import Mathlib

namespace radius_ratio_l467_46718

variable (VL VS rL rS : ℝ)
variable (hVL : VL = 432 * Real.pi)
variable (hVS : VS = 0.275 * VL)

theorem radius_ratio (h1 : (4 / 3) * Real.pi * rL^3 = VL)
                     (h2 : (4 / 3) * Real.pi * rS^3 = VS) :
  rS / rL = 2 / 3 := by
  sorry

end radius_ratio_l467_46718


namespace test_takers_percent_correct_l467_46712

theorem test_takers_percent_correct 
  (n : Set ℕ → ℝ) 
  (A B : Set ℕ) 
  (hB : n B = 0.75) 
  (hAB : n (A ∩ B) = 0.60) 
  (hneither : n (Set.univ \ (A ∪ B)) = 0.05) 
  : n A = 0.80 := by
  sorry

end test_takers_percent_correct_l467_46712


namespace perfect_square_expression_l467_46754
open Real

theorem perfect_square_expression (x : ℝ) :
  (12.86 * 12.86 + 12.86 * x + 0.14 * 0.14 = (12.86 + 0.14)^2) → x = 0.28 :=
by
  sorry

end perfect_square_expression_l467_46754


namespace existence_of_nonnegative_value_l467_46711

theorem existence_of_nonnegative_value :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x + 12 ≥ 0 := 
by
  sorry

end existence_of_nonnegative_value_l467_46711


namespace max_value_x2_y3_z_l467_46723

noncomputable def maximum_value (x y z : ℝ) : ℝ :=
  if x + y + z = 3 then x^2 * y^3 * z else 0

theorem max_value_x2_y3_z
  (x y z : ℝ)
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxyz : x + y + z = 3) :
  maximum_value x y z ≤ 9 / 16 := sorry

end max_value_x2_y3_z_l467_46723


namespace cube_root_of_neg_27_l467_46736

theorem cube_root_of_neg_27 : ∃ y : ℝ, y^3 = -27 ∧ y = -3 := by
  sorry

end cube_root_of_neg_27_l467_46736


namespace Geraldine_more_than_Jazmin_l467_46797

-- Define the number of dolls Geraldine and Jazmin have
def Geraldine_dolls : ℝ := 2186.0
def Jazmin_dolls : ℝ := 1209.0

-- State the theorem we need to prove
theorem Geraldine_more_than_Jazmin :
  Geraldine_dolls - Jazmin_dolls = 977.0 := 
by
  sorry

end Geraldine_more_than_Jazmin_l467_46797


namespace exists_unique_i_l467_46761

-- Let p be an odd prime number.
variable {p : ℕ} [Fact (Nat.Prime p)] (odd_prime : p % 2 = 1)

-- Let a be an integer in the sequence {2, 3, 4, ..., p-3, p-2}
variable (a : ℕ) (a_range : 2 ≤ a ∧ a ≤ p - 2)

-- Prove that there exists a unique i such that i * a ≡ 1 (mod p) and i ≠ a
theorem exists_unique_i (h1 : ∀ k, 1 ≤ k ∧ k ≤ p - 1 → Nat.gcd k p = 1) :
  ∃! (i : ℕ), 1 ≤ i ∧ i ≤ p - 1 ∧ i * a % p = 1 ∧ i ≠ a :=
by 
  sorry

end exists_unique_i_l467_46761


namespace evaluate_expression_at_two_l467_46789

theorem evaluate_expression_at_two: 
  (3 * 2^2 - 4 * 2 + 2) = 6 := 
by 
  sorry

end evaluate_expression_at_two_l467_46789


namespace unique_sum_of_cubes_lt_1000_l467_46767

theorem unique_sum_of_cubes_lt_1000 : 
  let max_cube := 11 
  let max_val := 1000 
  ∃ n : ℕ, n = 35 ∧ ∀ (a b : ℕ), 1 ≤ a ∧ a ≤ max_cube → 1 ≤ b ∧ b ≤ max_cube → a^3 + b^3 < max_val :=
sorry

end unique_sum_of_cubes_lt_1000_l467_46767


namespace determine_top_5_median_required_l467_46788

theorem determine_top_5_median_required (scores : Fin 9 → ℝ) (unique_scores : ∀ (i j : Fin 9), i ≠ j → scores i ≠ scores j) :
  ∃ median,
  (∀ (student_score : ℝ), 
    (student_score > median ↔ ∃ (idx_top : Fin 5), student_score = scores ⟨idx_top.1, sorry⟩)) :=
sorry

end determine_top_5_median_required_l467_46788


namespace carmen_burning_candles_l467_46724

theorem carmen_burning_candles (candle_hours_per_night: ℕ) (nights_per_candle: ℕ) (candles_used: ℕ) (total_nights: ℕ) : 
  candle_hours_per_night = 2 →
  nights_per_candle = 8 / candle_hours_per_night →
  candles_used = 6 →
  total_nights = candles_used * (nights_per_candle / candle_hours_per_night) →
  total_nights = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end carmen_burning_candles_l467_46724


namespace boat_license_combinations_l467_46726

theorem boat_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let digit_positions := 5
  (letter_choices * (digit_choices ^ digit_positions)) = 300000 :=
  sorry

end boat_license_combinations_l467_46726


namespace solve_for_nonzero_x_l467_46733

open Real

theorem solve_for_nonzero_x (x : ℝ) (hx : x ≠ 0) : (9 * x) ^ 18 = (18 * x) ^ 9 → x = 2 / 9 :=
by
  sorry

end solve_for_nonzero_x_l467_46733


namespace sufficient_but_not_necessary_l467_46768

-- Let's define the conditions and the theorem to be proved in Lean 4
theorem sufficient_but_not_necessary : ∀ x : ℝ, (x > 1 → x > 0) ∧ ¬(∀ x : ℝ, x > 0 → x > 1) := by
  sorry

end sufficient_but_not_necessary_l467_46768


namespace average_interest_rate_correct_l467_46763

-- Constants representing the conditions
def totalInvestment : ℝ := 5000
def rateA : ℝ := 0.035
def rateB : ℝ := 0.07

-- The condition that return from investment at 7% is twice that at 3.5%
def return_condition (x : ℝ) : Prop := 0.07 * x = 2 * 0.035 * (5000 - x)

-- The average rate of interest formula
noncomputable def average_rate_of_interest (x : ℝ) : ℝ := 
  (0.07 * x + 0.035 * (5000 - x)) / 5000

-- The theorem to prove the average rate is 5.25%
theorem average_interest_rate_correct : ∃ (x : ℝ), return_condition x ∧ average_rate_of_interest x = 0.0525 := 
by
  sorry

end average_interest_rate_correct_l467_46763


namespace find_value_of_f3_l467_46703

variable {R : Type} [LinearOrderedField R]

/-- f is an odd function -/
def is_odd_function (f : R → R) : Prop := ∀ x : R, f (-x) = -f x

/-- f is symmetric about the line x = 1 -/
def is_symmetric_about (f : R → R) (a : R) : Prop := ∀ x : R, f (a + x) = f (a - x)

variable (f : R → R)
variable (Hodd : is_odd_function f)
variable (Hsymmetric : is_symmetric_about f 1)
variable (Hf1 : f 1 = 2)

theorem find_value_of_f3 : f 3 = -2 :=
by
  sorry

end find_value_of_f3_l467_46703


namespace num_men_in_boat_l467_46727

theorem num_men_in_boat 
  (n : ℕ) (W : ℝ)
  (h1 : (W / n : ℝ) = W / n)
  (h2 : (W + 8) / n = W / n + 1)
  : n = 8 := 
sorry

end num_men_in_boat_l467_46727


namespace sqrt_real_domain_l467_46796

theorem sqrt_real_domain (x : ℝ) (h : 2 - x ≥ 0) : x ≤ 2 := 
sorry

end sqrt_real_domain_l467_46796


namespace probability_A_not_winning_l467_46775

theorem probability_A_not_winning 
  (prob_draw : ℚ := 1/2)
  (prob_B_wins : ℚ := 1/3) : 
  (prob_draw + prob_B_wins) = 5 / 6 := 
by
  sorry

end probability_A_not_winning_l467_46775


namespace phase_shift_equivalence_l467_46772

noncomputable def y_original (x : ℝ) : ℝ := 2 * Real.cos x ^ 2 - Real.sqrt 3 * Real.sin (2 * x)
noncomputable def y_target (x : ℝ) : ℝ := 2 * Real.sin (2 * x) + 1
noncomputable def phase_shift : ℝ := 5 * Real.pi / 12

theorem phase_shift_equivalence : 
  ∀ x : ℝ, y_original x = y_target (x - phase_shift) :=
sorry

end phase_shift_equivalence_l467_46772


namespace find_x8_l467_46721

theorem find_x8 (x : ℝ) (h : x + 1/x = Real.sqrt 2) : x^8 = 1 :=
by sorry

end find_x8_l467_46721


namespace area_square_hypotenuse_l467_46745

theorem area_square_hypotenuse 
(a : ℝ) 
(h1 : ∀ a: ℝ,  ∃ YZ: ℝ, YZ = a + 3) 
(h2: ∀ XY: ℝ, ∃ total_area: ℝ, XY^2 + XY * (XY + 3) + (2 * XY^2 + 6 * XY + 9) = 450) :
  ∃ XZ: ℝ, (2 * a^2 + 6 * a + 9 = XZ) → XZ = 201 := by
  sorry

end area_square_hypotenuse_l467_46745


namespace john_learns_vowels_in_fifteen_days_l467_46710

def days_to_learn_vowels (days_per_vowel : ℕ) (num_vowels : ℕ) : ℕ :=
  days_per_vowel * num_vowels

theorem john_learns_vowels_in_fifteen_days :
  days_to_learn_vowels 3 5 = 15 :=
by
  -- Proof goes here
  sorry

end john_learns_vowels_in_fifteen_days_l467_46710


namespace root_inverse_cubes_l467_46756

theorem root_inverse_cubes (a b c r s : ℝ) (h1 : a ≠ 0)
  (h2 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = r ∨ x = s) :
  (1 / r^3) + (1 / s^3) = (-b^3 + 3 * a * b * c) / c^3 :=
by
  sorry

end root_inverse_cubes_l467_46756


namespace alex_singles_percentage_l467_46749

theorem alex_singles_percentage (total_hits home_runs triples doubles: ℕ) 
  (h1 : total_hits = 50) 
  (h2 : home_runs = 2) 
  (h3 : triples = 3) 
  (h4 : doubles = 10) :
  ((total_hits - (home_runs + triples + doubles)) / total_hits : ℚ) * 100 = 70 := 
by
  sorry

end alex_singles_percentage_l467_46749


namespace divisible_by_11_of_sum_divisible_l467_46751

open Int

theorem divisible_by_11_of_sum_divisible (a b : ℤ) (h : 11 ∣ (a^2 + b^2)) : 11 ∣ a ∧ 11 ∣ b :=
sorry

end divisible_by_11_of_sum_divisible_l467_46751


namespace math_problem_l467_46770

theorem math_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 * x + y - x * y = 0) : 
  ((9 * x + y) * (9 / y + 1 / x) = x * y) ∧ ¬ ((x / 9) + y = 10) ∧ 
  ((x + y = 16) ↔ (x = 4 ∧ y = 12)) ∧ 
  ((x * y = 36) ↔ (x = 2 ∧ y = 18)) :=
by {
  sorry
}

end math_problem_l467_46770


namespace retailer_received_extra_boxes_l467_46746
-- Necessary import for mathematical proofs

-- Define the conditions
def dozen_boxes := 12
def dozens_ordered := 3
def discount_percent := 25

-- Calculate the total boxes ordered and the discount factor
def total_boxes := dozen_boxes * dozens_ordered
def discount_factor := (100 - discount_percent) / 100

-- Define the number of boxes paid for and the extra boxes received
def paid_boxes := total_boxes * discount_factor
def extra_boxes := total_boxes - paid_boxes

-- Statement of the proof problem
theorem retailer_received_extra_boxes : extra_boxes = 9 :=
by
    -- This is the place where the proof would be written
    sorry

end retailer_received_extra_boxes_l467_46746


namespace div_by_7_or_11_l467_46762

theorem div_by_7_or_11 (z x y : ℕ) (hx : x < 1000) (hz : z = 1000 * y + x) (hdiv7 : (x - y) % 7 = 0 ∨ (x - y) % 11 = 0) :
  z % 7 = 0 ∨ z % 11 = 0 :=
by
  sorry

end div_by_7_or_11_l467_46762


namespace golden_chest_diamonds_rubies_l467_46720

theorem golden_chest_diamonds_rubies :
  ∀ (diamonds rubies : ℕ), diamonds = 421 → rubies = 377 → diamonds - rubies = 44 :=
by
  intros diamonds rubies
  sorry

end golden_chest_diamonds_rubies_l467_46720


namespace ticTacToeWinningDiagonals_l467_46714

-- Define the tic-tac-toe board and the conditions
def ticTacToeBoard : Type := Fin 3 × Fin 3
inductive Player | X | O

def isWinningDiagonal (board : ticTacToeBoard → Option Player) : Prop :=
  (board (0, 0) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 2) = some Player.O) ∨
  (board (0, 2) = some Player.O ∧ board (1, 1) = some Player.O ∧ board (2, 0) = some Player.O)

-- Define the main problem statement
theorem ticTacToeWinningDiagonals : ∃ (n : ℕ), n = 40 :=
  sorry

end ticTacToeWinningDiagonals_l467_46714


namespace aunt_gave_each_20_l467_46700

theorem aunt_gave_each_20
  (jade_initial : ℕ)
  (julia_initial : ℕ)
  (total_after_aunt : ℕ)
  (equal_amount_from_aunt : ℕ)
  (h1 : jade_initial = 38)
  (h2 : julia_initial = jade_initial / 2)
  (h3 : total_after_aunt = 97)
  (h4 : jade_initial + julia_initial + 2 * equal_amount_from_aunt = total_after_aunt) :
  equal_amount_from_aunt = 20 := 
sorry

end aunt_gave_each_20_l467_46700


namespace merchant_loss_is_15_yuan_l467_46766

noncomputable def profit_cost_price : ℝ := (180 : ℝ) / 1.2
noncomputable def loss_cost_price : ℝ := (180 : ℝ) / 0.8

theorem merchant_loss_is_15_yuan :
  (180 + 180) - (profit_cost_price + loss_cost_price) = -15 := by
  sorry

end merchant_loss_is_15_yuan_l467_46766


namespace letters_received_per_day_l467_46750

-- Define the conditions
def packages_per_day := 20
def total_pieces_in_six_months := 14400
def days_in_month := 30
def months := 6

-- Calculate total days in six months
def total_days := months * days_in_month

-- Calculate pieces of mail per day
def pieces_per_day := total_pieces_in_six_months / total_days

-- Define the number of letters per day
def letters_per_day := pieces_per_day - packages_per_day

-- Prove that the number of letters per day is 60
theorem letters_received_per_day : letters_per_day = 60 := sorry

end letters_received_per_day_l467_46750


namespace magic_square_y_l467_46790

theorem magic_square_y (a b c d e y : ℚ) (h1 : y - 61 = a) (h2 : 2 * y - 125 = b) 
    (h3 : y + 25 + 64 = 3 + (y - 61) + (2 * y - 125)) : y = 272 / 3 :=
by
  sorry

end magic_square_y_l467_46790


namespace largest_whole_number_lt_div_l467_46705

theorem largest_whole_number_lt_div {x : ℕ} (hx : 8 * x < 80) : x ≤ 9 :=
by
  sorry

end largest_whole_number_lt_div_l467_46705


namespace ellipse_foci_on_y_axis_l467_46753

theorem ellipse_foci_on_y_axis (k : ℝ) (h1 : 5 + k > 3 - k) (h2 : 3 - k > 0) (h3 : 5 + k > 0) : -1 < k ∧ k < 3 :=
by 
  sorry

end ellipse_foci_on_y_axis_l467_46753


namespace incorrect_operation_l467_46715

theorem incorrect_operation (a : ℝ) : ¬ (a^3 + a^3 = 2 * a^6) :=
by
  sorry

end incorrect_operation_l467_46715


namespace stock_worth_l467_46773

theorem stock_worth (W : Real) 
  (profit_part : Real := 0.25 * W * 0.20)
  (loss_part1 : Real := 0.35 * W * 0.10)
  (loss_part2 : Real := 0.40 * W * 0.15)
  (overall_loss_eq : loss_part1 + loss_part2 - profit_part = 1200) : 
  W = 26666.67 :=
by
  sorry

end stock_worth_l467_46773


namespace total_robodinos_in_shipment_l467_46709

-- Definitions based on the conditions:
def percentage_on_display : ℝ := 0.30
def percentage_in_storage : ℝ := 0.70
def stored_robodinos : ℕ := 168

-- The main statement to prove:
theorem total_robodinos_in_shipment (T : ℝ) : (percentage_in_storage * T = stored_robodinos) → T = 240 := by
  sorry

end total_robodinos_in_shipment_l467_46709


namespace B_days_to_complete_work_l467_46785

theorem B_days_to_complete_work (B : ℕ) (hB : B ≠ 0)
  (A_work_days : ℕ := 9) (combined_days : ℕ := 6)
  (work_rate_A : ℚ := 1 / A_work_days) (work_rate_combined : ℚ := 1 / combined_days):
  (1 / B : ℚ) = work_rate_combined - work_rate_A → B = 18 :=
by
  intro h
  sorry

end B_days_to_complete_work_l467_46785


namespace equal_area_division_l467_46702

theorem equal_area_division (d : ℝ) : 
  (∃ x y, 0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 4 ∧ 
   (x = d ∨ x = 4) ∧ (y = 4 ∨ y = 0) ∧ 
   (2 : ℝ) * (4 - d) = 4) ↔ d = 2 :=
by
  sorry

end equal_area_division_l467_46702


namespace unique_triple_primes_l467_46758

theorem unique_triple_primes (p q r : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) 
  (h1 : p < q) (h2 : q < r) (h3 : (p^3 + q^3 + r^3) / (p + q + r) = 249) : r = 19 :=
sorry

end unique_triple_primes_l467_46758


namespace find_c_l467_46719

noncomputable def g (x c : ℝ) : ℝ := 1 / (3 * x + c)
noncomputable def g_inv (x : ℝ) : ℝ := (2 - 3 * x) / (3 * x)

theorem find_c (c : ℝ) : (∀ x, g_inv (g x c) = x) ↔ c = 3 / 2 := by
  sorry

end find_c_l467_46719


namespace garden_dimensions_l467_46759

variable {w l x : ℝ}

-- Definition of the problem conditions
def garden_length_eq_three_times_width (w l : ℝ) : Prop := l = 3 * w
def combined_area_eq (w x : ℝ) : Prop := (w + 2 * x) * (3 * w + 2 * x) = 432
def walkway_area_eq (w x : ℝ) : Prop := 8 * w * x + 4 * x^2 = 108

-- The main theorem statement
theorem garden_dimensions (w l x : ℝ)
  (h1 : garden_length_eq_three_times_width w l)
  (h2 : combined_area_eq w x)
  (h3 : walkway_area_eq w x) :
  w = 6 * Real.sqrt 3 ∧ l = 18 * Real.sqrt 3 :=
sorry

end garden_dimensions_l467_46759


namespace sum_of_1_to_17_is_odd_l467_46792

-- Define the set of natural numbers from 1 to 17
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

-- Proof that the sum of these numbers is odd
theorem sum_of_1_to_17_is_odd : (List.sum nums) % 2 = 1 := 
by
  sorry  -- Proof goes here

end sum_of_1_to_17_is_odd_l467_46792


namespace number_of_sides_on_die_l467_46755

theorem number_of_sides_on_die (n : ℕ) 
  (h1 : n ≥ 6) 
  (h2 : (∃ k : ℕ, k = 5) → (5 : ℚ) / (n ^ 2 : ℚ) = (5 : ℚ) / (36 : ℚ)) 
  : n = 6 :=
sorry

end number_of_sides_on_die_l467_46755


namespace train_speed_including_stoppages_l467_46743

theorem train_speed_including_stoppages (s : ℝ) (t : ℝ) (running_time_fraction : ℝ) :
  s = 48 ∧ t = 1/4 ∧ running_time_fraction = (1 - t) → (s * running_time_fraction = 36) :=
by
  sorry

end train_speed_including_stoppages_l467_46743


namespace bottle_caps_cost_l467_46784

-- Conditions
def cost_per_bottle_cap : ℕ := 2
def number_of_bottle_caps : ℕ := 6

-- Statement of the problem
theorem bottle_caps_cost : (cost_per_bottle_cap * number_of_bottle_caps) = 12 :=
by
  sorry

end bottle_caps_cost_l467_46784


namespace second_group_students_l467_46738

-- Define the number of groups and their respective sizes
def num_groups : ℕ := 4
def first_group_students : ℕ := 5
def third_group_students : ℕ := 7
def fourth_group_students : ℕ := 4
def total_students : ℕ := 24

-- Define the main theorem to prove
theorem second_group_students :
  (∃ second_group_students : ℕ,
    total_students = first_group_students + second_group_students + third_group_students + fourth_group_students ∧
    second_group_students = 8) :=
sorry

end second_group_students_l467_46738


namespace difference_rabbits_antelopes_l467_46777

variable (A R H W L : ℕ)
variable (x : ℕ)

def antelopes := 80
def rabbits := antelopes + x
def hyenas := (antelopes + rabbits) - 42
def wild_dogs := hyenas + 50
def leopards := rabbits / 2
def total_animals := 605

theorem difference_rabbits_antelopes
  (h1 : antelopes = 80)
  (h2 : rabbits = antelopes + x)
  (h3 : hyenas = (antelopes + rabbits) - 42)
  (h4 : wild_dogs = hyenas + 50)
  (h5 : leopards = rabbits / 2)
  (h6 : antelopes + rabbits + hyenas + wild_dogs + leopards = total_animals) : rabbits - antelopes = 70 := 
by
  -- Proof goes here
  sorry

end difference_rabbits_antelopes_l467_46777


namespace time_addition_and_sum_l467_46782

noncomputable def time_after_addition (hours_1 minutes_1 seconds_1 hours_2 minutes_2 seconds_2 : ℕ) : (ℕ × ℕ × ℕ) :=
  let total_seconds := seconds_1 + seconds_2
  let extra_minutes := total_seconds / 60
  let result_seconds := total_seconds % 60
  let total_minutes := minutes_1 + minutes_2 + extra_minutes
  let extra_hours := total_minutes / 60
  let result_minutes := total_minutes % 60
  let total_hours := hours_1 + hours_2 + extra_hours
  let result_hours := total_hours % 12
  (result_hours, result_minutes, result_seconds)

theorem time_addition_and_sum :
  let current_hours := 3
  let current_minutes := 0
  let current_seconds := 0
  let add_hours := 300
  let add_minutes := 55
  let add_seconds := 30
  let (final_hours, final_minutes, final_seconds) := time_after_addition current_hours current_minutes current_seconds add_hours add_minutes add_seconds
  final_hours + final_minutes + final_seconds = 88 :=
by
  sorry

end time_addition_and_sum_l467_46782


namespace tan_half_angles_l467_46744

theorem tan_half_angles (a b : ℝ) (ha : 3 * (Real.cos a + Real.cos b) + 5 * (Real.cos a * Real.cos b - 1) = 0) :
  ∃ z : ℝ, z = Real.tan (a / 2) * Real.tan (b / 2) ∧ (z = Real.sqrt (6 / 13) ∨ z = -Real.sqrt (6 / 13)) :=
by
  sorry

end tan_half_angles_l467_46744


namespace original_length_of_tape_l467_46794

-- Given conditions
variables (L : Real) (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
          (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4)

-- The theorem to prove
theorem original_length_of_tape (L : Real) 
  (used_by_Remaining_yesterday : L * (1 - 1 / 5) = 4 / 5 * L)
  (remaining_after_today : 1.5 = 4 / 5 * L * 1 / 4) :
  L = 7.5 :=
by
  sorry

end original_length_of_tape_l467_46794


namespace reflection_identity_l467_46748

-- Define the reflection function
def reflect (O P : ℝ × ℝ) : ℝ × ℝ := (2 * O.1 - P.1, 2 * O.2 - P.2)

-- Given three points and a point P
variables (O1 O2 O3 P : ℝ × ℝ)

-- Define the sequence of reflections
def sequence_reflection (P : ℝ × ℝ) : ℝ × ℝ :=
  reflect O3 (reflect O2 (reflect O1 P))

-- Lean 4 statement to prove the mathematical theorem
theorem reflection_identity :
  sequence_reflection O1 O2 O3 (sequence_reflection O1 O2 O3 P) = P :=
by sorry

end reflection_identity_l467_46748


namespace ravi_refrigerator_purchase_price_l467_46704

theorem ravi_refrigerator_purchase_price (purchase_price_mobile : ℝ) (sold_mobile : ℝ)
  (profit : ℝ) (loss : ℝ) (overall_profit : ℝ)
  (H1 : purchase_price_mobile = 8000)
  (H2 : loss = 0.04)
  (H3 : profit = 0.10)
  (H4 : overall_profit = 200) :
  ∃ R : ℝ, 0.96 * R + sold_mobile = R + purchase_price_mobile + overall_profit ∧ R = 15000 :=
by
  use 15000
  sorry

end ravi_refrigerator_purchase_price_l467_46704


namespace markov_coprime_squares_l467_46752

def is_coprime (x y : ℕ) : Prop :=
Nat.gcd x y = 1

theorem markov_coprime_squares (x y z : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) :
  x^2 + y^2 + z^2 = 3 * x * y * z →
  ∃ a b c: ℕ, (a, b, c) = (2, 1, 1) ∨ (a, b, c) = (1, 2, 1) ∨ (a, b, c) = (1, 1, 2) ∧ 
  (a ≠ 1 → ∃ p q : ℕ, is_coprime p q ∧ a = p^2 + q^2) :=
sorry

end markov_coprime_squares_l467_46752


namespace fraction_division_l467_46747

theorem fraction_division : (3 / 4) / (2 / 5) = 15 / 8 := 
by
  -- We need to convert this division into multiplication by the reciprocal
  -- (3 / 4) / (2 / 5) = (3 / 4) * (5 / 2)
  -- Now perform the multiplication of the numerators and denominators
  -- (3 * 5) / (4 * 2) = 15 / 8
  sorry

end fraction_division_l467_46747


namespace tom_watching_days_l467_46769

def show_a_season_1_time : Nat := 20 * 22
def show_a_season_2_time : Nat := 18 * 24
def show_a_season_3_time : Nat := 22 * 26
def show_a_season_4_time : Nat := 15 * 30

def show_b_season_1_time : Nat := 24 * 42
def show_b_season_2_time : Nat := 16 * 48
def show_b_season_3_time : Nat := 12 * 55

def show_c_season_1_time : Nat := 10 * 60
def show_c_season_2_time : Nat := 13 * 58
def show_c_season_3_time : Nat := 15 * 50
def show_c_season_4_time : Nat := 11 * 52
def show_c_season_5_time : Nat := 9 * 65

def show_a_total_time : Nat :=
  show_a_season_1_time + show_a_season_2_time +
  show_a_season_3_time + show_a_season_4_time

def show_b_total_time : Nat :=
  show_b_season_1_time + show_b_season_2_time + show_b_season_3_time

def show_c_total_time : Nat :=
  show_c_season_1_time + show_c_season_2_time +
  show_c_season_3_time + show_c_season_4_time +
  show_c_season_5_time

def total_time : Nat := show_a_total_time + show_b_total_time + show_c_total_time

def daily_watch_time : Nat := 120

theorem tom_watching_days : (total_time + daily_watch_time - 1) / daily_watch_time = 64 := sorry

end tom_watching_days_l467_46769


namespace system_solution_l467_46713

theorem system_solution (x y : ℚ) (h1 : 2 * x - 3 * y = 1) (h2 : (y + 1) / 4 + 1 = (x + 2) / 3) : x = 3 ∧ y = 5 / 3 :=
by
  sorry

end system_solution_l467_46713


namespace divisibility_expression_l467_46779

variable {R : Type*} [CommRing R] (x a b : R)

theorem divisibility_expression :
  ∃ k : R, (x + a + b) ^ 3 - x ^ 3 - a ^ 3 - b ^ 3 = (x + a) * (x + b) * k :=
sorry

end divisibility_expression_l467_46779


namespace sum_even_integers_l467_46732

theorem sum_even_integers (sum_first_50_even : Nat) (sum_from_100_to_200 : Nat) : 
  sum_first_50_even = 2550 → sum_from_100_to_200 = 7550 :=
by
  sorry

end sum_even_integers_l467_46732


namespace street_trees_one_side_number_of_street_trees_l467_46722

-- Conditions
def road_length : ℕ := 2575
def interval : ℕ := 25
def trees_at_endpoints : ℕ := 2

-- Question: number of street trees on one side of the road
theorem street_trees_one_side (road_length interval : ℕ) (trees_at_endpoints : ℕ) : ℕ :=
  (road_length / interval) + 1

-- Proof of the provided problem
theorem number_of_street_trees : street_trees_one_side road_length interval trees_at_endpoints = 104 :=
by
  sorry

end street_trees_one_side_number_of_street_trees_l467_46722


namespace B_completion_time_l467_46765

-- Definitions based on the conditions
def A_work : ℚ := 1 / 24
def B_work : ℚ := 1 / 16
def C_work : ℚ := 1 / 32  -- Since C takes twice the time as B, C_work = B_work / 2

-- Combined work rates based on the conditions
def combined_ABC_work := A_work + B_work + C_work
def combined_AB_work := A_work + B_work

-- Question: How long does B take to complete the job alone?
-- Answer: 16 days

theorem B_completion_time : 
  (combined_ABC_work = 1 / 8) ∧ 
  (combined_AB_work = 1 / 12) ∧ 
  (A_work = 1 / 24) ∧ 
  (C_work = B_work / 2) → 
  (1 / B_work = 16) := 
by 
  sorry

end B_completion_time_l467_46765


namespace board_divisible_into_hexominos_l467_46791

theorem board_divisible_into_hexominos {m n : ℕ} (h_m_gt_5 : m > 5) (h_n_gt_5 : n > 5) 
  (h_m_div_by_3 : m % 3 = 0) (h_n_div_by_4 : n % 4 = 0) : 
  (m * n) % 6 = 0 :=
by
  sorry

end board_divisible_into_hexominos_l467_46791


namespace petya_series_sum_l467_46708

theorem petya_series_sum (n k : ℕ) (h1 : (n + k) * (k + 1) = 20 * (n + 2 * k)) 
                                      (h2 : (n + k) * (k + 1) = 60 * n) :
  n = 29 ∧ k = 29 :=
by
  sorry

end petya_series_sum_l467_46708


namespace younger_age_is_12_l467_46730

theorem younger_age_is_12 
  (y elder : ℕ)
  (h_diff : elder = y + 20)
  (h_past : elder - 7 = 5 * (y - 7)) :
  y = 12 :=
by
  sorry

end younger_age_is_12_l467_46730


namespace find_k_l467_46764

theorem find_k (x y z k : ℝ) 
  (h1 : 9 / (x + y) = k / (x + 2 * z)) 
  (h2 : 9 / (x + y) = 14 / (z - y)) 
  (h3 : y = 2 * x) 
  (h4 : x + z = 10) :
  k = 46 :=
by
  sorry

end find_k_l467_46764


namespace domain_of_sqrt_tan_x_minus_sqrt_3_l467_46742

noncomputable def domain_of_function : Set Real :=
  {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

theorem domain_of_sqrt_tan_x_minus_sqrt_3 :
  { x : Real | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2 } = domain_of_function :=
by
  sorry

end domain_of_sqrt_tan_x_minus_sqrt_3_l467_46742


namespace total_face_value_of_notes_l467_46737

theorem total_face_value_of_notes :
  let face_value := 5
  let number_of_notes := 440 * 10^6
  face_value * number_of_notes = 2200000000 := 
by
  sorry

end total_face_value_of_notes_l467_46737


namespace tanya_dan_error_l467_46701

theorem tanya_dan_error 
  (a b c d e f g : ℤ)
  (h₁ : a < b) (h₂ : b < c) (h₃ : c < d) (h₄ : d < e) (h₅ : e < f) (h₆ : f < g)
  (h₇ : a % 2 = 1) (h₈ : b % 2 = 1) (h₉ : c % 2 = 1) (h₁₀ : d % 2 = 1) 
  (h₁₁ : e % 2 = 1) (h₁₂ : f % 2 = 1) (h₁₃ : g % 2 = 1)
  (h₁₄ : (a + b + c + d + e + f + g) / 7 - d = 3 / 7) :
  false :=
by sorry

end tanya_dan_error_l467_46701


namespace simplified_expression_eq_l467_46760

noncomputable def simplify_expression : ℚ :=
  1 / (1 / (1 / 3)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)

-- We need to prove that the simplifed expression is equal to 1 / 39
theorem simplified_expression_eq : simplify_expression = 1 / 39 :=
by sorry

end simplified_expression_eq_l467_46760


namespace inlet_pipe_rate_l467_46786

theorem inlet_pipe_rate (capacity : ℕ) (t_empty : ℕ) (t_with_inlet : ℕ) (R_out : ℕ) :
  capacity = 6400 →
  t_empty = 10 →
  t_with_inlet = 16 →
  R_out = capacity / t_empty →
  (R_out - (capacity / t_with_inlet)) / 60 = 4 :=
by
  intros h1 h2 h3 h4 
  sorry

end inlet_pipe_rate_l467_46786


namespace johns_running_hours_l467_46780

-- Define the conditions
variable (x : ℕ) -- let x represent the number of hours at 8 mph and 6 mph
variable (total_hours : ℕ) (total_distance : ℕ)
variable (speed_8 : ℕ) (speed_6 : ℕ) (speed_5 : ℕ)
variable (distance_8 : ℕ := speed_8 * x)
variable (distance_6 : ℕ := speed_6 * x)
variable (distance_5 : ℕ := speed_5 * (total_hours - 2 * x))

-- Total hours John completes the marathon
axiom h1: total_hours = 15

-- Total distance John completes in miles
axiom h2: total_distance = 95

-- Speed factors
axiom h3: speed_8 = 8
axiom h4: speed_6 = 6
axiom h5: speed_5 = 5

-- Distance equation
axiom h6: distance_8 + distance_6 + distance_5 = total_distance

-- Prove the number of hours John ran at each speed
theorem johns_running_hours : x = 5 :=
by
  sorry

end johns_running_hours_l467_46780


namespace amount_distributed_l467_46725

theorem amount_distributed (A : ℕ) (h : A / 14 = A / 18 + 80) : A = 5040 :=
sorry

end amount_distributed_l467_46725


namespace simplify_to_linear_binomial_l467_46771

theorem simplify_to_linear_binomial (k : ℝ) (x : ℝ) : 
  (-3 * k * x^2 + x - 1) + (9 * x^2 - 4 * k * x + 3 * k) = 
  (1 - 4 * k) * x + (3 * k - 1) → 
  k = 3 := by
  sorry

end simplify_to_linear_binomial_l467_46771


namespace gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l467_46729

theorem gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4 (k : Int) :
  Int.gcd ((360 * k)^2 + 6 * (360 * k) + 8) (360 * k + 4) = 4 := 
sorry

end gcd_a_squared_plus_6a_plus_8_and_a_plus_4_is_4_l467_46729


namespace hexagonal_prism_sum_maximum_l467_46757

noncomputable def hexagonal_prism_max_sum (h_u h_v h_w h_x h_y h_z : ℕ) (u v w x y z : ℝ) : ℝ :=
  u + v + w + x + y + z

def max_sum_possible (h_u h_v h_w h_x h_y h_z : ℕ) : ℝ :=
  if h_u = 4 ∧ h_v = 7 ∧ h_w = 10 ∨
     h_u = 4 ∧ h_x = 7 ∧ h_y = 10 ∨
     h_u = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_v = 4 ∧ h_x = 7 ∧ h_w = 10 ∨
     h_v = 4 ∧ h_y = 7 ∧ h_z = 10 ∨
     h_w = 4 ∧ h_x = 7 ∧ h_z = 10
  then 78
  else 0

theorem hexagonal_prism_sum_maximum (h_u h_v h_w h_x h_y h_z : ℕ) :
  max_sum_possible h_u h_v h_w h_x h_y h_z = 78 → ∃ (u v w x y z : ℝ), hexagonal_prism_max_sum h_u h_v h_w h_x h_y h_z u v w x y z = 78 := 
by 
  sorry

end hexagonal_prism_sum_maximum_l467_46757


namespace min_value_z_l467_46781

variable {x y : ℝ}

def constraint1 (x y : ℝ) : Prop := x + y ≤ 3
def constraint2 (x y : ℝ) : Prop := x - y ≥ -1
def constraint3 (y : ℝ) : Prop := y ≥ 1

theorem min_value_z (x y : ℝ) 
  (h1 : constraint1 x y) 
  (h2 : constraint2 x y) 
  (h3 : constraint3 y) 
  (hx_pos : x > 0) 
  (hy_pos : y > 0) : 
  ∃ x y, x > 0 ∧ y ≥ 1 ∧ x + y ≤ 3 ∧ x - y ≥ -1 ∧ (∀ x' y', x' > 0 ∧ y' ≥ 1 ∧ x' + y' ≤ 3 ∧ x' - y' ≥ -1 → (y' / x' ≥ y / x)) ∧ y / x = 1 / 2 := 
sorry

end min_value_z_l467_46781


namespace initial_garrison_men_l467_46799

theorem initial_garrison_men (M : ℕ) (H1 : ∃ provisions : ℕ, provisions = M * 60)
  (H2 : ∃ provisions_15 : ℕ, provisions_15 = M * 45)
  (H3 : ∀ provisions_15 (new_provisions: ℕ), (provisions_15 = M * 45 ∧ new_provisions = 20 * (M + 1250)) → provisions_15 = new_provisions) :
  M = 1000 :=
by
  sorry

end initial_garrison_men_l467_46799


namespace quadratic_equation_roots_l467_46739

theorem quadratic_equation_roots (a b k k1 k2 : ℚ)
  (h_roots : ∀ x : ℚ, k * (x^2 - x) + x + 2 = 0)
  (h_ab_condition : (a / b) + (b / a) = 3 / 7)
  (h_k_values : ∀ x : ℚ, 7 * x^2 - 20 * x - 21 = 0)
  (h_k1k2 : k1 + k2 = 20 / 7)
  (h_k1k2_prod : k1 * k2 = -21 / 7) :
  (k1 / k2) + (k2 / k1) = -104 / 21 :=
sorry

end quadratic_equation_roots_l467_46739


namespace sin_polar_circle_l467_46793

theorem sin_polar_circle (t : ℝ) (θ : ℝ) (r : ℝ) (h : ∀ θ, 0 ≤ θ ∧ θ ≤ t → r = Real.sin θ) :
  t = Real.pi := 
by
  sorry

end sin_polar_circle_l467_46793


namespace tetrahedron_sphere_surface_area_l467_46735

-- Define the conditions
variables (a : ℝ) (mid_AB_C : ℝ → Prop) (S : ℝ)
variables (h1 : a > 0)
variables (h2 : mid_AB_C a)
variables (h3 : S = 3 * Real.sqrt 2)

-- Theorem statement
theorem tetrahedron_sphere_surface_area (h1 : a = 2 * Real.sqrt 3) : 
  4 * Real.pi * ( (Real.sqrt 6 / 4) * a )^2 = 18 * Real.pi := by
  sorry

end tetrahedron_sphere_surface_area_l467_46735


namespace bob_initial_pennies_l467_46728

-- Definitions of conditions
variables (a b : ℕ)
def condition1 : Prop := b + 2 = 4 * (a - 2)
def condition2 : Prop := b - 2 = 3 * (a + 2)

-- Goal: Proving that b = 62
theorem bob_initial_pennies (h1 : condition1 a b) (h2 : condition2 a b) : b = 62 :=
by {
  sorry
}

end bob_initial_pennies_l467_46728


namespace complex_pure_imaginary_solution_l467_46798

theorem complex_pure_imaginary_solution (m : ℝ) 
  (h_real_part : m^2 + 2*m - 3 = 0) 
  (h_imaginary_part : m - 1 ≠ 0) : 
  m = -3 :=
sorry

end complex_pure_imaginary_solution_l467_46798


namespace travel_time_l467_46706

theorem travel_time (distance speed : ℝ) (h1 : distance = 300) (h2 : speed = 60) : 
  distance / speed = 5 := 
by
  sorry

end travel_time_l467_46706


namespace quadratic_equation_factored_form_correct_l467_46783

theorem quadratic_equation_factored_form_correct :
  ∀ x : ℝ, (x^2 - 4 * x - 1 = 0) → (x - 2)^2 = 5 :=
by
  intros x h
  sorry

end quadratic_equation_factored_form_correct_l467_46783


namespace unit_prices_max_toys_l467_46707

-- For question 1
theorem unit_prices (x y : ℕ)
  (h₁ : y = x + 25)
  (h₂ : 2*y + x = 200) : x = 50 ∧ y = 75 :=
by {
  sorry
}

-- For question 2
theorem max_toys (cost_a cost_b q_a q_b : ℕ)
  (h₁ : cost_a = 50)
  (h₂ : cost_b = 75)
  (h₃ : q_b = 2 * q_a)
  (h₄ : 50 * q_a + 75 * q_b ≤ 20000) : q_a ≤ 100 :=
by {
  sorry
}

end unit_prices_max_toys_l467_46707


namespace probability_N_taller_than_L_l467_46717

variable (M N L O : ℕ)

theorem probability_N_taller_than_L 
  (h1 : N < M) 
  (h2 : L > O) : 
  0 = 0 := 
sorry

end probability_N_taller_than_L_l467_46717


namespace average_weight_increase_l467_46731

theorem average_weight_increase (W_new : ℝ) (W_old : ℝ) (num_persons : ℝ): 
  W_new = 94 ∧ W_old = 70 ∧ num_persons = 8 → 
  (W_new - W_old) / num_persons = 3 :=
by
  intros h
  rcases h with ⟨h1, h2, h3⟩
  sorry

end average_weight_increase_l467_46731


namespace melanie_cats_l467_46741

theorem melanie_cats (jacob_cats : ℕ) (annie_cats : ℕ) (melanie_cats : ℕ) 
  (h_jacob : jacob_cats = 90)
  (h_annie : annie_cats = jacob_cats / 3)
  (h_melanie : melanie_cats = annie_cats * 2) :
  melanie_cats = 60 := by
  sorry

end melanie_cats_l467_46741


namespace determine_k_value_l467_46795

theorem determine_k_value (x y z k : ℝ) 
  (h1 : 5 / (x + y) = k / (x - z))
  (h2 : k / (x - z) = 9 / (z + y)) :
  k = 14 :=
sorry

end determine_k_value_l467_46795


namespace increasing_interval_l467_46778

-- Given function definition
def quad_func (x : ℝ) : ℝ := -x^2 + 1

-- Property to be proven: The function is increasing on the interval (-∞, 0]
theorem increasing_interval : ∀ x y : ℝ, x ≤ 0 → y ≤ 0 → x < y → quad_func x < quad_func y := by
  sorry

end increasing_interval_l467_46778


namespace updated_mean_166_l467_46774

/-- The mean of 50 observations is 200. Later, it was found that there is a decrement of 34 
from each observation. Prove that the updated mean of the observations is 166. -/
theorem updated_mean_166
  (mean : ℝ) (n : ℕ) (decrement : ℝ) (updated_mean : ℝ)
  (h1 : mean = 200) (h2 : n = 50) (h3 : decrement = 34) (h4 : updated_mean = 166) :
  mean - (decrement * n) / n = updated_mean :=
by
  sorry

end updated_mean_166_l467_46774


namespace power_half_mod_prime_l467_46787

-- Definitions of odd prime and coprime condition
def is_odd_prime (p : ℕ) : Prop := Nat.Prime p ∧ p % 2 = 1
def coprime (a p : ℕ) : Prop := Nat.gcd a p = 1

-- Main statement
theorem power_half_mod_prime (p a : ℕ) (hp : is_odd_prime p) (ha : coprime a p) :
  a ^ ((p - 1) / 2) % p = 1 ∨ a ^ ((p - 1) / 2) % p = p - 1 := 
  sorry

end power_half_mod_prime_l467_46787


namespace sum_of_solutions_l467_46716

theorem sum_of_solutions (a b c : ℚ) (h : a ≠ 0) (eq : 2 * x^2 - 7 * x - 9 = 0) : 
  (-b / a) = (7 / 2) := 
sorry

end sum_of_solutions_l467_46716


namespace arthur_walks_total_distance_l467_46740

theorem arthur_walks_total_distance :
  let east_blocks := 8
  let north_blocks := 10
  let west_blocks := 3
  let block_distance := 1 / 3
  let total_blocks := east_blocks + north_blocks + west_blocks
  let total_miles := total_blocks * block_distance
  total_miles = 7 :=
by
  sorry

end arthur_walks_total_distance_l467_46740


namespace min_value_of_expression_l467_46776

theorem min_value_of_expression : ∀ x : ℝ, ∃ (M : ℝ), (∀ x, 16^x - 4^x - 4^(x+1) + 3 ≥ M) ∧ M = -4 :=
by
  sorry

end min_value_of_expression_l467_46776


namespace main_theorem_l467_46734

-- Define the distribution
def P0 : ℝ := 0.4
def P2 : ℝ := 0.4
def P1 (p : ℝ) : ℝ := p

-- Define a hypothesis that the sum of probabilities is 1
def prob_sum_eq_one (p : ℝ) : Prop := P0 + P1 p + P2 = 1

-- Define the expected value of X
def E_X (p : ℝ) : ℝ := 0 * P0 + 1 * P1 p + 2 * P2

-- Define variance computation
def variance (p : ℝ) : ℝ := P0 * (0 - E_X p) ^ 2 + P1 p * (1 - E_X p) ^ 2 + P2 * (2 - E_X p) ^ 2

-- State the main theorem
theorem main_theorem : (∃ p : ℝ, prob_sum_eq_one p) ∧ variance 0.2 = 0.8 :=
by
  sorry

end main_theorem_l467_46734
