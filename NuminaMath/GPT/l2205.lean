import Mathlib

namespace total_volume_of_drink_l2205_220554

theorem total_volume_of_drink :
  ∀ (total_ounces : ℝ),
    (∀ orange_juice watermelon_juice grape_juice : ℝ,
      orange_juice = 0.25 * total_ounces →
      watermelon_juice = 0.4 * total_ounces →
      grape_juice = 0.35 * total_ounces →
      grape_juice = 105 →
      total_ounces = 300) :=
by
  intros total_ounces orange_juice watermelon_juice grape_juice ho hw hg hg_eq
  sorry

end total_volume_of_drink_l2205_220554


namespace parabola_focus_l2205_220562

-- Define the equation of the parabola
def parabola_eq (x y : ℝ) : Prop := y^2 = -8 * x

-- Define the coordinates of the focus
def focus (x y : ℝ) : Prop := x = -2 ∧ y = 0

-- The Lean statement that needs to be proved
theorem parabola_focus : ∀ (x y : ℝ), parabola_eq x y → focus x y :=
by
  intros x y h
  sorry

end parabola_focus_l2205_220562


namespace negation_of_existence_l2205_220568

theorem negation_of_existence (h: ∃ x : ℝ, 0 < x ∧ (Real.log x + x - 1 ≤ 0)) :
  ¬ (∀ x : ℝ, 0 < x → ¬ (Real.log x + x - 1 ≤ 0)) :=
sorry

end negation_of_existence_l2205_220568


namespace jane_reads_pages_l2205_220544

theorem jane_reads_pages (P : ℕ) (h1 : 7 * (P + 10) = 105) : P = 5 := by
  sorry

end jane_reads_pages_l2205_220544


namespace pool_cannot_be_filled_l2205_220538

noncomputable def pool := 48000 -- Pool capacity in gallons
noncomputable def hose_rate := 3 -- Rate of each hose in gallons per minute
noncomputable def number_of_hoses := 6 -- Number of hoses
noncomputable def leakage_rate := 18 -- Leakage rate in gallons per minute

theorem pool_cannot_be_filled : 
  (number_of_hoses * hose_rate - leakage_rate <= 0) -> False :=
by
  -- Skipping the proof with 'sorry' as per instructions
  sorry

end pool_cannot_be_filled_l2205_220538


namespace benny_missed_games_l2205_220500

def total_games : ℕ := 39
def attended_games : ℕ := 14
def missed_games : ℕ := total_games - attended_games

theorem benny_missed_games : missed_games = 25 := by
  sorry

end benny_missed_games_l2205_220500


namespace length_of_plot_l2205_220517

theorem length_of_plot (total_poles : ℕ) (distance : ℕ) (one_side : ℕ) (other_side : ℕ) 
  (poles_distance_condition : total_poles = 28) 
  (fencing_condition : distance = 10) 
  (side_condition : one_side = 50) 
  (rectangular_condition : total_poles = (2 * (one_side / distance) + 2 * (other_side / distance))) :
  other_side = 120 :=
by
  sorry

end length_of_plot_l2205_220517


namespace calculate_expression_l2205_220525

theorem calculate_expression (y : ℝ) : (20 * y^3) * (7 * y^2) * (1 / (2 * y)^3) = 17.5 * y^2 :=
by
  sorry

end calculate_expression_l2205_220525


namespace cuboid_surface_area_increase_l2205_220521

variables (L W H : ℝ)
def SA_original (L W H : ℝ) : ℝ := 2 * (L * W + L * H + W * H)

def SA_new (L W H : ℝ) : ℝ := 2 * ((1.50 * L) * (1.70 * W) + (1.50 * L) * (1.80 * H) + (1.70 * W) * (1.80 * H))

theorem cuboid_surface_area_increase :
  (SA_new L W H - SA_original L W H) / SA_original L W H * 100 = 315.5 :=
by
  sorry

end cuboid_surface_area_increase_l2205_220521


namespace addition_results_in_perfect_square_l2205_220540

theorem addition_results_in_perfect_square : ∃ n: ℕ, n * n = 4440 + 49 :=
by
  sorry

end addition_results_in_perfect_square_l2205_220540


namespace range_of_a_l2205_220542

noncomputable def f (x : ℝ) : ℝ := x + 1 / Real.exp x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x > a * x) ↔ (1 - Real.exp 1 < a ∧ a ≤ 1) := by
  sorry

end range_of_a_l2205_220542


namespace problem1_problem2_problem3_l2205_220561

-- Problem 1
theorem problem1 (x y : ℝ) : 4 * x^2 - y^4 = (2 * x + y^2) * (2 * x - y^2) :=
by
  -- proof omitted
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : 8 * x^2 - 24 * x * y + 18 * y^2 = 2 * (2 * x - 3 * y)^2 :=
by
  -- proof omitted
  sorry

-- Problem 3
theorem problem3 (x y : ℝ) : (x - y) * (3 * x + 1) - 2 * (x^2 - y^2) - (y - x)^2 = (x - y) * (1 - y) :=
by
  -- proof omitted
  sorry

end problem1_problem2_problem3_l2205_220561


namespace find_ab_exponent_l2205_220592

theorem find_ab_exponent (a b : ℝ) 
  (h : |a - 2| + (b + 1 / 2)^2 = 0) : 
  a^2022 * b^2023 = -1 / 2 := 
sorry

end find_ab_exponent_l2205_220592


namespace proof_l_shaped_area_l2205_220572

-- Define the overall rectangle dimensions
def overall_length : ℕ := 10
def overall_width : ℕ := 7

-- Define the dimensions of the removed rectangle
def removed_length : ℕ := overall_length - 3
def removed_width : ℕ := overall_width - 2

-- Calculate the areas
def overall_area : ℕ := overall_length * overall_width
def removed_area : ℕ := removed_length * removed_width
def l_shaped_area : ℕ := overall_area - removed_area

-- The theorem to be proved
theorem proof_l_shaped_area : l_shaped_area = 35 := by
  sorry

end proof_l_shaped_area_l2205_220572


namespace avg_class_weight_l2205_220504

def num_students_A : ℕ := 24
def num_students_B : ℕ := 16
def avg_weight_A : ℕ := 40
def avg_weight_B : ℕ := 35

/-- Theorem: The average weight of the whole class is 38 kg --/
theorem avg_class_weight :
  (num_students_A * avg_weight_A + num_students_B * avg_weight_B) / (num_students_A + num_students_B) = 38 :=
by
  -- Proof goes here
  sorry

end avg_class_weight_l2205_220504


namespace articles_bought_l2205_220537

theorem articles_bought (C : ℝ) (N : ℝ) (h1 : (N * C) = (30 * ((5 / 3) * C))) : N = 50 :=
by
  sorry

end articles_bought_l2205_220537


namespace build_wall_30_persons_l2205_220511

-- Defining the conditions
def work_rate (persons : ℕ) (days : ℕ) : ℚ := 1 / (persons * days)

-- Total work required to build the wall by 8 persons in 42 days
def total_work : ℚ := work_rate 8 42 * 8 * 42

-- Work rate for 30 persons
def combined_work_rate (persons : ℕ) : ℚ := persons * work_rate 8 42

-- Days required for 30 persons to complete the same work
def days_required (persons : ℕ) (work : ℚ) : ℚ := work / combined_work_rate persons

-- Expected result is 11.2 days for 30 persons
theorem build_wall_30_persons : days_required 30 total_work = 11.2 := 
by
  sorry

end build_wall_30_persons_l2205_220511


namespace john_flights_of_stairs_l2205_220523

theorem john_flights_of_stairs (x : ℕ) : 
    let flight_height := 10
    let rope_height := flight_height / 2
    let ladder_height := rope_height + 10
    let total_height := 70
    10 * x + rope_height + ladder_height = total_height → x = 5 :=
by
    intro h
    sorry

end john_flights_of_stairs_l2205_220523


namespace unique_solution_of_abc_l2205_220506

theorem unique_solution_of_abc (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) 
  (h_lt_ab_c : a < b) (h_lt_b_c: b < c) (h_eq_abc : a * b + b * c + c * a = a * b * c) : 
  a = 2 ∧ b = 3 ∧ c = 6 :=
by {
  -- Proof skipped, only the statement is provided.
  sorry
}

end unique_solution_of_abc_l2205_220506


namespace length_of_first_train_l2205_220594

noncomputable def length_first_train
  (speed_train1_kmh : ℝ)
  (speed_train2_kmh : ℝ)
  (time_sec : ℝ)
  (length_train2_m : ℝ) : ℝ :=
  let relative_speed_mps := (speed_train1_kmh + speed_train2_kmh) * (1000 / 3600)
  let total_distance_m := relative_speed_mps * time_sec
  total_distance_m - length_train2_m

theorem length_of_first_train :
  length_first_train 80 65 7.82006405004841 165 = 150.106201 :=
  by
  -- Proof steps would go here.
  sorry

end length_of_first_train_l2205_220594


namespace exists_infinitely_many_primes_dividing_form_l2205_220535

theorem exists_infinitely_many_primes_dividing_form (a : ℕ) (ha : 0 < a) :
  ∃ᶠ p in Filter.atTop, ∃ n : ℕ, p ∣ 2^(2*n) + a := 
sorry

end exists_infinitely_many_primes_dividing_form_l2205_220535


namespace inheritance_amount_l2205_220598

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end inheritance_amount_l2205_220598


namespace chip_credit_card_balance_l2205_220583

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l2205_220583


namespace train_crossing_time_l2205_220549

def train_length : ℝ := 140
def bridge_length : ℝ := 235.03
def speed_kmh : ℝ := 45

noncomputable def speed_mps : ℝ := speed_kmh * (1000 / 3600)
noncomputable def total_distance : ℝ := train_length + bridge_length

theorem train_crossing_time :
  (total_distance / speed_mps) = 30.0024 :=
by
  sorry

end train_crossing_time_l2205_220549


namespace top_and_bottom_area_each_l2205_220503

def long_side_area : ℕ := 2 * 8 * 6
def short_side_area : ℕ := 2 * 5 * 6
def total_sides_area : ℕ := long_side_area + short_side_area
def total_needed_area : ℕ := 236
def top_and_bottom_area : ℕ := total_needed_area - total_sides_area

theorem top_and_bottom_area_each :
  top_and_bottom_area / 2 = 40 := by
  sorry

end top_and_bottom_area_each_l2205_220503


namespace laura_park_time_l2205_220518

theorem laura_park_time
  (T : ℝ) -- Time spent at the park each trip in hours
  (walk_time : ℝ := 0.5) -- Time spent walking to and from the park each trip in hours
  (trips : ℕ := 6) -- Total number of trips
  (park_time_percentage : ℝ := 0.80) -- Percentage of total time spent at the park
  (total_park_time_eq : trips * T = park_time_percentage * (trips * (T + walk_time))) :
  T = 2 :=
by
  sorry

end laura_park_time_l2205_220518


namespace z_is_46_percent_less_than_y_l2205_220570

variable (w e y z : ℝ)

-- Conditions
def w_is_60_percent_of_e := w = 0.60 * e
def e_is_60_percent_of_y := e = 0.60 * y
def z_is_150_percent_of_w := z = w * 1.5000000000000002

-- Proof Statement
theorem z_is_46_percent_less_than_y (h1 : w_is_60_percent_of_e w e)
                                    (h2 : e_is_60_percent_of_y e y)
                                    (h3 : z_is_150_percent_of_w z w) :
                                    100 - (z / y * 100) = 46 :=
by
  sorry

end z_is_46_percent_less_than_y_l2205_220570


namespace find_three_digit_numbers_l2205_220563

theorem find_three_digit_numbers :
  ∃ A, (100 ≤ A ∧ A ≤ 999) ∧ (A^2 % 1000 = A) ↔ (A = 376) ∨ (A = 625) :=
by
  sorry

end find_three_digit_numbers_l2205_220563


namespace hawkeye_charged_4_times_l2205_220505

variables (C B L S : ℝ) (N : ℕ)
def hawkeye_charging_problem : Prop :=
  C = 3.5 ∧ B = 20 ∧ L = 6 ∧ S = B - L ∧ N = (S / C) → N = 4 

theorem hawkeye_charged_4_times : hawkeye_charging_problem C B L S N :=
by {
  repeat { sorry }
}

end hawkeye_charged_4_times_l2205_220505


namespace find_marks_in_biology_l2205_220550

/-- 
David's marks in various subjects and his average marks are given.
This statement proves David's marks in Biology assuming the conditions provided.
--/
theorem find_marks_in_biology
  (english : ℕ) (math : ℕ) (physics : ℕ) (chemistry : ℕ) (avg_marks : ℕ)
  (total_subjects : ℕ)
  (h_english : english = 91)
  (h_math : math = 65)
  (h_physics : physics = 82)
  (h_chemistry : chemistry = 67)
  (h_avg_marks : avg_marks = 78)
  (h_total_subjects : total_subjects = 5)
  : ∃ (biology : ℕ), biology = 85 := 
by
  sorry

end find_marks_in_biology_l2205_220550


namespace ellipse_eccentricity_l2205_220575

theorem ellipse_eccentricity (a c : ℝ) (h : 2 * a = 2 * (2 * c)) : (c / a) = 1 / 2 :=
by
  sorry

end ellipse_eccentricity_l2205_220575


namespace candy_per_day_eq_eight_l2205_220595

def candy_received_from_neighbors : ℝ := 11.0
def candy_received_from_sister : ℝ := 5.0
def days_candy_lasted : ℝ := 2.0

theorem candy_per_day_eq_eight :
  (candy_received_from_neighbors + candy_received_from_sister) / days_candy_lasted = 8.0 :=
by
  sorry

end candy_per_day_eq_eight_l2205_220595


namespace find_mn_l2205_220567

theorem find_mn (sec_x_plus_tan_x : ℝ) (sec_tan_eq : sec_x_plus_tan_x = 24 / 7) :
  ∃ (m n : ℕ) (h : Int.gcd m n = 1), (∃ y, y = (m:ℝ) / (n:ℝ) ∧ (y^2)*527^2 - 2*y*527*336 + 336^2 = 1) ∧
  m + n = boxed_mn :=
by
  sorry

end find_mn_l2205_220567


namespace johnson_oldest_child_age_l2205_220591

/-- The average age of the three Johnson children is 10 years. 
    The two younger children are 6 years old and 8 years old. 
    Prove that the age of the oldest child is 16 years. -/
theorem johnson_oldest_child_age :
  ∃ x : ℕ, (6 + 8 + x) / 3 = 10 ∧ x = 16 :=
by
  sorry

end johnson_oldest_child_age_l2205_220591


namespace truck_license_combinations_l2205_220576

theorem truck_license_combinations :
  let letter_choices := 3
  let digit_choices := 10
  let number_of_digits := 6
  letter_choices * (digit_choices ^ number_of_digits) = 3000000 :=
by
  sorry

end truck_license_combinations_l2205_220576


namespace arithmetic_evaluation_l2205_220558

theorem arithmetic_evaluation :
  -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 :=
by
  sorry

end arithmetic_evaluation_l2205_220558


namespace evie_collected_shells_for_6_days_l2205_220509

theorem evie_collected_shells_for_6_days (d : ℕ) (h1 : 10 * d - 2 = 58) : d = 6 := by
  sorry

end evie_collected_shells_for_6_days_l2205_220509


namespace percentage_of_full_marks_D_l2205_220531

theorem percentage_of_full_marks_D (full_marks a b c d : ℝ)
  (h_full_marks : full_marks = 500)
  (h_a : a = 360)
  (h_a_b : a = b - 0.10 * b)
  (h_b_c : b = c + 0.25 * c)
  (h_c_d : c = d - 0.20 * d) :
  d / full_marks * 100 = 80 :=
by
  sorry

end percentage_of_full_marks_D_l2205_220531


namespace quadratic_no_real_roots_probability_l2205_220565

theorem quadratic_no_real_roots_probability :
  (1 : ℝ) - 1 / 4 - 0 = 3 / 4 :=
by
  sorry

end quadratic_no_real_roots_probability_l2205_220565


namespace pqr_value_l2205_220556

noncomputable def complex_numbers (p q r : ℂ) := p * q + 5 * q = -20 ∧ q * r + 5 * r = -20 ∧ r * p + 5 * p = -20

theorem pqr_value (p q r : ℂ) (h : complex_numbers p q r) : p * q * r = 80 := by
  sorry

end pqr_value_l2205_220556


namespace fraction_zero_solution_l2205_220569

theorem fraction_zero_solution (x : ℝ) (h₁ : x - 1 = 0) (h₂ : x + 3 ≠ 0) : x = 1 :=
by
  sorry

end fraction_zero_solution_l2205_220569


namespace solution_set_of_inequality_l2205_220528

theorem solution_set_of_inequality :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_of_inequality_l2205_220528


namespace solve_seating_problem_l2205_220513

-- Define the conditions of the problem
def valid_seating_arrangements (n : ℕ) : Prop :=
  (∃ (x y : ℕ), x < y ∧ x + 1 < y ∧ y < n ∧ 
    (n ≥ 5 ∧ y - x - 1 > 0)) ∧
  (∃! (x' y' : ℕ), x' < y' ∧ x' + 1 < y' ∧ y' < n ∧ 
    (n ≥ 5 ∧ y' - x' - 1 > 0))

-- State the theorem
theorem solve_seating_problem : ∃ n : ℕ, valid_seating_arrangements n ∧ n = 5 :=
by
  sorry

end solve_seating_problem_l2205_220513


namespace envelope_weight_l2205_220588

-- Define the conditions as constants
def total_weight_kg : ℝ := 7.48
def num_envelopes : ℕ := 880
def kg_to_g_conversion : ℝ := 1000

-- Calculate the total weight in grams
def total_weight_g : ℝ := total_weight_kg * kg_to_g_conversion

-- Define the expected weight of one envelope in grams
def expected_weight_one_envelope_g : ℝ := 8.5

-- The proof statement
theorem envelope_weight :
  total_weight_g / num_envelopes = expected_weight_one_envelope_g := by
  sorry

end envelope_weight_l2205_220588


namespace value_of_k_h_10_l2205_220533

def h (x : ℝ) : ℝ := 4 * x - 5
def k (x : ℝ) : ℝ := 2 * x + 6

theorem value_of_k_h_10 : k (h 10) = 76 := by
  -- We provide only the statement as required, skipping the proof
  sorry

end value_of_k_h_10_l2205_220533


namespace pieces_present_l2205_220599

def total_pieces : ℕ := 32
def missing_pieces : ℕ := 10

theorem pieces_present : total_pieces - missing_pieces = 22 :=
by {
  sorry
}

end pieces_present_l2205_220599


namespace unique_a_value_l2205_220547

theorem unique_a_value (a : ℝ) :
  let M := { x : ℝ | x^2 = 2 }
  let N := { x : ℝ | a * x = 1 }
  N ⊆ M ↔ (a = 0 ∨ a = -Real.sqrt 2 / 2 ∨ a = Real.sqrt 2 / 2) :=
by
  sorry

end unique_a_value_l2205_220547


namespace length_of_bridge_is_correct_l2205_220520

noncomputable def train_length : ℝ := 150
noncomputable def crossing_time : ℝ := 29.997600191984642
noncomputable def train_speed_kmph : ℝ := 36
noncomputable def kmph_to_mps (v : ℝ) : ℝ := (v * 1000) / 3600
noncomputable def train_speed_mps : ℝ := kmph_to_mps train_speed_kmph
noncomputable def total_distance : ℝ := train_speed_mps * crossing_time
noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge_is_correct :
  bridge_length = 149.97600191984642 := by
  sorry

end length_of_bridge_is_correct_l2205_220520


namespace fifth_root_of_unity_l2205_220573

noncomputable def expression (x : ℂ) := 
  2 * x + 1 / (1 + x) + x / (1 + x^2) + x^2 / (1 + x^3) + x^3 / (1 + x^4)

theorem fifth_root_of_unity (x : ℂ) (hx : x^5 = 1) : 
  (expression x = 4) ∨ (expression x = -1 + Real.sqrt 5) ∨ (expression x = -1 - Real.sqrt 5) :=
sorry

end fifth_root_of_unity_l2205_220573


namespace xia_sheets_left_l2205_220582

def stickers_left (initial : ℕ) (shared : ℕ) (per_sheet : ℕ) : ℕ :=
  (initial - shared) / per_sheet

theorem xia_sheets_left :
  stickers_left 150 100 10 = 5 :=
by
  sorry

end xia_sheets_left_l2205_220582


namespace train_length_l2205_220555

theorem train_length (L : ℝ) (V : ℝ)
  (h1 : V = L / 8)
  (h2 : V = (L + 273) / 20) :
  L = 182 :=
  by
  sorry

end train_length_l2205_220555


namespace parallel_lines_slope_condition_l2205_220574

theorem parallel_lines_slope_condition (b : ℝ) (x y : ℝ) :
    (∀ (x y : ℝ), 3 * y - 3 * b = 9 * x) →
    (∀ (x y : ℝ), y + 2 = (b + 9) * x) →
    b = -6 :=
by
    sorry

end parallel_lines_slope_condition_l2205_220574


namespace convert_base_10_to_base_7_l2205_220543

def base10_to_base7 (n : ℕ) : ℕ := 
  match n with
  | 5423 => 21545
  | _ => 0

theorem convert_base_10_to_base_7 : base10_to_base7 5423 = 21545 := by
  rfl

end convert_base_10_to_base_7_l2205_220543


namespace different_colors_probability_l2205_220585

noncomputable def differentColorProbability : ℚ :=
  let redChips := 7
  let greenChips := 5
  let totalChips := redChips + greenChips
  let probRedThenGreen := (redChips / totalChips) * (greenChips / totalChips)
  let probGreenThenRed := (greenChips / totalChips) * (redChips / totalChips)
  (probRedThenGreen + probGreenThenRed)

theorem different_colors_probability :
  differentColorProbability = 35 / 72 :=
by sorry

end different_colors_probability_l2205_220585


namespace convert_BFACE_to_decimal_l2205_220579

def hex_BFACE : ℕ := 11 * 16^4 + 15 * 16^3 + 10 * 16^2 + 12 * 16^1 + 14 * 16^0

theorem convert_BFACE_to_decimal : hex_BFACE = 785102 := by
  sorry

end convert_BFACE_to_decimal_l2205_220579


namespace extra_flour_l2205_220578

-- Define the conditions
def recipe_flour : ℝ := 7.0
def mary_flour : ℝ := 9.0

-- Prove the number of extra cups of flour Mary puts in
theorem extra_flour : mary_flour - recipe_flour = 2 :=
by
  sorry

end extra_flour_l2205_220578


namespace mod_pow_sub_eq_l2205_220536

theorem mod_pow_sub_eq : 
  (45^1537 - 25^1537) % 8 = 4 := 
by
  have h1 : 45 % 8 = 5 := by norm_num
  have h2 : 25 % 8 = 1 := by norm_num
  sorry

end mod_pow_sub_eq_l2205_220536


namespace towels_per_person_l2205_220577

-- Define the conditions
def num_rooms : ℕ := 10
def people_per_room : ℕ := 3
def total_towels : ℕ := 60

-- Define the total number of people
def total_people : ℕ := num_rooms * people_per_room

-- Define the proposition to prove
theorem towels_per_person : total_towels / total_people = 2 :=
by sorry

end towels_per_person_l2205_220577


namespace sector_area_l2205_220548

theorem sector_area (r l : ℝ) (h1 : l + 2 * r = 10) (h2 : l = 2 * r) : 
  (1 / 2) * l * r = 25 / 4 :=
by 
  sorry

end sector_area_l2205_220548


namespace box_filling_rate_l2205_220519

theorem box_filling_rate (l w h t : ℝ) (hl : l = 7) (hw : w = 6) (hh : h = 2) (ht : t = 21) : 
  (l * w * h) / t = 4 := by
  sorry

end box_filling_rate_l2205_220519


namespace sin_double_angle_tangent_identity_l2205_220514

theorem sin_double_angle_tangent_identity (x : ℝ) 
  (h : Real.tan (x + Real.pi / 4) = 2) : 
  Real.sin (2 * x) = 3 / 5 :=
by
  -- proof is omitted
  sorry

end sin_double_angle_tangent_identity_l2205_220514


namespace max_value_of_f_l2205_220597

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end max_value_of_f_l2205_220597


namespace perfect_number_mod_9_l2205_220587

theorem perfect_number_mod_9 (N : ℕ) (hN : ∃ p, N = 2^(p-1) * (2^p - 1) ∧ Nat.Prime (2^p - 1)) (hN_ne_6 : N ≠ 6) : ∃ n : ℕ, N = 9 * n + 1 :=
by
  sorry

end perfect_number_mod_9_l2205_220587


namespace log_inequality_l2205_220596

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end log_inequality_l2205_220596


namespace correct_product_l2205_220580

theorem correct_product : 0.125 * 5.12 = 0.64 := sorry

end correct_product_l2205_220580


namespace smallest_n_l2205_220566

theorem smallest_n (n : ℕ) (h : 5 * n ≡ 850 [MOD 26]) : n = 14 :=
by
  sorry

end smallest_n_l2205_220566


namespace choir_average_age_solution_l2205_220522

noncomputable def choir_average_age (avg_f avg_m avg_c : ℕ) (n_f n_m n_c : ℕ) : ℕ :=
  (n_f * avg_f + n_m * avg_m + n_c * avg_c) / (n_f + n_m + n_c)

def choir_average_age_problem : Prop :=
  let avg_f := 32
  let avg_m := 38
  let avg_c := 10
  let n_f := 12
  let n_m := 18
  let n_c := 5
  choir_average_age avg_f avg_m avg_c n_f n_m n_c = 32

theorem choir_average_age_solution : choir_average_age_problem := by
  sorry

end choir_average_age_solution_l2205_220522


namespace abs_inequality_solution_set_l2205_220546

theorem abs_inequality_solution_set :
  { x : ℝ | abs (2 - x) < 5 } = { x : ℝ | -3 < x ∧ x < 7 } :=
by
  sorry

end abs_inequality_solution_set_l2205_220546


namespace solve_for_x_l2205_220551

theorem solve_for_x (x : ℚ) : (3 * x / 7 - 2 = 12) → (x = 98 / 3) :=
by
  intro h
  sorry

end solve_for_x_l2205_220551


namespace average_mark_of_excluded_students_l2205_220590

theorem average_mark_of_excluded_students (N A A_remaining N_excluded N_remaining T T_remaining T_excluded A_excluded : ℝ)
  (hN : N = 33) 
  (hA : A = 90) 
  (hA_remaining : A_remaining = 95)
  (hN_excluded : N_excluded = 3) 
  (hN_remaining : N_remaining = N - N_excluded) 
  (hT : T = N * A) 
  (hT_remaining : T_remaining = N_remaining * A_remaining) 
  (hT_eq : T = T_excluded + T_remaining) : 
  A_excluded = T_excluded / N_excluded :=
by
  have hTN : N = 33 := hN
  have hTA : A = 90 := hA
  have hTAR : A_remaining = 95 := hA_remaining
  have hTN_excluded : N_excluded = 3 := hN_excluded
  have hNrem : N_remaining = N - N_excluded := hN_remaining
  have hT_sum : T = N * A := hT
  have hTRem : T_remaining = N_remaining * A_remaining := hT_remaining
  have h_sum_eq : T = T_excluded + T_remaining := hT_eq
  sorry -- proof yet to be constructed

end average_mark_of_excluded_students_l2205_220590


namespace range_f_compare_sizes_final_comparison_l2205_220530

noncomputable def f (x : ℝ) := |2 * x - 1| + |x + 1|

theorem range_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = {y : ℝ | y ∈ Set.Ici (3 / 2)} :=
sorry

theorem compare_sizes (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
sorry

theorem final_comparison (a : ℝ) (ha : a ≥ 3 / 2) :
  |a - 1| + |a + 1| > 3 / (2 * a) ∧ 3 / (2 * a) > 7 / 2 - 2 * a :=
by
  exact compare_sizes a ha

end range_f_compare_sizes_final_comparison_l2205_220530


namespace selling_price_conditions_met_l2205_220510

-- Definitions based on the problem conditions
def initial_selling_price : ℝ := 50
def purchase_price : ℝ := 40
def initial_volume : ℝ := 500
def decrease_rate : ℝ := 10
def desired_profit : ℝ := 8000
def max_total_cost : ℝ := 10000

-- Definition for the selling price
def selling_price : ℝ := 80

-- Condition: Cost is below $10000 for the valid selling price
def valid_item_count (x : ℝ) : ℝ := initial_volume - decrease_rate * (x - initial_selling_price)

-- Cost calculation function
def total_cost (x : ℝ) : ℝ := purchase_price * valid_item_count x

-- Profit calculation function 
def profit (x : ℝ) : ℝ := (x - purchase_price) * valid_item_count x

-- Main theorem statement
theorem selling_price_conditions_met : 
  profit selling_price = desired_profit ∧ total_cost selling_price < max_total_cost :=
by
  sorry

end selling_price_conditions_met_l2205_220510


namespace polynomial_self_composition_l2205_220524

theorem polynomial_self_composition {p : Polynomial ℝ} {n : ℕ} (hn : 0 < n) :
  (∀ x, p.eval (p.eval x) = (p.eval x) ^ n) ↔ p = Polynomial.X ^ n :=
by sorry

end polynomial_self_composition_l2205_220524


namespace desired_average_l2205_220586

variable (avg_4_tests : ℕ)
variable (score_5th_test : ℕ)

theorem desired_average (h1 : avg_4_tests = 78) (h2 : score_5th_test = 88) : (4 * avg_4_tests + score_5th_test) / 5 = 80 :=
by
  sorry

end desired_average_l2205_220586


namespace smallest_y_square_l2205_220559

theorem smallest_y_square (y n : ℕ) (h1 : y = 10) (h2 : ∀ m : ℕ, (∃ z : ℕ, m * y = z^2) ↔ (m = n)) : n = 10 :=
sorry

end smallest_y_square_l2205_220559


namespace ice_cream_cone_cost_is_5_l2205_220502

noncomputable def cost_of_ice_cream_cone (x : ℝ) : Prop := 
  let total_cost_of_cones := 15 * x
  let total_cost_of_puddings := 5 * 2
  let extra_spent_on_cones := total_cost_of_cones - total_cost_of_puddings
  extra_spent_on_cones = 65

theorem ice_cream_cone_cost_is_5 : ∃ x : ℝ, cost_of_ice_cream_cone x ∧ x = 5 :=
by 
  use 5
  unfold cost_of_ice_cream_cone
  simp
  sorry

end ice_cream_cone_cost_is_5_l2205_220502


namespace solve_equation_l2205_220552

theorem solve_equation (x : ℝ) (h : x ≠ 2 / 3) :
  (6 * x + 2) / (3 * x^2 + 6 * x - 4) = (3 * x) / (3 * x - 2) ↔ (x = (Real.sqrt 6) / 3 ∨ x = -(Real.sqrt 6) / 3) :=
by sorry

end solve_equation_l2205_220552


namespace julieta_total_cost_l2205_220532

variable (initial_backpack_price : ℕ)
variable (initial_binder_price : ℕ)
variable (backpack_price_increase : ℕ)
variable (binder_price_reduction : ℕ)
variable (discount_rate : ℕ)
variable (num_binders : ℕ)

def calculate_total_cost (initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders : ℕ) : ℝ :=
  let new_backpack_price := initial_backpack_price + backpack_price_increase
  let new_binder_price := initial_binder_price - binder_price_reduction
  let total_bindable_cost := min num_binders ((num_binders + 1) / 2 * new_binder_price)
  let total_pre_discount := new_backpack_price + total_bindable_cost
  let discount_amount := total_pre_discount * discount_rate / 100
  let total_price := total_pre_discount - discount_amount
  total_price

theorem julieta_total_cost
  (initial_backpack_price : ℕ)
  (initial_binder_price : ℕ)
  (backpack_price_increase : ℕ)
  (binder_price_reduction : ℕ)
  (discount_rate : ℕ)
  (num_binders : ℕ)
  (h_initial_backpack : initial_backpack_price = 50)
  (h_initial_binder : initial_binder_price = 20)
  (h_backpack_inc : backpack_price_increase = 5)
  (h_binder_red : binder_price_reduction = 2)
  (h_discount : discount_rate = 10)
  (h_num_binders : num_binders = 3) :
  calculate_total_cost initial_backpack_price initial_binder_price backpack_price_increase binder_price_reduction discount_rate num_binders = 81.90 :=
by
  sorry

end julieta_total_cost_l2205_220532


namespace total_paths_A_to_C_via_B_l2205_220545

-- Define the conditions
def steps_from_A_to_B : Nat := 6
def steps_from_B_to_C : Nat := 6
def right_moves_A_to_B : Nat := 4
def down_moves_A_to_B : Nat := 2
def right_moves_B_to_C : Nat := 3
def down_moves_B_to_C : Nat := 3

-- Define binomial coefficient function
def binom (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of paths for each segment
def paths_A_to_B : Nat := binom steps_from_A_to_B down_moves_A_to_B
def paths_B_to_C : Nat := binom steps_from_B_to_C down_moves_B_to_C

-- Theorem stating the total number of distinct paths
theorem total_paths_A_to_C_via_B : paths_A_to_B * paths_B_to_C = 300 :=
by
  sorry

end total_paths_A_to_C_via_B_l2205_220545


namespace one_third_of_four_l2205_220515

theorem one_third_of_four (h : 1/6 * 20 = 15) : 1/3 * 4 = 10 :=
sorry

end one_third_of_four_l2205_220515


namespace complement_intersection_l2205_220571

open Set

variable (U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8})
variable (A : Set ℕ := {2, 5, 8})
variable (B : Set ℕ := {1, 3, 5, 7})

theorem complement_intersection (CUA : Set ℕ := {1, 3, 4, 6, 7}) :
  (CUA ∩ B) = {1, 3, 7} := by
  sorry

end complement_intersection_l2205_220571


namespace max_min_K_max_min_2x_plus_y_l2205_220560

noncomputable def circle_equation (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1

theorem max_min_K (x y : ℝ) (h : circle_equation x y) : 
  - (Real.sqrt 3) / 3 ≤ (y - 1) / (x - 2) ∧ (y - 1) / (x - 2) ≤ (Real.sqrt 3) / 3 :=
by sorry

theorem max_min_2x_plus_y (x y : ℝ) (h : circle_equation x y) :
  1 - Real.sqrt 5 ≤ 2 * x + y ∧ 2 * x + y ≤ 1 + Real.sqrt 5 :=
by sorry

end max_min_K_max_min_2x_plus_y_l2205_220560


namespace girls_with_short_hair_count_l2205_220507

-- Definitions based on the problem's conditions
def TotalPeople := 55
def Boys := 30
def FractionLongHair : ℚ := 3 / 5

-- The statement to prove
theorem girls_with_short_hair_count :
  (TotalPeople - Boys) - (TotalPeople - Boys) * FractionLongHair = 10 :=
by
  sorry

end girls_with_short_hair_count_l2205_220507


namespace inverse_proportion_point_l2205_220527

theorem inverse_proportion_point (a : ℝ) (h : (a, 7) ∈ {p : ℝ × ℝ | ∃ x y, y = 14 / x ∧ p = (x, y)}) : a = 2 :=
by
  sorry

end inverse_proportion_point_l2205_220527


namespace green_block_weight_l2205_220516

theorem green_block_weight
    (y : ℝ)
    (g : ℝ)
    (h1 : y = 0.6)
    (h2 : y = g + 0.2) :
    g = 0.4 :=
by
  sorry

end green_block_weight_l2205_220516


namespace fifth_number_in_10th_row_l2205_220541

theorem fifth_number_in_10th_row : 
  ∀ (n : ℕ), (∃ (a : ℕ), ∀ (m : ℕ), 1 ≤ m ∧ m ≤ 10 → (m = 10 → a = 67)) :=
by
  sorry

end fifth_number_in_10th_row_l2205_220541


namespace part1_solution_part2_solution_l2205_220529

open Real

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) + abs (x - 3)

theorem part1_solution : ∀ x, f x ≤ 4 ↔ (0 ≤ x) ∧ (x ≤ 4) :=
by
  intro x
  sorry

theorem part2_solution : ∀ m, (∀ x, f x > m^2 + m) ↔ (-2 < m) ∧ (m < 1) :=
by
  intro m
  sorry

end part1_solution_part2_solution_l2205_220529


namespace tom_candy_pieces_l2205_220553

def total_boxes : ℕ := 14
def give_away_boxes : ℕ := 8
def pieces_per_box : ℕ := 3

theorem tom_candy_pieces : (total_boxes - give_away_boxes) * pieces_per_box = 18 := 
by 
  sorry

end tom_candy_pieces_l2205_220553


namespace polynomial_addition_l2205_220501

variable (x : ℝ)

def p := 3 * x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 2
def q := -3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 4

theorem polynomial_addition : p x + q x = -3 * x^3 + 2 * x^2 + 2 := by
  sorry

end polynomial_addition_l2205_220501


namespace people_present_l2205_220593

-- Define the number of parents, pupils, and teachers as constants
def p := 73
def s := 724
def t := 744

-- The theorem to prove the total number of people present
theorem people_present : p + s + t = 1541 := 
by
  -- Proof is inserted here
  sorry

end people_present_l2205_220593


namespace speed_of_current_l2205_220581

variable (c : ℚ) -- Speed of the current in miles per hour
variable (d : ℚ) -- Distance to the certain point in miles

def boat_speed := 16 -- Boat's speed relative to water in mph
def upstream_time := (20:ℚ) / 60 -- Time upstream in hours 
def downstream_time := (15:ℚ) / 60 -- Time downstream in hours

theorem speed_of_current (h1 : d = (boat_speed - c) * upstream_time)
                         (h2 : d = (boat_speed + c) * downstream_time) :
    c = 16 / 7 :=
  by
  sorry

end speed_of_current_l2205_220581


namespace smallest_positive_multiple_l2205_220584

theorem smallest_positive_multiple (a : ℕ) (k : ℕ) (h : 17 * a ≡ 7 [MOD 101]) : 
  ∃ k, k = 17 * 42 := 
sorry

end smallest_positive_multiple_l2205_220584


namespace xyz_product_neg4_l2205_220508

theorem xyz_product_neg4 (x y z : ℝ) (h1 : x + 2 / y = 2) (h2 : y + 2 / z = 2) : x * y * z = -4 :=
by {
  sorry
}

end xyz_product_neg4_l2205_220508


namespace chord_length_is_correct_l2205_220512

noncomputable def length_of_chord {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : Real :=
  2 * Real.sqrt 3

theorem chord_length_is_correct {ρ θ : Real} 
 (h_line : ρ * Real.sin (π / 6 - θ) = 2) 
 (h_curve : ρ = 4 * Real.cos θ) : 
 length_of_chord h_line h_curve = 2 * Real.sqrt 3 :=
sorry

end chord_length_is_correct_l2205_220512


namespace find_B_and_distance_l2205_220564

noncomputable def pointA : ℝ × ℝ := (2, 4)

noncomputable def pointB : ℝ × ℝ := (-(1 + Real.sqrt 385) / 8, (-(1 + Real.sqrt 385) / 8) ^ 2)

noncomputable def distanceToOrigin (p : ℝ × ℝ) : ℝ :=
  Real.sqrt (p.1 ^ 2 + p.2 ^ 2)

theorem find_B_and_distance :
  (pointA.snd = pointA.fst ^ 2) ∧
  (pointB.snd = (-(1 + Real.sqrt 385) / 8) ^ 2) ∧
  (distanceToOrigin pointB = Real.sqrt ((-(1 + Real.sqrt 385) / 8) ^ 2 + (-(1 + Real.sqrt 385) / 8) ^ 4)) :=
  sorry

end find_B_and_distance_l2205_220564


namespace solve_for_y_solve_for_x_l2205_220557

variable (x y : ℝ)

theorem solve_for_y (h : 2 * x + 3 * y - 4 = 0) : y = (4 - 2 * x) / 3 := 
sorry

theorem solve_for_x (h : 2 * x + 3 * y - 4 = 0) : x = (4 - 3 * y) / 2 := 
sorry

end solve_for_y_solve_for_x_l2205_220557


namespace num_satisfying_inequality_l2205_220589

theorem num_satisfying_inequality : ∃ (s : Finset ℤ), (∀ n ∈ s, (n + 4) * (n - 8) ≤ 0) ∧ s.card = 13 := by
  sorry

end num_satisfying_inequality_l2205_220589


namespace gcd_proof_l2205_220526

def gcd_10010_15015 := Nat.gcd 10010 15015 = 5005

theorem gcd_proof : gcd_10010_15015 :=
by
  sorry

end gcd_proof_l2205_220526


namespace problem_statement_l2205_220534

noncomputable def a : ℝ := 2 * Real.log 0.99
noncomputable def b : ℝ := Real.log 0.98
noncomputable def c : ℝ := Real.sqrt 0.96 - 1

theorem problem_statement : a > b ∧ b > c := by
  sorry

end problem_statement_l2205_220534


namespace gallons_of_soup_l2205_220539

def bowls_per_minute : ℕ := 5
def ounces_per_bowl : ℕ := 10
def serving_time_minutes : ℕ := 15
def ounces_per_gallon : ℕ := 128

theorem gallons_of_soup :
  (5 * 10 * 15 / 128) = 6 :=
by
  sorry

end gallons_of_soup_l2205_220539
