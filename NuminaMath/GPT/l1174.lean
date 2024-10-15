import Mathlib

namespace NUMINAMATH_GPT_max_digit_sum_in_24_hour_format_l1174_117484

def digit_sum (n : ℕ) : ℕ := 
  (n / 10) + (n % 10)

theorem max_digit_sum_in_24_hour_format :
  (∃ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ digit_sum h + digit_sum m = 19) ∧
  ∀ (h m : ℕ), 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 → digit_sum h + digit_sum m ≤ 19 :=
by
  sorry

end NUMINAMATH_GPT_max_digit_sum_in_24_hour_format_l1174_117484


namespace NUMINAMATH_GPT_remaining_distance_l1174_117462

-- Definitions based on the conditions
def total_distance : ℕ := 78
def first_leg : ℕ := 35
def second_leg : ℕ := 18

-- The theorem we want to prove
theorem remaining_distance : total_distance - (first_leg + second_leg) = 25 := by
  sorry

end NUMINAMATH_GPT_remaining_distance_l1174_117462


namespace NUMINAMATH_GPT_proof_of_equivalence_l1174_117412

variables (x y : ℝ)

def expression := 49 * x^2 - 36 * y^2
def optionD := (-6 * y + 7 * x) * (6 * y + 7 * x)

theorem proof_of_equivalence : expression x y = optionD x y := 
by sorry

end NUMINAMATH_GPT_proof_of_equivalence_l1174_117412


namespace NUMINAMATH_GPT_wood_length_equation_l1174_117467

theorem wood_length_equation (x : ℝ) : 
  (∃ r : ℝ, r - x = 4.5 ∧ r/2 + 1 = x) → 1/2 * (x + 4.5) = x - 1 :=
sorry

end NUMINAMATH_GPT_wood_length_equation_l1174_117467


namespace NUMINAMATH_GPT_kayla_waiting_years_l1174_117483

def minimum_driving_age : ℕ := 18
def kimiko_age : ℕ := 26
def kayla_age : ℕ := kimiko_age / 2
def years_until_kayla_can_drive : ℕ := minimum_driving_age - kayla_age

theorem kayla_waiting_years : years_until_kayla_can_drive = 5 :=
by
  sorry

end NUMINAMATH_GPT_kayla_waiting_years_l1174_117483


namespace NUMINAMATH_GPT_each_group_has_two_bananas_l1174_117481

theorem each_group_has_two_bananas (G T : ℕ) (hG : G = 196) (hT : T = 392) : T / G = 2 :=
by
  sorry

end NUMINAMATH_GPT_each_group_has_two_bananas_l1174_117481


namespace NUMINAMATH_GPT_negative_value_option_D_l1174_117490

theorem negative_value_option_D :
  (-7) * (-6) > 0 ∧
  (-7) - (-15) > 0 ∧
  0 * (-2) * (-3) = 0 ∧
  (-6) + (-4) < 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_value_option_D_l1174_117490


namespace NUMINAMATH_GPT_smallest_with_20_divisors_is_144_l1174_117464

def has_exactly_20_divisors (n : ℕ) : Prop :=
  let factors := n.factors;
  let divisors_count := factors.foldr (λ a b => (a + 1) * b) 1;
  divisors_count = 20

theorem smallest_with_20_divisors_is_144 : ∀ (n : ℕ), has_exactly_20_divisors n → (n < 144) → False :=
by
  sorry

end NUMINAMATH_GPT_smallest_with_20_divisors_is_144_l1174_117464


namespace NUMINAMATH_GPT_least_possible_value_z_minus_x_l1174_117497

theorem least_possible_value_z_minus_x (x y z : ℤ) (h1 : Even x) (h2 : Odd y) (h3 : Odd z) (h4 : x < y) (h5 : y < z) (h6 : y - x > 5) : z - x = 9 := 
sorry

end NUMINAMATH_GPT_least_possible_value_z_minus_x_l1174_117497


namespace NUMINAMATH_GPT_Wayne_initially_collected_blocks_l1174_117420

-- Let's denote the initial blocks collected by Wayne as 'w'.
-- According to the problem:
-- - Wayne's father gave him 6 more blocks.
-- - He now has 15 blocks in total.
--
-- We need to prove that the initial number of blocks Wayne collected (w) is 9.

theorem Wayne_initially_collected_blocks : 
  ∃ w : ℕ, (w + 6 = 15) ↔ (w = 9) := by
  sorry

end NUMINAMATH_GPT_Wayne_initially_collected_blocks_l1174_117420


namespace NUMINAMATH_GPT_number_of_valid_pairings_l1174_117433

-- Definition for the problem
def validPairingCount (n : ℕ) (k: ℕ) : ℕ :=
  sorry -- Calculating the valid number of pairings is deferred

-- The problem statement to be proven:
theorem number_of_valid_pairings : validPairingCount 12 3 = 14 :=
sorry

end NUMINAMATH_GPT_number_of_valid_pairings_l1174_117433


namespace NUMINAMATH_GPT_roots_of_equation_l1174_117415

theorem roots_of_equation : ∀ x : ℝ, x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by sorry

end NUMINAMATH_GPT_roots_of_equation_l1174_117415


namespace NUMINAMATH_GPT_valentines_left_l1174_117410

theorem valentines_left (initial valentines_to_children valentines_to_neighbors valentines_to_coworkers : ℕ) (h_initial : initial = 30)
  (h1 : valentines_to_children = 8) (h2 : valentines_to_neighbors = 5) (h3 : valentines_to_coworkers = 3) : initial - (valentines_to_children + valentines_to_neighbors + valentines_to_coworkers) = 14 := by
  sorry

end NUMINAMATH_GPT_valentines_left_l1174_117410


namespace NUMINAMATH_GPT_two_digit_sum_reverse_l1174_117428

theorem two_digit_sum_reverse (a b : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ 9)
    (h₃ : 0 ≤ b) (h₄ : b ≤ 9)
    (h₅ : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
    (10 * a + b) + (10 * b + a) = 99 := 
by
  sorry

end NUMINAMATH_GPT_two_digit_sum_reverse_l1174_117428


namespace NUMINAMATH_GPT_simplify_fraction_l1174_117482

theorem simplify_fraction : 5 * (21 / 8) * (32 / -63) = -20 / 3 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1174_117482


namespace NUMINAMATH_GPT_parallelogram_perimeter_l1174_117422

theorem parallelogram_perimeter 
  (EF FG EH : ℝ)
  (hEF : EF = 40) (hFG : FG = 30) (hEH : EH = 50) : 
  2 * (EF + FG) = 140 := 
by 
  rw [hEF, hFG]
  norm_num

end NUMINAMATH_GPT_parallelogram_perimeter_l1174_117422


namespace NUMINAMATH_GPT_tv_weight_calculations_l1174_117437

theorem tv_weight_calculations
    (w1 h1 r1 : ℕ) -- Represents Bill's TV dimensions and weight ratio
    (w2 h2 r2 : ℕ) -- Represents Bob's TV dimensions and weight ratio
    (w3 h3 r3 : ℕ) -- Represents Steve's TV dimensions and weight ratio
    (ounce_to_pound: ℕ) -- Represents the conversion factor from ounces to pounds
    (bill_tv_weight bob_tv_weight steve_tv_weight : ℕ) -- Computed weights in pounds
    (weight_diff: ℕ):
  (w1 * h1 * r1) / ounce_to_pound = bill_tv_weight → -- Bill's TV weight calculation
  (w2 * h2 * r2) / ounce_to_pound = bob_tv_weight → -- Bob's TV weight calculation
  (w3 * h3 * r3) / ounce_to_pound = steve_tv_weight → -- Steve's TV weight calculation
  steve_tv_weight > (bill_tv_weight + bob_tv_weight) → -- Steve's TV is the heaviest
  steve_tv_weight - (bill_tv_weight + bob_tv_weight) = weight_diff → -- weight difference calculation
  True := sorry

end NUMINAMATH_GPT_tv_weight_calculations_l1174_117437


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1174_117427

theorem sufficient_but_not_necessary_condition (x : ℝ) : (x > 0 → |x| > 0) ∧ (¬ (|x| > 0 → x > 0)) :=
by 
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1174_117427


namespace NUMINAMATH_GPT_cannot_determine_right_triangle_l1174_117499

def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

theorem cannot_determine_right_triangle :
  ∀ A B C : ℝ, 
    (A = 2 * B ∧ A = 3 * C) →
    ¬ is_right_triangle A B C :=
by
  intro A B C h
  have h1 : A = 2 * B := h.1
  have h2 : A = 3 * C := h.2
  sorry

end NUMINAMATH_GPT_cannot_determine_right_triangle_l1174_117499


namespace NUMINAMATH_GPT_right_triangle_third_side_l1174_117463

theorem right_triangle_third_side (a b c : ℝ) (h₁ : a = 3) (h₂ : b = 4) (h₃ : c = 5) (h₄ : c = Real.sqrt (7) ∨ c = 5) :
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 := by
  sorry

end NUMINAMATH_GPT_right_triangle_third_side_l1174_117463


namespace NUMINAMATH_GPT_sweater_markup_percentage_l1174_117449

variables (W R : ℝ)
variables (h1 : 0.30 * R = 1.40 * W)

theorem sweater_markup_percentage :
  (R = (1.40 / 0.30) * W) →
  (R - W) / W * 100 = 367 := 
by
  intro hR
  sorry

end NUMINAMATH_GPT_sweater_markup_percentage_l1174_117449


namespace NUMINAMATH_GPT_exp_ineq_of_r_gt_one_l1174_117444

theorem exp_ineq_of_r_gt_one {x r : ℝ} (hx : x > 0) (hr : r > 1) : (1 + x)^r > 1 + r * x :=
by
  sorry

end NUMINAMATH_GPT_exp_ineq_of_r_gt_one_l1174_117444


namespace NUMINAMATH_GPT_person_speed_l1174_117452

theorem person_speed (d_meters : ℕ) (t_minutes : ℕ) (d_km t_hours : ℝ) :
  (d_meters = 1800) →
  (t_minutes = 12) →
  (d_km = d_meters / 1000) →
  (t_hours = t_minutes / 60) →
  d_km / t_hours = 9 :=
by
  intros
  sorry

end NUMINAMATH_GPT_person_speed_l1174_117452


namespace NUMINAMATH_GPT_sinB_law_of_sines_l1174_117448

variable (A B C : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Assuming a triangle with sides and angles as described
variable (a b : ℝ) (sinA sinB : ℝ)
variable (h₁ : a = 3) (h₂ : b = 5) (h₃ : sinA = 1 / 3)

theorem sinB_law_of_sines : sinB = 5 / 9 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_sinB_law_of_sines_l1174_117448


namespace NUMINAMATH_GPT_blocks_differ_in_two_ways_l1174_117457

/-- 
A child has a set of 120 distinct blocks. Each block is one of 3 materials (plastic, wood, metal), 
3 sizes (small, medium, large), 4 colors (blue, green, red, yellow), and 5 shapes (circle, hexagon, 
square, triangle, pentagon). How many blocks in the set differ from the 'metal medium blue hexagon' 
in exactly 2 ways?
-/
def num_blocks_differ_in_two_ways : Nat := 44

theorem blocks_differ_in_two_ways (blocks : Fin 120)
    (materials : Fin 3)
    (sizes : Fin 3)
    (colors : Fin 4)
    (shapes : Fin 5)
    (fixed_block : {m // m = 2} × {s // s = 1} × {c // c = 0} × {sh // sh = 1}) :
    num_blocks_differ_in_two_ways = 44 :=
by
  -- proof steps are omitted
  sorry

end NUMINAMATH_GPT_blocks_differ_in_two_ways_l1174_117457


namespace NUMINAMATH_GPT_overlap_coordinates_l1174_117431

theorem overlap_coordinates :
  ∃ m n : ℝ, 
    (m + n = 6.8) ∧ 
    ((2 * (7 + m) / 2 - 3) = (3 + n) / 2) ∧ 
    ((2 * (7 + m) / 2 - 3) = - (m - 7) / 2) :=
by
  sorry

end NUMINAMATH_GPT_overlap_coordinates_l1174_117431


namespace NUMINAMATH_GPT_cricket_target_runs_l1174_117488

def run_rate_first_20_overs : ℝ := 4.2
def overs_first_20 : ℝ := 20
def run_rate_remaining_30_overs : ℝ := 5.533333333333333
def overs_remaining_30 : ℝ := 30
def total_runs_first_20 : ℝ := run_rate_first_20_overs * overs_first_20
def total_runs_remaining_30 : ℝ := run_rate_remaining_30_overs * overs_remaining_30

theorem cricket_target_runs :
  (total_runs_first_20 + total_runs_remaining_30) = 250 :=
by
  sorry

end NUMINAMATH_GPT_cricket_target_runs_l1174_117488


namespace NUMINAMATH_GPT_range_of_a_l1174_117478

variable {x a : ℝ}

def p (x a : ℝ) : Prop := x^2 - 4*a*x + 3*a^2 < 0
def q (x : ℝ) : Prop := x^2 - 6*x + 8 > 0

theorem range_of_a (h : (∀ x a, p x a → q x) ∧ (∃ x a, q x ∧ ¬ p x a)) :
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1174_117478


namespace NUMINAMATH_GPT_remove_6_maximizes_probability_l1174_117402

def original_list : List Int := [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

-- Define what it means to maximize the probability of pairs summing to 12
def maximize_probability (l : List Int) : Prop :=
  ∀ x y, x ≠ y → x ∈ l → y ∈ l → x + y = 12

-- Prove that removing 6 maximizes the probability that the sum of the two chosen numbers is 12
theorem remove_6_maximizes_probability :
  maximize_probability (original_list.erase 6) :=
sorry

end NUMINAMATH_GPT_remove_6_maximizes_probability_l1174_117402


namespace NUMINAMATH_GPT_no_digit_B_divisible_by_4_l1174_117474

theorem no_digit_B_divisible_by_4 : 
  ∀ B : ℕ, B < 10 → ¬ (8 * 1000000 + B * 100000 + 4 * 10000 + 6 * 1000 + 3 * 100 + 5 * 10 + 1) % 4 = 0 :=
by
  intros B hB_lt_10
  sorry

end NUMINAMATH_GPT_no_digit_B_divisible_by_4_l1174_117474


namespace NUMINAMATH_GPT_fruits_calculation_l1174_117425

structure FruitStatus :=
  (initial_picked  : ℝ)
  (initial_eaten  : ℝ)

def apples_status : FruitStatus :=
  { initial_picked := 7.0 + 3.0 + 5.0, initial_eaten := 6.0 + 2.0 }

def pears_status : FruitStatus :=
  { initial_picked := 0, initial_eaten := 4.0 + 3.0 }  -- number of pears picked is unknown, hence 0

def oranges_status : FruitStatus :=
  { initial_picked := 8.0, initial_eaten := 8.0 }

def cherries_status : FruitStatus :=
  { initial_picked := 4.0, initial_eaten := 4.0 }

theorem fruits_calculation :
  (apples_status.initial_picked - apples_status.initial_eaten = 7.0) ∧
  (pears_status.initial_picked - pears_status.initial_eaten = 0) ∧  -- cannot be determined in the problem statement
  (oranges_status.initial_picked - oranges_status.initial_eaten = 0) ∧
  (cherries_status.initial_picked - cherries_status.initial_eaten = 0) :=
by {
  sorry
}

end NUMINAMATH_GPT_fruits_calculation_l1174_117425


namespace NUMINAMATH_GPT_basketball_team_total_players_l1174_117451

theorem basketball_team_total_players (total_points : ℕ) (min_points : ℕ) (max_points : ℕ) (team_size : ℕ)
  (h1 : total_points = 100)
  (h2 : min_points = 7)
  (h3 : max_points = 23)
  (h4 : ∀ (n : ℕ), n ≥ min_points)
  (h5 : max_points = 23)
  : team_size = 12 :=
sorry

end NUMINAMATH_GPT_basketball_team_total_players_l1174_117451


namespace NUMINAMATH_GPT_initial_machines_count_l1174_117438

theorem initial_machines_count (M : ℕ) (h1 : M * 8 = 8 * 1) (h2 : 72 * 6 = 12 * 2) : M = 64 :=
by
  sorry

end NUMINAMATH_GPT_initial_machines_count_l1174_117438


namespace NUMINAMATH_GPT_loss_percentage_is_five_l1174_117429

/-- Definitions -/
def original_price : ℝ := 490
def sold_price : ℝ := 465.50
def loss_amount : ℝ := original_price - sold_price

/-- Theorem -/
theorem loss_percentage_is_five :
  (loss_amount / original_price) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_is_five_l1174_117429


namespace NUMINAMATH_GPT_find_initial_time_l1174_117487

-- The initial distance d
def distance : ℕ := 288

-- Conditions
def initial_condition (v t : ℕ) : Prop :=
  distance = v * t

def new_condition (t : ℕ) : Prop :=
  distance = 32 * (3 * t / 2)

-- Proof Problem Statement
theorem find_initial_time (v t : ℕ) (h1 : initial_condition v t)
  (h2 : new_condition t) : t = 6 := by
  sorry

end NUMINAMATH_GPT_find_initial_time_l1174_117487


namespace NUMINAMATH_GPT_carson_gets_clawed_39_times_l1174_117436

-- Conditions
def number_of_wombats : ℕ := 9
def claws_per_wombat : ℕ := 4
def number_of_rheas : ℕ := 3
def claws_per_rhea : ℕ := 1

-- Theorem statement
theorem carson_gets_clawed_39_times :
  (number_of_wombats * claws_per_wombat + number_of_rheas * claws_per_rhea) = 39 :=
by
  sorry

end NUMINAMATH_GPT_carson_gets_clawed_39_times_l1174_117436


namespace NUMINAMATH_GPT_range_of_a_l1174_117496

noncomputable
def proposition_p (x : ℝ) : Prop := abs (x - (3 / 4)) <= (1 / 4)
noncomputable
def proposition_q (x a : ℝ) : Prop := (x - a) * (x - a - 1) <= 0

theorem range_of_a :
  (∀ x : ℝ, proposition_p x → ∃ x : ℝ, proposition_q x a) ∧
  (∃ x : ℝ, ¬(proposition_p x → proposition_q x a )) →
  0 ≤ a ∧ a ≤ (1 / 2) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1174_117496


namespace NUMINAMATH_GPT_typing_difference_l1174_117417

theorem typing_difference (m : ℕ) (h1 : 10 * m - 8 * m = 10) : m = 5 :=
by
  sorry

end NUMINAMATH_GPT_typing_difference_l1174_117417


namespace NUMINAMATH_GPT_keith_attended_games_l1174_117442

-- Definitions from the conditions
def total_games : ℕ := 20
def missed_games : ℕ := 9

-- The statement to prove
theorem keith_attended_games : (total_games - missed_games) = 11 :=
by
  sorry

end NUMINAMATH_GPT_keith_attended_games_l1174_117442


namespace NUMINAMATH_GPT_zeroSeq_arithmetic_not_geometric_l1174_117460

-- Define what it means for a sequence to be arithmetic
def isArithmeticSequence (seq : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, seq (n + 1) = seq n + d

-- Define what it means for a sequence to be geometric
def isGeometricSequence (seq : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, seq n ≠ 0 → seq (n + 1) = seq n * q

-- Define the sequence of zeros
def zeroSeq (n : ℕ) : ℝ := 0

theorem zeroSeq_arithmetic_not_geometric :
  isArithmeticSequence zeroSeq ∧ ¬ isGeometricSequence zeroSeq :=
by
  sorry

end NUMINAMATH_GPT_zeroSeq_arithmetic_not_geometric_l1174_117460


namespace NUMINAMATH_GPT_total_votes_cast_l1174_117445

-- Define the variables and constants
def total_votes (V : ℝ) : Prop :=
  let A := 0.32 * V
  let B := 0.28 * V
  let C := 0.22 * V
  let D := 0.18 * V
  -- Candidate A defeated Candidate B by 1200 votes
  0.32 * V - 0.28 * V = 1200 ∧
  -- Candidate A defeated Candidate C by 2200 votes
  0.32 * V - 0.22 * V = 2200 ∧
  -- Candidate B defeated Candidate D by 900 votes
  0.28 * V - 0.18 * V = 900

noncomputable def V := 30000

-- State the theorem
theorem total_votes_cast : total_votes V := by
  sorry

end NUMINAMATH_GPT_total_votes_cast_l1174_117445


namespace NUMINAMATH_GPT_simplify_expression_l1174_117411

theorem simplify_expression (x : ℝ) (hx : x ≠ 0) : 
  (18 * x^3) * (4 * x^2) * (1 / (2 * x)^3) = 9 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1174_117411


namespace NUMINAMATH_GPT_solve_equation_l1174_117456

theorem solve_equation :
  ∃ x : ℝ, (x - 2)^2 - (x + 3) * (x - 3) = 4 * x - 1 ∧ x = 7 / 4 := 
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1174_117456


namespace NUMINAMATH_GPT_edwards_initial_money_l1174_117466

variable (spent1 spent2 current remaining : ℕ)

def initial_money (spent1 spent2 current remaining : ℕ) : ℕ :=
  spent1 + spent2 + current

theorem edwards_initial_money :
  spent1 = 9 → spent2 = 8 → remaining = 17 →
  initial_money spent1 spent2 remaining remaining = 34 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_edwards_initial_money_l1174_117466


namespace NUMINAMATH_GPT_car_travel_time_l1174_117407

noncomputable def travelTimes 
  (t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime : ℝ) : Prop :=
t_Ningi_Zipra = 0.80 * t_Ngapara_Zipra ∧
t_Ngapara_Zipra = 60 ∧
totalTravelTime = t_Ngapara_Zipra + t_Ningi_Zipra

theorem car_travel_time :
  ∃ t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime,
  travelTimes t_Ngapara_Zipra t_Ningi_Zipra totalTravelTime ∧
  totalTravelTime = 108 :=
by
  sorry

end NUMINAMATH_GPT_car_travel_time_l1174_117407


namespace NUMINAMATH_GPT_range_of_a_l1174_117405

noncomputable def f (x a : ℝ) : ℝ :=
  if x ≤ 1 then 2^(|x - a|) else x + 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a ≥ f 1 a) ↔ (1 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1174_117405


namespace NUMINAMATH_GPT_truncated_cone_volume_l1174_117401

theorem truncated_cone_volume :
  let R := 10
  let r := 5
  let h_t := 10
  let V_large := (1/3:Real) * Real.pi * (R^2) * (20)
  let V_small := (1/3:Real) * Real.pi * (r^2) * (10)
  (V_large - V_small) = (1750/3) * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_truncated_cone_volume_l1174_117401


namespace NUMINAMATH_GPT_polygon_side_count_l1174_117473

theorem polygon_side_count (s : ℝ) (hs : s ≠ 0) : 
  ∀ (side_length_ratio : ℝ) (sides_first sides_second : ℕ),
  sides_first = 50 ∧ side_length_ratio = 3 ∧ 
  sides_first * side_length_ratio * s = sides_second * s → sides_second = 150 :=
by
  sorry

end NUMINAMATH_GPT_polygon_side_count_l1174_117473


namespace NUMINAMATH_GPT_factorize_cubic_l1174_117447

theorem factorize_cubic (x : ℝ) : x^3 - x = x * (x + 1) * (x - 1) :=
by sorry

end NUMINAMATH_GPT_factorize_cubic_l1174_117447


namespace NUMINAMATH_GPT_function_properties_l1174_117470

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 6)

theorem function_properties :
  (∀ x, f (x + Real.pi) = f x) ∧
  (f (Real.pi / 3) = 1) ∧
  (∀ x y, -Real.pi / 6 ≤ x → x ≤ y → y ≤ Real.pi / 3 → f x ≤ f y) := by
  sorry

end NUMINAMATH_GPT_function_properties_l1174_117470


namespace NUMINAMATH_GPT_sum_num_den_252_l1174_117404

theorem sum_num_den_252 (h : (252 : ℤ) / 100 = (63 : ℤ) / 25) : 63 + 25 = 88 :=
by
  sorry

end NUMINAMATH_GPT_sum_num_den_252_l1174_117404


namespace NUMINAMATH_GPT_today_is_thursday_l1174_117430

-- Define the days of the week as an enumerated type
inductive DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define the conditions for the lion and the unicorn
def lion_truth (d: DayOfWeek) : Bool :=
match d with
| Monday | Tuesday | Wednesday => false
| _ => true

def unicorn_truth (d: DayOfWeek) : Bool :=
match d with
| Thursday | Friday | Saturday => false
| _ => true

-- The statement made by the lion and the unicorn
def lion_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => lion_truth Sunday
| Tuesday => lion_truth Monday
| Wednesday => lion_truth Tuesday
| Thursday => lion_truth Wednesday
| Friday => lion_truth Thursday
| Saturday => lion_truth Friday
| Sunday => lion_truth Saturday

def unicorn_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => unicorn_truth Sunday
| Tuesday => unicorn_truth Monday
| Wednesday => unicorn_truth Tuesday
| Thursday => unicorn_truth Wednesday
| Friday => unicorn_truth Thursday
| Saturday => unicorn_truth Friday
| Sunday => unicorn_truth Saturday

-- Main theorem to prove the current day
theorem today_is_thursday (d: DayOfWeek) (lion_said: lion_statement d = false) (unicorn_said: unicorn_statement d = false) : d = Thursday :=
by
  -- Placeholder for actual proof
  sorry

end NUMINAMATH_GPT_today_is_thursday_l1174_117430


namespace NUMINAMATH_GPT_triangle_third_side_length_l1174_117403

theorem triangle_third_side_length 
  (a b c : ℝ) (ha : a = 7) (hb : b = 11) (hc : c = 3) :
  (4 < c ∧ c < 18) → c ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_triangle_third_side_length_l1174_117403


namespace NUMINAMATH_GPT_eliminate_denominators_l1174_117441

theorem eliminate_denominators (x : ℝ) :
  (4 * (2 * x - 1) - 3 * (3 * x - 4) = 12) ↔ ((2 * x - 1) / 3 - (3 * x - 4) / 4 = 1) := 
by
  sorry

end NUMINAMATH_GPT_eliminate_denominators_l1174_117441


namespace NUMINAMATH_GPT_two_squares_inequality_l1174_117492

theorem two_squares_inequality (a b : ℝ) : 2 * (a^2 + b^2) ≥ (a + b)^2 := 
sorry

end NUMINAMATH_GPT_two_squares_inequality_l1174_117492


namespace NUMINAMATH_GPT_gcd_204_85_l1174_117432

theorem gcd_204_85 : Nat.gcd 204 85 = 17 := by
  have h1 : 204 = 2 * 85 + 34 := by rfl
  have h2 : 85 = 2 * 34 + 17 := by rfl
  have h3 : 34 = 2 * 17 := by rfl
  sorry

end NUMINAMATH_GPT_gcd_204_85_l1174_117432


namespace NUMINAMATH_GPT_johns_burritos_l1174_117418

-- Definitions based on conditions:
def initial_burritos : Nat := 3 * 20
def burritos_given_away : Nat := initial_burritos / 3
def burritos_after_giving_away : Nat := initial_burritos - burritos_given_away
def burritos_eaten : Nat := 3 * 10
def burritos_left : Nat := burritos_after_giving_away - burritos_eaten

-- The theorem we need to prove:
theorem johns_burritos : burritos_left = 10 := by
  sorry

end NUMINAMATH_GPT_johns_burritos_l1174_117418


namespace NUMINAMATH_GPT_solve_inequality_system_l1174_117480

theorem solve_inequality_system :
  (∀ x : ℝ, (1 - 3 * (x - 1) < 8 - x) ∧ ((x - 3) / 2 + 2 ≥ x)) →
  ∃ (integers : Set ℤ), integers = {x : ℤ | -2 < (x : ℝ) ∧ (x : ℝ) ≤ 1} ∧ integers = {-1, 0, 1} :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_system_l1174_117480


namespace NUMINAMATH_GPT_product_plus_one_eq_216_l1174_117468

variable (a b c : ℝ)

theorem product_plus_one_eq_216 
  (h1 : a * b + a + b = 35)
  (h2 : b * c + b + c = 35)
  (h3 : c * a + c + a = 35)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a + 1) * (b + 1) * (c + 1) = 216 := 
sorry

end NUMINAMATH_GPT_product_plus_one_eq_216_l1174_117468


namespace NUMINAMATH_GPT_DollOutfit_l1174_117435

variables (VeraDress OlyaCoat VeraCoat NinaCoat : Prop)
axiom FirstAnswer : (VeraDress ∧ ¬OlyaCoat) ∨ (¬VeraDress ∧ OlyaCoat)
axiom SecondAnswer : (VeraCoat ∧ ¬NinaCoat) ∨ (¬VeraCoat ∧ NinaCoat)
axiom OnlyOneTrueFirstAnswer : (VeraDress ∨ OlyaCoat) ∧ ¬(VeraDress ∧ OlyaCoat)
axiom OnlyOneTrueSecondAnswer : (VeraCoat ∨ NinaCoat) ∧ ¬(VeraCoat ∧ NinaCoat)

theorem DollOutfit :
  VeraDress ∧ NinaCoat ∧ ¬OlyaCoat ∧ ¬VeraCoat ∧ ¬NinaCoat :=
sorry

end NUMINAMATH_GPT_DollOutfit_l1174_117435


namespace NUMINAMATH_GPT_find_k_l1174_117413

theorem find_k (k l : ℝ) (C : ℝ × ℝ) (OC : ℝ) (A B D : ℝ × ℝ)
  (hC_coords : C = (0, 3))
  (hl_val : l = 3)
  (line_eqn : ∀ x, y = k * x + l)
  (intersect_eqn : ∀ x, y = 1 / x)
  (hA_coords : A = (1 / 6, 6))
  (hD_coords : D = (1 / 6, 6))
  (dist_ABC : dist A B = dist B C)
  (dist_BCD : dist B C = dist C D)
  (OC_val : OC = 3) :
  k = 18 := 
sorry

end NUMINAMATH_GPT_find_k_l1174_117413


namespace NUMINAMATH_GPT_sqrt_of_225_eq_15_l1174_117443

theorem sqrt_of_225_eq_15 : Real.sqrt 225 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_of_225_eq_15_l1174_117443


namespace NUMINAMATH_GPT_miles_driven_each_day_l1174_117440

theorem miles_driven_each_day
  (total_distance : ℕ)
  (days_in_semester : ℕ)
  (h_total : total_distance = 1600)
  (h_days : days_in_semester = 80):
  total_distance / days_in_semester = 20 := by
  sorry

end NUMINAMATH_GPT_miles_driven_each_day_l1174_117440


namespace NUMINAMATH_GPT_toys_produced_each_day_l1174_117479

def toys_produced_per_week : ℕ := 6000
def work_days_per_week : ℕ := 4

theorem toys_produced_each_day :
  (toys_produced_per_week / work_days_per_week) = 1500 := 
by
  -- The details of the proof are omitted
  -- The correct answer given the conditions is 1500 toys
  sorry

end NUMINAMATH_GPT_toys_produced_each_day_l1174_117479


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1174_117439

theorem necessary_but_not_sufficient (x : ℝ) (h : x ≠ 1) : x^2 - 3 * x + 2 ≠ 0 :=
by
  intro h1
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1174_117439


namespace NUMINAMATH_GPT_reflection_transformation_l1174_117421

structure Point (α : Type) :=
(x : α)
(y : α)

def reflect_x_axis (p : Point ℝ) : Point ℝ :=
  {x := p.x, y := -p.y}

def reflect_x_eq_3 (p : Point ℝ) : Point ℝ :=
  {x := 6 - p.x, y := p.y}

def D : Point ℝ := {x := 4, y := 1}

def D' := reflect_x_axis D

def D'' := reflect_x_eq_3 D'

theorem reflection_transformation :
  D'' = {x := 2, y := -1} :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_reflection_transformation_l1174_117421


namespace NUMINAMATH_GPT_tetrahedron_vertex_equality_l1174_117455

theorem tetrahedron_vertex_equality
  (r1 r2 r3 r4 j1 j2 j3 j4 : ℝ) (hr1 : r1 > 0) (hr2 : r2 > 0) (hr3 : r3 > 0) (hr4 : r4 > 0)
  (hj1 : j1 > 0) (hj2 : j2 > 0) (hj3 : j3 > 0) (hj4 : j4 > 0) 
  (h1 : r2 * r3 + r3 * r4 + r4 * r2 = j2 * j3 + j3 * j4 + j4 * j2)
  (h2 : r1 * r3 + r3 * r4 + r4 * r1 = j1 * j3 + j3 * j4 + j4 * j1)
  (h3 : r1 * r2 + r2 * r4 + r4 * r1 = j1 * j2 + j2 * j4 + j4 * j1)
  (h4 : r1 * r2 + r2 * r3 + r3 * r1 = j1 * j2 + j2 * j3 + j3 * j1) :
  r1 = j1 ∧ r2 = j2 ∧ r3 = j3 ∧ r4 = j4 := by
  sorry

end NUMINAMATH_GPT_tetrahedron_vertex_equality_l1174_117455


namespace NUMINAMATH_GPT_evaluate_expression_l1174_117494

-- Given conditions 
def x := 3
def y := 2

-- Prove that y + y(y^x + x!) evaluates to 30.
theorem evaluate_expression : y + y * (y^x + Nat.factorial x) = 30 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1174_117494


namespace NUMINAMATH_GPT_alex_score_correct_l1174_117416

-- Conditions of the problem
def num_students := 20
def average_first_19 := 78
def new_average := 79

-- Alex's score calculation
def alex_score : ℕ :=
  let total_score_first_19 := 19 * average_first_19
  let total_score_all := num_students * new_average
  total_score_all - total_score_first_19

-- Problem statement: Prove Alex's score is 98
theorem alex_score_correct : alex_score = 98 := by
  sorry

end NUMINAMATH_GPT_alex_score_correct_l1174_117416


namespace NUMINAMATH_GPT_misread_weight_l1174_117486

theorem misread_weight (avg_initial : ℝ) (avg_correct : ℝ) (n : ℕ) (actual_weight : ℝ) (x : ℝ) : 
  avg_initial = 58.4 → avg_correct = 58.7 → n = 20 → actual_weight = 62 → 
  (n * avg_correct - n * avg_initial = actual_weight - x) → x = 56 :=
by
  intros
  sorry

end NUMINAMATH_GPT_misread_weight_l1174_117486


namespace NUMINAMATH_GPT_no_common_solution_l1174_117450

theorem no_common_solution :
  ¬(∃ y : ℚ, (6 * y^2 + 11 * y - 1 = 0) ∧ (18 * y^2 + y - 1 = 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_common_solution_l1174_117450


namespace NUMINAMATH_GPT_average_salary_rest_l1174_117471

theorem average_salary_rest (total_workers : ℕ) (avg_salary_all : ℝ)
  (num_technicians : ℕ) (avg_salary_technicians : ℝ) :
  total_workers = 21 →
  avg_salary_all = 8000 →
  num_technicians = 7 →
  avg_salary_technicians = 12000 →
  (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / (total_workers - num_technicians) = 6000 :=
by intros h1 h2 h3 h4; sorry

end NUMINAMATH_GPT_average_salary_rest_l1174_117471


namespace NUMINAMATH_GPT_percent_full_time_more_than_three_years_l1174_117489

variable (total_associates : ℕ)
variable (second_year_percentage : ℕ)
variable (third_year_percentage : ℕ)
variable (non_first_year_percentage : ℕ)
variable (part_time_percentage : ℕ)
variable (part_time_more_than_two_years_percentage : ℕ)
variable (full_time_more_than_three_years_percentage : ℕ)

axiom condition_1 : second_year_percentage = 30
axiom condition_2 : third_year_percentage = 20
axiom condition_3 : non_first_year_percentage = 60
axiom condition_4 : part_time_percentage = 10
axiom condition_5 : part_time_more_than_two_years_percentage = 5

theorem percent_full_time_more_than_three_years : 
  full_time_more_than_three_years_percentage = 10 := 
sorry

end NUMINAMATH_GPT_percent_full_time_more_than_three_years_l1174_117489


namespace NUMINAMATH_GPT_number_of_common_tangents_l1174_117475

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem number_of_common_tangents
  (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ)
  (h₁ : ∀ (x y : ℝ), x^2 + y^2 - 2 * x = 0 → (C₁ = (1, 0)) ∧ (r₁ = 1))
  (h₂ : ∀ (x y : ℝ), x^2 + y^2 - 4 * y + 3 = 0 → (C₂ = (0, 2)) ∧ (r₂ = 1))
  (d : distance C₁ C₂ = Real.sqrt 5) :
  4 = 4 := 
by sorry

end NUMINAMATH_GPT_number_of_common_tangents_l1174_117475


namespace NUMINAMATH_GPT_num_ways_to_queue_ABC_l1174_117426

-- Definitions for the problem
def num_people : ℕ := 5
def fixed_order_positions : ℕ := 3

-- Lean statement to prove the problem
theorem num_ways_to_queue_ABC (h : num_people = 5) (h_fop : fixed_order_positions = 3) : 
  (Nat.factorial num_people / Nat.factorial (num_people - fixed_order_positions)) * 1 = 20 := 
by
  sorry

end NUMINAMATH_GPT_num_ways_to_queue_ABC_l1174_117426


namespace NUMINAMATH_GPT_problem1_problem2_l1174_117476

-- Problem 1
theorem problem1 : 23 + (-13) + (-17) + 8 = 1 :=
by
  sorry

-- Problem 2
theorem problem2 : - (2^3) - (1 + 0.5) / (1/3) * (-3) = 11/2 :=
by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1174_117476


namespace NUMINAMATH_GPT_fraction_comparisons_l1174_117406

theorem fraction_comparisons :
  (1 / 8 : ℝ) * (3 / 7) < (1 / 8) ∧ 
  (9 / 8 : ℝ) * (1 / 5) > (9 / 8) * (1 / 8) ∧ 
  (2 / 3 : ℝ) < (2 / 3) / (6 / 11) := by
    sorry

end NUMINAMATH_GPT_fraction_comparisons_l1174_117406


namespace NUMINAMATH_GPT_unit_prices_minimum_B_seedlings_l1174_117493

-- Definition of the problem conditions and the results of Part 1
theorem unit_prices (x : ℝ) : 
  (1200 / (1.5 * x) + 10 = 900 / x) ↔ x = 10 :=
by
  sorry

-- Definition of the problem conditions and the result of Part 2
theorem minimum_B_seedlings (m : ℕ) : 
  (10 * m + 15 * (100 - m) ≤ 1314) ↔ m ≥ 38 :=
by
  sorry

end NUMINAMATH_GPT_unit_prices_minimum_B_seedlings_l1174_117493


namespace NUMINAMATH_GPT_triangle_side_lengths_l1174_117465

theorem triangle_side_lengths
  (x y z : ℕ)
  (h1 : x > y)
  (h2 : y > z)
  (h3 : x + y + z = 240)
  (h4 : 3 * x - 2 * (y + z) = 5 * z + 10)
  (h5 : x < y + z) :
  (x = 113 ∧ y = 112 ∧ z = 15) ∨
  (x = 114 ∧ y = 110 ∧ z = 16) ∨
  (x = 115 ∧ y = 108 ∧ z = 17) ∨
  (x = 116 ∧ y = 106 ∧ z = 18) ∨
  (x = 117 ∧ y = 104 ∧ z = 19) ∨
  (x = 118 ∧ y = 102 ∧ z = 20) ∨
  (x = 119 ∧ y = 100 ∧ z = 21) := by
  sorry

end NUMINAMATH_GPT_triangle_side_lengths_l1174_117465


namespace NUMINAMATH_GPT_apple_price_difference_l1174_117498

variable (S R F : ℝ)

theorem apple_price_difference (h1 : S + R > R + F) (h2 : F = S - 250) :
  (S + R) - (R + F) = 250 :=
by
  sorry

end NUMINAMATH_GPT_apple_price_difference_l1174_117498


namespace NUMINAMATH_GPT_six_times_number_eq_132_l1174_117409

theorem six_times_number_eq_132 (x : ℕ) (h : x / 11 = 2) : 6 * x = 132 :=
sorry

end NUMINAMATH_GPT_six_times_number_eq_132_l1174_117409


namespace NUMINAMATH_GPT_subset_B_of_A_l1174_117434

def A : Set ℕ := {2, 0, 3}
def B : Set ℕ := {2, 3}

theorem subset_B_of_A : B ⊆ A :=
by
  sorry

end NUMINAMATH_GPT_subset_B_of_A_l1174_117434


namespace NUMINAMATH_GPT_at_least_one_f_nonnegative_l1174_117424

theorem at_least_one_f_nonnegative 
  (m n : ℝ) (hm : m > 0) (hn : n > 0) (h : m * n > 1) : 
  (m^2 - m ≥ 0) ∨ (n^2 - n ≥ 0) :=
by sorry

end NUMINAMATH_GPT_at_least_one_f_nonnegative_l1174_117424


namespace NUMINAMATH_GPT_correct_average_weight_l1174_117495

theorem correct_average_weight (avg_weight : ℝ) (num_boys : ℕ) (incorrect_weight correct_weight : ℝ)
  (h1 : avg_weight = 58.4) (h2 : num_boys = 20) (h3 : incorrect_weight = 56) (h4 : correct_weight = 62) :
  (avg_weight * ↑num_boys + (correct_weight - incorrect_weight)) / ↑num_boys = 58.7 := by
  sorry

end NUMINAMATH_GPT_correct_average_weight_l1174_117495


namespace NUMINAMATH_GPT_wendy_first_album_pictures_l1174_117461

theorem wendy_first_album_pictures 
  (total_pictures : ℕ)
  (num_albums : ℕ)
  (pics_per_album : ℕ)
  (pics_in_first_album : ℕ)
  (h1 : total_pictures = 79)
  (h2 : num_albums = 5)
  (h3 : pics_per_album = 7)
  (h4 : total_pictures = pics_in_first_album + num_albums * pics_per_album) : 
  pics_in_first_album = 44 :=
by
  sorry

end NUMINAMATH_GPT_wendy_first_album_pictures_l1174_117461


namespace NUMINAMATH_GPT_max_M_is_7524_l1174_117454

-- Define the conditions
def is_valid_t (t : ℕ) : Prop :=
  let a := t / 1000
  let b := (t % 1000) / 100
  let c := (t % 100) / 10
  let d := t % 10
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a + c = 9 ∧
  b - d = 1 ∧
  (2 * (2 * a + d)) % (2 * b + c) = 0

-- Define function M
def M (a b c d : ℕ) : ℕ := 2000 * a + 100 * b + 10 * c + d

-- Define the maximum value of M
def max_valid_M : ℕ :=
  let m_values := [5544, 7221, 7322, 7524]
  m_values.foldl max 0

theorem max_M_is_7524 : max_valid_M = 7524 := by
  -- The proof would be written here. For now, we indicate the theorem as
  -- not yet proven.
  sorry

end NUMINAMATH_GPT_max_M_is_7524_l1174_117454


namespace NUMINAMATH_GPT_roots_polynomial_l1174_117472

noncomputable def roots_are (a b c : ℝ) : Prop :=
  a^3 - 18 * a^2 + 20 * a - 8 = 0 ∧ b^3 - 18 * b^2 + 20 * b - 8 = 0 ∧ c^3 - 18 * c^2 + 20 * c - 8 = 0

theorem roots_polynomial (a b c : ℝ) (h : roots_are a b c) : 
  (2 + a) * (2 + b) * (2 + c) = 128 :=
by
  sorry

end NUMINAMATH_GPT_roots_polynomial_l1174_117472


namespace NUMINAMATH_GPT_bananas_left_l1174_117414

-- Definitions based on conditions
def original_bananas : ℕ := 46
def bananas_removed : ℕ := 5

-- Statement of the problem using the definitions
theorem bananas_left : original_bananas - bananas_removed = 41 :=
by sorry

end NUMINAMATH_GPT_bananas_left_l1174_117414


namespace NUMINAMATH_GPT_net_difference_in_expenditure_l1174_117453

variable (P Q : ℝ)
-- Condition 1: Price increased by 25%
def new_price (P : ℝ) : ℝ := P * 1.25

-- Condition 2: Purchased 72% of the originally required amount
def new_quantity (Q : ℝ) : ℝ := Q * 0.72

-- Definition of original expenditure
def original_expenditure (P Q : ℝ) : ℝ := P * Q

-- Definition of new expenditure
def new_expenditure (P Q : ℝ) : ℝ := new_price P * new_quantity Q

-- Statement of the proof problem.
theorem net_difference_in_expenditure
  (P Q : ℝ) : new_expenditure P Q - original_expenditure P Q = -0.1 * original_expenditure P Q := 
by
  sorry

end NUMINAMATH_GPT_net_difference_in_expenditure_l1174_117453


namespace NUMINAMATH_GPT_min_troublemakers_l1174_117491

theorem min_troublemakers (n : ℕ) (students : ℕ → Prop) 
  (h : n = 29)
  (condition1 : ∀ i, students i → (students ((i - 1) % n) ↔ ¬ students ((i + 1) % n)))
  (condition2 : ∀ i, ¬ students i → (students ((i - 1) % n) ∧ students ((i + 1) % n)))
  : ∃ L : ℕ, (L ≤ 29 ∧ L ≥ 10) :=
by sorry

end NUMINAMATH_GPT_min_troublemakers_l1174_117491


namespace NUMINAMATH_GPT_originally_planned_days_l1174_117485

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end NUMINAMATH_GPT_originally_planned_days_l1174_117485


namespace NUMINAMATH_GPT_alexander_spends_total_amount_l1174_117423

theorem alexander_spends_total_amount :
  (5 * 1) + (2 * 2) = 9 :=
by
  sorry

end NUMINAMATH_GPT_alexander_spends_total_amount_l1174_117423


namespace NUMINAMATH_GPT_statement_1_correct_statement_3_correct_correct_statements_l1174_117459

-- Definition for Acute Angles
def is_acute_angle (α : Real) : Prop :=
  0 < α ∧ α < 90

-- Definition for First Quadrant Angles
def is_first_quadrant_angle (β : Real) : Prop :=
  ∃ k : Int, k * 360 < β ∧ β < 90 + k * 360

-- Conditions
theorem statement_1_correct (α : Real) : is_acute_angle α → is_first_quadrant_angle α :=
sorry

theorem statement_3_correct (β : Real) : is_first_quadrant_angle β :=
sorry

-- Final Proof Statement
theorem correct_statements (α β : Real) :
  (is_acute_angle α → is_first_quadrant_angle α) ∧ (is_first_quadrant_angle β) :=
⟨statement_1_correct α, statement_3_correct β⟩

end NUMINAMATH_GPT_statement_1_correct_statement_3_correct_correct_statements_l1174_117459


namespace NUMINAMATH_GPT_minimum_pencils_needed_l1174_117400

theorem minimum_pencils_needed (red_pencils blue_pencils : ℕ) (total_pencils : ℕ) 
  (h_red : red_pencils = 7) (h_blue : blue_pencils = 4) (h_total : total_pencils = red_pencils + blue_pencils) :
  (∃ n : ℕ, n = 8 ∧ n ≤ total_pencils ∧ (∀ m : ℕ, m < 8 → (m < red_pencils ∨ m < blue_pencils))) :=
by
  sorry

end NUMINAMATH_GPT_minimum_pencils_needed_l1174_117400


namespace NUMINAMATH_GPT_alex_downhill_time_l1174_117446

theorem alex_downhill_time
  (speed_flat : ℝ)
  (time_flat : ℝ)
  (speed_uphill : ℝ)
  (time_uphill : ℝ)
  (speed_downhill : ℝ)
  (distance_walked : ℝ)
  (total_distance : ℝ)
  (h_flat : speed_flat = 20)
  (h_time_flat : time_flat = 4.5)
  (h_uphill : speed_uphill = 12)
  (h_time_uphill : time_uphill = 2.5)
  (h_downhill : speed_downhill = 24)
  (h_walked : distance_walked = 8)
  (h_total : total_distance = 164)
  : (156 - (speed_flat * time_flat + speed_uphill * time_uphill)) / speed_downhill = 1.5 :=
by 
  sorry

end NUMINAMATH_GPT_alex_downhill_time_l1174_117446


namespace NUMINAMATH_GPT_roots_sum_of_quadratic_l1174_117477

theorem roots_sum_of_quadratic:
  (∃ a b : ℝ, (a ≠ b) ∧ (a * b = 5) ∧ (a + b = 8)) →
  (a + b = 8) :=
by
  sorry

end NUMINAMATH_GPT_roots_sum_of_quadratic_l1174_117477


namespace NUMINAMATH_GPT_number_of_sequences_less_than_1969_l1174_117408

theorem number_of_sequences_less_than_1969 :
  (∃ S : ℕ → ℕ, (∀ n : ℕ, S (n + 1) > (S n) * (S n)) ∧ S 1969 = 1969) →
  ∃ N : ℕ, N < 1969 :=
sorry

end NUMINAMATH_GPT_number_of_sequences_less_than_1969_l1174_117408


namespace NUMINAMATH_GPT_angle_between_hour_and_minute_hand_at_3_40_l1174_117469

def angle_between_hands (hour minute : ℕ) : ℝ :=
  let minute_angle := (360 / 60) * minute
  let hour_angle := (360 / 12) + (30 / 60) * minute
  abs (minute_angle - hour_angle)

theorem angle_between_hour_and_minute_hand_at_3_40 : angle_between_hands 3 40 = 130 :=
by
  sorry

end NUMINAMATH_GPT_angle_between_hour_and_minute_hand_at_3_40_l1174_117469


namespace NUMINAMATH_GPT_mean_proportional_234_104_l1174_117458

theorem mean_proportional_234_104 : Real.sqrt (234 * 104) = 156 :=
by 
  sorry

end NUMINAMATH_GPT_mean_proportional_234_104_l1174_117458


namespace NUMINAMATH_GPT_use_six_threes_to_get_100_use_five_threes_to_get_100_l1174_117419

theorem use_six_threes_to_get_100 : 100 = (333 / 3) - (33 / 3) :=
by
  -- proof steps go here
  sorry

theorem use_five_threes_to_get_100 : 100 = (33 * 3) + (3 / 3) :=
by
  -- proof steps go here
  sorry

end NUMINAMATH_GPT_use_six_threes_to_get_100_use_five_threes_to_get_100_l1174_117419
