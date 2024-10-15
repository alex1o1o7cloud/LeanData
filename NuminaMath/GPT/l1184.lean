import Mathlib

namespace NUMINAMATH_GPT_min_houses_needed_l1184_118414

theorem min_houses_needed (n : ℕ) (x : ℕ) (h : n > 0) : (x ≤ n ∧ (x: ℚ)/n < 0.06) → n ≥ 20 :=
sorry

end NUMINAMATH_GPT_min_houses_needed_l1184_118414


namespace NUMINAMATH_GPT_number_of_rows_of_red_notes_l1184_118408

theorem number_of_rows_of_red_notes (R : ℕ) :
  let red_notes_in_each_row := 6
  let blue_notes_per_red_note := 2
  let additional_blue_notes := 10
  let total_notes := 100
  (6 * R + 12 * R + 10 = 100) → R = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_rows_of_red_notes_l1184_118408


namespace NUMINAMATH_GPT_probability_face_cards_l1184_118433

theorem probability_face_cards :
  let first_card_hearts_face := 3 / 52
  let second_card_clubs_face_after_hearts := 3 / 51
  let combined_probability := first_card_hearts_face * second_card_clubs_face_after_hearts
  combined_probability = 1 / 294 :=
by 
  sorry

end NUMINAMATH_GPT_probability_face_cards_l1184_118433


namespace NUMINAMATH_GPT_value_of_a_l1184_118484

theorem value_of_a (a : ℕ) (h : a^3 = 21 * 49 * 45 * 25) : a = 105 := sorry

end NUMINAMATH_GPT_value_of_a_l1184_118484


namespace NUMINAMATH_GPT_number_of_real_pairs_l1184_118411

theorem number_of_real_pairs :
  ∃! (x y : ℝ), 11 * x^2 + 2 * x * y + 9 * y^2 + 8 * x - 12 * y + 6 = 0 :=
sorry

end NUMINAMATH_GPT_number_of_real_pairs_l1184_118411


namespace NUMINAMATH_GPT_probability_at_least_one_six_l1184_118443

theorem probability_at_least_one_six :
  let p_six := 1 / 6
  let p_not_six := 5 / 6
  let p_not_six_three_rolls := p_not_six ^ 3
  let p_at_least_one_six := 1 - p_not_six_three_rolls
  p_at_least_one_six = 91 / 216 :=
by
  sorry

end NUMINAMATH_GPT_probability_at_least_one_six_l1184_118443


namespace NUMINAMATH_GPT_number_of_oranges_l1184_118493

-- Definitions of the conditions
def peaches : ℕ := 9
def pears : ℕ := 18
def greatest_num_per_basket : ℕ := 3
def num_baskets_peaches := peaches / greatest_num_per_basket
def num_baskets_pears := pears / greatest_num_per_basket
def min_num_baskets := min num_baskets_peaches num_baskets_pears

-- Proof problem statement
theorem number_of_oranges (O : ℕ) (h1 : O % greatest_num_per_basket = 0) 
  (h2 : O / greatest_num_per_basket = min_num_baskets) : 
  O = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_number_of_oranges_l1184_118493


namespace NUMINAMATH_GPT_angle_reduction_l1184_118487

theorem angle_reduction (θ : ℝ) : θ = 1303 → ∃ k : ℤ, θ = 360 * k - 137 := 
by  
  intro h 
  use 4 
  simp [h] 
  sorry

end NUMINAMATH_GPT_angle_reduction_l1184_118487


namespace NUMINAMATH_GPT_division_equivalence_l1184_118496

theorem division_equivalence (a b c d : ℝ) (h1 : a = 11.7) (h2 : b = 2.6) (h3 : c = 117) (h4 : d = 26) :
  (11.7 / 2.6) = (117 / 26) ∧ (117 / 26) = 4.5 := 
by 
  sorry

end NUMINAMATH_GPT_division_equivalence_l1184_118496


namespace NUMINAMATH_GPT_school_student_ratio_l1184_118475

theorem school_student_ratio :
  ∀ (F S T : ℕ), (T = 200) → (S = T + 40) → (F + S + T = 920) → (F : ℚ) / (S : ℚ) = 2 / 1 :=
by
  intros F S T hT hS hSum
  sorry

end NUMINAMATH_GPT_school_student_ratio_l1184_118475


namespace NUMINAMATH_GPT_cos_identity_l1184_118483

theorem cos_identity 
  (x : ℝ) 
  (h : Real.sin (x - π / 3) = 3 / 5) : 
  Real.cos (x + π / 6) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cos_identity_l1184_118483


namespace NUMINAMATH_GPT_total_five_digit_odd_and_multiples_of_5_l1184_118406

def count_odd_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 5
  choices

def count_multiples_of_5_five_digit_numbers : ℕ :=
  let choices := 9 * 10 * 10 * 10 * 2
  choices

theorem total_five_digit_odd_and_multiples_of_5 : count_odd_five_digit_numbers + count_multiples_of_5_five_digit_numbers = 63000 :=
by
  -- Proof Placeholder
  sorry

end NUMINAMATH_GPT_total_five_digit_odd_and_multiples_of_5_l1184_118406


namespace NUMINAMATH_GPT_max_sum_mult_table_l1184_118450

def isEven (n : ℕ) : Prop := n % 2 = 0
def isOdd (n : ℕ) : Prop := ¬ isEven n
def entries : List ℕ := [3, 4, 6, 8, 9, 12]
def sumOfList (l : List ℕ) : ℕ := l.foldr (· + ·) 0

theorem max_sum_mult_table :
  ∃ (a b c d e f : ℕ), 
    a ∈ entries ∧ b ∈ entries ∧ c ∈ entries ∧ 
    d ∈ entries ∧ e ∈ entries ∧ f ∈ entries ∧ 
    (isEven a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isEven c ∨ isOdd a ∧ isOdd b ∧ isOdd c ∨ isOdd a ∧ isEven b ∧ isOdd c ∨ isEven a ∧ isOdd b ∧ isEven c) ∧ 
    (isEven d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isEven f ∨ isOdd d ∧ isOdd e ∧ isOdd f ∨ isOdd d ∧ isEven e ∧ isOdd f ∨ isEven d ∧ isOdd e ∧ isEven f) ∧ 
    (sumOfList [a, b, c] * sumOfList [d, e, f] = 425) := 
by
    sorry  -- Skipping the proof as instructed.

end NUMINAMATH_GPT_max_sum_mult_table_l1184_118450


namespace NUMINAMATH_GPT_number_subtracted_eq_l1184_118456

theorem number_subtracted_eq (x n : ℤ) (h1 : x + 1315 + 9211 - n = 11901) (h2 : x = 88320) : n = 86945 :=
by
  sorry

end NUMINAMATH_GPT_number_subtracted_eq_l1184_118456


namespace NUMINAMATH_GPT_hiking_trip_distance_l1184_118478

open Real

-- Define the given conditions
def distance_north : ℝ := 10
def distance_south : ℝ := 7
def distance_east1 : ℝ := 17
def distance_east2 : ℝ := 8

-- Define the net displacement conditions
def net_distance_north : ℝ := distance_north - distance_south
def net_distance_east : ℝ := distance_east1 + distance_east2

-- Prove the distance from the starting point
theorem hiking_trip_distance :
  sqrt ((net_distance_north)^2 + (net_distance_east)^2) = sqrt 634 := by
  sorry

end NUMINAMATH_GPT_hiking_trip_distance_l1184_118478


namespace NUMINAMATH_GPT_equal_split_payment_l1184_118472

variable (L M N : ℝ)

theorem equal_split_payment (h1 : L < N) (h2 : L > M) : 
  (L + M + N) / 3 - L = (M + N - 2 * L) / 3 :=
by sorry

end NUMINAMATH_GPT_equal_split_payment_l1184_118472


namespace NUMINAMATH_GPT_total_opponent_scores_is_45_l1184_118491

-- Definitions based on the conditions
def games : Fin 10 := Fin.mk 10 sorry

def team_scores : Fin 10 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 2
| ⟨2, _⟩ => 3
| ⟨3, _⟩ => 4
| ⟨4, _⟩ => 5
| ⟨5, _⟩ => 6
| ⟨6, _⟩ => 7
| ⟨7, _⟩ => 8
| ⟨8, _⟩ => 9
| ⟨9, _⟩ => 10
| _ => 0  -- Placeholder for out-of-bounds, should not be used

def lost_games : Fin 5 → ℕ
| ⟨0, _⟩ => 1
| ⟨1, _⟩ => 3
| ⟨2, _⟩ => 5
| ⟨3, _⟩ => 7
| ⟨4, _⟩ => 9

def opponent_score_lost : ℕ → ℕ := λ s => s + 1

def won_games : Fin 5 → ℕ
| ⟨0, _⟩ => 2
| ⟨1, _⟩ => 4
| ⟨2, _⟩ => 6
| ⟨3, _⟩ => 8
| ⟨4, _⟩ => 10

def opponent_score_won : ℕ → ℕ := λ s => s / 2

-- Main statement to prove total opponent scores
theorem total_opponent_scores_is_45 :
  let total_lost_scores := (lost_games 0 :: lost_games 1 :: lost_games 2 :: lost_games 3 :: lost_games 4 :: []).map opponent_score_lost
  let total_won_scores  := (won_games 0 :: won_games 1 :: won_games 2 :: won_games 3 :: won_games 4 :: []).map opponent_score_won
  total_lost_scores.sum + total_won_scores.sum = 45 :=
by sorry

end NUMINAMATH_GPT_total_opponent_scores_is_45_l1184_118491


namespace NUMINAMATH_GPT_ratio_of_fifth_to_second_l1184_118452

-- Definitions based on the conditions
def first_stack := 7
def second_stack := first_stack + 3
def third_stack := second_stack - 6
def fourth_stack := third_stack + 10

def total_blocks := 55

-- The number of blocks in the fifth stack
def fifth_stack := total_blocks - (first_stack + second_stack + third_stack + fourth_stack)

-- The ratio of the fifth stack to the second stack
def ratio := fifth_stack / second_stack

-- The theorem we want to prove
theorem ratio_of_fifth_to_second: ratio = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_fifth_to_second_l1184_118452


namespace NUMINAMATH_GPT_shaded_fraction_in_fifth_diagram_l1184_118441

-- Definitions for conditions
def geometric_sequence (a₀ r n : ℕ) : ℕ := a₀ * r^n

def total_triangles (n : ℕ) : ℕ := n^2

-- Lean theorem statement
theorem shaded_fraction_in_fifth_diagram 
  (a₀ r n : ℕ) 
  (h_geometric : a₀ = 1) 
  (h_ratio : r = 2)
  (h_step_number : n = 4):
  (geometric_sequence a₀ r n) / (total_triangles (n + 1)) = 16 / 25 :=
by
  sorry

end NUMINAMATH_GPT_shaded_fraction_in_fifth_diagram_l1184_118441


namespace NUMINAMATH_GPT_necessary_and_sufficient_l1184_118469

def point_on_curve (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : Prop :=
  f P = 0

theorem necessary_and_sufficient (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) :
  (point_on_curve P f ↔ f P = 0) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_l1184_118469


namespace NUMINAMATH_GPT_first_problem_second_problem_l1184_118415

variable (x : ℝ)

-- Proof for the first problem
theorem first_problem : 6 * x^3 / (-3 * x^2) = -2 * x := by
sorry

-- Proof for the second problem
theorem second_problem : (2 * x + 3) * (2 * x - 3) - 4 * (x - 2)^2 = 16 * x - 25 := by
sorry

end NUMINAMATH_GPT_first_problem_second_problem_l1184_118415


namespace NUMINAMATH_GPT_Lizzy_savings_after_loan_l1184_118453

theorem Lizzy_savings_after_loan :
  ∀ (initial_amount loan_amount : ℕ) (interest_percent : ℕ),
  initial_amount = 30 →
  loan_amount = 15 →
  interest_percent = 20 →
  initial_amount - loan_amount + loan_amount + loan_amount * interest_percent / 100 = 33 :=
by
  intros initial_amount loan_amount interest_percent h1 h2 h3
  sorry

end NUMINAMATH_GPT_Lizzy_savings_after_loan_l1184_118453


namespace NUMINAMATH_GPT_max_value_of_f_l1184_118473

noncomputable def f (ω a x : ℝ) : ℝ := Real.sin (ω * x) + a * Real.cos (ω * x)

theorem max_value_of_f 
  (ω a : ℝ) 
  (h1 : 0 < ω) 
  (h2 : (2 * Real.pi / ω) = Real.pi) 
  (h3 : ∃ k : ℤ, ω * (Real.pi / 12) + (k : ℝ) * Real.pi + Real.pi / 3 = Real.pi / 2 + (k : ℝ) * Real.pi) :
  ∃ x : ℝ, f ω a x = 2 := by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l1184_118473


namespace NUMINAMATH_GPT_find_value_of_a_l1184_118412

theorem find_value_of_a (a : ℝ) (h : ( (-2 - (2 * a - 1)) / (3 - (-2)) = -1 )) : a = 2 :=
sorry

end NUMINAMATH_GPT_find_value_of_a_l1184_118412


namespace NUMINAMATH_GPT_employed_males_percentage_l1184_118438

theorem employed_males_percentage (P : ℕ) (H1: P > 0)
    (employed_pct : ℝ) (female_pct : ℝ)
    (H_employed_pct : employed_pct = 0.64)
    (H_female_pct : female_pct = 0.140625) :
    (0.859375 * employed_pct * 100) = 54.96 :=
by
  sorry

end NUMINAMATH_GPT_employed_males_percentage_l1184_118438


namespace NUMINAMATH_GPT_proof_min_max_expected_wasted_minutes_l1184_118465

/-- The conditions given:
    - There are 8 people in the queue.
    - 5 people perform simple operations that take 1 minute each.
    - 3 people perform lengthy operations that take 5 minutes each.
--/
structure QueueStatus where
  total_people : Nat := 8
  simple_operations_people : Nat := 5
  lengthy_operations_people : Nat := 3
  simple_operation_time : Nat := 1
  lengthy_operation_time : Nat := 5

/-- Propositions to be proven:
    - Minimum possible total number of wasted person-minutes is 40.
    - Maximum possible total number of wasted person-minutes is 100.
    - Expected total number of wasted person-minutes in random order is 72.5.
--/
def min_wasted_person_minutes (qs: QueueStatus) : Nat := 40
def max_wasted_person_minutes (qs: QueueStatus) : Nat := 100
def expected_wasted_person_minutes (qs: QueueStatus) : Real := 72.5

theorem proof_min_max_expected_wasted_minutes (qs: QueueStatus) :
  min_wasted_person_minutes qs = 40 ∧ 
  max_wasted_person_minutes qs = 100 ∧ 
  expected_wasted_person_minutes qs = 72.5 := by
  sorry

end NUMINAMATH_GPT_proof_min_max_expected_wasted_minutes_l1184_118465


namespace NUMINAMATH_GPT_cos_B_in_triangle_l1184_118418

theorem cos_B_in_triangle (A B C : ℝ) (a b c : ℝ) 
  (h1 : a = (Real.sqrt 5 / 2) * b)
  (h2 : A = 2 * B)
  (h_triangle: A + B + C = Real.pi) : 
  Real.cos B = Real.sqrt 5 / 4 :=
sorry

end NUMINAMATH_GPT_cos_B_in_triangle_l1184_118418


namespace NUMINAMATH_GPT_smallest_n_condition_l1184_118410

theorem smallest_n_condition (n : ℕ) : 25 * n - 3 ≡ 0 [MOD 16] → n ≡ 11 [MOD 16] :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_condition_l1184_118410


namespace NUMINAMATH_GPT_alex_initial_jelly_beans_l1184_118468

variable (initial : ℕ)
variable (eaten : ℕ := 6)
variable (pile_weight : ℕ := 10)
variable (piles : ℕ := 3)

theorem alex_initial_jelly_beans :
  (initial - eaten = pile_weight * piles) → initial = 36 :=
by
  -- proof will be provided here
  sorry

end NUMINAMATH_GPT_alex_initial_jelly_beans_l1184_118468


namespace NUMINAMATH_GPT_overall_gain_percentage_l1184_118430

def cost_of_A : ℝ := 100
def selling_price_of_A : ℝ := 125
def cost_of_B : ℝ := 200
def selling_price_of_B : ℝ := 250
def cost_of_C : ℝ := 150
def selling_price_of_C : ℝ := 180

theorem overall_gain_percentage :
  ((selling_price_of_A + selling_price_of_B + selling_price_of_C) - (cost_of_A + cost_of_B + cost_of_C)) / (cost_of_A + cost_of_B + cost_of_C) * 100 = 23.33 := 
by
  sorry

end NUMINAMATH_GPT_overall_gain_percentage_l1184_118430


namespace NUMINAMATH_GPT_simplify_expression_l1184_118485

theorem simplify_expression (a : ℝ) (h1 : a ≠ 0) (h2 : a ≠ 2) :
  (a^2 - 6 * a + 9) / (a^2 - 2 * a) / (1 - 1 / (a - 2)) = (a - 3) / a :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1184_118485


namespace NUMINAMATH_GPT_find_k_value_l1184_118482

theorem find_k_value (k : ℝ) : (∀ (x y : ℝ), (x = 2 ∧ y = 5) → y = k * x + 3) → k = 1 := 
by 
  intro h
  have h1 := h 2 5 ⟨rfl, rfl⟩
  linarith

end NUMINAMATH_GPT_find_k_value_l1184_118482


namespace NUMINAMATH_GPT_triangle_area_proof_l1184_118466

-- Define the triangle sides and median
variables (AB BC BD AC : ℝ)

-- Assume given values
def AB_value : AB = 1 := by sorry 
def BC_value : BC = Real.sqrt 15 := by sorry
def BD_value : BD = 2 := by sorry

-- Assume AC calculated from problem
def AC_value : AC = 4 := by sorry

-- Final proof statement
theorem triangle_area_proof 
  (hAB : AB = 1)
  (hBC : BC = Real.sqrt 15)
  (hBD : BD = 2)
  (hAC : AC = 4) :
  (1 / 2) * AB * BC = (Real.sqrt 15) / 2 := 
sorry

end NUMINAMATH_GPT_triangle_area_proof_l1184_118466


namespace NUMINAMATH_GPT_lottery_ticket_might_win_l1184_118459

theorem lottery_ticket_might_win (p_win : ℝ) (h : p_win = 0.01) : 
  (∃ (n : ℕ), n = 1 ∧ 0 < p_win ∧ p_win < 1) :=
by 
  sorry

end NUMINAMATH_GPT_lottery_ticket_might_win_l1184_118459


namespace NUMINAMATH_GPT_infinite_solutions_exists_l1184_118477

theorem infinite_solutions_exists : 
  ∃ (S : Set (ℕ × ℕ)), (∀ (a b : ℕ), (a, b) ∈ S → 2 * a^2 - 3 * a + 1 = 3 * b^2 + b) 
  ∧ Set.Infinite S :=
sorry

end NUMINAMATH_GPT_infinite_solutions_exists_l1184_118477


namespace NUMINAMATH_GPT_p_cycling_speed_l1184_118498

-- J starts walking at 6 kmph at 12:00
def start_time : ℕ := 12 * 60  -- time in minutes for convenience
def j_speed : ℤ := 6  -- in kmph
def j_start_time : ℕ := start_time  -- 12:00 in minutes

-- P starts cycling at 13:30
def p_start_time : ℕ := (13 * 60) + 30  -- time in minutes for convenience

-- They are at their respective positions at 19:30
def end_time : ℕ := (19 * 60) + 30  -- time in minutes for convenience

-- At 19:30, J is 3 km behind P
def j_behind_p_distance : ℤ := 3  -- in kilometers

-- Prove that P's cycling speed = 8 kmph
theorem p_cycling_speed {p_speed : ℤ} :
  j_start_time = start_time →
  p_start_time = (13 * 60) + 30 →
  end_time = (19 * 60) + 30 →
  j_speed = 6 →
  j_behind_p_distance = 3 →
  p_speed = 8 :=
by
  sorry

end NUMINAMATH_GPT_p_cycling_speed_l1184_118498


namespace NUMINAMATH_GPT_evaluate_series_l1184_118463

-- Define the series S
noncomputable def S : ℝ := ∑' n : ℕ, (n + 1) / (3 ^ (n + 1))

-- Lean statement to show the evaluated series
theorem evaluate_series : (3:ℝ)^S = (3:ℝ)^(3 / 4) :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_evaluate_series_l1184_118463


namespace NUMINAMATH_GPT_village_duration_l1184_118420

theorem village_duration (vampire_drain : ℕ) (werewolf_eat : ℕ) (village_population : ℕ)
  (hv : vampire_drain = 3) (hw : werewolf_eat = 5) (hp : village_population = 72) :
  village_population / (vampire_drain + werewolf_eat) = 9 :=
by
  sorry

end NUMINAMATH_GPT_village_duration_l1184_118420


namespace NUMINAMATH_GPT_taxi_fare_max_distance_l1184_118424

-- Setting up the conditions
def starting_price : ℝ := 7
def additional_fare_per_km : ℝ := 2.4
def max_base_distance_km : ℝ := 3
def total_fare : ℝ := 19

-- Defining the maximum distance based on the given conditions
def max_distance : ℝ := 8

-- The theorem is to prove that the maximum distance is indeed 8 kilometers
theorem taxi_fare_max_distance :
  ∀ (x : ℝ), total_fare = starting_price + additional_fare_per_km * (x - max_base_distance_km) → x ≤ max_distance :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_taxi_fare_max_distance_l1184_118424


namespace NUMINAMATH_GPT_spadesuit_example_l1184_118476

-- Define the operation spadesuit
def spadesuit (a b : ℤ) : ℤ := abs (a - b)

-- Define the specific instance to prove
theorem spadesuit_example : spadesuit 2 (spadesuit 4 7) = 1 :=
by
  sorry

end NUMINAMATH_GPT_spadesuit_example_l1184_118476


namespace NUMINAMATH_GPT_problem_statement_l1184_118444

theorem problem_statement :
  ¬(∀ n : ℤ, n ≥ 0 → n = 0) ∧
  ¬(∀ q : ℚ, q ≠ 0 → q > 0 ∨ q < 0) ∧
  ¬(∀ a b : ℝ, abs a = abs b → a = b) ∧
  (∀ a : ℝ, abs a = abs (-a)) :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1184_118444


namespace NUMINAMATH_GPT_john_ratio_amounts_l1184_118439

/-- John gets $30 from his grandpa and some multiple of that amount from his grandma. 
He got $120 from the two grandparents. What is the ratio of the amount he got from 
his grandma to the amount he got from his grandpa? --/
theorem john_ratio_amounts (amount_grandpa amount_total : ℝ) (multiple : ℝ) :
  amount_grandpa = 30 → amount_total = 120 →
  amount_total = amount_grandpa + multiple * amount_grandpa →
  multiple = 3 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_john_ratio_amounts_l1184_118439


namespace NUMINAMATH_GPT_tan_inequality_solution_l1184_118422

variable (x : ℝ)
variable (k : ℤ)

theorem tan_inequality_solution (hx : Real.tan (2 * x - Real.pi / 4) ≤ 1) :
  ∃ k : ℤ,
  (k * Real.pi / 2 - Real.pi / 8 < x) ∧ (x ≤ k * Real.pi / 2 + Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_tan_inequality_solution_l1184_118422


namespace NUMINAMATH_GPT_overlapping_triangle_area_l1184_118407

/-- Given a rectangle with length 8 and width 4, folded along its diagonal, 
    the area of the overlapping part (grey triangle) is 10. --/
theorem overlapping_triangle_area : 
  let length := 8 
  let width := 4 
  let diagonal := (length^2 + width^2)^(1/2) 
  let base := (length^2 / (width^2 + length^2))^(1/2) * width 
  let height := width
  1 / 2 * base * height = 10 := by 
  sorry

end NUMINAMATH_GPT_overlapping_triangle_area_l1184_118407


namespace NUMINAMATH_GPT_maximize_revenue_l1184_118446

-- Define the conditions
def total_time_condition (x y : ℝ) : Prop := x + y ≤ 300
def total_cost_condition (x y : ℝ) : Prop := 2.5 * x + y ≤ 4500
def non_negative_condition (x y : ℝ) : Prop := x ≥ 0 ∧ y ≥ 0

-- Define the revenue function
def revenue (x y : ℝ) : ℝ := 0.3 * x + 0.2 * y

-- The proof statement
theorem maximize_revenue : 
  ∃ x y, total_time_condition x y ∧ total_cost_condition x y ∧ non_negative_condition x y ∧ 
  revenue x y = 70 := 
by
  sorry

end NUMINAMATH_GPT_maximize_revenue_l1184_118446


namespace NUMINAMATH_GPT_age_of_B_l1184_118404

variables (A B : ℕ)

-- Conditions
def condition1 := A + 10 = 2 * (B - 10)
def condition2 := A = B + 7

-- Theorem stating the present age of B
theorem age_of_B (h1 : condition1 A B) (h2 : condition2 A B) : B = 37 :=
by
  sorry

end NUMINAMATH_GPT_age_of_B_l1184_118404


namespace NUMINAMATH_GPT_percent_increase_l1184_118437

theorem percent_increase (original value new_value : ℕ) (h1 : original_value = 20) (h2 : new_value = 25) :
  ((new_value - original_value) / original_value) * 100 = 25 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_percent_increase_l1184_118437


namespace NUMINAMATH_GPT_walking_rate_ratio_l1184_118489

theorem walking_rate_ratio (R R' : ℚ) (D : ℚ) (h1: D = R * 14) (h2: D = R' * 12) : R' / R = 7 / 6 :=
by 
  sorry

end NUMINAMATH_GPT_walking_rate_ratio_l1184_118489


namespace NUMINAMATH_GPT_sequence_an_formula_l1184_118455

theorem sequence_an_formula (a : ℕ → ℕ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, n ≥ 1 → a (n + 1) = 2 * a n + 1) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 :=
by
  sorry

end NUMINAMATH_GPT_sequence_an_formula_l1184_118455


namespace NUMINAMATH_GPT_percentage_problem_l1184_118490

theorem percentage_problem
  (a b c : ℚ) :
  (8 = (2 / 100) * a) →
  (2 = (8 / 100) * b) →
  (c = b / a) →
  c = 1 / 16 :=
by
  sorry

end NUMINAMATH_GPT_percentage_problem_l1184_118490


namespace NUMINAMATH_GPT_percentage_goods_lost_l1184_118454

theorem percentage_goods_lost
    (cost_price selling_price loss_price : ℝ)
    (profit_percent loss_percent : ℝ)
    (h_profit : selling_price = cost_price * (1 + profit_percent / 100))
    (h_loss_value : loss_price = selling_price * (loss_percent / 100))
    (cost_price_assumption : cost_price = 100)
    (profit_percent_assumption : profit_percent = 10)
    (loss_percent_assumption : loss_percent = 45) :
    (loss_price / cost_price * 100) = 49.5 :=
sorry

end NUMINAMATH_GPT_percentage_goods_lost_l1184_118454


namespace NUMINAMATH_GPT_program_arrangements_l1184_118486

/-- Given 5 programs, if A, B, and C appear in a specific order, then the number of different
    arrangements is 20. -/
theorem program_arrangements (A B C A_order : ℕ) : 
  (A + B + C + A_order = 5) → 
  (A_order = 3) → 
  (B = 1) → 
  (C = 1) → 
  (A = 1) → 
  (A * B * C * A_order = 1) :=
  by sorry

end NUMINAMATH_GPT_program_arrangements_l1184_118486


namespace NUMINAMATH_GPT_numerator_greater_denominator_l1184_118434

theorem numerator_greater_denominator (x : ℝ) (h1 : -3 ≤ x) (h2 : x ≤ 3) (h3 : 5 * x + 3 > 8 - 3 * x) : (5 / 8) < x ∧ x ≤ 3 :=
by
  sorry

end NUMINAMATH_GPT_numerator_greater_denominator_l1184_118434


namespace NUMINAMATH_GPT_probability_of_drawing_green_ball_l1184_118405

variable (total_balls green_balls : ℕ)
variable (total_balls_eq : total_balls = 10)
variable (green_balls_eq : green_balls = 4)

theorem probability_of_drawing_green_ball (h_total : total_balls = 10) (h_green : green_balls = 4) :
  (green_balls : ℚ) / total_balls = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_probability_of_drawing_green_ball_l1184_118405


namespace NUMINAMATH_GPT_seating_arrangement_l1184_118445

theorem seating_arrangement (n x : ℕ) (h1 : 7 * x + 6 * (n - x) = 53) : x = 5 :=
sorry

end NUMINAMATH_GPT_seating_arrangement_l1184_118445


namespace NUMINAMATH_GPT_total_matches_round_robin_l1184_118419

/-- A round-robin chess tournament is organized in two groups with different numbers of players. 
Group A consists of 6 players, and Group B consists of 5 players. 
Each player in each group plays every other player in the same group exactly once. 
Prove that the total number of matches is 25. -/
theorem total_matches_round_robin 
  (nA : ℕ) (nB : ℕ) 
  (hA : nA = 6) (hB : nB = 5) : 
  (nA * (nA - 1) / 2) + (nB * (nB - 1) / 2) = 25 := 
  by
    sorry

end NUMINAMATH_GPT_total_matches_round_robin_l1184_118419


namespace NUMINAMATH_GPT_positive_integer_M_l1184_118409

theorem positive_integer_M (M : ℕ) (h : 14^2 * 35^2 = 70^2 * M^2) : M = 7 :=
sorry

end NUMINAMATH_GPT_positive_integer_M_l1184_118409


namespace NUMINAMATH_GPT_compute_fraction_l1184_118464

theorem compute_fraction :
  (1 * 2 + 2 * 4 - 3 * 8 + 4 * 16 + 5 * 32 - 6 * 64) /
  (2 * 4 + 4 * 8 - 6 * 16 + 8 * 32 + 10 * 64 - 12 * 128) =
  1 / 4 :=
by
  -- Proof will go here
  sorry

end NUMINAMATH_GPT_compute_fraction_l1184_118464


namespace NUMINAMATH_GPT_a_8_eq_5_l1184_118442

noncomputable def S (n : ℕ) : ℕ := sorry
noncomputable def a (n : ℕ) : ℕ := sorry

axiom S_eq : ∀ n m : ℕ, S n + S m = S (n + m)
axiom a1 : a 1 = 5
axiom Sn1 : ∀ n : ℕ, S (n + 1) = S n + 5

theorem a_8_eq_5 : a 8 = 5 :=
sorry

end NUMINAMATH_GPT_a_8_eq_5_l1184_118442


namespace NUMINAMATH_GPT_arrange_decimals_in_order_l1184_118492

theorem arrange_decimals_in_order 
  (a b c d : ℚ) 
  (h₀ : a = 6 / 10) 
  (h₁ : b = 676 / 1000) 
  (h₂ : c = 677 / 1000) 
  (h₃ : d = 67 / 100) : 
  a < d ∧ d < b ∧ b < c := 
by
  sorry

end NUMINAMATH_GPT_arrange_decimals_in_order_l1184_118492


namespace NUMINAMATH_GPT_midpoint_locus_l1184_118428

theorem midpoint_locus (c : ℝ) (H : 0 < c ∧ c ≤ Real.sqrt 2) :
  ∃ L, L = "curvilinear quadrilateral with arcs forming transitions" :=
sorry

end NUMINAMATH_GPT_midpoint_locus_l1184_118428


namespace NUMINAMATH_GPT_initial_games_count_l1184_118479

-- Definitions used in conditions
def games_given_away : ℕ := 99
def games_left : ℝ := 22.0

-- Theorem statement for the initial number of games
theorem initial_games_count : games_given_away + games_left = 121.0 := by
  sorry

end NUMINAMATH_GPT_initial_games_count_l1184_118479


namespace NUMINAMATH_GPT_geometric_sequence_first_term_and_ratio_l1184_118431

theorem geometric_sequence_first_term_and_ratio (b : ℕ → ℚ) 
  (hb2 : b 2 = 37 + 1/3) 
  (hb6 : b 6 = 2 + 1/3) : 
  ∃ (b1 q : ℚ), b 1 = b1 ∧ (∀ n, b n = b1 * q^(n-1)) ∧ b1 = 224 / 3 ∧ q = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_geometric_sequence_first_term_and_ratio_l1184_118431


namespace NUMINAMATH_GPT_taylor_correct_answers_percentage_l1184_118480

theorem taylor_correct_answers_percentage 
  (N : ℕ := 30)
  (alex_correct_alone_percentage : ℝ := 0.85)
  (alex_overall_percentage : ℝ := 0.83)
  (taylor_correct_alone_percentage : ℝ := 0.95)
  (alex_correct_alone : ℕ := 13)
  (alex_correct_total : ℕ := 25)
  (together_correct : ℕ := 12)
  (taylor_correct_alone : ℕ := 14)
  (taylor_correct_total : ℕ := 26) :
  ((taylor_correct_total : ℝ) / (N : ℝ)) * 100 = 87 :=
by
  sorry

end NUMINAMATH_GPT_taylor_correct_answers_percentage_l1184_118480


namespace NUMINAMATH_GPT_f_max_iff_l1184_118426

noncomputable def f : ℚ → ℝ := sorry

axiom f_zero : f 0 = 0
axiom f_pos (a : ℚ) (h : a ≠ 0) : f a > 0
axiom f_mul (a b : ℚ) : f (a * b) = f a * f b
axiom f_add_le (a b : ℚ) : f (a + b) ≤ f a + f b
axiom f_bound (m : ℤ) : f m ≤ 1989

theorem f_max_iff (a b : ℚ) (h : f a ≠ f b) : f (a + b) = max (f a) (f b) := 
sorry

end NUMINAMATH_GPT_f_max_iff_l1184_118426


namespace NUMINAMATH_GPT_gcd_840_1764_l1184_118447

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 := by
  sorry

end NUMINAMATH_GPT_gcd_840_1764_l1184_118447


namespace NUMINAMATH_GPT_percentage_books_returned_l1184_118481

theorem percentage_books_returned
    (initial_books : ℝ)
    (end_books : ℝ)
    (loaned_books : ℝ)
    (R : ℝ)
    (Percentage_Returned : ℝ) :
    initial_books = 75 →
    end_books = 65 →
    loaned_books = 50.000000000000014 →
    R = (75 - 65) →
    Percentage_Returned = (R / loaned_books) * 100 →
    Percentage_Returned = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_books_returned_l1184_118481


namespace NUMINAMATH_GPT_probability_region_D_l1184_118429

noncomputable def P_A : ℝ := 1 / 4
noncomputable def P_B : ℝ := 1 / 3
noncomputable def P_C : ℝ := 1 / 6

theorem probability_region_D (P_D : ℝ) (h : P_A + P_B + P_C + P_D = 1) : P_D = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_region_D_l1184_118429


namespace NUMINAMATH_GPT_sqrt_square_eq_self_l1184_118458

variable (a : ℝ)

theorem sqrt_square_eq_self (h : a > 0) : Real.sqrt (a ^ 2) = a :=
  sorry

end NUMINAMATH_GPT_sqrt_square_eq_self_l1184_118458


namespace NUMINAMATH_GPT_patty_fraction_3mph_l1184_118440

noncomputable def fraction_time_at_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) : ℝ :=
  t3 / (t3 + t6)

theorem patty_fraction_3mph (t3 t6 : ℝ) (h : 3 * t3 + 6 * t6 = 5 * (t3 + t6)) :
  fraction_time_at_3mph t3 t6 h = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_patty_fraction_3mph_l1184_118440


namespace NUMINAMATH_GPT_neg_P_l1184_118401

-- Define the proposition P
def P : Prop := ∃ x : ℝ, Real.exp x ≤ 0

-- State the negation of P
theorem neg_P : ¬P ↔ ∀ x : ℝ, Real.exp x > 0 := 
by 
  sorry

end NUMINAMATH_GPT_neg_P_l1184_118401


namespace NUMINAMATH_GPT_find_geometric_sequence_first_term_and_ratio_l1184_118403

theorem find_geometric_sequence_first_term_and_ratio 
  (a1 a2 a3 a4 a5 : ℕ) 
  (h : a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5)
  (geo_seq : a2 = a1 * 3 / 2 ∧ a3 = a2 * 3 / 2 ∧ a4 = a3 * 3 / 2 ∧ a5 = a4 * 3 / 2)
  (sum_cond : a1 + a2 + a3 + a4 + a5 = 211) :
  (a1 = 16) ∧ (3 / 2 = 3 / 2) := 
by {
  sorry
}

end NUMINAMATH_GPT_find_geometric_sequence_first_term_and_ratio_l1184_118403


namespace NUMINAMATH_GPT_percent_of_a_is_b_l1184_118494

variable {a b c : ℝ}

theorem percent_of_a_is_b (h1 : c = 0.25 * a) (h2 : c = 0.10 * b) : b = 2.5 * a :=
by sorry

end NUMINAMATH_GPT_percent_of_a_is_b_l1184_118494


namespace NUMINAMATH_GPT_arithmetic_sequence_unique_a_l1184_118416

theorem arithmetic_sequence_unique_a (a : ℝ) (b : ℕ → ℝ) (a_seq : ℕ → ℝ)
  (h1 : a_seq 1 = a) (h2 : a > 0)
  (h3 : b 1 - a_seq 1 = 1) (h4 : b 2 - a_seq 2 = 2)
  (h5 : b 3 - a_seq 3 = 3)
  (unique_a : ∀ (a' : ℝ), (a_seq 1 = a' ∧ a' > 0 ∧ b 1 - a' = 1 ∧ b 2 - a_seq 2 = 2 ∧ b 3 - a_seq 3 = 3) → a' = a) :
  a = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_unique_a_l1184_118416


namespace NUMINAMATH_GPT_prime_prod_identity_l1184_118457

theorem prime_prod_identity (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : 3 * p + 7 * q = 41) : (p + 1) * (q - 1) = 12 := 
by 
  sorry

end NUMINAMATH_GPT_prime_prod_identity_l1184_118457


namespace NUMINAMATH_GPT_greatest_sum_x_y_l1184_118432

theorem greatest_sum_x_y (x y : ℤ) (h : x^2 + y^2 = 36) : (x + y ≤ 9) := sorry

end NUMINAMATH_GPT_greatest_sum_x_y_l1184_118432


namespace NUMINAMATH_GPT_solve_equation_l1184_118427

theorem solve_equation 
  (x : ℝ) 
  (h : (2 * x - 1)^2 - (1 - 3 * x)^2 = 5 * (1 - x) * (x + 1)) : 
  x = 5 / 2 := 
sorry

end NUMINAMATH_GPT_solve_equation_l1184_118427


namespace NUMINAMATH_GPT_ellipse_standard_equation_l1184_118471

theorem ellipse_standard_equation (a b c : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 3 / 4) (h3 : b^2 = a^2 - c^2) :
  (x y : ℝ) →
  (x^2 / a^2 + y^2 / b^2 = 1 ∨ x^2 / b^2 + y^2 / a^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_standard_equation_l1184_118471


namespace NUMINAMATH_GPT_greatest_prime_factor_is_5_l1184_118436

-- Define the expression
def expr : Nat := (3^8 + 9^5)

-- State the theorem
theorem greatest_prime_factor_is_5 : ∃ p : Nat, Prime p ∧ p = 5 ∧ ∀ q : Nat, Prime q ∧ q ∣ expr → q ≤ 5 := by
  sorry

end NUMINAMATH_GPT_greatest_prime_factor_is_5_l1184_118436


namespace NUMINAMATH_GPT_lcm_of_pack_sizes_l1184_118448

theorem lcm_of_pack_sizes :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 19) 8) 11) 17) 23 = 772616 := by
  sorry

end NUMINAMATH_GPT_lcm_of_pack_sizes_l1184_118448


namespace NUMINAMATH_GPT_tan_585_eq_1_l1184_118461

theorem tan_585_eq_1 : Real.tan (585 * Real.pi / 180) = 1 := 
by
  sorry

end NUMINAMATH_GPT_tan_585_eq_1_l1184_118461


namespace NUMINAMATH_GPT_EF_squared_correct_l1184_118474

-- Define the problem setup and the proof goal.
theorem EF_squared_correct :
  ∀ (A B C D E F : Type)
  (side : ℝ)
  (h1 : side = 10)
  (BE DF AE CF : ℝ)
  (h2 : BE = 7)
  (h3 : DF = 7)
  (h4 : AE = 15)
  (h5 : CF = 15)
  (EF_squared : ℝ),
  EF_squared = 548 :=
by
  sorry

end NUMINAMATH_GPT_EF_squared_correct_l1184_118474


namespace NUMINAMATH_GPT_fourth_circle_radius_l1184_118488

theorem fourth_circle_radius (c : ℝ) (h : c > 0) :
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  fourth_radius = (c / 2) - r :=
by
  let r := c / 5
  let fourth_radius := (3 * c) / 10
  sorry

end NUMINAMATH_GPT_fourth_circle_radius_l1184_118488


namespace NUMINAMATH_GPT_winnie_keeps_10_lollipops_l1184_118460

def winnie_keep_lollipops : Prop :=
  let cherry := 72
  let wintergreen := 89
  let grape := 23
  let shrimp_cocktail := 316
  let total_lollipops := cherry + wintergreen + grape + shrimp_cocktail
  let friends := 14
  let lollipops_per_friend := total_lollipops / friends
  let winnie_keeps := total_lollipops % friends
  winnie_keeps = 10

theorem winnie_keeps_10_lollipops : winnie_keep_lollipops := by
  sorry

end NUMINAMATH_GPT_winnie_keeps_10_lollipops_l1184_118460


namespace NUMINAMATH_GPT_small_boxes_in_big_box_l1184_118467

theorem small_boxes_in_big_box (total_candles : ℕ) (candles_per_small : ℕ) (total_big_boxes : ℕ) 
  (h1 : total_candles = 8000) 
  (h2 : candles_per_small = 40) 
  (h3 : total_big_boxes = 50) :
  (total_candles / candles_per_small) / total_big_boxes = 4 :=
by
  sorry

end NUMINAMATH_GPT_small_boxes_in_big_box_l1184_118467


namespace NUMINAMATH_GPT_cube_edge_length_l1184_118451

theorem cube_edge_length (V : ℝ) (a : ℝ)
  (hV : V = (4 / 3) * Real.pi * (Real.sqrt 3 * a / 2) ^ 3)
  (hVolume : V = (9 * Real.pi) / 2) :
  a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_cube_edge_length_l1184_118451


namespace NUMINAMATH_GPT_first_year_exceeds_threshold_l1184_118462

def P (n : ℕ) : ℝ := 40000 * (1 + 0.2) ^ n
def exceeds_threshold (n : ℕ) : Prop := P n > 120000

theorem first_year_exceeds_threshold : ∃ n : ℕ, exceeds_threshold n ∧ 2013 + n = 2020 := 
by
  sorry

end NUMINAMATH_GPT_first_year_exceeds_threshold_l1184_118462


namespace NUMINAMATH_GPT_coordinates_of_point_A_l1184_118495

    theorem coordinates_of_point_A (x y : ℝ) (h1 : y = 0) (h2 : abs x = 3) : (x, y) = (3, 0) ∨ (x, y) = (-3, 0) :=
    sorry
    
end NUMINAMATH_GPT_coordinates_of_point_A_l1184_118495


namespace NUMINAMATH_GPT_line_points_product_l1184_118425

theorem line_points_product (x y : ℝ) (h1 : 8 = (1/4 : ℝ) * x) (h2 : y = (1/4 : ℝ) * 20) : x * y = 160 := 
by
  sorry

end NUMINAMATH_GPT_line_points_product_l1184_118425


namespace NUMINAMATH_GPT_probability_of_C_l1184_118400

theorem probability_of_C (P : ℕ → ℚ) (P_total : P 1 + P 2 + P 3 = 1)
  (P_A : P 1 = 1/3) (P_B : P 2 = 1/2) : P 3 = 1/6 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_C_l1184_118400


namespace NUMINAMATH_GPT_sin_sum_alpha_pi_over_3_l1184_118470

theorem sin_sum_alpha_pi_over_3 (alpha : ℝ) (h1 : Real.cos (alpha + 2/3 * Real.pi) = 4/5) (h2 : -Real.pi/2 < alpha ∧ alpha < 0) :
  Real.sin (alpha + Real.pi/3) + Real.sin alpha = -4 * Real.sqrt 3 / 5 :=
sorry

end NUMINAMATH_GPT_sin_sum_alpha_pi_over_3_l1184_118470


namespace NUMINAMATH_GPT_part1_range_a_part2_range_a_l1184_118402

-- Definitions of the propositions
def p (a : ℝ) := ∃ x : ℝ, x^2 + a * x + 2 = 0

def q (a : ℝ) := ∀ x : ℝ, 0 < x ∧ x < 1 → x^2 - a < 0

-- Part 1: If p is true, find the range of values for a
theorem part1_range_a (a : ℝ) :
  p a → (a ≤ -2*Real.sqrt 2 ∨ a ≥ 2*Real.sqrt 2) := sorry

-- Part 2: If one of p or q is true and the other is false, find the range of values for a
theorem part2_range_a (a : ℝ) :
  (p a ∧ ¬ q a) ∨ (¬ p a ∧ q a) →
  (a ≤ -2*Real.sqrt 2 ∨ (1 ≤ a ∧ a < 2*Real.sqrt 2)) := sorry

end NUMINAMATH_GPT_part1_range_a_part2_range_a_l1184_118402


namespace NUMINAMATH_GPT_circular_garden_area_l1184_118449

theorem circular_garden_area
  (r : ℝ) (h_r : r = 16)
  (C A : ℝ) (h_C : C = 2 * Real.pi * r) (h_A : A = Real.pi * r^2)
  (fence_cond : C = 1 / 8 * A) :
  A = 256 * Real.pi := by
  sorry

end NUMINAMATH_GPT_circular_garden_area_l1184_118449


namespace NUMINAMATH_GPT_moles_of_HCl_formed_l1184_118413

-- Define the reaction as given in conditions
def reaction (C2H6 Cl2 C2H4Cl2 HCl : ℝ) := C2H6 + Cl2 = C2H4Cl2 + 2 * HCl

-- Define the initial moles of reactants
def moles_C2H6 : ℝ := 2
def moles_Cl2 : ℝ := 2

-- State the expected moles of HCl produced
def expected_moles_HCl : ℝ := 4

-- The theorem stating the problem to prove
theorem moles_of_HCl_formed : ∃ HCl : ℝ, reaction moles_C2H6 moles_Cl2 0 HCl ∧ HCl = expected_moles_HCl :=
by
  -- Skipping detailed proof with sorry
  sorry

end NUMINAMATH_GPT_moles_of_HCl_formed_l1184_118413


namespace NUMINAMATH_GPT_min_people_in_photographs_l1184_118499

-- Definitions based on conditions
def photographs := (List (Nat × Nat × Nat))
def menInCenter (photos : photographs) := photos.map (fun (c, _, _) => c)

-- Condition: there are 10 photographs each with a distinct man in the center
def valid_photographs (photos: photographs) :=
  photos.length = 10 ∧ photos.map (fun (c, _, _) => c) = List.range 10

-- Theorem to be proved: The minimum number of different people in the photographs is at least 16
theorem min_people_in_photographs (photos: photographs) (h : valid_photographs photos) : 
  ∃ people : Finset Nat, people.card ≥ 16 := 
sorry

end NUMINAMATH_GPT_min_people_in_photographs_l1184_118499


namespace NUMINAMATH_GPT_find_a_l1184_118435

-- Define the conditions given in the problem
def binomial_term (r : ℕ) (a : ℝ) : ℝ :=
  Nat.choose 7 r * 2^(7-r) * (-a)^r

def coefficient_condition (a : ℝ) : Prop :=
  binomial_term 5 a = 84

-- The theorem stating the problem's solution
theorem find_a (a : ℝ) (h : coefficient_condition a) : a = -1 :=
  sorry

end NUMINAMATH_GPT_find_a_l1184_118435


namespace NUMINAMATH_GPT_total_rock_needed_l1184_118497

theorem total_rock_needed (a b : ℕ) (h₁ : a = 8) (h₂ : b = 8) : a + b = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_rock_needed_l1184_118497


namespace NUMINAMATH_GPT_brother_more_lambs_than_merry_l1184_118423

theorem brother_more_lambs_than_merry
  (merry_lambs : ℕ) (total_lambs : ℕ) (more_than_merry : ℕ)
  (h1 : merry_lambs = 10) 
  (h2 : total_lambs = 23)
  (h3 : more_than_merry + merry_lambs + merry_lambs = total_lambs) :
  more_than_merry = 3 :=
by
  sorry

end NUMINAMATH_GPT_brother_more_lambs_than_merry_l1184_118423


namespace NUMINAMATH_GPT_beckys_age_ratio_l1184_118421

theorem beckys_age_ratio (Eddie_age : ℕ) (Irene_age : ℕ)
  (becky_age: ℕ)
  (H1 : Eddie_age = 92)
  (H2 : Irene_age = 46)
  (H3 : Irene_age = 2 * becky_age) :
  becky_age / Eddie_age = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_beckys_age_ratio_l1184_118421


namespace NUMINAMATH_GPT_chris_first_day_breath_l1184_118417

theorem chris_first_day_breath (x : ℕ) (h1 : x + 10 = 20) : x = 10 :=
by
  sorry

end NUMINAMATH_GPT_chris_first_day_breath_l1184_118417
