import Mathlib

namespace boys_and_girls_l2116_211622

theorem boys_and_girls (B G : ℕ) (h1 : B + G = 30)
  (h2 : ∀ (i j : ℕ), i < B → j < B → i ≠ j → ∃ k, k < G ∧ ∀ l < B, l ≠ i → k ≠ l)
  (h3 : ∀ (i j : ℕ), i < G → j < G → i ≠ j → ∃ k, k < B ∧ ∀ l < G, l ≠ i → k ≠ l) :
  B = 15 ∧ G = 15 :=
by
  have hB : B ≤ G := sorry
  have hG : G ≤ B := sorry
  exact ⟨by linarith, by linarith⟩

end boys_and_girls_l2116_211622


namespace club_additional_members_l2116_211650

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l2116_211650


namespace roots_cubic_sum_l2116_211675

theorem roots_cubic_sum :
  (∃ x1 x2 x3 x4 : ℂ, (x1^4 + 5*x1^3 + 6*x1^2 + 5*x1 + 1 = 0) ∧
                       (x2^4 + 5*x2^3 + 6*x2^2 + 5*x2 + 1 = 0) ∧
                       (x3^4 + 5*x3^3 + 6*x3^2 + 5*x3 + 1 = 0) ∧
                       (x4^4 + 5*x4^3 + 6*x4^2 + 5*x4 + 1 = 0)) →
  (x1^3 + x2^3 + x3^3 + x4^3 = -54) :=
sorry

end roots_cubic_sum_l2116_211675


namespace selection_ways_l2116_211618

-- Define the problem parameters
def male_students : ℕ := 4
def female_students : ℕ := 3
def total_selected : ℕ := 3

-- Define the binomial coefficient function for combinatorial calculations
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define conditions
def both_genders_must_be_represented : Prop :=
  total_selected = 3 ∧ male_students >= 1 ∧ female_students >= 1

-- Problem statement: proof that the total ways to select 3 students is 30
theorem selection_ways : both_genders_must_be_represented → 
  (binomial male_students 2 * binomial female_students 1 +
   binomial male_students 1 * binomial female_students 2) = 30 :=
by
  sorry

end selection_ways_l2116_211618


namespace luncheon_cost_l2116_211616

variables (s c p : ℝ)

def eq1 := 5 * s + 8 * c + 2 * p = 5.10
def eq2 := 6 * s + 11 * c + 2 * p = 6.45

theorem luncheon_cost (h₁ : 5 * s + 8 * c + 2 * p = 5.10) (h₂ : 6 * s + 11 * c + 2 * p = 6.45) : 
  s + c + p = 1.35 :=
  sorry

end luncheon_cost_l2116_211616


namespace Sally_lost_20_Pokemon_cards_l2116_211604

theorem Sally_lost_20_Pokemon_cards (original_cards : ℕ) (received_cards : ℕ) (final_cards : ℕ) (lost_cards : ℕ) 
  (h1 : original_cards = 27) 
  (h2 : received_cards = 41) 
  (h3 : final_cards = 48) 
  (h4 : original_cards + received_cards - lost_cards = final_cards) : 
  lost_cards = 20 := 
sorry

end Sally_lost_20_Pokemon_cards_l2116_211604


namespace part_a_l2116_211644

def is_tricubic (k : ℕ) : Prop :=
  ∃ a b c : ℕ, k = a^3 + b^3 + c^3

theorem part_a : ∃ (n : ℕ), is_tricubic n ∧ ¬ is_tricubic (n + 2) ∧ ¬ is_tricubic (n + 28) :=
by 
  let n := 3 * (3*1+1)^3
  exists n
  sorry

end part_a_l2116_211644


namespace fruit_seller_price_l2116_211646

theorem fruit_seller_price (CP SP : ℝ) (h1 : SP = 0.90 * CP) (h2 : 1.10 * CP = 13.444444444444445) : 
  SP = 11 :=
sorry

end fruit_seller_price_l2116_211646


namespace harry_worked_total_hours_l2116_211608

theorem harry_worked_total_hours (x : ℝ) (H : ℝ) (H_total : ℝ) :
  (24 * x + 1.5 * x * H = 42 * x) → (H_total = 24 + H) → H_total = 36 :=
by
sorry

end harry_worked_total_hours_l2116_211608


namespace evaluate_number_l2116_211645

theorem evaluate_number (n : ℝ) (h : 22 + Real.sqrt (-4 + 6 * 4 * n) = 24) : n = 1 / 3 :=
by
  sorry

end evaluate_number_l2116_211645


namespace illuminated_cube_surface_area_l2116_211670

noncomputable def edge_length : ℝ := Real.sqrt (2 + Real.sqrt 3)
noncomputable def radius : ℝ := Real.sqrt 2
noncomputable def illuminated_area (a ρ : ℝ) : ℝ := Real.sqrt 3 * (Real.pi + 3)

theorem illuminated_cube_surface_area :
  illuminated_area edge_length radius = Real.sqrt 3 * (Real.pi + 3) := sorry

end illuminated_cube_surface_area_l2116_211670


namespace adult_dog_cost_is_100_l2116_211609

-- Define the costs for cats, puppies, and dogs.
def cat_cost : ℕ := 50
def puppy_cost : ℕ := 150

-- Define the number of each type of animal.
def number_of_cats : ℕ := 2
def number_of_adult_dogs : ℕ := 3
def number_of_puppies : ℕ := 2

-- The total cost
def total_cost : ℕ := 700

-- Define what needs to be proven: the cost of getting each adult dog ready for adoption.
theorem adult_dog_cost_is_100 (D : ℕ) (h : number_of_cats * cat_cost + number_of_adult_dogs * D + number_of_puppies * puppy_cost = total_cost) : D = 100 :=
by 
  sorry

end adult_dog_cost_is_100_l2116_211609


namespace rectangle_MQ_l2116_211682

theorem rectangle_MQ :
  ∀ (PQ QR PM MQ : ℝ),
    PQ = 4 →
    QR = 10 →
    PM = MQ →
    MQ = 2 * Real.sqrt 10 → 
    0 < MQ
:= by
  intros PQ QR PM MQ h1 h2 h3 h4
  sorry

end rectangle_MQ_l2116_211682


namespace find_a_l2116_211648

theorem find_a (a : ℝ) (i : ℂ) (h : i * i = -1) (h1 : (1 + a * i) * i = -3 + i) : a = 3 :=
by
  sorry

end find_a_l2116_211648


namespace total_songs_bought_l2116_211685

def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7

theorem total_songs_bought :
  (country_albums + pop_albums) * songs_per_album = 70 := by
  sorry

end total_songs_bought_l2116_211685


namespace largest_distance_between_spheres_l2116_211681

theorem largest_distance_between_spheres :
  let O1 := (3, -14, 8)
  let O2 := (-9, 5, -12)
  let d := Real.sqrt ((3 + 9)^2 + (-14 - 5)^2 + (8 + 12)^2)
  let r1 := 24
  let r2 := 50
  r1 + d + r2 = Real.sqrt 905 + 74 :=
by
  intro O1 O2 d r1 r2
  sorry

end largest_distance_between_spheres_l2116_211681


namespace primes_with_prime_remainders_l2116_211601

namespace PrimePuzzle

open Nat

def primes_between (a b : Nat) : List Nat :=
  (List.range' (a + 1) (b - a)).filter Nat.Prime

def prime_remainders (lst : List Nat) (m : Nat) : List Nat :=
  (lst.map (λ n => n % m)).filter Nat.Prime

theorem primes_with_prime_remainders : 
  primes_between 40 85 = [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] ∧ 
  prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12 = [5, 7, 7, 11, 11, 7, 11] ∧ 
  (prime_remainders [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83] 12).toFinset.card = 9 := 
by 
  sorry

end PrimePuzzle

end primes_with_prime_remainders_l2116_211601


namespace log2_a_plus_log2_b_zero_l2116_211642

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

theorem log2_a_plus_log2_b_zero 
    (a b : ℝ) 
    (h : (Nat.choose 6 3) * (a^3) * (b^3) = 20) 
    (hc : (a^2 + b / a)^(3) = 20 * x^(3)) :
  log2 a + log2 b = 0 :=
by
  sorry

end log2_a_plus_log2_b_zero_l2116_211642


namespace profit_percentage_l2116_211654

theorem profit_percentage (CP SP : ℝ) (h1 : CP = 500) (h2 : SP = 650) : 
  (SP - CP) / CP * 100 = 30 :=
by
  sorry

end profit_percentage_l2116_211654


namespace parabola_directrix_l2116_211617

theorem parabola_directrix (a : ℝ) (h : -1 / (4 * a) = 2) : a = -1 / 8 :=
by
  sorry

end parabola_directrix_l2116_211617


namespace not_inequality_l2116_211690

theorem not_inequality (x : ℝ) : ¬ (x^2 + 2*x - 3 < 0) :=
sorry

end not_inequality_l2116_211690


namespace range_of_k_l2116_211674

noncomputable def meets_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - y^2 = 4 ∧ y = k * x - 1

theorem range_of_k : 
  { k : ℝ | meets_hyperbola k } = { k : ℝ | k = 1 ∨ k = -1 ∨ - (Real.sqrt 5) / 2 ≤ k ∧ k ≤ (Real.sqrt 5) / 2 } :=
by
  sorry

end range_of_k_l2116_211674


namespace invested_sum_l2116_211699

theorem invested_sum (P r : ℝ) 
  (peter_total : P + 3 * P * r = 815) 
  (david_total : P + 4 * P * r = 870) 
  : P = 650 := 
by
  sorry

end invested_sum_l2116_211699


namespace brian_gallons_usage_l2116_211620

/-
Brian’s car gets 20 miles per gallon. 
On his last trip, he traveled 60 miles. 
How many gallons of gas did he use?
-/

theorem brian_gallons_usage (miles_per_gallon : ℝ) (total_miles : ℝ) (gallons_used : ℝ) 
    (h1 : miles_per_gallon = 20) 
    (h2 : total_miles = 60) 
    (h3 : gallons_used = total_miles / miles_per_gallon) : 
    gallons_used = 3 := 
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end brian_gallons_usage_l2116_211620


namespace interval_of_x_l2116_211629

theorem interval_of_x (x : ℝ) : 
  (2 < 4 * x ∧ 4 * x < 3) → (2 < 5 * x ∧ 5 * x < 3) → (1 / 2 < x ∧ x < 3 / 5) :=
by
  sorry

end interval_of_x_l2116_211629


namespace cards_given_l2116_211636

-- Defining the conditions
def initial_cards : ℕ := 4
def final_cards : ℕ := 12

-- The theorem to be proved
theorem cards_given : final_cards - initial_cards = 8 := by
  -- Proof will go here
  sorry

end cards_given_l2116_211636


namespace all_are_multiples_of_3_l2116_211692

theorem all_are_multiples_of_3 :
  (123 % 3 = 0) ∧
  (234 % 3 = 0) ∧
  (345 % 3 = 0) ∧
  (456 % 3 = 0) ∧
  (567 % 3 = 0) :=
by
  sorry

end all_are_multiples_of_3_l2116_211692


namespace salt_percentage_l2116_211669

theorem salt_percentage :
  ∀ (salt water : ℝ), salt = 10 → water = 90 → 
  100 * (salt / (salt + water)) = 10 :=
by
  intros salt water h_salt h_water
  sorry

end salt_percentage_l2116_211669


namespace find_angle_degree_l2116_211696

theorem find_angle_degree (x : ℝ) (h : 90 - x = 0.4 * (180 - x)) : x = 30 := by
  sorry

end find_angle_degree_l2116_211696


namespace toys_produced_in_week_l2116_211686

-- Define the number of working days in a week
def working_days_in_week : ℕ := 4

-- Define the number of toys produced per day
def toys_produced_per_day : ℕ := 1375

-- The statement to be proved
theorem toys_produced_in_week :
  working_days_in_week * toys_produced_per_day = 5500 :=
by
  sorry

end toys_produced_in_week_l2116_211686


namespace findMultipleOfSamsMoney_l2116_211625

-- Define the conditions specified in the problem
def SamMoney : ℕ := 75
def TotalMoney : ℕ := 200
def BillyHasLess (x : ℕ) : ℕ := x * SamMoney - 25

-- State the theorem to prove
theorem findMultipleOfSamsMoney (x : ℕ) 
  (h1 : SamMoney + BillyHasLess x = TotalMoney) : x = 2 :=
by
  -- Placeholder for the proof
  sorry

end findMultipleOfSamsMoney_l2116_211625


namespace largest_prime_divisor_25_sq_plus_72_sq_l2116_211638

theorem largest_prime_divisor_25_sq_plus_72_sq : ∃ p : ℕ, Nat.Prime p ∧ p ∣ (25^2 + 72^2) ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ (25^2 + 72^2) → q ≤ p :=
sorry

end largest_prime_divisor_25_sq_plus_72_sq_l2116_211638


namespace sum_remainders_l2116_211624

theorem sum_remainders :
  ∀ (a b c d : ℕ),
  a % 53 = 31 →
  b % 53 = 44 →
  c % 53 = 6 →
  d % 53 = 2 →
  (a + b + c + d) % 53 = 30 :=
by
  intros a b c d ha hb hc hd
  sorry

end sum_remainders_l2116_211624


namespace fred_spent_18_42_l2116_211665

variable (football_price : ℝ) (pokemon_price : ℝ) (baseball_price : ℝ)
variable (football_packs : ℕ) (pokemon_packs : ℕ) (baseball_decks : ℕ)

def total_cost (football_price : ℝ) (football_packs : ℕ) (pokemon_price : ℝ) (pokemon_packs : ℕ) (baseball_price : ℝ) (baseball_decks : ℕ) : ℝ :=
  football_packs * football_price + pokemon_packs * pokemon_price + baseball_decks * baseball_price

theorem fred_spent_18_42 :
  total_cost 2.73 2 4.01 1 8.95 1 = 18.42 :=
by
  sorry

end fred_spent_18_42_l2116_211665


namespace range_of_f_l2116_211668

def f (x : ℝ) : ℝ := x^2 - 2*x + 2

theorem range_of_f : Set.Icc 0 3 → (Set.Ico 1 5) :=
by
  sorry
  -- Here the proof steps would go, which are omitted based on your guidelines.

end range_of_f_l2116_211668


namespace original_price_l2116_211600

theorem original_price (x : ℝ) (h1 : 0.95 * x * 1.40 = 1.33 * x) (h2 : 1.33 * x = 2 * x - 1352.06) : x = 2018 := sorry

end original_price_l2116_211600


namespace prime_square_mod_12_l2116_211653

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l2116_211653


namespace gold_distribution_l2116_211623

theorem gold_distribution :
  ∃ (d : ℚ), 
    (4 * (a1: ℚ) + 6 * d = 3) ∧ 
    (3 * (a1: ℚ) + 24 * d = 4) ∧
    d = 7 / 78 :=
by {
  sorry
}

end gold_distribution_l2116_211623


namespace correct_operation_among_given_ones_l2116_211602

theorem correct_operation_among_given_ones
  (a : ℝ) :
  (a^2)^3 = a^6 :=
by {
  sorry
}

-- Auxiliary lemmas if needed (based on conditions):
lemma mul_powers_add_exponents (a : ℝ) (m n : ℕ) : a^m * a^n = a^(m + n) := by sorry

lemma power_of_a_power (a : ℝ) (m n : ℕ) : (a^m)^n = a^(m * n) := by sorry

lemma div_powers_subtract_exponents (a : ℝ) (m n : ℕ) : a^m / a^n = a^(m - n) := by sorry

lemma square_of_product (x y : ℝ) : (x * y)^2 = x^2 * y^2 := by sorry

end correct_operation_among_given_ones_l2116_211602


namespace total_sand_correct_l2116_211643

-- Define the conditions as variables and equations:
variables (x : ℕ) -- original days scheduled to complete
variables (total_sand : ℕ) -- total amount of sand in tons

-- Define the conditions in the problem:
def original_daily_amount := 15  -- tons per day as scheduled
def actual_daily_amount := 20  -- tons per day in reality
def days_ahead := 3  -- days finished ahead of schedule

-- Equation representing the planned and actual transportation:
def planned_sand := original_daily_amount * x
def actual_sand := actual_daily_amount * (x - days_ahead)

-- The goal is to prove:
theorem total_sand_correct : planned_sand = actual_sand → total_sand = 180 :=
by
  sorry

end total_sand_correct_l2116_211643


namespace candles_ratio_l2116_211662

-- Conditions
def kalani_bedroom_candles : ℕ := 20
def donovan_candles : ℕ := 20
def total_candles_house : ℕ := 50

-- Definitions for the number of candles in the living room and the ratio
def living_room_candles : ℕ := total_candles_house - kalani_bedroom_candles - donovan_candles
def ratio_of_candles : ℚ := kalani_bedroom_candles / living_room_candles

theorem candles_ratio : ratio_of_candles = 2 :=
by
  sorry

end candles_ratio_l2116_211662


namespace middle_number_of_consecutive_squares_l2116_211641

theorem middle_number_of_consecutive_squares (x : ℕ ) (h : x^2 + (x+1)^2 + (x+2)^2 = 2030) : x + 1 = 26 :=
sorry

end middle_number_of_consecutive_squares_l2116_211641


namespace area_square_A_32_l2116_211677

-- Define the areas of the squares in Figure B and Figure A and their relationship with the triangle areas
def identical_isosceles_triangles_with_squares (area_square_B : ℝ) (area_triangle_B : ℝ) (area_square_A : ℝ) (area_triangle_A : ℝ) :=
  area_triangle_B = (area_square_B / 2) * 4 ∧
  area_square_A / area_triangle_A = 4 / 9

theorem area_square_A_32 {area_square_B : ℝ} (h : area_square_B = 36) :
  identical_isosceles_triangles_with_squares area_square_B 72 32 72 :=
by
  sorry

end area_square_A_32_l2116_211677


namespace solution_system_l2116_211672

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l2116_211672


namespace notepad_days_last_l2116_211651

def fold_paper (n : Nat) : Nat := 2 ^ n

def lettersize_paper_pieces : Nat := 5
def folds : Nat := 3
def notes_per_day : Nat := 10

def smaller_note_papers_per_piece : Nat := fold_paper folds
def total_smaller_note_papers : Nat := lettersize_paper_pieces * smaller_note_papers_per_piece
def total_days : Nat := total_smaller_note_papers / notes_per_day

theorem notepad_days_last : total_days = 4 := by
  sorry

end notepad_days_last_l2116_211651


namespace distance_between_A_and_B_is_90_l2116_211615

variable (A B : Type)
variables (v_A v_B v'_A v'_B : ℝ)
variable (d : ℝ)

-- Conditions
axiom starts_simultaneously : True
axiom speed_ratio : v_A / v_B = 4 / 5
axiom A_speed_decrease : v'_A = 0.75 * v_A
axiom B_speed_increase : v'_B = 1.2 * v_B
axiom distance_when_B_reaches_A : ∃ k : ℝ, k = 30 -- Person A is 30 km away from location B

-- Goal
theorem distance_between_A_and_B_is_90 : d = 90 := by 
  sorry

end distance_between_A_and_B_is_90_l2116_211615


namespace find_a_2016_l2116_211634

-- Define the sequence a_n and its sum S_n
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Given conditions
axiom S_n_eq : ∀ n : ℕ, S n + (1 + (2 / n)) * a n = 4
axiom a_1_eq : a 1 = 1
axiom a_rec : ∀ n : ℕ, n ≥ 2 → a n = (n / (2 * (n - 1))) * a (n - 1)

-- The theorem to prove
theorem find_a_2016 : a 2016 = 2016 / 2^2015 := by
  sorry

end find_a_2016_l2116_211634


namespace find_third_number_l2116_211607

-- Definitions
def A : ℕ := 600
def B : ℕ := 840
def LCM : ℕ := 50400
def HCF : ℕ := 60

-- Theorem to be proven
theorem find_third_number (C : ℕ) (h_lcm : Nat.lcm (Nat.lcm A B) C = LCM) (h_hcf : Nat.gcd (Nat.gcd A B) C = HCF) : C = 6 :=
by -- proof
  sorry

end find_third_number_l2116_211607


namespace competition_end_time_l2116_211611

def time := ℕ × ℕ -- Representing time as a pair of hours and minutes

def start_time : time := (15, 15) -- 3:15 PM is represented as 15:15 in 24-hour format
def duration := 1825 -- Duration in minutes
def end_time : time := (21, 40) -- 9:40 PM is represented as 21:40 in 24-hour format

def add_minutes (t : time) (m : ℕ) : time :=
  let (h, min) := t
  let total_minutes := h * 60 + min + m
  (total_minutes / 60 % 24, total_minutes % 60)

theorem competition_end_time :
  add_minutes start_time duration = end_time :=
by
  -- The proof would go here
  sorry

end competition_end_time_l2116_211611


namespace Ronaldinho_age_2018_l2116_211661

variable (X : ℕ)

theorem Ronaldinho_age_2018 (h : X^2 = 2025) : X - (2025 - 2018) = 38 := by
  sorry

end Ronaldinho_age_2018_l2116_211661


namespace keith_bought_cards_l2116_211647

theorem keith_bought_cards (orig : ℕ) (now : ℕ) (bought : ℕ) 
  (h1 : orig = 40) (h2 : now = 18) (h3 : bought = orig - now) : bought = 22 := by
  sorry

end keith_bought_cards_l2116_211647


namespace initial_total_cards_l2116_211631

theorem initial_total_cards (x y : ℕ) (h1 : x / (x + y) = 1 / 3) (h2 : x / (x + y + 4) = 1 / 4) : x + y = 12 := 
sorry

end initial_total_cards_l2116_211631


namespace segment_EC_length_l2116_211663

noncomputable def length_of_segment_EC (a b c : ℕ) (angle_A_deg BC : ℝ) (BD_perp_AC CE_perp_AB : Prop) (angle_DBC_eq_3_angle_ECB : Prop) : ℝ :=
  a * (Real.sqrt b + Real.sqrt c)

theorem segment_EC_length
  (a b c : ℕ)
  (angle_A_deg BC : ℝ)
  (BD_perp_AC CE_perp_AB : Prop)
  (angle_DBC_eq_3_angle_ECB : Prop)
  (h1 : angle_A_deg = 45)
  (h2 : BC = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : angle_DBC_eq_3_angle_ECB)
  (h6 : length_of_segment_EC a b c angle_A_deg BC BD_perp_AC CE_perp_AB angle_DBC_eq_3_angle_ECB = 5 * (Real.sqrt 3 + Real.sqrt 1)) :
  a + b + c = 9 :=
  by
    sorry

end segment_EC_length_l2116_211663


namespace trig_identity_eq_one_l2116_211632

theorem trig_identity_eq_one :
  (Real.sin (160 * Real.pi / 180) + Real.sin (40 * Real.pi / 180)) *
  (Real.sin (140 * Real.pi / 180) + Real.sin (20 * Real.pi / 180)) +
  (Real.sin (50 * Real.pi / 180) - Real.sin (70 * Real.pi / 180)) *
  (Real.sin (130 * Real.pi / 180) - Real.sin (110 * Real.pi / 180)) =
  1 :=
sorry

end trig_identity_eq_one_l2116_211632


namespace find_a_l2116_211612

open Set
open Real

def A : Set ℝ := {-1, 1}
def B (a : ℝ) : Set ℝ := {x | a * x ^ 2 = 1}

theorem find_a (a : ℝ) (h : (A ∩ (B a)) = (B a)) : a = 1 :=
sorry

end find_a_l2116_211612


namespace equation_1_solve_equation_2_solve_l2116_211652

-- The first equation
theorem equation_1_solve (x : ℝ) (h : 4 * (x - 2) = 2 * x) : x = 4 :=
by
  sorry

-- The second equation
theorem equation_2_solve (x : ℝ) (h : (x + 1) / 4 = 1 - (1 - x) / 3) : x = -5 :=
by
  sorry

end equation_1_solve_equation_2_solve_l2116_211652


namespace power_mod_eq_one_l2116_211698

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l2116_211698


namespace bounded_poly_constant_l2116_211656

theorem bounded_poly_constant (P : Polynomial ℤ) (B : ℕ) (h_bounded : ∀ x : ℤ, abs (P.eval x) ≤ B) : 
  P.degree = 0 :=
sorry

end bounded_poly_constant_l2116_211656


namespace storks_more_than_birds_l2116_211619

def initial_birds := 2
def additional_birds := 3
def total_birds := initial_birds + additional_birds
def storks := 6
def difference := storks - total_birds

theorem storks_more_than_birds : difference = 1 :=
by
  sorry

end storks_more_than_birds_l2116_211619


namespace geometric_sequence_sum_terms_l2116_211689

noncomputable def geometric_sequence (a_1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_sum_terms :
  ∀ (a_1 q : ℕ), a_1 = 3 → 
  (geometric_sequence 3 q 1 + geometric_sequence 3 q 2 + geometric_sequence 3 q 3 = 21) →
  (q > 0) →
  (geometric_sequence 3 q 3 + geometric_sequence 3 q 4 + geometric_sequence 3 q 5 = 84) :=
by
  intros a_1 q h1 hsum hqpos
  sorry

end geometric_sequence_sum_terms_l2116_211689


namespace total_apples_l2116_211660

def packs : ℕ := 2
def apples_per_pack : ℕ := 4

theorem total_apples : packs * apples_per_pack = 8 := by
  sorry

end total_apples_l2116_211660


namespace modular_inverse_sum_eq_14_l2116_211687

theorem modular_inverse_sum_eq_14 : 
(9 + 13 + 15 + 16 + 12 + 3 + 14) % 17 = 14 := by
  sorry

end modular_inverse_sum_eq_14_l2116_211687


namespace find_total_kids_l2116_211639

-- Given conditions
def total_kids_in_camp (X : ℕ) : Prop :=
  let soccer_kids := X / 2
  let morning_soccer_kids := soccer_kids / 4
  let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
  afternoon_soccer_kids = 750

-- Theorem statement
theorem find_total_kids (X : ℕ) (h : total_kids_in_camp X) : X = 2000 :=
by
  sorry

end find_total_kids_l2116_211639


namespace bicycle_speed_l2116_211673

theorem bicycle_speed (d1 d2 v1 v_avg : ℝ)
  (h1 : d1 = 300) 
  (h2 : d1 + d2 = 450) 
  (h3 : v1 = 20) 
  (h4 : v_avg = 18) : 
  (d2 / ((d1 / v1) + d2 / (d2 * v_avg / 450)) = 15) :=
by 
  sorry

end bicycle_speed_l2116_211673


namespace percentage_discount_l2116_211655

-- Define the given conditions
def equal_contribution (total: ℕ) (num_people: ℕ) := total / num_people

def original_contribution (amount_paid: ℕ) (discount: ℕ) := amount_paid + discount

def total_original_cost (individual_original: ℕ) (num_people: ℕ) := individual_original * num_people

def discount_amount (original_cost: ℕ) (discounted_cost: ℕ) := original_cost - discounted_cost

def discount_percentage (discount: ℕ) (original_cost: ℕ) := (discount * 100) / original_cost

-- Given conditions
def given_total := 48
def given_num_people := 3
def amount_paid_each := equal_contribution given_total given_num_people
def discount_each := 4
def original_payment_each := original_contribution amount_paid_each discount_each
def original_total_cost := total_original_cost original_payment_each given_num_people
def paid_total := 48

-- Question: What is the percentage discount
theorem percentage_discount :
  discount_percentage (discount_amount original_total_cost paid_total) original_total_cost = 20 :=
by
  sorry

end percentage_discount_l2116_211655


namespace least_positive_int_satisfies_congruence_l2116_211605

theorem least_positive_int_satisfies_congruence :
  ∃ x : ℕ, (x + 3001) % 15 = 1723 % 15 ∧ x = 12 :=
by
  sorry

end least_positive_int_satisfies_congruence_l2116_211605


namespace change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l2116_211613

-- Given f(x) = x^2 - 5x
def f (x : ℝ) : ℝ := x^2 - 5 * x

-- Prove the change in f(x) when x is increased by 2 is 4x - 6
theorem change_in_f_when_x_increased_by_2 (x : ℝ) : f (x + 2) - f x = 4 * x - 6 := by
  sorry

-- Prove the change in f(x) when x is decreased by 2 is -4x + 14
theorem change_in_f_when_x_decreased_by_2 (x : ℝ) : f (x - 2) - f x = -4 * x + 14 := by
  sorry

end change_in_f_when_x_increased_by_2_change_in_f_when_x_decreased_by_2_l2116_211613


namespace elevator_translation_l2116_211671

-- Definitions based on conditions
def turning_of_steering_wheel : Prop := False
def rotation_of_bicycle_wheels : Prop := False
def motion_of_pendulum : Prop := False
def movement_of_elevator : Prop := True

-- Theorem statement
theorem elevator_translation :
  movement_of_elevator := by
  exact True.intro

end elevator_translation_l2116_211671


namespace calculate_series_l2116_211679

theorem calculate_series : 20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 200 := 
by
  sorry

end calculate_series_l2116_211679


namespace third_circle_radius_l2116_211603

theorem third_circle_radius (r1 r2 d : ℝ) (τ : ℝ) (h1: r1 = 1) (h2: r2 = 9) (h3: d = 17) : 
  τ = 225 / 64 :=
by
  sorry

end third_circle_radius_l2116_211603


namespace four_inv_mod_35_l2116_211694

theorem four_inv_mod_35 : ∃ x : ℕ, 4 * x ≡ 1 [MOD 35] ∧ x = 9 := 
by 
  use 9
  sorry

end four_inv_mod_35_l2116_211694


namespace gcd_of_4410_and_10800_l2116_211628

theorem gcd_of_4410_and_10800 : Nat.gcd 4410 10800 = 90 := 
by 
  sorry

end gcd_of_4410_and_10800_l2116_211628


namespace Rhett_rent_expense_l2116_211649

-- Define the problem statement using given conditions
theorem Rhett_rent_expense
  (late_payments : ℕ := 2)
  (no_late_fees : Bool := true)
  (fraction_of_salary : ℝ := 3 / 5)
  (monthly_salary : ℝ := 5000)
  (tax_rate : ℝ := 0.1) :
  let salary_after_taxes := monthly_salary * (1 - tax_rate)
  let total_late_rent := fraction_of_salary * salary_after_taxes
  let monthly_rent_expense := total_late_rent / late_payments
  monthly_rent_expense = 1350 := by
  sorry

end Rhett_rent_expense_l2116_211649


namespace original_divisor_in_terms_of_Y_l2116_211667

variables (N D Y : ℤ)
variables (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4)

theorem original_divisor_in_terms_of_Y (h1 : N = 45 * D + 13) (h2 : N = 6 * Y + 4) : 
  D = (2 * Y - 3) / 15 :=
sorry

end original_divisor_in_terms_of_Y_l2116_211667


namespace sin_alpha_eq_three_fifths_l2116_211637

theorem sin_alpha_eq_three_fifths (α : ℝ) 
  (h1 : α ∈ Set.Ioo (π / 2) π) 
  (h2 : Real.tan α = -3 / 4) 
  (h3 : Real.sin α > 0) 
  (h4 : Real.cos α < 0) 
  (h5 : Real.sin α ^ 2 + Real.cos α ^ 2 = 1) : 
  Real.sin α = 3 / 5 := 
sorry

end sin_alpha_eq_three_fifths_l2116_211637


namespace symmetric_points_origin_l2116_211630

theorem symmetric_points_origin (a b : ℝ) (h1 : 1 = -b) (h2 : a = 2) : a + b = 1 := by
  sorry

end symmetric_points_origin_l2116_211630


namespace tangent_line_through_points_of_tangency_l2116_211678

noncomputable def equation_of_tangent_line (x1 y1 x y : ℝ) : Prop :=
x1 * x + (y1 - 2) * (y - 2) = 4

theorem tangent_line_through_points_of_tangency
  (x1 y1 x2 y2 : ℝ)
  (h1 : equation_of_tangent_line x1 y1 2 (-2))
  (h2 : equation_of_tangent_line x2 y2 2 (-2)) :
  (2 * x1 - 4 * (y1 - 2) = 4) ∧ (2 * x2 - 4 * (y2 - 2) = 4) →
  ∃ a b c, (a = 1) ∧ (b = -2) ∧ (c = 2) ∧ (a * x + b * y + c = 0) :=
by
  sorry

end tangent_line_through_points_of_tangency_l2116_211678


namespace rhombus_diagonals_perpendicular_l2116_211693

section circumscribed_quadrilateral

variables {a b c d : ℝ}

-- Definition of a tangential quadrilateral satisfying Pitot's theorem.
def tangential_quadrilateral (a b c d : ℝ) :=
  a + c = b + d

-- Defining a rhombus in terms of its sides
def rhombus (a b c d : ℝ) :=
  a = b ∧ b = c ∧ c = d

-- The theorem we want to prove
theorem rhombus_diagonals_perpendicular
  (h : tangential_quadrilateral a b c d)
  (hr : rhombus a b c d) : 
  true := sorry

end circumscribed_quadrilateral

end rhombus_diagonals_perpendicular_l2116_211693


namespace rectangle_dimensions_l2116_211627

theorem rectangle_dimensions (x : ℝ) (h : 3 * x * x = 8 * x) : (x = 8 / 3 ∧ 3 * x = 8) :=
by {
  sorry
}

end rectangle_dimensions_l2116_211627


namespace even_function_on_neg_interval_l2116_211697

theorem even_function_on_neg_interval
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_incr : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → f x₁ ≤ f x₂)
  (h_min : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → 0 ≤ f x) :
  (∀ x : ℝ, -3 ≤ x → x ≤ -1 → 0 ≤ f x) ∧ (∀ x₁ x₂ : ℝ, -3 ≤ x₁ → x₁ < x₂ → x₂ ≤ -1 → f x₁ ≥ f x₂) :=
sorry

end even_function_on_neg_interval_l2116_211697


namespace area_of_fourth_rectangle_l2116_211635

variable (x y z w : ℝ)
variable (Area_EFGH Area_EIKJ Area_KLMN Perimeter : ℝ)

def conditions :=
  (Area_EFGH = x * y ∧ Area_EFGH = 20 ∧
   Area_EIKJ = x * w ∧ Area_EIKJ = 25 ∧
   Area_KLMN = z * w ∧ Area_KLMN = 15 ∧
   Perimeter = 2 * (x + z + y + w) ∧ Perimeter = 40)

theorem area_of_fourth_rectangle (h : conditions x y z w Area_EFGH Area_EIKJ Area_KLMN Perimeter) :
  (y * w = 340) :=
by
  sorry

end area_of_fourth_rectangle_l2116_211635


namespace download_time_ratio_l2116_211676

-- Define the conditions of the problem
def mac_download_time : ℕ := 10
def audio_glitches : ℕ := 2 * 4
def video_glitches : ℕ := 6
def time_with_glitches : ℕ := audio_glitches + video_glitches
def time_without_glitches : ℕ := 2 * time_with_glitches
def total_time : ℕ := 82

-- Define the Windows download time as a variable
def windows_download_time : ℕ := total_time - (mac_download_time + time_with_glitches + time_without_glitches)

-- Prove the required ratio
theorem download_time_ratio : 
  (windows_download_time / mac_download_time = 3) :=
by
  -- Perform a straightforward calculation as defined in the conditions and solution steps
  sorry

end download_time_ratio_l2116_211676


namespace emily_num_dresses_l2116_211640

theorem emily_num_dresses (M : ℕ) (D : ℕ) (E : ℕ) 
  (h1 : D = M + 12) 
  (h2 : M = E / 2) 
  (h3 : M + D + E = 44) : 
  E = 16 := 
by 
  sorry

end emily_num_dresses_l2116_211640


namespace find_missing_employee_l2116_211684

-- Definitions based on the problem context
def employee_numbers : List Nat := List.range (52)
def sample_size := 4

-- The given conditions, stating that these employees are in the sample
def in_sample (x : Nat) : Prop := x = 6 ∨ x = 32 ∨ x = 45 ∨ x = 19

-- Define systematic sampling method condition
def systematic_sample (nums : List Nat) (size interval : Nat) : Prop :=
  nums = List.map (fun i => 6 + i * interval % 52) (List.range size)

-- The employees in the sample must include 6
def start_num := 6
def interval := 13
def expected_sample := [6, 19, 32, 45]

-- The Lean theorem we need to prove
theorem find_missing_employee :
  systematic_sample expected_sample sample_size interval ∧
  in_sample 6 ∧ in_sample 32 ∧ in_sample 45 →
  in_sample 19 :=
by
  sorry

end find_missing_employee_l2116_211684


namespace tom_coins_worth_l2116_211659

-- Definitions based on conditions:
def total_coins : ℕ := 30
def value_difference_cents : ℕ := 90
def nickel_value_cents : ℕ := 5
def dime_value_cents : ℕ := 10

-- Main theorem statement:
theorem tom_coins_worth (n d : ℕ) (h1 : d = total_coins - n) 
    (h2 : (nickel_value_cents * n + dime_value_cents * d) - (dime_value_cents * n + nickel_value_cents * d) = value_difference_cents) : 
    (nickel_value_cents * n + dime_value_cents * d) = 180 :=
by
  sorry -- Proof omitted.

end tom_coins_worth_l2116_211659


namespace inequality_holds_l2116_211614

variable (a b c : ℝ)

theorem inequality_holds (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + b*c) / (a * (b + c)) + 
  (b^2 + c*a) / (b * (c + a)) + 
  (c^2 + a*b) / (c * (a + b)) ≥ 3 :=
sorry

end inequality_holds_l2116_211614


namespace jason_flames_per_minute_l2116_211606

theorem jason_flames_per_minute :
  (∀ (t : ℕ), t % 15 = 0 -> (5 * (t / 15) = 20)) :=
sorry

end jason_flames_per_minute_l2116_211606


namespace dow_jones_morning_value_l2116_211691

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end dow_jones_morning_value_l2116_211691


namespace math_expression_equivalent_l2116_211626

theorem math_expression_equivalent :
  ((π - 1)^0 + 4 * (Real.sin (Real.pi / 4)) - Real.sqrt 8 + abs (-3) = 4) :=
by
  sorry

end math_expression_equivalent_l2116_211626


namespace triangular_angles_l2116_211666

noncomputable def measure_of_B (A : ℝ) : ℝ :=
  Real.arcsin (Real.sqrt ((1 + Real.sin (2 * A)) / 3))

noncomputable def length_of_c (A : ℝ) : ℝ := 
  Real.sqrt (22 - 6 * Real.sqrt 13 * Real.cos (measure_of_B A))

noncomputable def area_of_triangle_ABC (A : ℝ) : ℝ := 
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * A)) / 3)

theorem triangular_angles 
  (a b c : ℝ) (b_pos : b = Real.sqrt 13) (a_pos : a = 3) (h : b * Real.cos c = (2 * a - c) * Real.cos (measure_of_B c)) :
  c = length_of_c c ∧
  (3 * Real.sqrt 13 / 2) * Real.sqrt ((1 + Real.sin (2 * c)) / 3) = area_of_triangle_ABC c :=
by
  sorry

end triangular_angles_l2116_211666


namespace apples_in_each_basket_l2116_211658

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end apples_in_each_basket_l2116_211658


namespace max_radius_of_inscribable_circle_l2116_211610

theorem max_radius_of_inscribable_circle
  (AB BC CD DA : ℝ) (x y z w : ℝ)
  (h1 : AB = 10) (h2 : BC = 12) (h3 : CD = 8) (h4 : DA = 14)
  (h5 : x + y = 10) (h6 : y + z = 12)
  (h7 : z + w = 8) (h8 : w + x = 14)
  (h9 : x + z = y + w) :
  ∃ r : ℝ, r = Real.sqrt 24.75 :=
by
  sorry

end max_radius_of_inscribable_circle_l2116_211610


namespace pond_contains_total_money_correct_l2116_211657

def value_of_dime := 10
def value_of_quarter := 25
def value_of_nickel := 5
def value_of_penny := 1

def cindy_dimes := 5
def eric_quarters := 3
def garrick_nickels := 8
def ivy_pennies := 60

def total_money : ℕ := 
  cindy_dimes * value_of_dime + 
  eric_quarters * value_of_quarter + 
  garrick_nickels * value_of_nickel + 
  ivy_pennies * value_of_penny

theorem pond_contains_total_money_correct:
  total_money = 225 := by
  sorry

end pond_contains_total_money_correct_l2116_211657


namespace sum_inverse_one_minus_roots_eq_half_l2116_211664

noncomputable def cubic_eq_roots (x : ℝ) : ℝ := 10 * x^3 - 25 * x^2 + 8 * x - 1

theorem sum_inverse_one_minus_roots_eq_half
  {p q s : ℝ} (hpqseq : cubic_eq_roots p = 0 ∧ cubic_eq_roots q = 0 ∧ cubic_eq_roots s = 0)
  (hpospq : 0 < p ∧ 0 < q ∧ 0 < s) (hlespq : p < 1 ∧ q < 1 ∧ s < 1) :
  (1 / (1 - p)) + (1 / (1 - q)) + (1 / (1 - s)) = 1 / 2 :=
sorry

end sum_inverse_one_minus_roots_eq_half_l2116_211664


namespace masha_comb_teeth_count_l2116_211688

theorem masha_comb_teeth_count (katya_teeth : ℕ) (masha_to_katya_ratio : ℕ) 
  (katya_teeth_eq : katya_teeth = 11) 
  (masha_to_katya_ratio_eq : masha_to_katya_ratio = 5) : 
  ∃ masha_teeth : ℕ, masha_teeth = 53 :=
by
  have katya_segments := 2 * katya_teeth - 1
  have masha_segments := masha_to_katya_ratio * katya_segments
  let masha_teeth := (masha_segments + 1) / 2
  use masha_teeth
  have masha_teeth_eq := (2 * masha_teeth - 1 = 105)
  sorry

end masha_comb_teeth_count_l2116_211688


namespace min_value_of_sum_of_squares_l2116_211695

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x - 2 * y - 3 * z = 4) : 
  (x^2 + y^2 + z^2) ≥ 8 / 7 :=
sorry

end min_value_of_sum_of_squares_l2116_211695


namespace find_a_minus_b_l2116_211683

theorem find_a_minus_b (a b : ℝ)
  (h1 : 6 = a * 3 + b)
  (h2 : 26 = a * 7 + b) :
  a - b = 14 := 
sorry

end find_a_minus_b_l2116_211683


namespace A_lt_B_l2116_211621

variable (x y : ℝ)

def A (x y : ℝ) : ℝ := - y^2 + 4 * x - 3
def B (x y : ℝ) : ℝ := x^2 + 2 * x + 2 * y

theorem A_lt_B (x y : ℝ) : A x y < B x y := 
by
  sorry

end A_lt_B_l2116_211621


namespace total_stars_l2116_211633

theorem total_stars (g s : ℕ) (hg : g = 10^11) (hs : s = 10^11) : g * s = 10^22 :=
by
  rw [hg, hs]
  sorry

end total_stars_l2116_211633


namespace valid_schedule_count_l2116_211680

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count_l2116_211680
