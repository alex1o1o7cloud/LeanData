import Mathlib

namespace sum_positive_132_l126_126988

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (d : ℝ)
variable (n : ℕ)

def arithmetic_sequence (a d : ℝ) (n : ℕ) : ℝ := a + n * d

theorem sum_positive_132 {a: ℕ → ℝ}
  (h1: a 66 < 0)
  (h2: a 67 > 0)
  (h3: a 67 > |a 66|):
  ∃ n, ∀ k < n, S k > 0 :=
by
  have h4 : (a 67 - a 66) > 0 := sorry
  have h5 : a 67 + a 66 > 0 := sorry
  have h6 : 66 * (a 67 + a 66) > 0 := sorry
  have h7 : S 132 = 66 * (a 67 + a 66) := sorry
  existsi 132
  intro k hk
  sorry

end sum_positive_132_l126_126988


namespace line_intersects_y_axis_at_0_2_l126_126731

theorem line_intersects_y_axis_at_0_2 :
  ∃ y : ℝ, (2, 8) ≠ (4, 14) ∧ ∀ x: ℝ, (3 * x + y = 2) ∧ x = 0 → y = 2 :=
by
  sorry

end line_intersects_y_axis_at_0_2_l126_126731


namespace problem_l126_126569

theorem problem (p q r : ℝ) (h1 : p + q + r = 5000) (h2 : r = (2 / 3) * (p + q)) : r = 2000 :=
by
  sorry

end problem_l126_126569


namespace range_of_function_l126_126167

theorem range_of_function : 
  ∀ y : ℝ, (∃ x : ℝ, y = x / (1 + x^2)) ↔ (-1 / 2 ≤ y ∧ y ≤ 1 / 2) := 
by sorry

end range_of_function_l126_126167


namespace dino_remaining_balance_is_4650_l126_126316

def gigA_hours : Nat := 20
def gigA_rate : Nat := 10

def gigB_hours : Nat := 30
def gigB_rate : Nat := 20

def gigC_hours : Nat := 5
def gigC_rate : Nat := 40

def gigD_hours : Nat := 15
def gigD_rate : Nat := 25

def gigE_hours : Nat := 10
def gigE_rate : Nat := 30

def january_expense : Nat := 500
def february_expense : Nat := 550
def march_expense : Nat := 520
def april_expense : Nat := 480

theorem dino_remaining_balance_is_4650 :
  let gigA_earnings := gigA_hours * gigA_rate
  let gigB_earnings := gigB_hours * gigB_rate
  let gigC_earnings := gigC_hours * gigC_rate
  let gigD_earnings := gigD_hours * gigD_rate
  let gigE_earnings := gigE_hours * gigE_rate

  let total_monthly_earnings := gigA_earnings + gigB_earnings + gigC_earnings + gigD_earnings + gigE_earnings

  let total_expenses := january_expense + february_expense + march_expense + april_expense

  let total_earnings_four_months := total_monthly_earnings * 4

  total_earnings_four_months - total_expenses = 4650 :=
by {
  sorry
}

end dino_remaining_balance_is_4650_l126_126316


namespace find_number_l126_126412

theorem find_number (x : ℝ) (h : 0.65 * x = 0.8 * x - 21) : x = 140 := by
  sorry

end find_number_l126_126412


namespace vincent_earnings_l126_126869

-- Definitions based on the problem conditions
def fantasy_book_cost : ℕ := 4
def literature_book_cost : ℕ := fantasy_book_cost / 2
def fantasy_books_sold_per_day : ℕ := 5
def literature_books_sold_per_day : ℕ := 8
def duration : ℕ := 5

-- Calculation functions
def daily_earnings_from_fantasy_books : ℕ := fantasy_books_sold_per_day * fantasy_book_cost
def daily_earnings_from_literature_books : ℕ := literature_books_sold_per_day * literature_book_cost
def total_daily_earnings : ℕ := daily_earnings_from_fantasy_books + daily_earnings_from_literature_books
def total_earnings_after_five_days : ℕ := total_daily_earnings * duration

-- Statement to prove
theorem vincent_earnings : total_earnings_after_five_days = 180 := 
by
  calc total_daily_earnings * duration = 180 : sorry

end vincent_earnings_l126_126869


namespace quadratic_roots_eqn_l126_126643

theorem quadratic_roots_eqn (b c : ℝ) (x1 x2 : ℝ) (h1 : x1 = -2) (h2 : x2 = 3) (h3 : b = -(x1 + x2)) (h4 : c = x1 * x2) : 
    (x^2 + b * x + c = 0) ↔ (x^2 - x - 6 = 0) :=
by
  sorry

end quadratic_roots_eqn_l126_126643


namespace alan_total_cost_is_84_l126_126272

noncomputable def price_AVN : ℝ := 12
noncomputable def multiplier : ℝ := 2
noncomputable def count_Dark : ℕ := 2
noncomputable def count_AVN : ℕ := 1
noncomputable def count_90s : ℕ := 5
noncomputable def percentage_90s : ℝ := 0.40

def main_theorem : Prop :=
  let price_Dark := price_AVN * multiplier in
  let total_cost_Dark := price_Dark * count_Dark in
  let total_cost_AVN := price_AVN * count_AVN in
  let total_cost_other := total_cost_Dark + total_cost_AVN in
  let cost_90s := percentage_90s * total_cost_other in
  let total_cost := total_cost_other + cost_90s in
  total_cost = 84

theorem alan_total_cost_is_84 : main_theorem :=
  sorry

end alan_total_cost_is_84_l126_126272


namespace morales_sisters_revenue_l126_126602

variable (Gabriela Alba Maricela : Nat)
variable (trees_per_grove : Nat := 110)
variable (oranges_per_tree : (Nat × Nat × Nat) := (600, 400, 500))
variable (oranges_per_cup : Nat := 3)
variable (price_per_cup : Nat := 4)

theorem morales_sisters_revenue :
  let G := trees_per_grove * oranges_per_tree.fst
  let A := trees_per_grove * oranges_per_tree.snd
  let M := trees_per_grove * oranges_per_tree.snd.snd
  let total_oranges := G + A + M
  let total_cups := total_oranges / oranges_per_cup
  let total_revenue := total_cups * price_per_cup
  total_revenue = 220000 :=
by 
  sorry

end morales_sisters_revenue_l126_126602


namespace Xia_shared_stickers_l126_126254

def stickers_shared (initial remaining sheets_per_sheet : ℕ) : ℕ :=
  initial - (remaining * sheets_per_sheet)

theorem Xia_shared_stickers :
  stickers_shared 150 5 10 = 100 :=
by
  sorry

end Xia_shared_stickers_l126_126254


namespace cost_of_one_dozen_pens_l126_126681

theorem cost_of_one_dozen_pens (x n : ℕ) (h₁ : 5 * n * x + 5 * x = 200) (h₂ : ∀ p : ℕ, p > 0 → p ≠ x * 5 → x * 5 ≠ x) :
  12 * 5 * x = 120 :=
by
  sorry

end cost_of_one_dozen_pens_l126_126681


namespace eval_expression_l126_126603

theorem eval_expression : 8 / 4 - 3^2 - 10 + 5 * 2 = -7 :=
by
  sorry

end eval_expression_l126_126603


namespace greatest_difference_l126_126856

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l126_126856


namespace total_volume_of_four_cubes_is_500_l126_126551

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l126_126551


namespace angle_A_eval_triangle_area_eval_l126_126197

-- Define the conditions from part (1)
noncomputable def angle_A (A B C : ℝ) (a c : ℝ) (h_angle_order : A < B ∧ B < C) (h_C_eq_2A : C = 2 * A) (h_c_sqrt3a : c = Real.sqrt 3 * a) : ℝ :=
A

-- The proof statement for part (1)
theorem angle_A_eval
  (A B C : ℝ) (a c : ℝ)
  (h_angle_order : A < B ∧ B < C)
  (h_C_eq_2A : C = 2 * A)
  (h_c_sqrt3a : c = Real.sqrt 3 * a) :
  angle_A A B C a c h_angle_order h_C_eq_2A h_c_sqrt3a = Real.pi / 6 :=
sorry

-- Define the conditions from part (2)
noncomputable def triangle_area (A B C a b c : ℝ) (h_consecutive_integers : a = b - 1 ∧ c = b + 1) (h_angle_order : A < B ∧ B < C) (h_C_eq_2A : C = 2 * A) : ℝ :=
0.5 * b * c * Real.sin A

-- The proof statement for part (2)
theorem triangle_area_eval
  (A B C a b c : ℝ)
  (h_consecutive_integers : a = b - 1 ∧ c = b + 1)
  (h_angle_order : A < B ∧ B < C)
  (h_C_eq_2A : C = 2 * A)
  (h_b_five : b = 5)
  (h_a_four : a = 4)
  (h_c_six : c = 6)
  (h_cosA : Real.cos A = 3 / 4)
  (h_sinA : Real.sin A = Real.sqrt 7 / 4):
  triangle_area A B C a b c h_consecutive_integers h_angle_order h_C_eq_2A = 15 * Real.sqrt 7 / 4 :=
sorry

end angle_A_eval_triangle_area_eval_l126_126197


namespace range_of_m_l126_126328

theorem range_of_m (m : ℝ) (h₁ : ∀ x : ℝ, -x^2 + 7*x + 8 ≥ 0 → x^2 - 7*x - 8 ≤ 0)
  (h₂ : ∀ x : ℝ, x^2 - 2*x + 1 - 4*m^2 ≤ 0 → 1 - 2*m ≤ x ∧ x ≤ 1 + 2*m)
  (not_p_sufficient_for_not_q : ∀ x : ℝ, ¬(-x^2 + 7*x + 8 ≥ 0) → ¬(x^2 - 2*x + 1 - 4*m^2 ≤ 0))
  (suff_non_necess : ∀ x : ℝ, (x^2 - 2*x + 1 - 4*m^2 ≤ 0) → ¬(x^2 - 7*x - 8 ≤ 0))
  : 0 < m ∧ m ≤ 1 := sorry

end range_of_m_l126_126328


namespace friends_meeting_both_movie_and_games_l126_126705

theorem friends_meeting_both_movie_and_games 
  (T M P G M_and_P P_and_G M_and_P_and_G : ℕ) 
  (hT : T = 31) 
  (hM : M = 10) 
  (hP : P = 20) 
  (hG : G = 5) 
  (hM_and_P : M_and_P = 4) 
  (hP_and_G : P_and_G = 0) 
  (hM_and_P_and_G : M_and_P_and_G = 2) : (M + P + G - M_and_P - T + M_and_P_and_G - 2) = 2 := 
by 
  sorry

end friends_meeting_both_movie_and_games_l126_126705


namespace minimum_distance_between_extrema_is_2_sqrt_pi_l126_126684

noncomputable def minimum_distance_adjacent_extrema (a : ℝ) (h : a > 0) : ℝ := 2 * Real.sqrt Real.pi

theorem minimum_distance_between_extrema_is_2_sqrt_pi (a : ℝ) (h : a > 0) :
  minimum_distance_adjacent_extrema a h = 2 * Real.sqrt Real.pi := 
sorry

end minimum_distance_between_extrema_is_2_sqrt_pi_l126_126684


namespace initial_water_amount_gallons_l126_126562

theorem initial_water_amount_gallons 
  (cup_capacity_oz : ℕ)
  (rows : ℕ)
  (chairs_per_row : ℕ)
  (water_left_oz : ℕ)
  (oz_per_gallon : ℕ)
  (total_gallons : ℕ)
  (h1 : cup_capacity_oz = 6)
  (h2 : rows = 5)
  (h3 : chairs_per_row = 10)
  (h4 : water_left_oz = 84)
  (h5 : oz_per_gallon = 128)
  (h6 : total_gallons = (rows * chairs_per_row * cup_capacity_oz + water_left_oz) / oz_per_gallon) :
  total_gallons = 3 := 
by sorry

end initial_water_amount_gallons_l126_126562


namespace total_space_compacted_l126_126830

-- Definitions according to the conditions
def num_cans : ℕ := 60
def space_per_can_before : ℝ := 30
def compaction_rate : ℝ := 0.20

-- Theorem statement
theorem total_space_compacted : num_cans * (space_per_can_before * compaction_rate) = 360 := by
  sorry

end total_space_compacted_l126_126830


namespace students_suggested_tomatoes_79_l126_126517

theorem students_suggested_tomatoes_79 (T : ℕ)
  (mashed_potatoes : ℕ)
  (h1 : mashed_potatoes = 144)
  (h2 : mashed_potatoes = T + 65) :
  T = 79 :=
by {
  -- Proof steps will go here
  sorry
}

end students_suggested_tomatoes_79_l126_126517


namespace exists_q_lt_1_l126_126507

variable {a : ℕ → ℝ}

theorem exists_q_lt_1 (h_nonneg : ∀ n, 0 ≤ a n)
  (h_rec : ∀ k m, a (k + m) ≤ a (k + m + 1) + a k * a m)
  (h_large_n : ∃ n₀, ∀ n ≥ n₀, n * a n < 0.2499) :
  ∃ q, 0 < q ∧ q < 1 ∧ (∃ n₀, ∀ n ≥ n₀, a n < q ^ n) :=
by
  sorry

end exists_q_lt_1_l126_126507


namespace mass_of_three_packages_l126_126670

noncomputable def total_mass {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : ℝ := 
  x + y + z

theorem mass_of_three_packages {x y z : ℝ} (h1 : x + y = 112) (h2 : y + z = 118) (h3 : z + x = 120) : total_mass h1 h2 h3 = 175 :=
by
  sorry

end mass_of_three_packages_l126_126670


namespace watch_cost_price_l126_126566

theorem watch_cost_price (cost_price : ℝ)
  (h1 : SP_loss = 0.90 * cost_price)
  (h2 : SP_gain = 1.08 * cost_price)
  (h3 : SP_gain - SP_loss = 540) :
  cost_price = 3000 := 
sorry

end watch_cost_price_l126_126566


namespace probability_exactly_two_singers_same_province_l126_126263

-- Defining the number of provinces and number of singers per province
def num_provinces : ℕ := 6
def singers_per_province : ℕ := 2

-- Total number of singers
def num_singers : ℕ := num_provinces * singers_per_province

-- Define the total number of ways to choose 4 winners from 12 contestants
def total_combinations : ℕ := Nat.choose num_singers 4

-- Define the number of favorable ways to select exactly two singers from the same province and two from two other provinces
def favorable_combinations : ℕ := 
  (Nat.choose num_provinces 1) *  -- Choose one province for the pair
  (Nat.choose (num_provinces - 1) 2) *  -- Choose two remaining provinces
  (Nat.choose singers_per_province 1) *
  (Nat.choose singers_per_province 1)

-- Calculate the probability
def probability : ℚ := favorable_combinations / total_combinations

-- Stating the theorem to be proved
theorem probability_exactly_two_singers_same_province : probability = 16 / 33 :=
by
  sorry

end probability_exactly_two_singers_same_province_l126_126263


namespace num_distinct_prime_factors_90_l126_126784

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l126_126784


namespace quadratic_roots_solution_l126_126452

noncomputable def quadratic_roots_differ_by_2 (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) : Prop :=
  let root1 := (-p + Real.sqrt (p^2 - 4*q)) / 2
  let root2 := (-p - Real.sqrt (p^2 - 4*q)) / 2
  abs (root1 - root2) = 2

theorem quadratic_roots_solution (p q : ℝ) (hq_pos : 0 < q) (hp_pos : 0 < p) :
  quadratic_roots_differ_by_2 p q hq_pos hp_pos →
  p = 2 * Real.sqrt (q + 1) :=
sorry

end quadratic_roots_solution_l126_126452


namespace union_eq_M_l126_126822

def M : Set ℝ := {y | ∃ x : ℝ, y = 2^x}
def S : Set ℝ := {y | ∃ x : ℝ, y = x - 1}

theorem union_eq_M : M ∪ S = M := by
  /- this part is for skipping the proof -/
  sorry

end union_eq_M_l126_126822


namespace franks_daily_reading_l126_126961

-- Define the conditions
def total_pages : ℕ := 612
def days_to_finish : ℕ := 6

-- State the theorem we want to prove
theorem franks_daily_reading : (total_pages / days_to_finish) = 102 :=
by
  sorry

end franks_daily_reading_l126_126961


namespace lifespan_represents_sample_l126_126120

-- Definitions
def survey_population := 2500
def provinces_and_cities := 11

-- Theorem stating that the lifespan of the urban residents surveyed represents a sample
theorem lifespan_represents_sample
  (number_of_residents : ℕ) (num_provinces : ℕ) 
  (h₁ : number_of_residents = survey_population)
  (h₂ : num_provinces = provinces_and_cities) :
  "Sample" = "Sample" :=
by 
  -- Proof skipped
  sorry

end lifespan_represents_sample_l126_126120


namespace common_difference_arithmetic_sequence_l126_126067

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 5 = 10) (h2 : a 12 = 31) : d = 3 :=
by
  sorry

end common_difference_arithmetic_sequence_l126_126067


namespace find_pairs_l126_126608

theorem find_pairs (m n : ℕ) (h1 : m > 0) (h2 : n > 0)
  (cond1 : (m^2 - n) ∣ (m + n^2))
  (cond2 : (n^2 - m) ∣ (n + m^2)) :
  (m = 2 ∧ n = 1) ∨ (m = 3 ∧ n = 2) := 
sorry

end find_pairs_l126_126608


namespace spinsters_count_l126_126416

theorem spinsters_count (S C : ℕ) (h1 : S / C = 2 / 9) (h2 : C = S + 42) : S = 12 := by
  sorry

end spinsters_count_l126_126416


namespace quad_vertex_transform_l126_126215

theorem quad_vertex_transform :
  ∀ (x y : ℝ) (h : y = -2 * x^2) (new_x new_y : ℝ) (h_translation : new_x = x + 3 ∧ new_y = y - 2),
  new_y = -2 * (new_x - 3)^2 + 2 :=
by
  intros x y h new_x new_y h_translation
  sorry

end quad_vertex_transform_l126_126215


namespace marie_erasers_l126_126211

theorem marie_erasers (initial_erasers : ℕ) (lost_erasers : ℕ) (final_erasers : ℕ) :
  initial_erasers = 95 → lost_erasers = 42 → final_erasers = initial_erasers - lost_erasers → final_erasers = 53 :=
by
  intros h_initial h_lost h_final
  rw [h_initial, h_lost] at h_final
  exact h_final

end marie_erasers_l126_126211


namespace ladder_length_l126_126419

/-- The length of the ladder leaning against a wall when it forms
    a 60 degree angle with the ground and the foot of the ladder 
    is 9.493063650744542 m from the wall is 18.986127301489084 m. -/
theorem ladder_length (L : ℝ) (adjacent : ℝ) (θ : ℝ) (cosθ : ℝ) :
  θ = Real.pi / 3 ∧ adjacent = 9.493063650744542 ∧ cosθ = Real.cos θ →
  L = 18.986127301489084 :=
by
  intro h
  sorry

end ladder_length_l126_126419


namespace problem_ABC_sum_l126_126746

-- Let A, B, and C be positive integers such that A and C, B and C, and A and B
-- have no common factor greater than 1.
-- If they satisfy the equation A * log_100 5 + B * log_100 4 = C,
-- then we need to prove that A + B + C = 4.

theorem problem_ABC_sum (A B C : ℕ) (h1 : 1 < A ∧ 1 < B ∧ 1 < C)
    (h2 : A.gcd B = 1 ∧ B.gcd C = 1 ∧ A.gcd C = 1)
    (h3 : A * Real.log 5 / Real.log 100 + B * Real.log 4 / Real.log 100 = C) :
    A + B + C = 4 :=
sorry

end problem_ABC_sum_l126_126746


namespace trigonometric_identity_l126_126973

open Real

theorem trigonometric_identity :
  (sin (20 * π / 180) * sin (80 * π / 180) - cos (160 * π / 180) * sin (10 * π / 180) = 1 / 2) :=
by
  -- Trigonometric calculations
  sorry

end trigonometric_identity_l126_126973


namespace domain_f_1_minus_2x_is_0_to_half_l126_126627

-- Define the domain of f(x) as a set.
def domain_f (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 1

-- Define the domain condition for f(1 - 2*x).
def domain_f_1_minus_2x (x : ℝ) : Prop := 0 ≤ 1 - 2 * x ∧ 1 - 2 * x ≤ 1

-- State the theorem: If x is in the domain of f(1 - 2*x), then x is in [0, 1/2].
theorem domain_f_1_minus_2x_is_0_to_half :
  ∀ x : ℝ, domain_f_1_minus_2x x ↔ (0 ≤ x ∧ x ≤ 1 / 2) := by
  sorry

end domain_f_1_minus_2x_is_0_to_half_l126_126627


namespace part_a_l126_126677

theorem part_a (x : ℝ) : 1 + (1 / (2 + 1 / ((4 * x + 1) / (2 * x + 1) - 1 / (2 + 1 / x)))) = 19 / 14 ↔ x = 1 / 2 := sorry

end part_a_l126_126677


namespace original_number_is_64_l126_126429

theorem original_number_is_64 (x : ℕ) : 500 + x = 9 * x - 12 → x = 64 :=
by
  sorry

end original_number_is_64_l126_126429


namespace minimum_value_y_is_2_l126_126052

noncomputable def minimum_value_y (x : ℝ) : ℝ :=
  x + (1 / x)

theorem minimum_value_y_is_2 (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = minimum_value_y x → y ≥ 2) :=
by
  sorry

end minimum_value_y_is_2_l126_126052


namespace ratio_of_visible_spots_l126_126936

theorem ratio_of_visible_spots (S S1 : ℝ) (h1 : ∀ (fold_type : ℕ), 
  (fold_type = 1 ∨ fold_type = 2 ∨ fold_type = 3) → 
  (if fold_type = 1 ∨ fold_type = 2 then S1 else S) = S1) : S1 / S = 2 / 3 := 
sorry

end ratio_of_visible_spots_l126_126936


namespace angle_measure_l126_126979

-- Define the complement function
def complement (α : ℝ) : ℝ := 180 - α

-- Given condition
variable (α : ℝ)
variable (h : complement α = 120)

-- Theorem to prove
theorem angle_measure : α = 60 :=
by sorry

end angle_measure_l126_126979


namespace find_k_l126_126976

def vector := (ℝ × ℝ)

def a : vector := (3, 1)
def b : vector := (1, 3)
def c (k : ℝ) : vector := (k, 2)

def subtract (v1 v2 : vector) : vector :=
  (v1.1 - v2.1, v1.2 - v2.2)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_k (k : ℝ) (h : dot_product (subtract a (c k)) b = 0) : k = 0 := by
  sorry

end find_k_l126_126976


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l126_126834

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l126_126834


namespace employed_females_part_time_percentage_l126_126646

theorem employed_females_part_time_percentage (P : ℕ) (hP1 : 0 < P)
  (h1 : ∀ x : ℕ, x = P * 6 / 10) -- 60% of P are employed
  (h2 : ∀ e : ℕ, e = P * 6 / 10) -- e is the number of employed individuals
  (h3 : ∀ f : ℕ, f = e * 4 / 10) -- 40% of employed are females
  (h4 : ∀ pt : ℕ, pt = f * 6 / 10) -- 60% of employed females are part-time
  (h5 : ∀ m : ℕ, m = P * 48 / 100) -- 48% of P are employed males
  (h6 : e = f + m) -- Employed individuals are either males or females
  : f * 6 / f * 10 = 60 := sorry

end employed_females_part_time_percentage_l126_126646


namespace factorize_expression_l126_126460

theorem factorize_expression (a b x y : ℝ) : 
  a^2 * b * (x - y)^3 - a * b^2 * (y - x)^2 = ab * (x - y)^2 * (a * x - a * y - b) :=
by
  sorry

end factorize_expression_l126_126460


namespace star_4_3_l126_126977

def star (a b : ℤ) : ℤ := a^2 - a * b + b^2

theorem star_4_3 : star 4 3 = 13 :=
by
  sorry

end star_4_3_l126_126977


namespace radius_of_circle_l126_126651

-- Define the polar coordinates equation
def polar_circle (ρ θ : ℝ) : Prop := ρ = 6 * Real.cos θ

-- Define the conversion to Cartesian coordinates and the circle equation
def cartesian_circle (x y : ℝ) : Prop := (x - 3) ^ 2 + y ^ 2 = 9

-- Prove that given the polar coordinates equation, the radius of the circle is 3
theorem radius_of_circle : ∀ (ρ θ : ℝ), polar_circle ρ θ → ∃ r, r = 3 := by
  sorry

end radius_of_circle_l126_126651


namespace tan_identity_at_30_degrees_l126_126041

theorem tan_identity_at_30_degrees :
  let A := 30
  let B := 30
  let deg_to_rad := pi / 180
  let tan := fun x : ℝ => Real.tan (x * deg_to_rad)
  (1 + tan A) * (1 + tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_identity_at_30_degrees_l126_126041


namespace equation_of_line_AC_l126_126766

-- Definitions of points and lines
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_equation (A B C : ℝ) (P : Point) : Prop :=
  A * P.x + B * P.y + C = 0

-- Given points and lines
def B : Point := ⟨-2, 0⟩
def altitude_on_AB (P : Point) : Prop := line_equation 1 3 (-26) P

-- Required equation of line AB
def line_AB (P : Point) : Prop := line_equation 3 (-1) 6 P

-- Angle bisector given in the condition
def angle_bisector (P : Point) : Prop := line_equation 1 1 (-2) P

-- Derived Point A
def A : Point := ⟨-1, 3⟩

-- Symmetric point B' with respect to the angle bisector
def B' : Point := ⟨2, 4⟩

-- Required equation of line AC
def line_AC (P : Point) : Prop := line_equation 1 (-3) 10 P

-- The proof statement
theorem equation_of_line_AC :
  ∀ P : Point, (line_AB B ∧ angle_bisector A ∧ P = A → P = B' → line_AC P) :=
by
  intros P h h1 h2
  sorry

end equation_of_line_AC_l126_126766


namespace neil_final_num_3_l126_126366

-- Assuming the conditions in the problem as definitions
noncomputable def die_prob := (1/3 : ℚ)

noncomputable def prob_neil_final_3 (prob_jerry_roll_1 : ℚ) (prob_jerry_roll_2 : ℚ) (prob_jerry_roll_3 : ℚ) :=
  (prob_jerry_roll_1 * die_prob) + (prob_jerry_roll_2 * (1/2)) + (prob_jerry_roll_3 * 1)

theorem neil_final_num_3 :
  let prob_jerry_roll := die_prob in
  (prob_jerry_roll * die_prob) + (prob_jerry_roll * (1/2)) + (prob_jerry_roll * 1) = 11/18 :=
by
  let prob_jerry_roll := die_prob
  let partial_prob_1 := prob_jerry_roll * die_prob
  let partial_prob_2 := prob_jerry_roll * (1/2)
  let partial_prob_3 := prob_jerry_roll * 1
  let total_prob := partial_prob_1 + partial_prob_2 + partial_prob_3
  show total_prob = 11/18
  sorry

end neil_final_num_3_l126_126366


namespace g_at_5_eq_9_l126_126664

-- Define the polynomial function g as given in the conditions
def g (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 3

-- Define the hypothesis that g(-5) = -3
axiom g_neg5 (a b c : ℝ) : g a b c (-5) = -3

-- State the theorem to prove that g(5) = 9 given the conditions
theorem g_at_5_eq_9 (a b c : ℝ) : g a b c 5 = 9 := 
by sorry

end g_at_5_eq_9_l126_126664


namespace sum_gcf_lcm_eq_28_l126_126883

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l126_126883


namespace angle_bao_proof_l126_126068

noncomputable def angle_bao : ℝ := sorry -- angle BAO in degrees

theorem angle_bao_proof 
    (CD_is_diameter : true)
    (A_on_extension_DC_beyond_C : true)
    (E_on_semicircle : true)
    (B_is_intersection_AE_semicircle : B ≠ E)
    (AB_eq_OE : AB = OE)
    (angle_EOD_30_degrees : EOD = 30) : 
    angle_bao = 7.5 :=
sorry

end angle_bao_proof_l126_126068


namespace least_possible_z_minus_x_l126_126644

theorem least_possible_z_minus_x (x y z : ℤ) (h₁ : x < y) (h₂ : y < z) (h₃ : y - x > 11) 
  (h₄ : Even x) (h₅ : Odd y) (h₆ : Odd z) : z - x = 15 :=
sorry

end least_possible_z_minus_x_l126_126644


namespace inequality_always_true_l126_126761

theorem inequality_always_true 
  (a b : ℝ) 
  (h1 : ab > 0) : 
  (b / a) + (a / b) ≥ 2 := 
by sorry

end inequality_always_true_l126_126761


namespace fraction_blue_balls_l126_126850

theorem fraction_blue_balls (total_balls : ℕ) (red_fraction : ℚ) (other_balls : ℕ) (remaining_blue_fraction : ℚ) 
  (h1 : total_balls = 360) 
  (h2 : red_fraction = 1/4) 
  (h3 : other_balls = 216) 
  (h4 : remaining_blue_fraction = 1/5) :
  (total_balls - (total_balls / 4) - other_balls) = total_balls * (5 * red_fraction / 270) := 
by
  sorry

end fraction_blue_balls_l126_126850


namespace tunnel_connects_land_l126_126216

noncomputable def surface_area (planet : Type) : ℝ := sorry
noncomputable def land_area (planet : Type) : ℝ := sorry
noncomputable def half_surface_area (planet : Type) : ℝ := surface_area planet / 2
noncomputable def can_dig_tunnel_through_center (planet : Type) : Prop := sorry

variable {TauCeti : Type}

-- Condition: Land occupies more than half of the entire surface area.
axiom land_more_than_half : land_area TauCeti > half_surface_area TauCeti

-- Proof problem statement: Prove that inhabitants can dig a tunnel through the center of the planet.
theorem tunnel_connects_land : can_dig_tunnel_through_center TauCeti :=
sorry

end tunnel_connects_land_l126_126216


namespace point_of_tangency_is_correct_l126_126750

theorem point_of_tangency_is_correct : 
  (∃ (x y : ℝ), y = x^2 + 20 * x + 63 ∧ x = y^2 + 56 * y + 875 ∧ x = -19 / 2 ∧ y = -55 / 2) :=
by
  sorry

end point_of_tangency_is_correct_l126_126750


namespace Anne_is_15_pounds_heavier_l126_126281

def Anne_weight : ℕ := 67
def Douglas_weight : ℕ := 52

theorem Anne_is_15_pounds_heavier : Anne_weight - Douglas_weight = 15 := by
  sorry

end Anne_is_15_pounds_heavier_l126_126281


namespace convert_to_dms_l126_126922

-- Define the conversion factors
def degrees_to_minutes (d : ℝ) : ℝ := d * 60
def minutes_to_seconds (m : ℝ) : ℝ := m * 60

-- The main proof statement
theorem convert_to_dms (d : ℝ) :
  d = 24.29 →
  (24, 17, 24) = (24, degrees_to_minutes (0.29), minutes_to_seconds 0.4) :=
by
  sorry

end convert_to_dms_l126_126922


namespace students_going_on_field_trip_l126_126776

-- Define conditions
def van_capacity : Nat := 7
def number_of_vans : Nat := 6
def number_of_adults : Nat := 9

-- Define the total capacity
def total_people_capacity : Nat := number_of_vans * van_capacity

-- Define the number of students
def number_of_students : Nat := total_people_capacity - number_of_adults

-- Prove the number of students is 33
theorem students_going_on_field_trip : number_of_students = 33 := by
  sorry

end students_going_on_field_trip_l126_126776


namespace dave_paid_3_more_than_doug_l126_126579

theorem dave_paid_3_more_than_doug :
  let total_slices := 10
  let plain_pizza_cost := 10
  let anchovy_fee := 3
  let total_cost := plain_pizza_cost + anchovy_fee
  let cost_per_slice := total_cost / total_slices
  let slices_with_anchovies := total_slices / 3
  let dave_slices := slices_with_anchovies + 2
  let doug_slices := total_slices - dave_slices
  let doug_pay := doug_slices * plain_pizza_cost / total_slices
  let dave_pay := total_cost - doug_pay
  dave_pay - doug_pay = 3 :=
by
  sorry

end dave_paid_3_more_than_doug_l126_126579


namespace fraction_min_sum_l126_126079

theorem fraction_min_sum (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 45 * b < 110 * a ∧ 110 * a < 50 * b) :
  a = 3 ∧ b = 7 :=
sorry

end fraction_min_sum_l126_126079


namespace timPaid_l126_126861

def timPayment (p : Real) (d : Real) : Real :=
  p - p * d

theorem timPaid (p : Real) (d : Real) (a : Real) : 
  p = 1200 ∧ d = 0.15 → a = 1020 :=
by
  intro h
  cases h with hp hd
  rw [hp, hd]
  have hdiscount : 1200 * 0.15 = 180 := by norm_num
  have hpayment : 1200 - 180 = 1020 := by norm_num
  rw [hdiscount, hpayment]
  sorry

end timPaid_l126_126861


namespace minimum_value_of_quadratic_function_l126_126773

variable (p q : ℝ) (hp : 0 < p) (hq : 0 < q)

theorem minimum_value_of_quadratic_function : 
  (∃ x : ℝ, x = p) ∧ (∀ x : ℝ, (x^2 - 2 * p * x + 4 * q) ≥ (p^2 - 2 * p * p + 4 * q)) :=
sorry

end minimum_value_of_quadratic_function_l126_126773


namespace part_1_part_2_part_3_l126_126652

/-- Defining a structure to hold the values of x and y as given in the problem --/
structure PhoneFeeData (α : Type) :=
  (x : α) (y : α)

def problem_data : List (PhoneFeeData ℝ) :=
  [
    ⟨1, 18.4⟩, ⟨2, 18.8⟩, ⟨3, 19.2⟩, ⟨4, 19.6⟩, ⟨5, 20⟩, ⟨6, 20.4⟩
  ]

noncomputable def phone_fee_equation (x : ℝ) : ℝ := 0.4 * x + 18

theorem part_1 :
  ∀ data ∈ problem_data, phone_fee_equation data.x = data.y :=
by
  sorry

theorem part_2 : phone_fee_equation 10 = 22 :=
by
  sorry

theorem part_3 : ∀ x : ℝ, phone_fee_equation x = 26 → x = 20 :=
by
  sorry

end part_1_part_2_part_3_l126_126652


namespace gain_percentage_second_book_l126_126183

theorem gain_percentage_second_book (C1 C2 SP1 SP2 : ℝ) (H1 : C1 + C2 = 360) (H2 : C1 = 210) (H3 : SP1 = C1 - (15 / 100) * C1) (H4 : SP1 = SP2) (H5 : SP2 = C2 + (19 / 100) * C2) : 
  (19 : ℝ) = 19 := 
by
  sorry

end gain_percentage_second_book_l126_126183


namespace triangle_angle_not_less_than_60_l126_126700

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l126_126700


namespace sum_of_three_equal_expressions_l126_126362

-- Definitions of variables and conditions
variables (a b c d e f g h i S : ℤ)
variable (ha : a = 4)
variable (hg : g = 13)
variable (hh : h = 6)
variable (heq1 : a + b + c + d = S)
variable (heq2 : d + e + f + g = S)
variable (heq3 : g + h + i = S)

-- Main statement we want to prove
theorem sum_of_three_equal_expressions : S = 19 + i :=
by
  -- substitution steps and equality reasoning would be carried out here
  sorry

end sum_of_three_equal_expressions_l126_126362


namespace rescue_team_assignment_count_l126_126226

def num_rescue_teams : ℕ := 6
def sites : Set String := {"A", "B", "C"}
def min_teams_at_A : ℕ := 2
def min_teams_per_site : ℕ := 1

theorem rescue_team_assignment_count : 
  ∃ (allocation : sites → ℕ), 
    (allocation "A" ≥ min_teams_at_A) ∧ 
    (∀ site ∈ sites, allocation site ≥ min_teams_per_site) ∧ 
    (∑ site in sites, allocation site = num_rescue_teams) ∧ 
    (nat.factorial num_rescue_teams / 
    (∏ site in sites, nat.factorial (allocation site))) = 360 :=
sorry

end rescue_team_assignment_count_l126_126226


namespace limit_sequence_l126_126945

open Filter
open Real

noncomputable def sequence_limit := 
  filter.tendsto (λ n : ℕ, (sqrt (3 * n - 1) - real.cbrt (125 * n^3 + n)) / (real.rpow n (1 / 5) - n)) at_top (nhds 5)

theorem limit_sequence: sequence_limit :=
  sorry

end limit_sequence_l126_126945


namespace incorrect_expression_l126_126910

theorem incorrect_expression :
  ¬((|(-5 : ℤ)|)^2 = 5) :=
by
sorry

end incorrect_expression_l126_126910


namespace math_problem_equivalence_l126_126920

section

variable (x y z : ℝ) (w : String)

theorem math_problem_equivalence (h₀ : x / 15 = 4 / 5) (h₁ : y = 80) (h₂ : z = 0.8) (h₃ : w = "八折"):
  x = 12 ∧ y = 80 ∧ z = 0.8 ∧ w = "八折" :=
by
  sorry

end

end math_problem_equivalence_l126_126920


namespace factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l126_126957

-- Proof 1: Factorize 3m^2 n - 12mn + 12n
theorem factor_3m2n_12mn_12n (m n : ℤ) : 3 * m^2 * n - 12 * m * n + 12 * n = 3 * n * (m - 2)^2 :=
by sorry

-- Proof 2: Factorize (a-b)x^2 + 4y^2(b-a)
theorem factor_abx2_4y2ba (a b x y : ℤ) : (a - b) * x^2 + 4 * y^2 * (b - a) = (a - b) * (x + 2 * y) * (x - 2 * y) :=
by sorry

-- Proof 3: Calculate 2023 * 51^2 - 2023 * 49^2
theorem calculate_result : 2023 * 51^2 - 2023 * 49^2 = 404600 :=
by sorry

end factor_3m2n_12mn_12n_factor_abx2_4y2ba_calculate_result_l126_126957


namespace calculate_percentage_l126_126925

/-- A candidate got a certain percentage of the votes polled and he lost to his rival by 2000 votes.
There were 10,000.000000000002 votes cast. What percentage of the votes did the candidate get? --/

def candidate_vote_percentage (P : ℝ) (total_votes : ℝ) (rival_margin : ℝ) : Prop :=
  (P / 100 * total_votes = total_votes - rival_margin) → P = 80

theorem calculate_percentage:
  candidate_vote_percentage P 10000.000000000002 2000 := 
by 
  sorry

end calculate_percentage_l126_126925


namespace sum_gcd_lcm_eight_twelve_l126_126876

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l126_126876


namespace organizingCommitteeWays_l126_126983

-- Define the problem context
def numberOfTeams : Nat := 5
def membersPerTeam : Nat := 8
def hostTeamSelection : Nat := 4
def otherTeamsSelection : Nat := 2

-- Define binomial coefficient
def binom (n k : Nat) : Nat := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of ways to select committee members
def totalCommitteeWays : Nat := numberOfTeams * 
                                 (binom membersPerTeam hostTeamSelection) * 
                                 ((binom membersPerTeam otherTeamsSelection) ^ (numberOfTeams - 1))

-- The theorem to prove
theorem organizingCommitteeWays : 
  totalCommitteeWays = 215134600 := 
    sorry

end organizingCommitteeWays_l126_126983


namespace distinct_prime_factors_of_90_l126_126791

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l126_126791


namespace length_of_rectangular_sheet_l126_126866

/-- The length of each rectangular sheet is 10 cm given that:
    1. Two identical rectangular sheets each have an area of 48 square centimeters,
    2. The covered area when overlapping the sheets is 72 square centimeters,
    3. The diagonal BD of the overlapping quadrilateral ABCD is 6 centimeters. -/
theorem length_of_rectangular_sheet :
  ∀ (length width : ℝ),
    width * length = 48 ∧
    2 * 48 - 72 = width * 6 ∧
    width * 6 = 24 →
    length = 10 :=
sorry

end length_of_rectangular_sheet_l126_126866


namespace sum_of_ages_l126_126199

-- Definitions for conditions
def age_product (a b c : ℕ) : Prop := a * b * c = 72
def younger_than_10 (k : ℕ) : Prop := k < 10

-- Main statement
theorem sum_of_ages (a b k : ℕ) (h_product : age_product a b k) (h_twin : a = b) (h_kiana : younger_than_10 k) : 
  a + b + k = 14 := sorry

end sum_of_ages_l126_126199


namespace sum_gcf_lcm_l126_126907

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l126_126907


namespace sum_gcf_lcm_eq_28_l126_126882

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l126_126882


namespace factor_3x2_minus_3y2_l126_126467

theorem factor_3x2_minus_3y2 (x y : ℝ) : 3 * x^2 - 3 * y^2 = 3 * (x + y) * (x - y) := by
  sorry

end factor_3x2_minus_3y2_l126_126467


namespace solve_system_l126_126678

theorem solve_system :
  ∃ x y : ℝ, (x + 2*y = 1 ∧ 3*x - 2*y = 7) → (x = 2 ∧ y = -1/2) :=
by
  sorry

end solve_system_l126_126678


namespace perfect_squares_solutions_l126_126560

noncomputable def isPerfectSquare (n : ℤ) : Prop :=
  ∃ m : ℤ, m * m = n

theorem perfect_squares_solutions :
  ∀ (a b : ℕ),
    0 < a → 0 < b →
    (isPerfectSquare (↑a * ↑a - 4 * ↑b)) →
    (isPerfectSquare (↑b * ↑b - 4 * ↑a)) →
      (a = 4 ∧ b = 4) ∨
      (a = 5 ∧ b = 6) ∨
      (a = 6 ∧ b = 5) :=
by
  -- Proof omitted
  sorry

end perfect_squares_solutions_l126_126560


namespace problem1_problem2_l126_126287

variables (a b : ℝ)

theorem problem1 : ((a^2)^3 / (-a)^2) = a^4 :=
sorry

theorem problem2 : ((a + 2 * b) * (a + b) - 3 * a * (a + b)) = -2 * a^2 + 2 * b^2 :=
sorry

end problem1_problem2_l126_126287


namespace geometric_series_abs_sum_range_l126_126970

open Function

variable {α : Type*} [LinearOrderedField α]

-- Definition of geometric series sum convergence
def geometric_series_sum (a q : α) (h : |q| < 1) : α :=
  a / (1 - q)

-- The problem statement
theorem geometric_series_abs_sum_range (a : α) (q : α) (h : |q| < 1) 
  (h_sum : geometric_series_sum a q (abs_lt.mpr h) = -2) : 
  (∑' n : ℕ, |a * q^n|) ∈ set.Ici (2 : α) :=
by
  sorry

end geometric_series_abs_sum_range_l126_126970


namespace community_theater_ticket_sales_l126_126117

theorem community_theater_ticket_sales (A C : ℕ) 
  (h1 : 12 * A + 4 * C = 840) 
  (h2 : A + C = 130) :
  A = 40 :=
sorry

end community_theater_ticket_sales_l126_126117


namespace log_increasing_a_gt_one_l126_126631

noncomputable def log (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem log_increasing_a_gt_one (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) (h₂ : log a 2 < log a 3) : a > 1 :=
by
  sorry

end log_increasing_a_gt_one_l126_126631


namespace sum_gcf_lcm_l126_126908

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l126_126908


namespace division_result_l126_126541

theorem division_result (a b : ℕ) (ha : a = 7) (hb : b = 3) :
    ((a^3 + b^3) / (a^2 - a * b + b^2) = 10) := 
by
  sorry

end division_result_l126_126541


namespace initial_members_in_family_c_l126_126255

theorem initial_members_in_family_c 
  (a b d e f : ℕ)
  (ha : a = 7)
  (hb : b = 8)
  (hd : d = 13)
  (he : e = 6)
  (hf : f = 10)
  (average_after_moving : (a - 1) + (b - 1) + (d - 1) + (e - 1) + (f - 1) + (x : ℕ) - 1 = 48) :
  x = 10 := by
  sorry

end initial_members_in_family_c_l126_126255


namespace number_of_possible_routes_l126_126496

def f (x y : ℕ) : ℕ :=
  if y = 2 then sorry else sorry -- Here you need the exact definition of f(x, y)

theorem number_of_possible_routes (n : ℕ) (h : n > 0) : 
  f n 2 = (1 / 2 : ℚ) * (n^2 + 3 * n + 2) := 
by 
  sorry

end number_of_possible_routes_l126_126496


namespace max_kings_on_chessboard_l126_126545

theorem max_kings_on_chessboard : 
  ∀ (board : matrix (fin 12) (fin 12) bool), 
  (∀ i j, board i j = tt → (∃ k l, (abs ((i : int) - k) ≤ 1 ∧ abs ((j : int) - l) ≤ 1) 
  ∧ board k l = tt ∧ (i ≠ k ∨ j ≠ l))) → 
  ∃ (S : finset (fin 12 × fin 12)), S.card = 56 ∧ 
  (∀ (i j) (h1 : (i, j) ∈ S) (h2 : (i', j') ∈ S), (abs ((i : int) - (i' : int)) ≤ 1 ∧ abs ((j : int) - (j' : int)) ≤ 1) 
  → ((i, j) ≠ (i', j')) → (abs ((i : int) - (i' : int)) = 1 ∧ abs ((j : int) - (j' : int)) = 0) 
  ∨ (abs ((i : int) - (i' : int)) = 0 ∧ abs ((j : int) - (j' : int)) = 1))) :=
by
  sorry

end max_kings_on_chessboard_l126_126545


namespace least_number_1056_div_26_l126_126692

/-- Define the given values and the divisibility condition -/
def least_number_to_add (n : ℕ) (d : ℕ) : ℕ :=
  let remainder := n % d
  d - remainder

/-- State the theorem to prove that the least number to add to 1056 to make it divisible by 26 is 10. -/
theorem least_number_1056_div_26 : least_number_to_add 1056 26 = 10 :=
by
  sorry -- Proof is omitted as per the instruction

end least_number_1056_div_26_l126_126692


namespace one_twentieth_of_eighty_l126_126490

/--
Given the conditions, to prove that \(\frac{1}{20}\) of 80 is equal to 4.
-/
theorem one_twentieth_of_eighty : (80 : ℚ) * (1 / 20) = 4 :=
by
  sorry

end one_twentieth_of_eighty_l126_126490


namespace LeanProof_l126_126815

noncomputable def ProblemStatement : Prop :=
  let AB_parallel_YZ := True -- given condition that AB is parallel to YZ
  let AZ := 36 
  let BQ := 15
  let QY := 20
  let similarity_ratio := BQ / QY = 3 / 4
  ∃ QZ : ℝ, AZ = (3 / 4) * QZ + QZ ∧ QZ = 144 / 7

theorem LeanProof : ProblemStatement :=
sorry

end LeanProof_l126_126815


namespace square_perimeter_from_area_l126_126270

def square_area (s : ℝ) : ℝ := s * s -- Definition of the area of a square based on its side length.
def square_perimeter (s : ℝ) : ℝ := 4 * s -- Definition of the perimeter of a square based on its side length.

theorem square_perimeter_from_area (s : ℝ) (h : square_area s = 900) : square_perimeter s = 120 :=
by {
  sorry -- Placeholder for the proof.
}

end square_perimeter_from_area_l126_126270


namespace solve_for_x_l126_126157

noncomputable def simplified_end_expr (x : ℝ) := x = 4 - Real.sqrt 7 
noncomputable def expressed_as_2_statement (x : ℝ) := (x ^ 2 - 4 * x + 5) = (4 * (x - 1))
noncomputable def domain_condition (x : ℝ) := (-5 < x) ∧ (x < 3)

theorem solve_for_x (x : ℝ) :
  domain_condition x →
  (expressed_as_2_statement x ↔ simplified_end_expr x) :=
by
  sorry

end solve_for_x_l126_126157


namespace number_of_true_false_questions_is_six_l126_126097

variable (x : ℕ)
variable (num_true_false num_free_response num_multiple_choice total_problems : ℕ)

axiom problem_conditions :
  (num_free_response = x + 7) ∧ 
  (num_multiple_choice = 2 * (x + 7)) ∧ 
  (total_problems = 45) ∧ 
  (total_problems = x + num_free_response + num_multiple_choice)

theorem number_of_true_false_questions_is_six (h : problem_conditions x num_true_false num_free_response num_multiple_choice total_problems) : 
  x = 6 :=
  sorry

end number_of_true_false_questions_is_six_l126_126097


namespace find_x_values_l126_126016

noncomputable def condition (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4

theorem find_x_values : 
  {x : ℝ | condition  x} = {x : ℝ | x < -3} ∪ {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end find_x_values_l126_126016


namespace arithmetic_sequence_root_arithmetic_l126_126650

theorem arithmetic_sequence_root_arithmetic (a : ℕ → ℝ) 
  (h_arith : ∀ n : ℕ, a (n+1) - a n = a 1 - a 0) 
  (h_root : ∀ x : ℝ, x^2 + 12 * x - 8 = 0 → (x = a 2 ∨ x = a 10)) : 
  a 6 = -6 := 
by
  -- We skip the proof as per instructions
  sorry

end arithmetic_sequence_root_arithmetic_l126_126650


namespace find_rate_percent_l126_126417

-- Define the conditions
def principal : ℝ := 1200
def time : ℝ := 4
def simple_interest : ℝ := 400

-- Define the rate that we need to prove
def rate : ℝ := 8.3333  -- approximately

-- Formalize the proof problem in Lean 4
theorem find_rate_percent
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = principal) (hT : T = time) (hSI : SI = simple_interest) :
  SI = (P * R * T) / 100 → R = rate :=
by
  intros h
  sorry

end find_rate_percent_l126_126417


namespace sum_gcf_lcm_l126_126905

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l126_126905


namespace smallest_n_l126_126250

theorem smallest_n (n : ℕ) (h : 503 * n % 48 = 1019 * n % 48) : n = 4 := by
  sorry

end smallest_n_l126_126250


namespace people_per_table_l126_126935

theorem people_per_table (initial_customers left_customers tables remaining_customers : ℕ) 
  (h1 : initial_customers = 21) 
  (h2 : left_customers = 12) 
  (h3 : tables = 3) 
  (h4 : remaining_customers = initial_customers - left_customers) 
  : remaining_customers / tables = 3 :=
by
  sorry

end people_per_table_l126_126935


namespace breadth_of_rectangular_plot_l126_126098

theorem breadth_of_rectangular_plot (b : ℝ) (A : ℝ) (l : ℝ)
  (h1 : A = 20 * b)
  (h2 : l = b + 10)
  (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l126_126098


namespace balloons_per_school_l126_126241

theorem balloons_per_school (yellow black total : ℕ) 
  (hyellow : yellow = 3414)
  (hblack : black = yellow + 1762)
  (htotal : total = yellow + black)
  (hdivide : total % 10 = 0) : 
  total / 10 = 859 :=
by sorry

end balloons_per_school_l126_126241


namespace line_intersection_y_axis_l126_126730

theorem line_intersection_y_axis :
  let p1 := (2, 8)
      p2 := (4, 14)
      m := (p2.2 - p1.2) / (p2.1 - p1.1)  -- slope calculation
      b := p1.2 - m * p1.1  -- y-intercept calculation
  in (b = 2) → (m = 3) → (p1 ≠ p2) → 
    (0, b) = @ y-intercept of the line passing through p1 and p2 :=
by
  intros p1 p2 m b h1 h2 h3
  -- placeholder for actual proof
  sorry

end line_intersection_y_axis_l126_126730


namespace rational_squares_solution_l126_126165

theorem rational_squares_solution {x y u v : ℕ} (x_pos : 0 < x) (y_pos : 0 < y) (u_pos : 0 < u) (v_pos : 0 < v) 
  (h1 : ∃ q : ℚ, q = (Real.sqrt (x * y) + Real.sqrt (u * v))) 
  (h2 : |(x / 9 : ℚ) - (y / 4 : ℚ)| = |(u / 3 : ℚ) - (v / 12 : ℚ)| ∧ |(u / 3 : ℚ) - (v / 12 : ℚ)| = u * v - x * y) :
  ∃ k : ℕ, x = 9 * k ∧ y = 4 * k ∧ u = 3 * k ∧ v = 12 * k := by
  sorry

end rational_squares_solution_l126_126165


namespace find_k_l126_126967

-- Define the arithmetic sequence and the sum of the first n terms
def a (n : ℕ) : ℤ := 2 * n + 2
def S (n : ℕ) : ℤ := n^2 + 3 * n

-- The main assertion
theorem find_k : ∃ (k : ℕ), k > 0 ∧ (S k - a (k + 5) = 44) ∧ k = 7 :=
by
  sorry

end find_k_l126_126967


namespace geometric_progression_ineq_l126_126471

variable (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ)

-- Condition: \(b_n\) is an increasing positive geometric progression
-- \( q > 1 \) because the progression is increasing
variable (q_pos : q > 1) 

-- Recursive definitions for the geometric progression
variable (geom_b₂ : b₂ = b₁ * q)
variable (geom_b₃ : b₃ = b₁ * q^2)
variable (geom_b₄ : b₄ = b₁ * q^3)
variable (geom_b₅ : b₅ = b₁ * q^4)
variable (geom_b₆ : b₆ = b₁ * q^5)

-- Given condition from the problem
variable (condition : b₄ + b₃ - b₂ - b₁ = 5)

-- Statement to prove
theorem geometric_progression_ineq (q b₁ b₂ b₃ b₄ b₅ b₆ : ℝ) 
  (q_pos : q > 1) 
  (geom_b₂ : b₂ = b₁ * q)
  (geom_b₃ : b₃ = b₁ * q^2)
  (geom_b₄ : b₄ = b₁ * q^3)
  (geom_b₅ : b₅ = b₁ * q^4)
  (geom_b₆ : b₆ = b₁ * q^5)
  (condition : b₃ + b₄ - b₂ - b₁ = 5) : b₆ + b₅ ≥ 20 := by
    sorry

end geometric_progression_ineq_l126_126471


namespace number_of_distinct_prime_factors_of_90_l126_126788

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l126_126788


namespace solution_l126_126802

theorem solution (x y : ℝ) (h1 : x * y = 6) (h2 : x^2 * y + x * y^2 + x + y = 63) : x^2 + y^2 = 69 := 
by 
  -- Insert proof here
  sorry

end solution_l126_126802


namespace smallest_N_triangle_ineq_l126_126462

theorem smallest_N_triangle_ineq (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c)
  (h₄ : c < a + b) : (a^2 + b^2 + a * b) / c^2 < 1 := 
sorry

end smallest_N_triangle_ineq_l126_126462


namespace investment_plans_count_l126_126573

theorem investment_plans_count :
  ∃ (plans : ℕ), plans = 60 ∧
    ∀ (projects : Finset ℕ) (cities : Finset ℕ), 
      projects.card = 3 ∧ cities.card = 4 →
      (∀ city ∈ cities, projects.count city ≤ 2) → 
      plans = 60 :=
by
  sorry

end investment_plans_count_l126_126573


namespace shaded_region_area_l126_126268

-- Definitions based on given conditions
def num_squares : ℕ := 25
def diagonal_length : ℝ := 10
def squares_in_large_square : ℕ := 16

-- The area of the entire shaded region
def area_of_shaded_region : ℝ := 78.125

-- Theorem to prove 
theorem shaded_region_area 
  (num_squares : ℕ) 
  (diagonal_length : ℝ) 
  (squares_in_large_square : ℕ) : 
  (num_squares = 25) → 
  (diagonal_length = 10) → 
  (squares_in_large_square = 16) → 
  area_of_shaded_region = 78.125 := 
by {
  sorry -- proof to be filled
}

end shaded_region_area_l126_126268


namespace gcd_891_810_l126_126105

theorem gcd_891_810 : Nat.gcd 891 810 = 81 := 
by
  sorry

end gcd_891_810_l126_126105


namespace problem_statement_l126_126087

variable {S R p a b c : ℝ}
variable (τ τ_a τ_b τ_c : ℝ)

theorem problem_statement
  (h1: S = τ * p)
  (h2: S = τ_a * (p - a))
  (h3: S = τ_b * (p - b))
  (h4: S = τ_c * (p - c))
  (h5: τ = S / p)
  (h6: τ_a = S / (p - a))
  (h7: τ_b = S / (p - b))
  (h8: τ_c = S / (p - c))
  (h9: abc / S = 4 * R) :
  1 / τ^3 - 1 / τ_a^3 - 1 / τ_b^3 - 1 / τ_c^3 = 12 * R / S^2 :=
  sorry

end problem_statement_l126_126087


namespace correct_diagram_is_B_l126_126601

-- Define the diagrams and their respected angles
def sector_angle_A : ℝ := 90
def sector_angle_B : ℝ := 135
def sector_angle_C : ℝ := 180

-- Define the target central angle for one third of the circle
def target_angle : ℝ := 120

-- The proof statement that Diagram B is the correct diagram with the sector angle closest to one third of the circle (120 degrees)
theorem correct_diagram_is_B (A B C : Prop) :
  (B = (sector_angle_A < target_angle ∧ target_angle < sector_angle_B)) := 
sorry

end correct_diagram_is_B_l126_126601


namespace percentage_given_away_l126_126004

theorem percentage_given_away
  (initial_bottles : ℕ)
  (drank_percentage : ℝ)
  (remaining_percentage : ℝ)
  (gave_away : ℝ):
  initial_bottles = 3 →
  drank_percentage = 0.90 →
  remaining_percentage = 0.70 →
  gave_away = initial_bottles - (drank_percentage * 1 + remaining_percentage) →
  (gave_away / 2) / 1 * 100 = 70 :=
by
  intros
  sorry

end percentage_given_away_l126_126004


namespace evaluate_expression_l126_126011

theorem evaluate_expression (b : ℕ) (h : b = 4) : (b ^ b - b * (b - 1) ^ b) ^ b = 21381376 := by
  sorry

end evaluate_expression_l126_126011


namespace max_ratio_of_odd_integers_is_nine_l126_126825

-- Define odd positive integers x and y whose mean is 55
def is_odd (n : ℕ) : Prop := n % 2 = 1
def is_positive (n : ℕ) : Prop := 0 < n
def mean_is_55 (x y : ℕ) : Prop := (x + y) / 2 = 55

-- The problem statement
theorem max_ratio_of_odd_integers_is_nine (x y : ℕ) 
  (hx : is_positive x) (hy : is_positive y)
  (ox : is_odd x) (oy : is_odd y)
  (mean : mean_is_55 x y) : 
  ∀ r, r = (x / y : ℚ) → r ≤ 9 :=
by
  sorry

end max_ratio_of_odd_integers_is_nine_l126_126825


namespace n_squared_divisible_by_144_l126_126805

theorem n_squared_divisible_by_144 (n : ℕ) (h1 : 0 < n) (h2 : ∃ t : ℕ, t = 12 ∧ ∀ d : ℕ, d ∣ n → d ≤ t) : 144 ∣ n^2 :=
sorry

end n_squared_divisible_by_144_l126_126805


namespace total_amount_in_account_after_two_years_l126_126680

-- Initial definitions based on conditions in the problem
def initial_investment : ℝ := 76800
def annual_interest_rate : ℝ := 0.125
def annual_contribution : ℝ := 5000

-- Function to calculate amount after n years with annual contributions
def total_amount_after_years (P : ℝ) (r : ℝ) (A : ℝ) (n : ℕ) : ℝ :=
  let rec helper (P : ℝ) (n : ℕ) :=
    if n = 0 then P
    else 
      let previous_amount := helper P (n - 1)
      (previous_amount * (1 + r) + A)
  helper P n

-- Theorem to prove the final total amount after 2 years
theorem total_amount_in_account_after_two_years :
  total_amount_after_years initial_investment annual_interest_rate annual_contribution 2 = 107825 :=
  by 
  -- proof goes here
  sorry

end total_amount_in_account_after_two_years_l126_126680


namespace div_pow_eq_l126_126001

theorem div_pow_eq : 23^11 / 23^5 = 148035889 := by
  sorry

end div_pow_eq_l126_126001


namespace number_of_distinct_prime_factors_90_l126_126794

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l126_126794


namespace q_zero_l126_126666

noncomputable def q (x : ℝ) : ℝ := sorry -- Definition of the polynomial q(x) is required here.

theorem q_zero : 
  (∀ n : ℕ, n ≤ 7 → q (3^n) = 1 / 3^n) →
  q 0 = 0 :=
by 
  sorry

end q_zero_l126_126666


namespace solve_medium_apple_cost_l126_126449

def cost_small_apple : ℝ := 1.5
def cost_big_apple : ℝ := 3.0
def num_small_apples : ℕ := 6
def num_medium_apples : ℕ := 6
def num_big_apples : ℕ := 8
def total_cost : ℝ := 45

noncomputable def cost_medium_apple (M : ℝ) : Prop :=
  (6 * cost_small_apple) + (6 * M) + (8 * cost_big_apple) = total_cost

theorem solve_medium_apple_cost : ∃ M : ℝ, cost_medium_apple M ∧ M = 2 := by
  sorry

end solve_medium_apple_cost_l126_126449


namespace machine_work_hours_l126_126374

theorem machine_work_hours (A B : ℝ) (x : ℝ) (hA : A = 1 / 8) (hB : B = A / 4)
  (hB_rate : B = 1 / 32) (B_time : B * 8 = 1 - x / 8) : x = 6 :=
by
  sorry

end machine_work_hours_l126_126374


namespace arithmetic_mean_of_fractions_l126_126248

theorem arithmetic_mean_of_fractions :
  (3 / 8 + 5 / 9 + 7 / 12) / 3 = 109 / 216 :=
by
  sorry

end arithmetic_mean_of_fractions_l126_126248


namespace mauve_red_paint_parts_l126_126587

noncomputable def parts_of_red_in_mauve : ℕ :=
let fuchsia_red_ratio := 5
let fuchsia_blue_ratio := 3
let total_fuchsia := 16
let added_blue := 14
let mauve_blue_ratio := 6

let total_fuchsia_parts := fuchsia_red_ratio + fuchsia_blue_ratio
let red_in_fuchsia := (fuchsia_red_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_fuchsia := (fuchsia_blue_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_mauve := blue_in_fuchsia + added_blue
let ratio_red_to_blue_in_mauve := red_in_fuchsia / blue_in_mauve
ratio_red_to_blue_in_mauve * mauve_blue_ratio

theorem mauve_red_paint_parts : parts_of_red_in_mauve = 3 :=
by sorry

end mauve_red_paint_parts_l126_126587


namespace daniel_age_l126_126003

def isAgeSet (s : Set ℕ) : Prop :=
  s = {4, 6, 8, 10, 12, 14}

def sumTo18 (s : Set ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = 18 ∧ a ≠ b

def youngerThan11 (s : Set ℕ) : Prop :=
  ∀ (a : ℕ), a ∈ s → a < 11

def staysHome (DanielAge : ℕ) (s : Set ℕ) : Prop :=
  6 ∈ s ∧ DanielAge ∈ s

theorem daniel_age :
  ∀ (ages : Set ℕ) (DanielAge : ℕ),
    isAgeSet ages →
    (∃ s, sumTo18 s ∧ s ⊆ ages) →
    (∃ s, youngerThan11 s ∧ s ⊆ ages ∧ 6 ∉ s) →
    staysHome DanielAge ages →
    DanielAge = 12 :=
by
  intros ages DanielAge isAgeSetAges sumTo18Ages youngerThan11Ages staysHomeDaniel
  sorry

end daniel_age_l126_126003


namespace circle_y_coords_sum_l126_126291

theorem circle_y_coords_sum (x y : ℝ) (hc : (x + 3)^2 + (y - 5)^2 = 64) (hx : x = 0) : y = 5 + Real.sqrt 55 ∨ y = 5 - Real.sqrt 55 → (5 + Real.sqrt 55) + (5 - Real.sqrt 55) = 10 := 
by
  intros
  sorry

end circle_y_coords_sum_l126_126291


namespace b5_plus_b9_l126_126028

variable {a : ℕ → ℕ} -- Geometric sequence
variable {b : ℕ → ℕ} -- Arithmetic sequence

axiom geom_progression {r x y : ℕ} : a x = a 1 * r^(x - 1) ∧ a y = a 1 * r^(y - 1)
axiom arith_progression {d x y : ℕ} : b x = b 1 + d * (x - 1) ∧ b y = b 1 + d * (y - 1)

axiom a3a11_equals_4a7 : a 3 * a 11 = 4 * a 7
axiom a7_equals_b7 : a 7 = b 7

theorem b5_plus_b9 : b 5 + b 9 = 8 := by
  apply sorry

end b5_plus_b9_l126_126028


namespace min_formula_l126_126260

theorem min_formula (a b : ℝ) : 
  min a b = (a + b - Real.sqrt ((a - b) ^ 2)) / 2 :=
by
  sorry

end min_formula_l126_126260


namespace remainder_1234567_div_256_l126_126438

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l126_126438


namespace exists_100_integers_with_distinct_pairwise_sums_l126_126817

-- Define number of integers and the constraint limit
def num_integers : ℕ := 100
def max_value : ℕ := 25000

-- Define the predicate for all pairwise sums being different
def pairwise_different_sums (as : Fin num_integers → ℕ) : Prop :=
  ∀ i j k l : Fin num_integers, i ≠ j ∧ k ≠ l → as i + as j ≠ as k + as l

-- Main theorem statement
theorem exists_100_integers_with_distinct_pairwise_sums :
  ∃ as : Fin num_integers → ℕ, (∀ i : Fin num_integers, as i > 0 ∧ as i ≤ max_value) ∧ pairwise_different_sums as :=
sorry

end exists_100_integers_with_distinct_pairwise_sums_l126_126817


namespace f_at_2_l126_126033

noncomputable def f (x : ℝ) (a b : ℝ) := a * Real.log x + b / x + x
noncomputable def g (x : ℝ) (a b : ℝ) := (a / x) - (b / (x ^ 2)) + 1

theorem f_at_2 (a b : ℝ) (ha : g 1 a b = 0) (hb : g 3 a b = 0) : f 2 a b = 1 / 2 - 4 * Real.log 2 :=
by
  sorry

end f_at_2_l126_126033


namespace balls_balance_l126_126868

theorem balls_balance (G Y W B : ℕ) (h1 : G = 2 * B) (h2 : Y = 5 * B / 2) (h3 : W = 3 * B / 2) :
  5 * G + 3 * Y + 3 * W = 22 * B :=
by
  sorry

end balls_balance_l126_126868


namespace sales_discount_l126_126714

theorem sales_discount
  (P N : ℝ)  -- original price and number of items sold
  (H1 : (1 - D / 100) * 1.3 = 1.17) -- condition when discount D is applied
  (D : ℝ)  -- sales discount percentage
  : D = 10 := by
  sorry

end sales_discount_l126_126714


namespace line_through_two_points_l126_126103

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) = (1, 3) ∨ (x, y) = (3, 7) → y = m * x + b) ∧ (m + b = 3) := by
{ sorry }

end line_through_two_points_l126_126103


namespace contrapositive_l126_126482

variable (k : ℝ)

theorem contrapositive (h : ¬∃ x : ℝ, x^2 - x - k = 0) : k ≤ 0 :=
sorry

end contrapositive_l126_126482


namespace correct_statements_l126_126181
noncomputable def is_pythagorean_triplet (a b c : ℕ) : Prop := a^2 + b^2 = c^2

theorem correct_statements {a b c : ℕ} (h1 : is_pythagorean_triplet a b c) (h2 : a^2 + b^2 = c^2) :
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → a^2 + b^2 = c^2)) ∧
  (∀ (a b c : ℕ), (is_pythagorean_triplet a b c → is_pythagorean_triplet (2 * a) (2 * b) (2 * c))) :=
by sorry

end correct_statements_l126_126181


namespace value_subtracted_from_result_l126_126138

theorem value_subtracted_from_result (N V : ℕ) (hN : N = 1152) (h: (N / 6) - V = 3) : V = 189 :=
by
  sorry

end value_subtracted_from_result_l126_126138


namespace spadesuit_eval_l126_126325

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_eval : spadesuit 2 (spadesuit 3 (spadesuit 1 2)) = 4 := 
by
  sorry

end spadesuit_eval_l126_126325


namespace daily_wage_of_man_l126_126570

-- Define the wages for men and women
variables (M W : ℝ)

-- Define the conditions given in the problem
def condition1 : Prop := 24 * M + 16 * W = 11600
def condition2 : Prop := 12 * M + 37 * W = 11600

-- The theorem we want to prove
theorem daily_wage_of_man (h1 : condition1 M W) (h2 : condition2 M W) : M = 350 :=
by
  sorry

end daily_wage_of_man_l126_126570


namespace find_k_inverse_proportion_l126_126526

theorem find_k_inverse_proportion :
  ∃ k : ℝ, k ≠ 0 ∧ (∀ x : ℝ, ∀ y : ℝ, (x = 1 ∧ y = 3) → (y = k / x)) ∧ k = 3 :=
by
  sorry

end find_k_inverse_proportion_l126_126526


namespace product_of_cubes_l126_126002

theorem product_of_cubes :
  ( (2^3 - 1) / (2^3 + 1) * (3^3 - 1) / (3^3 + 1) * (4^3 - 1) / (4^3 + 1) * 
    (5^3 - 1) / (5^3 + 1) * (6^3 - 1) / (6^3 + 1) * (7^3 - 1) / (7^3 + 1) 
  ) = 57 / 72 := 
by
  sorry

end product_of_cubes_l126_126002


namespace companyA_sold_bottles_l126_126539

-- Let CompanyA and CompanyB be the prices per bottle for the respective companies
def CompanyA_price : ℝ := 4
def CompanyB_price : ℝ := 3.5

-- Company B sold 350 bottles
def CompanyB_bottles : ℕ := 350

-- Total revenue of Company B
def CompanyB_revenue : ℝ := CompanyB_price * CompanyB_bottles

-- Additional condition that the revenue difference is $25
def revenue_difference : ℝ := 25

-- Define the total revenue equations for both scenarios
def revenue_scenario1 (x : ℕ) : Prop :=
  CompanyA_price * x = CompanyB_revenue + revenue_difference

def revenue_scenario2 (x : ℕ) : Prop :=
  CompanyA_price * x + revenue_difference = CompanyB_revenue

-- The problem translates to finding x such that either of these conditions hold
theorem companyA_sold_bottles : ∃ x : ℕ, revenue_scenario2 x ∧ x = 300 :=
by
  sorry

end companyA_sold_bottles_l126_126539


namespace alexa_emily_profit_l126_126144

def lemonade_stand_profit : ℕ :=
  let total_expenses := 10 + 5 + 3
  let price_per_cup := 4
  let cups_sold := 21
  let total_revenue := price_per_cup * cups_sold
  total_revenue - total_expenses

theorem alexa_emily_profit : lemonade_stand_profit = 66 :=
  by
  sorry

end alexa_emily_profit_l126_126144


namespace train_crossing_time_l126_126141

noncomputable def train_speed_kmph : ℕ := 72
noncomputable def platform_length_m : ℕ := 300
noncomputable def crossing_time_platform_s : ℕ := 33
noncomputable def train_speed_mps : ℕ := (train_speed_kmph * 5) / 18

theorem train_crossing_time (L : ℕ) (hL : L + platform_length_m = train_speed_mps * crossing_time_platform_s) :
  L / train_speed_mps = 18 :=
  by
    have : train_speed_mps = 20 := by
      sorry
    have : L = 360 := by
      sorry
    sorry

end train_crossing_time_l126_126141


namespace area_of_triangle_l126_126365

theorem area_of_triangle 
  (a b : ℝ) 
  (C : ℝ) 
  (h_a : a = 1) 
  (h_b : b = sqrt 3) 
  (h_C : C = 30 * real.pi / 180) : 
  1 / 2 * a * b * real.sin C = sqrt 3 / 4 :=
by 
  simp [h_a, h_b, h_C, real.sin_pi_div_two]
  sorry

end area_of_triangle_l126_126365


namespace range_of_a_l126_126768

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, (1 ≤ x) → ∀ y : ℝ, (1 ≤ y) → (x ≤ y) → (Real.exp (abs (x - a)) ≤ Real.exp (abs (y - a)))) : a ≤ 1 :=
sorry

end range_of_a_l126_126768


namespace angle_ABD_l126_126058

theorem angle_ABD (A B C D E F : Type)
  (quadrilateral : Prop)
  (angle_ABC : ℝ)
  (angle_BDE : ℝ)
  (angle_BDF : ℝ)
  (h1 : quadrilateral)
  (h2 : angle_ABC = 120)
  (h3 : angle_BDE = 30)
  (h4 : angle_BDF = 28) :
  (180 - angle_ABC = 60) :=
by
  sorry

end angle_ABD_l126_126058


namespace problem_solution_l126_126403

noncomputable def f (x : ℝ) := x / (|x| + 1)

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def range_of_f (f : ℝ → ℝ) (s : set ℝ) := ∀ y, y ∈ s ↔ ∃ x, f x = y

def increasing_function (f : ℝ → ℝ) := ∀ x y, x < y → f x ≤ f y

def exists_no_real_k (f g : ℝ → ℝ) := ∀ k, ¬ ∀ x, g x ≠ 0

theorem problem_solution :
  odd_function f ∧
  range_of_f f (set.Ioo (-1 : ℝ) 1) ∧
  increasing_function f ∧
  exists_no_real_k f (λ x, f x - k * x - k) :=
by
  sorry

end problem_solution_l126_126403


namespace second_factor_of_lcm_l126_126229

theorem second_factor_of_lcm (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) (lcm : ℕ) 
  (h1 : hcf = 20) 
  (h2 : A = 280)
  (h3 : factor1 = 13) 
  (h4 : lcm = hcf * factor1 * factor2) 
  (h5 : A = hcf * 14) : 
  factor2 = 14 :=
by 
  sorry

end second_factor_of_lcm_l126_126229


namespace added_amount_correct_l126_126036

theorem added_amount_correct (n x : ℕ) (h1 : n = 20) (h2 : 1/2 * n + x = 15) :
  x = 5 :=
by
  sorry

end added_amount_correct_l126_126036


namespace smallest_n_l126_126151

-- Define the costs.
def cost_red := 10 * 8  -- = 80
def cost_green := 18 * 12  -- = 216
def cost_blue := 20 * 15  -- = 300
def cost_yellow (n : Nat) := 24 * n

-- Define the LCM of the costs.
def LCM_cost : Nat := Nat.lcm (Nat.lcm cost_red cost_green) cost_blue

-- Problem statement: Prove that the smallest value of n such that 24 * n is the LCM of the candy costs is 150.
theorem smallest_n : ∃ n : Nat, cost_yellow n = LCM_cost ∧ n = 150 := 
by {
  -- This part is just a placeholder; the proof steps are omitted.
  sorry
}

end smallest_n_l126_126151


namespace sum_of_cubes_consecutive_integers_divisible_by_9_l126_126832

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end sum_of_cubes_consecutive_integers_divisible_by_9_l126_126832


namespace sin_180_is_zero_l126_126296

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l126_126296


namespace bhupathi_amount_l126_126916

variable (A B : ℝ)

theorem bhupathi_amount :
  (A + B = 1210 ∧ (4 / 15) * A = (2 / 5) * B) → B = 484 :=
by
  sorry

end bhupathi_amount_l126_126916


namespace largest_divisor_l126_126399

theorem largest_divisor (n : ℕ) (hn : Even n) : ∃ k, ∀ n, Even n → k ∣ (n * (n+2) * (n+4) * (n+6) * (n+8)) ∧ (∀ m, (∀ n, Even n → m ∣ (n * (n+2) * (n+4) * (n+6) * (n+8))) → m ≤ k) :=
by
  use 96
  { sorry }

end largest_divisor_l126_126399


namespace distance_between_petya_and_misha_l126_126686

theorem distance_between_petya_and_misha 
  (v1 v2 v3 : ℝ) -- Speeds of Misha, Dima, and Petya
  (t1 : ℝ) -- Time taken by Misha to finish the race
  (d : ℝ := 1000) -- Distance of the race
  (h1 : d - (v1 * (d / v1)) = 0)
  (h2 : d - 0.9 * v1 * (d / v1) = 100)
  (h3 : d - 0.81 * v1 * (d / v1) = 100) :
  (d - 0.81 * v1 * (d / v1) = 190) := 
sorry

end distance_between_petya_and_misha_l126_126686


namespace missing_dimension_of_carton_l126_126574

theorem missing_dimension_of_carton (x : ℕ) 
  (h1 : 0 < x)
  (h2 : 0 < 48)
  (h3 : 0 < 60)
  (h4 : 0 < 8)
  (h5 : 0 < 6)
  (h6 : 0 < 5)
  (h7 : (x * 48 * 60) / (8 * 6 * 5) = 300) : 
  x = 25 :=
by
  sorry

end missing_dimension_of_carton_l126_126574


namespace circle_general_eq_l126_126020
noncomputable def center_line (x : ℝ) := -4 * x
def tangent_line (x : ℝ) := 1 - x

def is_circle (center : ℝ × ℝ) (radius : ℝ) :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def is_on_line (p : ℝ × ℝ) := (p.2 = center_line p.1)

def is_tangent_at_p (center : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) :=
  is_circle center r ∧ p.2 = tangent_line p.1 ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

theorem circle_general_eq :
  ∀ (center : ℝ × ℝ), is_on_line center →
  ∀ r, is_tangent_at_p center (3, -2) r →
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2 →
  x^2 + y^2 - 2 * x + 8 * y + 9 = 0 := by
  sorry

end circle_general_eq_l126_126020


namespace ceil_x_pow_2_values_l126_126048

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l126_126048


namespace total_volume_of_four_cubes_is_500_l126_126553

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l126_126553


namespace convex_polygon_diagonals_l126_126346

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let V := n in
  let total_pairs := V * (V - 1) / 2 in
  let adjacent_pairs := n in
  let diagonals := total_pairs - adjacent_pairs in
  diagonals = 405 :=
  by
  -- number of sides is 30
  have hn : n = 30 := h
  -- calculate the total pairs
  have total_pairs_calc : total_pairs = 30 * 29 / 2 := by sorry
  -- calculate the adjacent pairs
  have adjacent_pairs_calc : adjacent_pairs = 30 := by sorry
  -- calculate the diagonals
  have diagonals_calc : diagonals = (30 * 29 / 2) - 30 := by sorry
  -- proved statement
  show 405 = 405, by rfl

end convex_polygon_diagonals_l126_126346


namespace max_teams_participation_l126_126809

theorem max_teams_participation (n : ℕ) (H : 9 * n * (n - 1) / 2 ≤ 200) : n ≤ 7 := by
  -- Proof to be filled in
  sorry

end max_teams_participation_l126_126809


namespace total_volume_of_four_boxes_l126_126549

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l126_126549


namespace gcd_lcm_sum_8_12_l126_126903

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l126_126903


namespace cross_shape_rectangle_count_l126_126919

def original_side_length := 30
def smallest_square_side_length := 1
def cut_corner_length := 10
def N : ℕ := sorry  -- total number of rectangles in the resultant graph paper
def result : ℕ := 14413

theorem cross_shape_rectangle_count :
  (1/10 : ℚ) * N = result := 
sorry

end cross_shape_rectangle_count_l126_126919


namespace distinct_prime_factors_count_l126_126793

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l126_126793


namespace number_of_ways_to_assign_friends_to_teams_l126_126800

theorem number_of_ways_to_assign_friends_to_teams (n m : ℕ) (h_n : n = 7) (h_m : m = 4) : m ^ n = 16384 :=
by
  rw [h_n, h_m]
  exact pow_succ' 4 6

end number_of_ways_to_assign_friends_to_teams_l126_126800


namespace sum_of_gcd_and_lcm_is_28_l126_126892

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l126_126892


namespace divisible_by_24_l126_126382

theorem divisible_by_24 (n : ℕ) (hn : n > 0) : 24 ∣ n * (n + 2) * (5 * n - 1) * (5 * n + 1) := 
by sorry

end divisible_by_24_l126_126382


namespace curve_focus_x_axis_l126_126844

theorem curve_focus_x_axis : 
    (x^2 - y^2 = 1)
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (a*x^2 + b*y^2 = 1 → False)
    )
    ∨ (∃ a b : ℝ, a ≠ 0 ∧ a ≠ b ∧ 
        (b*y^2 - a*x^2 = 1 → False)
    )
    ∨ (∃ c : ℝ, c ≠ 0 ∧ 
        (y = c*x^2 → False)
    ) :=
sorry

end curve_focus_x_axis_l126_126844


namespace mark_and_alice_probability_l126_126224

def probability_sunny_days : ℚ := 51 / 250

theorem mark_and_alice_probability :
  (∀ (day : ℕ), day < 5 → (∃ rain_prob sun_prob : ℚ, rain_prob = 0.8 ∧ sun_prob = 0.2 ∧ rain_prob + sun_prob = 1))
  → probability_sunny_days = 51 / 250 :=
by sorry

end mark_and_alice_probability_l126_126224


namespace sum_gcd_lcm_eight_twelve_l126_126875

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l126_126875


namespace smallest_n_l126_126009

theorem smallest_n :
  ∃ n : ℕ, n = 10 ∧ (n * (n + 1) > 100 ∧ ∀ m : ℕ, m < n → m * (m + 1) ≤ 100) := by
  sorry

end smallest_n_l126_126009


namespace amare_needs_more_fabric_l126_126277

theorem amare_needs_more_fabric :
  let first_two_dresses_in_feet := 2 * 5.5 * 3
  let next_two_dresses_in_feet := 2 * 6 * 3
  let last_two_dresses_in_feet := 2 * 6.5 * 3
  let total_fabric_needed := first_two_dresses_in_feet + next_two_dresses_in_feet + last_two_dresses_in_feet
  let fabric_amare_has := 10
  total_fabric_needed - fabric_amare_has = 98 :=
by {
  sorry
}

end amare_needs_more_fabric_l126_126277


namespace sum_of_all_angles_l126_126153

-- Defining the three triangles and their properties
structure Triangle :=
  (a1 a2 a3 : ℝ)
  (sum : a1 + a2 + a3 = 180)

def triangle_ABC : Triangle := {a1 := 1, a2 := 2, a3 := 3, sum := sorry}
def triangle_DEF : Triangle := {a1 := 4, a2 := 5, a3 := 6, sum := sorry}
def triangle_GHI : Triangle := {a1 := 7, a2 := 8, a3 := 9, sum := sorry}

theorem sum_of_all_angles :
  triangle_ABC.a1 + triangle_ABC.a2 + triangle_ABC.a3 +
  triangle_DEF.a1 + triangle_DEF.a2 + triangle_DEF.a3 +
  triangle_GHI.a1 + triangle_GHI.a2 + triangle_GHI.a3 = 540 := by
  sorry

end sum_of_all_angles_l126_126153


namespace Carla_is_2_years_older_than_Karen_l126_126055

-- Define the current age of Karen.
def Karen_age : ℕ := 2

-- Define the current age of Frank given that in 5 years he will be 36 years old.
def Frank_age : ℕ := 36 - 5

-- Define the current age of Ty given that Frank will be 3 times his age in 5 years.
def Ty_age : ℕ := 36 / 3

-- Define Carla's current age given that Ty is currently 4 years more than two times Carla's age.
def Carla_age : ℕ := (Ty_age - 4) / 2

-- Define the difference in age between Carla and Karen.
def Carla_Karen_age_diff : ℕ := Carla_age - Karen_age

-- The statement to be proven.
theorem Carla_is_2_years_older_than_Karen : Carla_Karen_age_diff = 2 := by
  -- The proof is not required, so we use sorry.
  sorry

end Carla_is_2_years_older_than_Karen_l126_126055


namespace proof_least_sum_l126_126661

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l126_126661


namespace solve_sys_eqns_l126_126093

def sys_eqns_solution (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

theorem solve_sys_eqns :
  ∃ (x y : ℝ),
  (sys_eqns_solution x y ∧
  ((x = 0 ∧ y = 0) ∨
  (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
  (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2))) :=
by
  sorry

end solve_sys_eqns_l126_126093


namespace cost_prices_max_profit_find_m_l126_126385

-- Part 1
theorem cost_prices (x y: ℕ) (h1 : 40 * x + 30 * y = 5000) (h2 : 10 * x + 50 * y = 3800) : 
  x = 80 ∧ y = 60 :=
sorry

-- Part 2
theorem max_profit (a: ℕ) (h1 : 70 ≤ a ∧ a ≤ 75) : 
  (20 * a + 6000) ≤ 7500 :=
sorry

-- Part 3
theorem find_m (m : ℝ) (h1 : 4 < m ∧ m < 8) (h2 : (20 - 5 * m) * 70 + 6000 = 5720) : 
  m = 4.8 :=
sorry

end cost_prices_max_profit_find_m_l126_126385


namespace minimum_value_ineq_l126_126372

variable {a b c : ℝ}

theorem minimum_value_ineq (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 + 2 * a + 1) * (b^2 + 2 * b + 1) * (c^2 + 2 * c + 1) / (a * b * c) ≥ 64 :=
sorry

end minimum_value_ineq_l126_126372


namespace calculate_liquids_l126_126121

def water_ratio := 60 -- mL of water for every 400 mL of flour
def milk_ratio := 80 -- mL of milk for every 400 mL of flour
def flour_ratio := 400 -- mL of flour in one portion

def flour_quantity := 1200 -- mL of flour available

def number_of_portions := flour_quantity / flour_ratio

def total_water := number_of_portions * water_ratio
def total_milk := number_of_portions * milk_ratio

theorem calculate_liquids :
  total_water = 180 ∧ total_milk = 240 :=
by
  -- Proof will be filled in here. Skipping with sorry for now.
  sorry

end calculate_liquids_l126_126121


namespace sum_zero_l126_126147

variable {a b c d : ℝ}

-- Pairwise distinct real numbers
axiom h1 : a ≠ b
axiom h2 : a ≠ c
axiom h3 : a ≠ d
axiom h4 : b ≠ c
axiom h5 : b ≠ d
axiom h6 : c ≠ d

-- Given condition
axiom h : (a^2 + b^2 - 1) * (a + b) = (b^2 + c^2 - 1) * (b + c) ∧ 
          (b^2 + c^2 - 1) * (b + c) = (c^2 + d^2 - 1) * (c + d)

theorem sum_zero : a + b + c + d = 0 :=
sorry

end sum_zero_l126_126147


namespace necessary_not_sufficient_to_form_triangle_l126_126408

-- Define the vectors and the condition
variables (a b c : ℝ × ℝ)

-- Define the condition that these vectors form a closed loop (triangle)
def forms_closed_loop (a b c : ℝ × ℝ) : Prop :=
  a + b + c = (0, 0)

-- Prove that the condition is necessary but not sufficient
theorem necessary_not_sufficient_to_form_triangle :
  forms_closed_loop a b c → ∃ (x : ℝ × ℝ), a ≠ x ∧ b ≠ -2 * x ∧ c ≠ x :=
sorry

end necessary_not_sufficient_to_form_triangle_l126_126408


namespace jellybeans_in_jar_l126_126852

theorem jellybeans_in_jar (num_kids_normal : ℕ) (num_absent : ℕ) (num_jellybeans_each : ℕ) (num_leftover : ℕ) 
  (h1 : num_kids_normal = 24) (h2 : num_absent = 2) (h3 : num_jellybeans_each = 3) (h4 : num_leftover = 34) : 
  (num_kids_normal - num_absent) * num_jellybeans_each + num_leftover = 100 :=
by sorry

end jellybeans_in_jar_l126_126852


namespace parallel_lines_a_eq_neg2_l126_126479

theorem parallel_lines_a_eq_neg2 (a : ℝ) :
  (∀ x y : ℝ, (ax + y - 1 - a = 0) ↔ (x - (1/2) * y = 0)) → a = -2 :=
by sorry

end parallel_lines_a_eq_neg2_l126_126479


namespace find_x_l126_126688

def twenty_four_is_30_percent_of (x : ℝ) : Prop := 24 = 0.3 * x

theorem find_x : ∃ x : ℝ, twenty_four_is_30_percent_of x ∧ x = 80 :=
by {
    use 80,
    split,
    {
        -- 24 = 0.3 * 80
        sorry
    },
    {
        -- x = 80
        refl
    }
}

end find_x_l126_126688


namespace distance_between_parallel_lines_correct_l126_126450

open Real

noncomputable def distance_between_parallel_lines : ℝ :=
  let a := (3, 1)
  let b := (2, 4)
  let d := (4, -6)
  let v := (b.1 - a.1, b.2 - a.2)
  let d_perp := (6, 4) -- a vector perpendicular to d
  let v_dot_d_perp := v.1 * d_perp.1 + v.2 * d_perp.2
  let d_perp_dot_d_perp := d_perp.1 * d_perp.1 + d_perp.2 * d_perp.2
  let proj_v_onto_d_perp := (v_dot_d_perp / d_perp_dot_d_perp * d_perp.1, v_dot_d_perp / d_perp_dot_d_perp * d_perp.2)
  sqrt (proj_v_onto_d_perp.1 * proj_v_onto_d_perp.1 + proj_v_onto_d_perp.2 * proj_v_onto_d_perp.2)

theorem distance_between_parallel_lines_correct :
  distance_between_parallel_lines = (3 * sqrt 13) / 13 := by
  sorry

end distance_between_parallel_lines_correct_l126_126450


namespace diagonal_count_of_convex_polygon_30_sides_l126_126348
-- Importing the entire Mathlib library for necessary mathematical constructs

theorem diagonal_count_of_convex_polygon_30_sides : 
  let n := 30 in
  let num_diagonals := (n * (n - 3)) / 2 in
  num_diagonals = 405 := 
by {
  let n := 30
  let num_diagonals := (n * (n - 3)) / 2
  show num_diagonals = 405,
  sorry
}

end diagonal_count_of_convex_polygon_30_sides_l126_126348


namespace distance_to_gym_l126_126280

theorem distance_to_gym (v d : ℝ) (h_walked_200_m: 200 / v > 0) (h_double_speed: 2 * v = 2) (h_time_diff: 200 / v - d / (2 * v) = 50) : d = 300 :=
by sorry

end distance_to_gym_l126_126280


namespace water_charging_standard_l126_126126

theorem water_charging_standard
  (x y : ℝ)
  (h1 : 10 * x + 5 * y = 35)
  (h2 : 10 * x + 8 * y = 44) : 
  x = 2 ∧ y = 3 :=
by
  sorry

end water_charging_standard_l126_126126


namespace sum_ages_l126_126732

theorem sum_ages (x : ℕ) (h_triple : True) (h_sons_age : ∀ a, a ∈ [16, 16, 16]) (h_beau_age : 42 = 42) :
  3 * (16 - x) = 42 - x → x = 3 := by
  sorry

end sum_ages_l126_126732


namespace man_speed_42_minutes_7_km_l126_126718

theorem man_speed_42_minutes_7_km 
  (distance : ℝ) (time_minutes : ℝ) (time_hours : ℝ)
  (h1 : distance = 7) 
  (h2 : time_minutes = 42) 
  (h3 : time_hours = time_minutes / 60) :
  distance / time_hours = 10 := by
  sorry

end man_speed_42_minutes_7_km_l126_126718


namespace sum_of_squares_not_perfect_square_l126_126516

theorem sum_of_squares_not_perfect_square (n : ℤ) : ¬ (∃ k : ℤ, k^2 = (n-2)^2 + (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2) :=
by
  sorry

end sum_of_squares_not_perfect_square_l126_126516


namespace volume_of_inscribed_cubes_l126_126719

noncomputable def tetrahedron_cube_volume (a m : ℝ) : ℝ × ℝ :=
  let V1 := (a * m / (a + m))^3
  let V2 := (a * m / (a + (Real.sqrt 2) * m))^3
  (V1, V2)

theorem volume_of_inscribed_cubes (a m : ℝ) (ha : 0 < a) (hm : 0 < m) :
  tetrahedron_cube_volume a m = 
  ( (a * m / (a + m))^3, 
    (a * m / (a + (Real.sqrt 2) * m))^3 ) :=
  by
    sorry

end volume_of_inscribed_cubes_l126_126719


namespace time_after_1750_minutes_is_1_10_pm_l126_126388

def add_minutes_to_time (hours : Nat) (minutes : Nat) : Nat × Nat :=
  let total_minutes := hours * 60 + minutes
  (total_minutes / 60, total_minutes % 60)

def time_after_1750_minutes (current_hour : Nat) (current_minute : Nat) : Nat × Nat :=
  let (new_hour, new_minute) := add_minutes_to_time current_hour current_minute
  let final_hour := (new_hour + 1750 / 60) % 24
  let final_minute := (new_minute + 1750 % 60) % 60
  (final_hour, final_minute)

theorem time_after_1750_minutes_is_1_10_pm : 
  time_after_1750_minutes 8 0 = (13, 10) :=
by {
  sorry
}

end time_after_1750_minutes_is_1_10_pm_l126_126388


namespace correct_calculation_l126_126909

variable (a b : ℝ)

theorem correct_calculation :
  -(a - b) = -a + b := by
  sorry

end correct_calculation_l126_126909


namespace shoot_down_probability_l126_126811

-- Define the probabilities
def P_hit_nose := 0.2
def P_hit_middle := 0.4
def P_hit_tail := 0.1
def P_miss := 0.3

-- Define the condition: probability of shooting down the plane with at most 2 shots
def condition := (P_hit_tail + (P_hit_nose * P_hit_nose) + (P_miss * P_hit_tail))

-- Proving the probability matches the required value
theorem shoot_down_probability : condition = 0.23 :=
by
  sorry

end shoot_down_probability_l126_126811


namespace sum_gcf_lcm_eq_28_l126_126879

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l126_126879


namespace tan_zero_l126_126738

theorem tan_zero : Real.tan 0 = 0 := 
by
  sorry

end tan_zero_l126_126738


namespace gcd_lcm_sum_8_12_l126_126895

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l126_126895


namespace total_animals_l126_126535

theorem total_animals : ∀ (D C R : ℕ), 
  C = 5 * D →
  R = D - 12 →
  R = 4 →
  (C + D + R = 100) :=
by
  intros D C R h1 h2 h3
  sorry

end total_animals_l126_126535


namespace smallest_positive_period_f_g_def_l126_126208

noncomputable def f (x : ℝ) : ℝ := (real.sqrt 2 / 2) * real.cos (2 * x + real.pi / 4) + real.sin x ^ 2

-- Part (I): Proving the smallest positive period of f(x)
theorem smallest_positive_period_f : (∃ T > 0, ∀ x : ℝ, f(x + T) = f(x)) ∧ ∀ T > 0, (∃ x : ℝ, f(x + T) ≠ f(x)) :=
sorry

-- Part (II): Proving the expression of g(x) in the interval [-π,0]
def g (x : ℝ) : ℝ :=
if x ∈ Icc (-real.pi) (0) then
  if x ∈ Icc (-real.pi / 2) (0) then -1 / 2 * real.sin (2 * x)
  else if x ∈ Icc (-real.pi) (-real.pi / 2) then 1 / 2 * real.sin (2 * x)
  else 0
else 0

theorem g_def (x : ℝ) (hx : x ∈ Icc (-real.pi) (0)) :
  g x = if x ∈ Icc (-real.pi / 2) (0) then -1 / 2 * real.sin (2 * x)
        else if x ∈ Icc (-real.pi) (-real.pi / 2) then 1 / 2 * real.sin (2 * x)
        else 0 :=
sorry

end smallest_positive_period_f_g_def_l126_126208


namespace items_left_in_store_l126_126149

def restocked : ℕ := 4458
def sold : ℕ := 1561
def storeroom : ℕ := 575

theorem items_left_in_store : restocked - sold + storeroom = 3472 := by
  sorry

end items_left_in_store_l126_126149


namespace miles_total_instruments_l126_126214

-- Definitions based on the conditions
def fingers : ℕ := 10
def hands : ℕ := 2
def heads : ℕ := 1
def trumpets : ℕ := fingers - 3
def guitars : ℕ := hands + 2
def trombones : ℕ := heads + 2
def french_horns : ℕ := guitars - 1
def total_instruments : ℕ := trumpets + guitars + trombones + french_horns

-- Main theorem
theorem miles_total_instruments : total_instruments = 17 := 
sorry

end miles_total_instruments_l126_126214


namespace area_of_triangle_l126_126543

theorem area_of_triangle:
  let line1 := λ x => 3 * x - 6
  let line2 := λ x => -2 * x + 18
  let y_axis: ℝ → ℝ := λ _ => 0
  let intersection := (4.8, line1 4.8)
  let y_intercept1 := (0, -6)
  let y_intercept2 := (0, 18)
  (1/2) * 24 * 4.8 = 57.6 := by
  sorry

end area_of_triangle_l126_126543


namespace remainder_of_division_l126_126443

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l126_126443


namespace quadratic_func_inequality_l126_126032

theorem quadratic_func_inequality (c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 4 * x + c)
  (h_increasing : ∀ x y, x ≤ y → -2 ≤ x → f x ≤ f y) :
  f 1 > f 0 ∧ f 0 > f (-2) :=
by
  sorry

end quadratic_func_inequality_l126_126032


namespace range_of_a_l126_126975

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → x < 4 → x^2 - 2 * x + 1 - a^2 < 0) ↔ a > 3 ∨ a < -3 :=
by
  sorry

end range_of_a_l126_126975


namespace Lizzy_total_after_loan_returns_l126_126083

theorem Lizzy_total_after_loan_returns : 
  let initial_amount := 50
  let alice_loan := 25 
  let alice_interest_rate := 0.15
  let bob_loan := 20
  let bob_interest_rate := 0.20
  let alice_interest := alice_loan * alice_interest_rate
  let bob_interest := bob_loan * bob_interest_rate
  let total_alice := alice_loan + alice_interest
  let total_bob := bob_loan + bob_interest
  let total_amount := total_alice + total_bob
  total_amount = 52.75 :=
by
  sorry

end Lizzy_total_after_loan_returns_l126_126083


namespace total_number_of_gifts_l126_126264

/-- Number of gifts calculation, given the distribution conditions with certain children -/
theorem total_number_of_gifts
  (n : ℕ) -- the total number of children
  (h1 : 2 * 4 + (n - 2) * 3 + 11 = 3 * n + 13) -- first scenario equation
  (h2 : 4 * 3 + (n - 4) * 6 + 10 = 6 * n - 2) -- second scenario equation
  : 3 * n + 13 = 28 := 
by 
  sorry

end total_number_of_gifts_l126_126264


namespace B_completes_in_40_days_l126_126708

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l126_126708


namespace total_volume_of_four_cubes_is_500_l126_126552

-- Definition of the edge length of each cube
def edge_length : ℝ := 5

-- Definition of the volume of one cube
def volume_of_cube (s : ℝ) : ℝ := s^3

-- Definition of the number of cubes
def number_of_cubes : ℕ := 4

-- Definition of the total volume
def total_volume (n : ℕ) (v : ℝ) : ℝ := n * v

-- The proposition we want to prove
theorem total_volume_of_four_cubes_is_500 :
  total_volume number_of_cubes (volume_of_cube edge_length) = 500 :=
by
  sorry

end total_volume_of_four_cubes_is_500_l126_126552


namespace shortest_distance_exp_graph_to_line_l126_126370

open Real

theorem shortest_distance_exp_graph_to_line :
  ∀ (P : ℝ × ℝ), P.2 = exp P.1 → 
  ∃ Q : ℝ × ℝ, Q = (0, 1) ∧ dist P (1, 1) = sqrt 2 / 2 := 
by
  sorry

end shortest_distance_exp_graph_to_line_l126_126370


namespace limit_sequence_is_5_l126_126947

noncomputable def sequence (n : ℕ) : ℝ :=
  (real.sqrt (3 * n - 1) - real.cbrt (125 * n ^ 3 + n)) / (real.rpow n (1 / 5) - n)

theorem limit_sequence_is_5 : 
  filter.tendsto (sequence) at_top (nhds 5) :=
sorry

end limit_sequence_is_5_l126_126947


namespace sum_of_roots_3x2_minus_12x_plus_12_eq_4_l126_126402

def sum_of_roots_quadratic (a b : ℚ) (h : a ≠ 0) : ℚ := -b / a

theorem sum_of_roots_3x2_minus_12x_plus_12_eq_4 :
  sum_of_roots_quadratic 3 (-12) (by norm_num) = 4 :=
sorry

end sum_of_roots_3x2_minus_12x_plus_12_eq_4_l126_126402


namespace op_example_l126_126202

variables {α β : ℚ}

def op (α β : ℚ) := α * β + 1

theorem op_example : op 2 (-3) = -5 :=
by
  -- The proof is omitted as requested
  sorry

end op_example_l126_126202


namespace quadratic_unique_solution_l126_126393

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l126_126393


namespace pool_capacity_is_800_l126_126116

-- Definitions for the given problem conditions
def fill_time_all_valves : ℝ := 36
def fill_time_first_valve : ℝ := 180
def fill_time_second_valve : ℝ := 240
def third_valve_more_than_first : ℝ := 30
def third_valve_more_than_second : ℝ := 10
def leak_rate : ℝ := 20

-- Function definition for the capacity of the pool
def capacity (W : ℝ) : Prop :=
  let V1 := W / fill_time_first_valve
  let V2 := W / fill_time_second_valve
  let V3 := (W / fill_time_first_valve) + third_valve_more_than_first
  let effective_rate := V1 + V2 + V3 - leak_rate
  (W / fill_time_all_valves) = effective_rate

-- Proof statement that the capacity of the pool is 800 cubic meters
theorem pool_capacity_is_800 : capacity 800 :=
by
  -- Proof is omitted
  sorry

end pool_capacity_is_800_l126_126116


namespace find_ac_pair_l126_126391

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l126_126391


namespace five_hash_neg_one_l126_126454

def hash (x y : ℤ) : ℤ := x * (y + 2) + x * y

theorem five_hash_neg_one : hash 5 (-1) = 0 :=
by
  sorry

end five_hash_neg_one_l126_126454


namespace div_iff_div_l126_126373

theorem div_iff_div {a b : ℤ} : (29 ∣ (3 * a + 2 * b)) ↔ (29 ∣ (11 * a + 17 * b)) := 
by sorry

end div_iff_div_l126_126373


namespace hunter_rats_l126_126821

-- Defining the conditions
variable (H : ℕ) (E : ℕ := H + 30) (K : ℕ := 3 * (H + E)) 
  
-- Defining the total number of rats condition
def total_rats : Prop := H + E + K = 200

-- Defining the goal: Prove Hunter has 10 rats
theorem hunter_rats (h : total_rats H) : H = 10 := by
  sorry

end hunter_rats_l126_126821


namespace sin_2alpha_pos_if_tan_alpha_pos_l126_126636

theorem sin_2alpha_pos_if_tan_alpha_pos (α : ℝ) (h : Real.tan α > 0) : Real.sin (2 * α) > 0 :=
sorry

end sin_2alpha_pos_if_tan_alpha_pos_l126_126636


namespace sum_of_maximum_and_minimum_of_u_l126_126470

theorem sum_of_maximum_and_minimum_of_u :
  ∀ (x y z : ℝ),
    0 ≤ x → 0 ≤ y → 0 ≤ z →
    3 * x + 2 * y + z = 5 →
    2 * x + y - 3 * z = 1 →
    3 * x + y - 7 * z = 3 * z - 2 →
    (-5 : ℝ) / 7 + (-1 : ℝ) / 11 = -62 / 77 :=
by
  sorry

end sum_of_maximum_and_minimum_of_u_l126_126470


namespace blue_flowers_percentage_l126_126813

theorem blue_flowers_percentage :
  let total_flowers := 96
  let green_flowers := 9
  let red_flowers := 3 * green_flowers
  let yellow_flowers := 12
  let accounted_flowers := green_flowers + red_flowers + yellow_flowers
  let blue_flowers := total_flowers - accounted_flowers
  (blue_flowers / total_flowers : ℝ) * 100 = 50 :=
by
  sorry

end blue_flowers_percentage_l126_126813


namespace incorrect_fraction_addition_l126_126722

theorem incorrect_fraction_addition (a b x y : ℤ) (h1 : 0 < b) (h2 : 0 < y) (h3 : (a + x) * (b * y) = (a * y + b * x) * (b + y)) :
  ∃ k : ℤ, x = -a * k^2 ∧ y = b * k :=
by
  sorry

end incorrect_fraction_addition_l126_126722


namespace max_valid_words_for_AU_language_l126_126987

noncomputable def maxValidWords : ℕ :=
  2^14 - 128

theorem max_valid_words_for_AU_language 
  (letters : Finset (String)) (validLengths : Set ℕ) (noConcatenation : Prop) :
  letters = {"a", "u"} ∧ validLengths = {n | 1 ≤ n ∧ n ≤ 13} ∧ noConcatenation →
  maxValidWords = 16256 :=
by
  sorry

end max_valid_words_for_AU_language_l126_126987


namespace ada_originally_in_seat2_l126_126751

inductive Seat
| S1 | S2 | S3 | S4 | S5 deriving Inhabited, DecidableEq

def moveRight : Seat → Option Seat
| Seat.S1 => some Seat.S2
| Seat.S2 => some Seat.S3
| Seat.S3 => some Seat.S4
| Seat.S4 => some Seat.S5
| Seat.S5 => none

def moveLeft : Seat → Option Seat
| Seat.S1 => none
| Seat.S2 => some Seat.S1
| Seat.S3 => some Seat.S2
| Seat.S4 => some Seat.S3
| Seat.S5 => some Seat.S4

structure FriendState :=
  (bea ceci dee edie : Seat)
  (ada_left : Bool) -- Ada is away for snacks, identified by her not being in the seat row.

def initial_seating := FriendState.mk Seat.S2 Seat.S3 Seat.S4 Seat.S5 true

def final_seating (init : FriendState) : FriendState :=
  let bea' := match moveRight init.bea with
              | some pos => pos
              | none => init.bea
  let ceci' := init.ceci -- Ceci moves left then back, net zero movement
  let (dee', edie') := match moveRight init.dee, init.dee with
                      | some new_ee, ed => (new_ee, ed) -- Dee and Edie switch and Edie moves right
                      | _, _ => (init.dee, init.edie) -- If moves are invalid
  FriendState.mk bea' ceci' dee' edie' init.ada_left

theorem ada_originally_in_seat2 (init : FriendState) : init = initial_seating → final_seating init ≠ initial_seating → init.bea = Seat.S2 :=
by
  intro h_init h_finalne
  sorry -- Proof steps go here

end ada_originally_in_seat2_l126_126751


namespace find_f_neg1_plus_f_7_l126_126342

-- Given a function f : ℝ → ℝ
axiom f : ℝ → ℝ

-- f satisfies the property of an even function
axiom even_f : ∀ x : ℝ, f (-x) = f x

-- f satisfies the periodicity of period 2
axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x

-- Also, we are given that f(1) = 1
axiom f_one : f 1 = 1

-- We need to prove that f(-1) + f(7) = 2
theorem find_f_neg1_plus_f_7 : f (-1) + f 7 = 2 :=
by
  sorry

end find_f_neg1_plus_f_7_l126_126342


namespace third_player_games_l126_126859

theorem third_player_games (p1 p2 p3 : ℕ) (h1 : p1 = 21) (h2 : p2 = 10)
  (total_games : p1 = p2 + p3) : p3 = 11 :=
by
  sorry

end third_player_games_l126_126859


namespace pqrs_inequality_l126_126204

theorem pqrs_inequality (p q r : ℝ) (h_condition : ∀ x : ℝ, (x < -6 ∨ |x - 30| ≤ 2) ↔ ((x - p) * (x - q)) / (x - r) ≥ 0)
  (h_pq : p < q) : p = 28 ∧ q = 32 ∧ r = -6 ∧ p + 2 * q + 3 * r = 78 :=
by
  sorry

end pqrs_inequality_l126_126204


namespace no_solution_integral_pairs_l126_126739

theorem no_solution_integral_pairs (a b : ℤ) : (1 / (a : ℚ) + 1 / (b : ℚ) = -1 / (a + b : ℚ)) → false :=
by
  sorry

end no_solution_integral_pairs_l126_126739


namespace problem_l126_126356

theorem problem (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 :=
by
  sorry

end problem_l126_126356


namespace total_newspapers_collected_l126_126290

-- Definitions based on the conditions
def Chris_collected : ℕ := 42
def Lily_collected : ℕ := 23

-- The proof statement
theorem total_newspapers_collected :
  Chris_collected + Lily_collected = 65 := by
  sorry

end total_newspapers_collected_l126_126290


namespace calculate_expression_l126_126436

theorem calculate_expression :
  (6 * 5 * 4 * 3 * 2 * 1 - 5 * 4 * 3 * 2 * 1) / (4 * 3 * 2 * 1) = 25 := 
by sorry

end calculate_expression_l126_126436


namespace nadia_pies_l126_126085

variables (T R B S : ℕ)

theorem nadia_pies (h₁: R = T / 2) 
                   (h₂: B = R - 14) 
                   (h₃: S = (R + B) / 2) 
                   (h₄: T = R + B + S) :
                   R = 21 ∧ B = 7 ∧ S = 14 := 
  sorry

end nadia_pies_l126_126085


namespace max_ab_plus_2bc_l126_126645

theorem max_ab_plus_2bc (A B C : ℝ) (AB AC BC : ℝ) (hB : B = 60) (hAC : AC = Real.sqrt 3) :
  (AB + 2 * BC) ≤ 2 * Real.sqrt 7 :=
sorry

end max_ab_plus_2bc_l126_126645


namespace line_intersects_x_axis_at_point_l126_126284

theorem line_intersects_x_axis_at_point :
  (∃ x, 5 * 0 - 2 * x = 10) ↔ (x = -5) ∧ (∃ x, 5 * y - 2 * x = 10 ∧ y = 0) :=
by
  sorry

end line_intersects_x_axis_at_point_l126_126284


namespace fit_max_blocks_l126_126690

/-- Prove the maximum number of blocks of size 1-in x 3-in x 2-in that can fit into a box of size 4-in x 3-in x 5-in is 10. -/
theorem fit_max_blocks :
  ∀ (block_dim box_dim : ℕ → ℕ ),
  block_dim 1 = 1 ∧ block_dim 2 = 3 ∧ block_dim 3 = 2 →
  box_dim 1 = 4 ∧ box_dim 2 = 3 ∧ box_dim 3 = 5 →
  ∃ max_blocks : ℕ, max_blocks = 10 :=
by
  sorry

end fit_max_blocks_l126_126690


namespace sine_180_eq_zero_l126_126293

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l126_126293


namespace dilution_plate_count_lower_than_actual_l126_126701

theorem dilution_plate_count_lower_than_actual
  (bacteria_count : ℕ)
  (colony_count : ℕ)
  (dilution_factor : ℕ)
  (plate_count : ℕ)
  (count_error_margin : ℕ)
  (method_estimation_error : ℕ)
  (H1 : method_estimation_error > 0)
  (H2 : colony_count = bacteria_count / dilution_factor - method_estimation_error)
  : colony_count < bacteria_count :=
by
  sorry

end dilution_plate_count_lower_than_actual_l126_126701


namespace solution_set_characterization_l126_126613

noncomputable def satisfies_inequality (x : ℝ) : Bool :=
  (3 / (x + 2) + 4 / (x + 6)) > 1

theorem solution_set_characterization :
  ∀ x : ℝ, (satisfies_inequality x) ↔ (x < -7 ∨ (-6 < x ∧ x < -2) ∨ x > 2) :=
by
  intro x
  unfold satisfies_inequality
  -- here we would provide the proof
  sorry

end solution_set_characterization_l126_126613


namespace fish_weight_l126_126924

variables (H T X : ℝ)
-- Given conditions
def tail_weight : Prop := X = 1
def head_weight : Prop := H = X + 0.5 * T
def torso_weight : Prop := T = H + X

theorem fish_weight (H T X : ℝ) 
  (h_tail : tail_weight X)
  (h_head : head_weight H T X)
  (h_torso : torso_weight H T X) : 
  H + T + X = 8 :=
sorry

end fish_weight_l126_126924


namespace children_on_bus_l126_126132

theorem children_on_bus (initial_children additional_children total_children : ℕ)
  (h1 : initial_children = 64)
  (h2 : additional_children = 14)
  (h3 : total_children = initial_children + additional_children) :
  total_children = 78 :=
by
  rw [h1, h2] at h3
  exact h3

end children_on_bus_l126_126132


namespace no_valid_prime_pairs_l126_126960

noncomputable def is_prime (n : ℕ) : Prop := Nat.Prime n

theorem no_valid_prime_pairs :
  ∀ x y : ℕ, is_prime x → is_prime y → y < x → x ≤ 200 → (x % y = 0) → ((x +1) % (y +1) = 0) → false :=
by
  sorry

end no_valid_prime_pairs_l126_126960


namespace exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l126_126502

-- Define the parabola as a function
def parabola (x : ℝ) : ℝ := x^2

-- N-gon properties
def is_convex_ngon (N : ℕ) (vertices : List (ℝ × ℝ)) : Prop :=
  -- Placeholder for checking properties; actual implementation would validate convexity and equilateral nature.
  sorry 

-- Statement for 2011-gon
theorem exists_convex_2011_gon_on_parabola :
  ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2011 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

-- Statement for 2012-gon
theorem not_exists_convex_2012_gon_on_parabola :
  ¬ ∃ (vertices : List (ℝ × ℝ)), is_convex_ngon 2012 vertices ∧ ∀ v ∈ vertices, v.2 = parabola v.1 :=
sorry

end exists_convex_2011_gon_on_parabola_not_exists_convex_2012_gon_on_parabola_l126_126502


namespace abs_diff_of_solutions_eq_5_point_5_l126_126658

theorem abs_diff_of_solutions_eq_5_point_5 (x y : ℝ)
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.7)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 8.2) :
  |x - y| = 5.5 :=
sorry

end abs_diff_of_solutions_eq_5_point_5_l126_126658


namespace total_oil_leak_l126_126586

-- Definitions for the given conditions
def before_repair_leak : ℕ := 6522
def during_repair_leak : ℕ := 5165
def total_leak : ℕ := 11687

-- The proof statement (without proof, only the statement)
theorem total_oil_leak :
  before_repair_leak + during_repair_leak = total_leak :=
sorry

end total_oil_leak_l126_126586


namespace trays_from_first_table_l126_126835

-- Definitions based on conditions
def trays_per_trip : ℕ := 4
def trips : ℕ := 3
def trays_from_second_table : ℕ := 2

-- Theorem statement to prove the number of trays picked up from the first table
theorem trays_from_first_table : trays_per_trip * trips - trays_from_second_table = 10 := by
  sorry

end trays_from_first_table_l126_126835


namespace dot_product_vec_a_vec_b_l126_126035

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem dot_product_vec_a_vec_b : dot_product vec_a vec_b = 1 := by
  sorry

end dot_product_vec_a_vec_b_l126_126035


namespace proof_students_scored_above_120_l126_126981

open ProbabilityMeasure

noncomputable def estimate_students_scored_above_120 (total_students : ℕ) (μ : ℝ) (σ : ℝ) (p_interval : ℝ) (score_threshold : ℝ) : ℕ :=
  let P := 0.5 - p_interval in
  let num_students := total_students * P in
  num_students.to_nat
  
theorem proof_students_scored_above_120 :
  estimate_students_scored_above_120 50 110 10 0.36 120 = 7 :=
by
  sorry

end proof_students_scored_above_120_l126_126981


namespace trig_identity_l126_126621

open Real

theorem trig_identity 
  (θ : ℝ)
  (h : tan (π / 4 + θ) = 3) : 
  sin (2 * θ) - 2 * cos θ ^ 2 = -3 / 4 :=
by
  sorry

end trig_identity_l126_126621


namespace synthetic_analytic_incorrect_statement_l126_126913

theorem synthetic_analytic_incorrect_statement
  (basic_methods : ∀ (P Q : Prop), (P → Q) ∨ (Q → P))
  (synthetic_forward : ∀ (P Q : Prop), (P → Q))
  (analytic_backward : ∀ (P Q : Prop), (Q → P)) :
  ¬ (∀ (P Q : Prop), (P → Q) ∧ (Q → P)) :=
by
  sorry

end synthetic_analytic_incorrect_statement_l126_126913


namespace B_completes_in_40_days_l126_126709

noncomputable def BCompletesWorkInDays (x : ℝ) : ℝ :=
  let A_rate := 1 / 45
  let B_rate := 1 / x
  let work_done_together := 9 * (A_rate + B_rate)
  let work_done_B_alone := 23 * B_rate
  let total_work := 1
  work_done_together + work_done_B_alone

theorem B_completes_in_40_days :
  BCompletesWorkInDays 40 = 1 :=
by
  sorry

end B_completes_in_40_days_l126_126709


namespace sum_gcd_lcm_eight_twelve_l126_126878

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l126_126878


namespace find_side_length_l126_126222

noncomputable def side_length_of_equilateral_triangle (t : ℝ) (Q : ℝ × ℝ) : Prop :=
  let D := (0, 0)
  let E := (t, 0)
  let F := (t/2, t * (Real.sqrt 3) / 2)
  let DQ := Real.sqrt ((Q.1 - D.1) ^ 2 + (Q.2 - D.2) ^ 2)
  let EQ := Real.sqrt ((Q.1 - E.1) ^ 2 + (Q.2 - E.2) ^ 2)
  let FQ := Real.sqrt ((Q.1 - F.1) ^ 2 + (Q.2 - F.2) ^ 2)
  DQ = 2 ∧ EQ = 2 * Real.sqrt 2 ∧ FQ = 3

theorem find_side_length :
  ∃ t Q, side_length_of_equilateral_triangle t Q → t = 2 * Real.sqrt 5 :=
sorry

end find_side_length_l126_126222


namespace math_problem_proof_l126_126734

theorem math_problem_proof : 
  ((9 - 8 + 7) ^ 2 * 6 + 5 - 4 ^ 2 * 3 + 2 ^ 3 - 1) = 347 := 
by sorry

end math_problem_proof_l126_126734


namespace ratio_ravi_kiran_l126_126108

-- Definitions for the conditions
def ratio_money_ravi_giri := 6 / 7
def money_ravi := 36
def money_kiran := 105

-- The proof problem
theorem ratio_ravi_kiran : (money_ravi : ℕ) / money_kiran = 12 / 35 := 
by 
  sorry

end ratio_ravi_kiran_l126_126108


namespace angle_at_3_15_l126_126037

theorem angle_at_3_15 : 
  let minute_hand_angle := 90.0
  let hour_at_3 := 90.0
  let hour_hand_at_3_15 := hour_at_3 + 7.5
  minute_hand_angle == 90.0 ∧ hour_hand_at_3_15 == 97.5 →
  abs (hour_hand_at_3_15 - minute_hand_angle) = 7.5 :=
by
  sorry

end angle_at_3_15_l126_126037


namespace gcd_lcm_sum_8_12_l126_126902

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l126_126902


namespace calculate_division_l126_126673

theorem calculate_division : 
  (- (1 / 28)) / ((1 / 2) - (1 / 4) + (1 / 7) - (1 / 14)) = - (1 / 9) :=
by
  sorry

end calculate_division_l126_126673


namespace product_of_roots_l126_126152

theorem product_of_roots :
  (∃ r s t : ℝ, (r + s + t) = 15 ∧ (r*s + s*t + r*t) = 50 ∧ (r*s*t) = -35) ∧ (∀ x : ℝ, x^3 - 15*x^2 + 50*x + 35 = (x - r) * (x - s) * (x - t)) :=
sorry

end product_of_roots_l126_126152


namespace completing_square_16x2_32x_512_eq_33_l126_126996

theorem completing_square_16x2_32x_512_eq_33:
  (∃ p q : ℝ, (16 * x ^ 2 + 32 * x - 512 = 0) → (x + p) ^ 2 = q ∧ q = 33) :=
by
  sorry

end completing_square_16x2_32x_512_eq_33_l126_126996


namespace coating_profit_l126_126418

theorem coating_profit (x y : ℝ) (h1 : 0.6 * x + 0.9 * (150 - x) ≤ 120)
  (h2 : 0.7 * x + 0.4 * (150 - x) ≤ 90) :
  (50 ≤ x ∧ x ≤ 100) → (y = -50 * x + 75000) → (x = 50 → y = 72500) :=
by
  intros hx hy hx_val
  sorry

end coating_profit_l126_126418


namespace sin_180_degree_l126_126302

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l126_126302


namespace largest_is_A_minus_B_l126_126659

noncomputable def A := 3 * 1005^1006
noncomputable def B := 1005^1006
noncomputable def C := 1004 * 1005^1005
noncomputable def D := 3 * 1005^1005
noncomputable def E := 1005^1005
noncomputable def F := 1005^1004

theorem largest_is_A_minus_B :
  max (A - B) (max (B - C) (max (C - D) (max (D - E) (E - F)))) = A - B :=
by {
  sorry
}

end largest_is_A_minus_B_l126_126659


namespace minimal_distance_l126_126524

noncomputable def minimum_distance_travel (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) : ℝ :=
  2 * Real.sqrt 19

theorem minimal_distance (a b c : ℝ) (ha : a = 2) (hb : b = Real.sqrt 7) (hc : c = 3) :
  minimum_distance_travel a b c ha hb hc = 2 * Real.sqrt 19 :=
by
  -- Proof is omitted
  sorry

end minimal_distance_l126_126524


namespace abs_difference_21st_term_l126_126865

def sequence_C (n : ℕ) : ℤ := 50 + 12 * (n - 1)
def sequence_D (n : ℕ) : ℤ := 50 - 14 * (n - 1)

theorem abs_difference_21st_term :
  |sequence_C 21 - sequence_D 21| = 520 := by
  sorry

end abs_difference_21st_term_l126_126865


namespace no_adjacent_performers_probability_l126_126839

-- A definition to model the probability of non-adjacent performers in a circle of 6 people.
def probability_no_adjacent_performers : ℚ :=
  -- Given conditions: fair coin tosses by six people, modeling permutations
  -- and specific valid configurations derived from the problem.
  9 / 32

-- Proving the final probability calculation is correct
theorem no_adjacent_performers_probability :
  probability_no_adjacent_performers = 9 / 32 :=
by
  -- Using sorry to indicate the proof needs to be filled in, acknowledging the correct answer.
  sorry

end no_adjacent_performers_probability_l126_126839


namespace unknown_number_value_l126_126640

theorem unknown_number_value (a x : ℕ) (h₁ : a = 105) (h₂ : a^3 = 21 * x * 35 * 63) : x = 25 := by
  sorry

end unknown_number_value_l126_126640


namespace min_value_expr_sum_of_squares_inequality_l126_126337

-- Given conditions
variables (a b : ℝ)
variables (ha : a > 0) (hb : b > 0) (hab : a + b = 2)

-- Problem (1): Prove minimum value of (2 / a + 8 / b) is 9
theorem min_value_expr : ∃ a b, a > 0 ∧ b > 0 ∧ a + b = 2 ∧ ((2 / a) + (8 / b) = 9) := sorry

-- Problem (2): Prove a^2 + b^2 ≥ 2
theorem sum_of_squares_inequality : a^2 + b^2 ≥ 2 :=
by { sorry }

end min_value_expr_sum_of_squares_inequality_l126_126337


namespace equal_sum_squares_l126_126223

open BigOperators

-- Definitions
def n := 10

-- Assuming x and y to be arrays that hold the number of victories and losses for each player respectively.
variables {x y : Fin n → ℝ}

-- Conditions
axiom pair_meet_once : ∀ i : Fin n, x i + y i = (n - 1)

-- Theorem to be proved
theorem equal_sum_squares : ∑ i : Fin n, x i ^ 2 = ∑ i : Fin n, y i ^ 2 :=
by
  sorry

end equal_sum_squares_l126_126223


namespace sum_of_GCF_and_LCM_l126_126888

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l126_126888


namespace no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l126_126567
-- Bringing in the entirety of Mathlib

-- Problem (a): There are no non-zero integers that increase by 7 or 9 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_7_or_9 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ (10 * X + d = 7 * n ∨ 10 * X + d = 9 * n)) :=
by sorry

-- Problem (b): There are no non-zero integers that increase by 4 times when the first digit is moved to the end
theorem no_nonzero_ints_increase_4 (n : ℕ) (h : n > 0) :
  ¬ (∃ d X m, n = d * 10^m + X ∧ 10 * X + d = 4 * n) :=
by sorry

end no_nonzero_ints_increase_7_or_9_no_nonzero_ints_increase_4_l126_126567


namespace sum_of_four_integers_l126_126022

noncomputable def originalSum (a b c d : ℤ) :=
  (a + b + c + d)

theorem sum_of_four_integers
  (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 8)
  (h2 : (a + b + d) / 3 + c = 12)
  (h3 : (a + c + d) / 3 + b = 32 / 3)
  (h4 : (b + c + d) / 3 + a = 28 / 3) :
  originalSum a b c d = 30 :=
sorry

end sum_of_four_integers_l126_126022


namespace constant_term_expansion_l126_126313

-- Definition of the binomial term
noncomputable def binomial_term (n k : ℕ) (a b : ℝ) : ℝ := nat.choose n k * a^(n-k) * b^k

-- Define the specific problem parameters
def general_term (r : ℕ) : ℝ := binomial_term 6 r (1:ℝ) (-1) * (1/x)^(6-r) * (x^(1/2))^r

-- Main statement proving the specific example
theorem constant_term_expansion: general_term 4 = 15 := by
  sorry

end constant_term_expansion_l126_126313


namespace circle_equation_through_points_l126_126017

theorem circle_equation_through_points 
  (M N : ℝ × ℝ)
  (hM : M = (5, 2))
  (hN : N = (3, 2))
  (hk : ∃ k : ℝ, (M.1 + N.1) / 2 = k ∧ (M.2 + N.2) / 2 = (2 * k - 3))
  : (∃ h : ℝ, ∀ x y: ℝ, (x - 4) ^ 2 + (y - 5) ^ 2 = h) ∧ (∃ r : ℝ, r = 10) := 
sorry

end circle_equation_through_points_l126_126017


namespace work_completion_problem_l126_126710

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l126_126710


namespace diamond_of_2_and_3_l126_126455

def diamond (a b : ℕ) : ℕ := a^3 * b^2 - b + 2

theorem diamond_of_2_and_3 : diamond 2 3 = 71 := by
  sorry

end diamond_of_2_and_3_l126_126455


namespace albert_snakes_count_l126_126143

noncomputable def garden_snake_length : ℝ := 10.0
noncomputable def boa_ratio : ℝ := 1 / 7.0
noncomputable def boa_length : ℝ := 1.428571429

theorem albert_snakes_count : 
  garden_snake_length = 10.0 ∧ 
  boa_ratio = 1 / 7.0 ∧ 
  boa_length = 1.428571429 → 
  2 = 2 :=
by
  intro h
  sorry   -- Proof will go here

end albert_snakes_count_l126_126143


namespace cube_root_solutions_l126_126824

theorem cube_root_solutions (p : ℕ) (hp : p > 3) :
    (∃ (k : ℤ) (h1 : k^2 ≡ -3 [ZMOD p]), ∀ x, x^3 ≡ 1 [ZMOD p] → 
        (x = 1 ∨ (x^2 + x + 1 ≡ 0 [ZMOD p])) )
    ∨ 
    (∀ x, x^3 ≡ 1 [ZMOD p] → x = 1) := 
sorry

end cube_root_solutions_l126_126824


namespace three_scientists_same_topic_l126_126685

theorem three_scientists_same_topic
  (scientists : Finset ℕ)
  (h_size : scientists.card = 17)
  (topics : Finset ℕ)
  (h_topics : topics.card = 3)
  (communicates : ℕ → ℕ → ℕ)
  (h_communicate : ∀ a b : ℕ, a ≠ b → b ∈ scientists → communicates a b ∈ topics) :
  ∃ (a b c : ℕ), a ∈ scientists ∧ b ∈ scientists ∧ c ∈ scientists ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  communicates a b = communicates b c ∧ communicates b c = communicates a c := 
sorry

end three_scientists_same_topic_l126_126685


namespace factor_expression_l126_126480

theorem factor_expression (a b c : ℝ) :
  a^3 * (b^3 - c^3) + b^3 * (c^3 - a^3) + c^3 * (a^3 - b^3) =
  (a - b) * (b - c) * (c - a) * (a^2 + b^2 + c^2 + ab + bc + ca) :=
by
  sorry

end factor_expression_l126_126480


namespace freddy_total_call_cost_l126_126756

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l126_126756


namespace solve_equation_l126_126092

theorem solve_equation (x : ℝ) (h : x = 5) :
  (3 * x - 5) / (x^2 - 7 * x + 12) + (5 * x - 1) / (x^2 - 5 * x + 6) = (8 * x - 13) / (x^2 - 6 * x + 8) := 
  by 
  rw [h]
  sorry

end solve_equation_l126_126092


namespace simplify_fraction_l126_126838

theorem simplify_fraction (n : Nat) : (2^(n+4) - 3 * 2^n) / (2 * 2^(n+3)) = 13 / 16 :=
by
  sorry

end simplify_fraction_l126_126838


namespace ackermann_3_2_l126_126591

-- Define the Ackermann function
def ackermann : ℕ → ℕ → ℕ
| 0, n => n + 1
| (m + 1), 0 => ackermann m 1
| (m + 1), (n + 1) => ackermann m (ackermann (m + 1) n)

-- Prove that A(3, 2) = 29
theorem ackermann_3_2 : ackermann 3 2 = 29 := by
  sorry

end ackermann_3_2_l126_126591


namespace dot_product_result_l126_126775

open Real

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-1, 2)

def scale_vec (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add_vec (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem dot_product_result :
  dot_product (add_vec (scale_vec 2 a) b) a = 6 :=
by
  sorry

end dot_product_result_l126_126775


namespace count_arithmetic_sequence_l126_126796

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l126_126796


namespace number_of_distinct_prime_factors_90_l126_126795

theorem number_of_distinct_prime_factors_90 : 
  (∃ (s : set ℕ), (∀ p ∈ s, (nat.prime p)) ∧ (s.prod id = 90) ∧ s.card = 3) :=
sorry

end number_of_distinct_prime_factors_90_l126_126795


namespace correct_product_l126_126814

-- Definitions for conditions
def reversed_product (a b : ℕ) : Prop :=
  let reversed_a := (a % 10) * 10 + (a / 10)
  reversed_a * b = 204

theorem correct_product (a b : ℕ) (h : reversed_product a b) : a * b = 357 := 
by
  sorry

end correct_product_l126_126814


namespace mistaken_divisor_is_12_l126_126057

theorem mistaken_divisor_is_12 (dividend : ℕ) (mistaken_divisor : ℕ) (correct_divisor : ℕ) 
  (mistaken_quotient : ℕ) (correct_quotient : ℕ) (remainder : ℕ) :
  remainder = 0 ∧ correct_divisor = 21 ∧ mistaken_quotient = 42 ∧ correct_quotient = 24 ∧ 
  dividend = mistaken_quotient * mistaken_divisor ∧ dividend = correct_quotient * correct_divisor →
  mistaken_divisor = 12 :=
by 
  sorry

end mistaken_divisor_is_12_l126_126057


namespace find_speed_of_goods_train_l126_126409

noncomputable def speed_of_goods_train (v_man : ℝ) (t_pass : ℝ) (d_goods : ℝ) : ℝ := 
  let v_man_mps := v_man * (1000 / 3600)
  let v_relative := d_goods / t_pass
  let v_goods_mps := v_relative - v_man_mps
  v_goods_mps * (3600 / 1000)

theorem find_speed_of_goods_train :
  speed_of_goods_train 45 8 340 = 108 :=
by sorry

end find_speed_of_goods_train_l126_126409


namespace trigonometric_quadrant_l126_126043

theorem trigonometric_quadrant (θ : ℝ) (h1 : Real.sin θ > Real.cos θ) (h2 : Real.sin θ * Real.cos θ < 0) : 
  (θ > π / 2) ∧ (θ < π) :=
by
  sorry

end trigonometric_quadrant_l126_126043


namespace min_value_sq_distance_l126_126472

theorem min_value_sq_distance {x y : ℝ} (h : x^2 + y^2 - 4 * x + 2 = 0) : 
  ∃ (m : ℝ), m = 2 ∧ (∀ x y, x^2 + y^2 - 4 * x + 2 = 0 → x^2 + (y - 2)^2 ≥ m) :=
sorry

end min_value_sq_distance_l126_126472


namespace vishal_investment_more_than_trishul_l126_126125

theorem vishal_investment_more_than_trishul :
  ∀ (V T R : ℝ), R = 2000 → T = R - 0.10 * R → V + T + R = 5780 → (V - T) / T * 100 = 10 :=
by
  intros V T R hR hT hSum
  sorry

end vishal_investment_more_than_trishul_l126_126125


namespace range_of_m_l126_126209

theorem range_of_m (m : ℝ) (f : ℝ → ℝ) 
(hf : ∀ x, f x = (Real.sqrt 3) * Real.sin ((Real.pi * x) / m))
(exists_extremum : ∃ x₀, (deriv f x₀ = 0) ∧ (x₀^2 + (f x₀)^2 < m^2)) :
(m > 2) ∨ (m < -2) :=
sorry

end range_of_m_l126_126209


namespace probability_walk_450_feet_or_less_l126_126312

theorem probability_walk_450_feet_or_less 
  (gates : List ℕ) (initial_gate new_gate : ℕ) 
  (n : ℕ) (dist_between_adjacent_gates : ℕ) 
  (valid_gates : gates.length = n)
  (distance : dist_between_adjacent_gates = 90) :
  n = 15 → 
  (initial_gate ∈ gates ∧ new_gate ∈ gates) → 
  ∃ (m1 m2 : ℕ), m1 = 59 ∧ m2 = 105 ∧ gcd m1 m2 = 1 ∧ 
  (∃ probability : ℚ, probability = (59 / 105 : ℚ) ∧ 
  (∃ sum_m1_m2 : ℕ, sum_m1_m2 = m1 + m2 ∧ sum_m1_m2 = 164)) :=
by
  sorry

end probability_walk_450_feet_or_less_l126_126312


namespace no_integer_roots_l126_126965

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l126_126965


namespace max_radius_of_sector_l126_126628

def sector_perimeter_area (r : ℝ) : ℝ := -r^2 + 10 * r

theorem max_radius_of_sector (R A : ℝ) (h : 2 * R + A = 20) : R = 5 :=
by
  sorry

end max_radius_of_sector_l126_126628


namespace total_travel_time_is_correct_l126_126114

-- Conditions as definitions
def total_distance : ℕ := 200
def initial_fraction : ℚ := 1 / 4
def initial_time : ℚ := 1 -- in hours
def lunch_time : ℚ := 1 -- in hours
def remaining_fraction : ℚ := 1 / 2
def pit_stop_time : ℚ := 0.5 -- in hours
def speed_increase : ℚ := 10

-- Derived/Calculated values needed for the problem statement
def initial_distance : ℚ := initial_fraction * total_distance
def initial_speed : ℚ := initial_distance / initial_time
def remaining_distance : ℚ := total_distance - initial_distance
def half_remaining_distance : ℚ := remaining_fraction * remaining_distance
def second_drive_time : ℚ := half_remaining_distance / initial_speed
def last_distance : ℚ := remaining_distance - half_remaining_distance
def last_speed : ℚ := initial_speed + speed_increase
def last_drive_time : ℚ := last_distance / last_speed

-- Total time calculation
def total_time : ℚ :=
  initial_time + lunch_time + second_drive_time + pit_stop_time + last_drive_time

-- Lean theorem statement
theorem total_travel_time_is_correct : total_time = 5.25 :=
  sorry

end total_travel_time_is_correct_l126_126114


namespace pave_hall_with_stones_l126_126564

def hall_length_m : ℕ := 36
def hall_breadth_m : ℕ := 15
def stone_length_dm : ℕ := 4
def stone_breadth_dm : ℕ := 5

def to_decimeters (m : ℕ) : ℕ := m * 10

def hall_length_dm : ℕ := to_decimeters hall_length_m
def hall_breadth_dm : ℕ := to_decimeters hall_breadth_m

def hall_area_dm2 : ℕ := hall_length_dm * hall_breadth_dm
def stone_area_dm2 : ℕ := stone_length_dm * stone_breadth_dm

def number_of_stones_required : ℕ := hall_area_dm2 / stone_area_dm2

theorem pave_hall_with_stones :
  number_of_stones_required = 2700 :=
sorry

end pave_hall_with_stones_l126_126564


namespace good_numbers_correct_l126_126500

noncomputable def good_numbers (n : ℕ) : ℝ :=
  1 / 2 * (8^n + 10^n) - 1

theorem good_numbers_correct (n : ℕ) : good_numbers n = 
  1 / 2 * (8^n + 10^n) - 1 := 
sorry

end good_numbers_correct_l126_126500


namespace evie_shells_l126_126162

theorem evie_shells (shells_per_day : ℕ) (days : ℕ) (gifted_shells : ℕ) 
  (h1 : shells_per_day = 10) 
  (h2 : days = 6)
  (h3 : gifted_shells = 2) : 
  shells_per_day * days - gifted_shells = 58 := 
by
  sorry

end evie_shells_l126_126162


namespace mean_sharpening_instances_l126_126737

def pencil_sharpening_instances : List ℕ :=
  [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

def mean (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

theorem mean_sharpening_instances :
  mean pencil_sharpening_instances = 18.1 := by
  sorry

end mean_sharpening_instances_l126_126737


namespace acute_angle_ACD_l126_126533

theorem acute_angle_ACD (α : ℝ) (h : α ≤ 120) :
  ∃ (ACD : ℝ), ACD = Real.arcsin ((Real.tan (α / 2)) / Real.sqrt 3) :=
sorry

end acute_angle_ACD_l126_126533


namespace new_average_page_count_l126_126575

theorem new_average_page_count
  (n : ℕ) (a : ℕ) (p1 p2 : ℕ)
  (h_n : n = 80) (h_a : a = 120)
  (h_p1 : p1 = 150) (h_p2 : p2 = 170) :
  (n - 2) ≠ 0 → 
  ((n * a - (p1 + p2)) / (n - 2) = 119) := 
by sorry

end new_average_page_count_l126_126575


namespace length_of_AB_l126_126194

noncomputable def isosceles_triangle (a b c : ℕ) : Prop :=
  a = b ∨ a = c ∨ b = c

theorem length_of_AB 
  (a b c d e : ℕ)
  (h_iso_ABC : isosceles_triangle a b c)
  (h_iso_CDE : isosceles_triangle c d e)
  (h_perimeter_CDE : c + d + e = 25)
  (h_perimeter_ABC : a + b + c = 24)
  (h_CE : c = 9)
  (h_AB_DE : a = e) : a = 7 :=
by
  sorry

end length_of_AB_l126_126194


namespace joao_claudia_scores_l126_126072

theorem joao_claudia_scores (joao_score claudia_score total_score : ℕ) 
  (h1 : claudia_score = joao_score + 13)
  (h2 : total_score = joao_score + claudia_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score = 68 ∧ claudia_score = 81 := by
  sorry

end joao_claudia_scores_l126_126072


namespace museum_revenue_from_college_students_l126_126064

/-!
In one day, 200 people visit The Metropolitan Museum of Art in New York City. Half of the visitors are residents of New York City. 
Of the NYC residents, 30% are college students. If the cost of a college student ticket is $4, we need to prove that 
the museum gets $120 from college students that are residents of NYC.
-/

theorem museum_revenue_from_college_students :
  let total_visitors := 200
  let residents_nyc := total_visitors / 2
  let college_students_percentage := 30 / 100
  let college_students := residents_nyc * college_students_percentage
  let ticket_cost := 4
  residents_nyc = 100 ∧ 
  college_students = 30 ∧ 
  ticket_cost * college_students = 120 := 
by
  sorry

end museum_revenue_from_college_students_l126_126064


namespace sum_of_gcd_and_lcm_is_28_l126_126893

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l126_126893


namespace one_fourth_of_56_equals_75_l126_126166

theorem one_fourth_of_56_equals_75 : (5.6 / 4) = 7 / 5 := 
by
  -- Temporarily omitting the actual proof
  sorry

end one_fourth_of_56_equals_75_l126_126166


namespace koby_boxes_l126_126073

theorem koby_boxes (x : ℕ) (sparklers_per_box : ℕ := 3) (whistlers_per_box : ℕ := 5) 
    (cherie_sparklers : ℕ := 8) (cherie_whistlers : ℕ := 9) (total_fireworks : ℕ := 33) : 
    (sparklers_per_box * x + cherie_sparklers) + (whistlers_per_box * x + cherie_whistlers) = total_fireworks → x = 2 :=
by
  sorry

end koby_boxes_l126_126073


namespace delta_y_over_delta_x_l126_126630

variable (Δx : ℝ)

def f (x : ℝ) : ℝ := 2 * x^2 - 1

theorem delta_y_over_delta_x : (f (1 + Δx) - f 1) / Δx = 4 + 2 * Δx :=
by
  sorry

end delta_y_over_delta_x_l126_126630


namespace minimum_abs_phi_l126_126632

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem minimum_abs_phi 
  (ω φ b : ℝ)
  (hω : ω > 0)
  (hb : 0 < b ∧ b < 2)
  (h_intersections : f ω φ (π / 6) = b ∧ f ω φ (5 * π / 6) = b ∧ f ω φ (7 * π / 6) = b)
  (h_minimum : f ω φ (3 * π / 2) = -2) : 
  |φ| = π / 2 :=
sorry

end minimum_abs_phi_l126_126632


namespace brick_width_is_10_cm_l126_126137

-- Define the conditions
def courtyard_length_meters := 25
def courtyard_width_meters := 16
def brick_length_cm := 20
def number_of_bricks := 20000

-- Convert courtyard dimensions to area in square centimeters
def area_of_courtyard_cm2 := courtyard_length_meters * 100 * courtyard_width_meters * 100

-- Total area covered by bricks
def total_brick_area_cm2 := area_of_courtyard_cm2

-- Area covered by one brick
def area_per_brick := total_brick_area_cm2 / number_of_bricks

-- Find the brick width
def brick_width_cm := area_per_brick / brick_length_cm

-- Prove the width of each brick is 10 cm
theorem brick_width_is_10_cm : brick_width_cm = 10 := 
by 
  -- Placeholder for the proof
  sorry

end brick_width_is_10_cm_l126_126137


namespace sum_of_remainders_eq_24_l126_126405

theorem sum_of_remainders_eq_24 (a b c : ℕ) 
  (h1 : a % 30 = 13) (h2 : b % 30 = 19) (h3 : c % 30 = 22) :
  (a + b + c) % 30 = 24 :=
by
  sorry

end sum_of_remainders_eq_24_l126_126405


namespace polygon_num_sides_l126_126423

-- Define the given conditions
def perimeter : ℕ := 150
def side_length : ℕ := 15

-- State the theorem to prove the number of sides of the polygon
theorem polygon_num_sides (P : ℕ) (s : ℕ) (hP : P = perimeter) (hs : s = side_length) : P / s = 10 :=
by
  sorry

end polygon_num_sides_l126_126423


namespace expected_length_of_string_is_12_l126_126519

noncomputable def expected_length_of_string : ℝ :=
  expected_value_of_length

theorem expected_length_of_string_is_12 :
  expected_length_of_string = 12 :=
sorry

end expected_length_of_string_is_12_l126_126519


namespace sin_180_degree_l126_126304

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l126_126304


namespace mod_last_digit_l126_126379

theorem mod_last_digit (N : ℕ) (a b : ℕ) (h : N = 10 * a + b) (hb : b < 10) : 
  N % 10 = b ∧ N % 2 = b % 2 ∧ N % 5 = b % 5 :=
by
  sorry

end mod_last_digit_l126_126379


namespace circle_equation_l126_126571

theorem circle_equation (x y : ℝ) (h1 : (1 - 1)^2 + (1 - 1)^2 = 2) (h2 : (0 - 1)^2 + (0 - 1)^2 = r_sq) :
  (x - 1)^2 + (y - 1)^2 = 2 :=
sorry

end circle_equation_l126_126571


namespace cannot_form_right_triangle_l126_126253

theorem cannot_form_right_triangle : ¬∃ a b c : ℕ, a = 4 ∧ b = 6 ∧ c = 11 ∧ (a^2 + b^2 = c^2) :=
by
  sorry

end cannot_form_right_triangle_l126_126253


namespace females_dont_listen_correct_l126_126112

/-- Number of males who listen to the station -/
def males_listen : ℕ := 45

/-- Number of females who don't listen to the station -/
def females_dont_listen : ℕ := 87

/-- Total number of people who listen to the station -/
def total_listen : ℕ := 120

/-- Total number of people who don't listen to the station -/
def total_dont_listen : ℕ := 135

/-- Number of females surveyed based on the problem description -/
def total_females_surveyed (total_peoples_total : ℕ) (males_dont_listen : ℕ) : ℕ := 
  total_peoples_total - (males_listen + males_dont_listen)

/-- Number of females who listen to the station -/
def females_listen (total_females : ℕ) : ℕ := total_females - females_dont_listen

/-- Proof that the number of females who do not listen to the station is 87 -/
theorem females_dont_listen_correct 
  (total_peoples_total : ℕ)
  (males_dont_listen : ℕ)
  (total_females := total_females_surveyed total_peoples_total males_dont_listen)
  (females_listen := females_listen total_females) :
  females_dont_listen = 87 :=
sorry

end females_dont_listen_correct_l126_126112


namespace union_complement_eq_l126_126343

open Set

def U : Set ℕ := {1, 2, 3}
def A : Set ℕ := {1}
def B : Set ℕ := {1, 2}

theorem union_complement_eq : A ∪ (U \ B) = {1, 3} := by
  sorry

end union_complement_eq_l126_126343


namespace problem1_problem2_problem3_problem4_l126_126288

-- Statement for problem 1
theorem problem1 : -12 + (-6) - (-28) = 10 :=
  by sorry

-- Statement for problem 2
theorem problem2 : (-8 / 5) * (15 / 4) / (-9) = 2 / 3 :=
  by sorry

-- Statement for problem 3
theorem problem3 : (-3 / 16 - 7 / 24 + 5 / 6) * (-48) = -17 :=
  by sorry

-- Statement for problem 4
theorem problem4 : -3^2 + (7 / 8 - 1) * (-2)^2 = -9.5 :=
  by sorry

end problem1_problem2_problem3_problem4_l126_126288


namespace negation_of_existence_l126_126234

theorem negation_of_existence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) :=
by
  sorry

end negation_of_existence_l126_126234


namespace four_digit_number_conditions_l126_126247

-- Define the needed values based on the problem conditions
def first_digit := 1
def second_digit := 3
def third_digit := 4
def last_digit := 9

def number := 1349

-- State the theorem
theorem four_digit_number_conditions :
  (second_digit = 3 * first_digit) ∧ 
  (last_digit = 3 * second_digit) ∧ 
  (number = 1349) :=
by
  -- This is where the proof would go
  sorry

end four_digit_number_conditions_l126_126247


namespace recommendation_plans_count_l126_126982

def num_male : ℕ := 3
def num_female : ℕ := 2
def num_recommendations : ℕ := 5

def num_spots_russian : ℕ := 2
def num_spots_japanese : ℕ := 2
def num_spots_spanish : ℕ := 1

def condition_russian (males : ℕ) : Prop := males > 0
def condition_japanese (males : ℕ) : Prop := males > 0

theorem recommendation_plans_count : 
  (∃ (males_r : ℕ) (males_j : ℕ), condition_russian males_r ∧ condition_japanese males_j ∧ 
  num_male - males_r - males_j >= 0 ∧ males_r + males_j ≤ num_male ∧ 
  num_female + (num_male - males_r - males_j) >= num_recommendations - (num_spots_russian + num_spots_japanese + num_spots_spanish)) →
  (∃ (x : ℕ), x = 24) := by
  sorry

end recommendation_plans_count_l126_126982


namespace angle_AA1_BD1_angle_BD1_DC1_angle_AD1_DC1_l126_126026

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  p1 : Point
  p2 : Point

def length (l : Line) : ℝ :=
  Real.sqrt ((l.p2.x - l.p1.x)^2 + (l.p2.y - l.p1.y)^2 + (l.p2.z - l.p1.z)^2)

def dot_product (l1 l2 : Line) : ℝ :=
  (l1.p2.x - l1.p1.x) * (l2.p2.x - l2.p1.x) +
  (l1.p2.y - l1.p1.y) * (l2.p2.y - l2.p1.y) +
  (l1.p2.z - l1.p1.z) * (l2.p2.z - l2.p1.z)

def angle_between_lines (l1 l2 : Line) : ℝ :=
  Real.arccos ((dot_product l1 l2) / (length l1 * length l2))

def cube_edge_length (l : ℝ) : Prop :=
  l > 0

variable (a : ℝ)

def pointA : Point := { x := 0, y := 0, z := 0 }
def pointB : Point := { x := a, y := 0, z := 0 }
def pointC : Point := { x := a, y := a, z := 0 }
def pointD : Point := { x := 0, y := a, z := 0 }
def pointA1 : Point := { x := 0, y := 0, z := a }
def pointB1 : Point := { x := a, y := 0, z := a }
def pointC1 : Point := { x := a, y := a, z := a }
def pointD1 : Point := { x := 0, y := a, z := a }

def lineAA1 : Line := { p1 := pointA, p2 := pointA1 }
def lineBD1 : Line := { p1 := pointB, p2 := pointD1 }
def lineDC1 : Line := { p1 := pointD, p2 := pointC1 }
def lineAD1 : Line := { p1 := pointA, p2 := pointD1 }

theorem angle_AA1_BD1 :
  angle_between_lines lineAA1 lineBD1 = Real.arccos (1 / Real.sqrt 3) :=
sorry

theorem angle_BD1_DC1 :
  angle_between_lines lineBD1 lineDC1 = Real.pi / 2 :=
sorry

theorem angle_AD1_DC1 :
  angle_between_lines lineAD1 lineDC1 = Real.pi / 3 :=
sorry

end angle_AA1_BD1_angle_BD1_DC1_angle_AD1_DC1_l126_126026


namespace charlie_has_largest_final_answer_l126_126728

theorem charlie_has_largest_final_answer :
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  charlie > alice ∧ charlie > bob :=
by
  -- Definitions of intermediate variables
  let alice := (15 - 2)^2 + 3
  let bob := 15^2 - 2 + 3
  let charlie := (15 - 2 + 3)^2
  -- Comparison assertions
  sorry

end charlie_has_largest_final_answer_l126_126728


namespace part_I_part_II_l126_126077

def S_n (n : ℕ) : ℕ := sorry
def a_n (n : ℕ) : ℕ := sorry

theorem part_I (n : ℕ) (h1 : 2 * S_n n = 3^n + 3) :
  a_n n = if n = 1 then 3 else 3^(n-1) :=
sorry

theorem part_II (n : ℕ) (h1 : a_n 1 = 1) (h2 : ∀ n : ℕ, a_n (n + 1) - a_n n = 2^n) :
  S_n n = 2^(n + 1) - n - 2 :=
sorry

end part_I_part_II_l126_126077


namespace necessary_but_not_sufficient_condition_for_purely_imaginary_l126_126258

theorem necessary_but_not_sufficient_condition_for_purely_imaginary (m : ℂ) :
  (1 - m^2 + (1 + m) * Complex.I = 0 → m = 1) ∧ 
  ((1 - m^2 + (1 + m) * Complex.I = 0 ↔ m = 1) = false) := by
  sorry

end necessary_but_not_sufficient_condition_for_purely_imaginary_l126_126258


namespace alice_paper_cranes_l126_126727

theorem alice_paper_cranes (T : ℕ)
  (h1 : T / 2 - T / 10 = 400) : T = 1000 :=
sorry

end alice_paper_cranes_l126_126727


namespace tan_alpha_trigonometric_expression_l126_126171

variable (α : ℝ)
variable (h1 : Real.sin (Real.pi + α) = 3 / 5)
variable (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2)

theorem tan_alpha (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : Real.tan α = 3 / 4 := 
sorry

theorem trigonometric_expression (h1 : Real.sin (Real.pi + α) = 3 / 5) (h2 : Real.pi < α ∧ α < 3 * Real.pi / 2) : 
  (Real.sin ((Real.pi + α) / 2) - Real.cos ((Real.pi + α) / 2)) / 
  (Real.sin ((Real.pi - α) / 2) - Real.cos ((Real.pi - α) / 2)) = -1 / 2 := 
sorry

end tan_alpha_trigonometric_expression_l126_126171


namespace sqrt_90000_eq_300_l126_126090

theorem sqrt_90000_eq_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_eq_300_l126_126090


namespace eval_expression_l126_126318

theorem eval_expression : (2^2 - 2) - (3^2 - 3) + (4^2 - 4) - (5^2 - 5) + (6^2 - 6) = 18 :=
by
  sorry

end eval_expression_l126_126318


namespace pq_sum_is_38_l126_126473

theorem pq_sum_is_38
  (p q : ℝ)
  (h_root : ∀ x, (2 * x^2) + (p * x) + q = 0 → x = 2 * Complex.I - 3 ∨ x = -2 * Complex.I - 3)
  (h_p_q : ∀ a b : ℂ, a + b = -p / 2 ∧ a * b = q / 2 → p = 12 ∧ q = 26) :
  p + q = 38 :=
sorry

end pq_sum_is_38_l126_126473


namespace vector_at_t5_l126_126576

theorem vector_at_t5 :
  ∃ (a : ℝ × ℝ × ℝ) (d : ℝ × ℝ × ℝ),
    a + (1 : ℝ) • d = (2, -1, 3) ∧
    a + (4 : ℝ) • d = (8, -5, 11) ∧
    a + (5 : ℝ) • d = (10, -19/3, 41/3) := 
sorry

end vector_at_t5_l126_126576


namespace f_g_relationship_l126_126353

def f (x : ℝ) : ℝ := 3 * x ^ 2 - x + 1
def g (x : ℝ) : ℝ := 2 * x ^ 2 + x - 1

theorem f_g_relationship (x : ℝ) : f x > g x :=
by
  -- proof goes here
  sorry

end f_g_relationship_l126_126353


namespace maximum_weekly_hours_l126_126213

-- Conditions
def regular_rate : ℝ := 8 -- $8 per hour for the first 20 hours
def overtime_rate : ℝ := regular_rate * 1.25 -- 25% higher than the regular rate
def max_weekly_earnings : ℝ := 460 -- Maximum of $460 in a week
def regular_hours : ℕ := 20 -- First 20 hours are regular hours
def regular_earnings : ℝ := regular_hours * regular_rate -- Earnings for regular hours
def max_overtime_earnings : ℝ := max_weekly_earnings - regular_earnings -- Maximum overtime earnings

-- Proof problem statement
theorem maximum_weekly_hours : regular_hours + (max_overtime_earnings / overtime_rate) = 50 := by
  sorry

end maximum_weekly_hours_l126_126213


namespace gcd_lcm_sum_8_12_l126_126898

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l126_126898


namespace no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l126_126008

theorem no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0 :
  ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
sorry

end no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l126_126008


namespace number_of_diagonals_30_sides_l126_126351

def number_of_diagonals (n : ℕ) : ℕ :=
  nat.choose n 2 - n

theorem number_of_diagonals_30_sides :
  number_of_diagonals 30 = 405 :=
by {
  rw [number_of_diagonals, nat.choose, nat.factorial, nat.factorial, nat.factorial],
  -- The proof would proceed to simplify the combination and subtraction, but we use sorry to skip details.
  sorry,
}

end number_of_diagonals_30_sides_l126_126351


namespace max_teams_l126_126810

theorem max_teams (n : ℕ) (cond1 : ∀ t, card t = 3) (cond2 : ∀ t1 t2 (ht1 : t1 ≠ t2), 
  ∀ p1 ∈ t1, ∀ p2 ∈ t2, p1 ≠ p2) (cond3 : 9 * n * (n - 1) / 2 ≤ 200) : 
  n ≤ 7 := 
sorry

end max_teams_l126_126810


namespace percentage_of_divisible_l126_126695

def count_divisible (n m : ℕ) : ℕ :=
(n / m)

def calculate_percentage (part total : ℕ) : ℚ :=
(part * 100 : ℚ) / (total : ℚ)

theorem percentage_of_divisible (n : ℕ) (k : ℕ) (h₁ : n = 150) (h₂ : k = 6) :
  calculate_percentage (count_divisible n k) n = 16.67 :=
by
  sorry

end percentage_of_divisible_l126_126695


namespace usamo_2003_q3_l126_126511

open Real

theorem usamo_2003_q3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2)
  + (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2)
  + (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2) ) ≤ 8 := 
sorry

end usamo_2003_q3_l126_126511


namespace freddy_spent_10_dollars_l126_126754

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end freddy_spent_10_dollars_l126_126754


namespace trigonometric_identity_l126_126025

theorem trigonometric_identity (α : ℝ)
 (h : Real.sin (α / 2) - 2 * Real.cos (α / 2) = 1) :
  (1 + Real.sin α + Real.cos α) / (1 + Real.sin α - Real.cos α) = 3 / 4 := 
sorry

end trigonometric_identity_l126_126025


namespace number_of_diagonals_in_30_sided_polygon_l126_126350

-- Definition of a convex polygon with 30 sides
def convex_polygon_sides := 30

-- The function to calculate the number of diagonals in a convex polygon with n sides
noncomputable def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

-- Prove that a convex polygon with 30 sides has 405 diagonals
theorem number_of_diagonals_in_30_sided_polygon :
  num_diagonals convex_polygon_sides = 405 :=
by
  calc
    num_diagonals 30 = 30 * (30 - 3) / 2 : rfl
    ... = 30 * 27 / 2 : rfl
    ... = 810 / 2 : rfl
    ... = 405 : rfl

end number_of_diagonals_in_30_sided_polygon_l126_126350


namespace q1_q2_l126_126468

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end q1_q2_l126_126468


namespace stratified_sampling_vision_test_l126_126538

theorem stratified_sampling_vision_test 
  (n_total : ℕ) (n_HS : ℕ) (n_selected : ℕ)
  (h1 : n_total = 165)
  (h2 : n_HS = 66)
  (h3 : n_selected = 15) :
  (n_HS * n_selected / n_total) = 6 := 
by 
  sorry

end stratified_sampling_vision_test_l126_126538


namespace work_together_time_l126_126266

theorem work_together_time (man_days : ℝ) (son_days : ℝ)
  (h_man : man_days = 5) (h_son : son_days = 7.5) :
  (1 / (1 / man_days + 1 / son_days)) = 3 :=
by
  -- Given the constraints, prove the result
  rw [h_man, h_son]
  sorry

end work_together_time_l126_126266


namespace value_of_5_T_3_l126_126596

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l126_126596


namespace isosceles_triangle_base_angle_l126_126376

theorem isosceles_triangle_base_angle (A B C : ℝ) (h_sum : A + B + C = 180) (h_iso : B = C) (h_one_angle : A = 80) : B = 50 :=
sorry

end isosceles_triangle_base_angle_l126_126376


namespace factorize_polynomial_l126_126606

theorem factorize_polynomial (a b : ℝ) : a^2 - 9 * b^2 = (a + 3 * b) * (a - 3 * b) := by
  sorry

end factorize_polynomial_l126_126606


namespace larger_of_two_numbers_l126_126415

theorem larger_of_two_numbers (H : Nat := 15) (f1 : Nat := 11) (f2 : Nat := 15) :
  let lcm := H * f1 * f2;
  ∃ (A B : Nat), A = H * f1 ∧ B = H * f2 ∧ A ≤ B := by
  sorry

end larger_of_two_numbers_l126_126415


namespace geom_seq_sum_l126_126363

theorem geom_seq_sum {a : ℕ → ℝ} (q : ℝ) (h1 : a 0 + a 1 + a 2 = 2)
    (h2 : a 3 + a 4 + a 5 = 16)
    (h_geom : ∀ n, a (n + 1) = q * a n) :
  a 6 + a 7 + a 8 = 128 :=
sorry

end geom_seq_sum_l126_126363


namespace factorization_correct_l126_126012

theorem factorization_correct (x : ℝ) : 
  98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) :=
by
  sorry

end factorization_correct_l126_126012


namespace greatest_difference_l126_126853

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l126_126853


namespace angle_of_inclination_l126_126031

theorem angle_of_inclination (m : ℝ) (h : m = -1) : 
  ∃ α : ℝ, α = 3 * Real.pi / 4 := 
sorry

end angle_of_inclination_l126_126031


namespace max_value_ratio_l126_126509

/-- Define the conditions on function f and variables x and y. -/
def conditions (f : ℝ → ℝ) (x y : ℝ) :=
  (∀ x, f (-x) + f x = 0) ∧
  (∀ x1 x2, x1 < x2 → f x1 < f x2) ∧
  f (x^2 - 6 * x) + f (y^2 - 4 * y + 12) ≤ 0

/-- The maximum value of (y - 2) / x under the given conditions. -/
theorem max_value_ratio (f : ℝ → ℝ) (x y : ℝ) (cond : conditions f x y) :
  (y - 2) / x ≤ (Real.sqrt 2) / 4 :=
sorry

end max_value_ratio_l126_126509


namespace messages_after_noon_l126_126953

theorem messages_after_noon (t n : ℕ) (h1 : t = 39) (h2 : n = 21) : t - n = 18 := by
  sorry

end messages_after_noon_l126_126953


namespace find_x_l126_126989

theorem find_x (x : ℕ) (h : 5 * x + 4 * x + x + 2 * x = 360) : x = 30 :=
by
  sorry

end find_x_l126_126989


namespace remainder_of_concatenated_number_l126_126657

def concatenated_number : ℕ :=
  -- Definition of the concatenated number
  -- That is 123456789101112...4344
  -- For simplicity, we'll just assign it directly
  1234567891011121314151617181920212223242526272829303132333435363738394041424344

theorem remainder_of_concatenated_number :
  concatenated_number % 45 = 9 :=
sorry

end remainder_of_concatenated_number_l126_126657


namespace distinct_prime_factors_of_90_l126_126780

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l126_126780


namespace square_area_when_a_eq_b_eq_c_l126_126816

theorem square_area_when_a_eq_b_eq_c {a b c : ℝ} (h : a = b ∧ b = c) :
  ∃ x : ℝ, (x = a * Real.sqrt 2) ∧ (x ^ 2 = 2 * a ^ 2) :=
by
  sorry

end square_area_when_a_eq_b_eq_c_l126_126816


namespace joan_change_received_l126_126367

/-- Definition of the cat toy cost -/
def cat_toy_cost : ℝ := 8.77

/-- Definition of the cage cost -/
def cage_cost : ℝ := 10.97

/-- Definition of the total cost -/
def total_cost : ℝ := cat_toy_cost + cage_cost

/-- Definition of the payment amount -/
def payment : ℝ := 20.00

/-- Definition of the change received -/
def change_received : ℝ := payment - total_cost

/-- Statement proving that Joan received $0.26 in change -/
theorem joan_change_received : change_received = 0.26 := by
  sorry

end joan_change_received_l126_126367


namespace ceil_x_pow_2_values_l126_126049

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l126_126049


namespace imo_1988_problem_29_l126_126626

variable (d r : ℕ)
variable (h1 : d > 1)
variable (h2 : 1059 % d = r)
variable (h3 : 1417 % d = r)
variable (h4 : 2312 % d = r)

theorem imo_1988_problem_29 :
  d - r = 15 := by sorry

end imo_1988_problem_29_l126_126626


namespace rowing_upstream_speed_l126_126420

theorem rowing_upstream_speed 
  (V_m : ℝ) (V_downstream : ℝ) (V_upstream : ℝ)
  (hyp1 : V_m = 30)
  (hyp2 : V_downstream = 35) :
  V_upstream = V_m - (V_downstream - V_m) := 
  sorry

end rowing_upstream_speed_l126_126420


namespace iggy_pace_l126_126495

theorem iggy_pace 
  (monday_miles : ℕ) (tuesday_miles : ℕ) (wednesday_miles : ℕ)
  (thursday_miles : ℕ) (friday_miles : ℕ) (total_hours : ℕ) 
  (h1 : monday_miles = 3) (h2 : tuesday_miles = 4) 
  (h3 : wednesday_miles = 6) (h4 : thursday_miles = 8) 
  (h5 : friday_miles = 3) (h6 : total_hours = 4) :
  (total_hours * 60) / (monday_miles + tuesday_miles + wednesday_miles + thursday_miles + friday_miles) = 10 :=
sorry

end iggy_pace_l126_126495


namespace museum_college_students_income_l126_126061

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l126_126061


namespace fleas_difference_l126_126715

-- Define the initial number of fleas and subsequent fleas after each treatment.
def initial_fleas (F : ℝ) := F
def after_first_treatment (F : ℝ) := F * 0.40
def after_second_treatment (F : ℝ) := (after_first_treatment F) * 0.55
def after_third_treatment (F : ℝ) := (after_second_treatment F) * 0.70
def after_fourth_treatment (F : ℝ) := (after_third_treatment F) * 0.80

-- Given condition
axiom final_fleas : initial_fleas 20 = after_fourth_treatment 20

-- Prove the number of fleas before treatment minus the number after treatment is 142
theorem fleas_difference (F : ℝ) (h : initial_fleas F = after_fourth_treatment 20) : 
  F - 20 = 142 :=
by {
  sorry
}

end fleas_difference_l126_126715


namespace count_paths_l126_126998

-- Define the lattice points and paths
def isLatticePoint (P : ℤ × ℤ) : Prop := true
def isLatticePath (P : ℕ → ℤ × ℤ) (n : ℕ) : Prop :=
  (∀ i, 0 < i → i ≤ n → abs ((P i).1 - (P (i - 1)).1) + abs ((P i).2 - (P (i - 1)).2) = 1)

-- Define F(n) with the given constraints
def numberOfPaths (n : ℕ) : ℕ :=
  -- Placeholder for the actual complex counting logic, which is not detailed here
  sorry

-- Identify F(n) from the initial conditions and the correct result
theorem count_paths (n : ℕ) :
  numberOfPaths n = Nat.choose (2 * n) n :=
sorry

end count_paths_l126_126998


namespace min_cost_proof_l126_126380

-- Define the costs and servings for each ingredient
def pasta_cost : ℝ := 1.12
def pasta_servings_per_box : ℕ := 5

def meatballs_cost : ℝ := 5.24
def meatballs_servings_per_pack : ℕ := 4

def tomato_sauce_cost : ℝ := 2.31
def tomato_sauce_servings_per_jar : ℕ := 5

def tomatoes_cost : ℝ := 1.47
def tomatoes_servings_per_pack : ℕ := 4

def lettuce_cost : ℝ := 0.97
def lettuce_servings_per_head : ℕ := 6

def olives_cost : ℝ := 2.10
def olives_servings_per_jar : ℕ := 8

def cheese_cost : ℝ := 2.70
def cheese_servings_per_block : ℕ := 7

-- Define the number of people to serve
def number_of_people : ℕ := 8

-- The total cost calculated
def total_cost : ℝ := 
  (2 * pasta_cost) +
  (2 * meatballs_cost) +
  (2 * tomato_sauce_cost) +
  (2 * tomatoes_cost) +
  (2 * lettuce_cost) +
  (1 * olives_cost) +
  (2 * cheese_cost)

-- The minimum total cost
def min_total_cost : ℝ := 29.72

theorem min_cost_proof : total_cost = min_total_cost :=
by sorry

end min_cost_proof_l126_126380


namespace solve_y_l126_126315

theorem solve_y (x y : ℤ) (h₁ : x = 3) (h₂ : x^3 - x - 2 = y + 2) : y = 20 :=
by
  -- Proof goes here
  sorry

end solve_y_l126_126315


namespace abs_sum_lt_ineq_l126_126804

theorem abs_sum_lt_ineq (x : ℝ) (a : ℝ) (h₀ : 0 < a) : 
  (∃ x : ℝ, |x - 4| + |x - 3| < a) ↔ (1 < a) :=
by
  sorry

end abs_sum_lt_ineq_l126_126804


namespace find_value_of_p_l126_126364

theorem find_value_of_p (p q r s t u v w : ℤ)
  (h1 : r + s = -2)
  (h2 : s + (-2) = 5)
  (h3 : t + u = 5)
  (h4 : u + v = 3)
  (h5 : v + w = 8)
  (h6 : w + t = 3)
  (h7 : q + r = s)
  (h8 : p + q = r) :
  p = -25 := by
  -- proof skipped
  sorry

end find_value_of_p_l126_126364


namespace william_washed_2_normal_cars_l126_126563

def time_spent_on_one_normal_car : Nat := 4 + 7 + 4 + 9

def time_spent_on_suv : Nat := 2 * time_spent_on_one_normal_car

def total_time_spent : Nat := 96

def time_spent_on_normal_cars : Nat := total_time_spent - time_spent_on_suv

def number_of_normal_cars : Nat := time_spent_on_normal_cars / time_spent_on_one_normal_car

theorem william_washed_2_normal_cars : number_of_normal_cars = 2 := by
  sorry

end william_washed_2_normal_cars_l126_126563


namespace find_denominator_l126_126932

noncomputable def original_denominator (d : ℝ) : Prop :=
  (7 / (d + 3)) = 2 / 3

theorem find_denominator : ∃ d : ℝ, original_denominator d ∧ d = 7.5 :=
by
  use 7.5
  unfold original_denominator
  sorry

end find_denominator_l126_126932


namespace xy_square_value_l126_126354

theorem xy_square_value (x y : ℝ) (h1 : x * (x + y) = 24) (h2 : y * (x + y) = 72) : (x + y)^2 = 96 :=
by
  sorry

end xy_square_value_l126_126354


namespace correct_option_b_l126_126911

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l126_126911


namespace sum_of_GCF_and_LCM_l126_126886

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l126_126886


namespace avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l126_126723

open Real
open List

-- Conditions
def x_vals : List ℝ := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
def y_vals : List ℝ := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
def sum_x : ℝ := 0.6
def sum_y : ℝ := 3.9
def sum_x_squared : ℝ := 0.038
def sum_y_squared : ℝ := 1.6158
def sum_xy : ℝ := 0.2474
def total_root_area : ℝ := 186

-- Proof problems
theorem avg_root_area : (List.sum x_vals / 10) = 0.06 := by
  sorry

theorem avg_volume : (List.sum y_vals / 10) = 0.39 := by
  sorry

theorem correlation_coefficient : 
  let mean_x := List.sum x_vals / 10;
  let mean_y := List.sum y_vals / 10;
  let numerator := List.sum (List.zipWith (λ x y => (x - mean_x) * (y - mean_y)) x_vals y_vals);
  let denominator := sqrt ((List.sum (List.map (λ x => (x - mean_x) ^ 2) x_vals)) * (List.sum (List.map (λ y => (y - mean_y) ^ 2) y_vals)));
  (numerator / denominator) = 0.97 := by 
  sorry

theorem total_volume_estimate : 
  let avg_x := sum_x / 10;
  let avg_y := sum_y / 10;
  (avg_y / avg_x) * total_root_area = 1209 := by
  sorry

end avg_root_area_avg_volume_correlation_coefficient_total_volume_estimate_l126_126723


namespace max_apartment_size_l126_126434

/-- Define the rental rate and the maximum rent Michael can afford. -/
def rental_rate : ℝ := 1.20
def max_rent : ℝ := 720

/-- State the problem in Lean: Prove that the maximum apartment size Michael should consider is 600 square feet. -/
theorem max_apartment_size :
  ∃ s : ℝ, rental_rate * s = max_rent ∧ s = 600 := by
  sorry

end max_apartment_size_l126_126434


namespace compute_expression_l126_126592

theorem compute_expression : (88 * 707 - 38 * 707) / 1414 = 25 :=
by
  sorry

end compute_expression_l126_126592


namespace sin_180_eq_zero_l126_126305

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l126_126305


namespace freddy_spent_10_dollars_l126_126753

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end freddy_spent_10_dollars_l126_126753


namespace equilateral_triangle_lines_l126_126311

-- Define the properties of an equilateral triangle
structure EquilateralTriangle :=
(sides_length : ℝ) -- All sides are of equal length
(angle : ℝ := 60)  -- All internal angles are 60 degrees

-- Define the concept that altitudes, medians, and angle bisectors coincide
structure CoincidingLines (T : EquilateralTriangle) :=
(altitude : T.angle = 60)
(median : T.angle = 60)
(angle_bisector : T.angle = 60)

-- Define a statement that proves the number of distinct lines in the equilateral triangle
theorem equilateral_triangle_lines (T : EquilateralTriangle) (L : CoincidingLines T) :  
  -- The total number of distinct lines consisting of altitudes, medians, and angle bisectors
  (3 = 3) :=
by
  sorry

end equilateral_triangle_lines_l126_126311


namespace total_leaves_l126_126995

def fernTypeA_fronds := 15
def fernTypeA_leaves_per_frond := 45
def fernTypeB_fronds := 20
def fernTypeB_leaves_per_frond := 30
def fernTypeC_fronds := 25
def fernTypeC_leaves_per_frond := 40

def fernTypeA_count := 4
def fernTypeB_count := 5
def fernTypeC_count := 3

theorem total_leaves : 
  fernTypeA_count * (fernTypeA_fronds * fernTypeA_leaves_per_frond) + 
  fernTypeB_count * (fernTypeB_fronds * fernTypeB_leaves_per_frond) + 
  fernTypeC_count * (fernTypeC_fronds * fernTypeC_leaves_per_frond) = 
  8700 := 
sorry

end total_leaves_l126_126995


namespace confidence_relationship_l126_126244
noncomputable def K_squared : ℝ := 3.918
noncomputable def critical_value : ℝ := 3.841
noncomputable def p_val : ℝ := 0.05

theorem confidence_relationship (K_squared : ℝ) (critical_value : ℝ) (p_val : ℝ) :
  K_squared ≥ critical_value -> p_val = 0.05 ->
  1 - p_val = 0.95 :=
by
  sorry

end confidence_relationship_l126_126244


namespace find_last_number_l126_126227

theorem find_last_number
  (A B C D : ℝ)
  (h1 : (A + B + C) / 3 = 6)
  (h2 : (B + C + D) / 3 = 5)
  (h3 : A + D = 11) :
  D = 4 :=
by
  sorry

end find_last_number_l126_126227


namespace arithmetic_sequence_length_correct_l126_126798

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l126_126798


namespace product_of_roots_eq_neg25_l126_126321

theorem product_of_roots_eq_neg25 : 
  ∀ (x : ℝ), 24 * x^2 + 36 * x - 600 = 0 → x * (x - ((-36 - 24 * x)/24)) = -25 :=
by
  sorry

end product_of_roots_eq_neg25_l126_126321


namespace circle_equation_l126_126238

/-
  Prove that the standard equation for the circle passing through points
  A(-6, 0), B(0, 2), and the origin O(0, 0) is (x+3)^2 + (y-1)^2 = 10.
-/
theorem circle_equation :
  ∃ (x y : ℝ), x = -6 ∨ x = 0 ∨ x = 0 ∧ y = 0 ∨ y = 2 ∨ y = 0 → (∀ P : ℝ × ℝ, P = (-6, 0) ∨ P = (0, 2) ∨ P = (0, 0) → (P.1 + 3)^2 + (P.2 - 1)^2 = 10) := 
sorry

end circle_equation_l126_126238


namespace num_ways_pay_l126_126369

theorem num_ways_pay : 
  let n : ℕ := 2010,
      num_solutions : ℕ := (Nat.choose (201 + 2) 2)
  in ∑ x y z in { (x' : ℕ) }, 2 * x' + 5 * y' + 10 * z' = 2010 = num_solutions :=
by
  sorry

end num_ways_pay_l126_126369


namespace peanuts_in_box_after_addition_l126_126918

theorem peanuts_in_box_after_addition : 4 + 12 = 16 := by
  sorry

end peanuts_in_box_after_addition_l126_126918


namespace triangle_has_at_most_one_obtuse_angle_l126_126252

-- Definitions
def Triangle (α β γ : ℝ) : Prop :=
  α + β + γ = 180

def Obtuse_angle (angle : ℝ) : Prop :=
  angle > 90

def Two_obtuse_angles (α β γ : ℝ) : Prop :=
  Obtuse_angle α ∧ Obtuse_angle β

-- Theorem Statement
theorem triangle_has_at_most_one_obtuse_angle (α β γ : ℝ) (h_triangle : Triangle α β γ) :
  ¬ Two_obtuse_angles α β γ := 
sorry

end triangle_has_at_most_one_obtuse_angle_l126_126252


namespace odd_function_example_l126_126180

theorem odd_function_example (f : ℝ → ℝ)
    (h_odd : ∀ x, f (-x) = -f x)
    (h_neg : ∀ x, x < 0 → f x = x + 2) : f 0 + f 3 = 1 :=
by
  sorry

end odd_function_example_l126_126180


namespace arithmetic_sequence_a_common_terms_C_exists_m_n_l126_126506

def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) := ∀ n, a (n + 1) - a n = d

def a_n (n : ℕ) : ℤ := 2 * n - 7
def b_n (n : ℕ) : ℤ := 3 * n - 11
def C_n (n : ℕ) : ℤ := 6 * n + 1

theorem arithmetic_sequence_a (a : ℕ → ℤ) (d : ℤ) (h : is_arithmetic_sequence a d) 
  (h₁ : a 6 = 5) (h₂ : a 2 ^ 2 + a 3 ^ 2 = a 4 ^ 2 + a 5 ^ 2) : 
  ∀ n, a n = 2 * n - 7 := sorry

theorem common_terms_C :
  ∀ n, C_n n = 6 * n + 1 := sorry

theorem exists_m_n (m n : ℕ) (h₁ : m ≠ 5) (h₂ : n ≠ 5) (h₃ : m ≠ n) :
  (m = 11 ∧ n = 1) ∨ (m = 2 ∧ n = 3) ∨ (m = 6 ∧ n = 11) :=
  sorry

end arithmetic_sequence_a_common_terms_C_exists_m_n_l126_126506


namespace aubree_animals_total_l126_126938

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l126_126938


namespace find_m_l126_126465

noncomputable def given_hyperbola (x y : ℝ) (m : ℝ) : Prop :=
    x^2 / m - y^2 / 3 = 1

noncomputable def hyperbola_eccentricity (m : ℝ) (e : ℝ) : Prop :=
    e = Real.sqrt (1 + 3 / m)

theorem find_m (m : ℝ) (h1 : given_hyperbola 1 1 m) (h2 : hyperbola_eccentricity m 2) : m = 1 :=
by
  sorry

end find_m_l126_126465


namespace angle_between_hands_at_3_15_l126_126038

theorem angle_between_hands_at_3_15 :
  let hour_angle_at_3 := 3 * 30
  let hour_hand_move_rate := 0.5
  let minute_angle := 15 * 6
  let hour_angle_at_3_15 := hour_angle_at_3 + 15 * hour_hand_move_rate
  abs (hour_angle_at_3_15 - minute_angle) = 7.5 := 
by
  sorry

end angle_between_hands_at_3_15_l126_126038


namespace solve_quadratic_l126_126518

-- Problem Definition
def quadratic_equation (x : ℝ) : Prop :=
  2 * x^2 - 6 * x + 3 = 0

-- Solution Definition
def solution1 (x : ℝ) : Prop :=
  x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2

-- Lean Theorem Statement
theorem solve_quadratic : ∀ x : ℝ, quadratic_equation x ↔ solution1 x :=
sorry

end solve_quadratic_l126_126518


namespace solve_quadratic_eq_solve_linear_system_l126_126704

theorem solve_quadratic_eq (x : ℚ) : 4 * (x - 1) ^ 2 - 25 = 0 ↔ x = 7 / 2 ∨ x = -3 / 2 := 
by sorry

theorem solve_linear_system (x y : ℚ) : (2 * x - y = 4) ∧ (3 * x + 2 * y = 1) ↔ (x = 9 / 7 ∧ y = -10 / 7) :=
by sorry

end solve_quadratic_eq_solve_linear_system_l126_126704


namespace lucas_seq_units_digit_M47_l126_126156

def lucas_seq : ℕ → ℕ := 
  sorry -- skipped sequence generation for brevity

def M (n : ℕ) : ℕ :=
  if n = 0 then 3 else
  if n = 1 then 1 else
  lucas_seq n -- will call the lucas sequence generator

-- Helper function to get the units digit of a number
def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem lucas_seq_units_digit_M47 : units_digit (M (M 6)) = 3 := 
sorry

end lucas_seq_units_digit_M47_l126_126156


namespace center_of_large_hexagon_within_small_hexagon_l126_126198

-- Define a structure for a regular hexagon with the necessary properties
structure RegularHexagon (α : Type) [LinearOrderedField α] :=
  (center : α × α)      -- Coordinates of the center
  (side_length : α)      -- Length of the side

-- Define the conditions: two regular hexagons with specific side length relationship
variables {α : Type} [LinearOrderedField α]
def hexagon_large : RegularHexagon α := 
  {center := (0, 0), side_length := 2}

def hexagon_small : RegularHexagon α := 
  {center := (0, 0), side_length := 1}

-- The theorem to prove
theorem center_of_large_hexagon_within_small_hexagon (hl : RegularHexagon α) (hs : RegularHexagon α) 
  (hc : hs.side_length = hl.side_length / 2) : (hl.center = hs.center) → 
  (∀ (x y : α × α), x = hs.center → (∃ r, y = hl.center → (y.1 - x.1) ^ 2 + (y.2 - x.2) ^ 2 < r ^ 2)) :=
by sorry

end center_of_large_hexagon_within_small_hexagon_l126_126198


namespace minimum_total_distance_l126_126523

-- Conditions:
def point (α : Type) := (α × α)
def distance (p1 p2 : point ℝ) : ℝ := 
  float.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)
def vertex := point ℝ

variables (A B C : vertex)
axiom AB_eq_3 : distance A B = 3
axiom AC_eq_2 : distance A C = 2
axiom BC_eq_sqrt_7 : distance B C = float.sqrt 7
axiom warehouse_pos : vertex -- Assuming existence but not fixing the actual position

-- Question == Answer
theorem minimum_total_distance (warehouse : vertex) 
  (dA := distance warehouse A)
  (dB := distance warehouse B)
  (dC := distance warehouse C) :
  let total_distance := dA + dB + dC 
  in total_distance * 2 = 2 * float.sqrt 19 :=
by sorry

end minimum_total_distance_l126_126523


namespace base_p_prime_values_zero_l126_126729

theorem base_p_prime_values_zero :
  (∀ p : ℕ, p.Prime → 2008 * p^3 + 407 * p^2 + 214 * p + 226 = 243 * p^2 + 382 * p + 471 → False) :=
by
  sorry

end base_p_prime_values_zero_l126_126729


namespace problem_l126_126456

noncomputable def f : ℝ → ℝ := sorry -- Placeholder for the function definition

theorem problem :
  (∀ x, f (-x) = -f x) → -- f is odd
  (∀ x, f (x + 2) = -1 / f x) → -- Functional equation
  (∀ x, 0 < x ∧ x < 1 → f x = 3 ^ x) → -- Definition on interval (0,1)
  f (Real.log (54) / Real.log 3) = -3 / 2 := sorry

end problem_l126_126456


namespace range_of_m_l126_126483

theorem range_of_m (m : ℝ) : (¬ ∃ x : ℝ, 4 ^ x + 2 ^ (x + 1) + m = 0) → m ≥ 0 := 
by
  sorry

end range_of_m_l126_126483


namespace quadratic_ineq_solutions_l126_126610

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l126_126610


namespace cos_triple_angle_l126_126355

theorem cos_triple_angle
  (θ : ℝ)
  (h : Real.cos θ = 1/3) :
  Real.cos (3 * θ) = -23 / 27 :=
by
  sorry

end cos_triple_angle_l126_126355


namespace hyperbola_focal_length_l126_126770

def is_hyperbola (x y a : ℝ) : Prop := (x^2) / (a^2) - (y^2) = 1
def is_perpendicular_asymptote (slope_asymptote slope_line : ℝ) : Prop := slope_asymptote * slope_line = -1

theorem hyperbola_focal_length {a : ℝ} (h1 : is_hyperbola x y a)
  (h2 : is_perpendicular_asymptote (1 / a) (-1)) : 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
sorry

end hyperbola_focal_length_l126_126770


namespace ceil_square_count_ceil_x_eq_15_l126_126045

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l126_126045


namespace train_crossing_time_l126_126934

noncomputable def relative_speed_kmh (speed_train : ℕ) (speed_man : ℕ) : ℕ := speed_train + speed_man

noncomputable def kmh_to_mps (speed_kmh : ℕ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def crossing_time (length_train : ℕ) (speed_train_kmh : ℕ) (speed_man_kmh : ℕ) : ℝ :=
  let relative_speed_kmh := relative_speed_kmh speed_train_kmh speed_man_kmh
  let relative_speed_mps := kmh_to_mps relative_speed_kmh
  length_train / relative_speed_mps

theorem train_crossing_time :
  crossing_time 210 25 2 = 28 :=
  by
  sorry

end train_crossing_time_l126_126934


namespace max_possible_player_salary_l126_126580

theorem max_possible_player_salary (n : ℕ) (min_salary total_salary : ℕ) (num_players : ℕ) 
  (h1 : num_players = 24) 
  (h2 : min_salary = 20000) 
  (h3 : total_salary = 960000)
  (h4 : n = 23 * min_salary + 500000) 
  (h5 : 23 * min_salary + 500000 ≤ total_salary) 
  : n = total_salary :=
by {
  -- The proof will replace this sorry.
  sorry
}

end max_possible_player_salary_l126_126580


namespace num_blue_balls_l126_126929

theorem num_blue_balls (total_balls blue_balls : ℕ) 
  (prob_all_blue : ℚ)
  (h_total : total_balls = 12)
  (h_prob : prob_all_blue = 1 / 55)
  (h_prob_eq : (blue_balls / 12) * ((blue_balls - 1) / 11) * ((blue_balls - 2) / 10) = prob_all_blue) :
  blue_balls = 4 :=
by
  -- Placeholder for proof
  sorry

end num_blue_balls_l126_126929


namespace schoolchildren_chocolate_l126_126837

theorem schoolchildren_chocolate (m d : ℕ) 
  (h1 : 7 * d + 2 * m > 36)
  (h2 : 8 * d + 4 * m < 48) :
  m = 1 ∧ d = 5 :=
by
  sorry

end schoolchildren_chocolate_l126_126837


namespace solve_for_x_l126_126676

theorem solve_for_x :
  ∀ (x : ℚ), x = 45 / (8 - 3 / 7) → x = 315 / 53 :=
by
  sorry

end solve_for_x_l126_126676


namespace lemonade_stand_profit_is_66_l126_126145

def lemonade_stand_profit
  (lemons_cost : ℕ := 10)
  (sugar_cost : ℕ := 5)
  (cups_cost : ℕ := 3)
  (price_per_cup : ℕ := 4)
  (cups_sold : ℕ := 21) : ℕ :=
  (price_per_cup * cups_sold) - (lemons_cost + sugar_cost + cups_cost)

theorem lemonade_stand_profit_is_66 :
  lemonade_stand_profit = 66 :=
by
  unfold lemonade_stand_profit
  simp
  sorry

end lemonade_stand_profit_is_66_l126_126145


namespace distinct_prime_factors_90_l126_126783

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l126_126783


namespace area_relationship_l126_126269

theorem area_relationship (x β : ℝ) (hβ : 0.60 * x^2 = β) : α = (4 / 3) * β :=
by
  -- conditions and goal are stated
  let α := 0.80 * x^2
  sorry

end area_relationship_l126_126269


namespace green_ball_probability_l126_126397

/-
  There are four containers:
  - Container A holds 5 red balls and 7 green balls.
  - Container B holds 7 red balls and 3 green balls.
  - Container C holds 8 red balls and 2 green balls.
  - Container D holds 4 red balls and 6 green balls.
  The probability of choosing containers A, B, C, and D is 1/4 each.
-/

def prob_A : ℚ := 1 / 4
def prob_B : ℚ := 1 / 4
def prob_C : ℚ := 1 / 4
def prob_D : ℚ := 1 / 4

def prob_Given_A : ℚ := 7 / 12
def prob_Given_B : ℚ := 3 / 10
def prob_Given_C : ℚ := 1 / 5
def prob_Given_D : ℚ := 3 / 5

def total_prob_green : ℚ :=
  prob_A * prob_Given_A + prob_B * prob_Given_B +
  prob_C * prob_Given_C + prob_D * prob_Given_D

theorem green_ball_probability : total_prob_green = 101 / 240 := 
by
  -- here would normally be the proof steps, but we use sorry to skip it.
  sorry

end green_ball_probability_l126_126397


namespace find_f_sqrt_10_l126_126487

-- Definitions and conditions provided in the problem
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x
def is_periodic_function (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x
def f_condition (f : ℝ → ℝ) : Prop := ∀ x : ℝ, 0 < x ∧ x < 1 → f x = x^2 - 8*x + 30

-- The problem specific conditions for f
variable (f : ℝ → ℝ)
variable (h_odd : is_odd_function f)
variable (h_periodic : is_periodic_function f 2)
variable (h_condition : f_condition f)

-- The statement to prove
theorem find_f_sqrt_10 : f (Real.sqrt 10) = -24 :=
by
  sorry

end find_f_sqrt_10_l126_126487


namespace scissors_total_l126_126396

theorem scissors_total (original_scissors : ℕ) (added_scissors : ℕ) (total_scissors : ℕ) 
  (h1 : original_scissors = 39)
  (h2 : added_scissors = 13)
  (h3 : total_scissors = original_scissors + added_scissors) : total_scissors = 52 :=
by
  rw [h1, h2] at h3
  exact h3

end scissors_total_l126_126396


namespace convex_polygon_diagonals_l126_126347

theorem convex_polygon_diagonals (n : ℕ) (h : n = 30) : 
  let total_pairs := (n * (n - 1)) / 2 in (total_pairs - n) = 405 := 
by
  simp only [total_pairs]
  sorry

end convex_polygon_diagonals_l126_126347


namespace probability_of_females_right_of_males_l126_126537

-- Defining the total and favorable outcomes
def total_outcomes : ℕ := Nat.factorial 5
def favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial 2

-- Defining the probability as a rational number
def probability_all_females_right : ℚ := favorable_outcomes / total_outcomes

-- Stating the theorem
theorem probability_of_females_right_of_males :
  probability_all_females_right = 1 / 10 :=
by
  -- Proof to be filled in
  sorry

end probability_of_females_right_of_males_l126_126537


namespace find_x_l126_126641

theorem find_x (x : ℝ) 
  (h1 : x = (1 / x * -x) - 5) 
  (h2 : x^2 - 3 * x + 2 ≥ 0) : 
  x = -6 := 
sorry

end find_x_l126_126641


namespace curve_symmetric_about_y_eq_x_l126_126228

theorem curve_symmetric_about_y_eq_x (x y : ℝ) (h : x * y * (x + y) = 1) :
  (y * x * (y + x) = 1) :=
by
  sorry

end curve_symmetric_about_y_eq_x_l126_126228


namespace even_integer_operations_l126_126154

theorem even_integer_operations (k : ℤ) (a : ℤ) (h : a = 2 * k) :
  (a * 5) % 2 = 0 ∧ (a ^ 2) % 2 = 0 ∧ (a ^ 3) % 2 = 0 :=
by
  sorry

end even_integer_operations_l126_126154


namespace correct_option_b_l126_126912

theorem correct_option_b (a : ℝ) : (-2 * a ^ 4) ^ 3 = -8 * a ^ 12 :=
sorry

end correct_option_b_l126_126912


namespace sum_gcd_lcm_eight_twelve_l126_126874

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l126_126874


namespace buns_distribution_not_equal_for_all_cases_l126_126849

theorem buns_distribution_not_equal_for_all_cases :
  ∀ (initial_buns : Fin 30 → ℕ),
  (∃ (p : ℕ → Fin 30 → Fin 30), 
    (∀ t, 
      (∀ i, 
        (initial_buns (p t i) = initial_buns i ∨ 
         initial_buns (p t i) = initial_buns i + 2 ∨ 
         initial_buns (p t i) = initial_buns i - 2))) → 
    ¬ ∀ n : Fin 30, initial_buns n = 2) := 
sorry

end buns_distribution_not_equal_for_all_cases_l126_126849


namespace gcd_lcm_sum_8_12_l126_126900

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l126_126900


namespace sampling_method_is_stratified_l126_126136

-- Given conditions
def unit_population : ℕ := 500 + 1000 + 800
def elderly_ratio : ℕ := 5
def middle_aged_ratio : ℕ := 10
def young_ratio : ℕ := 8
def total_selected : ℕ := 230

-- Prove that the sampling method used is stratified sampling
theorem sampling_method_is_stratified :
  (500 + 1000 + 800 = unit_population) ∧
  (total_selected = 230) ∧
  (500 * 230 / unit_population = elderly_ratio) ∧
  (1000 * 230 / unit_population = middle_aged_ratio) ∧
  (800 * 230 / unit_population = young_ratio) →
  sampling_method = stratified_sampling :=
by
  sorry

end sampling_method_is_stratified_l126_126136


namespace sea_horses_count_l126_126282

theorem sea_horses_count (S P : ℕ) 
  (h1 : S / P = 5 / 11) 
  (h2 : P = S + 85) 
  : S = 70 := sorry

end sea_horses_count_l126_126282


namespace abe_age_is_22_l126_126111

-- Define the conditions of the problem
def abe_age_condition (A : ℕ) : Prop := A + (A - 7) = 37

-- State the theorem
theorem abe_age_is_22 : ∃ A : ℕ, abe_age_condition A ∧ A = 22 :=
by
  sorry

end abe_age_is_22_l126_126111


namespace planes_meet_in_50_minutes_l126_126377

noncomputable def time_to_meet (d : ℕ) (vA vB : ℕ) : ℚ :=
  d / (vA + vB : ℚ)

theorem planes_meet_in_50_minutes
  (d : ℕ) (vA vB : ℕ)
  (h_d : d = 500) (h_vA : vA = 240) (h_vB : vB = 360) :
  (time_to_meet d vA vB * 60 : ℚ) = 50 := by
  sorry

end planes_meet_in_50_minutes_l126_126377


namespace find_certain_number_l126_126807

theorem find_certain_number (mystery_number certain_number : ℕ) (h1 : mystery_number = 47) 
(h2 : mystery_number + certain_number = 92) : certain_number = 45 :=
by
  sorry

end find_certain_number_l126_126807


namespace compare_values_l126_126477

variable (f : ℝ → ℝ)
variable (hf_even : ∀ x, f x = f (-x))
variable (hf_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x)

noncomputable def a : ℝ := f 1
noncomputable def b : ℝ := f (Real.log 3 / Real.log 0.5)
noncomputable def c : ℝ := f ((Real.log 3 / Real.log 2) - 1)

theorem compare_values (h_log1 : Real.log 3 / Real.log 0.5 < -1) 
                       (h_log2 : 0 < (Real.log 3 / Real.log 2) - 1 ∧ (Real.log 3 / Real.log 2) - 1 < 1) : 
  b < a ∧ a < c :=
by
  sorry

end compare_values_l126_126477


namespace base12_div_remainder_9_l126_126404

def base12_to_base10 (d0 d1 d2 d3: ℕ) : ℕ :=
  d0 * 12^3 + d1 * 12^2 + d2 * 12^1 + d3 * 12^0

theorem base12_div_remainder_9 : 
  let n := base12_to_base10 1 7 4 2 in
  n % 9 = 3 :=
by
  let n := base12_to_base10 1 7 4 2
  have : n = 2786 := rfl
  have : 2786 % 9 = 3 := by decide
  exact this

end base12_div_remainder_9_l126_126404


namespace abel_inequality_l126_126504

theorem abel_inequality (ξ ζ : ℝ → ℂ) (hξ1 : ∀ ω, 0 ≤ ξ ω ∧ ξ ω ≤ 1)
  (hξ2 : measurable ξ) (hζ : integrable ζ) :
  |∫ ω, (ξ ω) * (ζ ω) ∂measure_space.volume| ≤ 
  sup (set.Icc 0 1) (λ x, |∫ ω, (ζ ω) * indicator (λ ω', ξ ω' ≥ x) 1 ω ∂measure_space.volume|) := 
sorry

end abel_inequality_l126_126504


namespace percent_problem_l126_126494

theorem percent_problem (x y z w : ℝ) 
  (h1 : x = 1.20 * y) 
  (h2 : y = 0.40 * z) 
  (h3 : z = 0.70 * w) : 
  x = 0.336 * w :=
sorry

end percent_problem_l126_126494


namespace distinct_prime_factors_of_90_l126_126790

theorem distinct_prime_factors_of_90 : Nat.card (Finset.filter Nat.Prime (factors 90).toFinset) = 3 :=
by
  sorry

end distinct_prime_factors_of_90_l126_126790


namespace fraction_of_students_with_partner_l126_126148

theorem fraction_of_students_with_partner
  (a b : ℕ)
  (condition1 : ∀ seventh, seventh ≠ 0 → ∀ tenth, tenth ≠ 0 → a * b = 0)
  (condition2 : b / 4 = (3 * a) / 7) :
  (b / 4 + 3 * a / 7) / (b + a) = 6 / 19 :=
by
  sorry

end fraction_of_students_with_partner_l126_126148


namespace sum_le_two_of_cubics_sum_to_two_l126_126663

theorem sum_le_two_of_cubics_sum_to_two (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a^3 + b^3 = 2) : a + b ≤ 2 := 
sorry

end sum_le_two_of_cubics_sum_to_two_l126_126663


namespace contradiction_assumption_l126_126251

-- Define the proposition that a triangle has at most one obtuse angle
def at_most_one_obtuse_angle (T : Type) [triangle T] : Prop :=
  ∀ (A B C : T), ∠A > 90 → ∠B > 90 → false

-- Define the negation of the proposition
def negation_at_most_one_obtuse_angle (T : Type) [triangle T] : Prop :=
  ∃ (A B C : T), ∠A > 90 ∧ ∠B > 90

-- Prove that negation of the proposition implies "There are at least two obtuse angles in the triangle."
theorem contradiction_assumption (T : Type) [triangle T] :
  ¬ (at_most_one_obtuse_angle T) ↔ negation_at_most_one_obtuse_angle T :=
by sorry

end contradiction_assumption_l126_126251


namespace probability_allison_wins_l126_126937

open ProbabilityTheory

noncomputable def ADice : Pmf ℕ := ⟨λ n, if n = 4 then 1 else 0, by simp [sum_ite_eq _ 4]⟩
noncomputable def CDice : Pmf ℕ := uniformOfFin 6
noncomputable def EDice : Pmf ℕ := ⟨λ n, if n = 3 then 1/3 else if n = 4 then 1/2 else if n = 5 then 1/6 else 0, by simp [finset.sum_ite, finset.sum_const, mul_inv_cancel, ne_of_gt]⟩

theorem probability_allison_wins :
  ADice.prob (λ a, a > 3) * CDice.prob (λ c, c < 4) * EDice.prob (λ e, e < 4) = 1/6 := sorry

end probability_allison_wins_l126_126937


namespace hillary_minutes_read_on_saturday_l126_126777

theorem hillary_minutes_read_on_saturday :
  let total_minutes := 60
  let friday_minutes := 16
  let sunday_minutes := 16
  total_minutes - (friday_minutes + sunday_minutes) = 28 := by
sorry

end hillary_minutes_read_on_saturday_l126_126777


namespace emily_furniture_assembly_time_l126_126161

def num_chairs : Nat := 4
def num_tables : Nat := 2
def num_shelves : Nat := 3
def num_wardrobe : Nat := 1

def time_per_chair : Nat := 8
def time_per_table : Nat := 15
def time_per_shelf : Nat := 10
def time_per_wardrobe : Nat := 45

def total_time : Nat := 
  num_chairs * time_per_chair + 
  num_tables * time_per_table + 
  num_shelves * time_per_shelf + 
  num_wardrobe * time_per_wardrobe

theorem emily_furniture_assembly_time : total_time = 137 := by
  unfold total_time
  sorry

end emily_furniture_assembly_time_l126_126161


namespace circle_trajectory_l126_126713

theorem circle_trajectory (a b : ℝ) :
  ∃ x y : ℝ, (b - 3)^2 + a^2 = (b + 3)^2 → x^2 = 12 * y := 
sorry

end circle_trajectory_l126_126713


namespace polynomial_remainder_l126_126018

theorem polynomial_remainder (x : ℂ) (hx : x^5 = 1) :
  (x^25 + x^20 + x^15 + x^10 + x^5 + 1) % (x^5 - 1) = 6 :=
by
  -- Proof will go here
  sorry

end polynomial_remainder_l126_126018


namespace closed_chain_possible_l126_126702

-- Define the angle constraint
def angle_constraint (θ : ℝ) : Prop :=
  θ ≥ 150

-- Define meshing condition between two gears
def meshed_gears (θ : ℝ) : Prop :=
  angle_constraint θ

-- Define the general condition for a closed chain of gears
def closed_chain (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → meshed_gears 150

theorem closed_chain_possible : closed_chain 61 :=
by sorry

end closed_chain_possible_l126_126702


namespace compute_expression_l126_126000

theorem compute_expression : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end compute_expression_l126_126000


namespace quadratic_inequality_real_solutions_l126_126612

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l126_126612


namespace total_rainfall_in_january_l126_126742

theorem total_rainfall_in_january 
  (r1 r2 : ℝ)
  (h1 : r2 = 1.5 * r1)
  (h2 : r2 = 18) : 
  r1 + r2 = 30 := by
  sorry

end total_rainfall_in_january_l126_126742


namespace polynomial_divisibility_l126_126529

theorem polynomial_divisibility (C D : ℝ) (h : ∀ (ω : ℂ), ω^2 + ω + 1 = 0 → (ω^106 + C * ω + D = 0)) : C + D = -1 :=
by
  -- Add proof here
  sorry

end polynomial_divisibility_l126_126529


namespace problem_condition_l126_126203

theorem problem_condition (f : ℝ → ℝ) (a b : ℝ) :
  (∀ x : ℝ, f x = 5 * x - 3) → (∀ x : ℝ, |x + 0.4| < b → |f x + 1| < a) ↔ (0 < a ∧ 0 < b ∧ b ≤ a / 5) := by
  sorry

end problem_condition_l126_126203


namespace ratio_of_side_lengths_l126_126140

theorem ratio_of_side_lengths
  (pentagon_perimeter square_perimeter : ℕ)
  (pentagon_sides square_sides : ℕ)
  (pentagon_perimeter_eq : pentagon_perimeter = 100)
  (square_perimeter_eq : square_perimeter = 100)
  (pentagon_sides_eq : pentagon_sides = 5)
  (square_sides_eq : square_sides = 4) :
  (pentagon_perimeter / pentagon_sides) / (square_perimeter / square_sides) = 4 / 5 :=
by
  sorry

end ratio_of_side_lengths_l126_126140


namespace largest_k_inequality_l126_126958

theorem largest_k_inequality :
  ∃ k : ℝ, (∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → a + b + c = 3 → a^3 + b^3 + c^3 - 3 ≥ k * (3 - a * b - b * c - c * a)) ∧ k = 5 :=
sorry

end largest_k_inequality_l126_126958


namespace log_expression_evaluation_l126_126921

noncomputable def log2 : ℝ := Real.log 2
noncomputable def log5 : ℝ := Real.log 5

theorem log_expression_evaluation (condition : log2 + log5 = 1) :
  log2^2 + log2 * log5 + log5 - (Real.sqrt 2 - 1)^0 = 0 :=
by
  sorry

end log_expression_evaluation_l126_126921


namespace exists_not_holds_l126_126201

variable (S : Type) [Nonempty S] [Inhabited S]
variable (op : S → S → S)
variable (h : ∀ a b : S, op a (op b a) = b)

theorem exists_not_holds : ∃ a b : S, (op (op a b) a) ≠ a := sorry

end exists_not_holds_l126_126201


namespace Jason_reroll_probability_optimal_l126_126653

/-- Represents the action of rerolling dice to achieve a sum of 9 when
    the player optimizes their strategy. The probability 
    that the player chooses to reroll exactly two dice.
 -/
noncomputable def probability_reroll_two_dice : ℚ :=
  13 / 72

/-- Prove that the probability Jason chooses to reroll exactly two
    dice to achieve a sum of 9, given the optimal strategy, is 13/72.
 -/
theorem Jason_reroll_probability_optimal :
  probability_reroll_two_dice = 13 / 72 :=
sorry

end Jason_reroll_probability_optimal_l126_126653


namespace peter_fraction_equiv_l126_126386

def fraction_pizza_peter_ate (total_slices : ℕ) (slices_ate_alone : ℕ) (shared_slices_brother : ℚ) (shared_slices_sister : ℚ) : ℚ :=
  (slices_ate_alone / total_slices) + (shared_slices_brother / total_slices) + (shared_slices_sister / total_slices)

theorem peter_fraction_equiv :
  fraction_pizza_peter_ate 16 3 (1/2) (1/2) = 1/4 :=
by
  sorry

end peter_fraction_equiv_l126_126386


namespace solve_trig_equation_l126_126091
open Real

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (1 / 2) * abs (cos (2 * x) + (1 / 2)) = (sin (3 * x))^2 - (sin x) * (sin (3 * x))

-- Define the correct solution set 
def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (π / 6) + (k * (π / 2)) ∨ x = -(π / 6) + (k * (π / 2))

-- The theorem we need to prove
theorem solve_trig_equation : ∀ x : ℝ, original_equation x ↔ solution_set x :=
by sorry

end solve_trig_equation_l126_126091


namespace range_of_xy_l126_126637

-- Given conditions
variables {x y : ℝ} (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1)

-- To Prove
theorem range_of_xy (hx : x > 0) (hy : y > 0) (hxy : 2 / x + 8 / y = 1) : 64 ≤ x * y :=
sorry

end range_of_xy_l126_126637


namespace jonah_fishes_per_day_l126_126497

theorem jonah_fishes_per_day (J G J_total : ℕ) (days : ℕ) (total : ℕ)
  (hJ : J = 6) (hG : G = 8) (hdays : days = 5) (htotal : total = 90) 
  (fish_total : days * J + days * G + days * J_total = total) : 
  J_total = 4 :=
by
  sorry

end jonah_fishes_per_day_l126_126497


namespace correct_number_of_six_letter_words_l126_126955

def number_of_six_letter_words (alphabet_size : ℕ) : ℕ :=
  alphabet_size ^ 4

theorem correct_number_of_six_letter_words :
  number_of_six_letter_words 26 = 456976 :=
by
  -- We write 'sorry' to omit the detailed proof.
  sorry

end correct_number_of_six_letter_words_l126_126955


namespace system_solution_l126_126015

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) (h₄ : 0 < x₄) (h₅ : 0 < x₅)
  (h₆ : x₁ + x₂ = x₃^2) (h₇ : x₃ + x₄ = x₅^2) (h₈ : x₄ + x₅ = x₁^2) (h₉ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
by 
  sorry

end system_solution_l126_126015


namespace regular_pay_per_hour_l126_126421

theorem regular_pay_per_hour (R : ℝ) (h : 40 * R + 11 * (2 * R) = 186) : R = 3 :=
by
  sorry

end regular_pay_per_hour_l126_126421


namespace number_of_distinct_prime_factors_of_90_l126_126789

-- Definition of what it means to be a prime factor
def is_prime_factor (p n : ℕ) : Prop := nat.prime p ∧ p ∣ n

-- Definition of the number of distinct prime factors of a number n
def num_distinct_prime_factors (n : ℕ) : ℕ :=
  (finset.filter (λ p, is_prime_factor p n) (finset.range (n+1))).card

-- Statement of the proof problem
theorem number_of_distinct_prime_factors_of_90 : num_distinct_prime_factors 90 = 3 :=
by sorry

end number_of_distinct_prime_factors_of_90_l126_126789


namespace gcd_lcm_sum_8_12_l126_126896

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l126_126896


namespace triangle_angle_B_eq_60_l126_126687

theorem triangle_angle_B_eq_60 {A B C : ℝ} (h1 : B = 2 * A) (h2 : C = 3 * A) (h3 : A + B + C = 180) : B = 60 :=
by sorry

end triangle_angle_B_eq_60_l126_126687


namespace spongebob_price_l126_126095

variable (x : ℝ)

theorem spongebob_price (h : 30 * x + 12 * 1.5 = 78) : x = 2 :=
by
  -- Given condition: 30 * x + 12 * 1.5 = 78
  sorry

end spongebob_price_l126_126095


namespace sandwich_cost_l126_126873

theorem sandwich_cost (c : ℕ) 
  (sandwiches : ℕ := 3)
  (drinks : ℕ := 2)
  (cost_per_drink : ℕ := 4)
  (total_spent : ℕ := 26)
  (drink_cost : ℕ := drinks * cost_per_drink)
  (sandwich_spent : ℕ := total_spent - drink_cost) :
  (∀ s, sandwich_spent = s * sandwiches → s = 6) :=
by
  intros s hs
  have hsandwich_count : sandwiches = 3 := by rfl
  have hdrinks : drinks = 2 := by rfl
  have hcost_per_drink : cost_per_drink = 4 := by rfl
  have htotal_spent : total_spent = 26 := by rfl
  have hdrink_cost : drink_cost = 8 := by
    calc 
      drinks * cost_per_drink 
      = 2 * 4 : by rw [hdrinks, hcost_per_drink]
      = 8 : by norm_num
  have hsandwich_spent : sandwich_spent = 18 := by
    calc
      total_spent - drink_cost 
      = 26 - 8 : by rw [htotal_spent, hdrink_cost]
      = 18 : by norm_num
  rw hsandwich_count at hs
  rw hsandwich_spent at hs
  linarith

end sandwich_cost_l126_126873


namespace mr_smith_spends_l126_126669

def buffet_price 
  (adult_price : ℕ) 
  (child_price : ℕ) 
  (senior_discount : ℕ) 
  (num_full_price_adults : ℕ) 
  (num_children : ℕ) 
  (num_seniors : ℕ) : ℕ :=
  num_full_price_adults * adult_price + num_children * child_price + num_seniors * (adult_price - (adult_price * senior_discount / 100))

theorem mr_smith_spends (adult_price : ℕ) (child_price : ℕ) (senior_discount : ℕ) (num_full_price_adults : ℕ) (num_children : ℕ) (num_seniors : ℕ) : 
  adult_price = 30 → 
  child_price = 15 → 
  senior_discount = 10 → 
  num_full_price_adults = 3 → 
  num_children = 3 → 
  num_seniors = 1 → 
  buffet_price adult_price child_price senior_discount num_full_price_adults num_children num_seniors = 162 :=
by 
  intros h_adult_price h_child_price h_senior_discount h_num_full_price_adults h_num_children h_num_seniors
  rw [h_adult_price, h_child_price, h_senior_discount, h_num_full_price_adults, h_num_children, h_num_seniors]
  sorry

end mr_smith_spends_l126_126669


namespace x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l126_126488

variable {x y : ℝ}

theorem x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13
  (h1 : x + y = 10) 
  (h2 : x * y = 12) : 
  x^3 - y^3 = 176 * Real.sqrt 13 := 
by
  sorry

end x_plus_y_eq_10_and_xy_eq_12_implies_x3_minus_y3_eq_176_sqrt_13_l126_126488


namespace clay_weight_in_second_box_l126_126707

/-- Define the properties of the first and second boxes -/
structure Box where
  height : ℕ
  width : ℕ
  length : ℕ
  weight : ℕ

noncomputable def box1 : Box :=
  { height := 2, width := 3, length := 5, weight := 40 }

noncomputable def box2 : Box :=
  { height := 2 * 2, width := 3 * 3, length := 5, weight := 240 }

theorem clay_weight_in_second_box : 
  box2.weight = (box2.height * box2.width * box2.length) / 
                (box1.height * box1.width * box1.length) * box1.weight :=
by
  sorry

end clay_weight_in_second_box_l126_126707


namespace painting_time_l126_126803

variable (a d e : ℕ)

theorem painting_time (h : a * e * d = a * d * e) : (d * x = a^2 * e) := 
by
   sorry

end painting_time_l126_126803


namespace ab_cd_value_l126_126978

theorem ab_cd_value (a b c d : ℝ) 
  (h1 : a + b + c = 5)
  (h2 : a + b + d = -3)
  (h3 : a + c + d = 10)
  (h4 : b + c + d = 0) : 
  a * b + c * d = -31 :=
by
  sorry

end ab_cd_value_l126_126978


namespace stratified_sampling_first_level_l126_126721

-- Definitions from the conditions
def num_senior_teachers : ℕ := 90
def num_first_level_teachers : ℕ := 120
def num_second_level_teachers : ℕ := 170
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_second_level_teachers
def sample_size : ℕ := 38

-- Definition of the stratified sampling result
def num_first_level_selected : ℕ := (num_first_level_teachers * sample_size) / total_teachers

-- The statement to be proven
theorem stratified_sampling_first_level : num_first_level_selected = 12 :=
by
  sorry

end stratified_sampling_first_level_l126_126721


namespace number_of_possible_ceil_values_l126_126046

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l126_126046


namespace find_omega_l126_126340

noncomputable def f (ω x : ℝ) := sqrt 3 * Real.sin (ω * x) + Real.cos (ω * x)

theorem find_omega (ω x₁ x₂ : ℝ) (h_ω : ω > 0) (hx₁ : f ω x₁ = -2) (hx₂ : f ω x₂ = 0) (h_dist : abs (x₁ - x₂) = π) : ω = 1 / 2 :=
sorry

end find_omega_l126_126340


namespace abs_neg_2023_l126_126521

theorem abs_neg_2023 : abs (-2023) = 2023 := 
by
  sorry

end abs_neg_2023_l126_126521


namespace sum_gcf_lcm_l126_126904

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l126_126904


namespace star_sub_correctness_l126_126639

def star (x y : ℤ) : ℤ := x * y - 3 * x

theorem star_sub_correctness : (star 6 2) - (star 2 6) = -12 := by
  sorry

end star_sub_correctness_l126_126639


namespace Justin_and_Tim_games_l126_126155

theorem Justin_and_Tim_games (total_players : ℕ) (total_games : ℕ) 
    (total_players_split : total_players = 12) 
    (each_game : total_games = Nat.choose total_players 6) 
    (unique_matchups : ∀ (p1 p2 : Fin 12), ∃! group : Finset (Fin 12), group.card = 6 ∧ (p1 ∈ group ∧ p2 ∈ group)) :
  ∃ (games_with_Justin_and_Tim : ℕ), games_with_Justin_and_Tim = 210 :=
by
  -- Using the conditions and known equations, we assert that there exists a 
  -- certain number of games where Justin and Tim will play together.
  have eq : Nat.choose 10 4 = 210 := by simp [Nat.choose, Nat.factorial]; simp
  exact ⟨210, eq⟩

end Justin_and_Tim_games_l126_126155


namespace museum_earnings_from_nyc_college_students_l126_126065

def visitors := 200
def nyc_residents_fraction := 1 / 2
def college_students_fraction := 0.30
def ticket_price := 4

theorem museum_earnings_from_nyc_college_students : 
  ((visitors * nyc_residents_fraction * college_students_fraction) * ticket_price) = 120 := 
by 
  sorry

end museum_earnings_from_nyc_college_students_l126_126065


namespace integer_solution_pair_l126_126013

theorem integer_solution_pair (x y : ℤ) (h : x^2 + x * y = y^2) : (x = 0 ∧ y = 0) :=
by
  sorry

end integer_solution_pair_l126_126013


namespace shopkeeper_net_loss_percent_l126_126931

theorem shopkeeper_net_loss_percent (cp : ℝ)
  (sp1 sp2 sp3 sp4 : ℝ)
  (h_cp : cp = 1000)
  (h_sp1 : sp1 = cp * 1.1)
  (h_sp2 : sp2 = cp * 0.9)
  (h_sp3 : sp3 = cp * 1.2)
  (h_sp4 : sp4 = cp * 0.75) :
  ((cp + cp + cp + cp) - (sp1 + sp2 + sp3 + sp4)) / (cp + cp + cp + cp) * 100 = 1.25 :=
by sorry

end shopkeeper_net_loss_percent_l126_126931


namespace inequality_a_b_cubed_l126_126475

theorem inequality_a_b_cubed (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^3 < b^3 :=
sorry

end inequality_a_b_cubed_l126_126475


namespace sin_180_eq_zero_l126_126306

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l126_126306


namespace required_line_equation_l126_126748

-- Define the point P
structure Point where
  x : ℝ
  y : ℝ

-- Line structure with general form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- A point P on a line
def on_line (P : Point) (l : Line) : Prop :=
  l.a * P.x + l.b * P.y + l.c = 0

-- Perpendicular condition between two lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- The known line
def known_line : Line := {a := 1, b := -2, c := 3}

-- The given point
def P : Point := {x := -1, y := 3}

noncomputable def required_line : Line := {a := 2, b := 1, c := -1}

-- The theorem to be proved
theorem required_line_equation (l : Line) (P : Point) :
  (on_line P l) ∧ (perpendicular l known_line) ↔ l = required_line :=
  by
    sorry

end required_line_equation_l126_126748


namespace parabola_focus_l126_126842

theorem parabola_focus (x : ℝ) : ∃ f : ℝ × ℝ, f = (0, 1 / 4) ∧ ∀ y : ℝ, y = x^2 → f = (0, 1 / 4) :=
by
  sorry

end parabola_focus_l126_126842


namespace find_a_l126_126660

theorem find_a (a b : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 3) (h₂ : 3 / a + 6 / b = 2 / 3) : 
  a = 9 * b / (2 * b - 18) :=
by
  sorry

end find_a_l126_126660


namespace fraction_savings_on_makeup_l126_126997

theorem fraction_savings_on_makeup (savings : ℝ) (sweater_cost : ℝ) (makeup_cost : ℝ) (h_savings : savings = 80) (h_sweater : sweater_cost = 20) (h_makeup : makeup_cost = savings - sweater_cost) : makeup_cost / savings = 3 / 4 := by
  sorry

end fraction_savings_on_makeup_l126_126997


namespace total_animals_l126_126941

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l126_126941


namespace ratio_of_cream_l126_126010

def initial_coffee := 18
def cup_capacity := 22
def Emily_drank := 3
def Emily_added_cream := 4
def Ethan_added_cream := 4
def Ethan_drank := 3

noncomputable def cream_in_Emily := Emily_added_cream

noncomputable def cream_remaining_in_Ethan :=
  Ethan_added_cream - (Ethan_added_cream * Ethan_drank / (initial_coffee + Ethan_added_cream))

noncomputable def resulting_ratio := cream_in_Emily / cream_remaining_in_Ethan

theorem ratio_of_cream :
  resulting_ratio = 200 / 173 :=
by
  sorry

end ratio_of_cream_l126_126010


namespace cows_black_more_than_half_l126_126089

theorem cows_black_more_than_half (t b : ℕ) (h1 : t = 18) (h2 : t - 4 = b) : b - t / 2 = 5 :=
by
  sorry

end cows_black_more_than_half_l126_126089


namespace james_lifting_heavy_after_39_days_l126_126070

noncomputable def JamesInjuryHealingTime : Nat := 3
noncomputable def HealingTimeFactor : Nat := 5
noncomputable def WaitingTimeAfterHealing : Nat := 3
noncomputable def AdditionalWaitingTimeWeeks : Nat := 3

theorem james_lifting_heavy_after_39_days :
  let healing_time := JamesInjuryHealingTime * HealingTimeFactor
  let total_time_before_workout := healing_time + WaitingTimeAfterHealing
  let additional_waiting_time_days := AdditionalWaitingTimeWeeks * 7
  let total_time_before_lifting_heavy := total_time_before_workout + additional_waiting_time_days
  total_time_before_lifting_heavy = 39 := by
  sorry

end james_lifting_heavy_after_39_days_l126_126070


namespace mass_percentage_of_N_in_NH4Br_l126_126615

theorem mass_percentage_of_N_in_NH4Br :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Br := 79.90
  let molar_mass_NH4Br := (1 * molar_mass_N) + (4 * molar_mass_H) + (1 * molar_mass_Br)
  let mass_percentage_N := (molar_mass_N / molar_mass_NH4Br) * 100
  mass_percentage_N = 14.30 :=
by
  sorry

end mass_percentage_of_N_in_NH4Br_l126_126615


namespace solve_quadratic_eq_l126_126219

theorem solve_quadratic_eq (x : ℝ) :
  (x^2 + (x - 1) * (x + 3) = 3 * x + 5) ↔ (x = -2 ∨ x = 2) :=
by
  sorry

end solve_quadratic_eq_l126_126219


namespace width_of_canal_at_bottom_l126_126102

theorem width_of_canal_at_bottom (h : Real) (b : Real) : 
  (A = 1/2 * (top_width + b) * d) ∧ 
  (A = 840) ∧ 
  (top_width = 12) ∧ 
  (d = 84) 
  → b = 8 := 
by
  intros
  sorry

end width_of_canal_at_bottom_l126_126102


namespace sum_gcf_lcm_l126_126906

theorem sum_gcf_lcm (a b : ℕ) (h_a : a = 8) (h_b : b = 12) :
  Nat.gcd a b + Nat.lcm a b = 28 :=
by
  rw [h_a, h_b]
  -- here you'd typically use known theorems about gcd and lcm for specific numbers
  sorry

end sum_gcf_lcm_l126_126906


namespace number_of_diagonals_in_convex_polygon_l126_126352

/-- 
A theorem stating that the number of diagonals 
in a convex polygon with 30 sides is equal to 405.
-/
theorem number_of_diagonals_in_convex_polygon (n : ℕ) (h : n = 30) (convex : True) : (n * (n - 3)) / 2 = 405 := 
by 
  rw h
  norm_num
  done

end number_of_diagonals_in_convex_polygon_l126_126352


namespace line_tangent_72_l126_126616

theorem line_tangent_72 (k : ℝ) : 4 * x + 6 * y + k = 0 → y^2 = 32 * x → (48^2 - 4 * (8 * k) = 0 ↔ k = 72) :=
by
  sorry

end line_tangent_72_l126_126616


namespace ratio_divisor_to_remainder_l126_126056

theorem ratio_divisor_to_remainder (R D Q : ℕ) (hR : R = 46) (hD : D = 10 * Q) (hdvd : 5290 = D * Q + R) :
  D / R = 5 :=
by
  sorry

end ratio_divisor_to_remainder_l126_126056


namespace problem_l126_126463

theorem problem (m n : ℕ) 
  (m_pos : 0 < m) 
  (n_pos : 0 < n) 
  (h1 : m + 8 < n) 
  (h2 : (m + (m + 3) + (m + 8) + n + (n + 3) + (2 * n - 1)) / 6 = n + 1) 
  (h3 : (m + 8 + n) / 2 = n + 1) : m + n = 16 :=
  sorry

end problem_l126_126463


namespace freddy_total_call_cost_l126_126755

def lm : ℕ := 45
def im : ℕ := 31
def lc : ℝ := 0.05
def ic : ℝ := 0.25

theorem freddy_total_call_cost : lm * lc + im * ic = 10.00 := by
  sorry

end freddy_total_call_cost_l126_126755


namespace ants_meet_at_QS_l126_126123

theorem ants_meet_at_QS (P Q R S : Type)
  (dist_PQ : Nat)
  (dist_QR : Nat)
  (dist_PR : Nat)
  (ants_meet : 2 * (dist_PQ + (5 : Nat)) = dist_PQ + dist_QR + dist_PR)
  (perimeter : dist_PQ + dist_QR + dist_PR = 24)
  (distance_each_ant_crawls : (dist_PQ + 5) = 12) :
  5 = 5 :=
by
  sorry

end ants_meet_at_QS_l126_126123


namespace sum_of_gcd_and_lcm_is_28_l126_126889

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l126_126889


namespace tomatoes_initially_l126_126716

-- Conditions
def tomatoes_picked_yesterday : ℕ := 56
def tomatoes_picked_today : ℕ := 41
def tomatoes_left_after_yesterday : ℕ := 104

-- The statement to prove
theorem tomatoes_initially : tomatoes_left_after_yesterday + tomatoes_picked_yesterday + tomatoes_picked_today = 201 :=
  by
  -- Proof steps would go here
  sorry

end tomatoes_initially_l126_126716


namespace problem_inequality_solution_set_problem_minimum_value_l126_126174

noncomputable def f (x : ℝ) := x^2 / (x - 1)

theorem problem_inequality_solution_set : 
  ∀ x : ℝ, 1 < x ∧ x < (1 + Real.sqrt 5) / 2 → f x > 2 * x + 1 :=
sorry

theorem problem_minimum_value : ∀ x : ℝ, x > 1 → (f x ≥ 4) ∧ (f 2 = 4) :=
sorry

end problem_inequality_solution_set_problem_minimum_value_l126_126174


namespace find_ac_pair_l126_126392

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l126_126392


namespace find_y_l126_126489

theorem find_y (x y : ℤ) 
  (h1 : x^2 + 4 = y - 2) 
  (h2 : x = 6) : 
  y = 42 := 
by 
  sorry

end find_y_l126_126489


namespace sin_180_eq_0_l126_126299

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l126_126299


namespace probability_two_even_dice_l126_126584

noncomputable def probability_even_two_out_of_six : ℚ :=
  15 * (1 / 64)

theorem probability_two_even_dice :
  (∀ (d1 d2 d3 d4 d5 d6 : ℕ), (d1 ∈ finset.range 1 11) ∧ (d2 ∈ finset.range 1 11) ∧ (d3 ∈ finset.range 1 11) ∧ (d4 ∈ finset.range 1 11) ∧ (d5 ∈ finset.range 1 11) ∧ (d6 ∈ finset.range 1 11)) →
  (probability_even_two_out_of_six = 15 / 64) :=
by
  sorry

end probability_two_even_dice_l126_126584


namespace total_animals_l126_126940

-- Definitions of the initial conditions
def initial_beavers := 20
def initial_chipmunks := 40
def doubled_beavers := 2 * initial_beavers
def decreased_chipmunks := initial_chipmunks - 10

theorem total_animals (initial_beavers initial_chipmunks doubled_beavers decreased_chipmunks : ℕ)
    (h1 : doubled_beavers = 2 * initial_beavers)
    (h2 : decreased_chipmunks = initial_chipmunks - 10) :
    (initial_beavers + initial_chipmunks) + (doubled_beavers + decreased_chipmunks) = 130 :=
by 
  sorry

end total_animals_l126_126940


namespace inequality_transitive_l126_126407

theorem inequality_transitive (a b c : ℝ) : a * c^2 > b * c^2 → a > b :=
sorry

end inequality_transitive_l126_126407


namespace old_fridge_cost_l126_126503

-- Define the daily cost of Kurt's old refrigerator
variable (x : ℝ)

-- Define the conditions given in the problem
def new_fridge_cost_per_day : ℝ := 0.45
def savings_per_month : ℝ := 12
def days_in_month : ℝ := 30

-- State the theorem to prove
theorem old_fridge_cost :
  30 * x - 30 * new_fridge_cost_per_day = savings_per_month → x = 0.85 := 
by
  intro h
  sorry

end old_fridge_cost_l126_126503


namespace two_integers_difference_l126_126217

theorem two_integers_difference
  (x y : ℕ)
  (h_sum : x + y = 5)
  (h_cube_diff : x^3 - y^3 = 63)
  (h_gt : x > y) :
  x - y = 3 := 
sorry

end two_integers_difference_l126_126217


namespace work_completion_problem_l126_126711

theorem work_completion_problem :
  (∃ x : ℕ, 9 * (1 / 45 + 1 / x) + 23 * (1 / x) = 1) → x = 40 :=
sorry

end work_completion_problem_l126_126711


namespace short_pencil_cost_l126_126984

theorem short_pencil_cost (x : ℝ)
  (h1 : 200 * 0.8 + 40 * 0.5 + 35 * x = 194) : x = 0.4 :=
by {
  sorry
}

end short_pencil_cost_l126_126984


namespace line_through_point_and_area_l126_126163

theorem line_through_point_and_area (a b : ℝ) (x y : ℝ) 
  (hx : x = -2) (hy : y = 2) 
  (h_area : 1/2 * |a * b| = 1): 
  (2 * x + y + 2 = 0 ∨ x + 2 * y - 2 = 0) :=
  sorry

end line_through_point_and_area_l126_126163


namespace number_of_B_students_l126_126359

theorem number_of_B_students (x : ℝ) (h1 : 0.8 * x + x + 1.2 * x = 40) : x = 13 :=
  sorry

end number_of_B_students_l126_126359


namespace option_B_is_not_polynomial_l126_126406

-- Define what constitutes a polynomial
def is_polynomial (expr : String) : Prop :=
  match expr with
  | "-26m" => True
  | "3m+5n" => True
  | "0" => True
  | _ => False

-- Given expressions
def expr_A := "-26m"
def expr_B := "m-n=1"
def expr_C := "3m+5n"
def expr_D := "0"

-- The Lean statement confirming option B is not a polynomial
theorem option_B_is_not_polynomial : ¬is_polynomial expr_B :=
by
  -- Since this statement requires a proof, we use 'sorry' as a placeholder.
  sorry

end option_B_is_not_polynomial_l126_126406


namespace karen_cookies_grandparents_l126_126655

theorem karen_cookies_grandparents :
  ∀ (total_cookies cookies_kept class_size cookies_per_person : ℕ)
  (cookies_given_class cookies_left cookies_to_grandparents : ℕ),
  total_cookies = 50 →
  cookies_kept = 10 →
  class_size = 16 →
  cookies_per_person = 2 →
  cookies_given_class = class_size * cookies_per_person →
  cookies_left = total_cookies - cookies_kept - cookies_given_class →
  cookies_to_grandparents = cookies_left →
  cookies_to_grandparents = 8 :=
by
  intros
  sorry

end karen_cookies_grandparents_l126_126655


namespace angle_value_l126_126334

theorem angle_value (α : ℝ) (h1 : 0 ≤ α) (h2 : α < 360) 
(h3 : (Real.sin 215 * π / 180, Real.cos 215 * π / 180) = (Real.sin α, Real.cos α)) :
α = 235 :=
sorry

end angle_value_l126_126334


namespace ratio_of_a_to_c_l126_126531

theorem ratio_of_a_to_c (a b c d : ℚ) 
  (h1 : a / b = 5 / 4)
  (h2 : c / d = 4 / 3)
  (h3 : d / b = 1 / 5) :
  a / c = 75 / 16 :=
by
  sorry

end ratio_of_a_to_c_l126_126531


namespace geometric_sequence_a2_a4_sum_l126_126759

theorem geometric_sequence_a2_a4_sum :
  ∃ (a : ℕ → ℝ) (q : ℝ), (∀ n, a n = a 1 * q ^ (n - 1)) ∧
    (a 2 * a 4 = 9) ∧
    (9 * (a 1 * (1 - q^4) / (1 - q)) = 10 * (a 1 * (1 - q^2) / (1 - q))) ∧
    (a 2 + a 4 = 10) :=
by
  sorry

end geometric_sequence_a2_a4_sum_l126_126759


namespace average_productivity_l126_126585

theorem average_productivity (T : ℕ) (total_words : ℕ) (increased_time_fraction : ℚ) (increased_productivity_fraction : ℚ) :
  T = 100 →
  total_words = 60000 →
  increased_time_fraction = 0.2 →
  increased_productivity_fraction = 1.5 →
  (total_words / T : ℚ) = 600 :=
by
  sorry

end average_productivity_l126_126585


namespace greatest_difference_l126_126854

def difference_marbles : Nat :=
  let A_diff := 4 - 2
  let B_diff := 6 - 1
  let C_diff := 9 - 3
  max (max A_diff B_diff) C_diff

theorem greatest_difference :
  difference_marbles = 6 :=
by
  sorry

end greatest_difference_l126_126854


namespace smallest_gcd_of_lcm_eq_square_diff_l126_126547

theorem smallest_gcd_of_lcm_eq_square_diff (x y : ℕ) (h : Nat.lcm x y = (x - y) ^ 2) : Nat.gcd x y = 2 :=
sorry

end smallest_gcd_of_lcm_eq_square_diff_l126_126547


namespace num_distinct_prime_factors_90_l126_126785

-- Define what it means for a number to be a prime
def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ d : ℕ, d ∣ p → d = 1 ∨ d = p

-- Define what it means for a set of prime factors to be the distinct prime factors of a number
def prime_factors (n : ℕ) : set ℕ :=
  {p | is_prime p ∧ p ∣ n}

-- State the problem: proving the number of distinct prime factors of 90 is 3
theorem num_distinct_prime_factors_90 : nat.card (prime_factors 90) = 3 :=
sorry

end num_distinct_prime_factors_90_l126_126785


namespace camera_value_l126_126994

variables (V : ℝ)

def rental_fee_per_week (V : ℝ) := 0.1 * V
def total_rental_fee(V : ℝ) := 4 * rental_fee_per_week V
def johns_share_of_fee(V : ℝ) := 0.6 * (0.4 * total_rental_fee V)

theorem camera_value (h : johns_share_of_fee V = 1200): 
  V = 5000 :=
by
  sorry

end camera_value_l126_126994


namespace percentage_died_by_bombardment_l126_126926

theorem percentage_died_by_bombardment (P_initial : ℝ) (P_remaining : ℝ) (died_percentage : ℝ) (fear_percentage : ℝ) :
  P_initial = 3161 → P_remaining = 2553 → fear_percentage = 0.15 → 
  P_initial - (died_percentage/100) * P_initial - fear_percentage * (P_initial - (died_percentage/100) * P_initial) = P_remaining → 
  abs (died_percentage - 4.98) < 0.01 :=
by
  intros h1 h2 h3 h4
  sorry

end percentage_died_by_bombardment_l126_126926


namespace distinct_numbers_in_union_set_l126_126505

def first_seq_term (k : ℕ) : ℤ := 5 * ↑k - 3
def second_seq_term (m : ℕ) : ℤ := 9 * ↑m - 3

def first_seq_set : Finset ℤ := ((Finset.range 1003).image first_seq_term)
def second_seq_set : Finset ℤ := ((Finset.range 1003).image second_seq_term)

def union_set : Finset ℤ := first_seq_set ∪ second_seq_set

theorem distinct_numbers_in_union_set : union_set.card = 1895 := by
  sorry

end distinct_numbers_in_union_set_l126_126505


namespace aubree_animals_total_l126_126939

theorem aubree_animals_total (b_go c_go b_return c_return : ℕ) 
    (h1 : b_go = 20) (h2 : c_go = 40) 
    (h3 : b_return = b_go * 2) 
    (h4 : c_return = c_go - 10) : 
    b_go + c_go + b_return + c_return = 130 := by 
  sorry

end aubree_animals_total_l126_126939


namespace total_problems_l126_126675

-- Definitions based on conditions
def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def problems_per_page : ℕ := 4

-- Statement of the problem
theorem total_problems : math_pages + reading_pages * problems_per_page = 40 :=
by
  unfold math_pages reading_pages problems_per_page
  sorry

end total_problems_l126_126675


namespace magician_earnings_l126_126577

noncomputable def total_earnings (price_per_deck : ℕ) (initial_decks : ℕ) (end_decks : ℕ) (promotion_price : ℕ) (exchange_rate_start : ℚ) (exchange_rate_mid : ℚ) (foreign_sales_1 : ℕ) (domestic_sales : ℕ) (foreign_sales_2 : ℕ) : ℕ :=
  let foreign_earnings_1 := (foreign_sales_1 / 2) * promotion_price
  let foreign_earnings_2 := foreign_sales_2 * price_per_deck
  (domestic_sales / 2) * promotion_price + foreign_earnings_1 + foreign_earnings_2
  

-- Given conditions:
-- price_per_deck = 2
-- initial_decks = 5
-- end_decks = 3
-- promotion_price = 3
-- exchange_rate_start = 1
-- exchange_rate_mid = 1.5
-- foreign_sales_1 = 4
-- domestic_sales = 2
-- foreign_sales_2 = 1

theorem magician_earnings :
  total_earnings 2 5 3 3 1 1.5 4 2 1 = 11 :=
by
   sorry

end magician_earnings_l126_126577


namespace money_received_from_mom_l126_126210

-- Define the given conditions
def initial_amount : ℕ := 48
def amount_spent : ℕ := 11
def amount_after_getting_money : ℕ := 58
def amount_left_after_spending : ℕ := initial_amount - amount_spent

-- Define the proof statement
theorem money_received_from_mom : (amount_after_getting_money - amount_left_after_spending) = 21 :=
by
  -- placeholder for the proof
  sorry

end money_received_from_mom_l126_126210


namespace maximal_value_ratio_l126_126999

theorem maximal_value_ratio (a b c h : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_altitude : h = (a * b) / c) :
  ∃ θ : ℝ, a = c * Real.cos θ ∧ b = c * Real.sin θ ∧ (1 < Real.cos θ + Real.sin θ ∧ Real.cos θ + Real.sin θ ≤ Real.sqrt 2) ∧
  ( Real.cos θ * Real.sin θ = (1 + 2 * Real.cos θ * Real.sin θ - 1) / 2 ) → 
  (c + h) / (a + b) ≤ 3 * Real.sqrt 2 / 4 :=
sorry

end maximal_value_ratio_l126_126999


namespace range_of_m_l126_126186

theorem range_of_m {x1 x2 y1 y2 m : ℝ} 
  (h1 : x1 > x2) 
  (h2 : y1 > y2) 
  (ha : y1 = (m - 3) * x1 - 4) 
  (hb : y2 = (m - 3) * x2 - 4) : 
  m > 3 :=
sorry

end range_of_m_l126_126186


namespace constant_max_value_l126_126491

theorem constant_max_value (n : ℤ) (c : ℝ) (h1 : c * (n^2) ≤ 8100) (h2 : n = 8) :
  c ≤ 126.5625 :=
sorry

end constant_max_value_l126_126491


namespace total_peaches_in_each_basket_l126_126242

-- Define the given conditions
def red_peaches : ℕ := 7
def green_peaches : ℕ := 3

-- State the theorem
theorem total_peaches_in_each_basket : red_peaches + green_peaches = 10 :=
by
  -- Proof goes here, which we skip for now
  sorry

end total_peaches_in_each_basket_l126_126242


namespace arithmetic_sequence_length_correct_l126_126799

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end arithmetic_sequence_length_correct_l126_126799


namespace sum_of_cubes_of_consecutive_integers_div_by_9_l126_126833

theorem sum_of_cubes_of_consecutive_integers_div_by_9 (x : ℤ) : 
  let a := (x - 1) ^ 3
  let b := x ^ 3
  let c := (x + 1) ^ 3
  (a + b + c) % 9 = 0 :=
by
  sorry

end sum_of_cubes_of_consecutive_integers_div_by_9_l126_126833


namespace sequences_of_length_15_l126_126451

def odd_runs_of_A_even_runs_of_B (n : ℕ) : ℕ :=
  (if n = 1 then 1 else 0) + (if n = 2 then 1 else 0)

theorem sequences_of_length_15 : 
  odd_runs_of_A_even_runs_of_B 15 = 47260 :=
  sorry

end sequences_of_length_15_l126_126451


namespace janet_initial_action_figures_l126_126991

theorem janet_initial_action_figures (x : ℕ) :
  (x - 2 + 2 * (x - 2) = 24) -> x = 10 := 
by
  sorry

end janet_initial_action_figures_l126_126991


namespace complex_number_imaginary_axis_l126_126101

theorem complex_number_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) → (a = 0 ∨ a = 2) :=
by
  sorry

end complex_number_imaginary_axis_l126_126101


namespace exists_member_T_divisible_by_3_l126_126512

-- Define the set T of all numbers which are the sum of the squares of four consecutive integers
def T := { x : ℤ | ∃ n : ℤ, x = (n-1)^2 + n^2 + (n+1)^2 + (n+2)^2 }

-- Theorem to prove that there exists a member in T which is divisible by 3
theorem exists_member_T_divisible_by_3 : ∃ x ∈ T, x % 3 = 0 :=
by
  sorry

end exists_member_T_divisible_by_3_l126_126512


namespace certain_number_divides_expression_l126_126323

theorem certain_number_divides_expression : 
  ∃ m : ℕ, (∃ n : ℕ, n = 6 ∧ m ∣ (11 * n - 1)) ∧ m = 65 := 
by
  sorry

end certain_number_divides_expression_l126_126323


namespace find_max_term_of_sequence_l126_126774

theorem find_max_term_of_sequence :
  ∃ m : ℕ, (m = 8) ∧ ∀ n : ℕ, (0 < n → n ≠ m → a_n = (n - 7) / (n - 5 * Real.sqrt 2)) :=
by
  sorry

end find_max_term_of_sequence_l126_126774


namespace negation_of_p_correct_l126_126772

def p := ∀ x : ℝ, x > Real.sin x

theorem negation_of_p_correct :
  (¬ p) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end negation_of_p_correct_l126_126772


namespace line_fixed_point_l126_126265

theorem line_fixed_point (m : ℝ) : ∃ x y, (∀ m, y = m * x + (2 * m + 1)) ↔ (x = -2 ∧ y = 1) :=
by
  sorry

end line_fixed_point_l126_126265


namespace waiter_customers_before_lunch_l126_126582

theorem waiter_customers_before_lunch (X : ℕ) (A : X + 20 = 49) : X = 29 := by
  -- The proof is omitted based on the instructions
  sorry

end waiter_customers_before_lunch_l126_126582


namespace ceil_x_squared_values_count_l126_126051

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l126_126051


namespace Lagrange_interpolation_poly_l126_126257

noncomputable def Lagrange_interpolation (P : ℝ → ℝ) : Prop :=
  P (-1) = -11 ∧ P (1) = -3 ∧ P (2) = 1 ∧ P (3) = 13

theorem Lagrange_interpolation_poly :
  ∃ P : ℝ → ℝ, Lagrange_interpolation P ∧ ∀ x, P x = x^3 - 2*x^2 + 3*x - 5 :=
by
  sorry

end Lagrange_interpolation_poly_l126_126257


namespace yacht_capacity_l126_126536

theorem yacht_capacity :
  ∀ (x y : ℕ), (3 * x + 2 * y = 68) → (2 * x + 3 * y = 57) → (3 * x + 6 * y = 96) :=
by
  intros x y h1 h2
  sorry

end yacht_capacity_l126_126536


namespace inequality_of_four_numbers_l126_126178

theorem inequality_of_four_numbers 
  (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h1 : a ≤ 3 * b) (h2 : b ≤ 3 * a) (h3 : a ≤ 3 * c)
  (h4 : c ≤ 3 * a) (h5 : a ≤ 3 * d) (h6 : d ≤ 3 * a)
  (h7 : b ≤ 3 * c) (h8 : c ≤ 3 * b) (h9 : b ≤ 3 * d)
  (h10 : d ≤ 3 * b) (h11 : c ≤ 3 * d) (h12 : d ≤ 3 * c) : 
  a^2 + b^2 + c^2 + d^2 < 2 * (ab + ac + ad + bc + bd + cd) :=
sorry

end inequality_of_four_numbers_l126_126178


namespace ant_to_vertices_probability_l126_126175

noncomputable def event_A_probability : ℝ :=
  1 - (Real.sqrt 3 * Real.pi / 24)

theorem ant_to_vertices_probability :
  let side_length := 4
  let event_A := "the distance from the ant to all three vertices is more than 1"
  event_A_probability = 1 - Real.sqrt 3 * Real.pi / 24
:=
sorry

end ant_to_vertices_probability_l126_126175


namespace edward_friend_scores_l126_126159

theorem edward_friend_scores (total_points friend_points edward_points : ℕ) (h1 : total_points = 13) (h2 : edward_points = 7) (h3 : friend_points = total_points - edward_points) : friend_points = 6 := 
by
  rw [h1, h2] at h3
  exact h3

end edward_friend_scores_l126_126159


namespace cuboid_total_edge_length_cuboid_surface_area_l126_126331

variables (a b c : ℝ)

theorem cuboid_total_edge_length : 4 * (a + b + c) = 4 * (a + b + c) := 
by
  sorry

theorem cuboid_surface_area : 2 * (a * b + b * c + a * c) = 2 * (a * b + b * c + a * c) := 
by
  sorry

end cuboid_total_edge_length_cuboid_surface_area_l126_126331


namespace banana_ratio_proof_l126_126134

-- Definitions based on conditions
def initial_bananas := 310
def bananas_left_on_tree := 100
def bananas_eaten := 70

-- Auxiliary calculations for clarity
def bananas_cut := initial_bananas - bananas_left_on_tree
def bananas_remaining := bananas_cut - bananas_eaten

-- Theorem we need to prove
theorem banana_ratio_proof :
  bananas_remaining / bananas_eaten = 2 :=
by
  sorry

end banana_ratio_proof_l126_126134


namespace parabola_focus_l126_126100

theorem parabola_focus (x y : ℝ) (p : ℝ) (h_eq : x^2 = 8 * y) (h_form : x^2 = 4 * p * y) : 
  p = 2 ∧ y = (x^2 / 8) ∧ (0, p) = (0, 2) :=
by
  sorry

end parabola_focus_l126_126100


namespace total_distance_travelled_l126_126930

def walking_distance_flat_surface (speed_flat : ℝ) (time_flat : ℝ) : ℝ := speed_flat * time_flat
def running_distance_downhill (speed_downhill : ℝ) (time_downhill : ℝ) : ℝ := speed_downhill * time_downhill
def walking_distance_hilly (speed_hilly_walk : ℝ) (time_hilly_walk : ℝ) : ℝ := speed_hilly_walk * time_hilly_walk
def running_distance_hilly (speed_hilly_run : ℝ) (time_hilly_run : ℝ) : ℝ := speed_hilly_run * time_hilly_run

def total_distance (ds1 ds2 ds3 ds4 : ℝ) : ℝ := ds1 + ds2 + ds3 + ds4

theorem total_distance_travelled :
  let speed_flat := 8
  let time_flat := 3
  let speed_downhill := 24
  let time_downhill := 1.5
  let speed_hilly_walk := 6
  let time_hilly_walk := 2
  let speed_hilly_run := 18
  let time_hilly_run := 1
  total_distance (walking_distance_flat_surface speed_flat time_flat) (running_distance_downhill speed_downhill time_downhill)
                            (walking_distance_hilly speed_hilly_walk time_hilly_walk) (running_distance_hilly speed_hilly_run time_hilly_run) = 90 := 
by
  sorry

end total_distance_travelled_l126_126930


namespace find_divisor_l126_126648

theorem find_divisor
  (Dividend : ℕ)
  (Quotient : ℕ)
  (Remainder : ℕ)
  (h1 : Dividend = 686)
  (h2 : Quotient = 19)
  (h3 : Remainder = 2) :
  ∃ (Divisor : ℕ), (Dividend = (Divisor * Quotient) + Remainder) ∧ Divisor = 36 :=
by
  sorry

end find_divisor_l126_126648


namespace remainder_1234567_div_256_l126_126440

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l126_126440


namespace rectangle_area_percentage_increase_l126_126410

theorem rectangle_area_percentage_increase
  (L W : ℝ) -- Original length and width of the rectangle
  (L_new : L_new = 2 * L) -- New length of the rectangle
  (W_new : W_new = 2 * W) -- New width of the rectangle
  : (4 * L * W - L * W) / (L * W) * 100 = 300 := 
by
  sorry

end rectangle_area_percentage_increase_l126_126410


namespace rhombus_shorter_diagonal_l126_126389

theorem rhombus_shorter_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d2 = 20) (h2 : area = 120) (h3 : area = (d1 * d2) / 2) : d1 = 12 :=
by 
  sorry

end rhombus_shorter_diagonal_l126_126389


namespace midpoint_trajectory_l126_126113

theorem midpoint_trajectory (x y : ℝ) (x0 y0 : ℝ)
  (h_circle : x0^2 + y0^2 = 4)
  (h_tangent : x0 * x + y0 * y = 4)
  (h_x0 : x0 = 2 / x)
  (h_y0 : y0 = 2 / y) :
  x^2 * y^2 = x^2 + y^2 :=
sorry

end midpoint_trajectory_l126_126113


namespace partI_partII_l126_126623

noncomputable def f (x m : ℝ) : ℝ := Real.log x - m * x
noncomputable def f' (x m : ℝ) : ℝ := (1 / x) - m

theorem partI (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f x m = -1) → m = 1 := by
  sorry

theorem partII (x1 x2 : ℝ) (h1 : e ^ x1 ≤ x2) (h2 : f x1 1 = 0) (h3 : f x2 1 = 0) :
  ∃ y : ℝ, y = (x1 - x2) * f' (x1 + x2) 1 ∧ y = 2 / (1 + Real.exp 1) := by
  sorry

end partI_partII_l126_126623


namespace sum_gcf_lcm_eq_28_l126_126880

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l126_126880


namespace remainder_div_1234567_256_l126_126445

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l126_126445


namespace sin_180_eq_0_l126_126301

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l126_126301


namespace sine_180_eq_zero_l126_126294

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l126_126294


namespace distinct_prime_factors_90_l126_126782

theorem distinct_prime_factors_90 : ∃ (n : Nat), n = 3 ∧ prime 2 ∧ prime 3 ∧ prime 5 ∧ 90 = 2 * 3^2 * 5 :=
by
  sorry

end distinct_prime_factors_90_l126_126782


namespace mr_william_farm_tax_l126_126744

noncomputable def total_tax_collected : ℝ := 3840
noncomputable def mr_william_percentage : ℝ := 16.666666666666668 / 100  -- Convert percentage to decimal

theorem mr_william_farm_tax : (total_tax_collected * mr_william_percentage) = 640 := by
  sorry

end mr_william_farm_tax_l126_126744


namespace compare_y_l126_126177

-- Define the points M and N lie on the graph of y = -5/x
def on_inverse_proportion_curve (x y : ℝ) : Prop :=
  y = -5 / x

-- Main theorem to be proven
theorem compare_y (x1 y1 x2 y2 : ℝ) (h1 : on_inverse_proportion_curve x1 y1) (h2 : on_inverse_proportion_curve x2 y2) (hx : x1 > 0 ∧ x2 < 0) : y1 < y2 :=
by
  sorry

end compare_y_l126_126177


namespace gcd_lcm_sum_8_12_l126_126901

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l126_126901


namespace sin_180_eq_zero_l126_126307

theorem sin_180_eq_zero : Real.sin (180 * Real.pi / 180) = 0 := by
  -- Simplifying the angle, 180 degrees = π radians
  let angle := 180 * Real.pi / 180
  have h : angle = Real.pi := by
    simp [angle, Real.pi]
  rw h
  -- From the unit circle, we know the sine of angle π is 0
  exact Real.sin_pi

end sin_180_eq_zero_l126_126307


namespace repeating_decimal_to_fraction_l126_126604

/--
Express \(2.\overline{06}\) as a reduced fraction, given that \(0.\overline{01} = \frac{1}{99}\)
-/
theorem repeating_decimal_to_fraction : 
  (0.01:ℚ) = 1 / 99 → (2.06:ℚ) = 68 / 33 := 
by 
  sorry 

end repeating_decimal_to_fraction_l126_126604


namespace tan_seven_pi_over_six_l126_126607
  
theorem tan_seven_pi_over_six :
  Real.tan (7 * Real.pi / 6) = 1 / Real.sqrt 3 :=
sorry

end tan_seven_pi_over_six_l126_126607


namespace proof_least_sum_l126_126662

noncomputable def least_sum (m n : ℕ) (h1 : Nat.gcd (m + n) 330 = 1) 
                           (h2 : n^n ∣ m^m) (h3 : ¬(n ∣ m)) : ℕ :=
  m + n

theorem proof_least_sum :
  ∃ m n : ℕ, Nat.gcd (m + n) 330 = 1 ∧ n^n ∣ m^m ∧ ¬(n ∣ m) ∧ m + n = 390 :=
by
  sorry

end proof_least_sum_l126_126662


namespace Tim_paid_amount_l126_126862

theorem Tim_paid_amount (original_price : ℝ) (discount_percentage : ℝ) (discounted_price : ℝ) 
    (h1 : original_price = 1200) (h2 : discount_percentage = 0.15) 
    (discount_amount : ℝ) (h3 : discount_amount = original_price * discount_percentage) 
    (h4 : discounted_price = original_price - discount_amount) : discounted_price = 1020 := 
    by {
        sorry
    }

end Tim_paid_amount_l126_126862


namespace remainder_of_1999_pow_11_mod_8_l126_126109

theorem remainder_of_1999_pow_11_mod_8 :
  (1999 ^ 11) % 8 = 7 :=
  sorry

end remainder_of_1999_pow_11_mod_8_l126_126109


namespace most_stable_student_l126_126361

-- Define the variances for the four students
def variance_A (SA2 : ℝ) : Prop := SA2 = 0.15
def variance_B (SB2 : ℝ) : Prop := SB2 = 0.32
def variance_C (SC2 : ℝ) : Prop := SC2 = 0.5
def variance_D (SD2 : ℝ) : Prop := SD2 = 0.25

-- Theorem proving that the most stable student is A
theorem most_stable_student {SA2 SB2 SC2 SD2 : ℝ} 
  (hA : variance_A SA2) 
  (hB : variance_B SB2)
  (hC : variance_C SC2)
  (hD : variance_D SD2) : 
  SA2 < SB2 ∧ SA2 < SC2 ∧ SA2 < SD2 :=
by
  rw [variance_A, variance_B, variance_C, variance_D] at *
  sorry

end most_stable_student_l126_126361


namespace pages_per_day_l126_126752

-- Define the given conditions
def total_pages : ℕ := 957
def total_days : ℕ := 47

-- State the theorem based on the conditions and the required proof
theorem pages_per_day (p : ℕ) (d : ℕ) (h1 : p = total_pages) (h2 : d = total_days) :
  p / d = 20 := by
  sorry

end pages_per_day_l126_126752


namespace sin_180_is_zero_l126_126298

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l126_126298


namespace quadratic_inequality_real_solutions_l126_126611

theorem quadratic_inequality_real_solutions (c : ℝ) (h1 : 0 < c) (h2 : c < 16) :
  ∃ x : ℝ, x^2 - 8*x + c < 0 :=
sorry

end quadratic_inequality_real_solutions_l126_126611


namespace f_neg4_plus_f_0_range_of_a_l126_126969

open Real

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then log x / log 2 else if x < 0 then -log (-x) / log 2 else 0

/- Prove that f(-4) + f(0) = -2 given the function properties -/
theorem f_neg4_plus_f_0 : f (-4) + f 0 = -2 :=
sorry

/- Prove the range of a such that f(a) > f(-a) is a > 1 or -1 < a < 0 given the function properties -/
theorem range_of_a (a : ℝ) : f a > f (-a) ↔ a > 1 ∨ (-1 < a ∧ a < 0) :=
sorry

end f_neg4_plus_f_0_range_of_a_l126_126969


namespace sum_gcd_lcm_eight_twelve_l126_126877

theorem sum_gcd_lcm_eight_twelve : 
  let a := 8
  let b := 12
  gcd a b + lcm a b = 28 := sorry

end sum_gcd_lcm_eight_twelve_l126_126877


namespace sandwich_cost_l126_126872

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end sandwich_cost_l126_126872


namespace profit_percentage_B_l126_126425

-- Definitions based on conditions:
def CP_A : ℝ := 150  -- Cost price for A
def profit_percentage_A : ℝ := 0.20  -- Profit percentage for A
def SP_C : ℝ := 225  -- Selling price for C

-- Lean statement for the problem:
theorem profit_percentage_B : (SP_C - (CP_A * (1 + profit_percentage_A))) / (CP_A * (1 + profit_percentage_A)) * 100 = 25 := 
by 
  sorry

end profit_percentage_B_l126_126425


namespace remainder_1234567_div_256_l126_126439

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l126_126439


namespace no_integer_roots_l126_126966

def cubic_polynomial (a b c d x : ℤ) : ℤ :=
  a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots (a b c d : ℤ) (h1 : cubic_polynomial a b c d 1 = 2015) (h2 : cubic_polynomial a b c d 2 = 2017) :
  ∀ x : ℤ, cubic_polynomial a b c d x ≠ 2016 :=
by
  sorry

end no_integer_roots_l126_126966


namespace simplify_fraction_l126_126948

theorem simplify_fraction (a : ℝ) (h1 : a ≠ 4) (h2 : a ≠ -4) : 
  (2 * a / (a^2 - 16) - 1 / (a - 4) = 1 / (a + 4)) := 
by 
  sorry 

end simplify_fraction_l126_126948


namespace raccoon_carrots_hid_l126_126818

theorem raccoon_carrots_hid 
  (r : ℕ)
  (b : ℕ)
  (h1 : 5 * r = 8 * b)
  (h2 : b = r - 3) 
  : 5 * r = 40 :=
by
  sorry

end raccoon_carrots_hid_l126_126818


namespace rational_solutions_are_integers_l126_126762

-- Given two integers a and b, and two equations with rational solutions
variables (a b : ℤ)

-- The first equation is y - 2x = a
def eq1 (y x : ℚ) : Prop := y - 2 * x = a

-- The second equation is y^2 - xy + x^2 = b
def eq2 (y x : ℚ) : Prop := y^2 - x * y + x^2 = b

-- We want to prove that if y and x are rational solutions, they must be integers
theorem rational_solutions_are_integers (y x : ℚ) (h1 : eq1 a y x) (h2 : eq2 b y x) : 
    ∃ (y_int x_int : ℤ), y = y_int ∧ x = x_int :=
sorry

end rational_solutions_are_integers_l126_126762


namespace lydia_candy_problem_l126_126828

theorem lydia_candy_problem :
  ∃ m: ℕ, (∀ k: ℕ, (k * 24 = Nat.lcm (Nat.lcm 16 18) 20) → k ≥ m) ∧ 24 * m = Nat.lcm (Nat.lcm 16 18) 20 ∧ m = 30 :=
by
  sorry

end lydia_candy_problem_l126_126828


namespace find_f_cos_10_l126_126469

noncomputable def f : ℝ → ℝ := sorry

axiom f_def (x : ℝ) : f (Real.sin x) = Real.cos (3 * x)

theorem find_f_cos_10 : f (Real.cos (10 * Real.pi / 180)) = -1/2 := by
  sorry

end find_f_cos_10_l126_126469


namespace total_volume_of_four_cubes_is_500_l126_126555

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l126_126555


namespace new_person_weight_l126_126414

theorem new_person_weight (w : ℝ) (avg_increase : ℝ) (replaced_person_weight : ℝ) (num_people : ℕ) 
(H1 : avg_increase = 4.8) (H2 : replaced_person_weight = 62) (H3 : num_people = 12) : 
w = 119.6 :=
by
  -- We could provide the intermediate steps as definitions here but for the theorem statement, we just present the goal.
  sorry

end new_person_weight_l126_126414


namespace probability_six_distinct_numbers_l126_126400

theorem probability_six_distinct_numbers (eight_sided_dice : Finset ℕ) (h : eight_sided_dice.card = 8) : 
  (∃ dice_rolls : Finset (fin 8) → ℕ, dice_rolls.card = 6) → 
  (probability : ℚ) = 315 / 4096 :=
by
  sorry

end probability_six_distinct_numbers_l126_126400


namespace sum_of_incircle_areas_l126_126826

variables {a b c : ℝ} (ABC : Triangle ℝ) (s K r : ℝ)
  (hs : s = (a + b + c) / 2)
  (hK : K = Real.sqrt (s * (s - a) * (s - b) * (s - c)))
  (hr : r = K / s)

theorem sum_of_incircle_areas :
  let larger_circle_area := π * r^2
  let smaller_circle_area := π * (r / 2)^2
  larger_circle_area + 3 * smaller_circle_area = 7 * π * r^2 / 4 :=
sorry

end sum_of_incircle_areas_l126_126826


namespace measure_of_angle_C_l126_126980

variable (A B C : Real)

theorem measure_of_angle_C (h1 : 4 * Real.sin A + 2 * Real.cos B = 4) 
                           (h2 : (1/2) * Real.sin B + Real.cos A = Real.sqrt 3 / 2) :
                           C = Real.pi / 6 :=
by
  sorry

end measure_of_angle_C_l126_126980


namespace similar_triangles_height_l126_126689

theorem similar_triangles_height (h_small: ℝ) (area_ratio: ℝ) (h_large: ℝ) :
  h_small = 5 ∧ area_ratio = 1/9 ∧ h_large = 3 * h_small → h_large = 15 :=
by
  intro h 
  sorry

end similar_triangles_height_l126_126689


namespace factorization_correct_l126_126525

theorem factorization_correct {c d : ℤ} (h1 : c + 4 * d = 4) (h2 : c * d = -32) :
  c - d = 12 :=
by
  sorry

end factorization_correct_l126_126525


namespace triangle_sides_inequality_l126_126962

theorem triangle_sides_inequality (a b c : ℝ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) :
  a^2 - 2 * a * b + b^2 - c^2 < 0 :=
by
  sorry

end triangle_sides_inequality_l126_126962


namespace power_mean_inequality_l126_126758

variables {a b c : ℝ}
variables {n p q r : ℕ}

theorem power_mean_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hn : 0 < n)
  (hpqr_nonneg : 0 ≤ p ∧ 0 ≤ q ∧ 0 ≤ r)
  (sum_pqr : p + q + r = n) :
  a^n + b^n + c^n ≥ a^p * b^q * c^r + a^r * b^p * c^q + a^q * b^r * c^p :=
sorry

end power_mean_inequality_l126_126758


namespace complete_the_square_l126_126129

theorem complete_the_square (x : ℝ) : (x^2 + 2 * x - 1 = 0) -> ((x + 1)^2 = 2) :=
by
  intro h
  sorry

end complete_the_square_l126_126129


namespace one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l126_126513

theorem one_div_add_one_div_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) := 
sorry

theorem one_div_add_one_div_not_upper_bounded (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M := 
sorry

theorem one_div_add_one_div_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (2 ≤ (1 / a + 1 / b) ∧ ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M) := 
sorry

end one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l126_126513


namespace diagonals_in_convex_polygon_with_30_sides_l126_126344

theorem diagonals_in_convex_polygon_with_30_sides : 
  let n := 30 in
  ∑ i j in finset.range n, (i + 1 - 1) * (j + 1 - 1) * (nat.choose 30 2 - 30) / 2 = 202 := 
by
  sorry

end diagonals_in_convex_polygon_with_30_sides_l126_126344


namespace range_of_m_l126_126485

open Set

def set_A : Set ℝ := {x | -3 ≤ x ∧ x ≤ 4}
def set_B (m : ℝ) : Set ℝ := {x | (m - 1) ≤ x ∧ x ≤ (3 * m - 2)}

theorem range_of_m (m : ℝ) : (set_B m ⊆ set_A) ↔ m ≤ 4 :=
by sorry

end range_of_m_l126_126485


namespace program_arrangement_possible_l126_126360

theorem program_arrangement_possible (initial_programs : ℕ) (additional_programs : ℕ) 
  (h_initial: initial_programs = 6) (h_additional: additional_programs = 2) : 
  ∃ arrangements, arrangements = 56 :=
by
  sorry

end program_arrangement_possible_l126_126360


namespace sum_of_GCF_and_LCM_l126_126887

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l126_126887


namespace least_integer_square_double_l126_126691

theorem least_integer_square_double (x : ℤ) : x^2 = 2 * x + 50 → x = -5 :=
by
  sorry

end least_integer_square_double_l126_126691


namespace min_value_fraction_l126_126184

theorem min_value_fraction (m n : ℝ) (h₀ : m > 0) (h₁ : n > 0) (h₂ : m + 2 * n = 1) : 
  (1 / m + 1 / n) ≥ 3 + 2 * Real.sqrt 2 :=
by sorry

end min_value_fraction_l126_126184


namespace correct_statements_l126_126914

-- Statement B
def statementB : Prop := 
∀ x : ℝ, x < 1/2 → (∃ y : ℝ, y = 2 * x + 1 / (2 * x - 1) ∧ y = -1)

-- Statement D
def statementD : Prop :=
∃ y : ℝ, (∀ x : ℝ, y = 1 / (Real.sin x) ^ 2 + 4 / (Real.cos x) ^ 2) ∧ y = 9

-- Combined proof problem
theorem correct_statements : statementB ∧ statementD :=
sorry

end correct_statements_l126_126914


namespace car_catches_up_in_6_hours_l126_126142

-- Conditions
def speed_truck := 40 -- km/h
def speed_car_initial := 50 -- km/h
def speed_car_increment := 5 -- km/h
def distance_between := 135 -- km

-- Solution: car catches up in 6 hours
theorem car_catches_up_in_6_hours : 
  ∃ n : ℕ, n = 6 ∧ (n * speed_truck + distance_between) ≤ (n * speed_car_initial + (n * (n - 1) / 2 * speed_car_increment)) := 
by
  sorry

end car_catches_up_in_6_hours_l126_126142


namespace vincent_earnings_l126_126870

theorem vincent_earnings 
  (price_fantasy_book : ℕ)
  (num_fantasy_books_per_day : ℕ)
  (num_lit_books_per_day : ℕ)
  (num_days : ℕ)
  (h1 : price_fantasy_book = 4)
  (h2 : num_fantasy_books_per_day = 5)
  (h3 : num_lit_books_per_day = 8)
  (h4 : num_days = 5) :
  let price_lit_book := price_fantasy_book / 2
      daily_earnings_fantasy := price_fantasy_book * num_fantasy_books_per_day
      daily_earnings_lit := price_lit_book * num_lit_books_per_day
      total_daily_earnings := daily_earnings_fantasy + daily_earnings_lit
      total_earnings := total_daily_earnings * num_days
  in total_earnings = 180 := 
  by 
  {
    sorry
  }

end vincent_earnings_l126_126870


namespace point_in_fourth_quadrant_l126_126185

theorem point_in_fourth_quadrant (m : ℝ) : 0 < m ∧ 2 - m < 0 ↔ m > 2 := 
by 
  sorry

end point_in_fourth_quadrant_l126_126185


namespace robot_distance_covered_l126_126424

theorem robot_distance_covered :
  let start1 := -3
  let end1 := -8
  let end2 := 6
  let distance1 := abs (end1 - start1)
  let distance2 := abs (end2 - end1)
  distance1 + distance2 = 19 := by
  sorry

end robot_distance_covered_l126_126424


namespace exponent_evaluation_l126_126076

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end exponent_evaluation_l126_126076


namespace total_volume_of_four_cubes_is_500_l126_126554

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l126_126554


namespace limit_sequence_l126_126944

open Filter
open Real

noncomputable def sequence_limit := 
  filter.tendsto (λ n : ℕ, (sqrt (3 * n - 1) - real.cbrt (125 * n^3 + n)) / (real.rpow n (1 / 5) - n)) at_top (nhds 5)

theorem limit_sequence: sequence_limit :=
  sorry

end limit_sequence_l126_126944


namespace siblings_gmat_scores_l126_126860

-- Define the problem conditions
variables (x y z : ℝ)

theorem siblings_gmat_scores (h1 : x - y = 1/3) (h2 : z = (x + y) / 2) : 
  y = x - 1/3 ∧ z = x - 1/6 :=
by
  sorry

end siblings_gmat_scores_l126_126860


namespace hours_rained_l126_126169

theorem hours_rained (total_hours non_rain_hours rained_hours : ℕ)
 (h_total : total_hours = 8)
 (h_non_rain : non_rain_hours = 6)
 (h_rain_eq : rained_hours = total_hours - non_rain_hours) :
 rained_hours = 2 := 
by
  sorry

end hours_rained_l126_126169


namespace problem_solution_l126_126021

variables {R : Type} [LinearOrder R]

def M (x y : R) : R := max x y
def m (x y : R) : R := min x y

theorem problem_solution (p q r s t : R) (h : p < q) (h1 : q < r) (h2 : r < s) (h3 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q :=
by
  sorry

end problem_solution_l126_126021


namespace ThreeDigitEvenNumbersCount_l126_126634

theorem ThreeDigitEvenNumbersCount : 
  let a := 100
  let max := 998
  let d := 2
  let n := (max - a) / d + 1
  100 < 999 ∧ 100 % 2 = 0 ∧ max % 2 = 0 
  → d > 0 
  → n = 450 :=
by
  sorry

end ThreeDigitEvenNumbersCount_l126_126634


namespace sum_of_19th_set_is_29572_l126_126333

-- Define the sequence rules and properties.
noncomputable def first_element_of_set : ℕ → ℕ
| 0     := 1
| (n+1) := first_element_of_set n + 2 * (n + 1) - 1

def set_elements (n : ℕ) : List ℕ := 
List.range (n+1) |>.map (λ k => first_element_of_set n + 2 * k)

def sum_of_set (n : ℕ) : ℕ :=
(set_elements n).sum

-- Proving the specific case for \( \tilde{S}_{19} \)
theorem sum_of_19th_set_is_29572 : sum_of_set 19 = 29572 :=
by
  -- proof steps can be added here
  sorry

end sum_of_19th_set_is_29572_l126_126333


namespace no_integer_roots_p_eq_2016_l126_126963

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l126_126963


namespace tan_arithmetic_sequence_l126_126029

theorem tan_arithmetic_sequence {a : ℕ → ℝ}
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + n * d)
  (h_sum : a 1 + a 7 + a 13 = Real.pi) :
  Real.tan (a 2 + a 12) = - Real.sqrt 3 :=
sorry

end tan_arithmetic_sequence_l126_126029


namespace quadratic_function_property_l126_126827

theorem quadratic_function_property
    (a b c : ℝ)
    (f : ℝ → ℝ)
    (h_f_def : ∀ x, f x = a * x^2 + b * x + c)
    (h_vertex : f (-2) = a^2)
    (h_point : f (-1) = 6)
    (h_vertex_condition : -b / (2 * a) = -2)
    (h_a_neg : a < 0) :
    (a + c) / b = 1 / 2 :=
by
  sorry

end quadratic_function_property_l126_126827


namespace football_tournament_max_points_l126_126499

theorem football_tournament_max_points:
  ∃ N : ℕ, let teams := 15,
               matches := teams.choose 2,
               max_points := 105*3 in
    (∀ pts. ∃ six_teams : Finset ℕ, (six_teams.card = 6 ∧ ∀ x ∈ six_teams, pts x ≥ N) → 6 * N ≤ max_points) ∧
    (∀ pts. ∃ six_teams : Finset ℕ, (six_teams.card = 6 ∧ ∀ x ∈ six_teams, pts x ≥ 35) → 6 * 35 > max_points) :=
sorry

end football_tournament_max_points_l126_126499


namespace inequality_proof_l126_126207

theorem inequality_proof (a b c x y z : ℝ) (h1 : a > 0) (h2 : b > 0) 
(h3 : c > 0) (h4 : x > 0) (h5 : y > 0) (h6 : z > 0) 
(h7 : a * y + b * x = c) (h8 : c * x + a * z = b) 
(h9 : b * z + c * y = a) :
x / (1 - y * z) + y / (1 - z * x) + z / (1 - x * y) ≤ 2 :=
sorry

end inequality_proof_l126_126207


namespace ratio_of_x_to_y_l126_126493

theorem ratio_of_x_to_y (x y : ℝ) (h : (3 * x - 2 * y) / (2 * x + y) = 3 / 4) : x / y = 11 / 6 := 
by
  sorry

end ratio_of_x_to_y_l126_126493


namespace geometric_prog_common_ratio_one_l126_126357

variable {x y z : ℝ}
variable {r : ℝ}

theorem geometric_prog_common_ratio_one
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hxy : x ≠ y) (hyz : y ≠ z) (hzx : z ≠ x)
  (hgeom : ∃ a : ℝ, a = x * (y - z) ∧ a * r = y * (z - x) ∧ a * r^2 = z * (x - y))
  (hprod : (x * (y - z)) * (y * (z - x)) * (z * (x - y)) * r^3 = (y * (z - x))^2) : 
  r = 1 := sorry

end geometric_prog_common_ratio_one_l126_126357


namespace ice_cream_flavors_l126_126801

theorem ice_cream_flavors : (Nat.choose 8 3) = 56 := 
by {
    sorry
}

end ice_cream_flavors_l126_126801


namespace find_a_from_roots_l126_126971

theorem find_a_from_roots (a : ℝ) :
  let A := {x | (x = a) ∨ (x = a - 1)}
  2 ∈ A → a = 2 ∨ a = 3 :=
by
  intros A h
  sorry

end find_a_from_roots_l126_126971


namespace evaluate_f_2x_l126_126024

def f (x : ℝ) : ℝ := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end evaluate_f_2x_l126_126024


namespace triangle_area_l126_126030

theorem triangle_area (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (h₄ : a * a + b * b = c * c) :
  (1/2) * a * b = 30 :=
by
  sorry

end triangle_area_l126_126030


namespace arithmetic_sequence_property_l126_126193

theorem arithmetic_sequence_property 
  (a : ℕ → ℤ) 
  (h₁ : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 1 + 2 * a 8 + a 15 = 96) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_sequence_property_l126_126193


namespace road_length_l126_126279

theorem road_length 
  (D : ℕ) (N1 : ℕ) (t : ℕ) (d1 : ℝ) (N_extra : ℝ) 
  (h1 : D = 300) (h2 : N1 = 35) (h3 : t = 100) (h4 : d1 = 2.5) (h5 : N_extra = 52.5) : 
  ∃ L : ℝ, L = 3 := 
by {
  sorry
}

end road_length_l126_126279


namespace minimize_slope_at_one_l126_126476

open Real

noncomputable def f (a x : ℝ) : ℝ :=
  2 * a * x^2 - (1 / (a * x))

noncomputable def f_deriv (a x : ℝ) : ℝ :=
  4 * a * x - (1 / (a * x^2))

noncomputable def slope_at_one (a : ℝ) : ℝ :=
  f_deriv a 1

theorem minimize_slope_at_one : ∀ a : ℝ, a > 0 → slope_at_one a ≥ 4 ∧ (slope_at_one a = 4 ↔ a = 1 / 2) :=
by 
  sorry

end minimize_slope_at_one_l126_126476


namespace probability_coin_die_sum_even_l126_126243

noncomputable def probability_sum_even : ℝ :=
  sorry -- this will be calculated based on the given problem

theorem probability_coin_die_sum_even :
  let three_fair_coins := [true, true, true] -- Represents tossing three fair coins
  ∀ (coins : list bool),
  coins.length = 3 →
  let heads := coins.count(λ b, b = true) in
  let dice_rolls := list.nth_le [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6] (heads - 1) (by sorry) in
  (if heads = 0 then 1 else
     if heads = 1 then 1 / 2 else
     if heads = 2 then (1 / 2 * 1 / 2 + 1 / 2 * 1 / 2) else
     (1 / 2))
  = (9/16) :=
by sorry

end probability_coin_die_sum_even_l126_126243


namespace find_dividend_l126_126747

theorem find_dividend (divisor : ℕ) (partial_quotient : ℕ) (dividend : ℕ) 
                       (h_divisor : divisor = 12)
                       (h_partial_quotient : partial_quotient = 909809) 
                       (h_calculation : dividend = divisor * partial_quotient) : 
                       dividend = 10917708 :=
by
  rw [h_divisor, h_partial_quotient] at h_calculation
  exact h_calculation


end find_dividend_l126_126747


namespace six_applications_of_s_l126_126206

def s (θ : ℝ) : ℝ :=
  1 / (2 - θ)

theorem six_applications_of_s (θ : ℝ) : s (s (s (s (s (s θ))))) = -1 / 29 :=
by
  have h : θ = 30 := rfl
  rw h
  sorry

end six_applications_of_s_l126_126206


namespace total_volume_of_four_boxes_l126_126550

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l126_126550


namespace problem1_problem2_l126_126974

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a) * Real.exp x

theorem problem1 (a : ℝ) : 
  (∀ x ∈ Set.Icc a - 1 x, f a x ≤ f a (x + 1)) ∧ (∀ x < a - 1,  f a x ≥ f a (x - 1)) := 
sorry

noncomputable def F (x : ℝ) : ℝ := (x - 2) * Real.exp x - x + Real.log x

theorem problem2 : 
  ∃ m, ∃ x ∈ Set.Icc (1 / 4:ℝ) 1, f 2 x - x + @Real.log x ∈ -4 < m < - 3 := 
sorry

end problem1_problem2_l126_126974


namespace inequality_proof_l126_126074

open Real

theorem inequality_proof
  (a b c x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hx_cond : 1 / x + 1 / y + 1 / z = 1) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ x + b ^ y + c ^ z ≥ (4 * a * b * c * x * y * z) / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_proof_l126_126074


namespace last_operation_ends_at_eleven_am_l126_126812

-- Definitions based on conditions
def operation_duration : ℕ := 45 -- duration of each operation in minutes
def start_time : ℕ := 8 * 60 -- start time of the first operation in minutes since midnight
def interval : ℕ := 15 -- interval between operations in minutes
def total_operations : ℕ := 10 -- total number of operations

-- Compute the start time of the last operation (10th operation)
def start_time_last_operation : ℕ := start_time + interval * (total_operations - 1)

-- Compute the end time of the last operation
def end_time_last_operation : ℕ := start_time_last_operation + operation_duration

-- End time of the last operation expected to be 11:00 a.m. in minutes since midnight
def expected_end_time : ℕ := 11 * 60 

theorem last_operation_ends_at_eleven_am : 
  end_time_last_operation = expected_end_time := by
  sorry

end last_operation_ends_at_eleven_am_l126_126812


namespace avg_decreased_by_one_l126_126387

noncomputable def avg_decrease (n : ℕ) (average_initial : ℝ) (obs_new : ℝ) : ℝ :=
  (n * average_initial + obs_new) / (n + 1)

theorem avg_decreased_by_one (init_avg : ℝ) (obs_new : ℝ) (num_obs : ℕ)
  (h₁ : num_obs = 6)
  (h₂ : init_avg = 12)
  (h₃ : obs_new = 5) :
  init_avg - avg_decrease num_obs init_avg obs_new = 1 :=
by
  sorry

end avg_decreased_by_one_l126_126387


namespace max_min_sundays_in_month_l126_126182

def week_days : ℕ := 7
def min_month_days : ℕ := 28
def months_days (d : ℕ) : Prop := d = 28 ∨ d = 30 ∨ d = 31

theorem max_min_sundays_in_month (d : ℕ) (h1 : months_days d) :
  4 ≤ (d / week_days) + ite (d % week_days > 0) 1 0 ∧ (d / week_days) + ite (d % week_days > 0) 1 0 ≤ 5 :=
by
  sorry

end max_min_sundays_in_month_l126_126182


namespace calc_6_4_3_199_plus_100_l126_126150

theorem calc_6_4_3_199_plus_100 (a b : ℕ) (h_a : a = 199) (h_b : b = 100) :
  6 * a + 4 * a + 3 * a + a + b = 2886 :=
by
  sorry

end calc_6_4_3_199_plus_100_l126_126150


namespace probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l126_126985

def total_balls := 20
def red_balls := 10
def yellow_balls := 6
def white_balls := 4
def initial_white_balls_probability := (white_balls : ℚ) / total_balls
def initial_yellow_or_red_balls_probability := (yellow_balls + red_balls : ℚ) / total_balls

def removed_red_balls := 2
def removed_white_balls := 2
def remaining_balls := total_balls - (removed_red_balls + removed_white_balls)
def remaining_white_balls := white_balls - removed_white_balls
def remaining_white_balls_probability := (remaining_white_balls : ℚ) / remaining_balls

theorem probability_white_ball_initial : initial_white_balls_probability = 1 / 5 := by sorry
theorem probability_yellow_or_red_ball_initial : initial_yellow_or_red_balls_probability = 4 / 5 := by sorry
theorem probability_white_ball_after_removal : remaining_white_balls_probability = 1 / 8 := by sorry

end probability_white_ball_initial_probability_yellow_or_red_ball_initial_probability_white_ball_after_removal_l126_126985


namespace sine_product_identity_l126_126952

open Real

theorem sine_product_identity :
  sin 12 * sin 36 * sin 54 * sin 72 = 1 / 16 := by
  have h1 : sin 72 = cos 18 := by sorry
  have h2 : sin 54 = cos 36 := by sorry
  have h3 : ∀ θ, sin θ * cos θ = 1 / 2 * sin (2 * θ) := by sorry
  have h4 : ∀ θ, cos (2 * θ) = 2 * cos θ ^ 2 - 1 := by sorry
  have h5 : cos 36 = 1 - 2 * (sin 18) ^ 2 := by sorry
  have h6 : ∀ θ, sin (180 - θ) = sin θ := by sorry
  sorry

end sine_product_identity_l126_126952


namespace diagonals_of_30_sided_polygon_l126_126345

theorem diagonals_of_30_sided_polygon : 
  ∀ (n : ℕ), n = 30 → (n * (n - 3)) / 2 = 405 := 
by
  intro n h
  rw h
  simp
  sorry

end diagonals_of_30_sided_polygon_l126_126345


namespace MrKishore_petrol_expense_l126_126726

theorem MrKishore_petrol_expense 
  (rent milk groceries education misc savings salary expenses petrol : ℝ)
  (h_rent : rent = 5000)
  (h_milk : milk = 1500)
  (h_groceries : groceries = 4500)
  (h_education : education = 2500)
  (h_misc : misc = 700)
  (h_savings : savings = 1800)
  (h_salary : salary = 18000)
  (h_expenses_equation : expenses = rent + milk + groceries + education + petrol + misc)
  (h_savings_equation : savings = salary * 0.10)
  (h_total_equation : salary = expenses + savings) :
  petrol = 2000 :=
by
  sorry

end MrKishore_petrol_expense_l126_126726


namespace smallest_n_for_divisibility_property_l126_126665

theorem smallest_n_for_divisibility_property (k : ℕ) : ∃ n : ℕ, n = k + 2 ∧ ∀ (S : Finset ℤ), 
  S.card = n → 
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ (a ≠ b ∧ (a + b) % (2 * k + 1) = 0 ∨ (a - b) % (2 * k + 1) = 0) :=
by
sorry

end smallest_n_for_divisibility_property_l126_126665


namespace ofelia_ratio_is_two_l126_126515

noncomputable def OfeliaSavingsRatio : ℝ :=
  let january_savings := 10
  let may_savings := 160
  let x := (may_savings / january_savings)^(1/4)
  x

theorem ofelia_ratio_is_two : OfeliaSavingsRatio = 2 := by
  sorry

end ofelia_ratio_is_two_l126_126515


namespace steel_more_by_l126_126135

variable {S T C k : ℝ}
variable (k_greater_than_zero : k > 0)
variable (copper_weight : C = 90)
variable (S_twice_T : S = 2 * T)
variable (S_minus_C : S = C + k)
variable (total_eq : 20 * S + 20 * T + 20 * C = 5100)

theorem steel_more_by (k): k = 20 := by
  sorry

end steel_more_by_l126_126135


namespace least_positive_integer_divisible_by_primes_gt_5_l126_126544

theorem least_positive_integer_divisible_by_primes_gt_5 : ∃ n : ℕ, n = 7 * 11 * 13 ∧ ∀ k : ℕ, (k > 0 ∧ (k % 7 = 0) ∧ (k % 11 = 0) ∧ (k % 13 = 0)) → k ≥ 1001 := 
sorry

end least_positive_integer_divisible_by_primes_gt_5_l126_126544


namespace incorrect_description_is_A_l126_126558

-- Definitions for the conditions
def description_A := "Increasing the concentration of reactants increases the percentage of activated molecules, accelerating the reaction rate."
def description_B := "Increasing the pressure of a gaseous reaction system increases the number of activated molecules per unit volume, accelerating the rate of the gas reaction."
def description_C := "Raising the temperature of the reaction increases the percentage of activated molecules, increases the probability of effective collisions, and increases the reaction rate."
def description_D := "Catalysts increase the reaction rate by changing the reaction path and lowering the activation energy required for the reaction."

-- Problem Statement
theorem incorrect_description_is_A :
  description_A ≠ correct :=
  sorry

end incorrect_description_is_A_l126_126558


namespace proof_problem_l126_126327

theorem proof_problem 
  (a b : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : b ≠ 0) : 
  |a / b + b / a| ≥ 2 := 
sorry

end proof_problem_l126_126327


namespace value_of_expression_l126_126846

theorem value_of_expression (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 10) : (12 * y - 4)^2 = 80 := 
by 
  sorry

end value_of_expression_l126_126846


namespace amount_saved_l126_126071

-- Initial conditions as definitions
def initial_amount : ℕ := 6000
def cost_ballpoint_pen : ℕ := 3200
def cost_eraser : ℕ := 1000
def cost_candy : ℕ := 500

-- Mathematical equivalent proof problem as a Lean theorem statement
theorem amount_saved : initial_amount - (cost_ballpoint_pen + cost_eraser + cost_candy) = 1300 := 
by 
  -- Proof is omitted
  sorry

end amount_saved_l126_126071


namespace vincent_earnings_after_5_days_l126_126871

def fantasy_book_price : ℕ := 4
def daily_fantasy_books_sold : ℕ := 5
def literature_book_price : ℕ := fantasy_book_price / 2
def daily_literature_books_sold : ℕ := 8
def days : ℕ := 5

def daily_earnings : ℕ :=
  (fantasy_book_price * daily_fantasy_books_sold) +
  (literature_book_price * daily_literature_books_sold)

def total_earnings (d : ℕ) : ℕ :=
  daily_earnings * d

theorem vincent_earnings_after_5_days : total_earnings days = 180 := by
  sorry

end vincent_earnings_after_5_days_l126_126871


namespace factor_expression_l126_126459

theorem factor_expression (m n x y : ℝ) :
  m * (x - y) + n * (y - x) = (x - y) * (m - n) := by
  sorry

end factor_expression_l126_126459


namespace aubree_total_animals_l126_126942

noncomputable def total_animals_seen : Nat :=
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks

theorem aubree_total_animals :
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks = 130 := by
  -- Define all constants and conditions
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10

  -- State the equation
  show morning_total + new_beavers + new_chipmunks = 130 from sorry

end aubree_total_animals_l126_126942


namespace not_all_angles_less_than_60_l126_126697

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l126_126697


namespace find_equation_AC_l126_126765

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l126_126765


namespace students_wearing_other_colors_l126_126568

variable (total_students blue_percentage red_percentage green_percentage : ℕ)
variable (h_total : total_students = 600)
variable (h_blue : blue_percentage = 45)
variable (h_red : red_percentage = 23)
variable (h_green : green_percentage = 15)

theorem students_wearing_other_colors :
  (total_students * (100 - (blue_percentage + red_percentage + green_percentage)) / 100 = 102) :=
by
  sorry

end students_wearing_other_colors_l126_126568


namespace workers_together_time_l126_126256

theorem workers_together_time (hA : ℝ) (hB : ℝ) (jobA_time : hA = 10) (jobB_time : hB = 12) : 
  1 / ((1 / hA) + (1 / hB)) = (60 / 11) :=
by
  -- skipping the proof details
  sorry

end workers_together_time_l126_126256


namespace intersection_M_N_l126_126381

def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | x^2 ≤ 4}

theorem intersection_M_N : M ∩ N = {x | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l126_126381


namespace product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l126_126950

theorem product_of_two_numbers_less_than_the_smaller_of_the_two_factors
    (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (ha1 : a < 1) (hb1 : b < 1) : 
  a * b < min a b := 
sorry

end product_of_two_numbers_less_than_the_smaller_of_the_two_factors_l126_126950


namespace gcd_143_117_l126_126614

theorem gcd_143_117 : Nat.gcd 143 117 = 13 :=
by
  have h1 : 143 = 11 * 13 := by rfl
  have h2 : 117 = 9 * 13 := by rfl
  sorry

end gcd_143_117_l126_126614


namespace ratio_Q_P_l126_126231

theorem ratio_Q_P : 
  ∀ (P Q : ℚ), (∀ x : ℚ, x ≠ -3 → x ≠ 0 → x ≠ 5 → 
    (P / (x + 3) + Q / (x * (x - 5)) = (x^2 - 3*x + 12) / (x^3 + x^2 - 15*x))) →
    (Q / P) = 20 / 9 :=
by
  intros P Q h
  sorry

end ratio_Q_P_l126_126231


namespace hypotenuse_eq_medians_l126_126233

noncomputable def hypotenuse_length_medians (a b : ℝ) (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) : ℝ :=
  3 * Real.sqrt (336 / 13)

-- definition
theorem hypotenuse_eq_medians {a b : ℝ} (h1 : b^2 + (9 * a^2) / 4 = 48) (h2 : a^2 + (9 * b^2) / 4 = 36) :
    Real.sqrt (9 * (a^2 + b^2)) = 3 * Real.sqrt (336 / 13) :=
sorry

end hypotenuse_eq_medians_l126_126233


namespace inequality_semi_perimeter_l126_126378

variables {R r p : Real}

theorem inequality_semi_perimeter (h1 : 0 < R) (h2 : 0 < r) (h3 : 0 < p) :
  16 * R * r - 5 * r^2 ≤ p^2 ∧ p^2 ≤ 4 * R^2 + 4 * R * r + 3 * r^2 :=
sorry

end inequality_semi_perimeter_l126_126378


namespace eventually_stable_l126_126540

theorem eventually_stable {n : ℕ} (a : Fin n → ℤ) :
  ∃ m : ℕ, ∀ t ≥ m, ∀ i j : Fin n, i ≠ j →
  let b := (a t).update i (Int.gcd (a t).nth i (a t).nth j) in
  let c := b.update j (Int.lcm (a t).nth i (a t).nth j) in
  c = a t :=
sorry

end eventually_stable_l126_126540


namespace option_D_correct_l126_126131

theorem option_D_correct (a : ℝ) :
  3 * a ^ 2 - a ≠ 2 * a ∧
  a - (1 - 2 * a) ≠ a - 1 ∧
  -5 * (1 - a ^ 2) ≠ -5 - 5 * a ^ 2 ∧
  a ^ 3 + 7 * a ^ 3 - 5 * a ^ 3 = 3 * a ^ 3 :=
by
  sorry

end option_D_correct_l126_126131


namespace B_starts_cycling_after_A_l126_126431

theorem B_starts_cycling_after_A (t : ℝ) : 10 * t + 20 * (2 - t) = 60 → t = 2 :=
by
  intro h
  sorry

end B_starts_cycling_after_A_l126_126431


namespace solve_eq_solution_l126_126220

def eq_solution (x y : ℕ) : Prop := 3 ^ x = 2 ^ x * y + 1

theorem solve_eq_solution (x y : ℕ) (h1 : x > 0) (h2 : y > 0) : 
  eq_solution x y ↔ (x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 4 ∧ y = 5) :=
sorry

end solve_eq_solution_l126_126220


namespace greatest_b_l126_126461

theorem greatest_b (b : ℝ) : (-b^2 + 9 * b - 14 ≥ 0) → b ≤ 7 := sorry

end greatest_b_l126_126461


namespace s_6_of_30_eq_146_over_175_l126_126205

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem s_6_of_30_eq_146_over_175 : s (s (s (s (s (s 30))))) = 146 / 175 := sorry

end s_6_of_30_eq_146_over_175_l126_126205


namespace eleven_pow_2023_mod_eight_l126_126546

theorem eleven_pow_2023_mod_eight (h11 : 11 % 8 = 3) (h3 : 3^2 % 8 = 1) : 11^2023 % 8 = 3 :=
by
  sorry

end eleven_pow_2023_mod_eight_l126_126546


namespace total_volume_of_four_cubes_is_500_l126_126556

-- Definitions for the problem assumptions
def edge_length := 5
def volume_of_cube (edge_length : ℕ) := edge_length ^ 3
def number_of_boxes := 4

-- Main statement to prove
theorem total_volume_of_four_cubes_is_500 :
  (volume_of_cube edge_length) * number_of_boxes = 500 :=
by
  -- Proof steps will go here
  sorry

end total_volume_of_four_cubes_is_500_l126_126556


namespace museum_earnings_from_nyc_college_students_l126_126066

def visitors := 200
def nyc_residents_fraction := 1 / 2
def college_students_fraction := 0.30
def ticket_price := 4

theorem museum_earnings_from_nyc_college_students : 
  ((visitors * nyc_residents_fraction * college_students_fraction) * ticket_price) = 120 := 
by 
  sorry

end museum_earnings_from_nyc_college_students_l126_126066


namespace express_repeating_decimal_as_fraction_l126_126605

noncomputable def repeating_decimal_to_fraction : ℚ :=
  3 + 7 / 9  -- Representation of 3.\overline{7} as a Rational number representation

theorem express_repeating_decimal_as_fraction :
  (3 + 7 / 9 : ℚ) = 34 / 9 :=
by
  -- Placeholder for proof steps
  sorry

end express_repeating_decimal_as_fraction_l126_126605


namespace sin_double_angle_l126_126474

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 4) :
  Real.sin (2 * α) = -3 / 4 :=
sorry

end sin_double_angle_l126_126474


namespace statement_B_statement_C_l126_126559

variable (a b c : ℝ)

-- Condition: a > b
def condition1 := a > b

-- Condition: a / c^2 > b / c^2
def condition2 := a / c^2 > b / c^2

-- Statement B: If a > b, then a - 1 > b - 2
theorem statement_B (ha_gt_b : condition1 a b) : a - 1 > b - 2 :=
by sorry

-- Statement C: If a / c^2 > b / c^2, then a > b
theorem statement_C (ha_div_csqr_gt_hb_div_csqr : condition2 a b c) : a > b :=
by sorry

end statement_B_statement_C_l126_126559


namespace value_of_k_l126_126642

open Nat

def perm (n r : ℕ) : ℕ := factorial n / factorial (n - r)
def comb (n r : ℕ) : ℕ := factorial n / (factorial r * factorial (n - r))

theorem value_of_k : ∃ k : ℕ, perm 32 6 = k * comb 32 6 ∧ k = 720 := by
  use 720
  unfold perm comb
  sorry

end value_of_k_l126_126642


namespace find_constants_u_v_l126_126484

theorem find_constants_u_v
  (n p r1 r2 : ℝ)
  (h1 : r1 + r2 = n)
  (h2 : r1 * r2 = p) :
  ∃ u v, (r1^4 + r2^4 = -u) ∧ (r1^4 * r2^4 = v) ∧ u = -(n^4 - 4*p*n^2 + 2*p^2) ∧ v = p^4 :=
by
  sorry

end find_constants_u_v_l126_126484


namespace solve_fraction_equation_l126_126383

theorem solve_fraction_equation (x : ℝ) (h : x ≠ 1) : (3 * x - 1) / (4 * x - 4) = 2 / 3 → x = -5 :=
by
  intro h_eq
  sorry

end solve_fraction_equation_l126_126383


namespace find_d_l126_126164

theorem find_d (d : ℚ) (h_floor : ∃ x : ℤ, x^2 + 5 * x - 36 = 0 ∧ x = ⌊d⌋)
  (h_frac: ∃ y : ℚ, 3 * y^2 - 11 * y + 2 = 0 ∧ y = d - ⌊d⌋):
  d = 13 / 3 :=
by
  sorry

end find_d_l126_126164


namespace pairs_of_socks_now_l126_126170

def initial_socks : Nat := 28
def socks_thrown_away : Nat := 4
def socks_bought : Nat := 36

theorem pairs_of_socks_now : (initial_socks - socks_thrown_away + socks_bought) / 2 = 30 := by
  sorry

end pairs_of_socks_now_l126_126170


namespace remainder_1234567_div_256_l126_126437

/--
  Given the numbers 1234567 and 256, prove that the remainder when
  1234567 is divided by 256 is 57.
-/
theorem remainder_1234567_div_256 :
  1234567 % 256 = 57 :=
by
  sorry

end remainder_1234567_div_256_l126_126437


namespace arithmetic_sequence_a5_l126_126176

theorem arithmetic_sequence_a5 {a : ℕ → ℝ} (h₁ : a 2 + a 8 = 16) : a 5 = 8 :=
sorry

end arithmetic_sequence_a5_l126_126176


namespace distinct_prime_factors_of_90_l126_126781

theorem distinct_prime_factors_of_90 : 
  ∃ (s : Finset ℕ), s = {2, 3, 5} ∧ ∀ p ∈ s, Nat.Prime p ∧ 2 * 3 * 3 * 5 = 90 :=
by
  sorry

end distinct_prime_factors_of_90_l126_126781


namespace prime_between_30_and_40_with_remainder_1_l126_126845

theorem prime_between_30_and_40_with_remainder_1 (n : ℕ) : 
  n.Prime → 
  30 < n → n < 40 → 
  n % 6 = 1 → 
  n = 37 := 
sorry

end prime_between_30_and_40_with_remainder_1_l126_126845


namespace sufficient_but_not_necessary_l126_126336

def quadratic_real_roots (a : ℝ) : Prop :=
  (∃ x : ℝ, x^2 - 2 * x + a = 0)

theorem sufficient_but_not_necessary (a : ℝ) :
  (quadratic_real_roots 1) ∧ (∀ a > 1, ¬ quadratic_real_roots a) :=
sorry

end sufficient_but_not_necessary_l126_126336


namespace alex_minimum_additional_coins_l126_126274

theorem alex_minimum_additional_coins (friends coins : ℕ) (h_friends : friends = 15) (h_coins : coins = 105) : 
  ∃ add_coins, add_coins = (∑ i in range (friends + 1), i) - coins :=
by
  sorry

end alex_minimum_additional_coins_l126_126274


namespace gauss_company_percent_five_years_or_more_l126_126115

def num_employees_less_1_year (x : ℕ) : ℕ := 5 * x
def num_employees_1_to_2_years (x : ℕ) : ℕ := 5 * x
def num_employees_2_to_3_years (x : ℕ) : ℕ := 8 * x
def num_employees_3_to_4_years (x : ℕ) : ℕ := 3 * x
def num_employees_4_to_5_years (x : ℕ) : ℕ := 2 * x
def num_employees_5_to_6_years (x : ℕ) : ℕ := 2 * x
def num_employees_6_to_7_years (x : ℕ) : ℕ := 2 * x
def num_employees_7_to_8_years (x : ℕ) : ℕ := x
def num_employees_8_to_9_years (x : ℕ) : ℕ := x
def num_employees_9_to_10_years (x : ℕ) : ℕ := x

def total_employees (x : ℕ) : ℕ :=
  num_employees_less_1_year x +
  num_employees_1_to_2_years x +
  num_employees_2_to_3_years x +
  num_employees_3_to_4_years x +
  num_employees_4_to_5_years x +
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

def employees_with_5_years_or_more (x : ℕ) : ℕ :=
  num_employees_5_to_6_years x +
  num_employees_6_to_7_years x +
  num_employees_7_to_8_years x +
  num_employees_8_to_9_years x +
  num_employees_9_to_10_years x

theorem gauss_company_percent_five_years_or_more (x : ℕ) :
  (employees_with_5_years_or_more x : ℝ) / (total_employees x : ℝ) * 100 = 30 :=
by
  sorry

end gauss_company_percent_five_years_or_more_l126_126115


namespace green_paint_quarts_l126_126620

theorem green_paint_quarts (x : ℕ) (h : 5 * x = 3 * 15) : x = 9 := 
sorry

end green_paint_quarts_l126_126620


namespace total_children_l126_126590

theorem total_children {x y : ℕ} (h₁ : x = 18) (h₂ : y = 12) 
  (h₃ : x + y = 30) (h₄ : x = 18) (h₅ : y = 12) : 2 * x + 3 * y = 72 := 
by
  sorry

end total_children_l126_126590


namespace equation_of_line_AC_l126_126767

-- Definitions of points and lines
structure Point :=
  (x : ℝ)
  (y : ℝ)

def line_equation (A B C : ℝ) (P : Point) : Prop :=
  A * P.x + B * P.y + C = 0

-- Given points and lines
def B : Point := ⟨-2, 0⟩
def altitude_on_AB (P : Point) : Prop := line_equation 1 3 (-26) P

-- Required equation of line AB
def line_AB (P : Point) : Prop := line_equation 3 (-1) 6 P

-- Angle bisector given in the condition
def angle_bisector (P : Point) : Prop := line_equation 1 1 (-2) P

-- Derived Point A
def A : Point := ⟨-1, 3⟩

-- Symmetric point B' with respect to the angle bisector
def B' : Point := ⟨2, 4⟩

-- Required equation of line AC
def line_AC (P : Point) : Prop := line_equation 1 (-3) 10 P

-- The proof statement
theorem equation_of_line_AC :
  ∀ P : Point, (line_AB B ∧ angle_bisector A ∧ P = A → P = B' → line_AC P) :=
by
  intros P h h1 h2
  sorry

end equation_of_line_AC_l126_126767


namespace quadratic_ineq_solutions_l126_126609

theorem quadratic_ineq_solutions (c : ℝ) (h : c > 0) : c < 16 ↔ ∀ x : ℝ, x^2 - 8 * x + c < 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x = x1 ∨ x = x2 :=
by
  sorry

end quadratic_ineq_solutions_l126_126609


namespace kanul_raw_material_expense_l126_126820

theorem kanul_raw_material_expense
  (total_amount : ℝ)
  (machinery_cost : ℝ)
  (raw_materials_cost : ℝ)
  (cash_fraction : ℝ)
  (h_total_amount : total_amount = 137500)
  (h_machinery_cost : machinery_cost = 30000)
  (h_cash_fraction: cash_fraction = 0.20)
  (h_eq : total_amount = raw_materials_cost + machinery_cost + cash_fraction * total_amount) :
  raw_materials_cost = 80000 :=
by
  rw [h_total_amount, h_machinery_cost, h_cash_fraction] at h_eq
  sorry

end kanul_raw_material_expense_l126_126820


namespace circle_properties_l126_126080

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
  let eq1 := x^2 - 4 * y - 18
  let eq2 := -y^2 + 6 * x + 26
  let lhs := x^2 - 6 * x + y^2 - 4 * y
  let rhs := 44
  let center_x := 3
  let center_y := 2
  let radius := Real.sqrt 57
  let target := 5 + radius
  (center_x, center_y, target)

theorem circle_properties
  (x y : ℝ) :
  let (a, b, r) := circle_center_and_radius x y 
  a + b + r = 5 + Real.sqrt 57 :=
by
  sorry

end circle_properties_l126_126080


namespace alan_total_cost_is_84_l126_126271

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end alan_total_cost_is_84_l126_126271


namespace reflex_angle_at_G_correct_l126_126326

noncomputable def reflex_angle_at_G
    (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80)
    : ℝ :=
  360 - (180 - (180 - angle_BAG) - (180 - angle_GEL))

theorem reflex_angle_at_G_correct :
    (∀ (B A E L G : Type)
    (on_line : B = A ∨ A = E ∨ E = L) 
    (off_line : ¬(G = B ∨ G = A ∨ G = E ∨ G = L))
    (angle_BAG : ℝ)
    (angle_GEL : ℝ)
    (h1 : angle_BAG = 120)
    (h2 : angle_GEL = 80),
    reflex_angle_at_G B A E L G on_line off_line angle_BAG angle_GEL h1 h2 = 340) := sorry

end reflex_angle_at_G_correct_l126_126326


namespace expr_eval_l126_126542

theorem expr_eval : 180 / 6 * 2 + 5 = 65 := by
  sorry

end expr_eval_l126_126542


namespace temperature_increase_l126_126069

variable (T_morning T_afternoon : ℝ)

theorem temperature_increase : 
  (T_morning = -3) → (T_afternoon = 5) → (T_afternoon - T_morning = 8) :=
by
intros h1 h2
rw [h1, h2]
sorry

end temperature_increase_l126_126069


namespace difference_between_possible_values_of_x_l126_126053

noncomputable def difference_of_roots (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ℝ :=
  let sol1 := 11  -- First root
  let sol2 := -11 -- Second root
  sol1 - sol2

theorem difference_between_possible_values_of_x (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) :
  difference_of_roots x h = 22 :=
sorry

end difference_between_possible_values_of_x_l126_126053


namespace value_of_5_T_3_l126_126597

def operation (a b : ℕ) : ℕ := 4 * a + 6 * b

theorem value_of_5_T_3 : operation 5 3 = 38 :=
by
  -- proof (which is not required)
  sorry

end value_of_5_T_3_l126_126597


namespace max_distance_is_15_l126_126289

noncomputable def max_distance_between_cars (v_A v_B: ℝ) (a: ℝ) (D: ℝ) : ℝ :=
  if v_A > v_B ∧ D = a + 60 then (a * (1 - a / 60)) else 0

theorem max_distance_is_15 (v_A v_B: ℝ) (a: ℝ) (D: ℝ) :
  v_A > v_B ∧ D = a + 60 → max_distance_between_cars v_A v_B a D = 15 :=
by
  sorry

end max_distance_is_15_l126_126289


namespace weight_loss_in_april_l126_126668

-- Definitions based on given conditions
def total_weight_to_lose : ℕ := 10
def march_weight_loss : ℕ := 3
def may_weight_loss : ℕ := 3

-- Theorem statement
theorem weight_loss_in_april :
  total_weight_to_lose = march_weight_loss + 4 + may_weight_loss := 
sorry

end weight_loss_in_april_l126_126668


namespace three_five_seven_sum_fraction_l126_126588

theorem three_five_seven_sum_fraction :
  (3 * 5 * 7) * ((1 / 3) + (1 / 5) + (1 / 7)) = 71 :=
by
  sorry

end three_five_seven_sum_fraction_l126_126588


namespace yellow_surface_area_fraction_minimal_l126_126572

theorem yellow_surface_area_fraction_minimal 
  (total_cubes : ℕ)
  (edge_length : ℕ)
  (yellow_cubes : ℕ)
  (blue_cubes : ℕ)
  (total_surface_area : ℕ)
  (yellow_surface_area : ℕ)
  (yellow_fraction : ℚ) :
  total_cubes = 64 ∧
  edge_length = 4 ∧
  yellow_cubes = 16 ∧
  blue_cubes = 48 ∧
  total_surface_area = 6 * edge_length * edge_length ∧
  yellow_surface_area = 15 →
  yellow_fraction = (yellow_surface_area : ℚ) / total_surface_area :=
sorry

end yellow_surface_area_fraction_minimal_l126_126572


namespace remainder_of_division_l126_126442

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l126_126442


namespace sin_180_eq_0_l126_126300

theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 :=
by
  sorry

end sin_180_eq_0_l126_126300


namespace no_integer_solution_for_system_l126_126006

theorem no_integer_solution_for_system :
  (¬ ∃ x y : ℤ, 18 * x + 27 * y = 21 ∧ 27 * x + 18 * y = 69) :=
by
  sorry

end no_integer_solution_for_system_l126_126006


namespace casper_candy_problem_l126_126951

theorem casper_candy_problem (o y gr : ℕ) (n : ℕ) (h1 : 10 * o = 16 * y) (h2 : 16 * y = 18 * gr) (h3 : 18 * gr = 18 * n) :
    n = 40 :=
by
  sorry

end casper_candy_problem_l126_126951


namespace monotonicity_and_extreme_values_l126_126667

noncomputable def f (x : ℝ) : ℝ := Real.log x - x

theorem monotonicity_and_extreme_values :
  (∀ x : ℝ, 0 < x ∧ x < 1 → f x < f (1 - x)) ∧
  (∀ x : ℝ, x > 1 → f x < f 1) ∧
  f 1 = -1 :=
by 
  sorry

end monotonicity_and_extreme_values_l126_126667


namespace mini_toy_height_difference_l126_126239

variables (H_standard H_toy H_mini_diff : ℝ)

def poodle_heights : Prop :=
  H_standard = 28 ∧ H_toy = 14 ∧ H_standard - 8 = H_mini_diff + H_toy

theorem mini_toy_height_difference (H_standard H_toy H_mini_diff: ℝ) (h: poodle_heights H_standard H_toy H_mini_diff) :
  H_mini_diff = 6 :=
by {
  sorry
}

end mini_toy_height_difference_l126_126239


namespace bottles_from_B_l126_126599

-- Definitions for the bottles from each shop and the total number of bottles Don can buy
def bottles_from_A : Nat := 150
def bottles_from_C : Nat := 220
def total_bottles : Nat := 550

-- Lean statement to prove that the number of bottles Don buys from Shop B is 180
theorem bottles_from_B :
  total_bottles - (bottles_from_A + bottles_from_C) = 180 := 
by
  sorry

end bottles_from_B_l126_126599


namespace gcd_lcm_sum_8_12_l126_126899

open Int

theorem gcd_lcm_sum_8_12 : 
  gcd 8 12 + lcm 8 12 = 28 :=
by
  -- Lean proof goes here, but it's omitted according to the instructions
  sorry

end gcd_lcm_sum_8_12_l126_126899


namespace sum_gcf_lcm_eq_28_l126_126881

-- Define the numbers
def a : ℕ := 8
def b : ℕ := 12

-- Define the greatest common divisor
def gcf : ℕ := Nat.gcd a b

-- Define the least common multiple
def lcm : ℕ := Nat.lcm a b

-- The theorem statement we need to prove
theorem sum_gcf_lcm_eq_28 : gcf + lcm = 28 :=
by
  -- Sorry is used to skip the proof
  sorry

end sum_gcf_lcm_eq_28_l126_126881


namespace maximum_possible_angle_Z_l126_126649

theorem maximum_possible_angle_Z (X Y Z : ℝ) (h1 : Z ≤ Y) (h2 : Y ≤ X) (h3 : 2 * X = 6 * Z) (h4 : X + Y + Z = 180) : Z = 36 :=
by
  sorry

end maximum_possible_angle_Z_l126_126649


namespace ferris_wheel_time_l126_126133

theorem ferris_wheel_time (R T : ℝ) (t : ℝ) (h : ℝ → ℝ) :
  R = 30 → T = 90 → (∀ t, h t = R * Real.cos ((2 * Real.pi / T) * t) + R) → h t = 45 → t = 15 :=
by
  intros hR hT hFunc hHt
  sorry

end ferris_wheel_time_l126_126133


namespace additional_coins_needed_l126_126275

def num_friends : Nat := 15
def current_coins : Nat := 105

def total_coins_needed (n : Nat) : Nat :=
  n * (n + 1) / 2
  
theorem additional_coins_needed :
  let coins_needed := total_coins_needed num_friends
  let additional_coins := coins_needed - current_coins
  additional_coins = 15 :=
by
  sorry

end additional_coins_needed_l126_126275


namespace mans_rate_in_still_water_l126_126578

theorem mans_rate_in_still_water (R S : ℝ) (h1 : R + S = 18) (h2 : R - S = 4) : R = 11 :=
by {
  sorry
}

end mans_rate_in_still_water_l126_126578


namespace inequality_holds_l126_126968

theorem inequality_holds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_prod : a * b * c = 1) :
  (a + (1 / b))^2 + (b + (1 / c))^2 + (c + (1 / a))^2 ≥ 3 * (a + b + c + 1) := by
  sorry

end inequality_holds_l126_126968


namespace gcd_lcm_sum_8_12_l126_126894

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l126_126894


namespace students_taking_art_l126_126712

def total_students := 500
def students_taking_music := 40
def students_taking_both := 10
def students_taking_neither := 450

theorem students_taking_art : ∃ A, total_students = students_taking_music - students_taking_both + (A - students_taking_both) + students_taking_both + students_taking_neither ∧ A = 20 :=
by
  sorry

end students_taking_art_l126_126712


namespace compute_expression_l126_126292

theorem compute_expression (x : ℝ) (h : x = 8) : 
  (x^6 - 64 * x^3 + 1024) / (x^3 - 16) = 480 :=
by
  rw [h]
  sorry

end compute_expression_l126_126292


namespace running_current_each_unit_l126_126527

theorem running_current_each_unit (I : ℝ) (h1 : ∀i, i = 2 * I) (h2 : ∀i, i * 3 = 6 * I) (h3 : 6 * I = 240) : I = 40 :=
by
  sorry

end running_current_each_unit_l126_126527


namespace optimal_years_minimize_cost_l126_126104

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost_l126_126104


namespace initial_violet_marbles_eq_l126_126593

variable {initial_violet_marbles : Nat}
variable (red_marbles : Nat := 14)
variable (total_marbles : Nat := 78)

theorem initial_violet_marbles_eq :
  initial_violet_marbles = total_marbles - red_marbles := by
  sorry

end initial_violet_marbles_eq_l126_126593


namespace max_distance_l126_126034

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def curve_C (p : ℝ × ℝ) : Prop := 
  let x := p.1 
  let y := p.2 
  x^2 + y^2 - 2*y = 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-3/5 * t + 2, 4/5 * t)

def x_axis_intersection (l : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := l 0 
  (x, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance {M : ℝ × ℝ} {N : ℝ × ℝ}
  (curve_c : (ℝ × ℝ) → Prop)
  (line_l : ℝ → ℝ × ℝ)
  (h1 : curve_c = curve_C)
  (h2 : line_l = line_l)
  (M_def : x_axis_intersection line_l = M)
  (hNP : curve_c N) :
  distance M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_l126_126034


namespace diamonds_balance_emerald_l126_126435

theorem diamonds_balance_emerald (D E : ℝ) (h1 : 9 * D = 4 * E) (h2 : 9 * D + E = 4 * E) : 3 * D = E := by
  sorry

end diamonds_balance_emerald_l126_126435


namespace total_volume_of_four_boxes_l126_126548

-- Define the edge length of the cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_one_cube := edge_length ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 4

-- The total volume of the four cubes
def total_volume := number_of_cubes * volume_of_one_cube

-- Statement to prove that the total volume equals 500 cubic feet
theorem total_volume_of_four_boxes :
  total_volume = 500 :=
sorry

end total_volume_of_four_boxes_l126_126548


namespace horner_eval_at_minus_point_two_l126_126124

def f (x : ℝ) : ℝ :=
  1 + x + 0.5 * x^2 + 0.16667 * x^3 + 0.04167 * x^4 + 0.00833 * x^5

theorem horner_eval_at_minus_point_two :
  f (-0.2) = 0.81873 :=
by 
  sorry

end horner_eval_at_minus_point_two_l126_126124


namespace perpendicular_MP_MQ_l126_126760

variable (k m : ℝ)

def ellipse (x y : ℝ) := (x^2 / 4) + (y^2 / 3) = 1

def line (x y : ℝ) := y = k*x + m

def fixed_point_exists (k m : ℝ) : Prop :=
  let P := (-(4 * k) / m, 3 / m)
  let Q := (4, 4 * k + m)
  ∃ (M : ℝ), (M = 1 ∧ ((P.1 - M) * (Q.1 - M) + P.2 * Q.2 = 0))

theorem perpendicular_MP_MQ : fixed_point_exists k m := sorry

end perpendicular_MP_MQ_l126_126760


namespace partition_555_weights_l126_126915

theorem partition_555_weights :
  ∃ A B C : Finset ℕ, 
  (∀ x ∈ A, x ∈ Finset.range (555 + 1)) ∧ 
  (∀ y ∈ B, y ∈ Finset.range (555 + 1)) ∧ 
  (∀ z ∈ C, z ∈ Finset.range (555 + 1)) ∧ 
  A ∩ B = ∅ ∧ A ∩ C = ∅ ∧ B ∩ C = ∅ ∧ 
  A ∪ B ∪ C = Finset.range (555 + 1) ∧ 
  A.sum id = 51430 ∧ B.sum id = 51430 ∧ C.sum id = 51430 := sorry

end partition_555_weights_l126_126915


namespace common_number_in_sequence_l126_126059

theorem common_number_in_sequence 
  (a b c d e f g h i j : ℕ) 
  (h1 : (a + b + c + d + e) / 5 = 4) 
  (h2 : (f + g + h + i + j) / 5 = 9)
  (h3 : (a + b + c + d + e + f + g + h + i + j) / 10 = 7)
  (h4 : e = f) :
  e = 5 :=
by
  sorry

end common_number_in_sequence_l126_126059


namespace complete_square_eq_l126_126236

theorem complete_square_eq (x : ℝ) : x^2 - 4 * x - 1 = 0 → (x - 2)^2 = 5 :=
by
  sorry

end complete_square_eq_l126_126236


namespace tree_volume_estimation_proof_l126_126724

noncomputable def average_root_cross_sectional_area (x_i : list ℝ) := (x_i.sum) / (x_i.length)
noncomputable def average_volume (y_i : list ℝ) := (y_i.sum) / (y_i.length)
noncomputable def correlation_coefficient (x_i y_i : list ℝ) : ℝ :=
  let n := x_i.length in
  let x_bar := average_root_cross_sectional_area x_i in
  let y_bar := average_volume y_i in
  let numerator := (list.zip x_i y_i).sum (λ ⟨x, y⟩, (x - x_bar) * (y - y_bar)) in
  let denominator_x := (x_i.sum (λ x, (x - x_bar)^2)) in
  let denominator_y := (y_i.sum (λ y, (y - y_bar)^2)) in
  numerator / ((denominator_x * denominator_y).sqrt)

noncomputable def total_volume_estimate (total_area avg_y avg_x : ℝ) := (avg_y / avg_x) * total_area

theorem tree_volume_estimation_proof :
  let x_i := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06] in
  let y_i := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40] in
  let total_area := 186 in
  average_root_cross_sectional_area x_i = 0.06 ∧
  average_volume y_i = 0.39 ∧
  correlation_coefficient x_i y_i ≈ 0.97 ∧
  total_volume_estimate total_area 0.39 0.06 = 1209 :=
by
  sorry

end tree_volume_estimation_proof_l126_126724


namespace domain_proof_l126_126954

def domain_of_function : Set ℝ := {x : ℝ | x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x}

theorem domain_proof :
  (∀ x : ℝ, (x ≠ 7) → (x^2 - 16 ≥ 0) → (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x)) ∧
  (∀ x : ℝ, (x ≤ -4 ∨ (4 ≤ x ∧ x < 7) ∨ 7 < x) → (x ≠ 7) ∧ (x^2 - 16 ≥ 0)) :=
by
  sorry

end domain_proof_l126_126954


namespace base_h_addition_eq_l126_126314

theorem base_h_addition_eq (h : ℕ) :
  let n1 := 7 * h^3 + 3 * h^2 + 6 * h + 4
  let n2 := 8 * h^3 + 4 * h^2 + 2 * h + 1
  let sum := 1 * h^4 + 7 * h^3 + 2 * h^2 + 8 * h + 5
  n1 + n2 = sum → h = 8 :=
by
  intros n1 n2 sum h_eq
  sorry

end base_h_addition_eq_l126_126314


namespace gcd_lcm_sum_8_12_l126_126897

-- Define the problem statement
theorem gcd_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end gcd_lcm_sum_8_12_l126_126897


namespace avg_korean_language_score_l126_126522

theorem avg_korean_language_score (male_avg : ℝ) (female_avg : ℝ) (male_students : ℕ) (female_students : ℕ) 
    (male_avg_given : male_avg = 83.1) (female_avg_given : female_avg = 84) (male_students_given : male_students = 10) (female_students_given : female_students = 8) :
    (male_avg * male_students + female_avg * female_students) / (male_students + female_students) = 83.5 :=
by sorry

end avg_korean_language_score_l126_126522


namespace compare_squares_l126_126023

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2 * a := by
  -- the proof would go here, but we skip it according to the instruction
  sorry

end compare_squares_l126_126023


namespace at_least_one_not_less_than_100_l126_126081

-- Defining the original propositions
def p : Prop := ∀ (A_score : ℕ), A_score ≥ 100
def q : Prop := ∀ (B_score : ℕ), B_score < 100

-- Assertion to be proved in Lean
theorem at_least_one_not_less_than_100 (h1 : p) (h2 : q) : p ∨ ¬q := 
sorry

end at_least_one_not_less_than_100_l126_126081


namespace all_d_zero_l126_126375

def d (n m : ℕ) : ℤ := sorry -- or some explicit initial definition

theorem all_d_zero (n m : ℕ) (h₁ : n ≥ 0) (h₂ : 0 ≤ m) (h₃ : m ≤ n) :
  (m = 0 ∨ m = n → d n m = 0) ∧
  (0 < m ∧ m < n → m * d n m = m * d (n - 1) m + (2 * n - m) * d (n - 1) (m - 1))
:=
  sorry

end all_d_zero_l126_126375


namespace sum_of_number_and_square_eq_132_l126_126411

theorem sum_of_number_and_square_eq_132 (x : ℝ) (h : x + x^2 = 132) : x = 11 ∨ x = -12 :=
by
  sorry

end sum_of_number_and_square_eq_132_l126_126411


namespace annual_percentage_increase_20_l126_126235

variable (P0 P1 : ℕ) (r : ℚ)

-- Population initial condition
def initial_population : Prop := P0 = 10000

-- Population after 1 year condition
def population_after_one_year : Prop := P1 = 12000

-- Define the annual percentage increase formula
def percentage_increase (P0 P1 : ℕ) : ℚ := ((P1 - P0 : ℚ) / P0) * 100

-- State the theorem
theorem annual_percentage_increase_20
  (h1 : initial_population P0)
  (h2 : population_after_one_year P1) :
  percentage_increase P0 P1 = 20 := by
  sorry

end annual_percentage_increase_20_l126_126235


namespace parallel_statements_l126_126278

-- Definitions for lines and planes
variables {Line Plane : Type}

-- Parallelism between a line and another line or a plane
variables (a b : Line) (α : Plane)

-- Parallel relationship assertions
axiom parallel_lines (l1 l2 : Line) : Prop -- l1 is parallel to l2
axiom line_in_plane (l : Line) (p : Plane) : Prop -- line l is in plane p
axiom parallel_line_plane (l : Line) (p : Plane) : Prop -- line l is parallel to plane p

-- Problem statement
theorem parallel_statements :
  (parallel_lines a b ∧ line_in_plane b α → parallel_line_plane a α) ∧
  (parallel_lines a b ∧ parallel_line_plane a α → parallel_line_plane b α) :=
sorry

end parallel_statements_l126_126278


namespace sin_squared_alpha_plus_pi_over_4_l126_126172

theorem sin_squared_alpha_plus_pi_over_4 (α : ℝ) (h : Real.sin (2 * α) = 2 / 3) : 
  Real.sin (α + Real.pi / 4) ^ 2 = 5 / 6 := 
sorry

end sin_squared_alpha_plus_pi_over_4_l126_126172


namespace ratio_of_height_to_radius_l126_126720

theorem ratio_of_height_to_radius (r h : ℝ)
  (h_cone : r > 0 ∧ h > 0)
  (circumference_cone_base : 20 * 2 * Real.pi * r = 2 * Real.pi * Real.sqrt (r^2 + h^2))
  : h / r = Real.sqrt 399 := by
  sorry

end ratio_of_height_to_radius_l126_126720


namespace sum_of_cubes_consecutive_integers_divisible_by_9_l126_126831

theorem sum_of_cubes_consecutive_integers_divisible_by_9 (n : ℤ) : 
  (n^3 + (n+1)^3 + (n+2)^3) % 9 = 0 :=
sorry

end sum_of_cubes_consecutive_integers_divisible_by_9_l126_126831


namespace smallest_a_l126_126384

theorem smallest_a (a b c : ℚ)
  (h1 : a > 0)
  (h2 : b = -2 * a / 3)
  (h3 : c = a / 9 - 5 / 9)
  (h4 : (a + b + c).den = 1) : a = 5 / 4 :=
by
  sorry

end smallest_a_l126_126384


namespace number_of_distinct_prime_factors_of_90_l126_126786

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l126_126786


namespace yacht_actual_cost_l126_126583

theorem yacht_actual_cost
  (discount_percentage : ℝ)
  (amount_paid : ℝ)
  (original_cost : ℝ)
  (h1 : discount_percentage = 0.72)
  (h2 : amount_paid = 3200000)
  (h3 : amount_paid = (1 - discount_percentage) * original_cost) :
  original_cost = 11428571.43 :=
by
  sorry

end yacht_actual_cost_l126_126583


namespace determine_c_l126_126285

theorem determine_c (c d : ℝ) (hc : c < 0) (hd : d > 0) (hamp : ∀ x, y = c * Real.cos (d * x) → |y| ≤ 3) :
  c = -3 :=
sorry

end determine_c_l126_126285


namespace domain_of_f_l126_126682

noncomputable def f (x : ℝ) := (Real.sqrt (x + 3)) / x

theorem domain_of_f :
  { x : ℝ | x ≥ -3 ∧ x ≠ 0 } = { x : ℝ | ∃ y, f y ≠ 0 } :=
by
  sorry

end domain_of_f_l126_126682


namespace coin_selection_probability_l126_126189

noncomputable def probability_at_least_50_cents : ℚ := 
  let total_ways := Nat.choose 12 6 -- total ways to choose 6 coins out of 12
  let case1 := 1 -- 6 dimes
  let case2 := (Nat.choose 6 5) * (Nat.choose 4 1) -- 5 dimes and 1 nickel
  let case3 := (Nat.choose 6 4) * (Nat.choose 4 2) -- 4 dimes and 2 nickels
  let successful_ways := case1 + case2 + case3 -- total successful outcomes
  successful_ways / total_ways

theorem coin_selection_probability : 
  probability_at_least_50_cents = 127 / 924 := by 
  sorry

end coin_selection_probability_l126_126189


namespace min_value_of_f_on_interval_l126_126107

noncomputable def f (x : ℝ) : ℝ := (1 / x) - 2 * x

theorem min_value_of_f_on_interval :
  ∃ m : ℝ, is_glb (set.range (λ x, f x)) m ∧ m = -7 / 2 :=
by
  let I := set.Icc (1 : ℝ) (2 : ℝ)
  refine ⟨-7 / 2, _, rfl⟩
  sorry

end min_value_of_f_on_interval_l126_126107


namespace correct_expression_l126_126433

variables {a b c : ℝ}

theorem correct_expression :
  -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b :=
by
  sorry

end correct_expression_l126_126433


namespace total_red_yellow_black_l126_126757

/-- Calculate the total number of red, yellow, and black shirts Gavin has,
given that he has 420 shirts in total, 85 of them are blue, and 157 are
green. -/
theorem total_red_yellow_black (total_shirts : ℕ) (blue_shirts : ℕ) (green_shirts : ℕ) :
  total_shirts = 420 → blue_shirts = 85 → green_shirts = 157 → 
  (total_shirts - (blue_shirts + green_shirts) = 178) :=
by
  intros h1 h2 h3
  sorry

end total_red_yellow_black_l126_126757


namespace alan_total_cost_l126_126273

theorem alan_total_cost :
  let price_AVN_CD := 12 in
  let price_The_Dark_CD := price_AVN_CD * 2 in
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD in
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs in
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs in
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs in
  total_cost = 84 :=
by
  let price_AVN_CD := 12
  let price_The_Dark_CD := price_AVN_CD * 2
  let total_cost_The_Dark_CDs := 2 * price_The_Dark_CD
  let total_cost_before_90s_CDs := price_AVN_CD + total_cost_The_Dark_CDs
  let cost_90s_CDs := 0.4 * total_cost_before_90s_CDs
  let total_cost := total_cost_before_90s_CDs + cost_90s_CDs
  show total_cost = 84, from sorry

end alan_total_cost_l126_126273


namespace Darnell_saves_on_alternative_plan_l126_126595

theorem Darnell_saves_on_alternative_plan :
  ∀ (current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins : ℕ),
    current_cost = 12 →
    text_cost_per_30 = 1 →
    call_cost_per_20 = 3 →
    texts = 60 →
    mins = 60 →
    alternative_cost = (texts / 30) * text_cost_per_30 + (mins / 20) * call_cost_per_20 →
    current_cost - alternative_cost = 1 :=
by
  intros current_cost alternative_cost text_cost_per_30 call_cost_per_20 texts mins
    h_current_cost h_text_cost_per_30 h_call_cost_per_20 h_texts h_mins h_alternative_cost
  rw [h_current_cost, h_text_cost_per_30, h_call_cost_per_20, h_texts, h_mins, h_alternative_cost]
  have h1 : 60 / 30 = 2 := by admit
  have h2 : 60 / 20 = 3 := by admit
  rw [h1, h2]
  simp
  sorry

end Darnell_saves_on_alternative_plan_l126_126595


namespace no_maximum_value_l126_126249

-- Define the conditions and the expression in Lean
def expression (a b c d : ℝ) : ℝ := a^2 + b^2 + c^2 + d^2 + a*b + c*d

def condition (a b c d : ℝ) : Prop := a * d - b * c = 1

theorem no_maximum_value : ¬ ∃ M, ∀ a b c d, condition a b c d → expression a b c d ≤ M := by
  sorry

end no_maximum_value_l126_126249


namespace smallest_YZ_minus_XZ_l126_126864

theorem smallest_YZ_minus_XZ 
  (XZ YZ XY : ℕ)
  (h_sum : XZ + YZ + XY = 3001)
  (h_order : XZ < YZ ∧ YZ ≤ XY)
  (h_triangle_ineq1 : XZ + YZ > XY)
  (h_triangle_ineq2 : XZ + XY > YZ)
  (h_triangle_ineq3 : YZ + XY > XZ) :
  ∃ XZ YZ XY : ℕ, YZ - XZ = 1 := sorry

end smallest_YZ_minus_XZ_l126_126864


namespace area_of_right_triangle_integers_l126_126088

theorem area_of_right_triangle_integers (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (A : ℤ), A = (a * b) / 2 := 
sorry

end area_of_right_triangle_integers_l126_126088


namespace width_of_room_l126_126232

noncomputable def roomWidth (length : ℝ) (totalCost : ℝ) (costPerSquareMeter : ℝ) : ℝ :=
  let area := totalCost / costPerSquareMeter
  area / length

theorem width_of_room :
  roomWidth 5.5 24750 1200 = 3.75 :=
by
  sorry

end width_of_room_l126_126232


namespace proof_problem_l126_126341

noncomputable def f (x : ℝ) := 2 * Real.sin (π * x / 6 + π / 3)

theorem proof_problem :
  (∃ A B : ℝ × ℝ, A = (1, 2) ∧ B = (5, -1) ∧
    let OA := (A.1, A.2) in
    let OB := (B.1, B.2) in
    OA.1 * OB.1 + OA.2 * OB.2 = 3) ∧
  (∃ α β : ℝ, Real.tan α = 2 ∧ Real.tan β = -1 / 5 ∧ Real.tan (α - 2 * β) = 29 / 2) :=
sorry

end proof_problem_l126_126341


namespace probability_of_common_books_l126_126084

-- Definitions based on conditions
def total_ways_4_books (n : ℕ) (k : ℕ) : ℕ :=
  nat.choose n k

def favorable_outcomes (n k : ℕ) (common_books : ℕ) :=
  nat.choose n common_books * nat.choose (n - common_books) k * nat.choose (n - common_books - (k - common_books)) k

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

-- Given the conditions of the problem
def problem_statement : Prop :=
  let total := (total_ways_4_books 12 4) * (total_ways_4_books 12 4) in
  let favorable := favorable_outcomes 12 4 2 in
  probability favorable total = 36 / 105

-- The proof is omitted with sorry
theorem probability_of_common_books :
  problem_statement :=
sorry

end probability_of_common_books_l126_126084


namespace sum_of_cubes_l126_126841

theorem sum_of_cubes (x y z : ℝ) (h1 : x + y + z = 2) (h2 : x * y + y * z + z * x = -3) (h3 : x * y * z = 2) : 
  x^3 + y^3 + z^3 = 32 := 
sorry

end sum_of_cubes_l126_126841


namespace remainder_div_1234567_256_l126_126447

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l126_126447


namespace max_distance_AB_l126_126990

-- Define curve C1 in Cartesian coordinates
def C1 (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 4

-- Define curve C2 in Cartesian coordinates
def C2 (x y : ℝ) : Prop := x^2 + y^2 = 1

-- Define the problem to prove the maximum value of distance AB is 8
theorem max_distance_AB :
  ∀ (Ax Ay Bx By : ℝ),
    C1 Ax Ay →
    C2 Bx By →
    dist (Ax, Ay) (Bx, By) ≤ 8 :=
sorry

end max_distance_AB_l126_126990


namespace silk_per_dress_l126_126432

theorem silk_per_dress (initial_silk : ℕ) (friends : ℕ) (silk_per_friend : ℕ) (total_dresses : ℕ)
  (h1 : initial_silk = 600)
  (h2 : friends = 5)
  (h3 : silk_per_friend = 20)
  (h4 : total_dresses = 100)
  (remaining_silk := initial_silk - friends * silk_per_friend) :
  remaining_silk / total_dresses = 5 :=
by
  -- proof goes here
  sorry

end silk_per_dress_l126_126432


namespace linear_coefficient_l126_126683

theorem linear_coefficient (m x : ℝ) (h1 : (m - 3) * x ^ (m^2 - 2 * m - 1) - m * x + 6 = 0) (h2 : (m^2 - 2 * m - 1 = 2)) (h3 : m ≠ 3) : 
  ∃ a b c : ℝ, a * x ^ 2 + b * x + c = 0 ∧ b = 1 :=
by
  sorry

end linear_coefficient_l126_126683


namespace range_of_k_l126_126478

theorem range_of_k (k : ℝ) (x y : ℝ) : 
  (y = 2 * x - 5 * k + 7) → 
  (y = - (1 / 2) * x + 2) → 
  (x > 0) → 
  (y > 0) → 
  (1 < k ∧ k < 3) :=
by
  sorry

end range_of_k_l126_126478


namespace books_already_read_l126_126848

def total_books : ℕ := 20
def unread_books : ℕ := 5

theorem books_already_read : (total_books - unread_books = 15) :=
by
 -- Proof goes here
 sorry

end books_already_read_l126_126848


namespace best_fit_model_l126_126557

-- Definition of the given R^2 values for different models
def R2_A : ℝ := 0.62
def R2_B : ℝ := 0.63
def R2_C : ℝ := 0.68
def R2_D : ℝ := 0.65

-- Theorem statement that model with R2_C has the best fitting effect
theorem best_fit_model : R2_C = max R2_A (max R2_B (max R2_C R2_D)) :=
by
  sorry -- Proof is not required

end best_fit_model_l126_126557


namespace martin_speed_l126_126212

theorem martin_speed (distance time : ℝ) (h_distance : distance = 12) (h_time : time = 6) :
  distance / time = 2 :=
by
  rw [h_distance, h_time]
  norm_num

end martin_speed_l126_126212


namespace gcd_of_sum_and_sum_of_squares_l126_126371

theorem gcd_of_sum_and_sum_of_squares {a b : ℕ} (h : Nat.gcd a b = 1) :
  Nat.gcd (a + b) (a^2 + b^2) = 1 ∨ Nat.gcd (a + b) (a^2 + b^2) = 2 :=
sorry

end gcd_of_sum_and_sum_of_squares_l126_126371


namespace intersect_sets_l126_126633

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | abs x < 2}

theorem intersect_sets :
  (set_M ∩ set_N) = {x | -1 ≤ x ∧ x < 2} :=
sorry

end intersect_sets_l126_126633


namespace triangle_side_relation_l126_126959

theorem triangle_side_relation (a b c : ℝ) (α β γ : ℝ)
  (h1 : 3 * α + 2 * β = 180)
  (h2 : α + β + γ = 180) :
  a^2 + b * c - c^2 = 0 :=
sorry

end triangle_side_relation_l126_126959


namespace length_of_first_square_flag_l126_126286

theorem length_of_first_square_flag
  (x : ℝ)
  (h1x : x * 5 + 10 * 7 + 5 * 5 = 15 * 9) : 
  x = 8 :=
by
  sorry

end length_of_first_square_flag_l126_126286


namespace min_value_of_expr_l126_126514

noncomputable def min_value_expr (x y : ℝ) : ℝ :=
  ((x^2 + 1 / y^2 + 1) * (x^2 + 1 / y^2 - 1000)) +
  ((y^2 + 1 / x^2 + 1) * (y^2 + 1 / x^2 - 1000))

theorem min_value_of_expr :
  ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ min_value_expr x y = -498998 :=
by
  sorry

end min_value_of_expr_l126_126514


namespace percent_integers_no_remainder_6_equals_16_67_l126_126694

theorem percent_integers_no_remainder_6_equals_16_67 :
  let N := 150 in
  let divisible_by_6_count := N / 6 in
  let percentage := (divisible_by_6_count / N) * 100 in
  percentage = 16.67 :=
by
  sorry

end percent_integers_no_remainder_6_equals_16_67_l126_126694


namespace max_min_y_l126_126749

noncomputable def y (x : ℝ) : ℝ := (Real.sin x)^(2:ℝ) + 2 * (Real.sin x) * (Real.cos x) + 3 * (Real.cos x)^(2:ℝ)

theorem max_min_y : 
  ∀ x : ℝ, 
  2 - Real.sqrt 2 ≤ y x ∧ y x ≤ 2 + Real.sqrt 2 :=
by sorry

end max_min_y_l126_126749


namespace root_of_equation_l126_126019

theorem root_of_equation : 
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2) → (x = -4.5) :=
by sorry

end root_of_equation_l126_126019


namespace sum_of_GCF_and_LCM_l126_126884

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l126_126884


namespace percentage_of_second_division_l126_126191

theorem percentage_of_second_division
  (total_students : ℕ)
  (students_first_division : ℕ)
  (students_just_passed : ℕ)
  (h1: total_students = 300)
  (h2: students_first_division = 75)
  (h3: students_just_passed = 63) :
  (total_students - (students_first_division + students_just_passed)) * 100 / total_students = 54 := 
by
  -- Proof will be added later
  sorry

end percentage_of_second_division_l126_126191


namespace original_two_digit_number_l126_126430

theorem original_two_digit_number :
  ∃ (x : ℕ), 10 ≤ x ∧ x < 100 ∧ 500 + x = 9 * x - 12 ∧ x = 64 :=
by
  have h₁ : ∀ (x : ℕ), 500 + x = 9 * x - 12 → x = 64 := sorry
  use 64
  split
  all_goals { sorry }

end original_two_digit_number_l126_126430


namespace recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l126_126769

def a (n : ℕ) : ℕ := n
def b (n : ℕ) : ℕ := n^2
def c (n : ℕ) : ℕ := n^3
def d (n : ℕ) : ℕ := n^4
def e (n : ℕ) : ℕ := n^5

theorem recursive_relation_a (n : ℕ) : a (n+2) = 2 * a (n+1) - a n :=
by sorry

theorem recursive_relation_b (n : ℕ) : b (n+3) = 3 * b (n+2) - 3 * b (n+1) + b n :=
by sorry

theorem recursive_relation_c (n : ℕ) : c (n+4) = 4 * c (n+3) - 6 * c (n+2) + 4 * c (n+1) - c n :=
by sorry

theorem recursive_relation_d (n : ℕ) : d (n+5) = 5 * d (n+4) - 10 * d (n+3) + 10 * d (n+2) - 5 * d (n+1) + d n :=
by sorry

theorem recursive_relation_e (n : ℕ) : 
  e (n+6) = 6 * e (n+5) - 15 * e (n+4) + 20 * e (n+3) - 15 * e (n+2) + 6 * e (n+1) - e n :=
by sorry

end recursive_relation_a_recursive_relation_b_recursive_relation_c_recursive_relation_d_recursive_relation_e_l126_126769


namespace sine_180_eq_zero_l126_126295

theorem sine_180_eq_zero :
  sin (180 : ℝ) = 0 :=
sorry

end sine_180_eq_zero_l126_126295


namespace true_false_questions_count_l126_126096

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end true_false_questions_count_l126_126096


namespace smallest_class_number_l126_126928

-- Define the conditions
def num_classes : Nat := 24
def num_selected_classes : Nat := 4
def total_sum : Nat := 52
def sampling_interval : Nat := num_classes / num_selected_classes

-- The core theorem to be proved
theorem smallest_class_number :
  ∃ x : Nat, x + (x + sampling_interval) + (x + 2 * sampling_interval) + (x + 3 * sampling_interval) = total_sum ∧ x = 4 := by
  sorry

end smallest_class_number_l126_126928


namespace gray_eyed_black_haired_students_l126_126317

theorem gray_eyed_black_haired_students :
  ∀ (students : ℕ)
    (green_eyed_red_haired : ℕ)
    (black_haired : ℕ)
    (gray_eyed : ℕ),
    students = 60 →
    green_eyed_red_haired = 20 →
    black_haired = 40 →
    gray_eyed = 25 →
    (gray_eyed - (students - black_haired - green_eyed_red_haired)) = 25 := by
  intros students green_eyed_red_haired black_haired gray_eyed
  intros h_students h_green h_black h_gray
  sorry

end gray_eyed_black_haired_students_l126_126317


namespace inequality_holds_for_all_x_l126_126492

theorem inequality_holds_for_all_x (m : ℝ) : 
  (∀ x : ℝ, (m * x^2 + 2 * m * x - 4 < 2 * x^2 + 4 * x)) ↔ -2 < m ∧ m ≤ 2 := 
by
  sorry

end inequality_holds_for_all_x_l126_126492


namespace sum_of_gcd_and_lcm_is_28_l126_126890

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l126_126890


namespace papaya_cost_is_one_l126_126122

theorem papaya_cost_is_one (lemons_cost : ℕ) (mangos_cost : ℕ) (total_fruits : ℕ) (total_cost_paid : ℕ) :
    (lemons_cost = 2) → (mangos_cost = 4) → (total_fruits = 12) → (total_cost_paid = 21) → 
    let discounts := total_fruits / 4
    let lemons_bought := 6
    let mangos_bought := 2
    let papayas_bought := 4
    let total_discount := discounts
    let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
    total_cost_before_discount - total_discount = total_cost_paid → 
    P = 1 := 
by 
  intros h1 h2 h3 h4 
  let discounts := total_fruits / 4
  let lemons_bought := 6
  let mangos_bought := 2
  let papayas_bought := 4
  let total_discount := discounts
  let total_cost_before_discount := lemons_bought * lemons_cost + mangos_bought * mangos_cost + papayas_bought * P
  sorry

end papaya_cost_is_one_l126_126122


namespace total_guests_at_least_one_reunion_l126_126246

-- Definitions used in conditions
def attendeesOates := 42
def attendeesYellow := 65
def attendeesBoth := 7

-- Definition of the total number of guests attending at least one of the reunions
def totalGuests := attendeesOates + attendeesYellow - attendeesBoth

-- Theorem stating that the total number of guests is equal to 100
theorem total_guests_at_least_one_reunion : totalGuests = 100 :=
by
  -- skipping the proof with sorry
  sorry

end total_guests_at_least_one_reunion_l126_126246


namespace integer_value_of_a_l126_126703

theorem integer_value_of_a (a x y z k : ℤ) :
  (x = k) ∧ (y = 4 * k) ∧ (z = 5 * k) ∧ (y = 9 * a^2 - 2 * a - 8) ∧ (z = 10 * a + 2) → a = 5 :=
by 
  sorry

end integer_value_of_a_l126_126703


namespace original_square_area_is_correct_l126_126187

noncomputable def original_square_side_length (s : ℝ) :=
  let original_area := s^2
  let new_width := 0.8 * s
  let new_length := 5 * s
  let new_area := new_width * new_length
  let increased_area := new_area - original_area
  increased_area = 15.18

theorem original_square_area_is_correct (s : ℝ) (h : original_square_side_length s) : s^2 = 5.06 := by
  sorry

end original_square_area_is_correct_l126_126187


namespace Cornelia_three_times_Kilee_l126_126986

variable (x : ℕ)

def Kilee_current_age : ℕ := 20
def Cornelia_current_age : ℕ := 80

theorem Cornelia_three_times_Kilee (x : ℕ) :
  Cornelia_current_age + x = 3 * (Kilee_current_age + x) ↔ x = 10 :=
by
  sorry

end Cornelia_three_times_Kilee_l126_126986


namespace computation_is_correct_l126_126617

def large_multiplication : ℤ := 23457689 * 84736521

def denominator_subtraction : ℤ := 7589236 - 3145897

def computed_m : ℚ := large_multiplication / denominator_subtraction

theorem computation_is_correct : computed_m = 447214.999 :=
by 
  -- exact calculation to be provided
  sorry

end computation_is_correct_l126_126617


namespace darnell_saves_money_l126_126594

-- Define conditions
def current_plan_cost := 12
def text_cost := 1
def call_cost := 3
def texts_per_month := 60
def calls_per_month := 60
def texts_per_unit := 30
def calls_per_unit := 20

-- Define the costs for the alternative plan
def alternative_texting_cost := (text_cost * (texts_per_month / texts_per_unit))
def alternative_calling_cost := (call_cost * (calls_per_month / calls_per_unit))
def alternative_plan_cost := alternative_texting_cost + alternative_calling_cost

-- Define the problem to prove
theorem darnell_saves_money :
  current_plan_cost - alternative_plan_cost = 1 :=
by
  sorry

end darnell_saves_money_l126_126594


namespace prime_iff_good_fractions_l126_126332

def isGoodFraction (n : ℕ) (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ (a + b = n)

def canBeExpressedUsingGoodFractions (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (expressedFraction : ℕ → ℕ → Prop), expressedFraction a b ∧
  ∀ x y, expressedFraction x y → isGoodFraction n x y

theorem prime_iff_good_fractions {n : ℕ} (hn : n > 1) :
  Prime n ↔
    ∀ a b : ℕ, b < n → (a > 0 ∧ b > 0) → canBeExpressedUsingGoodFractions n a b :=
sorry

end prime_iff_good_fractions_l126_126332


namespace prob_240_yuan_refund_l126_126741


def spinner_probability (n : ℕ) (p : ℚ) : ℚ := (Nat.choose 3 n) * (p^n) * ((1 - p)^(3-n))

def refund_probability : ℚ :=
1 - (spinner_probability 0 (1/6)) - (spinner_probability 3 (1/6))

theorem prob_240_yuan_refund : refund_probability = 5/12 := by
  sorry

end prob_240_yuan_refund_l126_126741


namespace proof_parabola_statements_l126_126481

theorem proof_parabola_statements (b c : ℝ)
  (h1 : 1/2 - b + c < 0)
  (h2 : 2 - 2 * b + c < 0) :
  (b^2 > 2 * c) ∧
  (c > 1 → b > 3/2) ∧
  (∀ (m1 m2 : ℝ), m1 < m2 ∧ m2 < b → ∀ (y : ℝ), y = (1/2)*m1^2 - b*m1 + c → ∀ (y2 : ℝ), y2 = (1/2)*m2^2 - b*m2 + c → y > y2) ∧
  (¬(∃ x1 x2 : ℝ, (1/2) * x1^2 - b * x1 + c = 0 ∧ (1/2) * x2^2 - b * x2 + c = 0 ∧ x1 + x2 > 3)) :=
by sorry

end proof_parabola_statements_l126_126481


namespace fraction_of_smart_integers_divisible_by_25_l126_126160

def is_smart_integer (n : ℕ) : Prop :=
  even n ∧ 20 < n ∧ n < 120 ∧ (n.digits 10).sum = 10

def is_divisible_by_25 (n : ℕ) : Prop :=
  n % 25 = 0

theorem fraction_of_smart_integers_divisible_by_25 : 
  (Finset.filter is_divisible_by_25 (Finset.filter is_smart_integer (Finset.range 120))).card = 0 :=
by
  sorry

end fraction_of_smart_integers_divisible_by_25_l126_126160


namespace final_stack_height_l126_126654

theorem final_stack_height (x : ℕ) 
  (first_stack_height : ℕ := 7) 
  (second_stack_height : ℕ := first_stack_height + 5) 
  (final_stack_height : ℕ := second_stack_height + x) 
  (blocks_fell_first : ℕ := first_stack_height) 
  (blocks_fell_second : ℕ := second_stack_height - 2) 
  (blocks_fell_final : ℕ := final_stack_height - 3) 
  (total_blocks_fell : 33 = blocks_fell_first + blocks_fell_second + blocks_fell_final) 
  : x = 7 :=
  sorry

end final_stack_height_l126_126654


namespace remainder_of_division_l126_126444

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l126_126444


namespace race_probability_l126_126501

theorem race_probability (Px : ℝ) (Py : ℝ) (Pz : ℝ) 
  (h1 : Px = 1 / 6) 
  (h2 : Pz = 1 / 8) 
  (h3 : Px + Py + Pz = 0.39166666666666666) : Py = 0.1 := 
sorry

end race_probability_l126_126501


namespace tan_value_l126_126329

theorem tan_value (x : ℝ) (hx : x ∈ Set.Ioo (-π / 2) 0) (hcos : Real.cos x = 4 / 5) : Real.tan x = -3 / 4 :=
sorry

end tan_value_l126_126329


namespace min_abs_sum_l126_126619

theorem min_abs_sum (x y : ℝ) : (|x - 1| + |x| + |y - 1| + |y + 1|) ≥ 3 :=
sorry

end min_abs_sum_l126_126619


namespace triangle_in_base_7_l126_126598

theorem triangle_in_base_7 (triangle : ℕ) 
  (h1 : (triangle + 6) % 7 = 0) : 
  triangle = 1 := 
sorry

end triangle_in_base_7_l126_126598


namespace fraction_evaluation_l126_126693

theorem fraction_evaluation : (20 + 24) / (20 - 24) = -11 := by
  sorry

end fraction_evaluation_l126_126693


namespace expression_equals_5000_l126_126589

theorem expression_equals_5000 :
  12 * 171 + 29 * 9 + 171 * 13 + 29 * 16 = 5000 :=
by
  sorry

end expression_equals_5000_l126_126589


namespace inequality_solution_real_l126_126094

theorem inequality_solution_real (x : ℝ) :
  (x + 1) * (2 - x) < 4 ↔ true :=
by
  sorry

end inequality_solution_real_l126_126094


namespace subscriptions_to_grandfather_l126_126829

/-- 
Maggie earns $5.00 for every magazine subscription sold. 
She sold 4 subscriptions to her parents, 2 to the next-door neighbor, 
and twice that amount to another neighbor. Maggie earned $55 in total. 
Prove that the number of subscriptions Maggie sold to her grandfather is 1.
-/
theorem subscriptions_to_grandfather (G : ℕ) 
  (h1 : 5 * (4 + G + 2 + 4) = 55) : 
  G = 1 :=
by {
  sorry
}

end subscriptions_to_grandfather_l126_126829


namespace no_integer_roots_p_eq_2016_l126_126964

noncomputable def p (a b c d : ℤ) (x : ℤ) : ℤ := a * x^3 + b * x^2 + c * x + d

theorem no_integer_roots_p_eq_2016 
  (a b c d : ℤ)
  (h₁ : p a b c d 1 = 2015)
  (h₂ : p a b c d 2 = 2017) :
  ¬ ∃ x : ℤ, p a b c d x = 2016 :=
sorry

end no_integer_roots_p_eq_2016_l126_126964


namespace find_X_l126_126324

theorem find_X (X : ℝ) 
  (h : 2 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * X)) = 1600.0000000000002) : 
  X = 1.25 := 
sorry

end find_X_l126_126324


namespace trigonometric_identity_l126_126240

theorem trigonometric_identity :
  (Real.cos (12 * Real.pi / 180) * Real.cos (18 * Real.pi / 180) - 
   Real.sin (12 * Real.pi / 180) * Real.sin (18 * Real.pi / 180) = 
   Real.cos (30 * Real.pi / 180)) :=
by
  sorry

end trigonometric_identity_l126_126240


namespace area_of_smallest_square_that_encloses_circle_l126_126127

def radius : ℕ := 5

def diameter (r : ℕ) : ℕ := 2 * r

def side_length (d : ℕ) : ℕ := d

def area_of_square (s : ℕ) : ℕ := s * s

theorem area_of_smallest_square_that_encloses_circle :
  area_of_square (side_length (diameter radius)) = 100 := by
  sorry

end area_of_smallest_square_that_encloses_circle_l126_126127


namespace correct_calculation_l126_126130

theorem correct_calculation :
  (∀ a : ℝ, a^3 + a^2 ≠ a^5) ∧
  (∀ a : ℝ, a^3 / a^2 = a) ∧
  (∀ a : ℝ, 3 * a^3 * 2 * a^2 ≠ 6 * a^6) ∧
  (∀ a : ℝ, (a - 2)^2 ≠ a^2 - 4) :=
by
  sorry

end correct_calculation_l126_126130


namespace highest_score_is_174_l126_126099

theorem highest_score_is_174
  (avg_40_innings : ℝ)
  (highest_exceeds_lowest : ℝ)
  (avg_excl_two : ℝ)
  (total_runs_40 : ℝ)
  (total_runs_38 : ℝ)
  (sum_H_L : ℝ)
  (new_avg_38 : ℝ)
  (H : ℝ)
  (L : ℝ)
  (H_eq_L_plus_172 : H = L + 172)
  (total_runs_40_eq : total_runs_40 = 40 * avg_40_innings)
  (total_runs_38_eq : total_runs_38 = 38 * new_avg_38)
  (sum_H_L_eq : sum_H_L = total_runs_40 - total_runs_38)
  (new_avg_eq : new_avg_38 = avg_40_innings - 2)
  (sum_H_L_val : sum_H_L = 176)
  (avg_40_val : avg_40_innings = 50) :
  H = 174 :=
sorry

end highest_score_is_174_l126_126099


namespace sqrt_expression_l126_126458

theorem sqrt_expression (y : ℝ) (hy : y < 0) : 
  Real.sqrt (y / (1 - ((y - 2) / y))) = -y / Real.sqrt 2 := 
sorry

end sqrt_expression_l126_126458


namespace inverse_sum_l126_126508

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then 3 - x else 2 * x - x^2

theorem inverse_sum :
  let f_inv_2 := (1 + Real.sqrt 3)
  let f_inv_1 := 2
  let f_inv_4 := -1
  f_inv_2 + f_inv_1 + f_inv_4 = 2 + Real.sqrt 3 :=
by
  sorry

end inverse_sum_l126_126508


namespace no_cell_with_sum_2018_l126_126706

theorem no_cell_with_sum_2018 : ∀ (x : ℕ), 1 ≤ x ∧ x ≤ 4900 → (5 * x = 2018 → false) := 
by
  intros x hx
  have h_bound : 1 ≤ x ∧ x ≤ 4900 := hx
  sorry

end no_cell_with_sum_2018_l126_126706


namespace compute_f_1_g_3_l126_126078

def f (x : ℝ) := 3 * x - 5
def g (x : ℝ) := x + 1

theorem compute_f_1_g_3 : f (1 + g 3) = 10 := by
  sorry

end compute_f_1_g_3_l126_126078


namespace not_all_angles_less_than_60_l126_126698

-- Definitions relating to interior angles of a triangle
def triangle (α β γ : ℝ) : Prop := α + β + γ = 180

theorem not_all_angles_less_than_60 (α β γ : ℝ) 
(h_triangle : triangle α β γ) 
(h1 : α < 60) 
(h2 : β < 60) 
(h3 : γ < 60) : False :=
    -- The proof steps would be placed here
sorry

end not_all_angles_less_than_60_l126_126698


namespace book_pricing_and_min_cost_l126_126261

-- Define the conditions
def price_relation (a : ℝ) (ps_price : ℝ) : Prop :=
  ps_price = 1.2 * a

def book_count_relation (a : ℝ) (lit_count ps_count : ℕ) : Prop :=
  lit_count = 1200 / a ∧ ps_count = 1200 / (1.2 * a) ∧ lit_count - ps_count = 10

def min_cost_condition (x : ℕ) : Prop :=
  x ≤ 600

def total_cost (x : ℕ) : ℝ :=
  20 * x + 24 * (1000 - x)

-- The theorem combining all parts
theorem book_pricing_and_min_cost:
  ∃ (a : ℝ) (ps_price : ℝ) (lit_count ps_count : ℕ),
    price_relation a ps_price ∧
    book_count_relation a lit_count ps_count ∧
    a = 20 ∧ ps_price = 24 ∧
    (∀ (x : ℕ), min_cost_condition x → total_cost x ≥ 21600) ∧
    (total_cost 600 = 21600) :=
by
  sorry

end book_pricing_and_min_cost_l126_126261


namespace y_intercepts_parabola_l126_126457

theorem y_intercepts_parabola : 
  ∀ (y : ℝ), ¬(0 = 3 * y^2 - 5 * y + 12) :=
by 
  -- Given x = 0, we have the equation 3 * y^2 - 5 * y + 12 = 0.
  -- The discriminant ∆ = b^2 - 4ac = (-5)^2 - 4 * 3 * 12 = 25 - 144 = -119 which is less than 0.
  -- Since the discriminant is negative, the quadratic equation has no real roots.
  sorry

end y_intercepts_parabola_l126_126457


namespace probability_open_lock_l126_126763

/-- Given 5 keys and only 2 can open the lock, the probability of opening the lock by selecting one key randomly is 0.4. -/
theorem probability_open_lock (k : Finset ℕ) (h₁ : k.card = 5) (s : Finset ℕ) (h₂ : s.card = 2 ∧ s ⊆ k) :
  ∃ p : ℚ, p = 0.4 :=
by
  sorry

end probability_open_lock_l126_126763


namespace find_equation_AC_l126_126764

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end find_equation_AC_l126_126764


namespace john_pays_2010_dollars_l126_126368

-- Define the main problem as the number of ways to pay 2010$ using 2, 5, and 10$ notes.
theorem john_pays_2010_dollars :
  ∃ (count : ℕ), count = 20503 ∧
  ∀ (x y z : ℕ), (2 * x + 5 * y + 10 * z = 2010) → (x % 5 = 0) → (y % 2 = 0) → count = 20503 :=
by sorry

end john_pays_2010_dollars_l126_126368


namespace area_of_triangle_AEB_l126_126192

structure Rectangle :=
  (A B C D : Type)
  (AB : ℝ)
  (BC : ℝ)
  (F G E : Type)
  (DF : ℝ)
  (GC : ℝ)
  (AF_BG_intersect_at_E : Prop)

def rectangle_example : Rectangle := {
  A := Unit,
  B := Unit,
  C := Unit,
  D := Unit,
  AB := 8,
  BC := 4,
  F := Unit,
  G := Unit,
  E := Unit,
  DF := 2,
  GC := 3,
  AF_BG_intersect_at_E := true
}

theorem area_of_triangle_AEB (r : Rectangle) (h : r = rectangle_example) :
  ∃ area : ℝ, area = 128 / 3 :=
by
  sorry

end area_of_triangle_AEB_l126_126192


namespace distinct_prime_factors_count_l126_126792

theorem distinct_prime_factors_count {n : ℕ} (h : n = 3^2 * 2 * 5) : (nat.factors n).nodup.card = 3 :=
by {
  have h1: nat.factors n = [3, 3, 2, 5],
  { rw h, refl },
  have h2: list.dedup (nat.factors n) = [3, 2, 5],
  { rw h1, refl },
  show (nat.factors n).nodup.card = 3,
  {
    rw [list.nodup_iff_dedup_eq_self, h2],
    refl,
  },
  sorry,
}

end distinct_prime_factors_count_l126_126792


namespace remainder_of_3_pow_100_mod_7_is_4_l126_126128

theorem remainder_of_3_pow_100_mod_7_is_4
  (h1 : 3^1 ≡ 3 [MOD 7])
  (h2 : 3^2 ≡ 2 [MOD 7])
  (h3 : 3^3 ≡ 6 [MOD 7])
  (h4 : 3^4 ≡ 4 [MOD 7])
  (h5 : 3^5 ≡ 5 [MOD 7])
  (h6 : 3^6 ≡ 1 [MOD 7]) :
  3^100 ≡ 4 [MOD 7] :=
by
  sorry

end remainder_of_3_pow_100_mod_7_is_4_l126_126128


namespace intersection_point_of_lines_l126_126843

theorem intersection_point_of_lines :
  (∃ x y : ℝ, y = x ∧ y = -x + 2 ∧ (x = 1 ∧ y = 1)) :=
sorry

end intersection_point_of_lines_l126_126843


namespace probability_of_qualified_product_l126_126245

theorem probability_of_qualified_product :
  let p1 := 0.30   -- Proportion of the first batch
  let d1 := 0.05   -- Defect rate of the first batch
  let p2 := 0.70   -- Proportion of the second batch
  let d2 := 0.04   -- Defect rate of the second batch
  -- Probability of selecting a qualified product
  p1 * (1 - d1) + p2 * (1 - d2) = 0.957 :=
by
  sorry

end probability_of_qualified_product_l126_126245


namespace problem_statement_l126_126625

theorem problem_statement (x y : ℝ) (h1 : 1/x + 1/y = 5) (h2 : x * y + x + y = 7) : 
  x^2 * y + x * y^2 = 245 / 36 := 
by
  sorry

end problem_statement_l126_126625


namespace imaginary_unit_power_l126_126230

-- Definition of the imaginary unit i
def imaginary_unit_i : ℂ := Complex.I

theorem imaginary_unit_power :
  (imaginary_unit_i ^ 2015) = -imaginary_unit_i := by
  sorry

end imaginary_unit_power_l126_126230


namespace museum_revenue_from_college_students_l126_126063

/-!
In one day, 200 people visit The Metropolitan Museum of Art in New York City. Half of the visitors are residents of New York City. 
Of the NYC residents, 30% are college students. If the cost of a college student ticket is $4, we need to prove that 
the museum gets $120 from college students that are residents of NYC.
-/

theorem museum_revenue_from_college_students :
  let total_visitors := 200
  let residents_nyc := total_visitors / 2
  let college_students_percentage := 30 / 100
  let college_students := residents_nyc * college_students_percentage
  let ticket_cost := 4
  residents_nyc = 100 ∧ 
  college_students = 30 ∧ 
  ticket_cost * college_students = 120 := 
by
  sorry

end museum_revenue_from_college_students_l126_126063


namespace bottles_per_crate_l126_126819

theorem bottles_per_crate (num_bottles total_bottles bottles_not_placed num_crates : ℕ) 
    (h1 : total_bottles = 130)
    (h2 : bottles_not_placed = 10)
    (h3 : num_crates = 10) 
    (h4 : num_bottles = total_bottles - bottles_not_placed) :
    (num_bottles / num_crates) = 12 := 
by 
    sorry

end bottles_per_crate_l126_126819


namespace sam_money_left_l126_126836

/- Definitions -/

def initial_dimes : ℕ := 38
def initial_quarters : ℕ := 12
def initial_nickels : ℕ := 25
def initial_pennies : ℕ := 30

def price_per_candy_bar_dimes : ℕ := 4
def price_per_candy_bar_nickels : ℕ := 2
def candy_bars_bought : ℕ := 5

def price_per_lollipop_nickels : ℕ := 6
def price_per_lollipop_pennies : ℕ := 10
def lollipops_bought : ℕ := 2

def price_per_bag_of_chips_quarters : ℕ := 1
def price_per_bag_of_chips_dimes : ℕ := 3
def price_per_bag_of_chips_pennies : ℕ := 5
def bags_of_chips_bought : ℕ := 3

/- Proof problem statement -/

theorem sam_money_left : 
  (initial_dimes * 10 + initial_quarters * 25 + initial_nickels * 5 + initial_pennies * 1) - 
  (
    candy_bars_bought * (price_per_candy_bar_dimes * 10 + price_per_candy_bar_nickels * 5) + 
    lollipops_bought * (price_per_lollipop_nickels * 5 + price_per_lollipop_pennies * 1) +
    bags_of_chips_bought * (price_per_bag_of_chips_quarters * 25 + price_per_bag_of_chips_dimes * 10 + price_per_bag_of_chips_pennies * 1)
  ) = 325 := 
sorry

end sam_money_left_l126_126836


namespace solve_linear_system_l126_126221

variable {x y : ℚ}

theorem solve_linear_system (h1 : 4 * x - 3 * y = -17) (h2 : 5 * x + 6 * y = -4) :
  (x, y) = (-(74 / 13 : ℚ), -(25 / 13 : ℚ)) :=
by
  sorry

end solve_linear_system_l126_126221


namespace smallest_integer_switch_add_l126_126401

theorem smallest_integer_switch_add (a b: ℕ) (h1: n = 10 * a + b) 
  (h2: 3 * n = 10 * b + a + 5)
  (h3: 0 ≤ b) (h4: b < 10) (h5: 1 ≤ a) (h6: a < 10): n = 47 :=
by
  sorry

end smallest_integer_switch_add_l126_126401


namespace triangle_area_l126_126428

theorem triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ ∃ (A : ℝ), 
  A = Real.sqrt (6 * (6 - a) * (6 - b) * (6 - c)) ∧ A = 6 := by
  sorry

end triangle_area_l126_126428


namespace council_revote_l126_126498

theorem council_revote (x y x' y' m : ℝ) (h1 : x + y = 500)
    (h2 : y - x = m) (h3 : x' - y' = 1.5 * m) (h4 : x' + y' = 500) (h5 : x' = 11 / 10 * y) :
    x' - x = 156.25 := by
  -- Proof goes here
  sorry

end council_revote_l126_126498


namespace avianna_blue_candles_l126_126671

theorem avianna_blue_candles (r b : ℕ) (h1 : r = 45) (h2 : r/b = 5/3) : b = 27 :=
by sorry

end avianna_blue_candles_l126_126671


namespace average_std_dev_qualified_prob_wang_qiang_l126_126110

variables {μ σ : ℝ} {students : ℕ}
variables {scores : ℕ → ℝ}
variables {groupA_scores : Fin 24 → ℝ}
variables {groupB_scores : Fin 16 → ℝ}
variables (μ σ students groupA_scores groupB_scores)

-- Given conditions
def groupA_mean : ℝ := 70
def groupB_mean : ℝ := 80
def groupA_std_dev : ℝ := 4
def groupB_std_dev : ℝ := 6
def groupA : Fin 24 → ℝ := groupA_scores
def groupB : Fin 16 → ℝ := groupB_scores

-- Definitions for group properties
def mean (scores : List ℝ) : ℝ := (scores.sum) / (scores.length)
def variance (scores : List ℝ) : ℝ := (scores.map (λ x, (x - mean scores) ^ 2)).sum / scores.length
def std_dev (scores : List ℝ) : ℕ := real.sqrt (variance scores)

-- Translate average and standard deviation calculation
theorem average_std_dev :
  mean ((List.ofFn groupA) ++ (List.ofFn groupB)) = 74 ∧
  std_dev ((List.ofFn groupA) ++ (List.ofFn groupB)) ≈ 7.75 :=
begin
  sorry
end

-- Normal distribution properties and qualification evaluation
def normal_qualification (cutoff : ℝ) (p_threshold : ℝ) : Prop :=
  P (λ x, x < cutoff) + P (λ x, x > 1000 - cutoff) < p_threshold

theorem qualified : normal_qualification 60 0.05 :=
begin
  sorry
end

-- Probability that Wang Qiang wins first 3 games and earns 3 points
variables {p_win p_lose : ℕ}

def matchup_probability (win : ℕ) (lose : ℕ) : ℝ :=
  (nat.choose (4 + lose - 1) 3 * ((2/3)^3) * ((1/3)^lose) * (2/3))

theorem prob_wang_qiang : matchup_probability 3 1 = (2/25) :=
begin
  sorry
end

end average_std_dev_qualified_prob_wang_qiang_l126_126110


namespace trajectory_of_Q_l126_126771

variable (x y m n : ℝ)

def line_l (x y : ℝ) : Prop := 2 * x + 4 * y + 3 = 0

def point_P_on_line_l (x y m n : ℝ) : Prop := line_l m n

def origin (O : (ℝ × ℝ)) := O = (0, 0)

def Q_condition (O Q P : (ℝ × ℝ)) : Prop := 2 • O + 2 • Q = Q + P

theorem trajectory_of_Q (x y m n : ℝ) (O : (ℝ × ℝ)) (P Q : (ℝ × ℝ)) :
  point_P_on_line_l x y m n → origin O → Q_condition O Q P → 
  2 * x + 4 * y + 1 = 0 := 
sorry

end trajectory_of_Q_l126_126771


namespace codecracker_number_of_codes_l126_126195

theorem codecracker_number_of_codes : ∃ n : ℕ, n = 6 * 5^4 := by
  sorry

end codecracker_number_of_codes_l126_126195


namespace curtain_price_l126_126956

theorem curtain_price
  (C : ℝ)
  (h1 : 2 * C + 9 * 15 + 50 = 245) :
  C = 30 :=
sorry

end curtain_price_l126_126956


namespace inequality_proof_l126_126335

theorem inequality_proof (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 3) :
  (a^2 + 9) / (2*a^2 + (b+c)^2) + (b^2 + 9) / (2*b^2 + (c+a)^2) + (c^2 + 9) / (2*c^2 + (a+b)^2) ≤ 5 :=
by
  sorry

end inequality_proof_l126_126335


namespace remainder_of_division_l126_126441

theorem remainder_of_division:
  1234567 % 256 = 503 :=
sorry

end remainder_of_division_l126_126441


namespace cookies_in_fridge_l126_126867

theorem cookies_in_fridge (total_baked : ℕ) (cookies_Tim : ℕ) (cookies_Mike : ℕ) (cookies_Sarah : ℕ) (cookies_Anna : ℕ)
  (h_total_baked : total_baked = 1024)
  (h_cookies_Tim : cookies_Tim = 48)
  (h_cookies_Mike : cookies_Mike = 58)
  (h_cookies_Sarah : cookies_Sarah = 78)
  (h_cookies_Anna : cookies_Anna = (2 * (cookies_Tim + cookies_Mike)) - (cookies_Sarah / 2)) :
  total_baked - (cookies_Tim + cookies_Mike + cookies_Sarah + cookies_Anna) = 667 := by
sorry

end cookies_in_fridge_l126_126867


namespace diagonals_in_30_sided_polygon_l126_126349

def number_of_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_in_30_sided_polygon :
  number_of_diagonals 30 = 405 :=
by sorry

end diagonals_in_30_sided_polygon_l126_126349


namespace expand_expression_l126_126743

theorem expand_expression : ∀ (x : ℝ), 2 * (x + 3) * (x^2 - 2*x + 7) = 2*x^3 + 2*x^2 + 2*x + 42 := 
by
  intro x
  sorry

end expand_expression_l126_126743


namespace andrew_worked_days_l126_126146

-- Definitions per given conditions
def vacation_days_per_work_days (W : ℕ) : ℕ := W / 10
def days_taken_off_in_march := 5
def days_taken_off_in_september := 2 * days_taken_off_in_march
def total_days_off_taken := days_taken_off_in_march + days_taken_off_in_september
def remaining_vacation_days := 15
def total_vacation_days := total_days_off_taken + remaining_vacation_days

theorem andrew_worked_days (W : ℕ) :
  vacation_days_per_work_days W = total_vacation_days → W = 300 := by
  sorry

end andrew_worked_days_l126_126146


namespace robins_hair_length_l126_126674

-- Conditions:
-- Robin cut off 4 inches of his hair.
-- After cutting, his hair is now 13 inches long.
-- Question: How long was Robin's hair before he cut it? Answer: 17 inches

theorem robins_hair_length (current_length : ℕ) (cut_length : ℕ) (initial_length : ℕ) 
  (h_cut_length : cut_length = 4) 
  (h_current_length : current_length = 13) 
  (h_initial : initial_length = current_length + cut_length) :
  initial_length = 17 :=
sorry

end robins_hair_length_l126_126674


namespace number_of_distinct_prime_factors_of_90_l126_126787

theorem number_of_distinct_prime_factors_of_90 : 
  ∃ (S : Finset ℕ), (∀ p ∈ S, Prime p) ∧ ∏ p in S, p ∣ 90 ∧ S.card = 3 :=
by
  sorry

end number_of_distinct_prime_factors_of_90_l126_126787


namespace B_pow_2017_eq_B_l126_126656

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![ ![0, 1, 0], ![0, 0, 1], ![1, 0, 0] ]

theorem B_pow_2017_eq_B : B^2017 = B := by
  sorry

end B_pow_2017_eq_B_l126_126656


namespace curve_C2_equation_l126_126672

theorem curve_C2_equation (x y : ℝ) :
  (∀ x, y = 2 * Real.sin (2 * x + π / 3) → 
    y = 2 * Real.sin (4 * (( x - π / 6) / 2))) := 
  sorry

end curve_C2_equation_l126_126672


namespace smaller_angle_at_315_l126_126039

def full_circle_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def degrees_per_hour : ℝ := full_circle_degrees / hours_on_clock
def minute_position_at_315 : ℝ := 3 * degrees_per_hour
def hour_position_at_315 : ℝ := 3 * degrees_per_hour + degrees_per_hour / 4

theorem smaller_angle_at_315 :
  minute_position_at_315 = 90 → 
  hour_position_at_315 = 3 * degrees_per_hour + degrees_per_hour / 4 → 
  abs (hour_position_at_315 - minute_position_at_315) = 7.5 :=
by 
  intro h_minute h_hour 
  rw [h_minute, h_hour]
  sorry

end smaller_angle_at_315_l126_126039


namespace intersection_sets_l126_126082

def universal_set : Set ℝ := Set.univ
def set_A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def set_B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_sets (x : ℝ) : 
  (x ∈ set_A ∩ set_B) ↔ (-2 < x ∧ x < 4) :=
by sorry

end intersection_sets_l126_126082


namespace count_arithmetic_sequence_l126_126797

theorem count_arithmetic_sequence: 
  ∃ n : ℕ, (2 + (n - 1) * 3 = 2014) ∧ n = 671 := 
sorry

end count_arithmetic_sequence_l126_126797


namespace profit_percentage_B_l126_126426

theorem profit_percentage_B (price_A price_C : ℝ) (profit_A_percentage : ℝ) : 
  price_A = 150 → 
  price_C = 225 → 
  profit_A_percentage = 20 →
  let price_B := price_A + (profit_A_percentage / 100 * price_A) in
  let profit_B := price_C - price_B in
  let profit_B_percentage := (profit_B / price_B) * 100 in
  profit_B_percentage = 25 := 
by
  intros
  simp only
  sorry

end profit_percentage_B_l126_126426


namespace units_digit_of_product_l126_126168

def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

def units_digit (n : Nat) : Nat :=
  n % 10

def target_product : Nat :=
  factorial 1 * factorial 2 * factorial 3 * factorial 4

theorem units_digit_of_product : units_digit target_product = 8 :=
  by
    sorry

end units_digit_of_product_l126_126168


namespace museum_college_students_income_l126_126062

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l126_126062


namespace symmetric_complex_division_l126_126338

theorem symmetric_complex_division :
  (∀ (z1 z2 : ℂ), z1 = 3 - (1 : ℂ) * Complex.I ∧ z2 = -(Complex.re z1) + (Complex.im z1) * Complex.I 
   → (z1 / z2) = -4/5 + (3/5) * Complex.I) := sorry

end symmetric_complex_division_l126_126338


namespace probability_of_selecting_3_co_captains_is_correct_l126_126851

def teams : List ℕ := [4, 6, 7, 9]

def probability_of_selecting_3_co_captains (n : ℕ) : ℚ :=
  if n = 4 then 1/4
  else if n = 6 then 1/20
  else if n = 7 then 1/35
  else if n = 9 then 1/84
  else 0

def total_probability : ℚ :=
  (1/4) * (probability_of_selecting_3_co_captains 4 +
            probability_of_selecting_3_co_captains 6 +
            probability_of_selecting_3_co_captains 7 +
            probability_of_selecting_3_co_captains 9)

theorem probability_of_selecting_3_co_captains_is_correct :
  total_probability = 143 / 1680 :=
by
  -- The proof will be inserted here
  sorry

end probability_of_selecting_3_co_captains_is_correct_l126_126851


namespace greatest_difference_l126_126855

-- Definitions: Number of marbles in each basket
def basketA_red : Nat := 4
def basketA_yellow : Nat := 2
def basketB_green : Nat := 6
def basketB_yellow : Nat := 1
def basketC_white : Nat := 3
def basketC_yellow : Nat := 9

-- Define the differences
def diff_basketA : Nat := basketA_red - basketA_yellow
def diff_basketB : Nat := basketB_green - basketB_yellow
def diff_basketC : Nat := basketC_yellow - basketC_white

-- The goal is to prove that 6 is the greatest difference
theorem greatest_difference : max (max diff_basketA diff_basketB) diff_basketC = 6 :=
by 
  -- The proof is not provided
  sorry

end greatest_difference_l126_126855


namespace range_of_m_l126_126510

open Real

def f (x m: ℝ) : ℝ := x^2 - 2 * x + m^2 + 3 * m - 3

def p (m: ℝ) : Prop := ∃ x, f x m < 0

def q (m: ℝ) : Prop := (5 * m - 1 > 0) ∧ (m - 2 > 0)

theorem range_of_m (m : ℝ) : ¬ (p m ∨ q m) ∧ ¬ (p m ∧ q m) → (m ≤ -4 ∨ m ≥ 2) :=
by
  sorry

end range_of_m_l126_126510


namespace certain_number_eq_14_l126_126638

theorem certain_number_eq_14 (x y : ℤ) (h1 : 4 * x + y = 34) (h2 : y^2 = 4) : 2 * x - y = 14 :=
by
  sorry

end certain_number_eq_14_l126_126638


namespace calculate_final_amount_l126_126733

def calculate_percentage (percentage : ℝ) (amount : ℝ) : ℝ :=
  percentage * amount

theorem calculate_final_amount :
  let A := 3000
  let B := 0.20
  let C := 0.35
  let D := 0.05
  D * (C * (B * A)) = 10.50 := by
    sorry

end calculate_final_amount_l126_126733


namespace ceil_square_count_ceil_x_eq_15_l126_126044

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l126_126044


namespace length_of_crease_correct_l126_126422

noncomputable def length_of_crease (theta : ℝ) : ℝ := Real.sqrt (40 + 24 * Real.cos theta)

theorem length_of_crease_correct (theta : ℝ) : 
  length_of_crease theta = Real.sqrt (40 + 24 * Real.cos theta) := 
by 
  sorry

end length_of_crease_correct_l126_126422


namespace min_value_expression_l126_126740

open Real

/-- The minimum value of (14 - x) * (8 - x) * (14 + x) * (8 + x) is -4356. -/
theorem min_value_expression (x : ℝ) : ∃ (a : ℝ), a = (14 - x) * (8 - x) * (14 + x) * (8 + x) ∧ a ≥ -4356 :=
by
  use -4356
  sorry

end min_value_expression_l126_126740


namespace max_value_of_sum_l126_126823

theorem max_value_of_sum (x y z : ℝ) (h : x^2 + 4 * y^2 + 9 * z^2 = 3) : x + 2 * y + 3 * z ≤ 3 :=
sorry

end max_value_of_sum_l126_126823


namespace sector_area_l126_126054

theorem sector_area (r : ℝ) (α : ℝ) (h1 : 2 * r + α * r = 16) (h2 : α = 2) :
  1 / 2 * α * r^2 = 16 :=
by
  sorry

end sector_area_l126_126054


namespace map_distance_l126_126395

noncomputable def map_scale_distance (actual_distance_km : ℕ) (scale : ℕ) : ℕ :=
  let actual_distance_cm := actual_distance_km * 100000;  -- conversion from kilometers to centimeters
  actual_distance_cm / scale

theorem map_distance (d_km : ℕ) (scale : ℕ) (h1 : d_km = 500) (h2 : scale = 8000000) :
  map_scale_distance d_km scale = 625 :=
by
  rw [h1, h2]
  dsimp [map_scale_distance]
  norm_num
  sorry

end map_distance_l126_126395


namespace Set_card_le_two_l126_126200

noncomputable def Satisfies_conditions (S : Set ℕ) : Prop :=
∀ a b ∈ S, a < b → (b - a) ∣ Nat.lcm a b ∧ (Nat.lcm a b) / (b - a) ∈ S

theorem Set_card_le_two (S : Set ℕ) (h: Satisfies_conditions S) : S.toFinset.card ≤ 2 :=
sorry

end Set_card_le_two_l126_126200


namespace triangle_angle_not_less_than_60_l126_126699

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end triangle_angle_not_less_than_60_l126_126699


namespace equal_play_time_for_students_l126_126218

theorem equal_play_time_for_students 
  (total_students : ℕ) 
  (start_time end_time : ℕ) 
  (tables : ℕ) 
  (playing_students refereeing_students : ℕ) 
  (time_played : ℕ) :
  total_students = 6 →
  start_time = 8 * 60 →
  end_time = 11 * 60 + 30 →
  tables = 2 →
  playing_students = 4 →
  refereeing_students = 2 →
  time_played = (end_time - start_time) * tables / (total_students / refereeing_students) →
  time_played = 140 :=
by
  sorry

end equal_play_time_for_students_l126_126218


namespace smallest_n_l126_126322

theorem smallest_n (n : ℕ) (h₁ : n > 2016) (h₂ : n % 4 = 0) : 
  ¬(1^n + 2^n + 3^n + 4^n) % 10 = 0 → n = 2020 :=
by
  sorry

end smallest_n_l126_126322


namespace distinct_prime_factors_90_l126_126779

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l126_126779


namespace johns_average_speed_l126_126993

-- Definitions of conditions
def total_time_hours : ℝ := 6.5
def total_distance_miles : ℝ := 255

-- Stating the problem to be proven
theorem johns_average_speed :
  (total_distance_miles / total_time_hours) = 39.23 := 
sorry

end johns_average_speed_l126_126993


namespace printing_time_l126_126139

-- Definitions based on the problem conditions
def printer_rate : ℕ := 25 -- Pages per minute
def total_pages : ℕ := 325 -- Total number of pages to be printed

-- Statement of the problem rewritten as a Lean 4 statement
theorem printing_time : total_pages / printer_rate = 13 := by
  sorry

end printing_time_l126_126139


namespace jerry_total_cost_l126_126119

-- Definition of the costs and quantities
def cost_color : ℕ := 32
def cost_bw : ℕ := 27
def num_color : ℕ := 3
def num_bw : ℕ := 1

-- Definition of the total cost
def total_cost : ℕ := (cost_color * num_color) + (cost_bw * num_bw)

-- The theorem that needs to be proved
theorem jerry_total_cost : total_cost = 123 :=
by
  sorry

end jerry_total_cost_l126_126119


namespace Pradeep_marks_l126_126086

variable (T : ℕ) (P : ℕ) (F : ℕ)

def passing_marks := P * T / 100

theorem Pradeep_marks (hT : T = 925) (hP : P = 20) (hF : F = 25) :
  (passing_marks P T) - F = 160 :=
by
  sorry

end Pradeep_marks_l126_126086


namespace range_of_a_minus_abs_b_l126_126486

theorem range_of_a_minus_abs_b {a b : ℝ} (h1 : 1 < a ∧ a < 3) (h2 : -4 < b ∧ b < 2) :
  -3 < a - |b| ∧ a - |b| < 3 :=
by
  sorry

end range_of_a_minus_abs_b_l126_126486


namespace value_of_expression_l126_126330

theorem value_of_expression (m n : ℤ) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) :
  m^2 + m * n + n^2 = 7 ∨ m^2 + m * n + n^2 = 19 := by
  sorry

end value_of_expression_l126_126330


namespace total_price_of_houses_l126_126992

theorem total_price_of_houses (price_first price_second total_price : ℝ)
    (h1 : price_first = 200000)
    (h2 : price_second = 2 * price_first)
    (h3 : total_price = price_first + price_second) :
  total_price = 600000 := by
  sorry

end total_price_of_houses_l126_126992


namespace power_factor_200_l126_126075

theorem power_factor_200 :
  (let a := 3 in let b := 2 in (1 / 3)^(b - a) = 3) :=
by
  -- assume a and b definitions
  let a := 3
  let b := 2
  -- main statement
  show (1 / 3) ^ (b - a) = 3
  -- we skip the proof
  sorry

end power_factor_200_l126_126075


namespace birch_trees_probability_l126_126927

theorem birch_trees_probability :
  let non_birch_trees := 4 + 5,
      total_trees := 4 + 5 + 3,
      birch_trees := 3,
      slots := non_birch_trees + 1,
      successful_arrangements := Nat.choose slots birch_trees,
      total_arrangements := Nat.choose total_trees birch_trees,
      probability := successful_arrangements / total_arrangements
  in Rat.num_den (success_probability : ℚ) = (6, 11) ∧ 6 + 11 = 17 := 
by
  sorry

end birch_trees_probability_l126_126927


namespace school_sample_proof_l126_126863

open Probability

noncomputable def classes_in_schools : ℕ → ℕ
| 0 => 12  -- School A
| 1 => 6   -- School B
| 2 => 18  -- School C
| _ => 0

def total_classes : ℕ := classes_in_schools 0 + classes_in_schools 1 + classes_in_schools 2

def sampled_classes : ℕ := 6

def sample_proportion (school_index : ℕ) : ℚ :=
  (classes_in_schools school_index : ℚ) / (total_classes)

noncomputable def number_of_sampled_classes (school_index : ℕ) : ℕ :=
  (sampled_classes * sample_proportion school_index).to_nat

def random_selection : List ℕ := [number_of_sampled_classes 0, number_of_sampled_classes 1, number_of_sampled_classes 2]

def all_possible_outcomes : List (ℕ × ℕ) :=
  [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2), 
   (2,3), (3,1), (3,2), (2,3), (3,3), (1,3), (1,2)]

def event_D : List (ℕ × ℕ) :=
  [(0,0), (0,1), (0,2), (0,3), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2)]

noncomputable def probability_D : ℚ :=
  event_D.length / all_possible_outcomes.length

theorem school_sample_proof :
  number_of_sampled_classes 0 = 2 ∧ number_of_sampled_classes 1 = 1 ∧ number_of_sampled_classes 2 = 3 ∧
  probability_D = 3 / 5 :=
by
  sorry

end school_sample_proof_l126_126863


namespace average_of_first_15_even_numbers_is_16_l126_126413

-- Define the sum of the first 15 even numbers
def sum_first_15_even_numbers : ℕ :=
  2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20 + 22 + 24 + 26 + 28 + 30

-- Define the average of the first 15 even numbers
def average_of_first_15_even_numbers : ℕ :=
  sum_first_15_even_numbers / 15

-- Prove that the average is equal to 16
theorem average_of_first_15_even_numbers_is_16 : average_of_first_15_even_numbers = 16 :=
by
  -- Sorry placeholder for the proof
  sorry

end average_of_first_15_even_numbers_is_16_l126_126413


namespace find_principal_l126_126320

theorem find_principal
  (R : ℝ) (hR : R = 0.05)
  (I : ℝ) (hI : I = 0.02)
  (A : ℝ) (hA : A = 1120)
  (n : ℕ) (hn : n = 6)
  (R' : ℝ) (hR' : R' = ((1 + R) / (1 + I)) - 1) :
  P = 938.14 :=
by
  have compound_interest_formula := A / (1 + R')^n
  sorry

end find_principal_l126_126320


namespace isosceles_triangle_perimeter_l126_126060

-- An auxiliary definition to specify that the triangle is isosceles
def is_isosceles (a b c : ℕ) :=
  a = b ∨ b = c ∨ c = a

-- The main theorem statement
theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : is_isosceles a b 6): a + b + 6 = 15 :=
by
  -- the proof would go here
  sorry

end isosceles_triangle_perimeter_l126_126060


namespace number_of_arrangements_l126_126225

theorem number_of_arrangements (teams : Finset ℕ) (sites : Finset ℕ) :
  (∀ team, team ∈ teams → (team ∈ sites)) ∧ ((Finset.card sites = 3) ∧ (Finset.card teams = 6)) ∧ 
  (∃ (a b c : ℕ), a + b + c = 6 ∧ a >= 2 ∧ b >= 1 ∧ c >= 1) →
  ∃ (n : ℕ), n = 360 :=
sorry

end number_of_arrangements_l126_126225


namespace sin_180_degrees_l126_126310

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l126_126310


namespace inscribed_circle_radius_is_correct_l126_126262

noncomputable def radius_of_inscribed_circle (base height : ℝ) : ℝ := sorry

theorem inscribed_circle_radius_is_correct :
  radius_of_inscribed_circle 20 24 = 120 / 13 := sorry

end inscribed_circle_radius_is_correct_l126_126262


namespace inverse_proportion_point_l126_126358

theorem inverse_proportion_point (k : ℝ) (x1 y1 x2 y2 : ℝ)
  (h1 : y1 = k / x1) 
  (h2 : x1 = -2) 
  (h3 : y1 = 3)
  (h4 : x2 = 2) :
  y2 = -3 := 
by
  -- proof will be provided here
  sorry

end inverse_proportion_point_l126_126358


namespace angle_B_plus_angle_D_105_l126_126042

theorem angle_B_plus_angle_D_105
(angle_A : ℝ) (angle_AFG angle_AGF : ℝ)
(h1 : angle_A = 30)
(h2 : angle_AFG = angle_AGF)
: angle_B + angle_D = 105 := sorry

end angle_B_plus_angle_D_105_l126_126042


namespace total_cartridge_cost_l126_126118

theorem total_cartridge_cost:
  ∀ (bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity : ℕ),
  bw_cartridge_cost = 27 →
  color_cartridge_cost = 32 →
  bw_quantity = 1 →
  color_quantity = 3 →
  bw_quantity * bw_cartridge_cost + color_quantity * color_cartridge_cost = 123 :=
begin
  intros bw_cartridge_cost color_cartridge_cost bw_quantity color_quantity,
  intros h_bw_cost h_color_cost h_bw_qty h_color_qty,
  rw [h_bw_cost, h_color_cost, h_bw_qty, h_color_qty],
  norm_num,
end

end total_cartridge_cost_l126_126118


namespace sin_180_is_zero_l126_126297

noncomputable def sin_180_eq_zero : Prop :=
  let point_zero_deg := (1, 0)
  let point_180_deg := (-1, 0)
  let y_coord_of_180_deg := 0
  sin 180 = y_coord_of_180_deg

theorem sin_180_is_zero : sin_180_eq_zero :=
  sorry

end sin_180_is_zero_l126_126297


namespace ceil_x_squared_values_count_l126_126050

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l126_126050


namespace percent_decrease_in_hours_l126_126565

variable {W H : ℝ} (W_nonzero : W ≠ 0) (H_nonzero : H ≠ 0)

theorem percent_decrease_in_hours
  (wage_increase : W' = 1.25 * W)
  (income_unchanged : W * H = W' * H')
  : (H' = 0.8 * H) → H' = H * (1 - 0.2) := by
  sorry

end percent_decrease_in_hours_l126_126565


namespace sin_180_degrees_l126_126309

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l126_126309


namespace product_of_prs_l126_126679

theorem product_of_prs
  (p r s : ℕ)
  (H1 : 4 ^ p + 4 ^ 3 = 272)
  (H2 : 3 ^ r + 27 = 54)
  (H3 : 2 ^ (s + 2) + 10 = 42) : 
  p * r * s = 27 :=
sorry

end product_of_prs_l126_126679


namespace prob_one_head_one_tail_l126_126398

theorem prob_one_head_one_tail (h1 h2 : bool) (H : h1 = tt ∨ h1 = ff) (T : h2 = tt ∨ h2 = ff):
  (h1 = tt ∧ h2 = ff) ∨ (h1 = ff ∧ h2 = tt) →
  real := 1 / 2 :=
by
  sorry

end prob_one_head_one_tail_l126_126398


namespace greatest_difference_in_baskets_l126_126857

theorem greatest_difference_in_baskets :
  let A_red := 4
  let A_yellow := 2
  let B_green := 6
  let B_yellow := 1
  let C_white := 3
  let C_yellow := 9
  max (abs (A_red - A_yellow)) (max (abs (B_green - B_yellow)) (abs (C_white - C_yellow))) = 6 :=
by
  sorry

end greatest_difference_in_baskets_l126_126857


namespace sum_of_gcd_and_lcm_is_28_l126_126891

def gcd (x y : ℕ) : ℕ := x.gcd y
def lcm (x y : ℕ) : ℕ := x.lcm y

theorem sum_of_gcd_and_lcm_is_28 :
  gcd 8 12 + lcm 8 12 = 28 :=
by 
  -- Proof to be completed
  sorry

end sum_of_gcd_and_lcm_is_28_l126_126891


namespace burgers_per_day_l126_126158

def calories_per_burger : ℝ := 20
def total_calories_after_two_days : ℝ := 120

theorem burgers_per_day :
  total_calories_after_two_days / (2 * calories_per_burger) = 3 := 
by
  sorry

end burgers_per_day_l126_126158


namespace remainder_of_n_mod_5_l126_126618

theorem remainder_of_n_mod_5
  (n : Nat)
  (h1 : n^2 ≡ 4 [MOD 5])
  (h2 : n^3 ≡ 2 [MOD 5]) :
  n ≡ 3 [MOD 5] :=
sorry

end remainder_of_n_mod_5_l126_126618


namespace y_satisfies_quadratic_l126_126179

theorem y_satisfies_quadratic (x y : ℝ) 
  (h1 : 2 * x^2 + 6 * x + 5 * y + 1 = 0)
  (h2 : 2 * x + y + 3 = 0) : y^2 + 10 * y - 7 = 0 := 
sorry

end y_satisfies_quadratic_l126_126179


namespace gasoline_tank_capacity_l126_126717

theorem gasoline_tank_capacity
  (y : ℝ)
  (h_initial: y * (5 / 6) - y * (1 / 3) = 20) :
  y = 40 :=
sorry

end gasoline_tank_capacity_l126_126717


namespace a_n_formula_l126_126466

open Nat

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 0 then 0 
  else n * (n + 1) / 2

theorem a_n_formula (n : ℕ) (h : n > 0) 
  (S_n : ℕ → ℕ)
  (hS : ∀ n, S_n n = (n + 2) / 3 * a_n n) 
  : a_n n = n * (n + 1) / 2 := sorry

end a_n_formula_l126_126466


namespace outstanding_student_awards_l126_126534

theorem outstanding_student_awards :
  ∃ n : ℕ, 
  (n = Nat.choose 9 7) ∧ 
  (∀ (awards : ℕ) (classes : ℕ), awards = 10 → classes = 8 → n = 36) := 
by
  sorry

end outstanding_student_awards_l126_126534


namespace value_of_y_l126_126635

theorem value_of_y (y : ℕ) (hy : (1 / 8) * 2^36 = 8^y) : y = 11 :=
by
  sorry

end value_of_y_l126_126635


namespace sum_of_GCF_and_LCM_l126_126885

-- Definitions
def GCF (m n : ℕ) : ℕ := Nat.gcd m n
def LCM (m n : ℕ) : ℕ := Nat.lcm m n

-- Conditions
def m := 8
def n := 12

-- Statement of the problem
theorem sum_of_GCF_and_LCM : GCF m n + LCM m n = 28 := by
  sorry

end sum_of_GCF_and_LCM_l126_126885


namespace complete_graph_color_5_complete_graph_no_color_4_l126_126629

noncomputable theory

-- Definitions and theorem statements

open Finset

-- Complete Graph definition
def complete_graph (n : ℕ) : SimpleGraph (Fin n) := {
  adj := λ x y, x ≠ y,
  symm := by finish,
  loopless := by finish
}

-- Theorem 1: Coloring with 5 colors for any subset of 5 vertices
theorem complete_graph_color_5 (G : SimpleGraph (Fin 10)) (H : G = complete_graph 10) :
  ∃ f : G.edge → Fin 5, ∀ (S : Finset (Fin 10)), S.card = 5 → (S.pairwise_disjoint f) :=
sorry

-- Theorem 2: Impossibility of coloring with 4 colors for any subset of 4 vertices
theorem complete_graph_no_color_4 (G : SimpleGraph (Fin 10)) (H : G = complete_graph 10) :
  ¬ ∃ f : G.edge → Fin 4, ∀ (S : Finset (Fin 10)), S.card = 4 → (S.pairwise_disjoint f) :=
sorry


end complete_graph_color_5_complete_graph_no_color_4_l126_126629


namespace remainder_div_1234567_256_l126_126448

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l126_126448


namespace sum_first_100_odd_l126_126949

theorem sum_first_100_odd :
  (Finset.sum (Finset.range 100) (λ x => 2 * (x + 1) - 1)) = 10000 := by
  sorry

end sum_first_100_odd_l126_126949


namespace quadratic_unique_solution_l126_126394

theorem quadratic_unique_solution (a c : ℝ) :
  (∀ x : ℝ, a * x^2 + 36 * x + c = 0 ↔ x = (-36) / (2*a))  -- The quadratic equation has exactly one solution
  → a + c = 37  -- Given condition
  → a < c      -- Given condition
  → (a, c) = ( (37 - Real.sqrt 73) / 2, (37 + Real.sqrt 73) / 2 ) :=  -- Correct answer
by
  sorry

end quadratic_unique_solution_l126_126394


namespace distinct_prime_factors_90_l126_126778

def prime_factors (n : Nat) : List Nat :=
  sorry -- Implementation of prime factorization is skipped

noncomputable def num_distinct_prime_factors (n : Nat) : Nat :=
  (List.toFinset (prime_factors n)).card

theorem distinct_prime_factors_90 : num_distinct_prime_factors 90 = 3 :=
  by sorry

end distinct_prime_factors_90_l126_126778


namespace equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l126_126745
open BigOperators

-- First, we define the three equations and their constraints
def equation1_solution (k : ℤ) : ℤ × ℤ := (2 - 5 * k, -1 + 3 * k)
def equation2_solution (k : ℤ) : ℤ × ℤ := (8 - 5 * k, -4 + 3 * k)
def equation3_solution (k : ℤ) : ℤ × ℤ := (16 - 39 * k, -25 + 61 * k)

-- Define the proof that the supposed solutions hold for each equation
theorem equation1_solution_valid (k : ℤ) : 3 * (equation1_solution k).1 + 5 * (equation1_solution k).2 = 1 :=
by
  -- Proof steps would go here
  sorry

theorem equation2_solution_valid (k : ℤ) : 3 * (equation2_solution k).1 + 5 * (equation2_solution k).2 = 4 :=
by
  -- Proof steps would go here
  sorry

theorem equation3_solution_valid (k : ℤ) : 183 * (equation3_solution k).1 + 117 * (equation3_solution k).2 = 3 :=
by
  -- Proof steps would go here
  sorry

end equation1_solution_valid_equation2_solution_valid_equation3_solution_valid_l126_126745


namespace storks_more_than_birds_l126_126259

def initial_birds := 2
def additional_birds := 3
def total_birds := initial_birds + additional_birds
def storks := 6
def difference := storks - total_birds

theorem storks_more_than_birds : difference = 1 :=
by
  sorry

end storks_more_than_birds_l126_126259


namespace sin_180_degree_l126_126303

theorem sin_180_degree : Real.sin (Real.pi) = 0 := by sorry

end sin_180_degree_l126_126303


namespace parabola_x_intercepts_incorrect_l126_126464

-- Define the given quadratic function
noncomputable def f (x : ℝ) : ℝ := -1 / 2 * (x - 1)^2 + 2

-- The Lean statement for the problem
theorem parabola_x_intercepts_incorrect :
  ¬ ((f 3 = 0) ∧ (f (-3) = 0)) :=
by
  sorry

end parabola_x_intercepts_incorrect_l126_126464


namespace limit_sequence_is_5_l126_126946

noncomputable def sequence (n : ℕ) : ℝ :=
  (real.sqrt (3 * n - 1) - real.cbrt (125 * n ^ 3 + n)) / (real.rpow n (1 / 5) - n)

theorem limit_sequence_is_5 : 
  filter.tendsto (sequence) at_top (nhds 5) :=
sorry

end limit_sequence_is_5_l126_126946


namespace quadratic_no_real_roots_l126_126237

-- Define the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant of a quadratic equation
def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Conditions of the problem
def a : ℝ := 3
def b : ℝ := -6
def c : ℝ := 4

-- The proof statement
theorem quadratic_no_real_roots : discriminant a b c < 0 :=
by
  -- Calculate the discriminant to show it's negative
  let Δ := discriminant a b c
  show Δ < 0
  sorry

end quadratic_no_real_roots_l126_126237


namespace inverse_proportionality_ratio_l126_126520

variable {x y k x1 x2 y1 y2 : ℝ}

theorem inverse_proportionality_ratio
  (h1 : x * y = k)
  (hx1 : x1 ≠ 0)
  (hx2 : x2 ≠ 0)
  (hy1 : y1 ≠ 0)
  (hy2 : y2 ≠ 0)
  (hx_ratio : x1 / x2 = 3 / 4)
  (hxy1 : x1 * y1 = k)
  (hxy2 : x2 * y2 = k) :
  y1 / y2 = 4 / 3 := by
  sorry

end inverse_proportionality_ratio_l126_126520


namespace greatest_difference_in_baskets_l126_126858

theorem greatest_difference_in_baskets :
  let A_red := 4
  let A_yellow := 2
  let B_green := 6
  let B_yellow := 1
  let C_white := 3
  let C_yellow := 9
  max (abs (A_red - A_yellow)) (max (abs (B_green - B_yellow)) (abs (C_white - C_yellow))) = 6 :=
by
  sorry

end greatest_difference_in_baskets_l126_126858


namespace bottle_caps_proof_l126_126453

def bottle_caps_difference (found thrown : ℕ) := found - thrown

theorem bottle_caps_proof : bottle_caps_difference 50 6 = 44 := by
  sorry

end bottle_caps_proof_l126_126453


namespace lcm_36_225_l126_126319

theorem lcm_36_225 : Nat.lcm 36 225 = 900 := by
  -- Defining the factorizations as given
  let fact_36 : 36 = 2^2 * 3^2 := by rfl
  let fact_225 : 225 = 3^2 * 5^2 := by rfl

  -- Indicating what LCM we need to prove
  show Nat.lcm 36 225 = 900

  -- Proof (skipped)
  sorry

end lcm_36_225_l126_126319


namespace point_in_second_quadrant_l126_126528

theorem point_in_second_quadrant (a : ℝ) : 
  ∃ q : ℕ, q = 2 ∧ (-1, a^2 + 1).1 < 0 ∧ 0 < (-1, a^2 + 1).2 :=
by
  sorry

end point_in_second_quadrant_l126_126528


namespace country_math_l126_126647

theorem country_math (h : (1 / 3 : ℝ) * 4 = 6) : 
  ∃ x : ℝ, (1 / 6 : ℝ) * x = 15 ∧ x = 405 :=
by
  sorry

end country_math_l126_126647


namespace greatest_y_value_l126_126840

theorem greatest_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 :=
sorry

end greatest_y_value_l126_126840


namespace muffins_count_l126_126600

-- Lean 4 Statement
theorem muffins_count (doughnuts muffins : ℕ) (ratio_doughnuts_muffins : ℕ → ℕ → Prop)
  (h_ratio : ratio_doughnuts_muffins 5 1) (h_doughnuts : doughnuts = 50) :
  muffins = 10 :=
by
  sorry

end muffins_count_l126_126600


namespace total_shaded_area_l126_126581

-- Problem condition definitions
def side_length_carpet := 12
def ratio_large_square : ℕ := 4
def ratio_small_square : ℕ := 4

-- Problem statement
theorem total_shaded_area : 
  ∃ S T : ℚ, 
    12 / S = ratio_large_square ∧ S / T = ratio_small_square ∧ 
    (12 * (T * T)) + (S * S) = 15.75 := 
sorry

end total_shaded_area_l126_126581


namespace smallest_b_for_composite_l126_126007

theorem smallest_b_for_composite (x : ℤ) : 
  ∃ b : ℕ, b > 0 ∧ Even b ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^2)) ∧ b = 16 := 
by 
  sorry

end smallest_b_for_composite_l126_126007


namespace showUpPeopleFirstDay_l126_126736

def cansFood := 2000
def people1stDay (cansTaken_1stDay : ℕ) := cansFood - 1500 = cansTaken_1stDay
def peopleSnapped_1stDay := 500

theorem showUpPeopleFirstDay :
  (people1stDay peopleSnapped_1stDay) → (peopleSnapped_1stDay / 1) = 500 := 
by 
  sorry

end showUpPeopleFirstDay_l126_126736


namespace max_value_m_l126_126530

theorem max_value_m (m n : ℕ) (h : 8 * m + 9 * n = m * n + 6) : m ≤ 75 := 
sorry

end max_value_m_l126_126530


namespace max_area_of_rectangle_l126_126622

theorem max_area_of_rectangle (x y : ℝ) (h : 2 * x + 2 * y = 36) : (x * y) ≤ 81 :=
sorry

end max_area_of_rectangle_l126_126622


namespace find_f_of_3_l126_126173

theorem find_f_of_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (2 * x + 1) = x^2 - 2 * x) : f 3 = -1 :=
by 
  sorry

end find_f_of_3_l126_126173


namespace cylinder_base_radius_l126_126190

theorem cylinder_base_radius (a : ℝ) (h_a_pos : 0 < a) :
  ∃ (R : ℝ), R = 7 * a * Real.sqrt 3 / 24 := 
    sorry

end cylinder_base_radius_l126_126190


namespace sin_180_degrees_l126_126308

theorem sin_180_degrees : Real.sin (180 * Real.pi / 180) = 0 := 
by
  sorry

end sin_180_degrees_l126_126308


namespace length_of_AD_l126_126532

theorem length_of_AD 
  (A B C D : Type) 
  (vertex_angle_equal: ∀ {a b c d : Type}, a = A →
    ∀ (AB AC AD : ℝ), (AB = 24) → (AC = 54) → (AD = 36)) 
  (right_triangles : ∀ {a b : Type}, a = A → ∀ {AB AC : ℝ}, (AB > 0) → (AC > 0) → (AB ^ 2 + AC ^ 2 = AD ^ 2)) :
  ∃ (AD : ℝ), AD = 36 :=
by
  sorry

end length_of_AD_l126_126532


namespace find_point_B_coordinates_l126_126624

theorem find_point_B_coordinates : 
  ∃ B : ℝ × ℝ, 
    (∀ A C B : ℝ × ℝ, A = (2, 3) ∧ C = (0, 1) ∧ 
    (B.1 - A.1, B.2 - A.2) = (-2) • (C.1 - B.1, C.2 - B.2)) → B = (-2, -1) :=
by 
  sorry

end find_point_B_coordinates_l126_126624


namespace total_points_earned_l126_126561

def defeated_enemies := 15
def points_per_enemy := 12
def level_completion_points := 20
def special_challenges_completed := 5
def points_per_special_challenge := 10

theorem total_points_earned :
  defeated_enemies * points_per_enemy
  + level_completion_points
  + special_challenges_completed * points_per_special_challenge = 250 :=
by
  -- The proof would be developed here.
  sorry

end total_points_earned_l126_126561


namespace ali_seashells_final_count_l126_126276

theorem ali_seashells_final_count :
  385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25))) 
  - (1 / 4) * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - 0.10 * ((385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)) 
  - (2 / 3) * (385.5 - 45.75 - 34.25 + 0.20 * (385.5 - 45.75 - 34.25)))) = 82.485 :=
sorry

end ali_seashells_final_count_l126_126276


namespace billy_tickets_l126_126283

theorem billy_tickets (ferris_wheel_rides bumper_car_rides rides_per_ride total_tickets : ℕ) 
  (h1 : ferris_wheel_rides = 7)
  (h2 : bumper_car_rides = 3)
  (h3 : rides_per_ride = 5)
  (h4 : total_tickets = (ferris_wheel_rides + bumper_car_rides) * rides_per_ride) :
  total_tickets = 50 := 
by 
  sorry

end billy_tickets_l126_126283


namespace find_pairs_l126_126014

def isDivisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def satisfiesConditions (a b : ℕ) : Prop :=
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  isPrime (a + 6 * b + 2)) ∨
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  ¬ isPrime (a + 6 * b + 2))

theorem find_pairs (a b : ℕ) :
  (a = 5 ∧ b = 1) ∨ 
  (a = 17 ∧ b = 7) → 
  satisfiesConditions a b :=
by
  -- Proof to be completed
  sorry

end find_pairs_l126_126014


namespace shape_of_triangle_l126_126188

-- Define the problem conditions
variable {a b : ℝ}
variable {A B C : ℝ}
variable (triangle_condition : (a^2 / b^2 = tan A / tan B))

-- Define the theorem to be proved
theorem shape_of_triangle ABC
  (h : triangle_condition):
  (A = B ∨ A + B = π / 2) :=
sorry

end shape_of_triangle_l126_126188


namespace car_gas_cost_l126_126735

def car_mpg_city : ℝ := 30
def car_mpg_highway : ℝ := 40
def city_distance_one_way : ℝ := 60
def highway_distance_one_way : ℝ := 200
def gas_cost_per_gallon : ℝ := 3
def total_gas_cost : ℝ := 42

theorem car_gas_cost :
  (city_distance_one_way / car_mpg_city * 2 + highway_distance_one_way / car_mpg_highway * 2) * gas_cost_per_gallon = total_gas_cost := 
  sorry

end car_gas_cost_l126_126735


namespace distance_fall_l126_126972

-- Given conditions as definitions
def velocity (g : ℝ) (t : ℝ) := g * t

-- The theorem stating the relationship between time t0 and distance S
theorem distance_fall (g : ℝ) (t0 : ℝ) : 
  (∫ t in (0 : ℝ)..t0, velocity g t) = (1/2) * g * t0^2 :=
by 
  sorry

end distance_fall_l126_126972


namespace angle_B_magnitude_value_of_b_l126_126808
open Real

theorem angle_B_magnitude (B : ℝ) (h : 2 * sin B - 2 * sin B ^ 2 - cos (2 * B) = sqrt 3 - 1) :
  B = π / 3 ∨ B = 2 * π / 3 := sorry

theorem value_of_b (a B S : ℝ) (hB : B = π / 3) (ha : a = 6) (hS : S = 6 * sqrt 3) :
  let c := 4
  let b := 2 * sqrt 7
  let half_angle_B := 1 / 2 * a * c * sin B
  half_angle_B = S :=
by
  sorry

end angle_B_magnitude_value_of_b_l126_126808


namespace min_value_f_l126_126106

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem min_value_f : 
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 12), f(x) ≥ 1) ∧ (∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 12), f(x) = 1) :=
by
  sorry

end min_value_f_l126_126106


namespace find_x_l126_126696

theorem find_x :
  ∃ x : ℕ, x % 6 = 5 ∧ x % 7 = 6 ∧ x % 8 = 7 ∧ (∀ y : ℕ, y % 6 = 5 ∧ y % 7 = 6 ∧ y % 8 = 7 → y ≥ x) :=
sorry

end find_x_l126_126696


namespace geometric_sequence_solution_l126_126196

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) (a1 : ℝ) :=
  ∀ n, a n = a1 * q ^ (n - 1)

theorem geometric_sequence_solution {a : ℕ → ℝ} {q a1 : ℝ}
  (h1 : geometric_sequence a q a1)
  (h2 : a 3 + a 5 = 20)
  (h3 : a 4 = 8) :
  a 2 + a 6 = 34 := by
  sorry

end geometric_sequence_solution_l126_126196


namespace workshop_processing_equation_l126_126267

noncomputable def process_equation (x : ℝ) : Prop :=
  (4000 / x - 4200 / (1.5 * x) = 3)

theorem workshop_processing_equation (x : ℝ) (hx : x > 0) :
  process_equation x :=
by
  sorry

end workshop_processing_equation_l126_126267


namespace shaded_triangle_area_l126_126390

/--
The large equilateral triangle shown consists of 36 smaller equilateral triangles.
Each of the smaller equilateral triangles has an area of 10 cm². 
The area of the shaded triangle is K cm².
Prove that K = 110 cm².
-/
theorem shaded_triangle_area 
  (n : ℕ) (area_small : ℕ) (area_total : ℕ) (K : ℕ)
  (H1 : n = 36)
  (H2 : area_small = 10)
  (H3 : area_total = n * area_small)
  (H4 : K = 110)
: K = 110 :=
by
  -- Adding 'sorry' indicating missing proof steps.
  sorry

end shaded_triangle_area_l126_126390


namespace aubree_total_animals_l126_126943

noncomputable def total_animals_seen : Nat :=
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks

theorem aubree_total_animals :
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10
  morning_total + new_beavers + new_chipmunks = 130 := by
  -- Define all constants and conditions
  let initial_beavers := 20
  let initial_chipmunks := 40
  let morning_total := initial_beavers + initial_chipmunks
  let new_beavers := initial_beavers * 2
  let new_chipmunks := initial_chipmunks - 10

  -- State the equation
  show morning_total + new_beavers + new_chipmunks = 130 from sorry

end aubree_total_animals_l126_126943


namespace find_second_divisor_l126_126923

theorem find_second_divisor :
  ∃ x : ℕ, 377 / 13 / x * (1/4 : ℚ) / 2 = 0.125 ∧ x = 29 :=
by
  use 29
  -- Proof steps would go here
  sorry

end find_second_divisor_l126_126923


namespace quadratic_roots_range_l126_126806

theorem quadratic_roots_range (k : ℝ) : 
  (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
  (k * x₁^2 - 4 * x₁ + 1 = 0) ∧ (k * x₂^2 - 4 * x₂ + 1 = 0)) 
  ↔ (k < 4 ∧ k ≠ 0) := 
by
  sorry

end quadratic_roots_range_l126_126806


namespace polynomial_factorization_l126_126847

theorem polynomial_factorization (a b : ℤ) (h : (x^2 + x - 6) = (x + a) * (x + b)) :
  (a + b)^2023 = 1 :=
sorry

end polynomial_factorization_l126_126847


namespace wages_problem_l126_126427

variable {S W_y W_x : ℝ}
variable {D_x : ℝ}

theorem wages_problem
  (h1 : S = 45 * W_y)
  (h2 : S = 20 * (W_x + W_y))
  (h3 : S = D_x * W_x) :
  D_x = 36 :=
sorry

end wages_problem_l126_126427


namespace car_actual_speed_is_40_l126_126917

variable (v : ℝ) -- actual speed (we will prove it is 40 km/h)

-- Conditions
variable (hyp_speed : ℝ := v + 20) -- hypothetical speed
variable (distance : ℝ := 60) -- distance traveled
variable (time_difference : ℝ := 0.5) -- time difference in hours

-- Define the equation derived from the given conditions:
def speed_equation : Prop :=
  (distance / v) - (distance / hyp_speed) = time_difference

-- The theorem to prove:
theorem car_actual_speed_is_40 : speed_equation v → v = 40 :=
by
  sorry

end car_actual_speed_is_40_l126_126917


namespace smaller_angle_at_3_15_l126_126040

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l126_126040


namespace males_listen_l126_126933

theorem males_listen (total_listen : ℕ) (females_listen : ℕ) (known_total_listen : total_listen = 160)
  (known_females_listen : females_listen = 75) : (total_listen - females_listen) = 85 :=
by 
  sorry

end males_listen_l126_126933


namespace estimation_problems_l126_126725

noncomputable def average_root_cross_sectional_area (x : list ℝ) : ℝ :=
  (list.sum x) / (list.length x)

noncomputable def average_volume (y : list ℝ) : ℝ :=
  (list.sum y) / (list.length y)

noncomputable def sample_correlation_coefficient (x y : list ℝ) : ℝ :=
  let n := list.length x
      avg_x := average_root_cross_sectional_area x
      avg_y := average_volume y
      sum_xy := (list.zip_with (*) x y).sum
      sum_x2 := (x.map (λ xi, xi * xi)).sum
      sum_y2 := (y.map (λ yi, yi * yi)).sum
  in (sum_xy - n * avg_x * avg_y) / (real.sqrt ((sum_x2 - n * avg_x^2) * (sum_y2 - n * avg_y^2)))

theorem estimation_problems :
  let x := [0.04, 0.06, 0.04, 0.08, 0.08, 0.05, 0.05, 0.07, 0.07, 0.06]
      y := [0.25, 0.40, 0.22, 0.54, 0.51, 0.34, 0.36, 0.46, 0.42, 0.40]
      X := 186
  in
    average_root_cross_sectional_area x = 0.06 ∧
    average_volume y = 0.39 ∧
    abs (sample_correlation_coefficient x y - 0.97) < 0.01 ∧
    (average_volume y / average_root_cross_sectional_area x) * X = 1209 :=
by
  sorry

end estimation_problems_l126_126725


namespace variances_equal_thirtieth_percentile_y_l126_126027

noncomputable def sample_data_x (i : ℕ) (h : 1 ≤ i ∧ i ≤ 10) : ℝ := 2 * i

def sample_data_y (i : ℕ) (h : 1 ≤ i ∧ i ≤ 10) : ℝ := sample_data_x i h - 20

def mean (data : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range (n + 1), data (i + 1) sorry) / n

def variance (data : ℕ → ℝ) (n : ℕ) : ℝ :=
  let mean_val := mean data n in
  (∑ i in finset.range (n + 1), ((data (i + 1) sorry) - mean_val) ^ 2) / n

def percentile (data : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ := 
  let sorted_data := list.sort (≤) [data (i + 1) sorry | i in finset.range (n + 1)] in
  (sorted_data.nth ((n * p).toNat) + sorted_data.nth ((n * p).toNat + 1)) / 2

theorem variances_equal : variance sample_data_x 10 = variance sample_data_y 10 :=
  sorry

theorem thirtieth_percentile_y : percentile sample_data_y 10 0.3 = -13 :=
  sorry

end variances_equal_thirtieth_percentile_y_l126_126027


namespace vec_v_satisfies_l126_126005

open Matrix

def A := ![![0, 2], ![4, 0]] : Matrix (Fin 2) (Fin 2) ℚ
def v := ![0, 47 / 665] : Fin 2 → ℚ
def I2 := (1 : Matrix (Fin 2) (Fin 2) ℚ)

theorem vec_v_satisfies :
  (A^6 + 2 * A^4 + 3 * A^2 + I2) ⬝ v = ![0, 47] := 
  sorry

end vec_v_satisfies_l126_126005


namespace find_m_repeated_root_l126_126339

theorem find_m_repeated_root (m : ℝ) :
  (∃ x : ℝ, (x - 1) ≠ 0 ∧ (m - 1) - x = 0) → m = 2 :=
by
  sorry

end find_m_repeated_root_l126_126339


namespace remainder_div_1234567_256_l126_126446

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l126_126446


namespace number_of_possible_ceil_values_l126_126047

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l126_126047
