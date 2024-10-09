import Mathlib

namespace nicole_initial_candies_l1889_188905

theorem nicole_initial_candies (x : ℕ) (h1 : x / 3 + 5 + 10 = x) : x = 23 := by
  sorry

end nicole_initial_candies_l1889_188905


namespace total_shaded_area_l1889_188917

theorem total_shaded_area
  (carpet_side : ℝ)
  (large_square_side : ℝ)
  (small_square_side : ℝ)
  (ratio_large : carpet_side / large_square_side = 4)
  (ratio_small : large_square_side / small_square_side = 2) : 
  (1 * large_square_side^2 + 12 * small_square_side^2 = 64) := 
by 
  sorry

end total_shaded_area_l1889_188917


namespace translate_point_A_l1889_188900

theorem translate_point_A :
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  A1 = (3, 0) :=
by
  let A : ℝ × ℝ := (-1, 2)
  let x_translation : ℝ := 4
  let y_translation : ℝ := -2
  let A1 : ℝ × ℝ := (A.1 + x_translation, A.2 + y_translation)
  show A1 = (3, 0)
  sorry

end translate_point_A_l1889_188900


namespace odd_primes_mod_32_l1889_188906

-- Define the set of odd primes less than 2^5
def odd_primes_less_than_32 : List ℕ := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]

-- Define the product of all elements in the list
def N : ℕ := odd_primes_less_than_32.foldl (·*·) 1

-- State the theorem
theorem odd_primes_mod_32 :
  N % 32 = 9 :=
sorry

end odd_primes_mod_32_l1889_188906


namespace subsequent_flights_requirements_l1889_188984

-- Define the initial conditions
def late_flights : ℕ := 1
def on_time_flights : ℕ := 3
def total_initial_flights : ℕ := late_flights + on_time_flights

-- Define the number of subsequent flights needed
def subsequent_flights_needed (x : ℕ) : Prop :=
  let total_flights := total_initial_flights + x
  let on_time_total := on_time_flights + x
  (on_time_total : ℚ) / (total_flights : ℚ) > 0.40

-- State the theorem to prove
theorem subsequent_flights_requirements:
  ∃ x : ℕ, subsequent_flights_needed x := sorry

end subsequent_flights_requirements_l1889_188984


namespace range_of_a_l1889_188992

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + (a + 1) * x + 1 < 0) → (a < -3 ∨ a > 1) :=
by
  sorry

end range_of_a_l1889_188992


namespace Pria_drove_372_miles_l1889_188997

theorem Pria_drove_372_miles (advertisement_mileage : ℕ) (tank_capacity : ℕ) (mileage_difference : ℕ) 
(h1 : advertisement_mileage = 35) 
(h2 : tank_capacity = 12) 
(h3 : mileage_difference = 4) : 
(advertisement_mileage - mileage_difference) * tank_capacity = 372 :=
by sorry

end Pria_drove_372_miles_l1889_188997


namespace no_common_root_l1889_188910

theorem no_common_root (a b c d : ℝ) (ha : 0 < a) (hb : a < b) (hc : b < c) (hd : c < d) :
  ¬ ∃ x : ℝ, (x^2 + b * x + c = 0) ∧ (x^2 + a * x + d = 0) :=
by
  sorry

end no_common_root_l1889_188910


namespace find_packs_size_l1889_188930

theorem find_packs_size (y : ℕ) :
  (24 - 2 * y) * (36 + 4 * y) = 864 → y = 3 :=
by
  intro h
  -- Proof goes here
  sorry

end find_packs_size_l1889_188930


namespace problem_seven_integers_l1889_188953

theorem problem_seven_integers (a b c d e f g : ℕ) 
  (h1 : b = a + 1) 
  (h2 : c = b + 1) 
  (h3 : d = c + 1) 
  (h4 : e = d + 1) 
  (h5 : f = e + 1) 
  (h6 : g = f + 1) 
  (h_sum : a + b + c + d + e + f + g = 2017) : 
  a = 286 ∨ g = 286 :=
sorry

end problem_seven_integers_l1889_188953


namespace max_questions_wrong_to_succeed_l1889_188995

theorem max_questions_wrong_to_succeed :
  ∀ (total_questions : ℕ) (passing_percentage : ℚ),
  total_questions = 50 →
  passing_percentage = 0.75 →
  ∃ (max_wrong : ℕ), max_wrong = 12 ∧
    (total_questions - max_wrong) ≥ passing_percentage * total_questions := by
  intro total_questions passing_percentage h1 h2
  use 12
  constructor
  . rfl
  . sorry  -- Proof omitted

end max_questions_wrong_to_succeed_l1889_188995


namespace donation_amount_is_correct_l1889_188921

def stuffed_animals_barbara : ℕ := 9
def stuffed_animals_trish : ℕ := 2 * stuffed_animals_barbara
def stuffed_animals_sam : ℕ := stuffed_animals_barbara + 5
def stuffed_animals_linda : ℕ := stuffed_animals_sam - 7

def price_per_barbara : ℝ := 2
def price_per_trish : ℝ := 1.5
def price_per_sam : ℝ := 2.5
def price_per_linda : ℝ := 3

def total_amount_collected : ℝ := 
  stuffed_animals_barbara * price_per_barbara +
  stuffed_animals_trish * price_per_trish +
  stuffed_animals_sam * price_per_sam +
  stuffed_animals_linda * price_per_linda

def discount : ℝ := 0.10

def final_amount : ℝ := total_amount_collected * (1 - discount)

theorem donation_amount_is_correct : final_amount = 90.90 := sorry

end donation_amount_is_correct_l1889_188921


namespace last_two_videos_length_l1889_188986

noncomputable def ad1 : ℕ := 45
noncomputable def ad2 : ℕ := 30
noncomputable def pause1 : ℕ := 45
noncomputable def pause2 : ℕ := 30
noncomputable def video1 : ℕ := 120
noncomputable def video2 : ℕ := 270
noncomputable def total_time : ℕ := 960

theorem last_two_videos_length : 
    ∃ v : ℕ, 
    v = 210 ∧ 
    total_time = ad1 + ad2 + video1 + video2 + pause1 + pause2 + 2 * v :=
by
  sorry

end last_two_videos_length_l1889_188986


namespace find_number_l1889_188935

theorem find_number (x : ℝ) (h : x / 5 = 70 + x / 6) : x = 2100 :=
sorry

end find_number_l1889_188935


namespace inequality_solution_l1889_188972

-- Definitions
variables {a b : ℝ}

-- Hypothesis
variable (h : a > b)

-- Theorem
theorem inequality_solution : -2 * a < -2 * b :=
sorry

end inequality_solution_l1889_188972


namespace incorrect_transformation_l1889_188925

theorem incorrect_transformation (a b c : ℝ) (h1 : a = b) (h2 : c = 0) : ¬(a / c = b / c) :=
by
  sorry

end incorrect_transformation_l1889_188925


namespace c_sub_a_equals_90_l1889_188920

variables (a b c : ℝ)

theorem c_sub_a_equals_90 (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 90) : c - a = 90 :=
by
  sorry

end c_sub_a_equals_90_l1889_188920


namespace palindrome_count_l1889_188909

theorem palindrome_count :
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  (A_choices * B_choices * C_choices) = 900 :=
by
  let A_choices := 9
  let B_choices := 10
  let C_choices := 10
  show (A_choices * B_choices * C_choices) = 900
  sorry

end palindrome_count_l1889_188909


namespace triangle_side_lengths_log_l1889_188960

theorem triangle_side_lengths_log (m : ℕ) (log15 log81 logm : ℝ)
  (h1 : log15 = Real.log 15 / Real.log 10)
  (h2 : log81 = Real.log 81 / Real.log 10)
  (h3 : logm = Real.log m / Real.log 10)
  (h4 : 0 < log15 ∧ 0 < log81 ∧ 0 < logm)
  (h5 : log15 + log81 > logm)
  (h6 : log15 + logm > log81)
  (h7 : log81 + logm > log15)
  (h8 : m > 0) :
  6 ≤ m ∧ m < 1215 → 
  ∃ n : ℕ, n = 1215 - 6 ∧ n = 1209 :=
by
  sorry

end triangle_side_lengths_log_l1889_188960


namespace find_numbers_l1889_188942

theorem find_numbers
  (X Y : ℕ)
  (h1 : 10 ≤ X ∧ X < 100)
  (h2 : 10 ≤ Y ∧ Y < 100)
  (h3 : X = 2 * Y)
  (h4 : ∃ a b c d, X = 10 * a + b ∧ Y = 10 * c + d ∧ (c + d = a + b) ∧ (c = a - b ∨ d = a - b)) :
  X = 34 ∧ Y = 17 :=
sorry

end find_numbers_l1889_188942


namespace ratio_of_u_to_v_l1889_188957

theorem ratio_of_u_to_v {b u v : ℝ} 
  (h1 : b ≠ 0)
  (h2 : 0 = 12 * u + b)
  (h3 : 0 = 8 * v + b) : 
  u / v = 2 / 3 := 
by
  sorry

end ratio_of_u_to_v_l1889_188957


namespace train_pass_bridge_time_l1889_188981

noncomputable def length_of_train : ℝ := 485
noncomputable def length_of_bridge : ℝ := 140
noncomputable def speed_of_train_kmph : ℝ := 45 
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

theorem train_pass_bridge_time :
  (length_of_train + length_of_bridge) / speed_of_train_mps = 50 :=
by
  sorry

end train_pass_bridge_time_l1889_188981


namespace company_a_taxis_l1889_188974

variable (a b : ℕ)

theorem company_a_taxis
  (h1 : 5 * a < 56)
  (h2 : 6 * a > 56)
  (h3 : 4 * b < 56)
  (h4 : 5 * b > 56)
  (h5 : b = a + 3) :
  a = 10 := by
  sorry

end company_a_taxis_l1889_188974


namespace contestants_order_l1889_188975

variables (G E H F : ℕ) -- Scores of the participants, given that they are nonnegative

theorem contestants_order (h1 : E + G = F + H) (h2 : F + E = H + G) (h3 : G > E + F) : 
  G ≥ E ∧ G ≥ H ∧ G ≥ F ∧ E = H ∧ E ≥ F :=
by {
  sorry
}

end contestants_order_l1889_188975


namespace mother_present_age_l1889_188941

def person_present_age (P M : ℕ) : Prop :=
  P = (2 / 5) * M

def person_age_in_10_years (P M : ℕ) : Prop :=
  P + 10 = (1 / 2) * (M + 10)

theorem mother_present_age (P M : ℕ) (h1 : person_present_age P M) (h2 : person_age_in_10_years P M) : M = 50 :=
sorry

end mother_present_age_l1889_188941


namespace cos_double_alpha_two_alpha_minus_beta_l1889_188907

variable (α β : ℝ)
variable (α_pos : 0 < α)
variable (α_lt_pi : α < π)
variable (tan_α : Real.tan α = 2)

variable (β_pos : 0 < β)
variable (β_lt_pi : β < π)
variable (cos_β : Real.cos β = -((7 * Real.sqrt 2) / 10))

theorem cos_double_alpha (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2) : 
  Real.cos (2 * α) = -3 / 5 := by
  sorry

theorem two_alpha_minus_beta (hα : 0 < α ∧ α < π) (htan : Real.tan α = 2)
  (hβ : 0 < β ∧ β < π) (hcosβ : Real.cos β = -((7 * Real.sqrt 2) / 10)) : 
  2 * α - β = -π / 4 := by
  sorry

end cos_double_alpha_two_alpha_minus_beta_l1889_188907


namespace cuboid_diagonal_cubes_l1889_188912

def num_cubes_intersecting_diagonal (a b c : ℕ) : ℕ :=
  a + b + c - 2

theorem cuboid_diagonal_cubes :
  num_cubes_intersecting_diagonal 77 81 100 = 256 :=
by
  sorry

end cuboid_diagonal_cubes_l1889_188912


namespace retailer_discount_problem_l1889_188978

theorem retailer_discount_problem
  (CP MP SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (0.65 * CP))
  (h3 : SP = CP + (0.2375 * CP)) :
  (MP - SP) / MP * 100 = 25 :=
by
  sorry

end retailer_discount_problem_l1889_188978


namespace probability_properties_l1889_188962

noncomputable def P1 : ℝ := 1 / 4
noncomputable def P2 : ℝ := 1 / 4
noncomputable def P3 : ℝ := 1 / 2

theorem probability_properties :
  (P1 ≠ P3) ∧
  (P1 + P2 = P3) ∧
  (P1 + P2 + P3 = 1) ∧
  (P3 = 2 * P1) ∧
  (P3 = 2 * P2) :=
by
  sorry

end probability_properties_l1889_188962


namespace geometric_sequence_a4_l1889_188980

variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def condition1 : Prop := 3 * a 5 = a 6
def condition2 : Prop := a 2 = 1

-- Question
def question : Prop := a 4 = 9

theorem geometric_sequence_a4 (h1 : condition1 a) (h2 : condition2 a) : question a :=
sorry

end geometric_sequence_a4_l1889_188980


namespace gcd_2_l1889_188911

-- Define the two numbers obtained from the conditions.
def n : ℕ := 3589 - 23
def m : ℕ := 5273 - 41

-- State that the GCD of n and m is 2.
theorem gcd_2 : Nat.gcd n m = 2 := by
  sorry

end gcd_2_l1889_188911


namespace john_total_climb_height_l1889_188964

-- Define the heights and conditions
def num_flights : ℕ := 3
def height_per_flight : ℕ := 10
def total_stairs_height : ℕ := num_flights * height_per_flight
def rope_height : ℕ := total_stairs_height / 2
def ladder_height : ℕ := rope_height + 10

-- Prove that the total height John climbed is 70 feet
theorem john_total_climb_height : 
  total_stairs_height + rope_height + ladder_height = 70 := by
  sorry

end john_total_climb_height_l1889_188964


namespace ones_digit_of_9_pow_27_l1889_188961

-- Definitions representing the cyclical pattern
def ones_digit_of_9_power (n : ℕ) : ℕ :=
  if n % 2 = 1 then 9 else 1

-- The problem statement to be proven
theorem ones_digit_of_9_pow_27 : ones_digit_of_9_power 27 = 9 := 
by
  -- the detailed proof steps are omitted
  sorry

end ones_digit_of_9_pow_27_l1889_188961


namespace area_of_triangle_ADE_l1889_188944

noncomputable def triangle_area (A B C: ℝ × ℝ) : ℝ :=
  1 / 2 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem area_of_triangle_ADE (A B D E F : ℝ × ℝ) (h₁ : A.1 = 0 ∧ A.2 = 0) (h₂ : B.1 = 8 ∧ B.2 = 0)
  (h₃ : D.1 = 8 ∧ D.2= 8) (h₄ : E.1 = 4 * 3 / 5 ∧ E.2 = 0) 
  (h₅ : F.1 = 0 ∧ F.2 = 12) :
  triangle_area A D E = 288 / 25 := 
sorry

end area_of_triangle_ADE_l1889_188944


namespace line_segment_intersection_range_l1889_188904

theorem line_segment_intersection_range (P Q : ℝ × ℝ) (m : ℝ)
  (hP : P = (-1, 1)) (hQ : Q = (2, 2)) :
  ∃ m : ℝ, (x + m * y + m = 0) ∧ (-3 < m ∧ m < -2/3) := 
sorry

end line_segment_intersection_range_l1889_188904


namespace spoiled_milk_percentage_l1889_188963

theorem spoiled_milk_percentage (p_egg p_flour p_all_good : ℝ) (h_egg : p_egg = 0.40) (h_flour : p_flour = 0.75) (h_all_good : p_all_good = 0.24) : 
  (1 - (p_all_good / (p_egg * p_flour))) = 0.20 :=
by
  sorry

end spoiled_milk_percentage_l1889_188963


namespace midpoint_of_segment_l1889_188928

def A : ℝ × ℝ × ℝ := (10, -3, 5)
def B : ℝ × ℝ × ℝ := (-2, 7, -4)

theorem midpoint_of_segment :
  let M_x := (10 + -2 : ℝ) / 2
  let M_y := (-3 + 7 : ℝ) / 2
  let M_z := (5 + -4 : ℝ) / 2
  (M_x, M_y, M_z) = (4, 2, 0.5) :=
by
  let M_x : ℝ := (10 + -2) / 2
  let M_y : ℝ := (-3 + 7) / 2
  let M_z : ℝ := (5 + -4) / 2
  show (M_x, M_y, M_z) = (4, 2, 0.5)
  repeat { sorry }

end midpoint_of_segment_l1889_188928


namespace sector_arc_length_l1889_188973

theorem sector_arc_length (r : ℝ) (θ : ℝ) (L : ℝ) (h₁ : r = 1) (h₂ : θ = 60 * π / 180) : L = π / 3 :=
by
  sorry

end sector_arc_length_l1889_188973


namespace sum_of_series_l1889_188965

theorem sum_of_series : (1 / (1 * 2 * 3) + 1 / (2 * 3 * 4) + 1 / (3 * 4 * 5) + 1 / (4 * 5 * 6)) = 7 / 30 :=
by
  sorry

end sum_of_series_l1889_188965


namespace area_of_rectangular_plot_l1889_188927

theorem area_of_rectangular_plot (breadth : ℝ) (length : ℝ) 
    (h1 : breadth = 17) 
    (h2 : length = 3 * breadth) : 
    length * breadth = 867 := 
by
  sorry

end area_of_rectangular_plot_l1889_188927


namespace ratio_paid_back_to_initial_debt_l1889_188940

def initial_debt : ℕ := 40
def still_owed : ℕ := 30
def paid_back (initial_debt still_owed : ℕ) : ℕ := initial_debt - still_owed

theorem ratio_paid_back_to_initial_debt
  (initial_debt still_owed : ℕ) :
  (paid_back initial_debt still_owed : ℚ) / initial_debt = 1 / 4 :=
by 
  sorry

end ratio_paid_back_to_initial_debt_l1889_188940


namespace factor_correct_l1889_188994

-- Define the polynomial p(x)
def p (x : ℤ) : ℤ := 6 * (x + 4) * (x + 7) * (x + 9) * (x + 11) - 5 * x^2

-- Define the potential factors of p(x)
def f1 (x : ℤ) : ℤ := 3 * x^2 + 93 * x
def f2 (x : ℤ) : ℤ := 2 * x^2 + 178 * x + 5432

theorem factor_correct : ∀ x : ℤ, p x = f1 x * f2 x := by
  sorry

end factor_correct_l1889_188994


namespace problem_l1889_188967

-- Define first terms
def a_1 : ℕ := 12
def b_1 : ℕ := 48

-- Define the 100th term condition
def a_100 (d_a : ℚ) := 12 + 99 * d_a
def b_100 (d_b : ℚ) := 48 + 99 * d_b

-- Condition that the sum of the 100th terms is 200
def condition (d_a d_b : ℚ) := a_100 d_a + b_100 d_b = 200

-- Define the value of the sum of the first 100 terms
def sequence_sum (d_a d_b : ℚ) := 100 * 60 + (140 / 99) * ((99 * 100) / 2)

-- The proof theorem
theorem problem : ∀ d_a d_b : ℚ, condition d_a d_b → sequence_sum d_a d_b = 13000 :=
by
  intros d_a d_b h_cond
  sorry

end problem_l1889_188967


namespace harriet_current_age_l1889_188903

theorem harriet_current_age (peter_age harriet_age : ℕ) (mother_age : ℕ := 60) (h₁ : peter_age = mother_age / 2) 
  (h₂ : peter_age + 4 = 2 * (harriet_age + 4)) : harriet_age = 13 :=
by
  sorry

end harriet_current_age_l1889_188903


namespace find_pairs_l1889_188902

noncomputable def x (a b : ℝ) : ℝ := b^2 - (a - 1)/2
noncomputable def y (a b : ℝ) : ℝ := a^2 + (b + 1)/2
def valid_pair (a b : ℝ) : Prop := max (x a b) (y a b) ≤ 7 / 16

theorem find_pairs : valid_pair (1/4) (-1/4) :=
  sorry

end find_pairs_l1889_188902


namespace find_red_coin_l1889_188950

/- Define the function f(n) as the minimum number of scans required to determine the red coin
   - out of n coins with the given conditions.
   - Seyed has 998 white coins, 1 red coin, and 1 red-white coin.
-/

def f (n : Nat) : Nat := sorry

/- The main theorem to be proved: There exists an algorithm that can find the red coin using 
   the scanner at most 17 times for 1000 coins.
-/

theorem find_red_coin (n : Nat) (h : n = 1000) : f n ≤ 17 := sorry

end find_red_coin_l1889_188950


namespace max_cookies_ben_could_have_eaten_l1889_188936

theorem max_cookies_ben_could_have_eaten (c : ℕ) (h_total : c = 36)
  (h_beth : ∃ n: ℕ, (n = 2 ∨ n = 3) ∧ c = (n + 1) * ben)
  (h_max : ∀ n, (n = 2 ∨ n = 3) → n * 12 ≤ n * ben)
  : ben = 12 := 
sorry

end max_cookies_ben_could_have_eaten_l1889_188936


namespace find_y_l1889_188969

theorem find_y (x y : ℤ) (h1 : x = -4) (h2 : x^2 + 3 * x + 7 = y - 5) : y = 16 := 
by
  sorry

end find_y_l1889_188969


namespace books_cost_l1889_188987

theorem books_cost (total_cost_three_books cost_seven_books : ℕ) 
  (h₁ : total_cost_three_books = 45)
  (h₂ : cost_seven_books = 7 * (total_cost_three_books / 3)) : 
  cost_seven_books = 105 :=
  sorry

end books_cost_l1889_188987


namespace min_k_value_l1889_188982

noncomputable def minimum_k_condition (x y z k : ℝ) : Prop :=
  k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x * y * z)^2 - (x * y * z) + 1

theorem min_k_value :
  ∀ x y z : ℝ, x ≤ 0 → y ≤ 0 → z ≤ 0 → minimum_k_condition x y z (16 / 9) :=
by
  sorry

end min_k_value_l1889_188982


namespace sum_of_three_numbers_l1889_188948

theorem sum_of_three_numbers (a b c : ℝ) (h1 : a + b = 35) (h2 : b + c = 42) (h3 : c + a = 58) :
  a + b + c = 67.5 :=
by
  sorry

end sum_of_three_numbers_l1889_188948


namespace shirts_not_washed_l1889_188934

def total_shortsleeve_shirts : Nat := 40
def total_longsleeve_shirts : Nat := 23
def washed_shirts : Nat := 29

theorem shirts_not_washed :
  (total_shortsleeve_shirts + total_longsleeve_shirts) - washed_shirts = 34 :=
by
  sorry

end shirts_not_washed_l1889_188934


namespace negation_of_exists_cond_l1889_188924

theorem negation_of_exists_cond (x : ℝ) (h : x > 0) : ¬ (∃ x : ℝ, x > 0 ∧ x^3 - x + 1 > 0) ↔ (∀ x : ℝ, x > 0 → x^3 - x + 1 ≤ 0) :=
by 
  sorry

end negation_of_exists_cond_l1889_188924


namespace cards_from_country_correct_l1889_188947

def total_cards : ℝ := 403.0
def cards_from_home : ℝ := 287.0
def cards_from_country : ℝ := total_cards - cards_from_home

theorem cards_from_country_correct : cards_from_country = 116.0 := by
  -- proof to be added
  sorry

end cards_from_country_correct_l1889_188947


namespace value_of_S_l1889_188989

def pseudocode_value : ℕ := 1
def increment (S I : ℕ) : ℕ := S + I

def loop_steps : ℕ :=
  let S := pseudocode_value
  let S := increment S 1
  let S := increment S 3
  let S := increment S 5
  let S := increment S 7
  S

theorem value_of_S : loop_steps = 17 :=
  by sorry

end value_of_S_l1889_188989


namespace system_no_solution_iff_n_eq_neg_half_l1889_188939

theorem system_no_solution_iff_n_eq_neg_half (x y z n : ℝ) :
  (¬ ∃ x y z, 2 * n * x + y = 2 ∧ n * y + 2 * z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1/2 := by
  sorry

end system_no_solution_iff_n_eq_neg_half_l1889_188939


namespace range_of_k_l1889_188949

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x^2 + k * y^2 = 2) ∧ (∀ x y : ℝ, y ≠ 0 → x^2 + k * y^2 = 2 → (x = 0 ∧ (∃ a : ℝ, a > 1 ∧ y = a))) → 0 < k ∧ k < 1 :=
sorry

end range_of_k_l1889_188949


namespace cube_volume_l1889_188966

theorem cube_volume (A : ℝ) (hA : A = 96) (s : ℝ) (hS : A = 6 * s^2) : s^3 = 64 := by
  sorry

end cube_volume_l1889_188966


namespace notebooks_distributed_l1889_188954

theorem notebooks_distributed  (C : ℕ) (N : ℕ) 
  (h1 : N = C^2 / 8) 
  (h2 : N = 8 * C) : 
  N = 512 :=
by 
  sorry

end notebooks_distributed_l1889_188954


namespace find_dividend_l1889_188988

theorem find_dividend (q : ℕ) (d : ℕ) (r : ℕ) (D : ℕ) 
  (h_q : q = 15000)
  (h_d : d = 82675)
  (h_r : r = 57801)
  (h_D : D = 1240182801) :
  D = d * q + r := by 
  sorry

end find_dividend_l1889_188988


namespace calculateL_l1889_188933

-- Defining the constants T, H, and C
def T : ℕ := 5
def H : ℕ := 10
def C : ℕ := 3

-- Definition of the formula for L
def crushingLoad (T H C : ℕ) : ℚ := (15 * T^3 : ℚ) / (H^2 + C)

-- The theorem to prove
theorem calculateL : crushingLoad T H C = 1875 / 103 := by
  -- Proof goes here
  sorry

end calculateL_l1889_188933


namespace sufficient_but_not_necessary_not_necessary_condition_l1889_188983

theorem sufficient_but_not_necessary 
  (α : ℝ) (h : Real.sin α = Real.cos α) :
  Real.cos (2 * α) = 0 :=
by sorry

theorem not_necessary_condition 
  (α : ℝ) (h : Real.cos (2 * α) = 0) :
  ∃ β : ℝ, Real.sin β ≠ Real.cos β :=
by sorry

end sufficient_but_not_necessary_not_necessary_condition_l1889_188983


namespace solution_sets_equiv_solve_l1889_188955

theorem solution_sets_equiv_solve (a b : ℝ) :
  (∀ x : ℝ, (4 * x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 / 4) →
  (∀ x : ℝ, a * x^2 + b * x - 2 > 0 ↔ -2 < x ∧ x < -1 / 4) →
  a = -4 ∧ b = -9 := by
  sorry

end solution_sets_equiv_solve_l1889_188955


namespace values_are_equal_and_differ_in_precision_l1889_188976

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

end values_are_equal_and_differ_in_precision_l1889_188976


namespace geom_seq_inequality_l1889_188919

theorem geom_seq_inequality 
  (a : ℕ → ℝ) 
  (q : ℝ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_pos : ∀ n : ℕ, a n > 0) 
  (h_q : q ≠ 1) : 
  a 1 + a 4 > a 2 + a 3 := 
sorry

end geom_seq_inequality_l1889_188919


namespace square_area_eq_l1889_188908

-- Define the side length of the square and the diagonal relationship
variables (s : ℝ) (h : s * Real.sqrt 2 = s + 1)

-- State the theorem to solve
theorem square_area_eq :
  s * Real.sqrt 2 = s + 1 → (s ^ 2 = 3 + 2 * Real.sqrt 2) :=
by
  -- Assume the given condition
  intro h
  -- Insert proof steps here, analysis follows the provided solution steps.
  sorry

end square_area_eq_l1889_188908


namespace radius_of_roots_circle_l1889_188956

theorem radius_of_roots_circle (z : ℂ) (hz : (z - 2)^6 = 64 * z^6) : ∃ r : ℝ, r = 2 / 3 :=
by
  sorry

end radius_of_roots_circle_l1889_188956


namespace least_possible_average_of_integers_l1889_188932

theorem least_possible_average_of_integers :
  ∃ (a b c d : ℤ), a < b ∧ b < c ∧ c < d ∧ d = 90 ∧ a ≥ 21 ∧ (a + b + c + d) / 4 = 39 := by
sorry

end least_possible_average_of_integers_l1889_188932


namespace water_percentage_in_fresh_grapes_l1889_188971

theorem water_percentage_in_fresh_grapes 
  (P : ℝ) -- the percentage of water in fresh grapes
  (fresh_grapes_weight : ℝ := 40) -- weight of fresh grapes in kg
  (dry_grapes_weight : ℝ := 5) -- weight of dry grapes in kg
  (dried_grapes_water_percentage : ℝ := 20) -- percentage of water in dried grapes
  (solid_content : ℝ := 4) -- solid content in both fresh and dried grapes in kg
  : P = 90 :=
by
  sorry

end water_percentage_in_fresh_grapes_l1889_188971


namespace inequality_solution_l1889_188916

theorem inequality_solution (a : ℝ) (h : a > 0) :
  (if a = 2 then {x : ℝ | false}
   else if 0 < a ∧ a < 2 then {x : ℝ | 1 < x ∧ x ≤ 2 / a}
   else if a > 2 then {x : ℝ | 2 / a ≤ x ∧ x < 1}
   else ∅) =
    {x : ℝ | (a + 2) * x - 4 ≤ 2 * (x - 1)} :=
by
  sorry

end inequality_solution_l1889_188916


namespace cost_price_per_meter_l1889_188977

theorem cost_price_per_meter (selling_price : ℝ) (total_meters : ℕ) (profit_per_meter : ℝ)
  (h1 : selling_price = 8925)
  (h2 : total_meters = 85)
  (h3 : profit_per_meter = 5) :
  (selling_price - total_meters * profit_per_meter) / total_meters = 100 := by
  sorry

end cost_price_per_meter_l1889_188977


namespace evaluate_expression_l1889_188945

theorem evaluate_expression : 2^3 + 2^3 + 2^3 + 2^3 = 2^5 := by
  sorry

end evaluate_expression_l1889_188945


namespace inverse_proposition_true_l1889_188959

-- Define a rectangle and a square
structure Rectangle where
  length : ℝ
  width  : ℝ

def is_square (r : Rectangle) : Prop :=
  r.length = r.width ∧ r.length > 0 ∧ r.width > 0

-- Define the condition that a rectangle with equal adjacent sides is a square
def rectangle_with_equal_adjacent_sides_is_square : Prop :=
  ∀ r : Rectangle, r.length = r.width → is_square r

-- Define the inverse proposition that a square is a rectangle with equal adjacent sides
def square_is_rectangle_with_equal_adjacent_sides : Prop :=
  ∀ r : Rectangle, is_square r → r.length = r.width

-- The proof statement
theorem inverse_proposition_true :
  rectangle_with_equal_adjacent_sides_is_square → square_is_rectangle_with_equal_adjacent_sides :=
by
  sorry

end inverse_proposition_true_l1889_188959


namespace kira_breakfast_time_l1889_188929

theorem kira_breakfast_time (n_sausages : ℕ) (n_eggs : ℕ) (t_fry_per_sausage : ℕ) (t_scramble_per_egg : ℕ) (total_time : ℕ) :
  n_sausages = 3 → n_eggs = 6 → t_fry_per_sausage = 5 → t_scramble_per_egg = 4 → total_time = (n_sausages * t_fry_per_sausage + n_eggs * t_scramble_per_egg) →
  total_time = 39 :=
by
  intros h_sausages h_eggs h_fry h_scramble h_total
  rw [h_sausages, h_eggs, h_fry, h_scramble] at h_total
  exact h_total

end kira_breakfast_time_l1889_188929


namespace smallest_n_l1889_188985

theorem smallest_n (n : ℕ) (h : 5 * n % 26 = 220 % 26) : n = 18 :=
by
  -- Initial congruence simplification
  have h1 : 220 % 26 = 12 := by norm_num
  rw [h1] at h
  -- Reformulation of the problem
  have h2 : 5 * n % 26 = 12 := h
  -- Conclude the smallest n
  sorry

end smallest_n_l1889_188985


namespace max_value_of_sum_max_value_achievable_l1889_188938

theorem max_value_of_sum (a b c d : ℝ) 
  (h : a^6 + b^6 + c^6 + d^6 = 64) : a^7 + b^7 + c^7 + d^7 ≤ 128 :=
sorry

theorem max_value_achievable : ∃ a b c d : ℝ,
  a^6 + b^6 + c^6 + d^6 = 64 ∧ a^7 + b^7 + c^7 + d^7 = 128 :=
sorry

end max_value_of_sum_max_value_achievable_l1889_188938


namespace average_salary_decrease_l1889_188993

theorem average_salary_decrease 
    (avg_wage_illiterate_initial : ℝ)
    (avg_wage_illiterate_new : ℝ)
    (num_illiterate : ℕ)
    (num_literate : ℕ)
    (num_total : ℕ)
    (total_decrease : ℝ) :
    avg_wage_illiterate_initial = 25 →
    avg_wage_illiterate_new = 10 →
    num_illiterate = 20 →
    num_literate = 10 →
    num_total = num_illiterate + num_literate →
    total_decrease = (avg_wage_illiterate_initial - avg_wage_illiterate_new) * num_illiterate →
    total_decrease / num_total = 10 :=
by
  intros avg_wage_illiterate_initial_eq avg_wage_illiterate_new_eq num_illiterate_eq num_literate_eq num_total_eq total_decrease_eq
  sorry

end average_salary_decrease_l1889_188993


namespace kylie_total_beads_used_l1889_188991

noncomputable def beads_monday_necklaces : ℕ := 10 * 20
noncomputable def beads_tuesday_necklaces : ℕ := 2 * 20
noncomputable def beads_wednesday_bracelets : ℕ := 5 * 10
noncomputable def beads_thursday_earrings : ℕ := 3 * 5
noncomputable def beads_friday_anklets : ℕ := 4 * 8
noncomputable def beads_friday_rings : ℕ := 6 * 7

noncomputable def total_beads_used : ℕ :=
  beads_monday_necklaces +
  beads_tuesday_necklaces +
  beads_wednesday_bracelets +
  beads_thursday_earrings +
  beads_friday_anklets +
  beads_friday_rings

theorem kylie_total_beads_used : total_beads_used = 379 := by
  sorry

end kylie_total_beads_used_l1889_188991


namespace false_implies_not_all_ripe_l1889_188998

def all_ripe (basket : Type) [Nonempty basket] (P : basket → Prop) : Prop :=
  ∀ x : basket, P x

theorem false_implies_not_all_ripe
  (basket : Type)
  [Nonempty basket]
  (P : basket → Prop)
  (h : ¬ all_ripe basket P) :
  (∃ x, ¬ P x) ∧ ¬ all_ripe basket P :=
by
  sorry

end false_implies_not_all_ripe_l1889_188998


namespace stanley_run_walk_difference_l1889_188996

theorem stanley_run_walk_difference :
  ∀ (ran walked : ℝ), ran = 0.4 → walked = 0.2 → ran - walked = 0.2 :=
by
  intros ran walked h_ran h_walk
  rw [h_ran, h_walk]
  norm_num

end stanley_run_walk_difference_l1889_188996


namespace union_complement_l1889_188913

open Set

def U : Set ℤ := {x | -3 < x ∧ x < 3}

def A : Set ℤ := {1, 2}

def B : Set ℤ := {-2, -1, 2}

theorem union_complement :
  A ∪ (U \ B) = {0, 1, 2} := by
  sorry

end union_complement_l1889_188913


namespace squirrel_count_l1889_188951

theorem squirrel_count (n m : ℕ) (h1 : n = 12) (h2 : m = 12 + 12 / 3) : n + m = 28 := by
  sorry

end squirrel_count_l1889_188951


namespace count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l1889_188970

theorem count_positive_even_multiples_of_3_less_than_5000_perfect_squares :
  ∃ n : ℕ, (n = 11) ∧ ∀ k : ℕ, (k < 5000) → (k % 2 = 0) → (k % 3 = 0) → (∃ m : ℕ, k = m * m) → k ≤ 36 * 11 * 11 :=
by {
  sorry
}

end count_positive_even_multiples_of_3_less_than_5000_perfect_squares_l1889_188970


namespace hyperbola_eccentricity_range_l1889_188922

-- Definitions of hyperbola and distance condition
def hyperbola (x y a b : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def distance_condition (a b : ℝ) : Prop :=
  ∀ (x y : ℝ), hyperbola x y a b → (b * x + a * y - 2 * a * b) > a

-- The range of the eccentricity
theorem hyperbola_eccentricity_range (a b : ℝ) (h : hyperbola 0 1 a b) 
  (dist_cond : distance_condition a b) : 
  ∃ e : ℝ, e ≥ (2 * Real.sqrt 3 / 3) :=
sorry

end hyperbola_eccentricity_range_l1889_188922


namespace range_of_x_f_greater_than_4_l1889_188915

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then 2^(-x) else x^2

theorem range_of_x_f_greater_than_4 :
  { x : ℝ | f x > 4 } = { x : ℝ | x < -2 ∨ x > 2 } :=
by
  sorry

end range_of_x_f_greater_than_4_l1889_188915


namespace quadratic_intersects_xaxis_once_l1889_188968

theorem quadratic_intersects_xaxis_once (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + k = 0) ↔ k = 1 :=
by
  sorry

end quadratic_intersects_xaxis_once_l1889_188968


namespace circles_common_point_l1889_188937

theorem circles_common_point {n : ℕ} (hn : n ≥ 5) (circles : Fin n → Set Point)
  (hcommon : ∀ (a b c : Fin n), (circles a ∩ circles b ∩ circles c).Nonempty) :
  ∃ p : Point, ∀ i : Fin n, p ∈ circles i :=
sorry

end circles_common_point_l1889_188937


namespace relationship_inequality_l1889_188923

variable {a b c d : ℝ}

-- Define the conditions
def is_largest (a b c : ℝ) : Prop := a > b ∧ a > c
def positive_numbers (a b c d : ℝ) : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0
def ratio_condition (a b c d : ℝ) : Prop := a / b = c / d

-- The theorem statement
theorem relationship_inequality 
  (h_largest : is_largest a b c)
  (h_positive : positive_numbers a b c d)
  (h_ratio : ratio_condition a b c d) :
  a + d > b + c :=
sorry

end relationship_inequality_l1889_188923


namespace eat_both_veg_nonveg_l1889_188958

theorem eat_both_veg_nonveg (total_veg only_veg : ℕ) (h1 : total_veg = 31) (h2 : only_veg = 19) :
  (total_veg - only_veg) = 12 :=
by
  have h3 : total_veg - only_veg = 31 - 19 := by rw [h1, h2]
  exact h3

end eat_both_veg_nonveg_l1889_188958


namespace odd_prime_divisibility_two_prime_divisibility_l1889_188943

theorem odd_prime_divisibility (p a n : ℕ) (hp : p % 2 = 1) (hp_prime : Nat.Prime p)
  (ha : a > 0) (hn : n > 0) (div_cond : p^n ∣ a^p - 1) : p^(n-1) ∣ a - 1 :=
sorry

theorem two_prime_divisibility (a n : ℕ) (ha : a > 0) (hn : n > 0) (div_cond : 2^n ∣ a^2 - 1) : ¬ 2^(n-1) ∣ a - 1 :=
sorry

end odd_prime_divisibility_two_prime_divisibility_l1889_188943


namespace problem_proof_l1889_188926

theorem problem_proof (a b c : ℝ) (h1 : a ≠ b ∧ b ≠ c ∧ c ≠ a) 
  (h2 : a / (b - c) + b / (c - a) + c / (a - b) = 0) : 
  a / (b - c) ^ 2 + b / (c - a) ^ 2 + c / (a - b) ^ 2 = 0 :=
sorry

end problem_proof_l1889_188926


namespace paint_room_together_l1889_188999

variable (t : ℚ)
variable (Doug_rate : ℚ := 1/5)
variable (Dave_rate : ℚ := 1/7)
variable (Diana_rate : ℚ := 1/6)
variable (Combined_rate : ℚ := Doug_rate + Dave_rate + Diana_rate)
variable (break_time : ℚ := 2)

theorem paint_room_together:
  Combined_rate * (t - break_time) = 1 :=
sorry

end paint_room_together_l1889_188999


namespace symmetric_circle_equation_l1889_188990

-- Define the original circle and the line of symmetry
def original_circle (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 2
def line_of_symmetry (x y : ℝ) : Prop := 2 * x - y + 3 = 0

-- Proving the equation of the symmetric circle
theorem symmetric_circle_equation :
  (∀ x y : ℝ, original_circle x y ↔ (x + 3)^2 + (y - 2)^2 = 2) :=
by
  sorry

end symmetric_circle_equation_l1889_188990


namespace find_y_satisfies_equation_l1889_188918

theorem find_y_satisfies_equation :
  ∃ y : ℝ, 3 * y + 6 = |(-20 + 2)| :=
by
  sorry

end find_y_satisfies_equation_l1889_188918


namespace min_value_inequality_l1889_188914

theorem min_value_inequality (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  2 * a^2 + 1 / (a * b) + 1 / (a * (a - b)) - 10 * a * c + 25 * c^2 ≥ 4 :=
by
  sorry

end min_value_inequality_l1889_188914


namespace converse_statement_l1889_188931

theorem converse_statement (x : ℝ) :
  x^2 + 3 * x - 2 < 0 → x < 1 :=
sorry

end converse_statement_l1889_188931


namespace percentage_of_masters_l1889_188979

-- Definition of given conditions
def average_points_juniors := 22
def average_points_masters := 47
def overall_average_points := 41

-- Problem statement
theorem percentage_of_masters (x y : ℕ) (hx : x ≥ 0) (hy : y ≥ 0) 
    (h_avg_juniors : 22 * x = average_points_juniors * x)
    (h_avg_masters : 47 * y = average_points_masters * y)
    (h_overall_average : 22 * x + 47 * y = overall_average_points * (x + y)) : 
    (y : ℚ) / (x + y) * 100 = 76 := 
sorry

end percentage_of_masters_l1889_188979


namespace max_cubes_fit_l1889_188952

-- Define the conditions
def box_volume (length : ℕ) (width : ℕ) (height : ℕ) : ℕ := length * width * height
def cube_volume : ℕ := 27
def total_cubes (V_box : ℕ) (V_cube : ℕ) : ℕ := V_box / V_cube

-- Statement of the problem
theorem max_cubes_fit (length width height : ℕ) (V_box : ℕ) (V_cube q : ℕ) :
  length = 8 → width = 9 → height = 12 → V_box = box_volume length width height →
  V_cube = cube_volume → q = total_cubes V_box V_cube → q = 32 :=
by sorry

end max_cubes_fit_l1889_188952


namespace mean_median_difference_l1889_188946

open Real

/-- In a class of 100 students, these are the distributions of scores:
  - 10% scored 60 points
  - 30% scored 75 points
  - 25% scored 80 points
  - 20% scored 90 points
  - 15% scored 100 points

Prove that the difference between the mean and the median scores is 1.5. -/
theorem mean_median_difference :
  let total_students := 100 
  let score_60 := 0.10 * total_students
  let score_75 := 0.30 * total_students
  let score_80 := 0.25 * total_students
  let score_90 := 0.20 * total_students
  let score_100 := (100 - (score_60 + score_75 + score_80 + score_90))
  let median := 80
  let mean := (60 * score_60 + 75 * score_75 + 80 * score_80 + 90 * score_90 + 100 * score_100) / total_students
  mean - median = 1.5 :=
by
  sorry

end mean_median_difference_l1889_188946


namespace weekly_rental_cost_l1889_188901

theorem weekly_rental_cost (W : ℝ) 
  (monthly_cost : ℝ := 40)
  (months_in_year : ℝ := 12)
  (weeks_in_year : ℝ := 52)
  (savings : ℝ := 40)
  (total_year_cost_month : ℝ := months_in_year * monthly_cost)
  (total_year_cost_week : ℝ := total_year_cost_month + savings) :
  (total_year_cost_week / weeks_in_year) = 10 :=
by 
  sorry

end weekly_rental_cost_l1889_188901
