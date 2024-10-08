import Mathlib

namespace find_smaller_integer_l124_124284

noncomputable def average_equals_decimal (m n : ℕ) : Prop :=
  (m + n) / 2 = m + n / 100

theorem find_smaller_integer (m n : ℕ) (h1 : 10 ≤ m ∧ m < 100) (h2 : 10 ≤ n ∧ n < 100) (h3 : 25 ∣ n) (h4 : average_equals_decimal m n) : m = 49 :=
by
  sorry

end find_smaller_integer_l124_124284


namespace angle_in_third_quadrant_l124_124573

/-- 
Given that the terminal side of angle α is in the third quadrant,
prove that the terminal side of α/3 cannot be in the second quadrant.
-/
theorem angle_in_third_quadrant (α : ℝ) (k : ℤ)
  (h : π + 2 * k * π < α ∧ α < 3 / 2 * π + 2 * k * π) :
  ¬ (π / 2 < α / 3 ∧ α / 3 < π) :=
sorry

end angle_in_third_quadrant_l124_124573


namespace complement_is_correct_l124_124945

variable (U : Set ℕ) (A : Set ℕ)

def complement (U : Set ℕ) (A : Set ℕ) : Set ℕ :=
  { x ∈ U | x ∉ A }

theorem complement_is_correct :
  (U = {1, 2, 3, 4, 5, 6, 7}) →
  (A = {2, 4, 5}) →
  complement U A = {1, 3, 6, 7} :=
by
  sorry

end complement_is_correct_l124_124945


namespace find_root_and_coefficient_l124_124885

theorem find_root_and_coefficient (m: ℝ) (x: ℝ) (h₁: x ^ 2 - m * x - 6 = 0) (h₂: x = 3) :
  (x = 3 ∧ -2 = -6 / 3 ∨ m = 1) :=
by
  sorry

end find_root_and_coefficient_l124_124885


namespace measure_angle_PSR_is_40_l124_124697

noncomputable def isosceles_triangle (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : Triangle := sorry
noncomputable def square (D R S T : Point) : Square := sorry
noncomputable def angle (A B C : Point) (θ : ℝ) : Prop := sorry

def angle_PQR (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry
def angle_PRQ (P Q R : Point) (PQ PR : ℝ) (hPQ_PR : PQ = PR) : ℝ := sorry

theorem measure_angle_PSR_is_40
  (P Q R S T D : Point)
  (PQ PR : ℝ)
  (hPQ_PR : PQ = PR)
  (hQ_eq_D : Q = D)
  (hQPS : angle P Q S 100)
  (hDRST_square : square D R S T) : angle P S R 40 :=
by
  -- Proof omitted for brevity
  sorry

end measure_angle_PSR_is_40_l124_124697


namespace lisa_total_spoons_l124_124729

def num_children := 4
def spoons_per_child := 3
def decorative_spoons := 2
def large_spoons := 10
def teaspoons := 15

def total_spoons := num_children * spoons_per_child + decorative_spoons + large_spoons + teaspoons

theorem lisa_total_spoons : total_spoons = 39 := by
  sorry

end lisa_total_spoons_l124_124729


namespace maximum_value_of_3m_4n_l124_124486

noncomputable def max_value (m n : ℕ) : ℕ :=
  3 * m + 4 * n

theorem maximum_value_of_3m_4n 
  (m n : ℕ) 
  (h_even : ∀ i, i < m → (2 * (i + 1)) > 0) 
  (h_odd : ∀ j, j < n → (2 * j + 1) > 0)
  (h_sum : m * (m + 1) + n^2 ≤ 1987) 
  (h_odd_n : n % 2 = 1) :
  max_value m n ≤ 221 := 
sorry

end maximum_value_of_3m_4n_l124_124486


namespace no_solution_xy_l124_124990

theorem no_solution_xy (x y : ℕ) : ¬ (x * (x + 1) = 4 * y * (y + 1)) :=
sorry

end no_solution_xy_l124_124990


namespace exists_three_digit_number_l124_124051

theorem exists_three_digit_number : ∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ (100 * a + 10 * b + c = a^3 + b^3 + c^3) ∧ (100 * a + 10 * b + c ≥ 100 ∧ 100 * a + 10 * b + c < 1000) := 
sorry

end exists_three_digit_number_l124_124051


namespace smallest_number_with_unique_digits_summing_to_32_exists_l124_124937

theorem smallest_number_with_unique_digits_summing_to_32_exists : 
  ∃ n : ℕ, n / 10000 < 10 ∧ (n % 10 ≠ (n / 10) % 10) ∧ 
  ((n / 10) % 10 ≠ (n / 100) % 10) ∧ 
  ((n / 100) % 10 ≠ (n / 1000) % 10) ∧ 
  ((n / 1000) % 10 ≠ (n / 10000) % 10) ∧ 
  (n % 10 + (n / 10) % 10 + (n / 100) % 10 + (n / 1000) % 10 + (n / 10000) % 10 = 32) := 
sorry

end smallest_number_with_unique_digits_summing_to_32_exists_l124_124937


namespace geometric_sequence_formula_l124_124023

def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, a (n + 1) = r * a n

theorem geometric_sequence_formula (a : ℕ → ℝ) (h_geom : geom_seq a)
  (h1 : a 3 = 2) (h2 : a 6 = 16) :
  ∀ n : ℕ, a n = 2 ^ (n - 2) :=
by
  sorry

end geometric_sequence_formula_l124_124023


namespace cube_root_neg_eighth_l124_124721

theorem cube_root_neg_eighth : ∃ x : ℚ, x^3 = -1 / 8 ∧ x = -1 / 2 :=
by
  sorry

end cube_root_neg_eighth_l124_124721


namespace max_tied_teams_l124_124584

theorem max_tied_teams (n : ℕ) (h_n : n = 8) (tournament : Fin n → Fin n → Prop)
  (h_symmetric : ∀ i j, tournament i j ↔ tournament j i)
  (h_antisymmetric : ∀ i j, tournament i j → ¬ tournament j i)
  (h_total : ∀ i j, i ≠ j → tournament i j ∨ tournament j i) :
  ∃ (k : ℕ), k = 7 ∧ ∀ (wins : Fin n → ℕ), 
  (∀ i, wins i = 4 → ∃! j, i ≠ j ∧ tournament i j) → True :=
by sorry

end max_tied_teams_l124_124584


namespace sheep_drowned_proof_l124_124737

def animal_problem_statement (S : ℕ) : Prop :=
  let initial_sheep := 20
  let initial_cows := 10
  let initial_dogs := 14
  let total_animals_made_shore := 35
  let sheep_drowned := S
  let cows_drowned := 2 * S
  let dogs_survived := initial_dogs
  let animals_made_shore := initial_sheep + initial_cows + initial_dogs - (sheep_drowned + cows_drowned)
  30 - 3 * S = 35 - 14

theorem sheep_drowned_proof : ∃ S : ℕ, animal_problem_statement S ∧ S = 3 :=
by
  sorry

end sheep_drowned_proof_l124_124737


namespace more_flour_than_sugar_l124_124044

def cups_of_flour : Nat := 9
def cups_of_sugar : Nat := 6
def flour_added : Nat := 2
def flour_needed : Nat := cups_of_flour - flour_added -- 9 - 2 = 7

theorem more_flour_than_sugar : flour_needed - cups_of_sugar = 1 :=
by
  sorry

end more_flour_than_sugar_l124_124044


namespace geometric_sequence_S28_l124_124228

noncomputable def geom_sequence_sum (S : ℕ → ℝ) (a : ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, S (n * (n + 1) / 2) = a * (1 - r^n) / (1 - r)

theorem geometric_sequence_S28 {S : ℕ → ℝ} (a r : ℝ)
  (h1 : geom_sequence_sum S a r)
  (h2 : S 14 = 3)
  (h3 : 3 * S 7 = 3) :
  S 28 = 15 :=
by
  sorry

end geometric_sequence_S28_l124_124228


namespace treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l124_124686

namespace PirateTreasure

-- Given conditions
def num_pirates_excl_captain := 100
def max_coins := 1000
def remaining_coins_99_pirates := 51
def remaining_coins_77_pirates := 29

-- Problem Part (a): Prove the number of coins in treasure
theorem treasure_contains_645_coins : 
  ∃ (N : ℕ), N < max_coins ∧ (N % 99 = remaining_coins_99_pirates ∧ N % 77 = remaining_coins_77_pirates) ∧ N = 645 :=
  sorry

-- Problem Part (b): Prove the number of pirates Barbaroxa should choose
theorem max_leftover_coins_when_choosing_93_pirates :
  ∃ (n : ℕ), n ≤ num_pirates_excl_captain ∧ (∀ k, k ≤ num_pirates_excl_captain → (645 % k) ≤ (645 % k) ∧ n = 93) :=
  sorry

end PirateTreasure

end treasure_contains_645_coins_max_leftover_coins_when_choosing_93_pirates_l124_124686


namespace gcd_digits_bounded_by_lcm_l124_124064

theorem gcd_digits_bounded_by_lcm (a b : ℕ) (h_a : 10^6 ≤ a ∧ a < 10^7) (h_b : 10^6 ≤ b ∧ b < 10^7) (h_lcm : 10^10 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^11) : Nat.gcd a b < 10^4 :=
by
  sorry

end gcd_digits_bounded_by_lcm_l124_124064


namespace modular_inverse_of_35_mod_36_l124_124892

theorem modular_inverse_of_35_mod_36 : 
  ∃ a : ℤ, (35 * a) % 36 = 1 % 36 ∧ a = 35 := 
by 
  sorry

end modular_inverse_of_35_mod_36_l124_124892


namespace value_of_b_l124_124252

theorem value_of_b :
  (∃ b : ℝ, (1 / Real.log b / Real.log 3 + 1 / Real.log b / Real.log 4 + 1 / Real.log b / Real.log 5 = 1) → b = 60) :=
by
  sorry

end value_of_b_l124_124252


namespace melanie_food_total_weight_l124_124935

def total_weight (brie_oz : ℕ) (bread_lb : ℕ) (tomatoes_lb : ℕ) (zucchini_lb : ℕ) 
           (chicken_lb : ℕ) (raspberries_oz : ℕ) (blueberries_oz : ℕ) : ℕ :=
  let brie_lb := brie_oz / 16
  let raspberries_lb := raspberries_oz / 16
  let blueberries_lb := blueberries_oz / 16
  brie_lb + raspberries_lb + blueberries_lb + bread_lb + tomatoes_lb + zucchini_lb + chicken_lb

theorem melanie_food_total_weight : total_weight 8 1 1 2 (3 / 2) 8 8 = 7 :=
by
  -- result placeholder
  sorry

end melanie_food_total_weight_l124_124935


namespace claire_gerbils_l124_124771

theorem claire_gerbils (G H : ℕ) (h1 : G + H = 92) (h2 : (1/4 : ℚ) * G + (1/3 : ℚ) * H = 25) : G = 68 :=
sorry

end claire_gerbils_l124_124771


namespace union_of_A_and_B_l124_124743

def A : Set ℝ := {x | x < 2}
def B : Set ℝ := {1, 2}

theorem union_of_A_and_B : A ∪ B = {x | x ≤ 2} := sorry

end union_of_A_and_B_l124_124743


namespace annual_interest_rate_is_10_percent_l124_124397

noncomputable def principal (P : ℝ) := P = 1500
noncomputable def total_amount (A : ℝ) := A = 1815
noncomputable def time_period (t : ℝ) := t = 2
noncomputable def compounding_frequency (n : ℝ) := n = 1
noncomputable def interest_rate_compound_interest_formula (P A t n : ℝ) (r : ℝ) := 
  A = P * (1 + r / n) ^ (n * t)

theorem annual_interest_rate_is_10_percent : 
  ∀ (P A t n : ℝ) (r : ℝ), principal P → total_amount A → time_period t → compounding_frequency n → 
  interest_rate_compound_interest_formula P A t n r → r = 0.1 :=
by
  intros P A t n r hP hA ht hn h_formula
  sorry

end annual_interest_rate_is_10_percent_l124_124397


namespace one_inch_cubes_with_red_paint_at_least_two_faces_l124_124369

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_two_faces_l124_124369


namespace locus_of_points_line_or_point_l124_124798

theorem locus_of_points_line_or_point {n : ℕ} (A B : ℕ → ℝ) (k : ℝ) (h : ∀ i, 1 ≤ i ∧ i < n → (A (i + 1) - A i) / (B (i + 1) - B i) = k) :
  ∃ l : ℝ, ∀ i, 1 ≤ i ∧ i ≤ n → (A i + l*B i) = A 1 + l*B 1 :=
by
  sorry

end locus_of_points_line_or_point_l124_124798


namespace total_pennies_l124_124876

theorem total_pennies (rachelle gretchen rocky max taylor : ℕ) (h_r : rachelle = 720) (h_g : gretchen = rachelle / 2)
  (h_ro : rocky = gretchen / 3) (h_m : max = rocky * 4) (h_t : taylor = max / 5) :
  rachelle + gretchen + rocky + max + taylor = 1776 := 
by
  sorry

end total_pennies_l124_124876


namespace benjamin_distance_l124_124116

def speed := 10  -- Speed in kilometers per hour
def time := 8    -- Time in hours

def distance (s t : ℕ) := s * t  -- Distance formula

theorem benjamin_distance : distance speed time = 80 :=
by
  -- proof omitted
  sorry

end benjamin_distance_l124_124116


namespace rectangle_width_l124_124466

theorem rectangle_width (P l: ℕ) (hP : P = 50) (hl : l = 13) : 
  ∃ w : ℕ, 2 * l + 2 * w = P ∧ w = 12 := 
by
  sorry

end rectangle_width_l124_124466


namespace scientific_notation_of_8200000_l124_124190

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l124_124190


namespace quadratic_inequality_solution_l124_124216

theorem quadratic_inequality_solution :
  {x : ℝ | 2*x^2 - 3*x - 2 ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 2} :=
sorry

end quadratic_inequality_solution_l124_124216


namespace real_root_fraction_l124_124333

theorem real_root_fraction (a b : ℝ) 
  (h_cond_a : a^4 - 7 * a - 3 = 0) 
  (h_cond_b : b^4 - 7 * b - 3 = 0)
  (h_order : a > b) : 
  (a - b) / (a^4 - b^4) = 1 / 7 := 
sorry

end real_root_fraction_l124_124333


namespace find_numbers_l124_124059

theorem find_numbers (N : ℕ) (a b : ℕ) :
  N = 5 * a →
  N = 7 * b →
  N = 35 ∨ N = 70 ∨ N = 105 :=
by
  sorry

end find_numbers_l124_124059


namespace single_ticket_cost_l124_124589

/-- Define the conditions: sales total, attendee count, number of couple tickets, and cost of couple tickets. -/
def total_sales : ℤ := 2280
def total_attendees : ℕ := 128
def couple_tickets_sold : ℕ := 16
def cost_of_couple_ticket : ℤ := 35

/-- Define the derived conditions: people covered by couple tickets, single tickets sold, and sales from couple tickets. -/
def people_covered_by_couple_tickets : ℕ := couple_tickets_sold * 2
def single_tickets_sold : ℕ := total_attendees - people_covered_by_couple_tickets
def sales_from_couple_tickets : ℤ := couple_tickets_sold * cost_of_couple_ticket

/-- Define the core equation that ties single ticket sales to the total sales. -/
def core_equation (x : ℤ) : Bool := 
  sales_from_couple_tickets + single_tickets_sold * x = total_sales

-- Finally, the statement that needs to be proved.
theorem single_ticket_cost :
  ∃ x : ℤ, core_equation x ∧ x = 18 := by
  sorry

end single_ticket_cost_l124_124589


namespace juice_fraction_left_l124_124378

theorem juice_fraction_left (initial_juice : ℝ) (given_juice : ℝ) (remaining_juice : ℝ) : 
  initial_juice = 5 → given_juice = 18/4 → remaining_juice = initial_juice - given_juice → remaining_juice = 1/2 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end juice_fraction_left_l124_124378


namespace smallest_multiple_of_8_and_9_l124_124315

theorem smallest_multiple_of_8_and_9 : ∃ n : ℕ, n > 0 ∧ (n % 8 = 0) ∧ (n % 9 = 0) ∧ (∀ m : ℕ, m > 0 ∧ (m % 8 = 0) ∧ (m % 9 = 0) → n ≤ m) ∧ n = 72 :=
by
  sorry

end smallest_multiple_of_8_and_9_l124_124315


namespace find_initial_money_l124_124855

-- Definitions of the conditions
def basketball_card_cost : ℕ := 3
def baseball_card_cost : ℕ := 4
def basketball_packs : ℕ := 2
def baseball_decks : ℕ := 5
def change_received : ℕ := 24

-- Total cost calculation
def total_cost : ℕ := (basketball_card_cost * basketball_packs) + (baseball_card_cost * baseball_decks)

-- Initial money calculation
def initial_money : ℕ := total_cost + change_received

-- Proof statement
theorem find_initial_money : initial_money = 50 := 
by
  -- Proof steps would go here
  sorry

end find_initial_money_l124_124855


namespace system1_solution_system2_solution_l124_124647

-- Definition and proof for System (1)
theorem system1_solution (x y : ℝ) (h1 : x - y = 2) (h2 : 2 * x + y = 7) : x = 3 ∧ y = 1 := 
by 
  sorry

-- Definition and proof for System (2)
theorem system2_solution (x y : ℝ) (h1 : x - 2 * y = 3) (h2 : (1 / 2) * x + (3 / 4) * y = 13 / 4) : x = 5 ∧ y = 1 :=
by 
  sorry

end system1_solution_system2_solution_l124_124647


namespace sin_2gamma_proof_l124_124823

-- Assume necessary definitions and conditions
variables {A B C D P : Type}
variables [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited P]
variables (a b c d: ℝ)
variables (α β γ: ℝ)

-- Assume points A, B, C, D, P lie on a circle in that order and AB = BC = CD
axiom points_on_circle : a = b ∧ b = c ∧ c = d
axiom cos_apc : Real.cos α = 3/5
axiom cos_bpd : Real.cos β = 1/5

noncomputable def sin_2gamma : ℝ :=
  2 * Real.sin γ * Real.cos γ

-- Statement to prove sin(2 * γ) given the conditions
theorem sin_2gamma_proof : sin_2gamma γ = 8 * Real.sqrt 5 / 25 :=
sorry

end sin_2gamma_proof_l124_124823


namespace smallest_base_l124_124006

theorem smallest_base (b : ℕ) : (b^2 ≤ 100 ∧ 100 < b^3) → b = 5 :=
by
  intros h
  sorry

end smallest_base_l124_124006


namespace arrangement_count_l124_124888

def number_of_arrangements (slots total_geometry total_number_theory : ℕ) : ℕ :=
  Nat.choose slots total_geometry

theorem arrangement_count :
  number_of_arrangements 8 5 3 = 56 := 
by
  sorry

end arrangement_count_l124_124888


namespace bugs_meet_at_point_P_l124_124971

theorem bugs_meet_at_point_P (r1 r2 v1 v2 t : ℝ) (h1 : r1 = 7) (h2 : r2 = 3) (h3 : v1 = 4 * Real.pi) (h4 : v2 = 3 * Real.pi) :
  t = 14 :=
by
  repeat { sorry }

end bugs_meet_at_point_P_l124_124971


namespace cookies_difference_l124_124380

-- Define the initial conditions
def initial_cookies : ℝ := 57
def cookies_eaten : ℝ := 8.5
def cookies_bought : ℝ := 125.75

-- Problem statement
theorem cookies_difference (initial_cookies cookies_eaten cookies_bought : ℝ) : 
  cookies_bought - cookies_eaten = 117.25 := 
sorry

end cookies_difference_l124_124380


namespace max_possible_cables_l124_124508

theorem max_possible_cables (num_employees : ℕ) (num_brand_X : ℕ) (num_brand_Y : ℕ) 
  (max_connections : ℕ) (num_cables : ℕ) :
  num_employees = 40 →
  num_brand_X = 25 →
  num_brand_Y = 15 →
  max_connections = 3 →
  (∀ x : ℕ, x < max_connections → num_cables ≤ 3 * num_brand_Y) →
  num_cables = 45 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end max_possible_cables_l124_124508


namespace third_highest_score_l124_124010

theorem third_highest_score
  (mean15 : ℕ → ℚ) (mean12 : ℕ → ℚ) 
  (sum15 : ℕ) (sum12 : ℕ) (highest : ℕ) (third_highest : ℕ) (third_is_100: third_highest = 100) :
  (mean15 15 = 90) →
  (mean12 12 = 85) →
  (highest = 120) →
  (sum15 = 15 * 90) →
  (sum12 = 12 * 85) →
  (sum15 - sum12 = highest + 210) →
  third_highest = 100 := 
by
  intros hm15 hm12 hhigh hsum15 hsum12 hdiff
  sorry

end third_highest_score_l124_124010


namespace lines_are_parallel_l124_124816

-- Definitions of the conditions
variable (θ a p : Real)
def line1 := θ = a
def line2 := p * Real.sin (θ - a) = 1

-- The proof problem: Prove the two lines are parallel
theorem lines_are_parallel (h1 : line1 θ a) (h2 : line2 θ a p) : False :=
by
  sorry

end lines_are_parallel_l124_124816


namespace range_of_a_l124_124411

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a * x + 1 > 0) ↔ (-2 < a ∧ a < 2) :=
sorry

end range_of_a_l124_124411


namespace goods_train_speed_l124_124202

theorem goods_train_speed (train_length platform_length : ℝ) (time_sec : ℝ) : 
  train_length = 270.0416 ∧ platform_length = 250 ∧ time_sec = 26 → 
  (train_length + platform_length) / time_sec * 3.6 = 72.00576 :=
by
  sorry

end goods_train_speed_l124_124202


namespace point_B_coordinates_l124_124773

def move_up (x y : Int) (units : Int) : Int := y + units
def move_left (x y : Int) (units : Int) : Int := x - units

theorem point_B_coordinates :
  let A : Int × Int := (1, -1)
  let B : Int × Int := (move_left A.1 A.2 3, move_up A.1 A.2 2)
  B = (-2, 1) := 
by
  -- This is where the proof would go, but we omit it with "sorry"
  sorry

end point_B_coordinates_l124_124773


namespace total_logs_in_stack_l124_124300

/-- The total number of logs in a stack where the top row has 5 logs,
each succeeding row has one more log than the one above,
and the bottom row has 15 logs. -/
theorem total_logs_in_stack :
  let a := 5               -- first term (logs in the top row)
  let l := 15              -- last term (logs in the bottom row)
  let n := l - a + 1       -- number of terms (rows)
  let S := n / 2 * (a + l) -- sum of the arithmetic series
  S = 110 := sorry

end total_logs_in_stack_l124_124300


namespace integer_not_in_range_l124_124251

theorem integer_not_in_range (g : ℝ → ℤ) :
  (∀ x, x > -3 → g x = Int.ceil (2 / (x + 3))) ∧
  (∀ x, x < -3 → g x = Int.floor (2 / (x + 3))) →
  ∀ z : ℤ, (∃ x, g x = z) ↔ z ≠ 0 :=
by
  intros h z
  sorry

end integer_not_in_range_l124_124251


namespace expression_divisible_by_17_l124_124714

theorem expression_divisible_by_17 (n : ℕ) : 
  (6^(2*n) + 2^(n+2) + 12 * 2^n) % 17 = 0 :=
by
  sorry

end expression_divisible_by_17_l124_124714


namespace number_of_yellow_balls_l124_124641

theorem number_of_yellow_balls (x : ℕ) (h : (6 : ℝ) / (6 + x) = 0.3) : x = 14 :=
by
  sorry

end number_of_yellow_balls_l124_124641


namespace arc_length_l124_124552

theorem arc_length (r : ℝ) (α : ℝ) (h_r : r = 10) (h_α : α = 2 * Real.pi / 3) : 
  r * α = 20 * Real.pi / 3 := 
by {
sorry
}

end arc_length_l124_124552


namespace correct_sum_of_satisfying_values_l124_124792

def g (x : Nat) : Nat :=
  match x with
  | 0 => 0
  | 1 => 2
  | 2 => 1
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def f (x : Nat) : Nat :=
  match x with
  | 0 => 2
  | 1 => 1
  | 2 => 0
  | _ => 0  -- This handles the out-of-bounds case, though it's not needed here

def satisfies_condition (x : Nat) : Bool :=
  f (g x) > g (f x)

def sum_of_satisfying_values : Nat :=
  List.sum (List.filter satisfies_condition [0, 1, 2])

theorem correct_sum_of_satisfying_values : sum_of_satisfying_values = 2 :=
  sorry

end correct_sum_of_satisfying_values_l124_124792


namespace area_of_region_l124_124145

theorem area_of_region : ∃ A, (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y = 12 → A = 25 * Real.pi) :=
by
  -- Completing the square and identifying the circle
  -- We verify that the given equation represents a circle
  existsi (25 * Real.pi)
  intros x y h
  sorry

end area_of_region_l124_124145


namespace appropriate_sampling_methods_l124_124910

-- Conditions for the first survey
structure Population1 where
  high_income_families : Nat
  middle_income_families : Nat
  low_income_families : Nat
  total : Nat := high_income_families + middle_income_families + low_income_families

def survey1_population : Population1 :=
  { high_income_families := 125,
    middle_income_families := 200,
    low_income_families := 95
  }

-- Condition for the second survey
structure Population2 where
  art_specialized_students : Nat

def survey2_population : Population2 :=
  { art_specialized_students := 5 }

-- The main statement to prove
theorem appropriate_sampling_methods :
  (survey1_population.total >= 100 → stratified_sampling_for_survey1) ∧ 
  (survey2_population.art_specialized_students >= 3 → simple_random_sampling_for_survey2) :=
  sorry

end appropriate_sampling_methods_l124_124910


namespace leak_empty_tank_time_l124_124488

-- Definitions based on given conditions
def rate_A := 1 / 2 -- Rate of Pipe A (1 tank per 2 hours)
def rate_A_plus_L := 2 / 5 -- Combined rate of Pipe A and leak

-- Theorem states the time leak takes to empty full tank is 10 hours
theorem leak_empty_tank_time : 1 / (rate_A - rate_A_plus_L) = 10 :=
by
  -- Proof steps would go here
  sorry

end leak_empty_tank_time_l124_124488


namespace tan_product_identity_l124_124224

theorem tan_product_identity : (1 + Real.tan (Real.pi / 6)) * (1 + Real.tan (Real.pi / 3)) = 4 + 2 * Real.sqrt 3 :=
by
  sorry

end tan_product_identity_l124_124224


namespace commutative_not_associative_l124_124254

variable (k : ℝ) (h_k : 0 < k)

noncomputable def star (x y : ℝ) : ℝ := (x * y + k) / (x + y + k)

theorem commutative (x y : ℝ) (h_x : 0 < x) (h_y : 0 < y) :
  star k x y = star k y x :=
by sorry

theorem not_associative (x y z : ℝ) (h_x : 0 < x) (h_y : 0 < y) (h_z : 0 < z) :
  ¬(star k (star k x y) z = star k x (star k y z)) :=
by sorry

end commutative_not_associative_l124_124254


namespace train_speed_l124_124869

theorem train_speed (L V : ℝ) (h1 : L = V * 20) (h2 : L + 300.024 = V * 50) : V = 10.0008 :=
by
  sorry

end train_speed_l124_124869


namespace sin_diff_angle_identity_l124_124286

open Real

noncomputable def alpha : ℝ := sorry -- α is an obtuse angle

axiom h1 : 90 < alpha ∧ alpha < 180 -- α is an obtuse angle
axiom h2 : cos alpha = -3 / 5 -- given cosine value

theorem sin_diff_angle_identity :
  sin (π / 4 - alpha) = - (7 * sqrt 2) / 10 :=
by
  sorry

end sin_diff_angle_identity_l124_124286


namespace factorization_correct_l124_124398

theorem factorization_correct (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_correct_l124_124398


namespace sin_double_angle_second_quadrant_l124_124453

theorem sin_double_angle_second_quadrant (α : ℝ) (h1 : Real.cos α = -3/5) (h2 : α ∈ Set.Ioo (π / 2) π) :
    Real.sin (2 * α) = -24 / 25 := by
  sorry

end sin_double_angle_second_quadrant_l124_124453


namespace eight_letter_good_words_l124_124365

-- Definition of a good word sequence (only using A, B, and C)
inductive Letter
| A | B | C

-- Define the restriction condition for a good word
def is_valid_transition (a b : Letter) : Prop :=
  match a, b with
  | Letter.A, Letter.B => False
  | Letter.B, Letter.C => False
  | Letter.C, Letter.A => False
  | _, _ => True

-- Count the number of 8-letter good words
def count_good_words : ℕ :=
  let letters := [Letter.A, Letter.B, Letter.C]
  -- Initial 3 choices for the first letter
  let first_choices := letters.length
  -- Subsequent 7 letters each have 2 valid previous choices
  let subsequent_choices := 2 ^ 7
  first_choices * subsequent_choices

theorem eight_letter_good_words : count_good_words = 384 :=
by
  sorry

end eight_letter_good_words_l124_124365


namespace math_problem_l124_124989

noncomputable def base10_b := 25 + 1  -- 101_5 in base 10
noncomputable def base10_c := 343 + 98 + 21 + 4  -- 1234_7 in base 10
noncomputable def base10_d := 2187 + 324 + 45 + 6  -- 3456_9 in base 10

theorem math_problem (a : ℕ) (b c d : ℕ) (h_a : a = 2468)
  (h_b : b = base10_b) (h_c : c = base10_c) (h_d : d = base10_d) :
  (a / b) * c - d = 41708 :=
  by {
  sorry
}

end math_problem_l124_124989


namespace remainder_of_sum_mod_l124_124954

theorem remainder_of_sum_mod (n : ℤ) : ((7 + n) + (n + 5)) % 7 = (5 + 2 * n) % 7 :=
by
  sorry

end remainder_of_sum_mod_l124_124954


namespace mary_added_peanuts_l124_124240

-- Defining the initial number of peanuts
def initial_peanuts : ℕ := 4

-- Defining the final number of peanuts
def total_peanuts : ℕ := 10

-- Defining the number of peanuts added by Mary
def peanuts_added : ℕ := total_peanuts - initial_peanuts

-- The proof problem is to show that Mary added 6 peanuts
theorem mary_added_peanuts : peanuts_added = 6 :=
by
  -- We leave the proof part as a sorry as per instruction
  sorry

end mary_added_peanuts_l124_124240


namespace min_sticks_to_avoid_rectangles_l124_124448

noncomputable def min_stick_deletions (n : ℕ) : ℕ :=
  if n = 8 then 43 else 0 -- we define 43 as the minimum for an 8x8 chessboard

theorem min_sticks_to_avoid_rectangles : min_stick_deletions 8 = 43 :=
  by
    sorry

end min_sticks_to_avoid_rectangles_l124_124448


namespace find_A_l124_124247

variable (x A B C : ℝ)

theorem find_A :
  (∃ A B C : ℝ, (∀ x : ℝ, x ≠ -3 ∧ x ≠ 2 → 
  (1 / (x^3 + 2 * x^2 - 19 * x - 30) = 
  (A / (x + 3)) + (B / (x - 2)) + (C / (x - 2)^2)) ∧ 
  A = 1 / 25)) :=
by
  sorry

end find_A_l124_124247


namespace library_science_books_count_l124_124749

-- Definitions based on the problem conditions
def initial_science_books := 120
def borrowed_books := 40
def returned_books := 15
def books_on_hold := 10
def borrowed_from_other_library := 20
def lost_books := 2
def damaged_books := 1

-- Statement for the proof.
theorem library_science_books_count :
  initial_science_books - borrowed_books + returned_books - books_on_hold + borrowed_from_other_library - lost_books - damaged_books = 102 :=
by
  sorry

end library_science_books_count_l124_124749


namespace question_1_part_1_question_1_part_2_question_2_l124_124088

universe u

variables (U : Type u) [PartialOrder U]
noncomputable def A : Set ℝ := {x | (x - 2) * (x - 9) < 0}
noncomputable def B : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
noncomputable def C (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ 2 - a }

theorem question_1_part_1 : A ∩ B = {x | 2 < x ∧ x ≤ 5} :=
sorry

theorem question_1_part_2 : B ∪ (Set.compl A) = {x | x ≤ 5 ∨ x ≥ 9} :=
sorry

theorem question_2 (a : ℝ) (h : C a ∪ (Set.compl B) = Set.univ) : a ≤ -3 :=
sorry

end question_1_part_1_question_1_part_2_question_2_l124_124088


namespace find_f_neg_2017_l124_124657

-- Define f as given in the problem
def f (a b x : ℝ) : ℝ := a * x^3 + b * x - 2

-- State the given problem condition
def condition (a b : ℝ) : Prop :=
  f a b 2017 = 10

-- The main problem statement proving the solution
theorem find_f_neg_2017 (a b : ℝ) (h : condition a b) : f a b (-2017) = -14 :=
by
  -- We state this theorem and provide a sorry to skip the proof
  sorry

end find_f_neg_2017_l124_124657


namespace danny_initial_caps_l124_124731

-- Define the conditions
variables (lostCaps : ℕ) (currentCaps : ℕ)
-- Assume given conditions
axiom lost_caps_condition : lostCaps = 66
axiom current_caps_condition : currentCaps = 25

-- Define the total number of bottle caps Danny had at first
def originalCaps (lostCaps currentCaps : ℕ) : ℕ := lostCaps + currentCaps

-- State the theorem to prove the number of bottle caps Danny originally had is 91
theorem danny_initial_caps : originalCaps lostCaps currentCaps = 91 :=
by
  -- Insert the proof here when available
  sorry

end danny_initial_caps_l124_124731


namespace mixture_ratio_l124_124911

theorem mixture_ratio (V : ℝ) (a b c : ℕ)
  (h_pos : V > 0)
  (h_ratio : V = (3/8) * V + (5/11) * V + ((88 - 33 - 40)/88) * V) :
  a = 33 ∧ b = 40 ∧ c = 15 :=
by
  sorry

end mixture_ratio_l124_124911


namespace total_weight_of_envelopes_l124_124992

theorem total_weight_of_envelopes :
  (8.5 * 880 / 1000) = 7.48 :=
by
  sorry

end total_weight_of_envelopes_l124_124992


namespace expand_polynomial_eq_l124_124050

theorem expand_polynomial_eq :
  (3 * t^3 - 2 * t^2 + 4 * t - 1) * (2 * t^2 - 5 * t + 3) = 6 * t^5 - 19 * t^4 + 27 * t^3 - 28 * t^2 + 17 * t - 3 :=
by
  sorry

end expand_polynomial_eq_l124_124050


namespace perfect_square_461_l124_124551

theorem perfect_square_461 (x : ℤ) (y : ℤ) (hx : 5 ∣ x) (hy : 5 ∣ y) 
  (h : x^2 + 461 = y^2) : x^2 = 52900 :=
  sorry

end perfect_square_461_l124_124551


namespace rectangle_area_l124_124878

theorem rectangle_area (w d : ℝ) 
  (h1 : d = (w^2 + (3 * w)^2) ^ (1/2))
  (h2 : ∃ A : ℝ, A = w * 3 * w) :
  ∃ A : ℝ, A = 3 * (d^2 / 10) := 
by {
  sorry
}

end rectangle_area_l124_124878


namespace zach_needs_more_money_l124_124432

theorem zach_needs_more_money
  (bike_cost : ℕ) (allowance : ℕ) (mowing_payment : ℕ) (babysitting_rate : ℕ) 
  (current_savings : ℕ) (babysitting_hours : ℕ) :
  bike_cost = 100 →
  allowance = 5 →
  mowing_payment = 10 →
  babysitting_rate = 7 →
  current_savings = 65 →
  babysitting_hours = 2 →
  (bike_cost - (current_savings + (allowance + mowing_payment + babysitting_hours * babysitting_rate))) = 6 :=
by
  sorry

end zach_needs_more_money_l124_124432


namespace airplane_distance_difference_l124_124462

variable (a : ℝ)

theorem airplane_distance_difference :
  let wind_speed := 20
  (4 * a) - (3 * (a - wind_speed)) = a + 60 := by
  sorry

end airplane_distance_difference_l124_124462


namespace find_resistance_x_l124_124322

theorem find_resistance_x (y r x : ℝ) (h₁ : y = 5) (h₂ : r = 1.875) (h₃ : 1/r = 1/x + 1/y) : x = 3 :=
by
  sorry

end find_resistance_x_l124_124322


namespace find_n_l124_124027

theorem find_n (n : ℕ) (h1 : 0 ≤ n) (h2 : n < 101) (h3 : 100 * n % 101 = 72) : n = 29 := 
by
  sorry

end find_n_l124_124027


namespace pasta_ratio_l124_124420

theorem pasta_ratio (total_students : ℕ) (spaghetti : ℕ) (manicotti : ℕ) 
  (h1 : total_students = 650) 
  (h2 : spaghetti = 250) 
  (h3 : manicotti = 100) : 
  (spaghetti : ℤ) / (manicotti : ℤ) = 5 / 2 :=
by
  sorry

end pasta_ratio_l124_124420


namespace lisa_needs_4_weeks_to_eat_all_candies_l124_124078

-- Define the number of candies Lisa has initially.
def candies_initial : ℕ := 72

-- Define the number of candies Lisa eats per week based on the given conditions.
def candies_per_week : ℕ := (3 * 2) + (2 * 2) + (4 * 2) + 1

-- Define the number of weeks it takes for Lisa to eat all the candies.
def weeks_to_eat_all_candies (candies : ℕ) (weekly_candies : ℕ) : ℕ := 
  (candies + weekly_candies - 1) / weekly_candies

-- The theorem statement that proves Lisa needs 4 weeks to eat all 72 candies.
theorem lisa_needs_4_weeks_to_eat_all_candies :
  weeks_to_eat_all_candies candies_initial candies_per_week = 4 :=
by
  sorry

end lisa_needs_4_weeks_to_eat_all_candies_l124_124078


namespace find_side_length_l124_124994

theorem find_side_length
  (n : ℕ) 
  (h : (6 * n^2) / (6 * n^3) = 1 / 3) : 
  n = 3 := 
by
  sorry

end find_side_length_l124_124994


namespace digit_in_452nd_place_l124_124394

def repeating_sequence : List Nat := [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7]
def repeat_length : Nat := 18

theorem digit_in_452nd_place :
  (repeating_sequence.get ⟨(452 % repeat_length) - 1, sorry⟩ = 6) :=
sorry

end digit_in_452nd_place_l124_124394


namespace find_prime_squares_l124_124024

def is_prime (n : ℕ) : Prop := 
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_square (n : ℕ) : Prop := 
  ∃ k : ℕ, k * k = n

theorem find_prime_squares :
  ∀ (p q : ℕ), is_prime p → is_prime q → is_square (p^(q+1) + q^(p+1)) → (p = 2 ∧ q = 2) :=
by 
  intros p q h_prime_p h_prime_q h_square
  sorry

end find_prime_squares_l124_124024


namespace number_of_moles_of_OC_NH2_2_formed_l124_124757

-- Definition: Chemical reaction condition
def reaction_eqn (x y : ℕ) : Prop := 
  x ≥ 1 ∧ y ≥ 2 ∧ x * 2 = y

-- Theorem: Prove that combining 3 moles of CO2 and 6 moles of NH3 results in 3 moles of OC(NH2)2
theorem number_of_moles_of_OC_NH2_2_formed (x y : ℕ) 
(h₁ : reaction_eqn x y)
(h₂ : x = 3)
(h₃ : y = 6) : 
x =  y / 2 :=
by {
    -- Proof is not provided
    sorry 
}

end number_of_moles_of_OC_NH2_2_formed_l124_124757


namespace inequality_proof_l124_124732

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l124_124732


namespace find_deepaks_age_l124_124149

variable (R D : ℕ)

theorem find_deepaks_age
  (h1 : R / D = 4 / 3)
  (h2 : R + 2 = 26) :
  D = 18 := by
  sorry

end find_deepaks_age_l124_124149


namespace volume_of_sphere_l124_124237

theorem volume_of_sphere (r : ℝ) (h : r = 3) : (4 / 3) * π * r ^ 3 = 36 * π := 
by
  sorry

end volume_of_sphere_l124_124237


namespace total_money_received_l124_124049

-- Define the given prices and quantities
def adult_ticket_price : ℕ := 12
def child_ticket_price : ℕ := 4
def adult_tickets_sold : ℕ := 90
def child_tickets_sold : ℕ := 40

-- Define the theorem to prove the total amount received
theorem total_money_received :
  (adult_ticket_price * adult_tickets_sold + child_ticket_price * child_tickets_sold) = 1240 :=
by
  -- Proof goes here
  sorry

end total_money_received_l124_124049


namespace inequality_solution_l124_124524

noncomputable def g (x : ℝ) : ℝ := (3 * x - 8) * (x - 4) * (x + 1) / (x - 2)

theorem inequality_solution :
  { x : ℝ | g x ≥ 0 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | 2 < x ∧ x ≤ 8/3 } ∪ { x : ℝ | 4 ≤ x } :=
by sorry

end inequality_solution_l124_124524


namespace sum_of_coordinates_D_l124_124997

theorem sum_of_coordinates_D (x y : Int) :
  let N := (4, 10)
  let C := (14, 6)
  let D := (x, y)
  N = ((x + 14) / 2, (y + 6) / 2) →
  x + y = 8 :=
by
  intros
  sorry

end sum_of_coordinates_D_l124_124997


namespace value_of_2_pow_5_plus_5_l124_124993

theorem value_of_2_pow_5_plus_5 : 2^5 + 5 = 37 := by
  sorry

end value_of_2_pow_5_plus_5_l124_124993


namespace sum_of_numbers_l124_124511

theorem sum_of_numbers (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 8)
  (h4 : (a + b + c) / 3 = a + 12) (h5 : (a + b + c) / 3 = c - 20) :
  a + b + c = 48 :=
sorry

end sum_of_numbers_l124_124511


namespace symmetry_about_origin_l124_124331

theorem symmetry_about_origin (m : ℝ) (A B : ℝ × ℝ) (hA : A = (2, -1)) (hB : B = (-2, m)) (h_sym : B = (-A.1, -A.2)) :
  m = 1 :=
by
  sorry

end symmetry_about_origin_l124_124331


namespace remainder_3_101_add_5_mod_11_l124_124810

theorem remainder_3_101_add_5_mod_11 : (3 ^ 101 + 5) % 11 = 8 := 
by sorry

end remainder_3_101_add_5_mod_11_l124_124810


namespace solve_equation_solve_inequality_system_l124_124243

theorem solve_equation :
  ∃ x, 2 * x^2 - 4 * x - 1 = 0 ∧ (x = (2 + Real.sqrt 6) / 2 ∨ x = (2 - Real.sqrt 6) / 2) :=
sorry

theorem solve_inequality_system : 
  ∀ x, (2 * x + 3 > 1 → -1 < x) ∧
       (x - 2 ≤ (1 / 2) * (x + 2) → x ≤ 6) ∧ 
       (2 * x + 3 > 1 ∧ x - 2 ≤ (1 / 2) * (x + 2) ↔ (-1 < x ∧ x ≤ 6)) :=
sorry

end solve_equation_solve_inequality_system_l124_124243


namespace maximize_profit_at_200_l124_124446

noncomputable def cost (q : ℝ) : ℝ := 50000 + 200 * q
noncomputable def price (q : ℝ) : ℝ := 24200 - (1/5) * q^2
noncomputable def profit (q : ℝ) : ℝ := (price q) * q - (cost q)

theorem maximize_profit_at_200 : ∃ (q : ℝ), q = 200 ∧ ∀ (x : ℝ), x ≥ 0 → profit q ≥ profit x :=
by
  sorry

end maximize_profit_at_200_l124_124446


namespace expression_evaluation_l124_124227

def evaluate_expression : ℝ := (-1) ^ 51 + 3 ^ (2^3 + 5^2 - 7^2)

theorem expression_evaluation :
  evaluate_expression = -1 + (1 / 43046721) :=
by
  sorry

end expression_evaluation_l124_124227


namespace edward_rides_l124_124076

theorem edward_rides (total_tickets tickets_spent tickets_per_ride rides : ℕ)
    (h1 : total_tickets = 79)
    (h2 : tickets_spent = 23)
    (h3 : tickets_per_ride = 7)
    (h4 : rides = (total_tickets - tickets_spent) / tickets_per_ride) :
    rides = 8 := by sorry

end edward_rides_l124_124076


namespace right_triangle_hypotenuse_length_l124_124796

theorem right_triangle_hypotenuse_length :
  ∀ (a b h : ℕ), a = 15 → b = 36 → h^2 = a^2 + b^2 → h = 39 :=
by
  intros a b h ha hb hyp
  -- In the proof, we would use ha, hb, and hyp to show h = 39
  sorry

end right_triangle_hypotenuse_length_l124_124796


namespace abc_inequality_l124_124020

theorem abc_inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  (a * (a^2 + b * c)) / (b + c) + (b * (b^2 + c * a)) / (c + a) + (c * (c^2 + a * b)) / (a + b) ≥ a * b + b * c + c * a := 
by 
  sorry

end abc_inequality_l124_124020


namespace union_complement_eq_l124_124706

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 4}
def N : Set ℕ := {2, 5}

theorem union_complement_eq : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_eq_l124_124706


namespace parallel_vectors_xy_sum_l124_124455

theorem parallel_vectors_xy_sum (x y : ℚ) (k : ℚ) 
  (h1 : (2, 4, -5) = (2 * k, 4 * k, -5 * k)) 
  (h2 : (3, x, y) = (2 * k, 4 * k, -5 * k)) 
  (h3 : 3 = 2 * k) : 
  x + y = -3 / 2 :=
by
  sorry

end parallel_vectors_xy_sum_l124_124455


namespace count_5_numbers_after_996_l124_124963

theorem count_5_numbers_after_996 : 
  ∃ a b c d e, a = 997 ∧ b = 998 ∧ c = 999 ∧ d = 1000 ∧ e = 1001 :=
sorry

end count_5_numbers_after_996_l124_124963


namespace smallest_integer_l124_124677

-- Define a function to calculate the LCM of a list of numbers
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

-- List of divisors
def divisors : List ℕ := [4, 5, 6, 7, 8, 9, 10]

-- Calculating the required integer
noncomputable def required_integer : ℕ := lcm_list divisors + 1

-- The proof statement
theorem smallest_integer : required_integer = 2521 :=
  by 
  sorry

end smallest_integer_l124_124677


namespace ball_reaches_20_feet_at_1_75_seconds_l124_124048

noncomputable def ball_height (t : ℝ) : ℝ :=
  60 - 9 * t - 8 * t ^ 2

theorem ball_reaches_20_feet_at_1_75_seconds :
  ∃ t : ℝ, ball_height t = 20 ∧ t = 1.75 ∧ t ≥ 0 :=
by {
  sorry
}

end ball_reaches_20_feet_at_1_75_seconds_l124_124048


namespace smallest_fraction_l124_124319

theorem smallest_fraction (x : ℝ) (h : x > 2022) :
  min (min (min (min (x / 2022) (2022 / (x - 1))) ((x + 1) / 2022)) (2022 / x)) (2022 / (x + 1)) = 2022 / (x + 1) :=
sorry

end smallest_fraction_l124_124319


namespace february_max_diff_percentage_l124_124961

noncomputable def max_diff_percentage (D B F : ℕ) : ℚ :=
  let avg_others := (B + F) / 2
  let high_sales := max (max D B) F
  (high_sales - avg_others) / avg_others * 100

theorem february_max_diff_percentage :
  max_diff_percentage 8 5 6 = 45.45 := by
  sorry

end february_max_diff_percentage_l124_124961


namespace product_of_p_r_s_l124_124861

theorem product_of_p_r_s
  (p r s : ℕ)
  (h1 : 3^p + 3^4 = 90)
  (h2 : 2^r + 44 = 76)
  (h3 : 5^3 + 6^s = 1421) :
  p * r * s = 40 := 
sorry

end product_of_p_r_s_l124_124861


namespace total_present_ages_l124_124884

variable (P Q P' Q' : ℕ)

-- Condition 1: 6 years ago, \( p \) was half of \( q \) in age.
axiom cond1 : P = Q / 2

-- Condition 2: The ratio of their present ages is 3:4.
axiom cond2 : (P + 6) * 4 = (Q + 6) * 3

-- We need to prove: the total of their present ages is 21
theorem total_present_ages : P' + Q' = 21 :=
by
  -- We already have the variables and axioms in the context, so we just need to state the goal
  sorry

end total_present_ages_l124_124884


namespace orchestra_members_l124_124604

theorem orchestra_members (n : ℕ) (h₀ : 100 ≤ n) (h₁ : n ≤ 300)
    (h₂ : n % 4 = 3) (h₃ : n % 5 = 1) (h₄ : n % 7 = 5) : n = 231 := by
  sorry

end orchestra_members_l124_124604


namespace simplify_expression_as_single_fraction_l124_124626

variable (d : ℚ)

theorem simplify_expression_as_single_fraction :
  (5 + 4*d)/9 + 3 = (32 + 4*d)/9 := 
by
  sorry

end simplify_expression_as_single_fraction_l124_124626


namespace arithmetic_seq_of_equal_roots_l124_124288

theorem arithmetic_seq_of_equal_roots (a b c : ℝ) (h : b ≠ 0) 
    (h_eq_roots : ∃ x, b*x^2 - 4*b*x + 2*(a + c) = 0 ∧ (∀ y, b*y^2 - 4*b*y + 2*(a + c) = 0 → x = y)) : 
    b - a = c - b := 
by 
  -- placeholder for proof body
  sorry

end arithmetic_seq_of_equal_roots_l124_124288


namespace polynomial_relation_l124_124130

variables {a b c : ℝ}

theorem polynomial_relation
  (h1: a ≠ 0) (h2: b ≠ 0) (h3: c ≠ 0) (h4: a + b + c = 0) :
  ((a^7 + b^7 + c^7)^2) / ((a^2 + b^2 + c^2) * (a^3 + b^3 + c^3) * (a^4 + b^4 + c^4) * (a^5 + b^5 + c^5)) = 49 / 60 :=
sorry

end polynomial_relation_l124_124130


namespace number_of_triangles_l124_124525

open Nat

-- Define the number of combinations
def comb : Nat → Nat → Nat
  | n, k => if k > n then 0 else n.choose k

-- The given conditions
def points_on_OA := 5
def points_on_OB := 6
def point_O := 1
def total_points := points_on_OA + points_on_OB + point_O -- should equal 12

-- Lean proof problem statement
theorem number_of_triangles : comb total_points 3 - comb points_on_OA 3 - comb points_on_OB 3 = 165 := by
  sorry

end number_of_triangles_l124_124525


namespace given_condition_required_solution_l124_124016

-- Define the polynomial f.
noncomputable def f (x : ℝ) : ℝ := x^2 + x - 6

-- Given condition
theorem given_condition (x : ℝ) : f (x^2 + 2) = x^4 + 5 * x^2 := by sorry

-- Proving the required equivalence
theorem required_solution (x : ℝ) : f (x^2 - 2) = x^4 - 3 * x^2 - 4 := by sorry

end given_condition_required_solution_l124_124016


namespace car_trip_time_l124_124974

theorem car_trip_time (T A : ℕ) (h1 : 50 * T = 140 + 53 * A) (h2 : T = 4 + A) : T = 24 := by
  sorry

end car_trip_time_l124_124974


namespace triangle_inequality_l124_124959

open Real

theorem triangle_inequality (A B C : ℝ) (hA : 0 < A ∧ A < π) (hB : 0 < B ∧ B < π) (hC : 0 < C ∧ C < π) (h_sum : A + B + C = π) :
  sin A * cos C + A * cos B > 0 :=
by
  sorry

end triangle_inequality_l124_124959


namespace find_f_6_5_l124_124779

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

noncomputable def f : ℝ → ℝ := sorry

axiom even_f : is_even_function f
axiom periodic_f : ∀ x, f (x + 4) = f x
axiom f_in_interval : ∀ x, 1 ≤ x ∧ x ≤ 2 → f x = x - 2

theorem find_f_6_5 : f 6.5 = -0.5 := by
  sorry

end find_f_6_5_l124_124779


namespace complement_of_P_with_respect_to_U_l124_124354

universe u

def U : Set ℤ := {-1, 0, 1, 2}

def P : Set ℤ := {x | x * x < 2}

theorem complement_of_P_with_respect_to_U : U \ P = {2} :=
by
  sorry

end complement_of_P_with_respect_to_U_l124_124354


namespace find_x_l124_124607

theorem find_x (x y z w : ℕ) (h1 : x = y + 8) (h2 : y = z + 15) (h3 : z = w + 25) (h4 : w = 90) : x = 138 :=
by
  sorry

end find_x_l124_124607


namespace ratio_of_a_over_3_to_b_over_2_l124_124431

theorem ratio_of_a_over_3_to_b_over_2 (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : a * b ≠ 0) : (a / 3) / (b / 2) = 1 :=
by
  sorry

end ratio_of_a_over_3_to_b_over_2_l124_124431


namespace min_orange_chips_l124_124450

theorem min_orange_chips (p g o : ℕ)
    (h1: g ≥ (1 / 3) * p)
    (h2: g ≤ (1 / 4) * o)
    (h3: p + g ≥ 75) : o = 76 :=
    sorry

end min_orange_chips_l124_124450


namespace min_distance_symmetry_l124_124265

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^2 + x + 1

def line (x y : ℝ) : Prop := 2 * x - y = 3

theorem min_distance_symmetry :
  ∀ (P Q : ℝ × ℝ),
    line P.1 P.2 → line Q.1 Q.2 →
    (exists (x : ℝ), P = (x, f x)) ∧
    (exists (x : ℝ), Q = (x, f x)) →
    ∃ (d : ℝ), d = 2 * Real.sqrt 5 :=
sorry

end min_distance_symmetry_l124_124265


namespace initial_pants_l124_124309

theorem initial_pants (pairs_per_year : ℕ) (pants_per_pair : ℕ) (years : ℕ) (total_pants : ℕ) 
  (h1 : pairs_per_year = 4) (h2 : pants_per_pair = 2) (h3 : years = 5) (h4 : total_pants = 90) : 
  ∃ (initial_pants : ℕ), initial_pants = total_pants - (pairs_per_year * pants_per_pair * years) :=
by
  use 50
  sorry

end initial_pants_l124_124309


namespace y_coord_vertex_of_parabola_l124_124591

-- Define the quadratic equation of the parabola
def parabola (x : ℝ) : ℝ := 2 * x^2 + 16 * x + 29

-- Statement to prove
theorem y_coord_vertex_of_parabola : ∃ (x : ℝ), parabola x = 2 * (x + 4)^2 - 3 := sorry

end y_coord_vertex_of_parabola_l124_124591


namespace linda_savings_l124_124659

theorem linda_savings :
  ∀ (S : ℝ), (5 / 6 * S + 500 = S) → S = 3000 :=
by
  intros S h
  sorry

end linda_savings_l124_124659


namespace find_u_plus_v_l124_124146

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 7 * v = 17) (h2 : 5 * u + 3 * v = 1) : 
  u + v = - 6 / 11 :=
  sorry

end find_u_plus_v_l124_124146


namespace exam_papers_count_l124_124549

theorem exam_papers_count (F x : ℝ) :
  (∀ n : ℕ, n = 5) →    -- condition 1: equivalence of n to proportions count
  (6 * x + 7 * x + 8 * x + 9 * x + 10 * x = 40 * x) →    -- condition 2: sum of proportions
  (40 * x = 0.60 * n * F) →   -- condition 3: student obtained 60% of total marks
  (7 * x > 0.50 * F ∧ 8 * x > 0.50 * F ∧ 9 * x > 0.50 * F ∧ 10 * x > 0.50 * F ∧ 6 * x ≤ 0.50 * F) →  -- condition 4: more than 50% in 4 papers
  ∃ n : ℕ, n = 5 :=    -- prove: number of papers is 5
sorry

end exam_papers_count_l124_124549


namespace cost_of_paving_l124_124804

theorem cost_of_paving (L W R : ℝ) (hL : L = 6.5) (hW : W = 2.75) (hR : R = 600) : 
  L * W * R = 10725 := by
  rw [hL, hW, hR]
  -- To solve the theorem successively
  -- we would need to verify the product of the values
  -- given by the conditions.
  sorry

end cost_of_paving_l124_124804


namespace plane_split_into_regions_l124_124028

theorem plane_split_into_regions : 
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  ∃ regions : ℕ, regions = 7 :=
by
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  existsi 7
  sorry

end plane_split_into_regions_l124_124028


namespace minimum_m_value_l124_124148

theorem minimum_m_value (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : 24 * m = n^4) : m = 54 := sorry

end minimum_m_value_l124_124148


namespace right_triangle_third_side_product_l124_124159

theorem right_triangle_third_side_product :
  ∀ (a b : ℝ), (a = 6 ∧ b = 8 ∧ (a^2 + b^2 = c^2 ∨ a^2 = b^2 - c^2)) →
  (a * b = 53.0) :=
by
  intros a b h
  sorry

end right_triangle_third_side_product_l124_124159


namespace fruits_eaten_l124_124603

theorem fruits_eaten (initial_cherries initial_strawberries initial_blueberries left_cherries left_strawberries left_blueberries : ℕ)
  (h1 : initial_cherries = 16) (h2 : initial_strawberries = 10) (h3 : initial_blueberries = 20)
  (h4 : left_cherries = 6) (h5 : left_strawberries = 8) (h6 : left_blueberries = 15) :
  (initial_cherries - left_cherries) + (initial_strawberries - left_strawberries) + (initial_blueberries - left_blueberries) = 17 := 
by
  sorry

end fruits_eaten_l124_124603


namespace triangle_area_l124_124261

-- Define the lines and the x-axis
noncomputable def line1 (x : ℝ) : ℝ := 2 * x + 1
noncomputable def line2 (x : ℝ) : ℝ := 1 - 5 * x
noncomputable def x_axis (x : ℝ) : ℝ := 0

-- Define intersection points
noncomputable def intersect_x_axis1 : ℝ × ℝ := (-1 / 2, 0)
noncomputable def intersect_x_axis2 : ℝ × ℝ := (1 / 5, 0)
noncomputable def intersect_lines : ℝ × ℝ := (0, 1)

-- State the theorem for the area of the triangle
theorem triangle_area : 
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  (1 / 2) * d * h = 7 / 20 := 
by
  let d := abs (intersect_x_axis1.1 - intersect_x_axis2.1)
  let h := intersect_lines.2 
  sorry

end triangle_area_l124_124261


namespace maurice_earnings_l124_124588

theorem maurice_earnings (bonus_per_10_tasks : ℕ → ℕ) (num_tasks : ℕ) (total_earnings : ℕ) :
  (∀ n, n * (bonus_per_10_tasks n) = 6 * n) →
  num_tasks = 30 →
  total_earnings = 78 →
  bonus_per_10_tasks num_tasks / 10 = 3 →
  (total_earnings - (bonus_per_10_tasks num_tasks / 10) * 6) / num_tasks = 2 :=
by
  intros h_bonus h_num_tasks h_total_earnings h_bonus_count
  sorry

end maurice_earnings_l124_124588


namespace second_root_of_quadratic_l124_124395

theorem second_root_of_quadratic (p q r : ℝ) (quad_eqn : ∀ x, 2 * p * (q - r) * x^2 + 3 * q * (r - p) * x + 4 * r * (p - q) = 0) (root : 2 * p * (q - r) * 2^2 + 3 * q * (r - p) * 2 + 4 * r * (p - q) = 0) :
    ∃ r₂ : ℝ, r₂ = (r * (p - q)) / (p * (q - r)) :=
sorry

end second_root_of_quadratic_l124_124395


namespace angle_A_is_120_degrees_l124_124880

theorem angle_A_is_120_degrees
  (b c l_a : ℝ)
  (h : (1 / b) + (1 / c) = 1 / l_a) :
  ∃ A : ℝ, A = 120 :=
by
  sorry

end angle_A_is_120_degrees_l124_124880


namespace part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l124_124386

theorem part_a_max_cells_crossed (m n : ℕ) : 
  ∃ max_cells : ℕ, max_cells = m + n - 1 := sorry

theorem part_b_max_cells_crossed_by_needle : 
  ∃ max_cells : ℕ, max_cells = 285 := sorry

end part_a_max_cells_crossed_part_b_max_cells_crossed_by_needle_l124_124386


namespace proof_time_to_run_square_field_l124_124366

def side : ℝ := 40
def speed_kmh : ℝ := 9
def perimeter (side : ℝ) : ℝ := 4 * side

noncomputable def speed_mps (speed_kmh : ℝ) : ℝ := speed_kmh * (1000 / 3600)

noncomputable def time_to_run (perimeter : ℝ) (speed_mps : ℝ) : ℝ := perimeter / speed_mps

theorem proof_time_to_run_square_field :
  time_to_run (perimeter side) (speed_mps speed_kmh) = 64 :=
by
  sorry

end proof_time_to_run_square_field_l124_124366


namespace compare_2_roses_3_carnations_l124_124161

variable (x y : ℝ)

def condition1 : Prop := 6 * x + 3 * y > 24
def condition2 : Prop := 4 * x + 5 * y < 22

theorem compare_2_roses_3_carnations (h1 : condition1 x y) (h2 : condition2 x y) : 2 * x > 3 * y := sorry

end compare_2_roses_3_carnations_l124_124161


namespace find_J_salary_l124_124776

variable (J F M A : ℝ)

theorem find_J_salary (h1 : (J + F + M + A) / 4 = 8000) (h2 : (F + M + A + 6500) / 4 = 8900) :
  J = 2900 := by
  sorry

end find_J_salary_l124_124776


namespace terminating_fraction_count_l124_124258

theorem terminating_fraction_count : 
  (∃ (n : ℕ), 1 ≤ n ∧ n ≤ 299 ∧ (∃ k, n = 3 * k)) ∧ 
  (∃ (count : ℕ), count = 99) :=
by
  sorry

end terminating_fraction_count_l124_124258


namespace vector_perpendicular_l124_124900

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 3)
def vec_diff : ℝ × ℝ := (vec_a.1 - vec_b.1, vec_a.2 - vec_b.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_perpendicular :
  dot_product vec_a vec_diff = 0 := by
  sorry

end vector_perpendicular_l124_124900


namespace surface_area_is_correct_l124_124777

structure CubicSolid where
  base_layer : ℕ
  second_layer : ℕ
  third_layer : ℕ
  top_layer : ℕ

def conditions : CubicSolid := ⟨4, 4, 3, 1⟩

theorem surface_area_is_correct : 
  (conditions.base_layer + conditions.second_layer + conditions.third_layer + conditions.top_layer + 7 + 7 + 3 + 3) = 28 := 
  by
  sorry

end surface_area_is_correct_l124_124777


namespace x5_plus_y5_l124_124257

theorem x5_plus_y5 (x y : ℝ) 
  (h1 : x + y = 3) 
  (h2 : 1 / (x + y^2) + 1 / (x^2 + y) = 1 / 2) : 
  x^5 + y^5 = 252 :=
by
  -- Placeholder for the proof
  sorry

end x5_plus_y5_l124_124257


namespace least_k_square_divisible_by_240_l124_124690

theorem least_k_square_divisible_by_240 (k : ℕ) (h : ∃ m : ℕ, k ^ 2 = 240 * m) : k ≥ 60 :=
by
  sorry

end least_k_square_divisible_by_240_l124_124690


namespace infinite_geometric_series_sum_l124_124767

theorem infinite_geometric_series_sum (a : ℕ → ℝ) (a1 : a 1 = 1) (r : ℝ) (h : r = 1 / 3) (S : ℝ) (H : S = a 1 / (1 - r)) : S = 3 / 2 :=
by
  sorry

end infinite_geometric_series_sum_l124_124767


namespace remainder_sum_div_l124_124819

theorem remainder_sum_div (n : ℤ) : ((9 - n) + (n + 5)) % 9 = 5 := by
  sorry

end remainder_sum_div_l124_124819


namespace difference_of_cubes_l124_124582

theorem difference_of_cubes (x y : ℕ) (h1 : x = y + 3) (h2 : x + y = 5) : x^3 - y^3 = 63 :=
by sorry

end difference_of_cubes_l124_124582


namespace verify_quadratic_eq_l124_124171

def is_quadratic (eq : String) : Prop :=
  eq = "ax^2 + bx + c = 0"

theorem verify_quadratic_eq :
  is_quadratic "x^2 - 1 = 0" :=
by
  -- Auxiliary functions or steps can be introduced if necessary, but proof is omitted here.
  sorry

end verify_quadratic_eq_l124_124171


namespace correlation_statements_l124_124497

def heavy_snow_predicts_harvest_year (heavy_snow benefits_wheat : Prop) : Prop := benefits_wheat → heavy_snow
def great_teachers_produce_students (great_teachers outstanding_students : Prop) : Prop := great_teachers → outstanding_students
def smoking_is_harmful (smoking harmful_to_health : Prop) : Prop := smoking → harmful_to_health
def magpies_call_signifies_joy (magpies_call joy_signified : Prop) : Prop := joy_signified → magpies_call

theorem correlation_statements (heavy_snow benefits_wheat great_teachers outstanding_students smoking harmful_to_health magpies_call joy_signified : Prop)
  (H1 : heavy_snow_predicts_harvest_year heavy_snow benefits_wheat)
  (H2 : great_teachers_produce_students great_teachers outstanding_students)
  (H3 : smoking_is_harmful smoking harmful_to_health) :
  ¬ magpies_call_signifies_joy magpies_call joy_signified := sorry

end correlation_statements_l124_124497


namespace eccentricity_of_hyperbola_l124_124701

variable {a b c e : ℝ}
variable (h_hyperbola : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1)
variable (ha_pos : a > 0)
variable (hb_pos : b > 0)
variable (h_vertices : A1 = (-a, 0) ∧ A2 = (a, 0))
variable (h_imaginary_axis : B1 = (0, b) ∧ B2 = (0, -b))
variable (h_foci : F1 = (-c, 0) ∧ F2 = (c, 0))
variable (h_relation : a^2 + b^2 = c^2)
variable (h_tangent_circle : ∀ d, (d = 2*a) → (tangent (circle d) (rhombus F1 B1 F2 B2)))

theorem eccentricity_of_hyperbola : e = (1 + Real.sqrt 5) / 2 :=
sorry

end eccentricity_of_hyperbola_l124_124701


namespace find_unique_function_l124_124940

theorem find_unique_function (f : ℝ → ℝ) (hf1 : ∀ x, 0 ≤ x → 0 ≤ f x)
    (hf2 : ∀ x, 0 ≤ x → f (f x) + f x = 12 * x) :
    ∀ x, 0 ≤ x → f x = 3 * x := 
  sorry

end find_unique_function_l124_124940


namespace percentage_calculation_l124_124656

-- Definitions based on conditions
def x : ℕ := 5200
def p1 : ℚ := 0.50
def p2 : ℚ := 0.30
def p3 : ℚ := 0.15

-- The theorem stating the desired proof
theorem percentage_calculation : p3 * (p2 * (p1 * x)) = 117 := by
  sorry

end percentage_calculation_l124_124656


namespace find_number_of_violas_l124_124186

theorem find_number_of_violas (cellos : ℕ) (pairs : ℕ) (probability : ℚ) 
    (h1 : cellos = 800) 
    (h2 : pairs = 100) 
    (h3 : probability = 0.00020833333333333335) : 
    ∃ V : ℕ, V = 600 := 
by 
    sorry

end find_number_of_violas_l124_124186


namespace simplify_expression_l124_124627

theorem simplify_expression : 
  ((3 + 4 + 5 + 6 + 7) / 3 + (3 * 6 + 9)^2 / 9) = 268 / 3 := 
by 
  sorry

end simplify_expression_l124_124627


namespace supervisors_per_bus_l124_124622

theorem supervisors_per_bus (total_supervisors : ℕ) (total_buses : ℕ) (H1 : total_supervisors = 21) (H2 : total_buses = 7) : (total_supervisors / total_buses = 3) :=
by
  sorry

end supervisors_per_bus_l124_124622


namespace x_coordinate_of_point_l124_124931

theorem x_coordinate_of_point (x_1 n : ℝ) 
  (h1 : x_1 = (n / 5) - (2 / 5)) 
  (h2 : x_1 + 3 = ((n + 15) / 5) - (2 / 5)) : 
  x_1 = (n / 5) - (2 / 5) :=
by sorry

end x_coordinate_of_point_l124_124931


namespace determine_num_chickens_l124_124655

def land_acres : ℕ := 30
def land_cost_per_acre : ℕ := 20
def house_cost : ℕ := 120000
def num_cows : ℕ := 20
def cow_cost_per_cow : ℕ := 1000
def install_hours : ℕ := 6
def install_cost_per_hour : ℕ := 100
def equipment_cost : ℕ := 6000
def total_expenses : ℕ := 147700
def chicken_cost_per_chicken : ℕ := 5

def total_cost_before_chickens : ℕ := 
  (land_acres * land_cost_per_acre) + 
  house_cost + 
  (num_cows * cow_cost_per_cow) + 
  (install_hours * install_cost_per_hour) + 
  equipment_cost

def chickens_cost : ℕ := total_expenses - total_cost_before_chickens

def num_chickens : ℕ := chickens_cost / chicken_cost_per_chicken

theorem determine_num_chickens : num_chickens = 100 := by
  sorry

end determine_num_chickens_l124_124655


namespace log_base_equal_l124_124413

noncomputable def logx (b x : ℝ) := Real.log x / Real.log b

theorem log_base_equal {x : ℝ} (h : 0 < x ∧ x ≠ 1) :
  logx 81 x = logx 16 2 → x = 3 :=
by
  intro h1
  sorry

end log_base_equal_l124_124413


namespace probability_both_selected_l124_124742

theorem probability_both_selected (P_C : ℚ) (P_B : ℚ) (hC : P_C = 4/5) (hB : P_B = 3/5) : 
  ((4/5) * (3/5)) = (12/25) := by
  sorry

end probability_both_selected_l124_124742


namespace compressor_station_distances_compressor_station_distances_when_a_is_30_l124_124188

theorem compressor_station_distances (a : ℝ) (h : 0 < a ∧ a < 60) :
  ∃ x y z : ℝ, x + y = 3 * z ∧ z + y = x + a ∧ x + z = 60 :=
sorry

theorem compressor_station_distances_when_a_is_30 :
  ∃ x y z : ℝ, 
  (x + y = 3 * z) ∧ (z + y = x + 30) ∧ (x + z = 60) ∧ 
  (x = 35) ∧ (y = 40) ∧ (z = 25) :=
sorry

end compressor_station_distances_compressor_station_distances_when_a_is_30_l124_124188


namespace customers_who_left_tip_l124_124272

-- Define the initial number of customers
def initial_customers : ℕ := 39

-- Define the additional number of customers during lunch rush
def additional_customers : ℕ := 12

-- Define the number of customers who didn't leave a tip
def no_tip_customers : ℕ := 49

-- Prove the number of customers who did leave a tip
theorem customers_who_left_tip : (initial_customers + additional_customers) - no_tip_customers = 2 := by
  sorry

end customers_who_left_tip_l124_124272


namespace wechat_payment_meaning_l124_124117

theorem wechat_payment_meaning (initial_balance after_receive_balance : ℝ)
  (recv_amount sent_amount : ℝ)
  (h1 : recv_amount = 200)
  (h2 : initial_balance + recv_amount = after_receive_balance)
  (h3 : after_receive_balance - sent_amount = initial_balance)
  : sent_amount = 200 :=
by
  -- starting the proof becomes irrelevant
  sorry

end wechat_payment_meaning_l124_124117


namespace calc_a_squared_plus_b_squared_and_ab_l124_124191

theorem calc_a_squared_plus_b_squared_and_ab (a b : ℝ) 
  (h1 : (a + b)^2 = 7) 
  (h2 : (a - b)^2 = 3) :
  a^2 + b^2 = 5 ∧ a * b = 1 :=
by
  sorry

end calc_a_squared_plus_b_squared_and_ab_l124_124191


namespace complex_multiplication_l124_124837

def imaginary_unit := Complex.I

theorem complex_multiplication (h : imaginary_unit^2 = -1) : (3 + 2 * imaginary_unit) * imaginary_unit = -2 + 3 * imaginary_unit :=
by
  sorry

end complex_multiplication_l124_124837


namespace surface_area_of_large_cube_l124_124330

theorem surface_area_of_large_cube (l w h : ℕ) (cube_side : ℕ) 
  (volume_cuboid : ℕ := l * w * h) 
  (n_cubes := volume_cuboid / (cube_side ^ 3))
  (side_length_large_cube : ℕ := cube_side * (n_cubes^(1/3 : ℕ))) 
  (surface_area_large_cube : ℕ := 6 * (side_length_large_cube ^ 2)) :
  l = 25 → w = 10 → h = 4 → cube_side = 1 → surface_area_large_cube = 600 :=
by
  intros hl hw hh hcs
  subst hl
  subst hw
  subst hh
  subst hcs
  sorry

end surface_area_of_large_cube_l124_124330


namespace triangle_sine_inequality_l124_124037

theorem triangle_sine_inequality
  (a b c : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (habc : a + b > c)
  (hbac : b + c > a)
  (hact : c + a > b)
  : |(a / (a + b)) + (b / (b + c)) + (c / (c + a)) - (3 / 2)| < (8 * Real.sqrt 2 - 5 * Real.sqrt 5) / 6 := 
sorry

end triangle_sine_inequality_l124_124037


namespace a5_b5_sum_l124_124852

-- Definitions of arithmetic sequences
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
∀ n : ℕ, a (n + 1) = a n + d

noncomputable
def a : ℕ → ℝ := sorry -- defining the arithmetic sequences
noncomputable
def b : ℕ → ℝ := sorry

-- Common differences for the sequences
noncomputable
def d_a : ℝ := sorry
noncomputable
def d_b : ℝ := sorry

-- Conditions given in the problem
axiom a1_b1_sum : a 1 + b 1 = 7
axiom a3_b3_sum : a 3 + b 3 = 21
axiom a_is_arithmetic : arithmetic_seq a d_a
axiom b_is_arithmetic : arithmetic_seq b d_b

-- Theorem to be proved
theorem a5_b5_sum : a 5 + b 5 = 35 := 
by sorry

end a5_b5_sum_l124_124852


namespace tan_range_l124_124153

theorem tan_range :
  ∀ (x : ℝ), -Real.pi / 4 ≤ x ∧ x < 0 ∨ 0 < x ∧ x ≤ Real.pi / 4 → -1 ≤ Real.tan x ∧ Real.tan x < 0 ∨ 0 < Real.tan x ∧ Real.tan x ≤ 1 :=
by
  sorry

end tan_range_l124_124153


namespace max_travel_within_budget_l124_124941

noncomputable def rental_cost_per_day : ℝ := 30
noncomputable def insurance_fee_per_day : ℝ := 10
noncomputable def mileage_cost_per_mile : ℝ := 0.18
noncomputable def budget : ℝ := 75
noncomputable def minimum_required_travel : ℝ := 100

theorem max_travel_within_budget : ∀ (rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel), 
  rental_cost_per_day = 30 → 
  insurance_fee_per_day = 10 → 
  mileage_cost_per_mile = 0.18 → 
  budget = 75 →
  minimum_required_travel = 100 →
  (minimum_required_travel + (budget - rental_cost_per_day - insurance_fee_per_day - mileage_cost_per_mile * minimum_required_travel) / mileage_cost_per_mile) = 194 := 
by
  intros rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end max_travel_within_budget_l124_124941


namespace find_f_of_3_l124_124912

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_3 (h : ∀ y > 0, f ((4 * y + 1) / (y + 1)) = 1 / y) : f 3 = 0.5 := by
  sorry

end find_f_of_3_l124_124912


namespace exceeded_by_600_l124_124728

noncomputable def ken_collected : ℕ := 600
noncomputable def mary_collected (ken : ℕ) : ℕ := 5 * ken
noncomputable def scott_collected (mary : ℕ) : ℕ := mary / 3
noncomputable def total_collected (ken mary scott : ℕ) : ℕ := ken + mary + scott
noncomputable def goal : ℕ := 4000
noncomputable def exceeded_goal (total goal : ℕ) : ℕ := total - goal

theorem exceeded_by_600 : exceeded_goal (total_collected ken_collected (mary_collected ken_collected) (scott_collected (mary_collected ken_collected))) goal = 600 := by
  sorry

end exceeded_by_600_l124_124728


namespace train_length_is_correct_l124_124410

noncomputable def speed_kmhr : ℝ := 45
noncomputable def time_sec : ℝ := 30
noncomputable def bridge_length_m : ℝ := 235

noncomputable def speed_ms : ℝ := (speed_kmhr * 1000) / 3600
noncomputable def total_distance_m : ℝ := speed_ms * time_sec
noncomputable def train_length_m : ℝ := total_distance_m - bridge_length_m

theorem train_length_is_correct : train_length_m = 140 :=
by
  -- Placeholder to indicate that a proof should go here
  -- Proof is omitted as per the instructions
  sorry

end train_length_is_correct_l124_124410


namespace mask_production_decrease_l124_124053

theorem mask_production_decrease (x : ℝ) : 
  (1 : ℝ) * (1 - x)^2 = 0.64 → 100 * (1 - x)^2 = 64 :=
by
  intro h
  sorry

end mask_production_decrease_l124_124053


namespace product_is_zero_l124_124548

theorem product_is_zero (n : ℤ) (h : n = 3) :
  (n - 3) * (n - 2) * (n - 1) * n * (n + 1) * (n + 4) = 0 := 
by
  sorry

end product_is_zero_l124_124548


namespace ratio_addition_l124_124043

theorem ratio_addition (a b : ℕ) (h : a / b = 2 / 3) : (a + b) / b = 5 / 3 := 
by sorry

end ratio_addition_l124_124043


namespace p_sufficient_for_q_q_not_necessary_for_p_l124_124289

variable (x : ℝ)

def p := |x - 2| < 1
def q := 1 < x ∧ x < 5

theorem p_sufficient_for_q : p x → q x :=
by sorry

theorem q_not_necessary_for_p : ¬ (q x → p x) :=
by sorry

end p_sufficient_for_q_q_not_necessary_for_p_l124_124289


namespace can_form_triangle_l124_124400

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

theorem can_form_triangle :
  (is_triangle 3 5 7) ∧ ¬(is_triangle 3 3 7) ∧ ¬(is_triangle 4 4 8) ∧ ¬(is_triangle 4 5 9) :=
by
  -- Proof steps will be added here
  sorry

end can_form_triangle_l124_124400


namespace second_field_area_percent_greater_l124_124449

theorem second_field_area_percent_greater (r1 r2 : ℝ) (h : r1 / r2 = 2 / 5) : 
  (π * (r2^2) - π * (r1^2)) / (π * (r1^2)) * 100 = 525 := 
by
  sorry

end second_field_area_percent_greater_l124_124449


namespace arithmetic_sequence_term_2011_is_671st_l124_124559

theorem arithmetic_sequence_term_2011_is_671st:
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → (3 * n - 2 = 2011) → n = 671 :=
by 
  intros a1 d n ha1 hd h_eq;
  sorry

end arithmetic_sequence_term_2011_is_671st_l124_124559


namespace sum_a2_to_a5_eq_zero_l124_124871

theorem sum_a2_to_a5_eq_zero 
  (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h : ∀ x : ℝ, x * (1 - 2 * x)^4 = a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5) : 
  a_2 + a_3 + a_4 + a_5 = 0 :=
sorry

end sum_a2_to_a5_eq_zero_l124_124871


namespace new_ratio_after_2_years_l124_124611

-- Definitions based on conditions
variable (A : ℕ) -- Current age of a
variable (B : ℕ) -- Current age of b

-- Conditions
def ratio_a_b := A / B = 5 / 3
def current_age_b := B = 6

-- Theorem: New ratio after 2 years is 3:2
theorem new_ratio_after_2_years (h1 : ratio_a_b A B) (h2 : current_age_b B) : (A + 2) / (B + 2) = 3 / 2 := by
  sorry

end new_ratio_after_2_years_l124_124611


namespace rhombus_area_fraction_l124_124593

theorem rhombus_area_fraction :
  let grid_area := 36
  let vertices := [(2, 2), (4, 2), (3, 3), (3, 1)]
  let rhombus_area := 2
  rhombus_area / grid_area = 1 / 18 :=
by
  sorry

end rhombus_area_fraction_l124_124593


namespace percentage_saved_is_25_l124_124038

def monthly_salary : ℝ := 1000

def increase_percentage : ℝ := 0.10

def saved_amount_after_increase : ℝ := 175

def calculate_percentage_saved (x : ℝ) : Prop := 
  1000 - (1000 - (x / 100) * monthly_salary) * (1 + increase_percentage) = saved_amount_after_increase

theorem percentage_saved_is_25 :
  ∃ x : ℝ, x = 25 ∧ calculate_percentage_saved x :=
sorry

end percentage_saved_is_25_l124_124038


namespace carols_rectangle_length_l124_124399

theorem carols_rectangle_length :
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  carol_length = 5 :=
by
  let jordan_length := 2
  let jordan_width := 60
  let carol_width := 24
  let jordan_area := jordan_length * jordan_width
  let carol_length := jordan_area / carol_width
  show carol_length = 5
  sorry

end carols_rectangle_length_l124_124399


namespace option_c_correct_l124_124717

theorem option_c_correct (x y : ℝ) (h : x < y) : -x > -y := 
sorry

end option_c_correct_l124_124717


namespace average_of_remaining_two_numbers_l124_124592

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) / 6 = 3.95)
  (h2 : (a + b) / 2 = 3.4)
  (h3 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 :=
sorry

end average_of_remaining_two_numbers_l124_124592


namespace reporters_not_covering_politics_l124_124801

-- Definitions of basic quantities
variables (R P : ℝ) (percentage_local : ℝ) (percentage_no_local : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  R = 100 ∧
  percentage_local = 10 ∧
  percentage_no_local = 30 ∧
  percentage_local = 0.7 * P

-- Theorem statement for the problem
theorem reporters_not_covering_politics (h : conditions R P percentage_local percentage_no_local) :
  100 - P = 85.71 :=
by sorry

end reporters_not_covering_politics_l124_124801


namespace part_a_part_b_part_c_l124_124991

-- Define initial setup and conditions
def average (scores: List ℚ) : ℚ :=
  scores.sum / scores.length

-- Part (a)
theorem part_a (A B : List ℚ) (a b : ℚ) (A' : List ℚ) (B' : List ℚ) :
  average A = a ∧ average B = b ∧ average A' = a ∧ average B' = b ∧
  average A' > a ∧ average B' > b :=
sorry

-- Part (b)
theorem part_b (A B : List ℚ) : 
  ∀ a b : ℚ, (average A = a ∧ average B = b ∧ ∀ A' : List ℚ, average A' > a ∧ ∀ B' : List ℚ, average B' > b) :=
sorry

-- Part (c)
theorem part_c (A B C : List ℚ) (a b c : ℚ) (A' B' C' A'' B'' C'' : List ℚ) :
  average A = a ∧ average B = b ∧ average C = c ∧
  average A' = a ∧ average B' = b ∧ average C' = c ∧
  average A'' = a ∧ average B'' = b ∧ average C'' = c ∧
  average A' > a ∧ average B' > b ∧ average C' > c ∧
  average A'' > average A' ∧ average B'' > average B' ∧ average C'' > average C' :=
sorry

end part_a_part_b_part_c_l124_124991


namespace largest_sum_digits_24_hour_watch_l124_124889

theorem largest_sum_digits_24_hour_watch : 
  (∃ h m : ℕ, 0 ≤ h ∧ h < 24 ∧ 0 ≤ m ∧ m < 60 ∧ 
              (h / 10 + h % 10 + m / 10 + m % 10 = 24)) :=
by
  sorry

end largest_sum_digits_24_hour_watch_l124_124889


namespace arithmetic_sequence_fifth_term_l124_124327

theorem arithmetic_sequence_fifth_term (a d : ℤ) 
  (h1 : a + 9 * d = 3) 
  (h2 : a + 11 * d = 9) : 
  a + 4 * d = -12 :=
by
  sorry

end arithmetic_sequence_fifth_term_l124_124327


namespace marites_saves_120_per_year_l124_124736

def current_internet_speed := 10 -- Mbps
def current_monthly_bill := 20 -- dollars

def monthly_cost_20mbps := current_monthly_bill + 10 -- dollars
def monthly_cost_30mbps := current_monthly_bill * 2 -- dollars

def bundled_cost_20mbps := 80 -- dollars per month
def bundled_cost_30mbps := 90 -- dollars per month

def annual_cost_20mbps := bundled_cost_20mbps * 12 -- dollars per year
def annual_cost_30mbps := bundled_cost_30mbps * 12 -- dollars per year

theorem marites_saves_120_per_year :
  annual_cost_30mbps - annual_cost_20mbps = 120 := 
by
  sorry

end marites_saves_120_per_year_l124_124736


namespace find_functional_form_l124_124774

theorem find_functional_form (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  sorry

end find_functional_form_l124_124774


namespace jacket_final_price_l124_124081

/-- 
The initial price of the jacket is $20, 
the first discount is 40%, and the second discount is 25%. 
We need to prove that the final price of the jacket is $9.
-/
theorem jacket_final_price :
  let initial_price := 20
  let first_discount := 0.40
  let second_discount := 0.25
  let price_after_first_discount := initial_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  final_price = 9 :=
by
  sorry

end jacket_final_price_l124_124081


namespace find_m_l124_124637

theorem find_m (m : ℕ) (h : 10^(m-1) < 2^512 ∧ 2^512 < 10^m): 
  m = 155 :=
sorry

end find_m_l124_124637


namespace cos_60_eq_half_l124_124483

theorem cos_60_eq_half : Real.cos (60 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end cos_60_eq_half_l124_124483


namespace problem_statement_l124_124111

theorem problem_statement : (4^4 / 4^3) * 2^8 = 1024 := by
  sorry

end problem_statement_l124_124111


namespace correct_vector_equation_l124_124124

variables {V : Type*} [AddCommGroup V]

variables (A B C: V)

theorem correct_vector_equation : 
  (A - B) - (B - C) = A - C :=
sorry

end correct_vector_equation_l124_124124


namespace find_t_l124_124119

variable {a b c r s t : ℝ}

-- Conditions from part a)
def first_polynomial_has_roots (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c)) : Prop :=
  ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = 0 → x = a ∨ x = b ∨ x = c

def second_polynomial_has_roots (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a))) : Prop :=
  ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = 0 → x = (a + b) ∨ x = (b + c) ∨ x = (c + a)

-- Translate problem (find t) with conditions
theorem find_t (ha : ∀ x, x ^ 3 + 3 * x ^ 2 + 4 * x - 11 = (x - a) * (x - b) * (x - c))
    (hb : ∀ x, x ^ 3 + r * x ^ 2 + s * x + t = (x - (a + b)) * (x - (b + c)) * (x - (c + a)))
    (sum_roots : a + b + c = -3) 
    (prod_roots : a * b * c = -11):
  t = 23 := 
sorry

end find_t_l124_124119


namespace sum_of_non_solutions_l124_124949

theorem sum_of_non_solutions (A B C : ℝ) :
  (∀ x : ℝ, (x ≠ -C ∧ x ≠ -10) → (x + B) * (A * x + 40) / ((x + C) * (x + 10)) = 2) →
  (A = 2 ∧ B = 10 ∧ C = 20) →
  (-10 + -20 = -30) :=
by sorry

end sum_of_non_solutions_l124_124949


namespace purchase_price_is_60_l124_124182

variable (P S D : ℝ)
variable (GP : ℝ := 4)

theorem purchase_price_is_60
  (h1 : S = P + 0.25 * S)
  (h2 : D = 0.80 * S)
  (h3 : GP = D - P) :
  P = 60 :=
by
  sorry

end purchase_price_is_60_l124_124182


namespace complex_product_eq_50i_l124_124847

open Complex

theorem complex_product_eq_50i : 
  let Q := (4 : ℂ) + 3 * I
  let E := (2 * I : ℂ)
  let D := (4 : ℂ) - 3 * I
  Q * E * D = 50 * I :=
by
  -- Complex numbers and multiplication are handled here
  sorry

end complex_product_eq_50i_l124_124847


namespace geometric_sequence_sum_l124_124204

theorem geometric_sequence_sum
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * r)
  (h1 : a 1 + a 3 = 20)
  (h2 : a 2 + a 4 = 40)
  :
  a 3 + a 5 = 80 :=
sorry

end geometric_sequence_sum_l124_124204


namespace smallest_sum_of_pairwise_distinct_squares_l124_124533

theorem smallest_sum_of_pairwise_distinct_squares :
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  ∃ x y z : ℕ, a + b = x^2 ∧ b + c = z^2 ∧ c + a = y^2 ∧ a + b + c = 55 :=
sorry

end smallest_sum_of_pairwise_distinct_squares_l124_124533


namespace nine_digit_numbers_divisible_by_eleven_l124_124842

theorem nine_digit_numbers_divisible_by_eleven :
  ∃ (n : ℕ), n = 31680 ∧
    ∃ (num : ℕ), num < 10^9 ∧ num ≥ 10^8 ∧
      (∀ d : ℕ, 1 ≤ d ∧ d ≤ 9 → ∃ i : ℕ, i ≤ 8 ∧ (num / 10^i) % 10 = d) ∧
      (num % 11 = 0) := 
sorry

end nine_digit_numbers_divisible_by_eleven_l124_124842


namespace fraction_addition_l124_124782

variable (a : ℝ)

theorem fraction_addition : (3 / a) + (2 / a) = 5 / a := 
by sorry

end fraction_addition_l124_124782


namespace least_subtract_to_divisible_by_14_l124_124415

theorem least_subtract_to_divisible_by_14 (n : ℕ) (h : n = 7538): 
  (n % 14 = 6) -> ∃ m, (m = 6) ∧ ((n - m) % 14 = 0) :=
by
  sorry

end least_subtract_to_divisible_by_14_l124_124415


namespace find_n_18_l124_124126

def valid_denominations (n : ℕ) : Prop :=
  ∀ k < 106, ∃ a b c : ℕ, k = 7 * a + n * b + (n + 1) * c

def cannot_form_106 (n : ℕ) : Prop :=
  ¬ ∃ a b c : ℕ, 106 = 7 * a + n * b + (n + 1) * c

theorem find_n_18 : 
  ∃ n : ℕ, valid_denominations n ∧ cannot_form_106 n ∧ ∀ m < n, ¬ (valid_denominations m ∧ cannot_form_106 m) :=
sorry

end find_n_18_l124_124126


namespace number_of_comic_books_l124_124934

def fairy_tale_books := 305
def science_and_technology_books := fairy_tale_books + 115
def total_books := fairy_tale_books + science_and_technology_books
def comic_books := total_books * 4

theorem number_of_comic_books : comic_books = 2900 := by
  sorry

end number_of_comic_books_l124_124934


namespace total_seeds_l124_124797

theorem total_seeds (A B C : ℕ) (h₁ : A = B + 10) (h₂ : B = 30) (h₃ : C = 30) : A + B + C = 100 :=
by
  sorry

end total_seeds_l124_124797


namespace exists_x_eq_1_l124_124181

theorem exists_x_eq_1 (x y z t : ℕ) (h : x + y + z + t = 10) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  ∃ x, x = 1 :=
sorry

end exists_x_eq_1_l124_124181


namespace g_at_3_l124_124443

noncomputable def g : ℝ → ℝ := sorry

theorem g_at_3 (h : ∀ x : ℝ, g (3 ^ x) - x * g (3 ^ (-x)) = x) : g 3 = 0 :=
by
  sorry

end g_at_3_l124_124443


namespace percent_twelve_equals_eighty_four_l124_124313

theorem percent_twelve_equals_eighty_four (x : ℝ) (h : (12 / 100) * x = 84) : x = 700 :=
by
  sorry

end percent_twelve_equals_eighty_four_l124_124313


namespace fixed_point_l124_124546

theorem fixed_point (a : ℝ) : (a + 1) * (-4) - (2 * a + 5) * (-2) - 6 = 0 :=
by
  sorry

end fixed_point_l124_124546


namespace loss_percentage_25_l124_124270

variable (C S : ℝ)
variable (h : 15 * C = 20 * S)

theorem loss_percentage_25 (h : 15 * C = 20 * S) : (C - S) / C * 100 = 25 := by
  sorry

end loss_percentage_25_l124_124270


namespace min_blocks_to_remove_l124_124173

theorem min_blocks_to_remove (n : ℕ) (h₁ : n = 59) : ∃ k, ∃ m, (m*m*m ≤ n ∧ n < (m+1)*(m+1)*(m+1)) ∧ k = n - m*m*m ∧ k = 32 :=
by {
  sorry
}

end min_blocks_to_remove_l124_124173


namespace max_and_min_sum_of_vars_l124_124932

theorem max_and_min_sum_of_vars (x y z w : ℝ) (h : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ w)
  (eq : x^2 + y^2 + z^2 + w^2 + x + 2*y + 3*z + 4*w = 17 / 2) :
  ∃ max min : ℝ, max = 3 ∧ min = -2 + 5 / 2 * Real.sqrt 2 ∧
  (∀ (S : ℝ), S = x + y + z + w → S ≤ max ∧ S ≥ min) :=
by sorry

end max_and_min_sum_of_vars_l124_124932


namespace toothpick_count_l124_124718

theorem toothpick_count (height width : ℕ) (h_height : height = 20) (h_width : width = 10) : 
  (21 * width + 11 * height) = 430 :=
by
  sorry

end toothpick_count_l124_124718


namespace alpha_cubic_expression_l124_124571

theorem alpha_cubic_expression (α : ℝ) (hα : α^2 - 8 * α - 5 = 0) : α^3 - 7 * α^2 - 13 * α + 6 = 11 :=
sorry

end alpha_cubic_expression_l124_124571


namespace smallest_x_for_multiple_of_625_l124_124492

theorem smallest_x_for_multiple_of_625 (x : ℕ) (hx_pos : 0 < x) : (500 * x) % 625 = 0 → x = 5 :=
by
  sorry

end smallest_x_for_multiple_of_625_l124_124492


namespace determine_b_when_lines_parallel_l124_124158

theorem determine_b_when_lines_parallel (b : ℝ) : 
  (∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = (b + 9) * x) → b = -6 :=
by
  sorry

end determine_b_when_lines_parallel_l124_124158


namespace purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l124_124570

def z (m : ℝ) : Complex := Complex.mk (2 * m^2 - 3 * m - 2) (m^2 - 3 * m + 2)

theorem purely_imaginary_implies_m_eq_neg_half (m : ℝ) : 
  (z m).re = 0 ↔ m = -1 / 2 := sorry

theorem simplify_z_squared_over_z_add_5_plus_2i (z_zero : ℂ) :
  z 0 = ⟨-2, 2⟩ →
  (z 0)^2 / (z 0 + Complex.mk 5 2) = ⟨-32 / 25, -24 / 25⟩ := sorry

end purely_imaginary_implies_m_eq_neg_half_simplify_z_squared_over_z_add_5_plus_2i_l124_124570


namespace trig_quadrant_l124_124338

theorem trig_quadrant (α : ℝ) (h1 : Real.sin α < 0) (h2 : Real.tan α > 0) : 
  ∃ k : ℤ, α = (2 * k + 1) * π + α / 2 :=
sorry

end trig_quadrant_l124_124338


namespace inequality_problem_l124_124203

theorem inequality_problem 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
by sorry

end inequality_problem_l124_124203


namespace value_of_expression_l124_124207

theorem value_of_expression (x y : ℤ) (h1 : x = 1) (h2 : y = 630) : 
  2019 * x - 3 * y - 9 = 120 := 
by
  sorry

end value_of_expression_l124_124207


namespace point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l124_124506

def circle_equation (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y - m = 0

def inside_circle (x y m : ℝ) : Prop :=
  (x-1)^2 + (y+2)^2 < 5 + m

theorem point_A_inside_circle (m : ℝ) : -1 < m ∧ m < 4 ↔ inside_circle m (-2) m :=
sorry

def circle_equation_m_4 (x y : ℝ) : Prop :=
  circle_equation x y 4

def dist_square_to_point_H (x y : ℝ) : ℝ :=
  (x - 4)^2 + (y - 2)^2

theorem max_min_dist_square_on_circle (P : ℝ × ℝ) :
  circle_equation_m_4 P.1 P.2 →
  4 ≤ dist_square_to_point_H P.1 P.2 ∧ dist_square_to_point_H P.1 P.2 ≤ 64 :=
sorry

def line_equation (m x y : ℝ) : Prop :=
  y = x + m

theorem chord_through_origin (m : ℝ) :
  ∃ m : ℝ, line_equation m (1 : ℝ) (-2 : ℝ) ∧ 
  (m = -4 ∨ m = 1) :=
sorry

end point_A_inside_circle_max_min_dist_square_on_circle_chord_through_origin_l124_124506


namespace circle_eq_l124_124975

theorem circle_eq (A B : ℝ × ℝ) (hA1 : A = (5, 2)) (hA2 : B = (-1, 4)) (hx : ∃ (c : ℝ), (c, 0) = (c, 0)) :
  ∃ (C : ℝ) (D : ℝ) (x y : ℝ), (x + C) ^ 2 + y ^ 2 = D ∧ D = 20 ∧ (x - 1) ^ 2 + y ^ 2 = 20 :=
by
  sorry

end circle_eq_l124_124975


namespace number_of_units_sold_l124_124105

theorem number_of_units_sold (p : ℕ) (c : ℕ) (k : ℕ) (h : p * c = k) (h₁ : c = 800) (h₂ : k = 8000) : p = 10 :=
by
  sorry

end number_of_units_sold_l124_124105


namespace sum_of_remainders_eq_3_l124_124063

theorem sum_of_remainders_eq_3 (a b c : ℕ) (h1 : a % 59 = 28) (h2 : b % 59 = 15) (h3 : c % 59 = 19) (h4 : a = b + d ∨ b = c + d ∨ c = a + d) : 
  (a + b + c) % 59 = 3 :=
by {
  sorry -- Proof to be constructed
}

end sum_of_remainders_eq_3_l124_124063


namespace four_nabla_seven_l124_124572

-- Define the operation ∇
def nabla (a b : ℤ) : ℚ :=
  (a + b) / (1 + a * b)

theorem four_nabla_seven :
  nabla 4 7 = 11 / 29 :=
by
  sorry

end four_nabla_seven_l124_124572


namespace find_temperature_on_friday_l124_124434

variable (M T W Th F : ℕ)

def problem_conditions : Prop :=
  (M + T + W + Th) / 4 = 48 ∧
  (T + W + Th + F) / 4 = 46 ∧
  M = 44

theorem find_temperature_on_friday (h : problem_conditions M T W Th F) : F = 36 := by
  sorry

end find_temperature_on_friday_l124_124434


namespace sam_total_cents_l124_124576

def dimes_to_cents (dimes : ℕ) : ℕ := dimes * 10
def quarters_to_cents (quarters : ℕ) : ℕ := quarters * 25
def nickels_to_cents (nickels : ℕ) : ℕ := nickels * 5
def dollars_to_cents (dollars : ℕ) : ℕ := dollars * 100

noncomputable def total_cents (initial_dimes dad_dimes mom_dimes grandma_dollars sister_quarters_initial : ℕ)
                             (initial_quarters dad_quarters mom_quarters grandma_transform sister_quarters_donation : ℕ)
                             (initial_nickels dad_nickels mom_nickels grandma_conversion sister_nickels_donation : ℕ) : ℕ :=
  dimes_to_cents initial_dimes +
  quarters_to_cents initial_quarters +
  nickels_to_cents initial_nickels +
  dimes_to_cents dad_dimes +
  quarters_to_cents dad_quarters -
  nickels_to_cents mom_nickels -
  dimes_to_cents mom_dimes +
  dollars_to_cents grandma_dollars +
  quarters_to_cents sister_quarters_donation +
  nickels_to_cents sister_nickels_donation

theorem sam_total_cents :
  total_cents 9 7 2 3 4 5 2 0 0 3 2 1 = 735 := 
  by exact sorry

end sam_total_cents_l124_124576


namespace cost_of_apple_l124_124438

variable (A O : ℝ)

theorem cost_of_apple :
  (6 * A + 3 * O = 1.77) ∧ (2 * A + 5 * O = 1.27) → A = 0.21 :=
by
  intro h
  -- Proof goes here
  sorry

end cost_of_apple_l124_124438


namespace absent_children_l124_124612

theorem absent_children (total_children bananas_per_child_if_present bananas_per_child_if_absent children_present absent_children : ℕ) 
  (H1 : total_children = 740)
  (H2 : bananas_per_child_if_present = 2)
  (H3 : bananas_per_child_if_absent = 4)
  (H4 : children_present * bananas_per_child_if_absent = total_children * bananas_per_child_if_present)
  (H5 : children_present = total_children - absent_children) : 
  absent_children = 370 :=
sorry

end absent_children_l124_124612


namespace largest_consecutive_odd_number_sum_is_27_l124_124157

theorem largest_consecutive_odd_number_sum_is_27
  (a b c : ℤ)
  (h1 : a + b + c = 75)
  (h2 : c - a = 4)
  (h3 : a % 2 = 1)
  (h4 : b % 2 = 1)
  (h5 : c % 2 = 1) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_sum_is_27_l124_124157


namespace largest_divisor_is_15_l124_124424

def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

def largest_divisor (n : ℕ) : ℕ :=
  (n + 1) * (n + 3) * (n + 5) * (n + 7) * (n + 9) * (n + 11) * (n + 13)

theorem largest_divisor_is_15 : ∀ (n : ℕ), n > 0 → is_even n → 15 ∣ largest_divisor n ∧ (∀ m, m ∣ largest_divisor n → m ≤ 15) :=
by
  intros n pos even
  sorry

end largest_divisor_is_15_l124_124424


namespace printer_Y_time_l124_124490

theorem printer_Y_time (T_y : ℝ) : 
    (12 * (1 / (1 / T_y + 1 / 20)) = 1.8) → T_y = 10 := 
by 
sorry

end printer_Y_time_l124_124490


namespace final_number_lt_one_l124_124168

theorem final_number_lt_one :
  ∀ (numbers : Finset ℕ),
    (numbers = Finset.range 3000 \ Finset.range 1000) →
    (∀ (a b : ℕ), a ∈ numbers → b ∈ numbers → a ≤ b →
    ∃ (numbers' : Finset ℕ), numbers' = (numbers \ {a, b}) ∪ {a / 2}) →
    ∃ (x : ℕ), x ∈ numbers ∧ x < 1 :=
by
  sorry

end final_number_lt_one_l124_124168


namespace geoboard_quadrilaterals_l124_124636

-- Definitions of the quadrilaterals as required by the conditions of the problem.
def quadrilateral_area (quad : Type) : ℝ := sorry
def quadrilateral_perimeter (quad : Type) : ℝ := sorry

-- Declaration of Quadrilateral I and II on a geoboard.
def quadrilateral_i : Type := sorry
def quadrilateral_ii : Type := sorry

-- The proof problem statement.
theorem geoboard_quadrilaterals :
  quadrilateral_area quadrilateral_i = quadrilateral_area quadrilateral_ii ∧
  quadrilateral_perimeter quadrilateral_i < quadrilateral_perimeter quadrilateral_ii := by
  sorry

end geoboard_quadrilaterals_l124_124636


namespace card_game_probability_l124_124654

theorem card_game_probability :
  let A_wins := 4;  -- number of heads needed for A to win all cards
  let B_wins := 4;  -- number of tails needed for B to win all cards
  let total_flips := 5;  -- exactly 5 flips
  (Nat.choose total_flips 1 + Nat.choose total_flips 1) / (2^total_flips) = 5 / 16 :=
by
  sorry

end card_game_probability_l124_124654


namespace largest_n_for_factorable_polynomial_l124_124112

theorem largest_n_for_factorable_polynomial :
  (∃ (A B : ℤ), A * B = 72 ∧ ∀ (n : ℤ), n = 3 * B + A → n ≤ 217) ∧
  (∃ (A B : ℤ), A * B = 72 ∧ 3 * B + A = 217) :=
by
    sorry

end largest_n_for_factorable_polynomial_l124_124112


namespace linear_eq_solution_l124_124597

theorem linear_eq_solution (m x : ℝ) (h : |m| = 1) (h1: 1 - m ≠ 0):
  x = -(1/2) :=
sorry

end linear_eq_solution_l124_124597


namespace rectangle_diagonal_ratio_l124_124602

theorem rectangle_diagonal_ratio (s : ℝ) :
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  D / d = Real.sqrt 5 :=
by
  let d := (Real.sqrt 2) * s
  let D := (Real.sqrt 10) * s
  sorry

end rectangle_diagonal_ratio_l124_124602


namespace lisa_and_robert_total_photos_l124_124444

def claire_photos : Nat := 10
def lisa_photos (c : Nat) : Nat := 3 * c
def robert_photos (c : Nat) : Nat := c + 20

theorem lisa_and_robert_total_photos :
  let c := claire_photos
  let l := lisa_photos c
  let r := robert_photos c
  l + r = 60 :=
by
  sorry

end lisa_and_robert_total_photos_l124_124444


namespace triangle_angle_contradiction_l124_124898

theorem triangle_angle_contradiction (A B C : ℝ) (hA : A > 0) (hB : B > 0) (hC : C > 0)
  (sum_angles : A + B + C = 180) : 
  (¬ (A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60)) = (A > 60 ∧ B > 60 ∧ C > 60) :=
by sorry

end triangle_angle_contradiction_l124_124898


namespace fixed_point_of_tangent_line_l124_124052

theorem fixed_point_of_tangent_line (x y : ℝ) (h1 : x = 3) 
  (h2 : ∃ m : ℝ, (3 - m)^2 + (y - 2)^2 = 4) :
  ∃ (k l : ℝ), k = 4 / 3 ∧ l = 2 :=
by
  sorry

end fixed_point_of_tangent_line_l124_124052


namespace value_of_a_l124_124554

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, x > 0.5 → 1 - a / 2^x > 0) → a = Real.sqrt 2 :=
by
  sorry

end value_of_a_l124_124554


namespace brendan_total_wins_l124_124874

-- Define the number of matches won in each round
def matches_won_first_round : ℕ := 6
def matches_won_second_round : ℕ := 4
def matches_won_third_round : ℕ := 3
def matches_won_final_round : ℕ := 5

-- Define the total number of matches won
def total_matches_won : ℕ := 
  matches_won_first_round + matches_won_second_round + matches_won_third_round + matches_won_final_round

-- State the theorem that needs to be proven
theorem brendan_total_wins : total_matches_won = 18 := by
  sorry

end brendan_total_wins_l124_124874


namespace triangle_PQR_area_l124_124487

-- Define the points P, Q, and R
def P : (ℝ × ℝ) := (-2, 2)
def Q : (ℝ × ℝ) := (8, 2)
def R : (ℝ × ℝ) := (4, -4)

-- Define a function to calculate the area of triangle
def triangle_area (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

-- Lean statement to prove the area of triangle PQR is 30 square units
theorem triangle_PQR_area : triangle_area P Q R = 30 := by
  sorry

end triangle_PQR_area_l124_124487


namespace longest_side_of_rectangle_l124_124259

theorem longest_side_of_rectangle (l w : ℕ) 
  (h1 : 2 * l + 2 * w = 240) 
  (h2 : l * w = 1920) : 
  l = 101 ∨ w = 101 :=
sorry

end longest_side_of_rectangle_l124_124259


namespace ratio_of_terms_l124_124195

theorem ratio_of_terms (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ) :
  (∀ n, S_n n = (n * (2 * a_n n - (n - 1))) / 2) → 
  (∀ n, T_n n = (n * (2 * b_n n - (n - 1))) / 2) → 
  (∀ n, S_n n / T_n n = (n + 3) / (2 * n + 1)) → 
  S_n 6 / T_n 6 = 14 / 23 :=
by
  sorry

end ratio_of_terms_l124_124195


namespace integer_a_conditions_l124_124826

theorem integer_a_conditions (a : ℤ) :
  (∃ (x y : ℕ), x ≠ y ∧ (a * x * y + 1) ∣ (a * x^2 + 1) ^ 2) → a ≥ -1 :=
sorry

end integer_a_conditions_l124_124826


namespace minimum_m_n_squared_l124_124271

theorem minimum_m_n_squared (a b c m n : ℝ) (h1 : c > a) (h2 : c > b) (h3 : c = Real.sqrt (a^2 + b^2)) 
    (h4 : a * m + b * n + c = 0) : m^2 + n^2 ≥ 1 := by
  sorry

end minimum_m_n_squared_l124_124271


namespace find_product_of_variables_l124_124089

variables (a b c d : ℚ)

def system_of_equations (a b c d : ℚ) :=
  3 * a + 4 * b + 6 * c + 9 * d = 45 ∧
  4 * (d + c) = b + 1 ∧
  4 * b + 2 * c = a ∧
  2 * c - 2 = d

theorem find_product_of_variables :
  system_of_equations a b c d → a * b * c * d = 162 / 185 :=
by sorry

end find_product_of_variables_l124_124089


namespace arts_school_probability_l124_124314

theorem arts_school_probability :
  let cultural_courses := 3
  let arts_courses := 3
  let total_periods := 6
  let total_arrangements := Nat.factorial total_periods
  let no_adjacent_more_than_one_separator := (72 + 216 + 144)
  (no_adjacent_more_than_one_separator : ℝ) / (total_arrangements : ℝ) = (3 / 5 : ℝ) := 
by 
  sorry

end arts_school_probability_l124_124314


namespace rose_can_afford_l124_124193

noncomputable def total_cost_before_discount : ℝ :=
  2.40 + 9.20 + 6.50 + 12.25 + 4.75

noncomputable def discount : ℝ :=
  0.15 * total_cost_before_discount

noncomputable def total_cost_after_discount : ℝ :=
  total_cost_before_discount - discount

noncomputable def budget : ℝ :=
  30.00

noncomputable def remaining_budget : ℝ :=
  budget - total_cost_after_discount

theorem rose_can_afford :
  remaining_budget = 0.165 :=
by
  -- proof goes here
  sorry

end rose_can_afford_l124_124193


namespace no_valid_solution_l124_124831

theorem no_valid_solution (x y z : ℤ) (h1 : x = 11 * y + 4) 
  (h2 : 2 * x = 24 * y + 3) (h3 : x + z = 34 * y + 5) : 
  ¬ ∃ (y : ℤ), 13 * y - x + 7 * z = 0 :=
by
  sorry

end no_valid_solution_l124_124831


namespace evaluate_polynomial_at_3_l124_124783

noncomputable def f (x : ℝ) : ℝ := 0.5 * x^5 + 4 * x^4 - 3 * x^2 + x - 1

theorem evaluate_polynomial_at_3 : f 3 = 1 := by
  sorry

end evaluate_polynomial_at_3_l124_124783


namespace apple_street_length_l124_124600

theorem apple_street_length :
  ∀ (n : ℕ) (d : ℕ), 
    (n = 15) → (d = 200) → 
    (∃ l : ℝ, (l = ((n + 1) * d) / 1000) ∧ l = 3.2) :=
by
  intros
  sorry

end apple_street_length_l124_124600


namespace power_of_two_last_digit_product_divisible_by_6_l124_124545

theorem power_of_two_last_digit_product_divisible_by_6 (n : Nat) (h : 3 < n) :
  ∃ d m : Nat, (2^n = 10 * m + d) ∧ (m * d) % 6 = 0 :=
by
  sorry

end power_of_two_last_digit_product_divisible_by_6_l124_124545


namespace tony_lottery_winning_l124_124521

theorem tony_lottery_winning
  (tickets : ℕ) (winning_numbers : ℕ) (worth_per_number : ℕ) (identical_numbers : Prop)
  (h_tickets : tickets = 3) (h_winning_numbers : winning_numbers = 5) (h_worth_per_number : worth_per_number = 20)
  (h_identical_numbers : identical_numbers) :
  (tickets * (winning_numbers * worth_per_number) = 300) :=
by
  sorry

end tony_lottery_winning_l124_124521


namespace points_on_same_line_l124_124065

theorem points_on_same_line (p : ℝ) :
  (∃ m : ℝ, m = ( -3.5 - 0.5 ) / ( 3 - (-1)) ∧ ∀ x y : ℝ, 
    (x = -1 ∧ y = 0.5) ∨ (x = 3 ∧ y = -3.5) ∨ (x = 7 ∧ y = p) → y = m * x + (0.5 - m * (-1))) →
    p = -7.5 :=
by
  sorry

end points_on_same_line_l124_124065


namespace range_of_m_l124_124353

def prop_p (m : ℝ) : Prop := ∃ (x y : ℝ), x^2 + y^2 - 2*x - 4*y + m = 0
def prop_q (m : ℝ) : Prop := ∃ (x y : ℝ), (x^2) / (m-6) - (y^2) / (m+3) = 1

theorem range_of_m (m : ℝ) : ¬ (prop_p m ∧ prop_q m) → m ≥ -3 :=
sorry

end range_of_m_l124_124353


namespace subtraction_of_bases_l124_124610

def base8_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 8^2 + ((n % 100) / 10) * 8^1 + (n % 10) * 8^0

def base7_to_base10 (n : Nat) : Nat :=
  match n with
  | 0 => 0
  | _ => (n / 100) * 7^2 + ((n % 100) / 10) * 7^1 + (n % 10) * 7^0

theorem subtraction_of_bases :
  base8_to_base10 343 - base7_to_base10 265 = 82 :=
by
  sorry

end subtraction_of_bases_l124_124610


namespace alex_jellybeans_l124_124844

theorem alex_jellybeans (x : ℕ) : x = 254 → x ≥ 150 ∧ x % 15 = 14 ∧ x % 17 = 16 :=
by
  sorry

end alex_jellybeans_l124_124844


namespace oranges_taken_from_basket_l124_124562

-- Define the original number of oranges and the number left after taking some out.
def original_oranges : ℕ := 8
def oranges_left : ℕ := 3

-- Prove that the number of oranges taken from the basket equals 5.
theorem oranges_taken_from_basket : original_oranges - oranges_left = 5 := by
  sorry

end oranges_taken_from_basket_l124_124562


namespace ninth_term_arithmetic_sequence_l124_124722

def first_term : ℚ := 3 / 4
def seventeenth_term : ℚ := 6 / 7

theorem ninth_term_arithmetic_sequence :
  let a1 := first_term
  let a17 := seventeenth_term
  (a1 + a17) / 2 = 45 / 56 := 
sorry

end ninth_term_arithmetic_sequence_l124_124722


namespace fraction_of_girls_l124_124858

theorem fraction_of_girls (G T B : ℕ) (Fraction : ℚ)
  (h1 : Fraction * G = (1/3 : ℚ) * T)
  (h2 : (B : ℚ) / G = 1/2) :
  Fraction = 1/2 := by
  sorry

end fraction_of_girls_l124_124858


namespace evaluate_f_g_3_l124_124765

def g (x : ℝ) := x^3
def f (x : ℝ) := 3 * x - 2

theorem evaluate_f_g_3 : f (g 3) = 79 := by
  sorry

end evaluate_f_g_3_l124_124765


namespace area_of_triangle_l124_124019

noncomputable def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 8 = 1

def foci_distance (F1 F2 : ℝ × ℝ) : Prop := (F1.1, F1.2) = (-3, 0) ∧ (F2.1, F2.2) = (3, 0)

def point_on_hyperbola (x y : ℝ) : Prop := hyperbola x y

def distance_ratios (P F1 F2 : ℝ × ℝ) : Prop := 
  let PF1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
  let PF2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  PF1 / PF2 = 3 / 4

theorem area_of_triangle {P F1 F2 : ℝ × ℝ} 
  (H1 : foci_distance F1 F2)
  (H2 : point_on_hyperbola P.1 P.2)
  (H3 : distance_ratios P F1 F2) :
  let area := 1 / 2 * (6:ℝ) * (8:ℝ) * Real.sqrt 5
  area = 8 * Real.sqrt 5 := 
sorry

end area_of_triangle_l124_124019


namespace a9_value_l124_124039

theorem a9_value (a : ℕ → ℝ) (x : ℝ) (h : (1 + x) ^ 10 = 
  (a 0) + (a 1) * (1 - x) + (a 2) * (1 - x)^2 + 
  (a 3) * (1 - x)^3 + (a 4) * (1 - x)^4 + 
  (a 5) * (1 - x)^5 + (a 6) * (1 - x)^6 + 
  (a 7) * (1 - x)^7 + (a 8) * (1 - x)^8 + 
  (a 9) * (1 - x)^9 + (a 10) * (1 - x)^10) : 
  a 9 = -20 :=
sorry

end a9_value_l124_124039


namespace g_neg_2_eq_3_l124_124206

def g (x : ℝ) : ℝ := 2 * x^2 + 3 * x + 1

theorem g_neg_2_eq_3 : g (-2) = 3 :=
by
  sorry

end g_neg_2_eq_3_l124_124206


namespace distance_between_trees_l124_124541

def yard_length : ℝ := 1530
def number_of_trees : ℝ := 37
def number_of_gaps := number_of_trees - 1

theorem distance_between_trees :
  number_of_gaps ≠ 0 →
  (yard_length / number_of_gaps) = 42.5 :=
by
  sorry

end distance_between_trees_l124_124541


namespace number_of_solutions_l124_124740

theorem number_of_solutions : ∃ (s : Finset ℕ), (∀ x ∈ s, 100 ≤ x^2 ∧ x^2 ≤ 200) ∧ s.card = 5 :=
by
  sorry

end number_of_solutions_l124_124740


namespace probability_of_two_sunny_days_l124_124334

def prob_two_sunny_days (prob_sunny prob_rain : ℚ) (days : ℕ) : ℚ :=
  (days.choose 2) * (prob_sunny^2 * prob_rain^(days-2))

theorem probability_of_two_sunny_days :
  prob_two_sunny_days (2/5) (3/5) 3 = 36/125 :=
by 
  sorry

end probability_of_two_sunny_days_l124_124334


namespace eggs_in_each_basket_is_four_l124_124947

theorem eggs_in_each_basket_is_four 
  (n : ℕ)
  (h1 : n ∣ 16) 
  (h2 : n ∣ 28) 
  (h3 : n ≥ 2) : 
  n = 4 :=
sorry

end eggs_in_each_basket_is_four_l124_124947


namespace ab_non_positive_l124_124815

theorem ab_non_positive (a b : ℝ) (h : 2011 * a + 2012 * b = 0) : a * b ≤ 0 :=
sorry

end ab_non_positive_l124_124815


namespace riverside_high_badges_l124_124817

/-- Given the conditions on the sums of consecutive prime badge numbers of the debate team members,
prove that Giselle's badge number is 1014, given that the current year is 2025.
-/
theorem riverside_high_badges (p1 p2 p3 p4 : ℕ) (hp1 : Prime p1) (hp2 : Prime p2) (hp3 : Prime p3) (hp4 : Prime p4)
    (hconsec : p1 < p2 ∧ p2 < p3 ∧ p3 < p4 ∧ p4 = p3 + 6)
    (h1 : ∃ x, p1 + p3 = x) (h2 : ∃ y, p1 + p2 = y) (h3 : ∃ z, p2 + p3 = z ∧ z ≤ 31) 
    (h4 : p3 + p4 = 2025) : p4 = 1014 :=
by sorry

end riverside_high_badges_l124_124817


namespace DebateClubOfficerSelection_l124_124818

-- Definitions based on the conditions
def members : Finset ℕ := Finset.range 25 -- Members are indexed from 0 to 24
def Simon := 0
def Rachel := 1
def John := 2

-- Conditions regarding the officers
def is_officer (x : ℕ) (pres sec tre : ℕ) : Prop := 
  x = pres ∨ x = sec ∨ x = tre

def Simon_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Simon pres sec tre) → (is_officer Rachel pres sec tre)

def Rachel_condition (pres sec tre : ℕ) : Prop :=
  (is_officer Rachel pres sec tre) → (is_officer Simon pres sec tre) ∨ (is_officer John pres sec tre)

-- Statement of the problem in Lean
theorem DebateClubOfficerSelection : ∃ (pres sec tre : ℕ), 
  pres ≠ sec ∧ sec ≠ tre ∧ pres ≠ tre ∧ 
  pres ∈ members ∧ sec ∈ members ∧ tre ∈ members ∧ 
  Simon_condition pres sec tre ∧
  Rachel_condition pres sec tre :=
sorry

end DebateClubOfficerSelection_l124_124818


namespace no_valid_triples_l124_124416

theorem no_valid_triples (a b c : ℕ) (h₁ : 1 ≤ a) (h₂ : a ≤ b) (h₃ : b ≤ c) (h₄ : 6 * (a * b + b * c + c * a) = a * b * c) : false :=
by
  sorry

end no_valid_triples_l124_124416


namespace number_of_blue_balls_l124_124345

theorem number_of_blue_balls (T : ℕ) (h1 : (1 / 4) * T = green) (h2 : (1 / 8) * T = blue)
    (h3 : (1 / 12) * T = yellow) (h4 : 26 = white) (h5 : green + blue + yellow + white = T) :
    blue = 6 :=
by
  sorry

end number_of_blue_balls_l124_124345


namespace average_age_of_students_l124_124498

theorem average_age_of_students (A : ℝ) (h1 : ∀ n : ℝ, n = 20 → A + 1 = n) (h2 : ∀ k : ℝ, k = 40 → 19 * A + k = 20 * (A + 1)) : A = 20 :=
by
  sorry

end average_age_of_students_l124_124498


namespace infinite_divisibility_of_2n_plus_n2_by_100_l124_124725

theorem infinite_divisibility_of_2n_plus_n2_by_100 :
  ∃ᶠ n in at_top, 100 ∣ (2^n + n^2) :=
sorry

end infinite_divisibility_of_2n_plus_n2_by_100_l124_124725


namespace max_square_side_length_l124_124761

theorem max_square_side_length (AC BC : ℝ) (hAC : AC = 3) (hBC : BC = 7) : 
  ∃ s : ℝ, s = 2.1 := by
  sorry

end max_square_side_length_l124_124761


namespace number_of_divisors_of_2744_l124_124069

-- Definition of the integer and its prime factorization
def two := 2
def seven := 7
def n := two^3 * seven^3

-- Define the property for the number of divisors
def num_divisors (n : ℕ) : ℕ := (3 + 1) * (3 + 1)

-- Main proof statement
theorem number_of_divisors_of_2744 : num_divisors n = 16 := by
  sorry

end number_of_divisors_of_2744_l124_124069


namespace interest_rate_proof_l124_124759

noncomputable def compound_interest_rate (P A : ℝ) (n : ℕ) (r : ℝ) : Prop :=
  A = P * (1 + r)^n

noncomputable def interest_rate (initial  final: ℝ) (years : ℕ) : ℝ := 
  (4: ℝ)^(1/(years: ℝ)) - 1

theorem interest_rate_proof :
  compound_interest_rate 8000 32000 36 (interest_rate 8000 32000 36) ∧
  abs (interest_rate 8000 32000 36 * 100 - 3.63) < 0.01 :=
by
  -- Conditions from the problem for compound interest
  -- Using the formula for interest rate and the condition checks
  sorry

end interest_rate_proof_l124_124759


namespace base_d_digit_difference_l124_124845

theorem base_d_digit_difference (A C d : ℕ) (h1 : d > 8)
  (h2 : d * A + C + (d * C + C) = 2 * d^2 + 3 * d + 2) :
  (A - C = d + 1) :=
sorry

end base_d_digit_difference_l124_124845


namespace abs_x_plus_7_eq_0_has_no_solution_l124_124310

theorem abs_x_plus_7_eq_0_has_no_solution : ¬∃ x : ℝ, |x| + 7 = 0 :=
by
  sorry

end abs_x_plus_7_eq_0_has_no_solution_l124_124310


namespace right_angled_triangle_side_length_l124_124635

theorem right_angled_triangle_side_length :
  ∃ c : ℕ, (c = 5) ∧ (3^2 + 4^2 = c^2) ∧ (c = 4 + 1) := by
  sorry

end right_angled_triangle_side_length_l124_124635


namespace total_reactions_eq_100_l124_124808

variable (x : ℕ) -- Total number of reactions.
variable (thumbs_up : ℕ) -- Number of "thumbs up" reactions.
variable (thumbs_down : ℕ) -- Number of "thumbs down" reactions.
variable (S : ℕ) -- Net Score.

-- Conditions
axiom thumbs_up_eq_75percent_reactions : thumbs_up = 3 * x / 4
axiom thumbs_down_eq_25percent_reactions : thumbs_down = x / 4
axiom score_definition : S = thumbs_up - thumbs_down
axiom initial_score : S = 50

theorem total_reactions_eq_100 : x = 100 :=
by 
  sorry

end total_reactions_eq_100_l124_124808


namespace days_to_complete_work_l124_124505

theorem days_to_complete_work {D : ℝ} (h1 : D > 0)
  (h2 : (1 / D) + (2 / D) = 0.3) :
  D = 10 :=
sorry

end days_to_complete_work_l124_124505


namespace quadratic_real_roots_l124_124692

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, (m - 1) * x^2 + 4 * x - 1 = 0) ↔ m ≥ -3 ∧ m ≠ 1 := 
by 
  sorry

end quadratic_real_roots_l124_124692


namespace four_by_four_increasing_matrices_l124_124977

noncomputable def count_increasing_matrices (n : ℕ) : ℕ := sorry

theorem four_by_four_increasing_matrices :
  count_increasing_matrices 4 = 320 :=
sorry

end four_by_four_increasing_matrices_l124_124977


namespace calculate_product_l124_124872

theorem calculate_product (a : ℝ) : 2 * a * (3 * a) = 6 * a^2 := by
  -- This will skip the proof, denoted by 'sorry'
  sorry

end calculate_product_l124_124872


namespace no_odd_total_rows_columns_l124_124673

open Function

def array_odd_column_row_count (n : ℕ) (array : ℕ → ℕ → ℤ) : Prop :=
  n % 2 = 1 ∧
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ array i j = -1 ∨ array i j = 1) →
  (∃ (rows cols : Finset ℕ),
    rows.card + cols.card = n ∧
    ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r) k = -1 ∧
    ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c) k = -1
    )

theorem no_odd_total_rows_columns (n : ℕ) (array : ℕ → ℕ → ℤ) :
  n % 2 = 1 →
  (∀ i j, 0 ≤ array i j ∧ array i j ≤ 1 ∧ (array i j = -1 ∨ array i j = 1)) →
  ¬ (∃ rows cols : Finset ℕ,
       rows.card + cols.card = n ∧
       ∀ r ∈ rows, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array r k = -1) ∧
       ∀ c ∈ cols, ∃ k, 0 < k ∧ (k % 2 = 1) ∧ (array c k = -1)) :=
by
  intros h_array
  sorry

end no_odd_total_rows_columns_l124_124673


namespace members_count_l124_124966

theorem members_count
  (n : ℝ)
  (h1 : 191.25 = n / 4) :
  n = 765 :=
by
  sorry

end members_count_l124_124966


namespace James_total_area_l124_124832

def initial_length : ℕ := 13
def initial_width : ℕ := 18
def increase_dimension : ℕ := 2
def number_of_new_rooms : ℕ := 4
def larger_room_multiplier : ℕ := 2

noncomputable def new_length : ℕ := initial_length + increase_dimension
noncomputable def new_width : ℕ := initial_width + increase_dimension
noncomputable def area_of_one_new_room : ℕ := new_length * new_width
noncomputable def total_area_of_4_rooms : ℕ := area_of_one_new_room * number_of_new_rooms
noncomputable def area_of_larger_room : ℕ := area_of_one_new_room * larger_room_multiplier
noncomputable def total_area : ℕ := total_area_of_4_rooms + area_of_larger_room

theorem James_total_area : total_area = 1800 := 
by
  sorry

end James_total_area_l124_124832


namespace multiples_of_4_count_l124_124638

theorem multiples_of_4_count (a b : ℕ) (h₁ : a = 100) (h₂ : b = 400) :
  ∃ n : ℕ, n = 75 ∧ ∀ k : ℕ, (k >= a ∧ k <= b ∧ k % 4 = 0) ↔ (k / 4 - 25 ≥ 1 ∧ k / 4 - 25 ≤ n) :=
sorry

end multiples_of_4_count_l124_124638


namespace movie_of_the_year_condition_l124_124317

theorem movie_of_the_year_condition (total_lists : ℕ) (fraction : ℚ) (num_lists : ℕ) 
  (h1 : total_lists = 775) (h2 : fraction = 1 / 4) (h3 : num_lists = ⌈fraction * total_lists⌉) : 
  num_lists = 194 :=
by
  -- Using the conditions given,
  -- total_lists = 775,
  -- fraction = 1 / 4,
  -- num_lists = ⌈fraction * total_lists⌉
  -- We need to show num_lists = 194.
  sorry

end movie_of_the_year_condition_l124_124317


namespace buckets_required_l124_124903

theorem buckets_required (C : ℝ) (N : ℝ):
  (62.5 * (2 / 5) * C = N * C) → N = 25 :=
by
  sorry

end buckets_required_l124_124903


namespace triangle_perimeter_l124_124920

theorem triangle_perimeter (a b : ℝ) (f : ℝ → Prop) 
  (h₁ : a = 7) (h₂ : b = 11)
  (eqn : ∀ x, f x ↔ x^2 - 25 = 2 * (x - 5)^2)
  (h₃ : ∃ x, f x ∧ 4 < x ∧ x < 18) :
  ∃ p : ℝ, (p = a + b + 5 ∨ p = a + b + 15) :=
by
  sorry

end triangle_perimeter_l124_124920


namespace turnip_count_example_l124_124045

theorem turnip_count_example : 6 + 9 = 15 := 
by
  -- Sorry is used to skip the actual proof
  sorry

end turnip_count_example_l124_124045


namespace perpendicular_condition_line_through_point_l124_124924

-- Definitions for lines l1 and l2
def l1 (m : ℝ) (x y : ℝ) : Prop := (m + 2) * x + m * y = 6
def l2 (m : ℝ) (x y : ℝ) : Prop := m * x + y = 3

-- Part 1: Prove that l1 is perpendicular to l2 if and only if m = -3 or m = 0
theorem perpendicular_condition (m : ℝ) : 
  (∀ (x : ℝ), ∀ (y : ℝ), (l1 m x y ∧ l2 m x y) → (m = 0 ∨ m = -3)) :=
sorry

-- Part 2: Prove the equations of line l given the conditions
theorem line_through_point (m : ℝ) (l : ℝ → ℝ → Prop) : 
  (∀ (P : ℝ × ℝ), (P = (1, 2*m)) → (l2 m P.1 P.2) → 
  ((∀ (x y : ℝ), l x y → 2 * x - y = 0) ∨ (∀ (x y: ℝ), l x y → x + 2 * y - 5 = 0))) :=
sorry

end perpendicular_condition_line_through_point_l124_124924


namespace area_of_fourth_square_l124_124577

theorem area_of_fourth_square (AB BC AC CD AD : ℝ) (h_sum_ABC : AB^2 + 25 = 50)
  (h_sum_ACD : 50 + 49 = AD^2) : AD^2 = 99 :=
by
  sorry

end area_of_fourth_square_l124_124577


namespace ratio_sum_l124_124788

theorem ratio_sum {x y : ℚ} (h : x / y = 4 / 7) : (x + y) / y = 11 / 7 :=
sorry

end ratio_sum_l124_124788


namespace squares_with_center_35_65_l124_124534

theorem squares_with_center_35_65 : 
  (∃ (n : ℕ), n = 1190 ∧ ∀ (x y : ℕ), x ≠ y → (x, y) = (35, 65)) :=
sorry

end squares_with_center_35_65_l124_124534


namespace total_lives_remaining_l124_124630

theorem total_lives_remaining (initial_players quit_players : Nat) 
  (lives_3_players lives_4_players lives_2_players bonus_lives : Nat)
  (h1 : initial_players = 16)
  (h2 : quit_players = 7)
  (h3 : lives_3_players = 10)
  (h4 : lives_4_players = 8)
  (h5 : lives_2_players = 6)
  (h6 : bonus_lives = 4)
  (remaining_players : Nat)
  (h7 : remaining_players = initial_players - quit_players)
  (lives_before_bonus : Nat)
  (h8 : lives_before_bonus = 3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players)
  (bonus_total : Nat)
  (h9 : bonus_total = remaining_players * bonus_lives) :
  3 * lives_3_players + 4 * lives_4_players + 2 * lives_2_players + remaining_players * bonus_lives = 110 :=
by
  sorry

end total_lives_remaining_l124_124630


namespace max_volume_tetrahedron_l124_124905

-- Definitions and conditions
def SA : ℝ := 4
def AB : ℝ := 5
def SB_min : ℝ := 7
def SC_min : ℝ := 9
def BC_max : ℝ := 6
def AC_max : ℝ := 8

-- Proof statement
theorem max_volume_tetrahedron {SB SC BC AC : ℝ} (hSB : SB ≥ SB_min) (hSC : SC ≥ SC_min) (hBC : BC ≤ BC_max) (hAC : AC ≤ AC_max) :
  ∃ V : ℝ, V = 8 * Real.sqrt 6 ∧ V ≤ (1/3) * (1/2) * SA * AB * (2 * Real.sqrt 6) * BC := by
  sorry

end max_volume_tetrahedron_l124_124905


namespace plates_difference_l124_124403

def num_plates_sunshine := 26^3 * 10^3
def num_plates_prairie := 26^2 * 10^4
def difference := num_plates_sunshine - num_plates_prairie

theorem plates_difference :
  difference = 10816000 := by sorry

end plates_difference_l124_124403


namespace modular_inverse_7_10000_l124_124277

theorem modular_inverse_7_10000 :
  (7 * 8571) % 10000 = 1 := 
sorry

end modular_inverse_7_10000_l124_124277


namespace smallest_number_conditions_l124_124824

theorem smallest_number_conditions :
  ∃ n : ℤ, (n > 0) ∧
           (n % 2 = 1) ∧
           (n % 3 = 1) ∧
           (n % 4 = 1) ∧
           (n % 5 = 1) ∧
           (n % 6 = 1) ∧
           (n % 11 = 0) ∧
           (∀ m : ℤ, (m > 0) → 
             (m % 2 = 1) ∧
             (m % 3 = 1) ∧
             (m % 4 = 1) ∧
             (m % 5 = 1) ∧
             (m % 6 = 1) ∧
             (m % 11 = 0) → 
             (n ≤ m)) :=
sorry

end smallest_number_conditions_l124_124824


namespace smallest_positive_integer_divisible_conditions_l124_124339

theorem smallest_positive_integer_divisible_conditions :
  ∃ (M : ℕ), M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M = 419 :=
sorry

end smallest_positive_integer_divisible_conditions_l124_124339


namespace inequality_solution_l124_124235

theorem inequality_solution :
  ∀ x : ℝ, (5 / 24 + |x - 11 / 48| < 5 / 16 ↔ (1 / 8 < x ∧ x < 1 / 3)) :=
by
  intro x
  sorry

end inequality_solution_l124_124235


namespace a_4_value_l124_124306

-- Definitions and Theorem
variable {α : Type*} [LinearOrderedField α]

noncomputable def geometric_seq (a₀ : α) (q : α) (n : ℕ) : α := a₀ * q ^ n

theorem a_4_value (a₁ : α) (q : α) (h : geometric_seq a₁ q 1 * geometric_seq a₁ q 2 * geometric_seq a₁ q 6 = 8) : 
  geometric_seq a₁ q 3 = 2 :=
sorry

end a_4_value_l124_124306


namespace parabola_line_intersection_l124_124174

theorem parabola_line_intersection :
  let a := (3 + Real.sqrt 11) / 2
  let b := (3 - Real.sqrt 11) / 2
  let p1 := (a, (9 + Real.sqrt 11) / 2)
  let p2 := (b, (9 - Real.sqrt 11) / 2)
  (3 * a^2 - 9 * a + 4 = (9 + Real.sqrt 11) / 2) ∧
  (-a^2 + 3 * a + 6 = (9 + Real.sqrt 11) / 2) ∧
  ((9 + Real.sqrt 11) / 2 = a + 3) ∧
  (3 * b^2 - 9 * b + 4 = (9 - Real.sqrt 11) / 2) ∧
  (-b^2 + 3 * b + 6 = (9 - Real.sqrt 11) / 2) ∧
  ((9 - Real.sqrt 11) / 2 = b + 3) :=
by
  sorry

end parabola_line_intersection_l124_124174


namespace geom_seq_sum_l124_124893

variable (a : ℕ → ℝ) (r : ℝ)
variable (h_geometric : ∀ n, a (n + 1) = a n * r)
variable (h_pos : ∀ n, a n > 0)
variable (h_equation : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25)

theorem geom_seq_sum : a 3 + a 5 = 5 :=
by sorry

end geom_seq_sum_l124_124893


namespace third_jumper_height_l124_124485

/-- 
  Ravi can jump 39 inches high.
  Ravi can jump 1.5 times higher than the average height of three other jumpers.
  The three jumpers can jump 23 inches, 27 inches, and some unknown height x.
  Prove that the unknown height x is 28 inches.
-/
theorem third_jumper_height (x : ℝ) (h₁ : 39 = 1.5 * (23 + 27 + x) / 3) : 
  x = 28 :=
sorry

end third_jumper_height_l124_124485


namespace eunji_initial_money_l124_124666

-- Define the conditions
def snack_cost : ℕ := 350
def allowance : ℕ := 800
def money_left_after_pencil : ℕ := 550

-- Define what needs to be proven
theorem eunji_initial_money (initial_money : ℕ) :
  initial_money - snack_cost + allowance = money_left_after_pencil * 2 →
  initial_money = 650 :=
by
  sorry

end eunji_initial_money_l124_124666


namespace total_insect_legs_l124_124793

/--
This Lean statement defines the conditions and question,
proving that given 5 insects in the laboratory and each insect
having 6 legs, the total number of insect legs is 30.
-/
theorem total_insect_legs (n_insects : Nat) (legs_per_insect : Nat) (h1 : n_insects = 5) (h2 : legs_per_insect = 6) : (n_insects * legs_per_insect) = 30 :=
by
  sorry

end total_insect_legs_l124_124793


namespace triangle_is_isosceles_l124_124820

theorem triangle_is_isosceles 
  (A B C : ℝ) 
  (h : (Real.sin A + Real.sin B) * (Real.cos A + Real.cos B) = 2 * Real.sin C) 
  (h₀ : A + B + C = π) :
  (A = B) := 
sorry

end triangle_is_isosceles_l124_124820


namespace train_length_l124_124074

theorem train_length
  (speed_km_hr : ℕ)
  (time_sec : ℕ)
  (length_train : ℕ)
  (length_platform : ℕ)
  (h_eq_len : length_train = length_platform)
  (h_speed : speed_km_hr = 108)
  (h_time : time_sec = 60) :
  length_train = 900 :=
by
  sorry

end train_length_l124_124074


namespace cost_per_first_30_kg_is_10_l124_124526

-- Definitions of the constants based on the conditions
def cost_per_33_kg (p q : ℝ) : Prop := 30 * p + 3 * q = 360
def cost_per_36_kg (p q : ℝ) : Prop := 30 * p + 6 * q = 420
def cost_per_25_kg (p : ℝ) : Prop := 25 * p = 250

-- The statement we want to prove
theorem cost_per_first_30_kg_is_10 (p q : ℝ) 
  (h1 : cost_per_33_kg p q)
  (h2 : cost_per_36_kg p q)
  (h3 : cost_per_25_kg p) : 
  p = 10 :=
sorry

end cost_per_first_30_kg_is_10_l124_124526


namespace bottles_more_than_apples_l124_124326

def regular_soda : ℕ := 72
def diet_soda : ℕ := 32
def apples : ℕ := 78

def total_bottles : ℕ := regular_soda + diet_soda

theorem bottles_more_than_apples : total_bottles - apples = 26 := by
  -- Proof will go here
  sorry

end bottles_more_than_apples_l124_124326


namespace smallest_b_satisfying_inequality_l124_124700

theorem smallest_b_satisfying_inequality : ∀ b : ℝ, (b^2 - 16 * b + 55) ≥ 0 ↔ b ≤ 5 ∨ b ≥ 11 := sorry

end smallest_b_satisfying_inequality_l124_124700


namespace part1_part2_l124_124853

open Complex

noncomputable def z1 : ℂ := 1 - 2 * I
noncomputable def z2 : ℂ := 4 + 3 * I

theorem part1 : z1 * z2 = 10 - 5 * I := by
  sorry

noncomputable def z : ℂ := -Real.sqrt 2 - Real.sqrt 2 * I

theorem part2 (h_abs_z : abs z = 2)
              (h_img_eq_real : z.im = (3 * z1 - z2).re)
              (h_quadrant : z.re < 0 ∧ z.im < 0) : z = -Real.sqrt 2 - Real.sqrt 2 * I := by
  sorry

end part1_part2_l124_124853


namespace correct_reasoning_l124_124705

-- Define that every multiple of 9 is a multiple of 3
def multiple_of_9_is_multiple_of_3 : Prop :=
  ∀ n : ℤ, n % 9 = 0 → n % 3 = 0

-- Define that a certain odd number is a multiple of 9
def odd_multiple_of_9 (n : ℤ) : Prop :=
  n % 2 = 1 ∧ n % 9 = 0

-- The goal: Prove that the reasoning process is completely correct
theorem correct_reasoning (H1 : multiple_of_9_is_multiple_of_3)
                          (n : ℤ)
                          (H2 : odd_multiple_of_9 n) : 
                          (n % 3 = 0) :=
by
  -- Explanation of the proof here
  sorry

end correct_reasoning_l124_124705


namespace workshop_cost_l124_124778

theorem workshop_cost
  (x : ℝ)
  (h1 : 0 < x) -- Given the cost must be positive
  (h2 : (x / 4) - 15 = x / 7) :
  x = 140 :=
by
  sorry

end workshop_cost_l124_124778


namespace initial_action_figures_correct_l124_124239

def initial_action_figures (x : ℕ) : Prop :=
  x + 11 - 10 = 8

theorem initial_action_figures_correct :
  ∃ x : ℕ, initial_action_figures x ∧ x = 7 :=
by
  sorry

end initial_action_figures_correct_l124_124239


namespace reciprocals_and_opposites_l124_124748

theorem reciprocals_and_opposites (a b c d : ℝ) (h_ab : a * b = 1) (h_cd : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
  sorry

end reciprocals_and_opposites_l124_124748


namespace square_side_length_l124_124800

theorem square_side_length (s : ℚ) (h : s^2 = 9/16) : s = 3/4 := 
by
sorry

end square_side_length_l124_124800


namespace comparison_of_f_values_l124_124242

noncomputable def f (x : ℝ) := Real.cos x - x

theorem comparison_of_f_values :
  f (8 * Real.pi / 9) > f Real.pi ∧ f Real.pi > f (10 * Real.pi / 9) :=
by
  sorry

end comparison_of_f_values_l124_124242


namespace geometric_sequence_a5_l124_124278

variable {a : ℕ → ℝ}
variable (h₁ : a 3 * a 7 = 3)
variable (h₂ : a 3 + a 7 = 4)

theorem geometric_sequence_a5 : a 5 = Real.sqrt 3 := 
sorry

end geometric_sequence_a5_l124_124278


namespace sector_central_angle_in_radians_l124_124681

/-- 
Given a sector of a circle where the perimeter is 4 cm 
and the area is 1 cm², prove that the central angle 
of the sector in radians is 2.
-/
theorem sector_central_angle_in_radians 
  (r l : ℝ) 
  (h_perimeter : 2 * r + l = 4) 
  (h_area : (1 / 2) * l * r = 1) : 
  l / r = 2 :=
by
  sorry

end sector_central_angle_in_radians_l124_124681


namespace driving_time_per_trip_l124_124812

-- Define the conditions
def filling_time_per_trip : ℕ := 15
def number_of_trips : ℕ := 6
def total_moving_hours : ℕ := 7
def total_moving_time : ℕ := total_moving_hours * 60

-- Define the problem
theorem driving_time_per_trip :
  (total_moving_time - (filling_time_per_trip * number_of_trips)) / number_of_trips = 55 :=
by
  sorry

end driving_time_per_trip_l124_124812


namespace equivalent_trigonometric_identity_l124_124293

variable (α : ℝ)

theorem equivalent_trigonometric_identity
  (h1 : α ∈ Set.Ioo (-(Real.pi/2)) 0)
  (h2 : Real.sin (α + (Real.pi/4)) = -1/3) :
  (Real.sin (2*α) / Real.cos ((Real.pi/4) - α)) = 7/3 := 
by
  sorry

end equivalent_trigonometric_identity_l124_124293


namespace calculate_expression_l124_124097

variable {x y : ℝ}

theorem calculate_expression (hx : x ≠ 0) (hy : y ≠ 0) (h : x = 3 / y) :
  (3 * x - 3 / x) * (3 * y + 3 / y) = 9 * x^2 - y^2 :=
by
  sorry

end calculate_expression_l124_124097


namespace remaining_grandchild_share_l124_124363

theorem remaining_grandchild_share 
  (total : ℕ) 
  (half_share : ℕ) 
  (remaining : ℕ) 
  (n : ℕ) 
  (total_eq : total = 124600)
  (half_share_eq : half_share = total / 2)
  (remaining_eq : remaining = total - half_share)
  (n_eq : n = 10) 
  : remaining / n = 6230 := 
by sorry

end remaining_grandchild_share_l124_124363


namespace tan_angle_PAB_correct_l124_124435

noncomputable def tan_angle_PAB (AB BC CA : ℝ) (P inside ABC : Prop) (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop) : ℝ :=
  180 / 329

theorem tan_angle_PAB_correct :
  ∀ (AB BC CA : ℝ)
    (P_inside_ABC : Prop)
    (PAB_angle_eq_PBC_angle_eq_PCA_angle : Prop),
    AB = 12 → BC = 15 → CA = 17 →
    (tan_angle_PAB AB BC CA P_inside_ABC PAB_angle_eq_PBC_angle_eq_PCA_angle) = 180 / 329 :=
by
  intros
  sorry

end tan_angle_PAB_correct_l124_124435


namespace study_tour_arrangement_l124_124442

def number_of_arrangements (classes routes : ℕ) (max_selected_route : ℕ) : ℕ :=
  if classes = 4 ∧ routes = 4 ∧ max_selected_route = 2 then 240 else 0

theorem study_tour_arrangement :
  number_of_arrangements 4 4 2 = 240 :=
by sorry

end study_tour_arrangement_l124_124442


namespace y_z_add_x_eq_160_l124_124668

theorem y_z_add_x_eq_160 (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * (y + z) = 132) (h5 : z * (x + y) = 180) (h6 : x * y * z = 160) :
  y * (z + x) = 160 := 
by 
  sorry

end y_z_add_x_eq_160_l124_124668


namespace train_passes_jogger_in_36_seconds_l124_124568

/-- A jogger runs at 9 km/h, 240m ahead of a train moving at 45 km/h.
The train is 120m long. Prove the train passes the jogger in 36 seconds. -/
theorem train_passes_jogger_in_36_seconds
  (distance_ahead : ℝ)
  (jogger_speed_km_hr train_speed_km_hr train_length_m : ℝ)
  (jogger_speed_m_s train_speed_m_s relative_speed_m_s distance_to_cover time_to_pass : ℝ)
  (h1 : distance_ahead = 240)
  (h2 : jogger_speed_km_hr = 9)
  (h3 : train_speed_km_hr = 45)
  (h4 : train_length_m = 120)
  (h5 : jogger_speed_m_s = jogger_speed_km_hr * 1000 / 3600)
  (h6 : train_speed_m_s = train_speed_km_hr * 1000 / 3600)
  (h7 : relative_speed_m_s = train_speed_m_s - jogger_speed_m_s)
  (h8 : distance_to_cover = distance_ahead + train_length_m)
  (h9 : time_to_pass = distance_to_cover / relative_speed_m_s) :
  time_to_pass = 36 := 
sorry

end train_passes_jogger_in_36_seconds_l124_124568


namespace reciprocal_of_minus_one_over_2023_l124_124609

theorem reciprocal_of_minus_one_over_2023 : (1 / (- (1 / 2023))) = -2023 := 
by
  sorry

end reciprocal_of_minus_one_over_2023_l124_124609


namespace rectangle_bounds_product_l124_124887

theorem rectangle_bounds_product (b : ℝ) :
  (∃ b, y = 3 ∧ y = 7 ∧ x = -1 ∧ (x = b) 
   → (b = 3 ∨ b = -5) 
    ∧ (3 * -5 = -15)) :=
sorry

end rectangle_bounds_product_l124_124887


namespace units_digit_17_times_29_l124_124358

theorem units_digit_17_times_29 :
  (17 * 29) % 10 = 3 :=
by
  sorry

end units_digit_17_times_29_l124_124358


namespace cinema_cost_comparison_l124_124489

theorem cinema_cost_comparison (x : ℕ) (hx : x = 1000) :
  let cost_A := if x ≤ 100 then 30 * x else 24 * x + 600
  let cost_B := 27 * x
  cost_A < cost_B :=
by
  sorry

end cinema_cost_comparison_l124_124489


namespace incorrect_population_growth_statement_l124_124781

def population_growth_behavior (p: ℝ → ℝ) : Prop :=
(p 0 < p 1) ∧ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))

def stabilizes_at_K (p: ℝ → ℝ) (K: ℝ) : Prop :=
∃ t₀, ∀ t > t₀, p t = K

def K_value_definition (K: ℝ) (environmental_conditions: ℝ → ℝ) : Prop :=
∀ t, environmental_conditions t = K

theorem incorrect_population_growth_statement (p: ℝ → ℝ) (K: ℝ) (environmental_conditions: ℝ → ℝ)
(h1: population_growth_behavior p)
(h2: stabilizes_at_K p K)
(h3: K_value_definition K environmental_conditions) :
(p 0 > p 1) ∨ (¬ (∃ t₁ t₂, t₁ < t₂ ∧ (∀ t < t₁, p t < p (t + 1)) ∧
 (∀ t > t₁, (p t < p (t - 1)) ∨ (p t = p (t - 1))))) :=
sorry

end incorrect_population_growth_statement_l124_124781


namespace geom_seq_value_l124_124752

noncomputable def geom_sequence (a : ℕ → ℝ) (q : ℝ) :=
∀ (n : ℕ), a (n + 1) = a n * q

theorem geom_seq_value
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom_seq : geom_sequence a q)
  (h_a5 : a 5 = 2)
  (h_a6_a8 : a 6 * a 8 = 8) :
  (a 2018 - a 2016) / (a 2014 - a 2012) = 2 :=
sorry

end geom_seq_value_l124_124752


namespace number_of_positive_integer_pairs_l124_124766

theorem number_of_positive_integer_pairs (x y : ℕ) (h : 20 * x + 6 * y = 2006) : 
  ∃ n, n = 34 ∧ ∀ (x y : ℕ), 20 * x + 6 * y = 2006 → 0 < x → 0 < y → 
  (∃ k, x = 3 * k + 1 ∧ y = 331 - 10 * k ∧ 0 ≤ k ∧ k ≤ 33) :=
sorry

end number_of_positive_integer_pairs_l124_124766


namespace simplest_quadratic_radicals_l124_124699

theorem simplest_quadratic_radicals (a : ℝ) :
  (3 * a - 8 ≥ 0) ∧ (17 - 2 * a ≥ 0) → a = 5 :=
by
  intro h
  sorry

end simplest_quadratic_radicals_l124_124699


namespace distance_light_travels_100_years_l124_124430

def distance_light_travels_one_year : ℝ := 5870e9 * 10^3

theorem distance_light_travels_100_years : distance_light_travels_one_year * 100 = 587 * 10^12 :=
by
  rw [distance_light_travels_one_year]
  sorry

end distance_light_travels_100_years_l124_124430


namespace largest_possible_sum_l124_124329

theorem largest_possible_sum (clubsuit heartsuit : ℕ) (h₁ : clubsuit * heartsuit = 48) (h₂ : Even clubsuit) : 
  clubsuit + heartsuit ≤ 26 :=
sorry

end largest_possible_sum_l124_124329


namespace hands_per_hoopit_l124_124678

-- Defining conditions
def num_hoopits := 7
def num_neglarts := 8
def total_toes := 164
def toes_per_hand_hoopit := 3
def toes_per_hand_neglart := 2
def hands_per_neglart := 5

-- The statement to prove
theorem hands_per_hoopit : 
  ∃ (H : ℕ), (H * toes_per_hand_hoopit * num_hoopits + hands_per_neglart * toes_per_hand_neglart * num_neglarts = total_toes) → H = 4 :=
sorry

end hands_per_hoopit_l124_124678


namespace bottle_t_capsules_l124_124613

theorem bottle_t_capsules 
  (num_capsules_r : ℕ)
  (cost_r : ℝ)
  (cost_t : ℝ)
  (cost_per_capsule_difference : ℝ)
  (h1 : num_capsules_r = 250)
  (h2 : cost_r = 6.25)
  (h3 : cost_t = 3.00)
  (h4 : cost_per_capsule_difference = 0.005) :
  ∃ (num_capsules_t : ℕ), num_capsules_t = 150 := 
by
  sorry

end bottle_t_capsules_l124_124613


namespace total_handshakes_l124_124672

theorem total_handshakes (players_team1 players_team2 referees : ℕ) 
  (h1 : players_team1 = 11) (h2 : players_team2 = 11) (h3 : referees = 3) : 
  players_team1 * players_team2 + (players_team1 + players_team2) * referees = 187 := 
by
  sorry

end total_handshakes_l124_124672


namespace cars_in_fourth_store_l124_124390

theorem cars_in_fourth_store
  (mean : ℝ) 
  (a1 a2 a3 a5 : ℝ) 
  (num_stores : ℝ) 
  (mean_value : mean = 20.8) 
  (a1_value : a1 = 30) 
  (a2_value : a2 = 14) 
  (a3_value : a3 = 14) 
  (a5_value : a5 = 25) 
  (num_stores_value : num_stores = 5) :
  ∃ x : ℝ, (a1 + a2 + a3 + x + a5) / num_stores = mean ∧ x = 21 :=
by
  sorry

end cars_in_fourth_store_l124_124390


namespace product_of_two_numbers_l124_124995

theorem product_of_two_numbers (x y : ℝ) 
  (h1 : x ^ 2 + y ^ 2 = 289)
  (h2 : x + y = 23) : 
  x * y = 120 :=
by
  sorry

end product_of_two_numbers_l124_124995


namespace line_equation_l124_124515

theorem line_equation {L : ℝ → ℝ → Prop} (h1 : L (-3) (-2)) 
  (h2 : ∃ a : ℝ, a ≠ 0 ∧ (L a 0 ∧ L 0 a)) :
  (∀ x y, L x y ↔ 2 * x - 3 * y = 0) ∨ (∀ x y, L x y ↔ x + y + 5 = 0) :=
by 
  sorry

end line_equation_l124_124515


namespace average_of_original_set_l124_124789

theorem average_of_original_set (A : ℝ) (h1 : 7 * A = 125 * 7 / 5) : A = 25 := 
sorry

end average_of_original_set_l124_124789


namespace min_value_fraction_l124_124303

theorem min_value_fraction (x y : ℝ) (h₁ : x + y = 4) (h₂ : x > y) (h₃ : y > 0) : (∃ z : ℝ, z = (2 / (x - y)) + (1 / y) ∧ z = 2) :=
by
  sorry

end min_value_fraction_l124_124303


namespace unique_value_expression_l124_124336

theorem unique_value_expression (m n : ℤ) : 
  (mn + 13 * m + 13 * n - m^2 - n^2 = 169) → 
  ∃! (m n : ℤ), mn + 13 * m + 13 * n - m^2 - n^2 = 169 := 
by
  sorry

end unique_value_expression_l124_124336


namespace problem_solution_sets_l124_124709

theorem problem_solution_sets (x y : ℝ) :
  (x^2 * y + y^3 = 2 * x^2 + 2 * y^2 ∧ x * y + 1 = x + y) →
  ( (x = 0 ∧ y = 0) ∨ y = 2 ∨ x = 1 ∨ y = 1 ) :=
by
  sorry

end problem_solution_sets_l124_124709


namespace contrapositive_of_original_l124_124376

theorem contrapositive_of_original (a b : ℝ) :
  (a > b → a - 1 > b - 1) ↔ (a - 1 ≤ b - 1 → a ≤ b) :=
by
  sorry

end contrapositive_of_original_l124_124376


namespace forgot_days_l124_124463

def July_days : ℕ := 31
def days_took_capsules : ℕ := 27

theorem forgot_days : July_days - days_took_capsules = 4 :=
by
  sorry

end forgot_days_l124_124463


namespace floor_difference_l124_124755

theorem floor_difference (x : ℝ) (h : x = 15.3) : 
  (⌊ x^2 ⌋ - ⌊ x ⌋ * ⌊ x ⌋ + 5) = 14 := 
by
  -- Skipping proof
  sorry

end floor_difference_l124_124755


namespace precisely_hundred_million_l124_124928

-- Defining the options as an enumeration type
inductive Precision
| HundredBillion
| Billion
| HundredMillion
| Percent

-- The given figure in billions
def givenFigure : Float := 21.658

-- The correct precision is HundredMillion
def correctPrecision : Precision := Precision.HundredMillion

-- The theorem to prove the correctness of the figure's precision
theorem precisely_hundred_million : correctPrecision = Precision.HundredMillion :=
by
  sorry

end precisely_hundred_million_l124_124928


namespace dual_cassette_recorder_price_l124_124268

theorem dual_cassette_recorder_price :
  ∃ (x y : ℝ),
    (x - 0.05 * x = 380) ∧
    (y = x + 0.08 * x) ∧ 
    (y = 432) :=
by
  -- sorry to skip the proof.
  sorry

end dual_cassette_recorder_price_l124_124268


namespace adult_elephant_weekly_bananas_l124_124241

theorem adult_elephant_weekly_bananas (daily_bananas : Nat) (days_in_week : Nat) (H1 : daily_bananas = 90) (H2 : days_in_week = 7) :
  daily_bananas * days_in_week = 630 :=
by
  sorry

end adult_elephant_weekly_bananas_l124_124241


namespace valid_interval_for_a_l124_124256

theorem valid_interval_for_a (a : ℝ) :
  (6 - 3 * a > 0) ∧ (a > 0) ∧ (3 * a^2 + a - 2 ≥ 0) ↔ (2 / 3 ≤ a ∧ a < 2 ∧ a ≠ 5 / 3) :=
by
  sorry

end valid_interval_for_a_l124_124256


namespace range_of_a_l124_124244

noncomputable def g (a x : ℝ) : ℝ := x ^ 2 - 2 * a * x + 3

theorem range_of_a 
  (h_mono_inc : ∀ x1 x2 : ℝ, -1 < x1 ∧ x1 < x2 ∧ x2 < 1 → g a x1 ≤ g a x2)
  (h_nonneg : ∀ x : ℝ, -1 < x ∧ x < 1 → 0 ≤ g a x) :
  (-2 : ℝ) ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l124_124244


namespace circumscribed_circle_radius_l124_124830

theorem circumscribed_circle_radius (r : ℝ) (π : ℝ)
  (isosceles_right_triangle : Type) 
  (perimeter : isosceles_right_triangle → ℝ )
  (area : ℝ → ℝ)
  (h : ∀ (t : isosceles_right_triangle), perimeter t = area r) :
  r = (1 + Real.sqrt 2) / π :=
sorry

end circumscribed_circle_radius_l124_124830


namespace lollipop_distribution_l124_124770

theorem lollipop_distribution 
  (P1 P2 P_total L x : ℕ) 
  (h1 : P1 = 45) 
  (h2 : P2 = 15) 
  (h3 : L = 12) 
  (h4 : P_total = P1 + P2) 
  (h5 : P_total = 60) : 
  x = 5 := 
by 
  sorry

end lollipop_distribution_l124_124770


namespace number_of_four_digit_integers_with_digit_sum_nine_l124_124741

theorem number_of_four_digit_integers_with_digit_sum_nine :
  ∃ (n : ℕ), (n = 165) ∧ (
    ∃ (a b c d : ℕ), 
      1 ≤ a ∧ 
      a + b + c + d = 9 ∧ 
      (1 ≤ a ∧ a ≤ 9) ∧ 
      (0 ≤ b ∧ b ≤ 9) ∧ 
      (0 ≤ c ∧ c ≤ 9) ∧ 
      (0 ≤ d ∧ d ≤ 9)) := 
sorry

end number_of_four_digit_integers_with_digit_sum_nine_l124_124741


namespace minimum_value_of_eccentricity_sum_l124_124004

variable {a b m n c : ℝ} (ha : a > b) (hb : b > 0) (hm : m > 0) (hn : n > 0)
variable {e1 e2 : ℝ}

theorem minimum_value_of_eccentricity_sum 
  (h_equiv : a^2 + m^2 = 2 * c^2) 
  (e1_def : e1 = c / a) 
  (e2_def : e2 = c / m) : 
  (2 * e1^2 + (e2^2) / 2) = (9 / 4) :=
sorry

end minimum_value_of_eccentricity_sum_l124_124004


namespace little_sister_stole_roses_l124_124901

/-- Ricky has 40 roses. His little sister steals some roses. He wants to give away the rest of the roses in equal portions to 9 different people, and each person gets 4 roses. Prove how many roses his little sister stole. -/
theorem little_sister_stole_roses (total_roses stolen_roses remaining_roses people roses_per_person : ℕ)
  (h1 : total_roses = 40)
  (h2 : people = 9)
  (h3 : roses_per_person = 4)
  (h4 : remaining_roses = people * roses_per_person)
  (h5 : remaining_roses = total_roses - stolen_roses) :
  stolen_roses = 4 :=
by
  sorry

end little_sister_stole_roses_l124_124901


namespace find_middle_number_l124_124476

theorem find_middle_number (x y z : ℕ) (h1 : x < y) (h2 : y < z)
  (h3 : x + y = 22) (h4 : x + z = 29) (h5 : y + z = 31) (h6 : x = 10) :
  y = 12 :=
sorry

end find_middle_number_l124_124476


namespace length_of_CD_l124_124201

theorem length_of_CD (x y: ℝ) (h1: 5 * x = 3 * y) (u v: ℝ) (h2: u = x + 3) (h3: v = y - 3) (h4: 7 * u = 4 * v) : x + y = 264 :=
by
  sorry

end length_of_CD_l124_124201


namespace algebra_expression_opposite_l124_124205

theorem algebra_expression_opposite (a : ℚ) :
  3 * a + 1 = -(3 * (a - 1)) → a = 1 / 3 :=
by
  intro h
  sorry

end algebra_expression_opposite_l124_124205


namespace shaded_area_l124_124806

theorem shaded_area (R : ℝ) (r : ℝ) (hR : R = 10) (hr : r = R / 2) : 
  π * R^2 - 2 * (π * r^2) = 50 * π :=
by
  sorry

end shaded_area_l124_124806


namespace triangle_side_lengths_relationship_l124_124134

variable {a b c : ℝ}

def is_quadratic_mean (a b c : ℝ) : Prop :=
  (2 * b^2 = a^2 + c^2)

def is_geometric_mean (a b c : ℝ) : Prop :=
  (b * a = c^2)

theorem triangle_side_lengths_relationship (a b c : ℝ) :
  (is_quadratic_mean a b c ∧ is_geometric_mean a b c) → 
  ∃ a b c, (2 * b^2 = a^2 + c^2) ∧ (b * a = c^2) :=
sorry

end triangle_side_lengths_relationship_l124_124134


namespace prob_A_and_B_truth_is_0_48_l124_124101

-- Conditions: Define the probabilities
def prob_A_truth : ℝ := 0.8
def prob_B_truth : ℝ := 0.6

-- Target: Define the probability that both A and B tell the truth at the same time.
def prob_A_and_B_truth : ℝ := prob_A_truth * prob_B_truth

-- Statement: Prove that the probability that both A and B tell the truth at the same time is 0.48.
theorem prob_A_and_B_truth_is_0_48 : prob_A_and_B_truth = 0.48 := by
  sorry

end prob_A_and_B_truth_is_0_48_l124_124101


namespace number_of_real_roots_l124_124425

theorem number_of_real_roots (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : a * b^2 + 1 = 0) :
  (c > 0 → ∃ x1 x2 x3 : ℝ, 
    (x1 = b * Real.sqrt c ∨ x1 = -b * Real.sqrt c ∨ x1 = -c / b) ∧
    (x2 = b * Real.sqrt c ∨ x2 = -b * Real.sqrt c ∨ x2 = -c / b) ∧
    (x3 = b * Real.sqrt c ∨ x3 = -b * Real.sqrt c ∨ x3 = -c / b)) ∧
  (c < 0 → ∃ x1 : ℝ, x1 = -c / b) :=
by
  sorry

end number_of_real_roots_l124_124425


namespace total_amount_l124_124671

variable (A B C : ℕ)
variable (h1 : C = 495)
variable (h2 : (A - 10) * 18 = (B - 20) * 11)
variable (h3 : (B - 20) * 24 = (C - 15) * 18)

theorem total_amount (A B C : ℕ) (h1 : C = 495)
  (h2 : (A - 10) * 18 = (B - 20) * 11)
  (h3 : (B - 20) * 24 = (C - 15) * 18) :
  A + B + C = 1105 :=
sorry

end total_amount_l124_124671


namespace Finn_initial_goldfish_l124_124120

variable (x : ℕ)

-- Defining the conditions
def number_of_goldfish_initial (x : ℕ) : Prop :=
  ∃ y z : ℕ, y = 32 ∧ z = 57 ∧ x = y + z 

-- Theorem statement to prove Finn's initial number of goldfish
theorem Finn_initial_goldfish (x : ℕ) (h : number_of_goldfish_initial x) : x = 89 := by
  sorry

end Finn_initial_goldfish_l124_124120


namespace polynomial_evaluation_l124_124967

-- Define the polynomial p(x) and the condition p(x) - p'(x) = x^2 + 2x + 1
variable (p : ℝ → ℝ)
variable (hp : ∀ x, p x - (deriv p x) = x^2 + 2 * x + 1)

-- Statement to prove p(5) = 50 given the conditions
theorem polynomial_evaluation : p 5 = 50 := 
sorry

end polynomial_evaluation_l124_124967


namespace correct_product_of_0_035_and_3_84_l124_124764

theorem correct_product_of_0_035_and_3_84 : 
  (0.035 * 3.84 = 0.1344) := sorry

end correct_product_of_0_035_and_3_84_l124_124764


namespace diamond_2_3_l124_124437

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_2_3 : diamond 2 3 = 16 :=
by
  -- Imported definition and theorem structure.
  sorry

end diamond_2_3_l124_124437


namespace bowling_ball_weight_l124_124461

theorem bowling_ball_weight (b c : ℕ) (h1 : 10 * b = 5 * c) (h2 : 3 * c = 120) : b = 20 := by
  sorry

end bowling_ball_weight_l124_124461


namespace jerry_won_games_l124_124473

theorem jerry_won_games 
  (T : ℕ) (K D J : ℕ) 
  (h1 : T = 32) 
  (h2 : K = D + 5) 
  (h3 : D = J + 3) : 
  J = 7 := 
sorry

end jerry_won_games_l124_124473


namespace remainder_98765432101_div_240_l124_124451

theorem remainder_98765432101_div_240 :
  (98765432101 % 240) = 61 :=
by
  -- Proof to be filled in later
  sorry

end remainder_98765432101_div_240_l124_124451


namespace number_of_pieces_of_bubble_gum_l124_124197

theorem number_of_pieces_of_bubble_gum (cost_per_piece total_cost : ℤ) (h1 : cost_per_piece = 18) (h2 : total_cost = 2448) :
  total_cost / cost_per_piece = 136 :=
by
  rw [h1, h2]
  norm_num

end number_of_pieces_of_bubble_gum_l124_124197


namespace mrs_hilt_baked_pecan_pies_l124_124428

def total_pies (rows : ℕ) (pies_per_row : ℕ) : ℕ :=
  rows * pies_per_row

def pecan_pies (total_pies : ℕ) (apple_pies : ℕ) : ℕ :=
  total_pies - apple_pies

theorem mrs_hilt_baked_pecan_pies :
  let apple_pies := 14
  let rows := 6
  let pies_per_row := 5
  let total := total_pies rows pies_per_row
  pecan_pies total apple_pies = 16 :=
by
  sorry

end mrs_hilt_baked_pecan_pies_l124_124428


namespace exp_sum_l124_124470

theorem exp_sum (a x y : ℝ) (h1 : a^x = 2) (h2 : a^y = 3) : a^(2 * x + 3 * y) = 108 :=
sorry

end exp_sum_l124_124470


namespace graph_equiv_l124_124058

theorem graph_equiv {x y : ℝ} :
  (x^3 - 2 * x^2 * y + x * y^2 - 2 * y^3 = 0) ↔ (x = 2 * y) :=
sorry

end graph_equiv_l124_124058


namespace sufficient_and_necessary_l124_124166

theorem sufficient_and_necessary (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := by
  sorry

end sufficient_and_necessary_l124_124166


namespace pseudo_code_output_l124_124661

theorem pseudo_code_output (a b c : Int)
  (h1 : a = 3)
  (h2 : b = -5)
  (h3 : c = 8)
  (ha : a = -5)
  (hb : b = 8)
  (hc : c = -5) : 
  a = -5 ∧ b = 8 ∧ c = -5 :=
by
  sorry

end pseudo_code_output_l124_124661


namespace mask_price_reduction_l124_124407

theorem mask_price_reduction 
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (additional_sales_factor : ℝ)
  (desired_profit : ℝ)
  (x : ℝ)
  (h_initial_sales : initial_sales = 500)
  (h_initial_profit : initial_profit = 0.6)
  (h_additional_sales_factor : additional_sales_factor = 100 / 0.1)
  (h_desired_profit : desired_profit = 240) :
  (initial_profit - x) * (initial_sales + additional_sales_factor * x) = desired_profit → x = 0.3 :=
sorry

end mask_price_reduction_l124_124407


namespace true_proposition_among_provided_l124_124267

theorem true_proposition_among_provided :
  ∃ (x0 : ℝ), |x0| ≤ 0 :=
by
  exists 0
  simp

end true_proposition_among_provided_l124_124267


namespace fraction_comparison_l124_124123

theorem fraction_comparison :
  (2 : ℝ) * (4 : ℝ) > (7 : ℝ) → (4 / 7 : ℝ) > (1 / 2 : ℝ) :=
by
  sorry

end fraction_comparison_l124_124123


namespace int_solve_ineq_l124_124555

theorem int_solve_ineq (x : ℤ) : (x + 3)^3 ≤ 8 ↔ x ≤ -1 :=
by sorry

end int_solve_ineq_l124_124555


namespace ratio_of_parts_l124_124308

theorem ratio_of_parts (N : ℝ) (h1 : (1/4) * (2/5) * N = 14) (h2 : 0.40 * N = 168) : (2/5) * N / N = 1 / 2.5 :=
by
  sorry

end ratio_of_parts_l124_124308


namespace a_eq_b_if_b2_ab_1_divides_a2_ab_1_l124_124282

theorem a_eq_b_if_b2_ab_1_divides_a2_ab_1 (a b : ℕ) (ha_pos : a > 0) (hb_pos : b > 0)
  (h : b^2 + a * b + 1 ∣ a^2 + a * b + 1) : a = b :=
by
  sorry

end a_eq_b_if_b2_ab_1_divides_a2_ab_1_l124_124282


namespace prove_fraction_identity_l124_124547

theorem prove_fraction_identity 
  (x y z : ℝ)
  (h1 : (x * z) / (x + y) + (y * z) / (y + z) + (x * y) / (z + x) = -18)
  (h2 : (z * y) / (x + y) + (z * x) / (y + z) + (y * x) / (z + x) = 20) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 20.5 := 
by
  sorry

end prove_fraction_identity_l124_124547


namespace sandbag_weight_l124_124586

theorem sandbag_weight (s : ℝ) (f : ℝ) (h : ℝ) : 
  f = 0.75 ∧ s = 450 ∧ h = 0.65 → f * s + h * (f * s) = 556.875 :=
by
  intro hfs
  sorry

end sandbag_weight_l124_124586


namespace problem1_l124_124620

theorem problem1 (x : ℝ) (n : ℕ) (h : x^n = 2) : (3 * x^n)^2 - 4 * (x^2)^n = 20 :=
by
  sorry

end problem1_l124_124620


namespace price_of_shoes_on_tuesday_is_correct_l124_124716

theorem price_of_shoes_on_tuesday_is_correct :
  let price_thursday : ℝ := 30
  let price_friday : ℝ := price_thursday * 1.2
  let price_monday : ℝ := price_friday - price_friday * 0.15
  let price_tuesday : ℝ := price_monday - price_monday * 0.1
  price_tuesday = 27.54 := 
by
  sorry

end price_of_shoes_on_tuesday_is_correct_l124_124716


namespace equal_number_of_digits_l124_124786

noncomputable def probability_equal_digits : ℚ := (20 * (9/16)^3 * (7/16)^3)

theorem equal_number_of_digits :
  probability_equal_digits = 3115125 / 10485760 := by
  sorry

end equal_number_of_digits_l124_124786


namespace sequence_is_geometric_and_general_formula_l124_124423

theorem sequence_is_geometric_and_general_formula (a : ℕ → ℝ) (h0 : a 1 = 2 / 3)
  (h1 : ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) / (a (n + 1) + 1)) :
  ∃ r : ℝ, (0 < r ∧ r < 1 ∧ (∀ n : ℕ, a (n + 1) = (2:ℝ)^n / (1 + (2:ℝ)^n)) ∧
  ∀ n : ℕ, (1 / a (n + 1) - 1) = (1 / 2) * (1 / a n - 1)) := sorry

end sequence_is_geometric_and_general_formula_l124_124423


namespace brick_surface_area_l124_124590

theorem brick_surface_area (l w h : ℝ) (hl : l = 10) (hw : w = 4) (hh : h = 3) : 
  2 * (l * w + l * h + w * h) = 164 := 
by
  sorry

end brick_surface_area_l124_124590


namespace exists_unique_n_digit_number_with_one_l124_124421

def n_digit_number (n : ℕ) : Type := {l : List ℕ // l.length = n ∧ ∀ x ∈ l, x = 1 ∨ x = 2 ∨ x = 3}

theorem exists_unique_n_digit_number_with_one (n : ℕ) (hn : n > 0) :
  ∃ x : n_digit_number n, x.val.count 1 = 1 ∧ ∀ y : n_digit_number n, y ≠ x → x.val.append [1] ≠ y.val.append [1] :=
sorry

end exists_unique_n_digit_number_with_one_l124_124421


namespace quadratic_root_l124_124645

theorem quadratic_root (k : ℝ) (h : ∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) : k = -1 :=
sorry

end quadratic_root_l124_124645


namespace temperature_difference_l124_124987

theorem temperature_difference (highest lowest : ℝ) (h_high : highest = 27) (h_low : lowest = 17) :
  highest - lowest = 10 :=
by
  sorry

end temperature_difference_l124_124987


namespace total_sum_of_ages_l124_124015

theorem total_sum_of_ages (Y : ℕ) (interval : ℕ) (age1 age2 age3 age4 age5 : ℕ)
  (h1 : Y = 2) 
  (h2 : interval = 8) 
  (h3 : age1 = Y) 
  (h4 : age2 = Y + interval) 
  (h5 : age3 = Y + 2 * interval) 
  (h6 : age4 = Y + 3 * interval) 
  (h7 : age5 = Y + 4 * interval) : 
  age1 + age2 + age3 + age4 + age5 = 90 := 
by
  sorry

end total_sum_of_ages_l124_124015


namespace product_check_l124_124321

theorem product_check : 
  (1200 < 31 * 53 ∧ 31 * 53 < 2400) ∧ 
  ¬ (1200 < 32 * 84 ∧ 32 * 84 < 2400) ∧ 
  ¬ (1200 < 63 * 54 ∧ 63 * 54 < 2400) ∧ 
  (1200 < 72 * 24 ∧ 72 * 24 < 2400) :=
by 
  sorry

end product_check_l124_124321


namespace no_integer_solutions_l124_124285

theorem no_integer_solutions (a b c : ℤ) : ¬ (a^2 + b^2 = 8 * c + 6) :=
sorry

end no_integer_solutions_l124_124285


namespace find_fraction_l124_124953

theorem find_fraction 
  (f : ℚ) (t k : ℚ)
  (h1 : t = f * (k - 32)) 
  (h2 : t = 75)
  (h3 : k = 167) : 
  f = 5 / 9 :=
by
  sorry

end find_fraction_l124_124953


namespace min_value_of_f_l124_124441

noncomputable def f (x : ℝ) : ℝ := x^2 + 8 * x + 3

theorem min_value_of_f : ∃ x₀ : ℝ, (∀ x : ℝ, f x ≥ f x₀) ∧ f x₀ = -13 :=
by
  sorry

end min_value_of_f_l124_124441


namespace inversely_directly_proportional_l124_124036

theorem inversely_directly_proportional (m n z : ℝ) (x : ℝ) (h₁ : x = 4) (hz₁ : z = 16) (hz₂ : z = 64) (hy : ∃ y : ℝ, y = n * Real.sqrt z) (hx : ∃ m y : ℝ, x = m / y^2)
: x = 1 :=
by
  sorry

end inversely_directly_proportional_l124_124036


namespace mean_and_sum_l124_124009

-- Define the sum of five numbers to be 1/3
def sum_of_five_numbers : ℚ := 1 / 3

-- Define the mean of these five numbers
def mean_of_five_numbers : ℚ := sum_of_five_numbers / 5

-- State the theorem
theorem mean_and_sum (h : sum_of_five_numbers = 1 / 3) :
  mean_of_five_numbers = 1 / 15 ∧ (mean_of_five_numbers + sum_of_five_numbers = 2 / 5) :=
by
  sorry

end mean_and_sum_l124_124009


namespace number_of_female_athletes_l124_124865

theorem number_of_female_athletes (male_athletes female_athletes male_selected female_selected : ℕ)
  (h1 : male_athletes = 56)
  (h2 : female_athletes = 42)
  (h3 : male_selected = 8)
  (ratio : male_athletes / female_athletes = 4 / 3)
  (stratified_sampling : female_selected = (3 / 4) * male_selected)
  : female_selected = 6 := by
  sorry

end number_of_female_athletes_l124_124865


namespace find_number_l124_124918

theorem find_number (x : ℝ) (h : 0.40 * x - 11 = 23) : x = 85 :=
sorry

end find_number_l124_124918


namespace find_positive_number_l124_124569

theorem find_positive_number (x : ℝ) (h_pos : 0 < x) (h_eq : (2 / 3) * x = (49 / 216) * (1 / x)) : x = 24.5 :=
by
  sorry

end find_positive_number_l124_124569


namespace hyperbola_foci_coordinates_l124_124346

theorem hyperbola_foci_coordinates :
  ∀ x y : ℝ, (x^2 / 4) - (y^2 / 12) = 1 → (x, y) = (4, 0) ∨ (x, y) = (-4, 0) :=
by
  -- We assume the given equation of the hyperbola
  intro x y h
  -- sorry is used to skip the actual proof steps
  sorry

end hyperbola_foci_coordinates_l124_124346


namespace find_k_l124_124021

theorem find_k (k : ℝ) : (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 :=
by
  sorry

end find_k_l124_124021


namespace gemstones_count_l124_124320

theorem gemstones_count (F B S W SN : ℕ) 
  (hS : S = 1)
  (hSpaatz : S = F / 2 - 2)
  (hBinkie : B = 4 * F)
  (hWhiskers : W = S + 3)
  (hSnowball : SN = 2 * W) :
  B = 24 :=
by
  sorry

end gemstones_count_l124_124320


namespace abs_neg_2023_l124_124753

theorem abs_neg_2023 : |(-2023)| = 2023 := by
  sorry

end abs_neg_2023_l124_124753


namespace factorize_expression_l124_124359

theorem factorize_expression (x y : ℝ) : x^2 * y - 2 * x * y^2 + y^3 = y * (x - y)^2 := 
sorry

end factorize_expression_l124_124359


namespace range_of_m_l124_124850

theorem range_of_m (p q : Prop) (m : ℝ) (h₀ : ∀ x : ℝ, p ↔ (x^2 - 8 * x - 20 ≤ 0)) 
  (h₁ : ∀ x : ℝ, q ↔ (x^2 - 2 * x + 1 - m^2 ≤ 0)) (hm : m > 0) 
  (hsuff : (∃ x : ℝ, x > 10 ∨ x < -2) → (∃ x : ℝ, x < 1 - m ∨ x > 1 + m)) :
  0 < m ∧ m ≤ 3 :=
sorry

end range_of_m_l124_124850


namespace curve_equation_with_params_l124_124352

theorem curve_equation_with_params (a m x y : ℝ) (ha : a > 0) (hm : m ≠ 0) :
    (y^2) = m * (x^2 - a^2) ↔ mx^2 - y^2 = ma^2 := by
  sorry

end curve_equation_with_params_l124_124352


namespace hilton_final_marbles_l124_124985

theorem hilton_final_marbles :
  let initial_marbles := 26
  let found_marbles := 6
  let lost_marbles := 10
  let gift_multiplication_factor := 2
  let marbles_after_find_and_lose := initial_marbles + found_marbles - lost_marbles
  let gift_marbles := gift_multiplication_factor * lost_marbles
  let final_marbles := marbles_after_find_and_lose + gift_marbles
  final_marbles = 42 :=
by
  -- Proof to be filled
  sorry

end hilton_final_marbles_l124_124985


namespace ratio_sheep_horses_l124_124260

theorem ratio_sheep_horses
  (horse_food_per_day : ℕ)
  (total_horse_food : ℕ)
  (number_of_sheep : ℕ)
  (number_of_horses : ℕ)
  (gcd_sheep_horses : ℕ):
  horse_food_per_day = 230 →
  total_horse_food = 12880 →
  number_of_sheep = 40 →
  number_of_horses = total_horse_food / horse_food_per_day →
  gcd number_of_sheep number_of_horses = 8 →
  (number_of_sheep / gcd_sheep_horses = 5) ∧ (number_of_horses / gcd_sheep_horses = 7) :=
by
  intros
  sorry

end ratio_sheep_horses_l124_124260


namespace flashlight_distance_difference_l124_124041

/--
Veronica's flashlight can be seen from 1000 feet. Freddie's flashlight can be seen from a distance
three times that of Veronica's flashlight. Velma's flashlight can be seen from a distance 2000 feet
less than 5 times Freddie's flashlight distance. We want to prove that Velma's flashlight can be seen 
12000 feet farther than Veronica's flashlight.
-/
theorem flashlight_distance_difference :
  let v_d := 1000
  let f_d := 3 * v_d
  let V_d := 5 * f_d - 2000
  V_d - v_d = 12000 := by
    sorry

end flashlight_distance_difference_l124_124041


namespace distinct_int_divisible_by_12_l124_124180

variable {a b c d : ℤ}

theorem distinct_int_divisible_by_12 (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) :
  12 ∣ (a - b) * (a - c) * (a - d) * (b - c) * (b - d) * (c - d) :=
by
  sorry

end distinct_int_divisible_by_12_l124_124180


namespace smallest_integer_l124_124491

theorem smallest_integer (n : ℕ) (h : n > 0) (h1 : lcm 36 n / gcd 36 n = 24) : n = 96 :=
sorry

end smallest_integer_l124_124491


namespace no_solution_abs_eq_quadratic_l124_124813

theorem no_solution_abs_eq_quadratic (x : ℝ) : ¬ (|x - 4| = x^2 + 6 * x + 8) :=
by
  sorry

end no_solution_abs_eq_quadratic_l124_124813


namespace find_x_that_satisfies_f_l124_124616

noncomputable def power_function (α : ℝ) (x : ℝ) : ℝ := x ^ α

theorem find_x_that_satisfies_f (α : ℝ) (x : ℝ) (h : power_function α (-2) = -1/8) : 
  power_function α x = 27 → x = 1/3 :=
  by
  sorry

end find_x_that_satisfies_f_l124_124616


namespace polynomial_sum_equals_one_l124_124075

theorem polynomial_sum_equals_one (a a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (2*x + 1)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4) →
  a - a_1 + a_2 - a_3 + a_4 = 1 :=
by
  sorry

end polynomial_sum_equals_one_l124_124075


namespace inequality_neg_mul_l124_124664

theorem inequality_neg_mul (a b : ℝ) (h : a > b) : -3 * a < -3 * b :=
sorry

end inequality_neg_mul_l124_124664


namespace gcd_of_168_56_224_l124_124708

theorem gcd_of_168_56_224 : (Nat.gcd 168 56 = 56) ∧ (Nat.gcd 56 224 = 56) ∧ (Nat.gcd 168 224 = 56) :=
by
  sorry

end gcd_of_168_56_224_l124_124708


namespace find_x_l124_124178

theorem find_x (x y : ℤ) (h₁ : x + 3 * y = 10) (h₂ : y = 3) : x = 1 := 
by
  sorry

end find_x_l124_124178


namespace velociraptor_catch_time_l124_124841

/-- You encounter a velociraptor while out for a stroll. You run to the northeast at 10 m/s 
    with a 3-second head start. The velociraptor runs at 15√2 m/s but only runs either north or east at any given time. 
    Prove that the time until the velociraptor catches you is 6 seconds. -/
theorem velociraptor_catch_time (v_yours : ℝ) (t_head_start : ℝ) (v_velociraptor : ℝ)
  (v_eff : ℝ) (speed_advantage : ℝ) (headstart_distance : ℝ) :
  v_yours = 10 → t_head_start = 3 → v_velociraptor = 15 * Real.sqrt 2 →
  v_eff = 15 → speed_advantage = v_eff - v_yours → headstart_distance = v_yours * t_head_start →
  (headstart_distance / speed_advantage) = 6 :=
by
  sorry

end velociraptor_catch_time_l124_124841


namespace socks_combinations_correct_l124_124436

noncomputable def num_socks_combinations (colors patterns pairs : ℕ) : ℕ :=
  colors * (colors - 1) * patterns * (patterns - 1)

theorem socks_combinations_correct :
  num_socks_combinations 5 4 20 = 240 :=
by
  sorry

end socks_combinations_correct_l124_124436


namespace irrational_sqrt3_l124_124502

theorem irrational_sqrt3 : ¬ ∃ (a b : ℕ), b ≠ 0 ∧ (a * a = 3 * b * b) :=
by
  sorry

end irrational_sqrt3_l124_124502


namespace max_min_sundays_in_month_l124_124560

def week_days : ℕ := 7
def min_month_days : ℕ := 28
def months_days (d : ℕ) : Prop := d = 28 ∨ d = 30 ∨ d = 31

theorem max_min_sundays_in_month (d : ℕ) (h1 : months_days d) :
  4 ≤ (d / week_days) + ite (d % week_days > 0) 1 0 ∧ (d / week_days) + ite (d % week_days > 0) 1 0 ≤ 5 :=
by
  sorry

end max_min_sundays_in_month_l124_124560


namespace factorization_quad_l124_124951

theorem factorization_quad (c d : ℕ) (h_factor : (x^2 - 18 * x + 77 = (x - c) * (x - d)))
  (h_nonneg : c ≥ 0 ∧ d ≥ 0) (h_lt : c > d) : 4 * d - c = 17 := by
  sorry

end factorization_quad_l124_124951


namespace fraction_meaningful_iff_l124_124129

theorem fraction_meaningful_iff (x : ℝ) : (x ≠ 2) ↔ (x - 2 ≠ 0) := 
by
  sorry

end fraction_meaningful_iff_l124_124129


namespace part1_part2_case1_part2_case2_part2_case3_l124_124528

namespace InequalityProof

variable {a x : ℝ}

def f (a x : ℝ) := a * x^2 + x - a

theorem part1 (h : a = 1) : (x > 1 ∨ x < -2) → f a x > 1 :=
by sorry

theorem part2_case1 (h1 : a < 0) (h2 : a < -1/2) : (- (a + 1) / a) < x ∧ x < 1 → f a x > 1 :=
by sorry

theorem part2_case2 (h1 : a < 0) (h2 : a = -1/2) : x ≠ 1 → f a x > 1 :=
by sorry

theorem part2_case3 (h1 : a < 0) (h2 : 0 > a) (h3 : a > -1/2) : 1 < x ∧ x < - (a + 1) / a → f a x > 1 :=
by sorry

end InequalityProof

end part1_part2_case1_part2_case2_part2_case3_l124_124528


namespace smallest_n_satisfying_conditions_l124_124875

def contains_digit (num : ℕ) (d : ℕ) : Prop :=
  d ∈ num.digits 10

theorem smallest_n_satisfying_conditions :
  ∃ n : ℕ, contains_digit (n^2) 7 ∧ contains_digit ((n+1)^2) 7 ∧ ¬ contains_digit ((n+2)^2) 7 ∧ ∀ m : ℕ, m < n → ¬ (contains_digit (m^2) 7 ∧ contains_digit ((m+1)^2) 7 ∧ ¬ contains_digit ((m+2)^2) 7) := 
sorry

end smallest_n_satisfying_conditions_l124_124875


namespace ticket_cost_l124_124150

theorem ticket_cost 
  (V G : ℕ)
  (h1 : V + G = 320)
  (h2 : V = G - 212) :
  40 * V + 15 * G = 6150 := 
by
  sorry

end ticket_cost_l124_124150


namespace minimum_value_of_quadratic_function_l124_124864

noncomputable def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem minimum_value_of_quadratic_function 
  (f : ℝ → ℝ)
  (n : ℕ)
  (h1 : f n = 6)
  (h2 : f (n + 1) = 5)
  (h3 : f (n + 2) = 5)
  (hf : ∃ a b c : ℝ, f = quadratic_function a b c) :
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 5 :=
by
  sorry

end minimum_value_of_quadratic_function_l124_124864


namespace problem_statement_l124_124383

noncomputable def expr (x y z : ℝ) : ℝ :=
  (x^2 * y^2) / ((x^2 - y*z) * (y^2 - x*z)) +
  (x^2 * z^2) / ((x^2 - y*z) * (z^2 - x*y)) +
  (y^2 * z^2) / ((y^2 - x*z) * (z^2 - x*y))

theorem problem_statement (x y z : ℝ) (h₁ : x ≠ 0) (h₂ : y ≠ 0) (h₃ : z ≠ 0) (h₄ : x + y + z = -1) :
  expr x y z = 1 := by
  sorry

end problem_statement_l124_124383


namespace prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l124_124297

-- Problem 1: Original proposition converse, inverse, contrapositive
theorem prob1_converse (x y : ℝ) (h : x = 0 ∨ y = 0) : x * y = 0 :=
sorry

theorem prob1_inverse (x y : ℝ) (h : x * y ≠ 0) : x ≠ 0 ∧ y ≠ 0 :=
sorry

theorem prob1_contrapositive (x y : ℝ) (h : x ≠ 0 ∧ y ≠ 0) : x * y ≠ 0 :=
sorry

-- Problem 2: Original proposition converse, inverse, contrapositive
theorem prob2_converse (x y : ℝ) (h : x * y > 0) : x > 0 ∧ y > 0 :=
sorry

theorem prob2_inverse (x y : ℝ) (h : x ≤ 0 ∨ y ≤ 0) : x * y ≤ 0 :=
sorry

theorem prob2_contrapositive (x y : ℝ) (h : x * y ≤ 0) : x ≤ 0 ∨ y ≤ 0 :=
sorry

end prob1_converse_prob1_inverse_prob1_contrapositive_prob2_converse_prob2_inverse_prob2_contrapositive_l124_124297


namespace solve_equation_l124_124619

theorem solve_equation (x : ℝ) (h : x ≠ 0 ∧ x ≠ -1) : (x / (x + 1) = 1 + (1 / x)) ↔ (x = -1 / 2) :=
by
  sorry

end solve_equation_l124_124619


namespace refund_amount_l124_124347

def income_tax_paid : ℝ := 156000
def education_expenses : ℝ := 130000
def medical_expenses : ℝ := 10000
def tax_rate : ℝ := 0.13

def eligible_expenses : ℝ := education_expenses + medical_expenses
def max_refund : ℝ := tax_rate * eligible_expenses

theorem refund_amount : min (max_refund) (income_tax_paid) = 18200 := by
  sorry

end refund_amount_l124_124347


namespace find_tenth_term_l124_124328

/- Define the general term formula -/
def a (a1 d : ℤ) (n : ℤ) : ℤ := a1 + (n - 1) * d

/- Define the sum of the first n terms formula -/
def S (a1 d : ℤ) (n : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

theorem find_tenth_term
  (a1 d : ℤ)
  (h1 : a a1 d 2 + a a1 d 5 = 19)
  (h2 : S a1 d 5 = 40) :
  a a1 d 10 = 29 := by
  /- Sorry used to skip the proof steps. -/
  sorry

end find_tenth_term_l124_124328


namespace four_digit_number_l124_124099

theorem four_digit_number (a b c d : ℕ)
    (h1 : 0 ≤ a) (h2 : a ≤ 9)
    (h3 : 0 ≤ b) (h4 : b ≤ 9)
    (h5 : 0 ≤ c) (h6 : c ≤ 9)
    (h7 : 0 ≤ d) (h8 : d ≤ 9)
    (h9 : 2 * (1000 * a + 100 * b + 10 * c + d) + 1000 = 1000 * d + 100 * c + 10 * b + a)
    : (1000 * a + 100 * b + 10 * c + d) = 2996 :=
by
  sorry

end four_digit_number_l124_124099


namespace water_remainder_l124_124873

theorem water_remainder (n : ℕ) (f : ℕ → ℚ) (h_init : f 1 = 1) 
  (h_recursive : ∀ k, k ≥ 2 → f k = f (k - 1) * (k^2 - 1) / k^2) :
  f 7 = 1 / 50 := 
sorry

end water_remainder_l124_124873


namespace initial_books_donations_l124_124913

variable {X : ℕ} -- Initial number of book donations

def books_donated_during_week := 10 * 5
def books_borrowed := 140
def books_remaining := 210

theorem initial_books_donations :
  X + books_donated_during_week - books_borrowed = books_remaining → X = 300 :=
by
  intro h
  sorry

end initial_books_donations_l124_124913


namespace solution_of_equations_l124_124587

variables (x y z w : ℤ)

def system_of_equations :=
  x + y + z + w = 20 ∧
  y + 2 * z - 3 * w = 28 ∧
  x - 2 * y + z = 36 ∧
  -7 * x - y + 5 * z + 3 * w = 84

theorem solution_of_equations (x y z w : ℤ) :
  system_of_equations x y z w → (x, y, z, w) = (4, -6, 20, 2) :=
by sorry

end solution_of_equations_l124_124587


namespace nonneg_solution_iff_m_range_l124_124160

theorem nonneg_solution_iff_m_range (m : ℝ) :
  (∃ x : ℝ, x ≥ 0 ∧ (m / (x - 1) + 3 / (1 - x) = 1)) ↔ (m ≥ 2 ∧ m ≠ 3) :=
sorry

end nonneg_solution_iff_m_range_l124_124160


namespace edward_mowed_lawns_l124_124061

theorem edward_mowed_lawns (L : ℕ) (h1 : 8 * L + 7 = 47) : L = 5 :=
by
  sorry

end edward_mowed_lawns_l124_124061


namespace systematic_sampling_student_number_l124_124693

theorem systematic_sampling_student_number 
  (total_students : ℕ)
  (sample_size : ℕ)
  (interval_between_numbers : ℕ)
  (student_17_in_sample : ∃ n, 17 = n ∧ n ≤ total_students ∧ n % interval_between_numbers = 5)
  : ∃ m, m = 41 ∧ m ≤ total_students ∧ m % interval_between_numbers = 5 := 
sorry

end systematic_sampling_student_number_l124_124693


namespace trig_identity_simplification_l124_124185

theorem trig_identity_simplification (α : ℝ) :
  (2 * Real.sin (Real.pi - α) + Real.sin (2 * α)) / (2 * (Real.cos (α / 2))^2) = 2 * Real.sin α :=
by sorry

end trig_identity_simplification_l124_124185


namespace clayton_total_points_l124_124460

theorem clayton_total_points 
  (game1 game2 game3 : ℕ)
  (game1_points : game1 = 10)
  (game2_points : game2 = 14)
  (game3_points : game3 = 6)
  (game4 : ℕ)
  (game4_points : game4 = (game1 + game2 + game3) / 3) :
  game1 + game2 + game3 + game4 = 40 :=
sorry

end clayton_total_points_l124_124460


namespace number_of_odd_positive_integer_triples_sum_25_l124_124110

theorem number_of_odd_positive_integer_triples_sum_25 :
  ∃ n : ℕ, (
    n = 78 ∧
    ∃ (a b c : ℕ), 
      (a % 2 = 1) ∧ (b % 2 = 1) ∧ (c % 2 = 1) ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 25
  ) := 
sorry

end number_of_odd_positive_integer_triples_sum_25_l124_124110


namespace negation_of_proposition_l124_124469

theorem negation_of_proposition :
  (¬ (∀ a b : ℤ, a = 0 → a * b = 0)) ↔ (∃ a b : ℤ, a = 0 ∧ a * b ≠ 0) :=
by
  sorry

end negation_of_proposition_l124_124469


namespace number_of_cipher_keys_l124_124803

theorem number_of_cipher_keys (n : ℕ) (h : n % 2 = 0) : 
  ∃ K : ℕ, K = 4^(n^2 / 4) :=
by 
  sorry

end number_of_cipher_keys_l124_124803


namespace union_sets_l124_124904

open Set

variable {α : Type*}

def setA : Set ℝ := { x | -2 < x ∧ x < 0 }
def setB : Set ℝ := { x | -1 < x ∧ x < 1 }
def setC : Set ℝ := { x | -2 < x ∧ x < 1 }

theorem union_sets : setA ∪ setB = setC := 
by {
  sorry
}

end union_sets_l124_124904


namespace f_f_2_eq_2_l124_124393

noncomputable def f (x : ℝ) : ℝ :=
if x < 2 then 2 * Real.exp (x - 1)
else Real.log (x ^ 2 - 1) / Real.log 3

theorem f_f_2_eq_2 : f (f 2) = 2 :=
by
  sorry

end f_f_2_eq_2_l124_124393


namespace smallest_solution_to_equation_l124_124838

noncomputable def smallest_solution := (11 - Real.sqrt 445) / 6

theorem smallest_solution_to_equation:
  ∃ x : ℝ, (3 * x / (x - 3) + (3 * x^2 - 27) / x = 14) ∧ (x = smallest_solution) :=
sorry

end smallest_solution_to_equation_l124_124838


namespace repeating_decimal_as_fraction_l124_124791

theorem repeating_decimal_as_fraction :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ Int.natAbs (Int.gcd a b) = 1 ∧ a + b = 15 ∧ (a : ℚ) / b = 0.3636363636363636 :=
by
  sorry

end repeating_decimal_as_fraction_l124_124791


namespace convinced_of_twelve_models_vitya_review_58_offers_l124_124350

noncomputable def ln : ℝ → ℝ := Real.log

theorem convinced_of_twelve_models (n : ℕ) (h_n : n ≥ 13) :
  ∃ k : ℕ, (12 / n : ℝ) ^ k < 0.01 := sorry

theorem vitya_review_58_offers :
  ∃ k : ℕ, (12 / 13 : ℝ) ^ k < 0.01 ∧ k = 58 := sorry

end convinced_of_twelve_models_vitya_review_58_offers_l124_124350


namespace incorrect_option_D_l124_124836

-- Definitions based on conditions
def cumulative_progress (days : ℕ) : ℕ :=
  30 * days

-- The Lean statement representing the mathematically equivalent proof problem
theorem incorrect_option_D : cumulative_progress 11 = 330 ∧ ¬ (cumulative_progress 10 = 330) :=
by {
  sorry
}

end incorrect_option_D_l124_124836


namespace agnes_flight_cost_l124_124642

theorem agnes_flight_cost
  (booking_fee : ℝ) (cost_per_km : ℝ) (distance_XY : ℝ)
  (h1 : booking_fee = 120)
  (h2 : cost_per_km = 0.12)
  (h3 : distance_XY = 4500) :
  booking_fee + cost_per_km * distance_XY = 660 := 
by
  sorry

end agnes_flight_cost_l124_124642


namespace amount_of_b_l124_124071

variable (A B : ℝ)

theorem amount_of_b (h₁ : A + B = 2530) (h₂ : (3 / 5) * A = (2 / 7) * B) : B = 1714 :=
sorry

end amount_of_b_l124_124071


namespace solve_for_x_l124_124851

theorem solve_for_x (x : ℚ) 
  (h : (1/3 : ℚ) + 1/x = (7/9 : ℚ) + 1) : 
  x = 9/13 :=
by
  sorry

end solve_for_x_l124_124851


namespace camera_guarantee_l124_124957

def battery_trials (b : Fin 22 → Bool) : Prop :=
  let charged := Finset.filter (λ i => b i) (Finset.univ : Finset (Fin 22))
  -- Ensuring there are exactly 15 charged batteries
  (charged.card = 15) ∧
  -- The camera works if any set of three batteries are charged
  (∀ (trials : Finset (Finset (Fin 22))),
   trials.card = 10 →
   ∃ t ∈ trials, (t.card = 3 ∧ t ⊆ charged))

theorem camera_guarantee :
  ∃ (b : Fin 22 → Bool), battery_trials b := by
  sorry

end camera_guarantee_l124_124957


namespace part_a_part_b_l124_124422

-- Define what it means for a coloring to be valid.
def valid_coloring (n : ℕ) (colors : Fin n → Fin 3) : Prop :=
  ∀ (i : Fin n),
  ∃ j k : Fin n, 
  ((i + 1) % n = j ∧ (i + 2) % n = k ∧ colors i ≠ colors j ∧ colors i ≠ colors k ∧ colors j ≠ colors k)

-- Part (a)
theorem part_a (n : ℕ) (hn : 3 ∣ n) : ∃ (colors : Fin n → Fin 3), valid_coloring n colors :=
by sorry

-- Part (b)
theorem part_b (n : ℕ) : (∃ (colors : Fin n → Fin 3), valid_coloring n colors) → 3 ∣ n :=
by sorry

end part_a_part_b_l124_124422


namespace expression_for_f_pos_f_monotone_on_pos_l124_124392

section

variable (f : ℝ → ℝ)
variable (h_odd : ∀ x, f (-x) = -f x)
variable (h_neg : ∀ x, -1 ≤ x ∧ x < 0 → f x = 2 * x + 1 / x^2)

-- Part 1: Prove the expression for f(x) when x ∈ (0,1]
theorem expression_for_f_pos (x : ℝ) (hx : 0 < x ∧ x ≤ 1) : 
  f x = 2 * x - 1 / x^2 :=
sorry

-- Part 2: Prove the monotonicity of f(x) on (0,1]
theorem f_monotone_on_pos : 
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y ≤ 1 → f x < f y :=
sorry

end

end expression_for_f_pos_f_monotone_on_pos_l124_124392


namespace max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l124_124886

noncomputable def f (x : ℝ) := Real.sin (Real.cos x) + Real.cos (Real.sin x)

theorem max_f_gt_sqrt2 : (∃ x : ℝ, f x > Real.sqrt 2) :=
sorry

theorem f_is_periodic : ∀ x : ℝ, f (x - 2 * Real.pi) = f x :=
sorry

theorem f_pi_shift_pos : ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi → f (x + Real.pi) > 0 :=
sorry

end max_f_gt_sqrt2_f_is_periodic_f_pi_shift_pos_l124_124886


namespace molecular_weight_CaO_l124_124215

theorem molecular_weight_CaO (m : ℕ -> ℝ) (h : m 7 = 392) : m 1 = 56 :=
sorry

end molecular_weight_CaO_l124_124215


namespace replace_movies_cost_l124_124128

theorem replace_movies_cost
  (num_movies : ℕ)
  (trade_in_value_per_vhs : ℕ)
  (cost_per_dvd : ℕ)
  (h1 : num_movies = 100)
  (h2 : trade_in_value_per_vhs = 2)
  (h3 : cost_per_dvd = 10):
  (cost_per_dvd - trade_in_value_per_vhs) * num_movies = 800 :=
by sorry

end replace_movies_cost_l124_124128


namespace age_of_teacher_l124_124305

theorem age_of_teacher (S T : ℕ) (avg_students avg_total : ℕ) (num_students num_total : ℕ)
  (h1 : num_students = 50)
  (h2 : avg_students = 14)
  (h3 : num_total = 51)
  (h4 : avg_total = 15)
  (h5 : S = avg_students * num_students)
  (h6 : S + T = avg_total * num_total) :
  T = 65 := 
by {
  sorry
}

end age_of_teacher_l124_124305


namespace abs_ineq_solution_l124_124447

theorem abs_ineq_solution (x : ℝ) : abs (x - 2) + abs (x - 3) < 9 ↔ -2 < x ∧ x < 7 :=
sorry

end abs_ineq_solution_l124_124447


namespace largest_even_digit_multiple_of_five_l124_124325

theorem largest_even_digit_multiple_of_five : ∃ n : ℕ, n = 8860 ∧ n < 10000 ∧ (∀ digit ∈ (n.digits 10), digit % 2 = 0) ∧ n % 5 = 0 :=
by
  sorry

end largest_even_digit_multiple_of_five_l124_124325


namespace derek_bought_more_cars_l124_124198

-- Define conditions
variables (d₆ c₆ d₁₆ c₁₆ : ℕ)

-- Given conditions
def initial_conditions :=
  (d₆ = 90) ∧
  (d₆ = 3 * c₆) ∧
  (d₁₆ = 120) ∧
  (c₁₆ = 2 * d₁₆)

-- Prove the number of cars Derek bought in ten years
theorem derek_bought_more_cars (h : initial_conditions d₆ c₆ d₁₆ c₁₆) : c₁₆ - c₆ = 210 :=
by sorry

end derek_bought_more_cars_l124_124198


namespace fraction_cube_l124_124213

theorem fraction_cube (a b : ℚ) (h : (a / b) ^ 3 = 15625 / 1000000) : a / b = 1 / 4 :=
by
  sorry

end fraction_cube_l124_124213


namespace sum_of_x_coordinates_l124_124091

def exists_common_point (x : ℕ) : Prop :=
  (3 * x + 5) % 9 = (7 * x + 3) % 9

theorem sum_of_x_coordinates :
  ∃ x : ℕ, exists_common_point x ∧ x % 9 = 5 := 
by
  sorry

end sum_of_x_coordinates_l124_124091


namespace fraction_conversion_integer_l124_124047

theorem fraction_conversion_integer (x : ℝ) :
  (x + 1) / 0.4 - (0.2 * x - 1) / 0.7 = 1 →
  (10 * x + 10) / 4 - (2 * x - 10) / 7 = 1 :=
by sorry

end fraction_conversion_integer_l124_124047


namespace discount_percentage_l124_124894

variable (P : ℝ) -- Original price of the dress
variable (D : ℝ) -- Discount percentage

theorem discount_percentage
  (h1 : P * (1 - D / 100) = 68)
  (h2 : 68 * 1.25 = 85)
  (h3 : 85 - P = 5) :
  D = 15 :=
by
  sorry

end discount_percentage_l124_124894


namespace max_brownies_l124_124802

theorem max_brownies (m n : ℕ) (h1 : (m-2)*(n-2) = 2*(2*m + 2*n - 4)) : m * n ≤ 294 :=
by sorry

end max_brownies_l124_124802


namespace simplify_expression_l124_124370

variable (x : ℝ)

theorem simplify_expression : (2 * x + 20) + (150 * x + 20) = 152 * x + 40 :=
by
  sorry

end simplify_expression_l124_124370


namespace vector_combination_l124_124936

-- Definitions for vectors a, b, and c with the conditions provided
def a : ℝ × ℝ × ℝ := (-1, 3, 2)
def b : ℝ × ℝ × ℝ := (4, -6, 2)
def c (t : ℝ) : ℝ × ℝ × ℝ := (-3, 12, t)

-- The statement we want to prove
theorem vector_combination (t m n : ℝ)
  (h : c t = m • a + n • b) :
  t = 11 ∧ m + n = 11 / 2 :=
by
  sorry

end vector_combination_l124_124936


namespace right_triangle_perimeter_l124_124703

theorem right_triangle_perimeter (a b : ℕ) (h : a^2 + b^2 = 100) (r : ℕ := 1) :
  (a + b + 10) = 24 :=
sorry

end right_triangle_perimeter_l124_124703


namespace daily_expenditure_l124_124452

theorem daily_expenditure (total_spent : ℕ) (days_in_june : ℕ) (equal_consumption : Prop) :
  total_spent = 372 ∧ days_in_june = 30 ∧ equal_consumption → (372 / 30) = 12.40 := by
  sorry

end daily_expenditure_l124_124452


namespace math_problem_l124_124923

theorem math_problem (x : ℕ) (h : (2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x + 2^x = 512)) : (x + 2) * (x - 2) = 32 :=
sorry

end math_problem_l124_124923


namespace min_ab_eq_11_l124_124003

theorem min_ab_eq_11 (a b : ℕ) (h : 23 * a - 13 * b = 1) : a + b = 11 :=
sorry

end min_ab_eq_11_l124_124003


namespace fraction_addition_l124_124707

theorem fraction_addition : (2 / 5 + 3 / 8) = 31 / 40 :=
by
  sorry

end fraction_addition_l124_124707


namespace graveling_cost_correct_l124_124014

-- Define the dimensions of the rectangular lawn
def lawn_length : ℕ := 80 -- in meters
def lawn_breadth : ℕ := 50 -- in meters

-- Define the width of each road
def road_width : ℕ := 10 -- in meters

-- Define the cost per square meter for graveling the roads
def cost_per_sq_m : ℕ := 3 -- in Rs. per sq meter

-- Define the area of the road parallel to the length of the lawn
def area_road_parallel_length : ℕ := lawn_length * road_width

-- Define the effective length of the road parallel to the breadth of the lawn
def effective_road_parallel_breadth_length : ℕ := lawn_breadth - road_width

-- Define the area of the road parallel to the breadth of the lawn
def area_road_parallel_breadth : ℕ := effective_road_parallel_breadth_length * road_width

-- Define the total area to be graveled
def total_area_to_be_graveled : ℕ := area_road_parallel_length + area_road_parallel_breadth

-- Define the total cost of graveling
def total_graveling_cost : ℕ := total_area_to_be_graveled * cost_per_sq_m

-- Theorem: The total cost of graveling the two roads is Rs. 3600
theorem graveling_cost_correct : total_graveling_cost = 3600 := 
by
  unfold total_graveling_cost total_area_to_be_graveled area_road_parallel_length area_road_parallel_breadth effective_road_parallel_breadth_length lawn_length lawn_breadth road_width cost_per_sq_m
  exact rfl

end graveling_cost_correct_l124_124014


namespace geometric_sequence_tenth_fifth_terms_l124_124192

variable (a r : ℚ) (n : ℕ)

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem geometric_sequence_tenth_fifth_terms :
  (geometric_sequence 4 (4/3) 10 = 1048576 / 19683) ∧ (geometric_sequence 4 (4/3) 5 = 1024 / 81) :=
by
  sorry

end geometric_sequence_tenth_fifth_terms_l124_124192


namespace integral_3x_plus_sin_x_l124_124454

theorem integral_3x_plus_sin_x :
  ∫ x in (0 : ℝ)..(π / 2), (3 * x + Real.sin x) = (3 / 8) * π^2 + 1 :=
by
  sorry

end integral_3x_plus_sin_x_l124_124454


namespace fraction_meaningful_l124_124364

theorem fraction_meaningful (x : ℝ) : (¬ (x - 2 = 0)) ↔ (x ≠ 2) :=
by
  sorry

end fraction_meaningful_l124_124364


namespace exists_xy_binom_eq_l124_124375

theorem exists_xy_binom_eq (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ (x + y).choose 2 = a * x + b * y :=
by
  sorry

end exists_xy_binom_eq_l124_124375


namespace norm_of_5v_l124_124942

noncomputable def norm_scale (v : ℝ × ℝ) (c : ℝ) : ℝ := c * (Real.sqrt (v.1^2 + v.2^2))

theorem norm_of_5v (v : ℝ × ℝ) (h : Real.sqrt (v.1^2 + v.2^2) = 6) : norm_scale v 5 = 30 := by
  sorry

end norm_of_5v_l124_124942


namespace plane_angle_divides_cube_l124_124083

noncomputable def angle_between_planes (m n : ℕ) (h : m ≤ n) : ℝ :=
  Real.arctan (2 * m / (m + n))

theorem plane_angle_divides_cube (m n : ℕ) (h : m ≤ n) :
  ∃ α, α = angle_between_planes m n h :=
sorry

end plane_angle_divides_cube_l124_124083


namespace quadratic_distinct_real_roots_l124_124249

-- Defining the main hypothesis
theorem quadratic_distinct_real_roots (k : ℝ) :
  (k < 4 / 3) ∧ (k ≠ 1) ↔ (∀ x : ℂ, ((k-1) * x^2 - 2 * x + 3 = 0) → ∃ x₁ x₂ : ℂ, x₁ ≠ x₂ ∧ ((k-1) * x₁ ^ 2 - 2 * x₁ + 3 = 0) ∧ ((k-1) * x₂ ^ 2 - 2 * x₂ + 3 = 0)) := by
sorry

end quadratic_distinct_real_roots_l124_124249


namespace length_of_platform_l124_124156

theorem length_of_platform (length_of_train : ℕ) (speed_kmph : ℕ) (time_s : ℕ) (L : ℕ) :
  length_of_train = 160 → speed_kmph = 72 → time_s = 25 → (L = 340) :=
by
  sorry

end length_of_platform_l124_124156


namespace lateral_surface_area_of_cone_l124_124580

open Real

theorem lateral_surface_area_of_cone
  (SA : ℝ) (SB : ℝ)
  (cos_angle_SA_SB : ℝ) (angle_SA_base : ℝ)
  (area_SAB : ℝ) :
  cos_angle_SA_SB = 7 / 8 →
  angle_SA_base = π / 4 →
  area_SAB = 5 * sqrt 15 →
  SA = 4 * sqrt 5 →
  SB = SA →
  (1/2) * (sqrt 2 / 2 * SA) * (2 * π * SA) = 40 * sqrt 2 * π :=
sorry

end lateral_surface_area_of_cone_l124_124580


namespace car_owners_without_motorcycle_or_bicycle_l124_124025

noncomputable def total_adults := 500
noncomputable def car_owners := 400
noncomputable def motorcycle_owners := 200
noncomputable def bicycle_owners := 150
noncomputable def car_motorcycle_owners := 100
noncomputable def motorcycle_bicycle_owners := 50
noncomputable def car_bicycle_owners := 30

theorem car_owners_without_motorcycle_or_bicycle :
  car_owners - car_motorcycle_owners - car_bicycle_owners = 270 := by
  sorry

end car_owners_without_motorcycle_or_bicycle_l124_124025


namespace solve_for_x_l124_124391

theorem solve_for_x (x : ℤ) (h : x + 1 = 10) : x = 9 := 
by 
  sorry

end solve_for_x_l124_124391


namespace exists_mutual_shooters_l124_124189

theorem exists_mutual_shooters (n : ℕ) (h : 0 ≤ n) (d : Fin (2 * n + 1) → Fin (2 * n + 1) → ℝ)
  (hdistinct : ∀ i j k l : Fin (2 * n + 1), i ≠ j → k ≠ l → d i j ≠ d k l)
  (hc : ∀ i : Fin (2 * n + 1), ∃ j : Fin (2 * n + 1), i ≠ j ∧ (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k)) :
  ∃ i j : Fin (2 * n + 1), i ≠ j ∧
  (∀ k : Fin (2 * n + 1), k ≠ j → d i j < d i k) ∧
  (∀ k : Fin (2 * n + 1), k ≠ i → d j i < d j k) :=
by
  sorry

end exists_mutual_shooters_l124_124189


namespace number_of_valid_partitions_l124_124292

-- Define the condition to check if a list of integers has all elements same or exactly differ by 1
def validPartition (l : List ℕ) : Prop :=
  l ≠ [] ∧ (∀ (a b : ℕ), a ∈ l → b ∈ l → a = b ∨ a = b + 1 ∨ b = a + 1)

-- Count valid partitions of n (integer partitions meeting the given condition)
noncomputable def countValidPartitions (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Main theorem
theorem number_of_valid_partitions (n : ℕ) : countValidPartitions n = n :=
by
  sorry

end number_of_valid_partitions_l124_124292


namespace graph_not_pass_second_quadrant_l124_124275

theorem graph_not_pass_second_quadrant (a b : ℝ) (h1 : a > 1) (h2 : b < -1) :
  ¬ ∃ (x : ℝ), y = a^x + b ∧ x < 0 ∧ y > 0 :=
by
  sorry

end graph_not_pass_second_quadrant_l124_124275


namespace smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l124_124471

noncomputable def f (x : ℝ) : ℝ := (1/2) * Real.sin x - (Real.sqrt 3 / 2) * Real.cos x

theorem smallest_positive_period_2pi : ∀ x : ℝ, f (x + 2 * Real.pi) = f x := by
  sorry

theorem range_of_f : ∀ y : ℝ, y ∈ Set.range f ↔ -1 ≤ y ∧ y ≤ 1 := by
  sorry

theorem intervals_monotonically_increasing : 
  ∀ k : ℤ, 
  ∀ x : ℝ, 
  (2 * k * Real.pi - Real.pi / 6 ≤ x ∧ x ≤ 2 * k * Real.pi + 5 * Real.pi / 6) → 
  (f (x + Real.pi / 6) - f x) ≥ 0 := by
  sorry

end smallest_positive_period_2pi_range_of_f_intervals_monotonically_increasing_l124_124471


namespace cube_surface_area_l124_124307

/-- Given a cube with a space diagonal of 6, the surface area is 72. -/
theorem cube_surface_area (s : ℝ) (h : s * Real.sqrt 3 = 6) : 6 * s^2 = 72 :=
by
  sorry

end cube_surface_area_l124_124307


namespace sine_tangent_not_possible_1_sine_tangent_not_possible_2_l124_124631

theorem sine_tangent_not_possible_1 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.27413 ∧ Real.tan θ = 0.25719) :=
sorry

theorem sine_tangent_not_possible_2 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.25719 ∧ Real.tan θ = 0.27413) :=
sorry

end sine_tangent_not_possible_1_sine_tangent_not_possible_2_l124_124631


namespace initial_books_l124_124930

theorem initial_books (total_books_now : ℕ) (books_added : ℕ) (initial_books : ℕ) :
  total_books_now = 48 → books_added = 10 → initial_books = total_books_now - books_added → initial_books = 38 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end initial_books_l124_124930


namespace describe_graph_l124_124822

theorem describe_graph :
  ∀ (x y : ℝ), ((x + y) ^ 2 = x ^ 2 + y ^ 2 + 4 * x) ↔ (x = 0 ∨ y = 2) := 
by
  sorry

end describe_graph_l124_124822


namespace apple_counts_l124_124711

theorem apple_counts (x y : ℤ) (h1 : y - x = 2) (h2 : y = 3 * x - 4) : x = 3 ∧ y = 5 := 
by
  sorry

end apple_counts_l124_124711


namespace max_xy_value_l124_124005

theorem max_xy_value (x y : ℕ) (h : 27 * x + 35 * y ≤ 1000) : x * y ≤ 252 :=
sorry

end max_xy_value_l124_124005


namespace solve_inequality_1_find_range_of_a_l124_124539

def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem solve_inequality_1 :
  {x : ℝ | f x ≥ 5} = {x : ℝ | x ≤ -3} ∪ {x : ℝ | x ≥ 2} :=
by
  sorry
  
theorem find_range_of_a (a : ℝ) :
  (∀ x : ℝ, f x > a^2 - 2 * a - 5) ↔ -2 < a ∧ a < 4 :=
by
  sorry

end solve_inequality_1_find_range_of_a_l124_124539


namespace minimum_questions_to_determine_village_l124_124667

-- Step 1: Define the types of villages
inductive Village
| A : Village
| B : Village
| C : Village

-- Step 2: Define the properties of residents in each village
def tells_truth (v : Village) (p : Prop) : Prop :=
  match v with
  | Village.A => p
  | Village.B => ¬p
  | Village.C => p ∨ ¬p

-- Step 3: Define the problem context in Lean
theorem minimum_questions_to_determine_village :
    ∀ (tourist_village person_village : Village), ∃ (n : ℕ), n = 4 := by
  sorry

end minimum_questions_to_determine_village_l124_124667


namespace ranking_of_ABC_l124_124814

-- Define the ranking type
inductive Rank
| first
| second
| third

-- Define types for people
inductive Person
| A
| B
| C

open Rank Person

-- Alias for ranking of each person
def ranking := Person → Rank

-- Define the conditions
def A_statement (r : ranking) : Prop := r A ≠ first
def B_statement (r : ranking) : Prop := A_statement r ≠ false
def C_statement (r : ranking) : Prop := r C ≠ third

def B_lied : Prop := true
def C_told_truth : Prop := true

-- The equivalent problem, asked to prove the final result
theorem ranking_of_ABC (r : ranking) : 
  (B_lied ∧ C_told_truth ∧ B_statement r = false ∧ C_statement r = true) → 
  (r A = first ∧ r B = third ∧ r C = second) :=
sorry

end ranking_of_ABC_l124_124814


namespace value_of_other_number_l124_124988

theorem value_of_other_number (k : ℕ) (other_number : ℕ) (h1 : k = 2) (h2 : (5 + k) * (5 - k) = 5^2 - other_number) : other_number = 21 :=
  sorry

end value_of_other_number_l124_124988


namespace compute_expression_l124_124550

theorem compute_expression : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end compute_expression_l124_124550


namespace find_a_of_pure_imaginary_l124_124976

noncomputable def isPureImaginary (z : ℂ) : Prop :=
  ∃ b : ℝ, z = ⟨0, b⟩  -- complex number z is purely imaginary if it can be written as 0 + bi

theorem find_a_of_pure_imaginary (a : ℝ) (i : ℂ) (ha : i*i = -1) :
  isPureImaginary ((1 - i) * (a + i)) → a = -1 := by
  sorry

end find_a_of_pure_imaginary_l124_124976


namespace problem_a_b_n_geq_1_l124_124633

theorem problem_a_b_n_geq_1 (a b n : ℕ) (h1 : a > b) (h2 : b > 1) (h3 : Odd b) (h4 : n > 0)
  (h5 : b^n ∣ a^n - 1) : a^b > 3^n / n := 
by 
  sorry

end problem_a_b_n_geq_1_l124_124633


namespace problem1_problem2_l124_124127

-- Proof Problem 1: Prove that (x-y)^2 - (x+y)(x-y) = -2xy + 2y^2
theorem problem1 (x y : ℝ) : (x - y) ^ 2 - (x + y) * (x - y) = -2 * x * y + 2 * y ^ 2 := 
by
  sorry

-- Proof Problem 2: Prove that (12a^2b - 6ab^2) / (-3ab) = -4a + 2b
theorem problem2 (a b : ℝ) (h : -3 * a * b ≠ 0) : (12 * a^2 * b - 6 * a * b^2) / (-3 * a * b) = -4 * a + 2 * b := 
by
  sorry

end problem1_problem2_l124_124127


namespace sufficient_but_not_necessary_condition_l124_124361

theorem sufficient_but_not_necessary_condition (A B : Set ℝ) :
  (A = {x : ℝ | 1 < x ∧ x < 3}) →
  (B = {x : ℝ | x > -1}) →
  (∀ x, x ∈ A → x ∈ B) ∧ (∃ x, x ∈ B ∧ x ∉ A) :=
by
  sorry

end sufficient_but_not_necessary_condition_l124_124361


namespace max_sum_of_squares_70_l124_124662

theorem max_sum_of_squares_70 :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧
  a^2 + b^2 + c^2 + d^2 = 70 ∧ a + b + c + d = 16 :=
by
  sorry

end max_sum_of_squares_70_l124_124662


namespace all_lights_on_l124_124311

def light_on (n : ℕ) : Prop := sorry

axiom light_rule_1 (k : ℕ) (hk: light_on k): light_on (2 * k) ∧ light_on (2 * k + 1)
axiom light_rule_2 (k : ℕ) (hk: ¬ light_on k): ¬ light_on (4 * k + 1) ∧ ¬ light_on (4 * k + 3)
axiom light_2023_on : light_on 2023

theorem all_lights_on (n : ℕ) (hn : n < 2023) : light_on n :=
by sorry

end all_lights_on_l124_124311


namespace toms_friend_decks_l124_124229

theorem toms_friend_decks
  (cost_per_deck : ℕ)
  (tom_decks : ℕ)
  (total_spent : ℕ)
  (h1 : cost_per_deck = 8)
  (h2 : tom_decks = 3)
  (h3 : total_spent = 64) :
  (total_spent - tom_decks * cost_per_deck) / cost_per_deck = 5 := by
  sorry

end toms_friend_decks_l124_124229


namespace percentage_increase_in_expenses_l124_124669

variable (a b c : ℝ)

theorem percentage_increase_in_expenses :
  (10 / 100 * a + 30 / 100 * b + 20 / 100 * c) / (a + b + c) =
  (10 * a + 30 * b + 20 * c) / (100 * (a + b + c)) :=
by
  sorry

end percentage_increase_in_expenses_l124_124669


namespace nails_remaining_l124_124760

theorem nails_remaining (nails_initial : ℕ) (kitchen_fraction : ℚ) (fence_fraction : ℚ) (nails_used_kitchen : ℕ) (nails_remaining_after_kitchen : ℕ) (nails_used_fence : ℕ) (nails_remaining_final : ℕ) 
  (h1 : nails_initial = 400) 
  (h2 : kitchen_fraction = 0.30) 
  (h3 : nails_used_kitchen = kitchen_fraction * nails_initial) 
  (h4 : nails_remaining_after_kitchen = nails_initial - nails_used_kitchen) 
  (h5 : fence_fraction = 0.70) 
  (h6 : nails_used_fence = fence_fraction * nails_remaining_after_kitchen) 
  (h7 : nails_remaining_final = nails_remaining_after_kitchen - nails_used_fence) :
  nails_remaining_final = 84 := by
sorry

end nails_remaining_l124_124760


namespace intersection_complement_l124_124108

def A : Set ℝ := {1, 2, 3, 4, 5, 6}
def B : Set ℝ := {x | 2 < x ∧ x < 5 }
def C : Set ℝ := {x | x ≤ 2 ∨ x ≥ 5 }

theorem intersection_complement :
  (A ∩ C) = {1, 2, 5, 6} :=
by sorry

end intersection_complement_l124_124108


namespace marbles_per_friend_l124_124426

variable (initial_marbles remaining_marbles given_marbles_per_friend : ℕ)

-- conditions in a)
def condition_initial_marbles := initial_marbles = 500
def condition_remaining_marbles := 4 * remaining_marbles = 720
def condition_total_given_marbles := initial_marbles - remaining_marbles = 320
def condition_given_marbles_per_friend := given_marbles_per_friend * 4 = 320

-- question proof goal
theorem marbles_per_friend (initial_marbles: ℕ) (remaining_marbles: ℕ) (given_marbles_per_friend: ℕ) :
  (condition_initial_marbles initial_marbles) →
  (condition_remaining_marbles remaining_marbles) →
  (condition_total_given_marbles initial_marbles remaining_marbles) →
  (condition_given_marbles_per_friend given_marbles_per_friend) →
  given_marbles_per_friend = 80 :=
by
  intros hinitial hremaining htotal_given hgiven_per_friend
  sorry

end marbles_per_friend_l124_124426


namespace probability_of_selecting_storybook_l124_124833

theorem probability_of_selecting_storybook (reference_books storybooks picture_books : ℕ) 
  (h1 : reference_books = 5) (h2 : storybooks = 3) (h3 : picture_books = 2) :
  (storybooks : ℚ) / (reference_books + storybooks + picture_books) = 3 / 10 :=
by {
  sorry
}

end probability_of_selecting_storybook_l124_124833


namespace complex_z24_condition_l124_124136

open Complex

theorem complex_z24_condition (z : ℂ) (h : z + z⁻¹ = 2 * Real.cos (5 * π / 180)) : 
  z^24 + z⁻¹^24 = -1 := sorry

end complex_z24_condition_l124_124136


namespace num_students_in_second_class_l124_124583

theorem num_students_in_second_class 
  (avg1 : ℕ) (num1 : ℕ) (avg2 : ℕ) (overall_avg : ℕ) (n : ℕ) :
  avg1 = 50 → num1 = 30 → avg2 = 60 → overall_avg = 5625 → 
  (num1 * avg1 + n * avg2) = (num1 + n) * overall_avg → n = 50 :=
by sorry

end num_students_in_second_class_l124_124583


namespace simplify_fraction_l124_124401

theorem simplify_fraction (x : ℤ) : 
    (2 * x + 3) / 4 + (5 - 4 * x) / 3 = (-10 * x + 29) / 12 := 
by
  sorry

end simplify_fraction_l124_124401


namespace frank_columns_l124_124848

theorem frank_columns (people : ℕ) (brownies_per_person : ℕ) (rows : ℕ)
  (h1 : people = 6) (h2 : brownies_per_person = 3) (h3 : rows = 3) : 
  (people * brownies_per_person) / rows = 6 :=
by
  -- Proof goes here
  sorry

end frank_columns_l124_124848


namespace solve_system_l124_124919

theorem solve_system :
  ∃ (x y : ℕ), 
    (∃ d : ℕ, d ∣ 42 ∧ x^2 + y^2 = 468 ∧ d + (x * y) / d = 42) ∧ 
    (x = 12 ∧ y = 18) ∨ (x = 18 ∧ y = 12) :=
sorry

end solve_system_l124_124919


namespace distance_to_origin_l124_124658

theorem distance_to_origin (a : ℝ) (h: |a| = 5) : 3 - a = -2 ∨ 3 - a = 8 :=
sorry

end distance_to_origin_l124_124658


namespace cost_per_bar_l124_124629

variable (months_in_year : ℕ := 12)
variable (months_per_bar_of_soap : ℕ := 2)
variable (total_cost_for_year : ℕ := 48)

theorem cost_per_bar (h1 : months_per_bar_of_soap > 0)
                     (h2 : total_cost_for_year > 0) : 
    (total_cost_for_year / (months_in_year / months_per_bar_of_soap)) = 8 := 
by
  sorry

end cost_per_bar_l124_124629


namespace percentage_less_A_than_B_l124_124480

theorem percentage_less_A_than_B :
  ∀ (full_marks A_marks D_marks C_marks B_marks : ℝ),
    full_marks = 500 →
    A_marks = 360 →
    D_marks = 0.80 * full_marks →
    C_marks = (1 - 0.20) * D_marks →
    B_marks = (1 + 0.25) * C_marks →
    ((B_marks - A_marks) / B_marks) * 100 = 10 :=
  by intros full_marks A_marks D_marks C_marks B_marks
     intros h_full h_A h_D h_C h_B
     sorry

end percentage_less_A_than_B_l124_124480


namespace ratio_fourth_to_sixth_l124_124219

-- Definitions from the conditions
def fourth_level_students := 40
def sixth_level_students := 40
def seventh_level_students := 2 * fourth_level_students

-- Statement to prove
theorem ratio_fourth_to_sixth : 
  fourth_level_students / sixth_level_students = 1 :=
by
  -- Proof skipped
  sorry

end ratio_fourth_to_sixth_l124_124219


namespace min_AB_CD_value_l124_124644

def vector := (ℝ × ℝ)

def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def AB_CD (AC BD CB : vector) : ℝ :=
  let AB := (CB.1 + AC.1, CB.2 + AC.2)
  let CD := (CB.1 + BD.1, CB.2 + BD.2)
  dot_product AB CD

theorem min_AB_CD_value : ∀ (AC BD : vector), AC = (1, 2) → BD = (-2, 2) → 
  ∃ CB : vector, AB_CD AC BD CB = -9 / 4 :=
by
  intros AC BD hAC hBD
  sorry

end min_AB_CD_value_l124_124644


namespace part1_part2_l124_124335

open Set

variable (A B : Set ℝ) (m : ℝ)

def setA : Set ℝ := {x | x ^ 2 - 2 * x - 8 ≤ 0}

def setB (m : ℝ) : Set ℝ := {x | x ^ 2 - (2 * m - 3) * x + m ^ 2 - 3 * m ≤ 0}

theorem part1 (h : (setA ∩ setB 5) = Icc 2 4) : m = 5 := sorry

theorem part2 (h : setA ⊆ compl (setB m)) :
  m ∈ Iio (-2) ∪ Ioi 7 := sorry

end part1_part2_l124_124335


namespace spider_legs_total_l124_124790

def num_spiders : ℕ := 4
def legs_per_spider : ℕ := 8
def total_legs : ℕ := num_spiders * legs_per_spider

theorem spider_legs_total : total_legs = 32 := by
  sorry -- proof is skipped with 'sorry'

end spider_legs_total_l124_124790


namespace part_a_part_b_l124_124409

def balanced (V : Finset (ℝ × ℝ)) : Prop :=
  ∀ (A B : ℝ × ℝ), A ∈ V → B ∈ V → A ≠ B → ∃ C : ℝ × ℝ, C ∈ V ∧ (dist C A = dist C B)

def center_free (V : Finset (ℝ × ℝ)) : Prop :=
  ¬ ∃ (A B C P : ℝ × ℝ), A ∈ V → B ∈ V → C ∈ V → P ∈ V →
                         A ≠ B ∧ B ≠ C ∧ A ≠ C →
                         (dist P A = dist P B ∧ dist P B = dist P C)

theorem part_a (n : ℕ) (hn : 3 ≤ n) :
  ∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V :=
by sorry

theorem part_b : ∀ n : ℕ, 3 ≤ n →
  (∃ V : Finset (ℝ × ℝ), V.card = n ∧ balanced V ∧ center_free V ↔ n % 2 = 1) :=
by sorry

end part_a_part_b_l124_124409


namespace probability_point_between_X_and_Z_l124_124299

theorem probability_point_between_X_and_Z (XW XZ YW : ℝ) (h1 : XW = 4 * XZ) (h2 : XW = 8 * YW) :
  (XZ / XW) = 1 / 4 := by
  sorry

end probability_point_between_X_and_Z_l124_124299


namespace hexagon_area_l124_124281

theorem hexagon_area :
  let points := [(0, 0), (2, 4), (5, 4), (7, 0), (5, -4), (2, -4), (0, 0)]
  ∃ (area : ℝ), area = 52 := by
  sorry

end hexagon_area_l124_124281


namespace cos4_minus_sin4_15_eq_sqrt3_div2_l124_124482

theorem cos4_minus_sin4_15_eq_sqrt3_div2 :
  (Real.cos 15)^4 - (Real.sin 15)^4 = Real.sqrt 3 / 2 :=
sorry

end cos4_minus_sin4_15_eq_sqrt3_div2_l124_124482


namespace difference_SP_l124_124650

-- Definitions for amounts
variables (P Q R S : ℕ)

-- Conditions given in the problem
def total_amount := P + Q + R + S = 1000
def P_condition := P = 2 * Q
def S_condition := S = 4 * R
def Q_R_equal := Q = R

-- Statement of the problem that needs to be proven
theorem difference_SP (P Q R S : ℕ) (h1 : total_amount P Q R S) 
  (h2 : P_condition P Q) (h3 : S_condition S R) (h4 : Q_R_equal Q R) : 
  S - P = 250 :=
by 
  sorry

end difference_SP_l124_124650


namespace top_card_is_5_or_king_l124_124263

-- Define the number of cards in a deck
def total_cards : ℕ := 52

-- Define the number of 5s in a deck
def number_of_5s : ℕ := 4

-- Define the number of Kings in a deck
def number_of_kings : ℕ := 4

-- Define the number of favorable outcomes (cards that are either 5 or King)
def favorable_outcomes : ℕ := number_of_5s + number_of_kings

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_cards

-- Theorem: The probability that the top card is either a 5 or a King is 2/13
theorem top_card_is_5_or_king (h_total_cards : total_cards = 52)
    (h_number_of_5s : number_of_5s = 4)
    (h_number_of_kings : number_of_kings = 4) :
    probability = 2 / 13 := by
  -- Proof would go here
  sorry

end top_card_is_5_or_king_l124_124263


namespace largest_sphere_radius_l124_124510

noncomputable def torus_inner_radius := 3
noncomputable def torus_outer_radius := 5
noncomputable def torus_center_circle := (4, 0, 1)
noncomputable def torus_radius := 1
noncomputable def torus_table_plane := 0

theorem largest_sphere_radius :
  ∀ (r : ℝ), 
  ∀ (O P : ℝ × ℝ × ℝ), 
  (P = (4, 0, 1)) → 
  (O = (0, 0, r)) → 
  4^2 + (r - 1)^2 = (r + 1)^2 → 
  r = 4 := 
by
  intros
  sorry

end largest_sphere_radius_l124_124510


namespace part1_proof_part2_proof_part3_proof_part4_proof_l124_124538

variable {A B C : Type}
variables {a b c : ℝ}  -- Sides of the triangle
variables {h_a h_b h_c r r_a r_b r_c : ℝ}  -- Altitudes, inradius, and exradii of \triangle ABC

-- Part 1: Proving the sum of altitudes related to sides and inradius
theorem part1_proof : h_a + h_b + h_c = r * (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 2: Proving the sum of reciprocals of altitudes related to the reciprocal of inradius and exradii
theorem part2_proof : (1 / h_a) + (1 / h_b) + (1 / h_c) = 1 / r ∧ 1 / r = (1 / r_a) + (1 / r_b) + (1 / r_c) := sorry

-- Part 3: Combining results of parts 1 and 2 to prove product of sums
theorem part3_proof : (h_a + h_b + h_c) * ((1 / h_a) + (1 / h_b) + (1 / h_c)) = (a + b + c) * (1 / a + 1 / b + 1 / c) := sorry

-- Part 4: Final geometric identity
theorem part4_proof : (h_a + h_c) / r_a + (h_c + h_a) / r_b + (h_a + h_b) / r_c = 6 := sorry

end part1_proof_part2_proof_part3_proof_part4_proof_l124_124538


namespace number_of_one_dollar_coins_l124_124535

theorem number_of_one_dollar_coins (t : ℕ) :
  (∃ k : ℕ, 3 * k = t) → ∃ k : ℕ, k = t / 3 :=
by
  sorry

end number_of_one_dollar_coins_l124_124535


namespace count_board_configurations_l124_124632

-- Define the 3x3 board as a type with 9 positions
inductive Position 
| top_left | top_center | top_right
| middle_left | center | middle_right
| bottom_left | bottom_center | bottom_right

-- Define an enum for players' moves
inductive Mark
| X | O | Empty

-- Define a board as a mapping from positions to marks
def Board : Type := Position → Mark

-- Define the win condition for Carl
def win_condition (b : Board) : Prop := 
(b Position.center = Mark.O) ∧ 
((b Position.top_left = Mark.O ∧ b Position.top_center = Mark.O) ∨ 
(b Position.middle_left = Mark.O ∧ b Position.middle_right = Mark.O) ∨ 
(b Position.bottom_left = Mark.O ∧ b Position.bottom_center = Mark.O))

-- Define the condition for a filled board
def filled_board (b : Board) : Prop :=
∀ p : Position, b p ≠ Mark.Empty

-- The proof problem to show the total number of configurations is 30
theorem count_board_configurations : 
  ∃ (n : ℕ), n = 30 ∧
  (∃ b : Board, win_condition b ∧ filled_board b) := 
sorry

end count_board_configurations_l124_124632


namespace simplify_expression_l124_124821

theorem simplify_expression
  (a b c : ℝ) 
  (hnz_a : a ≠ 0) 
  (hnz_b : b ≠ 0) 
  (hnz_c : c ≠ 0) 
  (h_sum : a + b + c = 0) :
  (1 / (b^3 + c^3 - a^3)) + (1 / (a^3 + c^3 - b^3)) + (1 / (a^3 + b^3 - c^3)) = 1 / (a * b * c) :=
by
  sorry

end simplify_expression_l124_124821


namespace unique_B_squared_l124_124040

theorem unique_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) : 
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B2 = B * B :=
sorry

end unique_B_squared_l124_124040


namespace impossible_sequence_l124_124978

theorem impossible_sequence (α : ℝ) (hα : 0 < α ∧ α < 1) (a : ℕ → ℝ) (ha : ∀ n, 0 < a n) :
  (∀ n, 1 + a (n + 1) ≤ a n + (α / n) * a n) → false :=
by
  sorry

end impossible_sequence_l124_124978


namespace one_minus_repeating_eight_l124_124979

-- Given the condition
def b : ℚ := 8 / 9

-- The proof problem statement
theorem one_minus_repeating_eight : 1 - b = 1 / 9 := 
by
  sorry  -- proof to be provided

end one_minus_repeating_eight_l124_124979


namespace Mary_ends_with_31_eggs_l124_124427

theorem Mary_ends_with_31_eggs (a b : ℕ) (h1 : a = 27) (h2 : b = 4) : a + b = 31 := by
  sorry

end Mary_ends_with_31_eggs_l124_124427


namespace find_m_l124_124465

variables (x m : ℝ)

def equation (x m : ℝ) : Prop := 3 * x - 2 * m = 4

theorem find_m (h1 : equation 6 m) : m = 7 :=
by
  sorry

end find_m_l124_124465


namespace find_a_if_parallel_l124_124406

-- Define the parallel condition for the given lines
def is_parallel (a : ℝ) : Prop :=
  let slope1 := -a / 2
  let slope2 := 3
  slope1 = slope2

-- Prove that a = -6 under the parallel condition
theorem find_a_if_parallel (a : ℝ) (h : is_parallel a) : a = -6 := by
  sorry

end find_a_if_parallel_l124_124406


namespace cyclist_A_speed_l124_124720

theorem cyclist_A_speed (a b : ℝ) (h1 : b = a + 5)
    (h2 : 80 / a = 120 / b) : a = 10 :=
by
  sorry

end cyclist_A_speed_l124_124720


namespace find_m_collinear_l124_124769

-- Definition of a point in 2D space
structure Point2D where
  x : ℤ
  y : ℤ

-- Predicate to check if three points are collinear 
def collinear_points (p1 p2 p3 : Point2D) : Prop :=
  (p3.x - p2.x) * (p2.y - p1.y) = (p2.x - p1.x) * (p3.y - p2.y)

-- Given points A, B, and C
def A : Point2D := ⟨2, 3⟩
def B (m : ℤ) : Point2D := ⟨-4, m⟩
def C : Point2D := ⟨-12, -1⟩

-- Theorem stating the value of m such that points A, B, and C are collinear
theorem find_m_collinear : ∃ (m : ℤ), collinear_points A (B m) C ∧ m = 9 / 7 := sorry

end find_m_collinear_l124_124769


namespace problem_l124_124030

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : odd_function f
axiom f_property : ∀ x : ℝ, f (x + 2) = -f x
axiom f_at_1 : f 1 = 8

theorem problem : f 2012 + f 2013 + f 2014 = 8 := by
  sorry

end problem_l124_124030


namespace intersection_of_P_and_Q_l124_124408

def P (x : ℝ) : Prop := 2 ≤ x ∧ x < 4
def Q (x : ℝ) : Prop := 3 * x - 7 ≥ 8 - 2 * x

theorem intersection_of_P_and_Q :
  ∀ x, P x ∧ Q x ↔ 3 ≤ x ∧ x < 4 :=
by
  sorry

end intersection_of_P_and_Q_l124_124408


namespace polar_coordinates_of_2_neg2_l124_124223

noncomputable def polar_coordinates (x y : ℝ) : ℝ × ℝ :=
  let ρ := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan (y / x)
  (ρ, θ)

theorem polar_coordinates_of_2_neg2 :
  polar_coordinates 2 (-2) = (2 * Real.sqrt 2, -Real.pi / 4) :=
by
  sorry

end polar_coordinates_of_2_neg2_l124_124223


namespace total_distance_between_first_and_fifth_poles_l124_124768

noncomputable def distance_between_poles (n : ℕ) (d : ℕ) : ℕ :=
  d / n

theorem total_distance_between_first_and_fifth_poles :
  ∀ (n : ℕ) (d : ℕ), (n = 3 ∧ d = 90) → (4 * distance_between_poles n d = 120) :=
by
  sorry

end total_distance_between_first_and_fifth_poles_l124_124768


namespace sum_of_squares_l124_124501

-- Define the proposition as a universal statement 
theorem sum_of_squares (a b : ℝ) : a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by
  sorry

end sum_of_squares_l124_124501


namespace larger_pile_toys_l124_124581

-- Define the conditions
def total_toys (small_pile large_pile : ℕ) : Prop := small_pile + large_pile = 120
def larger_pile (small_pile large_pile : ℕ) : Prop := large_pile = 2 * small_pile

-- Define the proof problem
theorem larger_pile_toys (small_pile large_pile : ℕ) (h1 : total_toys small_pile large_pile) (h2 : larger_pile small_pile large_pile) : 
  large_pile = 80 := by
  sorry

end larger_pile_toys_l124_124581


namespace product_of_primes_95_l124_124086

theorem product_of_primes_95 (p q : Nat) (hp : Nat.Prime p) (hq : Nat.Prime q) (h : p + q = 95) : p * q = 178 := sorry

end product_of_primes_95_l124_124086


namespace find_a_plus_b_l124_124187

theorem find_a_plus_b (x a b : ℝ) (ha : a > 0) (hb : b > 0)
  (hx : x = a + Real.sqrt b)
  (hxeq : x^2 + 5*x + 5/x + 1/(x^2) = 42) : a + b = 5 :=
sorry

end find_a_plus_b_l124_124187


namespace determine_n_l124_124883

theorem determine_n (n : ℕ) (h : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^26) : n = 25 :=
by
  sorry

end determine_n_l124_124883


namespace total_soccer_games_l124_124114

theorem total_soccer_games (months : ℕ) (games_per_month : ℕ) (h_months : months = 3) (h_games_per_month : games_per_month = 9) : months * games_per_month = 27 :=
by
  sorry

end total_soccer_games_l124_124114


namespace susan_walked_9_miles_l124_124167

theorem susan_walked_9_miles (E S : ℕ) (h1 : E + S = 15) (h2 : E = S - 3) : S = 9 :=
by
  sorry

end susan_walked_9_miles_l124_124167


namespace smallest_number_is_33_l124_124477

theorem smallest_number_is_33 
  (x : ℕ) 
  (h1 : ∀ y z, y = 2 * x → z = 4 * x → (x + y + z) / 3 = 77) : 
  x = 33 :=
by
  sorry

end smallest_number_is_33_l124_124477


namespace profit_percentage_on_cost_price_l124_124095

theorem profit_percentage_on_cost_price (CP MP SP : ℝ)
    (h1 : CP = 100)
    (h2 : MP = 131.58)
    (h3 : SP = 0.95 * MP) :
    ((SP - CP) / CP) * 100 = 25 :=
by
  -- Sorry to skip the proof
  sorry

end profit_percentage_on_cost_price_l124_124095


namespace norma_cards_left_l124_124998

def initial_cards : ℕ := 88
def lost_cards : ℕ := 70
def remaining_cards (initial lost : ℕ) : ℕ := initial - lost

theorem norma_cards_left : remaining_cards initial_cards lost_cards = 18 := by
  sorry

end norma_cards_left_l124_124998


namespace number_of_girls_l124_124404

-- Given conditions
def ratio_girls_boys_teachers (girls boys teachers : ℕ) : Prop :=
  3 * (girls + boys + teachers) = 3 * girls + 2 * boys + 1 * teachers

def total_people (total girls boys teachers : ℕ) : Prop :=
  total = girls + boys + teachers

-- Define the main theorem
theorem number_of_girls 
  (k total : ℕ)
  (h1 : ratio_girls_boys_teachers (3 * k) (2 * k) k)
  (h2 : total_people total (3 * k) (2 * k) k)
  (h_total : total = 60) : 
  3 * k = 30 :=
  sorry

end number_of_girls_l124_124404


namespace prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l124_124509

-- Show that if \( p \) is a prime number, then \( p \) divides \( (p-1)! + 1 \).
theorem prime_divides_factorial_plus_one (p : ℕ) (hp : Nat.Prime p) : p ∣ (Nat.factorial (p - 1) + 1) :=
sorry

-- Show that if \( n \) is not a prime number, then \( n \) does not divide \( (n-1)! + 1 \).
theorem non_prime_not_divides_factorial_plus_one (n : ℕ) (hn : ¬Nat.Prime n) : ¬(n ∣ (Nat.factorial (n - 1) + 1)) :=
sorry

-- Calculate the remainder of the division of \((n-1)!\) by \( n \).
theorem factorial_mod_non_prime_is_zero (n : ℕ) (hn : ¬Nat.Prime n) : (Nat.factorial (n - 1)) % n = 0 :=
sorry

end prime_divides_factorial_plus_one_non_prime_not_divides_factorial_plus_one_factorial_mod_non_prime_is_zero_l124_124509


namespace intersection_points_l124_124238

noncomputable def parabola (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 4
noncomputable def line (x : ℝ) : ℝ := -x + 2

theorem intersection_points :
  (parabola (-1 / 3) = line (-1 / 3) ∧ parabola (-2) = line (-2)) ∧
  (parabola (-1 / 3) = 7 / 3) ∧ (parabola (-2) = 4) :=
by
  sorry

end intersection_points_l124_124238


namespace basic_astrophysics_degrees_l124_124144

open Real

theorem basic_astrophysics_degrees :
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  basic_astrophysics_percentage / 100 * circle_degrees = 43.2 :=
by
  let microphotonics := 12
  let home_electronics := 24
  let food_additives := 15
  let gmo := 29
  let industrial_lubricants := 8
  let total_percentage := microphotonics + home_electronics + food_additives + gmo + industrial_lubricants
  let basic_astrophysics_percentage := 100 - total_percentage
  let circle_degrees := 360
  exact sorry

end basic_astrophysics_degrees_l124_124144


namespace find_solutions_l124_124857

theorem find_solutions (x : ℝ) :
  (16 * x - x^2) / (x + 2) * (x + (16 - x) / (x + 2)) = 48 →
  (x = 1.2 ∨ x = -81.2) :=
by sorry

end find_solutions_l124_124857


namespace solve_for_x_l124_124357

theorem solve_for_x : ∀ x : ℕ, x + 1315 + 9211 - 1569 = 11901 → x = 2944 :=
by
  intro x
  intro h
  sorry

end solve_for_x_l124_124357


namespace carrots_picked_next_day_l124_124133

theorem carrots_picked_next_day :
  ∀ (initial_picked thrown_out additional_picked total : ℕ),
    initial_picked = 48 →
    thrown_out = 11 →
    total = 52 →
    additional_picked = total - (initial_picked - thrown_out) →
    additional_picked = 15 :=
by
  intros initial_picked thrown_out additional_picked total h_ip h_to h_total h_ap
  sorry

end carrots_picked_next_day_l124_124133


namespace ellipse_focus_value_l124_124096

theorem ellipse_focus_value (k : ℝ) (hk : 5 * (0:ℝ)^2 - k * (2:ℝ)^2 = 5) : k = -1 :=
by
  sorry

end ellipse_focus_value_l124_124096


namespace make_fraction_meaningful_l124_124291

theorem make_fraction_meaningful (x : ℝ) : (x - 1) ≠ 0 ↔ x ≠ 1 :=
by
  sorry

end make_fraction_meaningful_l124_124291


namespace solve_quadratic_l124_124614

noncomputable def quadratic_roots (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem solve_quadratic : ∀ x : ℝ, quadratic_roots 1 (-4) (-5) x ↔ (x = -1 ∨ x = 5) :=
by
  intro x
  rw [quadratic_roots]
  sorry

end solve_quadratic_l124_124614


namespace triangle_value_l124_124811

variable (triangle p : ℝ)

theorem triangle_value : (triangle + p = 75 ∧ 3 * (triangle + p) - p = 198) → triangle = 48 :=
by
  sorry

end triangle_value_l124_124811


namespace infinite_series_computation_l124_124585

theorem infinite_series_computation : 
  ∑' k : ℕ, (8^k) / ((2^k - 1) * (2^(k + 1) - 1)) = 4 :=
by
  sorry

end infinite_series_computation_l124_124585


namespace general_equation_of_curve_l124_124234

variable (θ x y : ℝ)

theorem general_equation_of_curve
  (h1 : x = Real.cos θ - 1)
  (h2 : y = Real.sin θ + 1) :
  (x + 1)^2 + (y - 1)^2 = 1 := sorry

end general_equation_of_curve_l124_124234


namespace range_of_m_l124_124273

theorem range_of_m (x y m : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (2 / x) + (3 / y) = 1)
  (h4 : 3 * x + 2 * y > m^2 + 2 * m) :
  -6 < m ∧ m < 4 :=
sorry

end range_of_m_l124_124273


namespace mass_percentage_of_H_in_ascorbic_acid_l124_124727

-- Definitions based on the problem conditions
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.01
def molar_mass_O : ℝ := 16.00

def ascorbic_acid_molecular_formula_C : ℝ := 6
def ascorbic_acid_molecular_formula_H : ℝ := 8
def ascorbic_acid_molecular_formula_O : ℝ := 6

noncomputable def ascorbic_acid_molar_mass : ℝ :=
  ascorbic_acid_molecular_formula_C * molar_mass_C + 
  ascorbic_acid_molecular_formula_H * molar_mass_H + 
  ascorbic_acid_molecular_formula_O * molar_mass_O

noncomputable def hydrogen_mass_in_ascorbic_acid : ℝ :=
  ascorbic_acid_molecular_formula_H * molar_mass_H

noncomputable def hydrogen_mass_percentage_in_ascorbic_acid : ℝ :=
  (hydrogen_mass_in_ascorbic_acid / ascorbic_acid_molar_mass) * 100

theorem mass_percentage_of_H_in_ascorbic_acid :
  hydrogen_mass_percentage_in_ascorbic_acid = 4.588 :=
by
  sorry

end mass_percentage_of_H_in_ascorbic_acid_l124_124727


namespace problem1_problem2_problem3_problem4_l124_124170

theorem problem1 : (-8) + 10 - 2 + (-1) = -1 := 
by
  sorry

theorem problem2 : 12 - 7 * (-4) + 8 / (-2) = 36 := 
by 
  sorry

theorem problem3 : ( (1/2) + (1/3) - (1/6) ) / (-1/18) = -12 := 
by 
  sorry

theorem problem4 : - 1 ^ 4 - (1 + 0.5) * (1/3) * (-4) ^ 2 = -33 / 32 := 
by 
  sorry


end problem1_problem2_problem3_problem4_l124_124170


namespace cubic_polynomial_root_l124_124890

theorem cubic_polynomial_root (a b c : ℕ) (h : 27 * x^3 - 9 * x^2 - 9 * x - 3 = 0) : 
  (a + b + c = 11) :=
sorry

end cubic_polynomial_root_l124_124890


namespace nth_equation_l124_124231

theorem nth_equation (n : ℕ) : 
  n^2 + (n + 1)^2 = (n * (n + 1) + 1)^2 - (n * (n + 1))^2 :=
by
  sorry

end nth_equation_l124_124231


namespace cheenu_time_difference_l124_124745

def cheenu_bike_time_per_mile (distance_bike : ℕ) (time_bike : ℕ) : ℕ := time_bike / distance_bike
def cheenu_walk_time_per_mile (distance_walk : ℕ) (time_walk : ℕ) : ℕ := time_walk / distance_walk
def time_difference (time1 : ℕ) (time2 : ℕ) : ℕ := time2 - time1

theorem cheenu_time_difference 
  (distance_bike : ℕ) (time_bike : ℕ) 
  (distance_walk : ℕ) (time_walk : ℕ) 
  (H_bike : distance_bike = 20) (H_time_bike : time_bike = 80) 
  (H_walk : distance_walk = 8) (H_time_walk : time_walk = 160) :
  time_difference (cheenu_bike_time_per_mile distance_bike time_bike) (cheenu_walk_time_per_mile distance_walk time_walk) = 16 := 
by
  sorry

end cheenu_time_difference_l124_124745


namespace second_term_arithmetic_sequence_l124_124523

theorem second_term_arithmetic_sequence (a d : ℝ) (h : a + (a + 2 * d) = 10) : 
  a + d = 5 :=
by
  sorry

end second_term_arithmetic_sequence_l124_124523


namespace area_of_L_equals_22_l124_124712

-- Define the dimensions of the rectangles
def big_rectangle_length := 8
def big_rectangle_width := 5
def small_rectangle_length := big_rectangle_length - 2
def small_rectangle_width := big_rectangle_width - 2

-- Define the areas
def area_big_rectangle := big_rectangle_length * big_rectangle_width
def area_small_rectangle := small_rectangle_length * small_rectangle_width

-- Define the area of the "L" shape
def area_L := area_big_rectangle - area_small_rectangle

-- State the theorem
theorem area_of_L_equals_22 : area_L = 22 := by
  -- The proof would go here
  sorry

end area_of_L_equals_22_l124_124712


namespace binomial_expansion_product_l124_124073

theorem binomial_expansion_product (a a1 a2 a3 a4 a5 : ℤ)
  (h1 : (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5)
  (h2 : (1 - (-1))^5 = a - a1 + a2 - a3 + a4 - a5) :
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := by
  sorry

end binomial_expansion_product_l124_124073


namespace walking_distance_l124_124082

theorem walking_distance (D : ℕ) (h : D / 15 = (D + 60) / 30) : D = 60 :=
by
  sorry

end walking_distance_l124_124082


namespace flight_time_l124_124933

def eagle_speed : ℕ := 15
def falcon_speed : ℕ := 46
def pelican_speed : ℕ := 33
def hummingbird_speed : ℕ := 30
def total_distance : ℕ := 248

theorem flight_time : (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) > 0 → 
                      total_distance / (eagle_speed + falcon_speed + pelican_speed + hummingbird_speed) = 2 :=
by
  -- Proof is skipped
  sorry

end flight_time_l124_124933


namespace geometric_triangle_q_range_l124_124290

theorem geometric_triangle_q_range (a : ℝ) (q : ℝ) (h : 0 < q) 
  (h1 : a + q * a > (q ^ 2) * a)
  (h2 : q * a + (q ^ 2) * a > a)
  (h3 : a + (q ^ 2) * a > q * a) : 
  q ∈ Set.Ioo ((Real.sqrt 5 - 1) / 2) ((1 + Real.sqrt 5) / 2) :=
sorry

end geometric_triangle_q_range_l124_124290


namespace UnionMathInstitute_students_l124_124349

theorem UnionMathInstitute_students :
  ∃ n : ℤ, n < 500 ∧ 
    n % 17 = 15 ∧ 
    n % 19 = 18 ∧ 
    n % 16 = 7 ∧ 
    n = 417 :=
by
  -- Problem setup and constraints
  sorry

end UnionMathInstitute_students_l124_124349


namespace domain_of_g_cauchy_schwarz_inequality_l124_124118

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

-- Question 1: Prove the domain of g(x) = log(f(x) - 2) is {x | 0.5 < x < 2.5}
theorem domain_of_g : {x : ℝ | 0.5 < x ∧ x < 2.5} = {x : ℝ | 0.5 < x ∧ x < 2.5} :=
by
  sorry

-- Minimum value of f(x)
def m : ℝ := 1

-- Question 2: Prove a^2 + b^2 + c^2 ≥ 1/3 given a + b + c = m
theorem cauchy_schwarz_inequality (a b c : ℝ) (h : a + b + c = m) : a^2 + b^2 + c^2 ≥ 1 / 3 :=
by
  sorry

end domain_of_g_cauchy_schwarz_inequality_l124_124118


namespace cube_side_length_equals_six_l124_124605

theorem cube_side_length_equals_six {s : ℝ} (h : 6 * s ^ 2 = s ^ 3) : s = 6 :=
by
  sorry

end cube_side_length_equals_six_l124_124605


namespace remainder_25197629_mod_4_l124_124543

theorem remainder_25197629_mod_4 : 25197629 % 4 = 1 := by
  sorry

end remainder_25197629_mod_4_l124_124543


namespace integer_solutions_determinant_l124_124863

theorem integer_solutions_determinant (a b c d : ℤ)
    (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
    a * d - b * c = 1 ∨ a * d - b * c = -1 :=
sorry

end integer_solutions_determinant_l124_124863


namespace highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l124_124433

theorem highest_power_of_two_factor_13_pow_4_minus_11_pow_4 :
  ∃ n : ℕ, n = 5 ∧ (2 ^ n ∣ (13 ^ 4 - 11 ^ 4)) ∧ ¬ (2 ^ (n + 1) ∣ (13 ^ 4 - 11 ^ 4)) :=
sorry

end highest_power_of_two_factor_13_pow_4_minus_11_pow_4_l124_124433


namespace negation_of_proposition_l124_124457

theorem negation_of_proposition (x : ℝ) :
  ¬ (∃ x > -1, x^2 + x - 2018 > 0) ↔ ∀ x > -1, x^2 + x - 2018 ≤ 0 := sorry

end negation_of_proposition_l124_124457


namespace geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l124_124643

theorem geometric_sequence_general_term_and_arithmetic_sequence_max_sum :
  (∃ a_n : ℕ → ℕ, ∃ b_n : ℕ → ℤ, ∃ T_n : ℕ → ℤ,
    (∀ n, a_n n = 2^(n-1)) ∧
    (a_n 1 + a_n 2 = 3) ∧
    (b_n 2 = a_n 3) ∧
    (b_n 3 = -b_n 5) ∧
    (∀ n, T_n n = n * (b_n 1 + b_n n) / 2) ∧
    (T_n 3 = 12) ∧
    (T_n 4 = 12)) :=
by
  sorry

end geometric_sequence_general_term_and_arithmetic_sequence_max_sum_l124_124643


namespace find_x_plus_y_l124_124676

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.cos y = 3000)
  (h2 : x + 3000 * Real.sin y = 2999) (h3 : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
  sorry

end find_x_plus_y_l124_124676


namespace gcd_of_987654_and_123456_l124_124624

theorem gcd_of_987654_and_123456 : Nat.gcd 987654 123456 = 6 := by
  sorry

end gcd_of_987654_and_123456_l124_124624


namespace correct_factorization_l124_124685

-- Definitions for the given conditions of different options
def condition_A (a : ℝ) : Prop := 2 * a^2 - 2 * a + 1 = 2 * a * (a - 1) + 1
def condition_B (x y : ℝ) : Prop := (x + y) * (x - y) = x^2 - y^2
def condition_C (x y : ℝ) : Prop := x^2 - 4 * x * y + 4 * y^2 = (x - 2 * y)^2
def condition_D (x : ℝ) : Prop := x^2 + 1 = x * (x + 1 / x)

-- The theorem to prove that option C is correct
theorem correct_factorization (x y : ℝ) : condition_C x y :=
by sorry

end correct_factorization_l124_124685


namespace factor_value_l124_124877

theorem factor_value 
  (m : ℝ) 
  (h : ∀ x : ℝ, x + 5 = 0 → (x^2 - m * x - 40) = 0) : 
  m = 3 := 
sorry

end factor_value_l124_124877


namespace fill_pipe_fraction_l124_124374

theorem fill_pipe_fraction (t : ℕ) (f : ℝ) (h : t = 30) (h' : f = 1) : f = 1 :=
by
  sorry

end fill_pipe_fraction_l124_124374


namespace simplify_expression_l124_124459

theorem simplify_expression (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b) (h4 : a ≠ -b) : 
  ((a^3 - a^2 * b) / (a^2 * b) - (a^2 * b - b^3) / (a * b - b^2) - (a * b) / (a^2 - b^2)) = 
  (-3 * a) / (a^2 - b^2) := 
by
  sorry

end simplify_expression_l124_124459


namespace magician_earning_l124_124986

-- Definitions based on conditions
def price_per_deck : ℕ := 2
def initial_decks : ℕ := 5
def remaining_decks : ℕ := 3

-- Theorem statement
theorem magician_earning :
  let sold_decks := initial_decks - remaining_decks
  let earning := sold_decks * price_per_deck
  earning = 4 := by
  sorry

end magician_earning_l124_124986


namespace sum_of_two_numbers_l124_124253

theorem sum_of_two_numbers (x y : ℝ) (h1 : x * y = 16) (h2 : 1/x = 3 * (1/y)) : 
  x + y = 16 * Real.sqrt 3 / 3 :=
by
  sorry

end sum_of_two_numbers_l124_124253


namespace smallest_possible_value_of_n_l124_124983

theorem smallest_possible_value_of_n 
  {a b c m n : ℕ} 
  (ha_pos : a > 0) 
  (hb_pos : b > 0) 
  (hc_pos : c > 0) 
  (h_ordering : a ≥ b ∧ b ≥ c) 
  (h_sum : a + b + c = 3010) 
  (h_factorial : a.factorial * b.factorial * c.factorial = m * 10^n) 
  (h_m_not_div_10 : ¬ (10 ∣ m)) 
  : n = 746 := 
sorry

end smallest_possible_value_of_n_l124_124983


namespace not_basic_logical_structure_l124_124046

def basic_structures : Set String := {"Sequential structure", "Conditional structure", "Loop structure"}

theorem not_basic_logical_structure : "Operational structure" ∉ basic_structures := by
  sorry

end not_basic_logical_structure_l124_124046


namespace isosceles_triangle_base_angle_l124_124514

theorem isosceles_triangle_base_angle (vertex_angle : ℝ) (base_angle : ℝ) 
  (h1 : vertex_angle = 60) 
  (h2 : 2 * base_angle + vertex_angle = 180) : 
  base_angle = 60 := 
by 
  sorry

end isosceles_triangle_base_angle_l124_124514


namespace log_function_domain_l124_124723

theorem log_function_domain (x : ℝ) : 
  (3 - x > 0) ∧ (x - 1 > 0) ∧ (x - 1 ≠ 1) -> (1 < x ∧ x < 3 ∧ x ≠ 2) :=
by
  intro h
  sorry

end log_function_domain_l124_124723


namespace xiao_ying_correct_answers_at_least_l124_124956

def total_questions : ℕ := 20
def points_correct : ℕ := 5
def points_incorrect : ℕ := 2
def excellent_points : ℕ := 80

theorem xiao_ying_correct_answers_at_least (x : ℕ) :
  (5 * x - 2 * (total_questions - x)) ≥ excellent_points → x ≥ 18 := by
  sorry

end xiao_ying_correct_answers_at_least_l124_124956


namespace exam_students_l124_124785

noncomputable def totalStudents (N : ℕ) (T : ℕ) := T = 70 * N
noncomputable def marksOfExcludedStudents := 5 * 50
noncomputable def remainingStudents (N : ℕ) := N - 5
noncomputable def remainingMarksCondition (N T : ℕ) := (T - marksOfExcludedStudents) / remainingStudents N = 90

theorem exam_students (N : ℕ) (T : ℕ) 
  (h1 : totalStudents N T) 
  (h2 : remainingMarksCondition N T) : 
  N = 10 :=
by 
  sorry

end exam_students_l124_124785


namespace right_triangle_proportion_l124_124939

/-- Given a right triangle ABC with ∠C = 90°, AB = c, AC = b, and BC = a, 
    and a point P on the hypotenuse AB (or its extension) such that 
    AP = m, BP = n, and CP = k, prove that a²m² + b²n² = c²k². -/
theorem right_triangle_proportion
  {a b c m n k : ℝ}
  (h_right : ∀ A B C : ℝ, A^2 + B^2 = C^2)
  (h1 : ∀ P : ℝ, m^2 + n^2 = k^2)
  (h_geometry : a^2 + b^2 = c^2) :
  a^2 * m^2 + b^2 * n^2 = c^2 * k^2 := 
sorry

end right_triangle_proportion_l124_124939


namespace amanda_jogging_distance_l124_124094

/-- Amanda's jogging path and the distance calculation. -/
theorem amanda_jogging_distance:
  let east_leg := 1.5
  let northwest_leg := 2
  let southwest_leg := 1
  -- Convert runs to displacement components
  let nw_x := northwest_leg / Real.sqrt 2
  let nw_y := northwest_leg / Real.sqrt 2
  let sw_x := southwest_leg / Real.sqrt 2
  let sw_y := southwest_leg / Real.sqrt 2
  -- Calculate net displacements
  let net_east := east_leg - (nw_x + sw_x)
  let net_north := nw_y - sw_y
  -- Final distance back to starting point
  let distance := Real.sqrt (net_east^2 + net_north^2)
  distance = Real.sqrt ((1.5 - 3 * Real.sqrt 2 / 2)^2 + (Real.sqrt 2 / 2)^2) := sorry

end amanda_jogging_distance_l124_124094


namespace complex_inequality_l124_124578

open Complex

noncomputable def condition (a b c : ℂ) := a * Complex.abs (b * c) + b * Complex.abs (c * a) + c * Complex.abs (a * b) = 0

theorem complex_inequality (a b c : ℂ) (h : condition a b c) :
  Complex.abs ((a - b) * (b - c) * (c - a)) ≥ 3 * Real.sqrt 3 * Complex.abs (a * b * c) := 
sorry

end complex_inequality_l124_124578


namespace no_common_points_lines_l124_124499

theorem no_common_points_lines (m : ℝ) : 
    ¬∃ x y : ℝ, (x + m^2 * y + 6 = 0) ∧ ((m - 2) * x + 3 * m * y + 2 * m = 0) ↔ m = 0 ∨ m = -1 := 
by 
    sorry

end no_common_points_lines_l124_124499


namespace tan_pi_div_4_add_alpha_l124_124139

theorem tan_pi_div_4_add_alpha (α : ℝ) (h : Real.sin α = 2 * Real.cos α) : 
  Real.tan (π / 4 + α) = -3 :=
by
  sorry

end tan_pi_div_4_add_alpha_l124_124139


namespace hexagon_inequality_l124_124948

noncomputable def ABCDEF := 3 * Real.sqrt 3 / 2
noncomputable def ACE := Real.sqrt 3
noncomputable def BDF := Real.sqrt 3
noncomputable def R₁ := Real.sqrt 3 / 4
noncomputable def R₂ := -Real.sqrt 3 / 4

theorem hexagon_inequality :
  min ACE BDF + R₂ - R₁ ≤ 3 * Real.sqrt 3 / 4 :=
by
  sorry

end hexagon_inequality_l124_124948


namespace intercept_sum_l124_124952

theorem intercept_sum (x y : ℝ) :
  (y - 3 = 6 * (x - 5)) →
  (∃ x_intercept, (y = 0) ∧ (x_intercept = 4.5)) →
  (∃ y_intercept, (x = 0) ∧ (y_intercept = -27)) →
  (4.5 + (-27) = -22.5) :=
by
  intros h_eq h_xint h_yint
  sorry

end intercept_sum_l124_124952


namespace find_f_f_2_l124_124031

def f (x : ℝ) : ℝ := 3 * x - 1

theorem find_f_f_2 :
  f (f 2) = 14 :=
by
sorry

end find_f_f_2_l124_124031


namespace weight_problem_l124_124106

variable (M T : ℕ)

theorem weight_problem
  (h1 : 220 = 3 * M + 10)
  (h2 : T = 2 * M)
  (h3 : 2 * T = 220) :
  M = 70 ∧ T = 140 :=
by
  sorry

end weight_problem_l124_124106


namespace inequality_solution_set_l124_124680

theorem inequality_solution_set (x : ℝ) : (2 * x + 1 ≥ 3) ∧ (4 * x - 1 < 7) ↔ (1 ≤ x ∧ x < 2) :=
by
  sorry

end inequality_solution_set_l124_124680


namespace number_of_pencils_l124_124982

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l124_124982


namespace complement_union_l124_124914

open Set

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 5, 6, 8})
  (hA : A = {1, 5, 8})(hB : B = {2}) :
  (U \ A) ∪ B = {0, 2, 3, 6} :=
by
  rw [hU, hA, hB]
  -- Intermediate steps would go here
  sorry

end complement_union_l124_124914


namespace calc_log_expression_l124_124414

theorem calc_log_expression : 2 * Real.log 5 + Real.log 4 = 2 :=
by
  sorry

end calc_log_expression_l124_124414


namespace James_beat_record_by_72_l124_124575

-- Define the conditions as given in the problem
def touchdowns_per_game : ℕ := 4
def points_per_touchdown : ℕ := 6
def games_in_season : ℕ := 15
def conversions : ℕ := 6
def points_per_conversion : ℕ := 2
def old_record : ℕ := 300

-- Define the necessary calculations based on the conditions
def points_from_touchdowns_per_game : ℕ := touchdowns_per_game * points_per_touchdown
def points_from_touchdowns_in_season : ℕ := games_in_season * points_from_touchdowns_per_game
def points_from_conversions : ℕ := conversions * points_per_conversion
def total_points_in_season : ℕ := points_from_touchdowns_in_season + points_from_conversions
def points_above_old_record : ℕ := total_points_in_season - old_record

-- State the proof problem
theorem James_beat_record_by_72 : points_above_old_record = 72 :=
by
  sorry

end James_beat_record_by_72_l124_124575


namespace alyssa_plums_correct_l124_124862

def total_plums : ℕ := 27
def jason_plums : ℕ := 10
def alyssa_plums : ℕ := 17

theorem alyssa_plums_correct : alyssa_plums = total_plums - jason_plums := by
  sorry

end alyssa_plums_correct_l124_124862


namespace train_distance_l124_124762

theorem train_distance (t : ℕ) (d : ℕ) (rate : d / t = 1 / 2) (total_time : ℕ) (h : total_time = 90) : ∃ distance : ℕ, distance = 45 := by
  sorry

end train_distance_l124_124762


namespace vector_magnitude_proof_l124_124066

theorem vector_magnitude_proof (a b : ℝ × ℝ) 
  (h₁ : ‖a‖ = 1) 
  (h₂ : ‖b‖ = 2)
  (h₃ : a - b = (Real.sqrt 3, Real.sqrt 2)) : 
‖a + (2:ℝ) • b‖ = Real.sqrt 17 := 
sorry

end vector_magnitude_proof_l124_124066


namespace second_set_parallel_lines_l124_124057

theorem second_set_parallel_lines (n : ℕ) (h : 7 * (n - 1) = 784) : n = 113 := 
by
  sorry

end second_set_parallel_lines_l124_124057


namespace proof_problem_l124_124835

-- Given conditions
variables {a b : Type}  -- Two non-coincident lines
variables {α β : Type}  -- Two non-coincident planes

-- Definitions of the relationships
def is_parallel_to (x y : Type) : Prop := sorry  -- Parallel relationship
def is_perpendicular_to (x y : Type) : Prop := sorry  -- Perpendicular relationship

-- Statements to verify
def statement1 (a α b : Type) : Prop := 
  (is_parallel_to a α ∧ is_parallel_to b α) → is_parallel_to a b

def statement2 (a α β : Type) : Prop :=
  (is_perpendicular_to a α ∧ is_perpendicular_to a β) → is_parallel_to α β

def statement3 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ l : Type, is_perpendicular_to l α ∧ is_parallel_to l β

def statement4 (α β : Type) : Prop :=
  is_perpendicular_to α β → ∃ γ : Type, is_perpendicular_to γ α ∧ is_perpendicular_to γ β

-- Proof problem: verifying which statements are true.
theorem proof_problem :
  ¬ (statement1 a α b) ∧ statement2 a α β ∧ statement3 α β ∧ statement4 α β :=
by
  sorry

end proof_problem_l124_124835


namespace find_number_l124_124691

theorem find_number (number : ℝ) (h : 0.003 * number = 0.15) : number = 50 :=
by
  sorry

end find_number_l124_124691


namespace average_of_expressions_l124_124295

theorem average_of_expressions (y : ℝ) :
  (1 / 3:ℝ) * ((2 * y + 5) + (3 * y + 4) + (7 * y - 2)) = 4 * y + 7 / 3 :=
by sorry

end average_of_expressions_l124_124295


namespace find_x_plus_inv_x_l124_124696

theorem find_x_plus_inv_x (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end find_x_plus_inv_x_l124_124696


namespace unique_triangle_constructions_l124_124344

structure Triangle :=
(a b c : ℝ) (A B C : ℝ)

-- Definitions for the conditions
def SSS (t : Triangle) : Prop := 
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0

def SAS (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180

def ASA (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.c > 0 ∧ t.A + t.B < 180

def SSA (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.A > 0 ∧ t.A < 180 

-- The formally stated proof goal
theorem unique_triangle_constructions (t : Triangle) :
  (SSS t ∨ SAS t ∨ ASA t) ∧ ¬(SSA t) :=
by
  sorry

end unique_triangle_constructions_l124_124344


namespace question_1_question_2_l124_124899

-- Condition: The coordinates of point P are given by the equations x = -3a - 4, y = 2 + a

-- Question 1: Prove coordinates when P lies on the x-axis
theorem question_1 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hy0 : y = 0) :
  a = -2 ∧ x = 2 ∧ y = 0 :=
sorry

-- Question 2: Prove coordinates when PQ is parallel to the y-axis
theorem question_2 (a : ℝ) (x : ℝ) (y : ℝ) (h1 : x = -3 * a - 4) (h2 : y = 2 + a) (hx5 : x = 5) :
  a = -3 ∧ x = 5 ∧ y = -1 :=
sorry

end question_1_question_2_l124_124899


namespace find_n_l124_124104

open Nat

theorem find_n (n : ℕ) (d : ℕ → ℕ) (h1 : d 1 = 1) (hk : d 6^2 + d 7^2 - 1 = n) :
  n = 1984 ∨ n = 144 :=
by
  sorry

end find_n_l124_124104


namespace expand_product_l124_124220

theorem expand_product (x : ℝ) : (x + 3) * (x + 9) = x^2 + 12 * x + 27 := 
by sorry

end expand_product_l124_124220


namespace no_solutions_ordered_triples_l124_124163

theorem no_solutions_ordered_triples :
  ¬ ∃ (x y z : ℤ), 
    x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
    -x^2 + 5 * y * z + 3 * z^2 = 55 ∧
    x^2 + 2 * x * y + 9 * z^2 = 150 :=
by
  sorry

end no_solutions_ordered_triples_l124_124163


namespace wrongly_entered_mark_l124_124332

theorem wrongly_entered_mark (x : ℕ) 
    (h1 : x - 33 = 52) : x = 85 :=
by
  sorry

end wrongly_entered_mark_l124_124332


namespace greatest_leftover_cookies_l124_124113

theorem greatest_leftover_cookies (n : ℕ) : ∃ k, k ≤ n ∧ k % 8 = 7 := sorry

end greatest_leftover_cookies_l124_124113


namespace point_translation_l124_124891

theorem point_translation :
  ∃ (x y : ℤ), x = -1 ∧ y = -2 ↔ 
  ∃ (x₀ y₀ : ℤ), 
    x₀ = -3 ∧ y₀ = 2 ∧ 
    x = x₀ + 2 ∧ 
    y = y₀ - 4 := by
  sorry

end point_translation_l124_124891


namespace spadesuit_value_l124_124034

-- Define the operation ♠ as a function
def spadesuit (a b : ℤ) : ℤ := |a - b|

theorem spadesuit_value : spadesuit 3 (spadesuit 5 8) = 0 :=
by
  -- Proof steps go here (we're skipping proof steps and directly writing sorry)
  sorry

end spadesuit_value_l124_124034


namespace problem_statement_l124_124617

-- Define the universal set
def U : Set ℕ := {x | x ≤ 6}

-- Define set A
def A : Set ℕ := {1, 3, 5}

-- Define set B
def B : Set ℕ := {4, 5, 6}

-- Define the complement of A with respect to U
def complement_A : Set ℕ := {x | x ∈ U ∧ x ∉ A}

-- Define the intersection of the complement of A and B
def intersect_complement_A_B : Set ℕ := {x | x ∈ complement_A ∧ x ∈ B}

-- Theorem statement to be proven
theorem problem_statement : intersect_complement_A_B = {4, 6} :=
by
  sorry

end problem_statement_l124_124617


namespace remainder_of_99_times_101_divided_by_9_is_0_l124_124029

theorem remainder_of_99_times_101_divided_by_9_is_0 : (99 * 101) % 9 = 0 :=
by
  sorry

end remainder_of_99_times_101_divided_by_9_is_0_l124_124029


namespace find_line_equation_l124_124276

open Real

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 2 * x

-- Define the line passing through (0,2)
def LineThruPoint (x y k : ℝ) : Prop := y = k * x + 2

-- Define when line intersects parabola
def LineIntersectsParabola (x1 y1 x2 y2 k : ℝ) : Prop :=
  LineThruPoint x1 y1 k ∧ LineThruPoint x2 y2 k ∧ Parabola x1 y1 ∧ Parabola x2 y2

-- Define when circle with diameter MN passes through origin O
def CircleThroughOrigin (x1 y1 x2 y2 : ℝ) : Prop :=
  x1 * x2 + y1 * y2 = 0

theorem find_line_equation (k : ℝ) 
    (h₀ : k ≠ 0)
    (h₁ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k)
    (h₂ : ∃ x1 y1 x2 y2, LineIntersectsParabola x1 y1 x2 y2 k ∧ CircleThroughOrigin x1 y1 x2 y2) :
  (∃ x y, LineThruPoint x y k ∧ y = -x + 2) :=
sorry

end find_line_equation_l124_124276


namespace derivative_at_2_l124_124196

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem derivative_at_2 : deriv f 2 = (1 - Real.log 2) / 4 :=
by
  sorry

end derivative_at_2_l124_124196


namespace question_a_question_b_l124_124625

-- Definitions
def isSolutionA (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 7

def isSolutionB (a b : ℤ) : Prop :=
  1 / (a : ℚ) + 1 / (b : ℚ) = 1 / 25

-- Statements
theorem question_a (a b : ℤ) : isSolutionA a b ↔ (a, b) ∈ [(6, -42), (-42, 6), (8, 56), (56, 8), (14, 14)] :=
sorry

theorem question_b (a b : ℤ) : isSolutionB a b ↔ (a, b) ∈ [(24, -600), (-600, 24), (26, 650), (650, 26), (50, 50)] :=
sorry

end question_a_question_b_l124_124625


namespace problem_divisibility_l124_124141

theorem problem_divisibility 
  (a b c : ℕ) 
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : b ∣ a^3)
  (h2 : c ∣ b^3)
  (h3 : a ∣ c^3) : 
  (a + b + c) ^ 13 ∣ a * b * c := 
sorry

end problem_divisibility_l124_124141


namespace pauls_total_cost_is_252_l124_124103

variable (price_shirt : ℕ) (num_shirts : ℕ)
variable (price_pants : ℕ) (num_pants : ℕ)
variable (price_suit : ℕ) (num_suit : ℕ)
variable (price_sweater : ℕ) (num_sweaters : ℕ)
variable (store_discount : ℕ) (coupon_discount : ℕ)

-- Define the given prices and discounts
def total_cost_before_discounts : ℕ :=
  (price_shirt * num_shirts) +
  (price_pants * num_pants) +
  (price_suit * num_suit) +
  (price_sweater * num_sweaters)

def store_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def coupon_discount_amount (total_cost : ℕ) (discount : ℕ) : ℕ :=
  (total_cost * discount) / 100

def total_cost_after_discounts : ℕ :=
  let initial_total := total_cost_before_discounts price_shirt num_shirts price_pants num_pants price_suit num_suit price_sweater num_sweaters
  let store_discount_value := store_discount_amount initial_total store_discount
  let subtotal_after_store_discount := initial_total - store_discount_value
  let coupon_discount_value := coupon_discount_amount subtotal_after_store_discount coupon_discount
  subtotal_after_store_discount - coupon_discount_value

theorem pauls_total_cost_is_252 :
  total_cost_after_discounts 15 4 40 2 150 1 30 2 20 10 = 252 := by
  sorry

end pauls_total_cost_is_252_l124_124103


namespace max_profit_at_nine_l124_124929

noncomputable def profit_function (x : ℝ) : ℝ :=
  -(1/3) * x ^ 3 + 81 * x - 234

theorem max_profit_at_nine :
  ∃ x, x = 9 ∧ ∀ y : ℝ, profit_function y ≤ profit_function 9 :=
by
  sorry

end max_profit_at_nine_l124_124929


namespace tan_theta_minus_pi_over_4_l124_124507

theorem tan_theta_minus_pi_over_4 (θ : Real) (k : ℤ)
  (h1 : - (π / 2) + (2 * k * π) < θ)
  (h2 : θ < 2 * k * π)
  (h3 : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 :=
sorry

end tan_theta_minus_pi_over_4_l124_124507


namespace canoe_rental_cost_l124_124984

theorem canoe_rental_cost (C : ℕ) (K : ℕ) :
  18 * K + C * (K + 5) = 405 → 
  3 * K = 2 * (K + 5) → 
  C = 15 :=
by
  intros revenue_eq ratio_eq
  sorry

end canoe_rental_cost_l124_124984


namespace per_can_price_difference_cents_l124_124867

   theorem per_can_price_difference_cents :
     let bulk_warehouse_price_per_case := 12.0
     let bulk_warehouse_cans_per_case := 48
     let bulk_warehouse_discount := 0.10
     let local_store_price_per_case := 6.0
     let local_store_cans_per_case := 12
     let local_store_promotion_factor := 1.5 -- represents the effect of the promotion (3 cases for the price of 2.5 cases)
     let bulk_warehouse_price_per_can := (bulk_warehouse_price_per_case * (1 - bulk_warehouse_discount)) / bulk_warehouse_cans_per_case
     let local_store_price_per_can := (local_store_price_per_case * local_store_promotion_factor) / (local_store_cans_per_case * 3)
     let price_difference_cents := (local_store_price_per_can - bulk_warehouse_price_per_can) * 100
     price_difference_cents = 19.17 :=
   by
     sorry
   
end per_can_price_difference_cents_l124_124867


namespace johns_uncommon_cards_l124_124540

def packs_bought : ℕ := 10
def cards_per_pack : ℕ := 20
def uncommon_fraction : ℚ := 1 / 4

theorem johns_uncommon_cards : packs_bought * (cards_per_pack * uncommon_fraction) = (50 : ℚ) := 
by 
  sorry

end johns_uncommon_cards_l124_124540


namespace solution_interval_l124_124565

noncomputable def f (x : ℝ) : ℝ := (1 / 2) ^ x - x^(1 / 3)

theorem solution_interval (x₀ : ℝ) 
  (h_solution : (1 / 2)^x₀ = x₀^(1 / 3)) : x₀ ∈ Set.Ioo (1 / 3) (1 / 2) :=
by
  sorry

end solution_interval_l124_124565


namespace bumper_cars_initial_count_l124_124780

variable {X : ℕ}

theorem bumper_cars_initial_count (h : (X - 6) + 3 = 6) : X = 9 := 
by
  sorry

end bumper_cars_initial_count_l124_124780


namespace least_number_remainder_l124_124496

theorem least_number_remainder (n : ℕ) (h : 20 ∣ (n - 5)) : n = 125 := sorry

end least_number_remainder_l124_124496


namespace cost_of_cucumbers_l124_124102

theorem cost_of_cucumbers (C : ℝ) (h1 : ∀ (T : ℝ), T = 0.80 * C)
  (h2 : 2 * (0.80 * C) + 3 * C = 23) : C = 5 := by
  sorry

end cost_of_cucumbers_l124_124102


namespace externally_tangent_circles_radius_l124_124973

theorem externally_tangent_circles_radius :
  ∃ r : ℝ, r > 0 ∧ (∀ x y, (x^2 + y^2 = 1 ∧ ((x - 3)^2 + y^2 = r^2)) → r = 2) :=
sorry

end externally_tangent_circles_radius_l124_124973


namespace find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l124_124092

-- Define the function f(x) and its derivative f'(x)
noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

-- Prove that a = 2 given the slope condition at x = 0
theorem find_a (a : ℝ) (h : f_prime 0 a = -1) : a = 2 :=
by sorry

-- Characteristics of the function f(x)
theorem monotonic_intervals (a : ℝ) (h : a = 2) :
  ∀ x : ℝ, (x ≤ Real.log 2 → f_prime x a ≤ 0) ∧ (x >= Real.log 2 → f_prime x a >= 0) :=
by sorry

-- Prove that e^x > x^2 + 1 when x > 0
theorem exp_gt_xsquare_plus_one (x : ℝ) (hx : x > 0) : Real.exp x > x^2 + 1 :=
by sorry

end find_a_monotonic_intervals_exp_gt_xsquare_plus_one_l124_124092


namespace root_in_interval_l124_124085

def f (x : ℝ) : ℝ := x^3 + 5 * x^2 - 3 * x + 1

theorem root_in_interval : ∃ A B : ℤ, B = A + 1 ∧ (∃ ξ : ℝ, f ξ = 0 ∧ (A : ℝ) < ξ ∧ ξ < (B : ℝ)) ∧ A = -6 ∧ B = -5 :=
by
  sorry

end root_in_interval_l124_124085


namespace value_of_k_l124_124172

theorem value_of_k {k : ℝ} :
  (∀ x : ℝ, (x^2 + k * x + 24 > 0) ↔ (x < -6 ∨ x > 4)) →
  k = 2 :=
by
  sorry

end value_of_k_l124_124172


namespace find_abs_3h_minus_4k_l124_124169

theorem find_abs_3h_minus_4k
  (h k : ℤ)
  (factor1_eq_zero : 3 * (-3)^3 - h * (-3) - 3 * k = 0)
  (factor2_eq_zero : 3 * 2^3 - h * 2 - 3 * k = 0) :
  |3 * h - 4 * k| = 615 :=
by
  sorry

end find_abs_3h_minus_4k_l124_124169


namespace find_speed_of_faster_train_l124_124371

noncomputable def speed_of_faster_train
  (length_each_train_m : ℝ)
  (speed_slower_kmph : ℝ)
  (time_pass_s : ℝ) : ℝ :=
  let distance_km := (2 * length_each_train_m / 1000)
  let time_pass_hr := (time_pass_s / 3600)
  let relative_speed_kmph := (distance_km / time_pass_hr)
  let speed_faster_kmph := (relative_speed_kmph - speed_slower_kmph)
  speed_faster_kmph

theorem find_speed_of_faster_train :
  speed_of_faster_train
    250   -- length_each_train_m
    30    -- speed_slower_kmph
    23.998080153587715 -- time_pass_s
  = 45 := sorry

end find_speed_of_faster_train_l124_124371


namespace shortest_distance_between_circles_l124_124665

theorem shortest_distance_between_circles :
  let c1 := (1, -3)
  let r1 := 2 * Real.sqrt 2
  let c2 := (-3, 1)
  let r2 := 1
  let distance_centers := Real.sqrt ((1 - -3)^2 + (-3 - 1)^2)
  let shortest_distance := distance_centers - (r1 + r2)
  shortest_distance = 2 * Real.sqrt 2 - 1 :=
by
  sorry

end shortest_distance_between_circles_l124_124665


namespace largest_constant_inequality_l124_124225

theorem largest_constant_inequality :
  ∃ C, (∀ x y z : ℝ, x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧ (C = Real.sqrt 2) :=
sorry

end largest_constant_inequality_l124_124225


namespace max_students_seated_l124_124594

-- Define the number of seats in the i-th row
def seats_in_row (i : ℕ) : ℕ := 10 + 2 * i

-- Define the maximum number of students that can be seated in the i-th row
def max_students_in_row (i : ℕ) : ℕ := (seats_in_row i + 1) / 2

-- Sum the maximum number of students for all 25 rows
def total_max_students : ℕ := (Finset.range 25).sum max_students_in_row

-- The theorem statement
theorem max_students_seated : total_max_students = 450 := by
  sorry

end max_students_seated_l124_124594


namespace pushkin_family_pension_l124_124608

def is_survivor_pension (pension : String) (main_provider_deceased : Bool) (provision_lifelong : Bool) (assigned_to_family : Bool) : Prop :=
  pension = "survivor's pension" ↔
    main_provider_deceased = true ∧
    provision_lifelong = true ∧
    assigned_to_family = true

theorem pushkin_family_pension :
  ∀ (pension : String),
    let main_provider_deceased := true
    let provision_lifelong := true
    let assigned_to_family := true
    is_survivor_pension pension main_provider_deceased provision_lifelong assigned_to_family →
    pension = "survivor's pension" :=
by
  intros pension
  intro h
  sorry

end pushkin_family_pension_l124_124608


namespace find_highest_score_l124_124355

theorem find_highest_score (average innings : ℕ) (avg_excl_two innings_excl_two H L : ℕ)
  (diff_high_low total_runs total_excl_two : ℕ)
  (h1 : diff_high_low = 150)
  (h2 : total_runs = average * innings)
  (h3 : total_excl_two = avg_excl_two * innings_excl_two)
  (h4 : total_runs - total_excl_two = H + L)
  (h5 : H - L = diff_high_low)
  (h6 : average = 62)
  (h7 : innings = 46)
  (h8 : avg_excl_two = 58)
  (h9 : innings_excl_two = 44)
  (h10 : total_runs = 2844)
  (h11 : total_excl_two = 2552) :
  H = 221 :=
by
  sorry

end find_highest_score_l124_124355


namespace match_graph_l124_124296

theorem match_graph (x : ℝ) (h : x ≤ 0) : 
  Real.sqrt (-2 * x^3) = -x * Real.sqrt (-2 * x) :=
by
  sorry

end match_graph_l124_124296


namespace height_difference_percentage_l124_124255

theorem height_difference_percentage (q p : ℝ) (h : p = 0.6 * q) : (q - p) / p * 100 = 66.67 := 
by
  sorry

end height_difference_percentage_l124_124255


namespace pencil_cost_is_11_l124_124854

-- Define the initial and remaining amounts
def initial_amount : ℤ := 15
def remaining_amount : ℤ := 4

-- Define the cost of the pencil
def cost_of_pencil : ℤ := initial_amount - remaining_amount

-- The statement we need to prove
theorem pencil_cost_is_11 : cost_of_pencil = 11 :=
by
  sorry

end pencil_cost_is_11_l124_124854


namespace sale_in_fifth_month_l124_124230

theorem sale_in_fifth_month 
    (a1 a2 a3 a4 a6 : ℕ) 
    (avg_sale : ℕ)
    (H_avg : avg_sale = 8500)
    (H_a1 : a1 = 8435) 
    (H_a2 : a2 = 8927) 
    (H_a3 : a3 = 8855) 
    (H_a4 : a4 = 9230) 
    (H_a6 : a6 = 6991) : 
    ∃ a5 : ℕ, (a1 + a2 + a3 + a4 + a5 + a6) / 6 = avg_sale ∧ a5 = 8562 := 
by
    sorry

end sale_in_fifth_month_l124_124230


namespace larger_of_two_numbers_l124_124107

theorem larger_of_two_numbers
  (A B hcf : ℕ)
  (factor1 factor2 : ℕ)
  (h_hcf : hcf = 23)
  (h_factor1 : factor1 = 9)
  (h_factor2 : factor2 = 10)
  (h_lcm : (A * B) / (hcf) = (hcf * factor1 * factor2))
  (h_A : A = hcf * 9)
  (h_B : B = hcf * 10) :
  max A B = 230 := by
  sorry

end larger_of_two_numbers_l124_124107


namespace similar_polygon_area_sum_l124_124763

theorem similar_polygon_area_sum 
  (t1 t2 a1 a2 b : ℝ)
  (h_ratio: t1 / t2 = a1^2 / a2^2)
  (t3 : ℝ := t1 + t2)
  (h_area_eq : t3 = b^2 * a1^2 / a2^2): 
  b = Real.sqrt (a1^2 + a2^2) :=
by
  sorry

end similar_polygon_area_sum_l124_124763


namespace triangle_inequality_l124_124651

theorem triangle_inequality (a b c : ℝ) (h : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a * b * c ≥ (-a + b + c) * (a - b + c) * (a + b - c) :=
sorry

end triangle_inequality_l124_124651


namespace each_person_has_5_bags_l124_124340

def people := 6
def weight_per_bag := 50
def max_plane_weight := 6000
def additional_capacity := 90

theorem each_person_has_5_bags :
  (max_plane_weight / weight_per_bag - additional_capacity) / people = 5 :=
by
  sorry

end each_person_has_5_bags_l124_124340


namespace rational_coefficients_count_l124_124154

theorem rational_coefficients_count : 
  ∃ n, n = 84 ∧ ∀ k, (0 ≤ k ∧ k ≤ 500) → 
            (k % 3 = 0 ∧ (500 - k) % 2 = 0) → 
            n = 84 :=
by
  sorry

end rational_coefficients_count_l124_124154


namespace sum_of_cubes_of_three_consecutive_integers_l124_124881

theorem sum_of_cubes_of_three_consecutive_integers (a : ℕ) (h : (a * a) + (a + 1) * (a + 1) + (a + 2) * (a + 2) = 2450) : a * a * a + (a + 1) * (a + 1) * (a + 1) + (a + 2) * (a + 2) * (a + 2) = 73341 :=
by
  sorry

end sum_of_cubes_of_three_consecutive_integers_l124_124881


namespace max_temp_range_l124_124121

theorem max_temp_range (avg_temp : ℝ) (lowest_temp : ℝ) (days : ℕ) (total_temp : ℝ) (range : ℝ) : 
  avg_temp = 45 → 
  lowest_temp = 42 → 
  days = 5 → 
  total_temp = avg_temp * days → 
  range = 6 := 
by 
  sorry

end max_temp_range_l124_124121


namespace ratio_problem_l124_124337

theorem ratio_problem (A B C : ℚ) (h : A / B = 3 / 2) (h' : B / C = 2 / 5) : (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := 
by
  sorry

end ratio_problem_l124_124337


namespace problem_solution_l124_124302

def is_quadratic (y : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), ∀ x, y x = a * x^2 + b * x + c

def not_quadratic_func := 
  let yA := fun x => -2 * x^2
  let yB := fun x => 2 * (x - 1)^2 + 1
  let yC := fun x => (x - 3)^2 - x^2
  let yD := fun a => a * (8 - a)
  (¬ is_quadratic yC) ∧ (is_quadratic yA) ∧ (is_quadratic yB) ∧ (is_quadratic yD)

theorem problem_solution : not_quadratic_func := 
sorry

end problem_solution_l124_124302


namespace range_of_m_l124_124908

variable {x m : ℝ}

-- Definition of the first condition: ∀ x in ℝ, |x| + |x - 1| > m
def condition1 (m : ℝ) := ∀ x : ℝ, |x| + |x - 1| > m

-- Definition of the second condition: ∀ x in ℝ, (-(7 - 3 * m))^x is decreasing
def condition2 (m : ℝ) := ∀ x : ℝ, (-(7 - 3 * m))^x > (-(7 - 3 * m))^(x + 1)

-- Main theorem to prove m < 1
theorem range_of_m (h1 : condition1 m) (h2 : condition2 m) : m < 1 :=
sorry

end range_of_m_l124_124908


namespace ara_height_l124_124060

/-
Conditions:
1. Shea's height increased by 25%.
2. Shea is now 65 inches tall.
3. Ara grew by three-quarters as many inches as Shea did.

Prove Ara's height is 61.75 inches.
-/

def shea_original_height (x : ℝ) : Prop := 1.25 * x = 65

def ara_growth (growth : ℝ) (shea_growth : ℝ) : Prop := growth = (3 / 4) * shea_growth

def shea_growth (original_height : ℝ) : ℝ := 0.25 * original_height

theorem ara_height (shea_orig_height : ℝ) (shea_now_height : ℝ) (ara_growth_inches : ℝ) :
  shea_original_height shea_orig_height → 
  shea_now_height = 65 →
  ara_growth ara_growth_inches (shea_now_height - shea_orig_height) →
  shea_orig_height + ara_growth_inches = 61.75 :=
by
  sorry

end ara_height_l124_124060


namespace islander_parity_l124_124733

-- Define the concept of knights and liars
def is_knight (x : ℕ) : Prop := x % 2 = 0 -- Knight count is even
def is_liar (x : ℕ) : Prop := ¬(x % 2 = 1) -- Liar count being odd is false, so even

-- Define the total inhabitants on the island and conditions
theorem islander_parity (K L : ℕ) (h₁ : is_knight K) (h₂ : is_liar L) (h₃ : K + L = 2021) : false := sorry

end islander_parity_l124_124733


namespace exists_multiple_with_equal_digit_sum_l124_124208

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem exists_multiple_with_equal_digit_sum (k : ℕ) (h : k > 0) : 
  ∃ n : ℕ, (n % k = 0) ∧ (sum_of_digits n = sum_of_digits (n * n)) :=
sorry

end exists_multiple_with_equal_digit_sum_l124_124208


namespace ratio_of_parallel_vectors_l124_124648

theorem ratio_of_parallel_vectors (m n : ℝ) 
  (h1 : ∃ k : ℝ, (m, 1, 3) = (k * 2, k * n, k)) : (m / n) = 18 :=
by
  sorry

end ratio_of_parallel_vectors_l124_124648


namespace time_spent_per_egg_in_seconds_l124_124840

-- Definitions based on the conditions in the problem
def minutes_per_roll : ℕ := 30
def number_of_rolls : ℕ := 7
def total_cleaning_time : ℕ := 225
def number_of_eggs : ℕ := 60

-- Problem statement
theorem time_spent_per_egg_in_seconds :
  (total_cleaning_time - number_of_rolls * minutes_per_roll) * 60 / number_of_eggs = 15 := by
  sorry

end time_spent_per_egg_in_seconds_l124_124840


namespace find_natural_numbers_l124_124387

theorem find_natural_numbers (n : ℕ) (h : n > 1) : 
  ((n - 1) ∣ (n^3 - 3)) ↔ (n = 2 ∨ n = 3) := 
by 
  sorry

end find_natural_numbers_l124_124387


namespace find_smallest_k_l124_124280

variable (k : ℕ)

theorem find_smallest_k :
  (∀ a : ℝ, 0 ≤ a ∧ a ≤ 1 → (∀ n : ℕ, n > 0 → a^k * (1-a)^n < 1 / (n+1)^3)) ↔ k = 4 :=
sorry

end find_smallest_k_l124_124280


namespace S2016_value_l124_124556

theorem S2016_value (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h1 : a 1 = -2016)
  (h2 : ∀ n, S (n+1) = S n + a (n+1))
  (h3 : ∀ n, a (n+1) = a n + d)
  (h4 : (S 2015) / 2015 - (S 2012) / 2012 = 3) : S 2016 = -2016 := 
sorry

end S2016_value_l124_124556


namespace expression_is_integer_if_k_eq_2_l124_124567

def binom (n k : ℕ) := n.factorial / (k.factorial * (n-k).factorial)

theorem expression_is_integer_if_k_eq_2 
  (n k : ℕ) (h1 : 1 ≤ k) (h2 : k < n) (h3 : k = 2) : 
  ∃ (m : ℕ), m = (n - 3 * k + 2) * binom n k / (k + 2) := sorry

end expression_is_integer_if_k_eq_2_l124_124567


namespace number_of_solutions_l124_124312

theorem number_of_solutions (x : ℤ) (h1 : 0 < x) (h2 : x < 150) (h3 : (x + 17) % 46 = 75 % 46) : 
  ∃ n : ℕ, n = 3 :=
sorry

end number_of_solutions_l124_124312


namespace skyscraper_anniversary_l124_124269

theorem skyscraper_anniversary (built_years_ago : ℕ) (anniversary_years : ℕ) (years_before : ℕ) :
    built_years_ago = 100 → anniversary_years = 200 → years_before = 5 → 
    (anniversary_years - years_before) - built_years_ago = 95 := by
  intros h1 h2 h3
  sorry

end skyscraper_anniversary_l124_124269


namespace intersection_in_second_quadrant_l124_124055

theorem intersection_in_second_quadrant (k : ℝ) (x y : ℝ) 
  (hk : 0 < k) (hk2 : k < 1/2) 
  (h1 : k * x - y = k - 1) 
  (h2 : k * y - x = 2 * k) : 
  x < 0 ∧ y > 0 := 
sorry

end intersection_in_second_quadrant_l124_124055


namespace functional_eq_solution_l124_124726

theorem functional_eq_solution (f : ℝ → ℝ) : 
  (∀ x y : ℝ, f (x + f (x + y)) + f (x * y) = x + f (x + y) + y * f x) →
  (∀ x : ℝ, f x = x) :=
by
  intro h
  sorry

end functional_eq_solution_l124_124726


namespace expression_in_terms_of_p_q_l124_124474

-- Define the roots and the polynomials conditions
variable (α β γ δ : ℝ)
variable (p q : ℝ)

-- The conditions of the problem
axiom roots_poly1 : α * β = 1 ∧ α + β = -p
axiom roots_poly2 : γ * δ = 1 ∧ γ + δ = -q

theorem expression_in_terms_of_p_q :
  (α - γ) * (β - γ) * (α + δ) * (β + δ) = q^2 - p^2 :=
sorry

end expression_in_terms_of_p_q_l124_124474


namespace inequality_solution_l124_124834

theorem inequality_solution (x : ℝ) :
  (x * (x + 2) > x * (3 - x) + 1) ↔ (x < -1/2 ∨ x > 1) :=
by sorry

end inequality_solution_l124_124834


namespace friday_can_determine_arrival_date_l124_124694

-- Define the conditions
def Robinson_crusoe (day : ℕ) : Prop := day % 365 = 0

-- Goal: Within 183 days, Friday can determine his arrival date.
theorem friday_can_determine_arrival_date : 
  (∀ day : ℕ, day < 183 → (Robinson_crusoe day ↔ ¬ Robinson_crusoe (day + 1)) ∨ (day % 365 = 0)) :=
sorry

end friday_can_determine_arrival_date_l124_124694


namespace condition_suff_and_nec_l124_124907

def p (x : ℝ) : Prop := |x + 2| ≤ 3
def q (x : ℝ) : Prop := x < -8

theorem condition_suff_and_nec (x : ℝ) : p x ↔ ¬ q x :=
by
  sorry

end condition_suff_and_nec_l124_124907


namespace find_value_of_y_l124_124137

theorem find_value_of_y (x y : ℕ) 
    (h1 : 2^x - 2^y = 3 * 2^12) 
    (h2 : x = 14) : 
    y = 13 := 
by
  sorry

end find_value_of_y_l124_124137


namespace tree_height_fraction_l124_124639

theorem tree_height_fraction :
  ∀ (initial_height growth_per_year : ℝ),
  initial_height = 4 ∧ growth_per_year = 0.5 →
  ((initial_height + 6 * growth_per_year) - (initial_height + 4 * growth_per_year)) / (initial_height + 4 * growth_per_year) = 1 / 6 :=
by
  intros initial_height growth_per_year h
  rcases h with ⟨h1, h2⟩
  sorry

end tree_height_fraction_l124_124639


namespace find_b_in_quadratic_eqn_l124_124142

theorem find_b_in_quadratic_eqn :
  ∃ (b : ℝ), ∃ (p : ℝ), 
  (∀ x, x^2 + b*x + 64 = (x + p)^2 + 16) → 
  b = 8 * Real.sqrt 3 :=
by 
  sorry

end find_b_in_quadratic_eqn_l124_124142


namespace polygon_sides_l124_124479

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 > 2970) :
  n = 19 :=
by
  sorry

end polygon_sides_l124_124479


namespace product_of_areas_eq_k3_times_square_of_volume_l124_124962

variables (a b c k : ℝ)

-- Defining the areas of bottom, side, and front of the box as provided
def area_bottom := k * a * b
def area_side := k * b * c
def area_front := k * c * a

-- Volume of the box
def volume := a * b * c

-- The lean statement to be proved
theorem product_of_areas_eq_k3_times_square_of_volume :
  (area_bottom a b k) * (area_side b c k) * (area_front c a k) = k^3 * (volume a b c)^2 :=
by
  sorry

end product_of_areas_eq_k3_times_square_of_volume_l124_124962


namespace distance_traveled_by_bus_l124_124969

noncomputable def total_distance : ℕ := 900
noncomputable def distance_by_plane : ℕ := total_distance / 3
noncomputable def distance_by_bus : ℕ := 360
noncomputable def distance_by_train : ℕ := (2 * distance_by_bus) / 3

theorem distance_traveled_by_bus :
  distance_by_plane + distance_by_train + distance_by_bus = total_distance :=
by
  sorry

end distance_traveled_by_bus_l124_124969


namespace train_cross_bridge_time_l124_124301

noncomputable def length_train : ℝ := 130
noncomputable def length_bridge : ℝ := 320
noncomputable def speed_kmh : ℝ := 54
noncomputable def speed_ms : ℝ := speed_kmh * 1000 / 3600

theorem train_cross_bridge_time :
  (length_train + length_bridge) / speed_ms = 30 := by
  sorry

end train_cross_bridge_time_l124_124301


namespace inequality_proof_l124_124002

theorem inequality_proof (x y z : ℝ) : 
  x^2 + 2 * y^2 + 3 * z^2 ≥ Real.sqrt 3 * (x * y + y * z + z * x) := 
  sorry

end inequality_proof_l124_124002


namespace profit_percentage_l124_124909

theorem profit_percentage (SP CP : ℝ) (H_SP : SP = 1800) (H_CP : CP = 1500) :
  ((SP - CP) / CP) * 100 = 20 :=
by
  sorry

end profit_percentage_l124_124909


namespace system_of_equations_proof_l124_124943

theorem system_of_equations_proof (a b x A B C : ℝ) (h1: a ≠ 0) 
  (h2: a * Real.sin x + b * Real.cos x = 0) 
  (h3: A * Real.sin (2 * x) + B * Real.cos (2 * x) = C) : 
  2 * a * b * A + (b ^ 2 - a ^ 2) * B + (a ^ 2 + b ^ 2) * C = 0 := 
sorry

end system_of_equations_proof_l124_124943


namespace find_term_number_l124_124960

-- Define the arithmetic sequence
def arithmetic_seq (a d : Int) (n : Int) := a + (n - 1) * d

-- Define the condition: first term and common difference
def a1 := 4
def d := 3

-- Prove that the 672nd term is 2017
theorem find_term_number (n : Int) (h : arithmetic_seq a1 d n = 2017) : n = 672 := by
  sorry

end find_term_number_l124_124960


namespace gcd_1821_2993_l124_124660

theorem gcd_1821_2993 : Nat.gcd 1821 2993 = 1 := 
by 
  sorry

end gcd_1821_2993_l124_124660


namespace factor_tree_X_value_l124_124739

theorem factor_tree_X_value :
  let F := 2 * 5
  let G := 7 * 3
  let Y := 7 * F
  let Z := 11 * G
  let X := Y * Z
  X = 16170 := by
sorry

end factor_tree_X_value_l124_124739


namespace present_age_of_father_l124_124972

-- Definitions based on the conditions
variables (F S : ℕ)
axiom cond1 : F = 3 * S + 3
axiom cond2 : F + 3 = 2 * (S + 3) + 8

-- The theorem to prove
theorem present_age_of_father : F = 27 :=
by
  sorry

end present_age_of_father_l124_124972


namespace geometric_sequence_sum_5_l124_124683

theorem geometric_sequence_sum_5 
  (a : ℕ → ℝ) 
  (h_geom : ∃ q, ∀ n, a (n + 1) = a n * q) 
  (h_a2 : a 2 = 2) 
  (h_a3 : a 3 = 4) : 
  (a 1 * (1 - (2:ℝ)^5) / (1 - (2:ℝ))) = 31 := 
by
  sorry

end geometric_sequence_sum_5_l124_124683


namespace max_s_value_l124_124532

theorem max_s_value (r s : ℕ) (hr : r ≥ s) (hs : s ≥ 3)
  (h : ((r - 2) * 180 / r : ℚ) / ((s - 2) * 180 / s) = 60 / 59) :
  s = 117 :=
by
  sorry

end max_s_value_l124_124532


namespace symmetry_sum_zero_l124_124944

theorem symmetry_sum_zero (v : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, v (-x) = -v x) : 
  v (-2.00) + v (-1.00) + v (1.00) + v (2.00) = 0 := 
by 
  sorry

end symmetry_sum_zero_l124_124944


namespace divisible_by_other_l124_124744

theorem divisible_by_other (y : ℕ) 
  (h1 : y = 20)
  (h2 : y % 4 = 0)
  (h3 : y % 8 ≠ 0) : (∃ n, n ≠ 4 ∧ y % n = 0 ∧ n = 5) :=
by 
  sorry

end divisible_by_other_l124_124744


namespace sum_of_three_consecutive_integers_product_990_l124_124846

theorem sum_of_three_consecutive_integers_product_990 
  (a b c : ℕ) 
  (h1 : b = a + 1)
  (h2 : c = b + 1)
  (h3 : a * b * c = 990) :
  a + b + c = 30 :=
sorry

end sum_of_three_consecutive_integers_product_990_l124_124846


namespace triangle_obtuse_l124_124557

theorem triangle_obtuse
  (A B : ℝ) 
  (hA : 0 < A ∧ A < π / 2)
  (hB : 0 < B ∧ B < π / 2)
  (h : Real.cos A > Real.sin B) : 
  π / 2 < π - (A + B) ∧ π - (A + B) < π :=
by
  sorry

end triangle_obtuse_l124_124557


namespace participants_who_drank_neither_l124_124373

-- Conditions
variables (total_participants : ℕ) (coffee_drinkers : ℕ) (juice_drinkers : ℕ) (both_drinkers : ℕ)

-- Initial Facts from the Conditions
def conditions := total_participants = 30 ∧ coffee_drinkers = 15 ∧ juice_drinkers = 18 ∧ both_drinkers = 7

-- The statement to prove
theorem participants_who_drank_neither : conditions total_participants coffee_drinkers juice_drinkers both_drinkers → 
  (total_participants - (coffee_drinkers + juice_drinkers - both_drinkers)) = 4 :=
by
  intros
  sorry

end participants_who_drank_neither_l124_124373


namespace c_minus_a_is_10_l124_124870

variable (a b c d k : ℝ)

theorem c_minus_a_is_10 (h1 : a + b = 90)
                        (h2 : b + c = 100)
                        (h3 : a + c + d = 180)
                        (h4 : a^2 + b^2 + c^2 + d^2 = k) :
  c - a = 10 :=
by sorry

end c_minus_a_is_10_l124_124870


namespace percent_of_amount_l124_124042

theorem percent_of_amount (Part Whole : ℝ) (hPart : Part = 120) (hWhole : Whole = 80) :
  (Part / Whole) * 100 = 150 :=
by
  rw [hPart, hWhole]
  sorry

end percent_of_amount_l124_124042


namespace range_of_x_l124_124209

variable (x y : ℝ)

def op (x y : ℝ) := x * (1 - y)

theorem range_of_x (h : op (x - 1) (x + 2) < 0) : x < -1 ∨ 1 < x :=
by
  dsimp [op] at h
  sorry

end range_of_x_l124_124209


namespace perfect_square_trinomial_l124_124274

theorem perfect_square_trinomial (m : ℝ) :
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 - 2*(m-3)*x + 16 = (x - a)^2) ↔ (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_trinomial_l124_124274


namespace quadractic_inequality_solution_l124_124787

theorem quadractic_inequality_solution (a b : ℝ) (h₁ : ∀ x : ℝ, -4 ≤ x ∧ x ≤ 3 → x^2 - (a+1) * x + b ≤ 0) : a + b = -14 :=
by 
  -- Proof construction is omitted
  sorry

end quadractic_inequality_solution_l124_124787


namespace b_squared_gt_4ac_l124_124402

theorem b_squared_gt_4ac (a b c : ℝ) (h : (a + b + c) * c < 0) : b^2 > 4 * a * c :=
by
  sorry

end b_squared_gt_4ac_l124_124402


namespace pencil_total_length_l124_124825

-- Definitions of the colored sections
def purple_length : ℝ := 3.5
def black_length : ℝ := 2.8
def blue_length : ℝ := 1.6
def green_length : ℝ := 0.9
def yellow_length : ℝ := 1.2

-- The theorem stating the total length of the pencil
theorem pencil_total_length : purple_length + black_length + blue_length + green_length + yellow_length = 10 := 
by
  sorry

end pencil_total_length_l124_124825


namespace person_speed_l124_124067

theorem person_speed (distance_m : ℝ) (time_min : ℝ) (h₁ : distance_m = 800) (h₂ : time_min = 5) : 
  let distance_km := distance_m / 1000
  let time_hr := time_min / 60
  distance_km / time_hr = 9.6 := 
by
  sorry

end person_speed_l124_124067


namespace mans_rate_in_still_water_l124_124245

theorem mans_rate_in_still_water (Vm Vs : ℝ) (h1 : Vm + Vs = 14) (h2 : Vm - Vs = 4) : Vm = 9 :=
by
  sorry

end mans_rate_in_still_water_l124_124245


namespace ellipse_foci_distance_2sqrt21_l124_124829

noncomputable def ellipse_foci_distance (a b : ℝ) : ℝ := 2 * Real.sqrt (a^2 - b^2)

theorem ellipse_foci_distance_2sqrt21 :
  let center : ℝ × ℝ := (5, 2)
  let a := 5
  let b := 2
  ellipse_foci_distance a b = 2 * Real.sqrt 21 :=
by
  sorry

end ellipse_foci_distance_2sqrt21_l124_124829


namespace simplify_expression_l124_124468

theorem simplify_expression :
  8 * (18 / 5) * (-40 / 27) = - (128 / 3) := 
by
  sorry

end simplify_expression_l124_124468


namespace no_x_satisfies_inequalities_l124_124475

theorem no_x_satisfies_inequalities : ¬ ∃ x : ℝ, 4 * x - 2 < (x + 2) ^ 2 ∧ (x + 2) ^ 2 < 9 * x - 5 :=
sorry

end no_x_satisfies_inequalities_l124_124475


namespace cookies_yesterday_l124_124698

theorem cookies_yesterday (cookies_today : ℕ) (difference : ℕ)
  (h1 : cookies_today = 140)
  (h2 : difference = 30) :
  cookies_today - difference = 110 :=
by
  sorry

end cookies_yesterday_l124_124698


namespace solution_set_of_inequalities_l124_124236

theorem solution_set_of_inequalities (x : ℝ) :
  (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end solution_set_of_inequalities_l124_124236


namespace power_equality_l124_124536

theorem power_equality : 
  ( (11 : ℝ) ^ (1 / 5) / (11 : ℝ) ^ (1 / 7) ) = (11 : ℝ) ^ (2 / 35) := 
by sorry

end power_equality_l124_124536


namespace least_subtracted_divisible_l124_124623

theorem least_subtracted_divisible :
  ∃ k, (5264 - 11) = 17 * k :=
by
  sorry

end least_subtracted_divisible_l124_124623


namespace greatest_x_value_l124_124140

noncomputable def greatest_possible_value (x : ℕ) : ℕ :=
  if (x % 5 = 0) ∧ (x^3 < 3375) then x else 0

theorem greatest_x_value :
  ∃ x, greatest_possible_value x = 10 ∧ (∀ y, ((y % 5 = 0) ∧ (y^3 < 3375)) → y ≤ x) :=
by
  sorry

end greatest_x_value_l124_124140


namespace proof_problem_l124_124304

theorem proof_problem
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) :
  (x + 1 / y ≥ 2) ∨ (y + 1 / z ≥ 2) ∨ (z + 1 / x ≥ 2) :=
sorry

end proof_problem_l124_124304


namespace no_real_solutions_l124_124250

theorem no_real_solutions :
  ¬ ∃ x : ℝ, (x - 3 * x + 8)^2 + 4 = -2 * |x| :=
by
  sorry

end no_real_solutions_l124_124250


namespace annika_total_kilometers_east_l124_124516

def annika_constant_rate : ℝ := 10 -- 10 minutes per kilometer
def distance_hiked_initially : ℝ := 2.5 -- 2.5 kilometers
def total_time_to_return : ℝ := 35 -- 35 minutes

theorem annika_total_kilometers_east :
  (total_time_to_return - (distance_hiked_initially * annika_constant_rate)) / annika_constant_rate + distance_hiked_initially = 3.5 := by
  sorry

end annika_total_kilometers_east_l124_124516


namespace percentage_of_whole_l124_124839

theorem percentage_of_whole (part whole percent : ℕ) (h1 : part = 120) (h2 : whole = 80) (h3 : percent = 150) : 
  part = (percent / 100) * whole :=
by
  sorry

end percentage_of_whole_l124_124839


namespace find_dimes_l124_124221

-- Definitions for the conditions
def total_dollars : ℕ := 13
def dollar_bills_1 : ℕ := 2
def dollar_bills_5 : ℕ := 1
def quarters : ℕ := 13
def nickels : ℕ := 8
def pennies : ℕ := 35
def value_dollar_bill_1 : ℝ := 1.0
def value_dollar_bill_5 : ℝ := 5.0
def value_quarter : ℝ := 0.25
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_dime : ℝ := 0.10

-- Theorem statement
theorem find_dimes (total_dollars dollar_bills_1 dollar_bills_5 quarters nickels pennies : ℕ)
  (value_dollar_bill_1 value_dollar_bill_5 value_quarter value_nickel value_penny value_dime : ℝ) :
  (2 * value_dollar_bill_1 + 1 * value_dollar_bill_5 + 13 * value_quarter + 8 * value_nickel + 35 * value_penny) + 
  (20 * value_dime) = ↑total_dollars :=
sorry

end find_dimes_l124_124221


namespace find_x_satisfying_sinx_plus_cosx_eq_one_l124_124859

theorem find_x_satisfying_sinx_plus_cosx_eq_one :
  ∀ x, 0 ≤ x ∧ x < 2 * Real.pi → (Real.sin x + Real.cos x = 1 ↔ x = 0) := by
  sorry

end find_x_satisfying_sinx_plus_cosx_eq_one_l124_124859


namespace sqrt_inequality_l124_124519

theorem sqrt_inequality (a b c : ℝ) (ha : 1 < a) (hb : 1 < b) (hc : 1 < c) (habc : a + b + c = 9) : 
  Real.sqrt (a * b + b * c + c * a) ≤ Real.sqrt a + Real.sqrt b + Real.sqrt c := 
sorry

end sqrt_inequality_l124_124519


namespace daily_savings_in_dollars_l124_124481

-- Define the total savings and the number of days
def total_savings_in_dimes : ℕ := 3
def number_of_days : ℕ := 30

-- Define the conversion factor from dimes to dollars
def dime_to_dollar : ℝ := 0.10

-- Prove that the daily savings in dollars is $0.01
theorem daily_savings_in_dollars : total_savings_in_dimes / number_of_days * dime_to_dollar = 0.01 :=
by sorry

end daily_savings_in_dollars_l124_124481


namespace mike_max_marks_l124_124751

theorem mike_max_marks
  (M : ℝ)
  (h1 : 0.30 * M = 234)
  (h2 : 234 = 212 + 22) : M = 780 := 
sorry

end mike_max_marks_l124_124751


namespace simplify_expression_l124_124925

theorem simplify_expression (y : ℝ) : y - 3 * (2 + y) + 4 * (2 - y) - 5 * (2 + 3 * y) = -21 * y - 8 :=
by
  sorry

end simplify_expression_l124_124925


namespace value_of_x_and_z_l124_124396

theorem value_of_x_and_z (x y z : ℤ) (h1 : x / y = 7 / 3) (h2 : y = 21) (h3 : z = 3 * y) : x = 49 ∧ z = 63 :=
by
  sorry

end value_of_x_and_z_l124_124396


namespace initial_oranges_l124_124109

open Nat

theorem initial_oranges (initial_oranges: ℕ) (eaten_oranges: ℕ) (stolen_oranges: ℕ) (returned_oranges: ℕ) (current_oranges: ℕ):
  eaten_oranges = 10 → 
  stolen_oranges = (initial_oranges - eaten_oranges) / 2 →
  returned_oranges = 5 →
  current_oranges = 30 →
  initial_oranges - eaten_oranges - stolen_oranges + returned_oranges = current_oranges →
  initial_oranges = 60 :=
by
  sorry

end initial_oranges_l124_124109


namespace higher_room_amount_higher_60_l124_124072

variable (higher_amount : ℕ)

theorem higher_room_amount_higher_60 
  (total_rent : ℕ) (amount_credited_50 : ℕ)
  (total_reduction : ℕ)
  (condition1 : total_rent = 400)
  (condition2 : amount_credited_50 = 50)
  (condition3 : total_reduction = total_rent / 4)
  (condition4 : 10 * higher_amount - 10 * amount_credited_50 = total_reduction) :
  higher_amount = 60 := 
sorry

end higher_room_amount_higher_60_l124_124072


namespace proof_problem_l124_124177

-- Conditions
def op1 := (15 + 3) / (8 - 2) = 3
def op2 := (9 + 4) / (14 - 7)

-- Statement
theorem proof_problem : op1 → op2 = 13 / 7 :=
by 
  intro h
  unfold op2
  sorry

end proof_problem_l124_124177


namespace expression_undefined_at_x_l124_124927

theorem expression_undefined_at_x (x : ℝ) : (x^2 - 18 * x + 81 = 0) → x = 9 :=
by {
  sorry
}

end expression_undefined_at_x_l124_124927


namespace larger_number_l124_124176

theorem larger_number (x y : ℤ) (h1 : x + y = 40) (h2 : x - y = 6) : x = 23 :=
by {
  sorry
}

end larger_number_l124_124176


namespace solve_equation_l124_124579

open Real

theorem solve_equation (t : ℝ) :
  ¬cos t = 0 ∧ ¬cos (2 * t) = 0 → 
  (tan (2 * t) / (cos t)^2 - tan t / (cos (2 * t))^2 = 0 ↔ 
    (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = π * ↑n + π / 6) ∨ (∃ n : ℤ, t = π * ↑n - π / 6)) :=
by
  intros h
  sorry

end solve_equation_l124_124579


namespace determine_N_l124_124388

theorem determine_N (N : ℕ) : (Nat.choose N 5 = 3003) ↔ (N = 15) :=
by
  sorry

end determine_N_l124_124388


namespace cuboid_surface_area_4_8_6_l124_124710

noncomputable def cuboid_surface_area (length width height : ℕ) : ℕ :=
  2 * (length * width + length * height + width * height)

theorem cuboid_surface_area_4_8_6 : cuboid_surface_area 4 8 6 = 208 := by
  sorry

end cuboid_surface_area_4_8_6_l124_124710


namespace painting_methods_correct_l124_124056

noncomputable def num_painting_methods : ℕ :=
  sorry 

theorem painting_methods_correct :
  num_painting_methods = 24 :=
by
  -- proof would go here
  sorry

end painting_methods_correct_l124_124056


namespace final_temperature_correct_l124_124634

-- Define the initial conditions
def initial_temperature : ℝ := 12
def decrease_per_hour : ℝ := 5
def time_duration : ℕ := 4

-- Define the expected final temperature
def expected_final_temperature : ℝ := -8

-- The theorem to prove that the final temperature after a given time is as expected
theorem final_temperature_correct :
  initial_temperature + (-decrease_per_hour * time_duration) = expected_final_temperature :=
by
  sorry

end final_temperature_correct_l124_124634


namespace race_distance_l124_124906

theorem race_distance (dA dB dC : ℝ) (h1 : dA = 1000) (h2 : dB = 900) (h3 : dB = 800) (h4 : dC = 700) (d : ℝ) (h5 : d = dA + 127.5) :
  d = 600 :=
sorry

end race_distance_l124_124906


namespace percentage_passed_in_both_l124_124828

def percentage_of_students_failing_hindi : ℝ := 30
def percentage_of_students_failing_english : ℝ := 42
def percentage_of_students_failing_both : ℝ := 28

theorem percentage_passed_in_both (P_H_E: percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both = 44) : 
  100 - (percentage_of_students_failing_hindi + percentage_of_students_failing_english - percentage_of_students_failing_both) = 56 := by
  sorry

end percentage_passed_in_both_l124_124828


namespace gcd_2873_1349_gcd_4562_275_l124_124001

theorem gcd_2873_1349 : Nat.gcd 2873 1349 = 1 := 
sorry

theorem gcd_4562_275 : Nat.gcd 4562 275 = 1 := 
sorry

end gcd_2873_1349_gcd_4562_275_l124_124001


namespace Jack_Income_Ratio_l124_124606

noncomputable def Ernie_current_income (x : ℕ) : ℕ :=
  (4 / 5) * x

noncomputable def Jack_current_income (combined_income Ernie_current_income : ℕ) : ℕ :=
  combined_income - Ernie_current_income

theorem Jack_Income_Ratio (Ernie_previous_income combined_income : ℕ) (h₁ : Ernie_previous_income = 6000) (h₂ : combined_income = 16800) :
  let Ernie_current := Ernie_current_income Ernie_previous_income
  let Jack_current := Jack_current_income combined_income Ernie_current
  (Jack_current / Ernie_previous_income) = 2 := by
  sorry

end Jack_Income_Ratio_l124_124606


namespace ratio_age_difference_to_pencils_l124_124115

-- Definitions of the given problem conditions
def AsafAge : ℕ := 50
def SumOfAges : ℕ := 140
def AlexanderAge : ℕ := SumOfAges - AsafAge

def PencilDifference : ℕ := 60
def TotalPencils : ℕ := 220
def AsafPencils : ℕ := (TotalPencils - PencilDifference) / 2
def AlexanderPencils : ℕ := AsafPencils + PencilDifference

-- Define the age difference and the ratio
def AgeDifference : ℕ := AlexanderAge - AsafAge
def Ratio : ℚ := AgeDifference / AsafPencils

theorem ratio_age_difference_to_pencils : Ratio = 1 / 2 := by
  sorry

end ratio_age_difference_to_pencils_l124_124115


namespace sum_a1_to_a12_l124_124566

variable {a : ℕ → ℕ}

axiom geom_seq (n : ℕ) : a n * a (n + 1) * a (n + 2) = 8
axiom a_1 : a 1 = 1
axiom a_2 : a 2 = 2

theorem sum_a1_to_a12 : (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 + a 11 + a 12) = 28 :=
by
  sorry

end sum_a1_to_a12_l124_124566


namespace vasya_fraction_l124_124958

variable (a b c d s : ℝ)

-- Anton drove half the distance Vasya did
axiom h1 : a = b / 2

-- Sasha drove as long as Anton and Dima together
axiom h2 : c = a + d

-- Dima drove one-tenth of the total distance
axiom h3 : d = s / 10

-- The total distance is the sum of distances driven by Anton, Vasya, Sasha, and Dima
axiom h4 : a + b + c + d = s

-- We need to prove that Vasya drove 0.4 of the total distance
theorem vasya_fraction (a b c d s : ℝ) (h1 : a = b / 2) (h2 : c = a + d) (h3 : d = s / 10) (h4 : a + b + c + d = s) : b = 0.4 * s :=
by
  sorry

end vasya_fraction_l124_124958


namespace greg_experienced_less_rain_l124_124262

theorem greg_experienced_less_rain (rain_day1 rain_day2 rain_day3 rain_house : ℕ) 
  (h1 : rain_day1 = 3) 
  (h2 : rain_day2 = 6) 
  (h3 : rain_day3 = 5) 
  (h4 : rain_house = 26) :
  rain_house - (rain_day1 + rain_day2 + rain_day3) = 12 :=
by
  sorry

end greg_experienced_less_rain_l124_124262


namespace inequality_holds_l124_124233

theorem inequality_holds (a b c : ℝ) 
  (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_holds_l124_124233


namespace max_remainder_when_divided_by_7_l124_124384

theorem max_remainder_when_divided_by_7 (y : ℕ) (r : ℕ) (h : r = y % 7) : r ≤ 6 ∧ ∃ k, y = 7 * k + r :=
by
  sorry

end max_remainder_when_divided_by_7_l124_124384


namespace sequence_value_l124_124175

noncomputable def f : ℝ → ℝ := sorry

theorem sequence_value :
  ∃ a : ℕ → ℝ, 
    (a 1 = f 1) ∧ 
    (∀ n : ℕ, f (a (n + 1)) = f (2 * a n + 1)) ∧ 
    (a 2017 = 2 ^ 2016 - 1) := sorry

end sequence_value_l124_124175


namespace tom_mileage_per_gallon_l124_124382

-- Definitions based on the given conditions
def daily_mileage : ℕ := 75
def cost_per_gallon : ℕ := 3
def amount_spent_in_10_days : ℕ := 45
def days : ℕ := 10

-- Main theorem to prove
theorem tom_mileage_per_gallon : 
  (amount_spent_in_10_days / cost_per_gallon) * 75 * days = 50 :=
by
  sorry

end tom_mileage_per_gallon_l124_124382


namespace rectangular_prism_cut_l124_124381

theorem rectangular_prism_cut
  (x y : ℕ)
  (original_volume : ℕ := 15 * 5 * 4) 
  (remaining_volume : ℕ := 120) 
  (cut_out_volume_eq : original_volume - remaining_volume = 5 * x * y) 
  (x_condition : 1 < x) 
  (x_condition_2 : x < 4) 
  (y_condition : 1 < y) 
  (y_condition_2 : y < 15) : 
  x + y = 15 := 
sorry

end rectangular_prism_cut_l124_124381


namespace complex_number_location_in_plane_l124_124011

def is_in_second_quadrant (x y : ℝ) : Prop :=
  x < 0 ∧ y > 0

theorem complex_number_location_in_plane :
  is_in_second_quadrant (-2) 5 :=
by
  sorry

end complex_number_location_in_plane_l124_124011


namespace function_relationship_l124_124670

theorem function_relationship (f : ℝ → ℝ)
  (h₁ : ∀ x, f (x + 1) = f (-x + 1))
  (h₂ : ∀ x, x ≥ 1 → f x = (1 / 2) ^ x - 1) :
  f (2 / 3) > f (3 / 2) ∧ f (3 / 2) > f (1 / 3) :=
by sorry

end function_relationship_l124_124670


namespace surface_area_geometric_mean_volume_geometric_mean_l124_124946

noncomputable def surfaces_areas_proof (r : ℝ) (π : ℝ) : Prop :=
  let F_1 := 6 * π * r^2
  let F_2 := 4 * π * r^2
  let F_3 := 9 * π * r^2
  F_1^2 = F_2 * F_3

noncomputable def volumes_proof (r : ℝ) (π : ℝ) : Prop :=
  let V_1 := 2 * π * r^3
  let V_2 := (4 / 3) * π * r^3
  let V_3 := π * r^3
  V_1^2 = V_2 * V_3

theorem surface_area_geometric_mean (r : ℝ) (π : ℝ) : surfaces_areas_proof r π := 
  sorry

theorem volume_geometric_mean (r : ℝ) (π : ℝ) : volumes_proof r π :=
  sorry

end surface_area_geometric_mean_volume_geometric_mean_l124_124946


namespace trajectory_of_midpoint_l124_124232

theorem trajectory_of_midpoint (x y : ℝ) (A B : ℝ × ℝ) 
  (hB : B = (4, 0)) (hA_on_circle : (A.1)^2 + (A.2)^2 = 4)
  (hM : ((x, y) = ( (A.1 + B.1)/2, (A.2 + B.2)/2))) :
  (x - 2)^2 + y^2 = 1 :=
sorry

end trajectory_of_midpoint_l124_124232


namespace first_interest_rate_is_correct_l124_124135

theorem first_interest_rate_is_correct :
  let A1 := 1500.0000000000007
  let A2 := 2500 - A1
  let yearly_income := 135
  (15.0 * (r / 100) + 6.0 * (A2 / 100) = yearly_income) -> r = 5.000000000000003 :=
sorry

end first_interest_rate_is_correct_l124_124135


namespace find_third_root_l124_124618

variables (a b : ℝ)

def poly (x : ℝ) : ℝ := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

def root1 := -3
def root2 := 4

axiom root1_cond : poly a b root1 = 0
axiom root2_cond : poly a b root2 = 0

theorem find_third_root (a b : ℝ) (h1 : poly a b root1 = 0) (h2 : poly a b root2 = 0) : 
  ∃ r3 : ℝ, r3 = -1/2 :=
sorry

end find_third_root_l124_124618


namespace sides_of_triangle_l124_124151

variable (a b c : ℝ)

theorem sides_of_triangle (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
  (h_ineq : (a^2 + b^2 + c^2)^2 > 2*(a^4 + b^4 + c^4)) :
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) :=
  sorry

end sides_of_triangle_l124_124151


namespace acute_angle_of_parallelogram_l124_124775

theorem acute_angle_of_parallelogram
  (a b : ℝ) (h : a < b)
  (parallelogram_division : ∀ x y : ℝ, x + y = a → b = x + 2 * Real.sqrt (x * y) + y) :
  ∃ α : ℝ, α = Real.arcsin ((b / a) - 1) :=
sorry

end acute_angle_of_parallelogram_l124_124775


namespace solve_for_nabla_l124_124017

theorem solve_for_nabla (nabla : ℤ) (h : 5 * (-4) = nabla + 4) : nabla = -24 :=
by {
  sorry
}

end solve_for_nabla_l124_124017


namespace sequence_formula_l124_124367

theorem sequence_formula (a : ℕ → ℝ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n / (1 + 2 * a n)) :
  ∀ n, a n = 1 / (2 * n - 1) :=
by
  sorry

end sequence_formula_l124_124367


namespace problem_part1_problem_part2_l124_124032

noncomputable def quadratic_roots_conditions (x1 x2 m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1)

noncomputable def existence_of_m (x1 x2 : ℝ) (m : ℝ) : Prop :=
  (x1 = 1) ∧ (x1 + x2 = 6) ∧ (x1 * x2 = 2 * m - 1) ∧ ((x1 - 1) * (x2 - 1) = 6 / (m - 5))

theorem problem_part1 : 
  ∃ x2 m, quadratic_roots_conditions 1 x2 m :=
sorry

theorem problem_part2 :
  ∃ m, ∃ x2, existence_of_m 1 x2 m ∧ m ≤ 5 :=
sorry

end problem_part1_problem_part2_l124_124032


namespace number_of_articles_l124_124856

theorem number_of_articles (C S : ℝ) (N : ℝ) 
    (h1 : N * C = 40 * S) 
    (h2 : (S - C) / C * 100 = 49.999999999999986) : 
    N = 60 :=
sorry

end number_of_articles_l124_124856


namespace max_sum_of_three_integers_with_product_24_l124_124318

theorem max_sum_of_three_integers_with_product_24 : ∃ (a b c : ℤ), (a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a * b * c = 24 ∧ a + b + c = 15) :=
by
  sorry

end max_sum_of_three_integers_with_product_24_l124_124318


namespace min_sum_xy_l124_124563

theorem min_sum_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y + x * y = 3) : x + y ≥ 2 :=
by
  sorry

end min_sum_xy_l124_124563


namespace find_b_value_l124_124843

theorem find_b_value (b : ℚ) (x : ℚ) (h1 : 3 * x + 9 = 0) (h2 : b * x + 15 = 5) : b = 10 / 3 :=
by
  sorry

end find_b_value_l124_124843


namespace trajectory_of_point_l124_124246

theorem trajectory_of_point (x y : ℝ)
  (h1 : (x - 1)^2 + (y - 1)^2 = ((3 * x + y - 4)^2) / 10) :
  x - 3 * y + 2 = 0 :=
sorry

end trajectory_of_point_l124_124246


namespace milk_leftover_l124_124663

theorem milk_leftover 
  (total_milk : ℕ := 24)
  (kids_percent : ℝ := 0.80)
  (cooking_percent : ℝ := 0.60)
  (neighbor_percent : ℝ := 0.25)
  (husband_percent : ℝ := 0.06) :
  let milk_after_kids := total_milk * (1 - kids_percent)
  let milk_after_cooking := milk_after_kids * (1 - cooking_percent)
  let milk_after_neighbor := milk_after_cooking * (1 - neighbor_percent)
  let milk_after_husband := milk_after_neighbor * (1 - husband_percent)
  milk_after_husband = 1.3536 :=
by 
  -- skip the proof for simplicity
  sorry

end milk_leftover_l124_124663


namespace range_of_k_l124_124484

noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  ((k+1)*x^2 + (k+3)*x + (2*k-8)) / ((2*k-1)*x^2 + (k+1)*x + (k-4))

theorem range_of_k 
  (k : ℝ) 
  (hk1 : k ≠ -1)
  (hk2 : (k+3)^2 - 4*(k+1)*(2*k-8) ≥ 0)
  (hk3 : (k+1)^2 - 4*(2*k-1)*(k-4) ≤ 0)
  (hk4 : (k+1)/(2*k-1) > 0) :
  k ∈ Set.Iio (-1) ∪ Set.Ioi (1 / 2) ∩ Set.Iic (41 / 7) := 
  sorry

end range_of_k_l124_124484


namespace f_leq_zero_l124_124746

noncomputable def f (x a : ℝ) := x * Real.log x - a * x^2 + (2 * a - 1) * x

theorem f_leq_zero (a x : ℝ) (h1 : 1/2 < a) (h2 : a ≤ 1) (hx : 0 < x) :
  f x a ≤ 0 :=
sorry

end f_leq_zero_l124_124746


namespace total_recovery_time_l124_124879

theorem total_recovery_time 
  (lions: ℕ := 3) (rhinos: ℕ := 2) (time_per_animal: ℕ := 2) :
  (lions + rhinos) * time_per_animal = 10 := by
  sorry

end total_recovery_time_l124_124879


namespace greatest_expression_value_l124_124090

noncomputable def greatest_expression : ℝ := 0.9986095661846496

theorem greatest_expression_value : greatest_expression = 0.9986095661846496 :=
by
  -- proof goes here
  sorry

end greatest_expression_value_l124_124090


namespace special_case_m_l124_124758

theorem special_case_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - 4 * x + 3 = 0 → y = mx^2 - 4 * x + 3 → (x = 0 ∧ m = 0) ∨ (x ≠ 0 ∧ m = 4/3)) :=
sorry

end special_case_m_l124_124758


namespace find_EF_squared_l124_124440

noncomputable def square_side := 15
noncomputable def BE := 6
noncomputable def DF := 6
noncomputable def AE := 14
noncomputable def CF := 14

theorem find_EF_squared (A B C D E F : ℝ) (AB BC CD DA : ℝ := square_side) :
  (BE = 6) → (DF = 6) → (AE = 14) → (CF = 14) → EF^2 = 72 :=
by
  -- Definitions and conditions usage according to (a)
  sorry

end find_EF_squared_l124_124440


namespace f_positive_for_specific_a_l124_124503

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x * Real.log x

theorem f_positive_for_specific_a (x : ℝ) (h : x > 0) :
  f x (Real.exp 3 / 4) > 0 := sorry

end f_positive_for_specific_a_l124_124503


namespace hamburgers_left_over_l124_124379

theorem hamburgers_left_over (h_made : ℕ) (h_served : ℕ) (h_total : h_made = 9) (h_served_count : h_served = 3) : h_made - h_served = 6 :=
by
  sorry

end hamburgers_left_over_l124_124379


namespace necessary_but_not_sufficient_condition_l124_124439

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (m < 1) → (∀ x y : ℝ, (x - m) ^ 2 + y ^ 2 = m ^ 2 → (x, y) ≠ (1, 1)) :=
sorry

end necessary_but_not_sufficient_condition_l124_124439


namespace min_value_2a_minus_ab_l124_124688

theorem min_value_2a_minus_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (ha_lt_11 : a < 11) (hb_lt_11 : b < 11) : 
  ∃ (min_val : ℤ), min_val = -80 ∧ ∀ x y : ℕ, 0 < x → 0 < y → x < 11 → y < 11 → 2 * x - x * y ≥ min_val :=
by
  use -80
  sorry

end min_value_2a_minus_ab_l124_124688


namespace divide_plane_into_regions_l124_124675

theorem divide_plane_into_regions (n : ℕ) (h₁ : n < 199) (h₂ : ∃ (k : ℕ), k = 99):
  n = 100 ∨ n = 198 :=
sorry

end divide_plane_into_regions_l124_124675


namespace remainder_polynomial_l124_124248

noncomputable def p (x : ℝ) : ℝ := sorry
noncomputable def r (x : ℝ) : ℝ := x^2 + x

theorem remainder_polynomial (p : ℝ → ℝ) (r : ℝ → ℝ) :
  (p 2 = 6) ∧ (p 4 = 20) ∧ (p 6 = 42) →
  (r 2 = 2^2 + 2) ∧ (r 4 = 4^2 + 4) ∧ (r 6 = 6^2 + 6) :=
sorry

end remainder_polynomial_l124_124248


namespace value_of_x_l124_124980

theorem value_of_x (x : ℝ) (h1 : (x^2 - 4) / (x + 2) = 0) : x = 2 := by
  sorry

end value_of_x_l124_124980


namespace find_digits_l124_124405

-- Define the digits range
def is_digit (x : ℕ) : Prop := 0 ≤ x ∧ x ≤ 9

-- Define the five-digit numbers
def num_abccc (a b c : ℕ) : ℕ := 10000 * a + 1000 * b + 111 * c
def num_abbbb (a b : ℕ) : ℕ := 10000 * a + 1111 * b

-- Problem statement
theorem find_digits (a b c : ℕ) (h_da : is_digit a) (h_db : is_digit b) (h_dc : is_digit c) :
  (num_abccc a b c) + 1 = (num_abbbb a b) ↔
  (a = 1 ∧ b = 0 ∧ c = 9) ∨ (a = 8 ∧ b = 9 ∧ c = 0) :=
sorry

end find_digits_l124_124405


namespace smallest_f_for_perfect_square_l124_124517

theorem smallest_f_for_perfect_square (f : ℕ) (h₁: 3150 = 2 * 3 * 5^2 * 7) (h₂: ∃ m : ℕ, 3150 * f = m^2) :
  f = 14 :=
sorry

end smallest_f_for_perfect_square_l124_124517


namespace Taran_original_number_is_12_l124_124561

open Nat

theorem Taran_original_number_is_12 (x : ℕ)
  (h1 : (5 * x) + 5 - 5 = 73 ∨ (5 * x) + 5 - 6 = 73 ∨ (5 * x) + 6 - 5 = 73 ∨ (5 * x) + 6 - 6 = 73 ∨ 
       (6 * x) + 5 - 5 = 73 ∨ (6 * x) + 5 - 6 = 73 ∨ (6 * x) + 6 - 5 = 73 ∨ (6 * x) + 6 - 6 = 73) : x = 12 := by
  sorry

end Taran_original_number_is_12_l124_124561


namespace smallest_of_three_numbers_l124_124702

theorem smallest_of_three_numbers : ∀ (a b c : ℕ), (a = 5) → (b = 8) → (c = 4) → min (min a b) c = 4 :=
by
  intros a b c ha hb hc
  rw [ha, hb, hc]
  sorry

end smallest_of_three_numbers_l124_124702


namespace rectangle_same_color_l124_124132

/-- In a 3 × 7 grid where each square is either black or white, 
  there exists a rectangle whose four corners are of the same color. -/
theorem rectangle_same_color (grid : Fin 3 × Fin 7 → Bool) :
  ∃ (r1 r2 : Fin 3) (c1 c2 : Fin 7), r1 ≠ r2 ∧ c1 ≠ c2 ∧ grid (r1, c1) = grid (r1, c2) ∧ grid (r2, c1) = grid (r2, c2) :=
by
  sorry

end rectangle_same_color_l124_124132


namespace sally_total_score_l124_124098

theorem sally_total_score :
  ∀ (correct incorrect unanswered : ℕ) (score_correct score_incorrect : ℝ),
    correct = 17 →
    incorrect = 8 →
    unanswered = 5 →
    score_correct = 1 →
    score_incorrect = -0.25 →
    (correct * score_correct +
     incorrect * score_incorrect +
     unanswered * 0) = 15 :=
by
  intros correct incorrect unanswered score_correct score_incorrect
  intros h_corr h_incorr h_unan h_sc h_si
  sorry

end sally_total_score_l124_124098


namespace canonical_equations_of_line_intersection_l124_124981

theorem canonical_equations_of_line_intersection
  (x y z : ℝ)
  (h1 : 2 * x - 3 * y + z + 6 = 0)
  (h2 : x - 3 * y - 2 * z + 3 = 0) :
  (∃ (m n p x0 y0 z0 : ℝ), 
  m * (x + 3) = n * y ∧ n * y = p * z ∧ 
  m = 9 ∧ n = 5 ∧ p = -3 ∧ 
  x0 = -3 ∧ y0 = 0 ∧ z0 = 0) :=
sorry

end canonical_equations_of_line_intersection_l124_124981


namespace rabbitAgeOrder_l124_124377

-- Define the ages of the rabbits as variables
variables (blue black red gray : ℕ)

-- Conditions based on the problem statement
noncomputable def rabbitConditions := 
  (blue ≠ max blue (max black (max red gray))) ∧  -- The blue-eyed rabbit is not the eldest
  (gray ≠ min blue (min black (min red gray))) ∧  -- The gray rabbit is not the youngest
  (red ≠ min blue (min black (min red gray))) ∧  -- The red-eyed rabbit is not the youngest
  (black > red) ∧ (gray > black)  -- The black rabbit is older than the red-eyed rabbit and younger than the gray rabbit

-- Required proof statement
theorem rabbitAgeOrder : rabbitConditions blue black red gray → gray > black ∧ black > red ∧ red > blue :=
by
  intro h
  sorry

end rabbitAgeOrder_l124_124377


namespace color_cartridge_cost_l124_124143

theorem color_cartridge_cost :
  ∃ C : ℝ, 
  (1 * 27) + (3 * C) = 123 ∧ C = 32 :=
by
  sorry

end color_cartridge_cost_l124_124143


namespace students_taking_all_three_classes_l124_124996

variable (students : Finset ℕ)
variable (yoga bridge painting : Finset ℕ)

variables (yoga_count bridge_count painting_count at_least_two exactly_two all_three : ℕ)

variable (total_students : students.card = 25)
variable (yoga_students : yoga.card = 12)
variable (bridge_students : bridge.card = 15)
variable (painting_students : painting.card = 11)
variable (at_least_two_classes : at_least_two = 10)
variable (exactly_two_classes : exactly_two = 7)

theorem students_taking_all_three_classes :
  all_three = 3 :=
sorry

end students_taking_all_three_classes_l124_124996


namespace find_valid_n_l124_124214

noncomputable def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

theorem find_valid_n (n : ℕ) (h1 : n > 0) (h2 : n < 200) (h3 : is_square (n^2 + (n + 1)^2)) :
  n = 3 ∨ n = 20 ∨ n = 119 :=
by
  sorry

end find_valid_n_l124_124214


namespace solve_system_of_equations_l124_124713

theorem solve_system_of_equations 
  (a b c s : ℝ) (x y z : ℝ)
  (h1 : y^2 - z * x = a * (x + y + z)^2)
  (h2 : x^2 - y * z = b * (x + y + z)^2)
  (h3 : z^2 - x * y = c * (x + y + z)^2)
  (h4 : a^2 + b^2 + c^2 - (a * b + b * c + c * a) = a + b + c) :
  (x = 0 ∧ y = 0 ∧ z = 0 ∧ x + y + z = 0) ∨
  ((x + y + z ≠ 0) ∧
   (x = (2 * c - a - b + 1) * s) ∧
   (y = (2 * a - b - c + 1) * s) ∧
   (z = (2 * b - c - a + 1) * s)) :=
by
  sorry

end solve_system_of_equations_l124_124713


namespace percent_of_day_is_hours_l124_124199

theorem percent_of_day_is_hours (h : ℝ) (day_hours : ℝ) (percent : ℝ) 
  (day_hours_def : day_hours = 24)
  (percent_def : percent = 29.166666666666668) :
  h = 7 :=
by
  sorry

end percent_of_day_is_hours_l124_124199


namespace average_rainfall_feb_1983_l124_124674

theorem average_rainfall_feb_1983 (total_rainfall : ℕ) (days_in_february : ℕ) (hours_per_day : ℕ) 
  (H1 : total_rainfall = 789) (H2 : days_in_february = 28) (H3 : hours_per_day = 24) : 
  total_rainfall / (days_in_february * hours_per_day) = 789 / 672 :=
by
  sorry

end average_rainfall_feb_1983_l124_124674


namespace angle_measure_l124_124356

theorem angle_measure (x : ℝ) :
  (180 - x) = 7 * (90 - x) → 
  x = 75 :=
by
  intro h
  sorry

end angle_measure_l124_124356


namespace ryan_learning_hours_l124_124601

theorem ryan_learning_hours (H_E : ℕ) (H_C : ℕ) (h1 : H_E = 6) (h2 : H_C = 2) : H_E - H_C = 4 := by
  sorry

end ryan_learning_hours_l124_124601


namespace custom_op_4_2_l124_124921

-- Define the custom operation
def custom_op (a b : ℝ) : ℝ := 5 * a + 2 * b

-- State the theorem to prove the result
theorem custom_op_4_2 : custom_op 4 2 = 24 :=
by
  sorry

end custom_op_4_2_l124_124921


namespace go_stones_perimeter_l124_124183

-- Define the conditions for the problem
def stones_wide : ℕ := 4
def stones_tall : ℕ := 8

-- Define what we want to prove based on the conditions
theorem go_stones_perimeter : 2 * stones_wide + 2 * stones_tall - 4 = 20 :=
by
  -- Proof would normally go here
  sorry

end go_stones_perimeter_l124_124183


namespace base_conversion_and_addition_l124_124772

theorem base_conversion_and_addition :
  let n1 := 2 * (8:ℕ)^2 + 4 * 8^1 + 3 * 8^0
  let d1 := 1 * 4^1 + 3 * 4^0
  let n2 := 2 * 7^2 + 0 * 7^1 + 4 * 7^0
  let d2 := 2 * 5^1 + 3 * 5^0
  n1 / d1 + n2 / d2 = 31 + 51 / 91 := by
  sorry

end base_conversion_and_addition_l124_124772


namespace solve_for_x_l124_124926

theorem solve_for_x : ∀ x, (8 * x^2 + 150 * x + 2) / (3 * x + 50) = 4 * x + 2 ↔ x = -7 / 2 := by
  sorry

end solve_for_x_l124_124926


namespace karen_piggy_bank_total_l124_124730

theorem karen_piggy_bank_total (a r n : ℕ) (h1 : a = 2) (h2 : r = 3) (h3 : n = 7) :
  (a * ((1 - r^n) / (1 - r))) = 2186 := by
  sorry

end karen_piggy_bank_total_l124_124730


namespace min_value_of_expression_l124_124500

theorem min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : 
  x^2 + 4 * y^2 + 2 * x * y ≥ 3 / 4 :=
sorry

end min_value_of_expression_l124_124500


namespace elsa_data_usage_l124_124544

theorem elsa_data_usage (D : ℝ) 
  (h_condition : D - 300 - (2/5) * (D - 300) = 120) : D = 500 := 
sorry

end elsa_data_usage_l124_124544


namespace inequality_solution_l124_124000

theorem inequality_solution :
  {x : ℝ | ((x > 4) ∧ (x < 5)) ∨ ((x > 6) ∧ (x < 7)) ∨ (x > 7)} =
  {x : ℝ | (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6) * (x - 7)) > 0} :=
sorry

end inequality_solution_l124_124000


namespace find_x_add_inv_l124_124950

theorem find_x_add_inv (x : ℝ) (h : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_add_inv_l124_124950


namespace addition_problem_l124_124494

theorem addition_problem (x y S : ℕ) 
    (h1 : x = S - 2000)
    (h2 : S = y + 6) :
    x = 6 ∧ y = 2000 ∧ S = 2006 :=
by
  -- The proof will go here
  sorry

end addition_problem_l124_124494


namespace breadth_of_rectangular_plot_l124_124897

variable (A b l : ℝ)

theorem breadth_of_rectangular_plot :
  (A = 15 * b) ∧ (l = b + 10) ∧ (A = l * b) → b = 5 :=
by
  intro h
  sorry

end breadth_of_rectangular_plot_l124_124897


namespace werewolf_is_A_l124_124866

def is_liar (x : ℕ) : Prop := sorry
def is_knight (x : ℕ) : Prop := sorry
def is_werewolf (x : ℕ) : Prop := sorry

axiom A : ℕ
axiom B : ℕ
axiom C : ℕ

-- Conditions from the problem
axiom A_statement : is_liar A ∨ is_liar B
axiom B_statement : is_werewolf C
axiom exactly_one_werewolf : 
  (is_werewolf A ∧ ¬ is_werewolf B ∧ ¬ is_werewolf C) ∨
  (is_werewolf B ∧ ¬ is_werewolf A ∧ ¬ is_werewolf C) ∨
  (is_werewolf C ∧ ¬ is_werewolf A ∧ ¬ is_werewolf B)
axiom werewolf_is_knight : ∀ x : ℕ, is_werewolf x → is_knight x

-- Prove the conclusion
theorem werewolf_is_A : 
  is_werewolf A ∧ is_knight A :=
sorry

end werewolf_is_A_l124_124866


namespace inequality_div_l124_124518

theorem inequality_div (m n : ℝ) (h : m > n) : (m / 5) > (n / 5) :=
sorry

end inequality_div_l124_124518


namespace distinct_pairs_disjoint_subsets_l124_124999

theorem distinct_pairs_disjoint_subsets (n : ℕ) : 
  ∃ k, k = (3^n + 1) / 2 := 
sorry

end distinct_pairs_disjoint_subsets_l124_124999


namespace c_seq_formula_l124_124922

def x_seq (n : ℕ) : ℕ := 2 * n - 1
def y_seq (n : ℕ) : ℕ := n ^ 2
def c_seq (n : ℕ) : ℕ := (2 * n - 1) ^ 2

theorem c_seq_formula (n : ℕ) : ∀ k, (c_seq k) = (2 * k - 1) ^ 2 :=
by
  sorry

end c_seq_formula_l124_124922


namespace largest_solution_achieves_largest_solution_l124_124360

theorem largest_solution (x : ℝ) (hx : ⌊x⌋ = 5 + 100 * (x - ⌊x⌋)) : x ≤ 104.99 :=
by
  -- Placeholder for the proof
  sorry

theorem achieves_largest_solution : ∃ (x : ℝ), ⌊x⌋ = 5 + 100 * (x - ⌊x⌋) ∧ x = 104.99 :=
by
  -- Placeholder for the proof
  sorry

end largest_solution_achieves_largest_solution_l124_124360


namespace chord_PQ_eqn_l124_124795

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 9
def midpoint_PQ (M : ℝ × ℝ) : Prop := M = (1, 2)
def line_PQ_eq (x y : ℝ) : Prop := x + 2 * y - 5 = 0

theorem chord_PQ_eqn : 
  (∃ P Q : ℝ × ℝ, circle_eq P.1 P.2 ∧ circle_eq Q.1 Q.2 ∧ midpoint_PQ ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)) →
  ∃ x y : ℝ, line_PQ_eq x y := 
sorry

end chord_PQ_eqn_l124_124795


namespace fifth_number_in_pascals_triangle_l124_124324

def factorial(n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def binomial (n k : ℕ) : ℕ :=
  factorial n / (factorial k * factorial (n - k))

theorem fifth_number_in_pascals_triangle : binomial 15 4 = 1365 := by
  sorry

end fifth_number_in_pascals_triangle_l124_124324


namespace charge_for_each_additional_fifth_mile_l124_124868

theorem charge_for_each_additional_fifth_mile
  (initial_charge : ℝ)
  (total_charge : ℝ)
  (distance_in_miles : ℕ)
  (distance_per_increment : ℝ)
  (x : ℝ) :
  initial_charge = 2.10 →
  total_charge = 17.70 →
  distance_in_miles = 8 →
  distance_per_increment = 1/5 →
  (total_charge - initial_charge) / ((distance_in_miles / distance_per_increment) - 1) = x →
  x = 0.40 :=
by
  intros h_initial_charge h_total_charge h_distance_in_miles h_distance_per_increment h_eq
  sorry

end charge_for_each_additional_fifth_mile_l124_124868


namespace garrison_men_initial_l124_124008

theorem garrison_men_initial (M : ℕ) (P : ℕ):
  (P = M * 40) →
  (P / 2 = (M + 2000) * 10) →
  M = 2000 :=
by
  intros h1 h2
  sorry

end garrison_men_initial_l124_124008


namespace incorrect_operation_B_l124_124916

theorem incorrect_operation_B : (4 + 5)^2 ≠ 4^2 + 5^2 := 
  sorry

end incorrect_operation_B_l124_124916


namespace rectangular_C₁_general_C₂_intersection_and_sum_l124_124689

-- Definition of curve C₁ in polar coordinates
def C₁_polar (ρ θ : ℝ) : Prop := ρ * Real.cos θ ^ 2 = Real.sin θ

-- Definition of curve C₂ in parametric form
def C₂_param (k x y : ℝ) : Prop := 
  x = 8 * k / (1 + k^2) ∧ y = 2 * (1 - k^2) / (1 + k^2)

-- Rectangular coordinate equation of curve C₁ is x² = y
theorem rectangular_C₁ (ρ θ : ℝ) (x y : ℝ) (h₁ : ρ * Real.cos θ ^ 2 = Real.sin θ)
  (h₂ : x = ρ * Real.cos θ) (h₃ : y = ρ * Real.sin θ) : x^2 = y :=
sorry

-- General equation of curve C₂ is x² / 16 + y² / 4 = 1 with y ≠ -2
theorem general_C₂ (k x y : ℝ) (h₁ : x = 8 * k / (1 + k^2))
  (h₂ : y = 2 * (1 - k^2) / (1 + k^2)) : x^2 / 16 + y^2 / 4 = 1 ∧ y ≠ -2 :=
sorry

-- Given point M and parametric line l, prove the value of sum reciprocals of distances to points of intersection with curve C₁ is √7
theorem intersection_and_sum (t m₁ m₂ x y : ℝ) 
  (M : ℝ × ℝ) (hM : M = (0, 1/2))
  (hline : x = Real.sqrt 3 * t ∧ y = 1/2 + t)
  (hintersect1 : 3 * m₁^2 - 2 * m₁ - 2 = 0)
  (hintersect2 : 3 * m₂^2 - 2 * m₂ - 2 = 0)
  (hroot1_2 : m₁ + m₂ = 2/3 ∧ m₁ * m₂ = -2/3) : 
  1 / abs (M.fst - x) + 1 / abs (M.snd - y) = Real.sqrt 7 :=
sorry

end rectangular_C₁_general_C₂_intersection_and_sum_l124_124689


namespace giant_spider_leg_cross_sectional_area_l124_124264

theorem giant_spider_leg_cross_sectional_area :
  let previous_spider_weight := 6.4
  let weight_multiplier := 2.5
  let pressure := 4
  let num_legs := 8

  let giant_spider_weight := weight_multiplier * previous_spider_weight
  let weight_per_leg := giant_spider_weight / num_legs
  let cross_sectional_area := weight_per_leg / pressure

  cross_sectional_area = 0.5 :=
by 
  sorry

end giant_spider_leg_cross_sectional_area_l124_124264


namespace tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l124_124077

theorem tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth (a : ℝ) (h : Real.tan a = 2) :
  Real.cos (2 * a) + Real.sin (2 * a) = 1 / 5 :=
by
  sorry

end tan_a_eq_two_imp_cos_2a_plus_sin_2a_eq_one_fifth_l124_124077


namespace area_change_l124_124860

theorem area_change (L B : ℝ) (hL : L > 0) (hB : B > 0) :
  let L' := 1.2 * L
  let B' := 0.8 * B
  let A := L * B
  let A' := L' * B'
  A' = 0.96 * A :=
by
  sorry

end area_change_l124_124860


namespace smaller_cone_volume_ratio_l124_124152

theorem smaller_cone_volume_ratio :
  let r := 12
  let theta1 := 120
  let theta2 := 240
  let arc_length_small := (theta1 / 360) * (2 * Real.pi * r)
  let arc_length_large := (theta2 / 360) * (2 * Real.pi * r)
  let r1 := arc_length_small / (2 * Real.pi)
  let r2 := arc_length_large / (2 * Real.pi)
  let l := r
  let h1 := Real.sqrt (l^2 - r1^2)
  let h2 := Real.sqrt (l^2 - r2^2)
  let V1 := (1 / 3) * Real.pi * r1^2 * h1
  let V2 := (1 / 3) * Real.pi * r2^2 * h2
  V1 / V2 = Real.sqrt 10 / 10 := sorry

end smaller_cone_volume_ratio_l124_124152


namespace no_graph_for_equation_l124_124456

theorem no_graph_for_equation (x y : ℝ) : 
  ¬ ∃ (x y : ℝ), x^2 + y^2 + 2*x + 4*y + 6 = 0 := 
by 
  sorry

end no_graph_for_equation_l124_124456


namespace probability_of_log_ge_than_1_l124_124649

noncomputable def probability_log_greater_than_one : ℝ := sorry

theorem probability_of_log_ge_than_1 :
  probability_log_greater_than_one = 1 / 2 :=
sorry

end probability_of_log_ge_than_1_l124_124649


namespace gray_region_area_l124_124553

theorem gray_region_area (r : ℝ) : 
  let inner_circle_radius := r
  let outer_circle_radius := r + 3
  let inner_circle_area := Real.pi * (r ^ 2)
  let outer_circle_area := Real.pi * ((r + 3) ^ 2)
  let gray_region_area := outer_circle_area - inner_circle_area
  gray_region_area = 6 * Real.pi * r + 9 * Real.pi := 
by
  sorry

end gray_region_area_l124_124553


namespace problems_per_page_is_five_l124_124917

-- Let M and R be the number of problems on each math and reading page respectively
variables (M R : ℕ)

-- Conditions given in problem
def two_math_pages := 2 * M
def four_reading_pages := 4 * R
def total_problems := two_math_pages + four_reading_pages

-- Assume the number of problems per page is the same for both math and reading as P
variable (P : ℕ)
def problems_per_page_equal := (2 * P) + (4 * P) = 30

theorem problems_per_page_is_five :
  (2 * P) + (4 * P) = 30 → P = 5 :=
by
  intro h
  sorry

end problems_per_page_is_five_l124_124917


namespace inequality_holds_l124_124200

theorem inequality_holds (x : ℝ) (n : ℕ) (hn : 0 < n) : 
  Real.sin (2 * x)^n + (Real.sin x^n - Real.cos x^n)^2 ≤ 1 := 
sorry

end inequality_holds_l124_124200


namespace necessary_and_sufficient_condition_l124_124621

variable (p q : Prop)

theorem necessary_and_sufficient_condition (hp : p) (hq : q) : ¬p ∨ ¬q = False :=
by {
    -- You are requested to fill out the proof here.
    sorry
}

end necessary_and_sufficient_condition_l124_124621


namespace arrangement_correct_l124_124138

def A := 4
def B := 1
def C := 2
def D := 5
def E := 6
def F := 3

def sum1 := A + B + C
def sum2 := A + D + F
def sum3 := B + E + D
def sum4 := C + F + E
def sum5 := A + E + F
def sum6 := B + D + C
def sum7 := B + C + F

theorem arrangement_correct :
  sum1 = 15 ∧ sum2 = 15 ∧ sum3 = 15 ∧ sum4 = 15 ∧ sum5 = 15 ∧ sum6 = 15 ∧ sum7 = 15 := 
by
  unfold sum1 sum2 sum3 sum4 sum5 sum6 sum7 
  unfold A B C D E F
  sorry

end arrangement_correct_l124_124138


namespace triple_hash_90_l124_124122

def hash (N : ℝ) : ℝ := 0.3 * N + 2

theorem triple_hash_90 : hash (hash (hash 90)) = 5.21 :=
by
  sorry

end triple_hash_90_l124_124122


namespace certain_number_plus_two_l124_124520

theorem certain_number_plus_two (x : ℤ) (h : x - 2 = 5) : x + 2 = 9 := by
  sorry

end certain_number_plus_two_l124_124520


namespace calculate_pow_zero_l124_124212

theorem calculate_pow_zero: (2023 - Real.pi) ≠ 0 → (2023 - Real.pi)^0 = 1 := by
  -- Proof
  sorry

end calculate_pow_zero_l124_124212


namespace percentage_of_green_ducks_smaller_pond_l124_124968

-- Definitions of the conditions
def num_ducks_smaller_pond : ℕ := 30
def num_ducks_larger_pond : ℕ := 50
def percentage_green_larger_pond : ℕ := 12
def percentage_green_total : ℕ := 15
def total_ducks : ℕ := num_ducks_smaller_pond + num_ducks_larger_pond

-- Calculation of the number of green ducks
def num_green_larger_pond := percentage_green_larger_pond * num_ducks_larger_pond / 100
def num_green_total := percentage_green_total * total_ducks / 100

-- Define the percentage of green ducks in the smaller pond
def percentage_green_smaller_pond (x : ℕ) :=
  x * num_ducks_smaller_pond / 100 + num_green_larger_pond = num_green_total

-- The theorem to be proven
theorem percentage_of_green_ducks_smaller_pond : percentage_green_smaller_pond 20 :=
  sorry

end percentage_of_green_ducks_smaller_pond_l124_124968


namespace complementary_angles_of_same_angle_are_equal_l124_124756

def complementary_angles (α β : ℝ) := α + β = 90 

theorem complementary_angles_of_same_angle_are_equal 
        (θ : ℝ) (α β : ℝ) 
        (h1 : complementary_angles θ α) 
        (h2 : complementary_angles θ β) : 
        α = β := 
by 
  sorry

end complementary_angles_of_same_angle_are_equal_l124_124756


namespace greatest_sum_solution_l124_124218

theorem greatest_sum_solution (x y : ℤ) (h : x^2 + y^2 = 20) : 
  x + y ≤ 6 :=
sorry

end greatest_sum_solution_l124_124218


namespace maximize_profit_l124_124164

noncomputable def I (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * (x - 1) * Real.exp (x - 2) + 2
  else if h' : 2 < x ∧ x ≤ 50 then 440 + 3050 / x - 9000 / x^2
  else 0 -- default case for Lean to satisfy definition

noncomputable def P (x : ℝ) : ℝ :=
  if h : 0 < x ∧ x ≤ 2 then 2 * x * (x - 1) * Real.exp (x - 2) - 448 * x - 180
  else if h' : 2 < x ∧ x ≤ 50 then -10 * x - 9000 / x + 2870
  else 0 -- default case for Lean to satisfy definition

theorem maximize_profit :
  (∀ x : ℝ, 0 < x ∧ x ≤ 50 → P x ≤ 2270) ∧ P 30 = 2270 :=
by
  sorry

end maximize_profit_l124_124164


namespace unique_zero_iff_a_in_range_l124_124100

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 3 * x^2 + 1

theorem unique_zero_iff_a_in_range (a : ℝ) :
  (∃ x0 : ℝ, f a x0 = 0 ∧ (∀ x1 : ℝ, f a x1 = 0 → x1 = x0) ∧ x0 > 0) ↔ a < -2 :=
by sorry

end unique_zero_iff_a_in_range_l124_124100


namespace n_pow4_sub_n_pow2_divisible_by_12_l124_124087

theorem n_pow4_sub_n_pow2_divisible_by_12 (n : ℤ) (h : n > 1) : 12 ∣ (n^4 - n^2) :=
by sorry

end n_pow4_sub_n_pow2_divisible_by_12_l124_124087


namespace average_visitors_per_day_in_november_l124_124595
-- Import the entire Mathlib library for necessary definitions and operations.

-- Define the average visitors per different days of the week.
def sunday_visitors := 510
def monday_visitors := 240
def tuesday_visitors := 240
def wednesday_visitors := 300
def thursday_visitors := 300
def friday_visitors := 200
def saturday_visitors := 200

-- Define the counts of each type of day in November.
def sundays := 5
def mondays := 4
def tuesdays := 4
def wednesdays := 4
def thursdays := 4
def fridays := 4
def saturdays := 4

-- Define the number of days in November.
def days_in_november := 30

-- State the theorem to prove the average number of visitors per day.
theorem average_visitors_per_day_in_november : 
  (5 * sunday_visitors + 
   4 * monday_visitors + 
   4 * tuesday_visitors + 
   4 * wednesday_visitors + 
   4 * thursday_visitors + 
   4 * friday_visitors + 
   4 * saturday_visitors) / days_in_november = 282 :=
by
  sorry

end average_visitors_per_day_in_november_l124_124595


namespace sum_of_digits_base2_345_l124_124527

open Nat -- open natural numbers namespace

theorem sum_of_digits_base2_345 : (Nat.digits 2 345).sum = 5 := by
  sorry -- proof to be filled in later

end sum_of_digits_base2_345_l124_124527


namespace angle_A_minimum_a_l124_124805

variable {α : Type} [LinearOrderedField α]

-- Part 1: Prove A = π / 3 given the specific equation in triangle ABC
theorem angle_A (a b c : α) (cos : α → α)
  (h : b^2 * c * cos c + c^2 * b * cos b = a * b^2 + a * c^2 - a^3) :
  ∃ A : α, A = π / 3 :=
sorry

-- Part 2: Prove the minimum value of a is 1 when b + c = 2
theorem minimum_a (a b c : α) (h : b + c = 2) :
  ∃ a : α, a = 1 :=
sorry

end angle_A_minimum_a_l124_124805


namespace radian_measure_of_sector_l124_124738

theorem radian_measure_of_sector
  (perimeter : ℝ) (area : ℝ) (radian_measure : ℝ)
  (h1 : perimeter = 8)
  (h2 : area = 4) :
  radian_measure = 2 :=
sorry

end radian_measure_of_sector_l124_124738


namespace blocks_to_get_home_l124_124574

-- Definitions based on conditions provided
def blocks_to_park := 4
def blocks_to_school := 7
def trips_per_day := 3
def total_daily_blocks := 66

-- The proof statement for the number of blocks Ray walks to get back home
theorem blocks_to_get_home 
  (h1: blocks_to_park = 4)
  (h2: blocks_to_school = 7)
  (h3: trips_per_day = 3)
  (h4: total_daily_blocks = 66) : 
  (total_daily_blocks / trips_per_day - (blocks_to_park + blocks_to_school) = 11) :=
by
  sorry

end blocks_to_get_home_l124_124574


namespace find_pairs_l124_124964

theorem find_pairs (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∃ q r : ℕ, a^2 + b^2 = (a + b) * q + r ∧ q^2 + r = 1977) →
  (a, b) = (50, 37) ∨ (a, b) = (37, 50) ∨ (a, b) = (50, 7) ∨ (a, b) = (7, 50) :=
by
  sorry

end find_pairs_l124_124964


namespace complement_M_eq_45_l124_124287

open Set Nat

/-- Define the universal set U and the set M in Lean -/
def U : Set ℕ := {1, 2, 3, 4, 5, 6}

def M : Set ℕ := {x | 6 % x = 0 ∧ x ∈ U}

/-- Lean theorem statement for the complement of M in U -/
theorem complement_M_eq_45 : (U \ M) = {4, 5} :=
by
  sorry

end complement_M_eq_45_l124_124287


namespace solve_garden_width_l124_124599

noncomputable def garden_width_problem (w l : ℕ) :=
  (w + l = 30) ∧ (w * l = 200) ∧ (l = w + 8) → w = 11

theorem solve_garden_width (w l : ℕ) : garden_width_problem w l :=
by
  intro h
  -- Omitting the actual proof
  sorry

end solve_garden_width_l124_124599


namespace triangle_formation_inequalities_l124_124640

theorem triangle_formation_inequalities (a b c d : ℝ)
  (h_abc_pos : 0 < a)
  (h_bcd_pos : 0 < b)
  (h_cde_pos : 0 < c)
  (h_def_pos : 0 < d)
  (tri_ineq_1 : a + b + c > d)
  (tri_ineq_2 : b + c + d > a)
  (tri_ineq_3 : a + d > b + c) :
  (a < (b + c + d) / 2) ∧ (b + c < a + d) ∧ (¬ (c + d < b / 2)) :=
by 
  sorry

end triangle_formation_inequalities_l124_124640


namespace Dan_team_lost_games_l124_124341

/-- Dan's high school played eighteen baseball games this year.
Two were at night and they won 15 games. Prove that they lost 3 games. -/
theorem Dan_team_lost_games (total_games won_games : ℕ) (h_total : total_games = 18) (h_won : won_games = 15) :
  total_games - won_games = 3 :=
by {
  sorry
}

end Dan_team_lost_games_l124_124341


namespace interest_rate_second_part_l124_124162

noncomputable def P1 : ℝ := 2799.9999999999995
noncomputable def P2 : ℝ := 4000 - P1
noncomputable def Interest1 : ℝ := P1 * (3 / 100)
noncomputable def TotalInterest : ℝ := 144
noncomputable def Interest2 : ℝ := TotalInterest - Interest1

theorem interest_rate_second_part :
  ∃ r : ℝ, Interest2 = P2 * (r / 100) ∧ r = 5 :=
by
  sorry

end interest_rate_second_part_l124_124162


namespace mike_total_work_time_l124_124542

theorem mike_total_work_time :
  let wash_time := 10
  let oil_change_time := 15
  let tire_change_time := 30
  let paint_time := 45
  let engine_service_time := 60

  let num_wash := 9
  let num_oil_change := 6
  let num_tire_change := 2
  let num_paint := 4
  let num_engine_service := 3
  
  let total_minutes := 
        num_wash * wash_time +
        num_oil_change * oil_change_time +
        num_tire_change * tire_change_time +
        num_paint * paint_time +
        num_engine_service * engine_service_time

  let total_hours := total_minutes / 60

  total_hours = 10 :=
  by
    -- Definitions of times per task
    let wash_time := 10
    let oil_change_time := 15
    let tire_change_time := 30
    let paint_time := 45
    let engine_service_time := 60

    -- Definitions of number of tasks performed
    let num_wash := 9
    let num_oil_change := 6
    let num_tire_change := 2
    let num_paint := 4
    let num_engine_service := 3

    -- Calculate total minutes
    let total_minutes := 
      num_wash * wash_time +
      num_oil_change * oil_change_time +
      num_tire_change * tire_change_time +
      num_paint * paint_time +
      num_engine_service * engine_service_time
    
    -- Calculate total hours
    let total_hours := total_minutes / 60

    -- Required equality to prove
    have : total_hours = 10 := sorry
    exact this

end mike_total_work_time_l124_124542


namespace stickers_on_first_day_l124_124704

theorem stickers_on_first_day (s e total : ℕ) (h1 : e = 22) (h2 : total = 61) (h3 : total = s + e) : s = 39 :=
by
  sorry

end stickers_on_first_day_l124_124704


namespace find_f_at_one_l124_124093

variable (a b : ℝ)
def f (x : ℝ) : ℝ := a * x^4 + b * x^2 + 2 * x - 8

theorem find_f_at_one (h_cond : f a b (-1) = 10) : f a b (1) = 14 := by
  sorry

end find_f_at_one_l124_124093


namespace johns_age_l124_124965

theorem johns_age (d j : ℕ) (h1 : j = d - 30) (h2 : j + d = 80) : j = 25 :=
by
  sorry

end johns_age_l124_124965


namespace interest_time_period_l124_124079

-- Define the constants given in the problem
def principal : ℝ := 4000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def interest_difference : ℝ := 480

-- Define the time period T
def time_period : ℝ := 2

-- Define a proof statement
theorem interest_time_period :
  (principal * rate1 * time_period) - (principal * rate2 * time_period) = interest_difference :=
by {
  -- We skip the proof since it's not required by the problem statement
  sorry
}

end interest_time_period_l124_124079


namespace simplify_expression_l124_124279

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x = 1) :
  (1 - x) ^ 2 - (x + 3) * (3 - x) - (x - 3) * (x - 1) = -10 :=
by 
  sorry

end simplify_expression_l124_124279


namespace necessary_but_not_sufficient_condition_l124_124343

-- Prove that x^2 ≥ -x is a necessary but not sufficient condition for |x| = x
theorem necessary_but_not_sufficient_condition (x : ℝ) : x^2 ≥ -x → |x| = x ↔ x ≥ 0 := 
sorry

end necessary_but_not_sufficient_condition_l124_124343


namespace probability_10_coins_at_most_3_heads_l124_124412

def probability_at_most_3_heads (n : ℕ) : ℚ :=
  let total_outcomes := 2^n
  let favorable_outcomes := (Nat.choose n 0) + (Nat.choose n 1) + (Nat.choose n 2) + (Nat.choose n 3)
  favorable_outcomes / total_outcomes

theorem probability_10_coins_at_most_3_heads : probability_at_most_3_heads 10 = 11 / 64 :=
by
  sorry

end probability_10_coins_at_most_3_heads_l124_124412


namespace count_square_of_integer_fraction_l124_124062

theorem count_square_of_integer_fraction :
  ∃ n_values : Finset ℤ, n_values = ({0, 15, 24} : Finset ℤ) ∧
  (∀ n ∈ n_values, ∃ k : ℤ, n / (30 - n) = k ^ 2) ∧
  n_values.card = 3 :=
by
  sorry

end count_square_of_integer_fraction_l124_124062


namespace probability_A_selected_l124_124799

def n : ℕ := 5
def k : ℕ := 2

def total_ways : ℕ := Nat.choose n k  -- C(n, k)

def favorable_ways : ℕ := Nat.choose (n - 1) (k - 1)  -- C(n-1, k-1)

theorem probability_A_selected : (favorable_ways : ℚ) / (total_ways : ℚ) = 2 / 5 :=
by
  sorry

end probability_A_selected_l124_124799


namespace paper_plates_cost_l124_124794

theorem paper_plates_cost (P C x : ℝ) 
(h1 : 100 * P + 200 * C = 6.00) 
(h2 : x * P + 40 * C = 1.20) : 
x = 20 := 
sorry

end paper_plates_cost_l124_124794


namespace triangle_area_l124_124022

theorem triangle_area (BC AC : ℝ) (angle_BAC : ℝ) (h1 : BC = 12) (h2 : AC = 5) (h3 : angle_BAC = π / 6) :
  1/2 * BC * (AC * Real.sin angle_BAC) = 15 :=
by
  sorry

end triangle_area_l124_124022


namespace find_cost_price_of_clock_l124_124294

namespace ClockCost

variable (C : ℝ)

def cost_price_each_clock (n : ℝ) (gain1 : ℝ) (gain2 : ℝ) (uniform_gain : ℝ) (price_difference : ℝ) :=
  let selling_price1 := 40 * C * (1 + gain1)
  let selling_price2 := 50 * C * (1 + gain2)
  let uniform_selling_price := n * C * (1 + uniform_gain)
  selling_price1 + selling_price2 - uniform_selling_price = price_difference

theorem find_cost_price_of_clock (C : ℝ) (h : cost_price_each_clock C 90 0.10 0.20 0.15 40) : C = 80 :=
  sorry

end ClockCost

end find_cost_price_of_clock_l124_124294


namespace total_coins_l124_124418
-- Import the necessary library

-- Defining the conditions
def quarters := 22
def dimes := quarters + 3
def nickels := quarters - 6

-- Main theorem statement
theorem total_coins : (quarters + dimes + nickels) = 63 := by
  sorry

end total_coins_l124_124418


namespace find_plaintext_from_ciphertext_l124_124915

theorem find_plaintext_from_ciphertext : 
  ∃ x : ℕ, ∀ a : ℝ, (a^3 - 2 = 6) → (1022 = a^x - 2) → x = 10 :=
by
  use 10
  intros a ha hc
  -- Proof omitted
  sorry

end find_plaintext_from_ciphertext_l124_124915


namespace dvaneft_shares_percentage_range_l124_124226

theorem dvaneft_shares_percentage_range :
  ∀ (x y z n m : ℝ),
    (4 * x * n = y * m) →
    (x * n + y * m = z * (m + n)) →
    (16 ≤ y - x ∧ y - x ≤ 20) →
    (42 ≤ z ∧ z ≤ 60) →
    (12.5 ≤ (n / (2 * (n + m)) * 100) ∧ (n / (2 * (n + m)) * 100) ≤ 15) :=
by
  intros x y z n m h1 h2 h3 h4
  sorry

end dvaneft_shares_percentage_range_l124_124226


namespace length_of_second_train_l124_124125

theorem length_of_second_train
  (length_first_train : ℝ)
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (clear_time_seconds : ℝ)
  (relative_speed_kmph : ℝ) :
  speed_first_train_kmph + speed_second_train_kmph = relative_speed_kmph →
  relative_speed_kmph * (5 / 18) * clear_time_seconds = length_first_train + 280 :=
by
  let length_first_train := 120
  let speed_first_train_kmph := 42
  let speed_second_train_kmph := 30
  let clear_time_seconds := 20
  let relative_speed_kmph := 72
  sorry

end length_of_second_train_l124_124125


namespace convert_108_kmph_to_mps_l124_124210

-- Definitions and assumptions
def kmph_to_mps (speed_kmph : ℕ) : ℚ :=
  speed_kmph * (1000 / 3600)

-- Theorem statement
theorem convert_108_kmph_to_mps : kmph_to_mps 108 = 30 := 
by
  sorry

end convert_108_kmph_to_mps_l124_124210


namespace quadratic_polynomial_solution_l124_124750

theorem quadratic_polynomial_solution :
  ∃ a b c : ℚ, 
    (∀ x : ℚ, ax*x + bx + c = 8 ↔ x = -2) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 2 ↔ x = 1) ∧ 
    (∀ x : ℚ, ax*x + bx + c = 10 ↔ x = 3) ∧ 
    a = 6 / 5 ∧ 
    b = -4 / 5 ∧ 
    c = 8 / 5 :=
by {
  sorry
}

end quadratic_polynomial_solution_l124_124750


namespace average_weight_estimation_exclude_friend_l124_124018

theorem average_weight_estimation_exclude_friend
    (w : ℝ)
    (H1 : 62.4 < w ∧ w < 72.1)
    (H2 : 60.3 < w ∧ w < 70.6)
    (H3 : w ≤ 65.9)
    (H4 : 63.7 < w ∧ w < 66.3)
    (H5 : 75.0 ≤ w ∧ w ≤ 78.5) :
    False ∧ ((63.7 < w ∧ w ≤ 65.9) → (w = 64.8)) :=
by
  sorry

end average_weight_estimation_exclude_friend_l124_124018


namespace ball_travel_distance_l124_124734

noncomputable def total_distance : ℝ :=
  200 + (2 * (200 * (1 / 3))) + (2 * (200 * ((1 / 3) ^ 2))) +
  (2 * (200 * ((1 / 3) ^ 3))) + (2 * (200 * ((1 / 3) ^ 4)))

theorem ball_travel_distance :
  total_distance = 397.2 :=
by
  sorry

end ball_travel_distance_l124_124734


namespace no_finite_set_A_exists_l124_124035

theorem no_finite_set_A_exists (A : Set ℕ) (h : Finite A ∧ ∀ a ∈ A, 2 * a ∈ A ∨ a / 3 ∈ A) : False :=
sorry

end no_finite_set_A_exists_l124_124035


namespace sum_of_ages_l124_124719

theorem sum_of_ages (age1 age2 age3 : ℕ) (h : age1 * age2 * age3 = 128) : age1 + age2 + age3 = 18 :=
sorry

end sum_of_ages_l124_124719


namespace only_positive_integer_x_l124_124558

theorem only_positive_integer_x (x : ℕ) (k : ℕ) (h1 : 2 * x + 1 = k^2) (h2 : x > 0) :
  ¬ (∃ y : ℕ, (y >= 2 * x + 2 ∧ y <= 3 * x + 2 ∧ ∃ m : ℕ, y = m^2)) → x = 4 := 
by sorry

end only_positive_integer_x_l124_124558


namespace ones_digit_of_power_35_35_pow_17_17_is_five_l124_124596

theorem ones_digit_of_power_35_35_pow_17_17_is_five :
  (35 ^ (35 * (17 ^ 17))) % 10 = 5 := by
  sorry

end ones_digit_of_power_35_35_pow_17_17_is_five_l124_124596


namespace M_gt_N_l124_124464

theorem M_gt_N (a b : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) :
  let M := a * b
  let N := a + b - 1
  M > N := by
  sorry

end M_gt_N_l124_124464


namespace find_hypotenuse_l124_124615

-- Let a, b be the legs of the right triangle, c be the hypotenuse.
-- Let h be the altitude to the hypotenuse and r be the radius of the inscribed circle.
variable (a b c h r : ℝ)

-- Assume conditions of a right-angled triangle
def right_angled (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2

-- Given the altitude to the hypotenuse
def altitude (h c : ℝ) : Prop :=
  ∃ a b : ℝ, right_angled a b c ∧ h = a * b / c

-- Given the radius of the inscribed circle
def inscribed_radius (r a b c : ℝ) : Prop :=
  r = (a + b - c) / 2

-- The proof problem statement
theorem find_hypotenuse (a b c h r : ℝ) 
  (h_right_angled : right_angled a b c)
  (h_altitude : altitude h c)
  (h_inscribed_radius : inscribed_radius r a b c) : 
  c = 2 * r^2 / (h - 2 * r) :=
  sorry

end find_hypotenuse_l124_124615


namespace cone_volume_ratio_l124_124458

noncomputable def ratio_of_volumes (r h : ℝ) : ℝ :=
  let S1 := r^2 * (2 * Real.pi - 3 * Real.sqrt 3) / 12
  let S2 := r^2 * (10 * Real.pi + 3 * Real.sqrt 3) / 12
  S1 / S2

theorem cone_volume_ratio (r h : ℝ) (hr : 0 < r) (hh : 0 < h) :
  ratio_of_volumes r h = (2 * Real.pi - 3 * Real.sqrt 3) / (10 * Real.pi + 3 * Real.sqrt 3) :=
  sorry

end cone_volume_ratio_l124_124458


namespace linda_loan_interest_difference_l124_124389

theorem linda_loan_interest_difference :
  let P : ℝ := 8000
  let r : ℝ := 0.10
  let t : ℕ := 3
  let n_monthly : ℕ := 12
  let n_annual : ℕ := 1
  let A_monthly : ℝ := P * (1 + r / (n_monthly : ℝ))^(n_monthly * t)
  let A_annual : ℝ := P * (1 + r)^t
  A_monthly - A_annual = 151.07 :=
by
  sorry

end linda_loan_interest_difference_l124_124389


namespace prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l124_124895

-- Definitions of probabilities of making a shot
def p_A : ℚ := 1 / 3
def p_B : ℚ := 1 / 2

-- Number of attempts
def num_attempts : ℕ := 3

-- Probability that B makes at most 2 shots
theorem prob_B_at_most_2_shots : 
  (1 - (num_attempts.choose 3) * (p_B ^ 3) * ((1 - p_B) ^ (num_attempts - 3))) = 7 / 8 :=
by 
  sorry

-- Probability that B makes exactly 2 more shots than A
theorem prob_B_exactly_2_more_than_A : 
  (num_attempts.choose 2) * (p_B ^ 2) * ((1 - p_B) ^ 1) * (num_attempts.choose 0) * ((1 - p_A) ^ num_attempts) +
  (num_attempts.choose 3) * (p_B ^ 3) * (num_attempts.choose 1) * (p_A ^ 1) * ((1 - p_A) ^ (num_attempts - 1)) = 1 / 6 :=
by 
  sorry

end prob_B_at_most_2_shots_prob_B_exactly_2_more_than_A_l124_124895


namespace min_value_of_D_l124_124419

noncomputable def D (x a : ℝ) : ℝ :=
  Real.sqrt ((x - a) ^ 2 + (Real.exp x - 2 * Real.sqrt a) ^ 2) + a + 2

theorem min_value_of_D (e : ℝ) (h_e : e = 2.71828) :
  ∀ a : ℝ, ∃ x : ℝ, D x a = Real.sqrt 2 + 1 :=
sorry

end min_value_of_D_l124_124419


namespace sum_remainders_l124_124938

theorem sum_remainders (a b c : ℕ) (h₁ : a % 30 = 7) (h₂ : b % 30 = 11) (h₃ : c % 30 = 23) : 
  (a + b + c) % 30 = 11 := 
by
  sorry

end sum_remainders_l124_124938


namespace quarters_spent_l124_124467

theorem quarters_spent (original : ℕ) (remaining : ℕ) (q : ℕ) 
  (h1 : original = 760) 
  (h2 : remaining = 342) 
  (h3 : q = original - remaining) : q = 418 := 
by
  sorry

end quarters_spent_l124_124467


namespace exist_identical_2x2_squares_l124_124080

theorem exist_identical_2x2_squares : 
  ∃ sq1 sq2 : Finset (Fin 5 × Fin 5), 
    sq1.card = 4 ∧ sq2.card = 4 ∧ 
    (∀ (i : Fin 5) (j : Fin 5), 
      (i = 0 ∧ j = 0) ∨ (i = 4 ∧ j = 4) → 
      (i, j) ∈ sq1 ∧ (i, j) ∈ sq2 ∧ 
      (sq1 ≠ sq2 → ∃ p ∈ sq1, p ∉ sq2)) :=
sorry

end exist_identical_2x2_squares_l124_124080


namespace min_a2_plus_b2_l124_124504

-- Define circle and line intercept conditions
def circle_center : ℝ × ℝ := (-2, 1)
def circle_radius : ℝ := 2
def line_eq (a b x y : ℝ) : Prop := a * x + 2 * b * y - 4 = 0
def chord_length (chord_len : ℝ) : Prop := chord_len = 4

-- Define the final minimum value to prove
def min_value (a b : ℝ) : ℝ := a^2 + b^2

-- Proving the specific value considering the conditions
theorem min_a2_plus_b2 (a b : ℝ) (h1 : b = a + 2) (h2 : chord_length 4) : min_value a b = 2 := by
  sorry

end min_a2_plus_b2_l124_124504


namespace rectangle_area_l124_124013

theorem rectangle_area (sqr_area : ℕ) (rect_width rect_length : ℕ) (h1 : sqr_area = 25)
    (h2 : rect_width = Int.sqrt sqr_area) (h3 : rect_length = 2 * rect_width) :
    rect_width * rect_length = 50 := by
  sorry

end rectangle_area_l124_124013


namespace min_perimeter_lateral_face_l124_124217

theorem min_perimeter_lateral_face (x h : ℝ) (V : ℝ) (P : ℝ): 
  (x > 0) → (h > 0) → (V = 4) → (V = x^2 * h) → 
  (∀ y : ℝ, y > 0 → 2*y + 2 * (4 / y^2) ≥ P) → P = 6 := 
by
  intro x_pos h_pos volume_eq volume_expr min_condition
  sorry

end min_perimeter_lateral_face_l124_124217


namespace sum_of_common_ratios_l124_124827

theorem sum_of_common_ratios (k p r : ℝ) (h₁ : k ≠ 0) (h₂ : p ≠ r) (h₃ : (k * (p ^ 2)) - (k * (r ^ 2)) = 4 * (k * p - k * r)) : 
  p + r = 4 :=
by
  -- Using the conditions provided, we can prove the sum of the common ratios is 4.
  sorry

end sum_of_common_ratios_l124_124827


namespace sum_of_reciprocals_l124_124882

variable {x y : ℝ}
variable (hx : x + y = 3 * x * y + 2)

theorem sum_of_reciprocals : (1 / x) + (1 / y) = 3 :=
by
  sorry

end sum_of_reciprocals_l124_124882


namespace bob_after_alice_l124_124747

def race_distance : ℕ := 15
def alice_speed : ℕ := 7
def bob_speed : ℕ := 9

def alice_time : ℕ := alice_speed * race_distance
def bob_time : ℕ := bob_speed * race_distance

theorem bob_after_alice : bob_time - alice_time = 30 := by
  sorry

end bob_after_alice_l124_124747


namespace condition_sufficient_not_necessary_l124_124735

theorem condition_sufficient_not_necessary (x : ℝ) :
  (0 < x ∧ x < 2) → (x < 2) ∧ ¬((x < 2) → (0 < x ∧ x < 2)) :=
by
  sorry

end condition_sufficient_not_necessary_l124_124735


namespace simplify_and_evaluate_expression_l124_124266

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = 1 + Real.sqrt 3) :
  ((x + 3) / (x^2 - 2*x + 1) * (x - 1) / (x^2 + 3*x) + 1 / x) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l124_124266


namespace division_remainder_l124_124529

theorem division_remainder : 
  ∀ (Dividend Divisor Quotient Remainder : ℕ), 
  Dividend = 760 → 
  Divisor = 36 → 
  Quotient = 21 → 
  Dividend = (Divisor * Quotient) + Remainder → 
  Remainder = 4 := 
by 
  intros Dividend Divisor Quotient Remainder h1 h2 h3 h4
  subst h1
  subst h2
  subst h3
  have h5 : 760 = 36 * 21 + Remainder := h4
  linarith

end division_remainder_l124_124529


namespace boys_to_girls_ratio_l124_124368

theorem boys_to_girls_ratio (x y : ℕ) 
  (h1 : 149 * x + 144 * y = 147 * (x + y)) : 
  x = (3 / 2 : ℚ) * y :=
by
  sorry

end boys_to_girls_ratio_l124_124368


namespace min_value_of_a_l124_124385

noncomputable def x (t a : ℝ) : ℝ :=
  5 * (t + 1)^2 + a / (t + 1)^5

theorem min_value_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ 2 * Real.sqrt ((24 / 7)^7) :=
sorry

end min_value_of_a_l124_124385


namespace train_length_l124_124372

theorem train_length 
  (L : ℝ) -- Length of each train in meters.
  (speed_fast : ℝ := 56) -- Speed of the faster train in km/hr.
  (speed_slow : ℝ := 36) -- Speed of the slower train in km/hr.
  (time_pass : ℝ := 72) -- Time taken for the faster train to pass the slower train in seconds.
  (km_to_m_s : ℝ := 5 / 18) -- Conversion factor from km/hr to m/s.
  (relative_speed : ℝ := (speed_fast - speed_slow) * km_to_m_s) -- Relative speed in m/s.
  (distance_covered : ℝ := relative_speed * time_pass) -- Distance covered in meters.
  (equal_length : 2 * L = distance_covered) -- Condition of the problem: 2L = distance covered.
  : L = 200.16 :=
sorry

end train_length_l124_124372


namespace intersection_of_sets_l124_124478

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l124_124478


namespace rectangle_side_length_l124_124849

theorem rectangle_side_length (a b c d : ℕ) 
  (h₁ : a = 3) 
  (h₂ : b = 6) 
  (h₃ : a / c = 3 / 4) : 
  c = 4 := 
by
  sorry

end rectangle_side_length_l124_124849


namespace purely_imaginary_l124_124513

theorem purely_imaginary {m : ℝ} (h1 : m^2 - 3 * m = 0) (h2 : m^2 - 5 * m + 6 ≠ 0) : m = 0 :=
sorry

end purely_imaginary_l124_124513


namespace total_num_novels_receiving_prizes_l124_124512

-- Definitions based on conditions
def total_prize_money : ℕ := 800
def first_place_prize : ℕ := 200
def second_place_prize : ℕ := 150
def third_place_prize : ℕ := 120
def remaining_award_amount : ℕ := 22

-- Total number of novels receiving prizes
theorem total_num_novels_receiving_prizes : 
  (3 + (total_prize_money - (first_place_prize + second_place_prize + third_place_prize)) / remaining_award_amount) = 18 :=
by {
  -- We leave the proof as an exercise (denoted by sorry)
  sorry
}

end total_num_novels_receiving_prizes_l124_124512


namespace douglas_votes_in_Y_is_46_l124_124695

variable (V : ℝ)
variable (P : ℝ)

def percentage_won_in_Y :=
  let total_voters_X := 2 * V
  let total_voters_Y := V
  let votes_in_X := 0.64 * total_voters_X
  let votes_in_Y := P / 100 * total_voters_Y
  let total_votes := 1.28 * V + (P / 100 * V)
  let combined_voters := 3 * V
  let combined_votes_percentage := 0.58 * combined_voters
  P = 46

theorem douglas_votes_in_Y_is_46
  (V_pos : V > 0)
  (H : 1.28 * V + (P / 100 * V) = 0.58 * 3 * V) :
  percentage_won_in_Y V P := by
  sorry

end douglas_votes_in_Y_is_46_l124_124695


namespace value_of_expression_in_third_quadrant_l124_124955

theorem value_of_expression_in_third_quadrant (α : ℝ) (h1 : 180 < α ∧ α < 270) :
  (2 * Real.sin α) / Real.sqrt (1 - Real.cos α ^ 2) = -2 := by
  sorry

end value_of_expression_in_third_quadrant_l124_124955


namespace greatest_difference_four_digit_numbers_l124_124784

theorem greatest_difference_four_digit_numbers : 
  ∃ (d1 d2 d3 d4 : ℕ), (d1 = 0 ∨ d1 = 3 ∨ d1 = 4 ∨ d1 = 8) ∧ 
                      (d2 = 0 ∨ d2 = 3 ∨ d2 = 4 ∨ d2 = 8) ∧ 
                      (d3 = 0 ∨ d3 = 3 ∨ d3 = 4 ∨ d3 = 8) ∧ 
                      (d4 = 0 ∨ d4 = 3 ∨ d4 = 4 ∨ d4 = 8) ∧ 
                      d1 ≠ d2 ∧ d1 ≠ d3 ∧ d1 ≠ d4 ∧ 
                      d2 ≠ d3 ∧ d2 ≠ d4 ∧ d3 ≠ d4 ∧ 
                      (∃ n1 n2, n1 = 1000 * 8 + 100 * 4 + 10 * 3 + 0 ∧ 
                                n2 = 1000 * 3 + 100 * 0 + 10 * 4 + 8 ∧ 
                                n1 - n2 = 5382) :=
by {
  sorry
}

end greatest_difference_four_digit_numbers_l124_124784


namespace cosine_eq_one_fifth_l124_124653

theorem cosine_eq_one_fifth {α : ℝ} 
  (h : Real.sin (5 * Real.pi / 2 + α) = 1 / 5) : 
  Real.cos α = 1 / 5 := 
sorry

end cosine_eq_one_fifth_l124_124653


namespace find_y_l124_124068

open Real

structure Vec3 where
  x : ℝ
  y : ℝ
  z : ℝ

def parallel (v₁ v₂ : Vec3) : Prop := ∃ s : ℝ, v₁ = ⟨s * v₂.x, s * v₂.y, s * v₂.z⟩

def orthogonal (v₁ v₂ : Vec3) : Prop := (v₁.x * v₂.x + v₁.y * v₂.y + v₁.z * v₂.z) = 0

noncomputable def correct_y (x y : Vec3) : Vec3 :=
  ⟨(8 : ℝ) - 2 * (2 : ℝ), (-4 : ℝ) - 2 * (2 : ℝ), (2 : ℝ) - 2 * (2 : ℝ)⟩

theorem find_y :
  ∀ (x y : Vec3),
    (x.x + y.x = 8) ∧ (x.y + y.y = -4) ∧ (x.z + y.z = 2) →
    (parallel x ⟨2, 2, 2⟩) →
    (orthogonal y ⟨1, -1, 0⟩) →
    y = ⟨4, -8, -2⟩ :=
by
  intros x y Hxy Hparallel Horthogonal
  sorry

end find_y_l124_124068


namespace calculate_division_l124_124054

theorem calculate_division :
  (- (3 / 4) - 5 / 9 + 7 / 12) / (- 1 / 36) = 26 := by
  sorry

end calculate_division_l124_124054


namespace miniVanTankCapacity_is_65_l124_124646

noncomputable def miniVanTankCapacity : ℝ :=
  let serviceCostPerVehicle := 2.10
  let fuelCostPerLiter := 0.60
  let numMiniVans := 3
  let numTrucks := 2
  let totalCost := 299.1
  let truckFactor := 1.2
  let V := (totalCost - serviceCostPerVehicle * (numMiniVans + numTrucks)) /
            (fuelCostPerLiter * (numMiniVans + numTrucks * (1 + truckFactor)))
  V

theorem miniVanTankCapacity_is_65 : miniVanTankCapacity = 65 :=
  sorry

end miniVanTankCapacity_is_65_l124_124646


namespace cut_scene_length_l124_124896

theorem cut_scene_length (original_length final_length : ℕ) (h1 : original_length = 60) (h2 : final_length = 52) : original_length - final_length = 8 := 
by 
  sorry

end cut_scene_length_l124_124896


namespace opposite_of_three_l124_124316

theorem opposite_of_three : -3 = -3 := 
by sorry

end opposite_of_three_l124_124316


namespace reciprocal_inequality_of_negatives_l124_124522

variable (a b : ℝ)

/-- Given that a < b < 0, prove that 1/a > 1/b. -/
theorem reciprocal_inequality_of_negatives (h1 : a < b) (h2 : b < 0) : (1/a) > (1/b) :=
sorry

end reciprocal_inequality_of_negatives_l124_124522


namespace projection_area_rectangular_board_l124_124754

noncomputable def projection_area (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) : ℝ :=
  let width := AB
  let height := BC
  let shadow_width := 5
  (1 / 2) * (width + shadow_width) * height

theorem projection_area_rectangular_board (AB BC NE MN : ℝ) (ABCD_perp_ground : Prop) (E_mid_AB : Prop) (light_at_M : Prop) :
  AB = 3 → BC = 2 → NE = 3 → MN = 5 → projection_area AB BC NE MN ABCD_perp_ground E_mid_AB light_at_M = 8 :=
by
  intros
  sorry

end projection_area_rectangular_board_l124_124754


namespace sum_consecutive_integers_l124_124362

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l124_124362


namespace sum_of_reciprocals_of_roots_l124_124323

open Real

-- Define the polynomial and its properties using Vieta's formulas
theorem sum_of_reciprocals_of_roots :
  ∀ p q : ℝ, 
  (p + q = 16) ∧ (p * q = 9) → 
  (1 / p + 1 / q = 16 / 9) :=
by
  intros p q h
  let ⟨h1, h2⟩ := h
  sorry

end sum_of_reciprocals_of_roots_l124_124323


namespace oscar_leap_longer_than_elmer_stride_l124_124131

theorem oscar_leap_longer_than_elmer_stride :
  ∀ (elmer_strides_per_gap oscar_leaps_per_gap gaps_between_poles : ℕ)
    (total_distance : ℝ),
  elmer_strides_per_gap = 60 →
  oscar_leaps_per_gap = 16 →
  gaps_between_poles = 60 →
  total_distance = 7920 →
  let elmer_stride_length := total_distance / (elmer_strides_per_gap * gaps_between_poles)
  let oscar_leap_length := total_distance / (oscar_leaps_per_gap * gaps_between_poles)
  oscar_leap_length - elmer_stride_length = 6.05 :=
by
  intros
  sorry

end oscar_leap_longer_than_elmer_stride_l124_124131


namespace sum_of_coefficients_l124_124351

theorem sum_of_coefficients :
  ∃ a b c d e : ℤ, 
    27 * (x : ℝ)^3 + 64 = (a * x + b) * (c * x^2 + d * x + e) ∧ 
    a + b + c + d + e = 20 :=
by
  sorry

end sum_of_coefficients_l124_124351


namespace peter_hunts_3_times_more_than_mark_l124_124495

theorem peter_hunts_3_times_more_than_mark : 
  ∀ (Sam Rob Mark Peter : ℕ),
  Sam = 6 →
  Rob = Sam / 2 →
  Mark = (Sam + Rob) / 3 →
  Sam + Rob + Mark + Peter = 21 →
  Peter = 3 * Mark :=
by
  intros Sam Rob Mark Peter h1 h2 h3 h4
  sorry

end peter_hunts_3_times_more_than_mark_l124_124495


namespace calculate_Delta_l124_124026

-- Define the Delta operation
def Delta (a b : ℚ) : ℚ := (a^2 + b^2) / (1 + a^2 * b^2)

-- Constants for the specific problem
def two := (2 : ℚ)
def three := (3 : ℚ)
def four := (4 : ℚ)

theorem calculate_Delta : Delta (Delta two three) four = 5945 / 4073 := by
  sorry

end calculate_Delta_l124_124026


namespace quadratic_sum_solutions_l124_124445

theorem quadratic_sum_solutions {a b : ℝ} (h : a ≥ b) (h1: a = 1 + Real.sqrt 17) (h2: b = 1 - Real.sqrt 17) :
  3 * a + 2 * b = 5 + Real.sqrt 17 := by
  sorry

end quadratic_sum_solutions_l124_124445


namespace no_solution_system_of_equations_l124_124084

theorem no_solution_system_of_equations :
  ¬ (∃ (x y : ℝ),
    (80 * x + 15 * y - 7) / (78 * x + 12 * y) = 1 ∧
    (2 * x^2 + 3 * y^2 - 11) / (y^2 - x^2 + 3) = 1 ∧
    78 * x + 12 * y ≠ 0 ∧
    y^2 - x^2 + 3 ≠ 0) :=
    by
      sorry

end no_solution_system_of_equations_l124_124084


namespace total_pictures_l124_124012

-- Definitions based on problem conditions
def Randy_pictures : ℕ := 5
def Peter_pictures : ℕ := Randy_pictures + 3
def Quincy_pictures : ℕ := Peter_pictures + 20
def Susan_pictures : ℕ := 2 * Quincy_pictures - 7
def Thomas_pictures : ℕ := Randy_pictures ^ 3

-- The proof statement
theorem total_pictures : Randy_pictures + Peter_pictures + Quincy_pictures + Susan_pictures + Thomas_pictures = 215 := by
  sorry

end total_pictures_l124_124012


namespace certain_positive_integer_value_l124_124342

theorem certain_positive_integer_value :
  ∃ (i m p : ℕ), (x = 2 ^ i * 3 ^ 2 * 5 ^ m * 7 ^ p) ∧ (i + 2 + m + p = 11) :=
by
  let x := 40320 -- 8!
  sorry

end certain_positive_integer_value_l124_124342


namespace remainder_div_3005_95_l124_124184

theorem remainder_div_3005_95 : 3005 % 95 = 60 := 
by {
  sorry
}

end remainder_div_3005_95_l124_124184


namespace tan_arccos_eq_2y_l124_124033

noncomputable def y_squared : ℝ :=
  (-1 + Real.sqrt 17) / 8

theorem tan_arccos_eq_2y (y : ℝ) (hy : 0 < y) (htan : Real.tan (Real.arccos y) = 2 * y) :
  y^2 = y_squared := sorry

end tan_arccos_eq_2y_l124_124033


namespace coffee_blend_l124_124807

variable (pA pB : ℝ) (cA cB : ℝ) (total_cost : ℝ) 

theorem coffee_blend (hA : pA = 4.60) 
                     (hB : pB = 5.95) 
                     (h_ratio : cB = 2 * cA) 
                     (h_total : 4.60 * cA + 5.95 * cB = 511.50) : 
                     cA = 31 := 
by
  sorry

end coffee_blend_l124_124807


namespace andy_late_minutes_l124_124417

theorem andy_late_minutes (school_starts_at : Nat) (normal_travel_time : Nat) 
  (stop_per_light : Nat) (red_lights : Nat) (construction_wait : Nat) 
  (left_house_at : Nat) : 
  let total_delay := (stop_per_light * red_lights) + construction_wait
  let total_travel_time := normal_travel_time + total_delay
  let arrive_time := left_house_at + total_travel_time
  let late_time := arrive_time - school_starts_at
  late_time = 7 :=
by
  sorry

end andy_late_minutes_l124_124417


namespace eat_jar_together_time_l124_124598

-- Define the rate of the child
def child_rate := 1 / 6

-- Define the rate of Karlson who eats twice as fast as the child
def karlson_rate := 2 * child_rate

-- Define the combined rate when both eat together
def combined_rate := child_rate + karlson_rate

-- Prove that the time taken together to eat one jar is 2 minutes
theorem eat_jar_together_time : (1 / combined_rate) = 2 :=
by
  -- Add the proof steps here
  sorry

end eat_jar_together_time_l124_124598


namespace range_of_m_l124_124348

theorem range_of_m (m : ℝ) :
  (∀ P : ℝ × ℝ, P.2 = 2 * P.1 + m → (abs (P.1^2 + (P.2 - 1)^2) = (1/2) * abs (P.1^2 + (P.2 - 4)^2)) → (-2 * Real.sqrt 5) ≤ m ∧ m ≤ (2 * Real.sqrt 5)) :=
sorry

end range_of_m_l124_124348


namespace Felipe_time_to_build_house_l124_124429

variables (E F : ℝ)
variables (Felipe_building_time_months : ℝ) (Combined_time : ℝ := 7.5) (Half_time_relation : F = 1 / 2 * E)

-- Felipe finished his house in 30 months
theorem Felipe_time_to_build_house :
  (F = 1 / 2 * E) →
  (F + E = Combined_time) →
  (Felipe_building_time_months = F * 12) →
  Felipe_building_time_months = 30 :=
by
  intros h1 h2 h3
  -- Combining the given conditions to prove the statement
  sorry

end Felipe_time_to_build_house_l124_124429


namespace total_cost_of_crayons_l124_124007

-- Definition of the initial conditions
def usual_price : ℝ := 2.5
def discount_rate : ℝ := 0.15
def packs_initial : ℕ := 4
def packs_to_buy : ℕ := 2

-- Calculate the discounted price for one pack
noncomputable def discounted_price : ℝ :=
  usual_price - (usual_price * discount_rate)

-- Calculate the total cost of packs after purchase and validate it
theorem total_cost_of_crayons :
  (packs_initial * usual_price) + (packs_to_buy * discounted_price) = 14.25 :=
by
  sorry

end total_cost_of_crayons_l124_124007


namespace triangle_interior_angle_ge_60_l124_124530

theorem triangle_interior_angle_ge_60 (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A < 60) (h3 : B < 60) (h4 : C < 60) : false := 
by
  sorry

end triangle_interior_angle_ge_60_l124_124530


namespace number_divisible_by_k_cube_l124_124194

theorem number_divisible_by_k_cube (k : ℕ) (h : k = 42) : ∃ n, (k^3) % n = 0 ∧ n = 74088 := by
  sorry

end number_divisible_by_k_cube_l124_124194


namespace find_b_l124_124682

variable (a b c : ℕ)

def conditions (a b c : ℕ) : Prop :=
  a = b + 2 ∧ 
  b = 2 * c ∧ 
  a + b + c = 42

theorem find_b (a b c : ℕ) (h : conditions a b c) : b = 16 := 
sorry

end find_b_l124_124682


namespace series_inequality_l124_124147

open BigOperators

theorem series_inequality :
  (∑ k in Finset.range 2012, (1 / (((k + 1) * Real.sqrt k) + (k * Real.sqrt (k + 1))))) > 0.97 :=
sorry

end series_inequality_l124_124147


namespace pyramid_height_correct_l124_124283

noncomputable def pyramid_height : ℝ :=
  let ab := 15 * Real.sqrt 3
  let bc := 14 * Real.sqrt 3
  let base_area := ab * bc
  let volume := 750
  let height := 3 * volume / base_area
  height

theorem pyramid_height_correct : pyramid_height = 25 / 7 :=
by
  sorry

end pyramid_height_correct_l124_124283


namespace initial_shipment_robot_rascals_l124_124179

theorem initial_shipment_robot_rascals 
(T : ℝ) 
(h1 : (0.7 * T = 168)) : 
  T = 240 :=
sorry

end initial_shipment_robot_rascals_l124_124179


namespace girls_more_than_boys_l124_124537

variables (B G : ℕ)
def ratio_condition : Prop := 3 * G = 4 * B
def total_students_condition : Prop := B + G = 49

theorem girls_more_than_boys
  (h1 : ratio_condition B G)
  (h2 : total_students_condition B G) :
  G = B + 7 :=
sorry

end girls_more_than_boys_l124_124537


namespace depth_notation_l124_124715

-- Definition of depth and altitude
def above_sea_level (h : ℝ) : Prop := h > 0 
def below_sea_level (h : ℝ) : Prop := h < 0 

-- Given conditions
axiom height_Dabaijing : above_sea_level 9050
axiom depth_Haidou1 : below_sea_level (-10907)

-- Proof goal
theorem depth_notation :
  ∀ (d : ℝ), above_sea_level 9050 → below_sea_level (-d) → d = 10907 :=
by
  intros d _ _
  exact sorry

end depth_notation_l124_124715


namespace inequality_not_always_true_l124_124222

theorem inequality_not_always_true {a b c : ℝ}
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a > b) (h₄ : c ≠ 0) : ¬ ∀ c : ℝ, (a / c > b / c) :=
by
  sorry

end inequality_not_always_true_l124_124222


namespace arithmetic_geom_seq_a5_l124_124684

theorem arithmetic_geom_seq_a5 (a : ℕ → ℝ) (s : ℕ → ℝ) (q : ℝ)
  (a1 : a 1 = 1)
  (S8 : s 8 = 17 * s 4) :
  a 5 = 16 :=
sorry

end arithmetic_geom_seq_a5_l124_124684


namespace prob_equal_even_odd_dice_l124_124809

def even_number_probability : ℚ := 1 / 2
def odd_number_probability : ℚ := 1 / 2

def probability_equal_even_odd (n : ℕ) (k : ℕ) : ℚ :=
  (Nat.choose n k) * (even_number_probability) ^ n

theorem prob_equal_even_odd_dice : 
  probability_equal_even_odd 8 4 = 35 / 128 :=
by
  sorry

end prob_equal_even_odd_dice_l124_124809


namespace probability_10_or_9_probability_at_least_7_l124_124902

-- Define the probabilities of hitting each ring
def p_10 : ℝ := 0.1
def p_9 : ℝ := 0.2
def p_8 : ℝ := 0.3
def p_7 : ℝ := 0.3
def p_below_7 : ℝ := 0.1

-- Define the events as their corresponding probabilities
def P_A : ℝ := p_10 -- Event of hitting the 10 ring
def P_B : ℝ := p_9 -- Event of hitting the 9 ring
def P_C : ℝ := p_8 -- Event of hitting the 8 ring
def P_D : ℝ := p_7 -- Event of hitting the 7 ring
def P_E : ℝ := p_below_7 -- Event of hitting below the 7 ring

-- Since the probabilities must sum to 1, we have the following fact about their sum
-- P_A + P_B + P_C + P_D + P_E = 1

theorem probability_10_or_9 : P_A + P_B = 0.3 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

theorem probability_at_least_7 : P_A + P_B + P_C + P_D = 0.9 :=
by 
  -- This would be filled in with the proof steps or assumptions
  sorry

end probability_10_or_9_probability_at_least_7_l124_124902


namespace repeating_decimal_division_l124_124165

-- Define x and y as the repeating decimals.
noncomputable def x : ℚ := 84 / 99
noncomputable def y : ℚ := 21 / 99

-- Proof statement of the equivalence.
theorem repeating_decimal_division : (x / y) = 4 := by
  sorry

end repeating_decimal_division_l124_124165


namespace rectangle_ratio_l124_124493

noncomputable def ratio_of_sides (a b : ℝ) : ℝ := a / b

theorem rectangle_ratio (a b d : ℝ) (h1 : d = Real.sqrt (a^2 + b^2)) (h2 : (a/b)^2 = b/d) : 
  ratio_of_sides a b = (Real.sqrt 5 - 1) / 3 :=
by sorry

end rectangle_ratio_l124_124493


namespace largest_whole_number_l124_124679

theorem largest_whole_number (x : ℕ) (h : 6 * x + 3 < 150) : x ≤ 24 :=
sorry

end largest_whole_number_l124_124679


namespace woman_first_half_speed_l124_124070

noncomputable def first_half_speed (total_time : ℕ) (second_half_speed : ℕ) (total_distance : ℕ) : ℕ :=
  let first_half_distance := total_distance / 2
  let second_half_distance := total_distance / 2
  let second_half_time := second_half_distance / second_half_speed
  let first_half_time := total_time - second_half_time
  first_half_distance / first_half_time

theorem woman_first_half_speed : first_half_speed 20 24 448 = 21 := by
  sorry

end woman_first_half_speed_l124_124070


namespace find_inequality_solution_set_l124_124687

noncomputable def inequality_solution_set : Set ℝ :=
  { x | (1 / (x * (x + 1))) - (1 / ((x + 1) * (x + 2))) < (1 / 4) }

theorem find_inequality_solution_set :
  inequality_solution_set = { x : ℝ | x < -2 } ∪ { x : ℝ | -1 < x ∧ x < 0 } ∪ { x : ℝ | 1 < x } :=
by
  sorry

end find_inequality_solution_set_l124_124687


namespace consecutive_composite_numbers_bound_l124_124298

theorem consecutive_composite_numbers_bound (n : ℕ) (hn: 0 < n) :
  ∃ (seq : Fin n → ℕ), (∀ i, ¬ Nat.Prime (seq i)) ∧ (∀ i, seq i < 4^(n+1)) :=
sorry

end consecutive_composite_numbers_bound_l124_124298


namespace garden_breadth_l124_124628

theorem garden_breadth (P L B : ℕ) (h₁ : P = 950) (h₂ : L = 375) (h₃ : P = 2 * (L + B)) : B = 100 := by
  sorry

end garden_breadth_l124_124628


namespace min_value_x_y_l124_124970

theorem min_value_x_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 4/y = 1) : x + y ≥ 9 :=
sorry

end min_value_x_y_l124_124970


namespace diff_of_squares_expression_l124_124724

theorem diff_of_squares_expression (m n : ℝ) :
  (3 * m + n) * (3 * m - n) = (3 * m)^2 - n^2 :=
by
  sorry

end diff_of_squares_expression_l124_124724


namespace minutkin_bedtime_l124_124531

def time_minutkin_goes_to_bed 
    (morning_time : ℕ) 
    (morning_turns : ℕ) 
    (night_turns : ℕ) 
    (morning_hours : ℕ) 
    (morning_minutes : ℕ)
    (hours_per_turn : ℕ) 
    (minutes_per_turn : ℕ) : Nat := 
    ((morning_hours * 60 + morning_minutes) - (night_turns * hours_per_turn * 60 + night_turns * minutes_per_turn)) % 1440 

theorem minutkin_bedtime : 
    time_minutkin_goes_to_bed 9 9 11 8 30 1 12 = 1290 :=
    sorry

end minutkin_bedtime_l124_124531


namespace compute_m_n_sum_l124_124472

theorem compute_m_n_sum :
  let AB := 10
  let BC := 15
  let height := 30
  let volume_ratio := 9
  let smaller_base_AB := AB / 3
  let smaller_base_BC := BC / 3
  let diagonal_AC := Real.sqrt (AB^2 + BC^2)
  let smaller_diagonal_A'C' := Real.sqrt ((smaller_base_AB)^2 + (smaller_base_BC)^2)
  let y_length := 145 / 9   -- derived from geometric considerations
  let YU := 20 + y_length
  let m := 325
  let n := 9
  YU = m / n ∧ Nat.gcd m n = 1 ∧ m + n = 334 :=
  by
  sorry

end compute_m_n_sum_l124_124472


namespace Tom_search_cost_l124_124211

theorem Tom_search_cost (first_5_days_rate: ℕ) (first_5_days: ℕ) (remaining_days_rate: ℕ) (total_days: ℕ) : 
  first_5_days_rate = 100 → 
  first_5_days = 5 → 
  remaining_days_rate = 60 → 
  total_days = 10 → 
  (first_5_days * first_5_days_rate + (total_days - first_5_days) * remaining_days_rate) = 800 := 
by 
  intros h1 h2 h3 h4 
  sorry

end Tom_search_cost_l124_124211


namespace quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l124_124155

theorem quadratic_has_negative_root_sufficiency 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) → (a < 0) :=
sorry

theorem quadratic_has_negative_root_necessity 
  (a : ℝ) : (∃ x : ℝ, x < 0 ∧ a * x^2 + 2 * x + 1 = 0) ↔ (a < 0) :=
sorry

end quadratic_has_negative_root_sufficiency_quadratic_has_negative_root_necessity_l124_124155


namespace time_to_cross_bridge_l124_124652

theorem time_to_cross_bridge (speed_km_hr : ℝ) (length_m : ℝ) (speed_conversion_factor : ℝ) (time_conversion_factor : ℝ) (expected_time : ℝ) :
  speed_km_hr = 5 →
  length_m = 1250 →
  speed_conversion_factor = 1000 →
  time_conversion_factor = 60 →
  expected_time = length_m / (speed_km_hr * (speed_conversion_factor / time_conversion_factor)) →
  expected_time = 15 :=
by
  intros
  sorry

end time_to_cross_bridge_l124_124652


namespace tom_annual_car_leasing_cost_l124_124564

theorem tom_annual_car_leasing_cost :
  let miles_mwf := 50 * 3  -- Miles driven on Monday, Wednesday, and Friday
  let miles_other_days := 100 * 4 -- Miles driven on the other days (Sunday, Tuesday, Thursday, Saturday)
  let weekly_miles := miles_mwf + miles_other_days -- Total miles driven per week

  let cost_per_mile := 0.1 -- Cost per mile
  let weekly_fee := 100 -- Weekly fee

  let weekly_cost := weekly_miles * cost_per_mile + weekly_fee -- Total weekly cost

  let weeks_per_year := 52
  let annual_cost := weekly_cost * weeks_per_year -- Annual cost

  annual_cost = 8060 :=
by
  sorry

end tom_annual_car_leasing_cost_l124_124564
