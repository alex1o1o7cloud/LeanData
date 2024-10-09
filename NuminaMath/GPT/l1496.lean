import Mathlib

namespace amina_wins_is_21_over_32_l1496_149615

/--
Amina and Bert alternate turns tossing a fair coin. Amina goes first and each player takes three turns.
The first player to toss a tail wins. If neither Amina nor Bert tosses a tail, then neither wins.
Prove that the probability that Amina wins is \( \frac{21}{32} \).
-/
def amina_wins_probability : ℚ :=
  let p_first_turn := 1 / 2
  let p_second_turn := (1 / 2) ^ 3
  let p_third_turn := (1 / 2) ^ 5
  p_first_turn + p_second_turn + p_third_turn

theorem amina_wins_is_21_over_32 :
  amina_wins_probability = 21 / 32 :=
sorry

end amina_wins_is_21_over_32_l1496_149615


namespace find_first_number_l1496_149673

theorem find_first_number (x : ℝ) : 
  (20 + 40 + 60) / 3 = (x + 60 + 35) / 3 + 5 → 
  x = 10 := 
by 
  sorry

end find_first_number_l1496_149673


namespace find_sum_3xyz_l1496_149656

variables (x y z : ℚ)

def equation1 : Prop := y + z = 18 - 4 * x
def equation2 : Prop := x + z = 16 - 4 * y
def equation3 : Prop := x + y = 9 - 4 * z

theorem find_sum_3xyz (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) : 
  3 * x + 3 * y + 3 * z = 43 / 2 := 
sorry

end find_sum_3xyz_l1496_149656


namespace gcd_45_75_eq_15_l1496_149664

theorem gcd_45_75_eq_15 : Nat.gcd 45 75 = 15 := by
  sorry

end gcd_45_75_eq_15_l1496_149664


namespace father_children_problem_l1496_149639

theorem father_children_problem {F C n : ℕ} 
  (hF_C : F = C) 
  (sum_ages_after_15_years : C + 15 * n = 2 * (F + 15)) 
  (father_age : F = 75) : 
  n = 7 :=
by
  sorry

end father_children_problem_l1496_149639


namespace handshake_count_l1496_149623

theorem handshake_count
  (total_people : ℕ := 40)
  (groupA_size : ℕ := 30)
  (groupB_size : ℕ := 10)
  (groupB_knowsA_5 : ℕ := 3)
  (groupB_knowsA_0 : ℕ := 7)
  (handshakes_between_A_and_B5 : ℕ := groupB_knowsA_5 * (groupA_size - 5))
  (handshakes_between_A_and_B0 : ℕ := groupB_knowsA_0 * groupA_size)
  (handshakes_within_B : ℕ := groupB_size * (groupB_size - 1) / 2) :
  handshakes_between_A_and_B5 + handshakes_between_A_and_B0 + handshakes_within_B = 330 :=
sorry

end handshake_count_l1496_149623


namespace exists_positive_m_dividing_f_100_l1496_149699

noncomputable def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_positive_m_dividing_f_100:
  ∃ (m : ℤ), m > 0 ∧ 19881 ∣ (3^100 * (m + 1) - 1) :=
by
  sorry

end exists_positive_m_dividing_f_100_l1496_149699


namespace total_marbles_l1496_149694

theorem total_marbles (bowl2_capacity : ℕ) (h₁ : bowl2_capacity = 600)
    (h₂ : 3 / 4 * bowl2_capacity = 450) : 600 + (3 / 4 * 600) = 1050 := by
  sorry

end total_marbles_l1496_149694


namespace Kyle_older_than_Julian_l1496_149613

variable (Tyson_age : ℕ)
variable (Frederick_age Julian_age Kyle_age : ℕ)

-- Conditions
def condition1 := Tyson_age = 20
def condition2 := Frederick_age = 2 * Tyson_age
def condition3 := Julian_age = Frederick_age - 20
def condition4 := Kyle_age = 25

-- The proof problem (statement only)
theorem Kyle_older_than_Julian :
  Tyson_age = 20 ∧
  Frederick_age = 2 * Tyson_age ∧
  Julian_age = Frederick_age - 20 ∧
  Kyle_age = 25 →
  Kyle_age - Julian_age = 5 := by
  intro h
  sorry

end Kyle_older_than_Julian_l1496_149613


namespace ramu_selling_price_l1496_149630

theorem ramu_selling_price (P R : ℝ) (profit_percent : ℝ) 
  (P_def : P = 42000)
  (R_def : R = 13000)
  (profit_percent_def : profit_percent = 17.272727272727273) :
  let total_cost := P + R
  let selling_price := total_cost * (1 + (profit_percent / 100))
  selling_price = 64500 := 
by
  sorry

end ramu_selling_price_l1496_149630


namespace combined_area_ratio_l1496_149678

theorem combined_area_ratio (s : ℝ) (h₁ : s > 0) : 
  let r := s / 2
  let area_semicircle := (1/2) * π * r^2
  let area_quarter_circle := (1/4) * π * r^2
  let area_square := s^2
  let combined_area := area_semicircle + area_quarter_circle
  let ratio := combined_area / area_square
  ratio = 3 * π / 16 :=
by
  sorry

end combined_area_ratio_l1496_149678


namespace solution_set_of_inequality_l1496_149602

variable (f : ℝ → ℝ)

def g (x : ℝ) : ℝ := f x - x - 1

theorem solution_set_of_inequality (h₁ : f 1 = 2) (h₂ : ∀ x, (deriv f x) < 1) :
  { x : ℝ | f x < x + 1 } = { x | 1 < x } :=
by
  sorry

end solution_set_of_inequality_l1496_149602


namespace opposite_neg_inv_three_l1496_149601

noncomputable def neg_inv_three : ℚ := -1 / 3
noncomputable def pos_inv_three : ℚ := 1 / 3

theorem opposite_neg_inv_three :
  -neg_inv_three = pos_inv_three :=
by
  sorry

end opposite_neg_inv_three_l1496_149601


namespace arithmetic_sequence_100th_term_l1496_149686

theorem arithmetic_sequence_100th_term (a b : ℤ)
  (h1 : 2 * a - a = a) -- definition of common difference d where d = a
  (h2 : b - 2 * a = a) -- b = 3a
  (h3 : a - 6 - b = -2 * a - 6) -- consistency of fourth term
  (h4 : 6 * a = -6) -- equation to solve for a
  : (a + 99 * (2 * a - a)) = -100 := 
sorry

end arithmetic_sequence_100th_term_l1496_149686


namespace A_form_k_l1496_149659

theorem A_form_k (m n : ℕ) (h_m : 2 ≤ m) (h_n : 2 ≤ n) :
  ∃ k : ℕ, (A : ℝ) = (n + Real.sqrt (n^2 - 4)) / 2 ^ m → A = (k + Real.sqrt (k^2 - 4)) / 2 :=
by
  sorry

end A_form_k_l1496_149659


namespace gcd_of_three_numbers_l1496_149627

theorem gcd_of_three_numbers (a b c d : ℕ) (ha : a = 72) (hb : b = 120) (hc : c = 168) (hd : d = 24) : 
  Nat.gcd (Nat.gcd a b) c = d :=
by
  rw [ha, hb, hc, hd]
  -- Placeholder for the actual proof
  exact sorry

end gcd_of_three_numbers_l1496_149627


namespace number_of_measures_of_C_l1496_149679

theorem number_of_measures_of_C (C D : ℕ) (h1 : C + D = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ C = k * D) : 
  ∃ n : ℕ, n = 17 :=
by
  sorry

end number_of_measures_of_C_l1496_149679


namespace range_of_m_l1496_149645

namespace ProofProblem

-- Define propositions P and Q in Lean
def P (m : ℝ) : Prop := 2 * m > 1
def Q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + m * x + 1 ≥ 0

-- Assumptions
variables (m : ℝ)
axiom hP_or_Q : P m ∨ Q m
axiom hP_and_Q_false : ¬(P m ∧ Q m)

-- We need to prove the range of m
theorem range_of_m : m ∈ (Set.Icc (-2 : ℝ) (1 / 2 : ℝ) ∪ Set.Ioi (2 : ℝ)) :=
sorry

end ProofProblem

end range_of_m_l1496_149645


namespace part1_part2_l1496_149618

-- Definitions
def A (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b - 2 * a - 1
def B (a b : ℝ) : ℝ := -a^2 + a * b + a + 3

-- First proof: When a = -1 and b = 10, prove 4A - (3A - 2B) = -45
theorem part1 : 4 * A (-1) 10 - (3 * A (-1) 10 - 2 * B (-1) 10) = -45 := by
  sorry

-- Second proof: If a and b are reciprocal, prove 4A - (3A - 2B) = 10
theorem part2 (a b : ℝ) (hab : a * b = 1) : 4 * A a b - (3 * A a b - 2 * B a b) = 10 := by
  sorry

end part1_part2_l1496_149618


namespace number_of_friends_is_five_l1496_149668

def total_cards : ℕ := 455
def cards_per_friend : ℕ := 91

theorem number_of_friends_is_five (n : ℕ) (h : total_cards = n * cards_per_friend) : n = 5 := 
sorry

end number_of_friends_is_five_l1496_149668


namespace point_M_coordinates_l1496_149644

theorem point_M_coordinates :
  ∃ M : ℝ × ℝ × ℝ, 
    M.1 = 0 ∧ M.2.1 = 0 ∧  
    (dist (1, 0, 2) (M.1, M.2.1, M.2.2) = dist (1, -3, 1) (M.1, M.2.1, M.2.2)) ∧ 
    M = (0, 0, -3) :=
by
  sorry

end point_M_coordinates_l1496_149644


namespace multiply_powers_l1496_149692

theorem multiply_powers (x : ℝ) : x^3 * x^3 = x^6 :=
by sorry

end multiply_powers_l1496_149692


namespace max_min_x2_min_xy_plus_y2_l1496_149652

theorem max_min_x2_min_xy_plus_y2 (x y : ℝ) (h : x^2 + x * y + y^2 = 3) :
  1 ≤ x^2 - x * y + y^2 ∧ x^2 - x * y + y^2 ≤ 9 :=
by sorry

end max_min_x2_min_xy_plus_y2_l1496_149652


namespace bottles_left_l1496_149672

theorem bottles_left (total_bottles : ℕ) (bottles_per_day : ℕ) (days : ℕ)
  (h_total : total_bottles = 264)
  (h_bottles_per_day : bottles_per_day = 15)
  (h_days : days = 11) :
  total_bottles - bottles_per_day * days = 99 :=
by
  sorry

end bottles_left_l1496_149672


namespace arrangement_count_BANANA_l1496_149655

theorem arrangement_count_BANANA : 
  let letters := ['B', 'A', 'N', 'A', 'N', 'A']
  let n := letters.length
  let a_count := letters.count ('A')
  let n_count := letters.count ('N')
  let unique_arrangements := n.factorial / (a_count.factorial * n_count.factorial)
  unique_arrangements = 60 :=
by
  sorry

end arrangement_count_BANANA_l1496_149655


namespace boat_speed_still_water_l1496_149646

-- Define the conditions
def speed_of_stream : ℝ := 4
def distance_downstream : ℕ := 68
def time_downstream : ℕ := 4

-- State the theorem
theorem boat_speed_still_water : 
  ∃V_b : ℝ, distance_downstream = (V_b + speed_of_stream) * time_downstream ∧ V_b = 13 :=
by 
  sorry

end boat_speed_still_water_l1496_149646


namespace opposite_of_negative_six_is_six_l1496_149617

theorem opposite_of_negative_six_is_six : ∀ (x : ℤ), (-6 + x = 0) → x = 6 :=
by
  intro x hx
  sorry

end opposite_of_negative_six_is_six_l1496_149617


namespace number_of_perfect_square_factors_450_l1496_149660

noncomputable def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def prime_factorization_450 := (2, 1) :: (3, 2) :: (5, 2) :: []

def perfect_square_factors (n : ℕ) : ℕ :=
  if n = 450 then 4 else 0

theorem number_of_perfect_square_factors_450 : perfect_square_factors 450 = 4 :=
by
  sorry

end number_of_perfect_square_factors_450_l1496_149660


namespace expression_value_l1496_149604

variables {a b c : ℝ}

theorem expression_value (h : a * b + b * c + c * a = 3) :
  (a * (b^2 + 3) / (a + b)) + (b * (c^2 + 3) / (b + c)) + (c * (a^2 + 3) / (c + a)) = 6 := 
  sorry

end expression_value_l1496_149604


namespace percentage_increase_l1496_149622

theorem percentage_increase (P Q R : ℝ) (x y : ℝ) 
  (h1 : P > 0) (h2 : Q > 0) (h3 : R > 0)
  (h4 : P = (1 + x / 100) * Q)
  (h5 : Q = (1 + y / 100) * R)
  (h6 : P = 2.4 * R) :
  x + y = 140 :=
sorry

end percentage_increase_l1496_149622


namespace probability_of_selecting_one_painted_face_and_one_unpainted_face_l1496_149671

noncomputable def probability_of_specific_selection :
  ℕ → ℕ → ℕ → ℚ
| total_cubes, painted_face_cubes, unpainted_face_cubes =>
  let total_pairs := (total_cubes * (total_cubes - 1)) / 2
  let success_pairs := painted_face_cubes * unpainted_face_cubes
  success_pairs / total_pairs

theorem probability_of_selecting_one_painted_face_and_one_unpainted_face :
  probability_of_specific_selection 36 13 17 = 221 / 630 :=
by
  sorry

end probability_of_selecting_one_painted_face_and_one_unpainted_face_l1496_149671


namespace problem_1_problem_2_l1496_149662

-- Define sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2 * x - 8 = 0}

-- First problem statement
theorem problem_1 (a : ℝ) : (A a ∩ B = A a ∪ B) → a = 5 :=
by
  -- proof omitted
  sorry

-- Second problem statement
theorem problem_2 (a : ℝ) : (∅ ⊆ A a ∩ B) ∧ (A a ∩ C = ∅) → a = -2 :=
by
  -- proof omitted
  sorry

end problem_1_problem_2_l1496_149662


namespace distinct_solutions_square_difference_l1496_149680

theorem distinct_solutions_square_difference 
  (Φ φ : ℝ) (h1 : Φ^2 = Φ + 2) (h2 : φ^2 = φ + 2) (h_distinct : Φ ≠ φ) :
  (Φ - φ)^2 = 9 :=
  sorry

end distinct_solutions_square_difference_l1496_149680


namespace find_value_of_z_l1496_149650

theorem find_value_of_z (z : ℂ) (h1 : ∀ a : ℝ, z = a * I) (h2 : ((z + 2) / (1 - I)).im = 0) : z = -2 * I :=
sorry

end find_value_of_z_l1496_149650


namespace polynomial_possible_integer_roots_l1496_149624

theorem polynomial_possible_integer_roots (b1 b2 : ℤ) :
  ∀ x : ℤ, (x ∣ 18) ↔ (x^3 + b2 * x^2 + b1 * x + 18 = 0) → 
  x = -18 ∨ x = -9 ∨ x = -6 ∨ x = -3 ∨ x = -2 ∨ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 6 ∨ x = 9 ∨ x = 18 :=
by {
  sorry
}


end polynomial_possible_integer_roots_l1496_149624


namespace exists_consecutive_integers_sum_cube_l1496_149690

theorem exists_consecutive_integers_sum_cube :
  ∃ (n : ℤ), ∃ (k : ℤ), 1981 * (n + 990) = k^3 :=
by
  sorry

end exists_consecutive_integers_sum_cube_l1496_149690


namespace proof_f_f_2008_eq_2008_l1496_149638

-- Define the function f
axiom f : ℝ → ℝ

-- The conditions given in the problem
axiom odd_f : ∀ x, f (-x) = -f x
axiom periodic_f : ∀ x, f (x + 6) = f x
axiom f_at_4 : f 4 = -2008

-- The goal to prove
theorem proof_f_f_2008_eq_2008 : f (f 2008) = 2008 :=
by
  sorry

end proof_f_f_2008_eq_2008_l1496_149638


namespace milkman_A_rent_share_l1496_149631

theorem milkman_A_rent_share : 
  let A_cows := 24
  let A_months := 3
  let B_cows := 10
  let B_months := 5
  let C_cows := 35
  let C_months := 4
  let D_cows := 21
  let D_months := 3
  let total_rent := 3250
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let fraction_A := A_cow_months / total_cow_months
  let A_rent_share := total_rent * fraction_A
  A_rent_share = 720 := 
by
  sorry

end milkman_A_rent_share_l1496_149631


namespace snail_stops_at_25_26_l1496_149696

def grid_width : ℕ := 300
def grid_height : ℕ := 50

def initial_position : ℕ × ℕ := (1, 1)

def snail_moves_in_spiral (w h : ℕ) (initial : ℕ × ℕ) : ℕ × ℕ := (25, 26)

theorem snail_stops_at_25_26 :
  snail_moves_in_spiral grid_width grid_height initial_position = (25, 26) :=
sorry

end snail_stops_at_25_26_l1496_149696


namespace removed_cubes_total_l1496_149682

-- Define the large cube composed of 125 smaller cubes (5x5x5 cube)
def large_cube := 5 * 5 * 5

-- Number of smaller cubes removed from each face to opposite face
def removed_faces := (5 * 5 + 5 * 5 + 5 * 3)

-- Overlapping cubes deducted
def overlapping_cubes := (3 + 1)

-- Final number of removed smaller cubes
def removed_total := removed_faces - overlapping_cubes

-- Lean theorem statement
theorem removed_cubes_total : removed_total = 49 :=
by
  -- Definitions provided above imply the theorem
  sorry

end removed_cubes_total_l1496_149682


namespace sin_690_l1496_149629

-- Defining the known conditions as hypotheses:
axiom sin_periodic (x : ℝ) : Real.sin (x + 360) = Real.sin x
axiom sin_odd (x : ℝ) : Real.sin (-x) = - Real.sin x
axiom sin_thirty : Real.sin 30 = 1 / 2

theorem sin_690 : Real.sin 690 = -1 / 2 :=
by
  -- Proof would go here, but it is skipped with sorry.
  sorry

end sin_690_l1496_149629


namespace schedule_problem_l1496_149637

def num_schedule_ways : Nat :=
  -- total ways to pick 3 out of 6 periods and arrange 3 courses
  let total_ways := Nat.choose 6 3 * Nat.factorial 3
  -- at least two consecutive courses (using Principle of Inclusion and Exclusion)
  let two_consecutive := 5 * 6 * 4
  let three_consecutive := 4 * 6
  let invalid_ways := two_consecutive + three_consecutive
  total_ways - invalid_ways

theorem schedule_problem (h : num_schedule_ways = 24) : num_schedule_ways = 24 := by {
  exact h
}

end schedule_problem_l1496_149637


namespace sequence_general_term_l1496_149628

-- Define the sequence using a recurrence relation for clarity in formal proof
def a (n : ℕ) : ℕ :=
  if h : n > 0 then 2^n + 1 else 3

theorem sequence_general_term :
  ∀ n : ℕ, n > 0 → a n = 2^n + 1 := 
by 
  sorry

end sequence_general_term_l1496_149628


namespace initial_red_balloons_l1496_149649

variable (initial_red : ℕ)
variable (given_away : ℕ := 24)
variable (left_with : ℕ := 7)

theorem initial_red_balloons : initial_red = given_away + left_with :=
by sorry

end initial_red_balloons_l1496_149649


namespace most_reasonable_sampling_method_l1496_149691

-- Definitions for the conditions
def significant_difference_by_stage : Prop := 
  -- There is a significant difference in vision condition at different educational stages
  sorry

def no_significant_difference_by_gender : Prop :=
  -- There is no significant difference in vision condition between male and female students
  sorry

-- Theorem statement
theorem most_reasonable_sampling_method 
  (h1 : significant_difference_by_stage) 
  (h2 : no_significant_difference_by_gender) : 
  -- The most reasonable sampling method is stratified sampling by educational stage
  sorry :=
by
  -- Proof skipped
  sorry

end most_reasonable_sampling_method_l1496_149691


namespace tank_capacity_is_24_l1496_149614

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l1496_149614


namespace common_chord_equation_l1496_149625

def circle1 (x y : ℝ) := x^2 + y^2 + 2*x + 2*y - 8 = 0
def circle2 (x y : ℝ) := x^2 + y^2 - 2*x + 10*y - 24 = 0

theorem common_chord_equation :
  ∃ (A B : ℝ × ℝ), circle1 A.1 A.2 ∧ circle2 A.1 A.2 ∧ circle1 B.1 B.2 ∧ circle2 B.1 B.2 ∧
                     ∀ (x y : ℝ), (x - 2*y + 4 = 0) ↔ ((x, y) = A ∨ (x, y) = B) :=
by
  sorry

end common_chord_equation_l1496_149625


namespace minimal_pyramid_height_l1496_149612

theorem minimal_pyramid_height (r x a : ℝ) (h₁ : 0 < r) (h₂ : a = 2 * r * x / (x - r)) (h₃ : x > 4 * r) :
  x = (6 + 2 * Real.sqrt 3) * r :=
by
  -- Proof steps would go here
  sorry

end minimal_pyramid_height_l1496_149612


namespace toy_store_fraction_l1496_149681

theorem toy_store_fraction
  (allowance : ℝ) (arcade_fraction : ℝ) (candy_store_amount : ℝ)
  (h1 : allowance = 1.50)
  (h2 : arcade_fraction = 3 / 5)
  (h3 : candy_store_amount = 0.40) :
  (0.60 - candy_store_amount) / (allowance - arcade_fraction * allowance) = 1 / 3 :=
by
  -- We're skipping the actual proof steps
  sorry

end toy_store_fraction_l1496_149681


namespace find_number_of_toonies_l1496_149642

variable (L T : ℕ)

def condition1 : Prop := L + T = 10
def condition2 : Prop := L + 2 * T = 14

theorem find_number_of_toonies (h1 : condition1 L T) (h2 : condition2 L T) : T = 4 :=
by
  sorry

end find_number_of_toonies_l1496_149642


namespace sum_of_special_multiples_l1496_149685

def smallest_two_digit_multiple_of_5 : ℕ := 10
def smallest_three_digit_multiple_of_7 : ℕ := 105

theorem sum_of_special_multiples :
  smallest_two_digit_multiple_of_5 + smallest_three_digit_multiple_of_7 = 115 :=
by
  sorry

end sum_of_special_multiples_l1496_149685


namespace sandy_marks_per_correct_sum_l1496_149621

theorem sandy_marks_per_correct_sum 
  (total_sums : ℕ)
  (total_marks : ℤ)
  (correct_sums : ℕ)
  (marks_per_incorrect_sum : ℤ)
  (marks_obtained : ℤ) 
  (marks_per_correct_sum : ℕ) :
  total_sums = 30 →
  total_marks = 45 →
  correct_sums = 21 →
  marks_per_incorrect_sum = 2 →
  marks_obtained = total_marks →
  marks_obtained = marks_per_correct_sum * correct_sums - marks_per_incorrect_sum * (total_sums - correct_sums) → 
  marks_per_correct_sum = 3 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end sandy_marks_per_correct_sum_l1496_149621


namespace chef_made_10_cakes_l1496_149657

-- Definitions based on the conditions
def total_eggs : ℕ := 60
def eggs_in_fridge : ℕ := 10
def eggs_per_cake : ℕ := 5

-- Calculated values based on the definitions
def eggs_for_cakes : ℕ := total_eggs - eggs_in_fridge
def number_of_cakes : ℕ := eggs_for_cakes / eggs_per_cake

-- Theorem to prove
theorem chef_made_10_cakes : number_of_cakes = 10 := by
  sorry

end chef_made_10_cakes_l1496_149657


namespace three_digit_numbers_sorted_desc_l1496_149677

theorem three_digit_numbers_sorted_desc :
  ∃ n, n = 84 ∧
    ∀ (h t u : ℕ), 100 <= 100 * h + 10 * t + u ∧ 100 * h + 10 * t + u <= 999 →
    1 ≤ h ∧ h ≤ 9 ∧ 0 ≤ t ∧ t ≤ 9 ∧ 0 ≤ u ∧ u ≤ 9 ∧ h > t ∧ t > u → 
    n = 84 := 
by
  sorry

end three_digit_numbers_sorted_desc_l1496_149677


namespace tree_height_equation_l1496_149609

theorem tree_height_equation (x : ℕ) : ∀ h : ℕ, h = 80 + 2 * x := by
  sorry

end tree_height_equation_l1496_149609


namespace average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l1496_149663

open Real

noncomputable def f (x : ℝ) := (2/3) * x ^ 3 + x ^ 2 + 2 * x

-- (1) Prove that the average velocity of the particle during the first second is 3 m/s
theorem average_velocity_first_second : (f 1 - f 0) / (1 - 0) = 3 := by
  sorry

-- (2) Prove that the instantaneous velocity at the end of the first second is 6 m/s
theorem instantaneous_velocity_end_first_second : deriv f 1 = 6 := by
  sorry

-- (3) Prove that the velocity of the particle reaches 14 m/s after 2 seconds
theorem velocity_reaches_14_after_2_seconds :
  ∃ x : ℝ, deriv f x = 14 ∧ x = 2 := by
  sorry

end average_velocity_first_second_instantaneous_velocity_end_first_second_velocity_reaches_14_after_2_seconds_l1496_149663


namespace count_multiples_of_12_l1496_149634

theorem count_multiples_of_12 (a b : ℤ) (h1 : a = 5) (h2 : b = 145) :
  ∃ n : ℕ, (12 * n + 12 ≤ b) ∧ (12 * n + 12 > a) ∧ n = 12 :=
by
  sorry

end count_multiples_of_12_l1496_149634


namespace michael_monica_age_ratio_l1496_149647

theorem michael_monica_age_ratio
  (x y : ℕ)
  (Patrick Michael Monica : ℕ)
  (h1 : Patrick = 3 * x)
  (h2 : Michael = 5 * x)
  (h3 : Monica = y)
  (h4 : y - Patrick = 64)
  (h5 : Patrick + Michael + Monica = 196) :
  Michael * 5 = Monica * 3 :=
by
  sorry

end michael_monica_age_ratio_l1496_149647


namespace f_no_zeros_in_interval_f_zeros_in_interval_l1496_149620

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem f_no_zeros_in_interval (x : ℝ) (hx1 : x > 1 / Real.exp 1) (hx2 : x < 1) :
  f x ≠ 0 := sorry

theorem f_zeros_in_interval (h1 : 1 < e) (x_exists : ∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :
  true := sorry

end f_no_zeros_in_interval_f_zeros_in_interval_l1496_149620


namespace smallest_n_l1496_149648

-- Definitions for arithmetic sequences with given conditions
def arithmetic_sequence_a (n : ℕ) (x : ℕ) : ℕ := 1 + (n-1) * x
def arithmetic_sequence_b (n : ℕ) (y : ℕ) : ℕ := 1 + (n-1) * y

-- Main theorem statement
theorem smallest_n (x y n : ℕ) (hxy : x < y) (ha1 : arithmetic_sequence_a 1 x = 1) (hb1 : arithmetic_sequence_b 1 y = 1) 
  (h_sum : arithmetic_sequence_a n x + arithmetic_sequence_b n y = 2556) : n = 3 :=
sorry

end smallest_n_l1496_149648


namespace inequality_for_positive_reals_l1496_149607

theorem inequality_for_positive_reals
  (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a / (a^4 + b^2) + b / (a^2 + b^4) ≤ 1 / (a * b) := 
sorry

end inequality_for_positive_reals_l1496_149607


namespace Bob_wins_game_l1496_149658

theorem Bob_wins_game :
  ∀ (initial_set : Set ℕ),
    47 ∈ initial_set →
    2016 ∈ initial_set →
    (∀ (a b : ℕ), a ∈ initial_set → b ∈ initial_set → a > b → (a - b) ∉ initial_set → (a - b) ∈ initial_set) →
    (∀ (S : Set ℕ), S ⊆ initial_set → ∃ (n : ℕ), ∀ m ∈ S, m > n) → false :=
by
  sorry

end Bob_wins_game_l1496_149658


namespace minimum_value_ineq_l1496_149640

theorem minimum_value_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) :
  (1 / (x + 3 * y) + 1 / (y + 3 * z) + 1 / (z + 3 * x)) ≥ (3 / 4) := sorry

end minimum_value_ineq_l1496_149640


namespace least_multiple_of_24_gt_500_l1496_149666

theorem least_multiple_of_24_gt_500 : ∃ x : ℕ, (x % 24 = 0) ∧ (x > 500) ∧ (∀ y : ℕ, (y % 24 = 0) ∧ (y > 500) → y ≥ x) ∧ (x = 504) := by
  sorry

end least_multiple_of_24_gt_500_l1496_149666


namespace sector_area_l1496_149605

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 2
noncomputable def r : ℝ := l / θ

theorem sector_area :
  (1 / 2) * l * r = 4 :=
by
  -- Proof goes here
  sorry

end sector_area_l1496_149605


namespace seed_total_after_trading_l1496_149698

theorem seed_total_after_trading :
  ∀ (Bom Gwi Yeon Eun : ℕ),
  Yeon = 3 * Gwi →
  Gwi = Bom + 40 →
  Eun = 2 * Gwi →
  Bom = 300 →
  Yeon_gives = 20 * Yeon / 100 →
  Bom_gives = 50 →
  let Yeon_after := Yeon - Yeon_gives
  let Gwi_after := Gwi + Yeon_gives
  let Bom_after := Bom - Bom_gives
  let Eun_after := Eun + Bom_gives
  Bom_after + Gwi_after + Yeon_after + Eun_after = 2340 :=
by
  intros Bom Gwi Yeon Eun hYeon hGwi hEun hBom hYeonGives hBomGives Yeon_after Gwi_after Bom_after Eun_after
  sorry

end seed_total_after_trading_l1496_149698


namespace goldfish_equal_months_l1496_149676

theorem goldfish_equal_months :
  ∃ (n : ℕ), 
    let B_n := 3 * 3^n 
    let G_n := 125 * 5^n 
    B_n = G_n ∧ n = 5 :=
by
  sorry

end goldfish_equal_months_l1496_149676


namespace intersection_point_l1496_149661

theorem intersection_point (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : x + y - 3 = 0) : x = 2 ∧ y = 1 :=
by
  sorry

end intersection_point_l1496_149661


namespace solve_equation1_solve_equation2_l1496_149608

-- Lean 4 statements for the given problems:
theorem solve_equation1 (x : ℝ) (h : x ≠ 0) : (2 / x = 3 / (x + 2)) ↔ (x = 4) := by
  sorry

theorem solve_equation2 (x : ℝ) (h : x ≠ 2) : ¬(5 / (x - 2) + 1 = (x - 7) / (2 - x)) := by
  sorry

end solve_equation1_solve_equation2_l1496_149608


namespace Meadow_sells_each_diaper_for_5_l1496_149611

-- Define the conditions as constants
def boxes_per_week := 30
def packs_per_box := 40
def diapers_per_pack := 160
def total_revenue := 960000

-- Calculate total packs and total diapers
def total_packs := boxes_per_week * packs_per_box
def total_diapers := total_packs * diapers_per_pack

-- The target price per diaper
def price_per_diaper := total_revenue / total_diapers

-- Statement of the proof theorem
theorem Meadow_sells_each_diaper_for_5 : price_per_diaper = 5 := by
  sorry

end Meadow_sells_each_diaper_for_5_l1496_149611


namespace sum_multiple_of_three_l1496_149693

theorem sum_multiple_of_three (a b : ℤ) (h₁ : ∃ m, a = 6 * m) (h₂ : ∃ n, b = 9 * n) : ∃ k, (a + b) = 3 * k :=
by
  sorry

end sum_multiple_of_three_l1496_149693


namespace math_books_count_l1496_149687

theorem math_books_count (total_books : ℕ) (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ) 
  (h1 : total_books = 100) 
  (h2 : history_books = 32) 
  (h3 : geography_books = 25) 
  (h4 : math_books = total_books - history_books - geography_books) 
  : math_books = 43 := 
by 
  rw [h1, h2, h3] at h4;
  exact h4;
-- use 'sorry' to skip the proof if needed
-- sorry

end math_books_count_l1496_149687


namespace goodColoringsOfPoints_l1496_149626

noncomputable def countGoodColorings (k m : ℕ) : ℕ :=
  (k * (k - 1) + 2) * 2 ^ m

theorem goodColoringsOfPoints :
  countGoodColorings 2011 2011 = (2011 * 2010 + 2) * 2 ^ 2011 :=
  by
    sorry

end goodColoringsOfPoints_l1496_149626


namespace division_631938_by_625_l1496_149616

theorem division_631938_by_625 :
  (631938 : ℚ) / 625 = 1011.1008 :=
by
  -- Add a placeholder proof. We do not provide the solution steps.
  sorry

end division_631938_by_625_l1496_149616


namespace find_y_l1496_149632

-- Definitions of vectors and parallel relationship
def vector_a : ℝ × ℝ := (4, 2)
def vector_b (y : ℝ) : ℝ × ℝ := (6, y)
def parallel (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a = (k * b.1, k * b.2)

-- The theorem we want to prove
theorem find_y (y : ℝ) (h : parallel vector_a (vector_b y)) : y = 3 :=
sorry

end find_y_l1496_149632


namespace how_many_years_later_will_tom_be_twice_tim_l1496_149606

-- Conditions
def toms_age := 15
def total_age := 21
def tims_age := total_age - toms_age

-- Define the problem statement
theorem how_many_years_later_will_tom_be_twice_tim (x : ℕ) 
  (h1 : toms_age + tims_age = total_age) 
  (h2 : toms_age = 15) 
  (h3 : ∀ y : ℕ, toms_age + y = 2 * (tims_age + y) ↔ y = x) : 
  x = 3 
:= sorry

end how_many_years_later_will_tom_be_twice_tim_l1496_149606


namespace steve_keeps_total_money_excluding_advance_l1496_149610

-- Definitions of the conditions
def totalCopies : ℕ := 1000000
def advanceCopies : ℕ := 100000
def pricePerCopy : ℕ := 2
def agentCommissionRate : ℚ := 0.1

-- Question and final proof
theorem steve_keeps_total_money_excluding_advance :
  let totalEarnings := totalCopies * pricePerCopy
  let agentCommission := agentCommissionRate * totalEarnings
  let moneyKept := totalEarnings - agentCommission
  moneyKept = 1800000 := by
  -- Proof goes here, but we skip it for now
  sorry

end steve_keeps_total_money_excluding_advance_l1496_149610


namespace ratio_of_boys_to_girls_l1496_149684

theorem ratio_of_boys_to_girls (G B : ℕ) (hg : G = 30) (hb : B = G + 18) : B / G = 8 / 5 :=
by
  sorry

end ratio_of_boys_to_girls_l1496_149684


namespace solve_equation_l1496_149683

theorem solve_equation (x : ℚ) :
  (x + 10) / (x - 4) = (x - 3) / (x + 6) ↔ x = -48 / 23 :=
by sorry

end solve_equation_l1496_149683


namespace value_of_fraction_l1496_149675

theorem value_of_fraction : (20 + 15) / (30 - 25) = 7 := by
  sorry

end value_of_fraction_l1496_149675


namespace difference_of_extremes_l1496_149636

def digits : List ℕ := [2, 0, 1, 3]

def largest_integer : ℕ := 3210
def smallest_integer_greater_than_1000 : ℕ := 1023
def expected_difference : ℕ := 2187

theorem difference_of_extremes :
  largest_integer - smallest_integer_greater_than_1000 = expected_difference := by
  sorry

end difference_of_extremes_l1496_149636


namespace n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l1496_149674

theorem n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd :
  ∀ (n m : ℤ), (n^2 + m^3) % 2 ≠ 0 → (n + m) % 2 = 1 :=
by sorry

end n_and_m_integers_and_n2_plus_m3_odd_then_n_plus_m_odd_l1496_149674


namespace average_rate_of_interest_l1496_149695

/-- Given:
    1. A woman has a total of $7500 invested,
    2. Part of the investment is at 5% interest,
    3. The remainder of the investment is at 7% interest,
    4. The annual returns from both investments are equal,
    Prove:
    The average rate of interest realized on her total investment is 5.8%.
-/
theorem average_rate_of_interest
  (total_investment : ℝ) (interest_5_percent : ℝ) (interest_7_percent : ℝ)
  (annual_return_equal : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent)
  (total_investment_eq : total_investment = 7500) : 
  (interest_5_percent / total_investment) = 0.058 :=
by
  -- conditions given
  have h1 : total_investment = 7500 := total_investment_eq
  have h2 : 0.05 * (total_investment - interest_7_percent) = 0.07 * interest_7_percent := annual_return_equal

  -- final step, sorry is used to skip the proof
  sorry

end average_rate_of_interest_l1496_149695


namespace find_n_l1496_149643

theorem find_n (n : ℕ) (b : Fin (n + 1) → ℝ) (h0 : b 0 = 45) (h1 : b 1 = 81) (hn : b n = 0) (rec : ∀ (k : ℕ), 1 ≤ k → k < n → b (k+1) = b (k-1) - 5 / b k) : 
  n = 730 :=
sorry

end find_n_l1496_149643


namespace least_common_multiple_of_first_10_integers_l1496_149600

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l1496_149600


namespace matilda_initial_bars_l1496_149654

theorem matilda_initial_bars (M : ℕ) 
  (shared_evenly : 5 * M = 20 * 2 / 5)
  (half_given_to_father : M / 2 * 5 = 10)
  (father_bars : 5 + 3 + 2 = 10) :
  M = 4 := 
by
  sorry

end matilda_initial_bars_l1496_149654


namespace expression_max_value_l1496_149670

open Real

theorem expression_max_value (x : ℝ) : ∃ M, M = 1/7 ∧ (∀ y : ℝ, y = x -> (y^3) / (y^6 + y^4 + y^3 - 3*y^2 + 9) ≤ M) :=
sorry

end expression_max_value_l1496_149670


namespace primary_schools_to_be_selected_l1496_149689

noncomputable def total_schools : ℕ := 150 + 75 + 25
noncomputable def proportion_primary : ℚ := 150 / total_schools
noncomputable def selected_primary : ℚ := proportion_primary * 30

theorem primary_schools_to_be_selected : selected_primary = 18 :=
by sorry

end primary_schools_to_be_selected_l1496_149689


namespace grocery_store_more_expensive_per_can_l1496_149635

theorem grocery_store_more_expensive_per_can :
  ∀ (bulk_case_price : ℝ) (bulk_cans_per_case : ℕ)
    (grocery_case_price : ℝ) (grocery_cans_per_case : ℕ),
  bulk_case_price = 12.00 →
  bulk_cans_per_case = 48 →
  grocery_case_price = 6.00 →
  grocery_cans_per_case = 12 →
  (grocery_case_price / grocery_cans_per_case - bulk_case_price / bulk_cans_per_case) * 100 = 25 :=
by
  intros _ _ _ _ h1 h2 h3 h4
  sorry

end grocery_store_more_expensive_per_can_l1496_149635


namespace correct_tile_for_b_l1496_149633

structure Tile where
  top : ℕ
  right : ℕ
  bottom : ℕ
  left : ℕ

def TileI : Tile := {top := 5, right := 3, bottom := 1, left := 6}
def TileII : Tile := {top := 2, right := 6, bottom := 3, left := 5}
def TileIII : Tile := {top := 6, right := 1, bottom := 4, left := 2}
def TileIV : Tile := {top := 4, right := 5, bottom := 2, left := 1}

def RectangleBTile := TileIII

theorem correct_tile_for_b : RectangleBTile = TileIII :=
  sorry

end correct_tile_for_b_l1496_149633


namespace extra_pieces_correct_l1496_149665

def pieces_per_package : ℕ := 7
def number_of_packages : ℕ := 5
def total_pieces : ℕ := 41

theorem extra_pieces_correct : total_pieces - (number_of_packages * pieces_per_package) = 6 :=
by
  sorry

end extra_pieces_correct_l1496_149665


namespace sequence_terms_l1496_149697

theorem sequence_terms (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 ^ n - 2) :
  (a 1 = 1) ∧ (∀ n, n ≥ 2 → a n = 2 * 3 ^ (n - 1)) := by
  sorry

end sequence_terms_l1496_149697


namespace angle_relationship_l1496_149651

theorem angle_relationship (u x y z w : ℝ)
    (H1 : ∀ (D E : ℝ), x + y + (360 - u - z) = 360)
    (H2 : ∀ (D E : ℝ), z + w + (360 - w - x) = 360) :
    x = (u + 2*z - y - w) / 2 := by
  sorry

end angle_relationship_l1496_149651


namespace box_and_apples_weight_l1496_149619

theorem box_and_apples_weight
  (total_weight : ℝ)
  (weight_after_half : ℝ)
  (h1 : total_weight = 62.8)
  (h2 : weight_after_half = 31.8) :
  ∃ (box_weight apple_weight : ℝ), box_weight = 0.8 ∧ apple_weight = 62 :=
by
  sorry

end box_and_apples_weight_l1496_149619


namespace cloth_sold_l1496_149667

theorem cloth_sold (C S P: ℝ) (N : ℕ) 
  (h1 : S = 3 * C)
  (h2 : P = 10 * S)
  (h3 : (200 : ℝ) = (P / (N * C)) * 100) : N = 15 := 
sorry

end cloth_sold_l1496_149667


namespace gcd_n_four_plus_sixteen_and_n_plus_three_l1496_149653

theorem gcd_n_four_plus_sixteen_and_n_plus_three (n : ℕ) (hn1 : n > 9) (hn2 : n ≠ 94) :
  Nat.gcd (n^4 + 16) (n + 3) = 1 :=
by
  sorry

end gcd_n_four_plus_sixteen_and_n_plus_three_l1496_149653


namespace find_constant_l1496_149688

noncomputable def f (x : ℝ) : ℝ := x + 4

theorem find_constant : ∃ c : ℝ, (∀ x : ℝ, x = 0.4 → (3 * f (x - c)) / f 0 + 4 = f (2 * x + 1)) ∧ c = 2 :=
by
  sorry

end find_constant_l1496_149688


namespace average_marks_l1496_149669

-- Given conditions
variables (M P C : ℝ)
variables (h1 : M + P = 32) (h2 : C = P + 20)

-- Statement to be proved
theorem average_marks : (M + C) / 2 = 26 :=
by
  -- The proof will be inserted here
  sorry

end average_marks_l1496_149669


namespace weeks_to_save_remaining_l1496_149641

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l1496_149641


namespace original_selling_price_is_990_l1496_149603

theorem original_selling_price_is_990 
( P : ℝ ) -- original purchase price
( SP_1 : ℝ := 1.10 * P ) -- original selling price
( P_new : ℝ := 0.90 * P ) -- new purchase price
( SP_2 : ℝ := 1.17 * P ) -- new selling price
( h : SP_2 - SP_1 = 63 ) : SP_1 = 990 :=
by {
  -- This is just the statement, proof is not provided
  sorry
}

end original_selling_price_is_990_l1496_149603
