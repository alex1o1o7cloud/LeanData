import Mathlib

namespace count_multiples_4_6_not_5_9_l244_24406

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end count_multiples_4_6_not_5_9_l244_24406


namespace number_of_players_tournament_l244_24461

theorem number_of_players_tournament (n : ℕ) : 
  (2 * n * (n - 1) = 272) → n = 17 :=
by
  sorry

end number_of_players_tournament_l244_24461


namespace dave_trips_l244_24417

theorem dave_trips :
  let trays_at_a_time := 12
  let trays_table_1 := 26
  let trays_table_2 := 49
  let trays_table_3 := 65
  let trays_table_4 := 38
  let total_trays := trays_table_1 + trays_table_2 + trays_table_3 + trays_table_4
  let trips := (total_trays + trays_at_a_time - 1) / trays_at_a_time
  trips = 15 := by
    repeat { sorry }

end dave_trips_l244_24417


namespace distance_school_house_l244_24425

def speed_to_school : ℝ := 6
def speed_from_school : ℝ := 4
def total_time : ℝ := 10

theorem distance_school_house : 
  ∃ D : ℝ, (D / speed_to_school + D / speed_from_school = total_time) ∧ (D = 24) :=
sorry

end distance_school_house_l244_24425


namespace line_equation_l244_24457

theorem line_equation (m n : ℝ) (p : ℝ) (h : p = 3) :
  ∃ b : ℝ, ∀ x y : ℝ, (y = n + 21) → (x = m + 3) → y = 7 * x + b ∧ b = n - 7 * m :=
by sorry

end line_equation_l244_24457


namespace total_songs_listened_l244_24495

theorem total_songs_listened (vivian_daily : ℕ) (fewer_songs : ℕ) (days_in_june : ℕ) (weekend_days : ℕ) :
  vivian_daily = 10 →
  fewer_songs = 2 →
  days_in_june = 30 →
  weekend_days = 8 →
  (vivian_daily * (days_in_june - weekend_days)) + ((vivian_daily - fewer_songs) * (days_in_june - weekend_days)) = 396 := 
by
  intros h1 h2 h3 h4
  sorry

end total_songs_listened_l244_24495


namespace find_f_2021_l244_24402

def f (x : ℝ) : ℝ := sorry

theorem find_f_2021 (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
    (h1 : f 1 = 5) (h4 : f 4 = 2) : f 2021 = -2015 :=
by
  sorry

end find_f_2021_l244_24402


namespace value_of_X_l244_24411

def M := 2007 / 3
def N := M / 3
def X := M - N

theorem value_of_X : X = 446 := by
  sorry

end value_of_X_l244_24411


namespace find_y_l244_24485

theorem find_y (y : ℝ) (h : 2 * y / 3 = 12) : y = 18 :=
by
  sorry

end find_y_l244_24485


namespace gcd_765432_654321_eq_3_l244_24446

theorem gcd_765432_654321_eq_3 :
  Nat.gcd 765432 654321 = 3 :=
sorry -- Proof is omitted

end gcd_765432_654321_eq_3_l244_24446


namespace hydrogen_atoms_count_l244_24479

-- Definitions for atomic weights
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Given conditions
def total_molecular_weight : ℝ := 88
def number_of_C_atoms : ℕ := 4
def number_of_O_atoms : ℕ := 2

theorem hydrogen_atoms_count (nh : ℕ) 
  (h_molecular_weight : total_molecular_weight = 88) 
  (h_C_atoms : number_of_C_atoms = 4) 
  (h_O_atoms : number_of_O_atoms = 2) :
  nh = 8 :=
by
  -- skipping proof
  sorry

end hydrogen_atoms_count_l244_24479


namespace number_of_squares_or_cubes_l244_24435

theorem number_of_squares_or_cubes (h1 : ∃ n, n = 28) (h2 : ∃ m, m = 9) (h3 : ∃ k, k = 2) : 
  ∃ t, t = 35 :=
sorry

end number_of_squares_or_cubes_l244_24435


namespace raw_score_is_correct_l244_24423

-- Define the conditions
def points_per_correct : ℝ := 1
def points_subtracted_per_incorrect : ℝ := 0.25
def total_questions : ℕ := 85
def answered_questions : ℕ := 82
def correct_answers : ℕ := 70

-- Define the number of incorrect answers
def incorrect_answers : ℕ := answered_questions - correct_answers
-- Calculate the raw score
def raw_score : ℝ := 
  (correct_answers * points_per_correct) - (incorrect_answers * points_subtracted_per_incorrect)

-- Prove the raw score is 67
theorem raw_score_is_correct : raw_score = 67 := 
by
  sorry

end raw_score_is_correct_l244_24423


namespace probability_product_is_square_l244_24487

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

noncomputable def probability_square_product : ℚ :=
  let total_outcomes   := 10 * 8
  let favorable_outcomes := 
    [(1,1), (1,4), (2,2), (4,1), (3,3), (2,8), (8,2), (5,5), (6,6), (7,7), (8,8)].length
  favorable_outcomes / total_outcomes

theorem probability_product_is_square : 
  probability_square_product = 11 / 80 :=
  sorry

end probability_product_is_square_l244_24487


namespace sins_prayers_l244_24444

structure Sins :=
  (pride : Nat)
  (slander : Nat)
  (laziness : Nat)
  (adultery : Nat)
  (gluttony : Nat)
  (self_love : Nat)
  (jealousy : Nat)
  (malicious_gossip : Nat)

def prayer_requirements (s : Sins) : Nat × Nat × Nat :=
  ( s.pride + 2 * s.laziness + 10 * s.adultery + s.gluttony,
    2 * s.pride + 2 * s.slander + 10 * s.adultery + 3 * s.self_love + 3 * s.jealousy + 7 * s.malicious_gossip,
    7 * s.slander + 10 * s.adultery + s.self_love + 2 * s.malicious_gossip )

theorem sins_prayers (sins : Sins) :
  sins.pride = 0 ∧
  sins.slander = 1 ∧
  sins.laziness = 0 ∧
  sins.adultery = 0 ∧
  sins.gluttony = 9 ∧
  sins.self_love = 1 ∧
  sins.jealousy = 0 ∧
  sins.malicious_gossip = 2 ∧
  (sins.pride + sins.slander + sins.laziness + sins.adultery + sins.gluttony + sins.self_love + sins.jealousy + sins.malicious_gossip = 12) ∧
  prayer_requirements sins = (9, 12, 10) :=
  by
  sorry

end sins_prayers_l244_24444


namespace m_range_l244_24407

noncomputable def range_of_m (m : ℝ) : Prop :=
  ∃ x : ℝ, 
    x ≠ 2 ∧ 
    (x + m) / (x - 2) - 3 = (x - 1) / (2 - x) ∧ 
    x ≥ 0

theorem m_range (m : ℝ) : 
  range_of_m m ↔ m ≥ -5 ∧ m ≠ -3 := 
sorry

end m_range_l244_24407


namespace intermediate_root_exists_l244_24467

open Polynomial

theorem intermediate_root_exists
  (a b c x1 x2 : ℝ) 
  (h1 : a * x1^2 + b * x1 + c = 0) 
  (h2 : -a * x2^2 + b * x2 + c = 0) :
  ∃ x3 : ℝ, (a / 2) * x3^2 + b * x3 + c = 0 ∧ (x1 ≤ x3 ∧ x3 ≤ x2 ∨ x1 ≥ x3 ∧ x3 ≥ x2) :=
sorry

end intermediate_root_exists_l244_24467


namespace sock_pairing_l244_24413

def sockPicker : Prop :=
  let white_socks := 5
  let brown_socks := 5
  let blue_socks := 2
  let total_socks := 12
  let choose (n k : ℕ) := Nat.choose n k
  (choose white_socks 2 + choose brown_socks 2 + choose blue_socks 2 = 21) ∧
  (choose (white_socks + brown_socks) 2 = 45) ∧
  (45 = 45)

theorem sock_pairing :
  sockPicker :=
by sorry

end sock_pairing_l244_24413


namespace blonde_hair_count_l244_24434

theorem blonde_hair_count (total_people : ℕ) (percentage_blonde : ℕ) (h_total : total_people = 600) (h_percentage : percentage_blonde = 30) : 
  (percentage_blonde * total_people / 100) = 180 :=
by
  -- Conditions from the problem
  have h1 : total_people = 600 := h_total
  have h2 : percentage_blonde = 30 := h_percentage
  -- Start the proof
  sorry

end blonde_hair_count_l244_24434


namespace complex_expression_l244_24484

def imaginary_unit (i : ℂ) : Prop := i^2 = -1

theorem complex_expression (i : ℂ) (h : imaginary_unit i) :
  (1 - i) ^ 2016 + (1 + i) ^ 2016 = 2 ^ 1009 :=
by
  sorry

end complex_expression_l244_24484


namespace smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l244_24450

open Nat

theorem smallest_natur_number_with_units_digit_6_and_transf_is_four_times (n : ℕ) :
  (n % 10 = 6 ∧ ∃ m, 6 * 10 ^ (m - 1) + n / 10 = 4 * n) → n = 153846 :=
by 
  sorry

end smallest_natur_number_with_units_digit_6_and_transf_is_four_times_l244_24450


namespace cost_price_of_computer_table_l244_24455

theorem cost_price_of_computer_table (SP : ℝ) (h1 : SP = 1.15 * CP ∧ SP = 6400) : CP = 5565.22 :=
by
  sorry

end cost_price_of_computer_table_l244_24455


namespace russian_writer_surname_l244_24410

def is_valid_surname (x y z w v u : ℕ) : Prop :=
  x = z ∧
  y = w ∧
  v = x + 9 ∧
  u = y + w - 2 ∧
  3 * x = y - 4 ∧
  x + y + z + w + v + u = 83

def position_to_letter (n : ℕ) : String :=
  if n = 4 then "Г"
  else if n = 16 then "О"
  else if n = 13 then "Л"
  else if n = 30 then "Ь"
  else "?"

theorem russian_writer_surname : ∃ x y z w v u : ℕ, 
  is_valid_surname x y z w v u ∧
  position_to_letter x ++ position_to_letter y ++ position_to_letter z ++ position_to_letter w ++ position_to_letter v ++ position_to_letter u = "Гоголь" :=
by
  sorry

end russian_writer_surname_l244_24410


namespace arithmetic_sequence_sum_l244_24426

variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n} represented by a function a : ℕ → ℝ

/-- Given that the sum of some terms of an arithmetic sequence is 25, prove the sum of other terms -/
theorem arithmetic_sequence_sum (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) : a 2 + a 8 = 10 := by
    sorry

end arithmetic_sequence_sum_l244_24426


namespace law_of_cosines_l244_24481

theorem law_of_cosines (a b c : ℝ) (A : ℝ) (h_triangle : a > 0 ∧ b > 0 ∧ c > 0 ∧ A ≥ 0 ∧ A ≤ π) :
  a^2 = b^2 + c^2 - 2 * b * c * Real.cos A :=
sorry

end law_of_cosines_l244_24481


namespace min_value_a_2b_3c_l244_24452

theorem min_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2 * y - 3 ≤ a * x + b * y + c ∧ a * x + b * y + c ≤ x + 2 * y + 3) :
  a + 2 * b - 3 * c ≥ -2 :=
sorry

end min_value_a_2b_3c_l244_24452


namespace fraction_of_beans_remaining_l244_24496

variables (J B R : ℝ)

-- Given conditions
def condition1 : Prop := J = 0.10 * (J + B)
def condition2 : Prop := J + R = 0.60 * (J + B)

theorem fraction_of_beans_remaining (h1 : condition1 J B) (h2 : condition2 J B R) :
  R / B = 5 / 9 :=
  sorry

end fraction_of_beans_remaining_l244_24496


namespace find_b2_a2_minus_a1_l244_24445

theorem find_b2_a2_minus_a1 
  (a₁ a₂ b₁ b₂ b₃ : ℝ)
  (d r : ℝ)
  (h_arith_seq : a₁ = -9 + d ∧ a₂ = a₁ + d)
  (h_geo_seq : b₁ = -9 * r ∧ b₂ = b₁ * r ∧ b₃ = b₂ * r ∧ (-9) * (-1) = b₁ * b₃)
  (h_d_val : a₂ - a₁ = d)
  (h_b2_val : b₂ = -1) : 
  b₂ * (a₂ - a₁) = -8 :=
sorry

end find_b2_a2_minus_a1_l244_24445


namespace greatest_integer_value_x_l244_24463

theorem greatest_integer_value_x :
  ∀ x : ℤ, (∃ k : ℤ, x^2 + 2 * x + 9 = k * (x - 5)) ↔ x ≤ 49 :=
by
  sorry

end greatest_integer_value_x_l244_24463


namespace stamps_cost_l244_24448

theorem stamps_cost (cost_one: ℝ) (cost_three: ℝ) (h: cost_one = 0.34) (h1: cost_three = 3 * cost_one) : 
  2 * cost_one = 0.68 := 
by
  sorry

end stamps_cost_l244_24448


namespace solve_for_x_l244_24422

theorem solve_for_x (x : ℝ) : 
  x^2 - 2 * x - 8 = -(x + 2) * (x - 6) → (x = 5 ∨ x = -2) :=
by
  intro h
  sorry

end solve_for_x_l244_24422


namespace relationship_y1_y2_y3_l244_24453

variable (y1 y2 y3 : ℝ)

def quadratic_function (x : ℝ) : ℝ := -x^2 + 4 * x - 5

theorem relationship_y1_y2_y3
  (h1 : quadratic_function (-4) = y1)
  (h2 : quadratic_function (-3) = y2)
  (h3 : quadratic_function (1) = y3) :
  y1 < y2 ∧ y2 < y3 :=
sorry

end relationship_y1_y2_y3_l244_24453


namespace most_stable_performance_l244_24428

structure Shooter :=
(average_score : ℝ)
(variance : ℝ)

def A := Shooter.mk 8.9 0.45
def B := Shooter.mk 8.9 0.42
def C := Shooter.mk 8.9 0.51

theorem most_stable_performance : 
  B.variance < A.variance ∧ B.variance < C.variance :=
by
  sorry

end most_stable_performance_l244_24428


namespace reduced_price_equals_50_l244_24424

noncomputable def reduced_price (P : ℝ) : ℝ := 0.75 * P

theorem reduced_price_equals_50 (P : ℝ) (X : ℝ) 
  (h1 : 1000 = X * P)
  (h2 : 1000 = (X + 5) * 0.75 * P) : reduced_price P = 50 :=
sorry

end reduced_price_equals_50_l244_24424


namespace paving_stone_width_l244_24472

theorem paving_stone_width 
    (length_courtyard : ℝ)
    (width_courtyard : ℝ)
    (length_paving_stone : ℝ)
    (num_paving_stones : ℕ)
    (total_area_courtyard : ℝ)
    (total_area_paving_stones : ℝ)
    (width_paving_stone : ℝ)
    (h1 : length_courtyard = 20)
    (h2 : width_courtyard = 16.5)
    (h3 : length_paving_stone = 2.5)
    (h4 : num_paving_stones = 66)
    (h5 : total_area_courtyard = length_courtyard * width_courtyard)
    (h6 : total_area_paving_stones = num_paving_stones * (length_paving_stone * width_paving_stone))
    (h7 : total_area_courtyard = total_area_paving_stones) :
    width_paving_stone = 2 :=
by
  sorry

end paving_stone_width_l244_24472


namespace minimum_value_of_f_l244_24456

noncomputable def f (x : ℝ) : ℝ := sorry  -- define f such that f(x + 199) = 4x^2 + 4x + 3 for x ∈ ℝ

theorem minimum_value_of_f : ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = 2 := by
  sorry  -- Prove that the minimum value of f(x) is 2

end minimum_value_of_f_l244_24456


namespace cost_difference_l244_24483

theorem cost_difference (S : ℕ) (h1 : 15 + S = 24) : 15 - S = 6 :=
by
  sorry

end cost_difference_l244_24483


namespace find_unique_number_l244_24469

theorem find_unique_number : 
  ∃ X : ℕ, 
    (X % 1000 = 376 ∨ X % 1000 = 625) ∧ 
    (X * (X - 1) % 10000 = 0) ∧ 
    (Nat.gcd X (X - 1) = 1) ∧ 
    ((X % 625 = 0) ∨ ((X - 1) % 625 = 0)) ∧ 
    ((X % 16 = 0) ∨ ((X - 1) % 16 = 0)) ∧ 
    X = 9376 :=
by sorry

end find_unique_number_l244_24469


namespace remainder_when_dividing_p_by_g_is_3_l244_24497

noncomputable def p (x : ℤ) : ℤ := x^5 - 2 * x^3 + 4 * x^2 + x + 5
noncomputable def g (x : ℤ) : ℤ := x + 2

theorem remainder_when_dividing_p_by_g_is_3 : p (-2) = 3 :=
by
  sorry

end remainder_when_dividing_p_by_g_is_3_l244_24497


namespace perimeters_ratio_l244_24400

noncomputable def ratio_perimeters_of_squares (area_ratio : ℚ) (ratio_area : area_ratio = 49 / 64) : ℚ :=
if h : area_ratio = 49 / 64 
then (7 / 8) 
else 0  -- This shouldn't happen since we enforce the condition

theorem perimeters_ratio (area_ratio : ℚ) (h : area_ratio = 49 / 64) : ratio_perimeters_of_squares area_ratio h = 7 / 8 :=
by {
  -- Proof goes here
  sorry
}

end perimeters_ratio_l244_24400


namespace problem1_problem2_l244_24403

variable {a b : ℝ}

theorem problem1 (h : a > b) : a - 3 > b - 3 :=
by sorry

theorem problem2 (h : a > b) : -4 * a < -4 * b :=
by sorry

end problem1_problem2_l244_24403


namespace polygon_sides_arithmetic_sequence_l244_24494

theorem polygon_sides_arithmetic_sequence 
  (n : ℕ) 
  (h1 : n ≥ 3) 
  (h2 : 2 * (180 * (n - 2)) = n * (100 + 140)) :
  n = 6 :=
  sorry

end polygon_sides_arithmetic_sequence_l244_24494


namespace intersect_A_B_l244_24427

def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x | -1 < x ∧ x ≤ 1}

theorem intersect_A_B : A ∩ B = {0, 1} :=
by
  sorry

end intersect_A_B_l244_24427


namespace triangle_is_obtuse_l244_24439

noncomputable def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  B = 2 * A ∧ a = 1 ∧ b = 4 / 3 ∧ (a^2 + b^2 < c^2)

theorem triangle_is_obtuse (A B C : ℝ) (a b c : ℝ) (h : triangle_ABC A B C a b c) : 
  B > π / 2 :=
by
  sorry

end triangle_is_obtuse_l244_24439


namespace expression_simplifies_l244_24449

variable {a b : ℚ}
variable (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ≠ b)

theorem expression_simplifies : (a^2 - b^2) / (a * b) - (a * b - b^2) / (a * b - a^2) = a / b := by
  -- TODO: Proof goes here
  sorry

end expression_simplifies_l244_24449


namespace solve_abc_l244_24473

theorem solve_abc (a b c : ℕ) (h1 : a > b ∧ b > c) 
  (h2 : 34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0) 
  (h3 : 79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0) : 
  a = 10 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end solve_abc_l244_24473


namespace find_a_l244_24430

theorem find_a
  (a : ℝ)
  (h_perpendicular : ∀ x y : ℝ, ax + 2 * y - 1 = 0 → 3 * x - 6 * y - 1 = 0 → true) :
  a = 4 :=
by
  -- sorry is used to skip the actual proof steps
  sorry

end find_a_l244_24430


namespace find_a8_l244_24464

-- Define the arithmetic sequence and the given conditions
variable {α : Type} [AddCommGroup α] [MulAction ℤ α]

def is_arithmetic_sequence (a : ℕ → α) := ∃ d, ∀ n, a (n + 1) = a n + d

-- Given conditions
variables {a : ℕ → ℝ}
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 5 + a 6 = 22
axiom h3 : a 3 = 7

theorem find_a8 : a 8 = 15 :=
by
  -- Proof omitted
  sorry

end find_a8_l244_24464


namespace find_x_solution_l244_24418

theorem find_x_solution
  (x y z : ℤ)
  (h1 : 4 * x + y + z = 80)
  (h2 : 2 * x - y - z = 40)
  (h3 : 3 * x + y - z = 20) :
  x = 20 :=
by
  -- Proof steps go here...
  sorry

end find_x_solution_l244_24418


namespace radical_conjugate_sum_l244_24431

theorem radical_conjugate_sum :
  (15 - Real.sqrt 500) + (15 + Real.sqrt 500) = 30 :=
by
  sorry

end radical_conjugate_sum_l244_24431


namespace minimum_value_of_func_l244_24476

-- Define the circle and the line constraints, and the question
namespace CircleLineProblem

def is_center_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 + 2 * x - 4 * y + 1 = 0

def line_divides_circle (a b : ℝ) : Prop :=
  ∀ x y : ℝ, is_center_of_circle x y → a * x - b * y + 3 = 0

noncomputable def func_to_minimize (a b : ℝ) : ℝ :=
  (2 / a) + (1 / (b - 1))

theorem minimum_value_of_func :
  ∃ (a b : ℝ), a > 0 ∧ b > 1 ∧ line_divides_circle a b ∧ func_to_minimize a b = 8 :=
by
  sorry

end CircleLineProblem

end minimum_value_of_func_l244_24476


namespace square_of_binomial_l244_24442

theorem square_of_binomial (c : ℝ) (h : ∃ a : ℝ, x^2 + 50 * x + c = (x + a)^2) : c = 625 :=
by
  sorry

end square_of_binomial_l244_24442


namespace onur_biking_distance_l244_24432

-- Definitions based only on given conditions
def Onur_biking_distance_per_day (O : ℕ) := O
def Hanil_biking_distance_per_day (O : ℕ) := O + 40
def biking_days_per_week := 5
def total_distance_per_week := 2700

-- Mathematically equivalent proof problem
theorem onur_biking_distance (O : ℕ) (cond : 5 * (O + (O + 40)) = 2700) : O = 250 := by
  sorry

end onur_biking_distance_l244_24432


namespace correct_calculation_value_l244_24480

theorem correct_calculation_value (x : ℕ) (h : (x * 5) + 7 = 27) : (x + 5) * 7 = 63 :=
by
  -- The conditions are used directly in the definitions
  -- Given the condition (x * 5) + 7 = 27
  let h1 := h
  -- Solve for x and use x in the correct calculation
  sorry

end correct_calculation_value_l244_24480


namespace total_resistance_l244_24498

theorem total_resistance (x y z : ℝ) (R_parallel r : ℝ)
    (hx : x = 3)
    (hy : y = 6)
    (hz : z = 4)
    (hR_parallel : 1 / R_parallel = 1 / x + 1 / y)
    (hr : r = R_parallel + z) :
    r = 6 := by
  sorry

end total_resistance_l244_24498


namespace part_a_int_values_part_b_int_values_l244_24462

-- Part (a)
theorem part_a_int_values (n : ℤ) :
  ∃ k : ℤ, (n^4 + 3) = k * (n^2 + n + 1) ↔ n = -3 ∨ n = -1 ∨ n = 0 :=
sorry

-- Part (b)
theorem part_b_int_values (n : ℤ) :
  ∃ m : ℤ, (n^3 + n + 1) = m * (n^2 - n + 1) ↔ n = 0 ∨ n = 1 :=
sorry

end part_a_int_values_part_b_int_values_l244_24462


namespace parallelogram_area_l244_24459

theorem parallelogram_area (base height : ℝ) (h_base : base = 24) (h_height : height = 10) :
  base * height = 240 :=
by
  rw [h_base, h_height]
  norm_num

end parallelogram_area_l244_24459


namespace count_blanks_l244_24405

theorem count_blanks (B : ℝ) (h1 : 10 + B = T) (h2 : 0.7142857142857143 = B / T) : B = 25 :=
by
  -- The conditions are taken into account as definitions or parameters
  -- We skip the proof itself by using 'sorry'
  sorry

end count_blanks_l244_24405


namespace relay_race_athlete_orders_l244_24415

def athlete_count : ℕ := 4
def cannot_run_first_leg (athlete : ℕ) : Prop := athlete = 1
def cannot_run_fourth_leg (athlete : ℕ) : Prop := athlete = 2

theorem relay_race_athlete_orders : 
  ∃ (number_of_orders : ℕ), number_of_orders = 14 := 
by 
  -- Proof is omitted because it’s not required as per instructions.
  sorry

end relay_race_athlete_orders_l244_24415


namespace find_leftover_amount_l244_24438

open Nat

def octal_to_decimal (n : ℕ) : ℕ :=
  let digits := [5, 5, 5, 5]
  List.foldr (λ (d : ℕ) (acc : ℕ) => d + 8 * acc) 0 digits

def expenses_total : ℕ := 1200 + 800 + 400

theorem find_leftover_amount : 
  let initial_amount := octal_to_decimal 5555
  let final_amount := initial_amount - expenses_total
  final_amount = 525 := by
    sorry

end find_leftover_amount_l244_24438


namespace moles_of_Br2_combined_l244_24493

-- Definition of the reaction relation
def chemical_reaction (CH4 Br2 CH3Br HBr : ℕ) : Prop :=
  CH4 = 1 ∧ HBr = 1

-- Statement of the proof problem
theorem moles_of_Br2_combined (CH4 Br2 CH3Br HBr : ℕ) (h : chemical_reaction CH4 Br2 CH3Br HBr) : Br2 = 1 :=
by
  sorry

end moles_of_Br2_combined_l244_24493


namespace meaningful_fraction_condition_l244_24471

theorem meaningful_fraction_condition (x : ℝ) : x - 2 ≠ 0 ↔ x ≠ 2 := 
by 
  sorry

end meaningful_fraction_condition_l244_24471


namespace quadratic_no_roots_c_positive_l244_24408

theorem quadratic_no_roots_c_positive
  (a b c : ℝ)
  (h_no_roots : ∀ x : ℝ, a * x^2 + b * x + c ≠ 0)
  (h_positive : a + b + c > 0) :
  c > 0 :=
sorry

end quadratic_no_roots_c_positive_l244_24408


namespace first_new_player_weight_l244_24454

theorem first_new_player_weight (x : ℝ) :
  (7 * 103) + x + 60 = 9 * 99 → 
  x = 110 := by
  sorry

end first_new_player_weight_l244_24454


namespace factor_expression_l244_24412

theorem factor_expression (x y : ℤ) : 231 * x^2 * y + 33 * x * y = 33 * x * y * (7 * x + 1) := by
  sorry

end factor_expression_l244_24412


namespace min_megabytes_for_plan_Y_more_economical_l244_24474

theorem min_megabytes_for_plan_Y_more_economical :
  ∃ (m : ℕ), 2500 + 10 * m < 15 * m ∧ m = 501 :=
by
  sorry

end min_megabytes_for_plan_Y_more_economical_l244_24474


namespace point_not_in_image_of_plane_l244_24414

def satisfies_plane (P : ℝ × ℝ × ℝ) (A B C D : ℝ) : Prop :=
  let (x, y, z) := P
  A * x + B * y + C * z + D = 0

theorem point_not_in_image_of_plane :
  let A := (2, -3, 1)
  let aA := 1
  let aB := 1
  let aC := -2
  let aD := 2
  let k := 5 / 2
  let a'A := aA
  let a'B := aB
  let a'C := aC
  let a'D := k * aD
  ¬ satisfies_plane A a'A a'B a'C a'D :=
by
  -- TODO: Proof needed
  sorry

end point_not_in_image_of_plane_l244_24414


namespace find_z_l244_24429

theorem find_z 
  (m : ℕ)
  (h1 : (1^(m+1) / 5^(m+1)) * (1^18 / z^18) = 1 / (2 * 10^35))
  (hm : m = 34) :
  z = 4 := 
sorry

end find_z_l244_24429


namespace most_stable_scores_l244_24499

-- Definitions for the variances of students A, B, and C
def s_A_2 : ℝ := 6
def s_B_2 : ℝ := 24
def s_C_2 : ℝ := 50

-- The proof that student A has the most stable math scores
theorem most_stable_scores : 
  s_A_2 < s_B_2 ∧ s_B_2 < s_C_2 → 
  ("Student A has the most stable scores" = "Student A has the most stable scores") :=
by
  intros h
  sorry

end most_stable_scores_l244_24499


namespace div_scaled_result_l244_24447

theorem div_scaled_result :
  (2994 : ℝ) / 14.5 = 171 :=
by
  have cond1 : (29.94 : ℝ) / 1.45 = 17.1 := sorry
  have cond2 : (2994 : ℝ) = 100 * 29.94 := sorry
  have cond3 : (14.5 : ℝ) = 10 * 1.45 := sorry
  sorry

end div_scaled_result_l244_24447


namespace math_problem_l244_24437

noncomputable def proof_problem (a b c : ℝ) : Prop :=
  (0 < a ∧ 0 < b ∧ 0 < c ∧ a * b * c = 1) →
  (1 / (a * (a + 1)) + 1 / (b * (b + 1)) + 1 / (c * (c + 1)) ≥  3 / 2)

theorem math_problem (a b c : ℝ) :
  proof_problem a b c :=
by
  sorry

end math_problem_l244_24437


namespace improper_fraction_2012a_div_b_l244_24488

theorem improper_fraction_2012a_div_b
  (a b : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : (a : ℚ) / b < (a + 1 : ℚ) / (b + 1)) :
  2012 * a > b :=
by 
  sorry

end improper_fraction_2012a_div_b_l244_24488


namespace train_cross_time_l244_24465

/-- Given the conditions:
1. Two trains run in opposite directions and cross a man in 17 seconds and some unknown time respectively.
2. They cross each other in 22 seconds.
3. The ratio of their speeds is 1 to 1.
Prove the time it takes for the first train to cross the man. -/
theorem train_cross_time (v_1 v_2 L_1 L_2 : ℝ) (t_2 : ℝ) (h1 : t_2 = 17) (h2 : v_1 = v_2)
  (h3 : (L_1 + L_2) / (v_1 + v_2) = 22) : (L_1 / v_1) = 27 := 
by 
  -- The actual proof will go here
  sorry

end train_cross_time_l244_24465


namespace solution_set_inequality_l244_24440

theorem solution_set_inequality 
  (a b : ℝ)
  (h1 : ∀ x, a * x^2 + b * x + 3 > 0 ↔ -1 < x ∧ x < 1/2) :
  ((-1:ℝ) < x ∧ x < 2) ↔ 3 * x^2 + b * x + a < 0 :=
by 
  -- Write the proof here
  sorry

end solution_set_inequality_l244_24440


namespace three_digit_number_ends_same_sequence_l244_24475

theorem three_digit_number_ends_same_sequence (N : ℕ) (a b c : ℕ) (h1 : 100 ≤ N ∧ N < 1000)
  (h2 : N % 10 = c)
  (h3 : (N / 10) % 10 = b)
  (h4 : (N / 100) % 10 = a)
  (h5 : a ≠ 0)
  (h6 : N^2 % 1000 = N) :
  N = 127 :=
by
  sorry

end three_digit_number_ends_same_sequence_l244_24475


namespace length_of_side_of_regular_tetradecagon_l244_24486

theorem length_of_side_of_regular_tetradecagon (P : ℝ) (n : ℕ) (h₀ : n = 14) (h₁ : P = 154) : P / n = 11 := 
by
  sorry

end length_of_side_of_regular_tetradecagon_l244_24486


namespace points_per_enemy_l244_24409

-- Definitions: total enemies, enemies not destroyed, points earned
def total_enemies : ℕ := 11
def enemies_not_destroyed : ℕ := 3
def points_earned : ℕ := 72

-- To prove: points per enemy
theorem points_per_enemy : points_earned / (total_enemies - enemies_not_destroyed) = 9 := 
by
  sorry

end points_per_enemy_l244_24409


namespace area_of_region_l244_24451

theorem area_of_region :
  ∀ (x y : ℝ), (|2 * x - 2| + |3 * y - 3| ≤ 30) → (area_of_figure = 300) :=
sorry

end area_of_region_l244_24451


namespace remainder_1534_base12_div_by_9_l244_24468

noncomputable def base12_to_base10 (n : ℕ) : ℕ :=
  1 * 12^3 + 5 * 12^2 + 3 * 12 + 4

theorem remainder_1534_base12_div_by_9 :
  (base12_to_base10 1534) % 9 = 4 :=
by
  sorry

end remainder_1534_base12_div_by_9_l244_24468


namespace count_ordered_triples_lcm_l244_24482

def lcm_of_pair (a b : ℕ) : ℕ := a * b / Nat.gcd a b

theorem count_ordered_triples_lcm :
  (∃ (count : ℕ), count = 70 ∧
   ∀ (a b c : ℕ), (0 < a) → (0 < b) → (0 < c) →
   lcm_of_pair a b = 1000 → lcm_of_pair b c = 2000 → lcm_of_pair c a = 2000 → count = 70) :=
sorry

end count_ordered_triples_lcm_l244_24482


namespace inequality_bounds_l244_24470

theorem inequality_bounds (a b c d e : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) :
  1 < (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) ∧
  (a/(a+b) + b/(b+c) + c/(c+d) + d/(d+e) + e/(e+a)) < 4 :=
sorry

end inequality_bounds_l244_24470


namespace log2_x_value_l244_24416

noncomputable def log_base (a b : ℝ) : ℝ := Real.log b / Real.log a

theorem log2_x_value
  (x : ℝ)
  (h : log_base (5 * x) (2 * x) = log_base (625 * x) (8 * x)) :
  log_base 2 x = Real.log 5 / (2 * Real.log 2 - 3 * Real.log 5) :=
by
  sorry

end log2_x_value_l244_24416


namespace speed_of_car_A_l244_24419

variable (V_A V_B T : ℕ)
variable (h1 : V_B = 35) (h2 : T = 10) (h3 : 2 * V_B * T = V_A * T)

theorem speed_of_car_A :
  V_A = 70 :=
by
  sorry

end speed_of_car_A_l244_24419


namespace expected_value_smallest_N_l244_24443
noncomputable def expectedValueN : ℝ := 6.54

def barryPicksPointsInsideUnitCircle (P : ℕ → ℝ × ℝ) : Prop :=
  ∀ n, (P n).fst^2 + (P n).snd^2 ≤ 1

def pointsIndependentAndUniform (P : ℕ → ℝ × ℝ) : Prop :=
  -- This is a placeholder representing the independent and uniform picking which 
  -- would be formally defined using probability measures in an advanced Lean library.
  sorry

theorem expected_value_smallest_N (P : ℕ → ℝ × ℝ)
  (h1 : barryPicksPointsInsideUnitCircle P)
  (h2 : pointsIndependentAndUniform P) :
  ∃ N : ℕ, N = expectedValueN :=
sorry

end expected_value_smallest_N_l244_24443


namespace find_a_tangent_slope_at_point_l244_24404

theorem find_a_tangent_slope_at_point :
  ∃ (a : ℝ), (∃ (y : ℝ), y = (fun (x : ℝ) => x^4 + a * x^2 + 1) (-1) ∧ (∃ (y' : ℝ), y' = (fun (x : ℝ) => 4 * x^3 + 2 * a * x) (-1) ∧ y' = 8)) ∧ a = -6 :=
by
  -- Used to skip the proof
  sorry

end find_a_tangent_slope_at_point_l244_24404


namespace largest_three_digit_congruent_to_twelve_mod_fifteen_l244_24466

theorem largest_three_digit_congruent_to_twelve_mod_fifteen :
  ∃ n : ℕ, 100 ≤ 15 * n + 12 ∧ 15 * n + 12 < 1000 ∧ (15 * n + 12 = 987) :=
sorry

end largest_three_digit_congruent_to_twelve_mod_fifteen_l244_24466


namespace quadratic_root_condition_l244_24491

theorem quadratic_root_condition (a b : ℝ) (h : (3:ℝ)^2 + 2 * a * 3 + 3 * b = 0) : 2 * a + b = -3 :=
by
  sorry

end quadratic_root_condition_l244_24491


namespace triple_sum_equals_seven_l244_24441

theorem triple_sum_equals_seven {k m n : ℕ} (hk : 0 < k) (hm : 0 < m) (hn : 0 < n)
  (hcoprime : Nat.gcd k m = 1 ∧ Nat.gcd k n = 1 ∧ Nat.gcd m n = 1)
  (hlog : k * Real.log 5 / Real.log 400 + m * Real.log 2 / Real.log 400 = n) :
  k + m + n = 7 := by
  sorry

end triple_sum_equals_seven_l244_24441


namespace rectangular_area_length_width_l244_24490

open Nat

theorem rectangular_area_length_width (lengthInMeters widthInMeters : ℕ) (h1 : lengthInMeters = 500) (h2 : widthInMeters = 60) :
  (lengthInMeters * widthInMeters = 30000) ∧ ((lengthInMeters * widthInMeters) / 10000 = 3) :=
by
  sorry

end rectangular_area_length_width_l244_24490


namespace can_construct_parallelogram_l244_24489

theorem can_construct_parallelogram {a b d1 d2 : ℝ} :
  (a = 3 ∧ b = 5 ∧ (a = b ∨ (‖a + b‖ ≥ ‖d1‖ ∧ ‖a + d1‖ ≥ ‖b‖ ∧ ‖b + d1‖ ≥ ‖a‖))) ∨
  (a ≠ 3 ∨ b ≠ 5 ∨ (a ≠ b ∧ (‖a + b‖ < ‖d1‖ ∨ ‖a + d1‖ < ‖b‖ ∨ ‖b + d1‖ < ‖a‖ ∨ ‖a + d1‖ < ‖d2‖ ∨ ‖b + d1‖ < ‖d2‖ ∨ ‖a + d2‖ < ‖d1‖ ∨ ‖b + d2‖ < ‖d1‖))) ↔ 
  (a = 3 ∧ b = 5 ∧ d1 = 0) :=
sorry

end can_construct_parallelogram_l244_24489


namespace price_of_each_sundae_l244_24477

theorem price_of_each_sundae
  (num_ice_cream_bars : ℕ)
  (num_sundaes : ℕ)
  (total_price : ℝ)
  (price_per_ice_cream_bar : ℝ)
  (total_cost_for_sundaes : ℝ) :
  num_ice_cream_bars = 225 →
  num_sundaes = 125 →
  total_price = 200 →
  price_per_ice_cream_bar = 0.60 →
  total_cost_for_sundaes = total_price - (num_ice_cream_bars * price_per_ice_cream_bar) →
  (total_cost_for_sundaes / num_sundaes) = 0.52 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end price_of_each_sundae_l244_24477


namespace manager_salary_l244_24436

theorem manager_salary (average_salary_employees : ℕ)
    (employee_count : ℕ) (new_average_salary : ℕ)
    (total_salary_before : ℕ)
    (total_salary_after : ℕ)
    (M : ℕ) :
    average_salary_employees = 1500 →
    employee_count = 20 →
    new_average_salary = 1650 →
    total_salary_before = employee_count * average_salary_employees →
    total_salary_after = (employee_count + 1) * new_average_salary →
    M = total_salary_after - total_salary_before →
    M = 4650 := by
    intros h1 h2 h3 h4 h5 h6
    rw [h6]
    sorry -- The proof is not required, so we use 'sorry' here.

end manager_salary_l244_24436


namespace find_y_value_l244_24478

noncomputable def y_value (y : ℝ) :=
  (3 * y)^2 + (7 * y)^2 + (1 / 2) * (3 * y) * (7 * y) = 1200

theorem find_y_value (y : ℝ) (hy : y_value y) : y = 10 :=
by
  sorry

end find_y_value_l244_24478


namespace alphanumeric_puzzle_l244_24458

/-- Alphanumeric puzzle proof problem -/
theorem alphanumeric_puzzle
  (A B C D E F H J K L : Nat)
  (h1 : A * B = B)
  (h2 : B * C = 10 * A + C)
  (h3 : C * D = 10 * B + C)
  (h4 : D * E = 100 * C + H)
  (h5 : E * F = 10 * D + K)
  (h6 : F * H = 100 * C + J)
  (h7 : H * J = 10 * K + J)
  (h8 : J * K = E)
  (h9 : K * L = L)
  (h10 : A * L = L) :
  A = 1 ∧ B = 3 ∧ C = 5 ∧ D = 7 ∧ E = 8 ∧ F = 9 ∧ H = 6 ∧ J = 4 ∧ K = 2 ∧ L = 0 :=
sorry

end alphanumeric_puzzle_l244_24458


namespace find_angle_C_l244_24421

variables {A B C : ℝ} {a b c : ℝ} 

theorem find_angle_C (h1 : a^2 + b^2 - c^2 + a*b = 0) (C_pos : 0 < C) (C_lt_pi : C < Real.pi) :
  C = (2 * Real.pi) / 3 :=
sorry

end find_angle_C_l244_24421


namespace inequality_example_l244_24460

theorem inequality_example (a b : ℝ) (h1 : a ≥ b) (h2 : b > 0) : 2 * a^3 - b^3 ≥ 2 * a * b^2 - a^2 * b := 
  sorry

end inequality_example_l244_24460


namespace Smiths_Backery_Pies_l244_24433

theorem Smiths_Backery_Pies : 
  ∀ (p : ℕ), (∃ (q : ℕ), q = 16 ∧ p = 4 * q + 6) → p = 70 :=
by
  intros p h
  cases' h with q hq
  cases' hq with hq1 hq2
  rw [hq1] at hq2
  have h_eq : p = 4 * 16 + 6 := hq2
  norm_num at h_eq
  exact h_eq

end Smiths_Backery_Pies_l244_24433


namespace first_tv_cost_is_672_l244_24420

-- width and height of the first TV
def width_first_tv : ℕ := 24
def height_first_tv : ℕ := 16
-- width and height of the new TV
def width_new_tv : ℕ := 48
def height_new_tv : ℕ := 32
-- cost of the new TV
def cost_new_tv : ℕ := 1152
-- extra cost per square inch for the first TV
def extra_cost_per_square_inch : ℕ := 1

noncomputable def cost_first_tv : ℕ :=
  let area_first_tv := width_first_tv * height_first_tv
  let area_new_tv := width_new_tv * height_new_tv
  let cost_per_square_inch_new_tv := cost_new_tv / area_new_tv
  let cost_per_square_inch_first_tv := cost_per_square_inch_new_tv + extra_cost_per_square_inch
  cost_per_square_inch_first_tv * area_first_tv

theorem first_tv_cost_is_672 : cost_first_tv = 672 := by
  sorry

end first_tv_cost_is_672_l244_24420


namespace maximum_volume_of_prism_l244_24401

noncomputable def maximum_volume_prism (s : ℝ) (θ : ℝ) (face_area_sum : ℝ) : ℝ := 
  if (s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36) then 27 
  else 0

theorem maximum_volume_of_prism : 
  ∀ (s θ face_area_sum), s = 6 ∧ θ = Real.pi / 3 ∧ face_area_sum = 36 → maximum_volume_prism s θ face_area_sum = 27 :=
by
  intros
  sorry

end maximum_volume_of_prism_l244_24401


namespace inequality_proof_l244_24492

theorem inequality_proof (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
(h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d) (h_sum : a + b + c + d ≥ 1) : 
  a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 ≥ 1 := 
by 
  sorry

end inequality_proof_l244_24492
