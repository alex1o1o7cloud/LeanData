import Mathlib

namespace NUMINAMATH_GPT_abs_diff_is_perfect_square_l448_44820

-- Define the conditions
variable (m n : ℤ) (h_odd_m : m % 2 = 1) (h_odd_n : n % 2 = 1)
variable (h_div : (n^2 - 1) ∣ (m^2 + 1 - n^2))

-- Theorem statement
theorem abs_diff_is_perfect_square : ∃ (k : ℤ), (m^2 + 1 - n^2) = k^2 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_is_perfect_square_l448_44820


namespace NUMINAMATH_GPT_cole_trip_time_l448_44875

theorem cole_trip_time 
  (D : ℕ) -- The distance D from home to work
  (T_total : ℕ) -- The total round trip time in hours
  (S1 S2 : ℕ) -- The average speeds (S1, S2) in km/h
  (h1 : S1 = 80) -- The average speed from home to work
  (h2 : S2 = 120) -- The average speed from work to home
  (h3 : T_total = 2) -- The total round trip time is 2 hours
  : (D : ℝ) / 80 + (D : ℝ) / 120 = 2 →
    (T_work : ℝ) = (D : ℝ) / 80 →
    (T_work * 60) = 72 := 
by {
  sorry
}

end NUMINAMATH_GPT_cole_trip_time_l448_44875


namespace NUMINAMATH_GPT_find_n_l448_44872

-- Define the function to sum the digits of a natural number n
def digit_sum (n : ℕ) : ℕ := 
  -- This is a dummy implementation for now
  -- Normally, we would implement the sum of the digits of n
  sorry 

-- The main theorem that we want to prove
theorem find_n : ∃ (n : ℕ), digit_sum n + n = 2011 ∧ n = 1991 :=
by
  -- Proof steps would go here, but we're skipping those with sorry.
  sorry

end NUMINAMATH_GPT_find_n_l448_44872


namespace NUMINAMATH_GPT_no_three_digit_number_l448_44881

theorem no_three_digit_number (N : ℕ) : 
  (100 ≤ N ∧ N < 1000 ∧ 
   (∀ k, k ∈ [1,2,3] → 5 < (N / 10^(k - 1) % 10)) ∧ 
   (N % 6 = 0) ∧ (N % 5 = 0)) → 
  false :=
by
sorry

end NUMINAMATH_GPT_no_three_digit_number_l448_44881


namespace NUMINAMATH_GPT_value_of_f_5_l448_44880

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * Real.sin x - 2

theorem value_of_f_5 (a b : ℝ) (hf : f a b (-5) = 17) : f a b 5 = -21 := by
  sorry

end NUMINAMATH_GPT_value_of_f_5_l448_44880


namespace NUMINAMATH_GPT_min_board_size_l448_44808

theorem min_board_size (n : ℕ) (total_area : ℕ) (domino_area : ℕ) 
  (h1 : total_area = 2008) 
  (h2 : domino_area = 2) 
  (h3 : ∀ domino_count : ℕ, domino_count = total_area / domino_area → (∃ m : ℕ, (m+1) * (m+1) ≥ domino_count * (2 + 4) → n = m)) :
  n = 77 :=
by
  sorry

end NUMINAMATH_GPT_min_board_size_l448_44808


namespace NUMINAMATH_GPT_multiples_of_6_and_8_l448_44841

open Nat

theorem multiples_of_6_and_8 (n m k : ℕ) (h₁ : n = 33) (h₂ : m = 25) (h₃ : k = 8) :
  (n - k) + (m - k) = 42 :=
by
  sorry

end NUMINAMATH_GPT_multiples_of_6_and_8_l448_44841


namespace NUMINAMATH_GPT_coefficient_of_expansion_l448_44892

theorem coefficient_of_expansion (m : ℝ) (h : m^3 * (Nat.choose 6 3) = -160) : m = -2 := by
  sorry

end NUMINAMATH_GPT_coefficient_of_expansion_l448_44892


namespace NUMINAMATH_GPT_age_difference_l448_44810

-- Denote the ages of A, B, and C as a, b, and c respectively.
variables (a b c : ℕ)

-- The given condition
def condition : Prop := a + b = b + c + 12

-- Prove that C is 12 years younger than A.
theorem age_difference (h : condition a b c) : c = a - 12 :=
by {
  -- skip the actual proof here, as instructed
  sorry
}

end NUMINAMATH_GPT_age_difference_l448_44810


namespace NUMINAMATH_GPT_difference_between_place_and_face_value_l448_44811

def numeral : Nat := 856973

def digit_of_interest : Nat := 7

def place_value : Nat := 7 * 10

def face_value : Nat := 7

theorem difference_between_place_and_face_value : place_value - face_value = 63 :=
by
  sorry

end NUMINAMATH_GPT_difference_between_place_and_face_value_l448_44811


namespace NUMINAMATH_GPT_mean_score_l448_44804

theorem mean_score (μ σ : ℝ)
  (h1 : 86 = μ - 7 * σ)
  (h2 : 90 = μ + 3 * σ) : μ = 88.8 := by
  -- Proof steps are not included as per requirements.
  sorry

end NUMINAMATH_GPT_mean_score_l448_44804


namespace NUMINAMATH_GPT_solve1_solve2_solve3_solve4_l448_44857

noncomputable section

-- Problem 1
theorem solve1 (x : ℝ) : x^2 + 2 * x = 0 ↔ x = 0 ∨ x = -2 := sorry

-- Problem 2
theorem solve2 (x : ℝ) : (x + 1)^2 - 144 = 0 ↔ x = 11 ∨ x = -13 := sorry

-- Problem 3
theorem solve3 (x : ℝ) : 3 * (x - 2)^2 = x * (x - 2) ↔ x = 2 ∨ x = 3 := sorry

-- Problem 4
theorem solve4 (x : ℝ) : x^2 + 5 * x - 1 = 0 ↔ x = (-5 + Real.sqrt 29) / 2 ∨ x = (-5 - Real.sqrt 29) / 2 := sorry

end NUMINAMATH_GPT_solve1_solve2_solve3_solve4_l448_44857


namespace NUMINAMATH_GPT_truck_travel_yards_l448_44832

variables (b t : ℝ)

theorem truck_travel_yards : 
  (2 * (2 * b / 7) / (2 * t)) * 240 / 3 = (80 * b) / (7 * t) :=
by 
  sorry

end NUMINAMATH_GPT_truck_travel_yards_l448_44832


namespace NUMINAMATH_GPT_doubled_dimensions_volume_l448_44845

theorem doubled_dimensions_volume (original_volume : ℝ) (length_factor width_factor height_factor : ℝ) 
  (h : original_volume = 3) 
  (hl : length_factor = 2)
  (hw : width_factor = 2)
  (hh : height_factor = 2) : 
  original_volume * length_factor * width_factor * height_factor = 24 :=
by
  sorry

end NUMINAMATH_GPT_doubled_dimensions_volume_l448_44845


namespace NUMINAMATH_GPT_no_integer_solutions_for_mn_squared_eq_1980_l448_44823

theorem no_integer_solutions_for_mn_squared_eq_1980 :
  ¬ ∃ m n : ℤ, m^2 + n^2 = 1980 := 
sorry

end NUMINAMATH_GPT_no_integer_solutions_for_mn_squared_eq_1980_l448_44823


namespace NUMINAMATH_GPT_value_of_a_l448_44834

noncomputable def f (x : ℝ) : ℝ := x^2 + 10
noncomputable def g (x : ℝ) : ℝ := x^2 - 5

theorem value_of_a (a : ℝ) (h₁ : a > 0) (h₂ : f (g a) = 18) :
  a = Real.sqrt (5 + 2 * Real.sqrt 2) ∨ a = Real.sqrt (5 - 2 * Real.sqrt 2) := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_l448_44834


namespace NUMINAMATH_GPT_strictly_increasing_intervals_l448_44879

-- Define the function y = cos^2(x + π/2)
noncomputable def y (x : ℝ) : ℝ := (Real.cos (x + Real.pi / 2))^2

-- Define the assertion
theorem strictly_increasing_intervals (k : ℤ) : 
  StrictMonoOn y (Set.Icc (k * Real.pi) (k * Real.pi + Real.pi / 2)) :=
sorry

end NUMINAMATH_GPT_strictly_increasing_intervals_l448_44879


namespace NUMINAMATH_GPT_find_inradius_of_scalene_triangle_l448_44884

noncomputable def side_a := 32
noncomputable def side_b := 40
noncomputable def side_c := 24
noncomputable def ic := 18
noncomputable def expected_inradius := 2 * Real.sqrt 17

theorem find_inradius_of_scalene_triangle (a b c : ℝ) (h : a = side_a) (h1 : b = side_b) (h2 : c = side_c) (ic_length : ℝ) (h3: ic_length = ic) : (Real.sqrt (ic_length ^ 2 - (b - ((a + b - c) / 2)) ^ 2)) = expected_inradius :=
by
  sorry

end NUMINAMATH_GPT_find_inradius_of_scalene_triangle_l448_44884


namespace NUMINAMATH_GPT_min_value_of_squared_sum_l448_44822

open Real

theorem min_value_of_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ m, m = (x^2 + y^2 + z^2) ∧ m = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_squared_sum_l448_44822


namespace NUMINAMATH_GPT_xy_value_l448_44830

theorem xy_value (x y : ℝ) (h : x * (x + 2 * y) = x^2 + 10) : x * y = 5 :=
by
  sorry

end NUMINAMATH_GPT_xy_value_l448_44830


namespace NUMINAMATH_GPT_find_hours_hired_l448_44897

def hourly_rate : ℝ := 15
def tip_rate : ℝ := 0.20
def total_paid : ℝ := 54

theorem find_hours_hired (h : ℝ) : 15 * h + 0.20 * 15 * h = 54 → h = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_hours_hired_l448_44897


namespace NUMINAMATH_GPT_solve_E_l448_44813

-- Definitions based on the conditions provided
variables {A H S M C O E : ℕ}

-- Given conditions
def algebra_books := A
def geometry_books := H
def history_books := C
def S_algebra_books := S
def M_geometry_books := M
def O_history_books := O
def E_algebra_books := E

-- Prove that E = (AM + AO - SH - SC) / (M + O - H - C) given the conditions
theorem solve_E (h1: A ≠ H) (h2: A ≠ S) (h3: A ≠ M) (h4: A ≠ C) (h5: A ≠ O) (h6: A ≠ E)
                (h7: H ≠ S) (h8: H ≠ M) (h9: H ≠ C) (h10: H ≠ O) (h11: H ≠ E)
                (h12: S ≠ M) (h13: S ≠ C) (h14: S ≠ O) (h15: S ≠ E)
                (h16: M ≠ C) (h17: M ≠ O) (h18: M ≠ E)
                (h19: C ≠ O) (h20: C ≠ E)
                (h21: O ≠ E)
                (pos1: 0 < A) (pos2: 0 < H) (pos3: 0 < S) (pos4: 0 < M) (pos5: 0 < C)
                (pos6: 0 < O) (pos7: 0 < E) :
  E = (A * M + A * O - S * H - S * C) / (M + O - H - C) :=
sorry

end NUMINAMATH_GPT_solve_E_l448_44813


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l448_44805

open Set

variable {α : Type*}

def M : Set ℝ := { x | 0 < x ∧ x ≤ 4 }
def N : Set ℝ := { x | 2 ≤ x ∧ x ≤ 3 }

theorem necessary_but_not_sufficient_condition :
  (N ⊆ M) ∧ (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l448_44805


namespace NUMINAMATH_GPT_triangle_shortest_side_l448_44896

theorem triangle_shortest_side (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] 
    (base : Real) (base_angle : Real) (sum_other_sides : Real)
    (h1 : base = 80) 
    (h2 : base_angle = 60) 
    (h3 : sum_other_sides = 90) : 
    ∃ shortest_side : Real, shortest_side = 17 :=
by 
    sorry

end NUMINAMATH_GPT_triangle_shortest_side_l448_44896


namespace NUMINAMATH_GPT_hyperbola_sufficiency_l448_44883

open Real

theorem hyperbola_sufficiency (k : ℝ) : 
  (9 - k < 0 ∧ k - 4 > 0) → 
  (∃ x y : ℝ, (x^2) / (9 - k) + (y^2) / (k - 4) = 1) :=
by
  intro hk
  sorry

end NUMINAMATH_GPT_hyperbola_sufficiency_l448_44883


namespace NUMINAMATH_GPT_mod_equiv_l448_44894

theorem mod_equiv :
  241 * 398 % 50 = 18 :=
by
  sorry

end NUMINAMATH_GPT_mod_equiv_l448_44894


namespace NUMINAMATH_GPT_increase_average_by_runs_l448_44824

theorem increase_average_by_runs :
  let total_runs_10_matches : ℕ := 10 * 32
  let runs_scored_next_match : ℕ := 87
  let total_runs_11_matches : ℕ := total_runs_10_matches + runs_scored_next_match
  let new_average_11_matches : ℚ := total_runs_11_matches / 11
  let increased_average : ℚ := 32 + 5
  new_average_11_matches = increased_average :=
by
  sorry

end NUMINAMATH_GPT_increase_average_by_runs_l448_44824


namespace NUMINAMATH_GPT_Hari_contribution_l448_44854

theorem Hari_contribution (H : ℕ) (Praveen_capital : ℕ := 3500) (months_Praveen : ℕ := 12) 
                          (months_Hari : ℕ := 7) (profit_ratio_P : ℕ := 2) (profit_ratio_H : ℕ := 3) : 
                          (Praveen_capital * months_Praveen) * profit_ratio_H = (H * months_Hari) * profit_ratio_P → 
                          H = 9000 :=
by
  sorry

end NUMINAMATH_GPT_Hari_contribution_l448_44854


namespace NUMINAMATH_GPT_min_chemistry_teachers_l448_44864

/--
A school has 7 maths teachers, 6 physics teachers, and some chemistry teachers.
Each teacher can teach a maximum of 3 subjects.
The minimum number of teachers required is 6.
Prove that the minimum number of chemistry teachers required is 1.
-/
theorem min_chemistry_teachers (C : ℕ) (math_teachers : ℕ := 7) (physics_teachers : ℕ := 6) 
  (max_subjects_per_teacher : ℕ := 3) (min_teachers_required : ℕ := 6) :
  7 + 6 + C ≤ 6 * 3 → C = 1 := 
by
  sorry

end NUMINAMATH_GPT_min_chemistry_teachers_l448_44864


namespace NUMINAMATH_GPT_solve_farm_l448_44899

def farm_problem (P H L T : ℕ) : Prop :=
  L = 4 * P + 2 * H ∧
  T = P + H ∧
  L = 3 * T + 36 →
  P = H + 36

-- Theorem statement
theorem solve_farm : ∃ P H L T : ℕ, farm_problem P H L T :=
by sorry

end NUMINAMATH_GPT_solve_farm_l448_44899


namespace NUMINAMATH_GPT_strawberry_jelly_sales_l448_44849

def jelly_sales (grape strawberry raspberry plum : ℕ) : Prop :=
  grape = 2 * strawberry ∧
  raspberry = 2 * plum ∧
  raspberry = grape / 3 ∧
  plum = 6

theorem strawberry_jelly_sales {grape strawberry raspberry plum : ℕ}
    (h : jelly_sales grape strawberry raspberry plum) : 
    strawberry = 18 :=
by
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2.1
  have h4 := h.2.2.2
  sorry

end NUMINAMATH_GPT_strawberry_jelly_sales_l448_44849


namespace NUMINAMATH_GPT_bad_carrots_count_l448_44809

-- Define the number of carrots each person picked and the number of good carrots
def carol_picked := 29
def mom_picked := 16
def good_carrots := 38

-- Define the total number of carrots picked and the total number of bad carrots
def total_carrots := carol_picked + mom_picked
def bad_carrots := total_carrots - good_carrots

-- State the theorem that the number of bad carrots is 7
theorem bad_carrots_count :
  bad_carrots = 7 :=
by
  sorry

end NUMINAMATH_GPT_bad_carrots_count_l448_44809


namespace NUMINAMATH_GPT_find_m_for_all_n_l448_44814

def sum_of_digits (k: ℕ) : ℕ :=
  k.digits 10 |>.sum

def A (k: ℕ) : ℕ :=
  -- Constructing the number A_k as described
  -- This is a placeholder for the actual implementation
  sorry

theorem find_m_for_all_n (n: ℕ) (hn: 0 < n) :
  ∃ m: ℕ, 0 < m ∧ n ∣ A m ∧ n ∣ m ∧ n ∣ sum_of_digits (A m) :=
sorry

end NUMINAMATH_GPT_find_m_for_all_n_l448_44814


namespace NUMINAMATH_GPT_equal_pair_b_l448_44801

def exprA1 := -3^2
def exprA2 := -2^3

def exprB1 := -6^3
def exprB2 := (-6)^3

def exprC1 := -6^2
def exprC2 := (-6)^2

def exprD1 := (-3 * 2)^2
def exprD2 := (-3) * 2^2

theorem equal_pair_b : exprB1 = exprB2 :=
by {
  -- proof steps should go here
  sorry
}

end NUMINAMATH_GPT_equal_pair_b_l448_44801


namespace NUMINAMATH_GPT_items_left_in_cart_l448_44856

-- Define the initial items in the shopping cart
def initial_items : ℕ := 18

-- Define the items deleted from the shopping cart
def deleted_items : ℕ := 10

-- Theorem statement: Prove the remaining items are 8
theorem items_left_in_cart : initial_items - deleted_items = 8 :=
by
  -- Sorry marks the place where the proof would go.
  sorry

end NUMINAMATH_GPT_items_left_in_cart_l448_44856


namespace NUMINAMATH_GPT_entrance_fee_per_person_l448_44874

theorem entrance_fee_per_person :
  let ticket_price := 50.00
  let processing_fee_rate := 0.15
  let parking_fee := 10.00
  let total_cost := 135.00
  let known_cost := 2 * ticket_price + processing_fee_rate * (2 * ticket_price) + parking_fee
  ∃ entrance_fee_per_person, 2 * entrance_fee_per_person + known_cost = total_cost :=
by
  sorry

end NUMINAMATH_GPT_entrance_fee_per_person_l448_44874


namespace NUMINAMATH_GPT_hyperbola_center_l448_44898

-- Definitions based on conditions
def hyperbola (x y : ℝ) : Prop := ((4 * x + 8) ^ 2 / 16) - ((5 * y - 5) ^ 2 / 25) = 1

-- Theorem statement
theorem hyperbola_center : ∀ x y : ℝ, hyperbola x y → (x, y) = (-2, 1) := 
  by
    sorry

end NUMINAMATH_GPT_hyperbola_center_l448_44898


namespace NUMINAMATH_GPT_hotel_total_towels_l448_44888

theorem hotel_total_towels :
  let rooms_A := 25
  let rooms_B := 30
  let rooms_C := 15
  let members_per_room_A := 5
  let members_per_room_B := 6
  let members_per_room_C := 4
  let towels_per_member_A := 3
  let towels_per_member_B := 2
  let towels_per_member_C := 4
  (rooms_A * members_per_room_A * towels_per_member_A) +
  (rooms_B * members_per_room_B * towels_per_member_B) +
  (rooms_C * members_per_room_C * towels_per_member_C) = 975
:= by
  sorry

end NUMINAMATH_GPT_hotel_total_towels_l448_44888


namespace NUMINAMATH_GPT_measuring_cup_size_l448_44895

-- Defining the conditions
def total_flour := 8
def flour_needed := 6
def scoops_removed := 8 

-- Defining the size of the cup
def cup_size (x : ℚ) := 8 - scoops_removed * x = flour_needed

-- Stating the theorem
theorem measuring_cup_size : ∃ x : ℚ, cup_size x ∧ x = 1 / 4 :=
by {
    sorry
}

end NUMINAMATH_GPT_measuring_cup_size_l448_44895


namespace NUMINAMATH_GPT_arithmetic_sequence_correct_l448_44882

-- Define the conditions
def last_term_eq_num_of_terms (a l n : Int) : Prop := l = n
def common_difference (d : Int) : Prop := d = 5
def sum_of_sequence (n a S : Int) : Prop :=
  S = n * (2 * a + (n - 1) * 5) / 2

-- The target arithmetic sequence
def seq : List Int := [-7, -2, 3]
def first_term : Int := -7
def num_terms : Int := 3
def sum_of_seq : Int := -6

-- Proof statement
theorem arithmetic_sequence_correct :
  last_term_eq_num_of_terms first_term seq.length num_terms ∧
  common_difference 5 ∧
  sum_of_sequence seq.length first_term sum_of_seq →
  seq = [-7, -2, 3] :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_correct_l448_44882


namespace NUMINAMATH_GPT_quadratic_root_sum_m_n_l448_44862

theorem quadratic_root_sum_m_n (m n : ℤ) :
  (∃ x : ℤ, x^2 + m * x + 2 * n = 0 ∧ x = 2) → m + n = -2 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_sum_m_n_l448_44862


namespace NUMINAMATH_GPT_alpha_beta_sum_two_l448_44886

theorem alpha_beta_sum_two (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 17 = 0)
  (hβ : β^3 - 3 * β^2 + 5 * β + 11 = 0) : 
  α + β = 2 :=
by
  sorry

end NUMINAMATH_GPT_alpha_beta_sum_two_l448_44886


namespace NUMINAMATH_GPT_molecular_weight_one_mole_l448_44870

theorem molecular_weight_one_mole
  (molecular_weight_7_moles : ℝ)
  (mole_count : ℝ)
  (h : molecular_weight_7_moles = 126)
  (k : mole_count = 7)
  : molecular_weight_7_moles / mole_count = 18 := 
sorry

end NUMINAMATH_GPT_molecular_weight_one_mole_l448_44870


namespace NUMINAMATH_GPT_correct_propositions_count_l448_44806

theorem correct_propositions_count (x y : ℝ) :
  (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0) ∧ -- original proposition
  (x^2 + y^2 ≠ 0) → (x ≠ 0 ∨ y ≠ 0) ∧ -- converse proposition
  (¬(x ≠ 0 ∨ y ≠ 0) ∨ x^2 + y^2 = 0) ∧ -- negation proposition
  (¬(x^2 + y^2 = 0) ∨ x ≠ 0 ∨ y ≠ 0) -- inverse proposition
  := by
  sorry

end NUMINAMATH_GPT_correct_propositions_count_l448_44806


namespace NUMINAMATH_GPT_rectangle_area_error_83_percent_l448_44847

theorem rectangle_area_error_83_percent (L W : ℝ) :
  let actual_area := L * W
  let measured_length := 1.14 * L
  let measured_width := 0.95 * W
  let calculated_area := measured_length * measured_width
  let error := calculated_area - actual_area
  (error / actual_area) * 100 = 8.3 := by
  sorry

end NUMINAMATH_GPT_rectangle_area_error_83_percent_l448_44847


namespace NUMINAMATH_GPT_weight_of_pants_l448_44802

def weight_socks := 2
def weight_underwear := 4
def weight_shirt := 5
def weight_shorts := 8
def total_allowed := 50

def weight_total (num_shirts num_shorts num_socks num_underwear : Nat) :=
  num_shirts * weight_shirt + num_shorts * weight_shorts + num_socks * weight_socks + num_underwear * weight_underwear

def items_in_wash := weight_total 2 1 3 4

theorem weight_of_pants :
  let weight_pants := total_allowed - items_in_wash
  weight_pants = 10 :=
by
  sorry

end NUMINAMATH_GPT_weight_of_pants_l448_44802


namespace NUMINAMATH_GPT_f_minus_5_eq_12_l448_44858

def f (x : ℝ) : ℝ := x^2 + 2*x - 3

theorem f_minus_5_eq_12 : f (-5) = 12 := 
by sorry

end NUMINAMATH_GPT_f_minus_5_eq_12_l448_44858


namespace NUMINAMATH_GPT_truck_driver_gas_l448_44835

variables (miles_per_gallon distance_to_station gallons_to_add gallons_in_tank total_gallons_needed : ℕ)
variables (current_gas_in_tank : ℕ)
variables (h1 : miles_per_gallon = 3)
variables (h2 : distance_to_station = 90)
variables (h3 : gallons_to_add = 18)

theorem truck_driver_gas :
  current_gas_in_tank = 12 :=
by
  -- Prove that the truck driver already has 12 gallons of gas in his tank,
  -- given the conditions provided.
  sorry

end NUMINAMATH_GPT_truck_driver_gas_l448_44835


namespace NUMINAMATH_GPT_math_proof_problem_l448_44842

-- Define constants
def x := 2000000000000
def y := 1111111111111

-- Prove the main statement
theorem math_proof_problem :
  2 * (x - y) = 1777777777778 := 
  by
    sorry

end NUMINAMATH_GPT_math_proof_problem_l448_44842


namespace NUMINAMATH_GPT_bank_transfer_amount_l448_44866

/-- Paul made two bank transfers. A service charge of 2% was added to each transaction.
The second transaction was reversed without the service charge. His account balance is now $307 if 
it was $400 before he made any transfers. Prove that the amount of the first bank transfer was 
$91.18. -/
theorem bank_transfer_amount (x : ℝ) (initial_balance final_balance : ℝ) (service_charge_rate : ℝ) 
  (second_transaction_reversed : Prop)
  (h_initial : initial_balance = 400)
  (h_final : final_balance = 307)
  (h_charge : service_charge_rate = 0.02)
  (h_reversal : second_transaction_reversed):
  initial_balance - (1 + service_charge_rate) * x = final_balance ↔
  x = 91.18 := 
by
  sorry

end NUMINAMATH_GPT_bank_transfer_amount_l448_44866


namespace NUMINAMATH_GPT_negation_of_proposition_l448_44839

theorem negation_of_proposition :
  (¬ ∀ (x : ℝ), |x| < 0) ↔ (∃ (x : ℝ), |x| ≥ 0) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l448_44839


namespace NUMINAMATH_GPT_greg_needs_additional_amount_l448_44853

def total_cost : ℤ := 90
def saved_amount : ℤ := 57
def additional_amount_needed : ℤ := total_cost - saved_amount

theorem greg_needs_additional_amount :
  additional_amount_needed = 33 :=
by
  sorry

end NUMINAMATH_GPT_greg_needs_additional_amount_l448_44853


namespace NUMINAMATH_GPT_find_x_solution_l448_44855

theorem find_x_solution :
  ∃ x, 2 ^ (x / 2) * (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x)) = 6 ∧
       x = 2 * Real.log 1.5 / Real.log 2 := by
  sorry

end NUMINAMATH_GPT_find_x_solution_l448_44855


namespace NUMINAMATH_GPT_total_water_capacity_l448_44818

-- Define the given conditions as constants
def numTrucks : ℕ := 5
def tanksPerTruck : ℕ := 4
def capacityPerTank : ℕ := 200

-- Define the claim as a theorem
theorem total_water_capacity :
  numTrucks * (tanksPerTruck * capacityPerTank) = 4000 :=
by
  sorry

end NUMINAMATH_GPT_total_water_capacity_l448_44818


namespace NUMINAMATH_GPT_red_balls_in_box_l448_44885

theorem red_balls_in_box {n : ℕ} (h : n = 6) (p : (∃ (r : ℕ), r / 6 = 1 / 3)) : ∃ r, r = 2 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_in_box_l448_44885


namespace NUMINAMATH_GPT_correct_equation_for_gift_exchanges_l448_44889

theorem correct_equation_for_gift_exchanges
  (x : ℕ)
  (H : (x * (x - 1)) = 56) :
  x * (x - 1) = 56 := 
by 
  exact H

end NUMINAMATH_GPT_correct_equation_for_gift_exchanges_l448_44889


namespace NUMINAMATH_GPT_penguin_seafood_protein_l448_44819

theorem penguin_seafood_protein
  (digest : ℝ) -- representing 30% 
  (digested : ℝ) -- representing 9 grams 
  (h : digest = 0.30) 
  (h1 : digested = 9) :
  ∃ x : ℝ, digested = digest * x ∧ x = 30 :=
by
  sorry

end NUMINAMATH_GPT_penguin_seafood_protein_l448_44819


namespace NUMINAMATH_GPT_interval_of_monotonic_increase_sum_greater_than_2e_l448_44867

noncomputable def f (a x : ℝ) : ℝ := a * x / (Real.log x)

theorem interval_of_monotonic_increase :
  ∀ (x : ℝ), (e < x → f 1 x > f 1 e) := 
sorry

theorem sum_greater_than_2e (x1 x2 : ℝ) (a : ℝ) (h1 : x1 ≠ x2) (hx1 : f 1 x1 = 1) (hx2 : f 1 x2 = 1) :
  x1 + x2 > 2 * Real.exp 1 :=
sorry

end NUMINAMATH_GPT_interval_of_monotonic_increase_sum_greater_than_2e_l448_44867


namespace NUMINAMATH_GPT_sum_fraction_series_eq_l448_44877

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end NUMINAMATH_GPT_sum_fraction_series_eq_l448_44877


namespace NUMINAMATH_GPT_number_of_women_in_first_class_l448_44812

-- Definitions for the conditions
def total_passengers : ℕ := 180
def percentage_women : ℝ := 0.65
def percentage_women_first_class : ℝ := 0.15

-- The desired proof statement
theorem number_of_women_in_first_class :
  (round (total_passengers * percentage_women * percentage_women_first_class) = 18) :=
by
  sorry  

end NUMINAMATH_GPT_number_of_women_in_first_class_l448_44812


namespace NUMINAMATH_GPT_value_of_expression_l448_44803

theorem value_of_expression : 2 * 2015 - 2015 = 2015 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l448_44803


namespace NUMINAMATH_GPT_smallest_angle_of_triangle_l448_44827

noncomputable def smallest_angle (a b : ℝ) (c : ℝ) (h_sum : a + b + c = 180) : ℝ :=
  min a (min b c)

theorem smallest_angle_of_triangle :
  smallest_angle 60 65 (180 - (60 + 65)) (by norm_num) = 55 :=
by
  -- The correct proof steps should be provided for the result
  sorry

end NUMINAMATH_GPT_smallest_angle_of_triangle_l448_44827


namespace NUMINAMATH_GPT_mary_visited_two_shops_l448_44873

-- Define the costs of items
def cost_shirt : ℝ := 13.04
def cost_jacket : ℝ := 12.27
def total_cost : ℝ := 25.31

-- Define the number of shops visited
def number_of_shops : ℕ := 2

-- Proof that Mary visited 2 shops given the conditions
theorem mary_visited_two_shops (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) (h_total : cost_shirt + cost_jacket = total_cost) : number_of_shops = 2 :=
by
  sorry

end NUMINAMATH_GPT_mary_visited_two_shops_l448_44873


namespace NUMINAMATH_GPT_fourth_buoy_distance_with_current_l448_44890

-- Define the initial conditions
def first_buoy_distance : ℕ := 20
def second_buoy_additional_distance : ℕ := 24
def third_buoy_additional_distance : ℕ := 28
def common_difference_increment : ℕ := 4
def ocean_current_push_per_segment : ℕ := 3
def number_of_segments : ℕ := 3

-- Define the mathematical proof problem
theorem fourth_buoy_distance_with_current :
  let fourth_buoy_additional_distance := third_buoy_additional_distance + common_difference_increment
  let first_to_second_buoy := first_buoy_distance + second_buoy_additional_distance
  let second_to_third_buoy := first_to_second_buoy + third_buoy_additional_distance
  let distance_before_current := second_to_third_buoy + fourth_buoy_additional_distance
  let total_current_push := ocean_current_push_per_segment * number_of_segments
  let final_distance := distance_before_current - total_current_push
  final_distance = 95 := by
  sorry

end NUMINAMATH_GPT_fourth_buoy_distance_with_current_l448_44890


namespace NUMINAMATH_GPT_even_and_nonneg_range_l448_44850

theorem even_and_nonneg_range : 
  (∀ x : ℝ, abs x = abs (-x) ∧ (abs x ≥ 0)) ∧ (∀ x : ℝ, x^2 + abs x = ( (-x)^2) + abs (-x) ∧ (x^2 + abs x ≥ 0)) := sorry

end NUMINAMATH_GPT_even_and_nonneg_range_l448_44850


namespace NUMINAMATH_GPT_pens_at_end_l448_44859

-- Define the main variable
variable (x : ℝ)

-- Define the conditions as functions
def initial_pens (x : ℝ) := x
def mike_gives (x : ℝ) := 0.5 * x
def after_mike (x : ℝ) := x + (mike_gives x)
def after_cindy (x : ℝ) := 2 * (after_mike x)
def give_sharon (x : ℝ) := 0.25 * (after_cindy x)

-- Define the final number of pens
def final_pens (x : ℝ) := (after_cindy x) - (give_sharon x)

-- The theorem statement
theorem pens_at_end (x : ℝ) : final_pens x = 2.25 * x :=
by sorry

end NUMINAMATH_GPT_pens_at_end_l448_44859


namespace NUMINAMATH_GPT_pto_shirts_total_cost_l448_44825

theorem pto_shirts_total_cost :
  let cost_Kindergartners : ℝ := 101 * 5.80
  let cost_FirstGraders : ℝ := 113 * 5.00
  let cost_SecondGraders : ℝ := 107 * 5.60
  let cost_ThirdGraders : ℝ := 108 * 5.25
  cost_Kindergartners + cost_FirstGraders + cost_SecondGraders + cost_ThirdGraders = 2317.00 := by
  sorry

end NUMINAMATH_GPT_pto_shirts_total_cost_l448_44825


namespace NUMINAMATH_GPT_cake_pieces_kept_l448_44868

theorem cake_pieces_kept (total_pieces : ℕ) (two_fifths_eaten : ℕ) (extra_pieces_eaten : ℕ)
  (h1 : total_pieces = 35)
  (h2 : two_fifths_eaten = 2 * total_pieces / 5)
  (h3 : extra_pieces_eaten = 3)
  (correct_answer : ℕ)
  (h4 : correct_answer = total_pieces - (two_fifths_eaten + extra_pieces_eaten)) :
  correct_answer = 18 := by
  sorry

end NUMINAMATH_GPT_cake_pieces_kept_l448_44868


namespace NUMINAMATH_GPT_remainder_of_3_pow_800_mod_17_l448_44800

theorem remainder_of_3_pow_800_mod_17 : (3^800) % 17 = 1 := by
  sorry

end NUMINAMATH_GPT_remainder_of_3_pow_800_mod_17_l448_44800


namespace NUMINAMATH_GPT_bruce_mango_purchase_l448_44829

theorem bruce_mango_purchase (m : ℕ) 
  (cost_grapes : 8 * 70 = 560)
  (cost_total : 560 + 55 * m = 1110) : 
  m = 10 :=
by
  sorry

end NUMINAMATH_GPT_bruce_mango_purchase_l448_44829


namespace NUMINAMATH_GPT_complement_intersection_complement_l448_44838

-- Define the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Define the statement of the proof problem
theorem complement_intersection_complement:
  (U \ (A ∩ B)) = {1, 4, 6} := by
  sorry

end NUMINAMATH_GPT_complement_intersection_complement_l448_44838


namespace NUMINAMATH_GPT_intersection_and_perpendicular_line_l448_44846

theorem intersection_and_perpendicular_line :
  ∃ (x y : ℝ), (3 * x + y - 1 = 0) ∧ (x + 2 * y - 7 = 0) ∧ (2 * x - y + 6 = 0) :=
by
  sorry

end NUMINAMATH_GPT_intersection_and_perpendicular_line_l448_44846


namespace NUMINAMATH_GPT_part1_1_part1_2_part1_3_part2_l448_44833

def operation (a b c : ℝ) : Prop := a^c = b

theorem part1_1 : operation 3 81 4 :=
by sorry

theorem part1_2 : operation 4 1 0 :=
by sorry

theorem part1_3 : operation 2 (1 / 4) (-2) :=
by sorry

theorem part2 (x y z : ℝ) (h1 : operation 3 7 x) (h2 : operation 3 8 y) (h3 : operation 3 56 z) : x + y = z :=
by sorry

end NUMINAMATH_GPT_part1_1_part1_2_part1_3_part2_l448_44833


namespace NUMINAMATH_GPT_proof_equilateral_inscribed_circle_l448_44878

variables {A B C : Type*}
variables (r : ℝ) (D : ℝ)

def is_equilateral_triangle (A B C : Type*) : Prop := 
  -- Define the equilateral condition, where all sides are equal
  true

def is_inscribed_circle_radius (D r : ℝ) : Prop := 
  -- Define the property that D is the center and r is the radius 
  true

def distance_center_to_vertex (D r x : ℝ) : Prop := 
  x = 3 * r

theorem proof_equilateral_inscribed_circle 
  (A B C : Type*) 
  (r D : ℝ) 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_inscribed_circle_radius D r) : 
  distance_center_to_vertex D r (1 / 16) :=
by sorry

end NUMINAMATH_GPT_proof_equilateral_inscribed_circle_l448_44878


namespace NUMINAMATH_GPT_susie_rooms_l448_44828

-- Define the conditions
def vacuum_time_per_room : ℕ := 20  -- in minutes
def total_vacuum_time : ℕ := 2 * 60  -- 2 hours in minutes

-- Define the number of rooms in Susie's house
def number_of_rooms (total_time room_time : ℕ) : ℕ := total_time / room_time

-- Prove that Susie has 6 rooms in her house
theorem susie_rooms : number_of_rooms total_vacuum_time vacuum_time_per_room = 6 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_susie_rooms_l448_44828


namespace NUMINAMATH_GPT_crackers_per_box_l448_44840

-- Given conditions
variables (x : ℕ)
variable (darren_boxes : ℕ := 4)
variable (calvin_boxes : ℕ := 2 * darren_boxes - 1)
variable (total_crackers : ℕ := 264)

-- Using the given conditions, create the proof statement to show x = 24
theorem crackers_per_box:
  11 * x = total_crackers → x = 24 :=
by
  sorry

end NUMINAMATH_GPT_crackers_per_box_l448_44840


namespace NUMINAMATH_GPT_bacteria_exceeds_day_l448_44844

theorem bacteria_exceeds_day :
  ∃ n : ℕ, 5 * 3^n > 200 ∧ ∀ m : ℕ, (m < n → 5 * 3^m ≤ 200) :=
sorry

end NUMINAMATH_GPT_bacteria_exceeds_day_l448_44844


namespace NUMINAMATH_GPT_cone_base_radius_l448_44848

-- Definitions based on conditions
def sphere_radius : ℝ := 1
def cone_height : ℝ := 2

-- Problem statement
theorem cone_base_radius {r : ℝ} 
  (h1 : ∀ x y z : ℝ, (x = sphere_radius ∧ y = sphere_radius ∧ z = sphere_radius) → 
                     (x + y + z = 3 * sphere_radius)) 
  (h2 : ∃ (O O1 O2 O3 : ℝ), (O = 0) ∧ (O1 = 1) ∧ (O2 = 1) ∧ (O3 = 1)) 
  (h3 : ∀ x y z : ℝ, (x + y + z = 3 * sphere_radius) → 
                     (y = z) → (x = z) → y * z + x * z + x * y = 3 * sphere_radius ^ 2)
  (h4 : ∀ h : ℝ, h = cone_height) :
  r = (Real.sqrt 3 / 6) :=
sorry

end NUMINAMATH_GPT_cone_base_radius_l448_44848


namespace NUMINAMATH_GPT_part1_part2_l448_44876

open Set

def A : Set ℝ := {x | x^2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem part1 : B (1/5) ⊆ A ∧ ¬ A ⊆ B (1/5) := by
  sorry
  
theorem part2 (a : ℝ) : (B a ⊆ A) ↔ a ∈ ({0, 1/3, 1/5} : Set ℝ) := by
  sorry

end NUMINAMATH_GPT_part1_part2_l448_44876


namespace NUMINAMATH_GPT_same_speed_is_4_l448_44860

namespace SpeedProof

theorem same_speed_is_4 (x : ℝ) (h_jack_speed : x^2 - 11 * x - 22 = x - 10) (h_jill_speed : x^2 - 5 * x - 60 = (x - 10) * (x + 6)) :
  x = 14 → (x - 10) = 4 :=
by
  sorry

end SpeedProof

end NUMINAMATH_GPT_same_speed_is_4_l448_44860


namespace NUMINAMATH_GPT_order_of_numbers_l448_44836

noncomputable def a : ℝ := 60.7
noncomputable def b : ℝ := 0.76
noncomputable def c : ℝ := Real.log 0.76

theorem order_of_numbers : (c < b) ∧ (b < a) :=
by
  have h1 : c = Real.log 0.76 := rfl
  have h2 : b = 0.76 := rfl
  have h3 : a = 60.7 := rfl
  have hc : c < 0 := sorry
  have hb : 0 < b := sorry
  have ha : 1 < a := sorry
  sorry 

end NUMINAMATH_GPT_order_of_numbers_l448_44836


namespace NUMINAMATH_GPT_correct_random_error_causes_l448_44815

-- Definitions based on conditions
def is_random_error_cause (n : ℕ) : Prop :=
  n = 1 ∨ n = 2 ∨ n = 3

-- Theorem: Valid causes of random errors are options (1), (2), and (3)
theorem correct_random_error_causes :
  (is_random_error_cause 1) ∧ (is_random_error_cause 2) ∧ (is_random_error_cause 3) :=
by
  sorry

end NUMINAMATH_GPT_correct_random_error_causes_l448_44815


namespace NUMINAMATH_GPT_symmetric_point_origin_l448_44852

theorem symmetric_point_origin (x y : ℤ) (h : x = -2 ∧ y = 3) :
    (-x, -y) = (2, -3) :=
by
  cases h with
  | intro hx hy =>
  simp only [hx, hy]
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l448_44852


namespace NUMINAMATH_GPT_num_four_digit_int_with_4_or_5_correct_l448_44861

def num_four_digit_int_with_4_or_5 : ℕ :=
  5416

theorem num_four_digit_int_with_4_or_5_correct (A B : ℕ) (hA : A = 9000) (hB : B = 3584) :
  num_four_digit_int_with_4_or_5 = A - B :=
by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_num_four_digit_int_with_4_or_5_correct_l448_44861


namespace NUMINAMATH_GPT_cream_ratio_l448_44891

noncomputable def John_creme_amount : ℚ := 3
noncomputable def Janet_initial_amount : ℚ := 8
noncomputable def Janet_creme_added : ℚ := 3
noncomputable def Janet_total_mixture : ℚ := Janet_initial_amount + Janet_creme_added
noncomputable def Janet_creme_ratio : ℚ := Janet_creme_added / Janet_total_mixture
noncomputable def Janet_drank_amount : ℚ := 3
noncomputable def Janet_drank_creme : ℚ := Janet_drank_amount * Janet_creme_ratio
noncomputable def Janet_creme_remaining : ℚ := Janet_creme_added - Janet_drank_creme

theorem cream_ratio :
  (John_creme_amount / Janet_creme_remaining) = (11 / 5) :=
by
  sorry

end NUMINAMATH_GPT_cream_ratio_l448_44891


namespace NUMINAMATH_GPT_difference_fraction_reciprocal_l448_44826

theorem difference_fraction_reciprocal :
  let f := (4 : ℚ) / 5
  let r := (5 : ℚ) / 4
  f - r = 9 / 20 :=
by
  sorry

end NUMINAMATH_GPT_difference_fraction_reciprocal_l448_44826


namespace NUMINAMATH_GPT_semi_minor_axis_l448_44807

theorem semi_minor_axis (a c : ℝ) (h_a : a = 5) (h_c : c = 2) : 
  ∃ b : ℝ, b = Real.sqrt (a^2 - c^2) ∧ b = Real.sqrt 21 :=
by
  use Real.sqrt 21
  sorry

end NUMINAMATH_GPT_semi_minor_axis_l448_44807


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_x_1_l448_44869

noncomputable def sufficient_but_not_necessary_condition (x : ℝ) : Prop :=
(x = 1 → (x = 1 ∨ x = 2)) ∧ ¬ ((x = 1 ∨ x = 2) → x = 1)

theorem sufficient_but_not_necessary_condition_for_x_1 :
  sufficient_but_not_necessary_condition 1 :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_for_x_1_l448_44869


namespace NUMINAMATH_GPT_expression_value_l448_44843

variable (m n : ℝ)

theorem expression_value (h : m - n = 1) : (m - n)^2 - 2 * m + 2 * n = -1 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l448_44843


namespace NUMINAMATH_GPT_ivy_collectors_edition_dolls_l448_44821

-- Definitions from the conditions
def dina_dolls : ℕ := 60
def ivy_dolls : ℕ := dina_dolls / 2
def collectors_edition_dolls : ℕ := (2 * ivy_dolls) / 3

-- Assertion
theorem ivy_collectors_edition_dolls : collectors_edition_dolls = 20 := by
  sorry

end NUMINAMATH_GPT_ivy_collectors_edition_dolls_l448_44821


namespace NUMINAMATH_GPT_triangle_AC_5_sqrt_3_l448_44816

theorem triangle_AC_5_sqrt_3 
  (A B C : ℝ)
  (BC AC : ℝ)
  (h1 : 2 * Real.sin (A - B) + Real.cos (B + C) = 2)
  (h2 : BC = 5) :
  AC = 5 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_GPT_triangle_AC_5_sqrt_3_l448_44816


namespace NUMINAMATH_GPT_max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l448_44817

noncomputable def y (x : ℝ) : ℝ := 3 * x + 4 / x
def max_value (x : ℝ) := y x ≤ -4 * Real.sqrt 3

theorem max_y_value_of_3x_plus_4_div_x (h : x < 0) : max_value x :=
sorry

theorem corresponds_value_of_x (x : ℝ) (h : x = -2 * Real.sqrt 3 / 3) : y x = -4 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_max_y_value_of_3x_plus_4_div_x_corresponds_value_of_x_l448_44817


namespace NUMINAMATH_GPT_part1_solution_part2_solution_l448_44837

noncomputable def part1_expr := (1 / (Real.sqrt 5 + 2)) - (Real.sqrt 3 - 1)^0 - Real.sqrt (9 - 4 * Real.sqrt 5)
theorem part1_solution : part1_expr = 2 := by
  sorry

noncomputable def part2_expr := 2 * Real.sqrt 3 * 612 * (7/2)
theorem part2_solution : part2_expr = 5508 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_part1_solution_part2_solution_l448_44837


namespace NUMINAMATH_GPT_smallest_perimeter_l448_44865

theorem smallest_perimeter (m n : ℕ) 
  (h1 : (m - 4) * (n - 4) = 8) 
  (h2 : ∀ k l : ℕ, (k - 4) * (l - 4) = 8 → 2 * k + 2 * l ≥ 2 * m + 2 * n) : 
  (m = 6 ∧ n = 8) ∨ (m = 8 ∧ n = 6) :=
sorry

end NUMINAMATH_GPT_smallest_perimeter_l448_44865


namespace NUMINAMATH_GPT_track_and_field_analysis_l448_44871

theorem track_and_field_analysis :
  let male_athletes := 12
  let female_athletes := 8
  let tallest_height := 190
  let shortest_height := 160
  let avg_male_height := 175
  let avg_female_height := 165
  let total_athletes := male_athletes + female_athletes
  let sample_size := 10
  let prob_selected := 1 / 2
  let prop_male := male_athletes / total_athletes * sample_size
  let prop_female := female_athletes / total_athletes * sample_size
  let overall_avg_height := (male_athletes / total_athletes) * avg_male_height + (female_athletes / total_athletes) * avg_female_height
  (tallest_height - shortest_height = 30) ∧
  (sample_size / total_athletes = prob_selected) ∧
  (prop_male = 6 ∧ prop_female = 4) ∧
  (overall_avg_height = 171) →
  (A = true ∧ B = true ∧ C = false ∧ D = true) :=
by
  sorry

end NUMINAMATH_GPT_track_and_field_analysis_l448_44871


namespace NUMINAMATH_GPT_balloon_ratio_l448_44893

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_balloon_ratio_l448_44893


namespace NUMINAMATH_GPT_cylinder_height_and_diameter_l448_44863

/-- The surface area of a sphere is the same as the curved surface area of a right circular cylinder.
    The height and diameter of the cylinder are the same, and the radius of the sphere is 4 cm.
    Prove that the height and diameter of the cylinder are both 8 cm. --/
theorem cylinder_height_and_diameter (r_sphere : ℝ) (r_cylinder h_cylinder : ℝ)
  (h1 : r_sphere = 4)
  (h2 : 4 * π * r_sphere^2 = 2 * π * r_cylinder * h_cylinder)
  (h3 : h_cylinder = 2 * r_cylinder) :
  h_cylinder = 8 ∧ r_cylinder = 4 :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_cylinder_height_and_diameter_l448_44863


namespace NUMINAMATH_GPT_ball_hits_ground_l448_44831

theorem ball_hits_ground (t : ℚ) : 
  (∃ t ≥ 0, (-4.9 * (t^2 : ℝ) + 5 * t + 10 = 0)) → t = 100 / 49 :=
by
  sorry

end NUMINAMATH_GPT_ball_hits_ground_l448_44831


namespace NUMINAMATH_GPT_parabola_vertex_on_x_axis_l448_44851

theorem parabola_vertex_on_x_axis (c : ℝ) : 
  (∃ h k : ℝ, y = (x : ℝ)^2 - 12 * x + c ∧
   (h = -12 / 2) ∧
   (k = c - 144 / 4) ∧
   (k = 0)) ↔ c = 36 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_on_x_axis_l448_44851


namespace NUMINAMATH_GPT_cost_comparison_for_30_pens_l448_44887

def cost_store_a (x : ℕ) : ℝ :=
  if x > 10 then 0.9 * x + 6
  else 1.5 * x

def cost_store_b (x : ℕ) : ℝ :=
  1.2 * x

theorem cost_comparison_for_30_pens :
  cost_store_a 30 < cost_store_b 30 :=
by
  have store_a_cost : cost_store_a 30 = 0.9 * 30 + 6 := by rfl
  have store_b_cost : cost_store_b 30 = 1.2 * 30 := by rfl
  rw [store_a_cost, store_b_cost]
  sorry

end NUMINAMATH_GPT_cost_comparison_for_30_pens_l448_44887
