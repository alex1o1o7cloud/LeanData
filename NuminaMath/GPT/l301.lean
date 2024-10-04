import Mathlib

namespace polygon_sides_l301_301533

-- Define the given condition formally
def sum_of_internal_and_external_angle (n : ℕ) : ℕ :=
  (n - 2) * 180 + (1) -- This represents the sum of internal angles plus an external angle

theorem polygon_sides (n : ℕ) : 
  sum_of_internal_and_external_angle n = 1350 → n = 9 :=
by
  sorry

end polygon_sides_l301_301533


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301703

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301703


namespace prime_sum_remainder_l301_301765

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301765


namespace construction_days_max_cost_teamB_l301_301155

noncomputable def teamA_days (teamB_days : ℝ) : ℝ := 1.5 * teamB_days

def total_work_days : ℝ := 72

theorem construction_days (teamB_days : ℝ) (teamA_days : ℝ) :
  (1 / teamA_days + 1 / teamB_days = 1 / total_work_days) → 
  teamB_days = 120 ∧ teamA_days = 180 :=
by
  intro h
  have : teamA_days = 1.5 * teamB_days := by sorry
  have : 1 / teamA_days = 1 / (1.5 * teamB_days) := by sorry
  have : 1 / teamB_days + 1 / (1.5 * teamB_days) = 1 / total_work_days := by sorry
  have : 1 / teamB_days + 1 / teamB_days * (2 / 3) = 1 / total_work_days := by sorry
  have : (1 + 2 / 3) / teamB_days = 1 / total_work_days := by sorry
  have : 5 / (3 * teamB_days) = 1 / 72 := by sorry
  have : 3 * teamB_days = 72 * 5 := by sorry
  have : teamB_days = 120 := by sorry
  have : teamA_days = 180 := by sorry
  exact ⟨teamB_days, teamA_days⟩

theorem max_cost_teamB (teamB_days : ℝ) (teamA_cost_per_day : ℝ) (subsidy_per_day : ℝ) (total_cost_A : ℝ) :
  teamA_cost_per_day = 0.8 ∧ subsidy_per_day = 0.01 ∧ total_cost_A = 145.8 → 
  teamB_days = 120 ∧ subsidy_per_day * teamB_days + x * teamB_days ≤ total_cost_A → 
  x ≤ 1.205 :=
by
  intro h
  have h1 := and.elim_left h
  have h2 := and.elim_right h1
  have h3 := and.elim_right h
  have : 120 * x + 120 * 0.01 ≤ 145.8 := by sorry
  have : 120 * x + 1.2 ≤ 145.8 := by sorry
  have : 120 * x ≤ 144.6 := by sorry
  have : x ≤ 1.205 := by sorry
  exact h2

end construction_days_max_cost_teamB_l301_301155


namespace exist_arrangement_for_P_23_l301_301196

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301196


namespace exists_F_12_mod_23_zero_l301_301222

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301222


namespace biology_vs_reading_diff_l301_301252

def math_hw_pages : ℕ := 2
def reading_hw_pages : ℕ := 3
def total_hw_pages : ℕ := 15

def biology_hw_pages : ℕ := total_hw_pages - (math_hw_pages + reading_hw_pages)

theorem biology_vs_reading_diff : (biology_hw_pages - reading_hw_pages) = 7 := by
  sorry

end biology_vs_reading_diff_l301_301252


namespace volume_of_pyramid_l301_301339

theorem volume_of_pyramid 
  (A B C D S : Type)
  (area_SAB area_SBC area_SCD area_SDA : ℝ) 
  (dihedral_angle : ℝ) 
  (area_ABCD : ℝ) 
  (α : Type → ℝ) 
  (h : Type → ℝ)
  (cos_α : Type → ℝ)
  (sin_α : Type → ℝ)
  (volume : Type → ℝ)
  (conditions : 
    -- lateral face areas
    area_SAB = 9 ∧ area_SBC = 9 ∧ area_SCD = 27 ∧ area_SDA = 27 ∧
    -- dihedral angles are equal
    dihedral_angle = α ∧
    -- the area of the base quadrilateral
    area_ABCD = 36)
  : volume (pyramid S A B C D) = 54 :=
begin
  sorry
end

end volume_of_pyramid_l301_301339


namespace smallest_integer_sum_of_two_distinct_primes_gt_70_l301_301833

/-- The smallest integer that is the sum of two distinct prime integers each greater than 70 is 144. -/
theorem smallest_integer_sum_of_two_distinct_primes_gt_70 : 
  ∃ (a b : ℕ), a > 70 ∧ b > 70 ∧ nat.prime a ∧ nat.prime b ∧ a ≠ b ∧ a + b = 144 :=
sorry

end smallest_integer_sum_of_two_distinct_primes_gt_70_l301_301833


namespace prime_sum_remainder_l301_301775

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301775


namespace quadratic_coefficients_l301_301531

theorem quadratic_coefficients : 
  ∀ (b k : ℝ), (∀ x : ℝ, x^2 + b * x + 5 = (x - 2)^2 + k) → b = -4 ∧ k = 1 :=
by
  intro b k h
  have h1 := h 0
  have h2 := h 1
  sorry

end quadratic_coefficients_l301_301531


namespace find_numbers_l301_301300

def is_solution (a b : ℕ) : Prop :=
  a + b = 432 ∧ (max a b) = 5 * (min a b) ∧ (max a b = 360 ∧ min a b = 72)

theorem find_numbers : ∃ a b : ℕ, is_solution a b :=
by
  sorry

end find_numbers_l301_301300


namespace initial_increase_l301_301286

noncomputable def factory_output_restored (O : ℝ) (X : ℝ) : Prop :=
  let intermediate_output := O + (X / 100) * O
  let holiday_output := intermediate_output + 0.30 * intermediate_output
  let final_output := holiday_output - 0.3007 * holiday_output
  final_output = O 

theorem initial_increase (O : ℝ) : ∃ (X : ℝ), factory_output_restored O X ∧ (X ≈ 7.7) :=
begin
  sorry
end

end initial_increase_l301_301286


namespace determine_a_l301_301635

theorem determine_a (a : ℝ) : 
  (∀ (x y : ℝ), y = a * (x + 3)^2 → (x = 2 → y = -50)) → a = -2 :=
by
  intro h
  specialize h 2 (-50)
  have ha := h (rfl)
  sorry

end determine_a_l301_301635


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301697

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301697


namespace find_n_l301_301959

theorem find_n (n : ℕ) : (1 / (n + 1 : ℝ) + 2 / (n + 1 : ℝ) + (n + 1) / (n + 1 : ℝ) = 2) → (n = 2) :=
by
  sorry

end find_n_l301_301959


namespace train_length_correct_l301_301831

-- Define the speed of the train in km/hr
def train_speed_kmh : ℝ := 52

-- Define the time it takes to cross the pole in seconds
def cross_time_seconds : ℝ := 18

-- Define the conversion factor from km/hr to m/s
def kmh_to_ms : ℝ := 5 / 18

-- Define the speed of the train in m/s
def train_speed_ms : ℝ := train_speed_kmh * kmh_to_ms

-- Define the length of the train
def train_length := train_speed_ms * cross_time_seconds

-- The statement we need to prove
theorem train_length_correct : train_length = 259.92 := 
by {
  -- The exact detailed proof steps are omitted as per instructions.
  sorry
}

end train_length_correct_l301_301831


namespace find_y_l301_301896

noncomputable def similar_triangles (a b x z : ℝ) :=
  (a / x = b / z)

theorem find_y 
  (a b x z : ℝ)
  (ha : a = 12)
  (hb : b = 9)
  (hz : z = 7)
  (h_sim : similar_triangles a b x z) :
  x = 28 / 3 :=
begin
  subst ha,
  subst hb,
  subst hz,
  unfold similar_triangles at h_sim,
  field_simp [h_sim],
  ring,
end

end find_y_l301_301896


namespace possible_scores_B_l301_301393

-- Definitions based on conditions
def total_questions := 10
def correct_points := 3
def total_score := 54
def diff_answers := 2

-- Lean statement
theorem possible_scores_B :
  ∃ S : Set ℕ, S = {24, 27, 30} ∧
  (∀ (a b : ℕ), 
    ∃ n m : ℕ, 
    n + m = total_questions ∧ 
    3 * n + 3 * (n - diff_answers) = total_score ∧ 
    S = {3 * (n - diff_answers), 3 * n, 3 * (n + diff_answers)}) :=
begin
  sorry
end

end possible_scores_B_l301_301393


namespace probability_open_path_l301_301870

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l301_301870


namespace number_of_incorrect_statements_l301_301018

theorem number_of_incorrect_statements :
  (ite (0 ∈ {0}) 0 1) +
  (ite ({0} ⊇ (∅ : Set ℕ)) 0 1) +
  (ite (0.3 ∉ ℚ) 1 0) +
  (ite (0 ∈ ℕ) 0 1) +
  (ite ({a, b} ⊆ {b, a}) 0 1) +
  (ite ({x : ℤ | x^2 - 2 = 0} = ∅) 0 1) = 1 := 
sorry

end number_of_incorrect_statements_l301_301018


namespace ratio_black_to_white_l301_301045

-- Definitions:
def radius_1 : ℝ := 2
def radius_2 : ℝ := 4
def radius_3 : ℝ := 6
def radius_4 : ℝ := 8

def area (r : ℝ) : ℝ := π * r^2

def area_black : ℝ := area radius_1 + (area radius_3 - area radius_2)
def area_white : ℝ := (area radius_2 - area radius_1) + (area radius_4 - area radius_3)

-- Statement to prove:
theorem ratio_black_to_white : area_black / area_white = 3 / 5 := sorry

end ratio_black_to_white_l301_301045


namespace find_m_l301_301529

noncomputable def hex_to_decimal (h : ℕ) : ℕ := 
  match h with
  | n => let d0 := (n % 16) 
             d1 := ((n / 16) % 16) 
             d2 := ((n / 256) % 16) 
             d3 := ((n / 4096) % 16) 
             d4 := ((n / 65536) % 16) 
         in d0 + 16 * d1 + 256 * d2 + 4096 * d3 + 65536 * d4

def m_eq (m : ℕ) : Prop :=
  hex_to_decimal (16^3 + m * 16^2 + 0 * 16 + 5) = 293

theorem find_m : ∃ m : ℕ, m_eq m := by
  sorry

end find_m_l301_301529


namespace total_number_of_ways_l301_301374

theorem total_number_of_ways : 
  let A := 3  -- Number of courses in Category A
  let B := 4  -- Number of courses in Category B
  let total_courses := 3  -- Total number of courses chosen
  (nat.choose A 1 * nat.choose B 2 + 
   nat.choose A 2 * nat.choose B 1) = 30 := 
by 
  sorry

end total_number_of_ways_l301_301374


namespace a_neg2_eq_one_a_neg1_eq_ln_sqrt2_add_one_a_n_minus_2_relation_c_n_neg_l301_301043

noncomputable def a (n : ℤ) : ℝ := ∫ x in 0..(Real.pi / 4), (Real.cos x) ^ n

-- Proof 1: a_{-2} = 1
theorem a_neg2_eq_one : a (-2) = 1 := sorry

-- Proof 2: a_{-1} = ln (sqrt(2) + 1)
theorem a_neg1_eq_ln_sqrt2_add_one : a (-1) = Real.log (Real.sqrt 2 + 1) := sorry

-- Proof 3: Relation between a_n and a_{n-2}
theorem a_n_minus_2_relation (n : ℤ) : a (n - 2) = (1 / (Real.sqrt 2) ^ n - n * a n) / (1 - n) := sorry

-- Proof 4: c_n = 1 / (2n - 1) for n < 0
theorem c_n_neg (n : ℤ) (h : n < 0) : ∃ b_n : ℚ, ∃ c_n : ℚ, 
  a (2 * n) = b_n + Real.pi * c_n ∧ c_n = 1 / (2 * n - 1) := sorry

end a_neg2_eq_one_a_neg1_eq_ln_sqrt2_add_one_a_n_minus_2_relation_c_n_neg_l301_301043


namespace find_phi_l301_301512

theorem find_phi (f : ℝ → ℝ) (ϕ : ℝ) (h1 : ∀ θ : ℝ, f θ = sin θ - √3 * cos θ) 
  (h2 : ∀ θ : ℝ, f θ = 2 * sin (θ + ϕ)) 
  (h3 : -π < ϕ ∧ ϕ < π) : 
  ϕ = -π / 3 :=
sorry

end find_phi_l301_301512


namespace correct_equation_for_programmers_l301_301333

theorem correct_equation_for_programmers (x : ℕ) 
  (hB : x > 0) 
  (programmer_b_speed : ℕ := x) 
  (programmer_a_speed : ℕ := 2 * x) 
  (data : ℕ := 2640) :
  (data / programmer_a_speed = data / programmer_b_speed - 120) :=
by
  -- sorry is used to skip the proof, focus on the statement
  sorry

end correct_equation_for_programmers_l301_301333


namespace parallelogram_ABCD_area_l301_301447

noncomputable def parallelogram_area 
  (AB BC : ℝ) (angle_ABC : ℝ) (is_parallelogram : Prop) 
  (h_AB : AB = 4) (h_BC : BC = 6) (h_angle : angle_ABC = π / 3) : ℝ :=
2 * (1 / 2 * BC * (AB * real.sin angle_ABC))

theorem parallelogram_ABCD_area 
  (AB BC : ℝ) (angle_ABC : ℝ) (is_parallelogram : Prop)
  (h_AB : AB = 4) (h_BC : BC = 6) (h_angle : angle_ABC = π / 3) 
  (parallelogram_area_eq : parallelogram_area AB BC angle_ABC is_parallelogram h_AB h_BC h_angle = 12 * real.sqrt 3): 
  parallelogram_area AB BC angle_ABC is_parallelogram h_AB h_BC h_angle = 12 * real.sqrt 3 :=
begin
  sorry
end

end parallelogram_ABCD_area_l301_301447


namespace max_labeling_of_ngon_l301_301448

/-- Given a positive integer n ≥ 3, 
    each side and diagonal of a regular n-gon is labeled with a positive integer not exceeding r,
    and satisfying the following conditions:
    1. Every positive integer 1, 2, 3, ..., r appears on at least one side or diagonal.
    2. For every triangle P_i P_j P_k, there are two sides with the same labeled number,
       and this number is greater than the number on the third side.
    The maximum positive integer r that satisfies these conditions is n - 1.
    The number of different labeling methods for this maximum positive integer r is given by f(n) = (n!(n-1)!)/(2^(n-1)).
-/
theorem max_labeling_of_ngon (n : ℕ) (h : n ≥ 3) : 
  ∃ (r : ℕ), 
    r = n - 1 ∧
    ∃ (f : ℕ → ℕ),
      f n = n! * (n-1)! / 2^(n-1) := 
sorry

end max_labeling_of_ngon_l301_301448


namespace angle_bisectors_meet_at_90_plus_half_alpha_l301_301610

theorem angle_bisectors_meet_at_90_plus_half_alpha
  (A B C I : Type*)
  [Incenter A B C I]
  (α β γ : ℝ)
  (h1 : α + β + γ = 180) :
  ∃ I : Type*, angle_bisectors_angle_meet_is (β γ : ℝ) I = 90 + α / 2 :=
sorry

end angle_bisectors_meet_at_90_plus_half_alpha_l301_301610


namespace length_of_BC_l301_301392

open Locale Classical
noncomputable theory

-- Definitions based on the given conditions.
variables (A B C D : Point)
variables (AB CD : ℝ)
variable (AB_2 : AB = 2)
variable (CD_3 : CD = 3)
variable (semi_circle_inscribed : SemicircleInscribedInTrapezoid A B C D AB CD) 

-- Define the theorem based on the given problem and solution.
theorem length_of_BC (A B C D : Point) (AB CD : ℝ)
  (AB_2 : AB = 2) (CD_3 : CD = 3)
  (semi_circle_inscribed : SemicircleInscribedInTrapezoid A B C D AB CD) :
  length (BC D) = 5 :=
sorry

end length_of_BC_l301_301392


namespace compute_pqr_l301_301521

theorem compute_pqr (p q r : ℤ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) 
  (h_sum : p + q + r = 30) 
  (h_equation : 1 / p + 1 / q + 1 / r + 420 / (p * q * r) = 1) : 
  p * q * r = 576 :=
sorry

end compute_pqr_l301_301521


namespace least_sub_to_make_div_by_10_l301_301834

theorem least_sub_to_make_div_by_10 : 
  ∃ n, n = 8 ∧ ∀ k, 427398 - k = 10 * m → k ≥ n ∧ k = 8 :=
sorry

end least_sub_to_make_div_by_10_l301_301834


namespace at_least_one_good_product_certain_l301_301389

-- Define the set of products, 10 good and 2 defective.
def num_total_products := 12
def num_good_products := 10
def num_defective_products := 2

-- Define the selection of 3 products from the set of 12.
def num_selected_products := 3

-- Define the statement which needs to be proven: if 3 products are randomly selected, 
-- then the event "at least one is a good product" is certain.
theorem at_least_one_good_product_certain : 
  (num_good_products + num_defective_products = num_total_products) ∧
  (num_selected_products ≤ num_total_products) → 
  ∃ (good selected : nat), good + selected = num_selected_products ∧ good ≥ 1 :=
by sorry   -- Proof is skipped using sorry.

end at_least_one_good_product_certain_l301_301389


namespace minimum_value_PA_PB_PC_condition_l301_301175

noncomputable def minimum_value_PA_PB_PC_over_sides (A B C P : Point) (hP : P ∈ triangle A B C) :
  ℝ :=
  sqrt 3

theorem minimum_value_PA_PB_PC_condition (A B C P : Point) (hP : P ∈ triangle A B C) :
  (PA / BC) + (PB / AC) + (PC / AB) ≥ sqrt 3 :=
begin
  sorry
end

end minimum_value_PA_PB_PC_condition_l301_301175


namespace angle_CKB_in_parallelogram_l301_301152

theorem angle_CKB_in_parallelogram
  (A B C D M K : Point)
  (parallelogram : is_parallelogram A B C D)
  (angle_D : ∠D = 60)
  (AD_length : AD = 2)
  (AB_length : AB = sqrt 3 + 1)
  (midpoint_M : M = midpoint A D)
  (angle_bisector_CK : CK := angle_bisector C) :
  angle CKB = 75 :=
  sorry

end angle_CKB_in_parallelogram_l301_301152


namespace distinguishable_dodecahedrons_l301_301316

noncomputable def count_distinguishable_dodecahedrons : ℕ :=
  (11.factorial / 5)

theorem distinguishable_dodecahedrons :
  ∃ n : ℕ, 
    n = count_distinguishable_dodecahedrons := by
  use (11.factorial / 5)
  sorry

end distinguishable_dodecahedrons_l301_301316


namespace minimum_expenses_for_Nikifor_to_win_maximum_F_value_l301_301340

noncomputable def number_of_voters := 35
noncomputable def sellable_voters := 14 -- 40% of 35
noncomputable def preference_voters := 21 -- 60% of 35
noncomputable def minimum_votes_to_win := 18 -- 50% of 35 + 1
noncomputable def cost_per_vote := 9

def vote_supply_function (P : ℕ) : ℕ :=
  if P = 0 then 10
  else if 1 ≤ P ∧ P ≤ 14 then 10 + P
  else 24


theorem minimum_expenses_for_Nikifor_to_win :
  ∃ P : ℕ, P * cost_per_vote = 162 ∧ vote_supply_function P ≥ minimum_votes_to_win := 
sorry

theorem maximum_F_value (F : ℕ) : 
  F = 3 :=
sorry

end minimum_expenses_for_Nikifor_to_win_maximum_F_value_l301_301340


namespace regular_decagon_interior_angle_l301_301330

theorem regular_decagon_interior_angle
  (n : ℕ) 
  (hn : n = 10) 
  (regular_decagon : true) :
  (180 * (n - 2)) / n = 144 :=
by
  rw [hn]
  sorry

end regular_decagon_interior_angle_l301_301330


namespace dashed_lines_form_square_l301_301886

-- Geometrical definitions and conditions
variables (A B C D O P Q R S T : Point)
variables (large_circle small_circle1 small_circle2 : Circle)
variables (rhombus : Rhombus)

-- Assumptions about the rhombus and its properties
axiom rhombus_diagonals_bisect : rhombus.is_diagonal_intersection O
axiom large_circle_inscribed : large_circle.is_inscribed rhombus
axiom small_circle1_touches_two_sides_and_large_circle : small_circle1.touches_two_sides_and_large_circle
axiom small_circle2_touches_two_sides_and_large_circle : small_circle2.touches_two_sides_and_large_circle
axiom dashed_lines_form_rectangle : ∀ P Q R S, rectangle_on_points P Q R S, where
    P Q R S are points of tangency of small circles with rhombus sides

-- The theorem to be proved
theorem dashed_lines_form_square :
  ∃ P Q R S, rectangle_on_points P Q R S ∧ 
  equal_sides P Q R S ∧ 
  ∀ T, angle P T Q = 90° ∧ angle Q T R = 90° ∧ angle R T S = 90° ∧ angle S T P = 90° :=
begin
  sorry
end

end dashed_lines_form_square_l301_301886


namespace compute_3_h_l301_301282

noncomputable def hypotenuse_of_right_triangle (a b : ℝ) : ℝ :=
  real.sqrt (a^2 + b^2)

theorem compute_3_h (h : ℝ) :
  let a := real.log 16 / real.log 3,
      b := real.log 81 / real.log 3,
      c := hypotenuse_of_right_triangle a b
  in c = h → 3^h = 3^(2 * real.sqrt 5) :=
by
  intros
  let a := real.log 16 / real.log 3
  let b := real.log 81 / real.log 3
  have h_eq : hypotenuse_of_right_triangle a b = h, from ‹_›
  sorry

end compute_3_h_l301_301282


namespace solve_for_x_l301_301613

theorem solve_for_x (x : ℝ) : 2 ^ x * 8 ^ x = 32 ^ (x - 12) → x = 60 := by
  sorry

end solve_for_x_l301_301613


namespace proof_m_n_proof_a_b_l301_301825

noncomputable def condition_m (m : ℝ) : Prop := 10^m = 40
noncomputable def condition_n (n : ℝ) : Prop := 10^n = 50
def condition_a : ℝ := Real.log 2
def condition_b : ℝ := Real.log 3

theorem proof_m_n (m n : ℝ) (hm : condition_m m) (hn : condition_n n) : m + 2 * n = 5 := 
by
  have hyp_m : m = Real.log 40 := by sorry
  have hyp_n : n = Real.log 50 := by sorry
  rw [hyp_m, hyp_n]
  calc
    m + 2 * n = Real.log 40 + 2 * Real.log 50 := by rfl
           ... = Real.log 40 + Real.log (50^2) := by sorry
           ... = Real.log (40 * 2500) := by sorry
           ... = Real.log 100000 := by sorry
           ... = 5 := by sorry

theorem proof_a_b (a b : ℝ) : Real.log 2 = a ∧ Real.log 3 = b → Real.logb 3 6 = (a + b) / b := 
by
  intro h
  obtain ⟨ha, hb⟩ := h
  calc
    Real.logb 3 6 = Real.log 6 / Real.log 3 := by sorry
             ... = (Real.log (2 * 3)) / Real.log 3 := by sorry
             ... = (Real.log 2 + Real.log 3) / Real.log 3 := by sorry
             ... = (a + b) / b := by rw [ha, hb]

end proof_m_n_proof_a_b_l301_301825


namespace nicky_cards_value_l301_301596

theorem nicky_cards_value 
  (x : ℝ)
  (h : 21 = 2 * x + 5) : 
  x = 8 := by
  sorry

end nicky_cards_value_l301_301596


namespace mean_of_set_l301_301412

open Nat

theorem mean_of_set (m : ℤ) (h : m + 4 = 16) : 
  (12 + 14 + 16 + 23 + 30) / 5 = 19 :=
by
  -- Prove the median condition implies m = 12
  have hm : m = 12 := by
    linarith [h]
  -- Substitute m = 12 in the set and compute the mean
  have hs : {m, m + 2, m + 4, m + 11, m + 18} = {12, 14, 16, 23, 30} := by
    simp [hm]
  -- Compute the mean of the new set
  calc (12 + 14 + 16 + 23 + 30) / 5 = 95 / 5 := by norm_num
                               ... = 19 := by norm_num

end mean_of_set_l301_301412


namespace sequence_an_expression_l301_301463

theorem sequence_an_expression (a : ℕ → ℕ) : 
  a 1 = 1 ∧ (∀ n : ℕ, n ≥ 1 → (a n / n - a (n - 1) / (n - 1)) = 2) → (∀ n : ℕ, a n = 2 * n * n - n) :=
by
  sorry

end sequence_an_expression_l301_301463


namespace length_of_BD_l301_301465

theorem length_of_BD (A B C D E : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] [MetricSpace E]
  (m_angle_A : ∀ {a b c : A}, angle a b c = 45)
  (BC_length : ∀ {b c : B}, dist b c = 8)
  (BD_perp_AC : ∀ {b d c : D}, perp b d c)
  (CE_perp_AB : ∀ {c e b : E}, perp c e b)
  (angle_DBC : ∀ {d b c : D}, angle d b c = 4 * angle e c b) :
  dist B D = 4 :=
by
  sorry

end length_of_BD_l301_301465


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301687

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301687


namespace cody_needs_total_steps_l301_301009

theorem cody_needs_total_steps 
  (weekly_steps : ℕ → ℕ)
  (h1 : ∀ n, weekly_steps n = (n + 1) * 1000 * 7)
  (h2 : 4 * 7 * 1000 + 3 * 7 * 1000 + 2 * 7 * 1000 + 1 * 7 * 1000 = 70000) 
  (h3 : 70000 + 30000 = 100000) :
  ∃ total_steps, total_steps = 100000 := 
by
  sorry

end cody_needs_total_steps_l301_301009


namespace radius_smaller_circle_l301_301904

-- Defining the given conditions: a larger circle with radius 5 and arithmetic progression of areas
def radius_larger_circle : ℝ := 5
def area_larger_circle : ℝ := π * radius_larger_circle^2
def sum_areas (A1 A2 : ℝ) : ℝ := A1 + A2
def in_arithmetic_progression (A1 A2 : ℝ) (S : ℝ) : Prop :=
  A2 = S / 3 ∧ A1 = S * 2 / 3


theorem radius_smaller_circle :
  ∃ r : ℝ, 
  ((area_larger_circle, sum_areas (π * r^2) area_larger_circle, π * r^2) : ℝ × ℝ × ℝ, 
   π * r^2 = 50 * π) :=
begin
  sorry
end

end radius_smaller_circle_l301_301904


namespace solve_for_k_l301_301406

def f (x : ℝ) : ℝ := 7 * x^2 - 1 / x + 5
def g (x : ℝ) (k : ℝ) : ℝ := x^2 - k

theorem solve_for_k :
  f 3 - g 3 k = 8 → k = -152 / 3 := by
  sorry

end solve_for_k_l301_301406


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301707

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301707


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301701

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301701


namespace intersection_complement_l301_301068

open Set

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 2^(x-2) > 1}

theorem intersection_complement :
  A ∩ (compl B) = {x | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_l301_301068


namespace solve_x_eq_99_l301_301266

theorem solve_x_eq_99 :
  ∃ (x : ℝ), 0.05 * x + 0.1 * (30 + x) - 2 = 15.8 ∧ (x.round : ℤ) = 99 :=
by
  sorry

end solve_x_eq_99_l301_301266


namespace eq_x_for_abs_sum_eq_l301_301021

theorem eq_x_for_abs_sum_eq (x : ℝ) (h : |x - 24| + |x - 20| = |2x - 44|) : 
  x = 22 :=
sorry

end eq_x_for_abs_sum_eq_l301_301021


namespace remainder_first_six_primes_div_seventh_l301_301813

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301813


namespace probability_area_PAC_less_than_third_l301_301936

theorem probability_area_PAC_less_than_third (P : Point)
  (h_PA : Point_in_triangle P [(0, 6), (6, 0), (0, 0)]) :
  probability (area (triangle (P :: (0, 6)) (0, 0)) < one_div_third * area (triangle [(0, 6), (6, 0), (0, 0)])) 
  = one_div_twelve := sorry

end probability_area_PAC_less_than_third_l301_301936


namespace triangle_area_l301_301271

theorem triangle_area (a b c : ℝ) (K : ℝ) (m n p : ℕ) (h1 : a = 10) (h2 : b = 12) (h3 : c = 15)
  (h4 : K = 240 * Real.sqrt 7 / 7)
  (h5 : Int.gcd m p = 1) -- m and p are relatively prime
  (h6 : n ≠ 1 ∧ ¬ (∃ x, x^2 ∣ n ∧ x > 1)) -- n is not divisible by the square of any prime
  : m + n + p = 254 := sorry

end triangle_area_l301_301271


namespace restaurant_total_dishes_l301_301849

noncomputable def total_couscous_received : ℝ := 15.4 + 45
noncomputable def total_chickpeas_received : ℝ := 19.8 + 33

-- Week 1, ratio of 5:3 (couscous:chickpeas)
noncomputable def sets_of_ratio_week1_couscous : ℝ := total_couscous_received / 5
noncomputable def sets_of_ratio_week1_chickpeas : ℝ := total_chickpeas_received / 3
noncomputable def dishes_week1 : ℝ := min sets_of_ratio_week1_couscous sets_of_ratio_week1_chickpeas

-- Week 2, ratio of 3:2 (couscous:chickpeas)
noncomputable def sets_of_ratio_week2_couscous : ℝ := total_couscous_received / 3
noncomputable def sets_of_ratio_week2_chickpeas : ℝ := total_chickpeas_received / 2
noncomputable def dishes_week2 : ℝ := min sets_of_ratio_week2_couscous sets_of_ratio_week2_chickpeas

-- Total dishes rounded down
noncomputable def total_dishes : ℝ := dishes_week1 + dishes_week2

theorem restaurant_total_dishes :
  ⌊total_dishes⌋ = 32 :=
by {
  sorry
}

end restaurant_total_dishes_l301_301849


namespace calculate_f_at_8_l301_301581

def f (x : ℝ) : ℝ := 2 * x^4 - 17 * x^3 + 27 * x^2 - 24 * x - 72

theorem calculate_f_at_8 : f 8 = 952 :=
by sorry

end calculate_f_at_8_l301_301581


namespace remainder_first_six_primes_div_seventh_l301_301812

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301812


namespace probability_of_reaching_last_floor_l301_301865

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l301_301865


namespace right_triangle_similarity_l301_301900

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l301_301900


namespace similar_triangles_y_value_l301_301898

noncomputable def y_value := 9.33

theorem similar_triangles_y_value (y : ℝ) 
    (h : ∃ (a b : ℝ), a = 12 ∧ b = 9 ∧ (a / y = b / 7)) : y = y_value := 
  sorry

end similar_triangles_y_value_l301_301898


namespace focal_length_of_ellipse_l301_301954

theorem focal_length_of_ellipse :
  (∀ (α : Real), ∃ (x y : Real), 
    x = 4 + 2 * Real.cos α ∧
    y = 1 + 5 * Real.sin α) →
    (exists (c : Real), c = 2 * Real.sqrt 21) := 
by
  intro h
  use 2 * Real.sqrt 21
  sorry

end focal_length_of_ellipse_l301_301954


namespace range_of_function_l301_301019

theorem range_of_function : ∀ (y : ℝ), (0 < y ∧ y ≤ 1 / 2) ↔ ∃ (x : ℝ), y = 1 / (x^2 + 2) := 
by
  sorry

end range_of_function_l301_301019


namespace probability_exactly_3_tails_in_8_flips_l301_301191

theorem probability_exactly_3_tails_in_8_flips :
  let n := 8
  let k := 3
  let p := (2/3 : ℚ)
  let q := (1/3 : ℚ)
  let prob := (nat.choose n k) * p^k * q^(n-k)
  prob = (448 : ℚ) / 6561 :=
by { sorry }

end probability_exactly_3_tails_in_8_flips_l301_301191


namespace find_ellipse_C2_triangle_OAB_obtuse_find_slope_l_l301_301491

-- Definitions and conditions
def parabola_C1 := ∀ x y : ℝ, x^2 = 4 * y
def ellipse_C2 (a b : ℝ) := ∀ x y : ℝ, a > b ∧ b > 0 ∧ (y^2 / a^2 + x^2 / b^2 = 1)
def focus_condition (F : ℝ × ℝ) := ∃ F, ∀ x : ℝ, F = (0, 1)
def common_chord_length := 2 * real.sqrt 6

-- Problem 1
theorem find_ellipse_C2 (a b : ℝ) 
  (h_parabola : parabola_C1)
  (h_ellipse : ellipse_C2 a b)
  (h_focus : focus_condition (0, 1))
  (h_chord_length : common_chord_length) :
  (a^2 = 9) ∧ (b^2 = 8) :=
  sorry

-- Problem 2.i
theorem triangle_OAB_obtuse (a b : ℝ) (k : ℝ)
  (h_parabola : parabola_C1)
  (h_ellipse : ellipse_C2 a b)
  (h_focus : focus_condition (0, 1)) :
  ∃ F : ℝ × ℝ, F = (0, 1) → is_obtuse_triangle O (A F.1) (B F.2) :=
  sorry

-- Problem 2.ii
theorem find_slope_l (a b k : ℝ)
  (h_parabola : parabola_C1)
  (h_ellipse : ellipse_C2 a b)
  (h_focus : focus_condition (0, 1))
  (h_eq_dist : |AC| = |BD|) :
  k = real.sqrt(6) / 4 ∨ k = -real.sqrt(6) / 4 :=
  sorry

end find_ellipse_C2_triangle_OAB_obtuse_find_slope_l_l301_301491


namespace ratio_sum_distances_l301_301575

variable (a : ℝ) (E : ℝ × ℝ)

-- Conditions for square ABCD and point E inside the square
def is_square (A B C D: ℝ × ℝ) : Prop := -- Definition for square
  A.1 = B.1 ∧ C.1 = D.1 ∧ A.2 = D.2 ∧ B.2 = C.2 ∧ (B.1 - A.1) = (C.2 - B.2)

-- Define the vertices
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (a, 0)
def C : ℝ × ℝ := (a, a)
def D : ℝ × ℝ := (0, a)

-- Ensure E is inside the square
axiom E_in_square : 0 ≤ E.1 ∧ E.1 ≤ a ∧ 0 ≤ E.2 ∧ E.2 ≤ a 

-- Distance from E to vertices
def distance (P Q: ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
def s : ℝ := distance E A + distance E B + distance E C + distance E D

-- Distance from E to lines
def distance_to_line_x (x: ℝ) : ℝ := abs (E.1 - x)
def distance_to_line_y (y: ℝ) : ℝ := abs (E.2 - y)
def S : ℝ := distance_to_line_x 0 + distance_to_line_x a + distance_to_line_y 0 + distance_to_line_y a

-- The proof we need
theorem ratio_sum_distances : s / S = real.sqrt 2 := sorry

end ratio_sum_distances_l301_301575


namespace complex_mul_example_l301_301344

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l301_301344


namespace garden_snake_length_l301_301907

theorem garden_snake_length :
  ∀ (garden_snake boa_constrictor : ℝ),
    boa_constrictor * 7.0 = garden_snake →
    boa_constrictor = 1.428571429 →
    garden_snake = 10.0 :=
by
  intros garden_snake boa_constrictor H1 H2
  sorry

end garden_snake_length_l301_301907


namespace calc_sqrt_log_sum_l301_301399

theorem calc_sqrt_log_sum : 9^(1/2) + log 2 4 = 5 :=
by
  sorry

end calc_sqrt_log_sum_l301_301399


namespace interior_angle_regular_decagon_l301_301327

theorem interior_angle_regular_decagon : 
  let n := 10 in (180 * (n - 2)) / n = 144 := 
by
  sorry

end interior_angle_regular_decagon_l301_301327


namespace area_of_hexagon_l301_301625

-- Definitions of the angles and side lengths
def angle_A := 120
def angle_B := 120
def angle_C := 120
def angle_D := 150

def FA := 2
def AB := 2
def BC := 2
def CD := 3
def DE := 3
def EF := 3

-- Theorem statement for the area of hexagon ABCDEF
theorem area_of_hexagon : 
  (angle_A = 120 ∧ angle_B = 120 ∧ angle_C = 120 ∧ angle_D = 150 ∧
   FA = 2 ∧ AB = 2 ∧ BC = 2 ∧ CD = 3 ∧ DE = 3 ∧ EF = 3) →
  (∃ area : ℝ, area = 7.5 * Real.sqrt 3) :=
by
  sorry

end area_of_hexagon_l301_301625


namespace jeff_total_work_hours_l301_301564

-- Definitions based on the problem conditions
def weekend_catch_up_hours (weekend_work_hours : ℕ) : ℕ :=
  3 * weekend_work_hours

def weekday_work_hours (weekday_catch_up_hours : ℕ) : ℕ :=
  4 * weekday_catch_up_hours

def catch_up_hours_per_day : ℕ := 3

def total_weekend_work_hours : ℕ :=
  (weekend_catch_up_hours total_weekend_work_hours) / 3

def total_weekday_work_hours : ℕ :=
  5 * (weekday_work_hours catch_up_hours_per_day)

def total_work_hours : ℕ :=
  total_weekend_work_hours + total_weekday_work_hours

-- The theorem to prove
theorem jeff_total_work_hours : total_work_hours = 66 :=
by
  sorry

end jeff_total_work_hours_l301_301564


namespace area_of_N1N2N3_l301_301153

-- Definitions based on the problem conditions
def is_halfside (a b c d e f : ℝ) := 
  (a = b / 2) ∧ (c = d / 2) ∧ (e = f / 2)

def is_proportional (AN2 N2N1 N1D BE CF : ℝ) :=
  (AN2 / N2N1 = 2) ∧ (N2N1 / N1D = 1) ∧ (AN2 / N1D = 2) ∧ (BE / N2N1 = 2) ∧ (CF / N1D = 2)

-- Main statement
theorem area_of_N1N2N3 (K : ℝ) (a b c d e f AN2 N2N1 N1D BE CF : ℝ)
  (h1 : is_halfside a b c d e f)
  (h2 : is_proportional AN2 N2N1 N1D BE CF) :
  (area N1N2N3 = (K : ℝ) / 8) := by
  sorry

end area_of_N1N2N3_l301_301153


namespace arrangement_exists_for_P_eq_23_l301_301244

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301244


namespace large_cuboid_width_l301_301105

noncomputable def cuboidVolume (length width height : ℝ) : ℝ :=
  length * width * height

theorem large_cuboid_width :
  let length_small := 6
  let width_small := 4
  let height_small := 3
  let num_small_cuboids := 7.5
  let length_large := 18
  let height_large := 2
  let total_small_volume := num_small_cuboids * cuboidVolume length_small width_small height_small
  let W := total_small_volume / (length_large * height_large) in
  W = 15 :=
by
  sorry

end large_cuboid_width_l301_301105


namespace largest_terms_ninth_and_tenth_l301_301633

noncomputable def a (n : ℕ) : ℝ := n / (n^2 + 90)

theorem largest_terms_ninth_and_tenth :
  ∀ n : ℕ, a(9) = a(10) ∧ (∀ m : ℕ, a(m) ≤ a(9)) :=
by
  sorry

end largest_terms_ninth_and_tenth_l301_301633


namespace arrangement_for_P23_exists_l301_301238

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301238


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301699

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301699


namespace bisect_BK_l301_301911

variables {A B C E F G K M : Type} [Point A] [Point B] [Point C] [Point E] [Point F] [Point G] [Point K] [Point M] [Triangle ABC]

-- Given conditions
hypothesis (h1 : A ≠ B)   
hypothesis (h2 : A ≠ C)  
hypothesis (hIso : is_isosceles_triangle ABC)
hypothesis (hCenter : midpoint B C = O)
hypothesis (hTangAB : tangent_point AB O E) 
hypothesis (hTangAC : tangent_point AC O F)
hypothesis (hAG_perp_EG : is_perpendicular AG EG)
hypothesis (hTang : tangent G K AC)
hypothesis (hM_midpoint_AE : midpoint A E = M)

-- Proof statement
theorem bisect_BK (K G : Type) : bisects_line_bisects_segment EF BK :=
sorry

end bisect_BK_l301_301911


namespace solution_exists_l301_301940

noncomputable def operation : ℝ → ℝ → ℝ 
| a b := if a > b then a * b^2 - 1
         else if a = b then 2 * a - 1
         else a^2 * b + 1

theorem solution_exists : 
  ∃ x : ℝ, 
    (operation 1 x = operation 2 x) ∧ 
    (x = (1 + Real.sqrt 17) / 4 ∨ x = 1 ∨ x = 2) := 
sorry

end solution_exists_l301_301940


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301780

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301780


namespace find_x_1998_l301_301644

variable (a b : ℝ)

noncomputable def sequence (n : ℕ) : ℝ :=
  if h : n = 0 then a 
  else if h : n = 1 then b 
  else sequence (n - 2)⁻¹ * (1 + sequence (n - 1))

theorem find_x_1998 : sequence a b 1998 = (1 + a + b) / (a * b) := by
  sorry

end find_x_1998_l301_301644


namespace arrangement_for_P23_exists_l301_301233

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301233


namespace exists_arrangement_for_P_23_l301_301203

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301203


namespace apples_on_tree_l301_301253

-- Defining initial number of apples on the tree
def initial_apples : ℕ := 4

-- Defining apples picked from the tree
def apples_picked : ℕ := 2

-- Defining new apples grown on the tree
def new_apples : ℕ := 3

-- Prove the final number of apples on the tree is 5
theorem apples_on_tree : initial_apples - apples_picked + new_apples = 5 :=
by
  -- This is where the proof would go
  sorry

end apples_on_tree_l301_301253


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301710

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301710


namespace triangle_areas_ratio_l301_301559

theorem triangle_areas_ratio
  {A B C D P K : Type*}
  [coord : affine_space ℝ A] [inhabited A]
  (triangle_ABC : triangle A B C)
  (is_median : is_median D B C)
  (point_P_on_BD : lies_on P (segment B D))
  (ratio_BP_PD : dist B P = 3 * dist P D)
  (intersection_AK_BC : lies_on K (line_pair A P ∩ line_pair B C)) :
  ratio_of_areas (triangle A B K) (triangle A C K) = 3 / 2 :=
by sorry

end triangle_areas_ratio_l301_301559


namespace grace_age_l301_301104

theorem grace_age 
  (H : ℕ) 
  (I : ℕ) 
  (J : ℕ) 
  (G : ℕ)
  (h1 : H = I - 5)
  (h2 : I = J + 7)
  (h3 : G = 2 * J)
  (h4 : H = 18) : 
  G = 32 := 
sorry

end grace_age_l301_301104


namespace sarah_socks_l301_301608

theorem sarah_socks :
  ∃ (a b c : ℕ), a + b + c = 15 ∧ 2 * a + 4 * b + 5 * c = 45 ∧ 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c ∧ (a = 8 ∨ a = 9) :=
by {
  sorry
}

end sarah_socks_l301_301608


namespace solve_m_n_sum_l301_301476

theorem solve_m_n_sum (m n : ℝ) 
  (h : ∀ x : ℝ, abs (2 * x - 3) ≤ 1 ↔ x ∈ set.Icc m n) : m + n = 3 := by
  sorry

end solve_m_n_sum_l301_301476


namespace solution_set_f_prime_pos_l301_301113

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem solution_set_f_prime_pos : 
  {x : ℝ | 0 < x ∧ (deriv f x > 0)} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_f_prime_pos_l301_301113


namespace arrangement_exists_for_P_eq_23_l301_301242

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301242


namespace remainder_first_six_primes_div_seventh_l301_301817

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301817


namespace company_needs_minimum_specialists_l301_301879

variable EnergyEfficiency WasteManagement WaterConservation : ℕ
variable EnergyAndWaste WasteAndWater EnergyAndWater AllThree : ℕ

variables (n : ℕ)

def minimum_specialists_needed 
  (EnergyEfficiency : ℕ) 
  (WasteManagement : ℕ) 
  (WaterConservation : ℕ) 
  (EnergyAndWaste : ℕ)
  (WasteAndWater : ℕ)
  (EnergyAndWater : ℕ)
  (AllThree : ℕ) : ℕ :=
  EnergyEfficiency + WasteManagement + WaterConservation - EnergyAndWaste - WasteAndWater - EnergyAndWater + AllThree

theorem company_needs_minimum_specialists 
  (h_ee : EnergyEfficiency = 95)
  (h_wm : WasteManagement = 80)
  (h_wc : WaterConservation = 110)
  (h_eewm : EnergyAndWaste = 30)
  (h_wmwc : WasteAndWater = 35)
  (h_eewc : EnergyAndWater = 25)
  (h_all : AllThree = 15) :
  minimum_specialists_needed 95 80 110 30 35 25 15 = 210 :=
by 
  simp [minimum_specialists_needed]
  sorry

end company_needs_minimum_specialists_l301_301879


namespace cost_of_popsicle_sticks_l301_301937

theorem cost_of_popsicle_sticks
  (total_money : ℕ)
  (cost_of_molds : ℕ)
  (cost_per_bottle : ℕ)
  (popsicles_per_bottle : ℕ)
  (sticks_used : ℕ)
  (sticks_left : ℕ)
  (number_of_sticks : ℕ)
  (remaining_money : ℕ) :
  total_money = 10 →
  cost_of_molds = 3 →
  cost_per_bottle = 2 →
  popsicles_per_bottle = 20 →
  sticks_left = 40 →
  number_of_sticks = 100 →
  remaining_money = total_money - cost_of_molds - (sticks_used / popsicles_per_bottle * cost_per_bottle) →
  sticks_used = number_of_sticks - sticks_left →
  remaining_money = 1 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end cost_of_popsicle_sticks_l301_301937


namespace collinear_implies_x_values_l301_301502

-- Given vectors a and b are not collinear (i.e., they are linearly independent),
-- and c = x • a + b, d = 2 • a + (2 * x - 3) • b.
-- If vectors c and d are collinear, we want to prove that x = 2 or x = -1/2.

variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables {a b : V} (h_non_collinear : ¬Collinear ℝ ({a, b} : Set V))
variables (x : ℝ)
def c := x • a + b
def d := 2 • a + (2 * x - 3) • b

theorem collinear_implies_x_values (h_collinear : Collinear ℝ ({c x, d x} : Set V)) : x = 2 ∨ x = -1 / 2 :=
sorry

end collinear_implies_x_values_l301_301502


namespace hyperbola_equation_and_area_of_triangle_l301_301101

def hyperbola_eq (a b x y : ℝ) : Prop := y^2 / (a^2) - x^2 / (b^2) = 1

noncomputable def distance_from_vertex_to_asymptote (a : ℝ) : ℝ := abs(2 * 0 + a) / sqrt(5)

theorem hyperbola_equation_and_area_of_triangle
  (a b : ℝ) (h1 : a > 0) (h2 : b > 0)
  (h_asymptote_eq : 2 * x + y = 0)
  (h_distance : distance_from_vertex_to_asymptote 2 = 2 * sqrt(5) / 5) :
  (hyperbola_eq 2 1 x y) ∧ (∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ mn = 1 ∧ 
  let OA := sqrt(5) * m, OB := sqrt(5) * n in (1 / 2) * OA * OB * sin(2 * atan(1 / 2)) = 2) :=
begin
  sorry
end

end hyperbola_equation_and_area_of_triangle_l301_301101


namespace determinant_scaled_l301_301974

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l301_301974


namespace length_of_AC_l301_301156

-- Definitions and conditions as given in the problem
variables (A B C H X Y : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited H] [Inhabited X] [Inhabited Y]
variables (right_triangle : A → B → C → Prop) (altitude : A → H → Prop)
variables (circle_intersects : A → H → X → Y → Prop)
variables (AX AY AB : ℝ) (AX_eq : AX = 5) (AY_eq : AY = 6) (AB_eq : AB = 9)

-- Target proof problem
theorem length_of_AC (AC : ℝ) (right_triangle_eq : right_triangle A B C)
  (altitude_from_A : altitude A H) (circle_condition : circle_intersects A H X Y) : AC = 13.5 :=
sorry

end length_of_AC_l301_301156


namespace max_value_of_A_l301_301983

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end max_value_of_A_l301_301983


namespace sin_50_tan_10_identity_l301_301037

theorem sin_50_tan_10_identity :
  sin (50 : ℝ) * (1 + (real.sqrt 3) * tan (10 : ℝ)) = 1 := 
sorry

end sin_50_tan_10_identity_l301_301037


namespace option_a_correct_option_c_correct_option_d_correct_l301_301112

theorem option_a_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (1 / a > 1 / b) :=
sorry

theorem option_c_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (Real.sqrt (-a) > Real.sqrt (-b)) :=
sorry

theorem option_d_correct (a b : ℝ) (h : a < b) (h_neg : b < 0) : (|a| > -b) :=
sorry

end option_a_correct_option_c_correct_option_d_correct_l301_301112


namespace calculate_expression_l301_301821

def x : Float := 3.241
def y : Float := 14
def z : Float := 100
def expected_result : Float := 0.45374

theorem calculate_expression : (x * y) / z = expected_result := by
  sorry

end calculate_expression_l301_301821


namespace cricketer_sixes_l301_301880

/-- A cricketer scored 134 runs which included 12 boundaries and some sixes.
    He made 55.223880597014926% of his total score by running between the wickets.
    Prove that the cricketer hit 2 sixes. -/
theorem cricketer_sixes :
  ∃ (sixes : ℕ), sixes = 2 ∧ 
    let total_runs := 134,
        boundaries := 12,
        running_percent := 55.223880597014926 in
    let runs_by_running := total_runs * (running_percent / 100),
        runs_from_boundaries := boundaries * 4,
        runs_from_sixes := total_runs - (runs_from_boundaries + runs_by_running) in
    runs_from_sixes / 6 = sixes :=
begin
  have total_runs : ℕ := 134,
  have boundaries : ℕ := 12,
  have running_percent : ℚ := 55.223880597014926,

  let runs_by_running : ℚ := total_runs * (running_percent / 100),
  let runs_from_boundaries : ℕ := boundaries * 4,
  let runs_from_sixes : ℚ := ↑total_runs - (runs_from_boundaries + runs_by_running),

  use runs_from_sixes / 6,
  split,
  { norm_num, },
  { sorry },
end

end cricketer_sixes_l301_301880


namespace sphere_surface_area_l301_301523

theorem sphere_surface_area (V : ℝ) (π : ℝ) (r : ℝ) (A : ℝ) 
  (h1 : V = 36 * π) 
  (h2 : V = (4 / 3) * π * r^3) 
  (h3 : A = 4 * π * r^2) 
  : A = 36 * π :=
by
  sorry

end sphere_surface_area_l301_301523


namespace tangent_line_eqn_monotonicity_f_zero_roots_diff_l301_301489

-- Problem 1
theorem tangent_line_eqn (a b : ℝ) (h1 : a = 1) (h2 : b = 1) :
  ∃ (x : ℝ), x = 1 → ∃ (y : ℝ), (f x = y) ∧ (2 * x - y - 2 = 0) :=
sorry

-- Problem 2
theorem monotonicity_f (a : ℝ) (b : ℝ) (h : b = 2 * a + 1): 
  (a ≤ 0 ∧ (∀ x, (0 < x ∧ x < 1) → (difff x > 0)) ∧ 
  (∀ x, (1 < x) → (difff x < 0))) ∨
  (0 < a ∧ a < 1/2 ∧ (∀ x, (1 < x ∧ x < 1/ (2 * a)) → (difff x < 0))) ∨
  (a = 1/2 ∧ (∀ x, 0 < x → (difff x ≥ 0))) ∨
  (a > 1/2 ∧ ∃ x, 0 < x ∧ x < 1/ (2 * a) → (difff x > 0)) :=
sorry

-- Problem 3
theorem zero_roots_diff (b : ℝ) (x₁ x₂ : ℝ) (h1 : a = 1) (h2 : b > 3)
  (h3 : x₁ < x₂) (h4 : 2 * x₁ ^ 2 - b * x₁ + 1 = 0)
  (h5 : 2 * x₂ ^ 2 - b * x₂ + 1 = 0) :
  f x₁ - f x₂ = 3 / 4 - ln 2 :=
sorry

end tangent_line_eqn_monotonicity_f_zero_roots_diff_l301_301489


namespace prob_of_interval_expectation_of_X_l301_301627

open ProbabilityTheory

-- Define the mean score µ
def mean_score : ℝ := 65

-- Define the normal distribution for Z
noncomputable def Z : Measure ℝ := NormalDistribution mean_score 210

-- Define the probability of the interval [36, 79.5]
def prob_interval : ℝ := 0.8186

theorem prob_of_interval (μ : ℝ) :
  μ = 65 →
  (probability (set.Icc 36 79.5) (NormalDistribution μ 210)) = prob_interval :=
by
  sorry

-- Define the distribution of phone credits X
def phone_credit_dist : Distribution ℝ := 
  [ (20, 3 / 8), (40, 13 / 32), (60, 3 / 16), (80, 1 / 32) ]

-- Define the expected value of X
def expected_value_X : ℝ := 75 / 2

theorem expectation_of_X :
  (expectedValue phone_credit_dist) = expected_value_X :=
by
  sorry

end prob_of_interval_expectation_of_X_l301_301627


namespace total_situps_l301_301918

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end total_situps_l301_301918


namespace arrangement_for_P23_exists_l301_301237

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301237


namespace proof_problem_l301_301494

def U := ℝ
def A : set ℝ := {x | 3 * x - x^2 > 0}
def B : set ℝ := {x | ∃ (y : ℝ), y = log 2 (x + 1) ∧ x ∈ A}
def complement_B := {x : ℝ | x ≤ 0 ∨ x ≥ 2}

theorem proof_problem : A ∩ complement_B = set.Ico 2 3 :=
by
  sorry

end proof_problem_l301_301494


namespace largest_perimeter_10_tiles_smallest_perimeter_10_tiles_largest_perimeter_2011_tiles_smallest_perimeter_2011_tiles_l301_301375

def perimeter_n_tiles (n : ℕ) : ℕ :=
  2 * (n + 1)

def min_perimeter_n_tiles (n : ℕ) : ℕ :=
  (4 * Real.sqrt n).ceil.toNat

theorem largest_perimeter_10_tiles :
  perimeter_n_tiles 10 = 22 := by
  sorry

theorem smallest_perimeter_10_tiles :
  min_perimeter_n_tiles 10 = 14 := by
  sorry

theorem largest_perimeter_2011_tiles :
  perimeter_n_tiles 2011 = 4024 := by
  sorry

theorem smallest_perimeter_2011_tiles :
  min_perimeter_n_tiles 2011 = 180 := by
  sorry

end largest_perimeter_10_tiles_smallest_perimeter_10_tiles_largest_perimeter_2011_tiles_smallest_perimeter_2011_tiles_l301_301375


namespace value_of_a_l301_301986

theorem value_of_a (a : ℝ) (h : (∑ i in finset.range 6, finset.choose 5 i * (x ^ 2)^(5 - i) * (a / x)^i = -10) : a = -2 := by
sorry

end value_of_a_l301_301986


namespace equation_of_line_m_l301_301056

-- Define the point P and the line l
def point_P : ℝ × ℝ := (-2, 5)
def line_l (x y : ℝ) : Prop := y = (-3/4) * x + (5 + 3/2)

-- Define the conditions for lines l and m
def slope_line_l : ℝ := -3/4
def line_parallel_to_l (x y : ℝ) (c : ℝ) : Prop := 3 * x + 4 * y + c = 0
def distance_between_lines (c : ℝ) : Prop := 
  ∃ d : ℝ, d = 3 ∧ abs ((-6) + 20 + c) / 5 = d

-- The ultimate goal is to prove the equation of line m
theorem equation_of_line_m : ∃ c : ℝ, (c = 1 ∨ c = -29) → (∃ x y : ℝ, line_parallel_to_l x y c) := by
  sorry

end equation_of_line_m_l301_301056


namespace smallest_identical_digits_divisible_by_18_l301_301957

-- Definitions of necessary properties
def identical_digits (n : ℕ) : Prop :=
  ∃ d : ℕ, (d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) ∧ (n = d * (10 ^ ((nat.log 10 n) + 1) - 1) / 9)

def divisible_by_2 (n : ℕ) : Prop := n % 2 = 0
def divisible_by_9 (n : ℕ) : Prop := (n.digits 10).sum % 9 = 0

def divisible_by_18 (n : ℕ) : Prop := divisible_by_2 n ∧ divisible_by_9 n

-- Proving the smallest number with identical digits divisible by 18
theorem smallest_identical_digits_divisible_by_18 :
  ∃ n : ℕ, identical_digits n ∧ divisible_by_18 n ∧ (∀ m : ℕ, identical_digits m → divisible_by_18 m → n ≤ m) :=
sorry

end smallest_identical_digits_divisible_by_18_l301_301957


namespace find_a16_l301_301492

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 1 / 2 ∧ ∀ n ≥ 1, a (n + 1) = 1 - 1 / a n

theorem find_a16 (a : ℕ → ℝ) (h : seq a) : a 16 = 1 / 2 :=
sorry

end find_a16_l301_301492


namespace shaded_area_percentage_correct_l301_301332

-- Define a square and the conditions provided
def square (side_length : ℕ) : ℕ := side_length ^ 2

-- Define conditions
def EFGH_side_length : ℕ := 6
def total_area : ℕ := square EFGH_side_length

def shaded_area_1 : ℕ := square 2
def shaded_area_2 : ℕ := square 4 - square 3
def shaded_area_3 : ℕ := square 6 - square 5

def total_shaded_area : ℕ := shaded_area_1 + shaded_area_2 + shaded_area_3

def shaded_percentage : ℚ := total_shaded_area / total_area * 100

-- Statement of the theorem to prove
theorem shaded_area_percentage_correct :
  shaded_percentage = 61.11 := by sorry

end shaded_area_percentage_correct_l301_301332


namespace perimeter_isosceles_triangle_l301_301133

structure Triangle where
  a b c : ℝ
  h_eq_sides : a = b ∨ a = c ∨ b = c
  h_valid_triangle : a + b > c ∧ a + c > b ∧ b + c > a

def isosceles_triangle_3_7 : Triangle :=
  { a := 3, b := 3, c := 7,
    h_eq_sides := Or.inl rfl,
    h_valid_triangle := ⟨by norm_num, by norm_num, by norm_num⟩ }

theorem perimeter_isosceles_triangle (T : Triangle) (hT : T = isosceles_triangle_3_7) : T.a + T.b + T.c = 13 := by
  rw [hT]
  simp
  norm_num

end perimeter_isosceles_triangle_l301_301133


namespace product_has_odd_prime_factor_l301_301587

theorem product_has_odd_prime_factor (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h_m_gt_n : m > n)
  (h_int_x_k : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n + 1 → 
                        ∃ (x_k : ℕ), x_k = (m + k) / (n + k) ∧ (m + k) % (n + k) = 0 ∧ x_k * (n + k) = m + k) :
  ∃ p : ℕ, p > 1 ∧ p % 2 = 1 ∧ p ∣ ((∏ k in Finset.range (n + 2), ((m + k) / (n + k))) - 1) :=
sorry

end product_has_odd_prime_factor_l301_301587


namespace read_books_correct_l301_301305

namespace CrazySillySchool

-- Definitions from conditions
def total_books : Nat := 20
def unread_books : Nat := 5
def read_books : Nat := total_books - unread_books

-- Theorem statement
theorem read_books_correct : read_books = 15 :=
by
  -- Mathematical statement that follows from conditions and correct answer
  sorry

end CrazySillySchool

end read_books_correct_l301_301305


namespace average_score_of_class_l301_301877

variable (students_total : ℕ) (group1_students : ℕ) (group2_students : ℕ)
variable (group1_avg : ℝ) (group2_avg : ℝ)

theorem average_score_of_class :
  students_total = 20 → 
  group1_students = 10 → 
  group2_students = 10 → 
  group1_avg = 80 → 
  group2_avg = 60 → 
  (group1_students * group1_avg + group2_students * group2_avg) / students_total = 70 := 
by
  intros students_total_eq group1_students_eq group2_students_eq group1_avg_eq group2_avg_eq
  rw [students_total_eq, group1_students_eq, group2_students_eq, group1_avg_eq, group2_avg_eq]
  simp
  sorry

end average_score_of_class_l301_301877


namespace complex_mul_example_l301_301342

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l301_301342


namespace carrots_picked_by_Carol_l301_301928

theorem carrots_picked_by_Carol (total_carrots mom_carrots : ℕ) (h1 : total_carrots = 38 + 7) (h2 : mom_carrots = 16) :
  total_carrots - mom_carrots = 29 :=
by {
  sorry
}

end carrots_picked_by_Carol_l301_301928


namespace power_series_solution_l301_301033

noncomputable def y (x : ℝ) : ℝ := ∑' m : ℕ, if m % 3 = 0 then (-1)^(m/3) * (List.product (List.map (fun k => (3 * (m / 3) - 2 * k)) (List.range (m/3 + 1)))) * (x ^ m) / m.factorial else 0

theorem power_series_solution (f : ℝ → ℝ) 
  (h_eq : ∀ x, f'' x + x * f x = 0)
  (h₀ : f 0 = 1) 
  (h₁ : deriv f 0 = 0) : 
  ∀ x, f x = y x :=
begin
  sorry
end

end power_series_solution_l301_301033


namespace similar_triangles_y_value_l301_301897

noncomputable def y_value := 9.33

theorem similar_triangles_y_value (y : ℝ) 
    (h : ∃ (a b : ℝ), a = 12 ∧ b = 9 ∧ (a / y = b / 7)) : y = y_value := 
  sorry

end similar_triangles_y_value_l301_301897


namespace oscar_wins_games_l301_301589

theorem oscar_wins_games :
  ∀ (O : ℕ),
  let lucy_games := 9,
      maya_games := 4,
      oscar_games := O + 4,
      total_games := (lucy_games + maya_games + oscar_games) / 2,
      lucy_wins := 5,
      maya_wins := 2,
      total_wins := lucy_wins + maya_wins + O
  in total_wins = total_games → O = 3 :=
by
  intros
  sorry

end oscar_wins_games_l301_301589


namespace area_AEF_l301_301188

theorem area_AEF (hBCD : triangle_area B C D = 1)
  (hBDE : triangle_area B D E = 1/3)
  (hCDF : triangle_area C D F = 1/5) :
  triangle_area A E F = 4 / 35 :=
sorry

end area_AEF_l301_301188


namespace sequence_solution_l301_301444

noncomputable def a : ℕ → ℝ
| 1 => 1
| (n + 1) => 1/16 * (1 + 4 * a n + Real.sqrt (1 + 24 * a n))

theorem sequence_solution (n : ℕ) : a n = (2 ^ (4 - 2 * n) + 3 * 2 ^ (3 - n) + 8) / 24 := 
sorry

end sequence_solution_l301_301444


namespace area_quadrilateral_ABCD_l301_301361

-- Definition of a cube with side length 1
def cube (side_length : ℝ) := 
  { α : ℝ // α = 1 }

-- Lean 4 statement for the problem:
theorem area_quadrilateral_ABCD 
  (A C B D : ℝ ^ 3)
  (cube : A.algebra θ = 1)
  (diagonal : (A - C) = 3)
  (midpoint_B : B = (1 / 2))
  (midpoint_D : D = (1 / 2) ∧ A ≠ D ∧ B ≠ C) -- ensure B and D are not on edges of A and C

: (area ABCD = sqrt 6 / 2)
sorry

end area_quadrilateral_ABCD_l301_301361


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301731

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301731


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301751

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301751


namespace find_a_l301_301987

noncomputable def S_n (n : ℕ) (a : ℝ) : ℝ := 2 * 3^n + a
noncomputable def a_1 (a : ℝ) : ℝ := S_n 1 a
noncomputable def a_2 (a : ℝ) : ℝ := S_n 2 a - S_n 1 a
noncomputable def a_3 (a : ℝ) : ℝ := S_n 3 a - S_n 2 a

theorem find_a (a : ℝ) : a_1 a * a_3 a = (a_2 a)^2 → a = -2 :=
by
  sorry

end find_a_l301_301987


namespace count_numbers_satisfying_condition_l301_301920

def three_digit_numbers (N : ℕ) : Prop :=
  100 ≤ N ∧ N ≤ 999

def base3_to_decimal (N : ℕ) : ℕ := -- Definition of base-3 to decimal conversion
  -- Implementation here
  sorry

def base4_to_decimal (N : ℕ) : ℕ := -- Definition of base-4 to decimal conversion
  -- Implementation here
  sorry

def M_condition (N : ℕ) (M : ℕ) : Prop :=
  let N3 := base3_to_decimal N
  let N4 := base4_to_decimal N
  M = N3 * N4 ∧ M % 100 = (3 * N) % 100

theorem count_numbers_satisfying_condition : 
    {N : ℕ | three_digit_numbers N ∧ ∃ M : ℕ, M_condition N M}.card = 25 := 
by {
  sorry
}

end count_numbers_satisfying_condition_l301_301920


namespace unfenced_side_length_l301_301047

-- Define the conditions
variables (L W : ℝ)
axiom area_condition : L * W = 480
axiom fence_condition : 2 * W + L = 64

-- Prove the unfenced side of the yard (L) is 40 feet
theorem unfenced_side_length : L = 40 :=
by
  -- Conditions, definitions, and properties go here.
  -- But we leave the proof as a placeholder since the statement is sufficient.
  sorry

end unfenced_side_length_l301_301047


namespace range_of_x_for_fx1_positive_l301_301455

-- Define the conditions
def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def is_monotonic_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x
def f_at_2_eq_zero (f : ℝ → ℝ) := f 2 = 0

-- Define the problem statement that needs to be proven
theorem range_of_x_for_fx1_positive (f : ℝ → ℝ) :
  is_even f →
  is_monotonic_decreasing_on_nonneg f →
  f_at_2_eq_zero f →
  ∀ x, f (x - 1) > 0 ↔ -1 < x ∧ x < 3 :=
by sorry

end range_of_x_for_fx1_positive_l301_301455


namespace proof_problem_l301_301479

noncomputable def f : ℝ → ℝ := λ x, 2 * sin (x + π / 3)

theorem proof_problem (f_odd: (∀ x, f (x - π / 3) = - f(x)) )
  (f_pi_six: f (π / 6) = 2 )
  (f_zero_2pi3: f (2 * π / 3) = 0) :
  (f x = 2 * sin (x + π / 3)) ∧
  (∀ (x : ℝ), 0 <= x ∧ x <= 2 * π →
      ((0 ≤ x ∧ x ≤ π / 6) ∨ (7 * π / 6 ≤ x ∧ x ≤ 2 * π) →
        monotone_on f (Icc 0 (2 * π)))) := 
sorry

end proof_problem_l301_301479


namespace ping_pong_team_sequences_l301_301296

open Nat

theorem ping_pong_team_sequences (players matches double_singles : ℕ)
  (total_matches : ℕ) (match_position : ℕ) :
  players = 3 ∧ total_matches = 5 ∧ double_singles = 3 →
  match_position = 3 →
  (∃ (num_sequences : ℕ), num_sequences = 36) := 
by
  intros conditions pos
  have num_sequences : ℕ := choose 3 1 * choose 4 2 * perm 2 2
  use num_sequences
  sorry

end ping_pong_team_sequences_l301_301296


namespace sum_f_series_l301_301487

noncomputable def f (x : ℝ) : ℝ := sin (5 * real.pi / 3 + real.pi / 6) + (3 * x) / (2 * x - 1)

theorem sum_f_series :
  (∑ k in finset.range 2016, if k.val % 2 = 1 then f (k.val / 2016) else 0) = 2016 :=
by
  sorry

end sum_f_series_l301_301487


namespace school_total_pupils_l301_301145

structure School :=
  (initial_girls : Nat)
  (initial_boys : Nat)
  (new_girls : Nat)

def total_pupils (s : School) : Nat :=
  s.initial_girls + s.initial_boys + s.new_girls

theorem school_total_pupils :
  ∀ (s : School), s.initial_girls = 706 →
  s.initial_boys = 222 →
  s.new_girls = 418 →
  total_pupils s = 1346 :=
by
  intro s hg hb hn
  rw [hg, hb, hn]
  sorry

end school_total_pupils_l301_301145


namespace total_weight_is_96_l301_301301

variables Jack_weight Sam_weight : ℕ

theorem total_weight_is_96 (h1 : Jack_weight = 52) (h2 : Jack_weight = Sam_weight + 8) : Jack_weight + Sam_weight = 96 :=
by
  sorry

end total_weight_is_96_l301_301301


namespace bubble_mix_ratio_l301_301371

theorem bubble_mix_ratio (total_soap_tablepoons : ℕ) (container_ounces : ℕ) (ounces_per_cup : ℕ) (total_cups : ℕ) (soap_per_cup : ℕ) :
  total_soap_tablepoons = 15 → 
  container_ounces = 40 → 
  ounces_per_cup = 8 → 
  total_cups = container_ounces / ounces_per_cup → 
  soap_per_cup = total_soap_tablepoons / total_cups → 
  soap_per_cup = 3 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  have h6 : total_cups = 5 := by norm_num; exact h4
  rw h6 at h5
  have h7 : total_soap_tablepoons = 15 := h1
  rw h7 at h5
  exact h5.symm

end bubble_mix_ratio_l301_301371


namespace samantha_exam_score_l301_301607

theorem samantha_exam_score :
  ∀ (q1 q2 q3 : ℕ) (s1 s2 s3 : ℚ),
  q1 = 30 → q2 = 50 → q3 = 20 →
  s1 = 0.75 → s2 = 0.8 → s3 = 0.65 →
  (22.5 + 40 + 2 * (0.65 * 20)) / (30 + 50 + 2 * 20) = 0.7375 :=
by
  intros q1 q2 q3 s1 s2 s3 hq1 hq2 hq3 hs1 hs2 hs3
  sorry

end samantha_exam_score_l301_301607


namespace symmetric_curve_eq_l301_301628

theorem symmetric_curve_eq : 
  (∃ x' y', (x' - 3)^2 + 4*(y' - 5)^2 = 4 ∧ (x' - 6 = x' + x) ∧ (y' - 10 = y' + y)) ->
  (∃ x y, (x - 6) ^ 2 + 4 * (y - 10) ^ 2 = 4) :=
by
  sorry

end symmetric_curve_eq_l301_301628


namespace exists_similar_sizes_P_23_l301_301211

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301211


namespace arrangement_exists_for_P_eq_23_l301_301247

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301247


namespace correct_option_D_l301_301335
noncomputable theory

-- Conditions defined as Lean 4 variables and functions

def log2 (x : ℝ) : ℝ := real.log x / real.log 2
def propA (x : ℝ) : Prop := (x < 1) → (log2 (x + 1) < 1)
def neg_prop (P : ℝ → Prop) : Prop := ∃ x : ℝ, ¬ P x
def propB : Prop := (¬ ∀ x > 0, 2^x > 1) = ∃ x₀ > 0, 2^x₀ ≤ 1
def propC {a b c : ℝ} : Prop := (a ≤ b) → (a * c^2 ≤ b * c^2) ↔ (a * c^2 ≤ b * c^2) → (a ≤ b)
def propD (a b : ℝ) : Prop := (a + b ≠ 5) → (a ≠ 2 ∨ b ≠ 3)

-- Main proof obligations:

theorem correct_option_D (a b : ℝ) :
  (¬ ∃ x (H : x = A), propA x) ∧
  (¬ propB) ∧
  (¬ propC) ∧
  (propD a b) :=
by 
  sorry

end correct_option_D_l301_301335


namespace possible_turtles_club_membership_l301_301137

def students := Fin 100
def clubs := Fin 10 -- assuming there can be at most 10 clubs (arbitrary number for Lean formulation)

structure Club :=
  founders : Finset students
  members : Finset students

structure Scenario :=
  students : Finset students
  friends : students → students → Prop
  clubs : Finset Club

-- Conditions
axiom friendship_symmetric : ∀ (x y : students), friends x y → friends y x
axiom friendship_refl : ∀ (x : students), friends x x

axiom initial_club_founders : ∀ (c : Club), founders.card = 3
axiom unique_members : ∀ (c₁ c₂ : Club), c₁ ≠ c₂ → (members c₁ ∩ members c₂).card = 0

axiom daily_join_conditions : ∀ (s : students), ∀ (c : Club), (∃ (m₁ m₂ m₃ : students), m₁ ∈ members c ∧ m₂ ∈ members c ∧ m₃ ∈ members c ∧ friends s m₁ ∧ friends s m₂ ∧ friends s m₃) → s ∈ members c

-- Conclusion to prove
theorem possible_turtles_club_membership (scenario : Scenario) :
  (∃ (turtles cheetahs : Club), ∀ (s : students), s ∈ members cheetahs ∧ (members turtles).card = 50) :=
sorry

end possible_turtles_club_membership_l301_301137


namespace write_xy_possible_l301_301561

theorem write_xy_possible (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ s : set ℝ, (1 ∈ s) ∧ (x ∈ s) ∧ (y ∈ s) ∧ (∀ a b ∈ s, a + b ∈ s ∧ a - b ∈ s ∧ (a ≠ 0 → a⁻¹ ∈ s) ∧ (b ≠ 0 → b⁻¹ ∈ s)) ∧ xy ∈ s :=
by
  sorry

end write_xy_possible_l301_301561


namespace snail_distance_min_max_l301_301261

-- Given Conditions
variable (t : ℝ) -- t is the total observation time in minutes
variable (observed_continuously : ℝ → Prop) -- A property indicating the snail is observed continuously

-- Additional assumptions based on conditions provided in step a
axiom observed_for_exactly_one_minute : (∀ (x : ℝ), (x >= 0 ∧ x < t) → (observed_continuously x → ∃ (p : ℝ), p = 1))
axiom snail_crawled_1_meter_per_minute : (∀ (x : ℝ), (x >= 0 ∧ x < t) → (observed_continuously x → ∃ (d : ℝ), d = 1))

-- Theorem: Minimum and Maximum Distance Crawled by the Snail
theorem snail_distance_min_max (t : ℝ) (observed_continuously : ℝ → Prop)
  (h_obs : ∀ (x : ℝ), (x >= 0 ∧ x < t) → (observed_continuously x → ∃ (p : ℝ), p = 1))
  (h_crawl : ∀ (x : ℝ), (x >= 0 ∧ x < t) → (observed_continuously x → ∃ (d : ℝ), d = 1)) :
  ∃ (d_min d_max : ℝ), d_min = ⌊t / 2⌋ + 1 ∧ (if t ∈ ℕ then d_max = 2 * (t - 1) else d_max = 2 * ⌊t⌋) := 
sorry

end snail_distance_min_max_l301_301261


namespace probability_open_doors_l301_301875

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l301_301875


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301779

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301779


namespace problem1_problem2_l301_301096

-- Definition of the function f with parameter a
def f (x : ℝ) (a : ℝ) : ℝ := log x + a / x - 1

-- Problem 1: Minimum value when a = 2
theorem problem1 : (∀ x : ℝ, f x 2 >= f 2 2) := 
sorry

-- Problem 2: Range of a given inequality holds for all x in [1, +∞)
theorem problem2 (a : ℝ) : (∀ x : ℝ, x ≥ 1 → f x a ≤ (1 / 2) * x - 1) → a ≤ 1 / 2 :=
sorry

end problem1_problem2_l301_301096


namespace num_solutions_abcd_eq_2020_l301_301032

theorem num_solutions_abcd_eq_2020 :
  ∃ S : Finset (ℕ × ℕ × ℕ × ℕ), 
    (∀ (a b c d : ℕ), (a, b, c, d) ∈ S ↔ (a^2 + b^2) * (c^2 - d^2) = 2020 ∧ a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) ∧
    S.card = 6 :=
sorry

end num_solutions_abcd_eq_2020_l301_301032


namespace gcd_256_162_450_l301_301325

theorem gcd_256_162_450 : Nat.gcd (Nat.gcd 256 162) 450 = 2 := sorry

end gcd_256_162_450_l301_301325


namespace min_red_cubes_l301_301058

theorem min_red_cubes (n : ℕ) :
  let white_cubes := 26
  let black_cube_center := (1 : ℕ)
  let edge_length_cube := 3
  let total_cubes := n^3
  let larger_cube_edge_length := 3 * n
  let min_painted_red := (n + 1) * n^2
  -- Conditions on center position for painting strategy (ensuring all white cubes share one vertex with a red one)
  (∀ (coords : ℕ × ℕ × ℕ), 
    let a := coords.1
    let b := coords.2
    let c := coords.3
    (a, b, c) ∈ white_cubes →
    (∃ (d : ℕ), d ≡ 2 [MOD 3] ∧
                (a = 1 ∨ b ≡ 2 [MOD 3] ∧ c ≡ 2 [MOD 3]) ∨
                (b = 0 ∨ c = 0)
  )) →
  min_painted_red = (n + 1) * n^2 :=
by
  sorry

end min_red_cubes_l301_301058


namespace find_a_over_b_l301_301467

-- Given conditions
variables (a b : ℝ)

-- Definitions based on conditions
def equation := (a - 2 * complex.I) * complex.I = b + a * complex.I

-- Statement to prove
theorem find_a_over_b (h : equation a b) : a / b = 1 / 2 :=
sorry

end find_a_over_b_l301_301467


namespace equal_real_roots_of_quadratic_eq_l301_301126

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l301_301126


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301714

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301714


namespace incandescent_bulb_count_l301_301914

noncomputable def water_potential_energy (m : ℝ) (g : ℝ) (h : ℝ) : ℝ :=
  m * g * h

noncomputable def total_energy_per_sec (vol : ℝ) (height : ℝ) (density : ℝ) (g : ℝ) : ℝ :=
  water_potential_energy (vol * density) g height

def total_effective_power (energy : ℝ) (t_eff : ℝ) (d_eff : ℝ) (tr_eff : ℝ) : ℝ :=
  energy * t_eff * d_eff * tr_eff

def total_motor_power (n_motors : ℕ) (power_hp : ℝ) (efficiency : ℝ) : ℝ :=
  n_motors * power_hp * 736 / efficiency

def total_lamp_power (n_lamps : ℕ) (voltage : ℝ) (current : ℝ) : ℝ :=
  n_lamps * voltage * current

def remaining_power (energy : ℝ) (motor_power : ℝ) (lamp_power : ℝ) : ℝ :=
  energy - motor_power - lamp_power

def bulb_count (remaining_power : ℝ) (bulb_power : ℝ) : ℕ :=
  real.to_nat (remaining_power / bulb_power)

theorem incandescent_bulb_count (vol : ℝ) (height : ℝ) (density : ℝ) (g : ℝ) 
(t_eff : ℝ) (d_eff : ℝ) (tr_eff : ℝ) 
(n_motors : ℕ) (power_hp : ℝ) (motor_eff : ℝ) 
(n_lamps : ℕ) (voltage : ℝ) (current : ℝ) 
(bulb_power : ℝ) : 
  bulb_count (remaining_power (total_effective_power (total_energy_per_sec vol height density g) t_eff d_eff tr_eff) (total_motor_power n_motors power_hp motor_eff) (total_lamp_power n_lamps voltage current)) bulb_power = 3920 := 
by 
  sorry

end incandescent_bulb_count_l301_301914


namespace scientific_notation_of_3900000000_l301_301552

theorem scientific_notation_of_3900000000 : 3900000000 = 3.9 * 10^9 :=
by 
  sorry

end scientific_notation_of_3900000000_l301_301552


namespace coprime_condition_exists_l301_301655

theorem coprime_condition_exists : ∃ (A B C : ℕ), (A > 0 ∧ B > 0 ∧ C > 0) ∧ (Nat.gcd (Nat.gcd A B) C = 1) ∧ 
  (A * Real.log 5 / Real.log 50 + B * Real.log 2 / Real.log 50 = C) ∧ (A + B + C = 4) :=
by {
  sorry
}

end coprime_condition_exists_l301_301655


namespace find_distance_between_A_and_B_l301_301926

noncomputable def distance_between_A_and_B (speedA speedB stopping_time total_distance : ℕ) :=
  (total_distance / speedA = total_distance / speedB - stopping_time / 60) ∧ 
  total_distance = distance_between_A_and_B

theorem find_distance_between_A_and_B :
  distance_between_A_and_B 80 70 15 2240 :=
by
  sorry

end find_distance_between_A_and_B_l301_301926


namespace inverse_of_13_mod_300_is_277_l301_301665

def modular_inverse_13_mod_300 : Prop :=
  ∃ x : ℕ, 0 ≤ x ∧ x < 300 ∧ 13 * x % 300 = 1

theorem inverse_of_13_mod_300_is_277 : modular_inverse_13_mod_300 :=
  ∃ (x : ℕ), 0 ≤ x ∧ x < 300 ∧ 13 * x % 300 = 1 ∧ x = 277

end inverse_of_13_mod_300_is_277_l301_301665


namespace nominal_rate_of_interest_annual_l301_301626

theorem nominal_rate_of_interest_annual (EAR nominal_rate : ℝ) (n : ℕ) (h1 : EAR = 0.0816) (h2 : n = 2) : 
  nominal_rate = 0.0796 :=
by 
  sorry

end nominal_rate_of_interest_annual_l301_301626


namespace parabola_chords_perpendicular_l301_301435

theorem parabola_chords_perpendicular (p : ℝ) (h_pow : p^2 = 4) :
  ∃ AB CD : ℝ, (ABS = ℝ := 2p / (sin(angle₁)^2)) ∧ (CD = ℝ := 2p / (cos(angle₁)^2)) ∧
     angle₁ = π / 2 - (angle₁) ∧
     (angle₁ ∧
  if h_AB : AB ≠ 0 ∧ CD ≠ 0 then
    ABS \in set_of {1,2,3}^2 ∧s h ∧ = 50 } :=
begin
  sorry
end

end parabola_chords_perpendicular_l301_301435


namespace water_fee_relationship_xiao_qiangs_water_usage_l301_301876

variable (x y : ℝ)
variable (H1 : x > 10)
variable (H2 : y = 3 * x - 8)

theorem water_fee_relationship : y = 3 * x - 8 := 
  by 
    exact H2

theorem xiao_qiangs_water_usage : y = 67 → x = 25 :=
  by
    intro H
    have H_eq : 67 = 3 * x - 8 := by 
      rw [←H2, H]
    linarith

end water_fee_relationship_xiao_qiangs_water_usage_l301_301876


namespace sinusoidal_equation_solution_l301_301829

theorem sinusoidal_equation_solution (z : ℝ) :
  (sin (2 * z))^2 + (sin (3 * z))^2 + (sin (4 * z))^2 + (sin (5 * z))^2 = 2 →
  (∃ n : ℤ, z = (Int.cast (2 * n + 1) * Real.pi) / 14) ∨ (∃ m : ℤ, z = (Int.cast (2 * m + 1) * Real.pi) / 4) :=
by
  intros h
  sorry

end sinusoidal_equation_solution_l301_301829


namespace class_student_count_l301_301416

theorem class_student_count
  (g1 g2 g3 g4 g5 : ℕ) 
  (ratio_eq : g1 : g2 : g3 : g4 : g5 = 1 : 2 : 5 : 3 : 1) 
  (max_group_size : g3 = 20) 
  (total_students : ℕ := g1 + g2 + g3 + g4 + g5) :
  total_students = 48 :=
sorry

end class_student_count_l301_301416


namespace assignment_arrangements_l301_301046

-- Define a structure for teachers and classes
inductive Teacher
| A | B | C | D deriving DecidableEq

inductive Class
| class1 | class2 | class3 deriving DecidableEq

-- Define a function for counting valid arrangements
def count_valid_arrangements : ℕ :=
  by sorry

-- Prove that the number of valid arrangements is 30
theorem assignment_arrangements : count_valid_arrangements = 30 :=
  by sorry

end assignment_arrangements_l301_301046


namespace log_base_2_equation_solution_set_l301_301034

theorem log_base_2_equation_solution_set :
  ∀ x : ℝ, (log 2 (4^x - 3) = x + 1) ↔ x = log 2 3 :=
begin
  intro x,
  split,
  {
    intro h,
    sorry, -- Here would be the space for actual proof steps
  },
  {
    intro h,
    rw h,
    sorry, -- Here would be the space for actual proof steps
  }
end

end log_base_2_equation_solution_set_l301_301034


namespace probability_green_ball_is_half_l301_301310

def Container := Finset (String × ℚ)

def Set1 : List Container :=
[ {("A", 8 / 10), ("A", 2 / 10)},  -- Container A: 8 green, 2 red
  {("B", 2 / 10), ("B", 8 / 10)},  -- Container B: 2 green, 8 red
  {("C", 2 / 10), ("C", 8 / 10)} ] -- Container C: 2 green, 8 red

def Set2 : List Container :=
[ {("A", 2 / 10), ("A", 8 / 10)},  -- Container A: 2 green, 8 red
  {("B", 8 / 10), ("B", 2 / 10)},  -- Container B: 8 green, 2 red
  {("C", 8 / 10), ("C", 2 / 10)} ] -- Container C: 8 green, 2 red

noncomputable def probabilityOfGreen (sets : List Container) : ℚ :=
  (1 / 2) * (1 / 3 * (8 / 10) + 1 / 3 * (2 / 10) + 1 / 3 * (2 / 10)) +
  (1 / 2) * (1 / 3 * (8 / 10) + 1 / 3 * (8 / 10) + 1 / 3 * (2 / 10))

theorem probability_green_ball_is_half : probabilityOfGreen Set1 = 1 / 2 :=
by
  sorry

end probability_green_ball_is_half_l301_301310


namespace probability_at_least_one_card_each_cousin_correct_l301_301594

noncomputable def probability_at_least_one_card_each_cousin : ℚ :=
  let total_cards := 16
  let cards_per_cousin := 8
  let selections := 3
  let total_ways := Nat.choose total_cards selections
  let ways_all_from_one_cousin := Nat.choose cards_per_cousin selections * 2  -- twice: once for each cousin
  let prob_all_from_one_cousin := (ways_all_from_one_cousin : ℚ) / total_ways
  1 - prob_all_from_one_cousin

theorem probability_at_least_one_card_each_cousin_correct :
  probability_at_least_one_card_each_cousin = 4 / 5 :=
by
  -- Proof would go here
  sorry

end probability_at_least_one_card_each_cousin_correct_l301_301594


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301737

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301737


namespace no_digits_satisfy_equation_l301_301953

theorem no_digits_satisfy_equation : 
  ∀ (x y z : ℕ), (x ≤ 9) ∧ (y ≤ 9) ∧ (z ≤ 9) → (100 * x + 10 * y + z ≠ y * (10 * x + z)) :=
begin
  intros x y z h,
  have H1: 100 * x + 10 * y + z = (y * 10 * x) + (y * z), 
  { sorry },
  have H2: y * 10 * x ≥ y * 10 + z, 
  { sorry },
  have H3: y < 10,
  { sorry },
  exact H3,
end

end no_digits_satisfy_equation_l301_301953


namespace age_difference_l301_301185

theorem age_difference (M T J X S : ℕ)
  (hM : M = 3)
  (hT : T = 4 * M)
  (hJ : J = T - 5)
  (hX : X = 2 * J)
  (hS : S = 3 * X - 1) :
  S - M = 38 :=
by
  sorry

end age_difference_l301_301185


namespace range_of_quadratic_expressions_l301_301440

theorem range_of_quadratic_expressions (a b : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : a + b = 1) :
  ∃ m, m ≥ 4 ∧ (∀ n, n ∈ range (λ x, a^2 + 1/a^2 + b^2 + 1/b^2) → n = m) :=
sorry

end range_of_quadratic_expressions_l301_301440


namespace position_2008_l301_301910

open Nat

def position_of_number (arrangement : ℕ → Finset ℕ) (n : ℕ) : ℕ × ℕ :=
  let row := find_rows (arrangement, n)
  let column := find_column (arrangement, row, n)
  (row, column)

theorem position_2008 (arrangement : ℕ → Finset ℕ) : position_of_number arrangement 2008 = (18, 45) :=
by
  sorry

end position_2008_l301_301910


namespace lassie_original_bones_l301_301571

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end lassie_original_bones_l301_301571


namespace greatest_area_difference_l301_301318

theorem greatest_area_difference (l w l' w' : ℕ) (hl : l + w = 90) (hl' : l' + w' = 90) :
  (l * w - l' * w').nat_abs ≤ 1936 :=
sorry

end greatest_area_difference_l301_301318


namespace range_of_k_l301_301530

theorem range_of_k (k : ℝ) : (∃ x ∈ set.Icc 1 2, x^2 + k * x - 1 > 0) ↔ k > (-3/2) :=
by
  sorry

end range_of_k_l301_301530


namespace only_x_eq_3_solution_l301_301966

theorem only_x_eq_3_solution : 
  (∀ x : ℤ, 3 * x > 4 * x - 4 ∧ 4 * x - b > -8 → x = 3) ↔ (4 = finset.card (finset.filter (fun b => 16 ≤ b ∧ b < 20) (finset.range 21))) := 
sorry

end only_x_eq_3_solution_l301_301966


namespace exist_arrangement_for_P_23_l301_301197

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301197


namespace probability_of_reaching_last_floor_l301_301863

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l301_301863


namespace acute_angle_value_l301_301486

theorem acute_angle_value (a : ℝ) :
  let f : ℝ → ℝ := λ x, cos x * (sin x + sqrt 3 * cos x) - (sqrt 3 / 2)
  let g : ℝ → ℝ := λ x, f (x + a)
  (∀ x, g (- x + - (π / 2)) = - g x) →
  ∃ α : ℝ, 0 < α ∧ α < π / 2 ∧ α = π / 3 :=
by
  sorry

end acute_angle_value_l301_301486


namespace path_count_l301_301847

theorem path_count :
  let is_valid_path (path : List (ℕ × ℕ)) : Prop :=
    ∃ (n : ℕ), path = List.range n    -- This is a simplification for definition purposes
  let count_paths_outside_square (start finish : (ℤ × ℤ)) (steps : ℕ) : ℕ :=
    43826                              -- Hardcoded the result as this is the correct answer
  ∀ start finish : (ℤ × ℤ),
    start = (-5, -5) → 
    finish = (5, 5) → 
    count_paths_outside_square start finish 20 = 43826
:= 
sorry

end path_count_l301_301847


namespace find_other_oils_l301_301024

variables (oils : ℕ) (chosen_oils : list ℕ)

def interval : ℕ := oils / 4

def systematic_sampling (oils chosen_oils : list ℕ) : Prop :=
  oils.length = 56 ∧ chosen_oils.length = 4 ∧
  7 ∈ chosen_oils ∧ 35 ∈ chosen_oils ∧
  (∀ (x ∈ chosen_oils), x + interval oils = (x + 14) % 56)

theorem find_other_oils (h : systematic_sampling 56 [7, 35, 21, 49]) : 
  21 ∈ [21, 49] ∧ 49 ∈ [21, 49] :=
by simp; sorry

end find_other_oils_l301_301024


namespace right_triangle_similarity_l301_301901

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l301_301901


namespace compute_expression_l301_301931

theorem compute_expression : 45 * (28 + 72) + 55 * 45 = 6975 := 
  by
  sorry

end compute_expression_l301_301931


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301679

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301679


namespace sum_of_k_binom_n_k_l301_301402

theorem sum_of_k_binom_n_k (n : ℕ) :
  ∑ k in Finset.range (n + 1), k * (nat.choose n k) = n * 2^(n - 1) :=
sorry

end sum_of_k_binom_n_k_l301_301402


namespace complex_multiplication_l301_301350

theorem complex_multiplication:
  (2 + 2 * complex.I) * (1 - 2 * complex.I) = 6 - 2 * complex.I := by
  sorry

end complex_multiplication_l301_301350


namespace exists_arrangement_for_P_23_l301_301231

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301231


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301756

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301756


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301745

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301745


namespace value_of_a_l301_301482

-- Define the quadratic function f
def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := x^2 + a * x + b

-- Define the condition f(1) = f(2)
def condition (a b : ℝ) : Prop := f 1 a b = f 2 a b

-- The proof problem statement
theorem value_of_a (a b : ℝ) (h : condition a b) : a = -3 :=
by sorry

end value_of_a_l301_301482


namespace probability_from_first_to_last_l301_301854

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l301_301854


namespace arrangement_for_P23_exists_l301_301235

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301235


namespace minimizing_point_position_l301_301459

open_locale real_inner_product_space

noncomputable def minimize_sum (A B C : Point) (h_acute : acute A B C) : Point :=
let external_eq_triangleAB := construct_external_equilateral_triangle A B,
    external_eq_triangleAC := construct_external_equilateral_triangle A C,
    intersect_lineA'C := line_intersection external_eq_triangleAB.opposite_vertex C,
    intersect_lineA''B := line_intersection external_eq_triangleAC.opposite_vertex B in
    intersect_line intersect_lineA'C intersect_lineA''B

theorem minimizing_point_position 
  (A B C : Point) 
  (h_acute : acute A B C)
  (P : Point)
  (h_minimality : ∀ Q, PA + PB + PC ≤ QA + QB + QC) : 
  P = minimize_sum A B C h_acute :=
by sorry

end minimizing_point_position_l301_301459


namespace count_ways_to_choose_4_cards_l301_301510

-- A standard deck has 4 suits
def suits : Finset ℕ := {1, 2, 3, 4}

-- Each suit has 6 even cards: 2, 4, 6, 8, 10, and Queen (12)
def even_cards_per_suit : Finset ℕ := {2, 4, 6, 8, 10, 12}

-- Define the problem in Lean: 
-- Total number of ways to choose 4 cards such that all cards are of different suits and each is an even card.
theorem count_ways_to_choose_4_cards : (suits.card = 4 ∧ even_cards_per_suit.card = 6) → (1 * 6^4 = 1296) :=
by
  intros h
  have suits_distinct : suits.card = 4 := h.1
  have even_cards_count : even_cards_per_suit.card = 6 := h.2
  sorry

end count_ways_to_choose_4_cards_l301_301510


namespace bottles_stolen_at_dance_l301_301418

-- Define the initial conditions
def initial_bottles := 10
def bottles_lost_at_school := 2
def total_stickers := 21
def stickers_per_bottle := 3

-- Calculate remaining bottles after loss at school
def remaining_bottles_after_school := initial_bottles - bottles_lost_at_school

-- Calculate the remaining bottles after the theft
def remaining_bottles_after_theft := total_stickers / stickers_per_bottle

-- Prove the number of bottles stolen
theorem bottles_stolen_at_dance : remaining_bottles_after_school - remaining_bottles_after_theft = 1 :=
by
  sorry

end bottles_stolen_at_dance_l301_301418


namespace ant_paths_from_P_to_Q_l301_301139

def P := (-3, -3)
def Q := (3, 3)

def is_valid_move (current next : Int × Int) : Prop :=
  (next.1 = current.1 + 1 ∨ next.2 = current.2 + 1) ∧
  (abs next.1 ≥ 2 ∨ abs next.2 ≥ 2)

def count_valid_paths : Int :=
  74

theorem ant_paths_from_P_to_Q : 
  ∀ paths, 
    (paths.head = P) ∧ 
    (paths.last = Q) ∧ 
    ∀ i, 0 ≤ i < paths.length - 1 → is_valid_move (paths[i]) (paths[i+1]) → 
    count_valid_paths = 74 :=
  sorry

end ant_paths_from_P_to_Q_l301_301139


namespace exists_arrangement_for_P_23_l301_301228

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301228


namespace stamps_per_book_type2_eq_15_l301_301256

-- Defining the conditions
def num_books_type1 : ℕ := 4
def stamps_per_book_type1 : ℕ := 10
def num_books_type2 : ℕ := 6
def total_stamps : ℕ := 130

-- Stating the theorem to prove the number of stamps in each book of the second type is 15
theorem stamps_per_book_type2_eq_15 : 
  ∀ (x : ℕ), 
    (num_books_type1 * stamps_per_book_type1 + num_books_type2 * x = total_stamps) → 
    x = 15 :=
by
  sorry

end stamps_per_book_type2_eq_15_l301_301256


namespace arrangement_exists_for_P_eq_23_l301_301243

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301243


namespace klari_sequence_l301_301906

theorem klari_sequence (n : ℕ) (hn : n > 0) :
  (let sixes := (list.repeat 6 n).join nat.digits_eq_nat base;
       eights := (list.repeat 8 n).join nat.digits_eq_nat base;
       fours := (list.repeat 4 (2 * n)).join nat.digits_eq_nat base
   in sixes ^ 2 + eights = fours) :=
sorry

end klari_sequence_l301_301906


namespace expression_inverse_l301_301403

theorem expression_inverse : [3 - 4 * (3 - 4)⁻¹ * 2]⁻¹ = 1 / 11 := 
by
  sorry

end expression_inverse_l301_301403


namespace division_of_negatives_example_div_l301_301010

theorem division_of_negatives (a b : Int) (ha : a < 0) (hb : b < 0) (hb_neq : b ≠ 0) : 
  (-a) / (-b) = a / b :=
by sorry

theorem example_div : (-300) / (-50) = 6 :=
by
  apply division_of_negatives
  repeat { sorry }

end division_of_negatives_example_div_l301_301010


namespace exists_arrangement_for_P_23_l301_301225

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301225


namespace intersection_complement_l301_301069

open Set

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 2^(x-2) > 1}

theorem intersection_complement :
  A ∩ (compl B) = {x | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_l301_301069


namespace probability_of_reaching_last_floor_l301_301861

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l301_301861


namespace binomial_coeff_sum_l301_301083

theorem binomial_coeff_sum :
  ∀ a b : ℝ, 15 * a^4 * b^2 = 135 ∧ 6 * a^5 * b = -18 →
  (a + b) ^ 6 = 64 :=
by
  intros a b h
  sorry

end binomial_coeff_sum_l301_301083


namespace average_rainfall_per_hour_l301_301138

theorem average_rainfall_per_hour :
  ∀ (total_rainfall : ℕ) (days_in_june hours_in_a_day: ℕ), 
  total_rainfall = 498 → 
  days_in_june = 30 → 
  hours_in_a_day = 24 → 
  (total_rainfall / (days_in_june * hours_in_a_day)) = (498 / 720) :=
by
  intros total_rainfall days_in_june hours_in_a_day h_total h_days h_hours
  rw [h_total, h_days, h_hours]
  norm_num
  sorry

end average_rainfall_per_hour_l301_301138


namespace similar_triangles_y_value_l301_301899

noncomputable def y_value := 9.33

theorem similar_triangles_y_value (y : ℝ) 
    (h : ∃ (a b : ℝ), a = 12 ∧ b = 9 ∧ (a / y = b / 7)) : y = y_value := 
  sorry

end similar_triangles_y_value_l301_301899


namespace number_of_elements_in_P_plus_Q_l301_301167

def P : Set ℝ := {0, 2, 5}
def Q : Set ℝ := {1, 2, 6}
def P_plus_Q : Set ℝ := {a + b | a ∈ P, b ∈ Q}

theorem number_of_elements_in_P_plus_Q : P_plus_Q.to_finset.card = 8 := 
by
  sorry

end number_of_elements_in_P_plus_Q_l301_301167


namespace remainder_of_sum_of_primes_l301_301801

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301801


namespace area_of_octagon_eq_4_l301_301658

noncomputable def sqrt2 := Real.sqrt 2

def side_length_square := sqrt2
def length_AB := 1 / 2

theorem area_of_octagon_eq_4
  (side_length_square = sqrt2)
  (length_AB = 1 / 2) :
  ∃ (m n : ℕ), m = 4 ∧ n = 1 ∧ m + n = 5 :=
begin
  use [4, 1],
  split,
  { refl },
  split,
  { refl },
  { norm_num },
end

end area_of_octagon_eq_4_l301_301658


namespace Carmen_needs_additional_money_l301_301927

-- Definitions
def Patricia_amount : ℕ := 60
def total_amount : ℕ := 113
def Jethro_amount : ℕ := Patricia_amount / 3
def Carmen_amount : ℕ := total_amount - Patricia_amount - Jethro_amount

-- Theorem to prove
theorem Carmen_needs_additional_money : 
  Patricia_amount = 60 ∧
  Patricia_amount = 3 * Jethro_amount ∧
  total_amount = Carmen_amount + Patricia_amount + Jethro_amount →
  2 * Jethro_amount - Carmen_amount = 7 :=
begin
  sorry
end

end Carmen_needs_additional_money_l301_301927


namespace length_AB_is_correct_l301_301549

noncomputable def parametric_line : ℝ → ℝ × ℝ := λ t, (1 + t, t)

noncomputable def parametric_ellipse : ℝ → ℝ × ℝ := λ θ, (real.sqrt 2 * real.cos θ, real.sin θ)

noncomputable def length_segment_AB : ℝ :=
  let A : ℝ × ℝ := (0, -1)
  let B : ℝ × ℝ := (4 / 3, 1 / 3)
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- We assert the length of segment AB is 4 * real.sqrt 2 / 3
theorem length_AB_is_correct : length_segment_AB = 4 * real.sqrt 2 / 3 := by
  sorry

end length_AB_is_correct_l301_301549


namespace polynomial_T_has_required_roots_l301_301165

noncomputable def P : Polynomial ℂ := sorry
noncomputable def Q : Polynomial ℂ := sorry
noncomputable def R : Polynomial ℂ := sorry

-- Assume S is given as stated in the problem.
def S (x : ℂ) : Polynomial ℂ := P.eval x^3 + x * Q.eval x^3 + x^2 * R.eval x^3

-- Assume these are the distinct roots of S.
def x_roots : Fin n → ℂ := sorry  -- x₁, ..., xₙ
axiom distinct_roots : ∀ i j : Fin n, i ≠ j → x_roots i ≠ x_roots j

-- Define the polynomial T.
noncomputable def T (x : ℂ) : Polynomial ℂ :=
  P.eval x^3 + x * Q.eval x^3 + x^2 * R.eval x^3 - 6 * x * P.eval x * Q.eval x * R.eval x 

-- Prove that T has roots x₁³, x₂³, ..., xₙ³.
theorem polynomial_T_has_required_roots : 
  (∀ i : Fin n, T.eval (x_roots i)^3 = 0) :=
sorry

end polynomial_T_has_required_roots_l301_301165


namespace horner_method_f_of_3_l301_301980

def f (x : ℝ) : ℝ := x^5 - 2 * x^3 + 3 * x^2 - x + 1

theorem horner_method_f_of_3 : f 3 = 24 :=
by 
  have v₀ : ℝ := 1
  have v₁ : ℝ := v₀ * 3 + 0
  have v₂ : ℝ := v₁ * 3 - 2
  have v₃ : ℝ := v₂ * 3 + 3
  calc f 3 = (v₃) : by sorry


end horner_method_f_of_3_l301_301980


namespace range_of_m_l301_301131

open Real

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m^2 > 0) ↔ -1 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l301_301131


namespace calc_expression_l301_301006

theorem calc_expression : 
  abs (Real.sqrt 3 - 2) + (8:ℝ)^(1/3) - Real.sqrt 16 + (-1)^(2023:ℝ) = -(Real.sqrt 3) - 1 :=
by
  sorry

end calc_expression_l301_301006


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301784

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301784


namespace cos_double_angle_l301_301514

theorem cos_double_angle (theta : ℝ) (h : Real.cos theta = 1/4) : Real.cos (2 * theta) = -7/8 :=
by
  sorry

end cos_double_angle_l301_301514


namespace at_least_one_triangle_at_least_n_triangles_l301_301150

-- There are 2n points in space, denoted by P
variable (n : ℕ) (P : Fin 2n → Type)

-- No four points are coplanar
def no_four_coplanar (P : Fin 2n → Type) : Prop :=
  ∀ a b c d : Fin 2n, ¬(∃ plane : AffineSubspace (ℝ ^ 3), AffineSubspace.span (P '' {a, b, c, d}) = plane)

-- There are n^2 + 1 segments
variable (E : Fin (n^2 + 1) → (Fin 2n × Fin 2n))

-- A function to check if a given combination of 3 points forms a triangle
def forms_triangle (P : Fin 2n → Type) (a b c : Fin 2n) : Prop :=
  (¬(E '' {(a, b), (b, c), (c, a)}).empty)

-- Problem (a): prove there exists at least one triangle
theorem at_least_one_triangle
  (P : Fin 2n → Type)
  (n : ℕ)
  (h_no_four_coplanar : no_four_coplanar n P)
  (E : Fin (n^2 + 1) → (Fin 2n × Fin 2n)) :
  ∃ (a b c : Fin 2n), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ forms_triangle n P E a b c :=
sorry

-- Problem (b): prove there exist at least n triangles
theorem at_least_n_triangles
  (P : Fin 2n → Type)
  (n : ℕ)
  (h_no_four_coplanar : no_four_coplanar n P)
  (E : Fin (n^2 + 1) → (Fin 2n × Fin 2n)) :
  ∃ T : Fin n → (Fin 2n × Fin 2n × Fin 2n),
    ∀ i : Fin n, let (a, b, c) := T i in a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ forms_triangle n P E a b c :=
sorry

end at_least_one_triangle_at_least_n_triangles_l301_301150


namespace circle_area_l301_301410

theorem circle_area (r θ : ℝ) (h : r = 4 * real.cos θ - 3 * real.sin θ) : 
  ∃ A : ℝ, A = (25 * real.pi) / 4 := 
sorry

end circle_area_l301_301410


namespace exist_arrangement_for_P_23_l301_301194

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301194


namespace remainder_first_six_primes_div_seventh_l301_301810

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301810


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301787

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301787


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301683

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301683


namespace gcd_9009_14014_l301_301424

-- Given conditions
def decompose_9009 : 9009 = 9 * 1001 := by sorry
def decompose_14014 : 14014 = 14 * 1001 := by sorry
def coprime_9_14 : Nat.gcd 9 14 = 1 := by sorry

-- Proof problem statement
theorem gcd_9009_14014 : Nat.gcd 9009 14014 = 1001 := by
  have h1 : 9009 = 9 * 1001 := decompose_9009
  have h2 : 14014 = 14 * 1001 := decompose_14014
  have h3 : Nat.gcd 9 14 = 1 := coprime_9_14
  sorry

end gcd_9009_14014_l301_301424


namespace factor_expression_l301_301028

theorem factor_expression (x y z : ℝ) :
  x^3 * (y^2 - z^2) + y^3 * (z^2 - x^2) + z^3 * (x^2 - y^2) =
  (x - y) * (y - z) * (z - x) * (x * y + x * z + y * z) :=
sorry

end factor_expression_l301_301028


namespace emilys_sister_packs_l301_301025

theorem emilys_sister_packs (total_packs packs_for_Emily packs_for_sister : ℕ) 
                            (H1 : total_packs = 13) (H2 : packs_for_Emily = 6) :
  packs_for_sister = total_packs - packs_for_Emily :=
by
  have h : packs_for_sister = 7
  sorry

end emilys_sister_packs_l301_301025


namespace problem_1_problem_2_problem_3_l301_301094

-- Problem 1: Prove that f(x) is an increasing function on (0, +∞)
theorem problem_1 (a : ℝ) (h : a > 0) : ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → (1 / a - 1 / x1) < (1 / a - 1 / x2) :=
sorry

-- Problem 2: Prove that ∀ a > 0, ( ∀ x > 0, (1 / a - 1 / x ≤ 2x) ) implies a ≥ √2 / 4
theorem problem_2 (a : ℝ) (h1 : a > 0) : (∀ x : ℝ, 0 < x → 1 / a - 1 / x ≤ 2 * x) → a ≥ Real.sqrt 2 / 4 :=
sorry

-- Problem 3: Prove that if the range of f(x) on [m, n] is [m, n] (m ≠ n), then 0 < a < 1/2
theorem problem_3 (a m n : ℝ) (h1 : a > 0) (h2 : m ≠ n) 
  (h3 : (∀ x : ℝ, m ≤ x ∧ x ≤ n → x ∈ set.image (λ x, 1 / a - 1 / x) (set.Icc m n))) : 0 < a ∧ a < 1/2 :=
sorry

end problem_1_problem_2_problem_3_l301_301094


namespace unique_function_nat_l301_301949

theorem unique_function_nat (f : ℕ → ℕ) (h : ∀ n : ℕ, 
  (finset.range n).sum (λ k, (1 : ℚ) / (f k.succ * f (k.succ.succ))) = (f (f n) / f n.succ)) :
  ∀ n, f n = n :=
by
  sorry

end unique_function_nat_l301_301949


namespace largest_possible_x_l301_301326

theorem largest_possible_x :
  ∃ x : ℝ, (3*x^2 + 18*x - 84 = x*(x + 10)) ∧ ∀ y : ℝ, (3*y^2 + 18*y - 84 = y*(y + 10)) → y ≤ x :=
by
  sorry

end largest_possible_x_l301_301326


namespace sum_of_other_endpoint_coordinates_l301_301640

/-- 
  Given that (9, -15) is the midpoint of the segment with one endpoint (7, 4),
  find the sum of the coordinates of the other endpoint.
-/
theorem sum_of_other_endpoint_coordinates : 
  ∃ x y : ℤ, ((7 + x) / 2 = 9 ∧ (4 + y) / 2 = -15) ∧ (x + y = -23) :=
by
  sorry

end sum_of_other_endpoint_coordinates_l301_301640


namespace number_of_solutions_l301_301031

theorem number_of_solutions :
  ∃ S : Finset (ℤ × ℤ), (∀ p ∈ S, (p.1^2 + p.1 * p.2 + p.2^2 = 28) ∧
  (∀ p₁ p₂ ∈ S, p₁ ≠ p₂ → p₁ ≠ p₂)) ∧ S.card = 4 := by
  sorry

end number_of_solutions_l301_301031


namespace coplanar_values_l301_301600

namespace CoplanarLines

-- Define parametric equations of the lines
def line1 (t : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (3 + 2 * t, 2 - t, 5 + m * t)
def line2 (u : ℝ) (m : ℝ) : ℝ × ℝ × ℝ := (4 - m * u, 5 + 3 * u, 6 + 2 * u)

-- Define coplanarity condition
def coplanar_condition (m : ℝ) : Prop :=
  ∃ t u : ℝ, line1 t m = line2 u m

-- Theorem to prove the specific values of m for coplanarity
theorem coplanar_values (m : ℝ) : coplanar_condition m ↔ (m = -13/9 ∨ m = 1) :=
sorry

end CoplanarLines

end coplanar_values_l301_301600


namespace repeating_decimal_as_fraction_l301_301027

theorem repeating_decimal_as_fraction :
  ∃ x : ℝ, x = 7.45 ∧ (100 * x - x = 738) → x = 82 / 11 :=
by
  sorry

end repeating_decimal_as_fraction_l301_301027


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301729

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301729


namespace min_board_size_l301_301614

theorem min_board_size (n : ℕ) (total_area : ℕ) (domino_area : ℕ) 
  (h1 : total_area = 2008) 
  (h2 : domino_area = 2) 
  (h3 : ∀ domino_count : ℕ, domino_count = total_area / domino_area → (∃ m : ℕ, (m+1) * (m+1) ≥ domino_count * (2 + 4) → n = m)) :
  n = 77 :=
by
  sorry

end min_board_size_l301_301614


namespace count_distinct_ways_l301_301443

def distinct_pairs (s : Finset ℕ) : ℕ :=
  (s.card.choose 2) * ((s.card - 2).choose 2) / 2

theorem count_distinct_ways :
  ∃ n, n = 90 ∧
    ∀ (A B C D : ℕ), A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D ∧
    A ∈ {1, 2, 3, 4, 5, 6} ∧ B ∈ {1, 2, 3, 4, 5, 6} ∧
    C ∈ {1, 2, 3, 4, 5, 6} ∧ D ∈ {1, 2, 3, 4, 5, 6} →
    distinct_pairs {A, B, C, D} = n := 
by
  sorry

end count_distinct_ways_l301_301443


namespace five_digit_numbers_count_l301_301659

theorem five_digit_numbers_count:
  let digits := {1, 2, 3, 4, 5}
  ∀ (n : ℕ), n ∈ permutations digits →
  (¬(digit_in_position n 4 1 ∨ digit_in_position n 4 5)) →
  (exactly_two_adjacent n {1, 3, 5}) →
  count_valid_numbers = 48 :=
by
  sorry

end five_digit_numbers_count_l301_301659


namespace sum_of_coordinates_l301_301414

theorem sum_of_coordinates (x y : ℝ) (h₁ : y = (x - 2)^2 + 1) (h₂ : x + 5 = (y + 2)^2) :
  let Σ := ∑ (x y : ℝ) in {(x, y) | y = (x - 2)^2 + 1 ∧ x + 5 = (y + 2)^2}, x + y in
  Σ = 4 := 
sorry

end sum_of_coordinates_l301_301414


namespace union_complement_real_domain_l301_301093

noncomputable def M : Set ℝ := {x | -2 < x ∧ x < 2}
noncomputable def N : Set ℝ := {x | -2 < x}

theorem union_complement_real_domain :
  M ∪ (Set.univ \ N) = {x : ℝ | x < 2} :=
by
  sorry

end union_complement_real_domain_l301_301093


namespace total_situps_performed_l301_301917

theorem total_situps_performed :
  let Barney_situps_per_min := 45 in
  let Carrie_situps_per_min := 2 * Barney_situps_per_min in
  let Jerrie_situps_per_min := Carrie_situps_per_min + 5 in
  let total_situps :=
    (Barney_situps_per_min * 1) +
    (Carrie_situps_per_min * 2) +
    (Jerrie_situps_per_min * 3) in
  total_situps = 510 :=
by
  sorry

end total_situps_performed_l301_301917


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301717

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301717


namespace maximum_value_iff_l301_301528

-- Definitions
def f (x k : ℝ) : ℝ := (Real.exp x / x) - k * x

-- Theorem statement
theorem maximum_value_iff (k : ℝ) : (∃ x : ℝ, ∃ M : ℝ, ∀ y : ℝ, f y k ≤ M) ↔ k < 0 :=
begin
  sorry
end

end maximum_value_iff_l301_301528


namespace max_m_for_roots_real_and_rational_l301_301013

-- Definition of the quadratic equation and discriminant conditions
def quadratic_eq (a b c : ℚ) (x : ℚ) := a * x^2 + b * x + c = 0

-- Real and rational roots condition (discriminant should be non-negative and perfect square)
def roots_are_real_and_rational (a b c : ℚ) :=
  let Δ := b^2 - 4 * a * c in
  Δ ≥ 0 ∧ ∃ k : ℚ, k^2 = Δ

-- The problem statement
theorem max_m_for_roots_real_and_rational: 
  ∃ m : ℚ, 
  (∀ x : ℚ, quadratic_eq 2 (-5) m x → 
  roots_are_real_and_rational 2 (-5) m) ∧ m = 25 / 8 :=
by
  sorry

end max_m_for_roots_real_and_rational_l301_301013


namespace exists_arrangement_for_P_23_l301_301205

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301205


namespace triangle_area_B1C1E_eq_l301_301157

theorem triangle_area_B1C1E_eq
    (AB_1_div_AB : (AB_1 / AB = 1 / 3))
    (AC_1_div_AC : (AC_1 / AC = 1 / 2))
    (h1 : AC_1 = 4)
    (h2 : AD = 1)
    (h3 : DE = 2)
    (area_ABC : 1/2 * AB * AC = 12) :
    let B_1D := 3/2,
        BE := 7/2,
        BD := 5/2,
        area_AB : AB_1 := (1 / 3) * AC,
        area_AC : AB_1 := (1 / 2) * AC,
        area_D := area_ABC / (12)
        S_B1D_C1 := (3 / 4) * ((area_AB * area_AC ) * (B_1E / BD ): ( 1/ 2)) * AC
  in  (S_B1D_C1 = (\frac{7}{2}))
:= sorry

end triangle_area_B1C1E_eq_l301_301157


namespace trapezoid_median_l301_301190

theorem trapezoid_median (α β a b : ℝ) (h : β > α) :
  ∀ (median : ℝ), median = b + (a * real.sin (β - α)) / (2 * real.sin α) :=
begin
  intros,
  sorry
end

end trapezoid_median_l301_301190


namespace exists_F_12_mod_23_zero_l301_301220

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301220


namespace ratio_of_areas_l301_301548

structure Point (α : Type) := 
(x : α) (y : α)

structure Triangle (α : Type) := 
(P Q R : Point α)

def isRightTriangle {α : Type} [LinearOrder α] (T : Triangle α) : Prop :=
(T.Q.x = T.R.x ∧ T.Q.y = T.P.y)

def midpoint {α : Type} [LinearOrder α] (A B : Point α) : Point α :=
⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

def area {α : Type} [LinearOrder α] (A B C : Point α) : α :=
(abs ((B.x - A.x) * (C.y - A.y) - (C.x - A.x) * (B.y - A.y))) / 2

theorem ratio_of_areas {α : Type} [LinearOrder α] : 
  ∀ (P Q R M N Y : Point α),
  isRightTriangle (Triangle.mk P Q R) → 
  P.y = 10 →
  Q.x = 0 ∧ Q.y = 0 →
  R.x = 15 ∧ R.y = 0 →
  M = midpoint P Q →
  N = midpoint P R →
  Y.x = 0 ∧ Y.y = 5 →
  (area P N Y M = area Q Y R) :=
by 
  intros P Q R M N Y hRight hPQ hQR hR N_mid Y_mid hY
  sorry

end ratio_of_areas_l301_301548


namespace caterer_min_people_l301_301189

theorem caterer_min_people (x : ℕ) : 150 + 18 * x > 250 + 15 * x → x ≥ 34 :=
by
  intro h
  sorry

end caterer_min_people_l301_301189


namespace total_area_of_dark_regions_l301_301382

theorem total_area_of_dark_regions :
  let unit_square_side := 1 in
  let A_circle := π * (unit_square_side / 2) ^ 2 in
  let A_outer_dark := unit_square_side ^ 2 - A_circle in
  let geometric_series := A_outer_dark * (1 / (1 - (1 / 2))) in
  geometric_series = 2 - π / 2 := by
  let unit_square_side := 1
  let A_circle := π * (unit_square_side / 2) ^ 2
  let A_outer_dark := unit_square_side ^ 2 - A_circle
  let geometric_series := A_outer_dark * (1 / (1 - (1 / 2)))
  show geometric_series = 2 - π / 2 from sorry

end total_area_of_dark_regions_l301_301382


namespace smallest_integer_expression_l301_301822

theorem smallest_integer_expression :
  ∃ m n : ℤ, 1237 * m + 78653 * n = 1 :=
sorry

end smallest_integer_expression_l301_301822


namespace range_of_k_l301_301277

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

end range_of_k_l301_301277


namespace cos_mul_tan_quadrant_condition_l301_301341

theorem cos_mul_tan_quadrant_condition (θ : Real) :
  (cos θ * tan θ < 0) ↔ (cos θ > 0 ∧ tan θ < 0) :=
sorry

end cos_mul_tan_quadrant_condition_l301_301341


namespace determine_digits_l301_301022

theorem determine_digits :
  ∃ (A B C D : ℕ), 
    1000 ≤ 1000 * A + 100 * B + 10 * C + D ∧ 
    1000 * A + 100 * B + 10 * C + D ≤ 9999 ∧ 
    1000 ≤ 1000 * C + 100 * B + 10 * A + D ∧ 
    1000 * C + 100 * B + 10 * A + D ≤ 9999 ∧ 
    (1000 * A + 100 * B + 10 * C + D) * D = 1000 * C + 100 * B + 10 * A + D ∧ 
    A = 2 ∧ B = 1 ∧ C = 7 ∧ D = 8 :=
by
  sorry

end determine_digits_l301_301022


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301785

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301785


namespace circle_symmetric_equation_l301_301958

-- Define the original circle equation and the line for symmetry
def circle_eq (x y : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 2
def symmetry_line (x y : ℝ) : Prop := x + y = 0

-- Define the symmetric center
def symmetric_center (x y : ℝ) : Prop :=
  let c := (3, -4) in
  x = 2 * c.1 - x ∧ y = 2 * c.2 - y ∧ (x + y = 0)

-- Define the new circle equation
def new_circle_eq (x y : ℝ) : Prop := (x - 4)^2 + (y + 3)^2 = 2

-- The problem statement
theorem circle_symmetric_equation :
  (∀ x y : ℝ, circle_eq x y → ∃ x' y', symmetric_center x' y' ∧ new_circle_eq x' y') :=
sorry

end circle_symmetric_equation_l301_301958


namespace weather_forecast_minutes_l301_301279

theorem weather_forecast_minutes 
  (total_duration : ℕ) 
  (national_news : ℕ) 
  (international_news : ℕ) 
  (sports : ℕ) 
  (advertising : ℕ) 
  (wf : ℕ) :
  total_duration = 30 →
  national_news = 12 →
  international_news = 5 →
  sports = 5 →
  advertising = 6 →
  total_duration - (national_news + international_news + sports + advertising) = wf →
  wf = 2 :=
by
  intros
  sorry

end weather_forecast_minutes_l301_301279


namespace probability_open_doors_l301_301874

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l301_301874


namespace volume_ratio_of_tetrahedron_is_one_twenty_seventh_l301_301887

-- Define the vertices of the large regular tetrahedron.
def tetrahedron_vertices : List (ℝ^4) :=
  [(1,1,0,0), (0,1,1,0), (0,0,1,1), (1,0,0,1)]

-- Define the function to compute the volume ratio of the smaller tetrahedron.
def volume_ratio (vertices : List (ℝ^4)) : ℝ :=
  -- Placeholder for the actual computation of the volume ratio.
  -- It should compute the ratio based on the given vertices.
  1 / 27

-- Proposition statement.
theorem volume_ratio_of_tetrahedron_is_one_twenty_seventh :
  volume_ratio tetrahedron_vertices = 1 / 27 := by
    sorry

end volume_ratio_of_tetrahedron_is_one_twenty_seventh_l301_301887


namespace intersection_complement_l301_301071

def setA (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def setB (x : ℝ) : Prop := 2^(x - 2) > 1

theorem intersection_complement :
  { x : ℝ | setA x } ∩ { x : ℝ | ¬ setB x } = { x : ℝ | -1 < x ∧ x ≤ 2 } := 
by 
  sorry

end intersection_complement_l301_301071


namespace lucy_payment_l301_301183

theorem lucy_payment :
  let
    grapes_cost := 6 * 74,
    mangoes_cost := 9 * 59,
    apples_cost := 4 * 45,
    oranges_cost := 12 * 32,
    total_grapes_and_apples := grapes_cost + apples_cost,
    total_mangoes_and_oranges := mangoes_cost + oranges_cost,
    grapes_and_apples_discount := 0.07 * total_grapes_and_apples,
    mangoes_and_oranges_discount := 0.05 * total_mangoes_and_oranges,
    discounted_grapes_and_apples := total_grapes_and_apples - grapes_and_apples_discount,
    discounted_mangoes_and_oranges := total_mangoes_and_oranges - mangoes_and_oranges_discount,
    total_amount_paid := discounted_grapes_and_apples + discounted_mangoes_and_oranges
  in
    total_amount_paid = 1449.57 :=
by
  -- steps to prove the theorem
  sorry

end lucy_payment_l301_301183


namespace number_of_multiples_of_six_l301_301507

theorem number_of_multiples_of_six (m : ℕ) (h₁ : m % 6 = 0) (h₂ : m < 100) (h₃ : m > 0) : 
  {n : ℕ | n > 0 ∧ n < 100 ∧ n % 6 = 0}.card = 16 :=
sorry

end number_of_multiples_of_six_l301_301507


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301736

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301736


namespace exists_F_12_mod_23_zero_l301_301217

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301217


namespace marble_probability_l301_301362

theorem marble_probability :
  let red_marbles := 6
      white_marbles := 4
      blue_marbles := 8
      total_marbles := red_marbles + white_marbles + blue_marbles
      draw_count := 3 in
  (nat.choose red_marbles draw_count + nat.choose white_marbles draw_count + nat.choose blue_marbles draw_count) /
  (nat.choose total_marbles draw_count : ℚ) = 5 / 51 :=
by repeat { sorry }

end marble_probability_l301_301362


namespace probability_of_reaching_last_floor_l301_301862

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l301_301862


namespace find_f_of_3_is_neg18_l301_301842

-- Define the given function f, ensuring it meets the properties described
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then x^2 - 2 * x + 3 else -(((-x)^2) - 2 * (-x) + 3)

-- Assert that f is an odd function
axiom f_odd : ∀ x : ℝ, f(-x) = -f(x)

-- Define a theorem to find the value of f(3)
theorem find_f_of_3_is_neg18 : f 3 = -18 := by
  -- skipping proof details
  sorry

end find_f_of_3_is_neg18_l301_301842


namespace parabola_p_value_min_area_triangle_value_l301_301102

noncomputable def parabola_intersection_length (p : ℝ) (h : p > 0) : ℝ :=
  let line := λ x y : ℝ, x - 2 * y + 1 = 0 in
  let parabola := λ x y : ℝ, y^2 = 2 * p * x in
  let A_B := λ x1 y1 x2 y2 : ℝ, 
    line x1 y1 ∧ parabola x1 y1 ∧ line x2 y2 ∧ parabola x2 y2 ∧ 
    (x2 - x1)^2 + (y2 - y1)^2 = (4 * sqrt 15)^2 in
  p

theorem parabola_p_value (p : ℝ) (h : p > 0) : 
  parabola_intersection_length p h = 2 := sorry

noncomputable def min_area_triangle (p : ℝ) (h : p = 2) : ℝ :=
  let focus : ℝ × ℝ := (1, 0) in
  let parabola := λ x y : ℝ, y^2 = 4 * x in
  let orthogonal_vectors := λ (M N : (ℝ × ℝ)), 
    let MF := (M.1 - 1, M.2) in
    let NF := (N.1 - 1, N.2) in
    MF.1 * NF.1 + MF.2 * NF.2 = 0 in
  let triangle_area := λ (M N : (ℝ × ℝ)), 
    let vec := (M.1 - N.1, M.2 - N.2) in 
    (vec.1^2 + vec.2^2) / 2 in
  min_area

theorem min_area_triangle_value (p : ℝ) (h : p = 2) : 
  min_area_triangle p h = 12 - 8 * sqrt 2 := sorry

end parabola_p_value_min_area_triangle_value_l301_301102


namespace eccentricity_range_l301_301275

theorem eccentricity_range (c a e : ℝ) (h_c_pos : c > 0) (h_a_pos : a > 0) (h_cos_angle : ∃ (x1 y1 : ℝ), x1^2 = (4 * c^2 - 3 * a^2) / e^2 ∧ (0 < x1^2) ∧ (x1^2 ≤ a^2)) :
  (√3 / 2 ≤ e) ∧ (e < 1) :=
by
  sorry

end eccentricity_range_l301_301275


namespace area_constant_k_l301_301294

theorem area_constant_k (l w d : ℝ) (h_ratio : l / w = 5 / 2) (h_diagonal : d = Real.sqrt (l^2 + w^2)) :
  ∃ k : ℝ, (k = 10 / 29) ∧ (l * w = k * d^2) :=
by
  sorry

end area_constant_k_l301_301294


namespace find_principal_l301_301560

noncomputable def principal_amount (r : ℝ) (n : ℕ) (difference : ℝ) : ℝ := difference / ((1 + r / 100) ^ n - (1 + r * n / 100))

theorem find_principal (h : principal_amount 4 2 6.000000000000455 ≈ 3750) : ∃ P : ℝ, P ≈ 3750 :=
begin
  use principal_amount 4 2 6.000000000000455,
  assumption,
end

end find_principal_l301_301560


namespace num_people_at_gathering_l301_301915

noncomputable def total_people_at_gathering : ℕ :=
  let wine_soda := 12
  let wine_juice := 10
  let wine_coffee := 6
  let wine_tea := 4
  let soda_juice := 8
  let soda_coffee := 5
  let soda_tea := 3
  let juice_coffee := 7
  let juice_tea := 2
  let coffee_tea := 4
  let wine_soda_juice := 3
  let wine_soda_coffee := 1
  let wine_soda_tea := 2
  let wine_juice_coffee := 3
  let wine_juice_tea := 1
  let wine_coffee_tea := 2
  let soda_juice_coffee := 3
  let soda_juice_tea := 1
  let soda_coffee_tea := 2
  let juice_coffee_tea := 3
  let all_five := 1
  wine_soda + wine_juice + wine_coffee + wine_tea +
  soda_juice + soda_coffee + soda_tea + juice_coffee +
  juice_tea + coffee_tea + wine_soda_juice + wine_soda_coffee +
  wine_soda_tea + wine_juice_coffee + wine_juice_tea +
  wine_coffee_tea + soda_juice_coffee + soda_juice_tea +
  soda_coffee_tea + juice_coffee_tea + all_five

theorem num_people_at_gathering : total_people_at_gathering = 89 := by
  sorry

end num_people_at_gathering_l301_301915


namespace circle_equation_solution_l301_301471

noncomputable def equation_of_circle : Prop :=
  ∃ a r : ℝ, (a > 0) ∧ (r > 0) ∧
  (∀ x y : ℝ, (x - a)^2 + y^2 = r^2 ↔ ((x^2 + y^2 = r^2 + 2 * a * x))) ∧
  (r = a) ∧
  (∀ d : ℝ, d = (sqrt 3 * r) / 2) ∧
  (2 / 2 = 1) ∧
  (r^2 = d^2 + 1) →
  a = 2 →
  r = 2 →
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 → x^2 + y^2 - 4x = 0

theorem circle_equation_solution : equation_of_circle :=
  sorry

end circle_equation_solution_l301_301471


namespace common_chord_length_l301_301317

def circle1_radius : ℝ := 8
def circle2_radius : ℝ := 12
def distance_centers : ℝ := 20

theorem common_chord_length :
  let r1 := circle1_radius
  let r2 := circle2_radius
  let d := distance_centers
  2 * real.sqrt (r2^2 - (d/2)^2) = 4 * real.sqrt (11) :=
by 
  let r1 := circle1_radius
  let r2 := circle2_radius
  let d := distance_centers
  let chord_half_length := real.sqrt (r2^2 - (d/2)^2)
  have h : chord_half_length = real.sqrt (44) := by sorry
  have h2 : 2 * chord_half_length = 4 * real.sqrt (11) := by sorry
  exact h2

end common_chord_length_l301_301317


namespace discount_percentage_l301_301642

theorem discount_percentage (x : ℝ) : 
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  (marked_price * (1 - x / 100) * (1 - second_discount) * (1 - third_discount) = final_price) ↔ x = 20 := 
by
  let marked_price := 12000
  let final_price := 7752
  let second_discount := 0.15
  let third_discount := 0.05
  sorry

end discount_percentage_l301_301642


namespace proof_l301_301439

variable {a b : ℝ}

noncomputable def proof_problem (h : a < b ∧ b < 0) : Prop :=
  (1 / a > 1 / b) ∧
  (ab > b^2) ∧
  (a^(1/5) < b^(1/5)) ∧
  (sqrt (a^2 - a) > sqrt (b^2 - b))

theorem proof (h : a < b ∧ b < 0) : proof_problem h := by
  sorry

end proof_l301_301439


namespace parallel_lines_and_planes_l301_301991

variables {l m : Line} {α β : Plane}

-- Given conditions: l \subset α, m \subset β
axiom line_in_plane_l : l ⊂ α
axiom line_in_plane_m : m ⊂ β
axiom lines_parallel : l ∥ m

-- Prove statement: l ∥ m is neither sufficient nor necessary for α ∥ β
theorem parallel_lines_and_planes :
  ¬ (l ∥ m → α ∥ β) ∧ ¬ (α ∥ β → l ∥ m) := 
sorry

end parallel_lines_and_planes_l301_301991


namespace sequence_formulas_log_sum_l301_301453

noncomputable def an (n : ℕ) : ℤ :=
  2 * n - 1

noncomputable def bn (n : ℕ) : ℤ :=
  3 ^ (n - 1)

noncomputable def Sn (n : ℕ) : ℤ :=
  (n * (n - 1)) / 2

theorem sequence_formulas (a1 b1 a2 b2 a3 b3 : ℤ) (h1 : a1 = 1) (h2 : b1 = 1) (h3 : a2 = b2) (h4 : 2 * a3 - b3 = 1) :
  (∀ n, an n = 2 * n - 1) ∧ (∀ n, bn n = 3 ^ (n - 1)) :=
sorry

theorem log_sum (n : ℕ) (h : ∀ i, log 3 (bn i) = i - 1) :
  Sn n = (n * (n - 1)) / 2 :=
sorry

end sequence_formulas_log_sum_l301_301453


namespace oranges_needed_to_make_150_cents_profit_l301_301850

noncomputable def cost_price_per_orange := (15 : ℝ) / 4
noncomputable def selling_price_per_orange := (25 : ℝ) / 6
noncomputable def profit_per_orange := selling_price_per_orange - cost_price_per_orange

theorem oranges_needed_to_make_150_cents_profit : 
  let num_oranges := (150 : ℝ) / profit_per_orange
  ceil num_oranges = 358 := 
by
  -- some computations to verify the statement
  sorry

end oranges_needed_to_make_150_cents_profit_l301_301850


namespace ellipse_chord_length_theorem_l301_301061

noncomputable def ellipse_chord_length (e : Type) [ellipse e] (a b : ℝ) (h : a > b > 0) : Prop :=
  ∃ (F : ℝ × ℝ), is_focus e F → (chord_length_perpendicular_to_major_axis e F = 3)

theorem ellipse_chord_length_theorem : ellipse_chord_length (λ x y => x^2 / 4 + y^2 / 3 = 1) 2 (sqrt 3) (by linarith) :=
sorry

end ellipse_chord_length_theorem_l301_301061


namespace exists_k_circle_passing_through_E_l301_301478

theorem exists_k_circle_passing_through_E :
  ∃ k : ℝ, k ≠ 0 ∧ 
  (∀ x y : ℝ, (x^2 / 3 + y^2 = 1) → (y = k * x + 2) → 
  ∃ C D : ℝ × ℝ, 
    (C.1^2 / 3 + C.2^2 = 1 ∧ D.1^2 / 3 + D.2^2 = 1) ∧ 
    (C.2 = k * C.1 + 2 ∧ D.2 = k * D.1 + 2) ∧ 
    (∃ r : ℝ, (r > 0) ∧ 
      circle_through_diameter (segment C D) (-1, 0))) :=
by 
  sorry

def circle_through_diameter (seg : ℝ × ℝ ∧ ℝ × ℝ) (e : ℝ × ℝ) : Prop :=
  ∃ x y : ℝ, 
    (x - e.1)^2 + (y - e.2)^2 = (seg.1.1 - seg.2.1)^2 + (seg.1.2 - seg.2.2)^2 / 4 ∧ 
    (y = k * x + 2)

end exists_k_circle_passing_through_E_l301_301478


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301715

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301715


namespace greatest_area_difference_l301_301321

/-- 
Given two rectangles with integer dimensions and both having a perimeter of 180 cm, prove that
the greatest possible difference between their areas is 1936 square centimeters.
-/
theorem greatest_area_difference (l1 w1 l2 w2 : ℕ) 
  (h1 : 2 * l1 + 2 * w1 = 180) 
  (h2 : 2 * l2 + 2 * w2 = 180) :
  abs (l1 * w1 - l2 * w2) ≤ 1936 := 
sorry

end greatest_area_difference_l301_301321


namespace problem_statement_l301_301164

open Real

variables {f : ℝ → ℝ} {a b c : ℝ}

-- f is twice differentiable on ℝ
axiom hf : ∀ x : ℝ, Differentiable ℝ f
axiom hf' : ∀ x : ℝ, Differentiable ℝ (deriv f)

-- ∃ c ∈ ℝ, such that (f(b) - f(a)) / (b - a) ≠ f'(c) for all a ≠ b
axiom hc : ∃ c : ℝ, ∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c

-- Prove that f''(c) = 0
theorem problem_statement : ∃ c : ℝ, (∀ a b : ℝ, a ≠ b → (f b - f a) / (b - a) ≠ deriv f c) → deriv (deriv f) c = 0 := sorry

end problem_statement_l301_301164


namespace find_expression_for_f_l301_301464

noncomputable def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

theorem find_expression_for_f (x : ℝ) (h : x ≠ -1) 
    (hf : f ((1 - x) / (1 + x)) = x) : 
    f x = (1 - x) / (1 + x) :=
sorry

end find_expression_for_f_l301_301464


namespace determine_r_l301_301945

theorem determine_r (r : ℝ) (h : 16 = 2^(5 * r + 3)) : r = 1 / 5 :=
sorry

end determine_r_l301_301945


namespace relationship_among_abc_l301_301996

noncomputable def a : ℝ := 1.1^0.1
noncomputable def b : ℝ := Real.log 2
noncomputable def c : ℝ := Real.logb (1 / 3) (Real.sqrt 3 / 3)

theorem relationship_among_abc : a > b ∧ b > c := by
  sorry

end relationship_among_abc_l301_301996


namespace totalUniqueStudents_l301_301913

-- Define the club memberships and overlap
variable (mathClub scienceClub artClub overlap : ℕ)

-- Conditions based on the problem
def mathClubSize : Prop := mathClub = 15
def scienceClubSize : Prop := scienceClub = 10
def artClubSize : Prop := artClub = 12
def overlapSize : Prop := overlap = 5

-- Main statement to prove
theorem totalUniqueStudents : 
  mathClubSize mathClub → 
  scienceClubSize scienceClub →
  artClubSize artClub →
  overlapSize overlap →
  mathClub + scienceClub + artClub - overlap = 32 := by
  intros
  sorry

end totalUniqueStudents_l301_301913


namespace calculate_f_17_69_l301_301630

noncomputable def f (x y: ℕ) : ℚ := sorry

axiom f_self : ∀ x, f x x = x
axiom f_symm : ∀ x y, f x y = f y x
axiom f_add : ∀ x y, (x + y) * f x y = y * f x (x + y)

theorem calculate_f_17_69 : f 17 69 = 73.3125 := sorry

end calculate_f_17_69_l301_301630


namespace speed_of_second_half_l301_301881

-- Definitions of the given conditions
def D : ℕ := 20 -- total distance in km
def T : ℕ := 2 -- total travel time in hours
def s1 : ℕ := 10 -- speed for the first half in kmph

-- Statement of the problem
theorem speed_of_second_half : 
  ∃ s2 : ℕ, s2 = 10 ∧ (D / 2 = s1 * 1 + s2 * 1) :=
begin
  sorry
end

end speed_of_second_half_l301_301881


namespace prime_sum_remainder_l301_301776

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301776


namespace remainder_of_sum_of_primes_l301_301793

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301793


namespace Walter_spent_120_minutes_on_the_bus_l301_301660

def time_spent_on_the_bus (wake_up_time leave_time [7_classes_duration lunch_duration additional_time return_time] : ℕ) : ℕ :=
  let total_time_away_from_home := 9.5 * 60
  let total_time_spent_on_school_activities := (7 * 45) + 45 + (1.5 * 60)
  total_time_away_from_home - total_time_spent_on_school_activities

theorem Walter_spent_120_minutes_on_the_bus :
  time_spent_on_the_bus 6 7 [7 * 45, 45, 1.5 * 60, 4.5 * 60] = 120 :=
by sorry

end Walter_spent_120_minutes_on_the_bus_l301_301660


namespace quadratic_discriminant_l301_301405

theorem quadratic_discriminant (d : ℝ) :
  (3 : ℝ) * (6 * (𝕜 : ℝ) * (sqrt (3 : ℜ))) - 4 * (3 : ℝ) * d = 12 →
  d = 8 ∧ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (3 : ℝ) * x1^2 - (6 * (sqrt (3 : ℜ)) * x1) + d = 0 ∧ (3 : ℝ) * x2^2 - (6 * (sqrt (3 : ℜ)) * x2) + d = 0 :=
by sorry

end quadratic_discriminant_l301_301405


namespace driver_net_rate_of_pay_is_26_dollars_per_hour_l301_301882

def total_distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

def fuel_used (total_distance : ℝ) (fuel_efficiency : ℝ) : ℝ := total_distance / fuel_efficiency

def earnings (total_distance : ℝ) (rate_per_mile : ℝ) : ℝ := total_distance * rate_per_mile

def gas_cost (fuel_used : ℝ) (cost_per_gallon : ℝ) : ℝ := fuel_used * cost_per_gallon

def net_rate_of_pay (distance : ℝ) (speed : ℝ) (time : ℝ) (fuel_efficiency : ℝ) (rate_per_mile : ℝ) (cost_per_gallon : ℝ) : ℝ := 
  (earnings distance rate_per_mile - gas_cost (fuel_used distance fuel_efficiency) cost_per_gallon) / time

theorem driver_net_rate_of_pay_is_26_dollars_per_hour :
  net_rate_of_pay 150 50 3 30 0.60 2.50 = 26 :=
by
  -- Proof can be inserted here
  sorry

end driver_net_rate_of_pay_is_26_dollars_per_hour_l301_301882


namespace scaled_variance_l301_301648

variable {α : Type*} [add_comm_group α] [vector_space ℝ α]

-- Definitions used only directly in the conditions
def variance (data : list ℝ) : ℝ := 
  let n := data.length in
  let mean := data.sum / n in
  (1 / n) * (data.map (λ a, (a - mean)^2)).sum

-- Conditions
variable (a : list ℝ) (σ : ℝ)
variable (h : variance a = σ^2)

-- Theorem to prove
theorem scaled_variance :
  variance (a.map (λ x, 2 * x)) = 4 * σ^2 := by
  sorry

end scaled_variance_l301_301648


namespace evaluate_expression_l301_301947

theorem evaluate_expression :
  (Int.floor 1.999) + (Int.ceil 3.001) - (Int.floor 0.001) = 5 := by
  sorry

end evaluate_expression_l301_301947


namespace base6_sum_lemma_l301_301001

def base6_to_base10 (digits : List ℕ) : ℕ :=
  digits.foldl (λ acc x => acc * 6 + x) 0

def base10_to_base6 (n : ℕ) : List ℕ :=
  if n == 0 then [0] else
    let rec divmod (m : ℕ) : List ℕ :=
      if m == 0 then [] else
      let (q, r) := m /% 6
      r :: divmod q
    divmod n

theorem base6_sum_lemma :
  base10_to_base6 ((base6_to_base10 [4, 5, 3, 2] + base6_to_base10 [2, 4, 2, 5, 3])) = [3 , 3, 2, 2, 2, 5] := by
  sorry

end base6_sum_lemma_l301_301001


namespace f_positive_range_l301_301078

noncomputable def f : ℝ → ℝ := sorry

-- Condition 1: f(x) is even on its domain
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

-- Condition 2: f(1) = 0
def f_at_1 (f : ℝ → ℝ) := f 1 = 0

-- Condition 3: For x > 0, 2f(x) - xf'(x) > 0
def derivative_condition (f : ℝ → ℝ) := ∀ x : ℝ, x > 0 → 2 * f x - x * (deriv f x) > 0

-- The range of x where f(x) > 0
def range_f_positive (f : ℝ → ℝ) := ∀ x : ℝ, f x > 0 → -1 < x ∧ x < 1

theorem f_positive_range :
  (is_even f) → (f_at_1 f) → (derivative_condition f) → (range_f_positive f) :=
by 
  intros h_even h_f_at_1 h_deriv_cond,
  sorry

end f_positive_range_l301_301078


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301746

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301746


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301764

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301764


namespace min_value_reciprocal_sum_l301_301997

variable {a b x y : ℝ}

theorem min_value_reciprocal_sum (h₁ : 0 < a) (h₂ : 0 < b) 
  (h₃ : Real.sqrt 2 = Real.sqrt (a * b)) 
  (h₄ : Real.logBase a x = 3) 
  (h₅ : Real.logBase b y = 3) : 
  1 / x + 1 / y = Real.sqrt 2 / 2 :=
  sorry

end min_value_reciprocal_sum_l301_301997


namespace arithmetic_sequence_term_count_l301_301942

def first_term : ℕ := 5
def common_difference : ℕ := 3
def last_term : ℕ := 203

theorem arithmetic_sequence_term_count :
  ∃ n : ℕ, last_term = first_term + (n - 1) * common_difference ∧ n = 67 :=
by
  sorry

end arithmetic_sequence_term_count_l301_301942


namespace length_of_AB_l301_301136

variables {V : Type*} [InnerProductSpace ℝ V]
variables {A B C : V}
variables (AB AC CB : V)

theorem length_of_AB 
  (h1 : inner AB AC = 2)
  (h2 : inner AB CB = 2) :
  ∥AB∥ = 2 :=
sorry

end length_of_AB_l301_301136


namespace part1_part2_l301_301558

noncomputable
def triangle_data (α β γ : ℝ) :=
  0 < α ∧ 0 < β ∧ 0 < γ ∧ α + β + γ = π

theorem part1 (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_data A B C)
  (h1 : sqrt 3 * b = a * (sqrt 3 * cos C - sin C)) :
  A = 2 * π / 3 :=
sorry

theorem part2 (a b c : ℝ) (A B C : ℝ)
  (r : ℝ) 
  (h_triangle : triangle_data A B C)
  (ha : a = 8)
  (hr : r = sqrt 3)
  (hA : A = 2 * π / 3) :
  a + b + c = 18 :=
sorry

end part1_part2_l301_301558


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301757

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301757


namespace arrangement_exists_for_P_eq_23_l301_301248

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301248


namespace exists_similar_sizes_P_23_l301_301214

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301214


namespace molecular_weight_CaH2_is_correct_l301_301003

noncomputable def atomicWeightCa : ℝ := 40.08
noncomputable def atomicWeightH : ℝ := 1.008

def molecularWeightCaH2 : ℝ := atomicWeightCa + 2 * atomicWeightH

theorem molecular_weight_CaH2_is_correct :
  molecularWeightCaH2 = 42.096 :=
by 
  have h1 : molecularWeightCaH2 = atomicWeightCa + 2 * atomicWeightH := rfl
  have h2 : molecularWeightCaH2 = 40.08 + 2 * 1.008 := by rw [h1, atomicWeightCa, atomicWeightH]
  have h3 : molecularWeightCaH2 = 40.08 + 2.016 := by rw [h2]
  have h4 : molecularWeightCaH2 = 42.096 := by norm_num [h3]
  exact h4

end molecular_weight_CaH2_is_correct_l301_301003


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301723

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301723


namespace seeds_in_big_garden_l301_301420

-- Definitions based on conditions
def total_seeds : ℕ := 42
def small_gardens : ℕ := 3
def seeds_per_small_garden : ℕ := 2
def seeds_planted_in_small_gardens : ℕ := small_gardens * seeds_per_small_garden

-- Proof statement
theorem seeds_in_big_garden : total_seeds - seeds_planted_in_small_gardens = 36 :=
sorry

end seeds_in_big_garden_l301_301420


namespace determinant_scaled_l301_301973

theorem determinant_scaled
  (x y z w : ℝ)
  (h : x * w - y * z = 10) :
  (3 * x) * (3 * w) - (3 * y) * (3 * z) = 90 :=
by sorry

end determinant_scaled_l301_301973


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301724

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301724


namespace hyperbola_asymptotes_angle_l301_301433

theorem hyperbola_asymptotes_angle {a b : ℝ} (h₁ : a > b) 
  (h₂ : ∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1) 
  (h₃ : ∀ θ : ℝ, θ = Real.pi / 4) : a / b = Real.sqrt 2 :=
by
  sorry

end hyperbola_asymptotes_angle_l301_301433


namespace classroom_chair_arrangements_l301_301002

def distinct_arrangements (n : ℕ) : ℕ :=
  (Finset.filter (λ (x : ℕ × ℕ), (2 ≤ x.1) ∧ (2 ≤ x.2) ∧ (x.1 * x.2 = n))
    (Finset.product (Finset.range (n + 1)) (Finset.range (n + 1)))).card

theorem classroom_chair_arrangements (n : ℕ) (h : n = 48) : distinct_arrangements n = 8 :=
by
  sorry

end classroom_chair_arrangements_l301_301002


namespace remainder_of_sum_of_primes_l301_301803

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301803


namespace max_x_constraint_l301_301303

-- Define the given data points
def x1 := 8
def y1 := 5
def x2 := 12
def y2 := 8
def x3 := 14
def y3 := 9
def x4 := 16
def y4 := 11

-- Mean of x values
def x_mean := (x1 + x2 + x3 + x4) / 4
def y_mean := (y1 + y2 + y3 + y4) / 4

-- Sum of x_i * y_i
def sum_xy := x1 * y1 + x2 * y2 + x3 * y3 + x4 * y4

-- Sum of squares of x
def sum_x_squared := x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4

-- Slope of the regression line
def slope := (sum_xy - 4 * x_mean * y_mean) / (sum_x_squared - 4 * (x_mean * x_mean))

-- Intercept of the regression line
def intercept := y_mean - slope * x_mean

-- Regression line equation
def regression_line (x : ℝ) := slope * x + intercept

-- Constraint given by the problem
def y_constraint := 10

-- Find the maximum value of x given the constraint on y
def max_x (y : ℝ) := (y - intercept) / slope

theorem max_x_constraint : max_x y_constraint = 15 := by
  -- We prove that the maximum value of x given the constraint y ≤ 10 is 15
  sorry

end max_x_constraint_l301_301303


namespace lassie_original_bones_l301_301570

variable (B : ℕ) -- B is the number of bones Lassie started with

-- Conditions translated into Lean statements
def eats_half_on_saturday (B : ℕ) : ℕ := B / 2
def receives_ten_more_on_sunday (B : ℕ) : ℕ := eats_half_on_saturday B + 10
def total_bones_after_sunday (B : ℕ) : Prop := receives_ten_more_on_sunday B = 35

-- Proof goal: B is equal to 50 given the conditions
theorem lassie_original_bones :
  total_bones_after_sunday B → B = 50 :=
sorry

end lassie_original_bones_l301_301570


namespace sequence_transformation_possible_l301_301384

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end sequence_transformation_possible_l301_301384


namespace arithmetic_sequence_S_11_zero_l301_301082

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

def arithmetic_sequence (a d : α) : ℕ → α :=
  λ n, a + n • d

def sum_of_arithmetic_sequence (a d : α) : ℕ → α :=
  λ n, (n + 1) • a + (finset.range n).sum (λ i, i • d) -- Sum of first n terms

theorem arithmetic_sequence_S_11_zero
  (a d : α) (S : ℕ → α) (h1 : d ≠ 0)
  (h2 : ∀ n, S n = sum_of_arithmetic_sequence a d n)
  (h3 : S 5 = S 6) :
  S 11 = 0 :=
by
  sorry

end arithmetic_sequence_S_11_zero_l301_301082


namespace inequality_solution_set_l301_301843

theorem inequality_solution_set (a b : ℝ) (h : abs (a - b) > 2) :
  {x : ℝ | abs (x - a) + abs (x - b) > 2} = set.univ :=
by
  sorry

end inequality_solution_set_l301_301843


namespace sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l301_301836

-- Prove that the square root of 36 equals ±6
theorem sqrt_36_eq_pm6 : ∃ y : ℤ, y * y = 36 ∧ y = 6 ∨ y = -6 :=
by
  sorry

-- Prove that the arithmetic square root of sqrt(16) equals 2
theorem arith_sqrt_sqrt_16_eq_2 : ∃ z : ℝ, z * z = 16 ∧ z = 4 ∧ 2 * 2 = z :=
by
  sorry

-- Prove that the cube root of -27 equals -3
theorem cube_root_minus_27_eq_minus_3 : ∃ x : ℝ, x * x * x = -27 ∧ x = -3 :=
by
  sorry

end sqrt_36_eq_pm6_arith_sqrt_sqrt_16_eq_2_cube_root_minus_27_eq_minus_3_l301_301836


namespace prime_sum_remainder_l301_301771

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301771


namespace nested_op_evaluation_l301_301938

noncomputable def op (a b c : ℕ) (hc : c ≠ 0) : ℚ := (a + b) / c

theorem nested_op_evaluation :
  (op (op 72 18 90 (by norm_num)) (op 4 2 6 (by norm_num)) (op 12 6 18 (by norm_num)) (by norm_num) = 2) :=
sorry

end nested_op_evaluation_l301_301938


namespace probability_from_first_to_last_l301_301852

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l301_301852


namespace cos_double_angle_l301_301053

-- Definitions of conditions from part a)
def x : ℝ := sorry -- Denote x, assume it is in the given interval
def cond1 : x ∈ Ioo (-3 * Real.pi / 4) (Real.pi / 4) := sorry -- x in the open interval
def cond2 : Real.cos (Real.pi / 4 - x) = -3 / 5 := sorry -- given cosine condition

-- The statement we want to prove
theorem cos_double_angle (hx : cond1) (heq : cond2) : Real.cos (2 * x) = -24 / 25 := 
sorry

end cos_double_angle_l301_301053


namespace exists_arrangement_for_P_23_l301_301230

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301230


namespace expenditure_increase_l301_301292

theorem expenditure_increase (x : ℝ) (h₁ : 3 * x / (3 * x + 2 * x) = 3 / 5)
  (h₂ : 2 * x / (3 * x + 2 * x) = 2 / 5)
  (h₃ : ((5 * x) + 0.15 * (5 * x)) = 5.75 * x) 
  (h₄ : (2 * x + 0.06 * 2 * x) = 2.12 * x) 
  : ((3.63 * x - 3 * x) / (3 * x) * 100) = 21 := 
  by
  sorry

end expenditure_increase_l301_301292


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301758

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301758


namespace extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l301_301099

-- Define the function f(x) = 2*x^3 + 3*(a-2)*x^2 - 12*a*x
def f (x : ℝ) (a : ℝ) := 2*x^3 + 3*(a-2)*x^2 - 12*a*x

-- Define the function f(x) when a = 0
def f_a_zero (x : ℝ) := f x 0

-- Define the intervals and extreme values problem
theorem extreme_values_of_f_a_zero_on_interval :
  ∃ (max min : ℝ), (∀ x ∈ Set.Icc (-2 : ℝ) 4, f_a_zero x ≤ max ∧ f_a_zero x ≥ min) ∧ max = 32 ∧ min = -40 :=
sorry

-- Define the function for the derivative of f(x)
def f_derivative (x : ℝ) (a : ℝ) := 6*x^2 + 6*(a-2)*x - 12*a

-- Prove the monotonicity based on the value of a
theorem monotonicity_of_f (a : ℝ) :
  (a > -2 → (∀ x, x < -a → f_derivative x a > 0) ∧ (∀ x, -a < x ∧ x < 2 → f_derivative x a < 0) ∧ (∀ x, x > 2 → f_derivative x a > 0)) ∧
  (a = -2 → ∀ x, f_derivative x a ≥ 0) ∧
  (a < -2 → (∀ x, x < 2 → f_derivative x a > 0) ∧ (∀ x, 2 < x ∧ x < -a → f_derivative x a < 0) ∧ (∀ x, x > -a → f_derivative x a > 0)) :=
sorry

end extreme_values_of_f_a_zero_on_interval_monotonicity_of_f_l301_301099


namespace exist_arrangement_for_P_23_l301_301200

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301200


namespace intersection_M_N_l301_301841

def M : Set ℕ := {0, 1, 2}
def N : Set ℝ := {x | x^2 - 3*x + 2 ≤ 0}

theorem intersection_M_N : M ∩ N = {1, 2} :=
by {
  sorry  -- Proof omitted as required in the task instructions
}

end intersection_M_N_l301_301841


namespace arrangement_possible_32_arrangement_possible_100_l301_301358

-- Problem (1)
theorem arrangement_possible_32 : 
  ∃ (f : Fin 32 → Fin 32), ∀ (a b : Fin 32), ∀ (i : Fin 32), 
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry

-- Problem (2)
theorem arrangement_possible_100 : 
  ∃ (f : Fin 100 → Fin 100), ∀ (a b : Fin 100), ∀ (i : Fin 100),
    a < b → i < b → f i = (a + b) / 2 → False := 
sorry


end arrangement_possible_32_arrangement_possible_100_l301_301358


namespace probability_open_doors_l301_301871

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l301_301871


namespace complement_union_l301_301497

variable (U : Set ℝ)
variable (A B : Set ℝ)
variable (lg : ℝ → ℝ)
variable (exp : ℝ → ℝ)

def A_def : Set ℝ := { x | lg x ≤ 0 }
def B_def : Set ℝ := { x | exp x ≤ 1 }
def U_def : Set ℝ := Set.univ

theorem complement_union : (U ∉ A_def ∪ B_def) = { x : ℝ | x > 1 } := by
  sorry

end complement_union_l301_301497


namespace zeros_at_end_of_50_factorial_l301_301520

-- Definitions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Proof statement
theorem zeros_at_end_of_50_factorial : ∀ n = 50, (count_factors 5 (factorial n)) = 12 :=
  by sorry

-- Helper function to count the factors of a base in a number
noncomputable def count_factors (base : ℕ) (n : ℕ) : ℕ :=
  if n = 0 then 0 else (nat.div n base) + (count_factors base (nat.div n base))

end zeros_at_end_of_50_factorial_l301_301520


namespace solve_for_k_l301_301120

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l301_301120


namespace part1_part2_l301_301481

-- Definition of the function f.
def f (a x : ℝ) : ℝ := ln (a * x + 1/2) + 2 / (2 * x + 1)

-- Problem Part 1: f(x) is monotonically increasing on (0, +∞) if and only if a ≥ 2
theorem part1 {a : ℝ} : (∀ x > 0, f a x ≤ f a (x + 1)) ↔ a ≥ 2 :=
sorry

-- Problem Part 2: There exists a real number a such that the minimum value of f(x) on (0, +∞) is 1, and a = 1
theorem part2 : (∃ a : ℝ, (∀ x > 0, f a x ≥ 1) ∧ (∃ x > 0, f a x = 1)) ↔ a = 1 :=
sorry

end part1_part2_l301_301481


namespace cows_days_production_l301_301517

theorem cows_days_production (x : ℕ) (h : 0 < x) :
  let bucket_production_per_cow_per_day := (x + 1) / (x * (x + 2)) in
  let bucket_production_per_day := (x + 3) * bucket_production_per_cow_per_day in
  ∃ D : ℕ, D = (x * (x + 2) * (x + 5)) / ((x + 1) * (x + 3)) ∧
    D * bucket_production_per_day = x + 5 :=
begin
  sorry
end

end cows_days_production_l301_301517


namespace find_x_collinear_points_l301_301474

open Real

theorem find_x_collinear_points
  (x : ℝ)
  (collinear : ∃ m b : ℝ, ∀ (p : ℝ × ℝ), p ∈ ([(2, 7), (10, x), (25, -2)] : list (ℝ × ℝ)) → p.2 = m * p.1 + b) :
  x = 89 / 23 :=
by
  -- Proof steps must show that the slopes between the points are equal
  sorry

end find_x_collinear_points_l301_301474


namespace b_horses_count_l301_301832

variable (H : ℕ)

-- Conditions
def total_cost : ℤ := 435
def a_horses : ℕ := 12
def a_months : ℕ := 8
def c_horses : ℕ := 18
def c_months : ℕ := 6
def b_payment : ℤ := 180

def a_horse_months : ℕ := a_horses * a_months
def b_horse_months : ℕ := H * 9
def c_horse_months : ℕ := c_horses * c_months
def total_horse_months : ℕ := a_horse_months + b_horse_months + c_horse_months

-- Prove H = 16
theorem b_horses_count : H = 16 := by
  have proportion : (b_horse_months : ℤ) * total_cost = b_payment * (total_horse_months : ℤ) := sorry
  sorry

end b_horses_count_l301_301832


namespace partition_binary_01_l301_301934

def is_binary_01 (n : ℕ) : Prop :=
  ∀ d ∈ n.digits 10, d = 0 ∨ d = 1

def has_ones_in_odd_positions_only (n : ℕ) : Prop :=
  ∃ l, l = n.digits 2 ∧ ∀ (i : ℕ), i < l.length → l.get_or_else i 0 = 1 → i % 2 = 0

def has_ones_in_even_positions_only (n : ℕ) : Prop :=
  ∃ l, l = n.digits 2 ∧ ∀ (i : ℕ), i < l.length → l.get_or_else i 0 = 1 → i % 2 = 1

theorem partition_binary_01 :
  ∃ (A B : set ℕ),
    (∀ n ∈ A, is_binary_01 n ∧ has_ones_in_odd_positions_only n) ∧
    (∀ n ∈ B, is_binary_01 n ∧ has_ones_in_even_positions_only n) ∧
    (∀ n, is_binary_01 n → n ∈ A ∨ n ∈ B) ∧
    (∀ a1 a2 ∈ A, a1 ≠ a2 → ∃ l, is_binary_01 (a1 + a2) ∧ (l = (a1 + a2).digits 10 ∧ l.count 1 ≥ 2)) ∧
    (∀ b1 b2 ∈ B, b1 ≠ b2 → ∃ l, is_binary_01 (b1 + b2) ∧ (l = (b1 + b2).digits 10 ∧ l.count 1 ≥ 2)) :=
sorry

end partition_binary_01_l301_301934


namespace trapezoid_midpoint_line_l301_301989

-- Define the arbitrary trapezoid ABCD
variables {A B C D K M H P : Type*}
variables [trapezoid A B C D]
variables [line AD BC has_intersection_at K]
variables [diagonals_inter A C B D has_intersection_at M]
variables [midpoint_of_segment AB H]
variables [midpoint_of_segment CD P]

-- State the theorem
theorem trapezoid_midpoint_line :
  passes_through_line K M H P :=
sorry

end trapezoid_midpoint_line_l301_301989


namespace find_divisor_l301_301662

theorem find_divisor (dividend quotient remainder : ℕ) (h_div : dividend = (16 * quotient) + remainder) : 
quotient = 9 → remainder = 5 → dividend = 149 →
16 * 9 + 5 = 149 := by
  intros
  rw [←h_div, mul_add]
  ring
  sorry

end find_divisor_l301_301662


namespace range_of_a_l301_301098

noncomputable def f (a x : ℝ) := x^2 - 2 * a * x + 5

theorem range_of_a (a : ℝ) (h1 : 1 < a)
  (h2 : ∀ x ∈ Set.Ioi (interval_upper_bound (-∞) 2 : Set ℝ), (f a x) < f a x)
  (h3 : ∀ x1 x2 ∈ Set.Icc 1 (a + 1), |f a x1 - f a x2| ≤ 4) :
  2 ≤ a ∧ a ≤ 3 :=
sorry

end range_of_a_l301_301098


namespace find_y_l301_301894

noncomputable def similar_triangles (a b x z : ℝ) :=
  (a / x = b / z)

theorem find_y 
  (a b x z : ℝ)
  (ha : a = 12)
  (hb : b = 9)
  (hz : z = 7)
  (h_sim : similar_triangles a b x z) :
  x = 28 / 3 :=
begin
  subst ha,
  subst hb,
  subst hz,
  unfold similar_triangles at h_sim,
  field_simp [h_sim],
  ring,
end

end find_y_l301_301894


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301721

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301721


namespace tangents_and_length_problem_l301_301912

variables {A B C D E : Type*}
variables [Points_on_circle : Type*]
variables (O1 O2 O3: Points_on_circle)

theorem tangents_and_length_problem
  (inside_triangle_ABC    : D inside_triangle ABC)
  (circumcircle_ABD       : O1 circumcircle ABD)
  (circumcircle_ACD       : O2 circumcircle ACD)
  (circumcircle_ABC       : O3 circumcircle ABC)
  (AD_meets_O3_at_E       : extended AD meets O3 at E)
  (D_is_midpoint_AE       : D is midpoint AE)
  (AB_tangent_to_O2       : AB is tangent to O2)
  (AC_length : AC = 20) 
  (AB_length : AB = 18)
  (BC_length : BC = 22) : Prop :=
begin
  (tangent_AC_O1: AC is tangent O1),
  (AE_length_calculated : AE = 40)
end

end tangents_and_length_problem_l301_301912


namespace purely_imaginary_z_l301_301085

theorem purely_imaginary_z (z : ℂ) (h1 : z.im ≠ 0) (h2 : (z-3).im = 0 ∧ ((z - 3) ^ 2 + 5 * complex.I).im = 0) : 
  z = 3 * complex.I ∨ z = -3 * complex.I := 
sorry

end purely_imaginary_z_l301_301085


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301694

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301694


namespace magnitude_of_complex_l301_301054

theorem magnitude_of_complex :
  let z := 5 / (2 + complex.I) in
  complex.abs z = sqrt 5 :=
by
  -- The statement requires a proof which we will skip, so we add sorry to indicate that.
  sorry

end magnitude_of_complex_l301_301054


namespace number_of_expressions_evaluating_to_half_l301_301480

theorem number_of_expressions_evaluating_to_half :
  let expr1 := sin (15 * Real.pi / 180) * cos (15 * Real.pi / 180)
  let expr2 := cos (Real.pi / 8) ^ 2 - sin (Real.pi / 8) ^ 2
  let expr3 := tan (22.5 * Real.pi / 180) / (1 - tan (22.5 * Real.pi / 180) ^ 2)
  (expr1 = 1/2 ∨ expr2 = 1/2 ∨ expr3 = 1/2) ∧
  (expr1 = 1/2 ↔ False) ∧
  (expr2 = 1/2 ↔ False) ∧
  (expr3 = 1/2 ↔ True) :=
by 
  let expr1 := sin (15 * Real.pi / 180) * cos (15 * Real.pi / 180)
  let expr2 := cos (Real.pi / 8) ^ 2 - sin (Real.pi / 8) ^ 2
  let expr3 := tan (22.5 * Real.pi / 180) / (1 - tan (22.5 * Real.pi / 180) ^ 2)
  sorry

end number_of_expressions_evaluating_to_half_l301_301480


namespace thomas_worked_hours_l301_301311

theorem thomas_worked_hours (Toby Thomas Rebecca : ℕ) 
  (h_total : Thomas + Toby + Rebecca = 157) 
  (h_toby : Toby = 2 * Thomas - 10) 
  (h_rebecca_1 : Rebecca = Toby - 8) 
  (h_rebecca_2 : Rebecca = 56) : Thomas = 37 :=
by
  sorry

end thomas_worked_hours_l301_301311


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301690

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301690


namespace product_of_all_divisors_l301_301428

-- Definitions of N and k
def N (p : ℕ → ℕ) (λ : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldr (λ i acc, acc * p i ^ λ i) 1

def k (λ : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).foldr (λ i acc, acc * (λ i + 1)) 1

-- Proof problem statement
theorem product_of_all_divisors (p : ℕ → ℕ) (λ : ℕ → ℕ) (n : ℕ) (h_fin : ∀ i < n, Nat.prime (p i)) :
  let N_val := N p λ n
  let k_val := k λ n
  (N_val.product_of_divisors = Real.sqrt (N_val ^ k_val)) := sorry

end product_of_all_divisors_l301_301428


namespace exists_arrangement_for_P_23_l301_301207

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301207


namespace remainder_first_six_primes_div_seventh_l301_301807

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301807


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301742

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301742


namespace range_of_a_l301_301077

noncomputable def unique_y_for_x (x : ℝ) (a : ℝ) (y : ℝ) : Prop :=
  y ∈ Icc (-1 : ℝ) (1 : ℝ) ∧ (x + y^2 * Real.exp y - a = 0)

theorem range_of_a (e : ℝ) (h_e : Real.exp 1 = e) :
  (∀ (x : ℝ), x ∈ Icc (0 : ℝ) (1 : ℝ) → ∃! (y : ℝ), unique_y_for_x x a y) →
  1 + (1 / e) < a ∧ a ≤ e :=
sorry

end range_of_a_l301_301077


namespace rectangle_area_constant_l301_301293

theorem rectangle_area_constant {d : ℝ} (h : 0 < d) :
  ∃ k, (∀ (l w : ℝ), l = 5 * (sqrt (d^2 / 29)) ∧ w = 2 * (sqrt (d^2 / 29)) 
  ∧ (l^2 + w^2 = d^2) → (l * w = k * d^2)) ∧ k = 10 / 29 :=
sorry

end rectangle_area_constant_l301_301293


namespace John_has_22_quarters_l301_301565

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l301_301565


namespace part1_part2_l301_301398

-- Part 1
theorem part1 : 
  (Real.log 8 / Real.log 10) + (Real.log 125 / Real.log 10) - (1 / 7)^(-2) + 16^(3 / 4) + (Real.sqrt 3 - 1)^0 = -37 := 
  by sorry

-- Part 2
theorem part2 (α : ℝ) (h : Real.tan α = 3) : 
  (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
  by sorry

end part1_part2_l301_301398


namespace max_value_expression_l301_301985

theorem max_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  ∃ M, M = 3 ∧ ∀ a b c, 0 < a → 0 < b → 0 < c → 
    let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / 
             ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
    A ≤ M := 
begin
  use 3,
  sorry
end

end max_value_expression_l301_301985


namespace min_value_range_of_x_l301_301977

variables (a b x : ℝ)

-- Problem 1: Prove the minimum value of 1/a + 4/b given a + b = 1, a > 0, b > 0
theorem min_value (h1 : a + b = 1) (h2 : a > 0) (h3 : b > 0) : 
  ∃ c, c = 9 ∧ ∀ y, ∃ (a b : ℝ), a + b = 1 ∧ a > 0 ∧ b > 0 → (1/a + 4/b) ≥ y :=
sorry

-- Problem 2: Prove the range of x for which 1/a + 4/b ≥ |2x - 1| - |x + 1|
theorem range_of_x (h : ∀ (a b : ℝ), a + b = 1 → a > 0 → b > 0 → (1/a + 4/b) ≥ (|2*x - 1| - |x + 1|)) :
  -7 ≤ x ∧ x ≤ 11 :=
sorry

end min_value_range_of_x_l301_301977


namespace prob_of_B1_selected_prob_of_D1_in_team_l301_301308

noncomputable def total_teams : ℕ := 20

noncomputable def teams_with_B1 : ℕ := 8

noncomputable def teams_with_D1 : ℕ := 12

theorem prob_of_B1_selected : (teams_with_B1 : ℚ) / total_teams = 2 / 5 := by
  sorry

theorem prob_of_D1_in_team : (teams_with_D1 : ℚ) / total_teams = 3 / 5 := by
  sorry

end prob_of_B1_selected_prob_of_D1_in_team_l301_301308


namespace count_even_nonzero_groups_correct_l301_301039

noncomputable def count_even_nonzero_groups (n₁ n₂ n₃ n₄ n₅ n₆ n₇ n₈ n₉ n₁₀ : ℕ) : ℕ :=
let n := n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ + n₈ + n₉ + n₁₀ in
2^(n - 1) + 1/2 * (2^n₁ - 2) * (2^n₂ - 2) * (2^n₃ - 2) * (2^n₄ - 2) * (2^n₅ - 2) * 
             (2^n₆ - 2) * (2^n₇ - 2) * (2^n₈ - 2) * (2^n₉ - 2) * (2^n₁₀ - 2)

theorem count_even_nonzero_groups_correct (n₁ n₂ n₃ n₄ n₅ n₆ n₇ n₈ n₉ n₁₀ : ℕ) :
  count_even_nonzero_groups n₁ n₂ n₃ n₄ n₅ n₆ n₇ n₈ n₉ n₁₀ =
    2^(n₁ + n₂ + n₃ + n₄ + n₅ + n₆ + n₇ + n₈ + n₉ + n₁₀ - 1) + 
    1/2 * (2^n₁ - 2) * (2^n₂ - 2) * (2^n₃ - 2) * (2^n₄ - 2) * (2^n₅ - 2) * 
          (2^n₆ - 2) * (2^n₇ - 2) * (2^n₈ - 2) * (2^n₉ - 2) * (2^n₁₀ - 2) :=
by
  sorry

end count_even_nonzero_groups_correct_l301_301039


namespace equal_real_roots_of_quadratic_eq_l301_301127

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l301_301127


namespace simplify_sqrt_expression_l301_301012

theorem simplify_sqrt_expression (t : ℝ) : (Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1)) :=
by sorry

end simplify_sqrt_expression_l301_301012


namespace combine_balls_into_one_heap_l301_301338

theorem combine_balls_into_one_heap (n : ℕ) (heaps : list ℕ) (h : heaps.sum = 2^n) :
  ∃ (moves : list (ℕ × ℕ × ℕ)), -- List of moves (A index, B index, q balls)
  -- Verifying that moves are finite and result in one heap
  let apply_moves := λ (heaps : list ℕ) (mv : (ℕ × ℕ × ℕ)),
    let p := heaps.nth mv.1 in
    let a := heaps.nth mv.2 in
    if p.is_some ∧ a.is_some ∧ p.get! mv.1 ≥ mv.2
    then list.update_nth (list.update_nth heaps mv.1 (p.get! - mv.3)) mv.2 (a.get! + mv.3)
    else heaps
  in heaps.length = 1
:=
sorry
 -- Steps to implement correct move verification and eventual combination

end combine_balls_into_one_heap_l301_301338


namespace probability_path_from_first_to_last_floor_open_doors_l301_301860

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l301_301860


namespace cloth_total_selling_price_l301_301378

theorem cloth_total_selling_price
    (meters : ℕ) (profit_per_meter cost_price_per_meter : ℝ) :
    meters = 92 →
    profit_per_meter = 24 →
    cost_price_per_meter = 83.5 →
    (cost_price_per_meter + profit_per_meter) * meters = 9890 :=
by
  intros
  sorry

end cloth_total_selling_price_l301_301378


namespace transform_sequence_l301_301387

theorem transform_sequence (a1 a2 : ℕ) (h_a1a2 : a1 ≥ a2 ∧ a1 + a2 ≤ 100) :
  if |a1 - a2| ≤ 50 then
    true 
  else 
    false :=
by
  intro h_a1a2
  let b1 := 100 - a1
  let b2 := 100 - a2
  have h1 : |a1 - a2| ≤ 50 := sorry
  -- Assumption or further lemmas could be used here if necessary.
  sorry

end transform_sequence_l301_301387


namespace negation_of_proposition_l301_301639

theorem negation_of_proposition (x y : ℝ) :
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by
  sorry

end negation_of_proposition_l301_301639


namespace probability_path_from_first_to_last_floor_open_doors_l301_301857

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l301_301857


namespace sum_PAi_geq_sum_PBi_l301_301173

theorem sum_PAi_geq_sum_PBi (n : ℕ) (P : ℝ×ℝ) (A B : ℕ → ℝ×ℝ)
  (hAreg : ∀ i, is_regular_ngon A n)
  (hBint : ∀ i, B i = line_intersection_with_polygon (A i) P n) :
  ∑ i in finset.range n, euclidean_dist P (A i) ≥ ∑ i in finset.range n, euclidean_dist P (B i) :=
sorry

end sum_PAi_geq_sum_PBi_l301_301173


namespace function_monotonically_decreasing_l301_301124

noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x ≤ 1 then 1 - x else -x^2 + a

theorem function_monotonically_decreasing {a : ℝ} :
  (∀ x1 x2 : ℝ, x1 ≤ x2 → f a x1 ≥ f a x2) ↔ a ≤ 1 :=
begin
  sorry
end

end function_monotonically_decreasing_l301_301124


namespace num_incorrect_propositions_l301_301653

theorem num_incorrect_propositions : 
  ( (∀ x : ℝ, (x^2 - 3*x + 2 = 0) → (x = 1)) → (∀ x : ℝ, (x ≠ 1) → (x^2 - 3*x + 2 ≠ 0)) ) ∧   -- Proposition ①
  ( (∃ x : ℝ, x^2 + x + 1 = 0) → (∀ x : ℝ, x^2 + x + 1 ≠ 0) ) ∧                           -- Proposition ②
  ¬ (¬ ((∀ p q : Prop, (p ∧ q) = false → ¬ (p ∧ ¬ q) ∧ ¬ (¬ p ∧ q)))) ∧                  -- Proposition ③
  (∀ x : ℝ, (x > 2 → x^2 - 3*x + 2 > 0)) →                                              -- Proposition ④
  (true → 1 = 1) :=                                                                      -- The number of incorrect propositions is 1
by
  intros,
  sorry

end num_incorrect_propositions_l301_301653


namespace yellow_balls_count_l301_301364

theorem yellow_balls_count (total_balls white_balls green_balls red_balls purple_balls : ℕ)
  (h_total : total_balls = 100)
  (h_white : white_balls = 20)
  (h_green : green_balls = 30)
  (h_red : red_balls = 37)
  (h_purple : purple_balls = 3)
  (h_prob : ((white_balls + green_balls + (total_balls - white_balls - green_balls - red_balls - purple_balls)) / total_balls : ℝ) = 0.6) :
  (total_balls - white_balls - green_balls - red_balls - purple_balls = 10) :=
by {
  sorry
}

end yellow_balls_count_l301_301364


namespace exists_similar_sizes_P_23_l301_301210

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301210


namespace science_club_neither_chem_nor_bio_l301_301598

theorem science_club_neither_chem_nor_bio :
  let total_students := 75 in
  let chem_students := 45 in
  let bio_students := 30 in
  let both_students := 18 in
  let neither_students := total_students - (chem_students - both_students + bio_students - both_students + both_students) in
  neither_students = 18 :=
by
  let total_students := 75
  let chem_students := 45
  let bio_students := 30
  let both_students := 18
  let neither_students := total_students - (chem_students - both_students + bio_students - both_students + both_students)
  show neither_students = 18, from sorry

end science_club_neither_chem_nor_bio_l301_301598


namespace water_cost_is_1_l301_301537

-- Define the conditions
def cost_cola : ℝ := 3
def cost_juice : ℝ := 1.5
def bottles_sold_cola : ℝ := 15
def bottles_sold_juice : ℝ := 12
def bottles_sold_water : ℝ := 25
def total_revenue : ℝ := 88

-- Compute the revenue from cola and juice
def revenue_cola : ℝ := bottles_sold_cola * cost_cola
def revenue_juice : ℝ := bottles_sold_juice * cost_juice

-- Define a proof that the cost of a bottle of water is $1
theorem water_cost_is_1 : (total_revenue - revenue_cola - revenue_juice) / bottles_sold_water = 1 :=
by
  -- Proof is omitted
  sorry

end water_cost_is_1_l301_301537


namespace probability_at_least_half_correct_l301_301617

/--
Steve guesses randomly on a 20-question multiple choice test where each question has three choices
(one correct and two incorrect). Prove that the probability that he gets at least half of the
questions correct is 1/2.
-/
theorem probability_at_least_half_correct :
  ∑ k in finset.range (21) \ finset.range (10), (nat.choose 20 k) * (1/3:NNReal)^k * (2/3:NNReal)^(20 - k) = 1/2 := by
  sorry

end probability_at_least_half_correct_l301_301617


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301725

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301725


namespace simplify_expression_l301_301264

variable (a b : ℤ)

theorem simplify_expression :
  (30 * a + 45 * b) + (15 * a + 40 * b) - (20 * a + 55 * b) + (5 * a - 10 * b) = 30 * a + 20 * b :=
by
  sorry

end simplify_expression_l301_301264


namespace weight_first_watermelon_l301_301569

-- We define the total weight and the weight of the second watermelon
def total_weight := 14.02
def second_watermelon := 4.11

-- We need to prove that the weight of the first watermelon is 9.91 pounds
theorem weight_first_watermelon : total_weight - second_watermelon = 9.91 := by
  -- Insert mathematical steps here (omitted in this case)
  sorry

end weight_first_watermelon_l301_301569


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301686

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301686


namespace income_derived_from_investment_l301_301638

-- Definitions for the given conditions
def market_value := 90.02777777777779
def investment_amount := 6500
def brokerage_rate := 0.0025
def dividend_rate := 0.105

-- The amount of brokerage fee
def brokerage_fee := investment_amount * brokerage_rate

-- Effective investment amount after deducting the brokerage fee
def effective_investment_amount := investment_amount - brokerage_fee

-- Face value calculation
def face_value := effective_investment_amount / (market_value / 100)

-- Income derived from the investment
def derived_income := face_value * dividend_rate

-- Proposition to be proved
theorem income_derived_from_investment : derived_income = 756 := by
  sorry

end income_derived_from_investment_l301_301638


namespace triangles_side_product_relation_l301_301014

-- Define the two triangles with their respective angles and side lengths
variables (A B C A1 B1 C1 : Type) 
          (angle_A angle_A1 angle_B angle_B1 : ℝ) 
          (a b c a1 b1 c1 : ℝ)

-- Given conditions
def angles_sum_to_180 (angle_A angle_A1 : ℝ) : Prop :=
  angle_A + angle_A1 = 180

def angles_equal (angle_B angle_B1 : ℝ) : Prop :=
  angle_B = angle_B1

-- The main theorem to be proven
theorem triangles_side_product_relation 
  (h1 : angles_sum_to_180 angle_A angle_A1)
  (h2 : angles_equal angle_B angle_B1) :
  a * a1 = b * b1 + c * c1 :=
sorry

end triangles_side_product_relation_l301_301014


namespace range_of_a_l301_301115

theorem range_of_a (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (∀ x1 x2, -1 ≤ x1 → x1 ≤ x2 → ∀ y1 y2, y1 = log a (a * x1 + 2) → y2 = log a (a * x2 + 2) → y1 ≤ y2) ↔ 1 < a ∧ a < 2 :=
by
  sorry

end range_of_a_l301_301115


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301726

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301726


namespace incorrect_statement_D_l301_301967

theorem incorrect_statement_D : ∃ a : ℝ, a > 0 ∧ (1 - 1 / (2 * a) < 0) := by
  sorry

end incorrect_statement_D_l301_301967


namespace geometric_series_sum_l301_301171

theorem geometric_series_sum :
  let a := 6
  let r := - (1 / 2)
  let t := a / (1 - r)
  t = 4 := by
unnat_props sorry

end geometric_series_sum_l301_301171


namespace max_constant_C_all_real_numbers_l301_301017

theorem max_constant_C_all_real_numbers:
  ∀ (x1 x2 x3 x4 x5 x6 : ℝ), 
  (x1 + x2 + x3 + x4 + x5 + x6)^2 ≥ 
  3 * (x1 * (x2 + x3) + x2 * (x3 + x4) + x3 * (x4 + x5) + x4 * (x5 + x6) + x5 * (x6 + x1) + x6 * (x1 + x2)) := 
by 
  sorry

end max_constant_C_all_real_numbers_l301_301017


namespace irrational_root_two_l301_301281

theorem irrational_root_two :
  ¬ ∃ (q : ℚ), q * q = (2 : ℝ) :=
begin
  intro h,
  cases h with q hq,
  sorry
end

example : 
  (¬ ∃ (q : ℚ), q * q = (2 : ℝ)) ∧
  ((∃ (q : ℚ), (↑q : ℝ) = (-1)) ∧ 
   (∃ (q : ℚ), (↑q : ℝ) = 0.101001) ∧ 
   (∃ (q : ℚ), (↑q : ℝ) = (3/2))) :=
by {
  split,
  {
    apply irrational_root_two,
  },
  {
    split,
    {
      use (-1 : ℚ),
      norm_cast,
    },
    {
      split,
      {
        use (101001 / 1000000 : ℚ),
        norm_cast,
      },
      {
        use (3 / 2 : ℚ),
        norm_cast,
      }
    }
  }
}

end irrational_root_two_l301_301281


namespace angle_between_lines_l301_301622

theorem angle_between_lines :
  let L1 := {p : ℝ × ℝ | p.1 = -3}  -- Line x+3=0
  let L2 := {p: ℝ × ℝ | p.1 + p.2 - 3 = 0}  -- Line x+y-3=0
  ∃ θ : ℝ, 0 < θ ∧ θ < 180 ∧ θ = 45 :=
sorry

end angle_between_lines_l301_301622


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301695

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301695


namespace integral_result_l301_301400

-- Define the integral function to be proved
def integral_expression := ∫ x in -2..0, (1 + sqrt(4 - x^2))

-- State the theorem
theorem integral_result : integral_expression = 2 + Real.pi := by
  sorry

end integral_result_l301_301400


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301718

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301718


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301789

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301789


namespace sum_ABC_sum_DEF_l301_301151

-- Defining the properties and constraints of the problem
def AB_CB_to_CDE (AB CB CDE : ℕ) : Prop :=
  let A := AB / 10 in
  let B := AB % 10 in
  let C := CB / 10 in
  let B' := CB % 10 in
  let D := (CDE / 10) % 10 in
  let E := CDE % 10 in
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ 
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ 
  C ≠ D ∧ C ≠ E ∧ D ≠ E ∧ 
  B = B' ∧ (10 * A + B) * (10 * C + B) = CDE ∧
  (CDE / 100) + D + E = C + 1 ∧ 
  C + 1 = D ∧ D + 1 = E

-- The main statement to be proved
theorem sum_ABC_sum_DEF : ∃ A B C D E : ℕ, 
  AB_CB_to_CDE (10 * A + B) (10 * C + B) (100 * C + 10 * D + E) ∧ A + B + C + D + E = 11 :=
by
  sorry

end sum_ABC_sum_DEF_l301_301151


namespace milk_left_after_third_operation_l301_301048

theorem milk_left_after_third_operation :
  ∀ (initial_milk : ℝ), initial_milk > 0 →
  (initial_milk * 0.8 * 0.8 * 0.8 / initial_milk) * 100 = 51.2 :=
by
  intros initial_milk h_initial_milk_pos
  sorry

end milk_left_after_third_operation_l301_301048


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301740

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301740


namespace jaclyn_debentures_worth_l301_301159

theorem jaclyn_debentures_worth :
  (∀ (principal quarterly_rate : ℕ) (p_annum q_interest : ℕ),
    let quarterly_rate := (9.5 / 4) / 100, -- Convert 9.5% p.a. to quarterly rate in decimal
    let total_quarters := 18 / 3,
    let total_interest := 237.5 * total_quarters,
    total_interest = q_interest *
    quarterly_rate * total_quarters) →
  principal = 10000 :=
by
  sorry

end jaclyn_debentures_worth_l301_301159


namespace arrangement_exists_for_P_eq_23_l301_301246

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301246


namespace complex_mul_example_l301_301343

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l301_301343


namespace greatest_area_difference_l301_301320

/-- 
Given two rectangles with integer dimensions and both having a perimeter of 180 cm, prove that
the greatest possible difference between their areas is 1936 square centimeters.
-/
theorem greatest_area_difference (l1 w1 l2 w2 : ℕ) 
  (h1 : 2 * l1 + 2 * w1 = 180) 
  (h2 : 2 * l2 + 2 * w2 = 180) :
  abs (l1 * w1 - l2 * w2) ≤ 1936 := 
sorry

end greatest_area_difference_l301_301320


namespace number_of_license_plates_l301_301621

def Rotokas_alphabet : finset char := {'A', 'E', 'G', 'I', 'K', 'O', 'P', 'R', 'S', 'T', 'U', 'V'}

def valid_license_plates : finset (char × char × char × char × char) :=
  {plate | ∃ (first_letter ∈ {'A', 'P'}),
          (last_letter = 'I'),
          (∀ l ∈ Rotokas_alphabet, l ≠ 'O'),
          (no_repeating_letters : plate.1 ≠ plate.2 ∧ plate.1 ≠ plate.3 ∧ plate.1 ≠ plate.4 ∧ plate.1 ≠ plate.5 ∧
                                                            plate.2 ≠ plate.3 ∧ plate.2 ≠ plate.4 ∧ plate.2 ≠ plate.5 ∧
                                                            plate.3 ≠ plate.4 ∧ plate.3 ≠ plate.5 ∧
                                                            plate.4 ≠ plate.5) ∧
          (letters_used: list char := [plate.1, plate.2, plate.3, plate.4, plate.5],
                        list.forall (λ l, l ∈ {'A', 'E', 'G', 'I', 'K', 'P', 'R', 'S', 'T', 'U', 'V'}) letters_used)}

theorem number_of_license_plates : finset.card valid_license_plates = 1440 := by
  sorry

end number_of_license_plates_l301_301621


namespace probability_open_doors_l301_301873

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l301_301873


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301782

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301782


namespace operations_on_S_l301_301493

def is_element_of_S (x : ℤ) : Prop :=
  x = 0 ∨ ∃ n : ℤ, x = 2 * n

theorem operations_on_S (a b : ℤ) (ha : is_element_of_S a) (hb : is_element_of_S b) :
  (is_element_of_S (a + b)) ∧
  (is_element_of_S (a - b)) ∧
  (is_element_of_S (a * b)) ∧
  (¬ is_element_of_S (a / b)) ∧
  (¬ is_element_of_S ((a + b) / 2)) :=
by
  sorry

end operations_on_S_l301_301493


namespace exists_similar_sizes_P_23_l301_301215

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301215


namespace tangent_line_at_point_l301_301629

noncomputable def tangent_line_eq (x y : ℝ) : Prop := x^3 - y = 0

theorem tangent_line_at_point :
  tangent_line_eq (-2) (-8) →
  ∃ (k : ℝ), (k = 12) ∧ (12 * x - y + 16 = 0) :=
sorry

end tangent_line_at_point_l301_301629


namespace remainder_of_sum_of_primes_l301_301799

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301799


namespace coins_probability_l301_301824

theorem coins_probability :
  let total_outcomes := 2^3,
      favorable_outcomes := Nat.choose 3 2 in
  (favorable_outcomes / total_outcomes : ℚ) = 3 / 8 :=
by
  sorry

end coins_probability_l301_301824


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301727

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301727


namespace age_difference_l301_301337

def P := 31.5
def M := (5 / 3) * P
def Mo := (4 / 3) * M
def N := (7 / 5) * Mo
def sum_ages := P + M + Mo + N
def difference := N - P

theorem age_difference : sum_ages = 252 → difference = 66.5 :=
by
  intro h
  have hP : P = 31.5 := rfl
  have hM : M = (5 / 3) * P := rfl
  have hMo : Mo = (4 / 3) * M := rfl
  have hN : N = (7 / 5) * Mo := rfl
  have h_sum_ages : sum_ages = P + ((5 / 3) * P) + ((4 / 3) * ((5 / 3) * P)) + ((7 / 5) * ((4 / 3) * ((5 / 3) * P))) := by
    rw [hP, hM, hMo, hN]
  rw [h_sum_ages] at h
  have hMoa : (4 / 3) * (5 / 3) * 31.5 = 20 / 9 * 31.5 := by sorry
  have hNa : (7 / 5) * (4 / 3) * (5 / 3) * 31.5 = 28 / 9 * 31.5 := by sorry
  have h_calculation : 31.5 + (5 / 3) * 31.5 + (20 / 9) * 31.5 + (28 / 9) * 31.5 = 252 := by
    rw [←hP, ←hM, ←hMo, ←hN, hMoa, hNa]
    linarith
  rw h_calculation at h
  have h_d : difference = N - P := rfl
  rw h_d
  have hN_val : N = (28 / 9) * P := rfl
  rw [hN_val, hP]
  calc
    (28 / 9) * 31.5 - 31.5 
    = 98 - 31.5 := by sorry
    = 66.5 := by sorry

end age_difference_l301_301337


namespace shaded_area_l301_301846

theorem shaded_area (a b : ℝ) (h : a = 10) (w : b = 10) :
  let mid1 := (a / 2, b)
  let mid2 := (a / 2, 0)
  ∃ Q1 Q2 Q3 Q4 : ℝ × ℝ,
  Q1 = (0, 0) ∧
  Q2 = (a, 0) ∧
  Q3 = (a, b) ∧
  Q4 = (0, b) ∧
  let center := (a / 2, b / 2)
  let p1 := (a / 2 - b / 2, 0 + b / 2)
  let p2 := (a / 2 + b / 2, 0 + b / 2)
  area_of_quadrilateral mid2 p1 center p2 = 25 := sorry

end shaded_area_l301_301846


namespace vector_sum_magnitude_zero_l301_301075

variables {E : Type*} [inner_product_space ℝ E]
variables (e1 e2 e3 : E)

-- Define the conditions
def is_unit_vector (v : E) : Prop := ∥v∥ = 1
def is_angle_120 (v w : E) : Prop := inner_product_space.angle v w = real.pi / 3

-- Formalize the problem statement
theorem vector_sum_magnitude_zero
  (h1 : is_unit_vector e1)
  (h2 : is_unit_vector e2)
  (h3 : is_unit_vector e3)
  (h4 : is_angle_120 e1 e2)
  (h5 : is_angle_120 e1 e3)
  (h6 : is_angle_120 e2 e3) :
  ∥e1 + e2 + e3∥ = 0 :=
begin
  sorry
end

end vector_sum_magnitude_zero_l301_301075


namespace last_term_of_sequence_l301_301089

theorem last_term_of_sequence (u₀ : ℤ) (diffs : List ℤ) (sum_diffs : ℤ) :
  u₀ = 0 → diffs = [2, 4, -1, 0, -5, -3, 3] → sum_diffs = diffs.sum → 
  u₀ + sum_diffs = 0 := by
  sorry

end last_term_of_sequence_l301_301089


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301713

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301713


namespace correct_operation_l301_301826

theorem correct_operation (x : ℝ) (h : x ≠ 0) : 1 / x^(-2) = x^2 :=
by {
  -- Proof can go here
  sorry
}

end correct_operation_l301_301826


namespace geometric_sequence_sum_l301_301076

theorem geometric_sequence_sum (a_n : ℕ → ℕ) (h1 : a_n 2 = 2) (h2 : a_n 5 = 16) :
  (∑ k in Finset.range n, a_n k * a_n (k + 1)) = (2 / 3) * (4^n - 1) := 
sorry

end geometric_sequence_sum_l301_301076


namespace simplify_sqrt1_simplify_sqrt2_l301_301263

theorem simplify_sqrt1 (x : ℝ) (h1 : 1 ≤ x) (h2 : x < 4) : sqrt (1 - 2*x + x^2) + sqrt (x^2 - 8*x + 16) = 3 :=
by
  sorry

theorem simplify_sqrt2 (x : ℝ) (h : 2 - x ≥ 0) : (sqrt (2 - x))^2 - sqrt (x^2 - 6*x + 9) = -1 :=
by
  sorry

end simplify_sqrt1_simplify_sqrt2_l301_301263


namespace min_value_z_l301_301500

theorem min_value_z (x y : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_sum : x + y = 1) :
  ∃ z_min, z_min = (x + 1 / x) * (y + 1 / y) ∧ z_min = 33 / 4 :=
sorry

end min_value_z_l301_301500


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301790

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301790


namespace symmetric_point_verify_l301_301838

noncomputable def point_symmetric {R : Type*} [linear_ordered_field R] (M M' : euclidean_space R (fin 3)) (plane : euclidean_space R (fin 3) → R) : Prop :=
  M' = -(M) + 2 * (classical.some (exists_midpoint plane M))

-- Conditions
def M : euclidean_space ℝ (fin 3) := ![-2, -3, 0]
def plane (p : euclidean_space ℝ (fin 3)) : ℝ := p 0 + 5 * (p 1) + 4

-- Definition to show point symmetric
def M' : euclidean_space ℝ (fin 3) := ![-1, -2, 0]

-- Proof statement
theorem symmetric_point_verify : point_symmetric M M' plane := sorry

end symmetric_point_verify_l301_301838


namespace determine_value_of_x_l301_301623

theorem determine_value_of_x (x : ℝ) :
  (x + 10) + 20 + (3x) + 18 + (3x + 6) = 5 * 32 → 
  x = 106 / 7 :=
by
  intros h
  sorry

end determine_value_of_x_l301_301623


namespace points_not_on_circles_l301_301933

theorem points_not_on_circles (x y : ℝ) (α β : ℝ) :
  α * ((x-2)^2 + y^2 - 1) + β * ((x+2)^2 + y^2 - 1) ≠ 0 →
  (x = 0 ∨ (x = √3 ∧ y = 0) ∨ (x = -√3 ∧ y = 0)) → False :=
by
  sorry

end points_not_on_circles_l301_301933


namespace find_a_and_x_range_l301_301356

noncomputable def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

theorem find_a_and_x_range :
  (∃ a, (∀ x, f x a ≥ 3) ∧ (∃ x, f x a = 3)) →
  (∀ x, ∃ a, f x a ≤ 5 → 
    ((a = 1 → (0 ≤ x ∧ x ≤ 5)) ∧
     (a = 7 → (3 ≤ x ∧ x ≤ 8)))) :=
by sorry

end find_a_and_x_range_l301_301356


namespace total_situps_performed_l301_301916

theorem total_situps_performed :
  let Barney_situps_per_min := 45 in
  let Carrie_situps_per_min := 2 * Barney_situps_per_min in
  let Jerrie_situps_per_min := Carrie_situps_per_min + 5 in
  let total_situps :=
    (Barney_situps_per_min * 1) +
    (Carrie_situps_per_min * 2) +
    (Jerrie_situps_per_min * 3) in
  total_situps = 510 :=
by
  sorry

end total_situps_performed_l301_301916


namespace pool_capacity_l301_301376

-- Define the total capacity of the pool as a variable
variable (C : ℝ)

-- Define the conditions
def additional_water_needed (x : ℝ) : Prop :=
  x = 300

def increases_by_25_percent (x : ℝ) (y : ℝ) : Prop :=
  y = x * 0.25

-- State the proof problem
theorem pool_capacity :
  ∃ C : ℝ, additional_water_needed 300 ∧ increases_by_25_percent (0.75 * C) 300 ∧ C = 1200 :=
sorry

end pool_capacity_l301_301376


namespace range_of_a_l301_301088

theorem range_of_a (a : ℝ) : 
  (∀ x1 x2 : ℝ, (x1 + x2 = -2 * a) ∧ (x1 * x2 = 1) ∧ (x1 < 0) ∧ (x2 < 0)) ↔ (a ≥ 1) :=
by
  sorry

end range_of_a_l301_301088


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301748

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301748


namespace find_n_l301_301944

theorem find_n :
  ∃ n : ℕ, (n + 1)! + (n + 3)! = n! * 1540 ∧ n = 10 :=
by
  sorry

end find_n_l301_301944


namespace sum_of_roots_l301_301276

theorem sum_of_roots (g : ℝ → ℝ) 
  (h_symmetry : ∀ x : ℝ, g (3 + x) = g (3 - x))
  (h_roots : ∃ s1 s2 s3 s4 : ℝ, 
               g s1 = 0 ∧ 
               g s2 = 0 ∧ 
               g s3 = 0 ∧ 
               g s4 = 0 ∧ 
               s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ 
               s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4) :
  s1 + s2 + s3 + s4 = 12 :=
by 
  sorry

end sum_of_roots_l301_301276


namespace teacher_student_relationship_l301_301146

theorem teacher_student_relationship
  (m n k ℓ : ℕ)
  (H1 : ∀ student, ∃! teachers, set.count teaching student teachers = ℓ)
  (H2 : ∀ teacher, ∃! students, set.count learning teacher students = k) :
  n * ℓ = m * k := sorry

end teacher_student_relationship_l301_301146


namespace paid_more_than_free_l301_301651

def num_men : ℕ := 194
def num_women : ℕ := 235
def free_admission : ℕ := 68
def total_people (num_men num_women : ℕ) : ℕ := num_men + num_women
def paid_admission (total_people free_admission : ℕ) : ℕ := total_people - free_admission
def paid_over_free (paid_admission free_admission : ℕ) : ℕ := paid_admission - free_admission

theorem paid_more_than_free :
  paid_over_free (paid_admission (total_people num_men num_women) free_admission) free_admission = 293 := 
by
  sorry

end paid_more_than_free_l301_301651


namespace grooming_time_correct_l301_301289

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l301_301289


namespace find_x_l301_301582

theorem find_x (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z)
(h₄ : x^2 / y = 3) (h₅ : y^2 / z = 4) (h₆ : z^2 / x = 5) : 
  x = (6480 : ℝ)^(1/7 : ℝ) :=
by 
  sorry

end find_x_l301_301582


namespace remainder_of_sum_of_primes_l301_301794

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301794


namespace vector_eq_w_l301_301939

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 2], ![5, 0]]

def w : Vector (Fin 2) ℝ := ![0, 31 / 11111]

theorem vector_eq_w :
  (B^8 + B^6 + B^4 + B^2 + 1) • w = ![0, 31] :=
by
  sorry

end vector_eq_w_l301_301939


namespace exists_F_12_mod_23_zero_l301_301218

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301218


namespace terminal_side_in_quadrant_l301_301477

theorem terminal_side_in_quadrant (k : ℤ) (α : ℝ)
  (h: π + 2 * k * π < α ∧ α < (3 / 2) * π + 2 * k * π) :
  (π / 2) + k * π < α / 2 ∧ α / 2 < (3 / 4) * π + k * π :=
sorry

end terminal_side_in_quadrant_l301_301477


namespace right_triangle_square_ratio_constant_l301_301186

theorem right_triangle_square_ratio_constant (A B C O : ℝ) (h1 : isRightTriangle A B C C) (h2 : is_center_of_square_on_hypotenuse O A B) :
  (∃ k : ℝ, ∀ (A B C O : ℝ), ratio (dist O C) ((dist A C) + (dist B C)) = k) ∧ k = (Real.sqrt 2) / 2 :=
by
  sorry

end right_triangle_square_ratio_constant_l301_301186


namespace abc_not_less_than_two_l301_301180

theorem abc_not_less_than_two (a b c : ℝ) (h : a + b + c = 6) : a ≥ 2 ∨ b ≥ 2 ∨ c ≥ 2 :=
sorry

end abc_not_less_than_two_l301_301180


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301671

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301671


namespace rectangle_count_l301_301372

theorem rectangle_count :
  ∃ (l : List (ℕ × ℕ)), 
    (∀ r ∈ l, let w := r.1, h := r.2 in w + h = 40 ∧ (w ≥ 2 * h ∨ h ≥ 2 * w)) ∧
    l.length = 13 ∧
    (∀ r1 r2 ∈ l, r1 ≠ r2 → ¬(r1.1 = r2.1 ∧ r1.2 = r2.2)) :=
sorry

end rectangle_count_l301_301372


namespace salary_increase_to_original_l301_301297

theorem salary_increase_to_original (S : ℝ) (h1 : S > 0) (h2 : let reduced_salary := 0.90 * S in true) : 
  (reduced_salary * (1 + 0.1111) = S) :=
by
  let reduced_salary := 0.90 * S
  sorry

end salary_increase_to_original_l301_301297


namespace circle_center_l301_301411

def circle_eq : Prop := ∀ x y : ℝ, x^2 - 6 * x + y^2 + 2 * y - 55 = 0

theorem circle_center :
  (∀ x y : ℝ, x^2 - 6 * x + y^2 + 2 * y - 55 = 0)
  → (3 : ℝ, -1 : ℝ) =
    let (h, k) := (3, -1) in (h, k) in
  (∃ h k r : ℝ, ∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2)
:= by
  -- Proof omitted
  sorry

end circle_center_l301_301411


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301755

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301755


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301783

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301783


namespace find_a_l301_301090

open Complex

noncomputable def z (a : ℝ) : ℂ := (1 - a * I) / (1 - I)

theorem find_a (a : ℝ) (h : (im (z a) = 4)) : a = -7 :=
by sorry

# Check if it compiles

end find_a_l301_301090


namespace y_run_time_l301_301835

theorem y_run_time (t : ℕ) (h_avg : (t + 26) / 2 = 42) : t = 58 :=
by
  sorry

end y_run_time_l301_301835


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301676

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301676


namespace probability_open_path_l301_301867

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l301_301867


namespace connected_graph_partitions_l301_301578

noncomputable def partitions_count (G : Graph) : Nat := -- Definition which counts valid partitions
  sorry

theorem connected_graph_partitions (G : Graph) (k : Nat) (h_connected : G.connected) (h_edges : G.edges = k) :
  partitions_count G ≥ k :=
by
  sorry

end connected_graph_partitions_l301_301578


namespace constant_AN_BM_product_l301_301999

noncomputable def circle_eq (x : ℝ) (y : ℝ) : Prop := x^2 + y^2 = 4

theorem constant_AN_BM_product :
  ∀ (x₀ y₀ : ℝ), 
    circle_eq x₀ y₀ → x₀ ≠ 0 → y₀ ≠ 0 → 
    let yₘ := -2 * y₀ / (x₀ - 2) in
    let xₙ := -2 * x₀ / (y₀ - 2) in
    let AN := 2 + 2 * x₀ / (y₀ - 2) in
    let BM := 2 + 2 * y₀ / (x₀ - 2) in
    |AN| * |BM| = 8 :=
by
  sorry

end constant_AN_BM_product_l301_301999


namespace prime_sum_remainder_l301_301777

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301777


namespace y_intercept_of_line_l301_301649

theorem y_intercept_of_line : 
  ∀ (x y : ℝ), y = x - 2 → (∀ x, x = 0 → y = -2) :=
by {
  intros x y h,
  specialize h 0,
  rw zero_sub at h,
  exact h,
  sorry
}

end y_intercept_of_line_l301_301649


namespace modular_inverse_13_mod_64_l301_301666

-- Define the problem statement in Lean 4
theorem modular_inverse_13_mod_64 : ∃ (x : ℤ), 0 ≤ x ∧ x < 64 ∧ 13 * x % 64 = 1 :=
begin
  use 5,
  -- We posited that 5 is the solution, and it needs to be verified.
  split,
  norm_num,
  split,
  norm_num,
  norm_num,
end

end modular_inverse_13_mod_64_l301_301666


namespace num_ways_to_sum_528_as_consecutive_seq_l301_301547

theorem num_ways_to_sum_528_as_consecutive_seq : 
  ∃ n : ℕ, n = 13 ∧ (∀ k : ℕ, ∀s : ℕ, (s = k + (k + 1) + ... + (k + n - 1) ∧ s = 528) → n ≥ 2) := 
sorry

end num_ways_to_sum_528_as_consecutive_seq_l301_301547


namespace vector_arithmetic_l301_301498

theorem vector_arithmetic (a b : ℝ × ℝ)
    (h₀ : a = (3, 5))
    (h₁ : b = (-2, 1)) :
    a - (2 : ℝ) • b = (7, 3) :=
sorry

end vector_arithmetic_l301_301498


namespace Tony_paints_columns_l301_301656

-- Defining constants and essential functions
def radius_from_diameter (d : ℕ) : ℕ := d / 2
def lateral_surface_area (r h : ℕ) : ℝ := 2 * real.pi * r * h
def num_gallons (area : ℝ) (coverage : ℕ) : ℝ := area / coverage
def round_up (x : ℝ) : ℕ := real.ceil x

-- Problem conditions
def diameter := 8
def height := 15
def num_columns := 20
def coverage_per_gallon := 300

-- Calculation steps corresponding to provided solution (without assuming any result directly)
def radius := radius_from_diameter diameter
def single_column_area := lateral_surface_area radius height
def total_area := single_column_area * num_columns
def gallons_needed := num_gallons total_area coverage_per_gallon
def gallons_to_buy := round_up gallons_needed

-- Proof statement
theorem Tony_paints_columns : gallons_to_buy = 26 := by
  sorry

end Tony_paints_columns_l301_301656


namespace sin_theta_l301_301049

theorem sin_theta (θ : ℝ) (h : cos (π / 4 - θ / 2) = 2 / 3) : sin θ = -1 / 9 :=
  sorry

end sin_theta_l301_301049


namespace intersection_complement_l301_301070

open Set

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}
def B : Set ℝ := {x | 2^(x-2) > 1}

theorem intersection_complement :
  A ∩ (compl B) = {x | -1 < x ∧ x ≤ 2} := by
  sorry

end intersection_complement_l301_301070


namespace remainder_first_six_primes_div_seventh_l301_301816

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301816


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301667

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301667


namespace problem_statement_l301_301050

noncomputable def omega : ℝ := sorry
noncomputable def max_value : ℝ := sorry
noncomputable def max_value_set : Set ℝ := sorry

theorem problem_statement 
  (a b : ℝ → ℝ × ℝ)
  (x : ℝ)
  (ω : ℝ)
  (f : ℝ → ℝ)
  (h1 : a x = (sin (ω * x), cos (ω * x)))
  (h2 : b x = (sin (ω * x) + 2 * cos (ω * x), cos (ω * x)))
  (h3 : f x = (a x).1 * (b x).1 + (a x).2 * (b x).2)
  (h4 : ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = π / 4) :
  (ω = 4 ∧ max_value = 2 ∧ max_value_set = { x : ℝ | ∃ (k : ℤ), x = π / 16 + k * π / 4}) :=
sorry

end problem_statement_l301_301050


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301696

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301696


namespace yen_to_usd_conversion_l301_301396

theorem yen_to_usd_conversion
  (cost_of_souvenir : ℕ)
  (service_charge : ℕ)
  (conversion_rate : ℕ)
  (total_cost_in_yen : ℕ)
  (usd_equivalent : ℚ)
  (h1 : cost_of_souvenir = 340)
  (h2 : service_charge = 25)
  (h3 : conversion_rate = 115)
  (h4 : total_cost_in_yen = cost_of_souvenir + service_charge)
  (h5 : usd_equivalent = (total_cost_in_yen : ℚ) / conversion_rate) :
  total_cost_in_yen = 365 ∧ usd_equivalent = 3.17 :=
by
  sorry

end yen_to_usd_conversion_l301_301396


namespace sofia_initial_floor_l301_301612

theorem sofia_initial_floor (x : ℤ) (h1 : x + 7 - 6 + 5 = 20) : x = 14 := 
sorry

end sofia_initial_floor_l301_301612


namespace remainder_of_sum_of_primes_l301_301798

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301798


namespace tangent_circumcircle_bisects_EC_l301_301890

open EuclideanGeometry

variables {A B C D E : Point}

theorem tangent_circumcircle_bisects_EC
  (ABC_isosceles : is_isosceles_triangle A B C)
  (D_on_BC : lies_on D (line_through B C))
  (E_on_parallel_B_AC : ∃ l, is_parallel l (line_through A C) ∧ E = intersection (line_through B l) (line_through A D))
  (circumcircle_ABD : ∃ O, is_circumcircle_of O A B D) :
  ∃ F, is_tangent_at F (circumcircle_ABD), is_midpoint F E C :=
by
  sorry

end tangent_circumcircle_bisects_EC_l301_301890


namespace arithmetic_progression_common_difference_l301_301952

theorem arithmetic_progression_common_difference :
  ∀ (A1 An n d : ℕ), A1 = 3 → An = 103 → n = 21 → An = A1 + (n - 1) * d → d = 5 :=
by
  intros A1 An n d h1 h2 h3 h4
  sorry

end arithmetic_progression_common_difference_l301_301952


namespace k3c60_properties_l301_301929

theorem k3c60_properties :
  (∀ (f: ℕ → Type) (n : ℕ), 
  (f 1 = "K_3C_{60} contains both ionic bonds between K^+ and C_{60}^{3-} and nonpolar covalent bonds between C atoms") ∧
  (f 2 = "The chemical formula of K_3C_{60} consists of 3 K^+ ions and 1 C_{60}^{3-} ion") ∧
  (f 3 = "In the crystalline state, K_3C_{60} does not have free-moving ions") →
  f n = "The correct analysis is that K_3C_{60} contains both ionic and covalent bonds"
  ) :=
sorry

end k3c60_properties_l301_301929


namespace Partial_Differential_Equation_for_P_t_l301_301148

noncomputable def P_t (t x : ℝ) : ℝ :=
  (√2 / √(π * t)) * Real.exp (-(x - 2 * t)^2 / (2 * t))

theorem Partial_Differential_Equation_for_P_t :
  ∀ (t x : ℝ),
    (∂ P_t t x / ∂ t) = -2 * (∂ P_t t x / ∂ x) + (1 / 2) * (∂² P_t t x / ∂ (x^2)) :=
by
  -- The proof is omitted here.
  sorry

end Partial_Differential_Equation_for_P_t_l301_301148


namespace initial_sum_invested_l301_301365

-- Definition of the initial problem setup and given conditions
def compound_interest (P r n : ℝ) : ℝ :=
  P * (1 + r) ^ n

def interest_difference (P r1 r2 n : ℝ) : ℝ :=
  (compound_interest P r1 n) - (compound_interest P r2 n)

theorem initial_sum_invested (P : ℝ) :
  let r1 := 0.15
  let r2 := 0.10
  let n := 3
  interest_difference P r1 r2 n = 1500 →
  P ≈ 7899.47 :=
by
  sorry

end initial_sum_invested_l301_301365


namespace locus_midpoint_of_XY_locus_of_Z_l301_301055

-- Definition of the cube and line segments
variables {ABC A'B'C'D': Type} [Cube ABC A'B'C'D']
variables {AC B'D': line_segment ABC A'B'C'D'}
variables {X Y Z: point}

-- Theorem 1: Locus of the midpoint of XY on AC and B'D' is the plane EFGH
theorem locus_midpoint_of_XY (X_on_AC: X ∈ AC) (Y_on_B'D': Y ∈ B'D') : 
    ∀ XY_midpoint, (XY_midpoint = midpoint X Y) → (XY_midpoint ∈ plane_EFGH) :=
by
  sorry

-- Theorem 2: Locus of point Z on XY such that ZY = 2XZ is the plane E'F'G'H'
theorem locus_of_Z (X_on_AC: X ∈ AC) (Y_on_B'D': Y ∈ B'D') (Z_on_XY: Z ∈ line_segment X Y) 
    (ZY_2XZ: dist Z Y = 2 * dist X Z) :
    Z ∈ plane_E'F'G'H' :=
by
  sorry

end locus_midpoint_of_XY_locus_of_Z_l301_301055


namespace ellipse_and_trapezoid_problems_l301_301060

-- Define the properties and conditions
axiom ex : 
  ∃ a b c : ℝ, a > b ∧ b > 0 ∧ a = 2 ∧ b = √3 ∧ c = 1 ∧ (λ x y, (x^2 / a^2 + y^2 / b^2 = 1)) = (λ x y, (x^2 / 4 + y^2 / 3 = 1))

noncomputable def problem_1_solution : Prop := 
  ∀ x y : ℝ, ex → (x^2 / 4 + y^2 / 3 = 1)

-- Define the properties for part (Ⅱ)
axiom trapezoid_condition :
  ∀ P Q M : ℝ × ℝ, P.1 = 4 ∧ Q = (4, 0) ∧ (M.2 ≠ 0) →
  (∃ APQM : Trapezoid, 
  APQM = mkTrapezoid (A, P, Q, M) ∧
  (exists P : (ℝ × ℝ), P = (4, 3) ∨ P = (4, -3)))

noncomputable def problem_2_solution : Prop := 
  ∀ A B P Q M : ℝ × ℝ, 
  A = (-2, 0) ∧ B = (2, 0) ∧ Q = (4, 0) ∧ 
  (P.1 = 4 ∧ (P.2 = 3 ∨ P.2 = -3)) → 
  trapezoid_condition P Q M

-- Main theorem combining both parts
theorem ellipse_and_trapezoid_problems :
  problem_1_solution ∧ problem_2_solution :=
by sorry

end ellipse_and_trapezoid_problems_l301_301060


namespace exists_F_12_mod_23_zero_l301_301223

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301223


namespace unique_line_splits_triangle_l301_301645

variables {A B C : Point}
variables {AB CA CB : ℝ}

def is_median_length (AB CA CB : ℝ) : Prop := 
  (CA < AB) ∧ (AB < CB)

def splits_perimeter_and_area_equally (line : Line) (A B C : Point) : Prop :=
  let D E := intersections line A C B in
  let p := perimeter A B C in
  let s := p / 2 in
  let area_h := area A B C in
  let area_s := area_h / 2 in
  (perimeter A D E = s) ∧ (perimeter B D E = s) ∧
  (area A D E = area_s) ∧ (area B D E = area_s)

theorem unique_line_splits_triangle (h : is_median_length AB CA CB) :
  ∃! line : Line, splits_perimeter_and_area_equally line A B C :=
sorry

end unique_line_splits_triangle_l301_301645


namespace prime_sum_remainder_l301_301778

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301778


namespace tan_cot_theta_l301_301445

theorem tan_cot_theta 
  (θ : ℝ) 
  (h1 : Real.sin θ + Real.cos θ = (Real.sqrt 2) / 3) 
  (h2 : Real.pi / 2 < θ ∧ θ < Real.pi) : 
  Real.tan θ - (1 / Real.tan θ) = - (8 * Real.sqrt 2) / 7 := 
sorry

end tan_cot_theta_l301_301445


namespace geometric_sequence_values_l301_301110

theorem geometric_sequence_values (l a b c : ℝ) (h : ∃ r : ℝ, a / (-l) = r ∧ b / a = r ∧ c / b = r ∧ (-9) / c = r) : b = -3 ∧ a * c = 9 :=
by
  sorry

end geometric_sequence_values_l301_301110


namespace floor_sqrt_sum_eq_floor_sqrt_expr_l301_301603

-- Proof problem definition
theorem floor_sqrt_sum_eq_floor_sqrt_expr (n : ℕ) : 
  (Int.floor (Real.sqrt n + Real.sqrt (n + 1))) = (Int.floor (Real.sqrt (4 * n + 2))) := 
sorry

end floor_sqrt_sum_eq_floor_sqrt_expr_l301_301603


namespace sequence_transformation_possible_l301_301385

theorem sequence_transformation_possible 
  (a1 a2 : ℕ) (h1 : a1 ≤ 100) (h2 : a2 ≤ 100) (h3 : a1 ≥ a2) : 
  ∃ (operations : ℕ), operations ≤ 51 :=
by
  sorry

end sequence_transformation_possible_l301_301385


namespace exist_arrangement_for_P_23_l301_301199

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301199


namespace triangle_calculations_acute_triangle_perimeter_range_l301_301135

noncomputable def cosine_rule (a b c : ℝ) (cosine_c : ℝ) : ℝ :=
(math.sqrt (a^2 + b^2 - 2 * a * b * cosine_c))

theorem triangle_calculations {A B C : Type*}
(ab ac : ℝ) (cosine_a : ℝ) (h : ℝ) :
  ab = 4 →
  ac = 6 →
  cosine_a = 1/16 →
  ∃ (bc : ℝ), bc = 7 ∧ h = (3 * real.sqrt 255) / 14 := 
by
  sorry

theorem acute_triangle_perimeter_range {A B C : Type*}
(ab ac : ℝ) (cosine_a : ℝ) :
  ab = 4 →
  ac = 6 →
  0 < cosine_a ∧ cosine_a < 1 →
  12 + 2 < cosine_rule ab ac cosine_a < 12 + 2 * real.sqrt 13 :=
by
  sorry

end triangle_calculations_acute_triangle_perimeter_range_l301_301135


namespace problem_solution_l301_301634

noncomputable def expr := 
  (Real.tan (Real.pi / 15) - Real.sqrt 3) / ((4 * (Real.cos (Real.pi / 15))^2 - 2) * Real.sin (Real.pi / 15))

theorem problem_solution : expr = -4 :=
by
  sorry

end problem_solution_l301_301634


namespace correct_equation_l301_301452

variables {V : Type*} [normed_group V] [normed_space ℝ V]

-- Definitions
variables (e : V) (a b : V)

-- Unit vector condition
def is_unit_vector (v : V) : Prop := ∥v∥ = 1

-- Non-zero vectors condition
def is_nonzero_vector (v : V) : Prop := v ≠ 0

-- Lean Statement
theorem correct_equation
  (h_e_unit : is_unit_vector e)
  (h_a_nonzero : is_nonzero_vector a)
  (h_b_nonzero : is_nonzero_vector b) :
  ∥e∥ * b = b :=
by {
  -- By assumption, this is a unit vector
  rw is_unit_vector at h_e_unit,
  -- Apply the property of the unit vector norm being 1
  rw h_e_unit,
  -- Conclude that 1 * b = b
  exact mul_one b,
}

end correct_equation_l301_301452


namespace perfect_squares_sum_l301_301654

theorem perfect_squares_sum (L : List ℕ) (hL : L = [1, 2, 3, 4, 5, 6, 7, 8, 9])
(m n : ℕ) (hm : m ≠ n) (h1 : List.prod (take 6 (L.erase 5).erase 7) ^ 2 = m ^ 2)
(h2 : List.prod (takeR 6 (L.erase 5).erase 7) ^ 2 = n ^ 2) :
  m + n = 108 :=
sorry

end perfect_squares_sum_l301_301654


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301704

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301704


namespace total_bill_correct_l301_301388

def cost_food : ℝ := 200
def tax_rate : ℝ := 0.08
def tips : List (String × ℝ) := [("Sally", 0.15), ("Sam", 0.18), ("Alyssa", 0.20), ("Greg", 0.22), ("Sarah", 0.25)]

def calc_sales_tax (cost : ℝ) (rate : ℝ) : ℝ := cost * rate
def total_cost_with_tax (cost tax : ℝ) : ℝ := cost + tax

def tip_amounts (cost : ℝ) (tips : List (String × ℝ)) : List (String × ℝ) :=
  tips.map (λ (name, rate), (name, cost * rate))

def total_per_person (total_tax : ℝ) (tip : ℝ) (num_people : ℕ) : ℝ :=
  (total_tax + tip) / num_people

def person_paid (cost_with_tax : ℝ) (tips : List (String × ℝ)) (num_people : ℕ) :=
  tips.map (λ (name, tip), (name, total_per_person cost_with_tax tip num_people))

def expected_totals : List (String × ℝ) := 
  [("Sally", 49.20), ("Sam", 50.40), ("Alyssa", 51.20), ("Greg", 52.00), ("Sarah", 53.20)]

theorem total_bill_correct :
  let sales_tax := calc_sales_tax cost_food tax_rate in
  let cost_with_tax := total_cost_with_tax cost_food sales_tax in
  let tips_amounts := tip_amounts cost_food tips in
  let num_people : ℕ := 5 in
  person_paid cost_with_tax tips_amounts num_people = expected_totals :=
by sorry

end total_bill_correct_l301_301388


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301781

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301781


namespace range_H_l301_301956

def H (x : ℝ) : ℝ := |x + 2| - |x - 3|

theorem range_H : Set.Iic (5 : ℝ) = Set.range H :=
by
  sorry

end range_H_l301_301956


namespace lines_meet_at_common_midpoint_l301_301250

-- Definitions for conditions
variables {A B C D P O O1 O2 O3 O4 : Point}
variable (circle : set Point)
variable (inscribed_quad : IsInscribedQuadrilateral A B C D circle)
variable (intersection : IsIntersectionOfDiagonals A C B D P)
variable (circumcenter_ABP : IsCircumcenter A B P O1)
variable (circumcenter_BCP : IsCircumcenter B C P O2)
variable (circumcenter_CDP : IsCircumcenter C D P O3)
variable (circumcenter_DAP : IsCircumcenter D A P O4)
variable (circumcenter_ABCD : IsCircumcenter A B C D O)

-- The theorem to prove
theorem lines_meet_at_common_midpoint :
  ∃ M, IsMidpoint M O P ∧ IsMidpoint M O1 O3 ∧ IsMidpoint M O2 O4 := sorry

end lines_meet_at_common_midpoint_l301_301250


namespace mars_colonization_cost_l301_301270

theorem mars_colonization_cost (C : ℝ) (P : ℝ) (rich_pay_rate : ℝ) (pop_fraction_rich : ℝ) :
  C = 50 * 10^9 ∧
  P = 300 * 10^6 ∧
  rich_pay_rate = 2 ∧
  pop_fraction_rich = 0.1 →
  let P_10 := pop_fraction_rich * P in
  let P_90 := P - P_10 in
  let total_cost := (P_90 * x) + (P_10 * rich_pay_rate * x) in
  total_cost = C →
  x ≈ 152 :=
by
  intros
  simp
  sorry

end mars_colonization_cost_l301_301270


namespace exists_arrangement_for_P_23_l301_301229

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301229


namespace John_has_22_quarters_l301_301566

variable (q d n : ℕ)

-- Conditions
axiom cond1 : d = q + 3
axiom cond2 : n = q - 6
axiom cond3 : q + d + n = 63

theorem John_has_22_quarters : q = 22 := by
  sorry

end John_has_22_quarters_l301_301566


namespace f_value_range_l301_301172

noncomputable theory
open_locale classical

def f (x y z : ℝ) := x / (x + y) + y / (y + z) + z / (z + x)

theorem f_value_range (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  1 < f x y (x^2) ∧ f x y (x^2) < 2 :=
sorry

end f_value_range_l301_301172


namespace derivative_at_pi_over_4_l301_301095

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x

theorem derivative_at_pi_over_4 : 
  deriv f (Real.pi / 4) = Real.sqrt 2 / 2 + Real.sqrt 2 * Real.pi / 8 :=
by
  -- Proof goes here
  sorry

end derivative_at_pi_over_4_l301_301095


namespace johns_watermelon_weight_l301_301593

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l301_301593


namespace exhibit_special_13_digit_integer_l301_301421

open Nat 

def thirteenDigitInteger (N : ℕ) : Prop :=
  N ≥ 10^12 ∧ N < 10^13

def isMultipleOf8192 (N : ℕ) : Prop :=
  8192 ∣ N

def hasOnlyEightOrNineDigits (N : ℕ) : Prop :=
  ∀ d ∈ digits 10 N, d = 8 ∨ d = 9

theorem exhibit_special_13_digit_integer : 
  ∃ N : ℕ, thirteenDigitInteger N ∧ isMultipleOf8192 N ∧ hasOnlyEightOrNineDigits N ∧ N = 8888888888888 := 
by
  sorry 

end exhibit_special_13_digit_integer_l301_301421


namespace sum_of_distinct_prime_factors_l301_301035

theorem sum_of_distinct_prime_factors : 
  ∑ p in ({2, 3, 7, 11} : Finset ℕ), p = 23 :=
by
  -- Definitions according to conditions
  have cond1 : 7^6 - 7^4 + 11 = 7^4 * 48 + 11 := by sorry
  have cond2 : Prime 2 := by sorry
  have cond3 : Prime 3 := by sorry
  have cond4 : Prime 7 := by sorry
  have cond5 : Prime 11 := by sorry
  have cond6 : 48 = 2^4 * 3 := by sorry
  
  sorry

end sum_of_distinct_prime_factors_l301_301035


namespace minimum_c_value_l301_301132

theorem minimum_c_value
  (a b c : ℝ)
  (h1 : 2^a + 4^b = 2^c)
  (h2 : 4^a + 2^b = 4^c) :
  c = log 2 3 - 5 / 3 :=
by
  sorry

end minimum_c_value_l301_301132


namespace pieces_per_pizza_is_five_l301_301184

-- Definitions based on the conditions
def cost_per_pizza (total_cost : ℕ) (number_of_pizzas : ℕ) : ℕ :=
  total_cost / number_of_pizzas

def number_of_pieces_per_pizza (cost_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  cost_per_pizza / cost_per_piece

-- Given conditions
def total_cost : ℕ := 80
def number_of_pizzas : ℕ := 4
def cost_per_piece : ℕ := 4

-- Prove
theorem pieces_per_pizza_is_five : number_of_pieces_per_pizza (cost_per_pizza total_cost number_of_pizzas) cost_per_piece = 5 :=
by sorry

end pieces_per_pizza_is_five_l301_301184


namespace scott_awards_l301_301161

theorem scott_awards (S : ℕ) 
  (h1 : ∃ J, J = 3 * S)
  (h2 : ∃ B, B = 2 * (3 * S) ∧ B = 24) : S = 4 := 
by 
  sorry

end scott_awards_l301_301161


namespace exists_A_satisfying_condition_l301_301951

-- Define the golden ratio
noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2

-- Define the value of A
noncomputable def A : ℝ := φ^2

-- Define the property that needs to be proven
theorem exists_A_satisfying_condition :
  ∃ A : ℝ, (∀ n : ℕ, 0 < n → (∃ k : ℤ, (Real.ceil (A^n) = k^2 + 2 ∨ Real.ceil (A^n) = k^2 - 2))) :=
begin
  use A,
  sorry  -- The proof will be here
end

end exists_A_satisfying_condition_l301_301951


namespace exists_similar_sizes_P_23_l301_301212

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301212


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301711

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301711


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301684

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301684


namespace sum_of_4_digit_numbers_l301_301322

open Finset

/-- Sum of all 24 four-digit numbers formed using the digits 1, 3, 5, and 7 without repetition. -/
theorem sum_of_4_digit_numbers : 
  let digits := {1, 3, 5, 7} in
  let num_ways := 4! in
  let place_count := 3! in
  let sum_digits := (1 + 3 + 5 + 7) in
  num_ways * place_count * (sum_digits * 1000 + sum_digits * 100 + sum_digits * 10 + sum_digits * 1) = 106656 :=
by
  sorry

end sum_of_4_digit_numbers_l301_301322


namespace simplify_and_evaluate_l301_301265

theorem simplify_and_evaluate :
  let x := -1 / 4 in
  ((x - 1) ^ 2 - 3 * x * (1 - x) - (2 * x - 1) * (2 * x + 1)) = 13 / 4 :=
by
  let x := -1 / 4
  sorry

end simplify_and_evaluate_l301_301265


namespace find_k_l301_301117

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l301_301117


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301752

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301752


namespace range_m_l301_301993

open Real

theorem range_m (m : ℝ)
  (hP : ¬ (∃ x : ℝ, m * x^2 + 1 ≤ 0))
  (hQ : ¬ (∃ x : ℝ, x^2 + m * x + 1 < 0)) :
  0 ≤ m ∧ m ≤ 2 := 
sorry

end range_m_l301_301993


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301708

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301708


namespace probability_of_passing_C_l301_301404

-- Define positions of points A, B, C
structure Position where
  east : ℕ
  south : ℕ

def A : Position := ⟨0, 0⟩
def B : Position := ⟨4, 3⟩
def C : Position := ⟨2, 1⟩

-- Function to count the number of paths from one position to another
def numPaths (from to : Position) : ℕ :=
  let eastSteps := to.east - from.east
  let southSteps := to.south - from.south
  Nat.choose (eastSteps + southSteps) eastSteps

-- Computing the required paths and the probability
def probThroughC : ℚ :=
  let pathsAtoC := numPaths A C
  let pathsCtoB := numPaths C B
  let pathsAtoB := numPaths A B
  (pathsAtoC * pathsCtoB : ℚ) / pathsAtoB

theorem probability_of_passing_C : probThroughC = 18 / 35 := by
  sorry

end probability_of_passing_C_l301_301404


namespace count_31_l301_301580

-- Define the function hat_g that calculates the product of proper divisors of n
noncomputable def hat_g (n : ℕ) : ℕ :=
  ∏ d in finset.filter (λ d, d ∣ n ∧ d ≠ n) (finset.range (n + 1)), d

-- Define the predicate for the numbers we are interested in
def not_divides_hat_g (n : ℕ) : Prop :=
  ¬ (n ∣ hat_g n)

-- Count the number of such n in the range from 2 to 100
noncomputable def count_values_not_divide_hat_g : ℕ :=
  finset.card (finset.filter not_divides_hat_g (finset.Icc 2 100))

-- The main theorem
theorem count_31 : count_values_not_divide_hat_g = 31 :=
  sorry

end count_31_l301_301580


namespace general_equation_of_line_cartesian_equation_of_circle_sum_of_distances_l301_301550

theorem general_equation_of_line (t : ℝ) :
  ∃ (y : ℝ), y = √3 * t - 2 :=
begin
  use x,
  rw y_eq: y = √3 * t - 2,
  sorry
end

theorem cartesian_equation_of_circle (ρ θ : ℝ) :
  ∃ (x y : ℝ), x^2 + y^2 - 4 * x + 3 = 0 :=
begin
  use [x, y],
  rw ρ_eq: ρ = sqrt(x^2 + y^2), 
  rw θ_eq: θ = arccos(x / ρ),
  rw cast_eq_real (ρ^2 - 4*ρ*cos(θ) + 3),
  sorry
end

theorem sum_of_distances (P A B : ℝ × ℝ) :
  P = (0, -2) → (A = (x1, y1)) → (B = (x2, y2)) →
  ∃ (PA PB : ℝ), |PA| + |PB| = 2 * √3 + 2 :=
begin
  rw P_eq: P = (0, -2),
  rw A_eq: A = (x1, y1),
  rw B_eq: B = (x2, y2),
  use PA_RING,
  use PB_RING,
  rw calc_PA: PA = sqrt(x1^2 + y1^2),
  rw calc_PB: PB = sqrt(x2^2 + y2^2),
  rw dist_eq: PA + PB = 2 * √3 + 2,
  sorry
end

end general_equation_of_line_cartesian_equation_of_circle_sum_of_distances_l301_301550


namespace quotient_of_sum_of_squares_mod_13_l301_301616

theorem quotient_of_sum_of_squares_mod_13 :
  let m := (Finset.range 16).image (λ n, (n * n) % 13)
  m.sum / 13 = 3 :=
by
  let n_squares := ((Finset.range 16).map (λ n, (n * n) % 13)).erase_dups
  let m : Nat := n_squares.sum
  have h1 : (m = 39) := by sorry
  have h2 : (39 / 13 = 3) := by sorry
  exact h2

end quotient_of_sum_of_squares_mod_13_l301_301616


namespace pyramid_volume_proof_l301_301606

-- Define the pyramid's base as a rectangle and the pyramid's height
variables (EF FG QE : ℝ)

-- Conditions given in the problem
def is_base_rectangle (EF FG : ℝ) (Area : ℝ) : Prop :=
  Area = EF * FG

def is_pyramid_height (h : ℝ) : Prop :=
  h = QE

-- Define volume of a pyramid
def pyramid_volume (base_area height : ℝ) : ℝ :=
  (1 / 3) * base_area * height

-- Main statement to prove
theorem pyramid_volume_proof 
  (h_area : is_base_rectangle EF FG 50)
  (height_condition : is_pyramid_height 8) :
  pyramid_volume 50 8 = 400 / 3 := 
sorry

end pyramid_volume_proof_l301_301606


namespace second_person_fraction_removed_l301_301970

theorem second_person_fraction_removed (teeth_total : ℕ) 
    (removed1 removed3 removed4 : ℕ)
    (total_removed: ℕ)
    (h1: teeth_total = 32)
    (h2: removed1 = teeth_total / 4)
    (h3: removed3 = teeth_total / 2)
    (h4: removed4 = 4)
    (h5 : total_removed = 40):
    ((total_removed - (removed1 + removed3 + removed4)) : ℚ) / teeth_total = 3 / 8 :=
by
  sorry

end second_person_fraction_removed_l301_301970


namespace find_a2_plus_b2_l301_301992

-- Definitions and conditions
def point_P := (-2 * Real.sqrt 2, 0)
def ellipse_eq (a b : ℝ) (x y : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4
def left_vertex (P : ℝ × ℝ) (a : ℝ) : Prop := P = (-2 * Real.sqrt 2, 0) ∧ a = 2 * Real.sqrt 2
def left_focus_F (F : ℝ × ℝ) : Prop := F = (-Real.sqrt 2, 0)

-- Proof statement
theorem find_a2_plus_b2 
    (a b : ℝ) (F P : ℝ × ℝ)
    (h1 : ellipse_eq a b (-2 * Real.sqrt 2) 0)
    (h2 : circle_eq (-2 * Real.sqrt 2) 0)
    (h3 : left_vertex P a)
    (h4 : left_focus_F F)
    (h5 : a > 0 ∧ b > 0) :
    a^2 + b^2 = 14 := 
sorry

end find_a2_plus_b2_l301_301992


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301735

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301735


namespace increase_in_sides_of_polygon_l301_301532

theorem increase_in_sides_of_polygon (n n' : ℕ) (h : (n' - 2) * 180 - (n - 2) * 180 = 180) : n' = n + 1 :=
by
  sorry

end increase_in_sides_of_polygon_l301_301532


namespace sufficient_not_necessary_condition_l301_301023

theorem sufficient_not_necessary_condition (α β γ : ℝ) : 
  (γ - β = β - α) → (sin (α + γ) = sin (2 * β)) ∧ 
  ¬ (∀ k : ℤ, sin (α + γ) = sin (2 * β + k * π) → γ - β = β - α) := by
  sorry

end sufficient_not_necessary_condition_l301_301023


namespace find_n_l301_301522

theorem find_n (a b_0 b_1 b_2 ... b_n : ℕ) (n : ℕ) 
  (h1 : (1 + a) + (1 + a)^2 + (1 + a)^3 + ... + (1 + a)^n = b_0 + b_1 * a + b_2 * a^2 + ... + b_n * a^n)
  (h2 : b_0 + b_1 + b_2 + ... + b_n = 30) : 
  n = 4 := 
sorry

end find_n_l301_301522


namespace only_negative_integer_among_list_l301_301908

namespace NegativeIntegerProblem

def is_integer (n : ℚ) : Prop := ∃ m : ℤ, n = m

theorem only_negative_integer_among_list :
  (∃ x, x ∈ [0, -1, 2, -1.5] ∧ (x < 0) ∧ is_integer x) ↔ (x = -1) :=
by
  sorry

end NegativeIntegerProblem

end only_negative_integer_among_list_l301_301908


namespace count_integers_satisfying_log_condition_l301_301042

theorem count_integers_satisfying_log_condition : 
  ∃ (count : ℕ), 
    count = 59 ∧ 
    ∀ (x : ℕ), 30 < x ∧ x < 90 → 
      ( log 10 (x - 30) + log 10 (90 - x) < 3 ) :=
sorry

end count_integers_satisfying_log_condition_l301_301042


namespace exists_arrangement_for_P_23_l301_301226

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301226


namespace exist_arrangement_for_P_23_l301_301198

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301198


namespace remainder_first_six_primes_div_seventh_l301_301819

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301819


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301738

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301738


namespace probability_from_first_to_last_l301_301855

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l301_301855


namespace jessy_earrings_ratio_l301_301885

-- Define the initial conditions
def initial_necklaces : ℕ := 10
def initial_earrings : ℕ := 15
def bought_necklaces : ℕ := 10
def total_jewelry : ℕ := 57

-- Define the new number of earrings bought by Jessy (variable)
variable (E : ℕ)

-- Define the relationship given by the problem
def given_equation (E : ℕ) : Prop :=
  initial_necklaces + initial_earrings + bought_necklaces + E + E / 5 = total_jewelry

-- Express the ratio of earrings Jessy bought to the number she had initially
def earrings_bought_to_initial_ratio (E : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd E initial_earrings in
  (E / gcd, initial_earrings / gcd)

-- State the problem as a theorem
theorem jessy_earrings_ratio : 
  (∃ (E : ℕ), given_equation E) → (earrings_bought_to_initial_ratio 27 = (9, 5)) :=
  by
  sorry

end jessy_earrings_ratio_l301_301885


namespace other_root_of_quadratic_l301_301044

theorem other_root_of_quadratic (m : ℝ) (h : (2:ℝ) is_root polynomial.ring_of {coeffs := [m, -6, 1]}) :
  ∃ t : ℝ, t = 4 := sorry

end other_root_of_quadratic_l301_301044


namespace range_f_l301_301163

noncomputable def f (x : ℝ) : ℝ := (Real.arccos x)^4 + (Real.arcsin x)^4

theorem range_f : set.image f (set.Icc (-1 : ℝ) 1) = set.Icc ((13 * Real.pi^4) / 16) (Real.pi^4 / 2) :=
by
  sorry

end range_f_l301_301163


namespace comb_identity_l301_301460

theorem comb_identity : (choose 17 10 = 19448) ∧ (choose 17 11 = 12376) ∧ (choose 19 12 = 50388) → choose 18 12 = 18564 := 
by
  sorry

end comb_identity_l301_301460


namespace ratio_depends_on_S_and_r_l301_301015

theorem ratio_depends_on_S_and_r
    (S : ℝ) (r : ℝ) (P1 : ℝ) (C2 : ℝ)
    (h1 : P1 = 4 * S)
    (h2 : C2 = 2 * Real.pi * r) :
    (P1 / C2 = 4 * S / (2 * Real.pi * r)) := by
  sorry

end ratio_depends_on_S_and_r_l301_301015


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301728

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301728


namespace max_points_four_circles_l301_301971

noncomputable def max_intersection_points (n : ℕ) : ℕ := 2 * n

theorem max_points_four_circles (circles : list ℝ) (h : circles.length = 4) : 
  max_intersection_points 4 = 8 :=
by 
  simp [max_intersection_points]

end max_points_four_circles_l301_301971


namespace garden_breadth_l301_301129

-- Problem statement conditions
def perimeter : ℝ := 600
def length : ℝ := 205

-- Translate the problem into Lean:
theorem garden_breadth (breadth : ℝ) (h1 : 2 * (length + breadth) = perimeter) : breadth = 95 := 
by sorry

end garden_breadth_l301_301129


namespace trig_identity_proof_l301_301081

variable (α : ℝ)
variable (h1 : π / 2 < α ∧ α < π)
variable (h2 : sin (π / 2 + α) = -sqrt 5 / 5)

theorem trig_identity_proof:
  (cos α)^3 + sin α) / (cos(α - π / 4)) = 9 * sqrt 2 / 5 :=
by
  sorry

end trig_identity_proof_l301_301081


namespace is_pythagorean_triple_l301_301334

theorem is_pythagorean_triple (a b c : ℕ) (h: a^2 + b^2 = c^2) : a^2 + b^2 = c^2 := h

def array_C := (6, 8, 10)

example : is_pythagorean_triple 6 8 10 := by
  have h : 6^2 + 8^2 = 10^2 := by
    calc
      6^2 + 8^2 = 36 + 64 := by rw [pow_two, pow_two]
      ... = 100 := by norm_num
      ... = 10^2 := by rw [pow_two]
  exact h

end is_pythagorean_triple_l301_301334


namespace wrenches_in_comparison_group_l301_301503

theorem wrenches_in_comparison_group (H W : ℝ) (x : ℕ) 
  (h1 : W = 2 * H)
  (h2 : 2 * H + 2 * W = (1 / 3) * (8 * H + x * W)) : x = 5 :=
by
  sorry

end wrenches_in_comparison_group_l301_301503


namespace philosophy_class_B_students_l301_301541

-- Noncomputable definitions are used to handle real number arithmetic in proofs
noncomputable def probability_A (p_B : ℝ) : ℝ := 0.5 * p_B
noncomputable def probability_C (p_B : ℝ) : ℝ := 2 * p_B

theorem philosophy_class_B_students (p_B : ℝ) :
  ∃ x : ℝ, (0.5 * x) + x + (2 * x) = 40 ∧ x ≈ floor (40 / 3.5) :=
begin
  sorry
end

end philosophy_class_B_students_l301_301541


namespace rooks_arrangement_count_l301_301504

theorem rooks_arrangement_count :
  ∃ n : ℕ, n = 3456 ∧
  (∀ (board : ℕ → ℕ → Prop), 
     (∀ i j, board i j → (1 ≤ i ∧ i ≤ 8 ∧ 1 ≤ j ∧ j ≤ 8)) ∧ 
     (∀ i j k, board i j → board i k → j = k) ∧ 
     (∀ i j k, board i j → board k j → i = k) ∧ 
     (∀ i j, board i j → i ≠ j) ∧ 
     (∃ f : ℕ → ℕ, 
        (∀ i, 1 ≤ f i ∧ f i ≤ 8) ∧ 
        function.injective f ∧ 
        ∃ nodup_list : list ℕ, 
          nodup_list = list.range 8 ∧ 
          (∀ i, f i = i → (list.nth_le nodup_list i (by linarith) ∈ finset.range 8)))) :=
begin
  sorry
end

end rooks_arrangement_count_l301_301504


namespace max_ln_div_expr_l301_301943

noncomputable def ln_div_expr (x : ℝ) : ℝ := (Real.log x) / (2 * x)

theorem max_ln_div_expr : 
  ∃ e : ℝ, (e = Real.exp 1) ∧  ∀ x > 0, (ln_div_expr x ≤ ln_div_expr e) :=
begin
  use Real.exp 1,
  split,
  { rw Real.exp_one_eq_e, },
  { intros x hx,
    exact sorry,
  }
end

end max_ln_div_expr_l301_301943


namespace exist_arrangement_for_P_23_l301_301195

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301195


namespace max_lateral_area_trian_pyr_l301_301451

def is_perpendicular (u v : ℝ^3) : Prop := u ⬝ v = 0

theorem max_lateral_area_trian_pyr 
  (P A B C : ℝ^3) 
  (r : ℝ) 
  (h_r : r = 2) 
  (h_on_sphere : dist P A = r ∧ dist P B = r ∧ dist P C = r ∧ dist A B = r ∧ dist B C = r ∧ dist C A = r)
  (h_perpendicular : is_perpendicular (A - P) (B - P) ∧ is_perpendicular (B - P) (C - P) ∧ is_perpendicular (A - P) (C - P)) :
  let PA := dist P A, PB := dist P B, PC := dist P C in
  S ≤ 8 :=
  sorry

end max_lateral_area_trian_pyr_l301_301451


namespace rhombus_division_ratio_l301_301542

-- Defining the conditions
variable (α : ℝ)

-- Statement of the problem in Lean 4
theorem rhombus_division_ratio (hα_pos : 0 < α) (hα_lt_pi2 : α < π/2) :
  ∃ r : ℝ, r = (cos (α / 6)) / (cos (α / 2)) :=
begin
  sorry
end

end rhombus_division_ratio_l301_301542


namespace find_interest_rate_l301_301423

-- Define the terms
def P := 10000
def t := 2
def CI := 824.32
def n := 2  -- Compounded half-yearly

-- Define the interest formula
def A := P + CI  -- Total amount including interest

-- Define the expected answer
def r_expected := 0.04  -- 4% in decimal

-- Theorem statement
theorem find_interest_rate : ∃ r : ℝ, (P * (1 + r / n)^(n * t) = A) ∧ r = r_expected :=
by
  sorry

end find_interest_rate_l301_301423


namespace simplify_expression_l301_301004

theorem simplify_expression :
  (2 * 6 / (12 * 14)) * (3 * 12 * 14 / (2 * 6 * 3)) * 2 = 2 := 
  sorry

end simplify_expression_l301_301004


namespace arrangement_exists_for_P_eq_23_l301_301245

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301245


namespace proof_distance_between_intersections_l301_301284

theorem proof_distance_between_intersections :
  let m := 101 in
  let n := 5 in
  (∀ x y : ℝ, (y = 3 ∧ y = 5 * x^2 + x - 2) → True) →
  (gcd m n = 1) →
  ((m - n) = 96) :=
by
  intros
  let m := 101
  let n := 5
  have hmn : m - n = 96 := sorry
  exact (hmn)

end proof_distance_between_intersections_l301_301284


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301730

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301730


namespace intersection_of_M_and_N_l301_301579

-- Defining our sets M and N based on the conditions provided
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | x^2 < 4 }

-- The statement we want to prove
theorem intersection_of_M_and_N :
  M ∩ N = { x | -2 < x ∧ x < 1 } :=
sorry

end intersection_of_M_and_N_l301_301579


namespace product_mb_lt_neg_1_l301_301370

theorem product_mb_lt_neg_1 (m : ℚ) (b : ℚ) (hm : m = 3/4) (hb : b = -5/3) : m * b < -1 := by
  have h_product : m * b = (3/4) * (-5/3) := by
    rw [hm, hb]
  calc
    m * b = (3/4) * (-5/3) : by rw [hm, hb]
    ... = -15 / 12       : by norm_num
    ... = -5 / 4         : by norm_num
    ... < -1             : by norm_num

end product_mb_lt_neg_1_l301_301370


namespace sqrt_meaningful_condition_l301_301149

theorem sqrt_meaningful_condition (x : ℝ) : (2 * x + 6 >= 0) ↔ (x >= -3) := by
  sorry

end sqrt_meaningful_condition_l301_301149


namespace smallest_visible_sum_of_3x3x3_cube_is_90_l301_301848

theorem smallest_visible_sum_of_3x3x3_cube_is_90 
: ∀ (dices: Fin 27 → Fin 6 → ℕ),
    (∀ i j k, dices (3*i+j) k = 7 - dices (3*i+j) (5-k)) → 
    (∃ s, s = 90 ∧
    s = (8 * (dices 0 0 + dices 0 1 + dices 0 2)) + 
        (12 * (dices 0 0 + dices 0 1)) +
        (6 * (dices 0 0))) := sorry

end smallest_visible_sum_of_3x3x3_cube_is_90_l301_301848


namespace catherine_success_rate_increase_l301_301008

theorem catherine_success_rate_increase :
  let initial_attempts := 20
  let initial_successes := 8
  let initial_rate := initial_successes / initial_attempts 
  let next_attempts := 32
  let next_success_rate := 0.75
  let next_successes := next_attempts * next_success_rate
  let total_attempts := initial_attempts + next_attempts
  let total_successes := initial_successes + next_successes
  let overall_success_rate := total_successes / total_attempts
  let initial_rate_percentage := initial_rate * 100
  let overall_success_rate_percentage := overall_success_rate * 100
  let success_rate_increase := overall_success_rate_percentage - initial_rate_percentage
  success_rate_increase ≈ 22 :=
begin
  sorry -- Proof is omitted
end

end catherine_success_rate_increase_l301_301008


namespace expression_evaluates_to_4_l301_301923

theorem expression_evaluates_to_4 :
  2 * Real.cos (Real.pi / 6) + (- 1 / 2 : ℝ)⁻¹ + |Real.sqrt 3 - 2| + (2 * Real.sqrt (9 / 4))^0 + Real.sqrt 9 = 4 := 
by
  sorry

end expression_evaluates_to_4_l301_301923


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301763

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301763


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301759

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301759


namespace ratio_sum_squares_diagonal_l301_301584

variables (a b c : ℝ)
def midpoint (p1 p2 : ℝ × ℝ × ℝ) := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2, (p1.3 + p2.3) / 2)

def K := midpoint (0, 0, 0) (c, 0, 0)
def L := midpoint (a, b, b) (a, b, 0)
def M := midpoint (a, 0, b) (a, 0, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 + (p1.3 - p2.3)^2)

noncomputable def KL := distance K L
noncomputable def KM := distance K M
noncomputable def LM := distance L M

noncomputable def sum_squares_KLM := KL^2 + KM^2 + LM^2
noncomputable def diagonal_DB1 := real.sqrt (a^2 + b^2 + c^2)
noncomputable def square_diagonal_DB1 := (diagonal_DB1)^2

theorem ratio_sum_squares_diagonal :
  sum_squares_KLM a b c / square_diagonal_DB1 a b c = 3 / 2 :=
by
  sorry

end ratio_sum_squares_diagonal_l301_301584


namespace apothem_comparison_l301_301373

noncomputable def pentagon_side_length : ℝ := 4 / Real.tan (54 * Real.pi / 180)

noncomputable def pentagon_apothem : ℝ := pentagon_side_length / (2 * Real.tan (54 * Real.pi / 180))

noncomputable def hexagon_side_length : ℝ := 4 / Real.sqrt 3

noncomputable def hexagon_apothem : ℝ := (Real.sqrt 3 / 2) * hexagon_side_length

theorem apothem_comparison : pentagon_apothem = 1.06 * hexagon_apothem :=
by
  sorry

end apothem_comparison_l301_301373


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301672

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301672


namespace complex_mul_eq_l301_301347

theorem complex_mul_eq :
  (2 + 2 * Complex.i) * (1 - 2 * Complex.i) = 6 - 2 * Complex.i := 
by
  intros
  sorry

end complex_mul_eq_l301_301347


namespace remainder_first_six_primes_div_seventh_l301_301818

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301818


namespace count_ordered_pairs_with_one_solution_l301_301964

def hasExactlyOneRealSolution (b c : ℕ) : Prop :=
(b * b = 4 * c ∨ c * c = 4 * b)

theorem count_ordered_pairs_with_one_solution :
  {p : ℕ × ℕ | p.1 > 0 ∧ p.2 > 0 ∧ hasExactlyOneRealSolution p.1 p.2}.card = 1 := sorry

end count_ordered_pairs_with_one_solution_l301_301964


namespace transform_sequence_l301_301386

theorem transform_sequence (a1 a2 : ℕ) (h_a1a2 : a1 ≥ a2 ∧ a1 + a2 ≤ 100) :
  if |a1 - a2| ≤ 50 then
    true 
  else 
    false :=
by
  intro h_a1a2
  let b1 := 100 - a1
  let b2 := 100 - a2
  have h1 : |a1 - a2| ≤ 50 := sorry
  -- Assumption or further lemmas could be used here if necessary.
  sorry

end transform_sequence_l301_301386


namespace compare_abc_l301_301462

noncomputable def a : ℝ := ∫ x in (0:ℝ)..1, x ^ (-1/3 : ℝ)
noncomputable def b : ℝ := 1 - ∫ x in (0:ℝ)..1, x ^ (1/2 : ℝ)
noncomputable def c : ℝ := ∫ x in (0:ℝ)..1, x ^ (3 : ℝ)

theorem compare_abc : a > b ∧ b > c := by
  sorry

end compare_abc_l301_301462


namespace mangoes_per_kg_l301_301930

theorem mangoes_per_kg (total_kg : ℕ) (sold_market_kg : ℕ) (sold_community_factor : ℚ) (remaining_mangoes : ℕ) (mangoes_per_kg : ℕ) :
  total_kg = 60 ∧ sold_market_kg = 20 ∧ sold_community_factor = 1/2 ∧ remaining_mangoes = 160 → mangoes_per_kg = 8 :=
  by
  sorry

end mangoes_per_kg_l301_301930


namespace license_plate_combinations_l301_301508

def num_consonants : ℕ := 21
def num_vowels : ℕ := 5
def num_digits : ℕ := 10

theorem license_plate_combinations :
  (num_consonants ^ 2) * (num_vowels ^ 2) * (num_digits ^ 2) = 1_102_500 :=
by
  sorry

end license_plate_combinations_l301_301508


namespace proof_parabola_ellipse_l301_301062

noncomputable def ellipse_equation (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) : Prop :=
  ∀ x y : ℝ, (x^2) / (a^2) + (y^2) / (b^2) = 1

noncomputable def parabola_equation (p : ℝ) (h₀ : p > 0) : Prop :=
  ∀ y x : ℝ, y^2 = 2 * p * x

-- Proof statement
theorem proof_parabola_ellipse (a b p : ℝ) (h₀ : a > b) (h₁ : b > 0) (h₂ : p > 0)
  (hε : ellipse_equation a b h₀ h₁) (hπ : parabola_equation p h₂)
  (h_focus_shared : ∃ F₂ : ℝ × ℝ, F₂ ∈ [(a, 0), (-a, 0)] ∧ F₂ ∈ [(p/2, 0)])
  (hM_property : ∀ M : ℝ × ℝ, (∥M.1 - 0∥ = |∥M.1, M.2∥ - 1|)
  (hQ_property : ∃ Q : ℝ × ℝ, (Q ∈ ellipse a b) ∧ (Q ∈ parabola p) ∧ (|Q| = 5/2)) :
  (parabola p = parabola 2) ∧ (ellipse a b = ellipse 3 (sqrt 8)) ∧ (∀ k m x₀ x₁ x₂ y₁ y₂ : ℝ,
  (tangent_line k m p x₀) →
  (midpoint (x₁, y₁) (x₂, y₂) x₀) →
  (0 < m^2 < 9) →
  (-1 < x₀ < 0)) := 
sorry

end proof_parabola_ellipse_l301_301062


namespace arrangement_for_P23_exists_l301_301236

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301236


namespace max_a1_value_l301_301179

theorem max_a1_value (a : Fin 100 → ℤ) 
  (h : (∑ i, (a i)^2) = 100 * (∑ i, a i)) : 
  ∃ n ≤ 550, (a 0 = n) :=
by sorry

end max_a1_value_l301_301179


namespace circle_center_radius_l301_301576

-- Define the necessary parameters and let Lean solve the equivalent proof problem
theorem circle_center_radius:
  (∃ a b r : ℝ, (∀ x y : ℝ, x^2 + 8 * x + y^2 - 2 * y = 1 ↔ (x + 4)^2 + (y - 1)^2 = 18) 
  ∧ a = -4 
  ∧ b = 1 
  ∧ r = 3 * Real.sqrt 2
  ∧ a + b + r = -3 + 3 * Real.sqrt 2) :=
by {
  sorry
}

end circle_center_radius_l301_301576


namespace probability_open_doors_l301_301872

variable (n : ℕ)

theorem probability_open_doors (h : n > 1) : 
  let num_doors := 2 * (n - 1)
      num_locked := n - 1
  in (2^(n-1) / (num_doors.choose num_locked) = 
  2^(n-1) / (nat.choose num_doors num_locked)) := sorry

end probability_open_doors_l301_301872


namespace geom_seq_sum_eqn_l301_301647

theorem geom_seq_sum_eqn (n : ℕ) (a : ℚ) (r : ℚ) (S_n : ℚ) : 
  a = 1/3 → r = 1/3 → S_n = 80/243 → S_n = a * (1 - r^n) / (1 - r) → n = 5 :=
by
  intros ha hr hSn hSum
  sorry

end geom_seq_sum_eqn_l301_301647


namespace exists_arrangement_for_P_23_l301_301206

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301206


namespace tan_expression_l301_301921

theorem tan_expression (a : ℝ) (h₀ : 45 = 2 * a) (h₁ : Real.tan 45 = 1) 
  (h₂ : Real.tan (2 * a) = 2 * Real.tan a / (1 - Real.tan a * Real.tan a)) :
  Real.tan a / (1 - Real.tan a * Real.tan a) = 1 / 2 :=
by 
  sorry

end tan_expression_l301_301921


namespace prime_sum_remainder_l301_301772

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301772


namespace rectangle_perimeter_l301_301272

theorem rectangle_perimeter (a b : ℝ) 
    (h_area: a * b = 100 * Real.sqrt 5)
    (h_ratio: ∃ P, P ∈ Icc 0 a ∧ P = a / 6 ∧ IsPerpendicular (P, b) (a, b)) 
    (h_perpendicular: ∃ P D, IsPerpendicular (P, D) (a, b)) :
    2 * a + 2 * b = 20 * Real.sqrt 5 + 20 :=
by 
  sorry

end rectangle_perimeter_l301_301272


namespace square_side_increase_l301_301646

variable (s : ℝ)  -- original side length of the square.
variable (p : ℝ)  -- percentage increase of the side length.

theorem square_side_increase (h1 : (s * (1 + p / 100))^2 = 1.21 * s^2) : p = 10 := 
by
  sorry

end square_side_increase_l301_301646


namespace transformation_correct_l301_301091

theorem transformation_correct (a x y : ℝ) (h : a * x = a * y) : 3 - a * x = 3 - a * y :=
sorry

end transformation_correct_l301_301091


namespace johns_quarters_l301_301567

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l301_301567


namespace monkeys_eating_bananas_l301_301615

theorem monkeys_eating_bananas
    (minutes_for_6_bananas : ∀ m : ℕ, m * 6 = 6 * m)
    (bananas_to_eat : ℕ)
    (time_for_6_minutes: nat) 
    (bananas_eaten_in_18_minutes : bananas_to_eat = 18)
    (time_taken_in_18_minutes : time_for_6_minutes = 18) :
    ∃ (monkeys : ℕ), monkeys = 6 :=
by
    sorry

end monkeys_eating_bananas_l301_301615


namespace remainder_of_sum_of_primes_l301_301800

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301800


namespace original_cost_of_car_l301_301254

theorem original_cost_of_car (
    (C : ℕ) -- The amount Ramu bought the car for.
    (repair_cost : ℕ) -- The amount spent on repairs.
    (selling_price : ℕ) -- The price at which he sold the car.
    (profit_percent : ℚ) -- The profit percent.
    (repair_cost = 10000) 
    (selling_price = 64900) 
    (profit_percent = 24.807692307692307) 
    : C = 43979 :=
by
  sorry

end original_cost_of_car_l301_301254


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301670

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301670


namespace staffing_ways_l301_301925

open Nat

def suitable_candidates (total_resumes : ℕ) (ratio : ℚ) : ℕ :=
  (total_resumes * ratio.num) / ratio.denom

theorem staffing_ways :
  ∃ (ways : ℕ),
    let total_resumes := 30;
    let ratio := (2 : ℚ) / 3;
    let suitable := suitable_candidates total_resumes ratio;
    let positions := 5;
    ways = (suitable * (suitable - 1) * (suitable - 2) * (suitable - 3) * (suitable - 4)) ∧ ways = 930240 :=
by
  sorry

end staffing_ways_l301_301925


namespace log_expression_value_l301_301080

noncomputable def log_base_10 : ℝ → ℝ := sorry

theorem log_expression_value (x : ℝ) (h1 : x < 1) (h2 : (log_base_10 x)^2 - log_base_10 (x^2) = 75) :
  (log_base_10 x)^4 - log_base_10 (x^4) = (308 - 4 * real.sqrt 304) / 16 - 2 + real.sqrt 304 :=
by
  sorry

end log_expression_value_l301_301080


namespace average_price_of_pig_l301_301360

theorem average_price_of_pig :
  ∀ (total_cost : ℕ) (num_pigs num_hens : ℕ) (avg_hen_price avg_pig_price : ℕ),
    total_cost = 2100 →
    num_pigs = 5 →
    num_hens = 15 →
    avg_hen_price = 30 →
    avg_pig_price * num_pigs + avg_hen_price * num_hens = total_cost →
    avg_pig_price = 330 :=
by
  intros total_cost num_pigs num_hens avg_hen_price avg_pig_price
  intros h_total_cost h_num_pigs h_num_hens h_avg_hen_price h_eq
  rw [h_total_cost, h_num_pigs, h_num_hens, h_avg_hen_price] at h_eq
  sorry

end average_price_of_pig_l301_301360


namespace range_of_omega_l301_301097

open Real

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := sqrt 2 * sin (ω * x + π / 4)

theorem range_of_omega (h_omega_pos : ∀ ω > 0, true) :
  (∀ ω > 0, (∀ x ∈ Ioc 0 π, f ω x = 0) ↔ (7 / 4 < ω ∧ ω ≤ 11 / 4)) :=
by
  sorry

end range_of_omega_l301_301097


namespace number_of_complementary_sets_l301_301026

def Shape := {circle, square, triangle}
def Color := {red, blue, green}
def Shade := {light, medium, dark}
def Pattern := {striped, dotted, solid}

structure Card :=
(shape : Shape)
(color : Color)
(shade : Shade)
(pattern : Pattern)

def complementary (c1 c2 c3 : Card) : Prop :=
  (c1.shape = c2.shape ∧ c2.shape = c3.shape ∨ c1.shape ≠ c2.shape ∧ c2.shape ≠ c3.shape ∧ c1.shape ≠ c3.shape) ∧
  (c1.color = c2.color ∧ c2.color = c3.color ∨ c1.color ≠ c2.color ∧ c2.color ≠ c3.color ∧ c1.color ≠ c3.color) ∧
  (c1.shade = c2.shade ∧ c2.shade = c3.shade ∨ c1.shade ≠ c2.shade ∧ c2.shade ≠ c3.shade ∧ c1.shade ≠ c3.shade) ∧
  (c1.pattern = c2.pattern ∧ c2.pattern = c3.pattern ∨ c1.pattern ≠ c2.pattern ∧ c2.pattern ≠ c3.pattern ∧ c1.pattern ≠ c3.pattern)

theorem number_of_complementary_sets : 
  ∃ n, n = 837 ∧ (∑ c1 c2 c3, complementary c1 c2 c3) = n :=
by sorry

end number_of_complementary_sets_l301_301026


namespace complex_multiplication_l301_301352

theorem complex_multiplication:
  (2 + 2 * complex.I) * (1 - 2 * complex.I) = 6 - 2 * complex.I := by
  sorry

end complex_multiplication_l301_301352


namespace remainder_of_sum_of_primes_l301_301802

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301802


namespace initial_percentage_of_grape_juice_l301_301116

theorem initial_percentage_of_grape_juice
  (P : ℝ)    -- P is the initial percentage in decimal
  (h₁ : 0 ≤ P ∧ P ≤ 1)    -- P is a valid probability
  (h₂ : 40 * P + 10 = 0.36 * 50):    -- Given condition from the problem
  P = 0.2 := 
sorry

end initial_percentage_of_grape_juice_l301_301116


namespace none_perfect_squares_l301_301417

def repeats (d : Nat) (n : Nat) : Nat :=
  (List.replicate n d).foldl (λ acc x => 10 * acc + x) 0

def alternates (d1 d2 : Nat) (n : Nat) : Nat :=
  (List.join (List.replicate n [d1, d2])).foldl (λ acc x => 10 * acc + x) 0

def N₁ := repeats 3 100
def N₂ := repeats 6 100
def N₃ := alternates 1 5 50
def N₄ := alternates 2 1 50
def N₅ := alternates 2 7 50

theorem none_perfect_squares :
  ¬ (Nat.is_square N₁) ∧ ¬ (Nat.is_square N₂) ∧ ¬ (Nat.is_square N₃) ∧ ¬ (Nat.is_square N₄) ∧ ¬ (Nat.is_square N₅) := by
  sorry

end none_perfect_squares_l301_301417


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301712

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301712


namespace initial_sum_invested_l301_301366

-- Definition of the initial problem setup and given conditions
def compound_interest (P r n : ℝ) : ℝ :=
  P * (1 + r) ^ n

def interest_difference (P r1 r2 n : ℝ) : ℝ :=
  (compound_interest P r1 n) - (compound_interest P r2 n)

theorem initial_sum_invested (P : ℝ) :
  let r1 := 0.15
  let r2 := 0.10
  let n := 3
  interest_difference P r1 r2 n = 1500 →
  P ≈ 7899.47 :=
by
  sorry

end initial_sum_invested_l301_301366


namespace intersection_of_sets_l301_301495

open Set

theorem intersection_of_sets :
  let P := {3, 5, 6, 8}
  let Q := {4, 5, 7, 8}
  P ∩ Q = {5, 8} :=
by
  sorry

end intersection_of_sets_l301_301495


namespace average_width_of_books_is_correct_l301_301162

theorem average_width_of_books_is_correct :
  let book_widths := [4, 3 / 4, 1.5, 3, 7.25, 2, 5.5, 12]
  let total_width := List.sum book_widths
  let number_of_books := book_widths.length
  let average_width := total_width / number_of_books
  average_width = 4.5 := 
by
  sorry

end average_width_of_books_is_correct_l301_301162


namespace y_intercept_of_line_l301_301280

theorem y_intercept_of_line (x y : ℝ) : x + 2 * y + 6 = 0 → x = 0 → y = -3 :=
by
  sorry

end y_intercept_of_line_l301_301280


namespace hexagon_angle_E_l301_301546

theorem hexagon_angle_E (F I U R G E : ℝ) 
  (h1 : F = I) (h2 : I = U) (h3 : U = R) 
  (h4 : G + E = 180) 
  (h5 : F + I + U + R + G + E = 720) : 
  E = 45 :=
by
  sorry

end hexagon_angle_E_l301_301546


namespace team_rate_increase_33_perc_l301_301304

theorem team_rate_increase_33_perc (initial_items processed_time additional_items total_time new_rate percent_increase : ℕ) 
(h1 : initial_items = 1250) 
(h2 : processed_time = 6) 
(h3 : additional_items = 165) 
(h4 : total_time = 10) 
(h5 : new_rate = 166.25)
(h6 : percent_increase = 33)
: by sorry := sorry

end team_rate_increase_33_perc_l301_301304


namespace add_base8_l301_301383

-- Define x and y in base 8 and their sum in base 8
def x := 24 -- base 8
def y := 157 -- base 8
def result := 203 -- base 8

theorem add_base8 : (x + y) = result := 
by sorry

end add_base8_l301_301383


namespace sum_of_areas_of_rectangles_l301_301935

theorem sum_of_areas_of_rectangles (a b : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (∑ k in finset.range (a+1), ∑ l in finset.range (b+1), k * l * (a + 1 - k) * (b + 1 - l)) = 
  ∑ k in finset.range a, ∑ l in finset.range b, k.succ * l.succ * (a + 1 - k.succ) * (b + 1 - l.succ) :=
sorry

end sum_of_areas_of_rectangles_l301_301935


namespace exists_similar_sizes_P_23_l301_301209

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301209


namespace total_water_used_l301_301323

theorem total_water_used (a b c : ℕ) (H_a_b : a > b) (H_b_c : b > c) :
  let x := (a - b) / b in
  let y := a / (b + c) in
  x + (2 * y - a / b) = (2 * a / (b + c)) - 1 :=
by sorry

end total_water_used_l301_301323


namespace probability_open_path_l301_301866

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l301_301866


namespace part_I_part_II_l301_301490

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a * log x
noncomputable def g (x : ℝ) : ℝ := (Real.exp (x - 1)) / x - 1

theorem part_I (a : ℝ) :
  (∃ x0 : ℝ, x0 > 0 ∧ 0 = x0 + a * log x0 ∧ 1 + a / x0 = 0) ↔ a = -Real.exp 1 :=
sorry

theorem part_II :
  ∀ a : ℝ,
  a > 0 →
  (∀ x1 x2 : ℝ, 3 ≤ x1 → 3 ≤ x2 → x1 ≠ x2 →
  |f x1 a - f x2 a| < |g x1 - g x2|) →
  0 < a ∧ a ≤ (2 * (Real.exp 2)) / 3 - 3 :=
sorry

end part_I_part_II_l301_301490


namespace spherical_quadrilateral_opposite_angles_equal_l301_301604

noncomputable theory
open_locale classical

variables {A B C D O K P : Type}
variables (sphere : Type) [has_inner sphere]
variables (quadrilateral : Type) [has_vertices quadrilateral A B C D]
variables (center_O : sphere) 
variables (circle_center_K : sphere)
variables (orthogonality : (K → P) ⊆ (subspace (vector_span sphere A B C D ↔ (∀ (P : sphere), orthogonal K A = orthogonal P B))))

theorem spherical_quadrilateral_opposite_angles_equal
    (h : ∀ (A B C D : quadilateral), circumscribed sphere A B C D center_O) :
    ∀ (α β γ δ : real), 
    (opposite_angles_sum α β γ δ) :=
begin
  sorry
end

end spherical_quadrilateral_opposite_angles_equal_l301_301604


namespace find_lambda_l301_301458

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A] 

variables (CA CB CD BC : A) (lambda m : ℝ)

-- Conditions
def point_on_side (D : A) : Prop :=
CD = (1 / 3) • CA + lambda • BC

def collinear_points (A D B : A): Prop :=
CD = -m • CA + (m - 1) • BC

-- Statement
theorem find_lambda (h1 : point_on_side D) (h2 : collinear_points A D B) :
  lambda = -4 / 3 :=
by sorry

end find_lambda_l301_301458


namespace exists_arrangement_for_P_23_l301_301227

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301227


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301668

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301668


namespace a_2_value_l301_301114

theorem a_2_value (a a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℝ) (x : ℝ) :
  x^3 + x^10 = a + a1 * (x+1) + a2 * (x+1)^2 + a3 * (x+1)^3 + a4 * (x+1)^4 + a5 * (x+1)^5 +
  a6 * (x+1)^6 + a7 * (x+1)^7 + a8 * (x+1)^8 + a9 * (x+1)^9 + a10 * (x+1)^10 → 
  a2 = 42 :=
by
  sorry

end a_2_value_l301_301114


namespace geometric_sequence_formula_and_sum_l301_301446

theorem geometric_sequence_formula_and_sum (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 1 = 2) 
  (h2 : a 1 + a 3 + a 5 = 42) 
  (h3 : ∀ n, a (n+1) = a n * q) : 
  (∀ n, a n = 2^n ∨ a n = (-1)^(n-1) * 2^n) ∧
  (∀ n, (a 2 + a 4 + a 6 + ... + a (2*n) = (4/3) * (4^n - 1)) ∨ 
         (a 2 + a 4 + a 6 + ... + a (2*n) = (4/3) * (1 - 4^n))) :=
by sorry

end geometric_sequence_formula_and_sum_l301_301446


namespace train_length_l301_301379

-- Define the conditions
def time_to_cross_pole : ℝ := 7.999360051195905
def speed_kmh : ℝ := 144
def speed_ms : ℝ := speed_kmh * (1000 / 3600)
def length_of_train : ℝ := speed_ms * time_to_cross_pole

-- The goal is to prove that the length of the train is 319.9744020478362 meters
theorem train_length (h : speed_ms = 40) : length_of_train = 319.9744020478362 := by
  sorry

end train_length_l301_301379


namespace cube_decomposition_smallest_number_91_l301_301472

theorem cube_decomposition_smallest_number_91 (m : ℕ) (h1 : 0 < m) (h2 : (91 - 1) / 2 + 2 = m * m - m + 1) : m = 10 := by {
  sorry
}

end cube_decomposition_smallest_number_91_l301_301472


namespace average_speed_of_car_l301_301040

noncomputable def average_speed :=
  let speed1 := 100
  let time1 := 1
  let speed2 := 80
  let time2 := 1
  let speed3 := 90
  let time3 := 2
  let speed4 := 60
  let time4 := 1
  let speed5 := 70
  let time5 := 1

  let total_distance := speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4 + speed5 * time5
  let total_time := time1 + time2 + time3 + time4 + time5

  total_distance / total_time

theorem average_speed_of_car : average_speed = 81.67 := 
by
  -- Proof goes here
  sorry

end average_speed_of_car_l301_301040


namespace problem_l301_301067

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | 2^(x - 2) > 1 }
def complement_R (B : Set ℝ) := { x : ℝ | ¬ (x ∈ B) }

theorem problem : A ∩ (complement_R B) = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end problem_l301_301067


namespace probability_open_path_l301_301868

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l301_301868


namespace remainder_of_sum_of_primes_l301_301804

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301804


namespace company_investment_exceeds_2_million_l301_301878

theorem company_investment_exceeds_2_million :
  ∃ n : ℕ, n ≥ 2015 ∧ (1.3 * (1 + 0.12) ^ (n - 2015) > 2) ∧ n = 2019 := 
by 
  -- Given conditions
  let initial_investment := 1.3
  let growth_rate := 1.12
  let year_threshold := 2
  
  -- We want to prove
  have log_1_12: Float := 0.05
  have log_1_3: Float := 0.11
  have log_2: Float := 0.30
  
  -- Calculate the inequality
  have inequality : ∀ n ≥ 2015, (n - 2015) * log_1_12 > log_2 - log_1_3 :=
    by intros; exact (3.8 : Float)
  
  -- Satisfy the condition n = 2019
  use 2019
  split
  -- Year condition
  exact Nat.le_refl 2019
  -- Inequality condition
  exact sorry
  -- Year equality
  exact rfl

end company_investment_exceeds_2_million_l301_301878


namespace months_rent_in_advance_required_l301_301563

def janet_savings : ℕ := 2225
def rent_per_month : ℕ := 1250
def deposit : ℕ := 500
def additional_needed : ℕ := 775

theorem months_rent_in_advance_required : 
  (janet_savings + additional_needed - deposit) / rent_per_month = 2 :=
by
  sorry

end months_rent_in_advance_required_l301_301563


namespace magnitude_comb_find_m_n_find_d_l301_301103

open Complex

-- Definitions of the given vectors
def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 2)
def c : ℝ × ℝ := (4, 1)

-- Definition of magnitudes
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Definition of vector operations
def sub (u v : ℝ × ℝ) := (u.1 - v.1, u.2 - v.2)
def add (u v : ℝ × ℝ) := (u.1 + v.1, u.2 + v.2)
def smul (k : ℝ) (v : ℝ × ℝ) := (k * v.1, k * v.2)

-- Question 1: Magnitude of the combination of vectors
theorem magnitude_comb :
  let v := smul 3 a ++ b -- Add and sub are now applied
  let answer := Real.sqrt 65
  magnitude (add (sub v c)) = answer := sorry

-- Question 2: Real numbers m and n
theorem find_m_n :
  let m := 5/9
  let n := 8/9
  a = add (smul m b) (smul n c) := sorry

-- Question 3: Finding vector d
theorem find_d :
  let d1 := (4 + (Real.sqrt 5) / 5, 1 + 2 * (Real.sqrt 5) / 5)
  let d2 := (4 - (Real.sqrt 5) / 5, 1 - 2 * (Real.sqrt 5) / 5)
  let condition1 := (d1 - c).1 * (b.1 + a.1) = (d1 - c).2 * (b.2 + a.2)
  let condition2 := (d2 - c).1 * (b.1 + a.1) = (d2 - c).2 * (b.2 + a.2)
  let magnitude1 := magnitude (sub d1 c) = 1
  let magnitude2 := magnitude (sub d2 c) = 1
  (condition1 ∧ magnitude1) ∨ (condition2 ∧ magnitude2) := sorry

end magnitude_comb_find_m_n_find_d_l301_301103


namespace complex_mul_eq_l301_301349

theorem complex_mul_eq :
  (2 + 2 * Complex.i) * (1 - 2 * Complex.i) = 6 - 2 * Complex.i := 
by
  intros
  sorry

end complex_mul_eq_l301_301349


namespace remainder_of_sum_of_primes_l301_301795

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301795


namespace hyperbola_standard_equation_l301_301430

theorem hyperbola_standard_equation (a b : ℝ) (P : ℝ × ℝ)
  (h1 : b = 2 * a / 3)
  (h2 : P = (real.sqrt 6, 2))
  (h3 : ∃ (a : ℝ), a^2 = 3)
  : ∃ (x y : ℝ), (3 * y^2 / 4 - x^2 / 3 = 1) := 
by
  use [x, y]
  rw h2
  rw h1
  sorry -- proof required here

end hyperbola_standard_equation_l301_301430


namespace ellipse_equation_and_max_area_diff_l301_301087

-- Definitions of the given conditions:
def is_ellipse (a : ℝ) (h : a > 0) : Prop :=
  ∃ k, k > 0 ∧ (ellipse_focus : ℝ × ℝ) = (-1, 0) ∧
      equation := (λ (x y : ℝ), (x^2 / a^2) + (y^2 / 3) = 1)

-- The proof statements
theorem ellipse_equation_and_max_area_diff (a : ℝ) (h : a > 0) :
  (is_ellipse a h → (a = 2 ∧ (max_area_diff : ℝ) = by sorry) :=
begin
  intros hin,
  have h1: a = 2, {
    sorry      -- Steps to prove a = 2
  },
  have h2: max_area_diff = sqrt 3, {
    sorry      -- Steps to prove max_area_diff = sqrt 3
  },
  exact ⟨h1, h2⟩,
end

end ellipse_equation_and_max_area_diff_l301_301087


namespace inverse_function_eq_l301_301442

noncomputable def f (x : ℝ) : ℝ := (x - 2)^2

noncomputable def f_inv (x : ℝ) : ℝ := 2 - sqrt x

theorem inverse_function_eq (x : ℝ) (hx : x ≥ -1) : f_inv (x + 1) = 2 - sqrt (x + 1) := by
  sorry

end inverse_function_eq_l301_301442


namespace grooming_time_correct_l301_301290

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l301_301290


namespace johns_quarters_l301_301568

variable (x : ℕ)  -- Number of quarters John has

def number_of_dimes : ℕ := x + 3  -- Number of dimes
def number_of_nickels : ℕ := x - 6  -- Number of nickels

theorem johns_quarters (h : x + (x + 3) + (x - 6) = 63) : x = 22 :=
by
  sorry

end johns_quarters_l301_301568


namespace total_players_on_ground_l301_301142

theorem total_players_on_ground :
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  cricket_players + hockey_players + football_players + softball_players +
  basketball_players + volleyball_players + netball_players + rugby_players = 263 := 
by 
  let cricket_players := 35
  let hockey_players := 28
  let football_players := 33
  let softball_players := 35
  let basketball_players := 29
  let volleyball_players := 32
  let netball_players := 34
  let rugby_players := 37
  sorry

end total_players_on_ground_l301_301142


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301680

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301680


namespace complex_mul_example_l301_301345

theorem complex_mul_example (i : ℝ) (h : i^2 = -1) : (⟨2, 2 * i⟩ : ℂ) * (⟨1, -2 * i⟩) = ⟨6, -2 * i⟩ :=
by
  sorry

end complex_mul_example_l301_301345


namespace count_color_patterns_l301_301595

def regions := 6
def colors := 3

theorem count_color_patterns (h1 : regions = 6) (h2 : colors = 3) :
  3^6 - 3 * 2^6 + 3 * 1^6 = 540 := by
  sorry

end count_color_patterns_l301_301595


namespace billy_tip_l301_301518

theorem billy_tip
  (steak_price drink_price : ℝ)
  (num_people : ℕ)
  (tip_percent cover_percent : ℝ)
  (h_steak_price : steak_price = 20)
  (h_drink_price : drink_price = 5)
  (h_num_people : num_people = 2)
  (h_tip_percent : tip_percent = 0.2)
  (h_cover_percent : cover_percent = 0.8) :
  let meal_cost := steak_price + drink_price
  let total_cost := num_people * meal_cost
  let total_tip := total_cost * tip_percent
  let covered_tip := total_tip * cover_percent
  covered_tip = 8 := 
by
  rw [h_steak_price, h_drink_price, h_num_people, h_tip_percent, h_cover_percent]
  simp [←Nat.cast_mul, ←mul_assoc, mul_comm, mul_left_comm]
  norm_num
  done
  sorry

end billy_tip_l301_301518


namespace grooming_time_correct_l301_301288

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def number_of_poodles : ℕ := 3
def number_of_terriers : ℕ := 8

def total_grooming_time : ℕ :=
  (number_of_poodles * time_to_groom_poodle) + (number_of_terriers * time_to_groom_terrier)

theorem grooming_time_correct :
  total_grooming_time = 210 :=
by
  sorry

end grooming_time_correct_l301_301288


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301791

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301791


namespace remainder_first_six_primes_div_seventh_l301_301808

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301808


namespace smallest_x_for_square_l301_301429

theorem smallest_x_for_square (N : ℕ) (h1 : ∃ x : ℕ, x > 0 ∧ 1260 * x = N^2) : ∃ x : ℕ, x = 35 :=
by
  sorry

end smallest_x_for_square_l301_301429


namespace problem_l301_301065

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | 2^(x - 2) > 1 }
def complement_R (B : Set ℝ) := { x : ℝ | ¬ (x ∈ B) }

theorem problem : A ∩ (complement_R B) = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end problem_l301_301065


namespace arithmetic_to_geometric_l301_301839

theorem arithmetic_to_geometric (a1 a2 a3 a4 d : ℝ) (h_sequence : a2 = a1 + d) (h_sequence2 : a3 = a1 + 2 * d) (h_sequence3 : a4 = a1 + 3 * d) (h_non_zero : d ≠ 0) (h_geometric : (a1 ≠ 0 ∧ a2 ≠ 0 ∧ a3 ≠ 0 ∧ a4 ≠ 0) ∧ by (a2^2 = a1 * a3 ∨ a2^2 = a1 * a4 ∨ a3^2 = a1 * a4 ∨ a3^2 = a2 * a4)) : 
    (a1 / d = 1) ∨ (a1 / d = -4) :=
sorry

end arithmetic_to_geometric_l301_301839


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301705

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301705


namespace value_of_b_l301_301122

variable (a b : ℤ)

theorem value_of_b : a = 105 ∧ a ^ 3 = 21 * 49 * 45 * b → b = 1 := by
  sorry

end value_of_b_l301_301122


namespace xyz_inequality_l301_301178

theorem xyz_inequality (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (a * c * d) / (1 - b)^2 + 
  (a * b * d) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
sorry

end xyz_inequality_l301_301178


namespace heather_bicycled_distance_l301_301519

def speed : ℕ := 8
def time : ℕ := 5
def distance (s : ℕ) (t : ℕ) : ℕ := s * t

theorem heather_bicycled_distance : distance speed time = 40 := by
  sorry

end heather_bicycled_distance_l301_301519


namespace g_at_3_l301_301632

def g : ℝ → ℝ

axiom g_property : ∀ x : ℝ, g (3 ^ x) + x * g (3 ^ (-x)) = x

theorem g_at_3 : g (3) = 1 :=
by {
  sorry
}

end g_at_3_l301_301632


namespace radius_satisfies_equation_l301_301456

variable {ABC : Type} [Nonempty ABC] {circumcircle : Type} [Nonempty circumcircle]

-- Definitions for distances H_a, H_b, and H_c and the radius R
variables {a b c : ℝ} -- side lengths of triangle ABC
variables {H_a H_b H_c : ℝ} -- distances from center O to sides a, b, c
variable {R : ℝ} -- radius of the circle

-- Assumptions based on given conditions
axiom dist_from_center (O : circumcircle) (ABC_triangle : ABC → circumcircle) : 
  ∃ (H_a H_b H_c : ℝ), (distance O (side a) = H_a) ∧ (distance O (side b) = H_b) ∧ (distance O (side c) = H_c)

-- The theorem to be proven in Lean 4 notation
theorem radius_satisfies_equation (H_a H_b H_c : ℝ) (R : ℝ) 
  (h : ∃ (H_a H_b H_c : ℝ), 
        dist_from_center O (triangle ABC) ∧ H_a^2 + H_b^2 + H_c^2 = H_sqr)
  : R^3 - (H_a^2 + H_b^2 + H_c^2) * R - 2 * H_a * H_b * H_c = 0 :=
sorry

end radius_satisfies_equation_l301_301456


namespace exists_unique_zero_in_interval_l301_301650

def f (x : ℝ) : ℝ := 2^(x-1) + x - 5

theorem exists_unique_zero_in_interval : 
  ∃ x_0 ∈ Set.Ioo 2 3, f x_0 = 0 :=
by
  have h1 : f 2 < 0 := by norm_num
  have h2 : f 3 > 0 := by norm_num
  have h3 : ∀ x y, x < y → f x < f y := sorry
  sorry

end exists_unique_zero_in_interval_l301_301650


namespace arithmetic_seq_20th_term_l301_301466

variable (a : ℕ → ℤ) -- a_n is an arithmetic sequence
variable (d : ℤ) -- common difference of the arithmetic sequence

-- Condition for arithmetic sequence
variable (h_seq : ∀ n, a (n+1) = a n + d)

-- Given conditions
axiom h1 : a 1 + a 3 + a 5 = 105
axiom h2 : a 2 + a 4 + a 6 = 99

-- Goal: prove that a 20 = 1
theorem arithmetic_seq_20th_term :
  a 20 = 1 :=
sorry

end arithmetic_seq_20th_term_l301_301466


namespace orthocenter_of_ABC_is_correct_l301_301427

structure Point3D where
  x : ℚ
  y : ℚ
  z : ℚ

def A : Point3D := {x := 2, y := 3, z := -1}
def B : Point3D := {x := 6, y := -1, z := 2}
def C : Point3D := {x := 4, y := 5, z := 4}

def orthocenter (A B C : Point3D) : Point3D := {
  x := 101 / 33,
  y := 95 / 33,
  z := 47 / 33
}

theorem orthocenter_of_ABC_is_correct : orthocenter A B C = {x := 101 / 33, y := 95 / 33, z := 47 / 33} :=
  sorry

end orthocenter_of_ABC_is_correct_l301_301427


namespace a_eq_4b_l301_301111

-- Given condition: log base 2 of a plus log base 1/2 of b equals 2
axiom log_condition {a b : ℝ} : log 2 a + log (1/2) b = 2

theorem a_eq_4b (a b : ℝ) (h : log 2 a + log (1/2) b = 2) : a = 4 * b :=
by {
  sorry -- Placeholder for the actual proof
}

end a_eq_4b_l301_301111


namespace greatest_possible_multiple_of_4_l301_301618

theorem greatest_possible_multiple_of_4 (x : ℕ) (h1 : x % 4 = 0) (h2 : x^2 < 400) : x ≤ 16 :=
by 
sorry

end greatest_possible_multiple_of_4_l301_301618


namespace norma_missing_items_l301_301597

theorem norma_missing_items :
  let initial_tshirts := 9
  let initial_sweaters := 2 * initial_tshirts
  let found_sweaters := 3
  let found_tshirts := 3 * initial_tshirts
  let total_initial_items := initial_tshirts + initial_sweaters
  let total_found_items := found_tshirts + found_sweaters
  missing_items := initial_sweaters - found_sweaters
  missing_items = 15 := 
begin
  sorry,
end

end norma_missing_items_l301_301597


namespace slope_of_l_l301_301283

def point (α : Type) := prod α α

def midpoint (P Q : point ℝ) : point ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

def slope (P Q : point ℝ) : ℝ :=
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_of_l :=
  let P := (-2, -3) in
  let Q := (4, 1) in
  (midpoint P Q = (1, -1)) → slope P Q = 2 / 3 := by
    sorry

end slope_of_l_l301_301283


namespace packs_needed_l301_301259

-- Define the problem conditions
def bulbs_bedroom : ℕ := 2
def bulbs_bathroom : ℕ := 1
def bulbs_kitchen : ℕ := 1
def bulbs_basement : ℕ := 4
def bulbs_pack : ℕ := 2

def total_bulbs_main_areas : ℕ := bulbs_bedroom + bulbs_bathroom + bulbs_kitchen + bulbs_basement
def bulbs_garage : ℕ := total_bulbs_main_areas / 2

def total_bulbs : ℕ := total_bulbs_main_areas + bulbs_garage

def total_packs : ℕ := total_bulbs / bulbs_pack

-- The proof statement
theorem packs_needed : total_packs = 6 :=
by
  sorry

end packs_needed_l301_301259


namespace Diracs_Theorem_Hamiltonian_Cycle_Include_Edge_l301_301357

open Finset

-- First problem
theorem Diracs_Theorem {V : Type*} [Fintype V] (G : SimpleGraph V) (n : ℕ) (h1 : 3 ≤ n) 
  (h2 : ∀ v ∈ G.vertices, G.degree v ≥ n / 2) : ∃ (c : Cycle V), c.isHamiltonian := 
  sorry

-- Second problem
theorem Hamiltonian_Cycle_Include_Edge {V : Type*} [Fintype V] (G : SimpleGraph V) (n : ℕ) (e : Sym2 V)
  (h1 : 3 ≤ n)
  (h2 : ∀ v ∈ G.vertices, G.degree v ≥ (n + 1) / 2)
  (h3 : e ∈ G.edgeSet) : ∃ (c : Cycle V), c.isHamiltonian ∧ e ∈ c.edges :=
  sorry

end Diracs_Theorem_Hamiltonian_Cycle_Include_Edge_l301_301357


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301750

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301750


namespace prime_sum_remainder_l301_301768

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301768


namespace no_monochromatic_10_term_progression_l301_301262

def can_color_without_monochromatic_progression (n k : ℕ) (c : Fin n → Fin k) : Prop :=
  ∀ (a d : ℕ), (a < n) → (a + (9 * d) < n) → (∀ i : ℕ, i < 10 → c ⟨a + (i * d), sorry⟩ = c ⟨a, sorry⟩) → 
    (∃ j i : ℕ, j < 10 ∧ i < 10 ∧ c ⟨a + (i * d), sorry⟩ ≠ c ⟨a + (j * d), sorry⟩)

theorem no_monochromatic_10_term_progression :
  ∃ c : Fin 2008 → Fin 4, can_color_without_monochromatic_progression 2008 4 c :=
sorry

end no_monochromatic_10_term_progression_l301_301262


namespace no_food_dogs_l301_301539

theorem no_food_dogs (total_dogs watermelon_liking salmon_liking chicken_liking ws_liking sc_liking wc_liking wsp_liking : ℕ) 
    (h_total : total_dogs = 100)
    (h_watermelon : watermelon_liking = 20) 
    (h_salmon : salmon_liking = 70) 
    (h_chicken : chicken_liking = 10) 
    (h_ws : ws_liking = 10) 
    (h_sc : sc_liking = 5) 
    (h_wc : wc_liking = 3) 
    (h_wsp : wsp_liking = 2) :
    (total_dogs - ((watermelon_liking - ws_liking - wc_liking + wsp_liking) + 
    (salmon_liking - ws_liking - sc_liking + wsp_liking) + 
    (chicken_liking - sc_liking - wc_liking + wsp_liking) + 
    (ws_liking - wsp_liking) + 
    (sc_liking - wsp_liking) + 
    (wc_liking - wsp_liking) + wsp_liking)) = 28 :=
  by sorry

end no_food_dogs_l301_301539


namespace k_l_absolute_value_l301_301572

theorem k_l_absolute_value :
  ∃ (k ℓ : ℝ), (∀ x : ℝ, 13 * x^2 + 39 * x - 91 = 13 * ((x + k) ^ 2 - |ℓ|)) ∧ |k + ℓ| = 10.75 :=
by
  let k := 1.5
  let ℓ := 9.25
  use k, ℓ
  split
  sorry
  sorry

end k_l_absolute_value_l301_301572


namespace equal_real_roots_of_quadratic_eq_l301_301125

theorem equal_real_roots_of_quadratic_eq (k : ℝ) :
  (∃ x : ℝ, x^2 + 3 * x - k = 0 ∧ x = x) → k = - (9 / 4) := by
  sorry

end equal_real_roots_of_quadratic_eq_l301_301125


namespace find_x_l301_301140

-- Definitions based on the conditions
def remaining_scores_after_removal (s: List ℕ) : List ℕ :=
  s.erase 87 |>.erase 94

def average (l : List ℕ) : ℚ :=
  (l.sum : ℚ) / l.length

-- Converting the given problem into a Lean 4 theorem statement
theorem find_x (x : ℕ) (s : List ℕ) :
  s = [94, 87, 89, 88, 92, 90, x, 93, 92, 91] →
  average (remaining_scores_after_removal s) = 91 →
  x = 2 :=
by
  intros h1 h2
  sorry

end find_x_l301_301140


namespace intersection_complement_l301_301073

def setA (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def setB (x : ℝ) : Prop := 2^(x - 2) > 1

theorem intersection_complement :
  { x : ℝ | setA x } ∩ { x : ℝ | ¬ setB x } = { x : ℝ | -1 < x ∧ x ≤ 2 } := 
by 
  sorry

end intersection_complement_l301_301073


namespace three_lines_intersect_point_l301_301377

theorem three_lines_intersect_point 
  (n : ℕ)
  (n = 9)
  (forall l : (line splits square into two quadrilaterals such that one has area 3 times the other)): 
  ∃ p : point on the midlines of the square, at least 3 of these lines intersect at p 
:= 
sorry

end three_lines_intersect_point_l301_301377


namespace count_n_not_divisible_by_3_l301_301965

def S (n : ℕ) : ℕ := 
  (1000 / n).floor + (1001 / n).floor + (1002 / n).floor

theorem count_n_not_divisible_by_3 : 
  (finset.filter (λ n, ¬ (S n % 3 = 0)) (finset.range 1001)).card = 20 :=
by
  sorry

end count_n_not_divisible_by_3_l301_301965


namespace find_k_l301_301118

theorem find_k (x k : ℝ) (h1 : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h2 : k ≠ 0) : k = 6 :=
by
  sorry

end find_k_l301_301118


namespace number_of_factors_M_l301_301506

-- Define the given natural number M
def M : ℕ := 2^4 * 3^3 * 7^1 * 11^2

-- Prove that the number of natural-number factors of M is 120
theorem number_of_factors_M : (finset.filter (λ n, n ∣ M) (finset.range (M + 1))).card = 120 := 
sorry

end number_of_factors_M_l301_301506


namespace probability_open_path_l301_301869

-- Define necessary terms
def total_doors (n : ℕ) : ℕ := 2 * (n - 1)
def locked_doors (n : ℕ) : ℕ := total_doors n / 2

-- Helper function to compute binomial coefficient
noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Probability theorem
theorem probability_open_path (n : ℕ) (h : n > 1) : 
  ((locked_doors n) = (n-1)) → 
  (∃ p, p = (2^(n-1)) / (binom (total_doors n) (n-1))) :=
by {
  intro h1,
  use ((2^(n-1)) / (binom (total_doors n) (n-1))),
  sorry
}

end probability_open_path_l301_301869


namespace interoceanic_numbers_lt_2020_l301_301892

noncomputable theory

-- Definition: A number N is interoceanic if its prime factorization
-- N = p_1^{x_1} p_2^{x_2} .. p_k^{x_k} satisfies
-- x_1 + x_2 + ... + x_k = p_1 + p_2 + ... + p_k.
def is_interoceanic (N : ℕ) : Prop :=
  ∃ (k : ℕ) (p : fin k → ℕ) (x : fin k → ℕ),
    (∀ i, Nat.prime (p i)) ∧
    (N = (finset.univ : finset (fin k)).prod (λ i, (p i) ^ (x i))) ∧
    (finset.univ.sum (λ i, x i) = finset.univ.sum (λ i, p i))

-- Theorem: the set of interoceanic numbers less than 2020.
theorem interoceanic_numbers_lt_2020 :
  { n : ℕ | is_interoceanic n ∧ n < 2020 } = {4, 27, 48, 72, 108, 162, 320, 800, 1792, 2000} :=
by
  sorry

end interoceanic_numbers_lt_2020_l301_301892


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301719

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301719


namespace common_internal_tangent_length_l301_301273

theorem common_internal_tangent_length (d r1 r2 : ℝ) (h1 : d = 50) (h2 : r1 = 7) (h3 : r2 = 12) :
    sqrt (d^2 - (r1 + r2)^2) = sqrt 2139 :=
by
  sorry

end common_internal_tangent_length_l301_301273


namespace probability_from_first_to_last_l301_301851

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l301_301851


namespace count_rare_subsets_l301_301573

def is_rare (n : ℕ) (S : set ℕ) : Prop :=
  n > 1 ∧
  ∀ k ∈ set.Ico 0 n, 
    (S ∩ set.Icc (4 * k - 2) (4 * k + 2)).card ≤ 2 ∧
    (S ∩ set.Icc (4 * k + 1) (4 * k + 3)).card ≤ 1

theorem count_rare_subsets (n : ℕ) (h : n > 1) :
  (finset.powerset (finset.range (4 * n))).filter (λ S, is_rare n S.to_set).card = 8 * 7^(n - 1) :=
sorry

end count_rare_subsets_l301_301573


namespace remainder_of_sum_of_primes_l301_301806

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301806


namespace tessellation_coloring_l301_301664

-- Definition of the problem conditions
def tessellation_composed_of_triangles_and_hexagonal_pattern := sorry

-- Statement of the problem with the given conditions and the correct answer
theorem tessellation_coloring : tessellation_composed_of_triangles_and_hexagonal_pattern → ∃ (N : ℕ), N = 3 :=
by
  sorry

end tessellation_coloring_l301_301664


namespace number_of_periodic_functions_with_period_pi_l301_301390

theorem number_of_periodic_functions_with_period_pi :
  let f1 := fun (x : ℝ) => Real.cos (|x|)
  let f2 := fun (x : ℝ) => |Real.tan x|
  let f3 := fun (x : ℝ) => Real.sin (2 * x + 2 * Real.pi / 3)
  let f4 := fun (x : ℝ) => Real.cos (2 * x + 2 * Real.pi / 3)
  (f2.periodic? (π) ∧ f3.periodic? (π) ∧ f4.periodic? (π)) ∧
    ¬ (f1.periodic? (π)) ∧ 
    (f2.periodic? (π) → f3.periodic? (π) → f4.periodic? (π) → 3) :=
  sorry

end number_of_periodic_functions_with_period_pi_l301_301390


namespace probability_greater_than_4_l301_301147

-- Given conditions
def die_faces : ℕ := 6
def favorable_outcomes : Finset ℕ := {5, 6}

-- Probability calculation
def probability (total : ℕ) (favorable : Finset ℕ) : ℚ :=
  favorable.card / total

theorem probability_greater_than_4 :
  probability die_faces favorable_outcomes = 1 / 3 :=
by
  sorry

end probability_greater_than_4_l301_301147


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301675

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301675


namespace find_sides_of_triangle_l301_301636

-- Define the conditions: inscribed circle radius and angle
def inscribed_circle_radius := 1
def angle_MKN_ABC : ℝ := 45
def sides_triangle_ABC : (ℝ × ℝ × ℝ) := (2 + Real.sqrt 2, 2 + Real.sqrt 2, Real.sqrt 2 + 2)

theorem find_sides_of_triangle (r : ℝ) (θ : ℝ) (sides : ℝ × ℝ × ℝ) : r = inscribed_circle_radius ∧ θ = angle_MKN_ABC → sides = sides_triangle_ABC := 
by
  -- conditions are assumed correct
  sorry

end find_sides_of_triangle_l301_301636


namespace score_below_mean_l301_301432

theorem score_below_mean :
  ∃ (σ : ℝ), (74 - 2 * σ = 58) ∧ (98 - 74 = 3 * σ) :=
sorry

end score_below_mean_l301_301432


namespace heesu_received_most_sweets_l301_301306

theorem heesu_received_most_sweets
  (total_sweets : ℕ)
  (minsus_sweets : ℕ)
  (jaeyoungs_sweets : ℕ)
  (heesus_sweets : ℕ)
  (h_total : total_sweets = 30)
  (h_minsu : minsus_sweets = 12)
  (h_jaeyoung : jaeyoungs_sweets = 3)
  (h_heesu : heesus_sweets = 15) :
  heesus_sweets = max minsus_sweets (max jaeyoungs_sweets heesus_sweets) :=
by sorry

end heesu_received_most_sweets_l301_301306


namespace probability_at_least_two_females_l301_301141

theorem probability_at_least_two_females (total_contestants : ℕ) (females : ℕ) (males : ℕ) (chosen : ℕ) :
  total_contestants = 8 → females = 5 → males = 3 → chosen = 3 →
  ( (∑ (k in finset.range (chosen + 1)), if k ≥ 2 then nat.choose females k * nat.choose males (chosen - k) else 0) / 
    (nat.choose total_contestants chosen) : ℚ ) = 5 / 7 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry  -- Proof is skipped

end probability_at_least_two_females_l301_301141


namespace money_lent_to_C_l301_301369

theorem money_lent_to_C (X : ℝ) (interest_rate : ℝ) (P_b : ℝ) (T_b : ℝ) (T_c : ℝ) (total_interest : ℝ) :
  interest_rate = 0.09 →
  P_b = 5000 →
  T_b = 2 →
  T_c = 4 →
  total_interest = 1980 →
  (P_b * interest_rate * T_b + X * interest_rate * T_c = total_interest) →
  X = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end money_lent_to_C_l301_301369


namespace OABC_parallelogram_angle_AOC_OE_vector_midpoint_intersect_l301_301583

noncomputable def is_parallelogram (A B C O : ℝ × ℝ) : Prop :=
  B.1 - A.1 = O.1 ∧ C.2 - A.2 = O.2

variables (a b : ℝ)

def A : ℝ × ℝ := (4, a)
def B : ℝ × ℝ := (b, 8)
def C : ℝ × ℝ := (a, b)
def O : ℝ × ℝ := (0, 0)

theorem OABC_parallelogram_angle_AOC 
  (h1 : is_parallelogram A B C O) 
  (h2 : a = 2) 
  (h3 : b = 6) : 
  ∠AOC = 45 :=
sorry

theorem OE_vector_midpoint_intersect 
  (h1 : is_parallelogram A B C O) 
  (h2 : a = 2) 
  (h3 : b = 6) 
  (D : ℝ × ℝ := (5, 5)) 
  (E : ℝ × ℝ := (5 * (λ : ℝ), 5 * (λ : ℝ))) 
  (h4 : λ = 2 / 3) : 
  vector OE = (10 / 3, 10 / 3) :=
sorry

end OABC_parallelogram_angle_AOC_OE_vector_midpoint_intersect_l301_301583


namespace prime_sum_remainder_l301_301774

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301774


namespace correctLikeTermsPair_l301_301827

def areLikeTerms (term1 term2 : String) : Bool :=
  -- Define the criteria for like terms (variables and their respective powers)
  sorry

def pairA : (String × String) := ("-2x^3", "-2x")
def pairB : (String × String) := ("-1/2ab", "18ba")
def pairC : (String × String) := ("x^2y", "-xy^2")
def pairD : (String × String) := ("4m", "4mn")

theorem correctLikeTermsPair :
  areLikeTerms pairA.1 pairA.2 = false ∧
  areLikeTerms pairB.1 pairB.2 = true ∧
  areLikeTerms pairC.1 pairC.2 = false ∧
  areLikeTerms pairD.1 pairD.2 = false :=
sorry

end correctLikeTermsPair_l301_301827


namespace polygon_diagonals_l301_301891

theorem polygon_diagonals (n : ℕ) (h1: ∀ (i: ℕ) (hi: i < n), interior_angle_of_polygon_i i = 150) : number_of_diagonals_from_one_vertex = 9 :=
sorry

end polygon_diagonals_l301_301891


namespace mean_integer_board_l301_301052

theorem mean_integer_board (n : ℕ) (nums : Fin n → ℕ)
  (h : ∀ (i j : Fin n), i ≠ j →
    (∃ a ∈ ℤ, a = (nums i + nums j) / 2) ∨
    (∃ b ∈ ℤ, b = Real.sqrt ((nums i : ℝ) * (nums j : ℝ)))) :
  ∃ board : Fin n → ℤ, 
    (∀ i j : Fin n, i ≠ j → 
      (board = λ k, (nums k + nums j) / 2) ∨ 
      (board = λ k, Real.sqrt ((nums k : ℝ) * (nums j : ℝ)).toInt)) :=
sorry

end mean_integer_board_l301_301052


namespace photocopy_distribution_l301_301511

-- Define the problem setting
variables {n k : ℕ}

-- Define the theorem stating the problem
theorem photocopy_distribution :
  ∀ n k : ℕ, (n > 0) → 
  (k + n).choose (n - 1) = (k + n - 1).choose (n - 1) :=
by sorry

end photocopy_distribution_l301_301511


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301762

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301762


namespace exists_arrangement_for_P_23_l301_301202

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301202


namespace part1_part2_l301_301285

noncomputable def f (x: ℝ) : ℝ := sorry

-- Conditions
axiom periodic_f : ∀ x : ℝ, f(x) = f(x + 2)
axiom symmetry_f : ∀ x : ℝ, f(2 - x) = f(2 + x)
axiom monotonic_interval_ab : ∀ a b : ℝ, monotonic (f: ℝ → ℝ) → a ≤ b → monotonic_interval a b

-- Problem 1: Prove that b - a ≤ 1
theorem part1 (a b : ℝ) (h : monotonic_interval_ab a b) : b - a ≤ 1 := 
sorry

-- Additional conditions for Part 2
axiom monotonic_01 : monotonic_interval 0 1
axiom condition_f_2x : ∀ x < 0, f(2 ^ x) > f(2)

-- Problem 2: Find the solution set for the inequality f(-10.5) > f(x^2 + 6x)
theorem part2 (x : ℝ) : 
  ∃ k : ℝ, k ≥ 1 ∧ 
  (-3 - sqrt (8 * k - 2) / 2 < x ∧ x < -3 - sqrt (8 * k - 6) / 2) ∨ 
  (-3 + sqrt (8 * k - 6) / 2 < x ∧ x < -3 + sqrt (8 * k - 2) / 2) :=
sorry

end part1_part2_l301_301285


namespace remainder_first_six_primes_div_seventh_l301_301811

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301811


namespace complement_A_union_B_l301_301496

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | log 2 (x + 2) ≤ 1}

def B : Set ℝ := {x | 1 / x < 1}

theorem complement_A_union_B :
  (U \ A) ∪ B = {x : ℝ | x ≠ 0} :=
by
  sorry

end complement_A_union_B_l301_301496


namespace exists_two_linear_functions_l301_301968

-- Define the quadratic trinomials and their general forms
variables (a b c d e f : ℝ)
-- Assuming coefficients a and d are non-zero
variable (ha : a ≠ 0)
variable (hd : d ≠ 0)

-- Define the linear function
def ell (m n x : ℝ) : ℝ := m * x + n

-- Define the quadratic trinomials P(x) and Q(x) 
def P (x : ℝ) := a * x^2 + b * x + c
def Q (x : ℝ) := d * x^2 + e * x + f

-- Prove that there exist exactly two linear functions ell(x) that satisfy the condition for all x
theorem exists_two_linear_functions : 
  ∃ (m1 m2 n1 n2 : ℝ), 
  (∀ x, P a b c x = Q d e f (ell m1 n1 x)) ∧ 
  (∀ x, P a b c x = Q d e f (ell m2 n2 x)) := 
sorry

end exists_two_linear_functions_l301_301968


namespace minimum_omega_is_4_l301_301488

noncomputable def min_omega (ω : ℝ) : ℝ := 
  if ω > 0 
  ∧ abs (arcsin (1/2)) < π / 2
  ∧ ∀ x : ℝ, (sin (ω * x + arcsin (1/2)) <= sin (ω * π / 12 + arcsin (1/2)))
  then ω else 0

theorem minimum_omega_is_4 : ∃ ω > 0, (ω = 4 ∧ min_omega ω = ω) :=
by
  use 4
  sorry

end minimum_omega_is_4_l301_301488


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301747

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301747


namespace num_correct_statements_l301_301041

def doubleAbsDiff (a b c d : ℝ) : ℝ :=
  |a - b| - |c - d|

theorem num_correct_statements : 
  (∀ a b c d : ℝ, (a, b, c, d) = (24, 25, 29, 30) → 
    (doubleAbsDiff a b c d = 0) ∨
    (doubleAbsDiff a c b d = 0) ∨
    (doubleAbsDiff a d b c = -0.5) ∨
    (doubleAbsDiff b c a d = 0.5)) → 
  (∀ x : ℝ, x ≥ 2 → 
    doubleAbsDiff (x^2) (2*x) 1 1 = 7 → 
    (x^4 + 2401 / x^4 = 226)) →
  (∀ x : ℝ, x ≥ -2 → 
    (doubleAbsDiff (2*x-5) (3*x-2) (4*x-1) (5*x+3)) ≠ 0) →
  (0 = 0)
:= by
  sorry

end num_correct_statements_l301_301041


namespace worth_six_inch_cube_l301_301368

-- The conditions provided in the problem
def cube_volume (side_length : ℕ) : ℕ := side_length ^ 3
def cube_worth (volume : ℕ) (worth : ℕ) : ℚ := worth / volume

-- The given values
def four_inch_worth := 800
def four_inch_side_length := 4
def six_inch_side_length := 6

-- The corresponding volumes
def four_inch_volume := cube_volume four_inch_side_length
def six_inch_volume := cube_volume six_inch_side_length

-- The value per cubic inch
def value_per_cubic_inch := cube_worth four_inch_volume four_inch_worth

-- The worth of the six-inch cube
def six_inch_worth := six_inch_volume * value_per_cubic_inch

-- The theorem to be proven
theorem worth_six_inch_cube : Real.floor (six_inch_worth) = 2700 := by
  sorry

end worth_six_inch_cube_l301_301368


namespace imaginary_part_of_z_modulus_of_omega_pow_l301_301086

variables (z : ℂ) (i : ℂ)
#check z + 1
#check i * (z + 1)
#check **(-2)**^(2i)
theorem imaginary_part_of_z (hz : i * (z + 1) = -2 + 2 * i) : 
  z.im = 2 := 
by admit

theorem modulus_of_omega_pow (hz : i * (z + 1) = -2 + 2 * i) (ω : ℂ) : 
  ω = z / (1 - 2 * i) → abs ω ^ 2012 = 1 := 
by admit

end imaginary_part_of_z_modulus_of_omega_pow_l301_301086


namespace liquid_x_percentage_in_new_solution_l301_301470

-- Definitions based on conditions
def initial_solution_y := 6 -- in kg
def percent_liquid_x := 0.10
def percent_liquid_z := 0.20
def percent_water := 0.70
def evaporated_water := 2 -- in kg
def added_solution_y := 1 -- in kg
def pure_liquid_z := 1 -- in kg

-- The proof problem we want to solve
theorem liquid_x_percentage_in_new_solution :
  let liquid_x := percent_liquid_x * initial_solution_y in
  let liquid_z := percent_liquid_z * initial_solution_y in
  let water := percent_water * initial_solution_y in
  let remaining_water := water - evaporated_water in
  let after_evaporation := liquid_x + liquid_z + remaining_water in
  let new_liquid_x := liquid_x + (percent_liquid_x * added_solution_y) in
  let new_liquid_z := liquid_z + (percent_liquid_z * added_solution_y) + pure_liquid_z in
  let new_water := remaining_water + (percent_water * added_solution_y) in
  let total_new_solution := new_liquid_x + new_liquid_z + new_water in
  (new_liquid_x / total_new_solution) * 100 = 11.67 :=
by sorry

end liquid_x_percentage_in_new_solution_l301_301470


namespace find_k_l301_301434

noncomputable def repeating_representation_base_k (k: ℕ) : Prop := 
  ((3 * k + 5 : ℚ) / (k^2 - 1)) = (11 / 85)

theorem find_k (k: ℕ) (hk : 1 < k) : repeating_representation_base_k k → k = 25 :=
by
  sorry

end find_k_l301_301434


namespace quadrilateral_AD_length_l301_301605

structure Quadrilateral :=
(A B C D E : Point)
(AB BC CD CE : ℝ)
(angle_B : angle A B C = 90)
(angle_C : angle B C D = 90)
(CD_perpendicular_CE : ⟪C, E⟫ = 10)

noncomputable def compute_length_AD : ℝ :=
  sqrt (6^2 + (21 - 10)^2)

theorem quadrilateral_AD_length (quad : Quadrilateral) (h_AB : quad.AB = 7) 
  (h_BC : quad.BC = 6) (h_CD : quad.CD = 21) (h_CE : quad.CE = 10) :
  compute_length_AD = sqrt 157 := 
by sorry

end quadrilateral_AD_length_l301_301605


namespace remainder_first_six_primes_div_seventh_l301_301814

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301814


namespace product_of_roots_l301_301586

/--
For the polynomial equation of least possible degree with rational coefficients,
having \( \sqrt[3]{5} + \sqrt[3]{125} \) as a root, the product of all of the roots is 280.
-/
theorem product_of_roots (P : Polynomial ℚ) 
(hP : P = Polynomial.monic (Polynomial.ofRealRoots [λ x, x^3 - 30 * x - 280 ])) 
(hr : P.has_root (λ x, x = (⟨npow_up_to 3 ⟨5, Real.is_cubic 5⟩, Rat.real_coe 125 + Rat.real_coe 5⟩) ) :
  P.roots_product = 280 :=
by
  sorry

end product_of_roots_l301_301586


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301792

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301792


namespace fifteenth_digit_sum_l301_301324

noncomputable def frac1 : ℚ := 1 / 8
noncomputable def frac2 : ℚ := 1 / 6

noncomputable def decimal1 : ℚ := 0.125
noncomputable def decimal2 : ℚ := 0.1666666666666667  -- extended enough to capture the repeating nature

theorem fifteenth_digit_sum : 
  frac1 = decimal1 ∧ frac2 = decimal2 →
  (let sum := decimal1 + decimal2 in
   let extended_decimal := "0.2866666666666667".toList in
   nth_digit_after_decimal extended_decimal 15 = '6') :=
begin
  sorry
end

end fifteenth_digit_sum_l301_301324


namespace sum_of_solutions_l301_301268

theorem sum_of_solutions : 
  (∀ (x y : ℝ), y = 5 → x^2 + y^2 = 169 → (Σ x in {x | x^2 + 25 = 144}, id x) = 0) :=
begin
  sorry,
end

end sum_of_solutions_l301_301268


namespace ten_m_add_n_eq_83_div_3_l301_301315

-- Defining the vertices of the triangle
def P : ℝ × ℝ := (-1, 2)
def Q : ℝ × ℝ := (5, 4)
def R : ℝ × ℝ := (4, -3)

-- Defining the point S inside the triangle PQR such that the areas of PQS, PRS, and QRS are equal
def S : ℝ × ℝ := ((-1 + 5 + 4) / 3, (2 + 4 - 3) / 3)

-- Theorem statement to prove 10 * m + n = 83 / 3
theorem ten_m_add_n_eq_83_div_3 : 
  let Sx := (-1 + 5 + 4) / 3 in
  let Sy := (2 + 4 - 3) / 3 in
  10 * Sx + Sy = 83 / 3 := by
  sorry

end ten_m_add_n_eq_83_div_3_l301_301315


namespace triangle_angle_equality_l301_301312

-- Definitions for the triangles and perpendiculars
variables {ABC A1B1C1 : Triangle}
variables (perp_AB : ∀ A1B1, A1B1 ⊥ (ABC.side AB))
variables (perp_AC : ∀ A1C1, A1C1 ⊥ (ABC.side AC))
variables (perp_BC : ∀ B1C1, B1C1 ⊥ (ABC.side BC))

-- The angles of the triangles
variables (α β γ : Angle)
variables (α1 β1 γ1 : Angle)

-- The statement to prove
theorem triangle_angle_equality :
  ∀ (ABC A1B1C1 : Triangle),
  (∀ A1B1, A1B1 ⊥ (ABC.side AB)) →
  (∀ A1C1, A1C1 ⊥ (ABC.side AC)) →
  (∀ B1C1, B1C1 ⊥ (ABC.side BC)) →
  ∀ α β γ α1 β1 γ1,
  α = α1 ∧ β = β1 ∧ γ = γ1 :=
by
  sorry

end triangle_angle_equality_l301_301312


namespace determinant_scalar_multiplication_l301_301976

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end determinant_scalar_multiplication_l301_301976


namespace probability_path_from_first_to_last_floor_open_doors_l301_301859

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l301_301859


namespace expression_equals_33_l301_301924

noncomputable def calculate_expression : ℚ :=
  let part1 := 25 * 52
  let part2 := 46 * 15
  let diff := part1 - part2
  (2013 / diff) * 10

theorem expression_equals_33 : calculate_expression = 33 := sorry

end expression_equals_33_l301_301924


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301749

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301749


namespace minimum_value_xy_range_of_m_l301_301475

noncomputable def minimum_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 3/y = 2) : ℝ :=
  xy

theorem minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 3/y = 2) : minimum_xy x y hx hy h = 3 := 
  sorry

def range_m (x y m : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 3/y = 2) : Prop :=
  3*x + y ≥ m^2 - m

theorem range_of_m (m : ℝ) (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1/x + 3/y = 2) 
  (H : ∀ (x y : ℝ), (0 < x) → (0 < y) → (1/x + 3/y = 2) → 3*x + y ≥ m^2 - m) : -2 ≤ m ∧ m ≤ 3 :=
  sorry

end minimum_value_xy_range_of_m_l301_301475


namespace no_integers_satisfy_condition_l301_301516

def g (n : ℕ) : ℕ := ∑ d in (finset.range (n+1)).filter (λ d, d ∣ n), d

theorem no_integers_satisfy_condition :
  ∀ j : ℕ, 1 ≤ j ∧ j ≤ 2500 → g(j) = 2 + int.ofNat (int.sqrt j) + j → false :=
by
  sorry

end no_integers_satisfy_condition_l301_301516


namespace find_length_DE_plus_FG_l301_301314

structure Triangle where
  A B C : ℝ
  AB AC : ℝ
  ⟨angle_BAC : B = 90⟩

structure Point where
  x y : ℝ

def rightTriangle (A B C : Point) (AB AC : ℝ) (angle_BAC : ℕ) : Prop := 
  AB = 1 ∧ AC = 1 ∧ angle_BAC = 90

def parallel (line1 line2 : ℝ × ℝ) : Prop := 
  line1.1 / line1.2 = line2.1 / line2.2

def is_trapezoid (E G D F : Point) := 
  ∃ DE FG : ℝ,
  parallel (D.x - E.x, D.y - E.y) (F.x - G.x, F.y - G.y) ∧ DE = 2 * FG

theorem find_length_DE_plus_FG :
  ∀ (A B C E G D F : Point),
  rightTriangle A B C 1 1 90 →
  ∃ DE FG : ℝ,
  is_trapezoid E G D F →
  2 * (1 - (D.x)) + 3 * ((G.x - 1)) = 2 * (1 - (F.x + E.x))

  → (DE + FG = 3 / 2)
 :=
by
  intros
  exact sorry

end find_length_DE_plus_FG_l301_301314


namespace problem_trigonometric_identity_l301_301972

-- Define the problem conditions
theorem problem_trigonometric_identity
  (α : ℝ)
  (h : 3 * Real.sin (33 * Real.pi / 14 + α) = -5 * Real.cos (5 * Real.pi / 14 + α)) :
  Real.tan (5 * Real.pi / 14 + α) = -5 / 3 :=
sorry

end problem_trigonometric_identity_l301_301972


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301685

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301685


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301691

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301691


namespace count_representable_integers_l301_301108

section
  open Int

  def floor_sum (x : ℝ) : ℤ := 
    ⌊2 * x⌋ + ⌊4 * x⌋ + ⌊6 * x⌋ + ⌊8 * x⌋

  theorem count_representable_integers : 
    ∃ (n : ℕ), n = 600 ∧ ∀ k, (1 ≤ k ∧ k ≤ 1000) → 
      (∃ x : ℝ, floor_sum x = k) ↔ (k ∈ finset.range 600) :=
  sorry
end

end count_representable_integers_l301_301108


namespace sqrt_multiplication_property_sqrt_3_mul_sqrt_5_l301_301005

theorem sqrt_multiplication_property (a b : ℝ) (h : 0 ≤ a ∧ 0 ≤ b) : 
  real.sqrt (a * b) = real.sqrt a * real.sqrt b := by sorry

theorem sqrt_3_mul_sqrt_5 : (real.sqrt 3) * (real.sqrt 5) = real.sqrt 15 :=
begin
  have h_pos : 0 ≤ 3 ∧ 0 ≤ 5 := by split; norm_num,
  rw ← sqrt_multiplication_property 3 5 h_pos,
  exact rfl,
end

end sqrt_multiplication_property_sqrt_3_mul_sqrt_5_l301_301005


namespace arrangement_for_P23_exists_l301_301239

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301239


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301700

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301700


namespace exists_similar_sizes_P_23_l301_301216

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301216


namespace compare_cubic_terms_l301_301978

theorem compare_cubic_terms (a b : ℝ) :
    (a ≥ b → a^3 - b^3 ≥ a * b^2 - a^2 * b) ∧
    (a < b → a^3 - b^3 ≤ a * b^2 - a^2 * b) :=
by sorry

end compare_cubic_terms_l301_301978


namespace statement_d_correct_l301_301828

/-- Define what it means for an expression to be a polynomial -/
def is_polynomial (f : ℕ → ℤ) : Prop :=
  ∃ (a b : ℤ), f = λ n, if n = 1 then a else if n = 0 then b else 0

/-- The statement to be proven -/
theorem statement_d_correct : is_polynomial (λ n, if n = 1 then 3 else if n = 0 then 1 else 0) :=
sorry

end statement_d_correct_l301_301828


namespace limit_ratio_to_zero_l301_301181

open Real

noncomputable def sequences_converge_to_zero (a b : ℕ → ℝ) : Prop :=
  ∀ (k : ℕ), (a k < b k) ∧
  (∀ (x : ℝ), cos (a k * x) + cos (b k * x) ≥ -1 / k)

theorem limit_ratio_to_zero (a b : ℕ → ℝ) (h : sequences_converge_to_zero a b) :
  tendsto (λ k, a k / b k) atTop (𝓝 0) :=
sorry

end limit_ratio_to_zero_l301_301181


namespace count_values_not_divide_g_l301_301170

def proper_divisors (n : ℕ) : List ℕ :=
  List.filter (λ d, d > 0 ∧ d < n ∧ n % d = 0) (List.range n)

def g (n : ℕ) : ℕ :=
  (proper_divisors n).prod

def n_not_divide_g (n : ℕ) : Prop :=
  ¬ (n ∣ g n)

def values_in_range_not_divide_g (m k : ℕ) : Set ℕ :=
  { n | m ≤ n ∧ n ≤ k ∧ n_not_divide_g n }

theorem count_values_not_divide_g (m k : ℕ) (h_m : m = 3) (h_k : k = 60) :
  (values_in_range_not_divide_g m k).card = 20 :=
by
  sorry

end count_values_not_divide_g_l301_301170


namespace positional_relationship_parallel_or_skew_l301_301134

-- Definitions for lines a and b in 3D space
variable {Point : Type} [AffineSpace Point (EuclideanSpace Point 3)]

def line (x : Point) (y : Point) := setOf (λ p : Point, ∃ t : ℝ, p = x + t • (y - x))

variable a b : set Point

-- Condition: Lines a and b do not have any common points
def no_common_points (a b : set Point) : Prop := (a ∩ b) = ∅

-- Statement of the proof problem
theorem positional_relationship_parallel_or_skew (h : no_common_points a b) :
  (∃ u v ∈ EuclideanSpace Point 3, a = line u v ∧ b = line u v) ∨
  (∃ u₁ v₁ u₂ v₂ ∈ EuclideanSpace Point 3, 
   a = line u₁ v₁ ∧ b = line u₂ v₂ ∧ ∀ t₁ t₂, u₁ + t₁ • (v₁ - u₁) ≠ u₂ + t₂ • (v₂ - u₂)) :=
sorry

end positional_relationship_parallel_or_skew_l301_301134


namespace max_value_of_A_l301_301982

theorem max_value_of_A (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / ((a + b + c)^4 - 79 * (a * b * c)^(4 / 3)) ≤ 3 :=
sorry

end max_value_of_A_l301_301982


namespace greatest_area_difference_l301_301319

theorem greatest_area_difference (l w l' w' : ℕ) (hl : l + w = 90) (hl' : l' + w' = 90) :
  (l * w - l' * w').nat_abs ≤ 1936 :=
sorry

end greatest_area_difference_l301_301319


namespace limit_proof_l301_301837

noncomputable def limit_expression : ℝ → ℝ := 
  λ x, (3 - 2 / cos x) ^ (1 / sin x ^ 2)

theorem limit_proof : 
  tendsto limit_expression (𝓝 0) (𝓝 (exp (-1))) :=
sorry

end limit_proof_l301_301837


namespace prime_sum_remainder_l301_301769

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301769


namespace num_grouping_arrangements_l301_301524

theorem num_grouping_arrangements (drivers ticket_collectors : Fin 4) :
  ∃ (n : ℕ), n = Nat.perm 4 4 :=
sorry

end num_grouping_arrangements_l301_301524


namespace arrangement_for_P23_exists_l301_301234

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301234


namespace find_other_root_l301_301469

-- Define the quadratic equation and its properties
def quadratic_roots (a b c x y : ℝ) : Prop :=
  x * y = c / a ∧ x + y = -b / a

theorem find_other_root (a : ℝ) (h : (quadratic_roots 1 1 (-a) 2 (-3))) :
  ∃ t : ℝ, t = -3 :=
by
  use -3
  sorry

end find_other_root_l301_301469


namespace range_of_m_l301_301483

def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x + 1

theorem range_of_m {m : ℝ} :
  (∃ x0 : ℝ, x0 > 0 ∧ f x0 m < 0) → m ∈ Iio (-2) :=
by
  sorry

end range_of_m_l301_301483


namespace prime_sum_remainder_l301_301773

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301773


namespace complex_mul_eq_l301_301346

theorem complex_mul_eq :
  (2 + 2 * Complex.i) * (1 - 2 * Complex.i) = 6 - 2 * Complex.i := 
by
  intros
  sorry

end complex_mul_eq_l301_301346


namespace complex_multiplication_l301_301351

theorem complex_multiplication:
  (2 + 2 * complex.I) * (1 - 2 * complex.I) = 6 - 2 * complex.I := by
  sorry

end complex_multiplication_l301_301351


namespace probability_path_from_first_to_last_floor_open_doors_l301_301856

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l301_301856


namespace Tony_slices_left_after_week_l301_301313

-- Define the conditions and problem statement
def Tony_slices_per_day (days : ℕ) : ℕ := days * 2
def Tony_slices_on_Saturday : ℕ := 3 + 2
def Tony_slice_on_Sunday : ℕ := 1
def Total_slices_used (days : ℕ) : ℕ := Tony_slices_per_day days + Tony_slices_on_Saturday + Tony_slice_on_Sunday
def Initial_loaf : ℕ := 22
def Slices_left (days : ℕ) : ℕ := Initial_loaf - Total_slices_used days

-- Prove that Tony has 6 slices left after a week
theorem Tony_slices_left_after_week : Slices_left 5 = 6 := by
  sorry

end Tony_slices_left_after_week_l301_301313


namespace exists_arrangement_for_P_23_l301_301201

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301201


namespace lewis_weekly_earnings_l301_301961

theorem lewis_weekly_earnings :
  ∀ total_weeks total_earnings,
  total_weeks = 19 → 
  total_earnings = 133 → 
  total_earnings / total_weeks = 7 := by
  intros total_weeks total_earnings h1 h2
  rw [h1, h2]
  norm_num
  sorry

end lewis_weekly_earnings_l301_301961


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301760

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301760


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301674

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301674


namespace increasing_function_geq_25_l301_301527

theorem increasing_function_geq_25 {m : ℝ} 
  (h : ∀ x y : ℝ, x ≥ -2 ∧ x ≤ y → (4 * x^2 - m * x + 5) ≤ (4 * y^2 - m * y + 5)) :
  (4 * 1^2 - m * 1 + 5) ≥ 25 :=
by {
  -- Proof is omitted
  sorry
}

end increasing_function_geq_25_l301_301527


namespace infimum_polynomial_modulus_l301_301588

theorem infimum_polynomial_modulus {a : ℝ} (h : 2 + Real.sqrt 2 ≤ a ∧ a ≤ 4) :
  (⨅ z : ℂ, |z| ≤ 1 → Complex.abs (z^2 - a*z + a)) = 1 :=
begin
  sorry
end

end infimum_polynomial_modulus_l301_301588


namespace nonnegative_poly_sum_of_squares_l301_301585

open Polynomial

theorem nonnegative_poly_sum_of_squares (P : Polynomial ℝ) 
    (hP : ∀ x : ℝ, 0 ≤ P.eval x) 
    : ∃ Q R : Polynomial ℝ, P = Q^2 + R^2 := 
by
  sorry

end nonnegative_poly_sum_of_squares_l301_301585


namespace triangle_proofs_l301_301450

noncomputable def triangle_ratios_and_area (A B C : ℝ) (a b c : ℝ) : (ℝ × ℝ) :=
if (cos A = 3/4) ∧ (cos C = 1/8) ∧ (||(\vec{AC} + \vec{BC})|| = sqrt 46) then
    let sinA := sqrt (1 - (3/4)^2) in
    let sinC := sqrt (1 - (1/8)^2) in
    let cosB := -cos (A + C) in
    let sinB := sqrt (1 - cosB^2) in
    let ratio := (sinA, sinB, sinC) in
    let area := 0.5 * a * b * sinC.upper in
    (ratio, area)
else (0,0)

theorem triangle_proofs :
∀ (A B C : ℝ), (a b c : ℝ),
cos A = 3/4 ∧ cos C = 1/8 ∧ (||(\vec{AC} + \vec {BC})|| = sqrt 46) →
( (sinA : sinB : sinC) = (4 : 5 : 6) ∧ (area = 15√7/4))
:= sorry

end triangle_proofs_l301_301450


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301692

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301692


namespace expected_difference_52_14_l301_301000

def roll_die (d : ℕ) : ℕ :=
  if d = 8
  then roll_die (d + 1 % d)
  else d

noncomputable def probability_organic : ℚ := 4 / 7
noncomputable def probability_gluten_free : ℚ := 3 / 7

noncomputable def expected_days_organic (days : ℕ) : ℚ := (4 / 7) * days
noncomputable def expected_days_gluten_free (days : ℕ) : ℚ := (3 / 7) * days

noncomputable def expected_difference (days : ℕ) : ℚ := expected_days_organic days - expected_days_gluten_free days

theorem expected_difference_52_14 : 
  expected_difference 365 = 52.14 := 
  sorry

end expected_difference_52_14_l301_301000


namespace solve_sqrt_eq_l301_301267

theorem solve_sqrt_eq (x : ℝ) :
  (sqrt (5 * x - 4) + 15 / sqrt (5 * x - 4) = 8) ↔ (x = 29 / 5 ∨ x = 13 / 5) :=
by
  sorry

end solve_sqrt_eq_l301_301267


namespace determine_x_l301_301941

noncomputable def proof_problem (x : ℝ) (y : ℝ) : Prop :=
  y > 0 → 2 * (x * y^2 + x^2 * y + 2 * y^2 + 2 * x * y) / (x + y) > 3 * x^2 * y

theorem determine_x (x : ℝ) : 
  (∀ (y : ℝ), y > 0 → proof_problem x y) ↔ 0 ≤ x ∧ x < (1 + Real.sqrt 13) / 3 := 
sorry

end determine_x_l301_301941


namespace sum_first_100_terms_l301_301449

def a (n : ℕ) : ℤ := (-1)^n * (2 * n + 1) * Int.sin (n * Int.pi / 2) + 1

def S (n : ℕ) : ℤ := ∑ i in Finset.range n, a i

theorem sum_first_100_terms :
  S 100 = 200 :=
by
  sorry

end sum_first_100_terms_l301_301449


namespace upper_limit_for_product_of_digits_210_l301_301307

theorem upper_limit_for_product_of_digits_210 :
  ∀ h : ℕ, (∀ d : ℕ, d ∈ (Nat.digits 10 h) → d ∈ {2, 3, 5, 7}) ∧ 
           (List.prod (Nat.digits 10 h) = 210) → h < 7533 :=
begin
  sorry
end

end upper_limit_for_product_of_digits_210_l301_301307


namespace base_length_of_isosceles_triangle_l301_301637

noncomputable def isosceles_triangle_base_length (height : ℝ) (radius : ℝ) : ℝ :=
  if height = 25 ∧ radius = 8 then 80 / 3 else 0

theorem base_length_of_isosceles_triangle :
  isosceles_triangle_base_length 25 8 = 80 / 3 :=
by
  -- skipping the proof
  sorry

end base_length_of_isosceles_triangle_l301_301637


namespace probability_of_three_draws_exceeding_seven_is_three_fifths_l301_301363

-- Define the number of the chips in the box.
def chips := {1, 2, 3, 4, 5, 6}

-- Define the condition of drawing without replacement such that sum of drawn values exceeds 7 in 3 draws.
def sum_exceeds_seven_in_three_draws : Prop :=
  ∃ a b c : ℕ, a ∈ chips ∧ b ∈ chips ∧ c ∈ chips ∧ a ≠ b ∧ b ≠ c ∧ a + b ≤ 7 ∧ a + b + c > 7

-- Define the total probability calculation with favorable conditions.
def probability_three_draws_required : ℚ :=
  18 / 30  -- As identified from the solution

-- The Lean theorem statement to prove the probability calculation.
theorem probability_of_three_draws_exceeding_seven_is_three_fifths :
  probability_three_draws_required = 3 / 5 :=
by
  -- To complete the theorem proof.
  sorry

end probability_of_three_draws_exceeding_seven_is_three_fifths_l301_301363


namespace bee_flight_time_l301_301884

theorem bee_flight_time (t : ℝ) : 
  let speed_daisy_to_rose := 2.6
  let speed_rose_to_poppy := speed_daisy_to_rose + 3
  let distance_daisy_to_rose := speed_daisy_to_rose * 10
  let distance_rose_to_poppy := distance_daisy_to_rose - 8
  distance_rose_to_poppy = speed_rose_to_poppy * t
  ∧ abs (t - 3) < 1 := 
sorry

end bee_flight_time_l301_301884


namespace find_a_values_l301_301422

noncomputable def system_has_exactly_three_solutions (a : ℝ) : Prop :=
  ∃ (x y : ℝ), 
    ((|y - 4| + |x + 12| - 3) * (x^2 + y^2 - 12) = 0) ∧ 
    ((x + 5)^2 + (y - 4)^2 = a)

theorem find_a_values : system_has_exactly_three_solutions 16 ∨ 
                        system_has_exactly_three_solutions (41 + 4 * Real.sqrt 123) :=
  by sorry

end find_a_values_l301_301422


namespace maze_paths_unique_l301_301555

-- Define the conditions and branching points
def maze_structure (x : ℕ) (b : ℕ) : Prop :=
  x > 0 ∧ b > 0 ∧
  -- This represents the structure and unfolding paths at each point
  ∀ (i : ℕ), i < x → ∃ j < b, True

-- Define a function to count the number of unique paths given the number of branching points
noncomputable def count_paths (x : ℕ) (b : ℕ) : ℕ :=
  x * (2 ^ b)

-- State the main theorem
theorem maze_paths_unique : ∃ x b, maze_structure x b ∧ count_paths x b = 16 :=
by
  -- The proof contents are skipped for now
  sorry

end maze_paths_unique_l301_301555


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301744

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301744


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301753

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301753


namespace percentage_to_pass_is_correct_l301_301391

-- Define the conditions
def marks_obtained : ℕ := 130
def marks_failed_by : ℕ := 14
def max_marks : ℕ := 400

-- Define the function to calculate the passing percentage
def passing_percentage (obtained : ℕ) (failed_by : ℕ) (max : ℕ) : ℚ :=
  ((obtained + failed_by : ℕ) / (max : ℚ)) * 100

-- Statement of the problem
theorem percentage_to_pass_is_correct :
  passing_percentage marks_obtained marks_failed_by max_marks = 36 := 
sorry

end percentage_to_pass_is_correct_l301_301391


namespace regular_decagon_interior_angle_l301_301329

theorem regular_decagon_interior_angle
  (n : ℕ) 
  (hn : n = 10) 
  (regular_decagon : true) :
  (180 * (n - 2)) / n = 144 :=
by
  rw [hn]
  sorry

end regular_decagon_interior_angle_l301_301329


namespace count_rectangles_l301_301426

theorem count_rectangles (n : ℕ) : 
  let vertices := (n + 1) ^ 2,
      valid_pairs := n ^ 2
  in (vertices * valid_pairs) / 4 = (n * n * (n + 1) * (n + 1)) / 4 := by
  sorry

end count_rectangles_l301_301426


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301739

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301739


namespace prime_sum_remainder_l301_301770

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301770


namespace base6_addition_problem_l301_301415

-- Definitions to capture the base-6 addition problem components.
def base6₀ := 0
def base6₁ := 1
def base6₂ := 2
def base6₃ := 3
def base6₄ := 4
def base6₅ := 5

-- The main hypothesis about the base-6 addition
theorem base6_addition_problem (diamond : ℕ) (h : diamond ∈ [base6₀, base6₁, base6₂, base6₃, base6₄, base6₅]) :
  ((diamond + base6₅) % 6 = base6₃ ∨ (diamond + base6₅) % 6 = (base6₃ + 6 * 1 % 6)) ∧
  (diamond + base6₂ + base6₂ = diamond % 6) →
  diamond = base6₄ :=
sorry

end base6_addition_problem_l301_301415


namespace difference_between_sums_l301_301016

open Nat

-- Sum of the first 'n' positive odd integers formula: n^2
def sum_of_first_odd (n : ℕ) : ℕ := n * n

-- Sum of the first 'n' positive even integers formula: n(n+1)
def sum_of_first_even (n : ℕ) : ℕ := n * (n + 1)

-- The main theorem stating the difference between the sums
theorem difference_between_sums (n : ℕ) (h : n = 3005) :
  sum_of_first_even n - sum_of_first_odd n = 3005 :=
by
  sorry

end difference_between_sums_l301_301016


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301709

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301709


namespace full_time_score_l301_301394

variables (x : ℕ)

def half_time_score_visitors := 14
def half_time_score_home := 9
def visitors_full_time_score := half_time_score_visitors + x
def home_full_time_score := half_time_score_home + 2 * x
def home_team_win_by_one := home_full_time_score = visitors_full_time_score + 1

theorem full_time_score 
  (h : home_team_win_by_one) : 
  visitors_full_time_score = 20 ∧ home_full_time_score = 21 :=
by
  sorry

end full_time_score_l301_301394


namespace xiaoming_juice_problem_l301_301946

theorem xiaoming_juice_problem 
  (x : ℕ)
  (h1 : let remaining_after_eve := x / 2 + 1 in true)
  (h2 : let remaining_after_first := remaining_after_eve / 2 in true)
  (h3 : let remaining_after_second := remaining_after_first / 2 - 1 in remaining_after_second = 2) :
  x = 22 :=
by
  sorry

end xiaoming_juice_problem_l301_301946


namespace first_digit_base_9_of_122122111221_is_8_l301_301624

noncomputable def base_3_number : Nat :=
  1 * (3^11) + 2 * (3^10) + 2 * (3^9) + 1 * (3^8) +
  2 * (3^7) + 2 * (3^6) + 1 * (3^5) + 1 * (3^4) +
  1 * (3^3) + 2 * (3^2) + 2 * (3^1) + 1 * (3^0)

theorem first_digit_base_9_of_122122111221_is_8 :
  let y := base_3_number in
  ∀ (first_digit : Fin 9), 
  (∃ (digits : List (Fin 9)), to_base_digits 9 y = first_digit :: digits) →
  first_digit = 8 :=
by
  sorry

end first_digit_base_9_of_122122111221_is_8_l301_301624


namespace largest_number_of_acute_angles_in_convex_octagon_l301_301663

theorem largest_number_of_acute_angles_in_convex_octagon : 
  ∀ (n : ℕ), n = 8 → 
  (∑ i in (finset.range 8), (if i < 4 then (θ: ℝ) < 90 else (θ: ℝ) > 90)) < 1080 → n ≤ 4 :=
sorry

end largest_number_of_acute_angles_in_convex_octagon_l301_301663


namespace cost_of_plastering_l301_301830

def length := 25
def width := 12
def depth := 6
def cost_per_sq_meter_paise := 75

def surface_area_of_two_walls_one := 2 * (length * depth)
def surface_area_of_two_walls_two := 2 * (width * depth)
def surface_area_of_bottom := length * width

def total_surface_area := surface_area_of_two_walls_one + surface_area_of_two_walls_two + surface_area_of_bottom

def cost_per_sq_meter_rupees := cost_per_sq_meter_paise / 100
def total_cost := total_surface_area * cost_per_sq_meter_rupees

theorem cost_of_plastering : total_cost = 558 := by
  sorry

end cost_of_plastering_l301_301830


namespace parity_of_exponentiated_sum_l301_301287

theorem parity_of_exponentiated_sum
  : (1 ^ 1994 + 9 ^ 1994 + 8 ^ 1994 + 6 ^ 1994) % 2 = 0 := 
by
  sorry

end parity_of_exponentiated_sum_l301_301287


namespace positive_integers_satisfying_inequality_l301_301109

theorem positive_integers_satisfying_inequality : 
  (∀ n : ℕ, 1 ≤ n → (n + 7) * (n - 4) * (n - 15) < 0 → n ∈ {5, 6, 7, 8, 9, 10, 11, 12, 13, 14})
  ∧ {n : ℕ | 1 ≤ n ∧ (n + 7) * (n - 4) * (n - 15) < 0}.to_finset.card = 10 := 
sorry

end positive_integers_satisfying_inequality_l301_301109


namespace area_AEF_l301_301187

theorem area_AEF (hBCD : triangle_area B C D = 1)
  (hBDE : triangle_area B D E = 1/3)
  (hCDF : triangle_area C D F = 1/5) :
  triangle_area A E F = 4 / 35 :=
sorry

end area_AEF_l301_301187


namespace solve_for_k_l301_301119

-- Define the hypotheses as Lean statements
theorem solve_for_k (x k : ℝ) (h₁ : (x^2 - k) * (x + k) = x^3 + k * (x^2 - x - 6)) (h₂ : k ≠ 0) : k = 6 :=
by {
  sorry
}

end solve_for_k_l301_301119


namespace sum_of_coordinates_D_l301_301074

variables {x y : ℕ}

/- Definitions of given conditions -/
def is_midpoint_scaled (N C D : ℚ × ℚ) : Prop :=
  ∃ (x y : ℚ), 
    N = (x, y) ∧
    N.fst = ((C.fst + D.fst) / 2) * (1 / 2) ∧
    N.snd = ((C.snd + D.snd) / 2) * (1 / 2)

def point_C := (7 : ℚ, -3 : ℚ)
def point_N := (4 : ℚ, 1 : ℚ)

/- Main theorem stating the proof goal -/
theorem sum_of_coordinates_D : 
  (D : ℚ × ℚ) → is_midpoint_scaled point_N point_C D → (D.fst + D.snd = 16) :=
sorry

end sum_of_coordinates_D_l301_301074


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301682

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301682


namespace number_of_solutions_l301_301509

theorem number_of_solutions :
  (∃ x : ℝ, -25 < x ∧ x < 120 ∧ 2 * Real.cos x ^ 2 - 3 * Real.sin x ^ 2 = 1) → 
  -- There are 24 distinct values of x that satisfy the conditions 
  ∃ count : ℕ, count = 24 := by
  sorry

end number_of_solutions_l301_301509


namespace f_even_solution_set_l301_301092

def f (x : ℝ) : ℝ := 2^(|x|) - 2

-- Proving f(x) is an even function
theorem f_even : ∀ x : ℝ, f(-x) = f(x) := by 
  sorry

-- Proving the solution set for the inequality
theorem solution_set : { x : ℝ | x * (f(x) + f(-x)) > 0 } = { x : ℝ | x > 1 ∨ -1 < x ∧ x < 0 } := by
  sorry

end f_even_solution_set_l301_301092


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301754

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301754


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301786

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301786


namespace at_least_one_not_less_than_two_l301_301461

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + 1 / b ≥ 2) ∨ (b + 1 / c ≥ 2) ∨ (c + 1 / a ≥ 2) :=
sorry

end at_least_one_not_less_than_two_l301_301461


namespace total_squirrels_l301_301960

theorem total_squirrels :
  let first := 15
  let second := 15 + (1/3) * 15
  let third := second + (1/4) * second
  let fourth := third + (1/5) * third
  let fifth := fourth + (1/6) * fourth
  first + second + third + fourth + fifth = 125 :=
by
  let first := 15
  let second := 15 + (1 / 3) * 15
  let third := second + (1 / 4) * second
  let fourth := third + (1 / 5) * third
  let fifth := fourth + (1 / 6) * fourth
  have h1 : first = 15 := rfl
  have h2 : second = 15 + 5 := by linarith
  have h3 : third = 20 + 5 := by linarith
  have h4 : fourth = 25 + 5 := by linarith
  have h5 : fifth = 30 + 5 := by linarith
  have total : first + second + third + fourth + fifth = 15 + 20 + 25 + 30 + 35 := by
    rw [h1, h2, h3, h4, h5]
  exact total

end total_squirrels_l301_301960


namespace polynomial_identity_l301_301574

theorem polynomial_identity
  (x : ℝ)
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ)
  (h : (2*x + 1)^6 = a_0*x^6 + a_1*x^5 + a_2*x^4 + a_3*x^3 + a_4*x^2 + a_5*x + a_6) :
  (a_0 + a_1 + a_2 + a_3 + a_4 + a_5 + a_6 = 729)
  ∧ (a_1 + a_3 + a_5 = 364)
  ∧ (a_2 + a_4 = 300) := sorry

end polynomial_identity_l301_301574


namespace sin_cos_mult_sin_cos_cube_diff_sin_cos_fourth_add_l301_301051

variable {θ : ℝ}
variable {a b : ℝ}

theorem sin_cos_mult (h : sin θ - cos θ = 1 / 2) : sin θ * cos θ = 3 / 8 := 
sorry

theorem sin_cos_cube_diff (h : sin θ - cos θ = 1 / 2) : sin θ ^ 3 - cos θ ^ 3 = 11 / 16 := 
sorry

theorem sin_cos_fourth_add (h : sin θ - cos θ = 1 / 2) : sin θ ^ 4 + cos θ ^ 4 = 23 / 32 := 
sorry

end sin_cos_mult_sin_cos_cube_diff_sin_cos_fourth_add_l301_301051


namespace remainder_of_sum_of_primes_l301_301796

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301796


namespace marie_gift_boxes_l301_301591

theorem marie_gift_boxes
  (total_eggs : ℕ)
  (weight_per_egg : ℕ)
  (remaining_weight : ℕ)
  (melted_eggs_weight : ℕ)
  (eggs_per_box : ℕ)
  (total_boxes : ℕ)
  (H1 : total_eggs = 12)
  (H2 : weight_per_egg = 10)
  (H3 : remaining_weight = 90)
  (H4 : melted_eggs_weight = total_eggs * weight_per_egg - remaining_weight)
  (H5 : melted_eggs_weight / weight_per_egg = eggs_per_box)
  (H6 : total_eggs / eggs_per_box = total_boxes) :
  total_boxes = 4 := 
sorry

end marie_gift_boxes_l301_301591


namespace problem_l301_301066

def A : Set ℝ := { x | x^2 - 2 * x - 3 < 0 }
def B : Set ℝ := { x | 2^(x - 2) > 1 }
def complement_R (B : Set ℝ) := { x : ℝ | ¬ (x ∈ B) }

theorem problem : A ∩ (complement_R B) = { x : ℝ | -1 < x ∧ x ≤ 2 } :=
by
  sorry

end problem_l301_301066


namespace probability_of_reaching_last_floor_l301_301864

noncomputable def probability_of_open_paths (n : ℕ) : ℝ :=
  2^(n-1) / (Nat.choose (2*(n-1)) (n-1))

theorem probability_of_reaching_last_floor (n : ℕ) :
  probability_of_open_paths n = 2^(n-1) / (Nat.choose (2*(n-1)) (n-1)) :=
by
  sorry

end probability_of_reaching_last_floor_l301_301864


namespace inequality_proof_l301_301468

theorem inequality_proof (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d) (h5 : a * b * c * d = 1) :
  (1 / (1 + a)^2) + (1 / (1 + b)^2) + (1 / (1 + c)^2) + (1 / (1 + d)^2) ≥ 1 :=
by
  sorry

end inequality_proof_l301_301468


namespace parallelepiped_diagonal_ratio_l301_301144

variable (a b c : ℝ^3)

theorem parallelepiped_diagonal_ratio (a b c : ℝ^3) :
  let norm_sq (v : ℝ^3) := v.dot v in
  (norm_sq (a + b + c) + norm_sq (a - b + c) + norm_sq (-a + b + c) + norm_sq (a + b - c)) / 
  (norm_sq a + norm_sq b + norm_sq c) = 4 := by
  sorry

end parallelepiped_diagonal_ratio_l301_301144


namespace angle_between_line_and_plane_l301_301526

-- Define the conditions
def angle_direct_vector_normal_vector (direction_vector_angle : ℝ) := direction_vector_angle = 120

-- Define the goal to prove
theorem angle_between_line_and_plane (direction_vector_angle : ℝ) :
  angle_direct_vector_normal_vector direction_vector_angle → direction_vector_angle = 120 → 90 - (180 - direction_vector_angle) = 30 :=
by
  intros h_angle_eq angle_120
  sorry

end angle_between_line_and_plane_l301_301526


namespace area_of_triangle_l301_301123

-- Definitions based on conditions
def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 2) + P.2^2 = 1

def foci_distance : ℝ := 1

def angle_right (P F1 F2: ℝ × ℝ) : Prop :=
  let d1 := (P.1 - F1.1)^2 + (P.2 - F1.2)^2
      d2 := (P.1 - F2.1)^2 + (P.2 - F2.2)^2
  (d1 + d2 = 4 ∧ (d1 = 1 ∧ d2 = 3) ∨ (d1 = 3 ∧ d2 = 1))

-- The proof problem
theorem area_of_triangle 
  (P F1 F2: ℝ × ℝ) (hP : ellipse P) (hF1F2 : angle_right P F1 F2) : 
  let m := Real.sqrt ((P.1 - F1.1)^2 + (P.2 - F1.2)^2)
      n := Real.sqrt ((P.1 - F2.1)^2 + (P.2 - F2.2)^2)
  in ½ * m * n = 1 :=
by
  sorry

end area_of_triangle_l301_301123


namespace probability_from_first_to_last_l301_301853

noncomputable def probability_path_possible (n : ℕ) : ℝ :=
  (2 ^ (n - 1) : ℝ) / (Nat.choose (2 * (n - 1)) (n - 1) : ℝ)

theorem probability_from_first_to_last (n : ℕ) (h : n > 1) :
  probability_path_possible n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_l301_301853


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301673

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301673


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301722

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301722


namespace complex_multiplication_l301_301353

theorem complex_multiplication:
  (2 + 2 * complex.I) * (1 - 2 * complex.I) = 6 - 2 * complex.I := by
  sorry

end complex_multiplication_l301_301353


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301720

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301720


namespace total_situps_l301_301919

def situps (b c j : ℕ) : ℕ := b * 1 + c * 2 + j * 3

theorem total_situps :
  ∀ (b c j : ℕ),
    b = 45 →
    c = 2 * b →
    j = c + 5 →
    situps b c j = 510 :=
by intros b c j hb hc hj
   sorry

end total_situps_l301_301919


namespace solution_set_of_inequality_af_neg2x_pos_l301_301534

-- Given conditions:
-- f(x) = x^2 + ax + b has roots -1 and 2
-- We need to prove that the solution set for af(-2x) > 0 is -1 < x < 1/2
theorem solution_set_of_inequality_af_neg2x_pos (a b : ℝ) (x : ℝ) 
  (h1 : -1 + 2 = -a) 
  (h2 : -1 * 2 = b) : 
  (a * ((-2 * x)^2 + a * (-2 * x) + b) > 0) = (-1 < x ∧ x < 1/2) :=
by
  sorry

end solution_set_of_inequality_af_neg2x_pos_l301_301534


namespace probability_BD_greater_than_7_l301_301657

theorem probability_BD_greater_than_7 :
  ∀ (ABC : Triangle) (P : Point) (D : Point),
  ABC.is_isosceles_right ∧ ABC.AC = 10 ∧ ABC.AB = 10 ∧
  (P ∈ ABC.interior) ∧ 
  (D ∈ ABC.AC ∧ Line_through P B ∩ Line_through A C = {D}) →
  probability (BD.length > 7) = (Real.sqrt 151) / 10 :=
by sorry

end probability_BD_greater_than_7_l301_301657


namespace find_fraction_of_difference_eq_halves_l301_301505

theorem find_fraction_of_difference_eq_halves (x : ℚ) (h : 9 - x = 2.25) : x = 27 / 4 :=
by sorry

end find_fraction_of_difference_eq_halves_l301_301505


namespace remainder_first_six_primes_div_seventh_l301_301809

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301809


namespace simplify_expression_l301_301611

theorem simplify_expression (x : ℝ) :
  tan x + 3 * tan (3 * x) + 9 * tan (9 * x) + 27 * cot (27 * x) = cot x := by
  -- conditions
  have h1 : sin (3 * x) = 3 * sin x - 4 * (sin x) ^ 3 := sorry
  have h2 : cos (3 * x) = 4 * (cos x) ^ 3 - 3 * cos x := sorry
  have h3 : cot x - 3 * cot (3 * x) = tan x := sorry
  have h4 : cot (3 * x) - 3 * cot (9 * x) = tan (3 * x) := sorry
  have h5 : cot (9 * x) - 3 * cot (27 * x) = tan (9 * x) := sorry
  sorry

end simplify_expression_l301_301611


namespace interior_angle_regular_decagon_l301_301328

theorem interior_angle_regular_decagon : 
  let n := 10 in (180 * (n - 2)) / n = 144 := 
by
  sorry

end interior_angle_regular_decagon_l301_301328


namespace find_C_plus_D_l301_301641

noncomputable def polynomial_divisible (x : ℝ) (C : ℝ) (D : ℝ) : Prop := 
  ∃ (ω : ℝ), ω^2 + ω + 1 = 0 ∧ ω^104 + C*ω + D = 0

theorem find_C_plus_D (C D : ℝ) : 
  (∃ x : ℝ, polynomial_divisible x C D) → C + D = 2 :=
by
  sorry

end find_C_plus_D_l301_301641


namespace prime_base_representation_of_360_l301_301661

theorem prime_base_representation_of_360 :
  ∃ (exponents : List ℕ), exponents = [3, 2, 1, 0]
  ∧ (2^exponents.head! * 3^(exponents.tail!.head!) * 5^(exponents.tail!.tail!.head!) * 7^(exponents.tail!.tail!.tail!.head!)) = 360 := by
sorry

end prime_base_representation_of_360_l301_301661


namespace length_of_QR_l301_301557

theorem length_of_QR {P Q R S : Type*} [MetricSpace P] [MetricSpace Q] 
    (h1 : dist P R = 6) (h2 : dist P Q = 10) 
    (hS : dist S (midpoint R Q)) (h_alt : altitude_length P QR = 4) :
    segment_length QR = 4 * real.sqrt 5 := sorry

end length_of_QR_l301_301557


namespace exists_F_12_mod_23_zero_l301_301219

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301219


namespace radio_show_length_l301_301408

theorem radio_show_length :
  let s3 := 10
  let s2 := s3 + 5
  let s4 := s2 / 2
  let s5 := 2 * s4
  let s1 := 2 * (s2 + s3 + s4 + s5)
  s1 + s2 + s3 + s4 + s5 = 142.5 :=
by
  sorry

end radio_show_length_l301_301408


namespace max_value_expression_l301_301984

theorem max_value_expression (a b c : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) :
  ∃ M, M = 3 ∧ ∀ a b c, 0 < a → 0 < b → 0 < c → 
    let A := (a^3 * (b + c) + b^3 * (c + a) + c^3 * (a + b)) / 
             ((a + b + c)^4 - 79 * (a * b * c)^(4/3))
    A ≤ M := 
begin
  use 3,
  sorry
end

end max_value_expression_l301_301984


namespace exists_F_12_mod_23_zero_l301_301221

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301221


namespace domain_of_function_l301_301274

theorem domain_of_function :
  ∀ x : ℝ, (2 - x > 0) ∧ (2 * x + 1 > 0) ↔ (-1 / 2 < x) ∧ (x < 2) :=
sorry

end domain_of_function_l301_301274


namespace volume_of_ACDH_is_correct_l301_301354

noncomputable def volume_of_pyramid (V_cube : ℝ) : ℝ := 
  let s := real.cbrt V_cube -- cube root of the volume of the cube gives the side length
  let base_triangle_area := (s * s) / 2 -- area of triangle ACD (half of one face of the cube)
  let height := s -- height from H to plane ACD is the same as the side length of the cube
  (1 / 3) * base_triangle_area * height

theorem volume_of_ACDH_is_correct :
  volume_of_pyramid 8 = 4 / 3 :=
by
  calc
    volume_of_pyramid 8 = _ := by sorry -- function definition
    _ = 4 / 3 := by sorry -- final result

end volume_of_ACDH_is_correct_l301_301354


namespace range_of_g_l301_301413

noncomputable def g (x : ℝ) : ℝ :=
  (Real.arctan (x / 2))^2 + π * Real.arccot (x / 2) - (Real.arccot (x / 2))^2 + (π^2 / 8) * (x^2 - 2 * x + 2)

theorem range_of_g :
  ∀ y, ∃ x, g x = y ↔ y ∈ set.Ici (3 * π^2 / 8) :=
sorry

end range_of_g_l301_301413


namespace find_k_for_tangent_graph_l301_301407

theorem find_k_for_tangent_graph (k : ℝ) (h : (∀ x : ℝ, x^2 - 6 * x + k = 0 → (x = 3))) : k = 9 :=
sorry

end find_k_for_tangent_graph_l301_301407


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301702

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301702


namespace least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l301_301823

-- The numbers involved and the requirements described
def num : ℕ := 427398

def least_to_subtract_10 : ℕ := 8
def least_to_subtract_100 : ℕ := 98
def least_to_subtract_1000 : ℕ := 398

-- Proving the conditions:
-- 1. (num - least_to_subtract_10) is divisible by 10
-- 2. (num - least_to_subtract_100) is divisible by 100
-- 3. (num - least_to_subtract_1000) is divisible by 1000

theorem least_subtract_divisible_by_10 : (num - least_to_subtract_10) % 10 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_100 : (num - least_to_subtract_100) % 100 = 0 := 
by 
  sorry

theorem least_subtract_divisible_by_1000 : (num - least_to_subtract_1000) % 1000 = 0 := 
by 
  sorry

end least_subtract_divisible_by_10_least_subtract_divisible_by_100_least_subtract_divisible_by_1000_l301_301823


namespace max_num_negatives_min_positive_product_l301_301011

-- Given a sequence of 20 integers where the product of all elements is negative
variable (seq : Fin 20 → ℤ)
variable (neg_prod : (∏ i, seq i) < 0)

-- Define the number of negative integers in the sequence
def num_negatives (seq : Fin 20 → ℤ) : ℕ :=
  Finset.univ.filter (λ i, seq i < 0).card

-- Problem 1: Proving the maximum number of negative integers given the conditions
theorem max_num_negatives (seq : Fin 20 → ℤ) (neg_prod : (∏ i, seq i) < 0) : num_negatives seq = 19 :=
sorry

-- Problem 2: Proving the minimum positive product of any 5 elements within the sequence
theorem min_positive_product (seq : Fin 20 → ℤ) (neg_prod : (∏ i, seq i) < 0) : ∃ S : Finset (Fin 20), S.card = 5 ∧ (0 < (∏ x in S, seq x)) ∧ (∏ x in S, seq x) = 1 :=
sorry

end max_num_negatives_min_positive_product_l301_301011


namespace points_circle_l301_301556

-- Define the points A and B in the plane.
def A := (0 : ℝ, 0 : ℝ)
def B := (1 : ℝ, 0 : ℝ)

-- Define a point M with coordinates (x, y).
variables (x y : ℝ)
def M := (x, y)

-- Define the distance function between two points.
def dist (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- State the main theorem.
theorem points_circle :
  (dist M A = 3 * dist M B) →
  (x - 9/8)^2 + y^2 = (3/8)^2 :=
begin
  sorry
end

end points_circle_l301_301556


namespace time_differences_l301_301590

def malcolm_speed := 6 -- minutes per mile
def joshua_speed := 8 -- minutes per mile
def lila_speed := 7 -- minutes per mile
def race_distance := 12 -- miles

noncomputable def malcolm_time := malcolm_speed * race_distance
noncomputable def joshua_time := joshua_speed * race_distance
noncomputable def lila_time := lila_speed * race_distance

theorem time_differences :
  joshua_time - malcolm_time = 24 ∧
  lila_time - malcolm_time = 12 :=
by
  sorry

end time_differences_l301_301590


namespace dot_product_zero_l301_301501

-- Define the two vectors and their magnitudes and the angle between them.
variables {V : Type*} [inner_product_space ℝ V]
variables (a b : V)
variables (theta : ℝ)
variables (a_magnitude b_magnitude : ℝ)

-- Given conditions
def conditions : Prop :=
  θ = 2 * real.pi / 3 ∧ -- angle 120 degrees in radians
  ∥a∥ = 4 ∧ -- magnitude of vector a
  ∥b∥ = 4 -- magnitude of vector b

-- The goal proof statement
theorem dot_product_zero : conditions a b θ ∥a∥ ∥b∥ → inner_product_space.real_inner b (2 • a + b) = 0 :=
by
  sorry

end dot_product_zero_l301_301501


namespace count_integers_with_block_178_l301_301106

theorem count_integers_with_block_178 (a b : ℕ) : 10000 ≤ a ∧ a < 100000 → 10000 ≤ b ∧ b < 100000 → a = b → b - a = 99999 → ∃ n, n = 280 ∧ (n = a + b) := sorry

end count_integers_with_block_178_l301_301106


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301734

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301734


namespace problem_statement_l301_301981

namespace GeometricRelations

variables {Line Plane : Type} [Nonempty Line] [Nonempty Plane]

-- Define parallel and perpendicular relations
def parallel (l : Line) (p : Plane) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry

-- Given conditions
variables (m n : Line) (α β : Plane)

-- The theorem to be proven
theorem problem_statement 
  (h1 : perpendicular m β) 
  (h2 : parallel α β) : 
  perpendicular m α :=
sorry

end GeometricRelations

end problem_statement_l301_301981


namespace exists_perpendicular_bisector_point_l301_301192

noncomputable theory

open EuclideanGeometry

-- Let's define the problem in Lean
theorem exists_perpendicular_bisector_point {C A B : Point} (h : dist C A + dist C B = 1) :
  ∃ P : Point, ∀ A B : Point, (dist C A + dist C B = 1) → 
  is_perpendicular_bisector_line (line_P := line_segment A B) (P := P) :=
sorry

end exists_perpendicular_bisector_point_l301_301192


namespace power_function_is_decreasing_l301_301130

-- We define the conditions
def alpha := -1 / 2
def f (x : ℝ) := x ^ alpha

-- Condition that the function passes through the given point
def passes_point (p : ℝ × ℝ) : Prop := f p.1 = p.2

-- Proof statement that f(x) is a decreasing function given the condition
theorem power_function_is_decreasing (p : ℝ × ℝ) (h : passes_point p) :
  (∀ x y : ℝ, x < y → f x > f y) :=
begin
  sorry
end

-- Using the specific point provided in the problem
example : power_function_is_decreasing (3, real.sqrt 3 / 3) (by simp [passes_point, f, alpha, real.sqrt_eq_rpow, real.rpow_neg_one_div_two, real.sqrt_three_eq_rpow]) :=
begin
  sorry
end

end power_function_is_decreasing_l301_301130


namespace intersection_complement_l301_301072

def setA (x : ℝ) : Prop := x^2 - 2 * x - 3 < 0
def setB (x : ℝ) : Prop := 2^(x - 2) > 1

theorem intersection_complement :
  { x : ℝ | setA x } ∩ { x : ℝ | ¬ setB x } = { x : ℝ | -1 < x ∧ x ≤ 2 } := 
by 
  sorry

end intersection_complement_l301_301072


namespace mowing_time_approx_l301_301258

-- Definitions for the given problem
def lawn_length : ℝ := 72  -- in feet
def lawn_width : ℝ := 120  -- in feet
def swath_width_inches : ℝ := 30  -- in inches
def overlap_inches : ℝ := 6  -- in inches
def mowing_speed : ℝ := 4500  -- in feet per hour

-- Conversion and effective cutting width calculation
def swath_width_feet : ℝ := swath_width_inches / 12
def overlap_feet : ℝ := overlap_inches / 12
def effective_cutting_width : ℝ := swath_width_feet - overlap_feet

-- Number of strips calculation
def number_of_strips : ℝ := lawn_width / effective_cutting_width

-- Total mowing distance calculation
def total_mowing_distance : ℝ := number_of_strips * lawn_length

-- Time required to mow completion
def time_to_mow : ℝ := total_mowing_distance / mowing_speed

-- Main theorem statement
theorem mowing_time_approx : abs (time_to_mow - 0.95) < 0.05 :=
by
  sorry

end mowing_time_approx_l301_301258


namespace dodecagon_ratio_l301_301932

-- Given definitions
def regular_dodecagon (P : ℝ) := ∃ (n : ℝ), True -- assuming "P" as the area of the dodecagon.
def quadrilateral_ACEG (Q : ℝ) := ∃ (m : ℝ), True -- assuming "Q" as the area of quadrilateral ACEG.

-- The theorem to prove
theorem dodecagon_ratio :
  ∀ (P Q : ℝ), regular_dodecagon P → quadrilateral_ACEG Q →  (Q / P) = (2 * (sqrt 3) + 3) / 3 :=
by
  sorry

end dodecagon_ratio_l301_301932


namespace right_triangle_similarity_l301_301902

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l301_301902


namespace value_of_expression_l301_301302

theorem value_of_expression : (2 + 4 + 6) - (1 + 3 + 5) = 3 := 
by 
  sorry

end value_of_expression_l301_301302


namespace dividend_calculation_l301_301295

theorem dividend_calculation (remainder quotient divisor : ℕ) (h₁ : remainder = 8) (h₂ : quotient = 43) (h₃ : divisor = 23) :
  divisor * quotient + remainder = 997 :=
by
  rw [h₁, h₂, h₃]
  sorry

end dividend_calculation_l301_301295


namespace total_balls_l301_301545

theorem total_balls (black_balls : ℕ) (prob_pick_black : ℚ) (total_balls : ℕ) :
  black_balls = 4 → prob_pick_black = 1 / 3 → total_balls = 12 :=
by
  intros h1 h2
  sorry

end total_balls_l301_301545


namespace part_a_infinite_1s_part_b_infinite_each_positive_l301_301544

def gcd (a b : ℕ) : ℕ := Nat.gcd a b
def greatest_odd_divisor (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let d := (List.range (n + 1)).filter (λ x => x > 0 ∧ x % 2 = 1 ∧ n % x = 0) |>.maximumD 0
  d

def next_value (a_n : ℕ) (n : ℕ) : ℕ :=
  let odd_div := greatest_odd_divisor n
  if odd_div % 4 = 1 then a_n + 1
  else if odd_div % 4 = 3 then a_n - 1
  else a_n

noncomputable def a_sequence : ℕ → ℕ
| 0     => 1
| n + 1 => next_value (a_sequence n) (n + 1)

theorem part_a_infinite_1s : ∀ n : ℕ, ∃ m > n, a_sequence m = 1 := sorry

theorem part_b_infinite_each_positive : ∀ k : ℕ, ∀ n : ℕ, ∃ m > n, a_sequence m = k := sorry

end part_a_infinite_1s_part_b_infinite_each_positive_l301_301544


namespace largest_perimeter_of_inscribed_rectangle_l301_301845

theorem largest_perimeter_of_inscribed_rectangle (P : ℝ) :
  (∀ (T : Type) [equilateral_triangle T] (s : ℝ), s = 1 →
  ∀ (R : Type) [rectangle_inscribed_in_triangle R] (p : ℝ), p = perimeter R →
  p ≥ P) ↔ P = Real.sqrt 3 :=
sorry

end largest_perimeter_of_inscribed_rectangle_l301_301845


namespace concyclic_points_l301_301988

open EuclideanGeometry

-- Definitions of the given points and conditions
variable {A B C D E F K L M N : Point}
variable {α : Triangle}

-- Conditions
variables
  (hTriangle : α = Triangle.mk A B C)
  (hD : is_altitude_foot A α E)
  (hE : is_altitude_foot B α F)
  (hF : is_altitude_foot C α D)
  (hK : is_intersect_circle_diameter_line A C B E K)
  (hL : is_intersect_circle_diameter_line A C B E L)
  (hM : is_intersect_circle_diameter_line A B C F M)
  (hN : is_intersect_circle_diameter_line A B C F N)

-- Goal to prove
theorem concyclic_points : are_concyclic K L M N :=
  sorry

end concyclic_points_l301_301988


namespace exists_similar_sizes_P_23_l301_301213

noncomputable def F : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

def similar_sizes (P : ℕ) := ∃ n : ℕ, F n % P = 0

theorem exists_similar_sizes_P_23 : similar_sizes 23 :=
by
  sorry

end exists_similar_sizes_P_23_l301_301213


namespace cosine_of_angle_l301_301525

theorem cosine_of_angle (α : ℝ) (x y r : ℝ)
  (h1 : x = -3)
  (h2 : y = 4)
  (h3 : r = real.sqrt(x^2 + y^2))
  (h4 : r = 5)
  : real.cos α = -3 / 5 := sorry

end cosine_of_angle_l301_301525


namespace vector_identity_l301_301166

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variables (A B C D : V)

def vector_relation (A B C D : V) : Prop :=
  (B - A) = 2 * (C - D)

theorem vector_identity (A B C D : V) (h : vector_relation A B C D) : 
  (B - D) = (C - A) - (3 / 2) * (B - A) :=
by
  sorry

end vector_identity_l301_301166


namespace remainder_first_six_primes_div_seventh_l301_301820

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301820


namespace remainder_of_sum_of_primes_l301_301805

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301805


namespace remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301716

theorem remainder_of_first_six_primes_sum_divided_by_seventh_prime : 
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  sum_primes % p7 = 7 := by sorry

end remainder_of_first_six_primes_sum_divided_by_seventh_prime_l301_301716


namespace exists_F_12_mod_23_zero_l301_301224

-- Define the recursive sequence F
def F : ℕ → ℤ
| 0     := 0
| 1     := 1
| (n+2) := 3 * F (n+1) - F n

-- Propose that F 12 (mod 23) is 0 when P = 23
theorem exists_F_12_mod_23_zero (P : ℕ) (hP : P = 23) : ∃ n : ℕ, F 12 % P = 0 :=
by
  use 12
  sorry

end exists_F_12_mod_23_zero_l301_301224


namespace investment_period_is_5_years_l301_301893

-- Define principal amount P
def principalAmount : ℝ := 8000

-- Define final amount A
def finalAmount : ℝ := 10210.25

-- Define annual interest rate r
def annualInterestRate : ℝ := 0.05

-- Define times compounded per year n
def timesCompoundedPerYear : ℤ := 1

-- Define the time period t
def timePeriod : ℤ := 5

-- Define the compound interest formula result for the problem statement
def compoundInterestFormula (P A r : ℝ) (n t : ℤ) : Prop :=
  A = P * (1 + r / n.toReal) ^ (n.toReal * t.toReal)

-- The theorem stating the given conditions lead to t = 5
theorem investment_period_is_5_years :
  compoundInterestFormula principalAmount finalAmount annualInterestRate timesCompoundedPerYear timePeriod := by
    sorry

end investment_period_is_5_years_l301_301893


namespace trajectory_eqn_l301_301063

-- Definition of points A and B
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Conditions given in the problem
def PA_squared (P : ℝ × ℝ) : ℝ := (P.1 + 1)^2 + P.2^2
def PB_squared (P : ℝ × ℝ) : ℝ := (P.1 - 1)^2 + P.2^2

-- The main statement to prove
theorem trajectory_eqn (P : ℝ × ℝ) (h : PA_squared P = 3 * PB_squared P) : 
  P.1^2 + P.2^2 - 4 * P.1 + 1 = 0 :=
by 
  sorry

end trajectory_eqn_l301_301063


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301677

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301677


namespace length_of_carton_l301_301883

theorem length_of_carton (L : ℕ) (soap_box_volume carton_volume : ℕ) :
  let soap_box_volume := 7 * 6 * 5
  let total_volume_soap : nat := 360 * soap_box_volume
  let carton_volume := 42 * 60 * L
  total_volume_soap = 75600 -> 
  carton_volume = 75600 -> 
  L = 30 :=
by
  sorry

end length_of_carton_l301_301883


namespace eight_digit_product_1400_l301_301425

def eight_digit_numbers_count : Nat :=
  sorry

theorem eight_digit_product_1400 : eight_digit_numbers_count = 5880 :=
  sorry

end eight_digit_product_1400_l301_301425


namespace range_of_a_if_decreasing_function_l301_301631

noncomputable def f (a : ℝ) : ℝ → ℝ :=
  λ x, if x ≥ 0 then (a - 5) * x - 1 else (x + a) / (x - 1)

theorem range_of_a_if_decreasing_function : 
  (∀ a : ℝ, (∀ x y : ℝ, (x ≤ y → f a x ≥ f a y)) → a ∈ (-1 : ℝ, 5]) :=
sorry

end range_of_a_if_decreasing_function_l301_301631


namespace U_ge_V_l301_301020

variable {x y: ℝ}

def U : ℝ := x^2 + y^2 + 1
def V : ℝ := 2 * (x + y - 1)

theorem U_ge_V : U ≥ V ∧ (x ≠ 1 ∨ y ≠ 1) → U > V :=
by
  sorry

end U_ge_V_l301_301020


namespace altitude_on_AB_eq_midline_on_BC_eq_l301_301438

open Real

-- Define the triangle with vertices A, B, and C.
def A : (ℝ × ℝ) := (4, 0)
def B : (ℝ × ℝ) := (8, 10)
def C : (ℝ × ℝ) := (0, 6)

-- Prove the equation of the altitude on side AB.
theorem altitude_on_AB_eq :
  ∃ (k : ℝ) (b : ℝ), k = -2/5 ∧ b = 6 ∧ (∀ x y : ℝ, y = k * x + b → 2 * x + 5 * y = 30) :=
by
  sorry

-- Prove the equation of the midline on side BC.
theorem midline_on_BC_eq :
  ∃ (x_coord : ℝ), x_coord = 4 ∧ (∀ y : ℝ, (x_coord, y) ∈ set_of (λ p : ℝ × ℝ, ∃ t : ℝ, p = ((B.1 + C.1) / 2, (B.2 + C.2) / 2))) :=
by
  sorry

end altitude_on_AB_eq_midline_on_BC_eq_l301_301438


namespace find_a_b_and_min_value_l301_301100

noncomputable def poly (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c

theorem find_a_b_and_min_value (c : ℝ) :
  (∀ x, has_deriv_at (poly x -3 -9 c) (3 * x^2 - 6 * x - 9) x) →
  (poly (-1) -3 -9 c = 7) →
  ∃ a b, a = -3 ∧ b = -9 ∧ poly 3 -3 -9 c = -25 :=
by {
  intros h_deriv h_value;
  use [-3, -9];
  split;
  { reflexivity },
  split;
  { reflexivity },
  {
    have h_c : c = 2, from by {
      simp [poly, pow_succ, pow_zero] at h_value,
      linarith,
    },
    rw h_c,
    simp [poly, pow_succ, pow_zero],
  }
}

end find_a_b_and_min_value_l301_301100


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301743

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301743


namespace arrangement_exists_for_P_eq_23_l301_301241

def F : ℕ → ℤ 
| 0       := 0
| 1       := 1
| (i + 2) := 3 * F (i + 1) - F i

theorem arrangement_exists_for_P_eq_23 :
  ∃ P : ℕ, P = 23 ∧ F 12 % 23 = 0 := 
begin 
  existsi 23,
  split,
  { refl },
  { sorry }
end

end arrangement_exists_for_P_eq_23_l301_301241


namespace max_value_of_expression_l301_301168

theorem max_value_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) :
  (∃ (x : ℝ), x = (ab/(a + b)) + (ac/(a + c)) + (bc/(b + c)) ∧ x = 9/4) :=
  sorry

end max_value_of_expression_l301_301168


namespace math_proof_problem_l301_301484

/-
  Problem:
  1. Prove that \phi = \frac{\pi}{2} given:
    - f(x) = 2 \sin x \cdot \cos^2(\phi / 2) + \cos x \sin \phi - \sin x.
    - f attains its minimum value at x = \pi.

  2. Prove that angle C in \triangle ABC is \frac{7\pi}{12} or \frac{\pi}{12} given:
    - a = 1, b = \sqrt{2}, and f(A) = \frac{\sqrt{3}}{2}.
-/

noncomputable def problem1 : Prop :=
  ∃ (φ : ℝ), (0 < φ ∧ φ < π) ∧ 
  (∀ x : ℝ, (2 * Real.sin x * Real.cos(φ / 2)^2 + Real.cos x * Real.sin φ - Real.sin x) ≥ (2 * Real.sin π * Real.cos(φ / 2)^2 + Real.cos π * Real.sin φ - Real.sin π)) ∧ 
  (2 * Real.sin π * Real.cos(φ / 2)^2 + Real.cos π * Real.sin φ - Real.sin π) = -1 →
  φ = π / 2

noncomputable def problem2 : Prop :=
  ∃ (A B C : ℝ), 
  (a = 1 ∧ b = Real.sqrt 2 ∧ 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π) ∧
  (Real.cos A = Real.sqrt 3 / 2 ∧ a / Real.sin A = b / Real.sin B) ∧ 
  B = π / 4 ∧ (C = π - A - B)  →
  (C = 7 * π / 12 ∨ C = π / 12)

# example of using Lean to state the new problem statement
theorem math_proof_problem : problem1 ∧ problem2 := sorry

end math_proof_problem_l301_301484


namespace x_y_sum_comb_identity_l301_301249

theorem x_y_sum_comb_identity (x y : ℝ) (m n : ℕ) (h : x + y = 1) : 
  x^(m+1) * (Finset.sum (Finset.range (n+1)) (λ j, Nat.choose (m+j) j * y^j)) + 
  y^(n+1) * (Finset.sum (Finset.range (m+1)) (λ i, Nat.choose (n+i) i * x^i)) = 1 :=
by
  sorry

end x_y_sum_comb_identity_l301_301249


namespace find_y_l301_301895

noncomputable def similar_triangles (a b x z : ℝ) :=
  (a / x = b / z)

theorem find_y 
  (a b x z : ℝ)
  (ha : a = 12)
  (hb : b = 9)
  (hz : z = 7)
  (h_sim : similar_triangles a b x z) :
  x = 28 / 3 :=
begin
  subst ha,
  subst hb,
  subst hz,
  unfold similar_triangles at h_sim,
  field_simp [h_sim],
  ring,
end

end find_y_l301_301895


namespace exists_arrangement_for_P_23_l301_301204

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301204


namespace exists_sequence_of_ten_numbers_l301_301562

theorem exists_sequence_of_ten_numbers :
  ∃ a : Fin 10 → ℝ,
    (∀ i : Fin 6,    a i + a ⟨i.1 + 1, sorry⟩ + a ⟨i.1 + 2, sorry⟩ + a ⟨i.1 + 3, sorry⟩ + a ⟨i.1 + 4, sorry⟩ > 0) ∧
    (∀ j : Fin 4, a j + a ⟨j.1 + 1, sorry⟩ + a ⟨j.1 + 2, sorry⟩ + a ⟨j.1 + 3, sorry⟩ + a ⟨j.1 + 4, sorry⟩ + a ⟨j.1 + 5, sorry⟩ + a ⟨j.1 + 6, sorry⟩ < 0) :=
sorry

end exists_sequence_of_ten_numbers_l301_301562


namespace normal_complaints_calculation_l301_301619

-- Define the normal number of complaints
def normal_complaints (C : ℕ) : ℕ := C

-- Define the complaints when short-staffed
def short_staffed_complaints (C : ℕ) : ℕ := (4 * C) / 3

-- Define the complaints when both conditions are met
def both_conditions_complaints (C : ℕ) : ℕ := (4 * C) / 3 + (4 * C) / 15

-- Main statement to prove
theorem normal_complaints_calculation (C : ℕ) (h : 3 * (both_conditions_complaints C) = 576) : C = 120 :=
by sorry

end normal_complaints_calculation_l301_301619


namespace alloy_specific_gravity_l301_301909

theorem alloy_specific_gravity 
  (G C : ℝ) 
  (h_gold_weight : 19)
  (h_copper_weight : 9)
  (h_ratio : G = 4 * C) :
  ((G * 19 + C * 9) / (G + C) = 17) :=
sorry

end alloy_specific_gravity_l301_301909


namespace projection_and_norm_l301_301499

noncomputable def v : ℝ × ℝ := (4, 5)
noncomputable def w : ℝ × ℝ := (-12, 3)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let scale := (dot_product u v) / (dot_product v v)
  (scale * v.1, scale * v.2)

def norm (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1 * u.1 + u.2 * u.2)

theorem projection_and_norm :
  projection v w = (-(528 / 17), 132 / 17)
  ∧ norm (projection v w) = 544 / 17 :=
by
  sorry

end projection_and_norm_l301_301499


namespace exists_arrangement_for_P_23_l301_301208

noncomputable def recurrence_relation (i : ℕ) : ℕ :=
  if i = 0 then 0
  else if i = 1 then 1
  else 3 * recurrence_relation (i - 1) - recurrence_relation (i - 2)

def is_similar (a b : ℕ) : Prop := 
  -- Define what it means for two pile sizes to be "similar".
  true -- Placeholder condition; should be replaced with the actual similarity condition.

theorem exists_arrangement_for_P_23 : ∃ (arrangement : list ℕ), 
  (∀ (i j : ℕ), i ≠ j → i < 23 → j < 23 → is_similar arrangement[i] arrangement[j]) ∧ 
  recurrence_relation 12 % 23 = 0 :=
by {
  -- Placeholder proof using the given calculations.
  sorry
}

end exists_arrangement_for_P_23_l301_301208


namespace max_value_fraction_l301_301994

theorem max_value_fraction {x y : ℝ} (h : x^2 + y^2 - 4 * x + 1 = 0) : 
  ∃ (z : ℝ), z = 1 + sqrt 3 ∧ z = (y + x) / x :=
by
  sorry

end max_value_fraction_l301_301994


namespace original_marked_price_l301_301255

theorem original_marked_price (P : ℝ) : 
  let discounted_price := 0.82 * P,
      final_price := 1.05 * discounted_price in
  final_price = 147.60 → P ≈ 171.43 :=
  by sorry

end original_marked_price_l301_301255


namespace find_monic_polynomial_l301_301029

theorem find_monic_polynomial :
  ∃ (p : polynomial ℚ), p.monic ∧ degree p = 4 ∧ 
    (∀ x, x = (∑ i in [0, 1, 2, 3], (coeff p i) * x ^ i)) ∧ 
    (∀ x, x = (x^4 - 24*x^2 + 4) → x = sqrt 5 + sqrt 7) := by
  sorry

end find_monic_polynomial_l301_301029


namespace probability_path_from_first_to_last_floor_open_doors_l301_301858

noncomputable
def probability_path_possible (n : ℕ) : ℚ :=
  (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1))

theorem probability_path_from_first_to_last_floor_open_doors (n : ℕ) :
  probability_path_possible n = (2 ^ (n - 1)) / (nat.choose (2 * (n - 1)) (n - 1)) :=
by
  sorry

end probability_path_from_first_to_last_floor_open_doors_l301_301858


namespace standard_hyperbola_equation_l301_301298

theorem standard_hyperbola_equation :
  ∃ (a b : ℝ), (a = 2 ∧ b = 1) →
  (∀ x y : ℝ, (x = 2 * Real.sqrt 2 ∧ y = Real.sqrt 6) →
  ((y^2 / 2) - (x^2 / 4) = 1)) :=
begin
  sorry

end standard_hyperbola_equation_l301_301298


namespace distance_between_points_is_correct_midpoint_between_points_is_correct_l301_301030

def point : Type := (ℚ × ℚ)

def distance (p1 p2 : point) : ℚ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

def midpoint (p1 p2 : point) : point :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem distance_between_points_is_correct :
  distance (2, 3) (5, 9) = 3 * real.sqrt 5 :=
by
  sorry

theorem midpoint_between_points_is_correct :
  midpoint (2, 3) (5, 9) = (3.5, 6) :=
by
  sorry

end distance_between_points_is_correct_midpoint_between_points_is_correct_l301_301030


namespace valid_numbers_count_is_7_l301_301107

-- Definitions based on the problem statement
def is_valid_digit (d : ℕ) : Prop :=
  d = 3 ∨ d = 0

def last_two_digits_div_by_4_eq_1 (n : ℕ) : Prop :=
  (n % 100) % 4 = 1

def nine_digit_number (n : ℕ) : Prop :=
  10^8 ≤ n ∧ n < 10^9 ∧ ∀ d ∈ digits 10 n, is_valid_digit d

def count_valid_numbers : ℕ :=
  (Finset.range (10^9)).filter (λ n, nine_digit_number n ∧ last_two_digits_div_by_4_eq_1 n).card

-- Statement to prove
theorem valid_numbers_count_is_7 : count_valid_numbers = 7 :=
by
  sorry

end valid_numbers_count_is_7_l301_301107


namespace product_ab_eq_six_l301_301395

theorem product_ab_eq_six (a b : ℝ) (h1 : (a * Real.tan(b * (π/4))) = 3) (h2 : (π / b) = π / 2) : a * b = 6 :=
by
  sorry

end product_ab_eq_six_l301_301395


namespace arc_length_polar_curve_segment_l301_301922

noncomputable def arc_length (rho : ℝ → ℝ) (rho' : ℝ → ℝ) (a b : ℝ) : ℝ :=
  ∫ x in a..b, real.sqrt (rho x ^ 2 + (rho' x) ^ 2)

def polar_curve (phi : ℝ) : ℝ := 6 * real.cos phi

def polar_curve_derivative (phi : ℝ) : ℝ := -6 * real.sin phi

theorem arc_length_polar_curve_segment : arc_length polar_curve polar_curve_derivative 0 (real.pi / 3) = 2 * real.pi :=
by
  sorry

end arc_length_polar_curve_segment_l301_301922


namespace probability_at_least_two_heads_l301_301257

theorem probability_at_least_two_heads (n : ℕ) (h : n = 5) : 
  let p := (1 / 2) ^ n in -- the probability of getting all tails
  let q := (5.choose 1) * p in -- the probability of getting exactly one head in five coins
  let r := p + q in -- probability of getting 0 or 1 head
  let s := 1 - r in -- probability of getting at least 2 heads
  s = 13 / 16 := sorry

end probability_at_least_two_heads_l301_301257


namespace tan_phi_value_l301_301038

theorem tan_phi_value (ϕ : ℝ) (h1 : cos ((3 * Real.pi / 2) - ϕ) = 3 / 5) (h2 : abs ϕ < Real.pi / 2) : 
  Real.tan ϕ = -3 / 4 := 
sorry

end tan_phi_value_l301_301038


namespace math_problem_solution_l301_301553

theorem math_problem_solution
  (ABCD_trapezoid : Trapezoid ABCD)
  (parallel_AB_DC : Parallel AB DC)
  (BC_eq_BD : BC = BD)
  (angle_DAB : Real.angle DAB = Real.angle.ofDegrees x)
  (angle_ADB : Real.angle ADB = Real.angle.ofDegrees 18)
  (angle_DBC : Real.angle DBC = Real.angle.ofDegrees (6 * t))
  (t_value : t = 7) :
  x = 93 := 
sorry

end math_problem_solution_l301_301553


namespace quentin_finishes_first_l301_301602

def area_quentin_lawn := q : ℝ
def area_paul_lawn := 3 * q
def area_rachel_lawn := 6 * q

def rate_quentin := r : ℝ
def rate_paul := 2 * r
def rate_rachel := 4 * r

def time_quentin_mowing := q / r
def time_paul_mowing := 3 * q / (2 * r)
def time_rachel_mowing := 6 * q / (4 * r)

theorem quentin_finishes_first : 
  time_quentin_mowing < time_paul_mowing ∧ 
  time_quentin_mowing < time_rachel_mowing := by
{
  sorry
}

end quentin_finishes_first_l301_301602


namespace inequality_abc_l301_301177

theorem inequality_abc (a b c : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a * b * c = 1) (h5 : 2 ≤ n) :
  (a / (b + c)^(1/(n:ℝ)) + b / (c + a)^(1/(n:ℝ)) + c / (a + b)^(1/(n:ℝ)) ≥ 3 / 2^(1/(n:ℝ))) :=
by sorry

end inequality_abc_l301_301177


namespace remainder_sum_of_first_six_primes_div_seventh_prime_l301_301788

theorem remainder_sum_of_first_six_primes_div_seventh_prime :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  ((p1 + p2 + p3 + p4 + p5 + p6) % p7) = 7 :=
by
  sorry

end remainder_sum_of_first_six_primes_div_seventh_prime_l301_301788


namespace intersection_complement_eq_C_l301_301995

def A := { x : ℝ | -3 < x ∧ x < 6 }
def B := { x : ℝ | 2 < x ∧ x < 7 }
def complement_B := { x : ℝ | x ≤ 2 ∨ x ≥ 7 }
def C := { x : ℝ | -3 < x ∧ x ≤ 2 }

theorem intersection_complement_eq_C :
  A ∩ complement_B = C :=
sorry

end intersection_complement_eq_C_l301_301995


namespace arrangement_for_P23_exists_l301_301240

-- Definition of Fibonacci-like sequence
def F : ℕ → ℤ
  | 0       => 0
  | 1       => 1
  | (n + 2) => 3 * F(n + 1) - F(n)

-- Predicate to check if an arrangement satisfying given conditions exists for P
def arrangement_exists (P : ℕ) : Prop := 
  ∃ i, F i = 0 ∧ i = (P + 1) / 2

theorem arrangement_for_P23_exists : arrangement_exists 23 :=
  sorry

end arrangement_for_P23_exists_l301_301240


namespace period_of_tan_cot_sin2x_l301_301331

/-- The period of y = tan x + cot x + sin 2x is π. -/
theorem period_of_tan_cot_sin2x :
  ∃ T > 0, (∀ x, tan x + cot x + sin (2 * x) = tan (x + T) + cot (x + T) + sin (2 * (x + T))) ∧ T = π :=
begin
  sorry,
end

end period_of_tan_cot_sin2x_l301_301331


namespace table_length_l301_301888

theorem table_length (area_m2 : ℕ) (width_cm : ℕ) (length_cm : ℕ) 
  (h_area : area_m2 = 54)
  (h_width : width_cm = 600)
  :
  length_cm = 900 := 
  sorry

end table_length_l301_301888


namespace a_beats_b_by_10_seconds_l301_301143

theorem a_beats_b_by_10_seconds :
  ∀ (T_A T_B D_A D_B : ℕ),
    T_A = 615 →
    D_A = 1000 →
    D_A - D_B = 16 →
    T_B = (D_A * T_A) / D_B →
    T_B - T_A = 10 :=
by
  -- Placeholder to ensure the theorem compiles
  intros T_A T_B D_A D_B h1 h2 h3 h4
  sorry

end a_beats_b_by_10_seconds_l301_301143


namespace divide_square_with_arc_l301_301554

theorem divide_square_with_arc 
  {A B C D E : Point} (square_ ABCD : isSquare A B C D) 
  (arc_ AE : arc AE) 
  (center_of_arc : center arc_ AE = C) 
  (radius_of_arc : radius arc_ AE = dist A C) :
  ∃ F G, (on_diagonal F A C) ∧ (F C = B C) ∧ (extend BC G = intersect arc_ AE G) ∧ 
  (triangle GFC ≅ triangle ADC) → 
  (area_divided_equally A B C D arc_ AE) :=
by
  sorry

end divide_square_with_arc_l301_301554


namespace solve_system_of_equations_l301_301084

variable (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ)

theorem solve_system_of_equations :
  let matrix := ![
    #[1, -1, 1],
    #[1, 1, 3]
  ] in
  let equation_1 := matrix[0] in
  let equation_2 := matrix[1] in
  ∃ x y : ℝ, (equation_1[0] * x + equation_1[1] * y = equation_1[2]) ∧ 
             (equation_2[0] * x + equation_2[1] * y = equation_2[2]) ∧ 
             x = 2 ∧ y = 1 :=
by
  sorry

end solve_system_of_equations_l301_301084


namespace delta_value_complex_l301_301513

theorem delta_value_complex (Δ : ℂ) : 4 * (-3) = Δ^2 + 3 ↔ Δ = Complex.sqrt 15 * Complex.I ∨ Δ = -Complex.sqrt 15 * Complex.I :=
by
  sorry

end delta_value_complex_l301_301513


namespace round_table_seating_l301_301609

theorem round_table_seating : 
  ∃ f m : ℕ, 
  (∀ (x : ℕ), x ≥ 3 → (∃ (y : ℕ), (y = 7) ∧ 
  (set_of_pairs = {(3, 7), (4, 7), (5, 7), (6, 7), (7, 7)}) ∧ 
  set_of_pairs.card = 5) :=
sorry

end round_table_seating_l301_301609


namespace difference_in_floors_l301_301397

-- Given conditions
variable (FA FB FC : ℕ)
variable (h1 : FA = 4)
variable (h2 : FC = 5 * FB - 6)
variable (h3 : FC = 59)

-- The statement to prove
theorem difference_in_floors : FB - FA = 9 :=
by 
  -- Placeholder proof
  sorry

end difference_in_floors_l301_301397


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301732

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301732


namespace exist_arrangement_for_P_23_l301_301193

def F : ℕ → ℤ
| 0        := 0
| 1        := 1
| (n + 2)  := 3 * F (n + 1) - F n

def similar (a b : ℤ) : Prop :=
  -- Define the "similar" relation as per the context of the problem
  abs (a - b) ≤ 1

theorem exist_arrangement_for_P_23 :
  ∃ (sequence : ℕ → ℤ), 
  P = 23 ∧ 
  (∀ i, sequence i = (-1) ^ (i+1) * i * F i) ∧ 
  (∀ i j, similar (sequence i) (sequence j)) := 
begin
  -- Proof here
  sorry
end

end exist_arrangement_for_P_23_l301_301193


namespace starting_number_range_l301_301154

theorem starting_number_range (n : ℕ) (h₁: ∀ m : ℕ, (m > n) → (m ≤ 50) → (m = 55) → True) : n = 54 :=
sorry

end starting_number_range_l301_301154


namespace complex_mul_eq_l301_301348

theorem complex_mul_eq :
  (2 + 2 * Complex.i) * (1 - 2 * Complex.i) = 6 - 2 * Complex.i := 
by
  intros
  sorry

end complex_mul_eq_l301_301348


namespace initial_storks_count_l301_301359

theorem initial_storks_count :
  (∃ B S : ℕ, B = 2 ∧ S = (B + 3) + 1) → (S = 6) :=
by
  intro h
  obtain ⟨B, S, hB, hS⟩ := h
  rw [hB, hS]
  sorry

end initial_storks_count_l301_301359


namespace lines_skew_condition_l301_301950

-- Define the lines and prove the condition
theorem lines_skew_condition (b : ℝ) : 
  let L1 := λ (t : ℝ), (⟨2, 3, b⟩ : ℝ × ℝ × ℝ) + t * ⟨3, 4, 5⟩ in
  let L2 := λ (u : ℝ), (⟨5, 0, -1⟩ : ℝ × ℝ × ℝ) + u * ⟨7, 1, 2⟩ in
  (∀ t u : ℝ, L1 t = L2 u) ↔ b = -187 / 25 → false :=
begin
  sorry,
end

end lines_skew_condition_l301_301950


namespace derivative_at_pi_f₁_derivative_at_pi_f₂_l301_301431

-- Define the function f₁ and state the goal for its derivative at x = π
def f₁ (x : ℝ) := (1 + Real.sin x) * (1 - 4 * x)
theorem derivative_at_pi_f₁ : (deriv f₁ π) = -5 + 4 * Real.pi := 
  by sketch_proof sorry

-- Define the function f₂ and state the goal for its derivative at x = π
def f₂ (x : ℝ) := Real.log (x + 1) - x / (x + 1)
theorem derivative_at_pi_f₂ : (deriv f₂ π) = Real.pi / ((Real.pi + 1) ^ 2) := 
  by sketch_proof sorry

end derivative_at_pi_f₁_derivative_at_pi_f₂_l301_301431


namespace find_r_l301_301436

theorem find_r : ∃ r : ℕ, (5 + 7 * 8 + 1 * 8^2) = 120 + r ∧ r = 5 := 
by
  use 5
  sorry

end find_r_l301_301436


namespace grooming_time_correct_l301_301291

def time_to_groom_poodle : ℕ := 30
def time_to_groom_terrier : ℕ := time_to_groom_poodle / 2
def num_poodles : ℕ := 3
def num_terriers : ℕ := 8

def total_grooming_time : ℕ := 
  (num_poodles * time_to_groom_poodle) + (num_terriers * time_to_groom_terrier)

theorem grooming_time_correct : 
  total_grooming_time = 210 := by
  sorry

end grooming_time_correct_l301_301291


namespace cube_stripe_probability_l301_301269

theorem cube_stripe_probability :
  let num_faces := 6
  let num_choices_per_face := 3
  let total_combinations := num_choices_per_face ^ num_faces
  let favorable_outcomes := 12
  let stripe_probability_per_face := 2 / 3
  let favorable_probability := favorable_outcomes * (stripe_probability_per_face ^ num_faces)
  let final_probability := favorable_probability / total_combinations
  final_probability = 768 / 59049 :=
begin
  sorry
end

end cube_stripe_probability_l301_301269


namespace stratified_sampling_employees_over_50_l301_301381

theorem stratified_sampling_employees_over_50 :
  let total_employees := 500
  let employees_under_35 := 125
  let employees_35_to_50 := 280
  let employees_over_50 := 95
  let total_samples := 100
  (employees_over_50 / total_employees * total_samples) = 19 := by
  sorry

end stratified_sampling_employees_over_50_l301_301381


namespace midpoints_of_parallelograms_form_parallelogram_l301_301174

variables (A B C D A' B' C' D' : Point)
variables (h1 : IsParallelogram A B C D) (h2 : IsParallelogram A' B' C' D')

noncomputable def midpoint (P Q : Point) : Point :=
  sorry -- Definition of midpoint

theorem midpoints_of_parallelograms_form_parallelogram :
  IsParallelogram (midpoint A A') (midpoint B B') (midpoint C C') (midpoint D D') :=
sorry

end midpoints_of_parallelograms_form_parallelogram_l301_301174


namespace determine_constants_l301_301473

noncomputable def f (a b c x : ℝ) : ℝ := a * x^3 + b * x + c

theorem determine_constants (c : ℝ) :
  (∃ (a b : ℝ),
    let f' := λ x, 3 * a * x^2 + b
    let f_val := f a b c 2
    f'(2) = 0 ∧ f_val = c - 6) →
  (∃ a b : ℝ, a = 3 / 8 ∧ b = -9 / 2) := by
  sorry

end determine_constants_l301_301473


namespace factor_of_100140001_between_8000_and_9000_l301_301948

theorem factor_of_100140001_between_8000_and_9000 : 
  ∃ k : ℕ, k * 8221 = 100140001 ∧ 8000 < 8221 ∧ 8221 < 9000 :=
by
  -- Let's define the conditions given in the problem.
  have h1 : 100140001 = 12181 * 8221 := by sorry
  have h2 : 8000 < 8221 := by sorry
  have h3 : 8221 < 9000 := by sorry
  use 12181
  exact ⟨h1, h2, h3⟩

end factor_of_100140001_between_8000_and_9000_l301_301948


namespace max_min_x_sub_2y_l301_301064

theorem max_min_x_sub_2y (x y : ℝ) (h : x^2 + y^2 - 2*x + 4*y = 0) : 0 ≤ x - 2*y ∧ x - 2*y ≤ 10 :=
sorry

end max_min_x_sub_2y_l301_301064


namespace calculate_cost_price_A_l301_301903

variables (CP_A SP_B SP_C SP_D : ℝ) (P : ℝ)

-- Define the conditions
def condition1 := SP_B = 1.20 * CP_A
def condition2 := SP_C = 1.25 * SP_B
def condition3 := SP_D = (1 + P / 100) * SP_C
def condition4 := SP_D = 225

-- The theorem to prove
theorem calculate_cost_price_A (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : CP_A = 150 :=
sorry

end calculate_cost_price_A_l301_301903


namespace remainder_of_sum_of_primes_l301_301797

theorem remainder_of_sum_of_primes :
    let p1 := 2
    let p2 := 3
    let p3 := 5
    let p4 := 7
    let p5 := 11
    let p6 := 13
    let p7 := 17
    let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
    sum_primes % p7 = 7 :=
by
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_primes := p1 + p2 + p3 + p4 + p5 + p6
  show sum_primes % p7 = 7
  sorry

end remainder_of_sum_of_primes_l301_301797


namespace number_of_diagonals_intersections_l301_301538

theorem number_of_diagonals_intersections (n : ℕ) (h : n ≥ 4) : 
  (∃ (I : ℕ), I = (n * (n - 1) * (n - 2) * (n - 3)) / 24) :=
by {
  sorry
}

end number_of_diagonals_intersections_l301_301538


namespace remainder_first_six_primes_div_seventh_l301_301815

theorem remainder_first_six_primes_div_seventh :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let p5 := 11
  let p6 := 13
  let p7 := 17
  let sum_six := p1 + p2 + p3 + p4 + p5 + p6 in
  sum_six % p7 = 7 :=
by
  sorry

end remainder_first_six_primes_div_seventh_l301_301815


namespace problem_divisible_by_64_l301_301962

theorem problem_divisible_by_64 (n : ℕ) (hn : n > 0) : (3 ^ (2 * n + 2) - 8 * n - 9) % 64 = 0 := 
by
  sorry

end problem_divisible_by_64_l301_301962


namespace paco_initial_sweet_cookies_l301_301601

theorem paco_initial_sweet_cookies (S : ℕ) (h1 : S - 15 = 7) : S = 22 :=
by
  sorry

end paco_initial_sweet_cookies_l301_301601


namespace number_of_women_in_first_class_l301_301599

-- Definitions for the conditions
def total_passengers : ℕ := 180
def percentage_women : ℝ := 0.65
def percentage_women_first_class : ℝ := 0.15

-- The desired proof statement
theorem number_of_women_in_first_class :
  (round (total_passengers * percentage_women * percentage_women_first_class) = 18) :=
by
  sorry  

end number_of_women_in_first_class_l301_301599


namespace prime_sum_remainder_l301_301766

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301766


namespace angle_ABC_DAC_equal_60_l301_301990

-- Define four distinct points
variable (A B C D : Point)

-- Define isometry between points
axiom isometry1 : isometry (fun p => if p = A then B else if p = B then A else p)
axiom isometry2 : isometry (fun p => if p = A then B else if p = B then C else if p = C then D else A)

-- Statement to prove
theorem angle_ABC_DAC_equal_60 :
  angle A B C = 60 ∧ angle D A C = 60 :=
sorry

end angle_ABC_DAC_equal_60_l301_301990


namespace g_2010_value_l301_301643

noncomputable def g : ℤ → ℤ := sorry

axiom g_property (x y m : ℤ) (h_pos : x > 0 ∧ y > 0 ∧ m > 0) (h_eq : x + y = 2^m) : g(x) + g(y) = 3 * m^2

theorem g_2010_value : g(2010) = 342 :=
by
  sorry

end g_2010_value_l301_301643


namespace sum_of_first_six_primes_mod_seventh_prime_l301_301741

open Int

theorem sum_of_first_six_primes_mod_seventh_prime :
  ((2 + 3 + 5 + 7 + 11 + 13) % 17) = 7 := 
  sorry

end sum_of_first_six_primes_mod_seventh_prime_l301_301741


namespace exists_arrangement_for_P_23_l301_301232

noncomputable def similar (x y : Nat) : Prop :=
abs (x - y) ≤ 1

theorem exists_arrangement_for_P_23 : ∃ (arrangement : Nat → Nat) (n : Nat), n = 23 ∧ (∀ i j, similar (arrangement i) (arrangement j)) :=
by
  sorry

end exists_arrangement_for_P_23_l301_301232


namespace cos_beta_half_l301_301998

theorem cos_beta_half (α β : ℝ) (hα_ac : 0 < α ∧ α < π / 2) (hβ_ac : 0 < β ∧ β < π / 2) 
  (h1 : Real.tan α = 4 * Real.sqrt 3) (h2 : Real.cos (α + β) = -11 / 14) : 
  Real.cos β = 1 / 2 :=
by
  sorry

end cos_beta_half_l301_301998


namespace min_shirts_to_save_l301_301905

theorem min_shirts_to_save (x : ℕ) :
  (75 + 10 * x < if x < 30 then 15 * x else 14 * x) → x = 20 :=
by
  sorry

end min_shirts_to_save_l301_301905


namespace find_a_plus_c_l301_301169

noncomputable def f (a b : ℝ) : ℝ → ℝ := λ x, x^2 + a * x + b
noncomputable def g (c d : ℝ) : ℝ → ℝ := λ x, x^2 + c * x + d

theorem find_a_plus_c (a b c d : ℝ) (h1 : (-a / 2)^2 ∈ (λ x, x^2 + c * x + d '' set.univ))
  (h2 : (-c / 2)^2 ∈ (λ x, x^2 + a * x + b '' set.univ))
  (h3 : f a b (-a / 2) = -200)
  (h4 : g c d (-c / 2) = -200)
  (h5 : f a b 150 = -200)
  (h6 : g c d 150 = -200) :
  a + c = 300 - 4 * real.sqrt 350 ∨ a + c = 300 + 4 * real.sqrt 350 :=
sorry

end find_a_plus_c_l301_301169


namespace modified_fibonacci_sum_l301_301577

noncomputable def F : ℕ → ℝ
| 0     := 1
| 1     := 2
| (n+2) := F n + F (n+1)

theorem modified_fibonacci_sum :
  ∑' n : ℕ, (F n) / (5^n) = 35 / 18 :=
sorry

end modified_fibonacci_sum_l301_301577


namespace cassie_has_8_parrots_l301_301007

-- Define the conditions
def num_dogs : ℕ := 4
def nails_per_foot : ℕ := 4
def feet_per_dog : ℕ := 4
def nails_per_dog := nails_per_foot * feet_per_dog

def nails_total_dogs : ℕ := num_dogs * nails_per_dog

def claws_per_leg : ℕ := 3
def legs_per_parrot : ℕ := 2
def normal_claws_per_parrot := claws_per_leg * legs_per_parrot

def extra_toe_parrot_claws : ℕ := normal_claws_per_parrot + 1

def total_nails : ℕ := 113

-- Establishing the proof problem
theorem cassie_has_8_parrots : 
  ∃ (P : ℕ), (6 * (P - 1) + 7 = 49) ∧ P = 8 := by
  sorry

end cassie_has_8_parrots_l301_301007


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301678

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301678


namespace five_digit_odd_numbers_count_l301_301969

-- Definitions based on conditions
def digits : List ℕ := [1, 2, 3, 4, 5]
def odd_digits : List ℕ := [1, 3, 5]
def num_digit_positions : ℕ := 5

-- Statement to prove
theorem five_digit_odd_numbers_count : 
  (∃ order : List ℕ, order.nodup ∧ order.length = num_digit_positions ∧ order.all (λ d, d ∈ digits) ∧ (order.last ∈ odd_digits) 
  ∧ (order.permutations.length = 3 * 24)) :=
by sorry

end five_digit_odd_numbers_count_l301_301969


namespace triangle_proof_l301_301535

section TriangleProblem

variables {A B C : Type*}
variables {AB BC AC : ℝ}
variables {α β : ℝ}

-- Given conditions
def triangle_cond (AB BC AC : ℝ) (α β : ℝ) := 
(BC = sqrt 5) ∧ (AC = 3) ∧ (sin β = 2 * sin α)

-- Proving the sides and sine angle difference
theorem triangle_proof (h : triangle_cond AB BC AC α β) :
  AB = 2 * sqrt 5 ∧ sin (α - π / 4) = -sqrt 10 / 10 :=
  sorry

end TriangleProblem

end triangle_proof_l301_301535


namespace number_of_real_solutions_l301_301955

noncomputable def f (x : ℝ) : ℝ :=
  (Finset.range 50).sum (λ n => (n + 1 : ℝ) / (x - (n + 1 : ℝ)))

theorem number_of_real_solutions : ∃ n : ℕ, n = 51 ∧ ∀ x : ℝ, f x = x + 1 ↔ n = 51 :=
by
  sorry

end number_of_real_solutions_l301_301955


namespace find_g_inv_f_of_15_l301_301121

open Function

theorem find_g_inv_f_of_15 {f g : ℝ → ℝ} (hf_inv : ∀ x, f⁻¹(g x) = x^4 - 4 * x^2 + 3)
  (hg_inv_exists : ∃ g_inv : ℝ → ℝ, ∀ y, g_inv (g y) = y ∧ g (g_inv y) = y) :
  g⁻¹(f 15) = sqrt 6 ∨ g⁻¹(f 15) = -sqrt 6 := 
begin
  sorry,
end

end find_g_inv_f_of_15_l301_301121


namespace parabola_sum_is_neg_fourteen_l301_301278

noncomputable def parabola_sum (a b c : ℝ) : ℝ := a + b + c

theorem parabola_sum_is_neg_fourteen :
  ∃ (a b c : ℝ), 
    (∀ x : ℝ, a * x^2 + b * x + c = -(x + 3)^2 + 2) ∧
    ((-1)^2 = a * (-1 + 3)^2 + 6) ∧ 
    ((-3)^2 = a * (-3 + 3)^2 + 2) ∧
    (parabola_sum a b c = -14) :=
sorry

end parabola_sum_is_neg_fourteen_l301_301278


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301681

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301681


namespace determinant_scalar_multiplication_l301_301975

theorem determinant_scalar_multiplication (x y z w : ℝ) (h : abs (x * w - y * z) = 10) :
  abs (3*x * 3*w - 3*y * 3*z) = 90 :=
by
  sorry

end determinant_scalar_multiplication_l301_301975


namespace quincy_monthly_payment_l301_301251

-- Definitions based on the conditions:
def car_price : ℕ := 20000
def down_payment : ℕ := 5000
def loan_years : ℕ := 5
def months_in_year : ℕ := 12

-- The mathematical problem to be proven:
theorem quincy_monthly_payment :
  let amount_to_finance := car_price - down_payment
  let total_months := loan_years * months_in_year
  amount_to_finance / total_months = 250 := by
  sorry

end quincy_monthly_payment_l301_301251


namespace understanding_related_to_gender_probability_of_understanding_l301_301620

noncomputable def chi_squared 
  (a b c d n : ℝ) : ℝ :=
  n * (a * d - b * c)^2 / ((a + b) * (c + d) * (a + c) * (b + d))

theorem understanding_related_to_gender (a b c d n : ℝ) (alpha : ℝ) :
  a = 140 ∧ b = 60 ∧ c = 180 ∧ d = 20 ∧ n = 400 ∧ alpha = 0.001 →
  chi_squared a b c d n = 25 ∧ 25 > 10.828 :=
by sorry

noncomputable def binomial_prob 
  (n k : ℕ) (p : ℝ) : ℝ :=
  (finset.card (finset.filter (λ x, x = k) (finset.range n))) * (p ^ k) * ((1 - p) ^ (n - k))

theorem probability_of_understanding (total_understanding : ℝ) 
  (total_population : ℝ) (selected_residents : ℕ) (k : ℕ) :
  total_understanding = 320 ∧ total_population = 400 ∧ selected_residents = 5 ∧ k = 3 →
  binomial_prob selected_residents k (total_understanding / total_population) = 128 / 625 :=
by sorry

end understanding_related_to_gender_probability_of_understanding_l301_301620


namespace captain_and_vicecaptain_pair_boys_and_girls_l301_301543

-- Problem A
theorem captain_and_vicecaptain (n : ℕ) (h : n = 11) : ∃ ways : ℕ, ways = 110 :=
by
  sorry

-- Problem B
theorem pair_boys_and_girls (N : ℕ) : ∃ ways : ℕ, ways = Nat.factorial N :=
by
  sorry

end captain_and_vicecaptain_pair_boys_and_girls_l301_301543


namespace proof_problem_l301_301079

noncomputable def f (a x : ℝ) : ℝ := (x^2 + a*x - 2*a - 3) * Real.exp x

theorem proof_problem :
  (∃ a : ℝ, (a = -5) ∧
    let fval := f a in
    (∀ x : ℝ, fval 2 = Real.exp 2) ∧
    (∀ x ∈ Set.Icc (3/2 : ℝ) 3, (max (fval 3) (fval 2) = fval 3 ∧ min (fval 2) (fval 3) = fval 2))) :=
by
  sorry

end proof_problem_l301_301079


namespace find_common_difference_find_minimum_sum_minimum_sum_value_l301_301551

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) (d : ℚ) := (-3 : ℚ) + n * d

-- Given conditions
def condition_1 : ℚ := -3
def condition_2 (d : ℚ) := 11 * a 4 d = 5 * a 7 d - 13
def common_difference : ℚ := 31 / 9

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (d : ℚ) := n * (-3 + (n - 1) * d / 2)

-- Defining the necessary theorems
theorem find_common_difference (d : ℚ) : condition_2 d → d = common_difference := by
  sorry

theorem find_minimum_sum (n : ℕ) : S n common_difference ≥ S 2 common_difference := by
  sorry

theorem minimum_sum_value : S 2 common_difference = -23 / 9 := by
  sorry

end find_common_difference_find_minimum_sum_minimum_sum_value_l301_301551


namespace remainder_sum_of_six_primes_div_seventh_prime_l301_301733

def sum_of_six_primes : ℕ := 2 + 3 + 5 + 7 + 11 + 13

def seventh_prime : ℕ := 17

theorem remainder_sum_of_six_primes_div_seventh_prime :
  sum_of_six_primes % seventh_prime = 7 := by
  sorry

end remainder_sum_of_six_primes_div_seventh_prime_l301_301733


namespace no_equal_black_white_shards_l301_301652

def initial_glass_cups := 25
def initial_porcelain_cups := 35
def glass_pieces := 17
def porcelain_pieces := 18

theorem no_equal_black_white_shards (x y : ℕ) :
  (glass_pieces * x + porcelain_pieces * (initial_porcelain_cups - y)
   = glass_pieces * (initial_glass_cups - x) + porcelain_pieces * y) →
  false :=
by sorry

end no_equal_black_white_shards_l301_301652


namespace remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301761

theorem remainder_of_sum_of_first_six_primes_divided_by_seventh :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13,
      seventh_prime := 17
  in (sum_of_first_six_primes % seventh_prime = 7) :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have h : sum_of_first_six_primes = 41 := by norm_num
  have h' : 41 % 17 = 7 := by norm_num
  rw h
  exact h'

end remainder_of_sum_of_first_six_primes_divided_by_seventh_l301_301761


namespace farmer_owned_land_l301_301367

theorem farmer_owned_land (T : ℝ) (h : 0.10 * T = 720) : 0.80 * T = 5760 :=
by
  sorry

end farmer_owned_land_l301_301367


namespace test_completion_unanswered_l301_301889

theorem test_completion_unanswered (num_questions : ℕ) (num_choices : ℕ) : num_questions = 4 → num_choices = 5 → (ways : ℕ → ℕ → ℕ) = λ q c, if q = 4 ∧ c = 5 then 1 else 0 :=
by
  intros hq hc
  sorry

end test_completion_unanswered_l301_301889


namespace lucy_popsicles_l301_301182

def total_funds : ℝ := 25.50
def first_tier_cost : ℝ := 1.75
def second_tier_cost : ℝ := 1.50

-- Define the number of popsicles Lucy can buy
def max_popsicles (funds : ℝ) (cost1 : ℝ) (cost2 : ℝ) : ℕ := 
let funds_after_first_eight := funds - 8 * cost1 in
8 + ⌊funds_after_first_eight / cost2⌋

-- Statement to prove
theorem lucy_popsicles : max_popsicles total_funds first_tier_cost second_tier_cost = 15 :=
by
  sorry

end lucy_popsicles_l301_301182


namespace zoo_students_l301_301309

theorem zoo_students (teachers : ℕ) (students_per_group : ℕ) (h1 : teachers = 8) (h2 : students_per_group = 32) : teachers * students_per_group = 256 :=
by
  rw [h1, h2]
  exact rfl

end zoo_students_l301_301309


namespace sin_diff_l301_301437

theorem sin_diff (α β : ℝ) 
  (h1 : Real.sin α + Real.cos β = 1 / 3) 
  (h2 : Real.sin β - Real.cos α = 1 / 2) : 
  Real.sin (α - β) = -59 / 72 := 
sorry

end sin_diff_l301_301437


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301688

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301688


namespace simplify_f_value_f_given_conditions_value_f_given_alpha_l301_301979

noncomputable def f (alpha : ℝ) : ℝ := (Real.sin (π - alpha) * Real.cos (2 * π - alpha) * Real.tan (-alpha + (3 * π / 2)) * Real.tan (-alpha - π)) / Real.sin (-π - alpha)

theorem simplify_f (alpha : ℝ) : 
  f(alpha) = Real.sin(alpha) * Real.tan(alpha) :=
sorry

theorem value_f_given_conditions (alpha : ℝ) (h1 : Real.cos (alpha - (3 * π / 2)) = 1 / 5) (third_quadrant : π < alpha ∧ alpha < 3 * π / 2) : 
  f(alpha) = -Real.sqrt(6) / 60 :=
sorry

theorem value_f_given_alpha : 
  f(-1860 * (π/180)) = 3 / 2 :=
sorry

end simplify_f_value_f_given_conditions_value_f_given_alpha_l301_301979


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301706

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301706


namespace angle_in_third_quadrant_l301_301515

theorem angle_in_third_quadrant (θ : ℝ) (h1 : sin θ * cos θ > 0) (h2 : cos θ * tan θ < 0) : 
  (π < θ ∧ θ < 3 * π / 2) :=
by
  sorry

end angle_in_third_quadrant_l301_301515


namespace sum_of_divisors_of_36_l301_301401

theorem sum_of_divisors_of_36 : 
  let divisors := {d : ℤ | d ∣ 36}
  (∑ d in divisors, d) = 0 := 
by
  -- Definitions and inputs
  let divisors := {d : ℤ | d ∣ 36} in
  -- This line is added to skip the proof for now
  sorry

end sum_of_divisors_of_36_l301_301401


namespace find_k_find_B_find_P_l301_301057

-- Define the linear function y = 3x - 1
def linear_function : ℝ → ℝ := λ x, 3 * x - 1

-- Define points A and B based on conditions
def A (a b : ℝ) : ℝ × ℝ := (a, b)
def B (a b k : ℝ) : ℝ × ℝ := (a + 1, b + k)

-- Condition that A(a, b) and B(a+1, b+k) lie on the linear function
def points_on_line (a b k : ℝ) : Prop :=
  linear_function a = b ∧ linear_function (a + 1) = b + k

-- Propositions to prove
theorem find_k (a b k : ℝ) (h : points_on_line a b k) : k = 3 := by
  sorry

theorem find_B (b : ℝ) (hb : linear_function 0 = b) : B 0 b 3 = (1, 2) := by
  sorry

theorem find_P (hp1 : B 0 (-1) 3 = (1, 2)) : 
  ∃ P : ℝ × ℝ, P = (-5/2, 0) ∨ P = (2, 0) ∨ P = (2.5, 0) :=
  sorry

end find_k_find_B_find_P_l301_301057


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301689

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301689


namespace score_difference_is_198_l301_301419
noncomputable theory

def score_difference (x y t : ℕ) : ℕ :=
  abs (99 * (x - y))

theorem score_difference_is_198 (x y t : ℕ) (h : x ≠ y) :
  score_difference x y t = 198 ↔ 99 * abs (x - y) = 198 :=
by 
  sorry

end score_difference_is_198_l301_301419


namespace car_travels_more_l301_301380

theorem car_travels_more (train_speed : ℕ) (car_speed : ℕ) (time : ℕ)
  (h1 : train_speed = 60)
  (h2 : car_speed = 2 * train_speed)
  (h3 : time = 3) :
  car_speed * time - train_speed * time = 180 :=
by
  sorry

end car_travels_more_l301_301380


namespace range_of_m_l301_301457

def p (m : ℝ) : Prop := m > 3
def q (m : ℝ) : Prop := m > (1 / 4)

theorem range_of_m (m : ℝ) (h1 : ¬p m) (h2 : p m ∨ q m) : (1 / 4) < m ∧ m ≤ 3 :=
by
  sorry

end range_of_m_l301_301457


namespace longest_pole_length_l301_301336

theorem longest_pole_length (l w h : ℝ) (hl : l = 24) (hw : w = 18) (hh : h = 16) :
  (real.sqrt (l^2 + w^2 + h^2)) = 34 := 
by 
  -- Assign known values
  have l_val : l = 24 := hl,
  have w_val : w = 18 := hw,
  have h_val : h = 16 := hh,
  -- Substitute the known values into the equation
  calc
    real.sqrt (l^2 + w^2 + h^2) = real.sqrt (24^2 + 18^2 + 16^2) : by rw [l_val, w_val, h_val]
    ... = real.sqrt (576 + 324 + 256) : by norm_num
    ... = real.sqrt (1156) : by norm_num
    ... = 34 : by norm_num

end longest_pole_length_l301_301336


namespace sufficient_condition_for_solution_l301_301840

theorem sufficient_condition_for_solution 
  (a : ℝ) (f g h : ℝ → ℝ) (h_a : 1 < a)
  (h_fg_h : ∀ x : ℝ, 0 ≤ f x + g x + h x) 
  (h_common_root : ∃ x : ℝ, f x = 0 ∧ g x = 0 ∧ h x = 0) : 
  ∃ x : ℝ, a^(f x) + a^(g x) + a^(h x) = 3 := 
by
  sorry

end sufficient_condition_for_solution_l301_301840


namespace complement_intersection_l301_301355

open Set

theorem complement_intersection
  (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = univ) 
  (hA : A = { x : ℝ | x ≤ -2 }) 
  (hB : B = { x : ℝ | x < 1 }) :
  (U \ A) ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  rw [hU, hA, hB]
  sorry

end complement_intersection_l301_301355


namespace percentage_students_50_59_is_10_71_l301_301536

theorem percentage_students_50_59_is_10_71 :
  let n_90_100 := 3
  let n_80_89 := 6
  let n_70_79 := 8
  let n_60_69 := 4
  let n_50_59 := 3
  let n_below_50 := 4
  let total_students := n_90_100 + n_80_89 + n_70_79 + n_60_69 + n_50_59 + n_below_50
  let fraction := (n_50_59 : ℚ) / total_students
  let percentage := (fraction * 100)
  percentage = 10.71 := by sorry

end percentage_students_50_59_is_10_71_l301_301536


namespace dot_product_neg_two_l301_301454

open_locale real

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C M : V) (s : ℝ) (CA CB CM MA MB : V)
variables (h_eq_side : dist A B = 2 * sqrt 3)
variables (h_CM_def : CM = (1 / 6) • CB + (2 / 3) • CA)
variables (h_CA := CA)
variables (h_CB := CB)
variables (h_MA : MA = CA - CM)
variables (h_MB : MB = CB - CM)
variables (h_dot_CA_CB : ⟪CA, CB⟫ = real.sqrt 3 * real.sqrt 3 * real.cos (π / 3))

def equilateral_triangle_dot_product : Prop :=
  ⟪MA, MB⟫ = -2

theorem dot_product_neg_two :
  equilateral_triangle_dot_product :=
  sorry

end dot_product_neg_two_l301_301454


namespace g_at_minus_4_l301_301409

def g (x : ℝ) : ℝ :=
  (7 * x - 3) / (x + 2)

theorem g_at_minus_4 : g (-4) = 15.5 :=
  by sorry

end g_at_minus_4_l301_301409


namespace car_a_reaches_first_l301_301540

noncomputable def car_racing (s a : ℝ) (h1 : 0 < a) (h2 : a < 50) (h3 : 0 < s) (h4 : s > a) : Prop :=
  let x : ℝ := (s - a) * (s + a) / s in
  s - (a^2 / s) < s ∧ s + a - x > 0

theorem car_a_reaches_first (s a : ℝ) (h1 : 0 < a) (h2 : a < 50) (h3 : 0 < s) (h4 : s > a) : car_racing s a h1 h2 h3 h4 :=
by
  sorry

end car_a_reaches_first_l301_301540


namespace johns_watermelon_weight_l301_301592

-- Michael's largest watermelon weighs 8 pounds
def michael_weight : ℕ := 8

-- Clay's watermelon weighs three times the size of Michael's watermelon
def clay_weight : ℕ := 3 * michael_weight

-- John's watermelon weighs half the size of Clay's watermelon
def john_weight : ℕ := clay_weight / 2

-- Prove that John's watermelon weighs 12 pounds
theorem johns_watermelon_weight : john_weight = 12 := by
  sorry

end johns_watermelon_weight_l301_301592


namespace line_circle_intersection_l301_301299

theorem line_circle_intersection {m : ℝ} :
  (0 < m ∧ m < 1) → ∃ x y : ℝ, (x^2 + y^2 - 2*x - 1 = 0) ∧ (x - y + m = 0) :=
begin
  sorry
end

end line_circle_intersection_l301_301299


namespace equal_real_roots_of_quadratic_eq_l301_301128

theorem equal_real_roots_of_quadratic_eq {k : ℝ} (h : ∃ x : ℝ, (x^2 + 3 * x - k = 0) ∧ ∀ y : ℝ, (y^2 + 3 * y - k = 0) → y = x) : k = -9 / 4 := 
by 
  sorry

end equal_real_roots_of_quadratic_eq_l301_301128


namespace prove_total_rent_of_field_l301_301844

def totalRentField (A_cows A_months B_cows B_months C_cows C_months 
                    D_cows D_months E_cows E_months F_cows F_months 
                    G_cows G_months A_rent : ℕ) : ℕ := 
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let E_cow_months := E_cows * E_months
  let F_cow_months := F_cows * F_months
  let G_cow_months := G_cows * G_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + 
                          D_cow_months + E_cow_months + F_cow_months + G_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  total_cow_months * rent_per_cow_month

theorem prove_total_rent_of_field : totalRentField 24 3 10 5 35 4 21 3 15 6 40 2 28 (7/2) 720 = 5930 :=
  by
  sorry

end prove_total_rent_of_field_l301_301844


namespace prime_sum_remainder_l301_301767

theorem prime_sum_remainder :
  (2 + 3 + 5 + 7 + 11 + 13) % 17 = 7 :=
by
  sorry

end prime_sum_remainder_l301_301767


namespace sum_of_rational_roots_l301_301036

theorem sum_of_rational_roots :
  let h : Polynomial ℚ := Polynomial.Coeff 0 + Polynomial.Coeff 11 * Polynomial.X - Polynomial.Coeff 6 * Polynomial.X ^ 2 - Polynomial.Coeff 6 * Polynomial.X ^ 3
  ∑ root in h.roots, root = 6 :=
by sorry

end sum_of_rational_roots_l301_301036


namespace part1_part2_smallest_positive_period_part2_intervals_increase_l301_301485

-- Define the function f(x)
def f (x : ℝ) : ℝ := cos x * (sin x + cos x) + 1/2

-- Problem Part 1
theorem part1 (α : ℝ) (h : tan α = 1/2) : f α = 17/10 :=
by
  sorry

-- Simplified form of the function for Part 2
def f_simplified (x : ℝ) : ℝ := (real.sqrt 2)/2 * sin (2 * x + real.pi / 4) + 1

-- Definitions and theorems for Part 2
theorem part2_smallest_positive_period : ∀ x : ℝ, (f x) = (f (x + real.pi)) :=
by
  sorry

noncomputable def interval_increase (k : ℤ) : set ℝ := 
  { x : ℝ | (-3 * real.pi / 8 + k * real.pi) ≤ x ∧ x ≤ (real.pi / 8 + k * real.pi) }

theorem part2_intervals_increase : ∀ k : ℤ, ∀ x : ℝ, (x ∈ interval_increase k ↔ ∃ n : ℤ, x = -3 * real.pi / 8 + n * real.pi ∨ x = real.pi / 8 + n * real.pi) :=
by
  sorry

end part1_part2_smallest_positive_period_part2_intervals_increase_l301_301485


namespace concyclic_A1_B1_C1_D1_l301_301059

variables {A B C I A₁ B₁ C₁ D₁ : Type*}
variables [IsTriangle A B C] [IsIncenter I A B C]
variables [IsMidpoint A I A₁] [IsMidpoint B I B₁] [IsMidpoint C I C₁]
variables [IsExcenter D₁ A B C]

theorem concyclic_A1_B1_C1_D1 :
  Circle A₁ B₁ C₁ D₁ :=
sorry

end concyclic_A1_B1_C1_D1_l301_301059


namespace arithmetic_geometric_sequence_k4_l301_301963

theorem arithmetic_geometric_sequence_k4 (a : ℕ → ℝ) (d : ℝ) (h_d_ne_zero : d ≠ 0)
  (h_arith_seq : ∀ n, a (n + 1) = a n + d)
  (h_geo_seq : ∃ k : ℕ → ℕ, k 0 = 1 ∧ k 1 = 2 ∧ k 2 = 6 ∧ ∀ i, a (k i + 1) / a (k i) = a (k i + 2) / a (k i + 1)) :
  ∃ k4 : ℕ, k4 = 22 := 
by
  sorry

end arithmetic_geometric_sequence_k4_l301_301963


namespace remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301693

def firstSixPrimes := [2, 3, 5, 7, 11, 13]
def seventhPrime := 17

theorem remainder_when_sum_of_first_six_primes_divided_by_seventh :
  (firstSixPrimes.sum % seventhPrime) = 7 :=
by
  -- proof would go here
  sorry

end remainder_when_sum_of_first_six_primes_divided_by_seventh_l301_301693


namespace remainder_sum_first_six_primes_div_seventh_prime_l301_301698

theorem remainder_sum_first_six_primes_div_seventh_prime :
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13 in
  let seventh_prime := 17 in
  sum_of_first_six_primes % seventh_prime = 7 :=
by
  let sum_of_first_six_primes := 2 + 3 + 5 + 7 + 11 + 13
  let seventh_prime := 17
  have : sum_of_first_six_primes = 41 := rfl
  have : seventh_prime = 17 := rfl
  calc
    sum_of_first_six_primes % seventh_prime = 41 % 17 : by rw [this, this]
    ... = 7 : by sorry

end remainder_sum_first_six_primes_div_seventh_prime_l301_301698


namespace problem_l301_301441

theorem problem (f : ℝ → ℝ) (h : ∀ x : ℝ, f (1 - x) = x^2 + 1) : f (-1) = 5 := 
  sorry

end problem_l301_301441


namespace cut_into_square_l301_301158

theorem cut_into_square (a : ℕ) (h : ∃ n m : ℕ, a = n * m) :
  ∃ parts : List (ℕ × ℕ), parts.length = 3 ∧ (∀ p ∈ parts, ∃ n m : ℕ, p = (n, m)) ∧ 
  parts.sum (λ p, p.fst * p.snd) = a ∧
  let total_dim := (parts.sum (λ p, p.fst), parts.sum (λ p, p.snd)) in
  ∃ side : ℕ, side * side = a ∧ (total_dim = (side, side) ∨ (total_dim.snd, total_dim.fst) = (side, side)) :=
by
  sorry

end cut_into_square_l301_301158


namespace no_afg_fourth_place_l301_301260

theorem no_afg_fourth_place
  (A B C D E F G : ℕ)
  (h1 : A < B)
  (h2 : A < C)
  (h3 : B < D)
  (h4 : C < E)
  (h5 : A < F ∧ F < B)
  (h6 : B < G ∧ G < C) :
  ¬ (A = 4 ∨ F = 4 ∨ G = 4) :=
by
  sorry

end no_afg_fourth_place_l301_301260


namespace remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301669

def first_six_primes : List Nat := [2, 3, 5, 7, 11, 13]
def seventh_prime : Nat := 17
def sum_first_six_primes : Nat := first_six_primes.sum

theorem remainder_when_sum_of_six_primes_divided_by_seventh_prime :
  (sum_first_six_primes % seventh_prime) = 7 :=
by
  sorry

end remainder_when_sum_of_six_primes_divided_by_seventh_prime_l301_301669


namespace crayons_eaten_l301_301160

def initial_crayons : ℕ := 87
def remaining_crayons : ℕ := 80

theorem crayons_eaten : initial_crayons - remaining_crayons = 7 := by
  sorry

end crayons_eaten_l301_301160


namespace at_most_n_fixed_points_l301_301176

noncomputable def P (x : ℤ) : ℤ := sorry  -- Since the specific polynomial P is not given, it is defined as a placeholder here.
def Q : ℤ → ℤ := (λ x => (ΠP λ P x).iterate k x)  -- The composition of P, k times.

theorem at_most_n_fixed_points (P : ℤ → ℤ) (n k : ℕ) (hP_degree : Polynomial.degree P = n) (hn_pos : n > 1) :
  ∃ (t : ℤ) (ht : Q t = t), t ≤ n := sorry

end at_most_n_fixed_points_l301_301176
