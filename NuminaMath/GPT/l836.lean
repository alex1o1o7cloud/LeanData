import Mathlib

namespace inequality_proof_l836_83639

theorem inequality_proof (n k : ℕ) (h₁ : 0 < n) (h₂ : 0 < k) (h₃ : k ≤ n) :
  1 + k / n ≤ (1 + 1 / n)^k ∧ (1 + 1 / n)^k < 1 + k / n + k^2 / n^2 :=
sorry

end inequality_proof_l836_83639


namespace find_missing_square_l836_83680

-- Defining the sequence as a list of natural numbers' squares
def square_sequence (n: ℕ) : ℕ := n * n

-- Proving the missing element in the given sequence is 36
theorem find_missing_square :
  (square_sequence 0 = 1) ∧ 
  (square_sequence 1 = 4) ∧ 
  (square_sequence 2 = 9) ∧ 
  (square_sequence 3 = 16) ∧ 
  (square_sequence 4 = 25) ∧ 
  (square_sequence 6 = 49) →
  square_sequence 5 = 36 :=
by {
  sorry
}

end find_missing_square_l836_83680


namespace mabel_shark_ratio_l836_83688

variables (F1 F2 sharks_total sharks_day1 sharks_day2 ratio : ℝ)
variables (fish_day1 := 15)
variables (shark_percentage := 0.25)
variables (total_sharks := 15)

noncomputable def ratio_of_fish_counts := (F2 / F1)

theorem mabel_shark_ratio 
    (fish_day1 : ℝ := 15)
    (shark_percentage : ℝ := 0.25)
    (total_sharks : ℝ := 15)
    (sharks_day1 := 0.25 * fish_day1)
    (sharks_day2 := total_sharks - sharks_day1)
    (F2 := sharks_day2 / shark_percentage)
    (ratio := F2 / fish_day1):
    ratio = 16 / 5 :=
by
  sorry

end mabel_shark_ratio_l836_83688


namespace area_of_region_l836_83672

theorem area_of_region :
  (∫ x, ∫ y in {y : ℝ | x^4 + y^4 = |x|^3 + |y|^3}, (1 : ℝ)) = 4 :=
sorry

end area_of_region_l836_83672


namespace tangent_slope_through_origin_l836_83644

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^a + 1

theorem tangent_slope_through_origin (a : ℝ) (h : curve a 1 = 2) 
  (tangent_passing_through_origin : ∀ y, (y - 2 = a * (1 - 0)) → y = 0): a = 2 := 
sorry

end tangent_slope_through_origin_l836_83644


namespace cone_lateral_surface_area_eq_sqrt_17_pi_l836_83612

theorem cone_lateral_surface_area_eq_sqrt_17_pi
  (r_cone r_sphere : ℝ) (h : ℝ)
  (V_sphere V_cone : ℝ)
  (h_cone_radius : r_cone = 1)
  (h_sphere_radius : r_sphere = 1)
  (h_volumes_eq : V_sphere = V_cone)
  (h_sphere_vol : V_sphere = (4 * π) / 3)
  (h_cone_vol : V_cone = (π * r_cone^2 * h) / 3) :
  (π * r_cone * (Real.sqrt (r_cone^2 + h^2))) = Real.sqrt 17 * π :=
sorry

end cone_lateral_surface_area_eq_sqrt_17_pi_l836_83612


namespace num_students_earning_B_l836_83693

variables (nA nB nC nF : ℕ)

-- Conditions from the problem
def condition1 := nA = 6 * nB / 10
def condition2 := nC = 15 * nB / 10
def condition3 := nF = 4 * nB / 10
def condition4 := nA + nB + nC + nF = 50

-- The theorem to prove
theorem num_students_earning_B (nA nB nC nF : ℕ) : 
  condition1 nA nB → 
  condition2 nC nB → 
  condition3 nF nB → 
  condition4 nA nB nC nF → 
  nB = 14 :=
by
  sorry

end num_students_earning_B_l836_83693


namespace hyperbola_focus_coordinates_l836_83645

open Real

theorem hyperbola_focus_coordinates :
  ∃ x y : ℝ, (2 * x^2 - y^2 + 8 * x + 4 * y - 28 = 0) ∧
           ((x = -2 - 4 * sqrt 3 ∧ y = 2) ∨ (x = -2 + 4 * sqrt 3 ∧ y = 2)) := by sorry

end hyperbola_focus_coordinates_l836_83645


namespace digit_after_decimal_is_4_l836_83661

noncomputable def sum_fractions : ℚ := (2 / 9) + (3 / 11)

theorem digit_after_decimal_is_4 :
  (sum_fractions - sum_fractions.floor) * 10 = 4 :=
by
  sorry

end digit_after_decimal_is_4_l836_83661


namespace bailey_towel_set_cost_l836_83682

def guest_bathroom_sets : ℕ := 2
def master_bathroom_sets : ℕ := 4
def cost_per_guest_set : ℝ := 40.00
def cost_per_master_set : ℝ := 50.00
def discount_rate : ℝ := 0.20

def total_cost_before_discount : ℝ := 
  (guest_bathroom_sets * cost_per_guest_set) + (master_bathroom_sets * cost_per_master_set)

def discount_amount : ℝ := total_cost_before_discount * discount_rate

def final_amount_spent : ℝ := total_cost_before_discount - discount_amount

theorem bailey_towel_set_cost : final_amount_spent = 224.00 := by sorry

end bailey_towel_set_cost_l836_83682


namespace number_of_logs_in_stack_l836_83653

theorem number_of_logs_in_stack :
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  total_logs = 114 := by
{
  let bottom := 15
  let top := 4
  let num_rows := bottom - top + 1
  let total_logs := num_rows * (bottom + top) / 2
  sorry
}

end number_of_logs_in_stack_l836_83653


namespace narrow_black_stripes_l836_83699

theorem narrow_black_stripes (w n b : ℕ) 
( h1 : b = w + 7 ) 
( h2 : w + n = b + 1 ) : 
n = 8 := 
sorry

end narrow_black_stripes_l836_83699


namespace b_share_220_l836_83614

theorem b_share_220 (A B C : ℝ) (h1 : A = B + 40) (h2 : C = A + 30) (h3 : B + A + C = 770) : B = 220 :=
by
  sorry

end b_share_220_l836_83614


namespace bob_weight_l836_83648

theorem bob_weight (j b : ℝ) (h1 : j + b = 200) (h2 : b - j = b / 3) : b = 120 :=
sorry

end bob_weight_l836_83648


namespace range_of_b_for_increasing_f_l836_83631

noncomputable def f (b x : ℝ) : ℝ :=
  if x > 1 then (2 * b - 1) / x + b + 3 else -x^2 + (2 - b) * x

theorem range_of_b_for_increasing_f :
  ∀ b : ℝ, (∀ x1 x2 : ℝ, x1 < x2 → f b x1 ≤ f b x2) ↔ -1/4 ≤ b ∧ b ≤ 0 := 
sorry

end range_of_b_for_increasing_f_l836_83631


namespace part1_part2_l836_83643

-- Part 1
theorem part1 (x y : ℝ) 
  (h1 : x + 2 * y = 9) 
  (h2 : 2 * x + y = 6) :
  (x - y = -3) ∧ (x + y = 5) :=
sorry

-- Part 2
theorem part2 (x y : ℝ) 
  (h1 : x + 2 = 5) 
  (h2 : y - 1 = 4) :
  x = 3 ∧ y = 5 :=
sorry

end part1_part2_l836_83643


namespace smallest_ducks_l836_83683

theorem smallest_ducks :
  ∃ D : ℕ, 
  ∃ C : ℕ, 
  ∃ H : ℕ, 
  (13 * D = 17 * C) ∧
  (11 * H = (6 / 5) * 13 * D) ∧
  (17 * C = (3 / 8) * 11 * H) ∧ 
  (13 * D = 520) :=
by 
  sorry

end smallest_ducks_l836_83683


namespace percent_of_100_is_30_l836_83679

theorem percent_of_100_is_30 : (30 / 100) * 100 = 30 := 
by
  sorry

end percent_of_100_is_30_l836_83679


namespace cylindrical_can_increase_l836_83616

theorem cylindrical_can_increase (R H y : ℝ)
  (h₁ : R = 5)
  (h₂ : H = 4)
  (h₃ : π * (R + y)^2 * (H + y) = π * (R + 2*y)^2 * H) :
  y = Real.sqrt 76 - 5 :=
by
  sorry

end cylindrical_can_increase_l836_83616


namespace find_other_number_product_find_third_number_sum_l836_83622

-- First Question
theorem find_other_number_product (x : ℚ) (h : x * (1/7 : ℚ) = -2) : x = -14 :=
sorry

-- Second Question
theorem find_third_number_sum (y : ℚ) (h : (1 : ℚ) + (-4) + y = -5) : y = -2 :=
sorry

end find_other_number_product_find_third_number_sum_l836_83622


namespace number_of_shortest_paths_l836_83638

-- Define the concept of shortest paths
def shortest_paths (m n : ℕ) : ℕ :=
  Nat.choose (m + n) m

-- State the theorem that needs to be proved
theorem number_of_shortest_paths (m n : ℕ) : shortest_paths m n = Nat.choose (m + n) m :=
by 
  sorry

end number_of_shortest_paths_l836_83638


namespace total_travel_time_l836_83663

noncomputable def washingtonToIdahoDistance : ℕ := 640
noncomputable def idahoToNevadaDistance : ℕ := 550
noncomputable def washingtonToIdahoSpeed : ℕ := 80
noncomputable def idahoToNevadaSpeed : ℕ := 50

theorem total_travel_time :
  (washingtonToIdahoDistance / washingtonToIdahoSpeed) + (idahoToNevadaDistance / idahoToNevadaSpeed) = 19 :=
by
  sorry

end total_travel_time_l836_83663


namespace altitude_of_triangle_l836_83656

theorem altitude_of_triangle (b h_t h_p : ℝ) (hb : b ≠ 0) 
  (area_eq : b * h_p = (1/2) * b * h_t) 
  (h_p_def : h_p = 100) : h_t = 200 :=
by
  sorry

end altitude_of_triangle_l836_83656


namespace max_m_minus_n_l836_83637

theorem max_m_minus_n (m n : ℝ) (h : (m + 1)^2 + (n + 1)^2 = 4) : m - n ≤ 2 * Real.sqrt 2 :=
by {
  -- Here is where the proof would take place.
  sorry
}

end max_m_minus_n_l836_83637


namespace classroomA_goal_is_200_l836_83647

def classroomA_fundraising_goal : ℕ :=
  let amount_from_two_families := 2 * 20
  let amount_from_eight_families := 8 * 10
  let amount_from_ten_families := 10 * 5
  let total_raised := amount_from_two_families + amount_from_eight_families + amount_from_ten_families
  let amount_needed := 30
  total_raised + amount_needed

theorem classroomA_goal_is_200 : classroomA_fundraising_goal = 200 := by
  sorry

end classroomA_goal_is_200_l836_83647


namespace highest_red_ball_probability_l836_83686

theorem highest_red_ball_probability :
  ∀ (total balls red yellow black : ℕ),
    total = 10 →
    red = 7 →
    yellow = 2 →
    black = 1 →
    (red / total) > (yellow / total) ∧ (red / total) > (black / total) :=
by
  intro total balls red yellow black
  intro h_total h_red h_yellow h_black
  sorry

end highest_red_ball_probability_l836_83686


namespace translate_down_by_2_l836_83635

theorem translate_down_by_2 (x y : ℝ) (h : y = -2 * x + 3) : y - 2 = -2 * x + 1 := 
by 
  sorry

end translate_down_by_2_l836_83635


namespace women_bathing_suits_count_l836_83636

theorem women_bathing_suits_count :
  ∀ (total_bathing_suits men_bathing_suits women_bathing_suits : ℕ),
    total_bathing_suits = 19766 →
    men_bathing_suits = 14797 →
    women_bathing_suits = total_bathing_suits - men_bathing_suits →
    women_bathing_suits = 4969 := by
sorry

end women_bathing_suits_count_l836_83636


namespace right_triangle_area_l836_83602

theorem right_triangle_area (a b c : ℕ) (h1 : a = 16) (h2 : b = 30) (h3 : c = 34) 
(h4 : a^2 + b^2 = c^2) : 
   1 / 2 * a * b = 240 :=
by 
  sorry

end right_triangle_area_l836_83602


namespace find_S13_l836_83624

-- Define the arithmetic sequence
variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- The sequence is arithmetic, i.e., there exists a common difference d
variable (d : ℤ)
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms is given by S_n
axiom sum_of_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 8 + a 12 = 12

-- We need to prove that S_{13} = 52
theorem find_S13 : S 13 = 52 :=
sorry

end find_S13_l836_83624


namespace find_a_and_b_l836_83603

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b

theorem find_a_and_b (a b : ℝ) (h_a : a < 0) (h_max : a + b = 3) (h_min : -a + b = -1) : a = -2 ∧ b = 1 :=
by
  sorry

end find_a_and_b_l836_83603


namespace arithmetic_seq_a6_l836_83634

theorem arithmetic_seq_a6 (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (0 < q) →
  a 1 = 1 →
  S 3 = 7/4 →
  S n = (1 - q^n) / (1 - q) →
  (∀ n, a n = 1 * q^(n - 1)) →
  a 6 = 1 / 32 :=
by
  sorry

end arithmetic_seq_a6_l836_83634


namespace measure_of_angle_C_l836_83697

-- Define the conditions using Lean 4 constructs
variable (a b c : ℝ)
variable (A B C : ℝ) -- Measures of angles in triangle ABC
variable (triangle_ABC : (a * a + b * b - c * c = a * b))

-- Statement of the proof problem
theorem measure_of_angle_C (h : a^2 + b^2 - c^2 = ab) (h2 : 0 < C ∧ C < π) : C = π / 3 :=
by
  -- Proof will go here but is omitted with sorry
  sorry

end measure_of_angle_C_l836_83697


namespace complement_union_correct_l836_83615

variable (U A B : Set ℕ)
variable (hU : U = {1, 2, 3, 4})
variable (hA : A = {2, 4})
variable (hB : B = {3, 4})

theorem complement_union_correct : ((U \ A) ∪ B) = {1, 3, 4} :=
by
  rw [hU, hA, hB]
  sorry

end complement_union_correct_l836_83615


namespace inequality_of_sums_l836_83654

theorem inequality_of_sums (a b c d : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_ineq : a > b ∧ b > c ∧ c > d) :
  (a + b + c + d)^2 > a^2 + 3 * b^2 + 5 * c^2 + 7 * d^2 :=
by
  sorry

end inequality_of_sums_l836_83654


namespace find_pairs_l836_83606

theorem find_pairs (x y : ℝ) (h1 : |x| + |y| = 1340) (h2 : x^3 + y^3 + 2010 * x * y = 670^3) :
  x + y = 670 ∧ x * y = -673350 :=
sorry

end find_pairs_l836_83606


namespace arithmetic_sequence_problem_l836_83657

-- Define what it means to be an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific terms in arithmetic sequence
def a (n : ℕ) : ℝ := sorry

-- Conditions given in the problem
axiom h1 : is_arithmetic_sequence a
axiom h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 120

-- The proof goal
theorem arithmetic_sequence_problem : a 9 - 1/3 * a 11 = 16 :=
by
  sorry

end arithmetic_sequence_problem_l836_83657


namespace not_a_factorization_l836_83627

open Nat

theorem not_a_factorization : ¬ (∃ (f g : ℝ → ℝ), (∀ (x : ℝ), x^2 + 6*x - 9 = f x * g x)) :=
by
  sorry

end not_a_factorization_l836_83627


namespace trajectory_of_M_lines_perpendicular_l836_83652

-- Define the given conditions
def parabola (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 = P.2

def midpoint_condition (P M : ℝ × ℝ) : Prop :=
  P.1 = 1/2 * M.1 ∧ P.2 = M.2

def trajectory_condition (M : ℝ × ℝ) : Prop :=
  M.1 ^ 2 = 4 * M.2

theorem trajectory_of_M (P M : ℝ × ℝ) (H1 : parabola P) (H2 : midpoint_condition P M) : 
  trajectory_condition M :=
sorry

-- Define the conditions for the second part
def line_through_F (A B : ℝ × ℝ) (F : ℝ × ℝ): Prop :=
  ∃ k : ℝ, A.2 = k * A.1 + F.2 ∧ B.2 = k * B.1 + F.2

def perpendicular_feet (A B A1 B1 : ℝ × ℝ) : Prop :=
  A1 = (A.1, -1) ∧ B1 = (B.1, -1)

def perpendicular_lines (A1 B1 F : ℝ × ℝ) : Prop :=
  let v1 := (-A1.1, F.2 - A1.2)
  let v2 := (-B1.1, F.2 - B1.2)
  v1.1 * v2.1 + v1.2 * v2.2 = 0

theorem lines_perpendicular (A B A1 B1 F : ℝ × ℝ) (H1 : trajectory_condition A) (H2 : trajectory_condition B) 
(H3 : line_through_F A B F) (H4 : perpendicular_feet A B A1 B1) :
  perpendicular_lines A1 B1 F :=
sorry

end trajectory_of_M_lines_perpendicular_l836_83652


namespace original_concentration_A_l836_83696

-- Definitions of initial conditions and parameters
def mass_A : ℝ := 2000 -- 2 kg in grams
def mass_B : ℝ := 3000 -- 3 kg in grams
def pour_out_A : ℝ := 0.15 -- 15% poured out from bottle A
def pour_out_B : ℝ := 0.30 -- 30% poured out from bottle B
def mixed_concentration1 : ℝ := 27.5 -- 27.5% concentration after first mix
def pour_out_restored : ℝ := 0.40 -- 40% poured out again

-- Using the calculated remaining mass and concentration to solve the proof
theorem original_concentration_A (x y : ℝ) 
  (h1 : 300 * x + 900 * y = 27.5 * (300 + 900)) 
  (h2 : (1700 * x + 300 * 27.5) * 0.4 / (2000 * 0.4) + (2100 * y + 900 * 27.5) * 0.4 / (3000 * 0.4) = 26) : 
  x = 20 :=
by 
  -- Skipping the proof. The proof should involve solving the system of equations.
  sorry

end original_concentration_A_l836_83696


namespace car_average_speed_l836_83678

theorem car_average_speed (distance time : ℕ) (h1 : distance = 715) (h2 : time = 11) : distance / time = 65 := by
  sorry

end car_average_speed_l836_83678


namespace set_union_complement_eq_l836_83651

def P : Set ℝ := {x | x^2 - 4 * x + 3 ≤ 0}
def Q : Set ℝ := {x | x^2 - 4 < 0}
def R_complement_Q : Set ℝ := {x | x ≤ -2 ∨ x ≥ 2}

theorem set_union_complement_eq :
  P ∪ R_complement_Q = {x | x ≤ -2} ∪ {x | x ≥ 1} :=
by {
  sorry
}

end set_union_complement_eq_l836_83651


namespace value_of_expression_l836_83685

theorem value_of_expression
  (a b : ℝ)
  (h₁ : a = 2 + Real.sqrt 3)
  (h₂ : b = 2 - Real.sqrt 3) :
  a^2 + 2 * a * b - b * (3 * a - b) = 13 :=
by
  sorry

end value_of_expression_l836_83685


namespace sharon_distance_to_mothers_house_l836_83676

noncomputable def total_distance (x : ℝ) :=
  x / 240

noncomputable def adjusted_speed (x : ℝ) :=
  x / 240 - 1 / 4

theorem sharon_distance_to_mothers_house (x : ℝ) (h1 : x / 240 = total_distance x) 
(h2 : adjusted_speed x = x / 240 - 1 / 4) 
(h3 : 120 + 120 * x / (x - 60) = 330) : 
x = 140 := 
by 
  sorry

end sharon_distance_to_mothers_house_l836_83676


namespace max_coins_Martha_can_take_l836_83623

/-- 
  Suppose a total of 2010 coins are distributed in 5 boxes with quantities 
  initially forming consecutive natural numbers. Martha can perform a 
  transformation where she takes one coin from a box with at least 4 coins and 
  distributes one coin to each of the other boxes. Prove that the maximum number 
  of coins that Martha can take away is 2004.
-/
theorem max_coins_Martha_can_take : 
  ∃ (a : ℕ), 2010 = a + (a+1) + (a+2) + (a+3) + (a+4) ∧ 
  ∀ (f : ℕ → ℕ) (h : (∃ b ≥ 4, f b = 400 + b)), 
  (∃ n : ℕ, f n = 4) → (∃ n : ℕ, f n = 3) → 
  (∃ n : ℕ, f n = 2) → (∃ n : ℕ, f n = 1) → 
  (∃ m : ℕ, f m = 2004) := 
by
  sorry

end max_coins_Martha_can_take_l836_83623


namespace find_number_of_people_l836_83610

def number_of_people (total_shoes : Nat) (shoes_per_person : Nat) : Nat :=
  total_shoes / shoes_per_person

theorem find_number_of_people :
  number_of_people 20 2 = 10 := 
by
  sorry

end find_number_of_people_l836_83610


namespace sum_of_first_9_terms_l836_83690

variable {a : ℕ → ℝ} -- the arithmetic sequence
variable {S : ℕ → ℝ} -- the sum function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

noncomputable def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

axiom arithmetic_sequence_condition (h : is_arithmetic_sequence a) : a 5 = 2

theorem sum_of_first_9_terms (h : is_arithmetic_sequence a) (h5: a 5 = 2) : sum_of_first_n_terms a 9 = 18 := by
  sorry

end sum_of_first_9_terms_l836_83690


namespace nebraska_more_plates_than_georgia_l836_83608

theorem nebraska_more_plates_than_georgia :
  (26 ^ 2 * 10 ^ 5) - (26 ^ 4 * 10 ^ 2) = 21902400 :=
by
  sorry

end nebraska_more_plates_than_georgia_l836_83608


namespace find_difference_l836_83673

variables (a b c : ℝ)

theorem find_difference (h1 : (a + b) / 2 = 45) (h2 : (b + c) / 2 = 50) : c - a = 10 := by
  sorry

end find_difference_l836_83673


namespace isosceles_triangle_perimeter_l836_83692

theorem isosceles_triangle_perimeter
  (a b : ℕ)
  (ha : a = 3)
  (hb : b = 7)
  (h_iso : ∃ (c : ℕ), (c = a ∨ c = b) ∧ ((a + b > c) ∧ (a + c > b) ∧ (b + c > a))) :
  a + b + b = 17 :=
by
  sorry

end isosceles_triangle_perimeter_l836_83692


namespace gcd_seq_consecutive_l836_83659

-- Define the sequence b_n
def seq (n : ℕ) : ℕ := n.factorial + 2 * n

-- Main theorem statement
theorem gcd_seq_consecutive (n : ℕ) : n ≥ 0 → Nat.gcd (seq n) (seq (n + 1)) = 2 :=
by
  intro h
  sorry

end gcd_seq_consecutive_l836_83659


namespace geometric_series_sum_l836_83604

theorem geometric_series_sum :
  let a := 1
  let r := 3
  let n := 9
  (1 * (3^n - 1) / (3 - 1)) = 9841 :=
by
  sorry

end geometric_series_sum_l836_83604


namespace ratio_five_to_one_l836_83611

theorem ratio_five_to_one (x : ℕ) (h : 5 / 1 = x / 9) : x = 45 :=
  sorry

end ratio_five_to_one_l836_83611


namespace proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l836_83655

variable (x y : ℤ)

def proposition_A := (x ≠ 1000 ∨ y ≠ 1002)
def proposition_B := (x + y ≠ 2002)

theorem proposition_A_necessary_for_B : proposition_B x y → proposition_A x y := by
  sorry

theorem proposition_A_not_sufficient_for_B : ¬ (proposition_A x y → proposition_B x y) := by
  sorry

end proposition_A_necessary_for_B_proposition_A_not_sufficient_for_B_l836_83655


namespace simplify_and_evaluate_l836_83632

theorem simplify_and_evaluate (x : ℚ) (h1 : x = -1/3) :
    (3 * x + 2) * (3 * x - 2) - 5 * x * (x - 1) - (2 * x - 1)^2 = 9 * x - 5 ∧
    (9 * x - 5) = -8 := 
by sorry

end simplify_and_evaluate_l836_83632


namespace average_payment_debt_l836_83665

theorem average_payment_debt :
  let total_payments := 65
  let first_20_payment := 410
  let increment := 65
  let remaining_payment := first_20_payment + increment
  let first_20_total := 20 * first_20_payment
  let remaining_total := 45 * remaining_payment
  let total_paid := first_20_total + remaining_total
  let average_payment := total_paid / total_payments
  average_payment = 455 := by sorry

end average_payment_debt_l836_83665


namespace complement_A_is_01_l836_83600

-- Define the universal set U as the set of all real numbers
def U : Set ℝ := Set.univ

-- Define the set A given the conditions
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}

-- State the theorem: complement of A is the interval [0, 1)
theorem complement_A_is_01 : Set.compl A = {x : ℝ | 0 ≤ x ∧ x < 1} :=
by
  sorry

end complement_A_is_01_l836_83600


namespace similar_right_triangle_hypotenuse_length_l836_83628

theorem similar_right_triangle_hypotenuse_length :
  ∀ (a b c d : ℝ), a = 15 → c = 39 → d = 45 → 
  (b^2 = c^2 - a^2) → 
  ∃ e : ℝ, e = (c * (d / b)) ∧ e = 48.75 :=
by
  intros a b c d ha hc hd hb
  sorry

end similar_right_triangle_hypotenuse_length_l836_83628


namespace minimize_transportation_cost_l836_83691

noncomputable def transportation_cost (x : ℝ) (distance : ℝ) (k : ℝ) (other_expense : ℝ) : ℝ :=
  k * (x * distance / x^2 + other_expense * distance / x)

theorem minimize_transportation_cost :
  ∀ (distance : ℝ) (max_speed : ℝ) (k : ℝ) (other_expense : ℝ) (x : ℝ),
  0 < x ∧ x ≤ max_speed ∧ max_speed = 50 ∧ distance = 300 ∧ k = 0.5 ∧ other_expense = 800 →
  transportation_cost x distance k other_expense = 150 * (x + 1600 / x) ∧
  (∀ y, (0 < y ∧ y ≤ max_speed) → transportation_cost y distance k other_expense ≥ 12000) ∧
  (transportation_cost 40 distance k other_expense = 12000)
  := 
  by intros distance max_speed k other_expense x H;
     sorry

end minimize_transportation_cost_l836_83691


namespace exists_a_star_b_eq_a_l836_83689

variable {S : Type*} [CommSemigroup S]

def exists_element_in_S (star : S → S → S) : Prop :=
  ∃ a : S, ∀ b : S, star a b = a

theorem exists_a_star_b_eq_a
  (star : S → S → S)
  (comm : ∀ a b : S, star a b = star b a)
  (assoc : ∀ a b c : S, star (star a b) c = star a (star b c))
  (exists_a : ∃ a : S, star a a = a) :
  exists_element_in_S star := sorry

end exists_a_star_b_eq_a_l836_83689


namespace cookies_eaten_l836_83670

theorem cookies_eaten (original remaining : ℕ) (h_original : original = 18) (h_remaining : remaining = 9) :
    original - remaining = 9 := by
  sorry

end cookies_eaten_l836_83670


namespace pencils_removed_l836_83633

theorem pencils_removed (initial_pencils removed_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 87) 
  (h2 : remaining_pencils = 83) 
  (h3 : removed_pencils = initial_pencils - remaining_pencils) : 
  removed_pencils = 4 :=
sorry

end pencils_removed_l836_83633


namespace johnson_family_seating_l836_83660

def johnson_family_boys : ℕ := 5
def johnson_family_girls : ℕ := 4
def total_chairs : ℕ := 9
def total_arrangements : ℕ := Nat.factorial total_chairs

noncomputable def seating_arrangements_with_at_least_3_boys : ℕ :=
  let three_boys_block_ways := 7 * (5 * 4 * 3) * Nat.factorial 6
  total_arrangements - three_boys_block_ways

theorem johnson_family_seating : seating_arrangements_with_at_least_3_boys = 60480 := by
  unfold seating_arrangements_with_at_least_3_boys
  sorry

end johnson_family_seating_l836_83660


namespace find_x_l836_83694

theorem find_x (x : ℕ) (h1 : (31 : ℕ) ≤ 100) (h2 : (58 : ℕ) ≤ 100) (h3 : (98 : ℕ) ≤ 100) (h4 : 0 < x) (h5 : x ≤ 100)
               (h_mean_mode : ((31 + 58 + 98 + x + x) / 5 : ℚ) = 1.5 * x) : x = 34 :=
by
  sorry

end find_x_l836_83694


namespace fraction_value_l836_83674

theorem fraction_value : (2 + 3 + 4 : ℚ) / (2 * 3 * 4) = 3 / 8 := 
by sorry

end fraction_value_l836_83674


namespace number_of_adults_l836_83626

theorem number_of_adults (A C S : ℕ) (h1 : C = A - 35) (h2 : S = 2 * C) (h3 : A + C + S = 127) : A = 58 :=
by
  sorry

end number_of_adults_l836_83626


namespace line_parallel_slope_l836_83669

theorem line_parallel_slope (m : ℝ) :
  (2 * 8 = m * m) →
  m = -4 :=
by
  intro h
  sorry

end line_parallel_slope_l836_83669


namespace lesser_number_is_14_l836_83684

theorem lesser_number_is_14 (x y : ℕ) (h₀ : x + y = 60) (h₁ : 4 * y - x = 10) : y = 14 :=
by 
  sorry

end lesser_number_is_14_l836_83684


namespace complex_transformation_l836_83668

open Complex

def dilation (z : ℂ) (center : ℂ) (scale : ℝ) : ℂ :=
  center + scale * (z - center)

def rotation90 (z : ℂ) : ℂ :=
  z * I

theorem complex_transformation (z : ℂ) (center : ℂ) (scale : ℝ) :
  center = -1 + 2 * I → scale = 2 → z = 3 + I →
  rotation90 (dilation z center scale) = 4 + 7 * I :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  dsimp [dilation]
  dsimp [rotation90]
  sorry

end complex_transformation_l836_83668


namespace intersection_of_S_and_T_l836_83698

noncomputable def S := {x : ℝ | x ≥ 2}
noncomputable def T := {x : ℝ | x ≤ 5}

theorem intersection_of_S_and_T : S ∩ T = {x : ℝ | 2 ≤ x ∧ x ≤ 5} :=
by
  sorry

end intersection_of_S_and_T_l836_83698


namespace pizza_volume_l836_83619

theorem pizza_volume (h : ℝ) (d : ℝ) (n : ℕ) 
  (h_cond : h = 1/2) 
  (d_cond : d = 16) 
  (n_cond : n = 8) 
  : (π * (d / 2) ^ 2 * h / n = 4 * π) :=
by
  sorry

end pizza_volume_l836_83619


namespace b_income_percentage_increase_l836_83662

theorem b_income_percentage_increase (A_m B_m C_m : ℕ) (annual_income_A : ℕ)
  (C_income : C_m = 15000)
  (annual_income_A_cond : annual_income_A = 504000)
  (ratio_cond : A_m / B_m = 5 / 2)
  (A_m_cond : A_m = annual_income_A / 12) :
  ((B_m - C_m) * 100 / C_m) = 12 :=
by
  sorry

end b_income_percentage_increase_l836_83662


namespace not_perfect_square_l836_83617

theorem not_perfect_square (a b : ℤ) (h : (a % 2 ≠ b % 2)) : ¬ ∃ k : ℤ, ((a + 3 * b) * (5 * a + 7 * b) = k^2) := 
by
  sorry

end not_perfect_square_l836_83617


namespace arccos_cos_eq_l836_83646

theorem arccos_cos_eq :
  Real.arccos (Real.cos 11) = 0.7168 := by
  sorry

end arccos_cos_eq_l836_83646


namespace students_interested_in_both_l836_83671

def numberOfStudentsInterestedInBoth (T S M N: ℕ) : ℕ := 
  S + M - (T - N)

theorem students_interested_in_both (T S M N: ℕ) (hT : T = 55) (hS : S = 43) (hM : M = 34) (hN : N = 4) : 
  numberOfStudentsInterestedInBoth T S M N = 26 := 
by 
  rw [hT, hS, hM, hN]
  sorry

end students_interested_in_both_l836_83671


namespace min_value_a_l836_83642

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ 1/2 → x^2 + a * x + 1 ≥ 0) → a ≥ -5/2 := 
sorry

end min_value_a_l836_83642


namespace geometric_sequence_sum_l836_83609

theorem geometric_sequence_sum (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n)
    (h1 : a 0 + a 1 = 324) (h2 : a 2 + a 3 = 36) : a 4 + a 5 = 4 :=
by
  sorry

end geometric_sequence_sum_l836_83609


namespace sqrt_of_9_l836_83618

theorem sqrt_of_9 : Real.sqrt 9 = 3 :=
by 
  sorry

end sqrt_of_9_l836_83618


namespace no_net_coin_change_l836_83629

noncomputable def probability_no_coin_change_each_round : ℚ :=
  (1 / 3) ^ 5

theorem no_net_coin_change :
  probability_no_coin_change_each_round = 1 / 243 := by
  sorry

end no_net_coin_change_l836_83629


namespace cost_of_cheaper_feed_l836_83681

theorem cost_of_cheaper_feed (C : ℝ) 
  (h1 : 35 * 0.36 = 12.6)
  (h2 : 18 * 0.53 = 9.54)
  (h3 : 17 * C + 9.54 = 12.6) :
  C = 0.18 := sorry

end cost_of_cheaper_feed_l836_83681


namespace number_of_sides_of_polygon_l836_83640

theorem number_of_sides_of_polygon (n : ℕ) (h : (n - 2) * 180 = 900) : n = 7 := 
sorry

end number_of_sides_of_polygon_l836_83640


namespace usable_parking_lot_percentage_l836_83675

theorem usable_parking_lot_percentage
  (length width : ℝ) (area_per_car : ℝ) (number_of_cars : ℝ)
  (h_len : length = 400)
  (h_wid : width = 500)
  (h_area_car : area_per_car = 10)
  (h_cars : number_of_cars = 16000) :
  ((number_of_cars * area_per_car) / (length * width) * 100) = 80 := 
by
  -- Proof omitted
  sorry

end usable_parking_lot_percentage_l836_83675


namespace collinear_probability_in_rectangular_array_l836_83695

noncomputable def prob_collinear (total_dots chosen_dots favorable_sets : ℕ) : ℚ :=
  favorable_sets / (Nat.choose total_dots chosen_dots)

theorem collinear_probability_in_rectangular_array :
  prob_collinear 20 4 2 = 2 / 4845 :=
by
  sorry

end collinear_probability_in_rectangular_array_l836_83695


namespace symmetric_point_l836_83649

theorem symmetric_point (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : |y| = 3) : 
  (2, -3) = (-x, -y) :=
sorry

end symmetric_point_l836_83649


namespace simple_interest_rate_l836_83664

theorem simple_interest_rate (P SI T : ℝ) (hP : P = 15000) (hSI : SI = 6000) (hT : T = 8) :
  ∃ R : ℝ, (SI = P * R * T / 100) ∧ R = 5 :=
by
  use 5
  field_simp [hP, hSI, hT]
  sorry

end simple_interest_rate_l836_83664


namespace area_below_line_l836_83620

-- Define the conditions provided in the problem.
def graph_eq (x y : ℝ) : Prop := x^2 - 14*x + 3*y + 70 = 21 + 11*y - y^2
def line_eq (x y : ℝ) : Prop := y = x - 3

-- State the final proof problem which is to find the area under the given conditions.
theorem area_below_line :
  ∃ area : ℝ, area = 8 * Real.pi ∧ 
  (∀ x y, graph_eq x y → y ≤ x - 3 → -area / 2 ≤ y ∧ y ≤ area / 2) := 
sorry

end area_below_line_l836_83620


namespace cats_sold_during_sale_l836_83687

-- Definitions based on conditions in a)
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def cats_left : ℕ := 8
def total_cats := siamese_cats + house_cats

-- Proof statement
theorem cats_sold_during_sale : total_cats - cats_left = 10 := by
  sorry

end cats_sold_during_sale_l836_83687


namespace student_correct_sums_l836_83630

-- Defining variables R and W along with the given conditions
variables (R W : ℕ)

-- Given conditions as Lean definitions
def condition1 := W = 5 * R
def condition2 := R + W = 180

-- Statement of the problem to prove R equals 30
theorem student_correct_sums :
  (W = 5 * R) → (R + W = 180) → R = 30 :=
by
  -- Import needed definitions and theorems from Mathlib
  sorry -- skipping the proof

end student_correct_sums_l836_83630


namespace solution_set_of_inequality_l836_83621

theorem solution_set_of_inequality (x : ℝ) : (2 * x + 3) * (4 - x) > 0 ↔ -3 / 2 < x ∧ x < 4 :=
by
  sorry

end solution_set_of_inequality_l836_83621


namespace expressions_equal_iff_l836_83666

theorem expressions_equal_iff (x y z : ℝ) : x + y + z = 0 ↔ x + yz = (x + y) * (x + z) :=
by
  sorry

end expressions_equal_iff_l836_83666


namespace valid_cube_placements_count_l836_83605

-- Define the initial cross configuration and the possible placements for the sixth square.
structure CrossConfiguration :=
  (squares : Finset (ℕ × ℕ)) -- Assume (ℕ × ℕ) represents the positions of the squares.

def valid_placements (config : CrossConfiguration) : Finset (ℕ × ℕ) :=
  -- Placeholder definition to represent the valid placements for the sixth square.
  sorry

theorem valid_cube_placements_count (config : CrossConfiguration) :
  (valid_placements config).card = 4 := 
by 
  sorry

end valid_cube_placements_count_l836_83605


namespace Cindy_correct_answer_l836_83658

theorem Cindy_correct_answer (x : ℕ) (h : (x - 14) / 4 = 28) : ((x - 5) / 7) * 4 = 69 := by
  sorry

end Cindy_correct_answer_l836_83658


namespace kerosene_sale_difference_l836_83601

noncomputable def rice_price : ℝ := 0.33
noncomputable def price_of_dozen_eggs := rice_price
noncomputable def price_of_one_egg := rice_price / 12
noncomputable def price_of_half_liter_kerosene := 4 * price_of_one_egg
noncomputable def price_of_one_liter_kerosene := 2 * price_of_half_liter_kerosene
noncomputable def kerosene_discounted := price_of_one_liter_kerosene * 0.95
noncomputable def kerosene_diff_cents := (price_of_one_liter_kerosene - kerosene_discounted) * 100

theorem kerosene_sale_difference :
  kerosene_diff_cents = 1.1 := by sorry

end kerosene_sale_difference_l836_83601


namespace exponent_multiplication_l836_83625

theorem exponent_multiplication (m n : ℕ) (h1 : m > 0) (h2 : n > 0) (a b : ℤ) (h3 : 3^m = a) (h4 : 3^n = b) : 3^(m + n) = a * b :=
by
  sorry

end exponent_multiplication_l836_83625


namespace contrapositive_equivalence_l836_83650

-- Define the original proposition and its contrapositive
def original_proposition (q p : Prop) := q → p
def contrapositive (q p : Prop) := ¬q → ¬p

-- The theorem to prove
theorem contrapositive_equivalence (q p : Prop) :
  (original_proposition q p) ↔ (contrapositive q p) :=
by
  sorry

end contrapositive_equivalence_l836_83650


namespace find_m_value_l836_83641

theorem find_m_value (f : ℝ → ℝ) (h1 : ∀ x, f ((x / 2) - 1) = 2 * x + 3) (h2 : f m = 6) : m = -(1 / 4) :=
sorry

end find_m_value_l836_83641


namespace soda_price_l836_83607

-- We define the conditions as given in the problem
def regular_price (P : ℝ) : Prop :=
  -- Regular price per can is P
  ∃ P, 
  -- 25 percent discount on regular price when purchased in 24-can cases
  (∀ (discounted_price_per_can : ℝ), discounted_price_per_can = 0.75 * P) ∧
  -- Price of 70 cans at the discounted price is $28.875
  (70 * 0.75 * P = 28.875)

-- We state the theorem to prove that the regular price per can is $0.55
theorem soda_price (P : ℝ) (h : regular_price P) : P = 0.55 :=
by
  sorry

end soda_price_l836_83607


namespace regular_rate_survey_l836_83677

theorem regular_rate_survey (R : ℝ) 
  (total_surveys : ℕ := 50)
  (rate_increase : ℝ := 0.30)
  (cellphone_surveys : ℕ := 35)
  (total_earnings : ℝ := 605) :
  35 * (1.30 * R) + 15 * R = 605 → R = 10 :=
by
  sorry

end regular_rate_survey_l836_83677


namespace max_value_Sn_l836_83613

theorem max_value_Sn (a₁ : ℚ) (r : ℚ) (S : ℕ → ℚ)
  (h₀ : a₁ = 3 / 2)
  (h₁ : r = -1 / 2)
  (h₂ : ∀ n, S n = a₁ * (1 - r ^ n) / (1 - r))
  : ∀ n, S n ≤ 3 / 2 ∧ (∃ m, S m = 3 / 2) :=
by sorry

end max_value_Sn_l836_83613


namespace largest_by_changing_first_digit_l836_83667

def value_with_digit_changed (d : Nat) : Float :=
  match d with
  | 1 => 0.86123
  | 2 => 0.78123
  | 3 => 0.76823
  | 4 => 0.76183
  | 5 => 0.76128
  | _ => 0.76123 -- default case

theorem largest_by_changing_first_digit :
  ∀ d : Nat, d ∈ [1, 2, 3, 4, 5] → value_with_digit_changed 1 ≥ value_with_digit_changed d :=
by
  intro d hd_list
  sorry

end largest_by_changing_first_digit_l836_83667
