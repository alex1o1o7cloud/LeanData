import Mathlib

namespace tank_overflow_time_l1385_138552

noncomputable def pipeARate : ℚ := 1 / 32
noncomputable def pipeBRate : ℚ := 3 * pipeARate
noncomputable def combinedRate (rateA rateB : ℚ) : ℚ := rateA + rateB

theorem tank_overflow_time : 
  combinedRate pipeARate pipeBRate = 1 / 8 ∧ (1 / combinedRate pipeARate pipeBRate = 8) :=
by
  sorry

end tank_overflow_time_l1385_138552


namespace miles_in_one_hour_eq_8_l1385_138575

-- Parameters as given in the conditions
variables (x : ℕ) (h1 : ∀ t : ℕ, t >= 6 → t % 6 = 0 ∨ t % 6 < 6)
variables (miles_in_one_hour : ℕ)
-- Given condition: The car drives 88 miles in 13 hours.
variable (miles_in_13_hours : miles_in_one_hour * 11 = 88)

-- Statement to prove: The car can drive 8 miles in one hour.
theorem miles_in_one_hour_eq_8 : miles_in_one_hour = 8 :=
by {
  -- Proof goes here
  sorry
}

end miles_in_one_hour_eq_8_l1385_138575


namespace log2_6_gt_2_sqrt_5_l1385_138550

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l1385_138550


namespace jan_more_miles_than_ian_l1385_138588

noncomputable def distance_diff (d t s : ℝ) : ℝ :=
  let han_distance := (s + 10) * (t + 2)
  let jan_distance := (s + 15) * (t + 3)
  jan_distance - (d + 100)

theorem jan_more_miles_than_ian {d t s : ℝ} (H : d = s * t) (H_han : d + 100 = (s + 10) * (t + 2)) : distance_diff d t s = 165 :=
by {
  sorry
}

end jan_more_miles_than_ian_l1385_138588


namespace difference_between_max_and_min_34_l1385_138595

theorem difference_between_max_and_min_34 
  (A B C D E: ℕ) 
  (h_avg: (A + B + C + D + E) / 5 = 50) 
  (h_max: E ≤ 58) 
  (h_distinct: A < B ∧ B < C ∧ C < D ∧ D < E) 
: E - A = 34 := 
sorry

end difference_between_max_and_min_34_l1385_138595


namespace part_a_exists_part_b_not_exists_l1385_138547

theorem part_a_exists :
  ∃ (a b : ℤ), (∀ x : ℝ, x^2 + a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + a*x + b = 0) :=
sorry

theorem part_b_not_exists :
  ¬ ∃ (a b : ℤ), (∀ x : ℝ, x^2 + 2*a*x + b ≠ 0) ∧ (∃ x : ℝ, ⌊x^2⌋ + 2*a*x + b = 0) :=
sorry

end part_a_exists_part_b_not_exists_l1385_138547


namespace triangle_area_from_squares_l1385_138596

noncomputable def area_of_triangle (S1 S2 : ℝ) : ℝ :=
  let side1 := Real.sqrt S1
  let side2 := Real.sqrt S2
  0.5 * side1 * side2

theorem triangle_area_from_squares
  (A1 A2 : ℝ)
  (h1 : A1 = 196)
  (h2 : A2 = 100) :
  area_of_triangle A1 A2 = 70 :=
by
  rw [h1, h2]
  unfold area_of_triangle
  rw [Real.sqrt_eq_rpow, Real.sqrt_eq_rpow]
  norm_num
  sorry

end triangle_area_from_squares_l1385_138596


namespace suff_but_not_nec_l1385_138534

theorem suff_but_not_nec (a b : ℝ) (h : a > b ∧ b > 0) : a^2 > b^2 :=
by {
  sorry
}

end suff_but_not_nec_l1385_138534


namespace conference_handshakes_l1385_138515

theorem conference_handshakes (n_leaders n_participants : ℕ) (n_total : ℕ) 
  (h_total : n_total = n_leaders + n_participants) 
  (h_leaders : n_leaders = 5) 
  (h_participants : n_participants = 25) 
  (h_total_people : n_total = 30) : 
  (n_leaders * (n_total - 1) - (n_leaders * (n_leaders - 1) / 2)) = 135 := 
by 
  sorry

end conference_handshakes_l1385_138515


namespace teacher_drank_milk_false_l1385_138531

-- Define the condition that the volume of milk a teacher can reasonably drink in a day is more appropriately measured in milliliters rather than liters.
def reasonable_volume_units := "milliliters"

-- Define the statement to be judged
def teacher_milk_intake := 250

-- Define the unit of the statement
def unit_of_statement := "liters"

-- The proof goal is to conclude that the statement "The teacher drank 250 liters of milk today" is false, given the condition on volume units.
theorem teacher_drank_milk_false (vol : ℕ) (unit : String) (reasonable_units : String) :
  vol = 250 ∧ unit = "liters" ∧ reasonable_units = "milliliters" → false :=
by
  sorry

end teacher_drank_milk_false_l1385_138531


namespace profit_percentage_is_correct_l1385_138582

-- Definitions for the given conditions
def SP : ℝ := 850
def Profit : ℝ := 255
def CP : ℝ := SP - Profit

-- The target proof statement
theorem profit_percentage_is_correct : 
  (Profit / CP) * 100 = 42.86 := by
  sorry

end profit_percentage_is_correct_l1385_138582


namespace Carla_total_counts_l1385_138526

def Monday_counts := (60 * 2) + (120 * 2) + (10 * 2)
def Tuesday_counts := (60 * 3) + (120 * 2) + (10 * 1)
def Wednesday_counts := (80 * 4) + (24 * 5)
def Thursday_counts := (60 * 1) + (80 * 2) + (120 * 3) + (10 * 4) + (24 * 5)
def Friday_counts := (60 * 1) + (120 * 2) + (80 * 2) + (10 * 3) + (24 * 3)

def total_counts := Monday_counts + Tuesday_counts + Wednesday_counts + Thursday_counts + Friday_counts

theorem Carla_total_counts : total_counts = 2552 :=
by 
  sorry

end Carla_total_counts_l1385_138526


namespace malcolm_initial_white_lights_l1385_138592

theorem malcolm_initial_white_lights :
  ∀ (red blue green remaining total_initial : ℕ),
    red = 12 →
    blue = 3 * red →
    green = 6 →
    remaining = 5 →
    total_initial = red + blue + green + remaining →
    total_initial = 59 :=
by
  intros red blue green remaining total_initial h1 h2 h3 h4 h5
  -- Add details if necessary for illustration
  -- sorry typically as per instructions
  sorry

end malcolm_initial_white_lights_l1385_138592


namespace area_of_triangle_ABC_equation_of_circumcircle_l1385_138590

-- Define points A, B, and C
structure Point where
  x : ℝ
  y : ℝ

def A : Point := { x := 1, y := 2 }
def B : Point := { x := 1, y := 3 }
def C : Point := { x := 3, y := 6 }

-- Theorem to prove the area of triangle ABC
theorem area_of_triangle_ABC : 
  let base := |B.y - A.y|
  let height := |C.x - A.x|
  (1/2) * base * height = 1 := sorry

-- Theorem to prove the equation of the circumcircle of triangle ABC
theorem equation_of_circumcircle : 
  let D := -10
  let E := -5
  let F := 15
  ∀ (x y : ℝ), (x - 5)^2 + (y - 5/2)^2 = 65/4 ↔ 
                x^2 + y^2 + D * x + E * y + F = 0 := sorry

end area_of_triangle_ABC_equation_of_circumcircle_l1385_138590


namespace mul_fraction_eq_l1385_138508

theorem mul_fraction_eq : 7 * (1 / 11) * 33 = 21 :=
by
  sorry

end mul_fraction_eq_l1385_138508


namespace alpha_value_l1385_138566

open Complex

theorem alpha_value (α β : ℂ) (h1 : β = 2 + 3 * I) (h2 : (α + β).im = 0) (h3 : (I * (2 * α - β)).im = 0) : α = 6 + 4 * I :=
by
  sorry

end alpha_value_l1385_138566


namespace female_democrats_count_l1385_138521

theorem female_democrats_count 
  (F M D : ℕ)
  (total_participants : F + M = 660)
  (total_democrats : F / 2 + M / 4 = 660 / 3)
  (female_democrats : D = F / 2) : 
  D = 110 := 
by
  sorry

end female_democrats_count_l1385_138521


namespace complement_of_45_is_45_l1385_138594

def angle_complement (A : Real) : Real :=
  90 - A

theorem complement_of_45_is_45:
  angle_complement 45 = 45 :=
by
  sorry

end complement_of_45_is_45_l1385_138594


namespace find_function_l1385_138578

theorem find_function (f : ℝ → ℝ)
  (h₁ : ∀ x : ℝ, x * (f (x + 1) - f x) = f x)
  (h₂ : ∀ x y : ℝ, |f x - f y| ≤ |x - y|) :
  ∃ k : ℝ, (∀ x : ℝ, f x = k * x) ∧ |k| ≤ 1 :=
sorry

end find_function_l1385_138578


namespace height_of_brick_l1385_138503

-- Definitions of given conditions
def length_brick : ℝ := 125
def width_brick : ℝ := 11.25
def length_wall : ℝ := 800
def height_wall : ℝ := 600
def width_wall : ℝ := 22.5
def number_bricks : ℝ := 1280

-- Prove that the height of each brick is 6.01 cm
theorem height_of_brick :
  ∃ H : ℝ,
    H = 6.01 ∧
    (number_bricks * (length_brick * width_brick * H) = length_wall * height_wall * width_wall) :=
by
  sorry

end height_of_brick_l1385_138503


namespace minimal_dominoes_needed_l1385_138523

-- Variables representing the number of dominoes and tetraminoes
variables (d t : ℕ)

-- Definitions related to the problem
def area_rectangle : ℕ := 2008 * 2010 -- Total area of the rectangle
def area_domino : ℕ := 1 * 2 -- Area of a single domino
def area_tetramino : ℕ := 2 * 3 - 2 -- Area of a single tetramino
def total_area_covered : ℕ := 2 * d + 4 * t -- Total area covered by dominoes and tetraminoes

-- The theorem we want to prove
theorem minimal_dominoes_needed :
  total_area_covered d t = area_rectangle → d = 0 :=
sorry

end minimal_dominoes_needed_l1385_138523


namespace answered_both_questions_correctly_l1385_138533

theorem answered_both_questions_correctly (P_A P_B P_A_prime_inter_B_prime : ℝ)
  (h1 : P_A = 70 / 100) (h2 : P_B = 55 / 100) (h3 : P_A_prime_inter_B_prime = 20 / 100) :
  P_A + P_B - (1 - P_A_prime_inter_B_prime) = 45 / 100 := 
by
  sorry

end answered_both_questions_correctly_l1385_138533


namespace no_such_triples_l1385_138518

noncomputable def no_triple_satisfy (a b c : ℤ) : Prop :=
  ∀ (x1 x2 x3 : ℤ), 
    x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    Int.gcd x1 x2 = 1 ∧ Int.gcd x2 x3 = 1 ∧ Int.gcd x1 x3 = 1 ∧
    (x1^3 - a^2 * x1^2 + b^2 * x1 - a * b + 3 * c = 0) ∧ 
    (x2^3 - a^2 * x2^2 + b^2 * x2 - a * b + 3 * c = 0) ∧ 
    (x3^3 - a^2 * x3^2 + b^2 * x3 - a * b + 3 * c = 0) →
    False

theorem no_such_triples : ∀ (a b c : ℤ), no_triple_satisfy a b c :=
by
  intros
  sorry

end no_such_triples_l1385_138518


namespace sum_first_five_terms_l1385_138554

variable {a : ℕ → ℝ}

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 1, ∀ n, a (n + 1) = a n * q

theorem sum_first_five_terms (h₁ : is_geometric_sequence a) 
  (h₂ : a 1 > 0) 
  (h₃ : a 1 * a 7 = 64) 
  (h₄ : a 3 + a 5 = 20) : 
  a 1 * (1 - (2 : ℝ) ^ 5) / (1 - 2) = 31 := 
by
  sorry

end sum_first_five_terms_l1385_138554


namespace arithmetic_sequence_nth_term_l1385_138591

noncomputable def nth_arithmetic_term (a : ℤ) (n : ℕ) : ℤ :=
  let a1 := a - 1
  let a2 := a + 1
  let a3 := 2 * a + 3
  if 2 * (a + 1) = (a - 1) + (2 * a + 3) then
    -1 + (n - 1) * 2
  else
    sorry

theorem arithmetic_sequence_nth_term (a : ℤ) (n : ℕ) (h : 2 * (a + 1) = (a - 1) + (2 * a + 3)) :
  nth_arithmetic_term a n = 2 * (n : ℤ) - 3 :=
by
  sorry

end arithmetic_sequence_nth_term_l1385_138591


namespace g_of_neg5_eq_651_over_16_l1385_138559

def f (x : ℝ) : ℝ := 4 * x + 6

def g (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 7

theorem g_of_neg5_eq_651_over_16 : g (-5) = 651 / 16 := by
  sorry

end g_of_neg5_eq_651_over_16_l1385_138559


namespace neg_p_iff_neg_q_l1385_138572

theorem neg_p_iff_neg_q (a : ℝ) : (¬ (a < 0)) ↔ (¬ (a^2 > a)) :=
by 
    sorry

end neg_p_iff_neg_q_l1385_138572


namespace lines_intersect_at_single_point_l1385_138560

theorem lines_intersect_at_single_point (m : ℚ)
    (h1 : ∃ x y : ℚ, y = 4 * x - 8 ∧ y = -3 * x + 9)
    (h2 : ∀ x y : ℚ, (y = 4 * x - 8 ∧ y = -3 * x + 9) → (y = 2 * x + m)) :
    m = -22/7 := by
  sorry

end lines_intersect_at_single_point_l1385_138560


namespace rectangle_square_division_l1385_138502

theorem rectangle_square_division (a b : ℝ) (n : ℕ) (h1 : (∃ (s1 : ℝ), s1^2 * (n : ℝ) = a * b))
                                            (h2 : (∃ (s2 : ℝ), s2^2 * (n + 76 : ℝ) = a * b)) :
    n = 324 := 
by
  sorry

end rectangle_square_division_l1385_138502


namespace vertex_of_parabola_l1385_138581

theorem vertex_of_parabola :
  ∃ (a b c : ℝ), 
      (4 * a - 2 * b + c = 9) ∧ 
      (16 * a + 4 * b + c = 9) ∧ 
      (49 * a + 7 * b + c = 16) ∧ 
      (-b / (2 * a) = 1) :=
by {
  -- we need to provide the proof here; sorry is a placeholder
  sorry
}

end vertex_of_parabola_l1385_138581


namespace largest_divisor_l1385_138577

theorem largest_divisor (n : ℕ) (hn : Even n) : ∃ k, ∀ n, Even n → k ∣ (n * (n+2) * (n+4) * (n+6) * (n+8)) ∧ (∀ m, (∀ n, Even n → m ∣ (n * (n+2) * (n+4) * (n+6) * (n+8))) → m ≤ k) :=
by
  use 96
  { sorry }

end largest_divisor_l1385_138577


namespace evaluate_expression_l1385_138561

theorem evaluate_expression : 
  (3^4 + 3^4 + 3^4) / (3^(-4) + 3^(-4)) = 9841.5 :=
by
  sorry

end evaluate_expression_l1385_138561


namespace find_common_ratio_l1385_138524

theorem find_common_ratio (q : ℝ) (a : ℕ → ℝ) 
  (h₀ : ∀ n, a (n + 1) = q * a n)
  (h₁ : a 0 = 4)
  (h₂ : q ≠ 1)
  (h₃ : 2 * a 4 = 4 * a 0 - 2 * a 2) :
  q = -1 := 
sorry

end find_common_ratio_l1385_138524


namespace max_area_BPC_l1385_138558

noncomputable def triangle_area_max (AB BC CA : ℝ) (D : ℝ) : ℝ :=
  if h₁ : AB = 13 ∧ BC = 15 ∧ CA = 14 then
    112.5 - 56.25 * Real.sqrt 3
  else 0

theorem max_area_BPC : triangle_area_max 13 15 14 D = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry

end max_area_BPC_l1385_138558


namespace cost_per_tissue_l1385_138543

-- Annalise conditions
def boxes : ℕ := 10
def packs_per_box : ℕ := 20
def tissues_per_pack : ℕ := 100
def total_spent : ℝ := 1000

-- Definition for total packs and total tissues
def total_packs : ℕ := boxes * packs_per_box
def total_tissues : ℕ := total_packs * tissues_per_pack

-- The math problem: Prove the cost per tissue
theorem cost_per_tissue : (total_spent / total_tissues) = 0.05 := by
  sorry

end cost_per_tissue_l1385_138543


namespace justin_reads_pages_l1385_138520

theorem justin_reads_pages (x : ℕ) 
  (h1 : 130 = x + 6 * (2 * x)) : x = 10 := 
sorry

end justin_reads_pages_l1385_138520


namespace al_bill_cal_probability_l1385_138525

-- Let's define the conditions and problem setup
def al_bill_cal_prob : ℚ :=
  let total_ways := 12 * 11 * 10
  let valid_ways := 12 -- This represent the summed valid cases as calculated
  valid_ways / total_ways

theorem al_bill_cal_probability :
  al_bill_cal_prob = 1 / 110 :=
  by
  -- Placeholder for calculation and proof
  sorry

end al_bill_cal_probability_l1385_138525


namespace forty_percent_of_number_l1385_138504

variables {N : ℝ}

theorem forty_percent_of_number (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 10) : 0.40 * N = 120 :=
by sorry

end forty_percent_of_number_l1385_138504


namespace simplify_expr1_simplify_expr2_simplify_expr3_l1385_138557

-- 1. Proving (1)(2x^{2})^{3}-x^{2}·x^{4} = 7x^{6}
theorem simplify_expr1 (x : ℝ) : (1 : ℝ) * (2 * x^2)^3 - x^2 * x^4 = 7 * x^6 := 
by 
  sorry

-- 2. Proving (a+b)^{2}-b(2a+b) = a^{2}
theorem simplify_expr2 (a b : ℝ) : (a + b)^2 - b * (2 * a + b) = a^2 := 
by 
  sorry

-- 3. Proving (x+1)(x-1)-x^{2} = -1
theorem simplify_expr3 (x : ℝ) : (x + 1) * (x - 1) - x^2 = -1 :=
by 
  sorry

end simplify_expr1_simplify_expr2_simplify_expr3_l1385_138557


namespace initial_balance_l1385_138532

theorem initial_balance (B : ℝ) (payment : ℝ) (new_balance : ℝ)
  (h1 : payment = 50) (h2 : new_balance = 120) (h3 : B - payment = new_balance) :
  B = 170 :=
by
  rw [h1, h2] at h3
  linarith

end initial_balance_l1385_138532


namespace solve_ineq_l1385_138570

noncomputable def f (x : ℝ) : ℝ := (2 / (x + 2)) + (4 / (x + 8)) - (7 / 3)

theorem solve_ineq (x : ℝ) : 
  (f x ≤ 0) ↔ (x ∈ Set.Ioc (-8) 4) := 
sorry

end solve_ineq_l1385_138570


namespace unique_triple_solution_zero_l1385_138548

theorem unique_triple_solution_zero (m n k : ℝ) :
  (∃ x : ℝ, m * x ^ 2 + n = 0) ∧
  (∃ x : ℝ, n * x ^ 2 + k = 0) ∧
  (∃ x : ℝ, k * x ^ 2 + m = 0) ↔
  (m = 0 ∧ n = 0 ∧ k = 0) := 
sorry

end unique_triple_solution_zero_l1385_138548


namespace ordered_pair_sol_l1385_138530

noncomputable def A (d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, 3], ![5, d]]

noncomputable def is_inverse_scalar_mul (d k : ℝ) : Prop :=
  (A d)⁻¹ = k • (A d)

theorem ordered_pair_sol (d k : ℝ) :
  is_inverse_scalar_mul d k → (d = -2 ∧ k = 1 / 19) :=
by
  intros h
  sorry

end ordered_pair_sol_l1385_138530


namespace joe_marshmallow_ratio_l1385_138537

theorem joe_marshmallow_ratio (J : ℕ) (h1 : 21 / 3 = 7) (h2 : 1 / 2 * J = 49 - 7) : J / 21 = 4 :=
by
  sorry

end joe_marshmallow_ratio_l1385_138537


namespace snowfall_difference_l1385_138549

-- Defining all conditions given in the problem
def BaldMountain_snowfall_meters : ℝ := 1.5
def BillyMountain_snowfall_meters : ℝ := 3.5
def MountPilot_snowfall_centimeters : ℝ := 126
def RockstonePeak_snowfall_millimeters : ℝ := 5250
def SunsetRidge_snowfall_meters : ℝ := 2.25

-- Conversion constants
def meters_to_centimeters : ℝ := 100
def millimeters_to_centimeters : ℝ := 0.1

-- Converting snowfall amounts to centimeters
def BaldMountain_snowfall_centimeters : ℝ := BaldMountain_snowfall_meters * meters_to_centimeters
def BillyMountain_snowfall_centimeters : ℝ := BillyMountain_snowfall_meters * meters_to_centimeters
def RockstonePeak_snowfall_centimeters : ℝ := RockstonePeak_snowfall_millimeters * millimeters_to_centimeters
def SunsetRidge_snowfall_centimeters : ℝ := SunsetRidge_snowfall_meters * meters_to_centimeters

-- Defining total combined snowfall
def combined_snowfall_centimeters : ℝ :=
  BillyMountain_snowfall_centimeters + MountPilot_snowfall_centimeters + RockstonePeak_snowfall_centimeters + SunsetRidge_snowfall_centimeters

-- Stating the proof statement
theorem snowfall_difference :
  combined_snowfall_centimeters - BaldMountain_snowfall_centimeters = 1076 := 
  by
    sorry

end snowfall_difference_l1385_138549


namespace sample_size_l1385_138579

theorem sample_size (w_under30 : ℕ) (w_30to40 : ℕ) (w_40plus : ℕ) (sample_40plus : ℕ) (total_sample : ℕ) :
  w_under30 = 2400 →
  w_30to40 = 3600 →
  w_40plus = 6000 →
  sample_40plus = 60 →
  total_sample = 120 :=
by
  intros
  sorry

end sample_size_l1385_138579


namespace find_c_l1385_138541

theorem find_c (a b c : ℝ) (k₁ k₂ : ℝ) 
  (h₁ : a * b = k₁) 
  (h₂ : b * c = k₂) 
  (h₃ : 40 * 5 = k₁) 
  (h₄ : 7 * 10 = k₂) 
  (h₅ : a = 16) : 
  c = 5.6 :=
  sorry

end find_c_l1385_138541


namespace fraction_savings_spent_on_furniture_l1385_138500

theorem fraction_savings_spent_on_furniture (savings : ℝ) (tv_cost : ℝ) (F : ℝ) 
  (h1 : savings = 840) (h2 : tv_cost = 210) 
  (h3 : F * savings + tv_cost = savings) : F = 3 / 4 :=
sorry

end fraction_savings_spent_on_furniture_l1385_138500


namespace find_m_l1385_138584

theorem find_m (a : ℕ → ℤ) (S : ℕ → ℤ) (m : ℕ) 
  (hS : ∀ n, S n = n^2 - 6 * n) :
  (forall m, (5 < a m ∧ a m < 8) → m = 7)
:= 
by
  sorry

end find_m_l1385_138584


namespace smoothie_one_serving_ingredients_in_cups_containers_needed_l1385_138589

theorem smoothie_one_serving_ingredients_in_cups :
  (0.2 + 0.1 + 0.2 + 1 * 0.125 + 2 * 0.0625 + 0.5).round = 1.25.round := sorry

theorem containers_needed :
  (5 * 1.25 / 1.5).ceil = 5 := sorry

end smoothie_one_serving_ingredients_in_cups_containers_needed_l1385_138589


namespace Adam_ate_more_than_Bill_l1385_138585

-- Definitions
def Sierra_ate : ℕ := 12
def Bill_ate : ℕ := Sierra_ate / 2
def total_pies_eaten : ℕ := 27
def Sierra_and_Bill_ate : ℕ := Sierra_ate + Bill_ate
def Adam_ate : ℕ := total_pies_eaten - Sierra_and_Bill_ate
def Adam_more_than_Bill : ℕ := Adam_ate - Bill_ate

-- Statement to prove
theorem Adam_ate_more_than_Bill :
  Adam_more_than_Bill = 3 :=
by
  sorry

end Adam_ate_more_than_Bill_l1385_138585


namespace selling_price_for_given_profit_selling_price_to_maximize_profit_l1385_138512

-- Define the parameters
def cost_price : ℝ := 40
def initial_selling_price : ℝ := 50
def initial_monthly_sales : ℝ := 500
def sales_decrement_per_unit_increase : ℝ := 10

-- Define the function for monthly sales based on price increment
def monthly_sales (x : ℝ) : ℝ := initial_monthly_sales - sales_decrement_per_unit_increase * x

-- Define the function for selling price based on price increment
def selling_price (x : ℝ) : ℝ := initial_selling_price + x

-- Define the function for monthly profit
def monthly_profit (x : ℝ) : ℝ :=
  let total_revenue := monthly_sales x * selling_price x 
  let total_cost := monthly_sales x * cost_price
  total_revenue - total_cost

-- Problem 1: Prove the selling price when monthly profit is 8750 yuan
theorem selling_price_for_given_profit : 
  ∃ x : ℝ, monthly_profit x = 8750 ∧ (selling_price x = 75 ∨ selling_price x = 65) :=
sorry

-- Problem 2: Prove the selling price that maximizes the monthly profit
theorem selling_price_to_maximize_profit : 
  ∀ x : ℝ, monthly_profit x ≤ monthly_profit 20 ∧ selling_price 20 = 70 :=
sorry

end selling_price_for_given_profit_selling_price_to_maximize_profit_l1385_138512


namespace even_product_divisible_by_1947_l1385_138514

theorem even_product_divisible_by_1947 (n : ℕ) (h_even : n % 2 = 0) :
  (∃ k: ℕ, 2 ≤ k ∧ k ≤ n / 2 ∧ 1947 ∣ (2 ^ k * k!)) → n ≥ 3894 :=
by
  sorry

end even_product_divisible_by_1947_l1385_138514


namespace solve_for_y_l1385_138513

theorem solve_for_y (y : ℝ) (h : y + 49 / y = 14) : y = 7 :=
sorry

end solve_for_y_l1385_138513


namespace moles_of_CH4_l1385_138568

theorem moles_of_CH4 (moles_Be2C moles_H2O : ℕ) (balanced_equation : 1 * Be2C + 4 * H2O = 2 * CH4 + 2 * BeOH2) 
  (h_Be2C : moles_Be2C = 3) (h_H2O : moles_H2O = 12) : 
  6 = 2 * moles_Be2C :=
by
  sorry

end moles_of_CH4_l1385_138568


namespace cubic_as_diff_of_squares_l1385_138546

theorem cubic_as_diff_of_squares (n : ℕ) (h : n > 1) :
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := 
sorry

end cubic_as_diff_of_squares_l1385_138546


namespace log_ratios_l1385_138586

noncomputable def ratio_eq : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem log_ratios
  {a b : ℝ}
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : Real.log a / Real.log 8 = Real.log b / Real.log 18)
  (h4 : Real.log b / Real.log 18 = Real.log (a + b) / Real.log 32) :
  b / a = ratio_eq :=
sorry

end log_ratios_l1385_138586


namespace number_of_boxes_l1385_138522

-- Define the given conditions
def total_chocolates : ℕ := 442
def chocolates_per_box : ℕ := 26

-- Prove the number of small boxes in the large box
theorem number_of_boxes : (total_chocolates / chocolates_per_box) = 17 := by
  sorry

end number_of_boxes_l1385_138522


namespace harmonic_progression_l1385_138564

theorem harmonic_progression (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
(h_harm : 1 / (a : ℝ) + 1 / (c : ℝ) = 2 / (b : ℝ))
(h_div : c % b = 0)
(h_inc : a < b ∧ b < c) :
  a = 20 → 
  (b, c) = (30, 60) ∨ (b, c) = (35, 140) ∨ (b, c) = (36, 180) ∨ (b, c) = (38, 380) ∨ (b, c) = (39, 780) :=
by sorry

end harmonic_progression_l1385_138564


namespace multiply_103_97_l1385_138571

theorem multiply_103_97 : 103 * 97 = 9991 := 
by
  sorry

end multiply_103_97_l1385_138571


namespace actual_distance_travelled_l1385_138562

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l1385_138562


namespace minimum_distance_between_extrema_is_2_sqrt_pi_l1385_138511

noncomputable def minimum_distance_adjacent_extrema (a : ℝ) (h : a > 0) : ℝ := 2 * Real.sqrt Real.pi

theorem minimum_distance_between_extrema_is_2_sqrt_pi (a : ℝ) (h : a > 0) :
  minimum_distance_adjacent_extrema a h = 2 * Real.sqrt Real.pi := 
sorry

end minimum_distance_between_extrema_is_2_sqrt_pi_l1385_138511


namespace original_price_l1385_138583

variable (x : ℝ)

-- Condition 1: Selling at 60% of the original price results in a 20 yuan loss
def condition1 : Prop := 0.6 * x + 20 = x * 0.8 - 15

-- The goal is to prove that the original price is 175 yuan under the given conditions
theorem original_price (h : condition1 x) : x = 175 :=
sorry

end original_price_l1385_138583


namespace laura_charges_for_truck_l1385_138573

theorem laura_charges_for_truck : 
  ∀ (car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars : ℕ),
  car_wash = 5 →
  suv_wash = 7 →
  num_suvs = 5 →
  num_trucks = 5 →
  num_cars = 7 →
  total_amount = 100 →
  car_wash * num_cars + suv_wash * num_suvs + truck_wash * num_trucks = total_amount →
  truck_wash = 6 :=
by
  intros car_wash suv_wash truck_wash total_amount num_suvs num_trucks num_cars h1 h2 h3 h4 h5 h6 h7
  sorry

end laura_charges_for_truck_l1385_138573


namespace blake_spent_on_apples_l1385_138529

noncomputable def apples_spending_problem : Prop :=
  let initial_amount := 300
  let change_received := 150
  let oranges_cost := 40
  let mangoes_cost := 60
  let total_spent := initial_amount - change_received
  let other_fruits_cost := oranges_cost + mangoes_cost
  let apples_cost := total_spent - other_fruits_cost
  apples_cost = 50

theorem blake_spent_on_apples : apples_spending_problem :=
by
  sorry

end blake_spent_on_apples_l1385_138529


namespace sum_of_possible_x_values_l1385_138501

theorem sum_of_possible_x_values (x : ℝ) : 
  (3 : ℝ)^(x^2 + 6*x + 9) = (27 : ℝ)^(x + 3) → x = 0 ∨ x = -3 → x = 0 ∨ x = -3 := 
sorry

end sum_of_possible_x_values_l1385_138501


namespace range_of_f_l1385_138598

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x+1) + 3

theorem range_of_f : Set.range f = Set.Ici 2 := 
by 
  sorry

end range_of_f_l1385_138598


namespace general_formula_for_a_n_l1385_138593

noncomputable def f (x : ℝ) : ℝ := x^2 - 4*x + 2

-- Defining a_n as a function of n assuming it's an arithmetic sequence.
noncomputable def a (x : ℝ) (n : ℕ) : ℝ :=
  if x = 1 then 2 * n - 4 else if x = 3 then 4 - 2 * n else 0

theorem general_formula_for_a_n (x : ℝ) (n : ℕ) (h1 : a x 1 = f (x + 1))
  (h2 : a x 2 = 0) (h3 : a x 3 = f (x - 1)) :
  (x = 1 → a x n = 2 * n - 4) ∧ (x = 3 → a x n = 4 - 2 * n) :=
by sorry

end general_formula_for_a_n_l1385_138593


namespace solve_system_of_equations_l1385_138538

theorem solve_system_of_equations (x y : ℝ) : 
  (x + y = x^2 + 2 * x * y + y^2) ∧ (x - y = x^2 - 2 * x * y + y^2) ↔ 
  (x = 0 ∧ y = 0) ∨ 
  (x = 1/2 ∧ y = 1/2) ∨ 
  (x = 1/2 ∧ y = -1/2) ∨ 
  (x = 1 ∧ y = 0) :=
by
  sorry

end solve_system_of_equations_l1385_138538


namespace simplify_and_evaluate_expression_l1385_138528

theorem simplify_and_evaluate_expression (x : ℝ) (h : x^2 - 2 * x - 2 = 0) :
    ( ( (x - 1)/x - (x - 2)/(x + 1) ) / ( (2 * x^2 - x) / (x^2 + 2 * x + 1) ) = 1 / 2 ) :=
by
    -- sorry to skip the proof
    sorry

end simplify_and_evaluate_expression_l1385_138528


namespace percentage_of_16_l1385_138539

theorem percentage_of_16 (p : ℝ) (h : (p / 100) * 16 = 0.04) : p = 0.25 :=
by
  sorry

end percentage_of_16_l1385_138539


namespace hockey_games_in_season_l1385_138536

theorem hockey_games_in_season
  (games_per_month : ℤ)
  (months_in_season : ℤ)
  (h1 : games_per_month = 25)
  (h2 : months_in_season = 18) :
  games_per_month * months_in_season = 450 :=
by
  sorry

end hockey_games_in_season_l1385_138536


namespace solution_set_l1385_138597

theorem solution_set (x : ℝ) : (⌊x⌋ + ⌈x⌉ = 7) ↔ (3 < x ∧ x < 4) :=
by
  sorry

end solution_set_l1385_138597


namespace value_of_expression_l1385_138556

variables (a b c d : ℝ)

def f (x : ℝ) : ℝ := a * x ^ 3 + b * x ^ 2 + c * x + d

theorem value_of_expression (h : f a b c d (-2) = -3) : 8 * a - 4 * b + 2 * c - d = 3 :=
by {
  sorry
}

end value_of_expression_l1385_138556


namespace popped_white_probability_l1385_138567

theorem popped_white_probability :
  let P_white := 2 / 3
  let P_yellow := 1 / 3
  let P_pop_given_white := 1 / 2
  let P_pop_given_yellow := 2 / 3

  let P_white_and_pop := P_white * P_pop_given_white
  let P_yellow_and_pop := P_yellow * P_pop_given_yellow
  let P_pop := P_white_and_pop + P_yellow_and_pop

  let P_white_given_pop := P_white_and_pop / P_pop

  P_white_given_pop = 3 / 5 := sorry

end popped_white_probability_l1385_138567


namespace prove_a2_minus_b2_l1385_138509

theorem prove_a2_minus_b2 : 
  ∀ (a b : ℚ), 
  a + b = 9 / 17 ∧ a - b = 1 / 51 → a^2 - b^2 = 3 / 289 :=
by
  intros a b h
  cases' h
  sorry

end prove_a2_minus_b2_l1385_138509


namespace john_baseball_cards_l1385_138569

theorem john_baseball_cards (new_cards old_cards cards_per_page : ℕ) (h1 : new_cards = 8) (h2 : old_cards = 16) (h3 : cards_per_page = 3) :
  (new_cards + old_cards) / cards_per_page = 8 := by
  sorry

end john_baseball_cards_l1385_138569


namespace clara_cookies_l1385_138527

theorem clara_cookies (n : ℕ) :
  (15 * n - 1) % 11 = 0 → n = 3 := 
sorry

end clara_cookies_l1385_138527


namespace weight_of_11th_person_l1385_138506

theorem weight_of_11th_person
  (n : ℕ) (avg1 avg2 : ℝ)
  (hn : n = 10)
  (havg1 : avg1 = 165)
  (havg2 : avg2 = 170)
  (W : ℝ) (X : ℝ)
  (hw : W = n * avg1)
  (havg2_eq : (W + X) / (n + 1) = avg2) :
  X = 220 :=
by
  sorry

end weight_of_11th_person_l1385_138506


namespace correct_option_l1385_138540

theorem correct_option : ∀ (x y : ℝ), 10 * x * y - 10 * y * x = 0 :=
by 
  intros x y
  sorry

end correct_option_l1385_138540


namespace least_subtracted_number_l1385_138517

theorem least_subtracted_number (r : ℕ) : r = 10^1000 % 97 := 
sorry

end least_subtracted_number_l1385_138517


namespace smallest_n_probability_l1385_138553

theorem smallest_n_probability (n : ℕ) : (1 / (n * (n + 1)) < 1 / 2023) → (n ≥ 45) :=
by
  sorry

end smallest_n_probability_l1385_138553


namespace fraction_of_capacity_l1385_138576

theorem fraction_of_capacity
    (bus_capacity : ℕ)
    (x : ℕ)
    (first_pickup : ℕ)
    (second_pickup : ℕ)
    (unable_to_board : ℕ)
    (bus_full : bus_capacity = x + (second_pickup - unable_to_board))
    (carry_fraction : x / bus_capacity = 3 / 5) : 
    true := 
sorry

end fraction_of_capacity_l1385_138576


namespace find_speed_first_car_l1385_138542

noncomputable def speed_first_car (v : ℝ) : Prop :=
  let t := (14 : ℝ) / 3
  let d_total := 490
  let d_second_car := 60 * t
  let d_first_car := v * t
  d_second_car + d_first_car = d_total

theorem find_speed_first_car : ∃ v : ℝ, speed_first_car v ∧ v = 45 :=
by
  sorry

end find_speed_first_car_l1385_138542


namespace cars_meet_time_l1385_138516

theorem cars_meet_time (s1 s2 : ℝ) (d : ℝ) (c : s1 = (5 / 4) * s2) 
  (h1 : s1 = 100) (h2 : d = 720) : d / (s1 + s2) = 4 :=
by 
  sorry

end cars_meet_time_l1385_138516


namespace vacation_cost_split_l1385_138507

theorem vacation_cost_split 
  (john_paid mary_paid lisa_paid : ℕ) 
  (total_amount : ℕ) 
  (share : ℕ)
  (j m : ℤ)
  (h1 : john_paid = 150)
  (h2 : mary_paid = 90)
  (h3 : lisa_paid = 210)
  (h4 : total_amount = 450)
  (h5 : share = total_amount / 3) 
  (h6 : john_paid - share = j) 
  (h7 : mary_paid - share = m) 
  : j - m = -60 :=
by
  sorry

end vacation_cost_split_l1385_138507


namespace root_condition_l1385_138510

-- Let f(x) = x^2 + ax + a^2 - a - 2
noncomputable def f (a x : ℝ) : ℝ := x^2 + a * x + a^2 - a - 2

theorem root_condition (a : ℝ) (h1 : ∀ ζ : ℝ, (ζ > 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0) ∧ (ζ < 1 → ζ^2 + a * ζ + a^2 - a - 2 = 0)) :
  -1 < a ∧ a < 1 :=
sorry

end root_condition_l1385_138510


namespace determinant_problem_l1385_138544

variables {p q r s : ℝ}

theorem determinant_problem
  (h : p * s - q * r = 5) :
  p * (4 * r + 2 * s) - (4 * p + 2 * q) * r = 10 := 
sorry

end determinant_problem_l1385_138544


namespace octopus_legs_l1385_138535

/-- Four octopuses made statements about their total number of legs.
    - Octopuses with 7 legs always lie.
    - Octopuses with 6 or 8 legs always tell the truth.
    - Blue: "Together we have 28 legs."
    - Green: "Together we have 27 legs."
    - Yellow: "Together we have 26 legs."
    - Red: "Together we have 25 legs."
   Prove that the Green octopus has 6 legs, and the Blue, Yellow, and Red octopuses each have 7 legs.
-/
theorem octopus_legs (L_B L_G L_Y L_R : ℕ) (H1 : (L_B + L_G + L_Y + L_R = 28 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 27 → L_B + L_G + L_Y + L_R = 27) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 26 → L_B ≠ 7) ∧ 
                                                  (L_B + L_G + L_Y + L_R = 25 → L_B ≠ 7)) : 
  (L_G = 6) ∧ (L_B = 7) ∧ (L_Y = 7) ∧ (L_R = 7) :=
sorry

end octopus_legs_l1385_138535


namespace solve_for_x_l1385_138505

theorem solve_for_x (a r s x : ℝ) (h1 : s > r) (h2 : r * (x + a) = s * (x - a)) :
  x = a * (s + r) / (s - r) :=
sorry

end solve_for_x_l1385_138505


namespace represent_nat_as_combinations_l1385_138574

theorem represent_nat_as_combinations (n : ℕ) :
  ∃ x y z : ℕ,
  (0 ≤ x ∧ x < y ∧ y < z ∨ 0 = x ∧ x = y ∧ y < z) ∧
  (n = Nat.choose x 1 + Nat.choose y 2 + Nat.choose z 3) :=
sorry

end represent_nat_as_combinations_l1385_138574


namespace sqrt_mul_example_complex_expression_example_l1385_138555

theorem sqrt_mul_example : Real.sqrt 3 * Real.sqrt 27 = 9 :=
by sorry

theorem complex_expression_example : 
  (Real.sqrt 2 + 1) * (Real.sqrt 2 - 1) - (Real.sqrt 3 - 2)^2 = 4 * Real.sqrt 3 - 6 :=
by sorry

end sqrt_mul_example_complex_expression_example_l1385_138555


namespace exp_problem_l1385_138545

theorem exp_problem (a b c : ℕ) (H1 : a = 1000) (H2 : b = 1000^1000) (H3 : c = 500^1000) :
  a * b / c = 2^1001 * 500 :=
sorry

end exp_problem_l1385_138545


namespace part_a_l1385_138519

theorem part_a : (2^41 + 1) % 83 = 0 :=
  sorry

end part_a_l1385_138519


namespace integer_k_values_l1385_138599

theorem integer_k_values (a b k : ℝ) (m : ℝ) (ha : a > 0) (hb : b > 0) (hba_int : ∃ n : ℤ, n ≠ 0 ∧ b = (n : ℝ) * a) 
  (hA : a = a * k + m) (hB : 8 * b = b * k + m) : k = 9 ∨ k = 15 := 
by
  sorry

end integer_k_values_l1385_138599


namespace molecular_weight_of_acid_l1385_138587

theorem molecular_weight_of_acid (molecular_weight : ℕ) (n : ℕ) (h : molecular_weight = 792) (hn : n = 9) :
  molecular_weight = 792 :=
by 
  sorry

end molecular_weight_of_acid_l1385_138587


namespace minimum_days_bacteria_count_exceeds_500_l1385_138551

theorem minimum_days_bacteria_count_exceeds_500 :
  ∃ n : ℕ, 4 * 3^n > 500 ∧ ∀ m : ℕ, m < n → 4 * 3^m ≤ 500 :=
by
  sorry

end minimum_days_bacteria_count_exceeds_500_l1385_138551


namespace students_count_l1385_138563

theorem students_count (x y : ℕ) (h1 : 3 * x + 20 = y) (h2 : 4 * x - 25 = y) : x = 45 :=
by {
  sorry
}

end students_count_l1385_138563


namespace smallest_y_76545_l1385_138580

theorem smallest_y_76545 (y : ℕ) (h1 : ∀ z : ℕ, 0 < z → (76545 * z = k ^ 2 → (3 ∣ z ∨ 5 ∣ z) → z = y)) : y = 7 :=
sorry

end smallest_y_76545_l1385_138580


namespace total_payment_correct_l1385_138565

def cost (n : ℕ) : ℕ :=
  if n <= 10 then n * 25
  else 10 * 25 + (n - 10) * (4 * 25 / 5)

def final_cost_with_discount (n : ℕ) : ℕ :=
  let initial_cost := cost n
  if n > 20 then initial_cost - initial_cost / 10
  else initial_cost

def orders_X := 60 * 20 / 100
def orders_Y := 60 * 25 / 100
def orders_Z := 60 * 55 / 100

def cost_X := final_cost_with_discount orders_X
def cost_Y := final_cost_with_discount orders_Y
def cost_Z := final_cost_with_discount orders_Z

theorem total_payment_correct : cost_X + cost_Y + cost_Z = 1279 := by
  sorry

end total_payment_correct_l1385_138565
