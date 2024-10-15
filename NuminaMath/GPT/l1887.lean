import Mathlib

namespace NUMINAMATH_GPT_prob_first_given_defective_correct_l1887_188702

-- Definitions from problem conditions
def first_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
def second_box : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
def defective_first_box : Set ℕ := {1, 2, 3}
def defective_second_box : Set ℕ := {1, 2}

-- Probability values as defined
def prob_first_box : ℚ := 1 / 2
def prob_second_box : ℚ := 1 / 2
def prob_defective_given_first : ℚ := 3 / 10
def prob_defective_given_second : ℚ := 1 / 10

-- Calculation of total probability of defective component
def prob_defective : ℚ := (prob_first_box * prob_defective_given_first) + (prob_second_box * prob_defective_given_second)

-- Bayes' Theorem application to find the required probability
def prob_first_given_defective : ℚ := (prob_first_box * prob_defective_given_first) / prob_defective

-- Lean statement to verify the computed probability is as expected
theorem prob_first_given_defective_correct : prob_first_given_defective = 3 / 4 :=
by
  unfold prob_first_given_defective prob_defective
  sorry

end NUMINAMATH_GPT_prob_first_given_defective_correct_l1887_188702


namespace NUMINAMATH_GPT_points_per_member_l1887_188744

theorem points_per_member
    (total_members : ℕ)
    (absent_members : ℕ)
    (total_points : ℕ)
    (present_members : ℕ)
    (points_per_member : ℕ)
    (h1 : total_members = 5)
    (h2 : absent_members = 2)
    (h3 : total_points = 18)
    (h4 : present_members = total_members - absent_members)
    (h5 : points_per_member = total_points / present_members) :
  points_per_member = 6 :=
by
  sorry

end NUMINAMATH_GPT_points_per_member_l1887_188744


namespace NUMINAMATH_GPT_perfect_squares_as_difference_l1887_188739

theorem perfect_squares_as_difference (N : ℕ) (hN : N = 20000) : 
  (∃ (n : ℕ), n = 71 ∧ 
    ∀ m < N, 
      (∃ a b : ℤ, 
        a^2 = m ∧
        b^2 = m + ((b + 1)^2 - b^2) - 1 ∧ 
        (b + 1)^2 - b^2 = 2 * b + 1)) :=
by 
  sorry

end NUMINAMATH_GPT_perfect_squares_as_difference_l1887_188739


namespace NUMINAMATH_GPT_arithmetic_geom_seq_a1_over_d_l1887_188724

theorem arithmetic_geom_seq_a1_over_d (a1 a2 a3 a4 d : ℝ) (hne : d ≠ 0)
  (hgeom1 : (a1 + 2*d)^2 = a1 * (a1 + 3*d))
  (hgeom2 : (a1 + d)^2 = a1 * (a1 + 3*d)) :
  (a1 / d = -4) ∨ (a1 / d = 1) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geom_seq_a1_over_d_l1887_188724


namespace NUMINAMATH_GPT_least_possible_coins_l1887_188756

theorem least_possible_coins : 
  ∃ b : ℕ, b % 7 = 3 ∧ b % 4 = 2 ∧ ∀ n : ℕ, (n % 7 = 3 ∧ n % 4 = 2) → b ≤ n :=
sorry

end NUMINAMATH_GPT_least_possible_coins_l1887_188756


namespace NUMINAMATH_GPT_correct_average_l1887_188776

theorem correct_average (incorrect_avg : ℝ) (n : ℕ) (wrong_num correct_num : ℝ)
  (h_avg : incorrect_avg = 23)
  (h_n : n = 10)
  (h_wrong : wrong_num = 26)
  (h_correct : correct_num = 36) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 24 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_correct_average_l1887_188776


namespace NUMINAMATH_GPT_vendor_has_maaza_l1887_188767

theorem vendor_has_maaza (liters_pepsi : ℕ) (liters_sprite : ℕ) (total_cans : ℕ) (gcd_pepsi_sprite : ℕ) (cans_pepsi : ℕ) (cans_sprite : ℕ) (cans_maaza : ℕ) (liters_per_can : ℕ) (total_liters_maaza : ℕ) :
  liters_pepsi = 144 →
  liters_sprite = 368 →
  total_cans = 133 →
  gcd_pepsi_sprite = Nat.gcd liters_pepsi liters_sprite →
  gcd_pepsi_sprite = 16 →
  cans_pepsi = liters_pepsi / gcd_pepsi_sprite →
  cans_sprite = liters_sprite / gcd_pepsi_sprite →
  cans_maaza = total_cans - (cans_pepsi + cans_sprite) →
  liters_per_can = gcd_pepsi_sprite →
  total_liters_maaza = cans_maaza * liters_per_can →
  total_liters_maaza = 1616 :=
by
  sorry

end NUMINAMATH_GPT_vendor_has_maaza_l1887_188767


namespace NUMINAMATH_GPT_power_subtraction_l1887_188741

theorem power_subtraction : 2^4 - 2^3 = 2^3 := by
  sorry

end NUMINAMATH_GPT_power_subtraction_l1887_188741


namespace NUMINAMATH_GPT_complement_of_A_eq_interval_l1887_188772

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x ≥ 1} ∪ {x | x < 0}
def complement_U_A : Set ℝ := {x | 0 ≤ x ∧ x < 1}

theorem complement_of_A_eq_interval : (U \ A) = complement_U_A := by
  sorry

end NUMINAMATH_GPT_complement_of_A_eq_interval_l1887_188772


namespace NUMINAMATH_GPT_regular_polygon_sides_l1887_188764

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) (interior_angle : ℝ) : 
  interior_angle = 144 → n = 10 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_regular_polygon_sides_l1887_188764


namespace NUMINAMATH_GPT_optimal_green_tiles_l1887_188710

variable (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ)

def conditions (n_red n_orange n_yellow n_green n_blue n_indigo : ℕ) :=
  n_indigo ≥ n_red + n_orange + n_yellow + n_green + n_blue ∧
  n_blue ≥ n_red + n_orange + n_yellow + n_green ∧
  n_green ≥ n_red + n_orange + n_yellow ∧
  n_yellow ≥ n_red + n_orange ∧
  n_orange ≥ n_red ∧
  n_red + n_orange + n_yellow + n_green + n_blue + n_indigo = 100

theorem optimal_green_tiles : 
  conditions n_red n_orange n_yellow n_green n_blue n_indigo → 
  n_green = 13 := by
    sorry

end NUMINAMATH_GPT_optimal_green_tiles_l1887_188710


namespace NUMINAMATH_GPT_intersection_proof_complement_proof_range_of_m_condition_l1887_188761

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | -3 < x ∧ x < 1}
def C (m : ℝ) : Set ℝ := {x | 2 - m ≤ x ∧ x ≤ 2 + m}

theorem intersection_proof : A ∩ B = {x | -2 ≤ x ∧ x < 1} := sorry

theorem complement_proof : (Set.univ \ B) = {x | x ≤ -3 ∨ x ≥ 1} := sorry

theorem range_of_m_condition (m : ℝ) : (A ∪ C m = A) → (m ≤ 2) := sorry

end NUMINAMATH_GPT_intersection_proof_complement_proof_range_of_m_condition_l1887_188761


namespace NUMINAMATH_GPT_tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l1887_188731

noncomputable def f (x m : ℝ) : ℝ := (Real.exp (x - 1) - 0.5 * x^2 + x - m * Real.log x)

theorem tangent_line_at_one (m : ℝ) :
  ∃ (y : ℝ → ℝ), (∀ x, y x = (1 - m) * x + m + 0.5) ∧ y 1 = f 1 m ∧ (tangent_slope : ℝ) = 1 - m ∧
    ∀ x, y x = f x m + y 0 :=
sorry

theorem m_positive_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  m > 0 :=
sorry

theorem ineq_if_equal_vals (m x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ m = f x₂ m) :
  2 * m > Real.exp (Real.log x₁ + Real.log x₂) :=
sorry

end NUMINAMATH_GPT_tangent_line_at_one_m_positive_if_equal_vals_ineq_if_equal_vals_l1887_188731


namespace NUMINAMATH_GPT_paint_house_l1887_188734

theorem paint_house (n s h : ℕ) (h_pos : 0 < h)
    (rate_eq : ∀ (x : ℕ), 0 < x → ∃ t : ℕ, x * t = n * h) :
    (n + s) * (nh / (n + s)) = n * h := 
sorry

end NUMINAMATH_GPT_paint_house_l1887_188734


namespace NUMINAMATH_GPT_steven_sixth_quiz_score_l1887_188762

theorem steven_sixth_quiz_score :
  ∃ x : ℕ, (75 + 80 + 85 + 90 + 100 + x) / 6 = 95 ∧ x = 140 :=
by
  sorry

end NUMINAMATH_GPT_steven_sixth_quiz_score_l1887_188762


namespace NUMINAMATH_GPT_find_non_zero_real_x_satisfies_equation_l1887_188786

theorem find_non_zero_real_x_satisfies_equation :
  ∃! x : ℝ, x ≠ 0 ∧ (9 * x) ^ 18 - (18 * x) ^ 9 = 0 ∧ x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_non_zero_real_x_satisfies_equation_l1887_188786


namespace NUMINAMATH_GPT_handshake_count_l1887_188716

theorem handshake_count : 
  let n := 5  -- number of representatives per company
  let c := 5  -- number of companies
  let total_people := n * c  -- total number of people
  let handshakes_per_person := total_people - n  -- each person shakes hands with 20 others
  (total_people * handshakes_per_person) / 2 = 250 := 
by
  sorry

end NUMINAMATH_GPT_handshake_count_l1887_188716


namespace NUMINAMATH_GPT_problem_solution_l1887_188745

theorem problem_solution :
  (2^2 + 4^2 + 6^2) / (1^2 + 3^2 + 5^2) - (1^2 + 3^2 + 5^2) / (2^2 + 4^2 + 6^2) = 1911 / 1960 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1887_188745


namespace NUMINAMATH_GPT_total_birds_in_marsh_l1887_188715

-- Define the number of geese and ducks as constants.
def geese : Nat := 58
def ducks : Nat := 37

-- The theorem that we need to prove.
theorem total_birds_in_marsh : geese + ducks = 95 :=
by
  -- Here, we add the sorry keyword to skip the proof part.
  sorry

end NUMINAMATH_GPT_total_birds_in_marsh_l1887_188715


namespace NUMINAMATH_GPT_book_pages_l1887_188791

theorem book_pages (pages_per_day : ℕ) (days : ℕ) (total_pages : ℕ) : 
  pages_per_day = 8 → days = 12 → total_pages = pages_per_day * days → total_pages = 96 :=
by 
  intro h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_book_pages_l1887_188791


namespace NUMINAMATH_GPT_sum_of_eight_terms_l1887_188777

theorem sum_of_eight_terms :
  (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) + (5 ^ 8) = 3125000 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_eight_terms_l1887_188777


namespace NUMINAMATH_GPT_quadratic_binomial_square_l1887_188726

theorem quadratic_binomial_square (a : ℚ) :
  (∃ r s : ℚ, (ax^2 + 22*x + 9 = (r*x + s)^2) ∧ s = 3 ∧ r = 11 / 3) → a = 121 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_binomial_square_l1887_188726


namespace NUMINAMATH_GPT_length_PD_l1887_188711

theorem length_PD (PA PB PC PD : ℝ) (hPA : PA = 5) (hPB : PB = 3) (hPC : PC = 4) :
  PD = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_length_PD_l1887_188711


namespace NUMINAMATH_GPT_jacob_twice_as_old_l1887_188703

theorem jacob_twice_as_old (x : ℕ) : 18 + x = 2 * (9 + x) → x = 0 := by
  intro h
  linarith

end NUMINAMATH_GPT_jacob_twice_as_old_l1887_188703


namespace NUMINAMATH_GPT_common_ratio_l1887_188789

theorem common_ratio
  (a b : ℝ)
  (h_arith : 2 * a = 1 + b)
  (h_geom : (a + 2) ^ 2 = 3 * (b + 5))
  (h_non_zero_a : a + 2 ≠ 0)
  (h_non_zero_b : b + 5 ≠ 0) :
  (a = 4 ∧ b = 7) ∧ (b + 5) / (a + 2) = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_common_ratio_l1887_188789


namespace NUMINAMATH_GPT_lcm_1230_924_l1887_188795

theorem lcm_1230_924 : Nat.lcm 1230 924 = 189420 :=
by
  /- Proof steps skipped -/
  sorry

end NUMINAMATH_GPT_lcm_1230_924_l1887_188795


namespace NUMINAMATH_GPT_ball_hits_ground_time_l1887_188749

theorem ball_hits_ground_time :
  ∃ t : ℝ, -20 * t^2 + 30 * t + 60 = 0 ∧ t = (3 + Real.sqrt 57) / 4 :=
by 
  sorry

end NUMINAMATH_GPT_ball_hits_ground_time_l1887_188749


namespace NUMINAMATH_GPT_tan_alpha_value_l1887_188780

open Real

theorem tan_alpha_value (α : ℝ) (h1 : sin α + cos α = -1 / 2) (h2 : 0 < α ∧ α < π) : tan α = -1 / 3 :=
sorry

end NUMINAMATH_GPT_tan_alpha_value_l1887_188780


namespace NUMINAMATH_GPT_equivalence_condition_l1887_188713

universe u

variables {U : Type u} (A B : Set U)

theorem equivalence_condition :
  (∃ (C : Set U), A ⊆ C ∧ B ⊆ Cᶜ) ↔ (A ∩ B = ∅) :=
sorry

end NUMINAMATH_GPT_equivalence_condition_l1887_188713


namespace NUMINAMATH_GPT_max_small_boxes_l1887_188719

-- Define the dimensions of the larger box in meters
def large_box_length : ℝ := 6
def large_box_width : ℝ := 5
def large_box_height : ℝ := 4

-- Define the dimensions of the smaller box in meters
def small_box_length : ℝ := 0.60
def small_box_width : ℝ := 0.50
def small_box_height : ℝ := 0.40

-- Calculate the volume of the larger box
def large_box_volume : ℝ := large_box_length * large_box_width * large_box_height

-- Calculate the volume of the smaller box
def small_box_volume : ℝ := small_box_length * small_box_width * small_box_height

-- State the theorem to prove the maximum number of smaller boxes that can fit in the larger box
theorem max_small_boxes : large_box_volume / small_box_volume = 1000 :=
by
  sorry

end NUMINAMATH_GPT_max_small_boxes_l1887_188719


namespace NUMINAMATH_GPT_even_function_a_eq_neg_one_l1887_188706

-- Definitions for the function f and the condition for it being an even function
def f (x a : ℝ) := (x - 1) * (x - a)

-- The theorem stating that if f is an even function, then a = -1
theorem even_function_a_eq_neg_one (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_a_eq_neg_one_l1887_188706


namespace NUMINAMATH_GPT_pushups_total_l1887_188793

theorem pushups_total (x melanie david karen john : ℕ) 
  (hx : x = 51)
  (h_melanie : melanie = 2 * x - 7)
  (h_david : david = x + 22)
  (h_avg : (x + melanie + david) / 3 = (x + (2 * x - 7) + (x + 22)) / 3)
  (h_karen : karen = (x + (2 * x - 7) + (x + 22)) / 3 - 5)
  (h_john : john = (x + 22) - 4) :
  john + melanie + karen = 232 := by
  sorry

end NUMINAMATH_GPT_pushups_total_l1887_188793


namespace NUMINAMATH_GPT_dealer_sold_70_hondas_l1887_188766

theorem dealer_sold_70_hondas
  (total_cars: ℕ)
  (percent_audi percent_toyota percent_acura percent_honda : ℝ)
  (total_audi := total_cars * percent_audi)
  (total_toyota := total_cars * percent_toyota)
  (total_acura := total_cars * percent_acura)
  (total_honda := total_cars * percent_honda )
  (h1 : total_cars = 200)
  (h2 : percent_audi = 0.15)
  (h3 : percent_toyota = 0.22)
  (h4 : percent_acura = 0.28)
  (h5 : percent_honda = 1 - (percent_audi + percent_toyota + percent_acura))
  : total_honda = 70 := 
  by
  sorry

end NUMINAMATH_GPT_dealer_sold_70_hondas_l1887_188766


namespace NUMINAMATH_GPT_find_X_l1887_188768

theorem find_X (X : ℕ) : 
  (∃ k : ℕ, X = 26 * k + k) ∧ (∃ m : ℕ, X = 29 * m + m) → (X = 270 ∨ X = 540) :=
by
  sorry

end NUMINAMATH_GPT_find_X_l1887_188768


namespace NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l1887_188712

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_3_pow_19_mod_10_l1887_188712


namespace NUMINAMATH_GPT_probability_three_or_more_same_l1887_188750

-- Let us define the total number of outcomes when rolling 5 8-sided dice
def total_outcomes : ℕ := 8 ^ 5

-- Define the number of favorable outcomes where at least three dice show the same number
def favorable_outcomes : ℕ := 4208

-- Define the probability as a fraction
def probability : ℚ := favorable_outcomes / total_outcomes

-- Now we state the theorem that this probability simplifies to 1052/8192
theorem probability_three_or_more_same : probability = 1052 / 8192 :=
sorry

end NUMINAMATH_GPT_probability_three_or_more_same_l1887_188750


namespace NUMINAMATH_GPT_find_y_l1887_188707

variable {a b y : ℝ}
variable (ha : a ≠ 0) (hb : b ≠ 0)

theorem find_y (h1 : (3 * a) ^ (4 * b) = a ^ b * y ^ b) : y = 81 * a ^ 3 := by
  sorry

end NUMINAMATH_GPT_find_y_l1887_188707


namespace NUMINAMATH_GPT_raised_bed_section_area_l1887_188735

theorem raised_bed_section_area :
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  area_of_raised_beds = 8800 :=
by 
  let length := 220
  let width := 120
  let total_area := length * width
  let half_area := total_area / 2
  let fraction_for_raised_beds := 2 / 3
  let area_of_raised_beds := half_area * fraction_for_raised_beds
  show area_of_raised_beds = 8800
  sorry

end NUMINAMATH_GPT_raised_bed_section_area_l1887_188735


namespace NUMINAMATH_GPT_ratio_netbooks_is_one_third_l1887_188748

open Nat

def total_computers (total : ℕ) : Prop := total = 72
def laptops_sold (laptops : ℕ) (total : ℕ) : Prop := laptops = total / 2
def desktops_sold (desktops : ℕ) : Prop := desktops = 12
def netbooks_sold (netbooks : ℕ) (total laptops desktops : ℕ) : Prop :=
  netbooks = total - (laptops + desktops)
def ratio_netbooks_total (netbooks total : ℕ) : Prop :=
  netbooks * 3 = total

theorem ratio_netbooks_is_one_third
  (total laptops desktops netbooks : ℕ)
  (h_total : total_computers total)
  (h_laptops : laptops_sold laptops total)
  (h_desktops : desktops_sold desktops)
  (h_netbooks : netbooks_sold netbooks total laptops desktops) :
  ratio_netbooks_total netbooks total :=
by
  sorry

end NUMINAMATH_GPT_ratio_netbooks_is_one_third_l1887_188748


namespace NUMINAMATH_GPT_range_of_m_l1887_188771

-- Define the propositions
def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

-- Problem statement to derive m's range
theorem range_of_m (m : ℝ) (h1: ¬ (p m ∧ q m)) (h2: p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end NUMINAMATH_GPT_range_of_m_l1887_188771


namespace NUMINAMATH_GPT_none_satisfied_l1887_188718

-- Define the conditions
variables {a b c x y z : ℝ}
  
-- Theorem that states that none of the given inequalities are satisfied strictly
theorem none_satisfied (h1 : x ≤ a) (h2 : y ≤ b) (h3 : z ≤ c) :
  ¬(x^2 * y + y^2 * z + z^2 * x < a^2 * b + b^2 * c + c^2 * a) ∧
  ¬(x^3 + y^3 + z^3 < a^3 + b^3 + c^3) :=
  by
    sorry

end NUMINAMATH_GPT_none_satisfied_l1887_188718


namespace NUMINAMATH_GPT_certain_number_divisible_by_9_l1887_188743

theorem certain_number_divisible_by_9 : ∃ N : ℕ, (∀ k : ℕ, (0 ≤ k ∧ k < 1110 → N + 9 * k ≤ 10000 ∧ (N + 9 * k) % 9 = 0)) ∧ N = 27 :=
by
  -- Given conditions:
  -- Numbers are in an arithmetic sequence with common difference 9.
  -- Total count of such numbers is 1110.
  -- The last number ≤ 10000 that is divisible by 9 is 9999.
  let L := 9999
  let n := 1110
  let d := 9
  -- First term in the sequence:
  let a := L - (n - 1) * d
  exists 27
  -- Proof of the conditions would follow here ...
  sorry

end NUMINAMATH_GPT_certain_number_divisible_by_9_l1887_188743


namespace NUMINAMATH_GPT_casper_initial_candies_l1887_188775

theorem casper_initial_candies : 
  ∃ x : ℕ, 
    (∃ y1 : ℕ, y1 = x / 2 - 3) ∧
    (∃ y2 : ℕ, y2 = y1 / 2 - 5) ∧
    (∃ y3 : ℕ, y3 = y2 / 2 - 2) ∧
    (y3 = 10) ∧
    x = 122 := 
sorry

end NUMINAMATH_GPT_casper_initial_candies_l1887_188775


namespace NUMINAMATH_GPT_ecuadorian_number_unique_l1887_188722

def is_Ecuadorian (n : ℕ) : Prop :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  n < 1000 ∧ c ≠ 0 ∧ n % 36 = 0 ∧ (n - (100 * c + 10 * b + a) > 0) ∧ (n - (100 * c + 10 * b + a)) % 36 = 0

theorem ecuadorian_number_unique (n : ℕ) : 
  is_Ecuadorian n → n = 864 :=
sorry

end NUMINAMATH_GPT_ecuadorian_number_unique_l1887_188722


namespace NUMINAMATH_GPT_find_a5_l1887_188797

variable {a : ℕ → ℝ}

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m : ℕ, ∃ q : ℝ, a (n + m) = a n * q ^ m

theorem find_a5
  (h : geometric_sequence a)
  (h3 : a 3 = 2)
  (h7 : a 7 = 8) :
  a 5 = 4 :=
sorry

end NUMINAMATH_GPT_find_a5_l1887_188797


namespace NUMINAMATH_GPT_positive_integer_in_base_proof_l1887_188794

noncomputable def base_conversion_problem (A B : ℕ) (n : ℕ) : Prop :=
  n = 9 * A + B ∧ n = 8 * B + A ∧ A < 9 ∧ B < 8 ∧ A ≠ 0 ∧ B ≠ 0

theorem positive_integer_in_base_proof (A B n : ℕ) (h : base_conversion_problem A B n) : n = 0 :=
sorry

end NUMINAMATH_GPT_positive_integer_in_base_proof_l1887_188794


namespace NUMINAMATH_GPT_cards_arrangement_count_is_10_l1887_188747

-- Define the problem in Lean statement terms
def valid_arrangements_count : ℕ :=
  -- number of arrangements of seven cards where one card can be removed 
  -- leaving the remaining six cards in either ascending or descending order
  10

-- Theorem stating that the number of valid arrangements is 10
theorem cards_arrangement_count_is_10 : valid_arrangements_count = 10 :=
by
  -- Proof is omitted (the explanation above corresponds to the omitted proof details)
  sorry

end NUMINAMATH_GPT_cards_arrangement_count_is_10_l1887_188747


namespace NUMINAMATH_GPT_solve_for_b_l1887_188754

noncomputable def system_has_solution (b : ℝ) : Prop :=
  ∃ (a : ℝ) (x y : ℝ),
    y = -b - x^2 ∧
    x^2 + y^2 + 8 * a^2 = 4 + 4 * a * (x + y)

theorem solve_for_b (b : ℝ) : system_has_solution b ↔ b ≤ 2 * Real.sqrt 2 + 1 / 4 := 
by 
  sorry

end NUMINAMATH_GPT_solve_for_b_l1887_188754


namespace NUMINAMATH_GPT_functional_equation_l1887_188784

noncomputable def f : ℝ → ℝ :=
  sorry

theorem functional_equation (h : ∀ x : ℝ, f x + 3 * f (8 - x) = x) : f 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_functional_equation_l1887_188784


namespace NUMINAMATH_GPT_range_of_m_l1887_188778

-- Define the function g as an even function on the interval [-2, 2] 
-- and monotonically decreasing on [0, 2]

variable {g : ℝ → ℝ}

axiom even_g : ∀ x, g x = g (-x)
axiom mono_dec_g : ∀ {x y}, 0 ≤ x → x ≤ y → g y ≤ g x
axiom domain_g : ∀ x, -2 ≤ x ∧ x ≤ 2

theorem range_of_m (m : ℝ) (hm : -2 ≤ m ∧ m ≤ 2) (h : g (1 - m) < g m) : -1 ≤ m ∧ m < 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1887_188778


namespace NUMINAMATH_GPT_number_of_diagonal_intersections_of_convex_n_gon_l1887_188746

theorem number_of_diagonal_intersections_of_convex_n_gon (n : ℕ) (h : 4 ≤ n) :
  (∀ P : Π m, m = n ↔ m ≥ 4, ∃ i : ℕ, i = n * (n - 1) * (n - 2) * (n - 3) / 24) := 
by
  sorry

end NUMINAMATH_GPT_number_of_diagonal_intersections_of_convex_n_gon_l1887_188746


namespace NUMINAMATH_GPT_fruit_vendor_l1887_188788

theorem fruit_vendor (x y a b : ℕ) (C1 : 60 * x + 40 * y = 3100) (C2 : x + y = 60) 
                     (C3 : 15 * a + 20 * b = 600) (C4 : 3 * a + 4 * b = 120)
                     (C5 : 3 * a + 4 * b + 3 * (x - a) + 4 * (y - b) = 250) :
  (x = 35 ∧ y = 25) ∧ (820 - 12 * a - 16 * b = 340) ∧ (a + b = 52 ∨ a + b = 53) :=
by
  sorry

end NUMINAMATH_GPT_fruit_vendor_l1887_188788


namespace NUMINAMATH_GPT_initial_people_in_line_l1887_188752

theorem initial_people_in_line (x : ℕ) (h1 : x + 22 = 83) : x = 61 :=
by sorry

end NUMINAMATH_GPT_initial_people_in_line_l1887_188752


namespace NUMINAMATH_GPT_initial_spinach_volume_l1887_188721

theorem initial_spinach_volume (S : ℝ) (h1 : 0.20 * S + 6 + 4 = 18) : S = 40 :=
by
  sorry

end NUMINAMATH_GPT_initial_spinach_volume_l1887_188721


namespace NUMINAMATH_GPT_unique_solution_l1887_188770

noncomputable def func_prop (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (f x)^2 / x - 1 / x)

theorem unique_solution (f : ℝ → ℝ) :
  func_prop f → ∀ x ≥ 1, f x = x + 1 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_l1887_188770


namespace NUMINAMATH_GPT_goose_eggs_hatching_l1887_188727

theorem goose_eggs_hatching (x : ℝ) :
  (∃ n_hatched : ℝ, 3 * (2 * n_hatched / 20) = 110 ∧ x = n_hatched / 550) →
  x = 2 / 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_goose_eggs_hatching_l1887_188727


namespace NUMINAMATH_GPT_favouring_more_than_one_is_39_l1887_188798

def percentage_favouring_more_than_one (x : ℝ) : Prop :=
  let sum_two : ℝ := 8 + 6 + 4 + 2 + 7 + 5 + 3 + 5 + 3 + 2
  let sum_three : ℝ := 1 + 0.5 + 0.3 + 0.8 + 0.2 + 0.1 + 1.5 + 0.7 + 0.3 + 0.4
  let all_five : ℝ := 0.2
  x = sum_two - sum_three - all_five

theorem favouring_more_than_one_is_39 : percentage_favouring_more_than_one 39 := 
by
  sorry

end NUMINAMATH_GPT_favouring_more_than_one_is_39_l1887_188798


namespace NUMINAMATH_GPT_find_x_of_series_eq_16_l1887_188799

noncomputable def series_sum (x : ℝ) : ℝ :=
  ∑' n : ℕ, (2 * n + 1) * x ^ n

theorem find_x_of_series_eq_16 (x : ℝ) (h : series_sum x = 16) : x = (4 - Real.sqrt 2) / 4 :=
by
  sorry

end NUMINAMATH_GPT_find_x_of_series_eq_16_l1887_188799


namespace NUMINAMATH_GPT_necessary_condition_for_x_greater_than_2_l1887_188729

-- Define the real number x
variable (x : ℝ)

-- The proof statement
theorem necessary_condition_for_x_greater_than_2 : (x > 2) → (x > 1) :=
by sorry

end NUMINAMATH_GPT_necessary_condition_for_x_greater_than_2_l1887_188729


namespace NUMINAMATH_GPT_selection_structure_count_is_three_l1887_188730

def requiresSelectionStructure (problem : ℕ) : Bool :=
  match problem with
  | 1 => true
  | 2 => false
  | 3 => true
  | 4 => true
  | _ => false

def countSelectionStructure : ℕ :=
  (if requiresSelectionStructure 1 then 1 else 0) +
  (if requiresSelectionStructure 2 then 1 else 0) +
  (if requiresSelectionStructure 3 then 1 else 0) +
  (if requiresSelectionStructure 4 then 1 else 0)

theorem selection_structure_count_is_three : countSelectionStructure = 3 :=
  by
    sorry

end NUMINAMATH_GPT_selection_structure_count_is_three_l1887_188730


namespace NUMINAMATH_GPT_original_price_of_cupcakes_l1887_188758

theorem original_price_of_cupcakes
  (revenue : ℕ := 32) 
  (cookies_sold : ℕ := 8) 
  (cupcakes_sold : ℕ := 16) 
  (cookie_price: ℕ := 2)
  (half_price_of_cookie: ℕ := 1) :
  (x : ℕ) → (16 * (x / 2)) + (8 * 1) = 32 → x = 3 := 
by
  sorry

end NUMINAMATH_GPT_original_price_of_cupcakes_l1887_188758


namespace NUMINAMATH_GPT_expand_and_simplify_l1887_188738

theorem expand_and_simplify (x : ℝ) : (17 * x - 9) * 3 * x = 51 * x^2 - 27 * x := 
by 
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l1887_188738


namespace NUMINAMATH_GPT_transformation_is_rotation_l1887_188785

-- Define the 90 degree rotation matrix
def rotation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- Define the transformation matrix to be proven
def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![0, -1],
  ![1, 0]
]

-- The theorem that proves they are equivalent
theorem transformation_is_rotation :
  transformation_matrix = rotation_matrix :=
by
  sorry

end NUMINAMATH_GPT_transformation_is_rotation_l1887_188785


namespace NUMINAMATH_GPT_bc_sum_eq_twelve_l1887_188769

theorem bc_sum_eq_twelve (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hb_lt : b < 12) (hc_lt : c < 12) 
  (h_eq : (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c) : b + c = 12 :=
by
  sorry

end NUMINAMATH_GPT_bc_sum_eq_twelve_l1887_188769


namespace NUMINAMATH_GPT_divisor_is_20_l1887_188708

theorem divisor_is_20 (D : ℕ) 
  (h1 : 242 % D = 11) 
  (h2 : 698 % D = 18) 
  (h3 : 940 % D = 9) :
  D = 20 :=
sorry

end NUMINAMATH_GPT_divisor_is_20_l1887_188708


namespace NUMINAMATH_GPT_prove_q_l1887_188742

-- Assume the conditions
variable (p q : Prop)
variable (hpq : p ∨ q) -- "p or q" is true
variable (hnp : ¬p)    -- "not p" is true

-- The theorem to prove q is true
theorem prove_q : q :=
by {
  sorry
}

end NUMINAMATH_GPT_prove_q_l1887_188742


namespace NUMINAMATH_GPT_diff_of_squares_l1887_188723

theorem diff_of_squares (a b : ℕ) : 
  (∃ x y : ℤ, a = x^2 - y^2) ∨ (∃ x y : ℤ, b = x^2 - y^2) ∨ (∃ x y : ℤ, a + b = x^2 - y^2) :=
sorry

end NUMINAMATH_GPT_diff_of_squares_l1887_188723


namespace NUMINAMATH_GPT_fractions_are_integers_l1887_188737

theorem fractions_are_integers (x y : ℕ) 
    (h : ∃ k : ℤ, (x^2 - 1) / (y + 1) + (y^2 - 1) / (x + 1) = k) :
    ∃ u v : ℤ, (x^2 - 1) = u * (y + 1) ∧ (y^2 - 1) = v * (x + 1) := 
by
  sorry

end NUMINAMATH_GPT_fractions_are_integers_l1887_188737


namespace NUMINAMATH_GPT_shifted_function_is_correct_l1887_188760

def original_function (x : ℝ) : ℝ :=
  (x - 1)^2 + 2

def shifted_up_function (x : ℝ) : ℝ :=
  original_function x + 3

def shifted_right_function (x : ℝ) : ℝ :=
  shifted_up_function (x - 4)

theorem shifted_function_is_correct : ∀ x : ℝ, shifted_right_function x = (x - 5)^2 + 5 := 
by
  sorry

end NUMINAMATH_GPT_shifted_function_is_correct_l1887_188760


namespace NUMINAMATH_GPT_arithm_prog_diff_max_l1887_188773

noncomputable def find_most_common_difference (a b c : Int) : Prop :=
  let d := a - b
  (b = a - d) ∧ (c = a - 2 * d) ∧
  (2 * a * 2 * a - 4 * 2 * a * c ≥ 0) ∧
  (2 * a * 2 * b - 4 * 2 * a * c ≥ 0) ∧
  (2 * b * 2 * b - 4 * 2 * b * c ≥ 0) ∧
  (2 * b * c - 4 * 2 * b * a ≥ 0) ∧
  (c * c - 4 * c * 2 * b ≥ 0) ∧
  ((2 * a * c - 4 * 2 * c * b) ≥ 0)

theorem arithm_prog_diff_max (a b c Dmax: Int) : 
  find_most_common_difference 4 (-1) (-6) ∧ Dmax = -5 :=
by 
  sorry

end NUMINAMATH_GPT_arithm_prog_diff_max_l1887_188773


namespace NUMINAMATH_GPT_cos_expression_l1887_188733

-- Define the condition for the line l and its relationship
def slope_angle_of_line_l (α : ℝ) : Prop :=
  ∃ l : ℝ, l = 2

-- Given the tangent condition for α
def tan_alpha (α : ℝ) : Prop :=
  Real.tan α = 2

theorem cos_expression (α : ℝ) (h1 : slope_angle_of_line_l α) (h2 : tan_alpha α) :
  Real.cos (2015 * Real.pi / 2 - 2 * α) = -4/5 :=
by sorry

end NUMINAMATH_GPT_cos_expression_l1887_188733


namespace NUMINAMATH_GPT_probability_of_pink_l1887_188740

theorem probability_of_pink (B P : ℕ) (h1 : (B : ℚ) / (B + P) = 6 / 7) (h2 : (B^2 : ℚ) / (B + P)^2 = 36 / 49) : 
  (P : ℚ) / (B + P) = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_pink_l1887_188740


namespace NUMINAMATH_GPT_curve_is_line_l1887_188709

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end NUMINAMATH_GPT_curve_is_line_l1887_188709


namespace NUMINAMATH_GPT_work_together_l1887_188782

theorem work_together (A_days B_days : ℕ) (hA : A_days = 8) (hB : B_days = 4)
  (A_work : ℚ := 1 / A_days)
  (B_work : ℚ := 1 / B_days) :
  (A_work + B_work = 3 / 8) :=
by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_work_together_l1887_188782


namespace NUMINAMATH_GPT_tolya_is_older_by_either_4_or_22_years_l1887_188755

-- Definitions of the problem conditions
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

def kolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2013

def tolya_conditions (y : ℕ) : Prop :=
  1985 ≤ y ∧ y + sum_of_digits y = 2014

-- The problem statement
theorem tolya_is_older_by_either_4_or_22_years (k_birth t_birth : ℕ) 
  (hk : kolya_conditions k_birth) (ht : tolya_conditions t_birth) :
  t_birth - k_birth = 4 ∨ t_birth - k_birth = 22 :=
sorry

end NUMINAMATH_GPT_tolya_is_older_by_either_4_or_22_years_l1887_188755


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l1887_188717

theorem quadratic_inequality_solution
  (x : ℝ)
  (h : x^2 - 5 * x + 6 < 0) :
  2 < x ∧ x < 3 :=
by sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l1887_188717


namespace NUMINAMATH_GPT_increasing_on_interval_solution_set_l1887_188763

noncomputable def f (x : ℝ) : ℝ := x / (x ^ 2 + 1)

/- Problem 1 -/
theorem increasing_on_interval : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → x2 < 1 → f x1 < f x2 :=
by
  sorry

/- Problem 2 -/
theorem solution_set : ∀ x : ℝ, f (2 * x - 1) + f x < 0 ↔ 0 < x ∧ x < 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_increasing_on_interval_solution_set_l1887_188763


namespace NUMINAMATH_GPT_remaining_cubes_count_l1887_188714

-- Define the initial number of cubes
def initial_cubes : ℕ := 64

-- Define the holes in the bottom layer
def holes_in_bottom_layer : ℕ := 6

-- Define the number of cubes removed per hole
def cubes_removed_per_hole : ℕ := 3

-- Define the calculation for missing cubes
def missing_cubes : ℕ := holes_in_bottom_layer * cubes_removed_per_hole

-- Define the calculation for remaining cubes
def remaining_cubes : ℕ := initial_cubes - missing_cubes

-- The theorem to prove
theorem remaining_cubes_count : remaining_cubes = 46 := by
  sorry

end NUMINAMATH_GPT_remaining_cubes_count_l1887_188714


namespace NUMINAMATH_GPT_sum_of_sequence_l1887_188720

theorem sum_of_sequence (avg : ℕ → ℕ → ℕ) (n : ℕ) (total_sum : ℕ) 
  (condition : avg 16 272 = 17) : 
  total_sum = 272 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_sequence_l1887_188720


namespace NUMINAMATH_GPT_electric_car_charging_cost_l1887_188700

/-- The fractional equation for the given problem,
    along with the correct solution for the average charging cost per kilometer. -/
theorem electric_car_charging_cost (
    x : ℝ
) : 
    (200 / x = 4 * (200 / (x + 0.6))) → x = 0.2 :=
by
  intros h_eq
  sorry

end NUMINAMATH_GPT_electric_car_charging_cost_l1887_188700


namespace NUMINAMATH_GPT_chess_piece_max_visitable_squares_l1887_188732

-- Define initial board properties and movement constraints
structure ChessBoard :=
  (rows : ℕ)
  (columns : ℕ)
  (movement : ℕ)
  (board_size : rows * columns = 225)

-- Define condition for unique visitation
def can_visit (movement : ℕ) (board_size : ℕ) : Prop :=
  ∃ (max_squares : ℕ), (max_squares ≤ board_size) ∧ (max_squares = 196)

-- Main theorem statement 
theorem chess_piece_max_visitable_squares (cb : ChessBoard) : 
  can_visit 196 225 :=
by sorry

end NUMINAMATH_GPT_chess_piece_max_visitable_squares_l1887_188732


namespace NUMINAMATH_GPT_determine_head_start_l1887_188725

def head_start (v : ℝ) (s : ℝ) : Prop :=
  let a_speed := 2 * v
  let distance := 142
  distance / a_speed = (distance - s) / v

theorem determine_head_start (v : ℝ) : head_start v 71 :=
  by
    sorry

end NUMINAMATH_GPT_determine_head_start_l1887_188725


namespace NUMINAMATH_GPT_sum_inequality_l1887_188765

open Real

theorem sum_inequality (a b c : ℝ) (h : a + b + c = 3) :
  (1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11)) ≤ 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_sum_inequality_l1887_188765


namespace NUMINAMATH_GPT_unique_bisecting_line_exists_l1887_188705

noncomputable def triangle_area := 1 / 2 * 6 * 8
noncomputable def triangle_perimeter := 6 + 8 + 10

theorem unique_bisecting_line_exists :
  ∃ (line : ℝ → ℝ), 
    (∃ x y : ℝ, x + y = 12 ∧ x * y = 30 ∧ 
      1 / 2 * x * y * (24 / triangle_perimeter) = 12) ∧
    (∃ x' y' : ℝ, x' + y' = 12 ∧ x' * y' = 24 ∧ 
      1 / 2 * x' * y' * (24 / triangle_perimeter) = 12) ∧
    ((x = x' ∧ y = y') ∨ (x = y' ∧ y = x')) :=
sorry

end NUMINAMATH_GPT_unique_bisecting_line_exists_l1887_188705


namespace NUMINAMATH_GPT_smallest_positive_integer_l1887_188728
-- Import the required library

-- State the problem in Lean
theorem smallest_positive_integer (x : ℕ) (h : 5 * x ≡ 17 [MOD 31]) : x = 13 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l1887_188728


namespace NUMINAMATH_GPT_rupert_candles_l1887_188701

theorem rupert_candles (peter_candles : ℕ) (rupert_times_older : ℝ) (h1 : peter_candles = 10) (h2 : rupert_times_older = 3.5) :
    ∃ rupert_candles : ℕ, rupert_candles = peter_candles * rupert_times_older := 
by
  sorry

end NUMINAMATH_GPT_rupert_candles_l1887_188701


namespace NUMINAMATH_GPT_intersecting_lines_l1887_188779

theorem intersecting_lines
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h_p1 : p1 = (3, 2))
  (h_line1 : ∀ x : ℝ, p1.2 = 3 * p1.1 + 4)
  (h_line2 : ∀ x : ℝ, p2.2 = - (1 / 3) * p2.1 + 3) :
  (∃ p : ℝ × ℝ, p = (-3 / 10, 31 / 10)) :=
by
  sorry

end NUMINAMATH_GPT_intersecting_lines_l1887_188779


namespace NUMINAMATH_GPT_sector_area_l1887_188796

theorem sector_area (theta : ℝ) (r : ℝ) (h_theta : theta = 2 * π / 3) (h_r : r = 3) : 
    (theta / (2 * π) * π * r^2) = 3 * π :=
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sector_area_l1887_188796


namespace NUMINAMATH_GPT_parallel_vectors_x_value_l1887_188757

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Define what it means for vectors to be parallel (they are proportional)
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem to prove
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = 9 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_parallel_vectors_x_value_l1887_188757


namespace NUMINAMATH_GPT_no_real_roots_l1887_188753

-- Define the coefficients of the quadratic equation
def a : ℝ := 1
def b : ℝ := 2
def c : ℝ := 4

-- Define the quadratic equation
def quadratic_eqn (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the discriminant
def discriminant : ℝ := b^2 - 4 * a * c

-- State the theorem: The quadratic equation has no real roots because the discriminant is negative
theorem no_real_roots : discriminant < 0 := by
  unfold discriminant
  unfold a b c
  sorry

end NUMINAMATH_GPT_no_real_roots_l1887_188753


namespace NUMINAMATH_GPT_problem1_problem2_l1887_188774

theorem problem1 : -1 + (-6) - (-4) + 0 = -3 := by
  sorry

theorem problem2 : 24 * (-1 / 4) / (-3 / 2) = 4 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1887_188774


namespace NUMINAMATH_GPT_smallest_sum_l1887_188792

theorem smallest_sum (a b c : ℕ) (h : (13 * a + 11 * b + 7 * c = 1001)) :
    a / 77 + b / 91 + c / 143 = 1 → a + b + c = 79 :=
by
  sorry

end NUMINAMATH_GPT_smallest_sum_l1887_188792


namespace NUMINAMATH_GPT_thirteen_y_minus_x_l1887_188736

theorem thirteen_y_minus_x (x y : ℤ) (hx1 : x = 11 * y + 4) (hx2 : 2 * x = 8 * (3 * y) + 3) : 13 * y - x = 1 :=
by
  sorry

end NUMINAMATH_GPT_thirteen_y_minus_x_l1887_188736


namespace NUMINAMATH_GPT_radius_increase_by_100_percent_l1887_188759

theorem radius_increase_by_100_percent (A A' r r' : ℝ) (π : ℝ)
  (h1 : A = π * r^2) -- initial area of the circle
  (h2 : A' = 4 * A) -- new area is 4 times the original area
  (h3 : A' = π * r'^2) -- new area formula with new radius
  : r' = 2 * r :=
by
  sorry

end NUMINAMATH_GPT_radius_increase_by_100_percent_l1887_188759


namespace NUMINAMATH_GPT_value_of_expression_l1887_188704

variable (x y : ℝ)

theorem value_of_expression (h1 : x + y = 6) (h2 : x * y = 1) :
  x + (x^4 / y^3) + (y^4 / x^3) + y = 228498 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1887_188704


namespace NUMINAMATH_GPT_triangle_area_tangent_log2_l1887_188783

open Real

noncomputable def log_base_2 (x : ℝ) : ℝ := log x / log 2

theorem triangle_area_tangent_log2 :
  let y := log_base_2
  let f := fun x : ℝ => y x
  let deriv := (deriv f 1)
  let tangent_line := fun x : ℝ => deriv * (x - 1) + f 1
  let x_intercept := 1
  let y_intercept := tangent_line 0
  
  (1 : ℝ) * (abs y_intercept) / 2 = 1 / (2 * log 2) := by
  sorry

end NUMINAMATH_GPT_triangle_area_tangent_log2_l1887_188783


namespace NUMINAMATH_GPT_quadratic_completion_l1887_188787

theorem quadratic_completion (b c : ℝ) (h : (x : ℝ) → x^2 + 1600 * x + 1607 = (x + b)^2 + c) (hb : b = 800) (hc : c = -638393) : 
  c / b = -797.99125 := by
  sorry

end NUMINAMATH_GPT_quadratic_completion_l1887_188787


namespace NUMINAMATH_GPT_circle_eq_of_points_value_of_m_l1887_188751

-- Define the points on the circle
def P : ℝ × ℝ := (0, -4)
def Q : ℝ × ℝ := (2, 0)
def R : ℝ × ℝ := (3, -1)

-- Statement 1: The equation of the circle passing through P, Q, and R
theorem circle_eq_of_points (C : ℝ × ℝ → Prop) :
  (C P ∧ C Q ∧ C R) ↔ ∀ x y : ℝ, C (x, y) ↔ (x - 1)^2 + (y + 2)^2 = 5 := sorry

-- Define the line intersecting the circle and the chord length condition |AB| = 4
def line_l (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

-- Statement 2: The value of m such that the chord length |AB| is 4
theorem value_of_m (m : ℝ) :
  (∃ A B : ℝ × ℝ, line_l m A.1 A.2 ∧ line_l m B.1 B.2 ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 16)) → m = 4 / 3 := sorry

end NUMINAMATH_GPT_circle_eq_of_points_value_of_m_l1887_188751


namespace NUMINAMATH_GPT_additional_charge_per_segment_l1887_188790

theorem additional_charge_per_segment :
  ∀ (initial_fee total_charge distance : ℝ), 
    initial_fee = 2.35 →
    total_charge = 5.5 →
    distance = 3.6 →
    (total_charge - initial_fee) / (distance / (2 / 5)) = 0.35 :=
by
  intros initial_fee total_charge distance h_initial_fee h_total_charge h_distance
  sorry

end NUMINAMATH_GPT_additional_charge_per_segment_l1887_188790


namespace NUMINAMATH_GPT_find_second_offset_l1887_188781

-- Define the given constants
def diagonal : ℝ := 30
def offset1 : ℝ := 10
def area : ℝ := 240

-- The theorem we want to prove
theorem find_second_offset : ∃ (offset2 : ℝ), area = (1 / 2) * diagonal * (offset1 + offset2) ∧ offset2 = 6 :=
sorry

end NUMINAMATH_GPT_find_second_offset_l1887_188781
