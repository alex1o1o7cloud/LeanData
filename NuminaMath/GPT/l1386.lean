import Mathlib

namespace NUMINAMATH_GPT_remainder_when_divided_by_20_l1386_138661

theorem remainder_when_divided_by_20
  (a b : ℤ) 
  (h1 : a % 60 = 49)
  (h2 : b % 40 = 29) :
  (a + b) % 20 = 18 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_20_l1386_138661


namespace NUMINAMATH_GPT_intersection_empty_set_l1386_138650

def M : Set ℝ := { y | ∃ x, x > 0 ∧ y = 2^x }
def N : Set ℝ := { y | ∃ x, y = Real.sqrt (2*x - x^2) }

theorem intersection_empty_set :
  M ∩ N = ∅ :=
by
  sorry

end NUMINAMATH_GPT_intersection_empty_set_l1386_138650


namespace NUMINAMATH_GPT_sum_of_two_numbers_l1386_138675

theorem sum_of_two_numbers (L S : ℕ) (hL : L = 22) (hExceeds : L = S + 10) : L + S = 34 := by
  sorry

end NUMINAMATH_GPT_sum_of_two_numbers_l1386_138675


namespace NUMINAMATH_GPT_shortest_distance_D_to_V_l1386_138684

-- Define distances
def distance_A_to_G : ℕ := 12
def distance_G_to_B : ℕ := 10
def distance_A_to_B : ℕ := 8
def distance_D_to_G : ℕ := 15
def distance_V_to_G : ℕ := 17

-- Prove the shortest distance from Dasha to Vasya
theorem shortest_distance_D_to_V : 
  let dD_to_V := distance_D_to_G + distance_V_to_G
  let dAlt := dD_to_V + distance_A_to_B - distance_A_to_G - distance_G_to_B
  (dAlt < dD_to_V) -> dAlt = 18 :=
by
  sorry

end NUMINAMATH_GPT_shortest_distance_D_to_V_l1386_138684


namespace NUMINAMATH_GPT_oula_deliveries_count_l1386_138602

-- Define the conditions for the problem
def num_deliveries_Oula (O : ℕ) (T : ℕ) : Prop :=
  T = (3 / 4 : ℚ) * O ∧ (100 * O - 100 * T = 2400)

-- Define the theorem we want to prove
theorem oula_deliveries_count : ∃ (O : ℕ), ∃ (T : ℕ), num_deliveries_Oula O T ∧ O = 96 :=
sorry

end NUMINAMATH_GPT_oula_deliveries_count_l1386_138602


namespace NUMINAMATH_GPT_cube_surface_area_including_inside_l1386_138613

theorem cube_surface_area_including_inside 
  (original_edge_length : ℝ) 
  (hole_side_length : ℝ) 
  (original_cube_surface_area : ℝ)
  (removed_hole_area : ℝ)
  (newly_exposed_internal_area : ℝ) 
  (total_surface_area : ℝ) 
  (h1 : original_edge_length = 3)
  (h2 : hole_side_length = 1)
  (h3 : original_cube_surface_area = 6 * (original_edge_length * original_edge_length))
  (h4 : removed_hole_area = 6 * (hole_side_length * hole_side_length))
  (h5 : newly_exposed_internal_area = 6 * 4 * (hole_side_length * hole_side_length))
  (h6 : total_surface_area = original_cube_surface_area - removed_hole_area + newly_exposed_internal_area) : 
  total_surface_area = 72 :=
by
  sorry

end NUMINAMATH_GPT_cube_surface_area_including_inside_l1386_138613


namespace NUMINAMATH_GPT_a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l1386_138638

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end NUMINAMATH_GPT_a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l1386_138638


namespace NUMINAMATH_GPT_original_price_of_book_l1386_138660

theorem original_price_of_book (final_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) 
  (h1 : final_price = 360) (h2 : increase_percentage = 0.20) 
  (h3 : final_price = (1 + increase_percentage) * original_price) : original_price = 300 := 
by
  sorry

end NUMINAMATH_GPT_original_price_of_book_l1386_138660


namespace NUMINAMATH_GPT_triangle_area_proof_l1386_138601

noncomputable def area_of_triangle (a b c : ℝ) (A B C : ℝ) : ℝ := 
  1 / 2 * a * c * Real.sin B

theorem triangle_area_proof (a b c : ℝ) (A B C : ℝ) (h1 : b = 3) 
  (h2 : Real.cos B = 1 / 4) 
  (h3 : Real.sin C = 2 * Real.sin A) 
  (h4 : c = 2 * a) 
  (h5 : 9 = 5 * a ^ 2 - 4 * a ^ 2 * Real.cos B): 
  area_of_triangle a b c A B C = 9 * Real.sqrt 15 / 16 :=
by 
  sorry

end NUMINAMATH_GPT_triangle_area_proof_l1386_138601


namespace NUMINAMATH_GPT_factorize_2070_l1386_138620

-- Define the conditions
def is_two_digit (n : ℕ) : Prop := n ≥ 10 ∧ n < 100
def is_unique_factorization (n a b : ℕ) : Prop := a * b = n ∧ is_two_digit a ∧ is_two_digit b

-- The final theorem statement
theorem factorize_2070 : 
  (∃ a b : ℕ, is_unique_factorization 2070 a b) ∧ 
  ∀ a b : ℕ, is_unique_factorization 2070 a b → (a = 30 ∧ b = 69) ∨ (a = 69 ∧ b = 30) :=
by 
  sorry

end NUMINAMATH_GPT_factorize_2070_l1386_138620


namespace NUMINAMATH_GPT_orchids_cut_l1386_138665

-- Define initial and final number of orchids in the vase
def initialOrchids : ℕ := 2
def finalOrchids : ℕ := 21

-- Formulate the claim to prove the number of orchids Jessica cut
theorem orchids_cut : finalOrchids - initialOrchids = 19 := by
  sorry

end NUMINAMATH_GPT_orchids_cut_l1386_138665


namespace NUMINAMATH_GPT_maxwells_walking_speed_l1386_138648

theorem maxwells_walking_speed 
    (brad_speed : ℕ) 
    (distance_between_homes : ℕ) 
    (maxwell_distance : ℕ)
    (meeting : maxwell_distance = 12)
    (brad_speed_condition : brad_speed = 6)
    (distance_between_homes_condition: distance_between_homes = 36) : 
    (maxwell_distance / (distance_between_homes - maxwell_distance) * brad_speed ) = 3 := by
  sorry

end NUMINAMATH_GPT_maxwells_walking_speed_l1386_138648


namespace NUMINAMATH_GPT_sum_of_reciprocals_l1386_138686

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 8 * x * y) : 
  (1 / x) + (1 / y) = 8 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l1386_138686


namespace NUMINAMATH_GPT_correct_equation_l1386_138611

variable (x : ℕ)

def three_people_per_cart_and_two_empty_carts (x : ℕ) :=
  x / 3 + 2

def two_people_per_cart_and_nine_walking (x : ℕ) :=
  (x - 9) / 2

theorem correct_equation (x : ℕ) :
  three_people_per_cart_and_two_empty_carts x = two_people_per_cart_and_nine_walking x :=
by
  sorry

end NUMINAMATH_GPT_correct_equation_l1386_138611


namespace NUMINAMATH_GPT_real_values_of_x_l1386_138656

theorem real_values_of_x :
  {x : ℝ | (∃ y, y = (x^2 + 2 * x^3 - 3 * x^4) / (x + 2 * x^2 - 3 * x^3) ∧ y ≥ -1)} =
  {x | -1 ≤ x ∧ x < -1/3 ∨ -1/3 < x ∧ x < 0 ∨ 0 < x ∧ x < 1 ∨ 1 < x} := 
sorry

end NUMINAMATH_GPT_real_values_of_x_l1386_138656


namespace NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_part3_l1386_138673

variable (m : ℝ)
def f (x : ℝ) := (m + 1) * x^2 - (m - 1) * x + (m - 1)

-- Part (1)
theorem part1 (h : ∀ x : ℝ, f m x < 1) : m < (1 - 2 * Real.sqrt 7) / 3 :=
sorry

-- Part (2)
theorem part2_case1 (h : m = -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≥ 1 :=
sorry

theorem part2_case2 (h : m > -1) : ∀ x, f m x ≥ (m + 1) * x ↔ x ≤ (m - 1) / (m + 1) ∨ x ≥ 1 :=
sorry

theorem part2_case3 (h : m < -1) : ∀ x, f m x ≥ (m + 1) * x ↔ 1 ≤ x ∧ x ≤ (m - 1) / (m + 1) :=
sorry

-- Part (3)
theorem part3 (h : ∀ x ∈ Set.Icc (-1/2 : ℝ) (1/2), f m x ≥ 0) : m ≥ 1 :=
sorry

end NUMINAMATH_GPT_part1_part2_case1_part2_case2_part2_case3_part3_l1386_138673


namespace NUMINAMATH_GPT_shekar_average_marks_l1386_138623

-- Define the scores for each subject
def mathematics := 76
def science := 65
def social_studies := 82
def english := 67
def biology := 55
def computer_science := 89
def history := 74
def geography := 63
def physics := 78
def chemistry := 71

-- Define the total number of subjects
def number_of_subjects := 10

-- State the theorem to prove the average marks
theorem shekar_average_marks :
  (mathematics + science + social_studies + english + biology +
   computer_science + history + geography + physics + chemistry) 
   / number_of_subjects = 72 := 
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_shekar_average_marks_l1386_138623


namespace NUMINAMATH_GPT_product_of_solutions_l1386_138622

theorem product_of_solutions (x : ℝ) :
  let a := -2
  let b := -8
  let c := -49
  ∀ x₁ x₂, (a * x₁^2 + b * x₁ + c = 0) ∧ (a * x₂^2 + b * x₂ + c = 0) → 
  x₁ * x₂ = 49/2 :=
sorry

end NUMINAMATH_GPT_product_of_solutions_l1386_138622


namespace NUMINAMATH_GPT_sum_of_roots_zero_l1386_138618

theorem sum_of_roots_zero (p q : ℝ) (h1 : p = -q) (h2 : ∀ x, x^2 + p * x + q = 0) : p + q = 0 := 
by {
  sorry 
}

end NUMINAMATH_GPT_sum_of_roots_zero_l1386_138618


namespace NUMINAMATH_GPT_gcd_lcm_of_300_105_l1386_138614

theorem gcd_lcm_of_300_105 :
  ∃ g l : ℕ, g = Int.gcd 300 105 ∧ l = Nat.lcm 300 105 ∧ g = 15 ∧ l = 2100 :=
by
  let g := Int.gcd 300 105
  let l := Nat.lcm 300 105
  have g_def : g = 15 := sorry
  have l_def : l = 2100 := sorry
  exact ⟨g, l, ⟨g_def, ⟨l_def, ⟨g_def, l_def⟩⟩⟩⟩

end NUMINAMATH_GPT_gcd_lcm_of_300_105_l1386_138614


namespace NUMINAMATH_GPT_eve_discovers_secret_l1386_138697

theorem eve_discovers_secret (x : ℕ) : ∃ (n : ℕ), ∃ (is_prime : ℕ → Prop), (∀ m : ℕ, (is_prime (x + n * m)) ∨ (¬is_prime (x + n * m))) :=
  sorry

end NUMINAMATH_GPT_eve_discovers_secret_l1386_138697


namespace NUMINAMATH_GPT_geometric_sequence_a1_value_l1386_138681

variable {a_1 q : ℝ}

theorem geometric_sequence_a1_value
  (h1 : a_1 * q^2 = 1)
  (h2 : a_1 * q^4 + (3 / 2) * a_1 * q^3 = 1) :
  a_1 = 4 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a1_value_l1386_138681


namespace NUMINAMATH_GPT_matt_peanut_revenue_l1386_138674

theorem matt_peanut_revenue
    (plantation_length : ℕ)
    (plantation_width : ℕ)
    (peanut_production : ℕ)
    (peanut_to_peanut_butter_rate_peanuts : ℕ)
    (peanut_to_peanut_butter_rate_butter : ℕ)
    (peanut_butter_price_per_kg : ℕ)
    (expected_revenue : ℕ) :
    plantation_length = 500 →
    plantation_width = 500 →
    peanut_production = 50 →
    peanut_to_peanut_butter_rate_peanuts = 20 →
    peanut_to_peanut_butter_rate_butter = 5 →
    peanut_butter_price_per_kg = 10 →
    expected_revenue = 31250 :=
by
  sorry

end NUMINAMATH_GPT_matt_peanut_revenue_l1386_138674


namespace NUMINAMATH_GPT_maisie_flyers_count_l1386_138680

theorem maisie_flyers_count (M : ℕ) (h1 : 71 = 2 * M + 5) : M = 33 :=
by
  sorry

end NUMINAMATH_GPT_maisie_flyers_count_l1386_138680


namespace NUMINAMATH_GPT_polynomial_factorization_l1386_138615

-- Definitions from conditions
def p (x : ℝ) : ℝ := x^6 - 2 * x^4 + 6 * x^3 + x^2 - 6 * x + 9
def q (x : ℝ) : ℝ := (x^3 - x + 3)^2

-- The theorem statement proving question == answer given conditions
theorem polynomial_factorization : ∀ x : ℝ, p x = q x :=
by
  sorry

end NUMINAMATH_GPT_polynomial_factorization_l1386_138615


namespace NUMINAMATH_GPT_cost_per_text_message_for_first_plan_l1386_138688

theorem cost_per_text_message_for_first_plan (x : ℝ) : 
  (9 + 60 * x = 60 * 0.40) → (x = 0.25) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_per_text_message_for_first_plan_l1386_138688


namespace NUMINAMATH_GPT_at_least_one_is_one_l1386_138664

theorem at_least_one_is_one (a b c : ℝ) 
  (h1 : a + b + c = (1 / a) + (1 / b) + (1 / c)) 
  (h2 : a * b * c = 1) : a = 1 ∨ b = 1 ∨ c = 1 := 
by 
  sorry

end NUMINAMATH_GPT_at_least_one_is_one_l1386_138664


namespace NUMINAMATH_GPT_fraction_equality_l1386_138627

theorem fraction_equality : 
  (3 ^ 8 + 3 ^ 6) / (3 ^ 8 - 3 ^ 6) = 5 / 4 :=
by
  -- Expression rewrite and manipulation inside parenthesis can be ommited
  sorry

end NUMINAMATH_GPT_fraction_equality_l1386_138627


namespace NUMINAMATH_GPT_tropical_island_parrots_l1386_138679

theorem tropical_island_parrots :
  let total_parrots := 150
  let red_fraction := 4 / 5
  let yellow_fraction := 1 - red_fraction
  let yellow_parrots := yellow_fraction * total_parrots
  yellow_parrots = 30 := sorry

end NUMINAMATH_GPT_tropical_island_parrots_l1386_138679


namespace NUMINAMATH_GPT_ef_length_l1386_138668

theorem ef_length (FR RG : ℝ) (cos_ERH : ℝ) (h1 : FR = 12) (h2 : RG = 6) (h3 : cos_ERH = 1 / 5) : EF = 30 :=
by
  sorry

end NUMINAMATH_GPT_ef_length_l1386_138668


namespace NUMINAMATH_GPT_functional_relationship_minimum_wage_l1386_138652

/-- Problem setup and conditions --/
def total_area : ℝ := 1200
def team_A_rate : ℝ := 100
def team_B_rate : ℝ := 50
def team_A_wage : ℝ := 4000
def team_B_wage : ℝ := 3000
def min_days_A : ℝ := 3

/-- Prove Part 1: y as a function of x --/
def y_of_x (x : ℝ) : ℝ := 24 - 2 * x

theorem functional_relationship (x : ℝ) :
  100 * x + 50 * y_of_x x = total_area := by
  sorry

/-- Prove Part 2: Minimum wage calculation --/
def total_wage (a b : ℝ) : ℝ := team_A_wage * a + team_B_wage * b

theorem minimum_wage :
  ∀ (a b : ℝ), 3 ≤ a → a ≤ b → b = 24 - 2 * a → 
  total_wage a b = 56000 → a = 8 ∧ b = 8 := by
  sorry

end NUMINAMATH_GPT_functional_relationship_minimum_wage_l1386_138652


namespace NUMINAMATH_GPT_initial_bowls_eq_70_l1386_138606

def customers : ℕ := 20
def bowls_per_customer : ℕ := 20
def reward_ratio := 10
def reward_bowls := 2
def remaining_bowls : ℕ := 30

theorem initial_bowls_eq_70 :
  let rewards_per_customer := (bowls_per_customer / reward_ratio) * reward_bowls
  let total_rewards := (customers / 2) * rewards_per_customer
  (remaining_bowls + total_rewards) = 70 :=
by
  sorry

end NUMINAMATH_GPT_initial_bowls_eq_70_l1386_138606


namespace NUMINAMATH_GPT_roots_abs_gt_4_or_l1386_138658

theorem roots_abs_gt_4_or
    (r1 r2 : ℝ)
    (q : ℝ) 
    (h1 : r1 ≠ r2)
    (h2 : r1 + r2 = -q)
    (h3 : r1 * r2 = -10) :
    |r1| > 4 ∨ |r2| > 4 :=
sorry

end NUMINAMATH_GPT_roots_abs_gt_4_or_l1386_138658


namespace NUMINAMATH_GPT_max_non_attacking_rooks_l1386_138647

theorem max_non_attacking_rooks (a b : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 299) (h3 : 1 ≤ b) (h4 : b ≤ 299) :
  ∃ max_rooks : ℕ, max_rooks = 400 :=
  sorry

end NUMINAMATH_GPT_max_non_attacking_rooks_l1386_138647


namespace NUMINAMATH_GPT_find_alpha_beta_sum_l1386_138626

theorem find_alpha_beta_sum
  (a : ℝ) (α β φ : ℝ)
  (h1 : 3 * Real.sin α + 4 * Real.cos α = a)
  (h2 : 3 * Real.sin β + 4 * Real.cos β = a)
  (h3 : α ≠ β)
  (h4 : 0 < α ∧ α < 2 * Real.pi)
  (h5 : 0 < β ∧ β < 2 * Real.pi)
  (hφ : φ = Real.arcsin (4/5)) :
  α + β = Real.pi - 2 * φ ∨ α + β = 3 * Real.pi - 2 * φ :=
by
  sorry

end NUMINAMATH_GPT_find_alpha_beta_sum_l1386_138626


namespace NUMINAMATH_GPT_platform_length_is_correct_l1386_138640

-- Given Definitions
def length_of_train : ℝ := 300
def time_to_cross_platform : ℝ := 42
def time_to_cross_pole : ℝ := 18

-- Definition to prove
theorem platform_length_is_correct :
  ∃ L : ℝ, L = 400 ∧ (length_of_train + L) / time_to_cross_platform = length_of_train / time_to_cross_pole :=
by
  sorry

end NUMINAMATH_GPT_platform_length_is_correct_l1386_138640


namespace NUMINAMATH_GPT_bus_A_speed_l1386_138610

-- Define the conditions
variables (v_A v_B : ℝ)
axiom equation1 : v_A - v_B = 15
axiom equation2 : v_A + v_B = 75

-- The main theorem we want to prove
theorem bus_A_speed : v_A = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_bus_A_speed_l1386_138610


namespace NUMINAMATH_GPT_tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l1386_138677

-- Condition: Given tan(α) = 2
variable (α : ℝ) (h₀ : Real.tan α = 2)

-- Statement (1): Prove tan(2α + π/4) = 9
theorem tan_double_alpha_plus_pi_over_four :
  Real.tan (2 * α + Real.pi / 4) = 9 := by
  sorry

-- Statement (2): Prove (6 sin α + cos α) / (3 sin α - 2 cos α) = 13 / 4
theorem sin_cos_fraction :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 := by
  sorry

end NUMINAMATH_GPT_tan_double_alpha_plus_pi_over_four_sin_cos_fraction_l1386_138677


namespace NUMINAMATH_GPT_exist_integers_xy_divisible_by_p_l1386_138632

theorem exist_integers_xy_divisible_by_p (p : ℕ) [Fact (Nat.Prime p)] : ∃ x y : ℤ, (x^2 + y^2 + 2) % p = 0 := by
  sorry

end NUMINAMATH_GPT_exist_integers_xy_divisible_by_p_l1386_138632


namespace NUMINAMATH_GPT_count_perfect_squares_lt_10_pow_9_multiple_36_l1386_138662

theorem count_perfect_squares_lt_10_pow_9_multiple_36 : 
  ∃ N : ℕ, ∀ n < 31622, (n % 6 = 0 → n^2 < 10^9 ∧ 36 ∣ n^2 → n ≤ 31620 → N = 5270) :=
by
  sorry

end NUMINAMATH_GPT_count_perfect_squares_lt_10_pow_9_multiple_36_l1386_138662


namespace NUMINAMATH_GPT_arc_length_of_sector_l1386_138628

theorem arc_length_of_sector (r α : ℝ) (hα : α = Real.pi / 5) (hr : r = 20) : r * α = 4 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_arc_length_of_sector_l1386_138628


namespace NUMINAMATH_GPT_smallest_a_divisible_by_65_l1386_138644

theorem smallest_a_divisible_by_65 (a : ℤ) 
  (h : ∀ (n : ℤ), (5 * n ^ 13 + 13 * n ^ 5 + 9 * a * n) % 65 = 0) : 
  a = 63 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_a_divisible_by_65_l1386_138644


namespace NUMINAMATH_GPT_initial_quantity_of_A_l1386_138666

noncomputable def initial_quantity_of_A_in_can (initial_total_mixture : ℤ) (x : ℤ) := 7 * x

theorem initial_quantity_of_A
  (initial_ratio_A : ℤ) (initial_ratio_B : ℤ) (initial_ratio_C : ℤ)
  (initial_total_mixture : ℤ) (drawn_off_mixture : ℤ) (new_quantity_of_B : ℤ)
  (new_ratio_A : ℤ) (new_ratio_B : ℤ) (new_ratio_C : ℤ)
  (h1 : initial_ratio_A = 7) (h2 : initial_ratio_B = 5) (h3 : initial_ratio_C = 3)
  (h4 : initial_total_mixture = 15 * x)
  (h5 : new_ratio_A = 7) (h6 : new_ratio_B = 9) (h7 : new_ratio_C = 3)
  (h8 : drawn_off_mixture = 18)
  (h9 : new_quantity_of_B = 5 * x - (5 / 15) * 18 + 18)
  (h10 : (7 * x - (7 / 15) * 18) / new_quantity_of_B = 7 / 9) :
  initial_quantity_of_A_in_can initial_total_mixture x = 54 :=
by
  sorry

end NUMINAMATH_GPT_initial_quantity_of_A_l1386_138666


namespace NUMINAMATH_GPT_neg_p_is_exists_x_l1386_138654

variable (x : ℝ)

def p : Prop := ∀ x, x^2 + x + 1 ≠ 0

theorem neg_p_is_exists_x : ¬ p ↔ ∃ x, x^2 + x + 1 = 0 := by
  sorry

end NUMINAMATH_GPT_neg_p_is_exists_x_l1386_138654


namespace NUMINAMATH_GPT_area_of_triangle_PQR_l1386_138676

-- Define the problem conditions
def PQ : ℝ := 4
def PR : ℝ := 4
def angle_P : ℝ := 45 -- degrees

-- Define the main problem
theorem area_of_triangle_PQR : 
  (PQ = PR) ∧ (angle_P = 45) ∧ (PR = 4) → 
  ∃ A, A = 8 := 
by
  sorry

end NUMINAMATH_GPT_area_of_triangle_PQR_l1386_138676


namespace NUMINAMATH_GPT_find_constant_c_l1386_138624

theorem find_constant_c (c : ℝ) (h : (x + 7) ∣ (c*x^3 + 19*x^2 - 3*c*x + 35)) : c = 3 := by
  sorry

end NUMINAMATH_GPT_find_constant_c_l1386_138624


namespace NUMINAMATH_GPT_number_of_maple_trees_planted_today_l1386_138690

-- Define the initial conditions
def initial_maple_trees : ℕ := 2
def poplar_trees : ℕ := 5
def final_maple_trees : ℕ := 11

-- State the main proposition
theorem number_of_maple_trees_planted_today : 
  (final_maple_trees - initial_maple_trees) = 9 := by
  sorry

end NUMINAMATH_GPT_number_of_maple_trees_planted_today_l1386_138690


namespace NUMINAMATH_GPT_expected_value_m_plus_n_l1386_138663

-- Define the main structures and conditions
def spinner_sectors : List ℚ := [-1.25, -1, 0, 1, 1.25]
def initial_value : ℚ := 1

-- Define a function that returns the largest expected value on the paper
noncomputable def expected_largest_written_value (sectors : List ℚ) (initial : ℚ) : ℚ :=
  -- The expected value calculation based on the problem and solution analysis
  11/6  -- This is derived from the correct solution steps not shown here

-- Define the final claim
theorem expected_value_m_plus_n :
  let m := 11
  let n := 6
  expected_largest_written_value spinner_sectors initial_value = 11/6 → m + n = 17 :=
by sorry

end NUMINAMATH_GPT_expected_value_m_plus_n_l1386_138663


namespace NUMINAMATH_GPT_marble_count_l1386_138693

theorem marble_count (r g b : ℝ) (h1 : g + b = 9) (h2 : r + b = 7) (h3 : r + g = 5) :
  r + g + b = 10.5 :=
by sorry

end NUMINAMATH_GPT_marble_count_l1386_138693


namespace NUMINAMATH_GPT_symmetric_point_origin_l1386_138685

def symmetric_point (p: ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

theorem symmetric_point_origin : 
  (symmetric_point (3, -2)) = (-3, 2) :=
by
  sorry

end NUMINAMATH_GPT_symmetric_point_origin_l1386_138685


namespace NUMINAMATH_GPT_john_memory_card_cost_l1386_138625

-- Define conditions
def pictures_per_day : ℕ := 10
def days_per_year : ℕ := 365
def years : ℕ := 3
def pictures_per_card : ℕ := 50
def cost_per_card : ℕ := 60

-- Define total days
def total_days (years : ℕ) (days_per_year : ℕ) : ℕ := years * days_per_year

-- Define total pictures
def total_pictures (pictures_per_day : ℕ) (total_days : ℕ) : ℕ := pictures_per_day * total_days

-- Define required cards
def required_cards (total_pictures : ℕ) (pictures_per_card : ℕ) : ℕ :=
  (total_pictures + pictures_per_card - 1) / pictures_per_card  -- ceiling division

-- Define total cost
def total_cost (required_cards : ℕ) (cost_per_card : ℕ) : ℕ := required_cards * cost_per_card

-- Prove the total cost equals $13,140
theorem john_memory_card_cost : total_cost (required_cards (total_pictures pictures_per_day (total_days years days_per_year)) pictures_per_card) cost_per_card = 13140 :=
by
  sorry

end NUMINAMATH_GPT_john_memory_card_cost_l1386_138625


namespace NUMINAMATH_GPT_paper_area_l1386_138689

theorem paper_area (L W : ℝ) 
(h1 : 2 * L + W = 34) 
(h2 : L + 2 * W = 38) : 
L * W = 140 := by
  sorry

end NUMINAMATH_GPT_paper_area_l1386_138689


namespace NUMINAMATH_GPT_find_x_l1386_138636

theorem find_x (x : ℕ) : (4 + x) / (7 + x) = 3 / 4 → x = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l1386_138636


namespace NUMINAMATH_GPT_solve_system_l1386_138657

theorem solve_system :
  ∀ (x y : ℝ) (triangle : ℝ), 
  (2 * x - 3 * y = 5) ∧ (x + y = triangle) ∧ (x = 4) →
  (y = 1) ∧ (triangle = 5) :=
by
  -- Skipping the proof steps
  sorry

end NUMINAMATH_GPT_solve_system_l1386_138657


namespace NUMINAMATH_GPT_cyclic_permutations_sum_41234_l1386_138605

theorem cyclic_permutations_sum_41234 :
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  3 * (n1 + n2 + n3 + n4) = 396618 :=
by
  let n1 := 41234
  let n2 := 34124
  let n3 := 23414
  let n4 := 12434
  show 3 * (n1 + n2 + n3 + n4) = 396618
  sorry

end NUMINAMATH_GPT_cyclic_permutations_sum_41234_l1386_138605


namespace NUMINAMATH_GPT_andy_loss_more_likely_than_win_l1386_138669

def prob_win_first := 0.30
def prob_lose_first := 0.70

def prob_win_second := 0.50
def prob_lose_second := 0.50

def prob_win_both := prob_win_first * prob_win_second
def prob_lose_both := prob_lose_first * prob_lose_second
def diff_probability := prob_lose_both - prob_win_both
def percentage_more_likely := (diff_probability / prob_win_both) * 100

theorem andy_loss_more_likely_than_win :
  percentage_more_likely = 133.33 := sorry

end NUMINAMATH_GPT_andy_loss_more_likely_than_win_l1386_138669


namespace NUMINAMATH_GPT_marble_total_weight_l1386_138649

theorem marble_total_weight :
  0.3333333333333333 + 0.3333333333333333 + 0.08333333333333333 + 0.21666666666666667 + 0.4583333333333333 + 0.12777777777777778 = 1.5527777777777777 :=
by
  sorry

end NUMINAMATH_GPT_marble_total_weight_l1386_138649


namespace NUMINAMATH_GPT_triangle_properties_l1386_138683

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (BD : ℝ) (D : ℝ) : 
  (a + c) * Real.sin A = Real.sin A + Real.sin C →
  c^2 + c = b^2 - 1 →
  D = (a + c) / 2 →
  BD = Real.sqrt 3 / 2 →
  B = 2 * Real.pi / 3 ∧ (1 / 2) * a * c * Real.sin B = Real.sqrt 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_properties_l1386_138683


namespace NUMINAMATH_GPT_least_square_of_conditions_l1386_138698

theorem least_square_of_conditions :
  ∃ (a x y : ℕ), 0 < a ∧ 0 < x ∧ 0 < y ∧ 
  (15 * a + 165 = x^2) ∧ 
  (16 * a - 155 = y^2) ∧ 
  (min (x^2) (y^2) = 481) := 
sorry

end NUMINAMATH_GPT_least_square_of_conditions_l1386_138698


namespace NUMINAMATH_GPT_problem_solution_l1386_138667

theorem problem_solution
  (m : ℝ) (n : ℝ)
  (h1 : m = 1 / (Real.sqrt 3 + Real.sqrt 2))
  (h2 : n = 1 / (Real.sqrt 3 - Real.sqrt 2)) :
  (m - 1) * (n - 1) = -2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_problem_solution_l1386_138667


namespace NUMINAMATH_GPT_factor_expression_l1386_138616

theorem factor_expression (x : ℝ) :
  4 * x * (x - 5) + 7 * (x - 5) + 12 * (x - 5) = (4 * x + 19) * (x - 5) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1386_138616


namespace NUMINAMATH_GPT_linear_term_coefficient_l1386_138635

-- Define the given equation
def equation (x : ℝ) : ℝ := x^2 - 2022*x - 2023

-- The goal is to prove that the coefficient of the linear term in equation is -2022
theorem linear_term_coefficient : ∀ x : ℝ, equation x = x^2 - 2022*x - 2023 → -2022 = -2022 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_linear_term_coefficient_l1386_138635


namespace NUMINAMATH_GPT_area_intersection_A_B_l1386_138670

noncomputable def A : Set (Real × Real) := {
  p | ∃ α β : ℝ, p.1 = 2 * Real.sin α + 2 * Real.sin β ∧ p.2 = 2 * Real.cos α + 2 * Real.cos β
}

noncomputable def B : Set (Real × Real) := {
  p | Real.sin (p.1 + p.2) * Real.cos (p.1 + p.2) ≥ 0
}

theorem area_intersection_A_B :
  let intersection := Set.inter A B
  let area : ℝ := 8 * Real.pi
  ∀ (x y : ℝ), (x, y) ∈ intersection → True := sorry

end NUMINAMATH_GPT_area_intersection_A_B_l1386_138670


namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1386_138691

theorem arithmetic_seq_sum (a : ℕ → ℝ) (d : ℝ) 
  (h₁ : ∀ n : ℕ, a (n + 1) = a n + d)
  (h₂ : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1386_138691


namespace NUMINAMATH_GPT_vector_sum_magnitude_l1386_138603

variable (a b : EuclideanSpace ℝ (Fin 3)) -- assuming 3-dimensional Euclidean space for vectors

-- Define the conditions
def mag_a : ℝ := 5
def mag_b : ℝ := 6
def dot_prod_ab : ℝ := -6

-- Prove the required magnitude condition
theorem vector_sum_magnitude (ha : ‖a‖ = mag_a) (hb : ‖b‖ = mag_b) (hab : inner a b = dot_prod_ab) :
  ‖a + b‖ = 7 :=
by
  sorry

end NUMINAMATH_GPT_vector_sum_magnitude_l1386_138603


namespace NUMINAMATH_GPT_count_squares_with_dot_l1386_138699

theorem count_squares_with_dot (n : ℕ) (dot_center : (n = 5)) :
  n = 5 → ∃ k, k = 19 :=
by sorry

end NUMINAMATH_GPT_count_squares_with_dot_l1386_138699


namespace NUMINAMATH_GPT_license_plate_difference_l1386_138653

theorem license_plate_difference :
  (26^4 * 10^3 - 26^5 * 10^2 = -731161600) :=
sorry

end NUMINAMATH_GPT_license_plate_difference_l1386_138653


namespace NUMINAMATH_GPT_x_plus_inv_x_eq_8_then_power_4_l1386_138607

theorem x_plus_inv_x_eq_8_then_power_4 (x : ℝ) (h : x + 1/x = 8) : x^4 + 1/x^4 = 3842 :=
sorry

end NUMINAMATH_GPT_x_plus_inv_x_eq_8_then_power_4_l1386_138607


namespace NUMINAMATH_GPT_radius_of_inscribed_circle_in_quarter_circle_l1386_138694

noncomputable def inscribed_circle_radius (R : ℝ) : ℝ :=
  R * (Real.sqrt 2 - 1)

theorem radius_of_inscribed_circle_in_quarter_circle 
  (R : ℝ) (hR : R = 6) : inscribed_circle_radius R = 6 * Real.sqrt 2 - 6 :=
by
  rw [inscribed_circle_radius, hR]
  -- Apply the necessary simplifications and manipulations from the given solution steps here
  sorry

end NUMINAMATH_GPT_radius_of_inscribed_circle_in_quarter_circle_l1386_138694


namespace NUMINAMATH_GPT_triangle_OAB_area_range_l1386_138643

noncomputable def area_of_triangle_OAB (m : ℝ) : ℝ :=
  4 * Real.sqrt (64 * m^2 + 4 * 64)

theorem triangle_OAB_area_range :
  ∀ m : ℝ, 64 ≤ area_of_triangle_OAB m :=
by
  intro m
  sorry

end NUMINAMATH_GPT_triangle_OAB_area_range_l1386_138643


namespace NUMINAMATH_GPT_geometric_sequence_fourth_term_l1386_138682

/-- In a geometric sequence with common ratio 2, where the sequence is denoted as {a_n},
and it is given that a_1 * a_3 = 6 * a_2, prove that a_4 = 24. -/
theorem geometric_sequence_fourth_term (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = 2 * a n)
  (h1 : a 1 * a 3 = 6 * a 2) : a 4 = 24 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_fourth_term_l1386_138682


namespace NUMINAMATH_GPT_rainfall_comparison_l1386_138655

-- Define the conditions
def rainfall_mondays (n_mondays : ℕ) (rain_monday : ℝ) : ℝ :=
  n_mondays * rain_monday

def rainfall_tuesdays (n_tuesdays : ℕ) (rain_tuesday : ℝ) : ℝ :=
  n_tuesdays * rain_tuesday

def rainfall_difference (total_monday : ℝ) (total_tuesday : ℝ) : ℝ :=
  total_tuesday - total_monday

-- The proof statement
theorem rainfall_comparison :
  rainfall_difference (rainfall_mondays 13 1.75) (rainfall_tuesdays 16 2.65) = 19.65 := by
  sorry

end NUMINAMATH_GPT_rainfall_comparison_l1386_138655


namespace NUMINAMATH_GPT_simplify_expression_l1386_138633

theorem simplify_expression :
  ( ∀ (a b c : ℕ), c > 0 ∧ (∀ p : ℕ, Prime p → ¬ p^2 ∣ c) →
  (a - b * Real.sqrt c = (28 - 16 * Real.sqrt 3) * 2 ^ (-2 - Real.sqrt 5))) :=
sorry

end NUMINAMATH_GPT_simplify_expression_l1386_138633


namespace NUMINAMATH_GPT_solution_correct_l1386_138692

noncomputable def solve_system (a b c : ℝ) : ℝ × ℝ × ℝ :=
  let x := (3 * c - a - b) / 4
  let y := (3 * b - a - c) / 4
  let z := (3 * a - b - c) / 4
  (x, y, z)

theorem solution_correct (a b c : ℝ) (x y z : ℝ) :
  (x + y + 2 * z = a) →
  (x + 2 * y + z = b) →
  (2 * x + y + z = c) →
  (x, y, z) = solve_system a b c :=
by sorry

end NUMINAMATH_GPT_solution_correct_l1386_138692


namespace NUMINAMATH_GPT_area_of_field_l1386_138645

theorem area_of_field (b l : ℝ) (h1 : l = b + 30) (h2 : 2 * (l + b) = 540) : l * b = 18000 := 
by
  sorry

end NUMINAMATH_GPT_area_of_field_l1386_138645


namespace NUMINAMATH_GPT_find_ordered_pair_l1386_138671

-- Definitions based on the conditions
variable (a c : ℝ)
def has_exactly_one_solution :=
  (-6)^2 - 4 * a * c = 0

def sum_is_twelve :=
  a + c = 12

def a_less_than_c :=
  a < c

-- The proof statement
theorem find_ordered_pair
  (h₁ : has_exactly_one_solution a c)
  (h₂ : sum_is_twelve a c)
  (h₃ : a_less_than_c a c) :
  a = 3 ∧ c = 9 := 
sorry

end NUMINAMATH_GPT_find_ordered_pair_l1386_138671


namespace NUMINAMATH_GPT_initial_mixture_l1386_138659

theorem initial_mixture (M : ℝ) (h1 : 0.20 * M + 20 = 0.36 * (M + 20)) : 
  M = 80 :=
by
  sorry

end NUMINAMATH_GPT_initial_mixture_l1386_138659


namespace NUMINAMATH_GPT_gretchen_work_hours_l1386_138639

noncomputable def walking_ratio (walking: ℤ) (sitting: ℤ) : Prop :=
  walking * 90 = sitting * 10

theorem gretchen_work_hours (walking_time: ℤ) (h: ℤ) (condition1: walking_ratio 40 (60 * h)) :
  h = 6 :=
by sorry

end NUMINAMATH_GPT_gretchen_work_hours_l1386_138639


namespace NUMINAMATH_GPT_total_revenue_from_selling_snakes_l1386_138637

-- Definitions based on conditions
def num_snakes := 3
def eggs_per_snake := 2
def standard_price := 250
def rare_multiplier := 4

-- Prove the total revenue Jake gets from selling all baby snakes is $2250
theorem total_revenue_from_selling_snakes : 
  (num_snakes * eggs_per_snake - 1) * standard_price + (standard_price * rare_multiplier) = 2250 := 
by
  sorry

end NUMINAMATH_GPT_total_revenue_from_selling_snakes_l1386_138637


namespace NUMINAMATH_GPT_quintuplets_babies_l1386_138629

theorem quintuplets_babies (a b c d : ℕ) 
  (h1 : d = 2 * c) 
  (h2 : c = 3 * b) 
  (h3 : b = 2 * a) 
  (h4 : 2 * a + 3 * b + 4 * c + 5 * d = 1200) : 
  5 * d = 18000 / 23 :=
by 
  sorry

end NUMINAMATH_GPT_quintuplets_babies_l1386_138629


namespace NUMINAMATH_GPT_correct_calculation_l1386_138608

theorem correct_calculation (x : ℕ) (h : 637 = x + 238) : x - 382 = 17 :=
by
  sorry

end NUMINAMATH_GPT_correct_calculation_l1386_138608


namespace NUMINAMATH_GPT_cost_of_projector_and_whiteboard_l1386_138609

variable (x : ℝ)

def cost_of_projector : ℝ := x
def cost_of_whiteboard : ℝ := x + 4000
def total_cost_eq_44000 : Prop := 4 * (x + 4000) + 3 * x = 44000

theorem cost_of_projector_and_whiteboard 
  (h : total_cost_eq_44000 x) : 
  cost_of_projector x = 4000 ∧ cost_of_whiteboard x = 8000 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_projector_and_whiteboard_l1386_138609


namespace NUMINAMATH_GPT_symmetric_shading_additional_squares_l1386_138600

theorem symmetric_shading_additional_squares :
  let initial_shaded : List (ℕ × ℕ) := [(1, 1), (2, 4), (4, 3)]
  let required_horizontal_symmetry := [(4, 1), (1, 6), (4, 6)]
  let required_vertical_symmetry := [(2, 3), (1, 3)]
  let total_additional_squares := required_horizontal_symmetry ++ required_vertical_symmetry
  let final_shaded := initial_shaded ++ total_additional_squares
  ∀ s ∈ total_additional_squares, s ∉ initial_shaded →
    final_shaded.length - initial_shaded.length = 5 :=
by
  sorry

end NUMINAMATH_GPT_symmetric_shading_additional_squares_l1386_138600


namespace NUMINAMATH_GPT_paco_cookie_problem_l1386_138678

theorem paco_cookie_problem (x : ℕ) (hx : x + 9 = 18) : x = 9 :=
by sorry

end NUMINAMATH_GPT_paco_cookie_problem_l1386_138678


namespace NUMINAMATH_GPT_marbles_remaining_l1386_138651

theorem marbles_remaining 
  (initial_remaining : ℕ := 400)
  (num_customers : ℕ := 20)
  (marbles_per_customer : ℕ := 15) :
  initial_remaining - (num_customers * marbles_per_customer) = 100 :=
by
  sorry

end NUMINAMATH_GPT_marbles_remaining_l1386_138651


namespace NUMINAMATH_GPT_kyle_lift_weight_l1386_138641

theorem kyle_lift_weight (this_year_weight last_year_weight : ℕ) 
  (h1 : this_year_weight = 80) 
  (h2 : this_year_weight = 3 * last_year_weight) : 
  (this_year_weight - last_year_weight) = 53 := by
  sorry

end NUMINAMATH_GPT_kyle_lift_weight_l1386_138641


namespace NUMINAMATH_GPT_sequence_ratio_l1386_138617

theorem sequence_ratio (S T a b : ℕ → ℚ) (h_sum_ratio : ∀ (n : ℕ), S n / T n = (7*n + 2) / (n + 3)) :
  a 7 / b 7 = 93 / 16 :=
by
  sorry

end NUMINAMATH_GPT_sequence_ratio_l1386_138617


namespace NUMINAMATH_GPT_jordan_more_novels_than_maxime_l1386_138621

theorem jordan_more_novels_than_maxime :
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  jordan_novels - maxime_novels = 51 :=
by
  let jordan_novels := 130
  let alexandre_novels := (1 / 10) * jordan_novels
  let camille_novels := 2 * alexandre_novels
  let total_novels := jordan_novels + alexandre_novels + camille_novels
  let maxime_novels := (1 / 2) * total_novels - 5
  sorry

end NUMINAMATH_GPT_jordan_more_novels_than_maxime_l1386_138621


namespace NUMINAMATH_GPT_hyperbola_find_a_b_l1386_138695

def hyperbola_conditions (a b : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)) ∧
  (∃ e : ℝ, e = 2) ∧ (∃ c : ℝ, c = 4)

theorem hyperbola_find_a_b (a b : ℝ) : hyperbola_conditions a b → a = 2 ∧ b = 2 * Real.sqrt 3 := 
sorry

end NUMINAMATH_GPT_hyperbola_find_a_b_l1386_138695


namespace NUMINAMATH_GPT_increase_in_average_l1386_138642

theorem increase_in_average (s1 s2 s3 s4 s5: ℝ)
  (h1: s1 = 92) (h2: s2 = 86) (h3: s3 = 89) (h4: s4 = 94) (h5: s5 = 91):
  ( ((s1 + s2 + s3 + s4 + s5) / 5) - ((s1 + s2 + s3) / 3) ) = 1.4 :=
by
  sorry

end NUMINAMATH_GPT_increase_in_average_l1386_138642


namespace NUMINAMATH_GPT_total_amount_is_152_l1386_138687

noncomputable def total_amount (p q r s t : ℝ) : ℝ := p + q + r + s + t

noncomputable def p_share (x : ℝ) : ℝ := 2 * x
noncomputable def q_share (x : ℝ) : ℝ := 1.75 * x
noncomputable def r_share (x : ℝ) : ℝ := 1.5 * x
noncomputable def s_share (x : ℝ) : ℝ := 1.25 * x
noncomputable def t_share (x : ℝ) : ℝ := 1.1 * x

theorem total_amount_is_152 (x : ℝ) (h1 : q_share x = 35) :
  total_amount (p_share x) (q_share x) (r_share x) (s_share x) (t_share x) = 152 := by
  sorry

end NUMINAMATH_GPT_total_amount_is_152_l1386_138687


namespace NUMINAMATH_GPT_decision_block_has_two_exits_l1386_138672

-- Define the conditions based on the problem
def output_block_exits := 1
def processing_block_exits := 1
def start_end_block_exits := 0
def decision_block_exits := 2

-- The proof statement
theorem decision_block_has_two_exits :
  (output_block_exits = 1) ∧
  (processing_block_exits = 1) ∧
  (start_end_block_exits = 0) ∧
  (decision_block_exits = 2) →
  decision_block_exits = 2 :=
by
  sorry

end NUMINAMATH_GPT_decision_block_has_two_exits_l1386_138672


namespace NUMINAMATH_GPT_g_neg501_l1386_138619

noncomputable def g : ℝ → ℝ := sorry

axiom g_eq (x y : ℝ) : g (x * y) + 2 * x = x * g y + g x

axiom g_neg1 : g (-1) = 7

theorem g_neg501 : g (-501) = 507 :=
by
  sorry

end NUMINAMATH_GPT_g_neg501_l1386_138619


namespace NUMINAMATH_GPT_fixed_point_f_l1386_138696

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.log (2 * x + 1) / Real.log a) + 2

theorem fixed_point_f (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : f a 0 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_f_l1386_138696


namespace NUMINAMATH_GPT_mn_value_l1386_138604

theorem mn_value (m n : ℤ) (h1 : 2 * m = 6) (h2 : m - n = 2) : m * n = 3 := by
  sorry

end NUMINAMATH_GPT_mn_value_l1386_138604


namespace NUMINAMATH_GPT_reversed_digit_multiple_of_sum_l1386_138630

variable (u v k : ℕ)

theorem reversed_digit_multiple_of_sum (h1 : 10 * u + v = k * (u + v)) :
  10 * v + u = (11 - k) * (u + v) :=
sorry

end NUMINAMATH_GPT_reversed_digit_multiple_of_sum_l1386_138630


namespace NUMINAMATH_GPT_area_ratio_of_squares_l1386_138631

open Real

theorem area_ratio_of_squares (a b : ℝ) (h : 4 * a = 4 * 4 * b) : (a^2) / (b^2) = 16 := 
by
  sorry

end NUMINAMATH_GPT_area_ratio_of_squares_l1386_138631


namespace NUMINAMATH_GPT_mandy_used_nutmeg_l1386_138646

theorem mandy_used_nutmeg (x : ℝ) (h1 : 0.67 = x + 0.17) : x = 0.50 :=
  by
  sorry

end NUMINAMATH_GPT_mandy_used_nutmeg_l1386_138646


namespace NUMINAMATH_GPT_combined_age_l1386_138612

-- Conditions as definitions
def AmyAge (j : ℕ) : ℕ :=
  j / 3

def ChrisAge (a : ℕ) : ℕ :=
  2 * a

-- Given condition
def JeremyAge : ℕ := 66

-- Question to prove
theorem combined_age : 
  let j := JeremyAge
  let a := AmyAge j
  let c := ChrisAge a
  a + j + c = 132 :=
by
  sorry

end NUMINAMATH_GPT_combined_age_l1386_138612


namespace NUMINAMATH_GPT_joe_used_fraction_paint_in_first_week_l1386_138634

variable (x : ℝ) -- Define the fraction x as a real number

-- Given conditions
def given_conditions : Prop := 
  let total_paint := 360
  let paint_first_week := x * total_paint
  let remaining_paint := (1 - x) * total_paint
  let paint_second_week := (1 / 2) * remaining_paint
  paint_first_week + paint_second_week = 225

-- The theorem to prove
theorem joe_used_fraction_paint_in_first_week (h : given_conditions x) : x = 1 / 4 :=
sorry

end NUMINAMATH_GPT_joe_used_fraction_paint_in_first_week_l1386_138634
