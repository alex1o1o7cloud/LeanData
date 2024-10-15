import Mathlib

namespace NUMINAMATH_GPT_power_of_m_divisible_by_33_l1017_101719

theorem power_of_m_divisible_by_33 (m : ℕ) (h : m > 0) (k : ℕ) (h_pow : (m ^ k) % 33 = 0) :
  ∃ n, n > 0 ∧ 11 ∣ m ^ n :=
by
  sorry

end NUMINAMATH_GPT_power_of_m_divisible_by_33_l1017_101719


namespace NUMINAMATH_GPT_last_digit_of_prime_l1017_101774

theorem last_digit_of_prime (n : ℕ) (h1 : 859433 = 214858 * 4 + 1) : (2 ^ 859433 - 1) % 10 = 1 := by
  sorry

end NUMINAMATH_GPT_last_digit_of_prime_l1017_101774


namespace NUMINAMATH_GPT_sequence_product_is_128_l1017_101748

-- Define the sequence of fractions
def fractional_sequence (n : ℕ) : Rat :=
  if n % 2 = 0 then 1 / (2 : ℕ) ^ ((n + 2) / 2)
  else (2 : ℕ) ^ ((n + 1) / 2)

-- The target theorem: prove the product of the sequence results in 128
theorem sequence_product_is_128 : 
  (List.prod (List.map fractional_sequence [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])) = 128 := 
by
  sorry

end NUMINAMATH_GPT_sequence_product_is_128_l1017_101748


namespace NUMINAMATH_GPT_distinct_banners_count_l1017_101795

def colors : Finset String := 
  {"red", "white", "blue", "green", "yellow"}

def valid_banners (strip1 strip2 strip3 : String) : Prop :=
  strip1 ∈ colors ∧ strip2 ∈ colors ∧ strip3 ∈ colors ∧
  strip1 ≠ strip2 ∧ strip2 ≠ strip3 ∧ strip3 ≠ strip1

theorem distinct_banners_count : 
  ∃ (banners : Finset (String × String × String)), 
    (∀ s1 s2 s3, (s1, s2, s3) ∈ banners ↔ valid_banners s1 s2 s3) ∧
    banners.card = 60 :=
by
  sorry

end NUMINAMATH_GPT_distinct_banners_count_l1017_101795


namespace NUMINAMATH_GPT_flight_time_is_approximately_50_hours_l1017_101734

noncomputable def flightTime (radius : ℝ) (speed : ℝ) : ℝ :=
  let circumference := 2 * Real.pi * radius
  circumference / speed

theorem flight_time_is_approximately_50_hours :
  let radius := 4200
  let speed := 525
  abs (flightTime radius speed - 50) < 1 :=
by
  sorry

end NUMINAMATH_GPT_flight_time_is_approximately_50_hours_l1017_101734


namespace NUMINAMATH_GPT_fraction_given_to_sofia_is_correct_l1017_101769

-- Pablo, Sofia, Mia, and Ana's initial egg counts
variables {m : ℕ}
def mia_initial (m : ℕ) := m
def sofia_initial (m : ℕ) := 3 * m
def pablo_initial (m : ℕ) := 12 * m
def ana_initial (m : ℕ) := m / 2

-- Total eggs and desired equal distribution
def total_eggs (m : ℕ) := 12 * m + 3 * m + m + m / 2
def equal_distribution (m : ℕ) := 33 * m / 4

-- Eggs each need to be equal
def sofia_needed (m : ℕ) := equal_distribution m - sofia_initial m
def mia_needed (m : ℕ) := equal_distribution m - mia_initial m
def ana_needed (m : ℕ) := equal_distribution m - ana_initial m

-- Fraction of eggs given to Sofia
def pablo_fraction_to_sofia (m : ℕ) := sofia_needed m / pablo_initial m

theorem fraction_given_to_sofia_is_correct (m : ℕ) :
  pablo_fraction_to_sofia m = 7 / 16 :=
sorry

end NUMINAMATH_GPT_fraction_given_to_sofia_is_correct_l1017_101769


namespace NUMINAMATH_GPT_cone_radius_l1017_101737

theorem cone_radius (r l : ℝ) 
  (surface_area_eq : π * r^2 + π * r * l = 12 * π)
  (net_is_semicircle : π * l = 2 * π * r) : 
  r = 2 :=
by
  sorry

end NUMINAMATH_GPT_cone_radius_l1017_101737


namespace NUMINAMATH_GPT_find_r_divisibility_l1017_101758

theorem find_r_divisibility (r : ℝ) :
  (∃ s : ℝ, 10 * (x - r)^2 * (x - s) = 10 * x^3 - 5 * x^2 - 52 * x + 56) → r = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_r_divisibility_l1017_101758


namespace NUMINAMATH_GPT_no_prime_sum_seventeen_l1017_101715

def is_prime (n : ℕ) : Prop := n ≥ 2 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem no_prime_sum_seventeen :
  ¬ ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ p + q = 17 := by
  sorry

end NUMINAMATH_GPT_no_prime_sum_seventeen_l1017_101715


namespace NUMINAMATH_GPT_max_prob_games_4_choose_best_of_five_l1017_101749

-- Definitions of probabilities for Team A and Team B in different game scenarios
def prob_win_deciding_game : ℝ := 0.5
def prob_A_non_deciding : ℝ := 0.6
def prob_B_non_deciding : ℝ := 0.4

-- Definitions of probabilities for different number of games in the series
def prob_xi_3 : ℝ := (prob_A_non_deciding)^3 + (prob_B_non_deciding)^3
def prob_xi_4 : ℝ := 3 * (prob_A_non_deciding^2 * prob_B_non_deciding * prob_A_non_deciding + prob_B_non_deciding^2 * prob_A_non_deciding * prob_B_non_deciding)
def prob_xi_5 : ℝ := 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2) * (2 * prob_win_deciding_game)

-- The statement that a series of 4 games has the highest probability
theorem max_prob_games_4 : prob_xi_4 > prob_xi_5 ∧ prob_xi_4 > prob_xi_3 :=
by {
  sorry
}

-- Definitions of winning probabilities in the series for Team A
def prob_A_win_best_of_3 : ℝ := (prob_A_non_deciding)^2 + 2 * (prob_A_non_deciding * prob_B_non_deciding * prob_win_deciding_game)
def prob_A_win_best_of_5 : ℝ := (prob_A_non_deciding)^3 + 3 * (prob_A_non_deciding^2 * prob_B_non_deciding) + 6 * (prob_A_non_deciding^2 * prob_B_non_deciding^2 * prob_win_deciding_game)

-- The statement that Team A has a higher chance of winning in a best-of-five series
theorem choose_best_of_five : prob_A_win_best_of_5 > prob_A_win_best_of_3 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_prob_games_4_choose_best_of_five_l1017_101749


namespace NUMINAMATH_GPT_find_f_3_l1017_101796

def f (x : ℝ) : ℝ := sorry

theorem find_f_3 : (∀ y : ℝ, y > 0 → f ((4 * y + 1) / (y + 1)) = 1 / y) → f 3 = 1 / 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_f_3_l1017_101796


namespace NUMINAMATH_GPT_solveEquation_l1017_101707

theorem solveEquation (x : ℝ) (hx : |x| ≥ 3) : (∃ x₁ x₂ : ℝ, (x₁ ≠ x₂ ∧ (x₁ / 3 + x₁ / Real.sqrt (x₁ ^ 2 - 9) = 35 / 12) ∧ (x₂ / 3 + x₂ / Real.sqrt (x₂ ^ 2 - 9) = 35 / 12)) ∧ x₁ + x₂ = 8.75) :=
sorry

end NUMINAMATH_GPT_solveEquation_l1017_101707


namespace NUMINAMATH_GPT_find_second_smallest_odd_number_l1017_101704

theorem find_second_smallest_odd_number (x : ℤ) (h : (x + (x + 2) + (x + 4) + (x + 6) = 112)) : (x + 2 = 27) :=
sorry

end NUMINAMATH_GPT_find_second_smallest_odd_number_l1017_101704


namespace NUMINAMATH_GPT_matchsticks_for_3_by_1996_grid_l1017_101777

def total_matchsticks_needed (rows cols : ℕ) : ℕ :=
  (cols * (rows + 1)) + (rows * (cols + 1))

theorem matchsticks_for_3_by_1996_grid : total_matchsticks_needed 3 1996 = 13975 := by
  sorry

end NUMINAMATH_GPT_matchsticks_for_3_by_1996_grid_l1017_101777


namespace NUMINAMATH_GPT_normal_line_at_x0_is_correct_l1017_101765

noncomputable def curve (x : ℝ) : ℝ := x^(2/3) - 20

def x0 : ℝ := -8

def normal_line_equation (x : ℝ) : ℝ := 3 * x + 8

theorem normal_line_at_x0_is_correct : 
  ∃ y0 : ℝ, curve x0 = y0 ∧ y0 = curve x0 ∧ normal_line_equation x0 = y0 :=
sorry

end NUMINAMATH_GPT_normal_line_at_x0_is_correct_l1017_101765


namespace NUMINAMATH_GPT_total_registration_methods_l1017_101785

theorem total_registration_methods (n : ℕ) (h : n = 5) : (2 ^ n) = 32 :=
by
  sorry

end NUMINAMATH_GPT_total_registration_methods_l1017_101785


namespace NUMINAMATH_GPT_molecular_weight_N2O5_l1017_101784

theorem molecular_weight_N2O5 :
  let atomic_weight_N := 14.01
  let atomic_weight_O := 16.00
  let molecular_weight_N2O5 := (2 * atomic_weight_N) + (5 * atomic_weight_O)
  molecular_weight_N2O5 = 108.02 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_N2O5_l1017_101784


namespace NUMINAMATH_GPT_smith_a_students_l1017_101751

-- Definitions representing the conditions

def johnson_a_students : ℕ := 12
def johnson_total_students : ℕ := 20
def smith_total_students : ℕ := 30

def johnson_ratio := johnson_a_students / johnson_total_students

-- Statement to prove
theorem smith_a_students :
  (johnson_a_students / johnson_total_students) = (18 / smith_total_students) :=
sorry

end NUMINAMATH_GPT_smith_a_students_l1017_101751


namespace NUMINAMATH_GPT_divides_f_of_nat_l1017_101790

variable {n : ℕ}

theorem divides_f_of_nat (n : ℕ) : 5 ∣ (76 * n^5 + 115 * n^4 + 19 * n) := 
sorry

end NUMINAMATH_GPT_divides_f_of_nat_l1017_101790


namespace NUMINAMATH_GPT_percentage_of_males_l1017_101767

theorem percentage_of_males (P : ℝ) (total_employees : ℝ) (below_50_male_count : ℝ) :
  total_employees = 2800 →
  0.70 * (P / 100 * total_employees) = below_50_male_count →
  below_50_male_count = 490 →
  P = 25 :=
by
  intros h_total h_eq h_below_50
  sorry

end NUMINAMATH_GPT_percentage_of_males_l1017_101767


namespace NUMINAMATH_GPT_correct_total_weight_6_moles_Al2_CO3_3_l1017_101794

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_C : ℝ := 12.01
def atomic_weight_O : ℝ := 16.00

def num_atoms_Al : ℕ := 2
def num_atoms_C : ℕ := 3
def num_atoms_O : ℕ := 9

def molecular_weight_Al2_CO3_3 : ℝ :=
  (num_atoms_Al * atomic_weight_Al) +
  (num_atoms_C * atomic_weight_C) +
  (num_atoms_O * atomic_weight_O)

def num_moles : ℝ := 6

def total_weight_6_moles_Al2_CO3_3 : ℝ := num_moles * molecular_weight_Al2_CO3_3

theorem correct_total_weight_6_moles_Al2_CO3_3 :
  total_weight_6_moles_Al2_CO3_3 = 1403.94 :=
by
  unfold total_weight_6_moles_Al2_CO3_3
  unfold num_moles
  unfold molecular_weight_Al2_CO3_3
  unfold num_atoms_Al num_atoms_C num_atoms_O atomic_weight_Al atomic_weight_C atomic_weight_O
  sorry

end NUMINAMATH_GPT_correct_total_weight_6_moles_Al2_CO3_3_l1017_101794


namespace NUMINAMATH_GPT_circle_equation_line_intersect_circle_l1017_101717

theorem circle_equation (x y : ℝ) : 
  y = x^2 - 4*x + 3 → (x = 0 ∧ y = 3) ∨ (y = 0 ∧ (x = 1 ∨ x = 3)) :=
sorry

theorem line_intersect_circle (m : ℝ) :
  (∀ x y : ℝ, (x + y + m = 0) ∨ ((x - 2)^2 + (y - 2)^2 = 5)) →
  (∀ x₁ y₁ x₂ y₂ : ℝ, 
    (x₁ + y₁ + m = 0) → ((x₁ - 2)^2 + (y₁ - 2)^2 = 5) →
    (x₂ + y₂ + m = 0) → ((x₂ - 2)^2 + (y₂ - 2)^2 = 5) →
    ((x₁ * x₂ + y₁ * y₂ = 0) → (m = -1 ∨ m = -3))) :=
sorry

end NUMINAMATH_GPT_circle_equation_line_intersect_circle_l1017_101717


namespace NUMINAMATH_GPT_t_shirts_to_buy_l1017_101764

variable (P T : ℕ)

def condition1 : Prop := 3 * P + 6 * T = 750
def condition2 : Prop := P + 12 * T = 750

theorem t_shirts_to_buy (h1 : condition1 P T) (h2 : condition2 P T) :
  400 / T = 8 :=
by
  sorry

end NUMINAMATH_GPT_t_shirts_to_buy_l1017_101764


namespace NUMINAMATH_GPT_math_club_team_selection_l1017_101757

theorem math_club_team_selection :
  let boys := 10
  let girls := 12
  let total := boys + girls
  let team_size := 8
  (Nat.choose total team_size - Nat.choose girls team_size - Nat.choose boys team_size = 319230) :=
by
  sorry

end NUMINAMATH_GPT_math_club_team_selection_l1017_101757


namespace NUMINAMATH_GPT_cosine_value_of_angle_between_vectors_l1017_101725

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_value_of_angle_between_vectors :
  cosine_angle a b = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end NUMINAMATH_GPT_cosine_value_of_angle_between_vectors_l1017_101725


namespace NUMINAMATH_GPT_handshakes_l1017_101710

open Nat

theorem handshakes : ∃ x : ℕ, 4 + 3 + 2 + 1 + x = 10 ∧ x = 2 :=
by
  existsi 2
  simp
  sorry

end NUMINAMATH_GPT_handshakes_l1017_101710


namespace NUMINAMATH_GPT_volume_of_sphere_l1017_101709

theorem volume_of_sphere
  (r : ℝ) (V : ℝ)
  (h₁ : r = 1/3)
  (h₂ : 2 * r = (16/9 * V)^(1/3)) :
  V = 1/6 :=
  sorry

end NUMINAMATH_GPT_volume_of_sphere_l1017_101709


namespace NUMINAMATH_GPT_john_total_distance_l1017_101773

-- Define the parameters according to the conditions
def daily_distance : ℕ := 1700
def number_of_days : ℕ := 6
def total_distance : ℕ := daily_distance * number_of_days

-- Lean theorem statement to prove the total distance run by John
theorem john_total_distance : total_distance = 10200 := by
  -- Here, the proof would go, but it is omitted as per instructions
  sorry

end NUMINAMATH_GPT_john_total_distance_l1017_101773


namespace NUMINAMATH_GPT_find_complex_number_z_l1017_101763

-- Given the complex number z and the equation \(\frac{z}{1+i} = i^{2015} + i^{2016}\)
-- prove that z = -2i
theorem find_complex_number_z (z : ℂ) (h : z / (1 + (1 : ℂ) * I) = I ^ 2015 + I ^ 2016) : z = -2 * I := 
by
  sorry

end NUMINAMATH_GPT_find_complex_number_z_l1017_101763


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1017_101750

noncomputable def is_increasing_on_R (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem necessary_but_not_sufficient (f : ℝ → ℝ) :
  (f 1 < f 2) → (¬∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) ∨ (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1017_101750


namespace NUMINAMATH_GPT_carlos_gold_quarters_l1017_101727

theorem carlos_gold_quarters (quarter_weight : ℚ) 
  (store_value_per_quarter : ℚ) 
  (melt_value_per_ounce : ℚ) 
  (quarters_per_ounce : ℚ := 1 / quarter_weight) 
  (spent_value : ℚ := quarters_per_ounce * store_value_per_quarter)
  (melted_value: ℚ := melt_value_per_ounce) :
  quarter_weight = 1/5 ∧ store_value_per_quarter = 0.25 ∧ melt_value_per_ounce = 100 → 
  melted_value / spent_value = 80 := 
by
  intros h
  sorry

end NUMINAMATH_GPT_carlos_gold_quarters_l1017_101727


namespace NUMINAMATH_GPT_diane_coffee_purchase_l1017_101782

theorem diane_coffee_purchase (c d : ℕ) (h1 : c + d = 7) (h2 : 90 * c + 60 * d % 100 = 0) : c = 6 :=
by
  sorry

end NUMINAMATH_GPT_diane_coffee_purchase_l1017_101782


namespace NUMINAMATH_GPT_positive_number_property_l1017_101766

theorem positive_number_property (x : ℝ) (h_pos : x > 0) (h_property : 0.01 * x * x = 4) : x = 20 :=
sorry

end NUMINAMATH_GPT_positive_number_property_l1017_101766


namespace NUMINAMATH_GPT_product_of_numbers_l1017_101713

variable (x y z : ℝ)

theorem product_of_numbers :
  x + y + z = 36 ∧ x = 3 * (y + z) ∧ y = 6 * z → x * y * z = 268 := 
by
  sorry

end NUMINAMATH_GPT_product_of_numbers_l1017_101713


namespace NUMINAMATH_GPT_remainder_problem_l1017_101729

theorem remainder_problem (n : ℤ) (h : n % 25 = 4) : (n + 15) % 5 = 4 := by
  sorry

end NUMINAMATH_GPT_remainder_problem_l1017_101729


namespace NUMINAMATH_GPT_simplify_expression_l1017_101711

-- Defining the variables involved
variables (b : ℝ)

-- The theorem statement that needs to be proven
theorem simplify_expression : 3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1017_101711


namespace NUMINAMATH_GPT_evaluate_expression_l1017_101761

theorem evaluate_expression :
  1002^3 - 1001 * 1002^2 - 1001^2 * 1002 + 1001^3 - 1000^3 = 2009007 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1017_101761


namespace NUMINAMATH_GPT_anniversary_sale_total_cost_l1017_101780

-- Definitions of conditions
def original_price_ice_cream : ℕ := 12
def discount_ice_cream : ℕ := 2
def sale_price_ice_cream : ℕ := original_price_ice_cream - discount_ice_cream

def price_per_five_cans_juice : ℕ := 2
def cans_per_five_pack : ℕ := 5

-- Definition of total cost
def total_cost : ℕ := 2 * sale_price_ice_cream + (10 / cans_per_five_pack) * price_per_five_cans_juice

-- The goal is to prove that total_cost is 24
theorem anniversary_sale_total_cost : total_cost = 24 :=
by
  sorry

end NUMINAMATH_GPT_anniversary_sale_total_cost_l1017_101780


namespace NUMINAMATH_GPT_NataliesSisterInitialDiaries_l1017_101776

theorem NataliesSisterInitialDiaries (D : ℕ)
  (h1 : 2 * D - (1 / 4) * 2 * D = 18) : D = 12 :=
by sorry

end NUMINAMATH_GPT_NataliesSisterInitialDiaries_l1017_101776


namespace NUMINAMATH_GPT_output_in_scientific_notation_l1017_101739

def output_kilowatt_hours : ℝ := 448000
def scientific_notation (n : ℝ) : Prop := n = 4.48 * 10^5

theorem output_in_scientific_notation : scientific_notation output_kilowatt_hours :=
by
  -- Proof steps are not required
  sorry

end NUMINAMATH_GPT_output_in_scientific_notation_l1017_101739


namespace NUMINAMATH_GPT_find_number_l1017_101786

-- Given conditions
variables (x y : ℕ)

-- The conditions from the problem statement
def digit_sum : Prop := x + y = 12
def reverse_condition : Prop := (10 * x + y) + 36 = 10 * y + x

-- The final statement
theorem find_number (h1 : digit_sum x y) (h2 : reverse_condition x y) : 10 * x + y = 48 :=
sorry

end NUMINAMATH_GPT_find_number_l1017_101786


namespace NUMINAMATH_GPT_distance_between_points_l1017_101760

theorem distance_between_points : 
  let p1 := (3, -2) 
  let p2 := (-7, 4) 
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = Real.sqrt 136 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_points_l1017_101760


namespace NUMINAMATH_GPT_solve_exponential_eq_l1017_101730

theorem solve_exponential_eq (x : ℝ) : 
  ((5 - 2 * x)^(x + 1) = 1) ↔ (x = -1 ∨ x = 2 ∨ x = 3) := by
  sorry

end NUMINAMATH_GPT_solve_exponential_eq_l1017_101730


namespace NUMINAMATH_GPT_vector_sum_eq_l1017_101743

variables (x y : ℝ)
def a : ℝ × ℝ := (2, 3)
def b : ℝ × ℝ := (3, 3)
def c : ℝ × ℝ := (7, 8)

theorem vector_sum_eq :
  ∃ (x y : ℝ), c = (x • a.1 + y • b.1, x • a.2 + y • b.2) ∧ x + y = 8 / 3 :=
by
  have h1 : 7 = 2 * x + 3 * y := sorry
  have h2 : 8 = 3 * x + 3 * y := sorry
  sorry

end NUMINAMATH_GPT_vector_sum_eq_l1017_101743


namespace NUMINAMATH_GPT_distance_to_building_materials_l1017_101787

theorem distance_to_building_materials (D : ℝ) 
  (h1 : 2 * 10 * 4 * D = 8000) : 
  D = 100 := 
by
  sorry

end NUMINAMATH_GPT_distance_to_building_materials_l1017_101787


namespace NUMINAMATH_GPT_LittleRedHeightCorrect_l1017_101728

noncomputable def LittleRedHeight : ℝ :=
let LittleMingHeight := 1.3 
let HeightDifference := 0.2 
LittleMingHeight - HeightDifference

theorem LittleRedHeightCorrect : LittleRedHeight = 1.1 := by
  sorry

end NUMINAMATH_GPT_LittleRedHeightCorrect_l1017_101728


namespace NUMINAMATH_GPT_min_knights_in_village_l1017_101792

theorem min_knights_in_village :
  ∃ (K L : ℕ), K + L = 7 ∧ 2 * K * L = 24 ∧ K ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_min_knights_in_village_l1017_101792


namespace NUMINAMATH_GPT_trigonometric_inequality_l1017_101778

theorem trigonometric_inequality (x : Real) (h1 : 0 < x) (h2 : x < (3 * Real.pi) / 8) :
  (1 / Real.sin (x / 3) + 1 / Real.sin (8 * x / 3) > (Real.sin (3 * x / 2)) / (Real.sin (x / 2) * Real.sin (2 * x))) :=
  by
  sorry

end NUMINAMATH_GPT_trigonometric_inequality_l1017_101778


namespace NUMINAMATH_GPT_num_of_nickels_l1017_101731

theorem num_of_nickels (x : ℕ) (hx_eq_dimes : ∀ n, n = x → n = x) (hx_eq_quarters : ∀ n, n = x → n = 2 * x) (total_value : 5 * x + 10 * x + 50 * x = 1950) : x = 30 :=
sorry

end NUMINAMATH_GPT_num_of_nickels_l1017_101731


namespace NUMINAMATH_GPT_isosceles_trapezoid_rotation_produces_frustum_l1017_101747

-- Definitions based purely on conditions
structure IsoscelesTrapezoid :=
(a b c d : ℝ) -- sides
(ha : a = c) -- isosceles property
(hb : b ≠ d) -- non-parallel sides

def rotateAroundSymmetryAxis (shape : IsoscelesTrapezoid) : Type :=
-- We need to define what the rotation of the trapezoid produces
sorry

theorem isosceles_trapezoid_rotation_produces_frustum (shape : IsoscelesTrapezoid) :
  rotateAroundSymmetryAxis shape = Frustum :=
sorry

end NUMINAMATH_GPT_isosceles_trapezoid_rotation_produces_frustum_l1017_101747


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l1017_101771

variable (p q : Prop)
-- Condition p: The base of a right prism is a rhombus.
def base_of_right_prism_is_rhombus := p
-- Condition q: A prism is a right rectangular prism.
def prism_is_right_rectangular := q

-- Proof: p is a necessary but not sufficient condition for q.
theorem necessary_but_not_sufficient (p q : Prop) 
  (h1 : base_of_right_prism_is_rhombus p)
  (h2 : prism_is_right_rectangular q) : 
  (q → p) ∧ ¬ (p → q) :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l1017_101771


namespace NUMINAMATH_GPT_prism_unique_triple_l1017_101706

theorem prism_unique_triple :
  ∃! (a b c : ℕ), a ≤ b ∧ b ≤ c ∧ b = 2000 ∧
                  (∃ b' c', b' = 2000 ∧ c' = 2000 ∧
                  (∃ k : ℚ, k = 1/2 ∧
                  (∃ x y z, x = a / 2 ∧ y = 1000 ∧ z = c / 2 ∧ a = 2000 ∧ c = 2000)))
/- The proof is omitted for this statement. -/
:= sorry

end NUMINAMATH_GPT_prism_unique_triple_l1017_101706


namespace NUMINAMATH_GPT_problem_inequality_l1017_101702

variable {a b c d : ℝ}

theorem problem_inequality (h1 : 0 ≤ a) (h2 : 0 ≤ d) (h3 : 0 < b) (h4 : 0 < c) (h5 : b + c ≥ a + d) :
  (b / (c + d)) + (c / (b + a)) ≥ (Real.sqrt 2) - (1 / 2) := 
sorry

end NUMINAMATH_GPT_problem_inequality_l1017_101702


namespace NUMINAMATH_GPT_combination_sum_l1017_101724

noncomputable def combination (n r : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial r * Nat.factorial (n - r))

theorem combination_sum :
  combination 3 2 + combination 4 2 + combination 5 2 + combination 6 2 + 
  combination 7 2 + combination 8 2 + combination 9 2 + combination 10 2 = 164 :=
by
  sorry

end NUMINAMATH_GPT_combination_sum_l1017_101724


namespace NUMINAMATH_GPT_find_X_l1017_101779

theorem find_X :
  (15.2 * 0.25 - 48.51 / 14.7) / X = ((13 / 44 - 2 / 11 - 5 / 66) / (5 / 2) * (6 / 5)) / (3.2 + 0.8 * (5.5 - 3.25)) ->
  X = 137.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_X_l1017_101779


namespace NUMINAMATH_GPT_hyperbola_h_k_a_b_sum_eq_l1017_101755

theorem hyperbola_h_k_a_b_sum_eq :
  ∃ (h k a b : ℝ), 
  h = 0 ∧ 
  k = 0 ∧ 
  a = 4 ∧ 
  (c : ℝ) = 8 ∧ 
  c^2 = a^2 + b^2 ∧ 
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
{ sorry }

end NUMINAMATH_GPT_hyperbola_h_k_a_b_sum_eq_l1017_101755


namespace NUMINAMATH_GPT_sector_angle_l1017_101775

theorem sector_angle (r : ℝ) (θ : ℝ) 
  (area_eq : (1 / 2) * θ * r^2 = 1)
  (perimeter_eq : 2 * r + θ * r = 4) : θ = 2 := 
by
  sorry

end NUMINAMATH_GPT_sector_angle_l1017_101775


namespace NUMINAMATH_GPT_express_x_in_terms_of_y_l1017_101722

theorem express_x_in_terms_of_y (x y : ℝ) (h : 3 * x - 4 * y = 8) : x = (4 * y + 8) / 3 :=
sorry

end NUMINAMATH_GPT_express_x_in_terms_of_y_l1017_101722


namespace NUMINAMATH_GPT_max_m_eq_4_inequality_a_b_c_l1017_101772

noncomputable def f (x : ℝ) : ℝ :=
  |x - 3| + |x + 2|

theorem max_m_eq_4 (m : ℝ) (h : ∀ x : ℝ, f x ≥ |m + 1|) : m ≤ 4 ∧ m ≥ -6 :=
  sorry

theorem inequality_a_b_c (a b c : ℝ) (h : a + 2 * b + c = 4) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 / (a + b) + 1 / (b + c) ≥ 1 :=
  sorry

end NUMINAMATH_GPT_max_m_eq_4_inequality_a_b_c_l1017_101772


namespace NUMINAMATH_GPT_avg_annual_reduction_l1017_101705

theorem avg_annual_reduction (x : ℝ) (hx : (1 - x)^2 = 0.64) : x = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_avg_annual_reduction_l1017_101705


namespace NUMINAMATH_GPT_cost_of_each_ticket_l1017_101723

theorem cost_of_each_ticket (x : ℝ) : 
  500 * x * 0.70 = 4 * 2625 → x = 30 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_ticket_l1017_101723


namespace NUMINAMATH_GPT_inequality_solution_set_l1017_101768

def solution_set (a b x : ℝ) : Set ℝ := {x | |a - b * x| - 5 ≤ 0}

theorem inequality_solution_set (x : ℝ) :
  solution_set 4 3 x = {x | - (1 : ℝ) / 3 ≤ x ∧ x ≤ 3} :=
by {
  sorry
}

end NUMINAMATH_GPT_inequality_solution_set_l1017_101768


namespace NUMINAMATH_GPT_hank_donates_90_percent_l1017_101718

theorem hank_donates_90_percent (x : ℝ) : 
  (100 * x + 0.75 * 80 + 50 = 200) → (x = 0.9) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_hank_donates_90_percent_l1017_101718


namespace NUMINAMATH_GPT_binom_1300_2_eq_844350_l1017_101700

theorem binom_1300_2_eq_844350 : (Nat.choose 1300 2) = 844350 := by
  sorry

end NUMINAMATH_GPT_binom_1300_2_eq_844350_l1017_101700


namespace NUMINAMATH_GPT_find_divisor_l1017_101759

theorem find_divisor (d : ℕ) (n : ℕ) (least : ℕ)
  (h1 : least = 2)
  (h2 : n = 433124)
  (h3 : ∀ d : ℕ, (d ∣ (n + least)) → d = 2) :
  d = 2 := 
sorry

end NUMINAMATH_GPT_find_divisor_l1017_101759


namespace NUMINAMATH_GPT_mike_corvette_average_speed_l1017_101701

theorem mike_corvette_average_speed
  (D : ℚ) (v : ℚ) (total_distance : ℚ)
  (first_half_distance : ℚ) (second_half_time_ratio : ℚ)
  (total_time : ℚ) (average_rate : ℚ) :
  total_distance = 640 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_time_ratio = 3 ∧
  average_rate = 40 →
  v = 80 :=
by
  intros h
  have total_distance_eq : total_distance = 640 := h.1
  have first_half_distance_eq : first_half_distance = total_distance / 2 := h.2.1
  have second_half_time_ratio_eq : second_half_time_ratio = 3 := h.2.2.1
  have average_rate_eq : average_rate = 40 := h.2.2.2
  sorry

end NUMINAMATH_GPT_mike_corvette_average_speed_l1017_101701


namespace NUMINAMATH_GPT_no_solution_to_equation_l1017_101781

theorem no_solution_to_equation :
  ¬ ∃ x : ℝ, x ≠ 5 ∧ (1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) :=
by 
  sorry

end NUMINAMATH_GPT_no_solution_to_equation_l1017_101781


namespace NUMINAMATH_GPT_tangent_lines_through_point_l1017_101783

theorem tangent_lines_through_point (x y : ℝ) (hp : (x, y) = (3, 1))
 : ∃ (a b c : ℝ), (y - 1 = (4 / 3) * (x - 3) ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_tangent_lines_through_point_l1017_101783


namespace NUMINAMATH_GPT_joan_mortgage_payback_months_l1017_101788

-- Define the conditions and statement

def first_payment : ℕ := 100
def total_amount : ℕ := 2952400

theorem joan_mortgage_payback_months :
  ∃ n : ℕ, 100 * (3^n - 1) / (3 - 1) = 2952400 ∧ n = 10 :=
by
  sorry

end NUMINAMATH_GPT_joan_mortgage_payback_months_l1017_101788


namespace NUMINAMATH_GPT_students_not_skating_nor_skiing_l1017_101726

theorem students_not_skating_nor_skiing (total_students skating_students skiing_students both_students : ℕ)
  (h_total : total_students = 30)
  (h_skating : skating_students = 20)
  (h_skiing : skiing_students = 9)
  (h_both : both_students = 5) :
  total_students - (skating_students + skiing_students - both_students) = 6 :=
by
  sorry

end NUMINAMATH_GPT_students_not_skating_nor_skiing_l1017_101726


namespace NUMINAMATH_GPT_molecular_weight_is_correct_l1017_101793

structure Compound :=
  (H C N Br O : ℕ)

structure AtomicWeights :=
  (H C N Br O : ℝ)

noncomputable def molecularWeight (compound : Compound) (weights : AtomicWeights) : ℝ :=
  compound.H * weights.H +
  compound.C * weights.C +
  compound.N * weights.N +
  compound.Br * weights.Br +
  compound.O * weights.O

def givenCompound : Compound :=
  { H := 2, C := 2, N := 1, Br := 1, O := 4 }

def givenWeights : AtomicWeights :=
  { H := 1.008, C := 12.011, N := 14.007, Br := 79.904, O := 15.999 }

theorem molecular_weight_is_correct : molecularWeight givenCompound givenWeights = 183.945 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_is_correct_l1017_101793


namespace NUMINAMATH_GPT_tom_has_65_fruits_left_l1017_101721

def initial_fruits : ℕ := 40 + 70 + 30 + 15

def sold_oranges : ℕ := (1 / 4) * 40
def sold_apples : ℕ := (2 / 3) * 70
def sold_bananas : ℕ := (5 / 6) * 30
def sold_kiwis : ℕ := (60 / 100) * 15

def fruits_remaining : ℕ :=
  40 - sold_oranges +
  70 - sold_apples +
  30 - sold_bananas +
  15 - sold_kiwis

theorem tom_has_65_fruits_left :
  fruits_remaining = 65 := by
  sorry

end NUMINAMATH_GPT_tom_has_65_fruits_left_l1017_101721


namespace NUMINAMATH_GPT_total_points_first_half_l1017_101799

noncomputable def raiders_wildcats_scores := 
  ∃ (a b d r : ℕ),
    (a = b + 1) ∧
    (a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 2) ∧
    (a + a * r ≤ 100) ∧
    (b + b + d ≤ 100)

theorem total_points_first_half : 
  raiders_wildcats_scores → 
  ∃ (total : ℕ), total = 25 :=
by
  sorry

end NUMINAMATH_GPT_total_points_first_half_l1017_101799


namespace NUMINAMATH_GPT_find_number_l1017_101754

theorem find_number (x : ℕ) (h : x * 625 = 584638125) : x = 935420 :=
sorry

end NUMINAMATH_GPT_find_number_l1017_101754


namespace NUMINAMATH_GPT_sum_geometric_series_is_correct_l1017_101762

def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem sum_geometric_series_is_correct
  (a r : ℚ) (n : ℕ)
  (h_a : a = 1/4)
  (h_r : r = 1/4)
  (h_n : n = 5) :
  geometric_series_sum a r n = 341 / 1024 :=
by
  rw [h_a, h_r, h_n]
  -- Now we can skip the proof.
  sorry

end NUMINAMATH_GPT_sum_geometric_series_is_correct_l1017_101762


namespace NUMINAMATH_GPT_specific_n_values_l1017_101720

theorem specific_n_values (n : ℕ) : 
  ∃ m : ℕ, 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → m % k = 0) ∧ 
    (m % (n + 1) ≠ 0) ∧ 
    (m % (n + 2) ≠ 0) ∧ 
    (m % (n + 3) ≠ 0) ↔ n = 1 ∨ n = 2 ∨ n = 6 := 
by
  sorry

end NUMINAMATH_GPT_specific_n_values_l1017_101720


namespace NUMINAMATH_GPT_expression_simplification_l1017_101708

theorem expression_simplification :
  (- (1 / 2)) ^ 2023 * 2 ^ 2024 = -2 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplification_l1017_101708


namespace NUMINAMATH_GPT_team_air_conditioner_installation_l1017_101735

theorem team_air_conditioner_installation (x : ℕ) (y : ℕ) 
  (h1 : 66 % x = 0) 
  (h2 : 60 % y = 0) 
  (h3 : x = y + 2) 
  (h4 : 66 / x = 60 / y) 
  : x = 22 ∧ y = 20 :=
by
  have h5 : x = 22 := sorry
  have h6 : y = 20 := sorry
  exact ⟨h5, h6⟩

end NUMINAMATH_GPT_team_air_conditioner_installation_l1017_101735


namespace NUMINAMATH_GPT_total_fruits_l1017_101797

theorem total_fruits (cucumbers : ℕ) (watermelons : ℕ) 
  (h1 : cucumbers = 18) 
  (h2 : watermelons = cucumbers + 8) : 
  cucumbers + watermelons = 44 := 
by {
  sorry
}

end NUMINAMATH_GPT_total_fruits_l1017_101797


namespace NUMINAMATH_GPT_apples_to_eat_raw_l1017_101744

/-- Proof of the number of apples left to eat raw given the conditions -/
theorem apples_to_eat_raw 
  (total_apples : ℕ)
  (pct_wormy : ℕ)
  (pct_moldy : ℕ)
  (wormy_apples_offset : ℕ)
  (wormy_apples bruised_apples moldy_apples apples_left : ℕ) 
  (h1 : total_apples = 120)
  (h2 : pct_wormy = 20)
  (h3 : pct_moldy = 30)
  (h4 : wormy_apples = pct_wormy * total_apples / 100)
  (h5 : moldy_apples = pct_moldy * total_apples / 100)
  (h6 : bruised_apples = wormy_apples + wormy_apples_offset)
  (h7 : wormy_apples_offset = 9)
  (h8 : apples_left = total_apples - (wormy_apples + moldy_apples + bruised_apples))
  : apples_left = 27 :=
sorry

end NUMINAMATH_GPT_apples_to_eat_raw_l1017_101744


namespace NUMINAMATH_GPT_area_of_triangle_ADE_l1017_101742

theorem area_of_triangle_ADE (A B C D E : Type) (AB BC AC : ℝ) (AD AE : ℝ)
  (h1 : AB = 8) (h2 : BC = 13) (h3 : AC = 15) (h4 : AD = 3) (h5 : AE = 11) :
  let s := (AB + BC + AC) / 2
  let area_ABC := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  let sinA := 2 * area_ABC / (AB * AC)
  let area_ADE := (1 / 2) * AD * AE * sinA
  area_ADE = (33 * Real.sqrt 3) / 4 :=
by 
  have s := (8 + 13 + 15) / 2
  have area_ABC := Real.sqrt (s * (s - 8) * (s - 13) * (s - 15))
  have sinA := 2 * area_ABC / (8 * 15)
  have area_ADE := (1 / 2) * 3 * 11 * sinA
  sorry

end NUMINAMATH_GPT_area_of_triangle_ADE_l1017_101742


namespace NUMINAMATH_GPT_find_c_l1017_101714

/-- Define the conditions given in the problem --/
def parabola_equation (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

def vertex_condition (a b c : ℝ) : Prop := 
  ∀ x, parabola_equation a b c x = a * (x - 3)^2 - 1

def passes_through_point (a b c : ℝ) : Prop := 
  parabola_equation a b c 1 = 5

/-- The main statement -/
theorem find_c (a b c : ℝ) 
  (h_vertex : vertex_condition a b c) 
  (h_point : passes_through_point a b c) :
  c = 12.5 :=
sorry

end NUMINAMATH_GPT_find_c_l1017_101714


namespace NUMINAMATH_GPT_bench_cost_l1017_101733

theorem bench_cost (B : ℕ) (h : B + 2 * B = 450) : B = 150 :=
by {
  sorry
}

end NUMINAMATH_GPT_bench_cost_l1017_101733


namespace NUMINAMATH_GPT_precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l1017_101770

-- Given definitions for precision adjustment
def initial_precision := 3

def new_precision_mult (x : ℕ): ℕ :=
  initial_precision - 1   -- Example: Multiplying by 10 moves decimal point right decreasing precision by 1

def new_precision_mult_large (x : ℕ): ℕ := 
  initial_precision - 2   -- Example: Multiplying by 35 generally decreases precision by 2

def new_precision_div (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 10 moves decimal point left increasing precision by 1

def new_precision_div_large (x : ℕ): ℕ := 
  initial_precision + 1   -- Example: Dividing by 35 generally increases precision by 1

-- Statements to prove
theorem precision_mult_10_decreases: 
  new_precision_mult 10 = 2 := 
by 
  sorry

theorem precision_mult_35_decreases: 
  new_precision_mult_large 35 = 1 := 
by 
  sorry

theorem precision_div_10_increases: 
  new_precision_div 10 = 4 := 
by 
  sorry

theorem precision_div_35_increases: 
  new_precision_div_large 35 = 4 := 
by 
  sorry

end NUMINAMATH_GPT_precision_mult_10_decreases_precision_mult_35_decreases_precision_div_10_increases_precision_div_35_increases_l1017_101770


namespace NUMINAMATH_GPT_students_taking_German_l1017_101746

theorem students_taking_German 
  (total_students : ℕ)
  (students_taking_French : ℕ)
  (students_taking_both : ℕ)
  (students_not_taking_either : ℕ) 
  (students_taking_German : ℕ) 
  (h1 : total_students = 69)
  (h2 : students_taking_French = 41)
  (h3 : students_taking_both = 9)
  (h4 : students_not_taking_either = 15)
  (h5 : students_taking_German = 22) :
  total_students - students_not_taking_either = students_taking_French + students_taking_German - students_taking_both :=
sorry

end NUMINAMATH_GPT_students_taking_German_l1017_101746


namespace NUMINAMATH_GPT_correct_value_of_a_l1017_101712

namespace ProofProblem

-- Condition 1: Definition of set M
def M : Set ℤ := {x | x^2 ≤ 1}

-- Condition 2: Definition of set N dependent on a parameter a
def N (a : ℤ) : Set ℤ := {a, a * a}

-- Question translated: Correct value of a such that M ∪ N = M
theorem correct_value_of_a (a : ℤ) : (M ∪ N a = M) → a = -1 :=
by
  sorry

end ProofProblem

end NUMINAMATH_GPT_correct_value_of_a_l1017_101712


namespace NUMINAMATH_GPT_cosine_of_angle_l1017_101789

theorem cosine_of_angle (α : ℝ) (h : Real.sin (Real.pi / 6 + α) = Real.sqrt 3 / 2) : 
  Real.cos (Real.pi / 3 - α) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_cosine_of_angle_l1017_101789


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_eq_zero_l1017_101740

theorem sum_of_squares_of_roots_eq_zero :
  let f : Polynomial ℝ := Polynomial.C 50 + Polynomial.monomial 3 (-2) + Polynomial.monomial 7 5 + Polynomial.monomial 10 1
  ∀ (r : ℝ), r ∈ Multiset.toFinset f.roots → r ^ 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_eq_zero_l1017_101740


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_ending_in_3_l1017_101741

theorem smallest_four_digit_divisible_by_53_ending_in_3 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 53 ∣ n ∧ n % 10 = 3 ∧ n = 1113 := 
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_53_ending_in_3_l1017_101741


namespace NUMINAMATH_GPT_price_decrease_percentage_l1017_101752

-- Definitions based on given conditions
def price_in_2007 (x : ℝ) : ℝ := x
def price_in_2008 (x : ℝ) : ℝ := 1.25 * x
def desired_price_in_2009 (x : ℝ) : ℝ := 1.1 * x

-- Theorem statement to prove the price decrease from 2008 to 2009
theorem price_decrease_percentage (x : ℝ) (h : x > 0) : 
  (1.25 * x - 1.1 * x) / (1.25 * x) = 0.12 := 
sorry

end NUMINAMATH_GPT_price_decrease_percentage_l1017_101752


namespace NUMINAMATH_GPT_negation_exists_zero_product_l1017_101745

variable {R : Type} [LinearOrderedField R]

variable (f g : R → R)

theorem negation_exists_zero_product :
  (¬ ∃ x : R, f x * g x = 0) ↔ ∀ x : R, f x ≠ 0 ∧ g x ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_negation_exists_zero_product_l1017_101745


namespace NUMINAMATH_GPT_curve_is_circle_l1017_101732

theorem curve_is_circle (s : ℝ) :
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  x^2 + y^2 = 1 :=
by
  let x := (3 - s^2) / (3 + s^2)
  let y := (4 * s) / (3 + s^2)
  sorry

end NUMINAMATH_GPT_curve_is_circle_l1017_101732


namespace NUMINAMATH_GPT_total_legs_l1017_101756

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end NUMINAMATH_GPT_total_legs_l1017_101756


namespace NUMINAMATH_GPT_value_of_a7_l1017_101738

-- Define an arithmetic sequence
structure ArithmeticSeq (a : Nat → ℤ) :=
  (d : ℤ)
  (a_eq : ∀ n, a (n+1) = a n + d)

-- Lean statement of the equivalent proof problem
theorem value_of_a7 (a : ℕ → ℤ) (H : ArithmeticSeq a) :
  (2 * a 4 - a 7 ^ 2 + 2 * a 10 = 0) → a 7 = 4 * H.d :=
by
  sorry

end NUMINAMATH_GPT_value_of_a7_l1017_101738


namespace NUMINAMATH_GPT_regular_polygon_enclosure_l1017_101798

theorem regular_polygon_enclosure (m n : ℕ) (h : m = 12)
    (h_enc : ∀ p : ℝ, p = 360 / ↑n → (2 * (180 / ↑n)) = (360 / ↑m)) :
    n = 12 :=
by
  sorry

end NUMINAMATH_GPT_regular_polygon_enclosure_l1017_101798


namespace NUMINAMATH_GPT_problem_statement_l1017_101703

theorem problem_statement
  (m : ℝ) 
  (h : m + (1/m) = 5) :
  m^2 + (1 / m^2) + 4 = 27 :=
by
  -- Parameter types are chosen based on the context and problem description.
  sorry

end NUMINAMATH_GPT_problem_statement_l1017_101703


namespace NUMINAMATH_GPT_largest_triangle_angle_l1017_101791

theorem largest_triangle_angle (h_ratio : ∃ (a b c : ℕ), a / b = 3 / 4 ∧ b / c = 4 / 9) 
  (h_external_angle : ∃ (θ1 θ2 θ3 θ4 : ℝ), θ1 = 3 * x ∧ θ2 = 4 * x ∧ θ3 = 9 * x ∧ θ4 = 3 * x ∧ θ1 + θ2 + θ3 = 180) :
  ∃ (θ3 : ℝ), θ3 = 101.25 := by
  sorry

end NUMINAMATH_GPT_largest_triangle_angle_l1017_101791


namespace NUMINAMATH_GPT_concert_ticket_revenue_l1017_101736

theorem concert_ticket_revenue :
  let price_student : ℕ := 9
  let price_non_student : ℕ := 11
  let total_tickets : ℕ := 2000
  let student_tickets : ℕ := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  revenue_student + revenue_non_student = 20960 :=
by
  -- Definitions
  let price_student := 9
  let price_non_student := 11
  let total_tickets := 2000
  let student_tickets := 520
  let non_student_tickets := total_tickets - student_tickets
  let revenue_student := student_tickets * price_student
  let revenue_non_student := non_student_tickets * price_non_student
  -- Proof
  sorry  -- Placeholder for the proof

end NUMINAMATH_GPT_concert_ticket_revenue_l1017_101736


namespace NUMINAMATH_GPT_car_travel_distance_l1017_101716

variable (b t : Real)
variable (h1 : b > 0)
variable (h2 : t > 0)

theorem car_travel_distance (b t : Real) (h1 : b > 0) (h2 : t > 0) :
  let rate := b / 4
  let inches_in_yard := 36
  let time_in_seconds := 5 * 60
  let distance_in_inches := (rate / t) * time_in_seconds
  let distance_in_yards := distance_in_inches / inches_in_yard
  distance_in_yards = (25 * b) / (12 * t) := by
  sorry

end NUMINAMATH_GPT_car_travel_distance_l1017_101716


namespace NUMINAMATH_GPT_cost_of_each_big_apple_l1017_101753

theorem cost_of_each_big_apple :
  ∀ (small_cost medium_cost : ℝ) (big_cost : ℝ) (num_small num_medium num_big : ℕ) (total_cost : ℝ),
  small_cost = 1.5 →
  medium_cost = 2 →
  num_small = 6 →
  num_medium = 6 →
  num_big = 8 →
  total_cost = 45 →
  total_cost = num_small * small_cost + num_medium * medium_cost + num_big * big_cost →
  big_cost = 3 :=
by
  intros small_cost medium_cost big_cost num_small num_medium num_big total_cost
  sorry

end NUMINAMATH_GPT_cost_of_each_big_apple_l1017_101753
