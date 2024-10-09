import Mathlib

namespace solve_for_x_l18_1838

theorem solve_for_x (x : ℝ) (h : 6 * x ^ (1 / 3) - 3 * (x / x ^ (2 / 3)) = -1 + 2 * x ^ (1 / 3) + 4) :
  x = 27 :=
by 
  sorry

end solve_for_x_l18_1838


namespace box_height_is_6_l18_1853

-- Defining the problem setup
variables (h : ℝ) (r_large r_small : ℝ)
variables (box_size : ℝ) (n_spheres : ℕ)

-- The conditions of the problem
def rectangular_box :=
  box_size = 5 ∧ r_large = 3 ∧ r_small = 1.5 ∧ n_spheres = 4 ∧
  (∀ k : ℕ, k < n_spheres → 
   ∃ C : ℝ, 
     (C = r_small) ∧ 
     -- Each smaller sphere is tangent to three sides of the box condition
     (C ≤ box_size))

def sphere_tangency (h r_large r_small : ℝ) :=
  h = 2 * r_large ∧ r_large + r_small = 4.5

def height_of_box (h : ℝ) := 2 * 3 = h

-- The mathematically equivalent proof problem
theorem box_height_is_6 (h : ℝ) (r_large : ℝ) (r_small : ℝ) (box_size : ℝ) (n_spheres : ℕ) 
  (conditions : rectangular_box box_size r_large r_small n_spheres) 
  (tangency : sphere_tangency h r_large r_small) :
  height_of_box h :=
by {
  -- Proof is omitted
  sorry
}

end box_height_is_6_l18_1853


namespace inverse_function_passes_through_point_a_l18_1882

theorem inverse_function_passes_through_point_a
  (a : ℝ) (ha_pos : 0 < a) (ha_neq_one : a ≠ 1) :
  ∃ (A : ℝ × ℝ), A = (2, 3) ∧ (∀ x, (a^(x-3) + 1) = 2 ↔ x = 3) → (2 - 1)/(3-3) = 0 :=
by
  sorry

end inverse_function_passes_through_point_a_l18_1882


namespace max_positive_integers_on_circle_l18_1806

theorem max_positive_integers_on_circle (a : ℕ → ℕ) (h: ∀ k : ℕ, 2 < k → a k > a (k-1) + a (k-2)) :
  ∃ n : ℕ, (∀ i < 2018, a i > 0 -> n ≤ 1009) :=
  sorry

end max_positive_integers_on_circle_l18_1806


namespace total_surface_area_prime_rectangular_solid_l18_1855

-- Definitions of the conditions
def is_prime (n : ℕ) : Prop := Prime n

def prime_edge_lengths (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c

def volume (a b c : ℕ) : ℕ := a * b * c

def surface_area (a b c : ℕ) : ℕ := 2 * (a * b + b * c + c * a)

-- The main theorem statement
theorem total_surface_area_prime_rectangular_solid :
  ∃ (a b c : ℕ), prime_edge_lengths a b c ∧ volume a b c = 105 ∧ surface_area a b c = 142 :=
sorry

end total_surface_area_prime_rectangular_solid_l18_1855


namespace find_expression_value_l18_1836

theorem find_expression_value (m : ℝ) (h : m + 1/m = 10) : m^2 + 1/m^2 + m^3 + 1/m^3 + 4 = 1072 := 
by 
  sorry

end find_expression_value_l18_1836


namespace find_number_l18_1879

theorem find_number (x : ℝ) (h : x - (3 / 5) * x = 64) : x = 160 :=
sorry

end find_number_l18_1879


namespace atomic_weight_of_Calcium_l18_1843

/-- Given definitions -/
def molecular_weight_CaOH₂ : ℕ := 74
def atomic_weight_O : ℕ := 16
def atomic_weight_H : ℕ := 1

/-- Given conditions -/
def total_weight_O_H : ℕ := 2 * atomic_weight_O + 2 * atomic_weight_H

/-- Problem statement -/
theorem atomic_weight_of_Calcium (H1 : molecular_weight_CaOH₂ = 74)
                                   (H2 : atomic_weight_O = 16)
                                   (H3 : atomic_weight_H = 1)
                                   (H4 : total_weight_O_H = 2 * atomic_weight_O + 2 * atomic_weight_H) :
  74 - (2 * 16 + 2 * 1) = 40 :=
by {
  sorry
}

end atomic_weight_of_Calcium_l18_1843


namespace sufficient_but_not_necessary_for_abs_eq_two_l18_1821

theorem sufficient_but_not_necessary_for_abs_eq_two (a : ℝ) :
  (a = -2 → |a| = 2) ∧ (|a| = 2 → a = 2 ∨ a = -2) :=
by
   sorry

end sufficient_but_not_necessary_for_abs_eq_two_l18_1821


namespace find_number_l18_1830

theorem find_number (x: ℝ) (h1: 0.10 * x + 0.15 * 50 = 10.5) : x = 30 :=
by
  sorry

end find_number_l18_1830


namespace infinite_geometric_series_first_term_l18_1877

theorem infinite_geometric_series_first_term 
  (r : ℝ) 
  (S : ℝ) 
  (a : ℝ) 
  (h1 : r = -3/7) 
  (h2 : S = 18) 
  (h3 : S = a / (1 - r)) : 
  a = 180 / 7 := by
  -- omitted proof
  sorry

end infinite_geometric_series_first_term_l18_1877


namespace two_dollar_coin_is_toonie_l18_1829

/-- We define the $2 coin in Canada -/
def two_dollar_coin_name : String := "toonie"

/-- Antonella's wallet problem setup -/
def Antonella_has_ten_coins := 10
def loonies_value := 1
def toonies_value := 2
def coins_after_purchase := 11
def purchase_amount := 3
def initial_toonies := 4

/-- Proving that the $2 coin is called a "toonie" -/
theorem two_dollar_coin_is_toonie :
  two_dollar_coin_name = "toonie" :=
by
  -- Here, we place the logical steps to derive that two_dollar_coin_name = "toonie"
  sorry

end two_dollar_coin_is_toonie_l18_1829


namespace freezer_temp_correct_l18_1895

variable (t_refrigeration : ℝ) (t_freezer : ℝ)

-- Given conditions
def refrigeration_temperature := t_refrigeration = 5
def freezer_temperature := t_freezer = -12

-- Goal: Prove that the freezer compartment's temperature is -12 degrees Celsius
theorem freezer_temp_correct : freezer_temperature t_freezer := by
  sorry

end freezer_temp_correct_l18_1895


namespace parallel_lines_iff_l18_1880

theorem parallel_lines_iff (a : ℝ) :
  (∀ x y : ℝ, x - y - 1 = 0 → x + a * y - 2 = 0) ↔ (a = -1) :=
by
  sorry

end parallel_lines_iff_l18_1880


namespace inverse_variation_l18_1866

theorem inverse_variation (k : ℝ) : 
  (∀ (x y : ℝ), x * y^2 = k) → 
  (∀ (x y : ℝ), x = 1 → y = 2 → k = 4) → 
  (x = 0.1111111111111111) → 
  (y = 6) :=
by 
  -- Assume the given conditions
  intros h h0 hx
  -- Proof goes here...
  sorry

end inverse_variation_l18_1866


namespace sam_drove_distance_l18_1845

theorem sam_drove_distance (m_distance : ℕ) (m_time : ℕ) (s_time : ℕ) (s_distance : ℕ)
  (m_distance_eq : m_distance = 120) (m_time_eq : m_time = 3) (s_time_eq : s_time = 4) :
  s_distance = (m_distance / m_time) * s_time :=
by
  sorry

end sam_drove_distance_l18_1845


namespace factorize_x_squared_plus_2x_l18_1870

theorem factorize_x_squared_plus_2x (x : ℝ) : x^2 + 2*x = x*(x + 2) :=
by sorry

end factorize_x_squared_plus_2x_l18_1870


namespace all_blue_figures_are_small_l18_1811

variables (Shape : Type) (Large Blue Small Square Triangle : Shape → Prop)

-- Given conditions
axiom h1 : ∀ (x : Shape), Large x → Square x
axiom h2 : ∀ (x : Shape), Blue x → Triangle x

-- The goal to prove
theorem all_blue_figures_are_small : ∀ (x : Shape), Blue x → Small x :=
by
  sorry

end all_blue_figures_are_small_l18_1811


namespace age_of_other_man_l18_1863

theorem age_of_other_man
  (n : ℕ) (average_age_before : ℕ) (average_age_after : ℕ) (age_of_one_man : ℕ) (average_age_women : ℕ) 
  (h1 : n = 9)
  (h2 : average_age_after = average_age_before + 4)
  (h3 : age_of_one_man = 36)
  (h4 : average_age_women = 52) :
  (68 - 36 = 32) := 
by
  sorry

end age_of_other_man_l18_1863


namespace all_numbers_are_2007_l18_1814

noncomputable def sequence_five_numbers (a b c d e : ℤ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ 
  (a = 2007 ∨ b = 2007 ∨ c = 2007 ∨ d = 2007 ∨ e = 2007) ∧ 
  (∃ r1, b = r1 * a ∧ c = r1 * b ∧ d = r1 * c ∧ e = r1 * d) ∧
  (∃ r2, a = r2 * b ∧ c = r2 * a ∧ d = r2 * c ∧ e = r2 * d) ∧
  (∃ r3, a = r3 * c ∧ b = r3 * a ∧ d = r3 * b ∧ e = r3 * d) ∧
  (∃ r4, a = r4 * d ∧ b = r4 * a ∧ c = r4 * b ∧ e = r4 * d) ∧
  (∃ r5, a = r5 * e ∧ b = r5 * a ∧ c = r5 * b ∧ d = r5 * c)

theorem all_numbers_are_2007 (a b c d e : ℤ) 
  (h : sequence_five_numbers a b c d e) : 
  a = 2007 ∧ b = 2007 ∧ c = 2007 ∧ d = 2007 ∧ e = 2007 :=
sorry

end all_numbers_are_2007_l18_1814


namespace irrational_number_problem_l18_1801

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem irrational_number_problem :
  ∀ x ∈ ({(0.4 : ℝ), (2 / 3 : ℝ), (2 : ℝ), - (Real.sqrt 5)} : Set ℝ), 
  is_irrational x ↔ x = - (Real.sqrt 5) :=
by
  intros x hx
  -- Other proof steps can go here
  sorry

end irrational_number_problem_l18_1801


namespace range_of_sqrt_x_minus_1_meaningful_l18_1802

theorem range_of_sqrt_x_minus_1_meaningful (x : ℝ) (h : 0 ≤ x - 1) : 1 ≤ x := 
sorry

end range_of_sqrt_x_minus_1_meaningful_l18_1802


namespace determine_positions_l18_1865

-- Defining the participants
inductive Participant
| Olya
| Oleg
| Pasha

open Participant

-- Defining the possible places
inductive Place
| First
| Second
| Third

open Place

-- Define the conditions
def condition1 (pos : Participant → Place) : Prop := 
  pos Olya = First ∨ pos Oleg = First ∨ pos Pasha = First

def condition2 (pos : Participant → Place) : Prop :=
  (pos Olya = First ∧ pos Olya = Second ∧ pos Olya = Third) ∨
  (pos Oleg = First ∧ pos Oleg = Second ∧ pos Oleg = Third) ∨
  (pos Pasha = First ∧ pos Pasha = Second ∧ pos Pasha = Third)

def condition3 (pos : Participant → Place) : Prop :=
  ∀ p, pos p ≠ First ∧ pos p ≠ Second ∧ pos p ≠ Third

def condition4 (pos : Participant → Place) : Prop :=
  (pos Olya = First → (pos Oleg = First ∨ pos Pasha = First)) ∧
  (pos Oleg = First → pos Olya ≠ First) ∧
  (pos Pasha = First → (pos Oleg = First ∨ pos Olya = First))

def always_true_or_false : Prop :=
  (∀ p, p = Olya ∨ p = Oleg ∨ p = Pasha )

-- Main theorem
theorem determine_positions (pos : Participant → Place) :
  condition1 pos ∧ condition2 pos ∧ condition3 pos ∧ condition4 pos ∧ always_true_or_false →
  pos Oleg = First ∧ pos Pasha = Second ∧ pos Olya = Third := 
by
  sorry

end determine_positions_l18_1865


namespace tax_computation_l18_1899

def income : ℕ := 56000
def first_portion_income : ℕ := 40000
def first_portion_rate : ℝ := 0.12
def remaining_income : ℕ := income - first_portion_income
def remaining_rate : ℝ := 0.20
def expected_tax : ℝ := 8000

theorem tax_computation :
  (first_portion_rate * first_portion_income) +
  (remaining_rate * remaining_income) = expected_tax := by
  sorry

end tax_computation_l18_1899


namespace pow_sub_nat_ge_seven_l18_1876

open Nat

theorem pow_sub_nat_ge_seven
  (m n : ℕ) 
  (h1 : m > 1)
  (h2 : 2^(2 * m + 1) - n^2 ≥ 0) : 
  2^(2 * m + 1) - n^2 ≥ 7 :=
sorry

end pow_sub_nat_ge_seven_l18_1876


namespace find_a_value_l18_1874

-- Define the problem conditions
theorem find_a_value (a : ℝ) :
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  (mean_y = 0.95 * mean_x + 2.6) → a = 2.2 :=
by
  -- Let bindings are for convenience to follow the problem statement
  let x_values := [0, 1, 3, 4]
  let y_values := [a, 4.3, 4.8, 6.7]
  let mean_x := (0 + 1 + 3 + 4) / 4
  let mean_y := (a + 4.3 + 4.8 + 6.7) / 4
  intro h
  sorry

end find_a_value_l18_1874


namespace first_house_bottles_l18_1820

theorem first_house_bottles (total_bottles : ℕ) 
  (cider_only : ℕ) (beer_only : ℕ) (half : ℕ → ℕ)
  (mixture : ℕ)
  (half_cider_bottles : ℕ)
  (half_beer_bottles : ℕ)
  (half_mixture_bottles : ℕ) : 
  total_bottles = 180 →
  cider_only = 40 →
  beer_only = 80 →
  mixture = total_bottles - (cider_only + beer_only) →
  half c = c / 2 →
  half_cider_bottles = half cider_only →
  half_beer_bottles = half beer_only →
  half_mixture_bottles = half mixture →
  half_cider_bottles + half_beer_bottles + half_mixture_bottles = 90 :=
by
  intros h_tot h_cid h_beer h_mix h_half half_cid half_beer half_mix
  sorry

end first_house_bottles_l18_1820


namespace stu_books_count_l18_1898

noncomputable def elmo_books : ℕ := 24
noncomputable def laura_books : ℕ := elmo_books / 3
noncomputable def stu_books : ℕ := laura_books / 2

theorem stu_books_count :
  stu_books = 4 :=
by
  sorry

end stu_books_count_l18_1898


namespace cubic_function_decreasing_l18_1852

-- Define the given function
def f (a x : ℝ) : ℝ := a * x^3 - 1

-- Define the condition that the function is decreasing on ℝ
def is_decreasing_on_R (a : ℝ) : Prop :=
  ∀ x : ℝ, 3 * a * x^2 ≤ 0 

-- Main theorem and its statement
theorem cubic_function_decreasing (a : ℝ) (h : is_decreasing_on_R a) : a < 0 :=
sorry

end cubic_function_decreasing_l18_1852


namespace part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l18_1878

-- Definitions for part (1)
def P_X_1 : ℚ := 1 / 6
def P_X_2 : ℚ := 5 / 36
def P_X_3 : ℚ := 25 / 216
def P_X_4 : ℚ := 125 / 216
def E_X : ℚ := 671 / 216

theorem part1_prob_dist (X : ℚ) :
  (X = 1 → P_X_1 = 1 / 6) ∧
  (X = 2 → P_X_2 = 5 / 36) ∧
  (X = 3 → P_X_3 = 25 / 216) ∧
  (X = 4 → P_X_4 = 125 / 216) := 
by sorry

theorem part1_expectation :
  E_X = 671 / 216 :=
by sorry

-- Definition for part (2)
def P_A_wins_n_throws (n : ℕ) : ℚ := 1 / 6 * (5 / 6) ^ (2 * n - 2)

theorem part2_prob_A_wins_n_throws (n : ℕ) (hn : n ≥ 1) :
  P_A_wins_n_throws n = 1 / 6 * (5 / 6) ^ (2 * n - 2) :=
by sorry

end part1_prob_dist_part1_expectation_part2_prob_A_wins_n_throws_l18_1878


namespace find_rate_percent_l18_1884

-- Define the conditions
def principal : ℝ := 1200
def time : ℝ := 4
def simple_interest : ℝ := 400

-- Define the rate that we need to prove
def rate : ℝ := 8.3333  -- approximately

-- Formalize the proof problem in Lean 4
theorem find_rate_percent
  (P : ℝ) (T : ℝ) (SI : ℝ) (R : ℝ)
  (hP : P = principal) (hT : T = time) (hSI : SI = simple_interest) :
  SI = (P * R * T) / 100 → R = rate :=
by
  intros h
  sorry

end find_rate_percent_l18_1884


namespace shaded_to_white_area_ratio_l18_1825

-- Define the problem
theorem shaded_to_white_area_ratio :
  let total_triangles_shaded := 5
  let total_triangles_white := 3
  let ratio_shaded_to_white := total_triangles_shaded / total_triangles_white
  ratio_shaded_to_white = (5 : ℚ)/(3 : ℚ) := by
  -- Proof steps should be provided here, but "sorry" is used to skip the proof.
  sorry

end shaded_to_white_area_ratio_l18_1825


namespace reciprocal_neg_one_thirteen_l18_1867

theorem reciprocal_neg_one_thirteen : -(1:ℝ) / 13⁻¹ = -13 := 
sorry

end reciprocal_neg_one_thirteen_l18_1867


namespace students_no_A_in_any_subject_l18_1834

def total_students : ℕ := 50
def a_in_history : ℕ := 9
def a_in_math : ℕ := 15
def a_in_science : ℕ := 12
def a_in_math_and_history : ℕ := 5
def a_in_history_and_science : ℕ := 3
def a_in_science_and_math : ℕ := 4
def a_in_all_three : ℕ := 1

theorem students_no_A_in_any_subject : 
  (total_students - (a_in_history + a_in_math + a_in_science 
                      - a_in_math_and_history - a_in_history_and_science - a_in_science_and_math 
                      + a_in_all_three)) = 28 := by
  sorry

end students_no_A_in_any_subject_l18_1834


namespace arithmetic_sum_sequence_l18_1871

theorem arithmetic_sum_sequence (a : ℕ → ℝ) (d : ℝ)
  (h : ∀ n, a (n + 1) = a n + d) :
  ∃ d', 
    a 4 + a 5 + a 6 - (a 1 + a 2 + a 3) = d' ∧
    a 7 + a 8 + a 9 - (a 4 + a 5 + a 6) = d' :=
by
  sorry

end arithmetic_sum_sequence_l18_1871


namespace percentage_of_600_eq_half_of_900_l18_1886

theorem percentage_of_600_eq_half_of_900 : 
  ∃ P : ℝ, (P / 100) * 600 = 0.5 * 900 ∧ P = 75 := by
  -- Proof goes here
  sorry

end percentage_of_600_eq_half_of_900_l18_1886


namespace phillip_initial_marbles_l18_1817

theorem phillip_initial_marbles
  (dilan_marbles : ℕ) (martha_marbles : ℕ) (veronica_marbles : ℕ) 
  (total_after_redistribution : ℕ) 
  (individual_marbles_after : ℕ) :
  dilan_marbles = 14 →
  martha_marbles = 20 →
  veronica_marbles = 7 →
  total_after_redistribution = 4 * individual_marbles_after →
  individual_marbles_after = 15 →
  ∃phillip_marbles : ℕ, phillip_marbles = 19 :=
by
  intro h_dilan h_martha h_veronica h_total_after h_individual
  have total_initial := 60 - (14 + 20 + 7)
  existsi total_initial
  sorry

end phillip_initial_marbles_l18_1817


namespace find_real_numbers_a_b_l18_1889

noncomputable def f (a b x : ℝ) : ℝ :=
  a * (Real.sin x * Real.cos x) - (Real.sqrt 3) * a * (Real.cos x) ^ 2 + Real.sqrt 3 / 2 * a + b

theorem find_real_numbers_a_b (a b : ℝ) (h1 : 0 < a)
    (h2 : ∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), -2 ≤ f a b x ∧ f a b x ≤ Real.sqrt 3)
    : a = 2 ∧ b = -2 + Real.sqrt 3 :=
sorry

end find_real_numbers_a_b_l18_1889


namespace pq_or_l18_1813

def p : Prop := 2 % 2 = 0
def q : Prop := 3 % 2 = 0

theorem pq_or : p ∨ q :=
by
  -- proof goes here
  sorry

end pq_or_l18_1813


namespace min_value_d1_d2_l18_1860

noncomputable def min_distance_sum : ℝ :=
  let d1 (u : ℝ) : ℝ := (1 / 5) * abs (3 * Real.cos u - 4 * Real.sin u - 10)
  let d2 (u : ℝ) : ℝ := 3 - Real.cos u
  let d_sum (u : ℝ) : ℝ := d1 u + d2 u
  ((5 - (4 * Real.sqrt 5 / 5)))

theorem min_value_d1_d2 :
  ∀ (P : ℝ × ℝ) (u : ℝ),
    P = (Real.cos u, Real.sin u) →
    (P.1 ^ 2 + P.2 ^ 2 = 1) →
    let d1 := (1 / 5) * abs (3 * P.1 - 4 * P.2 - 10)
    let d2 := 3 - P.1
    d1 + d2 ≥ (5 - (4 * Real.sqrt 5 / 5)) :=
by
  sorry

end min_value_d1_d2_l18_1860


namespace egg_production_difference_l18_1849

def eggs_last_year : ℕ := 1416
def eggs_this_year : ℕ := 4636
def eggs_difference (a b : ℕ) : ℕ := a - b

theorem egg_production_difference : eggs_difference eggs_this_year eggs_last_year = 3220 := 
by
  sorry

end egg_production_difference_l18_1849


namespace solve_for_x_l18_1885

variable (x : ℝ)

theorem solve_for_x (h : 0.05 * x + 0.12 * (30 + x) = 15.6) : x = 12 / 0.17 := by
  sorry

end solve_for_x_l18_1885


namespace how_much_together_l18_1805

def madeline_money : ℕ := 48
def brother_money : ℕ := madeline_money / 2

theorem how_much_together : madeline_money + brother_money = 72 := by
  sorry

end how_much_together_l18_1805


namespace find_a_l18_1823

namespace MathProof

theorem find_a (a : ℕ) (h_pos : a > 0) (h_eq : (a : ℚ) / (a + 18) = 47 / 50) : a = 282 :=
by
  sorry

end MathProof

end find_a_l18_1823


namespace glucose_solution_volume_l18_1812

theorem glucose_solution_volume (V : ℕ) (h : 500 / 10 = V / 20) : V = 1000 :=
sorry

end glucose_solution_volume_l18_1812


namespace placemat_length_correct_l18_1881

noncomputable def placemat_length (r : ℝ) : ℝ :=
  2 * r * Real.sin (Real.pi / 8)

theorem placemat_length_correct (r : ℝ) (n : ℕ) (w : ℝ) (y : ℝ) (h_r : r = 5)
  (h_n : n = 8) (h_w : w = 1)
  (h_y : y = placemat_length r) :
  y = 5 * Real.sqrt (2 - Real.sqrt 2) :=
by
  sorry

end placemat_length_correct_l18_1881


namespace birds_more_than_nests_l18_1819

theorem birds_more_than_nests : 
  let birds := 6 
  let nests := 3 
  (birds - nests) = 3 := 
by 
  sorry

end birds_more_than_nests_l18_1819


namespace parallel_lines_slope_equal_intercepts_lines_l18_1883

theorem parallel_lines_slope (m : ℝ) :
  (∀ x y, (2 * x - y - 3 = 0 ∧ x - m * y + 1 - 3 * m = 0) → 2 = (1 / m)) → m = 1 / 2 :=
by
  intro h
  sorry

theorem equal_intercepts_lines (m : ℝ) :
  (m ≠ 0 → (∀ x y, (x - m * y + 1 - 3 * m = 0) → (1 - 3 * m) / m = 3 * m - 1)) →
  (m = -1 ∨ m = 1 / 3) →
  ∀ x y, (x - m * y + 1 - 3 * m = 0) →
  (x + y + 4 = 0 ∨ 3 * x - y = 0) :=
by
  intro h hm
  sorry

end parallel_lines_slope_equal_intercepts_lines_l18_1883


namespace product_xyz_l18_1896

theorem product_xyz (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z)
  (h4 : x * y = 30 * (4:ℝ)^(1/3)) (h5 : x * z = 45 * (4:ℝ)^(1/3)) (h6 : y * z = 18 * (4:ℝ)^(1/3)) :
  x * y * z = 540 * Real.sqrt 3 :=
sorry

end product_xyz_l18_1896


namespace person_A_boxes_average_unit_price_after_promotion_l18_1842

-- Definitions based on the conditions.
def unit_price (x: ℕ) (y: ℕ) : ℚ := y / x

def person_A_spent : ℕ := 2400
def person_B_spent : ℕ := 3000
def promotion_discount : ℕ := 20
def boxes_difference : ℕ := 10

-- Main proofs
theorem person_A_boxes (unit_price: ℕ → ℕ → ℚ) 
  (person_A_spent person_B_spent boxes_difference: ℕ): 
  ∃ x, unit_price person_A_spent x = unit_price person_B_spent (x + boxes_difference) 
  ∧ x = 40 := 
by {
  sorry
}

theorem average_unit_price_after_promotion (unit_price: ℕ → ℕ → ℚ) 
  (promotion_discount: ℕ) (person_A_spent person_B_spent: ℕ) 
  (boxes_A_promotion boxes_B: ℕ): 
  person_A_spent / (boxes_A_promotion * 2) + 20 = 48 
  ∧ person_B_spent / (boxes_B * 2) + 20 = 50 :=
by {
  sorry
}

end person_A_boxes_average_unit_price_after_promotion_l18_1842


namespace intersection_M_N_l18_1873

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_M_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
sorry

end intersection_M_N_l18_1873


namespace friedas_probability_to_corner_l18_1816

-- Define the grid size and positions
def grid_size : Nat := 4
def start_position : ℕ × ℕ := (3, 3)
def corner_positions : List (ℕ × ℕ) := [(1, 1), (1, 4), (4, 1), (4, 4)]

-- Define the number of hops allowed
def max_hops : Nat := 4

-- Define a function to calculate the probability of reaching a corner square
-- within the given number of hops starting from the initial position.
noncomputable def prob_reach_corner (grid_size : ℕ) (start_position : ℕ × ℕ) 
                                     (corner_positions : List (ℕ × ℕ)) 
                                     (max_hops : ℕ) : ℚ :=
  -- Implementation details skipped
  sorry

-- Define the main theorem that states the desired probability
theorem friedas_probability_to_corner : 
  prob_reach_corner grid_size start_position corner_positions max_hops = 17 / 64 :=
sorry

end friedas_probability_to_corner_l18_1816


namespace woman_total_coins_l18_1888

theorem woman_total_coins
  (num_each_coin : ℕ)
  (h : 1 * num_each_coin + 5 * num_each_coin + 10 * num_each_coin + 25 * num_each_coin + 100 * num_each_coin = 351)
  : 5 * num_each_coin = 15 :=
by
  sorry

end woman_total_coins_l18_1888


namespace average_rate_dan_trip_l18_1894

/-- 
Given:
- Dan runs along a 4-mile stretch of river and then swims back along the same route.
- Dan runs at a rate of 10 miles per hour.
- Dan swims at a rate of 6 miles per hour.

Prove:
Dan's average rate for the entire trip is 0.125 miles per minute.
-/
theorem average_rate_dan_trip :
  let distance := 4 -- miles
  let run_rate := 10 -- miles per hour
  let swim_rate := 6 -- miles per hour
  let time_run_hours := distance / run_rate -- hours
  let time_swim_hours := distance / swim_rate -- hours
  let time_run_minutes := time_run_hours * 60 -- minutes
  let time_swim_minutes := time_swim_hours * 60 -- minutes
  let total_distance := distance + distance -- miles
  let total_time := time_run_minutes + time_swim_minutes -- minutes
  let average_rate := total_distance / total_time -- miles per minute
  average_rate = 0.125 :=
by sorry

end average_rate_dan_trip_l18_1894


namespace coins_player_1_received_l18_1861

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l18_1861


namespace at_least_two_consecutive_heads_probability_l18_1809

noncomputable def probability_at_least_two_consecutive_heads : ℚ := 
  let total_outcomes := 16
  let unfavorable_outcomes := 8
  1 - (unfavorable_outcomes / total_outcomes)

theorem at_least_two_consecutive_heads_probability :
  probability_at_least_two_consecutive_heads = 1 / 2 := 
by
  sorry

end at_least_two_consecutive_heads_probability_l18_1809


namespace initial_amount_in_cookie_jar_l18_1846

theorem initial_amount_in_cookie_jar (M : ℝ) (h : 15 / 100 * (85 / 100 * (100 - 10) / 100 * (100 - 15) / 100 * M) = 15) : M = 24.51 :=
sorry

end initial_amount_in_cookie_jar_l18_1846


namespace correct_option_is_C_l18_1892

-- Definitions of the expressions given in the conditions
def optionA (a : ℝ) : ℝ := 3 * a^5 - a^5
def optionB (a : ℝ) : ℝ := a^2 + a^5
def optionC (a : ℝ) : ℝ := a^5 + a^5
def optionD (x y : ℝ) : ℝ := x^2 * y + x * y^2

-- The problem is to prove that optionC is correct and the others are not
theorem correct_option_is_C (a x y : ℝ) :
  (optionC a = 2 * a^5) ∧ 
  (optionA a ≠ 3) ∧ 
  (optionB a ≠ a^7) ∧ 
  (optionD x y ≠ 2 * (x ^ 3) * (y ^ 3)) :=
by
  sorry

end correct_option_is_C_l18_1892


namespace tangent_product_20_40_60_80_l18_1862

theorem tangent_product_20_40_60_80 :
  Real.tan (20 * Real.pi / 180) * Real.tan (40 * Real.pi / 180) * Real.tan (60 * Real.pi / 180) * Real.tan (80 * Real.pi / 180) = 3 :=
by
  sorry

end tangent_product_20_40_60_80_l18_1862


namespace daniel_total_earnings_l18_1828

-- Definitions of conditions
def fabric_delivered_monday : ℕ := 20
def fabric_delivered_tuesday : ℕ := 2 * fabric_delivered_monday
def fabric_delivered_wednesday : ℕ := fabric_delivered_tuesday / 4
def total_fabric_delivered : ℕ := fabric_delivered_monday + fabric_delivered_tuesday + fabric_delivered_wednesday

def cost_per_yard : ℕ := 2
def total_earnings : ℕ := total_fabric_delivered * cost_per_yard

-- Proposition to be proved
theorem daniel_total_earnings : total_earnings = 140 := by
  sorry

end daniel_total_earnings_l18_1828


namespace proof_of_problem_statement_l18_1808

noncomputable def problem_statement : Prop :=
  ∀ (k : ℝ) (m : ℝ),
    (0 < m ∧ m < 3/2) → 
    (-3/(4 * m) = k) → 
    (k < -1/2)

theorem proof_of_problem_statement : problem_statement :=
  sorry

end proof_of_problem_statement_l18_1808


namespace smallest_n_for_congruence_l18_1841

theorem smallest_n_for_congruence :
  ∃ n : ℕ, n > 0 ∧ 7 ^ n % 4 = n ^ 7 % 4 ∧ ∀ m : ℕ, (m > 0 ∧ m < n → ¬ (7 ^ m % 4 = m ^ 7 % 4)) :=
by
  sorry

end smallest_n_for_congruence_l18_1841


namespace nonneg_real_inequality_l18_1833

theorem nonneg_real_inequality (a b c : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) :
    a^4 + b^4 + c^4 - 2 * (a^2 * b^2 + a^2 * c^2 + b^2 * c^2) + a^2 * b * c + b^2 * a * c + c^2 * a * b ≥ 0 :=
sorry

end nonneg_real_inequality_l18_1833


namespace total_clothing_ironed_l18_1872

-- Definitions based on conditions
def shirts_per_hour := 4
def pants_per_hour := 3
def hours_ironing_shirts := 3
def hours_ironing_pants := 5

-- Theorem statement based on the problem and its solution
theorem total_clothing_ironed : 
  (shirts_per_hour * hours_ironing_shirts + pants_per_hour * hours_ironing_pants) = 27 := 
by
  sorry

end total_clothing_ironed_l18_1872


namespace acres_used_for_corn_l18_1875

-- Define the conditions
def total_acres : ℝ := 5746
def ratio_beans : ℝ := 7.5
def ratio_wheat : ℝ := 3.2
def ratio_corn : ℝ := 5.6
def total_parts : ℝ := ratio_beans + ratio_wheat + ratio_corn

-- Define the statement to prove
theorem acres_used_for_corn : (total_acres / total_parts) * ratio_corn = 1975.46 :=
by
  -- Placeholder for the proof; to be completed separately
  sorry

end acres_used_for_corn_l18_1875


namespace total_miles_l18_1856

theorem total_miles (miles_Darius : Int) (miles_Julia : Int) (h1 : miles_Darius = 679) (h2 : miles_Julia = 998) :
  miles_Darius + miles_Julia = 1677 :=
by
  sorry

end total_miles_l18_1856


namespace largest_divisor_of_n_given_n_squared_divisible_by_72_l18_1897

theorem largest_divisor_of_n_given_n_squared_divisible_by_72 (n : ℕ) (h1 : 0 < n) (h2 : 72 ∣ n^2) :
  ∃ q, q = 12 ∧ q ∣ n :=
by
  sorry

end largest_divisor_of_n_given_n_squared_divisible_by_72_l18_1897


namespace yards_after_8_marathons_l18_1832

-- Define the constants and conditions
def marathon_miles := 26
def marathon_yards := 395
def yards_per_mile := 1760

-- Definition for total distance covered after 8 marathons
def total_miles := marathon_miles * 8
def total_yards := marathon_yards * 8

-- Convert the total yards into miles with remainder
def extra_miles := total_yards / yards_per_mile
def remainder_yards := total_yards % yards_per_mile

-- Prove the remainder yards is 1400
theorem yards_after_8_marathons : remainder_yards = 1400 := by
  -- Proof steps would go here
  sorry

end yards_after_8_marathons_l18_1832


namespace concentric_circles_area_difference_l18_1857

/-- Two concentric circles with radii 12 cm and 7 cm have an area difference of 95π cm² between them. -/
theorem concentric_circles_area_difference :
  let r1 := 12
  let r2 := 7
  let area_larger := Real.pi * r1^2
  let area_smaller := Real.pi * r2^2
  let area_difference := area_larger - area_smaller
  area_difference = 95 * Real.pi := by
sorry

end concentric_circles_area_difference_l18_1857


namespace compare_neg_fractions_l18_1859

theorem compare_neg_fractions : (- (2 / 3) < - (1 / 2)) :=
sorry

end compare_neg_fractions_l18_1859


namespace tan_product_pi_over_6_3_2_undefined_l18_1807

noncomputable def tan_pi_over_6 : ℝ := Real.tan (Real.pi / 6)
noncomputable def tan_pi_over_3 : ℝ := Real.tan (Real.pi / 3)
noncomputable def tan_pi_over_2 : ℝ := Real.tan (Real.pi / 2)

theorem tan_product_pi_over_6_3_2_undefined :
  ∃ (x y : ℝ), Real.tan (Real.pi / 6) = x ∧ Real.tan (Real.pi / 3) = y ∧ Real.tan (Real.pi / 2) = 0 :=
by
  sorry

end tan_product_pi_over_6_3_2_undefined_l18_1807


namespace irrational_infinitely_many_approximations_l18_1818

theorem irrational_infinitely_many_approximations (x : ℝ) (hx : Irrational x) (hx_pos : 0 < x) :
  ∃ᶠ (q : ℕ) in at_top, ∃ p : ℤ, |x - p / q| < 1 / q^2 :=
sorry

end irrational_infinitely_many_approximations_l18_1818


namespace exam_combinations_l18_1840

/-- In the "$3+1+2$" examination plan in Hubei Province, 2021,
there are three compulsory subjects: Chinese, Mathematics, and English.
Candidates must choose one subject from Physics and History.
Candidates must choose two subjects from Chemistry, Biology, Ideological and Political Education, and Geography.
Prove that the total number of different combinations of examination subjects is 12.
-/
theorem exam_combinations : exists n : ℕ, n = 12 :=
by
  have compulsory_choice := 1
  have physics_history_choice := 2
  have remaining_subjects_choice := Nat.choose 4 2
  exact Exists.intro (compulsory_choice * physics_history_choice * remaining_subjects_choice) sorry

end exam_combinations_l18_1840


namespace mean_value_of_interior_angles_pentagon_l18_1810

def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem mean_value_of_interior_angles_pentagon :
  sum_of_interior_angles 5 / 5 = 108 :=
by
  sorry

end mean_value_of_interior_angles_pentagon_l18_1810


namespace sum_first_six_terms_l18_1837

variable {S : ℕ → ℝ}

theorem sum_first_six_terms (h2 : S 2 = 4) (h4 : S 4 = 6) : S 6 = 7 := 
  sorry

end sum_first_six_terms_l18_1837


namespace third_number_correct_l18_1803

-- Given that the row of Pascal's triangle with 51 numbers corresponds to the binomial coefficients of 50.
def third_number_in_51_pascal_row : ℕ := Nat.choose 50 2

-- Prove that the third number in this row of Pascal's triangle is 1225.
theorem third_number_correct : third_number_in_51_pascal_row = 1225 := 
by 
  -- Calculation part can be filled in for the full proof.
  sorry

end third_number_correct_l18_1803


namespace negation_exists_l18_1869

theorem negation_exists (a : ℝ) :
  ¬ (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ ∀ x : ℝ, x^2 - a * x + 1 ≥ 0 :=
sorry

end negation_exists_l18_1869


namespace A_infinite_l18_1839

noncomputable def f : ℝ → ℝ := sorry

def A : Set ℝ := { a : ℝ | f a > a ^ 2 }

theorem A_infinite
  (h_f_def : ∀ x : ℝ, ∃ y : ℝ, y = f x)
  (h_inequality: ∀ x : ℝ, (f x) ^ 2 ≤ 2 * x ^ 2 * f (x / 2))
  (h_A_nonempty : A ≠ ∅) :
  Set.Infinite A := 
sorry

end A_infinite_l18_1839


namespace car_distance_l18_1890

theorem car_distance 
  (speed : ℝ) 
  (time : ℝ) 
  (distance : ℝ) 
  (h_speed : speed = 160) 
  (h_time : time = 5) 
  (h_dist_formula : distance = speed * time) : 
  distance = 800 :=
by sorry

end car_distance_l18_1890


namespace equality_of_integers_l18_1831

theorem equality_of_integers (a b : ℕ) (h1 : ∀ n : ℕ, ∃ m : ℕ, m > 0 ∧ (a^m + b^m) % (a^n + b^n) = 0) : a = b :=
sorry

end equality_of_integers_l18_1831


namespace true_proposition_l18_1847

-- Define propositions p and q
variable (p q : Prop)

-- Assume p is true and q is false
axiom h1 : p
axiom h2 : ¬q

-- Prove that p ∧ ¬q is true
theorem true_proposition (p q : Prop) (h1 : p) (h2 : ¬q) : p ∧ ¬q :=
by
  sorry

end true_proposition_l18_1847


namespace solve_system_of_equations_l18_1804

theorem solve_system_of_equations :
  ∃ x y : ℝ, (2 * x - 5 * y = -1) ∧ (-4 * x + y = -7) ∧ (x = 2) ∧ (y = 1) :=
by
  -- proof omitted
  sorry

end solve_system_of_equations_l18_1804


namespace find_a_l18_1824

def star (x y : ℤ × ℤ) : ℤ × ℤ := (x.1 - y.1, x.2 + y.2)

theorem find_a :
  ∃ (a b : ℤ), 
  star (5, 2) (1, 1) = (a, b) ∧
  star (a, b) (0, 1) = (2, 5) ∧
  a = 2 :=
sorry

end find_a_l18_1824


namespace sqrt_of_expression_l18_1887

theorem sqrt_of_expression : Real.sqrt (5^2 * 7^6) = 1715 := 
by
  sorry

end sqrt_of_expression_l18_1887


namespace correct_time_fraction_l18_1822

theorem correct_time_fraction : 
  let hours_with_glitch := [5]
  let minutes_with_glitch := [5, 15, 25, 35, 45, 55]
  let total_hours := 12
  let total_minutes_per_hour := 60
  let correct_hours := total_hours - hours_with_glitch.length
  let correct_minutes := total_minutes_per_hour - minutes_with_glitch.length
  (correct_hours * correct_minutes) / (total_hours * total_minutes_per_hour) = 33 / 40 :=
by
  sorry

end correct_time_fraction_l18_1822


namespace min_value_inverse_sum_l18_1891

theorem min_value_inverse_sum (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + 2 * b = 2) :
  (1 / a + 2 / b) ≥ 9 / 2 :=
sorry

end min_value_inverse_sum_l18_1891


namespace mural_width_l18_1850

theorem mural_width (l p r c t w : ℝ) (h₁ : l = 6) (h₂ : p = 4) (h₃ : r = 1.5) (h₄ : c = 10) (h₅ : t = 192) :
  4 * 6 * w + 10 * (6 * w / 1.5) = 192 → w = 3 :=
by
  intros
  sorry

end mural_width_l18_1850


namespace correct_exponentiation_l18_1868

variable (a : ℝ)

theorem correct_exponentiation : (a^2)^3 = a^6 := by
  sorry

end correct_exponentiation_l18_1868


namespace inequality_holds_l18_1851

theorem inequality_holds (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  x^4 + y^4 + 2 / (x^2 * y^2) ≥ 4 := 
by
  sorry

end inequality_holds_l18_1851


namespace find_A_d_minus_B_d_l18_1848

variable {d : ℕ} (A B : ℕ) (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2)

theorem find_A_d_minus_B_d (h₁ : d > 6) (h₂ : (d^1 * A + B) + (d^1 * A + A) = 1 * d^2 + 6 * d^1 + 2) :
  A - B = 3 :=
sorry

end find_A_d_minus_B_d_l18_1848


namespace original_solution_sugar_percentage_l18_1800

theorem original_solution_sugar_percentage :
  ∃ x : ℚ, (∀ (y : ℚ), (y = 14) → (∃ (z : ℚ), (z = 26) → (3 / 4 * x + 1 / 4 * z = y))) → x = 10 := 
  sorry

end original_solution_sugar_percentage_l18_1800


namespace solve_y_eq_l18_1826

theorem solve_y_eq :
  ∀ y: ℝ, y ≠ -1 → (y^3 - 3 * y^2) / (y^2 + 2 * y + 1) + 2 * y = -1 → 
  y = 1 / Real.sqrt 3 ∨ y = -1 / Real.sqrt 3 :=
by sorry

end solve_y_eq_l18_1826


namespace find_c_l18_1827

def p (x : ℝ) : ℝ := 3 * x - 8
def q (x : ℝ) (c : ℝ) : ℝ := 5 * x - c

theorem find_c (c : ℝ) (h : p (q 3 c) = 14) : c = 23 / 3 :=
by
  sorry

end find_c_l18_1827


namespace max_sum_of_arithmetic_sequence_l18_1864

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
(h1 : 3 * a 8 = 5 * a 13) 
(h2 : a 1 > 0)
(hS : ∀ n, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
S 20 > S 21 ∧ S 20 > S 10 ∧ S 20 > S 11 :=
sorry

end max_sum_of_arithmetic_sequence_l18_1864


namespace compute_x_y_power_sum_l18_1844

noncomputable def pi : ℝ := Real.pi

theorem compute_x_y_power_sum
  (x y : ℝ)
  (h1 : 1 < x)
  (h2 : 1 < y)
  (h3 : (Real.log x / Real.log 2)^5 + (Real.log y / Real.log 3)^5 + 32 = 16 * (Real.log x / Real.log 2) * (Real.log y / Real.log 3)) :
  x^pi + y^pi = 2^(pi * (16:ℝ)^(1/5)) + 3^(pi * (16:ℝ)^(1/5)) :=
by
  sorry

end compute_x_y_power_sum_l18_1844


namespace random_sampling_not_in_proving_methods_l18_1858

inductive Method
| Comparison
| RandomSampling
| SyntheticAndAnalytic
| ProofByContradictionAndScaling

open Method

def proving_methods : List Method :=
  [Comparison, SyntheticAndAnalytic, ProofByContradictionAndScaling]

theorem random_sampling_not_in_proving_methods : 
  RandomSampling ∉ proving_methods :=
sorry

end random_sampling_not_in_proving_methods_l18_1858


namespace infinite_series_sum_l18_1893

/-- The sum of the infinite series ∑ 1/(n(n+3)) for n from 1 to ∞ is 7/9. -/
theorem infinite_series_sum :
  ∑' n, (1 : ℝ) / (n * (n + 3)) = 7 / 9 :=
sorry

end infinite_series_sum_l18_1893


namespace find_some_ounce_size_l18_1854

variable (x : ℕ)
variable (h_total : 122 = 6 * 5 + 4 * x + 15 * 4)

theorem find_some_ounce_size : x = 8 := by
  sorry

end find_some_ounce_size_l18_1854


namespace sum_cubes_div_product_eq_three_l18_1815

-- Given that x, y, z are non-zero real numbers and x + y + z = 3,
-- we need to prove that the possible value of (x^3 + y^3 + z^3) / xyz is 3.

theorem sum_cubes_div_product_eq_three 
  (x y z : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) 
  (hxyz_sum : x + y + z = 3) : 
  (x^3 + y^3 + z^3) / (x * y * z) = 3 :=
by
  sorry

end sum_cubes_div_product_eq_three_l18_1815


namespace kamal_marks_in_mathematics_l18_1835

def kamal_marks_english : ℕ := 96
def kamal_marks_physics : ℕ := 82
def kamal_marks_chemistry : ℕ := 67
def kamal_marks_biology : ℕ := 85
def kamal_average_marks : ℕ := 79
def kamal_number_of_subjects : ℕ := 5

theorem kamal_marks_in_mathematics :
  let total_marks := kamal_average_marks * kamal_number_of_subjects
  let total_known_marks := kamal_marks_english + kamal_marks_physics + kamal_marks_chemistry + kamal_marks_biology
  total_marks - total_known_marks = 65 :=
by
  sorry

end kamal_marks_in_mathematics_l18_1835
