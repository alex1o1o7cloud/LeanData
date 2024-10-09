import Mathlib

namespace range_of_a_l1694_169417

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = Real.exp x + a * x) ∧ (∃ x, 0 < x ∧ (DifferentiableAt ℝ f x) ∧ (deriv f x = 0)) → a < -1 :=
by
  sorry

end range_of_a_l1694_169417


namespace same_solution_eq_l1694_169423

theorem same_solution_eq (a b : ℤ) (x y : ℤ) 
  (h₁ : 4 * x + 3 * y = 11)
  (h₂ : a * x + b * y = -2)
  (h₃ : 3 * x - 5 * y = 1)
  (h₄ : b * x - a * y = 6) :
  (a + b) ^ 2023 = 0 := by
  sorry

end same_solution_eq_l1694_169423


namespace quadratic_inequality_solution_l1694_169426

theorem quadratic_inequality_solution (a b : ℝ)
  (h1 : ∀ x, (x > -1 ∧ x < 2) ↔ ax^2 + x + b > 0) :
  a + b = 1 :=
sorry

end quadratic_inequality_solution_l1694_169426


namespace cosine_midline_l1694_169478

theorem cosine_midline (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_range : ∀ x, 1 ≤ a * Real.cos (b * x + c) + d ∧ a * Real.cos (b * x + c) + d ≤ 5) : 
  d = 3 := 
by 
  sorry

end cosine_midline_l1694_169478


namespace find_a_l1694_169418

theorem find_a (a : ℝ) (h_pos : 0 < a) 
(h : a + a^2 = 6) : a = 2 :=
sorry

end find_a_l1694_169418


namespace probability_of_y_gt_2x_l1694_169443

noncomputable def probability_y_gt_2x : ℝ := 
  (∫ x in (0:ℝ)..(1000:ℝ), ∫ y in (2*x)..(2000:ℝ), (1 / (1000 * 2000) : ℝ)) * (1000 * 2000)

theorem probability_of_y_gt_2x : probability_y_gt_2x = 0.5 := sorry

end probability_of_y_gt_2x_l1694_169443


namespace tangent_line_equation_l1694_169432

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  y₀ = 0 →
  m = 3 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = 3 * x - 3) :=
by
  intros x₀ y₀ m h₀ hm x y
  sorry

end tangent_line_equation_l1694_169432


namespace polynomial_problem_l1694_169405

theorem polynomial_problem :
  ∀ P : Polynomial ℤ,
    (∃ R : Polynomial ℤ, (X^2 + 6*X + 10) * P^2 - 1 = R^2) → 
    P = 0 :=
by { sorry }

end polynomial_problem_l1694_169405


namespace element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l1694_169409

-- defining element, nuclide, and valence based on protons, neutrons, and electrons
def Element (protons : ℕ) := protons
def Nuclide (protons neutrons : ℕ) := (protons, neutrons)
def ChemicalProperties (outermostElectrons : ℕ) := outermostElectrons
def HighestPositiveValence (mainGroupNum : ℕ) := mainGroupNum

-- The proof problems as Lean theorems
theorem element_type_determined_by_protons (protons : ℕ) :
  Element protons = protons := sorry

theorem nuclide_type_determined_by_protons_neutrons (protons neutrons : ℕ) :
  Nuclide protons neutrons = (protons, neutrons) := sorry

theorem chemical_properties_determined_by_outermost_electrons (outermostElectrons : ℕ) :
  ChemicalProperties outermostElectrons = outermostElectrons := sorry
  
theorem highest_positive_valence_determined_by_main_group_num (mainGroupNum : ℕ) :
  HighestPositiveValence mainGroupNum = mainGroupNum := sorry

end element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l1694_169409


namespace flower_total_l1694_169425

theorem flower_total (H C D : ℕ) (h1 : H = 34) (h2 : H = C - 13) (h3 : C = D + 23) : 
  H + C + D = 105 :=
by 
  sorry  -- Placeholder for the proof

end flower_total_l1694_169425


namespace boxes_in_pantry_l1694_169498

theorem boxes_in_pantry (b p c: ℕ) (h: p = 100) (hc: c = 50) (g: b = 225) (weeks: ℕ) (consumption: ℕ)
    (total_birdseed: ℕ) (new_boxes: ℕ) (initial_boxes: ℕ) : 
    weeks = 12 → consumption = (100 + 50) * weeks → total_birdseed = 1800 →
    new_boxes = 3 → total_birdseed = b * 8 → initial_boxes = 5 :=
by
  sorry

end boxes_in_pantry_l1694_169498


namespace calc_expression_solve_system_inequalities_l1694_169465

-- Proof Problem 1: Calculation
theorem calc_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2 / 3) ^ (-2 : ℤ) = -5 / 4 := 
by 
  sorry

-- Proof Problem 2: System of Inequalities Solution
variable (m : ℝ)
variable (x : ℝ)
  
theorem solve_system_inequalities (h : m < 0) : 
  (4 * x - 1 > x - 7) ∧ (-1 / 4 * x < 3 / 2 * m - 1) → x > 4 - 6 * m := 
by 
  sorry

end calc_expression_solve_system_inequalities_l1694_169465


namespace diamond_two_three_l1694_169449

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l1694_169449


namespace no_unique_day_in_august_l1694_169493

def july_has_five_tuesdays (N : ℕ) : Prop :=
  ∃ (d : ℕ), ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30

def july_august_have_30_days (N : ℕ) : Prop :=
  true -- We're asserting this unconditionally since both months have exactly 30 days in the problem

theorem no_unique_day_in_august (N : ℕ) (h1 : july_has_five_tuesdays N) (h2 : july_august_have_30_days N) :
  ¬(∃ d : ℕ, ∀ k : ℕ, k < 5 → (d + k * 7) ≤ 30 ∧ ∃! wday : ℕ, (d + k * 7 + wday) % 7 = 0) :=
sorry

end no_unique_day_in_august_l1694_169493


namespace polynomial_expansion_l1694_169420

-- Definitions of the polynomials
def p (w : ℝ) : ℝ := 3 * w^3 + 4 * w^2 - 7
def q (w : ℝ) : ℝ := 2 * w^3 - 3 * w^2 + 1

-- Statement of the theorem
theorem polynomial_expansion (w : ℝ) : 
  (p w) * (q w) = 6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 :=
by
  sorry

end polynomial_expansion_l1694_169420


namespace jed_speed_l1694_169408

theorem jed_speed
  (posted_speed_limit : ℕ := 50)
  (fine_per_mph_over_limit : ℕ := 16)
  (red_light_fine : ℕ := 75)
  (cellphone_fine : ℕ := 120)
  (parking_fine : ℕ := 50)
  (total_red_light_fines : ℕ := 2 * red_light_fine)
  (total_parking_fines : ℕ := 3 * parking_fine)
  (total_fine : ℕ := 1046)
  (non_speeding_fines : ℕ := total_red_light_fines + cellphone_fine + total_parking_fines)
  (speeding_fine : ℕ := total_fine - non_speeding_fines)
  (mph_over_limit : ℕ := speeding_fine / fine_per_mph_over_limit):
  (posted_speed_limit + mph_over_limit) = 89 :=
by
  sorry

end jed_speed_l1694_169408


namespace proof_A_cap_complement_B_l1694_169445

variable (A B U : Set ℕ) (h1 : A ⊆ U) (h2 : B ⊆ U)
variable (h3 : U = {1, 2, 3, 4})
variable (h4 : (U \ (A ∪ B)) = {4}) -- \ represents set difference, complement in the universal set
variable (h5 : B = {1, 2})

theorem proof_A_cap_complement_B : A ∩ (U \ B) = {3} := by
  sorry

end proof_A_cap_complement_B_l1694_169445


namespace factor_polynomial_l1694_169475

theorem factor_polynomial : ∀ y : ℝ, 3 * y^2 - 27 = 3 * (y + 3) * (y - 3) :=
by
  intros y
  sorry

end factor_polynomial_l1694_169475


namespace fractions_non_integer_l1694_169488

theorem fractions_non_integer (a b c d : ℤ) : 
  ∃ (a b c d : ℤ), 
    ¬((a-b) % 2 = 0 ∧ 
      (b-c) % 2 = 0 ∧ 
      (c-d) % 2 = 0 ∧ 
      (d-a) % 2 = 0) :=
sorry

end fractions_non_integer_l1694_169488


namespace train_speed_km_per_hr_l1694_169486

-- Definitions for the conditions
def length_of_train_meters : ℕ := 250
def time_to_cross_pole_seconds : ℕ := 10

-- Conversion factors
def meters_to_kilometers (m : ℕ) : ℚ := m / 1000
def seconds_to_hours (s : ℕ) : ℚ := s / 3600

-- Theorem stating that the speed of the train is 90 km/hr
theorem train_speed_km_per_hr : 
  meters_to_kilometers length_of_train_meters / seconds_to_hours time_to_cross_pole_seconds = 90 := 
by 
  -- We skip the actual proof with sorry
  sorry

end train_speed_km_per_hr_l1694_169486


namespace range_of_p_l1694_169483

-- Conditions: p is a prime number and the roots of the quadratic equation are integers 
def p_is_prime (p : ℕ) : Prop := Nat.Prime p

def roots_are_integers (p : ℕ) : Prop :=
  ∃ x y : ℤ, x ≠ y ∧ x * y = -204 * p ∧ (x + y) = p

-- Main statement: Prove the range of p
theorem range_of_p (p : ℕ) (hp : p_is_prime p) (hr : roots_are_integers p) : 11 < p ∧ p ≤ 21 :=
  sorry

end range_of_p_l1694_169483


namespace sum_of_digits_l1694_169407

variable {w x y z : ℕ}

theorem sum_of_digits :
  (w + x + y + z = 20) ∧ w ≠ x ∧ w ≠ y ∧ w ≠ z ∧ x ≠ y ∧ x ≠ z ∧ y ≠ z →
  (y + w = 11) ∧ (x + y = 9) ∧ (w + z = 10) :=
by
  sorry

end sum_of_digits_l1694_169407


namespace eggs_sally_bought_is_correct_l1694_169499

def dozen := 12

def eggs_sally_bought (dozens : Nat) : Nat :=
  dozens * dozen

theorem eggs_sally_bought_is_correct :
  eggs_sally_bought 4 = 48 :=
by
  sorry

end eggs_sally_bought_is_correct_l1694_169499


namespace shelves_filled_l1694_169471

theorem shelves_filled (total_teddy_bears teddy_bears_per_shelf : ℕ) (h1 : total_teddy_bears = 98) (h2 : teddy_bears_per_shelf = 7) : 
  total_teddy_bears / teddy_bears_per_shelf = 14 := 
by 
  sorry

end shelves_filled_l1694_169471


namespace complete_square_transform_l1694_169437

theorem complete_square_transform (x : ℝ) :
  x^2 - 8 * x + 2 = 0 → (x - 4)^2 = 14 :=
by
  intro h
  sorry

end complete_square_transform_l1694_169437


namespace x_quad_greater_l1694_169470

theorem x_quad_greater (x : ℝ) : x^4 > x - 1/2 :=
sorry

end x_quad_greater_l1694_169470


namespace solve_for_x2_minus_y2_minus_z2_l1694_169415

theorem solve_for_x2_minus_y2_minus_z2
  (x y z : ℝ)
  (h1 : x + y + z = 12)
  (h2 : x - y = 4)
  (h3 : y + z = 7) :
  x^2 - y^2 - z^2 = -12 :=
by
  sorry

end solve_for_x2_minus_y2_minus_z2_l1694_169415


namespace intersection_is_empty_l1694_169424

open Finset

namespace ComplementIntersection

-- Define the universal set U, sets M and N
def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {2, 4, 5}

-- The complement of M with respect to U
def complement_U_M : Finset ℕ := U \ M

-- The complement of N with respect to U
def complement_U_N : Finset ℕ := U \ N

-- The intersection of the complements
def intersection_complements : Finset ℕ := complement_U_M ∩ complement_U_N

-- The proof statement
theorem intersection_is_empty : intersection_complements = ∅ :=
by sorry

end ComplementIntersection

end intersection_is_empty_l1694_169424


namespace range_m_inequality_l1694_169494

theorem range_m_inequality (m : ℝ) : 
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → x^2 * Real.exp x < m) ↔ m > Real.exp 1 := 
  by
    sorry

end range_m_inequality_l1694_169494


namespace A_can_finish_remaining_work_in_4_days_l1694_169434

theorem A_can_finish_remaining_work_in_4_days
  (A_days : ℕ) (B_days : ℕ) (B_worked_days : ℕ) : 
  A_days = 12 → B_days = 15 → B_worked_days = 10 → 
  (4 * (1 / A_days) = 1 / 3 - B_worked_days * (1 / B_days)) :=
by
  intros hA hB hBwork
  sorry

end A_can_finish_remaining_work_in_4_days_l1694_169434


namespace self_employed_tax_amount_l1694_169477

-- Definitions for conditions
def gross_income : ℝ := 350000.0

def tax_rate_self_employed : ℝ := 0.06

-- Statement asserting the tax amount for self-employed individuals given the conditions
theorem self_employed_tax_amount :
  gross_income * tax_rate_self_employed = 21000.0 := by
  sorry

end self_employed_tax_amount_l1694_169477


namespace soda_preference_respondents_l1694_169416

noncomputable def fraction_of_soda (angle_soda : ℝ) (total_angle : ℝ) : ℝ :=
  angle_soda / total_angle

noncomputable def number_of_soda_preference (total_people : ℕ) (fraction : ℝ) : ℝ :=
  total_people * fraction

theorem soda_preference_respondents (total_people : ℕ) (angle_soda : ℝ) (total_angle : ℝ) : 
  total_people = 520 → angle_soda = 298 → total_angle = 360 → 
  number_of_soda_preference total_people (fraction_of_soda angle_soda total_angle) = 429 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold fraction_of_soda number_of_soda_preference
  -- further calculation steps
  sorry

end soda_preference_respondents_l1694_169416


namespace remaining_movie_time_l1694_169485

def start_time := 200 -- represents 3:20 pm in total minutes from midnight
def end_time := 350 -- represents 5:44 pm in total minutes from midnight
def total_movie_duration := 180 -- 3 hours in minutes

theorem remaining_movie_time : total_movie_duration - (end_time - start_time) = 36 :=
by
  sorry

end remaining_movie_time_l1694_169485


namespace percent_pelicans_non_swans_l1694_169495

noncomputable def percent_geese := 0.20
noncomputable def percent_swans := 0.30
noncomputable def percent_herons := 0.10
noncomputable def percent_ducks := 0.25
noncomputable def percent_pelicans := 0.15

theorem percent_pelicans_non_swans :
  (percent_pelicans / (1 - percent_swans)) * 100 = 21.43 := 
by 
  sorry

end percent_pelicans_non_swans_l1694_169495


namespace f_eq_for_neg_l1694_169467

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x * (2^(-x) + 1) else x * (2^x + 1)

-- Theorem to prove
theorem f_eq_for_neg (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, 0 ≤ x → f x = x * (2^(-x) + 1)) :
  ∀ x : ℝ, x < 0 → f x = x * (2^x + 1) :=
by
  intro x hx
  sorry

end f_eq_for_neg_l1694_169467


namespace find_original_number_l1694_169427

theorem find_original_number (x : ℤ) (h : (x + 19) % 25 = 0) : x = 6 :=
sorry

end find_original_number_l1694_169427


namespace prime_quadratic_roots_l1694_169463

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_integer_roots (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, (a * x * x + b * x + c = 0) ∧ (a * y * y + b * y + c = 0)

theorem prime_quadratic_roots (p : ℕ) (h_prime : is_prime p)
  (h_roots : has_integer_roots 1 (p : ℤ) (-444 * (p : ℤ))) :
  31 < p ∧ p ≤ 41 :=
sorry

end prime_quadratic_roots_l1694_169463


namespace rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l1694_169444

-- Define the digit constraints and the RD sum function
def is_digit (n : ℕ) : Prop := n < 10
def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def rd_sum (A B C D : ℕ) : ℕ :=
  let abcd := 1000 * A + 100 * B + 10 * C + D
  let dcba := 1000 * D + 100 * C + 10 * B + A
  abcd + dcba

-- Problem (a)
theorem rd_sum_4281 : rd_sum 4 2 8 1 = 6105 := sorry

-- Problem (b)
theorem rd_sum_formula (A B C D : ℕ) (hA : is_nonzero_digit A) (hD : is_nonzero_digit D) :
  ∃ m n, m = 1001 ∧ n = 110 ∧ rd_sum A B C D = m * (A + D) + n * (B + C) :=
  sorry

-- Problem (c)
theorem rd_sum_count_3883 :
  ∃ n, n = 18 ∧ ∃ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D ∧ rd_sum A B C D = 3883 :=
  sorry

-- Problem (d)
theorem count_self_equal_rd_sum : 
  ∃ n, n = 143 ∧ ∀ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D → (1001 * (A + D) + 110 * (B + C) ≤ 9999 → (1000 * A + 100 * B + 10 * C + D = rd_sum A B C D → 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ D ∧ D ≤ 9)) :=
  sorry

end rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l1694_169444


namespace correct_calculation_l1694_169402

noncomputable def option_A : Prop := (Real.sqrt 3 + Real.sqrt 2) ≠ Real.sqrt 5
noncomputable def option_B : Prop := (Real.sqrt 3 * Real.sqrt 5) = Real.sqrt 15 ∧ Real.sqrt 15 ≠ 15
noncomputable def option_C : Prop := Real.sqrt (32 / 8) = 2 ∧ (Real.sqrt (32 / 8) ≠ -2)
noncomputable def option_D : Prop := (2 * Real.sqrt 3) - Real.sqrt 3 = Real.sqrt 3

theorem correct_calculation : option_D :=
by
  sorry

end correct_calculation_l1694_169402


namespace Jean_average_speed_correct_l1694_169431

noncomputable def Jean_avg_speed_until_meet
    (total_distance : ℕ)
    (chantal_flat_distance : ℕ)
    (chantal_flat_speed : ℕ)
    (chantal_steep_distance : ℕ)
    (chantal_steep_ascend_speed : ℕ)
    (chantal_steep_descend_distance : ℕ)
    (chantal_steep_descend_speed : ℕ)
    (jean_meet_position_ratio : ℚ) : ℚ :=
  let chantal_flat_time := (chantal_flat_distance : ℚ) / chantal_flat_speed
  let chantal_steep_ascend_time := (chantal_steep_distance : ℚ) / chantal_steep_ascend_speed
  let chantal_steep_descend_time := (chantal_steep_descend_distance : ℚ) / chantal_steep_descend_speed
  let total_time_until_meet := chantal_flat_time + chantal_steep_ascend_time + chantal_steep_descend_time
  let jean_distance_until_meet := (jean_meet_position_ratio * chantal_steep_distance : ℚ) + chantal_flat_distance
  jean_distance_until_meet / total_time_until_meet

theorem Jean_average_speed_correct :
  Jean_avg_speed_until_meet 6 3 5 3 3 1 4 (1 / 3) = 80 / 37 :=
by
  sorry

end Jean_average_speed_correct_l1694_169431


namespace factorize_expression_l1694_169406

variable {R : Type} [CommRing R] (a x y : R)

theorem factorize_expression :
  a^2 * (x - y) + 9 * (y - x) = (x - y) * (a + 3) * (a - 3) :=
by
  sorry

end factorize_expression_l1694_169406


namespace num_divisors_of_30_l1694_169442

theorem num_divisors_of_30 : 
  (∀ n : ℕ, n > 0 → (30 = 2^1 * 3^1 * 5^1) → (∀ k : ℕ, 0 < k ∧ k ∣ 30 → ∃ m : ℕ, k = 2^m ∧ k ∣ 30)) → 
  ∃ num_divisors : ℕ, num_divisors = 8 := 
by 
  sorry

end num_divisors_of_30_l1694_169442


namespace neither_sufficient_nor_necessary_l1694_169411

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a ≠ 5) (h2 : b ≠ -5) : ¬((a + b ≠ 0) ↔ (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end neither_sufficient_nor_necessary_l1694_169411


namespace find_integer_l1694_169454

theorem find_integer (n : ℤ) (h1 : n + 10 > 11) (h2 : -4 * n > -12) : 
  n = 2 :=
sorry

end find_integer_l1694_169454


namespace value_of_expression_l1694_169469

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : a^2 - b^2 + 6 * b = 9 :=
by
  sorry

end value_of_expression_l1694_169469


namespace painter_total_cost_l1694_169421

-- Define the arithmetic sequence for house addresses
def south_side_arith_seq (n : ℕ) : ℕ := 5 + (n - 1) * 7
def north_side_arith_seq (n : ℕ) : ℕ := 6 + (n - 1) * 8

-- Define the counting of digits
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

-- Define the condition of painting cost for multiples of 10
def painting_cost (n : ℕ) : ℕ :=
  if n % 10 = 0 then 2 * digit_count n
  else digit_count n

-- Calculate total cost for side with given arithmetic sequence
def total_cost_for_side (side_arith_seq : ℕ → ℕ): ℕ :=
  List.range 25 |>.map (λ n => painting_cost (side_arith_seq (n + 1))) |>.sum

-- Main theorem to prove
theorem painter_total_cost : total_cost_for_side south_side_arith_seq + total_cost_for_side north_side_arith_seq = 147 := by
  sorry

end painter_total_cost_l1694_169421


namespace find_x_l1694_169455

theorem find_x (x : ℝ) (h : (x * 74) / 30 = 1938.8) : x = 786 := by
  sorry

end find_x_l1694_169455


namespace total_volume_of_quiche_l1694_169453

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end total_volume_of_quiche_l1694_169453


namespace scooter_gain_percent_l1694_169458

theorem scooter_gain_percent 
  (purchase_price : ℝ) (repair_costs : ℝ) (selling_price : ℝ) 
  (h1 : purchase_price = 800) (h2 : repair_costs = 200) (h3 : selling_price = 1200) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 = 20 :=
by
  sorry

end scooter_gain_percent_l1694_169458


namespace range_of_BC_in_triangle_l1694_169481

theorem range_of_BC_in_triangle 
  (A B C : ℝ) 
  (a c BC : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : a * Real.cos C = c * Real.sin A)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : BC = 2 * Real.sin A)
  (h5 : ∃ A1 A2, 0 < A1 ∧ A1 < Real.pi / 2 ∧ Real.pi / 2 < A2 ∧ A2 < Real.pi ∧ Real.sin A = Real.sin A1 ∧ Real.sin A = Real.sin A2)
  : BC ∈ Set.Ioo (Real.sqrt 2) 2 :=
sorry

end range_of_BC_in_triangle_l1694_169481


namespace matt_homework_time_l1694_169433

variable (T : ℝ)
variable (h_math : 0.30 * T = math_time)
variable (h_science : 0.40 * T = science_time)
variable (h_others : math_time + science_time + 45 = T)

theorem matt_homework_time (h_math : 0.30 * T = math_time)
                             (h_science : 0.40 * T = science_time)
                             (h_others : math_time + science_time + 45 = T) :
  T = 150 := by
  sorry

end matt_homework_time_l1694_169433


namespace inequality_holds_l1694_169450

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l1694_169450


namespace permutations_behind_Alice_l1694_169476

theorem permutations_behind_Alice (n : ℕ) (h : n = 7) : 
  (Nat.factorial n) = 5040 :=
by
  rw [h]
  rw [Nat.factorial]
  sorry

end permutations_behind_Alice_l1694_169476


namespace sum_arithmetic_sequence_max_l1694_169410

theorem sum_arithmetic_sequence_max (d : ℝ) (a : ℕ → ℝ) 
  (h1 : d < 0) (h2 : (a 1)^2 = (a 13)^2) :
  ∃ n, n = 6 ∨ n = 7 :=
by
  sorry

end sum_arithmetic_sequence_max_l1694_169410


namespace kite_diagonals_sum_l1694_169428

theorem kite_diagonals_sum (a b e f : ℝ) (h₁ : a ≥ b) 
    (h₂ : e < 2 * a) (h₃ : f < a + b) : 
    e + f < 2 * a + b := by 
    sorry

end kite_diagonals_sum_l1694_169428


namespace blue_square_area_percentage_l1694_169490

theorem blue_square_area_percentage (k : ℝ) (H1 : 0 < k) 
(Flag_area : ℝ := k^2) -- total area of the flag
(Cross_area : ℝ := 0.49 * Flag_area) -- total area of the cross and blue squares 
(one_blue_square_area : ℝ := Cross_area / 3) -- area of one blue square
(percentage : ℝ := one_blue_square_area / Flag_area * 100) :
percentage = 16.33 :=
by
  sorry

end blue_square_area_percentage_l1694_169490


namespace solve_for_r_l1694_169439

theorem solve_for_r (r : ℤ) : 24 - 5 = 3 * r + 7 → r = 4 :=
by
  intro h
  sorry

end solve_for_r_l1694_169439


namespace volume_of_sphere_l1694_169435

theorem volume_of_sphere
    (area1 : ℝ) (area2 : ℝ) (distance : ℝ)
    (h1 : area1 = 9 * π)
    (h2 : area2 = 16 * π)
    (h3 : distance = 1) :
    ∃ R : ℝ, (4 / 3) * π * R ^ 3 = 500 * π / 3 :=
by
  sorry

end volume_of_sphere_l1694_169435


namespace solution_triples_l1694_169492

noncomputable def find_triples (x y z : ℝ) : Prop :=
  x + y + z = 2008 ∧
  x^2 + y^2 + z^2 = 6024^2 ∧
  (1/x) + (1/y) + (1/z) = 1/2008

theorem solution_triples :
  ∃ (x y z : ℝ), find_triples x y z ∧ (x = 2008 ∧ y = 4016 ∧ z = -4016) :=
sorry

end solution_triples_l1694_169492


namespace find_y_l1694_169491

theorem find_y (x y : ℤ) (h1 : 2 * x - y = 11) (h2 : 4 * x + y ≠ 17) : y = -9 :=
by sorry

end find_y_l1694_169491


namespace max_value_of_n_l1694_169438

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variable (S_2015_pos : S 2015 > 0)
variable (S_2016_neg : S 2016 < 0)

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (S_2015_pos : S 2015 > 0)
  (S_2016_neg : S 2016 < 0) : 
  ∃ n, n = 1008 ∧ ∀ m, S m < S n := 
sorry

end max_value_of_n_l1694_169438


namespace triangle_balls_l1694_169473

theorem triangle_balls (n : ℕ) (num_tri_balls : ℕ) (num_sq_balls : ℕ) :
  (∀ n : ℕ, num_tri_balls = n * (n + 1) / 2)
  ∧ (num_sq_balls = num_tri_balls + 424)
  ∧ (∀ s : ℕ, s = n - 8 → s * s = num_sq_balls)
  → num_tri_balls = 820 :=
by sorry

end triangle_balls_l1694_169473


namespace people_believing_mostly_purple_l1694_169462

theorem people_believing_mostly_purple :
  ∀ (total : ℕ) (mostly_pink : ℕ) (both_mostly_pink_purple : ℕ) (neither : ℕ),
  total = 150 →
  mostly_pink = 80 →
  both_mostly_pink_purple = 40 →
  neither = 25 →
  (total - neither + both_mostly_pink_purple - mostly_pink) = 85 :=
by
  intros total mostly_pink both_mostly_pink_purple neither h_total h_mostly_pink h_both h_neither
  have people_identified_without_mostly_purple : ℕ := mostly_pink + both_mostly_pink_purple - mostly_pink + neither
  have leftover_people : ℕ := total - people_identified_without_mostly_purple
  have people_mostly_purple := both_mostly_pink_purple + leftover_people
  suffices people_mostly_purple = 85 by sorry
  sorry

end people_believing_mostly_purple_l1694_169462


namespace intersection_of_lines_l1694_169414

theorem intersection_of_lines : ∃ (x y : ℝ), (9 * x - 4 * y = 30) ∧ (7 * x + y = 11) ∧ (x = 2) ∧ (y = -3) := 
by
  sorry

end intersection_of_lines_l1694_169414


namespace total_songs_performed_l1694_169468

theorem total_songs_performed (lucy_songs : ℕ) (sarah_songs : ℕ) (beth_songs : ℕ) (jane_songs : ℕ) 
  (h1 : lucy_songs = 8)
  (h2 : sarah_songs = 5)
  (h3 : sarah_songs < beth_songs)
  (h4 : sarah_songs < jane_songs)
  (h5 : beth_songs < lucy_songs)
  (h6 : jane_songs < lucy_songs)
  (h7 : beth_songs = 6 ∨ beth_songs = 7)
  (h8 : jane_songs = 6 ∨ jane_songs = 7) :
  (lucy_songs + sarah_songs + beth_songs + jane_songs) / 3 = 9 :=
by
  sorry

end total_songs_performed_l1694_169468


namespace water_increase_factor_l1694_169487

theorem water_increase_factor 
  (initial_koolaid : ℝ := 2) 
  (initial_water : ℝ := 16) 
  (evaporated_water : ℝ := 4) 
  (final_koolaid_percentage : ℝ := 4) : 
  (initial_water - evaporated_water) * (final_koolaid_percentage / 100) * initial_koolaid = 4 := 
by
  sorry

end water_increase_factor_l1694_169487


namespace senior_year_allowance_more_than_twice_l1694_169430

noncomputable def middle_school_allowance : ℝ :=
  8 + 2

noncomputable def twice_middle_school_allowance : ℝ :=
  2 * middle_school_allowance

noncomputable def senior_year_increase : ℝ :=
  1.5 * middle_school_allowance

noncomputable def senior_year_allowance : ℝ :=
  middle_school_allowance + senior_year_increase

theorem senior_year_allowance_more_than_twice : 
  senior_year_allowance = twice_middle_school_allowance + 5 :=
by
  sorry

end senior_year_allowance_more_than_twice_l1694_169430


namespace geometric_seq_sum_first_4_terms_l1694_169496

theorem geometric_seq_sum_first_4_terms (a : ℕ → ℕ) (S : ℕ → ℕ) 
  (h1 : a 1 = 1)
  (h2 : ∀ n, a (n + 1) = a n * 2)
  (h3 : ∀ n, S (n + 1) = S n + a (n + 1)) :
  S 4 = 15 :=
by
  -- The actual proof would go here.
  sorry

end geometric_seq_sum_first_4_terms_l1694_169496


namespace books_finished_correct_l1694_169464

def miles_traveled : ℕ := 6760
def miles_per_book : ℕ := 450
def books_finished (miles_traveled miles_per_book : ℕ) : ℕ :=
  miles_traveled / miles_per_book

theorem books_finished_correct :
  books_finished miles_traveled miles_per_book = 15 :=
by
  -- The steps of the proof would go here
  sorry

end books_finished_correct_l1694_169464


namespace tetrahedron_volume_l1694_169400

noncomputable def volume_of_tetrahedron (A B C O : Point) (r : ℝ) :=
  1 / 3 * (Real.sqrt (3) / 4 * 2^2 * Real.sqrt 11)

theorem tetrahedron_volume 
  (A B C O : Point)
  (side_length : ℝ)
  (surface_area : ℝ)
  (radius : ℝ)
  (h : ℝ)
  (radius_eq : radius = Real.sqrt (37 / 3))
  (side_length_eq : side_length = 2)
  (surface_area_eq : surface_area = (4 * Real.pi * radius^2))
  (sphere_surface_area_eq : surface_area = 148 * Real.pi / 3)
  (height_eq : h^2 = radius^2 - (2 / 3 * 2 * Real.sqrt 3 / 2)^2)
  (height_value_eq : h = Real.sqrt 11) :
  volume_of_tetrahedron A B C O radius = Real.sqrt 33 / 3 := sorry

end tetrahedron_volume_l1694_169400


namespace distinct_arrangements_of_beads_l1694_169451

noncomputable def factorial (n : Nat) : Nat := if h : n = 0 then 1 else n * factorial (n - 1)

theorem distinct_arrangements_of_beads : 
  ∃ (arrangements : Nat), arrangements = factorial 8 / (8 * 2) ∧ arrangements = 2520 := 
by
  -- Sorry to skip the proof, only requiring the statement.
  sorry

end distinct_arrangements_of_beads_l1694_169451


namespace cauchy_schwarz_example_l1694_169497

theorem cauchy_schwarz_example (a b c : ℝ) (ha: 0 < a) (hb: 0 < b) (hc: 0 < c) : 
  a^2 + b^2 + c^2 ≥ a * b + b * c + c * a :=
by
  sorry

end cauchy_schwarz_example_l1694_169497


namespace exists_triangle_cut_into_2005_congruent_l1694_169447

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end exists_triangle_cut_into_2005_congruent_l1694_169447


namespace average_first_8_matches_l1694_169441

/--
Assume we have the following conditions:
1. The average score for 12 matches is 48 runs.
2. The average score for the last 4 matches is 64 runs.
Prove that the average score for the first 8 matches is 40 runs.
-/
theorem average_first_8_matches (A1 A2 : ℕ) :
  (A1 / 12 = 48) → 
  (A2 / 4 = 64) →
  ((A1 - A2) / 8 = 40) :=
by
  sorry

end average_first_8_matches_l1694_169441


namespace sequence_formula_l1694_169489

theorem sequence_formula (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 5)
  (h3 : ∀ n > 1, a (n + 1) = 2 * a n - a (n - 1)) :
  ∀ n, a n = 4 * n - 3 :=
by
  sorry

end sequence_formula_l1694_169489


namespace jason_initial_speed_correct_l1694_169448

noncomputable def jason_initial_speed (d : ℝ) (t1 t2 : ℝ) (v2 : ℝ) : ℝ :=
  let t_total := t1 + t2
  let d2 := v2 * t2
  let d1 := d - d2
  let v1 := d1 / t1
  v1

theorem jason_initial_speed_correct :
  jason_initial_speed 120 0.5 1 90 = 60 := 
by 
  sorry

end jason_initial_speed_correct_l1694_169448


namespace cost_per_pound_beef_is_correct_l1694_169480

variable (budget initial_chicken_cost pounds_beef remaining_budget_after_purchase : ℝ)
variable (spending_on_beef cost_per_pound_beef : ℝ)

axiom h1 : budget = 80
axiom h2 : initial_chicken_cost = 12
axiom h3 : pounds_beef = 5
axiom h4 : remaining_budget_after_purchase = 53
axiom h5 : spending_on_beef = budget - initial_chicken_cost - remaining_budget_after_purchase
axiom h6 : cost_per_pound_beef = spending_on_beef / pounds_beef

theorem cost_per_pound_beef_is_correct : cost_per_pound_beef = 3 :=
by
  sorry

end cost_per_pound_beef_is_correct_l1694_169480


namespace fraction_reducible_l1694_169452

theorem fraction_reducible (l : ℤ) : ∃ d : ℤ, d ≠ 1 ∧ d > 0 ∧ d = gcd (5 * l + 6) (8 * l + 7) := by 
  use 13
  sorry

end fraction_reducible_l1694_169452


namespace team_savings_with_discount_l1694_169403

def regular_shirt_cost : ℝ := 7.50
def regular_pants_cost : ℝ := 15.00
def regular_socks_cost : ℝ := 4.50
def discounted_shirt_cost : ℝ := 6.75
def discounted_pants_cost : ℝ := 13.50
def discounted_socks_cost : ℝ := 3.75
def team_size : ℕ := 12

theorem team_savings_with_discount :
  let regular_uniform_cost := regular_shirt_cost + regular_pants_cost + regular_socks_cost
  let discounted_uniform_cost := discounted_shirt_cost + discounted_pants_cost + discounted_socks_cost
  let savings_per_uniform := regular_uniform_cost - discounted_uniform_cost
  let total_savings := savings_per_uniform * team_size
  total_savings = 36 := by
  sorry

end team_savings_with_discount_l1694_169403


namespace determinant_inequality_l1694_169404

open Real

def det (a b c d : ℝ) : ℝ := a * d - b * c

theorem determinant_inequality (x : ℝ) :
  det 7 (x^2) 2 1 > det 3 (-2) 1 x ↔ -5/2 < x ∧ x < 1 :=
by
  sorry

end determinant_inequality_l1694_169404


namespace part_1_part_2_part_3_l1694_169413

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3_l1694_169413


namespace bat_pattern_area_l1694_169456

-- Define the areas of the individual components
def area_large_square : ℕ := 8
def num_large_squares : ℕ := 2

def area_medium_square : ℕ := 4
def num_medium_squares : ℕ := 2

def area_triangle : ℕ := 1
def num_triangles : ℕ := 3

-- Define the total area calculation
def total_area : ℕ :=
  (num_large_squares * area_large_square) +
  (num_medium_squares * area_medium_square) +
  (num_triangles * area_triangle)

-- The theorem statement
theorem bat_pattern_area : total_area = 27 := by
  sorry

end bat_pattern_area_l1694_169456


namespace prime_factor_of_difference_l1694_169461

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA9 : A ≤ 9) (hC : 1 ≤ C) (hC9 : C ≤ 9) (hA_ne_C : A ≠ C) :
  ∃ p : ℕ, Prime p ∧ p = 3 ∧ p ∣ 3 * (100 * A + 10 * B + C - (100 * C + 10 * B + A)) := by
  sorry

end prime_factor_of_difference_l1694_169461


namespace hiking_trip_time_l1694_169459

noncomputable def R_up : ℝ := 7
noncomputable def R_down : ℝ := 1.5 * R_up
noncomputable def Distance_down : ℝ := 21
noncomputable def T_down : ℝ := Distance_down / R_down
noncomputable def T_up : ℝ := T_down

theorem hiking_trip_time :
  T_up = 2 := by
      sorry

end hiking_trip_time_l1694_169459


namespace cos_double_alpha_proof_l1694_169440

theorem cos_double_alpha_proof (α : ℝ) (h1 : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = - 7 / 9 :=
by
  sorry

end cos_double_alpha_proof_l1694_169440


namespace slices_per_large_pizza_l1694_169474

theorem slices_per_large_pizza (total_pizzas : ℕ) (slices_eaten : ℕ) (slices_remaining : ℕ) 
  (H1 : total_pizzas = 2) (H2 : slices_eaten = 7) (H3 : slices_remaining = 9) : 
  (slices_remaining + slices_eaten) / total_pizzas = 8 := 
by
  sorry

end slices_per_large_pizza_l1694_169474


namespace total_earnings_first_two_weeks_l1694_169460

-- Conditions
variable (x : ℝ)  -- Xenia's hourly wage
variable (earnings_first_week : ℝ := 12 * x)  -- Earnings in the first week
variable (earnings_second_week : ℝ := 20 * x)  -- Earnings in the second week

-- Xenia earned $36 more in the second week than in the first
axiom h1 : earnings_second_week = earnings_first_week + 36

-- Proof statement
theorem total_earnings_first_two_weeks : earnings_first_week + earnings_second_week = 144 := by
  -- Proof is omitted
  sorry

end total_earnings_first_two_weeks_l1694_169460


namespace square_area_twice_triangle_perimeter_l1694_169422

noncomputable def perimeter_of_triangle (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length * side_length

theorem square_area_twice_triangle_perimeter (a b c : ℕ) (h1 : perimeter_of_triangle a b c = 22) (h2 : a = 5) (h3 : b = 7) (h4 : c = 10) : area_of_square (side_length_of_square (2 * perimeter_of_triangle a b c)) = 121 :=
by
  sorry

end square_area_twice_triangle_perimeter_l1694_169422


namespace y_paid_per_week_l1694_169412

variable (x y z : ℝ)

-- Conditions
axiom h1 : x + y + z = 900
axiom h2 : x = 1.2 * y
axiom h3 : z = 0.8 * y

-- Theorem to prove
theorem y_paid_per_week : y = 300 := by
  sorry

end y_paid_per_week_l1694_169412


namespace distance_before_rest_l1694_169401

theorem distance_before_rest (total_distance after_rest_distance : ℝ) (h1 : total_distance = 1) (h2 : after_rest_distance = 0.25) :
  total_distance - after_rest_distance = 0.75 :=
by sorry

end distance_before_rest_l1694_169401


namespace reciprocal_neg_one_over_2023_l1694_169457

theorem reciprocal_neg_one_over_2023 : 1 / (- (1 / 2023 : ℝ)) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_l1694_169457


namespace part1_part2_l1694_169484

noncomputable def f (a b x : ℝ) : ℝ := a * x^2 + b * x + 6

theorem part1 (a b : ℝ) (h1 : f a b 1 = 8) : a + b = 2 := by
  rw [f] at h1
  sorry

theorem part2 (a b : ℝ) (h1 : f a b (-1) = f a b 3) : f a b 2 = 6 := by
  rw [f] at h1
  sorry

end part1_part2_l1694_169484


namespace locus_centers_of_circles_l1694_169429

theorem locus_centers_of_circles (P : ℝ × ℝ) (a : ℝ) (a_pos : 0 < a):
  {O : ℝ × ℝ | dist O P = a} = {O : ℝ × ℝ | dist O P = a} :=
by
  sorry

end locus_centers_of_circles_l1694_169429


namespace calculate_total_cost_l1694_169466

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

theorem calculate_total_cost :
  let total_items := num_sandwiches + num_sodas
  let cost_before_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  let discount := if total_items > discount_threshold then cost_before_discount * discount_rate else 0
  let final_cost := cost_before_discount - discount
  final_cost = 38.7 :=
by
  sorry

end calculate_total_cost_l1694_169466


namespace min_knights_l1694_169482

noncomputable def is_lying (n : ℕ) (T : ℕ → Prop) (p : ℕ → Prop) : Prop :=
    (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m ∧ m < n))

open Nat

def islanders_condition (T : ℕ → Prop) (p : ℕ → Prop) :=
  ∀ n, n < 80 → (T n ∨ ¬T n) ∧ (T n → ∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m)) ∨ (¬T n → ¬∃ m, (m ≠ n) ∧ ¬p n → (m > n ∧ T m))

theorem min_knights : ∀ (T : ℕ → Prop) (p : ℕ → Prop), islanders_condition T p → ∃ k, k = 70 :=    
by
    sorry

end min_knights_l1694_169482


namespace fraction_one_two_three_sum_l1694_169472

def fraction_one_bedroom : ℝ := 0.12
def fraction_two_bedroom : ℝ := 0.26
def fraction_three_bedroom : ℝ := 0.38
def fraction_four_bedroom : ℝ := 0.24

theorem fraction_one_two_three_sum :
  fraction_one_bedroom + fraction_two_bedroom + fraction_three_bedroom = 0.76 :=
by
  sorry

end fraction_one_two_three_sum_l1694_169472


namespace distance_between_A_and_B_l1694_169419

theorem distance_between_A_and_B
  (vA vB D : ℝ)
  (hvB : vB = (3/2) * vA)
  (second_meeting_distance : 20 = D * 2 / 5) : 
  D = 50 := 
by
  sorry

end distance_between_A_and_B_l1694_169419


namespace value_divided_by_3_l1694_169446

-- Given condition
def given_condition (x : ℕ) : Prop := x - 39 = 54

-- Correct answer we need to prove
theorem value_divided_by_3 (x : ℕ) (h : given_condition x) : x / 3 = 31 := 
by
  sorry

end value_divided_by_3_l1694_169446


namespace cost_of_soap_per_year_l1694_169436

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l1694_169436


namespace bricks_needed_l1694_169479

theorem bricks_needed 
    (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ) 
    (wall_length_m : ℝ) (wall_height_m : ℝ) (wall_width_cm : ℝ)
    (H1 : brick_length = 25) (H2 : brick_width = 11.25) (H3 : brick_height = 6)
    (H4 : wall_length_m = 7) (H5 : wall_height_m = 6) (H6 : wall_width_cm = 22.5) :
    (wall_length_m * 100 * wall_height_m * 100 * wall_width_cm) / (brick_length * brick_width * brick_height) = 5600 :=
by
    sorry

end bricks_needed_l1694_169479
