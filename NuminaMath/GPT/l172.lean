import Mathlib

namespace ensemble_average_age_l172_172015

theorem ensemble_average_age (female_avg_age : ℝ) (num_females : ℕ) (male_avg_age : ℝ) (num_males : ℕ)
  (h1 : female_avg_age = 32) (h2 : num_females = 12) (h3 : male_avg_age = 40) (h4 : num_males = 18) :
  (num_females * female_avg_age + num_males * male_avg_age) / (num_females + num_males) =  36.8 :=
by sorry

end ensemble_average_age_l172_172015


namespace distinct_triangles_from_chord_intersections_l172_172449

theorem distinct_triangles_from_chord_intersections :
  let points := 9
  let chords := (points.choose 2)
  let intersections := (points.choose 4)
  let triangles := (points.choose 6)
  (chords > 0 ∧ intersections > 0 ∧ triangles > 0) →
  triangles = 84 :=
by
  intros
  sorry

end distinct_triangles_from_chord_intersections_l172_172449


namespace partial_fraction_product_is_correct_l172_172176

-- Given conditions
def fraction_decomposition (x A B C : ℝ) :=
  ( (x^2 + 5 * x - 14) / (x^3 - 3 * x^2 - x + 3) = A / (x - 1) + B / (x - 3) + C / (x + 1) )

-- Statement we want to prove
theorem partial_fraction_product_is_correct (A B C : ℝ) (h : ∀ x : ℝ, fraction_decomposition x A B C) :
  A * B * C = -25 / 2 :=
sorry

end partial_fraction_product_is_correct_l172_172176


namespace average_donation_is_integer_l172_172740

variable (num_classes : ℕ) (students_per_class : ℕ) (num_teachers : ℕ) (total_donation : ℕ)

def valid_students (n : ℕ) : Prop := 30 < n ∧ n ≤ 45

theorem average_donation_is_integer (h_classes : num_classes = 14)
                                    (h_teachers : num_teachers = 35)
                                    (h_donation : total_donation = 1995)
                                    (h_students_per_class : valid_students students_per_class)
                                    (h_total_people : ∃ n, 
                                      n = num_teachers + num_classes * students_per_class ∧ 30 < students_per_class ∧ students_per_class ≤ 45) :
  total_donation % (num_teachers + num_classes * students_per_class) = 0 ∧ 
  total_donation / (num_teachers + num_classes * students_per_class) = 3 := 
sorry

end average_donation_is_integer_l172_172740


namespace min_value_inequality_l172_172233

theorem min_value_inequality (a b : ℝ) (h : a * b = 1) : 4 * a^2 + 9 * b^2 ≥ 12 :=
by sorry

end min_value_inequality_l172_172233


namespace expression_value_l172_172826

theorem expression_value : (2^2003 + 5^2004)^2 - (2^2003 - 5^2004)^2 = 40 * 10^2003 := 
by
  sorry

end expression_value_l172_172826


namespace A_det_nonzero_A_inv_is_correct_l172_172389

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 4], ![2, 9]]

def A_inv : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![9, -4], ![-2, 1]]

theorem A_det_nonzero : det A ≠ 0 := 
  sorry

theorem A_inv_is_correct : A * A_inv = 1 := 
  sorry

end A_det_nonzero_A_inv_is_correct_l172_172389


namespace william_shared_marble_count_l172_172772

theorem william_shared_marble_count : ∀ (initial_marbles shared_marbles remaining_marbles : ℕ),
  initial_marbles = 10 → remaining_marbles = 7 → 
  shared_marbles = initial_marbles - remaining_marbles → 
  shared_marbles = 3 := by 
    intros initial_marbles shared_marbles remaining_marbles h_initial h_remaining h_shared
    rw [h_initial, h_remaining] at h_shared
    exact h_shared

end william_shared_marble_count_l172_172772


namespace find_a_value_l172_172763

-- Define the conditions
def inverse_variation (a b : ℝ) : Prop := ∃ k : ℝ, a * b^3 = k

-- Define the proof problem
theorem find_a_value
  (a b : ℝ)
  (h1 : inverse_variation a b)
  (h2 : a = 4)
  (h3 : b = 1) :
  ∃ a', a' = 1 / 2 ∧ inverse_variation a' 2 := 
sorry

end find_a_value_l172_172763


namespace age_ratio_l172_172508

/-- 
Axiom: Kareem's age is 42 and his son's age is 14. 
-/
axiom Kareem_age : ℕ
axiom Son_age : ℕ

/-- 
Conditions: 
  - Kareem's age after 10 years plus his son's age after 10 years equals 76.
  - Kareem's current age is 42.
  - His son's current age is 14.
-/
axiom age_condition : Kareem_age + 10 + Son_age + 10 = 76
axiom Kareem_current_age : Kareem_age = 42
axiom Son_current_age : Son_age = 14

/-- 
Theorem: The ratio of Kareem's age to his son's age is 3:1.
-/
theorem age_ratio : Kareem_age / Son_age = 3 / 1 := by {
  -- Proof skipped
  sorry 
}

end age_ratio_l172_172508


namespace quadratic_is_binomial_square_l172_172912

theorem quadratic_is_binomial_square 
  (a : ℤ) : 
  (∃ b : ℤ, 9 * (x: ℤ)^2 - 24 * x + a = (3 * x + b)^2) ↔ a = 16 := 
by 
  sorry

end quadratic_is_binomial_square_l172_172912


namespace marbles_problem_l172_172162

theorem marbles_problem (n : ℕ) :
  (n % 3 = 0 ∧ n % 4 = 0 ∧ n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0 ∧ n % 8 = 0) → 
  n - 10 = 830 :=
sorry

end marbles_problem_l172_172162


namespace Jerry_travel_time_l172_172227

theorem Jerry_travel_time
  (speed_j speed_b distance_j distance_b time_j time_b : ℝ)
  (h_speed_j : speed_j = 40)
  (h_speed_b : speed_b = 30)
  (h_distance_b : distance_b = distance_j + 5)
  (h_time_b : time_b = time_j + 1/3)
  (h_distance_j : distance_j = speed_j * time_j)
  (h_distance_b_eq : distance_b = speed_b * time_b) :
  time_j = 1/2 :=
by
  sorry

end Jerry_travel_time_l172_172227


namespace rotation_image_of_D_l172_172075

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem rotation_image_of_D :
  rotate_90_clockwise (-3, 2) = (2, 3) :=
by
  sorry

end rotation_image_of_D_l172_172075


namespace max_a_value_l172_172851

theorem max_a_value (a b c d : ℕ) (h1 : a < 3 * b) (h2 : b < 3 * c) (h3 : c < 4 * d) (h4 : b + d = 200) : a ≤ 449 :=
by sorry

end max_a_value_l172_172851


namespace union_sets_l172_172562

-- Define the sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {x : ℤ | (x + 2) * (x - 1) < 0}

-- The theorem to be proven
theorem union_sets : A ∪ B = {-1, 0, 1} :=
by
  sorry

end union_sets_l172_172562


namespace current_time_l172_172555

theorem current_time (t : ℝ) 
  (h1 : 6 * (t + 10) - (90 + 0.5 * (t - 5)) = 90 ∨ 6 * (t + 10) - (90 + 0.5 * (t - 5)) = -90) :
  t = 3 + 11 / 60 := sorry

end current_time_l172_172555


namespace blue_balls_count_l172_172708

theorem blue_balls_count:
  ∀ (T : ℕ),
  (1/4 * T) + (1/8 * T) + (1/12 * T) + 26 = T → 
  (1 / 8) * T = 6 := by
  intros T h
  sorry

end blue_balls_count_l172_172708


namespace estimate_mass_of_ice_floe_l172_172691

noncomputable def mass_of_ice_floe (d : ℝ) (D : ℝ) (m : ℝ) : ℝ :=
  (m * d) / (D - d)

theorem estimate_mass_of_ice_floe :
  mass_of_ice_floe 9.5 10 600 = 11400 := 
by
  sorry

end estimate_mass_of_ice_floe_l172_172691


namespace solve_absolute_inequality_l172_172829

theorem solve_absolute_inequality (x : ℝ) : |x - 1| - |x - 2| > 1 / 2 ↔ x > 7 / 4 :=
by sorry

end solve_absolute_inequality_l172_172829


namespace equivalent_proof_problem_l172_172347

theorem equivalent_proof_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 :=
by
  sorry

end equivalent_proof_problem_l172_172347


namespace small_monkey_dolls_cheaper_than_large_l172_172413

theorem small_monkey_dolls_cheaper_than_large (S : ℕ) 
  (h1 : 300 / 6 = 50) 
  (h2 : 300 / S = 75) 
  (h3 : 75 - 50 = 25) : 
  6 - S = 2 := 
sorry

end small_monkey_dolls_cheaper_than_large_l172_172413


namespace remainder_2519_div_7_l172_172028

theorem remainder_2519_div_7 : 2519 % 7 = 6 :=
by
  sorry

end remainder_2519_div_7_l172_172028


namespace inequality_proof_l172_172652

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (1 + 4 * a / (b + c)) * (1 + 4 * b / (c + a)) * (1 + 4 * c / (a + b)) > 25 := by
  sorry

end inequality_proof_l172_172652


namespace youngest_child_age_l172_172479

theorem youngest_child_age 
  (x : ℕ)
  (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 50) : 
  x = 6 := 
by 
  sorry

end youngest_child_age_l172_172479


namespace smallest_enclosing_sphere_radius_l172_172292

-- Define the radius of each small sphere and the center set
def radius (r : ℝ) : Prop := r = 2

def center_set (C : Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ c ∈ C, ∃ x y z : ℝ, 
    (x = 2 ∨ x = -2) ∧ 
    (y = 2 ∨ y = -2) ∧ 
    (z = 2 ∨ z = -2) ∧
    (c = (x, y, z))

-- Prove the radius of the smallest enclosing sphere is 2√3 + 2
theorem smallest_enclosing_sphere_radius (r : ℝ) (C : Set (ℝ × ℝ × ℝ)) 
  (h_radius : radius r) (h_center_set : center_set C) :
  ∃ R : ℝ, R = 2 * Real.sqrt 3 + 2 :=
sorry

end smallest_enclosing_sphere_radius_l172_172292


namespace monotonicity_and_range_of_a_l172_172237

noncomputable def f (x a : ℝ) := Real.log x - a * x - 2

theorem monotonicity_and_range_of_a (a : ℝ) (h : a ≠ 0) :
  ((∀ x > 0, (Real.log x - a * x - 2) < (Real.log (x + 1) - a * (x + 1) - 2)) ↔ (a < 0)) ∧
  ((∃ M, M = Real.log (1/a) - a * (1/a) - 2 ∧ M > a - 4) → 0 < a ∧ a < 1) := sorry

end monotonicity_and_range_of_a_l172_172237


namespace find_a14_l172_172615

-- Define the arithmetic sequence properties
def sum_of_first_n_terms (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

def nth_term (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

theorem find_a14 (a1 d : ℤ) (S11 : sum_of_first_n_terms a1 d 11 = 55)
  (a10 : nth_term a1 d 10 = 9) : nth_term a1 d 14 = 13 :=
sorry

end find_a14_l172_172615


namespace games_needed_to_declare_winner_l172_172698

def single_elimination_games (T : ℕ) : ℕ :=
  T - 1

theorem games_needed_to_declare_winner (T : ℕ) :
  (single_elimination_games 23 = 22) :=
by
  sorry

end games_needed_to_declare_winner_l172_172698


namespace peter_speed_l172_172945

theorem peter_speed (p : ℝ) (v_juan : ℝ) (d : ℝ) (t : ℝ) 
  (h1 : v_juan = p + 3) 
  (h2 : d = t * p + t * v_juan) 
  (h3 : t = 1.5) 
  (h4 : d = 19.5) : 
  p = 5 :=
by
  sorry

end peter_speed_l172_172945


namespace sum_ap_series_l172_172846

-- Definition of the arithmetic progression sum for given parameters
def ap_sum (a d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

-- Specific sum calculation for given p
def S_p (p : ℕ) : ℕ :=
  ap_sum p (2 * p - 1) 40

-- Total sum from p = 1 to p = 10
def total_sum : ℕ :=
  (Finset.range 10).sum (λ i => S_p (i + 1))

-- The theorem stating the desired proof
theorem sum_ap_series : total_sum = 80200 := by
  sorry

end sum_ap_series_l172_172846


namespace find_angle_x_l172_172719

theorem find_angle_x (x : ℝ) (h1 : x + x + 140 = 360) : x = 110 :=
by
  sorry

end find_angle_x_l172_172719


namespace inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l172_172525

theorem inequality_d_over_c_lt_d_plus_4_over_c_plus_4
  (a b c d : ℝ)
  (h1 : a > b)
  (h2 : c > d)
  (h3 : d > 0) :
  (d / c) < ((d + 4) / (c + 4)) :=
by
  sorry

end inequality_d_over_c_lt_d_plus_4_over_c_plus_4_l172_172525


namespace Jessie_l172_172651

theorem Jessie's_friends (total_muffins : ℕ) (muffins_per_person : ℕ) (num_people : ℕ) :
  total_muffins = 20 → muffins_per_person = 4 → num_people = total_muffins / muffins_per_person → num_people - 1 = 4 :=
by
  intros h1 h2 h3
  sorry

end Jessie_l172_172651


namespace find_fourth_mark_l172_172812

-- Definitions of conditions
def average_of_four (a b c d : ℕ) : Prop :=
  (a + b + c + d) / 4 = 60

def known_marks (a b c : ℕ) : Prop :=
  a = 30 ∧ b = 55 ∧ c = 65

-- Theorem statement
theorem find_fourth_mark {d : ℕ} (h_avg : average_of_four 30 55 65 d) (h_known : known_marks 30 55 65) : d = 90 := 
by 
  sorry

end find_fourth_mark_l172_172812


namespace points_eq_l172_172043

-- Definition of the operation 
def star (a b : ℝ) : ℝ := a^2 * b + a * b^2

-- The property we want to prove
theorem points_eq : {p : ℝ × ℝ | star p.1 p.2 = star p.2 p.1} =
    {p : ℝ × ℝ | p.1 = 0} ∪ {p : ℝ × ℝ | p.2 = 0} ∪ {p : ℝ × ℝ | p.1 + p.2 = 0} :=
by
  sorry

end points_eq_l172_172043


namespace find_g2_l172_172891

-- Define the conditions of the problem
def satisfies_condition (g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 4 * x^3 - 5 * x

-- Prove the desired value of g(2)
theorem find_g2 (g : ℝ → ℝ) (h : satisfies_condition g) : g 2 = -19 / 6 :=
by
  sorry

end find_g2_l172_172891


namespace Liza_initial_balance_l172_172962

theorem Liza_initial_balance
  (W: Nat)   -- Liza's initial balance on Tuesday
  (rent: Nat := 450)
  (deposit: Nat := 1500)
  (electricity: Nat := 117)
  (internet: Nat := 100)
  (phone: Nat := 70)
  (final_balance: Nat := 1563) 
  (balance_eq: W - rent + deposit - electricity - internet - phone = final_balance) 
  : W = 800 :=
sorry

end Liza_initial_balance_l172_172962


namespace differential_savings_l172_172878

theorem differential_savings (income : ℝ) (tax_rate1 tax_rate2 : ℝ) 
                            (old_tax_rate_eq : tax_rate1 = 0.40) 
                            (new_tax_rate_eq : tax_rate2 = 0.33) 
                            (income_eq : income = 45000) :
    ((tax_rate1 - tax_rate2) * income) = 3150 :=
by
  rw [old_tax_rate_eq, new_tax_rate_eq, income_eq]
  norm_num

end differential_savings_l172_172878


namespace circle_tangent_proof_l172_172276

noncomputable def circle_tangent_range : Set ℝ :=
  { k : ℝ | k > 0 ∧ ((3 - 2 * k)^2 + (1 - k)^2 > k) }

theorem circle_tangent_proof :
  ∀ k > 0, ((3 - 2 * k)^2 + (1 - k)^2 > k) ↔ (k ∈ (Set.Ioo 0 1 ∪ Set.Ioi 2)) :=
by
  sorry

end circle_tangent_proof_l172_172276


namespace lattice_points_count_l172_172083

-- A definition of lattice points and bounded region
def is_lattice_point (p : ℤ × ℤ) : Prop := true

def in_region (p : ℤ × ℤ) : Prop :=
  let (x, y) := p
  (y = abs x ∨ y = -x^2 + 4*x + 6) ∧ (y ≤ abs x ∧ y ≤ -x^2 + 4*x + 6)

-- The target statement to prove
theorem lattice_points_count : ∃ n, n = 23 ∧ ∀ p : ℤ × ℤ, is_lattice_point p → in_region p := sorry

end lattice_points_count_l172_172083


namespace additional_pencils_l172_172815

theorem additional_pencils (original_pencils new_pencils per_container distributed_pencils : ℕ)
  (h1 : original_pencils = 150)
  (h2 : per_container = 5)
  (h3 : distributed_pencils = 36)
  (h4 : new_pencils = distributed_pencils * per_container) :
  (new_pencils - original_pencils) = 30 :=
by
  -- Proof will go here
  sorry

end additional_pencils_l172_172815


namespace dog_older_than_max_by_18_l172_172085

-- Definition of the conditions
def human_to_dog_years_ratio : ℕ := 7
def max_age : ℕ := 3
def dog_age_in_human_years : ℕ := 3

-- Translate the question: How much older, in dog years, will Max's dog be?
def age_difference_in_dog_years : ℕ :=
  dog_age_in_human_years * human_to_dog_years_ratio - max_age

-- The proof statement
theorem dog_older_than_max_by_18 : age_difference_in_dog_years = 18 := by
  sorry

end dog_older_than_max_by_18_l172_172085


namespace psychologist_charge_difference_l172_172657

variables (F A : ℝ)

theorem psychologist_charge_difference
  (h1 : F + 4 * A = 375)
  (h2 : F + A = 174) :
  (F - A) = 40 :=
by sorry

end psychologist_charge_difference_l172_172657


namespace ratio_of_x_l172_172886

theorem ratio_of_x (x : ℝ) (h : x = Real.sqrt 7 + Real.sqrt 6) :
    ((x + 1 / x) / (x - 1 / x)) = (Real.sqrt 7 / Real.sqrt 6) :=
by
  sorry

end ratio_of_x_l172_172886


namespace notebook_cost_l172_172130

theorem notebook_cost (n p : ℝ) (h1 : n + p = 2.40) (h2 : n = 2 + p) : n = 2.20 := by
  sorry

end notebook_cost_l172_172130


namespace eval_diamond_expr_l172_172685

def diamond (a b : ℕ) : ℕ :=
  match (a, b) with
  | (1, 1) => 4
  | (1, 2) => 3
  | (1, 3) => 2
  | (1, 4) => 1
  | (2, 1) => 1
  | (2, 2) => 4
  | (2, 3) => 3
  | (2, 4) => 2
  | (3, 1) => 2
  | (3, 2) => 1
  | (3, 3) => 4
  | (3, 4) => 3
  | (4, 1) => 3
  | (4, 2) => 2
  | (4, 3) => 1
  | (4, 4) => 4
  | (_, _) => 0  -- This handles any case outside of 1,2,3,4 which should ideally not happen

theorem eval_diamond_expr : diamond (diamond 3 4) (diamond 2 1) = 2 := by
  sorry

end eval_diamond_expr_l172_172685


namespace sum_of_extremes_of_g_l172_172017

noncomputable def g (x : ℝ) : ℝ := abs (x - 1) + abs (x - 5) - abs (2 * x - 8)

theorem sum_of_extremes_of_g :
  (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≤ g 4) ∧ (∀ x, 1 ≤ x ∧ x ≤ 10 → g x ≥ g 1) → g 4 + g 1 = 2 :=
by
  sorry

end sum_of_extremes_of_g_l172_172017


namespace percentage_enclosed_by_hexagons_is_50_l172_172854

noncomputable def hexagon_area (s : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 2) * s^2

noncomputable def square_area (s : ℝ) : ℝ :=
  s^2

noncomputable def total_tiling_unit_area (s : ℝ) : ℝ :=
  hexagon_area s + 3 * square_area s

noncomputable def percentage_enclosed_by_hexagons (s : ℝ) : ℝ :=
  (hexagon_area s / total_tiling_unit_area s) * 100

theorem percentage_enclosed_by_hexagons_is_50 (s : ℝ) : percentage_enclosed_by_hexagons s = 50 := by
  sorry

end percentage_enclosed_by_hexagons_is_50_l172_172854


namespace max_positive_n_l172_172161

def a (n : ℕ) : ℤ := 19 - 2 * n

theorem max_positive_n (n : ℕ) (h : a n > 0) : n ≤ 9 :=
by
  sorry

end max_positive_n_l172_172161


namespace sufficient_but_not_necessary_condition_l172_172849

theorem sufficient_but_not_necessary_condition (a b : ℝ) (hb : b < -1) : |a| + |b| > 1 := 
by
  sorry

end sufficient_but_not_necessary_condition_l172_172849


namespace period_tan_2x_3_l172_172097

noncomputable def period_of_tan_transformed : Real :=
  let period_tan := Real.pi
  let coeff := 2/3
  (period_tan / coeff : Real)

theorem period_tan_2x_3 : period_of_tan_transformed = 3 * Real.pi / 2 :=
  sorry

end period_tan_2x_3_l172_172097


namespace largest_base5_to_base7_l172_172583

-- Define the largest four-digit number in base-5
def largest_base5_four_digit_number : ℕ := 4 * 5^3 + 4 * 5^2 + 4 * 5^1 + 4 * 5^0

-- Convert this number to base-7
def convert_to_base7 (n : ℕ) : ℕ := 
  let d3 := n / (7^3)
  let r3 := n % (7^3)
  let d2 := r3 / (7^2)
  let r2 := r3 % (7^2)
  let d1 := r2 / (7^1)
  let r1 := r2 % (7^1)
  let d0 := r1
  (d3 * 10^3) + (d2 * 10^2) + (d1 * 10^1) + d0

-- Theorem to prove m in base-7
theorem largest_base5_to_base7 : 
  convert_to_base7 largest_base5_four_digit_number = 1551 :=
by 
  -- skip the proof
  sorry

end largest_base5_to_base7_l172_172583


namespace initial_goldfish_eq_15_l172_172014

-- Let's define our setup as per the conditions provided
def fourGoldfishLeft := 4
def elevenGoldfishDisappeared := 11

-- Our main statement that we need to prove
theorem initial_goldfish_eq_15 : fourGoldfishLeft + elevenGoldfishDisappeared = 15 := by
  sorry

end initial_goldfish_eq_15_l172_172014


namespace sequence_condition_satisfies_l172_172200

def seq_prove_abs_lt_1 (a : ℕ → ℝ) : Prop :=
  (∃ i : ℕ, |a i| < 1)

theorem sequence_condition_satisfies (a : ℕ → ℝ)
  (h1 : a 1 * a 2 < 0)
  (h2 : ∀ n > 2, ∃ i j, 1 ≤ i ∧ i < j ∧ j < n ∧ (∀ k l, 1 ≤ k ∧ k < l ∧ l < n → |a i + a j| ≤ |a k + a l|)) :
  seq_prove_abs_lt_1 a :=
by
  sorry

end sequence_condition_satisfies_l172_172200


namespace minimum_width_l172_172665

theorem minimum_width (A l w : ℝ) (hA : A >= 150) (hl : l = 2 * w) (hA_def : A = w * l) : 
  w >= 5 * Real.sqrt 3 := 
  by
    -- Using the given conditions, we can prove that w >= 5 * sqrt(3)
    sorry

end minimum_width_l172_172665


namespace distance_in_scientific_notation_l172_172231

theorem distance_in_scientific_notation :
  ∃ a n : ℝ, 1 ≤ |a| ∧ |a| < 10 ∧ n = 4 ∧ 38000 = a * 10^n ∧ a = 3.8 :=
by
  sorry

end distance_in_scientific_notation_l172_172231


namespace percentage_paid_to_A_l172_172573

theorem percentage_paid_to_A (A B : ℝ) (h1 : A + B = 550) (h2 : B = 220) : (A / B) * 100 = 150 := by
  -- Proof omitted
  sorry

end percentage_paid_to_A_l172_172573


namespace apps_left_on_phone_l172_172930

-- Definitions for the given conditions
def initial_apps : ℕ := 15
def added_apps : ℕ := 71
def deleted_apps : ℕ := added_apps + 1

-- Proof statement
theorem apps_left_on_phone : initial_apps + added_apps - deleted_apps = 14 := by
  sorry

end apps_left_on_phone_l172_172930


namespace part_one_part_two_l172_172445

variable {a : ℕ → ℕ}

-- Conditions
axiom a1 : a 1 = 3
axiom recurrence_relation : ∀ n, a (n + 1) = 2 * (a n) + 1

-- Proof of the first part
theorem part_one: ∀ n, (a (n + 1) + 1) = 2 * (a n + 1) :=
by
  sorry

-- General formula for the sequence
theorem part_two: ∀ n, a n = 2^(n + 1) - 1 :=
by
  sorry

end part_one_part_two_l172_172445


namespace Vasya_not_11_more_than_Kolya_l172_172616

def is_L_shaped (n : ℕ) : Prop :=
  n % 2 = 1

def total_cells : ℕ :=
  14400

theorem Vasya_not_11_more_than_Kolya (k v : ℕ) :
  (is_L_shaped k) → (is_L_shaped v) → (k + v = total_cells) → (k % 2 = 0) → (v % 2 = 0) → (v - k ≠ 11) := 
by
  sorry

end Vasya_not_11_more_than_Kolya_l172_172616


namespace polygon_sides_l172_172598

theorem polygon_sides (n : ℕ) (h1 : (n - 2) * 180 - 180 = 2190) : n = 15 :=
sorry

end polygon_sides_l172_172598


namespace triangle_sides_l172_172144

theorem triangle_sides (a : ℕ) (h : a > 0) : 
  (a + 1) + (a + 2) > (a + 3) ∧ (a + 1) + (a + 3) > (a + 2) ∧ (a + 2) + (a + 3) > (a + 1) := 
by 
  sorry

end triangle_sides_l172_172144


namespace find_numbers_l172_172734

theorem find_numbers (A B C : ℝ) 
  (h1 : A - B = 1860) 
  (h2 : 0.075 * A = 0.125 * B) 
  (h3 : 0.15 * B = 0.05 * C) : 
  A = 4650 ∧ B = 2790 ∧ C = 8370 := 
by
  sorry

end find_numbers_l172_172734


namespace equation_has_real_roots_l172_172565

theorem equation_has_real_roots (k : ℝ) : ∀ (x : ℝ), 
  ∃ x, x = k^2 * (x - 1) * (x - 2) :=
by {
  sorry
}

end equation_has_real_roots_l172_172565


namespace fraction_irreducible_l172_172888

open Nat

theorem fraction_irreducible (m n : ℕ) : Nat.gcd (m * (n + 1) + 1) (m * (n + 1) - n) = 1 :=
  sorry

end fraction_irreducible_l172_172888


namespace ratio_difference_l172_172428

variables (p q r : ℕ) (x : ℕ)
noncomputable def shares_p := 3 * x
noncomputable def shares_q := 7 * x
noncomputable def shares_r := 12 * x

theorem ratio_difference (h1 : shares_q - shares_p = 2400) : shares_r - shares_q = 3000 :=
by sorry

end ratio_difference_l172_172428


namespace erick_total_money_collected_l172_172407

noncomputable def new_lemon_price (old_price increase : ℝ) : ℝ := old_price + increase
noncomputable def new_grape_price (old_price increase : ℝ) : ℝ := old_price + increase / 2

noncomputable def total_money_collected (lemons grapes : ℕ)
                                       (lemon_price grape_price lemon_increase : ℝ) : ℝ :=
  let new_lemon_price := new_lemon_price lemon_price lemon_increase
  let new_grape_price := new_grape_price grape_price lemon_increase
  lemons * new_lemon_price + grapes * new_grape_price

theorem erick_total_money_collected :
  total_money_collected 80 140 8 7 4 = 2220 := 
by
  sorry

end erick_total_money_collected_l172_172407


namespace find_value_of_x2001_plus_y2001_l172_172011

theorem find_value_of_x2001_plus_y2001 (x y : ℝ) (h1 : x - y = 2) (h2 : x^2 + y^2 = 4) : 
x ^ 2001 + y ^ 2001 = 2 ^ 2001 ∨ x ^ 2001 + y ^ 2001 = -2 ^ 2001 := by
  sorry

end find_value_of_x2001_plus_y2001_l172_172011


namespace jason_remaining_pokemon_cards_l172_172574

theorem jason_remaining_pokemon_cards :
  (3 - 2) = 1 :=
by 
  sorry

end jason_remaining_pokemon_cards_l172_172574


namespace sum_of_digits_133131_l172_172983

noncomputable def extract_digits_sum (n : Nat) : Nat :=
  let digits := n.digits 10
  digits.foldl (· + ·) 0

theorem sum_of_digits_133131 :
  let ABCDEF := 665655 / 5
  extract_digits_sum ABCDEF = 12 :=
by
  sorry

end sum_of_digits_133131_l172_172983


namespace product_remainder_31_l172_172986

theorem product_remainder_31 (m n : ℕ) (h₁ : m % 31 = 7) (h₂ : n % 31 = 12) : (m * n) % 31 = 22 :=
by
  sorry

end product_remainder_31_l172_172986


namespace algebraic_expression_value_l172_172186

-- Define the premises as a Lean statement
theorem algebraic_expression_value (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) :
  a * (b + c) + b * (a + c) + c * (a + b) = -1 :=
sorry

end algebraic_expression_value_l172_172186


namespace compound_interest_correct_l172_172198

-- define the problem conditions
def P : ℝ := 3000
def r : ℝ := 0.07
def n : ℕ := 25

-- the compound interest formula
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- state the theorem we want to prove
theorem compound_interest_correct :
  compound_interest P r n = 16281 := 
by
  sorry

end compound_interest_correct_l172_172198


namespace expression_never_equals_33_l172_172086

theorem expression_never_equals_33 (x y : ℤ) : 
  x^5 + 3 * x^4 * y - 5 * x^3 * y^2 - 15 * x^2 * y^3 + 4 * x * y^4 + 12 * y^5 ≠ 33 := 
sorry

end expression_never_equals_33_l172_172086


namespace find_k_l172_172979

theorem find_k 
  (k : ℤ) 
  (h : 2^2000 - 2^1999 - 2^1998 + 2^1997 = k * 2^1997) : 
  k = 3 :=
sorry

end find_k_l172_172979


namespace solve_for_x_l172_172841

theorem solve_for_x (i x : ℂ) (h : i^2 = -1) (eq : 3 - 2 * i * x = 5 + 4 * i * x) : x = i / 3 := 
by
  sorry

end solve_for_x_l172_172841


namespace expression_equals_required_value_l172_172693

-- Define the expression as needed
def expression : ℚ := (((((4 + 2)⁻¹ + 2)⁻¹) + 2)⁻¹) + 2

-- Define the theorem stating that the expression equals the required value
theorem expression_equals_required_value : 
  expression = 77 / 32 := 
sorry

end expression_equals_required_value_l172_172693


namespace charge_R_12_5_percent_more_l172_172464

-- Let R be the charge for a single room at hotel R.
-- Let G be the charge for a single room at hotel G.
-- Let P be the charge for a single room at hotel P.

def charge_R (R : ℝ) : Prop := true
def charge_G (G : ℝ) : Prop := true
def charge_P (P : ℝ) : Prop := true

axiom hotel_P_20_less_R (R P : ℝ) : charge_R R → charge_P P → P = 0.80 * R
axiom hotel_P_10_less_G (G P : ℝ) : charge_G G → charge_P P → P = 0.90 * G

theorem charge_R_12_5_percent_more (R G : ℝ) :
  charge_R R → charge_G G → (∃ P, charge_P P ∧ P = 0.80 * R ∧ P = 0.90 * G) → R = 1.125 * G :=
by sorry

end charge_R_12_5_percent_more_l172_172464


namespace pizza_slices_needed_l172_172682

theorem pizza_slices_needed (couple_slices : ℕ) (children : ℕ) (children_slices : ℕ) (pizza_slices : ℕ)
    (hc : couple_slices = 3)
    (hcouple : children = 6)
    (hch : children_slices = 1)
    (hpizza : pizza_slices = 4) : 
    (2 * couple_slices + children * children_slices) / pizza_slices = 3 := 
by
    sorry

end pizza_slices_needed_l172_172682


namespace min_m_value_l172_172311

theorem min_m_value :
  ∃ (x y m : ℝ), x - y + 2 ≥ 0 ∧ x + y - 2 ≤ 0 ∧ 2 * y ≥ x + 2 ∧
  (m > 0) ∧ (x^2 / 4 + y^2 = m^2) ∧ m = Real.sqrt 2 / 2 :=
sorry

end min_m_value_l172_172311


namespace range_of_m_l172_172591

def A (x : ℝ) := x^2 - 3 * x - 10 ≤ 0
def B (x m : ℝ) := m + 1 ≤ x ∧ x ≤ 2 * m - 1

theorem range_of_m (m : ℝ) (h : ∀ x, B x m → A x) : m ≤ 3 := by
  sorry

end range_of_m_l172_172591


namespace intersection_two_sets_l172_172910

theorem intersection_two_sets (M N : Set ℤ) (h1 : M = {1, 2, 3, 4}) (h2 : N = {-2, 2}) :
  M ∩ N = {2} := 
by
  sorry

end intersection_two_sets_l172_172910


namespace letters_identity_l172_172702

theorem letters_identity (l1 l2 l3 : Prop) 
  (h1 : l1 → l2 → false)
  (h2 : ¬(l1 ∧ l3))
  (h3 : ¬(l2 ∧ l3))
  (h4 : l3 → ¬l1 ∧ l2 ∧ ¬(l1 ∧ ¬l2)) :
  (¬l1 ∧ l2 ∧ ¬l3) :=
by 
  sorry

end letters_identity_l172_172702


namespace max_gcd_b_n_b_n_plus_1_l172_172532

noncomputable def b (n : ℕ) : ℚ := (2 ^ n - 1) / 3

theorem max_gcd_b_n_b_n_plus_1 : ∀ n : ℕ, Int.gcd (b n).num (b (n + 1)).num = 1 :=
by
  sorry

end max_gcd_b_n_b_n_plus_1_l172_172532


namespace find_a5_over_T9_l172_172617

-- Define arithmetic sequences and their sums
variables {a_n : ℕ → ℚ} {b_n : ℕ → ℚ}
variables {S_n : ℕ → ℚ} {T_n : ℕ → ℚ}

-- Conditions
def arithmetic_seq_a (a_n : ℕ → ℚ) : Prop :=
  ∀ n, a_n n = a_n 1 + (n - 1) * (a_n 2 - a_n 1)

def arithmetic_seq_b (b_n : ℕ → ℚ) : Prop :=
  ∀ n, b_n n = b_n 1 + (n - 1) * (b_n 2 - b_n 1)

def sum_a (S_n : ℕ → ℚ) (a_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n = n * (a_n 1 + a_n n) / 2

def sum_b (T_n : ℕ → ℚ) (b_n : ℕ → ℚ) : Prop :=
  ∀ n, T_n n = n * (b_n 1 + b_n n) / 2

def given_condition (S_n : ℕ → ℚ) (T_n : ℕ → ℚ) : Prop :=
  ∀ n, S_n n / T_n n = (n + 3) / (2 * n - 1)

-- Goal statement
theorem find_a5_over_T9 (h_a : arithmetic_seq_a a_n) (h_b : arithmetic_seq_b b_n)
  (sum_a_S : sum_a S_n a_n) (sum_b_T : sum_b T_n b_n) (cond : given_condition S_n T_n) :
  a_n 5 / T_n 9 = 4 / 51 :=
  sorry

end find_a5_over_T9_l172_172617


namespace necessary_but_not_sufficient_for_p_l172_172042

variable {p q r : Prop}

theorem necessary_but_not_sufficient_for_p 
  (h₁ : p → q) (h₂ : ¬ (q → p)) 
  (h₃ : q → r) (h₄ : ¬ (r → q)) 
  : (r → p) ∧ ¬ (p → r) :=
sorry

end necessary_but_not_sufficient_for_p_l172_172042


namespace yangyang_departure_time_l172_172865

noncomputable def departure_time : Nat := 373 -- 6:13 in minutes from midnight (6 * 60 + 13)

theorem yangyang_departure_time :
  let arrival_at_60_mpm := 413 -- 6:53 in minutes from midnight
  let arrival_at_75_mpm := 405 -- 6:45 in minutes from midnight
  let difference := arrival_at_60_mpm - arrival_at_75_mpm -- time difference
  let x := 40 -- time taken to walk to school at 60 meters per minute
  departure_time = arrival_at_60_mpm - x :=
by
  -- Definitions
  let arrival_at_60_mpm := 413
  let arrival_at_75_mpm := 405
  let difference := 8
  let x := 40
  have h : departure_time = (413 - 40) := rfl
  sorry

end yangyang_departure_time_l172_172865


namespace arithmetic_sequences_ratio_l172_172421

theorem arithmetic_sequences_ratio (a b S T : ℕ → ℕ) (h : ∀ n, S n / T n = 2 * n / (3 * n + 1)) :
  (a 2) / (b 3 + b 7) + (a 8) / (b 4 + b 6) = 9 / 14 :=
  sorry

end arithmetic_sequences_ratio_l172_172421


namespace remainder_of_division_l172_172242

theorem remainder_of_division (x r : ℕ) (h : 23 = 7 * x + r) : r = 2 :=
sorry

end remainder_of_division_l172_172242


namespace parabola_focus_coordinates_l172_172447

theorem parabola_focus_coordinates :
  ∀ x y : ℝ, y^2 - 4 * x = 0 → (x, y) = (1, 0) :=
by
  -- Use the equivalence given by the problem
  intros x y h
  sorry

end parabola_focus_coordinates_l172_172447


namespace barney_extra_weight_l172_172994

-- Define the weight of a regular dinosaur
def regular_dinosaur_weight : ℕ := 800

-- Define the combined weight of five regular dinosaurs
def five_regular_dinosaurs_weight : ℕ := 5 * regular_dinosaur_weight

-- Define the total weight of Barney and the five regular dinosaurs together
def total_combined_weight : ℕ := 9500

-- Define the weight of Barney
def barney_weight : ℕ := total_combined_weight - five_regular_dinosaurs_weight

-- The proof statement
theorem barney_extra_weight : barney_weight - five_regular_dinosaurs_weight = 1500 :=
by sorry

end barney_extra_weight_l172_172994


namespace final_probability_l172_172472

-- Define the structure of the problem
structure GameRound :=
  (green_ball : ℕ)
  (red_ball : ℕ)
  (blue_ball : ℕ)
  (white_ball : ℕ)

structure GameState :=
  (coins : ℕ)
  (players : ℕ)

-- Define the game rules and initial conditions
noncomputable def initial_coins := 5
noncomputable def rounds := 5

-- Probability-related functions and game logic
noncomputable def favorable_outcome_count : ℕ := 6
noncomputable def total_outcomes_per_round : ℕ := 120
noncomputable def probability_per_round : ℚ := favorable_outcome_count / total_outcomes_per_round

theorem final_probability :
  probability_per_round ^ rounds = 1 / 3200000 :=
by
  sorry

end final_probability_l172_172472


namespace max_min_value_function_l172_172008

noncomputable def given_function (x : ℝ) : ℝ :=
  (Real.sin x) ^ 2 + Real.cos x + 1

theorem max_min_value_function :
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≤ 9 / 4) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 9 / 4) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2) → given_function x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2), given_function x = 2) := by
  sorry

end max_min_value_function_l172_172008


namespace converse_inverse_contrapositive_l172_172036

theorem converse (x y : ℤ) : (x = 3 ∧ y = 2) → (x + y = 5) :=
by sorry

theorem inverse (x y : ℤ) : (x + y ≠ 5) → (x ≠ 3 ∨ y ≠ 2) :=
by sorry

theorem contrapositive (x y : ℤ) : (¬ (x = 3 ∧ y = 2)) → (¬ (x + y = 5)) :=
by sorry

end converse_inverse_contrapositive_l172_172036


namespace tan_triple_angle_l172_172007

theorem tan_triple_angle (θ : ℝ) (h : Real.tan θ = 3) : Real.tan (3 * θ) = 9 / 13 :=
by
  sorry

end tan_triple_angle_l172_172007


namespace solve_inequality_l172_172417

theorem solve_inequality (x : ℝ) (h1: 3 * x - 8 ≠ 0) :
  5 ≤ x / (3 * x - 8) ∧ x / (3 * x - 8) < 10 ↔ (8 / 3) < x ∧ x ≤ (20 / 7) := 
sorry

end solve_inequality_l172_172417


namespace sum_of_factors_is_17_l172_172018

theorem sum_of_factors_is_17 :
  ∃ (a b c d e f g : ℤ), 
  (16 * x^4 - 81 * y^4) =
    (a * x + b * y) * 
    (c * x^2 + d * x * y + e * y^2) * 
    (f * x + g * y) ∧ 
    a + b + c + d + e + f + g = 17 :=
by
  sorry

end sum_of_factors_is_17_l172_172018


namespace problem_part1_problem_part2_l172_172481

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a - 2 / (2^x + 1)

theorem problem_part1 (h : ∀ x : ℝ, f (-x) = -f x) : a = 1 :=
sorry

theorem problem_part2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2 :=
sorry

end problem_part1_problem_part2_l172_172481


namespace Tim_scores_expected_value_l172_172138

theorem Tim_scores_expected_value :
  let LAIMO := 15
  let FARML := 10
  let DOMO := 50
  let p := 1 / 3
  let expected_LAIMO := LAIMO * p
  let expected_FARML := FARML * p
  let expected_DOMO := DOMO * p
  expected_LAIMO + expected_FARML + expected_DOMO = 25 :=
by
  -- The Lean proof would go here
  sorry

end Tim_scores_expected_value_l172_172138


namespace find_polynomials_l172_172989

-- Definition of polynomials in Lean
noncomputable def polynomials : Type := Polynomial ℝ

-- Main theorem statement
theorem find_polynomials : 
  ∀ p : polynomials, 
    (∀ x : ℝ, p.eval (5 * x) ^ 2 - 3 = p.eval (5 * x^2 + 1)) → 
    (p.eval 0 ≠ 0 → (∃ c : ℝ, (p = Polynomial.C c) ∧ (c = (1 + Real.sqrt 13) / 2 ∨ c = (1 - Real.sqrt 13) / 2))) ∧ 
    (p.eval 0 = 0 → ∀ x : ℝ, p.eval x = 0) :=
by
  sorry

end find_polynomials_l172_172989


namespace magnitude_of_z_l172_172750

open Complex -- open the complex number namespace

theorem magnitude_of_z (z : ℂ) (h : z + I = 3) : Complex.abs z = Real.sqrt 10 :=
by
  sorry

end magnitude_of_z_l172_172750


namespace yuna_average_score_l172_172505

theorem yuna_average_score (avg_may_june : ℕ) (score_july : ℕ) (h1 : avg_may_june = 84) (h2 : score_july = 96) :
  (avg_may_june * 2 + score_july) / 3 = 88 := by
  sorry

end yuna_average_score_l172_172505


namespace perpendicular_lines_unique_a_l172_172964

open Real

theorem perpendicular_lines_unique_a (a : ℝ) 
  (l1 : ∀ x y : ℝ, (a - 1) * x + y - 1 = 0) 
  (l2 : ∀ x y : ℝ, 3 * x + a * y + 2 = 0) 
  (perpendicular : True) : 
  a = 3 / 4 := 
sorry

end perpendicular_lines_unique_a_l172_172964


namespace coordinates_after_5_seconds_l172_172639

-- Define the initial coordinates of point P
def initial_coordinates : ℚ × ℚ := (-10, 10)

-- Define the velocity vector of point P
def velocity_vector : ℚ × ℚ := (4, -3)

-- Asserting the coordinates of point P after 5 seconds
theorem coordinates_after_5_seconds : 
   initial_coordinates + 5 • velocity_vector = (10, -5) :=
by 
  sorry

end coordinates_after_5_seconds_l172_172639


namespace cost_fly_D_to_E_l172_172127

-- Definitions for the given conditions
def distance_DE : ℕ := 4750
def cost_per_km_plane : ℝ := 0.12
def booking_fee_plane : ℝ := 150

-- The proof statement about the total cost
theorem cost_fly_D_to_E : (distance_DE * cost_per_km_plane + booking_fee_plane = 720) :=
by sorry

end cost_fly_D_to_E_l172_172127


namespace two_colonies_limit_l172_172643

def doubles_each_day (size: ℕ) (day: ℕ) : ℕ := size * 2 ^ day

theorem two_colonies_limit (habitat_limit: ℕ) (initial_size: ℕ) : 
  (∀ t, doubles_each_day initial_size t = habitat_limit → t = 20) → 
  initial_size > 0 →
  ∀ t, doubles_each_day (2 * initial_size) t = habitat_limit → t = 20 :=
by
  sorry

end two_colonies_limit_l172_172643


namespace total_cost_second_set_l172_172551

variable (A V : ℝ)

-- Condition declarations
axiom cost_video_cassette : V = 300
axiom cost_second_set : 7 * A + 3 * V = 1110

-- Proof goal
theorem total_cost_second_set :
  7 * A + 3 * V = 1110 :=
by
  sorry

end total_cost_second_set_l172_172551


namespace find_f_21_l172_172402

def f : ℝ → ℝ := sorry

lemma f_condition (x : ℝ) : f (2 / x + 1) = Real.log x := sorry

theorem find_f_21 : f 21 = -1 := sorry

end find_f_21_l172_172402


namespace daisy_germination_rate_theorem_l172_172024

-- Define the conditions of the problem
variables (daisySeeds sunflowerSeeds : ℕ) (sunflowerGermination flowerProduction finalFlowerPlants : ℝ)
def conditions : Prop :=
  daisySeeds = 25 ∧ sunflowerSeeds = 25 ∧ sunflowerGermination = 0.80 ∧ flowerProduction = 0.80 ∧ finalFlowerPlants = 28

-- Define the statement that the germination rate of the daisy seeds is 60%
def germination_rate_of_daisy_seeds : Prop :=
  ∃ (daisyGerminationRate : ℝ), (conditions daisySeeds sunflowerSeeds sunflowerGermination flowerProduction finalFlowerPlants) →
  daisyGerminationRate = 0.60

-- The proof is omitted - note this is just the statement
theorem daisy_germination_rate_theorem : germination_rate_of_daisy_seeds 25 25 0.80 0.80 28 :=
sorry

end daisy_germination_rate_theorem_l172_172024


namespace sell_price_equal_percentage_l172_172607

theorem sell_price_equal_percentage (SP : ℝ) (CP : ℝ) :
  (SP - CP) / CP * 100 = (CP - 1280) / CP * 100 → 
  (1937.5 = CP + 0.25 * CP) → 
  SP = 1820 :=
by 
  -- Note: skip proof with sorry
  apply sorry

end sell_price_equal_percentage_l172_172607


namespace shaded_area_of_joined_squares_l172_172351

theorem shaded_area_of_joined_squares:
  ∀ (a b : ℕ) (area_of_shaded : ℝ),
  (a = 6) → (b = 8) → 
  (area_of_shaded = (6 * 6 : ℝ) + (8 * 8 : ℝ) / 2) →
  area_of_shaded = 50.24 := 
by
  intros a b area_of_shaded h1 h2 h3
  -- skipping the proof for now
  sorry

end shaded_area_of_joined_squares_l172_172351


namespace calculate_value_l172_172751

theorem calculate_value (a b c : ℤ) (h₁ : a = 5) (h₂ : b = -3) (h₃ : c = 4) : 2 * c / (a + b) = 4 :=
by
  rw [h₁, h₂, h₃]
  sorry

end calculate_value_l172_172751


namespace bike_ride_energetic_time_l172_172320

theorem bike_ride_energetic_time :
  ∃ x : ℚ, (22 * x + 15 * (7.5 - x) = 142) ∧ x = (59 / 14) :=
by
  sorry

end bike_ride_energetic_time_l172_172320


namespace rational_root_uniqueness_l172_172706

theorem rational_root_uniqueness (c : ℚ) :
  ∀ x1 x2 : ℚ, (x1 ≠ x2) →
  (x1^3 - 3 * c * x1^2 - 3 * x1 + c = 0) →
  (x2^3 - 3 * c * x2^2 - 3 * x2 + c = 0) →
  false := 
by
  intros x1 x2 h1 h2 h3
  sorry

end rational_root_uniqueness_l172_172706


namespace numberOfWaysToPlaceCoinsSix_l172_172699

def numberOfWaysToPlaceCoins (n : ℕ) : ℕ :=
  if n = 0 then 1 else 2 * numberOfWaysToPlaceCoins (n - 1)

theorem numberOfWaysToPlaceCoinsSix : numberOfWaysToPlaceCoins 6 = 32 :=
by
  sorry

end numberOfWaysToPlaceCoinsSix_l172_172699


namespace total_interest_rate_is_correct_l172_172844

theorem total_interest_rate_is_correct :
  let total_investment := 100000
  let interest_rate_first := 0.09
  let interest_rate_second := 0.11
  let invested_in_second := 29999.999999999993
  let invested_in_first := total_investment - invested_in_second
  let interest_first := invested_in_first * interest_rate_first
  let interest_second := invested_in_second * interest_rate_second
  let total_interest := interest_first + interest_second
  let total_interest_rate := (total_interest / total_investment) * 100
  total_interest_rate = 9.6 :=
by
  sorry

end total_interest_rate_is_correct_l172_172844


namespace smallest_value_of_n_l172_172880

theorem smallest_value_of_n (r g b : ℕ) (p : ℕ) (h_p : p = 20) 
                            (h_money : ∃ k, k = 12 * r ∨ k = 14 * g ∨ k = 15 * b ∨ k = 20 * n)
                            (n : ℕ) : n = 21 :=
by
  sorry

end smallest_value_of_n_l172_172880


namespace seashells_initial_count_l172_172183

theorem seashells_initial_count (S : ℕ)
  (h1 : S - 70 = 2 * 55) : S = 180 :=
by
  sorry

end seashells_initial_count_l172_172183


namespace missed_angle_l172_172727

theorem missed_angle (n : ℕ) (h1 : (n - 2) * 180 ≥ 3239) (h2 : n ≥ 3) : 3240 - 3239 = 1 :=
by
  sorry

end missed_angle_l172_172727


namespace find_numbers_l172_172927

theorem find_numbers :
  ∃ a d : ℝ, 
    ((a - d) + a + (a + d) = 12) ∧ 
    ((a - d) * a * (a + d) = 48) ∧
    (a = 4) ∧ 
    (d = -2) ∧ 
    (a - d = 6) ∧ 
    (a + d = 2) :=
by
  sorry

end find_numbers_l172_172927


namespace a_alone_time_to_complete_work_l172_172384

theorem a_alone_time_to_complete_work :
  (W : ℝ) →
  (A : ℝ) →
  (B : ℝ) →
  (h1 : A + B = W / 6) →
  (h2 : B = W / 12) →
  A = W / 12 :=
by
  -- Given conditions
  intros W A B h1 h2
  -- Proof is not needed as per instructions
  sorry

end a_alone_time_to_complete_work_l172_172384


namespace num_k_vals_l172_172156

-- Definitions of the conditions
def div_by_7 (n k : ℕ) : Prop :=
  (2 * 3^(6*n) + k * 2^(3*n + 1) - 1) % 7 = 0

-- Main theorem statement
theorem num_k_vals : 
  ∃ (S : Finset ℕ), (∀ k ∈ S, k < 100 ∧ ∀ n, div_by_7 n k) ∧ S.card = 14 := 
by
  sorry

end num_k_vals_l172_172156


namespace find_interest_rate_l172_172038

-- Define the given conditions
def initial_investment : ℝ := 2200
def additional_investment : ℝ := 1099.9999999999998
def total_investment : ℝ := initial_investment + additional_investment
def desired_income : ℝ := 0.06 * total_investment
def income_from_additional_investment : ℝ := 0.08 * additional_investment
def income_from_initial_investment (r : ℝ) : ℝ := initial_investment * r

-- State the proof problem
theorem find_interest_rate (r : ℝ) 
    (h : desired_income = income_from_additional_investment + income_from_initial_investment r) :
    r = 0.05 :=
sorry

end find_interest_rate_l172_172038


namespace line_passes_through_fixed_point_l172_172390

theorem line_passes_through_fixed_point (a b c : ℝ) (h : a - b + c = 0) : a * 1 + b * (-1) + c = 0 := 
by sorry

end line_passes_through_fixed_point_l172_172390


namespace gabby_mom_gave_20_l172_172975

theorem gabby_mom_gave_20 (makeup_set_cost saved_money more_needed total_needed mom_money : ℕ)
  (h1 : makeup_set_cost = 65)
  (h2 : saved_money = 35)
  (h3 : more_needed = 10)
  (h4 : total_needed = makeup_set_cost - saved_money)
  (h5 : total_needed - mom_money = more_needed) :
  mom_money = 20 :=
by
  sorry

end gabby_mom_gave_20_l172_172975


namespace ramu_spent_on_repairs_l172_172396

theorem ramu_spent_on_repairs 
    (initial_cost : ℝ) (selling_price : ℝ) (profit_percent : ℝ) (R : ℝ) 
    (h1 : initial_cost = 42000) 
    (h2 : selling_price = 64900) 
    (h3 : profit_percent = 18) 
    (h4 : profit_percent / 100 = (selling_price - (initial_cost + R)) / (initial_cost + R)) : 
    R = 13000 :=
by
  rw [h1, h2, h3] at h4
  sorry

end ramu_spent_on_repairs_l172_172396


namespace simplify_expression_l172_172277

variable (d : ℤ)

theorem simplify_expression :
  (5 + 4 * d) / 9 - 3 + 1 / 3 = (4 * d - 19) / 9 := by
  sorry

end simplify_expression_l172_172277


namespace people_between_katya_and_polina_l172_172606

-- Definitions based on given conditions
def is_next_to (a b : ℕ) : Prop := (b = a + 1) ∨ (b = a - 1)
def position_alena : ℕ := 1
def position_lena : ℕ := 5
def position_sveta (pos_sveta : ℕ) : Prop := pos_sveta + 1 = position_lena
def position_katya (pos_katya : ℕ) : Prop := pos_katya = 3
def position_polina (pos_polina : ℕ) : Prop := (is_next_to position_alena pos_polina)

-- The question: prove the number of people between Katya and Polina is 0
theorem people_between_katya_and_polina : 
  ∃ (pos_katya pos_polina : ℕ),
    position_katya pos_katya ∧ 
    position_polina pos_polina ∧ 
    pos_polina + 1 = pos_katya ∧
    pos_katya = 3 ∧ pos_polina = 2 := 
sorry

end people_between_katya_and_polina_l172_172606


namespace area_of_circumscribed_circle_l172_172132

noncomputable def circumradius_of_equilateral_triangle (s : ℝ) : ℝ :=
  s / Real.sqrt 3

noncomputable def area_of_circle (r : ℝ) : ℝ :=
  Real.pi * r^2

theorem area_of_circumscribed_circle (s : ℝ) (h : s = 12) : area_of_circle (circumradius_of_equilateral_triangle s) = 48 * Real.pi := by
  sorry

end area_of_circumscribed_circle_l172_172132


namespace debate_organizing_committees_count_l172_172112

theorem debate_organizing_committees_count :
    ∃ (n : ℕ), n = 5 * (Nat.choose 8 4) * (Nat.choose 8 3)^4 ∧ n = 3442073600 :=
by
  sorry

end debate_organizing_committees_count_l172_172112


namespace cheezit_bag_weight_l172_172326

-- Definitions based on the conditions of the problem
def cheezit_bags : ℕ := 3
def calories_per_ounce : ℕ := 150
def run_minutes : ℕ := 40
def calories_per_minute : ℕ := 12
def excess_calories : ℕ := 420

-- Main theorem stating the question with the solution
theorem cheezit_bag_weight (x : ℕ) : 
  (calories_per_ounce * cheezit_bags * x) - (run_minutes * calories_per_minute) = excess_calories → 
  x = 2 :=
by
  sorry

end cheezit_bag_weight_l172_172326


namespace find_a_bi_c_l172_172540

theorem find_a_bi_c (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h_eq : (a - (b : ℤ)*I)^2 + c = 13 - 8*I) :
  a = 2 ∧ b = 2 ∧ c = 13 :=
by
  sorry

end find_a_bi_c_l172_172540


namespace flight_cost_A_to_B_l172_172362

-- Definitions based on conditions in the problem
def distance_AB : ℝ := 2000
def flight_cost_per_km : ℝ := 0.10
def booking_fee : ℝ := 100

-- Statement: Given the distances and cost conditions, the flight cost from A to B is $300
theorem flight_cost_A_to_B : distance_AB * flight_cost_per_km + booking_fee = 300 := by
  sorry

end flight_cost_A_to_B_l172_172362


namespace ratio_of_first_to_second_l172_172354

theorem ratio_of_first_to_second (x y : ℕ) 
  (h1 : x + y + (1 / 3 : ℚ) * x = 110)
  (h2 : y = 30) :
  x / y = 2 :=
by
  sorry

end ratio_of_first_to_second_l172_172354


namespace determine_c_div_d_l172_172165

theorem determine_c_div_d (x y c d : ℝ) (h1 : 4 * x + 8 * y = c) (h2 : 5 * x - 10 * y = d) (h3 : d ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) : c / d = -4 / 5 :=
by
sorry

end determine_c_div_d_l172_172165


namespace two_point_three_five_as_fraction_l172_172248

theorem two_point_three_five_as_fraction : (2.35 : ℚ) = 47 / 20 :=
by
-- We'll skip the intermediate steps and just state the end result
-- because the prompt specifies not to include the solution steps.
sorry

end two_point_three_five_as_fraction_l172_172248


namespace find_functions_l172_172280

def satisfies_equation (f : ℤ → ℤ) : Prop :=
  ∀ a b : ℤ, f (2 * a) + 2 * f b = f (f (a + b))

theorem find_functions (f : ℤ → ℤ) (h : satisfies_equation f) : (∀ x, f x = 2 * x) ∨ (∀ x, f x = 0) :=
sorry

end find_functions_l172_172280


namespace ratio_G_to_C_is_1_1_l172_172160

variable (R C G : ℕ)

-- Given conditions
def Rover_has_46_spots : Prop := R = 46
def Cisco_has_half_R_minus_5 : Prop := C = R / 2 - 5
def Granger_Cisco_combined_108 : Prop := G + C = 108
def Granger_Cisco_equal : Prop := G = C

-- Theorem stating the final answer to the problem
theorem ratio_G_to_C_is_1_1 (h1 : Rover_has_46_spots R) 
                            (h2 : Cisco_has_half_R_minus_5 C R) 
                            (h3 : Granger_Cisco_combined_108 G C) 
                            (h4 : Granger_Cisco_equal G C) : 
                            G / C = 1 := by
  sorry

end ratio_G_to_C_is_1_1_l172_172160


namespace hannahs_weekly_pay_l172_172753

-- Define conditions
def hourly_wage : ℕ := 30
def total_hours : ℕ := 18
def dock_per_late : ℕ := 5
def late_times : ℕ := 3

-- The amount paid after deductions for being late
def pay_after_deductions : ℕ :=
  let wage_before_deductions := hourly_wage * total_hours
  let total_dock := dock_per_late * late_times
  wage_before_deductions - total_dock

-- The proof statement
theorem hannahs_weekly_pay : pay_after_deductions = 525 := 
  by
  -- No proof necessary; statement and conditions must be correctly written to run
  sorry

end hannahs_weekly_pay_l172_172753


namespace abby_bridget_chris_probability_l172_172019

noncomputable def seatingProbability : ℚ :=
  let totalArrangements := 720
  let favorableArrangements := 114
  favorableArrangements / totalArrangements

theorem abby_bridget_chris_probability :
  seatingProbability = 19 / 120 :=
by
  simp [seatingProbability]
  sorry

end abby_bridget_chris_probability_l172_172019


namespace find_original_denominator_l172_172267

theorem find_original_denominator (d : ℕ) 
  (h : (10 : ℚ) / (d + 7) = 1 / 3) : 
  d = 23 :=
by 
  sorry

end find_original_denominator_l172_172267


namespace find_values_of_ABC_l172_172212

-- Define the given conditions
def condition1 (A B C : ℕ) : Prop := A + B + C = 36
def condition2 (A B C : ℕ) : Prop := 
  (A + B) * 3 * 4 = (B + C) * 2 * 4 ∧ 
  (B + C) * 2 * 4 = (A + C) * 2 * 3

-- State the problem
theorem find_values_of_ABC (A B C : ℕ) 
  (h1 : condition1 A B C) 
  (h2 : condition2 A B C) : 
  A = 12 ∧ B = 4 ∧ C = 20 :=
sorry

end find_values_of_ABC_l172_172212


namespace proportion_third_number_l172_172425

theorem proportion_third_number
  (x : ℝ) (y : ℝ)
  (h1 : 0.60 * 4 = x * y)
  (h2 : x = 0.39999999999999997) :
  y = 6 :=
by
  sorry

end proportion_third_number_l172_172425


namespace retail_price_of_machine_l172_172003

theorem retail_price_of_machine 
  (wholesale_price : ℝ) 
  (discount_rate : ℝ) 
  (profit_rate : ℝ) 
  (selling_price : ℝ) 
  (P : ℝ)
  (h1 : wholesale_price = 90)
  (h2 : discount_rate = 0.10)
  (h3 : profit_rate = 0.20)
  (h4 : selling_price = wholesale_price * (1 + profit_rate))
  (h5 : (P * (1 - discount_rate)) = selling_price) : 
  P = 120 := by
  sorry

end retail_price_of_machine_l172_172003


namespace sum_of_factors_of_120_is_37_l172_172137

theorem sum_of_factors_of_120_is_37 :
  ∃ a b c d e : ℤ, (a * b = 120) ∧ (b = a + 1) ∧ (c * d * e = 120) ∧ (d = c + 1) ∧ (e = d + 1) ∧ (a + b + c + d + e = 37) :=
by
  sorry

end sum_of_factors_of_120_is_37_l172_172137


namespace problem1_problem2_l172_172113

-- Definition of the function
def f (a x : ℝ) := x^2 + a * x + 3

-- Problem statement 1: Prove that if f(x) ≥ a for all x ∈ ℜ, then a ≤ 3.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x ≥ a) → a ≤ 3 := sorry

-- Problem statement 2: Prove that if f(x) ≥ a for all x ∈ [-2, 2], then -6 ≤ a ≤ 2.
theorem problem2 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≥ a) → -6 ≤ a ∧ a ≤ 2 := sorry

end problem1_problem2_l172_172113


namespace find_integer_modulo_l172_172814

theorem find_integer_modulo : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [MOD 11] := by
  use 3
  sorry

end find_integer_modulo_l172_172814


namespace multiple_of_2_and_3_is_divisible_by_6_l172_172783

theorem multiple_of_2_and_3_is_divisible_by_6 (n : ℤ) (h1 : n % 2 = 0) (h2 : n % 3 = 0) : n % 6 = 0 :=
sorry

end multiple_of_2_and_3_is_divisible_by_6_l172_172783


namespace abc_value_l172_172749

variables (a b c : ℝ)

theorem abc_value (h1 : a * (b + c) = 156) (h2 : b * (c + a) = 168) (h3 : c * (a + b) = 180) :
  a * b * c = 288 * Real.sqrt 7 :=
sorry

end abc_value_l172_172749


namespace rotten_eggs_prob_l172_172235

theorem rotten_eggs_prob (T : ℕ) (P : ℝ) (R : ℕ) :
  T = 36 ∧ P = 0.0047619047619047615 ∧ P = (R / T) * ((R - 1) / (T - 1)) → R = 3 :=
by
  sorry

end rotten_eggs_prob_l172_172235


namespace cone_section_area_half_base_ratio_l172_172779

theorem cone_section_area_half_base_ratio (h_base h_upper h_lower : ℝ) (A_base A_upper : ℝ) 
  (h_total : h_upper + h_lower = h_base)
  (A_upper : A_upper = A_base / 2) :
  h_upper = h_lower :=
by
  sorry

end cone_section_area_half_base_ratio_l172_172779


namespace frogs_per_fish_per_day_l172_172070

theorem frogs_per_fish_per_day
  (f g n F : ℕ)
  (h1 : f = 30)
  (h2 : g = 15)
  (h3 : n = 9)
  (h4 : F = 32400) :
  F / f / (n * g) = 8 := by
  sorry

end frogs_per_fish_per_day_l172_172070


namespace elizabeth_wedding_gift_cost_l172_172298

-- Defining the given conditions
def cost_steak_knife_set : ℝ := 80.00
def num_steak_knife_sets : ℝ := 2
def cost_dinnerware_set : ℝ := 200.00
def discount_rate : ℝ := 0.10
def sales_tax_rate : ℝ := 0.05

-- Calculating total expense
def total_cost (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set : ℝ) : ℝ :=
  (cost_steak_knife_set * num_steak_knife_sets) + cost_dinnerware_set

def discounted_price (total_cost discount_rate : ℝ) : ℝ :=
  total_cost - (total_cost * discount_rate)

def final_price (discounted_price sales_tax_rate : ℝ) : ℝ :=
  discounted_price + (discounted_price * sales_tax_rate)

def elizabeth_spends (cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate : ℝ) : ℝ :=
  final_price (discounted_price (total_cost cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set) discount_rate) sales_tax_rate

theorem elizabeth_wedding_gift_cost
  (cost_steak_knife_set : ℝ)
  (num_steak_knife_sets : ℝ)
  (cost_dinnerware_set : ℝ)
  (discount_rate : ℝ)
  (sales_tax_rate : ℝ) :
  elizabeth_spends cost_steak_knife_set num_steak_knife_sets cost_dinnerware_set discount_rate sales_tax_rate = 340.20 := 
by
  sorry -- Proof is to be completed

end elizabeth_wedding_gift_cost_l172_172298


namespace greatest_three_digit_multiple_of_17_is_986_l172_172922

theorem greatest_three_digit_multiple_of_17_is_986:
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ 17 ∣ n ∧ (∀ m, 100 ≤ m ∧ m ≤ 999 ∧ 17 ∣ m → m ≤ 986) :=
sorry

end greatest_three_digit_multiple_of_17_is_986_l172_172922


namespace road_trip_cost_l172_172884

theorem road_trip_cost 
  (x : ℝ)
  (initial_cost_per_person: ℝ) 
  (redistributed_cost_per_person: ℝ)
  (cost_difference: ℝ) :
  initial_cost_per_person = x / 4 →
  redistributed_cost_per_person = x / 7 →
  cost_difference = 8 →
  initial_cost_per_person - redistributed_cost_per_person = cost_difference →
  x = 74.67 :=
by
  intro h1 h2 h3 h4
  -- starting the proof
  rw [h1, h2] at h4
  sorry

end road_trip_cost_l172_172884


namespace isosceles_triangle_perimeter_l172_172093

theorem isosceles_triangle_perimeter (a b c : ℝ) 
  (h1 : a = 4 ∨ b = 4 ∨ c = 4) 
  (h2 : a = 8 ∨ b = 8 ∨ c = 8) 
  (isosceles : a = b ∨ b = c ∨ a = c) : 
  a + b + c = 20 :=
by
  sorry

end isosceles_triangle_perimeter_l172_172093


namespace fraction_money_left_zero_l172_172509

-- Defining variables and conditions
variables {m c : ℝ} -- m: total money, c: total cost of CDs

-- Condition under the problem statement
def uses_one_fourth_of_money_to_buy_one_fourth_of_CDs (m c : ℝ) := (1 / 4) * m = (1 / 4) * c

-- The conjecture to be proven
theorem fraction_money_left_zero 
  (h: uses_one_fourth_of_money_to_buy_one_fourth_of_CDs m c) 
  (h_eq: c = m) : 
  (m - c) / m = 0 := 
by
  sorry

end fraction_money_left_zero_l172_172509


namespace population_of_missing_village_l172_172897

theorem population_of_missing_village 
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ) 
  (avg_pop : ℕ) 
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1023)
  (h4 : pop4 = 945)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000) :
  ∃ (pop_missing : ℕ), pop_missing = 1100 := 
by
  -- Placeholder for proof
  sorry

end population_of_missing_village_l172_172897


namespace power_of_power_l172_172103

theorem power_of_power {a : ℝ} : (a^2)^3 = a^6 := 
by
  sorry

end power_of_power_l172_172103


namespace samuel_remaining_distance_l172_172312

noncomputable def remaining_distance
  (total_distance : ℕ)
  (segment1_speed : ℕ) (segment1_time : ℕ)
  (segment2_speed : ℕ) (segment2_time : ℕ)
  (segment3_speed : ℕ) (segment3_time : ℕ)
  (segment4_speed : ℕ) (segment4_time : ℕ) : ℕ :=
  total_distance -
  (segment1_speed * segment1_time +
   segment2_speed * segment2_time +
   segment3_speed * segment3_time +
   segment4_speed * segment4_time)

theorem samuel_remaining_distance :
  remaining_distance 1200 60 2 70 3 50 4 80 5 = 270 :=
by
  sorry

end samuel_remaining_distance_l172_172312


namespace order_of_x_given_conditions_l172_172802

variables (x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅ : ℝ)

def system_equations :=
  x₁ + x₂ + x₃ = a₁ ∧
  x₂ + x₃ + x₄ = a₂ ∧
  x₃ + x₄ + x₅ = a₃ ∧
  x₄ + x₅ + x₁ = a₄ ∧
  x₅ + x₁ + x₂ = a₅

def a_descending_order :=
  a₁ > a₂ ∧
  a₂ > a₃ ∧
  a₃ > a₄ ∧
  a₄ > a₅

theorem order_of_x_given_conditions (h₁ : system_equations x₁ x₂ x₃ x₄ x₅ a₁ a₂ a₃ a₄ a₅) :
  a_descending_order a₁ a₂ a₃ a₄ a₅ →
  x₃ > x₁ ∧ x₁ > x₄ ∧ x₄ > x₂ ∧ x₂ > x₅ := sorry

end order_of_x_given_conditions_l172_172802


namespace sculpture_plus_base_height_l172_172034

def height_sculpture_feet : Nat := 2
def height_sculpture_inches : Nat := 10
def height_base_inches : Nat := 4

def height_sculpture_total_inches : Nat := height_sculpture_feet * 12 + height_sculpture_inches
def height_total_inches : Nat := height_sculpture_total_inches + height_base_inches

theorem sculpture_plus_base_height :
  height_total_inches = 38 := by
  sorry

end sculpture_plus_base_height_l172_172034


namespace binary_multiplication_correct_l172_172051

-- Define binary numbers as strings to directly use them in Lean
def binary_num1 : String := "1111"
def binary_num2 : String := "111"

-- Define a function to convert binary strings to natural numbers
def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => acc * 2 + (if c = '1' then 1 else 0)) 0

-- Define the target multiplication result
def binary_product_correct : Nat :=
  binary_to_nat "1001111"

theorem binary_multiplication_correct :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_product_correct :=
by
  sorry

end binary_multiplication_correct_l172_172051


namespace score_analysis_l172_172538

open Real

noncomputable def deviations : List ℝ := [8, -3, 12, -7, -10, -4, -8, 1, 0, 10]
def benchmark : ℝ := 85

theorem score_analysis :
  let highest_score := benchmark + List.maximum deviations
  let lowest_score := benchmark + List.minimum deviations
  let sum_deviations := List.sum deviations
  let average_deviation := sum_deviations / List.length deviations
  let average_score := benchmark + average_deviation
  highest_score = 97 ∧ lowest_score = 75 ∧ average_score = 84.9 :=
by
  sorry -- This is the placeholder for the proof

end score_analysis_l172_172538


namespace polar_coordinates_of_point_l172_172764

theorem polar_coordinates_of_point {x y : ℝ} (hx : x = -3) (hy : y = 1) :
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.pi - Real.arctan (y / abs x)
  r = Real.sqrt 10 ∧ θ = Real.pi - Real.arctan (1 / 3) := 
by
  rw [hx, hy]
  sorry

end polar_coordinates_of_point_l172_172764


namespace inequality_holds_l172_172406

variable {a b c : ℝ}

theorem inequality_holds (h : a > 0) (h' : b > 0) (h'' : c > 0) (h_abc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end inequality_holds_l172_172406


namespace solve_abs_eq_l172_172833

theorem solve_abs_eq (x : ℝ) : (|x - 3| = 5 - x) ↔ (x = 4) := 
by
  sorry

end solve_abs_eq_l172_172833


namespace part1_part2_part3_l172_172029

-- Definitions for conditions used in the proof problems
def eq1 (a b : ℝ) : Prop := 2 * a + b = 0
def eq2 (a x : ℝ) : Prop := x = a ^ 2

-- Part 1: Prove b = 4 and x = 4 given a = -2
theorem part1 (a b x : ℝ) (h1 : a = -2) (h2 : eq1 a b) (h3 : eq2 a x) : b = 4 ∧ x = 4 :=
by sorry

-- Part 2: Prove a = -3 and x = 9 given b = 6
theorem part2 (a b x : ℝ) (h1 : b = 6) (h2 : eq1 a b) (h3 : eq2 a x) : a = -3 ∧ x = 9 :=
by sorry

-- Part 3: Prove x = 2 given a^2*x + (a + b)^2*x = 8
theorem part3 (a b x : ℝ) (h : a^2 * x + (a + b)^2 * x = 8) : x = 2 :=
by sorry

end part1_part2_part3_l172_172029


namespace solve_system_of_equations_l172_172139

theorem solve_system_of_equations (x y_1 y_2 y_3: ℝ) (n : ℤ) (h1 : -3 ≤ n) (h2 : n ≤ 3)
  (h_eq1 : (1 - x^2) * y_1 = 2 * x)
  (h_eq2 : (1 - y_1^2) * y_2 = 2 * y_1)
  (h_eq3 : (1 - y_2^2) * y_3 = 2 * y_2)
  (h_eq4 : y_3 = x) :
  y_1 = Real.tan (2 * n * Real.pi / 7) ∧
  y_2 = Real.tan (4 * n * Real.pi / 7) ∧
  y_3 = Real.tan (n * Real.pi / 7) ∧
  x = Real.tan (n * Real.pi / 7) :=
sorry

end solve_system_of_equations_l172_172139


namespace expand_and_simplify_l172_172503

theorem expand_and_simplify : ∀ x : ℝ, (7 * x + 5) * 3 * x^2 = 21 * x^3 + 15 * x^2 :=
by
  intro x
  sorry

end expand_and_simplify_l172_172503


namespace equation_of_latus_rectum_l172_172271

theorem equation_of_latus_rectum (y x : ℝ) : (x = -1/4) ∧ (y^2 = x) ↔ (2 * (1 / 2) = 1) ∧ (l = - (1 / 2) / 2) := sorry

end equation_of_latus_rectum_l172_172271


namespace product_of_four_integers_negative_l172_172492

theorem product_of_four_integers_negative {a b c d : ℤ}
  (h : a * b * c * d < 0) :
  (∃ n : ℕ, n ≤ 3 ∧ (n = (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0))) :=
sorry

end product_of_four_integers_negative_l172_172492


namespace maximum_visibility_sum_l172_172215

theorem maximum_visibility_sum (X Y : ℕ) (h : X + 2 * Y = 30) :
  X * Y ≤ 112 :=
by
  sorry

end maximum_visibility_sum_l172_172215


namespace smallest_divisible_by_15_16_18_l172_172147

def factors_of_15 : Prop := 15 = 3 * 5
def factors_of_16 : Prop := 16 = 2^4
def factors_of_18 : Prop := 18 = 2 * 3^2

theorem smallest_divisible_by_15_16_18 (h1: factors_of_15) (h2: factors_of_16) (h3: factors_of_18) : 
  ∃ n, n > 0 ∧ n % 15 = 0 ∧ n % 16 = 0 ∧ n % 18 = 0 ∧ n = 720 :=
by
  sorry

end smallest_divisible_by_15_16_18_l172_172147


namespace yogurt_packs_ordered_l172_172996

theorem yogurt_packs_ordered (P : ℕ) (price_per_pack refund_amount : ℕ) (expired_percentage : ℚ)
  (h1 : price_per_pack = 12)
  (h2 : refund_amount = 384)
  (h3 : expired_percentage = 0.40)
  (h4 : refund_amount / price_per_pack = 32)
  (h5 : 32 / expired_percentage = P) :
  P = 80 :=
sorry

end yogurt_packs_ordered_l172_172996


namespace space_between_trees_l172_172915

theorem space_between_trees (n_trees : ℕ) (tree_space : ℕ) (total_length : ℕ) (spaces_between_trees : ℕ) (result_space : ℕ) 
  (h1 : n_trees = 8)
  (h2 : tree_space = 1)
  (h3 : total_length = 148)
  (h4 : spaces_between_trees = n_trees - 1)
  (h5 : result_space = (total_length - n_trees * tree_space) / spaces_between_trees) : 
  result_space = 20 := 
by sorry

end space_between_trees_l172_172915


namespace cost_of_adult_ticket_l172_172264

theorem cost_of_adult_ticket
  (A : ℝ) -- Cost of an adult ticket in dollars
  (x y : ℝ) -- Number of children tickets and number of adult tickets respectively
  (hx : x = 90) -- Condition: number of children tickets sold
  (hSum : x + y = 130) -- Condition: total number of tickets sold
  (hTotal : 4 * x + A * y = 840) -- Condition: total receipts from all tickets
  : A = 12 := 
by
  -- Proof is skipped as per instruction
  sorry

end cost_of_adult_ticket_l172_172264


namespace percentage_exceeds_l172_172978

theorem percentage_exceeds (N P : ℕ) (h1 : N = 50) (h2 : N = (P / 100) * N + 42) : P = 16 :=
sorry

end percentage_exceeds_l172_172978


namespace tangent_line_to_circle_l172_172584

theorem tangent_line_to_circle :
  ∀ (x y : ℝ), x^2 + y^2 = 5 → (x = 2 → y = -1 → 2 * x - y - 5 = 0) :=
by
  intros x y h_circle hx hy
  sorry

end tangent_line_to_circle_l172_172584


namespace daily_evaporation_rate_l172_172076

theorem daily_evaporation_rate (initial_amount : ℝ) (period : ℕ) (percentage_evaporated : ℝ) (h_initial : initial_amount = 10) (h_period : period = 50) (h_percentage : percentage_evaporated = 4) : 
  (percentage_evaporated / 100 * initial_amount) / period = 0.008 :=
by
  -- Ensures that the conditions translate directly into the Lean theorem statement
  rw [h_initial, h_period, h_percentage]
  -- Insert the required logical proof here
  sorry

end daily_evaporation_rate_l172_172076


namespace georgia_total_cost_l172_172247

def carnation_price : ℝ := 0.50
def dozen_price : ℝ := 4.00
def teachers : ℕ := 5
def friends : ℕ := 14

theorem georgia_total_cost :
  ((dozen_price * teachers) + dozen_price + (carnation_price * (friends - 12))) = 25.00 :=
by
  sorry

end georgia_total_cost_l172_172247


namespace find_positive_integer_l172_172377

theorem find_positive_integer (n : ℕ) (hn_pos : n > 0) :
  (∃ a b : ℕ, n = a^2 ∧ n + 100 = b^2) → n = 576 :=
by sorry

end find_positive_integer_l172_172377


namespace find_x_value_l172_172166

def solve_for_x (a b x : ℝ) (rectangle_perimeter triangle_height equated_areas : Prop) :=
  rectangle_perimeter -> triangle_height -> equated_areas -> x = 20 / 3

-- Definitions of the conditions
def rectangle_perimeter (a b : ℝ) : Prop := 2 * (a + b) = 60
def triangle_height : Prop := 60 > 0
def equated_areas (a b x : ℝ) : Prop := a * b = 30 * x

theorem find_x_value :
  ∃ a b x : ℝ, solve_for_x a b x (rectangle_perimeter a b) triangle_height (equated_areas a b x) :=
  sorry

end find_x_value_l172_172166


namespace wyatt_total_envelopes_l172_172773

theorem wyatt_total_envelopes :
  let b := 10
  let y := b - 4
  let t := b + y
  t = 16 :=
by
  let b := 10
  let y := b - 4
  let t := b + y
  sorry

end wyatt_total_envelopes_l172_172773


namespace solution_inequality_l172_172206

theorem solution_inequality (m : ℝ) :
  (∀ x : ℝ, x^2 - (m+3)*x + 3*m < 0 ↔ m ∈ Set.Icc 3 (-1) ∪ Set.Icc 6 7) →
  m = -1/2 ∨ m = 13/2 :=
sorry

end solution_inequality_l172_172206


namespace scientific_notation_l172_172415

theorem scientific_notation (x : ℝ) (a : ℝ) (n : ℤ) (h₁ : x = 5853) (h₂ : 1 ≤ |a|) (h₃ : |a| < 10) (h₄ : x = a * 10^n) : 
  a = 5.853 ∧ n = 3 :=
by sorry

end scientific_notation_l172_172415


namespace probability_XOXOXOX_is_1_div_35_l172_172761

def count_combinations (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

def num_ways_to_choose_positions_for_X (total_positions : ℕ) (num_X : ℕ) : ℕ := 
  count_combinations total_positions num_X

def num_ways_for_specific_arrangement_XOXOXOX : ℕ := 1

def probability_of_XOXOXOX (num_ways_total : ℕ) (num_ways_specific : ℕ) : ℚ := 
  num_ways_specific / num_ways_total

theorem probability_XOXOXOX_is_1_div_35 :
  probability_of_XOXOXOX (num_ways_to_choose_positions_for_X 7 4) num_ways_for_specific_arrangement_XOXOXOX = 1 / 35 := by
  sorry

end probability_XOXOXOX_is_1_div_35_l172_172761


namespace latus_rectum_parabola_l172_172885

theorem latus_rectum_parabola : 
  ∀ (x y : ℝ), (x = 4 * y^2) → (x = -1/16) :=
by 
  sorry

end latus_rectum_parabola_l172_172885


namespace andrew_paid_correct_amount_l172_172732

-- Definitions of the conditions
def cost_of_grapes : ℝ := 7 * 68
def cost_of_mangoes : ℝ := 9 * 48
def cost_of_apples : ℝ := 5 * 55
def cost_of_oranges : ℝ := 4 * 38

def total_cost_grapes_and_mangoes_before_discount : ℝ := cost_of_grapes + cost_of_mangoes
def discount_on_grapes_and_mangoes : ℝ := 0.10 * total_cost_grapes_and_mangoes_before_discount
def total_cost_grapes_and_mangoes_after_discount : ℝ := total_cost_grapes_and_mangoes_before_discount - discount_on_grapes_and_mangoes

def total_cost_all_fruits_before_tax : ℝ := total_cost_grapes_and_mangoes_after_discount + cost_of_apples + cost_of_oranges
def sales_tax : ℝ := 0.05 * total_cost_all_fruits_before_tax
def total_amount_to_pay : ℝ := total_cost_all_fruits_before_tax + sales_tax

-- Statement to be proved
theorem andrew_paid_correct_amount :
  total_amount_to_pay = 1306.41 :=
by
  sorry

end andrew_paid_correct_amount_l172_172732


namespace min_value_of_a_plus_2b_l172_172045

theorem min_value_of_a_plus_2b (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 / a + 1 / b = 1) : a + 2 * b = 4 :=
sorry

end min_value_of_a_plus_2b_l172_172045


namespace count_four_digit_numbers_ending_25_l172_172800

theorem count_four_digit_numbers_ending_25 : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 10000 ∧ n ≡ 25 [MOD 100]) → ∃ n : ℕ, n = 100 :=
by
  sorry

end count_four_digit_numbers_ending_25_l172_172800


namespace leo_trousers_count_l172_172145

theorem leo_trousers_count (S T : ℕ) (h1 : 5 * S + 9 * T = 140) (h2 : S = 10) : T = 10 :=
by
  sorry

end leo_trousers_count_l172_172145


namespace count_three_digit_numbers_between_l172_172274

theorem count_three_digit_numbers_between 
  (a b : ℕ) 
  (ha : a = 137) 
  (hb : b = 285) : 
  ∃ n, n = (b - a - 1) + 1 := 
sorry

end count_three_digit_numbers_between_l172_172274


namespace denomination_of_second_note_l172_172642

theorem denomination_of_second_note
  (x : ℕ)
  (y : ℕ)
  (z : ℕ)
  (h1 : x = y)
  (h2 : y = z)
  (h3 : x + y + z = 75)
  (h4 : 1 * x + y * x + 10 * x = 400):
  y = 5 := by
  sorry

end denomination_of_second_note_l172_172642


namespace percentage_employed_females_is_16_l172_172484

/- 
  In Town X, the population is divided into three age groups: 18-34, 35-54, and 55+.
  For each age group, the percentage of the employed population is 64%, and the percentage of employed males is 48%.
  We need to prove that the percentage of employed females in each age group is 16%.
-/

theorem percentage_employed_females_is_16
  (percentage_employed_population : ℝ)
  (percentage_employed_males : ℝ)
  (h1 : percentage_employed_population = 0.64)
  (h2 : percentage_employed_males = 0.48) :
  percentage_employed_population - percentage_employed_males = 0.16 :=
by
  rw [h1, h2]
  norm_num
  exact sorry

end percentage_employed_females_is_16_l172_172484


namespace max_pieces_from_cake_l172_172482

theorem max_pieces_from_cake (large_cake_area small_piece_area : ℕ) 
  (h_large_cake : large_cake_area = 15 * 15) 
  (h_small_piece : small_piece_area = 5 * 5) :
  large_cake_area / small_piece_area = 9 := 
by
  sorry

end max_pieces_from_cake_l172_172482


namespace regular_triangular_prism_cosine_l172_172000

-- Define the regular triangular prism and its properties
structure RegularTriangularPrism :=
  (side : ℝ) -- the side length of the base and the lateral edge

-- Define the vertices of the prism
structure Vertices :=
  (A : ℝ × ℝ × ℝ) 
  (B : ℝ × ℝ × ℝ) 
  (C : ℝ × ℝ × ℝ)
  (A1 : ℝ × ℝ × ℝ)
  (B1 : ℝ × ℝ × ℝ)
  (C1 : ℝ × ℝ × ℝ)

-- Define the cosine calculation
def cos_angle (prism : RegularTriangularPrism) (v : Vertices) : ℝ := sorry

-- Prove that the cosine of the angle between diagonals AB1 and BC1 is 1/4
theorem regular_triangular_prism_cosine (prism : RegularTriangularPrism) (v : Vertices)
  : cos_angle prism v = 1 / 4 :=
sorry

end regular_triangular_prism_cosine_l172_172000


namespace neg_P_l172_172993

def P := ∃ x : ℝ, (0 < x) ∧ (3^x < x^3)

theorem neg_P : ¬P ↔ ∀ x : ℝ, (0 < x) → (3^x ≥ x^3) :=
by
  sorry

end neg_P_l172_172993


namespace proof_problem_l172_172332

variable {α : Type*} [LinearOrderedField α]

theorem proof_problem 
  (a b x y : α) 
  (h0 : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y)
  (h1 : a + b + x + y < 2)
  (h2 : a + b^2 = x + y^2)
  (h3 : a^2 + b = x^2 + y) :
  a = x ∧ b = y := 
by
  sorry

end proof_problem_l172_172332


namespace ground_beef_lean_beef_difference_l172_172972

theorem ground_beef_lean_beef_difference (x y z : ℕ) 
  (h1 : x + y + z = 20) 
  (h2 : y + 2 * z = 18) :
  x - z = 2 :=
sorry

end ground_beef_lean_beef_difference_l172_172972


namespace commission_percentage_is_4_l172_172898

-- Define the given conditions
def commission := 12.50
def total_sales := 312.5

-- The problem is to prove the commission percentage
theorem commission_percentage_is_4 :
  (commission / total_sales) * 100 = 4 := by
  sorry

end commission_percentage_is_4_l172_172898


namespace equal_number_of_experienced_fishermen_and_children_l172_172940

theorem equal_number_of_experienced_fishermen_and_children 
  (n : ℕ)
  (total_fish : ℕ)
  (children_catch : ℕ)
  (fishermen_catch : ℕ)
  (h1 : total_fish = n^2 + 5 * n + 22)
  (h2 : fishermen_catch - 10 = children_catch)
  (h3 : total_fish = n * children_catch + 11 * fishermen_catch)
  (h4 : fishermen_catch > children_catch)
  : n = 11 := 
sorry

end equal_number_of_experienced_fishermen_and_children_l172_172940


namespace probability_succeeding_third_attempt_l172_172475

theorem probability_succeeding_third_attempt :
  let total_keys := 5
  let successful_keys := 2
  let attempts := 3
  let prob := successful_keys / total_keys * (successful_keys / (total_keys - 1)) * (successful_keys / (total_keys - 2))
  prob = 1 / 5 := by
sorry

end probability_succeeding_third_attempt_l172_172475


namespace cone_volume_l172_172786

theorem cone_volume (V_cyl : ℝ) (r h : ℝ) (h_cyl : V_cyl = 150 * Real.pi) :
  (1 / 3) * V_cyl = 50 * Real.pi :=
by
  rw [h_cyl]
  ring


end cone_volume_l172_172786


namespace probability_at_least_one_trip_l172_172348

theorem probability_at_least_one_trip (p_A_trip : ℚ) (p_B_trip : ℚ)
  (h1 : p_A_trip = 1/4) (h2 : p_B_trip = 1/5) :
  (1 - ((1 - p_A_trip) * (1 - p_B_trip))) = 2/5 :=
by
  sorry

end probability_at_least_one_trip_l172_172348


namespace solve_for_x_l172_172545

theorem solve_for_x (x : ℤ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 4) : 
  x = -15 := 
by
  sorry

end solve_for_x_l172_172545


namespace points_equidistant_from_circle_and_tangents_l172_172881

noncomputable def circle_radius := 4
noncomputable def tangent_distance := 6

theorem points_equidistant_from_circle_and_tangents :
  ∃! (P : ℝ × ℝ), dist P (0, 0) = circle_radius ∧
                 dist P (0, tangent_distance) = tangent_distance - circle_radius ∧
                 dist P (0, -tangent_distance) = tangent_distance - circle_radius :=
by {
  sorry
}

end points_equidistant_from_circle_and_tangents_l172_172881


namespace maple_taller_than_birch_l172_172050

def birch_tree_height : ℚ := 49 / 4
def maple_tree_height : ℚ := 102 / 5

theorem maple_taller_than_birch : maple_tree_height - birch_tree_height = 163 / 20 :=
by
  sorry

end maple_taller_than_birch_l172_172050


namespace find_number_l172_172904

theorem find_number (x : ℝ) (h : 0.7 * x = 48 + 22) : x = 100 :=
by
  sorry

end find_number_l172_172904


namespace remainder_when_divided_by_7_l172_172589

theorem remainder_when_divided_by_7
  (x : ℤ) (k : ℤ) (h : x = 52 * k + 19) : x % 7 = 5 :=
sorry

end remainder_when_divided_by_7_l172_172589


namespace george_borrow_amount_l172_172542

-- Define the conditions
def initial_fee_rate : ℝ := 0.05
def doubling_rate : ℝ := 2
def total_weeks : ℕ := 2
def total_fee : ℝ := 15

-- Define the problem statement
theorem george_borrow_amount : 
  ∃ (P : ℝ), (initial_fee_rate * P + initial_fee_rate * doubling_rate * P = total_fee) ∧ P = 100 :=
by
  -- Statement only, proof is skipped
  sorry

end george_borrow_amount_l172_172542


namespace smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l172_172621

def is_not_prime (n : ℕ) : Prop := ¬ Prime n

def is_not_square (n : ℕ) : Prop := ∀ m : ℕ, m * m ≠ n

def no_prime_factor_less_than_50 (n : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → p ∣ n → p ≥ 50

theorem smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50 :
  (∃ n : ℕ, 0 < n ∧ is_not_prime n ∧ is_not_square n ∧ no_prime_factor_less_than_50 n ∧
  (∀ m : ℕ, 0 < m ∧ is_not_prime m ∧ is_not_square m ∧ no_prime_factor_less_than_50 m → n ≤ m)) →
  ∃ n : ℕ, n = 3127 :=
by {
  sorry
}

end smallest_positive_integer_neither_prime_nor_square_no_prime_factor_less_than_50_l172_172621


namespace total_area_of_pyramid_faces_l172_172256

theorem total_area_of_pyramid_faces (base_edge lateral_edge : ℝ) (h : base_edge = 8) (k : lateral_edge = 5) : 
  4 * (1 / 2 * base_edge * 3) = 48 :=
by
  -- Base edge of the pyramid
  let b := base_edge
  -- Lateral edge of the pyramid
  let l := lateral_edge
  -- Half of the base
  let half_b := 4
  -- Height of the triangular face using Pythagorean theorem
  let h := 3
  -- Total area of four triangular faces
  have triangular_face_area : 1 / 2 * base_edge * h = 12 := sorry
  have total_area_of_faces : 4 * (1 / 2 * base_edge * h) = 48 := sorry
  exact total_area_of_faces

end total_area_of_pyramid_faces_l172_172256


namespace student_chose_number_l172_172739

theorem student_chose_number :
  ∃ x : ℕ, 7 * x - 150 = 130 ∧ x = 40 := sorry

end student_chose_number_l172_172739


namespace comb_identity_a_l172_172152

theorem comb_identity_a (r m k : ℕ) (h : 0 ≤ k ∧ k ≤ m ∧ m ≤ r) :
  Nat.choose r m * Nat.choose m k = Nat.choose r k * Nat.choose (r - k) (m - k) :=
sorry

end comb_identity_a_l172_172152


namespace alster_caught_two_frogs_l172_172754

-- Definitions and conditions
variables (alster quinn bret : ℕ)

-- Condition 1: Quinn catches twice the amount of frogs as Alster
def quinn_catches_twice_as_alster : Prop := quinn = 2 * alster

-- Condition 2: Bret catches three times the amount of frogs as Quinn
def bret_catches_three_times_as_quinn : Prop := bret = 3 * quinn

-- Condition 3: Bret caught 12 frogs
def bret_caught_twelve : Prop := bret = 12

-- Theorem: How many frogs did Alster catch? Alster caught 2 frogs
theorem alster_caught_two_frogs (h1 : quinn_catches_twice_as_alster alster quinn)
                                (h2 : bret_catches_three_times_as_quinn quinn bret)
                                (h3 : bret_caught_twelve bret) :
                                alster = 2 :=
by sorry

end alster_caught_two_frogs_l172_172754


namespace pablo_days_to_complete_puzzles_l172_172122

-- Define the given conditions 
def puzzle_pieces_300 := 300
def puzzle_pieces_500 := 500
def puzzles_300 := 8
def puzzles_500 := 5
def rate_per_hour := 100
def max_hours_per_day := 7

-- Calculate total number of pieces
def total_pieces_300 := puzzles_300 * puzzle_pieces_300
def total_pieces_500 := puzzles_500 * puzzle_pieces_500
def total_pieces := total_pieces_300 + total_pieces_500

-- Calculate the number of pieces Pablo can put together per day
def pieces_per_day := max_hours_per_day * rate_per_hour

-- Calculate the number of days required for Pablo to complete all puzzles
def days_to_complete := total_pieces / pieces_per_day

-- Proposition to prove
theorem pablo_days_to_complete_puzzles : days_to_complete = 7 := sorry

end pablo_days_to_complete_puzzles_l172_172122


namespace expression_indeterminate_l172_172101

-- Given variables a, b, c, d which are real numbers
variables {a b c d : ℝ}

-- Statement asserting that the expression is indeterminate under given conditions
theorem expression_indeterminate
  (h : true) :
  ¬∃ k, (a^2 + b^2 - c^2 - 2 * b * d)/(a^2 + c^2 - b^2 - 2 * c * d) = k :=
sorry

end expression_indeterminate_l172_172101


namespace positional_relationship_l172_172094

-- Defining the concepts of parallelism, containment, and positional relationships
structure Line -- subtype for a Line
structure Plane -- subtype for a Plane

-- Definitions and Conditions
def is_parallel_to (l : Line) (p : Plane) : Prop := sorry  -- A line being parallel to a plane
def is_contained_in (l : Line) (p : Plane) : Prop := sorry  -- A line being contained within a plane
def are_skew (l₁ l₂ : Line) : Prop := sorry  -- Two lines being skew
def are_parallel (l₁ l₂ : Line) : Prop := sorry  -- Two lines being parallel

-- Given conditions
variables (a b : Line) (α : Plane)
axiom Ha : is_parallel_to a α
axiom Hb : is_contained_in b α

-- The theorem to be proved
theorem positional_relationship (a b : Line) (α : Plane) 
  (Ha : is_parallel_to a α) 
  (Hb : is_contained_in b α) : 
  (are_skew a b ∨ are_parallel a b) :=
sorry

end positional_relationship_l172_172094


namespace population_exceeds_l172_172072

theorem population_exceeds (n : ℕ) : (∃ n, 4 * 3^n > 200) ∧ ∀ m, m < n → 4 * 3^m ≤ 200 := by
  sorry

end population_exceeds_l172_172072


namespace lice_checks_time_in_hours_l172_172121

-- Define the number of students in each grade
def kindergarteners : ℕ := 26
def first_graders : ℕ := 19
def second_graders : ℕ := 20
def third_graders : ℕ := 25

-- Define the time each check takes (in minutes)
def time_per_check : ℕ := 2

-- Define the conversion factor from minutes to hours
def minutes_per_hour : ℕ := 60

-- The theorem states that the total time in hours is 3
theorem lice_checks_time_in_hours : 
  ((kindergarteners + first_graders + second_graders + third_graders) * time_per_check) / minutes_per_hour = 3 := 
by
  sorry

end lice_checks_time_in_hours_l172_172121


namespace a_gt_b_neither_sufficient_nor_necessary_l172_172016

theorem a_gt_b_neither_sufficient_nor_necessary (a b : ℝ) : ¬ ((a > b → a^2 > b^2) ∧ (a^2 > b^2 → a > b)) := 
sorry

end a_gt_b_neither_sufficient_nor_necessary_l172_172016


namespace train_travel_time_l172_172044

theorem train_travel_time
  (a : ℝ) (s : ℝ) (t : ℝ)
  (ha : a = 3)
  (hs : s = 27)
  (h0 : ∀ t, 0 ≤ t) :
  t = Real.sqrt 18 :=
by
  sorry

end train_travel_time_l172_172044


namespace total_trees_planted_l172_172222

/-- A yard is 255 meters long, with a tree at each end and trees planted at intervals of 15 meters. -/
def yard_length : ℤ := 255

def tree_interval : ℤ := 15

def total_trees : ℤ := 18

theorem total_trees_planted (L : ℤ) (d : ℤ) (n : ℤ) : 
  L = yard_length →
  d = tree_interval →
  n = total_trees →
  n = (L / d) + 1 :=
by
  intros hL hd hn
  rw [hL, hd, hn]
  sorry

end total_trees_planted_l172_172222


namespace convert_157_base_10_to_base_7_l172_172420

-- Given
def base_10_to_base_7(n : ℕ) : String := "313"

-- Prove
theorem convert_157_base_10_to_base_7 : base_10_to_base_7 157 = "313" := by
  sorry

end convert_157_base_10_to_base_7_l172_172420


namespace operation_multiplication_in_P_l172_172303

-- Define the set P
def P : Set ℕ := {n | ∃ k : ℕ, n = k^2}

-- Define the operation "*" as multiplication within the set P
def operation (a b : ℕ) : ℕ := a * b

-- Define the property to be proved
theorem operation_multiplication_in_P (a b : ℕ)
  (ha : a ∈ P) (hb : b ∈ P) : operation a b ∈ P :=
sorry

end operation_multiplication_in_P_l172_172303


namespace find_other_endpoint_l172_172263

theorem find_other_endpoint (x1 y1 x_m y_m x y : ℝ) 
  (h1 : (x_m, y_m) = (3, 7))
  (h2 : (x1, y1) = (0, 11)) :
  (x, y) = (6, 3) ↔ (x_m = (x1 + x) / 2 ∧ y_m = (y1 + y) / 2) :=
by
  simp at h1 h2
  simp
  sorry

end find_other_endpoint_l172_172263


namespace common_ratio_is_half_l172_172720

variable {a₁ q : ℝ}

-- Given the conditions of the geometric sequence

-- First condition
axiom h1 : a₁ + a₁ * q ^ 2 = 10

-- Second condition
axiom h2 : a₁ * q ^ 3 + a₁ * q ^ 5 = 5 / 4

-- Proving that the common ratio q is 1/2
theorem common_ratio_is_half : q = 1 / 2 :=
by
  -- The proof details will be filled in here.
  sorry

end common_ratio_is_half_l172_172720


namespace farmer_ear_count_l172_172892

theorem farmer_ear_count
    (seeds_per_ear : ℕ)
    (price_per_ear : ℝ)
    (cost_per_bag : ℝ)
    (seeds_per_bag : ℕ)
    (profit : ℝ)
    (target_profit : ℝ) :
  seeds_per_ear = 4 →
  price_per_ear = 0.1 →
  cost_per_bag = 0.5 →
  seeds_per_bag = 100 →
  target_profit = 40 →
  profit = price_per_ear - ((cost_per_bag / seeds_per_bag) * seeds_per_ear) →
  target_profit / profit = 500 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end farmer_ear_count_l172_172892


namespace least_number_to_subtract_l172_172506

theorem least_number_to_subtract 
  (n : ℤ) 
  (h1 : 7 ∣ (90210 - n + 12)) 
  (h2 : 11 ∣ (90210 - n + 12)) 
  (h3 : 13 ∣ (90210 - n + 12)) 
  (h4 : 17 ∣ (90210 - n + 12)) 
  (h5 : 19 ∣ (90210 - n + 12)) : 
  n = 90198 :=
sorry

end least_number_to_subtract_l172_172506


namespace seashells_total_l172_172281

theorem seashells_total {sally tom jessica : ℕ} (h₁ : sally = 9) (h₂ : tom = 7) (h₃ : jessica = 5) : sally + tom + jessica = 21 := by
  sorry

end seashells_total_l172_172281


namespace squirrel_rainy_days_l172_172469

theorem squirrel_rainy_days (s r : ℕ) (h1 : 20 * s + 12 * r = 112) (h2 : s + r = 8) : r = 6 :=
by {
  -- sorry to skip the proof
  sorry
}

end squirrel_rainy_days_l172_172469


namespace circle_radius_l172_172520

theorem circle_radius (m : ℝ) (h : 2 * 1 + (-m / 2) = 0) :
  let radius := 1 / 2 * Real.sqrt (4 + m ^ 2 + 16)
  radius = 3 :=
by
  sorry

end circle_radius_l172_172520


namespace contrapositive_example_l172_172659

theorem contrapositive_example 
  (x y : ℝ) (h : x^2 + y^2 = 0 → x = 0 ∧ y = 0) : 
  (x ≠ 0 ∨ y ≠ 0) → x^2 + y^2 ≠ 0 :=
sorry

end contrapositive_example_l172_172659


namespace part_a_part_b_l172_172995

-- Define the tower of exponents function for convenience
def tower (base : ℕ) (height : ℕ) : ℕ :=
  if height = 0 then 1 else base^(tower base (height - 1))

-- Part a: Tower of 3s with height 99 is greater than Tower of 2s with height 100
theorem part_a : tower 3 99 > tower 2 100 := sorry

-- Part b: Tower of 3s with height 100 is greater than Tower of 3s with height 99
theorem part_b : tower 3 100 > tower 3 99 := sorry

end part_a_part_b_l172_172995


namespace manuscript_fee_tax_l172_172676

theorem manuscript_fee_tax (fee : ℕ) (tax_paid : ℕ) :
  (tax_paid = 0 ∧ fee ≤ 800) ∨ 
  (tax_paid = (14 * (fee - 800) / 100) ∧ 800 < fee ∧ fee ≤ 4000) ∨ 
  (tax_paid = 11 * fee / 100 ∧ fee > 4000) →
  tax_paid = 420 →
  fee = 3800 :=
by 
  intro h_eq h_tax;
  sorry

end manuscript_fee_tax_l172_172676


namespace value_of_a_minus_b_l172_172056

variables (a b : ℝ)

theorem value_of_a_minus_b (h1 : abs a = 3) (h2 : abs b = 5) (h3 : a > b) : a - b = 8 :=
sorry

end value_of_a_minus_b_l172_172056


namespace integer_pair_condition_l172_172921

theorem integer_pair_condition (m n : ℤ) (h : (m^2 + m * n + n^2 : ℚ) / (m + 2 * n) = 13 / 3) : m + 2 * n = 9 :=
sorry

end integer_pair_condition_l172_172921


namespace option_A_two_solutions_l172_172569

theorem option_A_two_solutions :
    (∀ (a b : ℝ) (A : ℝ), 
    (a = 3 ∧ b = 4 ∧ A = 45) ∨ 
    (a = 7 ∧ b = 14 ∧ A = 30) ∨ 
    (a = 2 ∧ b = 7 ∧ A = 60) ∨ 
    (a = 8 ∧ b = 5 ∧ A = 135) →
    (∃ a b A : ℝ, a = 3 ∧ b = 4 ∧ A = 45 ∧ 2 = 2)) :=
by
  sorry

end option_A_two_solutions_l172_172569


namespace numWaysElectOfficers_l172_172262

-- Definitions and conditions from part (a)
def numMembers : Nat := 30
def numPositions : Nat := 5
def members := ["Alice", "Bob", "Carol", "Dave"]
def allOrNoneCondition (S : List String) : Bool := 
  S.all (members.contains)

-- Function to count the number of ways to choose the officers
def countWays (n : Nat) (k : Nat) (allOrNone : Bool) : Nat :=
if allOrNone then
  -- All four members are positioned
  Nat.factorial k * (n - k)
else
  -- None of the four members are positioned
  let remaining := n - members.length
  remaining * (remaining - 1) * (remaining - 2) * (remaining - 3) * (remaining - 4)

theorem numWaysElectOfficers :
  let casesWithNone := countWays numMembers numPositions false
  let casesWithAll := countWays numMembers numPositions true
  (casesWithNone + casesWithAll) = 6378720 :=
by
  sorry

end numWaysElectOfficers_l172_172262


namespace selected_six_numbers_have_two_correct_statements_l172_172838

def selection := {n : ℕ // 1 ≤ n ∧ n ≤ 11}

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_multiple (a b : ℕ) : Prop := a ≠ b ∧ (b % a = 0 ∨ a % b = 0)

def is_double_multiple (a b : ℕ) : Prop := a ≠ b ∧ (2 * a = b ∨ 2 * b = a)

theorem selected_six_numbers_have_two_correct_statements (s : Finset selection) (h : s.card = 6) :
  ∃ n1 n2 : selection, is_coprime n1.1 n2.1 ∧ ∃ n1 n2 : selection, is_double_multiple n1.1 n2.1 :=
by
  -- The detailed proof is omitted.
  sorry

end selected_six_numbers_have_two_correct_statements_l172_172838


namespace integral_negative_of_negative_function_l172_172229

theorem integral_negative_of_negative_function {f : ℝ → ℝ} 
  (hf_cont : Continuous f) 
  (hf_neg : ∀ x, f x < 0) 
  {a b : ℝ} 
  (hab : a < b) 
  : ∫ x in a..b, f x < 0 := 
sorry

end integral_negative_of_negative_function_l172_172229


namespace garden_roller_area_l172_172974

theorem garden_roller_area (length : ℝ) (area_5rev : ℝ) (d1 d2 : ℝ) (π : ℝ) :
  length = 4 ∧ area_5rev = 88 ∧ π = 22 / 7 ∧ d2 = 1.4 →
  let circumference := π * d2
  let area_rev := circumference * length
  let new_area_5rev := 5 * area_rev
  new_area_5rev = 88 :=
by
  sorry

end garden_roller_area_l172_172974


namespace time_for_one_essay_l172_172249

-- We need to define the times for questions and paragraphs first.

def time_per_short_answer_question := 3 -- in minutes
def time_per_paragraph := 15 -- in minutes
def total_homework_time := 4 -- in hours
def num_essays := 2
def num_paragraphs := 5
def num_short_answer_questions := 15

-- Now we need to state the total homework time and define the goal
def computed_homework_time :=
  (time_per_short_answer_question * num_short_answer_questions +
   time_per_paragraph * num_paragraphs) / 60 + num_essays * sorry -- time for one essay in hours

theorem time_for_one_essay :
  (total_homework_time = computed_homework_time) → sorry = 1 :=
by
  sorry

end time_for_one_essay_l172_172249


namespace unique_tangent_lines_through_point_l172_172543

theorem unique_tangent_lines_through_point (P : ℝ × ℝ) (hP : P = (2, 4)) :
  ∃! l : ℝ × ℝ → Prop, (l P) ∧ (∀ p : ℝ × ℝ, l p → p ∈ {p : ℝ × ℝ | p.2 ^ 2 = 8 * p.1}) := sorry

end unique_tangent_lines_through_point_l172_172543


namespace find_ABC_l172_172210

theorem find_ABC :
    ∃ (A B C : ℚ), 
    (∀ x : ℚ, x ≠ 2 ∧ x ≠ 4 ∧ x ≠ 5 → 
        (x^2 - 9) / ((x - 2) * (x - 4) * (x - 5)) = A / (x - 2) + B / (x - 4) + C / (x - 5)) 
    ∧ A = 5 / 3 ∧ B = -7 / 2 ∧ C = 8 / 3 := 
sorry

end find_ABC_l172_172210


namespace track_length_l172_172436

theorem track_length
  (x : ℕ)
  (run1_Brenda : x / 2 + 80 = a)
  (run2_Sally : x / 2 + 100 = b)
  (run1_ratio : 80 / (x / 2 - 80) = c)
  (run2_ratio : (x / 2 - 100) / (x / 2 + 100) = c)
  : x = 520 :=
by sorry

end track_length_l172_172436


namespace probability_at_least_one_five_or_six_l172_172723

theorem probability_at_least_one_five_or_six
  (P_neither_five_nor_six: ℚ)
  (h: P_neither_five_nor_six = 4 / 9) :
  (1 - P_neither_five_nor_six) = 5 / 9 :=
by
  sorry

end probability_at_least_one_five_or_six_l172_172723


namespace quadratic_inequality_solution_set_l172_172399

theorem quadratic_inequality_solution_set :
  {x : ℝ | x * (x - 2) > 0} = {x : ℝ | x < 0} ∪ {x : ℝ | x > 2} :=
by
  sorry

end quadratic_inequality_solution_set_l172_172399


namespace arithmetic_sequence_sum_l172_172510

-- Definitions based on conditions from step a
def first_term : ℕ := 1
def last_term : ℕ := 36
def num_terms : ℕ := 8

-- The problem statement in Lean 4
theorem arithmetic_sequence_sum :
  (num_terms / 2) * (first_term + last_term) = 148 := by
  sorry

end arithmetic_sequence_sum_l172_172510


namespace train_a_distance_at_meeting_l172_172120

-- Define the problem conditions as constants
def distance := 75 -- distance between start points of Train A and B
def timeA := 3 -- time taken by Train A to complete the trip in hours
def timeB := 2 -- time taken by Train B to complete the trip in hours

-- Calculate the speeds
def speedA := distance / timeA -- speed of Train A in miles per hour
def speedB := distance / timeB -- speed of Train B in miles per hour

-- Calculate the combined speed and time to meet
def combinedSpeed := speedA + speedB
def timeToMeet := distance / combinedSpeed

-- Define the distance traveled by Train A at the time of meeting
def distanceTraveledByTrainA := speedA * timeToMeet

-- Theorem stating Train A has traveled 30 miles when it met Train B
theorem train_a_distance_at_meeting : distanceTraveledByTrainA = 30 := by
  sorry

end train_a_distance_at_meeting_l172_172120


namespace x_plus_y_possible_values_l172_172882

theorem x_plus_y_possible_values (x y : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : x < 20) (h4 : y < 20) (h5 : x + y + x * y = 99) : 
  x + y = 23 ∨ x + y = 18 :=
by
  sorry

end x_plus_y_possible_values_l172_172882


namespace max_geometric_progression_terms_l172_172514

theorem max_geometric_progression_terms :
  ∀ a0 q : ℕ, (∀ k, a0 * q^k ≥ 100 ∧ a0 * q^k < 1000) →
  (∃ r s : ℕ, r > s ∧ q = r / s) →
  (∀ n, ∃ r s : ℕ, (r^n < 1000) ∧ ((r / s)^n < 10)) →
  n ≤ 5 :=
sorry

end max_geometric_progression_terms_l172_172514


namespace polynomial_transformation_l172_172965

-- Given the conditions of the polynomial function g and the provided transformation
-- We aim to prove the equivalence in a mathematically formal way using Lean

theorem polynomial_transformation (g : ℝ → ℝ) (h : ∀ x : ℝ, g (x^2 + 2) = x^4 + 5 * x^2 + 1) :
  ∀ x : ℝ, g (x^2 - 2) = x^4 - 3 * x^2 - 3 :=
by
  intro x
  sorry

end polynomial_transformation_l172_172965


namespace pool_length_l172_172806

def volume_of_pool (width length depth : ℕ) : ℕ :=
  width * length * depth

def volume_of_water (volume : ℕ) (capacity : ℝ) : ℝ :=
  volume * capacity

theorem pool_length (L : ℕ) (width depth : ℕ) (capacity : ℝ) (drain_rate drain_time : ℕ) (h_capacity : capacity = 0.80)
  (h_width : width = 50) (h_depth : depth = 10)
  (h_drain_rate : drain_rate = 60) (h_drain_time : drain_time = 1000)
  (h_drain_volume : volume_of_water (volume_of_pool width L depth) capacity = drain_rate * drain_time) :
  L = 150 :=
by
  sorry

end pool_length_l172_172806


namespace pythagorean_triple_correct_l172_172592

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_correct : 
  is_pythagorean_triple 9 12 15 ∧ ¬ is_pythagorean_triple 3 4 6 ∧ ¬ is_pythagorean_triple 1 2 3 ∧ ¬ is_pythagorean_triple 6 12 13 :=
by
  sorry

end pythagorean_triple_correct_l172_172592


namespace remaining_paint_needed_l172_172859

-- Define the conditions
def total_paint_needed : ℕ := 70
def paint_bought : ℕ := 23
def paint_already_have : ℕ := 36

-- Lean theorem statement
theorem remaining_paint_needed : (total_paint_needed - (paint_already_have + paint_bought)) = 11 := by
  sorry

end remaining_paint_needed_l172_172859


namespace problem_statement_l172_172744

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - Real.log x

theorem problem_statement (a b : ℝ) (h1 : 0 < a) (h2 : ∀ x, 0 < x → f a b x ≥ f a b 1) : 
  Real.log a < -2 * b :=
by
  sorry

end problem_statement_l172_172744


namespace eqD_is_linear_l172_172523

-- Definitions for the given equations
def eqA (x y : ℝ) : Prop := 3 * x - 2 * y = 1
def eqB (x : ℝ) : Prop := 1 + (1 / x) = x
def eqC (x : ℝ) : Prop := x^2 = 9
def eqD (x : ℝ) : Prop := 2 * x - 3 = 5

-- Definition of a linear equation in one variable
def isLinear (eq : ℝ → Prop) : Prop :=
  ∃ a b c : ℝ, (∀ x : ℝ, eq x ↔ a * x + b = c)

-- Theorem stating that eqD is a linear equation
theorem eqD_is_linear : isLinear eqD :=
  sorry

end eqD_is_linear_l172_172523


namespace cos_two_pi_over_three_l172_172729

theorem cos_two_pi_over_three : Real.cos (2 * Real.pi / 3) = -1 / 2 := 
by
  sorry

end cos_two_pi_over_three_l172_172729


namespace age_multiple_l172_172837

variables {R J K : ℕ}

theorem age_multiple (h1 : R = J + 6) (h2 : R = K + 3) (h3 : (R + 4) * (K + 4) = 108) :
  ∃ M : ℕ, R + 4 = M * (J + 4) ∧ M = 2 :=
sorry

end age_multiple_l172_172837


namespace highest_value_of_a_divisible_by_8_l172_172308

theorem highest_value_of_a_divisible_by_8 :
  ∃ (a : ℕ), (0 ≤ a ∧ a ≤ 9) ∧ (8 ∣ (100 * a + 16)) ∧ 
  (∀ (b : ℕ), (0 ≤ b ∧ b ≤ 9) → 8 ∣ (100 * b + 16) → b ≤ a) :=
sorry

end highest_value_of_a_divisible_by_8_l172_172308


namespace proof_part1_proof_part2_l172_172991

noncomputable def f (x a : ℝ) : ℝ := x^3 - a * x^2 + 3 * x

def condition1 (a : ℝ) : Prop := ∀ x : ℝ, x ≥ 1 → 3 * x^2 - 2 * a * x + 3 ≥ 0

def condition2 (a : ℝ) : Prop := 3 * 3^2 - 2 * a * 3 + 3 = 0

theorem proof_part1 (a : ℝ) : condition1 a → a ≤ 3 := 
sorry

theorem proof_part2 (a : ℝ) (ha : a = 5) : 
  f 1 a = -1 ∧ f 3 a = -9 ∧ f 5 a = 15 :=
sorry

end proof_part1_proof_part2_l172_172991


namespace calc_expr_eq_l172_172098

-- Define the polynomial and expression
def expr (x : ℝ) : ℝ := x * (x * (x * (3 - 2 * x) - 4) + 8) + 3 * x^2

theorem calc_expr_eq (x : ℝ) : expr x = -2 * x^4 + 3 * x^3 - x^2 + 8 * x := 
by
  sorry

end calc_expr_eq_l172_172098


namespace round_trip_by_car_time_l172_172788

variable (time_walk time_car : ℕ)
variable (h1 : time_walk + time_car = 20)
variable (h2 : 2 * time_walk = 32)

theorem round_trip_by_car_time : 2 * time_car = 8 :=
by
  sorry

end round_trip_by_car_time_l172_172788


namespace find_a_l172_172823

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - Real.sqrt 2

theorem find_a (a : ℝ) (h : f a (f a (Real.sqrt 2)) = -Real.sqrt 2) : 
  a = Real.sqrt 2 / 2 :=
by
  sorry

end find_a_l172_172823


namespace perfect_square_conditions_l172_172460

theorem perfect_square_conditions (x y k : ℝ) :
  (∃ a : ℝ, x^2 + k * x * y + 81 * y^2 = a^2) ↔ (k = 18 ∨ k = -18) :=
sorry

end perfect_square_conditions_l172_172460


namespace badges_total_l172_172932

theorem badges_total :
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  hermione_badges + luna_badges + celestia_badges = 83 :=
by
  let hermione_badges := 14
  let luna_badges := 17
  let celestia_badges := 52
  sorry

end badges_total_l172_172932


namespace fraction_of_paper_per_book_l172_172628

theorem fraction_of_paper_per_book (total_fraction_used : ℚ) (num_books : ℕ) (h1 : total_fraction_used = 5 / 8) (h2 : num_books = 5) : 
  (total_fraction_used / num_books) = 1 / 8 :=
by
  sorry

end fraction_of_paper_per_book_l172_172628


namespace find_cookbooks_stashed_in_kitchen_l172_172804

-- Definitions of the conditions
def total_books := 99
def books_in_boxes := 3 * 15
def books_in_room := 21
def books_on_table := 4
def books_picked_up := 12
def current_books := 23

-- Main statement
theorem find_cookbooks_stashed_in_kitchen :
  let books_donated := books_in_boxes + books_in_room + books_on_table
  let books_left_initial := total_books - books_donated
  let books_left_before_pickup := current_books - books_picked_up
  books_left_initial - books_left_before_pickup = 18 := by
  sorry

end find_cookbooks_stashed_in_kitchen_l172_172804


namespace problem_f_17_l172_172485

/-- Assume that f(1) = 0 and f(m + n) = f(m) + f(n) + 4 * (9 * m * n - 1) for all natural numbers m and n.
    Prove that f(17) = 4832.
-/
theorem problem_f_17 (f : ℕ → ℤ) 
  (h1 : f 1 = 0) 
  (h_func : ∀ m n : ℕ, f (m + n) = f m + f n + 4 * (9 * m * n - 1)) 
  : f 17 = 4832 := 
sorry

end problem_f_17_l172_172485


namespace harry_spends_1920_annually_l172_172236

def geckoCount : Nat := 3
def iguanaCount : Nat := 2
def snakeCount : Nat := 4

def geckoFeedTimesPerMonth : Nat := 2
def iguanaFeedTimesPerMonth : Nat := 3
def snakeFeedTimesPerMonth : Nat := 1 / 2

def geckoFeedCostPerMeal : Nat := 8
def iguanaFeedCostPerMeal : Nat := 12
def snakeFeedCostPerMeal : Nat := 20

def annualCostHarrySpends (geckoCount guCount scCount : Nat) (geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth : Nat) (geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal : Nat) : Nat :=
  let geckoAnnualCost := geckoCount * (geckoFeedTimesPerMonth * 12 * geckoFeedCostPerMeal)
  let iguanaAnnualCost := iguanaCount * (iguanaFeedTimesPerMonth * 12 * iguanaFeedCostPerMeal)
  let snakeAnnualCost := snakeCount * ((12 / (2 : Nat)) * snakeFeedCostPerMeal)
  geckoAnnualCost + iguanaAnnualCost + snakeAnnualCost

theorem harry_spends_1920_annually : annualCostHarrySpends geckoCount iguanaCount snakeCount geckoFeedTimesPerMonth iguanaFeedTimesPerMonth snakeFeedTimesPerMonth geckoFeedCostPerMeal iguanaFeedCostPerMeal snakeFeedCostPerMeal = 1920 := 
  sorry

end harry_spends_1920_annually_l172_172236


namespace inequality_proof_l172_172067

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_ineq : a + b + c > 1 / a + 1 / b + 1 / c) : 
  a + b + c ≥ 3 / (a * b * c) :=
sorry

end inequality_proof_l172_172067


namespace wallet_amount_l172_172149

-- Definitions of given conditions
def num_toys := 28
def cost_per_toy := 10
def num_teddy_bears := 20
def cost_per_teddy_bear := 15

-- Calculation of total costs
def total_cost_of_toys := num_toys * cost_per_toy
def total_cost_of_teddy_bears := num_teddy_bears * cost_per_teddy_bear

-- Total amount of money in Louise's wallet
def total_cost := total_cost_of_toys + total_cost_of_teddy_bears

-- Proof that the total cost is $580
theorem wallet_amount : total_cost = 580 :=
by
  -- Skipping the proof for now
  sorry

end wallet_amount_l172_172149


namespace fraction_addition_simplest_form_l172_172716

theorem fraction_addition_simplest_form :
  (7 / 8) + (3 / 5) = 59 / 40 :=
by sorry

end fraction_addition_simplest_form_l172_172716


namespace solve_congruence_l172_172188

theorem solve_congruence :
  ∃ a m : ℕ, m ≥ 2 ∧ a < m ∧ a + m = 27 ∧ (10 * x + 3 ≡ 7 [MOD 15]) → x ≡ 12 [MOD 15] := 
by
  sorry

end solve_congruence_l172_172188


namespace incorrect_desc_is_C_l172_172605
noncomputable def incorrect_geometric_solid_desc : Prop :=
  ¬ (∀ (plane_parallel: Prop), 
      plane_parallel ∧ 
      (∀ (frustum: Prop), frustum ↔ 
        (∃ (base section_cut cone : Prop), 
          cone ∧ 
          (section_cut = plane_parallel) ∧ 
          (frustum = (base ∧ section_cut)))))

theorem incorrect_desc_is_C (plane_parallel frustum base section_cut cone : Prop) :
  incorrect_geometric_solid_desc := 
by
  sorry

end incorrect_desc_is_C_l172_172605


namespace combined_avg_score_l172_172319

-- Define the average scores
def avg_score_u : ℕ := 65
def avg_score_b : ℕ := 80
def avg_score_c : ℕ := 77

-- Define the ratio of the number of students
def ratio_u : ℕ := 4
def ratio_b : ℕ := 6
def ratio_c : ℕ := 5

-- Prove the combined average score
theorem combined_avg_score : (ratio_u * avg_score_u + ratio_b * avg_score_b + ratio_c * avg_score_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end combined_avg_score_l172_172319


namespace other_x_intercept_l172_172587

def foci1 := (0, -3)
def foci2 := (4, 0)
def x_intercept1 := (0, 0)

theorem other_x_intercept :
  (∃ x : ℝ, (|x - 4| + |-3| * x = 7)) → x = 11 / 4 := by
  sorry

end other_x_intercept_l172_172587


namespace min_width_for_fence_area_least_200_l172_172600

theorem min_width_for_fence_area_least_200 (w : ℝ) (h : w * (w + 20) ≥ 200) : w ≥ 10 :=
sorry

end min_width_for_fence_area_least_200_l172_172600


namespace student_age_is_24_l172_172334

-- Defining the conditions
variables (S M : ℕ)
axiom h1 : M = S + 26
axiom h2 : M + 2 = 2 * (S + 2)

-- The proof statement
theorem student_age_is_24 : S = 24 :=
by
  sorry

end student_age_is_24_l172_172334


namespace willie_final_stickers_l172_172627

-- Conditions
def willie_start_stickers : ℝ := 36.0
def emily_gives_willie : ℝ := 7.0

-- Theorem
theorem willie_final_stickers : willie_start_stickers + emily_gives_willie = 43.0 :=
by
  sorry

end willie_final_stickers_l172_172627


namespace transformed_roots_l172_172259

theorem transformed_roots 
  (a b c : ℝ)
  (h₁ : a ≠ 0)
  (h₂ : a * (-1)^2 + b * (-1) + c = 0)
  (h₃ : a * 2^2 + b * 2 + c = 0) :
  (a * 0^2 + b * 0 + c = 0) ∧ (a * 3^2 + b * 3 + c = 0) :=
by 
  sorry

end transformed_roots_l172_172259


namespace compute_expression_l172_172039

theorem compute_expression (x : ℤ) (h : x = 6) :
  ((x^9 - 24 * x^6 + 144 * x^3 - 512) / (x^3 - 8) = 43264) :=
by
  sorry

end compute_expression_l172_172039


namespace carter_stretching_legs_frequency_l172_172135

-- Given conditions
def tripDuration : ℤ := 14 * 60 -- in minutes
def foodStops : ℤ := 2
def gasStops : ℤ := 3
def pitStopDuration : ℤ := 20 -- in minutes
def totalTripDuration : ℤ := 18 * 60 -- in minutes

-- Prove that Carter stops to stretch his legs every 2 hours
theorem carter_stretching_legs_frequency :
  ∃ (stretchingStops : ℤ), (totalTripDuration - tripDuration = (foodStops + gasStops + stretchingStops) * pitStopDuration) ∧
    (stretchingStops * pitStopDuration = totalTripDuration - (tripDuration + (foodStops + gasStops) * pitStopDuration)) ∧
    (14 / stretchingStops = 2) :=
by sorry

end carter_stretching_legs_frequency_l172_172135


namespace combined_stickers_leftover_l172_172770

theorem combined_stickers_leftover (r p g : ℕ) (h_r : r % 5 = 1) (h_p : p % 5 = 4) (h_g : g % 5 = 3) :
  (r + p + g) % 5 = 3 :=
by
  sorry

end combined_stickers_leftover_l172_172770


namespace xy_sum_greater_two_l172_172919

theorem xy_sum_greater_two (x y : ℝ) (h1 : x^3 > y^2) (h2 : y^3 > x^2) : x + y > 2 := 
by 
  sorry

end xy_sum_greater_two_l172_172919


namespace praveen_hari_profit_ratio_l172_172446

theorem praveen_hari_profit_ratio
  (praveen_capital : ℕ := 3360)
  (hari_capital : ℕ := 8640)
  (time_praveen_invested : ℕ := 12)
  (time_hari_invested : ℕ := 7)
  (praveen_shares_full_time : ℕ := praveen_capital * time_praveen_invested)
  (hari_shares_full_time : ℕ := hari_capital * time_hari_invested)
  (gcd_common : ℕ := Nat.gcd praveen_shares_full_time hari_shares_full_time) :
  (praveen_shares_full_time / gcd_common) * 2 = 2 ∧ (hari_shares_full_time / gcd_common) * 2 = 3 := by
    sorry

end praveen_hari_profit_ratio_l172_172446


namespace find_non_equivalent_fraction_l172_172929

-- Define the fractions mentioned in the problem
def sevenSixths := 7 / 6
def optionA := 14 / 12
def optionB := 1 + 1 / 6
def optionC := 1 + 5 / 30
def optionD := 1 + 2 / 6
def optionE := 1 + 14 / 42

-- The main problem statement
theorem find_non_equivalent_fraction :
  optionD ≠ sevenSixths := by
  -- We put a 'sorry' here because we are not required to provide the proof
  sorry

end find_non_equivalent_fraction_l172_172929


namespace tan_beta_l172_172040

open Real

variable (α β : ℝ)

theorem tan_beta (h₁ : tan α = 1/3) (h₂ : tan (α + β) = 1/2) : tan β = 1/7 :=
by sorry

end tan_beta_l172_172040


namespace roots_of_quadratic_eq_l172_172275

noncomputable def r : ℂ := sorry
noncomputable def s : ℂ := sorry

def roots_eq (h : 3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : Prop :=
  (1 / r^3) + (1 / s^3) = 1

theorem roots_of_quadratic_eq (h:3 * r^2 + 4 * r + 2 = 0 ∧ 3 * s^2 + 4 * s + 2 = 0) : roots_eq h :=
sorry

end roots_of_quadratic_eq_l172_172275


namespace angleC_equals_40_of_angleA_40_l172_172840

-- Define an arbitrary quadrilateral type and its angle A and angle C
structure Quadrilateral :=
  (angleA : ℝ)  -- angleA is in degrees
  (angleC : ℝ)  -- angleC is in degrees

-- Given condition in the problem
def quadrilateral_with_A_40 : Quadrilateral :=
  { angleA := 40, angleC := 0 } -- Initialize angleC as a placeholder

-- Theorem stating the problem's claim
theorem angleC_equals_40_of_angleA_40 :
  quadrilateral_with_A_40.angleA = 40 → quadrilateral_with_A_40.angleC = 40 :=
by
  sorry  -- Proof is omitted for brevity

end angleC_equals_40_of_angleA_40_l172_172840


namespace range_of_m_l172_172721

-- Define the variables and main theorem
theorem range_of_m (m : ℝ) (a b c : ℝ) 
  (h₀ : a = 3) (h₁ : b = (1 - 2 * m)) (h₂ : c = 8)
  : -5 < m ∧ m < -2 :=
by
  -- Given that a, b, and c are sides of a triangle, we use the triangle inequality theorem
  -- This code will remain as a placeholder of that proof
  sorry

end range_of_m_l172_172721


namespace price_of_cheese_cookie_pack_l172_172516

theorem price_of_cheese_cookie_pack
    (cartons : ℕ) (boxes_per_carton : ℕ) (packs_per_box : ℕ) (total_cost : ℕ)
    (h_cartons : cartons = 12)
    (h_boxes_per_carton : boxes_per_carton = 12)
    (h_packs_per_box : packs_per_box = 10)
    (h_total_cost : total_cost = 1440) :
  (total_cost / (cartons * boxes_per_carton * packs_per_box) = 1) :=
by
  -- conditions are explicitly given in the theorem statement
  sorry

end price_of_cheese_cookie_pack_l172_172516


namespace exists_nat_square_starting_with_digits_l172_172576

theorem exists_nat_square_starting_with_digits (S : ℕ) : 
  ∃ (N k : ℕ), S * 10^k ≤ N^2 ∧ N^2 < (S + 1) * 10^k := 
by {
  sorry
}

end exists_nat_square_starting_with_digits_l172_172576


namespace john_income_increase_l172_172059

noncomputable def net_percentage_increase (initial_income : ℝ) (final_income_before_bonus : ℝ) (monthly_bonus : ℝ) (tax_deduction_rate : ℝ) : ℝ :=
  let weekly_bonus := monthly_bonus / 4
  let final_income_before_taxes := final_income_before_bonus + weekly_bonus
  let tax_deduction := tax_deduction_rate * final_income_before_taxes
  let net_final_income := final_income_before_taxes - tax_deduction
  ((net_final_income - initial_income) / initial_income) * 100

theorem john_income_increase :
  net_percentage_increase 40 60 100 0.10 = 91.25 := by
  sorry

end john_income_increase_l172_172059


namespace max_value_correct_l172_172531

noncomputable def max_value (x y : ℝ) (h : x + y = 5) : ℝ :=
  x^5 * y + x^4 * y + x^3 * y + x^2 * y + x * y + x * y^2 + x * y^3 + x * y^4 + x * y^5

theorem max_value_correct (x y : ℝ) (h : x + y = 5) : max_value x y h ≤ 22884 :=
  sorry

end max_value_correct_l172_172531


namespace graph_shift_l172_172478

theorem graph_shift (f : ℝ → ℝ) (h : f 0 = 2) : f (-1 + 1) = 2 :=
by
  have h1 : f 0 = 2 := h
  sorry

end graph_shift_l172_172478


namespace cubic_sum_l172_172796

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 :=
by
  sorry

end cubic_sum_l172_172796


namespace det_matrixB_eq_neg_one_l172_172381

variable (x y : ℝ)

def matrixB : Matrix (Fin 2) (Fin 2) ℝ := ![
  ![x, 3],
  ![-4, y]
]

theorem det_matrixB_eq_neg_one 
  (h : matrixB x y - (matrixB x y)⁻¹ = 2 • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
  Matrix.det (matrixB x y) = -1 := sorry

end det_matrixB_eq_neg_one_l172_172381


namespace total_tiles_l172_172767

theorem total_tiles (s : ℕ) (H1 : 2 * s - 1 = 57) : s^2 = 841 := by
  sorry

end total_tiles_l172_172767


namespace sum_of_cube_angles_l172_172317

theorem sum_of_cube_angles (W X Y Z : Point) (cube : Cube)
  (angle_WXY angle_XYZ angle_YZW angle_ZWX : ℝ)
  (h₁ : angle_WXY = 90)
  (h₂ : angle_XYZ = 90)
  (h₃ : angle_YZW = 90)
  (h₄ : angle_ZWX = 60) :
  angle_WXY + angle_XYZ + angle_YZW + angle_ZWX = 330 := by
  sorry

end sum_of_cube_angles_l172_172317


namespace age_difference_l172_172084

theorem age_difference (a1 a2 a3 a4 x y : ℕ) 
  (h1 : (a1 + a2 + a3 + a4 + x) / 5 = 28)
  (h2 : ((a1 + 1) + (a2 + 1) + (a3 + 1) + (a4 + 1) + y) / 5 = 30) : 
  y - (x + 1) = 5 := 
by
  sorry

end age_difference_l172_172084


namespace olivia_worked_hours_on_wednesday_l172_172167

-- Define the conditions
def hourly_rate := 9
def hours_monday := 4
def hours_friday := 6
def total_earnings := 117
def earnings_monday := hours_monday * hourly_rate
def earnings_friday := hours_friday * hourly_rate
def earnings_wednesday := total_earnings - (earnings_monday + earnings_friday)

-- Define the number of hours worked on Wednesday
def hours_wednesday := earnings_wednesday / hourly_rate

-- The theorem to prove
theorem olivia_worked_hours_on_wednesday : hours_wednesday = 3 :=
by
  -- Skip the proof
  sorry

end olivia_worked_hours_on_wednesday_l172_172167


namespace geometric_sequence_sum_2018_l172_172914

noncomputable def geometric_sum (n : ℕ) (a1 q : ℝ) : ℝ :=
  if q = 1 then n * a1 else a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum_2018 :
  ∀ (a : ℕ → ℝ) (S : ℕ → ℝ),
    (∀ n, S n = geometric_sum n (a 1) 2) →
    a 1 = 1 / 2 →
    (a 1 * 2^2)^2 = 8 * a 1 * 2^3 - 16 →
    S 2018 = 2^2017 - 1 / 2 :=
by sorry

end geometric_sequence_sum_2018_l172_172914


namespace equal_real_roots_a_value_l172_172405

theorem equal_real_roots_a_value (a : ℝ) :
  a ≠ 0 →
  let b := -4
  let c := 3
  b * b - 4 * a * c = 0 →
  a = 4 / 3 :=
by
  intros h_nonzero h_discriminant
  sorry

end equal_real_roots_a_value_l172_172405


namespace unique_solution_exists_l172_172143

theorem unique_solution_exists :
  ∃ (a b c d e : ℕ),
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧
  a + b = 1/7 * (c + d + e) ∧
  a + c = 1/5 * (b + d + e) ∧
  (a, b, c, d, e) = (1, 2, 3, 9, 9) :=
by {
  sorry
}

end unique_solution_exists_l172_172143


namespace difference_of_squares_evaluation_l172_172967

theorem difference_of_squares_evaluation :
  49^2 - 16^2 = 2145 :=
by sorry

end difference_of_squares_evaluation_l172_172967


namespace part1_part2_l172_172931

def setA := {x : ℝ | -3 < x ∧ x < 4}
def setB (a : ℝ) := {x : ℝ | x^2 - 4 * a * x + 3 * a^2 = 0}

theorem part1 (a : ℝ) : (setA ∩ setB a = ∅) ↔ (a ≤ -3 ∨ a ≥ 4) :=
sorry

theorem part2 (a : ℝ) : (setA ∪ setB a = setA) ↔ (-1 < a ∧ a < 4/3) :=
sorry

end part1_part2_l172_172931


namespace reduced_price_l172_172119

variable (P R : ℝ)
variable (price_reduction : R = 0.75 * P)
variable (buy_more_oil : 700 / R = 700 / P + 5)

theorem reduced_price (non_zero_P : P ≠ 0) (non_zero_R : R ≠ 0) : R = 35 := 
by
  sorry

end reduced_price_l172_172119


namespace number_of_coins_l172_172596

-- Define the conditions
def equal_number_of_coins (x : ℝ) :=
  ∃ n : ℝ, n = x

-- Define the total value condition
def total_value (x : ℝ) :=
  x + 0.50 * x + 0.25 * x = 70

-- The theorem to be proved
theorem number_of_coins (x : ℝ) (h1 : equal_number_of_coins x) (h2 : total_value x) : x = 40 :=
by sorry

end number_of_coins_l172_172596


namespace ratio_P_to_A_l172_172537

variable (M P A : ℕ) -- Define variables for Matthew, Patrick, and Alvin's egg rolls

theorem ratio_P_to_A (hM : M = 6) (hM_to_P : M = 3 * P) (hA : A = 4) : P / A = 1 / 2 := by
  sorry

end ratio_P_to_A_l172_172537


namespace square_root_of_9_eq_pm_3_l172_172423

theorem square_root_of_9_eq_pm_3 (x : ℝ) : x^2 = 9 → x = 3 ∨ x = -3 :=
sorry

end square_root_of_9_eq_pm_3_l172_172423


namespace radius_of_circle_B_l172_172961

theorem radius_of_circle_B (r_A r_D : ℝ) (r_B : ℝ) (hA : r_A = 2) (hD : r_D = 4) 
  (congruent_BC : r_B = r_B) (tangent_condition : true) -- placeholder conditions
  (center_pass : true) -- placeholder conditions
  : r_B = (4 / 3) * (Real.sqrt 7 - 1) :=
sorry

end radius_of_circle_B_l172_172961


namespace determine_A_plus_B_l172_172883

theorem determine_A_plus_B :
  ∃ (A B : ℚ), ((∀ x : ℚ, x ≠ 4 ∧ x ≠ 5 → 
  (Bx - 23) / (x^2 - 9 * x + 20) = A / (x - 4) + 5 / (x - 5)) ∧
  (A + B = 11 / 9)) :=
sorry

end determine_A_plus_B_l172_172883


namespace cost_of_goods_l172_172911

-- Define variables and conditions
variables (x y z : ℝ)

-- Assume the given conditions
axiom h1 : x + 2 * y + 3 * z = 136
axiom h2 : 3 * x + 2 * y + z = 240

-- Statement to prove
theorem cost_of_goods : x + y + z = 94 := 
sorry

end cost_of_goods_l172_172911


namespace regular_pentagon_cannot_tessellate_l172_172470

-- Definitions of polygons
def is_regular_triangle (angle : ℝ) : Prop := angle = 60
def is_square (angle : ℝ) : Prop := angle = 90
def is_regular_pentagon (angle : ℝ) : Prop := angle = 108
def is_hexagon (angle : ℝ) : Prop := angle = 120

-- Tessellation condition
def divides_evenly (a b : ℝ) : Prop := ∃ k : ℕ, b = k * a

-- The main statement
theorem regular_pentagon_cannot_tessellate :
  ¬ divides_evenly 108 360 :=
sorry

end regular_pentagon_cannot_tessellate_l172_172470


namespace cost_of_notebook_l172_172372

theorem cost_of_notebook (s n c : ℕ) 
    (h1 : s > 18) 
    (h2 : n ≥ 2) 
    (h3 : c > n) 
    (h4 : s * c * n = 2376) : 
    c = 11 := 
  sorry

end cost_of_notebook_l172_172372


namespace find_second_number_in_second_set_l172_172549

theorem find_second_number_in_second_set :
    (14 + 32 + 53) / 3 = 3 + (21 + x + 22) / 3 → x = 47 :=
by intro h
   sorry

end find_second_number_in_second_set_l172_172549


namespace find_other_number_l172_172331

theorem find_other_number 
  {A B : ℕ} 
  (h_A : A = 24)
  (h_hcf : Nat.gcd A B = 14)
  (h_lcm : Nat.lcm A B = 312) :
  B = 182 :=
by
  -- Proof skipped
  sorry

end find_other_number_l172_172331


namespace binary_operation_correct_l172_172954

-- Define the binary numbers involved
def bin1 := 0b110110 -- 110110_2
def bin2 := 0b101010 -- 101010_2
def bin3 := 0b100    -- 100_2

-- Define the operation in binary
def result := 0b111001101100 -- 111001101100_2

-- Lean statement to verify the operation result
theorem binary_operation_correct : (bin1 * bin2) / bin3 = result :=
by sorry

end binary_operation_correct_l172_172954


namespace total_games_attended_l172_172902

theorem total_games_attended 
  (games_this_month : ℕ)
  (games_last_month : ℕ)
  (games_next_month : ℕ)
  (total_games : ℕ) 
  (h : games_this_month = 11)
  (h2 : games_last_month = 17)
  (h3 : games_next_month = 16) 
  (htotal : total_games = 44) :
  games_this_month + games_last_month + games_next_month = total_games :=
by sorry

end total_games_attended_l172_172902


namespace calculation_equals_106_25_l172_172602

noncomputable def calculation : ℝ := 2.5 * 8.5 * (5.2 - 0.2)

theorem calculation_equals_106_25 : calculation = 106.25 := 
by
  sorry

end calculation_equals_106_25_l172_172602


namespace find_a4_l172_172339

-- Given expression of x^5
def polynomial_expansion (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) : Prop :=
  x^5 = a_0 + a_1 * (x+1) + a_2 * (x+1)^2 + a_3 * (x+1)^3 + a_4 * (x+1)^4 + a_5 * (x+1)^5

theorem find_a4 (x a_0 a_1 a_2 a_3 a_4 a_5 : ℝ) (h : polynomial_expansion x a_0 a_1 a_2 a_3 a_4 a_5) : a_4 = -5 :=
  sorry

end find_a4_l172_172339


namespace find_natural_n_l172_172834

theorem find_natural_n (n : ℕ) :
  (992768 ≤ n ∧ n ≤ 993791) ↔ 
  (∀ k : ℕ, k > 0 → k^2 + (n / k^2) = 1991) := sorry

end find_natural_n_l172_172834


namespace num_of_valid_numbers_l172_172230

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 10
  let b := n % 10
  a >= 1 ∧ a <= 9 ∧ b >= 0 ∧ b <= 9 ∧ (9 * a) % 10 = 4

theorem num_of_valid_numbers : ∃ n, n = 10 :=
by {
  sorry
}

end num_of_valid_numbers_l172_172230


namespace find_value_of_m_l172_172164

theorem find_value_of_m (m : ℤ) (x : ℤ) (h : (x - 3 ≠ 0) ∧ (x = 3)) : 
  ((x - 1) / (x - 3) = m / (x - 3)) → m = 2 :=
by
  sorry

end find_value_of_m_l172_172164


namespace length_of_field_l172_172515

-- Define the known conditions
def width := 50
def total_distance_run := 1800
def num_laps := 6

-- Define the problem statement
theorem length_of_field :
  ∃ L : ℕ, 6 * (2 * (L + width)) = total_distance_run ∧ L = 100 :=
by
  sorry

end length_of_field_l172_172515


namespace count_perfect_squares_diff_l172_172722

theorem count_perfect_squares_diff (a b : ℕ) : 
  ∃ (count : ℕ), 
  count = 25 ∧ 
  (∀ (a : ℕ), (∃ (b : ℕ), a^2 = 2 * b + 1 ∧ a^2 < 2500) ↔ (∃ (k : ℕ), 1 ≤ k ∧ k ≤ 25 ∧ 2 * k - 1 = a)) :=
by
  sorry

end count_perfect_squares_diff_l172_172722


namespace numWaysToPaintDoors_l172_172805

-- Define the number of doors and choices per door
def numDoors : ℕ := 3
def numChoicesPerDoor : ℕ := 2

-- Theorem statement that we want to prove
theorem numWaysToPaintDoors : numChoicesPerDoor ^ numDoors = 8 := by
  sorry

end numWaysToPaintDoors_l172_172805


namespace youngest_child_age_l172_172442

theorem youngest_child_age
  (ten_years_ago_avg_age : Nat) (family_initial_size : Nat) (present_avg_age : Nat)
  (age_difference : Nat) (age_ten_years_ago_total : Nat)
  (age_increase : Nat) (current_age_total : Nat)
  (current_family_size : Nat) (total_age_increment : Nat) :
  ten_years_ago_avg_age = 24 →
  family_initial_size = 4 →
  present_avg_age = 24 →
  age_difference = 2 →
  age_ten_years_ago_total = family_initial_size * ten_years_ago_avg_age →
  age_increase = family_initial_size * 10 →
  current_age_total = age_ten_years_ago_total + age_increase →
  current_family_size = family_initial_size + 2 →
  total_age_increment = current_family_size * present_avg_age →
  total_age_increment - current_age_total = 8 →
  ∃ (Y : Nat), Y + Y + age_difference = 8 ∧ Y = 3 :=
by
  intros
  sorry

end youngest_child_age_l172_172442


namespace single_colony_habitat_limit_reach_time_l172_172201

noncomputable def doubling_time (n : ℕ) : ℕ := 2^n

theorem single_colony_habitat_limit_reach_time :
  ∀ (S : ℕ), ∀ (n : ℕ), doubling_time (n + 1) = S → doubling_time (2 * (n - 1)) = S → n + 1 = 16 :=
by
  intros S n H1 H2
  sorry

end single_colony_habitat_limit_reach_time_l172_172201


namespace sibling_discount_is_correct_l172_172966

-- Defining the given conditions
def tuition_per_person : ℕ := 45
def total_cost_with_discount : ℕ := 75

-- Defining the calculation of sibling discount
def sibling_discount : ℕ :=
  let original_cost := 2 * tuition_per_person
  let discount := original_cost - total_cost_with_discount
  discount

-- Statement to prove
theorem sibling_discount_is_correct : sibling_discount = 15 :=
by
  unfold sibling_discount
  simp
  sorry

end sibling_discount_is_correct_l172_172966


namespace roots_quartic_ab_plus_a_plus_b_l172_172836

theorem roots_quartic_ab_plus_a_plus_b (a b : ℝ) 
  (h1 : Polynomial.eval a (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0)
  (h2 : Polynomial.eval b (Polynomial.C 1 + Polynomial.C (-4) * Polynomial.X + Polynomial.C (-6) * (Polynomial.X ^ 2) + Polynomial.C 1 * (Polynomial.X ^ 4)) = 0) :
  a * b + a + b = -1 := 
sorry

end roots_quartic_ab_plus_a_plus_b_l172_172836


namespace range_of_a_for_root_l172_172253

noncomputable def has_root_in_interval (a : ℝ) : Prop :=
  ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ (a * x^2 + 2 * x - 1) = 0

theorem range_of_a_for_root :
  { a : ℝ | has_root_in_interval a } = { a : ℝ | -1 ≤ a } :=
by 
  sorry

end range_of_a_for_root_l172_172253


namespace line_symmetric_y_axis_eqn_l172_172486

theorem line_symmetric_y_axis_eqn (x y : ℝ) : 
  (∀ x y : ℝ, x - y + 1 = 0 → x + y - 1 = 0) := 
sorry

end line_symmetric_y_axis_eqn_l172_172486


namespace parabola_vertex_point_sum_l172_172314

theorem parabola_vertex_point_sum (a b c : ℚ) 
  (h1 : ∃ (a b c : ℚ), ∀ x : ℚ, (y = a * x ^ 2 + b * x + c) = (y = - (1 / 3) * (x - 5) ^ 2 + 3)) 
  (h2 : ∀ x : ℚ, ((x = 2) ∧ (y = 0)) → (0 = a * 2 ^ 2 + b * 2 + c)) :
  a + b + c = -7 / 3 := 
sorry

end parabola_vertex_point_sum_l172_172314


namespace car_mileage_proof_l172_172418

noncomputable def car_average_mpg 
  (odometer_start: ℝ) (odometer_end: ℝ) 
  (fuel1: ℝ) (fuel2: ℝ) (odometer2: ℝ) 
  (fuel3: ℝ) (odometer3: ℝ) (final_fuel: ℝ) 
  (final_odometer: ℝ): ℝ :=
  (odometer_end - odometer_start) / 
  ((fuel1 + fuel2 + fuel3 + final_fuel): ℝ)

theorem car_mileage_proof:
  car_average_mpg 56200 57150 6 14 56600 10 56880 20 57150 = 19 :=
by
  sorry

end car_mileage_proof_l172_172418


namespace Linda_outfits_l172_172710

theorem Linda_outfits (skirts blouses shoes : ℕ) 
  (hskirts : skirts = 5) 
  (hblouses : blouses = 8) 
  (hshoes : shoes = 2) :
  skirts * blouses * shoes = 80 := by
  -- We provide the proof here
  sorry

end Linda_outfits_l172_172710


namespace number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l172_172359

-- Define the number of classes in each grade.
def num_classes_first_year : ℕ := 14
def num_classes_second_year : ℕ := 14
def num_classes_third_year : ℕ := 15

-- Prove the number of different ways to choose students from 1 class.
theorem number_of_ways_to_choose_one_class :
  (num_classes_first_year + num_classes_second_year + num_classes_third_year) = 43 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from one class in each grade.
theorem number_of_ways_to_choose_one_class_each_grade :
  (num_classes_first_year * num_classes_second_year * num_classes_third_year) = 2940 := 
by {
  -- Numerical calculation
  sorry
}

-- Prove the number of different ways to choose students from 2 classes from different grades.
theorem number_of_ways_to_choose_two_classes_different_grades :
  (num_classes_first_year * num_classes_second_year + num_classes_first_year * num_classes_third_year + num_classes_second_year * num_classes_third_year) = 616 := 
by {
  -- Numerical calculation
  sorry
}

end number_of_ways_to_choose_one_class_number_of_ways_to_choose_one_class_each_grade_number_of_ways_to_choose_two_classes_different_grades_l172_172359


namespace arith_seq_formula_geom_seq_sum_l172_172279

-- Definitions for condition 1: Arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  (a 4 = 7) ∧ (a 10 = 19)

-- Definitions for condition 2: Sum of the first n terms of {a_n}
def sum_arith_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Definitions for condition 3: Geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop :=
  (b 1 = 2) ∧ (∀ n, b (n + 1) = b n * 2)

-- Definitions for condition 4: Sum of the first n terms of {b_n}
def sum_geom_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n, T n = (b 1 * (1 - (2 ^ n))) / (1 - 2)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula (a : ℕ → ℤ) (S : ℕ → ℤ) :
  arithmetic_seq a ∧ sum_arith_seq S a → 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, S n = n ^ 2) :=
sorry

-- Proving the sum of the first n terms for geometric sequence
theorem geom_seq_sum (b : ℕ → ℤ) (T : ℕ → ℤ) (S : ℕ → ℤ) :
  geometric_seq b ∧ sum_geom_seq T b ∧ b 4 = S 4 → 
  (∀ n, T n = 2 ^ (n + 1) - 2) :=
sorry

end arith_seq_formula_geom_seq_sum_l172_172279


namespace hyperbola_equation_l172_172441

variable (a b : ℝ)
variable (c : ℝ) (h1 : c = 4)
variable (h2 : b / a = Real.sqrt 3)
variable (h3 : a ^ 2 + b ^ 2 = c ^ 2)

theorem hyperbola_equation : (a ^ 2 = 4) ∧ (b ^ 2 = 12) ↔ (∀ x y : ℝ, (x ^ 2 / a ^ 2) - (y ^ 2 / b ^ 2) = 1 → (x ^ 2 / 4) - (y ^ 2 / 12) = 1) := by
  sorry

end hyperbola_equation_l172_172441


namespace vacation_cost_division_l172_172189

theorem vacation_cost_division (total_cost : ℕ) (cost_per_person3 different_cost : ℤ) (n : ℕ)
  (h1 : total_cost = 375)
  (h2 : cost_per_person3 = total_cost / 3)
  (h3 : different_cost = cost_per_person3 - 50)
  (h4 : different_cost = total_cost / n) :
  n = 5 :=
  sorry

end vacation_cost_division_l172_172189


namespace largest_n_divisible_l172_172737

theorem largest_n_divisible : ∃ n : ℕ, (∀ k : ℕ, (k^3 + 150) % (k + 5) = 0 → k ≤ n) ∧ n = 20 := 
by
  sorry

end largest_n_divisible_l172_172737


namespace find_number_l172_172037

theorem find_number (x : ℚ) : (35 / 100) * x = (20 / 100) * 50 → x = 200 / 7 :=
by
  intros h
  sorry

end find_number_l172_172037


namespace angle_of_inclination_of_line_l172_172232

-- Definition of the line l
def line_eq (x : ℝ) : ℝ := x + 1

-- Statement of the theorem about the angle of inclination
theorem angle_of_inclination_of_line (x : ℝ) : 
  ∃ (θ : ℝ), θ = 45 ∧ line_eq x = x + 1 := 
sorry

end angle_of_inclination_of_line_l172_172232


namespace sqrt_one_half_eq_sqrt_two_over_two_l172_172905

theorem sqrt_one_half_eq_sqrt_two_over_two : Real.sqrt (1 / 2) = Real.sqrt 2 / 2 :=
by sorry

end sqrt_one_half_eq_sqrt_two_over_two_l172_172905


namespace expression_I_evaluation_expression_II_evaluation_l172_172831

theorem expression_I_evaluation :
  ( (3 / 2) ^ (-2: ℤ) - (49 / 81) ^ (0.5: ℝ) + (0.008: ℝ) ^ (-2 / 3: ℝ) * (2 / 25) ) = (5 / 3) := 
by
  sorry

theorem expression_II_evaluation :
  ( (Real.logb 2 2) ^ 2 + (Real.logb 10 20) * (Real.logb 10 5) ) = (17 / 9) := 
by
  sorry

end expression_I_evaluation_expression_II_evaluation_l172_172831


namespace nathan_final_temperature_l172_172872

theorem nathan_final_temperature : ∃ (final_temp : ℝ), final_temp = 77.4 :=
  let initial_temp : ℝ := 50
  let type_a_increase : ℝ := 2
  let type_b_increase : ℝ := 3.5
  let type_c_increase : ℝ := 4.8
  let type_d_increase : ℝ := 7.2
  let type_a_quantity : ℚ := 6
  let type_b_quantity : ℚ := 5
  let type_c_quantity : ℚ := 9
  let type_d_quantity : ℚ := 3
  let temp_after_a := initial_temp + 3 * type_a_increase
  let temp_after_b := temp_after_a + 2 * type_b_increase
  let temp_after_c := temp_after_b + 3 * type_c_increase
  let final_temp := temp_after_c
  ⟨final_temp, sorry⟩

end nathan_final_temperature_l172_172872


namespace aimee_poll_l172_172192

theorem aimee_poll (P : ℕ) (h1 : 35 ≤ 100) (h2 : 39 % (P/2) = 39) : P = 120 := 
by sorry

end aimee_poll_l172_172192


namespace MrSami_sold_20_shares_of_stock_x_l172_172163

theorem MrSami_sold_20_shares_of_stock_x
    (shares_v : ℕ := 68)
    (shares_w : ℕ := 112)
    (shares_x : ℕ := 56)
    (shares_y : ℕ := 94)
    (shares_z : ℕ := 45)
    (additional_shares_y : ℕ := 23)
    (increase_in_range : ℕ := 14)
    : (shares_x - (shares_y + additional_shares_y - ((shares_w - shares_z + increase_in_range) - shares_y - additional_shares_y)) = 20) :=
by
  sorry

end MrSami_sold_20_shares_of_stock_x_l172_172163


namespace both_players_same_score_probability_l172_172544

theorem both_players_same_score_probability :
  let p_A_score := 0.6
  let p_B_score := 0.8
  let p_A_miss := 1 - p_A_score
  let p_B_miss := 1 - p_B_score
  (p_A_score * p_B_score + p_A_miss * p_B_miss = 0.56) :=
by
  sorry

end both_players_same_score_probability_l172_172544


namespace bricks_needed_l172_172185

noncomputable def volume (length : ℝ) (width : ℝ) (height : ℝ) : ℝ := length * width * height

theorem bricks_needed
  (brick_length : ℝ)
  (brick_width : ℝ)
  (brick_height : ℝ)
  (wall_length : ℝ)
  (wall_height : ℝ)
  (wall_thickness : ℝ)
  (hl : brick_length = 40)
  (hw : brick_width = 11.25)
  (hh : brick_height = 6)
  (wl : wall_length = 800)
  (wh : wall_height = 600)
  (wt : wall_thickness = 22.5) :
  (volume wall_length wall_height wall_thickness / volume brick_length brick_width brick_height) = 4000 := by
  sorry

end bricks_needed_l172_172185


namespace max_value_l172_172133

theorem max_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 9 * a^2 + 4 * b^2 + c^2 = 91) :
  a + 2 * b + 3 * c ≤ 30.333 :=
by
  sorry

end max_value_l172_172133


namespace fifth_team_points_l172_172980

theorem fifth_team_points (points_A points_B points_C points_D points_E : ℕ) 
(hA : points_A = 1) 
(hB : points_B = 2) 
(hC : points_C = 5) 
(hD : points_D = 7) 
(h_sum : points_A + points_B + points_C + points_D + points_E = 20) : 
points_E = 5 := 
sorry

end fifth_team_points_l172_172980


namespace triangle_angle_contradiction_l172_172959

theorem triangle_angle_contradiction (α β γ : ℝ) (h : α + β + γ = 180) :
  (α > 60 ∧ β > 60 ∧ γ > 60) -> false :=
by
  sorry

end triangle_angle_contradiction_l172_172959


namespace union_sets_l172_172004

-- Define the sets A and B as conditions
def A : Set ℝ := {0, 1}  -- Since lg 1 = 0
def B : Set ℝ := {-1, 0}

-- Define that A union B equals {-1, 0, 1}
theorem union_sets : A ∪ B = {-1, 0, 1} := by
  sorry

end union_sets_l172_172004


namespace max_value_of_a_exists_max_value_of_a_l172_172226

theorem max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  a ≤ (Real.sqrt 6 / 3) :=
sorry

theorem exists_max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 1) : 
  ∃ a_max: ℝ, a_max = (Real.sqrt 6 / 3) ∧ (∀ a', (a' ≤ a_max)) :=
sorry

end max_value_of_a_exists_max_value_of_a_l172_172226


namespace complex_division_l172_172933

theorem complex_division (i : ℂ) (h : i ^ 2 = -1) : (3 - 4 * i) / i = -4 - 3 * i :=
by
  sorry

end complex_division_l172_172933


namespace time_between_shark_sightings_l172_172856

def earnings_per_photo : ℕ := 15
def fuel_cost_per_hour : ℕ := 50
def hunting_hours : ℕ := 5
def expected_profit : ℕ := 200

theorem time_between_shark_sightings :
  (hunting_hours * 60) / ((expected_profit + (fuel_cost_per_hour * hunting_hours)) / earnings_per_photo) = 10 :=
by 
  sorry

end time_between_shark_sightings_l172_172856


namespace find_natural_numbers_l172_172457

open Nat

theorem find_natural_numbers (n : ℕ) (h : ∃ m : ℤ, 2^n + 33 = m^2) : n = 4 ∨ n = 8 :=
sorry

end find_natural_numbers_l172_172457


namespace pure_imaginary_condition_l172_172857

variable (a : ℝ)

def isPureImaginary (z : ℂ) : Prop := z.re = 0

theorem pure_imaginary_condition :
  isPureImaginary (a - 17 / (4 - (i : ℂ))) → a = 4 := 
by
  sorry

end pure_imaginary_condition_l172_172857


namespace intersection_of_M_and_N_l172_172990

noncomputable def M : Set ℝ := {y | ∃ x : ℝ, y = x ^ 2 - 1}
noncomputable def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
noncomputable def intersection : Set ℝ := {z | -1 ≤ z ∧ z ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {z | -1 ≤ z ∧ z ≤ 3} := 
sorry

end intersection_of_M_and_N_l172_172990


namespace find_missing_number_l172_172203

theorem find_missing_number:
  ∃ x : ℕ, (306 / 34) * 15 + x = 405 := sorry

end find_missing_number_l172_172203


namespace glass_bottles_in_second_scenario_l172_172157

theorem glass_bottles_in_second_scenario
  (G P x : ℕ)
  (h1 : 3 * G = 600)
  (h2 : G = P + 150)
  (h3 : x * G + 5 * P = 1050) :
  x = 4 :=
by 
  -- Proof is omitted
  sorry

end glass_bottles_in_second_scenario_l172_172157


namespace combined_weight_is_correct_l172_172803

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end combined_weight_is_correct_l172_172803


namespace fraction_equality_l172_172091

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 :=
by
  sorry

end fraction_equality_l172_172091


namespace sin_C_eq_63_over_65_l172_172554

theorem sin_C_eq_63_over_65 (A B C : Real) (h₁ : 0 < A) (h₂ : A < π)
  (h₃ : 0 < B) (h₄ : B < π) (h₅ : 0 < C) (h₆ : C < π)
  (h₇ : A + B + C = π)
  (h₈ : Real.sin A = 5 / 13) (h₉ : Real.cos B = 3 / 5) : Real.sin C = 63 / 65 := 
by
  sorry

end sin_C_eq_63_over_65_l172_172554


namespace min_value_inequality_l172_172624

theorem min_value_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : a + b + c = 3) :
  (1 / (a + b)) + (1 / (b + c)) + (1 / (c + a)) ≥ 3 / 2 :=
sorry

end min_value_inequality_l172_172624


namespace greta_received_more_letters_l172_172630

noncomputable def number_of_letters_difference : ℕ :=
  let B := 40
  let M (G : ℕ) := 2 * (G + B)
  let total (G : ℕ) := G + B + M G
  let G := 50 -- Solved from the total equation
  G - B

theorem greta_received_more_letters : number_of_letters_difference = 10 :=
by
  sorry

end greta_received_more_letters_l172_172630


namespace henry_books_l172_172105

def initial_books := 99
def boxes := 3
def books_per_box := 15
def room_books := 21
def coffee_table_books := 4
def kitchen_books := 18
def picked_books := 12

theorem henry_books :
  (initial_books - (boxes * books_per_box + room_books + coffee_table_books + kitchen_books) + picked_books) = 23 :=
by
  sorry

end henry_books_l172_172105


namespace parabola_equation_l172_172811

theorem parabola_equation (p x0 : ℝ) (h_p : p > 0) (h_dist_focus : x0 + p / 2 = 10) (h_parabola : 2 * p * x0 = 36) :
  (2 * p = 4) ∨ (2 * p = 36) :=
by sorry

end parabola_equation_l172_172811


namespace combined_salaries_l172_172757

theorem combined_salaries (A B C D E : ℝ) 
  (hA : A = 9000) 
  (h_avg : (A + B + C + D + E) / 5 = 8200) :
  (B + C + D + E) = 32000 :=
by
  sorry

end combined_salaries_l172_172757


namespace circle_radius_is_six_l172_172701

open Real

theorem circle_radius_is_six
  (r : ℝ)
  (h : 2 * 3 * 2 * π * r = 2 * π * r^2) :
  r = 6 := sorry

end circle_radius_is_six_l172_172701


namespace trishul_investment_less_than_raghu_l172_172866

noncomputable def VishalInvestment (T : ℝ) : ℝ := 1.10 * T

noncomputable def TotalInvestment (T : ℝ) (R : ℝ) : ℝ :=
  T + VishalInvestment T + R

def RaghuInvestment : ℝ := 2100

def TotalSumInvested : ℝ := 6069

theorem trishul_investment_less_than_raghu :
  ∃ T : ℝ, TotalInvestment T RaghuInvestment = TotalSumInvested → (RaghuInvestment - T) / RaghuInvestment * 100 = 10 := by
  sorry

end trishul_investment_less_than_raghu_l172_172866


namespace jeans_price_increase_l172_172181

theorem jeans_price_increase 
  (C : ℝ) 
  (R : ℝ) 
  (F : ℝ) 
  (H1 : R = 1.40 * C)
  (H2 : F = 1.82 * C) 
  : (F - C) / C * 100 = 82 := 
sorry

end jeans_price_increase_l172_172181


namespace chess_games_won_l172_172695

theorem chess_games_won (W L : ℕ) (h1 : W + L = 44) (h2 : 4 * L = 7 * W) : W = 16 :=
by
  sorry

end chess_games_won_l172_172695


namespace probability_x_lt_2y_in_rectangle_l172_172207

-- Define the rectangle and the conditions
def in_rectangle (x y : ℝ) : Prop :=
  0 ≤ x ∧ x ≤ 4 ∧ 0 ≤ y ∧ y ≤ 3

-- Define the condition x < 2y
def condition_x_lt_2y (x y : ℝ) : Prop :=
  x < 2 * y

-- Define the probability calculation
theorem probability_x_lt_2y_in_rectangle :
  let rectangle_area := (4:ℝ) * 3
  let triangle_area := (1:ℝ) / 2 * 4 * 2
  let probability := triangle_area / rectangle_area
  probability = 1 / 3 :=
by
  sorry

end probability_x_lt_2y_in_rectangle_l172_172207


namespace necessary_condition_l172_172128

variables (a b : ℝ)

theorem necessary_condition (h : a > b) : a > b - 1 :=
sorry

end necessary_condition_l172_172128


namespace sum_of_8th_and_10th_terms_arithmetic_sequence_l172_172795

theorem sum_of_8th_and_10th_terms_arithmetic_sequence (a d : ℤ)
  (h1 : a + 3 * d = 25) (h2 : a + 5 * d = 61) :
  (a + 7 * d) + (a + 9 * d) = 230 := 
sorry

end sum_of_8th_and_10th_terms_arithmetic_sequence_l172_172795


namespace cost_price_l172_172780

theorem cost_price (MP SP C : ℝ) (h1 : MP = 112.5) (h2 : SP = 0.95 * MP) (h3 : SP = 1.25 * C) : 
  C = 85.5 :=
by
  sorry

end cost_price_l172_172780


namespace average_age_of_girls_l172_172992

theorem average_age_of_girls (total_students : ℕ) (boys_avg_age : ℝ) (school_avg_age : ℚ)
    (girls_count : ℕ) (total_age_school : ℝ) (boys_count : ℕ) 
    (total_age_boys : ℝ) (total_age_girls : ℝ): (total_students = 640) →
    (boys_avg_age = 12) →
    (school_avg_age = 47 / 4) →
    (girls_count = 160) →
    (total_students - girls_count = boys_count) →
    (boys_avg_age * boys_count = total_age_boys) →
    (school_avg_age * total_students = total_age_school) →
    (total_age_school - total_age_boys = total_age_girls) →
    total_age_girls / girls_count = 11 :=
by
  intros h_total_students h_boys_avg_age h_school_avg_age h_girls_count 
         h_boys_count h_total_age_boys h_total_age_school h_total_age_girls
  sorry

end average_age_of_girls_l172_172992


namespace meaningful_sqrt_l172_172306

theorem meaningful_sqrt (a : ℝ) (h : a - 4 ≥ 0) : a ≥ 4 :=
sorry

end meaningful_sqrt_l172_172306


namespace ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l172_172529

theorem ab_cd_ge_ac_bd_squared (a b c d : ℝ) : ((a^2 + b^2) * (c^2 + d^2)) ≥ (a * c + b * d)^2 := 
by sorry

theorem eq_condition_ad_eq_bc (a b c d : ℝ) (h : a * d = b * c) : ((a^2 + b^2) * (c^2 + d^2)) = (a * c + b * d)^2 := 
by sorry

end ab_cd_ge_ac_bd_squared_eq_condition_ad_eq_bc_l172_172529


namespace ticket_difference_l172_172272

-- Definitions representing the number of VIP and general admission tickets
def numTickets (V G : Nat) : Prop :=
  V + G = 320

def totalCost (V G : Nat) : Prop :=
  40 * V + 15 * G = 7500

-- Theorem stating that the difference between general admission and VIP tickets is 104
theorem ticket_difference (V G : Nat) (h1 : numTickets V G) (h2 : totalCost V G) : G - V = 104 := by
  sorry

end ticket_difference_l172_172272


namespace quadratic_real_roots_range_l172_172047

noncomputable def discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem quadratic_real_roots_range (m : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + m = 0) → m ≤ 9 / 4 :=
by
  sorry

end quadratic_real_roots_range_l172_172047


namespace bus_weight_conversion_l172_172928

noncomputable def round_to_nearest (x : ℚ) : ℤ := Int.floor (x + 0.5)

theorem bus_weight_conversion (kg_to_pound : ℚ) (bus_weight_kg : ℚ) 
  (h : kg_to_pound = 0.4536) (h_bus : bus_weight_kg = 350) : 
  round_to_nearest (bus_weight_kg / kg_to_pound) = 772 := by
  sorry

end bus_weight_conversion_l172_172928


namespace triangle_angle_sum_l172_172021

theorem triangle_angle_sum (x : ℝ) (h1 : 70 + 50 + x = 180) : x = 60 := by
  -- proof goes here
  sorry

end triangle_angle_sum_l172_172021


namespace solution_l172_172325

noncomputable def f (a b x : ℝ) : ℝ := a * Real.sin x + b * 3 * x + 4

def problem (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : Prop :=
  f a b (-Real.logb 3 3) = 3

theorem solution (a b : ℝ) (m : ℝ) (h1 : f a b m = 5) (h2 : m = Real.logb 3 10) : problem a b m h1 h2 :=
sorry

end solution_l172_172325


namespace celina_total_cost_l172_172629

def hoodieCost : ℝ := 80
def hoodieTaxRate : ℝ := 0.05

def flashlightCost := 0.20 * hoodieCost
def flashlightTaxRate : ℝ := 0.10

def bootsInitialCost : ℝ := 110
def bootsDiscountRate : ℝ := 0.10
def bootsTaxRate : ℝ := 0.05

def waterFilterCost : ℝ := 65
def waterFilterDiscountRate : ℝ := 0.25
def waterFilterTaxRate : ℝ := 0.08

def campingMatCost : ℝ := 45
def campingMatDiscountRate : ℝ := 0.15
def campingMatTaxRate : ℝ := 0.08

def backpackCost : ℝ := 105
def backpackTaxRate : ℝ := 0.08

def totalCost : ℝ := 
  let hoodieTotal := (hoodieCost * (1 + hoodieTaxRate))
  let flashlightTotal := (flashlightCost * (1 + flashlightTaxRate))
  let bootsTotal := ((bootsInitialCost * (1 - bootsDiscountRate)) * (1 + bootsTaxRate))
  let waterFilterTotal := ((waterFilterCost * (1 - waterFilterDiscountRate)) * (1 + waterFilterTaxRate))
  let campingMatTotal := ((campingMatCost * (1 - campingMatDiscountRate)) * (1 + campingMatTaxRate))
  let backpackTotal := (backpackCost * (1 + backpackTaxRate))
  hoodieTotal + flashlightTotal + bootsTotal + waterFilterTotal + campingMatTotal + backpackTotal

theorem celina_total_cost: totalCost = 413.91 := by
  sorry

end celina_total_cost_l172_172629


namespace solve_system_of_eqns_l172_172221

theorem solve_system_of_eqns :
  ∃ x y : ℝ, (x^2 + x * y + y = 1 ∧ y^2 + x * y + x = 5) ∧ ((x = -1 ∧ y = 3) ∨ (x = -1 ∧ y = -2)) :=
by
  sorry

end solve_system_of_eqns_l172_172221


namespace sales_worth_l172_172295

def old_scheme_remuneration (S : ℝ) : ℝ := 0.05 * S
def new_scheme_remuneration (S : ℝ) : ℝ := 1300 + 0.025 * (S - 4000)
def remuneration_difference (S : ℝ) : ℝ := new_scheme_remuneration S - old_scheme_remuneration S

theorem sales_worth (S : ℝ) (h : remuneration_difference S = 600) : S = 24000 :=
by
  sorry

end sales_worth_l172_172295


namespace image_of_2_in_set_B_l172_172335

theorem image_of_2_in_set_B (f : ℤ → ℤ) (h : ∀ x, f x = 2 * x + 1) : f 2 = 5 :=
by
  apply h

end image_of_2_in_set_B_l172_172335


namespace imaginary_unit_real_part_eq_l172_172002

theorem imaginary_unit_real_part_eq (a : ℝ) (i : ℂ) (h : i * i = -1) :
  (∃ r : ℝ, ((3 + i) * (a + 2 * i) / (1 + i) = r)) → a = 4 :=
by
  sorry

end imaginary_unit_real_part_eq_l172_172002


namespace tanner_savings_in_november_l172_172968

theorem tanner_savings_in_november(savings_sep : ℕ) (savings_oct : ℕ) 
(spending : ℕ) (leftover : ℕ) (N : ℕ) :
savings_sep = 17 →
savings_oct = 48 →
spending = 49 →
leftover = 41 →
((savings_sep + savings_oct + N - spending) = leftover) →
N = 25 :=
by
  intros h_sep h_oct h_spending h_leftover h_equation
  sorry

end tanner_savings_in_november_l172_172968


namespace lower_amount_rent_l172_172976

theorem lower_amount_rent (L : ℚ) (total_rent : ℚ) (reduction : ℚ)
  (h1 : total_rent = 2000)
  (h2 : reduction = 200)
  (h3 : 10 * (60 - L) = reduction) :
  L = 40 := by
  sorry

end lower_amount_rent_l172_172976


namespace intersectionAandB_l172_172307

def setA (x : ℝ) : Prop := abs (x + 3) + abs (x - 4) ≤ 9
def setB (x : ℝ) : Prop := ∃ t : ℝ, 0 < t ∧ x = 4 * t + 1 / t - 6

theorem intersectionAandB : {x : ℝ | setA x} ∩ {x : ℝ | setB x} = {x : ℝ | -2 ≤ x ∧ x ≤ 5} := 
by 
  sorry

end intersectionAandB_l172_172307


namespace find_starting_number_l172_172369

theorem find_starting_number (x : ℕ) (h1 : (50 + 250) / 2 = 150)
  (h2 : (x + 400) / 2 = 150 + 100) : x = 100 := by
  sorry

end find_starting_number_l172_172369


namespace ball_hits_ground_in_3_seconds_l172_172714

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 240

theorem ball_hits_ground_in_3_seconds :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 3 :=
sorry

end ball_hits_ground_in_3_seconds_l172_172714


namespace initial_avg_weight_l172_172593

theorem initial_avg_weight (A : ℝ) (h : 6 * A + 121 = 7 * 151) : A = 156 :=
by
sorry

end initial_avg_weight_l172_172593


namespace children_in_school_l172_172409

theorem children_in_school (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by
  sorry

end children_in_school_l172_172409


namespace women_at_each_table_l172_172661

theorem women_at_each_table (W : ℕ) (h1 : ∃ W, ∀ i : ℕ, (i < 7) → W + 2 = 7 * W + 14) (h2 : 7 * W + 14 = 63) : W = 7 :=
by
  sorry

end women_at_each_table_l172_172661


namespace immortal_flea_can_visit_every_natural_l172_172491

theorem immortal_flea_can_visit_every_natural :
  ∀ (k : ℕ), ∃ (jumps : ℕ → ℤ), (∀ n : ℕ, ∃ m : ℕ, jumps m = n) :=
by
  -- proof goes here
  sorry

end immortal_flea_can_visit_every_natural_l172_172491


namespace range_of_k_l172_172199

theorem range_of_k (f : ℝ → ℝ) (a : ℝ) (k : ℝ) 
  (h₀ : ∀ x > 0, f x = 2 - 1 / (a - x)^2) 
  (h₁ : ∀ x > 0, k^2 * x + f (1 / 4 * x + 1) > 0) : 
  k ≠ 0 :=
by
  -- proof goes here
  sorry

end range_of_k_l172_172199


namespace fourth_vertex_of_parallelogram_l172_172847

structure Point where
  x : ℝ
  y : ℝ

def Q := Point.mk 1 (-1)
def R := Point.mk (-1) 0
def S := Point.mk 0 1
def V := Point.mk (-2) 2

theorem fourth_vertex_of_parallelogram (Q R S V : Point) :
  Q = ⟨1, -1⟩ ∧ R = ⟨-1, 0⟩ ∧ S = ⟨0, 1⟩ → V = ⟨-2, 2⟩ := by 
  sorry

end fourth_vertex_of_parallelogram_l172_172847


namespace sum_of_reciprocals_is_one_l172_172078

theorem sum_of_reciprocals_is_one (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / (x : ℚ)) + (1 / (y : ℚ)) + (1 / (z : ℚ)) = 1 ↔ (x, y, z) = (2, 4, 4) ∨ 
                                                    (x, y, z) = (2, 3, 6) ∨ 
                                                    (x, y, z) = (3, 3, 3) :=
by 
  sorry

end sum_of_reciprocals_is_one_l172_172078


namespace savings_amount_l172_172704

-- Define the conditions for Celia's spending
def food_spending_per_week : ℝ := 100
def weeks : ℕ := 4
def rent_spending : ℝ := 1500
def video_streaming_services_spending : ℝ := 30
def cell_phone_usage_spending : ℝ := 50
def savings_rate : ℝ := 0.10

-- Define the total spending calculation
def total_spending : ℝ :=
  food_spending_per_week * weeks + rent_spending + video_streaming_services_spending + cell_phone_usage_spending

-- Define the savings calculation
def savings : ℝ :=
  savings_rate * total_spending

-- Prove the amount of savings
theorem savings_amount : savings = 198 :=
by
  -- This is the statement that needs to be proven, hence adding a placeholder proof.
  sorry

end savings_amount_l172_172704


namespace simplified_expression_l172_172117

variable (m : ℝ) (h : m = Real.sqrt 3)

theorem simplified_expression : (m - (m + 9) / (m + 1)) / ((m^2 + 3 * m) / (m + 1)) = 1 - Real.sqrt 3 :=
by
  rw [h]
  sorry

end simplified_expression_l172_172117


namespace simplify_cos_diff_l172_172257

theorem simplify_cos_diff :
  let a := Real.cos (36 * Real.pi / 180)
  let b := Real.cos (72 * Real.pi / 180)
  (b = 2 * a^2 - 1) → 
  (a = 1 - 2 * b^2) →
  a - b = 1 / 2 :=
by
  sorry

end simplify_cos_diff_l172_172257


namespace toothpicks_at_150th_stage_l172_172444

theorem toothpicks_at_150th_stage (a₁ d n : ℕ) (h₁ : a₁ = 6) (hd : d = 5) (hn : n = 150) :
  (n * (2 * a₁ + (n - 1) * d)) / 2 = 56775 :=
by
  sorry -- Proof to be completed.

end toothpicks_at_150th_stage_l172_172444


namespace find_K_values_l172_172953

-- Define summation of first K natural numbers
def sum_natural_numbers (K : ℕ) : ℕ :=
  K * (K + 1) / 2

-- Define the main problem conditions
theorem find_K_values (K N : ℕ) (hN_positive : N > 0) (hN_bound : N < 150) (h_sum_eq : sum_natural_numbers K = 3 * N^2) :
  K = 2 ∨ K = 12 ∨ K = 61 :=
  sorry

end find_K_values_l172_172953


namespace smallest_prime_with_conditions_l172_172353

def is_prime (n : ℕ) : Prop := ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10 
  let units := n % 10 
  units * 10 + tens

theorem smallest_prime_with_conditions : 
  ∃ (p : ℕ), is_prime p ∧ 20 ≤ p ∧ p < 30 ∧ (reverse_digits p) < 100 ∧ is_composite (reverse_digits p) ∧ p = 23 :=
by
  sorry

end smallest_prime_with_conditions_l172_172353


namespace find_k_value_l172_172647

variable (S : ℕ → ℤ) (n : ℕ)

-- Conditions
def is_arithmetic_sum (S : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, S n = n * (2 * a + (n - 1) * d) / 2

axiom S3_eq_S8 (S : ℕ → ℤ) (hS : is_arithmetic_sum S) : S 3 = S 8
axiom Sk_eq_S7 (S : ℕ → ℤ) (k : ℕ) (hS: is_arithmetic_sum S)  : S 7 = S k

theorem find_k_value (S : ℕ → ℤ) (hS: is_arithmetic_sum S) :  S 3 = S 8 → S 7 = S 4 :=
by
  sorry

end find_k_value_l172_172647


namespace value_of_a_l172_172839

theorem value_of_a (a : ℤ) (x y : ℝ) :
  (a - 2) ≠ 0 →
  (2 + |a| + 1 = 5) →
  a = -2 :=
by
  intro ha hdeg
  sorry

end value_of_a_l172_172839


namespace find_train_length_l172_172285

noncomputable def speed_kmh : ℝ := 45
noncomputable def bridge_length : ℝ := 245.03
noncomputable def time_seconds : ℝ := 30
noncomputable def speed_ms : ℝ := (speed_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := speed_ms * time_seconds
noncomputable def train_length : ℝ := total_distance - bridge_length

theorem find_train_length : train_length = 129.97 := 
by
  sorry

end find_train_length_l172_172285


namespace amount_given_by_mom_l172_172388

def amount_spent_by_Mildred : ℕ := 25
def amount_spent_by_Candice : ℕ := 35
def amount_left : ℕ := 40

theorem amount_given_by_mom : 
  (amount_spent_by_Mildred + amount_spent_by_Candice + amount_left) = 100 := by
  sorry

end amount_given_by_mom_l172_172388


namespace tomato_seed_cost_l172_172799

theorem tomato_seed_cost (T : ℝ) 
  (h1 : 3 * 2.50 + 4 * T + 5 * 0.90 = 18) : 
  T = 1.50 := 
by
  sorry

end tomato_seed_cost_l172_172799


namespace time_interval_between_recordings_is_5_seconds_l172_172684

theorem time_interval_between_recordings_is_5_seconds
  (instances_per_hour : ℕ)
  (seconds_per_hour : ℕ)
  (h1 : instances_per_hour = 720)
  (h2 : seconds_per_hour = 3600) :
  seconds_per_hour / instances_per_hour = 5 :=
by
  -- proof omitted
  sorry

end time_interval_between_recordings_is_5_seconds_l172_172684


namespace find_c_l172_172104

noncomputable def parabola_equation (a b c y : ℝ) : ℝ :=
  a * y^2 + b * y + c

theorem find_c (a b c : ℝ) (h_vertex : (-4, 2) = (-4, 2)) (h_point : (-2, 4) = (-2, 4)) :
  ∃ c : ℝ, parabola_equation a b c 0 = -2 :=
  by {
    use -2,
    sorry
  }

end find_c_l172_172104


namespace probability_2x_less_y_equals_one_over_eight_l172_172511

noncomputable def probability_2x_less_y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1 / 2) * 3 * 1.5
  let area_rectangle : ℚ := 6 * 3
  area_triangle / area_rectangle

theorem probability_2x_less_y_equals_one_over_eight :
  probability_2x_less_y_in_rectangle = 1 / 8 :=
by
  sorry

end probability_2x_less_y_equals_one_over_eight_l172_172511


namespace bus_stop_minutes_per_hour_l172_172608

/-- Given the average speed of a bus excluding stoppages is 60 km/hr
and including stoppages is 15 km/hr, prove that the bus stops for 45 minutes per hour. -/
theorem bus_stop_minutes_per_hour
  (speed_no_stops : ℝ := 60)
  (speed_with_stops : ℝ := 15) :
  ∃ t : ℝ, t = 45 :=
by
  sorry

end bus_stop_minutes_per_hour_l172_172608


namespace erasers_in_each_box_l172_172108

theorem erasers_in_each_box (boxes : ℕ) (price_per_eraser : ℚ) (total_money_made : ℚ) (total_erasers_sold : ℕ) (erasers_per_box : ℕ) :
  boxes = 48 → price_per_eraser = 0.75 → total_money_made = 864 → total_erasers_sold = 1152 → total_erasers_sold / boxes = erasers_per_box → erasers_per_box = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end erasers_in_each_box_l172_172108


namespace Stephen_total_distance_l172_172864

theorem Stephen_total_distance 
  (round_trips : ℕ := 10) 
  (mountain_height : ℕ := 40000) 
  (fraction_of_height : ℚ := 3/4) :
  (round_trips * (2 * (fraction_of_height * mountain_height))) = 600000 :=
by
  sorry

end Stephen_total_distance_l172_172864


namespace doubled_sum_of_squares_l172_172900

theorem doubled_sum_of_squares (a b : ℝ) : 
  2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := 
by
  sorry

end doubled_sum_of_squares_l172_172900


namespace smallest_boxes_l172_172997

-- Definitions based on the conditions:
def divisible_by (n d : Nat) : Prop := ∃ k, n = d * k

-- The statement to be proved:
theorem smallest_boxes (n : Nat) : 
  divisible_by n 5 ∧ divisible_by n 24 -> n = 120 :=
by sorry

end smallest_boxes_l172_172997


namespace lance_more_pebbles_l172_172672

-- Given conditions
def candy_pebbles : ℕ := 4
def lance_pebbles : ℕ := 3 * candy_pebbles

-- Proof statement
theorem lance_more_pebbles : lance_pebbles - candy_pebbles = 8 :=
by
  sorry

end lance_more_pebbles_l172_172672


namespace abc_sum_eq_sqrt34_l172_172071

noncomputable def abc_sum (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 16)
                          (h2 : ab + bc + ca = 9)
                          (h3 : a^2 + b^2 = 10)
                          (h4 : 0 ≤ a) (h5 : 0 ≤ b) (h6 : 0 ≤ c) : ℝ :=
a + b + c

theorem abc_sum_eq_sqrt34 (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 16)
  (h2 : ab + bc + ca = 9)
  (h3 : a^2 + b^2 = 10)
  (h4 : 0 ≤ a)
  (h5 : 0 ≤ b)
  (h6 : 0 ≤ c) :
  abc_sum a b c h1 h2 h3 h4 h5 h6 = Real.sqrt 34 :=
by
  sorry

end abc_sum_eq_sqrt34_l172_172071


namespace pages_copied_l172_172731

theorem pages_copied (cost_per_page total_cents : ℤ) (h1 : cost_per_page = 3) (h2 : total_cents = 1500) :
  total_cents / cost_per_page = 500 :=
by
  sorry

end pages_copied_l172_172731


namespace orange_juice_fraction_in_mixture_l172_172302

theorem orange_juice_fraction_in_mixture :
  let capacity1 := 800
  let capacity2 := 700
  let fraction1 := (1 : ℚ) / 4
  let fraction2 := (3 : ℚ) / 7
  let orange_juice1 := capacity1 * fraction1
  let orange_juice2 := capacity2 * fraction2
  let total_orange_juice := orange_juice1 + orange_juice2
  let total_volume := capacity1 + capacity2
  let fraction := total_orange_juice / total_volume
  fraction = (1 : ℚ) / 3 := by
  sorry

end orange_juice_fraction_in_mixture_l172_172302


namespace percent_of_a_is_4b_l172_172687

theorem percent_of_a_is_4b (b : ℝ) (a : ℝ) (h : a = 1.8 * b) : (4 * b / a) * 100 = 222.22 := 
by {
  sorry
}

end percent_of_a_is_4b_l172_172687


namespace sum_of_digits_eq_28_l172_172061

theorem sum_of_digits_eq_28 (A B C D E : ℕ) 
  (hA : 0 ≤ A ∧ A ≤ 9) 
  (hB : 0 ≤ B ∧ B ≤ 9) 
  (hC : 0 ≤ C ∧ C ≤ 9) 
  (hD : 0 ≤ D ∧ D ≤ 9) 
  (hE : 0 ≤ E ∧ E ≤ 9) 
  (unique_digits : (A ≠ B) ∧ (A ≠ C) ∧ (A ≠ D) ∧ (A ≠ E) ∧ (B ≠ C) ∧ (B ≠ D) ∧ (B ≠ E) ∧ (C ≠ D) ∧ (C ≠ E) ∧ (D ≠ E)) 
  (h : (10 * A + B) * (10 * C + D) = 111 * E) : 
  A + B + C + D + E = 28 :=
sorry

end sum_of_digits_eq_28_l172_172061


namespace cupcakes_initial_count_l172_172950

theorem cupcakes_initial_count (x : ℕ) (h1 : x - 5 + 10 = 24) : x = 19 :=
by sorry

end cupcakes_initial_count_l172_172950


namespace area_of_quadrilateral_l172_172638

noncomputable def quadrilateral_area
  (AB CD r : ℝ) (k : ℝ) 
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) : ℝ := 
  (3 * r^2 * abs (1 - k^2)) / (1 + k^2)

theorem area_of_quadrilateral
  (AB CD r : ℝ) (k : ℝ)
  (h_perpendicular : AB * CD = 0)
  (h_equal_diameters : AB = 2 * r ∧ CD = 2 * r)
  (h_ratio : BC / AD = k) :
  quadrilateral_area AB CD r k h_perpendicular h_equal_diameters h_ratio = (3 * r^2 * abs (1 - k^2)) / (1 + k^2) :=
sorry

end area_of_quadrilateral_l172_172638


namespace expression_evaluation_l172_172917

theorem expression_evaluation (m n : ℤ) (h1 : m = 2) (h2 : n = -1 ^ 2023) :
  (2 * m + n) * (2 * m - n) - (2 * m - n) ^ 2 + 2 * n * (m + n) = -12 := by
  sorry

end expression_evaluation_l172_172917


namespace log_equation_solution_l172_172777

theorem log_equation_solution (x : ℝ) (hx : 0 < x) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 ↔ (x = 4 ∨ x = 8) :=
by
  sorry

end log_equation_solution_l172_172777


namespace find_z_in_sequence_l172_172774

theorem find_z_in_sequence (x y z a b : ℤ) 
  (h1 : b = 1)
  (h2 : a + b = 0)
  (h3 : y + a = 1)
  (h4 : z + y = 3)
  (h5 : x + z = 2) :
  z = 1 :=
sorry

end find_z_in_sequence_l172_172774


namespace cubic_sum_identity_l172_172700

variables (x y z : ℝ)

theorem cubic_sum_identity (h1 : x + y + z = 10) (h2 : xy + xz + yz = 30) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 100 :=
sorry

end cubic_sum_identity_l172_172700


namespace nancy_yearly_payment_l172_172528

open Real

-- Define the monthly cost of the car insurance
def monthly_cost : ℝ := 80

-- Nancy's percentage contribution
def percentage : ℝ := 0.40

-- Calculate the monthly payment Nancy will make
def monthly_payment : ℝ := percentage * monthly_cost

-- Calculate the yearly payment Nancy will make
def yearly_payment : ℝ := 12 * monthly_payment

-- State the proof problem
theorem nancy_yearly_payment : yearly_payment = 384 :=
by
  -- Proof goes here
  sorry

end nancy_yearly_payment_l172_172528


namespace white_balls_count_l172_172648

theorem white_balls_count {T W : ℕ} (h1 : 3 * 4 = T) (h2 : T - 3 = W) : W = 9 :=
by 
    sorry

end white_balls_count_l172_172648


namespace percentage_of_customers_purchased_l172_172726

theorem percentage_of_customers_purchased (ad_cost : ℕ) (customers : ℕ) (price_per_sale : ℕ) (profit : ℕ)
  (h1 : ad_cost = 1000)
  (h2 : customers = 100)
  (h3 : price_per_sale = 25)
  (h4 : profit = 1000) :
  (profit / price_per_sale / customers) * 100 = 40 :=
by
  sorry

end percentage_of_customers_purchased_l172_172726


namespace correct_option_is_C_l172_172794

def option_A (x : ℝ) : Prop := (-x^2)^3 = -x^5
def option_B (x : ℝ) : Prop := x^2 + x^3 = x^5
def option_C (x : ℝ) : Prop := x^3 * x^4 = x^7
def option_D (x : ℝ) : Prop := 2 * x^3 - x^3 = 1

theorem correct_option_is_C (x : ℝ) : ¬ option_A x ∧ ¬ option_B x ∧ option_C x ∧ ¬ option_D x :=
by
  sorry

end correct_option_is_C_l172_172794


namespace workshop_workers_l172_172110

theorem workshop_workers (W N: ℕ) 
  (h1: 8000 * W = 70000 + 6000 * N) 
  (h2: W = 7 + N) : 
  W = 14 := 
  by 
    sorry

end workshop_workers_l172_172110


namespace find_all_functions_l172_172427

theorem find_all_functions (n : ℕ) (h_pos : 0 < n) (f : ℝ → ℝ) :
  (∀ x y : ℝ, (f x)^n * f (x + y) = (f x)^(n + 1) + x^n * f y) ↔
  (if n % 2 = 1 then ∀ x, f x = 0 ∨ f x = x else ∀ x, f x = 0 ∨ f x = x ∨ f x = -x) :=
sorry

end find_all_functions_l172_172427


namespace common_difference_is_two_l172_172283

-- Define the properties and conditions.
variables {a : ℕ → ℝ} {d : ℝ}

-- An arithmetic sequence definition.
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement to be proved.
theorem common_difference_is_two (h1 : a 1 + a 5 = 10) (h2 : a 4 = 7) (h3 : arithmetic_sequence a d) : 
  d = 2 :=
sorry

end common_difference_is_two_l172_172283


namespace parallel_line_through_point_l172_172769

theorem parallel_line_through_point (x y : ℝ) (m b : ℝ) (h₁ : y = -3 * x + b) (h₂ : x = 2) (h₃ : y = 1) :
  b = 7 :=
by
  -- x, y are components of the point P (2,1)
  -- equation of line parallel to y = -3x + 2 has slope -3 but different y-intercept
  -- y = -3x + b is the general form, and must pass through (2,1) => 1 = -3*2 + b
  -- Therefore, b must be 7
  sorry

end parallel_line_through_point_l172_172769


namespace find_cost_price_l172_172501

-- Condition 1: The owner charges his customer 15% more than the cost price.
def selling_price (C : Real) : Real := C * 1.15

-- Condition 2: A customer paid Rs. 8325 for the computer table.
def paid_amount : Real := 8325

-- Define the cost price and its expected value
def cost_price : Real := 7239.13

-- The theorem to prove that the cost price matches the expected value
theorem find_cost_price : 
  ∃ C : Real, selling_price C = paid_amount ∧ C = cost_price :=
by
  sorry

end find_cost_price_l172_172501


namespace total_amount_paid_is_correct_l172_172671

-- Define the initial conditions
def tireA_price : ℕ := 75
def tireA_discount : ℕ := 20
def tireB_price : ℕ := 90
def tireB_discount : ℕ := 30
def tireC_price : ℕ := 120
def tireC_discount : ℕ := 45
def tireD_price : ℕ := 150
def tireD_discount : ℕ := 60
def installation_fee : ℕ := 15
def disposal_fee : ℕ := 5

-- Calculate the total amount paid
def total_paid : ℕ :=
  let tireA_total := (tireA_price - tireA_discount) + installation_fee + disposal_fee
  let tireB_total := (tireB_price - tireB_discount) + installation_fee + disposal_fee
  let tireC_total := (tireC_price - tireC_discount) + installation_fee + disposal_fee
  let tireD_total := (tireD_price - tireD_discount) + installation_fee + disposal_fee
  tireA_total + tireB_total + tireC_total + tireD_total

-- Statement of the theorem
theorem total_amount_paid_is_correct :
  total_paid = 360 :=
by
  -- proof goes here
  sorry

end total_amount_paid_is_correct_l172_172671


namespace smallest_n_for_violet_candy_l172_172027

theorem smallest_n_for_violet_candy (p y o n : Nat) (h : 10 * p = 12 * y ∧ 12 * y = 18 * o ∧ 18 * o = 24 * n) :
  n = 8 :=
by 
  sorry

end smallest_n_for_violet_candy_l172_172027


namespace kendra_bought_3_hats_l172_172141

-- Define the price of a wooden toy
def price_of_toy : ℕ := 20

-- Define the price of a hat
def price_of_hat : ℕ := 10

-- Define the amount Kendra went to the shop with
def initial_amount : ℕ := 100

-- Define the number of wooden toys Kendra bought
def number_of_toys : ℕ := 2

-- Define the amount of change Kendra received
def change_received : ℕ := 30

-- Prove that Kendra bought 3 hats
theorem kendra_bought_3_hats : 
  initial_amount - change_received - (number_of_toys * price_of_toy) = 3 * price_of_hat := by
  sorry

end kendra_bought_3_hats_l172_172141


namespace company_total_employees_l172_172433

def total_employees_after_hiring (T : ℕ) (before_hiring_female_percentage : ℚ) (additional_male_workers : ℕ) (after_hiring_female_percentage : ℚ) : ℕ :=
  T + additional_male_workers

theorem company_total_employees (T : ℕ)
  (before_hiring_female_percentage : ℚ)
  (additional_male_workers : ℕ)
  (after_hiring_female_percentage : ℚ)
  (h_before_percent : before_hiring_female_percentage = 0.60)
  (h_additional_male : additional_male_workers = 28)
  (h_after_percent : after_hiring_female_percentage = 0.55)
  (h_equation : (before_hiring_female_percentage * T)/(T + additional_male_workers) = after_hiring_female_percentage) :
  total_employees_after_hiring T before_hiring_female_percentage additional_male_workers after_hiring_female_percentage = 336 :=
by {
  -- This is where you add the proof steps.
  sorry
}

end company_total_employees_l172_172433


namespace bridge_length_at_least_200_l172_172487

theorem bridge_length_at_least_200 :
  ∀ (length_train : ℝ) (speed_kmph : ℝ) (time_secs : ℝ),
  length_train = 200 ∧ speed_kmph = 32 ∧ time_secs = 20 →
  ∃ l : ℝ, l ≥ length_train :=
by
  sorry

end bridge_length_at_least_200_l172_172487


namespace no_power_of_q_l172_172211

theorem no_power_of_q (n : ℕ) (hn : n > 0) (q : ℕ) (hq : Prime q) : ¬ (∃ k : ℕ, n^q + ((n-1)/2)^2 = q^k) := 
by
  sorry  -- proof steps are not required as per instructions

end no_power_of_q_l172_172211


namespace total_balls_estimate_l172_172963

theorem total_balls_estimate 
  (total_balls : ℕ) 
  (red_balls : ℕ) 
  (frequency : ℚ)
  (h_red_balls : red_balls = 12)
  (h_frequency : frequency = 0.6) 
  (h_fraction : (red_balls : ℚ) / total_balls = frequency): 
  total_balls = 20 := 
by 
  sorry

end total_balls_estimate_l172_172963


namespace equality_condition_l172_172595

theorem equality_condition (a b c : ℝ) :
  a + b + c = (a + b) * (a + c) → a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end equality_condition_l172_172595


namespace odd_coefficients_in_polynomial_l172_172448

noncomputable def number_of_odd_coefficients (n : ℕ) : ℕ :=
  (2^n - 1) / 3 * 4 + 1

theorem odd_coefficients_in_polynomial (n : ℕ) (hn : 0 < n) :
  (x^2 + x + 1)^n = number_of_odd_coefficients n :=
sorry

end odd_coefficients_in_polynomial_l172_172448


namespace line_through_point_equidistant_l172_172455

open Real

structure Point where
  x : ℝ
  y : ℝ

def line_equation (a b c : ℝ) (p : Point) : Prop :=
  a * p.x + b * p.y + c = 0

def equidistant (p1 p2 : Point) (l : ℝ × ℝ × ℝ) : Prop :=
  let (a, b, c) := l
  let dist_from_p1 := abs (a * p1.x + b * p1.y + c) / sqrt (a^2 + b^2)
  let dist_from_p2 := abs (a * p2.x + b * p2.y + c) / sqrt (a^2 + b^2)
  dist_from_p1 = dist_from_p2

theorem line_through_point_equidistant (a b c : ℝ)
  (P : Point) (A : Point) (B : Point) :
  (P = ⟨1, 2⟩) →
  (A = ⟨2, 2⟩) →
  (B = ⟨4, -6⟩) →
  line_equation a b c P →
  equidistant A B (a, b, c) →
  (a = 2 ∧ b = 1 ∧ c = -4) :=
by
  sorry

end line_through_point_equidistant_l172_172455


namespace product_of_all_possible_values_of_x_l172_172282

def conditions (x : ℚ) : Prop := abs (18 / x - 4) = 3

theorem product_of_all_possible_values_of_x:
  ∃ x1 x2 : ℚ, conditions x1 ∧ conditions x2 ∧ ((18 * 18) / (x1 * x2) = 324 / 7) :=
sorry

end product_of_all_possible_values_of_x_l172_172282


namespace polynomial_factorization_l172_172818

theorem polynomial_factorization : ∃ q : Polynomial ℝ, (Polynomial.X ^ 4 - 6 * Polynomial.X ^ 2 + 25) = (Polynomial.X ^ 2 + 5) * q :=
by
  sorry

end polynomial_factorization_l172_172818


namespace largest_number_of_stamps_per_page_l172_172355

theorem largest_number_of_stamps_per_page :
  Nat.gcd (Nat.gcd 1200 1800) 2400 = 600 :=
sorry

end largest_number_of_stamps_per_page_l172_172355


namespace problem_l172_172633

noncomputable def a : ℝ := Real.exp 1 - 2
noncomputable def b : ℝ := 1 - Real.log 2
noncomputable def c : ℝ := Real.exp (Real.exp 1) - Real.exp 2

theorem problem (a_def : a = Real.exp 1 - 2) 
                (b_def : b = 1 - Real.log 2) 
                (c_def : c = Real.exp (Real.exp 1) - Real.exp 2) : 
                c > a ∧ a > b := 
by 
  rw [a_def, b_def, c_def]
  sorry

end problem_l172_172633


namespace rectangle_length_35_l172_172126

theorem rectangle_length_35
  (n_rectangles : ℕ) (area_abcd : ℝ) (rect_length_multiple : ℕ) (rect_width_multiple : ℕ) 
  (n_rectangles_eq : n_rectangles = 6)
  (area_abcd_eq : area_abcd = 4800)
  (rect_length_multiple_eq : rect_length_multiple = 3)
  (rect_width_multiple_eq : rect_width_multiple = 2) :
  ∃ y : ℝ, round y = 35 ∧ y^2 * (4/3) = area_abcd :=
by
  sorry


end rectangle_length_35_l172_172126


namespace range_of_a_l172_172096

-- Definitions of propositions p and q

def p (a : ℝ) : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

-- Theorem stating the range of values for a given p ∧ q is true

theorem range_of_a (a : ℝ) : (p a ∧ q a) → (a ≤ -2 ∨ a = 1) :=
by sorry

end range_of_a_l172_172096


namespace right_triangle_inradius_height_ratio_l172_172463

-- Define a right triangle with sides a, b, and hypotenuse c
variables {a b c : ℝ}
-- Define the altitude from the right angle vertex
variables {h : ℝ}
-- Define the inradius of the triangle
variables {r : ℝ}

-- Define the conditions: right triangle 
-- and the relationships for h and r
def is_right_triangle (a b c : ℝ) : Prop := a^2 + b^2 = c^2
def altitude (h : ℝ) (a b c : ℝ) : Prop := h = (a * b) / c
def inradius (r : ℝ) (a b c : ℝ) : Prop := r = (a + b - c) / 2

theorem right_triangle_inradius_height_ratio {a b c h r : ℝ} 
  (Hrt : is_right_triangle a b c)
  (Hh : altitude h a b c)
  (Hr : inradius r a b c) : 
  0.4 < r / h ∧ r / h < 0.5 :=
sorry

end right_triangle_inradius_height_ratio_l172_172463


namespace range_of_a_l172_172958

noncomputable def setM (a : ℝ) : Set ℝ := {x | x * (x - a - 1) < 0}
def setN : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem range_of_a (a : ℝ) : setM a ∪ setN = setN ↔ (-2 ≤ a ∧ a ≤ 2) :=
sorry

end range_of_a_l172_172958


namespace coefficient_x8_expansion_l172_172936

-- Define the problem statement in Lean
theorem coefficient_x8_expansion : 
  (Nat.choose 7 4) * (1 : ℤ)^3 * (-2 : ℤ)^4 = 560 :=
by
  sorry

end coefficient_x8_expansion_l172_172936


namespace medals_award_count_l172_172696

theorem medals_award_count :
  let total_ways (n k : ℕ) := n.factorial / (n - k).factorial
  ∃ (award_ways : ℕ), 
    let no_americans := total_ways 6 3
    let one_american := 4 * 3 * total_ways 6 2
    award_ways = no_americans + one_american ∧
    award_ways = 480 :=
by
  sorry

end medals_award_count_l172_172696


namespace Seth_bought_20_cartons_of_ice_cream_l172_172571

-- Definitions from conditions
def ice_cream_cost_per_carton : ℕ := 6
def yogurt_cost_per_carton : ℕ := 1
def num_yogurt_cartons : ℕ := 2
def extra_amount_spent_on_ice_cream : ℕ := 118

-- Let x be the number of cartons of ice cream Seth bought
def num_ice_cream_cartons (x : ℕ) : Prop :=
  ice_cream_cost_per_carton * x = num_yogurt_cartons * yogurt_cost_per_carton + extra_amount_spent_on_ice_cream

-- The proof goal
theorem Seth_bought_20_cartons_of_ice_cream : num_ice_cream_cartons 20 :=
by
  unfold num_ice_cream_cartons
  unfold ice_cream_cost_per_carton yogurt_cost_per_carton num_yogurt_cartons extra_amount_spent_on_ice_cream
  sorry

end Seth_bought_20_cartons_of_ice_cream_l172_172571


namespace father_age_when_rachel_is_25_l172_172681

-- Definitions for Rachel's age, Grandfather's age, Mother's age, and Father's age
def rachel_age : ℕ := 12
def grandfather_age : ℕ := 7 * rachel_age
def mother_age : ℕ := grandfather_age / 2
def father_age : ℕ := mother_age + 5
def years_until_rachel_is_25 : ℕ := 25 - rachel_age
def fathers_age_when_rachel_is_25 : ℕ := father_age + years_until_rachel_is_25

-- Theorem to prove that Rachel's father will be 60 years old when Rachel is 25 years old
theorem father_age_when_rachel_is_25 : fathers_age_when_rachel_is_25 = 60 := by
  sorry

end father_age_when_rachel_is_25_l172_172681


namespace find_point_P_l172_172273

-- Define the function
def f (x : ℝ) := x^4 - 2 * x

-- Define the derivative of the function
def f' (x : ℝ) := 4 * x^3 - 2

theorem find_point_P :
  ∃ (P : ℝ × ℝ), (f' P.1 = 2) ∧ (f P.1 = P.2) ∧ (P = (1, -1)) :=
by
  -- here would go the actual proof
  sorry

end find_point_P_l172_172273


namespace divisible_by_eight_l172_172534

def expr (n : ℕ) : ℕ := 3^(4*n + 1) + 5^(2*n + 1)

theorem divisible_by_eight (n : ℕ) : expr n % 8 = 0 :=
  sorry

end divisible_by_eight_l172_172534


namespace stmt_A_stmt_B_stmt_C_stmt_D_l172_172832
open Real

def x_and_y_conditions := ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 2 * y = 3

theorem stmt_A : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (2 * (x * x + y * y) = 4) :=
by sorry

theorem stmt_B : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x * y = 9 / 8) :=
by sorry

theorem stmt_C : x_and_y_conditions → ¬ (∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (sqrt (x) + sqrt (2 * y) = sqrt 6)) :=
by sorry

theorem stmt_D : x_and_y_conditions → ∃ x y : ℝ, (0 < x ∧ 0 < y ∧ x + 2 * y = 3) ∧ (x^2 + 4 * y^2 = 9 / 2) :=
by sorry

end stmt_A_stmt_B_stmt_C_stmt_D_l172_172832


namespace razorback_tshirt_shop_sales_l172_172738

theorem razorback_tshirt_shop_sales :
  let price_per_tshirt := 16 
  let tshirts_sold := 45 
  price_per_tshirt * tshirts_sold = 720 :=
by
  sorry

end razorback_tshirt_shop_sales_l172_172738


namespace rectangles_containment_existence_l172_172250

theorem rectangles_containment_existence :
  (∃ (rects : ℕ → ℕ × ℕ), (∀ n : ℕ, (rects n).fst > 0 ∧ (rects n).snd > 0) ∧
   (∀ n m : ℕ, n ≠ m → ¬((rects n).fst ≤ (rects m).fst ∧ (rects n).snd ≤ (rects m).snd))) →
  false :=
by
  sorry

end rectangles_containment_existence_l172_172250


namespace peter_age_problem_l172_172653

theorem peter_age_problem
  (P J : ℕ) 
  (h1 : J = P + 12)
  (h2 : P - 10 = 1/3 * (J - 10)) : P = 16 :=
sorry

end peter_age_problem_l172_172653


namespace find_alpha_l172_172172

theorem find_alpha (α : ℝ) (hα : 0 ≤ α ∧ α < 2 * Real.pi) 
  (l1 : ∀ x y : ℝ, x * Real.cos α - y - 1 = 0) 
  (l2 : ∀ x y : ℝ, x + y * Real.sin α + 1 = 0) :
  α = Real.pi / 4 ∨ α = 5 * Real.pi / 4 :=
sorry

end find_alpha_l172_172172


namespace find_investment_duration_l172_172468

theorem find_investment_duration :
  ∀ (A P R I : ℝ) (T : ℝ),
    A = 1344 →
    P = 1200 →
    R = 5 →
    I = A - P →
    I = (P * R * T) / 100 →
    T = 2.4 :=
by
  intros A P R I T hA hP hR hI1 hI2
  sorry

end find_investment_duration_l172_172468


namespace find_k_slope_eq_l172_172032

theorem find_k_slope_eq :
  ∃ k: ℝ, (∃ k: ℝ, ((k - 4) / 7 = (-2 - k) / 14) → k = 2) :=
by
  sorry

end find_k_slope_eq_l172_172032


namespace circular_arc_sum_l172_172218

theorem circular_arc_sum (n : ℕ) (h₁ : n > 0) :
  ∀ s : ℕ, (1 ≤ s ∧ s ≤ (n * (n + 1)) / 2) →
  ∃ arc_sum : ℕ, arc_sum = s := 
by
  sorry

end circular_arc_sum_l172_172218


namespace shorter_base_of_isosceles_trapezoid_l172_172411

theorem shorter_base_of_isosceles_trapezoid
  (a b : ℝ)
  (h : a > b)
  (h_division : (a + b) / 2 = (a - b) / 2 + 10) :
  b = 10 :=
by
  sorry

end shorter_base_of_isosceles_trapezoid_l172_172411


namespace average_percentage_score_is_71_l172_172504

-- Define the number of students.
def number_of_students : ℕ := 150

-- Define the scores and their corresponding frequencies.
def scores_and_frequencies : List (ℕ × ℕ) :=
  [(100, 10), (95, 20), (85, 45), (75, 30), (65, 25), (55, 15), (45, 5)]

-- Define the total points scored by all students.
def total_points_scored : ℕ := 
  scores_and_frequencies.foldl (λ acc pair => acc + pair.1 * pair.2) 0

-- Define the average percentage score.
def average_score : ℚ := total_points_scored / number_of_students

-- Statement of the proof problem.
theorem average_percentage_score_is_71 :
  average_score = 71.0 := by
  sorry

end average_percentage_score_is_71_l172_172504


namespace larger_segment_of_triangle_l172_172414

theorem larger_segment_of_triangle (a b c : ℝ) (h : ℝ) (hc : c = 100) (ha : a = 40) (hb : b = 90) 
  (h_triangle : a^2 + h^2 = x^2)
  (h_triangle2 : b^2 + h^2 = (100 - x)^2) :
  100 - x = 82.5 :=
sorry

end larger_segment_of_triangle_l172_172414


namespace combination_x_l172_172006
noncomputable def C (n k : ℕ) : ℕ := Nat.choose n k

theorem combination_x (x : ℕ) (H : C 25 (2 * x) = C 25 (x + 4)) : x = 4 ∨ x = 7 :=
by sorry

end combination_x_l172_172006


namespace minimum_value_expression_l172_172808

theorem minimum_value_expression (x : ℝ) : (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 :=
by
  sorry

end minimum_value_expression_l172_172808


namespace original_square_side_length_l172_172493

theorem original_square_side_length :
  ∃ n k : ℕ, (n + k) * (n + k) - n * n = 47 ∧ k ≤ 5 ∧ k % 2 = 1 ∧ n = 23 :=
by
  sorry

end original_square_side_length_l172_172493


namespace point_2023_0_cannot_lie_on_line_l172_172863

-- Define real numbers a and c with the condition ac > 0
variables (a c : ℝ)

-- The condition ac > 0
def ac_positive := (a * c > 0)

-- The statement that (2023, 0) cannot be on the line y = ax + c given the condition a * c > 0
theorem point_2023_0_cannot_lie_on_line (h : ac_positive a c) : ¬ (0 = 2023 * a + c) :=
sorry

end point_2023_0_cannot_lie_on_line_l172_172863


namespace a_2023_value_l172_172546

theorem a_2023_value :
  ∀ (a : ℕ → ℚ),
  a 1 = 5 ∧
  a 2 = 5 / 11 ∧
  (∀ n, 3 ≤ n → a n = (a (n - 2)) * (a (n - 1)) / (3 * (a (n - 2)) - (a (n - 1)))) →
  a 2023 = 5 / 10114 ∧ 5 + 10114 = 10119 :=
by
  sorry

end a_2023_value_l172_172546


namespace midpoint_of_five_points_on_grid_l172_172599

theorem midpoint_of_five_points_on_grid 
    (points : Fin 5 → ℤ × ℤ) :
    ∃ i j : Fin 5, i ≠ j ∧ ((points i).fst + (points j).fst) % 2 = 0 
    ∧ ((points i).snd + (points j).snd) % 2 = 0 :=
by sorry

end midpoint_of_five_points_on_grid_l172_172599


namespace dongzhi_daylight_hours_l172_172703

theorem dongzhi_daylight_hours:
  let total_hours_in_day := 24
  let daytime_ratio := 5
  let nighttime_ratio := 7
  let total_parts := daytime_ratio + nighttime_ratio
  let daylight_hours := total_hours_in_day * daytime_ratio / total_parts
  daylight_hours = 10 :=
by
  sorry

end dongzhi_daylight_hours_l172_172703


namespace tank_fill_time_l172_172725

-- Define the conditions
def start_time : ℕ := 1 -- 1 pm
def first_hour_rainfall : ℕ := 2 -- 2 inches rainfall in the first hour from 1 pm to 2 pm
def next_four_hours_rate : ℕ := 1 -- 1 inch/hour rainfall rate from 2 pm to 6 pm
def following_rate : ℕ := 3 -- 3 inches/hour rainfall rate from 6 pm onwards
def tank_height : ℕ := 18 -- 18 inches tall fish tank

-- Define what needs to be proved
theorem tank_fill_time : 
  ∃ t : ℕ, t = 22 ∧ (tank_height ≤ (first_hour_rainfall + 4 * next_four_hours_rate + (t - 6)) + (t - 6 - 4) * following_rate) := 
by 
  sorry

end tank_fill_time_l172_172725


namespace reflection_over_line_y_eq_x_l172_172575

theorem reflection_over_line_y_eq_x {x y x' y' : ℝ} (h_c : (x, y) = (6, -5)) (h_reflect : (x', y') = (y, x)) :
  (x', y') = (-5, 6) :=
  by
    simp [h_c, h_reflect]
    sorry

end reflection_over_line_y_eq_x_l172_172575


namespace p_is_contradictory_to_q_l172_172594

variable (a : ℝ)

def p := a > 0 → a^2 ≠ 0
def q := a ≤ 0 → a^2 = 0

theorem p_is_contradictory_to_q : (p a) ↔ ¬ (q a) :=
by
  sorry

end p_is_contradictory_to_q_l172_172594


namespace margaret_score_l172_172869

theorem margaret_score (average_score marco_score margaret_score : ℝ)
  (h1: average_score = 90)
  (h2: marco_score = average_score - 0.10 * average_score)
  (h3: margaret_score = marco_score + 5) : 
  margaret_score = 86 := 
by
  sorry

end margaret_score_l172_172869


namespace opposite_of_neg_half_l172_172400

-- Define the opposite of a number
def opposite (x : ℝ) : ℝ := -x

-- The theorem we want to prove
theorem opposite_of_neg_half : opposite (-1/2) = 1/2 :=
by
  -- Proof goes here
  sorry

end opposite_of_neg_half_l172_172400


namespace abc_inequality_l172_172095

open Real

noncomputable def posReal (x : ℝ) : Prop := x > 0

theorem abc_inequality (a b c : ℝ) 
  (hCond1 : posReal a) 
  (hCond2 : posReal b) 
  (hCond3 : posReal c) 
  (hCond4 : a * b * c = 1) : 
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 :=
by
  sorry

end abc_inequality_l172_172095


namespace total_ridges_on_all_records_l172_172440

theorem total_ridges_on_all_records :
  let ridges_per_record := 60
  let cases := 4
  let shelves_per_case := 3
  let records_per_shelf := 20
  let shelf_fullness_ratio := 0.60

  let total_capacity := cases * shelves_per_case * records_per_shelf
  let actual_records := total_capacity * shelf_fullness_ratio
  let total_ridges := actual_records * ridges_per_record
  
  total_ridges = 8640 :=
by
  sorry

end total_ridges_on_all_records_l172_172440


namespace forest_area_relationship_l172_172313

variable (a b c x : ℝ)

theorem forest_area_relationship
    (hb : b = a * (1 + x))
    (hc : c = a * (1 + x) ^ 2) :
    a * c = b ^ 2 := by
  sorry

end forest_area_relationship_l172_172313


namespace circle_radius_l172_172074

/-- Consider a square ABCD with a side length of 4 cm. A circle touches the extensions 
of sides AB and AD. From point C, two tangents are drawn to this circle, 
and the angle between the tangents is 60 degrees. -/
theorem circle_radius (side_length : ℝ) (angle_between_tangents : ℝ) : 
  side_length = 4 ∧ angle_between_tangents = 60 → 
  ∃ (radius : ℝ), radius = 4 * (Real.sqrt 2 + 1) :=
by
  sorry

end circle_radius_l172_172074


namespace hair_growth_l172_172214

theorem hair_growth (initial final : ℝ) (h_init : initial = 18) (h_final : final = 24) : final - initial = 6 :=
by
  sorry

end hair_growth_l172_172214


namespace sequence_term_expression_l172_172175

theorem sequence_term_expression (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = 3^n + 1) :
  (a 1 = 4) ∧ (∀ n, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by
  sorry

end sequence_term_expression_l172_172175


namespace reflected_line_eq_l172_172243

noncomputable def point_symmetric_reflection :=
  ∃ (A : ℝ × ℝ) (B : ℝ × ℝ) (A' : ℝ × ℝ),
  A = (-1 / 2, 0) ∧ B = (0, 1) ∧ A' = (1 / 2, 0) ∧ 
  ∀ (x y : ℝ), 2 * x + y = 1 ↔
  (y - 1) / (0 - 1) = x / (1 / 2 - 0)

theorem reflected_line_eq :
  point_symmetric_reflection :=
sorry

end reflected_line_eq_l172_172243


namespace mother_used_eggs_l172_172054

variable (initial_eggs : ℕ) (eggs_after_chickens : ℕ) (chickens : ℕ) (eggs_per_chicken : ℕ) (current_eggs : ℕ)

theorem mother_used_eggs (h1 : initial_eggs = 10)
                        (h2 : chickens = 2)
                        (h3 : eggs_per_chicken = 3)
                        (h4 : current_eggs = 11)
                        (eggs_laid : ℕ)
                        (h5 : eggs_laid = chickens * eggs_per_chicken)
                        (eggs_used : ℕ)
                        (h6 : eggs_after_chickens = initial_eggs - eggs_used + eggs_laid)
                        : eggs_used = 7 :=
by
  -- proof steps go here
  sorry

end mother_used_eggs_l172_172054


namespace eq_three_div_x_one_of_eq_l172_172632

theorem eq_three_div_x_one_of_eq (x : ℝ) (hx : 1 - 6 / x + 9 / (x ^ 2) = 0) : (3 / x) = 1 :=
sorry

end eq_three_div_x_one_of_eq_l172_172632


namespace f_2021_value_l172_172129

def A : Set ℚ := {x | x ≠ -1 ∧ x ≠ 0}

def f (x : ℚ) : ℝ := sorry -- Placeholder for function definition with its properties

axiom f_property : ∀ x ∈ A, f x + f (1 + 1 / x) = 1 / 2 * Real.log (|x|)

theorem f_2021_value : f 2021 = 1 / 2 * Real.log 2021 :=
by
  sorry

end f_2021_value_l172_172129


namespace jennifer_spent_124_dollars_l172_172658

theorem jennifer_spent_124_dollars 
  (initial_cans : ℕ := 40)
  (cans_per_set : ℕ := 5)
  (additional_cans_per_set : ℕ := 6)
  (total_cans_mark : ℕ := 30)
  (price_per_can_whole : ℕ := 2)
  (discount_threshold_whole : ℕ := 10)
  (discount_amount_whole : ℕ := 4) : 
  (initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) * price_per_can_whole - 
  (discount_amount_whole * ((initial_cans + additional_cans_per_set * (total_cans_mark / cans_per_set)) / discount_threshold_whole)) = 124 := by
  sorry

end jennifer_spent_124_dollars_l172_172658


namespace recruits_line_l172_172690

theorem recruits_line
  (x y z : ℕ) 
  (hx : x + y + z + 3 = 211) 
  (hx_peter : x = 50) 
  (hy_nikolai : y = 100) 
  (hz_denis : z = 170) 
  (hxy_ratio : x = 4 * z) : 
  x + y + z + 3 = 211 :=
by
  sorry

end recruits_line_l172_172690


namespace find_number_l172_172088

variable (N : ℕ)

theorem find_number (h : 6 * ((N / 8) + 8 - 30) = 12) : N = 192 := 
by
  sorry

end find_number_l172_172088


namespace coefficient_of_x_eq_2_l172_172987

variable (a : ℝ)

theorem coefficient_of_x_eq_2 (h : (5 * (-2)) + (4 * a) = 2) : a = 3 :=
sorry

end coefficient_of_x_eq_2_l172_172987


namespace spent_on_new_tires_is_correct_l172_172327

-- Conditions
def amount_spent_on_speakers : ℝ := 136.01
def amount_spent_on_cd_player : ℝ := 139.38
def total_amount_spent : ℝ := 387.85

-- Goal
def amount_spent_on_tires : ℝ := total_amount_spent - (amount_spent_on_speakers + amount_spent_on_cd_player)

theorem spent_on_new_tires_is_correct : 
  amount_spent_on_tires = 112.46 :=
by
  sorry

end spent_on_new_tires_is_correct_l172_172327


namespace product_relationship_l172_172238

variable {a_1 a_2 b_1 b_2 : ℝ}

theorem product_relationship (h1 : a_1 < a_2) (h2 : b_1 < b_2) : 
  a_1 * b_1 + a_2 * b_2 > a_1 * b_2 + a_2 * b_1 := 
sorry

end product_relationship_l172_172238


namespace blue_apples_l172_172001

theorem blue_apples (B : ℕ) (h : (12 / 5) * B = 12) : B = 5 :=
by
  sorry

end blue_apples_l172_172001


namespace trigonometric_identity_l172_172089

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - 3 * Real.cos α) = 3 := 
by 
  sorry

end trigonometric_identity_l172_172089


namespace f_decreasing_on_neg_infty_2_l172_172718

def f (x : ℝ) := x^2 - 4 * x + 3

theorem f_decreasing_on_neg_infty_2 :
  ∀ x y : ℝ, x < y → y ≤ 2 → f y < f x :=
by
  sorry

end f_decreasing_on_neg_infty_2_l172_172718


namespace willam_land_percentage_l172_172820

-- Definitions from conditions
def farm_tax_rate : ℝ := 0.6
def total_tax_collected : ℝ := 3840
def mr_willam_tax_paid : ℝ := 500

-- Goal to prove: percentage of Mr. Willam's land over total taxable land of the village
noncomputable def percentage_mr_willam_land : ℝ :=
  (mr_willam_tax_paid / total_tax_collected) * 100

theorem willam_land_percentage :
  percentage_mr_willam_land = 13.02 := 
  by 
  sorry

end willam_land_percentage_l172_172820


namespace compute_expression_l172_172450

theorem compute_expression (y : ℕ) (h : y = 3) : 
  (y^8 + 18 * y^4 + 81) / (y^4 + 9) = 90 :=
by
  sorry

end compute_expression_l172_172450


namespace fraction_irreducible_l172_172496

theorem fraction_irreducible (n : ℤ) : gcd (2 * n ^ 2 + 9 * n - 17) (n + 6) = 1 := by
  sorry

end fraction_irreducible_l172_172496


namespace joseph_total_cost_l172_172309

variable (cost_refrigerator cost_water_heater cost_oven : ℝ)

-- Conditions
axiom h1 : cost_refrigerator = 3 * cost_water_heater
axiom h2 : cost_oven = 500
axiom h3 : cost_oven = 2 * cost_water_heater

-- Theorem
theorem joseph_total_cost : cost_refrigerator + cost_water_heater + cost_oven = 1500 := by
  sorry

end joseph_total_cost_l172_172309


namespace original_number_l172_172563

theorem original_number (x : ℝ) (h : x * 1.5 = 105) : x = 70 :=
sorry

end original_number_l172_172563


namespace leaves_blew_away_correct_l172_172876

-- Definitions based on conditions
def original_leaves : ℕ := 356
def leaves_left : ℕ := 112
def leaves_blew_away : ℕ := original_leaves - leaves_left

-- Theorem statement based on the question and correct answer
theorem leaves_blew_away_correct : leaves_blew_away = 244 := by {
  -- Proof goes here (omitted for now)
  sorry
}

end leaves_blew_away_correct_l172_172876


namespace abs_h_eq_one_l172_172816

theorem abs_h_eq_one (h : ℝ) (roots_square_sum_eq : ∀ x : ℝ, x^2 + 6 * h * x + 8 = 0 → x^2 + (x + 6 * h)^2 = 20) : |h| = 1 :=
by
  sorry

end abs_h_eq_one_l172_172816


namespace eq_solution_set_l172_172830

theorem eq_solution_set (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a^b = b^(a^a)) :
  (a, b) = (1, 1) ∨ (a, b) = (2, 16) ∨ (a, b) = (3, 27) :=
by
  sorry

end eq_solution_set_l172_172830


namespace find_t_l172_172924

theorem find_t (t : ℝ) (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) :
  (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1)^(n-1)) → -- Geometric sequence condition
  (∀ n, S_n n = 2017 * 2016^n - 2018 * t) →     -- Given sum formula
  t = 2017 / 2018 :=
by
  sorry

end find_t_l172_172924


namespace no_day_income_is_36_l172_172527

theorem no_day_income_is_36 : ∀ (n : ℕ), 3 * 3^(n-1) ≠ 36 :=
by
  intro n
  sorry

end no_day_income_is_36_l172_172527


namespace find_x_of_equation_l172_172438

-- Defining the condition and setting up the proof goal
theorem find_x_of_equation
  (h : (1/2)^25 * (1/x)^12.5 = 1/(18^25)) :
  x = 0.1577 := 
sorry

end find_x_of_equation_l172_172438


namespace sum_of_coefficients_l172_172366

theorem sum_of_coefficients (A B C : ℤ)
  (h : ∀ x, x^3 + A * x^2 + B * x + C = (x + 3) * x * (x - 3))
  : A + B + C = -9 :=
sorry

end sum_of_coefficients_l172_172366


namespace gum_cost_example_l172_172416

def final_cost (pieces : ℕ) (cost_per_piece : ℕ) (discount_percentage : ℕ) : ℕ :=
  let total_cost := pieces * cost_per_piece
  let discount := total_cost * discount_percentage / 100
  total_cost - discount

theorem gum_cost_example :
  final_cost 1500 2 10 / 100 = 27 :=
by sorry

end gum_cost_example_l172_172416


namespace find_a_l172_172159

theorem find_a (a : ℝ) (h : (2 - -3) / (1 - a) = Real.tan (135 * Real.pi / 180)) : a = 6 :=
sorry

end find_a_l172_172159


namespace remaining_shirt_cost_l172_172755

theorem remaining_shirt_cost (total_shirts : ℕ) (cost_3_shirts : ℕ) (total_cost : ℕ) 
  (h1 : total_shirts = 5) 
  (h2 : cost_3_shirts = 3 * 15) 
  (h3 : total_cost = 85) :
  (total_cost - cost_3_shirts) / (total_shirts - 3) = 20 :=
by
  sorry

end remaining_shirt_cost_l172_172755


namespace tan_neg_480_eq_sqrt_3_l172_172821

theorem tan_neg_480_eq_sqrt_3 : Real.tan (-8 * Real.pi / 3) = Real.sqrt 3 :=
by
  sorry

end tan_neg_480_eq_sqrt_3_l172_172821


namespace find_u_plus_v_l172_172220

theorem find_u_plus_v (u v : ℤ) (huv : 0 < v ∧ v < u) (h_area : u * u + 3 * u * v = 451) : u + v = 21 := 
sorry

end find_u_plus_v_l172_172220


namespace find_selling_price_functional_relationship_and_max_find_value_of_a_l172_172178

section StoreProduct

variable (x : ℕ) (y : ℕ) (a k b : ℝ)

-- Definitions for the given conditions
def cost_price : ℝ := 50
def selling_price := x 
def sales_quantity := y 
def future_cost_increase := a

-- Given points
def point1 : ℝ × ℕ := (55, 90) 
def point2 : ℝ × ℕ := (65, 70)

-- Linear relationship between selling price and sales quantity
def linearfunc := y = k * x + b

-- Proof of the first statement
theorem find_selling_price (k := -2) (b := 200) : 
    (profit = 800 → (x = 60 ∨ x = 90)) :=
by
  -- People prove the theorem here
  sorry

-- Proof for the functional relationship between W and x
theorem functional_relationship_and_max (x := 75) : 
    W = -2*x^2 + 300*x - 10000 ∧ W_max = 1250 :=
by
  -- People prove the theorem here
  sorry

-- Proof for the value of a when the cost price increases
theorem find_value_of_a (cost_increase := 4) : 
    (W'_max = 960 → a = 4) :=
by
  -- People prove the theorem here
  sorry

end StoreProduct

end find_selling_price_functional_relationship_and_max_find_value_of_a_l172_172178


namespace symmetric_line_equation_l172_172941

theorem symmetric_line_equation {l : ℝ} (h1 : ∀ x y : ℝ, x + y - 1 = 0 → (-x) - y + 1 = l) : l = 0 :=
by
  sorry

end symmetric_line_equation_l172_172941


namespace train_platform_length_l172_172835

theorem train_platform_length (time_platform : ℝ) (time_man : ℝ) (speed_km_per_hr : ℝ) :
  time_platform = 34 ∧ time_man = 20 ∧ speed_km_per_hr = 54 →
  let speed_m_per_s := speed_km_per_hr * (5/18)
  let length_train := speed_m_per_s * time_man
  let time_to_cover_platform := time_platform - time_man
  let length_platform := speed_m_per_s * time_to_cover_platform
  length_platform = 210 := 
by {
  sorry
}

end train_platform_length_l172_172835


namespace remainder_1234_mul_5678_mod_1000_l172_172842

theorem remainder_1234_mul_5678_mod_1000 :
  (1234 * 5678) % 1000 = 652 := by
  sorry

end remainder_1234_mul_5678_mod_1000_l172_172842


namespace arithmetic_mean_l172_172530

theorem arithmetic_mean (a b : ℚ) (h1 : a = 3/7) (h2 : b = 6/11) :
  (a + b) / 2 = 75 / 154 :=
by
  sorry

end arithmetic_mean_l172_172530


namespace meeting_distance_from_top_l172_172577

section

def total_distance : ℝ := 12
def uphill_distance : ℝ := 6
def downhill_distance : ℝ := 6
def john_start_time : ℝ := 0.25
def john_uphill_speed : ℝ := 12
def john_downhill_speed : ℝ := 18
def jenny_uphill_speed : ℝ := 14
def jenny_downhill_speed : ℝ := 21

theorem meeting_distance_from_top : 
  ∃ (d : ℝ), d = 6 - 14 * ((0.25) + 6 / 14 - (1 / 2) - (6 - 18 * ((1 / 2) + d / 18))) / 14 ∧ d = 45 / 32 :=
sorry

end

end meeting_distance_from_top_l172_172577


namespace bhanu_spends_on_petrol_l172_172499

-- Define the conditions as hypotheses
variable (income : ℝ)
variable (spend_on_rent : income * 0.7 * 0.14 = 98)

-- Define the theorem to prove
theorem bhanu_spends_on_petrol : (income * 0.3 = 300) :=
by
  sorry

end bhanu_spends_on_petrol_l172_172499


namespace sequence_term_formula_l172_172184

def sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = 1/2 - 1/2 * a n

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n ≥ 2, a n = r * a (n - 1)

theorem sequence_term_formula (a : ℕ → ℝ) (S : ℕ → ℝ) :
  (∀ n ≥ 1, S n = 1/2 - 1/2 * a n) →
  (S 1 = 1/2 - 1/2 * a 1) →
  a 1 = 1/3 →
  (∀ n ≥ 2, S n = 1/2 - 1/2 * (a n) → S (n - 1) = 1/2 - 1/2 * (a (n - 1)) → a n = 1/3 * a (n-1)) →
  ∀ n, a n = (1/3)^n :=
by
  intro h1 h2 h3 h4
  sorry

end sequence_term_formula_l172_172184


namespace intersection_complement_l172_172918

-- Definitions of the sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {2, 5}

-- Complement of B with respect to U
def comp_B : Set ℕ := U \ B

-- Statement to be proven
theorem intersection_complement : A ∩ comp_B = {1, 3} :=
by 
  sorry

end intersection_complement_l172_172918


namespace sum_of_extreme_values_eq_four_l172_172502

-- Given conditions in problem statement
variables (x y z : ℝ)
variables (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 8)

-- Statement to be proved: sum of smallest and largest possible values of x is 4
theorem sum_of_extreme_values_eq_four : m + M = 4 :=
sorry

end sum_of_extreme_values_eq_four_l172_172502


namespace fit_nine_cross_pentominoes_on_chessboard_l172_172431

def cross_pentomino (A B C D E : Prop) :=
  A ∧ B ∧ C ∧ D ∧ E -- A cross pentomino is five connected 1x1 squares

def square1x1 : Prop := sorry -- a placeholder for a 1x1 square

def eight_by_eight_chessboard := Fin 8 × Fin 8 -- an 8x8 chessboard using finitely indexed squares

noncomputable def can_cut_nine_cross_pentominoes : Prop := sorry -- a placeholder proof verification

theorem fit_nine_cross_pentominoes_on_chessboard : can_cut_nine_cross_pentominoes  :=
by 
  -- Assume each cross pentomino consists of 5 connected 1x1 squares
  let cross := cross_pentomino square1x1 square1x1 square1x1 square1x1 square1x1
  -- We need to prove that we can cut out nine such crosses from the 8x8 chessboard
  sorry

end fit_nine_cross_pentominoes_on_chessboard_l172_172431


namespace gcd_pow_sub_l172_172662

theorem gcd_pow_sub (h1001 h1012 : ℕ) (h : 1001 ≤ 1012) : 
  (Nat.gcd (2 ^ 1001 - 1) (2 ^ 1012 - 1)) = 2047 := sorry

end gcd_pow_sub_l172_172662


namespace hockey_league_games_l172_172310

theorem hockey_league_games (n t : ℕ) (h1 : n = 15) (h2 : t = 1050) :
  ∃ k, ∀ team1 team2 : ℕ, team1 ≠ team2 → k = 10 :=
by
  -- Declare k as the number of times each team faces the other teams
  let k := 10
  -- Verify the total number of teams and games
  have hn : n = 15 := h1
  have ht : t = 1050 := h2
  -- For any two distinct teams, they face each other k times
  use k
  intros team1 team2 hneq
  -- Show that k equals 10 under given conditions
  exact rfl

end hockey_league_games_l172_172310


namespace trapezoid_perimeter_l172_172845

noncomputable def perimeter_of_trapezoid (AB CD BC AD AP DQ : ℕ) : ℕ :=
  AB + BC + CD + AD

theorem trapezoid_perimeter (AB CD BC AP DQ : ℕ) (hBC : BC = 50) (hAP : AP = 18) (hDQ : DQ = 7) :
  perimeter_of_trapezoid AB CD BC (AP + BC + DQ) AP DQ = 180 :=
by 
  unfold perimeter_of_trapezoid
  rw [hBC, hAP, hDQ]
  -- sorry to skip the proof
  sorry

end trapezoid_perimeter_l172_172845


namespace petrol_price_l172_172173

theorem petrol_price (P : ℝ) (h : 0.9 * P = 0.9 * P) : (250 / (0.9 * P) - 250 / P = 5) → P = 5.56 :=
by
  sorry

end petrol_price_l172_172173


namespace exponent_multiplication_l172_172955

-- Define the variables and exponentiation property
variable (a : ℝ)

-- State the theorem
theorem exponent_multiplication : a^3 * a^2 = a^5 := by
  sorry

end exponent_multiplication_l172_172955


namespace speed_of_stream_l172_172947

-- Definitions based on given conditions
def speed_still_water := 24 -- km/hr
def distance_downstream := 140 -- km
def time_downstream := 5 -- hours

-- Proof problem statement
theorem speed_of_stream (v : ℕ) :
  24 + v = distance_downstream / time_downstream → v = 4 :=
by
  sorry

end speed_of_stream_l172_172947


namespace stephanie_fewer_forks_l172_172174

noncomputable def fewer_forks := 
  (60 - 44) / 4

theorem stephanie_fewer_forks : fewer_forks = 4 := by
  sorry

end stephanie_fewer_forks_l172_172174


namespace exp_gt_f_n_y_between_0_and_x_l172_172278

open Real

noncomputable def f_n (x : ℝ) (n : ℕ) : ℝ :=
  (Finset.range (n + 1)).sum (λ k => x^k / k.factorial)

theorem exp_gt_f_n (x : ℝ) (n : ℕ) (h1 : 0 < x) :
  exp x > f_n x n :=
sorry

theorem y_between_0_and_x (x : ℝ) (n : ℕ) (y : ℝ)
  (h1 : 0 < x)
  (h2 : exp x = f_n x n + x^(n+1) / (n + 1).factorial * exp y) :
  0 < y ∧ y < x :=
sorry

end exp_gt_f_n_y_between_0_and_x_l172_172278


namespace three_sum_xyz_l172_172982

theorem three_sum_xyz (x y z : ℝ) 
  (h1 : y + z = 18 - 4 * x) 
  (h2 : x + z = 22 - 4 * y) 
  (h3 : x + y = 15 - 4 * z) : 
  3 * x + 3 * y + 3 * z = 55 / 2 := 
  sorry

end three_sum_xyz_l172_172982


namespace remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l172_172689

-- Definitions from the conditions
def a : ℕ := 3^302
def b : ℕ := 3^151 + 3^101 + 1

-- Theorem: Prove that the remainder when a + 302 is divided by b is 302.
theorem remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1 :
  (a + 302) % b = 302 :=
by {
  sorry
}

end remainder_3_302_plus_302_div_by_3_151_plus_3_101_plus_1_l172_172689


namespace transformation_result_l172_172344

noncomputable def initial_function (x : ℝ) : ℝ := Real.sin (2 * x)

noncomputable def translate_left (f : ℝ → ℝ) (a : ℝ) : ℝ → ℝ :=
  λ x => f (x + a)

noncomputable def compress_horizontal (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ :=
  λ x => f (k * x)

theorem transformation_result :
  (compress_horizontal (translate_left initial_function (Real.pi / 3)) 2) x = Real.sin (4 * x + (2 * Real.pi / 3)) :=
sorry

end transformation_result_l172_172344


namespace M1M2_product_l172_172887

theorem M1M2_product :
  ∀ (M1 M2 : ℝ),
  (∀ x : ℝ, x^2 - 5 * x + 6 ≠ 0 →
    (45 * x - 55) / (x^2 - 5 * x + 6) = M1 / (x - 2) + M2 / (x - 3)) →
  (M1 + M2 = 45) →
  (3 * M1 + 2 * M2 = 55) →
  M1 * M2 = 200 :=
by
  sorry

end M1M2_product_l172_172887


namespace cups_remaining_l172_172724

-- Definitions based on problem conditions
def initial_cups : ℕ := 12
def mary_morning_cups : ℕ := 1
def mary_evening_cups : ℕ := 1
def frank_afternoon_cups : ℕ := 1
def frank_late_evening_cups : ℕ := 2 * frank_afternoon_cups

-- Hypothesis combining all conditions:
def total_given_cups : ℕ :=
  mary_morning_cups + mary_evening_cups + frank_afternoon_cups + frank_late_evening_cups

-- Theorem to prove
theorem cups_remaining : initial_cups - total_given_cups = 7 :=
  sorry

end cups_remaining_l172_172724


namespace train_length_l172_172329

/-- 
Given that a train can cross an electric pole in 200 seconds and its speed is 18 km/h,
prove that the length of the train is 1000 meters.
-/
theorem train_length
  (time_to_cross : ℕ)
  (speed_kmph : ℕ)
  (h_time : time_to_cross = 200)
  (h_speed : speed_kmph = 18)
  : (speed_kmph * 1000 / 3600 * time_to_cross = 1000) :=
by
  sorry

end train_length_l172_172329


namespace swimming_speed_in_still_water_l172_172380

theorem swimming_speed_in_still_water :
  ∀ (speed_of_water person's_speed time distance: ℝ),
  speed_of_water = 8 →
  time = 1.5 →
  distance = 12 →
  person's_speed - speed_of_water = distance / time →
  person's_speed = 16 :=
by
  intro speed_of_water person's_speed time distance hw ht hd heff
  rw [hw, ht, hd] at heff
  -- steps to isolate person's_speed should be done here, but we leave it as sorry
  sorry

end swimming_speed_in_still_water_l172_172380


namespace a5_value_l172_172850

variable {a : ℕ → ℝ} (q : ℝ) (a2 a3 : ℝ)

-- Assume the conditions: geometric sequence, a_2 = 2, a_3 = -4
def is_geometric_sequence (a : ℕ → ℝ) : Prop := ∃ q, ∀ n, a (n + 1) = a n * q

-- Given conditions
axiom h1 : is_geometric_sequence a
axiom h2 : a 2 = 2
axiom h3 : a 3 = -4

-- Theorem to prove
theorem a5_value : a 5 = -16 :=
by
  -- Here you would provide the proof based on the conditions
  sorry

end a5_value_l172_172850


namespace jonathan_typing_time_l172_172315

theorem jonathan_typing_time 
(J : ℕ) 
(h_combined_rate : (1 / (J : ℝ)) + (1 / 30) + (1 / 24) = 1 / 10) : 
  J = 40 :=
by {
  sorry
}

end jonathan_typing_time_l172_172315


namespace N_divisible_by_9_l172_172125

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem N_divisible_by_9 (N : ℕ) (h : sum_of_digits N = sum_of_digits (5 * N)) : N % 9 = 0 := 
sorry

end N_divisible_by_9_l172_172125


namespace pencils_combined_length_l172_172668

theorem pencils_combined_length (length_pencil1 length_pencil2 : Nat) (h1 : length_pencil1 = 12) (h2 : length_pencil2 = 12) :
  length_pencil1 + length_pencil2 = 24 := by
  sorry

end pencils_combined_length_l172_172668


namespace solve_quadratic_l172_172323

theorem solve_quadratic (x : ℝ) (h1 : 2 * x^2 - 6 * x = 0) (h2 : x ≠ 0) : x = 3 := by
  sorry

end solve_quadratic_l172_172323


namespace cannot_achieve_80_cents_l172_172196

def is_possible_value (n : ℕ) : Prop :=
  ∃ (n_nickels n_dimes n_quarters n_half_dollars : ℕ), 
    n_nickels + n_dimes + n_quarters + n_half_dollars = 5 ∧
    5 * n_nickels + 10 * n_dimes + 25 * n_quarters + 50 * n_half_dollars = n

theorem cannot_achieve_80_cents : ¬ is_possible_value 80 :=
by sorry

end cannot_achieve_80_cents_l172_172196


namespace least_k_divisible_480_l172_172618

theorem least_k_divisible_480 (k : ℕ) (h : k^4 % 480 = 0) : k = 101250 :=
sorry

end least_k_divisible_480_l172_172618


namespace sum_of_consecutive_page_numbers_l172_172005

def consecutive_page_numbers_product_and_sum (n m : ℤ) :=
  n * m = 20412

theorem sum_of_consecutive_page_numbers (n : ℤ) (h1 : consecutive_page_numbers_product_and_sum n (n + 1)) : n + (n + 1) = 285 :=
by
  sorry

end sum_of_consecutive_page_numbers_l172_172005


namespace average_viewing_times_correct_l172_172099

-- Define the viewing times for each family member per week
def Evelyn_week1 : ℕ := 10
def Evelyn_week2 : ℕ := 8
def Evelyn_week3 : ℕ := 6

def Eric_week1 : ℕ := 8
def Eric_week2 : ℕ := 6
def Eric_week3 : ℕ := 5

def Kate_week2_episodes : ℕ := 12
def minutes_per_episode : ℕ := 40
def Kate_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def Kate_week3 : ℕ := 4

def John_week2 : ℕ := (Kate_week2_episodes * minutes_per_episode) / 60
def John_week3 : ℕ := 8

-- Calculate the averages
def average (total : ℚ) (weeks : ℚ) : ℚ := total / weeks

-- Define the total viewing time for each family member
def Evelyn_total : ℕ := Evelyn_week1 + Evelyn_week2 + Evelyn_week3
def Eric_total : ℕ := Eric_week1 + Eric_week2 + Eric_week3
def Kate_total : ℕ := 0 + Kate_week2 + Kate_week3
def John_total : ℕ := 0 + John_week2 + John_week3

-- Define the expected averages
def Evelyn_expected_avg : ℚ := 8
def Eric_expected_avg : ℚ := 19 / 3
def Kate_expected_avg : ℚ := 4
def John_expected_avg : ℚ := 16 / 3

-- The theorem to prove that the calculated averages are correct
theorem average_viewing_times_correct :
  average Evelyn_total 3 = Evelyn_expected_avg ∧
  average Eric_total 3 = Eric_expected_avg ∧
  average Kate_total 3 = Kate_expected_avg ∧
  average John_total 3 = John_expected_avg :=
by sorry

end average_viewing_times_correct_l172_172099


namespace time_to_shovel_snow_l172_172483

noncomputable def initial_rate : ℕ := 30
noncomputable def decay_rate : ℕ := 2
noncomputable def driveway_width : ℕ := 6
noncomputable def driveway_length : ℕ := 15
noncomputable def snow_depth : ℕ := 2

noncomputable def total_snow_volume : ℕ := driveway_width * driveway_length * snow_depth

def snow_shoveling_time (initial_rate decay_rate total_volume : ℕ) : ℕ :=
-- Function to compute the time needed, assuming definition provided
sorry

theorem time_to_shovel_snow 
  : snow_shoveling_time initial_rate decay_rate total_snow_volume = 8 :=
sorry

end time_to_shovel_snow_l172_172483


namespace parabola_focus_directrix_eq_l172_172318

open Real

def distance (p : ℝ × ℝ) (l : ℝ) : ℝ := abs (p.fst - l)

def parabola_eq (focus_x focus_y l : ℝ) : Prop :=
  ∀ x y, (distance (x, y) focus_x = distance (x, y) l) ↔ y^2 = 2 * x - 1

theorem parabola_focus_directrix_eq :
  parabola_eq 1 0 0 :=
by
  sorry

end parabola_focus_directrix_eq_l172_172318


namespace minimum_value_expression_l172_172760

theorem minimum_value_expression (x : ℝ) (h : x > 4) : 
  ∃ (m : ℝ), m = 6 ∧ ∀ y : ℝ, y = (x + 5) / (Real.sqrt (x - 4)) → y ≥ m :=
by
  -- proof goes here
  sorry

end minimum_value_expression_l172_172760


namespace problem_solution_l172_172853

theorem problem_solution
  (a b c d : ℝ)
  (h : a^2 + b^2 + c^2 + d^2 = 4) :
  (a + 2) * (b + 2) ≥ c * d :=
sorry

end problem_solution_l172_172853


namespace largest_triangle_perimeter_l172_172392

theorem largest_triangle_perimeter :
  ∀ (x : ℕ), 1 < x ∧ x < 15 → (7 + 8 + x = 29) :=
by
  intro x
  intro h
  sorry

end largest_triangle_perimeter_l172_172392


namespace total_pebbles_count_l172_172287

def white_pebbles : ℕ := 20
def red_pebbles : ℕ := white_pebbles / 2
def blue_pebbles : ℕ := red_pebbles / 3
def green_pebbles : ℕ := blue_pebbles + 5

theorem total_pebbles_count : white_pebbles + red_pebbles + blue_pebbles + green_pebbles = 41 := by
  sorry

end total_pebbles_count_l172_172287


namespace find_k_l172_172340

theorem find_k (x k : ℝ) :
  (∀ x, x ∈ Set.Ioo (-4 : ℝ) 3 ↔ x * (x^2 - 9) < k) → k = 0 :=
  by
  sorry

end find_k_l172_172340


namespace drawing_at_least_one_red_is_certain_l172_172033

-- Defining the balls and box conditions
structure Box :=
  (red_balls : ℕ) 
  (yellow_balls : ℕ) 

-- Let the box be defined as having 3 red balls and 2 yellow balls
def box : Box := { red_balls := 3, yellow_balls := 2 }

-- Define the event of drawing at least one red ball
def at_least_one_red (draws : ℕ) (b : Box) : Prop :=
  ∀ drawn_yellow, drawn_yellow < draws → drawn_yellow < b.yellow_balls

-- The conclusion we want to prove
theorem drawing_at_least_one_red_is_certain : at_least_one_red 3 box :=
by 
  sorry

end drawing_at_least_one_red_is_certain_l172_172033


namespace trim_hedges_purpose_l172_172582

-- Given possible answers
inductive Answer
| A : Answer
| B : Answer
| C : Answer
| D : Answer

-- Define the purpose of trimming hedges
def trimmingHedges : Answer :=
  Answer.B

-- Formal problem statement
theorem trim_hedges_purpose : trimmingHedges = Answer.B :=
  sorry

end trim_hedges_purpose_l172_172582


namespace simplified_expression_l172_172368

variable (x y : ℝ)

theorem simplified_expression (hx : x ≠ 0) (hy : y ≠ 0) :
  (3 / 5) * Real.sqrt (x * y^2) / ((-4 / 15) * Real.sqrt (y / x)) * ((-5 / 6) * Real.sqrt (x^3 * y)) =
  (15 * x^2 * y * Real.sqrt x) / 8 :=
by
  sorry

end simplified_expression_l172_172368


namespace profit_percentage_is_correct_l172_172069

-- Define the conditions
variables (market_price_per_pen : ℝ) (discount_percentage : ℝ) (total_pens_bought : ℝ) (cost_pens_market_price : ℝ)
variables (cost_price_per_pen : ℝ) (selling_price_per_pen : ℝ) (profit_per_pen : ℝ) (profit_percent : ℝ)

-- Conditions
def condition_1 : market_price_per_pen = 1 := by sorry
def condition_2 : discount_percentage = 0.01 := by sorry
def condition_3 : total_pens_bought = 80 := by sorry
def condition_4 : cost_pens_market_price = 36 := by sorry

-- Definitions based on conditions
def cost_price_per_pen_def : cost_price_per_pen = cost_pens_market_price / total_pens_bought := by sorry
def selling_price_per_pen_def : selling_price_per_pen = market_price_per_pen * (1 - discount_percentage) := by sorry
def profit_per_pen_def : profit_per_pen = selling_price_per_pen - cost_price_per_pen := by sorry
def profit_percent_def : profit_percent = (profit_per_pen / cost_price_per_pen) * 100 := by sorry

-- The statement to prove
theorem profit_percentage_is_correct : profit_percent = 120 :=
by
  have h1 : cost_price_per_pen = 36 / 80 := by sorry
  have h2 : selling_price_per_pen = 1 * (1 - 0.01) := by sorry
  have h3 : profit_per_pen = 0.99 - 0.45 := by sorry
  have h4 : profit_percent = (0.54 / 0.45) * 100 := by sorry
  sorry

end profit_percentage_is_correct_l172_172069


namespace find_k_l172_172136

theorem find_k (k : ℝ) (h : (3, 1) ∈ {(x, y) | y = k * x - 2} ∧ k ≠ 0) : k = 1 :=
by sorry

end find_k_l172_172136


namespace find_sum_invested_l172_172194

theorem find_sum_invested (P : ℝ) 
  (SI_1: ℝ) (SI_2: ℝ)
  (h1 : SI_1 = P * (15 / 100) * 2)
  (h2 : SI_2 = P * (12 / 100) * 2)
  (h3 : SI_1 - SI_2 = 900) :
  P = 15000 := by
sorry

end find_sum_invested_l172_172194


namespace range_of_a_l172_172893

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - 2 * x

theorem range_of_a 
  (a : ℝ) 
  (h : ∀ x : ℝ, 1 < x → 2 * a * Real.log x ≤ 2 * x^2 + f a (2 * x - 1)) :
  a ≤ 2 :=
sorry

end range_of_a_l172_172893


namespace y_squared_range_l172_172741

theorem y_squared_range (y : ℝ) 
  (h : Real.sqrt (Real.sqrt (y + 16)) - Real.sqrt (Real.sqrt (y - 16)) = 2) : 
  9200 ≤ y^2 ∧ y^2 ≤ 9400 := 
sorry

end y_squared_range_l172_172741


namespace second_pipe_fill_time_l172_172680

theorem second_pipe_fill_time :
  ∃ x : ℝ, x ≠ 0 ∧ (1 / 10 + 1 / x - 1 / 20 = 1 / 7.5) ∧ x = 60 :=
by
  sorry

end second_pipe_fill_time_l172_172680


namespace sum_of_areas_lt_side_length_square_l172_172213

variable (n : ℕ) (a : ℝ)
variable (S : Fin n → ℝ) (d : Fin n → ℝ)

-- Conditions
axiom areas_le_one : ∀ i, S i ≤ 1
axiom sum_d_le_a : (Finset.univ).sum d ≤ a
axiom areas_less_than_diameters : ∀ i, S i < d i

-- Theorem Statement
theorem sum_of_areas_lt_side_length_square :
  ((Finset.univ : Finset (Fin n)).sum S) < a :=
sorry

end sum_of_areas_lt_side_length_square_l172_172213


namespace rhombus_longer_diagonal_l172_172057

theorem rhombus_longer_diagonal
  (a b : ℝ)
  (ha : a = 65) -- Side length
  (hb : b = 28) -- Half the length of the shorter diagonal
  : 2 * Real.sqrt (a^2 - b^2) = 118 :=  -- Statement of the longer diagonal length
by
  sorry -- Proof goes here

end rhombus_longer_diagonal_l172_172057


namespace sqrt_expr_is_599_l172_172062

theorem sqrt_expr_is_599 : Real.sqrt ((26 * 25 * 24 * 23) + 1) = 599 := by
  sorry

end sqrt_expr_is_599_l172_172062


namespace sum_of_properly_paintable_numbers_l172_172300

-- Definitions based on conditions
def properly_paintable (a b c : ℕ) : Prop :=
  ∀ n : ℕ, (n % a = 0 ∧ n % b ≠ 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b = 1 ∧ n % c ≠ 3) ∨
           (n % a ≠ 0 ∧ n % b ≠ 1 ∧ n % c = 3) → n < 100

-- Main theorem to prove
theorem sum_of_properly_paintable_numbers : 
  (properly_paintable 3 3 6) ∧ (properly_paintable 4 2 8) → 
  100 * 3 + 10 * 3 + 6 + 100 * 4 + 10 * 2 + 8 = 764 :=
by
  sorry  -- The proof goes here, but it's not required

-- Note: The actual condition checks in the definition of properly_paintable 
-- might need more detailed splits into depending on specific post visits and a 
-- more rigorous formalization to comply with the exact checking as done above. 
-- This definition is a simplified logical structure to represent the condition.


end sum_of_properly_paintable_numbers_l172_172300


namespace daily_evaporation_l172_172066

variable (initial_water : ℝ) (percentage_evaporated : ℝ) (days : ℕ)
variable (evaporation_amount : ℝ)

-- Given conditions
def conditions_met : Prop :=
  initial_water = 10 ∧ percentage_evaporated = 0.4 ∧ days = 50

-- Question: Prove the amount of water evaporated each day is 0.08
theorem daily_evaporation (h : conditions_met initial_water percentage_evaporated days) :
  evaporation_amount = (initial_water * percentage_evaporated) / days :=
sorry

end daily_evaporation_l172_172066


namespace cells_at_end_of_8th_day_l172_172367

theorem cells_at_end_of_8th_day :
  let initial_cells := 5
  let factor := 3
  let toxin_factor := 1 / 2
  let cells_after_toxin := (initial_cells * factor * factor * factor * toxin_factor : ℤ)
  let final_cells := cells_after_toxin * factor 
  final_cells = 201 :=
by
  sorry

end cells_at_end_of_8th_day_l172_172367


namespace find_a_l172_172494

-- Define sets A and B
def A : Set ℕ := {1, 2, 5}
def B (a : ℕ) : Set ℕ := {2, a}

-- Given condition: A ∪ B = {1, 2, 3, 5}
def union_condition (a : ℕ) : Prop := A ∪ B a = {1, 2, 3, 5}

-- Theorem we want to prove
theorem find_a (a : ℕ) : union_condition a → a = 3 :=
by
  intro h
  sorry

end find_a_l172_172494


namespace expectation_of_X_l172_172092

-- Conditions:
-- Defect rate of the batch of products is 0.05
def defect_rate : ℚ := 0.05

-- 5 items are randomly selected for quality inspection
def n : ℕ := 5

-- The probability of obtaining a qualified product in each trial
def P : ℚ := 1 - defect_rate

-- Question:
-- The random variable X, representing the number of qualified products, follows a binomial distribution.
-- Expectation of X
def expectation_X : ℚ := n * P

-- Prove that the mathematical expectation E(X) is equal to 4.75
theorem expectation_of_X :
  expectation_X = 4.75 := 
sorry

end expectation_of_X_l172_172092


namespace not_necessarily_divisible_by_66_l172_172022

open Nat

-- Definition of what it means to be the product of four consecutive integers
def product_of_four_consecutive_integers (n : ℕ) : Prop :=
  ∃ k : ℤ, n = (k * (k + 1) * (k + 2) * (k + 3))

-- Lean theorem statement for the proof problem
theorem not_necessarily_divisible_by_66 (n : ℕ) 
  (h1 : product_of_four_consecutive_integers n) 
  (h2 : 11 ∣ n) : ¬ (66 ∣ n) :=
sorry

end not_necessarily_divisible_by_66_l172_172022


namespace solve_for_k_l172_172382

def f (n : ℤ) : ℤ :=
  if n % 2 = 1 then n + 5 else n / 2

theorem solve_for_k (k : ℤ) (h1 : k % 2 = 1) (h2 : f (f (f k)) = 57) : k = 223 :=
by
  -- Proof will be provided here
  sorry

end solve_for_k_l172_172382


namespace min_value_2x_y_l172_172666

noncomputable def min_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (heq : Real.log (x + 2 * y) = Real.log x + Real.log y) : ℝ :=
  2 * x + y

theorem min_value_2x_y : ∀ (x y : ℝ), 0 < x → 0 < y → Real.log (x + 2 * y) = Real.log x + Real.log y → 2 * x + y ≥ 9 :=
by
  intros x y hx hy heq
  sorry

end min_value_2x_y_l172_172666


namespace sum_of_two_numbers_is_10_l172_172337

variable (a b : ℝ)

theorem sum_of_two_numbers_is_10
  (h1 : a + b = 10)
  (h2 : a - b = 8)
  (h3 : a^2 - b^2 = 80) :
  a + b = 10 :=
by
  sorry

end sum_of_two_numbers_is_10_l172_172337


namespace union_set_solution_l172_172946

theorem union_set_solution (M N : Set ℝ) 
    (hM : M = { x | 0 ≤ x ∧ x ≤ 3 }) 
    (hN : N = { x | x < 1 }) : 
    M ∪ N = { x | x ≤ 3 } := 
by 
    sorry

end union_set_solution_l172_172946


namespace abs_a_eq_5_and_a_add_b_eq_0_l172_172397

theorem abs_a_eq_5_and_a_add_b_eq_0 (a b : ℤ) (h1 : |a| = 5) (h2 : a + b = 0) :
  a - b = 10 ∨ a - b = -10 :=
by
  sorry

end abs_a_eq_5_and_a_add_b_eq_0_l172_172397


namespace gcd_eq_gcd_of_eq_add_mul_l172_172939

theorem gcd_eq_gcd_of_eq_add_mul (a b q r : Int) (h_q : b > 0) (h_r : 0 ≤ r) (h_ar : a = b * q + r) : Int.gcd a b = Int.gcd b r :=
by
  -- Conditions: constraints and assertion
  exact sorry

end gcd_eq_gcd_of_eq_add_mul_l172_172939


namespace integer_not_always_greater_decimal_l172_172580

-- Definitions based on conditions
def is_decimal (d : ℚ) : Prop :=
  ∃ (i : ℤ) (f : ℚ), 0 ≤ f ∧ f < 1 ∧ d = i + f

def is_greater (a : ℤ) (b : ℚ) : Prop :=
  (a : ℚ) > b

theorem integer_not_always_greater_decimal : ¬ ∀ n : ℤ, ∀ d : ℚ, is_decimal d → (is_greater n d) :=
by
  sorry

end integer_not_always_greater_decimal_l172_172580


namespace graph_quadrant_exclusion_l172_172471

theorem graph_quadrant_exclusion (a b : ℝ) (h1 : 0 < a) (h2 : a < 1) (h3 : b < -1) :
  ∀ x : ℝ, ¬ ((a^x + b > 0) ∧ (x > 0)) :=
by
  sorry

end graph_quadrant_exclusion_l172_172471


namespace replace_movie_cost_l172_172370

def num_popular_action_movies := 20
def num_moderate_comedy_movies := 30
def num_unpopular_drama_movies := 10
def num_popular_comedy_movies := 15
def num_moderate_action_movies := 25

def trade_in_rate_action := 3
def trade_in_rate_comedy := 2
def trade_in_rate_drama := 1

def dvd_cost_popular := 12
def dvd_cost_moderate := 8
def dvd_cost_unpopular := 5

def johns_movie_cost : Nat :=
  let total_trade_in := 
    (num_popular_action_movies + num_moderate_action_movies) * trade_in_rate_action +
    (num_moderate_comedy_movies + num_popular_comedy_movies) * trade_in_rate_comedy +
    num_unpopular_drama_movies * trade_in_rate_drama
  let total_dvd_cost :=
    (num_popular_action_movies + num_popular_comedy_movies) * dvd_cost_popular +
    (num_moderate_comedy_movies + num_moderate_action_movies) * dvd_cost_moderate +
    num_unpopular_drama_movies * dvd_cost_unpopular
  total_dvd_cost - total_trade_in

theorem replace_movie_cost : johns_movie_cost = 675 := 
by
  sorry

end replace_movie_cost_l172_172370


namespace smallest_positive_integer_x_l172_172678

-- Definitions based on the conditions given
def is_multiple (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement of the problem
theorem smallest_positive_integer_x (x : ℕ) :
  (is_multiple (900 * x) 640) → x = 32 :=
sorry

end smallest_positive_integer_x_l172_172678


namespace volume_in_barrel_l172_172180

def is_divisible (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

theorem volume_in_barrel (x : ℕ) (V : ℕ) (hx : V = 30) 
  (h1 : V = x / 2 + x / 3 + x / 4 + x / 5 + x / 6) 
  (h2 : is_divisible (87 * x) 60) : 
  V = 29 := 
sorry

end volume_in_barrel_l172_172180


namespace expression_equals_one_l172_172564

variable {R : Type*} [Field R]
variables (x y z : R)

theorem expression_equals_one (h₁ : x ≠ y) (h₂ : x ≠ z) (h₃ : y ≠ z) :
    (x^2 / ((x - y) * (x - z)) + y^2 / ((y - x) * (y - z)) + z^2 / ((z - x) * (z - y))) = 1 :=
by sorry

end expression_equals_one_l172_172564


namespace inequality_abc_lt_l172_172150

variable (a b c : ℝ)

theorem inequality_abc_lt:
  c > b → b > a → a^2 * b + b^2 * c + c^2 * a < a * b^2 + b * c^2 + c * a^2 :=
by
  intros h1 h2
  sorry

end inequality_abc_lt_l172_172150


namespace op_value_l172_172519

noncomputable def op (a b c : ℝ) (k : ℤ) : ℝ :=
  b^2 - k * a^2 * c

theorem op_value : op 2 5 3 3 = -11 := by
  sorry

end op_value_l172_172519


namespace find_solutions_l172_172245

theorem find_solutions :
  ∀ x y : Real, 
  (3 / 20) + abs (x - (15 / 40)) < (7 / 20) →
  y = 2 * x + 1 →
  (7 / 20) < x ∧ x < (2 / 5) ∧ (17 / 10) ≤ y ∧ y ≤ (11 / 5) :=
by
  intros x y h₁ h₂
  sorry

end find_solutions_l172_172245


namespace directrix_of_parabola_l172_172048

theorem directrix_of_parabola (a b c : ℝ) (parabola_eqn : ∀ x : ℝ, y = 3 * x^2 - 6 * x + 2)
  (vertex : ∃ h k : ℝ, h = 1 ∧ k = -1)
  : ∃ y : ℝ, y = -13 / 12 := 
sorry

end directrix_of_parabola_l172_172048


namespace train_speed_ratio_l172_172068

variable (V1 V2 : ℝ)

theorem train_speed_ratio (H1 : V1 * 4 = D1) (H2 : V2 * 36 = D2) (H3 : D1 / D2 = 1 / 9) :
  V1 / V2 = 1 := 
by
  sorry

end train_speed_ratio_l172_172068


namespace find_larger_number_l172_172867

theorem find_larger_number (S L : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 6 * S + 15) : 
  L = 1635 := 
sorry

end find_larger_number_l172_172867


namespace bus_avg_speed_l172_172692

noncomputable def average_speed_of_bus 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) :
  ℕ :=
  (initial_distance_behind + bicycle_speed * catch_up_time) / catch_up_time

theorem bus_avg_speed 
  (bicycle_speed : ℕ) 
  (initial_distance_behind : ℕ) 
  (catch_up_time : ℕ) 
  (h_bicycle_speed : bicycle_speed = 15) 
  (h_initial_distance_behind : initial_distance_behind = 195)
  (h_catch_up_time : catch_up_time = 3) :
  average_speed_of_bus bicycle_speed initial_distance_behind catch_up_time = 80 :=
by
  sorry

end bus_avg_speed_l172_172692


namespace exponent_zero_nonneg_l172_172424

theorem exponent_zero_nonneg (a : ℝ) (h : a ≠ -1) : (a + 1) ^ 0 = 1 :=
sorry

end exponent_zero_nonneg_l172_172424


namespace necessary_but_not_sufficient_l172_172195

theorem necessary_but_not_sufficient (a b : ℝ) : 
  (a > b - 1) ∧ ¬(a > b - 1 → a > b) :=
sorry

end necessary_but_not_sufficient_l172_172195


namespace river_flow_volume_l172_172552

noncomputable def river_depth : ℝ := 2
noncomputable def river_width : ℝ := 45
noncomputable def flow_rate_kmph : ℝ := 4
noncomputable def flow_rate_mpm := flow_rate_kmph * 1000 / 60
noncomputable def cross_sectional_area := river_depth * river_width
noncomputable def volume_per_minute := cross_sectional_area * flow_rate_mpm

theorem river_flow_volume :
  volume_per_minute = 6000.3 := by
  sorry

end river_flow_volume_l172_172552


namespace ratio_of_slices_l172_172560

theorem ratio_of_slices
  (initial_slices : ℕ)
  (slices_eaten_for_lunch : ℕ)
  (remaining_slices_after_lunch : ℕ)
  (slices_left_for_tomorrow : ℕ)
  (slices_eaten_for_dinner : ℕ)
  (ratio : ℚ) :
  initial_slices = 12 → 
  slices_eaten_for_lunch = initial_slices / 2 →
  remaining_slices_after_lunch = initial_slices - slices_eaten_for_lunch →
  slices_left_for_tomorrow = 4 →
  slices_eaten_for_dinner = remaining_slices_after_lunch - slices_left_for_tomorrow →
  ratio = (slices_eaten_for_dinner : ℚ) / remaining_slices_after_lunch →
  ratio = 1 / 3 :=
by sorry

end ratio_of_slices_l172_172560


namespace product_four_integers_sum_to_50_l172_172466

theorem product_four_integers_sum_to_50 (E F G H : ℝ) 
  (h₀ : E + F + G + H = 50)
  (h₁ : E - 3 = F + 3)
  (h₂ : E - 3 = G * 3)
  (h₃ : E - 3 = H / 3) :
  E * F * G * H = 7461.9140625 := 
sorry

end product_four_integers_sum_to_50_l172_172466


namespace parallel_vectors_implies_m_eq_neg1_l172_172677

theorem parallel_vectors_implies_m_eq_neg1 (m : ℝ) :
  let a := (m, -1)
  let b := (1, m + 2)
  a.1 * b.2 = a.2 * b.1 → m = -1 :=
by
  intro h
  sorry

end parallel_vectors_implies_m_eq_neg1_l172_172677


namespace eval_expression_l172_172357

theorem eval_expression (x y z : ℝ) (hx : x = 1/3) (hy : y = 2/3) (hz : z = -9) :
  x^2 * y^3 * z = -8/27 :=
by
  subst hx
  subst hy
  subst hz
  sorry

end eval_expression_l172_172357


namespace mos_to_ory_bus_encounter_l172_172497

def encounter_buses (departure_time : Nat) (encounter_bus_time : Nat) (travel_time : Nat) : Nat := sorry

theorem mos_to_ory_bus_encounter :
  encounter_buses 0 30 5 = 10 :=
sorry

end mos_to_ory_bus_encounter_l172_172497


namespace geometric_sequence_product_l172_172065

theorem geometric_sequence_product :
  ∀ (a : ℕ → ℝ), (∀ n, a n > 0) →
  (∃ (a_1 a_99 : ℝ), (a_1 + a_99 = 10) ∧ (a_1 * a_99 = 16) ∧ a 1 = a_1 ∧ a 99 = a_99) →
  a 20 * a 50 * a 80 = 64 :=
by
  intro a hpos hex
  sorry

end geometric_sequence_product_l172_172065


namespace term_five_eq_nine_l172_172861

variable (S : ℕ → ℕ) (a : ℕ → ℕ)

-- The sum of the first n terms of the sequence equals n^2.
axiom sum_formula : ∀ n, S n = n^2

-- Definition of the nth term in terms of the sequence sum.
def a_n (n : ℕ) : ℕ := S n - S (n - 1)

-- Goal: Prove that the 5th term, a(5), equals 9.
theorem term_five_eq_nine : a_n S 5 = 9 :=
by
  sorry

end term_five_eq_nine_l172_172861


namespace census_entirety_is_population_l172_172153

-- Define the options as a type
inductive CensusOptions
| Part
| Whole
| Individual
| Population

-- Define the condition: the entire object under investigation in a census
def entirety_of_objects_under_investigation : CensusOptions := CensusOptions.Population

-- Prove that the entirety of objects under investigation in a census is called Population
theorem census_entirety_is_population :
  entirety_of_objects_under_investigation = CensusOptions.Population :=
sorry

end census_entirety_is_population_l172_172153


namespace factor_difference_of_squares_l172_172365

theorem factor_difference_of_squares (t : ℝ) : t^2 - 64 = (t - 8) * (t + 8) := by
  sorry

end factor_difference_of_squares_l172_172365


namespace exponentiation_rule_l172_172349

theorem exponentiation_rule (a : ℝ) : (a^4) * (a^4) = a^8 :=
by 
  sorry

end exponentiation_rule_l172_172349


namespace factorization_correct_l172_172304

def expression (x : ℝ) : ℝ := 16 * x^3 + 4 * x^2
def factored_expression (x : ℝ) : ℝ := 4 * x^2 * (4 * x + 1)

theorem factorization_correct (x : ℝ) : expression x = factored_expression x := 
by 
  sorry

end factorization_correct_l172_172304


namespace max_tied_teams_for_most_wins_l172_172473

theorem max_tied_teams_for_most_wins 
  (n : ℕ) 
  (h₀ : n = 6)
  (total_games : ℕ := n * (n - 1) / 2)
  (game_result : Π (i j : ℕ), i ≠ j → (0 = 1 → false) ∨ (1 = 1))
  (rank_by_wins : ℕ → ℕ) : true := sorry

end max_tied_teams_for_most_wins_l172_172473


namespace polygon_perimeter_l172_172360

theorem polygon_perimeter (side_length : ℝ) (ext_angle_deg : ℝ) (n : ℕ) (h1 : side_length = 8) 
  (h2 : ext_angle_deg = 90) (h3 : ext_angle_deg = 360 / n) : 
  4 * side_length = 32 := 
  by 
    sorry

end polygon_perimeter_l172_172360


namespace apartments_per_floor_l172_172957

theorem apartments_per_floor (floors apartments_per: ℕ) (total_people : ℕ) (each_apartment_houses : ℕ)
    (h1 : floors = 25)
    (h2 : each_apartment_houses = 2)
    (h3 : total_people = 200)
    (h4 : floors * apartments_per * each_apartment_houses = total_people) :
    apartments_per = 4 := 
sorry

end apartments_per_floor_l172_172957


namespace area_of_ring_between_outermost_and_middle_circle_l172_172010

noncomputable def pi : ℝ := Real.pi

theorem area_of_ring_between_outermost_and_middle_circle :
  let r_outermost := 12
  let r_middle := 8
  let A_outermost := pi * r_outermost^2
  let A_middle := pi * r_middle^2
  A_outermost - A_middle = 80 * pi :=
by 
  sorry

end area_of_ring_between_outermost_and_middle_circle_l172_172010


namespace time_to_cross_first_platform_l172_172906

-- Define the given conditions
def length_first_platform : ℕ := 140
def length_second_platform : ℕ := 250
def length_train : ℕ := 190
def time_cross_second_platform : Nat := 20
def speed := (length_train + length_second_platform) / time_cross_second_platform

-- The theorem to be proved
theorem time_to_cross_first_platform : 
  (length_train + length_first_platform) / speed = 15 :=
sorry

end time_to_cross_first_platform_l172_172906


namespace ab_bc_cd_da_le_four_l172_172434

theorem ab_bc_cd_da_le_four (a b c d : ℝ) (hpos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d) (hsum : a + b + c + d = 4) :
  a * b + b * c + c * d + d * a ≤ 4 :=
by
  sorry

end ab_bc_cd_da_le_four_l172_172434


namespace geometric_sequence_third_term_l172_172557

theorem geometric_sequence_third_term :
  ∀ (a r : ℕ), a = 2 ∧ a * r ^ 3 = 162 → a * r ^ 2 = 18 :=
by
  intros a r
  intro h
  have ha : a = 2 := h.1
  have h_fourth_term : a * r ^ 3 = 162 := h.2
  sorry

end geometric_sequence_third_term_l172_172557


namespace problem_statement_l172_172437

-- Definitions of conditions
def p (a : ℝ) : Prop := a < 0
def q (a : ℝ) : Prop := a^2 > a

-- Statement of the problem
theorem problem_statement (a : ℝ) (h1 : p a) (h2 : q a) : (¬ p a) → (¬ q a) → ∃ x, ¬ (¬ q x) → (¬ (¬ p x)) :=
by
  sorry

end problem_statement_l172_172437


namespace time_to_run_100_meters_no_wind_l172_172338

-- Definitions based on the conditions
variables (v w : ℝ)
axiom speed_with_wind : v + w = 9
axiom speed_against_wind : v - w = 7

-- The theorem statement to prove
theorem time_to_run_100_meters_no_wind : (100 / v) = 12.5 :=
by 
  sorry

end time_to_run_100_meters_no_wind_l172_172338


namespace probability_of_sum_16_with_duplicates_l172_172895

namespace DiceProbability

def is_valid_die_roll (n : ℕ) : Prop :=
  n ≥ 1 ∧ n ≤ 6

def is_valid_combination (x y z : ℕ) : Prop :=
  x + y + z = 16 ∧ 
  is_valid_die_roll x ∧ 
  is_valid_die_roll y ∧ 
  is_valid_die_roll z ∧ 
  (x = y ∨ y = z ∨ z = x)

theorem probability_of_sum_16_with_duplicates (P : ℚ) :
  (∃ x y z : ℕ, is_valid_combination x y z) → 
  P = 1 / 36 :=
sorry

end DiceProbability

end probability_of_sum_16_with_duplicates_l172_172895


namespace no_solution_15x_29y_43z_t2_l172_172822

theorem no_solution_15x_29y_43z_t2 (x y z t : ℕ) : ¬ (15 ^ x + 29 ^ y + 43 ^ z = t ^ 2) :=
by {
  -- We'll insert the necessary conditions for the proof here
  sorry -- proof goes here
}

end no_solution_15x_29y_43z_t2_l172_172822


namespace digging_foundation_l172_172790

-- Define given conditions
variable (m1 d1 m2 d2 k : ℝ)
variable (md_proportionality : m1 * d1 = k)
variable (k_value : k = 20 * 6)

-- Prove that for 30 men, it takes 4 days to dig the foundation
theorem digging_foundation : m1 = 20 ∧ d1 = 6 ∧ m2 = 30 → d2 = 4 :=
by
  sorry

end digging_foundation_l172_172790


namespace find_jack_euros_l172_172080

theorem find_jack_euros (E : ℕ) (h1 : 45 + 2 * E = 117) : E = 36 :=
by
  sorry

end find_jack_euros_l172_172080


namespace proof_f_value_l172_172363

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then 1 - x^2 else 2^x

theorem proof_f_value : f (1 / f (Real.log 6 / Real.log 2)) = 35 / 36 := by
  sorry

end proof_f_value_l172_172363


namespace parallel_line_passing_through_point_l172_172461

theorem parallel_line_passing_through_point :
  ∃ m b : ℝ, (∀ x y : ℝ, 4 * x + 2 * y = 8 → y = -2 * x + 4) ∧ b = 1 ∧ m = -2 ∧ b = 1 := by
  sorry

end parallel_line_passing_through_point_l172_172461


namespace symmetry_axis_of_f_l172_172269

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem symmetry_axis_of_f :
  ∃ k : ℤ, ∃ k_π_div_2 : ℝ, (f (k * π / 2 + π / 12) = f ((k * π / 2 + π / 12) + π)) :=
by {
  sorry
}

end symmetry_axis_of_f_l172_172269


namespace ones_digit_of_22_to_22_11_11_l172_172297

theorem ones_digit_of_22_to_22_11_11 : (22 ^ (22 * (11 ^ 11))) % 10 = 4 :=
by
  sorry

end ones_digit_of_22_to_22_11_11_l172_172297


namespace sum_of_digits_product_is_13_l172_172636

def base_eight_to_base_ten (n : ℕ) : ℕ := sorry
def product_base_eight (n1 n2 : ℕ) : ℕ := sorry
def digits_sum_base_ten (n : ℕ) : ℕ := sorry

theorem sum_of_digits_product_is_13 :
  let N1 := base_eight_to_base_ten 35
  let N2 := base_eight_to_base_ten 42
  let product := product_base_eight N1 N2
  digits_sum_base_ten product = 13 :=
by
  sorry

end sum_of_digits_product_is_13_l172_172636


namespace number_of_team_members_l172_172049

-- Let's define the conditions.
def packs : ℕ := 3
def pouches_per_pack : ℕ := 6
def total_pouches : ℕ := packs * pouches_per_pack
def coaches : ℕ := 3
def helpers : ℕ := 2
def total_people (members : ℕ) : ℕ := members + coaches + helpers

-- Prove the number of members on the baseball team.
theorem number_of_team_members (members : ℕ) (h : total_people members = total_pouches) : members = 13 :=
by
  sorry

end number_of_team_members_l172_172049


namespace ivan_income_tax_l172_172766

theorem ivan_income_tax :
  let salary_probation := 20000
  let probation_months := 2
  let salary_after_probation := 25000
  let after_probation_months := 8
  let bonus := 10000
  let tax_rate := 0.13
  let total_income := salary_probation * probation_months +
                      salary_after_probation * after_probation_months + bonus
  total_income * tax_rate = 32500 := sorry

end ivan_income_tax_l172_172766


namespace terminal_velocity_steady_speed_l172_172265

variable (g : ℝ) (t₁ t₂ : ℝ) (a₀ a₁ : ℝ) (v_terminal : ℝ)

-- Conditions
def acceleration_due_to_gravity := g = 10 -- m/s²
def initial_time := t₁ = 0 -- s
def intermediate_time := t₂ = 2 -- s
def initial_acceleration := a₀ = 50 -- m/s²
def final_acceleration := a₁ = 10 -- m/s²

-- Question: Prove the terminal velocity
theorem terminal_velocity_steady_speed 
  (h_g : acceleration_due_to_gravity g)
  (h_t1 : initial_time t₁)
  (h_t2 : intermediate_time t₂)
  (h_a0 : initial_acceleration a₀)
  (h_a1 : final_acceleration a₁) :
  v_terminal = 25 :=
  sorry

end terminal_velocity_steady_speed_l172_172265


namespace gcd_1213_1985_eq_1_l172_172462

theorem gcd_1213_1985_eq_1
  (h1: ¬ (1213 % 2 = 0))
  (h2: ¬ (1213 % 3 = 0))
  (h3: ¬ (1213 % 5 = 0))
  (h4: ¬ (1985 % 2 = 0))
  (h5: ¬ (1985 % 3 = 0))
  (h6: ¬ (1985 % 5 = 0)):
  Nat.gcd 1213 1985 = 1 := by
  sorry

end gcd_1213_1985_eq_1_l172_172462


namespace transformation_correct_l172_172364

theorem transformation_correct (a b c : ℝ) (h : a / c = b / c) (hc : c ≠ 0) : a = b :=
sorry

end transformation_correct_l172_172364


namespace cos_add_pi_over_4_l172_172742

theorem cos_add_pi_over_4 (α : ℝ) (h : Real.sin (α - π/4) = 1/3) : Real.cos (π/4 + α) = -1/3 := 
  sorry

end cos_add_pi_over_4_l172_172742


namespace arith_seq_ratio_l172_172142

variables {a₁ d : ℝ} (h₁ : d ≠ 0) (h₂ : (a₁ + 2*d)^2 ≠ a₁ * (a₁ + 8*d))

theorem arith_seq_ratio:
  (a₁ + 2*d) / (a₁ + 5*d) = 1 / 2 :=
sorry

end arith_seq_ratio_l172_172142


namespace cookie_radius_proof_l172_172255

-- Define the given equation of the cookie
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6 * x + 9 * y

-- Define the radius computation for the circle derived from the given equation
def cookie_radius (r : ℝ) : Prop :=
  r = 3 * Real.sqrt 5 / 2

-- The theorem to prove that the radius of the described cookie is as obtained
theorem cookie_radius_proof :
  ∀ x y : ℝ, cookie_equation x y → cookie_radius (Real.sqrt (45 / 4)) :=
by
  sorry

end cookie_radius_proof_l172_172255


namespace volume_ratio_of_cubes_l172_172115

def cube_volume (a : ℝ) : ℝ := a ^ 3

theorem volume_ratio_of_cubes :
  cube_volume 3 / cube_volume 18 = 1 / 216 :=
by
  sorry

end volume_ratio_of_cubes_l172_172115


namespace geometric_sequences_l172_172981

variable (a_n b_n : ℕ → ℕ) -- Geometric sequences
variable (S_n T_n : ℕ → ℕ) -- Sums of first n terms
variable (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4)

theorem geometric_sequences (n : ℕ) (h : ∀ n, S_n n / T_n n = (3^n + 1) / 4) : 
  (a_n 3) / (b_n 3) = 9 := 
sorry

end geometric_sequences_l172_172981


namespace total_green_marbles_l172_172225

-- Conditions
def Sara_green_marbles : ℕ := 3
def Tom_green_marbles : ℕ := 4

-- Problem statement: proving the total number of green marbles
theorem total_green_marbles : Sara_green_marbles + Tom_green_marbles = 7 := by
  sorry

end total_green_marbles_l172_172225


namespace no_nat_number_with_perfect_square_l172_172977

theorem no_nat_number_with_perfect_square (n : Nat) : 
  ¬ ∃ m : Nat, m * m = n^6 + 3 * n^5 - 5 * n^4 - 15 * n^3 + 4 * n^2 + 12 * n + 3 := 
  by
  sorry

end no_nat_number_with_perfect_square_l172_172977


namespace molecular_weight_calc_l172_172321

namespace MolecularWeightProof

def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90
def atomic_weight_O : ℝ := 16.00
def number_of_H : ℕ := 1
def number_of_Br : ℕ := 1
def number_of_O : ℕ := 3

theorem molecular_weight_calc :
  (number_of_H * atomic_weight_H + number_of_Br * atomic_weight_Br + number_of_O * atomic_weight_O) = 128.91 :=
by
  sorry

end MolecularWeightProof

end molecular_weight_calc_l172_172321


namespace isabella_haircut_length_l172_172513

-- Define the original length of Isabella's hair.
def original_length : ℕ := 18

-- Define the length of hair cut off.
def cut_off_length : ℕ := 9

-- The length of Isabella's hair after the haircut.
def length_after_haircut : ℕ := original_length - cut_off_length

-- Statement of the theorem we want to prove.
theorem isabella_haircut_length : length_after_haircut = 9 :=
by
  sorry

end isabella_haircut_length_l172_172513


namespace largest_digit_divisible_by_6_l172_172270

def divisibleBy2 (N : ℕ) : Prop :=
  ∃ k, N = 2 * k

def divisibleBy3 (N : ℕ) : Prop :=
  ∃ k, N = 3 * k

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N ≤ 9 ∧ divisibleBy2 N ∧ divisibleBy3 (26 + N) ∧ (∀ M : ℕ, M ≤ 9 ∧ divisibleBy2 M ∧ divisibleBy3 (26 + M) → M ≤ N) ∧ N = 4 :=
by
  sorry

end largest_digit_divisible_by_6_l172_172270


namespace maximum_x_value_l172_172679

theorem maximum_x_value (x y z : ℝ) (h1 : x + y + z = 10) (h2 : x * y + x * z + y * z = 20) : 
  x ≤ 10 / 3 := sorry

end maximum_x_value_l172_172679


namespace lowest_fraction_of_job_done_l172_172258

theorem lowest_fraction_of_job_done :
  ∀ (rateA rateB rateC rateB_plus_C : ℝ),
  (rateA = 1/4) → (rateB = 1/6) → (rateC = 1/8) →
  (rateB_plus_C = rateB + rateC) →
  rateB_plus_C = 7/24 := by
  intros rateA rateB rateC rateB_plus_C hA hB hC hBC
  sorry

end lowest_fraction_of_job_done_l172_172258


namespace men_seated_l172_172410

theorem men_seated (total_passengers : ℕ) (women_ratio : ℚ) (children_count : ℕ) (men_standing_ratio : ℚ) 
  (women_with_prams : ℕ) (disabled_passengers : ℕ) 
  (h_total_passengers : total_passengers = 48) 
  (h_women_ratio : women_ratio = 2 / 3) 
  (h_children_count : children_count = 5) 
  (h_men_standing_ratio : men_standing_ratio = 1 / 8) 
  (h_women_with_prams : women_with_prams = 3) 
  (h_disabled_passengers : disabled_passengers = 2) : 
  (total_passengers * (1 - women_ratio) - total_passengers * (1 - women_ratio) * men_standing_ratio = 14) :=
by sorry

end men_seated_l172_172410


namespace tangent_half_angle_sum_eq_product_l172_172371

variable {α β γ : ℝ}

theorem tangent_half_angle_sum_eq_product (h : α + β + γ = 2 * Real.pi) :
  Real.tan (α / 2) + Real.tan (β / 2) + Real.tan (γ / 2) =
  Real.tan (α / 2) * Real.tan (β / 2) * Real.tan (γ / 2) :=
sorry

end tangent_half_angle_sum_eq_product_l172_172371


namespace fill_tank_time_l172_172063

-- Definitions based on provided conditions
def pipeA_time := 60 -- Pipe A fills the tank in 60 minutes
def pipeB_time := 40 -- Pipe B fills the tank in 40 minutes

-- Theorem statement
theorem fill_tank_time (T : ℕ) : 
  (T / 2) / pipeB_time + (T / 2) * (1 / pipeA_time + 1 / pipeB_time) = 1 → 
  T = 48 :=
by
  intro h
  sorry

end fill_tank_time_l172_172063


namespace age_difference_l172_172009

variable (E Y : ℕ)

theorem age_difference (hY : Y = 35) (hE : E - 15 = 2 * (Y - 15)) : E - Y = 20 := by
  -- Assertions and related steps could be handled subsequently.
  sorry

end age_difference_l172_172009


namespace proof_problem_l172_172187

noncomputable def calc_a_star_b (a b : ℤ) : ℚ :=
1 / (a:ℚ) + 1 / (b:ℚ)

theorem proof_problem (a b : ℤ) (h1 : a + b = 10) (h2 : a * b = 24) :
  calc_a_star_b a b = 5 / 12 ∧ (a * b > a + b) := by
  sorry

end proof_problem_l172_172187


namespace limit_one_minus_reciprocal_l172_172998

theorem limit_one_minus_reciprocal (h : Filter.Tendsto (fun (n : ℕ) => 1 / n) Filter.atTop (nhds 0)) :
  Filter.Tendsto (fun (n : ℕ) => 1 - 1 / n) Filter.atTop (nhds 1) :=
sorry

end limit_one_minus_reciprocal_l172_172998


namespace determine_s_plus_u_l172_172030

theorem determine_s_plus_u (p r s u : ℂ) (q t : ℂ) (h₁ : q = 5)
    (h₂ : t = -p - r) (h₃ : p + q * I + r + s * I + t + u * I = 4 * I) : s + u = -1 :=
by
  sorry

end determine_s_plus_u_l172_172030


namespace additional_investment_l172_172567

-- Given the conditions
variables (x y : ℝ)
def interest_rate_1 := 0.02
def interest_rate_2 := 0.04
def invested_amount := 1000
def total_interest := 92

-- Theorem to prove
theorem additional_investment : 
  0.02 * invested_amount + 0.04 * (invested_amount + y) = total_interest → 
  y = 800 :=
by
  sorry

end additional_investment_l172_172567


namespace dilution_plate_count_lower_than_actual_l172_172948

theorem dilution_plate_count_lower_than_actual
  (bacteria_count : ℕ)
  (colony_count : ℕ)
  (dilution_factor : ℕ)
  (plate_count : ℕ)
  (count_error_margin : ℕ)
  (method_estimation_error : ℕ)
  (H1 : method_estimation_error > 0)
  (H2 : colony_count = bacteria_count / dilution_factor - method_estimation_error)
  : colony_count < bacteria_count :=
by
  sorry

end dilution_plate_count_lower_than_actual_l172_172948


namespace find_geometric_sequence_values_l172_172650

structure GeometricSequence (a b c d : ℝ) : Prop where
  ratio1 : b / a = c / b
  ratio2 : c / b = d / c

theorem find_geometric_sequence_values (x u v y : ℝ)
    (h1 : x + y = 20)
    (h2 : u + v = 34)
    (h3 : x^2 + u^2 + v^2 + y^2 = 1300) :
    (GeometricSequence x u v y ∧ ((x = 16 ∧ u = 4 ∧ v = 32 ∧ y = 2) ∨ (x = 4 ∧ u = 16 ∧ v = 2 ∧ y = 32))) :=
by
  sorry

end find_geometric_sequence_values_l172_172650


namespace brads_zip_code_l172_172087

theorem brads_zip_code (A B C D E : ℕ) (h1 : A + B + C + D + E = 20)
                        (h2 : B = A + 1) (h3 : C = A)
                        (h4 : D = 2 * A) (h5 : D + E = 13)
                        (h6 : Nat.Prime (A*10000 + B*1000 + C*100 + D*10 + E)) :
                        A*10000 + B*1000 + C*100 + D*10 + E = 34367 := 
sorry

end brads_zip_code_l172_172087


namespace decreasing_function_iff_a_range_l172_172453

noncomputable def f (a x : ℝ) : ℝ := (1 - 2 * a) ^ x

theorem decreasing_function_iff_a_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x > f a y) ↔ 0 < a ∧ a < 1/2 :=
by
  sorry

end decreasing_function_iff_a_range_l172_172453


namespace determinant_zero_l172_172148

def matrix_determinant (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1, x, y + z],
    ![1, x + y, z],
    ![1, x + z, y]
  ]

theorem determinant_zero (x y z : ℝ) : matrix_determinant x y z = 0 := 
by
  sorry

end determinant_zero_l172_172148


namespace sufficient_condition_transitive_l172_172934

theorem sufficient_condition_transitive
  (C B A : Prop) (h1 : (C → B)) (h2 : (B → A)) : (C → A) :=
  sorry

end sufficient_condition_transitive_l172_172934


namespace polygon_sides_l172_172581

theorem polygon_sides :
  ∀ (n : ℕ), (n > 2) → (n - 2) * 180 < 360 → n = 3 :=
by
  intros n hn1 hn2
  sorry

end polygon_sides_l172_172581


namespace machine_A_sprockets_per_hour_l172_172170

theorem machine_A_sprockets_per_hour :
  ∃ (A : ℝ), 
    (∃ (G : ℝ), 
      (G = 1.10 * A) ∧ 
      (∃ (T : ℝ), 
        (660 = A * (T + 10)) ∧ 
        (660 = G * T) 
      )
    ) ∧ 
    (A = 6) :=
by
  -- Conditions and variables will be introduced here...
  -- Proof can be implemented here
  sorry

end machine_A_sprockets_per_hour_l172_172170


namespace min_value_expression_l172_172454

theorem min_value_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (H : 1 / a + 1 / b = 1) :
  ∃ c : ℝ, (∀ (a b : ℝ), 0 < a → 0 < b → 1 / a + 1 / b = 1 → c ≤ 4 / (a - 1) + 9 / (b - 1)) ∧ (c = 6) :=
by
  sorry

end min_value_expression_l172_172454


namespace strategy_probabilities_l172_172343

noncomputable def P1 : ℚ := 1 / 3
noncomputable def P2 : ℚ := 1 / 2
noncomputable def P3 : ℚ := 2 / 3

theorem strategy_probabilities :
  (P1 < P2) ∧
  (P1 < P3) ∧
  (2 * P1 = P3) := by
  sorry

end strategy_probabilities_l172_172343


namespace find_value_of_expression_l172_172669

theorem find_value_of_expression (x : ℝ) (h : x^2 + (1 / x^2) = 5) : x^4 + (1 / x^4) = 23 :=
by
  sorry

end find_value_of_expression_l172_172669


namespace value_expression_l172_172901

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_expression (p q r s t : ℝ) (h : g p q r s t (-3) = 9) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = -9 := 
by
  sorry

end value_expression_l172_172901


namespace part1_part2_l172_172251

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.cos x

theorem part1 (hx : f (-x) = 2 * f x) : f x ^ 2 = 2 / 5 := 
  sorry

theorem part2 : 
  ∀ k : ℤ, ∃ a b : ℝ, [a, b] = [2 * π * k + (5 * π / 6), 2 * π * k + (11 * π / 6)] ∧ 
  ∀ x : ℝ, x ∈ Set.Icc a b → ∀ y : ℝ, y = f (π / 12 - x) → 
  ∃ δ > 0, ∀ ε > 0, 0 < |x - y| ∧ |x - y| < δ → y < x := 
  sorry

end part1_part2_l172_172251


namespace quadratic_root_value_l172_172813

theorem quadratic_root_value
  (a : ℝ) 
  (h : a^2 + 3 * a - 1010 = 0) :
  2 * a^2 + 6 * a + 4 = 2024 :=
by
  sorry

end quadratic_root_value_l172_172813


namespace total_tires_l172_172260

def cars := 15
def bicycles := 3
def pickup_trucks := 8
def tricycles := 1

def tires_per_car := 4
def tires_per_bicycle := 2
def tires_per_pickup_truck := 4
def tires_per_tricycle := 3

theorem total_tires : (cars * tires_per_car) + (bicycles * tires_per_bicycle) + (pickup_trucks * tires_per_pickup_truck) + (tricycles * tires_per_tricycle) = 101 :=
by
  sorry

end total_tires_l172_172260


namespace solve_equation_l172_172645

theorem solve_equation :
  ∃ x : ℝ, (x + 2) / 4 - (2 * x - 3) / 6 = 2 ∧ x = -12 :=
by
  sorry

end solve_equation_l172_172645


namespace number_of_dogs_is_correct_l172_172465

variable (D C B : ℕ)
variable (k : ℕ)

def validRatio (D C B : ℕ) : Prop := D = 7 * k ∧ C = 7 * k ∧ B = 8 * k
def totalDogsAndBunnies (D B : ℕ) : Prop := D + B = 330
def correctNumberOfDogs (D : ℕ) : Prop := D = 154

theorem number_of_dogs_is_correct (D C B k : ℕ) 
  (hRatio : validRatio D C B k)
  (hTotal : totalDogsAndBunnies D B) :
  correctNumberOfDogs D :=
by
  sorry

end number_of_dogs_is_correct_l172_172465


namespace range_of_b_l172_172114

theorem range_of_b (x b : ℝ) (hb : b > 0) : 
  (∃ x : ℝ, |x - 2| + |x + 1| < b) ↔ b > 3 :=
by
  sorry

end range_of_b_l172_172114


namespace find_k_value_l172_172301

theorem find_k_value (k : ℝ) (h : 64 / k = 4) : k = 16 :=
by
  sorry

end find_k_value_l172_172301


namespace no_solution_for_99_l172_172134

theorem no_solution_for_99 :
  ∃ n : ℕ, (¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = n) ∧
  (∀ m : ℕ, n < m → ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ 9 * x + 11 * y = m) ∧
  n = 99 :=
by
  sorry

end no_solution_for_99_l172_172134


namespace eight_n_plus_nine_is_perfect_square_l172_172713

theorem eight_n_plus_nine_is_perfect_square 
  (n : ℕ) (N : ℤ) 
  (hN : N = 2 ^ (4 * n + 1) - 4 ^ n - 1)
  (hdiv : 9 ∣ N) :
  ∃ k : ℤ, 8 * N + 9 = k ^ 2 :=
by
  sorry

end eight_n_plus_nine_is_perfect_square_l172_172713


namespace min_period_and_sym_center_l172_172268

open Real

noncomputable def func (x α β : ℝ) : ℝ :=
  sin (x - α) * cos (x - β)

theorem min_period_and_sym_center (α β : ℝ) :
  (∀ x, func (x + π) α β = func x α β) ∧ (func α 0 β = 0) :=
by
  sorry

end min_period_and_sym_center_l172_172268


namespace six_points_within_circle_l172_172566

/-- If six points are placed inside or on a circle with radius 1, then 
there always exist at least two points such that the distance between 
them is at most 1. -/
theorem six_points_within_circle : ∀ (points : Fin 6 → ℝ × ℝ), 
  (∀ i, (points i).1^2 + (points i).2^2 ≤ 1) → 
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 1 :=
by
  -- Condition: Circle of radius 1
  intro points h_points
  sorry

end six_points_within_circle_l172_172566


namespace not_geometric_sequence_of_transformed_l172_172116

theorem not_geometric_sequence_of_transformed (a b c : ℝ) (q : ℝ) (hq : q ≠ 1) 
  (h_geometric : b = a * q ∧ c = b * q) :
  ¬ (∃ q' : ℝ, 1 - b = (1 - a) * q' ∧ 1 - c = (1 - b) * q') :=
by
  sorry

end not_geometric_sequence_of_transformed_l172_172116


namespace cost_of_one_dozen_pens_l172_172155

theorem cost_of_one_dozen_pens
  (p q : ℕ)
  (h1 : 3 * p + 5 * q = 240)
  (h2 : p = 5 * q) :
  12 * p = 720 :=
by
  sorry

end cost_of_one_dozen_pens_l172_172155


namespace determine_p_l172_172909

theorem determine_p (p : ℝ) (h : (2 * p - 1) * (-1)^2 + 2 * (1 - p) * (-1) + 3 * p = 0) : p = 3 / 7 := by
  sorry

end determine_p_l172_172909


namespace sum_cubes_eq_power_l172_172541

/-- Given the conditions, prove that 1^3 + 2^3 + 3^3 + 4^3 = 10^2 -/
theorem sum_cubes_eq_power : 1 + 2 + 3 + 4 = 10 → 1^3 + 2^3 + 3^3 + 4^3 = 10^2 :=
by
  intro h
  sorry

end sum_cubes_eq_power_l172_172541


namespace brownie_leftover_is_zero_l172_172058

-- Define the dimensions of the pan
def pan_length : ℕ := 24
def pan_width : ℕ := 15

-- Define the dimensions of one piece of brownie
def piece_length : ℕ := 3
def piece_width : ℕ := 4

-- The total area of the pan
def pan_area : ℕ := pan_length * pan_width

-- The total area of one piece
def piece_area : ℕ := piece_length * piece_width

-- The number of full pieces that can be cut
def number_of_pieces : ℕ := pan_area / piece_area

-- The total used area when pieces are cut
def used_area : ℕ := number_of_pieces * piece_area

-- The leftover area
def leftover_area : ℕ := pan_area - used_area

theorem brownie_leftover_is_zero (pan_length pan_width piece_length piece_width : ℕ)
  (h1 : pan_length = 24) (h2 : pan_width = 15) 
  (h3 : piece_length = 3) (h4 : piece_width = 4) :
  pan_width * pan_length - (pan_width * pan_length / (piece_width * piece_length)) * (piece_width * piece_length) = 0 := 
by sorry

end brownie_leftover_is_zero_l172_172058


namespace problem_statement_l172_172526

theorem problem_statement
  (a b m n c : ℝ)
  (h1 : a = -b)
  (h2 : m * n = 1)
  (h3 : |c| = 3)
  : a + b + m * n - |c| = -2 := by
  sorry

end problem_statement_l172_172526


namespace common_chord_eqn_l172_172960

theorem common_chord_eqn (x y : ℝ) :
  (x^2 + y^2 + 2 * x - 6 * y + 1 = 0) ∧
  (x^2 + y^2 - 4 * x + 2 * y - 11 = 0) →
  3 * x - 4 * y + 6 = 0 :=
by
  intro h
  sorry

end common_chord_eqn_l172_172960


namespace max_value_f_l172_172707

noncomputable def f (x : ℝ) : ℝ := (4 * x - 4 * x^3) / (1 + 2 * x^2 + x^4)

theorem max_value_f : ∃ x : ℝ, (f x = 1) ∧ (∀ y : ℝ, f y ≤ 1) :=
sorry

end max_value_f_l172_172707


namespace bob_monthly_hours_l172_172971

noncomputable def total_hours_in_month : ℝ :=
  let daily_hours := 10
  let weekly_days := 5
  let weeks_in_month := 4.33
  daily_hours * weekly_days * weeks_in_month

theorem bob_monthly_hours :
  total_hours_in_month = 216.5 :=
by
  sorry

end bob_monthly_hours_l172_172971


namespace geom_seq_sum_l172_172376

theorem geom_seq_sum (a : ℕ → ℝ) (q : ℝ) (h1 : a 1 = 3)
  (h2 : a 1 + a 2 + a 3 = 21)
  (h3 : ∀ n, a (n + 1) = a n * q) : a 4 + a 5 + a 6 = 168 :=
sorry

end geom_seq_sum_l172_172376


namespace days_elapsed_l172_172373

theorem days_elapsed
  (initial_amount : ℕ)
  (daily_spending : ℕ)
  (total_savings : ℕ)
  (doubling_factor : ℕ)
  (additional_amount : ℕ)
  :
  initial_amount = 50 →
  daily_spending = 15 →
  doubling_factor = 2 →
  additional_amount = 10 →
  2 * (initial_amount - daily_spending) * total_savings + additional_amount = 500 →
  total_savings = 7 :=
by
  intros h_initial h_spending h_doubling h_additional h_total
  sorry

end days_elapsed_l172_172373


namespace deschamps_cows_l172_172873

theorem deschamps_cows (p v : ℕ) (h1 : p + v = 160) (h2 : 2 * p + 4 * v = 400) : v = 40 :=
by sorry

end deschamps_cows_l172_172873


namespace prove_x_minus_y_squared_l172_172568

noncomputable section

variables {x y a b : ℝ}

theorem prove_x_minus_y_squared (h1 : x * y = b) (h2 : x / y + y / x = a) : (x - y) ^ 2 = a * b - 2 * b := 
  sorry

end prove_x_minus_y_squared_l172_172568


namespace train_b_speed_l172_172784

variable (v : ℝ) -- the speed of Train B

theorem train_b_speed 
  (speedA : ℝ := 30) -- speed of Train A
  (head_start_hours : ℝ := 2) -- head start time in hours
  (overtake_distance : ℝ := 285) -- distance at which Train B overtakes Train A
  (train_a_travel_distance : ℝ := speedA * head_start_hours) -- distance Train A travels in the head start time
  (total_distance : ℝ := 345) -- total distance Train B travels to overtake Train A
  (train_a_travel_time : ℝ := overtake_distance / speedA) -- time taken by Train A to travel the overtake distance
  : v * train_a_travel_time = total_distance → v = 36.32 :=
by
  sorry

end train_b_speed_l172_172784


namespace JamesFlowers_l172_172228

noncomputable def numberOfFlowersJamesPlantedInADay (F : ℝ) := 0.5 * (F + 0.15 * F)

theorem JamesFlowers (F : ℝ) (H₁ : 6 * F + (F + 0.15 * F) = 315) : numberOfFlowersJamesPlantedInADay F = 25.3:=
by
  sorry

end JamesFlowers_l172_172228


namespace geometric_sequence_sixth_term_l172_172394

theorem geometric_sequence_sixth_term:
  ∃ q : ℝ, 
  ∀ (a₁ a₈ a₆ : ℝ), 
    a₁ = 6 ∧ a₈ = 768 ∧ a₈ = a₁ * q^7 ∧ a₆ = a₁ * q^5 
    → a₆ = 192 :=
by
  sorry

end geometric_sequence_sixth_term_l172_172394


namespace range_of_a_l172_172586

theorem range_of_a (a : ℝ) : 
  {x : ℝ | x^2 - 4 * x + 3 < 0} ⊆ {x : ℝ | 2^(1 - x) + a ≤ 0 ∧ x^2 - 2 * (a + 7) * x + 5 ≤ 0} → 
  -4 ≤ a ∧ a ≤ -1 := by
  sorry

end range_of_a_l172_172586


namespace gcd_polynomials_l172_172635

def P (n : ℤ) : ℤ := n^3 - 6 * n^2 + 11 * n - 6
def Q (n : ℤ) : ℤ := n^2 - 4 * n + 4

theorem gcd_polynomials (n : ℤ) (h : n ≥ 3) : Int.gcd (P n) (Q n) = n - 2 :=
by
  sorry

end gcd_polynomials_l172_172635


namespace calc_x2015_l172_172090

noncomputable def f (x a : ℝ) : ℝ := x / (a * (x + 2))

theorem calc_x2015 (a x x_0 : ℝ) (x_seq : ℕ → ℝ)
  (h_unique: ∀ x, f x a = x → x = 0) 
  (h_a_val: a = 1 / 2)
  (h_f_x0: f x_0 a = 1 / 1008)
  (h_seq: ∀ n, x_seq (n + 1) = f (x_seq n) a)
  (h_x0_val: x_seq 0 = x_0):
  x_seq 2015 = 1 / 2015 :=
by
  sorry

end calc_x2015_l172_172090


namespace naomi_total_time_l172_172512

-- Definitions
def time_to_parlor : ℕ := 60
def speed_ratio : ℕ := 2 -- because her returning speed is half of the going speed
def first_trip_delay : ℕ := 15
def coffee_break : ℕ := 10
def second_trip_delay : ℕ := 20
def detour_time : ℕ := 30

-- Calculate total round trip times
def first_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + first_trip_delay + coffee_break
def second_round_trip_time : ℕ := time_to_parlor + speed_ratio * time_to_parlor + second_trip_delay + detour_time

-- Hypothesis
def total_round_trip_time : ℕ := first_round_trip_time + second_round_trip_time

-- Main theorem statement
theorem naomi_total_time : total_round_trip_time = 435 := by
  sorry

end naomi_total_time_l172_172512


namespace probability_of_color_change_l172_172622

def traffic_light_cycle := 90
def green_duration := 45
def yellow_duration := 5
def red_duration := 40
def green_to_yellow := green_duration
def yellow_to_red := green_duration + yellow_duration
def red_to_green := traffic_light_cycle
def observation_interval := 4
def valid_intervals := [green_to_yellow - observation_interval + 1, green_to_yellow, 
                        yellow_to_red - observation_interval + 1, yellow_to_red, 
                        red_to_green - observation_interval + 1, red_to_green]
def total_valid_intervals := valid_intervals.length * observation_interval

theorem probability_of_color_change : 
  (total_valid_intervals : ℚ) / traffic_light_cycle = 2 / 15 := 
by
  sorry

end probability_of_color_change_l172_172622


namespace Gage_skating_time_l172_172124

theorem Gage_skating_time :
  let min_per_hr := 60
  let skating_6_days := 6 * (1 * min_per_hr + 20)
  let skating_4_days := 4 * (1 * min_per_hr + 35)
  let needed_total := 11 * 90
  let skating_10_days := skating_6_days + skating_4_days
  let minutes_on_eleventh_day := needed_total - skating_10_days
  minutes_on_eleventh_day = 130 :=
by
  sorry

end Gage_skating_time_l172_172124


namespace intersection_A_B_l172_172715

-- Define set A based on the given condition.
def setA : Set ℝ := {x | x^2 - 4 < 0}

-- Define set B based on the given condition.
def setB : Set ℝ := {x | x < 0}

-- Prove that the intersection of sets A and B is the given set.
theorem intersection_A_B : setA ∩ setB = {x | -2 < x ∧ x < 0} := by
  sorry

end intersection_A_B_l172_172715


namespace distance_house_to_market_l172_172429

-- Define each of the given conditions
def distance_to_school := 50
def distance_to_park_from_school := 25
def return_distance := 60
def total_distance_walked := 220

-- Proven distance to the market
def distance_to_market := 85

-- Statement to prove
theorem distance_house_to_market (d1 d2 d3 d4 : ℕ) 
  (h1 : d1 = distance_to_school) 
  (h2 : d2 = distance_to_park_from_school) 
  (h3 : d3 = return_distance) 
  (h4 : d4 = total_distance_walked) :
  d4 - (d1 + d2 + d3) = distance_to_market := 
by
  sorry

end distance_house_to_market_l172_172429


namespace value_of_3a_minus_b_l172_172158
noncomputable def solveEquation : Type := sorry

theorem value_of_3a_minus_b (a b : ℝ) (h1 : a = 3 + Real.sqrt 15) (h2 : b = 3 - Real.sqrt 15) (h3 : a ≥ b) :
  3 * a - b = 6 + 4 * Real.sqrt 15 :=
sorry

end value_of_3a_minus_b_l172_172158


namespace number_of_ferns_is_six_l172_172383

def num_fronds_per_fern : Nat := 7
def num_leaves_per_frond : Nat := 30
def total_leaves : Nat := 1260

theorem number_of_ferns_is_six :
  total_leaves = num_fronds_per_fern * num_leaves_per_frond * 6 :=
by
  sorry

end number_of_ferns_is_six_l172_172383


namespace find_k_l172_172451

variable {S : ℕ → ℤ} -- Assuming the sum function S for the arithmetic sequence 
variable {k : ℕ} -- k is a natural number

theorem find_k (h1 : S (k - 2) = -4) (h2 : S k = 0) (h3 : S (k + 2) = 8) (hk2 : k > 2) (hnaturalk : k ∈ Set.univ) : k = 6 := by
  sorry

end find_k_l172_172451


namespace cube_volume_ratio_l172_172578

theorem cube_volume_ratio (a b : ℝ) (h : (a^2 / b^2) = 9 / 25) :
  (b^3 / a^3) = 125 / 27 :=
by
  sorry

end cube_volume_ratio_l172_172578


namespace journey_total_time_l172_172711

def journey_time (d1 d2 : ℕ) (total_distance : ℕ) (car_speed walk_speed : ℕ) : ℕ :=
  d1 / car_speed + (total_distance - d1) / walk_speed

theorem journey_total_time :
  let total_distance := 150
  let car_speed := 30
  let walk_speed := 3
  let d1 := 50
  let d2 := 15
  
  journey_time d1 d2 total_distance car_speed walk_speed =
  max (journey_time d1 0 total_distance car_speed walk_speed / car_speed + 
       (total_distance - d1) / walk_speed)
      ((d1 / car_speed + (d1 - d2) / car_speed + (total_distance - d1 + d2) / car_speed)) :=
by
  sorry

end journey_total_time_l172_172711


namespace converse_of_statement_l172_172077

theorem converse_of_statement (x y : ℝ) :
  (¬ (x = 0 ∧ y = 0)) → (x^2 + y^2 ≠ 0) :=
by {
  sorry
}

end converse_of_statement_l172_172077


namespace total_worth_correct_l172_172020

def row1_gold_bars : ℕ := 5
def row1_weight_per_bar : ℕ := 2
def row1_cost_per_kg : ℕ := 20000

def row2_gold_bars : ℕ := 8
def row2_weight_per_bar : ℕ := 3
def row2_cost_per_kg : ℕ := 18000

def row3_gold_bars : ℕ := 3
def row3_weight_per_bar : ℕ := 5
def row3_cost_per_kg : ℕ := 22000

def row4_gold_bars : ℕ := 4
def row4_weight_per_bar : ℕ := 4
def row4_cost_per_kg : ℕ := 25000

def total_worth : ℕ :=
  (row1_gold_bars * row1_weight_per_bar * row1_cost_per_kg)
  + (row2_gold_bars * row2_weight_per_bar * row2_cost_per_kg)
  + (row3_gold_bars * row3_weight_per_bar * row3_cost_per_kg)
  + (row4_gold_bars * row4_weight_per_bar * row4_cost_per_kg)

theorem total_worth_correct : total_worth = 1362000 := by
  sorry

end total_worth_correct_l172_172020


namespace max_cookies_eaten_l172_172241

def prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem max_cookies_eaten 
  (total_cookies : ℕ)
  (andy_cookies : ℕ)
  (alexa_cookies : ℕ)
  (hx : andy_cookies + alexa_cookies = total_cookies)
  (hp : ∃ p : ℕ, prime p ∧ alexa_cookies = p * andy_cookies)
  (htotal : total_cookies = 30) :
  andy_cookies = 10 :=
  sorry

end max_cookies_eaten_l172_172241


namespace inequality_solution_l172_172208

theorem inequality_solution (x : ℝ) (h₀ : x ≠ 0) (h₂ : x ≠ 2) : 
  (x ∈ (Set.Ioi 0 ∩ Set.Iic (1/2)) ∪ (Set.Ioi 1.5 ∩ Set.Iio 2)) 
  ↔ ( (x + 1) / (x - 2) + (x + 3) / (3 * x) ≥ 4 ) := by
  sorry

end inequality_solution_l172_172208


namespace v_is_82_875_percent_of_z_l172_172612

theorem v_is_82_875_percent_of_z (x y z w v : ℝ) 
  (h1 : x = 1.30 * y)
  (h2 : y = 0.60 * z)
  (h3 : w = 1.25 * x)
  (h4 : v = 0.85 * w) : 
  v = 0.82875 * z :=
by
  sorry

end v_is_82_875_percent_of_z_l172_172612


namespace line_through_point_parallel_to_given_line_l172_172171

theorem line_through_point_parallel_to_given_line 
  (x y : ℝ) 
  (h₁ : (x, y) = (1, -4)) 
  (h₂ : ∀ m : ℝ, 2 * 1 + 3 * (-4) + m = 0 → m = 10)
  : 2 * x + 3 * y + 10 = 0 :=
sorry

end line_through_point_parallel_to_given_line_l172_172171


namespace go_to_yolka_together_l172_172787

noncomputable def anya_will_not_wait : Prop := true
noncomputable def boris_wait_time : ℕ := 10 -- in minutes
noncomputable def vasya_wait_time : ℕ := 15 -- in minutes
noncomputable def meeting_time_window : ℕ := 60 -- total available time in minutes

noncomputable def probability_all_go_together : ℝ :=
  (1 / 3) * (3500 / 3600)

theorem go_to_yolka_together :
  anya_will_not_wait ∧
  boris_wait_time = 10 ∧
  vasya_wait_time = 15 ∧
  meeting_time_window = 60 →
  probability_all_go_together = 0.324 :=
by
  intros
  sorry

end go_to_yolka_together_l172_172787


namespace daily_sales_change_l172_172686

theorem daily_sales_change
    (mon_sales : ℕ)
    (week_total_sales : ℕ)
    (days_in_week : ℕ)
    (avg_sales_per_day : ℕ)
    (other_days_total_sales : ℕ)
    (x : ℕ)
    (h1 : days_in_week = 7)
    (h2 : avg_sales_per_day = 5)
    (h3 : week_total_sales = avg_sales_per_day * days_in_week)
    (h4 : mon_sales = 2)
    (h5 : week_total_sales = mon_sales + other_days_total_sales)
    (h6 : other_days_total_sales = 33)
    (h7 : 2 + x + 2 + 2*x + 2 + 3*x + 2 + 4*x + 2 + 5*x + 2 + 6*x = other_days_total_sales) : 
  x = 1 :=
by
sorry

end daily_sales_change_l172_172686


namespace derivative_at_zero_l172_172597

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then (Real.log (1 + 2 * x^2 + x^3)) / x else 0

theorem derivative_at_zero : deriv f 0 = 2 := by
  sorry

end derivative_at_zero_l172_172597


namespace max_value_f_value_of_f_at_alpha_l172_172792

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * Real.sin x

theorem max_value_f :
  (∀ x, f x ≤ 3)
  ∧ (∃ x, f x = 3)
  ∧ {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} = {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} :=
sorry

theorem value_of_f_at_alpha {α : ℝ} (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_f_value_of_f_at_alpha_l172_172792


namespace sara_spent_on_hotdog_l172_172244

def total_cost_of_lunch: ℝ := 10.46
def cost_of_salad: ℝ := 5.10
def cost_of_hotdog: ℝ := total_cost_of_lunch - cost_of_salad

theorem sara_spent_on_hotdog :
  cost_of_hotdog = 5.36 := by
  sorry

end sara_spent_on_hotdog_l172_172244


namespace infinite_n_exists_l172_172182

theorem infinite_n_exists (p : ℕ) (hp : Nat.Prime p) (hp_gt_7 : 7 < p) :
  ∃ᶠ n in at_top, (n ≡ 1 [MOD 2016]) ∧ (p ∣ 2^n + n) :=
sorry

end infinite_n_exists_l172_172182


namespace ball_box_distribution_l172_172610

theorem ball_box_distribution:
  ∃ (C : ℕ → ℕ → ℕ) (A : ℕ → ℕ → ℕ),
  C 4 2 * A 3 3 = sorry := 
by sorry

end ball_box_distribution_l172_172610


namespace cumulative_percentage_decrease_l172_172209

theorem cumulative_percentage_decrease :
  let original_price := 100
  let first_reduction := original_price * 0.85
  let second_reduction := first_reduction * 0.90
  let third_reduction := second_reduction * 0.95
  let fourth_reduction := third_reduction * 0.80
  let final_price := fourth_reduction
  (original_price - final_price) / original_price * 100 = 41.86 := by
  sorry

end cumulative_percentage_decrease_l172_172209


namespace probability_perfect_square_sum_l172_172419

def is_perfect_square_sum (n : ℕ) : Prop :=
  n = 4 ∨ n = 9 ∨ n = 16

def count_perfect_square_sums : ℕ :=
  let possible_outcomes := 216
  let favorable_outcomes := 32
  favorable_outcomes

theorem probability_perfect_square_sum :
  (count_perfect_square_sums : ℚ) / 216 = 4 / 27 :=
by
  sorry

end probability_perfect_square_sum_l172_172419


namespace martin_crayons_l172_172140

theorem martin_crayons : (8 * 7 = 56) := by
  sorry

end martin_crayons_l172_172140


namespace d_divisibility_l172_172550

theorem d_divisibility (p d : ℕ) (h_p : 0 < p) (h_d : 0 < d)
  (h1 : Prime p) 
  (h2 : Prime (p + d)) 
  (h3 : Prime (p + 2 * d)) 
  (h4 : Prime (p + 3 * d)) 
  (h5 : Prime (p + 4 * d)) 
  (h6 : Prime (p + 5 * d)) : 
  (2 ∣ d) ∧ (3 ∣ d) ∧ (5 ∣ d) :=
by
  sorry

end d_divisibility_l172_172550


namespace candle_ratio_proof_l172_172361

noncomputable def candle_height_ratio := 
  ∃ (x y : ℝ), 
    (x / 6) * 3 = x / 2 ∧
    (y / 8) * 3 = 3 * y / 8 ∧
    (x / 2) = (5 * y / 8) →
    x / y = 5 / 4

theorem candle_ratio_proof : candle_height_ratio :=
by sorry

end candle_ratio_proof_l172_172361


namespace min_third_side_of_right_triangle_l172_172899

theorem min_third_side_of_right_triangle (a b : ℕ) (h : a = 7 ∧ b = 24) : 
  ∃ (c : ℝ), c = Real.sqrt (576 - 49) :=
by
  sorry

end min_third_side_of_right_triangle_l172_172899


namespace cos_fourth_power_sum_l172_172252

open Real

theorem cos_fourth_power_sum :
  (cos (0 : ℝ))^4 + (cos (π / 6))^4 + (cos (π / 3))^4 + (cos (π / 2))^4 +
  (cos (2 * π / 3))^4 + (cos (5 * π / 6))^4 + (cos π)^4 = 13 / 4 := 
by
  sorry

end cos_fourth_power_sum_l172_172252


namespace complex_div_imag_unit_l172_172807

theorem complex_div_imag_unit (i : ℂ) (h : i^2 = -1) : (1 + i) / (1 - i) = i :=
sorry

end complex_div_imag_unit_l172_172807


namespace angle_between_AB_CD_l172_172378

def point := (ℝ × ℝ × ℝ)

def A : point := (-3, 0, 1)
def B : point := (2, 1, -1)
def C : point := (-2, 2, 0)
def D : point := (1, 3, 2)

noncomputable def angle_between_lines (p1 p2 p3 p4 : point) : ℝ := sorry

theorem angle_between_AB_CD :
  angle_between_lines A B C D = Real.arccos (2 * Real.sqrt 105 / 35) :=
sorry

end angle_between_AB_CD_l172_172378


namespace one_plus_x_pow_gt_one_plus_nx_l172_172169

theorem one_plus_x_pow_gt_one_plus_nx (x : ℝ) (n : ℕ) (hx1 : x > -1) (hx2 : x ≠ 0)
  (hn1 : n ≥ 2) : (1 + x)^n > 1 + n * x :=
sorry

end one_plus_x_pow_gt_one_plus_nx_l172_172169


namespace complex_seventh_root_identity_l172_172944

open Complex

theorem complex_seventh_root_identity (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 11 :=
by
  sorry

end complex_seventh_root_identity_l172_172944


namespace new_unemployment_rate_is_66_percent_l172_172937

theorem new_unemployment_rate_is_66_percent
  (initial_unemployment_rate : ℝ)
  (initial_employment_rate : ℝ)
  (u_increases_by_10_percent : initial_unemployment_rate * 1.1 = new_unemployment_rate)
  (e_decreases_by_15_percent : initial_employment_rate * 0.85 = new_employment_rate)
  (sum_is_100_percent : initial_unemployment_rate + initial_employment_rate = 100) :
  new_unemployment_rate = 66 :=
by
  sorry

end new_unemployment_rate_is_66_percent_l172_172937


namespace complement_union_l172_172879

variable (U : Set ℕ)
variable (A : Set ℕ)
variable (B : Set ℕ)

theorem complement_union : 
  U = {0, 1, 2, 3, 4} →
  (U \ A = {1, 2}) →
  B = {1, 3} →
  (A ∪ B = {0, 1, 3, 4}) :=
by
  intros hU hA hB
  sorry

end complement_union_l172_172879


namespace geom_seq_common_ratio_l172_172620

noncomputable def log_custom_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem geom_seq_common_ratio (a : ℝ) :
  let u₁ := a + log_custom_base 2 3
  let u₂ := a + log_custom_base 4 3
  let u₃ := a + log_custom_base 8 3
  u₂ / u₁ = u₃ / u₂ →
  u₂ / u₁ = 1 / 3 :=
by
  intro h
  sorry

end geom_seq_common_ratio_l172_172620


namespace solution_set_of_f_gt_0_range_of_m_l172_172644

noncomputable def f (x : ℝ) : ℝ := abs (2 * x - 1) - abs (x + 2)

theorem solution_set_of_f_gt_0 :
  {x : ℝ | f x > 0} = {x : ℝ | x < -1 / 3} ∪ {x | x > 3} :=
by sorry

theorem range_of_m (m : ℝ) :
  (∃ x_0 : ℝ, f x_0 + 2 * m^2 < 4 * m) ↔ -1 / 2 < m ∧ m < 5 / 2 :=
by sorry

end solution_set_of_f_gt_0_range_of_m_l172_172644


namespace relay_team_member_distance_l172_172743

theorem relay_team_member_distance (n_people : ℕ) (total_distance : ℕ)
  (h1 : n_people = 5) (h2 : total_distance = 150) : total_distance / n_people = 30 :=
by 
  sorry

end relay_team_member_distance_l172_172743


namespace closest_distance_l172_172756

theorem closest_distance (x y z : ℕ)
  (h1 : x + y = 10)
  (h2 : y + z = 13)
  (h3 : z + x = 11) :
  min x (min y z) = 4 :=
by
  -- Here you would provide the proof steps in Lean, but for the statement itself, we leave it as sorry.
  sorry

end closest_distance_l172_172756


namespace vanessa_score_l172_172524

-- Define the total score of the team
def total_points : ℕ := 60

-- Define the score of the seven other players
def other_players_points : ℕ := 7 * 4

-- Mathematics statement for proof
theorem vanessa_score : total_points - other_players_points = 32 :=
by
    sorry

end vanessa_score_l172_172524


namespace Emir_needs_more_money_l172_172923

theorem Emir_needs_more_money
  (cost_dictionary : ℝ)
  (cost_dinosaur_book : ℝ)
  (cost_cookbook : ℝ)
  (cost_science_kit : ℝ)
  (cost_colored_pencils : ℝ)
  (saved_amount : ℝ)
  (total_cost : ℝ := cost_dictionary + cost_dinosaur_book + cost_cookbook + cost_science_kit + cost_colored_pencils)
  (more_money_needed : ℝ := total_cost - saved_amount) :
  cost_dictionary = 5.50 →
  cost_dinosaur_book = 11.25 →
  cost_cookbook = 5.75 →
  cost_science_kit = 8.40 →
  cost_colored_pencils = 3.60 →
  saved_amount = 24.50 →
  more_money_needed = 10.00 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end Emir_needs_more_money_l172_172923


namespace molecular_weight_correct_l172_172041

-- Define the atomic weights of the elements.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element in the compound.
def number_of_C : ℕ := 7
def number_of_H : ℕ := 6
def number_of_O : ℕ := 2

-- Define the molecular weight calculation.
def molecular_weight : ℝ := 
  (number_of_C * atomic_weight_C) +
  (number_of_H * atomic_weight_H) +
  (number_of_O * atomic_weight_O)

-- Step to prove that molecular weight is equal to 122.118 g/mol.
theorem molecular_weight_correct : molecular_weight = 122.118 := by
  sorry

end molecular_weight_correct_l172_172041


namespace inequality_a_b_c_d_l172_172862

theorem inequality_a_b_c_d
  (a b c d : ℝ)
  (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : 0 ≤ d)
  (h₄ : a * b + b * c + c * d + d * a = 1) :
  (a ^ 3 / (b + c + d) + b ^ 3 / (c + d + a) + c ^ 3 / (a + b + d) + d ^ 3 / (a + b + c)) ≥ (1 / 3) :=
by
  sorry

end inequality_a_b_c_d_l172_172862


namespace find_H_over_G_l172_172570

variable (G H : ℤ)
variable (x : ℝ)

-- Conditions
def condition (G H : ℤ) (x : ℝ) : Prop :=
  x ≠ -7 ∧ x ≠ 0 ∧ x ≠ 6 ∧
  (↑G / (x + 7) + ↑H / (x * (x - 6)) = (x^2 - 3 * x + 15) / (x^3 + x^2 - 42 * x))

-- Theorem Statement
theorem find_H_over_G (G H : ℤ) (x : ℝ) (h : condition G H x) : (H : ℝ) / G = 15 / 7 :=
sorry

end find_H_over_G_l172_172570


namespace overall_percentage_supporting_increased_funding_l172_172430

-- Definitions for the conditions
def percent_of_men_supporting (percent_men_supporting : ℕ := 60) : ℕ := percent_men_supporting
def percent_of_women_supporting (percent_women_supporting : ℕ := 80) : ℕ := percent_women_supporting
def number_of_men_surveyed (men_surveyed : ℕ := 100) : ℕ := men_surveyed
def number_of_women_surveyed (women_surveyed : ℕ := 900) : ℕ := women_surveyed

-- Theorem: the overall percent of people surveyed who supported increased funding is 78%
theorem overall_percentage_supporting_increased_funding : 
  (percent_of_men_supporting * number_of_men_surveyed + percent_of_women_supporting * number_of_women_surveyed) / 
  (number_of_men_surveyed + number_of_women_surveyed) = 78 := 
sorry

end overall_percentage_supporting_increased_funding_l172_172430


namespace fg_of_1_eq_15_l172_172395

def f (x : ℝ) := 2 * x - 3
def g (x : ℝ) := (x + 2) ^ 2

theorem fg_of_1_eq_15 : f (g 1) = 15 :=
by
  sorry

end fg_of_1_eq_15_l172_172395


namespace add_congruence_mul_congruence_l172_172341

namespace ModularArithmetic

-- Define the congruence relation mod m
def is_congruent_mod (a b m : ℤ) : Prop := ∃ k : ℤ, a - b = k * m

-- Part (a): Proving a + c ≡ b + d (mod m)
theorem add_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a + c) (b + d) m :=
  sorry

-- Part (b): Proving a ⋅ c ≡ b ⋅ d (mod m)
theorem mul_congruence {a b c d m : ℤ}
  (h₁ : is_congruent_mod a b m)
  (h₂ : is_congruent_mod c d m) :
  is_congruent_mod (a * c) (b * d) m :=
  sorry

end ModularArithmetic

end add_congruence_mul_congruence_l172_172341


namespace find_k_multiple_l172_172522

theorem find_k_multiple (a b k : ℕ) (h1 : a = b + 5) (h2 : a + b = 13) 
  (h3 : 3 * (a + 7) = k * (b + 7)) : k = 4 := sorry

end find_k_multiple_l172_172522


namespace find_c_l172_172294

theorem find_c (c : ℝ) (h : (-c / 4) + (-c / 7) = 22) : c = -56 :=
by
  sorry

end find_c_l172_172294


namespace additional_airplanes_needed_l172_172374

theorem additional_airplanes_needed (total_current_airplanes : ℕ) (airplanes_per_row : ℕ) 
  (h_current_airplanes : total_current_airplanes = 37) 
  (h_airplanes_per_row : airplanes_per_row = 8) : 
  ∃ additional_airplanes : ℕ, additional_airplanes = 3 ∧ 
  ((total_current_airplanes + additional_airplanes) % airplanes_per_row = 0) :=
by
  sorry

end additional_airplanes_needed_l172_172374


namespace algebraic_expression_value_l172_172548

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -2) : 
  (2 * y - x) ^ 2 - 2 * x + 4 * y - 1 = 7 :=
by
  sorry

end algebraic_expression_value_l172_172548


namespace triangle_with_incircle_radius_one_has_sides_5_4_3_l172_172603

variable {a b c : ℕ} (h1 : a ≥ b ∧ b ≥ c)
variable (h2 : ∃ (a b c : ℕ), (a + b + c) / 2 * 1 = (a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))

theorem triangle_with_incircle_radius_one_has_sides_5_4_3 :
  a = 5 ∧ b = 4 ∧ c = 3 :=
by
    sorry

end triangle_with_incircle_radius_one_has_sides_5_4_3_l172_172603


namespace infinite_positive_integer_solutions_l172_172146

theorem infinite_positive_integer_solutions : ∃ (a b c : ℕ), (∃ k : ℕ, k > 0 ∧ a = k * (k^3 + 1990) ∧ b = (k^3 + 1990) ∧ c = (k^3 + 1990)) ∧ (a^3 + 1990 * b^3) = c^4 :=
sorry

end infinite_positive_integer_solutions_l172_172146


namespace basis_transformation_l172_172286

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem basis_transformation (h_basis : ∀ (v : V), ∃ (x y z : ℝ), v = x • a + y • b + z • c) :
  ∀ (v : V), ∃ (x y z : ℝ), v = x • (a + b) + y • (a - c) + z • b :=
by {
  sorry  -- to skip the proof steps for now
}

end basis_transformation_l172_172286


namespace total_stones_l172_172177

theorem total_stones (x : ℕ) 
  (h1 : x + 6 * x = x * 7 ∧ 7 * x + 6 * x = 2 * x) 
  (h2 : 2 * x = 7 * x - 10) 
  (h3 : 14 * x / 2 = 7 * x) :
  2 * 2 + 14 * 2 + 2 + 7 * 2 + 6 * 2 = 60 := 
by {
  sorry
}

end total_stones_l172_172177


namespace prince_spending_l172_172224

theorem prince_spending (CDs_total : ℕ) (CDs_10_percent : ℕ) (CDs_10_cost : ℕ) (CDs_5_cost : ℕ) 
  (Prince_10_fraction : ℚ) (Prince_5_fraction : ℚ) 
  (total_10_CDs : ℕ) (total_5_CDs : ℕ) (Prince_10_CDs : ℕ) (Prince_5_CDs : ℕ) (total_cost : ℕ) :
  CDs_total = 200 →
  CDs_10_percent = 40 →
  CDs_10_cost = 10 →
  CDs_5_cost = 5 →
  Prince_10_fraction = 1/2 →
  Prince_5_fraction = 1 →
  total_10_CDs = CDs_total * CDs_10_percent / 100 →
  total_5_CDs = CDs_total - total_10_CDs →
  Prince_10_CDs = total_10_CDs * Prince_10_fraction →
  Prince_5_CDs = total_5_CDs * Prince_5_fraction →
  total_cost = (Prince_10_CDs * CDs_10_cost) + (Prince_5_CDs * CDs_5_cost) →
  total_cost = 1000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9 h10 h11
  sorry

end prince_spending_l172_172224


namespace solve_for_x_l172_172197

theorem solve_for_x (x : ℝ) (h1 : x^2 - 5 * x = 0) (h2 : x ≠ 0) : x = 5 := sorry

end solve_for_x_l172_172197


namespace tip_percentage_l172_172759

def julie_food_cost : ℝ := 10
def letitia_food_cost : ℝ := 20
def anton_food_cost : ℝ := 30
def julie_tip : ℝ := 4
def letitia_tip : ℝ := 4
def anton_tip : ℝ := 4

theorem tip_percentage : 
  (julie_tip + letitia_tip + anton_tip) / (julie_food_cost + letitia_food_cost + anton_food_cost) * 100 = 20 :=
by
  sorry

end tip_percentage_l172_172759


namespace find_initial_books_each_l172_172819

variable (x : ℝ)
variable (sandy_books : ℝ := x)
variable (tim_books : ℝ := 2 * x + 33)
variable (benny_books : ℝ := 3 * x - 24)
variable (total_books : ℝ := 100)

theorem find_initial_books_each :
  sandy_books + tim_books + benny_books = total_books → x = 91 / 6 := by
  sorry

end find_initial_books_each_l172_172819


namespace xiao_gao_actual_score_l172_172025

-- Definitions from the conditions:
def standard_score : ℕ := 80
def xiao_gao_recorded_score : ℤ := 12

-- Proof problem statement:
theorem xiao_gao_actual_score : (standard_score : ℤ) + xiao_gao_recorded_score = 92 :=
by
  sorry

end xiao_gao_actual_score_l172_172025


namespace tangent_triangle_area_l172_172611

noncomputable def area_of_tangent_triangle : ℝ :=
  let f : ℝ → ℝ := fun x => Real.log x
  let f' : ℝ → ℝ := fun x => 1 / x
  let tangent_line : ℝ → ℝ := fun x => x - 1
  let x_intercept : ℝ := 1
  let y_intercept : ℝ := -1
  let base := 1
  let height := 1
  (1 / 2) * base * height

theorem tangent_triangle_area :
  area_of_tangent_triangle = 1 / 2 :=
sorry

end tangent_triangle_area_l172_172611


namespace find_two_digit_number_l172_172776

theorem find_two_digit_number (x y a b : ℕ) :
  10 * x + y + 46 = 10 * a + b →
  a * b = 6 →
  a + b = 14 →
  (x = 7 ∧ y = 7) ∨ (x = 8 ∧ y = 6) :=
by {
  sorry
}

end find_two_digit_number_l172_172776


namespace toluene_production_l172_172712

def molar_mass_benzene : ℝ := 78.11 -- The molar mass of benzene in g/mol
def benzene_mass : ℝ := 156 -- The mass of benzene in grams
def methane_moles : ℝ := 2 -- The moles of methane

-- Define the balanced chemical reaction
def balanced_reaction (benzene methanol toluene hydrogen : ℝ) : Prop :=
  benzene + methanol = toluene + hydrogen

-- The main theorem statement
theorem toluene_production (h1 : balanced_reaction benzene_mass methane_moles 1 1)
  (h2 : benzene_mass / molar_mass_benzene = 2) :
  ∃ toluene_moles : ℝ, toluene_moles = 2 :=
by
  sorry

end toluene_production_l172_172712


namespace elf_distribution_finite_l172_172730

theorem elf_distribution_finite (infinite_rubies : ℕ → ℕ) (infinite_sapphires : ℕ → ℕ) :
  (∃ n : ℕ, ∀ i j : ℕ, i < n → j < n → (infinite_rubies i > infinite_rubies j → infinite_sapphires i < infinite_sapphires j) ∧
  (infinite_rubies i ≥ infinite_rubies j → infinite_sapphires i < infinite_sapphires j)) ↔
  ∃ k : ℕ, ∀ j : ℕ, j < k :=
sorry

end elf_distribution_finite_l172_172730


namespace common_sale_days_in_july_l172_172495

def BookstoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (d % 4 = 0)

def ShoeStoreSaleDays (d : ℕ) : Prop :=
  (d ≥ 1) ∧ (d ≤ 31) ∧ (∃ k : ℕ, d = 2 + k * 7)

theorem common_sale_days_in_july : ∃! d, (BookstoreSaleDays d) ∧ (ShoeStoreSaleDays d) :=
by {
  sorry
}

end common_sale_days_in_july_l172_172495


namespace arccos_one_eq_zero_l172_172324

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := 
by sorry

end arccos_one_eq_zero_l172_172324


namespace cost_of_jeans_and_shirts_l172_172903

theorem cost_of_jeans_and_shirts 
  (S : ℕ) (J : ℕ) (X : ℕ)
  (hS : S = 18)
  (h2J3S : 2 * J + 3 * S = 76)
  (h3J2S : 3 * J + 2 * S = X) :
  X = 69 :=
by
  sorry

end cost_of_jeans_and_shirts_l172_172903


namespace Lee_surpasses_Hernandez_in_May_l172_172775

def monthly_totals_Hernandez : List ℕ :=
  [4, 8, 9, 5, 7, 6]

def monthly_totals_Lee : List ℕ :=
  [3, 9, 10, 6, 8, 8]

def cumulative_sum (lst : List ℕ) : List ℕ :=
  List.scanl (· + ·) 0 lst

noncomputable def cumulative_Hernandez := cumulative_sum monthly_totals_Hernandez
noncomputable def cumulative_Lee := cumulative_sum monthly_totals_Lee

-- Lean 4 statement asserting when Lee surpasses Hernandez in cumulative home runs
theorem Lee_surpasses_Hernandez_in_May :
  cumulative_Hernandez[3] < cumulative_Lee[3] :=
sorry

end Lee_surpasses_Hernandez_in_May_l172_172775


namespace time_b_is_54_l172_172798

-- Define the time A takes to complete the work
def time_a := 27

-- Define the time B takes to complete the work as twice the time A takes
def time_b := 2 * time_a

-- Prove that B takes 54 days to complete the work
theorem time_b_is_54 : time_b = 54 :=
by
  sorry

end time_b_is_54_l172_172798


namespace solve_quadratic_eq_l172_172500

theorem solve_quadratic_eq {x : ℝ} (h : x^2 - 5*x + 6 = 0) : x = 2 ∨ x = 3 :=
sorry

end solve_quadratic_eq_l172_172500


namespace area_of_overlap_l172_172191

def area_of_square_1 : ℝ := 1
def area_of_square_2 : ℝ := 4
def area_of_square_3 : ℝ := 9
def area_of_square_4 : ℝ := 16
def total_area_of_rectangle : ℝ := 27.5
def unshaded_area : ℝ := 1.5

def total_area_of_squares : ℝ := area_of_square_1 + area_of_square_2 + area_of_square_3 + area_of_square_4
def total_area_covered_by_squares : ℝ := total_area_of_rectangle - unshaded_area

theorem area_of_overlap :
  total_area_of_squares - total_area_covered_by_squares = 4 := 
sorry

end area_of_overlap_l172_172191


namespace necessary_condition_for_A_l172_172709

variable {x a : ℝ}

def A : Set ℝ := { x | (x - 2) / (x + 1) ≤ 0 }

theorem necessary_condition_for_A (x : ℝ) (h : x ∈ A) (ha : x ≥ a) : a ≤ -1 :=
sorry

end necessary_condition_for_A_l172_172709


namespace sum_possible_values_for_k_l172_172935

theorem sum_possible_values_for_k :
  ∃ (k_vals : Finset ℕ), (∀ j k : ℕ, 0 < j → 0 < k → (1 / j + 1 / k = 1 / 4) → k ∈ k_vals) ∧ 
    k_vals.sum id = 51 :=
by 
  sorry

end sum_possible_values_for_k_l172_172935


namespace prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l172_172890

-- Definitions
def groupA : List ℕ := [2, 4, 6]
def groupB : List ℕ := [3, 5]
def card_count_A : ℕ := groupA.length
def card_count_B : ℕ := groupB.length

-- Condition 1: Probability of drawing the card with number 2 from group A
def prob_draw_2_groupA : ℚ := 1 / card_count_A

-- Condition 2: Game Rule Outcomes
def is_multiple_of_3 (n : ℕ) : Bool := n % 3 == 0

def outcomes : List (ℕ × ℕ) := [(2, 3), (2, 5), (4, 3), (4, 5), (6, 3), (6, 5)]

def winning_outcomes_A : List (ℕ × ℕ) :=List.filter (λ p => is_multiple_of_3 (p.1 * p.2)) outcomes
def winning_outcomes_B : List (ℕ × ℕ) := List.filter (λ p => ¬ is_multiple_of_3 (p.1 * p.2)) outcomes

def prob_win_A : ℚ := winning_outcomes_A.length / outcomes.length
def prob_win_B : ℚ := winning_outcomes_B.length / outcomes.length

-- Proof problems
theorem prob_draw_2_groupA_is_one_third : prob_draw_2_groupA = 1 / 3 := sorry

theorem game_rule_is_unfair : prob_win_A ≠ prob_win_B := sorry

end prob_draw_2_groupA_is_one_third_game_rule_is_unfair_l172_172890


namespace part_I_part_II_l172_172261

-- Define the triangle and sides
structure Triangle :=
  (A B C : ℝ)   -- angles in the triangle
  (a b c : ℝ)   -- sides opposite to respective angles

-- Express given conditions in the problem
def conditions (T: Triangle) : Prop :=
  2 * (1 / (Real.tan T.A) + 1 / (Real.tan T.C)) = 1 / (Real.sin T.A) + 1 / (Real.sin T.C)

-- First theorem statement
theorem part_I (T : Triangle) : conditions T → (T.a + T.c = 2 * T.b) :=
sorry

-- Second theorem statement
theorem part_II (T : Triangle) : conditions T → (T.B ≤ Real.pi / 3) :=
sorry

end part_I_part_II_l172_172261


namespace number_of_true_propositions_is_one_l172_172507

theorem number_of_true_propositions_is_one :
  (¬ ∀ x : ℝ, x^4 > x^2) ∧
  (¬ (∀ (p q : Prop), ¬ (p ∧ q) → (¬ p ∧ ¬ q))) ∧
  (¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0) →
  1 = 1 :=
by
  sorry

end number_of_true_propositions_is_one_l172_172507


namespace value_of_f_750_l172_172533

theorem value_of_f_750 (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, 0 < x → 0 < y → f (x * y) = f x / y^2)
    (hf500 : f 500 = 4) :
    f 750 = 16 / 9 :=
sorry

end value_of_f_750_l172_172533


namespace find_interest_rate_l172_172908
noncomputable def annualInterestRate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  P * (1 + r / n)^(n * t) = A

theorem find_interest_rate :
  annualInterestRate 5000 6050.000000000001 1 2 0.1 :=
by
  -- The proof goes here
  sorry

end find_interest_rate_l172_172908


namespace solve_eq_l172_172848

theorem solve_eq (x a b : ℝ) (h₁ : x^2 + 10 * x = 34) (h₂ : a = 59) (h₃ : b = 5) :
  a + b = 64 :=
by {
  -- insert proof here, eventually leading to a + b = 64
  sorry
}

end solve_eq_l172_172848


namespace distance_between_cities_l172_172810

theorem distance_between_cities
  (S : ℤ)
  (h1 : ∀ x : ℤ, 0 ≤ x ∧ x ≤ S → (gcd x (S - x) = 1 ∨ gcd x (S - x) = 3 ∨ gcd x (S - x) = 13)) :
  S = 39 := 
sorry

end distance_between_cities_l172_172810


namespace powers_of_i_l172_172674

theorem powers_of_i (i : ℂ) (h1 : i^1 = i) (h2 : i^2 = -1) (h3 : i^3 = -i) (h4 : i^4 = 1) : 
  i^22 + i^222 = -2 :=
by {
  -- Proof will go here
  sorry
}

end powers_of_i_l172_172674


namespace quadratic_eq_has_real_root_l172_172870

theorem quadratic_eq_has_real_root (a b : ℝ) :
  ¬(∀ x : ℝ, x^2 + a * x + b ≠ 0) → ∃ x : ℝ, x^2 + a * x + b = 0 :=
by
  sorry

end quadratic_eq_has_real_root_l172_172870


namespace existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l172_172412

def is_rational_non_integer (a : ℚ) : Prop :=
  ¬ (∃ n : ℤ, a = n)

theorem existence_of_rational_solutions_a :
  ∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x + 8 * y = k1 ∧ 8 * x + 3 * y = k2 :=
sorry

theorem nonexistence_of_rational_solutions_b :
  ¬ (∃ x y : ℚ, is_rational_non_integer x ∧ is_rational_non_integer y ∧
    ∃ k1 k2 : ℤ, 19 * x^2 + 8 * y^2 = k1 ∧ 8 * x^2 + 3 * y^2 = k2) :=
sorry

end existence_of_rational_solutions_a_nonexistence_of_rational_solutions_b_l172_172412


namespace factorize_expr_l172_172111

theorem factorize_expr (x : ℝ) : x^3 - 16 * x = x * (x + 4) * (x - 4) :=
sorry

end factorize_expr_l172_172111


namespace band_gigs_count_l172_172558

-- Definitions of earnings per role and total earnings
def leadSingerEarnings := 30
def guitaristEarnings := 25
def bassistEarnings := 20
def drummerEarnings := 25
def keyboardistEarnings := 20
def backupSingerEarnings := 15
def totalEarnings := 2055

-- Calculate total per gig earnings
def totalPerGigEarnings :=
  leadSingerEarnings + guitaristEarnings + bassistEarnings + drummerEarnings + keyboardistEarnings + backupSingerEarnings

-- Statement to prove the number of gigs played is 15
theorem band_gigs_count :
  totalEarnings / totalPerGigEarnings = 15 := 
by { sorry }

end band_gigs_count_l172_172558


namespace gardener_total_expenses_l172_172239

theorem gardener_total_expenses
  (tulips carnations roses : ℕ)
  (cost_per_flower : ℕ)
  (h1 : tulips = 250)
  (h2 : carnations = 375)
  (h3 : roses = 320)
  (h4 : cost_per_flower = 2) :
  (tulips + carnations + roses) * cost_per_flower = 1890 := 
by
  sorry

end gardener_total_expenses_l172_172239


namespace correct_statement_is_d_l172_172843

/-- A definition for all the conditions given in the problem --/
def very_small_real_form_set : Prop := false
def smallest_natural_number_is_one : Prop := false
def sets_equal : Prop := false
def empty_set_subset_of_any_set : Prop := true

/-- The main statement to be proven --/
theorem correct_statement_is_d : (very_small_real_form_set = false) ∧ 
                                 (smallest_natural_number_is_one = false) ∧ 
                                 (sets_equal = false) ∧ 
                                 (empty_set_subset_of_any_set = true) :=
by
  sorry

end correct_statement_is_d_l172_172843


namespace connie_total_markers_l172_172768

def red_markers : ℕ := 5420
def blue_markers : ℕ := 3875
def green_markers : ℕ := 2910
def yellow_markers : ℕ := 6740

def total_markers : ℕ := red_markers + blue_markers + green_markers + yellow_markers

theorem connie_total_markers : total_markers = 18945 := by
  sorry

end connie_total_markers_l172_172768


namespace smaller_circle_radius_l172_172385

theorem smaller_circle_radius (r R : ℝ) (A1 A2 : ℝ) (hR : R = 5.0) (hA : A1 + A2 = 25 * Real.pi)
  (hap : A2 = A1 + 25 * Real.pi / 2) : r = 5 * Real.sqrt 2 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end smaller_circle_radius_l172_172385


namespace highest_avg_speed_2_to_3_l172_172601

-- Define the time periods and distances traveled in those periods
def distance_8_to_9 : ℕ := 50
def distance_9_to_10 : ℕ := 70
def distance_10_to_11 : ℕ := 60
def distance_2_to_3 : ℕ := 80
def distance_3_to_4 : ℕ := 40

-- Define the average speed calculation for each period
def avg_speed (distance : ℕ) (hours : ℕ) : ℕ := distance / hours

-- Proposition stating that the highest average speed is from 2 pm to 3 pm
theorem highest_avg_speed_2_to_3 : 
  avg_speed distance_2_to_3 1 > avg_speed distance_8_to_9 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_9_to_10 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_10_to_11 1 ∧ 
  avg_speed distance_2_to_3 1 > avg_speed distance_3_to_4 1 := 
by 
  sorry

end highest_avg_speed_2_to_3_l172_172601


namespace compare_expressions_l172_172458

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l172_172458


namespace rain_difference_l172_172641

theorem rain_difference
    (rain_monday : ℕ → ℝ)
    (rain_tuesday : ℕ → ℝ)
    (rain_wednesday : ℕ → ℝ)
    (rain_thursday : ℕ → ℝ)
    (h_monday : ∀ n : ℕ, n = 10 → rain_monday n = 1.25)
    (h_tuesday : ∀ n : ℕ, n = 12 → rain_tuesday n = 2.15)
    (h_wednesday : ∀ n : ℕ, n = 8 → rain_wednesday n = 1.60)
    (h_thursday : ∀ n : ℕ, n = 6 → rain_thursday n = 2.80) :
    let total_rain_monday := 10 * 1.25
    let total_rain_tuesday := 12 * 2.15
    let total_rain_wednesday := 8 * 1.60
    let total_rain_thursday := 6 * 2.80
    (total_rain_tuesday + total_rain_thursday) - (total_rain_monday + total_rain_wednesday) = 17.3 :=
by
  sorry

end rain_difference_l172_172641


namespace at_least_one_miss_l172_172896

variables (p q : Prop)

-- Proposition stating the necessary and sufficient condition.
theorem at_least_one_miss : ¬(p ∧ q) ↔ (¬p ∨ ¬q) :=
by sorry

end at_least_one_miss_l172_172896


namespace salt_quantity_l172_172375

-- Conditions translated to Lean definitions
def cost_of_sugar_per_kg : ℝ := 1.50
def total_cost_sugar_2kg_and_salt (x : ℝ) : ℝ := 5.50
def total_cost_sugar_3kg_and_1kg_salt : ℝ := 5.00

-- Theorem statement
theorem salt_quantity (x : ℝ) : 
  2 * cost_of_sugar_per_kg + x * cost_of_sugar_per_kg / 3 = total_cost_sugar_2kg_and_salt x 
  → 3 * cost_of_sugar_per_kg + x = total_cost_sugar_3kg_and_1kg_salt 
  → x = 5 := 
sorry

end salt_quantity_l172_172375


namespace problem_l172_172809

theorem problem (y : ℝ) (h : 7 * y^2 + 6 = 5 * y + 14) : (14 * y - 2)^2 = 258 := by
  sorry

end problem_l172_172809


namespace find_x_l172_172609

-- Given condition
def condition (x : ℝ) : Prop := 3 * x - 5 * x + 8 * x = 240

-- Statement (problem to prove)
theorem find_x (x : ℝ) (h : condition x) : x = 40 :=
by 
  sorry

end find_x_l172_172609


namespace sum_and_round_to_nearest_ten_l172_172102

/-- A function to round a number to the nearest ten -/
def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + 10 - n % 10

/-- The sum of 54 and 29 rounded to the nearest ten is 80 -/
theorem sum_and_round_to_nearest_ten : round_to_nearest_ten (54 + 29) = 80 :=
by
  sorry

end sum_and_round_to_nearest_ten_l172_172102


namespace floor_neg_seven_four_is_neg_two_l172_172969

noncomputable def floor_neg_seven_four : Int :=
  Int.floor (-7 / 4)

theorem floor_neg_seven_four_is_neg_two : floor_neg_seven_four = -2 := by
  sorry

end floor_neg_seven_four_is_neg_two_l172_172969


namespace unique_identity_function_l172_172205

theorem unique_identity_function (f : ℕ+ → ℕ+) :
  (∀ (x y : ℕ+), 
    let a := x 
    let b := f y 
    let c := f (y + f x - 1)
    a + b > c ∧ a + c > b ∧ b + c > a) →
  (∀ x, f x = x) :=
by
  intro h
  sorry

end unique_identity_function_l172_172205


namespace rationalize_denominator_l172_172782

theorem rationalize_denominator (h : ∀ x: ℝ, x = 1 / (Real.sqrt 3 - 2)) : 
    1 / (Real.sqrt 3 - 2) = - Real.sqrt 3 - 2 :=
by
  sorry

end rationalize_denominator_l172_172782


namespace boat_speed_still_water_l172_172640

theorem boat_speed_still_water : 
  ∀ (b s : ℝ), (b + s = 11) → (b - s = 5) → b = 8 := 
by 
  intros b s h1 h2
  sorry

end boat_speed_still_water_l172_172640


namespace tony_bread_slices_left_l172_172131

def num_slices_left (initial_slices : ℕ) (d1 : ℕ) (d2 : ℕ) : ℕ :=
  initial_slices - (d1 + d2)

theorem tony_bread_slices_left :
  num_slices_left 22 (5 * 2) (2 * 2) = 8 := by
  sorry

end tony_bread_slices_left_l172_172131


namespace general_term_is_correct_l172_172299

variable (a : ℕ → ℤ)
variable (n : ℕ)

def is_arithmetic_sequence := ∃ d a₁, ∀ n, a n = a₁ + d * (n - 1)

axiom a_10_eq_30 : a 10 = 30
axiom a_20_eq_50 : a 20 = 50

noncomputable def general_term (n : ℕ) : ℤ := 2 * n + 10

theorem general_term_is_correct (a: ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : a 10 = 30)
  (h3 : a 20 = 50)
  : ∀ n, a n = general_term n :=
sorry

end general_term_is_correct_l172_172299


namespace solve_fractional_equation_l172_172705

theorem solve_fractional_equation :
  {x : ℝ | 1 / (x^2 + 8 * x - 6) + 1 / (x^2 + 5 * x - 6) + 1 / (x^2 - 14 * x - 6) = 0}
  = {3, -2, -6, 1} :=
by
  sorry

end solve_fractional_equation_l172_172705


namespace geometric_sequence_common_ratio_l172_172894

variable {α : Type*} [LinearOrderedField α]

def is_geometric_sequence (a : ℕ → α) (q : α) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_common_ratio
  (a : ℕ → α)
  (q : α)
  (h1 : is_geometric_sequence a q)
  (h2 : a 3 = 6)
  (h3 : a 0 + a 1 + a 2 = 18) :
  q = 1 ∨ q = - (1 / 2) := 
sorry

end geometric_sequence_common_ratio_l172_172894


namespace spaceship_distance_traveled_l172_172817

theorem spaceship_distance_traveled (d_ex : ℝ) (d_xy : ℝ) (d_total : ℝ) :
  d_ex = 0.5 → d_xy = 0.1 → d_total = 0.7 → (d_total - (d_ex + d_xy)) = 0.1 :=
by
  intros h1 h2 h3
  sorry

end spaceship_distance_traveled_l172_172817


namespace average_seven_numbers_l172_172673

theorem average_seven_numbers (A B C D E F G : ℝ) 
  (h1 : (A + B + C + D) / 4 = 4)
  (h2 : (D + E + F + G) / 4 = 4)
  (hD : D = 11) : 
  (A + B + C + D + E + F + G) / 7 = 3 :=
by
  sorry

end average_seven_numbers_l172_172673


namespace eq_fractions_l172_172634

theorem eq_fractions : 
  (1 + 1 / (1 + 1 / (1 + 1 / 2))) = 8 / 5 := 
  sorry

end eq_fractions_l172_172634


namespace circle_line_intersection_points_l172_172408

noncomputable def radius : ℝ := 6
noncomputable def distance : ℝ := 5

theorem circle_line_intersection_points :
  radius > distance -> number_of_intersection_points = 2 := 
by
  sorry

end circle_line_intersection_points_l172_172408


namespace nth_term_sequence_sum_first_n_terms_l172_172452

def a_n (n : ℕ) : ℕ :=
  (2 * n - 1) * (2 * n + 2)

def S_n (n : ℕ) : ℚ :=
  4 * (n * (n + 1) * (2 * n + 1)) / 6 + n * (n + 1) - 2 * n

theorem nth_term_sequence (n : ℕ) : a_n n = 4 * n^2 + 2 * n - 2 :=
  sorry

theorem sum_first_n_terms (n : ℕ) : S_n n = (4 * n^3 + 9 * n^2 - n) / 3 :=
  sorry

end nth_term_sequence_sum_first_n_terms_l172_172452


namespace part_I_min_value_part_II_a_range_l172_172118

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x - a) - abs (x + 3)

theorem part_I_min_value (x : ℝ) : f x 1 ≥ -7 / 2 :=
by sorry 

theorem part_II_a_range (x a : ℝ) (hx : 0 ≤ x) (hx' : x ≤ 3) (hf : f x a ≤ 4) : -4 ≤ a ∧ a ≤ 7 :=
by sorry

end part_I_min_value_part_II_a_range_l172_172118


namespace readers_scifi_l172_172748

variable (S L B T : ℕ)

-- Define conditions given in the problem
def totalReaders := 650
def literaryReaders := 550
def bothReaders := 150

-- Define the main problem to prove
theorem readers_scifi (S L B T : ℕ) (hT : T = totalReaders) (hL : L = literaryReaders) (hB : B = bothReaders) (hleq : T = S + L - B) : S = 250 :=
by
  -- Insert proof here
  sorry

end readers_scifi_l172_172748


namespace total_miles_traveled_l172_172107

noncomputable def initial_fee : ℝ := 2.0
noncomputable def charge_per_2_5_mile : ℝ := 0.35
noncomputable def total_charge : ℝ := 5.15

theorem total_miles_traveled :
  ∃ (miles : ℝ), total_charge = initial_fee + (charge_per_2_5_mile * miles * (5 / 2)) ∧ miles = 3.6 :=
by
  sorry

end total_miles_traveled_l172_172107


namespace find_second_number_l172_172852

theorem find_second_number (X : ℝ) : 
  (0.6 * 50 - 0.3 * X = 27) → X = 10 :=
by
  sorry

end find_second_number_l172_172852


namespace gcd_288_123_l172_172106

-- Define the conditions
def cond1 : 288 = 2 * 123 + 42 := by sorry
def cond2 : 123 = 2 * 42 + 39 := by sorry
def cond3 : 42 = 39 + 3 := by sorry
def cond4 : 39 = 13 * 3 := by sorry

-- Prove that GCD of 288 and 123 is 3
theorem gcd_288_123 : Nat.gcd 288 123 = 3 := by
  sorry

end gcd_288_123_l172_172106


namespace cos2_alpha_add_sin2_alpha_eq_eight_over_five_l172_172745

theorem cos2_alpha_add_sin2_alpha_eq_eight_over_five (x y : ℝ) (r : ℝ) (α : ℝ) 
(hx : x = 2) 
(hy : y = 1)
(hr : r = Real.sqrt (x^2 + y^2))
(hcos : Real.cos α = x / r)
(hsin : Real.sin α = y / r) :
  Real.cos α ^ 2 + Real.sin (2 * α) = 8 / 5 :=
sorry

end cos2_alpha_add_sin2_alpha_eq_eight_over_five_l172_172745


namespace infinite_triangles_with_sides_x_y_10_l172_172613

theorem infinite_triangles_with_sides_x_y_10 (x y : Nat) (hx : 0 < x) (hy : 0 < y) : 
  (∃ n : Nat, n > 5 ∧ ∀ m ≥ n, ∃ x y : Nat, 0 < x ∧ 0 < y ∧ x + y > 10 ∧ x + 10 > y ∧ y + 10 > x) :=
sorry

end infinite_triangles_with_sides_x_y_10_l172_172613


namespace exists_unique_c_l172_172988

theorem exists_unique_c (a : ℝ) (h₁ : 1 < a) :
  (∃ (c : ℝ), ∀ (x : ℝ), x ∈ Set.Icc a (2 * a) → ∃ (y : ℝ), y ∈ Set.Icc a (a ^ 2) ∧ (Real.log x / Real.log a + Real.log y / Real.log a = c)) ↔ a = 2 :=
by
  sorry

end exists_unique_c_l172_172988


namespace trip_time_difference_l172_172579

def travel_time (distance speed : ℕ) : ℕ :=
  distance / speed

theorem trip_time_difference
  (speed : ℕ)
  (speed_pos : 0 < speed)
  (distance1 : ℕ)
  (distance2 : ℕ)
  (time_difference : ℕ)
  (h1 : distance1 = 540)
  (h2 : distance2 = 600)
  (h_speed : speed = 60)
  (h_time_diff : time_difference = (travel_time distance2 speed) - (travel_time distance1 speed) * 60)
  : time_difference = 60 :=
by
  sorry

end trip_time_difference_l172_172579


namespace min_n_such_that_no_more_possible_l172_172223

-- Define a seven-cell corner as a specific structure within the grid
inductive Corner
| cell7 : Corner

-- Function to count the number of cells clipped out by n corners
def clipped_cells (n : ℕ) : ℕ := 7 * n

-- Statement to be proven
theorem min_n_such_that_no_more_possible (n : ℕ) (h_n : n ≥ 3) (h_max : n < 4) :
  ¬ ∃ k : ℕ, k > n ∧ clipped_cells k ≤ 64 :=
by {
  sorry -- Proof goes here
}

end min_n_such_that_no_more_possible_l172_172223


namespace opposite_of_negative_fraction_l172_172422

theorem opposite_of_negative_fraction : -(- (1/2023 : ℚ)) = 1/2023 := 
sorry

end opposite_of_negative_fraction_l172_172422


namespace prop_for_real_l172_172293

theorem prop_for_real (a : ℝ) : (∀ x : ℝ, x^2 + 2 * x - a > 0) → a < -1 :=
by
  sorry

end prop_for_real_l172_172293


namespace solve_linear_system_l172_172828

theorem solve_linear_system :
  ∃ x y z : ℝ, 
    (2 * x + y + z = -1) ∧ 
    (3 * y - z = -1) ∧ 
    (3 * x + 2 * y + 3 * z = -5) ∧ 
    (x = 1) ∧ 
    (y = -1) ∧ 
    (z = -2) :=
by
  sorry

end solve_linear_system_l172_172828


namespace no_solution_to_inequalities_l172_172179

theorem no_solution_to_inequalities :
  ∀ (x y z t : ℝ), 
    ¬ (|x| > |y - z + t| ∧
       |y| > |x - z + t| ∧
       |z| > |x - y + t| ∧
       |t| > |x - y + z|) :=
by
  intro x y z t
  sorry

end no_solution_to_inequalities_l172_172179


namespace sequence_geometric_l172_172079

theorem sequence_geometric (a : ℕ → ℕ) (n : ℕ) (hn : 0 < n):
  (a 1 = 1) →
  (∀ n, 0 < n → a (n + 1) = 2 * a n) →
  a n = 2^(n-1) :=
by
  intros
  sorry

end sequence_geometric_l172_172079


namespace hiker_displacement_l172_172490

theorem hiker_displacement :
  let start_point := (0, 0)
  let move_east := (24, 0)
  let move_north := (0, 20)
  let move_west := (-7, 0)
  let move_south := (0, -9)
  let final_position := (start_point.1 + move_east.1 + move_west.1, start_point.2 + move_north.2 + move_south.2)
  let distance_from_start := Real.sqrt (final_position.1^2 + final_position.2^2)
  distance_from_start = Real.sqrt 410
:= by 
  sorry

end hiker_displacement_l172_172490


namespace min_value_of_F_on_neg_infinity_l172_172858

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- Define the conditions provided in the problem
axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom g_odd : ∀ x : ℝ, g (-x) = - g x
noncomputable def F (x : ℝ) := a * f x + b * g x + 2
axiom F_max_on_pos : ∃ x ∈ (Set.Ioi 0), F x = 5

-- Prove the conclusion of the problem
theorem min_value_of_F_on_neg_infinity : ∃ y ∈ (Set.Iio 0), F y = -1 :=
sorry

end min_value_of_F_on_neg_infinity_l172_172858


namespace problem1_problem2_problem2_zero_problem2_neg_l172_172694

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + a*x + a
def g (a x : ℝ) : ℝ := a*(f a x) - a^2*(x + 1) - 2*x

-- Problem 1
theorem problem1 (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < x2 ∧ x2 < 1 ∧ f a x1 - x1 = 0 ∧ f a x2 - x2 = 0) →
  (0 < a ∧ a < 3 - 2*Real.sqrt 2) :=
sorry

-- Problem 2
theorem problem2 (a : ℝ) (h1 : a > 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ 
    if a < 1 then a-2 
    else -1/a) :=
sorry

theorem problem2_zero (h2 : a = 0) : 
  g a 1 = -2 :=
sorry

theorem problem2_neg (a : ℝ) (h3 : a < 0) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → g a x ≥ a - 2) :=
sorry

end problem1_problem2_problem2_zero_problem2_neg_l172_172694


namespace sum_remainder_l172_172825

theorem sum_remainder (a b c : ℕ) (h1 : a % 30 = 12) (h2 : b % 30 = 9) (h3 : c % 30 = 15) :
  (a + b + c) % 30 = 6 := 
sorry

end sum_remainder_l172_172825


namespace trioball_play_time_l172_172432

theorem trioball_play_time (total_duration : ℕ) (num_children : ℕ) (players_at_a_time : ℕ) 
  (equal_play_time : ℕ) (H1 : total_duration = 120) (H2 : num_children = 3) (H3 : players_at_a_time = 2)
  (H4 : equal_play_time = 240 / num_children)
  : equal_play_time = 80 := 
by 
  sorry

end trioball_play_time_l172_172432


namespace gcd_of_consecutive_digit_sums_is_1111_l172_172925

theorem gcd_of_consecutive_digit_sums_is_1111 (p q r s : ℕ) (hc : q = p+1 ∧ r = p+2 ∧ s = p+3) :
  ∃ d, d = 1111 ∧ ∀ n : ℕ, n = (1000 * p + 100 * q + 10 * r + s) + (1000 * s + 100 * r + 10 * q + p) → d ∣ n := by
  use 1111
  sorry

end gcd_of_consecutive_digit_sums_is_1111_l172_172925


namespace maximize_area_l172_172535

noncomputable def max_area : ℝ :=
  let l := 100
  let w := 100
  l * w

theorem maximize_area (l w : ℝ) (h1 : 2 * l + 2 * w = 400) (h2 : l ≥ 100) (h3 : w ≥ 50) :
  (l * w ≤ 10000) :=
sorry

end maximize_area_l172_172535


namespace smallest_solution_equation_l172_172778

noncomputable def equation (x : ℝ) : ℝ :=
  (3*x / (x-3)) + ((3*x^2 - 45) / x) + 3

theorem smallest_solution_equation : 
  ∃ x : ℝ, equation x = 14 ∧ x = (1 - Real.sqrt 649) / 12 :=
sorry

end smallest_solution_equation_l172_172778


namespace range_of_a_l172_172733

def quadratic_function (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, quadratic_function a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by
  sorry

end range_of_a_l172_172733


namespace initial_pages_l172_172670

/-
Given:
1. Sammy uses 25% of the pages for his science project.
2. Sammy uses another 10 pages for his math homework.
3. There are 80 pages remaining in the pad.

Prove that the initial number of pages in the pad (P) is 120.
-/

theorem initial_pages (P : ℝ) (h1 : P * 0.25 + 10 + 80 = P) : 
  P = 120 :=
by 
  sorry

end initial_pages_l172_172670


namespace distance_downstream_in_12min_l172_172789

-- Define the given constants
def boat_speed_still_water : ℝ := 15  -- km/hr
def current_speed : ℝ := 3  -- km/hr
def time_minutes : ℝ := 12  -- minutes

-- Prove the distance traveled downstream in 12 minutes
theorem distance_downstream_in_12min
  (b_velocity_still : ℝ)
  (c_velocity : ℝ)
  (time_m : ℝ)
  (h1 : b_velocity_still = boat_speed_still_water)
  (h2 : c_velocity = current_speed)
  (h3 : time_m = time_minutes) :
  let effective_speed := b_velocity_still + c_velocity
  let effective_speed_km_per_min := effective_speed / 60
  let distance := effective_speed_km_per_min * time_m
  distance = 3.6 :=
by
  sorry

end distance_downstream_in_12min_l172_172789


namespace second_train_speed_l172_172771

variable (t v : ℝ)

-- Defining the first condition: 20t = vt + 55
def condition1 : Prop := 20 * t = v * t + 55

-- Defining the second condition: 20t + vt = 495
def condition2 : Prop := 20 * t + v * t = 495

-- Prove that the speed of the second train is 16 km/hr under given conditions
theorem second_train_speed : ∃ t : ℝ, condition1 t 16 ∧ condition2 t 16 := sorry

end second_train_speed_l172_172771


namespace find_k_l172_172942

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : ∀ x ∈ Set.Icc (2 : ℝ) 4, y = k / x → y ≥ 5) : k = 20 :=
sorry

end find_k_l172_172942


namespace ratio_five_to_one_l172_172026

theorem ratio_five_to_one (x : ℕ) (h : 5 * 12 = x) : x = 60 :=
by
  sorry

end ratio_five_to_one_l172_172026


namespace zeros_of_f_x_minus_1_l172_172536

def f (x : ℝ) : ℝ := x^2 - 1

theorem zeros_of_f_x_minus_1 :
  (f (0 - 1) = 0) ∧ (f (2 - 1) = 0) :=
by
  sorry

end zeros_of_f_x_minus_1_l172_172536


namespace total_visitors_400_l172_172064

variables (V E U : ℕ)

def visitors_did_not_enjoy_understand (V : ℕ) := 3 * V / 4 + 100 = V
def visitors_enjoyed_equal_understood (E U : ℕ) := E = U
def total_visitors_satisfy_34 (V E : ℕ) := 3 * V / 4 = E

theorem total_visitors_400
  (h1 : ∀ V, visitors_did_not_enjoy_understand V)
  (h2 : ∀ E U, visitors_enjoyed_equal_understood E U)
  (h3 : ∀ V E, total_visitors_satisfy_34 V E) :
  V = 400 :=
by { sorry }

end total_visitors_400_l172_172064


namespace unknown_road_length_l172_172288

/-
  Given the lengths of four roads and the Triangle Inequality condition, 
  prove the length of the fifth road.
  Given lengths: a = 10 km, b = 5 km, c = 8 km, d = 21 km.
-/

theorem unknown_road_length
  (a b c d : ℕ) (h0 : a = 10) (h1 : b = 5) (h2 : c = 8) (h3 : d = 21)
  (x : ℕ) :
  2 < x ∧ x < 18 ∧ 16 < x ∧ x < 26 → x = 17 :=
by
  intros
  sorry

end unknown_road_length_l172_172288


namespace sqrt_meaningful_condition_l172_172556

theorem sqrt_meaningful_condition (x : ℝ) : (2 * x + 6 >= 0) ↔ (x >= -3) := by
  sorry

end sqrt_meaningful_condition_l172_172556


namespace percentage_decrease_is_14_percent_l172_172151

-- Definitions based on conditions
def original_price_per_pack : ℚ := 7 / 3
def new_price_per_pack : ℚ := 8 / 4

-- Statement to prove that percentage decrease is 14%
theorem percentage_decrease_is_14_percent :
  ((original_price_per_pack - new_price_per_pack) / original_price_per_pack) * 100 = 14 := by
  sorry

end percentage_decrease_is_14_percent_l172_172151


namespace average_length_of_strings_l172_172984

-- Define lengths of the three strings
def length1 := 4  -- length of the first string in inches
def length2 := 5  -- length of the second string in inches
def length3 := 7  -- length of the third string in inches

-- Define the total length and number of strings
def total_length := length1 + length2 + length3
def num_strings := 3

-- Define the average length calculation
def average_length := total_length / num_strings

-- The proof statement
theorem average_length_of_strings : average_length = 16 / 3 := 
by 
  sorry

end average_length_of_strings_l172_172984


namespace dihedral_angles_pyramid_l172_172342

noncomputable def dihedral_angles (a b : ℝ) : ℝ × ℝ :=
  let alpha := Real.arccos ((a * Real.sqrt 3) / Real.sqrt (4 * b ^ 2 - a ^ 2))
  let gamma := 2 * Real.arctan (b / Real.sqrt (4 * b ^ 2 - a ^ 2))
  (alpha, gamma)

theorem dihedral_angles_pyramid (a b alpha gamma : ℝ) (h1 : a > 0) (h2 : b > 0) :
  dihedral_angles a b = (alpha, gamma) :=
sorry

end dihedral_angles_pyramid_l172_172342


namespace cos_A_side_c_l172_172012

-- helper theorem for cosine rule usage
theorem cos_A (a b c : ℝ) (cosA cosB cosC : ℝ) (h : 3 * a * cosA = c * cosB + b * cosC) : cosA = 1 / 3 :=
by
  sorry

-- main statement combining conditions 1 and 2 with side value results
theorem side_c (a b c : ℝ) (cosA cosB cosC : ℝ) (h1 : 3 * a * cosA = c * cosB + b * cosC) (h2 : cosB + cosC = 0) (h3 : a = 1) : c = 2 :=
by
  have h_cosA : cosA = 1 / 3 := cos_A a b c cosA cosB cosC h1
  sorry

end cos_A_side_c_l172_172012


namespace jimmy_change_l172_172401

noncomputable def change_back (pen_cost notebook_cost folder_cost highlighter_cost sticky_notes_cost total_paid discount tax : ℝ) : ℝ :=
  let total_before_discount := (5 * pen_cost) + (6 * notebook_cost) + (4 * folder_cost) + (3 * highlighter_cost) + (2 * sticky_notes_cost)
  let total_after_discount := total_before_discount * (1 - discount)
  let final_total := total_after_discount * (1 + tax)
  (total_paid - final_total)

theorem jimmy_change :
  change_back 1.65 3.95 4.35 2.80 1.75 150 0.25 0.085 = 100.16 :=
by
  sorry

end jimmy_change_l172_172401


namespace points_opposite_sides_of_line_l172_172052

theorem points_opposite_sides_of_line (a : ℝ) :
  (1 + 1 - a) * (2 - 1 - a) < 0 ↔ 1 < a ∧ a < 2 :=
by sorry

end points_opposite_sides_of_line_l172_172052


namespace solve_inequality_system_l172_172970

theorem solve_inequality_system (x : ℝ) :
  (4 * x + 5 > x - 1) ∧ ((3 * x - 1) / 2 < x) ↔ (-2 < x ∧ x < 1) :=
by
  sorry

end solve_inequality_system_l172_172970


namespace NY_Mets_fans_count_l172_172477

noncomputable def NY_Yankees_fans (M: ℝ) : ℝ := (3/2) * M
noncomputable def Boston_Red_Sox_fans (M: ℝ) : ℝ := (5/4) * M
noncomputable def LA_Dodgers_fans (R: ℝ) : ℝ := (2/7) * R

theorem NY_Mets_fans_count :
  ∃ M : ℕ, let Y := NY_Yankees_fans M
           let R := Boston_Red_Sox_fans M
           let D := LA_Dodgers_fans R
           Y + M + R + D = 780 ∧ M = 178 :=
by
  sorry

end NY_Mets_fans_count_l172_172477


namespace mike_picked_32_limes_l172_172875

theorem mike_picked_32_limes (total_limes : ℕ) (alyssa_limes : ℕ) (mike_limes : ℕ) 
  (h1 : total_limes = 57) (h2 : alyssa_limes = 25) (h3 : mike_limes = total_limes - alyssa_limes) : 
  mike_limes = 32 :=
by
  sorry

end mike_picked_32_limes_l172_172875


namespace caitlin_age_l172_172654

theorem caitlin_age (aunt_anna_age : ℕ) (brianna_age : ℕ) (caitlin_age : ℕ) 
  (h1 : aunt_anna_age = 60)
  (h2 : brianna_age = aunt_anna_age / 3)
  (h3 : caitlin_age = brianna_age - 7)
  : caitlin_age = 13 :=
by
  sorry

end caitlin_age_l172_172654


namespace numbers_not_crossed_out_l172_172697

/-- Total numbers between 1 and 90 after crossing out multiples of 3 and 5 is 48. -/
theorem numbers_not_crossed_out : 
  let n := 90 
  let multiples_of_3 := n / 3 
  let multiples_of_5 := n / 5 
  let multiples_of_15 := n / 15 
  let crossed_out := multiples_of_3 + multiples_of_5 - multiples_of_15
  n - crossed_out = 48 :=
by {
  sorry
}

end numbers_not_crossed_out_l172_172697


namespace smallest_coterminal_angle_pos_radians_l172_172053

theorem smallest_coterminal_angle_pos_radians :
  ∀ (θ : ℝ), θ = -560 * (π / 180) → ∃ α : ℝ, α > 0 ∧ α = (8 * π) / 9 ∧ (∃ k : ℤ, θ + 2 * k * π = α) :=
by
  sorry

end smallest_coterminal_angle_pos_radians_l172_172053


namespace solve_inequality_l172_172717

theorem solve_inequality {x : ℝ} :
  (3 / (5 - 3 * x) > 1) ↔ (2/3 < x ∧ x < 5/3) :=
by
  sorry

end solve_inequality_l172_172717


namespace speed_of_man_in_still_water_l172_172656

variable (v_m v_s : ℝ)

theorem speed_of_man_in_still_water
  (h1 : (v_m + v_s) * 4 = 24)
  (h2 : (v_m - v_s) * 5 = 20) :
  v_m = 5 := 
sorry

end speed_of_man_in_still_water_l172_172656


namespace num_ordered_pairs_c_d_l172_172202

def is_solution (c d x y : ℤ) : Prop :=
  c * x + d * y = 2 ∧ x^2 + y^2 = 65

theorem num_ordered_pairs_c_d : 
  ∃ (S : Finset (ℤ × ℤ)), S.card = 136 ∧ 
  ∀ (c d : ℤ), (c, d) ∈ S ↔ ∃ (x y : ℤ), is_solution c d x y :=
sorry

end num_ordered_pairs_c_d_l172_172202


namespace math_books_count_l172_172855

theorem math_books_count (M H : ℕ) (h1 : M + H = 90) (h2 : 4 * M + 5 * H = 396) : M = 54 :=
sorry

end math_books_count_l172_172855


namespace f_10_half_l172_172746

noncomputable def f (x : ℝ) : ℝ := x^2 / (2 * x + 1)
noncomputable def fn (n : ℕ) (x : ℝ) : ℝ :=
  match n with
  | 0     => x
  | n + 1 => f (fn n x)

theorem f_10_half :
  fn 10 (1 / 2) = 1 / (3 ^ 1024 - 1) :=
sorry

end f_10_half_l172_172746


namespace distance_to_cut_pyramid_l172_172649

theorem distance_to_cut_pyramid (V A V1 : ℝ) (h1 : V > 0) (h2 : A > 0) :
  ∃ d : ℝ, d = (3 / A) * (V - (V^2 * (V - V1))^(1 / 3)) :=
by
  sorry

end distance_to_cut_pyramid_l172_172649


namespace inequality_proof_l172_172322

theorem inequality_proof (x y z : ℝ) (hx : x < 0) (hy : y < 0) (hz : z < 0) :
    (x * y * z) / ((1 + 5 * x) * (4 * x + 3 * y) * (5 * y + 6 * z) * (z + 18)) ≤ (1 : ℝ) / 5120 := 
by
  sorry

end inequality_proof_l172_172322


namespace eccentricity_condition_l172_172100

theorem eccentricity_condition (m : ℝ) (h : 0 < m) : 
  (m < (4 / 3) ∨ m > (3 / 4)) ↔ ((1 - m) > (1 / 4) ∨ ((m - 1) / m) > (1 / 4)) :=
by
  sorry

end eccentricity_condition_l172_172100


namespace donation_percentage_l172_172765

noncomputable def income : ℝ := 266666.67
noncomputable def remaining_income : ℝ := 0.25 * income
noncomputable def final_amount : ℝ := 40000

theorem donation_percentage :
  ∃ D : ℝ, D = 40 /\ (1 - D / 100) * remaining_income = final_amount :=
by
  sorry

end donation_percentage_l172_172765


namespace base_b_of_256_has_4_digits_l172_172266

theorem base_b_of_256_has_4_digits : ∃ (b : ℕ), b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 5 :=
by
  sorry

end base_b_of_256_has_4_digits_l172_172266


namespace evaluate_expression_l172_172949

theorem evaluate_expression :
  (3 ^ 1002 + 7 ^ 1003) ^ 2 - (3 ^ 1002 - 7 ^ 1003) ^ 2 = 56 * 10 ^ 1003 :=
by
  sorry

end evaluate_expression_l172_172949


namespace middle_card_number_is_6_l172_172123

noncomputable def middle_card_number : ℕ :=
  6

theorem middle_card_number_is_6 (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 17)
  (casey_cannot_determine : ∀ (x : ℕ), (a = x) → ∃ (y z : ℕ), y ≠ z ∧ a + y + z = 17 ∧ a < y ∧ y < z)
  (tracy_cannot_determine : ∀ (x : ℕ), (c = x) → ∃ (y z : ℕ), y ≠ z ∧ y + z + c = 17 ∧ y < z ∧ z < c)
  (stacy_cannot_determine : ∀ (x : ℕ), (b = x) → ∃ (y z : ℕ), y ≠ z ∧ y + b + z = 17 ∧ y < b ∧ b < z) : 
  b = middle_card_number :=
sorry

end middle_card_number_is_6_l172_172123


namespace fuel_capacity_ratio_l172_172874

noncomputable def oldCost : ℝ := 200
noncomputable def newCost : ℝ := 480
noncomputable def priceIncreaseFactor : ℝ := 1.20

theorem fuel_capacity_ratio (C C_new : ℝ) (h1 : newCost = C_new * oldCost * priceIncreaseFactor / C) : 
  C_new / C = 2 :=
sorry

end fuel_capacity_ratio_l172_172874


namespace original_number_of_men_l172_172284

variable (M W : ℕ)

def original_work_condition := M * W / 60 = W
def larger_group_condition := (M + 8) * W / 50 = W

theorem original_number_of_men : original_work_condition M W ∧ larger_group_condition M W → M = 48 :=
by
  sorry

end original_number_of_men_l172_172284


namespace count_pos_integers_three_digits_l172_172013

/-- The number of positive integers less than 50,000 having at most three distinct digits equals 7862. -/
theorem count_pos_integers_three_digits : 
  ∃ n : ℕ, n < 50000 ∧ (∀ d1 d2 d3 d4 d5 : ℕ, d1 ≠ d2 ∨ d1 ≠ d3 ∨ d1 ≠ d4 ∨ d1 ≠ d5 ∨ d2 ≠ d3 ∨ d2 ≠ d4 ∨ d2 ≠ d5 ∨ d3 ≠ d4 ∨ d3 ≠ d5 ∨ d4 ≠ d5) ∧ n = 7862 :=
sorry

end count_pos_integers_three_digits_l172_172013


namespace roots_exist_for_all_K_l172_172920

theorem roots_exist_for_all_K (K : ℝ) : ∃ x : ℝ, x = K^3 * (x - 1) * (x - 3) :=
by
  -- Applied conditions and approach
  sorry

end roots_exist_for_all_K_l172_172920


namespace find_x_l172_172572

theorem find_x (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : x = 9 := 
by 
sorry

end find_x_l172_172572


namespace find_two_digit_numbers_l172_172358

def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem find_two_digit_numbers :
  ∀ (A : ℕ), (10 ≤ A ∧ A ≤ 99) →
    (sum_of_digits A)^2 = sum_of_digits (A^2) →
    (A = 11 ∨ A = 12 ∨ A = 13 ∨ A = 20 ∨ A = 21 ∨ A = 22 ∨ A = 30 ∨ A = 31 ∨ A = 50) :=
by sorry

end find_two_digit_numbers_l172_172358


namespace stamps_max_l172_172938

theorem stamps_max (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 25) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, (n * price_per_stamp ≤ total_cents) ∧ (∀ m : ℕ, (m > n) → (m * price_per_stamp > total_cents)) ∧ n = 200 := 
by
  sorry

end stamps_max_l172_172938


namespace t_shirts_sold_l172_172488

theorem t_shirts_sold (total_money : ℕ) (money_per_tshirt : ℕ) (n : ℕ) 
  (h1 : total_money = 2205) (h2 : money_per_tshirt = 9) (h3 : total_money = n * money_per_tshirt) : 
  n = 245 :=
by
  sorry

end t_shirts_sold_l172_172488


namespace range_f_l172_172973

noncomputable def f (a b x : ℝ) : ℝ :=
  Real.sqrt (a * Real.cos x ^ 2 + b * Real.sin x ^ 2) + 
  Real.sqrt (a * Real.sin x ^ 2 + b * Real.cos x ^ 2)

theorem range_f (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) :
  Set.range (f a b) = Set.Icc (Real.sqrt a + Real.sqrt b) (Real.sqrt (2 * (a + b))) :=
sorry

end range_f_l172_172973


namespace water_purification_problem_l172_172393

variable (x : ℝ) (h : x > 0)

theorem water_purification_problem
  (h1 : ∀ (p : ℝ), p = 2400)
  (h2 : ∀ (eff : ℝ), eff = 1.2)
  (h3 : ∀ (time_saved : ℝ), time_saved = 40) :
  (2400 * 1.2 / x) - (2400 / x) = 40 := by
  sorry

end water_purification_problem_l172_172393


namespace largest_class_students_l172_172871

theorem largest_class_students :
  ∃ x : ℕ, (x + (x - 4) + (x - 8) + (x - 12) + (x - 16) + (x - 20) + (x - 24) +
  (x - 28) + (x - 32) + (x - 36) = 100) ∧ x = 28 :=
by
  sorry

end largest_class_students_l172_172871


namespace P_Q_sum_l172_172667

noncomputable def find_P_Q_sum (P Q : ℚ) : Prop :=
  ∀ x : ℚ, (x^2 + 3 * x + 7) * (x^2 + (51/7) * x - 2) = x^4 + P * x^3 + Q * x^2 + 45 * x - 14

theorem P_Q_sum :
  ∃ P Q : ℚ, find_P_Q_sum P Q ∧ (P + Q = 260 / 7) :=
by
  sorry

end P_Q_sum_l172_172667


namespace quadratic_completion_l172_172655

theorem quadratic_completion :
  ∀ x : ℝ, (x^2 - 4*x + 1 = 0) ↔ ((x - 2)^2 = 3) :=
by
  sorry

end quadratic_completion_l172_172655


namespace simplified_form_l172_172023

def simplify_expression (x : ℝ) : ℝ :=
  (3 * x - 2) * (6 * x ^ 8 + 3 * x ^ 7 - 2 * x ^ 3 + x)

theorem simplified_form (x : ℝ) : 
  simplify_expression x = 18 * x ^ 9 - 3 * x ^ 8 - 6 * x ^ 7 - 6 * x ^ 4 - 4 * x ^ 3 + x :=
by
  sorry

end simplified_form_l172_172023


namespace compute_expression_at_4_l172_172877

theorem compute_expression_at_4 (x : ℝ) (h : x = 4) : 
  (x^8 - 32*x^4 + 256) / (x^4 - 16) = 240 := by
  sorry

end compute_expression_at_4_l172_172877


namespace first_year_payment_l172_172387

theorem first_year_payment (X : ℝ) (second_year : ℝ) (third_year : ℝ) (fourth_year : ℝ) 
    (total_payments : ℝ) 
    (h1 : second_year = X + 2)
    (h2 : third_year = X + 5)
    (h3 : fourth_year = X + 9)
    (h4 : total_payments = X + second_year + third_year + fourth_year) :
    total_payments = 96 → X = 20 :=
by
    sorry

end first_year_payment_l172_172387


namespace find_number_of_white_balls_l172_172290

-- Define the conditions
variables (n k : ℕ)
axiom k_ge_2 : k ≥ 2
axiom prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100

-- State the theorem
theorem find_number_of_white_balls (n k : ℕ) (k_ge_2 : k ≥ 2) (prob_white_black : (n * k) / ((n + k) * (n + k - 1)) = n / 100) : n = 19 :=
sorry

end find_number_of_white_balls_l172_172290


namespace inequality_solution_set_l172_172752

theorem inequality_solution_set (x : ℝ) :
  ((1 / 2 - x) * (x - 1 / 3) > 0) ↔ (1 / 3 < x ∧ x < 1 / 2) :=
by 
  sorry

end inequality_solution_set_l172_172752


namespace line_intersects_circle_l172_172868

theorem line_intersects_circle (r d : ℝ) (hr : r = 5) (hd : d = 3 * Real.sqrt 2) : d < r :=
by
  rw [hr, hd]
  exact sorry

end line_intersects_circle_l172_172868


namespace there_are_six_bases_ending_in_one_for_625_in_decimal_l172_172631

theorem there_are_six_bases_ending_in_one_for_625_in_decimal :
  (∃ ls : List ℕ, ls = [2, 3, 4, 6, 8, 12] ∧ ∀ b ∈ ls, 2 ≤ b ∧ b ≤ 12 ∧ 624 % b = 0 ∧ List.length ls = 6) :=
by
  sorry

end there_are_six_bases_ending_in_one_for_625_in_decimal_l172_172631


namespace taxi_cost_per_mile_l172_172109

variable (x : ℝ)

-- Mike's total cost
def Mike_total_cost := 2.50 + 36 * x

-- Annie's total cost
def Annie_total_cost := 2.50 + 5.00 + 16 * x

-- The primary theorem to prove
theorem taxi_cost_per_mile : Mike_total_cost x = Annie_total_cost x → x = 0.25 := by
  sorry

end taxi_cost_per_mile_l172_172109


namespace total_area_of_field_l172_172614

noncomputable def total_field_area (A1 A2 : ℝ) : ℝ := A1 + A2

theorem total_area_of_field :
  ∀ (A1 A2 : ℝ),
    A1 = 405 ∧ (A2 - A1 = (1/5) * ((A1 + A2) / 2)) →
    total_field_area A1 A2 = 900 :=
by
  intros A1 A2 h
  sorry

end total_area_of_field_l172_172614


namespace sin_150_eq_half_l172_172168

theorem sin_150_eq_half : Real.sin (150 * Real.pi / 180) = 1 / 2 :=
by
  sorry

end sin_150_eq_half_l172_172168


namespace positive_integer_solutions_count_l172_172217

theorem positive_integer_solutions_count :
  (∃ (x y z : ℕ), 1 ≤ x ∧ x ≤ y ∧ y ≤ z ∧ x + y + z = 2010) → (336847 = 336847) :=
by {
  sorry
}

end positive_integer_solutions_count_l172_172217


namespace binom_18_4_eq_3060_l172_172403

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l172_172403


namespace solution_set_of_inequality_l172_172664

theorem solution_set_of_inequality (x : ℝ) : -x^2 + 3 * x - 2 > 0 ↔ 1 < x ∧ x < 2 :=
by
  sorry

end solution_set_of_inequality_l172_172664


namespace rotate180_of_point_A_l172_172660

-- Define the point A and the transformation
def point_A : ℝ × ℝ := (-3, 2)
def rotate180 (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- Theorem statement for the problem
theorem rotate180_of_point_A :
  rotate180 point_A = (3, -2) :=
sorry

end rotate180_of_point_A_l172_172660


namespace perfect_square_trinomial_l172_172637

theorem perfect_square_trinomial (b : ℝ) : 
  (∃ (x : ℝ), 4 * x^2 + b * x + 1 = (2 * x + 1) ^ 2) ↔ (b = 4 ∨ b = -4) := 
by 
  sorry

end perfect_square_trinomial_l172_172637


namespace ratio_proof_l172_172735

variable (a b c d : ℚ)

-- Given conditions
axiom h1 : b / a = 3
axiom h2 : c / b = 4
axiom h3 : d = 5 * b

-- Theorem to be proved
theorem ratio_proof : (a + b + d) / (b + c + d) = 19 / 30 := 
by 
  sorry

end ratio_proof_l172_172735


namespace math_problem_l172_172291

variables {x y : ℝ}

theorem math_problem (h1 : x + y = 6) (h2 : x * y = 5) :
  (2 / x + 2 / y = 12 / 5) ∧ ((x - y) ^ 2 = 16) ∧ (x ^ 2 + y ^ 2 = 26) :=
by
  sorry

end math_problem_l172_172291


namespace max_pasture_area_l172_172081

/-- A rectangular sheep pasture is enclosed on three sides by a fence, while the fourth side uses the 
side of a barn that is 500 feet long. The fence costs $10 per foot, and the total budget for the 
fence is $2000. Determine the length of the side parallel to the barn that will maximize the pasture area. -/
theorem max_pasture_area (length_barn : ℝ) (cost_per_foot : ℝ) (budget : ℝ) :
  length_barn = 500 ∧ cost_per_foot = 10 ∧ budget = 2000 → 
  ∃ x : ℝ, x = 100 ∧ (∀ y : ℝ, y ≥ 0 → 
    (budget / cost_per_foot) ≥ 2*y + x → 
    (y * x ≤ y * 100)) :=
by
  sorry

end max_pasture_area_l172_172081


namespace gcd_lcm_of_a_b_l172_172055

def a := 1560
def b := 1040

theorem gcd_lcm_of_a_b :
  (Nat.gcd a b = 520) ∧ (Nat.lcm a b = 1560) :=
by
  -- Proof is omitted.
  sorry

end gcd_lcm_of_a_b_l172_172055


namespace find_cost_price_l172_172663

variables (SP CP : ℝ)
variables (discount profit : ℝ)
variable (h1 : SP = 24000)
variable (h2 : discount = 0.10)
variable (h3 : profit = 0.08)

theorem find_cost_price 
  (h1 : SP = 24000)
  (h2 : discount = 0.10)
  (h3 : profit = 0.08)
  (h4 : SP * (1 - discount) = CP * (1 + profit)) :
  CP = 20000 := 
by
  sorry

end find_cost_price_l172_172663


namespace exponent_sum_l172_172333

theorem exponent_sum : (-3)^3 + (-3)^2 + (-3)^1 + 3^1 + 3^2 + 3^3 = 18 := by
  sorry

end exponent_sum_l172_172333


namespace at_least_one_not_less_than_2_l172_172675

-- Definitions for the problem
variables {a b c : ℝ}
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)

-- The Lean 4 statement for the problem
theorem at_least_one_not_less_than_2 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (2 ≤ a + 1/b) ∨ (2 ≤ b + 1/c) ∨ (2 ≤ c + 1/a) :=
sorry

end at_least_one_not_less_than_2_l172_172675


namespace total_logs_combined_l172_172480

theorem total_logs_combined 
  (a1 l1 a2 l2 : ℕ) 
  (n1 n2 : ℕ) 
  (S1 S2 : ℕ) 
  (h1 : a1 = 15) 
  (h2 : l1 = 10) 
  (h3 : n1 = 6) 
  (h4 : S1 = n1 * (a1 + l1) / 2) 
  (h5 : a2 = 9) 
  (h6 : l2 = 5) 
  (h7 : n2 = 5) 
  (h8 : S2 = n2 * (a2 + l2) / 2) : 
  S1 + S2 = 110 :=
by {
  sorry
}

end total_logs_combined_l172_172480


namespace barbata_interest_rate_l172_172296

theorem barbata_interest_rate
  (initial_investment: ℝ)
  (additional_investment: ℝ)
  (additional_rate: ℝ)
  (total_income_rate: ℝ)
  (total_income: ℝ)
  (h_total_investment_eq: initial_investment + additional_investment = 4800)
  (h_total_income_eq: 0.06 * (initial_investment + additional_investment) = total_income):
  (initial_investment * (r : ℝ) + additional_investment * additional_rate = total_income) →
  r = 0.04 := sorry

end barbata_interest_rate_l172_172296


namespace books_left_over_after_repacking_l172_172216

theorem books_left_over_after_repacking :
  ((1335 * 39) % 40) = 25 :=
sorry

end books_left_over_after_repacking_l172_172216


namespace total_sheets_of_paper_l172_172328

theorem total_sheets_of_paper (num_classes : ℕ) (students_per_class : ℕ) (sheets_per_student : ℕ)
  (h1 : num_classes = 4) (h2 : students_per_class = 20) (h3 : sheets_per_student = 5) :
  (num_classes * students_per_class * sheets_per_student) = 400 :=
by
  sorry

end total_sheets_of_paper_l172_172328


namespace tangent_circles_m_values_l172_172585

noncomputable def is_tangent (m : ℝ) : Prop :=
  let o1_center := (m, 0)
  let o2_center := (-1, 2 * m)
  let distance := Real.sqrt ((m + 1)^2 + (2 * m)^2)
  (distance = 5 ∨ distance = 1)

theorem tangent_circles_m_values :
  {m : ℝ | is_tangent m} = {-12 / 5, -2 / 5, 0, 2} := by
  sorry

end tangent_circles_m_values_l172_172585


namespace randy_biscuits_l172_172889

theorem randy_biscuits (F : ℕ) (initial_biscuits mother_biscuits brother_ate remaining_biscuits : ℕ) 
  (h_initial : initial_biscuits = 32)
  (h_mother : mother_biscuits = 15)
  (h_brother : brother_ate = 20)
  (h_remaining : remaining_biscuits = 40)
  : ((initial_biscuits + mother_biscuits + F) - brother_ate) = remaining_biscuits → F = 13 := 
by
  intros h_eq
  sorry

end randy_biscuits_l172_172889


namespace maximum_illuminated_surfaces_l172_172345

noncomputable def optimal_position (r R d : ℝ) (h : d > r + R) : ℝ :=
  d / (1 + Real.sqrt (R^3 / r^3))

theorem maximum_illuminated_surfaces (r R d : ℝ) (h : d > r + R) (h1 : r ≤ optimal_position r R d h) (h2 : optimal_position r R d h ≤ d - R) :
  (optimal_position r R d h = d / (1 + Real.sqrt (R^3 / r^3))) ∨ (optimal_position r R d h = r) :=
sorry

end maximum_illuminated_surfaces_l172_172345


namespace min_value_of_quadratic_l172_172204

def quadratic_function (x : ℝ) : ℝ :=
  x^2 - 12 * x + 35

theorem min_value_of_quadratic :
  ∀ x : ℝ, quadratic_function x ≥ quadratic_function 6 :=
by sorry

end min_value_of_quadratic_l172_172204


namespace cos_double_angle_identity_l172_172801

variable (α : Real)

theorem cos_double_angle_identity (h : Real.sin (Real.pi / 6 + α) = 1/3) :
  Real.cos (2 * Real.pi / 3 - 2 * α) = -7/9 :=
by
  sorry

end cos_double_angle_identity_l172_172801


namespace correct_equation_l172_172386

-- Define the necessary conditions and parameters
variables (x : ℝ)

-- Length of the rectangle
def length := x 

-- Width is 6 meters less than the length
def width := x - 6

-- The area of the rectangle
def area := 720

-- Proof statement
theorem correct_equation : 
  x * (x - 6) = 720 :=
sorry

end correct_equation_l172_172386


namespace marika_father_age_twice_l172_172619

theorem marika_father_age_twice (t : ℕ) (h : t = 2036) :
  let marika_age := 10 + (t - 2006)
  let father_age := 50 + (t - 2006)
  father_age = 2 * marika_age :=
by {
  -- let marika_age := 10 + (t - 2006),
  -- let father_age := 50 + (t - 2006),
  sorry
}

end marika_father_age_twice_l172_172619


namespace seventy_five_inverse_mod_seventy_six_l172_172683

-- Lean 4 statement for the problem.
theorem seventy_five_inverse_mod_seventy_six : (75 : ℤ) * 75 % 76 = 1 :=
by
  sorry

end seventy_five_inverse_mod_seventy_six_l172_172683


namespace x0_y0_sum_eq_31_l172_172860

theorem x0_y0_sum_eq_31 :
  ∃ x0 y0 : ℕ, (0 ≤ x0 ∧ x0 < 37) ∧ (0 ≤ y0 ∧ y0 < 37) ∧ 
  (2 * x0 ≡ 1 [MOD 37]) ∧ (3 * y0 ≡ 36 [MOD 37]) ∧ 
  (x0 + y0 = 31) :=
sorry

end x0_y0_sum_eq_31_l172_172860


namespace f_diff_eq_l172_172791

def f (n : ℕ) : ℚ := 1 / 4 * (n * (n + 1) * (n + 3))

theorem f_diff_eq (r : ℕ) : 
  f (r + 1) - f r = 1 / 4 * (3 * r^2 + 11 * r + 8) :=
by {
  sorry
}

end f_diff_eq_l172_172791


namespace length_of_train_l172_172626

-- Definitions of given conditions
def train_speed (kmh : ℤ) := 25
def man_speed (kmh : ℤ) := 2
def crossing_time (sec : ℤ) := 28

-- Relative speed calculation (in meters per second)
def relative_speed := (train_speed 1 + man_speed 1) * (5 / 18 : ℚ)

-- Distance calculation (in meters)
def distance_covered := relative_speed * (crossing_time 1 : ℚ)

-- The theorem statement: Length of the train equals distance covered in crossing time
theorem length_of_train : distance_covered = 210 := by
  sorry

end length_of_train_l172_172626


namespace find_largest_m_l172_172193

theorem find_largest_m (m : ℤ) : (m^2 - 11 * m + 24 < 0) → m ≤ 7 := sorry

end find_largest_m_l172_172193


namespace Sheelas_monthly_income_l172_172234

theorem Sheelas_monthly_income (I : ℝ) (h : 0.32 * I = 3800) : I = 11875 :=
by
  sorry

end Sheelas_monthly_income_l172_172234


namespace apples_difference_l172_172489

theorem apples_difference
    (adam_apples : ℕ)
    (jackie_apples : ℕ)
    (h_adam : adam_apples = 10)
    (h_jackie : jackie_apples = 2) :
    adam_apples - jackie_apples = 8 :=
by
    sorry

end apples_difference_l172_172489


namespace distribution_plans_l172_172758

theorem distribution_plans (teachers schools : ℕ) (h_teachers : teachers = 3) (h_schools : schools = 6) : 
  ∃ plans : ℕ, plans = 210 :=
by
  sorry

end distribution_plans_l172_172758


namespace strawberries_remaining_l172_172952

theorem strawberries_remaining (initial : ℝ) (eaten_yesterday : ℝ) (eaten_today : ℝ) :
  initial = 1.6 ∧ eaten_yesterday = 0.8 ∧ eaten_today = 0.3 → initial - eaten_yesterday - eaten_today = 0.5 :=
by
  sorry

end strawberries_remaining_l172_172952


namespace peter_total_food_l172_172951

-- Given conditions
variables (chicken hamburgers hot_dogs sides total_food : ℕ)
  (h1 : chicken = 16)
  (h2 : hamburgers = chicken / 2)
  (h3 : hot_dogs = hamburgers + 2)
  (h4 : sides = hot_dogs / 2)
  (total_food : ℕ := chicken + hamburgers + hot_dogs + sides)

theorem peter_total_food (h1 : chicken = 16)
                          (h2 : hamburgers = chicken / 2)
                          (h3 : hot_dogs = hamburgers + 2)
                          (h4 : sides = hot_dogs / 2) :
                          total_food = 39 :=
by sorry

end peter_total_food_l172_172951


namespace range_my_function_l172_172793

noncomputable def my_function (x : ℝ) := (x^2 + 4 * x + 3) / (x + 2)

theorem range_my_function : 
  Set.range my_function = Set.univ := 
sorry

end range_my_function_l172_172793


namespace cos_minus_sin_l172_172046

theorem cos_minus_sin (α : ℝ) (h1 : Real.sin (2 * α) = 1 / 4) (h2 : Real.pi / 4 < α ∧ α < Real.pi / 2) : 
  Real.cos α - Real.sin α = - (Real.sqrt 3) / 2 :=
sorry

end cos_minus_sin_l172_172046


namespace pears_seed_avg_l172_172289

def apple_seed_avg : ℕ := 6
def grape_seed_avg : ℕ := 3
def total_seeds_required : ℕ := 60
def apples_count : ℕ := 4
def pears_count : ℕ := 3
def grapes_count : ℕ := 9
def seeds_short : ℕ := 3
def total_seeds_obtained : ℕ := total_seeds_required - seeds_short

theorem pears_seed_avg :
  (apples_count * apple_seed_avg) + (grapes_count * grape_seed_avg) + (pears_count * P) = total_seeds_obtained → 
  P = 2 :=
by
  sorry

end pears_seed_avg_l172_172289


namespace greatest_number_that_divides_54_87_172_l172_172553

noncomputable def gcdThree (a b c : ℤ) : ℤ :=
  gcd (gcd a b) c

theorem greatest_number_that_divides_54_87_172
  (d r : ℤ)
  (h1 : 54 % d = r)
  (h2 : 87 % d = r)
  (h3 : 172 % d = r) :
  d = gcdThree 33 85 118 := by
  -- We would start the proof here, but it's omitted per instructions
  sorry

end greatest_number_that_divides_54_87_172_l172_172553


namespace cricket_matches_total_l172_172443

theorem cricket_matches_total
  (n : ℕ)
  (avg_all : ℝ)
  (avg_first4 : ℝ)
  (avg_last3 : ℝ)
  (h_avg_all : avg_all = 56)
  (h_avg_first4 : avg_first4 = 46)
  (h_avg_last3 : avg_last3 = 69.33333333333333)
  (h_total_runs : n * avg_all = 4 * avg_first4 + 3 * avg_last3) :
  n = 7 :=
by
  sorry

end cricket_matches_total_l172_172443


namespace min_value_l172_172426

theorem min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 1/a + 1/b = 1) : 
  ∃ c : ℝ, c = 4 ∧ 
  ∀ x y : ℝ, (x = 1 / (a - 1) ∧ y = 4 / (b - 1)) → (x + y ≥ c) :=
sorry

end min_value_l172_172426


namespace find_y_l172_172747

variable (α : ℝ) (y : ℝ)
axiom sin_alpha_neg_half : Real.sin α = -1 / 2
axiom point_on_terminal_side : 2^2 + y^2 = (Real.sin α)^2 + (Real.cos α)^2

theorem find_y : y = -2 * Real.sqrt 3 / 3 :=
by {
  sorry
}

end find_y_l172_172747


namespace complementary_angles_l172_172240

theorem complementary_angles (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 :=
by 
  sorry

end complementary_angles_l172_172240


namespace auction_starting_price_l172_172439

-- Defining the conditions
def bid_increment := 5         -- The dollar increment per bid
def bids_per_person := 5       -- Number of bids per person
def total_bidders := 2         -- Number of people bidding
def final_price := 65          -- Final price of the desk after all bids

-- Calculate derived conditions
def total_bids := bids_per_person * total_bidders
def total_increment := total_bids * bid_increment

-- The statement to be proved
theorem auction_starting_price : (final_price - total_increment) = 15 :=
by
  sorry

end auction_starting_price_l172_172439


namespace sum_powers_of_5_mod_8_l172_172588

theorem sum_powers_of_5_mod_8 :
  (List.sum (List.map (fun n => (5^n % 8)) (List.range 2011))) % 8 = 4 := 
  sorry

end sum_powers_of_5_mod_8_l172_172588


namespace angle_ratio_half_l172_172246

theorem angle_ratio_half (a b c : ℝ) (A B C : ℝ) (h1 : a^2 = b * (b + c))
  (h2 : A = 2 * B ∨ A + 2 * B = Real.pi) 
  (h3 : A + B + C = Real.pi) : 
  (B / A = 1 / 2) :=
sorry

end angle_ratio_half_l172_172246


namespace spider_travel_distance_l172_172316

theorem spider_travel_distance (r : ℝ) (journey3 : ℝ) (diameter : ℝ) (leg2 : ℝ) :
    r = 75 → journey3 = 110 → diameter = 2 * r → 
    leg2 = Real.sqrt (diameter^2 - journey3^2) → 
    diameter + leg2 + journey3 = 362 :=
by
  sorry

end spider_travel_distance_l172_172316


namespace length_of_CD_l172_172625

theorem length_of_CD
    (AB BC AC AD CD : ℝ)
    (h1 : AB = 6)
    (h2 : BC = 1 / 2 * AB)
    (h3 : AC = AB + BC)
    (h4 : AD = AC)
    (h5 : CD = AD + AC) :
    CD = 18 := by
  sorry

end length_of_CD_l172_172625


namespace probability_three_hearts_l172_172913

noncomputable def probability_of_three_hearts : ℚ :=
  (13/52) * (12/51) * (11/50)

theorem probability_three_hearts :
  probability_of_three_hearts = 26/2025 :=
by
  sorry

end probability_three_hearts_l172_172913


namespace max_k_value_l172_172762

noncomputable def f (x : ℝ) : ℝ := (Real.log x) / x

theorem max_k_value :
  (∀ x : ℝ, 0 < x → (∃ k : ℝ, k * x = Real.log x ∧ k ≤ f x)) ∧
  (∀ x : ℝ, 0 < x → f x ≤ 1 / Real.exp 1) ∧
  (∀ x : ℝ, 0 < x → (k = f x → k ≤ 1 / Real.exp 1)) := 
sorry

end max_k_value_l172_172762


namespace valid_tree_arrangements_l172_172797

-- Define the types of trees
inductive TreeType
| Birch
| Oak

-- Define the condition that each tree must be adjacent to a tree of the other type
def isValidArrangement (trees : List TreeType) : Prop :=
  ∀ (i : ℕ), i < trees.length - 1 → trees.nthLe i sorry ≠ trees.nthLe (i + 1) sorry

-- Define the main problem
theorem valid_tree_arrangements : ∃ (ways : Nat), ways = 16 ∧
  ∃ (arrangements : List (List TreeType)), arrangements.length = ways ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 7 ∧ isValidArrangement arrangement :=
sorry

end valid_tree_arrangements_l172_172797


namespace log_stack_total_l172_172336

theorem log_stack_total :
  let a := 5
  let l := 15
  let n := l - a + 1
  let S := n * (a + l) / 2
  S = 110 :=
sorry

end log_stack_total_l172_172336


namespace members_with_both_non_athletic_parents_l172_172785

-- Let's define the conditions
variable (total_members athletic_dads athletic_moms both_athletic none_have_dads : ℕ)
variable (H1 : total_members = 50)
variable (H2 : athletic_dads = 25)
variable (H3 : athletic_moms = 30)
variable (H4 : both_athletic = 10)
variable (H5 : none_have_dads = 5)

-- Define the conclusion we want to prove
theorem members_with_both_non_athletic_parents : 
  (total_members - (athletic_dads + athletic_moms - both_athletic) + none_have_dads - total_members) = 10 :=
sorry

end members_with_both_non_athletic_parents_l172_172785


namespace triangle_inequality_squares_l172_172350

theorem triangle_inequality_squares (a b c : ℝ) (h₁ : a < b + c) (h₂ : b < a + c) (h₃ : c < a + b) :
  a^2 + b^2 + c^2 < 2 * (a * b + b * c + a * c) :=
sorry

end triangle_inequality_squares_l172_172350


namespace find_number_l172_172916

theorem find_number (n : ℤ) (h : 7 * n = 3 * n + 12) : n = 3 :=
sorry

end find_number_l172_172916


namespace number_of_taxis_l172_172305

-- Define the conditions explicitly
def number_of_cars : ℕ := 3
def people_per_car : ℕ := 4
def number_of_vans : ℕ := 2
def people_per_van : ℕ := 5
def people_per_taxi : ℕ := 6
def total_people : ℕ := 58

-- Define the number of people in cars and vans
def people_in_cars := number_of_cars * people_per_car
def people_in_vans := number_of_vans * people_per_van
def people_in_taxis := total_people - (people_in_cars + people_in_vans)

-- The theorem we need to prove
theorem number_of_taxis : people_in_taxis / people_per_taxi = 6 := by
  sorry

end number_of_taxis_l172_172305


namespace solve_m_range_l172_172498

-- Define the propositions
def p (m : ℝ) := m + 1 ≤ 0

def q (m : ℝ) := ∀ x : ℝ, x^2 + m * x + 1 > 0

-- Provide the Lean statement for the problem
theorem solve_m_range (m : ℝ) (hpq_false : ¬ (p m ∧ q m)) (hpq_true : p m ∨ q m) :
  m ≤ -2 ∨ (-1 < m ∧ m < 2) :=
sorry

end solve_m_range_l172_172498


namespace gratuities_charged_l172_172330

-- Define the conditions in the problem
def total_bill : ℝ := 140
def sales_tax_rate : ℝ := 0.10
def ny_striploin_cost : ℝ := 80
def wine_cost : ℝ := 10

-- Calculate the total cost before tax and gratuities
def subtotal : ℝ := ny_striploin_cost + wine_cost

-- Calculate the taxes paid
def tax : ℝ := subtotal * sales_tax_rate

-- Calculate the total bill before gratuities
def total_before_gratuities : ℝ := subtotal + tax

-- Goal: Prove that gratuities charged is 41
theorem gratuities_charged : (total_bill - total_before_gratuities) = 41 := by sorry

end gratuities_charged_l172_172330


namespace complex_calculation_l172_172456

theorem complex_calculation (i : ℂ) (hi : i * i = -1) : (1 - i)^2 * i = 2 :=
by
  sorry

end complex_calculation_l172_172456


namespace find_x_value_l172_172926

theorem find_x_value (x : ℝ) :
  |x - 25| + |x - 21| = |3 * x - 75| → x = 71 / 3 :=
by
  sorry

end find_x_value_l172_172926


namespace calc_result_l172_172518

theorem calc_result : (-2 * -3 + 2) = 8 := sorry

end calc_result_l172_172518


namespace p_plus_q_identity_l172_172561

variable {α : Type*} [CommRing α]

-- Definitions derived from conditions
def p (x : α) : α := 3 * (x - 2)
def q (x : α) : α := (x + 2) * (x - 4)

-- Lean theorem stating the problem
theorem p_plus_q_identity (x : α) : p x + q x = x^2 + x - 14 :=
by
  unfold p q
  sorry

end p_plus_q_identity_l172_172561


namespace geometric_series_ratio_l172_172435

theorem geometric_series_ratio (a r : ℝ) 
  (h_series : ∑' n : ℕ, a * r^n = 18 )
  (h_odd_series : ∑' n : ℕ, a * r^(2*n + 1) = 8 ) : 
  r = 4 / 5 := 
sorry

end geometric_series_ratio_l172_172435


namespace determine_C_plus_D_l172_172190

theorem determine_C_plus_D (A B C D : ℕ) 
  (hA : A ≠ 0) 
  (h1 : A < 10) (h2 : B < 10) (h3 : C < 10) (h4 : D < 10) 
  (h_distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D) :
  (A * 100 + B * 10 + C) * D = A * 1000 + B * 100 + C * 10 + D → 
  C + D = 5 :=
by
    sorry

end determine_C_plus_D_l172_172190


namespace upper_bound_of_n_l172_172999

theorem upper_bound_of_n (m n : ℕ) (h_m : m ≥ 2)
  (h_div : ∀ a : ℕ, gcd a n = 1 → n ∣ a^m - 1) : 
  n ≤ 4 * m * (2^m - 1) := 
sorry

end upper_bound_of_n_l172_172999


namespace polynomial_value_at_2008_l172_172031

def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4

theorem polynomial_value_at_2008 (a₀ a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₄ ≠ 0)
  (h₀₃ : f a₀ a₁ a₂ a₃ a₄ 2003 = 24)
  (h₀₄ : f a₀ a₁ a₂ a₃ a₄ 2004 = -6)
  (h₀₅ : f a₀ a₁ a₂ a₃ a₄ 2005 = 4)
  (h₀₆ : f a₀ a₁ a₂ a₃ a₄ 2006 = -6)
  (h₀₇ : f a₀ a₁ a₂ a₃ a₄ 2007 = 24) :
  f a₀ a₁ a₂ a₃ a₄ 2008 = 274 :=
by sorry

end polynomial_value_at_2008_l172_172031


namespace solution_set_of_inequality_l172_172521

-- Define the function f satisfying the given conditions
variable (f : ℝ → ℝ)
-- f(x) is symmetric about the origin
variable (symmetric_f : ∀ x, f (-x) = -f x)
-- f(2) = 2
variable (f_at_2 : f 2 = 2)
-- For any 0 < x2 < x1, the slope condition holds
variable (slope_cond : ∀ x1 x2, 0 < x2 ∧ x2 < x1 → (f x1 - f x2) / (x1 - x2) < 1)

theorem solution_set_of_inequality :
  {x : ℝ | f x - x > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end solution_set_of_inequality_l172_172521


namespace ufo_convention_attendees_l172_172539

theorem ufo_convention_attendees 
  (F M : ℕ) 
  (h1 : F + M = 450) 
  (h2 : M = F + 26) : 
  M = 238 := 
sorry

end ufo_convention_attendees_l172_172539


namespace value_of_a_l172_172060

theorem value_of_a (a : ℕ) (h : a ^ 3 = 21 * 35 * 45 * 35) : a = 105 :=
by
  sorry

end value_of_a_l172_172060


namespace car_dealership_l172_172346

variable (sportsCars : ℕ) (sedans : ℕ) (trucks : ℕ)

theorem car_dealership (h1 : 3 * sedans = 5 * sportsCars) 
  (h2 : 3 * trucks = 3 * sportsCars) 
  (h3 : sportsCars = 45) : 
  sedans = 75 ∧ trucks = 45 := by
  sorry

end car_dealership_l172_172346


namespace average_speed_for_trip_l172_172547

-- Define the total distance of the trip
def total_distance : ℕ := 850

--  Define the distance and speed for the first part of the trip
def distance1 : ℕ := 400
def speed1 : ℕ := 20

-- Define the distance and speed for the remaining part of the trip
def distance2 : ℕ := 450
def speed2 : ℕ := 15

-- Define the calculated average speed for the entire trip
def average_speed : ℕ := 17

theorem average_speed_for_trip 
  (d_total : ℕ)
  (d1 : ℕ) (s1 : ℕ)
  (d2 : ℕ) (s2 : ℕ)
  (hsum : d1 + d2 = d_total)
  (d1_eq : d1 = distance1)
  (s1_eq : s1 = speed1)
  (d2_eq : d2 = distance2)
  (s2_eq : s2 = speed2) :
  (d_total / ((d1 / s1) + (d2 / s2))) = average_speed := by
  sorry

end average_speed_for_trip_l172_172547


namespace correct_factorization_A_l172_172559

theorem correct_factorization_A (x : ℝ) : x^2 - 4 * x + 4 = (x - 2)^2 :=
by sorry

end correct_factorization_A_l172_172559


namespace variance_of_sample_l172_172728

theorem variance_of_sample
  (x : ℝ)
  (h : (2 + 3 + x + 6 + 8) / 5 = 5) : 
  (1 / 5) * ((2 - 5) ^ 2 + (3 - 5) ^ 2 + (x - 5) ^ 2 + (6 - 5) ^ 2 + (8 - 5) ^ 2) = 24 / 5 :=
by
  sorry

end variance_of_sample_l172_172728


namespace troy_needs_additional_money_l172_172604

-- Defining the initial conditions
def price_of_new_computer : ℕ := 80
def initial_savings : ℕ := 50
def money_from_selling_old_computer : ℕ := 20

-- Defining the question and expected answer
def required_additional_money : ℕ :=
  price_of_new_computer - (initial_savings + money_from_selling_old_computer)

-- The proof statement
theorem troy_needs_additional_money : required_additional_money = 10 := by
  sorry

end troy_needs_additional_money_l172_172604


namespace N_subset_proper_M_l172_172379

open Set Int

def set_M : Set ℝ := {x | ∃ k : ℤ, x = (k + 2) / 4}
def set_N : Set ℝ := {x | ∃ k : ℤ, x = (2 * k + 1) / 4}

theorem N_subset_proper_M : set_N ⊂ set_M := by
  sorry

end N_subset_proper_M_l172_172379


namespace selling_price_before_brokerage_l172_172943

theorem selling_price_before_brokerage (cash_realized : ℝ) (brokerage_rate : ℝ) (final_cash : ℝ) : 
  final_cash = 104.25 → brokerage_rate = 1 / 400 → cash_realized = 104.51 :=
by
  intro h1 h2
  sorry

end selling_price_before_brokerage_l172_172943


namespace smallest_k_divides_l172_172985

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l172_172985


namespace div_remainder_l172_172356

theorem div_remainder (x : ℕ) (h : x = 2^40) : 
  (2^160 + 160) % (2^80 + 2^40 + 1) = 159 :=
by
  sorry

end div_remainder_l172_172356


namespace percentage_is_50_l172_172035

theorem percentage_is_50 (P : ℝ) (h1 : P = 0.20 * 15 + 47) : P = 50 := 
by
  -- skip the proof
  sorry

end percentage_is_50_l172_172035


namespace power_mod_equality_l172_172646

theorem power_mod_equality (n : ℕ) : 
  (47 % 8 = 7) → (23 % 8 = 7) → (47 ^ 2500 - 23 ^ 2500) % 8 = 0 := 
by
  intro h1 h2
  sorry

end power_mod_equality_l172_172646


namespace regular_icosahedron_edges_l172_172459

-- Define the concept of a regular icosahedron.
structure RegularIcosahedron :=
  (vertices : ℕ)
  (faces : ℕ)
  (edges : ℕ)

-- Define the properties of a regular icosahedron.
def regular_icosahedron_properties (ico : RegularIcosahedron) : Prop :=
  ico.vertices = 12 ∧ ico.faces = 20 ∧ ico.edges = 30

-- Statement of the proof problem: The number of edges in a regular icosahedron is 30.
theorem regular_icosahedron_edges : ∀ (ico : RegularIcosahedron), regular_icosahedron_properties ico → ico.edges = 30 :=
by
  sorry

end regular_icosahedron_edges_l172_172459


namespace find_ratio_squares_l172_172082

variables (x y z a b c : ℝ)

theorem find_ratio_squares 
  (h1 : x / a + y / b + z / c = 5) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 25 := 
sorry

end find_ratio_squares_l172_172082


namespace band_first_set_songs_count_l172_172391

theorem band_first_set_songs_count 
  (total_repertoire : ℕ) (second_set : ℕ) (encore : ℕ) (avg_third_fourth : ℕ)
  (h_total_repertoire : total_repertoire = 30)
  (h_second_set : second_set = 7)
  (h_encore : encore = 2)
  (h_avg_third_fourth : avg_third_fourth = 8)
  : ∃ (x : ℕ), x + second_set + encore + avg_third_fourth * 2 = total_repertoire := 
  sorry

end band_first_set_songs_count_l172_172391


namespace optimal_order_l172_172623

variables (p1 p2 p3 : ℝ)
variables (hp3_lt_p1 : p3 < p1) (hp1_lt_p2 : p1 < p2)

theorem optimal_order (hcond1 : p2 * (p1 + p3 - p1 * p3) > p1 * (p2 + p3 - p2 * p3))
    : true :=
by {
  -- the details of the proof would go here, but we skip it with sorry
  sorry
}

end optimal_order_l172_172623


namespace value_of_sum_l172_172398

theorem value_of_sum (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d ≠ 0)
  (hc_solution : c^2 + a * c + b = 0) (hd_solution : d^2 + a * d + b = 0)
  (ha_solution : a^2 + c * a + d = 0) (hb_solution : b^2 + c * b + d = 0)
: a + b + c + d = -2 := sorry -- The proof is omitted as requested

end value_of_sum_l172_172398


namespace percentage_saved_l172_172352

theorem percentage_saved (rent milk groceries education petrol misc savings : ℝ) 
  (salary : ℝ) 
  (h_rent : rent = 5000) 
  (h_milk : milk = 1500) 
  (h_groceries : groceries = 4500) 
  (h_education : education = 2500) 
  (h_petrol : petrol = 2000) 
  (h_misc : misc = 700) 
  (h_savings : savings = 1800) 
  (h_salary : salary = rent + milk + groceries + education + petrol + misc + savings) : 
  (savings / salary) * 100 = 10 :=
by
  sorry

end percentage_saved_l172_172352


namespace square_of_binomial_l172_172907

theorem square_of_binomial (a : ℝ) :
  (∃ b : ℝ, (9:ℝ) * x^2 + 24 * x + a = (3 * x + b)^2) → a = 16 :=
by
  sorry

end square_of_binomial_l172_172907


namespace find_n_l172_172824

theorem find_n (n : ℕ) (h1 : Nat.lcm n 16 = 52) (h2 : Nat.gcd n 16 = 8) : n = 26 := by
  sorry

end find_n_l172_172824


namespace arun_remaining_work_days_l172_172219

noncomputable def arun_and_tarun_work_in_days (W : ℝ) := 10
noncomputable def arun_alone_work_in_days (W : ℝ) := 60
noncomputable def arun_tarun_together_days := 4

theorem arun_remaining_work_days (W : ℝ) :
  (arun_and_tarun_work_in_days W = 10) ∧
  (arun_alone_work_in_days W = 60) ∧
  (let complete_work_days := arun_tarun_together_days;
  let remaining_work := W - (complete_work_days / arun_and_tarun_work_in_days W * W);
  let arun_remaining_days := (remaining_work / W) * arun_alone_work_in_days W;
  arun_remaining_days = 36) :=
sorry

end arun_remaining_work_days_l172_172219


namespace exists_x_odd_n_l172_172073

theorem exists_x_odd_n (n : ℤ) (h : n % 2 = 1) : 
  ∃ x : ℤ, n^2 ∣ x^2 - n*x - 1 := by
  sorry

end exists_x_odd_n_l172_172073


namespace ellipse_k_values_l172_172476

theorem ellipse_k_values (k : ℝ) :
  (∃ k, (∃ e, e = 1/2 ∧
    (∃ a b : ℝ, a = Real.sqrt (k+8) ∧ b = 3 ∧
      ∃ c, (c = Real.sqrt (abs ((a^2) - (b^2)))) ∧ (e = c/b ∨ e = c/a)) ∧
      k = 4 ∨ k = -5/4)) :=
  sorry

end ellipse_k_values_l172_172476


namespace length_AB_l172_172590

open Real

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

theorem length_AB (x1 y1 x2 y2 : ℝ) 
  (hA : y1^2 = 4 * x1) (hB : y2^2 = 4 * x2) 
  (hLine: (y2 - y1) * 1 = (x2 - x1) *0)
  (hSum : x1 + x2 = 6) : 
  dist (x1, y1) (x2, y2) = 8 := 
sorry

end length_AB_l172_172590


namespace find_q_l172_172827

theorem find_q (q : ℕ) (h1 : 32 = 2^5) (h2 : 32^5 = 2^q) : q = 25 := by
  sorry

end find_q_l172_172827


namespace combine_like_terms_substitute_expression_complex_expression_l172_172154

-- Part 1
theorem combine_like_terms (a b : ℝ) : 
  10 * (a - b)^2 - 12 * (a - b)^2 + 9 * (a - b)^2 = 7 * (a - b)^2 :=
by
  sorry

-- Part 2
theorem substitute_expression (x y : ℝ) (h1 : x^2 - 2 * y = -5) : 
  4 * x^2 - 8 * y + 24 = 4 :=
by
  sorry

-- Part 3
theorem complex_expression (a b c d : ℝ) 
  (h1 : a - 2 * b = 1009.5) 
  (h2 : 2 * b - c = -2024.6666)
  (h3 : c - d = 1013.1666) : 
  (a - c) + (2 * b - d) - (2 * b - c) = -2 :=
by
  sorry

end combine_like_terms_substitute_expression_complex_expression_l172_172154


namespace speed_of_stream_l172_172474

variable (x : ℝ) -- Let the speed of the stream be x kmph

-- Conditions
variable (speed_of_boat_in_still_water : ℝ)
variable (time_upstream_twice_time_downstream : Prop)

-- Given conditions
axiom h1 : speed_of_boat_in_still_water = 48
axiom h2 : time_upstream_twice_time_downstream → 1 / (speed_of_boat_in_still_water - x) = 2 * (1 / (speed_of_boat_in_still_water + x))

-- Theorem to prove
theorem speed_of_stream (h2: time_upstream_twice_time_downstream) : x = 16 := by
  sorry

end speed_of_stream_l172_172474


namespace find_value_of_f2_sub_f3_l172_172254

variable (f : ℝ → ℝ)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem find_value_of_f2_sub_f3 (h_odd : is_odd_function f) (h_sum : f (-2) + f 0 + f 3 = 2) :
  f 2 - f 3 = -2 :=
by
  sorry

end find_value_of_f2_sub_f3_l172_172254


namespace return_time_is_2_hours_l172_172736

noncomputable def distance_home_city_hall := 6
noncomputable def speed_to_city_hall := 3 -- km/h
noncomputable def additional_distance_return := 2 -- km
noncomputable def speed_return := 4 -- km/h
noncomputable def total_trip_time := 4 -- hours

theorem return_time_is_2_hours :
  (distance_home_city_hall + additional_distance_return) / speed_return = 2 :=
by
  sorry

end return_time_is_2_hours_l172_172736


namespace service_charge_percentage_is_correct_l172_172956

-- Define the conditions
def orderAmount : ℝ := 450
def totalAmountPaid : ℝ := 468
def serviceCharge : ℝ := totalAmountPaid - orderAmount

-- Define the target percentage
def expectedServiceChargePercentage : ℝ := 4.0

-- Proof statement: the service charge percentage is expectedServiceChargePercentage
theorem service_charge_percentage_is_correct : 
  (serviceCharge / orderAmount) * 100 = expectedServiceChargePercentage :=
by
  sorry

end service_charge_percentage_is_correct_l172_172956


namespace value_of_a_l172_172467

theorem value_of_a (a : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = x^2 - a * x + 4) (h₂ : ∀ x, f (x + 1) = f (1 - x)) :
  a = 2 :=
sorry

end value_of_a_l172_172467


namespace speed_of_current_l172_172404

theorem speed_of_current (v : ℝ) : 
  (∀ s, s = 3 → s / (3 - v) = 2.3076923076923075) → v = 1.7 := 
by
  intro h
  sorry

end speed_of_current_l172_172404


namespace largest_number_is_A_l172_172781

noncomputable def numA : ℝ := 4.25678
noncomputable def numB : ℝ := 4.2567777 -- repeating 7
noncomputable def numC : ℝ := 4.25676767 -- repeating 67
noncomputable def numD : ℝ := 4.25675675 -- repeating 567
noncomputable def numE : ℝ := 4.25672567 -- repeating 2567

theorem largest_number_is_A : numA > numB ∧ numA > numC ∧ numA > numD ∧ numA > numE := by
  sorry

end largest_number_is_A_l172_172781


namespace gcd_of_powers_l172_172517

theorem gcd_of_powers (m n : ℕ) (h1 : m = 2^2024 - 1) (h2 : n = 2^2007 - 1) : 
  Nat.gcd m n = 131071 :=
by
  sorry

end gcd_of_powers_l172_172517


namespace tan_alpha_tan_beta_l172_172688

/-- Given the cosine values of the sum and difference of two angles, 
    find the value of the product of their tangents. -/
theorem tan_alpha_tan_beta (α β : ℝ) 
  (h1 : Real.cos (α + β) = 1/3) 
  (h2 : Real.cos (α - β) = 1/5) : 
  Real.tan α * Real.tan β = -1/4 := sorry

end tan_alpha_tan_beta_l172_172688
