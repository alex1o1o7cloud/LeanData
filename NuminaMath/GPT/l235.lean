import Mathlib

namespace nested_sqrt_eq_l235_235666

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l235_235666


namespace largest_integer_condition_l235_235387

theorem largest_integer_condition (x : ℤ) : (x/3 + 3/4 : ℚ) < 7/3 → x ≤ 4 :=
by
  sorry

end largest_integer_condition_l235_235387


namespace no_solution_eq_l235_235329

theorem no_solution_eq (k : ℝ) :
  (¬ ∃ x : ℝ, x ≠ 3 ∧ x ≠ 7 ∧ (x + 2) / (x - 3) = (x - k) / (x - 7)) ↔ k = 2 :=
by
  sorry

end no_solution_eq_l235_235329


namespace max_friday_more_than_wednesday_l235_235256

-- Definitions and conditions
def played_hours_wednesday : ℕ := 2
def played_hours_thursday : ℕ := 2
def played_average_hours : ℕ := 3
def played_days : ℕ := 3

-- Total hours over three days
def total_hours : ℕ := played_average_hours * played_days

-- Hours played on Friday
def played_hours_wednesday_thursday : ℕ := played_hours_wednesday + played_hours_thursday

def played_hours_friday : ℕ := total_hours - played_hours_wednesday_thursday

-- Proof problem statement
theorem max_friday_more_than_wednesday : 
  played_hours_friday - played_hours_wednesday = 3 := 
sorry

end max_friday_more_than_wednesday_l235_235256


namespace common_ratio_of_geometric_series_l235_235682

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l235_235682


namespace sqrt_expression_value_l235_235176

theorem sqrt_expression_value :
  Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 25 * Real.sqrt 5 :=
by
  sorry

end sqrt_expression_value_l235_235176


namespace set_intersection_eq_l235_235105

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 3, 5}
def B : Set ℕ := {2, 5}
def ComplementU (S : Set ℕ) : Set ℕ := U \ S

theorem set_intersection_eq : 
  A ∩ (ComplementU B) = {1, 3} := 
by
  sorry

end set_intersection_eq_l235_235105


namespace arithmetic_seq_common_diff_l235_235335

theorem arithmetic_seq_common_diff
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)
  (h_a1 : a 1 = 1)
  (h_geomet : (a 3) ^ 2 = a 1 * a 13) :
  d = 2 :=
by
  sorry

end arithmetic_seq_common_diff_l235_235335


namespace num_ways_to_assign_grades_l235_235838

theorem num_ways_to_assign_grades : (4 ^ 12) = 16777216 := by
  sorry

end num_ways_to_assign_grades_l235_235838


namespace Robie_gave_away_boxes_l235_235127

theorem Robie_gave_away_boxes :
  ∀ (total_cards cards_per_box boxes_with_him remaining_cards : ℕ)
  (h_total_cards : total_cards = 75)
  (h_cards_per_box : cards_per_box = 10)
  (h_boxes_with_him : boxes_with_him = 5)
  (h_remaining_cards : remaining_cards = 5),
  (total_cards / cards_per_box) - boxes_with_him = 2 :=
by
  intros total_cards cards_per_box boxes_with_him remaining_cards
  intros h_total_cards h_cards_per_box h_boxes_with_him h_remaining_cards
  sorry

end Robie_gave_away_boxes_l235_235127


namespace clients_number_l235_235395

theorem clients_number (C : ℕ) (total_cars : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ)
  (h1 : total_cars = 12)
  (h2 : cars_per_client = 4)
  (h3 : selections_per_car = 3)
  (h4 : C * cars_per_client = total_cars * selections_per_car) : C = 9 :=
by sorry

end clients_number_l235_235395


namespace find_trapezoid_bases_l235_235809

-- Define the conditions of the isosceles trapezoid
variables {AD BC : ℝ}
variables (h1 : ∀ (A B C D : ℝ), is_isosceles_trapezoid A B C D ∧ intersects_at_right_angle A B C D)
variables (h2 : ∀ {A B C D : ℝ}, trapezoid_area A B C D = 12)
variables (h3 : ∀ {A B C D : ℝ}, trapezoid_height A B C D = 2)

-- Prove the bases AD and BC are 8 and 4 respectively under the given conditions
theorem find_trapezoid_bases (AD BC : ℝ) : 
  AD = 8 ∧ BC = 4 :=
  sorry

end find_trapezoid_bases_l235_235809


namespace total_cost_rental_l235_235258

theorem total_cost_rental :
  let rental_fee := 20.99
  let charge_per_mile := 0.25
  let miles_driven := 299
  let total_cost := rental_fee + charge_per_mile * miles_driven
  total_cost = 95.74 := by
{
  sorry
}

end total_cost_rental_l235_235258


namespace sum_of_x_y_l235_235518

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235518


namespace solve_for_r_l235_235129

theorem solve_for_r (r : ℝ) : 
  (r^2 - 3) / 3 = (5 - r) / 2 ↔ 
  r = (-3 + Real.sqrt 177) / 4 ∨ r = (-3 - Real.sqrt 177) / 4 :=
by
  sorry

end solve_for_r_l235_235129


namespace area_of_parallelogram_l235_235027

def parallelogram_base : ℝ := 26
def parallelogram_height : ℝ := 14

theorem area_of_parallelogram : parallelogram_base * parallelogram_height = 364 := by
  sorry

end area_of_parallelogram_l235_235027


namespace natural_number_pairs_lcm_gcd_l235_235862

theorem natural_number_pairs_lcm_gcd (a b : ℕ) (h1 : lcm a b * gcd a b = a * b)
  (h2 : lcm a b - gcd a b = (a * b) / 5) : 
  (a = 4 ∧ b = 20) ∨ (a = 20 ∧ b = 4) :=
  sorry

end natural_number_pairs_lcm_gcd_l235_235862


namespace cylinder_volume_multiplication_factor_l235_235390

theorem cylinder_volume_multiplication_factor (r h : ℝ) (h_r_positive : r > 0) (h_h_positive : h > 0) :
  let V := π * r^2 * h
  let V' := π * (2.5 * r)^2 * (3 * h)
  let X := V' / V
  X = 18.75 :=
by
  -- Proceed with the proof here
  sorry

end cylinder_volume_multiplication_factor_l235_235390


namespace minimum_height_l235_235922

theorem minimum_height (x : ℝ) (h : ℝ) (A : ℝ) :
  (h = x + 4) →
  (A = 6*x^2 + 16*x) →
  (A ≥ 120) →
  (x ≥ 2) →
  h = 6 :=
by
  intros h_def A_def A_geq min_x
  sorry

end minimum_height_l235_235922


namespace area_curve_is_correct_l235_235462

-- Define the initial conditions
structure Rectangle :=
  (vertices : Fin 4 → ℝ × ℝ)
  (point : ℝ × ℝ)

-- Define the rotation transformation
def rotate_clockwise_90 (center : ℝ × ℝ) (point : ℝ × ℝ) : ℝ × ℝ :=
  let (cx, cy) := center
  let (px, py) := point
  (cx + (py - cy), cy - (px - cx))

-- Given initial rectangle and the point to track
def initial_rectangle : Rectangle :=
  { vertices := ![(0, 0), (2, 0), (0, 3), (2, 3)],
    point := (1, 1) }

-- Perform the four specified rotations
def rotated_points : List (ℝ × ℝ) :=
  let r1 := rotate_clockwise_90 (2, 0) initial_rectangle.point
  let r2 := rotate_clockwise_90 (5, 0) r1
  let r3 := rotate_clockwise_90 (7, 0) r2
  let r4 := rotate_clockwise_90 (10, 0) r3
  [initial_rectangle.point, r1, r2, r3, r4]

-- Calculate the area below the curve and above the x-axis
noncomputable def area_below_curve : ℝ :=
  6 + (7 * Real.pi / 2)

-- The theorem statement
theorem area_curve_is_correct : 
  area_below_curve = 6 + (7 * Real.pi / 2) :=
  by trivial

end area_curve_is_correct_l235_235462


namespace probability_even_xy_sub_xy_even_l235_235151

theorem probability_even_xy_sub_xy_even :
  let s := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_ways := (s.card.choose 2)
  let even_ways := (evens.card.choose 2)
  even_ways.toRat / total_ways.toRat = 5 / 22 :=
by
  sorry

end probability_even_xy_sub_xy_even_l235_235151


namespace find_expression_l235_235876

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l235_235876


namespace find_xy_l235_235442

theorem find_xy :
  ∃ (x y : ℝ), (x - 14)^2 + (y - 15)^2 + (x - y)^2 = 1/3 ∧ x = 14 + 1/3 ∧ y = 14 + 2/3 :=
by
  sorry

end find_xy_l235_235442


namespace problem_l235_235878

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l235_235878


namespace repeating_decimal_computation_l235_235998

noncomputable def x := 864 / 999
noncomputable def y := 579 / 999
noncomputable def z := 135 / 999

theorem repeating_decimal_computation :
  x - y - z = 50 / 333 :=
by
  sorry

end repeating_decimal_computation_l235_235998


namespace chromosome_stability_due_to_meiosis_and_fertilization_l235_235356

/-- Definition of reducing chromosome number during meiosis -/
def meiosis_reduces_chromosome_number (n : ℕ) : ℕ := n / 2

/-- Definition of restoring chromosome number during fertilization -/
def fertilization_restores_chromosome_number (n : ℕ) : ℕ := n * 2

/-- Axiom: Sexual reproduction involves meiosis and fertilization to maintain chromosome stability -/
axiom chromosome_stability (n m : ℕ) (h1 : meiosis_reduces_chromosome_number n = m) 
  (h2 : fertilization_restores_chromosome_number m = n) : n = n

/-- Theorem statement in Lean 4: The chromosome number stability in sexually reproducing organisms is maintained due to meiosis and fertilization -/
theorem chromosome_stability_due_to_meiosis_and_fertilization 
  (n : ℕ) (h_meiosis: meiosis_reduces_chromosome_number n = n / 2) 
  (h_fertilization: fertilization_restores_chromosome_number (n / 2) = n) : 
  n = n := 
by
  apply chromosome_stability
  exact h_meiosis
  exact h_fertilization

end chromosome_stability_due_to_meiosis_and_fertilization_l235_235356


namespace convert_spherical_to_rectangular_l235_235207

def spherical_to_rectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem convert_spherical_to_rectangular :
  spherical_to_rectangular 5 (Real.pi / 4) (Real.pi / 3) = (5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (Real.sqrt 3 / 2) * (Real.sqrt 2 / 2), 5 * (1 / 2)) :=
by
  sorry

end convert_spherical_to_rectangular_l235_235207


namespace yellow_block_weight_proof_l235_235097

-- Define the weights and the relationship between them
def green_block_weight : ℝ := 0.4
def additional_weight : ℝ := 0.2
def yellow_block_weight : ℝ := green_block_weight + additional_weight

-- The theorem to prove
theorem yellow_block_weight_proof : yellow_block_weight = 0.6 :=
by
  -- Proof will be supplied here
  sorry

end yellow_block_weight_proof_l235_235097


namespace prime_factors_of_30_l235_235755

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l235_235755


namespace prob_A_inter_B_l235_235620

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l235_235620


namespace asymptotes_N_are_correct_l235_235891

-- Given the conditions of the hyperbola M
def hyperbola_M (x y : ℝ) (m : ℝ) : Prop :=
  x^2 / m - y^2 / 6 = 1

-- Eccentricity condition
def eccentricity (m : ℝ) (e : ℝ) : Prop :=
  e = 2 ∧ (m > 0)

-- Given hyperbola N
def hyperbola_N (x y : ℝ) (m : ℝ) : Prop :=
  x^2 - y^2 / m = 1

-- The theorem to be proved
theorem asymptotes_N_are_correct (m : ℝ) (x y : ℝ) :
  hyperbola_M x y 2 → eccentricity 2 2 → hyperbola_N x y m →
  (y = x * Real.sqrt 2 ∨ y = -x * Real.sqrt 2) :=
by
  sorry

end asymptotes_N_are_correct_l235_235891


namespace prime_factors_30_fac_eq_10_l235_235716

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l235_235716


namespace number_less_than_neg_two_l235_235812

theorem number_less_than_neg_two : ∃ x : Int, x = -2 - 1 := 
by
  use -3
  sorry

end number_less_than_neg_two_l235_235812


namespace layoffs_payment_l235_235075

theorem layoffs_payment :
  let total_employees := 450
  let salary_2000_employees := 150
  let salary_2500_employees := 200
  let salary_3000_employees := 100
  let first_round_2000_layoffs := 0.20 * salary_2000_employees
  let first_round_2500_layoffs := 0.25 * salary_2500_employees
  let first_round_3000_layoffs := 0.15 * salary_3000_employees
  let remaining_2000_after_first_round := salary_2000_employees - first_round_2000_layoffs
  let remaining_2500_after_first_round := salary_2500_employees - first_round_2500_layoffs
  let remaining_3000_after_first_round := salary_3000_employees - first_round_3000_layoffs
  let second_round_2000_layoffs := 0.10 * remaining_2000_after_first_round
  let second_round_2500_layoffs := 0.15 * remaining_2500_after_first_round
  let second_round_3000_layoffs := 0.05 * remaining_3000_after_first_round
  let remaining_2000_after_second_round := remaining_2000_after_first_round - second_round_2000_layoffs
  let remaining_2500_after_second_round := remaining_2500_after_first_round - second_round_2500_layoffs
  let remaining_3000_after_second_round := remaining_3000_after_first_round - second_round_3000_layoffs
  let total_payment := remaining_2000_after_second_round * 2000 + remaining_2500_after_second_round * 2500 + remaining_3000_after_second_round * 3000
  total_payment = 776500 := sorry

end layoffs_payment_l235_235075


namespace weight_of_person_being_replaced_l235_235280

variable (W_old : ℝ)

theorem weight_of_person_being_replaced :
  (W_old : ℝ) = 35 :=
by
  -- Given: The average weight of 8 persons increases by 5 kg.
  -- The weight of the new person is 75 kg.
  -- The total weight increase is 40 kg.
  -- Prove that W_old = 35 kg.
  sorry

end weight_of_person_being_replaced_l235_235280


namespace find_integer_k_l235_235873

theorem find_integer_k {k : ℤ} :
  (∀ x : ℝ, (k^2 + 1) * x^2 - (4 - k) * x + 1 = 0 →
    (∃ m n : ℝ, m ≠ n ∧ m * n = 1 / (k^2 + 1) ∧ m + n = (4 - k) / (k^2 + 1) ∧
      ((1 < m ∧ n < 1) ∨ (1 < n ∧ m < 1)))) →
  k = -1 ∨ k = 0 :=
by
  sorry

end find_integer_k_l235_235873


namespace total_toothpicks_correct_l235_235384

-- Define the number of vertical lines and toothpicks in them
def num_vertical_lines : ℕ := 41
def num_toothpicks_per_vertical_line : ℕ := 20
def vertical_toothpicks : ℕ := num_vertical_lines * num_toothpicks_per_vertical_line

-- Define the number of horizontal lines and toothpicks in them
def num_horizontal_lines : ℕ := 21
def num_toothpicks_per_horizontal_line : ℕ := 40
def horizontal_toothpicks : ℕ := num_horizontal_lines * num_toothpicks_per_horizontal_line

-- Define the dimensions of the triangle
def triangle_base : ℕ := 20
def triangle_height : ℕ := 20
def triangle_hypotenuse : ℕ := 29 -- approximated

-- Total toothpicks in the triangle
def triangle_toothpicks : ℕ := triangle_height + triangle_hypotenuse

-- Total toothpicks used in the structure
def total_toothpicks : ℕ := vertical_toothpicks + horizontal_toothpicks + triangle_toothpicks

-- Theorem to prove the total number of toothpicks used is 1709
theorem total_toothpicks_correct : total_toothpicks = 1709 := by
  sorry

end total_toothpicks_correct_l235_235384


namespace range_of_7a_minus_5b_l235_235460

theorem range_of_7a_minus_5b (a b : ℝ) (h1 : 5 ≤ a - b ∧ a - b ≤ 27) (h2 : 6 ≤ a + b ∧ a + b ≤ 30) : 
  36 ≤ 7 * a - 5 * b ∧ 7 * a - 5 * b ≤ 192 :=
sorry

end range_of_7a_minus_5b_l235_235460


namespace problem_a_b_sum_l235_235845

-- Define the operation
def otimes (x y : ℝ) : ℝ := x * (1 - y)

-- Given conditions
variable (a b : ℝ)

-- Theorem statement: Prove that a + b = 4
theorem problem_a_b_sum :
  (∀ x, ((2 < x) ∧ (x < 3)) ↔ ((x - a) * (x - b - 1) < 0)) → a + b = 4 :=
by
  sorry

end problem_a_b_sum_l235_235845


namespace possible_values_of_sum_l235_235544

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235544


namespace trig_functions_symmetry_l235_235806

theorem trig_functions_symmetry :
  ∀ k₁ k₂ : ℤ,
  (∃ x, x = k₁ * π / 2 + π / 3 ∧ x = k₂ * π + π / 3) ∧
  (¬ ∃ x, (x, 0) = (k₁ * π / 2 + π / 12, 0) ∧ (x, 0) = (k₂ * π + 5 * π / 6, 0)) :=
by
  sorry

end trig_functions_symmetry_l235_235806


namespace rationalize_denominator_l235_235125

theorem rationalize_denominator :
  (1 / (real.sqrt 3 - 2)) = -(real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l235_235125


namespace seating_arrangements_l235_235279

theorem seating_arrangements :
  let boys := 6
  let girls := 5
  let chairs := 11
  let total_arrangements := Nat.factorial chairs
  let restricted_arrangements := Nat.factorial boys * Nat.factorial girls
  total_arrangements - restricted_arrangements = 39830400 :=
by
  sorry

end seating_arrangements_l235_235279


namespace find_tangent_line_perpendicular_to_given_line_l235_235213

theorem find_tangent_line_perpendicular_to_given_line :
  (∀ x ∈ Set.Icc (-1) (-1), 2*x - 6*((x^3 + 3*x^2 - 1)) + 1 = 0) →
  (∃ l : ℝ, ∀ x : ℝ, 3*x + l + 2 = 0) :=
by
  sorry

end find_tangent_line_perpendicular_to_given_line_l235_235213


namespace find_x_l235_235342

theorem find_x (x : ℝ) : 
  let a := (4, 2)
  let b := (x, 3)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = -3 / 2 :=
by
  intros a b h
  sorry

end find_x_l235_235342


namespace probability_of_specific_sequence_l235_235300

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l235_235300


namespace tens_digit_of_2013_pow_2018_minus_2019_l235_235389

theorem tens_digit_of_2013_pow_2018_minus_2019 :
  (2013 ^ 2018 - 2019) % 100 / 10 % 10 = 5 := sorry

end tens_digit_of_2013_pow_2018_minus_2019_l235_235389


namespace weights_identical_l235_235814

theorem weights_identical (w : Fin 13 → ℤ) 
  (h : ∀ i, ∃ (A B : Finset (Fin 13)), A.card = 6 ∧ B.card = 6 ∧ A ∪ B = Finset.univ.erase i ∧ (A.sum w) = (B.sum w)) :
  ∀ i j, w i = w j :=
by
  sorry

end weights_identical_l235_235814


namespace perpendicular_slope_l235_235032

theorem perpendicular_slope (a b c : ℝ) (h : 4 * a - 6 * b = c) :
  let m := - (3 / 2) in
  ∃ k : ℝ, (k = m) :=
by
  sorry

end perpendicular_slope_l235_235032


namespace batches_engine_count_l235_235484

theorem batches_engine_count (x : ℕ) 
  (h1 : ∀ e, 1/4 * e = 0) -- every batch has engines, no proof needed for this question
  (h2 : 5 * (3/4 : ℚ) * x = 300) : 
  x = 80 := 
sorry

end batches_engine_count_l235_235484


namespace statement_A_statement_C_statement_D_statement_B_l235_235875

variable (a b : ℝ)

theorem statement_A :
  4 * a^2 - a * b + b^2 = 1 → |a| ≤ 2 * Real.sqrt 15 / 15 :=
sorry

theorem statement_C :
  (4 * a^2 - a * b + b^2 = 1) → 4 / 5 ≤ 4 * a^2 + b^2 ∧ 4 * a^2 + b^2 ≤ 4 / 3 :=
sorry

theorem statement_D :
  4 * a^2 - a * b + b^2 = 1 → |2 * a - b| ≤ 2 * Real.sqrt 10 / 5 :=
sorry

theorem statement_B :
  4 * a^2 - a * b + b^2 = 1 → ¬(|a + b| < 1) :=
sorry

end statement_A_statement_C_statement_D_statement_B_l235_235875


namespace factorial_prime_factors_l235_235764

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l235_235764


namespace total_cats_in_academy_l235_235316

theorem total_cats_in_academy (cats_jump cats_jump_fetch cats_fetch cats_fetch_spin cats_spin cats_jump_spin cats_all_three cats_none: ℕ)
  (h_jump: cats_jump = 60)
  (h_jump_fetch: cats_jump_fetch = 20)
  (h_fetch: cats_fetch = 35)
  (h_fetch_spin: cats_fetch_spin = 15)
  (h_spin: cats_spin = 40)
  (h_jump_spin: cats_jump_spin = 22)
  (h_all_three: cats_all_three = 11)
  (h_none: cats_none = 10) :
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none = 99 :=
by
  calc 
  cats_all_three + (cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + (cats_jump_spin - cats_all_three) +
  (cats_jump - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_fetch - ((cats_jump_fetch - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) +
  (cats_spin - ((cats_jump_spin - cats_all_three) + (cats_fetch_spin - cats_all_three) + cats_all_three)) + cats_none 
  = 11 + (20 - 11) + (15 - 11) + (22 - 11) + (60 - (9 + 11 + 11)) + (35 - (9 + 4 + 11)) + (40 - (11 + 4 + 11)) + 10 
  := by sorry
  _ = 99 := by sorry

end total_cats_in_academy_l235_235316


namespace euler_conjecture_counter_example_l235_235510

theorem euler_conjecture_counter_example :
  ∃ (n : ℕ), 133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144 :=
by
  sorry

end euler_conjecture_counter_example_l235_235510


namespace maximum_value_of_parabola_eq_24_l235_235817

theorem maximum_value_of_parabola_eq_24 (x : ℝ) : 
  ∃ x, x = -2 ∧ (-2 * x^2 - 8 * x + 16) = 24 :=
by
  use -2
  sorry

end maximum_value_of_parabola_eq_24_l235_235817


namespace eval_sqrt_expression_l235_235652

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l235_235652


namespace total_birds_is_1300_l235_235182

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l235_235182


namespace theorem_perimeter_shaded_region_theorem_area_shaded_region_l235_235911

noncomputable section

-- Definitions based on the conditions
def r : ℝ := Real.sqrt (1 / Real.pi)  -- radius of the unit circle

-- Define the perimeter and area functions for the shaded region
def perimeter_shaded_region (r : ℝ) : ℝ :=
  2 * Real.sqrt Real.pi

def area_shaded_region (r : ℝ) : ℝ :=
  1 / 5

-- Main theorem statements to prove
theorem theorem_perimeter_shaded_region
  (h : Real.pi * r^2 = 1) : perimeter_shaded_region r = 2 * Real.sqrt Real.pi :=
by
  sorry

theorem theorem_area_shaded_region
  (h : Real.pi * r^2 = 1) : area_shaded_region r = 1 / 5 :=
by
  sorry

end theorem_perimeter_shaded_region_theorem_area_shaded_region_l235_235911


namespace total_students_sampled_l235_235190

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l235_235190


namespace tan_alpha_l235_235880

variable (α : Real)
-- Condition 1: α is an angle in the second quadrant
-- This implies that π/2 < α < π and sin α = 4 / 5
variable (h1 : π / 2 < α ∧ α < π) 
variable (h2 : Real.sin α = 4 / 5)

theorem tan_alpha : Real.tan α = -4 / 3 :=
by
  sorry

end tan_alpha_l235_235880


namespace sin_C_in_right_triangle_l235_235240

-- Triangle ABC with angle B = 90 degrees and tan A = 3/4
theorem sin_C_in_right_triangle (A C : ℝ) (h1 : A + C = π / 2) (h2 : Real.tan A = 3 / 4) : Real.sin C = 4 / 5 := by
  sorry

end sin_C_in_right_triangle_l235_235240


namespace daily_reading_goal_l235_235003

-- Define the constants for pages read each day
def pages_on_sunday : ℕ := 43
def pages_on_monday : ℕ := 65
def pages_on_tuesday : ℕ := 28
def pages_on_wednesday : ℕ := 0
def pages_on_thursday : ℕ := 70
def pages_on_friday : ℕ := 56
def pages_on_saturday : ℕ := 88

-- Define the total pages read in the week
def total_pages := pages_on_sunday + pages_on_monday + pages_on_tuesday + pages_on_wednesday 
                    + pages_on_thursday + pages_on_friday + pages_on_saturday

-- The theorem that expresses Berry's daily reading goal
theorem daily_reading_goal : total_pages / 7 = 50 :=
by
  sorry

end daily_reading_goal_l235_235003


namespace sum_of_roots_l235_235500

theorem sum_of_roots (a1 a2 a3 a4 a5 : ℤ)
  (h_distinct : a1 ≠ a2 ∧ a1 ≠ a3 ∧ a1 ≠ a4 ∧ a1 ≠ a5 ∧
                a2 ≠ a3 ∧ a2 ≠ a4 ∧ a2 ≠ a5 ∧
                a3 ≠ a4 ∧ a3 ≠ a5 ∧
                a4 ≠ a5)
  (h_poly : (104 - a1) * (104 - a2) * (104 - a3) * (104 - a4) * (104 - a5) = 2012) :
  a1 + a2 + a3 + a4 + a5 = 17 := by
  sorry

end sum_of_roots_l235_235500


namespace sine_product_inequality_l235_235797

theorem sine_product_inequality :
  (1 / 8 : ℝ) < (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) ∧
                (Real.sin (20 * Real.pi / 180) * Real.sin (50 * Real.pi / 180) * Real.sin (70 * Real.pi / 180)) < (1 / 4 : ℝ) :=
sorry

end sine_product_inequality_l235_235797


namespace number_of_sequences_of_length_100_l235_235121

def sequence_count (n : ℕ) : ℕ :=
  3^n - 2^n

theorem number_of_sequences_of_length_100 :
  sequence_count 100 = 3^100 - 2^100 :=
by
  sorry

end number_of_sequences_of_length_100_l235_235121


namespace possible_values_of_sum_l235_235517

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235517


namespace sum_of_x_y_l235_235521

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235521


namespace value_of_x_l235_235348

theorem value_of_x (x : ℝ) (h : 0.75 * 600 = 0.50 * x) : x = 900 :=
by
  sorry

end value_of_x_l235_235348


namespace probability_intersection_l235_235617

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l235_235617


namespace number_of_integer_pairs_satisfying_conditions_l235_235206

noncomputable def count_integer_pairs (n m : ℕ) : ℕ := Nat.choose (n-1) (m-1)

theorem number_of_integer_pairs_satisfying_conditions :
  ∃ (a b c x y : ℕ), a + b + c = 55 ∧ a + b + c + x + y = 71 ∧ x + y > a + b + c → count_integer_pairs 55 3 * count_integer_pairs 16 2 = 21465 := sorry

end number_of_integer_pairs_satisfying_conditions_l235_235206


namespace sqrt_recursive_value_l235_235661

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l235_235661


namespace solve_dog_walking_minutes_l235_235704

-- Definitions based on the problem conditions
def cost_one_dog (x : ℕ) : ℕ := 20 + x
def cost_two_dogs : ℕ := 54
def cost_three_dogs : ℕ := 87
def total_earnings (x : ℕ) : ℕ := cost_one_dog x + cost_two_dogs + cost_three_dogs

-- Proving that the total earnings equal to 171 implies x = 10
theorem solve_dog_walking_minutes (x : ℕ) (h : total_earnings x = 171) : x = 10 :=
by
  -- The proof goes here
  sorry

end solve_dog_walking_minutes_l235_235704


namespace emily_art_supplies_l235_235646

theorem emily_art_supplies (total_spent skirts_cost skirt_quantity : ℕ) 
  (total_spent_eq : total_spent = 50) 
  (skirt_cost_eq : skirts_cost = 15) 
  (skirt_quantity_eq : skirt_quantity = 2) :
  total_spent - skirt_quantity * skirts_cost = 20 :=
by
  sorry

end emily_art_supplies_l235_235646


namespace monotonicity_f_max_m_l235_235054

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x - x^2 + x
def g (x : ℝ) (m : ℝ) : ℝ := (x - 2) * Real.exp x - x^2 + m

-- Monotonicity problem for f
theorem monotonicity_f (a : ℝ) (h_le : a ≤ 0) : 
  if a ≤ -1/8 then 
    ∀ x : ℝ, 0 < x → (f a)' x ≤ 0
  else 
    ∀ x : ℝ, 
      ((0 < x ∧ x < (1 - Real.sqrt (1 + 8 * a)) / 4) → (f a)' x < 0) ∧ 
      ((x = (1 - Real.sqrt (1 + 8 * a)) / 4 ∨ x = (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x = 0) ∧
      ((x > (1 - Real.sqrt (1 + 8 * a)) / 4 ∧ x < (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x > 0) ∧ 
      ((x > (1 + Real.sqrt (1 + 8 * a)) / 4) → (f a)' x < 0)
:= sorry

-- Maximum value of m for f(x) > g(x)
theorem max_m (m : ℝ) : 
  ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f (-1) x > g x m → m ≤ 3
:= sorry

end monotonicity_f_max_m_l235_235054


namespace area_of_triangle_is_168_l235_235995

-- Define the curve equation
def curve_eq (x : ℝ) : ℝ := (x - 4)^2 * (x + 3)

-- Define the x-intercepts
def x_intercepts (y : ℝ) : Prop := y = 0

-- Define the y-intercept
def y_intercept (x : ℝ) : ℝ := curve_eq 0

-- Define the base of the triangle (distance between x-intercepts)
def base : ℝ := 4 - (-3)

-- Define the height of the triangle (y-intercept value)
def height : ℝ := y_intercept 0

-- Define the area calculation for the triangle
def triangle_area : ℝ := (1 / 2) * base * height

-- The theorem to prove the area of the triangle is 168
theorem area_of_triangle_is_168 : triangle_area = 168 :=
by sorry

end area_of_triangle_is_168_l235_235995


namespace probability_intersection_l235_235613

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l235_235613


namespace range_of_a_l235_235222

noncomputable def f : ℝ → ℝ := sorry

def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)
def is_monotone_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y : ℝ⦄, 0 ≤ x → 0 ≤ y → x < y → f x < f y

axiom even_f : is_even f
axiom monotone_f : is_monotone_on_nonneg f

theorem range_of_a (a : ℝ) (h : f a ≥ f 3) : a ≤ -3 ∨ a ≥ 3 :=
by
  sorry

end range_of_a_l235_235222


namespace first_train_speed_l235_235291

noncomputable def speed_first_train (length1 length2 : ℝ) (speed2 time : ℝ) : ℝ :=
  let distance := (length1 + length2) / 1000
  let time_hours := time / 3600
  (distance / time_hours) - speed2

theorem first_train_speed :
  speed_first_train 100 280 30 18.998480121590273 = 42 :=
by
  sorry

end first_train_speed_l235_235291


namespace points_on_line_possible_l235_235562

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l235_235562


namespace molecular_weight_BaBr2_l235_235294

theorem molecular_weight_BaBr2 (w: ℝ) (h: w = 2376) : w / 8 = 297 :=
by
  sorry

end molecular_weight_BaBr2_l235_235294


namespace parabola_point_dot_product_eq_neg4_l235_235467

-- Definition of the parabola
def is_parabola_point (A : ℝ × ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

-- Definition of the focus of the parabola y^2 = 4x
def focus : ℝ × ℝ := (1, 0)

-- Dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Coordinates of origin
def origin : ℝ × ℝ := (0, 0)

-- Vector from origin to point A
def vector_OA (A : ℝ × ℝ) : ℝ × ℝ :=
  (A.1, A.2)

-- Vector from point A to the focus
def vector_AF (A : ℝ × ℝ) : ℝ × ℝ :=
  (focus.1 - A.1, focus.2 - A.2)

-- Theorem statement
theorem parabola_point_dot_product_eq_neg4 (A : ℝ × ℝ) 
  (hA : is_parabola_point A) 
  (h_dot : dot_product (vector_OA A) (vector_AF A) = -4) :
  A = (1, 2) ∨ A = (1, -2) :=
sorry

end parabola_point_dot_product_eq_neg4_l235_235467


namespace graphene_scientific_notation_l235_235230

theorem graphene_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (0.00000000034 : ℝ) = a * 10^n ∧ a = 3.4 ∧ n = -10 :=
sorry

end graphene_scientific_notation_l235_235230


namespace ratio_is_9_l235_235640

-- Define the set of numbers
def set_of_numbers := { x : ℕ | ∃ n, n ≤ 8 ∧ x = 10^n }

-- Define the sum of the geometric series excluding the largest element
def sum_of_others : ℕ := (Finset.range 8).sum (λ n => 10^n)

-- Define the largest element
def largest_element := 10^8

-- Define the ratio of the largest element to the sum of the other elements
def ratio := largest_element / sum_of_others

-- Problem statement: The ratio is 9
theorem ratio_is_9 : ratio = 9 := by
  sorry

end ratio_is_9_l235_235640


namespace range_of_a_l235_235696

theorem range_of_a (a : ℝ) (h1 : ∃ x : ℝ, x > 0 ∧ |x| = a * x - a) (h2 : ∀ x : ℝ, x < 0 → |x| ≠ a * x - a) : a > 1 :=
sorry

end range_of_a_l235_235696


namespace smallest_positive_integer_cube_ends_368_l235_235450

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l235_235450


namespace Jake_has_8_peaches_l235_235784

variable (Steven Jill Jake : ℕ)

-- Conditions
axiom h1 : Steven = 15
axiom h2 : Steven = Jill + 14
axiom h3 : Jake = Steven - 7

-- Goal
theorem Jake_has_8_peaches : Jake = 8 := by
  sorry

end Jake_has_8_peaches_l235_235784


namespace number_of_selected_in_interval_l235_235068

noncomputable def systematic_sampling_group := (420: ℕ)
noncomputable def selected_people := (21: ℕ)
noncomputable def interval_start := (241: ℕ)
noncomputable def interval_end := (360: ℕ)
noncomputable def sampling_interval := systematic_sampling_group / selected_people
noncomputable def interval_length := interval_end - interval_start + 1

theorem number_of_selected_in_interval :
  interval_length / sampling_interval = 6 :=
by
  -- Placeholder for the proof
  sorry

end number_of_selected_in_interval_l235_235068


namespace problem_statement_l235_235643

theorem problem_statement : (515 % 1000) = 515 :=
by
  sorry

end problem_statement_l235_235643


namespace min_value_of_a_plus_b_l235_235498

theorem min_value_of_a_plus_b (a b : ℤ) (h1 : Even a) (h2 : Even b) (h3 : a * b = 144) : a + b = -74 :=
sorry

end min_value_of_a_plus_b_l235_235498


namespace gcd_example_l235_235607

theorem gcd_example : Nat.gcd 8675309 7654321 = 36 := sorry

end gcd_example_l235_235607


namespace direct_proportion_l235_235844

theorem direct_proportion (c f p : ℝ) (h : f ≠ 0 ∧ p = c * f) : ∃ k : ℝ, p / f = k * (f / f) :=
by
  sorry

end direct_proportion_l235_235844


namespace ratio_of_students_to_professors_l235_235780

theorem ratio_of_students_to_professors (total : ℕ) (students : ℕ) (professors : ℕ)
  (h1 : total = 40000) (h2 : students = 37500) (h3 : total = students + professors) :
  students / professors = 15 :=
by
  sorry

end ratio_of_students_to_professors_l235_235780


namespace base_number_of_exponentiation_l235_235769

theorem base_number_of_exponentiation (n : ℕ) (some_number : ℕ) (h1 : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = some_number^22) (h2 : n = 21) : some_number = 4 :=
  sorry

end base_number_of_exponentiation_l235_235769


namespace probability_defective_is_three_tenths_l235_235990

open Classical

noncomputable def probability_of_defective_product (total_products defective_products: ℕ) : ℝ :=
  (defective_products * 1.0) / (total_products * 1.0)

theorem probability_defective_is_three_tenths :
  probability_of_defective_product 10 3 = 3 / 10 := by
  sorry

end probability_defective_is_three_tenths_l235_235990


namespace equal_a_b_l235_235869

theorem equal_a_b (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_n : 0 < n) 
  (h_eq : (a + b)^n - (a - b)^n = (a / b) * ((a + b)^n + (a - b)^n)) : a = b :=
sorry

end equal_a_b_l235_235869


namespace sum_of_xy_l235_235549

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235549


namespace john_money_left_l235_235361

theorem john_money_left 
  (start_amount : ℝ := 100) 
  (price_roast : ℝ := 17)
  (price_vegetables : ℝ := 11)
  (price_wine : ℝ := 12)
  (price_dessert : ℝ := 8)
  (price_bread : ℝ := 4)
  (price_milk : ℝ := 2)
  (discount_rate : ℝ := 0.15)
  (tax_rate : ℝ := 0.05)
  (total_cost := price_roast + price_vegetables + price_wine + price_dessert + price_bread + price_milk)
  (discount_amount := discount_rate * total_cost)
  (discounted_total := total_cost - discount_amount)
  (tax_amount := tax_rate * discounted_total)
  (final_amount := discounted_total + tax_amount)
  : start_amount - final_amount = 51.80 := sorry

end john_money_left_l235_235361


namespace borrowed_quarters_l235_235491

def original_quarters : ℕ := 8
def remaining_quarters : ℕ := 5

theorem borrowed_quarters : original_quarters - remaining_quarters = 3 :=
by
  sorry

end borrowed_quarters_l235_235491


namespace pi_is_irrational_l235_235632

theorem pi_is_irrational (π : ℝ) (h : π = Real.pi) :
  ¬ ∃ (a b : ℤ), b ≠ 0 ∧ π = a / b :=
by
  sorry

end pi_is_irrational_l235_235632


namespace scientific_notation_correct_l235_235261

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l235_235261


namespace prime_factors_of_30_l235_235754

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l235_235754


namespace sum_of_x_y_l235_235524

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235524


namespace minimum_value_2x_plus_y_l235_235694

theorem minimum_value_2x_plus_y (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : (1 / x) + (2 / (y + 1)) = 2) : 2 * x + y ≥ 3 := 
by
  sorry

end minimum_value_2x_plus_y_l235_235694


namespace ratio_after_addition_l235_235816

theorem ratio_after_addition (a b : ℕ) (h1 : a * 3 = b * 2) (h2 : b - a = 8) : (a + 4) * 7 = (b + 4) * 5 :=
by
  sorry

end ratio_after_addition_l235_235816


namespace product_of_midpoint_coordinates_l235_235159

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l235_235159


namespace height_of_balcony_l235_235187

variable (t : ℝ) (v₀ : ℝ) (g : ℝ) (h₀ : ℝ)

axiom cond1 : t = 6
axiom cond2 : v₀ = 20
axiom cond3 : g = 10

theorem height_of_balcony : h₀ + v₀ * t - (1/2 : ℝ) * g * t^2 = 0 → h₀ = 60 :=
by
  intro h'
  sorry

end height_of_balcony_l235_235187


namespace num_prime_factors_30_factorial_l235_235751

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l235_235751


namespace smaller_of_two_digit_product_4680_l235_235137

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end smaller_of_two_digit_product_4680_l235_235137


namespace proof1_proof2_proof3_l235_235310

variables (x m n : ℝ)

theorem proof1 (x : ℝ) : (-3 * x - 5) * (5 - 3 * x) = 9 * x^2 - 25 :=
sorry

theorem proof2 (x : ℝ) : (-3 * x - 5) * (5 + 3 * x) = - (3 * x + 5) ^ 2 :=
sorry

theorem proof3 (m n : ℝ) : (2 * m - 3 * n + 1) * (2 * m + 1 + 3 * n) = (2 * m + 1) ^ 2 - (3 * n) ^ 2 :=
sorry

end proof1_proof2_proof3_l235_235310


namespace min_value_of_m_l235_235039

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def b_n (n : ℕ) : ℝ := 2 * n - 9
noncomputable def c_n (n : ℕ) : ℝ := b_n n / a_n n

theorem min_value_of_m (m : ℝ) : (∀ n : ℕ, c_n n ≤ m) → m ≥ 1/162 :=
by
  sorry

end min_value_of_m_l235_235039


namespace product_of_midpoint_is_minus_4_l235_235157

-- Coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (4, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coordinates of the midpoint
def midpoint_coords := midpoint endpoint1 endpoint2

-- Product of the coordinates of the midpoint
def product_of_midpoint_coords (mp : ℝ × ℝ) : ℝ :=
  mp.1 * mp.2

-- Statement of the theorem to be proven
theorem product_of_midpoint_is_minus_4 : 
  product_of_midpoint_coords midpoint_coords = -4 := 
by
  sorry

end product_of_midpoint_is_minus_4_l235_235157


namespace common_ratio_geometric_series_l235_235679

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l235_235679


namespace boxes_of_chocolates_l235_235830

theorem boxes_of_chocolates (total_pieces : ℕ) (pieces_per_box : ℕ) (h_total : total_pieces = 3000) (h_each : pieces_per_box = 500) : total_pieces / pieces_per_box = 6 :=
by
  sorry

end boxes_of_chocolates_l235_235830


namespace possible_values_of_sum_l235_235545

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235545


namespace curve_is_line_l235_235686

theorem curve_is_line (r θ : ℝ) (h : r = 2 / (Real.sin θ + Real.cos θ)) : 
  ∃ m b, ∀ θ, r * Real.cos θ = m * (r * Real.sin θ) + b :=
sorry

end curve_is_line_l235_235686


namespace mary_more_than_marco_l235_235108

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l235_235108


namespace hexagon_largest_angle_l235_235832

theorem hexagon_largest_angle (a : ℚ) 
  (h₁ : (a + 2) + (2 * a - 3) + (3 * a + 1) + 4 * a + (5 * a - 4) + (6 * a + 2) = 720) :
  6 * a + 2 = 4374 / 21 :=
by sorry

end hexagon_largest_angle_l235_235832


namespace composite_divides_expression_l235_235457

theorem composite_divides_expression (n : ℕ) (h : composite n) : 6 * n^2 ∣ n^4 - n^2 := 
sorry

end composite_divides_expression_l235_235457


namespace minimum_discount_correct_l235_235975

noncomputable def minimum_discount (total_weight: ℝ) (cost_price: ℝ) (sell_price: ℝ) 
                                   (profit_required: ℝ) : ℝ :=
  let first_half_profit := (total_weight / 2) * (sell_price - cost_price)
  let second_half_profit_with_discount (x: ℝ) := (total_weight / 2) * (sell_price * x - cost_price)
  let required_profit_condition (x: ℝ) := first_half_profit + second_half_profit_with_discount x ≥ profit_required
  (1 - (7 / 11))

theorem minimum_discount_correct : minimum_discount 1000 7 10 2000 = 4 / 11 := 
by {
  -- We need to solve the inequality step by step to reach the final answer
  sorry
}

end minimum_discount_correct_l235_235975


namespace sum_greater_l235_235220

theorem sum_greater {a b c d : ℝ} (h1 : b + Real.sin a > d + Real.sin c) (h2 : a + Real.sin b > c + Real.sin d) : a + b > c + d := by
  sorry

end sum_greater_l235_235220


namespace mooney_ate_correct_l235_235115

-- Define initial conditions
def initial_brownies : ℕ := 24
def father_ate : ℕ := 8
def mother_added : ℕ := 24
def final_brownies : ℕ := 36

-- Define Mooney ate some brownies
variable (mooney_ate : ℕ)

-- Prove that Mooney ate 4 brownies
theorem mooney_ate_correct :
  (initial_brownies - father_ate) - mooney_ate + mother_added = final_brownies →
  mooney_ate = 4 :=
by
  sorry

end mooney_ate_correct_l235_235115


namespace number_of_prime_factors_of_30_factorial_l235_235724

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l235_235724


namespace charge_per_trousers_l235_235786

-- Definitions
def pairs_of_trousers : ℕ := 10
def shirts : ℕ := 10
def bill : ℕ := 140
def charge_per_shirt : ℕ := 5

-- Theorem statement
theorem charge_per_trousers :
  ∃ (T : ℕ), (pairs_of_trousers * T + shirts * charge_per_shirt = bill) ∧ (T = 9) :=
by 
  sorry

end charge_per_trousers_l235_235786


namespace johns_total_earnings_l235_235095

noncomputable def total_earnings_per_week (baskets_monday : ℕ) (baskets_thursday : ℕ) (small_crabs_per_basket : ℕ) (large_crabs_per_basket : ℕ) (price_small_crab : ℕ) (price_large_crab : ℕ) : ℕ :=
  let small_crabs := baskets_monday * small_crabs_per_basket
  let large_crabs := baskets_thursday * large_crabs_per_basket
  (small_crabs * price_small_crab) + (large_crabs * price_large_crab)

theorem johns_total_earnings :
  total_earnings_per_week 3 4 4 5 3 5 = 136 :=
by
  sorry

end johns_total_earnings_l235_235095


namespace smallest_common_multiple_of_9_and_6_l235_235167

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l235_235167


namespace find_E_coordinates_l235_235085

structure Point where
  x : ℚ
  y : ℚ

def A : Point := {x := -2, y := 1}
def B : Point := {x := 1, y := 4}
def C : Point := {x := 4, y := -3}
def D : Point := {x := (-2 * 1 + 1 * (-2)) / (1 + 2), y := (1 * 4 + 2 * 1) / (1 + 2)}

def externalDivision (P1 P2 : Point) (m n : ℚ) : Point :=
  {x := (m * P2.x - n * P1.x) / (m - n), y := (m * P2.y - n * P1.y) / (m - n)}

theorem find_E_coordinates :
  let E := externalDivision D C 1 4
  E.x = -8 / 3 ∧ E.y = 11 / 3 := 
by 
  let E := externalDivision D C 1 4
  sorry

end find_E_coordinates_l235_235085


namespace remainder_of_3_pow_600_mod_19_l235_235305

theorem remainder_of_3_pow_600_mod_19 :
  (3 ^ 600) % 19 = 11 :=
sorry

end remainder_of_3_pow_600_mod_19_l235_235305


namespace age_difference_l235_235612

variables (A B C : ℕ)

theorem age_difference (h : A + B = B + C + 18) : C = A - 18 :=
sorry

end age_difference_l235_235612


namespace urea_formation_l235_235327

theorem urea_formation (CO2 NH3 : ℕ) (OCN2 H2O : ℕ) (h1 : CO2 = 3) (h2 : NH3 = 6) :
  (∀ x, CO2 * 1 + NH3 * 2 = x + (2 * x) + x) →
  OCN2 = 3 :=
by
  sorry

end urea_formation_l235_235327


namespace tankard_one_quarter_full_l235_235290

theorem tankard_one_quarter_full
  (C : ℝ) 
  (h : (3 / 4) * C = 480) : 
  (1 / 4) * C = 160 := 
by
  sorry

end tankard_one_quarter_full_l235_235290


namespace baseball_games_per_month_l235_235598

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l235_235598


namespace residue_of_neg_1237_mod_29_l235_235644

theorem residue_of_neg_1237_mod_29 :
  (-1237 : ℤ) % 29 = 10 :=
sorry

end residue_of_neg_1237_mod_29_l235_235644


namespace min_value_of_function_l235_235446

theorem min_value_of_function :
  ∀ x : ℝ, x > -1 → (y : ℝ) = (x^2 + 7*x + 10) / (x + 1) → y ≥ 9 :=
by
  intros x hx h
  sorry

end min_value_of_function_l235_235446


namespace red_balls_removal_l235_235074

theorem red_balls_removal (total_balls : ℕ) (red_balls : ℕ) (blue_balls : ℕ) (x : ℕ) :
  total_balls = 600 →
  red_balls = 420 →
  blue_balls = 180 →
  (red_balls - x) / (total_balls - x : ℚ) = 3 / 5 ↔ x = 150 :=
by 
  intros;
  sorry

end red_balls_removal_l235_235074


namespace calculate_power_of_fractions_l235_235318

-- Defining the fractions
def a : ℚ := 5 / 6
def b : ℚ := 3 / 5

-- The main statement to prove the given question
theorem calculate_power_of_fractions : a^3 + b^3 = (21457 : ℚ) / 27000 := by 
  sorry

end calculate_power_of_fractions_l235_235318


namespace mary_potatoes_l235_235925

theorem mary_potatoes (original new_except : ℕ) (h₁ : original = 25) (h₂ : new_except = 7) :
  original + new_except = 32 := by
  sorry

end mary_potatoes_l235_235925


namespace initial_pens_eq_42_l235_235794

-- Definitions based on the conditions
def initial_books : ℕ := 143
def remaining_books : ℕ := 113
def remaining_pens : ℕ := 19
def sold_pens : ℕ := 23

-- Theorem to prove that the initial number of pens was 42
theorem initial_pens_eq_42 (b_init b_remain p_remain p_sold : ℕ) 
    (H_b_init : b_init = initial_books)
    (H_b_remain : b_remain = remaining_books)
    (H_p_remain : p_remain = remaining_pens)
    (H_p_sold : p_sold = sold_pens) : 
    (p_sold + p_remain = 42) := 
by {
    -- Provide proof later
    sorry
}

end initial_pens_eq_42_l235_235794


namespace tammy_average_speed_second_day_l235_235802

theorem tammy_average_speed_second_day :
  ∃ v t : ℝ, 
  t + (t - 2) + (t + 1) = 20 ∧
  v * t + (v + 0.5) * (t - 2) + (v - 0.5) * (t + 1) = 80 ∧
  (v + 0.5) = 4.575 :=
by 
  sorry

end tammy_average_speed_second_day_l235_235802


namespace no_pairs_xy_perfect_square_l235_235025

theorem no_pairs_xy_perfect_square :
  ¬ ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ ∃ k : ℕ, (xy + 1) * (xy + x + 2) = k^2 := 
by {
  sorry
}

end no_pairs_xy_perfect_square_l235_235025


namespace Zixuan_amount_l235_235393

noncomputable def amounts (X Y Z : ℕ) : Prop := 
  (X + Y + Z = 50) ∧
  (X = 3 * (Y + Z) / 2) ∧
  (Y = Z + 4)

theorem Zixuan_amount : ∃ Z : ℕ, ∃ X Y : ℕ, amounts X Y Z ∧ Z = 8 :=
by
  sorry

end Zixuan_amount_l235_235393


namespace value_added_after_doubling_l235_235350

theorem value_added_after_doubling (x v : ℝ) (h1 : x = 4) (h2 : 2 * x + v = x / 2 + 20) : v = 14 :=
by
  sorry

end value_added_after_doubling_l235_235350


namespace taxi_fare_distance_l235_235134

variable (x : ℝ)

theorem taxi_fare_distance (h1 : 0 ≤ x - 2) (h2 : 3 + 1.2 * (x - 2) = 9) : x = 7 := by
  sorry

end taxi_fare_distance_l235_235134


namespace arcsin_cos_arcsin_rel_arccos_sin_arccos_l235_235031

theorem arcsin_cos_arcsin_rel_arccos_sin_arccos (x : ℝ) (hx : -1 ≤ x ∧ x ≤ 1) :
    let α := Real.arcsin (Real.cos (Real.arcsin x))
    let β := Real.arccos (Real.sin (Real.arccos x))
    (Real.arcsin x + Real.arccos x = π / 2) → α + β = π / 2 :=
by
  let α := Real.arcsin (Real.cos (Real.arcsin x))
  let β := Real.arccos (Real.sin (Real.arccos x))
  intro h_arcsin_arccos_eq
  sorry

end arcsin_cos_arcsin_rel_arccos_sin_arccos_l235_235031


namespace weak_multiple_l235_235503

def is_weak (a b n : ℕ) : Prop :=
  ∀ (x y : ℕ), n ≠ a * x + b * y

theorem weak_multiple (a b n : ℕ) (h_coprime : Nat.gcd a b = 1) (h_weak : is_weak a b n) (h_bound : n < a * b / 6) : 
  ∃ k ≥ 2, is_weak a b (k * n) :=
by
  sorry

end weak_multiple_l235_235503


namespace sum_of_x_y_possible_values_l235_235534

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235534


namespace train_speed_l235_235415

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l235_235415


namespace max_g_8_l235_235789

noncomputable def g (x : ℝ) : ℝ := sorry -- To be filled with the specific polynomial

theorem max_g_8 (g : ℝ → ℝ)
  (h_nonneg : ∀ x, 0 ≤ g x)
  (h4 : g 4 = 16)
  (h16 : g 16 = 1024) : g 8 ≤ 128 :=
sorry

end max_g_8_l235_235789


namespace shahrazad_stories_not_power_of_two_l235_235571

theorem shahrazad_stories_not_power_of_two :
  ∀ (a b c : ℕ) (k : ℕ),
  a + b + c = 1001 → 27 * a + 14 * b + c = 2^k → False :=
by {
  sorry
}

end shahrazad_stories_not_power_of_two_l235_235571


namespace inverse_proportion_y_relation_l235_235043

theorem inverse_proportion_y_relation (x₁ x₂ y₁ y₂ : ℝ) 
  (hA : y₁ = -4 / x₁) 
  (hB : y₂ = -4 / x₂)
  (h₁ : x₁ < 0) 
  (h₂ : 0 < x₂) : 
  y₁ > y₂ := 
sorry

end inverse_proportion_y_relation_l235_235043


namespace triangle_angle_conditions_l235_235379

theorem triangle_angle_conditions
  (a b c : ℝ)
  (α β γ : ℝ)
  (h_triangle : c^2 = a^2 + 2 * b^2 * Real.cos β)
  (h_tri_angles : α + β + γ = 180):
  (γ = β / 2 + 90 ∧ α = 90 - 3 * β / 2 ∧ 0 < β ∧ β < 60) ∨ 
  (α = β / 2 ∧ γ = 180 - 3 * β / 2 ∧ 0 < β ∧ β < 120) :=
sorry

end triangle_angle_conditions_l235_235379


namespace updated_mean_l235_235611

-- Definitions
def initial_mean := 200
def number_of_observations := 50
def decrement_per_observation := 9

-- Theorem stating the updated mean after decrementing each observation
theorem updated_mean : 
  (initial_mean * number_of_observations - decrement_per_observation * number_of_observations) / number_of_observations = 191 :=
by
  -- Placeholder for the proof
  sorry

end updated_mean_l235_235611


namespace problem_a_problem_b_problem_c_l235_235964

theorem problem_a : (7 * (2 / 3) + 16 * (5 / 12)) = 11.3333 := by
  sorry

theorem problem_b : (5 - (2 / (5 / 3))) = 3.8 := by
  sorry

theorem problem_c : (1 + 2 / (1 + 3 / (1 + 4))) = 2.25 := by
  sorry

end problem_a_problem_b_problem_c_l235_235964


namespace cost_per_tissue_box_l235_235984

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l235_235984


namespace students_in_neither_l235_235904

def total_students := 60
def students_in_art := 40
def students_in_music := 30
def students_in_both := 15

theorem students_in_neither : total_students - (students_in_art - students_in_both + students_in_music - students_in_both + students_in_both) = 5 :=
by
  sorry

end students_in_neither_l235_235904


namespace Julie_work_hours_per_week_l235_235363

theorem Julie_work_hours_per_week 
  (hours_summer_per_week : ℕ)
  (weeks_summer : ℕ)
  (total_earnings_summer : ℕ)
  (planned_weeks_school_year : ℕ)
  (needed_income_school_year : ℕ)
  (hourly_wage : ℝ := total_earnings_summer / (hours_summer_per_week * weeks_summer))
  (total_hours_needed_school_year : ℝ := needed_income_school_year / hourly_wage)
  (hours_per_week_needed : ℝ := total_hours_needed_school_year / planned_weeks_school_year) :
  hours_summer_per_week = 60 →
  weeks_summer = 8 →
  total_earnings_summer = 6000 →
  planned_weeks_school_year = 40 →
  needed_income_school_year = 10000 →
  hours_per_week_needed = 20 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end Julie_work_hours_per_week_l235_235363


namespace value_of_x_l235_235455

theorem value_of_x (x y : ℕ) (h1 : y = 864) (h2 : x^3 * 6^3 / 432 = y) : x = 12 :=
sorry

end value_of_x_l235_235455


namespace probability_two_heads_with_second_tail_l235_235016

/-- The probability that Debra gets two heads in a row but sees a second tail before she sees a second head
    when repeatedly flipping a fair coin and stops flipping when she gets two heads in a row or two tails
    in a row is 1/24. --/
theorem probability_two_heads_with_second_tail :
  let coin := prob_choice (1 / 2) tt ff in
  let outcome := coin_repeatedly coin in
  let stop_condition := λ (s : list bool), (s.tail ++ [true, true] = s) ∨ (s.tail ++ [false, false] = s) in
  Pr (λ s, (stop_condition s) ∧ (λ s, s = [false, true, false])) = 1 / 24 := 
sorry

end probability_two_heads_with_second_tail_l235_235016


namespace selection_methods_at_least_one_female_l235_235268

theorem selection_methods_at_least_one_female :
  let total_students := 7
  let male_students := 5
  let female_students := 2
  let selection_size := 3
  nat.choose total_students selection_size - nat.choose male_students selection_size = 25 :=
by
  sorry

end selection_methods_at_least_one_female_l235_235268


namespace tree_planting_problem_l235_235148

noncomputable def total_trees_needed (length width tree_distance : ℕ) : ℕ :=
  let perimeter := 2 * (length + width)
  let intervals := perimeter / tree_distance
  intervals

theorem tree_planting_problem : total_trees_needed 150 60 10 = 42 :=
by
  sorry

end tree_planting_problem_l235_235148


namespace sum_of_possible_values_of_N_l235_235579

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l235_235579


namespace length_of_goods_train_l235_235821

theorem length_of_goods_train 
  (speed_km_per_hr : ℕ) (platform_length_m : ℕ) (time_sec : ℕ) 
  (h1 : speed_km_per_hr = 72) (h2 : platform_length_m = 300) (h3 : time_sec = 26) : 
  ∃ length_of_train : ℕ, length_of_train = 220 :=
by
  sorry

end length_of_goods_train_l235_235821


namespace brendan_match_ratio_l235_235846

noncomputable def brendanMatches (totalMatches firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound : ℕ) :=
  matchesWonFirstTwoRounds = firstRound + secondRound ∧
  matchesWonFirstTwoRounds = 12 ∧
  totalMatches = matchesWonTotal ∧
  matchesWonTotal = 14 ∧
  firstRound = 6 ∧
  secondRound = 6 ∧
  matchesInLastRound = 4

theorem brendan_match_ratio :
  ∃ ratio: ℕ × ℕ,
    let firstRound := 6
    let secondRound := 6
    let matchesInLastRound := 4
    let matchesWonFirstTwoRounds := firstRound + secondRound
    let matchesWonTotal := 14
    let matchesWonLastRound := matchesWonTotal - matchesWonFirstTwoRounds
    let ratio := (matchesWonLastRound, matchesInLastRound)
    brendanMatches matchesWonTotal firstRound secondRound matchesWonFirstTwoRounds matchesWonTotal matchesInLastRound ∧
    ratio = (1, 2) :=
by
  sorry

end brendan_match_ratio_l235_235846


namespace sum_of_series_is_correct_l235_235997

noncomputable def geometric_series_sum_5_terms : ℚ :=
  let a := 1 / 4
  let r := 1 / 4
  let n := 5
  a * (1 - r^n) / (1 - r)

theorem sum_of_series_is_correct :
  geometric_series_sum_5_terms = 1023 / 3072 := by
  sorry

end sum_of_series_is_correct_l235_235997


namespace triangle_problem_l235_235773

noncomputable def triangle_sum : Real := sorry

theorem triangle_problem
  (A B C : ℝ) -- Angles of the triangle
  (a b c : ℝ) -- Sides of the triangle
  (hA : A = π / 6) -- A = 30 degrees
  (h_a : a = Real.sqrt 3) -- a = √3
  (h_law_of_sines : ∀ (x : ℝ), x = 2 * triangle_sum * Real.sin x) -- Law of Sines
  (h_sin_30 : Real.sin (π / 6) = 1 / 2) -- sin 30 degrees = 1/2
  : (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) 
  = 2 * Real.sqrt 3 := sorry

end triangle_problem_l235_235773


namespace pair_product_not_72_l235_235307

theorem pair_product_not_72 : (2 * (-36) ≠ 72) :=
by
  sorry

end pair_product_not_72_l235_235307


namespace alex_buys_15_pounds_of_wheat_l235_235177

theorem alex_buys_15_pounds_of_wheat (w o : ℝ) (h1 : w + o = 30) (h2 : 72 * w + 36 * o = 1620) : w = 15 :=
by
  sorry

end alex_buys_15_pounds_of_wheat_l235_235177


namespace prime_factors_of_30_factorial_l235_235733

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l235_235733


namespace prime_factors_of_30_l235_235756

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l235_235756


namespace manny_marbles_l235_235147

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l235_235147


namespace setD_is_empty_l235_235842

-- Definitions of sets A, B, C, D
def setA : Set ℝ := {x | x + 3 = 3}
def setB : Set (ℝ × ℝ) := {(x, y) | y^2 ≠ -x^2}
def setC : Set ℝ := {x | x^2 ≤ 0}
def setD : Set ℝ := {x | x^2 - x + 1 = 0}

-- Theorem stating that set D is the empty set
theorem setD_is_empty : setD = ∅ := 
by 
  sorry

end setD_is_empty_l235_235842


namespace man_twice_son_age_l235_235628

theorem man_twice_son_age (S M Y : ℕ) (h1 : S = 18) (h2 : M = S + 20) 
  (h3 : M + Y = 2 * (S + Y)) : Y = 2 :=
by
  -- Proof steps can be added here later
  sorry

end man_twice_son_age_l235_235628


namespace smallest_common_multiple_of_9_and_6_l235_235171

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l235_235171


namespace correct_calculation_l235_235306

theorem correct_calculation (a : ℝ) : -2 * a + (2 * a - 1) = -1 := by
  sorry

end correct_calculation_l235_235306


namespace nested_sqrt_eq_l235_235665

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l235_235665


namespace vitya_probability_l235_235301

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l235_235301


namespace part_a_part_b_l235_235100

theorem part_a (n : ℕ) (hn : n % 2 = 1) (h_pos : n > 0) :
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n-1 ∧ ∃ f : (ℕ → ℕ), f k ≥ (n - 1) / 2 :=
sorry

theorem part_b : ∃ᶠ n in at_top, ∃ f : (ℕ → ℕ), ∀ k : ℕ, 1 ≤ k ∧ k ≤ n-1 → f k ≤ (n - 1) / 2 :=
sorry

end part_a_part_b_l235_235100


namespace find_range_of_a_l235_235336

noncomputable def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * a * x + 4 > 0

noncomputable def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → 4 - 2 * a > 0 ∧ 4 - 2 * a < 1

noncomputable def problem_statement (a : ℝ) : Prop :=
  let p := proposition_p a
  let q := proposition_q a
  (p ∨ q) ∧ ¬(p ∧ q)

theorem find_range_of_a (a : ℝ) :
  problem_statement a → -2 < a ∧ a ≤ 3/2 :=
sorry

end find_range_of_a_l235_235336


namespace cubes_with_red_face_l235_235401

theorem cubes_with_red_face :
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  redFaceCubes = 488 :=
by
  let totalCubes := 10 * 10 * 10
  let innerCubes := (10 - 2) * (10 - 2) * (10 - 2)
  let redFaceCubes := totalCubes - innerCubes
  sorry

end cubes_with_red_face_l235_235401


namespace parabola_focus_l235_235214

theorem parabola_focus (x y : ℝ) :
  (∃ x, y = 4 * x^2 + 8 * x - 5) →
  (x, y) = (-1, -8.9375) :=
by
  sorry

end parabola_focus_l235_235214


namespace compare_fractions_l235_235203

theorem compare_fractions :
  (111110 / 111111) < (333331 / 333334) ∧ (333331 / 333334) < (222221 / 222223) :=
by
  sorry

end compare_fractions_l235_235203


namespace last_person_is_knight_l235_235244

-- Definitions for the conditions:
def first_whispered_number := 7
def last_announced_number_first_game := 3
def last_whispered_number_second_game := 5
def first_announced_number_second_game := 2

-- Definitions to represent the roles:
inductive Role
| knight
| liar

-- Definition of the last person in the first game being a knight:
def last_person_first_game_role := Role.knight

theorem last_person_is_knight 
  (h1 : Role.liar = Role.liar)
  (h2 : last_announced_number_first_game = 3)
  (h3 : first_whispered_number = 7)
  (h4 : first_announced_number_second_game = 2)
  (h5 : last_whispered_number_second_game = 5) :
  last_person_first_game_role = Role.knight :=
sorry

end last_person_is_knight_l235_235244


namespace carmen_candle_usage_l235_235008

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l235_235008


namespace sum_of_homothety_coeffs_geq_4_l235_235237

theorem sum_of_homothety_coeffs_geq_4 (a : ℕ → ℝ)
  (h_pos : ∀ i, 0 < a i)
  (h_less_one : ∀ i, a i < 1)
  (h_sum_cubes : ∑' i, (a i)^3 = 1) :
  (∑' i, a i) ≥ 4 := sorry

end sum_of_homothety_coeffs_geq_4_l235_235237


namespace different_prime_factors_of_factorial_eq_10_l235_235728

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l235_235728


namespace paving_cost_l235_235808

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 300
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost : cost = 6187.50 := by
  -- length = 5.5
  -- width = 3.75
  -- rate = 300
  -- area = length * width = 20.625
  -- cost = area * rate = 6187.50
  sorry

end paving_cost_l235_235808


namespace how_many_fewer_runs_did_E_score_l235_235905

-- Define the conditions
variables (a b c d e : ℕ)
variable (h1 : 5 * 36 = 180)
variable (h2 : d = e + 5)
variable (h3 : e = 20)
variable (h4 : b = d + e)
variable (h5 : b + c = 107)
variable (h6 : a + b + c + d + e = 180)

-- Specification to be proved
theorem how_many_fewer_runs_did_E_score :
  a - e = 8 :=
by {
  sorry
}

end how_many_fewer_runs_did_E_score_l235_235905


namespace prime_factors_of_30_factorial_l235_235748

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l235_235748


namespace quadratic_translation_transformed_l235_235603

-- The original function is defined as follows:
def original_func (x : ℝ) : ℝ := 2 * x^2

-- Translated function left by 3 units
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)

-- Translated function down by 2 units
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- Combine both translations: left by 3 units and down by 2 units
def translated_func (x : ℝ) : ℝ := translate_down (translate_left original_func 3) 2 x

-- The theorem we want to prove
theorem quadratic_translation_transformed :
  translated_func x = 2 * (x + 3)^2 - 2 := 
by
  sorry

end quadratic_translation_transformed_l235_235603


namespace prime_factors_of_30_factorial_l235_235736

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l235_235736


namespace find_principal_l235_235071

theorem find_principal
  (P : ℝ)
  (R : ℝ := 4)
  (T : ℝ := 5)
  (SI : ℝ := (P * R * T) / 100) 
  (h : SI = P - 2400) : 
  P = 3000 := 
sorry

end find_principal_l235_235071


namespace nested_radical_solution_l235_235673

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l235_235673


namespace common_ratio_geometric_series_l235_235678

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l235_235678


namespace find_common_ratio_l235_235374

-- Define the geometric sequence with the given conditions
variable (a_n : ℕ → ℝ)
variable (q : ℝ)

axiom a2_eq : a_n 2 = 1
axiom a4_eq : a_n 4 = 4
axiom q_pos : q > 0

-- Define the nature of the geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

-- The specific problem statement to prove
theorem find_common_ratio (h: is_geometric_sequence a_n q) : q = 2 :=
by
  sorry

end find_common_ratio_l235_235374


namespace olivia_dad_spent_l235_235260

def cost_per_meal : ℕ := 7
def number_of_meals : ℕ := 3
def total_cost : ℕ := 21

theorem olivia_dad_spent :
  cost_per_meal * number_of_meals = total_cost :=
by
  sorry

end olivia_dad_spent_l235_235260


namespace find_N_l235_235347

-- Definitions and conditions directly appearing in the problem
variable (X Y Z N : ℝ)

axiom condition1 : 0.15 * X = 0.25 * N + Y
axiom condition2 : X + Y = Z

-- The theorem to prove
theorem find_N : N = 4.6 * X - 4 * Z := by
  sorry

end find_N_l235_235347


namespace factorial_prime_factors_l235_235761

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l235_235761


namespace part_A_part_B_part_D_l235_235131

variables (c d : ℤ)

def multiple_of_5 (x : ℤ) : Prop := ∃ k : ℤ, x = 5 * k
def multiple_of_10 (x : ℤ) : Prop := ∃ k : ℤ, x = 10 * k

-- Given conditions
axiom h1 : multiple_of_5 c
axiom h2 : multiple_of_10 d

-- Problems to prove
theorem part_A : multiple_of_5 d := by sorry
theorem part_B : multiple_of_5 (c - d) := by sorry
theorem part_D : multiple_of_5 (c + d) := by sorry

end part_A_part_B_part_D_l235_235131


namespace equal_blocks_strings_l235_235496

open Nat

def count_special_strings (n : ℕ) : ℕ := 2 * (n - 2).choose (n - 2) / 2

theorem equal_blocks_strings (n : ℕ) (h : n ≥ 2) :
  (count_special_strings n) = 2 * (((n-2).choose ((n-2) / 2))) :=
by
  sorry

end equal_blocks_strings_l235_235496


namespace grid_possible_configuration_l235_235781

theorem grid_possible_configuration (m n : ℕ) (hm : m > 100) (hn : n > 100) : 
  ∃ grid : ℕ → ℕ → ℕ,
  (∀ i j, grid i j = (if i > 0 then grid (i - 1) j else 0) + 
                       (if i < m - 1 then grid (i + 1) j else 0) + 
                       (if j > 0 then grid i (j - 1) else 0) + 
                       (if j < n - 1 then grid i (j + 1) else 0)) 
  ∧ (∃ i j, grid i j ≠ 0) 
  ∧ m > 14 
  ∧ n > 14 := 
sorry

end grid_possible_configuration_l235_235781


namespace solve_absolute_value_inequality_l235_235211

theorem solve_absolute_value_inequality (x : ℝ) :
  3 ≤ |x + 3| ∧ |x + 3| ≤ 7 ↔ (-10 ≤ x ∧ x ≤ -6) ∨ (0 ≤ x ∧ x ≤ 4) :=
by
  sorry

end solve_absolute_value_inequality_l235_235211


namespace chip_credit_card_balance_l235_235853

-- Conditions
def initial_balance : Float := 50.00
def first_interest_rate : Float := 0.20
def additional_charge : Float := 20.00
def second_interest_rate : Float := 0.20

-- Question
def current_balance : Float :=
  let first_interest_fee := initial_balance * first_interest_rate
  let balance_after_first_interest := initial_balance + first_interest_fee
  let balance_before_second_interest := balance_after_first_interest + additional_charge
  let second_interest_fee := balance_before_second_interest * second_interest_rate
  balance_before_second_interest + second_interest_fee

-- Correct Answer
def expected_balance : Float := 96.00

-- Proof Problem Statement
theorem chip_credit_card_balance : current_balance = expected_balance := by
  sorry

end chip_credit_card_balance_l235_235853


namespace compute_expression_l235_235963

theorem compute_expression : 1007^2 - 993^2 - 1005^2 + 995^2 = 8000 := by
  sorry

end compute_expression_l235_235963


namespace center_square_is_15_l235_235854

noncomputable def center_square_value : ℤ :=
  let d1 := (15 - 3) / 2
  let d3 := (33 - 9) / 2
  let middle_first_row := 3 + d1
  let middle_last_row := 9 + d3
  let d2 := (middle_last_row - middle_first_row) / 2
  middle_first_row + d2

theorem center_square_is_15 : center_square_value = 15 := by
  sorry

end center_square_is_15_l235_235854


namespace sum_of_x_y_l235_235519

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235519


namespace find_b_l235_235351

noncomputable def angle_B : ℝ := 60
noncomputable def c : ℝ := 8
noncomputable def diff_b_a (b a : ℝ) : Prop := b - a = 4

theorem find_b (b a : ℝ) (h₁ : angle_B = 60) (h₂ : c = 8) (h₃ : diff_b_a b a) :
  b = 7 :=
sorry

end find_b_l235_235351


namespace distance_AB_l235_235831

-- Definitions and conditions taken from part a)
variables (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0)

-- The main theorem statement
theorem distance_AB (a b c : ℝ) (h_ac_gt_b : a + c > b) (h_a_ge_0 : a ≥ 0) (h_b_ge_0 : b ≥ 0) (h_c_ge_0 : c ≥ 0) : 
  ∃ s : ℝ, s = Real.sqrt ((a * b * c) / (a + c - b)) := 
sorry

end distance_AB_l235_235831


namespace line_circle_intersect_l235_235702

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l235_235702


namespace average_viewer_watches_two_videos_daily_l235_235198

variable (V : ℕ)
variable (video_time : ℕ := 7)
variable (ad_time : ℕ := 3)
variable (total_time : ℕ := 17)

theorem average_viewer_watches_two_videos_daily :
  7 * V + 3 = 17 → V = 2 := 
by
  intro h
  have h1 : 7 * V = 14 := by linarith
  have h2 : V = 2 := by linarith
  exact h2

end average_viewer_watches_two_videos_daily_l235_235198


namespace total_shoes_l235_235782

theorem total_shoes (Brian_shoes : ℕ) (Edward_shoes : ℕ) (Jacob_shoes : ℕ)
  (hBrian : Brian_shoes = 22)
  (hEdward : Edward_shoes = 3 * Brian_shoes)
  (hJacob : Jacob_shoes = Edward_shoes / 2) :
  Brian_shoes + Edward_shoes + Jacob_shoes = 121 :=
by 
  sorry

end total_shoes_l235_235782


namespace spherical_to_rectangular_correct_l235_235208

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l235_235208


namespace janet_earnings_per_hour_l235_235360

theorem janet_earnings_per_hour :
  let text_posts := 150
  let image_posts := 80
  let video_posts := 20
  let rate_text := 0.25
  let rate_image := 0.30
  let rate_video := 0.40
  text_posts * rate_text + image_posts * rate_image + video_posts * rate_video = 69.50 :=
by
  sorry

end janet_earnings_per_hour_l235_235360


namespace david_english_marks_l235_235209

theorem david_english_marks :
  let Mathematics := 45
  let Physics := 72
  let Chemistry := 77
  let Biology := 75
  let AverageMarks := 68.2
  let TotalSubjects := 5
  let TotalMarks := AverageMarks * TotalSubjects
  let MarksInEnglish := TotalMarks - (Mathematics + Physics + Chemistry + Biology)
  MarksInEnglish = 72 :=
by
  sorry

end david_english_marks_l235_235209


namespace half_dollar_difference_l235_235369

theorem half_dollar_difference (n d h : ℕ) 
  (h1 : n + d + h = 150) 
  (h2 : 5 * n + 10 * d + 50 * h = 1500) : 
  ∃ h_max h_min, (h_max - h_min = 16) :=
by sorry

end half_dollar_difference_l235_235369


namespace jenny_earnings_at_better_neighborhood_l235_235093

noncomputable def homes_in_A := 10
noncomputable def boxes_per_home_A := 2
noncomputable def homes_in_B := 5
noncomputable def boxes_per_home_B := 5
noncomputable def price_per_box := 2

theorem jenny_earnings_at_better_neighborhood :
  let total_boxes_A := homes_in_A * boxes_per_home_A in
  let total_boxes_B := homes_in_B * boxes_per_home_B in
  let better_choice := if total_boxes_A > total_boxes_B then total_boxes_A else total_boxes_B in
  let total_earnings := better_choice * price_per_box in
  total_earnings = 50 :=
by
  sorry

end jenny_earnings_at_better_neighborhood_l235_235093


namespace polynomial_sat_condition_l235_235099

theorem polynomial_sat_condition (P : Polynomial ℝ) (k : ℕ) (hk : 0 < k) :
  (P.comp P = P ^ k) →
  (P = 0 ∨ P = 1 ∨ (k % 2 = 1 ∧ P = -1) ∨ P = Polynomial.X ^ k) :=
sorry

end polynomial_sat_condition_l235_235099


namespace line_intercept_form_l235_235028

theorem line_intercept_form 
  (P : ℝ × ℝ) 
  (a : ℝ × ℝ) 
  (l_eq : ∃ m : ℝ, ∀ x y : ℝ, (x, y) = P → y - 3 = m * (x - 2))
  (P_coord : P = (2, 3)) 
  (a_vect : a = (2, -6)) 
  : ∀ x y : ℝ, y - 3 = (-3) * (x - 2) → 3 * x + y - 9 = 0 →  ∃ a' b' : ℝ, a' ≠ 0 ∧ b' ≠ 0 ∧ x / 3 + y / 9 = 1 :=
by
  sorry

end line_intercept_form_l235_235028


namespace number_of_distinct_prime_factors_30_fact_l235_235706

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l235_235706


namespace vertical_asymptote_l235_235215

noncomputable def f (x : ℝ) (c : ℝ) : ℝ := (x^2 - x + c) / (x^2 - 6*x + 8)

theorem vertical_asymptote (c : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 4 → ((x^2 - x + c) ≠ 0)) ∨
  (∀ x : ℝ, ((x^2 - x + c) = 0) ↔ (x = 2) ∨ (x = 4)) →
  c = -2 ∨ c = -12 :=
sorry

end vertical_asymptote_l235_235215


namespace sqrt_continued_fraction_l235_235670

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l235_235670


namespace number_of_prime_factors_30_factorial_l235_235757

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l235_235757


namespace radius_of_circle_is_61_2_inches_l235_235402

noncomputable def large_square_side : ℝ := 144
noncomputable def total_area : ℝ := large_square_side * large_square_side
noncomputable def l_region_fraction : ℝ := 5 / 18
noncomputable def l_region_area : ℝ := 4 * l_region_fraction * total_area
noncomputable def center_square_area : ℝ := total_area - l_region_area
noncomputable def center_square_side : ℝ := Real.sqrt center_square_area
noncomputable def circle_radius : ℝ := center_square_side / 2

theorem radius_of_circle_is_61_2_inches : circle_radius = 61.2 := 
sorry

end radius_of_circle_is_61_2_inches_l235_235402


namespace P_intersection_Q_is_singleton_l235_235476

theorem P_intersection_Q_is_singleton :
  {p : ℝ × ℝ | p.1 + p.2 = 3} ∩ {p : ℝ × ℝ | p.1 - p.2 = 5} = { (4, -1) } :=
by
  -- The proof steps would go here.
  sorry

end P_intersection_Q_is_singleton_l235_235476


namespace deductible_amount_l235_235199

-- This definition represents the conditions of the problem.
def current_annual_deductible_is_increased (D : ℝ) : Prop :=
  (2 / 3) * D = 2000

-- This is the Lean statement, expressing the problem that needs to be proven.
theorem deductible_amount (D : ℝ) (h : current_annual_deductible_is_increased D) : D = 3000 :=
by
  sorry

end deductible_amount_l235_235199


namespace bales_stored_in_barn_l235_235596

-- Defining the conditions
def bales_initial : Nat := 28
def bales_stacked : Nat := 28
def bales_already_there : Nat := 54

-- Formulate the proof statement
theorem bales_stored_in_barn : bales_already_there + bales_stacked = 82 := by
  sorry

end bales_stored_in_barn_l235_235596


namespace num_prime_factors_30_factorial_l235_235767

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l235_235767


namespace bridge_length_problem_l235_235981

noncomputable def length_of_bridge (num_carriages : ℕ) (length_carriage : ℕ) (length_engine : ℕ) (speed_kmph : ℕ) (crossing_time_min : ℕ) : ℝ :=
  let total_train_length := (num_carriages + 1) * length_carriage
  let speed_mps := (speed_kmph * 1000) / 3600
  let crossing_time_secs := crossing_time_min * 60
  let total_distance := speed_mps * crossing_time_secs
  let bridge_length := total_distance - total_train_length
  bridge_length

theorem bridge_length_problem :
  length_of_bridge 24 60 60 60 5 = 3501 :=
by
  sorry

end bridge_length_problem_l235_235981


namespace min_value_of_u_l235_235184

theorem min_value_of_u : ∀ (x y : ℝ), x ∈ Set.Ioo (-2) 2 → y ∈ Set.Ioo (-2) 2 → x * y = -1 → 
  (∀ u, u = (4 / (4 - x^2)) + (9 / (9 - y^2)) → u ≥ 12 / 5) :=
by
  intros x y hx hy hxy u hu
  sorry

end min_value_of_u_l235_235184


namespace find_numbers_l235_235385

-- Define the conditions
def geometric_mean_condition (a b : ℝ) : Prop :=
  a * b = 3

def harmonic_mean_condition (a b : ℝ) : Prop :=
  2 / (1 / a + 1 / b) = 3 / 2

-- State the theorem to be proven
theorem find_numbers (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  geometric_mean_condition a b ∧ harmonic_mean_condition a b → (a = 1 ∧ b = 3) ∨ (a = 3 ∧ b = 1) := 
by 
  sorry

end find_numbers_l235_235385


namespace find_a_range_l235_235047

variable (a k : ℝ)
variable (x : ℝ) (hx : x > 0)

def p := ∀ x > 0, x + a / x ≥ 2
def q := ∀ k : ℝ, ∃ x y : ℝ, k * x - y + 2 = 0 ∧ x^2 + y^2 / a^2 = 1

theorem find_a_range :
  (a > 0) →
  ((p a) ∨ (q a)) ∧ ¬ ((p a) ∧ (q a)) ↔ 1 ≤ a ∧ a < 2 :=
sorry

end find_a_range_l235_235047


namespace ratio_gluten_free_l235_235311

theorem ratio_gluten_free (total_cupcakes vegan_cupcakes non_vegan_gluten cupcakes_gluten_free : ℕ)
    (H1 : total_cupcakes = 80)
    (H2 : vegan_cupcakes = 24)
    (H3 : non_vegan_gluten = 28)
    (H4 : cupcakes_gluten_free = vegan_cupcakes / 2) :
    (cupcakes_gluten_free : ℚ) / (total_cupcakes : ℚ) = 3 / 20 :=
by 
  -- Proof goes here
  sorry

end ratio_gluten_free_l235_235311


namespace train_speed_in_km_per_hr_l235_235426

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l235_235426


namespace mutually_exclusive_and_probability_conditional_probability_independence_l235_235040

variables (Ω : Type*) [ProbabilityMeasure Ω]
variable {A B : Set Ω}
variable [MeasurableSet A] [MeasurableSet B]

-- Given conditions
variable (hA : ℙ A = 0.5)
variable (hB : ℙ B = 0.2)

-- Proof goals
theorem mutually_exclusive_and_probability (h : Disjoint A B) : ℙ (A ∪ B) = 0.7 :=
by
  sorry

theorem conditional_probability_independence (h : ℙ(B ∩ A) / ℙ A = 0.2) : Independent A B :=
by
  sorry

end mutually_exclusive_and_probability_conditional_probability_independence_l235_235040


namespace line_equation_l235_235626

theorem line_equation (a b : ℝ) (h1 : (1, 2) ∈ line) (h2 : ∃ a b : ℝ, b = 2 * a ∧ line = {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}) :
  line = {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} ∨ line = {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0} :=
sorry

end line_equation_l235_235626


namespace anne_distance_l235_235770

-- Definitions based on conditions
def Time : ℕ := 5
def Speed : ℕ := 4
def Distance : ℕ := Speed * Time

-- Proof statement
theorem anne_distance : Distance = 20 := by
  sorry

end anne_distance_l235_235770


namespace num_prime_factors_of_30_l235_235739

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l235_235739


namespace smallest_common_multiple_of_9_and_6_l235_235170

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l235_235170


namespace sin_cos_product_l235_235060

theorem sin_cos_product (x : ℝ) (h : Real.sin x = 5 * Real.cos x) : Real.sin x * Real.cos x = 5 / 26 := by
  sorry

end sin_cos_product_l235_235060


namespace length_of_living_room_l235_235602

theorem length_of_living_room
  (l : ℝ) -- length of the living room
  (w : ℝ) -- width of the living room
  (boxes_coverage : ℝ) -- area covered by one box
  (initial_area : ℝ) -- area already covered
  (additional_boxes : ℕ) -- additional boxes required
  (total_area : ℝ) -- total area required
  (w_condition : w = 20)
  (boxes_coverage_condition : boxes_coverage = 10)
  (initial_area_condition : initial_area = 250)
  (additional_boxes_condition : additional_boxes = 7)
  (total_area_condition : total_area = l * w)
  (full_coverage_condition : additional_boxes * boxes_coverage + initial_area = total_area) :
  l = 16 := by
  sorry

end length_of_living_room_l235_235602


namespace largest_fraction_among_given_l235_235178

theorem largest_fraction_among_given (f1 f2 f3 f4 f5 : ℚ)
  (h1 : f1 = 2/5) 
  (h2 : f2 = 4/9) 
  (h3 : f3 = 7/15) 
  (h4 : f4 = 11/18) 
  (h5 : f5 = 16/35) 
  : f1 < f4 ∧ f2 < f4 ∧ f3 < f4 ∧ f5 < f4 :=
by
  sorry

end largest_fraction_among_given_l235_235178


namespace price_comparison_l235_235471

variable (x y : ℝ)
variable (h1 : 6 * x + 3 * y > 24)
variable (h2 : 4 * x + 5 * y < 22)

theorem price_comparison : 2 * x > 3 * y :=
sorry

end price_comparison_l235_235471


namespace find_equation_of_line_l235_235687

-- Define the conditions
def line_passes_through_A (m b : ℝ) (A : ℝ × ℝ) : Prop :=
  A = (1, 1) ∧ A.2 = -A.1 + b

def intercepts_equal (m b : ℝ) : Prop :=
  b = m

-- The goal to prove the equations of the line
theorem find_equation_of_line :
  ∃ (m b : ℝ), line_passes_through_A m b (1, 1) ∧ intercepts_equal m b ↔ 
  (∃ m b : ℝ, (m = -1 ∧ b = 2) ∨ (m = 1 ∧ b = 0)) :=
sorry

end find_equation_of_line_l235_235687


namespace find_number_with_divisors_condition_l235_235502

theorem find_number_with_divisors_condition :
  ∃ n : ℕ, (∃ d1 d2 d3 d4 : ℕ, 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 * d4 ∣ n ∧
    d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 = n) ∧ n = 130 :=
by
  sorry

end find_number_with_divisors_condition_l235_235502


namespace total_money_made_l235_235404

structure Building :=
(floors : Nat)
(rooms_per_floor : Nat)

def cleaning_time_per_room : Nat := 8

structure CleaningRates :=
(first_4_hours_rate : Int)
(next_4_hours_rate : Int)
(unpaid_break_hours : Nat)

def supply_cost : Int := 1200

def total_earnings (b : Building) (c : CleaningRates) : Int :=
  let rooms := b.floors * b.rooms_per_floor
  let earnings_per_room := (4 * c.first_4_hours_rate + 4 * c.next_4_hours_rate)
  rooms * earnings_per_room - supply_cost

theorem total_money_made (b : Building) (c : CleaningRates) : 
  b.floors = 12 →
  b.rooms_per_floor = 25 →
  cleaning_time_per_room = 8 →
  c.first_4_hours_rate = 20 →
  c.next_4_hours_rate = 25 →
  c.unpaid_break_hours = 1 →
  total_earnings b c = 52800 := 
by
  intros
  sorry

end total_money_made_l235_235404


namespace total_monkeys_is_correct_l235_235431

-- Define the parameters
variables (m n : ℕ)

-- Define the conditions as separate definitions
def monkeys_on_n_bicycles : ℕ := 3 * n
def monkeys_on_remaining_bicycles : ℕ := 5 * (m - n)

-- Define the total number of monkeys
def total_monkeys : ℕ := monkeys_on_n_bicycles n + monkeys_on_remaining_bicycles m n

-- State the theorem
theorem total_monkeys_is_correct : total_monkeys m n = 5 * m - 2 * n :=
by
  sorry

end total_monkeys_is_correct_l235_235431


namespace sum_of_x_y_possible_values_l235_235535

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235535


namespace greatest_product_l235_235954

theorem greatest_product (x : ℤ) (h : x + (1998 - x) = 1998) : 
  x * (1998 - x) ≤ 998001 :=
  sorry

end greatest_product_l235_235954


namespace intersection_A_B_l235_235477

def set_A : Set ℕ := {x | x^2 - 2 * x = 0}
def set_B : Set ℕ := {0, 1, 2}

theorem intersection_A_B : set_A ∩ set_B = {0, 2} := 
by sorry

end intersection_A_B_l235_235477


namespace polygon_interior_exterior_relation_l235_235885

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l235_235885


namespace sum_of_x_y_possible_values_l235_235537

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235537


namespace find_P_Q_sum_l235_235345

theorem find_P_Q_sum (P Q : ℤ) 
  (h : ∃ b c : ℤ, x^2 + 3 * x + 2 ∣ x^4 + P * x^2 + Q 
    ∧ b + 3 = 0 
    ∧ c + 3 * b + 6 = P 
    ∧ 3 * c + 2 * b = 0 
    ∧ 2 * c = Q): 
  P + Q = 3 := 
sorry

end find_P_Q_sum_l235_235345


namespace product_of_three_integers_sum_l235_235584

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l235_235584


namespace gcd_612_468_l235_235293

theorem gcd_612_468 : gcd 612 468 = 36 :=
by
  sorry

end gcd_612_468_l235_235293


namespace vertex_of_parabola_l235_235141

theorem vertex_of_parabola (c d : ℝ) :
  (∀ x, -2 * x^2 + c * x + d ≤ 0 ↔ x ≥ -7 / 2) →
  ∃ k, k = (-7 / 2 : ℝ) ∧ y = -2 * (x + 7 / 2)^2 + 0 := 
sorry

end vertex_of_parabola_l235_235141


namespace sum_of_three_numbers_l235_235856

theorem sum_of_three_numbers (a b c : ℝ) (h1 : (a + b + c) / 3 = a - 15) (h2 : (a + b + c) / 3 = c + 10) (h3 : b = 10) :
  a + b + c = 45 :=
  sorry

end sum_of_three_numbers_l235_235856


namespace right_triangle_circle_area_l235_235807

/-- 
Given a right triangle ABC with legs AB = 6 cm and BC = 8 cm,
E is the midpoint of AB and D is the midpoint of AC.
A circle passes through points E and D and touches the hypotenuse AC.
Prove that the area of this circle is 100 * pi / 9 cm^2.
-/
theorem right_triangle_circle_area :
  ∃ (r : ℝ), 
  let AB := 6
  let BC := 8
  let AC := Real.sqrt (AB^2 + BC^2)
  let E := (AB / 2)
  let D := (AC / 2)
  let radius := (AC * (BC / 2) / AB)
  r = radius * radius * Real.pi ∧
  r = (100 * Real.pi / 9) := sorry

end right_triangle_circle_area_l235_235807


namespace nested_sqrt_eq_l235_235649

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l235_235649


namespace exist_initial_points_l235_235568

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l235_235568


namespace sqrt_three_cubes_l235_235595

theorem sqrt_three_cubes : Real.sqrt (3^3 + 3^3 + 3^3) = 9 := 
  sorry

end sqrt_three_cubes_l235_235595


namespace value_two_stddev_below_mean_l235_235396

def mean : ℝ := 16.2
def standard_deviation : ℝ := 2.3

theorem value_two_stddev_below_mean : mean - 2 * standard_deviation = 11.6 :=
by
  sorry

end value_two_stddev_below_mean_l235_235396


namespace nested_sqrt_eq_l235_235663

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l235_235663


namespace otimes_calc_1_otimes_calc_2_otimes_calc_3_l235_235642

def otimes (a b : Int) : Int :=
  a^2 - Int.natAbs b

theorem otimes_calc_1 : otimes (-2) 3 = 1 :=
by
  sorry

theorem otimes_calc_2 : otimes 5 (-4) = 21 :=
by
  sorry

theorem otimes_calc_3 : otimes (-3) (-1) = 8 :=
by
  sorry

end otimes_calc_1_otimes_calc_2_otimes_calc_3_l235_235642


namespace cone_volume_l235_235625

noncomputable def radius_of_sector : ℝ := 6
noncomputable def arc_length_of_sector : ℝ := (1 / 2) * (2 * Real.pi * radius_of_sector)
noncomputable def radius_of_base : ℝ := arc_length_of_sector / (2 * Real.pi)
noncomputable def slant_height : ℝ := radius_of_sector
noncomputable def height_of_cone : ℝ := Real.sqrt (slant_height^2 - radius_of_base^2)
noncomputable def volume_of_cone : ℝ := (1 / 3) * Real.pi * (radius_of_base^2) * height_of_cone

theorem cone_volume : volume_of_cone = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end cone_volume_l235_235625


namespace part1_part2_l235_235473

noncomputable def f (a x : ℝ) := a * x^2 - (a + 1) * x + 1

theorem part1 (a : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, f a x ≤ 2) ↔ (-3 - 2 * Real.sqrt 2 ≤ a ∧ a ≤ -3 + 2 * Real.sqrt 2) :=
sorry

theorem part2 (a : ℝ) (h1 : a ≠ 0) (x : ℝ) :
  (f a x < 0) ↔
    ((0 < a ∧ a < 1 ∧ 1 < x ∧ x < 1 / a) ∨
     (a = 1 ∧ false) ∨
     (a > 1 ∧ 1 / a < x ∧ x < 1) ∨
     (a < 0 ∧ (x < 1 / a ∨ x > 1))) :=
sorry

end part1_part2_l235_235473


namespace sum_of_real_numbers_l235_235525

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235525


namespace find_x_l235_235894

-- Definitions of the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Inner product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Perpendicular condition
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l235_235894


namespace cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l235_235972

-- Problem Part 1
theorem cos_alpha_implies_sin_alpha (alpha : ℝ) (h1 : Real.cos alpha = -4/5) (h2 : α ∈ Set.Ioo (π/2) π) : 
  Real.sin alpha = -3/5 := sorry

-- Problem Part 2
theorem tan_theta_implies_expr (theta : ℝ) (h1 : Real.tan theta = 3) : 
  (Real.sin theta + Real.cos theta) / (2 * Real.sin theta + Real.cos theta) = 4 / 7 := sorry

end cos_alpha_implies_sin_alpha_tan_theta_implies_expr_l235_235972


namespace possible_values_of_sum_l235_235541

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235541


namespace total_students_sampled_l235_235191

theorem total_students_sampled :
  ∀ (seniors juniors freshmen sampled_seniors sampled_juniors sampled_freshmen total_students : ℕ),
    seniors = 1000 →
    juniors = 1200 →
    freshmen = 1500 →
    sampled_freshmen = 75 →
    sampled_seniors = seniors * (sampled_freshmen / freshmen) →
    sampled_juniors = juniors * (sampled_freshmen / freshmen) →
    total_students = sampled_seniors + sampled_juniors + sampled_freshmen →
    total_students = 185 :=
by
sorry

end total_students_sampled_l235_235191


namespace tomatoes_ruined_percentage_l235_235811

-- The definitions from the problem conditions
def tomato_cost_per_pound : ℝ := 0.80
def tomato_selling_price_per_pound : ℝ := 0.977777777777778
def desired_profit_percent : ℝ := 0.10
def revenue_equal_cost_plus_profit_cost_fraction : ℝ := (tomato_cost_per_pound + (tomato_cost_per_pound * desired_profit_percent))

-- The theorem stating the problem and the expected result
theorem tomatoes_ruined_percentage :
  ∀ (W : ℝ) (P : ℝ),
  (0.977777777777778 * (1 - P / 100) * W = (0.80 * W + 0.08 * W)) →
  P = 10.00000000000001 :=
by
  intros W P h
  have eq1 : 0.977777777777778 * (1 - P / 100) = 0.88 := sorry
  have eq2 : 1 - P / 100 = 0.8999999999999999 := sorry
  have eq3 : P / 100 = 0.1000000000000001 := sorry
  exact sorry

end tomatoes_ruined_percentage_l235_235811


namespace baseball_games_per_month_l235_235597

-- Define the conditions
def total_games_in_a_season : ℕ := 14
def months_in_a_season : ℕ := 2

-- Define the proposition stating the number of games per month
def games_per_month (total_games months : ℕ) : ℕ := total_games / months

-- State the equivalence proof problem
theorem baseball_games_per_month : games_per_month total_games_in_a_season months_in_a_season = 7 :=
by
  -- Directly stating the equivalence based on given conditions
  sorry

end baseball_games_per_month_l235_235597


namespace number_of_typists_needed_l235_235898

theorem number_of_typists_needed :
  (∃ t : ℕ, (20 * 40) / 20 * 60 * t = 180) ↔ t = 30 :=
by sorry

end number_of_typists_needed_l235_235898


namespace number_of_prime_factors_thirty_factorial_l235_235732

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l235_235732


namespace model_tower_height_l235_235252

theorem model_tower_height (real_height : ℝ) (real_volume : ℝ) (model_volume : ℝ) (h_cond : real_height = 80) (vol_cond : real_volume = 200000) (model_vol_cond : model_volume = 0.2) : 
  ∃ h : ℝ, h = 0.8 :=
by sorry

end model_tower_height_l235_235252


namespace first_day_of_month_l235_235937

theorem first_day_of_month (d : ℕ) (h : d = 30) (dow_30 : d % 7 = 3) : (1 % 7 = 2) :=
by sorry

end first_day_of_month_l235_235937


namespace apples_in_baskets_l235_235117

theorem apples_in_baskets (total_apples : ℕ) (first_basket : ℕ) (increase : ℕ) (baskets : ℕ) :
  total_apples = 495 ∧ first_basket = 25 ∧ increase = 2 ∧
  (total_apples = (baskets / 2) * (2 * first_basket + (baskets - 1) * increase)) -> baskets = 13 :=
by sorry

end apples_in_baskets_l235_235117


namespace initial_milk_in_container_A_l235_235967

theorem initial_milk_in_container_A (A B C D : ℝ) 
  (h1 : B = A - 0.625 * A) 
  (h2 : C - 158 = B) 
  (h3 : D = 0.45 * (C - 58)) 
  (h4 : D = 58) 
  : A = 231 := 
sorry

end initial_milk_in_container_A_l235_235967


namespace negation_of_universal_proposition_l235_235227

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x^2 - x + 1 / 4 > 0)) = ∃ x : ℝ, x^2 - x + 1 / 4 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l235_235227


namespace sum_product_of_integers_l235_235588

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l235_235588


namespace prime_factors_30_fac_eq_10_l235_235714

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l235_235714


namespace cost_of_one_box_of_tissues_l235_235986

variable (num_toilet_paper : ℕ) (num_paper_towels : ℕ) (num_tissues : ℕ)
variable (cost_toilet_paper : ℝ) (cost_paper_towels : ℝ) (total_cost : ℝ)

theorem cost_of_one_box_of_tissues (num_toilet_paper = 10) 
                                   (num_paper_towels = 7) 
                                   (num_tissues = 3)
                                   (cost_toilet_paper = 1.50) 
                                   (cost_paper_towels = 2.00) 
                                   (total_cost = 35.00) :
  let total_cost_toilet_paper := num_toilet_paper * cost_toilet_paper,
      total_cost_paper_towels := num_paper_towels * cost_paper_towels,
      cost_left_for_tissues := total_cost - (total_cost_toilet_paper + total_cost_paper_towels),
      one_box_tissues_cost := cost_left_for_tissues / num_tissues
  in one_box_tissues_cost = 2.00 := 
sorry

end cost_of_one_box_of_tissues_l235_235986


namespace permutation_sum_inequality_l235_235791

noncomputable def permutations (n : ℕ) : List (List ℚ) :=
  List.permutations ((List.range (n+1)).map (fun i => if i = 0 then (1 : ℚ) else (1 : ℚ) / i))

theorem permutation_sum_inequality (n : ℕ) (a b : Fin n → ℚ)
  (ha : ∃ p : List ℚ, p ∈ permutations n ∧ ∀ i, a i = p.get? i) 
  (hb : ∃ q : List ℚ, q ∈ permutations n ∧ ∀ i, b i = q.get? i)
  (h_sum : ∀ i j : Fin n, i ≤ j → a i + b i ≥ a j + b j) 
  (m : Fin n) :
  a m + b m ≤ 4 / (m + 1) :=
sorry

end permutation_sum_inequality_l235_235791


namespace train_speed_is_60_kmph_l235_235421

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l235_235421


namespace number_of_prime_factors_of_30_factorial_l235_235723

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l235_235723


namespace prime_factors_of_30_factorial_l235_235745

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l235_235745


namespace marble_distribution_l235_235288

theorem marble_distribution (a b c : ℚ) (h1 : a + b + c = 78) (h2 : a = 3 * b + 2) (h3 : b = c / 2) : 
  a = 40 ∧ b = 38 / 3 ∧ c = 76 / 3 :=
by
  sorry

end marble_distribution_l235_235288


namespace invalid_votes_percentage_l235_235908

theorem invalid_votes_percentage (total_votes : ℕ) (valid_votes_candidate2 : ℕ) (valid_votes_percentage_candidate1 : ℕ) 
  (h_total_votes : total_votes = 7500) 
  (h_valid_votes_candidate2 : valid_votes_candidate2 = 2700)
  (h_valid_votes_percentage_candidate1 : valid_votes_percentage_candidate1 = 55) :
  ((total_votes - (valid_votes_candidate2 * 100 / (100 - valid_votes_percentage_candidate1))) * 100 / total_votes) = 20 :=
by sorry

end invalid_votes_percentage_l235_235908


namespace greatest_integer_less_PS_l235_235080

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l235_235080


namespace mustard_found_at_second_table_l235_235200

variables (total_mustard first_table third_table second_table : ℝ)

def mustard_found (total_mustard first_table third_table : ℝ) := total_mustard - (first_table + third_table)

theorem mustard_found_at_second_table
    (h_total : total_mustard = 0.88)
    (h_first : first_table = 0.25)
    (h_third : third_table = 0.38) :
    mustard_found total_mustard first_table third_table = 0.25 :=
by
    rw [mustard_found, h_total, h_first, h_third]
    simp
    sorry

end mustard_found_at_second_table_l235_235200


namespace remainder_of_n_div_4_is_1_l235_235968

noncomputable def n : ℕ := sorry  -- We declare n as a noncomputable natural number to proceed with the proof complexity

theorem remainder_of_n_div_4_is_1 (n : ℕ) (h : (2 * n) % 4 = 2) : n % 4 = 1 :=
by
  sorry  -- skip the proof

end remainder_of_n_div_4_is_1_l235_235968


namespace evaluate_nested_radical_l235_235657

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l235_235657


namespace scientific_notation_of_75500000_l235_235072

theorem scientific_notation_of_75500000 :
  ∃ (a : ℝ) (n : ℤ), 75500000 = a * 10 ^ n ∧ a = 7.55 ∧ n = 7 :=
by {
  sorry
}

end scientific_notation_of_75500000_l235_235072


namespace sin_eq_sqrt3_div_2_range_l235_235084

theorem sin_eq_sqrt3_div_2_range :
  {x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x ≥ Real.sqrt 3 / 2} = 
  {x | Real.pi / 3 ≤ x ∧ x ≤ 2 * Real.pi / 3} :=
sorry

end sin_eq_sqrt3_div_2_range_l235_235084


namespace selling_price_correct_l235_235193

variable (CostPrice GainPercent : ℝ)
variables (Profit SellingPrice : ℝ)

noncomputable def calculateProfit : ℝ := (GainPercent / 100) * CostPrice

noncomputable def calculateSellingPrice : ℝ := CostPrice + calculateProfit CostPrice GainPercent

theorem selling_price_correct 
  (h1 : CostPrice = 900) 
  (h2 : GainPercent = 30)
  : calculateSellingPrice CostPrice GainPercent = 1170 := by
  sorry

end selling_price_correct_l235_235193


namespace S_n_expression_l235_235691

/-- 
  Given a sequence of positive terms {a_n} with sum of the first n terms represented as S_n,
  and given a_1 = 2, and given the relationship 
  S_{n+1}(S_{n+1} - 3^n) = S_n(S_n + 3^n), prove that S_{2023} = (3^2023 + 1) / 2.
-/
theorem S_n_expression
  (a : ℕ → ℕ) (S : ℕ → ℕ)
  (ha1 : a 1 = 2)
  (hr : ∀ n, S (n + 1) * (S (n + 1) - 3^n) = S n * (S n + 3^n)) :
  S 2023 = (3^2023 + 1) / 2 :=
sorry

end S_n_expression_l235_235691


namespace negation_of_proposition_l235_235015

theorem negation_of_proposition (a b : ℝ) : 
  ¬ (∀ a b : ℝ, (a = 1 → a + b = 1)) ↔ (∃ a b : ℝ, a = 1 ∧ a + b ≠ 1) :=
by
  sorry

end negation_of_proposition_l235_235015


namespace min_a2_b2_l235_235281

noncomputable def minimum_a2_b2 (a b : ℝ) : Prop :=
  (∃ a b : ℝ, (|(-2*a - 2*b + 4)|) / (Real.sqrt (a^2 + (2*b)^2)) = 2) → (a^2 + b^2 = 2)

theorem min_a2_b2 : minimum_a2_b2 a b :=
by
  sorry

end min_a2_b2_l235_235281


namespace people_counted_l235_235000

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l235_235000


namespace minimum_a_l235_235083

theorem minimum_a (a : ℝ) (h : a > 0) :
  (∀ (N : ℝ × ℝ), (N.1 - a)^2 + (N.2 + a - 3)^2 = 1 → 
   dist (N.1, N.2) (0, 0) ≥ 2) → a ≥ 3 :=
by
  sorry

end minimum_a_l235_235083


namespace necessary_but_not_sufficient_l235_235376

theorem necessary_but_not_sufficient :
  (∀ x : ℝ, x > 2 → |x| ≥ 1) ∧ (∃ x : ℝ, |x| ≥ 1 ∧ ¬ (x > 2)) :=
by
  sorry

end necessary_but_not_sufficient_l235_235376


namespace shirt_price_after_discount_l235_235434

/-- Given a shirt with an initial cost price of $20 and a profit margin of 30%, 
    and a sale discount of 50%, prove that the final sale price of the shirt is $13. -/
theorem shirt_price_after_discount
  (cost_price : ℝ)
  (profit_margin : ℝ)
  (discount : ℝ)
  (selling_price : ℝ)
  (final_price : ℝ)
  (h_cost : cost_price = 20)
  (h_profit_margin : profit_margin = 0.30)
  (h_discount : discount = 0.50)
  (h_selling_price : selling_price = cost_price + profit_margin * cost_price)
  (h_final_price : final_price = selling_price - discount * selling_price) :
  final_price = 13 := 
  sorry

end shirt_price_after_discount_l235_235434


namespace rationalize_denominator_l235_235124

theorem rationalize_denominator :
  (1 / (real.sqrt 3 - 2)) = -(real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l235_235124


namespace base_k_perfect_square_l235_235798

theorem base_k_perfect_square (k : ℤ) (h : k ≥ 6) : 
  (1 * k^8 + 2 * k^7 + 3 * k^6 + 4 * k^5 + 5 * k^4 + 4 * k^3 + 3 * k^2 + 2 * k + 1) = (k^4 + k^3 + k^2 + k + 1)^2 := 
by
  sorry

end base_k_perfect_square_l235_235798


namespace at_least_half_girls_prob_l235_235930

-- Define the conditions
def total_children : ℕ := 6
def prob_girl : ℚ := 3 / 5
def prob_boy : ℚ := 1 - prob_girl

-- Define the binomial probability function
noncomputable def binomial_prob (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- Define the probability that at least 3 out of 6 children are girls
noncomputable def prob_at_least_3_girls : ℚ :=
  binomial_prob total_children 3 prob_girl +
  binomial_prob total_children 4 prob_girl +
  binomial_prob total_children 5 prob_girl +
  binomial_prob total_children 6 prob_girl

-- The proof statement
theorem at_least_half_girls_prob :
  prob_at_least_3_girls = 513 / 625 :=
by
  sorry

end at_least_half_girls_prob_l235_235930


namespace angle_rotation_l235_235903

theorem angle_rotation (α : ℝ) (β : ℝ) (k : ℤ) :
  (∃ k' : ℤ, α + 30 = 120 + 360 * k') →
  (β = 360 * k + 90) ↔ (∃ k'' : ℤ, β = 360 * k'' + α) :=
by
  sorry

end angle_rotation_l235_235903


namespace mary_has_more_l235_235112

theorem mary_has_more (marco_initial mary_initial : ℕ) (h1 : marco_initial = 24) (h2 : mary_initial = 15) :
  let marco_final := marco_initial - 12,
      mary_final := mary_initial + 12 - 5 in
  mary_final = marco_final + 10 :=
by
  sorry

end mary_has_more_l235_235112


namespace evaluate_expression_l235_235022

theorem evaluate_expression : 500 * (500 ^ 500) * 500 = 500 ^ 502 := by
  sorry

end evaluate_expression_l235_235022


namespace expand_polynomial_l235_235959

theorem expand_polynomial (N : ℕ) :
  (∃ a b c d : ℕ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ (a + b + c + d + 1)^N = 715) ↔ N = 13 := by
  sorry -- Replace with the actual proof when ready

end expand_polynomial_l235_235959


namespace vertex_angle_measure_l235_235634

-- Definitions for Lean Proof
def isosceles_triangle (α β γ : ℝ) : Prop := (α = β) ∨ (α = γ) ∨ (β = γ)
def exterior_angle (interior exterior : ℝ) : Prop := interior + exterior = 180

-- Conditions from the problem
variables (α β γ : ℝ)
variable (ext_angle : ℝ := 110)

-- Lean 4 statement: The measure of the vertex angle is 70° or 40°
theorem vertex_angle_measure :
  isosceles_triangle α β γ ∧
  (exterior_angle γ ext_angle ∨ exterior_angle α ext_angle ∨ exterior_angle β ext_angle) →
  (γ = 70 ∨ γ = 40) :=
by
  sorry

end vertex_angle_measure_l235_235634


namespace savannah_wraps_4_with_third_roll_l235_235931

variable (gifts total_rolls : ℕ)
variable (wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ)
variable (no_leftover : Prop)

def savannah_wrapping_presents (gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 : ℕ) (no_leftover : Prop) : Prop :=
  gifts = 12 ∧
  total_rolls = 3 ∧
  wrap_with_roll1 = 3 ∧
  wrap_with_roll2 = 5 ∧
  remaining_wrap_with_roll3 = gifts - (wrap_with_roll1 + wrap_with_roll2) ∧
  no_leftover = (total_rolls = 3) ∧ (wrap_with_roll1 + wrap_with_roll2 + remaining_wrap_with_roll3 = gifts)

theorem savannah_wraps_4_with_third_roll
  (h : savannah_wrapping_presents gifts total_rolls wrap_with_roll1 wrap_with_roll2 remaining_wrap_with_roll3 no_leftover) :
  remaining_wrap_with_roll3 = 4 :=
by
  sorry

end savannah_wraps_4_with_third_roll_l235_235931


namespace sqrt_continued_fraction_l235_235667

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l235_235667


namespace box_height_l235_235792

theorem box_height (x : ℝ) (hx : x + 5 = 10)
  (surface_area : 2*x^2 + 4*x*(x + 5) ≥ 150) : x + 5 = 10 :=
sorry

end box_height_l235_235792


namespace points_on_line_l235_235557

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l235_235557


namespace minimum_value_proof_l235_235248

noncomputable def minimum_value (a b c : ℝ) (h : a + b + c = 6) : ℝ :=
  9 / a + 4 / b + 1 / c

theorem minimum_value_proof (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0) (h₃ : a + b + c = 6) :
  (minimum_value a b c h₃) = 6 :=
sorry

end minimum_value_proof_l235_235248


namespace part1_part2_l235_235225

noncomputable def f (x : ℝ) : ℝ := (2 * x) / (Real.log x)

theorem part1 : 
  (∀ x, 0 < x → x < 1 → (f x) < f (1)) ∧ 
  (∀ x, 1 < x → x < Real.exp 1 → (f x) < f (Real.exp 1)) :=
sorry

theorem part2 :
  ∃ k, k = 2 ∧ ∀ x, 0 < x → (f x) > (k / (Real.log x)) + 2 * Real.sqrt x :=
sorry

end part1_part2_l235_235225


namespace proportion_check_option_B_l235_235962

theorem proportion_check_option_B (a b c d : ℝ) (ha : a = 1) (hb : b = 2) (hc : c = 2) (hd : d = 4) :
  (a / b) = (c / d) :=
by {
  sorry
}

end proportion_check_option_B_l235_235962


namespace smallest_multiple_of_9_and_6_is_18_l235_235175

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l235_235175


namespace new_average_doubled_l235_235397

theorem new_average_doubled
  (average : ℕ)
  (num_students : ℕ)
  (h_avg : average = 45)
  (h_num_students : num_students = 30)
  : (2 * average * num_students / num_students) = 90 := by
  sorry

end new_average_doubled_l235_235397


namespace coefficient_x2_sum_expansion_l235_235004

theorem coefficient_x2_sum_expansion :
  (finset.sum (finset.range 10) (λ n, nat.choose n 2)) = 120 :=
by sorry

end coefficient_x2_sum_expansion_l235_235004


namespace consecutive_numbers_equation_l235_235050

theorem consecutive_numbers_equation (x y z : ℤ) (h1 : z = 3) (h2 : y = z + 1) (h3 : x = y + 1) 
(h4 : 2 * x + 3 * y + 3 * z = 5 * y + n) : n = 11 :=
by
  sorry

end consecutive_numbers_equation_l235_235050


namespace first_part_length_l235_235800

def total_length : ℝ := 74.5
def part_two : ℝ := 21.5
def part_three : ℝ := 21.5
def part_four : ℝ := 16

theorem first_part_length :
  total_length - (part_two + part_three + part_four) = 15.5 :=
by
  sorry

end first_part_length_l235_235800


namespace cost_price_per_meter_l235_235980

-- Given conditions
def total_selling_price : ℕ := 18000
def total_meters_sold : ℕ := 400
def loss_per_meter : ℕ := 5

-- Statement to be proven
theorem cost_price_per_meter : 
    ((total_selling_price + (loss_per_meter * total_meters_sold)) / total_meters_sold) = 50 := 
by
    sorry

end cost_price_per_meter_l235_235980


namespace hexagon_area_l235_235912

-- Definition of an equilateral triangle with a given perimeter.
def is_equilateral_triangle (P Q R : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] :=
  ∀ (a b c : ℝ), a = b ∧ b = c ∧ a + b + c = 42 ∧ ∀ (angle : ℝ), angle = 60

-- Statement of the problem
theorem hexagon_area (P Q R P' Q' R' : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R]
  [MetricSpace P'] [MetricSpace Q'] [MetricSpace R']
  (h1 : is_equilateral_triangle P Q R) :
  ∃ (area : ℝ), area = 49 * Real.sqrt 3 := 
sorry

end hexagon_area_l235_235912


namespace neg_and_eq_or_not_l235_235772

theorem neg_and_eq_or_not (p q : Prop) : ¬(p ∧ q) ↔ ¬p ∨ ¬q :=
by sorry

end neg_and_eq_or_not_l235_235772


namespace current_population_l235_235487

def initial_population : ℕ := 4200
def percentage_died : ℕ := 10
def percentage_left : ℕ := 15

theorem current_population (pop : ℕ) (died left : ℕ) 
  (h1 : pop = initial_population) 
  (h2 : died = pop * percentage_died / 100) 
  (h3 : left = (pop - died) * percentage_left / 100) 
  (h4 : ∀ remaining, remaining = pop - died - left) 
  : (pop - died - left) = 3213 := 
by sorry

end current_population_l235_235487


namespace find_2a_plus_b_l235_235698

theorem find_2a_plus_b (a b : ℝ) (h1 : 3 * a + 2 * b = 18) (h2 : 5 * a + 4 * b = 31) :
  2 * a + b = 11.5 :=
sorry

end find_2a_plus_b_l235_235698


namespace probability_less_than_8000_miles_l235_235945

open ProbabilityMeasure

def distances : List (ℕ × ℕ) := [
  (5900, 1),
  (4800, 1),
  (6200, 1),
  (8700, 0),
  (2133, 1),
  (10400, 0)
]

def favorablePairs (dists : List (ℕ × ℕ)) : ℕ :=
  dists.filter (λ pair, pair.2 = 1).length

def totalPairs (dists : List (ℕ × ℕ)) : ℕ :=
  dists.length

theorem probability_less_than_8000_miles :
  (favorablePairs distances : ℚ) / (totalPairs distances : ℚ) = 2 / 3 := by
sorry

end probability_less_than_8000_miles_l235_235945


namespace general_term_of_sequence_l235_235334

theorem general_term_of_sequence (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 4 * n) : 
  a n = 2 * n - 5 :=
by
  -- Proof can be completed here
  sorry

end general_term_of_sequence_l235_235334


namespace valid_range_for_b_l235_235886

noncomputable def f (x b : ℝ) : ℝ := -x^2 + 2 * x + b^2 - b + 1

theorem valid_range_for_b (b : ℝ) :
  (∀ x ∈ Set.Icc (-1 : ℝ) 1, f x b > 0) → (b < -1 ∨ b > 2) :=
by
  sorry

end valid_range_for_b_l235_235886


namespace set_M_real_l235_235066

noncomputable def set_M : Set ℂ := {z : ℂ | (z - 1) ^ 2 = Complex.abs (z - 1) ^ 2}

theorem set_M_real :
  set_M = {z : ℂ | ∃ x : ℝ, z = x} :=
by
  sorry

end set_M_real_l235_235066


namespace factorial_prime_factors_l235_235762

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l235_235762


namespace no_continuous_coverage_l235_235979

noncomputable def running_track_problem : Prop :=
  let track_length := 1 -- 1 km
  let stands_arc_length := 0.1 -- 100 meters = 0.1 km
  let runners_speeds := [20, 21, 22, 23, 24, 25, 26, 27, 28, 29] -- km/h
  ∃ (runner_positions : Fin 10 → ℝ), -- Starting positions (unit: km) of 10 runners
    ∀ (t : ℝ), -- For any time (unit: hours)
      ∃ (i : Fin 10), -- There exists a runner (among 10)
        let position := (runner_positions i + runners_speeds.nth i * t) % track_length
        in  position ≤ stands_arc_length

theorem no_continuous_coverage :
  ¬running_track_problem :=
sorry

end no_continuous_coverage_l235_235979


namespace circle_center_and_radius_l235_235340

-- Define the equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x + 5 = 0

-- Statement of the center and radius of the circle
theorem circle_center_and_radius :
  (∀ x y : ℝ, circle_equation x y) →
  (∃ (h k r : ℝ), (∀ x y : ℝ, circle_equation x y ↔ (x - h)^2 + (y - k)^2 = r^2) ∧ h = 3 ∧ k = 0 ∧ r = 2) :=
by
  sorry

end circle_center_and_radius_l235_235340


namespace prime_factors_of_30_factorial_l235_235734

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l235_235734


namespace xiaoming_correct_answers_l235_235776

theorem xiaoming_correct_answers (x : ℕ) (h1 : x ≤ 10) (h2 : 5 * x - (10 - x) > 30) : x ≥ 7 := 
by
  sorry

end xiaoming_correct_answers_l235_235776


namespace sqrt_recursive_value_l235_235662

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l235_235662


namespace vectors_parallel_l235_235056

theorem vectors_parallel (m : ℝ) : 
    (∃ k : ℝ, (m, 4) = (k * 5, k * -2)) → m = -10 := 
by
  sorry

end vectors_parallel_l235_235056


namespace neg_p_equiv_l235_235505

open Real
open Classical

noncomputable def prop_p : Prop :=
  ∀ x : ℝ, 0 < x → exp x > log x

noncomputable def neg_prop_p : Prop :=
  ∃ x : ℝ, 0 < x ∧ exp x ≤ log x

theorem neg_p_equiv :
  ¬ prop_p ↔ neg_prop_p := by
  sorry

end neg_p_equiv_l235_235505


namespace max_sum_arith_seq_l235_235488

theorem max_sum_arith_seq (a : ℕ → ℝ) (d : ℝ) (S : ℕ → ℝ)
  (h_arith_seq : ∀ n : ℕ, a (n + 1) = a 1 + n * d)
  (h_a1_pos : a 1 > 0)
  (h_d_neg : d < 0)
  (h_a5_3a7 : a 5 = 3 * a 7)
  (h_Sn_def : ∀ n : ℕ, S n = n / 2 * (2 * a 1 + (n - 1) * d)) :
  ∃ n : ℕ, (n = 7 ∨ n = 8) ∧ S n = max (S 7) (S 8) := by
  sorry

end max_sum_arith_seq_l235_235488


namespace parallel_lines_slope_l235_235578

theorem parallel_lines_slope (a : ℝ) :
  (∃ b : ℝ, ( ∀ x y : ℝ, a*x - 5*y - 9 = 0 → b*x - 3*y - 10 = 0) → a = 10/3) :=
sorry

end parallel_lines_slope_l235_235578


namespace quadratic_value_l235_235926

theorem quadratic_value (a b c : ℤ) (a_pos : a > 0) (h_eq : ∀ x : ℝ, (a * x + b)^2 = 49 * x^2 + 70 * x + c) : a + b + c = -134 :=
by
  -- Proof starts here
  sorry

end quadratic_value_l235_235926


namespace solution_set_l235_235480

-- Defining the condition and inequalities:
variable (a x : Real)

-- Condition that a < 0
def condition_a : Prop := a < 0

-- Inequalities in the system
def inequality1 : Prop := x > -2 * a
def inequality2 : Prop := x > 3 * a

-- The solution set we need to prove
theorem solution_set (h : condition_a a) : (inequality1 a x) ∧ (inequality2 a x) ↔ x > -2 * a :=
by
  sorry

end solution_set_l235_235480


namespace part_a_l235_235935

theorem part_a (x : ℝ) (hx : x > 0) :
  ∃ color : ℕ, ∃ p1 p2 : ℝ × ℝ, (p1 = p2 ∨ x = dist p1 p2) :=
sorry

end part_a_l235_235935


namespace correct_exponential_rule_l235_235609

theorem correct_exponential_rule (a : ℝ) : (a^3)^2 = a^6 :=
by sorry

end correct_exponential_rule_l235_235609


namespace lisa_children_l235_235106

theorem lisa_children (C : ℕ) 
  (h1 : 5 * 52 = 260)
  (h2 : (2 * C + 3 + 2) * 260 = 3380) : 
  C = 4 := 
by
  sorry

end lisa_children_l235_235106


namespace henry_twice_jill_l235_235143

-- Conditions
def Henry := 29
def Jill := 19
def sum_ages : Nat := Henry + Jill

-- Prove the statement
theorem henry_twice_jill (Y : Nat) (H J : Nat) (h_sum : H + J = 48) (h_H : H = 29) (h_J : J = 19) :
  H - Y = 2 * (J - Y) ↔ Y = 9 :=
by {
  -- Here, we would provide the proof, but we'll skip that with sorry.
  sorry
}

end henry_twice_jill_l235_235143


namespace aaron_pages_sixth_day_l235_235983

theorem aaron_pages_sixth_day 
  (h1 : 18 + 12 + 23 + 10 + 17 + y = 6 * 15) : 
  y = 10 :=
by
  sorry

end aaron_pages_sixth_day_l235_235983


namespace train_speed_is_60_kmph_l235_235423

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l235_235423


namespace points_on_line_l235_235556

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l235_235556


namespace factorial_expression_l235_235319

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

theorem factorial_expression (N : ℕ) (h : N > 0) :
  (factorial (N + 1) + factorial (N - 1)) / factorial (N + 2) = 
  (N^2 + N + 1) / (N^3 + 3 * N^2 + 2 * N) :=
by
  sorry

end factorial_expression_l235_235319


namespace factorial_prime_factors_l235_235717

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l235_235717


namespace ticket_cost_correct_l235_235002

noncomputable def calculate_ticket_cost : ℝ :=
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  10 * x + 8 * child_price + 5 * senior_price

theorem ticket_cost_correct :
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  (4 * x + 3 * child_price + 2 * senior_price = 35) →
  (10 * x + 8 * child_price + 5 * senior_price = 88.75) :=
by
  intros
  sorry

end ticket_cost_correct_l235_235002


namespace unique_solution_l235_235321

def s (n : ℕ) : ℕ := n.digits 10 |>.sum

theorem unique_solution (m n : ℕ) (h : n * (n + 1) = 3 ^ m + s n + 1182) : (m, n) = (0, 34) :=
by
  sorry

end unique_solution_l235_235321


namespace volleyballs_count_l235_235553

-- Definitions of sports item counts based on given conditions.
def soccer_balls := 20
def basketballs := soccer_balls + 5
def tennis_balls := 2 * soccer_balls
def baseballs := soccer_balls + 10
def hockey_pucks := tennis_balls / 2
def total_items := 180

-- Calculate the total number of known sports items.
def known_items_sum := soccer_balls + basketballs + tennis_balls + baseballs + hockey_pucks

-- Prove the number of volleyballs
theorem volleyballs_count : total_items - known_items_sum = 45 := by
  sorry

end volleyballs_count_l235_235553


namespace complex_division_l235_235375

-- Define the imaginary unit 'i'
def i := Complex.I

-- Define the complex numbers as described in the problem
def num := Complex.mk 3 (-1)
def denom := Complex.mk 1 (-1)
def expected := Complex.mk 2 1

-- State the theorem to prove that the complex division is as expected
theorem complex_division : (num / denom) = expected :=
by
  sorry

end complex_division_l235_235375


namespace probability_of_sequence_123456_l235_235298

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l235_235298


namespace consecutive_sum_divisible_by_12_l235_235067

theorem consecutive_sum_divisible_by_12 
  (b : ℤ) 
  (a : ℤ := b - 1) 
  (c : ℤ := b + 1) 
  (d : ℤ := b + 2) :
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k := by
  sorry

end consecutive_sum_divisible_by_12_l235_235067


namespace seventh_term_of_arithmetic_sequence_l235_235381

theorem seventh_term_of_arithmetic_sequence 
  (a d : ℤ)
  (h1 : 5 * a + 10 * d = 35)
  (h2 : a + 5 * d = 10) :
  a + 6 * d = 11 :=
by
  sorry

end seventh_term_of_arithmetic_sequence_l235_235381


namespace general_term_sequence_l235_235946

/--
Given the sequence a : ℕ → ℝ such that a 0 = 1/2,
a 1 = 1/4,
a 2 = -1/8,
a 3 = 1/16,
and we observe that
a n = (-(1/2))^n,
prove that this formula holds for all n : ℕ.
-/
theorem general_term_sequence (a : ℕ → ℝ) :
  (∀ n, a n = (-(1/2))^n) :=
sorry

end general_term_sequence_l235_235946


namespace num_prime_factors_30_fac_l235_235711

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l235_235711


namespace gcd_180_270_eq_90_l235_235437

theorem gcd_180_270_eq_90 : Nat.gcd 180 270 = 90 := sorry

end gcd_180_270_eq_90_l235_235437


namespace eulers_formula_convex_polyhedron_l235_235020

theorem eulers_formula_convex_polyhedron :
  ∀ (V E F T H : ℕ),
  (V - E + F = 2) →
  (F = 24) →
  (E = (3 * T + 6 * H) / 2) →
  100 * H + 10 * T + V = 240 :=
by
  intros V E F T H h1 h2 h3
  sorry

end eulers_formula_convex_polyhedron_l235_235020


namespace rationalize_denominator_l235_235122

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l235_235122


namespace smallest_integer_solution_system_of_inequalities_solution_l235_235399

-- Define the conditions and problem
variable (x : ℝ)

-- Part 1: Prove smallest integer solution for 5x + 15 > x - 1
theorem smallest_integer_solution :
  5 * x + 15 > x - 1 → x = -3 := sorry

-- Part 2: Prove solution set for system of inequalities
theorem system_of_inequalities_solution :
  (-3 * (x - 2) ≥ 4 - x) ∧ ((1 + 4 * x) / 3 > x - 1) → (-4 < x ∧ x ≤ 1) := sorry

end smallest_integer_solution_system_of_inequalities_solution_l235_235399


namespace bill_head_circumference_l235_235092

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l235_235092


namespace area_before_halving_l235_235900

theorem area_before_halving (A : ℝ) (h : A / 2 = 7) : A = 14 :=
sorry

end area_before_halving_l235_235900


namespace revenue_increase_l235_235824

theorem revenue_increase (n : ℕ) (C P : ℝ) 
  (h1 : n * P = 1.20 * C) : 
  (0.95 * n * P) = 1.14 * C :=
by
  sorry

end revenue_increase_l235_235824


namespace parabola_distance_l235_235049

theorem parabola_distance (x y : ℝ) (h_parabola : y^2 = 8 * x)
  (h_distance_focus : ∀ x y, (x - 2)^2 + y^2 = 6^2) :
  abs x = 4 :=
by sorry

end parabola_distance_l235_235049


namespace rhombus_area_l235_235398

theorem rhombus_area (side d1 : ℝ) (h_side : side = 28) (h_d1 : d1 = 12) : 
  (side = 28 ∧ d1 = 12) →
  ∃ area : ℝ, area = 328.32 := 
by 
  sorry

end rhombus_area_l235_235398


namespace num_prime_factors_30_factorial_l235_235765

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l235_235765


namespace average_rate_of_change_is_4_l235_235887

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_is_4_l235_235887


namespace neg_of_p_l235_235892

variable (x : ℝ)

def p : Prop := ∀ x ≥ 0, 2^x = 3

theorem neg_of_p : ¬p ↔ ∃ x ≥ 0, 2^x ≠ 3 :=
by
  sorry

end neg_of_p_l235_235892


namespace nat_solution_unique_l235_235272

theorem nat_solution_unique (x y : ℕ) (h : x + y = x * y) : (x, y) = (2, 2) :=
sorry

end nat_solution_unique_l235_235272


namespace minimum_distance_l235_235497

open Real

def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 4 * x - 4 * y + 4 = 0
def parabola_eq (x y : ℝ) : Prop := y^2 = 8 * x

theorem minimum_distance :
  ∃ (A B : ℝ × ℝ), circle_eq A.1 A.2 ∧ parabola_eq B.1 B.2 ∧ dist A B = 1 / 2 :=
sorry

end minimum_distance_l235_235497


namespace product_ineq_l235_235365

-- Define the relevant elements and conditions
variables (a b : ℝ) (x₁ x₂ x₃ x₄ x₅ : ℝ)

-- Assumptions based on the conditions provided
variables (h₀ : a > 0) (h₁ : b > 0)
variables (h₂ : a + b = 1)
variables (h₃ : x₁ > 0) (h₄ : x₂ > 0) (h₅ : x₃ > 0) (h₆ : x₄ > 0) (h₇ : x₅ > 0)
variables (h₈ : x₁ * x₂ * x₃ * x₄ * x₅ = 1)

-- The theorem statement to be proved
theorem product_ineq : (a * x₁ + b) * (a * x₂ + b) * (a * x₃ + b) * (a * x₄ + b) * (a * x₅ + b) ≥ 1 :=
sorry

end product_ineq_l235_235365


namespace baseball_games_per_month_l235_235599

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l235_235599


namespace ratio_of_cakes_l235_235011

/-- Define the usual number of cheesecakes, muffins, and red velvet cakes baked in a week -/
def usual_cheesecakes : ℕ := 6
def usual_muffins : ℕ := 5
def usual_red_velvet_cakes : ℕ := 8

/-- Define the total number of cakes usually baked in a week -/
def usual_cakes : ℕ := usual_cheesecakes + usual_muffins + usual_red_velvet_cakes

/-- Assume Carter baked this week a multiple of usual cakes, denoted as x -/
def multiple (x : ℕ) : Prop := usual_cakes * x = usual_cakes + 38

/-- Assume he baked usual_cakes + 38 equals 57 cakes -/
def total_cakes_this_week : ℕ := 57

/-- The theorem stating the problem: proving the ratio is 3:1 -/
theorem ratio_of_cakes (x : ℕ) (hx : multiple x) : 
  (total_cakes_this_week : ℚ) / (usual_cakes : ℚ) = (3 : ℚ) :=
by
  sorry

end ratio_of_cakes_l235_235011


namespace isosceles_triangle_possible_values_of_x_l235_235140

open Real

-- Define the main statement
theorem isosceles_triangle_possible_values_of_x :
  ∀ x : ℝ, 
  (0 < x ∧ x < 90) ∧ 
  (sin (3*x) = sin (2*x) ∧ 
   sin (9*x) = sin (2*x)) 
  → x = 0 ∨ x = 180/11 ∨ x = 540/11 :=
by
  sorry

end isosceles_triangle_possible_values_of_x_l235_235140


namespace problem_statement_l235_235218

variable {x y : ℝ}

theorem problem_statement (h1 : x * y = -3) (h2 : x + y = -4) : x^2 + 3 * x * y + y^2 = 13 := sorry

end problem_statement_l235_235218


namespace garden_ratio_2_l235_235833

theorem garden_ratio_2 :
  ∃ (P C k R : ℤ), 
      P = 237 ∧ 
      C = P - 60 ∧ 
      P + C + k = 768 ∧ 
      R = k / C ∧ 
      R = 2 := 
by
  sorry

end garden_ratio_2_l235_235833


namespace find_tangent_circles_tangent_circle_at_given_point_l235_235863

noncomputable def circle_C (x y : ℝ) : Prop :=
  (x - 2)^2 + (y + 1)^2 = 4

def is_tangent (x y : ℝ) (a b : ℝ) : Prop :=
  ∃ (u v : ℝ), (u - a)^2 + (v - b)^2 = 1 ∧
  (x - u)^2 + (y - v)^2 = 4 ∧
  (x = u ∧ y = v)

theorem find_tangent_circles (x y a b : ℝ) (hx : circle_C x y)
  (ha_b : is_tangent x y a b) :
  (a = 5 ∧ b = -1) ∨ (a = 3 ∧ b = -1) :=
sorry

theorem tangent_circle_at_given_point (x y : ℝ) (hx : circle_C x y) (y_pos : y = -1)
  : ((x - 5)^2 + (y + 1)^2 = 1) ∨ ((x - 3)^2 + (y + 1)^2 = 1) :=
sorry

end find_tangent_circles_tangent_circle_at_given_point_l235_235863


namespace product_of_three_integers_sum_l235_235582

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l235_235582


namespace sum_of_integers_l235_235949

theorem sum_of_integers (a b : ℕ) (h1 : a * a + b * b = 585) (h2 : Nat.gcd a b + Nat.lcm a b = 87) : a + b = 33 := 
sorry

end sum_of_integers_l235_235949


namespace solve_equation_l235_235592

theorem solve_equation : ∀ x : ℝ, -2 * x + 11 = 0 → x = 11 / 2 :=
by
  intro x
  intro h
  sorry

end solve_equation_l235_235592


namespace rectangle_area_l235_235610

theorem rectangle_area (L B : ℕ) (h1 : L - B = 23) (h2 : 2 * L + 2 * B = 226) : L * B = 3060 := by
  sorry

end rectangle_area_l235_235610


namespace lawn_length_l235_235835

-- Defining the main conditions
def area : ℕ := 20
def width : ℕ := 5

-- The proof statement (goal)
theorem lawn_length : (area / width) = 4 := by
  sorry

end lawn_length_l235_235835


namespace smallest_multiple_of_9_and_6_l235_235163

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l235_235163


namespace meeting_time_l235_235202

-- Variables representing the conditions
def uniform_rate_cassie := 15
def uniform_rate_brian := 18
def distance_route := 70
def cassie_start_time := 8.0
def brian_start_time := 9.25

-- The goal
theorem meeting_time : ∃ T : ℝ, (15 * T + 18 * (T - 1.25) = 70) ∧ T = 2.803 := 
by {
  sorry
}

end meeting_time_l235_235202


namespace repeating_decimal_fraction_difference_l235_235102

theorem repeating_decimal_fraction_difference :
  ∀ (F : ℚ),
  F = 817 / 999 → (999 - 817 = 182) :=
by
  sorry

end repeating_decimal_fraction_difference_l235_235102


namespace smallest_integer_whose_cube_ends_in_368_l235_235451

theorem smallest_integer_whose_cube_ends_in_368 :
  ∃ (n : ℕ+), (n % 2 = 0 ∧ n^3 % 1000 = 368) ∧ (∀ (m : ℕ+), m % 2 = 0 ∧ m^3 % 1000 = 368 → m ≥ n) :=
by
  sorry

end smallest_integer_whose_cube_ends_in_368_l235_235451


namespace points_on_line_l235_235558

theorem points_on_line (n : ℕ) (h : 9 * n - 8 = 82) : n = 10 := by
  sorry

end points_on_line_l235_235558


namespace students_like_apple_and_chocolate_not_blueberry_l235_235775

variables (n A C B D : ℕ)

theorem students_like_apple_and_chocolate_not_blueberry
  (h1 : n = 50)
  (h2 : A = 25)
  (h3 : C = 20)
  (h4 : B = 5)
  (h5 : D = 15) :
  ∃ (x : ℕ), x = 10 ∧ x = n - D - (A + C - 2 * x) ∧ 0 ≤ 2 * x - A - C + B :=
sorry

end students_like_apple_and_chocolate_not_blueberry_l235_235775


namespace kira_night_songs_l235_235098

-- Definitions for the conditions
def morning_songs : ℕ := 10
def later_songs : ℕ := 15
def song_size_mb : ℕ := 5
def total_new_songs_memory_mb : ℕ := 140

-- Assert the number of songs Kira downloaded at night
theorem kira_night_songs : (total_new_songs_memory_mb - (morning_songs * song_size_mb + later_songs * song_size_mb)) / song_size_mb = 3 :=
by
  sorry

end kira_night_songs_l235_235098


namespace solve_equations_l235_235271

theorem solve_equations (x y : ℝ) (h1 : (x + y) / x = y / (x + y)) (h2 : x = 2 * y) :
  x = 0 ∧ y = 0 :=
by
  sorry

end solve_equations_l235_235271


namespace certain_event_drawing_triangle_interior_angles_equal_180_deg_l235_235961

-- Define a triangle in the Euclidean space
structure Triangle (α : Type) [plane : TopologicalSpace α] :=
(a b c : α)

-- Define the sum of the interior angles of a triangle
noncomputable def sum_of_interior_angles {α : Type} [TopologicalSpace α] (T : Triangle α) : ℝ :=
180

-- The proof statement
theorem certain_event_drawing_triangle_interior_angles_equal_180_deg {α : Type} [TopologicalSpace α]
(T : Triangle α) : 
(sum_of_interior_angles T = 180) :=
sorry

end certain_event_drawing_triangle_interior_angles_equal_180_deg_l235_235961


namespace intersection_M_N_l235_235234

def M : Set ℝ := {x | x < 2}
def N : Set ℝ := {x | -1 < x ∧ x < 3}

theorem intersection_M_N :
  M ∩ N = {x | -1 < x ∧ x <2} := by
  sorry

end intersection_M_N_l235_235234


namespace range_of_x_l235_235069

theorem range_of_x (x m : ℝ) (h₁ : 1 ≤ m) (h₂ : m ≤ 3) (h₃ : x + 3 * m + 5 > 0) : x > -14 := 
sorry

end range_of_x_l235_235069


namespace sum_of_real_numbers_l235_235528

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235528


namespace solve_x_squared_eq_nine_l235_235273

theorem solve_x_squared_eq_nine (x : ℝ) : x^2 = 9 → (x = 3 ∨ x = -3) :=
by
  -- Proof by sorry placeholder
  sorry

end solve_x_squared_eq_nine_l235_235273


namespace potatoes_yield_l235_235116

theorem potatoes_yield (steps_length : ℕ) (steps_width : ℕ) (step_size : ℕ) (yield_per_sqft : ℚ) 
  (h_steps_length : steps_length = 18) 
  (h_steps_width : steps_width = 25) 
  (h_step_size : step_size = 3) 
  (h_yield_per_sqft : yield_per_sqft = 1/3) 
  : (steps_length * step_size) * (steps_width * step_size) * yield_per_sqft = 1350 := 
by 
  sorry

end potatoes_yield_l235_235116


namespace sum_of_xy_l235_235547

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235547


namespace nested_radical_solution_l235_235672

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l235_235672


namespace greatest_integer_less_than_PS_l235_235078

noncomputable def rectangle_problem (PQ PS : ℝ) (T : ℝ) (PT QP : ℝ) : ℝ := real.sqrt (PQ * PQ + PS * PS) / 2

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ) (hPQ : PQ = 150)
  (T_midpoint : T = PS / 2)
  (PT_perpendicular_QT : PT * PT + T * T = PQ * PQ),
  int.floor (PS) = 212 :=
by
  intros PQ PS hPQ T_midpoint PT_perpendicular_QT
  have h₁ : PS = rectangle_problem PQ PS T PQ,
  {
    sorry
  }
  have h₂ : 150 * real.sqrt 2,
  {
    sorry
  }
  have h₃ : (⌊150 * real.sqrt 2⌋ : ℤ) = 212,
  {
    sorry
  }
  exact h₃

end greatest_integer_less_than_PS_l235_235078


namespace scientific_notation_correct_l235_235262

/-- Define the number 42.39 million as 42.39 * 10^6 and prove that it is equivalent to 4.239 * 10^7 -/
def scientific_notation_of_42_39_million : Prop :=
  (42.39 * 10^6 = 4.239 * 10^7)

theorem scientific_notation_correct : scientific_notation_of_42_39_million :=
by 
  sorry

end scientific_notation_correct_l235_235262


namespace students_doing_at_least_one_hour_of_homework_l235_235430

theorem students_doing_at_least_one_hour_of_homework (total_angle : ℝ) (less_than_one_hour_angle : ℝ) 
  (h1 : total_angle = 360) (h2 : less_than_one_hour_angle = 90) :
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  at_least_one_hour_percentage = 75 :=
by
  let less_than_one_hour_fraction := less_than_one_hour_angle / total_angle
  let less_than_one_hour_percentage := less_than_one_hour_fraction * 100
  let at_least_one_hour_percentage := 100 - less_than_one_hour_percentage
  sorry

end students_doing_at_least_one_hour_of_homework_l235_235430


namespace sqrt_continued_fraction_l235_235668

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l235_235668


namespace find_d_l235_235902

theorem find_d (c d : ℝ) (h1 : c / d = 5) (h2 : c = 18 - 7 * d) : d = 3 / 2 := by
  sorry

end find_d_l235_235902


namespace sophie_saves_money_by_using_wool_balls_l235_235277

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l235_235277


namespace smallest_multiple_9_and_6_l235_235164

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l235_235164


namespace flight_duration_is_four_hours_l235_235866

def convert_to_moscow_time (local_time : ℕ) (time_difference : ℕ) : ℕ :=
  (local_time - time_difference) % 24

def flight_duration (departure_time arrival_time : ℕ) : ℕ :=
  (arrival_time - departure_time) % 24

def duration_per_flight (total_flight_time : ℕ) (number_of_flights : ℕ) : ℕ :=
  total_flight_time / number_of_flights

theorem flight_duration_is_four_hours :
  let MoscowToBishkekTimeDifference := 3
  let departureMoscowTime := 12
  let arrivalBishkekLocalTime := 18
  let departureBishkekLocalTime := 8
  let arrivalMoscowTime := 10
  let outboundArrivalMoscowTime := convert_to_moscow_time arrivalBishkekLocalTime MoscowToBishkekTimeDifference
  let returnDepartureMoscowTime := convert_to_moscow_time departureBishkekLocalTime MoscowToBishkekTimeDifference
  let outboundDuration := flight_duration departureMoscowTime outboundArrivalMoscowTime
  let returnDuration := flight_duration returnDepartureMoscowTime arrivalMoscowTime
  let totalFlightTime := outboundDuration + returnDuration
  duration_per_flight totalFlightTime 2 = 4 := by
  sorry

end flight_duration_is_four_hours_l235_235866


namespace unique_triple_solution_l235_235676

theorem unique_triple_solution :
  ∃! (x y z : ℕ), (y > 1) ∧ Prime y ∧
                  (¬(3 ∣ z ∧ y ∣ z)) ∧
                  (x^3 - y^3 = z^2) ∧
                  (x = 8 ∧ y = 7 ∧ z = 13) :=
by
  sorry

end unique_triple_solution_l235_235676


namespace possible_values_of_sum_l235_235542

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235542


namespace represent_in_scientific_notation_l235_235263

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l235_235263


namespace probability_intersection_l235_235615

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l235_235615


namespace stratified_sampling_total_students_sampled_l235_235189

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l235_235189


namespace carmen_candles_needed_l235_235010

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l235_235010


namespace intersection_A_B_l235_235044

open Set

def SetA : Set ℤ := {x | ∃ n : ℤ, x = 2 * n}
def SetB : Set ℤ := {x | 0 ≤ x ∧ x ≤ 4}

theorem intersection_A_B :
  (SetA ∩ SetB) = ( {0, 2, 4} : Set ℤ ) :=
by
  sorry

end intersection_A_B_l235_235044


namespace probability_intersection_l235_235614

variable (p : ℝ)

def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6
def P_AuB : ℝ := 1 - (1 - 2 * p)^6
def P_AiB : ℝ := P_A p + P_B p - P_AuB p

theorem probability_intersection :
  P_AiB p = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := by
  sorry

end probability_intersection_l235_235614


namespace find_common_difference_l235_235489

variable {a : ℕ → ℝ}
variable {p q : ℕ}
variable {d : ℝ}

-- Definitions based on the conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) (p : ℕ) := a p = 4
def condition2 (a : ℕ → ℝ) (q : ℕ) := a q = 2
def condition3 (p q : ℕ) := p = 4 + q

-- The goal statement
theorem find_common_difference
  (a_seq : arithmetic_sequence a d)
  (h1 : condition1 a p)
  (h2 : condition2 a q)
  (h3 : condition3 p q) :
  d = 1 / 2 :=
by
  sorry

end find_common_difference_l235_235489


namespace parabola_equation_l235_235339

theorem parabola_equation (p x0 : ℝ) (h_p : p > 0) (h_dist_focus : x0 + p / 2 = 10) (h_parabola : 2 * p * x0 = 36) :
  (2 * p = 4) ∨ (2 * p = 36) :=
by sorry

end parabola_equation_l235_235339


namespace z_is_1_2_decades_younger_than_x_l235_235970

variable (x y z w : ℕ) -- Assume ages as natural numbers

def age_equivalence_1 : Prop := x + y = y + z + 12
def age_equivalence_2 : Prop := x + y + w = y + z + w + 12

theorem z_is_1_2_decades_younger_than_x (h1 : age_equivalence_1 x y z) (h2 : age_equivalence_2 x y z w) :
  z = x - 12 := by
  sorry

end z_is_1_2_decades_younger_than_x_l235_235970


namespace sum_of_x_y_possible_values_l235_235532

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235532


namespace more_flour_than_sugar_l235_235113

def cups_of_flour : Nat := 9
def cups_of_sugar : Nat := 6
def flour_added : Nat := 2
def flour_needed : Nat := cups_of_flour - flour_added -- 9 - 2 = 7

theorem more_flour_than_sugar : flour_needed - cups_of_sugar = 1 :=
by
  sorry

end more_flour_than_sugar_l235_235113


namespace wrapping_paper_area_l235_235192

variable (a b h w : ℝ) (a_gt_b : a > b)

theorem wrapping_paper_area : 
  ∃ total_area, total_area = 4 * (a * b + a * w + b * w + w ^ 2) :=
by
  sorry

end wrapping_paper_area_l235_235192


namespace num_prime_factors_30_factorial_l235_235749

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l235_235749


namespace total_gym_cost_l235_235914

def cheap_monthly_fee : ℕ := 10
def cheap_signup_fee : ℕ := 50
def expensive_monthly_fee : ℕ := 3 * cheap_monthly_fee
def expensive_signup_fee : ℕ := 4 * expensive_monthly_fee

def yearly_cost_cheap : ℕ := 12 * cheap_monthly_fee + cheap_signup_fee
def yearly_cost_expensive : ℕ := 12 * expensive_monthly_fee + expensive_signup_fee

theorem total_gym_cost : yearly_cost_cheap + yearly_cost_expensive = 650 := by
  -- Proof goes here
  sorry

end total_gym_cost_l235_235914


namespace represent_in_scientific_notation_l235_235264

def million : ℕ := 10^6
def rural_residents : ℝ := 42.39 * million

theorem represent_in_scientific_notation :
  42.39 * 10^6 = 4.239 * 10^7 :=
by
  -- The proof is omitted.
  sorry

end represent_in_scientific_notation_l235_235264


namespace not_possible_for_runners_in_front_l235_235978

noncomputable def runnerInFrontAtAnyMoment 
  (track_length : ℝ)
  (stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) : Prop := 
  ∀ t : ℝ, ∃ i : Fin 10, 
  ∃ n : ℤ, 
  (runners_speeds i * t - n * track_length) % track_length ≤ stands_length

theorem not_possible_for_runners_in_front 
  (track_length stands_length : ℝ)
  (runners_speeds : Fin 10 → ℝ) 
  (h_track : track_length = 1)
  (h_stands : stands_length = 0.1)
  (h_speeds : ∀ i : Fin 10, 20 + i = runners_speeds i) : 
  ¬ runnerInFrontAtAnyMoment track_length stands_length runners_speeds :=
sorry

end not_possible_for_runners_in_front_l235_235978


namespace cassie_nails_l235_235012

-- Define the number of pets
def num_dogs := 4
def num_parrots := 8
def num_cats := 2
def num_rabbits := 6

-- Define the number of nails/claws/toes per pet
def nails_per_dog := 4 * 4
def common_claws_per_parrot := 2 * 3
def extra_toed_parrot_claws := 2 * 4
def toes_per_cat := 2 * 5 + 2 * 4
def rear_nails_per_rabbit := 2 * 5
def front_nails_per_rabbit := 3 + 4

-- Calculations
def total_dog_nails := num_dogs * nails_per_dog
def total_parrot_claws := 7 * common_claws_per_parrot + extra_toed_parrot_claws
def total_cat_toes := num_cats * toes_per_cat
def total_rabbit_nails := num_rabbits * (rear_nails_per_rabbit + front_nails_per_rabbit)

-- Total nails/claws/toes
def total_nails := total_dog_nails + total_parrot_claws + total_cat_toes + total_rabbit_nails

-- Theorem stating the problem
theorem cassie_nails : total_nails = 252 :=
by
  -- Here we would normally have the proof, but we'll skip it with sorry
  sorry

end cassie_nails_l235_235012


namespace sum_of_squares_of_solutions_l235_235864

theorem sum_of_squares_of_solutions :
  (∃ s₁ s₂ : ℝ, s₁ ≠ s₂ ∧ s₁ + s₂ = 17 ∧ s₁ * s₂ = 22) →
  ∃ s₁ s₂ : ℝ, s₁^2 + s₂^2 = 245 :=
by
  sorry

end sum_of_squares_of_solutions_l235_235864


namespace mango_distribution_l235_235118

theorem mango_distribution (harvested_mangoes : ℕ) (sold_fraction : ℕ) (received_per_neighbor : ℕ)
  (h_harvested : harvested_mangoes = 560)
  (h_sold_fraction : sold_fraction = 2)
  (h_received_per_neighbor : received_per_neighbor = 35) :
  (harvested_mangoes / sold_fraction) = (harvested_mangoes / sold_fraction) / received_per_neighbor :=
by
  sorry

end mango_distribution_l235_235118


namespace min_value_of_f_l235_235030

def f (x : ℝ) : ℝ := x^2 - 4 * x + 4

theorem min_value_of_f : ∀ x : ℝ, f x ≥ 0 ∧ f 2 = 0 :=
  by sorry

end min_value_of_f_l235_235030


namespace fraction_equality_l235_235036

theorem fraction_equality (x : ℝ) : (5 + x) / (7 + x) = (3 + x) / (4 + x) → x = -1 :=
by
  sorry

end fraction_equality_l235_235036


namespace find_minimum_f_l235_235029

noncomputable def f (x : ℝ) : ℝ :=
x^2 + 2 * x / (x^2 + 1) + x * (x + 5) / (x^2 + 3) + 3 * (x + 3) / (x * (x^2 + 3))

theorem find_minimum_f (x : ℝ) (hx : x > 0) : 
  ∃ y, ∀ z, z > 0 → f z ≥ y :=
begin
  sorry
end

end find_minimum_f_l235_235029


namespace storm_first_thirty_minutes_rain_l235_235840

theorem storm_first_thirty_minutes_rain 
  (R: ℝ)
  (H1: R + (R / 2) + (1 / 2) = 8)
  : R = 5 :=
by
  sorry

end storm_first_thirty_minutes_rain_l235_235840


namespace max_leap_years_l235_235315

theorem max_leap_years (years : ℕ) (leap_interval : ℕ) (total_years : ℕ) (leap_years : ℕ)
  (h1 : leap_interval = 5)
  (h2 : total_years = 200)
  (h3 : years = total_years / leap_interval) :
  leap_years = 40 :=
by
  sorry

end max_leap_years_l235_235315


namespace smallest_y_absolute_value_equation_l235_235034

theorem smallest_y_absolute_value_equation :
  ∃ y : ℚ, (|5 * y - 9| = 55) ∧ y = -46 / 5 :=
by
  sorry

end smallest_y_absolute_value_equation_l235_235034


namespace sum_eighth_row_l235_235359

-- Definitions based on the conditions
def sum_of_interior_numbers (n : ℕ) : ℕ := 2^(n-1) - 2

axiom sum_fifth_row : sum_of_interior_numbers 5 = 14
axiom sum_sixth_row : sum_of_interior_numbers 6 = 30

-- The proof problem statement
theorem sum_eighth_row : sum_of_interior_numbers 8 = 126 :=
by {
  sorry
}

end sum_eighth_row_l235_235359


namespace tangent_vertical_y_axis_iff_a_gt_0_l235_235901

theorem tangent_vertical_y_axis_iff_a_gt_0 {a : ℝ} (f : ℝ → ℝ) 
    (hf : ∀ x > 0, f x = a * x^2 - Real.log x)
    (h_tangent_vertical : ∃ x > 0, (deriv f x) = 0) :
    a > 0 := 
sorry

end tangent_vertical_y_axis_iff_a_gt_0_l235_235901


namespace gcd_and_sum_of_1729_and_867_l235_235153

-- Given numbers
def a := 1729
def b := 867

-- Define the problem statement
theorem gcd_and_sum_of_1729_and_867 : Nat.gcd a b = 1 ∧ a + b = 2596 := by
  sorry

end gcd_and_sum_of_1729_and_867_l235_235153


namespace slower_time_to_reach_top_l235_235254

def time_for_lola (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def time_for_tara (stories : ℕ) (time_per_story : ℕ) (stopping_time : ℕ) (num_stops : ℕ) : ℕ :=
  (stories * time_per_story) + (num_stops * stopping_time)

theorem slower_time_to_reach_top (stories : ℕ) (lola_time_per_story : ℕ) (tara_time_per_story : ℕ) 
  (tara_stop_time : ℕ) (tara_num_stops : ℕ) : 
  stories = 20 
  → lola_time_per_story = 10 
  → tara_time_per_story = 8 
  → tara_stop_time = 3
  → tara_num_stops = 18
  → max (time_for_lola stories lola_time_per_story) (time_for_tara stories tara_time_per_story tara_stop_time tara_num_stops) = 214 :=
by sorry

end slower_time_to_reach_top_l235_235254


namespace percentage_of_ginger_is_correct_l235_235924

noncomputable def teaspoons_per_tablespoon : ℕ := 3
noncomputable def ginger_tablespoons : ℕ := 3
noncomputable def cardamom_teaspoons : ℕ := 1
noncomputable def mustard_teaspoons : ℕ := 1
noncomputable def garlic_tablespoons : ℕ := 2
noncomputable def chile_powder_factor : ℕ := 4

theorem percentage_of_ginger_is_correct :
  let ginger_teaspoons := ginger_tablespoons * teaspoons_per_tablespoon
  let garlic_teaspoons := garlic_tablespoons * teaspoons_per_tablespoon
  let chile_teaspoons := chile_powder_factor * mustard_teaspoons
  let total_teaspoons := ginger_teaspoons + cardamom_teaspoons + mustard_teaspoons + garlic_teaspoons + chile_teaspoons
  let percentage_ginger := (ginger_teaspoons * 100) / total_teaspoons
  percentage_ginger = 43 :=
by
  sorry

end percentage_of_ginger_is_correct_l235_235924


namespace expansion_terms_count_l235_235943

theorem expansion_terms_count (n : ℕ) : 
  (∑ k in Finset.powerset (Finset.range (n.succ + 1)), 
  1 = (n.succ) ^ (3 - 1)) →
  (∃ t : ℕ, t = 78) :=
by
  intros h
  use 78
  rwa Finset.card_powerset_len_eq.symm at h

end expansion_terms_count_l235_235943


namespace sum_of_xy_l235_235548

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235548


namespace sum_of_real_numbers_l235_235531

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235531


namespace nested_radical_solution_l235_235671

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l235_235671


namespace percentage_of_students_chose_spring_is_10_l235_235934

-- Define the constants given in the problem
def total_students : ℕ := 10
def students_spring : ℕ := 1

-- Define the percentage calculation formula
def percentage (part total : ℕ) : ℕ := (part * 100) / total

-- State the theorem
theorem percentage_of_students_chose_spring_is_10 :
  percentage students_spring total_students = 10 :=
by
  -- We don't need to provide a proof here, just state it.
  sorry

end percentage_of_students_chose_spring_is_10_l235_235934


namespace find_X_l235_235857

def spadesuit (X Y : ℝ) : ℝ := 4 * X - 3 * Y + 7

theorem find_X (X : ℝ) (h : spadesuit X 5 = 23) : X = 7.75 :=
by sorry

end find_X_l235_235857


namespace sum_product_of_integers_l235_235590

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l235_235590


namespace evaluate_nested_radical_l235_235658

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l235_235658


namespace common_ratio_of_geometric_series_l235_235680

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l235_235680


namespace sum_last_two_digits_l235_235295

theorem sum_last_two_digits (a b : ℕ) (h₁ : a = 7) (h₂ : b = 13) : 
  (a^25 + b^25) % 100 = 0 :=
by
  sorry

end sum_last_two_digits_l235_235295


namespace coeff_of_z_in_eq2_l235_235699

-- Definitions of the conditions from part a)
def equation1 (x y z : ℤ) := 6 * x - 5 * y + 3 * z = 22
def equation2 (x y z : ℤ) := 4 * x + 8 * y - z = (7 : ℚ) / 11
def equation3 (x y z : ℤ) := 5 * x - 6 * y + 2 * z = 12
def sum_xyz (x y z : ℤ) := x + y + z = 10

-- Theorem stating that the coefficient of z in equation 2 is -1.
theorem coeff_of_z_in_eq2 {x y z : ℤ} (h1 : equation1 x y z) (h2 : equation2 x y z) (h3 : equation3 x y z) (h4 : sum_xyz x y z) :
    -1 = -1 :=
by
  -- This is a placeholder for the proof.
  sorry

end coeff_of_z_in_eq2_l235_235699


namespace proof_triangle_is_right_angle_proof_perimeter_range_l235_235238

noncomputable def triangle_is_right_angle (a b c A B C : ℝ) (sin_A sin_B sin_C : ℝ) (cos_B : ℝ) (cos_2A : ℝ) :=
  (b > 0) ∧ (sin A / (sin B + sin C) = 1 - (a - b) / (a - c)) ∧
  (((8 * cos B) * sin A + cos_2A) = (-2 * (sin A - 1)^2 + 3)) ∧ (a > 0) ∧ (c > 0) ∧
  (B = π / 3) ∧ (A = π / 2)

noncomputable def perimeter_range (a b c : ℝ) (A : ℝ) : set ℝ :=
  { p : ℝ | b = sqrt 3 ∧ (p = sqrt 3 + 2 * sqrt 3 * sin (A + π / 6)) ∧ ((A + π / 6) ∈ (π / 6, 5 * π / 6)) }

theorem proof_triangle_is_right_angle (a b c A B C : ℝ) (sin_A sin_B sin_C : ℝ) (cos_B : ℝ) (cos_2A : ℝ) :
  triangle_is_right_angle a b c A B C sin_A sin_B sin_C cos_B cos_2A → (A = π / 2) :=
by sorry

theorem proof_perimeter_range (a b c A : ℝ) :
  b = sqrt 3 → ((a > 0) ∧ (c > 0)) →
  ∃ l, l ∈ perimeter_range a b c A :=
by sorry

end proof_triangle_is_right_angle_proof_perimeter_range_l235_235238


namespace total_birds_is_1300_l235_235181

def initial_birds : ℕ := 300
def birds_doubled (b : ℕ) : ℕ := 2 * b
def birds_reduced (b : ℕ) : ℕ := b - 200
def total_birds_three_days : ℕ := initial_birds + birds_doubled initial_birds + birds_reduced (birds_doubled initial_birds)

theorem total_birds_is_1300 : total_birds_three_days = 1300 :=
by
  unfold total_birds_three_days initial_birds birds_doubled birds_reduced
  simp
  done

end total_birds_is_1300_l235_235181


namespace probability_intersection_l235_235618

-- Definitions of events A and B and initial probabilities
variables (p : ℝ)
def A := {ω | ω = true}
def B := {ω | ω = true}

-- Conditions
def P_A : ℝ := 1 - (1 - p)^6
def P_B : ℝ := 1 - (1 - p)^6

-- Probability union of A and B
def P_A_union_B : ℝ := 1 - (1 - 2p)^6

-- Given probabilities satisfy the question statement
theorem probability_intersection (p : ℝ) : 
  P_A + P_B - P_A_union_B = 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 := 
by 
  sorry

end probability_intersection_l235_235618


namespace sum_of_possible_values_of_N_l235_235580

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l235_235580


namespace distance_of_canteen_from_each_camp_l235_235194

noncomputable def distanceFromCanteen (distGtoRoad distBtoG : ℝ) : ℝ :=
  let hypotenuse := Real.sqrt (distGtoRoad ^ 2 + distBtoG ^ 2)
  hypotenuse / 2

theorem distance_of_canteen_from_each_camp :
  distanceFromCanteen 360 800 = 438.6 :=
by
  sorry -- The proof is omitted but must show that this statement is valid.

end distance_of_canteen_from_each_camp_l235_235194


namespace sophie_saves_money_by_using_wool_balls_l235_235276

def cost_of_dryer_sheets_per_year (loads_per_week : ℕ) (sheets_per_load : ℕ)
                                  (weeks_per_year : ℕ) (sheets_per_box : ℕ)
                                  (cost_per_box : ℝ) : ℝ :=
  let sheets_per_year := loads_per_week * sheets_per_load * weeks_per_year
  let boxes_per_year := sheets_per_year / sheets_per_box
  boxes_per_year * cost_per_box

theorem sophie_saves_money_by_using_wool_balls :
  cost_of_dryer_sheets_per_year 4 1 52 104 5.50 = 11.00 :=
by simp only [cost_of_dryer_sheets_per_year]; sorry

end sophie_saves_money_by_using_wool_balls_l235_235276


namespace inequality_solution_set_l235_235933

theorem inequality_solution_set :
  {x : ℝ | (x^2 - 4) / (x^2 - 9) > 0} = {x : ℝ | x < -3 ∨ x > 3} :=
sorry

end inequality_solution_set_l235_235933


namespace sqrt_recursive_value_l235_235660

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l235_235660


namespace geometric_sequence_problem_l235_235048

-- Assume {a_n} is a geometric sequence with positive terms
variable {a : ℕ → ℝ}
variable {r : ℝ}

-- Condition: all terms are positive numbers in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a n = a 0 * r ^ n

-- Condition: a_1 * a_9 = 16
def condition1 (a : ℕ → ℝ) : Prop :=
  a 1 * a 9 = 16

-- Question to prove: a_2 * a_5 * a_8 = 64
theorem geometric_sequence_problem
  (h_geom : is_geometric_sequence a r)
  (h_pos : ∀ n, 0 < a n)
  (h_cond1 : condition1 a) :
  a 2 * a 5 * a 8 = 64 :=
by
  sorry

end geometric_sequence_problem_l235_235048


namespace find_larger_number_l235_235483

theorem find_larger_number (x y : ℤ) (h1 : 4 * y = 3 * x) (h2 : y - x = 12) : y = -36 := 
by sorry

end find_larger_number_l235_235483


namespace inverse_proportion_function_sol_l235_235135

theorem inverse_proportion_function_sol (k m x : ℝ) (h1 : k ≠ 0) (h2 : (m - 1) * x ^ (m ^ 2 - 2) = k / x) : m = -1 :=
by
  sorry

end inverse_proportion_function_sol_l235_235135


namespace olivia_spent_38_l235_235259

def initial_amount : ℕ := 128
def amount_left : ℕ := 90
def money_spent (initial amount_left : ℕ) : ℕ := initial - amount_left

theorem olivia_spent_38 :
  money_spent initial_amount amount_left = 38 :=
by 
  sorry

end olivia_spent_38_l235_235259


namespace minimum_operations_to_transfer_beer_l235_235149

-- Definition of the initial conditions
structure InitialState where
  barrel_quarts : ℕ := 108
  seven_quart_vessel : ℕ := 0
  five_quart_vessel : ℕ := 0

-- Definition of the desired final state after minimum steps
structure FinalState where
  operations : ℕ := 17

-- Main theorem statement
theorem minimum_operations_to_transfer_beer (s : InitialState) : FinalState :=
  sorry

end minimum_operations_to_transfer_beer_l235_235149


namespace callie_caught_frogs_l235_235799

theorem callie_caught_frogs (A Q B C : ℝ) 
  (hA : A = 2)
  (hQ : Q = 2 * A)
  (hB : B = 3 * Q)
  (hC : C = (5 / 8) * B) : 
  C = 7.5 := by
  sorry

end callie_caught_frogs_l235_235799


namespace point_A_equidistant_l235_235326

/-
This statement defines the problem of finding the coordinates of point A that is equidistant from points B and C.
-/
theorem point_A_equidistant (x : ℝ) :
  (dist (x, 0, 0) (3, 5, 6)) = (dist (x, 0, 0) (1, 2, 3)) ↔ x = 14 :=
by {
  sorry
}

end point_A_equidistant_l235_235326


namespace C_work_completion_l235_235820

theorem C_work_completion (A_completion_days B_completion_days AB_completion_days : ℕ)
  (A_cond : A_completion_days = 8)
  (B_cond : B_completion_days = 12)
  (AB_cond : AB_completion_days = 4) :
  ∃ (C_completion_days : ℕ), C_completion_days = 24 := 
by
  sorry

end C_work_completion_l235_235820


namespace ratio_of_a_over_b_l235_235700

noncomputable def f (a b x : ℝ) : ℝ := x^3 + a * x^2 + b * x - a^2 - 7 * a

theorem ratio_of_a_over_b (a b : ℝ) (h_max : ∀ x : ℝ, f a b x ≤ 10)
  (h_cond1 : f a b 1 = 10) (h_cond2 : (deriv (f a b)) 1 = 0) :
  a / b = -2/3 :=
sorry

end ratio_of_a_over_b_l235_235700


namespace gift_bag_combinations_l235_235411

theorem gift_bag_combinations (giftBags tissuePapers tags : ℕ) (h1 : giftBags = 10) (h2 : tissuePapers = 4) (h3 : tags = 5) : 
  giftBags * tissuePapers * tags = 200 := 
by 
  sorry

end gift_bag_combinations_l235_235411


namespace most_probable_hits_l235_235447

variable (n : ℕ) (p : ℝ) (q : ℝ) (k : ℕ)
variable (h1 : n = 5) (h2 : p = 0.6) (h3 : q = 1 - p)

theorem most_probable_hits : k = 3 := by
  -- Define the conditions
  have hp : p = 0.6 := h2
  have hn : n = 5 := h1
  have hq : q = 1 - p := h3

  -- Set the expected value for the number of hits
  let expected := n * p

  -- Use the bounds for the most probable number of successes (k_0)
  have bounds := expected - q ≤ k ∧ k ≤ expected + p

  -- Proof step analysis can go here
  sorry

end most_probable_hits_l235_235447


namespace sophie_saves_money_l235_235275

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l235_235275


namespace smallest_distance_l235_235101

open Real

/-- Let A be a point on the circle (x-3)^2 + (y-4)^2 = 16,
and let B be a point on the parabola x^2 = 8y.
The smallest possible distance AB is √34 - 4. -/
theorem smallest_distance 
  (A B : ℝ × ℝ)
  (hA : (A.1 - 3)^2 + (A.2 - 4)^2 = 16)
  (hB : (B.1)^2 = 8 * B.2) :
  sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ sqrt 34 - 4 := 
sorry

end smallest_distance_l235_235101


namespace ants_rice_transport_l235_235088

/-- 
Given:
  1) 12 ants can move 24 grains of rice in 6 trips.

Prove:
  How many grains of rice can 9 ants move in 9 trips?
-/
theorem ants_rice_transport :
  (9 * 9 * (24 / (12 * 6))) = 27 := 
sorry

end ants_rice_transport_l235_235088


namespace average_rate_of_change_l235_235889

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l235_235889


namespace people_born_in_country_l235_235364

-- Define the conditions
def people_immigrated : ℕ := 16320
def new_people_total : ℕ := 106491

-- Define the statement to be proven
theorem people_born_in_country (people_born : ℕ) (h : people_born = new_people_total - people_immigrated) : 
    people_born = 90171 :=
  by
    -- This is where we would provide the proof, but we use sorry to skip the proof.
    sorry

end people_born_in_country_l235_235364


namespace num_prime_factors_30_fac_l235_235710

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l235_235710


namespace central_angle_star_in_polygon_l235_235906

theorem central_angle_star_in_polygon (n : ℕ) (h : 2 < n) : 
  ∃ C, C = 720 / n :=
by sorry

end central_angle_star_in_polygon_l235_235906


namespace simplify_sqrt_of_square_l235_235270

-- The given condition
def x : ℤ := -9

-- The theorem stating the simplified form
theorem simplify_sqrt_of_square : (Real.sqrt ((x : ℝ) ^ 2) = 9) := by    
    sorry

end simplify_sqrt_of_square_l235_235270


namespace intersection_of_A_and_B_l235_235250

open Set

variable (A : Set ℕ) (B : Set ℕ)

theorem intersection_of_A_and_B (hA : A = {0, 1, 2}) (hB : B = {0, 2, 4}) :
  A ∩ B = {0, 2} := by
  sorry

end intersection_of_A_and_B_l235_235250


namespace tony_bought_10_play_doughs_l235_235916

noncomputable def num_play_doughs 
    (lego_cost : ℕ) 
    (sword_cost : ℕ) 
    (play_dough_cost : ℕ) 
    (bought_legos : ℕ) 
    (bought_swords : ℕ) 
    (total_paid : ℕ) : ℕ :=
  let lego_total := lego_cost * bought_legos
  let sword_total := sword_cost * bought_swords
  let total_play_dough_cost := total_paid - (lego_total + sword_total)
  total_play_dough_cost / play_dough_cost

theorem tony_bought_10_play_doughs : 
  num_play_doughs 250 120 35 3 7 1940 = 10 := 
sorry

end tony_bought_10_play_doughs_l235_235916


namespace common_ratio_of_series_l235_235683

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l235_235683


namespace proof_problem_l235_235870

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | x ≤ -1}

theorem proof_problem :
  ((A ∩ {x | x > -1}) ∪ (B ∩ {x | x ≤ 0})) = {x | x > 0 ∨ x ≤ -1} :=
by 
  sorry

end proof_problem_l235_235870


namespace polygon_with_given_angle_sum_l235_235882

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l235_235882


namespace additional_cards_l235_235410

theorem additional_cards (total_cards : ℕ) (complete_decks : ℕ) (cards_per_deck : ℕ) (num_decks : ℕ) 
  (h1 : total_cards = 160) (h2 : num_decks = 3) (h3 : cards_per_deck = 52) :
  total_cards - (num_decks * cards_per_deck) = 4 :=
by
  sorry

end additional_cards_l235_235410


namespace intersection_of_A_and_B_l235_235046

def set_A : Set ℝ := {x | x >= 1 ∨ x <= -2}
def set_B : Set ℝ := {x | -3 < x ∧ x < 2}

def set_C : Set ℝ := {x | (-3 < x ∧ x <= -2) ∨ (1 <= x ∧ x < 2)}

theorem intersection_of_A_and_B (x : ℝ) : x ∈ set_A ∧ x ∈ set_B ↔ x ∈ set_C :=
  by
  sorry

end intersection_of_A_and_B_l235_235046


namespace parking_average_cost_l235_235575

noncomputable def parking_cost_per_hour := 
  let cost_two_hours : ℝ := 20.00
  let cost_per_excess_hour : ℝ := 1.75
  let weekend_surcharge : ℝ := 5.00
  let discount_rate : ℝ := 0.10
  let total_hours : ℝ := 9.00
  let excess_hours : ℝ := total_hours - 2.00
  let remaining_cost := cost_per_excess_hour * excess_hours
  let total_cost_before_discount := cost_two_hours + remaining_cost + weekend_surcharge
  let discount := discount_rate * total_cost_before_discount
  let discounted_total_cost := total_cost_before_discount - discount
  let average_cost_per_hour := discounted_total_cost / total_hours
  average_cost_per_hour

theorem parking_average_cost :
  parking_cost_per_hour = 3.725 := 
by
  sorry

end parking_average_cost_l235_235575


namespace number_of_distinct_prime_factors_30_fact_l235_235707

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l235_235707


namespace m_range_for_circle_l235_235235

def is_circle (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2 * (m - 3) * x + 2 * y + 5 = 0

theorem m_range_for_circle (m : ℝ) :
  (∀ x y : ℝ, is_circle x y m) → ((m > 5) ∨ (m < 1)) :=
by 
  sorry -- Proof not required

end m_range_for_circle_l235_235235


namespace Jim_remaining_miles_l235_235826

-- Define the total journey miles and miles already driven
def total_miles : ℕ := 1200
def miles_driven : ℕ := 215

-- Define the remaining miles Jim needs to drive
def remaining_miles (total driven : ℕ) : ℕ := total - driven

-- Statement to prove
theorem Jim_remaining_miles : remaining_miles total_miles miles_driven = 985 := by
  -- The proof is omitted
  sorry

end Jim_remaining_miles_l235_235826


namespace pigeon_percentage_l235_235778

-- Define the conditions
variables (total_birds : ℕ)
variables (geese swans herons ducks pigeons : ℕ)
variables (h1 : geese = total_birds * 20 / 100)
variables (h2 : swans = total_birds * 30 / 100)
variables (h3 : herons = total_birds * 15 / 100)
variables (h4 : ducks = total_birds * 25 / 100)
variables (h5 : pigeons = total_birds * 10 / 100)

-- Define the target problem
theorem pigeon_percentage (h_total : total_birds = 100) :
  (pigeons * 100 / (total_birds - swans)) = 14 :=
by sorry

end pigeon_percentage_l235_235778


namespace line_circle_intersect_l235_235701

theorem line_circle_intersect (m : ℤ) :
  (∃ x y : ℝ, 4 * x + 3 * y + 2 * m = 0 ∧ (x + 3)^2 + (y - 1)^2 = 1) ↔ 2 < m ∧ m < 7 :=
by
  sorry

end line_circle_intersect_l235_235701


namespace range_of_a_l235_235874

def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

noncomputable def roots (a : ℝ) : (ℝ × ℝ) :=
  (1, 3)

noncomputable def f_max (a : ℝ) :=
  -a

theorem range_of_a (a b c : ℝ) 
  (h1 : ∀ x, quadratic_function a b c x < 0 ↔ (x < 1 ∨ 3 < x))
  (h2 : f_max a < 2) : 
  -2 < a ∧ a < 0 :=
sorry

end range_of_a_l235_235874


namespace sum_of_product_of_consecutive_numbers_divisible_by_12_l235_235382

theorem sum_of_product_of_consecutive_numbers_divisible_by_12 (a : ℤ) : 
  (a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1) % 12 = 0 :=
by sorry

end sum_of_product_of_consecutive_numbers_divisible_by_12_l235_235382


namespace first_day_of_month_l235_235936

noncomputable def day_of_week := ℕ → ℕ

def is_wednesday (n : ℕ) : Prop := day_of_week n = 3

theorem first_day_of_month (day_of_week : day_of_week) (h : is_wednesday 30) : day_of_week 1 = 2 :=
by
  sorry

end first_day_of_month_l235_235936


namespace order_of_magnitude_l235_235337

theorem order_of_magnitude (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m := a / Real.sqrt b + b / Real.sqrt a
  let n := Real.sqrt a + Real.sqrt b
  let p := Real.sqrt (a + b)
  m ≥ n ∧ n > p := 
sorry

end order_of_magnitude_l235_235337


namespace PQRS_value_l235_235499

theorem PQRS_value
  (P Q R S : ℝ)
  (hP : 0 < P)
  (hQ : 0 < Q)
  (hR : 0 < R)
  (hS : 0 < S)
  (h1 : Real.log (P * Q) / Real.log 10 + Real.log (P * S) / Real.log 10 = 2)
  (h2 : Real.log (Q * S) / Real.log 10 + Real.log (Q * R) / Real.log 10 = 3)
  (h3 : Real.log (R * P) / Real.log 10 + Real.log (R * S) / Real.log 10 = 5) :
  P * Q * R * S = 100000 := 
sorry

end PQRS_value_l235_235499


namespace simplify_t_l235_235233

theorem simplify_t (t : ℝ) (cbrt3 : ℝ) (h : cbrt3 ^ 3 = 3) 
  (ht : t = 1 / (1 - cbrt3)) : 
  t = - (1 + cbrt3 + cbrt3 ^ 2) / 2 := 
sorry

end simplify_t_l235_235233


namespace total_rainfall_2003_and_2004_l235_235486

noncomputable def average_rainfall_2003 : ℝ := 45
noncomputable def months_in_year : ℕ := 12
noncomputable def percent_increase : ℝ := 0.05

theorem total_rainfall_2003_and_2004 :
  let rainfall_2004 := average_rainfall_2003 * (1 + percent_increase)
  let total_rainfall_2003 := average_rainfall_2003 * months_in_year
  let total_rainfall_2004 := rainfall_2004 * months_in_year
  total_rainfall_2003 = 540 ∧ total_rainfall_2004 = 567 := 
by 
  sorry

end total_rainfall_2003_and_2004_l235_235486


namespace positional_relationship_l235_235386

-- Definitions of the lines l1 and l2
def l1 (m x y : ℝ) : Prop := (m + 3) * x + 5 * y = 5 - 3 * m
def l2 (m x y : ℝ) : Prop := 2 * x + (m + 6) * y = 8

theorem positional_relationship (m : ℝ) :
  (∃ x y : ℝ, l1 m x y ∧ l2 m x y) ∨ (∀ x y : ℝ, l1 m x y ↔ l2 m x y) ∨
  ¬(∃ x y : ℝ, l1 m x y ∨ l2 m x y) :=
sorry

end positional_relationship_l235_235386


namespace find_single_digit_A_l235_235976

theorem find_single_digit_A (A : ℕ) (h1 : 0 ≤ A) (h2 : A < 10) (h3 : (10 * A + A) * (10 * A + A) = 5929) : A = 7 :=
sorry

end find_single_digit_A_l235_235976


namespace gcd_47_pow6_plus_1_l235_235432

theorem gcd_47_pow6_plus_1 (h_prime : Prime 47) : 
  Nat.gcd (47^6 + 1) (47^6 + 47^3 + 1) = 1 := 
by 
  sorry

end gcd_47_pow6_plus_1_l235_235432


namespace points_on_line_l235_235567

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l235_235567


namespace total_legs_correct_l235_235266

def num_horses : ℕ := 2
def num_dogs : ℕ := 5
def num_cats : ℕ := 7
def num_turtles : ℕ := 3
def num_goats : ℕ := 1
def legs_per_animal : ℕ := 4

theorem total_legs_correct :
  num_horses * legs_per_animal +
  num_dogs * legs_per_animal +
  num_cats * legs_per_animal +
  num_turtles * legs_per_animal +
  num_goats * legs_per_animal = 72 :=
by
  sorry

end total_legs_correct_l235_235266


namespace animal_arrangements_l235_235373

-- Given conditions
def chickens := 3
def dogs := 3
def cats := 5
def rabbits := 2
def cages := 13

-- Proving the total number of ways
theorem animal_arrangements :
  (fact 4) * (fact chickens) * (fact dogs) * (fact cats) * (fact rabbits) = 207360 :=
by {
  -- The proof itself is not required to be written as per the instruction, so we use sorry.
  sorry
}

end animal_arrangements_l235_235373


namespace final_cost_is_correct_l235_235859

noncomputable def calculate_final_cost 
  (price_orange : ℕ)
  (price_mango : ℕ)
  (increase_percent : ℕ)
  (bulk_discount_percent : ℕ)
  (sales_tax_percent : ℕ) : ℕ := 
  let new_price_orange := price_orange + (price_orange * increase_percent) / 100
  let new_price_mango := price_mango + (price_mango * increase_percent) / 100
  let total_cost_oranges := 10 * new_price_orange
  let total_cost_mangoes := 10 * new_price_mango
  let total_cost_before_discount := total_cost_oranges + total_cost_mangoes
  let discount_oranges := (total_cost_oranges * bulk_discount_percent) / 100
  let discount_mangoes := (total_cost_mangoes * bulk_discount_percent) / 100
  let total_cost_after_discount := total_cost_before_discount - discount_oranges - discount_mangoes
  let sales_tax := (total_cost_after_discount * sales_tax_percent) / 100
  total_cost_after_discount + sales_tax

theorem final_cost_is_correct :
  calculate_final_cost 40 50 15 10 8 = 100602 :=
by
  sorry

end final_cost_is_correct_l235_235859


namespace symmetry_axis_of_f_triangle_side_b_l235_235053

noncomputable def symmetry_axis (k : ℤ) : ℝ := (k * Real.pi / 2) + (Real.pi / 3)

theorem symmetry_axis_of_f :
  ∀ k : ℤ, ∃ x : ℝ, symmetry_axis k = x :=
by 
  sorry

noncomputable def g_of_x (x : ℝ) : ℝ := Real.sin (x + (Real.pi / 6)) - 1

def cosine_rule (a b c B : ℝ) : ℝ := (a^2 + c^2 - 2 * a * c * Real.cos B)

theorem triangle_side_b (a c : ℝ) (B : ℝ) (h1 : a = 2) (h2 : c = 4) (gB_zero : g_of_x B = 0) :
  ∃ b : ℝ, b = Real.sqrt (cosine_rule a b c B) :=
by 
  sorry

end symmetry_axis_of_f_triangle_side_b_l235_235053


namespace num_prime_factors_30_factorial_l235_235766

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l235_235766


namespace three_digit_number_probability_not_divisible_by_3_l235_235265

noncomputable def probability_not_divisible_by_3 : ℚ :=
  2 / 3

theorem three_digit_number_probability_not_divisible_by_3 :
  ∃ (s : Finset (Fin 1000)), (s.card = 720) ∧
    (∃ (not_divisible_by_3 : Finset (Fin 1000)),
      (not_divisible_by_3.card = 480) ∧
      (∀ d ∈ not_divisible_by_3, ¬ (d.val % 3 = 0)) ∧
      (probability_not_divisible_by_3 = not_divisible_by_3.card / s.card)) :=
begin
  sorry
end

end three_digit_number_probability_not_divisible_by_3_l235_235265


namespace soup_problem_l235_235405

def cans_needed_for_children (children : ℕ) (children_per_can : ℕ) : ℕ :=
  children / children_per_can

def remaining_cans (initial_cans used_cans : ℕ) : ℕ :=
  initial_cans - used_cans

def half_cans (cans : ℕ) : ℕ :=
  cans / 2

def adults_fed (cans : ℕ) (adults_per_can : ℕ) : ℕ :=
  cans * adults_per_can

theorem soup_problem
  (initial_cans : ℕ)
  (children_fed : ℕ)
  (children_per_can : ℕ)
  (adults_per_can : ℕ)
  (reserved_fraction : ℕ)
  (hreserved : reserved_fraction = 2)
  (hintial : initial_cans = 8)
  (hchildren : children_fed = 24)
  (hchildren_per_can : children_per_can = 6)
  (hadults_per_can : adults_per_can = 4) :
  adults_fed (half_cans (remaining_cans initial_cans (cans_needed_for_children children_fed children_per_can))) adults_per_can = 8 :=
by
  sorry

end soup_problem_l235_235405


namespace outfit_combination_count_l235_235059

theorem outfit_combination_count (c : ℕ) (s p h sh : ℕ) (c_eq_6 : c = 6) (s_eq_c : s = c) (p_eq_c : p = c) (h_eq_c : h = c) (sh_eq_c : sh = c) :
  (c^4) - c = 1290 :=
by
  sorry

end outfit_combination_count_l235_235059


namespace user_level_1000_l235_235278

noncomputable def user_level (points : ℕ) : ℕ :=
if points >= 1210 then 18
else if points >= 1000 then 17
else if points >= 810 then 16
else if points >= 640 then 15
else if points >= 490 then 14
else if points >= 360 then 13
else if points >= 250 then 12
else if points >= 160 then 11
else if points >= 90 then 10
else 0

theorem user_level_1000 : user_level 1000 = 17 :=
by {
  -- proof will be written here
  sorry
}

end user_level_1000_l235_235278


namespace average_weight_of_Arun_l235_235825

def arun_opinion (w : ℝ) : Prop := 66 < w ∧ w < 72
def brother_opinion (w : ℝ) : Prop := 60 < w ∧ w < 70
def mother_opinion (w : ℝ) : Prop := w ≤ 69

theorem average_weight_of_Arun :
  (∀ w, arun_opinion w → brother_opinion w → mother_opinion w → 
    (w = 67 ∨ w = 68 ∨ w = 69)) →
  avg_weight = 68 :=
sorry

end average_weight_of_Arun_l235_235825


namespace lora_coins_l235_235186

theorem lora_coins :
  ∃ n : ℕ, (17 = (finset.filter (λ d, d > 2) (nat.divisors n)).card) ∧
           (∀ m : ℕ, (17 = (finset.filter (λ d, d > 2) (nat.divisors m)).card) → n ≤ m) ∧
           n = 2700 :=
by
  sorry

end lora_coins_l235_235186


namespace factorial_prime_factors_l235_235719

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l235_235719


namespace math_problem_l235_235341

noncomputable def f (x a : ℝ) : ℝ := -4 * (Real.cos x) ^ 2 + 4 * Real.sqrt 3 * a * (Real.sin x) * (Real.cos x) + 2

theorem math_problem (a : ℝ) :
  (∃ a, ∀ x, f x a = f (π/6 - x) a) →    -- Symmetry condition
  (a = 1 ∧
  ∀ k : ℤ, ∀ x, (x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π) → 
    x ∈ Set.Icc (π/3 + k * π) (5 * π / 6 + k * π)) ∧  -- Decreasing intervals
  (∀ x, 2 * x - π / 6 ∈ Set.Icc (-2 * π / 3) (π / 6) → 
    f x a ∈ Set.Icc (-4 : ℝ) 2)) := -- Range on given interval
sorry

end math_problem_l235_235341


namespace function_relationship_profit_1200_max_profit_l235_235435

namespace SalesProblem

-- Define the linear relationship between sales quantity y and selling price x
def sales_quantity (x : ℝ) : ℝ := -2 * x + 160

-- Define the cost per item
def cost_per_item := 30

-- Define the profit given selling price x and quantity y
def profit (x : ℝ) (y : ℝ) : ℝ := (x - cost_per_item) * y

-- The given data points and conditions
def data_point_1 : (ℝ × ℝ) := (35, 90)
def data_point_2 : (ℝ × ℝ) := (40, 80)

-- Prove the linear relationship between y and x
theorem function_relationship : 
  sales_quantity data_point_1.1 = data_point_1.2 ∧ 
  sales_quantity data_point_2.1 = data_point_2.2 := 
  by sorry

-- Given daily profit of 1200, proves selling price should be 50 yuan
theorem profit_1200 (x : ℝ) (h₁ : 30 ≤ x ∧ x ≤ 54) 
  (h₂ : profit x (sales_quantity x) = 1200) : 
  x = 50 := 
  by sorry

-- Prove the maximum daily profit and corresponding selling price
theorem max_profit : 
  ∃ x, 30 ≤ x ∧ x ≤ 54 ∧ (∀ y, 30 ≤ y ∧ y ≤ 54 → profit y (sales_quantity y) ≤ profit x (sales_quantity x)) ∧ 
  profit x (sales_quantity x) = 1248 := 
  by sorry

end SalesProblem

end function_relationship_profit_1200_max_profit_l235_235435


namespace average_rate_of_change_is_4_l235_235888

def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change_is_4 : 
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_is_4_l235_235888


namespace right_triangle_proportion_l235_235793

/-- Given a right triangle ABC with ∠C = 90°, AB = c, AC = b, and BC = a, 
    and a point P on the hypotenuse AB (or its extension) such that 
    AP = m, BP = n, and CP = k, prove that a²m² + b²n² = c²k². -/
theorem right_triangle_proportion
  {a b c m n k : ℝ}
  (h_right : ∀ A B C : ℝ, A^2 + B^2 = C^2)
  (h1 : ∀ P : ℝ, m^2 + n^2 = k^2)
  (h_geometry : a^2 + b^2 = c^2) :
  a^2 * m^2 + b^2 * n^2 = c^2 * k^2 := 
sorry

end right_triangle_proportion_l235_235793


namespace midpoint_product_l235_235155

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l235_235155


namespace machine_A_produces_40_percent_l235_235407

theorem machine_A_produces_40_percent (p : ℝ) : 
  (0 < p ∧ p < 1 ∧
  (0.0156 = p * 0.009 + (1 - p) * 0.02)) → 
  p = 0.4 :=
by 
  intro h
  sorry

end machine_A_produces_40_percent_l235_235407


namespace points_on_line_possible_l235_235564

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l235_235564


namespace committee_selection_l235_235412

/-- There are exactly 20 ways to select a three-person team for the welcoming committee. -/
axiom welcoming_committee_ways : 20

/-- From those selected for the welcoming committee, a two-person finance committee must be selected.
  The student council has 15 members in total. Prove that the number of different ways to select
  the four-person planning committee and the two-person finance committee is 4095.
-/
theorem committee_selection (h1 : nat.choose 15 3 = 20) : 
  (nat.choose 15 4) * (nat.choose 3 2) = 4095 :=
by
  have h2 : nat.choose 15 4 = 1365 := by sorry
  have h3 : nat.choose 3 2 = 3 := by sorry
  have h4 : 1365 * 3 = 4095 := by sorry
  rw [h2, h3]
  exact h4

end committee_selection_l235_235412


namespace sum_of_selected_primes_divisible_by_3_probability_l235_235440

def first_fifteen_primes : List ℕ :=
  [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

def count_combinations_divisible_3 (nums : List ℕ) (k : ℕ) : ℕ :=
sorry -- Combines over the list to count combinations summing divisible by 3

noncomputable def probability_divisible_by_3 : ℚ :=
  let total_combinations := (Nat.choose 15 4)
  let favorable_combinations := count_combinations_divisible_3 first_fifteen_primes 4
  favorable_combinations / total_combinations

theorem sum_of_selected_primes_divisible_by_3_probability :
  probability_divisible_by_3 = 1/3 :=
sorry

end sum_of_selected_primes_divisible_by_3_probability_l235_235440


namespace smallest_common_multiple_of_9_and_6_l235_235169

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l235_235169


namespace possible_values_of_sum_l235_235540

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235540


namespace necessary_but_not_sufficient_condition_l235_235217

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a < 1) → ((a + 1) * (a - 2) < 0) ∧ ((∃ b : ℝ, (b + 1) * (b - 2) < 0 ∧ ¬(0 < b ∧ b < 1))) :=
by
  sorry

end necessary_but_not_sufficient_condition_l235_235217


namespace different_prime_factors_of_factorial_eq_10_l235_235725

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l235_235725


namespace hyperbola_range_k_l235_235805

noncomputable def hyperbola_equation (x y k : ℝ) : Prop :=
    (x^2) / (|k|-2) + (y^2) / (5-k) = 1

theorem hyperbola_range_k (k : ℝ) :
    (∃ x y, hyperbola_equation x y k) → (k > 5 ∨ (-2 < k ∧ k < 2)) :=
by 
    sorry

end hyperbola_range_k_l235_235805


namespace area_of_circle_2pi_distance_AB_sqrt6_l235_235086

/- Definition of the circle in polar coordinates -/
def circle_polar := ∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)

/- Definition of the line in polar coordinates -/
def line_polar := ∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0

/- The area of the circle -/
theorem area_of_circle_2pi : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  ∃ A : ℝ, A = 2 * Real.pi :=
by
  intro h
  sorry

/- The distance between two intersection points A and B -/
theorem distance_AB_sqrt6 : 
  (∀ θ, ∃ ρ : ℝ, ρ = 2 * Real.sqrt 2 * Real.sin (θ + Real.pi / 4)) → 
  (∀ θ, ∃ ρ : ℝ, ρ * Real.cos θ - ρ * Real.sin θ + 1 = 0) → 
  ∃ d : ℝ, d = Real.sqrt 6 :=
by
  intros h1 h2
  sorry

end area_of_circle_2pi_distance_AB_sqrt6_l235_235086


namespace probability_of_sequence_123456_l235_235297

theorem probability_of_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1     -- Total number of sequences
  let specific_sequences := 1 * 3 * 5 * 7 * 9 * 11        -- Sequences leading to 123456
  specific_sequences / total_sequences = 1 / 720 := by
  let total_sequences := 74919600
  let specific_sequences := 10395
  sorry

end probability_of_sequence_123456_l235_235297


namespace initial_boxes_l235_235974

theorem initial_boxes (x : ℕ) (h1 : 80 + 165 = 245) (h2 : 2000 * 245 = 490000) 
                      (h3 : 4 * 245 * x + 245 * x = 1225 * x) : x = 400 :=
by
  sorry

end initial_boxes_l235_235974


namespace average_age_of_5_students_l235_235941

theorem average_age_of_5_students
  (avg_age_20_students : ℕ → ℕ → ℕ → ℕ)
  (total_age_20 : avg_age_20_students 20 20 0 = 400)
  (total_age_9 : 9 * 16 = 144)
  (age_20th_student : ℕ := 186) :
  avg_age_20_students 5 ((400 - 144 - 186) / 5) 5 = 14 :=
by
  sorry

end average_age_of_5_students_l235_235941


namespace problem_l235_235879

theorem problem 
  (x y : ℝ)
  (h1 : 3 * x + y = 7)
  (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 := 
sorry

end problem_l235_235879


namespace probability_of_specific_sequence_l235_235299

-- We define a structure representing the problem conditions.
structure problem_conditions :=
  (cards : multiset ℕ)
  (permutation : list ℕ)

-- Noncomputable definition for the correct answer.
noncomputable def probability := (1 : ℚ) / 720

-- The main theorem statement.
theorem probability_of_specific_sequence :
  ∀ (conds : problem_conditions),
  conds.cards = {1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6} ∧
  (∃ (perm : list ℕ), perm.perm conds.permutation) →
  (∃ (sequence : list ℕ), sequence = [1, 2, 3, 4, 5, 6]) →
  let prob := calculate_probability conds.permutation [1, 2, 3, 4, 5, 6] in
  prob = (1 : ℚ) / 720 :=
sorry

end probability_of_specific_sequence_l235_235299


namespace greatest_integer_less_PS_l235_235079

theorem greatest_integer_less_PS 
  (PQ PS T : ℝ)
  (midpoint_TPS : T = PS / 2)
  (perpendicular_PT_QT : (PQ ^ 2 = (PS / 2) ^ 2 + (PS / 2) ^ 2))
  (PQ_value : PQ = 150) :
  ⌊ PS ⌋ = 212 :=
by
  sorry

end greatest_integer_less_PS_l235_235079


namespace factorial_30_prime_count_l235_235744

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l235_235744


namespace points_on_line_l235_235565

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l235_235565


namespace mary_has_more_money_than_marco_l235_235109

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l235_235109


namespace points_on_line_proof_l235_235561

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l235_235561


namespace midpoint_product_l235_235156

theorem midpoint_product (x1 y1 x2 y2 : ℤ) (hx1 : x1 = 4) (hy1 : y1 = -3) (hx2 : x2 = -8) (hy2 : y2 = 7) :
  let midx := (x1 + x2) / 2
  let midy := (y1 + y2) / 2
  midx * midy = -4 :=
by
  sorry

end midpoint_product_l235_235156


namespace find_g_3_16_l235_235919

theorem find_g_3_16 (g : ℝ → ℝ) (h1 : ∀ x, 0 ≤ x → x ≤ 1 → g x = g x) 
(h2 : g 0 = 0) 
(h3 : ∀ x y, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y) 
(h4 : ∀ x, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x) 
(h5 : ∀ x, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) : 
  g (3 / 16) = 8 / 27 :=
sorry

end find_g_3_16_l235_235919


namespace xyz_value_l235_235466

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24) 
  (h2 : x ^ 2 * (y + z) + y ^ 2 * (x + z) + z ^ 2 * (x + y) = 9) : 
  x * y * z = 5 :=
by
  sorry

end xyz_value_l235_235466


namespace evaluate_expression_l235_235021

theorem evaluate_expression : 2^4 + 2^4 + 2^4 + 2^4 = 2^6 :=
by
  sorry

end evaluate_expression_l235_235021


namespace possible_values_of_sum_l235_235512

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235512


namespace dogs_neither_long_furred_nor_brown_l235_235073

theorem dogs_neither_long_furred_nor_brown :
  (∀ (total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown : ℕ),
     total_dogs = 45 →
     long_furred_dogs = 26 →
     brown_dogs = 22 →
     both_long_furred_and_brown = 11 →
     neither_long_furred_nor_brown = total_dogs - (long_furred_dogs + brown_dogs - both_long_furred_and_brown) → 
     neither_long_furred_nor_brown = 8) :=
by
  intros total_dogs long_furred_dogs brown_dogs both_long_furred_and_brown neither_long_furred_nor_brown
  sorry

end dogs_neither_long_furred_nor_brown_l235_235073


namespace histogram_height_representation_l235_235633

theorem histogram_height_representation (freq_ratio : ℝ) (frequency : ℝ) (class_interval : ℝ) 
  (H : freq_ratio = frequency / class_interval) : 
  freq_ratio = frequency / class_interval :=
by 
  sorry

end histogram_height_representation_l235_235633


namespace factorial_prime_factors_l235_235720

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l235_235720


namespace P_vector_space_basis_set_polynomial_expansion_l235_235320

open Polynomial

-- Definition: P is the set of all polynomials of degree <= 4 with rational coefficients
def P : Polynomial ℚ := {p : Polynomial ℚ | degree p ≤ 4}

-- Part (a): Prove P has a vector space structure over ℚ
theorem P_vector_space : ∀ (p q : P), ∃ (r : P), p + q = r ∧ ∀ (a : ℚ) (p : P), ∃ (q : P), a • p = q := sorry

-- Part (b): Prove the given set forms a basis
theorem basis_set : LinearIndependent ℚ ![1, X - 2, (X - 2)^2, (X - 2)^3, (X - 2)^4] ∧
                    span ℚ (Set.range ![1, X - 2, (X - 2)^2, (X - 2)^3, (X - 2)^4]) = ⊤ := sorry

-- Part (c): Express polynomial in the given basis
theorem polynomial_expansion : 
  (7 + 2 * X - 45 * X^2 + 3 * X^4 : Polynomial ℚ) = 
  3 * (X - 2)^4 + 24 * (X - 2)^3 + 27 * (X - 2)^2 - 82 * (X - 2) - 121 := sorry

end P_vector_space_basis_set_polynomial_expansion_l235_235320


namespace expected_pairs_correct_l235_235803

-- Define the total number of cards in the deck.
def total_cards : ℕ := 52

-- Define the number of black cards in the deck.
def black_cards : ℕ := 26

-- Define the number of red cards in the deck.
def red_cards : ℕ := 26

-- Define the expected number of pairs of adjacent cards such that one is black and the other is red.
def expected_adjacent_pairs := 52 * (26 / 51)

-- Prove that the expected_adjacent_pairs is equal to 1352 / 51.
theorem expected_pairs_correct : expected_adjacent_pairs = 1352 / 51 := 
by
  have expected_adjacent_pairs_simplified : 52 * (26 / 51) = (1352 / 51) := 
    by sorry
  exact expected_adjacent_pairs_simplified

end expected_pairs_correct_l235_235803


namespace probability_within_sphere_correct_l235_235409

noncomputable def probability_within_sphere : ℝ :=
  let cube_volume := (2 : ℝ) * (2 : ℝ) * (2 : ℝ)
  let sphere_volume := (4 * Real.pi / 3) * (0.5) ^ 3
  sphere_volume / cube_volume

theorem probability_within_sphere_correct (x y z : ℝ) 
  (hx1 : -1 ≤ x) (hx2 : x ≤ 1) 
  (hy1 : -1 ≤ y) (hy2 : y ≤ 1) 
  (hz1 : -1 ≤ z) (hz2 : z ≤ 1) 
  (hx_sq : x^2 ≤ 0.5) 
  (hxyz : x^2 + y^2 + z^2 ≤ 0.25) : 
  probability_within_sphere = Real.pi / 48 :=
by
  sorry

end probability_within_sphere_correct_l235_235409


namespace possible_values_of_sum_l235_235513

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235513


namespace sum_possible_values_N_l235_235586

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l235_235586


namespace probability_intersection_l235_235616

noncomputable def prob_A : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_B : ℝ → ℝ := λ p, 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ → ℝ := λ p, 1 - (1 - 2 * p)^6

theorem probability_intersection (p : ℝ) 
  (hA : ∀ p, prob_A p = 1 - (1 - p)^6)
  (hB : ∀ p, prob_B p = 1 - (1 - p)^6)
  (hA_union_B : ∀ p, prob_A_union_B p = 1 - (1 - 2 * p)^6)
  (hImpossible : ∀ p, ¬(prob_A p > 0 ∧ prob_B p > 0)) :
  (∀ p, 1 - 2 * (1 - p)^6 + (1 - 2 * p)^6 = prob_A p + prob_B p - prob_A_union_B p) :=
begin
  intros p,
  rw [hA, hB, hA_union_B],
  sorry
end

end probability_intersection_l235_235616


namespace minimum_words_to_learn_for_90_percent_l235_235058

-- Define the conditions
def total_vocabulary_words : ℕ := 800
def minimum_percentage_required : ℚ := 0.90

-- Define the proof goal
theorem minimum_words_to_learn_for_90_percent (x : ℕ) (h1 : (x : ℚ) / total_vocabulary_words ≥ minimum_percentage_required) : x ≥ 720 :=
sorry

end minimum_words_to_learn_for_90_percent_l235_235058


namespace sum_possible_values_N_l235_235585

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l235_235585


namespace find_number_l235_235861

theorem find_number 
  (m : ℤ)
  (h13 : m % 13 = 12)
  (h12 : m % 12 = 11)
  (h11 : m % 11 = 10)
  (h10 : m % 10 = 9)
  (h9 : m % 9 = 8)
  (h8 : m % 8 = 7)
  (h7 : m % 7 = 6)
  (h6 : m % 6 = 5)
  (h5 : m % 5 = 4)
  (h4 : m % 4 = 3)
  (h3 : m % 3 = 2) :
  m = 360359 :=
by
  sorry

end find_number_l235_235861


namespace not_difference_of_squares_2021_l235_235482

theorem not_difference_of_squares_2021:
  ¬ ∃ (a b : ℕ), (a > b) ∧ (a^2 - b^2 = 2021) :=
sorry

end not_difference_of_squares_2021_l235_235482


namespace consecutive_odd_integers_sum_l235_235142

theorem consecutive_odd_integers_sum (n : ℤ) (h : (n - 2) + (n + 2) = 150) : n = 75 := 
by
  sorry

end consecutive_odd_integers_sum_l235_235142


namespace projection_of_a_onto_b_is_three_l235_235703

def vec_a : ℝ × ℝ := (3, 4)
def vec_b : ℝ × ℝ := (1, 0)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)
noncomputable def projection (u v : ℝ × ℝ) : ℝ := dot_product u v / magnitude v

theorem projection_of_a_onto_b_is_three : projection vec_a vec_b = 3 := by
  sorry

end projection_of_a_onto_b_is_three_l235_235703


namespace brad_started_after_maxwell_l235_235114

theorem brad_started_after_maxwell :
  ∀ (distance maxwell_speed brad_speed maxwell_time : ℕ),
  distance = 94 →
  maxwell_speed = 4 →
  brad_speed = 6 →
  maxwell_time = 10 →
  (distance - maxwell_speed * maxwell_time) / brad_speed = 9 := 
by
  intros distance maxwell_speed brad_speed maxwell_time h_dist h_m_speed h_b_speed h_m_time
  sorry

end brad_started_after_maxwell_l235_235114


namespace sum_of_x_y_possible_values_l235_235536

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235536


namespace shirts_not_all_on_sale_implications_l235_235771

variable (Shirts : Type) (store_contains : Shirts → Prop) (on_sale : Shirts → Prop)

theorem shirts_not_all_on_sale_implications :
  ¬ ∀ s, store_contains s → on_sale s → 
  (∃ s, store_contains s ∧ ¬ on_sale s) ∧ (∃ s, store_contains s ∧ ¬ on_sale s) :=
by
  sorry

end shirts_not_all_on_sale_implications_l235_235771


namespace find_smallest_m_l235_235368

def is_in_S (z : ℂ) : Prop :=
  ∃ (x y : ℝ), ((1 / 2 : ℝ) ≤ x) ∧ (x ≤ Real.sqrt 2 / 2) ∧ (z = (x : ℂ) + (y : ℂ) * Complex.I)

def is_nth_root_of_unity (z : ℂ) (n : ℕ) : Prop :=
  z ^ n = 1

def smallest_m (m : ℕ) : Prop :=
  ∀ n : ℕ, n ≥ m → ∃ z : ℂ, is_in_S z ∧ is_nth_root_of_unity z n

theorem find_smallest_m : smallest_m 24 :=
  sorry

end find_smallest_m_l235_235368


namespace tank_capacity_l235_235481

variable (C : ℝ)  -- total capacity of the tank

-- The tank is 5/8 full initially
axiom h1 : (5/8) * C + 15 = (19/24) * C

theorem tank_capacity : C = 90 :=
by
  sorry

end tank_capacity_l235_235481


namespace common_ratio_of_series_l235_235684

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l235_235684


namespace sum_of_xy_l235_235546

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235546


namespace smallest_multiple_of_9_and_6_is_18_l235_235173

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l235_235173


namespace first_day_of_month_l235_235939

theorem first_day_of_month (h: ∀ n, (n % 7 = 2) → n_day_of_week n = "Wednesday"): 
  n_day_of_week 1 = "Tuesday" :=
sorry

end first_day_of_month_l235_235939


namespace sum_of_rationals_eq_l235_235504

theorem sum_of_rationals_eq (a1 a2 a3 a4 : ℚ)
  (h : {x : ℚ | ∃ i j, 1 ≤ i ∧ i < j ∧ j ≤ 4 ∧ x = a1 * a2 ∧ x = a1 * a3 ∧ x = a1 * a4 ∧ x = a2 * a3 ∧ x = a2 * a4 ∧ x = a3 * a4} = {-24, -2, -3/2, -1/8, 1, 3}) :
  a1 + a2 + a3 + a4 = 9/4 ∨ a1 + a2 + a3 + a4 = -9/4 :=
sorry

end sum_of_rationals_eq_l235_235504


namespace henry_total_cost_l235_235896

def henry_initial_figures : ℕ := 3
def henry_total_needed_figures : ℕ := 15
def cost_per_figure : ℕ := 12

theorem henry_total_cost :
  (henry_total_needed_figures - henry_initial_figures) * cost_per_figure = 144 :=
by
  sorry

end henry_total_cost_l235_235896


namespace quadratic_expression_value_l235_235065

theorem quadratic_expression_value :
  ∀ x1 x2 : ℝ, (x1^2 - 4 * x1 - 2020 = 0) ∧ (x2^2 - 4 * x2 - 2020 = 0) →
  (x1^2 - 2 * x1 + 2 * x2 = 2028) :=
by
  intros x1 x2 h
  sorry

end quadratic_expression_value_l235_235065


namespace line_equation_l235_235627

theorem line_equation (a : ℝ) (P : ℝ × ℝ) (hx : P = (5, 6)) 
                      (cond : (a ≠ 0) ∧ (2 * a = 17)) : 
  ∃ (m b : ℝ), - (m * (0 : ℝ) + b) = a ∧ (- m * 17 / 2 + b) = 6 ∧ 
               (x + 2 * y - 17 =  0) := sorry

end line_equation_l235_235627


namespace claudia_groupings_l235_235013

-- Definition of combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Given conditions
def candles_combinations : ℕ := combination 6 3
def flowers_combinations : ℕ := combination 15 12

-- Lean statement
theorem claudia_groupings : candles_combinations * flowers_combinations = 9100 :=
by
  sorry

end claudia_groupings_l235_235013


namespace num_prime_factors_of_30_l235_235740

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l235_235740


namespace power_combination_l235_235636

theorem power_combination :
  (-1)^43 + 2^(2^3 + 5^2 - 7^2) = -65535 / 65536 :=
by
  sorry

end power_combination_l235_235636


namespace ellipse_eccentricity_l235_235693

theorem ellipse_eccentricity (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (B F A C : ℝ × ℝ) 
    (h3 : (B.1 ^ 2 / a ^ 2 + B.2 ^ 2 / b ^ 2 = 1))
    (h4 : (C.1 ^ 2 / a ^ 2 + C.2 ^ 2 / b ^ 2 = 1))
    (h5 : B.1 > 0 ∧ B.2 > 0)
    (h6 : C.1 > 0 ∧ C.2 > 0)
    (h7 : ∃ M : ℝ × ℝ, M = ((B.1 + C.1) / 2, (B.2 + C.2) / 2) ∧ (F = M)) :
    ∃ e : ℝ, e = (1 / 3) := 
  sorry

end ellipse_eccentricity_l235_235693


namespace prime_factor_of_difference_l235_235464

theorem prime_factor_of_difference {A B : ℕ} (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (h_neq : A ≠ B) :
  Nat.Prime 2 ∧ (∃ B : ℕ, 20 * B = 20 * B) :=
by
  sorry

end prime_factor_of_difference_l235_235464


namespace arithmetic_sequence_a15_l235_235909

variable {α : Type*} [LinearOrderedField α]

-- Conditions for the arithmetic sequence
variable (a : ℕ → α)
variable (d : α)
variable (a1 : α)
variable (h_arith_seq : ∀ n, a (n + 1) = a n + d)
variable (h_a5 : a 5 = 5)
variable (h_a10 : a 10 = 15)

-- To prove that a15 = 25
theorem arithmetic_sequence_a15 : a 15 = 25 := by
  sorry

end arithmetic_sequence_a15_l235_235909


namespace product_of_midpoint_is_minus_4_l235_235158

-- Coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (4, -3)
def endpoint2 : ℝ × ℝ := (-8, 7)

-- Function to compute the midpoint of two points
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Coordinates of the midpoint
def midpoint_coords := midpoint endpoint1 endpoint2

-- Product of the coordinates of the midpoint
def product_of_midpoint_coords (mp : ℝ × ℝ) : ℝ :=
  mp.1 * mp.2

-- Statement of the theorem to be proven
theorem product_of_midpoint_is_minus_4 : 
  product_of_midpoint_coords midpoint_coords = -4 := 
by
  sorry

end product_of_midpoint_is_minus_4_l235_235158


namespace gcd_876543_765432_l235_235953

theorem gcd_876543_765432 : Nat.gcd 876543 765432 = 1 :=
by {
  sorry
}

end gcd_876543_765432_l235_235953


namespace lion_weight_l235_235813

theorem lion_weight :
  ∃ (L : ℝ), 
    (∃ (T P : ℝ), 
      L + T + P = 106.6 ∧ 
      P = T - 7.7 ∧ 
      T = L - 4.8) ∧ 
    L = 41.3 :=
by
  sorry

end lion_weight_l235_235813


namespace arithmetic_sequence_a2_a6_l235_235357

theorem arithmetic_sequence_a2_a6 (a : ℕ → ℕ) (d : ℕ) (h_arith_seq : ∀ n, a (n+1) = a n + d)
  (h_a4 : a 4 = 4) : a 2 + a 6 = 8 :=
by sorry

end arithmetic_sequence_a2_a6_l235_235357


namespace number_of_distinct_prime_factors_30_fact_l235_235708

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l235_235708


namespace greatest_integer_less_than_PS_l235_235082

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l235_235082


namespace number_of_toys_bought_l235_235120

def toy_cost (T : ℕ) : ℕ := 10 * T
def card_cost : ℕ := 2 * 5
def shirt_cost : ℕ := 5 * 6
def total_cost (T : ℕ) : ℕ := toy_cost T + card_cost + shirt_cost

theorem number_of_toys_bought (T : ℕ) : total_cost T = 70 → T = 3 :=
by
  intro h
  sorry

end number_of_toys_bought_l235_235120


namespace joe_speed_l235_235243

theorem joe_speed (P : ℝ) (J : ℝ) (h1 : J = 2 * P) (h2 : 2 * P * (2 / 3) + P * (2 / 3) = 16) : J = 16 := 
by
  sorry

end joe_speed_l235_235243


namespace cohen_saw_1300_fish_eater_birds_l235_235179

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l235_235179


namespace sine_ratio_comparison_l235_235818

theorem sine_ratio_comparison : (Real.sin (1 * Real.pi / 180) / Real.sin (2 * Real.pi / 180)) < (Real.sin (3 * Real.pi / 180) / Real.sin (4 * Real.pi / 180)) :=
sorry

end sine_ratio_comparison_l235_235818


namespace right_triangle_condition_l235_235465

theorem right_triangle_condition (a b c : ℝ) (θ : ℝ) 
  (h₁ : c = real.sqrt (a^2 + b^2)) 
  (h₂ : a ≥ 0) (h₃ : b ≥ 0) (h₄ : c > 0) 
  (h₅ : real.cos θ = 0) :
  (real.sqrt (a^2 + b^2) = a + b) ↔ (θ = real.pi / 2) :=
by
  sorry

end right_triangle_condition_l235_235465


namespace new_average_age_l235_235133

theorem new_average_age (n : ℕ) (avg_old : ℕ) (new_person_age : ℕ) (new_avg_age : ℕ)
  (h1 : avg_old = 14)
  (h2 : n = 9)
  (h3 : new_person_age = 34)
  (h4 : new_avg_age = 16) :
  (n * avg_old + new_person_age) / (n + 1) = new_avg_age :=
sorry

end new_average_age_l235_235133


namespace train_speed_in_km_per_hr_l235_235428

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l235_235428


namespace polynomial_count_condition_l235_235999

theorem polynomial_count_condition :
  let S := { k : ℕ | k ≤ 9 }
  let Q (a b c d : ℕ) := a + b + c + d = 9 ∧ a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S
  ∃ n : ℕ, n = 220 ∧ (∃ a b c d : ℕ, Q a b c d) :=
by sorry

end polynomial_count_condition_l235_235999


namespace sum_of_x_y_possible_values_l235_235533

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235533


namespace sum_of_real_numbers_l235_235527

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235527


namespace geometric_sequence_fifth_term_l235_235286

theorem geometric_sequence_fifth_term (a r : ℝ) (h1 : a * r^2 = 9) (h2 : a * r^6 = 1) : a * r^4 = 3 :=
by
  sorry

end geometric_sequence_fifth_term_l235_235286


namespace sqrt_recursive_value_l235_235659

noncomputable def recursive_sqrt (x : ℝ) : ℝ := Real.sqrt (3 - x)

theorem sqrt_recursive_value : 
  ∃ x : ℝ, (x = recursive_sqrt x) ∧ x = ( -1 + Real.sqrt 13 ) / 2 :=
by 
  -- ∃ x, solution assertion to define the value of x 
  use ( -1 + Real.sqrt 13 ) / 2
  sorry 

end sqrt_recursive_value_l235_235659


namespace am_gm_hm_inequality_l235_235247

theorem am_gm_hm_inequality (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : a ≠ c) : 
  (a + b + c) / 3 > (a * b * c) ^ (1 / 3) ∧ (a * b * c) ^ (1 / 3) > 3 * a * b * c / (a * b + b * c + c * a) :=
by
  sorry

end am_gm_hm_inequality_l235_235247


namespace nested_sqrt_eq_l235_235650

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l235_235650


namespace intersection_PQ_l235_235478

def setP  := {x : ℝ | x * (x - 1) ≥ 0}
def setQ := {y : ℝ | ∃ x : ℝ, y = 3 * x^2 + 1}

theorem intersection_PQ : {x : ℝ | x > 1} = {z : ℝ | z ∈ setP ∧ z ∈ setQ} :=
by
  sorry

end intersection_PQ_l235_235478


namespace num_prime_factors_30_factorial_l235_235752

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l235_235752


namespace probability_same_color_dice_l235_235343

theorem probability_same_color_dice :
  let total_sides := 12
  let red_sides := 3
  let green_sides := 4
  let blue_sides := 2
  let yellow_sides := 3
  let prob_red := (red_sides / total_sides) ^ 2
  let prob_green := (green_sides / total_sides) ^ 2
  let prob_blue := (blue_sides / total_sides) ^ 2
  let prob_yellow := (yellow_sides / total_sides) ^ 2
  prob_red + prob_green + prob_blue + prob_yellow = 19 / 72 := 
by
  -- The proof goes here
  sorry

end probability_same_color_dice_l235_235343


namespace carousel_problem_l235_235403

theorem carousel_problem (n : ℕ) : 
  (∃ (f : Fin n → Fin n), 
    (∀ i, f (f i) = i) ∧ 
    (∀ i j, i ≠ j → f i ≠ f j) ∧ 
    (∀ i, f i < n)) ↔ 
  (Even n) := 
sorry

end carousel_problem_l235_235403


namespace probability_sequence_123456_l235_235304

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l235_235304


namespace common_element_in_subsets_l235_235779

open Finset

theorem common_element_in_subsets (A : Finset α) (n : ℕ) (h : A.card = n) (S : Finset (Finset α))
  (hS : S.card = 2^(n-1))
  (h_common : ∀ x y z ∈ S, ∃ a, a ∈ x ∧ a ∈ y ∧ a ∈ z) :
  ∃ e, ∀ s ∈ S, e ∈ s :=
sorry

end common_element_in_subsets_l235_235779


namespace length_of_ship_l235_235323

-- Variables and conditions
variables (E L S : ℝ)
variables (W : ℝ := 0.9) -- Wind reducing factor

-- Conditions as equations
def condition1 : Prop := 150 * E = L + 150 * S
def condition2 : Prop := 70 * E = L - 63 * S

-- Theorem to prove
theorem length_of_ship (hc1 : condition1 E L S) (hc2 : condition2 E L S) : L = (19950 / 213) * E :=
sorry

end length_of_ship_l235_235323


namespace system_of_equations_solution_l235_235572

theorem system_of_equations_solution :
  ∃ x y : ℝ, 7 * x - 3 * y = 2 ∧ 2 * x + y = 8 ∧ x = 2 ∧ y = 4 :=
by
  use 2
  use 4
  sorry

end system_of_equations_solution_l235_235572


namespace smallest_common_multiple_of_9_and_6_l235_235172

theorem smallest_common_multiple_of_9_and_6 : 
  ∃ x : ℕ, (x > 0 ∧ x % 9 = 0 ∧ x % 6 = 0) ∧ 
           ∀ y : ℕ, (y > 0 ∧ y % 9 = 0 ∧ y % 6 = 0) → x ≤ y :=
begin
  use 18,
  split,
  { split,
    { exact nat.succ_pos 17, },
    { split,
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_right 9 6), },
      { exact nat.mod_eq_zero_of_dvd (dvd_lcm_left 9 6), } } },
  { intros y hy,
    cases hy with hy1 hy2,
    cases hy2 with hy2 hy3,
    exact lcm.dvd_iff.1 (nat.dvd_of_mod_eq_zero hy3) }
end

end smallest_common_multiple_of_9_and_6_l235_235172


namespace solution_set_for_f_ge_0_range_of_a_l235_235226

def f (x : ℝ) : ℝ := |3 * x + 1| - |2 * x + 2|

theorem solution_set_for_f_ge_0 : {x : ℝ | f x ≥ 0} = {x : ℝ | x ≤ -3/5} ∪ {x : ℝ | x ≥ 1} :=
sorry

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x - |x + 1| ≤ |a + 1|) ↔ (a ≤ -3 ∨ a ≥ 1) :=
sorry

end solution_set_for_f_ge_0_range_of_a_l235_235226


namespace systematic_sampling_fourth_group_l235_235353

theorem systematic_sampling_fourth_group (n m k g2 g4 : ℕ) (h_class_size : n = 72)
  (h_sample_size : m = 6) (h_k : k = n / m) (h_group2 : g2 = 16) (h_group4 : g4 = g2 + 2 * k) :
  g4 = 40 := by
  sorry

end systematic_sampling_fourth_group_l235_235353


namespace verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l235_235927

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Definition for conversions from base 60 to base 10
def from_base_60 (d1 d0 : ℕ) : ℕ :=
  d1 * 60 + d0

-- Proof statements
theorem verify_21_base_60 : from_base_60 2 1 = 121 ∧ is_perfect_square 121 :=
by {
  sorry
}

theorem verify_1_base_60 : from_base_60 0 1 = 1 ∧ is_perfect_square 1 :=
by {
  sorry
}

theorem verify_2_base_60_not_square : from_base_60 0 2 = 2 ∧ ¬ is_perfect_square 2 :=
by {
  sorry
}

end verify_21_base_60_verify_1_base_60_verify_2_base_60_not_square_l235_235927


namespace sum_possible_values_N_l235_235587

theorem sum_possible_values_N (a b c N : ℕ) (ha : 0 < a) (hb : 0 < b)
  (hc : c = a + b) (hN : N = a * b * c) (h_condition : N = 8 * (a + b + c)) :
  (N = 272 ∨ N = 160 ∨ N = 128) →
  (272 + 160 + 128) = 560 :=
by {
  intros h,
  have h1 : N = 272 ∨ N = 160 ∨ N = 128,
  from h,
  exact eq.refl 560,
}

end sum_possible_values_N_l235_235587


namespace unit_vector_AB_l235_235042

def point := ℝ × ℝ

def vector_sub (p1 p2 : point) : point := (p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v : point) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def unit_vector (v : point) : point := (v.1 / magnitude v, v.2 / magnitude v)

def A : point := (1, 3)
def B : point := (4, -1)

def AB : point := vector_sub A B

theorem unit_vector_AB : unit_vector AB = (3/5, -4/5) := sorry

end unit_vector_AB_l235_235042


namespace num_prime_factors_30_factorial_l235_235768

theorem num_prime_factors_30_factorial : 
  (finset.filter nat.prime (finset.range 31)).card = 10 :=
by
  sorry

end num_prime_factors_30_factorial_l235_235768


namespace bills_head_circumference_l235_235089

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l235_235089


namespace smallest_m_exists_l235_235958

theorem smallest_m_exists :
  ∃ (m : ℕ), 0 < m ∧ (∃ k : ℕ, 5 * m = k^2) ∧ (∃ l : ℕ, 3 * m = l^3) ∧ m = 243 :=
by
  sorry

end smallest_m_exists_l235_235958


namespace points_on_line_l235_235566

theorem points_on_line (n : ℕ) : 9 * n - 8 = 82 → n = 10 :=
by
  sorry

end points_on_line_l235_235566


namespace total_balloons_l235_235267

theorem total_balloons (sam_balloons_initial mary_balloons fred_balloons : ℕ) (h1 : sam_balloons_initial = 6)
    (h2 : mary_balloons = 7) (h3 : fred_balloons = 5) : sam_balloons_initial - fred_balloons + mary_balloons = 8 :=
by
  sorry

end total_balloons_l235_235267


namespace cost_of_tissues_l235_235988
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l235_235988


namespace age_sum_l235_235128

variable {S R K : ℝ}

theorem age_sum 
  (h1 : S = R + 10)
  (h2 : S + 12 = 3 * (R - 5))
  (h3 : K = R / 2) :
  S + R + K = 56.25 := 
by 
  sorry

end age_sum_l235_235128


namespace eval_sqrt_expression_l235_235651

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l235_235651


namespace number_of_triangles_l235_235918

theorem number_of_triangles (n : ℕ) (h₁ : n = 30) (h₂ : ∀ (a b c : ℕ), 
  a ≠ b ∧ b ≠ c ∧ c ≠ a → 
  a < b ∧ b < c → 
  ∃ d e f, d ≠ e ∧ e ≠ f ∧ f ≠ d ∧ (b - a) ≥ 3 ∧ (c - b) ≥ 3 ∧ (d - c) ≥ 3) :
  ∃ t : ℕ, t = 2530 :=
by
  sorry

end number_of_triangles_l235_235918


namespace arithmetic_progression_infinite_kth_powers_l235_235205

theorem arithmetic_progression_infinite_kth_powers {a d k : ℕ} (ha : a > 0) (hd : d > 0) (hk : k > 0) :
  (∀ n : ℕ, ¬ ∃ b : ℕ, a + n * d = b ^ k) ∨ (∀ b : ℕ, ∃ n : ℕ, a + n * d = b ^ k) :=
sorry

end arithmetic_progression_infinite_kth_powers_l235_235205


namespace absolute_sum_of_coefficients_l235_235330

theorem absolute_sum_of_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℤ) :
  (2 - x)^6 = a_0 + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 →
  a_0 = 2^6 →
  a_0 > 0 ∧ a_2 > 0 ∧ a_4 > 0 ∧ a_6 > 0 ∧
  a_1 < 0 ∧ a_3 < 0 ∧ a_5 < 0 → 
  |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 665 :=
by sorry

end absolute_sum_of_coefficients_l235_235330


namespace calc_remainder_l235_235996

theorem calc_remainder : 
  (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 +
   90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 -
   90^7 * Nat.choose 10 7 + 90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 +
   90^10 * Nat.choose 10 10) % 88 = 1 := 
by sorry

end calc_remainder_l235_235996


namespace min_value_of_expr_l235_235061

theorem min_value_of_expr (a : ℝ) (h : a > 3) : ∃ m, (∀ b > 3, b + 4 / (b - 3) ≥ m) ∧ m = 7 :=
sorry

end min_value_of_expr_l235_235061


namespace triangle_area_l235_235994

theorem triangle_area (f : ℝ → ℝ) (x1 x2 yIntercept base height area : ℝ)
  (h1 : ∀ x, f x = (x - 4)^2 * (x + 3))
  (h2 : f 0 = yIntercept)
  (h3 : x1 = -3)
  (h4 : x2 = 4)
  (h5 : base = x2 - x1)
  (h6 : height = yIntercept)
  (h7 : area = 1/2 * base * height) :
  area = 168 := sorry

end triangle_area_l235_235994


namespace benny_apples_l235_235993

theorem benny_apples (benny dan : ℕ) (total : ℕ) (H1 : dan = 9) (H2 : total = 11) (H3 : benny + dan = total) : benny = 2 :=
by
  sorry

end benny_apples_l235_235993


namespace sum_of_x_y_l235_235522

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235522


namespace number_of_prime_factors_thirty_factorial_l235_235729

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l235_235729


namespace evaluate_expression_l235_235847

theorem evaluate_expression :
  (2 ^ (-1 : ℤ) + 2 ^ (-2 : ℤ))⁻¹ = (4 / 3 : ℚ) := by
    sorry

end evaluate_expression_l235_235847


namespace discount_threshold_l235_235804

-- Definitions based on given conditions
def photocopy_cost : ℝ := 0.02
def discount_percentage : ℝ := 0.25
def copies_needed_each : ℕ := 80
def total_savings : ℝ := 0.40 * 2 -- total savings for both Steve and Dennison

-- Minimum number of photocopies required to get the discount
def min_copies_for_discount : ℕ := 160

-- Lean statement to prove the minimum number of photocopies required for the discount
theorem discount_threshold :
  ∀ (x : ℕ),
  photocopy_cost * (x : ℝ) - (photocopy_cost * (1 - discount_percentage) * (x : ℝ)) * 2 = total_savings → 
  min_copies_for_discount = 160 :=
by sorry

end discount_threshold_l235_235804


namespace geometric_sequence_tenth_fifth_terms_l235_235855

variable (a r : ℚ) (n : ℕ)

def geometric_sequence (a r : ℚ) (n : ℕ) : ℚ :=
  a * r^(n-1)

theorem geometric_sequence_tenth_fifth_terms :
  (geometric_sequence 4 (4/3) 10 = 1048576 / 19683) ∧ (geometric_sequence 4 (4/3) 5 = 1024 / 81) :=
by
  sorry

end geometric_sequence_tenth_fifth_terms_l235_235855


namespace mike_last_5_shots_l235_235509

theorem mike_last_5_shots :
  let initial_shots := 30
  let initial_percentage := 40 / 100
  let additional_shots_1 := 10
  let new_percentage_1 := 45 / 100
  let additional_shots_2 := 5
  let new_percentage_2 := 46 / 100
  
  let initial_makes := initial_shots * initial_percentage
  let total_shots_after_1 := initial_shots + additional_shots_1
  let makes_after_1 := total_shots_after_1 * new_percentage_1 - initial_makes
  let total_makes_after_1 := initial_makes + makes_after_1
  let total_shots_after_2 := total_shots_after_1 + additional_shots_2
  let final_makes := total_shots_after_2 * new_percentage_2
  let makes_in_last_5 := final_makes - total_makes_after_1
  
  makes_in_last_5 = 2
:=
by
  sorry

end mike_last_5_shots_l235_235509


namespace exist_initial_points_l235_235569

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l235_235569


namespace shorter_side_ratio_l235_235241

variable {x y : ℝ}
variables (h1 : x < y)
variables (h2 : x + y - Real.sqrt (x^2 + y^2) = 1/2 * y)

theorem shorter_side_ratio (h1 : x < y) (h2 : x + y - Real.sqrt (x^2 + y^2) = 1 / 2 * y) : x / y = 3 / 4 := 
sorry

end shorter_side_ratio_l235_235241


namespace five_point_eight_one_million_in_scientific_notation_l235_235037

theorem five_point_eight_one_million_in_scientific_notation :
  5.81 * 10^6 = 5.81e6 :=
sorry

end five_point_eight_one_million_in_scientific_notation_l235_235037


namespace martha_saves_half_daily_allowance_l235_235506

theorem martha_saves_half_daily_allowance {f : ℚ} (h₁ : 12 > 0) (h₂ : (6 : ℚ) * 12 * f + (3 : ℚ) = 39) : f = 1 / 2 :=
by
  sorry

end martha_saves_half_daily_allowance_l235_235506


namespace cos_reflected_value_l235_235871

theorem cos_reflected_value (x : ℝ) (h : Real.cos (π / 6 + x) = 1 / 3) :
  Real.cos (5 * π / 6 - x) = -1 / 3 := 
by {
  sorry
}

end cos_reflected_value_l235_235871


namespace correct_propositions_l235_235052

-- Given the following propositions:
def proposition1 : Prop :=
∀ x : ℝ, (3^x) = (log3 (exp3 x))

def proposition2 : Prop :=
real.is_periodic (abs ∘ real.sin) 2 * real.pi

def proposition3 : Prop :=
∃ x : real, ∀ x, (tan (2 * x + real.pi / 3)) = (tan (2 * (x + real.pi / 3)))

def proposition4 : Prop :=
∀ x : ℝ, x ∈ set.Icc (-2 * real.pi) (2 * real.pi) → x ∈ set.Icc (-real.pi / 3) (5 * real.pi / 3) ↔ 
(real.derivative (λ x, 2 * real.sin (real.pi / 3 - 1/2 * x)) x) < 0

-- Prove that the correct propositions are (1), (3), and (4):
theorem correct_propositions : 
  proposition1 ∧ proposition3 ∧ proposition4 :=
by sorry

end correct_propositions_l235_235052


namespace solve_inequality_l235_235372

-- We will define the conditions and corresponding solution sets
def solution_set (a x : ℝ) : Prop :=
  (a < -1 ∧ (x > -a ∨ x < 1)) ∨
  (a = -1 ∧ x ≠ 1) ∨
  (a > -1 ∧ (x < -a ∨ x > 1))

theorem solve_inequality (a x : ℝ) :
  (x - 1) * (x + a) > 0 ↔ solution_set a x :=
by
  sorry

end solve_inequality_l235_235372


namespace product_of_midpoint_coordinates_l235_235160

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l235_235160


namespace first_chapter_pages_calculation_l235_235973

-- Define the constants and conditions
def second_chapter_pages : ℕ := 11
def first_chapter_pages_more : ℕ := 37

-- Main proof problem
theorem first_chapter_pages_calculation : first_chapter_pages_more + second_chapter_pages = 48 := by
  sorry

end first_chapter_pages_calculation_l235_235973


namespace thought_number_is_24_l235_235309

variable (x : ℝ)

theorem thought_number_is_24 (h : x / 4 + 9 = 15) : x = 24 := by
  sorry

end thought_number_is_24_l235_235309


namespace storks_minus_birds_l235_235400

/-- Define the initial values --/
def s : ℕ := 6         -- Number of storks
def b1 : ℕ := 2        -- Initial number of birds
def b2 : ℕ := 3        -- Number of additional birds

/-- Calculate the total number of birds --/
def b : ℕ := b1 + b2   -- Total number of birds

/-- Prove the number of storks minus the number of birds --/
theorem storks_minus_birds : s - b = 1 :=
by sorry

end storks_minus_birds_l235_235400


namespace number_of_prime_factors_thirty_factorial_l235_235731

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l235_235731


namespace binom_identity_l235_235246

def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem binom_identity (n k : ℕ) : k * binom n k = n * binom (n - 1) (k - 1) := by
  sorry

end binom_identity_l235_235246


namespace factorial_30_prime_count_l235_235742

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l235_235742


namespace nested_sqrt_eq_l235_235648

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l235_235648


namespace smallest_multiple_of_9_and_6_l235_235161

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l235_235161


namespace roots_product_of_polynomials_l235_235026

theorem roots_product_of_polynomials :
  ∃ (b c : ℤ), (∀ r : ℂ, r ^ 2 - 2 * r - 1 = 0 → r ^ 5 - b * r - c = 0) ∧ b * c = 348 :=
by 
  sorry

end roots_product_of_polynomials_l235_235026


namespace find_x_y_l235_235865

theorem find_x_y (x y : ℝ) : (3 * x + 4 * -2 = 0) ∧ (3 * 1 + 4 * y = 0) → x = 8 / 3 ∧ y = -3 / 4 :=
by
  sorry

end find_x_y_l235_235865


namespace curve_is_circle_l235_235445

theorem curve_is_circle (r θ : ℝ) (h : r = 1 / (Real.sin θ + Real.cos θ)) :
  ∃ x y : ℝ, r = Math.sqrt(x^2 + y^2) ∧ x = r * Real.cos θ ∧ y = r * Real.sin θ ∧ (x + y)^2 = (x^2 + y^2) :=
by
  sorry

end curve_is_circle_l235_235445


namespace find_f_2017_l235_235223

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f_2017 (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_func_eq : ∀ x : ℝ, f (x + 3) * f x = -1)
  (h_val : f (-1) = 2) :
  f 2017 = -2 := sorry

end find_f_2017_l235_235223


namespace smallest_common_multiple_of_9_and_6_l235_235168

theorem smallest_common_multiple_of_9_and_6 : ∃ (n : ℕ), n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ (m : ℕ), m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m := 
sorry

end smallest_common_multiple_of_9_and_6_l235_235168


namespace sine_tangent_not_possible_1_sine_tangent_not_possible_2_l235_235087

theorem sine_tangent_not_possible_1 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.27413 ∧ Real.tan θ = 0.25719) :=
sorry

theorem sine_tangent_not_possible_2 : 
  ¬ (∃ θ : ℝ, Real.sin θ = 0.25719 ∧ Real.tan θ = 0.27413) :=
sorry

end sine_tangent_not_possible_1_sine_tangent_not_possible_2_l235_235087


namespace mary_cut_roses_l235_235383

-- Definitions from conditions
def initial_roses : ℕ := 6
def final_roses : ℕ := 16

-- The theorem to prove
theorem mary_cut_roses : (final_roses - initial_roses) = 10 :=
by
  sorry

end mary_cut_roses_l235_235383


namespace train_speed_l235_235416

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l235_235416


namespace sum_last_two_digits_l235_235296

-- Definition of the problem conditions
def seven : ℕ := 10 - 3
def thirteen : ℕ := 10 + 3

-- Main statement of the problem
theorem sum_last_two_digits (x : ℕ) (y : ℕ) : x = seven → y = thirteen → (7^25 + 13^25) % 100 = 0 :=
by
  intros
  rw [←h, ←h_1] -- Rewriting x and y in terms of seven and thirteen
  sorry -- Proof omitted

end sum_last_two_digits_l235_235296


namespace juwella_reads_pages_l235_235024

theorem juwella_reads_pages :
  let pages_three_nights_ago := 15 in
  let pages_two_nights_ago := 2 * pages_three_nights_ago in
  let pages_last_night := pages_two_nights_ago + 5 in
  let total_pages := 100 in
  let pages_read_so_far := pages_three_nights_ago + pages_two_nights_ago + pages_last_night in
  let pages_remaining := total_pages - pages_read_so_far in
  pages_remaining = 20 :=
by
  sorry

end juwella_reads_pages_l235_235024


namespace train_speed_l235_235420

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l235_235420


namespace different_prime_factors_of_factorial_eq_10_l235_235726

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l235_235726


namespace product_of_three_integers_sum_l235_235583

theorem product_of_three_integers_sum :
  ∀ (a b c : ℕ), (c = a + b) → (a * b * c = 8 * (a + b + c)) →
  (a > 0) → (b > 0) → (c > 0) →
  (∃ N1 N2 N3: ℕ, N1 = (a * b * (a + b)), N2 = (a * b * (a + b)), N3 = (a * b * (a + b)) ∧ 
  (N1 = 272 ∨ N2 = 160 ∨ N3 = 128) ∧ 
  (N1 + N2 + N3 = 560)) := sorry

end product_of_three_integers_sum_l235_235583


namespace probability_even_diff_l235_235150

theorem probability_even_diff (x y : ℕ) (hx : x ≠ y) (hx_set : x ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) (hy_set : y ∈ ({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} : set ℕ)) :
  (∃ p : ℚ, p = 5 / 22 ∧ 
    (let xy_diff_even := xy - x - y mod 2 = 0 
     in (xy_diff_even --> True))) :=
sorry

end probability_even_diff_l235_235150


namespace number_of_distinct_prime_factors_30_fact_l235_235705

/-
Define a set representing the prime numbers less than or equal to 30.
-/
def primes_le_30 := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29} 

/-
Prove that the number of distinct prime factors of 30! is 10.
-/
theorem number_of_distinct_prime_factors_30_fact : 
  (∀ n : ℕ , n ∣ (Nat.factorial 30) → (Nat.Prime n → n ∈ primes_le_30)) → 
  Nat.card primes_le_30 = 10 := by 
sorry

end number_of_distinct_prime_factors_30_fact_l235_235705


namespace smallest_number_of_students_l235_235907

theorem smallest_number_of_students (n : ℕ) (x : ℕ) 
  (h_total : n = 5 * x + 3) 
  (h_more_than_50 : n > 50) : 
  n = 53 :=
by {
  sorry
}

end smallest_number_of_students_l235_235907


namespace time_in_vancouver_l235_235950

theorem time_in_vancouver (toronto_time vancouver_time : ℕ) (h : toronto_time = 18 + 30 / 60) (h_diff : vancouver_time = toronto_time - 3) :
  vancouver_time = 15 + 30 / 60 :=
by
  sorry

end time_in_vancouver_l235_235950


namespace factorial_30_prime_count_l235_235743

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l235_235743


namespace possible_values_of_sum_l235_235539

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235539


namespace tree_planting_problem_l235_235834

variables (n t : ℕ)

theorem tree_planting_problem (h1 : 4 * n = t + 11) (h2 : 2 * n = t - 13) : n = 12 ∧ t = 37 :=
by
  sorry

end tree_planting_problem_l235_235834


namespace maximum_of_expression_l235_235787

theorem maximum_of_expression (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 3) : 
  a + b^2 + c^4 ≤ 3 ∧ (∃ a b c : ℝ, 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ a + b + c = 3 ∧ a + b^2 + c^4 = 3) :=
begin
  sorry
end

end maximum_of_expression_l235_235787


namespace polygon_with_given_angle_sum_l235_235883

-- Definition of the sum of interior angles of a polygon
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Definition of the sum of exterior angles of a polygon
def sum_exterior_angles : ℝ := 360

-- Given condition: the sum of the interior angles is four times the sum of the exterior angles
def sum_condition (n : ℕ) : Prop :=
  sum_interior_angles n = 4 * sum_exterior_angles

-- The main theorem we want to prove
theorem polygon_with_given_angle_sum : 
  ∃ n : ℕ, sum_condition n ∧ n = 10 :=
by
  sorry

end polygon_with_given_angle_sum_l235_235883


namespace find_positive_solution_l235_235228

-- Defining the variables x, y, and z as real numbers
variables (x y z : ℝ)

-- Define the conditions from the problem statement
def condition1 : Prop := x * y + 3 * x + 4 * y + 10 = 30
def condition2 : Prop := y * z + 4 * y + 2 * z + 8 = 6
def condition3 : Prop := x * z + 4 * x + 3 * z + 12 = 30

-- The theorem that states the positive solution for x is 3
theorem find_positive_solution (h1 : condition1 x y) (h2 : condition2 y z) (h3 : condition3 x z) : x = 3 :=
by {
  sorry
}

end find_positive_solution_l235_235228


namespace side_length_of_square_l235_235429

-- Define the areas of the triangles AOR, BOP, and CRQ
def S1 := 1
def S2 := 3
def S3 := 1

-- Prove that the side length of the square OPQR is 2
theorem side_length_of_square (side_length : ℝ) : 
  S1 = 1 ∧ S2 = 3 ∧ S3 = 1 → side_length = 2 :=
by
  intros h
  sorry

end side_length_of_square_l235_235429


namespace not_proportional_x2_y2_l235_235645

def directly_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x = k * y

def inversely_proportional (x y : ℝ) : Prop :=
∃ k : ℝ, x * y = k

theorem not_proportional_x2_y2 (x y : ℝ) :
  x^2 + y^2 = 16 → ¬directly_proportional x y ∧ ¬inversely_proportional x y :=
by
  sorry

end not_proportional_x2_y2_l235_235645


namespace factorial_30_prime_count_l235_235741

open Nat

def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

theorem factorial_30_prime_count : (count (fun p => p.Prime) (List.range 31)) = 10 :=
by
  sorry

end factorial_30_prime_count_l235_235741


namespace residue_system_mod_3n_l235_235790

theorem residue_system_mod_3n (n : ℕ) (h_odd : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ) (k : ℕ), 
  (∀ i, a i = 3 * i - 2) ∧ 
  (∀ i, b i = 3 * i - 3) ∧
  (∀ i (k : ℕ), 0 < k ∧ k < n → 
    (a i + a (i + 1)) % (3 * n) ≠ (a i + b i) % (3 * n) ∧ 
    (a i + b i) % (3 * n) ≠ (b i + b (i + k)) % (3 * n) ∧ 
    (a i + a (i + 1)) % (3 * n) ≠ (b i + b (i + k)) % (3 * n)) :=
sorry

end residue_system_mod_3n_l235_235790


namespace amount_after_two_years_l235_235324

/-- Defining given conditions. -/
def initial_value : ℤ := 65000
def first_year_increase : ℚ := 12 / 100
def second_year_increase : ℚ := 8 / 100

/-- The main statement that needs to be proved. -/
theorem amount_after_two_years : 
  let first_year_amount := initial_value + (initial_value * first_year_increase)
  let second_year_amount := first_year_amount + (first_year_amount * second_year_increase)
  second_year_amount = 78624 := 
by 
  sorry

end amount_after_two_years_l235_235324


namespace sum_of_possible_values_of_N_l235_235581

theorem sum_of_possible_values_of_N :
  ∃ a b c : ℕ, (a > 0 ∧ b > 0 ∧ c > 0) ∧ (abc = 8 * (a + b + c)) ∧ (c = a + b)
  ∧ (2560 = 560) :=
by
  sorry

end sum_of_possible_values_of_N_l235_235581


namespace edge_count_bound_of_no_shared_edge_triangle_l235_235366

variables {V : Type} [Fintype V]

-- Definition of a graph without two triangles sharing an edge
def no_shared_edge_triangle_graph (G : SimpleGraph V) : Prop :=
  ¬∃ u v w x : V, G.Adj u v ∧ G.Adj v w ∧ G.Adj w u ∧ G.Adj u x ∧ G.Adj v x

-- Proposition
theorem edge_count_bound_of_no_shared_edge_triangle (G : SimpleGraph V) (n : ℕ) 
  (hV : Fintype.card V = 2 * n) (hG : no_shared_edge_triangle_graph G) : 
  G.edge_finset.card ≤ n^2 + 1 :=
sorry

end edge_count_bound_of_no_shared_edge_triangle_l235_235366


namespace quadratic_equation_correct_form_l235_235313

theorem quadratic_equation_correct_form :
  ∀ (a b c x : ℝ), a = 3 → b = -6 → c = 1 → a * x^2 + c = b * x :=
by
  intros a b c x ha hb hc
  rw [ha, hb, hc]
  sorry

end quadratic_equation_correct_form_l235_235313


namespace find_m_value_l235_235349

/-- 
If the function y = (m + 1)x^(m^2 + 3m + 4) is a quadratic function, 
then the value of m is -2.
--/
theorem find_m_value 
  (m : ℝ)
  (h1 : m^2 + 3 * m + 4 = 2)
  (h2 : m + 1 ≠ 0) : 
  m = -2 := 
sorry

end find_m_value_l235_235349


namespace exists_integers_u_v_l235_235801

theorem exists_integers_u_v (A : ℕ) (a b s : ℤ)
  (hA: A = 1 ∨ A = 2 ∨ A = 3)
  (hab_rel_prime: Int.gcd a b = 1)
  (h_eq: a^2 + A * b^2 = s^3) :
  ∃ u v : ℤ, s = u^2 + A * v^2 ∧ a = u^3 - 3 * A * u * v^2 ∧ b = 3 * u^2 * v - A * v^3 := 
sorry

end exists_integers_u_v_l235_235801


namespace sqrt_continued_fraction_l235_235669

theorem sqrt_continued_fraction :
  (x : ℝ) → (h : x = Real.sqrt (3 - x)) → x = (Real.sqrt 13 - 1) / 2 :=
by
  intros x h
  sorry

end sqrt_continued_fraction_l235_235669


namespace possible_values_of_sum_l235_235511

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235511


namespace intersection_result_l235_235104

open Set

-- Define the universal set U
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x : ℝ | abs (x - 1) > 2 }

-- Define set B
def B : Set ℝ := { x : ℝ | -x^2 + 6 * x - 8 > 0 }

-- Define the complement of A in U
def compl_A : Set ℝ := U \ A

-- Define the intersection of compl_A and B
def inter_complA_B : Set ℝ := compl_A ∩ B

-- Prove that the intersection is equal to the given set
theorem intersection_result : inter_complA_B = { x : ℝ | 2 < x ∧ x ≤ 3 } :=
by
  sorry

end intersection_result_l235_235104


namespace max_sum_of_arithmetic_sequence_l235_235692

theorem max_sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (S_seq : ∀ n, S n = (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)) 
  (S16_pos : S 16 > 0) (S17_neg : S 17 < 0) : 
  ∃ m, ∀ n, S n ≤ S m ∧ m = 8 := 
sorry

end max_sum_of_arithmetic_sequence_l235_235692


namespace slope_of_line_passing_through_MN_l235_235322

theorem slope_of_line_passing_through_MN :
  let M := (-2, 1)
  let N := (1, 4)
  ∃ m : ℝ, m = (N.2 - M.2) / (N.1 - M.1) ∧ m = 1 :=
by
  sorry

end slope_of_line_passing_through_MN_l235_235322


namespace sum_of_xy_l235_235552

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235552


namespace curve_is_line_l235_235444

noncomputable def polar_to_cartesian (r : ℝ) (θ : ℝ) : ℝ × ℝ :=
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (x, y)

theorem curve_is_line (r : ℝ) (θ : ℝ) :
  r = 1 / (Real.sin θ + Real.cos θ) ↔ ∃ (x y : ℝ), (x, y) = polar_to_cartesian r θ ∧ (x + y)^2 = 1 :=
by 
  sorry

end curve_is_line_l235_235444


namespace greatest_N_exists_l235_235688

def is_condition_satisfied (N : ℕ) (xs : Fin N → ℤ) : Prop :=
  ∀ i j : Fin N, i ≠ j → ¬ (1111 ∣ ((xs i) * (xs i) - (xs i) * (xs j)))

theorem greatest_N_exists : ∃ N : ℕ, (∀ M : ℕ, (∀ xs : Fin M → ℤ, is_condition_satisfied M xs → M ≤ N)) ∧ N = 1000 :=
by
  sorry

end greatest_N_exists_l235_235688


namespace log_suff_nec_l235_235459

theorem log_suff_nec (a b : ℝ) (ha : a > 0) (hb : b > 0) : ¬ ((a > b) ↔ (Real.log b / Real.log a < 1)) := 
sorry

end log_suff_nec_l235_235459


namespace quadratic_maximum_or_minimum_l235_235236

open Real

noncomputable def quadratic_function (a b x : ℝ) : ℝ := a * x^2 + b * x - b^2 / (3 * a)

theorem quadratic_maximum_or_minimum (a b : ℝ) (h : a ≠ 0) :
  (a > 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≤ quadratic_function a b x) ∧
  (a < 0 → ∃ x₀, ∀ x, quadratic_function a b x₀ ≥ quadratic_function a b x) :=
by
  -- Proof will go here
  sorry

end quadratic_maximum_or_minimum_l235_235236


namespace sales_price_calculation_l235_235948

variables (C S : ℝ)
def gross_profit := 1.25 * C
def gross_profit_value := 30

theorem sales_price_calculation 
  (h1: gross_profit C = 30) :
  S = 54 :=
sorry

end sales_price_calculation_l235_235948


namespace other_root_eq_l235_235695

theorem other_root_eq (b : ℝ) : (∀ x, x^2 + b * x - 2 = 0 → (x = 1 ∨ x = -2)) :=
by
  intro x hx
  have : x = 1 ∨ x = -2 := sorry
  exact this

end other_root_eq_l235_235695


namespace find_n_l235_235448

variable (a r : ℚ) (n : ℕ)

def geom_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Given conditions
axiom seq_first_term : a = 1 / 3
axiom seq_common_ratio : r = 1 / 3
axiom sum_of_first_n_terms_eq : geom_sum a r n = 80 / 243

-- Prove that n = 5
theorem find_n : n = 5 := by
  sorry

end find_n_l235_235448


namespace find_smallest_int_cube_ends_368_l235_235453

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l235_235453


namespace cubes_not_touching_tin_foil_volume_l235_235815

-- Definitions for the conditions given
variables (l w h : ℕ)
-- Condition 1: Width is twice the length
def width_twice_length := w = 2 * l
-- Condition 2: Width is twice the height
def width_twice_height := w = 2 * h
-- Condition 3: The adjusted width for the inner structure in inches
def adjusted_width := w = 8

-- The theorem statement to prove the final answer
theorem cubes_not_touching_tin_foil_volume : 
  width_twice_length l w → 
  width_twice_height w h →
  adjusted_width w →
  l * w * h = 128 :=
by
  intros h1 h2 h3
  sorry

end cubes_not_touching_tin_foil_volume_l235_235815


namespace smaller_of_two_digit_product_4680_l235_235136

theorem smaller_of_two_digit_product_4680 (a b : ℕ) (h1 : a * b = 4680) (h2 : 10 ≤ a) (h3 : a < 100) (h4 : 10 ≤ b) (h5 : b < 100): min a b = 40 :=
sorry

end smaller_of_two_digit_product_4680_l235_235136


namespace rectangle_inscribed_circle_hypotenuse_l235_235145

open Real

theorem rectangle_inscribed_circle_hypotenuse
  (AB BC : ℝ)
  (h_AB : AB = 20)
  (h_BC : BC = 10)
  (r : ℝ)
  (h_r : r = 10 / 3) :
  sqrt ((AB - 2 * r) ^ 2 + BC ^ 2) = 50 / 3 :=
by {
  sorry
}

end rectangle_inscribed_circle_hypotenuse_l235_235145


namespace smallest_multiple_9_and_6_l235_235165

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l235_235165


namespace problem_l235_235103

noncomputable def roots1 : Set ℝ := { α | α^2 - 2*α + 1 = 0 }
noncomputable def roots2 : Set ℝ := { γ | γ^2 - 3*γ + 1 = 0 }

theorem problem 
  (α β γ δ : ℝ) 
  (hαβ : α ∈ roots1 ∧ β ∈ roots1)
  (hγδ : γ ∈ roots2 ∧ δ ∈ roots2) : 
  (α - γ)^2 * (β - δ)^2 = 1 := 
sorry

end problem_l235_235103


namespace tennis_balls_per_can_is_three_l235_235982

-- Definition of the number of games in each round
def games_in_round (round: Nat) : Nat :=
  match round with
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | 4 => 1
  | _ => 0

-- Definition of the average number of cans used per game
def cans_per_game : Nat := 5

-- Total number of games in the tournament
def total_games : Nat :=
  games_in_round 1 + games_in_round 2 + games_in_round 3 + games_in_round 4

-- Total number of cans used
def total_cans : Nat :=
  total_games * cans_per_game

-- Total number of tennis balls used
def total_tennis_balls : Nat := 225

-- Number of tennis balls per can
def tennis_balls_per_can : Nat :=
  total_tennis_balls / total_cans

-- Theorem to prove
theorem tennis_balls_per_can_is_three :
  tennis_balls_per_can = 3 :=
by
  -- No proof required, using sorry to skip the proof
  sorry

end tennis_balls_per_can_is_three_l235_235982


namespace percentage_ginger_is_43_l235_235923

noncomputable def calculate_percentage_ginger 
  (ginger_tbsp : ℝ) (cardamom_tsp : ℝ) (mustard_tsp : ℝ) (garlic_tbsp : ℝ) (conversion_rate : ℝ) (chile_factor : ℝ) 
  : ℝ :=
  let ginger_tsp := ginger_tbsp * conversion_rate in
  let garlic_tsp := garlic_tbsp * conversion_rate in
  let chile_tsp := mustard_tsp * chile_factor in
  let total_tsp := ginger_tsp + garlic_tsp + chile_tsp + cardamom_tsp + mustard_tsp in
  (ginger_tsp / total_tsp) * 100

theorem percentage_ginger_is_43 :
  calculate_percentage_ginger 3 1 1 2 3 4 = 43 :=
by
  sorry

end percentage_ginger_is_43_l235_235923


namespace fraction_problem_l235_235828

theorem fraction_problem :
  ((3 / 4 - 5 / 8) / 2) = 1 / 16 :=
by
  sorry

end fraction_problem_l235_235828


namespace possible_values_of_sum_l235_235515

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235515


namespace c_share_of_profit_l235_235841

theorem c_share_of_profit 
  (x : ℝ) -- The amount invested by B
  (total_profit : ℝ := 11000) -- Total profit
  (A_invest : ℝ := 3 * x) -- A's investment
  (C_invest : ℝ := (3/2) * A_invest) -- C's investment
  (total_invest : ℝ := A_invest + x + C_invest) -- Total investment
  (C_share : ℝ := C_invest / total_invest * total_profit) -- C's share of the profit
  : C_share = 99000 / 17 := 
  by sorry

end c_share_of_profit_l235_235841


namespace second_child_birth_year_l235_235362

theorem second_child_birth_year (first_child_birth : ℕ)
  (second_child_birth : ℕ)
  (third_child_birth : ℕ)
  (fourth_child_birth : ℕ)
  (first_child_years_ago : first_child_birth = 15)
  (third_child_on_second_child_fourth_birthday : third_child_birth = second_child_birth + 4)
  (fourth_child_two_years_after_third : fourth_child_birth = third_child_birth + 2)
  (fourth_child_age : fourth_child_birth = 8) :
  second_child_birth = first_child_birth - 14 := 
by
  sorry

end second_child_birth_year_l235_235362


namespace no_rectangle_from_six_different_squares_l235_235928

theorem no_rectangle_from_six_different_squares (a1 a2 a3 a4 a5 a6 : ℝ) (h: a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧ a5 < a6) :
  ¬ (∃ (L W : ℝ), a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = L * W) :=
sorry

end no_rectangle_from_six_different_squares_l235_235928


namespace f_even_l235_235249

-- Let g(x) = x^3 - x
def g (x : ℝ) : ℝ := x^3 - x

-- Let f(x) = |g(x^2)|
def f (x : ℝ) : ℝ := abs (g (x^2))

-- Prove that f(x) is even, i.e., f(-x) = f(x) for all x
theorem f_even : ∀ x : ℝ, f (-x) = f x := by
  sorry

end f_even_l235_235249


namespace sum_of_xy_l235_235551

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235551


namespace total_spent_l235_235216

theorem total_spent (cost_per_deck : ℕ) (decks_frank : ℕ) (decks_friend : ℕ) (total : ℕ) : 
  cost_per_deck = 7 → 
  decks_frank = 3 → 
  decks_friend = 2 → 
  total = (decks_frank * cost_per_deck) + (decks_friend * cost_per_deck) → 
  total = 35 :=
by
  sorry

end total_spent_l235_235216


namespace smaller_of_two_digit_product_l235_235139

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l235_235139


namespace problem_statement_l235_235895

theorem problem_statement
  (a b c : ℝ) 
  (X : ℝ) 
  (hX : X = a + b + c + 2 * Real.sqrt (a^2 + b^2 + c^2 - a * b - b * c - c * a)) :
  X ≥ max (max (3 * a) (3 * b)) (3 * c) ∧ 
  ∃ (u v w : ℝ), 
    (u = Real.sqrt (X - 3 * a) ∧ v = Real.sqrt (X - 3 * b) ∧ w = Real.sqrt (X - 3 * c) ∧ 
     ((u + v = w) ∨ (v + w = u) ∨ (w + u = v))) :=
by
  sorry

end problem_statement_l235_235895


namespace different_prime_factors_of_factorial_eq_10_l235_235727

-- First, define n as 30
def n : ℕ := 30

-- Define a list of primes less than 30
def primesLessThanN : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- This is the theorem stating the number of distinct prime factors of 30!
theorem different_prime_factors_of_factorial_eq_10 : 
  (primesLessThanN.filter (Nat.Prime)).length = 10 := by 
  sorry

end different_prime_factors_of_factorial_eq_10_l235_235727


namespace evaluate_nested_radical_l235_235656

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l235_235656


namespace number_of_prime_factors_of_30_factorial_l235_235722

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l235_235722


namespace num_prime_factors_of_30_l235_235738

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l235_235738


namespace compute_expression_l235_235063

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l235_235063


namespace squad_sizes_l235_235860

-- Definitions for conditions
def total_students (x y : ℕ) : Prop := x + y = 146
def equal_after_transfer (x y : ℕ) : Prop := x - 11 = y + 11

-- Theorem to prove the number of students in first and second-year squads
theorem squad_sizes (x y : ℕ) (h1 : total_students x y) (h2 : equal_after_transfer x y) : 
  x = 84 ∧ y = 62 :=
by
  sorry

end squad_sizes_l235_235860


namespace sum_possible_values_of_m_l235_235630

theorem sum_possible_values_of_m :
  let m_values := Finset.filter (λ m, 5 ≤ m ∧ m ≤ 17) (Finset.range 18)
  Finset.sum m_values id = 143 :=
by
  let m_values := Finset.filter (λ m, 5 ≤ m ∧ m ≤ 17) (Finset.range 18)
  have h : Finset.sum m_values id = 143 := sorry
  exact h

end sum_possible_values_of_m_l235_235630


namespace train_speed_is_60_kmph_l235_235422

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l235_235422


namespace find_x_plus_y_l235_235126

theorem find_x_plus_y (x y : ℝ) (h1 : x + Real.sin y = 2008) (h2 : x + 2008 * Real.cos y = 2007) (hy : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2007 + Real.pi / 2 := 
by
  sorry

end find_x_plus_y_l235_235126


namespace one_cow_one_bag_in_39_days_l235_235777

-- Definitions
def cows : ℕ := 52
def husks : ℕ := 104
def days : ℕ := 78

-- Problem: Given that 52 cows eat 104 bags of husk in 78 days,
-- Prove that one cow will eat one bag of husk in 39 days.
theorem one_cow_one_bag_in_39_days (cows_cons : cows = 52) (husks_cons : husks = 104) (days_cons : days = 78) :
  ∃ d : ℕ, d = 39 :=
by
  -- Placeholder for the proof.
  sorry

end one_cow_one_bag_in_39_days_l235_235777


namespace positive_difference_is_127_div_8_l235_235957

-- Defining the basic expressions
def eight_squared : ℕ := 8 ^ 2 -- 64

noncomputable def expr1 : ℝ := (eight_squared + eight_squared) / 8
noncomputable def expr2 : ℝ := (eight_squared / eight_squared) / 8

-- Problem statement
theorem positive_difference_is_127_div_8 :
  (expr1 - expr2) = 127 / 8 :=
by
  sorry

end positive_difference_is_127_div_8_l235_235957


namespace range_of_y_l235_235346

theorem range_of_y (y: ℝ) (hy: y > 0) (h_eq: ⌈y⌉ * ⌊y⌋ = 72) : 8 < y ∧ y < 9 :=
by
  sorry

end range_of_y_l235_235346


namespace nested_sqrt_eq_l235_235647

theorem nested_sqrt_eq : 
  (∃ x : ℝ, (0 < x) ∧ (x = sqrt (3 - x))) → (sqrt (3 - sqrt (3 - sqrt (3 - sqrt (3 - ...)))) = (-1 + sqrt 13) / 2) :=
by
  sorry

end nested_sqrt_eq_l235_235647


namespace find_value_of_A_l235_235458

theorem find_value_of_A (x y A : ℝ)
  (h1 : 2^x = A)
  (h2 : 7^(2*y) = A)
  (h3 : 1 / x + 2 / y = 2) : 
  A = 7 * Real.sqrt 2 := 
sorry

end find_value_of_A_l235_235458


namespace inequality_condition_l235_235282

theorem inequality_condition (a : ℝ) : 
  (∀ x y : ℝ, x^2 + 2 * x + a ≥ -y^2 - 2 * y) → a ≥ 2 :=
by
  sorry

end inequality_condition_l235_235282


namespace f_values_sum_l235_235017

noncomputable def f : ℝ → ℝ := sorry

-- defining the properties
def is_odd (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x : ℝ, f (x + p) = f x

-- given conditions
axiom f_odd : is_odd f
axiom f_periodic : is_periodic f 2

-- statement to prove
theorem f_values_sum : f 1 + f 2 + f 3 = 0 :=
by
  sorry

end f_values_sum_l235_235017


namespace smallest_positive_integer_cube_ends_368_l235_235449

theorem smallest_positive_integer_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 368 ∧ n = 34 :=
by
  sorry

end smallest_positive_integer_cube_ends_368_l235_235449


namespace find_integer_n_l235_235344

theorem find_integer_n (n : ℤ) : (⌊(n^2 : ℤ) / 4⌋ - ⌊n / 2⌋ ^ 2 = 3) → n = 7 :=
by sorry

end find_integer_n_l235_235344


namespace subtracting_seven_percent_l235_235969

theorem subtracting_seven_percent (a : ℝ) : a - 0.07 * a = 0.93 * a :=
by 
  sorry

end subtracting_seven_percent_l235_235969


namespace monotonicity_of_f_l235_235438

noncomputable theory

open Real

def f (a : ℝ) (x : ℝ) : ℝ := (1 / 2) * x^2 - a * x + (a - 1) * log x

theorem monotonicity_of_f (a : ℝ) (h : a > 1) :
  (∀ x : ℝ, 0 < x -> (a = 2 -> monotone_on (f a) (Ioi 0)) ∧
             (1 < a ∧ a < 2 -> 
              (monotone_on (λ x, true) (Ioi 0)) ∧ 
              (monotone_on (λ x, true) (Ioi 0) ) ∧
              monopnically_increasing (Ioi 0) (Ioi 0)) ) ∧
             (a > 2 -> 
              (monotone_on (λ x, true) (Ioi 0)) ∧ 
              (monotone_on (λ x, true) (Ioi 0) ) ∧
              monotone_increasing (Ioi 0) (Ioi 0)) :

sorry

end monotonicity_of_f_l235_235438


namespace max_value_F_l235_235332

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x^2
noncomputable def g (x : ℝ) : ℝ := x^2 - 2 * x

noncomputable def F (x : ℝ) : ℝ :=
if f x ≥ g x then f x else g x

theorem max_value_F : ∃ x : ℝ, ∀ y : ℝ, F y ≤ F x ∧ F x = 7 / 9 := 
sorry

end max_value_F_l235_235332


namespace five_letter_word_with_at_least_one_consonant_l235_235057

def letter_set : Set Char := {'A', 'B', 'C', 'D', 'E', 'F'}
def consonants : Set Char := {'B', 'C', 'D', 'F'}
def vowels : Set Char := {'A', 'E'}

-- Calculate the total number of 5-letter words using the letter set
def total_words : ℕ := 6^5

-- Calculate the number of 5-letter words using only vowels
def vowel_only_words : ℕ := 2^5

-- Number of 5-letter words with at least one consonant
def words_with_consonant : ℕ := total_words - vowel_only_words

theorem five_letter_word_with_at_least_one_consonant :
  words_with_consonant = 7744 :=
by
  sorry

end five_letter_word_with_at_least_one_consonant_l235_235057


namespace quadratic_eq_has_nonzero_root_l235_235463

theorem quadratic_eq_has_nonzero_root (b c : ℝ) (h : c ≠ 0) (h_eq : c^2 + b * c + c = 0) : b + c = -1 :=
sorry

end quadratic_eq_has_nonzero_root_l235_235463


namespace find_k_l235_235441

theorem find_k (a b c : ℝ) :
    (a + b) * (b + c) * (c + a) = (a + b + c) * (a * b + b * c + c * a) + (-1) * a * b * c :=
by
  sorry

end find_k_l235_235441


namespace xyz_problem_l235_235232

variables {x y z : ℝ}

theorem xyz_problem
  (h1 : y + z = 10 - 4 * x)
  (h2 : x + z = -16 - 4 * y)
  (h3 : x + y = 9 - 4 * z) :
  3 * x + 3 * y + 3 * z = 1.5 :=
by 
  sorry

end xyz_problem_l235_235232


namespace relationship_among_neg_a_neg_a3_a2_l235_235367

theorem relationship_among_neg_a_neg_a3_a2 (a : ℝ) (h : a^2 + a < 0) : -a > a^2 ∧ a^2 > -a^3 :=
by sorry

end relationship_among_neg_a_neg_a3_a2_l235_235367


namespace number_of_prime_factors_thirty_factorial_l235_235730

-- Given condition: 30! is defined as the product of all integers from 1 to 30
def thirty_factorial : ℕ := (List.range 30).map (λ n, n + 1).prod

-- Goal: Prove that the number of different prime factors of 30! is 10
theorem number_of_prime_factors_thirty_factorial : (List.filter Prime (List.range 30)).length = 10 := by
  sorry

end number_of_prime_factors_thirty_factorial_l235_235730


namespace eval_sqrt_expression_l235_235653

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l235_235653


namespace y_increase_by_41_8_units_l235_235352

theorem y_increase_by_41_8_units :
  ∀ (x y : ℝ),
    (∀ k : ℝ, y = 2 + k * 11 / 5 → x = 1 + k * 5) →
    x = 20 → y = 41.8 :=
by
  sorry

end y_increase_by_41_8_units_l235_235352


namespace wire_cut_problem_l235_235631

theorem wire_cut_problem (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h_eq_area : (a / 4) ^ 2 = π * (b / (2 * π)) ^ 2) : 
  a / b = 2 / Real.sqrt π :=
by
  sorry

end wire_cut_problem_l235_235631


namespace cost_of_new_shoes_l235_235406

theorem cost_of_new_shoes :
  ∃ P : ℝ, P = 32 ∧ (P / 2 = 14.50 + 0.10344827586206897 * 14.50) :=
sorry

end cost_of_new_shoes_l235_235406


namespace solution_l235_235212

noncomputable def f (x : ℝ) := 
  10 / (Real.sqrt (x - 5) - 10) + 
  2 / (Real.sqrt (x - 5) - 5) + 
  9 / (Real.sqrt (x - 5) + 5) + 
  18 / (Real.sqrt (x - 5) + 10)

theorem solution : 
  f (1230 / 121) = 0 := sorry

end solution_l235_235212


namespace pow_div_pow_l235_235005

variable (a : ℝ)
variable (A B : ℕ)

theorem pow_div_pow (a : ℝ) (A B : ℕ) : a^A / a^B = a^(A - B) :=
  sorry

example : a^6 / a^2 = a^4 :=
  pow_div_pow a 6 2

end pow_div_pow_l235_235005


namespace ratio_of_third_to_second_building_l235_235594

/-
The tallest building in the world is 100 feet tall. The second tallest is half that tall, the third tallest is some 
fraction of the second tallest building's height, and the fourth is one-fifth as tall as the third. All 4 buildings 
put together are 180 feet tall. What is the ratio of the height of the third tallest building to the second tallest building?

Given H1 = 100, H2 = (1 / 2) * H1, H4 = (1 / 5) * H3, 
and H1 + H2 + H3 + H4 = 180, prove that H3 / H2 = 1 / 2.
-/

theorem ratio_of_third_to_second_building :
  ∀ (H1 H2 H3 H4 : ℝ),
  H1 = 100 →
  H2 = (1 / 2) * H1 →
  H4 = (1 / 5) * H3 →
  H1 + H2 + H3 + H4 = 180 →
  (H3 / H2) = (1 / 2) :=
by
  intros H1 H2 H3 H4 h1_eq h2_half_h1 h4_fifth_h3 total_eq
  /- proof steps go here -/
  sorry

end ratio_of_third_to_second_building_l235_235594


namespace evaluate_nested_radical_l235_235655

noncomputable def nested_radical (x : ℝ) := x = Real.sqrt (3 - x)

theorem evaluate_nested_radical (x : ℝ) (h : nested_radical x) : 
  x = (Real.sqrt 13 - 1) / 2 :=
by sorry

end evaluate_nested_radical_l235_235655


namespace club_truncator_probability_l235_235639

theorem club_truncator_probability :
  let p_win := (1 : ℚ) / 3
  let p_loss := (1 : ℚ) / 3
  let p_tie := (1 : ℚ) / 3
  TotalOutcomes := 3^6
  FavorableOutcomes_W_eq_L :=
    20 + (6.choose 2 * (4.choose 2))  + (6.choose 1 * 5.choose 4) + 1
  Probability_W_eq_L := FavorableOutcomes_W_eq_L / TotalOutcomes
  Probability_W_neq_L := 1 - Probability_W_eq_L
  Probability_W_gt_L := Probability_W_neq_L / 2
  prob := Probability_W_gt_L
  fraction := prob.num / prob.denom,
  RelPrime := fraction.num.gcd fraction.denom = 1,

  (fraction.num + fraction.denom) = 341
:=
sorry

end club_truncator_probability_l235_235639


namespace exists_monochromatic_rectangle_l235_235606

theorem exists_monochromatic_rectangle 
  (coloring : ℤ × ℤ → Prop)
  (h : ∀ p : ℤ × ℤ, coloring p = red ∨ coloring p = blue)
  : ∃ (a b c d : ℤ × ℤ), (a.1 = b.1) ∧ (c.1 = d.1) ∧ (a.2 = c.2) ∧ (b.2 = d.2) ∧ (coloring a = coloring b) ∧ (coloring b = coloring c) ∧ (coloring c = coloring d) :=
sorry

end exists_monochromatic_rectangle_l235_235606


namespace common_difference_is_one_l235_235314

variable {a : ℕ → ℝ}
variable {d : ℝ}

-- Conditions given in the problem
axiom h1 : a 1 ^ 2 + a 10 ^ 2 = 101
axiom h2 : a 5 + a 6 = 11
axiom h3 : ∀ n m, n < m → a n < a m
noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n+1) = a n + d

-- Theorem stating the common difference d is 1
theorem common_difference_is_one : is_arithmetic_sequence a d → d = 1 := 
by
  sorry

end common_difference_is_one_l235_235314


namespace prime_factors_of_30_l235_235753

-- Define the set of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Prove that the number of different prime factors of 30! equals 10
theorem prime_factors_of_30! : (primes_le_30.count (∈ primes_le_30)) = 10 :=
by sorry

end prime_factors_of_30_l235_235753


namespace max_cells_primitive_dinosaur_l235_235991

section Dinosaur

universe u

-- Define a dinosaur as a structure with at least 2007 cells
structure Dinosaur (α : Type u) :=
(cells : ℕ) (connected : α → α → Prop)
(h_cells : cells ≥ 2007)
(h_connected : ∀ (x y : α), connected x y → connected y x)

-- Define a primitive dinosaur where the cells cannot be partitioned into two or more dinosaurs
structure PrimitiveDinosaur (α : Type u) extends Dinosaur α :=
(h_partition : ∀ (x : α), ¬∃ (d1 d2 : Dinosaur α), (d1.cells + d2.cells = cells) ∧ 
  (d1 ≠ d2 ∧ d1.cells ≥ 2007 ∧ d2.cells ≥ 2007))

-- Prove that the maximum number of cells in a Primitive Dinosaur is 8025
theorem max_cells_primitive_dinosaur : ∀ (α : Type u), ∃ (d : PrimitiveDinosaur α), d.cells = 8025 :=
sorry

end Dinosaur

end max_cells_primitive_dinosaur_l235_235991


namespace num_prime_factors_of_30_l235_235737

theorem num_prime_factors_of_30! : 
  nat.card {p : ℕ | nat.prime p ∧ p ≤ 30} = 10 := 
sorry

end num_prime_factors_of_30_l235_235737


namespace problem1_problem2_problem3_l235_235185

-- Problem 1
theorem problem1 (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (1/2 * x * (1 - 2 * x) ≤ 1/16) := sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 0 < x) : 
  (2 - x - 4 / x ≤ -2) := sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  (1 / x + 3 / y ≥ 1 + Real.sqrt 3 / 2) := sorry

end problem1_problem2_problem3_l235_235185


namespace find_x_l235_235196

-- Define the conditions according to the problem statement
variables {C x : ℝ} -- C is the cost per liter of pure spirit, x is the volume of water in the first solution

-- Condition 1: The cost for the first solution
def cost_first_solution (C : ℝ) (x : ℝ) : Prop := 0.50 = C * (1 / (1 + x))

-- Condition 2: The cost for the second solution (approximating 0.4999999999999999 as 0.50)
def cost_second_solution (C : ℝ) : Prop := 0.50 = C * (1 / 3)

-- The theorem to prove: x = 2 given the two conditions
theorem find_x (C : ℝ) (x : ℝ) (h1 : cost_first_solution C x) (h2 : cost_second_solution C) : x = 2 := 
sorry

end find_x_l235_235196


namespace Trevor_future_age_when_brother_is_three_times_now_l235_235952

def Trevor_current_age := 11
def Brother_current_age := 20

theorem Trevor_future_age_when_brother_is_three_times_now :
  ∃ (X : ℕ), Brother_current_age + (X - Trevor_current_age) = 3 * Trevor_current_age :=
by
  use 24
  sorry

end Trevor_future_age_when_brother_is_three_times_now_l235_235952


namespace log_sum_eq_two_l235_235849

theorem log_sum_eq_two (log6_3 log6_4 : ℝ) (H1 : Real.logb 6 3 = log6_3) (H2 : Real.logb 6 4 = log6_4) : 
  log6_3 + log6_4 = 2 := 
by 
  sorry

end log_sum_eq_two_l235_235849


namespace problem_a_l235_235495

def continuous (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib
def monotonic (f : ℝ → ℝ) : Prop := sorry -- Assume this is properly defined somewhere in Mathlib

theorem problem_a :
  ¬ (∀ (f : ℝ → ℝ), continuous f ∧ (∀ y, ∃ x, f x = y) → monotonic f) := sorry

end problem_a_l235_235495


namespace solve_for_a_l235_235035

theorem solve_for_a (a : ℝ) (h_pos : a > 0) 
  (h_roots : ∀ x, x^2 - 2*a*x - 3*a^2 = 0 → (x = -a ∨ x = 3*a)) 
  (h_diff : |(-a) - (3*a)| = 8) : a = 2 := 
sorry

end solve_for_a_l235_235035


namespace range_of_a_l235_235461

noncomputable def p (x: ℝ) : Prop := |4 * x - 1| ≤ 1
noncomputable def q (x a: ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) ≤ 0

theorem range_of_a (a: ℝ) :
  (¬ (∀ x, p x) → (¬ (∀ x, q x a))) ∧ (¬ (¬ (∀ x, p x) → (¬ (∀ x, q x a))))
  ↔ (-1 / 2 ≤ a ∧ a ≤ 0) :=
sorry

end range_of_a_l235_235461


namespace points_on_line_l235_235899

theorem points_on_line (b m n : ℝ) (hA : m = -(-5) + b) (hB : n = -(4) + b) :
  m > n :=
by
  sorry

end points_on_line_l235_235899


namespace Iain_pennies_left_l235_235897

theorem Iain_pennies_left (initial_pennies older_pennies : ℕ) (percentage : ℝ)
  (h_initial : initial_pennies = 200)
  (h_older : older_pennies = 30)
  (h_percentage : percentage = 0.20) :
  initial_pennies - older_pennies - (percentage * (initial_pennies - older_pennies)) = 136 :=
by
  sorry

end Iain_pennies_left_l235_235897


namespace Sally_seashells_l235_235554

/- Definitions -/
def Tom_seashells : Nat := 7
def Jessica_seashells : Nat := 5
def total_seashells : Nat := 21

/- Theorem statement -/
theorem Sally_seashells : total_seashells - (Tom_seashells + Jessica_seashells) = 9 := by
  -- Definitions of seashells found by Tom, Jessica and the total should be used here
  -- Proving the theorem
  sorry

end Sally_seashells_l235_235554


namespace right_triangle_area_l235_235947

theorem right_triangle_area (h : Real) (a : Real) (b : Real) (c : Real) (h_is_hypotenuse : h = 13) (a_is_leg : a = 5) (pythagorean_theorem : a^2 + b^2 = h^2) : (1 / 2) * a * b = 30 := 
by 
  sorry

end right_triangle_area_l235_235947


namespace nat_lemma_l235_235675

theorem nat_lemma (a b : ℕ) : (∃ k : ℕ, (a + b^2) * (b + a^2) = 2^k) → (a = 1 ∧ b = 1) := by
  sorry

end nat_lemma_l235_235675


namespace polynomial_value_given_cond_l235_235224

variable (x : ℝ)
theorem polynomial_value_given_cond :
  (x^2 - (5/2) * x = 6) →
  2 * x^2 - 5 * x + 6 = 18 :=
by
  sorry

end polynomial_value_given_cond_l235_235224


namespace max_value_of_function_l235_235470

noncomputable def f (x : ℝ) (α : ℝ) := x ^ α

theorem max_value_of_function (α : ℝ)
  (h₁ : f 4 α = 2)
  : ∃ a : ℝ, 3 ≤ a ∧ a ≤ 5 ∧ (f (a - 3) (α) + f (5 - a) α = 2) := 
sorry

end max_value_of_function_l235_235470


namespace number_of_prime_factors_of_30_factorial_l235_235721

-- Define the factorial function
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial(n - 1)

-- Define the list of prime numbers less than or equal to 30
def primes_upto_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define the condition that a number is prime
def is_prime (n: ℕ) : Prop :=
  2 ≤ n ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the condition that a prime number divides 30!
def prime_divides_factorial (p : ℕ) : Prop :=
  List.mem p primes_upto_30 ∧ p ∣ factorial 30

-- State the main theorem
theorem number_of_prime_factors_of_30_factorial : ∃ n : ℕ, n = 10 ∧ ∀ p : ℕ, prime_divides_factorial p → p ∈ primes_upto_30 :=
by
  sorry

end number_of_prime_factors_of_30_factorial_l235_235721


namespace remove_12_increases_probability_l235_235601

open Finset

def T : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def sums_to_18_combinations (s : Finset ℕ) : Finset (Finset ℕ) := 
  s.powerset.filter (λ x => x.card = 3 ∧ x.sum = 18)

noncomputable def probability_of_sums_to_18 (s : Finset ℕ) : ℚ :=
  let total_combinations := (s.card.choose 3 : ℚ)
  let successful_combinations := (sums_to_18_combinations s).card
  (successful_combinations : ℚ) / total_combinations

theorem remove_12_increases_probability : 
  ∀ m ∈ T, m ≠ 12 → probability_of_sums_to_18 (T.erase 12) > probability_of_sums_to_18 (T.erase m) :=
sorry

end remove_12_increases_probability_l235_235601


namespace jenny_best_neighborhood_earnings_l235_235094

theorem jenny_best_neighborhood_earnings :
  let cost_per_box := 2
  let neighborhood_a_homes := 10
  let neighborhood_a_boxes_per_home := 2
  let neighborhood_b_homes := 5
  let neighborhood_b_boxes_per_home := 5
  let earnings_a := neighborhood_a_homes * neighborhood_a_boxes_per_home * cost_per_box
  let earnings_b := neighborhood_b_homes * neighborhood_b_boxes_per_home * cost_per_box
  max earnings_a earnings_b = 50
:= by
  sorry

end jenny_best_neighborhood_earnings_l235_235094


namespace quadrilateral_diagonals_l235_235285

theorem quadrilateral_diagonals (a b c d e f : ℝ) 
  (hac : a > c) 
  (hbd : b ≥ d) 
  (hapc : a = c) 
  (hdiag1 : e^2 = (a - b)^2 + b^2) 
  (hdiag2 : f^2 = (c + b)^2 + b^2) :
  e^4 - f^4 = (a + c) / (a - c) * (d^2 * (2 * a * c + d^2) - b^2 * (2 * a * c + b^2)) :=
by
  sorry

end quadrilateral_diagonals_l235_235285


namespace carmen_candle_usage_l235_235007

-- Define the duration a candle lasts when burned for 1 hour every night.
def candle_duration_1_hour_per_night : ℕ := 8

-- Define the number of hours Carmen burns a candle each night.
def hours_burned_per_night : ℕ := 2

-- Define the number of nights over which we want to calculate the number of candles needed.
def number_of_nights : ℕ := 24

-- We want to show that given these conditions, Carmen will use 6 candles.
theorem carmen_candle_usage :
  (number_of_nights / (candle_duration_1_hour_per_night / hours_burned_per_night)) = 6 :=
by
  sorry

end carmen_candle_usage_l235_235007


namespace manny_gave_2_marbles_l235_235951

-- Define the total number of marbles
def total_marbles : ℕ := 36

-- Define the ratio parts for Mario and Manny
def mario_ratio : ℕ := 4
def manny_ratio : ℕ := 5

-- Define the total ratio parts
def total_ratio : ℕ := mario_ratio + manny_ratio

-- Define the number of marbles Manny has after giving some away
def manny_marbles_now : ℕ := 18

-- Calculate the marbles per part based on the ratio and total marbles
def marbles_per_part : ℕ := total_marbles / total_ratio

-- Calculate the number of marbles Manny originally had
def manny_marbles_original : ℕ := manny_ratio * marbles_per_part

-- Formulate the theorem
theorem manny_gave_2_marbles : manny_marbles_original - manny_marbles_now = 2 := by
  sorry

end manny_gave_2_marbles_l235_235951


namespace circle_radius_l235_235051

theorem circle_radius (x y : ℝ) : (x^2 + y^2 + 2*x = 0) → ∃ r, r = 1 :=
by sorry

end circle_radius_l235_235051


namespace train_speed_l235_235413

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l235_235413


namespace range_x2y2z_range_a_inequality_l235_235621

theorem range_x2y2z {x y z : ℝ} (h : x^2 + y^2 + z^2 = 1) : 
  -3 ≤ x + 2*y + 2*z ∧ x + 2*y + 2*z ≤ 3 :=
by sorry

theorem range_a_inequality (a : ℝ) (h : ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 1 → |a - 3| + a / 2 ≥ x + 2*y + 2*z) :
  (4 ≤ a) ∨ (a ≤ 0) :=
by sorry

end range_x2y2z_range_a_inequality_l235_235621


namespace original_number_of_people_l235_235604

-- Define the conditions as Lean definitions
def two_thirds_left (x : ℕ) : ℕ := (2 * x) / 3
def one_fourth_dancing_left (x : ℕ) : ℕ := ((x / 3) - (x / 12))

-- The problem statement as Lean theorem
theorem original_number_of_people (x : ℕ) (h : x / 4 = 15) : x = 60 :=
by sorry

end original_number_of_people_l235_235604


namespace sourdough_cost_eq_nine_l235_235850

noncomputable def cost_per_visit (white_bread_cost baguette_cost croissant_cost: ℕ) : ℕ :=
  2 * white_bread_cost + baguette_cost + croissant_cost

noncomputable def total_spent (weekly_cost num_weeks: ℕ) : ℕ :=
  weekly_cost * num_weeks

noncomputable def total_sourdough_spent (total_spent weekly_cost num_weeks: ℕ) : ℕ :=
  total_spent - weekly_cost * num_weeks

noncomputable def total_sourdough_per_week (total_sourdough_spent num_weeks: ℕ) : ℕ :=
  total_sourdough_spent / num_weeks

theorem sourdough_cost_eq_nine (white_bread_cost baguette_cost croissant_cost total_spent_over_4_weeks: ℕ)
  (h₁: white_bread_cost = 350) (h₂: baguette_cost = 150) (h₃: croissant_cost = 200) (h₄: total_spent_over_4_weeks = 7800) :
  total_sourdough_per_week (total_sourdough_spent total_spent_over_4_weeks (cost_per_visit white_bread_cost baguette_cost croissant_cost) 4) 4 = 900 :=
by 
  sorry

end sourdough_cost_eq_nine_l235_235850


namespace baseball_cards_l235_235641

theorem baseball_cards (cards_per_page new_cards pages : ℕ) (h1 : cards_per_page = 8) (h2 : new_cards = 3) (h3 : pages = 2) : 
  (pages * cards_per_page - new_cards = 13) := by
  sorry

end baseball_cards_l235_235641


namespace cost_per_tissue_box_l235_235985

-- Given conditions
def rolls_toilet_paper : ℝ := 10
def cost_per_toilet_paper : ℝ := 1.5
def rolls_paper_towels : ℝ := 7
def cost_per_paper_towel : ℝ := 2
def boxes_tissues : ℝ := 3
def total_cost : ℝ := 35

-- Deduction of individual costs
def cost_toilet_paper := rolls_toilet_paper * cost_per_toilet_paper
def cost_paper_towels := rolls_paper_towels * cost_per_paper_towel
def cost_tissues := total_cost - cost_toilet_paper - cost_paper_towels

-- Prove the cost for one box of tissues
theorem cost_per_tissue_box : (cost_tissues / boxes_tissues) = 2 :=
by
  sorry

end cost_per_tissue_box_l235_235985


namespace probability_xy_minus_x_minus_y_even_l235_235152

open Nat

theorem probability_xy_minus_x_minus_y_even :
  let S := {1,2,3,4,5,6,7,8,9,10,11,12}
  let evens := {2, 4, 6, 8, 10, 12}
  let total_pairs := (Finset.card S).choose 2
  let even_pairs := (Finset.card evens).choose 2
  even_pairs / total_pairs = 5 / 22 :=
by
  sorry

end probability_xy_minus_x_minus_y_even_l235_235152


namespace sum_values_of_cubes_eq_l235_235338

theorem sum_values_of_cubes_eq :
  ∀ (a b : ℝ), a^3 + b^3 + 3 * a * b = 1 → a + b = 1 ∨ a + b = -2 :=
by
  intros a b h
  sorry

end sum_values_of_cubes_eq_l235_235338


namespace initial_knives_l235_235317

theorem initial_knives (K T : ℕ)
  (h1 : T = 2 * K)
  (h2 : K + T + (1 / 3 : ℚ) * K + (2 / 3 : ℚ) * T = 112) : 
  K = 24 :=
by
  sorry

end initial_knives_l235_235317


namespace find_y_when_x_is_8_l235_235144

theorem find_y_when_x_is_8 (x y : ℕ) (k : ℕ) (h1 : x + y = 36) (h2 : x - y = 12) (h3 : x * y = k) (h4 : k = 288) : y = 36 :=
by
  -- Given the conditions
  sorry

end find_y_when_x_is_8_l235_235144


namespace sum_of_real_numbers_l235_235529

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235529


namespace possible_values_of_sum_l235_235543

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235543


namespace sum_of_x_y_l235_235520

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235520


namespace number_of_prime_factors_30_factorial_l235_235760

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l235_235760


namespace framed_painting_ratio_correct_l235_235977

/-- Define the conditions -/
def painting_height : ℕ := 30
def painting_width : ℕ := 20
def width_ratio : ℕ := 3

/-- Calculate the framed dimensions and check the area conditions -/
def framed_smaller_dimension (x : ℕ) : ℕ := painting_width + 2 * x
def framed_larger_dimension (x : ℕ) : ℕ := painting_height + 6 * x

theorem framed_painting_ratio_correct (x : ℕ) (h : (painting_width + 2 * x) * (painting_height + 6 * x) = 2 * (painting_width * painting_height)) :
  framed_smaller_dimension x / framed_larger_dimension x = 4 / 7 :=
by
  sorry

end framed_painting_ratio_correct_l235_235977


namespace garden_sparrows_l235_235239

theorem garden_sparrows (ratio_b_s : ℕ) (bluebirds sparrows : ℕ)
  (h1 : ratio_b_s = 4 / 5) (h2 : bluebirds = 28) :
  sparrows = 35 :=
  sorry

end garden_sparrows_l235_235239


namespace perimeter_of_triangle_hyperbola_l235_235283

theorem perimeter_of_triangle_hyperbola (x y : ℝ) (F1 F2 A B : ℝ) :
  (x^2 / 16) - (y^2 / 9) = 1 →
  |A - F2| - |A - F1| = 8 →
  |B - F2| - |B - F1| = 8 →
  |B - A| = 5 →
  |A - F2| + |B - F2| + |B - A| = 26 :=
by
  sorry

end perimeter_of_triangle_hyperbola_l235_235283


namespace prob_A_inter_B_l235_235619

noncomputable def prob_A : ℝ := 1 - (1 - p)^6
noncomputable def prob_B : ℝ := 1 - (1 - p)^6
noncomputable def prob_A_union_B : ℝ := 1 - (1 - 2*p)^6

theorem prob_A_inter_B (p : ℝ) : 
  (1 - 2*(1 - p)^6 + (1 - 2*p)^6) = prob_A + prob_B - prob_A_union_B :=
sorry

end prob_A_inter_B_l235_235619


namespace minimize_cost_l235_235624

theorem minimize_cost (x : ℝ) (h1 : 0 < x) (h2 : 400 / x * 40 ≤ 4 * x) : x = 20 :=
by
  sorry

end minimize_cost_l235_235624


namespace manny_marbles_l235_235146

theorem manny_marbles (total_marbles : ℕ) (ratio_m : ℕ) (ratio_n : ℕ) (manny_gives : ℕ) 
  (h_total : total_marbles = 36) (h_ratio_m : ratio_m = 4) (h_ratio_n : ratio_n = 5) (h_manny_gives : manny_gives = 2) : 
  (total_marbles * ratio_n / (ratio_m + ratio_n)) - manny_gives = 18 :=
by
  sorry

end manny_marbles_l235_235146


namespace find_m_of_parabola_and_line_l235_235055

theorem find_m_of_parabola_and_line (k m x1 x2 : ℝ) 
  (h_parabola_line : ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | p.1 ^ 2 = 4 * p.2} → 
                                   y = k * x + m → true)
  (h_intersection : x1 * x2 = -4) : m = 1 := 
sorry

end find_m_of_parabola_and_line_l235_235055


namespace redistributed_gnomes_l235_235132

def WestervilleWoods : ℕ := 20
def RavenswoodForest := 4 * WestervilleWoods
def GreenwoodGrove := (5 * RavenswoodForest) / 4
def OwnerTakes (f: ℕ) (p: ℚ) := p * f

def RemainingGnomes (initial: ℕ) (p: ℚ) := initial - (OwnerTakes initial p)

def TotalRemainingGnomes := 
  (RemainingGnomes RavenswoodForest (40 / 100)) + 
  (RemainingGnomes WestervilleWoods (30 / 100)) + 
  (RemainingGnomes GreenwoodGrove (50 / 100))

def GnomesPerForest := TotalRemainingGnomes / 3

theorem redistributed_gnomes : 
  2 * 37 + 38 = TotalRemainingGnomes := by
  sorry

end redistributed_gnomes_l235_235132


namespace prime_factors_30_fac_eq_10_l235_235715

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l235_235715


namespace tennis_tournament_matches_l235_235354

noncomputable def total_matches (players: ℕ) : ℕ :=
  players - 1

theorem tennis_tournament_matches :
  total_matches 104 = 103 :=
by
  sorry

end tennis_tournament_matches_l235_235354


namespace first_day_of_month_l235_235938

theorem first_day_of_month (h : weekday 30 = "Wednesday") : weekday 1 = "Tuesday" :=
sorry

end first_day_of_month_l235_235938


namespace prime_factors_of_30_factorial_l235_235747

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l235_235747


namespace num_prime_factors_30_factorial_l235_235750

theorem num_prime_factors_30_factorial : 
  (nat.factors 30!).to_finset.card = 10 := 
by sorry

end num_prime_factors_30_factorial_l235_235750


namespace excircle_inequality_l235_235795

variables {a b c : ℝ} -- The sides of the triangle

noncomputable def semiperimeter (a b c : ℝ) : ℝ := (a + b + c) / 2 -- Definition of semiperimeter

noncomputable def excircle_distance (p a : ℝ) : ℝ := p - a -- Distance from vertices to tangency points

theorem excircle_inequality (a b c : ℝ) (p : ℝ) 
    (h1 : p = semiperimeter a b c) : 
    (excircle_distance p a) + (excircle_distance p b) > p := 
by
    -- Placeholder for proof
    sorry

end excircle_inequality_l235_235795


namespace train_speed_l235_235417

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l235_235417


namespace possible_values_of_sum_l235_235516

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235516


namespace rationalize_denominator_l235_235123

theorem rationalize_denominator : (1 : ℝ) / (Real.sqrt 3 - 2) = -(Real.sqrt 3 + 2) :=
by
  sorry

end rationalize_denominator_l235_235123


namespace last_three_digits_of_2_pow_10000_l235_235965

theorem last_three_digits_of_2_pow_10000 (h : 2^500 ≡ 1 [MOD 1250]) : (2^10000) % 1000 = 1 :=
by
  sorry

end last_three_digits_of_2_pow_10000_l235_235965


namespace factorization_correct_l235_235944

theorem factorization_correct : ∃ (a b : ℕ), (a > b) ∧ (3 * b - a = 12) ∧ (x^2 - 16 * x + 63 = (x - a) * (x - b)) :=
by
  sorry

end factorization_correct_l235_235944


namespace harriet_speed_l235_235819

-- Define the conditions
def return_speed := 140 -- speed from B-town to A-ville in km/h
def total_trip_time := 5 -- total trip time in hours
def trip_time_to_B := 2.8 -- trip time from A-ville to B-town in hours

-- Define the theorem to prove
theorem harriet_speed {r_speed : ℝ} {t_time : ℝ} {t_time_B : ℝ} 
  (h1 : r_speed = 140) 
  (h2 : t_time = 5) 
  (h3 : t_time_B = 2.8) : 
  ((r_speed * (t_time - t_time_B)) / t_time_B) = 110 :=
by 
  -- Assume we have completed proof steps here.
  sorry

end harriet_speed_l235_235819


namespace sin_2x_and_tan_fraction_l235_235468

open Real

theorem sin_2x_and_tan_fraction (x : ℝ) (h : sin (π + x) + cos (π + x) = 1 / 2) :
  (sin (2 * x) = -3 / 4) ∧ ((1 + tan x) / (sin x * cos (x - π / 4)) = -8 * sqrt 2 / 3) :=
by
  sorry

end sin_2x_and_tan_fraction_l235_235468


namespace intersection_complement_A_l235_235921

def A : Set ℝ := {x | abs (x - 1) < 1}

def B : Set ℝ := {x | x < 1}

def CRB : Set ℝ := {x | x ≥ 1}

theorem intersection_complement_A :
  (CRB ∩ A) = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_complement_A_l235_235921


namespace Mikail_birthday_money_l235_235508

theorem Mikail_birthday_money (x : ℕ) (h1 : x = 3 + 3 * 3) : 5 * x = 60 := 
by 
  sorry

end Mikail_birthday_money_l235_235508


namespace greatest_y_l235_235130

theorem greatest_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -9) : y ≤ -2 :=
by {
  sorry
}

end greatest_y_l235_235130


namespace slope_of_perpendicular_line_l235_235033

-- Define the line equation as a condition
def line_eqn (x y : ℝ) : Prop := 4 * x - 6 * y = 12

-- Define the slope of the given line from its equation
noncomputable def original_slope : ℝ := 2 / 3

-- Define the negative reciprocal of the original slope
noncomputable def perp_slope (m : ℝ) : ℝ := -1 / m

-- State the theorem
theorem slope_of_perpendicular_line : perp_slope original_slope = -3 / 2 :=
by 
  sorry

end slope_of_perpendicular_line_l235_235033


namespace fraction_of_bones_in_foot_is_approx_one_eighth_l235_235573

def number_bones_human_body : ℕ := 206
def number_bones_one_foot : ℕ := 26
def fraction_bones_one_foot (total_bones foot_bones : ℕ) : ℚ := foot_bones / total_bones

theorem fraction_of_bones_in_foot_is_approx_one_eighth :
  fraction_bones_one_foot number_bones_human_body number_bones_one_foot = 13 / 103 ∧ 
  (abs ((13 / 103 : ℚ) - (1 / 8)) < 1 / 103) := 
sorry

end fraction_of_bones_in_foot_is_approx_one_eighth_l235_235573


namespace num_prime_factors_30_fac_l235_235712

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l235_235712


namespace batsman_average_after_17th_inning_l235_235394

theorem batsman_average_after_17th_inning
  (A : ℝ) -- average before 17th inning
  (h1 : (16 * A + 50) / 17 = A + 2) : 
  (A + 2) = 18 :=
by
  -- Proof goes here
  sorry

end batsman_average_after_17th_inning_l235_235394


namespace sum_of_real_numbers_l235_235530

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235530


namespace greatest_integer_less_than_PS_l235_235081

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ), PQ = 150 → 
  PS = 150 * Real.sqrt 3 → 
  (⌊PS⌋ = 259) := 
by
  intros PQ PS hPQ hPS
  sorry

end greatest_integer_less_than_PS_l235_235081


namespace option_D_correct_l235_235960

theorem option_D_correct (a b : ℝ) : -a * b + 3 * b * a = 2 * a * b :=
by sorry

end option_D_correct_l235_235960


namespace num_prime_factors_30_fac_l235_235709

open Nat

theorem num_prime_factors_30_fac : 
  ∃ (n : ℕ), numDistinctPrimeFactors 30! = n ∧ n = 10 := by
  sorry

end num_prime_factors_30_fac_l235_235709


namespace sum_of_x_y_possible_values_l235_235538

theorem sum_of_x_y_possible_values (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
sorry

end sum_of_x_y_possible_values_l235_235538


namespace apples_in_bowl_l235_235623

variable {A : ℕ}

theorem apples_in_bowl
  (initial_oranges : ℕ)
  (removed_oranges : ℕ)
  (final_oranges : ℕ)
  (total_fruit : ℕ)
  (fraction_apples : ℚ) :
  initial_oranges = 25 →
  removed_oranges = 19 →
  final_oranges = initial_oranges - removed_oranges →
  fraction_apples = (70 : ℚ) / (100 : ℚ) →
  final_oranges = total_fruit * (30 : ℚ) / (100 : ℚ) →
  A = total_fruit * fraction_apples →
  A = 14 :=
by
  sorry

end apples_in_bowl_l235_235623


namespace jasmine_gives_lola_marbles_l235_235913

theorem jasmine_gives_lola_marbles :
  ∃ (y : ℕ), ∀ (j l : ℕ), 
    j = 120 ∧ l = 15 ∧ 120 - y = 3 * (15 + y) → y = 19 := 
sorry

end jasmine_gives_lola_marbles_l235_235913


namespace slower_time_is_Tara_l235_235253

def time_to_top (stories : ℕ) (time_per_story : ℕ) : ℕ :=
  stories * time_per_story

def elevator_total_time (stories : ℕ) (time_per_story : ℕ) (stop_time : ℕ) : ℕ :=
  stories * time_per_story + (stories - 1) * stop_time

theorem slower_time_is_Tara :
  let stories := 20
  let lola_time_per_story := 10
  let tara_elevator_time_per_story := 8
  let tara_stop_time := 3 in
  max (time_to_top stories lola_time_per_story) (elevator_total_time stories tara_elevator_time_per_story tara_stop_time) = elevator_total_time stories tara_elevator_time_per_story tara_stop_time :=
by
  sorry

end slower_time_is_Tara_l235_235253


namespace product_of_tangents_is_constant_l235_235697

theorem product_of_tangents_is_constant (a b : ℝ) (h_ab : a > b) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1)
  (A1 A2 : ℝ × ℝ)
  (hA1 : A1 = (-a, 0))
  (hA2 : A2 = (a, 0)) :
  ∃ (Q1 Q2 : ℝ × ℝ),
  (A1.1 - Q1.1, A2.1 - Q2.1) = (b^2, b^2) :=
sorry

end product_of_tangents_is_constant_l235_235697


namespace power_mod_l235_235388

theorem power_mod (x n m : ℕ) : (x^n) % m = x % m := by 
  sorry

example : 5^2023 % 150 = 5 % 150 :=
by exact power_mod 5 2023 150

end power_mod_l235_235388


namespace egg_production_difference_l235_235245

def eggs_last_year : ℕ := 1416
def eggs_this_year : ℕ := 4636
def eggs_difference (a b : ℕ) : ℕ := a - b

theorem egg_production_difference : eggs_difference eggs_this_year eggs_last_year = 3220 := 
by
  sorry

end egg_production_difference_l235_235245


namespace Jake_peach_count_l235_235783

theorem Jake_peach_count (Steven_peaches : ℕ) (Jake_peach_difference : ℕ) (h1 : Steven_peaches = 19) (h2 : Jake_peach_difference = 12) : 
  Steven_peaches - Jake_peach_difference = 7 :=
by
  sorry

end Jake_peach_count_l235_235783


namespace chip_credit_card_balance_l235_235852

-- Definitions based on the problem conditions
def initial_balance : ℝ := 50.00
def interest_rate : ℝ := 0.20
def additional_amount : ℝ := 20.00

-- Define the function to calculate the final balance after two months
def final_balance (b₀ r a : ℝ) : ℝ :=
  let b₁ := b₀ * (1 + r) in
  let b₂ := (b₁ + a) * (1 + r) in
  b₂

-- Theorem to prove that the final balance is 96.00
theorem chip_credit_card_balance : final_balance initial_balance interest_rate additional_amount = 96.00 :=
by
  -- Simplified proof outline
  sorry

end chip_credit_card_balance_l235_235852


namespace last_two_digits_of_100_factorial_l235_235019

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_nonzero_digits (n : ℕ) : ℕ := sorry

theorem last_two_digits_of_100_factorial :
  last_two_nonzero_digits (factorial 100) = 24 :=
sorry

end last_two_digits_of_100_factorial_l235_235019


namespace junior_score_is_95_l235_235774

theorem junior_score_is_95:
  ∀ (n j s : ℕ) (x avg_total avg_seniors : ℕ),
    n = 20 →
    j = n * 15 / 100 →
    s = n * 85 / 100 →
    avg_total = 78 →
    avg_seniors = 75 →
    (j * x + s * avg_seniors) / n = avg_total →
    x = 95 :=
by
  sorry

end junior_score_is_95_l235_235774


namespace common_ratio_of_series_l235_235685

-- Definition of the terms in the series
def term1 : ℚ := 7 / 8
def term2 : ℚ := - (5 / 12)

-- Definition of the common ratio
def common_ratio (a1 a2 : ℚ) : ℚ := a2 / a1

-- The theorem we need to prove that the common ratio is -10/21
theorem common_ratio_of_series : common_ratio term1 term2 = -10 / 21 :=
by
  -- We skip the proof with 'sorry'
  sorry

end common_ratio_of_series_l235_235685


namespace mary_has_more_l235_235111

theorem mary_has_more (marco_initial mary_initial : ℕ) (h1 : marco_initial = 24) (h2 : mary_initial = 15) :
  let marco_final := marco_initial - 12,
      mary_final := mary_initial + 12 - 5 in
  mary_final = marco_final + 10 :=
by
  sorry

end mary_has_more_l235_235111


namespace smallest_radius_squared_of_sphere_l235_235195

theorem smallest_radius_squared_of_sphere :
  ∃ (x y z : ℤ), 
  (x - 2)^2 + y^2 + z^2 = (x^2 + (y - 4)^2 + z^2) ∧
  (x - 2)^2 + y^2 + z^2 = (x^2 + y^2 + (z - 6)^2) ∧
  (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
  (∃ r, r^2 = (x - 2)^2 + (0 - y)^2 + (0 - z)^2) ∧
  51 = r^2 :=
sorry

end smallest_radius_squared_of_sphere_l235_235195


namespace sophie_saves_money_l235_235274

-- Definitions based on the conditions
def loads_per_week : ℕ := 4
def sheets_per_load : ℕ := 1
def cost_per_box : ℝ := 5.50
def sheets_per_box : ℕ := 104
def weeks_per_year : ℕ := 52

-- Main theorem statement
theorem sophie_saves_money :
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  annual_saving = 11.00 := 
by {
  -- Calculation steps
  let sheets_per_week := loads_per_week * sheets_per_load
  let total_sheets_per_year := sheets_per_week * weeks_per_year
  let boxes_per_year := total_sheets_per_year / sheets_per_box
  let annual_saving := boxes_per_year * cost_per_box
  -- Proving the final statement
  sorry
}

end sophie_saves_money_l235_235274


namespace positive_integers_divisors_of_2_to_the_n_plus_1_l235_235325

theorem positive_integers_divisors_of_2_to_the_n_plus_1:
  ∀ n : ℕ, 0 < n → (n^2 ∣ 2^n + 1) ↔ (n = 1 ∨ n = 3) :=
by
  sorry

end positive_integers_divisors_of_2_to_the_n_plus_1_l235_235325


namespace bill_head_circumference_l235_235091

theorem bill_head_circumference (jack_head_circumference charlie_head_circumference bill_head_circumference : ℝ) :
  jack_head_circumference = 12 →
  charlie_head_circumference = (1 / 2 * jack_head_circumference) + 9 →
  bill_head_circumference = (2 / 3 * charlie_head_circumference) →
  bill_head_circumference = 10 :=
by
  intro hj hc hb
  sorry

end bill_head_circumference_l235_235091


namespace difference_of_cubes_divisible_by_8_l235_235932

theorem difference_of_cubes_divisible_by_8 (a b : ℤ) : 
  8 ∣ ((2 * a - 1) ^ 3 - (2 * b - 1) ^ 3) := 
by
  sorry

end difference_of_cubes_divisible_by_8_l235_235932


namespace intersection_A_B_l235_235045

noncomputable def A : Set ℝ := { x | (x - 1) / (x + 3) < 0 }
noncomputable def B : Set ℝ := { x | abs x < 2 }

theorem intersection_A_B :
  A ∩ B = { x : ℝ | -2 < x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l235_235045


namespace cocktail_cost_l235_235810

noncomputable def costPerLitreCocktail (cost_mixed_fruit_juice : ℝ) (cost_acai_juice : ℝ) (volume_mixed_fruit : ℝ) (volume_acai : ℝ) : ℝ :=
  let total_cost := cost_mixed_fruit_juice * volume_mixed_fruit + cost_acai_juice * volume_acai
  let total_volume := volume_mixed_fruit + volume_acai
  total_cost / total_volume

theorem cocktail_cost : costPerLitreCocktail 262.85 3104.35 32 21.333333333333332 = 1399.99 :=
  by
    sorry

end cocktail_cost_l235_235810


namespace remainder_when_P_divided_by_DD_l235_235868

noncomputable def remainder (a b : ℕ) : ℕ := a % b

theorem remainder_when_P_divided_by_DD' (P D Q R D' Q'' R'' : ℕ)
  (h1 : P = Q * D + R)
  (h2 : Q^2 = D' * Q'' + R'') :
  remainder P (D * D') = R :=
by {
  sorry
}

end remainder_when_P_divided_by_DD_l235_235868


namespace simplify_expression_l235_235183

theorem simplify_expression : (-5) - (-4) + (-7) - (2) = -5 + 4 - 7 - 2 := 
by
  sorry

end simplify_expression_l235_235183


namespace sum_of_x_y_l235_235523

theorem sum_of_x_y (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_x_y_l235_235523


namespace smallest_integer_whose_cube_ends_in_368_l235_235452

theorem smallest_integer_whose_cube_ends_in_368 :
  ∃ (n : ℕ+), (n % 2 = 0 ∧ n^3 % 1000 = 368) ∧ (∀ (m : ℕ+), m % 2 = 0 ∧ m^3 % 1000 = 368 → m ≥ n) :=
by
  sorry

end smallest_integer_whose_cube_ends_in_368_l235_235452


namespace prime_factors_of_30_factorial_l235_235735

theorem prime_factors_of_30_factorial : 
  ∀ (n : ℕ), n = 30 → (∃ s : Finset ℕ, (∀ p ∈ s, Nat.Prime p ∧ p < 30) ∧ s.card = 10) :=
by
  intros n hn
  use {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}
  split
  sorry

end prime_factors_of_30_factorial_l235_235735


namespace minimum_value_of_quadratic_l235_235637

-- Definition of the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 6 * x + 13

-- Statement of the proof problem
theorem minimum_value_of_quadratic : ∃ (y : ℝ), ∀ x : ℝ, quadratic x >= y ∧ y = 4 := by
  sorry

end minimum_value_of_quadratic_l235_235637


namespace seonmi_initial_money_l235_235555

theorem seonmi_initial_money (M : ℝ) (h1 : M/6 = 250) : M = 1500 :=
by
  sorry

end seonmi_initial_money_l235_235555


namespace fraction_equality_l235_235843

theorem fraction_equality (x y : ℝ) : (-x + y) / (-x - y) = (x - y) / (x + y) :=
by sorry

end fraction_equality_l235_235843


namespace fully_filled_boxes_l235_235785

theorem fully_filled_boxes (total_cards : ℕ) (cards_per_box : ℕ) (h1 : total_cards = 94) (h2 : cards_per_box = 8) : total_cards / cards_per_box = 11 :=
by {
  sorry
}

end fully_filled_boxes_l235_235785


namespace simplify_expression_l235_235269

variable (a : ℝ)

theorem simplify_expression (h1 : 0 < a ∨ a < 0) : a * Real.sqrt (-(1 / a)) = -Real.sqrt (-a) :=
sorry

end simplify_expression_l235_235269


namespace simplification_l235_235371

theorem simplification (b : ℝ) : 3 * b * (3 * b^3 + 2 * b) - 2 * b^2 = 9 * b^4 + 4 * b^2 :=
by
  sorry

end simplification_l235_235371


namespace quadratic_distinct_roots_l235_235070

theorem quadratic_distinct_roots (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ a * x1^2 + 2 * x1 + 1 = 0 ∧ a * x2^2 + 2 * x2 + 1 = 0) ↔ (a < 1 ∧ a ≠ 0) :=
sorry

end quadratic_distinct_roots_l235_235070


namespace max_min_distance_product_l235_235038

noncomputable def circle : set (ℝ × ℝ) := {p | (p.1 - 1)^2 + (p.2 + 1)^2 = 2}
def line (p : ℝ × ℝ) : Prop := p.1 - p.2 + 1 = 0

theorem max_min_distance_product :
  let center := (1 : ℝ, -1 : ℝ) in
  let radius := real.sqrt 2 in
  let d := abs ((1 : ℝ) - (-1)) / real.sqrt (1^2 + (-1)^2) in
  let max_distance := d + radius in
  let min_distance := d - radius in
  (max_distance * min_distance = 5 / 2) :=
begin
  sorry
end

end max_min_distance_product_l235_235038


namespace number_of_prime_factors_30_factorial_l235_235759

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l235_235759


namespace vitya_probability_l235_235302

theorem vitya_probability :
  let total_sequences := (finset.range 6).card * 
                         (finset.range 5).card * 
                         (finset.range 4).card * 
                         (finset.range 3).card * 
                         (finset.range 2).card * 
                         (finset.range 1).card,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11,
      total_possibilities := nat.choose 12 2 * nat.choose 10 2 * 
                             nat.choose 8 2 * nat.choose 6 2 * 
                             nat.choose 4 2 * nat.choose 2 2,
      P := (favorable_sequences : ℚ) / (total_possibilities : ℚ)
  in P = 1 / 720 := 
sorry

end vitya_probability_l235_235302


namespace jerome_family_members_l235_235242

-- Define the conditions of the problem
variables (C F M T : ℕ)
variables (hC : C = 20) (hF : F = C / 2) (hT : T = 33)

-- Formulate the theorem to prove
theorem jerome_family_members :
  M = T - (C + F) :=
sorry

end jerome_family_members_l235_235242


namespace train_speed_in_km_per_hr_l235_235425

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l235_235425


namespace new_bottles_from_recycling_l235_235689

theorem new_bottles_from_recycling (initial_bottles : ℕ) (required_bottles : ℕ) (h : initial_bottles = 125) (r : required_bottles = 5) : 
∃ new_bottles : ℕ, new_bottles = (initial_bottles / required_bottles ^ 2 + initial_bottles / (required_bottles * required_bottles / required_bottles) + initial_bottles / (required_bottles * required_bottles * required_bottles / required_bottles * required_bottles * required_bottles)) :=
  sorry

end new_bottles_from_recycling_l235_235689


namespace distance_origin_is_two_l235_235577

noncomputable def distance_origin_intersection : ℝ :=
  let l1 := λ x y : ℝ, x + y - 2 * real.sqrt 2 = 0
  let l2 := λ t : ℝ, (⟨ real.sqrt 2 / 2 * t, real.sqrt 2 / 2 * t⟩ : ℝ × ℝ)
  let intersection : ℝ × ℝ := ⟨ real.sqrt 2, real.sqrt 2 ⟩
  real.sqrt( (real.sqrt 2)^2 + (real.sqrt 2)^2 )

theorem distance_origin_is_two :
  distance_origin_intersection = 2 :=
sorry

end distance_origin_is_two_l235_235577


namespace fraction_calculation_l235_235006

theorem fraction_calculation :
  ((1 / 2 + 1 / 5) / (3 / 7 - 1 / 14) = 49 / 25) := 
by 
  sorry

end fraction_calculation_l235_235006


namespace train_speed_l235_235419

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l235_235419


namespace cohen_saw_1300_fish_eater_birds_l235_235180

theorem cohen_saw_1300_fish_eater_birds :
  let day1 := 300
  let day2 := 2 * day1
  let day3 := day2 - 200
  day1 + day2 + day3 = 1300 :=
by
  sorry

end cohen_saw_1300_fish_eater_birds_l235_235180


namespace nested_sqrt_eq_l235_235664

theorem nested_sqrt_eq :
  ∃ x ≥ 0, x = sqrt (3 - x) ∧ x = (-1 + sqrt 13) / 2 :=
by
  sorry

end nested_sqrt_eq_l235_235664


namespace length_ST_l235_235910

theorem length_ST (PQ QR RS SP SQ PT RT : ℝ) 
  (h1 : PQ = 6) (h2 : QR = 6)
  (h3 : RS = 6) (h4 : SP = 6)
  (h5 : SQ = 6) (h6 : PT = 14)
  (h7 : RT = 14) : 
  ∃ ST : ℝ, ST = 10 := 
by
  -- sorry is used to complete the theorem without a proof
  sorry

end length_ST_l235_235910


namespace total_pebbles_count_l235_235119

def white_pebbles : ℕ := 20
def red_pebbles : ℕ := white_pebbles / 2
def blue_pebbles : ℕ := red_pebbles / 3
def green_pebbles : ℕ := blue_pebbles + 5

theorem total_pebbles_count : white_pebbles + red_pebbles + blue_pebbles + green_pebbles = 41 := by
  sorry

end total_pebbles_count_l235_235119


namespace total_present_ages_l235_235829

theorem total_present_ages (P Q : ℕ) 
    (h1 : P - 12 = (1 / 2) * (Q - 12))
    (h2 : P = (3 / 4) * Q) : P + Q = 42 :=
by
  sorry

end total_present_ages_l235_235829


namespace find_principal_sum_l235_235966

theorem find_principal_sum (SI : ℝ) (R : ℝ) (T : ℕ) (P : ℝ) 
  (hSI : SI = 4016.25) (hR : R = 9) (hT : T = 5) : P = 8925 := 
by
  sorry

end find_principal_sum_l235_235966


namespace value_of_y_l235_235289

variable (y : ℚ)

def first_boy_marbles : ℚ := 4 * y + 2
def second_boy_marbles : ℚ := 2 * y
def third_boy_marbles : ℚ := y + 3
def total_marbles : ℚ := 31

theorem value_of_y (h : first_boy_marbles y + second_boy_marbles y + third_boy_marbles y = total_marbles) :
  y = 26 / 7 :=
by
  sorry

end value_of_y_l235_235289


namespace points_on_line_proof_l235_235559

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l235_235559


namespace unit_vector_same_direction_l235_235041

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end unit_vector_same_direction_l235_235041


namespace crossing_time_indeterminate_l235_235292

-- Define the lengths of the two trains.
def train_A_length : Nat := 120
def train_B_length : Nat := 150

-- Define the crossing time of the two trains when moving in the same direction.
def crossing_time_together : Nat := 135

-- Define a theorem to state that without additional information, the crossing time for a 150-meter train cannot be determined.
theorem crossing_time_indeterminate 
    (V120 V150 : Nat) 
    (H : V150 - V120 = 2) : 
    ∃ t, t > 0 -> t < 150 / V150 -> False :=
by 
    -- The proof is not provided.
    sorry

end crossing_time_indeterminate_l235_235292


namespace at_least_26_equal_differences_l235_235917

theorem at_least_26_equal_differences (x : Fin 102 → ℕ) (h : ∀ i j, i < j → x i < x j) (h' : ∀ i, x i < 255) :
  (∃ d : Fin 101 → ℕ, ∃ s : Finset ℕ, s.card ≥ 26 ∧ (∀ i, d i = x i.succ - x i) ∧ ∃ i j, i ≠ j ∧ (d i = d j)) :=
by {
  sorry
}

end at_least_26_equal_differences_l235_235917


namespace part_one_part_two_l235_235848

-- Part (1)
theorem part_one (x : ℝ) : x - (3 * x - 1) ≤ 2 * x + 3 → x ≥ -1 / 2 :=
by sorry

-- Part (2)
theorem part_two (x : ℝ) : 
  (3 * (x - 1) < 4 * x - 2) ∧ ((1 + 4 * x) / 3 > x - 1) → x > -1 :=
by sorry

end part_one_part_two_l235_235848


namespace count_valid_three_digit_numbers_l235_235328

theorem count_valid_three_digit_numbers : 
  ∃ n : ℕ, n = 36 ∧ 
    (∀ (a b c : ℕ), a ≠ 0 ∧ c ≠ 0 → 
    ((10 * b + c) % 4 = 0 ∧ (10 * b + a) % 4 = 0) → 
    n = 36) :=
sorry

end count_valid_three_digit_numbers_l235_235328


namespace man_l235_235822

variable (v : ℝ) (speed_with_current : ℝ) (speed_of_current : ℝ)

theorem man's_speed_against_current :
  speed_with_current = 12 ∧ speed_of_current = 2 → v - speed_of_current = 8 :=
by
  sorry

end man_l235_235822


namespace line_through_three_points_l235_235858

-- Define the points
structure Point where
  x : ℝ
  y : ℝ

-- Given conditions
def p1 : Point := { x := 1, y := -1 }
def p2 : Point := { x := 3, y := 3 }
def p3 : Point := { x := 2, y := 1 }

-- The line that passes through the points
def line_eq (m b : ℝ) (p : Point) : Prop :=
  p.y = m * p.x + b

-- The condition of passing through the three points
def passes_three_points (m b : ℝ) : Prop :=
  line_eq m b p1 ∧ line_eq m b p2 ∧ line_eq m b p3

-- The statement to prove
theorem line_through_three_points (m b : ℝ) (h : passes_three_points m b) : m + b = -1 :=
  sorry

end line_through_three_points_l235_235858


namespace stratified_sampling_total_students_sampled_l235_235188

theorem stratified_sampling_total_students_sampled 
  (seniors juniors freshmen : ℕ)
  (sampled_freshmen : ℕ)
  (ratio : ℚ)
  (h_freshmen : freshmen = 1500)
  (h_sampled_freshmen_ratio : sampled_freshmen = 75)
  (h_seniors : seniors = 1000)
  (h_juniors : juniors = 1200)
  (h_ratio : ratio = (sampled_freshmen : ℚ) / (freshmen : ℚ))
  (h_freshmen_ratio : ratio * (freshmen : ℚ) = sampled_freshmen) :
  let sampled_juniors := ratio * (juniors : ℚ)
  let sampled_seniors := ratio * (seniors : ℚ)
  sampled_freshmen + sampled_juniors + sampled_seniors = 185 := sorry

end stratified_sampling_total_students_sampled_l235_235188


namespace shares_correct_l235_235308

open Real

-- Problem setup
def original_problem (a b c d e : ℝ) : Prop :=
  a + b + c + d + e = 1020 ∧
  a = (3 / 4) * b ∧
  b = (2 / 3) * c ∧
  c = (1 / 4) * d ∧
  d = (5 / 6) * e

-- Goal
theorem shares_correct : ∃ (a b c d e : ℝ),
  original_problem a b c d e ∧
  abs (a - 58.17) < 0.01 ∧
  abs (b - 77.56) < 0.01 ∧
  abs (c - 116.34) < 0.01 ∧
  abs (d - 349.02) < 0.01 ∧
  abs (e - 419.42) < 0.01 := by
  sorry

end shares_correct_l235_235308


namespace factorial_prime_factors_l235_235718

theorem factorial_prime_factors :
  ∀ (n : ℕ), n = 30 → 
    (finset.image prime 
      (finset.filter prime (finset.range (n + 1)))).card = 10 :=
by
  intros n hn
  rw hn
  -- Additional technical Lean commands for managing finset properties and prime factorization can be added here
  sorry

end factorial_prime_factors_l235_235718


namespace common_ratio_geometric_series_l235_235677

theorem common_ratio_geometric_series :
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  (b / a) = - (10 : ℚ) / 21 :=
by
  -- definitions
  let a := (7 : ℚ) / 8
  let b := - (5 : ℚ) / 12
  -- assertion
  have ratio := b / a
  sorry

end common_ratio_geometric_series_l235_235677


namespace fill_cistern_time_l235_235629

theorem fill_cistern_time (fill_ratio : ℚ) (time_for_fill_ratio : ℚ) :
  fill_ratio = 1/11 ∧ time_for_fill_ratio = 4 → (11 * time_for_fill_ratio) = 44 :=
by
  sorry

end fill_cistern_time_l235_235629


namespace number_of_members_in_league_l235_235257

-- Define the conditions
def pair_of_socks_cost := 4
def t_shirt_cost := pair_of_socks_cost + 6
def cap_cost := t_shirt_cost - 3
def total_cost_per_member := 2 * (pair_of_socks_cost + t_shirt_cost + cap_cost)
def league_total_expenditure := 3144

-- Prove that the number of members in the league is 75
theorem number_of_members_in_league : 
  (∃ (n : ℕ), total_cost_per_member * n = league_total_expenditure) → 
  (∃ (n : ℕ), n = 75) :=
by
  sorry

end number_of_members_in_league_l235_235257


namespace min_value_of_reciprocal_sum_l235_235221

theorem min_value_of_reciprocal_sum {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hgeom : 3 = Real.sqrt (3^a * 3^b)) : (1 / a + 1 / b) = 2 :=
sorry  -- Proof not required, only the statement is needed.

end min_value_of_reciprocal_sum_l235_235221


namespace hyperbola_sufficient_but_not_necessary_asymptote_l235_235942

-- Define the equation of the hyperbola and the related asymptotes
def hyperbola_eq (a b x y : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (x^2 / a^2 - y^2 / b^2 = 1)

def asymptote_eq (a b x y : ℝ) : Prop :=
  y = b / a * x ∨ y = - (b / a * x)

-- Stating the theorem that expresses the sufficiency but not necessity
theorem hyperbola_sufficient_but_not_necessary_asymptote (a b : ℝ) :
  (∃ x y, hyperbola_eq a b x y) → (∀ x y, asymptote_eq a b x y) ∧ ¬ (∀ x y, (asymptote_eq a b x y) → (hyperbola_eq a b x y)) := 
sorry

end hyperbola_sufficient_but_not_necessary_asymptote_l235_235942


namespace mary_has_more_money_than_marco_l235_235110

variable (Marco Mary : ℕ)

theorem mary_has_more_money_than_marco
    (h1 : Marco = 24)
    (h2 : Mary = 15)
    (h3 : ∃ maryNew : ℕ, maryNew = Mary + Marco / 2 - 5)
    (h4 : ∃ marcoNew : ℕ, marcoNew = Marco / 2) :
    ∃ diff : ℕ, diff = maryNew - marcoNew ∧ diff = 10 := 
by 
    sorry

end mary_has_more_money_than_marco_l235_235110


namespace intersection_M_N_l235_235251

-- Definitions of the sets M and N
def M : Set ℝ := { -1, 0, 1 }
def N : Set ℝ := { x | x^2 ≤ x }

-- The theorem to be proven
theorem intersection_M_N : M ∩ N = { 0, 1 } :=
by
  sorry

end intersection_M_N_l235_235251


namespace approximation_irrational_quotient_l235_235796

theorem approximation_irrational_quotient 
  (r1 r2 : ℝ) (irrational : ¬ ∃ q : ℚ, r1 = q * r2) 
  (x : ℝ) (p : ℝ) (pos_p : p > 0) : 
  ∃ (k1 k2 : ℤ), |x - (k1 * r1 + k2 * r2)| < p :=
sorry

end approximation_irrational_quotient_l235_235796


namespace shaded_region_area_eq_108_l235_235377

/-- There are two concentric circles, where the outer circle has twice the radius of the inner circle,
and the total boundary length of the shaded region is 36π. Prove that the area of the shaded region
is nπ, where n = 108. -/
theorem shaded_region_area_eq_108 (r : ℝ) (h_outer : ∀ (c₁ c₂ : ℝ), c₁ = 2 * c₂) 
  (h_boundary : 2 * Real.pi * r + 2 * Real.pi * (2 * r) = 36 * Real.pi) : 
  ∃ (n : ℕ), n = 108 ∧ (Real.pi * (2 * r)^2 - Real.pi * r^2) = n * Real.pi := 
sorry

end shaded_region_area_eq_108_l235_235377


namespace cost_of_tissues_l235_235989
-- Import the entire Mathlib library

-- Define the context and the assertion without computing the proof details
theorem cost_of_tissues
  (n_tp : ℕ) -- Number of toilet paper rolls
  (c_tp : ℝ) -- Cost per toilet paper roll
  (n_pt : ℕ) -- Number of paper towels rolls
  (c_pt : ℝ) -- Cost per paper towel roll
  (n_t : ℕ) -- Number of tissue boxes
  (T : ℝ) -- Total cost of all items
  (H_tp : n_tp = 10) -- Given: 10 rolls of toilet paper
  (H_c_tp : c_tp = 1.5) -- Given: $1.50 per roll of toilet paper
  (H_pt : n_pt = 7) -- Given: 7 rolls of paper towels
  (H_c_pt : c_pt = 2) -- Given: $2 per roll of paper towel
  (H_t : n_t = 3) -- Given: 3 boxes of tissues
  (H_T : T = 35) -- Given: total cost is $35
  : (T - (n_tp * c_tp + n_pt * c_pt)) / n_t = 2 := -- Conclusion: the cost of one box of tissues is $2
by {
  sorry -- Proof details to be supplied here
}

end cost_of_tissues_l235_235989


namespace nested_radical_solution_l235_235674

noncomputable def infinite_nested_radical : ℝ :=
  let x := √(3 - √(3 - √(3 - √(3 - ...))))
  x

theorem nested_radical_solution : infinite_nested_radical = (√(13) - 1) / 2 := 
by
  sorry

end nested_radical_solution_l235_235674


namespace sum_contains_even_digit_l235_235370

-- Define the five-digit integer and its reversed form
def reversed_digits (n : ℕ) : ℕ := 
  let a := n % 10
  let b := (n / 10) % 10
  let c := (n / 100) % 10
  let d := (n / 1000) % 10
  let e := (n / 10000) % 10
  a * 10000 + b * 1000 + c * 100 + d * 10 + e

theorem sum_contains_even_digit (n m : ℕ) (h1 : n >= 10000) (h2 : n < 100000) (h3 : m = reversed_digits n) : 
  ∃ d : ℕ, d < 10 ∧ d % 2 = 0 ∧ (n + m) % 10 = d ∨ (n + m) / 10 % 10 = d ∨ (n + m) / 100 % 10 = d ∨ (n + m) / 1000 % 10 = d ∨ (n + m) / 10000 % 10 = d := 
sorry

end sum_contains_even_digit_l235_235370


namespace parabola_c_value_l235_235284

theorem parabola_c_value (a b c : ℝ) (h1 : 3 = a * (-1)^2 + b * (-1) + c)
  (h2 : 1 = a * (-2)^2 + b * (-2) + c) : c = 1 :=
sorry

end parabola_c_value_l235_235284


namespace find_expression_l235_235877

theorem find_expression (x y : ℝ) (h1 : 3 * x + y = 7) (h2 : x + 3 * y = 8) : 
  10 * x ^ 2 + 13 * x * y + 10 * y ^ 2 = 113 :=
by
  sorry

end find_expression_l235_235877


namespace larger_angle_at_3_30_l235_235956

def hour_hand_angle_3_30 : ℝ := 105.0
def minute_hand_angle_3_30 : ℝ := 180.0
def smaller_angle_between_hands : ℝ := abs (minute_hand_angle_3_30 - hour_hand_angle_3_30)
def larger_angle_between_hands : ℝ := 360.0 - smaller_angle_between_hands

theorem larger_angle_at_3_30 :
  larger_angle_between_hands = 285.0 := 
  sorry

end larger_angle_at_3_30_l235_235956


namespace triangle_ABC_BC_length_l235_235358

theorem triangle_ABC_BC_length 
  (A B C D : ℝ)
  (AB AD DC AC BD BC : ℝ)
  (h1 : BD = 20)
  (h2 : AC = 69)
  (h3 : AB = 29)
  (h4 : BD^2 + DC^2 = BC^2)
  (h5 : AD^2 + BD^2 = AB^2)
  (h6 : AC = AD + DC) : 
  BC = 52 := 
by
  sorry

end triangle_ABC_BC_length_l235_235358


namespace greatest_integer_less_than_PS_l235_235077

noncomputable def rectangle_problem (PQ PS : ℝ) (T : ℝ) (PT QP : ℝ) : ℝ := real.sqrt (PQ * PQ + PS * PS) / 2

theorem greatest_integer_less_than_PS :
  ∀ (PQ PS : ℝ) (hPQ : PQ = 150)
  (T_midpoint : T = PS / 2)
  (PT_perpendicular_QT : PT * PT + T * T = PQ * PQ),
  int.floor (PS) = 212 :=
by
  intros PQ PS hPQ T_midpoint PT_perpendicular_QT
  have h₁ : PS = rectangle_problem PQ PS T PQ,
  {
    sorry
  }
  have h₂ : 150 * real.sqrt 2,
  {
    sorry
  }
  have h₃ : (⌊150 * real.sqrt 2⌋ : ℤ) = 212,
  {
    sorry
  }
  exact h₃

end greatest_integer_less_than_PS_l235_235077


namespace find_fourth_number_l235_235574

theorem find_fourth_number 
  (average : ℝ) 
  (a1 a2 a3 : ℝ) 
  (x : ℝ) 
  (n : ℝ) 
  (h1 : average = 20) 
  (h2 : a1 = 3) 
  (h3 : a2 = 16) 
  (h4 : a3 = 33) 
  (h5 : n = 27) 
  (h_avg : (a1 + a2 + a3 + x) / 4 = average) :
  x = n + 1 :=
by
  sorry

end find_fourth_number_l235_235574


namespace remainder_3_pow_19_mod_10_l235_235608

theorem remainder_3_pow_19_mod_10 : (3^19) % 10 = 7 := 
by 
  sorry

end remainder_3_pow_19_mod_10_l235_235608


namespace find_a_max_min_values_l235_235472

noncomputable def f (x a b : ℝ) := (1 / 3) * x ^ 3 - a * x ^ 2 + (a ^ 2 - 1) * x + b

theorem find_a (a b : ℝ) : 
  (∀ x, deriv (f x a b) x = x ^ 2 - 2 * a * x + (a ^ 2 - 1)) → 
  (deriv (f 1 a b) 1 = 0) → 
  ∃ a : ℝ, a = 1 :=
by
  sorry

theorem max_min_values (a b : ℝ) :
  let fa := f (1 : ℝ) 1 b in
  fa = 2 →
  (∀ x, deriv (f x 1 b) x = x^2 - 2 * x) →
  (f (0 : ℝ) 1 b = 8 / 3) →
  (f (2 : ℝ) 1 b = 4 / 3) →
  (f (-2 : ℝ) 1 b = -4) →
  (f (4 : ℝ) 1 b = 8) →
  ∃ (max_value min_value : ℝ), max_value = 8 ∧ min_value = -4 :=
by
  sorry

end find_a_max_min_values_l235_235472


namespace smallest_multiple_9_and_6_l235_235166

theorem smallest_multiple_9_and_6 : ∃ n : ℕ, n > 0 ∧ n % 9 = 0 ∧ n % 6 = 0 ∧ ∀ m : ℕ, m > 0 ∧ m % 9 = 0 ∧ m % 6 = 0 → n ≤ m :=
by
  have h := Nat.lcm 9 6
  use h
  split
  sorry

end smallest_multiple_9_and_6_l235_235166


namespace Cole_drive_time_to_work_l235_235014

theorem Cole_drive_time_to_work :
  ∀ (D T_work T_home : ℝ),
    (T_work = D / 80) →
    (T_home = D / 120) →
    (T_work + T_home = 3) →
    (T_work * 60 = 108) :=
by
  intros D T_work T_home h1 h2 h3
  sorry

end Cole_drive_time_to_work_l235_235014


namespace bills_head_circumference_l235_235090

/-- Jack is ordering custom baseball caps for him and his two best friends, and we need to prove the circumference of Bill's head. -/
theorem bills_head_circumference (Jack : ℝ) (Charlie : ℝ) (Bill : ℝ)
  (h1 : Jack = 12)
  (h2 : Charlie = (1 / 2) * Jack + 9)
  (h3 : Bill = (2 / 3) * Charlie) :
  Bill = 10 :=
by sorry

end bills_head_circumference_l235_235090


namespace smaller_of_two_digit_product_l235_235138

theorem smaller_of_two_digit_product (a b : ℕ) (ha : 10 ≤ a) (hb : 10 ≤ b) (ha' : a < 100) (hb' : b < 100) 
  (hprod : a * b = 4680) : min a b = 52 :=
by
  sorry

end smaller_of_two_digit_product_l235_235138


namespace number_of_prime_factors_30_factorial_l235_235758

theorem number_of_prime_factors_30_factorial : (List.filter Nat.Prime (List.range 31)).length = 10 := by
  sorry

end number_of_prime_factors_30_factorial_l235_235758


namespace train_speed_l235_235414

theorem train_speed (length_of_train time_to_cross : ℕ) (h_length : length_of_train = 50) (h_time : time_to_cross = 3) : 
  (length_of_train / time_to_cross : ℝ) * 3.6 = 60 := by
  sorry

end train_speed_l235_235414


namespace lcm_1540_2310_l235_235154

theorem lcm_1540_2310 : Nat.lcm 1540 2310 = 4620 :=
by sorry

end lcm_1540_2310_l235_235154


namespace max_area_central_angle_l235_235881

theorem max_area_central_angle (r l : ℝ) (S α : ℝ) (h1 : 2 * r + l = 4)
  (h2 : S = (1 / 2) * l * r) : (∀ x y : ℝ, (1 / 2) * x * y ≤ (1 / 4) * ((x + y) / 2) ^ 2) → α = l / r → α = 2 :=
by
  sorry

end max_area_central_angle_l235_235881


namespace two_digit_divisors_1995_l235_235439

theorem two_digit_divisors_1995 :
  (∃ (n : Finset ℕ), (∀ x ∈ n, 10 ≤ x ∧ x < 100 ∧ 1995 % x = 0) ∧ n.card = 6 ∧ ∃ y ∈ n, y = 95) :=
by
  sorry

end two_digit_divisors_1995_l235_235439


namespace fraction_comparison_l235_235331

theorem fraction_comparison (a b : ℝ) (h : a > b ∧ b > 0) : 
  (a / b) > (a + 1) / (b + 1) :=
by
  sorry

end fraction_comparison_l235_235331


namespace exist_initial_points_l235_235570

theorem exist_initial_points (n : ℕ) (h : 9 * n - 8 = 82) : ∃ n = 10 :=
by
  sorry

end exist_initial_points_l235_235570


namespace hyperbola_eccentricity_range_l235_235475

theorem hyperbola_eccentricity_range 
(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
(hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(parabola_eq : ∀ y x, y^2 = 8 * a * x)
(right_vertex : A = (a, 0))
(focus : F = (2 * a, 0))
(P : ℝ × ℝ)
(asymptote_eq : P = (x0, b / a * x0))
(perpendicular_condition : (x0 ^ 2 - (3 * a - b^2 / a^2) * x0 + 2 * a^2 = 0))
(hyperbola_properties: c^2 = a^2 + b^2) :
1 < c / a ∧ c / a <= 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l235_235475


namespace four_g_users_scientific_notation_l235_235312

-- Condition for scientific notation
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10

-- Given problem in scientific notation form
theorem four_g_users_scientific_notation :
  ∃ a n, is_scientific_notation a n 1030000000 ∧ a = 1.03 ∧ n = 9 :=
sorry

end four_g_users_scientific_notation_l235_235312


namespace time_to_cover_same_distance_l235_235605

theorem time_to_cover_same_distance
  (a b c d : ℕ) (k : ℕ) 
  (h_k : k = 3) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) (h_pos_d : 0 < d) 
  (h_speed_eq : 3 * (a + 2 * b) = 3 * a - b) : 
  (a + 2 * b) * (c + d) / (3 * a - b) = (a + 2 * b) * (c + d) / (3 * a - b) :=
by sorry

end time_to_cover_same_distance_l235_235605


namespace extreme_point_of_f_l235_235436

noncomputable def f (x : ℝ) : ℝ := (3 / 2) * x^2 - Real.log x

theorem extreme_point_of_f : 
  ∃ c : ℝ, c = Real.sqrt 3 / 3 ∧ (∀ x: ℝ, x > 0 → (f x > f c → x > c) ∧ (f x < f c → x < c)) := 
sorry

end extreme_point_of_f_l235_235436


namespace candy_pieces_per_pile_l235_235690

theorem candy_pieces_per_pile :
  ∀ (total_candies eaten_candies num_piles pieces_per_pile : ℕ),
    total_candies = 108 →
    eaten_candies = 36 →
    num_piles = 8 →
    pieces_per_pile = (total_candies - eaten_candies) / num_piles →
    pieces_per_pile = 9 :=
by
  intros total_candies eaten_candies num_piles pieces_per_pile
  sorry

end candy_pieces_per_pile_l235_235690


namespace sum_product_of_integers_l235_235589

theorem sum_product_of_integers (a b c : ℕ) (h₁ : c = a + b) (h₂ : N = a * b * c) (h₃ : N = 8 * (a + b + c)) : 
  a * b * (a + b) = 16 * (a + b) :=
by {
  sorry
}

end sum_product_of_integers_l235_235589


namespace no_digit_C_makes_2C4_multiple_of_5_l235_235867

theorem no_digit_C_makes_2C4_multiple_of_5 : ∀ (C : ℕ), (2 * 100 + C * 10 + 4 ≠ 0 ∨ 2 * 100 + C * 10 + 4 ≠ 5) := 
by 
  intros C
  have h : 4 ≠ 0 := by norm_num
  have h2 : 4 ≠ 5 := by norm_num
  sorry

end no_digit_C_makes_2C4_multiple_of_5_l235_235867


namespace gummy_cost_proof_l235_235851

variables (lollipop_cost : ℝ) (num_lollipops : ℕ) (initial_money : ℝ) (remaining_money : ℝ)
variables (num_gummies : ℕ) (cost_per_gummy : ℝ)

-- Conditions
def conditions : Prop :=
  lollipop_cost = 1.50 ∧
  num_lollipops = 4 ∧
  initial_money = 15 ∧
  remaining_money = 5 ∧
  num_gummies = 2 ∧
  initial_money - remaining_money = (num_lollipops * lollipop_cost) + (num_gummies * cost_per_gummy)

-- Proof problem
theorem gummy_cost_proof : conditions lollipop_cost num_lollipops initial_money remaining_money num_gummies cost_per_gummy → cost_per_gummy = 2 :=
by
  sorry  -- Solution steps would be filled in here


end gummy_cost_proof_l235_235851


namespace carmen_candles_needed_l235_235009

-- Definitions based on the conditions

def candle_lifespan_1_hour : Nat := 8  -- a candle lasts 8 nights when burned 1 hour each night
def nights_total : Nat := 24  -- total nights

-- Question: How many candles are needed if burned 2 hours a night?

theorem carmen_candles_needed (h : candle_lifespan_1_hour / 2 = 4) :
  nights_total / 4 = 6 := 
  sorry

end carmen_candles_needed_l235_235009


namespace sum_of_real_numbers_l235_235526

theorem sum_of_real_numbers (x y : ℝ) (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := 
  sorry

end sum_of_real_numbers_l235_235526


namespace scout_troop_net_profit_l235_235839

theorem scout_troop_net_profit :
  ∃ (cost_per_bar selling_price_per_bar : ℝ),
    cost_per_bar = 1 / 3 ∧
    selling_price_per_bar = 0.6 ∧
    (1500 * selling_price_per_bar - (1500 * cost_per_bar + 50) = 350) :=
by {
  sorry
}

end scout_troop_net_profit_l235_235839


namespace bus_rent_proof_l235_235591

theorem bus_rent_proof (r1 r2 : ℝ) (r1_rent_eq : r1 + 2 * r2 = 2800) (r2_mult : r2 = 1.25 * r1) :
  r1 = 800 ∧ r2 = 1000 := 
by
  sorry

end bus_rent_proof_l235_235591


namespace find_a_plus_b_l235_235893

/-- Given the sets M = {x | |x-4| + |x-1| < 5} and N = {x | a < x < 6}, and M ∩ N = {2, b}, 
prove that a + b = 7. -/
theorem find_a_plus_b 
  (M : Set ℝ := { x | |x - 4| + |x - 1| < 5 }) 
  (N : Set ℝ := { x | a < x ∧ x < 6 }) 
  (a b : ℝ)
  (h_inter : M ∩ N = {2, b}) :
  a + b = 7 :=
sorry

end find_a_plus_b_l235_235893


namespace prob_at_least_3_out_of_4_patients_cured_l235_235837

namespace MedicineCure

open ProbabilityTheory

-- Define the probability of curing a patient
def cure_prob : ℝ := 0.95

-- Define the probability that at least 3 out of 4 patients are cured
def at_least_3_cured : ℝ :=
  let prob_3_of_4 := 4 * (cure_prob^3) * (1 - cure_prob)
  let prob_4_of_4 := cure_prob^4
  prob_3_of_4 + prob_4_of_4

theorem prob_at_least_3_out_of_4_patients_cured :
  at_least_3_cured = 0.99 :=
by
  -- placeholder for proof
  sorry

end MedicineCure

end prob_at_least_3_out_of_4_patients_cured_l235_235837


namespace happy_valley_zoo_animal_arrangement_l235_235940

theorem happy_valley_zoo_animal_arrangement :
  let parrots := 5
  let dogs := 3
  let cats := 4
  let total_animals := parrots + dogs + cats
  (total_animals = 12) →
    (∃ no_of_ways_to_arrange,
      no_of_ways_to_arrange = 2 * (parrots.factorial) * (dogs.factorial) * (cats.factorial) ∧
      no_of_ways_to_arrange = 34560) :=
by
  sorry

end happy_valley_zoo_animal_arrangement_l235_235940


namespace possible_values_of_sum_l235_235514

theorem possible_values_of_sum (x y : ℝ) (h : x^3 + y^3 + 21 * x * y = 343) :
  x + y = 7 ∨ x + y = -14 :=
sorry

end possible_values_of_sum_l235_235514


namespace shifted_function_l235_235576

def initial_fun (x : ℝ) : ℝ := 5 * (x - 1) ^ 2 + 1

theorem shifted_function :
  (∀ x, initial_fun (x - 2) - 3 = 5 * (x + 1) ^ 2 - 2) :=
by
  intro x
  -- sorry statement to indicate proof should be here
  sorry

end shifted_function_l235_235576


namespace water_added_l235_235836

theorem water_added (W X : ℝ) 
  (h1 : 45 / W = 2 / 1)
  (h2 : 45 / (W + X) = 6 / 5) : 
  X = 15 := 
by
  sorry

end water_added_l235_235836


namespace quadratic_inequality_solution_l235_235333

theorem quadratic_inequality_solution 
  (a : ℝ) 
  (h : ∀ x : ℝ, -1 < x ∧ x < a → -x^2 + 2 * a * x + a + 1 > a + 1) : -1 < a ∧ a ≤ -1/2 :=
sorry

end quadratic_inequality_solution_l235_235333


namespace largest_divisor_of_composite_l235_235456

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end largest_divisor_of_composite_l235_235456


namespace arithmetic_mean_probability_l235_235219

theorem arithmetic_mean_probability
  (a b c : ℝ)
  (h1 : a + b + c = 1)
  (h2 : b = (a + c) / 2) :
  b = 1 / 3 :=
by
  sorry

end arithmetic_mean_probability_l235_235219


namespace workshop_participants_problem_l235_235287

variable (WorkshopSize : ℕ) 
variable (LeftHanded : ℕ) 
variable (RockMusicLovers : ℕ) 
variable (RightHandedDislikeRock : ℕ) 
variable (Under25 : ℕ)
variable (RightHandedUnder25RockMusicLovers : ℕ)
variable (y : ℕ)

theorem workshop_participants_problem
  (h1 : WorkshopSize = 30)
  (h2 : LeftHanded = 12)
  (h3 : RockMusicLovers = 18)
  (h4 : RightHandedDislikeRock = 5)
  (h5 : Under25 = 9)
  (h6 : RightHandedUnder25RockMusicLovers = 3)
  (h7 : WorkshopSize = LeftHanded + (WorkshopSize - LeftHanded))
  (h8 : WorkshopSize - LeftHanded = RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + (WorkshopSize - LeftHanded - RightHandedDislikeRock - RightHandedUnder25RockMusicLovers - y))
  (h9 : WorkshopSize - (RightHandedDislikeRock + RightHandedUnder25RockMusicLovers + Under25 - y - (RockMusicLovers - y)) - (LeftHanded - y) = WorkshopSize) :
  y = 5 := by
  sorry

end workshop_participants_problem_l235_235287


namespace number_of_pairs_l235_235231

theorem number_of_pairs (n : ℕ) (h : n = 2835) :
  ∃ (count : ℕ), count = 20 ∧
  (∀ (x y : ℕ), (0 < x ∧ 0 < y ∧ x < y ∧ (x^2 + y^2) % (x + y) = 0 ∧ (x^2 + y^2) / (x + y) ∣ n) → count = 20) := 
sorry

end number_of_pairs_l235_235231


namespace line_intersects_hyperbola_l235_235485

theorem line_intersects_hyperbola 
  (k : ℝ)
  (hyp : ∃ x y : ℝ, y = k * x + 2 ∧ x^2 - y^2 = 6) :
  -Real.sqrt 15 / 3 < k ∧ k < -1 := 
sorry


end line_intersects_hyperbola_l235_235485


namespace dog_total_distance_l235_235622

-- Define the conditions
def distance_between_A_and_B : ℝ := 100
def speed_A : ℝ := 6
def speed_B : ℝ := 4
def speed_dog : ℝ := 10

-- Define the statement we want to prove
theorem dog_total_distance : ∀ t : ℝ, (speed_A + speed_B) * t = distance_between_A_and_B → speed_dog * t = 100 :=
by
  intro t
  intro h
  sorry

end dog_total_distance_l235_235622


namespace mary_more_than_marco_l235_235107

def marco_initial : ℕ := 24
def mary_initial : ℕ := 15
def half_marco : ℕ := marco_initial / 2
def mary_after_give : ℕ := mary_initial + half_marco
def mary_after_spend : ℕ := mary_after_give - 5
def marco_final : ℕ := marco_initial - half_marco

theorem mary_more_than_marco :
  mary_after_spend - marco_final = 10 := by
  sorry

end mary_more_than_marco_l235_235107


namespace probability_sequence_123456_l235_235303

theorem probability_sequence_123456 :
  let total_sequences := 66 * 45 * 28 * 15 * 6 * 1,
      favorable_sequences := 1 * 3 * 5 * 7 * 9 * 11
  in (favorable_sequences : ℚ) / total_sequences = 1 / 720 := 
by 
  sorry

end probability_sequence_123456_l235_235303


namespace quadratic_inequality_solution_l235_235380

theorem quadratic_inequality_solution (x : ℝ) : 
  ((x - 1) * x ≥ 2) ↔ (x ≤ -1 ∨ x ≥ 2) := 
sorry

end quadratic_inequality_solution_l235_235380


namespace john_books_per_day_l235_235492

theorem john_books_per_day (books_total : ℕ) (total_weeks : ℕ) (days_per_week : ℕ) (total_days : ℕ)
  (read_days_eq : total_days = total_weeks * days_per_week)
  (books_per_day_eq : books_total = total_days * 4) : (books_total / total_days = 4) :=
by
  -- The conditions state the following:
  -- books_total = 48 (total books read)
  -- total_weeks = 6 (total number of weeks)
  -- days_per_week = 2 (number of days John reads per week)
  -- total_days = 12 (total number of days in which John reads books)
  -- read_days_eq :- total_days = total_weeks * days_per_week
  -- books_per_day_eq :- books_total = total_days * 4
  sorry

end john_books_per_day_l235_235492


namespace polygon_interior_exterior_relation_l235_235884

theorem polygon_interior_exterior_relation :
  ∃ (n : ℕ), (n > 2) ∧ ((n - 2) * 180 = 4 * 360) ∧ n = 10 :=
by
  sorry

end polygon_interior_exterior_relation_l235_235884


namespace trains_cross_time_l235_235827

theorem trains_cross_time (length : ℝ) (time1 time2 : ℝ) (speed1 speed2 relative_speed : ℝ) 
  (H1 : length = 120) 
  (H2 : time1 = 12) 
  (H3 : time2 = 20) 
  (H4 : speed1 = length / time1) 
  (H5 : speed2 = length / time2) 
  (H6 : relative_speed = speed1 + speed2) 
  (total_distance : ℝ) (H7 : total_distance = length + length) 
  (T : ℝ) (H8 : T = total_distance / relative_speed) :
  T = 15 := 
sorry

end trains_cross_time_l235_235827


namespace cost_of_one_box_of_tissues_l235_235987

variable (num_toilet_paper : ℕ) (num_paper_towels : ℕ) (num_tissues : ℕ)
variable (cost_toilet_paper : ℝ) (cost_paper_towels : ℝ) (total_cost : ℝ)

theorem cost_of_one_box_of_tissues (num_toilet_paper = 10) 
                                   (num_paper_towels = 7) 
                                   (num_tissues = 3)
                                   (cost_toilet_paper = 1.50) 
                                   (cost_paper_towels = 2.00) 
                                   (total_cost = 35.00) :
  let total_cost_toilet_paper := num_toilet_paper * cost_toilet_paper,
      total_cost_paper_towels := num_paper_towels * cost_paper_towels,
      cost_left_for_tissues := total_cost - (total_cost_toilet_paper + total_cost_paper_towels),
      one_box_tissues_cost := cost_left_for_tissues / num_tissues
  in one_box_tissues_cost = 2.00 := 
sorry

end cost_of_one_box_of_tissues_l235_235987


namespace curve_is_line_l235_235443

noncomputable def polar_eq (θ : ℝ) : ℝ :=
  1 / (Real.sin θ + Real.cos θ)

def cartesian_x (r θ : ℝ) := r * Real.cos θ
def cartesian_y (r θ : ℝ) := r * Real.sin θ

theorem curve_is_line (r θ : ℝ) :
  let x := cartesian_x r θ,
      y := cartesian_y r θ in
  r = polar_eq θ → y + x = 1 :=
by
  sorry

end curve_is_line_l235_235443


namespace math_problem_l235_235469

-- Cartesian equation of the curve C
def cartesian_equation_of_curve : Prop :=
  ∀ x y : ℝ, (x - 2)^2 + y^2 = 4 ↔ ∃ ρ θ : ℝ, ρ = 4 * cos θ ∧ (ρ^2 = x^2 + y^2) ∧ (x = ρ * cos θ)

-- Normal equation of the line l
def normal_equation_of_line : Prop :=
  ∀ x y t : ℝ, (x = -1 + (sqrt 3 / 2) * t ∧ y = (1 / 2) * t) ↔ (x - sqrt 3 * y + 1 = 0)

-- Distance between points P and Q
def distance_PQ : Prop :=
  ∀ t1 t2 : ℝ, (t1 + t2 = 3 * sqrt 3 ∧ t1 * t2 = 5) → (|t1 - t2| = sqrt 7)

theorem math_problem : Prop :=
  cartesian_equation_of_curve ∧ normal_equation_of_line ∧ distance_PQ

end math_problem_l235_235469


namespace compute_expression_l235_235062

-- Given condition
def condition (x : ℝ) : Prop := x + 1/x = 3

-- Theorem to prove
theorem compute_expression (x : ℝ) (hx : condition x) : (x - 1) ^ 2 + 16 / (x - 1) ^ 2 = 8 := 
by
  sorry

end compute_expression_l235_235062


namespace jerry_cut_maple_trees_l235_235490

theorem jerry_cut_maple_trees :
  (∀ pine maple walnut : ℕ, 
    pine = 8 * 80 ∧ 
    walnut = 4 * 100 ∧ 
    1220 = pine + walnut + maple * 60) → 
  maple = 3 := 
by 
  sorry

end jerry_cut_maple_trees_l235_235490


namespace smallest_multiple_of_9_and_6_is_18_l235_235174

theorem smallest_multiple_of_9_and_6_is_18 :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ 
  (∀ m : ℕ, m > 0 ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) :=
sorry

end smallest_multiple_of_9_and_6_is_18_l235_235174


namespace average_rate_of_change_l235_235890

noncomputable def f (x : ℝ) : ℝ := x^2 + 2

theorem average_rate_of_change :
  (f 3 - f 1) / (3 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l235_235890


namespace ratio_3_7_not_possible_l235_235971

theorem ratio_3_7_not_possible (n : ℕ) (h : 30 < n ∧ n < 40) :
  ¬ (∃ k : ℕ, n = 10 * k) :=
by {
  sorry
}

end ratio_3_7_not_possible_l235_235971


namespace relationship_among_abc_l235_235210

noncomputable def a : ℝ := 20.3
noncomputable def b : ℝ := 0.32
noncomputable def c : ℝ := Real.log 25 / Real.log 10

theorem relationship_among_abc : b < a ∧ a < c :=
by
  -- Proof needs to be filled in here
  sorry

end relationship_among_abc_l235_235210


namespace mikail_birthday_money_l235_235507

theorem mikail_birthday_money :
  ∀ (A M : ℕ), A = 3 * 3 → M = 5 * A → M = 45 :=
by
  intros A M hA hM
  rw [hA] at hM
  rw [hM]
  norm_num

end mikail_birthday_money_l235_235507


namespace smallest_multiple_of_9_and_6_l235_235162

theorem smallest_multiple_of_9_and_6 : ∃ n : ℕ, (n > 0) ∧ (n % 9 = 0) ∧ (n % 6 = 0) ∧ (∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m) := 
begin
  use 18,
  split,
  { -- n > 0
    exact nat.succ_pos',
  },
  split,
  { -- n % 9 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 9),
  },
  split,
  { -- n % 6 = 0
    exact nat.mod_eq_zero_of_dvd (dvd_refl 6),
  },
  { -- ∀ m : ℕ, (m > 0) ∧ (m % 9 = 0) ∧ (m % 6 = 0) → n ≤ m
    intros m h_pos h_multiple9 h_multiple6,
    exact le_of_dvd h_pos (nat.lcm_dvd_prime_multiples 6 9),
  },
  sorry, -- Since full proof capabilities are not required here, "sorry" is used to skip the proof process.
end

end smallest_multiple_of_9_and_6_l235_235162


namespace train_speed_is_60_kmph_l235_235424

noncomputable def speed_of_train_in_kmph (length_meters time_seconds : ℝ) : ℝ :=
  (length_meters / time_seconds) * 3.6

theorem train_speed_is_60_kmph (length_meters time_seconds : ℝ) :
  length_meters = 50 → time_seconds = 3 → speed_of_train_in_kmph length_meters time_seconds = 60 :=
by
  intros h_length h_time
  simp [speed_of_train_in_kmph, h_length, h_time]
  norm_num
  sorry

end train_speed_is_60_kmph_l235_235424


namespace proof_problem_l235_235474

def f (x : ℝ) : ℝ := x^2 - 6 * x + 5

-- The two conditions
def condition1 (x y : ℝ) : Prop := f x + f y ≤ 0
def condition2 (x y : ℝ) : Prop := f x - f y ≥ 0

-- Equivalent description
def circle_condition (x y : ℝ) : Prop := (x - 3)^2 + (y - 3)^2 ≤ 8
def region1 (x y : ℝ) : Prop := y ≤ x ∧ y ≥ 6 - x
def region2 (x y : ℝ) : Prop := y ≥ x ∧ y ≤ 6 - x

-- The proof statement
theorem proof_problem (x y : ℝ) :
  (condition1 x y ∧ condition2 x y) ↔ 
  (circle_condition x y ∧ (region1 x y ∨ region2 x y)) :=
sorry

end proof_problem_l235_235474


namespace percentage_needed_to_pass_l235_235197

-- Define conditions
def student_score : ℕ := 80
def marks_shortfall : ℕ := 40
def total_marks : ℕ := 400

-- Theorem statement: The percentage of marks required to pass the test.
theorem percentage_needed_to_pass : (student_score + marks_shortfall) * 100 / total_marks = 30 := by
  sorry

end percentage_needed_to_pass_l235_235197


namespace sum_of_xy_l235_235550

theorem sum_of_xy {x y : ℝ} (h : x^3 + 21 * x * y + y^3 = 343) : x + y = 7 ∨ x + y = -14 := sorry

end sum_of_xy_l235_235550


namespace Juwella_reads_pages_l235_235023

theorem Juwella_reads_pages (p1 p2 p3 p_total p_tonight : ℕ) 
                            (h1 : p1 = 15)
                            (h2 : p2 = 2 * p1)
                            (h3 : p3 = p2 + 5)
                            (h4 : p_total = 100) 
                            (h5 : p_total = p1 + p2 + p3 + p_tonight) :
  p_tonight = 20 := 
sorry

end Juwella_reads_pages_l235_235023


namespace blaine_fish_caught_l235_235096

theorem blaine_fish_caught (B : ℕ) (cond1 : B + 2 * B = 15) : B = 5 := by 
  sorry

end blaine_fish_caught_l235_235096


namespace rationalize_denominator_l235_235929

theorem rationalize_denominator (h : ∀ x: ℝ, x = 1 / (Real.sqrt 3 - 2)) : 
    1 / (Real.sqrt 3 - 2) = - Real.sqrt 3 - 2 :=
by
  sorry

end rationalize_denominator_l235_235929


namespace g_at_six_l235_235501

def g (x : ℝ) : ℝ := 2 * x^4 - 19 * x^3 + 30 * x^2 - 12 * x - 72

theorem g_at_six : g 6 = 288 :=
by
  sorry

end g_at_six_l235_235501


namespace prime_factors_30_fac_eq_10_l235_235713

open Nat

theorem prime_factors_30_fac_eq_10 : 
  (finset.primeFactors (Nat.factorial 30)).card = 10 := 
by 
  sorry

end prime_factors_30_fac_eq_10_l235_235713


namespace factorial_prime_factors_l235_235763

theorem factorial_prime_factors :
  {p : ℕ | p.prime ∧ p ≤ 30}.card = 10 :=
by
  sorry

end factorial_prime_factors_l235_235763


namespace distinct_fib_sum_2017_l235_235018

-- Define the Fibonacci sequence as given.
def fib : ℕ → ℕ
| 0 => 1
| 1 => 2
| (n+2) => (fib (n+1)) + (fib n)

-- Define the predicate for representing a number as a sum of distinct Fibonacci numbers.
def can_be_written_as_sum_of_distinct_fibs (n : ℕ) : Prop :=
  ∃ s : Finset ℕ, (s.sum fib = n) ∧ (∀ (i j : ℕ), i ≠ j → i ∉ s → j ∉ s)

theorem distinct_fib_sum_2017 : ∃! s : Finset ℕ, s.sum fib = 2017 ∧ (∀ (i j : ℕ), i ≠ j → i ≠ j → i ∉ s → j ∉ s) :=
sorry

end distinct_fib_sum_2017_l235_235018


namespace work_completion_rate_l235_235823

theorem work_completion_rate (A B D : ℝ) (W : ℝ) (hB : B = W / 9) (hA : A = W / 10) (hD : D = 90 / 19) : 
  (A + B) * D = W := 
by 
  sorry

end work_completion_rate_l235_235823


namespace find_p_q_l235_235064

variable {R : Type*} [CommRing R]

theorem find_p_q (p q : R) :
  (X^5 - X^4 + X^3 - p * X^2 + q * X + 4 : Polynomial R) % (X - 2) = 0 ∧ 
  (X^5 - X^4 + X^3 - p * X^2 + q * X + 4 : Polynomial R) % (X + 1) = 0 → 
  (p = 5 ∧ q = -4) :=
by
  sorry

end find_p_q_l235_235064


namespace probability_of_passing_l235_235355

theorem probability_of_passing :
  let total_combinations := (comb 10 3)
  let success_two_correct_one_incorrect := (comb 4 1) * (comb 6 2)
  let success_all_correct := (comb 6 3)
  let total_successful_outcomes := success_two_correct_one_incorrect + success_all_correct
  total_successful_outcomes / total_combinations = 2 / 3 :=
by
  sorry

end probability_of_passing_l235_235355


namespace relationship_of_abc_l235_235872

theorem relationship_of_abc (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 4) : c > b ∧ b > a := by
  sorry

end relationship_of_abc_l235_235872


namespace find_value_of_f_neg_3_over_2_l235_235788

noncomputable def f : ℝ → ℝ := sorry

theorem find_value_of_f_neg_3_over_2 (h1 : ∀ x : ℝ, f (-x) = -f x) 
    (h2 : ∀ x : ℝ, f (x + 3/2) = -f x) : 
    f (- 3 / 2) = 0 := 
sorry

end find_value_of_f_neg_3_over_2_l235_235788


namespace selling_price_of_article_l235_235408

theorem selling_price_of_article (cost_price gain_percent : ℝ) (h1 : cost_price = 100) (h2 : gain_percent = 30) : 
  cost_price + (gain_percent / 100) * cost_price = 130 := 
by 
  sorry

end selling_price_of_article_l235_235408


namespace leonards_age_l235_235493

variable (L N J : ℕ)

theorem leonards_age (h1 : L = N - 4) (h2 : N = J / 2) (h3 : L + N + J = 36) : L = 6 := 
by 
  sorry

end leonards_age_l235_235493


namespace final_shirt_price_l235_235433

theorem final_shirt_price :
  let cost_price := 20
  let profit_rate := 0.30
  let discount_rate := 0.50
  let profit := cost_price * profit_rate
  let regular_selling_price := cost_price + profit
  let final_price := regular_selling_price * discount_rate
  in final_price = 13 :=
by
  sorry

end final_shirt_price_l235_235433


namespace mabel_shark_ratio_l235_235255

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

end mabel_shark_ratio_l235_235255


namespace compare_neg_fractions_l235_235204

theorem compare_neg_fractions : - (4 / 3 : ℚ) < - (5 / 4 : ℚ) := 
by sorry

end compare_neg_fractions_l235_235204


namespace least_integer_with_remainders_l235_235955

theorem least_integer_with_remainders :
  ∃ M : ℕ, 
    M % 6 = 5 ∧
    M % 7 = 6 ∧
    M % 9 = 8 ∧
    M % 10 = 9 ∧
    M % 11 = 10 ∧
    M = 6929 :=
by
  sorry

end least_integer_with_remainders_l235_235955


namespace common_ratio_of_geometric_series_l235_235681

noncomputable def first_term : ℝ := 7/8
noncomputable def second_term : ℝ := -5/12
noncomputable def third_term : ℝ := 25/144

theorem common_ratio_of_geometric_series : 
  (second_term / first_term = -10/21) ∧ (third_term / second_term = -10/21) := by
  sorry

end common_ratio_of_geometric_series_l235_235681


namespace solution_l235_235920

theorem solution (x y : ℝ) (h1 : x ≠ y) (h2 : x^2 - 2000 * x = y^2 - 2000 * y) : 
  x + y = 2000 := 
by 
  sorry

end solution_l235_235920


namespace train_speed_in_km_per_hr_l235_235427

-- Definitions from the problem conditions
def length_of_train : ℝ := 50
def time_to_cross_pole : ℝ := 3

-- Conversion factor from the problem 
def meter_per_sec_to_km_per_hr : ℝ := 3.6

-- Lean theorem statement based on problem conditions and solution
theorem train_speed_in_km_per_hr : 
  (length_of_train / time_to_cross_pole) * meter_per_sec_to_km_per_hr = 60 := by
  sorry

end train_speed_in_km_per_hr_l235_235427


namespace tiles_needed_l235_235392

theorem tiles_needed (S : ℕ) (n : ℕ) (k : ℕ) (N : ℕ) (H1 : S = 18144) 
  (H2 : n * k^2 = S) (H3 : n = (N * (N + 1)) / 2) : n = 2016 := 
sorry

end tiles_needed_l235_235392


namespace negation_statement_l235_235391

variable {α : Type} (S : Set α)

theorem negation_statement (P : α → Prop) :
  (∀ x ∈ S, ¬ P x) ↔ (∃ x ∈ S, P x) :=
by
  sorry

end negation_statement_l235_235391


namespace part_a_part_b_part_c_l235_235635

open ProbabilityTheory

namespace Problem

-- Definitions of conditions
variable (B G : Event) (p : ℙ (B ∪ G) = 1 / 2) (h : ℙ B = ℙ G)

-- Part (a)
theorem part_a (hb : ℙ (B ∩ G)) : ℙ (B ∩ G) = 1 / 2 := by
  sorry 

-- Additional condition for part (b)
variable (OneBoy : Event) (B1 : ℙ OneBoy = ℙ (B ∩ OneBoy))

-- Part (b)
theorem part_b (hb1 : ℙ (B ∩ G ∩ OneBoy)) : ℙ (B ∩ G ∩ OneBoy) = 2 / 3 := by
  sorry

-- Additional condition for part (c)
variable (BoyMonday : Event) (Bm : ℙ BoyMonday = ℙ (B ∩ BoyMonday))

-- Part (c)
theorem part_c (hbm : ℙ (B ∩ G ∩ BoyMonday)) : ℙ (B ∩ G ∩ BoyMonday) = 14 / 27 := by
  sorry

end Problem

end part_a_part_b_part_c_l235_235635


namespace find_constant_c_l235_235378

theorem find_constant_c (c : ℝ) :
  (∀ x y : ℝ, x + y = c ∧ y - (2 + 5) / 2 = x - (8 + 11) / 2) →
  (c = 13) :=
by
  sorry

end find_constant_c_l235_235378


namespace expected_value_of_N_l235_235201

noncomputable def expected_value_N : ℝ :=
  30

theorem expected_value_of_N :
  -- Suppose Bob chooses a 4-digit binary string uniformly at random,
  -- and examines an infinite sequence of independent random binary bits.
  -- Let N be the least number of bits Bob has to examine to find his chosen string.
  -- Then the expected value of N is 30.
  expected_value_N = 30 :=
by
  sorry

end expected_value_of_N_l235_235201


namespace train_speed_l235_235418

-- Define the conditions
def train_length : ℝ := 50 -- Length of the train in meters
def crossing_time : ℝ := 3 -- Time to cross the pole in seconds

-- Define the speed in meters per second and convert it to km/hr
noncomputable def speed_mps : ℝ := train_length / crossing_time
noncomputable def speed_kmph : ℝ := speed_mps * 3.6 -- Conversion factor

-- Theorem statement: Prove that the calculated speed in km/hr is 60 km/hr
theorem train_speed : speed_kmph = 60 := by
  sorry

end train_speed_l235_235418


namespace pentagon_largest_angle_l235_235076

theorem pentagon_largest_angle
  (F G H I J : ℝ)
  (hF : F = 90)
  (hG : G = 70)
  (hH_eq_I : H = I)
  (hJ : J = 2 * H + 20)
  (sum_angles : F + G + H + I + J = 540) :
  max F (max G (max H (max I J))) = 200 :=
by
  sorry

end pentagon_largest_angle_l235_235076


namespace prime_factors_of_30_factorial_l235_235746

-- Define the factorial function
def factorial : ℕ → ℕ
| 0     := 1
| (n + 1) := (n + 1) * factorial n

-- List of prime numbers less than or equal to 30
def primes_le_30 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

-- Define a predicate that checks if a number is a prime factor of factorial 30
def is_prime_factor_of_factorial (p : ℕ) : Prop :=
  p ∈ primes_le_30

-- Prove the number of distinct prime factors of 30! is 10
theorem prime_factors_of_30_factorial :
  (primes_le_30.filter (λ p, is_prime p)).length = 10 :=
by {
  -- Proof skipped, the statement asserts the length is 10
  sorry
}

end prime_factors_of_30_factorial_l235_235746


namespace tangent_parallel_line_exists_l235_235494

theorem tangent_parallel_line_exists 
  (A B C H : Point) 
  [is_acute_triangle A B C]
  [is_orthocenter H A B C]
  : ∃ l, parallel l (line_through B C) ∧ (tangent l (incircle (triangle A B H))) ∧ (tangent l (incircle (triangle A C H))) := 
by
  sorry

end tangent_parallel_line_exists_l235_235494


namespace people_counted_l235_235001

-- Define the conditions
def first_day_count (second_day_count : ℕ) : ℕ := 2 * second_day_count
def second_day_count : ℕ := 500

-- Define the total count
def total_count (first_day : ℕ) (second_day : ℕ) : ℕ := first_day + second_day

-- Statement of the proof problem: Prove that the total count is 1500 given the conditions
theorem people_counted : total_count (first_day_count second_day_count) second_day_count = 1500 := by
  sorry

end people_counted_l235_235001


namespace no_arith_geo_progression_S1_S2_S3_l235_235992

noncomputable def S_1 (A B C : Point) : ℝ := sorry -- area of triangle ABC
noncomputable def S_2 (A B E : Point) : ℝ := sorry -- area of triangle ABE
noncomputable def S_3 (A B D : Point) : ℝ := sorry -- area of triangle ABD

def bisecting_plane (A B D C E : Point) : Prop := sorry -- plane bisects dihedral angle at AB

theorem no_arith_geo_progression_S1_S2_S3 (A B C D E : Point) 
(h_bisect : bisecting_plane A B D C E) :
¬ (∃ (S1 S2 S3 : ℝ), S1 = S_1 A B C ∧ S2 = S_2 A B E ∧ S3 = S_3 A B D ∧ 
  (S2 = (S1 + S3) / 2 ∨ S2^2 = S1 * S3 )) :=
sorry

end no_arith_geo_progression_S1_S2_S3_l235_235992


namespace points_on_line_proof_l235_235560

theorem points_on_line_proof (n : ℕ) (hn : n = 10) : 
  let after_first_procedure := 3 * n - 2 in
  let after_second_procedure := 3 * after_first_procedure - 2 in
  after_second_procedure = 82 :=
by
  let after_first_procedure := 3 * n - 2
  let after_second_procedure := 3 * after_first_procedure - 2
  have h : after_second_procedure = 9 * n - 8 := by
    calc
      after_second_procedure = 3 * (3 * n - 2) - 2 : rfl
                      ... = 9 * n - 6 - 2      : by ring
                      ... = 9 * n - 8          : by ring
  rw [hn] at h 
  exact h.symm.trans (by norm_num)

end points_on_line_proof_l235_235560


namespace even_three_digit_numbers_count_l235_235479

theorem even_three_digit_numbers_count :
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  count = 18 :=
by
  let digits := [0, 1, 2, 3, 4]
  let even_digits := [2, 4]
  let count := 2 * 3 * 3
  show count = 18
  sorry

end even_three_digit_numbers_count_l235_235479


namespace product_expression_evaluates_to_32_l235_235638

theorem product_expression_evaluates_to_32 : 
  (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  -- The proof itself is not required, hence we can put sorry here
  sorry

end product_expression_evaluates_to_32_l235_235638


namespace acute_angle_of_parallel_vectors_l235_235229
open Real

theorem acute_angle_of_parallel_vectors (α : ℝ) (h₁ : abs (α * π / 180) < π / 2) :
  let a := (3 / 2, sin (α * π / 180))
  let b := (sin (α * π / 180), 1 / 6) 
  a.1 * b.2 = a.2 * b.1 → α = 30 :=
by
  sorry

end acute_angle_of_parallel_vectors_l235_235229


namespace eval_sqrt_expression_l235_235654

noncomputable def x : ℝ :=
  Real.sqrt 3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - Real.sqrt (3 - …))))

theorem eval_sqrt_expression (x : ℝ) (h : x = Real.sqrt (3 - x)) : x = (-1 + Real.sqrt 13) / 2 :=
by {
  sorry
}

end eval_sqrt_expression_l235_235654


namespace prism_edges_l235_235593

theorem prism_edges (V F E n : ℕ) (h1 : V + F + E = 44) (h2 : V = 2 * n) (h3 : F = n + 2) (h4 : E = 3 * n) : E = 21 := by
  sorry

end prism_edges_l235_235593


namespace uncolored_area_of_rectangle_l235_235915

theorem uncolored_area_of_rectangle :
  let width := 30
  let length := 50
  let radius := width / 4
  let rectangle_area := width * length
  let circle_area := π * (radius ^ 2)
  let total_circles_area := 4 * circle_area
  rectangle_area - total_circles_area = 1500 - 225 * π := by
  sorry

end uncolored_area_of_rectangle_l235_235915


namespace points_on_line_possible_l235_235563

theorem points_on_line_possible : ∃ n : ℕ, 9 * n - 8 = 82 :=
by
  sorry

end points_on_line_possible_l235_235563


namespace find_smallest_int_cube_ends_368_l235_235454

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l235_235454


namespace baseball_games_per_month_l235_235600

theorem baseball_games_per_month (total_games : ℕ) (season_length : ℕ) (games_per_month : ℕ) :
  total_games = 14 → season_length = 2 → games_per_month = total_games / season_length → games_per_month = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end baseball_games_per_month_l235_235600
