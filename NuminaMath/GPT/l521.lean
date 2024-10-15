import Mathlib

namespace NUMINAMATH_GPT_bowls_initially_bought_l521_52139

theorem bowls_initially_bought 
  (x : ℕ) 
  (cost_per_bowl : ℕ := 13) 
  (revenue_per_bowl : ℕ := 17)
  (sold_bowls : ℕ := 108)
  (profit_percentage : ℝ := 23.88663967611336) 
  (approx_x : ℝ := 139) :
  (23.88663967611336 / 100) * (cost_per_bowl : ℝ) * (x : ℝ) = 
    (sold_bowls * revenue_per_bowl) - (sold_bowls * cost_per_bowl) → 
  abs ((x : ℝ) - approx_x) < 0.5 :=
by
  sorry

end NUMINAMATH_GPT_bowls_initially_bought_l521_52139


namespace NUMINAMATH_GPT_car_speed_first_hour_l521_52195

theorem car_speed_first_hour (x : ℕ) (hx : x = 65) : 
  let speed_second_hour := 45 
  let average_speed := 55
  (x + 45) / 2 = 55 
  :=
  by
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l521_52195


namespace NUMINAMATH_GPT_dodecahedron_has_150_interior_diagonals_l521_52176

def dodecahedron_diagonals (vertices : ℕ) (adjacent : ℕ) : ℕ :=
  let total := vertices * (vertices - adjacent - 1) / 2
  total

theorem dodecahedron_has_150_interior_diagonals :
  dodecahedron_diagonals 20 4 = 150 :=
by
  sorry

end NUMINAMATH_GPT_dodecahedron_has_150_interior_diagonals_l521_52176


namespace NUMINAMATH_GPT_max_value_of_f_l521_52177

noncomputable def f (x : ℝ) : ℝ :=
  Real.sqrt (x^4 - 3*x^2 - 6*x + 13) - Real.sqrt (x^4 - x^2 + 1)

theorem max_value_of_f : ∃ x : ℝ, f x = Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_max_value_of_f_l521_52177


namespace NUMINAMATH_GPT_train_speed_l521_52148

noncomputable def train_length : ℝ := 2500
noncomputable def time_to_cross_pole : ℝ := 35

noncomputable def speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed :
  speed_in_kmph train_length time_to_cross_pole = 257.14 := by
  sorry

end NUMINAMATH_GPT_train_speed_l521_52148


namespace NUMINAMATH_GPT_f_2017_plus_f_2016_l521_52157

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = - f x
axiom f_even_shift : ∀ x : ℝ, f (x + 2) = f (-x + 2)
axiom f_at_neg1 : f (-1) = -1

theorem f_2017_plus_f_2016 : f 2017 + f 2016 = 1 :=
by
  sorry

end NUMINAMATH_GPT_f_2017_plus_f_2016_l521_52157


namespace NUMINAMATH_GPT_find_min_n_l521_52121

theorem find_min_n (k : ℕ) : ∃ n, 
  (∀ (m : ℕ), (k = 2 * m → n = 100 * (m + 1)) ∨ (k = 2 * m + 1 → n = 100 * (m + 1) + 1)) ∧
  (∀ n', (∀ (m : ℕ), (k = 2 * m → n' ≥ 100 * (m + 1)) ∨ (k = 2 * m + 1 → n' ≥ 100 * (m + 1) + 1)) → n' ≥ n) :=
by {
  sorry
}

end NUMINAMATH_GPT_find_min_n_l521_52121


namespace NUMINAMATH_GPT_fraction_of_arith_geo_seq_l521_52126

theorem fraction_of_arith_geo_seq (a : ℕ → ℝ) (d : ℝ) (h_d : d ≠ 0)
  (h_seq_arith : ∀ n, a (n+1) = a n + d)
  (h_seq_geo : (a 1 + 2 * d)^2 = a 1 * (a 1 + 8 * d)) :
  (a 1 + a 3 + a 9) / (a 2 + a 4 + a 10) = 13 / 16 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_arith_geo_seq_l521_52126


namespace NUMINAMATH_GPT_no_solutions_abs_eq_3x_plus_6_l521_52131

theorem no_solutions_abs_eq_3x_plus_6 : ¬ ∃ x : ℝ, |x| = 3 * (|x| + 2) :=
by {
  sorry
}

end NUMINAMATH_GPT_no_solutions_abs_eq_3x_plus_6_l521_52131


namespace NUMINAMATH_GPT_floor_sqrt_80_eq_8_l521_52110

theorem floor_sqrt_80_eq_8 (h1: 8 * 8 = 64) (h2: 9 * 9 = 81) (h3: 8 < Real.sqrt 80) (h4: Real.sqrt 80 < 9) :
  Int.floor (Real.sqrt 80) = 8 :=
sorry

end NUMINAMATH_GPT_floor_sqrt_80_eq_8_l521_52110


namespace NUMINAMATH_GPT_parallelLines_perpendicularLines_l521_52116

-- Problem A: Parallel lines
theorem parallelLines (a : ℝ) : 
  (∀x y : ℝ, y = -x + 2 * a → y = (a^2 - 2) * x + 2 → -1 = a^2 - 2) → 
  a = -1 := 
sorry

-- Problem B: Perpendicular lines
theorem perpendicularLines (a : ℝ) : 
  (∀x y : ℝ, y = (2 * a - 1) * x + 3 → y = 4 * x - 3 → (2 * a - 1) * 4 = -1) →
  a = 3 / 8 := 
sorry

end NUMINAMATH_GPT_parallelLines_perpendicularLines_l521_52116


namespace NUMINAMATH_GPT_center_of_circle_eq_minus_two_four_l521_52133

theorem center_of_circle_eq_minus_two_four : 
  ∀ (x y : ℝ), x^2 + 4 * x + y^2 - 8 * y + 16 = 0 → (x, y) = (-2, 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_center_of_circle_eq_minus_two_four_l521_52133


namespace NUMINAMATH_GPT_f_identity_l521_52104

def f (x : ℝ) : ℝ := (2 * x + 1)^5 - 5 * (2 * x + 1)^4 + 10 * (2 * x + 1)^3 - 10 * (2 * x + 1)^2 + 5 * (2 * x + 1) - 1

theorem f_identity (x : ℝ) : f x = 32 * x^5 :=
by
  -- the proof is omitted
  sorry

end NUMINAMATH_GPT_f_identity_l521_52104


namespace NUMINAMATH_GPT_program_output_is_1023_l521_52161

-- Definition placeholder for program output.
def program_output : ℕ := 1023

-- Theorem stating the program's output.
theorem program_output_is_1023 : program_output = 1023 := 
by 
  -- Proof details are omitted.
  sorry

end NUMINAMATH_GPT_program_output_is_1023_l521_52161


namespace NUMINAMATH_GPT_sum_possible_values_A_B_l521_52118

theorem sum_possible_values_A_B : 
  ∀ (A B : ℕ), 
  (0 ≤ A ∧ A ≤ 9) ∧ 
  (0 ≤ B ∧ B ≤ 9) ∧ 
  ∃ k : ℕ, 28 + A + B = 9 * k 
  → (A + B = 8 ∨ A + B = 17) 
  → A + B = 25 :=
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_A_B_l521_52118


namespace NUMINAMATH_GPT_joe_used_225_gallons_l521_52167

def initial_paint : ℕ := 360

def paint_first_week (initial : ℕ) : ℕ := initial / 4

def remaining_paint_after_first_week (initial : ℕ) : ℕ :=
  initial - paint_first_week initial

def paint_second_week (remaining : ℕ) : ℕ := remaining / 2

def total_paint_used (initial : ℕ) : ℕ :=
  paint_first_week initial + paint_second_week (remaining_paint_after_first_week initial)

theorem joe_used_225_gallons :
  total_paint_used initial_paint = 225 :=
by
  sorry

end NUMINAMATH_GPT_joe_used_225_gallons_l521_52167


namespace NUMINAMATH_GPT_apprentice_time_l521_52178

theorem apprentice_time
  (x y : ℝ)
  (h1 : 7 * x + 4 * y = 5 / 9)
  (h2 : 11 * x + 8 * y = 17 / 18)
  (hy : y > 0) :
  1 / y = 24 :=
by
  sorry

end NUMINAMATH_GPT_apprentice_time_l521_52178


namespace NUMINAMATH_GPT_triangle_angle_120_l521_52197

theorem triangle_angle_120 (a b c : ℝ) (B : ℝ) (hB : B = 120) :
  a^2 + a * c + c^2 - b^2 = 0 := by
sorry

end NUMINAMATH_GPT_triangle_angle_120_l521_52197


namespace NUMINAMATH_GPT_find_number_l521_52147

theorem find_number (x : ℝ) (h₁ : 0.40 * x = 130 + 190) : x = 800 :=
sorry

end NUMINAMATH_GPT_find_number_l521_52147


namespace NUMINAMATH_GPT_price_reductions_l521_52124

theorem price_reductions (a : ℝ) : 18400 * (1 - a / 100)^2 = 16000 :=
sorry

end NUMINAMATH_GPT_price_reductions_l521_52124


namespace NUMINAMATH_GPT_vector_transitivity_l521_52136

variables (V : Type*) [AddCommGroup V] [Module ℝ V]
variables (a b c : V)

theorem vector_transitivity (h1 : a = b) (h2 : b = c) : a = c :=
by {
  sorry
}

end NUMINAMATH_GPT_vector_transitivity_l521_52136


namespace NUMINAMATH_GPT_find_x_given_total_area_l521_52145

theorem find_x_given_total_area :
  ∃ x : ℝ, (16 * x^2 + 36 * x^2 + 6 * x^2 + 3 * x^2 = 1100) ∧ (x = Real.sqrt (1100 / 61)) :=
sorry

end NUMINAMATH_GPT_find_x_given_total_area_l521_52145


namespace NUMINAMATH_GPT_max_notebooks_l521_52123

-- Definitions based on the conditions
def joshMoney : ℕ := 1050
def notebookCost : ℕ := 75

-- Statement to prove
theorem max_notebooks (x : ℕ) : notebookCost * x ≤ joshMoney → x ≤ 14 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_max_notebooks_l521_52123


namespace NUMINAMATH_GPT_FriedChickenDinner_orders_count_l521_52183

-- Defining the number of pieces of chicken used by each type of order
def piecesChickenPasta := 2
def piecesBarbecueChicken := 3
def piecesFriedChickenDinner := 8

-- Defining the number of orders for Chicken Pasta and Barbecue Chicken
def numChickenPastaOrders := 6
def numBarbecueChickenOrders := 3

-- Defining the total pieces of chicken needed for all orders
def totalPiecesOfChickenNeeded := 37

-- Defining the number of pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPasta : Nat := piecesChickenPasta * numChickenPastaOrders
def piecesNeededBarbecueChicken : Nat := piecesBarbecueChicken * numBarbecueChickenOrders

-- Defining the total pieces of chicken needed for Chicken Pasta and Barbecue orders
def piecesNeededChickenPastaAndBarbecue : Nat := piecesNeededChickenPasta + piecesNeededBarbecueChicken

-- Calculating the pieces of chicken needed for Fried Chicken Dinner orders
def piecesNeededFriedChickenDinner : Nat := totalPiecesOfChickenNeeded - piecesNeededChickenPastaAndBarbecue

-- Defining the number of Fried Chicken Dinner orders
def numFriedChickenDinnerOrders : Nat := piecesNeededFriedChickenDinner / piecesFriedChickenDinner

-- Proving Victor has 2 Fried Chicken Dinner orders
theorem FriedChickenDinner_orders_count : numFriedChickenDinnerOrders = 2 := by
  unfold numFriedChickenDinnerOrders
  unfold piecesNeededFriedChickenDinner
  unfold piecesNeededChickenPastaAndBarbecue
  unfold piecesNeededBarbecueChicken
  unfold piecesNeededChickenPasta
  unfold totalPiecesOfChickenNeeded
  unfold numBarbecueChickenOrders
  unfold piecesBarbecueChicken
  unfold numChickenPastaOrders
  unfold piecesChickenPasta
  sorry

end NUMINAMATH_GPT_FriedChickenDinner_orders_count_l521_52183


namespace NUMINAMATH_GPT_chocolates_exceeding_200_l521_52189

-- Define the initial amount of chocolates
def initial_chocolates : ℕ := 3

-- Define the function that computes the amount of chocolates on the nth day
def chocolates_on_day (n : ℕ) : ℕ := initial_chocolates * 3 ^ (n - 1)

-- Define the proof problem
theorem chocolates_exceeding_200 : ∃ (n : ℕ), chocolates_on_day n > 200 :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_chocolates_exceeding_200_l521_52189


namespace NUMINAMATH_GPT_distance_between_x_intercepts_l521_52107

theorem distance_between_x_intercepts :
  let slope1 := 4
  let slope2 := -2
  let point := (8, 20)
  let line1 (x : ℝ) := slope1 * (x - point.1) + point.2
  let line2 (x : ℝ) := slope2 * (x - point.1) + point.2
  let x_intercept1 := (0 - point.2) / slope1 + point.1
  let x_intercept2 := (0 - point.2) / slope2 + point.1
  abs (x_intercept1 - x_intercept2) = 15 := sorry

end NUMINAMATH_GPT_distance_between_x_intercepts_l521_52107


namespace NUMINAMATH_GPT_negation_proposition_l521_52156

theorem negation_proposition {x : ℝ} (h : ∀ x > 0, Real.sin x > 0) : ∃ x > 0, Real.sin x ≤ 0 :=
sorry

end NUMINAMATH_GPT_negation_proposition_l521_52156


namespace NUMINAMATH_GPT_probability_single_trial_l521_52100

open Real

theorem probability_single_trial :
  ∃ p : ℝ, 0 ≤ p ∧ p ≤ 1 ∧ (1 - p)^4 = 16 / 81 ∧ p = 1 / 3 :=
by
  -- The proof steps have been skipped.
  sorry

end NUMINAMATH_GPT_probability_single_trial_l521_52100


namespace NUMINAMATH_GPT_quadratic_solution_exists_for_any_a_b_l521_52159

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_quadratic_solution_exists_for_any_a_b_l521_52159


namespace NUMINAMATH_GPT_projection_of_b_onto_a_l521_52135

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_squared := a.1 * a.1 + a.2 * a.2
  let scalar := dot_product / magnitude_squared
  (scalar * a.1, scalar * a.2)

theorem projection_of_b_onto_a :
  vector_projection (2, -1) (6, 2) = (4, -2) :=
by
  simp [vector_projection]
  sorry

end NUMINAMATH_GPT_projection_of_b_onto_a_l521_52135


namespace NUMINAMATH_GPT_second_part_of_sum_l521_52163

-- Defining the problem conditions
variables (x : ℚ)
def sum_parts := (2 * x) + (1/2 * x) + (1/4 * x)

theorem second_part_of_sum :
  sum_parts x = 104 →
  (1/2 * x) = 208 / 11 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_part_of_sum_l521_52163


namespace NUMINAMATH_GPT_system_sol_l521_52122

theorem system_sol {x y : ℝ} (h1 : x + 2 * y = -1) (h2 : 2 * x + y = 3) : x - y = 4 := by
  sorry

end NUMINAMATH_GPT_system_sol_l521_52122


namespace NUMINAMATH_GPT_inequality_solution_l521_52112

theorem inequality_solution :
  { x : ℝ | (x^3 - 4 * x) / (x^2 - 1) > 0 } = { x : ℝ | x < -2 ∨ (0 < x ∧ x < 1) ∨ 2 < x } :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l521_52112


namespace NUMINAMATH_GPT_fraction_identity_l521_52108

theorem fraction_identity (m n : ℕ) (h : (m : ℚ) / n = 3 / 7) : ((m + n) : ℚ) / n = 10 / 7 := 
sorry

end NUMINAMATH_GPT_fraction_identity_l521_52108


namespace NUMINAMATH_GPT_largest_divisor_of_5_consecutive_integers_l521_52117

theorem largest_divisor_of_5_consecutive_integers :
  ∃ d : ℤ, (∀ n : ℤ, d ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧ d = 120 :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_5_consecutive_integers_l521_52117


namespace NUMINAMATH_GPT_red_pairs_count_l521_52169

theorem red_pairs_count (blue_shirts red_shirts total_pairs blue_blue_pairs : ℕ)
  (h1 : blue_shirts = 63) 
  (h2 : red_shirts = 81) 
  (h3 : total_pairs = 72) 
  (h4 : blue_blue_pairs = 21)
  : (red_shirts - (blue_shirts - blue_blue_pairs * 2)) / 2 = 30 :=
by
  sorry

end NUMINAMATH_GPT_red_pairs_count_l521_52169


namespace NUMINAMATH_GPT_vectors_parallel_x_value_l521_52149

theorem vectors_parallel_x_value :
  ∀ (x : ℝ), (∀ a b : ℝ × ℝ, a = (2, 1) → b = (4, x+1) → (a.1 / b.1 = a.2 / b.2)) → x = 1 :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_vectors_parallel_x_value_l521_52149


namespace NUMINAMATH_GPT_average_weight_of_remaining_boys_l521_52132

theorem average_weight_of_remaining_boys :
  ∀ (total_boys remaining_boys_num : ℕ)
    (avg_weight_22 remaining_boys_avg_weight total_class_avg_weight : ℚ),
    total_boys = 30 →
    remaining_boys_num = total_boys - 22 →
    avg_weight_22 = 50.25 →
    total_class_avg_weight = 48.89 →
    (remaining_boys_num : ℚ) * remaining_boys_avg_weight =
    total_boys * total_class_avg_weight - 22 * avg_weight_22 →
    remaining_boys_avg_weight = 45.15 :=
by
  intros total_boys remaining_boys_num avg_weight_22 remaining_boys_avg_weight total_class_avg_weight
         h_total_boys h_remaining_boys_num h_avg_weight_22 h_total_class_avg_weight h_equation
  sorry

end NUMINAMATH_GPT_average_weight_of_remaining_boys_l521_52132


namespace NUMINAMATH_GPT_line_intersects_circle_l521_52155

theorem line_intersects_circle (a : ℝ) :
  ∃ (x y : ℝ), (y = a * x + 1) ∧ ((x - 1) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end NUMINAMATH_GPT_line_intersects_circle_l521_52155


namespace NUMINAMATH_GPT_positive_difference_largest_prime_factors_l521_52180

theorem positive_difference_largest_prime_factors :
  let p1 := 139
  let p2 := 29
  p1 - p2 = 110 := sorry

end NUMINAMATH_GPT_positive_difference_largest_prime_factors_l521_52180


namespace NUMINAMATH_GPT_arithmetic_mean_pq_l521_52137

variable (p q r : ℝ)

-- Definitions from conditions
def condition1 := (p + q) / 2 = 10
def condition2 := (q + r) / 2 = 26
def condition3 := r - p = 32

-- Theorem statement
theorem arithmetic_mean_pq : condition1 p q → condition2 q r → condition3 p r → (p + q) / 2 = 10 :=
by
  intros h1 h2 h3
  exact h1

end NUMINAMATH_GPT_arithmetic_mean_pq_l521_52137


namespace NUMINAMATH_GPT_smallest_number_first_digit_is_9_l521_52166

def sum_of_digits (n : Nat) : Nat :=
  (n.digits 10).sum

def first_digit (n : Nat) : Nat :=
  n.digits 10 |>.headD 0

theorem smallest_number_first_digit_is_9 :
  ∃ N : Nat, sum_of_digits N = 2020 ∧ ∀ M : Nat, (sum_of_digits M = 2020 → N ≤ M) ∧ first_digit N = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_first_digit_is_9_l521_52166


namespace NUMINAMATH_GPT_exists_subset_no_three_ap_l521_52185

-- Define the set S_n
def S (n : ℕ) : Finset ℕ := (Finset.range ((3^n + 1) / 2 + 1)).image (λ i => i + 1)

-- Define the property of no three elements forming an arithmetic progression
def no_three_form_ap (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a < b → b < c → 2 * b ≠ a + c

-- Define the theorem statement
theorem exists_subset_no_three_ap (n : ℕ) :
  ∃ M : Finset ℕ, M ⊆ S n ∧ M.card = 2^n ∧ no_three_form_ap M :=
sorry

end NUMINAMATH_GPT_exists_subset_no_three_ap_l521_52185


namespace NUMINAMATH_GPT_example_problem_l521_52152

def diamond (a b : ℕ) : ℕ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem example_problem : diamond 3 2 = 125 := by
  sorry

end NUMINAMATH_GPT_example_problem_l521_52152


namespace NUMINAMATH_GPT_varphi_le_one_varphi_l521_52143

noncomputable def f (a x : ℝ) := -a * Real.log x

-- Definition of the minimum value function φ for a > 0
noncomputable def varphi (a : ℝ) := -a * Real.log a

theorem varphi_le_one (a : ℝ) (h : 0 < a) : varphi a ≤ 1 := 
by sorry

theorem varphi'_le (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) : 
    (1 - Real.log a) ≤ (1 - Real.log b) := 
by sorry

end NUMINAMATH_GPT_varphi_le_one_varphi_l521_52143


namespace NUMINAMATH_GPT_remainder_when_summed_divided_by_15_l521_52160

theorem remainder_when_summed_divided_by_15 (k j : ℤ) (x y : ℤ)
  (hx : x = 60 * k + 47)
  (hy : y = 45 * j + 26) :
  (x + y) % 15 = 13 := 
sorry

end NUMINAMATH_GPT_remainder_when_summed_divided_by_15_l521_52160


namespace NUMINAMATH_GPT_molecular_weight_correct_l521_52115

noncomputable def molecular_weight_compound : ℝ :=
  (3 * 12.01) + (6 * 1.008) + (1 * 16.00)

theorem molecular_weight_correct :
  molecular_weight_compound = 58.078 := by
  sorry

end NUMINAMATH_GPT_molecular_weight_correct_l521_52115


namespace NUMINAMATH_GPT_find_k_l521_52128

def green_balls : ℕ := 7

noncomputable def probability_green (k : ℕ) : ℚ := green_balls / (green_balls + k)
noncomputable def probability_purple (k : ℕ) : ℚ := k / (green_balls + k)

noncomputable def winning_for_green : ℤ := 3
noncomputable def losing_for_purple : ℤ := -1

noncomputable def expected_value (k : ℕ) : ℚ :=
  (probability_green k) * (winning_for_green : ℚ) + (probability_purple k) * (losing_for_purple : ℚ)

theorem find_k (k : ℕ) (h : expected_value k = 1) : k = 7 :=
  sorry

end NUMINAMATH_GPT_find_k_l521_52128


namespace NUMINAMATH_GPT_animal_lifespan_probability_l521_52105

theorem animal_lifespan_probability
    (P_B : ℝ) (hP_B : P_B = 0.8)
    (P_A : ℝ) (hP_A : P_A = 0.4)
    : (P_A / P_B = 0.5) :=
by
    sorry

end NUMINAMATH_GPT_animal_lifespan_probability_l521_52105


namespace NUMINAMATH_GPT_vector_sum_length_l521_52196

open Real

noncomputable def vector_length (v : ℝ × ℝ) : ℝ :=
sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vector_angle_cosine (v w : ℝ × ℝ) : ℝ :=
dot_product v w / (vector_length v * vector_length w)

theorem vector_sum_length (a b : ℝ × ℝ)
  (ha : vector_length a = 2)
  (hb : vector_length b = 2)
  (hab_angle : vector_angle_cosine a b = cos (π / 3)):
  vector_length (a.1 + b.1, a.2 + b.2) = 2 * sqrt 3 :=
by sorry

end NUMINAMATH_GPT_vector_sum_length_l521_52196


namespace NUMINAMATH_GPT_volume_of_pyramid_l521_52103

theorem volume_of_pyramid 
  (QR RS : ℝ) (PT : ℝ) 
  (hQR_pos : 0 < QR) (hRS_pos : 0 < RS) (hPT_pos : 0 < PT)
  (perp1 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * QR) * (x * y) = 0)
  (perp2 : ∃ x : ℝ, ∀ y : ℝ, y ≠ x -> (PT * RS) * (x * y) = 0) :
  QR = 10 -> RS = 5 -> PT = 9 -> 
  (1/3) * QR * RS * PT = 150 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_pyramid_l521_52103


namespace NUMINAMATH_GPT_nancy_crayons_l521_52191

theorem nancy_crayons (packs : Nat) (crayons_per_pack : Nat) (total_crayons : Nat) 
  (h1 : packs = 41) (h2 : crayons_per_pack = 15) (h3 : total_crayons = packs * crayons_per_pack) : 
  total_crayons = 615 := by
  sorry

end NUMINAMATH_GPT_nancy_crayons_l521_52191


namespace NUMINAMATH_GPT_lasagna_pieces_l521_52168

theorem lasagna_pieces (m a k r l : ℕ → ℝ)
  (hm : m 1 = 1)                -- Manny's consumption
  (ha : a 0 = 0)                -- Aaron's consumption
  (hk : ∀ n, k n = 2 * (m 1))   -- Kai's consumption
  (hr : ∀ n, r n = (1 / 2) * (m 1)) -- Raphael's consumption
  (hl : ∀ n, l n = 2 + (r n))   -- Lisa's consumption
  : m 1 + a 0 + k 1 + r 1 + l 1 = 6 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_lasagna_pieces_l521_52168


namespace NUMINAMATH_GPT_profit_per_meter_correct_l521_52188

noncomputable def total_selling_price := 6788
noncomputable def num_meters := 78
noncomputable def cost_price_per_meter := 58.02564102564102
noncomputable def total_cost_price := 4526 -- rounded total
noncomputable def total_profit := 2262 -- calculated total profit
noncomputable def profit_per_meter := 29

theorem profit_per_meter_correct :
  (total_selling_price - total_cost_price) / num_meters = profit_per_meter :=
by
  sorry

end NUMINAMATH_GPT_profit_per_meter_correct_l521_52188


namespace NUMINAMATH_GPT_sequence_problem_l521_52172

theorem sequence_problem (a : ℕ → ℝ) (pos_terms : ∀ n, a n > 0)
  (h1 : a 1 = 2)
  (recurrence : ∀ n, (a n + 1) * a (n + 2) = 1)
  (h2 : a 2 = a 6) :
  a 11 + a 12 = (11 / 18) + ((Real.sqrt 5 - 1) / 2) := by
  sorry

end NUMINAMATH_GPT_sequence_problem_l521_52172


namespace NUMINAMATH_GPT_sin_tan_identity_of_cos_eq_tan_identity_l521_52129

open Real

variable (α : ℝ)
variable (hα : α ∈ Ioo 0 π)   -- α is in the interval (0, π)
variable (hcos : cos (2 * α) = 2 * cos (α + π / 4))

theorem sin_tan_identity_of_cos_eq_tan_identity : 
  sin (2 * α) = 1 ∧ tan α = 1 :=
by
  sorry

end NUMINAMATH_GPT_sin_tan_identity_of_cos_eq_tan_identity_l521_52129


namespace NUMINAMATH_GPT_compute_expression_l521_52109

theorem compute_expression : 19 * 42 + 81 * 19 = 2337 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l521_52109


namespace NUMINAMATH_GPT_new_persons_joined_l521_52120

theorem new_persons_joined :
  ∀ (A : ℝ) (N : ℕ) (avg_new : ℝ) (avg_combined : ℝ), 
  N = 15 → avg_new = 15 → avg_combined = 15.5 → 1 = (N * avg_combined + N * avg_new - 232.5) / (avg_combined - avg_new) := by
  intros A N avg_new avg_combined
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_new_persons_joined_l521_52120


namespace NUMINAMATH_GPT_find_number_of_students_l521_52140

theorem find_number_of_students (N T : ℕ) 
  (avg_mark_all : T = 80 * N) 
  (avg_mark_exclude : (T - 150) / (N - 5) = 90) : 
  N = 30 := by
  sorry

end NUMINAMATH_GPT_find_number_of_students_l521_52140


namespace NUMINAMATH_GPT_find_AC_l521_52111

def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

theorem find_AC :
  let AB := (2, 3)
  let BC := (1, -4)
  vector_add AB BC = (3, -1) :=
by 
  sorry

end NUMINAMATH_GPT_find_AC_l521_52111


namespace NUMINAMATH_GPT_ratio_of_largest_to_smallest_root_in_geometric_progression_l521_52106

theorem ratio_of_largest_to_smallest_root_in_geometric_progression 
    (a b c d : ℝ) (r s t : ℝ) 
    (h_poly : 81 * r^3 - 243 * r^2 + 216 * r - 64 = 0)
    (h_geo_prog : r > 0 ∧ s > 0 ∧ t > 0 ∧ ∃ (k : ℝ),  k > 0 ∧ s = r * k ∧ t = s * k) :
    ∃ (k : ℝ), k = r^2 ∧ s = r * k ∧ t = s * k := 
sorry

end NUMINAMATH_GPT_ratio_of_largest_to_smallest_root_in_geometric_progression_l521_52106


namespace NUMINAMATH_GPT_minimum_value_of_expression_l521_52101

theorem minimum_value_of_expression (x y : ℝ) : 
    ∃ (x y : ℝ), (2 * x * y - 1) ^ 2 + (x - y) ^ 2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l521_52101


namespace NUMINAMATH_GPT_john_spending_l521_52181

open Nat Real

noncomputable def cost_of_silver (silver_ounce: Real) (silver_price: Real) : Real :=
  silver_ounce * silver_price

noncomputable def quantity_of_gold (silver_ounce: Real): Real :=
  2 * silver_ounce

noncomputable def cost_per_ounce_gold (silver_price: Real) (multiplier: Real): Real :=
  silver_price * multiplier

noncomputable def cost_of_gold (gold_ounce: Real) (gold_price: Real) : Real :=
  gold_ounce * gold_price

noncomputable def total_cost (cost_silver: Real) (cost_gold: Real): Real :=
  cost_silver + cost_gold

theorem john_spending :
  let silver_ounce := 1.5
  let silver_price := 20
  let gold_multiplier := 50
  let cost_silver := cost_of_silver silver_ounce silver_price
  let gold_ounce := quantity_of_gold silver_ounce
  let gold_price := cost_per_ounce_gold silver_price gold_multiplier
  let cost_gold := cost_of_gold gold_ounce gold_price
  let total := total_cost cost_silver cost_gold
  total = 3030 :=
by
  sorry

end NUMINAMATH_GPT_john_spending_l521_52181


namespace NUMINAMATH_GPT_alice_preferred_numbers_l521_52170

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

def is_not_multiple_of_3 (n : ℕ) : Prop :=
  ¬ (n % 3 = 0)

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def alice_pref_num (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 150 ∧ is_multiple_of_7 n ∧ is_not_multiple_of_3 n ∧ is_prime (digit_sum n)

theorem alice_preferred_numbers :
  ∀ n, alice_pref_num n ↔ n = 119 ∨ n = 133 ∨ n = 140 := 
sorry

end NUMINAMATH_GPT_alice_preferred_numbers_l521_52170


namespace NUMINAMATH_GPT_pencils_per_friend_l521_52186

theorem pencils_per_friend (total_pencils num_friends : ℕ) (h_total : total_pencils = 24) (h_friends : num_friends = 3) : total_pencils / num_friends = 8 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_pencils_per_friend_l521_52186


namespace NUMINAMATH_GPT_complex_coordinates_l521_52175

theorem complex_coordinates : (⟨(-1:ℝ), (-1:ℝ)⟩ : ℂ) = (⟨0,1⟩ : ℂ) * (⟨-2,0⟩ : ℂ) / (⟨1,1⟩ : ℂ) :=
by
  sorry

end NUMINAMATH_GPT_complex_coordinates_l521_52175


namespace NUMINAMATH_GPT_percentage_forgot_homework_l521_52146

def total_students_group_A : ℕ := 30
def total_students_group_B : ℕ := 50
def forget_percentage_A : ℝ := 0.20
def forget_percentage_B : ℝ := 0.12

theorem percentage_forgot_homework :
  let num_students_forgot_A := forget_percentage_A * total_students_group_A
  let num_students_forgot_B := forget_percentage_B * total_students_group_B
  let total_students_forgot := num_students_forgot_A + num_students_forgot_B
  let total_students := total_students_group_A + total_students_group_B
  let percentage_forgot := (total_students_forgot / total_students) * 100
  percentage_forgot = 15 := sorry

end NUMINAMATH_GPT_percentage_forgot_homework_l521_52146


namespace NUMINAMATH_GPT_log_ratio_l521_52162

noncomputable def log_base (b x : ℝ) := Real.log x / Real.log b

theorem log_ratio (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b)
  (h3 : log_base 4 a = log_base 6 b)
  (h4 : log_base 6 b = log_base 9 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := sorry

end NUMINAMATH_GPT_log_ratio_l521_52162


namespace NUMINAMATH_GPT_joey_return_speed_l521_52171

theorem joey_return_speed
    (h1: 1 = (2 : ℝ) / u)
    (h2: (4 : ℝ) / (1 + t) = 3)
    (h3: u = 2)
    (h4: t = 1 / 3) :
    (2 : ℝ) / t = 6 :=
by
  sorry

end NUMINAMATH_GPT_joey_return_speed_l521_52171


namespace NUMINAMATH_GPT_value_of_f_1985_l521_52164

def f : ℝ → ℝ := sorry -- Assuming the existence of f, let ℝ be the type of real numbers

-- Given condition as a hypothesis
axiom functional_eq (x y : ℝ) : f (x + y) = f (x^2) + f (2 * y)

-- The main theorem we want to prove
theorem value_of_f_1985 : f 1985 = 0 :=
by
  sorry

end NUMINAMATH_GPT_value_of_f_1985_l521_52164


namespace NUMINAMATH_GPT_sum_of_four_consecutive_even_integers_l521_52187

theorem sum_of_four_consecutive_even_integers (x : ℕ) (hx : x > 4) :
  (x - 4) * (x - 2) * x * (x + 2) = 48 * (4 * x) → (x - 4) + (x - 2) + x + (x + 2) = 28 := by
{
  sorry
}

end NUMINAMATH_GPT_sum_of_four_consecutive_even_integers_l521_52187


namespace NUMINAMATH_GPT_price_per_gallon_in_NC_l521_52182

variable (P : ℝ)
variable (price_nc := P) -- price per gallon in North Carolina
variable (price_va := P + 1) -- price per gallon in Virginia
variable (gallons_nc := 10) -- gallons bought in North Carolina
variable (gallons_va := 10) -- gallons bought in Virginia
variable (total_cost := 50) -- total amount spent on gas

theorem price_per_gallon_in_NC :
  (gallons_nc * price_nc) + (gallons_va * price_va) = total_cost → price_nc = 2 :=
by
  sorry

end NUMINAMATH_GPT_price_per_gallon_in_NC_l521_52182


namespace NUMINAMATH_GPT_polynomial_abc_value_l521_52141

theorem polynomial_abc_value (a b c : ℝ) (h : a * (x^2) + b * x + c = (x - 1) * (x - 2)) : a * b * c = -6 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_abc_value_l521_52141


namespace NUMINAMATH_GPT_value_of_a_minus_b_l521_52158

theorem value_of_a_minus_b (a b : ℤ) (h1 : 2020 * a + 2024 * b = 2040) (h2 : 2022 * a + 2026 * b = 2044) :
  a - b = 1002 :=
sorry

end NUMINAMATH_GPT_value_of_a_minus_b_l521_52158


namespace NUMINAMATH_GPT_hot_dogs_served_today_l521_52194

-- Define the number of hot dogs served during lunch
def h_dogs_lunch : ℕ := 9

-- Define the number of hot dogs served during dinner
def h_dogs_dinner : ℕ := 2

-- Define the total number of hot dogs served today
def total_h_dogs : ℕ := h_dogs_lunch + h_dogs_dinner

-- Theorem stating that the total number of hot dogs served today is 11
theorem hot_dogs_served_today : total_h_dogs = 11 := by
  sorry

end NUMINAMATH_GPT_hot_dogs_served_today_l521_52194


namespace NUMINAMATH_GPT_greatest_value_of_x_l521_52190

theorem greatest_value_of_x (x : ℝ) : 
  (∃ (M : ℝ), (∀ y : ℝ, (y ^ 2 - 14 * y + 45 <= 0) → y <= M) ∧ (M ^ 2 - 14 * M + 45 <= 0)) ↔ M = 9 :=
by
  sorry

end NUMINAMATH_GPT_greatest_value_of_x_l521_52190


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l521_52130

theorem ratio_of_x_intercepts (b : ℝ) (hb : b ≠ 0) (u v: ℝ)
  (h1: 0 = 8 * u + b) (h2: 0 = 4 * v + b) : u / v = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_ratio_of_x_intercepts_l521_52130


namespace NUMINAMATH_GPT_arithmetic_sum_nine_l521_52173

noncomputable def arithmetic_sequence_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a 1 + a n)

theorem arithmetic_sum_nine (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 4 = 9)
  (h3 : a 6 = 11) : arithmetic_sequence_sum a 9 = 90 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sum_nine_l521_52173


namespace NUMINAMATH_GPT_total_transaction_loss_l521_52151

-- Define the cost and selling prices given the conditions
def cost_price_house (h : ℝ) := (7 / 10) * h = 15000
def cost_price_store (s : ℝ) := (5 / 4) * s = 15000

-- Define the loss calculation for the transaction
def transaction_loss : Prop :=
  ∃ (h s : ℝ),
    (7 / 10) * h = 15000 ∧
    (5 / 4) * s = 15000 ∧
    h + s - 2 * 15000 = 3428.57

-- The theorem stating the transaction resulted in a loss of $3428.57
theorem total_transaction_loss : transaction_loss :=
by
  sorry

end NUMINAMATH_GPT_total_transaction_loss_l521_52151


namespace NUMINAMATH_GPT_dodecahedron_edge_coloring_l521_52174

-- Define the properties of the dodecahedron
structure Dodecahedron :=
  (faces : Fin 12)          -- 12 pentagonal faces
  (edges : Fin 30)         -- 30 edges
  (vertices : Fin 20)      -- 20 vertices
  (edge_faces : Fin 30 → Fin 2) -- Each edge contributes to two faces

-- Prove the number of valid edge colorations such that each face has an even number of red edges
theorem dodecahedron_edge_coloring : 
    (∃ num_colorings : ℕ, num_colorings = 2^11) :=
sorry

end NUMINAMATH_GPT_dodecahedron_edge_coloring_l521_52174


namespace NUMINAMATH_GPT_proof_expr_28_times_35_1003_l521_52179

theorem proof_expr_28_times_35_1003 :
  (5^1003 + 7^1004)^2 - (5^1003 - 7^1004)^2 = 28 * 35^1003 :=
by
  sorry

end NUMINAMATH_GPT_proof_expr_28_times_35_1003_l521_52179


namespace NUMINAMATH_GPT_polynomial_unique_f_g_l521_52114

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

theorem polynomial_unique_f_g :
  (∀ x : ℝ, (x^2 + x + 1) * f (x^2 - x + 1) = (x^2 - x + 1) * g (x^2 + x + 1)) →
  (∃ k : ℝ, ∀ x : ℝ, f x = k * x ∧ g x = k * x) :=
sorry

end NUMINAMATH_GPT_polynomial_unique_f_g_l521_52114


namespace NUMINAMATH_GPT_range_of_a1_l521_52150

theorem range_of_a1 (a : ℕ → ℝ) (h1 : ∀ n, a (n + 1) = 1 / (2 - a n)) (h2 : ∀ n, a (n + 1) > a n) :
  a 1 < 1 :=
sorry

end NUMINAMATH_GPT_range_of_a1_l521_52150


namespace NUMINAMATH_GPT_unique_scalar_matrix_l521_52127

theorem unique_scalar_matrix (N : Matrix (Fin 3) (Fin 3) ℝ) :
  (∀ v : Fin 3 → ℝ, Matrix.mulVec N v = 5 • v) → 
  N = !![5, 0, 0; 0, 5, 0; 0, 0, 5] :=
by
  intro hv
  sorry -- Proof omitted as per instructions

end NUMINAMATH_GPT_unique_scalar_matrix_l521_52127


namespace NUMINAMATH_GPT_tiles_difference_between_tenth_and_eleventh_square_l521_52153

-- Define the side length of the nth square
def side_length (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

-- Define the area of the nth square
def area (n : ℕ) : ℕ :=
  (side_length n) ^ 2

-- The math proof statement
theorem tiles_difference_between_tenth_and_eleventh_square : area 11 - area 10 = 88 :=
by 
  -- Proof goes here, but we use sorry to skip it for now
  sorry

end NUMINAMATH_GPT_tiles_difference_between_tenth_and_eleventh_square_l521_52153


namespace NUMINAMATH_GPT_range_of_m_l521_52134

def proposition_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2 * x + 2 ≥ m

def proposition_q (m : ℝ) : Prop :=
  ∀ x y : ℝ, x < y → -(7 - 3*m)^x > -(7 - 3*m)^y

theorem range_of_m (m : ℝ) :
  (proposition_p m ∧ ¬ proposition_q m) ∨ (¬ proposition_p m ∧ proposition_q m) ↔ (1 < m ∧ m < 2) :=
sorry

end NUMINAMATH_GPT_range_of_m_l521_52134


namespace NUMINAMATH_GPT_margin_in_terms_of_selling_price_l521_52192

variable (C S M : ℝ) (n : ℕ) (h : M = (1 / 2) * (S - (1 / n) * C))

theorem margin_in_terms_of_selling_price :
  M = ((n - 1) / (2 * n - 1)) * S :=
sorry

end NUMINAMATH_GPT_margin_in_terms_of_selling_price_l521_52192


namespace NUMINAMATH_GPT_discount_per_coupon_l521_52102

-- Definitions and conditions from the problem
def num_cans : ℕ := 9
def cost_per_can : ℕ := 175 -- in cents
def num_coupons : ℕ := 5
def total_payment : ℕ := 2000 -- $20 in cents
def change_received : ℕ := 550 -- $5.50 in cents
def amount_paid := total_payment - change_received

-- Mathematical proof problem
theorem discount_per_coupon :
  let total_cost_without_coupons := num_cans * cost_per_can 
  let total_discount := total_cost_without_coupons - amount_paid
  let discount_per_coupon := total_discount / num_coupons
  discount_per_coupon = 25 :=
by
  sorry

end NUMINAMATH_GPT_discount_per_coupon_l521_52102


namespace NUMINAMATH_GPT_not_necessarily_divisible_by_28_l521_52138

theorem not_necessarily_divisible_by_28 (k : ℤ) (h : 7 ∣ (k * (k + 1) * (k + 2))) : ¬ (28 ∣ (k * (k + 1) * (k + 2))) :=
sorry

end NUMINAMATH_GPT_not_necessarily_divisible_by_28_l521_52138


namespace NUMINAMATH_GPT_max_k_l521_52199

def A : Finset ℕ := {0,1,2,3,4,5,6,7,8,9}

def valid_collection (B : ℕ → Finset ℕ) (k : ℕ) : Prop :=
  ∀ i j : ℕ, i < k → j < k → i ≠ j → (B i ∩ B j).card ≤ 2

theorem max_k (B : ℕ → Finset ℕ) : ∃ k, valid_collection B k → k ≤ 175 := sorry

end NUMINAMATH_GPT_max_k_l521_52199


namespace NUMINAMATH_GPT_find_number_l521_52154

noncomputable def percentage_of (p : ℝ) (n : ℝ) := p / 100 * n

noncomputable def fraction_of (f : ℝ) (n : ℝ) := f * n

theorem find_number :
  ∃ x : ℝ, percentage_of 40 60 = fraction_of (4/5) x + 4 ∧ x = 25 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l521_52154


namespace NUMINAMATH_GPT_total_cost_computers_l521_52193

theorem total_cost_computers (B T : ℝ) 
  (cA : ℝ := 1.4 * B) 
  (cB : ℝ := B) 
  (tA : ℝ := T) 
  (tB : ℝ := T + 20) 
  (total_cost_A : ℝ := cA * tA)
  (total_cost_B : ℝ := cB * tB):
  total_cost_A = total_cost_B → 70 * B = total_cost_A := 
by
  sorry

end NUMINAMATH_GPT_total_cost_computers_l521_52193


namespace NUMINAMATH_GPT_money_lent_to_C_l521_52113

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

end NUMINAMATH_GPT_money_lent_to_C_l521_52113


namespace NUMINAMATH_GPT_nes_sale_price_l521_52142

noncomputable def price_of_nes
    (snes_value : ℝ)
    (tradein_rate : ℝ)
    (cash_given : ℝ)
    (change_received : ℝ)
    (game_value : ℝ) : ℝ :=
  let tradein_credit := snes_value * tradein_rate
  let additional_cost := cash_given - change_received
  let total_cost := tradein_credit + additional_cost
  let nes_price := total_cost - game_value
  nes_price

theorem nes_sale_price 
  (snes_value : ℝ)
  (tradein_rate : ℝ)
  (cash_given : ℝ)
  (change_received : ℝ)
  (game_value : ℝ) :
  snes_value = 150 → tradein_rate = 0.80 → cash_given = 80 → change_received = 10 → game_value = 30 →
  price_of_nes snes_value tradein_rate cash_given change_received game_value = 160 := by
  intros
  sorry

end NUMINAMATH_GPT_nes_sale_price_l521_52142


namespace NUMINAMATH_GPT_squirrel_pine_cones_l521_52144

theorem squirrel_pine_cones (x y : ℕ) (hx : 26 - 10 + 9 + (x + 14)/2 = x/2) (hy : y + 5 - 18 + 9 + (x + 14)/2 = x/2) :
  x = 86 := sorry

end NUMINAMATH_GPT_squirrel_pine_cones_l521_52144


namespace NUMINAMATH_GPT_roots_are_simplified_sqrt_form_l521_52165

theorem roots_are_simplified_sqrt_form : 
  ∃ m p n : ℕ, gcd m p = 1 ∧ gcd p n = 1 ∧ gcd m n = 1 ∧
    (∀ x : ℝ, (3 * x^2 - 8 * x + 1 = 0) ↔ 
    (x = (m : ℝ) + (Real.sqrt n)/(p : ℝ) ∨ x = (m : ℝ) - (Real.sqrt n)/(p : ℝ))) ∧
    n = 13 :=
by
  sorry

end NUMINAMATH_GPT_roots_are_simplified_sqrt_form_l521_52165


namespace NUMINAMATH_GPT_find_largest_x_l521_52184

theorem find_largest_x : 
  ∃ x : ℝ, (4 * x ^ 3 - 17 * x ^ 2 + x + 10 = 0) ∧ 
           (∀ y : ℝ, 4 * y ^ 3 - 17 * y ^ 2 + y + 10 = 0 → y ≤ x) ∧ 
           x = (25 + Real.sqrt 545) / 8 :=
sorry

end NUMINAMATH_GPT_find_largest_x_l521_52184


namespace NUMINAMATH_GPT_selection_schemes_l521_52125

theorem selection_schemes (boys girls : ℕ) (hb : boys = 4) (hg : girls = 2) :
  (boys * girls = 8) :=
by
  -- Proof goes here
  intros
  sorry

end NUMINAMATH_GPT_selection_schemes_l521_52125


namespace NUMINAMATH_GPT_a_eq_one_sufficient_not_necessary_P_subset_M_iff_l521_52119

open Set

-- Define sets P and M based on conditions
def P : Set ℝ := {x | x > 2}
def M (a : ℝ) : Set ℝ := {x | x > a}

-- State the theorem
theorem a_eq_one_sufficient_not_necessary (a : ℝ) : (a = 1) → (P ⊆ M a) := 
by
  sorry

theorem P_subset_M_iff (a : ℝ) : (P ⊆ M a) ↔ (a < 2) :=
by
  sorry

end NUMINAMATH_GPT_a_eq_one_sufficient_not_necessary_P_subset_M_iff_l521_52119


namespace NUMINAMATH_GPT_find_m_l521_52198

-- Definitions based on the conditions
def parabola (x y : ℝ) : Prop := y^2 = 2 * x
def symmetric_about_line (x1 y1 x2 y2 m : ℝ) : Prop := (y1 - y2) / (x1 - x2) = -1
def product_y (y1 y2 : ℝ) : Prop := y1 * y2 = -1 / 2

-- Theorem to be proven
theorem find_m 
  (x1 y1 x2 y2 m : ℝ)
  (h1 : parabola x1 y1)
  (h2 : parabola x2 y2)
  (h3 : symmetric_about_line x1 y1 x2 y2 m)
  (h4 : product_y y1 y2) :
  m = 9 / 4 :=
sorry

end NUMINAMATH_GPT_find_m_l521_52198
