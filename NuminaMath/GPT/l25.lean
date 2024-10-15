import Mathlib

namespace NUMINAMATH_GPT_opposite_sides_range_a_l25_2533

theorem opposite_sides_range_a (a: ℝ) :
  ((1 - 2 * a + 1) * (a + 4 + 1) < 0) ↔ (a < -5 ∨ a > 1) :=
by
  sorry

end NUMINAMATH_GPT_opposite_sides_range_a_l25_2533


namespace NUMINAMATH_GPT_number_of_combinations_with_constraints_l25_2534

theorem number_of_combinations_with_constraints :
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose n k
  let invalid_combinations := 2
  total_combinations - invalid_combinations = 13 :=
by
  let n : ℕ := 6
  let k : ℕ := 2
  let total_combinations := Nat.choose 6 2
  let invalid_combinations := 2
  show total_combinations - invalid_combinations = 13
  sorry

end NUMINAMATH_GPT_number_of_combinations_with_constraints_l25_2534


namespace NUMINAMATH_GPT_zamena_solution_l25_2574

def digit_assignment (A M E H Z N : ℕ) : Prop :=
  1 ≤ A ∧ A ≤ 5 ∧ 1 ≤ M ∧ M ≤ 5 ∧ 1 ≤ E ∧ E ≤ 5 ∧ 1 ≤ H ∧ H ≤ 5 ∧ 1 ≤ Z ∧ Z ≤ 5 ∧ 1 ≤ N ∧ N ≤ 5 ∧
  A ≠ M ∧ A ≠ E ∧ A ≠ H ∧ A ≠ Z ∧ A ≠ N ∧
  M ≠ E ∧ M ≠ H ∧ M ≠ Z ∧ M ≠ N ∧
  E ≠ H ∧ E ≠ Z ∧ E ≠ N ∧
  H ≠ Z ∧ H ≠ N ∧
  Z ≠ N ∧
  3 > A ∧ A > M ∧ M < E ∧ E < H ∧ H < A 

theorem zamena_solution : 
  ∃ (A M E H Z N : ℕ), digit_assignment A M E H Z N ∧ (Z * 100000 + A * 10000 + M * 1000 + E * 100 + N * 10 + A = 541234) := 
by
  sorry

end NUMINAMATH_GPT_zamena_solution_l25_2574


namespace NUMINAMATH_GPT_parabola_standard_form_l25_2566

theorem parabola_standard_form (a : ℝ) (x y : ℝ) :
  (∀ a : ℝ, (2 * a + 3) * x + y - 4 * a + 2 = 0 → 
  x = 2 ∧ y = -8) → 
  (y^2 = 32 * x ∨ x^2 = - (1/2) * y) :=
by 
  intros h
  sorry

end NUMINAMATH_GPT_parabola_standard_form_l25_2566


namespace NUMINAMATH_GPT_steak_and_egg_meal_cost_is_16_l25_2586

noncomputable def steak_and_egg_cost (x : ℝ) := 
  (x + 14) / 2 + 0.20 * (x + 14) = 21

theorem steak_and_egg_meal_cost_is_16 (x : ℝ) (h : steak_and_egg_cost x) : x = 16 := 
by 
  sorry

end NUMINAMATH_GPT_steak_and_egg_meal_cost_is_16_l25_2586


namespace NUMINAMATH_GPT_num_trombone_players_l25_2517

def weight_per_trumpet := 5
def weight_per_clarinet := 5
def weight_per_trombone := 10
def weight_per_tuba := 20
def weight_per_drum := 15

def num_trumpets := 6
def num_clarinets := 9
def num_tubas := 3
def num_drummers := 2
def total_weight := 245

theorem num_trombone_players : 
  let weight_trumpets := num_trumpets * weight_per_trumpet
  let weight_clarinets := num_clarinets * weight_per_clarinet
  let weight_tubas := num_tubas * weight_per_tuba
  let weight_drums := num_drummers * weight_per_drum
  let weight_others := weight_trumpets + weight_clarinets + weight_tubas + weight_drums
  let weight_trombones := total_weight - weight_others
  weight_trombones / weight_per_trombone = 8 :=
by
  sorry

end NUMINAMATH_GPT_num_trombone_players_l25_2517


namespace NUMINAMATH_GPT_exponential_inequality_l25_2535

theorem exponential_inequality (n : ℕ) (h : n ≥ 5) : 2^n > n^2 + 1 :=
sorry

end NUMINAMATH_GPT_exponential_inequality_l25_2535


namespace NUMINAMATH_GPT_widgets_made_per_week_l25_2541

theorem widgets_made_per_week
  (widgets_per_hour : Nat)
  (hours_per_day : Nat)
  (days_per_week : Nat)
  (total_widgets : Nat) :
  widgets_per_hour = 20 →
  hours_per_day = 8 →
  days_per_week = 5 →
  total_widgets = widgets_per_hour * hours_per_day * days_per_week →
  total_widgets = 800 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_widgets_made_per_week_l25_2541


namespace NUMINAMATH_GPT_distance_to_moscow_at_4PM_l25_2538

noncomputable def exact_distance_at_4PM (d12: ℝ) (d13: ℝ) (d15: ℝ) : ℝ :=
  d15 - 12

theorem distance_to_moscow_at_4PM  (h12 : 81.5 ≤ 82 ∧ 82 ≤ 82.5)
                                  (h13 : 70.5 ≤ 71 ∧ 71 ≤ 71.5)
                                  (h15 : 45.5 ≤ 46 ∧ 46 ≤ 46.5) :
  exact_distance_at_4PM 82 71 46 = 34 :=
by
  sorry

end NUMINAMATH_GPT_distance_to_moscow_at_4PM_l25_2538


namespace NUMINAMATH_GPT_residue_of_7_pow_2023_mod_19_l25_2571

theorem residue_of_7_pow_2023_mod_19 : (7^2023) % 19 = 3 :=
by 
  -- The main goal is to construct the proof that matches our explanation.
  sorry

end NUMINAMATH_GPT_residue_of_7_pow_2023_mod_19_l25_2571


namespace NUMINAMATH_GPT_total_ages_l25_2582

theorem total_ages (Xavier Yasmin : ℕ) (h1 : Xavier = 2 * Yasmin) (h2 : Xavier + 6 = 30) : Xavier + Yasmin = 36 :=
by
  sorry

end NUMINAMATH_GPT_total_ages_l25_2582


namespace NUMINAMATH_GPT_laptop_total_selling_price_l25_2557

-- Define the original price of the laptop
def originalPrice : ℝ := 1200

-- Define the discount rate
def discountRate : ℝ := 0.30

-- Define the redemption coupon amount
def coupon : ℝ := 50

-- Define the tax rate
def taxRate : ℝ := 0.15

-- Calculate the discount amount
def discountAmount : ℝ := originalPrice * discountRate

-- Calculate the sale price after discount
def salePrice : ℝ := originalPrice - discountAmount

-- Calculate the new sale price after applying the coupon
def newSalePrice : ℝ := salePrice - coupon

-- Calculate the tax amount
def taxAmount : ℝ := newSalePrice * taxRate

-- Calculate the total selling price after tax
def totalSellingPrice : ℝ := newSalePrice + taxAmount

-- Prove that the total selling price is 908.5 dollars
theorem laptop_total_selling_price : totalSellingPrice = 908.5 := by
  unfold totalSellingPrice newSalePrice taxAmount salePrice discountAmount
  norm_num
  sorry

end NUMINAMATH_GPT_laptop_total_selling_price_l25_2557


namespace NUMINAMATH_GPT_min_transport_cost_l25_2530

/- Definitions for the problem conditions -/
def villageA_vegetables : ℕ := 80
def villageB_vegetables : ℕ := 60
def destinationX_requirement : ℕ := 65
def destinationY_requirement : ℕ := 75

def cost_A_to_X : ℕ := 50
def cost_A_to_Y : ℕ := 30
def cost_B_to_X : ℕ := 60
def cost_B_to_Y : ℕ := 45

def W (x : ℕ) : ℕ :=
  cost_A_to_X * x +
  cost_A_to_Y * (villageA_vegetables - x) +
  cost_B_to_X * (destinationX_requirement - x) +
  cost_B_to_Y * (x - 5) + 6075 - 225

/- Prove that the minimum total cost W is 6100 -/
theorem min_transport_cost : ∃ (x : ℕ), 5 ≤ x ∧ x ≤ 65 ∧ W x = 6100 :=
by sorry

end NUMINAMATH_GPT_min_transport_cost_l25_2530


namespace NUMINAMATH_GPT_cost_of_two_dogs_l25_2558

theorem cost_of_two_dogs (original_price : ℤ) (profit_margin : ℤ) (num_dogs : ℤ) (final_price : ℤ) :
  original_price = 1000 →
  profit_margin = 30 →
  num_dogs = 2 →
  final_price = original_price + (profit_margin * original_price / 100) →
  num_dogs * final_price = 2600 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_two_dogs_l25_2558


namespace NUMINAMATH_GPT_alcohol_mixture_l25_2506

variable {a b c d : ℝ} (ha : a ≠ d) (hbc : d ≠ c)

theorem alcohol_mixture (hcd : a ≥ d ∧ d ≥ c ∨ a ≤ d ∧ d ≤ c) :
  x = b * (d - c) / (a - d) :=
by 
  sorry

end NUMINAMATH_GPT_alcohol_mixture_l25_2506


namespace NUMINAMATH_GPT_no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l25_2522

-- Part (a)
theorem no_six_digit_starting_with_five_12_digit_square : ∀ (x y : ℕ), (5 * 10^5 ≤ x) → (x < 6 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ¬∃ z : ℕ, (10^11 ≤ z) ∧ (z < 10^12) ∧ x * 10^6 + y = z^2 := sorry

-- Part (b)
theorem six_digit_starting_with_one_12_digit_square : ∀ (x y : ℕ), (10^5 ≤ x) → (x < 2 * 10^5) → (10^5 ≤ y) → (y < 10^6) → ∃ z : ℕ, (10^11 ≤ z) ∧ (z < 2 * 10^11) ∧ x * 10^6 + y = z^2 := sorry

-- Part (c)
theorem smallest_k_for_n_digit_number_square : ∀ (n : ℕ), ∃ (k : ℕ), k = n + 1 ∧ ∀ (x : ℕ), (10^(n-1) ≤ x) → (x < 10^n) → ∃ y : ℕ, (10^(n + k - 1) ≤ x * 10^k + y) ∧ (x * 10^k + y) < 10^(n + k) ∧ ∃ z : ℕ, x * 10^k + y = z^2 := sorry

end NUMINAMATH_GPT_no_six_digit_starting_with_five_12_digit_square_six_digit_starting_with_one_12_digit_square_smallest_k_for_n_digit_number_square_l25_2522


namespace NUMINAMATH_GPT_construct_rectangle_l25_2547

structure Rectangle where
  side1 : ℝ
  side2 : ℝ
  diagonal : ℝ
  sum_diag_side : ℝ := side2 + diagonal

theorem construct_rectangle (b a d : ℝ) (r : Rectangle) :
  r.side2 = a ∧ r.side1 = b ∧ r.sum_diag_side = a + d :=
by
  sorry

end NUMINAMATH_GPT_construct_rectangle_l25_2547


namespace NUMINAMATH_GPT_smallest_area_right_triangle_l25_2592

theorem smallest_area_right_triangle {a b : ℕ} (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (A : ℝ), A = 6 * Real.sqrt 7 :=
sorry

end NUMINAMATH_GPT_smallest_area_right_triangle_l25_2592


namespace NUMINAMATH_GPT_sum_of_arithmetic_seq_minimum_value_n_equals_5_l25_2509

variable {a : ℕ → ℝ} -- Define a sequence of real numbers
variable {S : ℕ → ℝ} -- Define the sum function for the sequence

-- Assume conditions
axiom a3_a8_neg : a 3 + a 8 < 0
axiom S11_pos : S 11 > 0

-- Prove the minimum value of S_n occurs at n = 5
theorem sum_of_arithmetic_seq_minimum_value_n_equals_5 :
  ∃ n, (∀ m < 5, S m ≥ S n) ∧ (∀ m > 5, S m > S n) ∧ n = 5 :=
sorry

end NUMINAMATH_GPT_sum_of_arithmetic_seq_minimum_value_n_equals_5_l25_2509


namespace NUMINAMATH_GPT_problem_1_problem_2_l25_2518

-- Definitions for set A and B when a = 3 for (1)
def A : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }
def B : Set ℝ := { x | x^2 - 5 * x + 6 ≤ 0 }

-- Theorem for (1)
theorem problem_1 : A ∪ (Bᶜ) = Set.univ := sorry

-- Function to describe B based on a for (2)
def B_a (a : ℝ) : Set ℝ := { x | x^2 - (a + 2) * x + 2 * a ≤ 0 }
def A_set : Set ℝ := { x | -x^2 + 5 * x - 4 > 0 }

-- Theorem for (2)
theorem problem_2 (a : ℝ) : (1 < a ∧ a < 4) → (A_set ∩ B_a a ≠ ∅ ∧ B_a a ⊆ A_set ∧ B_a a ≠ A_set) := sorry

end NUMINAMATH_GPT_problem_1_problem_2_l25_2518


namespace NUMINAMATH_GPT_farm_field_ploughing_l25_2578

theorem farm_field_ploughing (A D : ℕ) 
  (h1 : ∀ farmerA_initial_capacity: ℕ, farmerA_initial_capacity = 120)
  (h2 : ∀ farmerB_initial_capacity: ℕ, farmerB_initial_capacity = 100)
  (h3 : ∀ farmerA_adjustment: ℕ, farmerA_adjustment = 10)
  (h4 : ∀ farmerA_reduced_capacity: ℕ, farmerA_reduced_capacity = farmerA_initial_capacity - (farmerA_adjustment * farmerA_initial_capacity / 100))
  (h5 : ∀ farmerB_reduced_capacity: ℕ, farmerB_reduced_capacity = 90)
  (h6 : ∀ extra_days: ℕ, extra_days = 3)
  (h7 : ∀ remaining_hectares: ℕ, remaining_hectares = 60)
  (h8 : ∀ initial_combined_effort: ℕ, initial_combined_effort = (farmerA_initial_capacity + farmerB_initial_capacity) * D)
  (h9 : ∀ total_combined_effort: ℕ, total_combined_effort = (farmerA_reduced_capacity + farmerB_reduced_capacity) * (D + extra_days))
  (h10 : ∀ area_covered: ℕ, area_covered = total_combined_effort + remaining_hectares)
  : initial_combined_effort = A ∧ D = 30 ∧ A = 6600 :=
by
  sorry

end NUMINAMATH_GPT_farm_field_ploughing_l25_2578


namespace NUMINAMATH_GPT_baker_cakes_l25_2540

theorem baker_cakes (P x : ℝ) (h1 : P * x = 320) (h2 : 0.80 * P * (x + 2) = 320) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_baker_cakes_l25_2540


namespace NUMINAMATH_GPT_find_B_l25_2537

-- Define the polynomial function and its properties
def polynomial (z : ℤ) (A B : ℤ) : ℤ :=
  z^4 - 6 * z^3 + A * z^2 + B * z + 9

-- Prove that B = -9 under the given conditions
theorem find_B (A B : ℤ) (r1 r2 r3 r4 : ℤ)
  (h1 : polynomial r1 A B = 0)
  (h2 : polynomial r2 A B = 0)
  (h3 : polynomial r3 A B = 0)
  (h4 : polynomial r4 A B = 0)
  (h5 : r1 + r2 + r3 + r4 = 6)
  (h6 : r1 > 0)
  (h7 : r2 > 0)
  (h8 : r3 > 0)
  (h9 : r4 > 0) :
  B = -9 :=
by
  sorry

end NUMINAMATH_GPT_find_B_l25_2537


namespace NUMINAMATH_GPT_total_length_of_board_l25_2549

-- Define variables for the lengths
variable (S L : ℝ)

-- Given conditions as Lean definitions
def condition1 : Prop := 2 * S = L + 4
def condition2 : Prop := S = 8.0

-- The goal is to prove the total length of the board is 20.0 feet
theorem total_length_of_board (h1 : condition1 S L) (h2 : condition2 S) : S + L = 20.0 := by
  sorry

end NUMINAMATH_GPT_total_length_of_board_l25_2549


namespace NUMINAMATH_GPT_find_angle_B_find_sin_C_l25_2551

-- Statement for proving B = π / 4 given the conditions
theorem find_angle_B (a b c : ℝ) (A B C : ℝ) 
  (h1 : a * Real.sin A + c * Real.sin C - Real.sqrt 2 * a * Real.sin C = b * Real.sin B) 
  (hABC : A + B + C = Real.pi) :
  B = Real.pi / 4 := 
sorry

-- Statement for proving sin C when cos A = 1 / 3
theorem find_sin_C (A C : ℝ) 
  (hA : Real.cos A = 1 / 3)
  (hABC : A + Real.pi / 4 + C = Real.pi) :
  Real.sin C = (4 + Real.sqrt 2) / 6 := 
sorry

end NUMINAMATH_GPT_find_angle_B_find_sin_C_l25_2551


namespace NUMINAMATH_GPT_find_a_l25_2595

noncomputable def angle := 30 * Real.pi / 180 -- In radians

noncomputable def tan_angle : ℝ := Real.tan angle

theorem find_a (a : ℝ) (h1 : tan_angle = 1 / Real.sqrt 3) : 
  x - a * y + 3 = 0 → a = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l25_2595


namespace NUMINAMATH_GPT_sum_of_two_integers_l25_2515

theorem sum_of_two_integers (x y : ℕ) (h₁ : x^2 + y^2 = 145) (h₂ : x * y = 40) : x + y = 15 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_sum_of_two_integers_l25_2515


namespace NUMINAMATH_GPT_math_proof_equiv_l25_2532

def A := 5
def B := 3
def C := 2
def D := 0
def E := 0
def F := 1
def G := 0

theorem math_proof_equiv : (A * 1000 + B * 100 + C * 10 + D) + (E * 100 + F * 10 + G) = 5300 :=
by
  sorry

end NUMINAMATH_GPT_math_proof_equiv_l25_2532


namespace NUMINAMATH_GPT_solve_quadratic_eq_l25_2508

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l25_2508


namespace NUMINAMATH_GPT_inequality_solution_l25_2526

variable {x : ℝ}

theorem inequality_solution (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) : 
  (2 * Real.cos x ≤ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| 
  ∧ |Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))| ≤ Real.sqrt 2) ↔ 
  (Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4) :=
sorry

end NUMINAMATH_GPT_inequality_solution_l25_2526


namespace NUMINAMATH_GPT_solve_puzzle_l25_2562

theorem solve_puzzle
  (EH OY AY OH : ℕ)
  (h1 : EH = 4 * OY)
  (h2 : AY = 4 * OH) :
  EH + OY + AY + OH = 150 :=
sorry

end NUMINAMATH_GPT_solve_puzzle_l25_2562


namespace NUMINAMATH_GPT_hawks_points_l25_2525

theorem hawks_points (E H : ℕ) (h1 : E + H = 82) (h2 : E = H + 6) : H = 38 :=
sorry

end NUMINAMATH_GPT_hawks_points_l25_2525


namespace NUMINAMATH_GPT_log_tangent_ratio_l25_2511

open Real

theorem log_tangent_ratio (α β : ℝ) 
  (h1 : sin (α + β) = 1 / 2) 
  (h2 : sin (α - β) = 1 / 3) : 
  log 5 * (tan α / tan β) = 1 := 
sorry

end NUMINAMATH_GPT_log_tangent_ratio_l25_2511


namespace NUMINAMATH_GPT_sales_difference_l25_2504
noncomputable def max_min_difference (sales : List ℕ) : ℕ :=
  (sales.maximum.getD 0) - (sales.minimum.getD 0)

theorem sales_difference :
  max_min_difference [1200, 1450, 1950, 1700] = 750 :=
by
  sorry

end NUMINAMATH_GPT_sales_difference_l25_2504


namespace NUMINAMATH_GPT_piglet_balloons_l25_2569

theorem piglet_balloons (n w o total_balloons: ℕ) (H1: w = 2 * n) (H2: o = 4 * n) (H3: n + w + o = total_balloons) (H4: total_balloons = 44) : n - (7 * n - total_balloons) = 2 :=
by
  sorry

end NUMINAMATH_GPT_piglet_balloons_l25_2569


namespace NUMINAMATH_GPT_contrapositive_example_l25_2581

theorem contrapositive_example (a b : ℝ) (h : a^2 + b^2 < 4) : a + b ≠ 3 :=
sorry

end NUMINAMATH_GPT_contrapositive_example_l25_2581


namespace NUMINAMATH_GPT_students_pass_both_subjects_l25_2528

theorem students_pass_both_subjects
  (F_H F_E F_HE : ℝ)
  (h1 : F_H = 0.25)
  (h2 : F_E = 0.48)
  (h3 : F_HE = 0.27) :
  (100 - (F_H + F_E - F_HE) * 100) = 54 :=
by
  sorry

end NUMINAMATH_GPT_students_pass_both_subjects_l25_2528


namespace NUMINAMATH_GPT_calculate_expression_l25_2584

theorem calculate_expression :
  2 * Real.sin (60 * Real.pi / 180) + abs (Real.sqrt 3 - 3) + (Real.pi - 1)^0 = 4 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l25_2584


namespace NUMINAMATH_GPT_range_of_a_l25_2524

theorem range_of_a (a b c : ℝ) 
  (h1 : a^2 - b*c - 8*a + 7 = 0)
  (h2 : b^2 + c^2 + b*c - 6*a + 6 = 0) : 
  1 ≤ a ∧ a ≤ 9 :=
sorry

end NUMINAMATH_GPT_range_of_a_l25_2524


namespace NUMINAMATH_GPT_classify_event_l25_2516

-- Define the conditions of the problem
def involves_variables_and_uncertainties (event: String) : Prop := 
  event = "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'"

-- Define the type of event as a string
def event_type : String := "random"

-- The theorem to prove the classification of the event
theorem classify_event : involves_variables_and_uncertainties "Turning on the TV and broadcasting 'Chinese Intangible Cultural Heritage'" →
  event_type = "random" :=
by
  intro h
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_classify_event_l25_2516


namespace NUMINAMATH_GPT_pencil_cost_is_correct_l25_2552

-- Defining the cost of a pen as x and the cost of a pencil as y in cents
def cost_of_pen_and_pencil (x y : ℕ) : Prop :=
  3 * x + 5 * y = 345 ∧ 4 * x + 2 * y = 280

-- Stating the theorem that proves y = 39
theorem pencil_cost_is_correct (x y : ℕ) (h : cost_of_pen_and_pencil x y) : y = 39 :=
by
  sorry

end NUMINAMATH_GPT_pencil_cost_is_correct_l25_2552


namespace NUMINAMATH_GPT_tangent_slope_of_circle_l25_2521

theorem tangent_slope_of_circle {x1 y1 x2 y2 : ℝ}
  (hx1 : x1 = 1) (hy1 : y1 = 1) (hx2 : x2 = 6) (hy2 : y2 = 4) :
  ∀ m : ℝ, m = -5 / 3 ↔
    (∃ (r : ℝ), r = (y2 - y1) / (x2 - x1) ∧ m = -1 / r) :=
by
  sorry

end NUMINAMATH_GPT_tangent_slope_of_circle_l25_2521


namespace NUMINAMATH_GPT_union_M_N_l25_2543

def M : Set ℝ := { x | x^2 - x = 0 }
def N : Set ℝ := { y | y^2 + y = 0 }

theorem union_M_N : (M ∪ N) = {-1, 0, 1} := 
by 
  sorry

end NUMINAMATH_GPT_union_M_N_l25_2543


namespace NUMINAMATH_GPT_rulers_left_l25_2585

variable (rulers_in_drawer : Nat)
variable (rulers_taken : Nat)

theorem rulers_left (h1 : rulers_in_drawer = 46) (h2 : rulers_taken = 25) : 
  rulers_in_drawer - rulers_taken = 21 := by
  sorry

end NUMINAMATH_GPT_rulers_left_l25_2585


namespace NUMINAMATH_GPT_ratio_x_y_z_w_l25_2583

theorem ratio_x_y_z_w (x y z w : ℝ) 
(h1 : 0.10 * x = 0.20 * y)
(h2 : 0.30 * y = 0.40 * z)
(h3 : 0.50 * z = 0.60 * w) : 
  (x / w) = 8 
  ∧ (y / w) = 4 
  ∧ (z / w) = 3
  ∧ (w / w) = 2.5 := 
sorry

end NUMINAMATH_GPT_ratio_x_y_z_w_l25_2583


namespace NUMINAMATH_GPT_max_expression_value_l25_2573

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end NUMINAMATH_GPT_max_expression_value_l25_2573


namespace NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l25_2590

-- Proving the first calculation: 3 * 232 + 456 = 1152
theorem calculation1 : 3 * 232 + 456 = 1152 := 
by 
  sorry

-- Proving the second calculation: 760 * 5 - 2880 = 920
theorem calculation2 : 760 * 5 - 2880 = 920 :=
by 
  sorry

-- Proving the third calculation: 805 / 7 = 115 (integer division)
theorem calculation3 : 805 / 7 = 115 :=
by 
  sorry

-- Proving the fourth calculation: 45 + 255 / 5 = 96
theorem calculation4 : 45 + 255 / 5 = 96 :=
by 
  sorry

end NUMINAMATH_GPT_calculation1_calculation2_calculation3_calculation4_l25_2590


namespace NUMINAMATH_GPT_maximum_smallest_angle_l25_2554

-- Definition of points on the plane
structure Point2D :=
  (x : ℝ)
  (y : ℝ)

-- Function to calculate the angle between three points (p1, p2, p3)
def angle (p1 p2 p3 : Point2D) : ℝ := 
  -- Placeholder for the actual angle calculation
  sorry

-- Condition: Given five points on a plane
variables (A B C D E : Point2D)

-- Maximum value of the smallest angle formed by any triple is 36 degrees
theorem maximum_smallest_angle :
  ∃ α : ℝ, (∀ p1 p2 p3 : Point2D, α ≤ angle p1 p2 p3) ∧ α = 36 :=
sorry

end NUMINAMATH_GPT_maximum_smallest_angle_l25_2554


namespace NUMINAMATH_GPT_tangent_circle_line_radius_l25_2596

theorem tangent_circle_line_radius (m : ℝ) :
  (∀ x y : ℝ, (x - 1)^2 + y^2 = m → x + y = 1 → dist (1, 0) (x, y) = Real.sqrt m) →
  m = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_tangent_circle_line_radius_l25_2596


namespace NUMINAMATH_GPT_inequality_solution_l25_2572

theorem inequality_solution (x : ℝ) : 
  (0 < x ∧ x ≤ 3) ∨ (4 ≤ x) ↔ (3 * (x - 3) * (x - 4)) / x ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_solution_l25_2572


namespace NUMINAMATH_GPT_triangle_angle_identity_l25_2570

theorem triangle_angle_identity
  (α β γ : ℝ)
  (h_triangle : α + β + γ = π)
  (sin_α_ne_zero : Real.sin α ≠ 0)
  (sin_β_ne_zero : Real.sin β ≠ 0)
  (sin_γ_ne_zero : Real.sin γ ≠ 0) :
  (Real.cos α / (Real.sin β * Real.sin γ) +
   Real.cos β / (Real.sin α * Real.sin γ) +
   Real.cos γ / (Real.sin α * Real.sin β) = 2) := by
  sorry

end NUMINAMATH_GPT_triangle_angle_identity_l25_2570


namespace NUMINAMATH_GPT_exists_distinct_pure_powers_l25_2512

-- Definitions and conditions
def is_pure_kth_power (k m : ℕ) : Prop := ∃ t : ℕ, m = t ^ k

-- The main theorem statement
theorem exists_distinct_pure_powers (n : ℕ) (hn : 0 < n) :
  ∃ (a : Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → a i ≠ a j) ∧ 
    is_pure_kth_power 2009 (Finset.univ.sum a) ∧ 
    is_pure_kth_power 2010 (Finset.univ.prod a) :=
sorry

end NUMINAMATH_GPT_exists_distinct_pure_powers_l25_2512


namespace NUMINAMATH_GPT_find_x_l25_2510

theorem find_x (x : ℝ) (h : 9 / (x + 4) = 1) : x = 5 :=
sorry

end NUMINAMATH_GPT_find_x_l25_2510


namespace NUMINAMATH_GPT_subtraction_problem_l25_2545

variable (x : ℕ) -- Let's assume x is a natural number for this problem

theorem subtraction_problem (h : x - 46 = 15) : x - 29 = 32 := 
by 
  sorry -- Proof to be filled in

end NUMINAMATH_GPT_subtraction_problem_l25_2545


namespace NUMINAMATH_GPT_regular_dodecahedron_edges_l25_2559

-- Define a regular dodecahedron as a type
inductive RegularDodecahedron : Type
| mk : RegularDodecahedron

-- Define a function that returns the number of edges for a regular dodecahedron
def numberOfEdges (d : RegularDodecahedron) : Nat :=
  30

-- The mathematical statement to be proved
theorem regular_dodecahedron_edges (d : RegularDodecahedron) : numberOfEdges d = 30 := by
  sorry

end NUMINAMATH_GPT_regular_dodecahedron_edges_l25_2559


namespace NUMINAMATH_GPT_order_of_a_b_c_l25_2505

noncomputable def a := Real.sqrt 3 - Real.sqrt 2
noncomputable def b := Real.sqrt 6 - Real.sqrt 5
noncomputable def c := Real.sqrt 7 - Real.sqrt 6

theorem order_of_a_b_c : a > b ∧ b > c :=
by
  sorry

end NUMINAMATH_GPT_order_of_a_b_c_l25_2505


namespace NUMINAMATH_GPT_f_is_decreasing_max_k_value_l25_2553

noncomputable def f (x : ℝ) : ℝ := (1 + Real.log (x + 1)) / x

theorem f_is_decreasing : ∀ x > 0, (∃ y > x, f y < f x) :=
by
  sorry

theorem max_k_value : ∃ k : ℕ, (∀ x > 0, f x > k / (x + 1)) ∧ k = 3 :=
by
  sorry

end NUMINAMATH_GPT_f_is_decreasing_max_k_value_l25_2553


namespace NUMINAMATH_GPT_volume_conversion_l25_2568

theorem volume_conversion (a : Nat) (b : Nat) (c : Nat) (d : Nat) (e : Nat) (f : Nat)
  (h1 : a = 1) (h2 : b = 3) (h3 : c = a^3) (h4 : d = b^3) (h5 : c = 1) (h6 : d = 27) 
  (h7 : 1 = 1) (h8 : 27 = 27) (h9 : e = 5) 
  (h10 : f = e * d) : 
  f = 135 := 
sorry

end NUMINAMATH_GPT_volume_conversion_l25_2568


namespace NUMINAMATH_GPT_selling_price_of_cycle_l25_2536

theorem selling_price_of_cycle (original_price : ℝ) (loss_percentage : ℝ) (loss_amount : ℝ) (selling_price : ℝ) :
  original_price = 2000 →
  loss_percentage = 10 →
  loss_amount = (loss_percentage / 100) * original_price →
  selling_price = original_price - loss_amount →
  selling_price = 1800 :=
by
  intros
  sorry

end NUMINAMATH_GPT_selling_price_of_cycle_l25_2536


namespace NUMINAMATH_GPT_smallest_n_l25_2546

theorem smallest_n (n : ℕ) (h₁ : n > 2016) (h₂ : n % 4 = 0) : 
  ¬(1^n + 2^n + 3^n + 4^n) % 10 = 0 → n = 2020 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l25_2546


namespace NUMINAMATH_GPT_expand_and_simplify_l25_2503

theorem expand_and_simplify (x : ℝ) : 3 * (x - 3) * (x + 10) + 2 * x = 3 * x^2 + 23 * x - 90 :=
by
  sorry

end NUMINAMATH_GPT_expand_and_simplify_l25_2503


namespace NUMINAMATH_GPT_planted_fraction_correct_l25_2529

noncomputable def field_planted_fraction (leg1 leg2 : ℕ) (square_distance : ℕ) : ℚ :=
  let hypotenuse := Real.sqrt (leg1^2 + leg2^2)
  let total_area := (leg1 * leg2) / 2
  let square_side := square_distance
  let square_area := square_side^2
  let planted_area := total_area - square_area
  planted_area / total_area

theorem planted_fraction_correct :
  field_planted_fraction 5 12 4 = 367 / 375 :=
by
  sorry

end NUMINAMATH_GPT_planted_fraction_correct_l25_2529


namespace NUMINAMATH_GPT_max_volume_prism_l25_2513

theorem max_volume_prism (a b h : ℝ) (h_congruent_lateral : a = b) (sum_areas_eq_48 : a * h + b * h + a * b = 48) : 
  ∃ V : ℝ, V = 64 :=
by
  sorry

end NUMINAMATH_GPT_max_volume_prism_l25_2513


namespace NUMINAMATH_GPT_even_combinations_result_in_486_l25_2560

-- Define the operations possible (increase by 2, increase by 3, multiply by 2)
inductive Operation
| inc2
| inc3
| mul2

open Operation

-- Function to apply an operation to a number
def applyOperation : Operation → ℕ → ℕ
| inc2, n => n + 2
| inc3, n => n + 3
| mul2, n => n * 2

-- Function to apply a list of operations to the initial number 1
def applyOperationsList (ops : List Operation) : ℕ :=
ops.foldl (fun acc op => applyOperation op acc) 1

-- Count the number of combinations that result in an even number
noncomputable def evenCombosCount : ℕ :=
(List.replicate 6 [inc2, inc3, mul2]).foldl (fun acc x => acc * x.length) 1
  |> λ _ => (3 ^ 5) * 2

theorem even_combinations_result_in_486 :
  evenCombosCount = 486 :=
sorry

end NUMINAMATH_GPT_even_combinations_result_in_486_l25_2560


namespace NUMINAMATH_GPT_max_side_length_of_triangle_l25_2561

theorem max_side_length_of_triangle (a b c : ℕ) (h1 : a < b) (h2 : b < c) (h3 : a + b + c = 24) :
  a + b > c ∧ a + c > b ∧ b + c > a ∧ c = 11 :=
by sorry

end NUMINAMATH_GPT_max_side_length_of_triangle_l25_2561


namespace NUMINAMATH_GPT_maximum_rubles_l25_2579

-- We define the initial number of '1' and '2' cards
def num_ones : ℕ := 2013
def num_twos : ℕ := 2013
def total_digits : ℕ := num_ones + num_twos

-- Definition of the problem statement
def problem_statement : Prop :=
  ∃ (max_rubles : ℕ), 
    max_rubles = 5 ∧
    ∀ (current_k : ℕ), 
      current_k = 5 → 
      ∃ (moves : ℕ), 
        moves ≤ max_rubles ∧
        (current_k - moves * 2) % 11 = 0

-- The expected solution is proving the maximum rubles is 5
theorem maximum_rubles : problem_statement :=
by
  sorry

end NUMINAMATH_GPT_maximum_rubles_l25_2579


namespace NUMINAMATH_GPT_part_1_select_B_prob_part_2_select_BC_prob_l25_2507

-- Definitions for the four students
inductive Student
| A
| B
| C
| D

open Student

-- Definition for calculating probability
def probability (favorable total : Nat) : Rat :=
  favorable / total

-- Part (1)
theorem part_1_select_B_prob : probability 1 4 = 1 / 4 :=
  sorry

-- Part (2)
theorem part_2_select_BC_prob : probability 2 12 = 1 / 6 :=
  sorry

end NUMINAMATH_GPT_part_1_select_B_prob_part_2_select_BC_prob_l25_2507


namespace NUMINAMATH_GPT_find_eccentricity_of_ellipse_l25_2542

noncomputable def ellipseEccentricity (k : ℝ) : ℝ :=
  let a := Real.sqrt (k + 2)
  let b := Real.sqrt (k + 1)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem find_eccentricity_of_ellipse (k : ℝ) (h1 : k + 2 = 4) (h2 : Real.sqrt (k + 2) = 2) :
  ellipseEccentricity k = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_find_eccentricity_of_ellipse_l25_2542


namespace NUMINAMATH_GPT_distance_from_focus_to_asymptote_l25_2544

theorem distance_from_focus_to_asymptote
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : a = b)
  (h2 : |a| / Real.sqrt 2 = 2) :
  Real.sqrt 2 * 2 = 2 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_focus_to_asymptote_l25_2544


namespace NUMINAMATH_GPT_expected_value_X_correct_prob_1_red_ball_B_correct_l25_2555

-- Boxes configuration
structure BoxConfig where
  white_A : ℕ -- Number of white balls in box A
  red_A : ℕ -- Number of red balls in box A
  white_B : ℕ -- Number of white balls in box B
  red_B : ℕ -- Number of red balls in box B

-- Given the problem configuration
def initialConfig : BoxConfig := {
  white_A := 2,
  red_A := 2,
  white_B := 1,
  red_B := 3,
}

-- Define random variable X (number of red balls drawn from box A)
def prob_X (X : ℕ) (cfg : BoxConfig) : ℚ :=
  if X = 0 then 1 / 6
  else if X = 1 then 2 / 3
  else if X = 2 then 1 / 6
  else 0

-- Expected value of X
noncomputable def expected_value_X (cfg : BoxConfig) : ℚ :=
  0 * (prob_X 0 cfg) + 1 * (prob_X 1 cfg) + 2 * (prob_X 2 cfg)

-- Probability of drawing 1 red ball from box B
noncomputable def prob_1_red_ball_B (cfg : BoxConfig) (X : ℕ) : ℚ :=
  if X = 0 then 1 / 2
  else if X = 1 then 2 / 3
  else if X = 2 then 5 / 6
  else 0

-- Total probability of drawing 1 red ball from box B
noncomputable def total_prob_1_red_ball_B (cfg : BoxConfig) : ℚ :=
  (prob_X 0 cfg * (prob_1_red_ball_B cfg 0))
  + (prob_X 1 cfg * (prob_1_red_ball_B cfg 1))
  + (prob_X 2 cfg * (prob_1_red_ball_B cfg 2))


theorem expected_value_X_correct : expected_value_X initialConfig = 1 := by
  sorry

theorem prob_1_red_ball_B_correct : total_prob_1_red_ball_B initialConfig = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_expected_value_X_correct_prob_1_red_ball_B_correct_l25_2555


namespace NUMINAMATH_GPT_factorization_correct_l25_2514

theorem factorization_correct (x y : ℝ) : 
  x * (x - y) - y * (x - y) = (x - y) ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l25_2514


namespace NUMINAMATH_GPT_quadratic_solution_l25_2593

theorem quadratic_solution (x : ℝ) : x ^ 2 - 4 * x + 3 = 0 ↔ x = 1 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_solution_l25_2593


namespace NUMINAMATH_GPT_smallest_d_for_divisibility_by_3_l25_2539

def sum_of_digits (d : ℕ) : ℕ := 5 + 4 + 7 + d + 0 + 6

theorem smallest_d_for_divisibility_by_3 (d : ℕ) :
  (sum_of_digits 2) % 3 = 0 ∧ ∀ k, k < 2 → sum_of_digits k % 3 ≠ 0 := 
sorry

end NUMINAMATH_GPT_smallest_d_for_divisibility_by_3_l25_2539


namespace NUMINAMATH_GPT_original_price_l25_2567

theorem original_price (P : ℝ) (h1 : ∃ P : ℝ, (120 : ℝ) = P + 0.2 * P) : P = 100 :=
by
  obtain ⟨P, h⟩ := h1
  sorry

end NUMINAMATH_GPT_original_price_l25_2567


namespace NUMINAMATH_GPT_even_function_has_specific_m_l25_2520

theorem even_function_has_specific_m (m : ℝ) (f : ℝ → ℝ) (h_def : ∀ x : ℝ, f x = x^2 + (m - 1) * x - 3) (h_even : ∀ x : ℝ, f x = f (-x)) :
  m = 1 :=
by
  sorry

end NUMINAMATH_GPT_even_function_has_specific_m_l25_2520


namespace NUMINAMATH_GPT_dot_product_eq_l25_2550

def vector1 : ℝ × ℝ := (-3, 0)
def vector2 : ℝ × ℝ := (7, 9)

theorem dot_product_eq :
  (vector1.1 * vector2.1 + vector1.2 * vector2.2) = -21 :=
by
  sorry

end NUMINAMATH_GPT_dot_product_eq_l25_2550


namespace NUMINAMATH_GPT_find_y_l25_2597

theorem find_y (t y : ℝ) (h1 : -3 = 2 - t) (h2 : y = 4 * t + 7) : y = 27 :=
sorry

end NUMINAMATH_GPT_find_y_l25_2597


namespace NUMINAMATH_GPT_power_function_value_l25_2531

noncomputable def f (x : ℝ) : ℝ := x^2

theorem power_function_value :
  f 3 = 9 :=
by
  -- Since f(x) = x^2 and f passes through (-2, 4)
  -- f(x) = x^2, so f(3) = 3^2 = 9
  sorry

end NUMINAMATH_GPT_power_function_value_l25_2531


namespace NUMINAMATH_GPT_pet_store_initial_puppies_l25_2523

theorem pet_store_initial_puppies
  (sold: ℕ) (cages: ℕ) (puppies_per_cage: ℕ)
  (remaining_puppies: ℕ)
  (h1: sold = 30)
  (h2: cages = 6)
  (h3: puppies_per_cage = 8)
  (h4: remaining_puppies = cages * puppies_per_cage):
  (sold + remaining_puppies) = 78 :=
by
  sorry

end NUMINAMATH_GPT_pet_store_initial_puppies_l25_2523


namespace NUMINAMATH_GPT_sum_999_is_1998_l25_2502

theorem sum_999_is_1998 : 999 + 999 = 1998 :=
by
  sorry

end NUMINAMATH_GPT_sum_999_is_1998_l25_2502


namespace NUMINAMATH_GPT_decorations_per_box_l25_2565

-- Definitions based on given conditions
def used_decorations : ℕ := 35
def given_away_decorations : ℕ := 25
def number_of_boxes : ℕ := 4

-- Theorem stating the problem
theorem decorations_per_box : (used_decorations + given_away_decorations) / number_of_boxes = 15 := by
  sorry

end NUMINAMATH_GPT_decorations_per_box_l25_2565


namespace NUMINAMATH_GPT_members_playing_both_l25_2598

theorem members_playing_both
  (N B T Neither : ℕ)
  (hN : N = 40)
  (hB : B = 20)
  (hT : T = 18)
  (hNeither : Neither = 5) :
  (B + T) - (N - Neither) = 3 := by
-- to complete the proof
sorry

end NUMINAMATH_GPT_members_playing_both_l25_2598


namespace NUMINAMATH_GPT_grazing_area_of_goat_l25_2500

/-- 
Consider a circular park with a diameter of 50 feet, and a square monument with 10 feet on each side.
Sally ties her goat on one corner of the monument with a 20-foot rope. Calculate the total grazing area
around the monument considering the space limited by the park's boundary.
-/
theorem grazing_area_of_goat : 
  let park_radius := 25
  let monument_side := 10
  let rope_length := 20
  let monument_radius := monument_side / 2 
  let grazing_quarter_circle := (1 / 4) * Real.pi * rope_length^2
  let ungrazable_area := (1 / 4) * Real.pi * monument_radius^2
  grazing_quarter_circle - ungrazable_area = 93.75 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_grazing_area_of_goat_l25_2500


namespace NUMINAMATH_GPT_find_ratio_b_a_l25_2501

theorem find_ratio_b_a (a b : ℝ) 
  (h : ∀ x : ℝ, (2 * a - b) * x + (a + b) > 0 ↔ x > -3) : 
  b / a = 5 / 4 :=
sorry

end NUMINAMATH_GPT_find_ratio_b_a_l25_2501


namespace NUMINAMATH_GPT_sum_a_b_eq_5_l25_2587

theorem sum_a_b_eq_5 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * b = a - 2) (h4 : (-2)^2 = b * (2 * b + 2)) : a + b = 5 :=
sorry

end NUMINAMATH_GPT_sum_a_b_eq_5_l25_2587


namespace NUMINAMATH_GPT_hyeyoung_walked_correct_l25_2527

/-- The length of the promenade near Hyeyoung's house is 6 kilometers (km). -/
def promenade_length : ℕ := 6

/-- Hyeyoung walked from the starting point to the halfway point of the trail. -/
def hyeyoung_walked : ℕ := promenade_length / 2

/-- The distance Hyeyoung walked is 3 kilometers (km). -/
theorem hyeyoung_walked_correct : hyeyoung_walked = 3 := by
  sorry

end NUMINAMATH_GPT_hyeyoung_walked_correct_l25_2527


namespace NUMINAMATH_GPT_parallel_line_distance_l25_2580

theorem parallel_line_distance 
    (A_upper : ℝ) (A_middle : ℝ) (A_lower : ℝ)
    (A_total : ℝ) (A_half : ℝ)
    (h_upper : A_upper = 3)
    (h_middle : A_middle = 5)
    (h_lower : A_lower = 2) 
    (h_total : A_total = A_upper + A_middle + A_lower)
    (h_half : A_half = A_total / 2) :
    ∃ d : ℝ, d = 2 + 0.6 ∧ A_middle * 0.6 = 3 := 
sorry

end NUMINAMATH_GPT_parallel_line_distance_l25_2580


namespace NUMINAMATH_GPT_find_m_l25_2519

theorem find_m (x y m : ℤ) 
  (h1 : x + 2 * y = 5 * m) 
  (h2 : x - 2 * y = 9 * m) 
  (h3 : 3 * x + 2 * y = 19) : 
  m = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_m_l25_2519


namespace NUMINAMATH_GPT_inequality_subtraction_real_l25_2564

theorem inequality_subtraction_real (a b c : ℝ) (h : a > b) : a - c > b - c :=
sorry

end NUMINAMATH_GPT_inequality_subtraction_real_l25_2564


namespace NUMINAMATH_GPT_minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l25_2576

theorem minimum_value_of_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
sorry

theorem exists_x_y_for_minimum_value : ∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x + 3 * y = 5 * x * y ∧ 3 * x + 4 * y = 5 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_3x_plus_4y_exists_x_y_for_minimum_value_l25_2576


namespace NUMINAMATH_GPT_largest_possible_m_l25_2556

theorem largest_possible_m (x y : ℕ) (h1 : x > y) (hx : Nat.Prime x) (hy : Nat.Prime y) (hxy : x < 10) (hyy : y < 10) (h_prime_10xy : Nat.Prime (10 * x + y)) : ∃ m : ℕ, m = x * y * (10 * x + y) ∧ 1000 ≤ m ∧ m ≤ 9999 ∧ ∀ n : ℕ, (n = x * y * (10 * x + y) ∧ 1000 ≤ n ∧ n ≤ 9999) → n ≤ 1533 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_m_l25_2556


namespace NUMINAMATH_GPT_shell_highest_point_time_l25_2599

theorem shell_highest_point_time (a b c : ℝ) (h₁ : a ≠ 0)
  (h₂ : a * 7^2 + b * 7 + c = a * 14^2 + b * 14 + c) :
  (-b / (2 * a)) = 10.5 :=
by
  -- The proof is omitted as per the instructions
  sorry

end NUMINAMATH_GPT_shell_highest_point_time_l25_2599


namespace NUMINAMATH_GPT_sequence_v5_value_l25_2594

theorem sequence_v5_value (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) - v n)
  (h_v3 : v 3 = 17) (h_v6 : v 6 = 524) : v 5 = 198.625 :=
sorry

end NUMINAMATH_GPT_sequence_v5_value_l25_2594


namespace NUMINAMATH_GPT_min_diagonal_length_of_trapezoid_l25_2563

theorem min_diagonal_length_of_trapezoid (a b h d1 d2 : ℝ) 
  (h_area : a * h + b * h = 2)
  (h_diag : d1^2 + d2^2 = h^2 + (a + b)^2) 
  : d1 ≥ Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_diagonal_length_of_trapezoid_l25_2563


namespace NUMINAMATH_GPT_radius_of_larger_circle_l25_2591

theorem radius_of_larger_circle (r : ℝ) (R : ℝ) 
  (h1 : ∀ a b c : ℝ, a = 2 ∧ b = 2 ∧ c = 2) 
  (h2 : ∀ x y z : ℝ, (x = 4) ∧ (y = 4) ∧ (z = 4) ) 
  (h3 : ∀ A B : ℝ, A * 2 = 2) : 
  R = 2 + 2 * Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_larger_circle_l25_2591


namespace NUMINAMATH_GPT_work_completion_time_l25_2588

theorem work_completion_time (W : ℝ) : 
  let A_effort := 1 / 11
  let B_effort := 1 / 20
  let C_effort := 1 / 55
  (2 * A_effort + B_effort + C_effort) = 1 / 4 → 
  8 * (2 * A_effort + B_effort + C_effort) = 1 :=
by { sorry }

end NUMINAMATH_GPT_work_completion_time_l25_2588


namespace NUMINAMATH_GPT_sum_of_acute_angles_l25_2577

theorem sum_of_acute_angles (angle1 angle2 angle3 angle4 angle5 angle6 : ℝ)
  (h1 : angle1 = 30) (h2 : angle2 = 30) (h3 : angle3 = 30) (h4 : angle4 = 30) (h5 : angle5 = 30) (h6 : angle6 = 30) :
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + 
  (angle1 + angle2) + (angle2 + angle3) + (angle3 + angle4) + (angle4 + angle5) + (angle5 + angle6)) = 480 :=
  sorry

end NUMINAMATH_GPT_sum_of_acute_angles_l25_2577


namespace NUMINAMATH_GPT_gcd_of_polynomials_l25_2575

theorem gcd_of_polynomials (b : ℤ) (h : 2460 ∣ b) : 
  Int.gcd (b^2 + 6 * b + 30) (b + 5) = 30 :=
sorry

end NUMINAMATH_GPT_gcd_of_polynomials_l25_2575


namespace NUMINAMATH_GPT_domain_of_function_l25_2548

-- Definitions based on conditions
def function_domain (x : ℝ) : Prop := (x > -1) ∧ (x ≠ 1)

-- Prove the domain is the desired set
theorem domain_of_function :
  ∀ x, function_domain x ↔ ((-1 < x ∧ x < 1) ∨ (1 < x)) :=
  by
    sorry

end NUMINAMATH_GPT_domain_of_function_l25_2548


namespace NUMINAMATH_GPT_probability_at_least_one_admitted_l25_2589

-- Define the events and probabilities
variables (A B : Prop)
variables (P_A : ℝ) (P_B : ℝ)
variables (independent : Prop)

-- Assume the given conditions
def P_A_def : Prop := P_A = 0.6
def P_B_def : Prop := P_B = 0.7
def independent_def : Prop := independent = true  -- simplistic representation for independence

-- Statement: Prove the probability that at least one of them is admitted is 0.88
theorem probability_at_least_one_admitted : 
  P_A = 0.6 → P_B = 0.7 → independent = true →
  (1 - (1 - P_A) * (1 - P_B)) = 0.88 :=
by
  intros
  sorry

end NUMINAMATH_GPT_probability_at_least_one_admitted_l25_2589
