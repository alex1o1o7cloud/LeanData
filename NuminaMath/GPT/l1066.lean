import Mathlib

namespace NUMINAMATH_GPT_ratio_of_points_to_away_home_game_l1066_106681

-- Definitions
def first_away_game_points (A : ℕ) : ℕ := A
def second_away_game_points (A : ℕ) : ℕ := A + 18
def third_away_game_points (A : ℕ) : ℕ := A + 20
def last_home_game_points : ℕ := 62
def next_game_points : ℕ := 55
def total_points (A : ℕ) : ℕ := A + (A + 18) + (A + 20) + 62 + 55

-- Given that the total points should be four times the points of the last home game
def target_points : ℕ := 4 * 62

-- The main theorem to prove
theorem ratio_of_points_to_away_home_game : ∀ A : ℕ,
  total_points A = target_points → 62 = 2 * A :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_points_to_away_home_game_l1066_106681


namespace NUMINAMATH_GPT_volume_of_prism_l1066_106603

variables (a b c : ℝ)
variables (ab_prod : a * b = 36) (ac_prod : a * c = 48) (bc_prod : b * c = 72)

theorem volume_of_prism : a * b * c = 352.8 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_prism_l1066_106603


namespace NUMINAMATH_GPT_compound_interest_rate_is_10_percent_l1066_106638

theorem compound_interest_rate_is_10_percent
  (P : ℝ) (CI : ℝ) (t : ℝ) (A : ℝ) (n : ℝ) (r : ℝ)
  (hP : P = 4500) (hCI : CI = 945.0000000000009) (ht : t = 2) (hn : n = 1) (hA : A = P + CI)
  (h_eq : A = P * (1 + r / n)^(n * t)) :
  r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_rate_is_10_percent_l1066_106638


namespace NUMINAMATH_GPT_betty_min_sugar_flour_oats_l1066_106670

theorem betty_min_sugar_flour_oats :
  ∃ (s f o : ℕ), f ≥ 4 + 2 * s ∧ f ≤ 3 * s ∧ o = f + s ∧ s = 4 :=
by
  sorry

end NUMINAMATH_GPT_betty_min_sugar_flour_oats_l1066_106670


namespace NUMINAMATH_GPT_height_flagstaff_l1066_106665

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end NUMINAMATH_GPT_height_flagstaff_l1066_106665


namespace NUMINAMATH_GPT_find_original_cost_price_l1066_106647

theorem find_original_cost_price (C S C_new S_new : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) (h3 : S_new = S - 16.80) (h4 : S_new = 1.04 * C_new) : C = 80 :=
by
  sorry

end NUMINAMATH_GPT_find_original_cost_price_l1066_106647


namespace NUMINAMATH_GPT_usb_drive_available_space_l1066_106696

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end NUMINAMATH_GPT_usb_drive_available_space_l1066_106696


namespace NUMINAMATH_GPT_polynomial_solution_l1066_106682

theorem polynomial_solution (P : ℝ → ℝ) (h₀ : P 0 = 0) (h₁ : ∀ x : ℝ, P x = (1/2) * (P (x+1) + P (x-1))) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end NUMINAMATH_GPT_polynomial_solution_l1066_106682


namespace NUMINAMATH_GPT_students_exceed_pets_l1066_106664

-- Defining the conditions
def num_students_per_classroom := 25
def num_rabbits_per_classroom := 3
def num_guinea_pigs_per_classroom := 3
def num_classrooms := 5

-- Main theorem to prove
theorem students_exceed_pets:
  let total_students := num_students_per_classroom * num_classrooms
  let total_rabbits := num_rabbits_per_classroom * num_classrooms
  let total_guinea_pigs := num_guinea_pigs_per_classroom * num_classrooms
  let total_pets := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 :=
by 
  sorry

end NUMINAMATH_GPT_students_exceed_pets_l1066_106664


namespace NUMINAMATH_GPT_find_value_of_x_squared_and_reciprocal_squared_l1066_106642

theorem find_value_of_x_squared_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + (1/x)^2 = 2 := 
sorry

end NUMINAMATH_GPT_find_value_of_x_squared_and_reciprocal_squared_l1066_106642


namespace NUMINAMATH_GPT_intersection_points_parabola_l1066_106654

noncomputable def parabola : ℝ → ℝ := λ x => x^2

noncomputable def directrix : ℝ → ℝ := λ x => -1

noncomputable def other_line (m c : ℝ) : ℝ → ℝ := λ x => m * x + c

theorem intersection_points_parabola {m c : ℝ} (h1 : ∃ x1 x2 : ℝ, other_line m c x1 = parabola x1 ∧ other_line m c x2 = parabola x2) :
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 ≠ x2) → 
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 = x2) := 
by
  sorry

end NUMINAMATH_GPT_intersection_points_parabola_l1066_106654


namespace NUMINAMATH_GPT_probability_X_eq_Y_l1066_106612

-- Define the conditions as functions or predicates.
def is_valid_pair (x y : ℝ) : Prop :=
  -5 * Real.pi ≤ x ∧ x ≤ 5 * Real.pi ∧ -5 * Real.pi ≤ y ∧ y ≤ 5 * Real.pi ∧ Real.cos (Real.cos x) = Real.cos (Real.cos y)

-- Final statement asserting the required probability.
theorem probability_X_eq_Y :
  ∃ (prob : ℝ), prob = 1 / 11 ∧ ∀ (x y : ℝ), is_valid_pair x y → (x = y ∨ x ≠ y ∧ prob = 1/11) :=
  sorry

end NUMINAMATH_GPT_probability_X_eq_Y_l1066_106612


namespace NUMINAMATH_GPT_find_b_l1066_106677

theorem find_b (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = 3^n + b)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1))
  (h_geometric : ∃ r, ∀ n ≥ 1, a n = a 1 * r^(n-1)) : b = -1 := 
sorry

end NUMINAMATH_GPT_find_b_l1066_106677


namespace NUMINAMATH_GPT_find_x_of_perpendicular_l1066_106698

-- Definitions based on the conditions in a)
def a (x : ℝ) : ℝ × ℝ := (x, x + 1)
def b : ℝ × ℝ := (1, 2)

-- The mathematical proof problem in Lean 4 statement: prove that the dot product is zero implies x = -2/3
theorem find_x_of_perpendicular (x : ℝ) (h : (a x).fst * b.fst + (a x).snd * b.snd = 0) : x = -2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_find_x_of_perpendicular_l1066_106698


namespace NUMINAMATH_GPT_total_pencils_is_60_l1066_106684

def original_pencils : ℕ := 33
def added_pencils : ℕ := 27
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end NUMINAMATH_GPT_total_pencils_is_60_l1066_106684


namespace NUMINAMATH_GPT_find_b_l1066_106607

theorem find_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - 2 * a = 4) : b = 4 :=
sorry

end NUMINAMATH_GPT_find_b_l1066_106607


namespace NUMINAMATH_GPT_mrs_white_expected_yield_l1066_106622

noncomputable def orchard_yield : ℝ :=
  let length_in_feet : ℝ := 10 * 3
  let width_in_feet : ℝ := 30 * 3
  let total_area : ℝ := length_in_feet * width_in_feet
  let half_area : ℝ := total_area / 2
  let tomato_yield : ℝ := half_area * 0.75
  let cucumber_yield : ℝ := half_area * 0.4
  tomato_yield + cucumber_yield

theorem mrs_white_expected_yield :
  orchard_yield = 1552.5 := sorry

end NUMINAMATH_GPT_mrs_white_expected_yield_l1066_106622


namespace NUMINAMATH_GPT_find_valid_pairs_l1066_106697

noncomputable def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ d : ℕ, d ∣ n → d = 1 ∨ d = n)

def distinct_two_digit_primes : List (ℕ × ℕ) :=
  [(13, 53), (19, 47), (23, 43), (29, 37)]

def average (p q : ℕ) : ℕ := (p + q) / 2

def number1 (p q : ℕ) : ℕ := 100 * p + q
def number2 (p q : ℕ) : ℕ := 100 * q + p

theorem find_valid_pairs (p q : ℕ)
  (hp : is_prime p) (hq : is_prime q)
  (hpq : p ≠ q)
  (havg : average p q ∣ number1 p q ∧ average p q ∣ number2 p q) :
  (p, q) ∈ distinct_two_digit_primes ∨ (q, p) ∈ distinct_two_digit_primes :=
sorry

end NUMINAMATH_GPT_find_valid_pairs_l1066_106697


namespace NUMINAMATH_GPT_square_side_length_l1066_106667

theorem square_side_length (A : ℝ) (h : A = 25) : ∃ s : ℝ, s * s = A ∧ s = 5 :=
by
  sorry

end NUMINAMATH_GPT_square_side_length_l1066_106667


namespace NUMINAMATH_GPT_six_times_expression_l1066_106621

theorem six_times_expression {x y Q : ℝ} (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q :=
by
  sorry

end NUMINAMATH_GPT_six_times_expression_l1066_106621


namespace NUMINAMATH_GPT_rhombus_area_of_square_l1066_106669

theorem rhombus_area_of_square (h : ∀ (c : ℝ), c = 96) : ∃ (a : ℝ), a = 288 := 
by
  sorry

end NUMINAMATH_GPT_rhombus_area_of_square_l1066_106669


namespace NUMINAMATH_GPT_number_line_move_l1066_106627

theorem number_line_move (A B: ℤ):  A = -3 → B = A + 4 → B = 1 := by
  intros hA hB
  rw [hA] at hB
  rw [hB]
  sorry

end NUMINAMATH_GPT_number_line_move_l1066_106627


namespace NUMINAMATH_GPT_minimum_value_of_z_l1066_106613

theorem minimum_value_of_z 
  (x y : ℝ) 
  (h1 : x - 2 * y + 2 ≥ 0) 
  (h2 : 2 * x - y - 2 ≤ 0) 
  (h3 : y ≥ 0) :
  ∃ (z : ℝ), z = 3 * x + y ∧ z = -6 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_z_l1066_106613


namespace NUMINAMATH_GPT_cubic_polynomial_at_zero_l1066_106645

noncomputable def f (x : ℝ) : ℝ := by sorry

theorem cubic_polynomial_at_zero :
  (∃ f : ℝ → ℝ, f 2 = 15 ∨ f 2 = -15 ∧
                 f 4 = 15 ∨ f 4 = -15 ∧
                 f 5 = 15 ∨ f 5 = -15 ∧
                 f 6 = 15 ∨ f 6 = -15 ∧
                 f 8 = 15 ∨ f 8 = -15 ∧
                 f 9 = 15 ∨ f 9 = -15 ∧
                 ∀ x, ∃ c a b d, f x = c * x^3 + a * x^2 + b * x + d ) →
  |f 0| = 135 :=
by sorry

end NUMINAMATH_GPT_cubic_polynomial_at_zero_l1066_106645


namespace NUMINAMATH_GPT_time_to_cross_platform_l1066_106688

/-- Definitions of the conditions in the problem. -/
def train_length : ℕ := 1500
def platform_length : ℕ := 1800
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

/-- Proof statement: The time for the train to pass the platform. -/
theorem time_to_cross_platform : (total_distance / train_speed) = 220 := by
  sorry

end NUMINAMATH_GPT_time_to_cross_platform_l1066_106688


namespace NUMINAMATH_GPT_translation_correct_l1066_106632

def parabola1 (x : ℝ) : ℝ := -2 * (x + 2)^2 + 3
def parabola2 (x : ℝ) : ℝ := -2 * (x - 1)^2 - 1

theorem translation_correct :
  ∀ x : ℝ, parabola2 (x - 3) = parabola1 x - 4 :=
by
  sorry

end NUMINAMATH_GPT_translation_correct_l1066_106632


namespace NUMINAMATH_GPT_sqrt_floor_eq_l1066_106692

theorem sqrt_floor_eq (n : ℤ) (h : n ≥ 0) : 
  (⌊Real.sqrt n + Real.sqrt (n + 2)⌋) = ⌊Real.sqrt (4 * n + 1)⌋ :=
sorry

end NUMINAMATH_GPT_sqrt_floor_eq_l1066_106692


namespace NUMINAMATH_GPT_find_second_number_l1066_106662

theorem find_second_number (A B : ℝ) (h1 : A = 6400) (h2 : 0.05 * A = 0.2 * B + 190) : B = 650 :=
by
  sorry

end NUMINAMATH_GPT_find_second_number_l1066_106662


namespace NUMINAMATH_GPT_smallest_single_discount_l1066_106626

noncomputable def discount1 : ℝ := (1 - 0.20) * (1 - 0.20)
noncomputable def discount2 : ℝ := (1 - 0.10) * (1 - 0.15)
noncomputable def discount3 : ℝ := (1 - 0.08) * (1 - 0.08) * (1 - 0.08)

theorem smallest_single_discount : ∃ n : ℕ, (1 - n / 100) < discount1 ∧ (1 - n / 100) < discount2 ∧ (1 - n / 100) < discount3 ∧ n = 37 := sorry

end NUMINAMATH_GPT_smallest_single_discount_l1066_106626


namespace NUMINAMATH_GPT_boxes_per_case_l1066_106668

/-- Let's define the variables for the problem.
    We are given that Shirley sold 10 boxes of trefoils,
    and she needs to deliver 5 cases of boxes. --/
def total_boxes : ℕ := 10
def number_of_cases : ℕ := 5

/-- We need to prove that the number of boxes in each case is 2. --/
theorem boxes_per_case :
  total_boxes / number_of_cases = 2 :=
by
  -- Definition step where we specify the calculation
  unfold total_boxes number_of_cases
  -- The problem requires a division operation
  norm_num
  -- The result should be correct according to the solution steps
  done

end NUMINAMATH_GPT_boxes_per_case_l1066_106668


namespace NUMINAMATH_GPT_expression_equality_l1066_106624

-- Define the conditions
variables {a b x : ℝ}
variable (h1 : x = a / b)
variable (h2 : a ≠ 2 * b)
variable (h3 : b ≠ 0)

-- Define and state the theorem
theorem expression_equality : (2 * a + b) / (a + 2 * b) = (2 * x + 1) / (x + 2) :=
by 
  intros
  sorry

end NUMINAMATH_GPT_expression_equality_l1066_106624


namespace NUMINAMATH_GPT_expected_waiting_time_approx_l1066_106636

noncomputable def expectedWaitingTime : ℚ :=
  (10 * (1/2) + 30 * (1/3) + 50 * (1/36) + 70 * (1/12) + 90 * (1/18))

theorem expected_waiting_time_approx :
  abs (expectedWaitingTime - 27.22) < 1 :=
by
  sorry

end NUMINAMATH_GPT_expected_waiting_time_approx_l1066_106636


namespace NUMINAMATH_GPT_reduced_price_of_oil_l1066_106689

/-- 
Given:
1. The original price per kg of oil is P.
2. The reduced price per kg of oil is 0.65P.
3. Rs. 800 can buy 5 kgs more oil at the reduced price than at the original price.
4. The equation 5P - 5 * 0.65P = 800 holds true.

Prove that the reduced price per kg of oil is Rs. 297.14.
-/
theorem reduced_price_of_oil (P : ℝ) (h1 : 5 * P - 5 * 0.65 * P = 800) : 
        0.65 * P = 297.14 := 
    sorry

end NUMINAMATH_GPT_reduced_price_of_oil_l1066_106689


namespace NUMINAMATH_GPT_calc_perm_product_l1066_106606

-- Define the permutation function
def permutation (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Lean statement to prove the given problem
theorem calc_perm_product : permutation 6 2 * permutation 4 2 = 360 := 
by
  -- Test the calculations if necessary, otherwise use sorry
  sorry

end NUMINAMATH_GPT_calc_perm_product_l1066_106606


namespace NUMINAMATH_GPT_trader_sold_pens_l1066_106610

theorem trader_sold_pens (C : ℝ) (N : ℕ) (hC : C > 0) (h_gain : N * (2 / 5) = 40) : N = 100 :=
by
  sorry

end NUMINAMATH_GPT_trader_sold_pens_l1066_106610


namespace NUMINAMATH_GPT_arithmetic_seq_S10_l1066_106623

open BigOperators

variables (a : ℕ → ℚ) (d : ℚ)

-- Definitions based on the conditions
def arithmetic_seq (a : ℕ → ℚ) (d : ℚ) := ∀ n, a (n + 1) = a n + d

-- Conditions given in the problem
axiom h1 : a 5 = 1
axiom h2 : a 1 + a 7 + a 10 = a 4 + a 6

-- We aim to prove the sum of the first 10 terms
def S (n : ℕ) :=
  ∑ i in Finset.range n, a (i + 1)

theorem arithmetic_seq_S10 : arithmetic_seq a d → S a 10 = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_S10_l1066_106623


namespace NUMINAMATH_GPT_garage_sale_items_count_l1066_106648

theorem garage_sale_items_count (n_high n_low: ℕ) :
  n_high = 17 ∧ n_low = 24 → total_items = 40 :=
by
  let n_high: ℕ := 17
  let n_low: ℕ := 24
  let total_items: ℕ := (n_high - 1) + (n_low - 1) + 1
  sorry

end NUMINAMATH_GPT_garage_sale_items_count_l1066_106648


namespace NUMINAMATH_GPT_scientific_notation_of_122254_l1066_106643

theorem scientific_notation_of_122254 :
  122254 = 1.22254 * 10^5 :=
sorry

end NUMINAMATH_GPT_scientific_notation_of_122254_l1066_106643


namespace NUMINAMATH_GPT_factorization_of_polynomial_l1066_106652

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorization_of_polynomial_l1066_106652


namespace NUMINAMATH_GPT_last_digit_is_zero_last_ten_digits_are_zero_l1066_106658

-- Condition: The product includes a factor of 10
def includes_factor_of_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10

-- Conclusion: The last digit of the product must be 0
theorem last_digit_is_zero (n : ℕ) (h : includes_factor_of_10 n) : 
  n % 10 = 0 :=
sorry

-- Condition: The product includes the factors \(5^{10}\) and \(2^{10}\)
def includes_10_to_the_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10^10

-- Conclusion: The last ten digits of the product must be 0000000000
theorem last_ten_digits_are_zero (n : ℕ) (h : includes_10_to_the_10 n) : 
  n % 10^10 = 0 :=
sorry

end NUMINAMATH_GPT_last_digit_is_zero_last_ten_digits_are_zero_l1066_106658


namespace NUMINAMATH_GPT_gcd_455_299_eq_13_l1066_106637

theorem gcd_455_299_eq_13 : Nat.gcd 455 299 = 13 := by
  sorry

end NUMINAMATH_GPT_gcd_455_299_eq_13_l1066_106637


namespace NUMINAMATH_GPT_evaluate_cubic_difference_l1066_106605

theorem evaluate_cubic_difference (x y : ℚ) (h1 : x + y = 10) (h2 : 2 * x - y = 16) :
  x^3 - y^3 = 17512 / 27 :=
by sorry

end NUMINAMATH_GPT_evaluate_cubic_difference_l1066_106605


namespace NUMINAMATH_GPT_chapatis_ordered_l1066_106604

theorem chapatis_ordered (C : ℕ) 
  (chapati_cost : ℕ) (plates_rice : ℕ) (rice_cost : ℕ)
  (plates_mixed_veg : ℕ) (mixed_veg_cost : ℕ)
  (ice_cream_cups : ℕ) (ice_cream_cost : ℕ)
  (total_amount_paid : ℕ)
  (cost_eq : chapati_cost = 6)
  (plates_rice_eq : plates_rice = 5)
  (rice_cost_eq : rice_cost = 45)
  (plates_mixed_veg_eq : plates_mixed_veg = 7)
  (mixed_veg_cost_eq : mixed_veg_cost = 70)
  (ice_cream_cups_eq : ice_cream_cups = 6)
  (ice_cream_cost_eq : ice_cream_cost = 40)
  (total_paid_eq : total_amount_paid = 1051) :
  6 * C + 5 * 45 + 7 * 70 + 6 * 40 = 1051 → C = 16 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_chapatis_ordered_l1066_106604


namespace NUMINAMATH_GPT_squirrels_and_nuts_l1066_106615

theorem squirrels_and_nuts (number_of_squirrels number_of_nuts : ℕ) 
    (h1 : number_of_squirrels = 4) 
    (h2 : number_of_squirrels = number_of_nuts + 2) : 
    number_of_nuts = 2 :=
by
  sorry

end NUMINAMATH_GPT_squirrels_and_nuts_l1066_106615


namespace NUMINAMATH_GPT_jack_walked_distance_l1066_106687

def jack_walking_time: ℝ := 1.25
def jack_walking_rate: ℝ := 3.2
def jack_distance_walked: ℝ := 4

theorem jack_walked_distance:
  jack_walking_rate * jack_walking_time = jack_distance_walked :=
by
  sorry

end NUMINAMATH_GPT_jack_walked_distance_l1066_106687


namespace NUMINAMATH_GPT_bernardo_probability_is_correct_l1066_106620

noncomputable def bernardo_larger_probability : ℚ :=
  let total_bernardo_combinations := (Nat.choose 10 3 : ℚ)
  let total_silvia_combinations := (Nat.choose 8 3 : ℚ)
  let bernardo_has_10 := (Nat.choose 8 2 : ℚ) / total_bernardo_combinations
  let bernardo_not_has_10 := ((total_silvia_combinations - 1) / total_silvia_combinations) / 2
  bernardo_has_10 * 1 + (1 - bernardo_has_10) * bernardo_not_has_10

theorem bernardo_probability_is_correct :
  bernardo_larger_probability = 19 / 28 := by
  sorry

end NUMINAMATH_GPT_bernardo_probability_is_correct_l1066_106620


namespace NUMINAMATH_GPT_number_is_45_percent_of_27_l1066_106693

theorem number_is_45_percent_of_27 (x : ℝ) (h : 27 / x = 45 / 100) : x = 60 := 
by
  sorry

end NUMINAMATH_GPT_number_is_45_percent_of_27_l1066_106693


namespace NUMINAMATH_GPT_smallest_identical_digit_divisible_by_18_l1066_106631

theorem smallest_identical_digit_divisible_by_18 :
  ∃ n : Nat, (∀ d : Nat, d < n → ∃ a : Nat, (n = a * (10 ^ d - 1) / 9 + 1 ∧ (∃ k : Nat, n = 18 * k))) ∧ n = 666 :=
by
  sorry

end NUMINAMATH_GPT_smallest_identical_digit_divisible_by_18_l1066_106631


namespace NUMINAMATH_GPT_factorize_expr_l1066_106602

def my_expr (a b : ℤ) : ℤ := 4 * a^2 * b - b

theorem factorize_expr (a b : ℤ) : my_expr a b = b * (2 * a + 1) * (2 * a - 1) := by
  sorry

end NUMINAMATH_GPT_factorize_expr_l1066_106602


namespace NUMINAMATH_GPT_max_a1_l1066_106659

theorem max_a1 (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, n > 0 → a n > 0)
  (h_eq : ∀ n : ℕ, n > 0 → 2 + a n * (a (n + 1) - a (n - 1)) = 0 ∨ 2 - a n * (a (n + 1) - a (n - 1)) = 0)
  (h_a20 : a 20 = a 20) :
  ∃ max_a1 : ℝ, max_a1 = 512 := 
sorry

end NUMINAMATH_GPT_max_a1_l1066_106659


namespace NUMINAMATH_GPT_find_integer_pairs_l1066_106617

theorem find_integer_pairs (m n : ℤ) (h1 : m * n ≥ 0) (h2 : m^3 + n^3 + 99 * m * n = 33^3) :
  (m = -33 ∧ n = -33) ∨ ∃ k : ℕ, k ≤ 33 ∧ m = k ∧ n = 33 - k ∨ m = 33 - k ∧ n = k :=
by
  sorry

end NUMINAMATH_GPT_find_integer_pairs_l1066_106617


namespace NUMINAMATH_GPT_karen_bonus_problem_l1066_106699

theorem karen_bonus_problem (n already_graded last_two target : ℕ) (h_already_graded : already_graded = 8)
  (h_last_two : last_two = 290) (h_target : target = 600) (max_score : ℕ)
  (h_max_score : max_score = 150) (required_avg : ℕ) (h_required_avg : required_avg = 75) :
  ∃ A : ℕ, (A = 70) ∧ (target = 600) ∧ (last_two = 290) ∧ (already_graded = 8) ∧
  (required_avg = 75) := by
  sorry

end NUMINAMATH_GPT_karen_bonus_problem_l1066_106699


namespace NUMINAMATH_GPT_yeast_population_correct_l1066_106614

noncomputable def yeast_population_estimation 
    (count_per_small_square : ℕ)
    (dimension_large_square : ℝ)
    (dilution_factor : ℝ)
    (thickness : ℝ)
    (total_volume : ℝ) 
    : ℝ :=
    (count_per_small_square:ℝ) / ((dimension_large_square * dimension_large_square * thickness) / 400) * dilution_factor * total_volume

theorem yeast_population_correct:
    yeast_population_estimation 5 1 10 0.1 10 = 2 * 10^9 :=
by
    sorry

end NUMINAMATH_GPT_yeast_population_correct_l1066_106614


namespace NUMINAMATH_GPT_second_smallest_prime_perimeter_l1066_106650

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def scalene_triangle (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def prime_perimeter (a b c : ℕ) : Prop := 
  is_prime (a + b + c)

def different_primes (a b c : ℕ) : Prop := 
  is_prime a ∧ is_prime b ∧ is_prime c

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ), 
  scalene_triangle a b c ∧ 
  different_primes a b c ∧ 
  prime_perimeter a b c ∧ 
  a + b + c = 29 := 
sorry

end NUMINAMATH_GPT_second_smallest_prime_perimeter_l1066_106650


namespace NUMINAMATH_GPT_sam_grew_3_carrots_l1066_106628

-- Let Sandy's carrots and the total number of carrots be defined
def sandy_carrots : ℕ := 6
def total_carrots : ℕ := 9

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := total_carrots - sandy_carrots

-- The theorem to prove
theorem sam_grew_3_carrots : sam_carrots = 3 := by
  sorry

end NUMINAMATH_GPT_sam_grew_3_carrots_l1066_106628


namespace NUMINAMATH_GPT_man_l1066_106674

theorem man's_salary (S : ℝ) 
  (h_food : S * (1 / 5) > 0)
  (h_rent : S * (1 / 10) > 0)
  (h_clothes : S * (3 / 5) > 0)
  (h_left : S * (1 / 10) = 19000) : 
  S = 190000 := by
  sorry

end NUMINAMATH_GPT_man_l1066_106674


namespace NUMINAMATH_GPT_alice_favorite_number_l1066_106679

theorem alice_favorite_number :
  ∃ (n : ℕ), 50 < n ∧ n < 100 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ (n / 10 + n % 10) % 5 = 0 ∧ n = 55 :=
by
  sorry

end NUMINAMATH_GPT_alice_favorite_number_l1066_106679


namespace NUMINAMATH_GPT_initial_roses_in_vase_l1066_106616

/-- 
There were some roses in a vase. Mary cut roses from her flower garden 
and put 16 more roses in the vase. There are now 22 roses in the vase.
Prove that the initial number of roses in the vase was 6. 
-/
theorem initial_roses_in_vase (initial_roses added_roses current_roses : ℕ) 
  (h_add : added_roses = 16) 
  (h_current : current_roses = 22) 
  (h_current_eq : current_roses = initial_roses + added_roses) : 
  initial_roses = 6 := 
by
  subst h_add
  subst h_current
  linarith

end NUMINAMATH_GPT_initial_roses_in_vase_l1066_106616


namespace NUMINAMATH_GPT_max_sides_of_convex_polygon_with_arithmetic_angles_l1066_106657

theorem max_sides_of_convex_polygon_with_arithmetic_angles :
  ∀ (n : ℕ), (∃ α : ℝ, α > 0 ∧ α + (n - 1) * 1 < 180) → 
  n * (2 * α + (n - 1)) / 2 = (n - 2) * 180 → n ≤ 27 :=
by
  sorry

end NUMINAMATH_GPT_max_sides_of_convex_polygon_with_arithmetic_angles_l1066_106657


namespace NUMINAMATH_GPT_total_spent_on_toys_l1066_106625

-- Definitions for costs
def cost_car : ℝ := 14.88
def cost_skateboard : ℝ := 4.88
def cost_truck : ℝ := 5.86

-- The statement to prove
theorem total_spent_on_toys : cost_car + cost_skateboard + cost_truck = 25.62 := by
  sorry

end NUMINAMATH_GPT_total_spent_on_toys_l1066_106625


namespace NUMINAMATH_GPT_binomial_constant_term_l1066_106641

theorem binomial_constant_term : 
  (∃ c : ℕ, ∀ x : ℝ, (x + (1 / (3 * x)))^8 = c * (x ^ (4 * 2 - 8) / 3)) → 
  ∃ c : ℕ, c = 28 :=
sorry

end NUMINAMATH_GPT_binomial_constant_term_l1066_106641


namespace NUMINAMATH_GPT_circle_tangent_to_parabola_and_x_axis_eqn_l1066_106630

theorem circle_tangent_to_parabola_and_x_axis_eqn :
  (∃ (h k : ℝ), k^2 = 2 * h ∧ (x - h)^2 + (y - k)^2 = 2 * h ∧ k > 0) →
    (∀ (x y : ℝ), x^2 + y^2 - x - 2 * y + 1 / 4 = 0) := by
  sorry

end NUMINAMATH_GPT_circle_tangent_to_parabola_and_x_axis_eqn_l1066_106630


namespace NUMINAMATH_GPT_prism_volume_is_correct_l1066_106686

noncomputable def prism_volume 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : ℝ :=
  a * b * c

theorem prism_volume_is_correct 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : prism_volume a b c hab hbc hca hc_longest = 30 * Real.sqrt 10 :=
sorry

end NUMINAMATH_GPT_prism_volume_is_correct_l1066_106686


namespace NUMINAMATH_GPT_find_f_value_l1066_106678

theorem find_f_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / x^2) : 
  f (1 / 2) = 15 :=
sorry

end NUMINAMATH_GPT_find_f_value_l1066_106678


namespace NUMINAMATH_GPT_length_of_AB_l1066_106649

theorem length_of_AB 
  (AB BC CD AD : ℕ)
  (h1 : AB = 1 * BC / 2)
  (h2 : BC = 6 * CD / 5)
  (h3 : AB + BC + CD = 56)
  : AB = 12 := sorry

end NUMINAMATH_GPT_length_of_AB_l1066_106649


namespace NUMINAMATH_GPT_tom_total_payment_l1066_106680

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end NUMINAMATH_GPT_tom_total_payment_l1066_106680


namespace NUMINAMATH_GPT_factorization_correct_l1066_106640

theorem factorization_correct:
  ∃ a b : ℤ, (25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) ∧ (a + 2 * b = -24) :=
by
  sorry

end NUMINAMATH_GPT_factorization_correct_l1066_106640


namespace NUMINAMATH_GPT_wheel_horizontal_distance_l1066_106683

noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_revolution_fraction : ℝ := 3 / 4
noncomputable def wheel_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem wheel_horizontal_distance :
  wheel_circumference wheel_radius * wheel_revolution_fraction = 3 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_wheel_horizontal_distance_l1066_106683


namespace NUMINAMATH_GPT_min_value_3x_plus_4y_l1066_106685

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 28 :=
sorry

end NUMINAMATH_GPT_min_value_3x_plus_4y_l1066_106685


namespace NUMINAMATH_GPT_train_cross_signal_in_18_sec_l1066_106671

-- Definitions of the given conditions
def train_length := 300 -- meters
def platform_length := 350 -- meters
def time_cross_platform := 39 -- seconds

-- Speed of the train
def train_speed := (train_length + platform_length) / time_cross_platform -- meters/second

-- Time to cross the signal pole
def time_cross_signal_pole := train_length / train_speed -- seconds

theorem train_cross_signal_in_18_sec : time_cross_signal_pole = 18 := by sorry

end NUMINAMATH_GPT_train_cross_signal_in_18_sec_l1066_106671


namespace NUMINAMATH_GPT_find_k_l1066_106695

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : a 2 = -1)
  (h2 : 2 * a 1 + a 3 = -1)
  (h3 : arithmetic_sequence a d)
  (h4 : sum_of_sequence S a)
  (h5 : S k = -99) :
  k = 11 := 
by
  sorry

end NUMINAMATH_GPT_find_k_l1066_106695


namespace NUMINAMATH_GPT_percentage_of_150_l1066_106618

theorem percentage_of_150 : (1 / 5 * (1 / 100) * 150 : ℝ) = 0.3 := by
  sorry

end NUMINAMATH_GPT_percentage_of_150_l1066_106618


namespace NUMINAMATH_GPT_journey_duration_is_9_hours_l1066_106608

noncomputable def journey_time : ℝ :=
  let d1 := 90 -- Distance traveled by Tom and Dick by car before Tom got off
  let d2 := 60 -- Distance Dick backtracked to pick up Harry
  let T := (d1 / 30) + ((120 - d1) / 5) -- Time taken for Tom's journey
  T

theorem journey_duration_is_9_hours : journey_time = 9 := 
by 
  sorry

end NUMINAMATH_GPT_journey_duration_is_9_hours_l1066_106608


namespace NUMINAMATH_GPT_marbles_problem_l1066_106611

theorem marbles_problem (a : ℚ) (h1: 34 * a = 156) : a = 78 / 17 := 
by
  sorry

end NUMINAMATH_GPT_marbles_problem_l1066_106611


namespace NUMINAMATH_GPT_small_pos_int_n_l1066_106633

theorem small_pos_int_n (a : ℕ → ℕ) (n : ℕ) (a1_val : a 1 = 7)
  (recurrence: ∀ n, a (n + 1) = a n * (a n + 2)) :
  ∃ n : ℕ, a n > 2 ^ 4036 ∧ ∀ m : ℕ, (m < n) → a m ≤ 2 ^ 4036 :=
by
  sorry

end NUMINAMATH_GPT_small_pos_int_n_l1066_106633


namespace NUMINAMATH_GPT_min_value_of_x_l1066_106690

-- Define the conditions and state the problem
theorem min_value_of_x (x : ℝ) : (∀ a : ℝ, a > 0 → x^2 < 1 + a) → x ≥ -1 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_x_l1066_106690


namespace NUMINAMATH_GPT_proportion_equation_correct_l1066_106635

theorem proportion_equation_correct (x y : ℝ) (h1 : 2 * x = 3 * y) (h2 : x ≠ 0) (h3 : y ≠ 0) : 
  x / 3 = y / 2 := 
  sorry

end NUMINAMATH_GPT_proportion_equation_correct_l1066_106635


namespace NUMINAMATH_GPT_cupcakes_left_correct_l1066_106646

-- Definitions based on conditions
def total_cupcakes : ℕ := 10 * 12 + 1 * 12 / 2
def total_students : ℕ := 48
def absent_students : ℕ := 6 
def field_trip_students : ℕ := 8
def teachers : ℕ := 2
def teachers_aids : ℕ := 2

-- Function to calculate the number of present people
def total_present_people : ℕ :=
  total_students - absent_students - field_trip_students + teachers + teachers_aids

-- Function to calculate the cupcakes left
def cupcakes_left : ℕ := total_cupcakes - total_present_people

-- The theorem to prove
theorem cupcakes_left_correct : cupcakes_left = 85 := 
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_cupcakes_left_correct_l1066_106646


namespace NUMINAMATH_GPT_compare_f_g_l1066_106661

def R (m n : ℕ) : ℕ := sorry
def L (m n : ℕ) : ℕ := sorry

def f (m n : ℕ) : ℕ := R m n + L m n - sorry
def g (m n : ℕ) : ℕ := R m n + L m n - sorry

theorem compare_f_g (m n : ℕ) : f m n ≤ g m n := sorry

end NUMINAMATH_GPT_compare_f_g_l1066_106661


namespace NUMINAMATH_GPT_pictures_left_l1066_106619

def zoo_pics : ℕ := 802
def museum_pics : ℕ := 526
def beach_pics : ℕ := 391
def amusement_park_pics : ℕ := 868
def duplicates_deleted : ℕ := 1395

theorem pictures_left : 
  (zoo_pics + museum_pics + beach_pics + amusement_park_pics - duplicates_deleted) = 1192 := 
by
  sorry

end NUMINAMATH_GPT_pictures_left_l1066_106619


namespace NUMINAMATH_GPT_find_x_for_which_f_f_x_eq_f_x_l1066_106694

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem find_x_for_which_f_f_x_eq_f_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_which_f_f_x_eq_f_x_l1066_106694


namespace NUMINAMATH_GPT_perfect_square_octal_last_digit_l1066_106651

theorem perfect_square_octal_last_digit (a b c : ℕ) (n : ℕ) (h1 : a ≠ 0) (h2 : (abc:ℕ) = n^2) :
  c = 1 :=
sorry

end NUMINAMATH_GPT_perfect_square_octal_last_digit_l1066_106651


namespace NUMINAMATH_GPT_sum_of_integers_satisfying_l1066_106663

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_satisfying_l1066_106663


namespace NUMINAMATH_GPT_probability_blue_or_purple_is_correct_l1066_106629

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10 + 4

def blue_jelly_beans : ℕ := 10

def purple_jelly_beans : ℕ := 4

def blue_or_purple_jelly_beans : ℕ := blue_jelly_beans + purple_jelly_beans

def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_correct :
  probability_blue_or_purple = 7 / 19 :=
by
  sorry

end NUMINAMATH_GPT_probability_blue_or_purple_is_correct_l1066_106629


namespace NUMINAMATH_GPT_range_of_a_l1066_106601

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + 2 * a * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 1) :=
sorry

end NUMINAMATH_GPT_range_of_a_l1066_106601


namespace NUMINAMATH_GPT_intersection_M_N_l1066_106600

open Set

def M : Set ℝ := {-2, -1, 0, 1, 2}
def N : Set ℝ := {x | (x + 1) * (x - 2) ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1066_106600


namespace NUMINAMATH_GPT_omar_total_time_l1066_106691

-- Conditions
def lap_distance : ℝ := 400
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 200
def speed_first_segment : ℝ := 6
def speed_second_segment : ℝ := 4
def number_of_laps : ℝ := 7

-- Correct answer we want to prove
def total_time_proven : ℝ := 9 * 60 + 23 -- in seconds

-- Theorem statement claiming total time is 9 minutes and 23 seconds
theorem omar_total_time :
  let time_first_segment := first_segment_distance / speed_first_segment
  let time_second_segment := second_segment_distance / speed_second_segment
  let single_lap_time := time_first_segment + time_second_segment
  let total_time := number_of_laps * single_lap_time
  total_time = total_time_proven := sorry

end NUMINAMATH_GPT_omar_total_time_l1066_106691


namespace NUMINAMATH_GPT_maria_paper_count_l1066_106655

-- Defining the initial number of sheets and the actions taken
variables (x y : ℕ)
def initial_sheets := 50 + 41
def remaining_sheets_after_giving_away := initial_sheets - x
def whole_sheets := remaining_sheets_after_giving_away - y
def half_sheets := y

-- The theorem we want to prove
theorem maria_paper_count (x y : ℕ) :
  whole_sheets x y = initial_sheets - x - y ∧ 
  half_sheets y = y :=
by sorry

end NUMINAMATH_GPT_maria_paper_count_l1066_106655


namespace NUMINAMATH_GPT_year_weeks_span_l1066_106673

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end NUMINAMATH_GPT_year_weeks_span_l1066_106673


namespace NUMINAMATH_GPT_bruce_total_payment_l1066_106656

-- Define the conditions
def quantity_grapes : Nat := 7
def rate_grapes : Nat := 70
def quantity_mangoes : Nat := 9
def rate_mangoes : Nat := 55

-- Define the calculation for total amount paid
def total_amount_paid : Nat :=
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes)

-- Proof statement
theorem bruce_total_payment : total_amount_paid = 985 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_bruce_total_payment_l1066_106656


namespace NUMINAMATH_GPT_sin_double_angle_l1066_106609

theorem sin_double_angle (α : ℝ) (h : Real.cos (Real.pi / 4 - α) = Real.sqrt 2 / 4) :
  Real.sin (2 * α) = -3 / 4 :=
sorry

end NUMINAMATH_GPT_sin_double_angle_l1066_106609


namespace NUMINAMATH_GPT_smallest_N_conditions_l1066_106634

theorem smallest_N_conditions:
  ∃N : ℕ, N % 9 = 8 ∧
           N % 8 = 7 ∧
           N % 7 = 6 ∧
           N % 6 = 5 ∧
           N % 5 = 4 ∧
           N % 4 = 3 ∧
           N % 3 = 2 ∧
           N % 2 = 1 ∧
           N = 2519 :=
sorry

end NUMINAMATH_GPT_smallest_N_conditions_l1066_106634


namespace NUMINAMATH_GPT_unique_symmetric_matrix_pair_l1066_106666

theorem unique_symmetric_matrix_pair (a b : ℝ) :
  (∃! M : Matrix (Fin 2) (Fin 2) ℝ, M = M.transpose ∧ Matrix.trace M = a ∧ Matrix.det M = b)
  ↔ (∃ t : ℝ, a = 2 * t ∧ b = t^2) :=
by
  sorry

end NUMINAMATH_GPT_unique_symmetric_matrix_pair_l1066_106666


namespace NUMINAMATH_GPT_sara_height_l1066_106639

def Julie := 33
def Mark := Julie + 1
def Roy := Mark + 2
def Joe := Roy + 3
def Sara := Joe + 6

theorem sara_height : Sara = 45 := by
  sorry

end NUMINAMATH_GPT_sara_height_l1066_106639


namespace NUMINAMATH_GPT_total_apples_picked_l1066_106676

theorem total_apples_picked (Mike_apples Nancy_apples Keith_apples : ℕ)
  (hMike : Mike_apples = 7)
  (hNancy : Nancy_apples = 3)
  (hKeith : Keith_apples = 6) :
  Mike_apples + Nancy_apples + Keith_apples = 16 :=
by
  sorry

end NUMINAMATH_GPT_total_apples_picked_l1066_106676


namespace NUMINAMATH_GPT_fewer_bands_l1066_106653

theorem fewer_bands (J B Y : ℕ) (h1 : J = B + 10) (h2 : B - 4 = 8) (h3 : Y = 24) :
  Y - J = 2 :=
sorry

end NUMINAMATH_GPT_fewer_bands_l1066_106653


namespace NUMINAMATH_GPT_Ashutosh_completion_time_l1066_106672

def Suresh_work_rate := 1 / 15
def Ashutosh_work_rate := 1 / 25
def Suresh_work_time := 9

def job_completed_by_Suresh_in_9_hours := Suresh_work_rate * Suresh_work_time
def remaining_job := 1 - job_completed_by_Suresh_in_9_hours

theorem Ashutosh_completion_time : 
  Ashutosh_work_rate * t = remaining_job -> t = 10 :=
by
  sorry

end NUMINAMATH_GPT_Ashutosh_completion_time_l1066_106672


namespace NUMINAMATH_GPT_solve_for_ratio_l1066_106644

noncomputable def slope_tangent_y_equals_x_squared (x1 : ℝ) : ℝ :=
  2 * x1

noncomputable def slope_tangent_y_equals_x_cubed (x2 : ℝ) : ℝ :=
  3 * x2 * x2

noncomputable def y1_compute (x1 : ℝ) : ℝ :=
  x1 * x1

noncomputable def y2_compute (x2 : ℝ) : ℝ :=
  x2 * x2 * x2

theorem solve_for_ratio (x1 x2 : ℝ)
    (tangent_l_same : slope_tangent_y_equals_x_squared x1 = slope_tangent_y_equals_x_cubed x2)
    (y_tangent_l_same : y1_compute x1 = y2_compute x2) :
  x1 / x2 = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_ratio_l1066_106644


namespace NUMINAMATH_GPT_mary_thought_animals_l1066_106675

-- Definitions based on conditions
def double_counted_sheep : ℕ := 7
def forgotten_pigs : ℕ := 3
def actual_animals : ℕ := 56

-- Statement to be proven
theorem mary_thought_animals (double_counted_sheep forgotten_pigs actual_animals : ℕ) :
  (actual_animals + double_counted_sheep - forgotten_pigs) = 60 := 
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_mary_thought_animals_l1066_106675


namespace NUMINAMATH_GPT_exists_triangle_with_sin_angles_l1066_106660

theorem exists_triangle_with_sin_angles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2 * (a^2*b^2 + a^2*c^2 + b^2*c^2)) : 
    ∃ (α β γ : ℝ), α + β + γ = Real.pi ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c :=
by
  sorry

end NUMINAMATH_GPT_exists_triangle_with_sin_angles_l1066_106660
