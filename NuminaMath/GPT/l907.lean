import Mathlib

namespace NUMINAMATH_GPT_proof_P_and_Q_l907_90713

/-!
Proposition P: The line y=2x is perpendicular to the line x+2y=0.
Proposition Q: The projections of skew lines in the same plane could be parallel lines.
Prove: P ∧ Q is true.
-/

def proposition_P : Prop := 
  let slope1 := 2
  let slope2 := -1 / 2
  slope1 * slope2 = -1

def proposition_Q : Prop :=
  ∃ (a b : ℝ), (∃ (p q r s : ℝ),
    (a * r + b * p = 0) ∧ (a * s + b * q = 0)) ∧
    (a ≠ 0 ∨ b ≠ 0)

theorem proof_P_and_Q : proposition_P ∧ proposition_Q :=
  by
  -- We need to prove the conjunction of both propositions is true.
  sorry

end NUMINAMATH_GPT_proof_P_and_Q_l907_90713


namespace NUMINAMATH_GPT_remainder_when_150_divided_by_k_is_2_l907_90745

theorem remainder_when_150_divided_by_k_is_2
  (k : ℕ) (q : ℤ)
  (hk_pos : k > 0)
  (hk_condition : 120 = q * k^2 + 8) :
  150 % k = 2 :=
sorry

end NUMINAMATH_GPT_remainder_when_150_divided_by_k_is_2_l907_90745


namespace NUMINAMATH_GPT_find_m_l907_90760

theorem find_m (m : ℕ) (h : 8 ^ 36 * 6 ^ 21 = 3 * 24 ^ m) : m = 43 :=
sorry

end NUMINAMATH_GPT_find_m_l907_90760


namespace NUMINAMATH_GPT_area_of_tangency_triangle_l907_90724

theorem area_of_tangency_triangle 
  (r1 r2 r3 : ℝ) 
  (h1 : r1 = 2)
  (h2 : r2 = 3)
  (h3 : r3 = 4) 
  (mutually_tangent : ∀ {c1 c2 c3 : ℝ}, c1 + c2 = r1 + r2 ∧ c2 + c3 = r2 + r3 ∧ c1 + c3 = r1 + r3 ) :
  ∃ area : ℝ, area = 3 * (Real.sqrt 6) / 2 :=
by
  sorry

end NUMINAMATH_GPT_area_of_tangency_triangle_l907_90724


namespace NUMINAMATH_GPT_tan_product_identity_l907_90742

theorem tan_product_identity : (1 + Real.tan (Real.pi / 180 * 17)) * (1 + Real.tan (Real.pi / 180 * 28)) = 2 := by
  sorry

end NUMINAMATH_GPT_tan_product_identity_l907_90742


namespace NUMINAMATH_GPT_solve_for_x_l907_90762

theorem solve_for_x (A B C D: Type) 
(y z w x : ℝ) 
(h_triangle : ∃ a b c : Type, True) 
(h_D_on_extension : ∃ D_on_extension : Type, True)
(h_AD_GT_BD : ∃ s : Type, True) 
(h_x_at_D : ∃ t : Type, True) 
(h_y_at_A : ∃ u : Type, True) 
(h_z_at_B : ∃ v : Type, True) 
(h_w_at_C : ∃ w : Type, True)
(h_triangle_angle_sum : y + z + w = 180):
x = 180 - z - w := by
  sorry

end NUMINAMATH_GPT_solve_for_x_l907_90762


namespace NUMINAMATH_GPT_complement_B_range_a_l907_90790

open Set

variable (A B : Set ℝ) (a : ℝ)

def mySetA : Set ℝ := {x | 2 * a - 2 < x ∧ x < a}
def mySetB : Set ℝ := {x | 3 / (x - 1) ≥ 1}

theorem complement_B_range_a (h : mySetA a ⊆ compl mySetB) : 
  compl mySetB = {x | x ≤ 1} ∪ {x | x > 4} ∧ (a ≤ 1 ∨ a ≥ 2) :=
by
  sorry

end NUMINAMATH_GPT_complement_B_range_a_l907_90790


namespace NUMINAMATH_GPT_max_problems_to_miss_to_pass_l907_90700

theorem max_problems_to_miss_to_pass (total_problems : ℕ) (pass_percentage : ℝ) :
  total_problems = 50 → pass_percentage = 0.85 → 7 = ↑total_problems * (1 - pass_percentage) :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_max_problems_to_miss_to_pass_l907_90700


namespace NUMINAMATH_GPT_radius_of_triangle_DEF_l907_90708

noncomputable def radius_of_inscribed_circle (DE DF EF : ℝ) : ℝ :=
  let s := (DE + DF + EF) / 2
  let K := Real.sqrt (s * (s - DE) * (s - DF) * (s - EF))
  K / s

theorem radius_of_triangle_DEF :
  radius_of_inscribed_circle 26 15 17 = 121 / 29 := by
sorry

end NUMINAMATH_GPT_radius_of_triangle_DEF_l907_90708


namespace NUMINAMATH_GPT_shells_collected_by_savannah_l907_90743

def num_shells_jillian : ℕ := 29
def num_shells_clayton : ℕ := 8
def total_shells_distributed : ℕ := 54

theorem shells_collected_by_savannah (S : ℕ) :
  num_shells_jillian + S + num_shells_clayton = total_shells_distributed → S = 17 :=
by
  sorry

end NUMINAMATH_GPT_shells_collected_by_savannah_l907_90743


namespace NUMINAMATH_GPT_john_profit_l907_90773

-- Definitions based on given conditions
def total_newspapers := 500
def selling_price_per_newspaper : ℝ := 2
def discount_percentage : ℝ := 0.75
def percentage_sold : ℝ := 0.80

-- Derived basic definitions
def cost_price_per_newspaper := selling_price_per_newspaper * (1 - discount_percentage)
def total_cost_price := cost_price_per_newspaper * total_newspapers
def newspapers_sold := total_newspapers * percentage_sold
def revenue := selling_price_per_newspaper * newspapers_sold
def profit := revenue - total_cost_price

-- Theorem stating the profit
theorem john_profit : profit = 550 := by
  sorry

#check john_profit

end NUMINAMATH_GPT_john_profit_l907_90773


namespace NUMINAMATH_GPT_log_negative_l907_90749

open Real

theorem log_negative (a : ℝ) (h : a > 0) : log (-a) = log a := sorry

end NUMINAMATH_GPT_log_negative_l907_90749


namespace NUMINAMATH_GPT_sticks_per_stool_is_two_l907_90772

-- Conditions
def sticks_from_chair := 6
def sticks_from_table := 9
def sticks_needed_per_hour := 5
def num_chairs := 18
def num_tables := 6
def num_stools := 4
def hours_to_keep_warm := 34

-- Question and Answer in Lean 4 statement
theorem sticks_per_stool_is_two : 
  (hours_to_keep_warm * sticks_needed_per_hour) - (num_chairs * sticks_from_chair + num_tables * sticks_from_table) = 2 * num_stools := 
  by
    sorry

end NUMINAMATH_GPT_sticks_per_stool_is_two_l907_90772


namespace NUMINAMATH_GPT_find_floor_abs_S_l907_90780

-- Conditions
-- For integers from 1 to 1500, x_1 + 2 = x_2 + 4 = x_3 + 6 = ... = x_1500 + 3000 = ∑(n=1 to 1500) x_n + 3001
def condition (x : ℕ → ℤ) (S : ℤ) : Prop :=
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 1500 →
    x a + 2 * a = S + 3001

-- Problem statement
theorem find_floor_abs_S (x : ℕ → ℤ) (S : ℤ)
  (h : condition x S) :
  (⌊|S|⌋ : ℤ) = 1500 :=
sorry

end NUMINAMATH_GPT_find_floor_abs_S_l907_90780


namespace NUMINAMATH_GPT_find_value_of_a5_l907_90740

variable (a : ℕ → ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a_1 d : ℝ), ∀ n, a n = a_1 + (n - 1) * d

variable (h_arith : is_arithmetic_sequence a)
variable (h : a 2 + a 8 = 12)

theorem find_value_of_a5 : a 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_value_of_a5_l907_90740


namespace NUMINAMATH_GPT_min_tiles_for_square_l907_90758

theorem min_tiles_for_square (a b : ℕ) (ha : a = 6) (hb : b = 4) (harea_tile : a * b = 24)
  (h_lcm : Nat.lcm a b = 12) : 
  let area_square := (Nat.lcm a b) * (Nat.lcm a b) 
  let num_tiles_required := area_square / (a * b)
  num_tiles_required = 6 :=
by
  sorry

end NUMINAMATH_GPT_min_tiles_for_square_l907_90758


namespace NUMINAMATH_GPT_ms_walker_drives_24_miles_each_way_l907_90761

theorem ms_walker_drives_24_miles_each_way
  (D : ℝ)
  (H1 : 1 / 60 * D + 1 / 40 * D = 1) :
  D = 24 := 
sorry

end NUMINAMATH_GPT_ms_walker_drives_24_miles_each_way_l907_90761


namespace NUMINAMATH_GPT_bakery_combinations_l907_90756

theorem bakery_combinations (h : ∀ (a b c : ℕ), a + b + c = 8 ∧ a > 0 ∧ b > 0 ∧ c > 0) : 
  ∃ count : ℕ, count = 25 := 
sorry

end NUMINAMATH_GPT_bakery_combinations_l907_90756


namespace NUMINAMATH_GPT_inequality_system_solution_l907_90729

theorem inequality_system_solution (x : ℝ) :
  (3 * x - 1 > 2 * (x + 1) ∧ (x + 2) / 3 > x - 2) ↔ (3 < x ∧ x < 4) := 
by
  sorry

end NUMINAMATH_GPT_inequality_system_solution_l907_90729


namespace NUMINAMATH_GPT_compute_expression_l907_90734

theorem compute_expression : -8 * 4 - (-6 * -3) + (-10 * -5) = 0 := by sorry

end NUMINAMATH_GPT_compute_expression_l907_90734


namespace NUMINAMATH_GPT_triangle_area_l907_90755

theorem triangle_area
  (a b : ℝ)
  (C : ℝ)
  (h₁ : a = 2)
  (h₂ : b = 3)
  (h₃ : C = π / 3) :
  (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_GPT_triangle_area_l907_90755


namespace NUMINAMATH_GPT_circle_center_and_radius_l907_90792

noncomputable def circle_eq : Prop :=
  ∀ x y : ℝ, (x^2 + y^2 + 2 * x - 2 * y - 2 = 0) ↔ (x + 1)^2 + (y - 1)^2 = 4

theorem circle_center_and_radius :
  ∃ center : ℝ × ℝ, ∃ r : ℝ, 
  center = (-1, 1) ∧ r = 2 ∧ circle_eq :=
by
  sorry

end NUMINAMATH_GPT_circle_center_and_radius_l907_90792


namespace NUMINAMATH_GPT_distinct_valid_sets_count_l907_90710

-- Define non-negative powers of 2 and 3
def is_non_neg_power (n : ℕ) : Prop :=
  ∃ (a b : ℕ), n = 2^a ∨ n = 3^b

-- Define the condition for sum of elements in set S to be 2014
def valid_sets (S : Finset ℕ) : Prop :=
  (∀ x ∈ S, is_non_neg_power x) ∧ (S.sum id = 2014)

theorem distinct_valid_sets_count : ∃ (number_of_distinct_sets : ℕ), number_of_distinct_sets = 64 :=
  sorry

end NUMINAMATH_GPT_distinct_valid_sets_count_l907_90710


namespace NUMINAMATH_GPT_find_triples_l907_90766

-- Definitions of the problem conditions
def is_prime (p : ℕ) : Prop := Nat.Prime p

def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

def satisfies_equation (a b p : ℕ) : Prop := a^p = factorial b + p

-- The main theorem statement based on the problem conditions
theorem find_triples :
  (satisfies_equation 2 2 2 ∧ is_prime 2) ∧
  (satisfies_equation 3 4 3 ∧ is_prime 3) ∧
  (∀ (a b p : ℕ), (satisfies_equation a b p ∧ is_prime p) → (a, b, p) = (2, 2, 2) ∨ (a, b, p) = (3, 4, 3)) :=
by
  -- Proof to be filled
  sorry

end NUMINAMATH_GPT_find_triples_l907_90766


namespace NUMINAMATH_GPT_cos_alpha_values_l907_90704

theorem cos_alpha_values (α : ℝ) (h : Real.sin (π + α) = -3 / 5) :
  Real.cos α = 4 / 5 ∨ Real.cos α = -4 / 5 := 
sorry

end NUMINAMATH_GPT_cos_alpha_values_l907_90704


namespace NUMINAMATH_GPT_sum_of_extreme_numbers_is_846_l907_90769

theorem sum_of_extreme_numbers_is_846 :
  let digits := [0, 2, 4, 6]
  let is_valid_hundreds_digit (d : Nat) := d ≠ 0
  let create_three_digit_number (h t u : Nat) := h * 100 + t * 10 + u
  let max_num := create_three_digit_number 6 4 2
  let min_num := create_three_digit_number 2 0 4
  max_num + min_num = 846 := by
  sorry

end NUMINAMATH_GPT_sum_of_extreme_numbers_is_846_l907_90769


namespace NUMINAMATH_GPT_area_of_right_triangle_l907_90714

theorem area_of_right_triangle (A B C : ℝ) (hA : A = 64) (hB : B = 36) (hC : C = 100) : 
  (1 / 2) * (Real.sqrt A) * (Real.sqrt B) = 24 :=
by
  sorry

end NUMINAMATH_GPT_area_of_right_triangle_l907_90714


namespace NUMINAMATH_GPT_inequality_proof_l907_90754

theorem inequality_proof
  (x y z : ℝ)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0)
  (hxyz : x * y * z ≥ 1) :
  (x^3 - x^2) / (x^5 + y^2 + z^2) +
  (y^5 - y^2) / (y^5 + z^2 + x^2) +
  (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l907_90754


namespace NUMINAMATH_GPT_arithmetic_sequence_first_term_l907_90737

theorem arithmetic_sequence_first_term (a : ℕ → ℤ) (d : ℤ) 
  (h1 : a 5 = 9) (h2 : 2 * a 3 = a 2 + 6) : a 1 = -3 :=
by
  -- a_5 = a_1 + 4d
  have h3 : a 5 = a 1 + 4 * d := sorry
  
  -- 2a_3 = a_2 + 6, which means 2 * (a_1 + 2d) = (a_1 + d) + 6
  have h4 : 2 * (a 1 + 2 * d) = (a 1 + d) + 6 := sorry
  
  -- solve the system of linear equations to find a_1 = -3
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_first_term_l907_90737


namespace NUMINAMATH_GPT_area_per_car_l907_90770

/-- Given the length and width of the parking lot, 
and the percentage of usable area, 
and the number of cars that can be parked,
prove that the area per car is as expected. -/
theorem area_per_car 
  (length width : ℝ) 
  (usable_percentage : ℝ) 
  (number_of_cars : ℕ) 
  (h_length : length = 400) 
  (h_width : width = 500) 
  (h_usable_percentage : usable_percentage = 0.80) 
  (h_number_of_cars : number_of_cars = 16000) :
  (length * width * usable_percentage) / number_of_cars = 10 :=
by
  sorry

end NUMINAMATH_GPT_area_per_car_l907_90770


namespace NUMINAMATH_GPT_find_m_n_l907_90798

theorem find_m_n (m n : ℤ) :
  (∀ x : ℤ, (x + 4) * (x - 2) = x^2 + m * x + n) → (m = 2 ∧ n = -8) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_m_n_l907_90798


namespace NUMINAMATH_GPT_eq_m_neg_one_l907_90775

theorem eq_m_neg_one (m : ℝ) (x : ℝ) (h1 : (m-1) * x^(m^2 + 1) + 2*x - 3 = 0) (h2 : m - 1 ≠ 0) (h3 : m^2 + 1 = 2) : 
  m = -1 :=
sorry

end NUMINAMATH_GPT_eq_m_neg_one_l907_90775


namespace NUMINAMATH_GPT_part_I_part_II_l907_90732

noncomputable def f (a x : ℝ) : ℝ := |x - 1| + a * |x - 2|

theorem part_I (a : ℝ) (h_min : ∃ m, ∀ x, f a x ≥ m) : -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem part_II (a : ℝ) (h_bound : ∀ x, f a x ≥ 1/2) : a = 1/3 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l907_90732


namespace NUMINAMATH_GPT_combined_salaries_correct_l907_90721

noncomputable def combined_salaries_BCDE (A B C D E : ℕ) : Prop :=
  (A = 8000) →
  ((A + B + C + D + E) / 5 = 8600) →
  (B + C + D + E = 35000)

theorem combined_salaries_correct 
  (A B C D E : ℕ) 
  (hA : A = 8000) 
  (havg : (A + B + C + D + E) / 5 = 8600) : 
  B + C + D + E = 35000 :=
sorry

end NUMINAMATH_GPT_combined_salaries_correct_l907_90721


namespace NUMINAMATH_GPT_white_area_correct_l907_90797

/-- The dimensions of the sign and the letter components -/
def sign_width : ℕ := 18
def sign_height : ℕ := 6
def vertical_bar_height : ℕ := 6
def vertical_bar_width : ℕ := 1
def horizontal_bar_length : ℕ := 4
def horizontal_bar_width : ℕ := 1

/-- The areas of the components of each letter -/
def area_C : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)
def area_O : ℕ := 2 * (vertical_bar_height * vertical_bar_width) + 2 * (horizontal_bar_length * horizontal_bar_width)
def area_L : ℕ := (vertical_bar_height * vertical_bar_width) + (horizontal_bar_length * horizontal_bar_width)

/-- The total area of the sign -/
def total_sign_area : ℕ := sign_height * sign_width

/-- The total black area covered by the letters "COOL" -/
def total_black_area : ℕ := area_C + 2 * area_O + area_L

/-- The area of the white portion of the sign -/
def white_area : ℕ := total_sign_area - total_black_area

/-- Proof that the area of the white portion of the sign is 42 square units -/
theorem white_area_correct : white_area = 42 := by
  -- Calculation steps (skipped, though the result is expected to be 42)
  sorry

end NUMINAMATH_GPT_white_area_correct_l907_90797


namespace NUMINAMATH_GPT_Elberta_has_23_dollars_l907_90763

theorem Elberta_has_23_dollars :
  let granny_smith_amount := 63
  let anjou_amount := 1 / 3 * granny_smith_amount
  let elberta_amount := anjou_amount + 2
  elberta_amount = 23 := by
  sorry

end NUMINAMATH_GPT_Elberta_has_23_dollars_l907_90763


namespace NUMINAMATH_GPT_exists_three_distinct_div_l907_90785

theorem exists_three_distinct_div (a b c : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a < b) (h5 : b < c) :
  ∀ m : ℕ, ∃ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ abc ∣ (x * y * z) ∧ m ≤ x ∧ x < m + 2*c ∧ m ≤ y ∧ y < m + 2*c ∧ m ≤ z ∧ z < m + 2*c :=
by
  sorry

end NUMINAMATH_GPT_exists_three_distinct_div_l907_90785


namespace NUMINAMATH_GPT_permutation_6_2_eq_30_l907_90783

theorem permutation_6_2_eq_30 :
  (Nat.factorial 6) / (Nat.factorial (6 - 2)) = 30 :=
by
  sorry

end NUMINAMATH_GPT_permutation_6_2_eq_30_l907_90783


namespace NUMINAMATH_GPT_cost_plane_l907_90709

def cost_boat : ℝ := 254.00
def savings_boat : ℝ := 346.00

theorem cost_plane : cost_boat + savings_boat = 600 := 
by 
  sorry

end NUMINAMATH_GPT_cost_plane_l907_90709


namespace NUMINAMATH_GPT_bill_age_l907_90786

theorem bill_age (C : ℕ) (h1 : ∀ B : ℕ, B = 2 * C - 1) (h2 : C + (2 * C - 1) = 26) : 
  ∃ B : ℕ, B = 17 := 
by
  sorry

end NUMINAMATH_GPT_bill_age_l907_90786


namespace NUMINAMATH_GPT_min_value_proof_l907_90752

noncomputable def min_value (α γ : ℝ) : ℝ :=
  (3 * Real.cos α + 4 * Real.sin γ - 7)^2 + (3 * Real.sin α + 4 * Real.cos γ - 12)^2

theorem min_value_proof (α γ : ℝ) : ∃ α γ : ℝ, min_value α γ = 36 :=
by
  use (Real.arcsin 12/13), (Real.pi/2 - Real.arcsin 12/13)
  sorry

end NUMINAMATH_GPT_min_value_proof_l907_90752


namespace NUMINAMATH_GPT_happy_valley_zoo_animal_arrangement_l907_90701

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

end NUMINAMATH_GPT_happy_valley_zoo_animal_arrangement_l907_90701


namespace NUMINAMATH_GPT_least_k_for_168_l907_90718

theorem least_k_for_168 (k : ℕ) :
  (k^3 % 168 = 0) ↔ k ≥ 42 :=
sorry

end NUMINAMATH_GPT_least_k_for_168_l907_90718


namespace NUMINAMATH_GPT_smallest_possible_input_l907_90779

def F (n : ℕ) := 9 * n + 120

theorem smallest_possible_input : ∃ n : ℕ, n > 0 ∧ F n = 129 :=
by {
  -- Here we would provide the proof steps, but we use sorry for now.
  sorry
}

end NUMINAMATH_GPT_smallest_possible_input_l907_90779


namespace NUMINAMATH_GPT_percentage_parents_agree_l907_90725

def total_parents : ℕ := 800
def disagree_parents : ℕ := 640

theorem percentage_parents_agree : 
  ((total_parents - disagree_parents) / total_parents : ℚ) * 100 = 20 := 
by 
  sorry

end NUMINAMATH_GPT_percentage_parents_agree_l907_90725


namespace NUMINAMATH_GPT_gerald_paid_l907_90747

theorem gerald_paid (G : ℝ) (h : 0.8 * G = 200) : G = 250 := by
  sorry

end NUMINAMATH_GPT_gerald_paid_l907_90747


namespace NUMINAMATH_GPT_min_value_of_a_for_inverse_l907_90746

theorem min_value_of_a_for_inverse (a : ℝ) : 
  (∀ x y : ℝ, x ≥ a → y ≥ a → (x^2 + 4*x ≤ y^2 + 4*y ↔ x ≤ y)) → a = -2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_a_for_inverse_l907_90746


namespace NUMINAMATH_GPT_and_15_and_l907_90730

def x_and (x : ℝ) : ℝ := 8 - x
def and_x (x : ℝ) : ℝ := x - 8

theorem and_15_and : and_x (x_and 15) = -15 :=
by
  sorry

end NUMINAMATH_GPT_and_15_and_l907_90730


namespace NUMINAMATH_GPT_power_sum_l907_90791

theorem power_sum (a b c : ℝ) (h1 : a + b + c = 1)
                  (h2 : a^2 + b^2 + c^2 = 3)
                  (h3 : a^3 + b^3 + c^3 = 4)
                  (h4 : a^4 + b^4 + c^4 = 5) :
  a^5 + b^5 + c^5 = 6 :=
  sorry

end NUMINAMATH_GPT_power_sum_l907_90791


namespace NUMINAMATH_GPT_fermats_little_theorem_l907_90716

theorem fermats_little_theorem 
  (a n : ℕ) 
  (h₁ : 0 < a) 
  (h₂ : 0 < n) 
  (h₃ : Nat.gcd a n = 1) 
  (phi : ℕ := (Nat.totient n)) 
  : n ∣ (a ^ phi - 1) := sorry

end NUMINAMATH_GPT_fermats_little_theorem_l907_90716


namespace NUMINAMATH_GPT_cubic_yard_to_cubic_feet_l907_90703

theorem cubic_yard_to_cubic_feet (h : 1 = 3) : 1 = 27 := 
by
  sorry

end NUMINAMATH_GPT_cubic_yard_to_cubic_feet_l907_90703


namespace NUMINAMATH_GPT_triangle_expression_simplification_l907_90764

variable (a b c : ℝ)

theorem triangle_expression_simplification (h1 : a + b > c) 
                                           (h2 : a + c > b) 
                                           (h3 : b + c > a) :
  |a - b - c| + |b - a - c| - |c - a + b| = a - b + c :=
sorry

end NUMINAMATH_GPT_triangle_expression_simplification_l907_90764


namespace NUMINAMATH_GPT_sum_infinite_series_l907_90771

theorem sum_infinite_series :
  (∑' n : ℕ, 1 / (n + 1) / (n + 4)) = 1 / 3 :=
sorry

end NUMINAMATH_GPT_sum_infinite_series_l907_90771


namespace NUMINAMATH_GPT_exists_real_root_iff_l907_90707

theorem exists_real_root_iff {m : ℝ} :
  (∃x : ℝ, 25 - abs (x + 1) - 4 * 5 - abs (x + 1) - m = 0) ↔ (-3 < m ∧ m < 0) :=
by
  sorry

end NUMINAMATH_GPT_exists_real_root_iff_l907_90707


namespace NUMINAMATH_GPT_probability_part_not_scrap_l907_90788

noncomputable def probability_not_scrap : Prop :=
  let p_scrap_first := 0.01
  let p_scrap_second := 0.02
  let p_not_scrap_first := 1 - p_scrap_first
  let p_not_scrap_second := 1 - p_scrap_second
  let p_not_scrap := p_not_scrap_first * p_not_scrap_second
  p_not_scrap = 0.9702

theorem probability_part_not_scrap : probability_not_scrap :=
by simp [probability_not_scrap] ; sorry

end NUMINAMATH_GPT_probability_part_not_scrap_l907_90788


namespace NUMINAMATH_GPT_row_speed_with_stream_l907_90731

theorem row_speed_with_stream (v : ℝ) (s : ℝ) (h1 : s = 2) (h2 : v - s = 12) : v + s = 16 := by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_row_speed_with_stream_l907_90731


namespace NUMINAMATH_GPT_acute_angle_inequality_l907_90782

theorem acute_angle_inequality (a b : ℝ) (α β : ℝ) (γ : ℝ) (h : γ < π / 2) :
  (a^2 + b^2) * Real.cos (α - β) ≤ 2 * a * b :=
sorry

end NUMINAMATH_GPT_acute_angle_inequality_l907_90782


namespace NUMINAMATH_GPT_set_intersection_nonempty_l907_90748

theorem set_intersection_nonempty {a : ℕ} (h : ({0, a} ∩ {1, 2} : Set ℕ) ≠ ∅) :
  a = 1 ∨ a = 2 := by
  sorry

end NUMINAMATH_GPT_set_intersection_nonempty_l907_90748


namespace NUMINAMATH_GPT_cake_flour_amount_l907_90781

theorem cake_flour_amount (sugar_cups : ℕ) (flour_already_in : ℕ) (extra_flour_needed : ℕ) (total_flour : ℕ) 
  (h1 : sugar_cups = 7) 
  (h2 : flour_already_in = 2)
  (h3 : extra_flour_needed = 2)
  (h4 : total_flour = sugar_cups + extra_flour_needed) : 
  total_flour = 9 := 
sorry

end NUMINAMATH_GPT_cake_flour_amount_l907_90781


namespace NUMINAMATH_GPT_percentage_voting_for_biff_equals_45_l907_90706

variable (total : ℕ) (votingForMarty : ℕ) (undecidedPercent : ℝ)

theorem percentage_voting_for_biff_equals_45 :
  total = 200 →
  votingForMarty = 94 →
  undecidedPercent = 0.08 →
  let totalDecided := (1 - undecidedPercent) * total
  let votingForBiff := totalDecided - votingForMarty
  let votingForBiffPercent := (votingForBiff / total) * 100
  votingForBiffPercent = 45 :=
by
  intros h1 h2 h3
  let totalDecided := (1 - 0.08 : ℝ) * 200
  let votingForBiff := totalDecided - 94
  let votingForBiffPercent := (votingForBiff / 200) * 100
  sorry

end NUMINAMATH_GPT_percentage_voting_for_biff_equals_45_l907_90706


namespace NUMINAMATH_GPT_laundry_per_hour_l907_90793

-- Definitions based on the conditions
def total_laundry : ℕ := 80
def total_hours : ℕ := 4

-- Theorems to prove the number of pieces per hour
theorem laundry_per_hour : total_laundry / total_hours = 20 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_laundry_per_hour_l907_90793


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l907_90774

theorem quadratic_inequality_solution (x : ℝ) : 
    (x^2 - 3*x - 4 > 0) ↔ (x < -1 ∨ x > 4) :=
sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l907_90774


namespace NUMINAMATH_GPT_rectangular_prism_diagonal_inequality_l907_90789

variable (a b c l : ℝ)

theorem rectangular_prism_diagonal_inequality (h : l^2 = a^2 + b^2 + c^2) : 
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := sorry

end NUMINAMATH_GPT_rectangular_prism_diagonal_inequality_l907_90789


namespace NUMINAMATH_GPT_num_roots_l907_90778

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 3 * x - 2

theorem num_roots : ∃! x : ℝ, f x = 0 := 
sorry

end NUMINAMATH_GPT_num_roots_l907_90778


namespace NUMINAMATH_GPT_ratio_of_numbers_l907_90739

-- Definitions for the conditions
variable (S L : ℕ)

-- Given conditions
def condition1 : Prop := S + L = 44
def condition2 : Prop := S = 20
def condition3 : Prop := L = 6 * S

-- The theorem to be proven
theorem ratio_of_numbers (h1 : condition1 S L) (h2 : condition2 S) (h3 : condition3 S L) : L / S = 6 := 
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l907_90739


namespace NUMINAMATH_GPT_container_capacity_l907_90787

theorem container_capacity
  (C : ℝ)  -- Total capacity of the container in liters
  (h1 : C / 2 + 20 = 3 * C / 4)  -- Condition combining the water added and the fractional capacities
  : C = 80 := 
sorry

end NUMINAMATH_GPT_container_capacity_l907_90787


namespace NUMINAMATH_GPT_divisible_by_65_l907_90719

theorem divisible_by_65 (n : ℕ) : 65 ∣ (5^n * (2^(2*n) - 3^n) + 2^n - 7^n) :=
sorry

end NUMINAMATH_GPT_divisible_by_65_l907_90719


namespace NUMINAMATH_GPT_locus_of_M_equation_of_l_l907_90715
open Real

-- Step 1: Define the given circles
def circle_F1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle_F2 (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 36

-- Step 2: Define the condition of tangency for the moving circle M
def external_tangent_F1 (cx cy r : ℝ) : Prop := (cx + 2)^2 + cy^2 = (2 + r)^2
def internal_tangent_F2 (cx cy r : ℝ) : Prop := (cx - 2)^2 + cy^2 = (6 - r)^2

-- Step 4: Prove the locus C is an ellipse with the equation excluding x = -4
theorem locus_of_M (cx cy : ℝ) : 
  (∃ r : ℝ, external_tangent_F1 cx cy r ∧ internal_tangent_F2 cx cy r) ↔
  (cx ≠ -4 ∧ (cx^2) / 16 + (cy^2) / 12 = 1) :=
sorry

-- Step 5: Define the conditions for the midpoint of segment AB
def midpoint_Q (x1 y1 x2 y2 : ℝ) : Prop := (x1 + x2) / 2 = 2 ∧ (y1 + y2) / 2 = -1

-- Step 6: Prove the equation of line l
theorem equation_of_l (x1 y1 x2 y2 : ℝ) (h1 : midpoint_Q x1 y1 x2 y2) 
  (h2 : (x1^2 / 16 + y1^2 / 12 = 1) ∧ (x2^2 / 16 + y2^2 / 12 = 1)) :
  3 * (x1 - x2) - 2 * (y1 - y2) = 8 :=
sorry

end NUMINAMATH_GPT_locus_of_M_equation_of_l_l907_90715


namespace NUMINAMATH_GPT_max_distance_circle_to_line_l907_90717

-- Definitions for the circle and line
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 2 * y = 0
def line_eq (x y : ℝ) : Prop := x + y + 2 = 0

-- Proof statement
theorem max_distance_circle_to_line 
  (x y : ℝ)
  (h_circ : circle_eq x y)
  (h_line : ∀ (x y : ℝ), line_eq x y → true) :
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_max_distance_circle_to_line_l907_90717


namespace NUMINAMATH_GPT_race_outcomes_210_l907_90727

-- Define the participants
def participants : List String := ["Abe", "Bobby", "Charles", "Devin", "Edwin", "Fern", "Grace"]

-- The question is to prove the number of different 1st-2nd-3rd place outcomes is 210.
theorem race_outcomes_210 (h : participants.length = 7) : (7 * 6 * 5 = 210) :=
  by sorry

end NUMINAMATH_GPT_race_outcomes_210_l907_90727


namespace NUMINAMATH_GPT_final_purchase_price_correct_l907_90794

-- Definitions
def initial_house_value : ℝ := 100000
def profit_percentage_Mr_Brown : ℝ := 0.10
def renovation_percentage : ℝ := 0.05
def profit_percentage_Mr_Green : ℝ := 0.07
def loss_percentage_Mr_Brown : ℝ := 0.10

-- Calculations
def purchase_price_mr_brown : ℝ := initial_house_value * (1 + profit_percentage_Mr_Brown)
def total_cost_mr_brown : ℝ := purchase_price_mr_brown * (1 + renovation_percentage)
def purchase_price_mr_green : ℝ := total_cost_mr_brown * (1 + profit_percentage_Mr_Green)
def final_purchase_price_mr_brown : ℝ := purchase_price_mr_green * (1 - loss_percentage_Mr_Brown)

-- Statement to prove
theorem final_purchase_price_correct : 
  final_purchase_price_mr_brown = 111226.50 :=
by
  sorry -- Proof is omitted

end NUMINAMATH_GPT_final_purchase_price_correct_l907_90794


namespace NUMINAMATH_GPT_negation_of_universal_proposition_l907_90711

theorem negation_of_universal_proposition :
  (∀ x : ℝ, x^2 + 1 > 0) → ¬(∃ x : ℝ, x^2 + 1 ≤ 0) := sorry

end NUMINAMATH_GPT_negation_of_universal_proposition_l907_90711


namespace NUMINAMATH_GPT_A_remaining_time_equals_B_remaining_time_l907_90702

variable (d_A d_B remaining_Distance_A remaining_Time_A remaining_Distance_B remaining_Time_B total_Distance : ℝ)

-- Given conditions as definitions
def A_traveled_more : d_A = d_B + 180 := sorry
def total_distance_between_X_Y : total_Distance = 900 := sorry
def sum_distance_traveled : d_A + d_B = total_Distance := sorry
def B_remaining_time : remaining_Time_B = 4.5 := sorry
def B_remaining_distance : remaining_Distance_B = total_Distance - d_B := sorry

-- Prove that: A travels the same remaining distance in the same time as B
theorem A_remaining_time_equals_B_remaining_time :
  remaining_Distance_A = remaining_Distance_B ∧ remaining_Time_A = remaining_Time_B := sorry

end NUMINAMATH_GPT_A_remaining_time_equals_B_remaining_time_l907_90702


namespace NUMINAMATH_GPT_part1_part2_l907_90753

def f (x a : ℝ) := |x - a| + x

theorem part1 (a : ℝ) (h_a : a = 1) : 
  {x : ℝ | f x a ≥ x + 2} = {x | x ≥ 3} ∪ {x | x ≤ -1} := 
by
  sorry

theorem part2 (a : ℝ) (h : {x : ℝ | f x a ≤ 3 * x} = {x | x ≥ 2}) : 
  a = 6 := 
by
  sorry

end NUMINAMATH_GPT_part1_part2_l907_90753


namespace NUMINAMATH_GPT_john_annual_payment_l907_90736

open Real

-- Definitions extracted from the problem:
def epipen_cost : ℝ := 500
def insurance_coverage : ℝ := 0.75
def epipen_frequency_per_year : ℕ := 2
def john_payment_per_epipen : ℝ := epipen_cost * (1 - insurance_coverage)

-- The statement to be proved:
theorem john_annual_payment :
  john_payment_per_epipen * epipen_frequency_per_year = 250 :=
by
  sorry

end NUMINAMATH_GPT_john_annual_payment_l907_90736


namespace NUMINAMATH_GPT_car_speed_first_hour_l907_90767

theorem car_speed_first_hour (x : ℕ) (h1 : 60 > 0) (h2 : 40 > 0) (h3 : 2 > 0) (avg_speed : 40 = (x + 60) / 2) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_car_speed_first_hour_l907_90767


namespace NUMINAMATH_GPT_fixed_point_tangent_circle_l907_90741

theorem fixed_point_tangent_circle (x y a b t : ℝ) :
  (x ^ 2 + (y - 2) ^ 2 = 16) ∧ (a * 0 + b * 2 - 12 = 0) ∧ (y = -6) ∧ 
  (t * x - 8 * y = 0) → 
  (0, 0) = (0, 0) :=
by 
  sorry

end NUMINAMATH_GPT_fixed_point_tangent_circle_l907_90741


namespace NUMINAMATH_GPT_original_triangle_area_l907_90738

-- Define the variables
variable (A_new : ℝ) (r : ℝ)

-- The conditions from the problem
def conditions := r = 5 ∧ A_new = 100

-- Goal: Prove that the original area is 4
theorem original_triangle_area (A_orig : ℝ) (h : conditions r A_new) : A_orig = 4 := by
  sorry

end NUMINAMATH_GPT_original_triangle_area_l907_90738


namespace NUMINAMATH_GPT_remainder_of_power_mod_l907_90795

theorem remainder_of_power_mod :
  ∀ (x n m : ℕ), 
  x = 5 → n = 2021 → m = 17 →
  x^n % m = 11 := by
sorry

end NUMINAMATH_GPT_remainder_of_power_mod_l907_90795


namespace NUMINAMATH_GPT_carson_clawed_39_times_l907_90759

def wombats_count := 9
def wombat_claws_per := 4
def rheas_count := 3
def rhea_claws_per := 1

def wombat_total_claws := wombats_count * wombat_claws_per
def rhea_total_claws := rheas_count * rhea_claws_per
def total_claws := wombat_total_claws + rhea_total_claws

theorem carson_clawed_39_times : total_claws = 39 :=
  by sorry

end NUMINAMATH_GPT_carson_clawed_39_times_l907_90759


namespace NUMINAMATH_GPT_coordinates_of_point_l907_90726

theorem coordinates_of_point (a : ℝ) (h : a - 3 = 0) : (a + 2, a - 3) = (5, 0) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_point_l907_90726


namespace NUMINAMATH_GPT_units_digit_G_2000_l907_90765

-- Define the sequence G
def G (n : ℕ) : ℕ := 2 ^ (2 ^ n) + 5 ^ (5 ^ n)

-- The main goal is to show that the units digit of G 2000 is 1
theorem units_digit_G_2000 : (G 2000) % 10 = 1 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_G_2000_l907_90765


namespace NUMINAMATH_GPT_xy_divides_x2_plus_y2_plus_one_l907_90720

theorem xy_divides_x2_plus_y2_plus_one 
    (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (h : (x * y) ∣ (x^2 + y^2 + 1)) :
  (x^2 + y^2 + 1) / (x * y) = 3 := by
  sorry

end NUMINAMATH_GPT_xy_divides_x2_plus_y2_plus_one_l907_90720


namespace NUMINAMATH_GPT_contrapositive_of_zero_implication_l907_90776

theorem contrapositive_of_zero_implication (a b : ℝ) :
  (a = 0 ∨ b = 0 → a * b = 0) → (a * b ≠ 0 → (a ≠ 0 ∧ b ≠ 0)) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_contrapositive_of_zero_implication_l907_90776


namespace NUMINAMATH_GPT_sin_inequality_of_triangle_l907_90723

theorem sin_inequality_of_triangle (B C : ℝ) (hB : 0 < B) (hB_lt_pi : B < π) 
(hC : 0 < C) (hC_lt_pi : C < π) :
  (B > C) ↔ (Real.sin B > Real.sin C) := 
  sorry

end NUMINAMATH_GPT_sin_inequality_of_triangle_l907_90723


namespace NUMINAMATH_GPT_eval_expression_l907_90751

theorem eval_expression :
  let a := 3
  let b := 2
  (2 ^ a ∣ 200) ∧ ¬(2 ^ (a + 1) ∣ 200) ∧ (5 ^ b ∣ 200) ∧ ¬(5 ^ (b + 1) ∣ 200)
→ (1 / 3)^(b - a) = 3 :=
by sorry

end NUMINAMATH_GPT_eval_expression_l907_90751


namespace NUMINAMATH_GPT_betty_total_oranges_l907_90722

-- Definitions for the given conditions
def boxes : ℝ := 3.0
def oranges_per_box : ℝ := 24

-- Theorem statement to prove the correct answer to the problem
theorem betty_total_oranges : boxes * oranges_per_box = 72 := by
  sorry

end NUMINAMATH_GPT_betty_total_oranges_l907_90722


namespace NUMINAMATH_GPT_find_m_l907_90744

theorem find_m (m : ℝ) :
  (∃ x : ℝ, (m - 1) * x^2 + 2 * x + m^2 - 1 = 0) ∧ (m - 1 ≠ 0) → m = -1 :=
by
  sorry

end NUMINAMATH_GPT_find_m_l907_90744


namespace NUMINAMATH_GPT_solve_for_x_l907_90750

theorem solve_for_x : (42 / (7 - 3 / 7) = 147 / 23) :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l907_90750


namespace NUMINAMATH_GPT_path_traveled_by_A_l907_90768

-- Define the initial conditions
def RectangleABCD (A B C D : ℝ × ℝ) :=
  dist A B = 3 ∧ dist C D = 3 ∧ dist B C = 5 ∧ dist D A = 5

-- Define the transformations
def rotated90Clockwise (D : ℝ × ℝ) (A : ℝ × ℝ) (A' : ℝ × ℝ) : Prop :=
  -- 90-degree clockwise rotation moves point A to A'
  A' = (D.1 + D.2 - A.2, D.2 - D.1 + A.1)

def translated3AlongDC (D C A' : ℝ × ℝ) (A'' : ℝ × ℝ) : Prop :=
  -- Translation by 3 units along line DC moves point A' to A''
  A'' = (A'.1 - 3, A'.2)

-- Define the total path traveled
noncomputable def totalPathTraveled (rotatedPath translatedPath : ℝ) : ℝ :=
  rotatedPath + translatedPath

-- Prove the total path is 2.5*pi + 3
theorem path_traveled_by_A (A B C D A' A'' : ℝ × ℝ) (hRect : RectangleABCD A B C D) (hRotate : rotated90Clockwise D A A') (hTranslate : translated3AlongDC D C A' A'') :
  totalPathTraveled (2.5 * Real.pi) 3 = (2.5 * Real.pi + 3) := by
  sorry

end NUMINAMATH_GPT_path_traveled_by_A_l907_90768


namespace NUMINAMATH_GPT_Benny_total_hours_l907_90735

def hours_per_day : ℕ := 7
def days_worked : ℕ := 14

theorem Benny_total_hours : hours_per_day * days_worked = 98 := by
  sorry

end NUMINAMATH_GPT_Benny_total_hours_l907_90735


namespace NUMINAMATH_GPT_m_div_x_eq_4_div_5_l907_90712

variable (a b : ℝ)
variable (h_pos_a : 0 < a)
variable (h_pos_b : 0 < b)
variable (h_ratio : a / b = 4 / 5)

def x := a * 1.25

def m := b * 0.80

theorem m_div_x_eq_4_div_5 : m / x = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_m_div_x_eq_4_div_5_l907_90712


namespace NUMINAMATH_GPT_range_of_a_l907_90784

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x + a| > 2) ↔ a < -1 ∨ a > 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l907_90784


namespace NUMINAMATH_GPT_smallest_perimeter_of_square_sides_l907_90705

/-
  Define a predicate for the triangle inequality condition for squares of integers.
-/
def triangle_ineq_squares (a b c : ℕ) : Prop :=
  (a < b) ∧ (b < c) ∧ (a^2 + b^2 > c^2) ∧ (a^2 + c^2 > b^2) ∧ (b^2 + c^2 > a^2)

/-
  Statement that proves the smallest possible perimeter given the conditions.
-/
theorem smallest_perimeter_of_square_sides : 
  ∃ a b c : ℕ, a < b ∧ b < c ∧ triangle_ineq_squares a b c ∧ a^2 + b^2 + c^2 = 77 :=
sorry

end NUMINAMATH_GPT_smallest_perimeter_of_square_sides_l907_90705


namespace NUMINAMATH_GPT_max_k_value_l907_90796

def A : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

noncomputable def B (i : ℕ) := {b : Finset ℕ // b ⊆ A ∧ b ≠ ∅ ∧ ∀ j ≠ i, ∃ k : Finset ℕ, k ⊆ A ∧ k ≠ ∅ ∧ (b ∩ k).card ≤ 2}

theorem max_k_value : ∃ k, k = 175 :=
  by
    sorry

end NUMINAMATH_GPT_max_k_value_l907_90796


namespace NUMINAMATH_GPT_number_of_students_l907_90799

-- Define John's total winnings
def john_total_winnings : ℤ := 155250

-- Define the proportion of winnings given to each student
def proportion_per_student : ℚ := 1 / 1000

-- Define the total amount received by students
def total_received_by_students : ℚ := 15525

-- Calculate the amount each student received
def amount_per_student : ℚ := john_total_winnings * proportion_per_student

-- Theorem to prove the number of students
theorem number_of_students : total_received_by_students / amount_per_student = 100 :=
by
  -- Lean will be expected to fill in this proof
  sorry

end NUMINAMATH_GPT_number_of_students_l907_90799


namespace NUMINAMATH_GPT_two_digit_number_l907_90777

theorem two_digit_number (a : ℕ) (N M : ℕ) :
  (10 ≤ a) ∧ (a ≤ 99) ∧ (2 * a + 1 = N^2) ∧ (3 * a + 1 = M^2) → a = 40 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_number_l907_90777


namespace NUMINAMATH_GPT_min_value_proof_l907_90728

noncomputable def min_value (x y : ℝ) : ℝ :=
  (y / x) + (1 / y)

theorem min_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + 2 * y = 1) : 
  (min_value x y) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_proof_l907_90728


namespace NUMINAMATH_GPT_complement_union_l907_90733

open Set

universe u

variable {U : Type u} [Fintype U] [DecidableEq U]
variable {A B : Set U}

def complement (s : Set U) : Set U := {x | x ∉ s}

theorem complement_union {U : Set ℕ} (A B : Set ℕ) 
  (h1 : complement A ∩ B = {1})
  (h2 : A ∩ B = {3})
  (h3 : complement A ∩ complement B = {2}) :
  complement (A ∪ B) = {2} :=
by sorry

end NUMINAMATH_GPT_complement_union_l907_90733


namespace NUMINAMATH_GPT_contrapositive_proposition_contrapositive_equiv_l907_90757

theorem contrapositive_proposition (x : ℝ) (h : -1 < x ∧ x < 1) : (x^2 < 1) :=
sorry

theorem contrapositive_equiv (x : ℝ) (h : x^2 ≥ 1) : x ≥ 1 ∨ x ≤ -1 :=
sorry

end NUMINAMATH_GPT_contrapositive_proposition_contrapositive_equiv_l907_90757
