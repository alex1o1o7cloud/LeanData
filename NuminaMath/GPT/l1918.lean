import Mathlib

namespace circle_center_l1918_191837

theorem circle_center (n : ℝ) (r : ℝ) (h1 : r = 7) (h2 : ∀ x : ℝ, x^2 + (x^2 - n)^2 = 49 → x^4 - x^2 * (2*n - 1) + n^2 - 49 = 0)
  (h3 : ∃! y : ℝ, y^2 + (1 - 2*n) * y + n^2 - 49 = 0) :
  (0, n) = (0, 197 / 4) := 
sorry

end circle_center_l1918_191837


namespace paint_cans_for_25_rooms_l1918_191828

theorem paint_cans_for_25_rooms (cans rooms : ℕ) (H1 : cans * 30 = rooms) (H2 : cans * 25 = rooms - 5 * cans) :
  cans = 15 :=
by
  sorry

end paint_cans_for_25_rooms_l1918_191828


namespace female_athletes_in_sample_l1918_191817

theorem female_athletes_in_sample (M F S : ℕ) (hM : M = 56) (hF : F = 42) (hS : S = 28) :
  (F * (S / (M + F))) = 12 :=
by
  rw [hM, hF, hS]
  norm_num
  sorry

end female_athletes_in_sample_l1918_191817


namespace power_mod_congruence_l1918_191876

theorem power_mod_congruence (h : 3^400 ≡ 1 [MOD 500]) : 3^800 ≡ 1 [MOD 500] :=
by {
  sorry
}

end power_mod_congruence_l1918_191876


namespace neg_p_is_necessary_but_not_sufficient_for_neg_q_l1918_191869

variables (p q : Prop)

-- Given conditions: (p → q) and ¬(q → p)
theorem neg_p_is_necessary_but_not_sufficient_for_neg_q
  (h1 : p → q)
  (h2 : ¬ (q → p)) :
  (¬ p → ¬ q) ∧ ¬ (¬ p ↔ ¬ q) :=
sorry

end neg_p_is_necessary_but_not_sufficient_for_neg_q_l1918_191869


namespace jerry_bought_3_pounds_l1918_191824

-- Definitions based on conditions:
def cost_mustard_oil := 2 * 13
def cost_pasta_sauce := 5
def total_money := 50
def money_left := 7
def cost_gluten_free_pasta_per_pound := 4

-- The proof goal based on the correct answer:
def pounds_gluten_free_pasta : Nat :=
  let total_spent := total_money - money_left
  let spent_on_mustard_and_sauce := cost_mustard_oil + cost_pasta_sauce
  let spent_on_pasta := total_spent - spent_on_mustard_and_sauce
  spent_on_pasta / cost_gluten_free_pasta_per_pound

theorem jerry_bought_3_pounds :
  pounds_gluten_free_pasta = 3 := by
  -- the proof should follow here
  sorry

end jerry_bought_3_pounds_l1918_191824


namespace initial_percentage_of_water_is_12_l1918_191834

noncomputable def initial_percentage_of_water (initial_volume : ℕ) (added_water : ℕ) (final_percentage : ℕ) : ℕ :=
  let final_volume := initial_volume + added_water
  let final_water_amount := (final_percentage * final_volume) / 100
  let initial_water_amount := final_water_amount - added_water
  (initial_water_amount * 100) / initial_volume

theorem initial_percentage_of_water_is_12 :
  initial_percentage_of_water 20 2 20 = 12 :=
by
  sorry

end initial_percentage_of_water_is_12_l1918_191834


namespace find_B_l1918_191897

noncomputable def A : ℝ := 1 / 49
noncomputable def C : ℝ := -(1 / 7)

theorem find_B :
  (∀ x : ℝ, 1 / (x^3 + 2 * x^2 - 25 * x - 50) 
            = (A / (x - 2)) + (B / (x + 5)) + (C / ((x + 5)^2))) 
    → B = - (11 / 490) :=
sorry

end find_B_l1918_191897


namespace opposite_signs_add_same_signs_sub_l1918_191853

-- Definitions based on the conditions
variables {a b : ℤ}

-- 1. Case when a and b have opposite signs
theorem opposite_signs_add (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b < 0) :
  a + b = 1 ∨ a + b = -1 := 
sorry

-- 2. Case when a and b have the same sign
theorem same_signs_sub (h₁ : |a| = 4) (h₂ : |b| = 3) (h₃ : a * b > 0) :
  a - b = 1 ∨ a - b = -1 := 
sorry

end opposite_signs_add_same_signs_sub_l1918_191853


namespace minimum_keys_needed_l1918_191809

theorem minimum_keys_needed (total_cabinets : ℕ) (boxes_per_cabinet : ℕ)
(boxes_needed : ℕ) (boxes_per_cabinet : ℕ) 
(warehouse_key : ℕ) (boxes_per_cabinet: ℕ)
(h1 : total_cabinets = 8)
(h2 : boxes_per_cabinet = 4)
(h3 : (boxes_needed = 52))
(h4 : boxes_per_cabinet = 4)
(h5 : warehouse_key = 1):
    6 + 2 + 1 = 9 := 
    sorry

end minimum_keys_needed_l1918_191809


namespace three_consecutive_odds_l1918_191878

theorem three_consecutive_odds (x : ℤ) (h3 : x + 4 = 133) : 
  x + (x + 4) = 3 * (x + 2) - 131 := 
by {
  sorry
}

end three_consecutive_odds_l1918_191878


namespace find_m_l1918_191814

-- Define vectors as tuples
def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (m, -1)
def c (m : ℝ) : ℝ × ℝ := (4, m)

-- Define vector subtraction
def sub_vect (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)

-- Define dot product
def dot_prod (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove the condition that (a - b) ⊥ c implies m = 4
theorem find_m (m : ℝ) (h : dot_prod (sub_vect a (b m)) (c m) = 0) : m = 4 :=
by
  sorry

end find_m_l1918_191814


namespace first_discount_percentage_l1918_191816

-- Definitions based on the conditions provided
def listed_price : ℝ := 400
def final_price : ℝ := 334.4
def additional_discount : ℝ := 5

-- The equation relating these quantities
theorem first_discount_percentage (D : ℝ) (h : listed_price * (1 - D / 100) * (1 - additional_discount / 100) = final_price) : D = 12 :=
sorry

end first_discount_percentage_l1918_191816


namespace max_candies_takeable_l1918_191815

theorem max_candies_takeable : 
  ∃ (max_take : ℕ), max_take = 159 ∧
  ∀ (boxes: Fin 5 → ℕ), 
    boxes 0 = 11 → 
    boxes 1 = 22 → 
    boxes 2 = 33 → 
    boxes 3 = 44 → 
    boxes 4 = 55 →
    (∀ (i : Fin 5), 
      ∀ (new_boxes : Fin 5 → ℕ),
      (new_boxes i = boxes i - 4) ∧ 
      (∀ (j : Fin 5), j ≠ i → new_boxes j = boxes j + 1) →
      boxes i = 0 → max_take = new_boxes i) :=
sorry

end max_candies_takeable_l1918_191815


namespace largest_angle_of_consecutive_interior_angles_pentagon_l1918_191844

theorem largest_angle_of_consecutive_interior_angles_pentagon (x : ℕ)
  (h1 : (x - 3) + (x - 2) + (x - 1) + x + (x + 1) = 540) :
  x + 1 = 110 := sorry

end largest_angle_of_consecutive_interior_angles_pentagon_l1918_191844


namespace cosine_relationship_l1918_191843

open Real

noncomputable def functional_relationship (x y : ℝ) : Prop :=
  y = -(4 / 5) * sqrt (1 - x ^ 2) + (3 / 5) * x

theorem cosine_relationship (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2)
  (h5 : cos (α + β) = - 4 / 5) (h6 : sin β = x) (h7 : cos α = y) (h8 : 4 / 5 < x) (h9 : x < 1) :
  functional_relationship x y :=
sorry

end cosine_relationship_l1918_191843


namespace regina_total_cost_l1918_191885

-- Definitions
def daily_cost : ℝ := 30
def mileage_cost : ℝ := 0.25
def days_rented : ℝ := 3
def miles_driven : ℝ := 450
def fixed_fee : ℝ := 15

-- Proposition for total cost
noncomputable def total_cost : ℝ := daily_cost * days_rented + mileage_cost * miles_driven + fixed_fee

-- Theorem statement
theorem regina_total_cost : total_cost = 217.5 := by
  sorry

end regina_total_cost_l1918_191885


namespace number_is_18_l1918_191831

theorem number_is_18 (x : ℝ) (h : (7 / 3) * x = 42) : x = 18 :=
sorry

end number_is_18_l1918_191831


namespace arithmetic_sequence_sum_l1918_191823

open Nat

theorem arithmetic_sequence_sum :
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  3 * S = 3774 := 
by
  let a := 50
  let d := 3
  let l := 98
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end arithmetic_sequence_sum_l1918_191823


namespace statement_A_statement_D_l1918_191838

variable (a b c d : ℝ)

-- Statement A: If ac² > bc², then a > b
theorem statement_A (h1 : a * c^2 > b * c^2) (h2 : c ≠ 0) : a > b := by
  sorry

-- Statement D: If a > b > 0, then a + 1/b > b + 1/a
theorem statement_D (h1 : a > b) (h2 : b > 0) : a + 1 / b > b + 1 / a := by
  sorry

end statement_A_statement_D_l1918_191838


namespace A_roster_method_l1918_191882

open Set

def A : Set ℤ := {x : ℤ | (∃ (n : ℤ), n > 0 ∧ 6 / (5 - x) = n) }

theorem A_roster_method :
  A = {-1, 2, 3, 4} :=
  sorry

end A_roster_method_l1918_191882


namespace minimum_value_w_l1918_191858

theorem minimum_value_w : 
  ∀ x y : ℝ, ∃ (w : ℝ), w = 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30 → w ≥ 26.25 :=
by
  intro x y
  use 2 * x^2 + 3 * y^2 + 8 * x - 5 * y + 30
  sorry

end minimum_value_w_l1918_191858


namespace displacement_during_interval_l1918_191864

noncomputable def velocity (t : ℝ) : ℝ := 3 * t^2 + 2 * t

theorem displacement_during_interval :
  (∫ t in (0 : ℝ)..3, velocity t) = 36 :=
by
  sorry

end displacement_during_interval_l1918_191864


namespace marching_band_formations_l1918_191841

theorem marching_band_formations :
  (∃ (s t : ℕ), s * t = 240 ∧ 8 ≤ t ∧ t ≤ 30) →
  ∃ (z : ℕ), z = 4 := sorry

end marching_band_formations_l1918_191841


namespace problem_l1918_191874

def f (x a b : ℝ) : ℝ := a * x ^ 3 - b * x + 1

theorem problem (a b : ℝ) (h : f 2 a b = -1) : f (-2) a b = 3 :=
by {
  sorry
}

end problem_l1918_191874


namespace identity_is_only_sum_free_preserving_surjection_l1918_191886

def is_surjective (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, ∃ m : ℕ, f m = n

def is_sum_free (A : Set ℕ) : Prop :=
  ∀ x y : ℕ, x ∈ A → y ∈ A → x + y ∉ A

noncomputable def identity_function_property : Prop :=
  ∀ f : ℕ → ℕ, is_surjective f →
  (∀ A : Set ℕ, is_sum_free A → is_sum_free (Set.image f A)) →
  ∀ n : ℕ, f n = n

theorem identity_is_only_sum_free_preserving_surjection : identity_function_property := sorry

end identity_is_only_sum_free_preserving_surjection_l1918_191886


namespace find_z_solutions_l1918_191898

open Real

noncomputable def is_solution (z : ℝ) : Prop :=
  sin z + sin (2 * z) + sin (3 * z) = cos z + cos (2 * z) + cos (3 * z)

theorem find_z_solutions (z : ℝ) : 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k - 1)) ∨ 
  (∃ k : ℤ, z = 2 * π / 3 * (3 * k + 1)) ∨ 
  (∃ k : ℤ, z = π / 8 * (4 * k + 1)) ↔
  is_solution z :=
by
  sorry

end find_z_solutions_l1918_191898


namespace f_three_l1918_191832

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_f_succ : ∀ x : ℝ, f (x + 1) = f (-x + 1)
axiom f_one : f 1 = 1 

-- Goal
theorem f_three : f 3 = -1 :=
by
  -- The proof will be provided here
  sorry

end f_three_l1918_191832


namespace set_A_range_l1918_191889

def A := {y : ℝ | ∃ x : ℝ, y = -x^2 ∧ (-1 ≤ x ∧ x ≤ 2)}

theorem set_A_range :
  A = {y : ℝ | -4 ≤ y ∧ y ≤ 0} :=
sorry

end set_A_range_l1918_191889


namespace no_such_b_c_exist_l1918_191833

theorem no_such_b_c_exist :
  ¬ ∃ (b c : ℝ), (∃ (k l : ℤ), (k ≠ l ∧ (k ^ 2 + b * ↑k + c = 0) ∧ (l ^ 2 + b * ↑l + c = 0))) ∧
                  (∃ (m n : ℤ), (m ≠ n ∧ (2 * (m ^ 2) + (b + 1) * ↑m + (c + 1) = 0) ∧ 
                                        (2 * (n ^ 2) + (b + 1) * ↑n + (c + 1) = 0))) :=
sorry

end no_such_b_c_exist_l1918_191833


namespace find_a_equidistant_l1918_191880

theorem find_a_equidistant :
  ∀ a : ℝ, (abs (a - 2) = abs (6 - 2 * a)) →
    (a = 8 / 3 ∨ a = 4) :=
by
  intro a h
  sorry

end find_a_equidistant_l1918_191880


namespace trapezoidal_field_perimeter_l1918_191865

-- Definitions derived from the conditions
def length_of_longer_parallel_side : ℕ := 15
def length_of_shorter_parallel_side : ℕ := 9
def total_perimeter_of_rectangle : ℕ := 52

-- Correct Answer
def correct_perimeter_of_trapezoidal_field : ℕ := 46

-- Theorem statement
theorem trapezoidal_field_perimeter 
  (a b w : ℕ)
  (h1 : a = length_of_longer_parallel_side)
  (h2 : b = length_of_shorter_parallel_side)
  (h3 : 2 * (a + w) = total_perimeter_of_rectangle)
  (h4 : w = 11) -- from the solution calculation
  : a + b + 2 * w = correct_perimeter_of_trapezoidal_field :=
by
  sorry

end trapezoidal_field_perimeter_l1918_191865


namespace sqrt_equation_solution_l1918_191872

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt ((2 + Real.sqrt 3) ^ x) + Real.sqrt ((2 - Real.sqrt 3) ^ x) = 4) ↔ (x = 2 ∨ x = -2) := 
by
  sorry

end sqrt_equation_solution_l1918_191872


namespace max_value_of_linear_combination_of_m_n_k_l1918_191893

-- The style grants us maximum flexibility for definitions.
theorem max_value_of_linear_combination_of_m_n_k 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (m n k : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i % 3 = 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → b i % 3 = 2)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ k → c i % 3 = 0)
  (h4 : Function.Injective a)
  (h5 : Function.Injective b)
  (h6 : Function.Injective c)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)
  (h_sum : (Finset.range m).sum a + (Finset.range n).sum b + (Finset.range k).sum c = 2007)
  : 4 * m + 3 * n + 5 * k ≤ 256 := by
  sorry

end max_value_of_linear_combination_of_m_n_k_l1918_191893


namespace polynomial_divisible_by_24_l1918_191852

-- Defining the function
def f (n : ℕ) : ℕ :=
n^4 + 2*n^3 + 11*n^2 + 10*n

-- Statement of the theorem
theorem polynomial_divisible_by_24 (n : ℕ) (h : n > 0) : f n % 24 = 0 :=
sorry

end polynomial_divisible_by_24_l1918_191852


namespace yellow_balls_count_l1918_191870

theorem yellow_balls_count (r y : ℕ) (h1 : r = 9) (h2 : (r : ℚ) / (r + y) = 1 / 3) : y = 18 := 
by
  sorry

end yellow_balls_count_l1918_191870


namespace base7_digits_of_143_l1918_191818

theorem base7_digits_of_143 : ∃ d1 d2 d3 : ℕ, (d1 < 7 ∧ d2 < 7 ∧ d3 < 7) ∧ (143 = d1 * 49 + d2 * 7 + d3) ∧ (d1 = 2 ∧ d2 = 6 ∧ d3 = 3) :=
by
  sorry

end base7_digits_of_143_l1918_191818


namespace fred_has_18_stickers_l1918_191812

def jerry_stickers := 36
def george_stickers (jerry : ℕ) := jerry / 3
def fred_stickers (george : ℕ) := george + 6

theorem fred_has_18_stickers :
  let j := jerry_stickers
  let g := george_stickers j 
  fred_stickers g = 18 :=
by
  sorry

end fred_has_18_stickers_l1918_191812


namespace heather_biked_per_day_l1918_191856

def total_kilometers_biked : ℝ := 320
def days_biked : ℝ := 8
def kilometers_per_day : ℝ := 40

theorem heather_biked_per_day : total_kilometers_biked / days_biked = kilometers_per_day := 
by
  -- Proof will be inserted here
  sorry

end heather_biked_per_day_l1918_191856


namespace y_sum_equals_three_l1918_191891

noncomputable def sum_of_y_values (solutions : List (ℝ × ℝ × ℝ)) : ℝ :=
  solutions.foldl (fun acc (_, y, _) => acc + y) 0

theorem y_sum_equals_three (solutions : List (ℝ × ℝ × ℝ))
  (h1 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → x + y * z = 5)
  (h2 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → y + x * z = 8)
  (h3 : ∀ (x y z : ℝ), (x, y, z) ∈ solutions → z + x * y = 12) :
  sum_of_y_values solutions = 3 := sorry

end y_sum_equals_three_l1918_191891


namespace problem_equivalent_l1918_191846

theorem problem_equivalent :
  500 * 2019 * 0.0505 * 20 = 2019^2 :=
by
  sorry

end problem_equivalent_l1918_191846


namespace chess_pieces_missing_l1918_191896

theorem chess_pieces_missing 
  (total_pieces : ℕ) (pieces_present : ℕ) (h1 : total_pieces = 32) (h2 : pieces_present = 28) : 
  total_pieces - pieces_present = 4 := 
by
  -- Sorry proof
  sorry

end chess_pieces_missing_l1918_191896


namespace neg_p_equiv_l1918_191863

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x / (x - 1) > 0

theorem neg_p_equiv :
  ¬p I ↔ ∃ x ∈ I, x / (x - 1) ≤ 0 ∨ x - 1 = 0 :=
by
  sorry

end neg_p_equiv_l1918_191863


namespace exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l1918_191854

theorem exists_n_such_that_5_pow_n_has_six_consecutive_zeros :
  ∃ n : ℕ, n < 1000000 ∧ ∃ k : ℕ, k = 20 ∧ 5 ^ n % (10 ^ k) < (10 ^ (k - 6)) :=
by
  -- proof goes here
  sorry

end exists_n_such_that_5_pow_n_has_six_consecutive_zeros_l1918_191854


namespace proof_ab_value_l1918_191894

theorem proof_ab_value (a b : ℝ) (h1 : a - b = 3) (h2 : a^2 + b^2 = 29) : a * b = 10 := 
by
  sorry

end proof_ab_value_l1918_191894


namespace fraction_zero_condition_l1918_191825

theorem fraction_zero_condition (x : ℝ) (h : (abs x - 2) / (2 - x) = 0) : x = -2 :=
by
  sorry

end fraction_zero_condition_l1918_191825


namespace circle_intersection_range_l1918_191840

noncomputable def circle1_eq (x y : ℝ) : Prop := x^2 + y^2 = 25
noncomputable def circle2_eq (x y r : ℝ) : Prop := (x - 7)^2 + y^2 = r^2

theorem circle_intersection_range (r : ℝ) (h : r > 0) :
  (∃ x y : ℝ, circle1_eq x y ∧ circle2_eq x y r) ↔ 2 < r ∧ r < 12 :=
sorry

end circle_intersection_range_l1918_191840


namespace sin_product_identity_l1918_191819

noncomputable def sin_15_deg := Real.sin (15 * Real.pi / 180)
noncomputable def sin_30_deg := Real.sin (30 * Real.pi / 180)
noncomputable def sin_75_deg := Real.sin (75 * Real.pi / 180)

theorem sin_product_identity :
  sin_15_deg * sin_30_deg * sin_75_deg = 1 / 8 :=
by
  sorry

end sin_product_identity_l1918_191819


namespace ellipse_AB_length_l1918_191820

theorem ellipse_AB_length :
  ∀ (F1 F2 A B : ℝ × ℝ) (x y : ℝ),
  (x^2 / 25 + y^2 / 9 = 1) →
  (F1 = (5, 0) ∨ F1 = (-5, 0)) →
  (F2 = (if F1 = (5, 0) then (-5, 0) else (5, 0))) →
  ({p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} A ∨ {p : ℝ × ℝ | p.1^2 / 25 + p.2^2 / 9 = 1} B) →
  ((A = F1) ∨ (B = F1)) →
  (abs (F2.1 - A.1) + abs (F2.2 - A.2) + abs (F2.1 - B.1) + abs (F2.2 - B.2) = 12) →
  abs (A.1 - B.1) + abs (A.2 - B.2) = 8 :=
by
  sorry

end ellipse_AB_length_l1918_191820


namespace ab_value_l1918_191811

theorem ab_value (a b : ℝ) (h1 : a + b = 8) (h2 : a^3 + b^3 = 172) : ab = 85 / 6 := 
by
  sorry

end ab_value_l1918_191811


namespace pool_people_count_l1918_191800

theorem pool_people_count (P : ℕ) (total_money : ℝ) (cost_per_person : ℝ) (leftover_money : ℝ) 
  (h1 : total_money = 30) 
  (h2 : cost_per_person = 2.50) 
  (h3 : leftover_money = 5) 
  (h4 : total_money - leftover_money = cost_per_person * P) : 
  P = 10 :=
sorry

end pool_people_count_l1918_191800


namespace p_sufficient_not_necessary_q_l1918_191851

-- Define the conditions p and q
def p (x : ℝ) : Prop := 2 < x ∧ x < 4
def q (x : ℝ) : Prop := x > 2 ∨ x < -3

-- Prove the relationship between p and q
theorem p_sufficient_not_necessary_q : 
  (∀ x : ℝ, p x → q x) ∧ ¬(∀ x : ℝ, q x → p x) :=
by
  sorry

end p_sufficient_not_necessary_q_l1918_191851


namespace poly_eq_l1918_191803

-- Definition of the polynomials f(x) and g(x)
def f (x : ℝ) := x^4 + 4*x^3 + 8*x
def g (x : ℝ) := 10*x^4 + 30*x^3 + 29*x^2 + 2*x + 5

-- Define p(x) as a function that satisfies the given condition
def p (x : ℝ) := 9*x^4 + 26*x^3 + 29*x^2 - 6*x + 5

-- Prove that the function p(x) satisfies the equation
theorem poly_eq : ∀ x : ℝ, p x + f x = g x :=
by
  intro x
  -- Add a marker to indicate that this is where the proof would go
  sorry

end poly_eq_l1918_191803


namespace solve_fractional_equation_l1918_191883

theorem solve_fractional_equation (x : ℝ) (h₀ : 2 = 3 * (x + 1) / (4 - x)) : x = 1 :=
sorry

end solve_fractional_equation_l1918_191883


namespace cos_neg_pi_over_3_l1918_191895

noncomputable def angle := - (Real.pi / 3)

theorem cos_neg_pi_over_3 : Real.cos angle = 1 / 2 :=
by
  sorry

end cos_neg_pi_over_3_l1918_191895


namespace f_alpha_l1918_191808

variables (α : Real) (x : Real)

noncomputable def f (x : Real) : Real := 
  (Real.cos (Real.pi + x) * Real.sin (2 * Real.pi - x)) / Real.cos (Real.pi - x)

lemma sin_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) : 
  Real.sin α = 2 * Real.sqrt 2 / 3 :=
sorry

lemma tan_alpha {α : Real} (hsin : Real.sin α = 2 * Real.sqrt 2 / 3) (hcos : Real.cos α = 1 / 3) :
  Real.tan α = 2 * Real.sqrt 2 :=
sorry

theorem f_alpha {α : Real} (hcos : Real.cos α = 1 / 3) (hα : 0 < α ∧ α < Real.pi) :
  f α = -2 * Real.sqrt 2 / 3 :=
sorry

end f_alpha_l1918_191808


namespace smallest_positive_integer_n_l1918_191879

def contains_digit_9 (n : ℕ) : Prop := 
  ∃ m : ℕ, (10^m) ∣ n ∧ (n / 10^m) % 10 = 9

theorem smallest_positive_integer_n :
  ∃ n : ℕ, (∀ k : ℕ, k > 0 ∧ k < n → 
  (∃ a b : ℕ, k = 2^a * 5^b * 3) ∧ contains_digit_9 k ∧ (k % 3 = 0))
  → n = 90 :=
sorry

end smallest_positive_integer_n_l1918_191879


namespace distance_greater_than_two_l1918_191807

theorem distance_greater_than_two (x : ℝ) (h : |x| > 2) : x > 2 ∨ x < -2 :=
sorry

end distance_greater_than_two_l1918_191807


namespace distance_between_hyperbola_vertices_l1918_191821

theorem distance_between_hyperbola_vertices :
  ∀ (x y : ℝ), (x^2 / 121 - y^2 / 49 = 1) → (22 = 2 * 11) :=
by
  sorry

end distance_between_hyperbola_vertices_l1918_191821


namespace ratio_markus_age_son_age_l1918_191813

variable (M S G : ℕ)

theorem ratio_markus_age_son_age (h1 : G = 20) (h2 : S = 2 * G) (h3 : M + S + G = 140) : M / S = 2 := by
  sorry

end ratio_markus_age_son_age_l1918_191813


namespace range_of_a_monotonically_decreasing_l1918_191829

-- Definitions
def f (a x : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

-- Lean statement
theorem range_of_a_monotonically_decreasing {a : ℝ} : 
  (∀ x y : ℝ, -2 ≤ x → x ≤ 4 → -2 ≤ y → y ≤ 4 → x < y → f a y < f a x) ↔ a ≤ -3 := 
by 
  sorry

end range_of_a_monotonically_decreasing_l1918_191829


namespace factor_expression_l1918_191887

theorem factor_expression (x : ℝ) : 5 * x * (x + 2) + 9 * (x + 2) = (x + 2) * (5 * x + 9) :=
by
  sorry

end factor_expression_l1918_191887


namespace sum_and_times_l1918_191845

theorem sum_and_times 
  (a : ℕ) (ha : a = 99) 
  (b : ℕ) (hb : b = 301) 
  (c : ℕ) (hc : c = 200) : 
  a + b = 2 * c :=
by 
  -- skipping proof 
  sorry

end sum_and_times_l1918_191845


namespace f_g_5_eq_163_l1918_191826

def g (x : ℤ) : ℤ := 4 * x + 9
def f (x : ℤ) : ℤ := 6 * x - 11

theorem f_g_5_eq_163 : f (g 5) = 163 := by
  sorry

end f_g_5_eq_163_l1918_191826


namespace complex_exponential_sum_identity_l1918_191850

theorem complex_exponential_sum_identity :
    12 * Complex.exp (Real.pi * Complex.I / 7) + 12 * Complex.exp (19 * Real.pi * Complex.I / 14) =
    24 * Real.cos (5 * Real.pi / 28) * Complex.exp (3 * Real.pi * Complex.I / 4) :=
sorry

end complex_exponential_sum_identity_l1918_191850


namespace dorothy_money_left_l1918_191888

def annual_income : ℝ := 60000
def tax_rate : ℝ := 0.18
def tax_amount : ℝ := annual_income * tax_rate
def money_left : ℝ := annual_income - tax_amount

theorem dorothy_money_left : money_left = 49200 := 
by
  sorry

end dorothy_money_left_l1918_191888


namespace farmer_has_42_cows_left_l1918_191802

-- Define the conditions
def initial_cows := 51
def added_cows := 5
def sold_fraction := 1 / 4

-- Lean statement to prove the number of cows left
theorem farmer_has_42_cows_left :
  (initial_cows + added_cows) - (sold_fraction * (initial_cows + added_cows)) = 42 :=
by
  -- skipping the proof part
  sorry

end farmer_has_42_cows_left_l1918_191802


namespace simplify_expr1_simplify_expr2_l1918_191861

variable (x y : ℝ)

theorem simplify_expr1 : 
  3 * x^2 - 2 * x * y + y^2 - 3 * x^2 + 3 * x * y = x * y + y^2 :=
by
  sorry

theorem simplify_expr2 : 
  (7 * x^2 - 3 * x * y) - 6 * (x^2 - 1/3 * x * y) = x^2 - x * y :=
by
  sorry

end simplify_expr1_simplify_expr2_l1918_191861


namespace y_coordinate_of_third_vertex_eq_l1918_191875

theorem y_coordinate_of_third_vertex_eq (x1 x2 y1 y2 : ℝ)
    (h1 : x1 = 0) 
    (h2 : y1 = 3) 
    (h3 : x2 = 10) 
    (h4 : y2 = 3) 
    (h5 : x1 ≠ x2) 
    (h6 : y1 = y2) 
    : ∃ y3 : ℝ, y3 = 3 + 5 * Real.sqrt 3 := 
by
  sorry

end y_coordinate_of_third_vertex_eq_l1918_191875


namespace sum_of_arithmetic_sequence_l1918_191884

theorem sum_of_arithmetic_sequence (S : ℕ → ℝ) (a₁ d : ℝ) 
  (h1 : ∀ n, S n = n * a₁ + (n - 1) * n / 2 * d)
  (h2 : S 1 / S 4 = 1 / 10) :
  S 3 / S 5 = 2 / 5 := 
sorry

end sum_of_arithmetic_sequence_l1918_191884


namespace combined_weight_of_jake_and_sister_l1918_191848

theorem combined_weight_of_jake_and_sister
  (J : ℕ) (S : ℕ)
  (h₁ : J = 113)
  (h₂ : J - 33 = 2 * S)
  : J + S = 153 :=
sorry

end combined_weight_of_jake_and_sister_l1918_191848


namespace maisy_new_job_hours_l1918_191867

-- Define the conditions
def current_job_earnings : ℚ := 80
def new_job_wage_per_hour : ℚ := 15
def new_job_bonus : ℚ := 35
def earnings_difference : ℚ := 15

-- Define the problem
theorem maisy_new_job_hours (h : ℚ) 
  (h1 : current_job_earnings = 80) 
  (h2 : new_job_wage_per_hour * h + new_job_bonus = current_job_earnings + earnings_difference) :
  h = 4 :=
  sorry

end maisy_new_job_hours_l1918_191867


namespace find_sum_invested_l1918_191860

theorem find_sum_invested (P : ℝ)
  (h1 : P * 18 / 100 * 2 - P * 12 / 100 * 2 = 504) :
  P = 4200 := 
sorry

end find_sum_invested_l1918_191860


namespace power_function_m_l1918_191877

theorem power_function_m (m : ℝ) 
  (h_even : ∀ x : ℝ, x^m = (-x)^m) 
  (h_decreasing : ∀ x y : ℝ, 0 < x → x < y → x^m > y^m) : m = -2 :=
sorry

end power_function_m_l1918_191877


namespace sum_g_eq_half_l1918_191866

noncomputable def g (n : ℕ) : ℝ := ∑' k, if h : k ≥ 3 then (1 / (k : ℝ) ^ n) else 0

theorem sum_g_eq_half : (∑' n, if h : n ≥ 3 then g n else 0) = 1 / 2 := by
  sorry

end sum_g_eq_half_l1918_191866


namespace area_ratio_greater_than_two_ninths_l1918_191873

variable {α : Type*} [LinearOrder α] [LinearOrderedField α]

def area_triangle (A B C : α) : α := sorry -- Placeholder for the area function
noncomputable def triangle_division (A B C P Q R : α) : Prop :=
  -- Placeholder for division condition
  -- Here you would check that P, Q, and R divide the perimeter of triangle ABC into three equal parts
  sorry

theorem area_ratio_greater_than_two_ninths (A B C P Q R : α) :
  triangle_division A B C P Q R → area_triangle P Q R > (2 / 9) * area_triangle A B C :=
by
  sorry -- The proof goes here

end area_ratio_greater_than_two_ninths_l1918_191873


namespace ages_of_boys_l1918_191857

theorem ages_of_boys (a b c : ℕ) (h : a + b + c = 29) (h₁ : a = b) (h₂ : c = 11) : a = 9 ∧ b = 9 := 
by
  sorry

end ages_of_boys_l1918_191857


namespace first_machine_rate_l1918_191881

theorem first_machine_rate (x : ℕ) (h1 : 30 * x + 30 * 65 = 3000) : x = 35 := sorry

end first_machine_rate_l1918_191881


namespace ratio_hours_per_day_l1918_191855

theorem ratio_hours_per_day 
  (h₁ : ∀ h : ℕ, h * 30 = 1200 + (h - 40) * 45 → 40 ≤ h ∧ 6 * 3 ≤ 40)
  (h₂ : 6 * 3 + (x - 6 * 3) / 2 = 24)
  (h₃ : x = 1290) :
  (24 / 2) / 6 = 2 := 
by
  sorry

end ratio_hours_per_day_l1918_191855


namespace value_of_mn_l1918_191806

theorem value_of_mn (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n) (h_eq : m^4 - n^4 = 3439) : m * n = 90 := 
by sorry

end value_of_mn_l1918_191806


namespace multiple_of_24_l1918_191849

theorem multiple_of_24 (n : ℕ) (h : n > 0) : 
  ∃ k₁ k₂ : ℕ, (6 * n - 1)^2 - 1 = 24 * k₁ ∧ (6 * n + 1)^2 - 1 = 24 * k₂ :=
by
  sorry

end multiple_of_24_l1918_191849


namespace total_points_scored_l1918_191836

theorem total_points_scored (m1 m2 m3 m4 m5 m6 j1 j2 j3 j4 j5 j6 : ℕ) :
  m1 = 5 → j1 = m1 + 2 →
  m2 = 7 → j2 = m2 - 3 →
  m3 = 10 → j3 = m3 / 2 →
  m4 = 12 → j4 = m4 * 2 →
  m5 = 6 → j5 = m5 →
  j6 = 8 → m6 = j6 + 4 →
  m1 + m2 + m3 + m4 + m5 + m6 + j1 + j2 + j3 + j4 + j5 + j6 = 106 :=
by
  intros
  sorry

end total_points_scored_l1918_191836


namespace negation_of_at_most_four_l1918_191822

theorem negation_of_at_most_four (n : ℕ) : ¬(n ≤ 4) → n ≥ 5 := 
by
  sorry

end negation_of_at_most_four_l1918_191822


namespace solution_set_of_inequality_l1918_191871

theorem solution_set_of_inequality:
  {x : ℝ | 1 < abs (2 * x - 1) ∧ abs (2 * x - 1) < 3} = 
  {x : ℝ | -1 < x ∧ x < 0} ∪ 
  {x : ℝ | 1 < x ∧ x < 2} :=
by
  sorry

end solution_set_of_inequality_l1918_191871


namespace integer_fraction_condition_l1918_191830

theorem integer_fraction_condition (p : ℕ) (h_pos : 0 < p) :
  (∃ k : ℤ, k > 0 ∧ (5 * p + 15) = k * (3 * p - 9)) ↔ (4 ≤ p ∧ p ≤ 19) :=
by
  sorry

end integer_fraction_condition_l1918_191830


namespace flash_catches_ace_l1918_191835

theorem flash_catches_ace (v : ℝ) (x : ℝ) (y : ℝ) (hx : x > 1) :
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  flash_distance = (xy / (x - 1)) :=
by
  let t := y / (v * (x - 1))
  let ace_distance := v * t
  let flash_distance := x * v * t
  have h1 : x * v * t = xy / (x - 1) := sorry
  exact h1

end flash_catches_ace_l1918_191835


namespace ch_sub_ch_add_sh_sub_sh_add_l1918_191805

noncomputable def sh (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2
noncomputable def ch (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) / 2

theorem ch_sub (x y : ℝ) : ch (x - y) = ch x * ch y - sh x * sh y := sorry
theorem ch_add (x y : ℝ) : ch (x + y) = ch x * ch y + sh x * sh y := sorry
theorem sh_sub (x y : ℝ) : sh (x - y) = sh x * ch y - ch x * sh y := sorry
theorem sh_add (x y : ℝ) : sh (x + y) = sh x * ch y + ch x * sh y := sorry

end ch_sub_ch_add_sh_sub_sh_add_l1918_191805


namespace time_after_2051_hours_l1918_191859

theorem time_after_2051_hours (h₀ : 9 ≤ 11): 
  (9 + 2051 % 12) % 12 = 8 :=
by {
  -- proving the statement here
  sorry
}

end time_after_2051_hours_l1918_191859


namespace largest_of_five_numbers_l1918_191827

theorem largest_of_five_numbers : ∀ (a b c d e : ℝ), 
  a = 0.938 → b = 0.9389 → c = 0.93809 → d = 0.839 → e = 0.893 → b = max a (max b (max c (max d e))) :=
by
  intros a b c d e ha hb hc hd he
  rw [ha, hb, hc, hd, he]
  sorry

end largest_of_five_numbers_l1918_191827


namespace fraction_mango_sold_l1918_191899

theorem fraction_mango_sold :
  ∀ (choco_total mango_total choco_sold unsold: ℕ) (x : ℚ),
    choco_total = 50 →
    mango_total = 54 →
    choco_sold = (3 * 50) / 5 →
    unsold = 38 →
    (choco_total + mango_total) - (choco_sold + x * mango_total) = unsold →
    x = 4 / 27 :=
by
  intros choco_total mango_total choco_sold unsold x
  sorry

end fraction_mango_sold_l1918_191899


namespace exists_n_divisible_by_5_l1918_191868

open Int

theorem exists_n_divisible_by_5 
  (a b c d m : ℤ) 
  (h1 : 5 ∣ (a * m^3 + b * m^2 + c * m + d)) 
  (h2 : ¬ (5 ∣ d)) :
  ∃ n : ℤ, 5 ∣ (d * n^3 + c * n^2 + b * n + a) :=
by
  sorry

end exists_n_divisible_by_5_l1918_191868


namespace triangle_pentagon_side_ratio_l1918_191810

theorem triangle_pentagon_side_ratio (triangle_perimeter : ℕ) (pentagon_perimeter : ℕ) 
  (h1 : triangle_perimeter = 60) (h2 : pentagon_perimeter = 60) :
  (triangle_perimeter / 3 : ℚ) / (pentagon_perimeter / 5 : ℚ) = 5 / 3 :=
by {
  sorry
}

end triangle_pentagon_side_ratio_l1918_191810


namespace red_marbles_in_A_l1918_191839

-- Define the number of marbles in baskets A, B, and C
variables (R : ℕ)
def basketA := R + 2 -- Basket A: R red, 2 yellow
def basketB := 6 + 1 -- Basket B: 6 green, 1 yellow
def basketC := 3 + 9 -- Basket C: 3 white, 9 yellow

-- Define the greatest difference condition
def greatest_difference (A B C : ℕ) := max (max (A - B) (B - C)) (max (A - C) (C - B))

-- Define the hypothesis based on the conditions
axiom H1 : greatest_difference 3 9 0 = 6

-- The theorem we need to prove: The number of red marbles in Basket A is 8
theorem red_marbles_in_A : R = 8 := 
by {
  -- The proof would go here, but we'll use sorry to skip it
  sorry
}

end red_marbles_in_A_l1918_191839


namespace simplify_radical_subtraction_l1918_191804

theorem simplify_radical_subtraction : 
  (Real.sqrt 18 - Real.sqrt 8) = Real.sqrt 2 := 
by
  sorry

end simplify_radical_subtraction_l1918_191804


namespace Neil_candy_collected_l1918_191862

variable (M H N : ℕ)

-- Conditions
def Maggie_collected := M = 50
def Harper_collected := H = M + (30 * M) / 100
def Neil_collected := N = H + (40 * H) / 100

-- Theorem statement 
theorem Neil_candy_collected
  (hM : Maggie_collected M)
  (hH : Harper_collected M H)
  (hN : Neil_collected H N) :
  N = 91 := by
  sorry

end Neil_candy_collected_l1918_191862


namespace problem_2014_minus_4102_l1918_191892

theorem problem_2014_minus_4102 : 2014 - 4102 = -2088 := 
by
  -- The proof is omitted as per the requirement
  sorry

end problem_2014_minus_4102_l1918_191892


namespace cos_arccos_minus_arctan_eq_l1918_191842

noncomputable def cos_arccos_minus_arctan: Real :=
  Real.cos (Real.arccos (4 / 5) - Real.arctan (1 / 2))

theorem cos_arccos_minus_arctan_eq : cos_arccos_minus_arctan = (11 * Real.sqrt 5) / 25 := by
  sorry

end cos_arccos_minus_arctan_eq_l1918_191842


namespace identical_digits_has_37_factor_l1918_191847

theorem identical_digits_has_37_factor (a : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) : 37 ∣ (100 * a + 10 * a + a) :=
by
  sorry

end identical_digits_has_37_factor_l1918_191847


namespace determine_defective_coin_l1918_191801

-- Define the properties of the coins
structure Coin :=
(denomination : ℕ)
(weight : ℕ)

-- Given coins
def c1 : Coin := ⟨1, 1⟩
def c2 : Coin := ⟨2, 2⟩
def c3 : Coin := ⟨3, 3⟩
def c5 : Coin := ⟨5, 5⟩

-- Assume one coin is defective
variable (defective : Coin)
variable (differing_weight : ℕ)
#check differing_weight

theorem determine_defective_coin :
  (∃ (defective : Coin), ∀ (c : Coin), 
    c ≠ defective → c.weight = c.denomination) → 
  ((c2.weight + c3.weight = c5.weight → defective = c1) ∧
   (c1.weight + c2.weight = c3.weight → defective = c5) ∧
   (c2.weight ≠ 2 → defective = c2) ∧
   (c3.weight ≠ 3 → defective = c3)) :=
by
  sorry

end determine_defective_coin_l1918_191801


namespace maximum_value_of_f_l1918_191890

noncomputable def f (x : ℝ) : ℝ := (2 - x) * Real.exp x

theorem maximum_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ f x) ∧ f x = Real.exp 1 :=
sorry

end maximum_value_of_f_l1918_191890
