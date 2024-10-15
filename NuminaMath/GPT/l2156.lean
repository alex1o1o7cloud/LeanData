import Mathlib

namespace NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l2156_215674

theorem sum_of_reciprocals_of_squares (a b : ℕ) (h : a * b = 3) : (1 : ℚ)/a^2 + (1 : ℚ)/b^2 = 10/9 := 
sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_squares_l2156_215674


namespace NUMINAMATH_GPT_tent_ratio_l2156_215654

-- Define variables for tents in different parts of the camp
variables (N E C S T : ℕ)

-- Given conditions
def northernmost (N : ℕ) := N = 100
def center (C N : ℕ) := C = 4 * N
def southern (S : ℕ) := S = 200
def total (T N C E S : ℕ) := T = N + C + E + S

-- Main theorem statement for the proof
theorem tent_ratio (N E C S T : ℕ) 
  (hn : northernmost N)
  (hc : center C N) 
  (hs : southern S)
  (ht : total T N C E S) :
  E / N = 2 :=
by sorry

end NUMINAMATH_GPT_tent_ratio_l2156_215654


namespace NUMINAMATH_GPT_selling_price_l2156_215622

def cost_price : ℝ := 76.92
def profit_rate : ℝ := 0.30

theorem selling_price : cost_price * (1 + profit_rate) = 100.00 := by
  sorry

end NUMINAMATH_GPT_selling_price_l2156_215622


namespace NUMINAMATH_GPT_perpendicular_line_through_point_l2156_215628

open Real

theorem perpendicular_line_through_point (B : ℝ × ℝ) (x y : ℝ) (c : ℝ)
  (hB : B = (3, 0)) (h_perpendicular : 2 * x + y - 5 = 0) :
  x - 2 * y + 3 = 0 :=
sorry

end NUMINAMATH_GPT_perpendicular_line_through_point_l2156_215628


namespace NUMINAMATH_GPT_find_interest_rate_l2156_215691

theorem find_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 100)
  (hA : A = 121.00000000000001)
  (hn : n = 2)
  (ht : t = 1)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.2 :=
by
  sorry

end NUMINAMATH_GPT_find_interest_rate_l2156_215691


namespace NUMINAMATH_GPT_last_three_digits_7_pow_105_l2156_215602

theorem last_three_digits_7_pow_105 : (7^105) % 1000 = 783 :=
  sorry

end NUMINAMATH_GPT_last_three_digits_7_pow_105_l2156_215602


namespace NUMINAMATH_GPT_maximum_area_of_sector_l2156_215695

theorem maximum_area_of_sector (r l : ℝ) (h₁ : 2 * r + l = 10) : 
  (1 / 2 * l * r) ≤ 25 / 4 := 
sorry

end NUMINAMATH_GPT_maximum_area_of_sector_l2156_215695


namespace NUMINAMATH_GPT_ages_of_Linda_and_Jane_l2156_215673

theorem ages_of_Linda_and_Jane : 
  ∃ (J L : ℕ), 
    (L = 2 * J + 3) ∧ 
    (∃ (p : ℕ), Nat.Prime p ∧ p = L - J) ∧ 
    (L + J = 4 * J - 5) ∧ 
    (L = 19 ∧ J = 8) :=
by
  sorry

end NUMINAMATH_GPT_ages_of_Linda_and_Jane_l2156_215673


namespace NUMINAMATH_GPT_customers_in_each_car_l2156_215630

-- Conditions given in the problem
def cars : ℕ := 10
def purchases_sports : ℕ := 20
def purchases_music : ℕ := 30

-- Total purchases are equal to the total number of customers
def total_purchases : ℕ := purchases_sports + purchases_music
def total_customers (C : ℕ) : ℕ := cars * C

-- Lean statement to prove that the number of customers in each car is 5
theorem customers_in_each_car : (∃ C : ℕ, total_customers C = total_purchases) ∧ (∀ C : ℕ, total_customers C = total_purchases → C = 5) :=
by
  sorry

end NUMINAMATH_GPT_customers_in_each_car_l2156_215630


namespace NUMINAMATH_GPT_range_of_k_l2156_215620

theorem range_of_k (k : ℤ) (x : ℤ) 
  (h1 : -4 * x - k ≤ 0) 
  (h2 : x = -1 ∨ x = -2) : 
  8 ≤ k ∧ k < 12 :=
sorry

end NUMINAMATH_GPT_range_of_k_l2156_215620


namespace NUMINAMATH_GPT_combined_length_of_trains_is_correct_l2156_215665

noncomputable def combined_length_of_trains : ℕ :=
  let speed_A := 120 * 1000 / 3600 -- speed of train A in m/s
  let speed_B := 100 * 1000 / 3600 -- speed of train B in m/s
  let speed_motorbike := 64 * 1000 / 3600 -- speed of motorbike in m/s
  let relative_speed_A := (120 - 64) * 1000 / 3600 -- relative speed of train A with respect to motorbike in m/s
  let relative_speed_B := (100 - 64) * 1000 / 3600 -- relative speed of train B with respect to motorbike in m/s
  let length_A := relative_speed_A * 75 -- length of train A in meters
  let length_B := relative_speed_B * 90 -- length of train B in meters
  length_A + length_B

theorem combined_length_of_trains_is_correct :
  combined_length_of_trains = 2067 :=
  by
  sorry

end NUMINAMATH_GPT_combined_length_of_trains_is_correct_l2156_215665


namespace NUMINAMATH_GPT_proof_problem_l2156_215649

open Classical

variable (x y z : ℝ)

theorem proof_problem
  (cond1 : 0 < x ∧ x < 1)
  (cond2 : 0 < y ∧ y < 1)
  (cond3 : 0 < z ∧ z < 1)
  (cond4 : x * y * z = (1 - x) * (1 - y) * (1 - z)) :
  ((1 - x) * y ≥ 1/4) ∨ ((1 - y) * z ≥ 1/4) ∨ ((1 - z) * x ≥ 1/4) := by
  sorry

end NUMINAMATH_GPT_proof_problem_l2156_215649


namespace NUMINAMATH_GPT_stella_doll_price_l2156_215643

theorem stella_doll_price 
  (dolls_count clocks_count glasses_count : ℕ)
  (price_per_clock price_per_glass cost profit : ℕ)
  (D : ℕ)
  (h1 : dolls_count = 3)
  (h2 : clocks_count = 2)
  (h3 : glasses_count = 5)
  (h4 : price_per_clock = 15)
  (h5 : price_per_glass = 4)
  (h6 : cost = 40)
  (h7 : profit = 25)
  (h8 : 3 * D + 2 * price_per_clock + 5 * price_per_glass = cost + profit) :
  D = 5 :=
by
  sorry

end NUMINAMATH_GPT_stella_doll_price_l2156_215643


namespace NUMINAMATH_GPT_pizzeria_provolone_shred_l2156_215652

theorem pizzeria_provolone_shred 
    (cost_blend : ℝ) 
    (cost_mozzarella : ℝ) 
    (cost_romano : ℝ) 
    (cost_provolone : ℝ) 
    (prop_mozzarella : ℝ) 
    (prop_romano : ℝ) 
    (prop_provolone : ℝ) 
    (shredded_mozzarella : ℕ) 
    (shredded_romano : ℕ) 
    (shredded_provolone_needed : ℕ) :
  cost_blend = 696.05 ∧ 
  cost_mozzarella = 504.35 ∧ 
  cost_romano = 887.75 ∧ 
  cost_provolone = 735.25 ∧ 
  prop_mozzarella = 2 ∧ 
  prop_romano = 1 ∧ 
  prop_provolone = 2 ∧ 
  shredded_mozzarella = 20 ∧ 
  shredded_romano = 10 → 
  shredded_provolone_needed = 20 :=
by {
  sorry -- proof to be provided
}

end NUMINAMATH_GPT_pizzeria_provolone_shred_l2156_215652


namespace NUMINAMATH_GPT_problem1_problem2_l2156_215677

variable {m x : ℝ}

-- Definition of the function f
def f (x m : ℝ) : ℝ := |x - m| + |x|

-- Statement for Problem (1)
theorem problem1 (h : f 1 m = 1) : 
  ∀ x, f x 1 < 2 ↔ (-1 / 2) < x ∧ x < (3 / 2) := 
sorry

-- Statement for Problem (2)
theorem problem2 (h : ∀ x, f x m ≥ m^2) : 
  -1 ≤ m ∧ m ≤ 1 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l2156_215677


namespace NUMINAMATH_GPT_age_of_youngest_child_l2156_215624

theorem age_of_youngest_child (x : ℕ) (h : x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 70) : x = 8 :=
by
  sorry

end NUMINAMATH_GPT_age_of_youngest_child_l2156_215624


namespace NUMINAMATH_GPT_farmer_land_acres_l2156_215633

theorem farmer_land_acres
  (initial_ratio_corn : Nat)
  (initial_ratio_sugar_cane : Nat)
  (initial_ratio_tobacco : Nat)
  (new_ratio_corn : Nat)
  (new_ratio_sugar_cane : Nat)
  (new_ratio_tobacco : Nat)
  (additional_tobacco_acres : Nat)
  (total_land_acres : Nat) :
  initial_ratio_corn = 5 →
  initial_ratio_sugar_cane = 2 →
  initial_ratio_tobacco = 2 →
  new_ratio_corn = 2 →
  new_ratio_sugar_cane = 2 →
  new_ratio_tobacco = 5 →
  additional_tobacco_acres = 450 →
  total_land_acres = 1350 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_farmer_land_acres_l2156_215633


namespace NUMINAMATH_GPT_range_of_a_l2156_215645

def satisfies_p (x : ℝ) : Prop := (2 * x - 1) / (x - 1) ≤ 0

def satisfies_q (x a : ℝ) : Prop := x^2 - (2 * a + 1) * x + a * (a + 1) < 0

def sufficient_but_not_necessary (p q : ℝ → Prop) : Prop :=
  ∀ x, p x → q x ∧ ∃ x, q x ∧ ¬(p x)

theorem range_of_a :
  (∀ (x a : ℝ), satisfies_p x → satisfies_q x a → 0 ≤ a ∧ a < 1 / 2) ↔ (∀ a, 0 ≤ a ∧ a < 1 / 2) := by sorry

end NUMINAMATH_GPT_range_of_a_l2156_215645


namespace NUMINAMATH_GPT_not_prime_5n_plus_3_l2156_215663

theorem not_prime_5n_plus_3 (n a b : ℕ) (hn_pos : n > 0) (ha_pos : a > 0) (hb_pos : b > 0)
  (ha : 2 * n + 1 = a^2) (hb : 3 * n + 1 = b^2) : ¬Prime (5 * n + 3) :=
by
  sorry

end NUMINAMATH_GPT_not_prime_5n_plus_3_l2156_215663


namespace NUMINAMATH_GPT_initial_number_of_girls_l2156_215681

theorem initial_number_of_girls (n A : ℕ) (new_girl_weight : ℕ := 80) (original_girl_weight : ℕ := 40)
  (avg_increase : ℕ := 2)
  (condition : n * (A + avg_increase) - n * A = 40) :
  n = 20 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_girls_l2156_215681


namespace NUMINAMATH_GPT_slices_per_pizza_l2156_215660

theorem slices_per_pizza (total_slices number_of_pizzas slices_per_pizza : ℕ) 
  (h_total_slices : total_slices = 168) 
  (h_number_of_pizzas : number_of_pizzas = 21) 
  (h_division : total_slices / number_of_pizzas = slices_per_pizza) : 
  slices_per_pizza = 8 :=
sorry

end NUMINAMATH_GPT_slices_per_pizza_l2156_215660


namespace NUMINAMATH_GPT_minimum_number_of_guests_l2156_215696

theorem minimum_number_of_guests (total_food : ℝ) (max_food_per_guest : ℝ) (H₁ : total_food = 406) (H₂ : max_food_per_guest = 2.5) : 
  ∃ n : ℕ, (n : ℝ) ≥ 163 ∧ total_food / max_food_per_guest ≤ (n : ℝ) := 
by
  sorry

end NUMINAMATH_GPT_minimum_number_of_guests_l2156_215696


namespace NUMINAMATH_GPT_root_interval_k_l2156_215623

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem root_interval_k (k : ℤ) (h_cont : Continuous f) (h_mono : Monotone f)
  (h1 : f 2 < 0) (h2 : f 3 > 0) : k = 4 :=
by
  -- The proof part is omitted as per instruction.
  sorry

end NUMINAMATH_GPT_root_interval_k_l2156_215623


namespace NUMINAMATH_GPT_winning_percentage_l2156_215690

theorem winning_percentage (total_votes winner_votes : ℕ) 
  (h1 : winner_votes = 1344) 
  (h2 : winner_votes - 288 = total_votes - winner_votes) : 
  (winner_votes * 100 / total_votes = 56) :=
sorry

end NUMINAMATH_GPT_winning_percentage_l2156_215690


namespace NUMINAMATH_GPT_chemical_reaction_proof_l2156_215605

-- Define the given number of moles for each reactant
def moles_NaOH : ℕ := 4
def moles_NH4Cl : ℕ := 3

-- Define the balanced chemical equation stoichiometry
def stoichiometry_ratio_NaOH_NH4Cl : ℕ := 1

-- Define the product formation based on the limiting reactant
theorem chemical_reaction_proof
  (moles_NaOH : ℕ)
  (moles_NH4Cl : ℕ)
  (stoichiometry_ratio_NaOH_NH4Cl : ℕ)
  (h1 : moles_NaOH = 4)
  (h2 : moles_NH4Cl = 3)
  (h3 : stoichiometry_ratio_NaOH_NH4Cl = 1):
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = 3 * 1) ∧
  (3 = moles_NH4Cl) ∧
  (1 = moles_NaOH - moles_NH4Cl) :=
by {
  -- Provide assumptions based on the problem
  sorry
}

end NUMINAMATH_GPT_chemical_reaction_proof_l2156_215605


namespace NUMINAMATH_GPT_principal_amount_l2156_215640

theorem principal_amount (r : ℝ) (n : ℕ) (t : ℕ) (A : ℝ) :
    r = 0.12 → n = 2 → t = 20 →
    ∃ P : ℝ, A = P * (1 + r / n)^(n * t) :=
by
  intros hr hn ht
  have P := A / (1 + r / n)^(n * t)
  use P
  sorry

end NUMINAMATH_GPT_principal_amount_l2156_215640


namespace NUMINAMATH_GPT_sequence_expression_l2156_215651

noncomputable def seq (n : ℕ) : ℝ := 
  match n with
  | 0 => 1  -- note: indexing from 1 means a_1 corresponds to seq 0 in Lean
  | m+1 => seq m / (3 * seq m + 1)

theorem sequence_expression (n : ℕ) : 
  ∀ n, seq (n + 1) = 1 / (3 * (n + 1) - 2) := 
sorry

end NUMINAMATH_GPT_sequence_expression_l2156_215651


namespace NUMINAMATH_GPT_wrapping_cube_wrapping_prism_a_wrapping_prism_b_l2156_215610

theorem wrapping_cube (ways_cube : ℕ) :
  ways_cube = 3 :=
  sorry

theorem wrapping_prism_a (ways_prism_a : ℕ) (a : ℝ) :
  (ways_prism_a = 5) ↔ (a > 0) :=
  sorry

theorem wrapping_prism_b (ways_prism_b : ℕ) (b : ℝ) :
  (ways_prism_b = 7) ↔ (b > 0) :=
  sorry

end NUMINAMATH_GPT_wrapping_cube_wrapping_prism_a_wrapping_prism_b_l2156_215610


namespace NUMINAMATH_GPT_problem_solution_l2156_215617

section
variables (a b : ℝ)

-- Definition of the \* operation
def star_op (a b : ℝ) : ℝ := (a + 1) * (b - 1)

-- Definition of a^{*2} as a \* a
def star_square (a : ℝ) : ℝ := star_op a a

-- Define the specific problem instance with x = 2
def problem_expr : ℝ := star_op 3 (star_square 2) - star_op 2 2 + 1

-- Theorem stating the correct answer
theorem problem_solution : problem_expr = 6 := by
  -- Proof steps, marked as 'sorry'
  sorry

end

end NUMINAMATH_GPT_problem_solution_l2156_215617


namespace NUMINAMATH_GPT_cubic_no_negative_roots_l2156_215637

noncomputable def cubic_eq (x : ℝ) : ℝ := x^3 - 9 * x^2 + 23 * x - 15

theorem cubic_no_negative_roots {x : ℝ} : cubic_eq x = 0 → 0 ≤ x := sorry

end NUMINAMATH_GPT_cubic_no_negative_roots_l2156_215637


namespace NUMINAMATH_GPT_ratio_length_breadth_l2156_215675

noncomputable def b : ℝ := 18
noncomputable def l : ℝ := 972 / b

theorem ratio_length_breadth
  (A : ℝ)
  (h1 : b = 18)
  (h2 : l * b = 972) :
  (l / b) = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_length_breadth_l2156_215675


namespace NUMINAMATH_GPT_integer_solutions_of_system_l2156_215639

theorem integer_solutions_of_system :
  {x : ℤ | - 2 * x + 7 < 10 ∧ (7 * x + 1) / 5 - 1 ≤ x} = {-1, 0, 1, 2} :=
by
  sorry

end NUMINAMATH_GPT_integer_solutions_of_system_l2156_215639


namespace NUMINAMATH_GPT_biking_days_in_week_l2156_215611

def onurDistancePerDay : ℕ := 250
def hanilDistanceMorePerDay : ℕ := 40
def weeklyDistance : ℕ := 2700

theorem biking_days_in_week : (weeklyDistance / (onurDistancePerDay + hanilDistanceMorePerDay + onurDistancePerDay)) = 5 :=
by
  sorry

end NUMINAMATH_GPT_biking_days_in_week_l2156_215611


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l2156_215676

theorem arithmetic_sequence_sum :
  ∀ (a_n : ℕ → ℤ) (S_n : ℕ → ℤ),
  (∀ n : ℕ, S_n n = (n * (2 * (a_n 1) + (n - 1) * (a_n 2 - a_n 1))) / 2) →
  S_n 17 = 170 →
  a_n 7 + a_n 8 + a_n 12 = 30 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l2156_215676


namespace NUMINAMATH_GPT_volleyball_shotput_cost_l2156_215658

theorem volleyball_shotput_cost (x y : ℝ) :
  (2*x + 3*y = 95) ∧ (5*x + 7*y = 230) :=
  sorry

end NUMINAMATH_GPT_volleyball_shotput_cost_l2156_215658


namespace NUMINAMATH_GPT__l2156_215655

open Nat

/-- Function to check the triangle inequality theorem -/
def canFormTriangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

example : canFormTriangle 6 4 5 := by
  /- Proof omitted -/
  sorry

end NUMINAMATH_GPT__l2156_215655


namespace NUMINAMATH_GPT_frequency_of_2_l2156_215698

def num_set := "20231222"
def total_digits := 8
def count_of_2 := 5

theorem frequency_of_2 : (count_of_2 : ℚ) / total_digits = 5 / 8 := by
  sorry

end NUMINAMATH_GPT_frequency_of_2_l2156_215698


namespace NUMINAMATH_GPT_D_300_l2156_215686

def D (n : ℕ) : ℕ :=
sorry

theorem D_300 : D 300 = 73 := 
by 
sorry

end NUMINAMATH_GPT_D_300_l2156_215686


namespace NUMINAMATH_GPT_integer_pairs_solution_l2156_215632

theorem integer_pairs_solution (x y : ℤ) (k : ℤ) :
  2 * x^2 - 6 * x * y + 3 * y^2 = -1 ↔
  ∃ n : ℤ, x = (2 + Real.sqrt 3)^k / 2 ∨ x = -(2 + Real.sqrt 3)^k / 2 ∧
           y = x + (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) ∨ 
           y = x - (2 + Real.sqrt 3)^k / (2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_integer_pairs_solution_l2156_215632


namespace NUMINAMATH_GPT_similar_triangle_perimeter_l2156_215650

noncomputable def is_similar_triangles (a b c a' b' c' : ℝ) := 
  ∃ (k : ℝ), k > 0 ∧ (a = k * a') ∧ (b = k * b') ∧ (c = k * c')

noncomputable def is_isosceles (a b c : ℝ) := (a = b) ∨ (a = c) ∨ (b = c)

theorem similar_triangle_perimeter :
  ∀ (a b c a' b' c' : ℝ),
    is_isosceles a b c → 
    is_similar_triangles a b c a' b' c' →
    c' = 42 →
    (a = 12) → 
    (b = 12) → 
    (c = 14) →
    (b' = 36) →
    (a' = 36) →
    a' + b' + c' = 114 :=
by
  intros
  sorry

end NUMINAMATH_GPT_similar_triangle_perimeter_l2156_215650


namespace NUMINAMATH_GPT_product_of_roots_l2156_215609

variable {x1 x2 : ℝ}

theorem product_of_roots (h : ∀ x, -x^2 + 3*x = 0 → (x = x1 ∨ x = x2)) :
  x1 * x2 = 0 :=
by
  sorry

end NUMINAMATH_GPT_product_of_roots_l2156_215609


namespace NUMINAMATH_GPT_positive_difference_l2156_215607

theorem positive_difference (x y : ℚ) (h1 : x + y = 40) (h2 : 3 * y - 4 * x = 20) : y - x = 80 / 7 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_l2156_215607


namespace NUMINAMATH_GPT_fourth_term_is_fifteen_l2156_215669

-- Define the problem parameters
variables (a d : ℕ)

-- Define the conditions
def sum_first_third_term : Prop := (a + (a + 2 * d) = 10)
def fourth_term_def : ℕ := a + 3 * d

-- Declare the theorem to be proved
theorem fourth_term_is_fifteen (h1 : sum_first_third_term a d) : fourth_term_def a d = 15 :=
sorry

end NUMINAMATH_GPT_fourth_term_is_fifteen_l2156_215669


namespace NUMINAMATH_GPT_gain_percentage_is_twenty_l2156_215672

theorem gain_percentage_is_twenty (SP CP Gain : ℝ) (h0 : SP = 90) (h1 : Gain = 15) (h2 : SP = CP + Gain) : (Gain / CP) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_gain_percentage_is_twenty_l2156_215672


namespace NUMINAMATH_GPT_angle_complement_supplement_l2156_215683

theorem angle_complement_supplement (θ : ℝ) (h1 : 90 - θ = (1/3) * (180 - θ)) : θ = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_complement_supplement_l2156_215683


namespace NUMINAMATH_GPT_speed_conversion_l2156_215697

theorem speed_conversion (speed_kmph : ℕ) (conversion_rate : ℚ) : (speed_kmph = 600) ∧ (conversion_rate = 0.6) → (speed_kmph * conversion_rate / 60 = 6) :=
by
  sorry

end NUMINAMATH_GPT_speed_conversion_l2156_215697


namespace NUMINAMATH_GPT_xyz_sum_l2156_215667

theorem xyz_sum (x y z : ℕ) (h1 : xyz = 240) (h2 : xy + z = 46) (h3 : x + yz = 64) : x + y + z = 20 :=
sorry

end NUMINAMATH_GPT_xyz_sum_l2156_215667


namespace NUMINAMATH_GPT_count_no_carry_pairs_l2156_215627

theorem count_no_carry_pairs : 
  ∃ n, n = 1125 ∧ ∀ (a b : ℕ), (2000 ≤ a ∧ a < 2999 ∧ b = a + 1) → 
  (∀ i, (0 ≤ i ∧ i < 4) → ((a / (10 ^ i) % 10 + b / (10 ^ i) % 10) < 10)) := sorry

end NUMINAMATH_GPT_count_no_carry_pairs_l2156_215627


namespace NUMINAMATH_GPT_solve_trig_equation_l2156_215635

open Real

theorem solve_trig_equation (x : ℝ) (n : ℤ) :
  (2 * tan (6 * x) ^ 4 + 4 * sin (4 * x) * sin (8 * x) - cos (8 * x) - cos (16 * x) + 2) / sqrt (cos x - sqrt 3 * sin x) = 0 
  ∧ cos x - sqrt 3 * sin x > 0 →
  ∃ (k : ℤ), x = 2 * π * k ∨ x = -π / 6 + 2 * π * k ∨ x = -π / 3 + 2 * π * k ∨ x = -π / 2 + 2 * π * k ∨ x = -2 * π / 3 + 2 * π * k :=
sorry

end NUMINAMATH_GPT_solve_trig_equation_l2156_215635


namespace NUMINAMATH_GPT_porch_width_l2156_215692

theorem porch_width (L_house W_house total_area porch_length W : ℝ)
  (h1 : L_house = 20.5) (h2 : W_house = 10) (h3 : total_area = 232) (h4 : porch_length = 6) (h5 : total_area = (L_house * W_house) + (porch_length * W)) :
  W = 4.5 :=
by 
  sorry

end NUMINAMATH_GPT_porch_width_l2156_215692


namespace NUMINAMATH_GPT_percentage_of_sum_l2156_215636

theorem percentage_of_sum (x y P : ℝ) (h1 : 0.50 * (x - y) = (P / 100) * (x + y)) (h2 : y = 0.25 * x) : P = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_sum_l2156_215636


namespace NUMINAMATH_GPT_longest_side_range_of_obtuse_triangle_l2156_215625

theorem longest_side_range_of_obtuse_triangle (a b c : ℝ) (h₁ : a = 1) (h₂ : b = 2) :
  a^2 + b^2 < c^2 → (Real.sqrt 5 < c ∧ c < 3) ∨ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_longest_side_range_of_obtuse_triangle_l2156_215625


namespace NUMINAMATH_GPT_molecular_weight_of_NH4Cl_l2156_215642

theorem molecular_weight_of_NH4Cl (weight_8_moles : ℕ) (weight_per_mole : ℕ) :
  weight_8_moles = 424 →
  weight_per_mole = 53 →
  weight_8_moles / 8 = weight_per_mole :=
by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_molecular_weight_of_NH4Cl_l2156_215642


namespace NUMINAMATH_GPT_mean_of_remaining_four_numbers_l2156_215689

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h: (a + b + c + d + 105) / 5 = 90) :
  (a + b + c + d) / 4 = 86.25 :=
by
  sorry

end NUMINAMATH_GPT_mean_of_remaining_four_numbers_l2156_215689


namespace NUMINAMATH_GPT_minimum_value_l2156_215668

open Real

-- Statement of the conditions
def conditions (a b c : ℝ) : Prop :=
  -0.5 < a ∧ a < 0.5 ∧ -0.5 < b ∧ b < 0.5 ∧ -0.5 < c ∧ c < 0.5

-- Expression to be minimized
noncomputable def expression (a b c : ℝ) : ℝ :=
  1 / ((1 - a) * (1 - b) * (1 - c)) + 1 / ((1 + a) * (1 + b) * (1 + c))

-- Minimum value to prove
theorem minimum_value (a b c : ℝ) (h : conditions a b c) : expression a b c ≥ 4.74 :=
sorry

end NUMINAMATH_GPT_minimum_value_l2156_215668


namespace NUMINAMATH_GPT_joanna_reading_rate_l2156_215664

variable (P : ℝ)

theorem joanna_reading_rate (h : 3 * P + 6.5 * P + 6 * P = 248) : P = 16 := by
  sorry

end NUMINAMATH_GPT_joanna_reading_rate_l2156_215664


namespace NUMINAMATH_GPT_total_vehicles_l2156_215648

theorem total_vehicles (morn_minivans afternoon_minivans evening_minivans night_minivans : Nat)
                       (morn_sedans afternoon_sedans evening_sedans night_sedans : Nat)
                       (morn_SUVs afternoon_SUVs evening_SUVs night_SUVs : Nat)
                       (morn_trucks afternoon_trucks evening_trucks night_trucks : Nat)
                       (morn_motorcycles afternoon_motorcycles evening_motorcycles night_motorcycles : Nat) :
                       morn_minivans = 20 → afternoon_minivans = 22 → evening_minivans = 15 → night_minivans = 10 →
                       morn_sedans = 17 → afternoon_sedans = 13 → evening_sedans = 19 → night_sedans = 12 →
                       morn_SUVs = 12 → afternoon_SUVs = 15 → evening_SUVs = 18 → night_SUVs = 20 →
                       morn_trucks = 8 → afternoon_trucks = 10 → evening_trucks = 14 → night_trucks = 20 →
                       morn_motorcycles = 5 → afternoon_motorcycles = 7 → evening_motorcycles = 10 → night_motorcycles = 15 →
                       morn_minivans + afternoon_minivans + evening_minivans + night_minivans +
                       morn_sedans + afternoon_sedans + evening_sedans + night_sedans +
                       morn_SUVs + afternoon_SUVs + evening_SUVs + night_SUVs +
                       morn_trucks + afternoon_trucks + evening_trucks + night_trucks +
                       morn_motorcycles + afternoon_motorcycles + evening_motorcycles + night_motorcycles = 282 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_vehicles_l2156_215648


namespace NUMINAMATH_GPT_people_sharing_bill_l2156_215647

theorem people_sharing_bill (total_bill : ℝ) (tip_percent : ℝ) (share_per_person : ℝ) (n : ℝ) :
  total_bill = 211.00 →
  tip_percent = 0.15 →
  share_per_person = 26.96 →
  abs (n - 9) < 1 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_people_sharing_bill_l2156_215647


namespace NUMINAMATH_GPT_hyperbola_condition_l2156_215688

theorem hyperbola_condition (m : ℝ) : ((m - 2) * (m + 3) < 0) ↔ (-3 < m ∧ m < 0) := by
  sorry

end NUMINAMATH_GPT_hyperbola_condition_l2156_215688


namespace NUMINAMATH_GPT_number_of_medium_boxes_l2156_215621

def large_box_tape := 4
def medium_box_tape := 2
def small_box_tape := 1
def label_tape := 1

def num_large_boxes := 2
def num_small_boxes := 5
def total_tape := 44

theorem number_of_medium_boxes :
  let tape_used_large_boxes := num_large_boxes * (large_box_tape + label_tape)
  let tape_used_small_boxes := num_small_boxes * (small_box_tape + label_tape)
  let tape_used_medium_boxes := total_tape - (tape_used_large_boxes + tape_used_small_boxes)
  let medium_box_total_tape := medium_box_tape + label_tape
  let num_medium_boxes := tape_used_medium_boxes / medium_box_total_tape
  num_medium_boxes = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_medium_boxes_l2156_215621


namespace NUMINAMATH_GPT_residue_7_1234_mod_13_l2156_215641

theorem residue_7_1234_mod_13 :
  (7 : ℤ) ^ 1234 % 13 = 12 :=
by
  -- given conditions as definitions
  have h1 : (7 : ℤ) % 13 = 7 := by norm_num
  
  -- auxiliary calculations
  have h2 : (49 : ℤ) % 13 = 10 := by norm_num
  have h3 : (100 : ℤ) % 13 = 9 := by norm_num
  have h4 : (81 : ℤ) % 13 = 3 := by norm_num
  have h5 : (729 : ℤ) % 13 = 1 := by norm_num

  -- the actual problem we want to prove
  sorry

end NUMINAMATH_GPT_residue_7_1234_mod_13_l2156_215641


namespace NUMINAMATH_GPT_find_positive_n_l2156_215679

def consecutive_product (k : ℕ) : ℕ := k * (k + 1) * (k + 2)

theorem find_positive_n (n k : ℕ) (hn : 0 < n) (hk : 0 < k) :
  n^6 + 5*n^3 + 4*n + 116 = consecutive_product k ↔ n = 3 := 
by 
  sorry

end NUMINAMATH_GPT_find_positive_n_l2156_215679


namespace NUMINAMATH_GPT_inequality_hold_l2156_215600

variables (x y : ℝ)

theorem inequality_hold (h : x^4 + y^4 ≥ 2) : |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := 
sorry

end NUMINAMATH_GPT_inequality_hold_l2156_215600


namespace NUMINAMATH_GPT_toms_weekly_revenue_l2156_215656

def crabs_per_bucket : Nat := 12
def number_of_buckets : Nat := 8
def price_per_crab : Nat := 5
def days_per_week : Nat := 7

theorem toms_weekly_revenue :
  (crabs_per_bucket * number_of_buckets * price_per_crab * days_per_week) = 3360 :=
by
  sorry

end NUMINAMATH_GPT_toms_weekly_revenue_l2156_215656


namespace NUMINAMATH_GPT_download_speeds_l2156_215666

theorem download_speeds (x : ℕ) (s4 : ℕ := 4) (s5 : ℕ := 60) :
  (600 / x - 600 / (15 * x) = 140) → (x = s4 ∧ 15 * x = s5) := by
  sorry

end NUMINAMATH_GPT_download_speeds_l2156_215666


namespace NUMINAMATH_GPT_fraction_of_pelicans_moved_l2156_215699

-- Conditions
variables (P : ℕ)
variables (n_Sharks : ℕ := 60) -- Number of sharks in Pelican Bay
variables (n_Pelicans_original : ℕ := 2 * P) -- Twice the original number of Pelicans in Shark Bite Cove
variables (n_Pelicans_remaining : ℕ := 20) -- Number of remaining Pelicans in Shark Bite Cove

-- Proof to show fraction that moved
theorem fraction_of_pelicans_moved (h : 2 * P = n_Sharks) : (P - n_Pelicans_remaining) / P = 1 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_fraction_of_pelicans_moved_l2156_215699


namespace NUMINAMATH_GPT_ermias_balls_more_is_5_l2156_215608

-- Define the conditions
def time_per_ball : ℕ := 20
def alexia_balls : ℕ := 20
def total_time : ℕ := 900

-- Define Ermias's balls
def ermias_balls_more (x : ℕ) : ℕ := alexia_balls + x

-- Alexia's total inflation time
def alexia_total_time : ℕ := alexia_balls * time_per_ball

-- Ermias's total inflation time given x more balls than Alexia
def ermias_total_time (x : ℕ) : ℕ := (ermias_balls_more x) * time_per_ball

-- Total time taken by both Alexia and Ermias
def combined_time (x : ℕ) : ℕ := alexia_total_time + ermias_total_time x

-- Proven that Ermias inflated 5 more balls than Alexia given the total time condition
theorem ermias_balls_more_is_5 : (∃ x : ℕ, combined_time x = total_time) := 
by {
  sorry
}

end NUMINAMATH_GPT_ermias_balls_more_is_5_l2156_215608


namespace NUMINAMATH_GPT_angle_equality_l2156_215604

variables {Point Circle : Type}
variables (K O1 O2 P1 P2 Q1 Q2 M1 M2 : Point)
variables (W1 W2 : Circle)
variables (midpoint : Point → Point → Point)
variables (is_center : Point → Circle → Prop)
variables (intersects_at : Circle → Circle → Point → Prop)
variables (common_tangent_points : Circle → Circle → (Point × Point) × (Point × Point) → Prop)
variables (intersect_circle_at : Circle → Line → Point → Point → Prop)
variables (angle : Point → Point → Point → ℝ) -- to denote the angle measure between three points

-- Conditions
axiom K_intersection : intersects_at W1 W2 K
axiom O1_center : is_center O1 W1
axiom O2_center : is_center O2 W2
axiom tangents_meet_at : common_tangent_points W1 W2 ((P1, Q1), (P2, Q2))
axiom M1_midpoint : M1 = midpoint P1 Q1
axiom M2_midpoint : M2 = midpoint P2 Q2

-- The statement to prove
theorem angle_equality : angle O1 K O2 = angle M1 K M2 := 
  sorry

end NUMINAMATH_GPT_angle_equality_l2156_215604


namespace NUMINAMATH_GPT_sum_of_areas_of_triangles_l2156_215613

noncomputable def triangle_sum_of_box (a b c : ℝ) :=
  let face_triangles_area := 4 * ((a * b + a * c + b * c) / 2)
  let perpendicular_triangles_area := 4 * ((a * c + b * c) / 2)
  let oblique_triangles_area := 8 * Real.sqrt ((a + b + c) / 2 * ((a + b + c) / 2 - a) * ((a + b + c) / 2 - b) * ((a + b + c) / 2 - c))
  face_triangles_area + perpendicular_triangles_area + oblique_triangles_area

theorem sum_of_areas_of_triangles :
  triangle_sum_of_box 2 3 4 = 168 + k * Real.sqrt p := sorry

end NUMINAMATH_GPT_sum_of_areas_of_triangles_l2156_215613


namespace NUMINAMATH_GPT_Alyssa_total_spent_l2156_215684

-- Declare the costs of grapes and cherries.
def costOfGrapes : ℝ := 12.08
def costOfCherries : ℝ := 9.85

-- Total amount spent by Alyssa.
def totalSpent : ℝ := 21.93

-- Statement to prove that the sum of the costs is equal to the total spent.
theorem Alyssa_total_spent (g : ℝ) (c : ℝ) (t : ℝ) 
  (hg : g = costOfGrapes) 
  (hc : c = costOfCherries) 
  (ht : t = totalSpent) :
  g + c = t := by
  sorry

end NUMINAMATH_GPT_Alyssa_total_spent_l2156_215684


namespace NUMINAMATH_GPT_discount_percentage_l2156_215682

theorem discount_percentage (wm_cost dryer_cost after_discount before_discount discount_amount : ℝ)
    (h0 : wm_cost = 100) 
    (h1 : dryer_cost = wm_cost - 30) 
    (h2 : after_discount = 153) 
    (h3 : before_discount = wm_cost + dryer_cost) 
    (h4 : discount_amount = before_discount - after_discount) 
    (h5 : (discount_amount / before_discount) * 100 = 10) : 
    True := sorry

end NUMINAMATH_GPT_discount_percentage_l2156_215682


namespace NUMINAMATH_GPT_sequence_6th_term_l2156_215603

theorem sequence_6th_term 
    (a₁ a₂ a₃ a₄ a₅ a₆ : ℚ)
    (h₁ : a₁ = 3)
    (h₅ : a₅ = 54)
    (h₂ : a₂ = (a₁ + a₃) / 3)
    (h₃ : a₃ = (a₂ + a₄) / 3)
    (h₄ : a₄ = (a₃ + a₅) / 3)
    (h₆ : a₅ = (a₄ + a₆) / 3) :
    a₆ = 1133 / 7 :=
by
  sorry

end NUMINAMATH_GPT_sequence_6th_term_l2156_215603


namespace NUMINAMATH_GPT_greatest_length_of_pieces_l2156_215629

/-- Alicia has three ropes with lengths of 28 inches, 42 inches, and 70 inches.
She wants to cut these ropes into equal length pieces for her art project, and she doesn't want any leftover pieces.
Prove that the greatest length of each piece she can cut is 7 inches. -/
theorem greatest_length_of_pieces (a b c : ℕ) (h1 : a = 28) (h2 : b = 42) (h3 : c = 70) :
  ∃ (d : ℕ), d > 0 ∧ d ∣ a ∧ d ∣ b ∧ d ∣ c ∧ ∀ e : ℕ, e > 0 ∧ e ∣ a ∧ e ∣ b ∧ e ∣ c → e ≤ d := sorry

end NUMINAMATH_GPT_greatest_length_of_pieces_l2156_215629


namespace NUMINAMATH_GPT_partial_fraction_sum_l2156_215687

theorem partial_fraction_sum :
  (∃ A B C D E : ℝ, 
    (∀ x : ℝ, x ≠ 0 → x ≠ -1 → x ≠ -2 → x ≠ -3 → x ≠ -5 → 
    (1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 5)) = 
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 5))) ∧
    (A + B + C + D + E = 1 / 30)) :=
sorry

end NUMINAMATH_GPT_partial_fraction_sum_l2156_215687


namespace NUMINAMATH_GPT_number_of_red_balls_l2156_215693

theorem number_of_red_balls (total_balls : ℕ) (probability : ℚ) (num_red_balls : ℕ) 
  (h1 : total_balls = 12) 
  (h2 : probability = 1 / 22) 
  (h3 : (num_red_balls * (num_red_balls - 1) : ℚ) / (total_balls * (total_balls - 1)) = probability) :
  num_red_balls = 3 := 
by
  sorry

end NUMINAMATH_GPT_number_of_red_balls_l2156_215693


namespace NUMINAMATH_GPT_remainder_of_sum_is_12_l2156_215644

theorem remainder_of_sum_is_12 (D k1 k2 : ℤ) (h1 : 242 = k1 * D + 4) (h2 : 698 = k2 * D + 8) : (242 + 698) % D = 12 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_sum_is_12_l2156_215644


namespace NUMINAMATH_GPT_sum_of_coefficients_l2156_215615

theorem sum_of_coefficients (a b c : ℝ) (w : ℂ) (h_roots : ∃ w : ℂ, (∃ i : ℂ, i^2 = -1) ∧ 
  (x + ax^2 + bx + c)^3 = (w + 3*im)* (w + 9*im)*(2*w - 4)) :
  a + b + c = -136 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l2156_215615


namespace NUMINAMATH_GPT_average_speed_jeffrey_l2156_215601
-- Import the necessary Lean library.

-- Initial conditions in the problem, restated as Lean definitions.
def distance_jog (d : ℝ) : Prop := d = 3
def speed_jog (s : ℝ) : Prop := s = 4
def distance_walk (d : ℝ) : Prop := d = 4
def speed_walk (s : ℝ) : Prop := s = 3

-- Target statement to prove using Lean.
theorem average_speed_jeffrey :
  ∀ (dj sj dw sw : ℝ), distance_jog dj → speed_jog sj → distance_walk dw → speed_walk sw →
    (dj + dw) / ((dj / sj) + (dw / sw)) = 3.36 := 
  by
    intros dj sj dw sw hj hs hw hw
    sorry

end NUMINAMATH_GPT_average_speed_jeffrey_l2156_215601


namespace NUMINAMATH_GPT_primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l2156_215685

-- Define Part (a)
theorem primitive_root_coprime_distinct_residues (m k : ℕ) (h: Nat.gcd m k = 1) :
  ∃ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∀ i j s t, (i ≠ s ∨ j ≠ t) → (a i * b j) % (m * k) ≠ (a s * b t) % (m * k) :=
sorry

-- Define Part (b)
theorem noncoprime_non_distinct_residues (m k : ℕ) (h: Nat.gcd m k > 1) :
  ∀ (a : Fin m → ℕ) (b : Fin k → ℕ),
    ∃ i j x t, (i ≠ x ∨ j ≠ t) ∧ (a i * b j) % (m * k) = (a x * b t) % (m * k) :=
sorry

end NUMINAMATH_GPT_primitive_root_coprime_distinct_residues_noncoprime_non_distinct_residues_l2156_215685


namespace NUMINAMATH_GPT_find_constants_monotonicity_l2156_215618

noncomputable def f (x a b : ℝ) := (x^2 + a * x) * Real.exp x + b

theorem find_constants (a b : ℝ) (h_tangent : (f 0 a b = 1) ∧ (deriv (f · a b) 0 = -2)) :
  a = -2 ∧ b = 1 := by
  sorry

theorem monotonicity (a b : ℝ) (h_constants : a = -2 ∧ b = 1) :
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) > 0 → x > Real.sqrt 2 ∨ x < -Real.sqrt 2)) ∧
  (∀ x : ℝ, (Real.exp x * (x^2 - 2) < 0 → -Real.sqrt 2 < x ∧ x < Real.sqrt 2)) := by
  sorry

end NUMINAMATH_GPT_find_constants_monotonicity_l2156_215618


namespace NUMINAMATH_GPT_man_rowing_upstream_speed_l2156_215671

theorem man_rowing_upstream_speed (V_down V_m V_up V_s : ℕ) 
  (h1 : V_down = 41)
  (h2 : V_m = 33)
  (h3 : V_down = V_m + V_s)
  (h4 : V_up = V_m - V_s) 
  : V_up = 25 := 
by
  sorry

end NUMINAMATH_GPT_man_rowing_upstream_speed_l2156_215671


namespace NUMINAMATH_GPT_candidate_percentage_l2156_215657

theorem candidate_percentage (P : ℝ) (l : ℝ) (V : ℝ) : 
  l = 5000.000000000007 ∧ 
  V = 25000.000000000007 ∧ 
  V - 2 * (P / 100) * V = l →
  P = 40 :=
by
  sorry

end NUMINAMATH_GPT_candidate_percentage_l2156_215657


namespace NUMINAMATH_GPT_fish_count_l2156_215634

variables
  (x g s r : ℕ)
  (h1 : x - g = (2 / 3 : ℚ) * x - 1)
  (h2 : x - r = (2 / 3 : ℚ) * x + 4)
  (h3 : x = g + s + r)

theorem fish_count :
  s - g = 2 :=
by
  sorry

end NUMINAMATH_GPT_fish_count_l2156_215634


namespace NUMINAMATH_GPT_greatest_divisible_by_13_l2156_215606

def is_distinct_nonzero_digits (A B C : ℕ) : Prop :=
  0 < A ∧ A < 10 ∧ 0 < B ∧ B < 10 ∧ 0 < C ∧ C < 10 ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C

def number (A B C : ℕ) : ℕ :=
  10000 * A + 1000 * B + 100 * C + 10 * B + A

theorem greatest_divisible_by_13 :
  ∃ (A B C : ℕ), is_distinct_nonzero_digits A B C ∧ number A B C % 13 = 0 ∧ number A B C = 96769 :=
sorry

end NUMINAMATH_GPT_greatest_divisible_by_13_l2156_215606


namespace NUMINAMATH_GPT_butterfly_probability_l2156_215646

-- Define the vertices of the cube
inductive Vertex
| A | B | C | D | E | F | G | H

open Vertex

-- Define the edges of the cube
def edges : Vertex → List Vertex
| A => [B, D, E]
| B => [A, C, F]
| C => [B, D, G]
| D => [A, C, H]
| E => [A, F, H]
| F => [B, E, G]
| G => [C, F, H]
| H => [D, E, G]

-- Define a function to simulate the butterfly's movement
noncomputable def move : Vertex → ℕ → List (Vertex × ℕ)
| v, 0 => [(v, 0)]
| v, n + 1 =>
  let nextMoves := edges v
  nextMoves.bind (λ v' => move v' n)

-- Define the probability calculation part
noncomputable def probability_of_visiting_all_vertices (n_moves : ℕ) : ℚ :=
  let total_paths := (3 ^ n_moves : ℕ)
  let valid_paths := 27 -- Based on given final solution step
  valid_paths / total_paths

-- Statement of the problem in Lean 4
theorem butterfly_probability :
  probability_of_visiting_all_vertices 11 = 27 / 177147 :=
by
  sorry

end NUMINAMATH_GPT_butterfly_probability_l2156_215646


namespace NUMINAMATH_GPT_exists_equilateral_triangle_l2156_215638

variables {d1 d2 d3 : AffineSubspace ℝ (EuclideanSpace ℝ (Fin 2))}

theorem exists_equilateral_triangle (hne1 : d1 ≠ d2) (hne2 : d2 ≠ d3) (hne3 : d1 ≠ d3) : 
  ∃ (A1 A2 A3 : EuclideanSpace ℝ (Fin 2)), 
  (A1 ∈ d1 ∧ A2 ∈ d2 ∧ A3 ∈ d3) ∧ 
  dist A1 A2 = dist A2 A3 ∧ dist A2 A3 = dist A3 A1 := 
sorry

end NUMINAMATH_GPT_exists_equilateral_triangle_l2156_215638


namespace NUMINAMATH_GPT_proof_of_expression_value_l2156_215662

theorem proof_of_expression_value (m n : ℝ) 
  (h1 : m^2 - 2019 * m = 1) 
  (h2 : n^2 - 2019 * n = 1) : 
  (m^2 - 2019 * m + 3) * (n^2 - 2019 * n + 4) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_proof_of_expression_value_l2156_215662


namespace NUMINAMATH_GPT_C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l2156_215694

noncomputable def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 4 + Real.sin α)

noncomputable def polar_C2 (ρ θ m : ℝ) : ℝ := ρ * (Real.cos θ - m * Real.sin θ) + 1

theorem C1_Cartesian_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, parametric_C1 α = (x, y)) ↔ (x - 2)^2 + (y - 4)^2 = 1 := sorry

theorem C2_Cartesian_equation :
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, polar_C2 ρ θ m = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)
  ↔ x - m * y + 1 = 0 := sorry

def closest_point_on_C1_to_x_axis : ℝ × ℝ := (2, 3)

theorem m_value_when_C2_passes_through_P :
  ∃ (m : ℝ), x - m * y + 1 = 0 ∧ x = 2 ∧ y = 3 ∧ m = 1 := sorry

end NUMINAMATH_GPT_C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l2156_215694


namespace NUMINAMATH_GPT_sum_div_mult_sub_result_l2156_215661

-- Define the problem with conditions and expected answer
theorem sum_div_mult_sub_result :
  3521 + 480 / 60 * 3 - 521 = 3024 :=
by 
  sorry

end NUMINAMATH_GPT_sum_div_mult_sub_result_l2156_215661


namespace NUMINAMATH_GPT_problem_eqn_l2156_215631

theorem problem_eqn (a b c : ℝ) :
  (∃ r₁ r₂ : ℝ, r₁^2 + 3 * r₁ - 1 = 0 ∧ r₂^2 + 3 * r₂ - 1 = 0) ∧
  (∀ x : ℝ, (x^2 + 3 * x - 1 = 0) → (x^4 + a * x^2 + b * x + c = 0)) →
  a + b + 4 * c = -7 :=
by
  sorry

end NUMINAMATH_GPT_problem_eqn_l2156_215631


namespace NUMINAMATH_GPT_koschei_coin_count_l2156_215670

theorem koschei_coin_count (a : ℕ) :
  (a % 10 = 7) ∧
  (a % 12 = 9) ∧
  (300 ≤ a ∧ a ≤ 400) →
  a = 357 :=
sorry

end NUMINAMATH_GPT_koschei_coin_count_l2156_215670


namespace NUMINAMATH_GPT_smallest_gcd_l2156_215619

theorem smallest_gcd (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (H1 : Nat.gcd x y = 270) (H2 : Nat.gcd x z = 105) : Nat.gcd y z = 15 :=
sorry

end NUMINAMATH_GPT_smallest_gcd_l2156_215619


namespace NUMINAMATH_GPT_stamps_per_book_type2_eq_15_l2156_215616

-- Defining the conditions
def num_books_type1 : ℕ := 4
def stamps_per_book_type1 : ℕ := 10
def num_books_type2 : ℕ := 6
def total_stamps : ℕ := 130

-- Stating the theorem to prove the number of stamps in each book of the second type is 15
theorem stamps_per_book_type2_eq_15 : 
  ∀ (x : ℕ), 
    (num_books_type1 * stamps_per_book_type1 + num_books_type2 * x = total_stamps) → 
    x = 15 :=
by
  sorry

end NUMINAMATH_GPT_stamps_per_book_type2_eq_15_l2156_215616


namespace NUMINAMATH_GPT_plane_centroid_l2156_215678

theorem plane_centroid (a b : ℝ) (h : 1 / a ^ 2 + 1 / b ^ 2 + 1 / 25 = 1 / 4) :
  let p := a / 3
  let q := b / 3
  let r := 5 / 3
  (1 / p ^ 2 + 1 / q ^ 2 + 1 / r ^ 2) = 369 / 400 :=
by
  sorry

end NUMINAMATH_GPT_plane_centroid_l2156_215678


namespace NUMINAMATH_GPT_probability_at_least_one_shows_one_is_correct_l2156_215614

/-- Two fair 8-sided dice are rolled. What is the probability that at least one of the dice shows a 1? -/
def probability_at_least_one_shows_one : ℚ :=
  let total_outcomes := 8 * 8
  let neither_one := 7 * 7
  let at_least_one := total_outcomes - neither_one
  at_least_one / total_outcomes

theorem probability_at_least_one_shows_one_is_correct :
  probability_at_least_one_shows_one = 15 / 64 :=
by
  unfold probability_at_least_one_shows_one
  sorry

end NUMINAMATH_GPT_probability_at_least_one_shows_one_is_correct_l2156_215614


namespace NUMINAMATH_GPT_time_to_return_l2156_215612

-- Given conditions
def distance : ℝ := 1000
def return_speed : ℝ := 142.85714285714286

-- Goal to prove
theorem time_to_return : distance / return_speed = 7 := 
by
  sorry

end NUMINAMATH_GPT_time_to_return_l2156_215612


namespace NUMINAMATH_GPT_sin_neg_135_eq_neg_sqrt_2_over_2_l2156_215626

theorem sin_neg_135_eq_neg_sqrt_2_over_2 :
  Real.sin (-135 * Real.pi / 180) = -Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_sin_neg_135_eq_neg_sqrt_2_over_2_l2156_215626


namespace NUMINAMATH_GPT_ribbons_count_l2156_215653

theorem ribbons_count (ribbons : ℕ) 
  (yellow_frac purple_frac orange_frac : ℚ)
  (black_ribbons : ℕ)
  (h1 : yellow_frac = 1/4)
  (h2 : purple_frac = 1/3)
  (h3 : orange_frac = 1/6)
  (h4 : ribbons - (yellow_frac * ribbons + purple_frac * ribbons + orange_frac * ribbons) = black_ribbons) :
  ribbons * orange_frac = 160 / 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_ribbons_count_l2156_215653


namespace NUMINAMATH_GPT_distance_between_points_l2156_215659

open Real

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  (x1, y1) = (-3, 1) →
  (x2, y2) = (5, -5) →
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end NUMINAMATH_GPT_distance_between_points_l2156_215659


namespace NUMINAMATH_GPT_train_speed_in_m_per_s_l2156_215680

theorem train_speed_in_m_per_s (speed_kmph : ℕ) (h : speed_kmph = 162) :
  (speed_kmph * 1000) / 3600 = 45 :=
by {
  sorry
}

end NUMINAMATH_GPT_train_speed_in_m_per_s_l2156_215680
