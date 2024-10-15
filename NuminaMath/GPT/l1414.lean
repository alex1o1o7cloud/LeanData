import Mathlib

namespace NUMINAMATH_GPT_arithmetic_seq_sum_l1414_141490

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
  (h2 : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_sum_l1414_141490


namespace NUMINAMATH_GPT_find_sum_zero_l1414_141474

open Complex

noncomputable def complex_numbers_satisfy (a1 a2 a3 : ℂ) : Prop :=
  a1^2 + a2^2 + a3^2 = 0 ∧
  a1^3 + a2^3 + a3^3 = 0 ∧
  a1^4 + a2^4 + a3^4 = 0

theorem find_sum_zero (a1 a2 a3 : ℂ) (h : complex_numbers_satisfy a1 a2 a3) :
  a1 + a2 + a3 = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_sum_zero_l1414_141474


namespace NUMINAMATH_GPT_train_length_is_150_l1414_141417

noncomputable def train_length (v_km_hr : ℝ) (t_sec : ℝ) : ℝ :=
  let v_m_s := v_km_hr * (5 / 18)
  v_m_s * t_sec

theorem train_length_is_150 :
  train_length 122 4.425875438161669 = 150 :=
by
  -- It follows directly from the given conditions and known conversion factor
  -- The actual proof steps would involve arithmetic simplifications.
  sorry

end NUMINAMATH_GPT_train_length_is_150_l1414_141417


namespace NUMINAMATH_GPT_exists_pretty_hexagon_max_area_pretty_hexagon_l1414_141488

-- Define the condition of a "pretty" hexagon
structure PrettyHexagon (L ℓ h : ℝ) : Prop :=
  (diag1 : (L + ℓ)^2 + h^2 = 1)
  (diag2 : (L + ℓ)^2 + h^2 = 1)
  (diag3 : (L + ℓ)^2 + h^2 = 1)
  (diag4 : (L + ℓ)^2 + h^2 = 1)
  (L_pos : L > 0) (L_lt_1 : L < 1)
  (ℓ_pos : ℓ > 0) (ℓ_lt_1 : ℓ < 1)
  (h_pos : h > 0) (h_lt_1 : h < 1)

-- Area of the hexagon given L, ℓ, and h
def hexagon_area (L ℓ h : ℝ) := 2 * (L + ℓ) * h

-- Question (a): Existence of a pretty hexagon with a given area
theorem exists_pretty_hexagon (k : ℝ) (hk : 0 < k ∧ k < 1) : 
  ∃ L ℓ h : ℝ, PrettyHexagon L ℓ h ∧ hexagon_area L ℓ h = k :=
sorry

-- Question (b): Maximum area of any pretty hexagon is at most 1
theorem max_area_pretty_hexagon : 
  ∀ L ℓ h : ℝ, PrettyHexagon L ℓ h → hexagon_area L ℓ h ≤ 1 :=
sorry

end NUMINAMATH_GPT_exists_pretty_hexagon_max_area_pretty_hexagon_l1414_141488


namespace NUMINAMATH_GPT_smallest_diff_l1414_141496

noncomputable def triangleSides : ℕ → ℕ → ℕ → Prop := λ AB BC AC =>
  AB < BC ∧ BC ≤ AC ∧ AB + BC + AC = 2007

theorem smallest_diff (AB BC AC : ℕ) (h : triangleSides AB BC AC) : BC - AB = 1 :=
  sorry

end NUMINAMATH_GPT_smallest_diff_l1414_141496


namespace NUMINAMATH_GPT_gross_profit_percentage_l1414_141411

theorem gross_profit_percentage (sales_price gross_profit cost : ℝ) 
  (h1 : sales_price = 81) 
  (h2 : gross_profit = 51) 
  (h3 : cost = sales_price - gross_profit) : 
  (gross_profit / cost) * 100 = 170 := 
by
  simp [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_gross_profit_percentage_l1414_141411


namespace NUMINAMATH_GPT_problem_solution_l1414_141465

variables {p q r : ℝ}

theorem problem_solution (h1 : (p + q) * (q + r) * (r + p) / (p * q * r) = 24)
  (h2 : (p - 2 * q) * (q - 2 * r) * (r - 2 * p) / (p * q * r) = 10) :
  ∃ m n : ℕ, (m.gcd n = 1 ∧ (p/q + q/r + r/p = m/n) ∧ m + n = 39) :=
sorry

end NUMINAMATH_GPT_problem_solution_l1414_141465


namespace NUMINAMATH_GPT_solve_fractional_eq_l1414_141475

theorem solve_fractional_eq (x : ℝ) (h₀ : x ≠ 2) (h₁ : x ≠ -2) :
  (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) → (x = 3 / 2) :=
by sorry

end NUMINAMATH_GPT_solve_fractional_eq_l1414_141475


namespace NUMINAMATH_GPT_guinea_pig_food_ratio_l1414_141468

-- Definitions of amounts of food consumed by each guinea pig
def first_guinea_pig_food : ℕ := 2
variable (x : ℕ)
def second_guinea_pig_food : ℕ := x
def third_guinea_pig_food : ℕ := x + 3

-- Total food requirement condition
def total_food_required := first_guinea_pig_food + second_guinea_pig_food x + third_guinea_pig_food x = 13

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- The goal is to prove this ratio given the conditions
theorem guinea_pig_food_ratio (h : total_food_required x) : ratio (second_guinea_pig_food x) first_guinea_pig_food = 2 := by
  sorry

end NUMINAMATH_GPT_guinea_pig_food_ratio_l1414_141468


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l1414_141419

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_A_B: ∃ A B : ℝ, x = -c ∧ |AF| = b^2 / a ∧ |CF| = a + c) :
  e > 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l1414_141419


namespace NUMINAMATH_GPT_power_of_power_example_l1414_141486

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end NUMINAMATH_GPT_power_of_power_example_l1414_141486


namespace NUMINAMATH_GPT_faye_earned_total_l1414_141455

-- Definitions of the necklace sales
def bead_necklaces := 3
def bead_price := 7
def gemstone_necklaces := 7
def gemstone_price := 10
def pearl_necklaces := 2
def pearl_price := 12
def crystal_necklaces := 5
def crystal_price := 15

-- Total amount calculation
def total_amount := 
  bead_necklaces * bead_price + 
  gemstone_necklaces * gemstone_price + 
  pearl_necklaces * pearl_price + 
  crystal_necklaces * crystal_price

-- Proving the total amount equals $190
theorem faye_earned_total : total_amount = 190 := by
  sorry

end NUMINAMATH_GPT_faye_earned_total_l1414_141455


namespace NUMINAMATH_GPT_new_room_correct_size_l1414_141484

-- Definitions of conditions
def current_bedroom := 309 -- sq ft
def current_bathroom := 150 -- sq ft
def current_space := current_bedroom + current_bathroom
def new_room_size := 2 * current_space

-- Proving the new room size
theorem new_room_correct_size : new_room_size = 918 := by
  sorry

end NUMINAMATH_GPT_new_room_correct_size_l1414_141484


namespace NUMINAMATH_GPT_girls_in_class4_1_l1414_141469

theorem girls_in_class4_1 (total_students grade: ℕ)
    (total_girls: ℕ)
    (students_class4_1: ℕ)
    (boys_class4_2: ℕ)
    (h1: total_students = 72)
    (h2: total_girls = 35)
    (h3: students_class4_1 = 36)
    (h4: boys_class4_2 = 19) :
    (total_girls - (total_students - students_class4_1 - boys_class4_2) = 18) :=
by
    sorry

end NUMINAMATH_GPT_girls_in_class4_1_l1414_141469


namespace NUMINAMATH_GPT_satisfies_differential_eqn_l1414_141450

noncomputable def y (x : ℝ) : ℝ := 5 * Real.exp (-2 * x) + (1 / 3) * Real.exp x

theorem satisfies_differential_eqn : ∀ x : ℝ, (deriv y x) + 2 * y x = Real.exp x :=
by
  -- The proof is to be provided
  sorry

end NUMINAMATH_GPT_satisfies_differential_eqn_l1414_141450


namespace NUMINAMATH_GPT_smallest_m_n_l1414_141436

noncomputable def g (m n : ℕ) (x : ℝ) : ℝ := Real.arccos (Real.log (↑n * x) / Real.log (↑m))

theorem smallest_m_n (m n : ℕ) (h1 : 1 < m) (h2 : ∀ x : ℝ, -1 ≤ Real.log (↑n * x) / Real.log (↑m) ∧
                      Real.log (↑n * x) / Real.log (↑m) ≤ 1 ∧
                      (forall a b : ℝ,  a ≤ x ∧ x ≤ b -> b - a = 1 / 1007)) :
  m + n = 1026 :=
sorry

end NUMINAMATH_GPT_smallest_m_n_l1414_141436


namespace NUMINAMATH_GPT_cube_difference_l1414_141435

theorem cube_difference (n : ℕ) (h: 0 < n) : (n + 1)^3 - n^3 = 3 * n^2 + 3 * n + 1 := 
sorry

end NUMINAMATH_GPT_cube_difference_l1414_141435


namespace NUMINAMATH_GPT_smallest_k_l1414_141477

def u (n : ℕ) : ℕ := n^4 + 3 * n^2 + 2

def delta (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => u
  | k+1 => fun n => delta k u (n+1) - delta k u n

theorem smallest_k (n : ℕ) : ∃ k, (forall m, delta k u m = 0) ∧ 
                            (forall j, (∀ m, delta j u m = 0) → j ≥ k) := sorry

end NUMINAMATH_GPT_smallest_k_l1414_141477


namespace NUMINAMATH_GPT_de_morgan_union_de_morgan_inter_l1414_141493

open Set

variable {α : Type*} (A B : Set α)

theorem de_morgan_union : ∀ (A B : Set α), 
  compl (A ∪ B) = compl A ∩ compl B := 
by 
  intro A B
  sorry

theorem de_morgan_inter : ∀ (A B : Set α), 
  compl (A ∩ B) = compl A ∪ compl B := 
by 
  intro A B
  sorry

end NUMINAMATH_GPT_de_morgan_union_de_morgan_inter_l1414_141493


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1414_141425

theorem arithmetic_sequence_problem
  (a : ℕ → ℕ)
  (h1 : a 2 + a 3 = 15)
  (h2 : a 3 + a 4 = 20) :
  a 4 + a 5 = 25 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1414_141425


namespace NUMINAMATH_GPT_pythagorean_triples_l1414_141461

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triples :
  is_pythagorean_triple 3 4 5 ∧ is_pythagorean_triple 6 8 10 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triples_l1414_141461


namespace NUMINAMATH_GPT_range_of_m_l1414_141462

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotonically_decreasing_in_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_m (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing_in_domain f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → -2 < m ∧ m < 1 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1414_141462


namespace NUMINAMATH_GPT_fractions_equiv_x_zero_l1414_141454

theorem fractions_equiv_x_zero (x b : ℝ) (h : x + 3 * b ≠ 0) : 
  (x + 2 * b) / (x + 3 * b) = 2 / 3 ↔ x = 0 :=
by sorry

end NUMINAMATH_GPT_fractions_equiv_x_zero_l1414_141454


namespace NUMINAMATH_GPT_inequality_proof_l1414_141407

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (habc : a * b * (1 / (a * b)) = 1) :
  a^2 + b^2 + (1 / (a * b))^2 + 3 ≥ 2 * (1 / a + 1 / b + a * b) := 
by sorry

end NUMINAMATH_GPT_inequality_proof_l1414_141407


namespace NUMINAMATH_GPT_red_balls_removal_condition_l1414_141495

theorem red_balls_removal_condition (total_balls : ℕ) (initial_red_balls : ℕ) (r : ℕ) : 
  total_balls = 600 → 
  initial_red_balls = 420 → 
  60 * (total_balls - r) = 100 * (initial_red_balls - r) → 
  r = 150 :=
by
  sorry

end NUMINAMATH_GPT_red_balls_removal_condition_l1414_141495


namespace NUMINAMATH_GPT_sally_last_10_shots_made_l1414_141404

def sally_initial_shots : ℕ := 30
def sally_initial_success_rate : ℝ := 0.60
def sally_additional_shots : ℕ := 10
def sally_final_success_rate : ℝ := 0.65

theorem sally_last_10_shots_made (x : ℕ) 
  (h1 : sally_initial_success_rate * sally_initial_shots = 18)
  (h2 : sally_final_success_rate * (sally_initial_shots + sally_additional_shots) = 26) :
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_sally_last_10_shots_made_l1414_141404


namespace NUMINAMATH_GPT_fraction_of_phone_numbers_l1414_141460

-- Define the total number of valid 7-digit phone numbers
def totalValidPhoneNumbers : Nat := 7 * 10^6

-- Define the number of valid phone numbers that begin with 3 and end with 5
def validPhoneNumbersBeginWith3EndWith5 : Nat := 10^5

-- Prove the fraction of phone numbers that begin with 3 and end with 5 is 1/70
theorem fraction_of_phone_numbers (h : validPhoneNumbersBeginWith3EndWith5 = 10^5) 
(h2 : totalValidPhoneNumbers = 7 * 10^6) : 
validPhoneNumbersBeginWith3EndWith5 / totalValidPhoneNumbers = 1 / 70 := 
sorry

end NUMINAMATH_GPT_fraction_of_phone_numbers_l1414_141460


namespace NUMINAMATH_GPT_triangle_perimeter_ABF_l1414_141494

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 21) = 1

-- Define the line
def line (x : ℝ) : Prop := x = -2

-- Define the foci of the ellipse
def right_focus : ℝ := 2
def left_focus : ℝ := -2

-- Points A and B are on the ellipse and line
def point_A (x y : ℝ) : Prop := ellipse x y ∧ line x
def point_B (x y : ℝ) : Prop := ellipse x y ∧ line x

-- Point F is the right focus of the ellipse
def point_F (x y : ℝ) : Prop := x = right_focus ∧ y = 0

-- Perimeter of the triangle ABF
def perimeter (A B F : ℝ × ℝ) : ℝ :=
  sorry -- Calculation of the perimeter of triangle ABF

-- Theorem statement that perimeter is 20
theorem triangle_perimeter_ABF 
  (A B F : ℝ × ℝ) 
  (hA : point_A (A.fst) (A.snd)) 
  (hB : point_B (B.fst) (B.snd))
  (hF : point_F (F.fst) (F.snd)) :
  perimeter A B F = 20 :=
sorry

end NUMINAMATH_GPT_triangle_perimeter_ABF_l1414_141494


namespace NUMINAMATH_GPT_percentage_of_second_solution_correct_l1414_141432

noncomputable def percentage_of_alcohol_in_second_solution : ℝ :=
  let total_liters := 80
  let percentage_final_solution := 0.49
  let volume_first_solution := 24
  let percentage_first_solution := 0.4
  let volume_second_solution := 56
  let total_alcohol_in_final_solution := total_liters * percentage_final_solution
  let total_alcohol_first_solution := volume_first_solution * percentage_first_solution
  let x := (total_alcohol_in_final_solution - total_alcohol_first_solution) / volume_second_solution
  x

theorem percentage_of_second_solution_correct : 
  percentage_of_alcohol_in_second_solution = 0.5285714286 := by sorry

end NUMINAMATH_GPT_percentage_of_second_solution_correct_l1414_141432


namespace NUMINAMATH_GPT_biased_coin_probability_l1414_141426

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- Define the probability mass function for a binomial distribution
def binomial_pmf (n k : ℕ) (p : ℝ) : ℝ :=
  binomial n k * p^k * (1 - p)^(n - k)

-- Define the problem conditions
def problem_conditions : Prop :=
  let p := 1 / 3
  binomial_pmf 5 1 p = binomial_pmf 5 2 p ∧ p ≠ 0 ∧ (1 - p) ≠ 0

-- The target probability to prove
def target_probability := 40 / 243

-- The theorem statement
theorem biased_coin_probability : problem_conditions → binomial_pmf 5 3 (1 / 3) = target_probability :=
by
  intro h
  sorry

end NUMINAMATH_GPT_biased_coin_probability_l1414_141426


namespace NUMINAMATH_GPT_magnitude_of_b_l1414_141472

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)

-- Defining the conditions
def vector_a_magnitude : Prop := ‖a‖ = 1
def vector_angle_condition : Prop := θ = Real.pi / 3
def linear_combination_magnitude : Prop := ‖2 • a - b‖ = 2 * Real.sqrt 3
def b_magnitude : Prop := ‖b‖ = 4

-- The statement we want to prove
theorem magnitude_of_b (h1 : vector_a_magnitude a) (h2 : vector_angle_condition θ) (h3 : linear_combination_magnitude a b) : b_magnitude b :=
sorry

end NUMINAMATH_GPT_magnitude_of_b_l1414_141472


namespace NUMINAMATH_GPT_number_of_members_l1414_141416

variable (n : ℕ)

-- Conditions
def each_member_contributes_n_cents : Prop := n * n = 64736

-- Theorem that relates to the number of members being 254
theorem number_of_members (h : each_member_contributes_n_cents n) : n = 254 :=
sorry

end NUMINAMATH_GPT_number_of_members_l1414_141416


namespace NUMINAMATH_GPT_number_of_sodas_bought_l1414_141497

-- Definitions based on conditions
def cost_sandwich : ℝ := 1.49
def cost_two_sandwiches : ℝ := 2 * cost_sandwich
def cost_soda : ℝ := 0.87
def total_cost : ℝ := 6.46

-- We need to prove that the number of sodas bought is 4 given these conditions
theorem number_of_sodas_bought : (total_cost - cost_two_sandwiches) / cost_soda = 4 := by
  sorry

end NUMINAMATH_GPT_number_of_sodas_bought_l1414_141497


namespace NUMINAMATH_GPT_plane_through_point_and_line_l1414_141478

noncomputable def point_on_plane (A B C D : ℤ) (x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

def line_eq_1 (x y : ℤ) : Prop :=
  3 * x + 4 * y - 20 = 0

def line_eq_2 (y z : ℤ) : Prop :=
  -3 * y + 2 * z + 18 = 0

theorem plane_through_point_and_line 
  (A B C D : ℤ)
  (h_point : point_on_plane A B C D 1 9 (-8))
  (h_line1 : ∀ x y, line_eq_1 x y → point_on_plane A B C D x y 0)
  (h_line2 : ∀ y z, line_eq_2 y z → point_on_plane A B C D 0 y z)
  (h_gcd : Int.gcd (Int.gcd (Int.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1) 
  (h_pos : A > 0) :
  A = 75 ∧ B = -29 ∧ C = 86 ∧ D = 274 :=
sorry

end NUMINAMATH_GPT_plane_through_point_and_line_l1414_141478


namespace NUMINAMATH_GPT_not_all_prime_distinct_l1414_141480

theorem not_all_prime_distinct (a1 a2 a3 : ℕ) (h1 : a1 ≠ a2) (h2 : a2 ≠ a3) (h3 : a1 ≠ a3)
  (h4 : 0 < a1) (h5 : 0 < a2) (h6 : 0 < a3)
  (h7 : a1 ∣ (a2 + a3 + a2 * a3)) (h8 : a2 ∣ (a3 + a1 + a3 * a1)) (h9 : a3 ∣ (a1 + a2 + a1 * a2)) :
  ¬ (Nat.Prime a1 ∧ Nat.Prime a2 ∧ Nat.Prime a3) :=
by
  sorry

end NUMINAMATH_GPT_not_all_prime_distinct_l1414_141480


namespace NUMINAMATH_GPT_tens_digit_of_even_not_divisible_by_10_l1414_141447

theorem tens_digit_of_even_not_divisible_by_10 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) :
  (N ^ 20) % 100 / 10 % 10 = 7 :=
sorry

end NUMINAMATH_GPT_tens_digit_of_even_not_divisible_by_10_l1414_141447


namespace NUMINAMATH_GPT_largest_digit_divisible_by_6_l1414_141446

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N = 8 ∧ (45670 + N) % 6 = 0 :=
sorry

end NUMINAMATH_GPT_largest_digit_divisible_by_6_l1414_141446


namespace NUMINAMATH_GPT_arrangements_ABC_together_l1414_141485

noncomputable def permutation_count_ABC_together (n : Nat) (unit_size : Nat) (remaining : Nat) : Nat :=
  (Nat.factorial unit_size) * (Nat.factorial (remaining + 1))

theorem arrangements_ABC_together : permutation_count_ABC_together 6 3 3 = 144 :=
by
  sorry

end NUMINAMATH_GPT_arrangements_ABC_together_l1414_141485


namespace NUMINAMATH_GPT_problem1_problem2_l1414_141445

-- Problem 1: Prove that 3 * sqrt(20) - sqrt(45) + sqrt(1 / 5) = (16 * sqrt(5)) / 5
theorem problem1 : 3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1 / 5) = (16 * Real.sqrt 5) / 5 := 
sorry

-- Problem 2: Prove that (sqrt(6) - 2 * sqrt(3))^2 - (2 * sqrt(5) + sqrt(2)) * (2 * sqrt(5) - sqrt(2)) = -12 * sqrt(2)
theorem problem2 : (Real.sqrt 6 - 2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_problem1_problem2_l1414_141445


namespace NUMINAMATH_GPT_stock_percent_change_l1414_141439

-- define initial value of stock
def initial_stock_value (x : ℝ) := x

-- define value after first day's decrease
def value_after_day_one (x : ℝ) := 0.85 * x

-- define value after second day's increase
def value_after_day_two (x : ℝ) := 1.25 * value_after_day_one x

-- Theorem stating the overall percent change is 6.25%
theorem stock_percent_change (x : ℝ) (h : x > 0) :
  ((value_after_day_two x - initial_stock_value x) / initial_stock_value x) * 100 = 6.25 := by sorry

end NUMINAMATH_GPT_stock_percent_change_l1414_141439


namespace NUMINAMATH_GPT_james_weekly_hours_l1414_141473

def james_meditation_total : ℕ :=
  let weekly_minutes := (30 * 2 * 6) + (30 * 2 * 2) -- 1 hour/day for 6 days + 2 hours on Sunday
  weekly_minutes / 60

def james_yoga_total : ℕ :=
  let weekly_minutes := (45 * 2) -- 45 minutes on Monday and Friday
  weekly_minutes / 60

def james_bikeride_total : ℕ :=
  let weekly_minutes := 90
  weekly_minutes / 60

def james_dance_total : ℕ :=
  2 -- 2 hours on Saturday

def james_total_activity_hours : ℕ :=
  james_meditation_total + james_yoga_total + james_bikeride_total + james_dance_total

theorem james_weekly_hours : james_total_activity_hours = 13 := by
  sorry

end NUMINAMATH_GPT_james_weekly_hours_l1414_141473


namespace NUMINAMATH_GPT_domain_of_function_l1414_141421

theorem domain_of_function : 
  {x : ℝ | 0 < x ∧ 4 - x^2 > 0} = {x : ℝ | 0 < x ∧ x < 2} :=
sorry

end NUMINAMATH_GPT_domain_of_function_l1414_141421


namespace NUMINAMATH_GPT_max_xy_l1414_141449

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_xy_l1414_141449


namespace NUMINAMATH_GPT_y_coordinate_of_point_l1414_141415

theorem y_coordinate_of_point (x y : ℝ) (m : ℝ)
  (h₁ : x = 10)
  (h₂ : y = m * x + -2)
  (m_def : m = (0 - (-4)) / (4 - (-4)))
  (h₃ : y = 3) : y = 3 :=
sorry

end NUMINAMATH_GPT_y_coordinate_of_point_l1414_141415


namespace NUMINAMATH_GPT_total_animals_received_l1414_141412

-- Define the conditions
def cats : ℕ := 40
def additionalCats : ℕ := 20
def dogs : ℕ := cats - additionalCats

-- Prove the total number of animals received
theorem total_animals_received : (cats + dogs) = 60 := by
  -- The proof itself is not required in this task
  sorry

end NUMINAMATH_GPT_total_animals_received_l1414_141412


namespace NUMINAMATH_GPT_power_division_l1414_141414

theorem power_division : 3^18 / (27^3) = 19683 := by
  have h1 : 27 = 3^3 := by sorry
  have h2 : (3^3)^3 = 3^(3*3) := by sorry
  have h3 : 27^3 = 3^9 := by
    rw [h1]
    exact h2
  rw [h3]
  have h4 : 3^18 / 3^9 = 3^(18 - 9) := by sorry
  rw [h4]
  norm_num

end NUMINAMATH_GPT_power_division_l1414_141414


namespace NUMINAMATH_GPT_max_n_factoring_polynomial_l1414_141463

theorem max_n_factoring_polynomial :
  ∃ n A B : ℤ, (3 * n + A = 217) ∧ (A * B = 72) ∧ (3 * B + A = n) :=
sorry

end NUMINAMATH_GPT_max_n_factoring_polynomial_l1414_141463


namespace NUMINAMATH_GPT_isosceles_triangle_circumscribed_radius_and_height_l1414_141499

/-
Conditions:
- The isosceles triangle has two equal sides of 20 inches.
- The base of the triangle is 24 inches.

Prove:
1. The radius of the circumscribed circle is 5 inches.
2. The height of the triangle is 16 inches.
-/

theorem isosceles_triangle_circumscribed_radius_and_height 
  (h_eq_sides : ∀ A B C : Type, ∀ (AB AC : ℝ), ∀ (BC : ℝ), AB = 20 → AC = 20 → BC = 24) 
  (R : ℝ) (h : ℝ) : 
  R = 5 ∧ h = 16 := 
sorry

end NUMINAMATH_GPT_isosceles_triangle_circumscribed_radius_and_height_l1414_141499


namespace NUMINAMATH_GPT_part1_part2_l1414_141402

noncomputable def f (a x : ℝ) : ℝ := a * x^2 + x - a

-- Part 1
theorem part1 (a x : ℝ) (h1 : |a| ≤ 1) (h2 : |x| ≤ 1) : |f a x| ≤ 5/4 :=
by
  sorry

-- Part 2
theorem part2 (a : ℝ) (h : ∃ x ∈ Set.Icc (-1:ℝ) (1:ℝ), f a x = 17/8) : a = -2 :=
by
  sorry

end NUMINAMATH_GPT_part1_part2_l1414_141402


namespace NUMINAMATH_GPT_range_of_a_l1414_141458

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (|x - 1| - |x - 3|) > a) → a < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1414_141458


namespace NUMINAMATH_GPT_seashells_given_l1414_141444

theorem seashells_given (initial left given : ℕ) (h1 : initial = 8) (h2 : left = 2) (h3 : given = initial - left) : given = 6 := by
  sorry

end NUMINAMATH_GPT_seashells_given_l1414_141444


namespace NUMINAMATH_GPT_perp_vec_m_l1414_141476

theorem perp_vec_m (m : ℝ) : (1 : ℝ) * (-1 : ℝ) + 2 * m = 0 → m = 1 / 2 :=
by 
  intro h
  -- Translate the given condition directly
  sorry

end NUMINAMATH_GPT_perp_vec_m_l1414_141476


namespace NUMINAMATH_GPT_number_of_women_l1414_141481

variable (W : ℕ) (x : ℝ)

-- Conditions
def daily_wage_men_and_women (W : ℕ) (x : ℝ) : Prop :=
  24 * 350 + W * x = 11600

def half_men_and_37_women (W : ℕ) (x : ℝ) : Prop :=
  12 * 350 + 37 * x = 24 * 350 + W * x

def daily_wage_man := (350 : ℝ)

-- Proposition to prove
theorem number_of_women (W : ℕ) (x : ℝ) (h1 : daily_wage_men_and_women W x)
  (h2 : half_men_and_37_women W x) : W = 16 := 
  by
  sorry

end NUMINAMATH_GPT_number_of_women_l1414_141481


namespace NUMINAMATH_GPT_find_k_values_l1414_141487

open Set

def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (k : ℝ) : Set ℝ := {x | x^2 - (k + 1) * x + k = 0}

theorem find_k_values (k : ℝ) : (A ∩ B k = B k) ↔ k ∈ ({1, -3} : Set ℝ) := by
  sorry

end NUMINAMATH_GPT_find_k_values_l1414_141487


namespace NUMINAMATH_GPT_sin_4theta_l1414_141413

theorem sin_4theta (θ : ℝ) (h : Complex.exp (Complex.I * θ) = (4 + Complex.I * Real.sqrt 7) / 5) :
  Real.sin (4 * θ) = (144 * Real.sqrt 7) / 625 := by
  sorry

end NUMINAMATH_GPT_sin_4theta_l1414_141413


namespace NUMINAMATH_GPT_filtration_concentration_l1414_141408

-- Variables and conditions used in the problem
variable (P P0 : ℝ) (k t : ℝ)
variable (h1 : P = P0 * Real.exp (-k * t))
variable (h2 : Real.exp (-2 * k) = 0.8)

-- Main statement: Prove the concentration after 5 hours is approximately 57% of the original
theorem filtration_concentration :
  (P0 * Real.exp (-5 * k)) / P0 = 0.57 :=
by sorry

end NUMINAMATH_GPT_filtration_concentration_l1414_141408


namespace NUMINAMATH_GPT_john_bought_two_shirts_l1414_141452

/-- The number of shirts John bought, given the conditions:
1. The first shirt costs $6 more than the second shirt.
2. The first shirt costs $15.
3. The total cost of the shirts is $24,
is equal to 2. -/
theorem john_bought_two_shirts
  (S : ℝ) 
  (first_shirt_cost : ℝ := 15)
  (second_shirt_cost : ℝ := S)
  (cost_difference : first_shirt_cost = second_shirt_cost + 6)
  (total_cost : first_shirt_cost + second_shirt_cost = 24)
  : 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_john_bought_two_shirts_l1414_141452


namespace NUMINAMATH_GPT_sin_double_angle_l1414_141482

open Real 

theorem sin_double_angle (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : cos (α - β) = 12 / 13) 
  (h5 : sin (α + β) = -3 / 5) : 
  sin (2 * α) = -56 / 65 := 
by 
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1414_141482


namespace NUMINAMATH_GPT_find_k_l1414_141466

theorem find_k 
  (x y k : ℚ) 
  (h1 : y = 4 * x - 1) 
  (h2 : y = -1 / 3 * x + 11) 
  (h3 : y = 2 * x + k) : 
  k = 59 / 13 :=
sorry

end NUMINAMATH_GPT_find_k_l1414_141466


namespace NUMINAMATH_GPT_mr_green_expects_expected_potatoes_yield_l1414_141498

theorem mr_green_expects_expected_potatoes_yield :
  ∀ (length_steps width_steps: ℕ) (step_length yield_per_sqft: ℝ),
  length_steps = 18 →
  width_steps = 25 →
  step_length = 2.5 →
  yield_per_sqft = 0.75 →
  (length_steps * step_length) * (width_steps * step_length) * yield_per_sqft = 2109.375 :=
by
  intros length_steps width_steps step_length yield_per_sqft
  intros h_length_steps h_width_steps h_step_length h_yield_per_sqft
  rw [h_length_steps, h_width_steps, h_step_length, h_yield_per_sqft]
  sorry

end NUMINAMATH_GPT_mr_green_expects_expected_potatoes_yield_l1414_141498


namespace NUMINAMATH_GPT_gcd_7429_12345_l1414_141429

theorem gcd_7429_12345 : Int.gcd 7429 12345 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_7429_12345_l1414_141429


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1414_141470

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1414_141470


namespace NUMINAMATH_GPT_proof_P_otimes_Q_l1414_141491

-- Define the sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def Q : Set ℝ := { x | 1 < x }

-- Define the operation ⊗ between sets
def otimes (P Q : Set ℝ) : Set ℝ := { x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q }

-- Prove that P ⊗ Q = [0,1] ∪ (2, +∞)
theorem proof_P_otimes_Q :
  otimes P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (2 < x)} :=
by
 sorry

end NUMINAMATH_GPT_proof_P_otimes_Q_l1414_141491


namespace NUMINAMATH_GPT_find_m_l1414_141428

-- Define the lines l1 and l2
def line1 (x y : ℝ) (m : ℝ) : Prop := x + m^2 * y + 6 = 0
def line2 (x y : ℝ) (m : ℝ) : Prop := (m - 2) * x + 3 * m * y + 2 * m = 0

-- The statement that two lines are parallel
def lines_parallel (m : ℝ) : Prop :=
  ∀ (x y : ℝ), line1 x y m → line2 x y m

-- The mathematically equivalent proof problem
theorem find_m (m : ℝ) (H_parallel : lines_parallel m) : m = 0 ∨ m = -1 :=
sorry

end NUMINAMATH_GPT_find_m_l1414_141428


namespace NUMINAMATH_GPT_age_difference_l1414_141489

variable (P M Mo N : ℚ)

-- Given conditions as per problem statement
axiom ratio_P_M : (P / M) = 3 / 5
axiom ratio_M_Mo : (M / Mo) = 3 / 4
axiom ratio_Mo_N : (Mo / N) = 5 / 7
axiom sum_ages : P + M + Mo + N = 228

-- Statement to prove
theorem age_difference (ratio_P_M : (P / M) = 3 / 5)
                        (ratio_M_Mo : (M / Mo) = 3 / 4)
                        (ratio_Mo_N : (Mo / N) = 5 / 7)
                        (sum_ages : P + M + Mo + N = 228) :
  N - P = 69.5 := 
sorry

end NUMINAMATH_GPT_age_difference_l1414_141489


namespace NUMINAMATH_GPT_discount_problem_l1414_141456

theorem discount_problem (x : ℝ) (h : 560 * (1 - x / 100) * 0.70 = 313.6) : x = 20 := 
by
  sorry

end NUMINAMATH_GPT_discount_problem_l1414_141456


namespace NUMINAMATH_GPT_rachel_picked_2_apples_l1414_141406

def apples_picked (initial_apples picked_apples final_apples : ℕ) : Prop :=
  initial_apples - picked_apples = final_apples

theorem rachel_picked_2_apples (initial_apples final_apples : ℕ)
  (h_initial : initial_apples = 9)
  (h_final : final_apples = 7) :
  apples_picked initial_apples 2 final_apples :=
by
  rw [h_initial, h_final]
  sorry

end NUMINAMATH_GPT_rachel_picked_2_apples_l1414_141406


namespace NUMINAMATH_GPT_max_f_l1414_141453

open Real

noncomputable def f (x y z : ℝ) := (1 - y * z + z) * (1 - z * x + x) * (1 - x * y + y)

theorem max_f (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  f x y z ≤ 1 ∧ (x = 1 ∧ y = 1 ∧ z = 1 → f x y z = 1) := sorry

end NUMINAMATH_GPT_max_f_l1414_141453


namespace NUMINAMATH_GPT_degree_measure_supplement_complement_l1414_141422

noncomputable def supp_degree_complement (α : ℕ) := 180 - (90 - α)

theorem degree_measure_supplement_complement : 
  supp_degree_complement 36 = 126 :=
by sorry

end NUMINAMATH_GPT_degree_measure_supplement_complement_l1414_141422


namespace NUMINAMATH_GPT_angle_opposite_c_exceeds_l1414_141438

theorem angle_opposite_c_exceeds (a b : ℝ) (c : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 2) (h_c : c >= 4) : 
  C >= 120 := 
sorry

end NUMINAMATH_GPT_angle_opposite_c_exceeds_l1414_141438


namespace NUMINAMATH_GPT_team_win_percentage_remaining_l1414_141418

theorem team_win_percentage_remaining (won_first_30: ℝ) (total_games: ℝ) (total_wins: ℝ)
  (h1: won_first_30 = 0.40 * 30)
  (h2: total_games = 120)
  (h3: total_wins = 0.70 * total_games) :
  (total_wins - won_first_30) / (total_games - 30) * 100 = 80 :=
by
  sorry


end NUMINAMATH_GPT_team_win_percentage_remaining_l1414_141418


namespace NUMINAMATH_GPT_part1_part2_l1414_141441

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l1414_141441


namespace NUMINAMATH_GPT_sum_of_cubes_of_roots_l1414_141400

theorem sum_of_cubes_of_roots :
  ∀ (x1 x2 : ℝ), (2 * x1^2 - 5 * x1 + 1 = 0) ∧ (2 * x2^2 - 5 * x2 + 1 = 0) →
  (x1 + x2 = 5 / 2) ∧ (x1 * x2 = 1 / 2) →
  (x1^3 + x2^3 = 95 / 8) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_cubes_of_roots_l1414_141400


namespace NUMINAMATH_GPT_oyster_crab_ratio_l1414_141464

theorem oyster_crab_ratio
  (O1 C1 : ℕ)
  (h1 : O1 = 50)
  (h2 : C1 = 72)
  (h3 : ∃ C2 : ℕ, C2 = (2 * C1) / 3)
  (h4 : ∃ O2 : ℕ, O1 + C1 + O2 + C2 = 195) :
  ∃ ratio : ℚ, ratio = O2 / O1 ∧ ratio = (1 : ℚ) / 2 := 
by 
  sorry

end NUMINAMATH_GPT_oyster_crab_ratio_l1414_141464


namespace NUMINAMATH_GPT_cost_per_person_is_125_l1414_141420

-- Defining the conditions
def totalCost : ℤ := 25000000000
def peopleSharing : ℤ := 200000000

-- Define the expected cost per person based on the conditions
def costPerPerson : ℤ := totalCost / peopleSharing

-- Proving that the cost per person is 125 dollars.
theorem cost_per_person_is_125 : costPerPerson = 125 := by
  sorry

end NUMINAMATH_GPT_cost_per_person_is_125_l1414_141420


namespace NUMINAMATH_GPT_probability_of_events_l1414_141409

-- Define the sets of tiles in each box
def boxA : Set ℕ := {n | 1 ≤ n ∧ n ≤ 25}
def boxB : Set ℕ := {n | 15 ≤ n ∧ n ≤ 40}

-- Define the specific conditions
def eventA (tile : ℕ) : Prop := tile ≤ 20
def eventB (tile : ℕ) : Prop := (Odd tile ∨ tile > 35)

-- Define the probabilities as calculations
def prob_eventA : ℚ := 20 / 25
def prob_eventB : ℚ := 15 / 26

-- The final probability given independence
def combined_prob : ℚ := prob_eventA * prob_eventB

-- The theorem statement we want to prove
theorem probability_of_events :
  combined_prob = 6 / 13 := 
by 
  -- proof details would go here
  sorry

end NUMINAMATH_GPT_probability_of_events_l1414_141409


namespace NUMINAMATH_GPT_ellipse_iff_k_range_l1414_141483

theorem ellipse_iff_k_range (k : ℝ) :
  (∃ x y, (x ^ 2 / (1 - k)) + (y ^ 2 / (1 + k)) = 1) ↔ (-1 < k ∧ k < 1 ∧ k ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_iff_k_range_l1414_141483


namespace NUMINAMATH_GPT_find_a3_l1414_141437

theorem find_a3 (a0 a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, x^4 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4) →
  a3 = -8 :=
by
  sorry

end NUMINAMATH_GPT_find_a3_l1414_141437


namespace NUMINAMATH_GPT_wool_production_equivalence_l1414_141451

variable (x y z w v : ℕ)

def wool_per_sheep_of_breed_A_per_day : ℚ :=
  (y:ℚ) / ((x:ℚ) * (z:ℚ))

def wool_per_sheep_of_breed_B_per_day : ℚ :=
  2 * wool_per_sheep_of_breed_A_per_day x y z

def total_wool_produced_by_breed_B (x y z w v: ℕ) : ℚ :=
  (w:ℚ) * wool_per_sheep_of_breed_B_per_day x y z * (v:ℚ)

theorem wool_production_equivalence :
  total_wool_produced_by_breed_B x y z w v = 2 * (y:ℚ) * (w:ℚ) * (v:ℚ) / ((x:ℚ) * (z:ℚ)) := by
  sorry

end NUMINAMATH_GPT_wool_production_equivalence_l1414_141451


namespace NUMINAMATH_GPT_find_s_l1414_141440

theorem find_s 
  (a b c x s z : ℕ)
  (h1 : a + b = x)
  (h2 : x + c = s)
  (h3 : s + a = z)
  (h4 : b + c + z = 16) : 
  s = 8 := 
sorry

end NUMINAMATH_GPT_find_s_l1414_141440


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1414_141405

theorem simplify_and_evaluate_expression :
  (2 * (-1/2) + 3 * 1)^2 - (2 * (-1/2) + 1) * (2 * (-1/2) - 1) = 4 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1414_141405


namespace NUMINAMATH_GPT_find_x_pow_3a_minus_b_l1414_141431

variable (x : ℝ) (a b : ℝ)
theorem find_x_pow_3a_minus_b (h1 : x^a = 2) (h2 : x^b = 9) : x^(3 * a - b) = 8 / 9 :=
  sorry

end NUMINAMATH_GPT_find_x_pow_3a_minus_b_l1414_141431


namespace NUMINAMATH_GPT_rate_of_current_l1414_141410

variable (c : ℝ)

-- Define the given conditions
def speed_still_water : ℝ := 4.5
def time_ratio : ℝ := 2

-- Define the effective speeds
def speed_downstream : ℝ := speed_still_water + c
def speed_upstream : ℝ := speed_still_water - c

-- Define the condition that it takes twice as long to row upstream as downstream
def rowing_equation : Prop := 1 / speed_upstream = 2 * (1 / speed_downstream)

-- The Lean theorem stating the problem we need to prove
theorem rate_of_current (h : rowing_equation) : c = 1.5 := by
  sorry

end NUMINAMATH_GPT_rate_of_current_l1414_141410


namespace NUMINAMATH_GPT_second_rooster_weight_l1414_141401

theorem second_rooster_weight (cost_per_kg : ℝ) (weight_1 : ℝ) (total_earnings : ℝ) (weight_2 : ℝ) :
  cost_per_kg = 0.5 →
  weight_1 = 30 →
  total_earnings = 35 →
  total_earnings = weight_1 * cost_per_kg + weight_2 * cost_per_kg →
  weight_2 = 40 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_second_rooster_weight_l1414_141401


namespace NUMINAMATH_GPT_prism_surface_area_l1414_141479

-- Define the base of the prism as an isosceles trapezoid ABCD
structure Trapezoid :=
(AB CD : ℝ)
(BC : ℝ)
(AD : ℝ)

-- Define the properties of the prism
structure Prism :=
(base : Trapezoid)
(diagonal_cross_section_area : ℝ)

-- Define the specific isosceles trapezoid from the problem
def myTrapezoid : Trapezoid :=
{ AB := 13, CD := 13, BC := 11, AD := 21 }

-- Define the specific prism from the problem with the given conditions
noncomputable def myPrism : Prism :=
{ base := myTrapezoid, diagonal_cross_section_area := 180 }

-- Define the total surface area as a function
noncomputable def total_surface_area (p : Prism) : ℝ :=
2 * (1 / 2 * (p.base.AD + p.base.BC) * (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2))) +
(p.base.AB + p.base.BC + p.base.CD + p.base.AD) * (p.diagonal_cross_section_area / (Real.sqrt ((1 / 2 * (p.base.AD + p.base.BC)) ^ 2 + (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2)) ^ 2)))

-- The proof problem in Lean
theorem prism_surface_area :
  total_surface_area myPrism = 906 :=
sorry

end NUMINAMATH_GPT_prism_surface_area_l1414_141479


namespace NUMINAMATH_GPT_largest_possible_value_l1414_141433

-- Definitions for the conditions
def lower_x_bound := -4
def upper_x_bound := -2
def lower_y_bound := 2
def upper_y_bound := 4

-- The proposition to prove
theorem largest_possible_value (x y : ℝ) 
    (h1 : lower_x_bound ≤ x) (h2 : x ≤ upper_x_bound)
    (h3 : lower_y_bound ≤ y) (h4 : y ≤ upper_y_bound) :
    ∃ v, v = (x + y) / x ∧ ∀ (w : ℝ), w = (x + y) / x → w ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_largest_possible_value_l1414_141433


namespace NUMINAMATH_GPT_relationship_a_b_c_d_l1414_141457

theorem relationship_a_b_c_d 
  (a b c d : ℤ)
  (h : (a + b + 1) * (d + a + 2) = (c + d + 1) * (b + c + 2)) : 
  a + b + c + d = -2 := 
sorry

end NUMINAMATH_GPT_relationship_a_b_c_d_l1414_141457


namespace NUMINAMATH_GPT_find_reciprocal_sum_l1414_141471

theorem find_reciprocal_sum
  (m n : ℕ)
  (h_sum : m + n = 72)
  (h_hcf : Nat.gcd m n = 6)
  (h_lcm : Nat.lcm m n = 210) :
  (1 / (m : ℚ)) + (1 / (n : ℚ)) = 6 / 105 :=
by
  sorry

end NUMINAMATH_GPT_find_reciprocal_sum_l1414_141471


namespace NUMINAMATH_GPT_food_requirement_l1414_141424

/-- Peter has six horses. Each horse eats 5 pounds of oats, three times a day, and 4 pounds of grain twice a day. -/
def totalFoodRequired (horses : ℕ) (days : ℕ) (oatsMeal : ℕ) (oatsMealsPerDay : ℕ) (grainMeal : ℕ) (grainMealsPerDay : ℕ) : ℕ :=
  let dailyOats := oatsMeal * oatsMealsPerDay
  let dailyGrain := grainMeal * grainMealsPerDay
  let dailyFood := dailyOats + dailyGrain
  let totalDailyFood := dailyFood * horses
  totalDailyFood * days

theorem food_requirement :
  totalFoodRequired 6 5 5 3 4 2 = 690 :=
by sorry

end NUMINAMATH_GPT_food_requirement_l1414_141424


namespace NUMINAMATH_GPT_greatest_divisor_same_remainder_l1414_141448

theorem greatest_divisor_same_remainder (a b c : ℕ) (d1 d2 d3 : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113)
(hd1 : d1 = b - a) (hd2 : d2 = c - b) (hd3 : d3 = c - a) :
  Nat.gcd (Nat.gcd d1 d2) d3 = 6 :=
by
  -- some computation here which we are skipping
  sorry

end NUMINAMATH_GPT_greatest_divisor_same_remainder_l1414_141448


namespace NUMINAMATH_GPT_octagon_perimeter_l1414_141459

theorem octagon_perimeter (n : ℕ) (side_length : ℝ) (h1 : n = 8) (h2 : side_length = 2) : 
  n * side_length = 16 :=
by
  sorry

end NUMINAMATH_GPT_octagon_perimeter_l1414_141459


namespace NUMINAMATH_GPT_range_of_m_l1414_141423

open Real

theorem range_of_m 
    (a : ℝ) (ha_pos : a > 0) (ha_ne_one : a ≠ 1) 
    (m : ℝ)
    (h : m * (a + 1/a) / sqrt 2 > 1) : 
    m ≥ sqrt 2 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_m_l1414_141423


namespace NUMINAMATH_GPT_find_d_l1414_141427

noncomputable def d_value (a b c : ℝ) := (2 * a + 2 * b + 2 * c - (3 / 4)^2) / 3

theorem find_d (a b c d : ℝ) (h : 2 * a^2 + 2 * b^2 + 2 * c^2 + 3 = 2 * d + (2 * a + 2 * b + 2 * c - 3 * d)^(1/2)) : 
  d = 23 / 48 :=
sorry

end NUMINAMATH_GPT_find_d_l1414_141427


namespace NUMINAMATH_GPT_exponentiation_equation_l1414_141430

theorem exponentiation_equation : 4^2011 * (-0.25)^2010 - 1 = 3 := 
by { sorry }

end NUMINAMATH_GPT_exponentiation_equation_l1414_141430


namespace NUMINAMATH_GPT_total_time_proof_l1414_141403

variable (mow_time : ℕ) (fertilize_time : ℕ) (total_time : ℕ)

-- Based on the problem conditions.
axiom mow_time_def : mow_time = 40
axiom fertilize_time_def : fertilize_time = 2 * mow_time
axiom total_time_def : total_time = mow_time + fertilize_time

-- The proof goal
theorem total_time_proof : total_time = 120 := by
  sorry

end NUMINAMATH_GPT_total_time_proof_l1414_141403


namespace NUMINAMATH_GPT_total_trip_hours_l1414_141443

-- Define the given conditions
def speed1 := 50 -- Speed in mph for the first 4 hours
def time1 := 4 -- First 4 hours
def distance1 := speed1 * time1 -- Distance covered in the first 4 hours

def speed2 := 80 -- Speed in mph for additional hours
def average_speed := 65 -- Average speed for the entire trip

-- Define the proof problem
theorem total_trip_hours (T : ℕ) (A : ℕ) :
  distance1 + (speed2 * A) = average_speed * T ∧ T = time1 + A → T = 8 :=
by
  sorry

end NUMINAMATH_GPT_total_trip_hours_l1414_141443


namespace NUMINAMATH_GPT_number_of_birds_is_122_l1414_141434

-- Defining the variables
variables (b m i : ℕ)

-- Define the conditions as part of an axiom
axiom heads_count : b + m + i = 300
axiom legs_count : 2 * b + 4 * m + 6 * i = 1112

-- We aim to prove the number of birds is 122
theorem number_of_birds_is_122 (h1 : b + m + i = 300) (h2 : 2 * b + 4 * m + 6 * i = 1112) : b = 122 := by
  sorry

end NUMINAMATH_GPT_number_of_birds_is_122_l1414_141434


namespace NUMINAMATH_GPT_maria_made_144_cookies_l1414_141492

def cookies (C : ℕ) : Prop :=
  (2 * 1 / 4 * C = 72)

theorem maria_made_144_cookies: ∃ (C : ℕ), cookies C ∧ C = 144 :=
by
  existsi 144
  unfold cookies
  sorry

end NUMINAMATH_GPT_maria_made_144_cookies_l1414_141492


namespace NUMINAMATH_GPT_sum_reciprocal_eq_eleven_eighteen_l1414_141467

noncomputable def sum_reciprocal (n : ℕ) : ℝ := ∑' (n : ℕ), 1 / (n * (n + 3))

theorem sum_reciprocal_eq_eleven_eighteen :
  sum_reciprocal = 11 / 18 :=
by
  sorry

end NUMINAMATH_GPT_sum_reciprocal_eq_eleven_eighteen_l1414_141467


namespace NUMINAMATH_GPT_sum_of_fourth_powers_l1414_141442

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 :=
by sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_l1414_141442
