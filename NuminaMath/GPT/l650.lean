import Mathlib

namespace NUMINAMATH_GPT_upstream_speed_l650_65089

-- Speed of the man in still water
def V_m : ℕ := 32

-- Speed of the man rowing downstream
def V_down : ℕ := 42

-- Speed of the stream
def V_s : ℕ := V_down - V_m

-- Speed of the man rowing upstream
def V_up : ℕ := V_m - V_s

theorem upstream_speed (V_m : ℕ) (V_down : ℕ) (V_s : ℕ) (V_up : ℕ) : 
  V_m = 32 → 
  V_down = 42 → 
  V_s = V_down - V_m → 
  V_up = V_m - V_s → 
  V_up = 22 := 
by intros; 
   repeat {sorry}

end NUMINAMATH_GPT_upstream_speed_l650_65089


namespace NUMINAMATH_GPT_term_largest_binomial_coeff_constant_term_in_expansion_l650_65098

theorem term_largest_binomial_coeff {n : ℕ} (h : n = 8) :
  ∃ (k : ℕ) (coeff : ℤ), coeff * x ^ k = 1120 * x^4 :=
by
  sorry

theorem constant_term_in_expansion :
  ∃ (const : ℤ), const = 1280 :=
by
  sorry

end NUMINAMATH_GPT_term_largest_binomial_coeff_constant_term_in_expansion_l650_65098


namespace NUMINAMATH_GPT_find_f_at_9_over_2_l650_65026

variable (f : ℝ → ℝ)

-- Domain of f is ℝ
axiom domain_f : ∀ x : ℝ, f x = f x

-- f(x+1) is an odd function
axiom odd_f : ∀ x : ℝ, f (x + 1) = -f (-(x - 1))

-- f(x+2) is an even function
axiom even_f : ∀ x : ℝ, f (x + 2) = f (-(x - 2))

-- When x is in [1,2], f(x) = ax^2 + b
variables (a b : ℝ)
axiom on_interval : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → f x = a * x^2 + b

-- f(0) + f(3) = 6
axiom sum_f : f 0 + f 3 = 6 

theorem find_f_at_9_over_2 : f (9/2) = 5/2 := 
by sorry

end NUMINAMATH_GPT_find_f_at_9_over_2_l650_65026


namespace NUMINAMATH_GPT_Faye_apps_left_l650_65038

theorem Faye_apps_left (total_apps gaming_apps utility_apps deleted_gaming_apps deleted_utility_apps remaining_apps : ℕ)
  (h1 : total_apps = 12) 
  (h2 : gaming_apps = 5) 
  (h3 : utility_apps = total_apps - gaming_apps) 
  (h4 : remaining_apps = total_apps - (deleted_gaming_apps + deleted_utility_apps))
  (h5 : deleted_gaming_apps = gaming_apps) 
  (h6 : deleted_utility_apps = 3) : 
  remaining_apps = 4 :=
by
  sorry

end NUMINAMATH_GPT_Faye_apps_left_l650_65038


namespace NUMINAMATH_GPT_fruit_store_problem_l650_65044

-- Define the conditions
def total_weight : Nat := 140
def total_cost : Nat := 1000

def purchase_price_A : Nat := 5
def purchase_price_B : Nat := 9

def selling_price_A : Nat := 8
def selling_price_B : Nat := 13

-- Define the total purchase price equation
def purchase_cost (x : Nat) : Nat := purchase_price_A * x + purchase_price_B * (total_weight - x)

-- Define the profit calculation
def profit (x : Nat) (y : Nat) : Nat := (selling_price_A - purchase_price_A) * x + (selling_price_B - purchase_price_B) * y

-- State the problem
theorem fruit_store_problem :
  ∃ x y : Nat, x + y = total_weight ∧ purchase_cost x = total_cost ∧ profit x y = 495 :=
by
  sorry

end NUMINAMATH_GPT_fruit_store_problem_l650_65044


namespace NUMINAMATH_GPT_parallel_lines_l650_65015

-- Definitions for the equations of the lines
def l1 (a : ℝ) (x y : ℝ) := (a - 1) * x + 2 * y + 10 = 0
def l2 (a : ℝ) (x y : ℝ) := x + a * y + 3 = 0

-- Theorem stating the conditions under which the lines l1 and l2 are parallel
theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, l1 a x y) = (∀ x y : ℝ, l2 a x y) → a = -1 ∨ a = 2 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_l650_65015


namespace NUMINAMATH_GPT_regular_polygon_sides_l650_65039

theorem regular_polygon_sides (interior_angle exterior_angle : ℕ)
  (h1 : interior_angle = exterior_angle + 60)
  (h2 : interior_angle + exterior_angle = 180) :
  ∃ n : ℕ, n = 6 :=
by
  have ext_angle_eq : exterior_angle = 60 := sorry
  have ext_angles_sum : exterior_angle * 6 = 360 := sorry
  exact ⟨6, by linarith⟩

end NUMINAMATH_GPT_regular_polygon_sides_l650_65039


namespace NUMINAMATH_GPT_train_crossing_time_l650_65099

theorem train_crossing_time 
  (train_length : ℕ) 
  (train_speed_kmph : ℕ) 
  (conversion_factor : ℚ := 1000/3600) 
  (train_speed_mps : ℚ := train_speed_kmph * conversion_factor) :
  train_length = 100 →
  train_speed_kmph = 72 →
  train_speed_mps = 20 →
  train_length / train_speed_mps = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_crossing_time_l650_65099


namespace NUMINAMATH_GPT_dummies_remainder_l650_65042

/-
  Prove that if the number of Dummies in one bag is such that when divided among 10 kids, 3 pieces are left over,
  then the number of Dummies in four bags when divided among 10 kids leaves 2 pieces.
-/
theorem dummies_remainder (n : ℕ) (h : n % 10 = 3) : (4 * n) % 10 = 2 := 
by {
  sorry
}

end NUMINAMATH_GPT_dummies_remainder_l650_65042


namespace NUMINAMATH_GPT_isosceles_triangle_l650_65002

def shape_of_triangle (A B C : Real) (h : 2 * Real.sin A * Real.cos B = Real.sin C) : Prop :=
  A = B

theorem isosceles_triangle {A B C : Real} (h : 2 * Real.sin A * Real.cos B = Real.sin C) :
  shape_of_triangle A B C h := 
  sorry

end NUMINAMATH_GPT_isosceles_triangle_l650_65002


namespace NUMINAMATH_GPT_minimize_expression_l650_65048

theorem minimize_expression (a b c d : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 :=
by
  sorry

end NUMINAMATH_GPT_minimize_expression_l650_65048


namespace NUMINAMATH_GPT_binary_101110_to_octal_l650_65040

-- Definition: binary number 101110 represents some decimal number
def binary_101110 : ℕ := 0 * 2^0 + 1 * 2^1 + 1 * 2^2 + 1 * 2^3 + 0 * 2^4 + 1 * 2^5

-- Definition: decimal number 46 represents some octal number
def decimal_46 := 46

-- A utility function to convert decimal to octal (returns the digits as a list)
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n < 8 then [n]
  else decimal_to_octal (n / 8) ++ [n % 8]

-- Hypothesis: the binary 101110 equals the decimal 46
lemma binary_101110_eq_46 : binary_101110 = decimal_46 := by sorry

-- Hypothesis: the decimal 46 converts to the octal number 56 (in list form)
def octal_56 := [5, 6]

-- Theorem: binary 101110 converts to the octal number 56
theorem binary_101110_to_octal :
  decimal_to_octal binary_101110 = octal_56 := by
  rw [binary_101110_eq_46]
  sorry

end NUMINAMATH_GPT_binary_101110_to_octal_l650_65040


namespace NUMINAMATH_GPT_max_n_for_factoring_l650_65006

theorem max_n_for_factoring (n : ℤ) :
  (∃ A B : ℤ, (5 * B + A = n) ∧ (A * B = 90)) → n = 451 :=
by
  sorry

end NUMINAMATH_GPT_max_n_for_factoring_l650_65006


namespace NUMINAMATH_GPT_lcm_gcd_eq_product_l650_65017

theorem lcm_gcd_eq_product {a b : ℕ} (h : Nat.lcm a b + Nat.gcd a b = a * b) : a = 2 ∧ b = 2 :=
  sorry

end NUMINAMATH_GPT_lcm_gcd_eq_product_l650_65017


namespace NUMINAMATH_GPT_probability_of_exactly_three_positives_l650_65053

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_exactly_three_positives :
  let p := 2/5
  let n := 7
  let k := 3
  let positive_prob := p^k
  let negative_prob := (1 - p)^(n - k)
  let binomial_coefficient := choose n k
  binomial_coefficient * positive_prob * negative_prob = 22680/78125 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_exactly_three_positives_l650_65053


namespace NUMINAMATH_GPT_zoe_total_cost_correct_l650_65027

theorem zoe_total_cost_correct :
  (6 * 0.5) + (6 * (1 + 2 * 0.75)) + (6 * 2 * 3) = 54 :=
by
  sorry

end NUMINAMATH_GPT_zoe_total_cost_correct_l650_65027


namespace NUMINAMATH_GPT_problems_left_to_grade_l650_65023

def worksheets : ℕ := 17
def graded_worksheets : ℕ := 8
def problems_per_worksheet : ℕ := 7

theorem problems_left_to_grade : (worksheets - graded_worksheets) * problems_per_worksheet = 63 := by
  sorry

end NUMINAMATH_GPT_problems_left_to_grade_l650_65023


namespace NUMINAMATH_GPT_cone_water_fill_percentage_l650_65000

noncomputable def volumeFilledPercentage (h r : ℝ) : ℝ :=
  let original_cone_volume := (1 / 3) * Real.pi * r^2 * h
  let water_cone_volume := (1 / 3) * Real.pi * ((2 / 3) * r)^2 * ((2 / 3) * h)
  let ratio := water_cone_volume / original_cone_volume
  ratio * 100


theorem cone_water_fill_percentage (h r : ℝ) :
  volumeFilledPercentage h r = 29.6296 :=
by
  sorry

end NUMINAMATH_GPT_cone_water_fill_percentage_l650_65000


namespace NUMINAMATH_GPT_final_value_l650_65094

noncomputable def f : ℕ → ℝ := sorry

axiom f_mul_add (a b : ℕ) : f (a + b) = f a * f b
axiom f_one : f 1 = 2

theorem final_value : 
  (f 1)^2 + f 2 / f 1 + (f 2)^2 + f 4 / f 3 + (f 3)^2 + f 6 / f 5 + (f 4)^2 + f 8 / f 7 = 16 := 
sorry

end NUMINAMATH_GPT_final_value_l650_65094


namespace NUMINAMATH_GPT_percent_equality_l650_65077

theorem percent_equality :
  (1 / 4 : ℝ) * 100 = (10 / 100 : ℝ) * 250 :=
by
  sorry

end NUMINAMATH_GPT_percent_equality_l650_65077


namespace NUMINAMATH_GPT_prob_less_than_9_is_correct_l650_65030

-- Define the probabilities
def prob_ring_10 := 0.24
def prob_ring_9 := 0.28
def prob_ring_8 := 0.19

-- Define the condition for scoring less than 9, which does not include hitting the 10 or 9 ring.
def prob_less_than_9 := 1 - prob_ring_10 - prob_ring_9

-- Now we state the theorem we want to prove.
theorem prob_less_than_9_is_correct : prob_less_than_9 = 0.48 :=
by {
  -- Proof would go here
  sorry
}

end NUMINAMATH_GPT_prob_less_than_9_is_correct_l650_65030


namespace NUMINAMATH_GPT_range_of_m_l650_65010

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ := 
  if x ≤ 1 then 2^x + 1 else 1 - Real.log (x) / Real.log 2

-- The problem is to find the range of m such that f(1 - m^2) > f(2m - 2). We assert the range of m as given in the correct answer.
theorem range_of_m : {m : ℝ | f (1 - m^2) > f (2 * m - 2)} = 
  {m : ℝ | -3 < m ∧ m < 1} ∪ {m : ℝ | m > 3 / 2} :=
sorry

end NUMINAMATH_GPT_range_of_m_l650_65010


namespace NUMINAMATH_GPT_first_generation_tail_length_l650_65001

theorem first_generation_tail_length
  (length_first_gen : ℝ)
  (H : (1.25:ℝ) * (1.25:ℝ) * length_first_gen = 25) :
  length_first_gen = 16 := by
  sorry

end NUMINAMATH_GPT_first_generation_tail_length_l650_65001


namespace NUMINAMATH_GPT_radius_of_smaller_circle_l650_65081

theorem radius_of_smaller_circle (A1 : ℝ) (r1 r2 : ℝ) (h1 : π * r2^2 = 4 * A1)
    (h2 : r2 = 4) : r1 = 2 :=
by
  sorry

end NUMINAMATH_GPT_radius_of_smaller_circle_l650_65081


namespace NUMINAMATH_GPT_box_volume_l650_65035

theorem box_volume (x y : ℝ) (hx : 0 < x ∧ x < 6) (hy : 0 < y ∧ y < 8) :
  (16 - 2 * x) * (12 - 2 * y) * y = 192 * y - 32 * y^2 - 24 * x * y + 4 * x * y^2 :=
by
  sorry

end NUMINAMATH_GPT_box_volume_l650_65035


namespace NUMINAMATH_GPT_complement_union_l650_65007

open Set

def set_A : Set ℝ := {x | x ≤ 0}
def set_B : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

theorem complement_union (A B : Set ℝ) (hA : A = set_A) (hB : B = set_B) :
  (univ \ (A ∪ B) = {x | 1 < x}) := by
  rw [hA, hB]
  sorry

end NUMINAMATH_GPT_complement_union_l650_65007


namespace NUMINAMATH_GPT_sum_of_prime_factors_of_143_l650_65036

theorem sum_of_prime_factors_of_143 :
  (Nat.Prime 11) ∧ (Nat.Prime 13) ∧ (143 = 11 * 13) → 11 + 13 = 24 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_prime_factors_of_143_l650_65036


namespace NUMINAMATH_GPT_problem1_problem2_l650_65080

-- Define the function f
def f (x a : ℝ) : ℝ := |x + a| + |3 * x - 1|

-- Part (1) statement
theorem problem1 (x : ℝ) : f x (-1) ≤ 1 ↔ (1/4 ≤ x ∧ x ≤ 1/2) :=
by
    sorry

-- Part (2) statement
theorem problem2 (x a : ℝ) (h : 1/4 ≤ x ∧ x ≤ 1) : f x a ≤ |3 * x + 1| ↔ -7/3 ≤ a ∧ a ≤ 1 :=
by
    sorry

end NUMINAMATH_GPT_problem1_problem2_l650_65080


namespace NUMINAMATH_GPT_number_of_family_members_l650_65062

-- Define the number of legs for each type of animal.
def bird_legs : ℕ := 2
def dog_legs : ℕ := 4
def cat_legs : ℕ := 4

-- Define the number of animals.
def birds : ℕ := 4
def dogs : ℕ := 3
def cats : ℕ := 18

-- Define the total number of legs of all animals.
def total_animal_feet : ℕ := birds * bird_legs + dogs * dog_legs + cats * cat_legs

-- Define the total number of heads of all animals.
def total_animal_heads : ℕ := birds + dogs + cats

-- Main theorem: If the total number of feet in the house is 74 more than the total number of heads, find the number of family members.
theorem number_of_family_members (F : ℕ) (h : total_animal_feet + 2 * F = total_animal_heads + F + 74) : F = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_family_members_l650_65062


namespace NUMINAMATH_GPT_solution_set_ineq_l650_65021

open Set

theorem solution_set_ineq (a x : ℝ) (h : 0 < a ∧ a < 1) : 
 (a < x ∧ x < 1/a) ↔ ((x - a) * (x - 1/a) > 0) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_ineq_l650_65021


namespace NUMINAMATH_GPT_intersection_A_notB_l650_65067

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A according to the given condition
def A : Set ℝ := { x | |x - 1| > 1 }

-- Define set B according to the given condition
def B : Set ℝ := { x | (x - 1) * (x - 4) > 0 }

-- Define the complement of set B in U
def notB : Set ℝ := { x | 1 ≤ x ∧ x ≤ 4 }

-- Lean statement to prove A ∩ notB = { x | 2 < x ∧ x ≤ 4 }
theorem intersection_A_notB :
  A ∩ notB = { x | 2 < x ∧ x ≤ 4 } :=
sorry

end NUMINAMATH_GPT_intersection_A_notB_l650_65067


namespace NUMINAMATH_GPT_sum_of_coordinates_A_l650_65011

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_A_l650_65011


namespace NUMINAMATH_GPT_arina_should_accept_anton_offer_l650_65029

noncomputable def total_shares : ℕ := 300000
noncomputable def arina_shares : ℕ := 90001
noncomputable def need_to_be_largest : ℕ := 104999 
noncomputable def shares_needed : ℕ := 14999
noncomputable def largest_shareholder_total : ℕ := 105000

noncomputable def maxim_shares : ℕ := 104999
noncomputable def inga_shares : ℕ := 30000
noncomputable def yuri_shares : ℕ := 30000
noncomputable def yulia_shares : ℕ := 30000
noncomputable def anton_shares : ℕ := 15000

noncomputable def maxim_price_per_share : ℕ := 11
noncomputable def inga_price_per_share : ℕ := 1250 / 100
noncomputable def yuri_price_per_share : ℕ := 1150 / 100
noncomputable def yulia_price_per_share : ℕ := 1300 / 100
noncomputable def anton_price_per_share : ℕ := 14

noncomputable def anton_total_cost : ℕ := anton_shares * anton_price_per_share
noncomputable def yuri_total_cost : ℕ := yuri_shares * yuri_price_per_share
noncomputable def inga_total_cost : ℕ := inga_shares * inga_price_per_share
noncomputable def yulia_total_cost : ℕ := yulia_shares * yulia_price_per_share

theorem arina_should_accept_anton_offer :
  anton_total_cost = 210000 := by
  sorry

end NUMINAMATH_GPT_arina_should_accept_anton_offer_l650_65029


namespace NUMINAMATH_GPT_age_difference_between_Mandy_and_sister_l650_65076

variable (Mandy_age Brother_age Sister_age : ℕ)

-- Given conditions
def Mandy_is_3_years_old : Mandy_age = 3 := by sorry
def Brother_is_4_times_older : Brother_age = 4 * Mandy_age := by sorry
def Sister_is_5_years_younger_than_brother : Sister_age = Brother_age - 5 := by sorry

-- Prove the question
theorem age_difference_between_Mandy_and_sister :
  Mandy_age = 3 ∧ Brother_age = 4 * Mandy_age ∧ Sister_age = Brother_age - 5 → Sister_age - Mandy_age = 4 := 
by 
  sorry

end NUMINAMATH_GPT_age_difference_between_Mandy_and_sister_l650_65076


namespace NUMINAMATH_GPT_prime_divisor_property_l650_65028

-- Given conditions
variable (p k : ℕ)
variable (prime_p : Nat.Prime p)
variable (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1)

-- The theorem we need to prove
theorem prime_divisor_property (p k : ℕ) (prime_p : Nat.Prime p) (divisor_p : p ∣ (2 ^ (2 ^ k)) + 1) : (2 ^ (k + 1)) ∣ (p - 1) := 
by 
  sorry

end NUMINAMATH_GPT_prime_divisor_property_l650_65028


namespace NUMINAMATH_GPT_first_player_always_wins_l650_65078

theorem first_player_always_wins (A B : ℤ) (hA : A ≠ 0) (hB : B ≠ 0) : A + B + 1998 = 0 → 
  (∃ (a b c : ℤ), (a = A ∨ a = B ∨ a = 1998) ∧ (b = A ∨ b = B ∨ b = 1998) ∧ (c = A ∨ c = B ∨ c = 1998) ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  (∃ (r1 r2 : ℚ), r1 ≠ r2 ∧ r1 * r1 * a + r1 * b + c = 0 ∧ r2 * r2 * a + r2 * b + c = 0)) :=
sorry

end NUMINAMATH_GPT_first_player_always_wins_l650_65078


namespace NUMINAMATH_GPT_find_salary_June_l650_65025

variable (J F M A May_s June_s : ℝ)
variable (h1 : J + F + M + A = 4 * 8000)
variable (h2 : F + M + A + May_s = 4 * 8450)
variable (h3 : May_s = 6500)
variable (h4 : M + A + May_s + June_s = 4 * 9000)
variable (h5 : June_s = 1.2 * May_s)

theorem find_salary_June : June_s = 7800 := by
  sorry

end NUMINAMATH_GPT_find_salary_June_l650_65025


namespace NUMINAMATH_GPT_planting_equation_l650_65090

def condition1 (x : ℕ) : ℕ := 5 * x + 3
def condition2 (x : ℕ) : ℕ := 6 * x - 4

theorem planting_equation (x : ℕ) : condition1 x = condition2 x := by
  sorry

end NUMINAMATH_GPT_planting_equation_l650_65090


namespace NUMINAMATH_GPT_rahul_share_is_100_l650_65069

-- Definitions of the conditions
def rahul_rate := 1/3
def rajesh_rate := 1/2
def total_payment := 250

-- Definition of their work rate when they work together
def combined_rate := rahul_rate + rajesh_rate

-- Definition of the total value of the work done in one day when both work together
noncomputable def combined_work_value := total_payment / combined_rate

-- Definition of Rahul's share for the work done in one day
noncomputable def rahul_share := rahul_rate * combined_work_value

-- The theorem we need to prove
theorem rahul_share_is_100 : rahul_share = 100 := by
  sorry

end NUMINAMATH_GPT_rahul_share_is_100_l650_65069


namespace NUMINAMATH_GPT_simplify_expression_l650_65073

theorem simplify_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l650_65073


namespace NUMINAMATH_GPT_mrs_sheridan_cats_l650_65056

theorem mrs_sheridan_cats (initial_cats : ℝ) (given_away_cats : ℝ) (remaining_cats : ℝ) :
  initial_cats = 17.0 → given_away_cats = 14.0 → remaining_cats = (initial_cats - given_away_cats) → remaining_cats = 3.0 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mrs_sheridan_cats_l650_65056


namespace NUMINAMATH_GPT_parabola_focus_coordinates_l650_65096

theorem parabola_focus_coordinates :
  ∃ (focus : ℝ × ℝ), focus = (0, 1 / 18) ∧ 
    ∃ (p : ℝ), y = 9 * x^2 → x^2 = 4 * p * y ∧ p = 1 / 18 :=
by
  sorry

end NUMINAMATH_GPT_parabola_focus_coordinates_l650_65096


namespace NUMINAMATH_GPT_average_salary_excluding_manager_l650_65074

theorem average_salary_excluding_manager
    (A : ℝ)
    (manager_salary : ℝ)
    (total_employees : ℕ)
    (salary_increase : ℝ)
    (h1 : total_employees = 24)
    (h2 : manager_salary = 4900)
    (h3 : salary_increase = 100)
    (h4 : 24 * A + manager_salary = 25 * (A + salary_increase)) :
    A = 2400 := by
  sorry

end NUMINAMATH_GPT_average_salary_excluding_manager_l650_65074


namespace NUMINAMATH_GPT_rectangle_area_from_perimeter_l650_65003

theorem rectangle_area_from_perimeter
  (a : ℝ)
  (shorter_side := 12 * a)
  (longer_side := 22 * a)
  (P := 2 * (shorter_side + longer_side))
  (hP : P = 102) :
  (shorter_side * longer_side = 594) := by
  sorry

end NUMINAMATH_GPT_rectangle_area_from_perimeter_l650_65003


namespace NUMINAMATH_GPT_find_a_values_l650_65016

theorem find_a_values (a n : ℕ) (h1 : 7 * a * n - 3 * n = 2020) :
    a = 68 ∨ a = 289 := sorry

end NUMINAMATH_GPT_find_a_values_l650_65016


namespace NUMINAMATH_GPT_union_sets_l650_65082

theorem union_sets :
  let A := { x : ℝ | x^2 - x - 2 < 0 }
  let B := { x : ℝ | x > -2 ∧ x < 0 }
  A ∪ B = { x : ℝ | x > -2 ∧ x < 2 } :=
by
  sorry

end NUMINAMATH_GPT_union_sets_l650_65082


namespace NUMINAMATH_GPT_consecutive_integers_equation_l650_65033

theorem consecutive_integers_equation
  (X Y : ℕ)
  (h_consecutive : Y = X + 1)
  (h_equation : 2 * X^2 + 4 * X + 5 * Y + 3 = (X + Y)^2 + 9 * (X + Y) + 4) :
  X + Y = 15 := by
  sorry

end NUMINAMATH_GPT_consecutive_integers_equation_l650_65033


namespace NUMINAMATH_GPT_fixed_point_of_family_of_lines_l650_65084

theorem fixed_point_of_family_of_lines :
  ∀ (m : ℝ), ∃ (x y : ℝ), (2 * x - m * y + 1 - 3 * m = 0) ∧ (x = -1 / 2) ∧ (y = -3) :=
by
  intro m
  use -1 / 2, -3
  constructor
  · sorry
  constructor
  · rfl
  · rfl

end NUMINAMATH_GPT_fixed_point_of_family_of_lines_l650_65084


namespace NUMINAMATH_GPT_min_trucks_for_crates_l650_65050

noncomputable def min_trucks (total_weight : ℕ) (max_weight_per_crate : ℕ) (truck_capacity : ℕ) : ℕ :=
  if total_weight % truck_capacity = 0 then total_weight / truck_capacity
  else total_weight / truck_capacity + 1

theorem min_trucks_for_crates :
  ∀ (total_weight max_weight_per_crate truck_capacity : ℕ),
    total_weight = 10 →
    max_weight_per_crate = 1 →
    truck_capacity = 3 →
    min_trucks total_weight max_weight_per_crate truck_capacity = 5 :=
by
  intros total_weight max_weight_per_crate truck_capacity h_total h_max h_truck
  rw [h_total, h_max, h_truck]
  sorry

end NUMINAMATH_GPT_min_trucks_for_crates_l650_65050


namespace NUMINAMATH_GPT_child_tickets_sold_l650_65019

noncomputable def price_adult_ticket : ℝ := 7
noncomputable def price_child_ticket : ℝ := 4
noncomputable def total_tickets_sold : ℝ := 900
noncomputable def total_revenue : ℝ := 5100

theorem child_tickets_sold : ∃ (C : ℝ), price_child_ticket * C + price_adult_ticket * (total_tickets_sold - C) = total_revenue ∧ C = 400 :=
by
  sorry

end NUMINAMATH_GPT_child_tickets_sold_l650_65019


namespace NUMINAMATH_GPT_contrapositive_proof_l650_65068

variable {p q : Prop}

theorem contrapositive_proof : (p → q) ↔ (¬q → ¬p) :=
  by sorry

end NUMINAMATH_GPT_contrapositive_proof_l650_65068


namespace NUMINAMATH_GPT_different_colors_probability_l650_65091

-- Definitions of the chips in the bag
def purple_chips := 7
def green_chips := 6
def orange_chips := 5
def total_chips := purple_chips + green_chips + orange_chips

-- Calculating probabilities for drawing chips of different colors and ensuring the final probability of different colors is correct
def probability_different_colors : ℚ :=
  let P := purple_chips
  let G := green_chips
  let O := orange_chips
  let T := total_chips
  (P / T) * ((G + O) / T) + (G / T) * ((P + O) / T) + (O / T) * ((P + G) / T)

theorem different_colors_probability : probability_different_colors = (107 / 162) := by
  sorry

end NUMINAMATH_GPT_different_colors_probability_l650_65091


namespace NUMINAMATH_GPT_castor_chess_players_l650_65057

theorem castor_chess_players (total_players : ℕ) (never_lost_to_ai : ℕ)
  (h1 : total_players = 40) (h2 : never_lost_to_ai = total_players / 4) :
  (total_players - never_lost_to_ai) = 30 :=
by
  sorry

end NUMINAMATH_GPT_castor_chess_players_l650_65057


namespace NUMINAMATH_GPT_smallest_number_satisfying_conditions_l650_65075

theorem smallest_number_satisfying_conditions :
  ∃ b : ℕ, b ≡ 3 [MOD 5] ∧ b ≡ 2 [MOD 4] ∧ b ≡ 2 [MOD 6] ∧ b = 38 := 
by
  sorry

end NUMINAMATH_GPT_smallest_number_satisfying_conditions_l650_65075


namespace NUMINAMATH_GPT_mike_toys_l650_65034

theorem mike_toys (M A T : ℕ) 
  (h1 : A = 4 * M) 
  (h2 : T = A + 2)
  (h3 : M + A + T = 56) 
  : M = 6 := 
by 
  sorry

end NUMINAMATH_GPT_mike_toys_l650_65034


namespace NUMINAMATH_GPT_roots_of_polynomial_l650_65005

theorem roots_of_polynomial : 
  (∀ x : ℝ, (x^3 - 6*x^2 + 11*x - 6) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l650_65005


namespace NUMINAMATH_GPT_total_people_on_bus_l650_65064

def students_left := 42
def students_right := 38
def students_back := 5
def students_aisle := 15
def teachers := 2
def bus_driver := 1

theorem total_people_on_bus : students_left + students_right + students_back + students_aisle + teachers + bus_driver = 103 :=
by
  sorry

end NUMINAMATH_GPT_total_people_on_bus_l650_65064


namespace NUMINAMATH_GPT_solution_set_of_inequality_l650_65087

def f : Int → Int
| -1 => -1
| 0 => -1
| 1 => 1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

def g : Int → Int
| -1 => 1
| 0 => 1
| 1 => -1
| _ => 0 -- Assuming this default as table provided values only for -1, 0, 1

theorem solution_set_of_inequality :
  {x | f (g x) > 0} = { -1, 0 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l650_65087


namespace NUMINAMATH_GPT_isabel_uploaded_pictures_l650_65032

theorem isabel_uploaded_pictures :
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  total_pictures = 25 :=
by
  let album1 := 10
  let total_other_pictures := 5 * 3
  let total_pictures := album1 + total_other_pictures
  show total_pictures = 25
  sorry

end NUMINAMATH_GPT_isabel_uploaded_pictures_l650_65032


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l650_65004

theorem necessary_and_sufficient_condition (t : ℝ) :
  ((t + 1) * (1 - |t|) > 0) ↔ (t < 1 ∧ t ≠ -1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l650_65004


namespace NUMINAMATH_GPT_find_n_l650_65095

-- Defining the conditions given in the problem
def condition_eq (n : ℝ) : Prop :=
  10 * 1.8 - (n * 1.5 / 0.3) = 50

-- Stating the goal: Prove that the number n is -6.4
theorem find_n : condition_eq (-6.4) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_find_n_l650_65095


namespace NUMINAMATH_GPT_max_sum_of_circle_eq_eight_l650_65066

noncomputable def max_sum_of_integer_solutions (r : ℕ) : ℕ :=
  if r = 6 then 8 else 0

theorem max_sum_of_circle_eq_eight 
  (h1 : ∃ (x y : ℤ), (x - 1)^2 + (y - 1)^2 = 36 ∧ (r : ℕ) = 6) :
  max_sum_of_integer_solutions r = 8 := 
by
  sorry

end NUMINAMATH_GPT_max_sum_of_circle_eq_eight_l650_65066


namespace NUMINAMATH_GPT_sum_of_diagonals_l650_65018

noncomputable def length_AB : ℝ := 31
noncomputable def length_sides : ℝ := 81

def hexagon_inscribed_in_circle (A B C D E F : Type) : Prop :=
-- Assuming A, B, C, D, E, F are suitable points on a circle
-- Definitions to be added as per detailed proof needs
sorry

theorem sum_of_diagonals (A B C D E F : Type) :
    hexagon_inscribed_in_circle A B C D E F →
    (length_AB + length_sides + length_sides + length_sides + length_sides + length_sides = 384) := 
by
  sorry

end NUMINAMATH_GPT_sum_of_diagonals_l650_65018


namespace NUMINAMATH_GPT_time_to_fill_partial_bucket_l650_65059

-- Definitions for the conditions
def time_to_fill_full_bucket : ℝ := 135
def r := 2 / 3

-- The time to fill 2/3 of the bucket should be proven as 90
theorem time_to_fill_partial_bucket : time_to_fill_full_bucket * r = 90 := 
by 
  -- Prove that 90 is the correct time to fill two-thirds of the bucket
  sorry

end NUMINAMATH_GPT_time_to_fill_partial_bucket_l650_65059


namespace NUMINAMATH_GPT_find_fourth_digit_l650_65083

theorem find_fourth_digit (a b c d : ℕ) (h : 0 ≤ a ∧ a < 8 ∧ 0 ≤ b ∧ b < 8 ∧ 0 ≤ c ∧ c < 8 ∧ 0 ≤ d ∧ d < 8)
  (h_eq : 511 * a + 54 * b - 92 * c - 999 * d = 0) : d = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_fourth_digit_l650_65083


namespace NUMINAMATH_GPT_system_solution_l650_65093

theorem system_solution (x y : ℤ) (h1 : x + y = 1) (h2 : 2*x + y = 5) : x = 4 ∧ y = -3 :=
by {
  sorry
}

end NUMINAMATH_GPT_system_solution_l650_65093


namespace NUMINAMATH_GPT_simplify_333_div_9999_mul_99_l650_65065

theorem simplify_333_div_9999_mul_99 :
  (333 / 9999) * 99 = 37 / 101 :=
by
  -- Sorry for skipping proof
  sorry

end NUMINAMATH_GPT_simplify_333_div_9999_mul_99_l650_65065


namespace NUMINAMATH_GPT_spending_on_other_items_is_30_percent_l650_65020

-- Define the total amount Jill spent excluding taxes
variable (T : ℝ)

-- Define the amounts spent on clothing, food, and other items as percentages of T
def clothing_spending : ℝ := 0.50 * T
def food_spending : ℝ := 0.20 * T
def other_items_spending (x : ℝ) : ℝ := x * T

-- Define the tax rates
def clothing_tax_rate : ℝ := 0.04
def food_tax_rate : ℝ := 0.0
def other_items_tax_rate : ℝ := 0.10

-- Define the taxes paid on each category
def clothing_tax : ℝ := clothing_tax_rate * clothing_spending T
def food_tax : ℝ := food_tax_rate * food_spending T
def other_items_tax (x : ℝ) : ℝ := other_items_tax_rate * other_items_spending T x

-- Define the total tax paid as a percentage of the total amount spent excluding taxes
def total_tax_paid : ℝ := 0.05 * T

-- The main theorem stating that the percentage of the amount spent on other items is 30%
theorem spending_on_other_items_is_30_percent (x : ℝ) (h : total_tax_paid T = clothing_tax T + other_items_tax T x) :
  x = 0.30 :=
sorry

end NUMINAMATH_GPT_spending_on_other_items_is_30_percent_l650_65020


namespace NUMINAMATH_GPT_loss_percentage_l650_65008

theorem loss_percentage (CP SP : ℝ) (h_CP : CP = 1300) (h_SP : SP = 1040) :
  ((CP - SP) / CP) * 100 = 20 :=
by
  sorry

end NUMINAMATH_GPT_loss_percentage_l650_65008


namespace NUMINAMATH_GPT_johns_weekly_earnings_percentage_increase_l650_65014

theorem johns_weekly_earnings_percentage_increase (initial final : ℝ) :
  initial = 30 →
  final = 50 →
  ((final - initial) / initial) * 100 = 66.67 :=
by
  intros h_initial h_final
  rw [h_initial, h_final]
  norm_num
  sorry

end NUMINAMATH_GPT_johns_weekly_earnings_percentage_increase_l650_65014


namespace NUMINAMATH_GPT_find_p_over_q_at_neg1_l650_65055

noncomputable def p (x : ℝ) : ℝ := (-27 / 8) * x
noncomputable def q (x : ℝ) : ℝ := (x + 5) * (x - 1)

theorem find_p_over_q_at_neg1 : p (-1) / q (-1) = 27 / 64 := by
  -- Skipping the proof
  sorry

end NUMINAMATH_GPT_find_p_over_q_at_neg1_l650_65055


namespace NUMINAMATH_GPT_circle_condition_k_l650_65072

theorem circle_condition_k (k : ℝ) : 
  (∃ (h : ℝ), (x^2 + y^2 - 2*x + 6*y + k = 0)) → k < 10 :=
by
  sorry

end NUMINAMATH_GPT_circle_condition_k_l650_65072


namespace NUMINAMATH_GPT_min_points_in_symmetric_set_l650_65079

theorem min_points_in_symmetric_set (T : Set (ℝ × ℝ)) (h1 : ∀ {a b : ℝ}, (a, b) ∈ T → (a, -b) ∈ T)
                                      (h2 : ∀ {a b : ℝ}, (a, b) ∈ T → (-a, b) ∈ T)
                                      (h3 : ∀ {a b : ℝ}, (a, b) ∈ T → (-b, -a) ∈ T)
                                      (h4 : (1, 4) ∈ T) : 
    ∃ (S : Finset (ℝ × ℝ)), 
          (∀ p ∈ S, p ∈ T) ∧
          (∀ q ∈ T, ∃ p ∈ S, q = (p.1, p.2) ∨ q = (p.1, -p.2) ∨ q = (-p.1, p.2) ∨ q = (-p.1, -p.2) ∨ q = (-p.2, -p.1) ∨ q = (-p.2, p.1) ∨ q = (p.2, p.1) ∨ q = (p.2, -p.1)) ∧
          S.card = 8 := sorry

end NUMINAMATH_GPT_min_points_in_symmetric_set_l650_65079


namespace NUMINAMATH_GPT_maximal_value_6tuple_l650_65092

theorem maximal_value_6tuple :
  ∀ (a b c d e f : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧ e ≥ 0 ∧ f ≥ 0 ∧ 
  a + b + c + d + e + f = 6 → 
  a * b * c + b * c * d + c * d * e + d * e * f + e * f * a + f * a * b ≤ 8 ∧ 
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 2 ∧
  ((a, b, c, d, e, f) = (0, 0, t, 2, 2, 2 - t) ∨ 
   (a, b, c, d, e, f) = (0, t, 2, 2 - t, 0, 0) ∨ 
   (a, b, c, d, e, f) = (t, 2, 2 - t, 0, 0, 0) ∨ 
   (a, b, c, d, e, f) = (2, 2 - t, 0, 0, 0, t) ∨
   (a, b, c, d, e, f) = (2 - t, 0, 0, 0, t, 2) ∨
   (a, b, c, d, e, f) = (0, 0, 0, t, 2, 2 - t))) := 
sorry

end NUMINAMATH_GPT_maximal_value_6tuple_l650_65092


namespace NUMINAMATH_GPT_julia_tuesday_l650_65051

variable (M : ℕ) -- The number of kids Julia played with on Monday
variable (T : ℕ) -- The number of kids Julia played with on Tuesday

-- Conditions
def condition1 : Prop := M = T + 8
def condition2 : Prop := M = 22

-- Theorem to prove
theorem julia_tuesday : condition1 M T → condition2 M → T = 14 := by
  sorry

end NUMINAMATH_GPT_julia_tuesday_l650_65051


namespace NUMINAMATH_GPT_evaluate_expression_l650_65037

theorem evaluate_expression : (24^36 / 72^18) = 8^18 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l650_65037


namespace NUMINAMATH_GPT_polynomial_root_interval_l650_65049

open Real

theorem polynomial_root_interval (b : ℝ) (x : ℝ) :
  (x^4 + b*x^3 + x^2 + b*x - 1 = 0) → (b ≤ -2 * sqrt 3 ∨ b ≥ 0) :=
sorry

end NUMINAMATH_GPT_polynomial_root_interval_l650_65049


namespace NUMINAMATH_GPT_polynomial_divisibility_p_q_l650_65061

theorem polynomial_divisibility_p_q (p' q' : ℝ) :
  (∀ x, x^5 - x^4 + x^3 - p' * x^2 + q' * x - 6 = 0 → (x = -1 ∨ x = 2)) →
  p' = 0 ∧ q' = -9 :=
by sorry

end NUMINAMATH_GPT_polynomial_divisibility_p_q_l650_65061


namespace NUMINAMATH_GPT_ratio_of_hair_lengths_l650_65058

theorem ratio_of_hair_lengths 
  (logan_hair : ℕ)
  (emily_hair : ℕ)
  (kate_hair : ℕ)
  (h1 : logan_hair = 20)
  (h2 : emily_hair = logan_hair + 6)
  (h3 : kate_hair = 7)
  : kate_hair / emily_hair = 7 / 26 :=
by sorry

end NUMINAMATH_GPT_ratio_of_hair_lengths_l650_65058


namespace NUMINAMATH_GPT_pounds_of_beef_l650_65009

theorem pounds_of_beef (meals_price : ℝ) (total_sales : ℝ) (meat_per_meal : ℝ) (relationship : ℝ) (total_meat_used : ℝ) (beef_pounds : ℝ) :
  (total_sales = 400) → (meals_price = 20) → (meat_per_meal = 1.5) → (relationship = 0.5) → (20 * meals_price = total_sales) → (total_meat_used = 30) →
  (beef_pounds + beef_pounds * relationship = total_meat_used) → beef_pounds = 20 :=
by
  intros
  sorry

end NUMINAMATH_GPT_pounds_of_beef_l650_65009


namespace NUMINAMATH_GPT_max_area_equilateral_triangle_in_rectangle_l650_65043

-- Define the problem parameters
def rect_width : ℝ := 12
def rect_height : ℝ := 15

-- State the theorem to be proved
theorem max_area_equilateral_triangle_in_rectangle 
  (width height : ℝ) (h_width : width = rect_width) (h_height : height = rect_height) :
  ∃ area : ℝ, area = 369 * Real.sqrt 3 - 540 := 
sorry

end NUMINAMATH_GPT_max_area_equilateral_triangle_in_rectangle_l650_65043


namespace NUMINAMATH_GPT_initial_number_of_men_l650_65071

theorem initial_number_of_men (n : ℕ) (A : ℕ)
  (h1 : 2 * n = 16)
  (h2 : 60 - 44 = 16)
  (h3 : 60 = 2 * 30)
  (h4 : 44 = 21 + 23) :
  n = 8 :=
by
  sorry

end NUMINAMATH_GPT_initial_number_of_men_l650_65071


namespace NUMINAMATH_GPT_place_numbers_in_table_l650_65085

theorem place_numbers_in_table (nums : Fin 100 → ℝ) (h_distinct : Function.Injective nums) :
  ∃ (table : Fin 10 → Fin 10 → ℝ),
    (∀ i j, table i j = nums ⟨10 * i + j, sorry⟩) ∧
    (∀ i j k l, (i, j) ≠ (k, l) → (i = k ∧ (j = l + 1 ∨ j = l - 1) ∨ j = l ∧ (i = k + 1 ∨ i = k - 1)) →
      |table i j - table k l| ≠ 1) := sorry  -- Proof omitted

end NUMINAMATH_GPT_place_numbers_in_table_l650_65085


namespace NUMINAMATH_GPT_flour_price_increase_l650_65097

theorem flour_price_increase (x : ℝ) (hx : x > 0) :
  (9600 / (1.5 * x) - 6000 / x = 0.4) :=
by 
  sorry

end NUMINAMATH_GPT_flour_price_increase_l650_65097


namespace NUMINAMATH_GPT_no_sol_for_frac_eq_l650_65070

theorem no_sol_for_frac_eq (x y : ℕ) (h : x > 1) : ¬ (y^5 + 1 = (x^7 - 1) / (x - 1)) :=
sorry

end NUMINAMATH_GPT_no_sol_for_frac_eq_l650_65070


namespace NUMINAMATH_GPT_compute_expression_l650_65031

theorem compute_expression : (88 * 707 - 38 * 707) / 1414 = 25 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l650_65031


namespace NUMINAMATH_GPT_range_of_a_l650_65063

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (hf : ∀ x : ℝ, f x = a*x^3 + Real.log x) :
  (∃ x : ℝ, x > 0 ∧ (deriv f x = 0)) → a < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l650_65063


namespace NUMINAMATH_GPT_ranking_of_anna_bella_carol_l650_65022

-- Define three people and their scores
variables (Anna Bella Carol : ℕ)

-- Define conditions based on problem statements
axiom Anna_not_highest : ∃ x : ℕ, x > Anna
axiom Bella_not_lowest : ∃ x : ℕ, x < Bella
axiom Bella_higher_than_Carol : Bella > Carol

-- The theorem to be proven
theorem ranking_of_anna_bella_carol (h : Anna < Bella ∧ Carol < Anna) :
  (Bella > Anna ∧ Anna > Carol) :=
by sorry

end NUMINAMATH_GPT_ranking_of_anna_bella_carol_l650_65022


namespace NUMINAMATH_GPT_max_value_l650_65060

noncomputable def max_fraction (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : ℝ :=
  (x + y + z)^2 / (x^2 + y^2 + z^2)

theorem max_value (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  max_fraction x y z hx hy hz ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_value_l650_65060


namespace NUMINAMATH_GPT_total_cost_with_discount_and_tax_l650_65088

theorem total_cost_with_discount_and_tax
  (sandwich_cost : ℝ := 2.44)
  (soda_cost : ℝ := 0.87)
  (num_sandwiches : ℕ := 2)
  (num_sodas : ℕ := 4)
  (discount : ℝ := 0.15)
  (tax_rate : ℝ := 0.09) : 
  (num_sandwiches * sandwich_cost * (1 - discount) + num_sodas * soda_cost) * (1 + tax_rate) = 8.32 :=
by
  sorry

end NUMINAMATH_GPT_total_cost_with_discount_and_tax_l650_65088


namespace NUMINAMATH_GPT_inequality_holds_for_positive_vars_l650_65046

theorem inequality_holds_for_positive_vars (x y : ℝ) (hx : x > 0) (hy : y > 0) : 
    x^2 + y^2 + 1 ≥ x * y + x + y :=
sorry

end NUMINAMATH_GPT_inequality_holds_for_positive_vars_l650_65046


namespace NUMINAMATH_GPT_avg_speed_including_stoppages_l650_65012

theorem avg_speed_including_stoppages (speed_without_stoppages : ℝ) (stoppage_time_per_hour : ℝ) 
  (h₁ : speed_without_stoppages = 60) (h₂ : stoppage_time_per_hour = 0.5) : 
  (speed_without_stoppages * (1 - stoppage_time_per_hour)) / 1 = 30 := 
  by 
  sorry

end NUMINAMATH_GPT_avg_speed_including_stoppages_l650_65012


namespace NUMINAMATH_GPT_least_common_multiple_l650_65047

open Int

theorem least_common_multiple {a b c : ℕ} 
  (h1 : Nat.lcm a b = 18) 
  (h2 : Nat.lcm b c = 20) : Nat.lcm a c = 90 := 
sorry

end NUMINAMATH_GPT_least_common_multiple_l650_65047


namespace NUMINAMATH_GPT_ab_leq_one_l650_65045

theorem ab_leq_one (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : a + b = 2) : ab ≤ 1 := by
  sorry

end NUMINAMATH_GPT_ab_leq_one_l650_65045


namespace NUMINAMATH_GPT_packing_heights_difference_l650_65013

-- Definitions based on conditions
def diameter := 8   -- Each pipe has a diameter of 8 cm
def num_pipes := 160 -- Each crate contains 160 pipes

-- Heights of the crates based on the given packing methods
def height_crate_A := 128 -- Calculated height for Crate A

noncomputable def height_crate_B := 8 + 60 * Real.sqrt 3 -- Calculated height for Crate B

-- Positive difference in the total heights of the two packings
noncomputable def delta_height := height_crate_A - height_crate_B

-- The goal to prove
theorem packing_heights_difference :
  delta_height = 120 - 60 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_packing_heights_difference_l650_65013


namespace NUMINAMATH_GPT_elvie_age_l650_65024

variable (E : ℕ) (A : ℕ)

theorem elvie_age (hA : A = 11) (h : E + A + (E * A) = 131) : E = 10 :=
by
  sorry

end NUMINAMATH_GPT_elvie_age_l650_65024


namespace NUMINAMATH_GPT_cafeteria_students_count_l650_65086

def total_students : ℕ := 90

def initial_in_cafeteria : ℕ := total_students * 2 / 3

def initial_outside : ℕ := total_students / 3

def ran_inside : ℕ := initial_outside / 3

def ran_outside : ℕ := 3

def net_change_in_cafeteria : ℕ := ran_inside - ran_outside

def final_in_cafeteria : ℕ := initial_in_cafeteria + net_change_in_cafeteria

theorem cafeteria_students_count : final_in_cafeteria = 67 := 
by
  sorry

end NUMINAMATH_GPT_cafeteria_students_count_l650_65086


namespace NUMINAMATH_GPT_range_of_x_l650_65041

noncomputable def f (x : ℝ) : ℝ := Real.exp x + x^3

theorem range_of_x (x : ℝ) (h : f (x^2) < f (3*x - 2)) : 1 < x ∧ x < 2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l650_65041


namespace NUMINAMATH_GPT_range_of_a_l650_65054

theorem range_of_a (a : ℝ) : (¬ ∃ x : ℝ, x^2 + (a - 1) * x + 1 ≤ 0) → -1 < a ∧ a < 3 :=
sorry

end NUMINAMATH_GPT_range_of_a_l650_65054


namespace NUMINAMATH_GPT_correct_statement_l650_65052

def angle_terminal_side (a b : ℝ) : Prop :=
∃ k : ℤ, a = b + k * 360

def obtuse_angle (θ : ℝ) : Prop :=
90 < θ ∧ θ < 180

def third_quadrant_angle (θ : ℝ) : Prop :=
180 < θ ∧ θ < 270

def first_quadrant_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

def acute_angle (θ : ℝ) : Prop :=
0 < θ ∧ θ < 90

theorem correct_statement :
  ¬∀ a b, angle_terminal_side a b → a = b ∧
  ¬∀ θ, obtuse_angle θ → θ < θ - 360 ∧
  ¬∀ θ, first_quadrant_angle θ → acute_angle θ ∧
  ∀ θ, acute_angle θ → first_quadrant_angle θ :=
by
  sorry

end NUMINAMATH_GPT_correct_statement_l650_65052
