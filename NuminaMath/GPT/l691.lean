import Mathlib

namespace box_surface_area_l691_69125

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l691_69125


namespace fraction_of_A_eq_l691_69139

noncomputable def fraction_A (A B C T : ℕ) : ℚ :=
  A / (T - A)

theorem fraction_of_A_eq :
  ∃ (A B C T : ℕ), T = 360 ∧ A = B + 10 ∧ B = 2 * (A + C) / 7 ∧ T = A + B + C ∧ fraction_A A B C T = 1 / 3 :=
by
  sorry

end fraction_of_A_eq_l691_69139


namespace application_methods_count_l691_69103

theorem application_methods_count :
  let S := 5; -- number of students
  let U := 3; -- number of universities
  let unrestricted := U^S; -- unrestricted distribution
  let restricted_one_university_empty := (U - 1)^S * U; -- one university empty
  let restricted_two_universities_empty := 0; -- invalid scenario
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty;
  valid_methods - U = 144 :=
by
  let S := 5
  let U := 3
  let unrestricted := U^S
  let restricted_one_university_empty := (U - 1)^S * U
  let restricted_two_universities_empty := 0
  let valid_methods := unrestricted - restricted_one_university_empty - restricted_two_universities_empty
  have : valid_methods - U = 144 := by sorry
  exact this

end application_methods_count_l691_69103


namespace eliana_total_steps_l691_69122

-- Define the conditions given in the problem
def steps_first_day_exercise : Nat := 200
def steps_first_day_additional : Nat := 300
def steps_first_day : Nat := steps_first_day_exercise + steps_first_day_additional

def steps_second_day : Nat := 2 * steps_first_day
def steps_additional_on_third_day : Nat := 100
def steps_third_day : Nat := steps_second_day + steps_additional_on_third_day

-- Mathematical proof problem proving that the total number of steps is 2600
theorem eliana_total_steps : steps_first_day + steps_second_day + steps_third_day = 2600 := 
by
  sorry

end eliana_total_steps_l691_69122


namespace area_square_ratio_l691_69152

theorem area_square_ratio (r : ℝ) (h1 : r > 0)
  (s1 : ℝ) (hs1 : s1^2 = r^2)
  (s2 : ℝ) (hs2 : s2^2 = (4/5) * r^2) : 
  (s1^2 / s2^2) = (5 / 4) :=
by 
  sorry

end area_square_ratio_l691_69152


namespace radius_of_smaller_base_l691_69111

theorem radius_of_smaller_base (C1 C2 : ℝ) (r : ℝ) (l : ℝ) (A : ℝ) 
    (h1 : C2 = 3 * C1) 
    (h2 : l = 3) 
    (h3 : A = 84 * Real.pi) 
    (h4 : C1 = 2 * Real.pi * r) 
    (h5 : C2 = 2 * Real.pi * (3 * r)) :
    r = 7 := 
by
  -- proof steps here
  sorry

end radius_of_smaller_base_l691_69111


namespace desired_percentage_alcohol_l691_69123

noncomputable def original_volume : ℝ := 6
noncomputable def original_percentage : ℝ := 0.40
noncomputable def added_alcohol : ℝ := 1.2
noncomputable def final_solution_volume : ℝ := original_volume + added_alcohol
noncomputable def final_alcohol_volume : ℝ := (original_percentage * original_volume) + added_alcohol
noncomputable def desired_percentage : ℝ := (final_alcohol_volume / final_solution_volume) * 100

theorem desired_percentage_alcohol :
  desired_percentage = 50 := by
  sorry

end desired_percentage_alcohol_l691_69123


namespace sum_of_first_20_terms_l691_69180

variable {a : ℕ → ℕ}

-- Conditions given in the problem
axiom seq_property : ∀ n, a n + 2 * a (n + 1) = 3 * n + 2
axiom arithmetic_sequence : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- Theorem to be proved
theorem sum_of_first_20_terms (a : ℕ → ℕ) (S20 := (Finset.range 20).sum a) :
  S20 = 210 :=
  sorry

end sum_of_first_20_terms_l691_69180


namespace find_b_plus_c_l691_69155

variable {a b c d : ℝ}

theorem find_b_plus_c
  (h1 : a + b = 4)
  (h2 : c + d = 3)
  (h3 : a + d = 2) :
  b + c = 5 := 
  by
  sorry

end find_b_plus_c_l691_69155


namespace length_of_field_l691_69141

variable (w : ℕ)   -- Width of the rectangular field
variable (l : ℕ)   -- Length of the rectangular field
variable (pond_side : ℕ)  -- Side length of the square pond
variable (pond_area field_area : ℕ)  -- Areas of the pond and field
variable (cond1 : l = 2 * w)  -- Condition 1: Length is double the width
variable (cond2 : pond_side = 4)  -- Condition 2: Side of the pond is 4 meters
variable (cond3 : pond_area = pond_side * pond_side)  -- Condition 3: Area of square pond
variable (cond4 : pond_area = (1 / 8) * field_area)  -- Condition 4: Area of pond is 1/8 of the area of the field

theorem length_of_field :
  pond_area = pond_side * pond_side →
  pond_area = (1 / 8) * (l * w) →
  l = 2 * w →
  w = 8 →
  l = 16 :=
by
  intro h1 h2 h3 h4
  sorry

end length_of_field_l691_69141


namespace cube_splitting_odd_numbers_l691_69172

theorem cube_splitting_odd_numbers (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 2 * k + 1 = 333) : m = 18 :=
sorry

end cube_splitting_odd_numbers_l691_69172


namespace Tiffany_total_score_l691_69129

def points_per_treasure_type : Type := ℕ × ℕ × ℕ
def treasures_per_level : Type := ℕ × ℕ × ℕ

def points (bronze silver gold : ℕ) : ℕ :=
  bronze * 6 + silver * 15 + gold * 30

def treasures_level1 : treasures_per_level := (2, 3, 1)
def treasures_level2 : treasures_per_level := (3, 1, 2)
def treasures_level3 : treasures_per_level := (5, 2, 1)

def total_points (l1 l2 l3 : treasures_per_level) : ℕ :=
  let (b1, s1, g1) := l1
  let (b2, s2, g2) := l2
  let (b3, s3, g3) := l3
  points b1 s1 g1 + points b2 s2 g2 + points b3 s3 g3

theorem Tiffany_total_score :
  total_points treasures_level1 treasures_level2 treasures_level3 = 270 :=
by
  sorry

end Tiffany_total_score_l691_69129


namespace women_population_percentage_l691_69175

theorem women_population_percentage (W M : ℕ) (h : M = 2 * W) : (W : ℚ) / (M : ℚ) = (50 : ℚ) / 100 :=
by
  -- Proof omitted
  sorry

end women_population_percentage_l691_69175


namespace count_ns_divisible_by_5_l691_69137

open Nat

theorem count_ns_divisible_by_5 : 
  let f (n : ℕ) := 2 * n^5 + 2 * n^4 + 3 * n^2 + 3 
  ∃ (N : ℕ), N = 19 ∧ 
  (∀ (n : ℕ), 2 ≤ n ∧ n ≤ 100 → f n % 5 = 0 → 
  (∃ (m : ℕ), 1 ≤ m ∧ m ≤ 19 ∧ n = 5 * m + 1)) :=
by
  sorry

end count_ns_divisible_by_5_l691_69137


namespace percentage_respondents_liked_B_l691_69100

variables (X Y : ℝ)
variables (likedA likedB likedBoth likedNeither : ℝ)
variables (totalRespondents : ℕ)

-- Conditions from the problem
def liked_conditions : Prop :=
    totalRespondents ≥ 100 ∧ 
    likedA = X ∧ 
    likedB = Y ∧ 
    likedBoth = 23 ∧ 
    likedNeither = 23

-- Proof statement
theorem percentage_respondents_liked_B (h : liked_conditions X Y likedA likedB likedBoth likedNeither totalRespondents) :
  Y = 100 - X :=
sorry

end percentage_respondents_liked_B_l691_69100


namespace swapped_coefficients_have_roots_l691_69128

theorem swapped_coefficients_have_roots 
  (a b c p q r : ℝ)
  (h1 : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0))
  (h2 : ∀ x : ℝ, ¬ (p * x^2 + q * x + r = 0))
  (h3 : b^2 < 4 * p * c)
  (h4 : q^2 < 4 * a * r) :
  ∃ x : ℝ, a * x^2 + q * x + c = 0 ∧ ∃ y : ℝ, p * y^2 + b * y + r = 0 :=
by
  sorry

end swapped_coefficients_have_roots_l691_69128


namespace cuberoot_inequality_l691_69135

theorem cuberoot_inequality (a b : ℝ) : a < b → (∃ x y : ℝ, x^3 = a ∧ y^3 = b ∧ (x = y ∨ x > y)) := 
sorry

end cuberoot_inequality_l691_69135


namespace proposition_4_l691_69196

variables {Line Plane : Type}
variables {a b : Line} {α β : Plane}

-- Definitions of parallel and perpendicular relationships
class Parallel (l : Line) (p : Plane) : Prop
class Perpendicular (l : Line) (p : Plane) : Prop
class Contains (p : Plane) (l : Line) : Prop

theorem proposition_4
  (h1: Perpendicular a β)
  (h2: Parallel a b)
  (h3: Contains α b) : Perpendicular α β :=
sorry

end proposition_4_l691_69196


namespace range_of_a_l691_69126

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ (1 ≤ a ∧ a < 5) := by
  sorry

end range_of_a_l691_69126


namespace collinear_condition_perpendicular_condition_l691_69145

namespace Vectors

-- Definitions for vectors a and b
def a : ℝ × ℝ := (4, -2)
def b (x : ℝ) : ℝ × ℝ := (x, 1)

-- Collinear condition
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 = k * b.1 ∧ a.2 = k * b.2

-- Perpendicular condition
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Proof statement for collinear condition
theorem collinear_condition (x : ℝ) (h : collinear a (b x)) : x = -2 := sorry

-- Proof statement for perpendicular condition
theorem perpendicular_condition (x : ℝ) (h : perpendicular a (b x)) : x = 1 / 2 := sorry

end Vectors

end collinear_condition_perpendicular_condition_l691_69145


namespace inequality_proof_l691_69113

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + b^2 + c^2 + a * b * c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a * b * c ≤ 4 := by
  sorry

end inequality_proof_l691_69113


namespace hexagon_cyclic_identity_l691_69185

variables (a a' b b' c c' a₁ b₁ c₁ : ℝ)

theorem hexagon_cyclic_identity :
  a₁ * b₁ * c₁ = a * b * c + a' * b' * c' + a * a' * a₁ + b * b' * b₁ + c * c' * c₁ :=
by
  sorry

end hexagon_cyclic_identity_l691_69185


namespace find_number_l691_69107

theorem find_number (x : ℝ) : 8050 * x = 80.5 → x = 0.01 :=
by
  sorry

end find_number_l691_69107


namespace monotonically_decreasing_iff_a_lt_1_l691_69138

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2 * x

theorem monotonically_decreasing_iff_a_lt_1 {a : ℝ} (h : ∀ x > 0, (deriv (f a) x) < 0) : a < 1 :=
sorry

end monotonically_decreasing_iff_a_lt_1_l691_69138


namespace prime_product_mod_32_l691_69158

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l691_69158


namespace determine_b_for_inverse_function_l691_69156

theorem determine_b_for_inverse_function (b : ℝ) :
  (∀ x, (2 - 3 * (1 / (2 * x + b))) / (3 * (1 / (2 * x + b))) = x) ↔ b = 3 / 2 := by
  sorry

end determine_b_for_inverse_function_l691_69156


namespace enclosed_area_abs_eq_54_l691_69140

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end enclosed_area_abs_eq_54_l691_69140


namespace sum_interior_angles_equal_diagonals_l691_69181

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l691_69181


namespace oranges_per_box_l691_69191

theorem oranges_per_box
  (total_oranges : ℕ)
  (boxes : ℕ)
  (h1 : total_oranges = 35)
  (h2 : boxes = 7) :
  total_oranges / boxes = 5 := by
  sorry

end oranges_per_box_l691_69191


namespace sqrt_defined_iff_le_l691_69150

theorem sqrt_defined_iff_le (x : ℝ) : (∃ y : ℝ, y^2 = 4 - x) ↔ (x ≤ 4) :=
by
  sorry

end sqrt_defined_iff_le_l691_69150


namespace cookie_radius_and_area_l691_69124

def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2 * x + 4 * y

theorem cookie_radius_and_area :
  (∃ r : ℝ, r = Real.sqrt 13) ∧ (∃ A : ℝ, A = 13 * Real.pi) :=
by
  sorry

end cookie_radius_and_area_l691_69124


namespace problem_l691_69178

theorem problem (p q r : ℝ)
    (h1 : p * 1^2 + q * 1 + r = 5)
    (h2 : p * 2^2 + q * 2 + r = 3) :
  p + q + 2 * r = 10 := 
sorry

end problem_l691_69178


namespace rationalize_denominator_l691_69151

theorem rationalize_denominator :
  let A := 9
  let B := 7
  let C := -18
  let D := 0
  let S := 2
  let F := 111
  (A + B + C + D + S + F = 111) ∧ 
  (
    (1 / (Real.sqrt 5 + Real.sqrt 6 + 2 * Real.sqrt 2)) * 
    ((Real.sqrt 5 + Real.sqrt 6) - 2 * Real.sqrt 2) * 
    (3 - 2 * Real.sqrt 30) / 
    (3^2 - (2 * Real.sqrt 30)^2) = 
    (9 * Real.sqrt 5 + 7 * Real.sqrt 6 - 18 * Real.sqrt 2) / 111
  ) := by
  sorry

end rationalize_denominator_l691_69151


namespace estimated_probability_is_2_div_9_l691_69190

def groups : List (List ℕ) :=
  [[3, 4, 3], [4, 3, 2], [3, 4, 1], [3, 4, 2], [2, 3, 4], [1, 4, 2], [2, 4, 3], [3, 3, 1], [1, 1, 2],
   [3, 4, 2], [2, 4, 1], [2, 4, 4], [4, 3, 1], [2, 3, 3], [2, 1, 4], [3, 4, 4], [1, 4, 2], [1, 3, 4]]

def count_desired_groups (gs : List (List ℕ)) : Nat :=
  gs.foldl (fun acc g =>
    if g.contains 1 ∧ g.contains 2 ∧ g.length ≥ 3 then acc + 1 else acc) 0

theorem estimated_probability_is_2_div_9 :
  (count_desired_groups groups) = 4 →
  4 / 18 = 2 / 9 :=
by
  intro h
  sorry

end estimated_probability_is_2_div_9_l691_69190


namespace simplify_fraction_l691_69114

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l691_69114


namespace least_number_l691_69184

theorem least_number (n : ℕ) (h1 : n % 31 = 3) (h2 : n % 9 = 3) : n = 282 :=
sorry

end least_number_l691_69184


namespace complement_of_A_in_U_l691_69163

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {3, 4, 5}

-- Define the complement of A in U
theorem complement_of_A_in_U : (U \ A) = {1, 2} :=
by
  sorry

end complement_of_A_in_U_l691_69163


namespace work_rate_sum_l691_69119

theorem work_rate_sum (A B : ℝ) (W : ℝ) (h1 : (A + B) * 4 = W) (h2 : A * 8 = W) : (A + B) * 4 = W :=
by
  -- placeholder for actual proof
  sorry

end work_rate_sum_l691_69119


namespace find_f_2011_l691_69199

theorem find_f_2011 (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = f (-x)) 
  (h2 : ∀ x, f (x + 1) * f (x - 1) = 1) 
  (h3 : ∀ x, f x > 0) : 
  f 2011 = 1 := 
sorry

end find_f_2011_l691_69199


namespace eval_expression_l691_69168

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end eval_expression_l691_69168


namespace bus_passengers_l691_69136

variable (P : ℕ) -- P represents the initial number of passengers

theorem bus_passengers (h1 : P + 16 - 17 = 49) : P = 50 :=
by
  sorry

end bus_passengers_l691_69136


namespace alicia_average_speed_correct_l691_69148

/-
Alicia drove 320 miles in 6 hours.
Alicia drove another 420 miles in 7 hours.
Prove Alicia's average speed for the entire journey is 56.92 miles per hour.
-/

def alicia_total_distance : ℕ := 320 + 420
def alicia_total_time : ℕ := 6 + 7
def alicia_average_speed : ℚ := alicia_total_distance / alicia_total_time

theorem alicia_average_speed_correct : alicia_average_speed = 56.92 :=
by
  -- Proof goes here
  sorry

end alicia_average_speed_correct_l691_69148


namespace circle_area_difference_l691_69179

/-- 
Prove that the area of the circle with radius r1 = 30 inches is 675π square inches greater than 
the area of the circle with radius r2 = 15 inches.
-/
theorem circle_area_difference (r1 r2 : ℝ) (h1 : r1 = 30) (h2 : r2 = 15) :
  π * r1^2 - π * r2^2 = 675 * π := 
by {
  -- Placeholders to indicate where the proof would go
  sorry 
}

end circle_area_difference_l691_69179


namespace only_n_eq_1_solution_l691_69164

theorem only_n_eq_1_solution (n : ℕ) (h : n > 0): 
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
by
  sorry

end only_n_eq_1_solution_l691_69164


namespace price_of_individual_rose_l691_69171

-- Definitions based on conditions

def price_of_dozen := 36  -- one dozen roses cost $36
def price_of_two_dozen := 50 -- two dozen roses cost $50
def total_money := 680 -- total available money
def total_roses := 317 -- total number of roses that can be purchased

-- Define the question as a theorem
theorem price_of_individual_rose : 
  ∃ (x : ℕ), (12 * (total_money / price_of_two_dozen) + 
              (total_money % price_of_two_dozen) / price_of_dozen * 12 + 
              (total_money % price_of_two_dozen % price_of_dozen) / x = total_roses) ∧ (x = 6) :=
by
  sorry

end price_of_individual_rose_l691_69171


namespace arithmetic_geometric_mean_l691_69102

theorem arithmetic_geometric_mean (a b : ℝ) (h1 : a + b = 48) (h2 : a * b = 440) : a^2 + b^2 = 1424 := 
by 
  -- Proof goes here
  sorry

end arithmetic_geometric_mean_l691_69102


namespace john_total_spent_l691_69182

def silver_ounces : ℝ := 2.5
def silver_price_per_ounce : ℝ := 25
def gold_ounces : ℝ := 3.5
def gold_price_multiplier : ℝ := 60
def platinum_ounces : ℝ := 4.5
def platinum_price_per_ounce_gbp : ℝ := 80
def palladium_ounces : ℝ := 5.5
def palladium_price_per_ounce_eur : ℝ := 100

def usd_per_gbp_monday : ℝ := 1.3
def usd_per_gbp_friday : ℝ := 1.4
def usd_per_eur_wednesday : ℝ := 1.15
def usd_per_eur_saturday : ℝ := 1.2

def discount_rate : ℝ := 0.05
def tax_rate : ℝ := 0.08

def total_amount_john_spends_usd : ℝ := 
  (silver_ounces * silver_price_per_ounce * (1 - discount_rate)) + 
  (gold_ounces * (gold_price_multiplier * silver_price_per_ounce) * (1 - discount_rate)) + 
  (((platinum_ounces * platinum_price_per_ounce_gbp) * (1 + tax_rate)) * usd_per_gbp_monday) + 
  ((palladium_ounces * palladium_price_per_ounce_eur) * usd_per_eur_wednesday)

theorem john_total_spent : total_amount_john_spends_usd = 6184.815 := by
  sorry

end john_total_spent_l691_69182


namespace lines_are_parallel_and_not_coincident_l691_69194

theorem lines_are_parallel_and_not_coincident (a : ℝ) :
  (a * (a - 1) - 3 * 2 = 0) ∧ (3 * (a - 7) - a * 3 * a ≠ 0) ↔ a = 3 :=
by
  sorry

end lines_are_parallel_and_not_coincident_l691_69194


namespace geometric_sequence_common_ratio_range_l691_69133

theorem geometric_sequence_common_ratio_range (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 < 0) 
  (h2 : ∀ n : ℕ, 0 < n → a n < a (n + 1))
  (hq : ∀ n : ℕ, a (n + 1) = a n * q) :
  0 < q ∧ q < 1 :=
sorry

end geometric_sequence_common_ratio_range_l691_69133


namespace jennie_speed_difference_l691_69106

theorem jennie_speed_difference :
  (∀ (d t1 t2 : ℝ), (d = 200) → (t1 = 5) → (t2 = 4) → (40 = d / t1) → (50 = d / t2) → (50 - 40 = 10)) :=
by
  intros d t1 t2 h_d h_t1 h_t2 h_speed_heavy h_speed_no_traffic
  sorry

end jennie_speed_difference_l691_69106


namespace remainder_of_division_l691_69105

-- Define the dividend and divisor
def dividend : ℕ := 3^303 + 303
def divisor : ℕ := 3^101 + 3^51 + 1

-- State the theorem to be proven
theorem remainder_of_division:
  (dividend % divisor) = 303 := by
  sorry

end remainder_of_division_l691_69105


namespace tray_height_l691_69162

-- Declare the main theorem with necessary given conditions.
theorem tray_height (a b c : ℝ) (side_length : ℝ) (cut_distance : ℝ) (angle : ℝ) : 
  (side_length = 150) →
  (cut_distance = Real.sqrt 50) →
  (angle = 45) →
  a^2 + b^2 = c^2 → -- Condition from Pythagorean theorem
  a = side_length * Real.sqrt 2 / 2 - cut_distance → -- Calculation for half diagonal minus cut distance
  b = (side_length * Real.sqrt 2 / 2 - cut_distance) / 2 → -- Perpendicular from R to the side
  side_length = 150 → -- Ensure consistency of side length
  b^2 + c^2 = side_length^2 → -- Ensure we use another Pythagorean relation
  c = Real.sqrt 7350 → -- Derived c value
  c = Real.sqrt 1470 := -- Simplified form of c.
  sorry

end tray_height_l691_69162


namespace ferry_time_difference_l691_69169

theorem ferry_time_difference :
  ∃ (t : ℕ), (∀ (dP : ℕ) (sP : ℕ) (sQ : ℕ), dP = sP * 3 →
   dP = 24 →
   sP = 8 →
   sQ = sP + 1 →
   t = (dP * 3) / sQ - 3) ∧ t = 5 := 
  sorry

end ferry_time_difference_l691_69169


namespace abc_inequality_l691_69146

theorem abc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_cond : a * b + b * c + c * a = 1) :
  (a + b + c) ≥ Real.sqrt 3 ∧ (a + b + c = Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = Real.sqrt 1 / Real.sqrt 3) :=
by sorry

end abc_inequality_l691_69146


namespace consistency_condition_l691_69104

theorem consistency_condition (x y z a b c d : ℝ)
  (h1 : y + z = a)
  (h2 : x + y = b)
  (h3 : x + z = c)
  (h4 : x + y + z = d) : a + b + c = 2 * d :=
by sorry

end consistency_condition_l691_69104


namespace neg_sin_leq_1_l691_69170

theorem neg_sin_leq_1 :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end neg_sin_leq_1_l691_69170


namespace student_missed_number_l691_69176

theorem student_missed_number (student_sum : ℕ) (n : ℕ) (actual_sum : ℕ) : 
  student_sum = 575 → 
  actual_sum = n * (n + 1) / 2 → 
  n = 34 → 
  actual_sum - student_sum = 20 := 
by 
  sorry

end student_missed_number_l691_69176


namespace has_three_zeros_iff_b_lt_neg3_l691_69147

def f (x b : ℝ) : ℝ := x^3 - b * x^2 - 4

theorem has_three_zeros_iff_b_lt_neg3 (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ f x₃ b = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ↔ b < -3 := 
sorry

end has_three_zeros_iff_b_lt_neg3_l691_69147


namespace gcd_97_power_l691_69192

theorem gcd_97_power (h : Nat.Prime 97) : 
  Nat.gcd (97^7 + 1) (97^7 + 97^3 + 1) = 1 := 
by 
  sorry

end gcd_97_power_l691_69192


namespace minimum_value_expression_l691_69101

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l691_69101


namespace evaluate_expression_l691_69157

variables (x y : ℕ)

theorem evaluate_expression : x = 2 → y = 4 → y * (y - 2 * x + 1) = 4 :=
by
  intro h1 h2
  sorry

end evaluate_expression_l691_69157


namespace max_geq_four_ninths_sum_min_leq_quarter_sum_l691_69109

theorem max_geq_four_ninths_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  max a (max b c) >= 4 / 9 * (a + b + c) :=
by 
  sorry

theorem min_leq_quarter_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  min a (min b c) <= 1 / 4 * (a + b + c) :=
by 
  sorry

end max_geq_four_ninths_sum_min_leq_quarter_sum_l691_69109


namespace push_ups_total_l691_69144

theorem push_ups_total (d z : ℕ) (h1 : d = 51) (h2 : d = z + 49) : d + z = 53 := by
  sorry

end push_ups_total_l691_69144


namespace p_and_q_together_complete_in_10_days_l691_69195

noncomputable def p_time := 50 / 3
noncomputable def q_time := 25
noncomputable def r_time := 50

theorem p_and_q_together_complete_in_10_days 
  (h1 : 1 / p_time = 1 / q_time + 1 / r_time)
  (h2 : r_time = 50)
  (h3 : q_time = 25) :
  (p_time * q_time) / (p_time + q_time) = 10 :=
by
  sorry

end p_and_q_together_complete_in_10_days_l691_69195


namespace fixed_point_for_line_l691_69187

theorem fixed_point_for_line (m : ℝ) : (m * (1 - 1) + (1 - 1) = 0) :=
by
  sorry

end fixed_point_for_line_l691_69187


namespace find_percentage_of_number_l691_69167

theorem find_percentage_of_number (P : ℝ) (N : ℝ) (h1 : P * N = (4 / 5) * N - 21) (h2 : N = 140) : P * 100 = 65 := 
by 
  sorry

end find_percentage_of_number_l691_69167


namespace projections_possibilities_l691_69134

-- Define the conditions: a and b are non-perpendicular skew lines, and α is a plane
variables {a b : Line} (α : Plane)

-- Non-perpendicular skew lines definition (external knowledge required for proper setup if not inbuilt)
def non_perpendicular_skew_lines (a b : Line) : Prop := sorry

-- Projections definition (external knowledge required for proper setup if not inbuilt)
def projections (a : Line) (α : Plane) : Line := sorry

-- The projections result in new conditions
def projected_parallel (a b : Line) (α : Plane) : Prop := sorry
def projected_perpendicular (a b : Line) (α : Plane) : Prop := sorry
def projected_same_line (a b : Line) (α : Plane) : Prop := sorry
def projected_line_and_point (a b : Line) (α : Plane) : Prop := sorry

-- Given the given conditions
variables (ha : non_perpendicular_skew_lines a b)

-- Prove the resultant conditions where the projections satisfy any 3 of the listed possibilities: parallel, perpendicular, line and point.
theorem projections_possibilities :
    (projected_parallel a b α ∨ projected_perpendicular a b α ∨ projected_line_and_point a b α) ∧
    ¬ projected_same_line a b α := sorry

end projections_possibilities_l691_69134


namespace sqrt10_parts_sqrt6_value_sqrt3_opposite_l691_69143

-- Problem 1
theorem sqrt10_parts : 3 < Real.sqrt 10 ∧ Real.sqrt 10 < 4 → (⌊Real.sqrt 10⌋ = 3 ∧ Real.sqrt 10 - 3 = Real.sqrt 10 - ⌊Real.sqrt 10⌋) :=
by
  sorry

-- Problem 2
theorem sqrt6_value (a b : ℝ) : a = Real.sqrt 6 - 2 ∧ b = 3 → (a + b - Real.sqrt 6 = 1) :=
by
  sorry

-- Problem 3
theorem sqrt3_opposite (x y : ℝ) : x = 13 ∧ y = Real.sqrt 3 - 1 → (-(x - y) = Real.sqrt 3 - 14) :=
by
  sorry

end sqrt10_parts_sqrt6_value_sqrt3_opposite_l691_69143


namespace truck_travel_distance_l691_69120

theorem truck_travel_distance
  (miles_traveled : ℕ)
  (gallons_used : ℕ)
  (new_gallons : ℕ)
  (rate : ℕ)
  (distance : ℕ) :
  (miles_traveled = 300) ∧
  (gallons_used = 10) ∧
  (new_gallons = 15) ∧
  (rate = miles_traveled / gallons_used) ∧
  (distance = rate * new_gallons)
  → distance = 450 :=
by
  sorry

end truck_travel_distance_l691_69120


namespace max_f_on_interval_l691_69189

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 + Real.sin x

theorem max_f_on_interval : 
  ∃ x ∈ Set.Icc (2 * Real.pi / 5) (3 * Real.pi / 4), f x = (1 + Real.sqrt 2) / 2 :=
by
  sorry

end max_f_on_interval_l691_69189


namespace find_rate_of_interest_l691_69197

def simple_interest (principal : ℕ) (rate : ℕ) (time : ℕ) : ℕ :=
  principal * rate * time / 100

theorem find_rate_of_interest :
  ∀ (R : ℕ),
  simple_interest 5000 R 2 + simple_interest 3000 R 4 = 2640 → R = 12 :=
by
  intros R h
  sorry

end find_rate_of_interest_l691_69197


namespace range_of_a_l691_69173

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ ((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) →
  (-2:ℝ) ≤ a ∧ a < (6 / 5:ℝ) :=
by
  sorry

end range_of_a_l691_69173


namespace part1_part2_l691_69117

theorem part1 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C) : 
  B = 2 * Real.pi / 3 := 
sorry

theorem part2 
  (a b c : ℝ) 
  (A C : ℝ) 
  (h1 : a + 2 * c = b * Real.cos C + Real.sqrt 3 * b * Real.sin C)
  (h2 : b = 3) : 
  6 < (a + b + c) ∧ (a + b + c) ≤ 3 + 2 * Real.sqrt 3 :=
sorry

end part1_part2_l691_69117


namespace car_speed_l691_69165

/-- 
If a tire rotates at 400 revolutions per minute, and the circumference of the tire is 6 meters, 
the speed of the car is 144 km/h.
-/
theorem car_speed (rev_per_min : ℕ) (circumference : ℝ) (speed : ℝ) :
  rev_per_min = 400 → circumference = 6 → speed = 144 :=
by
  intro h_rev h_circ
  sorry

end car_speed_l691_69165


namespace rectangle_area_l691_69154

theorem rectangle_area (b : ℕ) (l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := 
by
  sorry

end rectangle_area_l691_69154


namespace election_valid_vote_counts_l691_69198

noncomputable def totalVotes : ℕ := 900000
noncomputable def invalidPercentage : ℝ := 0.25
noncomputable def validVotes : ℝ := totalVotes * (1.0 - invalidPercentage)
noncomputable def fractionA : ℝ := 7 / 15
noncomputable def fractionB : ℝ := 5 / 15
noncomputable def fractionC : ℝ := 3 / 15
noncomputable def validVotesA : ℝ := fractionA * validVotes
noncomputable def validVotesB : ℝ := fractionB * validVotes
noncomputable def validVotesC : ℝ := fractionC * validVotes

theorem election_valid_vote_counts :
  validVotesA = 315000 ∧ validVotesB = 225000 ∧ validVotesC = 135000 := by
  sorry

end election_valid_vote_counts_l691_69198


namespace bun_eating_problem_l691_69186

theorem bun_eating_problem
  (n k : ℕ)
  (H1 : 5 * n / 10 + 3 * k / 10 = 180) -- This corresponds to the condition that Zhenya eats 5 buns in 10 minutes, and Sasha eats 3 buns in 10 minutes, for a total of 180 minutes.
  (H2 : n + k = 70) -- This corresponds to the total number of buns eaten.
  : n = 40 ∧ k = 30 :=
by
  sorry

end bun_eating_problem_l691_69186


namespace findQuadraticFunctionAndVertex_l691_69132

noncomputable section

def quadraticFunction (x : ℝ) (b c : ℝ) : ℝ :=
  (1 / 2) * x^2 + b * x + c

theorem findQuadraticFunctionAndVertex :
  (∃ b c : ℝ, quadraticFunction 0 b c = -1 ∧ quadraticFunction 2 b c = -3) →
  (quadraticFunction x (-2) (-1) = (1 / 2) * x^2 - 2 * x - 1) ∧
  (∃ (vₓ vᵧ : ℝ), vₓ = 2 ∧ vᵧ = -3 ∧ quadraticFunction vₓ (-2) (-1) = vᵧ)  :=
by
  sorry

end findQuadraticFunctionAndVertex_l691_69132


namespace JaneTotalEarningsIs138_l691_69153

structure FarmData where
  chickens : ℕ
  ducks : ℕ
  quails : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenPricePerDozen : ℕ
  duckPricePerDozen : ℕ
  quailPricePerDozen : ℕ

def JaneFarmData : FarmData := {
  chickens := 10,
  ducks := 8,
  quails := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenPricePerDozen := 2,
  duckPricePerDozen := 3,
  quailPricePerDozen := 4
}

def eggsLaid (f : FarmData) : ℕ × ℕ × ℕ :=
((f.chickens * f.chickenEggsPerWeek), 
 (f.ducks * f.duckEggsPerWeek), 
 (f.quails * f.quailEggsPerWeek))

def earningsForWeek1 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := duckEggs / 12
let quailDozens := (quailEggs / 12) / 2
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek2 (f : FarmData) : ℕ :=
let (chickenEggs, duckEggs, quailEggs) := eggsLaid f
let chickenDozens := chickenEggs / 12
let duckDozens := (3 * duckEggs / 4) / 12
let quailDozens := quailEggs / 12
(chickenDozens * f.chickenPricePerDozen) + (duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def earningsForWeek3 (f : FarmData) : ℕ :=
let (_, duckEggs, quailEggs) := eggsLaid f
let duckDozens := duckEggs / 12
let quailDozens := quailEggs / 12
(duckDozens * f.duckPricePerDozen) + (quailDozens * f.quailPricePerDozen)

def totalEarnings (f : FarmData) : ℕ :=
earningsForWeek1 f + earningsForWeek2 f + earningsForWeek3 f

theorem JaneTotalEarningsIs138 : totalEarnings JaneFarmData = 138 := by
  sorry

end JaneTotalEarningsIs138_l691_69153


namespace region_area_l691_69112

noncomputable def area_of_region := 
  let a := 0
  let b := Real.sqrt 2 / 2
  ∫ x in a..b, (Real.arccos x) - (Real.arcsin x)

theorem region_area : area_of_region = 2 - Real.sqrt 2 :=
by
  sorry

end region_area_l691_69112


namespace least_value_x_l691_69174

theorem least_value_x (x : ℕ) (p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q)
  (h_distinct : p ≠ q) (h_diff : q - p = 3) (h_even_prim : x / (11 * p * q) = 2) : x = 770 := by
  sorry

end least_value_x_l691_69174


namespace problem_solution_l691_69108

def lean_problem (a : ℝ) : Prop :=
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1)^x₁ > (2 * a - 1)^x₂) →
  a > 1 / 2 ∧ a ≤ 2 / 3

theorem problem_solution (a : ℝ) : lean_problem a :=
  sorry -- Proof is to be filled in

end problem_solution_l691_69108


namespace lee_can_make_cookies_l691_69161

def cookies_per_cup_of_flour (cookies : ℕ) (flour_cups : ℕ) : ℕ :=
  cookies / flour_cups

def flour_needed (sugar_cups : ℕ) (flour_to_sugar_ratio : ℕ) : ℕ :=
  sugar_cups * flour_to_sugar_ratio

def total_cookies (cookies_per_cup : ℕ) (total_flour : ℕ) : ℕ :=
  cookies_per_cup * total_flour

theorem lee_can_make_cookies
  (cookies : ℕ)
  (flour_cups : ℕ)
  (sugar_cups : ℕ)
  (flour_to_sugar_ratio : ℕ)
  (h1 : cookies = 24)
  (h2 : flour_cups = 4)
  (h3 : sugar_cups = 3)
  (h4 : flour_to_sugar_ratio = 2) :
  total_cookies (cookies_per_cup_of_flour cookies flour_cups)
    (flour_needed sugar_cups flour_to_sugar_ratio) = 36 :=
by
  sorry

end lee_can_make_cookies_l691_69161


namespace roy_total_pens_l691_69121

def number_of_pens (blue black red green purple : ℕ) : ℕ :=
  blue + black + red + green + purple

theorem roy_total_pens (blue black red green purple : ℕ)
  (h1 : blue = 8)
  (h2 : black = 4 * blue)
  (h3 : red = blue + black - 5)
  (h4 : green = red / 2)
  (h5 : purple = blue + green - 3) :
  number_of_pens blue black red green purple = 114 := by
  sorry

end roy_total_pens_l691_69121


namespace find_number_l691_69166

theorem find_number (x : ℤ) (h : 35 - 3 * x = 14) : x = 7 :=
by {
  sorry -- This is where the proof would go.
}

end find_number_l691_69166


namespace maximize_revenue_l691_69130

noncomputable def revenue (p : ℝ) : ℝ :=
  p * (150 - 6 * p)

theorem maximize_revenue : ∃ (p : ℝ), p = 12.5 ∧ p ≤ 30 ∧ ∀ q ≤ 30, revenue q ≤ revenue 12.5 := by 
  sorry

end maximize_revenue_l691_69130


namespace segment_length_cd_l691_69159

theorem segment_length_cd
  (AB : ℝ)
  (M : ℝ)
  (N : ℝ)
  (P : ℝ)
  (C : ℝ)
  (D : ℝ)
  (h₁ : AB = 60)
  (h₂ : N = M / 2)
  (h₃ : P = (AB - M) / 2)
  (h₄ : C = N / 2)
  (h₅ : D = P / 2) :
  |C - D| = 15 :=
by
  sorry

end segment_length_cd_l691_69159


namespace interval_k_is_40_l691_69131

def total_students := 1200
def sample_size := 30

theorem interval_k_is_40 : (total_students / sample_size) = 40 :=
by
  sorry

end interval_k_is_40_l691_69131


namespace subset_M_N_l691_69183

-- Definition of the sets
def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1/x < 3 }

theorem subset_M_N : M ⊆ N :=
by
  -- sorry to skip the proof
  sorry

end subset_M_N_l691_69183


namespace ratio_of_x_to_y_l691_69149

variable {x y : ℝ}

theorem ratio_of_x_to_y (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l691_69149


namespace no_combination_of_five_coins_is_75_l691_69118

theorem no_combination_of_five_coins_is_75 :
  ∀ (a b c d e : ℕ), 
    (a + b + c + d + e = 5) →
    ∀ (v : ℤ), 
      v = a * 1 + b * 5 + c * 10 + d * 25 + e * 50 → 
      v ≠ 75 :=
by
  intro a b c d e h1 v h2
  sorry

end no_combination_of_five_coins_is_75_l691_69118


namespace orchid_bushes_planted_tomorrow_l691_69115

theorem orchid_bushes_planted_tomorrow 
  (initial : ℕ) (planted_today : ℕ) (final : ℕ) (planted_tomorrow : ℕ) :
  initial = 47 →
  planted_today = 37 →
  final = 109 →
  planted_tomorrow = final - (initial + planted_today) →
  planted_tomorrow = 25 :=
by
  intros h_initial h_planted_today h_final h_planted_tomorrow
  rw [h_initial, h_planted_today, h_final] at h_planted_tomorrow
  exact h_planted_tomorrow


end orchid_bushes_planted_tomorrow_l691_69115


namespace usamo_2003_q3_l691_69160

open Real

theorem usamo_2003_q3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2)
  + (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2)
  + (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2) ) ≤ 8 := 
sorry

end usamo_2003_q3_l691_69160


namespace mike_ride_distance_l691_69177

-- Definitions from conditions
def mike_cost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annie_cost : ℝ := 2.50 + 5.00 + 0.25 * 16

-- Theorem to prove
theorem mike_ride_distance (m : ℕ) (h : mike_cost m = annie_cost) : m = 36 := by
  sorry

end mike_ride_distance_l691_69177


namespace set_contains_all_nonnegative_integers_l691_69188

theorem set_contains_all_nonnegative_integers (S : Set ℕ) :
  (∃ a b, a ∈ S ∧ b ∈ S ∧ 1 < a ∧ 1 < b ∧ Nat.gcd a b = 1) →
  (∀ x y, x ∈ S → y ∈ S → y ≠ 0 → (x * y) ∈ S ∧ (x % y) ∈ S) →
  (∀ n, n ∈ S) :=
by
  intros h1 h2
  sorry

end set_contains_all_nonnegative_integers_l691_69188


namespace fraction_value_l691_69193

-- Define the constants
def eight := 8
def four := 4

-- Statement to prove
theorem fraction_value : (eight + four) / (eight - four) = 3 := 
by
  sorry

end fraction_value_l691_69193


namespace min_value_x_plus_y_l691_69127

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + y ≥ 4 :=
by
  sorry

end min_value_x_plus_y_l691_69127


namespace parallelogram_height_same_area_l691_69142

noncomputable def rectangle_area (length width : ℕ) : ℕ := length * width

theorem parallelogram_height_same_area (length width base height : ℕ) 
  (h₁ : rectangle_area length width = base * height) 
  (h₂ : length = 12) 
  (h₃ : width = 6) 
  (h₄ : base = 12) : 
  height = 6 := 
sorry

end parallelogram_height_same_area_l691_69142


namespace mn_condition_l691_69110

theorem mn_condition {m n : ℕ} (h : m * n = 121) : (m + 1) * (n + 1) = 144 :=
sorry

end mn_condition_l691_69110


namespace jenny_change_l691_69116

-- Definitions for the conditions
def single_sided_cost_per_page : ℝ := 0.10
def double_sided_cost_per_page : ℝ := 0.17
def pages_per_essay : ℕ := 25
def single_sided_copies : ℕ := 5
def double_sided_copies : ℕ := 2
def pen_cost_before_tax : ℝ := 1.50
def number_of_pens : ℕ := 7
def sales_tax_rate : ℝ := 0.10
def payment_amount : ℝ := 2 * 20.00

-- Hypothesis for the total costs and calculations
noncomputable def total_single_sided_cost : ℝ := single_sided_copies * pages_per_essay * single_sided_cost_per_page
noncomputable def total_double_sided_cost : ℝ := double_sided_copies * pages_per_essay * double_sided_cost_per_page
noncomputable def total_pen_cost_before_tax : ℝ := number_of_pens * pen_cost_before_tax
noncomputable def total_sales_tax : ℝ := sales_tax_rate * total_pen_cost_before_tax
noncomputable def total_pen_cost : ℝ := total_pen_cost_before_tax + total_sales_tax
noncomputable def total_printing_cost : ℝ := total_single_sided_cost + total_double_sided_cost
noncomputable def total_cost : ℝ := total_printing_cost + total_pen_cost
noncomputable def change : ℝ := payment_amount - total_cost

-- The proof statement
theorem jenny_change : change = 7.45 := by
  sorry

end jenny_change_l691_69116
