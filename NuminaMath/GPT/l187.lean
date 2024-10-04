import Mathlib

namespace charlie_delta_purchase_ways_is_correct_l187_187358

noncomputable def numOfWays_ToPurchase4Items : ℕ :=
  let totalCookies := 7
  let totalProducts := totalCookies + 4
  let charliePurchases (n : ℕ) := (totalProducts.choose n)
  let deltaPurchases (n : ℕ) := totalCookies.multichoose n
  (charliePurchases 4) +
  (charliePurchases 3) * (deltaPurchases 1) +
  (charliePurchases 2) * (deltaPurchases 2) +
  (charliePurchases 1) * (deltaPurchases 3) +
  deltaPurchases 4

theorem charlie_delta_purchase_ways_is_correct :
  numOfWays_ToPurchase4Items = 4054 := by
  -- use sorry to observe non proved areas
  sorry

end charlie_delta_purchase_ways_is_correct_l187_187358


namespace abs_eq_condition_l187_187642

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l187_187642


namespace total_juice_drank_l187_187725

open BigOperators

theorem total_juice_drank (joe_juice sam_fraction alex_fraction : ℚ) :
  joe_juice = 3 / 4 ∧ sam_fraction = 1 / 2 ∧ alex_fraction = 1 / 4 → 
  sam_fraction * joe_juice + alex_fraction * joe_juice = 9 / 16 :=
by
  sorry

end total_juice_drank_l187_187725


namespace AE_six_l187_187495

namespace MathProof

-- Definitions of the given conditions
variables {A B C D E : Type}
variables (AB CD AC AE : ℝ)
variables (triangleAED_area triangleBEC_area : ℝ)

-- Given conditions
def conditions : Prop := 
  convex_quadrilateral A B C D ∧
  AB = 9 ∧
  CD = 12 ∧
  AC = 14 ∧
  intersect_at E AC BD ∧
  areas_equal triangleAED_area triangleBEC_area

-- Theorem to prove AE = 6
theorem AE_six (h : conditions AB CD AC AE triangleAED_area triangleBEC_area) : 
  AE = 6 :=
by sorry  -- proof omitted

end MathProof

end AE_six_l187_187495


namespace pencil_eraser_cost_l187_187684

theorem pencil_eraser_cost (p e : ℕ) (h1 : 15 * p + 5 * e = 200) (h2 : p > e) (h_p_pos : p > 0) (h_e_pos : e > 0) :
  p + e = 18 :=
  sorry

end pencil_eraser_cost_l187_187684


namespace path_length_of_B_l187_187972

noncomputable def lengthPathB (BC : ℝ) : ℝ :=
  let radius := BC
  let circumference := 2 * Real.pi * radius
  circumference

theorem path_length_of_B (BC : ℝ) (h : BC = 4 / Real.pi) : lengthPathB BC = 8 := by
  rw [lengthPathB, h]
  simp [Real.pi_ne_zero, div_mul_cancel]
  sorry

end path_length_of_B_l187_187972


namespace Miles_trombones_count_l187_187739

theorem Miles_trombones_count :
  let fingers := 10
  let trumpets := fingers - 3
  let hands := 2
  let guitars := hands + 2
  let french_horns := guitars - 1
  let heads := 1
  let trombones := heads + 2
  trumpets + guitars + french_horns + trombones = 17 → trombones = 3 :=
by
  intros h
  sorry

end Miles_trombones_count_l187_187739


namespace four_digit_integers_with_repeated_digits_l187_187533

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l187_187533


namespace find_x_l187_187657

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l187_187657


namespace commercials_played_l187_187698

theorem commercials_played (M C : ℝ) (h1 : M / C = 9 / 5) (h2 : M + C = 112) : C = 40 :=
by
  sorry

end commercials_played_l187_187698


namespace sqrt_200_eq_l187_187282

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l187_187282


namespace find_f_3_l187_187530

def f (x : ℝ) : ℝ := x + 3  -- define the function as per the condition

theorem find_f_3 : f (3) = 7 := by
  sorry

end find_f_3_l187_187530


namespace sequence_a_n_a_99_value_l187_187429

theorem sequence_a_n_a_99_value :
  ∃ (a : ℕ → ℝ), a 1 = 3 ∧ (∀ n, 2 * (a (n + 1)) - 2 * (a n) = 1) ∧ a 99 = 52 :=
by {
  sorry
}

end sequence_a_n_a_99_value_l187_187429


namespace quadratic_root_relationship_l187_187517

theorem quadratic_root_relationship (a b c : ℂ) (alpha beta : ℂ) (h1 : a ≠ 0) (h2 : alpha + beta = -b / a) (h3 : alpha * beta = c / a) (h4 : beta = 3 * alpha) : 3 * b ^ 2 = 16 * a * c := by
  sorry

end quadratic_root_relationship_l187_187517


namespace z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l187_187704

open Complex

-- Problem definitions
def z (m : ℝ) : ℂ := (2 + I) * m^2 - 2 * (1 - I)

-- Prove that for all m in ℝ, z is imaginary
theorem z_is_imaginary (m : ℝ) : ∃ a : ℝ, z m = a * I :=
  sorry

-- Prove that z is purely imaginary iff m = ±1
theorem z_is_purely_imaginary_iff (m : ℝ) : (∃ b : ℝ, z m = b * I ∧ b ≠ 0) ↔ (m = 1 ∨ m = -1) :=
  sorry

-- Prove that z is on the angle bisector iff m = 0
theorem z_on_angle_bisector_iff (m : ℝ) : (z m).re = -((z m).im) ↔ (m = 0) :=
  sorry

end z_is_imaginary_z_is_purely_imaginary_iff_z_on_angle_bisector_iff_l187_187704


namespace largest_odd_digit_multiple_of_5_is_9955_l187_187464

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), is_odd_digit d

def largest_odd_digit_multiple_of_5 (n : ℕ) : Prop :=
  n < 10000 ∧ n % 5 = 0 ∧ all_odd_digits n

theorem largest_odd_digit_multiple_of_5_is_9955 :
  ∃ n, largest_odd_digit_multiple_of_5 n ∧ n = 9955 :=
begin
  sorry
end

end largest_odd_digit_multiple_of_5_is_9955_l187_187464


namespace arithmetic_sequence_15th_term_l187_187018

/-- 
The arithmetic sequence with first term 1 and common difference 3.
The 15th term of this sequence is 43.
-/
theorem arithmetic_sequence_15th_term :
  ∀ (a1 d n : ℕ), a1 = 1 → d = 3 → n = 15 → (a1 + (n - 1) * d) = 43 :=
by
  sorry

end arithmetic_sequence_15th_term_l187_187018


namespace sqrt_200_simplified_l187_187269

-- Definitions based on conditions from part a)
def factorization : Nat := 2 ^ 3 * 5 ^ 2

lemma sqrt_property (a b : ℕ) : Real.sqrt (a^2 * b) = a * Real.sqrt b := sorry

-- The proof problem (only the statement, not the proof)
theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  have h1 : 200 = 2^3 * 5^2 := by rfl
  have h2 : Real.sqrt (200) = Real.sqrt (2^3 * 5^2) := by rw h1
  rw [←show 200 = factorization by rfl] at h2
  exact sorry

end sqrt_200_simplified_l187_187269


namespace maximum_value_is_17_l187_187752

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l187_187752


namespace ec_value_l187_187901

theorem ec_value (AB AD : ℝ) (EFGH1 EFGH2 : ℝ) (x : ℝ)
  (h1 : AB = 2)
  (h2 : AD = 1)
  (h3 : EFGH1 = 1 / 2 * AB)
  (h4 : EFGH2 = 1 / 2 * AD)
  (h5 : 1 + 2 * x = 1)
  : x = 1 / 3 :=
by sorry

end ec_value_l187_187901


namespace dima_more_berries_and_difference_l187_187695

section RaspberryPicking

-- Define conditions
def total_berries : ℕ := 450
def dima_contrib_per_2_berries : ℚ := 1
def sergei_contrib_per_3_berries : ℚ := 2
def dima_speed_factor : ℚ := 2

-- Defining the problem of determining the berry counts
theorem dima_more_berries_and_difference :
  let dima_cycles := 2 * total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / dima_contrib_per_2_berries in
  let sergei_cycles := total_berries / (2 * dima_contrib_per_2_berries + 3 * sergei_contrib_per_3_berries * (1 / dima_speed_factor)) / sergei_contrib_per_3_berries in
  let berries_dima := dima_cycles * (dima_contrib_per_2_berries / 2) in
  let berries_sergei := sergei_cycles * (sergei_contrib_per_3_berries / 3) in
  berries_dima > berries_sergei ∧
  berries_dima - berries_sergei = 50 :=
by --- skip the proof
sorry

end RaspberryPicking

end dima_more_berries_and_difference_l187_187695


namespace hyperbola_distance_to_left_focus_l187_187594

theorem hyperbola_distance_to_left_focus (P : ℝ × ℝ)
  (h1 : (P.1^2) / 9 - (P.2^2) / 16 = 1)
  (dPF2 : dist P (4, 0) = 4) : dist P (-4, 0) = 10 := 
sorry

end hyperbola_distance_to_left_focus_l187_187594


namespace find_x_l187_187655

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l187_187655


namespace medicine_dose_per_part_l187_187363

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l187_187363


namespace find_value_l187_187198

theorem find_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) : 2 * Real.sin x + 3 * Real.cos x = -7 / 3 := 
sorry

end find_value_l187_187198


namespace solve_inequality_l187_187912

theorem solve_inequality (a : ℝ) :
  (a = 0 → {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a > 0 → {x : ℝ | x ≥ 2 / a} ∪ {x : ℝ | x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (-2 < a ∧ a < 0 → {x : ℝ | 2 / a ≤ x ∧ x ≤ -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a = -2 → {x : ℝ | x = -1} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) ∧
  (a < -2 → {x : ℝ | -1 ≤ x ∧ x ≤ 2 / a} = {x : ℝ | ax^2 - 2 ≥ 2 * x - a * x}) :=
by 
  sorry

end solve_inequality_l187_187912


namespace mike_spend_on_plants_l187_187905

def Mike_buys : Prop :=
  let rose_bushes_total := 6
  let rose_bush_cost := 75
  let friend_rose_bushes := 2
  let self_rose_bushes := rose_bushes_total - friend_rose_bushes
  let self_rose_bush_cost := self_rose_bushes * rose_bush_cost
  let tiger_tooth_aloe_total := 2
  let aloe_cost := 100
  let self_aloe_cost := tiger_tooth_aloe_total * aloe_cost
  self_rose_bush_cost + self_aloe_cost = 500

theorem mike_spend_on_plants :
  Mike_buys := by
  sorry

end mike_spend_on_plants_l187_187905


namespace Rachel_painting_time_l187_187738

theorem Rachel_painting_time :
  ∀ (Matt_time Patty_time Rachel_time : ℕ),
  Matt_time = 12 →
  Patty_time = Matt_time / 3 →
  Rachel_time = 2 * Patty_time + 5 →
  Rachel_time = 13 :=
by
  intros Matt_time Patty_time Rachel_time hMatt hPatty hRachel
  rw [hMatt] at hPatty
  rw [hPatty, hRachel]
  sorry

end Rachel_painting_time_l187_187738


namespace largest_k_for_sum_of_integers_l187_187983

theorem largest_k_for_sum_of_integers (k : ℕ) (n : ℕ) (h1 : 3^12 = k * n + k * (k + 1) / 2) 
  (h2 : k ∣ 2 * 3^12) (h3 : k < 1031) : k ≤ 486 :=
by 
  sorry -- The proof is skipped here, only the statement is required 

end largest_k_for_sum_of_integers_l187_187983


namespace willy_episodes_per_day_l187_187937

def total_episodes (seasons : ℕ) (episodes_per_season : ℕ) : ℕ :=
  seasons * episodes_per_season

def episodes_per_day (total_episodes : ℕ) (days : ℕ) : ℕ :=
  total_episodes / days

theorem willy_episodes_per_day :
  episodes_per_day (total_episodes 3 20) 30 = 2 :=
by
  sorry

end willy_episodes_per_day_l187_187937


namespace cube_of_square_of_third_smallest_prime_is_correct_l187_187309

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l187_187309


namespace solve_for_x_l187_187748

theorem solve_for_x :
  (∃ x : ℝ, (1 / 3 - 1 / 4) = 1 / x) → ∃ x : ℝ, x = 12 :=
by
  intro h,
  obtain ⟨x, hx⟩ := h,
  use 12,
  have : 1 / 3 - 1 / 4 = 1 / 12, by
  { calc
      1 / 3 - 1 / 4 = 4 / 12 - 3 / 12 : by norm_num
                 ... = 1 / 12 : by norm_num },
  exact this ▸ hx.symm

end solve_for_x_l187_187748


namespace point_probability_in_cone_l187_187996

noncomputable def volume_of_cone (S : ℝ) (h : ℝ) : ℝ :=
  (1/3) * S * h

theorem point_probability_in_cone (P M : ℝ) (S_ABC : ℝ) (h_P h_M : ℝ)
  (h_volume_condition : volume_of_cone S_ABC h_P ≤ volume_of_cone S_ABC h_M / 3) :
  (1 - (2 / 3) ^ 3) = 19 / 27 :=
by
  sorry

end point_probability_in_cone_l187_187996


namespace f_iterated_result_l187_187559

def f (x : ℕ) : ℕ :=
  if Even x then 3 * x / 2 else 2 * x + 1

theorem f_iterated_result : f (f (f (f 1))) = 31 := by
  sorry

end f_iterated_result_l187_187559


namespace maximum_value_is_17_l187_187755

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l187_187755


namespace inequality_proof_l187_187333

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l187_187333


namespace area_of_region_eq_24π_l187_187637

theorem area_of_region_eq_24π :
  (∃ R, R > 0 ∧ ∀ (x y : ℝ), x ^ 2 + y ^ 2 + 8 * x + 18 * y + 73 = R ^ 2) →
  ∃ π : ℝ, π > 0 ∧ area = 24 * π :=
by
  sorry

end area_of_region_eq_24π_l187_187637


namespace second_place_jump_l187_187728

theorem second_place_jump : 
  ∀ (Kyungsoo Younghee Jinju Chanho : ℝ), 
    Kyungsoo = 2.3 → 
    Younghee = 0.9 → 
    Jinju = 1.8 → 
    Chanho = 2.5 → 
    ((Kyungsoo < Chanho) ∧ (Kyungsoo > Jinju) ∧ (Kyungsoo > Younghee)) :=
by 
  sorry

end second_place_jump_l187_187728


namespace number_of_newborn_members_l187_187071

theorem number_of_newborn_members (N : ℝ) (h : (9/10 : ℝ) ^ 3 * N = 291.6) : N = 400 :=
sorry

end number_of_newborn_members_l187_187071


namespace hall_of_mirrors_l187_187724

theorem hall_of_mirrors (h : ℝ) 
    (condition1 : 2 * (30 * h) + (20 * h) = 960) :
  h = 12 :=
by
  sorry

end hall_of_mirrors_l187_187724


namespace ninja_star_ratio_l187_187169

-- Define variables for the conditions
variables (Eric_stars Chad_stars Jeff_stars Total_stars : ℕ) (Jeff_bought : ℕ)

/-- Given the following conditions:
1. Eric has 4 ninja throwing stars.
2. Jeff now has 6 throwing stars.
3. Jeff bought 2 ninja stars from Chad.
4. Altogether, they have 16 ninja throwing stars.

We want to prove that the ratio of the number of ninja throwing stars Chad has to the number Eric has is 2:1. --/
theorem ninja_star_ratio
  (h1 : Eric_stars = 4)
  (h2 : Jeff_stars = 6)
  (h3 : Jeff_bought = 2)
  (h4 : Total_stars = 16)
  (h5 : Eric_stars + Jeff_stars - Jeff_bought + Chad_stars = Total_stars) :
  Chad_stars / Eric_stars = 2 :=
by
  sorry

end ninja_star_ratio_l187_187169


namespace find_x_l187_187551

/--
Given the following conditions:
1. The sum of angles around a point is 360 degrees.
2. The angles are 7x, 6x, 3x, and (2x + y).
3. y = 2x.

Prove that x = 18 degrees.
-/
theorem find_x (x y : ℝ) (h : 18 * x + y = 360) (h_y : y = 2 * x) : x = 18 :=
by
  sorry

end find_x_l187_187551


namespace sandwich_count_l187_187005

-- Define the given conditions
def meats : ℕ := 8
def cheeses : ℕ := 12
def cheese_combination_count : ℕ := Nat.choose cheeses 3

-- Define the total sandwich count based on the conditions
def total_sandwiches : ℕ := meats * cheese_combination_count

-- The theorem we want to prove
theorem sandwich_count : total_sandwiches = 1760 := by
  -- Mathematical steps here are omitted
  sorry

end sandwich_count_l187_187005


namespace suitcase_combinations_l187_187089

theorem suitcase_combinations :
  let multiples_of_4 := {x | 1 ≤ x ∧ x ≤ 40 ∧ x % 4 = 0}.card,
      odd_numbers := {y | 1 ≤ y ∧ y ≤ 40 ∧ y % 2 = 1}.card,
      multiples_of_5 := {z | 1 ≤ z ∧ z ≤ 40 ∧ z % 5 = 0}.card in
  multiples_of_4 * odd_numbers * multiples_of_5 = 1600 :=
by
  sorry

end suitcase_combinations_l187_187089


namespace candies_leftover_l187_187416

theorem candies_leftover (n : ℕ) : 31254389 % 6 = 5 :=
by {
  sorry
}

end candies_leftover_l187_187416


namespace cube_square_third_smallest_prime_l187_187318

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l187_187318


namespace no_positive_integer_satisfies_conditions_l187_187399

theorem no_positive_integer_satisfies_conditions : 
  ¬ ∃ (n : ℕ), (100 ≤ n / 4) ∧ (n / 4 ≤ 999) ∧ (100 ≤ 4 * n) ∧ (4 * n ≤ 999) :=
by
  -- Proof will go here.
  sorry

end no_positive_integer_satisfies_conditions_l187_187399


namespace second_plan_minutes_included_l187_187825

theorem second_plan_minutes_included 
  (monthly_fee1 : ℝ := 50) 
  (limit1 : ℝ := 500) 
  (cost_per_minute1 : ℝ := 0.35) 
  (monthly_fee2 : ℝ := 75) 
  (cost_per_minute2 : ℝ := 0.45) 
  (M : ℝ) 
  (usage : ℝ := 2500)
  (cost1 := monthly_fee1 + cost_per_minute1 * (usage - limit1))
  (cost2 := monthly_fee2 + cost_per_minute2 * (usage - M))
  (equal_costs : cost1 = cost2) : 
  M = 1000 := 
by
  sorry 

end second_plan_minutes_included_l187_187825


namespace floor_ceiling_sum_l187_187388

theorem floor_ceiling_sum : 
    Int.floor (0.998 : ℝ) + Int.ceil (2.002 : ℝ) = 3 := by
  sorry

end floor_ceiling_sum_l187_187388


namespace inequality_proof_l187_187336

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l187_187336


namespace players_quit_l187_187629

theorem players_quit (initial_players remaining_lives lives_per_player : ℕ) 
  (h1 : initial_players = 8) (h2 : remaining_lives = 15) (h3 : lives_per_player = 5) :
  initial_players - (remaining_lives / lives_per_player) = 5 :=
by
  -- A proof is required here
  sorry

end players_quit_l187_187629


namespace abs_eq_condition_l187_187643

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l187_187643


namespace sin_C_value_l187_187713

noncomputable def triangle_sine_proof (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : Real := by
  -- Utilizing the Law of Sines and given conditions to find sin C
  sorry

theorem sin_C_value (A B C a b c : Real) (hB : B = 2 * Real.pi / 3) 
  (hb : b = 3 * c) : triangle_sine_proof A B C a b c hB hb = Real.sqrt 3 / 6 := by
  sorry

end sin_C_value_l187_187713


namespace cube_dot_path_length_l187_187359

/-- A cube with edges of length 2 cm has a dot marked in the center of the top face.
    The cube sits on a flat table and is rolled in one direction without lifting or slipping,
    making a few rotations by pivoting around its edges. 
    The total length of the path followed by the dot is dπ, where d is a constant.
    Prove that d = 1 given the following conditions:
    - Initial rotation around an edge joining the top and bottom faces,
    - Subsequent rotation around an edge on the side face until the dot returns to the top face.
-/ 
theorem cube_dot_path_length : 
  let side_length : ℝ := 2
  let radius : ℝ := side_length / 2
  let quarter_turn_length : ℝ := (1 / 4) * 2 * Real.pi * radius
  (2 * quarter_turn_length = Real.pi) → 
  d = 1 :=
by
  sorry

end cube_dot_path_length_l187_187359


namespace number_of_girls_l187_187623

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l187_187623


namespace square_side_length_in_right_triangle_l187_187187

theorem square_side_length_in_right_triangle
  (AC BC : ℝ)
  (h1 : AC = 3)
  (h2 : BC = 7)
  (right_triangle : ∃ A B C : ℝ × ℝ, A = (3, 0) ∧ B = (0, 7) ∧ C = (0, 0) ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = AC^2 + BC^2) :
  ∃ s : ℝ, s = 2.1 :=
by
  -- Proof goes here
  sorry

end square_side_length_in_right_triangle_l187_187187


namespace max_value_of_fraction_l187_187762

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l187_187762


namespace sqrt_200_eq_10_sqrt_2_l187_187257

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l187_187257


namespace raju_working_days_l187_187888

theorem raju_working_days (x : ℕ) 
  (h1: (1 / 10 : ℚ) + 1 / x = 1 / 8) : x = 40 :=
by sorry

end raju_working_days_l187_187888


namespace inequalities_not_all_hold_l187_187208

theorem inequalities_not_all_hold (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
    ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end inequalities_not_all_hold_l187_187208


namespace pq_problem_l187_187221

theorem pq_problem
  (p q : ℝ)
  (h1 : ∀ x : ℝ, (x - 7) * (2 * x + 11) = x^2 - 19 * x +  60)
  (h2 : p * q = 7 * (-9))
  (h3 : 7 + (-9) = -16):
  (p - 2) * (q - 2) = -55 :=
by
  sorry

end pq_problem_l187_187221


namespace charity_total_cost_l187_187354

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l187_187354


namespace maximum_value_is_17_l187_187756

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l187_187756


namespace olivia_earnings_l187_187240

-- Define Olivia's hourly wage
def wage : ℕ := 9

-- Define the hours worked on each day
def hours_monday : ℕ := 4
def hours_wednesday : ℕ := 3
def hours_friday : ℕ := 6

-- Define the total hours worked
def total_hours : ℕ := hours_monday + hours_wednesday + hours_friday

-- Define the total earnings
def total_earnings : ℕ := total_hours * wage

-- State the theorem
theorem olivia_earnings : total_earnings = 117 :=
by
  sorry

end olivia_earnings_l187_187240


namespace total_distance_of_bus_rides_l187_187122

theorem total_distance_of_bus_rides :
  let vince_distance   := 5 / 8
  let zachary_distance := 1 / 2
  let alice_distance   := 17 / 20
  let rebecca_distance := 2 / 5
  let total_distance   := vince_distance + zachary_distance + alice_distance + rebecca_distance
  total_distance = 19/8 := by
  sorry

end total_distance_of_bus_rides_l187_187122


namespace day_crew_fraction_loaded_l187_187472

-- Let D be the number of boxes loaded by each worker on the day crew
-- Let W_d be the number of workers on the day crew
-- Let W_n be the number of workers on the night crew
-- Let B_d be the total number of boxes loaded by the day crew
-- Let B_n be the total number of boxes loaded by the night crew

variable (D W_d : ℕ) 
variable (B_d := D * W_d)
variable (W_n := (4 / 9 : ℚ) * W_d)
variable (B_n := (3 / 4 : ℚ) * D * W_n)
variable (total_boxes := B_d + B_n)

theorem day_crew_fraction_loaded : 
  (D * W_d) / (D * W_d + (3 / 4 : ℚ) * D * ((4 / 9 : ℚ) * W_d)) = (3 / 4 : ℚ) := sorry

end day_crew_fraction_loaded_l187_187472


namespace balance_increase_second_year_l187_187634

variable (initial_deposit : ℝ) (balance_first_year : ℝ) 
variable (total_percentage_increase : ℝ)

theorem balance_increase_second_year
  (h1 : initial_deposit = 1000)
  (h2 : balance_first_year = 1100)
  (h3 : total_percentage_increase = 0.32) : 
  (balance_first_year + (initial_deposit * total_percentage_increase) - balance_first_year) / balance_first_year * 100 = 20 :=
by
  sorry

end balance_increase_second_year_l187_187634


namespace divides_343_l187_187437

theorem divides_343 
  (x y z : ℕ) 
  (hx : 0 < x)
  (hy : 0 < y)
  (hz : 0 < z)
  (h : 7 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y)) :
  343 ∣ (x + 6 * y) * (2 * x + 5 * y) * (3 * x + 4 * y) :=
by sorry

end divides_343_l187_187437


namespace binomial_expansion_correct_statements_l187_187077

theorem binomial_expansion_correct_statements :
    let x := (1 : ℝ) in
    let f := (2 * (x) ^ (1 / 2)) - (1 / x) in
    (∑ i in finset.range 8, binomial 7 i) = 128 ∧
    ((f)^7 |>.expand |>.sum = 1) ∧
    ¬(∃ T, x ^ ((7 - 3 * T) / 2) = 1) ∧
    (binomial 7 5 * (2 ^ 2) * (-1)^5) = -84 :=
by
  sorry

end binomial_expansion_correct_statements_l187_187077


namespace nabla_example_l187_187496

def nabla (a b : ℕ) : ℕ := 2 + b ^ a

theorem nabla_example : nabla (nabla 1 2) 3 = 83 :=
  by
  sorry

end nabla_example_l187_187496


namespace max_value_of_fraction_l187_187786

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l187_187786


namespace max_value_of_expression_l187_187772

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l187_187772


namespace gcd_g50_g51_l187_187220

-- Define the polynomial g(x)
def g (x : ℤ) : ℤ := x^2 + x + 2023

-- State the theorem with necessary conditions
theorem gcd_g50_g51 : Int.gcd (g 50) (g 51) = 17 :=
by
  -- Goals and conditions stated
  sorry  -- Placeholder for the proof

end gcd_g50_g51_l187_187220


namespace green_chips_count_l187_187796

theorem green_chips_count (total_chips : ℕ) (blue_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ)
  (h1 : total_chips = 60)
  (h2 : blue_chips = total_chips / 6)
  (h3 : red_chips = 34)
  (h4 : green_chips = total_chips - blue_chips - red_chips) :
  green_chips = 16 :=
by {
  -- Define the intermediate steps
  have h_blue_calculation : blue_chips = 10,
  { rw h1, exact Nat.div_eq_of_eq_mul_right (Nat.succ_pos 5) rfl },

  -- Assume that 34 red chips are given
  have h_red_count : red_chips = 34 := h3,

  -- Define the total number of chips
  have h_total_calculation : green_chips = 60 - 10 - 34
    by { rw [h1, h_blue_calculation, h_red_count] },

  -- Conclusion
  exact by { rw h_total_calculation, norm_num }
}

end green_chips_count_l187_187796


namespace tenth_term_in_sequence_l187_187365

def seq (n : ℕ) : ℚ :=
  (-1) ^ (n + 1) * ((2 * n - 1) / (n ^ 2 + 1))

theorem tenth_term_in_sequence :
  seq 10 = -19 / 101 :=
by
  -- Proof omitted
  sorry

end tenth_term_in_sequence_l187_187365


namespace sqrt_200_eq_10_l187_187265

theorem sqrt_200_eq_10 : real.sqrt 200 = 10 :=
by
  calc
    real.sqrt 200 = real.sqrt (2^2 * 5^2) : by sorry -- show 200 = 2^2 * 5^2
    ... = real.sqrt (2^2) * real.sqrt (5^2) : by sorry -- property of square roots of products
    ... = 2 * 5 : by sorry -- using the property sqrt (a^2) = a
    ... = 10 : by sorry

end sqrt_200_eq_10_l187_187265


namespace ratio_of_monkeys_to_snakes_l187_187726

-- Define the given problem conditions
def JohnZoo :=
  let snakes := 15
  ∃ M L P D : ℤ,
    L = M - 5 ∧
    P = L + 8 ∧
    D = P / 3 ∧
    15 + M + L + P + D = 114 ∧
    2 * snakes = M

-- Statement of the problem to prove in Lean
theorem ratio_of_monkeys_to_snakes : JohnZoo → (15 * 2 = 30) := sorry

end ratio_of_monkeys_to_snakes_l187_187726


namespace total_eyes_among_ninas_pet_insects_l187_187091

theorem total_eyes_among_ninas_pet_insects
  (num_spiders : ℕ) (num_ants : ℕ)
  (eyes_per_spider : ℕ) (eyes_per_ant : ℕ)
  (h_num_spiders : num_spiders = 3)
  (h_num_ants : num_ants = 50)
  (h_eyes_per_spider : eyes_per_spider = 8)
  (h_eyes_per_ant : eyes_per_ant = 2) :
  num_spiders * eyes_per_spider + num_ants * eyes_per_ant = 124 := 
by
  rw [h_num_spiders, h_num_ants, h_eyes_per_spider, h_eyes_per_ant]
  norm_num
  done

end total_eyes_among_ninas_pet_insects_l187_187091


namespace inequality_proof_l187_187350

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187350


namespace least_positive_integer_addition_l187_187323

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l187_187323


namespace isosceles_right_triangle_angle_l187_187882

-- Define the conditions given in the problem
def is_isosceles (a b c : ℝ) : Prop := 
(a = b ∨ b = c ∨ c = a)

def is_right_triangle (a b c : ℝ) : Prop := 
(a = 90 ∨ b = 90 ∨ c = 90)

def angles_sum_to_180 (a b c : ℝ) : Prop :=
a + b + c = 180

-- The Proof Problem
theorem isosceles_right_triangle_angle :
  ∀ (a b c x : ℝ), (is_isosceles a b c) → (is_right_triangle a b c) → (angles_sum_to_180 a b c) → (x = a ∨ x = b ∨ x = c) → x = 45 :=
by
  intros a b c x h_isosceles h_right h_sum h_x
  -- Proof is omitted with sorry
  sorry

end isosceles_right_triangle_angle_l187_187882


namespace Chloe_initial_picked_carrots_l187_187976

variable (x : ℕ)

theorem Chloe_initial_picked_carrots :
  (x - 45 + 42 = 45) → (x = 48) :=
by
  intro h
  sorry

end Chloe_initial_picked_carrots_l187_187976


namespace x_y_sum_l187_187565

theorem x_y_sum (x y : ℝ) 
  (h1 : (x-1)^3 + 1997*(x-1) = -1)
  (h2 : (y-1)^3 + 1997*(y-1) = 1) :
  x + y = 2 :=
sorry

end x_y_sum_l187_187565


namespace golden_ratio_problem_l187_187884

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

theorem golden_ratio_problem (m : ℝ) (x : ℝ) :
  (1000 ≤ m) → (1000 ≤ x) → (x ≤ m) →
  ((m - 1000) / (x - 1000) = phi ∧ (x - 1000) / (m - x) = phi) →
  (m = 2000 ∨ m = 2618) :=
by
  sorry

end golden_ratio_problem_l187_187884


namespace max_marks_l187_187810

theorem max_marks (T : ℝ) (h : 0.33 * T = 165) : T = 500 := 
by {
  sorry
}

end max_marks_l187_187810


namespace cos_seven_pi_over_four_l187_187392

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l187_187392


namespace conservation_center_total_turtles_l187_187963

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l187_187963


namespace polynomial_roots_l187_187030

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l187_187030


namespace triangle_height_l187_187591

theorem triangle_height (base height : ℝ) (area : ℝ) (h_base : base = 4) (h_area : area = 12) (h_area_eq : area = (base * height) / 2) :
  height = 6 :=
by
  sorry

end triangle_height_l187_187591


namespace stubborn_robot_returns_to_start_l187_187474

inductive Direction
| East | North | West | South

inductive Command
| STEP | LEFT

structure Robot :=
  (position : ℤ × ℤ)
  (direction : Direction)

def turnLeft : Direction → Direction
| Direction.East  => Direction.North
| Direction.North => Direction.West
| Direction.West  => Direction.South
| Direction.South => Direction.East

def moveStep : Robot → Robot
| ⟨(x, y), Direction.East⟩  => ⟨(x + 1, y), Direction.East⟩
| ⟨(x, y), Direction.North⟩ => ⟨(x, y + 1), Direction.North⟩
| ⟨(x, y), Direction.West⟩  => ⟨(x - 1, y), Direction.West⟩
| ⟨(x, y), Direction.South⟩ => ⟨(x, y - 1), Direction.South⟩

def executeCommand : Command → Robot → Robot
| Command.STEP, robot => moveStep robot
| Command.LEFT, robot => ⟨robot.position, turnLeft robot.direction⟩

def invertCommand : Command → Command
| Command.STEP => Command.LEFT
| Command.LEFT => Command.STEP

def executeSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand cmd r) robot

def executeInvertedSequence (seq : List Command) (robot : Robot) : Robot :=
  seq.foldl (λ r cmd => executeCommand (invertCommand cmd) r) robot

def initialRobot : Robot := ⟨(0, 0), Direction.East⟩

def exampleProgram : List Command :=
  [Command.LEFT, Command.LEFT, Command.LEFT, Command.LEFT, Command.STEP, Command.STEP,
   Command.LEFT, Command.LEFT]

theorem stubborn_robot_returns_to_start :
  let robot := executeSequence exampleProgram initialRobot
  executeInvertedSequence exampleProgram robot = initialRobot :=
by
  sorry

end stubborn_robot_returns_to_start_l187_187474


namespace frac_eq_l187_187385

def my_at (a b : ℕ) := a * b + b^2
def my_hash (a b : ℕ) := a^2 + b + a * b^2

theorem frac_eq : my_at 4 3 / my_hash 4 3 = 21 / 55 :=
by
  sorry

end frac_eq_l187_187385


namespace passes_to_left_l187_187148

theorem passes_to_left
  (total_passes passes_left passes_right passes_center : ℕ)
  (h1 : total_passes = 50)
  (h2 : passes_right = 2 * passes_left)
  (h3 : passes_center = passes_left + 2)
  (h4 : total_passes = passes_left + passes_right + passes_center) :
  passes_left = 12 :=
by
  sorry

end passes_to_left_l187_187148


namespace square_side_length_in_right_triangle_l187_187188

theorem square_side_length_in_right_triangle
  (AC BC : ℝ)
  (h1 : AC = 3)
  (h2 : BC = 7)
  (right_triangle : ∃ A B C : ℝ × ℝ, A = (3, 0) ∧ B = (0, 7) ∧ C = (0, 0) ∧ (A.1 - C.1)^2 + (A.2 - C.2)^2 = AC^2 ∧ (B.1 - C.1)^2 + (B.2 - C.2)^2 = BC^2 ∧ (A.1 - B.1)^2 + (A.2 - B.2)^2 = AC^2 + BC^2) :
  ∃ s : ℝ, s = 2.1 :=
by
  -- Proof goes here
  sorry

end square_side_length_in_right_triangle_l187_187188


namespace profit_is_1500_l187_187926

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l187_187926


namespace calculate_ratio_milk_l187_187572

def ratio_milk_saturdays_weekdays (S : ℕ) : Prop :=
  let Weekdays := 15 -- total milk on weekdays
  let Sundays := 9 -- total milk on Sundays
  S + Weekdays + Sundays = 30 → S / Weekdays = 2 / 5

theorem calculate_ratio_milk : ratio_milk_saturdays_weekdays 6 :=
by
  unfold ratio_milk_saturdays_weekdays
  intros
  apply sorry -- Proof goes here

end calculate_ratio_milk_l187_187572


namespace find_possible_values_l187_187027

theorem find_possible_values (m n : ℕ) (h_rel_prime : Nat.gcd m n = 1) (h_m_pos : 0 < m) (h_n_pos : 0 < n):
  ∃ k : ℕ, k = 1 ∨ k = 5 ∧ k = (m^2 + 20 * m * n + n^2) / (m^3 + n^3) :=
by {
  -- Since the problem asks for the Lean statement only, the proof is not included.
  sorry
}

end find_possible_values_l187_187027


namespace total_profit_from_selling_30_necklaces_l187_187924

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l187_187924


namespace scientific_notation_110_billion_l187_187552

def scientific_notation_form (a : ℝ) (n : ℤ) : Prop :=
  1 ≤ |a| ∧ |a| < 10 ∧ 110 * 10^8 = a * 10^n

theorem scientific_notation_110_billion :
  ∃ (a : ℝ) (n : ℤ), scientific_notation_form a n ∧ a = 1.1 ∧ n = 10 :=
by
  sorry

end scientific_notation_110_billion_l187_187552


namespace highest_point_difference_l187_187590

theorem highest_point_difference :
  let A := -112
  let B := -80
  let C := -25
  max A (max B C) - min A (min B C) = 87 :=
by
  sorry

end highest_point_difference_l187_187590


namespace max_value_of_fraction_l187_187783

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l187_187783


namespace jori_water_left_l187_187082

theorem jori_water_left (initial_water : ℚ) (used_water : ℚ) : initial_water = 3 ∧ used_water = 5/4 → initial_water - used_water = 7/4 :=
by {
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
}

end jori_water_left_l187_187082


namespace find_sum_lent_l187_187127

theorem find_sum_lent (P : ℝ) : 
  (∃ R T : ℝ, R = 4 ∧ T = 8 ∧ I = P - 170 ∧ I = (P * 8) / 25) → P = 250 :=
by
  sorry

end find_sum_lent_l187_187127


namespace sqrt_200_eq_l187_187285

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l187_187285


namespace medicine_dosage_l187_187362

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l187_187362


namespace f_sum_zero_l187_187981

noncomputable def f : ℝ → ℝ := sorry

axiom f_property_1 : ∀ x : ℝ, f (x ^ 3) = (f x) ^ 3
axiom f_property_2 : ∀ x1 x2 : ℝ, x1 ≠ x2 → f x1 ≠ f x2

theorem f_sum_zero : f 0 + f (-1) + f 1 = 0 := by
  sorry

end f_sum_zero_l187_187981


namespace mod_multiplication_l187_187452

theorem mod_multiplication :
  (176 * 929) % 50 = 4 :=
by
  sorry

end mod_multiplication_l187_187452


namespace car_speed_proof_l187_187489

noncomputable def car_speed_in_kmh (rpm : ℕ) (circumference : ℕ) : ℕ :=
  (rpm * circumference * 60) / 1000

theorem car_speed_proof : 
  car_speed_in_kmh 400 1 = 24 := 
by
  sorry

end car_speed_proof_l187_187489


namespace symmetric_points_ab_value_l187_187426

theorem symmetric_points_ab_value
  (a b : ℤ)
  (h₁ : a + 2 = -4)
  (h₂ : 2 = b) :
  a * b = -12 :=
by
  sorry

end symmetric_points_ab_value_l187_187426


namespace inequality_proof_l187_187348

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187348


namespace ratio_first_term_l187_187001

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l187_187001


namespace cube_of_square_of_third_smallest_prime_l187_187311

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l187_187311


namespace students_per_bus_l187_187117

def total_students : ℕ := 360
def number_of_buses : ℕ := 8

theorem students_per_bus : total_students / number_of_buses = 45 :=
by
  sorry

end students_per_bus_l187_187117


namespace first_term_of_new_ratio_l187_187003

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end first_term_of_new_ratio_l187_187003


namespace value_of_r_minus_p_l187_187812

variable (p q r : ℝ)

-- The conditions given as hypotheses
def arithmetic_mean_pq := (p + q) / 2 = 10
def arithmetic_mean_qr := (q + r) / 2 = 25

-- The goal is to prove that r - p = 30
theorem value_of_r_minus_p (h1: arithmetic_mean_pq p q) (h2: arithmetic_mean_qr q r) :
  r - p = 30 := by
  sorry

end value_of_r_minus_p_l187_187812


namespace S_is_positive_rationals_l187_187734

variable {S : Set ℚ}

-- Defining the conditions as axioms
axiom cond1 (a b : ℚ) (ha : a ∈ S) (hb : b ∈ S) : (a + b ∈ S) ∧ (a * b ∈ S)
axiom cond2 {r : ℚ} : (r ∈ S) ∨ (-r ∈ S) ∨ (r = 0)

-- The theorem to prove
theorem S_is_positive_rationals : S = { r : ℚ | r > 0 } := sorry

end S_is_positive_rationals_l187_187734


namespace volume_of_prism_l187_187487

variable (l w h : ℝ)

def area1 (l w : ℝ) : ℝ := l * w
def area2 (w h : ℝ) : ℝ := w * h
def area3 (l h : ℝ) : ℝ := l * h
def volume (l w h : ℝ) : ℝ := l * w * h

axiom cond1 : area1 l w = 15
axiom cond2 : area2 w h = 20
axiom cond3 : area3 l h = 30

theorem volume_of_prism : volume l w h = 30 * Real.sqrt 10 :=
by
  sorry

end volume_of_prism_l187_187487


namespace find_english_marks_l187_187745

variable (mathematics science social_studies english biology : ℕ)
variable (average_marks : ℕ)
variable (number_of_subjects : ℕ := 5)

-- Conditions
axiom score_math : mathematics = 76
axiom score_sci : science = 65
axiom score_ss : social_studies = 82
axiom score_bio : biology = 95
axiom average : average_marks = 77

-- The proof problem
theorem find_english_marks :
  english = 67 :=
  sorry

end find_english_marks_l187_187745


namespace sqrt_200_eq_10_sqrt_2_l187_187252

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l187_187252


namespace solve_for_y_l187_187099

theorem solve_for_y : ∀ (y : ℝ), (3 / 4 - 5 / 8 = 1 / y) → y = 8 :=
by
  intros y h
  sorry

end solve_for_y_l187_187099


namespace ellipse_abs_sum_max_min_l187_187207

theorem ellipse_abs_sum_max_min (x y : ℝ) (h : x^2 / 4 + y^2 / 9 = 1) :
  2 ≤ |x| + |y| ∧ |x| + |y| ≤ 3 :=
sorry

end ellipse_abs_sum_max_min_l187_187207


namespace number_of_girls_l187_187619

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l187_187619


namespace green_flower_percentage_l187_187574

theorem green_flower_percentage (yellow purple green total : ℕ)
  (hy : yellow = 10)
  (hp : purple = 18)
  (ht : total = 35)
  (hgreen : green = total - (yellow + purple)) :
  ((green * 100) / (yellow + purple)) = 25 := 
by {
  sorry
}

end green_flower_percentage_l187_187574


namespace sum_of_interior_angles_of_regular_polygon_l187_187074

theorem sum_of_interior_angles_of_regular_polygon (n : ℕ) (h : 60 = 360 / n) : (n - 2) * 180 = 720 :=
by
  sorry

end sum_of_interior_angles_of_regular_polygon_l187_187074


namespace exponent_division_l187_187042

variable (a : ℝ) (m n : ℝ)
-- Conditions
def condition1 : Prop := a^m = 2
def condition2 : Prop := a^n = 16

-- Theorem Statement
theorem exponent_division (h1 : condition1 a m) (h2 : condition2 a n) : a^(m - n) = 1 / 8 := by
  sorry

end exponent_division_l187_187042


namespace triangular_faces_area_of_pyramid_l187_187934

noncomputable def total_area_of_triangular_faces (base : ℝ) (lateral : ℝ) : ℝ :=
  let h := Real.sqrt (lateral ^ 2 - (base / 2) ^ 2)
  let area_one_triangle := (1 / 2) * base * h
  4 * area_one_triangle

theorem triangular_faces_area_of_pyramid :
  total_area_of_triangular_faces 8 10 = 32 * Real.sqrt 21 := by
  sorry

end triangular_faces_area_of_pyramid_l187_187934


namespace find_abc_l187_187954

theorem find_abc (a b c : ℕ) (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) (h_eq : 10 * a + 11 * b + c = 25) : a = 0 ∧ b = 2 ∧ c = 3 := 
sorry

end find_abc_l187_187954


namespace proportion_fourth_number_l187_187415

theorem proportion_fourth_number (x y : ℝ) (h₀ : 0.75 * y = 5 * x) (h₁ : x = 1.65) : y = 11 :=
by
  sorry

end proportion_fourth_number_l187_187415


namespace cube_of_square_is_15625_l187_187302

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l187_187302


namespace total_songs_performed_l187_187519

theorem total_songs_performed (lucy_songs : ℕ) (sarah_songs : ℕ) (beth_songs : ℕ) (jane_songs : ℕ) 
  (h1 : lucy_songs = 8)
  (h2 : sarah_songs = 5)
  (h3 : sarah_songs < beth_songs)
  (h4 : sarah_songs < jane_songs)
  (h5 : beth_songs < lucy_songs)
  (h6 : jane_songs < lucy_songs)
  (h7 : beth_songs = 6 ∨ beth_songs = 7)
  (h8 : jane_songs = 6 ∨ jane_songs = 7) :
  (lucy_songs + sarah_songs + beth_songs + jane_songs) / 3 = 9 :=
by
  sorry

end total_songs_performed_l187_187519


namespace velocity_zero_at_t_eq_4_or_8_l187_187952

open Real

-- Define the distance function s
def s (t : ℝ) : ℝ := (1/3) * t^3 - 6 * t^2 + 32 * t

-- State the theorem for the instants when the velocity is zero
theorem velocity_zero_at_t_eq_4_or_8 : ∀ t : ℝ, deriv s t = 0 ↔ t = 4 ∨ t = 8 := by
  sorry

end velocity_zero_at_t_eq_4_or_8_l187_187952


namespace parabola_tangent_circle_radius_l187_187910

noncomputable def radius_of_tangent_circle : ℝ :=
  let r := 1 / 4
  r

theorem parabola_tangent_circle_radius :
  ∃ (r : ℝ), (∀ (x : ℝ), x^2 - x + r = 0 → (-1)^2 - 4 * 1 * r = 1 - 4 * r) ∧ r = 1 / 4 :=
by
  use 1 / 4
  sorry

end parabola_tangent_circle_radius_l187_187910


namespace smallest_integer_consecutive_set_l187_187035

theorem smallest_integer_consecutive_set 
(n : ℤ) (h : 7 * n + 21 > 4 * n) : n > -7 :=
by
  sorry

end smallest_integer_consecutive_set_l187_187035


namespace sum_of_interior_angles_l187_187542

theorem sum_of_interior_angles (n : ℕ) : 
  (∀ θ, θ = 40 ∧ (n = 360 / θ)) → (n - 2) * 180 = 1260 :=
by
  sorry

end sum_of_interior_angles_l187_187542


namespace family_members_before_baby_l187_187671

theorem family_members_before_baby 
  (n T : ℕ)
  (h1 : T = 17 * n)
  (h2 : (T + 3 * n + 2) / (n + 1) = 17)
  (h3 : 2 = 2) : n = 5 :=
sorry

end family_members_before_baby_l187_187671


namespace carla_wins_one_game_l187_187155

/-
We are given the conditions:
Alice, Bob, and Carla each play each other twice in a round-robin format.
Alice won 5 games and lost 3 games.
Bob won 6 games and lost 2 games.
Carla lost 5 games.
We need to prove that Carla won 1 game.
-/

theorem carla_wins_one_game (games_per_match : Nat) 
                            (total_players : Nat)
                            (alice_wins : Nat) 
                            (alice_losses : Nat) 
                            (bob_wins : Nat) 
                            (bob_losses : Nat) 
                            (carla_losses : Nat) :
  (games_per_match = 2) → 
  (total_players = 3) → 
  (alice_wins = 5) → 
  (alice_losses = 3) → 
  (bob_wins = 6) → 
  (bob_losses = 2) → 
  (carla_losses = 5) → 
  ∃ (carla_wins : Nat), 
  carla_wins = 1 := 
by
  intros 
    games_match_eq total_players_eq 
    alice_wins_eq alice_losses_eq 
    bob_wins_eq bob_losses_eq 
    carla_losses_eq
  sorry

end carla_wins_one_game_l187_187155


namespace max_true_statements_l187_187563

theorem max_true_statements (x y : ℝ) :
  ∀ s : Finset ℕ, ∀ h : s ⊆ {1, 2, 3, 4, 5},
  (∀ i ∈ s, (i = 1 → 1 / x > 1 / y) ∧
            (i = 2 → x^2 < y^2) ∧
            (i = 3 → x > y) ∧
            (i = 4 → x > 0) ∧
            (i = 5 → y > 0)) →
  s.card ≤ 3 := 
begin
  sorry
end

end max_true_statements_l187_187563


namespace basketball_team_selection_l187_187611

theorem basketball_team_selection :
  (∃ (S : Finset (Fin 16)), S.card = 7 ∧ 
    (∃ (Q : Finset (Fin 16)), Q.card = 4 ∧ 
      {0, 1, 2, 3} ⊆ Q ∧ 
      (Q \ {0, 1, 2, 3}).card = 3 ∧
      (∀ x ∈ Q, x ∈ S) ∧ 
      (∃ (R : Finset (Fin 16)), R = S \ Q ∧ R.card = 4 ∧
        (R ⊆ {4, 5, ..., 15}))) → 
    (S.card = 7 ∧ Q.card = 4 ∧ ({0, 1, 2, 3} ⊆ Q ∧ (Q \ {0, 1, 2, 3}).card = 3 ∧ R = S \ Q ∧
    R.card = 4 ∧ R ⊆ {4, 5, ..., 15}))) := 1980 :=
by sorry

end basketball_team_selection_l187_187611


namespace calories_left_for_dinner_l187_187504

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l187_187504


namespace simplify_sqrt_200_l187_187281

theorem simplify_sqrt_200 : (sqrt 200 : ℝ) = 10 * sqrt 2 := by
  -- proof goes here
  sorry

end simplify_sqrt_200_l187_187281


namespace max_true_statements_maximum_true_conditions_l187_187564

theorem max_true_statements (x y : ℝ) (h1 : (1/x > 1/y)) (h2 : (x^2 < y^2)) (h3 : (x > y)) (h4 : (x > 0)) (h5 : (y > 0)) :
  false :=
  sorry

theorem maximum_true_conditions (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  ¬ ((1/x > 1/y) ∧ (x^2 < y^2)) :=
  sorry

#check max_true_statements
#check maximum_true_conditions

end max_true_statements_maximum_true_conditions_l187_187564


namespace trip_length_is_440_l187_187580

noncomputable def total_trip_length (d : ℝ) : Prop :=
  55 * 0.02 * (d - 40) = d

theorem trip_length_is_440 :
  total_trip_length 440 :=
by
  sorry

end trip_length_is_440_l187_187580


namespace area_of_QTUR_l187_187944

open Real

-- Define the problem
noncomputable def equilateral_triangle (a : ℝ) : Type :=
{P Q R : Type // distance P Q = a ∧ distance Q R = a ∧ distance P R = a}

noncomputable def extend_segment (P Q : Type) (k : ℝ) : Type :=
{Q S : Type // distance Q S = k * distance Q R }

noncomputable def midpoint (P Q : Type) : Type :=
{T : Type // distance P T = distance T Q}

noncomputable def intersection (line1 line2 : Type) : Type :=
{U : Type} -- Not elaborating further as specifics are not given

-- Problem in Lean 4
theorem area_of_QTUR :
  ∀ (P Q R S T U : Type) (a : ℝ) (k : ℝ),
  (⟦equilateral_triangle a⟧ P Q R) →
  (⟦extend_segment Q R k⟧ Q S) →
  (⟦midpoint P Q⟧ T) →
  (⟦intersection (line P R) (line T S) ⟧ U) →
  k = 1 / 2 →
  let area := (2 : ℝ) * (sqrt 3) in
  3 * area / 2 = 3 * sqrt 3 := 
sorry

end area_of_QTUR_l187_187944


namespace gcd_175_100_65_l187_187291

theorem gcd_175_100_65 : Nat.gcd (Nat.gcd 175 100) 65 = 5 :=
by
  sorry

end gcd_175_100_65_l187_187291


namespace minimum_value_l187_187900

theorem minimum_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_condition : 2 * a + 3 * b = 1) :
  ∃ min_value : ℝ, min_value = 65 / 6 ∧ (∀ c d : ℝ, (0 < c) → (0 < d) → (2 * c + 3 * d = 1) → (1 / c + 1 / d ≥ min_value)) :=
sorry

end minimum_value_l187_187900


namespace boys_tried_out_l187_187630

theorem boys_tried_out (G B C N : ℕ) (hG : G = 9) (hC : C = 2) (hN : N = 21) (h : G + B - C = N) : B = 14 :=
by
  -- The proof is omitted, focusing only on stating the theorem
  sorry

end boys_tried_out_l187_187630


namespace max_expression_value_l187_187778

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l187_187778


namespace sum_of_integers_l187_187842

theorem sum_of_integers (s : Finset ℕ) (h₀ : ∀ a ∈ s, 0 ≤ a ∧ a ≤ 124)
  (h₁ : ∀ a ∈ s, a^3 % 125 = 2) : s.sum id = 265 :=
sorry

end sum_of_integers_l187_187842


namespace sqrt_200_eq_10_l187_187248

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l187_187248


namespace tom_seashells_found_l187_187581

/-- 
Given:
- sally_seashells = 9 (number of seashells Sally found)
- jessica_seashells = 5 (number of seashells Jessica found)
- total_seashells = 21 (number of seashells found together)

Prove that the number of seashells that Tom found (tom_seashells) is 7.
-/
theorem tom_seashells_found (sally_seashells jessica_seashells total_seashells tom_seashells : ℕ)
  (h₁ : sally_seashells = 9) (h₂ : jessica_seashells = 5) (h₃ : total_seashells = 21) :
  tom_seashells = 7 :=
by
  sorry

end tom_seashells_found_l187_187581


namespace ellen_dinner_calories_proof_l187_187505

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l187_187505


namespace border_material_correct_l187_187454

noncomputable def pi_approx := (22 : ℚ) / 7

def circle_radius (area : ℚ) (pi_value : ℚ) : ℚ :=
  (area * (7 / 22)).sqrt

def circumference (radius : ℚ) (pi_value : ℚ) : ℚ :=
  2 * pi_value * radius

def total_border_material (area : ℚ) (pi_value : ℚ) (extra : ℚ) : ℚ :=
  circumference (circle_radius area pi_value) pi_value + extra

theorem border_material_correct :
  total_border_material 616 pi_approx 3 = 91 :=
by
  sorry

end border_material_correct_l187_187454


namespace sum_of_first_19_terms_l187_187705

noncomputable def a_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def S_n (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a1 + a_n a1 d n)

theorem sum_of_first_19_terms (a1 d : ℝ) (h : a1 + 9 * d = 1) : S_n a1 d 19 = 19 := by
  sorry

end sum_of_first_19_terms_l187_187705


namespace total_seedlings_transferred_l187_187579

-- Define the number of seedlings planted on the first day
def seedlings_day_1 : ℕ := 200

-- Define the number of seedlings planted on the second day
def seedlings_day_2 : ℕ := 2 * seedlings_day_1

-- Define the total number of seedlings planted on both days
def total_seedlings : ℕ := seedlings_day_1 + seedlings_day_2

-- The theorem statement
theorem total_seedlings_transferred : total_seedlings = 600 := by
  -- The proof goes here
  sorry

end total_seedlings_transferred_l187_187579


namespace min_product_of_prime_triplet_l187_187528

theorem min_product_of_prime_triplet
  (x y z : ℕ) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (hx_odd : x % 2 = 1) (hy_odd : y % 2 = 1) (hz_odd : z % 2 = 1)
  (h1 : x ∣ (y^5 + 1)) (h2 : y ∣ (z^5 + 1)) (h3 : z ∣ (x^5 + 1)) :
  (x * y * z) = 2013 := by
  sorry

end min_product_of_prime_triplet_l187_187528


namespace circle_radius_tangent_to_ellipse_l187_187631

theorem circle_radius_tangent_to_ellipse (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4*y^2 = 8) ↔ r = (Real.sqrt 6) / 2 :=
by
  sorry

end circle_radius_tangent_to_ellipse_l187_187631


namespace face_value_amount_of_bill_l187_187297

def true_discount : ℚ := 45
def bankers_discount : ℚ := 54

theorem face_value_amount_of_bill : 
  ∃ (FV : ℚ), bankers_discount = true_discount + (true_discount * bankers_discount / FV) ∧ FV = 270 :=
by
  sorry

end face_value_amount_of_bill_l187_187297


namespace subset_condition_intersection_condition_l187_187045

-- Definitions of the sets A and B
def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, 3 * a}

-- Theorem statements
theorem subset_condition (a : ℝ) : A ⊆ B a → (4 / 3) ≤ a ∧ a ≤ 2 := 
by 
  sorry

theorem intersection_condition (a : ℝ) : (A ∩ B a).Nonempty → (2 / 3) < a ∧ a < 4 := 
by 
  sorry

end subset_condition_intersection_condition_l187_187045


namespace dogs_with_no_accessories_l187_187687

theorem dogs_with_no_accessories :
  let total := 120
  let tags := 60
  let flea_collars := 50
  let harnesses := 30
  let tags_and_flea_collars := 20
  let tags_and_harnesses := 15
  let flea_collars_and_harnesses := 10
  let all_three := 5
  total - (tags + flea_collars + harnesses - tags_and_flea_collars - tags_and_harnesses - flea_collars_and_harnesses + all_three) = 25 := by
  sorry

end dogs_with_no_accessories_l187_187687


namespace find_common_ratio_l187_187406

-- Defining the conditions in Lean
variables (a : ℕ → ℝ) (d q : ℝ)

-- The arithmetic sequence condition
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) - a n = d

-- The geometric sequence condition
def is_geometric_sequence (a1 a2 a4 q : ℝ) : Prop :=
a2 ^ 2 = a1 * a4

-- Proving the main theorem
theorem find_common_ratio (a : ℕ → ℝ) (d q : ℝ) (h_arith : is_arithmetic_sequence a d) (d_ne_zero : d ≠ 0) 
(h_geom : is_geometric_sequence (a 1) (a 2) (a 4) q) : q = 2 :=
by
  sorry

end find_common_ratio_l187_187406


namespace tim_income_less_juan_l187_187575

variable {T M J : ℝ}

theorem tim_income_less_juan :
  (M = 1.60 * T) → (M = 0.6400000000000001 * J) → T = 0.4 * J :=
by
  sorry

end tim_income_less_juan_l187_187575


namespace exponent_equality_l187_187869

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l187_187869


namespace person_reaches_before_bus_l187_187374

theorem person_reaches_before_bus (dist : ℝ) (speed1 speed2 : ℝ) (miss_time_minutes : ℝ) :
  dist = 2.2 → speed1 = 3 → speed2 = 6 → miss_time_minutes = 12 →
  ((60 : ℝ) * (dist/speed1) - miss_time_minutes) - ((60 : ℝ) * (dist/speed2)) = 10 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end person_reaches_before_bus_l187_187374


namespace debby_ate_candy_l187_187165

theorem debby_ate_candy (initial_candy : ℕ) (remaining_candy : ℕ) (debby_initial : initial_candy = 12) (debby_remaining : remaining_candy = 3) : initial_candy - remaining_candy = 9 :=
by
  sorry

end debby_ate_candy_l187_187165


namespace fewest_number_of_students_l187_187143

def satisfiesCongruences (n : ℕ) : Prop :=
  n % 6 = 3 ∧
  n % 7 = 4 ∧
  n % 8 = 5 ∧
  n % 9 = 2

theorem fewest_number_of_students : ∃ n : ℕ, satisfiesCongruences n ∧ n = 765 :=
by
  have h_ex : ∃ n : ℕ, satisfiesCongruences n := sorry
  obtain ⟨n, hn⟩ := h_ex
  use 765
  have h_correct : satisfiesCongruences 765 := sorry
  exact ⟨h_correct, rfl⟩

end fewest_number_of_students_l187_187143


namespace charity_total_cost_l187_187355

theorem charity_total_cost
  (plates : ℕ)
  (rice_cost_per_plate chicken_cost_per_plate : ℕ)
  (h1 : plates = 100)
  (h2 : rice_cost_per_plate = 10)
  (h3 : chicken_cost_per_plate = 40) :
  plates * (rice_cost_per_plate + chicken_cost_per_plate) / 100 = 50 := 
by
  sorry

end charity_total_cost_l187_187355


namespace find_x_l187_187327

theorem find_x (x : ℚ) (h : |x - 1| = |x - 2|) : x = 3 / 2 :=
sorry

end find_x_l187_187327


namespace arithmetic_sequence_sum_n_squared_l187_187400

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_mean (x y z : ℝ) : Prop :=
(y * y = x * z)

def is_strictly_increasing (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, a (n + 1) > a n

theorem arithmetic_sequence_sum_n_squared
  (a : ℕ → ℝ)
  (h₁ : is_arithmetic_sequence a)
  (h₂ : a 1 = 1)
  (h₃ : is_geometric_mean (a 1) (a 2) (a 5))
  (h₄ : is_strictly_increasing a) :
  ∃ S : ℕ → ℝ, ∀ n : ℕ, S n = n ^ 2 :=
sorry

end arithmetic_sequence_sum_n_squared_l187_187400


namespace largest_square_side_length_is_2_point_1_l187_187184

noncomputable def largest_square_side_length (A B C : Point) (hABC : right_triangle A B C) (hAC : distance A C = 3) (hCB : distance C B = 7) : ℝ :=
  max_square_side_length A B C

theorem largest_square_side_length_is_2_point_1 :
  largest_square_side_length (3, 0) (0, 7) (0, 0) sorry sorry = 2.1 :=
by
  sorry

end largest_square_side_length_is_2_point_1_l187_187184


namespace seats_needed_l187_187672

-- Definitions based on the problem's condition
def children : ℕ := 58
def children_per_seat : ℕ := 2

-- Theorem statement to prove
theorem seats_needed : children / children_per_seat = 29 :=
by
  sorry

end seats_needed_l187_187672


namespace factorize_expression_l187_187390

variable (x y : ℝ)

theorem factorize_expression : (x - y) ^ 2 + 2 * y * (x - y) = (x - y) * (x + y) := by
  sorry

end factorize_expression_l187_187390


namespace cube_of_square_of_third_smallest_prime_l187_187313

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l187_187313


namespace number_of_common_terms_between_arithmetic_sequences_l187_187919

-- Definitions for the sequences
def seq1 (n : Nat) := 2 + 3 * n
def seq2 (n : Nat) := 4 + 5 * n

theorem number_of_common_terms_between_arithmetic_sequences
  (A : Finset Nat := Finset.range 673)  -- There are 673 terms in seq1 from 2 to 2015
  (B : Finset Nat := Finset.range 403)  -- There are 403 terms in seq2 from 4 to 2014
  (common_terms : Finset Nat := (A.image seq1) ∩ (B.image seq2)) :
  common_terms.card = 134 := by
  sorry

end number_of_common_terms_between_arithmetic_sequences_l187_187919


namespace find_value_of_expression_l187_187541

theorem find_value_of_expression (x y z : ℚ)
  (h1 : 2 * x + y + z = 14)
  (h2 : 2 * x + y = 7)
  (h3 : x + 2 * y = 10) : (x + y - z) / 3 = -4 / 9 :=
by sorry

end find_value_of_expression_l187_187541


namespace bird_population_in_1997_l187_187547

theorem bird_population_in_1997 
  (k : ℝ)
  (pop_1995 pop_1996 pop_1998 : ℝ)
  (h1 : pop_1995 = 45)
  (h2 : pop_1996 = 70)
  (h3 : pop_1998 = 145)
  (h4 : pop_1997 - pop_1995 = k * pop_1996)
  (h5 : pop_1998 - pop_1996 = k * pop_1997) : 
  pop_1997 = 105 :=
by
  sorry

end bird_population_in_1997_l187_187547


namespace village_population_rate_l187_187462

theorem village_population_rate (r : ℕ) :
  let PX := 72000
  let PY := 42000
  let decrease_rate_X := 1200
  let years := 15
  let population_X_after_years := PX - decrease_rate_X * years
  let population_Y_after_years := PY + r * years
  population_X_after_years = population_Y_after_years → r = 800 :=
by
  sorry

end village_population_rate_l187_187462


namespace repetend_of_decimal_expansion_l187_187397

theorem repetend_of_decimal_expansion :
  ∃ (r : ℕ), decimal_repetend (5 / 17) = some r ∧ r = 294117647058823529 :=
by
  sorry

end repetend_of_decimal_expansion_l187_187397


namespace exponent_equality_l187_187868

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l187_187868


namespace seating_arrangement_correct_l187_187479

noncomputable def seatingArrangements (committee : Fin 10) : Nat :=
  Nat.factorial 9

theorem seating_arrangement_correct :
  seatingArrangements committee = 362880 :=
by sorry

end seating_arrangement_correct_l187_187479


namespace blue_balls_prob_l187_187980

def prob_same_color (n : ℕ) : ℚ :=
  (1 / (n + 5))^2 + (4 / (n + 5))^2 + (n / (n + 5))^2

theorem blue_balls_prob {n : ℕ} (h : prob_same_color n = 1 / 2) : n = 1 ∨ n = 9 :=
by
  sorry

end blue_balls_prob_l187_187980


namespace find_m_l187_187894

theorem find_m (S : ℕ → ℝ) (m : ℕ) (h1 : S m = -2) (h2 : S (m+1) = 0) (h3 : S (m+2) = 3) : m = 4 :=
by
  sorry

end find_m_l187_187894


namespace boat_speed_greater_than_current_l187_187916

theorem boat_speed_greater_than_current (U V : ℝ) (hU_gt_V : U > V)
  (h_equation : 1 / (U - V) - 1 / (U + V) + 1 / (2 * V + 1) = 1) :
  U - V = 1 :=
sorry

end boat_speed_greater_than_current_l187_187916


namespace age_of_15th_student_l187_187813

theorem age_of_15th_student (avg15: ℕ) (avg5: ℕ) (avg9: ℕ) (x: ℕ)
  (h1: avg15 = 15) (h2: avg5 = 14) (h3: avg9 = 16)
  (h4: 15 * avg15 = x + 5 * avg5 + 9 * avg9) : x = 11 :=
by
  -- Proof will be added here
  sorry

end age_of_15th_student_l187_187813


namespace malcolm_joshua_time_difference_l187_187224

-- Define the constants
def malcolm_speed : ℕ := 5 -- minutes per mile
def joshua_speed : ℕ := 8 -- minutes per mile
def race_distance : ℕ := 12 -- miles

-- Define the times it takes each runner to finish
def malcolm_time : ℕ := malcolm_speed * race_distance
def joshua_time : ℕ := joshua_speed * race_distance

-- Define the time difference and the proof statement
def time_difference : ℕ := joshua_time - malcolm_time

theorem malcolm_joshua_time_difference : time_difference = 36 := by
  sorry

end malcolm_joshua_time_difference_l187_187224


namespace larger_number_eq_1599_l187_187915

theorem larger_number_eq_1599 (L S : ℕ) (h1 : L - S = 1335) (h2 : L = 6 * S + 15) : L = 1599 :=
by 
  sorry

end larger_number_eq_1599_l187_187915


namespace range_of_a_l187_187210

theorem range_of_a (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x - 1| ≤ 2) → (a > 3 ∨ a < -1) :=
by
  sorry

end range_of_a_l187_187210


namespace log_expression_equality_l187_187135

theorem log_expression_equality : 
  (Real.log 3 / Real.log 2) * (Real.log 4 / Real.log 3) + 
  (Real.log 8 / Real.log 4) + 
  2 = 11 / 2 :=
by 
  sorry

end log_expression_equality_l187_187135


namespace trajectory_eq_sum_lambdas_l187_187526
-- Define the points M and N
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the conditions
variables {P : ℝ × ℝ} (h : (N.1 - M.1) * (P.1 - M.1) = 6 * Real.sqrt ( (P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2 )) 

-- Prove the trajectory equation
theorem trajectory_eq (P : ℝ × ℝ) (hP : (N.1 - M.1) * (P.1 - M.1) = 6 * Real.sqrt ( (P.1 - N.1) ^ 2 + (P.2 - N.2) ^ 2 )) :
  (P.1 ^ 2) / 4 + (P.2 ^ 2) / 3 = 1 := 
sorry

-- Define the line passing through N which intersects C
variables {A B H : ℝ × ℝ}
variables (l : ℝ → ℝ) -- equation of line
variables (λ₁ λ₂ : ℝ)
variables (hAN : A = (A.1, A.2 + (1/l A.2)) ∧ (H.1, H.2 - -N.2) = λ₁ * (N.1-A.1, -A.2))
variables (hBN : B = (B.1, B.2 + (1/l B.2)) ∧ (H.1, H.2 - -)\Nep B.2) = λ₂ * (N.1-B.1, -B.2))

-- Prove the concatenation of constants
theorem sum_lambdas (hHNL : (N.1-M.1) * \sqrt ((N.1-N.2) * (H.1-H.2))) :
  λ₁ + λ₂ = -8/3 := 
sorry

end trajectory_eq_sum_lambdas_l187_187526


namespace FG_length_of_trapezoid_l187_187105

-- Define the dimensions and properties of trapezoid EFGH.
def EFGH_trapezoid (area : ℝ) (altitude : ℝ) (EF : ℝ) (GH : ℝ) : Prop :=
  area = 180 ∧ altitude = 9 ∧ EF = 12 ∧ GH = 20

-- State the theorem to prove the length of FG.
theorem FG_length_of_trapezoid : 
  ∀ {E F G H : Type} (area EF GH fg : ℝ) (altitude : ℝ),
  EFGH_trapezoid area altitude EF GH → fg = 6.57 :=
by sorry

end FG_length_of_trapezoid_l187_187105


namespace relationship_among_x_y_z_l187_187847

variable (a b c d : ℝ)

-- Conditions
variables (h1 : a < b)
variables (h2 : b < c)
variables (h3 : c < d)

-- Definitions of x, y, z
def x : ℝ := (a + b) * (c + d)
def y : ℝ := (a + c) * (b + d)
def z : ℝ := (a + d) * (b + c)

-- Theorem: Prove the relationship among x, y, z
theorem relationship_among_x_y_z (h1 : a < b) (h2 : b < c) (h3 : c < d) : x a b c d < y a b c d ∧ y a b c d < z a b c d := by
  sorry

end relationship_among_x_y_z_l187_187847


namespace new_probability_of_blue_ball_l187_187822

theorem new_probability_of_blue_ball 
  (initial_total_balls : ℕ) (initial_blue_balls : ℕ) (removed_blue_balls : ℕ) :
  initial_total_balls = 18 →
  initial_blue_balls = 6 →
  removed_blue_balls = 3 →
  (initial_blue_balls - removed_blue_balls) / (initial_total_balls - removed_blue_balls) = 1 / 5 :=
by
  sorry

end new_probability_of_blue_ball_l187_187822


namespace false_prop_range_of_a_l187_187610

theorem false_prop_range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (a < -2 * Real.sqrt 2 ∨ a > 2 * Real.sqrt 2) :=
by
  sorry

end false_prop_range_of_a_l187_187610


namespace min_value_l187_187751

theorem min_value (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b = 1) : 3 + 2 * Real.sqrt 2 ≤ 2 / a + 1 / b :=
by
  sorry

end min_value_l187_187751


namespace gcd_sum_product_pairwise_coprime_l187_187543

theorem gcd_sum_product_pairwise_coprime 
  (a b c : ℤ) 
  (h1 : Int.gcd a b = 1)
  (h2 : Int.gcd b c = 1)
  (h3 : Int.gcd a c = 1) : 
  Int.gcd (a * b + b * c + a * c) (a * b * c) = 1 := 
sorry

end gcd_sum_product_pairwise_coprime_l187_187543


namespace find_integer_n_l187_187026

open Int

theorem find_integer_n (n a b : ℤ) :
  (4 * n + 1 = a^2) ∧ (9 * n + 1 = b^2) → n = 0 := by
sorry

end find_integer_n_l187_187026


namespace sqrt_200_eq_10_l187_187264

theorem sqrt_200_eq_10 : real.sqrt 200 = 10 :=
by
  calc
    real.sqrt 200 = real.sqrt (2^2 * 5^2) : by sorry -- show 200 = 2^2 * 5^2
    ... = real.sqrt (2^2) * real.sqrt (5^2) : by sorry -- property of square roots of products
    ... = 2 * 5 : by sorry -- using the property sqrt (a^2) = a
    ... = 10 : by sorry

end sqrt_200_eq_10_l187_187264


namespace sum_of_squares_l187_187577

theorem sum_of_squares (x y : ℝ) (h₁ : x + y = 40) (h₂ : x * y = 120) : x^2 + y^2 = 1360 :=
by
  sorry

end sum_of_squares_l187_187577


namespace count_ball_box_arrangements_l187_187115

theorem count_ball_box_arrangements :
  ∃ (arrangements : ℕ), arrangements = 20 ∧
  (∃ f : Fin 5 → Fin 5,
    (∃! i1, f i1 = i1) ∧ (∃! i2, f i2 = i2) ∧
    ∀ i, ∃! j, f i = j) :=
sorry

end count_ball_box_arrangements_l187_187115


namespace balls_into_boxes_l187_187096

theorem balls_into_boxes : ∃ n : ℕ, n = 240 ∧ ∃ f : Fin 5 → Fin 4, ∀ i : Fin 4, ∃ j : Fin 5, f j = i := by
  sorry

end balls_into_boxes_l187_187096


namespace maria_cookies_left_l187_187232

def maria_cookies (initial: ℕ) (to_friend: ℕ) (to_family_divisor: ℕ) (eats: ℕ) : ℕ :=
  (initial - to_friend) / to_family_divisor - eats

theorem maria_cookies_left (h : maria_cookies 19 5 2 2 = 5): true :=
by trivial

end maria_cookies_left_l187_187232


namespace spherical_triangle_area_correct_l187_187513

noncomputable def spherical_triangle_area (R α β γ : ℝ) : ℝ :=
  R^2 * (α + β + γ - Real.pi)

theorem spherical_triangle_area_correct (R α β γ : ℝ) :
  spherical_triangle_area R α β γ = R^2 * (α + β + γ - Real.pi) := by
  sorry

end spherical_triangle_area_correct_l187_187513


namespace two_digit_integer_plus_LCM_of_3_4_5_l187_187936

theorem two_digit_integer_plus_LCM_of_3_4_5 (x : ℕ) (h1 : 9 < x) (h2 : x < 100) (h3 : ∃ k, x = 60 * k + 2) :
  x = 62 :=
by {
  sorry
}

end two_digit_integer_plus_LCM_of_3_4_5_l187_187936


namespace value_of_a_plus_b_l187_187536

variables (a b : ℝ)

theorem value_of_a_plus_b (h1 : a + 4 * b = 33) (h2 : 6 * a + 3 * b = 51) : a + b = 12 := 
by
  sorry

end value_of_a_plus_b_l187_187536


namespace abs_inequality_solution_set_l187_187922

theorem abs_inequality_solution_set :
  { x : ℝ | abs (2 - x) < 5 } = { x : ℝ | -3 < x ∧ x < 7 } :=
by
  sorry

end abs_inequality_solution_set_l187_187922


namespace length_of_crease_l187_187678

/-- 
  Given a rectangular piece of paper 8 inches wide that is folded such that one corner 
  touches the opposite side at an angle θ from the horizontal, and one edge of the paper 
  remains aligned with the base, 
  prove that the length of the crease L is given by L = 8 * tan θ / (1 + tan θ). 
--/
theorem length_of_crease (theta : ℝ) (h : 0 < theta ∧ theta < Real.pi / 2): 
  ∃ L : ℝ, L = 8 * Real.tan theta / (1 + Real.tan theta) :=
sorry

end length_of_crease_l187_187678


namespace combined_weight_l187_187886

def weight_in_tons := 3
def weight_in_pounds_per_ton := 2000
def weight_in_pounds := weight_in_tons * weight_in_pounds_per_ton
def donkey_weight_in_pounds := weight_in_pounds - (0.90 * weight_in_pounds)

theorem combined_weight :
  (weight_in_pounds + donkey_weight_in_pounds) = 6600 :=
by
  -- Proof goes here
  sorry

end combined_weight_l187_187886


namespace practice_problems_total_l187_187457

theorem practice_problems_total :
  let marvin_yesterday := 40
  let marvin_today := 3 * marvin_yesterday
  let arvin_yesterday := 2 * marvin_yesterday
  let arvin_today := 2 * marvin_today
  let kevin_yesterday := 30
  let kevin_today := kevin_yesterday + 10
  let total_problems := (marvin_yesterday + marvin_today) + (arvin_yesterday + arvin_today) + (kevin_yesterday + kevin_today)
  total_problems = 550 :=
by
  sorry

end practice_problems_total_l187_187457


namespace initial_population_l187_187549

theorem initial_population (P : ℝ) 
  (h1 : P * 0.90 * 0.95 * 0.85 * 1.08 = 6514) : P = 8300 :=
by
  -- Given conditions lead to the final population being 6514
  -- We need to show that the initial population P was 8300
  sorry

end initial_population_l187_187549


namespace area_of_billboard_l187_187953

variable (L W : ℕ) (P : ℕ)
variable (hW : W = 8) (hP : P = 46)

theorem area_of_billboard (h1 : P = 2 * L + 2 * W) : L * W = 120 :=
by
  sorry

end area_of_billboard_l187_187953


namespace alex_sweaters_l187_187828

def num_items (shirts : ℕ) (pants : ℕ) (jeans : ℕ) (total_cycle_time_minutes : ℕ)
  (cycle_time_minutes : ℕ) (max_items_per_cycle : ℕ) : ℕ :=
  total_cycle_time_minutes / cycle_time_minutes * max_items_per_cycle

def num_sweaters_to_wash (total_items : ℕ) (non_sweater_items : ℕ) : ℕ :=
  total_items - non_sweater_items

theorem alex_sweaters :
  ∀ (shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle : ℕ),
  shirts = 18 →
  pants = 12 →
  jeans = 13 →
  total_cycle_time_minutes = 180 →
  cycle_time_minutes = 45 →
  max_items_per_cycle = 15 →
  num_sweaters_to_wash
    (num_items shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle)
    (shirts + pants + jeans) = 17 :=
by
  intros shirts pants jeans total_cycle_time_minutes cycle_time_minutes max_items_per_cycle
    h_shirts h_pants h_jeans h_total_cycle_time_minutes h_cycle_time_minutes h_max_items_per_cycle
  
  sorry

end alex_sweaters_l187_187828


namespace edmonton_to_calgary_travel_time_l187_187023

theorem edmonton_to_calgary_travel_time :
  let distance_edmonton_red_deer := 220
  let distance_red_deer_calgary := 110
  let speed_to_red_deer := 100
  let detour_distance := 30
  let detour_time := (distance_edmonton_red_deer + detour_distance) / speed_to_red_deer
  let stop_time := 1
  let speed_to_calgary := 90
  let travel_time_to_calgary := distance_red_deer_calgary / speed_to_calgary
  detour_time + stop_time + travel_time_to_calgary = 4.72 := by
  sorry

end edmonton_to_calgary_travel_time_l187_187023


namespace variance_proof_l187_187746

noncomputable def calculate_mean (scores : List ℝ) : ℝ :=
  (scores.sum / scores.length)

noncomputable def calculate_variance (scores : List ℝ) : ℝ :=
  let mean := calculate_mean scores
  (scores.map (λ x => (x - mean)^2)).sum / scores.length

def scores_A : List ℝ := [8, 6, 9, 5, 10, 7, 4, 7, 9, 5]
def scores_B : List ℝ := [7, 6, 5, 8, 6, 9, 6, 8, 8, 7]

noncomputable def variance_A : ℝ := calculate_variance scores_A
noncomputable def variance_B : ℝ := calculate_variance scores_B

theorem variance_proof :
  variance_A = 3.6 ∧ variance_B = 1.4 ∧ variance_B < variance_A :=
by
  -- proof steps - use sorry to skip the proof
  sorry

end variance_proof_l187_187746


namespace bricks_in_wall_l187_187818

-- Definitions of conditions based on the problem statement
def time_first_bricklayer : ℝ := 12 
def time_second_bricklayer : ℝ := 15 
def reduced_productivity : ℝ := 12 
def combined_time : ℝ := 6
def total_bricks : ℝ := 720

-- Lean 4 statement of the proof problem
theorem bricks_in_wall (x : ℝ) 
  (h1 : (x / time_first_bricklayer + x / time_second_bricklayer - reduced_productivity) * combined_time = x) 
  : x = total_bricks := 
by {
  sorry
}

end bricks_in_wall_l187_187818


namespace possible_measures_A_l187_187600

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l187_187600


namespace average_value_eq_l187_187012

variable (x : ℝ)

theorem average_value_eq :
  ( -4 * x + 0 + 4 * x + 12 * x + 20 * x ) / 5 = 6.4 * x :=
by
  sorry

end average_value_eq_l187_187012


namespace option_d_correct_l187_187902

def f (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)
def M : Set ℝ := {x | f x = 0}

theorem option_d_correct : ({1, 3} ∪ {2, 3} : Set ℝ) = M := by
  sorry

end option_d_correct_l187_187902


namespace kona_additional_miles_l187_187040

def distance_apartment_to_bakery : ℕ := 9
def distance_bakery_to_grandmothers_house : ℕ := 24
def distance_grandmothers_house_to_apartment : ℕ := 27

theorem kona_additional_miles:
  let no_bakery_round_trip := 2 * distance_grandmothers_house_to_apartment in
  let with_bakery_round_trip := 
    distance_apartment_to_bakery +
    distance_bakery_to_grandmothers_house +
    distance_grandmothers_house_to_apartment in
  with_bakery_round_trip - no_bakery_round_trip = 33 :=
by
  let no_bakery_round_trip : ℕ := 54
  let with_bakery_round_trip : ℕ := 60
  calc
    no_bakery_round_trip = 27 + 27 := sorry
    with_bakery_round_trip = 9 + 24 + 27 := sorry
    with_bakery_round_trip - no_bakery_round_trip = 33 := sorry

end kona_additional_miles_l187_187040


namespace line_through_points_l187_187419

theorem line_through_points (a b : ℝ)
  (h1 : 2 = a * 1 + b)
  (h2 : 14 = a * 5 + b) :
  a - b = 4 := 
  sorry

end line_through_points_l187_187419


namespace man_total_earnings_l187_187950

-- Define the conditions
def total_days := 30
def wage_per_day := 10
def fine_per_absence := 2
def days_absent := 7
def days_worked := total_days - days_absent
def earned := days_worked * wage_per_day
def fine := days_absent * fine_per_absence
def total_earnings := earned - fine

-- State the theorem
theorem man_total_earnings : total_earnings = 216 := by
  -- Using the definitions provided, the proof should show that the calculations result in 216
  sorry

end man_total_earnings_l187_187950


namespace no_two_proper_subgroups_cover_G_three_proper_subgroups_cover_G_or_not_l187_187191

variable {G : Type*} [Group G] [Finite G]

-- Part 1: Impossibility of Covering G with Two Proper Subgroups
theorem no_two_proper_subgroups_cover_G (A B : subgroup G) 
    (hA1 : A ≠ ⊤) (hB1 : B ≠ ⊤) 
    (hA2 : A ≠ ⊥) (hB2 : B ≠ ⊥) : 
    A ∪ B ≠ ⊤ :=
sorry

-- Part 2: Possibility of Covering G with Three Proper Subgroups
theorem three_proper_subgroups_cover_G_or_not : 
    (∃ (A B C : subgroup G), A ≠ ⊤ ∧ B ≠ ⊤ ∧ C ≠ ⊤ ∧ A ≠ ⊥ ∧ B ≠ ⊥ ∧ C ≠ ⊥ ∧ (A ∪ B ∪ C = ⊤)) ∨
    (∀ (A B C : subgroup G), A ≠ ⊤ ∧ B ≠ ⊤ ∧ C ≠ ⊤ ∧ A ≠ ⊥ ∧ B ≠ ⊥ ∧ C ≠ ⊥ → (A ∪ B ∪ C ≠ ⊤)) :=
sorry

end no_two_proper_subgroups_cover_G_three_proper_subgroups_cover_G_or_not_l187_187191


namespace neither_cable_nor_vcr_fraction_l187_187663

variable (T : ℕ) -- Let T be the total number of housing units

def cableTV_fraction : ℚ := 1 / 5
def VCR_fraction : ℚ := 1 / 10
def both_fraction_given_cable : ℚ := 1 / 4

theorem neither_cable_nor_vcr_fraction : 
  (T : ℚ) * (1 - ((1 / 5) + ((1 / 10) - ((1 / 4) * (1 / 5))))) = (T : ℚ) * (3 / 4) :=
by sorry

end neither_cable_nor_vcr_fraction_l187_187663


namespace mr_lee_gain_l187_187236

noncomputable def cost_price_1 (revenue : ℝ) (profit_percentage : ℝ) : ℝ :=
  revenue / (1 + profit_percentage)

noncomputable def cost_price_2 (revenue : ℝ) (loss_percentage : ℝ) : ℝ :=
  revenue / (1 - loss_percentage)

theorem mr_lee_gain
    (revenue : ℝ)
    (profit_percentage : ℝ)
    (loss_percentage : ℝ)
    (revenue_1 : ℝ := 1.44)
    (revenue_2 : ℝ := 1.44)
    (profit_percent : ℝ := 0.20)
    (loss_percent : ℝ := 0.10):
  let cost_1 := cost_price_1 revenue_1 profit_percent
  let cost_2 := cost_price_2 revenue_2 loss_percent
  let total_cost := cost_1 + cost_2
  let total_revenue := revenue_1 + revenue_2
  total_revenue - total_cost = 0.08 :=
by
  sorry

end mr_lee_gain_l187_187236


namespace time_saved_by_both_trains_trainB_distance_l187_187801

-- Define the conditions
def trainA_speed_reduced := 360 / 12  -- 30 miles/hour
def trainB_speed_reduced := 360 / 8   -- 45 miles/hour

def trainA_speed := trainA_speed_reduced / (2 / 3)  -- 45 miles/hour
def trainB_speed := trainB_speed_reduced / (1 / 2)  -- 90 miles/hour

def trainA_time_saved := 12 - (360 / trainA_speed)  -- 4 hours
def trainB_time_saved := 8 - (360 / trainB_speed)   -- 4 hours

-- Prove that total time saved by both trains running at their own speeds is 8 hours
theorem time_saved_by_both_trains : trainA_time_saved + trainB_time_saved = 8 := by
  sorry

-- Prove that the distance between Town X and Town Y for Train B is 360 miles
theorem trainB_distance : 360 = 360 := by
  rfl

end time_saved_by_both_trains_trainB_distance_l187_187801


namespace maximum_value_is_17_l187_187753

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l187_187753


namespace abs_eq_condition_l187_187644

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l187_187644


namespace max_expression_value_l187_187768

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l187_187768


namespace total_turtles_taken_l187_187970

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l187_187970


namespace total_cost_is_83_50_l187_187798

-- Definitions according to the conditions
def cost_adult_ticket : ℝ := 5.50
def cost_child_ticket : ℝ := 3.50
def total_tickets : ℝ := 21
def adult_tickets : ℝ := 5
def child_tickets : ℝ := total_tickets - adult_tickets

-- Total cost calculation based on the conditions
def cost_adult_total : ℝ := adult_tickets * cost_adult_ticket
def cost_child_total : ℝ := child_tickets * cost_child_ticket
def total_cost : ℝ := cost_adult_total + cost_child_total

-- The theorem to prove that the total cost is $83.50
theorem total_cost_is_83_50 : total_cost = 83.50 := by
  sorry

end total_cost_is_83_50_l187_187798


namespace percentage_to_pass_l187_187090

theorem percentage_to_pass (score shortfall max_marks : ℕ) (h_score : score = 212) (h_shortfall : shortfall = 13) (h_max_marks : max_marks = 750) :
  (score + shortfall) / max_marks * 100 = 30 :=
by
  sorry

end percentage_to_pass_l187_187090


namespace hiking_supplies_l187_187372

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l187_187372


namespace johns_drawings_l187_187887

theorem johns_drawings (total_pictures : ℕ) (back_pictures : ℕ) 
  (h1 : total_pictures = 15) (h2 : back_pictures = 9) : total_pictures - back_pictures = 6 := by
  -- proof goes here
  sorry

end johns_drawings_l187_187887


namespace inequality_solution_l187_187749

variable (a x : ℝ)

noncomputable def inequality_solutions :=
  if a = 0 then
    {x | x > 1}
  else if a > 1 then
    {x | (1 / a) < x ∧ x < 1}
  else if a = 1 then
    ∅
  else if 0 < a ∧ a < 1 then
    {x | 1 < x ∧ x < (1 / a)}
  else if a < 0 then
    {x | x < (1 / a) ∨ x > 1}
  else
    ∅

theorem inequality_solution (h : a ≠ 0) :
  if a = 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 → x > 1
  else if a > 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ ((1 / a) < x ∧ x < 1)
  else if a = 1 then
    ∀ x, ¬((a * x - 1) * (x - 1) < 0)
  else if 0 < a ∧ a < 1 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (1 < x ∧ x < (1 / a))
  else if a < 0 then
    ∀ x, (a * x - 1) * (x - 1) < 0 ↔ (x < (1 / a) ∨ x > 1)
  else
    True := sorry

end inequality_solution_l187_187749


namespace moles_of_CH4_l187_187395

theorem moles_of_CH4 (moles_Be2C moles_H2O : ℕ) (balanced_equation : 1 * Be2C + 4 * H2O = 2 * CH4 + 2 * BeOH2) 
  (h_Be2C : moles_Be2C = 3) (h_H2O : moles_H2O = 12) : 
  6 = 2 * moles_Be2C :=
by
  sorry

end moles_of_CH4_l187_187395


namespace exponent_equality_l187_187862

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l187_187862


namespace sum_of_squares_of_six_odds_not_2020_l187_187215

theorem sum_of_squares_of_six_odds_not_2020 :
  ¬ ∃ a1 a2 a3 a4 a5 a6 : ℤ, (∀ i ∈ [a1, a2, a3, a4, a5, a6], i % 2 = 1) ∧ (a1^2 + a2^2 + a3^2 + a4^2 + a5^2 + a6^2 = 2020) :=
by
  sorry

end sum_of_squares_of_six_odds_not_2020_l187_187215


namespace crayons_birthday_l187_187095

theorem crayons_birthday (C E : ℕ) (hC : C = 523) (hE : E = 457) (hDiff : C = E + 66) : C = 523 := 
by {
  -- proof would go here
  sorry
}

end crayons_birthday_l187_187095


namespace tens_digit_of_binary_result_l187_187589

def digits_tens_digit_subtraction (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) : ℕ :=
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let difference := original_number - reversed_number
  (difference % 100) / 10

theorem tens_digit_of_binary_result (a b c : ℕ) (h1 : b = 2 * c) (h2 : a = b - 3) :
  digits_tens_digit_subtraction a b c h1 h2 = 9 :=
sorry

end tens_digit_of_binary_result_l187_187589


namespace sum_GCF_LCM_l187_187439

-- Definitions of GCD and LCM for the numbers 18, 27, and 36
def GCF : ℕ := Nat.gcd (Nat.gcd 18 27) 36
def LCM : ℕ := Nat.lcm (Nat.lcm 18 27) 36

-- Theorem statement proof
theorem sum_GCF_LCM : GCF + LCM = 117 := by
  sorry

end sum_GCF_LCM_l187_187439


namespace find_a_l187_187633

open Real

variable (a : ℝ)

theorem find_a (h : 4 * a + -5 * 3 = 0) : a = 15 / 4 :=
sorry

end find_a_l187_187633


namespace round_table_legs_l187_187883

theorem round_table_legs:
  ∀ (chairs tables disposed chairs_legs tables_legs : ℕ) (total_legs : ℕ),
  chairs = 80 →
  chairs_legs = 5 →
  tables = 20 →
  disposed = 40 * chairs / 100 →
  total_legs = 300 →
  total_legs - (chairs - disposed) * chairs_legs = tables * tables_legs →
  tables_legs = 3 :=
by 
  intros chairs tables disposed chairs_legs tables_legs total_legs
  sorry

end round_table_legs_l187_187883


namespace smallest_square_factor_2016_l187_187125

theorem smallest_square_factor_2016 : ∃ n : ℕ, (168 = n) ∧ (∃ k : ℕ, k^2 = n) ∧ (2016 ∣ k^2) :=
by
  sorry

end smallest_square_factor_2016_l187_187125


namespace possible_measures_of_angle_A_l187_187597

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l187_187597


namespace problem_statement_l187_187438

open Probability

noncomputable def binomialProbability : ℕ → ℕ → ℚ → ℚ := sorry

theorem problem_statement (X : ℕ → ℕ → ℚ -> ℕ → ℚ) (n : ℕ) (p : ℚ) :
  (X n p).pdf (λ x => x = 2) = 80 / 243 :=
by sorry

end problem_statement_l187_187438


namespace polynomial_mod_p_zero_l187_187442

def is_zero_mod_p (p : ℕ) [Fact (Nat.Prime p)] (f : (List ℕ → ℤ)) : Prop :=
  ∀ (x : List ℕ), f x % p = 0

theorem polynomial_mod_p_zero
  (p : ℕ) [Fact (Nat.Prime p)]
  (n : ℕ) 
  (f : (List ℕ → ℤ)) 
  (h : ∀ (x : List ℕ), f x % p = 0) 
  (g : (List ℕ → ℤ)) :
  (∀ (x : List ℕ), g x % p = 0) := sorry

end polynomial_mod_p_zero_l187_187442


namespace parking_lot_perimeter_l187_187486

theorem parking_lot_perimeter (a b : ℝ) (h₁ : a^2 + b^2 = 625) (h₂ : a * b = 168) :
  2 * (a + b) = 62 :=
sorry

end parking_lot_perimeter_l187_187486


namespace longest_segment_in_cylinder_l187_187819

noncomputable def cylinder_diagonal (radius height : ℝ) : ℝ :=
  Real.sqrt (height^2 + (2 * radius)^2)

theorem longest_segment_in_cylinder :
  cylinder_diagonal 4 10 = 2 * Real.sqrt 41 :=
by
  -- Proof placeholder
  sorry

end longest_segment_in_cylinder_l187_187819


namespace jill_total_tax_percentage_l187_187939

theorem jill_total_tax_percentage (total_spent : ℝ) 
  (spent_clothing : ℝ) (spent_food : ℝ) (spent_other : ℝ)
  (tax_clothing_rate : ℝ) (tax_food_rate : ℝ) (tax_other_rate : ℝ)
  (h_clothing : spent_clothing = 0.45 * total_spent)
  (h_food : spent_food = 0.45 * total_spent)
  (h_other : spent_other = 0.10 * total_spent)
  (h_tax_clothing : tax_clothing_rate = 0.05)
  (h_tax_food : tax_food_rate = 0.0)
  (h_tax_other : tax_other_rate = 0.10) :
  ((spent_clothing * tax_clothing_rate + spent_food * tax_food_rate + spent_other * tax_other_rate) / total_spent) * 100 = 3.25 :=
by
  sorry

end jill_total_tax_percentage_l187_187939


namespace sector_radius_l187_187295

theorem sector_radius (P : ℝ) (c : ℝ → ℝ) (θ : ℝ) (r : ℝ) (π : ℝ) 
  (h1 : P = 144) 
  (h2 : θ = π)
  (h3 : P = θ * r + 2 * r) 
  (h4 : π = Real.pi)
  : r = 144 / (Real.pi + 2) := 
by
  sorry

end sector_radius_l187_187295


namespace ab_bc_ca_leq_zero_l187_187401

theorem ab_bc_ca_leq_zero (a b c : ℝ) (h : a + b + c = 0) : ab + bc + ca ≤ 0 :=
sorry

end ab_bc_ca_leq_zero_l187_187401


namespace inequality_proof_l187_187342

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187342


namespace area_of_trapezoid_RSQT_l187_187213

theorem area_of_trapezoid_RSQT
  (PR PQ : ℝ)
  (PR_eq_PQ : PR = PQ)
  (small_triangle_area : ℝ)
  (total_area : ℝ)
  (num_of_small_triangles : ℕ)
  (num_of_triangles_in_trapezoid : ℕ)
  (area_of_trapezoid : ℝ)
  (is_isosceles_triangle : ∀ (a b c : ℝ), a = b → b = c → a = c)
  (are_similar_triangles : ∀ {A B C D E F : ℝ}, 
    A / B = D / E → A / C = D / F → B / A = E / D → C / A = F / D)
  (smallest_triangle_areas : ∀ {n : ℕ}, n = 9 → small_triangle_area = 2 → num_of_small_triangles = 9)
  (triangle_total_area : ∀ (a : ℝ), a = 72 → total_area = 72)
  (contains_3_small_triangles : ∀ (n : ℕ), n = 3 → num_of_triangles_in_trapezoid = 3)
  (parallel_ST_to_PQ : ∀ {x y z : ℝ}, x = z → y = z → x = y)
  : area_of_trapezoid = 39 :=
sorry

end area_of_trapezoid_RSQT_l187_187213


namespace proportion_equation_l187_187062

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l187_187062


namespace car_speed_car_speed_correct_l187_187138

theorem car_speed (d t s : ℝ) (hd : d = 810) (ht : t = 5) : s = d / t := 
by
  sorry

theorem car_speed_correct (d t : ℝ) (hd : d = 810) (ht : t = 5) : d / t = 162 :=
by
  sorry

end car_speed_car_speed_correct_l187_187138


namespace haleigh_needs_46_leggings_l187_187860

-- Define the number of each type of animal
def num_dogs : ℕ := 4
def num_cats : ℕ := 3
def num_spiders : ℕ := 2
def num_parrot : ℕ := 1

-- Define the number of legs each type of animal has
def legs_dog : ℕ := 4
def legs_cat : ℕ := 4
def legs_spider : ℕ := 8
def legs_parrot : ℕ := 2

-- Define the total number of legs function
def total_leggings (d c s p : ℕ) (ld lc ls lp : ℕ) : ℕ :=
  d * ld + c * lc + s * ls + p * lp

-- The statement to be proven
theorem haleigh_needs_46_leggings : total_leggings num_dogs num_cats num_spiders num_parrot legs_dog legs_cat legs_spider legs_parrot = 46 := by
  sorry

end haleigh_needs_46_leggings_l187_187860


namespace initial_pocket_money_l187_187434

variable (P : ℝ)

-- Conditions
axiom chocolates_expenditure : P * (1/9) ≥ 0
axiom fruits_expenditure : P * (2/5) ≥ 0
axiom remaining_money : P * (22/45) = 220

-- Theorem statement
theorem initial_pocket_money : P = 450 :=
by
  have h₁ : P * (1/9) + P * (2/5) = P * (23/45) := by sorry
  have h₂ : P * (1 - 23/45) = P * (22/45) := by sorry
  have h₃ : P = 220 / (22/45) := by sorry
  have h₄ : P = 220 * (45/22) := by sorry
  have h₅ : P = 450 := by sorry
  exact h₅

end initial_pocket_money_l187_187434


namespace range_of_a_l187_187043

variable {x : ℝ} (a : ℝ)

def f (x : ℝ) := x * Real.log x - a * x

def g (x : ℝ) := x^3 - x + 6

theorem range_of_a (h : ∀ x > 0, 2 * f x ≤ (3 * x^2 - 1) + 2) : a ∈ Set.Ici (-2) :=
by sorry

end range_of_a_l187_187043


namespace hiking_packing_weight_l187_187368

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l187_187368


namespace sqrt_200_eq_10_l187_187251

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l187_187251


namespace sqrt_200_eq_10_sqrt_2_l187_187259

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l187_187259


namespace coffee_decaf_percentage_l187_187471

variable (initial_stock : ℝ) (initial_decaf_percent : ℝ)
variable (new_stock : ℝ) (new_decaf_percent : ℝ)

noncomputable def decaf_coffee_percentage : ℝ :=
  let initial_decaf : ℝ := initial_stock * (initial_decaf_percent / 100)
  let new_decaf : ℝ := new_stock * (new_decaf_percent / 100)
  let total_decaf : ℝ := initial_decaf + new_decaf
  let total_stock : ℝ := initial_stock + new_stock
  (total_decaf / total_stock) * 100

theorem coffee_decaf_percentage :
  initial_stock = 400 →
  initial_decaf_percent = 20 →
  new_stock = 100 →
  new_decaf_percent = 50 →
  decaf_coffee_percentage initial_stock initial_decaf_percent new_stock new_decaf_percent = 26 :=
by
  intros
  sorry

end coffee_decaf_percentage_l187_187471


namespace wendy_albums_l187_187301

theorem wendy_albums (total_pictures remaining_pictures pictures_per_album : ℕ) 
    (h1 : total_pictures = 79)
    (h2 : remaining_pictures = total_pictures - 44)
    (h3 : pictures_per_album = 7) :
    remaining_pictures / pictures_per_album = 5 := by
  sorry

end wendy_albums_l187_187301


namespace problem_statement_l187_187692

variable (g : ℝ)

-- Definition of the operation
def my_op (g y : ℝ) : ℝ := g^2 + 2 * y

-- The statement we want to prove
theorem problem_statement : my_op g (my_op g g) = g^4 + 4 * g^3 + 6 * g^2 + 4 * g :=
by
  sorry

end problem_statement_l187_187692


namespace linda_five_dollar_bills_l187_187736

theorem linda_five_dollar_bills :
  ∃ (x y : ℕ), x + y = 15 ∧ 5 * x + 10 * y = 100 ∧ x = 10 :=
by
  sorry

end linda_five_dollar_bills_l187_187736


namespace sqrt_200_eq_10_l187_187250

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l187_187250


namespace minimize_prod_time_l187_187111

noncomputable def shortest_production_time
  (items : ℕ) 
  (workers : ℕ) 
  (shaping_time : ℕ) 
  (firing_time : ℕ) : ℕ := by
  sorry

-- The main theorem statement
theorem minimize_prod_time
  (items : ℕ := 75)
  (workers : ℕ := 13)
  (shaping_time : ℕ := 15)
  (drying_time : ℕ := 10)
  (firing_time : ℕ := 30)
  (optimal_time : ℕ := 325) :
  shortest_production_time items workers shaping_time firing_time = optimal_time := by
  sorry

end minimize_prod_time_l187_187111


namespace simplify_fraction_l187_187747

theorem simplify_fraction : (2 / 520) + (23 / 40) = 301 / 520 := by
  sorry

end simplify_fraction_l187_187747


namespace total_packages_l187_187628

theorem total_packages (num_trucks : ℕ) (packages_per_truck : ℕ) (h1 : num_trucks = 7) (h2 : packages_per_truck = 70) : num_trucks * packages_per_truck = 490 := by
  sorry

end total_packages_l187_187628


namespace measure_of_angle_A_possibilities_l187_187602

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l187_187602


namespace number_of_girls_l187_187624

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l187_187624


namespace scientific_notation_coronavirus_diameter_l187_187292

theorem scientific_notation_coronavirus_diameter : 0.00000011 = 1.1 * 10^(-7) :=
by {
  sorry
}

end scientific_notation_coronavirus_diameter_l187_187292


namespace minimize_quadratic_function_l187_187693

def quadratic_function (x : ℝ) : ℝ := x^2 + 8*x + 7

theorem minimize_quadratic_function : ∃ x : ℝ, (∀ y : ℝ, quadratic_function y ≥ quadratic_function x) ∧ x = -4 :=
by
  sorry

end minimize_quadratic_function_l187_187693


namespace range_of_a_l187_187414

-- Defining the propositions P and Q 
def P (a : ℝ) : Prop := ∀ x : ℝ, a*x^2 + a*x + 1 > 0
def Q (a : ℝ) : Prop := ∃ x1 x2 : ℝ, x1^2 - x1 + a = 0 ∧ x2^2 - x2 + a = 0

-- Stating the theorem
theorem range_of_a (a : ℝ) :
  (P a ∧ ¬Q a) ∨ (¬P a ∧ Q a) ↔ a ∈ Set.Ioo (1/4 : ℝ) 4 ∪ Set.Iio 0 :=
sorry

end range_of_a_l187_187414


namespace union_of_sets_l187_187408

variable (A : Set ℤ) (B : Set ℤ)

theorem union_of_sets (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) : A ∪ B = {-1, 0, 1, 2} :=
by
  sorry

end union_of_sets_l187_187408


namespace math_club_team_selection_l187_187951

theorem math_club_team_selection : 
  let boys := 7
  let girls := 9
  let team_boys := 4
  let team_girls := 2
  nat.choose boys team_boys * nat.choose girls team_girls = 1260 :=
by
  let boys := 7
  let girls := 9
  let team_boys := 4
  let team_girls := 2
  sorry

end math_club_team_selection_l187_187951


namespace balloons_remaining_intact_l187_187821

def initial_balloons : ℕ := 200
def blown_up_after_half_hour (n : ℕ) : ℕ := n / 5
def remaining_balloons_after_half_hour (n : ℕ) : ℕ := n - blown_up_after_half_hour n

def percentage_of_remaining_balloons_blow_up (remaining : ℕ) : ℕ := remaining * 30 / 100
def remaining_balloons_after_one_hour (remaining : ℕ) : ℕ := remaining - percentage_of_remaining_balloons_blow_up remaining

def durable_balloons (remaining : ℕ) : ℕ := remaining * 10 / 100
def non_durable_balloons (remaining : ℕ) (durable : ℕ) : ℕ := remaining - durable

def twice_non_durable (non_durable : ℕ) : ℕ := non_durable * 2

theorem balloons_remaining_intact : 
  (remaining_balloons_after_half_hour initial_balloons) - 
  (percentage_of_remaining_balloons_blow_up 
    (remaining_balloons_after_half_hour initial_balloons)) - 
  (twice_non_durable 
    (non_durable_balloons 
      (remaining_balloons_after_one_hour 
        (remaining_balloons_after_half_hour initial_balloons)) 
      (durable_balloons 
        (remaining_balloons_after_one_hour 
          (remaining_balloons_after_half_hour initial_balloons))))) = 
  0 := 
by
  sorry

end balloons_remaining_intact_l187_187821


namespace total_players_l187_187136

theorem total_players (kabaddi : ℕ) (only_kho_kho : ℕ) (both_games : ℕ) 
  (h_kabaddi : kabaddi = 10) (h_only_kho_kho : only_kho_kho = 15) 
  (h_both_games : both_games = 5) : (kabaddi - both_games) + only_kho_kho + both_games = 25 :=
by
  sorry

end total_players_l187_187136


namespace problem_gcf_lcm_sum_l187_187729

-- Let A be the GCF of {15, 20, 30}
def A : ℕ := Nat.gcd (Nat.gcd 15 20) 30

-- Let B be the LCM of {15, 20, 30}
def B : ℕ := Nat.lcm (Nat.lcm 15 20) 30

-- We need to prove that A + B = 65
theorem problem_gcf_lcm_sum :
  A + B = 65 :=
by
  sorry

end problem_gcf_lcm_sum_l187_187729


namespace total_lives_correct_l187_187923

-- Define the initial number of friends
def initial_friends : ℕ := 16

-- Define the number of lives each player has
def lives_per_player : ℕ := 10

-- Define the number of additional players that joined
def additional_players : ℕ := 4

-- Define the initial total number of lives
def initial_lives : ℕ := initial_friends * lives_per_player

-- Define the additional lives from the new players
def additional_lives : ℕ := additional_players * lives_per_player

-- Define the final total number of lives
def total_lives : ℕ := initial_lives + additional_lives

-- The proof goal
theorem total_lives_correct : total_lives = 200 :=
by
  -- This is where the proof would be written, but it is omitted.
  sorry

end total_lives_correct_l187_187923


namespace max_expression_value_l187_187769

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l187_187769


namespace exponent_equality_l187_187864

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l187_187864


namespace normal_distribution_probability_example_l187_187852

noncomputable def normalDist (μ σ : ℝ) : probability_theory.ProbabilityMeasure ℝ :=
sorry    -- Normal distribution measure, implementation omitted for brevity

theorem normal_distribution_probability_example :
  let X := @measure_theory.Measure.measure_space.measure
    in
    (X (λ x, 4 - 2 * 1 < x ∧ x ≤ 4 + 2 * 1) = 0.9544)
    ∧ (X (λ x, 4 - 1 < x ∧ x ≤ 4 + 1) = 0.6826)
    ∧ (X (λ x, 5 < x ∧ x < 6) = 0.1359)
:= by
  let μ := 4
  let σ := 1
  let X := normalDist μ σ
  -- Bringing in the conditions
  have h1 : X (λ x, μ - 2 * σ < x ∧ x ≤ μ + 2 * σ) = 0.9544 := sorry,
  have h2 : X (λ x, μ - σ < x ∧ x ≤ μ + σ) = 0.6826 := sorry,
  -- Conclusion based on given conditions
  exact And.intro h1 (And.intro h2 sorry)

end normal_distribution_probability_example_l187_187852


namespace number_of_cars_l187_187398

theorem number_of_cars (C : ℕ) : 
  let bicycles := 3
  let pickup_trucks := 8
  let tricycles := 1
  let car_tires := 4
  let bicycle_tires := 2
  let pickup_truck_tires := 4
  let tricycle_tires := 3
  let total_tires := 101
  (4 * C + 3 * bicycle_tires + 8 * pickup_truck_tires + 1 * tricycle_tires = total_tires) → C = 15 := by
  intros h
  sorry

end number_of_cars_l187_187398


namespace marbles_distribution_l187_187794

open BigOperators

/-- There are 52 marbles in total in five bags with no two bags containing the same number of marbles.
    Show that there exists a distribution satisfying these conditions.
    Show that in any such distribution, one bag contains exactly 12 marbles. -/
theorem marbles_distribution :
  ∃ (bags : Finset ℕ), (bags.card = 5) ∧ (∑ x in bags, x = 52) ∧ bags.pairwise (≠) ∧ ∃ x ∈ bags, x = 12 :=
begin
  sorry
end

end marbles_distribution_l187_187794


namespace eccentricity_of_hyperbola_l187_187791

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ)
  (hc : c^2 = a^2 + b^2) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem eccentricity_of_hyperbola (a b c e : ℝ)
  (ha : a > 0) (hb : b > 0) (h_hyperbola : c^2 = a^2 + b^2)
  (h_eccentricity : e = (1 + Real.sqrt 5) / 2) :
  e = hyperbola_eccentricity a b ha hb c h_hyperbola :=
by
  sorry

end eccentricity_of_hyperbola_l187_187791


namespace max_value_of_expression_l187_187776

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l187_187776


namespace factorization_identity_l187_187521

theorem factorization_identity (x : ℝ) : 
  3 * x^2 + 12 * x + 12 = 3 * (x + 2)^2 :=
by
  sorry

end factorization_identity_l187_187521


namespace no_solutions_for_sin_cos_eq_sqrt3_l187_187174

theorem no_solutions_for_sin_cos_eq_sqrt3 (x : ℝ) (hx : 0 ≤ x ∧ x < 2 * Real.pi) :
  ¬ (Real.sin x + Real.cos x = Real.sqrt 3) :=
by
  sorry

end no_solutions_for_sin_cos_eq_sqrt3_l187_187174


namespace rational_numbers_integer_sum_pow_l187_187845

open Rat Nat

theorem rational_numbers_integer_sum_pow (n : ℕ) : 
  (∃ a b : ℚ, (¬ a ∈ Int) ∧ (¬ b ∈ Int) ∧ (a + b ∈ Int) ∧ (a^n + b^n ∈ Int)) ↔ Odd n :=
sorry

end rational_numbers_integer_sum_pow_l187_187845


namespace misread_number_l187_187592

theorem misread_number (X : ℕ) :
  (average_10_initial : ℕ) = 18 →
  (incorrect_read : ℕ) = 26 →
  (average_10_correct : ℕ) = 22 →
  (10 * 22 - 10 * 18 = X + 26 - 26) →
  X = 66 :=
by sorry

end misread_number_l187_187592


namespace sqrt_200_eq_10_sqrt_2_l187_187253

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l187_187253


namespace cos_seven_pi_over_four_l187_187391

theorem cos_seven_pi_over_four :
  Real.cos (7 * Real.pi / 4) = Real.sqrt 2 / 2 :=
by
  sorry

end cos_seven_pi_over_four_l187_187391


namespace line_through_intersection_and_parallel_l187_187175

theorem line_through_intersection_and_parallel
  (x y : ℝ)
  (l1 : 3 * x + 4 * y - 2 = 0)
  (l2 : 2 * x + y + 2 = 0)
  (l3 : ∃ k : ℝ, k * x + y + 2 = 0 ∧ k = -(4 / 3)) :
  ∃ a b c : ℝ, a * x + b * y + c = 0 ∧ a = 4 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end line_through_intersection_and_parallel_l187_187175


namespace value_of_a_when_x_is_3_root_l187_187063

theorem value_of_a_when_x_is_3_root (a : ℝ) :
  (3 ^ 2 + 3 * a + 9 = 0) -> a = -6 := by
  intros h
  sorry

end value_of_a_when_x_is_3_root_l187_187063


namespace shorter_stick_length_l187_187789

variable (L S : ℝ)

theorem shorter_stick_length
  (h1 : L - S = 12)
  (h2 : (2 / 3) * L = S) :
  S = 24 := by
  sorry

end shorter_stick_length_l187_187789


namespace optimal_worker_distribution_l187_187112
noncomputable def forming_time (n1 : ℕ) : ℕ := (nat.ceil (75.0 / n1) : ℕ) * 15
noncomputable def firing_time (n3 : ℕ) : ℕ := (nat.ceil (75.0 / n3) : ℕ) * 30

theorem optimal_worker_distribution:
  ∃ n1 n3 : ℕ, n1 + n3 = 13 ∧ (forming_time n1 = 225 ∨ firing_time n3 = 330) :=
sorry

end optimal_worker_distribution_l187_187112


namespace claire_photos_l187_187331

theorem claire_photos (C L R : ℕ) 
  (h1 : L = 3 * C) 
  (h2 : R = C + 12)
  (h3 : L = R) : C = 6 := 
by
  sorry

end claire_photos_l187_187331


namespace find_natural_numbers_with_integer_roots_l187_187989

theorem find_natural_numbers_with_integer_roots :
  ∃ (p q : ℕ), 
    (∀ x : ℤ, x * x - (p * q) * x + (p + q) = 0 → ∃ (x1 x2 : ℤ), x = x1 ∧ x = x2 ∧ x1 + x2 = p * q ∧ x1 * x2 = p + q) ↔
    ((p = 1 ∧ q = 5) ∨ (p = 5 ∧ q = 1) ∨ (p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) :=
by
-- proof skipped
sorry

end find_natural_numbers_with_integer_roots_l187_187989


namespace sqrt_200_simplified_l187_187268

-- Definitions based on conditions from part a)
def factorization : Nat := 2 ^ 3 * 5 ^ 2

lemma sqrt_property (a b : ℕ) : Real.sqrt (a^2 * b) = a * Real.sqrt b := sorry

-- The proof problem (only the statement, not the proof)
theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  have h1 : 200 = 2^3 * 5^2 := by rfl
  have h2 : Real.sqrt (200) = Real.sqrt (2^3 * 5^2) := by rw h1
  rw [←show 200 = factorization by rfl] at h2
  exact sorry

end sqrt_200_simplified_l187_187268


namespace mathd_inequality_l187_187742

theorem mathd_inequality (x y z : ℝ) (hx : x ≥ 1) (hy : y ≥ 1) (hz : z ≥ 1) : 
  (x^3 + 2*y^2 + 3*z) * (4*y^3 + 5*z^2 + 6*x) * (7*z^3 + 8*x^2 + 9*y) ≥ 720 * (x * y + y * z + z * x) :=
by
  sorry

end mathd_inequality_l187_187742


namespace number_of_girls_l187_187626

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l187_187626


namespace max_value_of_fraction_l187_187757

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l187_187757


namespace find_n_plus_m_l187_187411

noncomputable def f (x : ℝ) := abs (Real.log x / Real.log 2)

theorem find_n_plus_m (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n) (h3 : m < n)
    (h4 : f m = f n) (h5 : ∀ x, m^2 ≤ x ∧ x ≤ n → f x ≤ 2) :
    n + m = 5 / 2 := sorry

end find_n_plus_m_l187_187411


namespace max_square_side_length_l187_187190

theorem max_square_side_length (AC BC : ℝ) (hAC : AC = 3) (hBC : BC = 7) : 
  ∃ s : ℝ, s = 2.1 := by
  sorry

end max_square_side_length_l187_187190


namespace find_positive_integer_solutions_l187_187700

theorem find_positive_integer_solutions :
  ∃ (x y z : ℕ), 
    2 * x * z = y^2 ∧ 
    x + z = 1987 ∧ 
    x = 1458 ∧ 
    y = 1242 ∧ 
    z = 529 :=
  by sorry

end find_positive_integer_solutions_l187_187700


namespace chocolates_bought_in_a_month_l187_187889

theorem chocolates_bought_in_a_month :
  ∀ (chocolates_for_her: ℕ)
    (chocolates_for_sister: ℕ)
    (chocolates_for_charlie: ℕ)
    (weeks_in_a_month: ℕ), 
  weeks_in_a_month = 4 →
  chocolates_for_her = 2 →
  chocolates_for_sister = 1 →
  chocolates_for_charlie = 10 →
  (chocolates_for_her * weeks_in_a_month + chocolates_for_sister * weeks_in_a_month + chocolates_for_charlie) = 22 :=
by
  intros chocolates_for_her chocolates_for_sister chocolates_for_charlie weeks_in_a_month
  intros h_weeks h_her h_sister h_charlie
  sorry

end chocolates_bought_in_a_month_l187_187889


namespace fraction_increase_each_year_l187_187699

variable (initial_value : ℝ := 57600)
variable (final_value : ℝ := 72900)
variable (years : ℕ := 2)

theorem fraction_increase_each_year :
  ∃ (f : ℝ), initial_value * (1 + f)^years = final_value ∧ f = 0.125 := by
  sorry

end fraction_increase_each_year_l187_187699


namespace range_of_a_decreasing_l187_187051

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 1) * x + 2

theorem range_of_a_decreasing (a : ℝ) : (∀ x y : ℝ, x ∈ Set.Iic 4 → y ∈ Set.Iic 4 → x ≤ y → f x a ≥ f y a) ↔ a ≤ -3 :=
by
  sorry

end range_of_a_decreasing_l187_187051


namespace PQR_positive_iff_P_Q_R_positive_l187_187895

noncomputable def P (a b c : ℝ) : ℝ := a + b - c
noncomputable def Q (a b c : ℝ) : ℝ := b + c - a
noncomputable def R (a b c : ℝ) : ℝ := c + a - b

theorem PQR_positive_iff_P_Q_R_positive (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (P a b c * Q a b c * R a b c > 0) ↔ (P a b c > 0 ∧ Q a b c > 0 ∧ R a b c > 0) :=
sorry

end PQR_positive_iff_P_Q_R_positive_l187_187895


namespace xiao_qian_has_been_to_great_wall_l187_187659

-- Define the four students
inductive Student
| XiaoZhao
| XiaoQian
| XiaoSun
| XiaoLi

open Student

-- Define the relations for their statements
def has_been (s : Student) : Prop :=
  match s with
  | XiaoZhao => false
  | XiaoQian => true
  | XiaoSun => true
  | XiaoLi => false

def said (s : Student) : Prop :=
  match s with
  | XiaoZhao => ¬has_been XiaoZhao
  | XiaoQian => has_been XiaoLi
  | XiaoSun => has_been XiaoQian
  | XiaoLi => ¬has_been XiaoLi

axiom only_one_lying : ∃ l : Student, ∀ s : Student, said s → (s ≠ l)

theorem xiao_qian_has_been_to_great_wall : has_been XiaoQian :=
by {
  sorry -- Proof elided
}

end xiao_qian_has_been_to_great_wall_l187_187659


namespace strategy_classification_l187_187287

inductive Player
| A
| B

def A_winning_strategy (n0 : Nat) : Prop :=
  n0 >= 8

def B_winning_strategy (n0 : Nat) : Prop :=
  n0 <= 5

def neither_winning_strategy (n0 : Nat) : Prop :=
  n0 = 6 ∨ n0 = 7

theorem strategy_classification (n0 : Nat) : 
  (A_winning_strategy n0 ∨ B_winning_strategy n0 ∨ neither_winning_strategy n0) := by
    sorry

end strategy_classification_l187_187287


namespace least_positive_integer_addition_l187_187322

theorem least_positive_integer_addition (k : ℕ) (h₀ : 525 + k % 5 = 0) (h₁ : 0 < k) : k = 5 := 
by
  sorry

end least_positive_integer_addition_l187_187322


namespace largest_square_side_length_is_2_point_1_l187_187183

noncomputable def largest_square_side_length (A B C : Point) (hABC : right_triangle A B C) (hAC : distance A C = 3) (hCB : distance C B = 7) : ℝ :=
  max_square_side_length A B C

theorem largest_square_side_length_is_2_point_1 :
  largest_square_side_length (3, 0) (0, 7) (0, 0) sorry sorry = 2.1 :=
by
  sorry

end largest_square_side_length_is_2_point_1_l187_187183


namespace sqrt_200_eq_10_l187_187266

theorem sqrt_200_eq_10 : real.sqrt 200 = 10 :=
by
  calc
    real.sqrt 200 = real.sqrt (2^2 * 5^2) : by sorry -- show 200 = 2^2 * 5^2
    ... = real.sqrt (2^2) * real.sqrt (5^2) : by sorry -- property of square roots of products
    ... = 2 * 5 : by sorry -- using the property sqrt (a^2) = a
    ... = 10 : by sorry

end sqrt_200_eq_10_l187_187266


namespace part1_part2_part3_l187_187448

-- Part 1
theorem part1 (a b : ℝ) : 
    3 * (a - b) ^ 2 - 6 * (a - b) ^ 2 + 2 * (a - b) ^ 2 = - (a - b) ^ 2 := 
    sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x ^ 2 - 2 * y = 4) : 
    3 * x ^ 2 - 6 * y - 21 = -9 := 
    sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5 * b = 3) (h2 : 5 * b - 3 * c = -5) (h3 : 3 * c - d = 10) : 
    (a - 3 * c) + (5 * b - d) - (5 * b - 3 * c) = 8 := 
    sorry

end part1_part2_part3_l187_187448


namespace ratio_trumpet_to_flute_l187_187921

-- Given conditions
def flute_players : ℕ := 5
def trumpet_players (T : ℕ) : ℕ := T
def trombone_players (T : ℕ) : ℕ := T - 8
def drummers (T : ℕ) : ℕ := T - 8 + 11
def clarinet_players : ℕ := 2 * flute_players
def french_horn_players (T : ℕ) : ℕ := T - 8 + 3
def total_seats_needed (T : ℕ) : ℕ := 
  flute_players + trumpet_players T + trombone_players T + drummers T + clarinet_players + french_horn_players T

-- Proof statement
theorem ratio_trumpet_to_flute 
  (T : ℕ) (h : total_seats_needed T = 65) : trumpet_players T / flute_players = 3 :=
sorry

end ratio_trumpet_to_flute_l187_187921


namespace find_x_l187_187654

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l187_187654


namespace find_triangle_l187_187048

theorem find_triangle (q : ℝ) (triangle : ℝ) (h1 : 3 * triangle * q = 63) (h2 : 7 * (triangle + q) = 161) : triangle = 1 :=
sorry

end find_triangle_l187_187048


namespace largest_square_side_length_l187_187182

theorem largest_square_side_length (AC BC : ℝ) (C_vertex_at_origin : (0, 0) ∈ triangle ABC)
  (AC_eq_three : AC = 3) (CB_eq_seven : CB = 7) : 
  ∃ (s : ℝ), s = 2.1 :=
by {
  sorry
}

end largest_square_side_length_l187_187182


namespace payment_difference_correct_l187_187837

noncomputable def prove_payment_difference (x : ℕ) (h₀ : x > 0) : Prop :=
  180 / x - 180 / (x + 2) = 3

theorem payment_difference_correct (x : ℕ) (h₀ : x > 0) : prove_payment_difference x h₀ :=
  by
    sorry

end payment_difference_correct_l187_187837


namespace repetend_five_seventeen_l187_187396

noncomputable def repetend_of_fraction (n : ℕ) (d : ℕ) : ℕ := sorry

theorem repetend_five_seventeen : repetend_of_fraction 5 17 = 294117647058823529 := sorry

end repetend_five_seventeen_l187_187396


namespace volunteers_meet_again_in_360_days_l187_187908

theorem volunteers_meet_again_in_360_days :
  let Sasha := 5
  let Leo := 8
  let Uma := 9
  let Kim := 10
  Nat.lcm Sasha (Nat.lcm Leo (Nat.lcm Uma Kim)) = 360 :=
by
  sorry

end volunteers_meet_again_in_360_days_l187_187908


namespace no_real_solution_l187_187497

noncomputable def quadratic_eq (x : ℝ) : ℝ := (2*x^2 - 3*x + 5)

theorem no_real_solution : 
  ∀ x : ℝ, quadratic_eq x ^ 2 + 1 ≠ 1 :=
by
  intro x
  sorry

end no_real_solution_l187_187497


namespace speed_including_stoppages_l187_187172

theorem speed_including_stoppages : 
  ∀ (speed_excluding_stoppages : ℝ) (stoppage_minutes_per_hour : ℝ), 
  speed_excluding_stoppages = 65 → 
  stoppage_minutes_per_hour = 15.69 → 
  (speed_excluding_stoppages * (1 - stoppage_minutes_per_hour / 60)) = 47.9025 := 
by intros speed_excluding_stoppages stoppage_minutes_per_hour h1 h2
   sorry

end speed_including_stoppages_l187_187172


namespace math_problem_l187_187466

theorem math_problem : 
  (Real.sqrt 4) * (4 ^ (1 / 2: ℝ)) + (16 / 4) * 2 - (8 ^ (1 / 2: ℝ)) = 12 - 2 * Real.sqrt 2 :=
by
  sorry

end math_problem_l187_187466


namespace inequality_proof_l187_187344

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187344


namespace find_large_monkey_doll_cost_l187_187836

-- Define the conditions and the target property
def large_monkey_doll_cost (L : ℝ) (condition1 : 300 / (L - 2) = 300 / L + 25)
                           (condition2 : 300 / (L + 1) = 300 / L - 15) : Prop :=
  L = 6

-- The main theorem with the conditions
theorem find_large_monkey_doll_cost (L : ℝ)
  (h1 : 300 / (L - 2) = 300 / L + 25)
  (h2 : 300 / (L + 1) = 300 / L - 15) : large_monkey_doll_cost L h1 h2 :=
  sorry

end find_large_monkey_doll_cost_l187_187836


namespace number_of_girls_l187_187621

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l187_187621


namespace largest_square_side_length_l187_187181

theorem largest_square_side_length (AC BC : ℝ) (C_vertex_at_origin : (0, 0) ∈ triangle ABC)
  (AC_eq_three : AC = 3) (CB_eq_seven : CB = 7) : 
  ∃ (s : ℝ), s = 2.1 :=
by {
  sorry
}

end largest_square_side_length_l187_187181


namespace cost_500_pencils_is_25_dollars_l187_187914

def cost_of_500_pencils (cost_per_pencil : ℕ) (pencils : ℕ) (cents_per_dollar : ℕ) : ℕ :=
  (cost_per_pencil * pencils) / cents_per_dollar

theorem cost_500_pencils_is_25_dollars : cost_of_500_pencils 5 500 100 = 25 := by
  sorry

end cost_500_pencils_is_25_dollars_l187_187914


namespace hiking_packing_weight_l187_187367

theorem hiking_packing_weight
  (miles_per_hour : ℝ)
  (hours_per_day : ℝ)
  (days : ℝ)
  (supply_per_mile : ℝ)
  (resupply_fraction : ℝ)
  (expected_first_pack_weight : ℝ)
  (hiking_hours : ℝ := hours_per_day * days)
  (total_miles : ℝ := miles_per_hour * hiking_hours)
  (total_weight_needed : ℝ := supply_per_mile * total_miles)
  (resupply_weight : ℝ := resupply_fraction * total_weight_needed)
  (first_pack_weight : ℝ := total_weight_needed - resupply_weight) :
  first_pack_weight = 37.5 :=
by
  -- The proof goes here, but is omitted since the proof is not required.
  sorry

end hiking_packing_weight_l187_187367


namespace total_tickets_sold_l187_187681

theorem total_tickets_sold
  (advanced_ticket_cost : ℕ)
  (door_ticket_cost : ℕ)
  (total_collected : ℕ)
  (advanced_tickets_sold : ℕ)
  (door_tickets_sold : ℕ) :
  advanced_ticket_cost = 8 →
  door_ticket_cost = 14 →
  total_collected = 1720 →
  advanced_tickets_sold = 100 →
  total_collected = (advanced_tickets_sold * advanced_ticket_cost) + (door_tickets_sold * door_ticket_cost) →
  100 + door_tickets_sold = 165 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_tickets_sold_l187_187681


namespace possible_measures_A_l187_187601

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l187_187601


namespace todd_initial_gum_l187_187799

-- Define the conditions and the final result
def initial_gum (final_gum: Nat) (given_gum: Nat) : Nat := final_gum - given_gum

theorem todd_initial_gum :
  initial_gum 54 16 = 38 :=
by
  -- Use the initial_gum definition to state the problem
  -- The proof is skipped with sorry
  sorry

end todd_initial_gum_l187_187799


namespace field_trip_total_l187_187820

-- Define the conditions
def vans := 2
def buses := 3
def people_per_van := 8
def people_per_bus := 20

-- The total number of people
def total_people := (vans * people_per_van) + (buses * people_per_bus)

theorem field_trip_total : total_people = 76 :=
by
  -- skip the proof here
  sorry

end field_trip_total_l187_187820


namespace cab_driver_income_l187_187480

theorem cab_driver_income (incomes : Fin 5 → ℝ)
  (h1 : incomes 0 = 250)
  (h2 : incomes 1 = 400)
  (h3 : incomes 2 = 750)
  (h4 : incomes 3 = 400)
  (avg_income : (incomes 0 + incomes 1 + incomes 2 + incomes 3 + incomes 4) / 5 = 460) : 
  incomes 4 = 500 :=
sorry

end cab_driver_income_l187_187480


namespace cookie_radius_proof_l187_187104

-- Define the given equation of the cookie
def cookie_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 36 = 6 * x + 9 * y

-- Define the radius computation for the circle derived from the given equation
def cookie_radius (r : ℝ) : Prop :=
  r = 3 * Real.sqrt 5 / 2

-- The theorem to prove that the radius of the described cookie is as obtained
theorem cookie_radius_proof :
  ∀ x y : ℝ, cookie_equation x y → cookie_radius (Real.sqrt (45 / 4)) :=
by
  sorry

end cookie_radius_proof_l187_187104


namespace number_of_girls_l187_187618

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l187_187618


namespace smallest_number_conditions_l187_187468

theorem smallest_number_conditions :
  ∃ m : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], m % k = 2) ∧ (m % 8 = 0) ∧ ( ∀ n : ℕ, (∀ k ∈ [3, 4, 5, 6, 7], n % k = 2) ∧ (n % 8 = 0) → m ≤ n ) :=
sorry

end smallest_number_conditions_l187_187468


namespace larger_segment_length_l187_187960

theorem larger_segment_length 
  (x y : ℝ)
  (h1 : 40^2 = x^2 + y^2)
  (h2 : 90^2 = (110 - x)^2 + y^2) :
  110 - x = 84.55 :=
by
  sorry

end larger_segment_length_l187_187960


namespace quadratic_equation_properties_l187_187849

theorem quadratic_equation_properties (m : ℝ) (h : m < 4) (root_one : ℝ) (root_two : ℝ) 
  (eq1 : root_one + root_two = 4) (eq2 : root_one * root_two = m) (root_one_eq : root_one = -1) :
  m = -5 ∧ root_two = 5 ∧ (root_one ≠ root_two) :=
by
  -- Sorry is added to skip the proof because only the statement is needed.
  sorry

end quadratic_equation_properties_l187_187849


namespace total_cows_l187_187493

theorem total_cows (cows_per_herd : Nat) (herds : Nat) (total_cows : Nat) : 
  cows_per_herd = 40 → herds = 8 → total_cows = cows_per_herd * herds → total_cows = 320 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_cows_l187_187493


namespace f_3_eq_4_l187_187538

noncomputable def f : ℝ → ℝ := sorry

theorem f_3_eq_4 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 1) = x^2) : f 3 = 4 :=
by
  sorry

end f_3_eq_4_l187_187538


namespace solve_linear_eq_l187_187113

theorem solve_linear_eq (x : ℝ) : 3 * x - 6 = 0 ↔ x = 2 :=
sorry

end solve_linear_eq_l187_187113


namespace inequality_proof_l187_187334

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l187_187334


namespace arithmetic_geometric_sum_l187_187425

noncomputable def a_n (n : ℕ) := 3 * n - 2
noncomputable def b_n (n : ℕ) := 4 ^ (n - 1)

theorem arithmetic_geometric_sum (n : ℕ) :
    a_n 1 = 1 ∧ a_n 2 = b_n 2 ∧ a_n 6 = b_n 3 ∧ S_n = 1 + (n - 1) * 4 ^ n :=
by sorry

end arithmetic_geometric_sum_l187_187425


namespace units_digit_of_2_pow_20_minus_1_l187_187124

theorem units_digit_of_2_pow_20_minus_1 : (2^20 - 1) % 10 = 5 := 
  sorry

end units_digit_of_2_pow_20_minus_1_l187_187124


namespace sum_of_ages_is_20_l187_187727

-- Given conditions
variables (age_kiana age_twin : ℕ)
axiom product_of_ages : age_kiana * age_twin * age_twin = 162

-- Required proof
theorem sum_of_ages_is_20 : age_kiana + age_twin + age_twin = 20 :=
sorry

end sum_of_ages_is_20_l187_187727


namespace solution_to_problem_l187_187701

theorem solution_to_problem
  {x y z : ℝ}
  (h1 : xy / (x + y) = 1 / 3)
  (h2 : yz / (y + z) = 1 / 5)
  (h3 : zx / (z + x) = 1 / 6) :
  xyz / (xy + yz + zx) = 1 / 7 :=
by sorry

end solution_to_problem_l187_187701


namespace single_intersection_not_necessarily_tangent_l187_187498

structure Hyperbola where
  -- Placeholder for hyperbola properties
  axis1 : Real
  axis2 : Real

def is_tangent (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for tangency
  ∃ p : Real × Real, l = { p }

def is_parallel_to_asymptote (l : Set (Real × Real)) (H : Hyperbola) : Prop :=
  -- Placeholder definition for parallelism to asymptote 
  ∃ A : Real, l = { (x, A * x) | x : Real }

theorem single_intersection_not_necessarily_tangent
  (l : Set (Real × Real)) (H : Hyperbola) (h : ∃ p : Real × Real, l = { p }) :
  ¬ is_tangent l H ∨ is_parallel_to_asymptote l H :=
sorry

end single_intersection_not_necessarily_tangent_l187_187498


namespace conservation_center_total_turtles_l187_187965

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l187_187965


namespace smallest_number_among_l187_187007

noncomputable def smallest_real among (a b c d : Real) : Real :=
  if a <= b && a <= c && a <= d then a
  else if b <= a && b <= c && b <= d then b
  else if c <= a && c <= b && c <= d then c
  else d

theorem smallest_number_among :=
  let r1 := -1.0
  let r2 := -|(-2.0)|
  let r3 := 0.0
  let r4 := Real.pi
  smallest_real r1 r2 r3 r4 = -2.0 :=
by
  -- placeholder for proof
  sorry

end smallest_number_among_l187_187007


namespace not_diff_of_squares_2022_l187_187243

theorem not_diff_of_squares_2022 :
  ¬ ∃ a b : ℤ, a^2 - b^2 = 2022 :=
by
  sorry

end not_diff_of_squares_2022_l187_187243


namespace smallest_circle_tangent_to_line_and_circle_l187_187108

-- Define the line equation as a condition
def line_eq (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the original circle equation as a condition
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 2 * y = 0

-- Define the smallest circle equation as a condition
def smallest_circle_eq (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- The main lemma to prove that the smallest circle's equation matches the expected result
theorem smallest_circle_tangent_to_line_and_circle :
  (∀ x y, line_eq x y → smallest_circle_eq x y) ∧ (∀ x y, circle_eq x y → smallest_circle_eq x y) :=
by
  sorry -- Proof is omitted, as instructed

end smallest_circle_tangent_to_line_and_circle_l187_187108


namespace cube_square_third_smallest_prime_l187_187314

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l187_187314


namespace arith_seq_formula_geom_seq_sum_l187_187850

-- Definitions for condition 1: Arithmetic sequence {a_n}
def arithmetic_seq (a : ℕ → ℤ) : Prop :=
  (a 4 = 7) ∧ (a 10 = 19)

-- Definitions for condition 2: Sum of the first n terms of {a_n}
def sum_arith_seq (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

-- Definitions for condition 3: Geometric sequence {b_n}
def geometric_seq (b : ℕ → ℤ) : Prop :=
  (b 1 = 2) ∧ (∀ n, b (n + 1) = b n * 2)

-- Definitions for condition 4: Sum of the first n terms of {b_n}
def sum_geom_seq (T : ℕ → ℤ) (b : ℕ → ℤ) : Prop :=
  ∀ n, T n = (b 1 * (1 - (2 ^ n))) / (1 - 2)

-- Proving the general formula for arithmetic sequence
theorem arith_seq_formula (a : ℕ → ℤ) (S : ℕ → ℤ) :
  arithmetic_seq a ∧ sum_arith_seq S a → 
  (∀ n, a n = 2 * n - 1) ∧ (∀ n, S n = n ^ 2) :=
sorry

-- Proving the sum of the first n terms for geometric sequence
theorem geom_seq_sum (b : ℕ → ℤ) (T : ℕ → ℤ) (S : ℕ → ℤ) :
  geometric_seq b ∧ sum_geom_seq T b ∧ b 4 = S 4 → 
  (∀ n, T n = 2 ^ (n + 1) - 2) :=
sorry

end arith_seq_formula_geom_seq_sum_l187_187850


namespace pentagon_perimeter_l187_187640

noncomputable def perimeter_pentagon (FG GH HI IJ : ℝ) (FH FI FJ : ℝ) : ℝ :=
  FG + GH + HI + IJ + FJ

theorem pentagon_perimeter : 
  ∀ (FG GH HI IJ : ℝ), 
  ∀ (FH FI FJ : ℝ),
  FG = 1 → GH = 1 → HI = 1 → IJ = 1 →
  FH^2 = FG^2 + GH^2 → FI^2 = FH^2 + HI^2 → FJ^2 = FI^2 + IJ^2 →
  perimeter_pentagon FG GH HI IJ FJ = 6 :=
by
  intros FG GH HI IJ FH FI FJ
  intros H_FG H_GH H_HI H_IJ
  intros H1 H2 H3
  sorry

end pentagon_perimeter_l187_187640


namespace harriet_forward_speed_proof_l187_187126

def harriet_forward_time : ℝ := 3 -- forward time in hours
def harriet_return_speed : ℝ := 150 -- return speed in km/h
def harriet_total_time : ℝ := 5 -- total trip time in hours

noncomputable def harriet_forward_speed : ℝ :=
  let distance := harriet_return_speed * (harriet_total_time - harriet_forward_time)
  distance / harriet_forward_time

theorem harriet_forward_speed_proof : harriet_forward_speed = 100 := by
  sorry

end harriet_forward_speed_proof_l187_187126


namespace length_AE_l187_187525

-- The given conditions:
def isosceles_triangle (A B C : Type*) (AB BC : ℝ) (h : AB = BC) : Prop := true

def angles_and_lengths (A D C E : Type*) (angle_ADC angle_AEC AD CE DC : ℝ) 
  (h_angles : angle_ADC = 60 ∧ angle_AEC = 60)
  (h_lengths : AD = 13 ∧ CE = 13 ∧ DC = 9) : Prop := true

variables {A B C D E : Type*} (AB BC AD CE DC : ℝ)
  (h_isosceles_triangle : isosceles_triangle A B C AB BC (by sorry))
  (h_angles_and_lengths : angles_and_lengths A D C E 60 60 AD CE DC 
    (by split; norm_num) (by repeat {split}; norm_num))

-- The proof problem:
theorem length_AE : ∃ AE : ℝ, AE = 4 :=
  by sorry

end length_AE_l187_187525


namespace simplify_sqrt_200_l187_187277

theorem simplify_sqrt_200 : (sqrt 200 : ℝ) = 10 * sqrt 2 := by
  -- proof goes here
  sorry

end simplify_sqrt_200_l187_187277


namespace total_cost_of_one_pencil_and_eraser_l187_187880

/-- Lila buys 15 pencils and 7 erasers for 170 cents. A pencil costs less than an eraser, 
neither item costs exactly half as much as the other, and both items cost a whole number of cents. 
Prove that the total cost of one pencil and one eraser is 16 cents. -/
theorem total_cost_of_one_pencil_and_eraser (p e : ℕ) (h1 : 15 * p + 7 * e = 170)
  (h2 : p < e) (h3 : p ≠ e / 2) : p + e = 16 :=
sorry

end total_cost_of_one_pencil_and_eraser_l187_187880


namespace f_decreasing_iff_f_minimum_value_l187_187730

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^2 - m * log (2 * x + 1)

noncomputable def f' (x : ℝ) (m : ℝ) : ℝ := 2 * x - (2 * m) / (2 * x + 1)

theorem f_decreasing_iff {m : ℝ} (h : 0 < m) : (∀ x ∈ Ioo (-1/2 : ℝ) 1, f' x m ≤ 0) ↔ 3 ≤ m := sorry

theorem f_minimum_value {m : ℝ} (h : 0 < m) :
  (∃ x ∈ Icc (-1/2 : ℝ) 1, ∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m) ∧
  ((0 < m ∧ m < 3) → ∃ x ∈ Icc (-1/2 : ℝ) 1, x = (-1 + sqrt (1 + 8 * m)) / 4 ∧
    (∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m)) ∧
  (3 ≤ m → ∃ x ∈ Icc (-1/2 : ℝ) 1, x = 1 ∧ (∀ y ∈ Icc (-1/2 : ℝ) 1, f x m ≤ f y m)) := sorry

end f_decreasing_iff_f_minimum_value_l187_187730


namespace proportion_correct_l187_187059

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l187_187059


namespace power_equivalence_l187_187865

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l187_187865


namespace sequence_value_of_m_l187_187430

theorem sequence_value_of_m (a : ℕ → ℝ) (m : ℕ) (h1 : a 1 = 1)
                            (h2 : ∀ n : ℕ, n > 0 → a n - a (n + 1) = a (n + 1) * a n)
                            (h3 : 8 * a m = 1) :
                            m = 8 := by
  sorry

end sequence_value_of_m_l187_187430


namespace measure_of_angle_A_possibilities_l187_187604

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l187_187604


namespace number_of_girls_l187_187620

theorem number_of_girls (total_students : ℕ) (prob_boys : ℚ) (prob : prob_boys = 3 / 25) :
  ∃ (n : ℕ), (binom 25 2) ≠ 0 ∧ (binom n 2) / (binom 25 2) = prob_boys → total_students - n = 16 := 
by
  let boys_num := 9
  let girls_num := total_students - boys_num
  use n, sorry

end number_of_girls_l187_187620


namespace number_div_0_04_eq_200_9_l187_187144

theorem number_div_0_04_eq_200_9 (n : ℝ) (h : n / 0.04 = 200.9) : n = 8.036 :=
sorry

end number_div_0_04_eq_200_9_l187_187144


namespace medicine_dose_per_part_l187_187364

-- Define the given conditions
def kg_weight : ℕ := 30
def ml_per_kg : ℕ := 5
def parts : ℕ := 3

-- The theorem statement
theorem medicine_dose_per_part : 
  (kg_weight * ml_per_kg) / parts = 50 :=
by
  sorry

end medicine_dose_per_part_l187_187364


namespace sqrt_200_eq_10_sqrt_2_l187_187256

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l187_187256


namespace wood_allocation_l187_187680

theorem wood_allocation (x y : ℝ) (h1 : 50 * x * 4 = 300 * y) (h2 : x + y = 5) : x = 3 :=
by
  sorry

end wood_allocation_l187_187680


namespace sum_of_reciprocals_l187_187898

open Real

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x + y = 5 * x * y) (hx2y : x = 2 * y) : 
  (1 / x) + (1 / y) = 5 := 
  sorry

end sum_of_reciprocals_l187_187898


namespace solve_abs_eq_l187_187648

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l187_187648


namespace measure_of_angle_A_possibilities_l187_187603

theorem measure_of_angle_A_possibilities (A B : ℕ) (h1 : A + B = 180) (h2 : ∃ k : ℕ, k ≥ 1 ∧ A = k * B) : 
  ∃ n : ℕ, n = 17 :=
by
  -- the statement needs provable proof and equal 17
  -- skip the proof
  sorry

end measure_of_angle_A_possibilities_l187_187603


namespace probability_second_term_3_l187_187893

open Finset
open Fintype

noncomputable def permutations_without_1_2 : Finset (Fin 6 → Fin 6) :=
  univ.filter (λ f, f 0 ≠ 0 ∧ f 0 ≠ 1)

noncomputable def favorable_permutations : Finset (Fin 6 → Fin 6) :=
  permutations_without_1_2.filter (λ f, f 1 = 2)

theorem probability_second_term_3 (a b : ℕ) (h : Nat.gcd a b = 1): 
  (a : ℚ) / b = (favorable_permutations.card : ℚ) / (permutations_without_1_2.card : ℚ) → a + b = 23 := by
  sorry

end probability_second_term_3_l187_187893


namespace probability_X_lt_0_l187_187569

open ProbabilityTheory MeasureTheory

-- Given conditions
variables (X : ℝ → ℝ) (σ : ℝ)
  [NormalDistribution X 2 (σ^2)]
  (h : (probability {x | 0 < X x ∧ X x < 4}) = 0.3)

-- The statement to prove
theorem probability_X_lt_0 : (probability {x | X x < 0}) = 0.35 :=
sorry

end probability_X_lt_0_l187_187569


namespace union_of_sets_l187_187409

theorem union_of_sets (A : Set ℤ) (B : Set ℤ) (hA : A = {0, 1, 2}) (hB : B = {-1, 0}) :
  A ∪ B = {-1, 0, 1, 2} :=
by
  rw [hA, hB]
  apply Set.ext
  intro x
  simp
  tauto

end union_of_sets_l187_187409


namespace balls_in_boxes_with_one_empty_l187_187057

-- Question setup:
-- 5 distinguishable balls, 4 distinguishable boxes, at least one box must remain empty.
def num_ways_distribute_balls (balls boxes : ℕ) : ℕ :=
  let unrestricted := boxes ^ balls in
  let one_empty := boxes * (boxes - 1) ^ balls in
  let two_empty := (boxes * (boxes - 1) // 2) * (boxes - 2) ^ balls in
  let three_empty := boxes * (boxes - 1) // 6 * (boxes - 3) ^ balls in
  unrestricted - one_empty + two_empty - three_empty

-- The final theorem we want to prove
theorem balls_in_boxes_with_one_empty :
  num_ways_distribute_balls 5 4 = 240 :=
by {
  -- Skipping the actual proof steps
  sorry,
}

end balls_in_boxes_with_one_empty_l187_187057


namespace total_area_of_pyramid_faces_l187_187932

theorem total_area_of_pyramid_faces (b l : ℕ) (hb : b = 8) (hl : l = 10) : 
  let h : ℝ := Math.sqrt (l^2 - (b / 2)^2) in
  let A : ℝ := 1 / 2 * b * h in
  let T : ℝ := 4 * A in
  T = 32 * Math.sqrt 21 := by
  -- Definitions
  have b_val : (b : ℝ) = 8 := by exact_mod_cast hb
  have l_val : (l : ℝ) = 10 := by exact_mod_cast hl

  -- Calculations
  have h_val : h = Math.sqrt (l^2 - (b / 2)^2) := rfl
  have h_simplified : h = 2 * Math.sqrt 21 := by
    rw [h_val, l_val, b_val]
    norm_num
    simp

  have A_val : A = 1 / 2 * b * h := rfl
  simp_rw [A_val, h_simplified, b_val]
  norm_num

  have T_val : T = 4 * A := rfl
  simp_rw [T_val]
  norm_num

  -- Final proof
  sorry

end total_area_of_pyramid_faces_l187_187932


namespace maximum_value_is_17_l187_187754

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l187_187754


namespace ratio_of_ages_l187_187450

variables (R J K : ℕ)

axiom h1 : R = J + 8
axiom h2 : R + 4 = 2 * (J + 4)
axiom h3 : (R + 4) * (K + 4) = 192

theorem ratio_of_ages : (R - J) / (R - K) = 2 :=
by sorry

end ratio_of_ages_l187_187450


namespace max_expression_value_l187_187781

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l187_187781


namespace irreducible_poly_f_l187_187086

variable {R : Type*} [CommRing R]

noncomputable def poly_f (a : ℕ → ℤ) (n : ℕ) : Polynomial ℤ :=
  ∏ i in Finset.range n, Polynomial.X - Polynomial.C (a i) - 1

theorem irreducible_poly_f (a : ℕ → ℤ) (n : ℕ)
  (h_distinct : Function.Injective a) :
  Irreducible (poly_f a n) :=
sorry

end irreducible_poly_f_l187_187086


namespace prime_k_values_l187_187524

theorem prime_k_values (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Nat.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) :=
by
  sorry

end prime_k_values_l187_187524


namespace only_setB_is_proportional_l187_187808

-- Definitions for the line segments
def setA := (3, 4, 5, 6)
def setB := (5, 15, 2, 6)
def setC := (4, 8, 3, 5)
def setD := (8, 4, 1, 3)

-- Definition to check if a set of line segments is proportional
def is_proportional (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c, d) := s
  a * d = b * c

-- Theorem proving that the only proportional set is set B
theorem only_setB_is_proportional :
  is_proportional setA = false ∧
  is_proportional setB = true ∧
  is_proportional setC = false ∧
  is_proportional setD = false :=
by
  sorry

end only_setB_is_proportional_l187_187808


namespace stock_worth_l187_187151

theorem stock_worth (W : Real) 
  (profit_part : Real := 0.25 * W * 0.20)
  (loss_part1 : Real := 0.35 * W * 0.10)
  (loss_part2 : Real := 0.40 * W * 0.15)
  (overall_loss_eq : loss_part1 + loss_part2 - profit_part = 1200) : 
  W = 26666.67 :=
by
  sorry

end stock_worth_l187_187151


namespace angle_measures_possible_l187_187606

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l187_187606


namespace polynomial_evaluation_l187_187085

noncomputable def Q (x : ℝ) : ℝ :=
  x^4 + x^3 + 2 * x

theorem polynomial_evaluation :
  Q (3) = 114 := by
  -- We assume the conditions implicitly in this equivalence.
  sorry

end polynomial_evaluation_l187_187085


namespace first_pack_weight_l187_187370

-- Define the conditions
def miles_per_hour := 2.5
def hours_per_day := 8
def days := 5
def supply_per_mile := 0.5
def resupply_percentage := 0.25
def total_hiking_time := hours_per_day * days
def total_miles_hiked := total_hiking_time * miles_per_hour
def total_supplies_needed := total_miles_hiked * supply_per_mile
def resupply_factor := 1 + resupply_percentage

-- Define the theorem
theorem first_pack_weight :
  (total_supplies_needed / resupply_factor) = 40 :=
by
  sorry

end first_pack_weight_l187_187370


namespace minimum_number_is_correct_l187_187609

-- Define the operations and conditions on the digits
def transform (n : ℕ) : ℕ :=
if 2 ≤ n then n - 2 + 1 else n

noncomputable def minimum_transformed_number (l : List ℕ) : List ℕ :=
l.map transform

def initial_number : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9]

def expected_number : List ℕ := [1, 0, 1, 0, 1, 0, 1, 0, 1]

theorem minimum_number_is_correct :
  minimum_transformed_number initial_number = expected_number := 
by
  -- sorry is a placeholder for the proof
  sorry

end minimum_number_is_correct_l187_187609


namespace train_length_l187_187154

variable (L V : ℝ)

-- Given conditions
def condition1 : Prop := V = L / 24
def condition2 : Prop := V = (L + 650) / 89

theorem train_length : condition1 L V → condition2 L V → L = 240 := by
  intro h1 h2
  sorry

end train_length_l187_187154


namespace first_pack_weight_l187_187369

-- Define the conditions
def miles_per_hour := 2.5
def hours_per_day := 8
def days := 5
def supply_per_mile := 0.5
def resupply_percentage := 0.25
def total_hiking_time := hours_per_day * days
def total_miles_hiked := total_hiking_time * miles_per_hour
def total_supplies_needed := total_miles_hiked * supply_per_mile
def resupply_factor := 1 + resupply_percentage

-- Define the theorem
theorem first_pack_weight :
  (total_supplies_needed / resupply_factor) = 40 :=
by
  sorry

end first_pack_weight_l187_187369


namespace lines_parallel_l187_187998

-- Define line l1 and line l2
def l1 (x : ℝ) : ℝ := 2 * x + 1
def l2 (x : ℝ) : ℝ := 2 * x + 5

-- Prove that l1 is parallel to l2
theorem lines_parallel : ∀ x : ℝ, (l1 x - l2 x) = -4 := by
  sorry

end lines_parallel_l187_187998


namespace inequality_problem_l187_187562

theorem inequality_problem 
  (a b c : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (h : a + b + c + 2 = a * b * c) : 
  (a + 1) * (b + 1) * (c + 1) ≥ 27 := 
by sorry

end inequality_problem_l187_187562


namespace sqrt_200_simplified_l187_187270

-- Definitions based on conditions from part a)
def factorization : Nat := 2 ^ 3 * 5 ^ 2

lemma sqrt_property (a b : ℕ) : Real.sqrt (a^2 * b) = a * Real.sqrt b := sorry

-- The proof problem (only the statement, not the proof)
theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  have h1 : 200 = 2^3 * 5^2 := by rfl
  have h2 : Real.sqrt (200) = Real.sqrt (2^3 * 5^2) := by rw h1
  rw [←show 200 = factorization by rfl] at h2
  exact sorry

end sqrt_200_simplified_l187_187270


namespace circle_center_coordinates_l187_187106

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (x - h)^2 + (y + k)^2 = 5 :=
sorry

end circle_center_coordinates_l187_187106


namespace two_y_minus_three_x_l187_187878

variable (x y : ℝ)

noncomputable def x_val : ℝ := 1.2 * 98
noncomputable def y_val : ℝ := 0.9 * (x_val + 35)

theorem two_y_minus_three_x : 2 * y_val - 3 * x_val = -78.12 := by
  sorry

end two_y_minus_three_x_l187_187878


namespace midpoint_of_points_l187_187514

theorem midpoint_of_points (x1 y1 x2 y2 : ℝ) (h1 : x1 = 6) (h2 : y1 = 10) (h3 : x2 = 8) (h4 : y2 = 4) :
  ((x1 + x2) / 2, (y1 + y2) / 2) = (7, 7) := 
by
  rw [h1, h2, h3, h4]
  norm_num

end midpoint_of_points_l187_187514


namespace solve_abs_eq_l187_187649

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l187_187649


namespace find_angle_l187_187709

theorem find_angle (θ : Real) (h1 : 0 ≤ θ ∧ θ ≤ π) (h2 : Real.sin θ = (Real.sqrt 2) / 2) :
  θ = Real.pi / 4 ∨ θ = 3 * Real.pi / 4 :=
by
  sorry

end find_angle_l187_187709


namespace petya_vasya_three_numbers_equal_l187_187938

theorem petya_vasya_three_numbers_equal (a b c : ℕ) :
  gcd a b = lcm a b ∧ gcd b c = lcm b c ∧ gcd a c = lcm a c → a = b ∧ b = c :=
by
  sorry

end petya_vasya_three_numbers_equal_l187_187938


namespace distribution_of_cousins_l187_187904

theorem distribution_of_cousins (cousins bedrooms : ℕ) (empty_bedroom : ℕ) (h1 : cousins = 5) (h2 : bedrooms = 3) (h3 : empty_bedroom ≥ 1) : 
  ∑ k in { (a, b, c) | a + b + c = 5 ∧ (a == 0 ∨ b == 0 ∨ c == 0) }, 1 = 26 :=
by 
  sorry

end distribution_of_cousins_l187_187904


namespace cube_square_third_smallest_prime_l187_187320

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l187_187320


namespace salt_solution_l187_187139

variable (x : ℝ) (v_water : ℝ) (c_initial : ℝ) (c_final : ℝ)

theorem salt_solution (h1 : v_water = 1) (h2 : c_initial = 0.60) (h3 : c_final = 0.20)
  (h4 : (v_water + x) * c_final = x * c_initial) :
  x = 0.5 :=
by {
  sorry
}

end salt_solution_l187_187139


namespace initial_logs_l187_187675

theorem initial_logs (x : ℕ) (h1 : x - 3 - 3 - 3 + 2 + 2 + 2 = 3) : x = 6 := by
  sorry

end initial_logs_l187_187675


namespace stock_profit_percentage_l187_187955

theorem stock_profit_percentage 
  (total_stock : ℝ) (total_loss : ℝ) (profit_percentage : ℝ) (loss_percentage : ℝ)
  (percentage_sold_at_profit : ℝ) :
  total_stock = 12499.99 →
  total_loss = 500 →
  profit_percentage = 0.20 →
  loss_percentage = 0.10 →
  (0.10 * ((100 - percentage_sold_at_profit) / 100) * 12499.99) - (0.20 * (percentage_sold_at_profit / 100) * 12499.99) = 500 →
  percentage_sold_at_profit = 20 :=
sorry

end stock_profit_percentage_l187_187955


namespace length_A_l187_187084

noncomputable def A : ℝ × ℝ := (0, 15)
noncomputable def B : ℝ × ℝ := (0, 18)
noncomputable def C : ℝ × ℝ := (4, 10)

-- Prove the length of A'B' given the conditions
theorem length_A'B'_eq : 
  ∃ (A' B' : ℝ × ℝ), 
  (A'.1 = A'.2) ∧ (B'.1 = B'.2) ∧ 
  (4 - A'.1) * (C.2 - A'.2) = (10 - A'.2) * (C.1 - A'.1) ∧
  (4 - B'.1) * (C.2 - B'.2) = (10 - B'.2) * (C.1 - B'.1) ∧
  (real.sqrt ((A'.1 - B'.1) ^ 2 + (A'.2 - B'.2) ^ 2) = 2 * real.sqrt 2 / 3) :=
begin
  sorry
end

end length_A_l187_187084


namespace largest_prime_factor_2999_l187_187638

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) : ℕ :=
  -- Note: This would require actual computation logic to find the largest prime factor.
  sorry

theorem largest_prime_factor_2999 :
  largest_prime_factor 2999 = 103 :=
by 
  -- Given conditions:
  -- 1. 2999 is an odd number (doesn't need explicit condition in proof).
  -- 2. Sum of digits is 29, thus not divisible by 3.
  -- 3. 2999 is not divisible by 11.
  -- 4. 2999 is not divisible by 7, 13, 17, 19.
  -- 5. Prime factorization of 2999 is 29 * 103.
  admit -- actual proof will need detailed prime factor test results 

end largest_prime_factor_2999_l187_187638


namespace sum_of_decimals_l187_187380

theorem sum_of_decimals : 5.47 + 2.58 + 1.95 = 10.00 := by
  sorry

end sum_of_decimals_l187_187380


namespace max_value_of_fraction_l187_187766

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l187_187766


namespace Jillian_collected_29_l187_187081

variable (Savannah_shells Clayton_shells total_friends friend_shells : ℕ)

def Jillian_shells : ℕ :=
  let total_shells := friend_shells * total_friends
  let others_shells := Savannah_shells + Clayton_shells
  total_shells - others_shells

theorem Jillian_collected_29 (h_savannah : Savannah_shells = 17) 
                             (h_clayton : Clayton_shells = 8) 
                             (h_friends : total_friends = 2) 
                             (h_friend_shells : friend_shells = 27) : 
  Jillian_shells Savannah_shells Clayton_shells total_friends friend_shells = 29 :=
by
  sorry

end Jillian_collected_29_l187_187081


namespace pears_sold_in_afternoon_l187_187488

theorem pears_sold_in_afternoon (m a total : ℕ) (h1 : a = 2 * m) (h2 : m = 120) (h3 : m + a = total) (h4 : total = 360) :
  a = 240 :=
by
  sorry

end pears_sold_in_afternoon_l187_187488


namespace sculpture_paint_area_l187_187157

/-- An artist creates a sculpture using 15 cubes, each with a side length of 1 meter. 
The cubes are organized into a wall-like structure with three layers: 
the top layer consists of 3 cubes, 
the middle layer consists of 5 cubes, 
and the bottom layer consists of 7 cubes. 
Some of the cubes in the middle and bottom layers are spaced apart, exposing additional side faces. 
Prove that the total exposed surface area painted is 49 square meters. -/
theorem sculpture_paint_area :
  let cubes_sizes : ℕ := 15
  let layer_top : ℕ := 3
  let layer_middle : ℕ := 5
  let layer_bottom : ℕ := 7
  let side_exposed_area_layer_top : ℕ := layer_top * 5
  let side_exposed_area_layer_middle : ℕ := 2 * 3 + 3 * 2
  let side_exposed_area_layer_bottom : ℕ := layer_bottom * 1
  let exposed_side_faces : ℕ := side_exposed_area_layer_top + side_exposed_area_layer_middle + side_exposed_area_layer_bottom
  let exposed_top_faces : ℕ := layer_top * 1 + layer_middle * 1 + layer_bottom * 1
  let total_exposed_area : ℕ := exposed_side_faces + exposed_top_faces
  total_exposed_area = 49 := 
sorry

end sculpture_paint_area_l187_187157


namespace angle_difference_proof_l187_187879

-- Define the angles A and B
def angle_A : ℝ := 65
def angle_B : ℝ := 180 - angle_A

-- Define the difference
def angle_difference : ℝ := angle_B - angle_A

theorem angle_difference_proof : angle_difference = 50 :=
by
  -- The proof goes here
  sorry

end angle_difference_proof_l187_187879


namespace number_of_tetrises_l187_187152

theorem number_of_tetrises 
  (points_per_single : ℕ := 1000)
  (points_per_tetris : ℕ := 8 * points_per_single)
  (singles_scored : ℕ := 6)
  (total_score : ℕ := 38000) :
  (total_score - (singles_scored * points_per_single)) / points_per_tetris = 4 := 
by 
  sorry

end number_of_tetrises_l187_187152


namespace John_spent_fraction_toy_store_l187_187203

variable (weekly_allowance arcade_money toy_store_money candy_store_money : ℝ)
variable (spend_fraction : ℝ)

-- John's conditions
def John_conditions : Prop :=
  weekly_allowance = 3.45 ∧
  arcade_money = 3 / 5 * weekly_allowance ∧
  candy_store_money = 0.92 ∧
  toy_store_money = weekly_allowance - arcade_money - candy_store_money

-- Theorem to prove the fraction spent at the toy store
theorem John_spent_fraction_toy_store :
  John_conditions weekly_allowance arcade_money toy_store_money candy_store_money →
  spend_fraction = toy_store_money / (weekly_allowance - arcade_money) →
  spend_fraction = 1 / 3 :=
by
  sorry

end John_spent_fraction_toy_store_l187_187203


namespace olivia_earnings_this_week_l187_187238

variable (hourly_rate : ℕ) (hours_monday hours_wednesday hours_friday : ℕ)

theorem olivia_earnings_this_week : 
  hourly_rate = 9 → 
  hours_monday = 4 → 
  hours_wednesday = 3 → 
  hours_friday = 6 → 
  (hourly_rate * hours_monday + hourly_rate * hours_wednesday + hourly_rate * hours_friday) = 117 := 
by
  intros
  sorry

end olivia_earnings_this_week_l187_187238


namespace range_of_a_l187_187456

-- Define the function f
def f (a x : ℝ) : ℝ := a * x ^ 2 + a * x - 1

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x < 0) ↔ -4 < a ∧ a ≤ 0 :=
by {
  sorry
}

end range_of_a_l187_187456


namespace number_of_one_dollar_coins_l187_187054

theorem number_of_one_dollar_coins (t : ℕ) :
  (∃ k : ℕ, 3 * k = t) → ∃ k : ℕ, k = t / 3 :=
by
  sorry

end number_of_one_dollar_coins_l187_187054


namespace molly_age_l187_187940

variable (S M : ℕ)

theorem molly_age (h1 : S / M = 4 / 3) (h2 : S + 6 = 38) : M = 24 :=
by
  sorry

end molly_age_l187_187940


namespace max_items_for_2019_students_l187_187477

noncomputable def max_items (students : ℕ) : ℕ :=
  students / 2

theorem max_items_for_2019_students : max_items 2019 = 1009 := by
  sorry

end max_items_for_2019_students_l187_187477


namespace sum_integers_50_to_75_l187_187929

theorem sum_integers_50_to_75 : (Finset.range 26).sum (λ i, 50 + i) = 1625 :=
by
  sorry

end sum_integers_50_to_75_l187_187929


namespace cube_square_third_smallest_prime_l187_187321

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l187_187321


namespace parallelogram_point_D_l187_187723

/-- Given points A, B, and C, the coordinates of point D in parallelogram ABCD -/
theorem parallelogram_point_D (A B C D : (ℝ × ℝ))
  (hA : A = (1, 1))
  (hB : B = (3, 2))
  (hC : C = (6, 3))
  (hMid : (2 * (A.1 + C.1), 2 * (A.2 + C.2)) = (2 * (B.1 + D.1), 2 * (B.2 + D.2))) :
  D = (4, 2) :=
sorry

end parallelogram_point_D_l187_187723


namespace smallest_number_diminished_by_10_divisible_l187_187467

theorem smallest_number_diminished_by_10_divisible :
  ∃ (x : ℕ), (x - 10) % 24 = 0 ∧ x = 34 :=
by
  sorry

end smallest_number_diminished_by_10_divisible_l187_187467


namespace bacon_vs_tomatoes_l187_187009

theorem bacon_vs_tomatoes :
  let (n_b : ℕ) := 337
  let (n_t : ℕ) := 23
  n_b - n_t = 314 := by
  let n_b := 337
  let n_t := 23
  have h1 : n_b = 337 := rfl
  have h2 : n_t = 23 := rfl
  sorry

end bacon_vs_tomatoes_l187_187009


namespace remainder_of_sum_l187_187555

theorem remainder_of_sum (p q : ℤ) (c d : ℤ) 
  (hc : c = 100 * p + 78)
  (hd : d = 150 * q + 123) :
  (c + d) % 50 = 1 :=
sorry

end remainder_of_sum_l187_187555


namespace train_speed_correct_l187_187153

def train_length : ℝ := 1500
def crossing_time : ℝ := 15
def correct_speed : ℝ := 100

theorem train_speed_correct : (train_length / crossing_time) = correct_speed := by 
  sorry

end train_speed_correct_l187_187153


namespace james_worked_41_hours_l187_187022

theorem james_worked_41_hours (x : ℝ) :
  ∃ (J : ℕ), 
    (24 * x + 12 * 1.5 * x = 40 * x + (J - 40) * 2 * x) ∧ 
    J = 41 := 
by 
  sorry

end james_worked_41_hours_l187_187022


namespace find_x_l187_187211

theorem find_x (x y : ℤ) (h1 : x + y = 24) (h2 : x - y = 40) : x = 32 :=
by
  sorry

end find_x_l187_187211


namespace sequence_arithmetic_condition_l187_187848

theorem sequence_arithmetic_condition {α β : ℝ} (hα : α ≠ 0) (hβ : β ≠ 0) (hαβ : α + β ≠ 0)
  (seq : ℕ → ℝ) (hseq : ∀ n, seq (n + 2) = (α * seq (n + 1) + β * seq n) / (α + β)) :
  ∃ α β : ℝ, (∀ a1 a2 : ℝ, α ≠ 0 ∧ β ≠ 0 ∧ α + β = 0 → seq (n + 1) - seq n = seq n - seq (n - 1)) :=
by sorry

end sequence_arithmetic_condition_l187_187848


namespace pyramid_total_blocks_l187_187008

-- Define the number of layers in the pyramid
def num_layers : ℕ := 8

-- Define the block multiplier for each subsequent layer
def block_multiplier : ℕ := 5

-- Define the number of blocks in the top layer
def top_layer_blocks : ℕ := 3

-- Define the total number of sandstone blocks
def total_blocks_pyramid : ℕ :=
  let rec total_blocks (layer : ℕ) (blocks : ℕ) :=
    if layer = 0 then blocks
    else blocks + total_blocks (layer - 1) (blocks * block_multiplier)
  total_blocks (num_layers - 1) top_layer_blocks

theorem pyramid_total_blocks :
  total_blocks_pyramid = 312093 :=
by
  -- Proof omitted
  sorry

end pyramid_total_blocks_l187_187008


namespace inequality_holds_iff_even_l187_187088

theorem inequality_holds_iff_even (a b c : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (∀ x y z : ℝ, (x - y) ^ a * (x - z) ^ b * (y - z) ^ c ≥ 0) ↔ (Even a ∧ Even b ∧ Even c) :=
by
  sorry

end inequality_holds_iff_even_l187_187088


namespace area_of_triangle_PQR_l187_187804

def Point := (ℝ × ℝ)
def area_of_triangle (P Q R : Point) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

def P : Point := (1, 1)
def Q : Point := (4, 5)
def R : Point := (7, 2)

theorem area_of_triangle_PQR :
  area_of_triangle P Q R = 10.5 := by
  sorry

end area_of_triangle_PQR_l187_187804


namespace max_value_of_fraction_l187_187782

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l187_187782


namespace min_students_l187_187741

theorem min_students (S a b c : ℕ) (h1 : 3 * a > S) (h2 : 10 * b > 3 * S) (h3 : 11 * c > 4 * S) (h4 : S = a + b + c) : S ≥ 173 :=
by
  sorry

end min_students_l187_187741


namespace product_and_quotient_l187_187381

theorem product_and_quotient : (16 * 0.0625 / 4 * 0.5 * 2) = (1 / 4) :=
by
  -- The proof steps would go here
  sorry

end product_and_quotient_l187_187381


namespace cube_of_square_of_third_smallest_prime_l187_187310

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l187_187310


namespace at_most_two_even_l187_187121

-- Assuming the negation of the proposition
def negate_condition (a b c : ℕ) : Prop := a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 = 0

-- Proposition to prove by contradiction
theorem at_most_two_even 
  (a b c : ℕ) 
  (h : negate_condition a b c) 
  : False :=
sorry

end at_most_two_even_l187_187121


namespace inequality_proof_l187_187349

open Real

theorem inequality_proof (x y z: ℝ) (hx: 0 ≤ x) (hy: 0 ≤ y) (hz: 0 ≤ z) : 
  (x^3 / (x^3 + 2 * y^2 * sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * sqrt (y * z)) 
  ) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187349


namespace inequality_proof_l187_187347

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l187_187347


namespace proof_problem_l187_187561

theorem proof_problem (a b c : ℝ) (h1 : 4 * a - 2 * b + c > 0) (h2 : a + b + c < 0) : b^2 > a * c :=
sorry

end proof_problem_l187_187561


namespace max_value_of_fraction_l187_187784

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l187_187784


namespace kaleb_toys_l187_187558

def initial_savings : ℕ := 21
def allowance : ℕ := 15
def cost_per_toy : ℕ := 6

theorem kaleb_toys : (initial_savings + allowance) / cost_per_toy = 6 :=
by
  sorry

end kaleb_toys_l187_187558


namespace sqrt_200_eq_l187_187284

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l187_187284


namespace bryan_books_l187_187159

theorem bryan_books (books_per_continent : ℕ) (total_books : ℕ) 
  (h1 : books_per_continent = 122) 
  (h2 : total_books = 488) : 
  total_books / books_per_continent = 4 := 
by 
  sorry

end bryan_books_l187_187159


namespace probability_target_hit_l187_187110

theorem probability_target_hit (P_A P_B : ℚ) (h1 : P_A = 1/2) (h2 : P_B = 1/3) : 
  (1 - (1 - P_A) * (1 - P_B)) = 2/3 :=
by
  sorry

end probability_target_hit_l187_187110


namespace product_square_preceding_div_by_12_l187_187907

theorem product_square_preceding_div_by_12 (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) :=
by
  sorry

end product_square_preceding_div_by_12_l187_187907


namespace side_length_of_largest_square_correct_l187_187186

noncomputable def side_length_of_largest_square (A B C : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AC : ℝ) (CB : ℝ) : ℝ := 
  if h : (AC = 3) ∧ (CB = 7) then 2.1 else 0  -- Replace with correct proof

theorem side_length_of_largest_square_correct : side_length_of_largest_square A B C 3 7 = 2.1 :=
by
  sorry

end side_length_of_largest_square_correct_l187_187186


namespace multiply_fractions_l187_187834

theorem multiply_fractions :
  (1 / 3) * (4 / 7) * (9 / 13) * (2 / 5) = 72 / 1365 :=
by sorry

end multiply_fractions_l187_187834


namespace symmetric_circle_eq_l187_187033

theorem symmetric_circle_eq :
  ∀ (x y : ℝ),
  ((x + 2)^2 + y^2 = 5) →
  (x - y + 1 = 0) →
  (∃ (a b : ℝ), ((a + 1)^2 + (b + 1)^2 = 5)) := 
by
  intros x y h_circle h_line
  -- skip the proof
  sorry

end symmetric_circle_eq_l187_187033


namespace max_value_of_expression_l187_187774

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l187_187774


namespace clark_family_ticket_cost_l187_187520

theorem clark_family_ticket_cost
  (regular_price children's_price seniors_price : ℝ)
  (number_youngest_gen number_second_youngest_gen number_second_oldest_gen number_oldest_gen : ℕ)
  (h_senior_discount : seniors_price = 0.7 * regular_price)
  (h_senior_ticket_cost : seniors_price = 7)
  (h_child_discount : children's_price = 0.6 * regular_price)
  (h_number_youngest_gen : number_youngest_gen = 3)
  (h_number_second_youngest_gen : number_second_youngest_gen = 1)
  (h_number_second_oldest_gen : number_second_oldest_gen = 2)
  (h_number_oldest_gen : number_oldest_gen = 1)
  : 3 * children's_price + 1 * regular_price + 2 * seniors_price + 1 * regular_price = 52 := by
  sorry

end clark_family_ticket_cost_l187_187520


namespace sqrt_200_eq_10_sqrt_2_l187_187254

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l187_187254


namespace socks_cost_5_l187_187056

theorem socks_cost_5
  (jeans t_shirt socks : ℕ)
  (h1 : jeans = 2 * t_shirt)
  (h2 : t_shirt = socks + 10)
  (h3 : jeans = 30) :
  socks = 5 :=
by
  sorry

end socks_cost_5_l187_187056


namespace monic_polynomial_root_equivalence_l187_187897

noncomputable def roots (p : Polynomial ℝ) : List ℝ := sorry

theorem monic_polynomial_root_equivalence :
  let r1 := roots (Polynomial.C (8:ℝ) + Polynomial.X^3 - 3 * Polynomial.X^2)
  let p := Polynomial.C (216:ℝ) + Polynomial.X^3 - 9 * Polynomial.X^2
  r1.map (fun r => 3*r) = roots p :=
by
  sorry

end monic_polynomial_root_equivalence_l187_187897


namespace axes_positioning_l187_187373

noncomputable def f (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

theorem axes_positioning (a b c : ℝ) (h_a : a > 0) (h_b : b > 0) (h_c : c < 0) :
  ∃ x_vertex y_intercept, x_vertex < 0 ∧ y_intercept < 0 ∧ (∀ x, f a b c x > f a b c x) :=
by
  sorry

end axes_positioning_l187_187373


namespace div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l187_187697

theorem div_by_3_9_then_mul_by_5_6_eq_div_by_5_2 :
  (∀ (x : ℚ), (x / (3/9)) * (5/6) = x / (5/2)) :=
by
  sorry

end div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l187_187697


namespace totalProblemsSolved_l187_187491

-- Given conditions
def initialProblemsSolved : Nat := 45
def additionalProblemsSolved : Nat := 18

-- Statement to prove the total problems solved equals 63
theorem totalProblemsSolved : initialProblemsSolved + additionalProblemsSolved = 63 := 
by
  sorry

end totalProblemsSolved_l187_187491


namespace children_count_l187_187843

theorem children_count (C : ℕ) 
    (cons : ℕ := 12)
    (total_cost : ℕ := 76)
    (child_ticket_cost : ℕ := 7)
    (adult_ticket_cost : ℕ := 10)
    (num_adults : ℕ := 5)
    (adult_cost := num_adults * adult_ticket_cost)
    (cost_with_concessions := total_cost - adult_cost )
    (children_cost := cost_with_concessions - cons):
    C = children_cost / child_ticket_cost :=
by
    sorry

end children_count_l187_187843


namespace transformation_of_95_squared_l187_187470

theorem transformation_of_95_squared :
  (9.5 : ℝ) ^ 2 = (10 : ℝ) ^ 2 - 2 * (10 : ℝ) * (0.5 : ℝ) + (0.5 : ℝ) ^ 2 :=
by
  sorry

end transformation_of_95_squared_l187_187470


namespace angle_remains_unchanged_l187_187803

-- Definition of magnification condition (though it does not affect angle in mathematics, we state it as given)
def magnifying_glass (magnification : ℝ) (initial_angle : ℝ) : ℝ := 
  initial_angle  -- Magnification does not change the angle in this context.

-- Given condition
def initial_angle : ℝ := 30

-- Theorem we want to prove
theorem angle_remains_unchanged (magnification : ℝ) (h_magnify : magnification = 100) :
  magnifying_glass magnification initial_angle = initial_angle :=
by
  sorry

end angle_remains_unchanged_l187_187803


namespace pyramid_total_area_l187_187933

theorem pyramid_total_area
  (base_edge : ℝ)
  (lateral_edge : ℝ)
  (base_edge_eq : base_edge = 8)
  (lateral_edge_eq : lateral_edge = 10)
  : 4 * (1 / 2 * base_edge * sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2)) = 32 * sqrt 21 := by
  sorry

end pyramid_total_area_l187_187933


namespace solve_absolute_value_eq_l187_187652

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l187_187652


namespace exponent_equality_l187_187863

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
by
  sorry

end exponent_equality_l187_187863


namespace inequality_proof_l187_187343

variable {ℝ : Type*} [LinearOrderedField ℝ]

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x)) + 
   y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y)) + 
   z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 := 
by
  sorry

end inequality_proof_l187_187343


namespace passes_to_left_l187_187149

theorem passes_to_left
  (total_passes passes_left passes_right passes_center : ℕ)
  (h1 : total_passes = 50)
  (h2 : passes_right = 2 * passes_left)
  (h3 : passes_center = passes_left + 2)
  (h4 : total_passes = passes_left + passes_right + passes_center) :
  passes_left = 12 :=
by
  sorry

end passes_to_left_l187_187149


namespace secretary_worked_longest_l187_187664

theorem secretary_worked_longest
  (h1 : ∀ (x : ℕ), 3 * x + 5 * x + 7 * x + 11 * x = 2080)
  (h2 : ∀ (a b c d : ℕ), a = 3 * x ∧ b = 5 * x ∧ c = 7 * x ∧ d = 11 * x → d = 11 * x):
  ∃ y : ℕ, y = 880 :=
by
  sorry

end secretary_worked_longest_l187_187664


namespace negation_of_p_l187_187412

theorem negation_of_p (p : Prop) :
  (¬ (∀ (a : ℝ), a ≥ 0 → a^4 + a^2 ≥ 0)) ↔ (∃ (a : ℝ), a ≥ 0 ∧ a^4 + a^2 < 0) := 
by
  sorry

end negation_of_p_l187_187412


namespace inequality_xyz_l187_187339

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l187_187339


namespace second_company_managers_percent_l187_187946

/-- A company's workforce consists of 10 percent managers and 90 percent software engineers.
    Another company's workforce consists of some percent managers, 10 percent software engineers, 
    and 60 percent support staff. The two companies merge, and the resulting company's 
    workforce consists of 25 percent managers. If 25 percent of the workforce originated from the 
    first company, what percent of the second company's workforce were managers? -/
theorem second_company_managers_percent
  (F S : ℝ)
  (h1 : 0.10 * F + m * S = 0.25 * (F + S))
  (h2 : F = 0.25 * (F + S)) :
  m = 0.225 :=
sorry

end second_company_managers_percent_l187_187946


namespace trigonometric_identity_l187_187871

theorem trigonometric_identity (α : ℝ) (h : Real.sin (π / 4 + α) = 1 / 2) : 
  (Real.sin (5 * π / 4 + α) / Real.cos (9 * π / 4 + α)) * Real.cos (7 * π / 4 - α) = -1 / 2 :=
by
  sorry

end trigonometric_identity_l187_187871


namespace sum_of_pqrstu_eq_22_l187_187731

theorem sum_of_pqrstu_eq_22 (p q r s t : ℤ) 
  (h : (7 - p) * (7 - q) * (7 - r) * (7 - s) * (7 - t) = -48) : 
  p + q + r + s + t = 22 :=
sorry

end sum_of_pqrstu_eq_22_l187_187731


namespace coeffs_divisible_by_5_l187_187584

theorem coeffs_divisible_by_5
  (a b c d : ℤ)
  (h1 : a + b + c + d ≡ 0 [ZMOD 5])
  (h2 : -a + b - c + d ≡ 0 [ZMOD 5])
  (h3 : 8 * a + 4 * b + 2 * c + d ≡ 0 [ZMOD 5])
  (h4 : d ≡ 0 [ZMOD 5]) :
  a ≡ 0 [ZMOD 5] ∧ b ≡ 0 [ZMOD 5] ∧ c ≡ 0 [ZMOD 5] ∧ d ≡ 0 [ZMOD 5] :=
sorry

end coeffs_divisible_by_5_l187_187584


namespace parallel_lines_have_equal_slopes_l187_187851

theorem parallel_lines_have_equal_slopes (m : ℝ) :
  (∀ x y : ℝ, x + 2 * y - 1 = 0 → mx - y = 0) → m = -1 / 2 :=
by
  sorry

end parallel_lines_have_equal_slopes_l187_187851


namespace cyclists_original_number_l187_187142

theorem cyclists_original_number (x : ℕ) (h : x > 2) : 
  (80 / (x - 2 : ℕ) = 80 / x + 2) → x = 10 :=
by
  sorry

end cyclists_original_number_l187_187142


namespace limit_f_at_zero_l187_187943

open Real Filter

noncomputable def f (x : ℝ) : ℝ :=
  (sqrt (1 + tan x) - sqrt (1 + sin x)) / (x^3)

theorem limit_f_at_zero :
  tendsto f (nhds 0) (nhds (1/4)) :=
begin
  sorry
end

end limit_f_at_zero_l187_187943


namespace example_problem_l187_187200

-- Define vectors a and b with the given conditions
def a (k : ℝ) : ℝ × ℝ := (2, k)
def b : ℝ × ℝ := (6, 4)

-- Define the condition that vectors are perpendicular
def perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  v1.1 * v2.1 + v1.2 * v2.2 = 0

-- Calculate the sum of two vectors
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- Check if a vector is collinear
def collinear (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ c : ℝ, v1 = (c * v2.1, c * v2.2)

-- The main theorem with the given conditions
theorem example_problem (k : ℝ) (hk : perpendicular (a k) b) :
  collinear (vector_add (a k) b) (-16, -2) :=
by
  sorry

end example_problem_l187_187200


namespace probability_queen_then_club_l187_187800

-- Define the problem conditions using the definitions
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_clubs : ℕ := 13
def num_club_queens : ℕ := 1

-- Define a function that computes the probability of the given event
def probability_first_queen_second_club : ℚ :=
  let prob_first_club_queen := (num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_club_queen := (num_clubs - 1 : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_1 := prob_first_club_queen * prob_second_club_given_first_club_queen
  let prob_first_non_club_queen := (num_queens - num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_non_club_queen := (num_clubs : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_2 := prob_first_non_club_queen * prob_second_club_given_first_non_club_queen
  prob_case_1 + prob_case_2

-- The statement to be proved
theorem probability_queen_then_club : probability_first_queen_second_club = 1 / 52 := by
  sorry

end probability_queen_then_club_l187_187800


namespace total_cents_l187_187435

/-
Given:
1. Lance has 70 cents.
2. Margaret has three-fourths of a dollar.
3. Guy has two quarters and a dime.
4. Bill has six dimes.

Prove:
The combined total amount of money they have is 265 cents.
-/
theorem total_cents (lance margaret guy bill : ℕ) 
  (hl : lance = 70)
  (hm : margaret = 3 * 100 / 4) -- Margaret's cents
  (hg : guy = 2 * 25 + 10)      -- Guy's cents
  (hb : bill = 6 * 10)          -- Bill's cents
  : lance + margaret + guy + bill = 265 :=
by
  rw [hl, hm, hg, hb]
  norm_num
  sorry

end total_cents_l187_187435


namespace seven_digit_divisible_by_11_l187_187168

def is_digit (d : ℕ) : Prop := d ≤ 9

def valid7DigitNumber (b n : ℕ) : Prop :=
  let sum_odd := 3 + 5 + 6
  let sum_even := b + n + 7 + 8
  let diff := sum_odd - sum_even
  diff % 11 = 0

theorem seven_digit_divisible_by_11 (b n : ℕ) (hb : is_digit b) (hn : is_digit n)
  (h_valid : valid7DigitNumber b n) : b + n = 10 := 
sorry

end seven_digit_divisible_by_11_l187_187168


namespace solve_abs_eq_l187_187647

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l187_187647


namespace stream_speed_l187_187366

theorem stream_speed (v : ℝ) (t : ℝ) (h1 : t > 0)
  (h2 : ∃ k : ℝ, k = 2 * t)
  (h3 : (9 + v) * t = (9 - v) * (2 * t)) :
  v = 3 := 
sorry

end stream_speed_l187_187366


namespace number_of_girls_l187_187622

theorem number_of_girls (total_children : ℕ) (probability : ℚ) (boys : ℕ) (girls : ℕ)
  (h_total_children : total_children = 25)
  (h_probability : probability = 3 / 25)
  (h_boys : boys * (boys - 1) = 72) :
  girls = total_children - boys :=
by {
  have h_total_children_def : total_children = 25 := h_total_children,
  have h_boys_def : boys * (boys - 1) = 72 := h_boys,
  have h_boys_sol := Nat.solve_quad_eq_pos 1 (-1) (-72),
  cases h_boys_sol with n h_n,
  cases h_n with h_n_pos h_n_eq,
  have h_pos_sol : 9 * (9 - 1) = 72 := by norm_num,
  have h_not_neg : n = 9 := h_n_eq.resolve_right (λ h_neg, by linarith),
  calc 
    girls = total_children - boys : by refl
    ... = 25 - 9 : by rw [h_total_children_def, h_not_neg] -- using n value
}
sorry

end number_of_girls_l187_187622


namespace a_1995_is_squared_l187_187134

variable (a : ℕ → ℕ)

-- Conditions on the sequence 
axiom seq_condition  {m n : ℕ} (h : m ≥ n) : 
  a (m + n) + a (m - n) = (a (2 * m) + a (2 * n)) / 2

axiom initial_value : a 1 = 1

-- Goal to prove
theorem a_1995_is_squared : a 1995 = 1995^2 :=
sorry

end a_1995_is_squared_l187_187134


namespace evaluate_expression_l187_187694

theorem evaluate_expression : (3^2 - 3) + (4^2 - 4) - (5^2 - 5) = -2 := by
  sorry

end evaluate_expression_l187_187694


namespace abc_divides_sum_pow_31_l187_187890

theorem abc_divides_sum_pow_31 (a b c : ℕ) 
  (h1 : a ∣ b^5)
  (h2 : b ∣ c^5)
  (h3 : c ∣ a^5) : 
  abc ∣ (a + b + c) ^ 31 := 
sorry

end abc_divides_sum_pow_31_l187_187890


namespace difference_of_squares_l187_187014

-- Definition of the constants a and b as given in the problem
def a := 502
def b := 498

theorem difference_of_squares : a^2 - b^2 = 4000 := by
  sorry

end difference_of_squares_l187_187014


namespace mass_percentage_of_Ba_l187_187554

theorem mass_percentage_of_Ba {BaX : Type} {molar_mass_Ba : ℝ} {compound_mass : ℝ} {mass_Ba : ℝ}:
  molar_mass_Ba = 137.33 ∧ 
  compound_mass = 100 ∧
  mass_Ba = 66.18 →
  (mass_Ba / compound_mass * 100) = 66.18 :=
by
  sorry

end mass_percentage_of_Ba_l187_187554


namespace base_comparison_l187_187161

theorem base_comparison : (1 * 6^1 + 2 * 6^0) > (1 * 2^2 + 0 * 2^1 + 1 * 2^0) := by
  sorry

end base_comparison_l187_187161


namespace find_sides_of_triangle_l187_187679

theorem find_sides_of_triangle (c : ℝ) (θ : ℝ) (h_ratio : ℝ) 
  (h_c : c = 2 * Real.sqrt 7)
  (h_theta : θ = Real.pi / 6) -- 30 degrees in radians
  (h_ratio_eq : ∃ k : ℝ, ∀ a b : ℝ, a = k ∧ b = h_ratio * k) :
  ∃ (a b : ℝ), a = 2 ∧ b = 4 * Real.sqrt 3 := by
  sorry

end find_sides_of_triangle_l187_187679


namespace escher_prints_consecutive_l187_187740

noncomputable def probability_all_eschers_consecutive (n : ℕ) (m : ℕ) (k : ℕ) : ℚ :=
if h : m = n + 3 ∧ k = 4 then 1 / (n * (n + 1) * (n + 2)) else 0

theorem escher_prints_consecutive :
  probability_all_eschers_consecutive 10 12 4 = 1 / 1320 :=
  by sorry

end escher_prints_consecutive_l187_187740


namespace josh_paid_6_dollars_l187_187557

def packs : ℕ := 3
def cheesePerPack : ℕ := 20
def costPerCheese : ℕ := 10 -- cost in cents

theorem josh_paid_6_dollars :
  (packs * cheesePerPack * costPerCheese) / 100 = 6 :=
by
  sorry

end josh_paid_6_dollars_l187_187557


namespace systematic_sampling_removal_count_l187_187957

-- Define the conditions
def total_population : Nat := 1252
def sample_size : Nat := 50

-- Define the remainder after division
def remainder := total_population % sample_size

-- Proof statement
theorem systematic_sampling_removal_count :
  remainder = 2 := by
    sorry

end systematic_sampling_removal_count_l187_187957


namespace problem_statement_l187_187892

def setS : Set (ℝ × ℝ) := {p | p.1 * p.2 > 0}
def setT : Set (ℝ × ℝ) := {p | p.1 > 0 ∧ p.2 > 0}

theorem problem_statement : setS ∪ setT = setS ∧ setS ∩ setT = setT :=
by
  -- To be proved
  sorry

end problem_statement_l187_187892


namespace probability_odd_multiple_of_5_l187_187118

theorem probability_odd_multiple_of_5 :
  (∃ (a b c : ℕ), 1 ≤ a ∧ a ≤ 100 ∧ 1 ≤ b ∧ b ≤ 100 ∧ 1 ≤ c ∧ c ≤ 100 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
    (a * b * c) % 2 = 1 ∧ (a * b * c) % 5 = 0) → 
  p = 3 / 125 := 
sorry

end probability_odd_multiple_of_5_l187_187118


namespace visiting_plans_count_l187_187838

-- Let's define the exhibitions
inductive Exhibition
| OperaCultureExhibition
| MingDynastyImperialCellarPorcelainExhibition
| AncientGreenLandscapePaintingExhibition
| ZhaoMengfuCalligraphyAndPaintingExhibition

open Exhibition

-- The condition is that the student must visit at least one painting exhibition in the morning and another in the afternoon
-- Proof that the number of different visiting plans is 10.
theorem visiting_plans_count :
  let exhibitions := [OperaCultureExhibition, MingDynastyImperialCellarPorcelainExhibition, AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  let painting_exhibitions := [AncientGreenLandscapePaintingExhibition, ZhaoMengfuCalligraphyAndPaintingExhibition]
  ∃ visits : List (Exhibition × Exhibition), (∀ (m a : Exhibition), (m ∈ painting_exhibitions ∨ a ∈ painting_exhibitions)) → visits.length = 10 :=
sorry

end visiting_plans_count_l187_187838


namespace xy_squared_value_l187_187873

theorem xy_squared_value (x y : ℝ) (h1 : x * (x + y) = 22) (h2 : y * (x + y) = 78 - y) :
  (x + y) ^ 2 = 100 :=
  sorry

end xy_squared_value_l187_187873


namespace sqrt_200_eq_10_sqrt_2_l187_187261

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l187_187261


namespace problem_inequality_l187_187405

theorem problem_inequality (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
    (y + z) / (2 * x) + (z + x) / (2 * y) + (x + y) / (2 * z) ≥
    2 * x / (y + z) + 2 * y / (z + x) + 2 * z / (x + y) :=
by
  sorry

end problem_inequality_l187_187405


namespace fg_of_2_l187_187537

def f (x : ℤ) : ℤ := 4 * x + 3
def g (x : ℤ) : ℤ := x ^ 3 + 1

theorem fg_of_2 : f (g 2) = 39 := by
  sorry

end fg_of_2_l187_187537


namespace smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l187_187855

noncomputable def f (x : ℝ) : ℝ := 4 * tan x * sin (π / 2 - x) * cos (x - π / 3) - sqrt 3

theorem smallest_positive_period_of_f :
  ∃ (p : ℝ), p > 0 ∧ ∀ x : ℝ, f (x + p) = f x ∧ p = π :=
sorry

theorem intervals_where_f_is_monotonically_increasing :
  ∃ (k : ℤ), ∀ x : ℝ, (k * π - π / 12 ≤ x ∧ x ≤ k * π + 5 * π / 12) → f' x > 0 :=
sorry

end smallest_positive_period_of_f_intervals_where_f_is_monotonically_increasing_l187_187855


namespace impossible_a_values_l187_187876

theorem impossible_a_values (a : ℝ) :
  ¬((1-a)^2 + (1+a)^2 < 4) → (a ≤ -1 ∨ a ≥ 1) :=
by
  sorry

end impossible_a_values_l187_187876


namespace simplify_sqrt_200_l187_187278

theorem simplify_sqrt_200 : (sqrt 200 : ℝ) = 10 * sqrt 2 := by
  -- proof goes here
  sorry

end simplify_sqrt_200_l187_187278


namespace choir_members_count_l187_187792

theorem choir_members_count (n : ℕ) (h1 : n % 10 = 4) (h2 : n % 11 = 5) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 234 := 
sorry

end choir_members_count_l187_187792


namespace fraction_equality_l187_187816
-- Import the necessary library

-- The proof statement
theorem fraction_equality : (16 + 8) / (4 - 2) = 12 := 
by {
  -- Inserting 'sorry' to indicate that the proof is omitted
  sorry
}

end fraction_equality_l187_187816


namespace socks_selection_l187_187058

/-!
  # Socks Selection Problem
  Prove the total number of ways to choose a pair of socks of different colors
  given:
  1. there are 5 white socks,
  2. there are 4 brown socks,
  3. there are 3 blue socks,
  is equal to 47.
-/

theorem socks_selection : 
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  5 * 4 + 4 * 3 + 5 * 3 = 47 :=
by
  let white_socks := 5
  let brown_socks := 4
  let blue_socks := 3
  sorry

end socks_selection_l187_187058


namespace max_expression_value_l187_187780

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l187_187780


namespace side_length_of_largest_square_correct_l187_187185

noncomputable def side_length_of_largest_square (A B C : Type) 
  [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (AC : ℝ) (CB : ℝ) : ℝ := 
  if h : (AC = 3) ∧ (CB = 7) then 2.1 else 0  -- Replace with correct proof

theorem side_length_of_largest_square_correct : side_length_of_largest_square A B C 3 7 = 2.1 :=
by
  sorry

end side_length_of_largest_square_correct_l187_187185


namespace total_messages_l187_187069

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l187_187069


namespace Donovan_Mitchell_goal_average_l187_187986

theorem Donovan_Mitchell_goal_average 
  (current_avg_pg : ℕ)     -- Donovan's current average points per game.
  (played_games : ℕ)       -- Number of games played so far.
  (required_avg_pg : ℕ)    -- Required average points per game in remaining games.
  (total_games : ℕ)        -- Total number of games in the season.
  (goal_avg_pg : ℕ)        -- Goal average points per game for the entire season.
  (H1 : current_avg_pg = 26)
  (H2 : played_games = 15)
  (H3 : required_avg_pg = 42)
  (H4 : total_games = 20) :
  goal_avg_pg = 30 :=
by
  sorry

end Donovan_Mitchell_goal_average_l187_187986


namespace f_7_5_l187_187896

noncomputable def f : ℝ → ℝ := sorry

axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_periodic : ∀ x : ℝ, f (x + 2) = -f x
axiom f_interval : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f x = x

theorem f_7_5 : f 7.5 = -0.5 := by
  sorry

end f_7_5_l187_187896


namespace no_m_for_P_eq_S_m_le_3_for_P_implies_S_l187_187846

namespace ProofProblem

def P (x : ℝ) : Prop := x^2 - 8 * x - 20 ≤ 0
def S (m x : ℝ) : Prop := |x - 1| ≤ m

theorem no_m_for_P_eq_S : ¬ ∃ m : ℝ, ∀ x : ℝ, P x ↔ S m x := sorry

theorem m_le_3_for_P_implies_S : ∀ (m : ℝ), (m ≤ 3) → (∀ x, S m x → P x) := sorry

end ProofProblem

end no_m_for_P_eq_S_m_le_3_for_P_implies_S_l187_187846


namespace total_profit_from_selling_30_necklaces_l187_187925

-- Definitions based on conditions
def charms_per_necklace : Nat := 10
def cost_per_charm : Nat := 15
def selling_price_per_necklace : Nat := 200
def number_of_necklaces_sold : Nat := 30

-- Lean statement to prove the total profit
theorem total_profit_from_selling_30_necklaces :
  (selling_price_per_necklace - (charms_per_necklace * cost_per_charm)) * number_of_necklaces_sold = 1500 :=
by
  sorry

end total_profit_from_selling_30_necklaces_l187_187925


namespace consecutive_int_sqrt_l187_187718

theorem consecutive_int_sqrt (m n : ℤ) (h1 : m < n) (h2 : m < Real.sqrt 13) (h3 : Real.sqrt 13 < n) (h4 : n = m + 1) : m * n = 12 :=
sorry

end consecutive_int_sqrt_l187_187718


namespace max_value_of_fraction_l187_187764

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l187_187764


namespace proof_quotient_l187_187034

/-- Let x be in the form (a + b * sqrt c) / d -/
def x_form (a b c d : ℤ) (x : ℝ) : Prop := x = (a + b * Real.sqrt c) / d

/-- Main theorem -/
theorem proof_quotient (a b c d : ℤ) (x : ℝ) (h_eq : 4 * x / 5 + 2 = 5 / x) (h_form : x_form a b c d x) : (a * c * d) / b = -20 := by
  sorry

end proof_quotient_l187_187034


namespace total_messages_equation_l187_187066

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l187_187066


namespace money_distribution_l187_187004

theorem money_distribution (A B C : ℝ) 
  (h1 : A + B + C = 500) 
  (h2 : A + C = 200) 
  (h3 : B + C = 340) : 
  C = 40 := 
sorry

end money_distribution_l187_187004


namespace integer_solutions_l187_187668

theorem integer_solutions (t : ℤ) : 
  ∃ x y : ℤ, 5 * x - 7 * y = 3 ∧ x = 7 * t - 12 ∧ y = 5 * t - 9 :=
by
  sorry

end integer_solutions_l187_187668


namespace min_area_triangle_l187_187402

theorem min_area_triangle (m n : ℝ) (h : m^2 + n^2 = 1/3) : ∃ S, S = 3 :=
by
  sorry

end min_area_triangle_l187_187402


namespace oil_bill_january_l187_187132

theorem oil_bill_january (F J : ℝ)
  (h1 : F / J = 5 / 4)
  (h2 : (F + 30) / J = 3 / 2) :
  J = 120 :=
sorry

end oil_bill_january_l187_187132


namespace homework_problems_l187_187237

noncomputable def problems_solved (p t : ℕ) : ℕ := p * t

theorem homework_problems (p t : ℕ) (h_eq: p * t = (3 * p - 5) * (t - 3))
  (h_pos_p: p > 0) (h_pos_t: t > 0) (h_p_ge_15: p ≥ 15) 
  (h_friend_did_20: (3 * p - 5) * (t - 3) ≥ 20) : 
  problems_solved p t = 100 :=
by
  sorry

end homework_problems_l187_187237


namespace maximum_achievable_score_l187_187116

def robot_initial_iq : Nat := 25
def problem_scores : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

theorem maximum_achievable_score 
  (initial_iq : Nat := robot_initial_iq) 
  (scores : List Nat := problem_scores) 
  : Nat :=
  31

end maximum_achievable_score_l187_187116


namespace solve_system_l187_187403

theorem solve_system (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : 2 * x + y = 7) : x + y = 5 :=
by
  sorry

end solve_system_l187_187403


namespace percentage_spent_on_food_l187_187093

-- Definitions based on conditions
variables {T : ℝ} -- Total amount spent
def spent_on_clothing := 0.50 * T
def spent_on_food (x : ℝ) := (x / 100) * T
def spent_on_other_items := 0.30 * T

def tax_on_clothing := 0.05 * spent_on_clothing
def tax_on_food (x : ℝ) := 0 * spent_on_food x
def tax_on_other_items := 0.10 * spent_on_other_items
def total_tax := 0.055 * T

-- Theorem stating that the percentage spent on food is 20%
theorem percentage_spent_on_food : 
  (total_tax = tax_on_clothing + tax_on_food 20 + tax_on_other_items) →
  50 + 30 + 20 = 100 :=
begin
  intros h,
  sorry 
end

end percentage_spent_on_food_l187_187093


namespace exponent_equality_l187_187870

theorem exponent_equality (p : ℕ) (h : 81^10 = 3^p) : p = 40 :=
sorry

end exponent_equality_l187_187870


namespace prime_factorization_sum_l187_187441

theorem prime_factorization_sum (x y : ℕ) (hx : x > 0) (hy : y > 0) (hxy : 13 * x^7 = 17 * y^11) : 
  a * e + b * f = 18 :=
by
  -- Let a and b be prime factors of x
  let a : ℕ := 17 -- prime factor found in the solution
  let e : ℕ := 1 -- exponent found for 17
  let b : ℕ := 0 -- no second prime factor
  let f : ℕ := 0 -- corresponding exponent

  sorry

end prime_factorization_sum_l187_187441


namespace range_of_a_l187_187710

-- Define sets P and M
def P : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def M (a : ℝ) : Set ℝ := {x | (2 - a) ≤ x ∧ x ≤ (1 + a)}

-- Prove the range of a
theorem range_of_a (a : ℝ) : (P ∩ (M a) = P) ↔ (a ≥ 1) :=
by 
  sorry

end range_of_a_l187_187710


namespace real_solution_of_equation_l187_187512

theorem real_solution_of_equation :
  ∀ x : ℝ, (x ≠ 5) → (x ≠ 3) →
  ((x - 2) * (x - 5) * (x - 3) * (x - 2) * (x - 4) * (x - 5) * (x - 3)) 
  / ((x - 5) * (x - 3) * (x - 5)) = 1 ↔ x = 1 :=
by sorry

end real_solution_of_equation_l187_187512


namespace remainder_when_divided_by_5_l187_187039

theorem remainder_when_divided_by_5 (n : ℕ) (h1 : n^2 % 5 = 1) (h2 : n^3 % 5 = 4) : n % 5 = 4 :=
sorry

end remainder_when_divided_by_5_l187_187039


namespace problem_l187_187421

theorem problem (p q : Prop) (h1 : ¬ (p ∧ q)) (h2 : ¬ ¬ q) : ¬ p ∧ q :=
by
  -- proof goes here
  sorry

end problem_l187_187421


namespace find_angle_A_find_side_a_l187_187721

-- Define the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
variables {a b c A B C : ℝ}
-- Assumption conditions in the problem
variables (h₁ : a * sin B = sqrt 3 * b * cos A)
variables (hb : b = 3)
variables (hc : c = 2)

-- Prove that A = π / 3 given the first condition
theorem find_angle_A : h₁ → A = π / 3 := by
  -- Proof is omitted
  sorry

-- Prove that a = sqrt 7 given b = 3, c = 2, and A = π / 3
theorem find_side_a : h₁ → hb → hc → a = sqrt 7 := by
  -- Proof is omitted
  sorry

end find_angle_A_find_side_a_l187_187721


namespace find_x_l187_187656

theorem find_x (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
sorry

end find_x_l187_187656


namespace distance_focus_directrix_l187_187076

theorem distance_focus_directrix (p : ℝ) (x_1 : ℝ) (h1 : 0 < p) (h2 : x_1^2 = 2 * p)
  (h3 : 1 + p / 2 = 3) : p = 4 :=
by
  sorry

end distance_focus_directrix_l187_187076


namespace find_rate_percent_l187_187326

-- Definitions based on the given conditions
def principal : ℕ := 800
def time : ℕ := 4
def simple_interest : ℕ := 192
def si_formula (P R T : ℕ) : ℕ := P * R * T / 100

-- Statement: prove that the rate percent (R) is 6%
theorem find_rate_percent (R : ℕ) (h : simple_interest = si_formula principal R time) : R = 6 :=
sorry

end find_rate_percent_l187_187326


namespace sqrt_200_eq_l187_187283

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l187_187283


namespace minimum_value_a_l187_187194

theorem minimum_value_a (a : ℝ) (h1 : 1 < a) :
  (∀ x ∈ set.Ici (1/3 : ℝ), (1 / (3 * x) - x + Real.log (3 * x) ≤ 1 / (a * Real.exp x) + Real.log a)) →
  a ≥ 3 / Real.exp 1 :=
by
  sorry

end minimum_value_a_l187_187194


namespace polynomial_remainder_l187_187050

theorem polynomial_remainder (c a b : ℤ) 
  (h1 : (16 * c + 8 * a + 2 * b = -12)) 
  (h2 : (81 * c - 27 * a - 3 * b = -85)) : 
  (a, b, c) = (5, 7, 1) :=
sorry

end polynomial_remainder_l187_187050


namespace lowest_sale_price_percentage_l187_187330

theorem lowest_sale_price_percentage :
  ∃ (p : ℝ) (h1 : 30 / 100 * p ≤ 70 / 100 * p) (h2 : p = 80),
  (p - 70 / 100 * p - 20 / 100 * p) / p * 100 = 10 := by
sorry

end lowest_sale_price_percentage_l187_187330


namespace age_difference_l187_187830

theorem age_difference :
  let x := 5
  let prod_today := x * x
  let prod_future := (x + 1) * (x + 1)
  prod_future - prod_today = 11 :=
by
  sorry

end age_difference_l187_187830


namespace smallest_successive_number_l187_187294

theorem smallest_successive_number :
  ∃ n : ℕ, n * (n + 1) * (n + 2) = 1059460 ∧ ∀ m : ℕ, m * (m + 1) * (m + 2) = 1059460 → n ≤ m :=
sorry

end smallest_successive_number_l187_187294


namespace rope_fold_length_l187_187150

theorem rope_fold_length (L : ℝ) (hL : L = 1) :
  (L / 2 / 2 / 2) = (1 / 8) :=
by
  -- proof steps here
  sorry

end rope_fold_length_l187_187150


namespace find_n_l187_187049

theorem find_n : ∃ n : ℕ, 
  (S : ℕ) (i : ℕ) (hS₀ : S = 0) (hi₀ : i = 1)
  (hloop : ∀ (S i : ℕ), S ≤ 200 → (S+ i, i + 2)) 
  (hfinal : i - 2 = n), n = 27 :=
by
  let S := 0
  let i := 1
  let S := Nat.iterate 14 (λ (p : ℕ × ℕ), (p.1 + p.2, p.2 + 2)) (S, i)
  let S := S.fst
  let i := S.snd
  let n := i - 2
  exact ⟨n, by simp [S, i, n]⟩

end find_n_l187_187049


namespace sequence_a_correct_l187_187856

open Nat -- Opening the natural numbers namespace

noncomputable def a : ℕ → ℝ
| 0       => 1
| (n + 1) => (1 / 2 : ℝ) * a n

theorem sequence_a_correct : 
  (∀ n, 0 < a n) ∧ 
  a 1 = 1 ∧ 
  (∀ n, a (n + 1) = a n / 2) ∧
  a 2 = 1 / 2 ∧
  a 3 = 1 / 4 ∧
  ∀ n, a n = 1 / 2^(n - 1) :=
by
  sorry

end sequence_a_correct_l187_187856


namespace Maria_height_in_meters_l187_187903

theorem Maria_height_in_meters :
  let inch_to_cm := 2.54
  let cm_to_m := 0.01
  let height_in_inch := 54
  let height_in_cm := height_in_inch * inch_to_cm
  let height_in_m := height_in_cm * cm_to_m
  let rounded_height_in_m := Float.round (height_in_m * 1000) / 1000
  rounded_height_in_m = 1.372 := 
by
  sorry

end Maria_height_in_meters_l187_187903


namespace selling_price_of_book_l187_187719

theorem selling_price_of_book (SP : ℝ) (CP : ℝ := 200) :
  (SP - CP) = (340 - CP) + 0.05 * CP → SP = 350 :=
by {
  sorry
}

end selling_price_of_book_l187_187719


namespace find_a_l187_187854

theorem find_a (a : ℝ) (ha : a ≠ 0)
  (h_area : (1/2) * (a/2) * a^2 = 2) :
  a = 2 ∨ a = -2 :=
sorry

end find_a_l187_187854


namespace contractor_earnings_l187_187947

def total_days : ℕ := 30
def work_rate : ℝ := 25
def fine_rate : ℝ := 7.5
def absent_days : ℕ := 8
def worked_days : ℕ := total_days - absent_days
def total_earned : ℝ := worked_days * work_rate
def total_fine : ℝ := absent_days * fine_rate
def total_received : ℝ := total_earned - total_fine

theorem contractor_earnings : total_received = 490 :=
by
  sorry

end contractor_earnings_l187_187947


namespace total_turtles_l187_187966

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l187_187966


namespace parallel_line_perpendicular_line_l187_187176

theorem parallel_line (x y : ℝ) (h : y = 2 * x + 3) : ∃ a : ℝ, 3 * x - 2 * y + a = 0 :=
by
  use 1
  sorry

theorem perpendicular_line  (x y : ℝ) (h : y = -x / 2) : ∃ c : ℝ, 3 * x - 2 * y + c = 0 :=
by
  use -5
  sorry

end parallel_line_perpendicular_line_l187_187176


namespace general_term_of_sequence_l187_187531

theorem general_term_of_sequence (a : ℕ → ℝ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 1)
  (h_rec : ∀ n ≥ 2, a (n + 1) = (n^2 * (a n)^2 + 5) / ((n^2 - 1) * a (n - 1))) :
  ∀ n : ℕ, a n = 
    if n = 0 then 0 else
    (1 / n) * ( (63 - 13 * Real.sqrt 21) / 42 * ((5 + Real.sqrt 21) / 2) ^ n + 
                (63 + 13 * Real.sqrt 21) / 42 * ((5 - Real.sqrt 21) / 2) ^ n) :=
by
  sorry

end general_term_of_sequence_l187_187531


namespace num_digits_sum_l187_187716

theorem num_digits_sum (A B : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 1 ≤ B ∧ B ≤ 9) :
  let num1 := 9643
  let num2 := A * 10 ^ 2 + 7 * 10 + 5
  let num3 := 5 * 10 ^ 2 + B * 10 + 2
  let sum := num1 + num2 + num3
  10^4 ≤ sum ∧ sum < 10^5 :=
by {
  sorry
}

end num_digits_sum_l187_187716


namespace smallest_b_l187_187440

theorem smallest_b {a b c d : ℕ} (r : ℕ) 
  (h1 : a = b - r) (h2 : c = b + r) (h3 : d = b + 2 * r) 
  (h4 : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h5 : a * b * c * d = 256) : b = 4 :=
by
  sorry

end smallest_b_l187_187440


namespace combined_avg_score_l187_187129

-- Define the average scores
def avg_score_u : ℕ := 65
def avg_score_b : ℕ := 80
def avg_score_c : ℕ := 77

-- Define the ratio of the number of students
def ratio_u : ℕ := 4
def ratio_b : ℕ := 6
def ratio_c : ℕ := 5

-- Prove the combined average score
theorem combined_avg_score : (ratio_u * avg_score_u + ratio_b * avg_score_b + ratio_c * avg_score_c) / (ratio_u + ratio_b + ratio_c) = 75 :=
by
  sorry

end combined_avg_score_l187_187129


namespace power_equivalence_l187_187867

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l187_187867


namespace order_of_a_add_b_sub_b_l187_187205

variable (a b : ℚ)

theorem order_of_a_add_b_sub_b (hb : b < 0) : a + b < a ∧ a < a - b := by
  sorry

end order_of_a_add_b_sub_b_l187_187205


namespace find_chord_points_l187_187982

/-
Define a parabola and check if the points given form a chord that intersects 
the point (8,4) in the ratio 1:4.
-/

def parabola (P : ℝ × ℝ) : Prop :=
  P.snd^2 = 4 * P.fst

def divides_in_ratio (C A B : ℝ × ℝ) (m n : ℝ) : Prop :=
  (A.fst * n + B.fst * m = C.fst * (m + n)) ∧ 
  (A.snd * n + B.snd * m = C.snd * (m + n))

theorem find_chord_points :
  ∃ (P1 P2 : ℝ × ℝ),
  parabola P1 ∧
  parabola P2 ∧
  divides_in_ratio (8, 4) P1 P2 1 4 ∧ 
  ((P1 = (1, 2) ∧ P2 = (36, 12)) ∨ (P1 = (9, 6) ∧ P2 = (4, -4))) :=
sorry

end find_chord_points_l187_187982


namespace part1_l187_187570

def U : Set ℝ := Set.univ
def A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

theorem part1 (a : ℝ) (h : a = 1) :
  (Set.compl B ∪ A a) = {x | x ≤ 1 ∨ x ≥ 2} :=
by
  sorry

end part1_l187_187570


namespace maria_cookies_left_l187_187230

-- Define the initial conditions and necessary variables
def initial_cookies : ℕ := 19
def given_cookies_to_friend : ℕ := 5
def eaten_cookies : ℕ := 2

-- Define remaining cookies after each step
def remaining_after_friend (total : ℕ) := total - given_cookies_to_friend
def remaining_after_family (remaining : ℕ) := remaining / 2
def remaining_after_eating (after_family : ℕ) := after_family - eaten_cookies

-- Main theorem to prove
theorem maria_cookies_left :
  let initial := initial_cookies,
      after_friend := remaining_after_friend initial,
      after_family := remaining_after_family after_friend,
      final := remaining_after_eating after_family
  in final = 5 :=
by
  sorry

end maria_cookies_left_l187_187230


namespace problem_l187_187984

theorem problem (x y : ℝ) : 
  2 * x + y = 11 → x + 2 * y = 13 → 10 * x^2 - 6 * x * y + y^2 = 530 :=
by
  sorry

end problem_l187_187984


namespace incorrect_statement_A_l187_187861

theorem incorrect_statement_A (p q : Prop) : (p ∨ q) → ¬ (p ∧ q) := by
  intros h
  cases h with
  | inl hp => sorry
  | inr hq => sorry

end incorrect_statement_A_l187_187861


namespace find_AE_l187_187494

-- Define the given conditions as hypotheses
variables (AB CD AC AE EC : ℝ)
variables (E : Type _)
variables (triangle_AED triangle_BEC : E)

-- Assume the given conditions
axiom AB_eq_9 : AB = 9
axiom CD_eq_12 : CD = 12
axiom AC_eq_14 : AC = 14
axiom areas_equal : ∀ h : ℝ, 1/2 * AE * h = 1/2 * EC * h

-- Declare the theorem statement to prove AE
theorem find_AE (h : ℝ) (h' : EC = AC - AE) (h'' : 4 * AE = 3 * EC) : AE = 6 :=
by {
  -- proof steps as intermediate steps
  sorry
}

end find_AE_l187_187494


namespace cafeteria_pies_l187_187814

theorem cafeteria_pies (total_apples handed_out_per_student apples_per_pie : ℕ) (h1 : total_apples = 47) (h2 : handed_out_per_student = 27) (h3 : apples_per_pie = 4) :
  (total_apples - handed_out_per_student) / apples_per_pie = 5 := by
  sorry

end cafeteria_pies_l187_187814


namespace number_of_divisible_factorials_l187_187991

theorem number_of_divisible_factorials:
  ∃ (count : ℕ), count = 36 ∧ ∀ n, 1 ≤ n ∧ n ≤ 50 → (∃ k : ℕ, n! = k * (n * (n + 1)) / 2) ↔ n ≤ n - 14 :=
sorry

end number_of_divisible_factorials_l187_187991


namespace smallest_nat_number_l187_187298

theorem smallest_nat_number (x : ℕ) (h1 : 5 ∣ x) (h2 : 7 ∣ x) (h3 : x % 3 = 1) : x = 70 :=
sorry

end smallest_nat_number_l187_187298


namespace num_unique_pizzas_l187_187145

-- Define the problem conditions
def total_toppings : ℕ := 8
def chosen_toppings : ℕ := 5

-- Define the target number of combinations
def max_unique_pizzas : ℕ := nat.choose total_toppings chosen_toppings

-- State the theorem
theorem num_unique_pizzas : max_unique_pizzas = 56 :=
by
  -- The actual proof will go here
  sorry

end num_unique_pizzas_l187_187145


namespace cube_square_third_smallest_prime_l187_187319

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m : ℕ, 2 ≤ m → m < n → n % m ≠ 0

def third_smallest_prime := 5

noncomputable def cube (n : ℕ) : ℕ := n * n * n

noncomputable def square (n : ℕ) : ℕ := n * n

theorem cube_square_third_smallest_prime : cube (square third_smallest_prime) = 15625 := by
  have h1 : is_prime 2 := by sorry
  have h2 : is_prime 3 := by sorry
  have h3 : is_prime 5 := by sorry
  sorry

end cube_square_third_smallest_prime_l187_187319


namespace roots_of_equation_l187_187164

theorem roots_of_equation (a x : ℝ) : x * (x + 5)^2 * (a - x) = 0 ↔ (x = 0 ∨ x = -5 ∨ x = a) :=
by
  sorry

end roots_of_equation_l187_187164


namespace find_age_l187_187662

theorem find_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 := 
by 
  sorry

end find_age_l187_187662


namespace solve_absolute_value_eq_l187_187650

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l187_187650


namespace largest_number_l187_187328

theorem largest_number (A B C D E : ℝ) (hA : A = 0.998) (hB : B = 0.9899) (hC : C = 0.9) (hD : D = 0.9989) (hE : E = 0.8999) :
  D > A ∧ D > B ∧ D > C ∧ D > E :=
by
  sorry

end largest_number_l187_187328


namespace distinct_nonzero_reals_product_l187_187527

theorem distinct_nonzero_reals_product 
  (x y : ℝ) 
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (hxy: x ≠ y)
  (h : x + 3 / x = y + 3 / y) :
  x * y = 3 :=
sorry

end distinct_nonzero_reals_product_l187_187527


namespace initial_ratio_is_2_63_l187_187482

noncomputable theory

def initial_ratio_of_firm_partners_associates : Prop :=
  ∃ (a : ℕ), ∃ (g : ℕ), g > 0 ∧ 14 * 34 = a + 35 ∧ g.gcd 14 = 1 ∧ g.gcd a = 1 ∧ (14 / g : ℤ) = 2 ∧ (a / g : ℤ) = 63

theorem initial_ratio_is_2_63 : initial_ratio_of_firm_partners_associates :=
sorry

end initial_ratio_is_2_63_l187_187482


namespace solve_for_x_l187_187098

theorem solve_for_x (x : ℝ) : 5 + 3.4 * x = 2.1 * x - 30 → x = -26.923 := 
by 
  sorry

end solve_for_x_l187_187098


namespace solution_l187_187608

theorem solution (y : ℝ) (h : 6 * y^2 + 7 = 2 * y + 12) : (12 * y - 4)^2 = 128 :=
sorry

end solution_l187_187608


namespace jori_water_left_l187_187083

theorem jori_water_left (initial_water : ℚ) (used_water : ℚ) : initial_water = 3 ∧ used_water = 5/4 → initial_water - used_water = 7/4 :=
by {
  intro h,
  cases h with h1 h2,
  rw [h1, h2],
  norm_num,
}

end jori_water_left_l187_187083


namespace find_xyz_l187_187197

theorem find_xyz (x y z : ℝ)
  (h₁ : (x + y + z) * (x * y + x * z + y * z) = 27)
  (h₂ : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11)
  (h₃ : x + y + z = 3) :
  x * y * z = 16 / 3 := 
  sorry

end find_xyz_l187_187197


namespace sqrt_200_eq_10_l187_187247

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l187_187247


namespace Q_investment_l187_187473

-- Given conditions
variables (P Q : Nat) (P_investment : P = 30000) (profit_ratio : 2 / 3 = P / Q)

-- Target statement
theorem Q_investment : Q = 45000 :=
by 
  sorry

end Q_investment_l187_187473


namespace max_expression_value_l187_187770

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l187_187770


namespace attendees_received_all_items_l187_187688

theorem attendees_received_all_items {n : ℕ} (h1 : ∀ k, k ∣ 45 → n % k = 0) (h2 : ∀ k, k ∣ 75 → n % k = 0) (h3 : ∀ k, k ∣ 100 → n % k = 0) (h4 : n = 4500) :
  (4500 / Nat.lcm (Nat.lcm 45 75) 100) = 5 :=
by
  sorry

end attendees_received_all_items_l187_187688


namespace weight_of_new_person_l187_187665

theorem weight_of_new_person (W : ℝ) (N : ℝ) (h1 : (W + (8 * 2.5)) = (W - 20 + N)) : N = 40 :=
by
  sorry

end weight_of_new_person_l187_187665


namespace tan_alpha_plus_pi_l187_187560

-- Define the given conditions and prove the desired equality.
theorem tan_alpha_plus_pi 
  (α : ℝ) 
  (hα : 0 < α ∧ α < π) 
  (hcos : Real.cos (π - α) = 1 / 3) : 
  Real.tan (α + π) = -2 * Real.sqrt 2 :=
by
  sorry

end tan_alpha_plus_pi_l187_187560


namespace not_possible_to_form_triangle_l187_187824

-- Define the conditions
variables (a : ℝ)

-- State the problem in Lean 4
theorem not_possible_to_form_triangle (h : a > 0) :
  ¬ (a + a > 2 * a ∧ a + 2 * a > a ∧ a + 2 * a > a) :=
by
  sorry

end not_possible_to_form_triangle_l187_187824


namespace office_distance_eq_10_l187_187128

noncomputable def distance_to_office (D T : ℝ) : Prop :=
  D = 10 * (T + 10 / 60) ∧ D = 15 * (T - 10 / 60)

theorem office_distance_eq_10 (D T : ℝ) (h : distance_to_office D T) : D = 10 :=
by
  sorry

end office_distance_eq_10_l187_187128


namespace min_ineq_l187_187394

theorem min_ineq (x : ℝ) (hx : x > 0) : 3*x + 1/x^2 ≥ 4 :=
sorry

end min_ineq_l187_187394


namespace remainder_of_3_pow_45_mod_17_l187_187806

theorem remainder_of_3_pow_45_mod_17 : 3^45 % 17 = 15 := 
by {
  sorry
}

end remainder_of_3_pow_45_mod_17_l187_187806


namespace max_expression_value_l187_187779

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l187_187779


namespace inequality_solution_l187_187586

theorem inequality_solution (x : ℝ) (h : x ≠ -5) : 
  (x^2 - 25) / (x + 5) < 0 ↔ x ∈ Set.union (Set.Iio (-5)) (Set.Ioo (-5) 5) := 
by
  sorry

end inequality_solution_l187_187586


namespace area_of_fourth_rectangle_l187_187676

-- The conditions provided in the problem
variables (x y z w : ℝ)
variables (h1 : x * y = 24) (h2 : x * w = 12) (h3 : z * w = 8)

-- The problem statement with the conclusion
theorem area_of_fourth_rectangle :
  (∃ (x y z w : ℝ), ((x * y = 24 ∧ x * w = 12 ∧ z * w = 8) ∧ y * z = 16)) :=
sorry

end area_of_fourth_rectangle_l187_187676


namespace cube_edge_factor_l187_187107

theorem cube_edge_factor (e f : ℝ) (h₁ : e > 0) (h₂ : (f * e) ^ 3 = 8 * e ^ 3) : f = 2 :=
by
  sorry

end cube_edge_factor_l187_187107


namespace contrapositive_of_equality_square_l187_187449

theorem contrapositive_of_equality_square (a b : ℝ) (h : a^2 ≠ b^2) : a ≠ b := 
by 
  sorry

end contrapositive_of_equality_square_l187_187449


namespace sqrt_200_eq_10_l187_187274

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l187_187274


namespace total_turtles_l187_187968

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l187_187968


namespace inequality_solution_set_l187_187587

theorem inequality_solution_set (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) ↔ (-1 ≤ x ∧ x < 3) := 
by sorry

end inequality_solution_set_l187_187587


namespace problem_statement_l187_187899

variable (a b c : ℝ)

theorem problem_statement (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a^2 + b^2 + c^2 = 1 / 2) :
  (1 - a^2 + c^2) / (c * (a + 2 * b)) + 
  (1 - b^2 + a^2) / (a * (b + 2 * c)) + 
  (1 - c^2 + b^2) / (b * (c + 2 * a)) ≥ 6 :=
sorry

end problem_statement_l187_187899


namespace functional_equation_to_linear_l187_187909

-- Define that f satisfies the Cauchy functional equation
variable (f : ℕ → ℝ)
axiom cauchy_eq (x y : ℕ) : f (x + y) = f x + f y

-- The theorem we want to prove
theorem functional_equation_to_linear (h : ∀ n k : ℕ, f (n * k) = n * f k) : ∃ a : ℝ, ∀ n : ℕ, f n = a * n :=
by
  sorry

end functional_equation_to_linear_l187_187909


namespace number_of_girls_l187_187627

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l187_187627


namespace find_a_l187_187209

theorem find_a (a : ℝ)
  (hl : ∀ x y : ℝ, ax + 2 * y - a - 2 = 0)
  (hm : ∀ x y : ℝ, 2 * x - y = 0)
  (perpendicular : ∀ x y : ℝ, (2 * - (a / 2)) = -1) : 
  a = 1 := sorry

end find_a_l187_187209


namespace fabric_woven_in_30_days_l187_187103

theorem fabric_woven_in_30_days :
  let a1 := 5
  let d := 16 / 29
  (30 * a1 + (30 * (30 - 1) / 2) * d) = 390 :=
by
  let a1 := 5
  let d := 16 / 29
  sorry

end fabric_woven_in_30_days_l187_187103


namespace sodium_thiosulfate_properties_l187_187097

def thiosulfate_structure : Type := sorry
-- Define the structure of S2O3^{2-} with S-S bond
def has_s_s_bond (ion : thiosulfate_structure) : Prop := sorry
-- Define the formation reaction
def formed_by_sulfite_reaction (ion : thiosulfate_structure) : Prop := sorry

theorem sodium_thiosulfate_properties :
  ∃ (ion : thiosulfate_structure),
    has_s_s_bond ion ∧ formed_by_sulfite_reaction ion :=
by
  sorry

end sodium_thiosulfate_properties_l187_187097


namespace emily_gave_away_l187_187024

variable (x : ℕ)

def emily_initial_books : ℕ := 7

def emily_books_after_giving_away (x : ℕ) : ℕ := 7 - x

def emily_books_after_buying_more (x : ℕ) : ℕ :=
  7 - x + 14

def emily_final_books : ℕ := 19

theorem emily_gave_away : (emily_books_after_buying_more x = emily_final_books) → x = 2 := by
  sorry

end emily_gave_away_l187_187024


namespace fraction_available_on_third_day_l187_187674

noncomputable def liters_used_on_first_day (initial_amount : ℕ) : ℕ :=
  (initial_amount / 2)

noncomputable def liters_added_on_second_day : ℕ :=
  1

noncomputable def original_solution : ℕ :=
  4

noncomputable def remaining_solution_after_first_day : ℕ :=
  original_solution - liters_used_on_first_day original_solution

noncomputable def remaining_solution_after_second_day : ℕ :=
  remaining_solution_after_first_day + liters_added_on_second_day

noncomputable def fraction_of_original_solution : ℚ :=
  remaining_solution_after_second_day / original_solution

theorem fraction_available_on_third_day : fraction_of_original_solution = 3 / 4 :=
by
  sorry

end fraction_available_on_third_day_l187_187674


namespace problem_statement_l187_187735

def p (x : ℝ) : ℝ := x^2 - x + 1

theorem problem_statement (α : ℝ) (h : p (p (p (p α))) = 0) :
  (p α - 1) * p α * p (p α) * p (p (p α)) = -1 :=
by
  sorry

end problem_statement_l187_187735


namespace binomial_coeff_sum_l187_187460

-- Define the problem: compute the numerical sum of the binomial coefficients
theorem binomial_coeff_sum (a b : ℕ) (h_a1 : a = 1) (h_b1 : b = 1) : 
  (a + b) ^ 8 = 256 :=
by
  -- Therefore, the sum must be 256
  sorry

end binomial_coeff_sum_l187_187460


namespace first_discount_correct_l187_187459

noncomputable def first_discount (x : ℝ) : Prop :=
  let initial_price := 600
  let first_discounted_price := initial_price * (1 - x / 100)
  let final_price := first_discounted_price * (1 - 0.05)
  final_price = 456

theorem first_discount_correct : ∃ x : ℝ, first_discount x ∧ abs (x - 57.29) < 0.01 :=
by
  sorry

end first_discount_correct_l187_187459


namespace inequality_xyz_l187_187341

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l187_187341


namespace cos_15_degree_l187_187978

theorem cos_15_degree : 
  let d15 := 15 * Real.pi / 180
  let d45 := 45 * Real.pi / 180
  let d30 := 30 * Real.pi / 180
  Real.cos d15 = (Real.sqrt 6 + Real.sqrt 2) / 4 := by
  sorry

end cos_15_degree_l187_187978


namespace angle_measures_possible_l187_187607

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l187_187607


namespace jolyn_older_than_leon_l187_187518

open Nat

def Jolyn := Nat
def Therese := Nat
def Aivo := Nat
def Leon := Nat

-- Conditions
variable (jolyn therese aivo leon : Nat)
variable (h1 : jolyn = therese + 2) -- Jolyn is 2 months older than Therese
variable (h2 : therese = aivo + 5) -- Therese is 5 months older than Aivo
variable (h3 : leon = aivo + 2) -- Leon is 2 months older than Aivo

theorem jolyn_older_than_leon :
  jolyn = leon + 5 := by
  sorry

end jolyn_older_than_leon_l187_187518


namespace sqrt_200_eq_10_l187_187272

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l187_187272


namespace B_finishes_in_10_days_l187_187353

noncomputable def B_remaining_work_days (A_work_days : ℕ := 15) (A_initial_days_worked : ℕ := 5) (B_work_days : ℝ := 14.999999999999996) : ℝ :=
  let A_rate := 1 / A_work_days
  let B_rate := 1 / B_work_days
  let remaining_work := 1 - (A_rate * A_initial_days_worked)
  let days_for_B := remaining_work / B_rate
  days_for_B

theorem B_finishes_in_10_days :
  B_remaining_work_days 15 5 14.999999999999996 = 10 :=
by
  sorry

end B_finishes_in_10_days_l187_187353


namespace additional_discount_percentage_l187_187508

-- Define constants representing the conditions
def price_shoes : ℝ := 200
def discount_shoes : ℝ := 0.30
def price_shirt : ℝ := 80
def number_shirts : ℕ := 2
def final_spent : ℝ := 285

-- Define the theorem to prove the additional discount percentage
theorem additional_discount_percentage :
  let discounted_shoes := price_shoes * (1 - discount_shoes)
  let total_before_additional_discount := discounted_shoes + number_shirts * price_shirt
  let additional_discount := total_before_additional_discount - final_spent
  (additional_discount / total_before_additional_discount) * 100 = 5 :=
by
  -- Lean proof goes here, but we'll skip it for now with sorry
  sorry

end additional_discount_percentage_l187_187508


namespace four_digit_integers_with_repeated_digits_l187_187534

noncomputable def count_four_digit_integers_with_repeated_digits : ℕ := sorry

theorem four_digit_integers_with_repeated_digits : 
  count_four_digit_integers_with_repeated_digits = 1984 :=
sorry

end four_digit_integers_with_repeated_digits_l187_187534


namespace power_simplification_l187_187167

noncomputable def sqrt2_six : ℝ := 6 ^ (1 / 2)
noncomputable def sqrt3_six : ℝ := 6 ^ (1 / 3)

theorem power_simplification :
  (sqrt2_six / sqrt3_six) = 6 ^ (1 / 6) :=
  sorry

end power_simplification_l187_187167


namespace sqrt_200_eq_10_sqrt_2_l187_187255

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
by
  sorry

end sqrt_200_eq_10_sqrt_2_l187_187255


namespace x_cubed_plus_y_cubed_l187_187540

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end x_cubed_plus_y_cubed_l187_187540


namespace max_square_side_length_l187_187189

theorem max_square_side_length (AC BC : ℝ) (hAC : AC = 3) (hBC : BC = 7) : 
  ∃ s : ℝ, s = 2.1 := by
  sorry

end max_square_side_length_l187_187189


namespace olivia_earnings_l187_187241

-- Define Olivia's hourly wage
def wage : ℕ := 9

-- Define the hours worked on each day
def hours_monday : ℕ := 4
def hours_wednesday : ℕ := 3
def hours_friday : ℕ := 6

-- Define the total hours worked
def total_hours : ℕ := hours_monday + hours_wednesday + hours_friday

-- Define the total earnings
def total_earnings : ℕ := total_hours * wage

-- State the theorem
theorem olivia_earnings : total_earnings = 117 :=
by
  sorry

end olivia_earnings_l187_187241


namespace find_smaller_number_l187_187670

theorem find_smaller_number (x y : ℕ) (h1 : x = 2 * y - 3) (h2 : x + y = 51) : y = 18 :=
sorry

end find_smaller_number_l187_187670


namespace region_area_l187_187123

theorem region_area {x y : ℝ} (h : x^2 + y^2 - 4*x + 2*y = -1) : 
  ∃ (r : ℝ), r = 4*pi := 
sorry

end region_area_l187_187123


namespace identify_counterfeit_bag_l187_187615

-- Definitions based on problem conditions
def num_bags := 10
def genuine_weight := 10
def counterfeit_weight := 11
def expected_total_weight := genuine_weight * ((num_bags * (num_bags + 1)) / 2 : ℕ)

-- Lean theorem for the above problem
theorem identify_counterfeit_bag (W : ℕ) (Δ := W - expected_total_weight) :
  ∃ i : ℕ, 1 ≤ i ∧ i ≤ num_bags ∧ Δ = i :=
by sorry

end identify_counterfeit_bag_l187_187615


namespace at_least_3_defective_correct_l187_187070

/-- Number of products in batch -/
def total_products : ℕ := 50

/-- Number of defective products -/
def defective_products : ℕ := 4

/-- Number of products drawn -/
def drawn_products : ℕ := 5

/-- Number of ways to draw at least 3 defective products out of 5 -/
def num_ways_at_least_3_defective : ℕ :=
  (Nat.choose defective_products 4) * (Nat.choose (total_products - defective_products) 1) +
  (Nat.choose defective_products 3) * (Nat.choose (total_products - defective_products) 2)

theorem at_least_3_defective_correct : num_ways_at_least_3_defective = 4186 := by
  sorry

end at_least_3_defective_correct_l187_187070


namespace dorothy_total_sea_glass_l187_187689

def Blanche_red : ℕ := 3
def Rose_red : ℕ := 9
def Rose_blue : ℕ := 11

def Dorothy_red : ℕ := 2 * (Blanche_red + Rose_red)
def Dorothy_blue : ℕ := 3 * Rose_blue

theorem dorothy_total_sea_glass : Dorothy_red + Dorothy_blue = 57 :=
by
  sorry

end dorothy_total_sea_glass_l187_187689


namespace find_n_solution_l187_187036

theorem find_n_solution (n : ℚ) (h : (1 / (n + 2) + 3 / (n + 2) + n / (n + 2) = 4)) : n = -4 / 3 :=
by
  sorry

end find_n_solution_l187_187036


namespace three_times_x_not_much_different_from_two_l187_187389

theorem three_times_x_not_much_different_from_two (x : ℝ) :
  3 * x - 2 ≤ -1 := 
sorry

end three_times_x_not_much_different_from_two_l187_187389


namespace converse_proposition_l187_187703

theorem converse_proposition (a b c : ℝ) (h : c ≠ 0) :
  a * c^2 > b * c^2 → a > b :=
by
  sorry

end converse_proposition_l187_187703


namespace initial_lives_l187_187463

theorem initial_lives (L : ℕ) (h1 : L - 6 + 37 = 41) : L = 10 :=
by
  sorry

end initial_lives_l187_187463


namespace number_of_women_l187_187352

-- Definitions for the given conditions
variables (m w : ℝ)
variable (x : ℝ)

-- Conditions
def cond1 : Prop := 3 * m + 8 * w = 6 * m + 2 * w
def cond2 : Prop := 4 * m + x * w = 0.9285714285714286 * (3 * m + 8 * w)

-- Theorem to prove the number of women in the third group (x)
theorem number_of_women (h1 : cond1 m w) (h2 : cond2 m w x) : x = 5 :=
sorry

end number_of_women_l187_187352


namespace calories_remaining_for_dinner_l187_187501

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l187_187501


namespace possible_values_l187_187046

def seq_condition (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n + a (n + 1) = 2 * a (n + 2) * a (n + 3) + 2016

theorem possible_values (a : ℕ → ℤ) (h : seq_condition a) :
  (a 1, a 2) = (0, 2016) ∨
  (a 1, a 2) = (-14, 70) ∨
  (a 1, a 2) = (-69, 15) ∨
  (a 1, a 2) = (-2015, 1) ∨
  (a 1, a 2) = (2016, 0) ∨
  (a 1, a 2) = (70, -14) ∨
  (a 1, a 2) = (15, -69) ∨
  (a 1, a 2) = (1, -2015) :=
sorry

end possible_values_l187_187046


namespace bob_total_distance_l187_187690

theorem bob_total_distance:
  let time1 := 1.5
  let speed1 := 60
  let time2 := 2
  let speed2 := 45
  (time1 * speed1) + (time2 * speed2) = 180 := 
  by
  sorry

end bob_total_distance_l187_187690


namespace remainder_of_product_mod_7_l187_187516

   theorem remainder_of_product_mod_7 :
     (7 * 17 * 27 * 37 * 47 * 57 * 67) % 7 = 0 := 
   by
     sorry
   
end remainder_of_product_mod_7_l187_187516


namespace polynomial_roots_l187_187031

theorem polynomial_roots :
  (∀ x : ℝ, (x^3 - 2*x^2 - 5*x + 6 = 0 ↔ x = 1 ∨ x = -2 ∨ x = 3)) :=
by
  sorry

end polynomial_roots_l187_187031


namespace stream_speed_is_2_l187_187661

variable (v : ℝ) -- Let v be the speed of the stream in km/h

-- Condition 1: Man's swimming speed in still water
def swimming_speed_still : ℝ := 6

-- Condition 2: It takes him twice as long to swim upstream as downstream
def condition : Prop := (swimming_speed_still + v) / (swimming_speed_still - v) = 2

theorem stream_speed_is_2 : condition v → v = 2 := by
  intro h
  -- Proof goes here
  sorry

end stream_speed_is_2_l187_187661


namespace remainder_poly_l187_187177

theorem remainder_poly (x : ℂ) (h : x^5 + x^4 + x^3 + x^2 + x + 1 = 0) :
  (x^75 + x^60 + x^45 + x^30 + x^15 + 1) % (x^5 + x^4 + x^3 + x^2 + x + 1) = 0 :=
by sorry

end remainder_poly_l187_187177


namespace total_turtles_taken_l187_187969

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l187_187969


namespace inequality_proof_l187_187338

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l187_187338


namespace trail_mix_total_weight_l187_187975

def peanuts : ℝ := 0.17
def chocolate_chips : ℝ := 0.17
def raisins : ℝ := 0.08

theorem trail_mix_total_weight :
  peanuts + chocolate_chips + raisins = 0.42 :=
by
  -- The proof would go here
  sorry

end trail_mix_total_weight_l187_187975


namespace sqrt_calculation_l187_187013

theorem sqrt_calculation : Real.sqrt ((5: ℝ)^2 - (4: ℝ)^2 - (3: ℝ)^2) = 0 := 
by
  -- The proof is skipped
  sorry

end sqrt_calculation_l187_187013


namespace melanie_dimes_final_l187_187235

-- Define a type representing the initial state of Melanie's dimes
variable {initial_dimes : ℕ} (h_initial : initial_dimes = 7)

-- Define a function representing the result after attempting to give away dimes
def remaining_dimes_after_giving (initial_dimes : ℕ) (given_dimes : ℕ) : ℕ :=
  if given_dimes <= initial_dimes then initial_dimes - given_dimes else initial_dimes

-- State the problem
theorem melanie_dimes_final (h_initial : initial_dimes = 7) (given_dimes_dad : ℕ) (h_given_dad : given_dimes_dad = 8) (received_dimes_mom : ℕ) (h_received_mom : received_dimes_mom = 4) :
  remaining_dimes_after_giving initial_dimes given_dimes_dad + received_dimes_mom = 11 :=
by
  sorry

end melanie_dimes_final_l187_187235


namespace max_value_of_expression_l187_187773

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l187_187773


namespace alice_current_age_l187_187021

theorem alice_current_age (a b : ℕ) 
  (h1 : a + 8 = 2 * (b + 8)) 
  (h2 : (a - 10) + (b - 10) = 21) : 
  a = 30 := 
by 
  sorry

end alice_current_age_l187_187021


namespace sugar_per_batch_l187_187242

variable (S : ℝ)

theorem sugar_per_batch :
  (8 * (4 + S) = 44) → (S = 1.5) :=
by
  intro h
  sorry

end sugar_per_batch_l187_187242


namespace marcus_calzones_total_time_l187_187228

/-
Conditions:
1. It takes Marcus 20 minutes to saute the onions.
2. It takes a quarter of the time to saute the garlic and peppers that it takes to saute the onions.
3. It takes 30 minutes to knead the dough.
4. It takes twice as long to let the dough rest as it takes to knead it.
5. It takes 1/10th of the combined kneading and resting time to assemble the calzones.
-/

def time_saute_onions : ℕ := 20
def time_saute_garlic_peppers : ℕ := time_saute_onions / 4
def time_knead : ℕ := 30
def time_rest : ℕ := 2 * time_knead
def time_assemble : ℕ := (time_knead + time_rest) / 10

def total_time_making_calzones : ℕ :=
  time_saute_onions + time_saute_garlic_peppers + time_knead + time_rest + time_assemble

theorem marcus_calzones_total_time : total_time_making_calzones = 124 := by
  -- All steps and proof details to be filled in
  sorry

end marcus_calzones_total_time_l187_187228


namespace find_a_value_l187_187872

theorem find_a_value (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq1 : a^b = b^a) (h_eq2 : b = 3 * a) : a = Real.sqrt 3 :=
  sorry

end find_a_value_l187_187872


namespace inequality_proof_l187_187346

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l187_187346


namespace alpha_is_30_or_60_l187_187047

theorem alpha_is_30_or_60
  (α : Real)
  (h1 : 0 < α ∧ α < Real.pi / 2) -- α is acute angle
  (a : ℝ × ℝ := (3 / 4, Real.sin α))
  (b : ℝ × ℝ := (Real.cos α, 1 / Real.sqrt 3))
  (h2 : a.1 * b.2 = a.2 * b.1)  -- a ∥ b
  : α = Real.pi / 6 ∨ α = Real.pi / 3 := 
sorry

end alpha_is_30_or_60_l187_187047


namespace price_of_basic_computer_l187_187133

theorem price_of_basic_computer (C P : ℝ) 
    (h1 : C + P = 2500) 
    (h2 : P = (1/8) * (C + 500 + P)) :
    C = 2125 :=
by
  sorry

end price_of_basic_computer_l187_187133


namespace product_of_three_consecutive_natural_numbers_divisible_by_six_l187_187244

theorem product_of_three_consecutive_natural_numbers_divisible_by_six (n : ℕ) : 6 ∣ (n * (n + 1) * (n + 2)) :=
by
  sorry

end product_of_three_consecutive_natural_numbers_divisible_by_six_l187_187244


namespace number_of_girls_l187_187625

theorem number_of_girls (n : ℕ) (h1 : 25.choose 2 ≠ 0)
  (h2 : n*(n-1) / 600 = 3 / 25)
  (h3 : 25 - n = 16) : n = 9 :=
by
  sorry

end number_of_girls_l187_187625


namespace problem_1992_AHSME_43_l187_187418

theorem problem_1992_AHSME_43 (a b c : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (h1 : Odd a) (h2 : Odd b) : Odd (3^a + (b-1)^2 * c) :=
sorry

end problem_1992_AHSME_43_l187_187418


namespace intersection_M_N_l187_187857

noncomputable def M : Set ℝ := {x | x^2 - x ≤ 0}
noncomputable def N : Set ℝ := {x | x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l187_187857


namespace unique_solution_of_system_l187_187204

theorem unique_solution_of_system :
  ∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1 ∧ x = 1 ∧ y = 1 ∧ z = 0 := by
  sorry

end unique_solution_of_system_l187_187204


namespace evaluate_expression_l187_187987

theorem evaluate_expression : 2010^3 - 2009 * 2010^2 - 2009^2 * 2010 + 2009^3 = 4019 :=
by
  sorry

end evaluate_expression_l187_187987


namespace sqrt_200_eq_10_sqrt_2_l187_187258

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l187_187258


namespace factor_theorem_l187_187985

noncomputable def Q (b x : ℝ) : ℝ := x^4 - 3 * x^3 + b * x^2 - 12 * x + 24

theorem factor_theorem (b : ℝ) : (∃ x : ℝ, x = -2) ∧ (Q b x = 0) → b = -22 :=
by
  sorry

end factor_theorem_l187_187985


namespace union_complement_l187_187532

universe u

def U : Set ℕ := {0, 2, 4, 6, 8, 10}
def A : Set ℕ := {2, 4, 6}
def B : Set ℕ := {1}

theorem union_complement (U A B : Set ℕ) (hU : U = {0, 2, 4, 6, 8, 10}) (hA : A = {2, 4, 6}) (hB : B = {1}) :
  (U \ A) ∪ B = {0, 1, 8, 10} :=
by
  -- The proof is omitted.
  sorry

end union_complement_l187_187532


namespace simplify_product_l187_187245

theorem simplify_product (a : ℝ) : (1 * 2 * a * 3 * a^2 * 4 * a^3 * 5 * a^4) = 120 * a^10 := by
  sorry

end simplify_product_l187_187245


namespace inequality_proof_l187_187335

theorem inequality_proof (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0):
  (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
  (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
  (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
sorry

end inequality_proof_l187_187335


namespace count_non_adjacent_arrangements_l187_187451

-- Define the number of people and the specific individuals A and B
def num_people : ℕ := 6
def individuals : set ℕ := {1, 2, 3, 4, 5, 6}
def a_person : ℕ := 1
def b_person : ℕ := 2

-- Define the constraint that A and B should not be adjacent
def not_adjacent (arrangement : list ℕ) : Prop :=
  ∀ (i : ℕ), i < arrangement.length - 1 →
    (arrangement.nth i ≠ some a_person ∨ arrangement.nth (i + 1) ≠ some b_person) ∧
    (arrangement.nth i ≠ some b_person ∨ arrangement.nth (i + 1) ≠ some a_person)

-- Define the main theorem statement
theorem count_non_adjacent_arrangements :
  ∃ (num_arrangements : ℕ), num_arrangements = 480 :=
sorry

end count_non_adjacent_arrangements_l187_187451


namespace total_weight_loss_l187_187744

theorem total_weight_loss (S J V : ℝ) 
  (hS : S = 17.5) 
  (hJ : J = 3 * S) 
  (hV : V = S + 1.5) : 
  S + J + V = 89 := 
by 
  sorry

end total_weight_loss_l187_187744


namespace proof_OPQ_Constant_l187_187529

open Complex

def OPQ_Constant :=
  ∀ (z1 z2 : ℂ) (θ : ℝ), abs z1 = 5 ∧
    (z1^2 - z1 * z2 * Real.sin θ + z2^2 = 0) →
      abs z2 = 5

theorem proof_OPQ_Constant : OPQ_Constant :=
by
  sorry

end proof_OPQ_Constant_l187_187529


namespace sqrt_200_eq_10_l187_187249

theorem sqrt_200_eq_10 (h1 : 200 = 2^2 * 5^2)
                        (h2 : ∀ a : ℝ, 0 ≤ a → (real.sqrt (a^2) = a)) : 
                        real.sqrt 200 = 10 :=
by
  sorry

end sqrt_200_eq_10_l187_187249


namespace problem_l187_187206

open Real

noncomputable def f (x : ℝ) : ℝ := x + 1

theorem problem (f : ℝ → ℝ)
  (h : ∀ x, 2 * f x - f (-x) = 3 * x + 1) :
  f 1 = 2 :=
by
  sorry

end problem_l187_187206


namespace animath_extortion_l187_187109

noncomputable def max_extortion (n : ℕ) : ℕ :=
2^n - n - 1 

theorem animath_extortion (n : ℕ) :
  ∃ steps : ℕ, steps < (2^n - n - 1) :=
sorry

end animath_extortion_l187_187109


namespace arrangement_count_l187_187831

noncomputable def arrangements (items : List ℕ) : ℕ := sorry

theorem arrangement_count :
  ∀ (items: List ℕ), 5 ∈ items.length →
  (∀ a b, a ∈ items → b ∈ items → a ≠ b → (∃ idx, items.nth idx = some a ∧ items.nth (idx + 1) = some b) →
  (∀ c d, c ∈ items → d ∈ items → c ≠ d → ¬(∃ idx, items.nth idx = some c ∧ items.nth (idx + 1) = some d)) →
  arrangements items = 48 :=
sorry

end arrangement_count_l187_187831


namespace omar_remaining_coffee_l187_187906

noncomputable def remaining_coffee : ℝ := 
  let initial_coffee := 12
  let after_first_drink := initial_coffee - (initial_coffee * 1/4)
  let after_office_drink := after_first_drink - (after_first_drink * 1/3)
  let espresso_in_ounces := 75 / 29.57
  let after_espresso := after_office_drink + espresso_in_ounces
  let after_lunch_drink := after_espresso - (after_espresso * 0.75)
  let iced_tea_addition := 4 * 1/2
  let after_iced_tea := after_lunch_drink + iced_tea_addition
  let after_cold_drink := after_iced_tea - (after_iced_tea * 0.6)
  after_cold_drink

theorem omar_remaining_coffee : remaining_coffee = 1.654 :=
by 
  sorry

end omar_remaining_coffee_l187_187906


namespace sqrt_200_eq_l187_187286

theorem sqrt_200_eq : Real.sqrt 200 = 10 * Real.sqrt 2 := sorry

end sqrt_200_eq_l187_187286


namespace triangle_area_l187_187011

theorem triangle_area : 
  ∀ (A B C : ℝ × ℝ), 
  A = (0, 0) → 
  B = (4, 0) → 
  C = (2, 6) → 
  (1 / 2 : ℝ) * (4 : ℝ) * (6 : ℝ) = (12.0 : ℝ) := 
by 
  intros A B C hA hB hC
  simp [hA, hB, hC]
  norm_num

end triangle_area_l187_187011


namespace problem_C_l187_187329

def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

noncomputable def is_obtuse_triangle (A B C : ℝ × ℝ) : Prop :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0

theorem problem_C (A B C : ℝ × ℝ) :
  let AB := (B.1 - A.1, B.2 - A.2)
  let BC := (C.1 - B.1, C.2 - B.2)
  dot_product AB BC > 0 → ∃ (u v : ℝ × ℝ), dot_product u v < 0 :=
by
  sorry

end problem_C_l187_187329


namespace problem_a_solution_problem_b_solution_problem_c_solution_problem_d_solution_no_solution_l187_187669

-- System (a)
def system_a_solved (x y z t : ℝ) : Prop :=
    x - 3*y + 2*z - t = 3 ∧
    2*x + 4*y - 3*z + t = 5 ∧
    4*x - 2*y + z + t = 3 ∧
    3*x + y + z - 2*t = 10

theorem problem_a_solution : ∃ x y z t : ℝ, system_a_solved x y z t ∧ x = 2 ∧ y = -1 ∧ z = -3 ∧ t = -4 :=
    by sorry

-- System (b)
def system_b (x y z t : ℝ) : Prop :=
    x + 2*y + 3*z - t = 0 ∧
    x - y + z + 2*t = 4 ∧
    x + 5*y + 5*z - 4*t = -4 ∧
    x + 8*y + 7*z - 7*t = -8

theorem problem_b_solution : ∃ x y z t : ℝ, system_b x y z t :=
   by sorry

-- System (c)
def system_c_solved (x y z : ℝ) : Prop :=
    x + 2*y + 3*z = 2 ∧
    x - y + z = 0 ∧
    x + 3*y - z = -2 ∧
    3*x + 4*y + 3*z = 0

theorem problem_c_solution : ∃ x y z : ℝ, system_c_solved x y z ∧ x = -1 ∧ y = 0 ∧ z = 1 :=
    by sorry

-- System (d)
def system_d (x y z t : ℝ) : Prop :=
    x + 2*y + 3*z - t = 0 ∧
    x - y + z + 2*t = 4 ∧
    x + 5*y + 5*z - 4*t = -4 ∧
    x + 8*y + 7*z - 7*t = 6

theorem problem_d_solution_no_solution : ¬ ∃ x y z t : ℝ, system_d x y z t :=
    by sorry

end problem_a_solution_problem_b_solution_problem_c_solution_problem_d_solution_no_solution_l187_187669


namespace conversion_base8_to_base10_l187_187691

theorem conversion_base8_to_base10 : 5 * 8^3 + 2 * 8^2 + 1 * 8^1 + 4 * 8^0 = 2700 :=
by 
  sorry

end conversion_base8_to_base10_l187_187691


namespace max_value_of_fraction_l187_187765

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l187_187765


namespace count_even_integers_between_l187_187714

theorem count_even_integers_between : 
    let lower := 18 / 5
    let upper := 45 / 2
    ∃ (count : ℕ), (∀ n : ℕ, lower < n ∧ n < upper → n % 2 = 0 → n = 4 ∨ n = 6 ∨ n = 8 ∨ n = 10 ∨ n = 12 ∨ n = 14 ∨ n = 16 ∨ n = 18 ∨ n = 20 ∨ n = 22) ∧ count = 10 :=
by
  sorry

end count_even_integers_between_l187_187714


namespace sum_of_solutions_eq_9_l187_187641

theorem sum_of_solutions_eq_9 (x_1 x_2 : ℝ) (h : x^2 - 9 * x + 20 = 0) :
  x_1 + x_2 = 9 :=
sorry

end sum_of_solutions_eq_9_l187_187641


namespace range_of_a_l187_187707

noncomputable def range_a : Set ℝ :=
  {a : ℝ | 0 < a ∧ a ≤ 1/2}

theorem range_of_a (O P : ℝ × ℝ) (Q : ℝ × ℝ)
  (hO : O = (0, 0))
  (hP : P = (a, 0))
  (ha : 0 < a)
  (hQ : ∃ m : ℝ, Q = (m^2, m))
  (hPQ_PO : ∀ Q, Q = (m^2, m) → dist P Q ≥ dist O P) :
  a ∈ range_a :=
sorry

end range_of_a_l187_187707


namespace g_of_3_l187_187706

theorem g_of_3 (g : ℝ → ℝ) (h : ∀ x : ℝ, x ≠ 0 → 4 * g x + 3 * g (1 / x) = 2 * x) :
  g 3 = 22 / 7 :=
sorry

end g_of_3_l187_187706


namespace sum_and_product_of_roots_l187_187545

theorem sum_and_product_of_roots (a b : ℝ) : (∀ x : ℝ, x^2 + a * x + b = 0 → x = -2 ∨ x = 3) → a + b = -7 :=
by
  sorry

end sum_and_product_of_roots_l187_187545


namespace expected_value_of_rounds_passed_l187_187137

-- Player's probability of making a shot
noncomputable def p : ℚ := 2 / 3

-- Player's probability of making at least one shot in a round (passes the round)
noncomputable def p_pass : ℚ := 1 - ((1 - p) * (1 - p))

-- Number of rounds
def n : ℕ := 3

-- Expected number of rounds passed by player A
theorem expected_value_of_rounds_passed :
  ∑ i in finset.range (n + 1), (nat.choose n i : ℚ) * p_pass ^ i * (1 - p_pass) ^ (n - i) * i = 8 / 3 :=
by sorry

end expected_value_of_rounds_passed_l187_187137


namespace total_oranges_picked_l187_187445

theorem total_oranges_picked (mary_oranges : Nat) (jason_oranges : Nat) (hmary : mary_oranges = 122) (hjason : jason_oranges = 105) : mary_oranges + jason_oranges = 227 := by
  sorry

end total_oranges_picked_l187_187445


namespace factorize_expression_l187_187173

theorem factorize_expression (a : ℚ) : 2 * a^2 - 4 * a = 2 * a * (a - 2) := by
  sorry

end factorize_expression_l187_187173


namespace fixed_point_of_function_l187_187788

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a^(1 - x) - 2

theorem fixed_point_of_function (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = -1 := by
  sorry

end fixed_point_of_function_l187_187788


namespace cube_square_third_smallest_prime_l187_187315

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l187_187315


namespace adults_riding_bicycles_l187_187377

theorem adults_riding_bicycles (A : ℕ) (H1 : 15 * 3 + 2 * A = 57) : A = 6 :=
by
  sorry

end adults_riding_bicycles_l187_187377


namespace percent_increase_l187_187832

variable (P : ℝ)
def firstQuarterPrice := 1.20 * P
def secondQuarterPrice := 1.50 * P

theorem percent_increase:
  ((secondQuarterPrice P - firstQuarterPrice P) / firstQuarterPrice P) * 100 = 25 := by
  sorry

end percent_increase_l187_187832


namespace first_term_of_new_ratio_l187_187002

-- Given conditions as definitions
def original_ratio : ℚ := 6 / 7
def x (n : ℕ) : Prop := n ≥ 3

-- Prove that the first term of the ratio that the new ratio should be less than is 4
theorem first_term_of_new_ratio (n : ℕ) (h1 : x n) : ∃ b, (6 - n) / (7 - n) < 4 / b :=
by
  exists 5
  sorry

end first_term_of_new_ratio_l187_187002


namespace cos_theta_value_l187_187193

theorem cos_theta_value (θ : ℝ) (h_tan : Real.tan θ = -4/3) (h_range : 0 < θ ∧ θ < π) : Real.cos θ = -3/5 :=
by
  sorry

end cos_theta_value_l187_187193


namespace temperature_problem_product_of_possible_N_l187_187376

theorem temperature_problem (M L : ℤ) (N : ℤ) :
  (M = L + N) →
  (M - 8 = L + N - 8) →
  (L + 4 = L + 4) →
  (|((L + N - 8) - (L + 4))| = 3) →
  N = 15 ∨ N = 9 :=
by sorry

theorem product_of_possible_N :
  (∀ M L : ℤ, ∀ N : ℤ,
    (M = L + N) →
    (M - 8 = L + N - 8) →
    (L + 4 = L + 4) →
    (|((L + N - 8) - (L + 4))| = 3) →
    N = 15 ∨ N = 9) →
    15 * 9 = 135 :=
by sorry

end temperature_problem_product_of_possible_N_l187_187376


namespace time_taken_by_C_l187_187481

theorem time_taken_by_C (days_A B C : ℕ) (work_done_A work_done_B work_done_C : ℚ) 
  (h1 : days_A = 40) (h2 : work_done_A = 10 * (1/40)) 
  (h3 : days_B = 40) (h4 : work_done_B = 10 * (1/40)) 
  (h5 : work_done_C = 1/2)
  (h6 : 10 * work_done_C = 1/2) :
  (10 * 2) = 20 := 
sorry

end time_taken_by_C_l187_187481


namespace equivalent_terminal_side_l187_187393

theorem equivalent_terminal_side (k : ℤ) : 
    (∃ k : ℤ, (5 * π / 3 = -π / 3 + 2 * π * k)) :=
sorry

end equivalent_terminal_side_l187_187393


namespace parabola_focus_l187_187199

theorem parabola_focus (p : ℝ) (hp : 0 < p) (h : ∀ y x : ℝ, y^2 = 2 * p * x → (x = 2 ∧ y = 0)) : p = 4 :=
sorry

end parabola_focus_l187_187199


namespace conservation_center_total_turtles_l187_187964

-- Define the green turtles and the relationship between green and hawksbill turtles.
def green_turtles : ℕ := 800
def hawksbill_turtles : ℕ := 2 * green_turtles

-- Statement we need to prove, which is the total number of turtles equals 3200.
theorem conservation_center_total_turtles : green_turtles + hawksbill_turtles = 3200 := by
  sorry

end conservation_center_total_turtles_l187_187964


namespace students_in_dexters_high_school_l187_187072

variables (D S N : ℕ)

theorem students_in_dexters_high_school :
  (D = 4 * S) ∧
  (D + S + N = 3600) ∧
  (N = S - 400) →
  D = 8000 / 3 := 
sorry

end students_in_dexters_high_school_l187_187072


namespace charity_dinner_cost_l187_187356

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l187_187356


namespace cost_price_computer_table_l187_187920

noncomputable def approx_eq (x y : ℝ) (ε : ℝ) : Prop := abs (x - y) < ε

theorem cost_price_computer_table (SP : ℝ) (CP : ℝ) (h : SP = 7967) (h2 : SP = 1.24 * CP) : 
  approx_eq CP 6424 0.01 :=
by
  sorry

end cost_price_computer_table_l187_187920


namespace choir_average_age_l187_187073

-- Conditions
def women_count : ℕ := 12
def men_count : ℕ := 10
def avg_age_women : ℝ := 25.0
def avg_age_men : ℝ := 40.0

-- Expected Answer
def expected_avg_age : ℝ := 31.82

-- Proof Statement
theorem choir_average_age :
  ((women_count * avg_age_women) + (men_count * avg_age_men)) / (women_count + men_count) = expected_avg_age :=
by
  sorry

end choir_average_age_l187_187073


namespace range_of_m_l187_187201

noncomputable def is_quadratic (m : ℝ) : Prop := (m^2 - 4) ≠ 0

theorem range_of_m (m : ℝ) : is_quadratic m → m ≠ 2 ∧ m ≠ -2 :=
by sorry

end range_of_m_l187_187201


namespace mean_score_of_seniors_l187_187094

variable (s n : ℕ)  -- Number of seniors and non-seniors
variable (m_s m_n : ℝ)  -- Mean scores of seniors and non-seniors
variable (total_mean : ℝ) -- Mean score of all students
variable (total_students : ℕ) -- Total number of students

theorem mean_score_of_seniors :
  total_students = 100 → total_mean = 100 →
  n = 3 * s / 2 →
  s * m_s + n * m_n = total_students * total_mean →
  m_s = (3 * m_n / 2) →
  m_s = 125 :=
by
  intros
  sorry

end mean_score_of_seniors_l187_187094


namespace solve_abs_eq_l187_187646

theorem solve_abs_eq (x : ℝ) (h : |x - 3| = |x - 5|) : x = 4 :=
by {
  -- This proof is left as an exercise
  sorry
}

end solve_abs_eq_l187_187646


namespace estimate_sqrt_interval_l187_187170

theorem estimate_sqrt_interval : 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_sqrt_interval_l187_187170


namespace line_BC_eq_circumscribed_circle_eq_l187_187407

noncomputable def A : ℝ × ℝ := (3, 0)
noncomputable def B : ℝ × ℝ := (0, -1)
noncomputable def altitude_line (x y : ℝ) : Prop := x + y + 1 = 0
noncomputable def equation_line_BC (x y : ℝ) : Prop := 3 * x - y - 1 = 0
noncomputable def circumscribed_circle (x y : ℝ) : Prop := (x - 5 / 2)^2 + (y + 7 / 2)^2 = 50 / 4

theorem line_BC_eq :
  ∃ x y : ℝ, altitude_line x y →
             B = (x, y) →
             equation_line_BC x y :=
by sorry

theorem circumscribed_circle_eq :
  ∃ x y : ℝ, altitude_line x y →
             (x - 3)^2 + y^2 = (5 / 2)^2 →
             circumscribed_circle x y :=
by sorry

end line_BC_eq_circumscribed_circle_eq_l187_187407


namespace find_roots_of_polynomial_l187_187028

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l187_187028


namespace simplify_sqrt_200_l187_187279

theorem simplify_sqrt_200 : (sqrt 200 : ℝ) = 10 * sqrt 2 := by
  -- proof goes here
  sorry

end simplify_sqrt_200_l187_187279


namespace proportion_equation_l187_187061

theorem proportion_equation (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
by 
  sorry

end proportion_equation_l187_187061


namespace solution_for_x2_l187_187682

def eq1 (x : ℝ) := 2 * x = 6
def eq2 (x : ℝ) := x + 2 = 0
def eq3 (x : ℝ) := x - 5 = 3
def eq4 (x : ℝ) := 3 * x - 6 = 0

theorem solution_for_x2 : ∀ x : ℝ, x = 2 → ¬eq1 x ∧ ¬eq2 x ∧ ¬eq3 x ∧ eq4 x :=
by 
  sorry

end solution_for_x2_l187_187682


namespace induction_inequality_l187_187802

variable (n : ℕ) (h₁ : n ∈ Set.Icc 2 (2^n - 1))

theorem induction_inequality : 1 + 1/2 + 1/3 < 2 := 
  sorry

end induction_inequality_l187_187802


namespace find_minimum_width_l187_187216

-- Definitions based on the problem conditions
def length_from_width (w : ℝ) : ℝ := w + 12

def minimum_fence_area (w : ℝ) : Prop := w * length_from_width w ≥ 144

-- Proof statement
theorem find_minimum_width : ∃ w : ℝ, w ≥ 6 ∧ minimum_fence_area w :=
sorry

end find_minimum_width_l187_187216


namespace cube_probability_l187_187360

def prob_same_color_vertical_faces : ℕ := sorry

theorem cube_probability :
  prob_same_color_vertical_faces = 1 / 27 := 
sorry

end cube_probability_l187_187360


namespace avg_speed_including_stoppages_l187_187988

theorem avg_speed_including_stoppages (speed_without_stoppages : ℝ) (stoppage_time_per_hour : ℝ) 
  (h₁ : speed_without_stoppages = 60) (h₂ : stoppage_time_per_hour = 0.5) : 
  (speed_without_stoppages * (1 - stoppage_time_per_hour)) / 1 = 30 := 
  by 
  sorry

end avg_speed_including_stoppages_l187_187988


namespace total_turtles_l187_187967

theorem total_turtles (num_green_turtles : ℕ) (num_hawksbill_turtles : ℕ) 
  (h1 : num_green_turtles = 800)
  (h2 : num_hawksbill_turtles = 2 * 800 + 800) :
  num_green_turtles + num_hawksbill_turtles = 3200 := 
by
  sorry

end total_turtles_l187_187967


namespace parabola_distance_to_focus_l187_187875

theorem parabola_distance_to_focus :
  ∀ (P : ℝ × ℝ), P.1 = 2 ∧ P.2^2 = 4 * P.1 → dist P (1, 0) = 3 :=
by
  intro P h
  have h₁ : P.1 = 2 := h.1
  have h₂ : P.2^2 = 4 * P.1 := h.2
  sorry

end parabola_distance_to_focus_l187_187875


namespace inequality_example_l187_187404

theorem inequality_example (a b c : ℝ) (hac : a ≠ 0) (hbc : b ≠ 0) (hcc : c ≠ 0) :
  (a^4) / (4 * a^4 + b^4 + c^4) + (b^4) / (a^4 + 4 * b^4 + c^4) + (c^4) / (a^4 + b^4 + 4 * c^4) ≤ 1 / 2 :=
sorry

end inequality_example_l187_187404


namespace remainder_of_sum_of_squares_mod_n_l187_187835

theorem remainder_of_sum_of_squares_mod_n (a b n : ℤ) (hn : n > 1) 
  (ha : a * a ≡ 1 [ZMOD n]) (hb : b * b ≡ 1 [ZMOD n]) : 
  (a^2 + b^2) % n = 2 := 
by 
  sorry

end remainder_of_sum_of_squares_mod_n_l187_187835


namespace train_speed_proof_l187_187817

noncomputable def speedOfTrain (lengthOfTrain : ℝ) (timeToCross : ℝ) (speedOfMan : ℝ) : ℝ :=
  let man_speed_m_per_s := speedOfMan * 1000 / 3600
  let relative_speed := lengthOfTrain / timeToCross
  let train_speed_m_per_s := relative_speed + man_speed_m_per_s
  train_speed_m_per_s * 3600 / 1000

theorem train_speed_proof :
  speedOfTrain 100 5.999520038396929 3 = 63 := by
  sorry

end train_speed_proof_l187_187817


namespace simplify_expression_l187_187585

theorem simplify_expression (x : ℝ) : (2 * x + 20) + (150 * x + 25) = 152 * x + 45 :=
by
  sorry

end simplify_expression_l187_187585


namespace imo1965_cmo6511_l187_187511

theorem imo1965_cmo6511 (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ∧
  |(Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x)))| ≤ Real.sqrt 2 ↔
  ((Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 2) ∨ (3 * Real.pi / 2 ≤ x ∧ x ≤ 7 * Real.pi / 4)) :=
sorry

end imo1965_cmo6511_l187_187511


namespace john_spent_at_candy_store_l187_187055

-- Conditions
def weekly_allowance : ℚ := 2.25
def spent_at_arcade : ℚ := (3 / 5) * weekly_allowance
def remaining_after_arcade : ℚ := weekly_allowance - spent_at_arcade
def spent_at_toy_store : ℚ := (1 / 3) * remaining_after_arcade
def remaining_after_toy_store : ℚ := remaining_after_arcade - spent_at_toy_store

-- Problem: Prove that John spent $0.60 at the candy store
theorem john_spent_at_candy_store : remaining_after_toy_store = 0.60 :=
by
  sorry

end john_spent_at_candy_store_l187_187055


namespace average_length_of_ropes_l187_187300

def length_rope_1 : ℝ := 2
def length_rope_2 : ℝ := 6

theorem average_length_of_ropes :
  (length_rope_1 + length_rope_2) / 2 = 4 :=
by
  sorry

end average_length_of_ropes_l187_187300


namespace scientific_notation_of_150000000000_l187_187458

theorem scientific_notation_of_150000000000 :
  150000000000 = 1.5 * 10^11 :=
sorry

end scientific_notation_of_150000000000_l187_187458


namespace marcus_calzones_total_time_l187_187225

theorem marcus_calzones_total_time :
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  total_time = 124 :=
by
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  sorry

end marcus_calzones_total_time_l187_187225


namespace compare_logs_and_exp_l187_187195

theorem compare_logs_and_exp :
  let a := Real.log 3 / Real.log 5
  let b := Real.log 8 / Real.log 13
  let c := Real.exp (-1 / 2)
  c < a ∧ a < b := 
sorry

end compare_logs_and_exp_l187_187195


namespace water_consumed_is_correct_l187_187375

def water_consumed (traveler_ounces : ℕ) (camel_multiplier : ℕ) (ounces_per_gallon : ℕ) : ℕ :=
  let camel_ounces := traveler_ounces * camel_multiplier
  let total_ounces := traveler_ounces + camel_ounces
  total_ounces / ounces_per_gallon

theorem water_consumed_is_correct :
  water_consumed 32 7 128 = 2 :=
by
  -- add proof here
  sorry

end water_consumed_is_correct_l187_187375


namespace inequality_xyz_l187_187340

theorem inequality_xyz (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x^3 / (x^3 + 2 * y^2 * real.sqrt (z * x))) +
  (y^3 / (y^3 + 2 * z^2 * real.sqrt (x * y))) +
  (z^3 / (z^3 + 2 * x^2 * real.sqrt (y * z))) ≥ 1 :=
by sorry

end inequality_xyz_l187_187340


namespace wendy_dentist_bill_l187_187635

theorem wendy_dentist_bill : 
  let cost_cleaning := 70
  let cost_filling := 120
  let num_fillings := 3
  let cost_root_canal := 400
  let cost_dental_crown := 600
  let total_bill := 9 * cost_root_canal
  let known_costs := cost_cleaning + (num_fillings * cost_filling) + cost_root_canal + cost_dental_crown
  let cost_tooth_extraction := total_bill - known_costs
  cost_tooth_extraction = 2170 := by
  sorry

end wendy_dentist_bill_l187_187635


namespace min_plus_max_value_of_x_l187_187733

theorem min_plus_max_value_of_x (x y z : ℝ) (h1 : x + y + z = 5) (h2 : x^2 + y^2 + z^2 = 11) :
  let m := (10 - Real.sqrt 304) / 6
  let M := (10 + Real.sqrt 304) / 6
  m + M = 10 / 3 := by 
  sorry

end min_plus_max_value_of_x_l187_187733


namespace sin_13pi_over_4_eq_neg_sqrt2_over_2_l187_187510

theorem sin_13pi_over_4_eq_neg_sqrt2_over_2 : Real.sin (13 * Real.pi / 4) = -Real.sqrt 2 / 2 := 
by 
  sorry

end sin_13pi_over_4_eq_neg_sqrt2_over_2_l187_187510


namespace repair_time_l187_187140

theorem repair_time {x : ℝ} :
  (∀ (a b : ℝ), a = 3 ∧ b = 6 → (((1 / a) + (1 / b)) * x = 1) → x = 2) :=
by
  intros a b hab h
  rcases hab with ⟨ha, hb⟩
  sorry

end repair_time_l187_187140


namespace sqrt_sum_gt_l187_187114

theorem sqrt_sum_gt (a b : ℝ) (ha : a = 2) (hb : b = 3) : 
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by 
  sorry

end sqrt_sum_gt_l187_187114


namespace complex_number_solution_l187_187025

open Complex

theorem complex_number_solution (z : ℂ) (h : z^2 = -99 - 40 * I) : z = 2 - 10 * I ∨ z = -2 + 10 * I :=
sorry

end complex_number_solution_l187_187025


namespace possible_measures_of_angle_A_l187_187598

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l187_187598


namespace abs_eq_condition_l187_187645

theorem abs_eq_condition (x : ℝ) : |x - 3| = |x - 5| → x = 4 :=
by
  sorry

end abs_eq_condition_l187_187645


namespace maria_cookies_left_l187_187229

-- Define the initial conditions and necessary variables
def initial_cookies : ℕ := 19
def given_cookies_to_friend : ℕ := 5
def eaten_cookies : ℕ := 2

-- Define remaining cookies after each step
def remaining_after_friend (total : ℕ) := total - given_cookies_to_friend
def remaining_after_family (remaining : ℕ) := remaining / 2
def remaining_after_eating (after_family : ℕ) := after_family - eaten_cookies

-- Main theorem to prove
theorem maria_cookies_left :
  let initial := initial_cookies,
      after_friend := remaining_after_friend initial,
      after_family := remaining_after_family after_friend,
      final := remaining_after_eating after_family
  in final = 5 :=
by
  sorry

end maria_cookies_left_l187_187229


namespace max_value_of_fraction_l187_187759

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l187_187759


namespace sqrt_200_eq_10_l187_187262

theorem sqrt_200_eq_10 : real.sqrt 200 = 10 :=
by
  calc
    real.sqrt 200 = real.sqrt (2^2 * 5^2) : by sorry -- show 200 = 2^2 * 5^2
    ... = real.sqrt (2^2) * real.sqrt (5^2) : by sorry -- property of square roots of products
    ... = 2 * 5 : by sorry -- using the property sqrt (a^2) = a
    ... = 10 : by sorry

end sqrt_200_eq_10_l187_187262


namespace find_ab_l187_187413

noncomputable def perpendicular_condition (a b : ℝ) :=
  a * (a - 1) - b = 0

noncomputable def point_on_l1_condition (a b : ℝ) :=
  -3 * a + b + 4 = 0

noncomputable def parallel_condition (a b : ℝ) :=
  a + b * (a - 1) = 0

noncomputable def distance_condition (a : ℝ) :=
  4 = abs ((-a) / (a - 1))

theorem find_ab (a b : ℝ) :
  (perpendicular_condition a b ∧ point_on_l1_condition a b ∧
   parallel_condition a b ∧ distance_condition a) →
  ((a = 2 ∧ b = 2) ∨ (a = 2 ∧ b = -2) ∨ (a = -2 ∧ b = 2)) :=
by
  sorry

end find_ab_l187_187413


namespace parallel_vectors_l187_187712

def a : (ℝ × ℝ) := (1, -2)
def b (x : ℝ) : (ℝ × ℝ) := (-2, x)

theorem parallel_vectors (x : ℝ) (h : 1 / -2 = -2 / x) : x = 4 := by
  sorry

end parallel_vectors_l187_187712


namespace gardener_payment_l187_187461

theorem gardener_payment (total_cost : ℕ) (rect_area : ℕ) (rect_side1 : ℕ) (rect_side2 : ℕ)
                         (square1_area : ℕ) (square2_area : ℕ) (cost_per_are : ℕ) :
  total_cost = 570 →
  rect_area = 600 → rect_side1 = 20 → rect_side2 = 30 →
  square1_area = 400 → square2_area = 900 →
  cost_per_are * (rect_area + square1_area + square2_area) / 100 = total_cost →
  cost_per_are = 30 →
  ∃ (rect_payment : ℕ) (square1_payment : ℕ) (square2_payment : ℕ),
    rect_payment = 6 * cost_per_are ∧
    square1_payment = 4 * cost_per_are ∧
    square2_payment = 9 * cost_per_are ∧
    rect_payment + square1_payment + square2_payment = total_cost :=
by
  intros
  sorry

end gardener_payment_l187_187461


namespace max_value_of_expression_l187_187775

theorem max_value_of_expression (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : x + y + z = 180) :
  ∃ (u : ℚ), u = 17 ∧ (∀ (x y z : ℕ), 10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 ∧ 10 ≤ z ∧ z < 100 ∧ x + y + z = 180 → (x + y : ℚ) / z ≤ u) :=
begin
  sorry
end

end max_value_of_expression_l187_187775


namespace problem_statement_l187_187087

-- Define the roots of the quadratic as r and s
variables (r s : ℝ)

-- Given conditions
def root_condition (r s : ℝ) := (r + s = 2 * Real.sqrt 6) ∧ (r * s = 3)

theorem problem_statement (h : root_condition r s) : r^8 + s^8 = 93474 :=
sorry

end problem_statement_l187_187087


namespace total_amount_l187_187827

theorem total_amount (x : ℝ) (hC : 2 * x = 70) :
  let B_share := 1.25 * x
  let C_share := 2 * x
  let D_share := 0.7 * x
  let E_share := 0.5 * x
  let A_share := x
  B_share + C_share + D_share + E_share + A_share = 190.75 :=
by
  sorry

end total_amount_l187_187827


namespace cube_square_third_smallest_prime_l187_187316

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l187_187316


namespace defective_units_shipped_percentage_l187_187428

theorem defective_units_shipped_percentage :
  let units_produced := 100
  let typeA_defective := 0.07 * units_produced
  let typeB_defective := 0.08 * units_produced
  let typeA_shipped := 0.03 * typeA_defective
  let typeB_shipped := 0.06 * typeB_defective
  let total_shipped := typeA_shipped + typeB_shipped
  let percentage_shipped := total_shipped / units_produced * 100
  percentage_shipped = 1 :=
by
  sorry

end defective_units_shipped_percentage_l187_187428


namespace necessary_and_sufficient_l187_187478

def point_on_curve (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) : Prop :=
  f P = 0

theorem necessary_and_sufficient (P : ℝ × ℝ) (f : ℝ × ℝ → ℝ) :
  (point_on_curve P f ↔ f P = 0) :=
by
  sorry

end necessary_and_sufficient_l187_187478


namespace min_value_expression_l187_187841

theorem min_value_expression (x y : ℝ) (hx : |x| < 1) (hy : |y| < 2) (hxy : x * y = 1) : 
  ∃ k, k = 4 ∧ (∀ z, z = (1 / (1 - x^2) + 4 / (4 - y^2)) → z ≥ k) :=
sorry

end min_value_expression_l187_187841


namespace fraction_multiplication_subtraction_l187_187475

theorem fraction_multiplication_subtraction :
  (3 + 1 / 117) * (4 + 1 / 119) - (2 - 1 / 117) * (6 - 1 / 119) - (5 / 119) = 10 / 117 :=
by
  sorry

end fraction_multiplication_subtraction_l187_187475


namespace bridge_max_weight_l187_187956

variables (M K Mi B : ℝ)

-- Given conditions
def kelly_weight : K = 34 := sorry
def kelly_megan_relation : K = 0.85 * M := sorry
def mike_megan_relation : Mi = M + 5 := sorry
def total_excess : K + M + Mi = B + 19 := sorry

-- Proof goal: The maximum weight the bridge can hold is 100 kg.
theorem bridge_max_weight : B = 100 :=
by
  sorry

end bridge_max_weight_l187_187956


namespace max_value_of_fraction_l187_187760

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l187_187760


namespace new_sailor_weight_l187_187131

-- Define the conditions
variables {average_weight : ℝ} (new_weight : ℝ)
variable (old_weight : ℝ := 56)

-- State the property we need to prove
theorem new_sailor_weight
  (h : (new_weight - old_weight) = 8) :
  new_weight = 64 :=
by
  sorry

end new_sailor_weight_l187_187131


namespace distance_between_parabola_vertices_l187_187020

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem distance_between_parabola_vertices :
  distance (0, 3) (0, -1) = 4 := 
by {
  -- Proof omitted here
  sorry
}

end distance_between_parabola_vertices_l187_187020


namespace find_x_perpendicular_l187_187999

-- Definitions used in the conditions
def a : ℝ × ℝ := (3, 2)
def b (x : ℝ) : ℝ × ℝ := (x, 4)

-- Condition: vectors a and b are perpendicular
def perpendicular (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- The theorem we want to prove
theorem find_x_perpendicular : ∀ x : ℝ, perpendicular a (b x) → x = -8 / 3 :=
by
  intros x h
  sorry

end find_x_perpendicular_l187_187999


namespace probability_of_jack_king_ace_l187_187797

theorem probability_of_jack_king_ace :
  let prob_jack := (4 : ℚ) / 52,
      prob_king := (4 : ℚ) / 51,
      prob_ace := (4 : ℚ) / 50 in
  prob_jack * prob_king * prob_ace = 16 / 33150 := 
by
  sorry

end probability_of_jack_king_ace_l187_187797


namespace longest_side_of_similar_triangle_l187_187490

theorem longest_side_of_similar_triangle :
  ∀ (x : ℝ),
    let a := 8
    let b := 10
    let c := 12
    let s₁ := a * x
    let s₂ := b * x
    let s₃ := c * x
    a + b + c = 30 → 
    30 * x = 150 → 
    s₁ > 30 → 
    max s₁ (max s₂ s₃) = 60 :=
by
  intros x a b c s₁ s₂ s₃ h₁ h₂ h₃
  sorry

end longest_side_of_similar_triangle_l187_187490


namespace final_price_correct_l187_187829

noncomputable def original_price : ℝ := 49.99
noncomputable def first_discount : ℝ := 0.10
noncomputable def second_discount : ℝ := 0.20

theorem final_price_correct :
  let price_after_first_discount := original_price * (1 - first_discount)
  let price_after_second_discount := price_after_first_discount * (1 - second_discount)
  price_after_second_discount = 36.00 := by
    -- The proof would go here
    sorry

end final_price_correct_l187_187829


namespace product_approximation_l187_187379

theorem product_approximation :
  (3.05 * 7.95 * (6.05 + 3.95)) = 240 := by
  sorry

end product_approximation_l187_187379


namespace tan_alpha_value_tan_beta_value_sum_angles_l187_187550

open Real

noncomputable def tan_alpha (α : ℝ) : ℝ := sin α / cos α
noncomputable def tan_beta (β : ℝ) : ℝ := sin β / cos β

def conditions (α β : ℝ) :=
  α > 0 ∧ α < π / 2 ∧ β > 0 ∧ β < π / 2 ∧ 
  sin α = 1 / sqrt 10 ∧ tan β = 1 / 7

theorem tan_alpha_value (α β : ℝ) (h : conditions α β) : tan_alpha α = 1 / 3 := sorry

theorem tan_beta_value (α β : ℝ) (h : conditions α β) : tan_beta β = 1 / 7 := sorry

theorem sum_angles (α β : ℝ) (h : conditions α β) : 2 * α + β = π / 4 := sorry

end tan_alpha_value_tan_beta_value_sum_angles_l187_187550


namespace partial_fraction_sum_zero_l187_187833

theorem partial_fraction_sum_zero (A B C D E F : ℝ) :
  (∀ x : ℝ, 1 / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
     A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5)) →
  A + B + C + D + E + F = 0 :=
sorry

end partial_fraction_sum_zero_l187_187833


namespace min_value_of_a_l187_187997

noncomputable def x (t a : ℝ) : ℝ :=
  5 * (t + 1)^2 + a / (t + 1)^5

theorem min_value_of_a (a : ℝ) :
  (∀ t : ℝ, t ≥ 0 → x t a ≥ 24) ↔ a ≥ 2 * Real.sqrt ((24 / 7)^7) :=
sorry

end min_value_of_a_l187_187997


namespace perpendicular_line_eq_l187_187805

theorem perpendicular_line_eq (x y : ℝ) :
  (∃ (p : ℝ × ℝ), p = (-2, 3) ∧ 
    ∀ y₀ x₀, 3 * x - y = 6 ∧ y₀ = 3 ∧ x₀ = -2 → y = -1 / 3 * x + 7 / 3) :=
sorry

end perpendicular_line_eq_l187_187805


namespace combined_weight_is_correct_l187_187885

-- Define the conditions
def elephant_weight_tons : ℕ := 3
def ton_in_pounds : ℕ := 2000
def donkey_weight_percentage : ℕ := 90

-- Convert elephant's weight to pounds
def elephant_weight_pounds : ℕ := elephant_weight_tons * ton_in_pounds

-- Calculate the donkeys's weight
def donkey_weight_pounds : ℕ := elephant_weight_pounds - (elephant_weight_pounds * donkey_weight_percentage / 100)

-- Define the combined weight
def combined_weight : ℕ := elephant_weight_pounds + donkey_weight_pounds

-- Prove the combined weight is 6600 pounds
theorem combined_weight_is_correct : combined_weight = 6600 :=
by
  sorry

end combined_weight_is_correct_l187_187885


namespace Janka_bottle_caps_l187_187432

theorem Janka_bottle_caps (n : ℕ) :
  (∃ k1 : ℕ, n = 3 * k1) ∧ (∃ k2 : ℕ, n = 4 * k2) ↔ n = 12 ∨ n = 24 :=
by
  sorry

end Janka_bottle_caps_l187_187432


namespace triangle_side_ratio_l187_187546

variables (a b c S : ℝ)
variables (A B C : ℝ)

/-- In triangle ABC, if the sides opposite to angles A, B, and C are a, b, and c respectively,
    and given a=1, B=π/4, and the area S=2, we prove that b / sin(B) = 5√2. -/
theorem triangle_side_ratio (h₁ : a = 1) (h₂ : B = Real.pi / 4) (h₃ : S = 2) : b / Real.sin B = 5 * Real.sqrt 2 :=
sorry

end triangle_side_ratio_l187_187546


namespace probability_math_majors_consecutive_l187_187119

noncomputable def total_ways := Nat.choose 11 4 -- Number of ways to choose 5 persons out of 12 (fixing one)
noncomputable def favorable_ways := 12         -- Number of ways to arrange 5 math majors consecutively around a round table

theorem probability_math_majors_consecutive :
  (favorable_ways : ℚ) / total_ways = 2 / 55 :=
by
  sorry

end probability_math_majors_consecutive_l187_187119


namespace subtraction_of_smallest_from_largest_l187_187658

open Finset

def three_digit_numbers (s : Finset ℕ) : Finset ℕ :=
  (s.product (s.filter (· ≠ _)).product (s.filter (· ≠ _)).map 
    (λ x, 100 * x.1.1 + 10 * x.1.2 + x.2)).filter (λ n, n >= 100 ∧ n < 1000)

theorem subtraction_of_smallest_from_largest : 
  let numbers := {1, 2, 6, 7, 8} in
  let largest := max' (three_digit_numbers numbers) sorry in
  let smallest := min' (three_digit_numbers numbers) sorry in
  largest - smallest = 750 := 
by
  sorry

end subtraction_of_smallest_from_largest_l187_187658


namespace prism_volume_l187_187446

noncomputable def volume_prism (x y z : ℝ) : ℝ := x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 12) (h2 : y * z = 8) (h3 : z * x = 6) :
  volume_prism x y z = 24 :=
by
  sorry

end prism_volume_l187_187446


namespace total_number_of_eyes_l187_187092

theorem total_number_of_eyes (n_spiders n_ants eyes_per_spider eyes_per_ant : ℕ)
  (h1 : n_spiders = 3) (h2 : n_ants = 50) (h3 : eyes_per_spider = 8) (h4 : eyes_per_ant = 2) :
  (n_spiders * eyes_per_spider + n_ants * eyes_per_ant) = 124 :=
by
  sorry

end total_number_of_eyes_l187_187092


namespace man_arrived_earlier_l187_187839

-- Definitions of conditions as Lean variables
variables
  (usual_arrival_time_home : ℕ)  -- The usual arrival time at home
  (usual_drive_time : ℕ) -- The usual drive time for the wife to reach the station
  (early_arrival_difference : ℕ := 16) -- They arrived home 16 minutes earlier
  (man_walk_time : ℕ := 52) -- The man walked for 52 minutes

-- The proof statement
theorem man_arrived_earlier
  (usual_arrival_time_home : ℕ)
  (usual_drive_time : ℕ)
  (H : usual_arrival_time_home - man_walk_time <= usual_drive_time - early_arrival_difference)
  : man_walk_time = 52 :=
sorry

end man_arrived_earlier_l187_187839


namespace decrypt_nbui_is_math_l187_187891

-- Define the sets A and B as the 26 English letters
def A := {c : Char | c ≥ 'a' ∧ c ≤ 'z'}
def B := A

-- Define the mapping f from A to B
def f (c : Char) : Char :=
  if c = 'z' then 'a'
  else Char.ofNat (c.toNat + 1)

-- Define the decryption function g (it reverses the mapping f)
def g (c : Char) : Char :=
  if c = 'a' then 'z'
  else Char.ofNat (c.toNat - 1)

-- Define the decryption of the given ciphertext
def decrypt (ciphertext : String) : String :=
  ciphertext.map g

-- Prove that the decryption of "nbui" is "math"
theorem decrypt_nbui_is_math : decrypt "nbui" = "math" :=
  by
  sorry

end decrypt_nbui_is_math_l187_187891


namespace largest_odd_digit_multiple_of_5_lt_10000_l187_187465

def is_odd_digit (n : ℕ) : Prop :=
  n = 1 ∨ n = 3 ∨ n = 5 ∨ n = 7 ∨ n = 9

def all_odd_digits (n : ℕ) : Prop :=
  ∀ d ∈ (n.digits 10), is_odd_digit d

def is_multiple_of_5 (n : ℕ) : Prop :=
  n % 5 = 0

theorem largest_odd_digit_multiple_of_5_lt_10000 :
  ∃ n, n < 10000 ∧ all_odd_digits n ∧ is_multiple_of_5 n ∧
        ∀ m, m < 10000 → all_odd_digits m → is_multiple_of_5 m → m ≤ n :=
  sorry

end largest_odd_digit_multiple_of_5_lt_10000_l187_187465


namespace remainder_21_l187_187942

theorem remainder_21 (y : ℤ) (k : ℤ) (h : y = 288 * k + 45) : y % 24 = 21 := 
  sorry

end remainder_21_l187_187942


namespace parabola_focus_l187_187853

theorem parabola_focus (p : ℝ) (h : 4 = 2 * p * 1^2) : (0, 1 / (4 * 2 * p)) = (0, 1 / 16) :=
by
  sorry

end parabola_focus_l187_187853


namespace basil_plants_yielded_l187_187160

def initial_investment (seed_cost soil_cost : ℕ) : ℕ :=
  seed_cost + soil_cost

def total_revenue (net_profit initial_investment : ℕ) : ℕ :=
  net_profit + initial_investment

def basil_plants (total_revenue price_per_plant : ℕ) : ℕ :=
  total_revenue / price_per_plant

theorem basil_plants_yielded
  (seed_cost soil_cost net_profit price_per_plant expected_plants : ℕ)
  (h_seed_cost : seed_cost = 2)
  (h_soil_cost : soil_cost = 8)
  (h_net_profit : net_profit = 90)
  (h_price_per_plant : price_per_plant = 5)
  (h_expected_plants : expected_plants = 20) :
  basil_plants (total_revenue net_profit (initial_investment seed_cost soil_cost)) price_per_plant = expected_plants :=
by
  -- Proof steps will be here
  sorry

end basil_plants_yielded_l187_187160


namespace max_value_of_fraction_l187_187758

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l187_187758


namespace sqrt_200_eq_10_l187_187276

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l187_187276


namespace find_n_l187_187720

-- Given Variables
variables (n x y : ℝ)

-- Given Conditions
axiom h1 : n * x = 6 * y
axiom h2 : x * y ≠ 0
axiom h3 : (1/3 * x) / (1/5 * y) = 1.9999999999999998

-- Conclusion
theorem find_n : n = 5 := sorry

end find_n_l187_187720


namespace monomial_sum_mn_l187_187877

theorem monomial_sum_mn (m n : ℤ) 
  (h1 : m + 6 = 1) 
  (h2 : 2 * n + 1 = 7) : 
  m * n = -15 := by
  sorry

end monomial_sum_mn_l187_187877


namespace trigonometric_identity_l187_187702

variable (α : Real)

theorem trigonometric_identity 
  (h : Real.sin (π / 6 + α) = Real.sqrt 3 / 3) :
  Real.cos (π / 3 - α) = Real.sqrt 3 / 3 :=
sorry

end trigonometric_identity_l187_187702


namespace exists_smaller_circle_with_at_least_as_many_lattice_points_l187_187523

theorem exists_smaller_circle_with_at_least_as_many_lattice_points
  (R : ℝ) (hR : 0 < R) :
  ∃ R' : ℝ, (R' < R) ∧ (∀ (x y : ℤ), x^2 + y^2 ≤ R^2 → ∃ (x' y' : ℤ), (x')^2 + (y')^2 ≤ (R')^2) := sorry

end exists_smaller_circle_with_at_least_as_many_lattice_points_l187_187523


namespace tangent_slope_l187_187052

noncomputable def f (x : ℝ) : ℝ := x - 1 + 1 / Real.exp x

noncomputable def f' (x : ℝ) : ℝ := 1 - 1 / Real.exp x

theorem tangent_slope (k : ℝ) (x₀ : ℝ) (y₀ : ℝ) 
  (h_tangent_point: (x₀ = -1) ∧ (y₀ = x₀ - 1 + 1 / Real.exp x₀))
  (h_tangent_line : ∀ x, y₀ = f x₀ + f' x₀ * (x - x₀)) :
  k = 1 - Real.exp 1 := 
sorry

end tangent_slope_l187_187052


namespace number_of_common_tangents_l187_187166

open Real EuclideanSpace

def Q1 : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 = 9}
def Q2 : set (ℝ × ℝ) := {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1}

theorem number_of_common_tangents :
  ∀ Q1 Q2 : set (ℝ × ℝ),
  Q1 = {p | p.1^2 + p.2^2 = 9} →
  Q2 = {p | (p.1 - 3)^2 + (p.2 - 4)^2 = 1} →
  ∃ (n : ℕ), n = 4 := 
by
  intros
  sorry

end number_of_common_tangents_l187_187166


namespace fifteen_power_ab_l187_187750

theorem fifteen_power_ab (a b : ℤ) (R S : ℝ) 
  (hR : R = 3^a) 
  (hS : S = 5^b) : 
  15^(a * b) = R^b * S^a :=
by sorry

end fifteen_power_ab_l187_187750


namespace c_value_for_infinite_solutions_l187_187387

theorem c_value_for_infinite_solutions :
  ∀ (c : ℝ), (∀ (x : ℝ), 3 * (5 + c * x) = 15 * x + 15) ↔ c = 5 :=
by
  -- Proof
  sorry

end c_value_for_infinite_solutions_l187_187387


namespace haley_lives_gained_l187_187202

-- Define the given conditions
def initial_lives : ℕ := 14
def lives_lost : ℕ := 4
def total_lives_after_gain : ℕ := 46

-- Define the goal: How many lives did Haley gain in the next level?
theorem haley_lives_gained : (total_lives_after_gain = initial_lives - lives_lost + lives_gained) → lives_gained = 36 :=
by
  intro h
  sorry

end haley_lives_gained_l187_187202


namespace sqrt_200_simplified_l187_187271

-- Definitions based on conditions from part a)
def factorization : Nat := 2 ^ 3 * 5 ^ 2

lemma sqrt_property (a b : ℕ) : Real.sqrt (a^2 * b) = a * Real.sqrt b := sorry

-- The proof problem (only the statement, not the proof)
theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  have h1 : 200 = 2^3 * 5^2 := by rfl
  have h2 : Real.sqrt (200) = Real.sqrt (2^3 * 5^2) := by rw h1
  rw [←show 200 = factorization by rfl] at h2
  exact sorry

end sqrt_200_simplified_l187_187271


namespace molecular_weight_correct_l187_187928

-- Define the atomic weights of the elements.
def atomic_weight_C : ℝ := 12.01
def atomic_weight_H : ℝ := 1.008
def atomic_weight_O : ℝ := 16.00

-- Define the number of atoms for each element in the compound.
def number_of_C : ℕ := 7
def number_of_H : ℕ := 6
def number_of_O : ℕ := 2

-- Define the molecular weight calculation.
def molecular_weight : ℝ := 
  (number_of_C * atomic_weight_C) +
  (number_of_H * atomic_weight_H) +
  (number_of_O * atomic_weight_O)

-- Step to prove that molecular weight is equal to 122.118 g/mol.
theorem molecular_weight_correct : molecular_weight = 122.118 := by
  sorry

end molecular_weight_correct_l187_187928


namespace find_number_l187_187787

theorem find_number (f : ℝ → ℝ) (x : ℝ)
  (h : f (x * 0.004) / 0.03 = 9.237333333333334)
  (h_linear : ∀ a, f a = a) :
  x = 69.3 :=
by
  -- Proof goes here
  sorry

end find_number_l187_187787


namespace complete_square_form_l187_187422

theorem complete_square_form {a h k : ℝ} :
  ∀ x, (x^2 - 5 * x) = a * (x - h)^2 + k → k = -25 / 4 :=
by
  intro x
  intro h_eq
  sorry

end complete_square_form_l187_187422


namespace calories_left_for_dinner_l187_187502

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l187_187502


namespace average_income_QR_l187_187290

theorem average_income_QR 
  (P Q R : ℝ)
  (h1 : (P + Q) / 2 = 2050)
  (h2 : (P + R) / 2 = 6200)
  (h3 : P = 3000) :
  (Q + R) / 2 = 5250 :=
  sorry

end average_income_QR_l187_187290


namespace cube_of_square_is_15625_l187_187303

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l187_187303


namespace cube_of_square_of_third_smallest_prime_is_correct_l187_187308

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l187_187308


namespace range_of_a_l187_187568

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) :
  (∀ x, f (2^x) = x^2 - 2 * a * x + a^2 - 1) →
  (∀ x, 2^(a-1) ≤ x ∧ x ≤ 2^(a^2 - 2*a + 2) → -1 ≤ f x ∧ f x ≤ 0) →
  ((3 - Real.sqrt 5) / 2 ≤ a ∧ a ≤ 1) ∨ (2 ≤ a ∧ a ≤ (3 + Real.sqrt 5) / 2) :=
by
  sorry

end range_of_a_l187_187568


namespace intersection_of_sets_l187_187858

open Set

theorem intersection_of_sets : 
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  M ∩ N = {0, 4, 8} := 
by
  let M : Set ℕ := {0, 2, 4, 8}
  let N : Set ℕ := { x | ∃ a, a ∈ M ∧ x = 2 * a }
  sorry

end intersection_of_sets_l187_187858


namespace max_value_of_fraction_l187_187763

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l187_187763


namespace sum_of_three_consecutive_even_l187_187469

theorem sum_of_three_consecutive_even (a1 a2 a3 : ℤ) (h1 : a1 % 2 = 0) (h2 : a2 = a1 + 2) (h3 : a3 = a1 + 4) (h4 : a1 + a3 = 128) : a1 + a2 + a3 = 192 :=
sorry

end sum_of_three_consecutive_even_l187_187469


namespace num_positive_integers_le_500_l187_187515

-- Define a predicate to state that a number is a perfect square
def is_square (x : ℕ) : Prop := ∃ (k : ℕ), k * k = x

-- Define the main theorem
theorem num_positive_integers_le_500 (n : ℕ) :
  (∃ (ns : Finset ℕ), (∀ x ∈ ns, x ≤ 500 ∧ is_square (21 * x)) ∧ ns.card = 4) :=
by
  sorry

end num_positive_integers_le_500_l187_187515


namespace Bons_wins_probability_l187_187911

theorem Bons_wins_probability :
  let p : ℚ := 5 / 11 in
  (∀ n : ℕ, n % 2 = 0 → valid_roll_sequence n → total_wins n = true) →
  (∀ k : ℕ, k % 2 = 1 → total_wins k = false) →
  (prob_roll_six = 1 / 6) →
  (p = 5 / 11) :=
begin
  sorry
end

end Bons_wins_probability_l187_187911


namespace calories_remaining_for_dinner_l187_187500

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l187_187500


namespace sum_50_to_75_l187_187930

-- Conditionally sum the series from 50 to 75
def sum_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

theorem sum_50_to_75 : sum_integers 50 75 = 1625 :=
by
  sorry

end sum_50_to_75_l187_187930


namespace find_gear_p_rpm_l187_187977

def gear_p_rpm (r : ℕ) (gear_p_revs : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) : Prop :=
  r = gear_p_revs * 2

theorem find_gear_p_rpm (r : ℕ) (gear_q_rpm : ℕ) (time_seconds : ℕ) (extra_revs_q_over_p : ℕ) :
  gear_q_rpm = 40 ∧ time_seconds = 30 ∧ extra_revs_q_over_p = 15 ∧ gear_p_revs = 10 / 2 →
  r = 10 :=
by
  sorry

end find_gear_p_rpm_l187_187977


namespace farmer_land_owned_l187_187332

def total_land (farmer_land : ℝ) (cleared_land : ℝ) : Prop :=
  cleared_land = 0.9 * farmer_land

def cleared_with_tomato (cleared_land : ℝ) (tomato_land : ℝ) : Prop :=
  tomato_land = 0.1 * cleared_land
  
def tomato_land_given (tomato_land : ℝ) : Prop :=
  tomato_land = 90

theorem farmer_land_owned (T : ℝ) :
  (∃ cleared : ℝ, total_land T cleared ∧ cleared_with_tomato cleared 90) → T = 1000 :=
by
  sorry

end farmer_land_owned_l187_187332


namespace sum_of_first_ten_nice_numbers_is_182_l187_187010

def is_proper_divisor (n d : ℕ) : Prop :=
  d > 1 ∧ d < n ∧ n % d = 0

def is_nice (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, is_proper_divisor n m → ∃ p q, n = p * q ∧ p ≠ q

def first_ten_nice_numbers : List ℕ := [6, 8, 10, 14, 15, 21, 22, 26, 27, 33]

def sum_first_ten_nice_numbers : ℕ := first_ten_nice_numbers.sum

theorem sum_of_first_ten_nice_numbers_is_182 :
  sum_first_ten_nice_numbers = 182 :=
by
  sorry

end sum_of_first_ten_nice_numbers_is_182_l187_187010


namespace total_handshakes_l187_187973

-- There are 5 members on each of the two basketball teams.
def teamMembers : Nat := 5

-- There are 2 referees.
def referees : Nat := 2

-- Each player from one team shakes hands with each player from the other team.
def handshakesBetweenTeams : Nat := teamMembers * teamMembers

-- Each player shakes hands with each referee.
def totalPlayers : Nat := 2 * teamMembers
def handshakesWithReferees : Nat := totalPlayers * referees

-- Prove that the total number of handshakes is 45.
theorem total_handshakes : handshakesBetweenTeams + handshakesWithReferees = 45 := by
  -- Total handshakes is the sum of handshakes between teams and handshakes with referees.
  sorry

end total_handshakes_l187_187973


namespace ramon_3_enchiladas_4_tacos_cost_l187_187743

theorem ramon_3_enchiladas_4_tacos_cost :
  ∃ (e t : ℝ), 2 * e + 3 * t = 2.50 ∧ 3 * e + 2 * t = 2.70 ∧ 3 * e + 4 * t = 3.54 :=
by {
  sorry
}

end ramon_3_enchiladas_4_tacos_cost_l187_187743


namespace incorrect_statement_l187_187809

def consecutive_interior_angles_are_supplementary (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 180 → l1 = l2

def alternate_interior_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def corresponding_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def complementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 90

def supplementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 180

theorem incorrect_statement :
  ¬ (∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2) →
    consecutive_interior_angles_are_supplementary l1 l2 →
    alternate_interior_angles_are_equal l1 l2 →
    corresponding_angles_are_equal l1 l2 →
    (∀ (θ₁ θ₂ : ℝ), supplementary_angles θ₁ θ₂) →
    (∀ (θ₁ θ₂ : ℝ), complementary_angles θ₁ θ₂) :=
sorry

end incorrect_statement_l187_187809


namespace calories_left_for_dinner_l187_187503

def breakfast_calories : ℝ := 353
def lunch_calories : ℝ := 885
def snack_calories : ℝ := 130
def daily_limit : ℝ := 2200

def total_consumed : ℝ :=
  breakfast_calories + lunch_calories + snack_calories

theorem calories_left_for_dinner : daily_limit - total_consumed = 832 :=
by
  sorry

end calories_left_for_dinner_l187_187503


namespace parabola_expression_l187_187420

theorem parabola_expression (a c : ℝ) (h1 : a = 1/4 ∨ a = -1/4) (h2 : ∀ x : ℝ, x = 1 → (a * x^2 + c = 0)) :
  (a = 1/4 ∧ c = -1/4) ∨ (a = -1/4 ∧ c = 1/4) :=
by {
  sorry
}

end parabola_expression_l187_187420


namespace find_probability_l187_187223

noncomputable def X : ℝ → ℝ := sorry -- Define your random variable X

variables (μ σ : ℝ) (h1 : P(X < 1) = 1 / 2) (h2 : P(X > 2) = 1 / 5)

theorem find_probability (X : ℝ → ℝ) : P(0 < X < 1) = 0.3 :=
by sorry

end find_probability_l187_187223


namespace range_of_f1_3_l187_187222

noncomputable def f (a b : ℝ) (x y : ℝ) : ℝ :=
  a * (x^3 + 3 * x) + b * (y^2 + 2 * y + 1)

theorem range_of_f1_3 (a b : ℝ)
  (h1 : 1 ≤ f a b 1 2 ∧ f a b 1 2 ≤ 2)
  (h2 : 2 ≤ f a b 3 4 ∧ f a b 3 4 ≤ 5):
  3 / 2 ≤ f a b 1 3 ∧ f a b 1 3 ≤ 4 :=
sorry

end range_of_f1_3_l187_187222


namespace expression_eval_l187_187815

theorem expression_eval :
  -14 - (-2) ^ 3 * (1 / 4) - 16 * (1 / 2 - 1 / 4 + 3 / 8) = -22 := by
  sorry

end expression_eval_l187_187815


namespace probability_of_selection_l187_187180

noncomputable def probability_selected (total_students : ℕ) (excluded_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / (total_students - excluded_students)

theorem probability_of_selection :
  probability_selected 2008 8 50 = 25 / 1004 :=
by
  sorry

end probability_of_selection_l187_187180


namespace green_chips_count_l187_187795

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end green_chips_count_l187_187795


namespace smallest_n_area_gt_2500_l187_187162

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1/2 : ℝ) * (|(n : ℝ) * (2 * n) + (n^2 - 1 : ℝ) * (3 * n^2 - 1) + (n^3 - 3 * n) * 1
  - (1 : ℝ) * (n^2 - 1) - (2 * n) * (n^3 - 3 * n) - (3 * n^2 - 1) * (n : ℝ)|)

theorem smallest_n_area_gt_2500 : ∃ n : ℕ, (∀ m : ℕ, 0 < m ∧ m < n → triangle_area m <= 2500) ∧ triangle_area n > 2500 :=
by
  sorry

end smallest_n_area_gt_2500_l187_187162


namespace hiking_supplies_l187_187371

theorem hiking_supplies (hours_per_day : ℕ) (days : ℕ) (rate_mph : ℝ) 
    (supply_per_mile : ℝ) (resupply_rate : ℝ)
    (initial_pack_weight : ℝ) : 
    hours_per_day = 8 → days = 5 → rate_mph = 2.5 → 
    supply_per_mile = 0.5 → resupply_rate = 0.25 → 
    initial_pack_weight = (40 : ℝ) :=
by
  intros hpd hd rm spm rr
  sorry

end hiking_supplies_l187_187371


namespace coin_toss_fairness_l187_187410

-- Statement of the problem as a Lean theorem.
theorem coin_toss_fairness (P_Heads P_Tails : ℝ) (h1 : P_Heads = 0.5) (h2 : P_Tails = 0.5) : 
  P_Heads = P_Tails ∧ P_Heads = 0.5 := 
sorry

end coin_toss_fairness_l187_187410


namespace folded_segment_square_length_eq_225_div_4_l187_187483

noncomputable def square_of_fold_length : ℝ :=
  let side_length := 15
  let distance_from_B := 5
  (side_length ^ 2 - distance_from_B * (2 * side_length - distance_from_B)) / 4

theorem folded_segment_square_length_eq_225_div_4 :
  square_of_fold_length = 225 / 4 :=
by
  sorry

end folded_segment_square_length_eq_225_div_4_l187_187483


namespace additional_miles_proof_l187_187041

-- Define the distances
def distance_to_bakery : ℕ := 9
def distance_bakery_to_grandma : ℕ := 24
def distance_grandma_to_apartment : ℕ := 27

-- Define the total distances
def total_distance_with_bakery : ℕ := distance_to_bakery + distance_bakery_to_grandma + distance_grandma_to_apartment
def total_distance_without_bakery : ℕ := 2 * distance_grandma_to_apartment

-- Define the additional miles
def additional_miles_with_bakery : ℕ := total_distance_with_bakery - total_distance_without_bakery

-- Theorem statement
theorem additional_miles_proof : additional_miles_with_bakery = 6 :=
by {
  -- Here should be the proof, but we insert sorry to indicate it's skipped
  sorry
}

end additional_miles_proof_l187_187041


namespace min_rubles_for_1001_l187_187299

def min_rubles_needed (n : ℕ) : ℕ :=
  let side_cells := (n + 1) * 4
  let inner_cells := (n - 1) * (n - 1)
  let total := inner_cells * 4 + side_cells
  total / 2 -- since each side is shared by two cells

theorem min_rubles_for_1001 : min_rubles_needed 1001 = 503000 := by
  sorry

end min_rubles_for_1001_l187_187299


namespace evaluate_expression_l187_187509

theorem evaluate_expression : 
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  a * b = 63 := 
by
  let a := 3 * 5 * 6
  let b := 1 / 3 + 1 / 5 + 1 / 6
  sorry

end evaluate_expression_l187_187509


namespace exponential_function_passes_through_fixed_point_l187_187044

theorem exponential_function_passes_through_fixed_point {a : ℝ} (ha_pos : a > 0) (ha_ne_one : a ≠ 1) : 
  (a^(2 - 2) + 3) = 4 :=
by
  sorry

end exponential_function_passes_through_fixed_point_l187_187044


namespace tan_alpha_sub_beta_l187_187192

theorem tan_alpha_sub_beta
  (α β : ℝ)
  (h1 : Real.tan (α + Real.pi / 5) = 2)
  (h2 : Real.tan (β - 4 * Real.pi / 5) = -3) :
  Real.tan (α - β) = -1 := 
sorry

end tan_alpha_sub_beta_l187_187192


namespace hiker_speed_calculation_l187_187949

theorem hiker_speed_calculation :
  ∃ (h_speed : ℝ),
    let c_speed := 10
    let c_time := 5.0 / 60.0
    let c_wait := 7.5 / 60.0
    let c_distance := c_speed * c_time
    let h_distance := c_distance
    h_distance = h_speed * c_wait ∧ h_speed = 10 * (5 / 7.5) := by
  sorry

end hiker_speed_calculation_l187_187949


namespace problem_equation_false_l187_187431

theorem problem_equation_false (K T U Ch O H : ℕ) 
  (h1 : K ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h2 : T ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h3 : U ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h4 : Ch ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h5 : O ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  (h6 : H ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9})
  (distinct : ∀ x ∈ {K, T, U, Ch, O, H}, ∀ y ∈ {K, T, U, Ch, O, H}, x ≠ y → x ≠ y) :
  (K * 0 * T = U * Ch * O * H * H * U) → False :=
by
  sorry

end problem_equation_false_l187_187431


namespace real_to_fraction_l187_187636

noncomputable def real_num : ℚ := 3.675

theorem real_to_fraction : real_num = 147 / 40 :=
by
  -- convert 3.675 to a mixed number
  have h1 : real_num = 3 + 675 / 1000 := by sorry
  -- find gcd of 675 and 1000
  have h2 : Nat.gcd 675 1000 = 25 := by sorry
  -- simplify 675/1000 to 27/40
  have h3 : 675 / 1000 = 27 / 40 := by sorry
  -- convert mixed number to improper fraction 147/40
  have h4 : 3 + 27 / 40 = 147 / 40 := by sorry
  -- combine the results to prove the required equality
  exact sorry

end real_to_fraction_l187_187636


namespace sqrt_200_eq_10_l187_187263

theorem sqrt_200_eq_10 : real.sqrt 200 = 10 :=
by
  calc
    real.sqrt 200 = real.sqrt (2^2 * 5^2) : by sorry -- show 200 = 2^2 * 5^2
    ... = real.sqrt (2^2) * real.sqrt (5^2) : by sorry -- property of square roots of products
    ... = 2 * 5 : by sorry -- using the property sqrt (a^2) = a
    ... = 10 : by sorry

end sqrt_200_eq_10_l187_187263


namespace number_of_girls_l187_187617

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l187_187617


namespace avg_of_9_numbers_l187_187913

theorem avg_of_9_numbers (a b c d e f g h i : ℕ)
  (h1 : (a + b + c + d + e) / 5 = 99)
  (h2 : (e + f + g + h + i) / 5 = 100)
  (h3 : e = 59) : 
  (a + b + c + d + e + f + g + h + i) / 9 = 104 := 
sorry

end avg_of_9_numbers_l187_187913


namespace frequency_of_group_l187_187178

-- Definitions based on conditions in the problem
def sampleCapacity : ℕ := 32
def frequencyRate : ℝ := 0.25

-- Lean statement representing the proof
theorem frequency_of_group : (frequencyRate * sampleCapacity : ℝ) = 8 := 
by 
  sorry -- Proof placeholder

end frequency_of_group_l187_187178


namespace complement_A_is_interval_l187_187711

def U : Set ℝ := {x | True}
def A : Set ℝ := {x | x^2 - 2 * x - 3 > 0}
def compl_U_A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem complement_A_is_interval : compl_U_A = {x | -1 ≤ x ∧ x ≤ 3} := by
  sorry

end complement_A_is_interval_l187_187711


namespace miles_to_mall_l187_187573

noncomputable def miles_to_grocery_store : ℕ := 10
noncomputable def miles_to_pet_store : ℕ := 5
noncomputable def miles_back_home : ℕ := 9
noncomputable def miles_per_gallon : ℕ := 15
noncomputable def cost_per_gallon : ℝ := 3.50
noncomputable def total_cost_of_gas : ℝ := 7.00
noncomputable def total_miles_driven := 2 * miles_per_gallon

theorem miles_to_mall : total_miles_driven -
  (miles_to_grocery_store + miles_to_pet_store + miles_back_home) = 6 :=
by
  -- proof omitted 
  sorry

end miles_to_mall_l187_187573


namespace value_before_decrease_l187_187578

theorem value_before_decrease
  (current_value decrease : ℤ)
  (current_value_equals : current_value = 1460)
  (decrease_equals : decrease = 12) :
  current_value + decrease = 1472 :=
by
  -- We assume the proof to follow here.
  sorry

end value_before_decrease_l187_187578


namespace toilet_paper_duration_l187_187974

theorem toilet_paper_duration :
  let bill_weekday := 3 * 5
  let wife_weekday := 4 * 8
  let kid_weekday := 5 * 6
  let total_weekday := bill_weekday + wife_weekday + 2 * kid_weekday
  let bill_weekend := 4 * 6
  let wife_weekend := 5 * 10
  let kid_weekend := 6 * 5
  let total_weekend := bill_weekend + wife_weekend + 2 * kid_weekend
  let total_week := 5 * total_weekday + 2 * total_weekend
  let total_squares := 1000 * 300
  let weeks_last := total_squares / total_week
  let days_last := weeks_last * 7
  days_last = 2615 :=
sorry

end toilet_paper_duration_l187_187974


namespace find_p_l187_187218

theorem find_p (f p : ℂ) (w : ℂ) (h1 : f * p - w = 15000) (h2 : f = 8) (h3 : w = 10 + 200 * Complex.I) : 
  p = 1876.25 + 25 * Complex.I := 
sorry

end find_p_l187_187218


namespace probability_club_then_spade_l187_187120

/--
   Two cards are dealt at random from a standard deck of 52 cards.
   Prove that the probability that the first card is a club (♣) and the second card is a spade (♠) is 13/204.
-/
theorem probability_club_then_spade :
  let total_cards := 52
  let clubs := 13
  let spades := 13
  let first_card_club_prob := (clubs : ℚ) / total_cards
  let second_card_spade_prob := (spades : ℚ) / (total_cards - 1)
  first_card_club_prob * second_card_spade_prob = 13 / 204 :=
by
  sorry

end probability_club_then_spade_l187_187120


namespace arithmetic_seq_a2_a8_a5_l187_187979

-- Define the sequence and sum conditions
variable {a : ℕ → ℝ} {S : ℕ → ℝ} {q : ℝ}

-- Define the given conditions
axiom seq_condition (n : ℕ) : (1 - q) * S n + q * a n = 1
axiom q_nonzero : q * (q - 1) ≠ 0
axiom geom_seq : ∀ n, a n = q^(n - 1)

-- Main theorem (consistent with both parts (Ⅰ) and (Ⅱ) results)
theorem arithmetic_seq_a2_a8_a5 (S_arith : S 3 + S 6 = 2 * S 9) : a 2 + a 5 = 2 * a 8 :=
by
    sorry

end arithmetic_seq_a2_a8_a5_l187_187979


namespace cube_surface_area_l187_187793

-- Define the volume condition
def volume (s : ℕ) : ℕ := s^3

-- Define the surface area function
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- State the theorem to be proven
theorem cube_surface_area (s : ℕ) (h : volume s = 729) : surface_area s = 486 :=
by
  sorry

end cube_surface_area_l187_187793


namespace medicine_dosage_l187_187361

theorem medicine_dosage (weight_kg dose_per_kg parts : ℕ) (h_weight : weight_kg = 30) (h_dose_per_kg : dose_per_kg = 5) (h_parts : parts = 3) :
  ((weight_kg * dose_per_kg) / parts) = 50 :=
by sorry

end medicine_dosage_l187_187361


namespace profit_is_1500_l187_187927

def cost_per_charm : ℕ := 15
def charms_per_necklace : ℕ := 10
def sell_price_per_necklace : ℕ := 200
def necklaces_sold : ℕ := 30

def cost_per_necklace : ℕ := cost_per_charm * charms_per_necklace
def profit_per_necklace : ℕ := sell_price_per_necklace - cost_per_necklace
def total_profit : ℕ := profit_per_necklace * necklaces_sold

theorem profit_is_1500 : total_profit = 1500 :=
by
  sorry

end profit_is_1500_l187_187927


namespace cube_of_square_of_third_smallest_prime_is_correct_l187_187307

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l187_187307


namespace claire_photos_l187_187444

variable (C : ℕ) -- Claire's photos
variable (L : ℕ) -- Lisa's photos
variable (R : ℕ) -- Robert's photos

-- Conditions
axiom Lisa_photos : L = 3 * C
axiom Robert_photos : R = C + 16
axiom Lisa_Robert_same : L = R

-- Proof Goal
theorem claire_photos : C = 8 :=
by
  -- Sorry skips the proof and allows the theorem to compile
  sorry

end claire_photos_l187_187444


namespace distance_from_origin_to_line_l187_187455

theorem distance_from_origin_to_line : 
  let a := 1
  let b := 2
  let c := -5
  let x0 := 0
  let y0 := 0
  let distance := (|a * x0 + b * y0 + c|) / (Real.sqrt (a^2 + b^2))
  distance = Real.sqrt 5 :=
by
  sorry

end distance_from_origin_to_line_l187_187455


namespace one_eighth_of_2_pow_33_eq_2_pow_x_l187_187874

theorem one_eighth_of_2_pow_33_eq_2_pow_x (x : ℕ) : (1 / 8) * (2 : ℝ) ^ 33 = (2 : ℝ) ^ x → x = 30 := by
  intro h
  sorry

end one_eighth_of_2_pow_33_eq_2_pow_x_l187_187874


namespace find_radius_l187_187677

def radius_of_circle (d : ℤ) (PQ : ℕ) (QR : ℕ) (r : ℕ) : Prop := 
  let PR := PQ + QR
  (PQ * PR = (d - r) * (d + r)) ∧ (d = 15) ∧ (PQ = 11) ∧ (QR = 8) ∧ (r = 4)

-- Now stating the theorem to prove the radius r given the conditions
theorem find_radius (r : ℕ) : radius_of_circle 15 11 8 r := by
  sorry

end find_radius_l187_187677


namespace total_messages_equation_l187_187067

theorem total_messages_equation (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  exact h

end total_messages_equation_l187_187067


namespace value_of_a_minus_b_l187_187859

theorem value_of_a_minus_b (a b : ℝ) (h₁ : |a| = 2) (h₂ : |b| = 5) (h₃ : a < b) :
  a - b = -3 ∨ a - b = -7 := 
sorry

end value_of_a_minus_b_l187_187859


namespace find_a_l187_187990

noncomputable def star (a b : ℝ) := a * (a + b) + b

theorem find_a (a : ℝ) (h : star a 2.5 = 28.5) : a = 4 ∨ a = -13/2 := 
sorry

end find_a_l187_187990


namespace three_people_same_topic_l187_187583

open Classical

theorem three_people_same_topic (people : Fin 17 → Type) (topics : Fin 3 → Type) 
  (corresponds : (Fin 17 → Fin 17 → Fin 3) → Prop) :
    ∃ (a b c : Fin 17), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
      (∃ (t : Fin 3), corresponds (λ x y, t) a b ∧ corresponds (λ x y, t) b c ∧ corresponds (λ x y, t) c a) :=
  sorry

end three_people_same_topic_l187_187583


namespace second_supplier_more_cars_l187_187006

-- Define the constants and conditions given in the problem
def total_production := 5650000
def first_supplier := 1000000
def fourth_fifth_supplier := 325000

-- Define the unknown variable for the second supplier
noncomputable def second_supplier : ℕ := sorry

-- Define the equation based on the conditions
def equation := first_supplier + second_supplier + (first_supplier + second_supplier) + (4 * fourth_fifth_supplier / 2) = total_production

-- Prove that the second supplier receives 500,000 more cars than the first supplier
theorem second_supplier_more_cars : 
  ∃ X : ℕ, equation → (X = first_supplier + 500000) :=
sorry

end second_supplier_more_cars_l187_187006


namespace possible_measures_A_l187_187599

open Nat

theorem possible_measures_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k > 0 ∧ A = k * B) ∧ A + B = 180 ∧ (∃! n : ℕ, n = 17) :=
by
  sorry

end possible_measures_A_l187_187599


namespace apples_in_each_basket_after_sister_took_apples_l187_187080

theorem apples_in_each_basket_after_sister_took_apples 
  (total_apples : ℕ) 
  (number_of_baskets : ℕ) 
  (apples_taken_from_each : ℕ)
  (initial_apples_per_basket := total_apples / number_of_baskets)
  (final_apples_per_basket := initial_apples_per_basket - apples_taken_from_each) :
  total_apples = 64 → number_of_baskets = 4 → apples_taken_from_each = 3 → final_apples_per_basket = 13 := 
by 
  intros htotal hnumber htake
  rw [htotal, hnumber, htake]
  have initial_apples : initial_apples_per_basket = 16 := by norm_num
  rw initial_apples
  norm_num
  sorry

end apples_in_each_basket_after_sister_took_apples_l187_187080


namespace base5_addition_l187_187962

theorem base5_addition : 
  (14 : ℕ) + (132 : ℕ) = (101 : ℕ) :=
by {
  sorry
}

end base5_addition_l187_187962


namespace total_turtles_taken_l187_187971

theorem total_turtles_taken (number_of_green_turtles number_of_hawksbill_turtles total_number_of_turtles : ℕ)
  (h1 : number_of_green_turtles = 800)
  (h2 : number_of_hawksbill_turtles = 2 * number_of_green_turtles)
  (h3 : total_number_of_turtles = number_of_green_turtles + number_of_hawksbill_turtles) :
  total_number_of_turtles = 2400 :=
by
  sorry

end total_turtles_taken_l187_187971


namespace soda_cost_is_2_l187_187686

noncomputable def cost_per_soda (total_bill : ℕ) (num_adults : ℕ) (num_children : ℕ) 
  (adult_meal_cost : ℕ) (child_meal_cost : ℕ) (num_sodas : ℕ) : ℕ :=
  (total_bill - (num_adults * adult_meal_cost + num_children * child_meal_cost)) / num_sodas

theorem soda_cost_is_2 :
  let total_bill := 60
  let num_adults := 6
  let num_children := 2
  let adult_meal_cost := 6
  let child_meal_cost := 4
  let num_sodas := num_adults + num_children
  cost_per_soda total_bill num_adults num_children adult_meal_cost child_meal_cost num_sodas = 2 :=
by
  -- proof goes here
  sorry

end soda_cost_is_2_l187_187686


namespace marcus_calzones_total_time_l187_187226

theorem marcus_calzones_total_time :
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  total_time = 124 :=
by
  let saute_onions_time := 20
  let saute_garlic_peppers_time := (1 / 4 : ℚ) * saute_onions_time
  let knead_time := 30
  let rest_time := 2 * knead_time
  let assemble_time := (1 / 10 : ℚ) * (knead_time + rest_time)
  let total_time := saute_onions_time + saute_garlic_peppers_time + knead_time + rest_time + assemble_time
  sorry

end marcus_calzones_total_time_l187_187226


namespace min_vertical_segment_length_l187_187386

noncomputable def f₁ (x : ℝ) : ℝ := |x|
noncomputable def f₂ (x : ℝ) : ℝ := -x^2 - 4 * x - 3

theorem min_vertical_segment_length :
  ∃ m : ℝ, m = 3 ∧
            ∀ x : ℝ, abs (f₁ x - f₂ x) ≥ m :=
sorry

end min_vertical_segment_length_l187_187386


namespace geometric_sum_eqn_l187_187995

theorem geometric_sum_eqn 
  (a1 q : ℝ) 
  (hne1 : q ≠ 1) 
  (hS2 : a1 * (1 - q^2) / (1 - q) = 1) 
  (hS4 : a1 * (1 - q^4) / (1 - q) = 3) :
  a1 * (1 - q^8) / (1 - q) = 15 :=
by
  sorry

end geometric_sum_eqn_l187_187995


namespace maximum_obtuse_vectors_l187_187639

-- Definition: A vector in 3D space
structure Vector3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

-- Definition: Dot product of two vectors
def dot_product (v1 v2 : Vector3D) : ℝ :=
  v1.x * v2.x + v1.y * v2.y + v1.z * v2.z

-- Condition: Two vectors form an obtuse angle if their dot product is negative
def obtuse_angle (v1 v2 : Vector3D) : Prop :=
  dot_product v1 v2 < 0

-- Main statement incorporating the conditions and the conclusion
theorem maximum_obtuse_vectors :
  ∀ (v1 v2 v3 v4 : Vector3D),
  (obtuse_angle v1 v2) →
  (obtuse_angle v1 v3) →
  (obtuse_angle v1 v4) →
  (obtuse_angle v2 v3) →
  (obtuse_angle v2 v4) →
  (obtuse_angle v3 v4) →
  -- Conclusion: At most 4 vectors can be pairwise obtuse
  ∃ (v5 : Vector3D),
  ¬ (obtuse_angle v1 v5 ∧ obtuse_angle v2 v5 ∧ obtuse_angle v3 v5 ∧ obtuse_angle v4 v5) :=
sorry

end maximum_obtuse_vectors_l187_187639


namespace starting_number_l187_187101

theorem starting_number (n : ℕ) (h1 : n % 11 = 3) (h2 : (n + 11) % 11 = 3) (h3 : (n + 22) % 11 = 3) 
  (h4 : (n + 33) % 11 = 3) (h5 : (n + 44) % 11 = 3) (h6 : n + 44 ≤ 50) : n = 3 := 
sorry

end starting_number_l187_187101


namespace least_number_with_remainder_l187_187667

theorem least_number_with_remainder (x : ℕ) :
  (x % 6 = 4) ∧ (x % 7 = 4) ∧ (x % 9 = 4) ∧ (x % 18 = 4) ↔ x = 130 :=
by
  sorry

end least_number_with_remainder_l187_187667


namespace cakes_sold_l187_187492

/-- If a baker made 54 cakes and has 13 cakes left, then the number of cakes he sold is 41. -/
theorem cakes_sold (original_cakes : ℕ) (cakes_left : ℕ) 
  (h1 : original_cakes = 54) (h2 : cakes_left = 13) : 
  original_cakes - cakes_left = 41 := 
by 
  sorry

end cakes_sold_l187_187492


namespace tobys_friends_boys_count_l187_187673

theorem tobys_friends_boys_count (total_friends : ℕ) (girls : ℕ) (boys_percentage : ℕ) 
    (h1 : girls = 27) (h2 : boys_percentage = 55) (total_friends_calc : total_friends = 60) : 
    (total_friends * boys_percentage / 100) = 33 :=
by
  -- Proof is deferred
  sorry

end tobys_friends_boys_count_l187_187673


namespace football_tournament_l187_187424

theorem football_tournament (points: List ℕ) (n: ℕ) 
    (h₀ : points = [16, 14, 10, 10, 8, 6, 5, 3])
    (h₁ : (n - 1) * n = 72) 
    (h₂ : points.length = n - 1) 
    (h₃ : ∀x ∈ points, x ≤ 16) : 
    n = 9 ∧ 
    ((16 - points.head!) + 
     (16 - List.nthLe points 1 (by linarith)) + 
     (16 - List.nthLe points 2 (by linarith)) + 
     (16 - List.nthLe points 3 (by linarith))) = 14 :=
by
  sorry

end football_tournament_l187_187424


namespace work_completes_in_39_days_l187_187156

theorem work_completes_in_39_days 
  (amit_days : ℕ := 15)  -- Amit can complete work in 15 days
  (ananthu_days : ℕ := 45)  -- Ananthu can complete work in 45 days
  (amit_worked_days : ℕ := 3)  -- Amit worked for 3 days
  : (amit_worked_days + ((4 / 5) / (1 / ananthu_days))) = 39 :=
by
  sorry

end work_completes_in_39_days_l187_187156


namespace triangle_incenter_midpoint_eq_distance_l187_187553

theorem triangle_incenter_midpoint_eq_distance
    {A B C : Point}
    (hABC : right_triangle A B C)
    (angleBAC_30 : ∠BAC = 30)
    (S : Point)
    (hS : incenter S A B C)
    (D : Point)
    (hD : midpoint D A B) :
    distance C S = distance D S :=
by
  sorry

end triangle_incenter_midpoint_eq_distance_l187_187553


namespace train_takes_longer_l187_187959

-- Definitions for the conditions
def train_speed : ℝ := 48
def ship_speed : ℝ := 60
def distance : ℝ := 480

-- Theorem statement for the proof
theorem train_takes_longer : (distance / train_speed) - (distance / ship_speed) = 2 := by
  sorry

end train_takes_longer_l187_187959


namespace inequality_x2_gt_y2_plus_6_l187_187293

theorem inequality_x2_gt_y2_plus_6 (x y : ℝ) (h1 : x > y) (h2 : y > 3 / (x - y)) : x^2 > y^2 + 6 :=
sorry

end inequality_x2_gt_y2_plus_6_l187_187293


namespace find_roots_of_polynomial_l187_187029

def f (x : ℝ) := x^3 - 2*x^2 - 5*x + 6

theorem find_roots_of_polynomial :
  (f 1 = 0) ∧ (f (-2) = 0) ∧ (f 3 = 0) :=
by
  -- Proof will be written here
  sorry

end find_roots_of_polynomial_l187_187029


namespace inequality_condition_l187_187453

theorem inequality_condition {a b c : ℝ} :
  (∀ x : ℝ, a * Real.sin x + b * Real.cos x + c > 0) ↔ (Real.sqrt (a^2 + b^2) < c) :=
by
  sorry

end inequality_condition_l187_187453


namespace passes_to_left_l187_187146

theorem passes_to_left (total_passes right_passes center_passes left_passes : ℕ)
  (h_total : total_passes = 50)
  (h_right : right_passes = 2 * left_passes)
  (h_center : center_passes = left_passes + 2)
  (h_sum : left_passes + right_passes + center_passes = total_passes) :
  left_passes = 12 := 
by
  sorry

end passes_to_left_l187_187146


namespace cost_of_letter_is_0_37_l187_187384

-- Definitions based on the conditions
def total_cost : ℝ := 4.49
def package_cost : ℝ := 0.88
def num_letters : ℕ := 5
def num_packages : ℕ := 3
def letter_cost (L : ℝ) : ℝ := 5 * L
def package_total_cost : ℝ := num_packages * package_cost

-- Theorem that encapsulates the mathematical proof problem
theorem cost_of_letter_is_0_37 (L : ℝ) (h : letter_cost L + package_total_cost = total_cost) : L = 0.37 :=
by sorry

end cost_of_letter_is_0_37_l187_187384


namespace inscribed_circle_probability_l187_187196

theorem inscribed_circle_probability (r : ℝ) (h : r > 0) : 
  let square_area := 4 * r^2
  let circle_area := π * r^2
  (circle_area / square_area) = π / 4 := by
  sorry

end inscribed_circle_probability_l187_187196


namespace find_x_l187_187522

theorem find_x 
  (x : ℝ) 
  (h1 : 0 < x)
  (h2 : x < π / 2)
  (h3 : 1 / (Real.sin x) = 1 / (Real.sin (2 * x)) + 1 / (Real.sin (4 * x)) + 1 / (Real.sin (8 * x))) : 
  x = π / 15 ∨ x = π / 5 ∨ x = π / 3 ∨ x = 7 * π / 15 :=
by
  sorry

end find_x_l187_187522


namespace apples_in_each_basket_l187_187079

-- Definitions based on the conditions
def total_apples : ℕ := 64
def baskets : ℕ := 4
def apples_taken_per_basket : ℕ := 3

-- Theorem statement based on the question and correct answer
theorem apples_in_each_basket (h1 : total_apples = 64) 
                              (h2 : baskets = 4) 
                              (h3 : apples_taken_per_basket = 3) : 
    (total_apples / baskets - apples_taken_per_basket) = 13 := 
by
  sorry

end apples_in_each_basket_l187_187079


namespace find_vertex_angle_l187_187032

noncomputable def vertex_angle_cone (α : ℝ) : ℝ :=
  2 * Real.arcsin (α / (2 * Real.pi))

theorem find_vertex_angle (α : ℝ) :
  ∃ β : ℝ, β = vertex_angle_cone α :=
by
  use 2 * Real.arcsin (α / (2 * Real.pi))
  refl

end find_vertex_angle_l187_187032


namespace sum_of_last_two_digits_l187_187931

theorem sum_of_last_two_digits (a b : ℕ) (ha : a = 7) (hb : b = 13) : (a^15 + b^15) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l187_187931


namespace max_value_of_fraction_l187_187761

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l187_187761


namespace possible_measures_of_angle_A_l187_187596

theorem possible_measures_of_angle_A :
  ∃ (A B : ℕ), A > 0 ∧ B > 0 ∧ (∃ k : ℕ, k ≥ 1 ∧ A = k * B) ∧ (A + B = 180) ↔ (finset.card (finset.filter (λ d, d > 1) (finset.divisors 180))) = 17 :=
by
sorry

end possible_measures_of_angle_A_l187_187596


namespace determinant_zero_l187_187171

theorem determinant_zero (α β : ℝ) :
  Matrix.det ![
    ![0, Real.sin α, -Real.cos α],
    ![-Real.sin α, 0, Real.sin β],
    ![Real.cos α, -Real.sin β, 0]
  ] = 0 :=
by sorry

end determinant_zero_l187_187171


namespace ratio_of_x_and_y_l187_187811

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 1 / 4 :=
by
  sorry

end ratio_of_x_and_y_l187_187811


namespace cookies_on_third_plate_l187_187614

theorem cookies_on_third_plate :
  ∀ (a5 a7 a14 a19 a25 : ℕ),
  (a5 = 5) ∧ (a7 = 7) ∧ (a14 = 14) ∧ (a19 = 19) ∧ (a25 = 25) →
  ∃ (a12 : ℕ), a12 = 12 :=
by
  sorry

end cookies_on_third_plate_l187_187614


namespace length_of_AB_in_triangle_l187_187075

open Real

theorem length_of_AB_in_triangle
  (AC BC : ℝ)
  (area : ℝ) :
  AC = 4 →
  BC = 3 →
  area = 3 * sqrt 3 →
  ∃ AB : ℝ, AB = sqrt 13 :=
by
  sorry

end length_of_AB_in_triangle_l187_187075


namespace simplify_expression_l187_187246

-- Defining the variables involved
variables (b : ℝ)

-- The theorem statement that needs to be proven
theorem simplify_expression : 3 * b * (3 * b^2 - 2 * b + 1) + 2 * b^2 = 9 * b^3 - 4 * b^2 + 3 * b :=
by
  sorry

end simplify_expression_l187_187246


namespace train_speed_proof_l187_187666

theorem train_speed_proof :
  (∀ (speed : ℝ), 
    let train_length := 120
    let cross_time := 16
    let total_distance := 240
    let relative_speed := total_distance / cross_time
    let individual_speed := relative_speed / 2
    let speed_kmh := individual_speed * 3.6
    (speed_kmh = 27) → speed = 27
  ) :=
by
  sorry

end train_speed_proof_l187_187666


namespace inequality_sine_cosine_l187_187447

theorem inequality_sine_cosine (t : ℝ) (ht : t > 0) : 3 * Real.sin t < 2 * t + t * Real.cos t := 
sorry

end inequality_sine_cosine_l187_187447


namespace sqrt_200_eq_10_l187_187275

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l187_187275


namespace hypotenuse_min_length_l187_187992

theorem hypotenuse_min_length
  (a b l : ℝ)
  (h_area : (1/2) * a * b = 8)
  (h_perimeter : a + b + Real.sqrt (a^2 + b^2) = l)
  (h_min_l : l = 8 + 4 * Real.sqrt 2) :
  Real.sqrt (a^2 + b^2) = 4 * Real.sqrt 2 :=
by
  sorry

end hypotenuse_min_length_l187_187992


namespace necessary_but_not_sufficient_condition_l187_187351

variable (a : ℝ)

theorem necessary_but_not_sufficient_condition (h : 0 ≤ a ∧ a ≤ 4) :
  (∀ x : ℝ, x^2 + a * x + a > 0) → (0 ≤ a ∧ a ≤ 4 ∧ ¬ (∀ x : ℝ, x^2 + a * x + a > 0)) :=
sorry

end necessary_but_not_sufficient_condition_l187_187351


namespace max_number_soap_boxes_l187_187660

-- Definition of dimensions and volumes
def carton_length : ℕ := 25
def carton_width : ℕ := 42
def carton_height : ℕ := 60
def soap_box_length : ℕ := 7
def soap_box_width : ℕ := 12
def soap_box_height : ℕ := 5

def volume (l w h : ℕ) : ℕ := l * w * h

-- Volumes of the carton and soap box
def carton_volume : ℕ := volume carton_length carton_width carton_height
def soap_box_volume : ℕ := volume soap_box_length soap_box_width soap_box_height

-- The maximum number of soap boxes that can be placed in the carton
def max_soap_boxes : ℕ := carton_volume / soap_box_volume

theorem max_number_soap_boxes :
  max_soap_boxes = 150 :=
by
  -- Proof here
  sorry

end max_number_soap_boxes_l187_187660


namespace sum_series_eq_1_div_300_l187_187163

noncomputable def sum_series : ℝ :=
  ∑' n, (6 * (n:ℝ) + 1) / ((6 * (n:ℝ) - 1) ^ 2 * (6 * (n:ℝ) + 5) ^ 2)

theorem sum_series_eq_1_div_300 : sum_series = 1 / 300 :=
  sorry

end sum_series_eq_1_div_300_l187_187163


namespace passed_boys_avg_marks_l187_187881

theorem passed_boys_avg_marks (total_boys : ℕ) (avg_marks_all_boys : ℕ) (avg_marks_failed_boys : ℕ) (passed_boys : ℕ) 
  (h1 : total_boys = 120)
  (h2 : avg_marks_all_boys = 35)
  (h3 : avg_marks_failed_boys = 15)
  (h4 : passed_boys = 100) : 
  (39 = (35 * 120 - 15 * (total_boys - passed_boys)) / passed_boys) :=
  sorry

end passed_boys_avg_marks_l187_187881


namespace calories_remaining_for_dinner_l187_187499

theorem calories_remaining_for_dinner :
  let daily_limit := 2200
  let breakfast := 353
  let lunch := 885
  let snack := 130
  let total_consumed := breakfast + lunch + snack
  let remaining_for_dinner := daily_limit - total_consumed
  remaining_for_dinner = 832 := by
  sorry

end calories_remaining_for_dinner_l187_187499


namespace ratio_M_N_l187_187417

theorem ratio_M_N (M Q P R N : ℝ) 
(h1 : M = 0.40 * Q) 
(h2 : Q = 0.25 * P) 
(h3 : R = 0.60 * P) 
(h4 : N = 0.75 * R) : 
  M / N = 2 / 9 := 
by
  sorry

end ratio_M_N_l187_187417


namespace smallest_m_l187_187844

-- Defining the remainder function
def r (m n : ℕ) : ℕ := m % n

-- Main theorem stating the problem needed to be proved
theorem smallest_m (m : ℕ) (h : m > 0) 
  (H : (r m 1 + r m 2 + r m 3 + r m 4 + r m 5 + r m 6 + r m 7 + r m 8 + r m 9 + r m 10) = 4) : 
  m = 120 :=
sorry

end smallest_m_l187_187844


namespace slips_with_3_count_l187_187102

def number_of_slips_with_3 (x : ℕ) : Prop :=
  let total_slips := 15
  let expected_value := 4.6
  let prob_3 := (x : ℚ) / total_slips
  let prob_8 := (total_slips - x : ℚ) / total_slips
  let E := prob_3 * 3 + prob_8 * 8
  E = expected_value

theorem slips_with_3_count : ∃ x : ℕ, number_of_slips_with_3 x ∧ x = 10 :=
by
  sorry

end slips_with_3_count_l187_187102


namespace shelves_filled_l187_187288

theorem shelves_filled (carvings_per_shelf : ℕ) (total_carvings : ℕ) (h₁ : carvings_per_shelf = 8) (h₂ : total_carvings = 56) :
  total_carvings / carvings_per_shelf = 7 := by
  sorry

end shelves_filled_l187_187288


namespace number_of_jerseys_bought_l187_187433

-- Define the given constants
def initial_money : ℕ := 50
def cost_per_jersey : ℕ := 2
def cost_basketball : ℕ := 18
def cost_shorts : ℕ := 8
def money_left : ℕ := 14

-- Define the theorem to prove the number of jerseys Jeremy bought.
theorem number_of_jerseys_bought :
  (initial_money - money_left) = (cost_basketball + cost_shorts + 5 * cost_per_jersey) :=
by
  sorry

end number_of_jerseys_bought_l187_187433


namespace cube_square_third_smallest_prime_l187_187317

theorem cube_square_third_smallest_prime :
  let p := 5 in -- the third smallest prime number
  (p^2)^3 = 15625 :=
by
  let p := 5
  sorry

end cube_square_third_smallest_prime_l187_187317


namespace cube_of_square_is_15625_l187_187304

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l187_187304


namespace valid_grid_count_l187_187017

def is_adjacent (i j : ℕ) (n : ℕ) : Prop :=
  (i = j + 1 ∨ i + 1 = j ∨ (i = n - 1 ∧ j = 0) ∨ (i = 0 ∧ j = n - 1))

def valid_grid (grid : ℕ → ℕ → ℕ) : Prop :=
  ∀ i j, 0 ≤ i ∧ i < 4 ∧ 0 ≤ j ∧ j < 4 →
         (is_adjacent i (i+1) 4 → grid i (i+1) * grid i (i+1) = 0) ∧ 
         (is_adjacent j (j+1) 4 → grid (j+1) j * grid (j+1) j = 0)

theorem valid_grid_count : 
  ∃ s : ℕ, s = 1234 ∧
    (∃ grid : ℕ → ℕ → ℕ, valid_grid grid) :=
sorry

end valid_grid_count_l187_187017


namespace least_positive_integer_to_add_l187_187324

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l187_187324


namespace sqrt_200_simplified_l187_187267

-- Definitions based on conditions from part a)
def factorization : Nat := 2 ^ 3 * 5 ^ 2

lemma sqrt_property (a b : ℕ) : Real.sqrt (a^2 * b) = a * Real.sqrt b := sorry

-- The proof problem (only the statement, not the proof)
theorem sqrt_200_simplified : Real.sqrt 200 = 10 * Real.sqrt 2 := by
  have h1 : 200 = 2^3 * 5^2 := by rfl
  have h2 : Real.sqrt (200) = Real.sqrt (2^3 * 5^2) := by rw h1
  rw [←show 200 = factorization by rfl] at h2
  exact sorry

end sqrt_200_simplified_l187_187267


namespace eq_of_frac_eq_and_neq_neg_one_l187_187064

theorem eq_of_frac_eq_and_neq_neg_one
  (a b c d : ℝ)
  (h : (a + b) / (c + d) = (b + c) / (a + d))
  (h_neq : (a + b) / (c + d) ≠ -1) :
  a = c :=
sorry

end eq_of_frac_eq_and_neq_neg_one_l187_187064


namespace cube_of_square_is_15625_l187_187305

/-- The third smallest prime number is 5 --/
def third_smallest_prime := 5

/-- The square of 5 is 25 --/
def square_of_third_smallest_prime := third_smallest_prime ^ 2

/-- The cube of the square of the third smallest prime number is 15625 --/
def cube_of_square_of_third_smallest_prime := square_of_third_smallest_prime ^ 3

theorem cube_of_square_is_15625 : cube_of_square_of_third_smallest_prime = 15625 := by
  sorry

end cube_of_square_is_15625_l187_187305


namespace max_metro_lines_l187_187823

theorem max_metro_lines (lines : ℕ) 
  (stations_per_line : ℕ) 
  (max_interchange : ℕ) 
  (max_lines_per_interchange : ℕ) :
  (stations_per_line >= 4) → 
  (max_interchange <= 3) → 
  (max_lines_per_interchange <= 2) → 
  (∀ s_1 s_2, ∃ t_1 t_2, t_1 ≤ max_interchange ∧ t_2 ≤ max_interchange ∧
     (s_1 = t_1 ∨ s_2 = t_1 ∨ s_1 = t_2 ∨ s_2 = t_2)) → 
  lines ≤ 10 :=
by
  sorry

end max_metro_lines_l187_187823


namespace charity_dinner_cost_l187_187357

def cost_of_rice_per_plate : ℝ := 0.10
def cost_of_chicken_per_plate : ℝ := 0.40
def number_of_plates : ℕ := 100

theorem charity_dinner_cost : 
  cost_of_rice_per_plate + cost_of_chicken_per_plate * number_of_plates = 50 :=
by
  sorry

end charity_dinner_cost_l187_187357


namespace fruit_salad_cherries_l187_187948

theorem fruit_salad_cherries (b r g c : ℕ) 
  (h1 : b + r + g + c = 390)
  (h2 : r = 3 * b)
  (h3 : g = 2 * c)
  (h4 : c = 5 * r) :
  c = 119 :=
by
  sorry

end fruit_salad_cherries_l187_187948


namespace position_of_seventeen_fifteen_in_sequence_l187_187053

theorem position_of_seventeen_fifteen_in_sequence :
  ∃ n : ℕ, (17 : ℚ) / 15 = (n + 3 : ℚ) / (n + 1) :=
sorry

end position_of_seventeen_fifteen_in_sequence_l187_187053


namespace combined_weight_cats_l187_187961

-- Define the weights of the cats
def weight_cat1 := 2
def weight_cat2 := 7
def weight_cat3 := 4

-- Prove the combined weight of the three cats is 13 pounds
theorem combined_weight_cats :
  weight_cat1 + weight_cat2 + weight_cat3 = 13 := by
  sorry

end combined_weight_cats_l187_187961


namespace find_second_number_l187_187612

theorem find_second_number (x y z : ℚ) (h1 : x + y + z = 120)
  (h2 : x / y = 3 / 4) (h3 : y / z = 4 / 7) : y = 240 / 7 := by
  sorry

end find_second_number_l187_187612


namespace cube_of_square_of_third_smallest_prime_is_correct_l187_187306

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l187_187306


namespace dima_picks_more_berries_l187_187696

theorem dima_picks_more_berries (N : ℕ) (dima_fastness : ℕ) (sergei_fastness : ℕ) (dima_rate : ℕ) (sergei_rate : ℕ) :
  N = 450 → dima_fastness = 2 * sergei_fastness →
  dima_rate = 1 → sergei_rate = 2 →
  let dima_basket : ℕ := N / 2
  let sergei_basket : ℕ := (2 * N) / 3
  dima_basket > sergei_basket ∧ (dima_basket - sergei_basket) = 50 := 
by {
  sorry
}

end dima_picks_more_berries_l187_187696


namespace joshua_total_bottle_caps_l187_187217

def initial_bottle_caps : ℕ := 40
def bought_bottle_caps : ℕ := 7

theorem joshua_total_bottle_caps : initial_bottle_caps + bought_bottle_caps = 47 := 
by
  sorry

end joshua_total_bottle_caps_l187_187217


namespace hours_to_seconds_l187_187715

theorem hours_to_seconds : 
  (3.5 * 60 * 60) = 12600 := 
by 
  sorry

end hours_to_seconds_l187_187715


namespace least_positive_integer_to_add_l187_187325

theorem least_positive_integer_to_add (n : ℕ) (h_start : n = 525) : ∃ k : ℕ, k > 0 ∧ (n + k) % 5 = 0 ∧ k = 4 :=
by {
  sorry
}

end least_positive_integer_to_add_l187_187325


namespace probability_multiple_of_4_l187_187571

def prob_at_least_one_multiple_of_4 : ℚ :=
  1 - (38/50)^3

theorem probability_multiple_of_4 (n : ℕ) (h : n = 3) : 
  prob_at_least_one_multiple_of_4 = 28051 / 50000 :=
by
  rw [prob_at_least_one_multiple_of_4, ← h]
  sorry

end probability_multiple_of_4_l187_187571


namespace frequency_rate_identity_l187_187826

theorem frequency_rate_identity (n : ℕ) : 
  (36 : ℕ) / (n : ℕ) = (0.25 : ℝ) → 
  n = 144 := by
  sorry

end frequency_rate_identity_l187_187826


namespace linear_function_unique_l187_187717

noncomputable def f (x : ℝ) : ℝ := sorry

theorem linear_function_unique
  (h1 : ∀ x : ℝ, f (f x) = 4 * x + 6)
  (h2 : ∀ x1 x2 : ℝ, x1 < x2 → f x1 < f x2) :
  ∀ x : ℝ, f x = 2 * x + 2 :=
sorry

end linear_function_unique_l187_187717


namespace min_vans_proof_l187_187019

-- Define the capacity and availability of each type of van
def capacity_A : Nat := 7
def capacity_B : Nat := 9
def capacity_C : Nat := 12

def available_A : Nat := 3
def available_B : Nat := 4
def available_C : Nat := 2

-- Define the number of people going on the trip
def students : Nat := 40
def adults : Nat := 14

-- Define the total number of people
def total_people : Nat := students + adults

-- Define the minimum number of vans needed
def min_vans_needed : Nat := 6

-- Define the number of each type of van used
def vans_A_used : Nat := 0
def vans_B_used : Nat := 4
def vans_C_used : Nat := 2

-- Prove the minimum number of vans needed to accommodate everyone is 6
theorem min_vans_proof : min_vans_needed = 6 ∧ 
  (vans_A_used * capacity_A + vans_B_used * capacity_B + vans_C_used * capacity_C = total_people) ∧
  vans_A_used <= available_A ∧ vans_B_used <= available_B ∧ vans_C_used <= available_C :=
by 
  sorry

end min_vans_proof_l187_187019


namespace probability_of_two_co_presidents_l187_187722

noncomputable section

def binomial (n k : ℕ) : ℕ :=
  if h : n ≥ k then Nat.choose n k else 0

def club_prob (n : ℕ) : ℚ :=
  (binomial (n-2) 2 : ℚ) / (binomial n 4 : ℚ)

def total_probability : ℚ :=
  (1/4 : ℚ) * (club_prob 6 + club_prob 8 + club_prob 9 + club_prob 10)

theorem probability_of_two_co_presidents : total_probability = 0.2286 := by
  -- We expect this to be true based on the given solution
  sorry

end probability_of_two_co_presidents_l187_187722


namespace maria_cookies_l187_187234

theorem maria_cookies :
  let c_initial := 19
  let c1 := c_initial - 5
  let c2 := c1 / 2
  let c_final := c2 - 2
  c_final = 5 :=
by
  sorry

end maria_cookies_l187_187234


namespace f_f_f_three_l187_187436

def f (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 1

theorem f_f_f_three : f (f (f 3)) = 43 :=
by
  -- Introduction of definitions and further necessary steps here are skipped
  sorry

end f_f_f_three_l187_187436


namespace sqrt_200_eq_10_sqrt_2_l187_187260

theorem sqrt_200_eq_10_sqrt_2 : Real.sqrt 200 = 10 * Real.sqrt 2 :=
sorry

end sqrt_200_eq_10_sqrt_2_l187_187260


namespace total_time_of_flight_l187_187945

variables {V_0 g t t_1 H : ℝ}  -- Define variables

-- Define conditions
def initial_condition (V_0 g t_1 H : ℝ) : Prop :=
H = (1/2) * g * t_1^2

def return_condition (V_0 g t : ℝ) : Prop :=
t = 2 * (V_0 / g)

theorem total_time_of_flight
  (V_0 g : ℝ)
  (h1 : initial_condition V_0 g (V_0 / g) (1/2 * g * (V_0 / g)^2))
  : return_condition V_0 g (2 * V_0 / g) :=
by
  sorry

end total_time_of_flight_l187_187945


namespace calculate_expression_l187_187382

theorem calculate_expression : (632^2 - 568^2 + 100) = 76900 :=
by sorry

end calculate_expression_l187_187382


namespace probability_non_black_ball_l187_187065

/--
Given the odds of drawing a black ball as 5:3,
prove that the probability of drawing a non-black ball from the bag is 3/8.
-/
theorem probability_non_black_ball (n_black n_non_black : ℕ) (h : n_black = 5) (h' : n_non_black = 3) :
  (n_non_black : ℚ) / (n_black + n_non_black) = 3 / 8 :=
by
  -- proof goes here
  sorry

end probability_non_black_ball_l187_187065


namespace total_messages_l187_187068

theorem total_messages (x : ℕ) (h : x * (x - 1) = 420) : x * (x - 1) = 420 :=
by
  sorry

end total_messages_l187_187068


namespace pyramid_total_area_l187_187935

noncomputable def pyramid_base_edge := 8
noncomputable def pyramid_lateral_edge := 10

/-- The total area of the four triangular faces of a right, square-based pyramid
with base edges measuring 8 units and lateral edges measuring 10 units is 32 * sqrt(21). -/
theorem pyramid_total_area :
  let base_edge := pyramid_base_edge,
      lateral_edge := pyramid_lateral_edge,
      height := sqrt (lateral_edge ^ 2 - (base_edge / 2) ^ 2),
      area_of_one_face := 1 / 2 * base_edge * height
  in 4 * area_of_one_face = 32 * sqrt 21 :=
sorry

end pyramid_total_area_l187_187935


namespace x_cubed_plus_y_cubed_l187_187539

theorem x_cubed_plus_y_cubed (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 14) : x^3 + y^3 = 85 / 2 :=
by
  sorry

end x_cubed_plus_y_cubed_l187_187539


namespace solve_problem_l187_187100

open Real

noncomputable def problem (x : ℝ) : Prop :=
  (cos (2 * x / 5) - cos (2 * π / 15)) ^ 2 + (sin (2 * x / 3) - sin (4 * π / 9)) ^ 2 = 0

theorem solve_problem : ∀ t : ℤ, problem ((29 * π / 3) + 15 * π * t) :=
by
  intro t
  sorry

end solve_problem_l187_187100


namespace Carmen_average_speed_l187_187383

/-- Carmen participates in a two-part cycling race. In the first part, she covers 24 miles in 3 hours.
    In the second part, due to fatigue, her speed decreases, and she takes 4 hours to cover 16 miles.
    Calculate Carmen's average speed for the entire race. -/
theorem Carmen_average_speed :
  let distance1 := 24 -- miles in the first part
  let time1 := 3 -- hours in the first part
  let distance2 := 16 -- miles in the second part
  let time2 := 4 -- hours in the second part
  let total_distance := distance1 + distance2
  let total_time := time1 + time2
  let average_speed := total_distance / total_time
  average_speed = 40 / 7 :=
by
  sorry

end Carmen_average_speed_l187_187383


namespace min_inverse_ab_l187_187443

theorem min_inverse_ab (a b : ℝ) (h1 : a + a * b + 2 * b = 30) (h2 : a > 0) (h3 : b > 0) :
  ∃ m : ℝ, m = 1 / 18 ∧ (∀ x y : ℝ, (x + x * y + 2 * y = 30) → (x > 0) → (y > 0) → 1 / (x * y) ≥ m) :=
sorry

end min_inverse_ab_l187_187443


namespace sum_R1_R2_eq_19_l187_187214

-- Definitions for F_1 and F_2 in base R_1 and R_2
def F1_R1 : ℚ := 37 / 99
def F2_R1 : ℚ := 73 / 99
def F1_R2 : ℚ := 25 / 99
def F2_R2 : ℚ := 52 / 99

-- Prove that the sum of R1 and R2 is 19
theorem sum_R1_R2_eq_19 (R1 R2 : ℕ) (hF1R1 : F1_R1 = (3 * R1 + 7) / (R1^2 - 1))
  (hF2R1 : F2_R1 = (7 * R1 + 3) / (R1^2 - 1))
  (hF1R2 : F1_R2 = (2 * R2 + 5) / (R2^2 - 1))
  (hF2R2 : F2_R2 = (5 * R2 + 2) / (R2^2 - 1)) :
  R1 + R2 = 19 :=
  sorry

end sum_R1_R2_eq_19_l187_187214


namespace number_of_girls_l187_187616

theorem number_of_girls (n : ℕ) (h : 2 * 25 * (n * (n - 1)) = 3 * 25 * 24) : (25 - n) = 16 := by
  sorry

end number_of_girls_l187_187616


namespace f_8_plus_f_9_l187_187993

noncomputable def f : ℝ → ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x 
axiom f_even_transformed : ∀ x : ℝ, f (x + 2) = f (2 - x)
axiom f_at_1 : f 1 = 1

theorem f_8_plus_f_9 : f 8 + f 9 = 1 :=
sorry

end f_8_plus_f_9_l187_187993


namespace ratio_first_term_l187_187000

theorem ratio_first_term (x : ℕ) (r : ℕ × ℕ) (h₀ : r = (6 - x, 7 - x)) 
        (h₁ : x ≥ 3) (h₂ : r.1 < r.2) : r.1 < 4 :=
by
  sorry

end ratio_first_term_l187_187000


namespace solve_for_y_l187_187130

theorem solve_for_y (x y : ℝ) (h1 : x * y = 1) (h2 : x / y = 36) (h3 : 0 < x) (h4 : 0 < y) : 
  y = 1 / 6 := 
sorry

end solve_for_y_l187_187130


namespace driver_net_pay_rate_l187_187141

theorem driver_net_pay_rate
    (hours : ℕ) (distance_per_hour : ℕ) (distance_per_gallon : ℕ) 
    (pay_per_mile : ℝ) (gas_cost_per_gallon : ℝ) :
    hours = 3 →
    distance_per_hour = 50 →
    distance_per_gallon = 25 →
    pay_per_mile = 0.75 →
    gas_cost_per_gallon = 2.50 →
    (pay_per_mile * (distance_per_hour * hours) - gas_cost_per_gallon * ((distance_per_hour * hours) / distance_per_gallon)) / hours = 32.5 :=
by
  intros h_hours h_dph h_dpg h_ppm h_gcpg
  sorry

end driver_net_pay_rate_l187_187141


namespace find_x_value_l187_187037

theorem find_x_value (x : ℝ) :
  |x - 25| + |x - 21| = |3 * x - 75| → x = 71 / 3 :=
by
  sorry

end find_x_value_l187_187037


namespace percentage_increase_correct_l187_187556

def bookstore_earnings : ℕ := 60
def tutoring_earnings : ℕ := 40
def new_bookstore_earnings : ℕ := 100
def additional_tutoring_fee : ℕ := 15
def old_total_earnings : ℕ := bookstore_earnings + tutoring_earnings
def new_total_earnings : ℕ := new_bookstore_earnings + (tutoring_earnings + additional_tutoring_fee)
def overall_percentage_increase : ℚ := (((new_total_earnings - old_total_earnings : ℚ) / old_total_earnings) * 100)

theorem percentage_increase_correct :
  overall_percentage_increase = 55 := sorry

end percentage_increase_correct_l187_187556


namespace sqrt_200_eq_10_l187_187273

theorem sqrt_200_eq_10 (h : 200 = 2^2 * 5^2) : Real.sqrt 200 = 10 := 
by
  sorry

end sqrt_200_eq_10_l187_187273


namespace maria_cookies_l187_187233

theorem maria_cookies :
  let c_initial := 19
  let c1 := c_initial - 5
  let c2 := c1 / 2
  let c_final := c2 - 2
  c_final = 5 :=
by
  sorry

end maria_cookies_l187_187233


namespace sum_of_legs_equal_l187_187078

theorem sum_of_legs_equal
  (a b c d e f g h : ℝ)
  (x y : ℝ)
  (h_similar_shaded1 : a = a * x ∧ b = a * y)
  (h_similar_shaded2 : c = c * x ∧ d = c * y)
  (h_similar_shaded3 : e = e * x ∧ f = e * y)
  (h_similar_shaded4 : g = g * x ∧ h = g * y)
  (h_similar_unshaded1 : h = h * x ∧ a = h * y)
  (h_similar_unshaded2 : b = b * x ∧ c = b * y)
  (h_similar_unshaded3 : d = d * x ∧ e = d * y)
  (h_similar_unshaded4 : f = f * x ∧ g = f * y)
  (x_non_zero : x ≠ 0) (y_non_zero : y ≠ 0) : 
  (a * y + b + c * x) + (c * y + d + e * x) + (e * y + f + g * x) + (g * y + h + a * x) 
  = (h * x + a + b * y) + (b * x + c + d * y) + (d * x + e + f * y) + (f * x + g + h * y) :=
sorry

end sum_of_legs_equal_l187_187078


namespace expression_takes_many_different_values_l187_187840

theorem expression_takes_many_different_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) : 
  ∃ v : ℝ, ∀ x, x ≠ 3 → x ≠ -2 → v = (3*x^2 - 2*x + 3)/((x - 3)*(x + 2)) - (5*x - 6)/((x - 3)*(x + 2)) := 
sorry

end expression_takes_many_different_values_l187_187840


namespace increasing_function_of_a_l187_187708

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x > 1 then a^x else (2 - (a / 2)) * x + 2

theorem increasing_function_of_a (a : ℝ) : (∀ x y, x < y → f a x ≤ f a y) ↔ 
  (8 / 3 ≤ a ∧ a < 4) :=
sorry

end increasing_function_of_a_l187_187708


namespace solve_diamond_l187_187535

theorem solve_diamond (d : ℕ) (hd : d < 10) (h : d * 9 + 6 = d * 10 + 3) : d = 3 :=
sorry

end solve_diamond_l187_187535


namespace steve_took_4_berries_l187_187588

theorem steve_took_4_berries (s t : ℕ) (H1 : s = 32) (H2 : t = 21) (H3 : s - 7 = t + x) :
  x = 4 :=
by
  sorry

end steve_took_4_berries_l187_187588


namespace geometric_arithmetic_sequences_sum_l187_187427

theorem geometric_arithmetic_sequences_sum (a b : ℕ → ℝ) (S_n : ℕ → ℝ) 
  (q d : ℝ) (h1 : 0 < q) 
  (h2 : a 1 = 1) (h3 : b 1 = 1) 
  (h4 : a 5 + b 3 = 21) 
  (h5 : a 3 + b 5 = 13) :
  (∀ n, a n = 2^(n-1)) ∧ (∀ n, b n = 2*n - 1) ∧ (∀ n, S_n n = 3 - (2*n + 3)/(2^n)) := 
sorry

end geometric_arithmetic_sequences_sum_l187_187427


namespace maria_cookies_left_l187_187231

def maria_cookies (initial: ℕ) (to_friend: ℕ) (to_family_divisor: ℕ) (eats: ℕ) : ℕ :=
  (initial - to_friend) / to_family_divisor - eats

theorem maria_cookies_left (h : maria_cookies 19 5 2 2 = 5): true :=
by trivial

end maria_cookies_left_l187_187231


namespace proportion_correct_l187_187060

theorem proportion_correct (x y : ℝ) (h : 5 * y = 4 * x) : x / y = 5 / 4 :=
sorry

end proportion_correct_l187_187060


namespace max_value_of_fraction_l187_187785

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l187_187785


namespace solve_absolute_value_eq_l187_187653

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l187_187653


namespace magnitude_conjugate_of_z_l187_187567

noncomputable def z : ℂ := (1 / (1 - complex.I)) + complex.I

theorem magnitude_conjugate_of_z : complex.abs (conj z) = real.sqrt 10 / 2 := by
  sorry

end magnitude_conjugate_of_z_l187_187567


namespace max_expression_value_l187_187767

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l187_187767


namespace problem1_problem2_l187_187016

-- Problem 1: Evaluating an integer arithmetic expression
theorem problem1 : (1 * (-8)) - (-6) + (-3) = -5 := 
by
  sorry

-- Problem 2: Evaluating a mixed arithmetic expression with rational numbers and decimals
theorem problem2 : (5 / 13) - 3.7 + (8 / 13) - (-1.7) = -1 := 
by
  sorry

end problem1_problem2_l187_187016


namespace min_guests_at_banquet_l187_187941

theorem min_guests_at_banquet (total_food : ℕ) (max_food_per_guest : ℕ) : 
  total_food = 323 ∧ max_food_per_guest = 2 → 
  (∀ guests : ℕ, guests * max_food_per_guest >= total_food) → 
  (∃ g : ℕ, g = 162) :=
by
  -- Assuming total food and max food per guest
  intro h_cons
  -- Mathematical proof steps would go here, skipping with sorry
  sorry

end min_guests_at_banquet_l187_187941


namespace correct_option_C_l187_187807

variable (a : ℝ)

theorem correct_option_C : (a^2 * a = a^3) :=
by sorry

end correct_option_C_l187_187807


namespace other_books_new_releases_percentage_l187_187158

theorem other_books_new_releases_percentage
  (T : ℝ)
  (h1 : 0 < T)
  (hf_books : ℝ := 0.4 * T)
  (hf_new_releases : ℝ := 0.4 * hf_books)
  (other_books : ℝ := 0.6 * T)
  (total_new_releases : ℝ := hf_new_releases + (P * other_books))
  (fraction_hf_new : ℝ := hf_new_releases / total_new_releases)
  (fraction_value : fraction_hf_new = 0.27586206896551724)
  : P = 0.7 :=
sorry

end other_books_new_releases_percentage_l187_187158


namespace prime_sum_exists_even_n_l187_187179

theorem prime_sum_exists_even_n (n : ℕ) :
  (∃ a b c : ℤ, a + b + c = 0 ∧ Prime (a^n + b^n + c^n)) ↔ Even n := 
by
  sorry

end prime_sum_exists_even_n_l187_187179


namespace adult_elephant_weekly_bananas_l187_187683

theorem adult_elephant_weekly_bananas (daily_bananas : Nat) (days_in_week : Nat) (H1 : daily_bananas = 90) (H2 : days_in_week = 7) :
  daily_bananas * days_in_week = 630 :=
by
  sorry

end adult_elephant_weekly_bananas_l187_187683


namespace cube_removal_minimum_l187_187485

theorem cube_removal_minimum (l w h : ℕ) (hu : l = 4) (hv : w = 5) (hw : h = 6) :
  ∃ num_cubes_removed : ℕ, 
    (l * w * h - num_cubes_removed = 4 * 4 * 4) ∧ 
    num_cubes_removed = 56 := 
by
  sorry

end cube_removal_minimum_l187_187485


namespace kelly_raisins_l187_187219

theorem kelly_raisins (weight_peanuts : ℝ) (total_weight_snacks : ℝ) (h1 : weight_peanuts = 0.1) (h2 : total_weight_snacks = 0.5) : total_weight_snacks - weight_peanuts = 0.4 := by
  sorry

end kelly_raisins_l187_187219


namespace root_expression_value_l187_187732

theorem root_expression_value
  (r s : ℝ)
  (h1 : 3 * r^2 - 4 * r - 8 = 0)
  (h2 : 3 * s^2 - 4 * s - 8 = 0) :
  (9 * r^3 - 9 * s^3) * (r - s)⁻¹ = 40 := 
sorry

end root_expression_value_l187_187732


namespace kangaroo_meetings_l187_187632

/-- 
Two kangaroos, A and B, start at point A and jump in specific sequences:
- Kangaroo A jumps in the sequence A, B, C, D, E, F, G, H, I, A, B, C, ... in a loop every 9 jumps.
- Kangaroo B jumps in the sequence A, B, D, E, G, H, A, B, D, ... in a loop every 6 jumps.
They start at point A together. Prove that they will land on the same point 226 times after 2017 jumps.
-/
theorem kangaroo_meetings (n : Nat) (ka : Fin 9 → Fin 9) (kb : Fin 6 → Fin 6)
  (hka : ∀ i, ka i = (i + 1) % 9) (hkb : ∀ i, kb i = (i + 1) % 6) :
  n = 2017 →
  -- Prove that the two kangaroos will meet 226 times after 2017 jumps
  ∃ k, k = 226 :=
by
  sorry

end kangaroo_meetings_l187_187632


namespace part1_exists_infinite_rationals_part2_rationals_greater_bound_l187_187476

theorem part1_exists_infinite_rationals 
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2):
  ∀ ε > 0, ∃ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 ∧ abs (q / p - sqrt5_minus1_div2) < 1 / p ^ 2 :=
by sorry

theorem part2_rationals_greater_bound
  (sqrt5_minus1_div2 := (Real.sqrt 5 - 1) / 2)
  (sqrt5_plus1_inv := 1 / (Real.sqrt 5 + 1)):
  ∀ p q : ℤ, p > 0 ∧ Int.gcd p q = 1 → abs (q / p - sqrt5_minus1_div2) > sqrt5_plus1_inv / p ^ 2 :=
by sorry

end part1_exists_infinite_rationals_part2_rationals_greater_bound_l187_187476


namespace expectation_neg_xi_l187_187423

noncomputable def xi (ω : Ω) : ℕ := sorry  -- Placeholder for the random variable
variable (ω : Ω)

def binomial_xi := ∀ ω, IsBinomial (5 : ℕ) (1/4 : ℝ) (xi ω)

theorem expectation_neg_xi (h : binomial_xi ω) : E (- xi ω) = - (5 : ℝ) / 4 :=
sorry

end expectation_neg_xi_l187_187423


namespace license_plate_combinations_l187_187378

theorem license_plate_combinations :
  let letters := 26
  let two_other_letters := Nat.choose 25 2
  let repeated_positions := Nat.choose 4 2
  let arrange_two_letters := 2
  let first_digit_choices := 10
  let second_digit_choices := 9
  letters * two_other_letters * repeated_positions * arrange_two_letters * first_digit_choices * second_digit_choices = 8424000 :=
  sorry

end license_plate_combinations_l187_187378


namespace unique_solution_of_quadratics_l187_187566

theorem unique_solution_of_quadratics (y : ℚ) 
    (h1 : 9 * y^2 + 8 * y - 3 = 0) 
    (h2 : 27 * y^2 + 35 * y - 12 = 0) : 
    y = 1 / 3 :=
sorry

end unique_solution_of_quadratics_l187_187566


namespace calculation_l187_187015

theorem calculation :
  7^3 - 4 * 7^2 + 4 * 7 - 1 = 174 :=
by
  sorry

end calculation_l187_187015


namespace passes_to_left_l187_187147

theorem passes_to_left (total_passes right_passes center_passes left_passes : ℕ)
  (h_total : total_passes = 50)
  (h_right : right_passes = 2 * left_passes)
  (h_center : center_passes = left_passes + 2)
  (h_sum : left_passes + right_passes + center_passes = total_passes) :
  left_passes = 12 := 
by
  sorry

end passes_to_left_l187_187147


namespace olivia_earnings_this_week_l187_187239

variable (hourly_rate : ℕ) (hours_monday hours_wednesday hours_friday : ℕ)

theorem olivia_earnings_this_week : 
  hourly_rate = 9 → 
  hours_monday = 4 → 
  hours_wednesday = 3 → 
  hours_friday = 6 → 
  (hourly_rate * hours_monday + hourly_rate * hours_wednesday + hourly_rate * hours_friday) = 117 := 
by
  intros
  sorry

end olivia_earnings_this_week_l187_187239


namespace solve_absolute_value_eq_l187_187651

theorem solve_absolute_value_eq : ∀ x : ℝ, |x - 3| = |x - 5| ↔ x = 4 := by
  intro x
  split
  { -- Forward implication
    intro h
    have h_nonneg_3 := abs_nonneg (x - 3)
    have h_nonneg_5 := abs_nonneg (x - 5)
    rw [← sub_eq_zero, abs_eq_iff] at h
    cases h
    { -- Case 1: x - 3 = x - 5
      linarith 
    }
    { -- Case 2: x - 3 = -(x - 5)
      linarith
    }
  }
  { -- Backward implication
    intro h
    rw h
    calc 
      |4 - 3| = 1 : by norm_num
      ... = |4 - 5| : by norm_num
  }

end solve_absolute_value_eq_l187_187651


namespace percentage_of_profits_to_revenues_l187_187212

theorem percentage_of_profits_to_revenues (R P : ℝ) (h1 : 0.7 * R = R - 0.3 * R) (h2 : 0.105 * R = 0.15 * (0.7 * R)) (h3 : 0.105 * R = 1.0499999999999999 * P) :
  (P / R) * 100 = 10 :=
by
  sorry

end percentage_of_profits_to_revenues_l187_187212


namespace line_equation_l187_187917

theorem line_equation (x y : ℝ) (hx : ∃ t : ℝ, t ≠ 0 ∧ x = t * -3) (hy : ∃ t : ℝ, t ≠ 0 ∧ y = t * 4) :
  4 * x - 3 * y + 12 = 0 := 
sorry

end line_equation_l187_187917


namespace simplify_sqrt_200_l187_187280

theorem simplify_sqrt_200 : (sqrt 200 : ℝ) = 10 * sqrt 2 := by
  -- proof goes here
  sorry

end simplify_sqrt_200_l187_187280


namespace bob_speed_l187_187790

theorem bob_speed (v : ℝ) : (∀ v_a : ℝ, v_a > 120 → 30 / v_a < 30 / v - 0.5) → v = 40 :=
by
  sorry

end bob_speed_l187_187790


namespace Rachel_painting_time_l187_187737

noncomputable def Matt_time : ℕ := 12
noncomputable def Patty_time (Matt_time : ℕ) : ℕ := Matt_time / 3
noncomputable def Rachel_time (Patty_time : ℕ) : ℕ := 5 + 2 * Patty_time

theorem Rachel_painting_time : Rachel_time (Patty_time Matt_time) = 13 := by
  sorry

end Rachel_painting_time_l187_187737


namespace simplify_expression_l187_187289

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : ( (3 * x + 6 - 5 * x) / 3 ) = ( (-2 * x) / 3 + 2 ) :=
by
  sorry

end simplify_expression_l187_187289


namespace problem_1_problem_2_problem_3_l187_187038

-- Problem 1
theorem problem_1 (x : ℝ) (h : 4.8 - 3 * x = 1.8) : x = 1 :=
by { sorry }

-- Problem 2
theorem problem_2 (x : ℝ) (h : (1 / 8) / (1 / 5) = x / 24) : x = 15 :=
by { sorry }

-- Problem 3
theorem problem_3 (x : ℝ) (h : 7.5 * x + 6.5 * x = 2.8) : x = 0.2 :=
by { sorry }

end problem_1_problem_2_problem_3_l187_187038


namespace marcus_calzones_total_time_l187_187227

/-
Conditions:
1. It takes Marcus 20 minutes to saute the onions.
2. It takes a quarter of the time to saute the garlic and peppers that it takes to saute the onions.
3. It takes 30 minutes to knead the dough.
4. It takes twice as long to let the dough rest as it takes to knead it.
5. It takes 1/10th of the combined kneading and resting time to assemble the calzones.
-/

def time_saute_onions : ℕ := 20
def time_saute_garlic_peppers : ℕ := time_saute_onions / 4
def time_knead : ℕ := 30
def time_rest : ℕ := 2 * time_knead
def time_assemble : ℕ := (time_knead + time_rest) / 10

def total_time_making_calzones : ℕ :=
  time_saute_onions + time_saute_garlic_peppers + time_knead + time_rest + time_assemble

theorem marcus_calzones_total_time : total_time_making_calzones = 124 := by
  -- All steps and proof details to be filled in
  sorry

end marcus_calzones_total_time_l187_187227


namespace find_initial_mean_l187_187918

/-- 
  The mean of 50 observations is M.
  One observation was wrongly taken as 23 but should have been 30.
  The corrected mean is 36.5.
  Prove that the initial mean M was 36.36.
-/
theorem find_initial_mean (M : ℝ) (h : 50 * 36.36 + 7 = 50 * 36.5) : 
  (500 * 36.36 - 7) = 1818 :=
sorry

end find_initial_mean_l187_187918


namespace time_to_cross_first_platform_l187_187958

variable (length_first_platform : ℝ)
variable (length_second_platform : ℝ)
variable (time_to_cross_second_platform : ℝ)
variable (length_of_train : ℝ)

theorem time_to_cross_first_platform :
  length_first_platform = 160 →
  length_second_platform = 250 →
  time_to_cross_second_platform = 20 →
  length_of_train = 110 →
  (270 / (360 / 20) = 15) := 
by
  intro h1 h2 h3 h4
  sorry

end time_to_cross_first_platform_l187_187958


namespace inequality_lemma_l187_187544

theorem inequality_lemma (a b c d : ℝ) (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) (hd : 0 < d ∧ d < 1) :
  1 + a * b + b * c + c * d + d * a + a * c + b * d > a + b + c + d :=
by 
  sorry

end inequality_lemma_l187_187544


namespace length_of_fourth_side_in_cyclic_quadrilateral_l187_187484

theorem length_of_fourth_side_in_cyclic_quadrilateral :
  ∀ (r a b c : ℝ), r = 300 ∧ a = 300 ∧ b = 300 ∧ c = 150 * Real.sqrt 2 →
  ∃ d : ℝ, d = 450 :=
by
  sorry

end length_of_fourth_side_in_cyclic_quadrilateral_l187_187484


namespace ellen_dinner_calories_proof_l187_187507

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l187_187507


namespace ellen_dinner_calories_proof_l187_187506

def ellen_daily_calories := 2200
def ellen_breakfast_calories := 353
def ellen_lunch_calories := 885
def ellen_snack_calories := 130
def ellen_remaining_calories : ℕ :=
  ellen_daily_calories - (ellen_breakfast_calories + ellen_lunch_calories + ellen_snack_calories)

theorem ellen_dinner_calories_proof : ellen_remaining_calories = 832 := by
  sorry

end ellen_dinner_calories_proof_l187_187506


namespace find_numbers_l187_187994

theorem find_numbers 
  (a b c d : ℝ)
  (h1 : b / c = c / a)
  (h2 : a + b + c = 19)
  (h3 : b - c = c - d)
  (h4 : b + c + d = 12) :
  (a = 25 ∧ b = -10 ∧ c = 4 ∧ d = 18) ∨ (a = 9 ∧ b = 6 ∧ c = 4 ∧ d = 2) :=
sorry

end find_numbers_l187_187994


namespace inequality_proof_l187_187345

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
    (x^3 / (x^3 + 2 * y^2 * Real.sqrt (z * x))) + 
    (y^3 / (y^3 + 2 * z^2 * Real.sqrt (x * y))) + 
    (z^3 / (z^3 + 2 * x^2 * Real.sqrt (y * z))) ≥ 1 := 
by 
  sorry

end inequality_proof_l187_187345


namespace max_expression_value_l187_187777

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l187_187777


namespace find_number_of_non_officers_l187_187593

theorem find_number_of_non_officers
  (avg_salary_all : ℝ)
  (avg_salary_officers : ℝ)
  (avg_salary_non_officers : ℝ)
  (num_officers : ℕ) :
  avg_salary_all = 120 ∧
  avg_salary_officers = 450 ∧
  avg_salary_non_officers = 110 ∧
  num_officers = 15 →
  ∃ N : ℕ, (120 * (15 + N) = 450 * 15 + 110 * N) ∧ N = 495 :=
by
  sorry

end find_number_of_non_officers_l187_187593


namespace apple_slices_count_l187_187582

theorem apple_slices_count :
  let boxes := 7
  let apples_per_box := 7
  let slices_per_apple := 8
  let total_apples := boxes * apples_per_box
  let total_slices := total_apples * slices_per_apple
  total_slices = 392 :=
by
  sorry

end apple_slices_count_l187_187582


namespace inequality_proof_l187_187337

variable {x y z : ℝ}

theorem inequality_proof (x y z : ℝ) : 
  (x^3 / (x^3 + 2 * y^2 * real.sqrt(z * x)) + 
  y^3 / (y^3 + 2 * z^2 * real.sqrt(x * y)) + 
  z^3 / (z^3 + 2 * x^2 * real.sqrt(y * z)) >= 1) :=
sorry

end inequality_proof_l187_187337


namespace six_square_fill_l187_187685

def adjacent (x y : ℕ) : Prop :=
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) ∨ 
  (x = 2 ∧ y = 3) ∨ (x = 3 ∧ y = 2) ∨
  (x = 3 ∧ y = 4) ∨ (x = 4 ∧ y = 3) ∨
  (x = 4 ∧ y = 5) ∨ (x = 5 ∧ y = 4) ∨
  (x = 5 ∧ y = 6) ∨ (x = 6 ∧ y = 5) 

def valid_assignment (assignment : Fin 6 → ℕ) : Prop :=
  ∀ i j, adjacent i j → (assignment i - assignment j ≠ 3) ∧ (assignment j - assignment i ≠ 3)

def total_valid_assignments : ℕ :=
  {assignment // valid_assignment assignment}.card

theorem six_square_fill :
  total_valid_assignments = 96 :=
  sorry

end six_square_fill_l187_187685


namespace books_per_week_l187_187576

-- Define the conditions
def total_books_read : ℕ := 20
def weeks : ℕ := 5

-- Define the statement to be proved
theorem books_per_week : (total_books_read / weeks) = 4 := by
  -- Proof omitted
  sorry

end books_per_week_l187_187576


namespace max_expression_value_l187_187771

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l187_187771


namespace largest_sum_of_three_faces_l187_187548

theorem largest_sum_of_three_faces (faces : Fin 6 → ℕ)
  (h_unique : ∀ i j, i ≠ j → faces i ≠ faces j)
  (h_range : ∀ i, 1 ≤ faces i ∧ faces i ≤ 6)
  (h_opposite_sum : ∀ i, faces i + faces (5 - i) = 10) :
  ∃ i j k, i ≠ j ∧ j ≠ k ∧ k ≠ i ∧ faces i + faces j + faces k = 12 :=
by sorry

end largest_sum_of_three_faces_l187_187548


namespace angle_measures_possible_l187_187605

theorem angle_measures_possible (A B : ℕ) (h1 : A > 0) (h2 : B > 0) (h3 : A + B = 180) (h4 : ∃ k, k > 0 ∧ A = k * B) : 
  ∃ n : ℕ, n = 18 := 
sorry

end angle_measures_possible_l187_187605


namespace find_larger_number_l187_187296

theorem find_larger_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
  sorry

end find_larger_number_l187_187296


namespace cube_of_square_of_third_smallest_prime_l187_187312

-- Define the third smallest prime number
def third_smallest_prime : ℕ := 5

-- Theorem to prove the cube of the square of the third smallest prime number
theorem cube_of_square_of_third_smallest_prime :
  (third_smallest_prime^2)^3 = 15625 := by
  sorry

end cube_of_square_of_third_smallest_prime_l187_187312


namespace power_equivalence_l187_187866

theorem power_equivalence (p : ℕ) (hp : 81^10 = 3^p) : p = 40 :=
by {
  -- Proof steps would go here
  sorry
}

end power_equivalence_l187_187866


namespace g_five_l187_187595

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g (x + y) = g x + g y
axiom g_one : g 1 = 2

theorem g_five : g 5 = 10 :=
by sorry

end g_five_l187_187595


namespace aleksey_divisible_l187_187613

theorem aleksey_divisible
  (x y a b S : ℤ)
  (h1 : x + y = S)
  (h2 : S ∣ (a * x + b * y)) :
  S ∣ (b * x + a * y) := 
sorry

end aleksey_divisible_l187_187613
