import Mathlib

namespace NUMINAMATH_GPT_train_probability_correct_l1357_135716

/-- Define the necessary parameters and conditions --/
noncomputable def train_arrival_prob (train_start train_wait max_time_Alex max_time_train : ℝ) : ℝ :=
  let total_possible_area := max_time_Alex * max_time_train
  let overlap_area := (max_time_train - train_wait) * train_wait + (train_wait) * max_time_train / 2
  overlap_area / total_possible_area

/-- Main theorem stating that the probability is 3/10 --/
theorem train_probability_correct :
  train_arrival_prob 0 15 75 60 = 3 / 10 :=
by sorry

end NUMINAMATH_GPT_train_probability_correct_l1357_135716


namespace NUMINAMATH_GPT_range_of_a_l1357_135723

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then (a - 2) * x - 1 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ 2 < a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1357_135723


namespace NUMINAMATH_GPT_reciprocal_of_sum_is_correct_l1357_135770

theorem reciprocal_of_sum_is_correct : (1 / (1 / 4 + 1 / 6)) = 12 / 5 := by
  sorry

end NUMINAMATH_GPT_reciprocal_of_sum_is_correct_l1357_135770


namespace NUMINAMATH_GPT_seated_men_l1357_135798

def passengers : Nat := 48
def fraction_of_women : Rat := 2/3
def fraction_of_men_standing : Rat := 1/8

theorem seated_men (men women standing seated : Nat) 
  (h1 : women = passengers * fraction_of_women)
  (h2 : men = passengers - women)
  (h3 : standing = men * fraction_of_men_standing)
  (h4 : seated = men - standing) :
  seated = 14 := by
  sorry

end NUMINAMATH_GPT_seated_men_l1357_135798


namespace NUMINAMATH_GPT_not_prime_257_1092_1092_l1357_135788

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem not_prime_257_1092_1092 :
  is_prime 1093 →
  ¬ is_prime (257 ^ 1092 + 1092) :=
by
  intro h_prime_1093
  -- Detailed steps are omitted, proof goes here
  sorry

end NUMINAMATH_GPT_not_prime_257_1092_1092_l1357_135788


namespace NUMINAMATH_GPT_postage_arrangements_11_cents_l1357_135790

-- Definitions for the problem settings, such as stamp denominations and counts
def stamp_collection : List (ℕ × ℕ) := [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (10, 10)]

-- Function to calculate all unique arrangements of stamps that sum to a given value (11 cents)
def count_arrangements (total_cents : ℕ) : ℕ :=
  -- The implementation would involve a combinatorial counting taking into account the problem conditions
  sorry

-- The main theorem statement asserting the solution
theorem postage_arrangements_11_cents :
  count_arrangements 11 = 71 :=
  sorry

end NUMINAMATH_GPT_postage_arrangements_11_cents_l1357_135790


namespace NUMINAMATH_GPT_books_read_in_common_l1357_135767

theorem books_read_in_common (T D B total X : ℕ) 
  (hT : T = 23) 
  (hD : D = 12) 
  (hB : B = 17) 
  (htotal : total = 47)
  (h_eq : (T - X) + (D - X) + B + 1 = total) : 
  X = 3 :=
by
  -- Here would go the proof details.
  sorry

end NUMINAMATH_GPT_books_read_in_common_l1357_135767


namespace NUMINAMATH_GPT_pyramid_volume_eq_l1357_135750

noncomputable def volume_of_pyramid (base_length1 base_length2 height : ℝ) : ℝ :=
  (1 / 3) * base_length1 * base_length2 * height

theorem pyramid_volume_eq (base_length1 base_length2 height : ℝ) (h1 : base_length1 = 1) (h2 : base_length2 = 2) (h3 : height = 1) :
  volume_of_pyramid base_length1 base_length2 height = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_pyramid_volume_eq_l1357_135750


namespace NUMINAMATH_GPT_total_fruits_l1357_135710

-- Define the given conditions
variable (a o : ℕ)
variable (ratio : a = 2 * o)
variable (half_apples_to_ann : a / 2 - 3 = 4)
variable (apples_to_cassie : a - a / 2 - 3 = 0)
variable (oranges_kept : 5 = o - 3)

theorem total_fruits (a o : ℕ) (ratio : a = 2 * o) 
  (half_apples_to_ann : a / 2 - 3 = 4) 
  (apples_to_cassie : a - a / 2 - 3 = 0) 
  (oranges_kept : 5 = o - 3) : a + o = 21 := 
sorry

end NUMINAMATH_GPT_total_fruits_l1357_135710


namespace NUMINAMATH_GPT_initial_floor_l1357_135786

theorem initial_floor (x y z : ℤ)
  (h1 : y = x - 7)
  (h2 : z = y + 3)
  (h3 : 13 = z + 8) :
  x = 9 :=
sorry

end NUMINAMATH_GPT_initial_floor_l1357_135786


namespace NUMINAMATH_GPT_ellipse_slope_product_constant_l1357_135761

noncomputable def ellipse_constant_slope_product (a b : ℝ) (P M : ℝ × ℝ) (N : ℝ × ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧
  (P.1^2 / a^2 + P.2^2 / b^2 = 1) ∧
  (M.1^2 / a^2 + M.2^2 / b^2 = 1) ∧
  (N.1 = -M.1 ∧ N.2 = -M.2) ∧
  (∃ k_PM k_PN : ℝ, k_PM = (P.2 - M.2) / (P.1 - M.1) ∧ k_PN = (P.2 - N.2) / (P.1 - N.1)) ∧
  ((P.2 - M.2) / (P.1 - M.1) * (P.2 - N.2) / (P.1 - N.1) = -b^2 / a^2)

theorem ellipse_slope_product_constant (a b : ℝ) (P M N : ℝ × ℝ) :
  ellipse_constant_slope_product a b P M N := 
sorry

end NUMINAMATH_GPT_ellipse_slope_product_constant_l1357_135761


namespace NUMINAMATH_GPT_combined_weight_is_150_l1357_135754

-- Definitions based on conditions
def tracy_weight : ℕ := 52
def jake_weight : ℕ := tracy_weight + 8
def weight_range : ℕ := 14
def john_weight : ℕ := tracy_weight - 14

-- Proving the combined weight
theorem combined_weight_is_150 :
  tracy_weight + jake_weight + john_weight = 150 := by
  sorry

end NUMINAMATH_GPT_combined_weight_is_150_l1357_135754


namespace NUMINAMATH_GPT_base_conversion_subtraction_l1357_135726

theorem base_conversion_subtraction :
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  n1 - n2 = 7422 :=
by
  let n1 := 5 * 7^4 + 2 * 7^3 + 1 * 7^2 + 4 * 7^1 + 3 * 7^0
  let n2 := 1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0
  show n1 - n2 = 7422
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l1357_135726


namespace NUMINAMATH_GPT_smallest_square_number_l1357_135785

theorem smallest_square_number (x y : ℕ) (hx : ∃ a, x = a ^ 2) (hy : ∃ b, y = b ^ 3) 
  (h_simp: ∃ c d, x / (y ^ 3) = c ^ 3 / d ^ 2 ∧ c > 1 ∧ d > 1): x = 64 := by
  sorry

end NUMINAMATH_GPT_smallest_square_number_l1357_135785


namespace NUMINAMATH_GPT_common_value_of_7a_and_2b_l1357_135731

variable (a b : ℝ)

theorem common_value_of_7a_and_2b (h1 : 7 * a = 2 * b) (h2 : 42 * a * b = 674.9999999999999) :
  7 * a = 15 :=
by
  -- This place will contain the proof steps
  sorry

end NUMINAMATH_GPT_common_value_of_7a_and_2b_l1357_135731


namespace NUMINAMATH_GPT_projectile_reaches_49_first_time_at_1_point_4_l1357_135752

-- Define the equation for the height of the projectile
def height (t : ℝ) : ℝ := -20 * t^2 + 100 * t

-- State the theorem to prove
theorem projectile_reaches_49_first_time_at_1_point_4 :
  ∃ t : ℝ, height t = 49 ∧ (∀ t' : ℝ, height t' = 49 → t ≤ t') :=
sorry

end NUMINAMATH_GPT_projectile_reaches_49_first_time_at_1_point_4_l1357_135752


namespace NUMINAMATH_GPT_sum_of_digits_of_special_number_l1357_135721

theorem sum_of_digits_of_special_number :
  ∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9 ∧ (100 * x + 10 * y + z = x.factorial + y.factorial + z.factorial) →
  (x + y + z = 10) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_digits_of_special_number_l1357_135721


namespace NUMINAMATH_GPT_fraction_multiplication_l1357_135706

theorem fraction_multiplication :
  (2 / (3 : ℚ)) * (4 / 7) * (5 / 9) * (11 / 13) = 440 / 2457 :=
by
  sorry

end NUMINAMATH_GPT_fraction_multiplication_l1357_135706


namespace NUMINAMATH_GPT_cylinder_original_radius_l1357_135794

theorem cylinder_original_radius 
  (r h : ℝ) 
  (hr_eq : h = 3)
  (volume_increase_radius : Real.pi * (r + 8)^2 * 3 = Real.pi * r^2 * 11) :
  r = 8 :=
by
  -- the proof steps will be here
  sorry

end NUMINAMATH_GPT_cylinder_original_radius_l1357_135794


namespace NUMINAMATH_GPT_ratio_of_areas_is_one_ninth_l1357_135713

-- Define the side lengths of Square A and Square B
variables (x : ℝ)
def side_length_a := x
def side_length_b := 3 * x

-- Define the areas of Square A and Square B
def area_a := side_length_a x * side_length_a x
def area_b := side_length_b x * side_length_b x

-- The theorem to prove the ratio of areas
theorem ratio_of_areas_is_one_ninth : (area_a x) / (area_b x) = (1 / 9) :=
by sorry

end NUMINAMATH_GPT_ratio_of_areas_is_one_ninth_l1357_135713


namespace NUMINAMATH_GPT_complex_magnitude_problem_l1357_135734

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_complex_magnitude_problem_l1357_135734


namespace NUMINAMATH_GPT_avg_speed_is_65_l1357_135711

theorem avg_speed_is_65
  (speed1: ℕ) (speed2: ℕ) (time1: ℕ) (time2: ℕ)
  (h_speed1: speed1 = 85)
  (h_speed2: speed2 = 45)
  (h_time1: time1 = 1)
  (h_time2: time2 = 1) :
  (speed1 + speed2) / (time1 + time2) = 65 := by
  sorry

end NUMINAMATH_GPT_avg_speed_is_65_l1357_135711


namespace NUMINAMATH_GPT_find_c1_in_polynomial_q_l1357_135700

theorem find_c1_in_polynomial_q
  (m : ℕ)
  (hm : m ≥ 5)
  (hm_odd : m % 2 = 1)
  (D : ℕ → ℕ)
  (hD_q : ∃ (c3 c2 c1 c0 : ℤ), ∀ (m : ℕ), m % 2 = 1 ∧ m ≥ 5 → D m = (c3 * m^3 + c2 * m^2 + c1 * m + c0)) :
  ∃ (c1 : ℤ), c1 = 11 :=
sorry

end NUMINAMATH_GPT_find_c1_in_polynomial_q_l1357_135700


namespace NUMINAMATH_GPT_joanie_loan_difference_l1357_135733

theorem joanie_loan_difference:
  let P := 6000
  let r := 0.12
  let t := 4
  let n_quarterly := 4
  let n_annually := 1
  let A_quarterly := P * (1 + r / n_quarterly)^(n_quarterly * t)
  let A_annually := P * (1 + r / n_annually)^t
  A_quarterly - A_annually = 187.12 := sorry

end NUMINAMATH_GPT_joanie_loan_difference_l1357_135733


namespace NUMINAMATH_GPT_john_chips_consumption_l1357_135744

/-- John starts the week with a routine. Every day, he eats one bag of chips for breakfast,
  two bags for lunch, and doubles the amount he had for lunch for dinner.
  Prove that by the end of the week, John consumed 49 bags of chips. --/
theorem john_chips_consumption : 
  ∀ (days_in_week : ℕ) (chips_breakfast : ℕ) (chips_lunch : ℕ) (chips_dinner : ℕ), 
    days_in_week = 7 ∧ chips_breakfast = 1 ∧ chips_lunch = 2 ∧ chips_dinner = 2 * chips_lunch →
    days_in_week * (chips_breakfast + chips_lunch + chips_dinner) = 49 :=
by
  intros days_in_week chips_breakfast chips_lunch chips_dinner
  sorry

end NUMINAMATH_GPT_john_chips_consumption_l1357_135744


namespace NUMINAMATH_GPT_children_in_circle_l1357_135712

theorem children_in_circle (n m : ℕ) (k : ℕ) 
  (h1 : n = m) 
  (h2 : n + m = 2 * k) :
  ∃ k', n + m = 4 * k' :=
by
  sorry

end NUMINAMATH_GPT_children_in_circle_l1357_135712


namespace NUMINAMATH_GPT_alyssa_puppies_l1357_135789

theorem alyssa_puppies (initial now given : ℕ) (h1 : initial = 12) (h2 : now = 5) : given = 7 :=
by
  have h3 : given = initial - now := by sorry
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_alyssa_puppies_l1357_135789


namespace NUMINAMATH_GPT_points_enclosed_in_circle_l1357_135748

open Set

variable (points : Set (ℝ × ℝ))
variable (radius : ℝ)
variable (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points → 
  ∃ (c : ℝ × ℝ), dist c A ≤ radius ∧ dist c B ≤ radius ∧ dist c C ≤ radius)

theorem points_enclosed_in_circle
  (h1 : ∀ (A B C : ℝ × ℝ), A ∈ points → B ∈ points → C ∈ points →
    ∃ (c : ℝ × ℝ), dist c A ≤ 1 ∧ dist c B ≤ 1 ∧ dist c C ≤ 1) :
  ∃ (c : ℝ × ℝ), ∀ (p : ℝ × ℝ), p ∈ points → dist c p ≤ 1 :=
sorry

end NUMINAMATH_GPT_points_enclosed_in_circle_l1357_135748


namespace NUMINAMATH_GPT_arithmetic_evaluation_l1357_135776

theorem arithmetic_evaluation : 8 + 18 / 3 - 4 * 2 = 6 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_evaluation_l1357_135776


namespace NUMINAMATH_GPT_total_animals_in_farm_l1357_135779

theorem total_animals_in_farm (C B : ℕ) (h1 : C = 5) (h2 : 2 * C + 4 * B = 26) : C + B = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_animals_in_farm_l1357_135779


namespace NUMINAMATH_GPT_parabola_opens_upward_l1357_135728

theorem parabola_opens_upward (a : ℝ) (b : ℝ) (h : a > 0) : ∀ x : ℝ, 3*x^2 + 2 = a*x^2 + b → a = 3 ∧ b = 2 → ∀ x : ℝ, 3 * x^2 + 2 ≤ a * x^2 + b := 
by
  sorry

end NUMINAMATH_GPT_parabola_opens_upward_l1357_135728


namespace NUMINAMATH_GPT_ratio_singers_joined_second_to_remaining_first_l1357_135737

-- Conditions
def total_singers : ℕ := 30
def singers_first_verse : ℕ := total_singers / 2
def remaining_after_first : ℕ := total_singers - singers_first_verse
def singers_joined_third_verse : ℕ := 10
def all_singing : ℕ := total_singers

-- Definition for singers who joined in the second verse
def singers_joined_second_verse : ℕ := all_singing - singers_joined_third_verse - singers_first_verse

-- The target proof
theorem ratio_singers_joined_second_to_remaining_first :
  (singers_joined_second_verse : ℚ) / remaining_after_first = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_singers_joined_second_to_remaining_first_l1357_135737


namespace NUMINAMATH_GPT_system_of_equations_solution_l1357_135778

theorem system_of_equations_solution :
  ∃ x y : ℚ, (4 * x - 3 * y = -8) ∧ (5 * x + 9 * y = -18) ∧ x = -14 / 3 ∧ y = -32 / 9 :=
by {
  sorry  -- Proof goes here
}

end NUMINAMATH_GPT_system_of_equations_solution_l1357_135778


namespace NUMINAMATH_GPT_abc_div_def_eq_1_div_20_l1357_135757

-- Definitions
variables (a b c d e f : ℝ)

-- Conditions
axiom condition1 : a / b = 1 / 3
axiom condition2 : b / c = 2
axiom condition3 : c / d = 1 / 2
axiom condition4 : d / e = 3
axiom condition5 : e / f = 1 / 10

-- Proof statement
theorem abc_div_def_eq_1_div_20 : (a * b * c) / (d * e * f) = 1 / 20 :=
by 
  -- The actual proof is omitted, as the problem only requires the statement.
  sorry

end NUMINAMATH_GPT_abc_div_def_eq_1_div_20_l1357_135757


namespace NUMINAMATH_GPT_frac_two_over_x_values_l1357_135707

theorem frac_two_over_x_values (x : ℝ) (h : 1 - 9 / x + 20 / (x ^ 2) = 0) :
  (2 / x = 1 / 2 ∨ 2 / x = 0.4) :=
sorry

end NUMINAMATH_GPT_frac_two_over_x_values_l1357_135707


namespace NUMINAMATH_GPT_sixty_percent_of_N_l1357_135787

noncomputable def N : ℝ :=
  let x := (45 : ℝ)
  let frac := (3/4 : ℝ) * (1/3) * (2/5) * (1/2)
  20 * x / frac

theorem sixty_percent_of_N : (0.60 : ℝ) * N = 540 := by
  sorry

end NUMINAMATH_GPT_sixty_percent_of_N_l1357_135787


namespace NUMINAMATH_GPT_system_solution_unique_l1357_135799

theorem system_solution_unique : 
  ∀ (x y z : ℝ),
  (4 * x^2) / (1 + 4 * x^2) = y ∧
  (4 * y^2) / (1 + 4 * y^2) = z ∧
  (4 * z^2) / (1 + 4 * z^2) = x 
  → (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end NUMINAMATH_GPT_system_solution_unique_l1357_135799


namespace NUMINAMATH_GPT_new_interest_rate_l1357_135756

theorem new_interest_rate 
    (i₁ : ℝ) (r₁ : ℝ) (p : ℝ) (additional_interest : ℝ) (i₂ : ℝ) (r₂ : ℝ)
    (h1 : r₁ = 0.05)
    (h2 : i₁ = 101.20)
    (h3 : additional_interest = 20.24)
    (h4 : i₂ = i₁ + additional_interest)
    (h5 : p = i₁ / (r₁ * 1))
    (h6 : i₂ = p * r₂ * 1) :
  r₂ = 0.06 :=
by
  sorry

end NUMINAMATH_GPT_new_interest_rate_l1357_135756


namespace NUMINAMATH_GPT_tom_to_luke_ratio_l1357_135795

theorem tom_to_luke_ratio (Tom Luke Anthony : ℕ) 
  (hAnthony : Anthony = 44) 
  (hTom : Tom = 33) 
  (hLuke : Luke = Anthony / 4) : 
  Tom / Nat.gcd Tom Luke = 3 ∧ Luke / Nat.gcd Tom Luke = 1 := 
by
  sorry

end NUMINAMATH_GPT_tom_to_luke_ratio_l1357_135795


namespace NUMINAMATH_GPT_carol_packs_l1357_135791

theorem carol_packs (invitations_per_pack total_invitations packs_bought : ℕ) 
  (h1 : invitations_per_pack = 9)
  (h2 : total_invitations = 45) 
  (h3 : packs_bought = total_invitations / invitations_per_pack) : 
  packs_bought = 5 :=
by 
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end NUMINAMATH_GPT_carol_packs_l1357_135791


namespace NUMINAMATH_GPT_yard_length_l1357_135753

theorem yard_length (n : ℕ) (d : ℕ) (k : ℕ) (h : k = n - 1) (hd : d = 5) (hn : n = 51) : (k * d) = 250 := 
by
  sorry

end NUMINAMATH_GPT_yard_length_l1357_135753


namespace NUMINAMATH_GPT_x_value_when_y_2000_l1357_135760

noncomputable def x_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) : ℝ :=
  if hy : y = 2000 then (1 / (50 : ℝ)^(1/3)) else x

-- Theorem statement
theorem x_value_when_y_2000 (x y : ℝ) (hxy_pos : 0 < x ∧ 0 < y) (hxy_inv : ∀ x' y', x'^3 * y' = x^3 * y) (h_init : x = 2 ∧ y = 5) :
  x_when_y_2000 x y hxy_pos hxy_inv h_init = 1 / (50 : ℝ)^(1/3) :=
sorry

end NUMINAMATH_GPT_x_value_when_y_2000_l1357_135760


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l1357_135715

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℤ) 
  (q : ℤ) 
  (h1 : a 1 + a 3 = 10) 
  (h2 : a 4 + a 6 = 1 / 4) : 
  q = 1 / 2 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l1357_135715


namespace NUMINAMATH_GPT_car_travel_distance_l1357_135718

-- Definitions based on the conditions
def car_speed : ℕ := 60  -- The actual speed of the car
def faster_speed : ℕ := car_speed + 30  -- Speed if the car traveled 30 km/h faster
def time_difference : ℚ := 0.5  -- 30 minutes less in hours

-- The distance D we need to prove
def distance_traveled : ℚ := 90

-- Main statement to be proven
theorem car_travel_distance : ∀ (D : ℚ),
  (D / car_speed) = (D / faster_speed) + time_difference →
  D = distance_traveled :=
by
  intros D h
  sorry

end NUMINAMATH_GPT_car_travel_distance_l1357_135718


namespace NUMINAMATH_GPT_walking_rate_on_escalator_l1357_135735

/-- If the escalator moves at 7 feet per second, is 180 feet long, and a person takes 20 seconds to cover this length, then the rate at which the person walks on the escalator is 2 feet per second. -/
theorem walking_rate_on_escalator 
  (escalator_rate : ℝ)
  (length : ℝ)
  (time : ℝ)
  (v : ℝ)
  (h_escalator_rate : escalator_rate = 7)
  (h_length : length = 180)
  (h_time : time = 20)
  (h_distance_formula : length = (v + escalator_rate) * time) :
  v = 2 :=
by
  sorry

end NUMINAMATH_GPT_walking_rate_on_escalator_l1357_135735


namespace NUMINAMATH_GPT_milk_tea_sales_l1357_135762

-- Definitions
def relationship (x y : ℕ) : Prop := y = 10 * x + 2

-- Theorem statement
theorem milk_tea_sales (x y : ℕ) :
  relationship x y → (y = 822 → x = 82) :=
by
  intros h_rel h_y
  sorry

end NUMINAMATH_GPT_milk_tea_sales_l1357_135762


namespace NUMINAMATH_GPT_smallest_value_N_l1357_135746

theorem smallest_value_N (N : ℕ) (a b c : ℕ) (h1 : N = a * b * c) (h2 : (a - 1) * (b - 1) * (c - 1) = 252) : N = 392 :=
sorry

end NUMINAMATH_GPT_smallest_value_N_l1357_135746


namespace NUMINAMATH_GPT_range_of_m_l1357_135701

theorem range_of_m (m : ℝ) :
  (∃ x0 : ℝ, m * x0^2 + 1 ≤ 0) ∧ (∀ x : ℝ, x^2 + m * x + 1 > 0) → -2 < m ∧ m < 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1357_135701


namespace NUMINAMATH_GPT_last_integer_in_sequence_div3_l1357_135729

theorem last_integer_in_sequence_div3 (a0 : ℤ) (sequence : ℕ → ℤ)
  (h0 : a0 = 1000000000)
  (h_seq : ∀ n, sequence n = a0 / (3^n)) :
  ∃ k, sequence k = 2 ∧ ∀ m, sequence m < 2 → sequence m < 1 := 
sorry

end NUMINAMATH_GPT_last_integer_in_sequence_div3_l1357_135729


namespace NUMINAMATH_GPT_anne_gave_sweettarts_to_three_friends_l1357_135780

theorem anne_gave_sweettarts_to_three_friends (sweettarts : ℕ) (eaten : ℕ) (friends : ℕ) 
  (h1 : sweettarts = 15) (h2 : eaten = 5) (h3 : sweettarts = friends * eaten) :
  friends = 3 := 
by 
  sorry

end NUMINAMATH_GPT_anne_gave_sweettarts_to_three_friends_l1357_135780


namespace NUMINAMATH_GPT_cylinder_volume_increase_l1357_135797

theorem cylinder_volume_increase 
  (r h : ℝ) 
  (V : ℝ := π * r^2 * h) 
  (new_h : ℝ := 3 * h) 
  (new_r : ℝ := 2 * r) : 
  (π * new_r^2 * new_h) = 12 * V := 
by
  sorry

end NUMINAMATH_GPT_cylinder_volume_increase_l1357_135797


namespace NUMINAMATH_GPT_trigonometric_expression_value_l1357_135722

variable {α : ℝ}
axiom tan_alpha_eq : Real.tan α = 2

theorem trigonometric_expression_value :
  (1 + 2 * Real.cos (Real.pi / 2 - α) * Real.cos (-10 * Real.pi - α)) /
  (Real.cos (3 * Real.pi / 2 - α) ^ 2 - Real.sin (9 * Real.pi / 2 - α) ^ 2) = 3 :=
by
  have h_tan_alpha : Real.tan α = 2 := tan_alpha_eq
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l1357_135722


namespace NUMINAMATH_GPT_hawks_points_l1357_135775

theorem hawks_points (x y z : ℤ) 
  (h_total_points: x + y = 82)
  (h_margin: x - y = 18)
  (h_eagles_points: x = 12 + z) : 
  y = 32 := 
sorry

end NUMINAMATH_GPT_hawks_points_l1357_135775


namespace NUMINAMATH_GPT_both_complementary_angles_acute_is_certain_event_l1357_135730

def complementary_angles (A B : ℝ) : Prop :=
  A + B = 90

def acute_angle (θ : ℝ) : Prop :=
  0 < θ ∧ θ < 90

theorem both_complementary_angles_acute_is_certain_event (A B : ℝ) (h1 : complementary_angles A B) (h2 : acute_angle A) (h3 : acute_angle B) : (A < 90) ∧ (B < 90) :=
by
  sorry

end NUMINAMATH_GPT_both_complementary_angles_acute_is_certain_event_l1357_135730


namespace NUMINAMATH_GPT_calculate_dividend_l1357_135742

def faceValue : ℕ := 100
def premiumPercent : ℕ := 20
def dividendPercent : ℕ := 5
def investment : ℕ := 14400
def costPerShare : ℕ := faceValue + (premiumPercent * faceValue / 100)
def numberOfShares : ℕ := investment / costPerShare
def dividendPerShare : ℕ := faceValue * dividendPercent / 100
def totalDividend : ℕ := numberOfShares * dividendPerShare

theorem calculate_dividend :
  totalDividend = 600 := 
by
  sorry

end NUMINAMATH_GPT_calculate_dividend_l1357_135742


namespace NUMINAMATH_GPT_suraj_average_after_9th_innings_l1357_135774

theorem suraj_average_after_9th_innings (A : ℕ) 
  (h1 : 8 * A + 90 = 9 * (A + 6)) : 
  (A + 6) = 42 :=
by
  sorry

end NUMINAMATH_GPT_suraj_average_after_9th_innings_l1357_135774


namespace NUMINAMATH_GPT_min_f_abs_l1357_135763

def f (x y : ℤ) : ℤ := 5 * x^2 + 11 * x * y - 5 * y^2

theorem min_f_abs (x y : ℤ) (h : x ≠ 0 ∨ y ≠ 0) : (∃ m, ∀ x y : ℤ, (x ≠ 0 ∨ y ≠ 0) → |f x y| ≥ m) ∧ 5 = 5 :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_min_f_abs_l1357_135763


namespace NUMINAMATH_GPT_a_5_eq_neg1_l1357_135758

-- Given conditions
def S (n : ℕ) : ℤ := n^2 - 10 * n

def a (n : ℕ) : ℤ := S n - S (n - 1)

-- The theorem to prove
theorem a_5_eq_neg1 : a 5 = -1 :=
by sorry

end NUMINAMATH_GPT_a_5_eq_neg1_l1357_135758


namespace NUMINAMATH_GPT_additional_payment_each_friend_l1357_135717

theorem additional_payment_each_friend (initial_cost : ℕ) (earned_amount : ℕ) (total_friends : ℕ) (final_friends : ℕ) 
(h_initial_cost : initial_cost = 1700) (h_earned_amount : earned_amount = 500) 
(h_total_friends : total_friends = 6) (h_final_friends : final_friends = 5) : 
  ((initial_cost - earned_amount) / total_friends) / final_friends = 40 :=
sorry

end NUMINAMATH_GPT_additional_payment_each_friend_l1357_135717


namespace NUMINAMATH_GPT_union_of_A_and_B_l1357_135702

def setA : Set ℝ := { x | -3 ≤ 2 * x - 1 ∧ 2 * x - 1 ≤ 3 }
def setB : Set ℝ := { x | 1 < x }

theorem union_of_A_and_B :
  setA ∪ setB = { x | -1 ≤ x } := sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1357_135702


namespace NUMINAMATH_GPT_fraction_product_correct_l1357_135732

theorem fraction_product_correct : (3 / 5) * (4 / 7) * (5 / 9) = 4 / 21 :=
by
  sorry

end NUMINAMATH_GPT_fraction_product_correct_l1357_135732


namespace NUMINAMATH_GPT_number_of_pens_bought_l1357_135724

theorem number_of_pens_bought 
  (P : ℝ) -- Marked price of one pen
  (N : ℝ) -- Number of pens bought
  (discount : ℝ := 0.01)
  (profit_percent : ℝ := 29.130434782608695)
  (Total_Cost := 46 * P)
  (Selling_Price_per_Pen := P * (1 - discount))
  (Total_Revenue := N * Selling_Price_per_Pen)
  (Profit := Total_Revenue - Total_Cost)
  (actual_profit_percent := (Profit / Total_Cost) * 100) :
  actual_profit_percent = profit_percent → N = 60 := 
by 
  intro h
  sorry

end NUMINAMATH_GPT_number_of_pens_bought_l1357_135724


namespace NUMINAMATH_GPT_system_unique_solution_l1357_135709

theorem system_unique_solution 
  (x y z : ℝ) 
  (h1 : x + y + z = 3 * x * y) 
  (h2 : x^2 + y^2 + z^2 = 3 * x * z) 
  (h3 : x^3 + y^3 + z^3 = 3 * y * z) 
  (hx : 0 ≤ x) 
  (hy : 0 ≤ y) 
  (hz : 0 ≤ z) : 
  (x = 1 ∧ y = 1 ∧ z = 1) := 
sorry

end NUMINAMATH_GPT_system_unique_solution_l1357_135709


namespace NUMINAMATH_GPT_jacket_initial_reduction_percent_l1357_135766

theorem jacket_initial_reduction_percent (P : ℝ) (x : ℝ) (h : P * (1 - x / 100) * 0.70 * 1.5873 = P) : x = 10 :=
sorry

end NUMINAMATH_GPT_jacket_initial_reduction_percent_l1357_135766


namespace NUMINAMATH_GPT_bookstore_shoe_store_common_sales_l1357_135705

-- Define the conditions
def bookstore_sale_days (d: ℕ) : Prop := d % 4 = 0 ∧ d >= 4 ∧ d <= 28
def shoe_store_sale_days (d: ℕ) : Prop := (d - 2) % 6 = 0 ∧ d >= 2 ∧ d <= 26

-- Define the question to be proven as a theorem
theorem bookstore_shoe_store_common_sales : 
  ∃ (n: ℕ), n = 2 ∧ (
    ∀ (d: ℕ), 
      ((bookstore_sale_days d ∧ shoe_store_sale_days d) → n = 2) 
      ∧ (d < 4 ∨ d > 28 ∨ d < 2 ∨ d > 26 → n = 2)
  ) :=
sorry

end NUMINAMATH_GPT_bookstore_shoe_store_common_sales_l1357_135705


namespace NUMINAMATH_GPT_problem_1_problem_2_l1357_135768

-- Problem 1 Lean statement
theorem problem_1 :
  (1 - 1^4 - (1/2) * (3 - (-3)^2)) = 2 :=
by sorry

-- Problem 2 Lean statement
theorem problem_2 :
  ((3/8 - 1/6 - 3/4) * 24) = -13 :=
by sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1357_135768


namespace NUMINAMATH_GPT_fraction_of_green_balls_l1357_135793

theorem fraction_of_green_balls (T G : ℝ)
    (h1 : (1 / 8) * T = 6)
    (h2 : (1 / 12) * T + (1 / 8) * T + 26 = T - G)
    (h3 : (1 / 8) * T = 6)
    (h4 : 26 ≥ 0):
  G / T = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_green_balls_l1357_135793


namespace NUMINAMATH_GPT_smallest_value_of_2a_plus_1_l1357_135741

theorem smallest_value_of_2a_plus_1 (a : ℝ) 
  (h : 6 * a^2 + 5 * a + 4 = 3) : 
  ∃ b : ℝ, b = 2 * a + 1 ∧ b = 0 := 
sorry

end NUMINAMATH_GPT_smallest_value_of_2a_plus_1_l1357_135741


namespace NUMINAMATH_GPT_sum_medians_less_than_perimeter_l1357_135720

noncomputable def median_a (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * b^2 + 2 * c^2 - a^2).sqrt

noncomputable def median_b (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * c^2 - b^2).sqrt

noncomputable def median_c (a b c : ℝ) : ℝ :=
  (1 / 2) * (2 * a^2 + 2 * b^2 - c^2).sqrt

noncomputable def sum_of_medians (a b c : ℝ) : ℝ :=
  median_a a b c + median_b a b c + median_c a b c

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

noncomputable def semiperimeter (a b c : ℝ) : ℝ :=
  perimeter a b c / 2

theorem sum_medians_less_than_perimeter (a b c : ℝ) :
  semiperimeter a b c < sum_of_medians a b c ∧ sum_of_medians a b c < perimeter a b c :=
by
  sorry

end NUMINAMATH_GPT_sum_medians_less_than_perimeter_l1357_135720


namespace NUMINAMATH_GPT_problem1_problem2_l1357_135773

open Nat

def binomial (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem problem1 : binomial 8 5 + binomial 100 98 * binomial 7 7 = 5006 := by
  sorry

theorem problem2 : binomial 5 0 + binomial 5 1 + binomial 5 2 + binomial 5 3 + binomial 5 4 + binomial 5 5 = 32 := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_l1357_135773


namespace NUMINAMATH_GPT_prism_faces_l1357_135781

-- Define the number of edges in a prism and the function to calculate faces based on edges.
def number_of_faces (E : ℕ) : ℕ :=
  if E % 3 = 0 then (E / 3) + 2 else 0

-- The main statement to be proven: A prism with 18 edges has 8 faces.
theorem prism_faces (E : ℕ) (hE : E = 18) : number_of_faces E = 8 :=
by
  -- Just outline the argument here for clarity, the detailed steps will go in an actual proof.
  sorry

end NUMINAMATH_GPT_prism_faces_l1357_135781


namespace NUMINAMATH_GPT_MarysTotalCandies_l1357_135777

-- Definitions for the conditions
def MegansCandies : Nat := 5
def MarysInitialCandies : Nat := 3 * MegansCandies
def MarysCandiesAfterAdding : Nat := MarysInitialCandies + 10

-- Theorem to prove that Mary has 25 pieces of candy in total
theorem MarysTotalCandies : MarysCandiesAfterAdding = 25 :=
by
  sorry

end NUMINAMATH_GPT_MarysTotalCandies_l1357_135777


namespace NUMINAMATH_GPT_number_of_chairs_borrowed_l1357_135704

-- Define the conditions
def red_chairs := 4
def yellow_chairs := 2 * red_chairs
def blue_chairs := yellow_chairs - 2
def total_initial_chairs : Nat := red_chairs + yellow_chairs + blue_chairs
def chairs_left_in_the_afternoon := 15

-- Define the question
def chairs_borrowed_by_Lisa : Nat := total_initial_chairs - chairs_left_in_the_afternoon

-- The theorem to state the proof problem
theorem number_of_chairs_borrowed : chairs_borrowed_by_Lisa = 3 := by
  -- Proof to be added
  sorry

end NUMINAMATH_GPT_number_of_chairs_borrowed_l1357_135704


namespace NUMINAMATH_GPT_work_completion_time_l1357_135740

-- Let's define the initial conditions
def total_days := 100
def initial_people := 10
def days1 := 20
def work_done1 := 1 / 4
def days2 (remaining_work_per_person: ℚ) := (3/4) / remaining_work_per_person
def remaining_people := initial_people - 2
def remaining_work_per_person_per_day := remaining_people * (work_done1 / (initial_people * days1))

-- Theorem stating that the total number of days to complete the work is 95
theorem work_completion_time : 
  days1 + days2 remaining_work_per_person_per_day = 95 := 
  by
    sorry -- Proof to be filled in

end NUMINAMATH_GPT_work_completion_time_l1357_135740


namespace NUMINAMATH_GPT_positive_sequence_unique_l1357_135747

theorem positive_sequence_unique (x : Fin 2021 → ℝ) (h : ∀ i : Fin 2020, x i.succ = (x i ^ 3 + 2) / (3 * x i ^ 2)) (h' : x 2020 = x 0) : ∀ i, x i = 1 := by
  sorry

end NUMINAMATH_GPT_positive_sequence_unique_l1357_135747


namespace NUMINAMATH_GPT_sufficiency_but_not_necessary_l1357_135743

theorem sufficiency_but_not_necessary (x y : ℝ) : |x| + |y| ≤ 1 → x^2 + y^2 ≤ 1 ∧ ¬(x^2 + y^2 ≤ 1 → |x| + |y| ≤ 1) :=
by
  sorry

end NUMINAMATH_GPT_sufficiency_but_not_necessary_l1357_135743


namespace NUMINAMATH_GPT_breadth_of_rectangular_plot_l1357_135769

theorem breadth_of_rectangular_plot (b : ℝ) (A : ℝ) (l : ℝ)
  (h1 : A = 20 * b)
  (h2 : l = b + 10)
  (h3 : A = l * b) : b = 10 := by
  sorry

end NUMINAMATH_GPT_breadth_of_rectangular_plot_l1357_135769


namespace NUMINAMATH_GPT_number_of_petri_dishes_l1357_135771

noncomputable def total_germs : ℝ := 0.036 * 10^5
noncomputable def germs_per_dish : ℝ := 99.99999999999999

theorem number_of_petri_dishes : 36 = total_germs / germs_per_dish :=
by sorry

end NUMINAMATH_GPT_number_of_petri_dishes_l1357_135771


namespace NUMINAMATH_GPT_parabola_translation_eq_l1357_135703

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := -x^2 + 2

-- Define the translated parabola function
def translated_parabola (x : ℝ) : ℝ := - (x - 2)^2 - 1

-- State the theorem to prove the translated function
theorem parabola_translation_eq :
  ∀ x : ℝ, translated_parabola x = - (x - 2)^2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_parabola_translation_eq_l1357_135703


namespace NUMINAMATH_GPT_M_intersection_N_eq_N_l1357_135792

def M := { x : ℝ | x < 4 }
def N := { x : ℝ | x ≤ -2 }

theorem M_intersection_N_eq_N : M ∩ N = N :=
by
  sorry

end NUMINAMATH_GPT_M_intersection_N_eq_N_l1357_135792


namespace NUMINAMATH_GPT_find_square_subtraction_l1357_135738

theorem find_square_subtraction (x y : ℝ) (h1 : x = Real.sqrt 5) (h2 : y = Real.sqrt 2) : (x - y)^2 = 7 - 2 * Real.sqrt 10 :=
by
  sorry

end NUMINAMATH_GPT_find_square_subtraction_l1357_135738


namespace NUMINAMATH_GPT_sequence_general_formula_l1357_135745

theorem sequence_general_formula (a : ℕ+ → ℝ) (h₀ : a 1 = 7 / 8)
  (h₁ : ∀ n : ℕ+, a (n + 1) = 1 / 2 * a n + 1 / 3) :
  ∀ n : ℕ+, a n = 5 / 24 * (1 / 2)^(n - 1 : ℕ) + 2 / 3 :=
by
  sorry

end NUMINAMATH_GPT_sequence_general_formula_l1357_135745


namespace NUMINAMATH_GPT_sequence_induction_l1357_135764

theorem sequence_induction (a b : ℕ → ℕ)
  (h₁ : a 1 = 2)
  (h₂ : b 1 = 4)
  (h₃ : ∀ n : ℕ, 0 < n → 2 * b n = a n + a (n + 1))
  (h₄ : ∀ n : ℕ, 0 < n → (a (n + 1))^2 = b n * b (n + 1)) :
  (∀ n : ℕ, 0 < n → a n = n * (n + 1)) ∧ (∀ n : ℕ, 0 < n → b n = (n + 1)^2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_induction_l1357_135764


namespace NUMINAMATH_GPT_problem_solution_l1357_135719

-- Lean 4 statement of the proof problem
theorem problem_solution (m : ℝ) (U : Set ℝ := Univ) (A : Set ℝ := {x | x^2 + 3*x + 2 = 0}) 
  (B : Set ℝ := {x | x^2 + (m + 1)*x + m = 0}) (h : ∀ x, x ∈ (U \ A) → x ∉ B) : 
  m = 1 ∨ m = 2 :=
by 
  -- This is where the proof would normally go
  sorry

end NUMINAMATH_GPT_problem_solution_l1357_135719


namespace NUMINAMATH_GPT_bucket_P_turns_to_fill_the_drum_l1357_135783

-- Define the capacities of the buckets
def capacity_P := 3
def capacity_Q := 1

-- Define the total number of turns for both buckets together to fill the drum
def turns_together := 60

-- Define the total capacity of the drum that gets filled in the given scenario of the problem
def total_capacity := turns_together * (capacity_P + capacity_Q)

-- The question: How many turns does it take for bucket P alone to fill this total capacity?
def turns_P_alone : ℕ :=
  total_capacity / capacity_P

theorem bucket_P_turns_to_fill_the_drum :
  turns_P_alone = 80 :=
by
  sorry

end NUMINAMATH_GPT_bucket_P_turns_to_fill_the_drum_l1357_135783


namespace NUMINAMATH_GPT_square_assembly_possible_l1357_135708

theorem square_assembly_possible (Area1 Area2 Area3 : ℕ) (h1 : Area1 = 29) (h2 : Area2 = 18) (h3 : Area3 = 10) (h_total : Area1 + Area2 + Area3 = 57) : 
  ∃ s : ℝ, s^2 = 57 ∧ true :=
by
  sorry

end NUMINAMATH_GPT_square_assembly_possible_l1357_135708


namespace NUMINAMATH_GPT_integer_a_values_l1357_135755

theorem integer_a_values (a : ℤ) :
  (∃ x : ℤ, x^3 + 3 * x^2 + a * x - 7 = 0) ↔ a = -70 ∨ a = -29 ∨ a = -5 ∨ a = 3 :=
by
  sorry

end NUMINAMATH_GPT_integer_a_values_l1357_135755


namespace NUMINAMATH_GPT_percentage_increase_l1357_135736

def originalPrice : ℝ := 300
def newPrice : ℝ := 390

theorem percentage_increase :
  ((newPrice - originalPrice) / originalPrice) * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1357_135736


namespace NUMINAMATH_GPT_verify_system_of_equations_l1357_135772

/-- Define a structure to hold the conditions of the problem -/
structure TreePurchasing :=
  (cost_A : ℕ)
  (cost_B : ℕ)
  (diff_A_B : ℕ)
  (total_cost : ℕ)
  (x : ℕ)
  (y : ℕ)

/-- Given conditions for purchasing trees -/
def example_problem : TreePurchasing :=
  { cost_A := 100,
    cost_B := 80,
    diff_A_B := 8,
    total_cost := 8000,
    x := 0,
    y := 0 }

/-- The theorem to prove that the equations match given conditions -/
theorem verify_system_of_equations (data : TreePurchasing) (h_diff : data.x - data.y = data.diff_A_B) (h_cost : data.cost_A * data.x + data.cost_B * data.y = data.total_cost) : 
  (data.x - data.y = 8) ∧ (100 * data.x + 80 * data.y = 8000) :=
  by
    sorry

end NUMINAMATH_GPT_verify_system_of_equations_l1357_135772


namespace NUMINAMATH_GPT_compute_binomial_sum_l1357_135765

def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem compute_binomial_sum :
  binomial 12 11 + binomial 12 1 = 24 :=
by
  sorry

end NUMINAMATH_GPT_compute_binomial_sum_l1357_135765


namespace NUMINAMATH_GPT_arithmetic_progression_common_difference_zero_l1357_135796

theorem arithmetic_progression_common_difference_zero {a d : ℤ} (h₁ : a = 12) 
  (h₂ : ∀ n : ℕ, a + n * d = (a + (n + 1) * d + a + (n + 2) * d) / 2) : d = 0 :=
  sorry

end NUMINAMATH_GPT_arithmetic_progression_common_difference_zero_l1357_135796


namespace NUMINAMATH_GPT_distance_to_workplace_l1357_135739

def driving_speed : ℕ := 40
def driving_time : ℕ := 3
def total_distance := driving_speed * driving_time
def one_way_distance := total_distance / 2

theorem distance_to_workplace : one_way_distance = 60 := by
  sorry

end NUMINAMATH_GPT_distance_to_workplace_l1357_135739


namespace NUMINAMATH_GPT_probability_of_finding_transmitter_l1357_135782

def total_license_plates : ℕ := 900
def inspected_vehicles : ℕ := 18

theorem probability_of_finding_transmitter : (inspected_vehicles : ℝ) / (total_license_plates : ℝ) = 0.02 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_finding_transmitter_l1357_135782


namespace NUMINAMATH_GPT_greatest_monthly_drop_l1357_135759

-- Definition of monthly price changes
def price_change_jan : ℝ := -1.00
def price_change_feb : ℝ := 2.50
def price_change_mar : ℝ := 0.00
def price_change_apr : ℝ := -3.00
def price_change_may : ℝ := -1.50
def price_change_jun : ℝ := 1.00

-- Proving the month with the greatest monthly drop in price
theorem greatest_monthly_drop :
  (price_change_apr < price_change_jan) ∧
  (price_change_apr < price_change_feb) ∧
  (price_change_apr < price_change_mar) ∧
  (price_change_apr < price_change_may) ∧
  (price_change_apr < price_change_jun) :=
by
  sorry

end NUMINAMATH_GPT_greatest_monthly_drop_l1357_135759


namespace NUMINAMATH_GPT_solve_for_a_b_l1357_135714

open Complex

theorem solve_for_a_b (a b : ℝ) (h : (mk 1 2) / (mk a b) = mk 1 1) : 
  a = 3 / 2 ∧ b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_solve_for_a_b_l1357_135714


namespace NUMINAMATH_GPT_solve_system_l1357_135751

theorem solve_system 
  (x y z : ℝ)
  (h1 : x + 2 * y = 10)
  (h2 : y = 3)
  (h3 : x - 3 * y + z = 7) :
  x = 4 ∧ y = 3 ∧ z = 12 :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1357_135751


namespace NUMINAMATH_GPT_min_val_m_l1357_135784

theorem min_val_m (m n : ℕ) (h_pos_m : m > 0) (h_pos_n : n > 0) (h : 24 * m = n ^ 4) : m = 54 :=
sorry

end NUMINAMATH_GPT_min_val_m_l1357_135784


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_coords_l1357_135725

/--
Cylindrical coordinates (r, θ, z)
Rectangular coordinates (x, y, z)
-/
theorem cylindrical_to_rectangular_coords (r θ z : ℝ) (hx : x = r * Real.cos θ)
    (hy : y = r * Real.sin θ) (hz : z = z) :
    (r, θ, z) = (5, Real.pi / 4, 2) → (x, y, z) = (5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_coords_l1357_135725


namespace NUMINAMATH_GPT_prop1_prop2_prop3_prop4_exists_l1357_135727

variable {R : Type*} [LinearOrderedField R]
def f (b c x : R) : R := abs x * x + b * x + c

theorem prop1 (b c x : R) (h : b > 0) : 
  ∀ {x y : R}, x ≤ y → f b c x ≤ f b c y := 
sorry

theorem prop2 (b c : R) (h : b < 0) : 
  ¬ ∃ a : R, ∀ x : R, f b c x ≥ f b c a := 
sorry

theorem prop3 (b c x : R) : 
  f b c (-x) = f b c x + 2*c := 
sorry

theorem prop4_exists (c : R) : 
  ∃ b : R, ∃ x y z : R, f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z := 
sorry

end NUMINAMATH_GPT_prop1_prop2_prop3_prop4_exists_l1357_135727


namespace NUMINAMATH_GPT_product_evaluation_l1357_135749

theorem product_evaluation :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) = 5^32 - 4^32 :=
by 
sorry

end NUMINAMATH_GPT_product_evaluation_l1357_135749
