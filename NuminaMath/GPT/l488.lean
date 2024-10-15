import Mathlib

namespace NUMINAMATH_GPT_joan_dozen_of_eggs_l488_48836

def number_of_eggs : ℕ := 72
def dozen : ℕ := 12

theorem joan_dozen_of_eggs : (number_of_eggs / dozen) = 6 := by
  sorry

end NUMINAMATH_GPT_joan_dozen_of_eggs_l488_48836


namespace NUMINAMATH_GPT_probability_factor_90_less_than_10_l488_48812

-- Definitions from conditions
def number_factors_90 : ℕ := 12
def factors_90_less_than_10 : ℕ := 6

-- The corresponding proof problem
theorem probability_factor_90_less_than_10 : 
  (factors_90_less_than_10 / number_factors_90 : ℚ) = 1 / 2 :=
by
  sorry  -- proof to be filled in

end NUMINAMATH_GPT_probability_factor_90_less_than_10_l488_48812


namespace NUMINAMATH_GPT_pharmacist_weights_exist_l488_48809

theorem pharmacist_weights_exist :
  ∃ (a b c : ℝ), a + b = 100 ∧ a + c = 101 ∧ b + c = 102 ∧ a < 90 ∧ b < 90 ∧ c < 90 :=
by
  sorry

end NUMINAMATH_GPT_pharmacist_weights_exist_l488_48809


namespace NUMINAMATH_GPT_S9_equals_27_l488_48821

variables {a : ℕ → ℤ} {S : ℕ → ℤ} {d : ℤ}

-- (Condition 1) The sequence is an arithmetic sequence: a_{n+1} = a_n + d
axiom arithmetic_seq : ∀ n : ℕ, a (n + 1) = a n + d

-- (Condition 2) The sum S_n is the sum of the first n terms of the sequence
axiom sum_first_n_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- (Condition 3) Given a_1 = 2 * a_3 - 3
axiom given_condition : a 1 = 2 * a 3 - 3

-- Prove that S_9 = 27
theorem S9_equals_27 : S 9 = 27 :=
by
  sorry

end NUMINAMATH_GPT_S9_equals_27_l488_48821


namespace NUMINAMATH_GPT_zero_point_neg_x₀_l488_48825

-- Define odd function property
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define zero point condition for the function
def is_zero_point (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  f x₀ = Real.exp x₀

-- The main theorem to be proved
theorem zero_point_neg_x₀ (f : ℝ → ℝ) (x₀ : ℝ)
  (h_odd : is_odd_function f)
  (h_zero : is_zero_point f x₀) :
  f (-x₀) * Real.exp x₀ + 1 = 0 :=
sorry

end NUMINAMATH_GPT_zero_point_neg_x₀_l488_48825


namespace NUMINAMATH_GPT_root_quadratic_eq_k_value_l488_48815

theorem root_quadratic_eq_k_value (k : ℤ) :
  (∃ x : ℤ, x = 5 ∧ 2 * x ^ 2 + 3 * x - k = 0) → k = 65 :=
by
  sorry

end NUMINAMATH_GPT_root_quadratic_eq_k_value_l488_48815


namespace NUMINAMATH_GPT_car_b_speed_l488_48854

def speed_of_car_b (Vb Va : ℝ) (tA tB : ℝ) (dist total_dist : ℝ) : Prop :=
  Va = 3 * Vb ∧ tA = 6 ∧ tB = 2 ∧ dist = 1000 ∧ total_dist = Va * tA + Vb * tB

theorem car_b_speed : ∃ Vb Va tA tB dist total_dist, speed_of_car_b Vb Va tA tB dist total_dist ∧ Vb = 50 :=
by
  sorry

end NUMINAMATH_GPT_car_b_speed_l488_48854


namespace NUMINAMATH_GPT_find_ks_l488_48869

theorem find_ks (k : ℕ) : 
  (∃ (m n : ℕ), m > 0 ∧ n > 0 ∧ m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end NUMINAMATH_GPT_find_ks_l488_48869


namespace NUMINAMATH_GPT_seedling_probability_l488_48881

theorem seedling_probability (germination_rate survival_rate : ℝ)
    (h_germ : germination_rate = 0.9) (h_survival : survival_rate = 0.8) : 
    germination_rate * survival_rate = 0.72 :=
by
  rw [h_germ, h_survival]
  norm_num

end NUMINAMATH_GPT_seedling_probability_l488_48881


namespace NUMINAMATH_GPT_range_of_a_l488_48877

variable (a : ℝ)

theorem range_of_a (ha : a ≥ 1/4) : ¬ ∃ x : ℝ, a * x^2 + x + 1 < 0 := sorry

end NUMINAMATH_GPT_range_of_a_l488_48877


namespace NUMINAMATH_GPT_value_of_each_walmart_gift_card_l488_48875

variable (best_buy_value : ℕ) (best_buy_count : ℕ) (walmart_count : ℕ) (points_sent_bb : ℕ) (points_sent_wm : ℕ) (total_returnable : ℕ)

-- Conditions given in the problem
def conditions : Prop :=
  best_buy_value = 500 ∧
  best_buy_count = 6 ∧
  walmart_count = 9 ∧
  points_sent_bb = 1 ∧
  points_sent_wm = 2 ∧
  total_returnable = 3900

-- Result to prove
theorem value_of_each_walmart_gift_card : conditions best_buy_value best_buy_count walmart_count points_sent_bb points_sent_wm total_returnable →
  (total_returnable - ((best_buy_count - points_sent_bb) * best_buy_value)) / (walmart_count - points_sent_wm) = 200 :=
by
  intros h
  rcases h with
    ⟨hbv, hbc, hwc, hsbb, hswm, htr⟩
  sorry

end NUMINAMATH_GPT_value_of_each_walmart_gift_card_l488_48875


namespace NUMINAMATH_GPT_marys_score_l488_48808

def score (c w : ℕ) : ℕ := 30 + 4 * c - w
def valid_score_range (s : ℕ) : Prop := s > 90 ∧ s ≤ 170

theorem marys_score : ∃ c w : ℕ, c + w ≤ 35 ∧ score c w = 170 ∧ 
  ∀ (s : ℕ), (valid_score_range s ∧ ∃ c' w', score c' w' = s ∧ c' + w' ≤ 35) → 
  (s = 170) :=
by
  sorry

end NUMINAMATH_GPT_marys_score_l488_48808


namespace NUMINAMATH_GPT_sequence_positive_from_26_l488_48889

-- Define the sequence with its general term formula.
def a_n (n : ℕ) : ℤ := 4 * n - 102

-- State the theorem that for all n ≥ 26, a_n > 0.
theorem sequence_positive_from_26 (n : ℕ) (h : n ≥ 26) : a_n n > 0 := by
  sorry

end NUMINAMATH_GPT_sequence_positive_from_26_l488_48889


namespace NUMINAMATH_GPT_diet_equivalence_l488_48893

variable (B E L D A : ℕ)

theorem diet_equivalence :
  (17 * B = 170 * L) →
  (100000 * A = 50 * L) →
  (10 * B = 4 * E) →
  12 * E = 600000 * A :=
sorry

end NUMINAMATH_GPT_diet_equivalence_l488_48893


namespace NUMINAMATH_GPT_find_real_numbers_l488_48810

theorem find_real_numbers (x y : ℝ) (h₁ : x + y = 3) (h₂ : x^5 + y^5 = 33) :
  (x = 1 ∧ y = 2) ∨ (x = 2 ∧ y = 1) := by
  sorry

end NUMINAMATH_GPT_find_real_numbers_l488_48810


namespace NUMINAMATH_GPT_correct_factorization_l488_48823

theorem correct_factorization {x y : ℝ} :
  (2 * x ^ 2 - 8 * y ^ 2 = 2 * (x + 2 * y) * (x - 2 * y)) ∧
  ¬(x ^ 2 + 3 * x * y + 9 * y ^ 2 = (x + 3 * y) ^ 2)
    ∧ ¬(2 * x ^ 2 - 4 * x * y + 9 * y ^ 2 = (2 * x - 3 * y) ^ 2)
    ∧ ¬(x * (x - y) + y * (y - x) = (x - y) * (x + y)) := 
by sorry

end NUMINAMATH_GPT_correct_factorization_l488_48823


namespace NUMINAMATH_GPT_frost_time_with_sprained_wrist_l488_48858

-- Definitions
def normal_time_per_cake : ℕ := 5
def additional_time_for_10_cakes : ℕ := 30
def normal_time_for_10_cakes : ℕ := 10 * normal_time_per_cake
def sprained_time_for_10_cakes : ℕ := normal_time_for_10_cakes + additional_time_for_10_cakes

-- Theorems
theorem frost_time_with_sprained_wrist : ∀ x : ℕ, 
  (10 * x = sprained_time_for_10_cakes) ↔ (x = 8) := 
sorry

end NUMINAMATH_GPT_frost_time_with_sprained_wrist_l488_48858


namespace NUMINAMATH_GPT_simplify_expression_l488_48833

variables {a b c : ℝ}
variable (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0)
variable (h₃ : b - 2 / c ≠ 0)

theorem simplify_expression :
  (a - 2 / b) / (b - 2 / c) = c / b :=
sorry

end NUMINAMATH_GPT_simplify_expression_l488_48833


namespace NUMINAMATH_GPT_geometric_progression_solution_l488_48841

theorem geometric_progression_solution (b4 b2 b6 : ℚ) (h1 : b4 - b2 = -45 / 32) (h2 : b6 - b4 = -45 / 512) :
  (∃ (b1 q : ℚ), b4 = b1 * q^3 ∧ b2 = b1 * q ∧ b6 = b1 * q^5 ∧ 
    ((b1 = 6 ∧ q = 1 / 4) ∨ (b1 = -6 ∧ q = -1 / 4))) :=
by
  sorry

end NUMINAMATH_GPT_geometric_progression_solution_l488_48841


namespace NUMINAMATH_GPT_geometric_sequence_condition_l488_48888

theorem geometric_sequence_condition {a : ℕ → ℝ} (h_geom : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) : 
  (a 3 * a 5 = 16) ↔ a 4 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_condition_l488_48888


namespace NUMINAMATH_GPT_intersection_M_N_eq_set_l488_48867

-- Define sets M and N
def M : Set ℝ := {x : ℝ | x^2 < 4}
def N : Set ℝ := {x : ℝ | x^2 - 2*x - 3 < 0}

-- The theorem to be proved
theorem intersection_M_N_eq_set : (M ∩ N) = {x : ℝ | -1 < x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_intersection_M_N_eq_set_l488_48867


namespace NUMINAMATH_GPT_probability_factor_of_36_is_1_over_4_l488_48872

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end NUMINAMATH_GPT_probability_factor_of_36_is_1_over_4_l488_48872


namespace NUMINAMATH_GPT_circular_garden_remaining_grass_area_l488_48884

noncomputable def remaining_grass_area (diameter : ℝ) (path_width: ℝ) : ℝ :=
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let path_area := path_width * diameter
  circle_area - path_area

theorem circular_garden_remaining_grass_area :
  remaining_grass_area 10 2 = 25 * Real.pi - 20 := sorry

end NUMINAMATH_GPT_circular_garden_remaining_grass_area_l488_48884


namespace NUMINAMATH_GPT_distance_from_Q_to_EG_l488_48851

noncomputable def distance_to_line : ℝ :=
  let E := (0, 5)
  let F := (5, 5)
  let G := (5, 0)
  let H := (0, 0)
  let N := (2.5, 0)
  let Q := (25 / 7, 10 / 7)
  let line_y := 5
  let distance := abs (line_y - Q.2)
  distance

theorem distance_from_Q_to_EG : distance_to_line = 25 / 7 :=
by
  sorry

end NUMINAMATH_GPT_distance_from_Q_to_EG_l488_48851


namespace NUMINAMATH_GPT_cubic_polynomial_Q_l488_48890

noncomputable def Q (x : ℝ) : ℝ := 27 * x^3 - 162 * x^2 + 297 * x - 156

theorem cubic_polynomial_Q {a b c : ℝ} 
  (h_roots : ∀ x, x^3 - 6 * x^2 + 11 * x - 6 = 0 → x = a ∨ x = b ∨ x = c)
  (h_vieta_sum : a + b + c = 6)
  (h_vieta_prod_sum : ab + bc + ca = 11)
  (h_vieta_prod : abc = 6)
  (hQ : Q a = b + c) 
  (hQb : Q b = a + c) 
  (hQc : Q c = a + b) 
  (hQ_sum : Q (a + b + c) = -27) :
  Q x = 27 * x^3 - 162 * x^2 + 297 * x - 156 :=
by { sorry }

end NUMINAMATH_GPT_cubic_polynomial_Q_l488_48890


namespace NUMINAMATH_GPT_at_least_one_triangle_l488_48892

theorem at_least_one_triangle {n : ℕ} (h1 : n ≥ 2) (points : Finset ℕ) (segments : Finset (ℕ × ℕ)) : 
(points.card = 2 * n) ∧ (segments.card = n^2 + 1) → 
∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ ((a, b) ∈ segments ∨ (b, a) ∈ segments) ∧ ((b, c) ∈ segments ∨ (c, b) ∈ segments) ∧ ((c, a) ∈ segments ∨ (a, c) ∈ segments) := 
by 
  sorry

end NUMINAMATH_GPT_at_least_one_triangle_l488_48892


namespace NUMINAMATH_GPT_functional_equation_solution_l488_48826

theorem functional_equation_solution (f : ℝ → ℝ) (h : ∀ x y : ℝ, f x * f y = f (x - y)) :
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
sorry

end NUMINAMATH_GPT_functional_equation_solution_l488_48826


namespace NUMINAMATH_GPT_sum_of_four_integers_l488_48847

noncomputable def originalSum (a b c d : ℤ) :=
  (a + b + c + d)

theorem sum_of_four_integers
  (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 8)
  (h2 : (a + b + d) / 3 + c = 12)
  (h3 : (a + c + d) / 3 + b = 32 / 3)
  (h4 : (b + c + d) / 3 + a = 28 / 3) :
  originalSum a b c d = 30 :=
sorry

end NUMINAMATH_GPT_sum_of_four_integers_l488_48847


namespace NUMINAMATH_GPT_taller_tree_height_is_108_l488_48848

variables (H : ℝ)

-- Conditions
def taller_tree_height := H
def shorter_tree_height := H - 18
def ratio_condition := (H - 18) / H = 5 / 6

-- Theorem to prove
theorem taller_tree_height_is_108 (hH : 0 < H) (h_ratio : ratio_condition H) : taller_tree_height H = 108 :=
sorry

end NUMINAMATH_GPT_taller_tree_height_is_108_l488_48848


namespace NUMINAMATH_GPT_complement_union_l488_48894

namespace SetProof

variable (U M N : Set ℕ)
variable (H_U : U = {1, 2, 3, 4, 5})
variable (H_M : M = {1, 2})
variable (H_N : N = {3, 4})

theorem complement_union :
  (U \ (M ∪ N)) = {5} := by
  sorry

end SetProof

end NUMINAMATH_GPT_complement_union_l488_48894


namespace NUMINAMATH_GPT_social_media_usage_in_week_l488_48863

def days_in_week : ℕ := 7
def daily_phone_usage : ℕ := 16
def daily_social_media_usage : ℕ := daily_phone_usage / 2

theorem social_media_usage_in_week :
  daily_social_media_usage * days_in_week = 56 :=
by
  sorry

end NUMINAMATH_GPT_social_media_usage_in_week_l488_48863


namespace NUMINAMATH_GPT_hyperbola_s_squared_l488_48897

theorem hyperbola_s_squared 
  (s : ℝ) 
  (a b : ℝ) 
  (h1 : a = 3)
  (h2 : b^2 = 144 / 13) 
  (h3 : (2, s) ∈ {p : ℝ × ℝ | (p.2)^2 / a^2 - (p.1)^2 / b^2 = 1}) :
  s^2 = 441 / 36 :=
by sorry

end NUMINAMATH_GPT_hyperbola_s_squared_l488_48897


namespace NUMINAMATH_GPT_transistors_in_2010_l488_48886

-- Define initial conditions
def initial_transistors : ℕ := 500000
def years_passed : ℕ := 15
def tripling_period : ℕ := 3
def tripling_factor : ℕ := 3

-- Define the function to compute the number of transistors after a number of years
noncomputable def final_transistors (initial : ℕ) (years : ℕ) (period : ℕ) (factor : ℕ) : ℕ :=
  initial * factor ^ (years / period)

-- State the proposition we aim to prove
theorem transistors_in_2010 : final_transistors initial_transistors years_passed tripling_period tripling_factor = 121500000 := 
by 
  sorry

end NUMINAMATH_GPT_transistors_in_2010_l488_48886


namespace NUMINAMATH_GPT_geometric_sequence_sufficient_not_necessary_l488_48845

theorem geometric_sequence_sufficient_not_necessary (a b c : ℝ) :
  (∃ r : ℝ, a = b * r ∧ b = c * r) → (b^2 = a * c) ∧ ¬ ( (b^2 = a * c) → (∃ r : ℝ, a = b * r ∧ b = c * r) ) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sufficient_not_necessary_l488_48845


namespace NUMINAMATH_GPT_compute_fraction_l488_48800

theorem compute_fraction :
  ( (12^4 + 500) * (24^4 + 500) * (36^4 + 500) * (48^4 + 500) * (60^4 + 500) ) /
  ( (6^4 + 500) * (18^4 + 500) * (30^4 + 500) * (42^4 + 500) * (54^4 + 500) ) = -182 :=
by
  sorry

end NUMINAMATH_GPT_compute_fraction_l488_48800


namespace NUMINAMATH_GPT_imaginary_part_of_z_l488_48827

open Complex

-- Definition of the complex number as per the problem statement
def z : ℂ := (2 - 3 * Complex.I) * Complex.I

-- The theorem stating that the imaginary part of the given complex number is 2
theorem imaginary_part_of_z : z.im = 2 :=
by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_z_l488_48827


namespace NUMINAMATH_GPT_total_votes_is_240_l488_48840

variable {x : ℕ} -- Total number of votes (natural number)
variable {S : ℤ} -- Score (integer)

-- Given conditions
axiom score_condition : S = 120
axiom votes_condition : 3 * x / 4 - x / 4 = S

theorem total_votes_is_240 : x = 240 :=
by
  -- Proof should go here
  sorry

end NUMINAMATH_GPT_total_votes_is_240_l488_48840


namespace NUMINAMATH_GPT_calculate_distance_l488_48817

theorem calculate_distance (t : ℕ) (h_t : t = 4) : 5 * t^2 + 2 * t = 88 :=
by
  rw [h_t]
  norm_num

end NUMINAMATH_GPT_calculate_distance_l488_48817


namespace NUMINAMATH_GPT_intersection_M_N_l488_48898

def M : Set ℝ := { x | x^2 + x - 2 < 0 }
def N : Set ℝ := { x | 0 < x ∧ x ≤ 2 }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l488_48898


namespace NUMINAMATH_GPT_xyz_ineq_l488_48804

theorem xyz_ineq (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y + y * z + z * x = 1) : 
  x * y * z * (x + y + z) ≤ 1 / 3 := 
sorry

end NUMINAMATH_GPT_xyz_ineq_l488_48804


namespace NUMINAMATH_GPT_tangents_product_l488_48882

theorem tangents_product (x y : ℝ) 
  (h1 : Real.tan x - Real.tan y = 7) 
  (h2 : 2 * Real.sin (2 * (x - y)) = Real.sin (2 * x) * Real.sin (2 * y)) :
  Real.tan x * Real.tan y = -7/6 := 
sorry

end NUMINAMATH_GPT_tangents_product_l488_48882


namespace NUMINAMATH_GPT_kristine_travel_distance_l488_48856

theorem kristine_travel_distance :
  ∃ T : ℝ, T + T / 2 + T / 6 = 500 ∧ T = 300 := by
  sorry

end NUMINAMATH_GPT_kristine_travel_distance_l488_48856


namespace NUMINAMATH_GPT_find_x_l488_48866

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -1)
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ := v₁.1 * v₂.1 + v₁.2 * v₂.2

theorem find_x
  (h : dot_product vector_a (vector_b x) = 0) :
  x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l488_48866


namespace NUMINAMATH_GPT_percentage_decrease_l488_48861

theorem percentage_decrease (A C : ℝ) (h1 : C > A) (h2 : A > 0) (h3 : C = 1.20 * A) : 
  ∃ y : ℝ, A = C - (y/100) * C ∧ y = 50 / 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_percentage_decrease_l488_48861


namespace NUMINAMATH_GPT_gcd_of_fraction_in_lowest_terms_l488_48895

theorem gcd_of_fraction_in_lowest_terms (n : ℤ) (h : n % 2 = 1) : Int.gcd (2 * n + 2) (3 * n + 2) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_of_fraction_in_lowest_terms_l488_48895


namespace NUMINAMATH_GPT_bank_balance_after_two_years_l488_48805

-- Define the original amount deposited
def original_amount : ℝ := 5600

-- Define the interest rate
def interest_rate : ℝ := 0.07

-- Define the interest for each year based on the original amount
def interest_per_year : ℝ := original_amount * interest_rate

-- Define the total amount after two years
def total_amount_after_two_years : ℝ := original_amount + interest_per_year + interest_per_year

-- Define the target value
def target_value : ℝ := 6384

-- The theorem we aim to prove
theorem bank_balance_after_two_years : 
  total_amount_after_two_years = target_value := 
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_bank_balance_after_two_years_l488_48805


namespace NUMINAMATH_GPT_seminar_total_cost_l488_48896

theorem seminar_total_cost 
  (regular_fee : ℝ)
  (discount_rate : ℝ)
  (num_teachers : ℕ) 
  (food_allowance_per_teacher : ℝ)
  (total_cost : ℝ)
  (h1 : regular_fee = 150)
  (h2 : discount_rate = 0.05)
  (h3 : num_teachers = 10) 
  (h4 : food_allowance_per_teacher = 10)
  (h5 : total_cost = regular_fee * num_teachers * (1 - discount_rate) + food_allowance_per_teacher * num_teachers) :
  total_cost = 1525 := 
sorry

end NUMINAMATH_GPT_seminar_total_cost_l488_48896


namespace NUMINAMATH_GPT_range_of_m_l488_48811

noncomputable def prop_p (m : ℝ) : Prop :=
0 < m ∧ m < 1 / 3

noncomputable def prop_q (m : ℝ) : Prop :=
0 < m ∧ m < 15

theorem range_of_m (m : ℝ) : (prop_p m ∧ ¬ prop_q m) ∨ (¬ prop_p m ∧ prop_q m) ↔ 1 / 3 ≤ m ∧ m < 15 :=
sorry

end NUMINAMATH_GPT_range_of_m_l488_48811


namespace NUMINAMATH_GPT_regular_polygon_area_l488_48839
open Real

theorem regular_polygon_area (R : ℝ) (n : ℕ) (hR : 0 < R) (hn : 8 ≤ n) (h_area : (1/2) * n * R^2 * sin (360 / n * (π / 180)) = 4 * R^2) :
  n = 10 := 
sorry

end NUMINAMATH_GPT_regular_polygon_area_l488_48839


namespace NUMINAMATH_GPT_semicircle_radius_l488_48850

-- Definition of the problem conditions
variables (a h : ℝ) -- base and height of the triangle
variable (R : ℝ)    -- radius of the semicircle

-- Statement of the proof problem
theorem semicircle_radius (h_pos : 0 < h) (a_pos : 0 < a) 
(semicircle_condition : ∀ R > 0, a * (h - R) = 2 * R * h) : R = a * h / (a + 2 * h) :=
sorry

end NUMINAMATH_GPT_semicircle_radius_l488_48850


namespace NUMINAMATH_GPT_fraction_left_handed_l488_48876

def total_participants (k : ℕ) := 15 * k

def red (k : ℕ) := 5 * k
def blue (k : ℕ) := 5 * k
def green (k : ℕ) := 3 * k
def yellow (k : ℕ) := 2 * k

def left_handed_red (k : ℕ) := (1 / 3) * red k
def left_handed_blue (k : ℕ) := (2 / 3) * blue k
def left_handed_green (k : ℕ) := (1 / 2) * green k
def left_handed_yellow (k : ℕ) := (1 / 4) * yellow k

def total_left_handed (k : ℕ) := left_handed_red k + left_handed_blue k + left_handed_green k + left_handed_yellow k

theorem fraction_left_handed (k : ℕ) : 
  (total_left_handed k) / (total_participants k) = 7 / 15 := 
sorry

end NUMINAMATH_GPT_fraction_left_handed_l488_48876


namespace NUMINAMATH_GPT_range_of_a_l488_48879

def A (x : ℝ) : Prop := x^2 - 6*x + 5 ≤ 0
def B (x a : ℝ) : Prop := x < a + 1

theorem range_of_a (a : ℝ) : (∃ x : ℝ, A x ∧ B x a) ↔ a > 0 := by
  sorry

end NUMINAMATH_GPT_range_of_a_l488_48879


namespace NUMINAMATH_GPT_merchant_boxes_fulfill_order_l488_48831

theorem merchant_boxes_fulfill_order :
  ∃ (a b c d e : ℕ), 16 * a + 17 * b + 23 * c + 39 * d + 40 * e = 100 := sorry

end NUMINAMATH_GPT_merchant_boxes_fulfill_order_l488_48831


namespace NUMINAMATH_GPT_last_number_in_first_set_l488_48878

variables (x y : ℕ)

def mean (a b c d e : ℕ) : ℕ :=
  (a + b + c + d + e) / 5

theorem last_number_in_first_set :
  (mean 28 x 42 78 y = 90) ∧ (mean 128 255 511 1023 x = 423) → y = 104 :=
by 
  sorry

end NUMINAMATH_GPT_last_number_in_first_set_l488_48878


namespace NUMINAMATH_GPT_cube_weight_l488_48806

theorem cube_weight (l1 l2 V1 V2 k : ℝ) (h1: l2 = 2 * l1) (h2: V1 = l1^3) (h3: V2 = (2 * l1)^3) (h4: w2 = 48) (h5: V2 * k = w2) (h6: V1 * k = w1):
  w1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_cube_weight_l488_48806


namespace NUMINAMATH_GPT_prob_none_given_not_A_l488_48883

-- Definitions based on the conditions
def prob_single (h : ℕ → Prop) : ℝ := 0.2
def prob_double (h1 h2 : ℕ → Prop) : ℝ := 0.1
def prob_triple_given_AB : ℝ := 0.5

-- Assume that h1, h2, and h3 represent the hazards A, B, and C respectively.
variables (A B C : ℕ → Prop)

-- The ultimate theorem we want to prove
theorem prob_none_given_not_A (P : ℕ → Prop) :
  ((1 - (0.2 * 3 + 0.1 * 3) + (prob_triple_given_AB * (prob_single A + prob_double A B))) / (1 - 0.2) = 11 / 9) :=
by
  sorry

end NUMINAMATH_GPT_prob_none_given_not_A_l488_48883


namespace NUMINAMATH_GPT_percentage_of_total_population_absent_l488_48846

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_of_total_population_absent_l488_48846


namespace NUMINAMATH_GPT_part_one_part_two_l488_48887

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 5 then (16 / (9 - x) - 1) else (11 - (2 / 45) * x ^ 2)

theorem part_one (k : ℝ) (h : 1 ≤ k ∧ k ≤ 4) : k * (16 / (9 - 3) - 1) = 4 → k = 12 / 5 :=
by sorry

theorem part_two (y x : ℝ) (h_y : y = 4) :
  (1 ≤ x ∧ x ≤ 5 ∧ 4 * (16 / (9 - x) - 1) ≥ 4) ∨
  (5 < x ∧ x ≤ 15 ∧ 4 * (11 - (2/45) * x ^ 2) ≥ 4) :=
by sorry

end NUMINAMATH_GPT_part_one_part_two_l488_48887


namespace NUMINAMATH_GPT_middle_school_students_count_l488_48857

variable (M H m h : ℕ)
variable (total_students : ℕ := 36)
variable (percentage_middle : ℕ := 20)
variable (percentage_high : ℕ := 25)

theorem middle_school_students_count :
  total_students = 36 ∧ (m = h) →
  (percentage_middle / 100 * M = m) ∧
  (percentage_high / 100 * H = h) →
  M + H = total_students →
  M = 16 :=
by sorry

end NUMINAMATH_GPT_middle_school_students_count_l488_48857


namespace NUMINAMATH_GPT_find_diminished_value_l488_48842

theorem find_diminished_value :
  ∃ (x : ℕ), 1015 - x = Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28 :=
by
  use 7
  simp
  unfold Nat.lcm
  sorry

end NUMINAMATH_GPT_find_diminished_value_l488_48842


namespace NUMINAMATH_GPT_sin_double_angle_l488_48818

theorem sin_double_angle (α : ℝ)
  (h : Real.cos (Real.pi / 4 - α) = -3 / 5) :
  Real.sin (2 * α) = -7 / 25 := by
sorry

end NUMINAMATH_GPT_sin_double_angle_l488_48818


namespace NUMINAMATH_GPT_problem1_problem2_l488_48860

theorem problem1 :
  (2 / 3) * Real.sqrt 24 / (-Real.sqrt 3) * (1 / 3) * Real.sqrt 27 = - (4 / 3) * Real.sqrt 6 :=
sorry

theorem problem2 :
  Real.sqrt 3 * Real.sqrt 12 + (Real.sqrt 3 + 1)^2 = 10 + 2 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l488_48860


namespace NUMINAMATH_GPT_triangle_right_triangle_l488_48819

-- Defining the sides of the triangle
variables (a b c : ℝ)

-- Theorem statement
theorem triangle_right_triangle (h : (a + b)^2 = c^2 + 2 * a * b) : a^2 + b^2 = c^2 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_right_triangle_l488_48819


namespace NUMINAMATH_GPT_Somu_years_back_l488_48835

-- Define the current ages of Somu and his father, and the relationship between them
variables (S F : ℕ)
variable (Y : ℕ)

-- Hypotheses based on the problem conditions
axiom age_of_Somu : S = 14
axiom age_relation : S = F / 3

-- Define the condition for years back when Somu was one-fifth his father's age
axiom years_back_condition : S - Y = (F - Y) / 5

-- Problem statement: Prove that 7 years back, Somu was one-fifth of his father's age
theorem Somu_years_back : Y = 7 :=
by
  sorry

end NUMINAMATH_GPT_Somu_years_back_l488_48835


namespace NUMINAMATH_GPT_norris_money_left_l488_48824

def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def hugo_spent  : ℕ := 75
def total_savings : ℕ := sept_savings + oct_savings + nov_savings
def norris_left : ℕ := total_savings - hugo_spent

theorem norris_money_left : norris_left = 10 := by
  unfold norris_left total_savings sept_savings oct_savings nov_savings hugo_spent
  sorry

end NUMINAMATH_GPT_norris_money_left_l488_48824


namespace NUMINAMATH_GPT_determine_a_for_line_l488_48899

theorem determine_a_for_line (a : ℝ) (h : a ≠ 0)
  (intercept_condition : ∃ (k : ℝ), 
    ∀ x y : ℝ, (a * x - 6 * y - 12 * a = 0) → (x = 12) ∧ (y = 2 * a * x / 6) ∧ (12 = 3 * (-2 * a))) : 
  a = -2 :=
by
  sorry

end NUMINAMATH_GPT_determine_a_for_line_l488_48899


namespace NUMINAMATH_GPT_betty_cupcakes_per_hour_l488_48829

theorem betty_cupcakes_per_hour (B : ℕ) (Dora_rate : ℕ) (betty_break_hours : ℕ) (total_hours : ℕ) (cupcake_diff : ℕ) :
  Dora_rate = 8 →
  betty_break_hours = 2 →
  total_hours = 5 →
  cupcake_diff = 10 →
  (total_hours - betty_break_hours) * B = Dora_rate * total_hours - cupcake_diff →
  B = 10 :=
by
  intros hDora_rate hbreak_hours htotal_hours hcupcake_diff hcupcake_eq
  sorry

end NUMINAMATH_GPT_betty_cupcakes_per_hour_l488_48829


namespace NUMINAMATH_GPT_complement_U_A_l488_48813

def U : Set ℝ := {x | ∃ y, y = Real.sqrt x}
def A : Set ℝ := {x | 3 ≤ 2 * x - 1 ∧ 2 * x - 1 < 5}

theorem complement_U_A : (U \ A) = {x | (0 ≤ x ∧ x < 2) ∨ (3 ≤ x)} := sorry

end NUMINAMATH_GPT_complement_U_A_l488_48813


namespace NUMINAMATH_GPT_playground_area_l488_48820

noncomputable def calculate_area (w s : ℝ) : ℝ := s * s

theorem playground_area (w s : ℝ) (h1 : s = 3 * w + 10) (h2 : 4 * s = 480) : calculate_area w s = 14400 := by
  sorry

end NUMINAMATH_GPT_playground_area_l488_48820


namespace NUMINAMATH_GPT_power_addition_identity_l488_48880

theorem power_addition_identity : 
  (-2)^23 + 5^(2^4 + 3^3 - 4^2) = -8388608 + 5^27 := by
  sorry

end NUMINAMATH_GPT_power_addition_identity_l488_48880


namespace NUMINAMATH_GPT_min_value_eq_ab_squared_l488_48862

noncomputable def min_value (x a b : ℝ) : ℝ := 1 / (x^a * (1 - x)^b)

theorem min_value_eq_ab_squared (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x, 0 < x ∧ x < 1 ∧ min_value x a b = (a + b)^2 :=
by
  sorry

end NUMINAMATH_GPT_min_value_eq_ab_squared_l488_48862


namespace NUMINAMATH_GPT_consecutive_numbers_product_l488_48865

theorem consecutive_numbers_product : 
  ∃ n : ℕ, (n + n + 1 = 11) ∧ (n * (n + 1) * (n + 2) = 210) :=
sorry

end NUMINAMATH_GPT_consecutive_numbers_product_l488_48865


namespace NUMINAMATH_GPT_number_of_two_bedroom_units_l488_48853

-- Definitions based on the conditions
def is_solution (x y : ℕ) : Prop :=
  (x + y = 12) ∧ (360 * x + 450 * y = 4950)

theorem number_of_two_bedroom_units : ∃ y : ℕ, is_solution (12 - y) y ∧ y = 7 :=
by
  sorry

end NUMINAMATH_GPT_number_of_two_bedroom_units_l488_48853


namespace NUMINAMATH_GPT_area_of_triangle_BFE_l488_48868

theorem area_of_triangle_BFE (A B C D E F : ℝ × ℝ) (u v : ℝ) 
  (h_rectangle : (0, 0) = A ∧ (3 * u, 0) = B ∧ (3 * u, 3 * v) = C ∧ (0, 3 * v) = D)
  (h_E : E = (0, 2 * v))
  (h_F : F = (2 * u, 0))
  (h_area_rectangle : 3 * u * 3 * v = 48) :
  ∃ (area : ℝ), area = |3 * u * 2 * v| / 2 ∧ area = 24 :=
by 
  sorry

end NUMINAMATH_GPT_area_of_triangle_BFE_l488_48868


namespace NUMINAMATH_GPT_value_of_f_neg_a_l488_48832

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 0 := by
  sorry

end NUMINAMATH_GPT_value_of_f_neg_a_l488_48832


namespace NUMINAMATH_GPT_period_of_f_max_value_of_f_and_values_l488_48838

noncomputable def f (x : ℝ) : ℝ := (1 - Real.sin (2 * x)) / (Real.sin x - Real.cos x)

-- Statement 1: The period of f(x) is 2π
theorem period_of_f : ∀ x, f (x + 2 * Real.pi) = f x := by
  sorry

-- Statement 2: The maximum value of f(x) is √2 and it is attained at x = 2kπ + 3π/4, k ∈ ℤ
theorem max_value_of_f_and_values :
  (∀ x, f x ≤ Real.sqrt 2) ∧
  (∃ k : ℤ, f (2 * k * Real.pi + 3 * Real.pi / 4) = Real.sqrt 2) := by
  sorry

end NUMINAMATH_GPT_period_of_f_max_value_of_f_and_values_l488_48838


namespace NUMINAMATH_GPT_log12_div_log15_eq_2m_n_div_1_m_n_l488_48852

variable (m n : Real)

theorem log12_div_log15_eq_2m_n_div_1_m_n 
  (h1 : Real.log 2 = m) 
  (h2 : Real.log 3 = n) : 
  Real.log 12 / Real.log 15 = (2 * m + n) / (1 - m + n) :=
by sorry

end NUMINAMATH_GPT_log12_div_log15_eq_2m_n_div_1_m_n_l488_48852


namespace NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l488_48864

theorem line_intersects_parabola_at_one_point (k : ℝ) : (∃ y : ℝ, -y^2 - 4 * y + 2 = k) ↔ k = 6 :=
by 
  sorry

end NUMINAMATH_GPT_line_intersects_parabola_at_one_point_l488_48864


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_for_tangency_l488_48891

-- Given conditions
variables (ρ θ D E : ℝ)

-- Definition of the circle in polar coordinates and the condition for tangency with the radial axis
def circle_eq : Prop := ρ = D * Real.cos θ + E * Real.sin θ

-- Statement of the proof problem
theorem necessary_and_sufficient_condition_for_tangency :
  (circle_eq ρ θ D E) → (D = 0 ∧ E ≠ 0) :=
sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_for_tangency_l488_48891


namespace NUMINAMATH_GPT_maximum_a3_S10_l488_48885

-- Given definitions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def conditions (a : ℕ → ℝ) : Prop :=
  is_arithmetic_sequence a ∧ (∀ n, a n > 0) ∧ (a 1 + a 3 + a 8 = a 4 ^ 2)

-- The problem statement
theorem maximum_a3_S10 (a : ℕ → ℝ) (h : conditions a) : 
  (∃ S : ℝ, S = a 3 * ((10 / 2) * (a 1 + a 10)) ∧ S ≤ 375 / 4) :=
sorry

end NUMINAMATH_GPT_maximum_a3_S10_l488_48885


namespace NUMINAMATH_GPT_roy_older_than_julia_l488_48822

variable {R J K x : ℝ}

theorem roy_older_than_julia (h1 : R = J + x)
                            (h2 : R = K + x / 2)
                            (h3 : R + 2 = 2 * (J + 2))
                            (h4 : (R + 2) * (K + 2) = 192) :
                            x = 2 :=
by
  sorry

end NUMINAMATH_GPT_roy_older_than_julia_l488_48822


namespace NUMINAMATH_GPT_minimum_groups_needed_l488_48855

theorem minimum_groups_needed :
  ∃ (g : ℕ), g = 5 ∧ ∀ n k : ℕ, n = 30 → k ≤ 7 → n / k = g :=
by
  sorry

end NUMINAMATH_GPT_minimum_groups_needed_l488_48855


namespace NUMINAMATH_GPT_probability_age_less_than_20_l488_48874

theorem probability_age_less_than_20 (total_people : ℕ) (over_30_years : ℕ) 
  (less_than_20_years : ℕ) (h1 : total_people = 120) (h2 : over_30_years = 90) 
  (h3 : less_than_20_years = total_people - over_30_years) : 
  (less_than_20_years : ℚ) / total_people = 1 / 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_probability_age_less_than_20_l488_48874


namespace NUMINAMATH_GPT_aprons_to_sew_tomorrow_l488_48828

def total_aprons : ℕ := 150
def already_sewn : ℕ := 13
def sewn_today (already_sewn : ℕ) : ℕ := 3 * already_sewn
def sewn_tomorrow (total_aprons : ℕ) (already_sewn : ℕ) (sewn_today : ℕ) : ℕ :=
  let remaining := total_aprons - (already_sewn + sewn_today)
  remaining / 2

theorem aprons_to_sew_tomorrow : sewn_tomorrow total_aprons already_sewn (sewn_today already_sewn) = 49 :=
  by 
    sorry

end NUMINAMATH_GPT_aprons_to_sew_tomorrow_l488_48828


namespace NUMINAMATH_GPT_molecular_weight_constant_l488_48834

-- Given the molecular weight of a compound
def molecular_weight (w : ℕ) := w = 1188

-- Statement about molecular weight of n moles
def weight_of_n_moles (n : ℕ) := n * 1188

theorem molecular_weight_constant (moles : ℕ) : 
  ∀ (w : ℕ), molecular_weight w → ∀ (n : ℕ), weight_of_n_moles n = n * w :=
by
  intro w h n
  sorry

end NUMINAMATH_GPT_molecular_weight_constant_l488_48834


namespace NUMINAMATH_GPT_simplify_expression_l488_48871

theorem simplify_expression : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l488_48871


namespace NUMINAMATH_GPT_article_usage_correct_l488_48859

def blank1 := "a"
def blank2 := ""  -- Representing "不填" (no article) as an empty string for simplicity

theorem article_usage_correct :
  (blank1 = "a" ∧ blank2 = "") :=
by
  sorry

end NUMINAMATH_GPT_article_usage_correct_l488_48859


namespace NUMINAMATH_GPT_coefficient_a2b2_in_expansion_l488_48837

theorem coefficient_a2b2_in_expansion :
  -- Combining the coefficients: \binom{4}{2} and \binom{6}{3}
  (Nat.choose 4 2) * (Nat.choose 6 3) = 120 :=
by
  -- No proof required, using sorry to indicate that.
  sorry

end NUMINAMATH_GPT_coefficient_a2b2_in_expansion_l488_48837


namespace NUMINAMATH_GPT_angle_measure_l488_48801

theorem angle_measure (α : ℝ) 
  (h1 : 90 - α + (180 - α) = 180) : 
  α = 45 := 
by 
  sorry

end NUMINAMATH_GPT_angle_measure_l488_48801


namespace NUMINAMATH_GPT_identical_lines_pairs_count_l488_48873

theorem identical_lines_pairs_count : 
  ∃ P : Finset (ℝ × ℝ), (∀ p ∈ P, 
    (∃ a b, p = (a, b) ∧ 
      (∀ x y, 2 * x + a * y + b = 0 ↔ b * x + 3 * y - 9 = 0))) ∧ P.card = 2 :=
sorry

end NUMINAMATH_GPT_identical_lines_pairs_count_l488_48873


namespace NUMINAMATH_GPT_largest_divisor_of_composite_l488_48807

theorem largest_divisor_of_composite (n : ℕ) (h : n > 1 ∧ ¬ Nat.Prime n) : 12 ∣ (n^4 - n^2) :=
sorry

end NUMINAMATH_GPT_largest_divisor_of_composite_l488_48807


namespace NUMINAMATH_GPT_no_solns_to_equation_l488_48816

noncomputable def no_solution : Prop :=
  ∀ (n m r : ℕ), (1 ≤ n) → (1 ≤ m) → (1 ≤ r) → n^5 + 49^m ≠ 1221^r

theorem no_solns_to_equation : no_solution :=
sorry

end NUMINAMATH_GPT_no_solns_to_equation_l488_48816


namespace NUMINAMATH_GPT_decrease_percent_in_revenue_l488_48843

theorem decrease_percent_in_revenue
  (T C : ℝ) -- T = original tax, C = original consumption
  (h1 : 0 < T) -- ensuring that T is positive
  (h2 : 0 < C) -- ensuring that C is positive
  (new_tax : ℝ := 0.75 * T) -- new tax is 75% of original tax
  (new_consumption : ℝ := 1.10 * C) -- new consumption is 110% of original consumption
  (original_revenue : ℝ := T * C) -- original revenue
  (new_revenue : ℝ := (0.75 * T) * (1.10 * C)) -- new revenue
  (decrease_percent : ℝ := ((T * C - (0.75 * T) * (1.10 * C)) / (T * C)) * 100) -- decrease percent
  : decrease_percent = 17.5 :=
by
  sorry

end NUMINAMATH_GPT_decrease_percent_in_revenue_l488_48843


namespace NUMINAMATH_GPT_find_a_l488_48849

noncomputable def parabola_eq (a b c : ℤ) (x : ℤ) : ℤ :=
  a * x^2 + b * x + c

theorem find_a (a b c : ℤ)
  (h_vertex : ∀ x, parabola_eq a b c x = a * (x - 2)^2 + 5) 
  (h_point : parabola_eq a b c 1 = 6) :
  a = 1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l488_48849


namespace NUMINAMATH_GPT_flagstaff_height_is_correct_l488_48803

noncomputable def flagstaff_height : ℝ := 40.25 * 12.5 / 28.75

theorem flagstaff_height_is_correct :
  flagstaff_height = 17.5 :=
by 
  -- These conditions are implicit in the previous definition
  sorry

end NUMINAMATH_GPT_flagstaff_height_is_correct_l488_48803


namespace NUMINAMATH_GPT_sequence_eventually_constant_l488_48870

theorem sequence_eventually_constant (n : ℕ) (h : n ≥ 1) : 
  ∃ s, ∀ k ≥ s, (2 ^ (2 ^ k) % n) = (2 ^ (2 ^ (k + 1)) % n) :=
sorry

end NUMINAMATH_GPT_sequence_eventually_constant_l488_48870


namespace NUMINAMATH_GPT_trapezoid_perimeter_l488_48830

noncomputable def length_AD : ℝ := 8
noncomputable def length_BC : ℝ := 18
noncomputable def length_AB : ℝ := 12 -- Derived from tangency and symmetry considerations
noncomputable def length_CD : ℝ := 18

theorem trapezoid_perimeter (ABCD : Π (a b c d : Type), a → b → c → d → Prop)
  (AD BC AB CD : ℝ)
  (h1 : AD = 8) (h2 : BC = 18) (h3 : AB = 12) (h4 : CD = 18)
  : AD + BC + AB + CD = 56 :=
by
  rw [h1, h2, h3, h4]
  norm_num

end NUMINAMATH_GPT_trapezoid_perimeter_l488_48830


namespace NUMINAMATH_GPT_triangle_possible_sides_l488_48814

theorem triangle_possible_sides (a b c : ℕ) (h₁ : a + b + c = 7) (h₂ : a + b > c) (h₃ : a + c > b) (h₄ : b + c > a) :
  a = 1 ∨ a = 2 ∨ a = 3 :=
by {
  sorry
}

end NUMINAMATH_GPT_triangle_possible_sides_l488_48814


namespace NUMINAMATH_GPT_domain_of_function_l488_48802

theorem domain_of_function (x : ℝ) : 
  {x | ∃ k : ℤ, - (Real.pi / 3) + (2 : ℝ) * k * Real.pi ≤ x ∧ x ≤ (Real.pi / 3) + (2 : ℝ) * k * Real.pi} :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_domain_of_function_l488_48802


namespace NUMINAMATH_GPT_count_five_digit_numbers_ending_in_6_divisible_by_3_l488_48844

theorem count_five_digit_numbers_ending_in_6_divisible_by_3 : 
  (∃ (n : ℕ), n = 3000 ∧
  ∀ (x : ℕ), (x ≥ 10000 ∧ x ≤ 99999) ∧ (x % 10 = 6) ∧ (x % 3 = 0) ↔ 
  (∃ (k : ℕ), x = 10026 + k * 30 ∧ k < 3000)) :=
by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_count_five_digit_numbers_ending_in_6_divisible_by_3_l488_48844
