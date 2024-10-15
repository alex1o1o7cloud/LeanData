import Mathlib

namespace NUMINAMATH_GPT_range_a_empty_intersection_range_a_sufficient_condition_l947_94752

noncomputable def A (x : ℝ) : Prop := -10 < x ∧ x < 2
noncomputable def B (x : ℝ) (a : ℝ) : Prop := x ≥ 1 + a ∨ x ≤ 1 - a
noncomputable def A_inter_B_empty (a : ℝ) : Prop := ∀ x : ℝ, A x → ¬ B x a
noncomputable def neg_p (x : ℝ) : Prop := x ≥ 2 ∨ x ≤ -10
noncomputable def neg_p_implies_q (a : ℝ) : Prop := ∀ x : ℝ, neg_p x → B x a

theorem range_a_empty_intersection : (∀ x : ℝ, A x → ¬ B x 11) → 11 ≤ a := by
  sorry

theorem range_a_sufficient_condition : (∀ x : ℝ, neg_p x → B x 1) → 0 < a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_GPT_range_a_empty_intersection_range_a_sufficient_condition_l947_94752


namespace NUMINAMATH_GPT_percentage_tax_raise_expecting_population_l947_94718

def percentage_affirmative_responses_tax : ℝ := 0.4
def percentage_affirmative_responses_money : ℝ := 0.3
def percentage_affirmative_responses_bonds : ℝ := 0.5
def percentage_affirmative_responses_gold : ℝ := 0.0

def fraction_liars : ℝ := 0.1
def fraction_economists : ℝ := 1 - fraction_liars

theorem percentage_tax_raise_expecting_population : 
  (percentage_affirmative_responses_tax - fraction_liars) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_percentage_tax_raise_expecting_population_l947_94718


namespace NUMINAMATH_GPT_cashback_discount_percentage_l947_94711

noncomputable def iphoneOriginalPrice : ℝ := 800
noncomputable def iwatchOriginalPrice : ℝ := 300
noncomputable def iphoneDiscountRate : ℝ := 0.15
noncomputable def iwatchDiscountRate : ℝ := 0.10
noncomputable def finalPrice : ℝ := 931

noncomputable def iphoneDiscountedPrice : ℝ := iphoneOriginalPrice * (1 - iphoneDiscountRate)
noncomputable def iwatchDiscountedPrice : ℝ := iwatchOriginalPrice * (1 - iwatchDiscountRate)
noncomputable def totalDiscountedPrice : ℝ := iphoneDiscountedPrice + iwatchDiscountedPrice
noncomputable def cashbackAmount : ℝ := totalDiscountedPrice - finalPrice
noncomputable def cashbackRate : ℝ := (cashbackAmount / totalDiscountedPrice) * 100

theorem cashback_discount_percentage : cashbackRate = 2 := by
  sorry

end NUMINAMATH_GPT_cashback_discount_percentage_l947_94711


namespace NUMINAMATH_GPT_trig_inequality_2016_l947_94737

theorem trig_inequality_2016 :
  let a := Real.sin (Real.cos (2016 * Real.pi / 180))
  let b := Real.sin (Real.sin (2016 * Real.pi / 180))
  let c := Real.cos (Real.sin (2016 * Real.pi / 180))
  let d := Real.cos (Real.cos (2016 * Real.pi / 180))
  c > d ∧ d > b ∧ b > a := by
  sorry

end NUMINAMATH_GPT_trig_inequality_2016_l947_94737


namespace NUMINAMATH_GPT_baron_munchausen_is_telling_truth_l947_94767

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_10_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def not_divisible_by_10 (n : ℕ) : Prop :=
  ¬(n % 10 = 0)

theorem baron_munchausen_is_telling_truth :
  ∃ a b : ℕ, a ≠ b ∧ is_10_digit a ∧ is_10_digit b ∧ not_divisible_by_10 a ∧ not_divisible_by_10 b ∧
  (a - digit_sum (a^2) = b - digit_sum (b^2)) := sorry

end NUMINAMATH_GPT_baron_munchausen_is_telling_truth_l947_94767


namespace NUMINAMATH_GPT_math_crackers_initial_l947_94706

def crackers_initial (gave_each : ℕ) (left : ℕ) (num_friends : ℕ) : ℕ :=
  (gave_each * num_friends) + left

theorem math_crackers_initial :
  crackers_initial 7 17 3 = 38 :=
by
  -- The definition of crackers_initial and the theorem statement should be enough.
  -- The exact proof is left as a sorry placeholder.
  sorry

end NUMINAMATH_GPT_math_crackers_initial_l947_94706


namespace NUMINAMATH_GPT_inequality_solution_l947_94746

theorem inequality_solution (x : ℝ) :
  (x + 2 > 3 * (2 - x) ∧ x < (x + 3) / 2) ↔ 1 < x ∧ x < 3 := sorry

end NUMINAMATH_GPT_inequality_solution_l947_94746


namespace NUMINAMATH_GPT_driver_days_off_l947_94792

theorem driver_days_off 
  (drivers : ℕ) 
  (cars : ℕ) 
  (maintenance_rate : ℚ) 
  (days_in_month : ℕ)
  (needed_driver_days : ℕ)
  (x : ℚ) :
  drivers = 54 →
  cars = 60 →
  maintenance_rate = 0.25 →
  days_in_month = 30 →
  needed_driver_days = 45 * days_in_month →
  54 * (30 - x) = needed_driver_days →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_driver_days_off_l947_94792


namespace NUMINAMATH_GPT_solution_l947_94775

-- Define the equation
def equation (x : ℝ) := x^2 + 4*x + 3 + (x + 3)*(x + 5) = 0

-- State that x = -3 is a solution to the equation
theorem solution : equation (-3) :=
by
  unfold equation
  simp
  sorry

end NUMINAMATH_GPT_solution_l947_94775


namespace NUMINAMATH_GPT_Andrew_runs_2_miles_each_day_l947_94725

theorem Andrew_runs_2_miles_each_day
  (A : ℕ)
  (Peter_runs : ℕ := A + 3)
  (total_miles_after_5_days : 5 * (A + Peter_runs) = 35) :
  A = 2 :=
by
  sorry

end NUMINAMATH_GPT_Andrew_runs_2_miles_each_day_l947_94725


namespace NUMINAMATH_GPT_max_value_of_M_l947_94740

def J (k : ℕ) := 10^(k + 3) + 256

def M (k : ℕ) := Nat.factors (J k) |>.count 2

theorem max_value_of_M (k : ℕ) (hk : k > 0) :
  M k = 8 := by
  sorry

end NUMINAMATH_GPT_max_value_of_M_l947_94740


namespace NUMINAMATH_GPT_solve_quadratic_eq_l947_94749

theorem solve_quadratic_eq (x : ℝ) : x^2 = 6 * x ↔ (x = 0 ∨ x = 6) := by
  sorry

end NUMINAMATH_GPT_solve_quadratic_eq_l947_94749


namespace NUMINAMATH_GPT_money_total_l947_94784

theorem money_total (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 350) (h3 : C = 100) : A + B + C = 450 :=
by {
  sorry
}

end NUMINAMATH_GPT_money_total_l947_94784


namespace NUMINAMATH_GPT_not_symmetric_about_point_l947_94791

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

theorem not_symmetric_about_point : ¬ (∀ h : ℝ, f (1 + h) = f (1 - h)) :=
by
  sorry

end NUMINAMATH_GPT_not_symmetric_about_point_l947_94791


namespace NUMINAMATH_GPT_larger_number_1655_l947_94704

theorem larger_number_1655 (L S : ℕ) (h1 : L - S = 1325) (h2 : L = 5 * S + 5) : L = 1655 :=
by sorry

end NUMINAMATH_GPT_larger_number_1655_l947_94704


namespace NUMINAMATH_GPT_number_of_people_who_didnt_do_both_l947_94700

def total_graduates : ℕ := 73
def graduates_both : ℕ := 13

theorem number_of_people_who_didnt_do_both : total_graduates - graduates_both = 60 :=
by
  sorry

end NUMINAMATH_GPT_number_of_people_who_didnt_do_both_l947_94700


namespace NUMINAMATH_GPT_angles_identity_l947_94772
open Real

theorem angles_identity (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2))
  (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end NUMINAMATH_GPT_angles_identity_l947_94772


namespace NUMINAMATH_GPT_percentage_relationship_l947_94714

theorem percentage_relationship (a b : ℝ) (h : a = 1.2 * b) : ¬ (b = 0.8 * a) :=
by
  -- assumption: a = 1.2 * b
  -- goal: ¬ (b = 0.8 * a)
  sorry

end NUMINAMATH_GPT_percentage_relationship_l947_94714


namespace NUMINAMATH_GPT_correct_multiplication_factor_l947_94715

theorem correct_multiplication_factor (x : ℕ) : ((139 * x) - 1251 = 139 * 34) → x = 43 := by
  sorry

end NUMINAMATH_GPT_correct_multiplication_factor_l947_94715


namespace NUMINAMATH_GPT_min_value_of_function_l947_94703

theorem min_value_of_function (p : ℝ) : 
  ∃ x : ℝ, (x^2 - 2 * p * x + 2 * p^2 + 2 * p - 1) = -2 := sorry

end NUMINAMATH_GPT_min_value_of_function_l947_94703


namespace NUMINAMATH_GPT_distance_in_interval_l947_94762

open Set Real

def distance_to_town (d : ℝ) : Prop :=
d < 8 ∧ 7 < d ∧ 6 < d

theorem distance_in_interval (d : ℝ) : distance_to_town d → d ∈ Ioo 7 8 :=
by
  intro h
  have d_in_Ioo_8 := h.left
  have d_in_Ioo_7 := h.right.left
  have d_in_Ioo_6 := h.right.right
  /- The specific steps for combining inequalities aren't needed for the final proof. -/
  sorry

end NUMINAMATH_GPT_distance_in_interval_l947_94762


namespace NUMINAMATH_GPT_tracy_additional_miles_l947_94730

def total_distance : ℕ := 1000
def michelle_distance : ℕ := 294
def twice_michelle_distance : ℕ := 2 * michelle_distance
def katie_distance : ℕ := michelle_distance / 3
def tracy_distance := total_distance - (michelle_distance + katie_distance)
def additional_miles := tracy_distance - twice_michelle_distance

-- The statement to prove:
theorem tracy_additional_miles : additional_miles = 20 := by
  sorry

end NUMINAMATH_GPT_tracy_additional_miles_l947_94730


namespace NUMINAMATH_GPT_trigonometric_identity_l947_94726

theorem trigonometric_identity :
  (2 * Real.sin (10 * Real.pi / 180) - Real.cos (20 * Real.pi / 180)) / Real.cos (70 * Real.pi / 180) = - Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trigonometric_identity_l947_94726


namespace NUMINAMATH_GPT_two_fifths_in_fraction_l947_94710

theorem two_fifths_in_fraction : 
  (∃ (k : ℚ), k = (9/3) / (2/5) ∧ k = 15/2) :=
by 
  sorry

end NUMINAMATH_GPT_two_fifths_in_fraction_l947_94710


namespace NUMINAMATH_GPT_sequence_satisfies_recurrence_l947_94774

theorem sequence_satisfies_recurrence (n : ℕ) (a : ℕ → ℕ) (h : ∀ k, 2 ≤ k → k ≤ n - 1 → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1) :
  n = 3 ∨ n = 4 := by
  sorry

end NUMINAMATH_GPT_sequence_satisfies_recurrence_l947_94774


namespace NUMINAMATH_GPT_value_of_a_l947_94794

theorem value_of_a
  (a : ℝ)
  (h1 : ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1)
  (h2 : ∀ (ρ : ℝ), ρ = a)
  (h3 : ∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1 ∧ ρ = a ∧ θ = 0)  :
  a = Real.sqrt 2 / 2 := 
sorry

end NUMINAMATH_GPT_value_of_a_l947_94794


namespace NUMINAMATH_GPT_axis_of_symmetry_sine_function_l947_94783

theorem axis_of_symmetry_sine_function :
  ∃ k : ℤ, x = k * (π / 2) := sorry

end NUMINAMATH_GPT_axis_of_symmetry_sine_function_l947_94783


namespace NUMINAMATH_GPT_parallel_lines_m_eq_one_l947_94795

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y + 8 = 0 ∧ (m + 1) * x + y + (m - 2) = 0 → m = 1) :=
by
  intro x y h
  let L1_slope := -2 / m
  let L2_slope := -(m + 1)
  have h_slope : L1_slope = L2_slope := sorry
  have m_positive : m = 1 := sorry
  exact m_positive

end NUMINAMATH_GPT_parallel_lines_m_eq_one_l947_94795


namespace NUMINAMATH_GPT_sum_of_numbers_l947_94764

theorem sum_of_numbers : 148 + 35 + 17 + 13 + 9 = 222 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_numbers_l947_94764


namespace NUMINAMATH_GPT_find_k_and_other_root_l947_94796

def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_k_and_other_root (k β : ℝ) (h1 : quadratic_eq 4 k 2 (-0.5)) (h2 : 4 * (-0.5) ^ 2 + k * (-0.5) + 2 = 0) : 
  k = 6 ∧ β = -1 ∧ quadratic_eq 4 k 2 β := 
by 
  sorry

end NUMINAMATH_GPT_find_k_and_other_root_l947_94796


namespace NUMINAMATH_GPT_evaluate_ceiling_sum_l947_94778

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end NUMINAMATH_GPT_evaluate_ceiling_sum_l947_94778


namespace NUMINAMATH_GPT_part_a_l947_94793

def A (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x * y) = x * f y

theorem part_a (f : ℝ → ℝ) (h : A f) : ∀ x y : ℝ, f (x + y) = f x + f y :=
sorry

end NUMINAMATH_GPT_part_a_l947_94793


namespace NUMINAMATH_GPT_inheritance_problem_l947_94776

def wifeAmounts (K J M : ℝ) : Prop :=
  K + J + M = 396 ∧
  J = K + 10 ∧
  M = J + 10

def husbandAmounts (wifeAmount : ℝ) (husbandMultiplier : ℝ := 1) : ℝ :=
  husbandMultiplier * wifeAmount

theorem inheritance_problem (K J M : ℝ)
  (h1 : wifeAmounts K J M)
  : ∃ wifeOf : String → String,
    wifeOf "John Smith" = "Katherine" ∧
    wifeOf "Henry Snooks" = "Jane" ∧
    wifeOf "Tom Crow" = "Mary" ∧
    husbandAmounts K = K ∧
    husbandAmounts J 1.5 = 1.5 * J ∧
    husbandAmounts M 2 = 2 * M :=
by 
  sorry

end NUMINAMATH_GPT_inheritance_problem_l947_94776


namespace NUMINAMATH_GPT_abs_square_implication_l947_94747

theorem abs_square_implication (a b : ℝ) (h : abs a > abs b) : a^2 > b^2 :=
by sorry

end NUMINAMATH_GPT_abs_square_implication_l947_94747


namespace NUMINAMATH_GPT_compute_fraction_pow_mul_l947_94782

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end NUMINAMATH_GPT_compute_fraction_pow_mul_l947_94782


namespace NUMINAMATH_GPT_S8_value_l947_94716

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {q : ℝ}

-- The sequence {a_n} is a geometric sequence with common ratio q
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = q * a n

theorem S8_value 
  (h_geo : is_geometric_sequence a q)
  (h_S4 : S 4 = 3)
  (h_S12_S8 : S 12 - S 8 = 12) :
  S 8 = 9 := 
sorry

end NUMINAMATH_GPT_S8_value_l947_94716


namespace NUMINAMATH_GPT_vegetarian_family_l947_94748

theorem vegetarian_family (eat_veg eat_non_veg eat_both : ℕ) (total_veg : ℕ) 
  (h1 : eat_non_veg = 8) (h2 : eat_both = 11) (h3 : total_veg = 26)
  : eat_veg = total_veg - eat_both := by
  sorry

end NUMINAMATH_GPT_vegetarian_family_l947_94748


namespace NUMINAMATH_GPT_fraction_of_boys_participated_l947_94760

-- Definitions based on given conditions
def total_students (B G : ℕ) : Prop := B + G = 800
def participating_girls (G : ℕ) : Prop := (3 / 4 : ℚ) * G = 150
def total_participants (P : ℕ) : Prop := P = 550
def participating_girls_count (PG : ℕ) : Prop := PG = 150

-- Definition of the fraction of participating boys
def fraction_participating_boys (X : ℚ) (B : ℕ) (PB : ℕ) : Prop := X * B = PB

-- The problem of proving the fraction of boys who participated
theorem fraction_of_boys_participated (B G PB : ℕ) (X : ℚ)
  (h1 : total_students B G)
  (h2 : participating_girls G)
  (h3 : total_participants 550)
  (h4 : participating_girls_count 150)
  (h5 : PB = 550 - 150) :
  fraction_participating_boys X B PB → X = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_fraction_of_boys_participated_l947_94760


namespace NUMINAMATH_GPT_find_principal_l947_94719

noncomputable def principal_amount (P : ℝ) : Prop :=
  let r := 0.05
  let t := 2
  let SI := P * r * t
  let CI := P * (1 + r) ^ t - P
  CI - SI = 15

theorem find_principal : principal_amount 6000 :=
by
  simp [principal_amount]
  sorry

end NUMINAMATH_GPT_find_principal_l947_94719


namespace NUMINAMATH_GPT_cannot_determine_congruency_l947_94741

-- Define the congruency criteria for triangles
def SSS (a1 b1 c1 a2 b2 c2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ c1 = c2
def SAS (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2
def ASA (angle1 b1 angle2 angle3 b2 angle4 : ℝ) : Prop := angle1 = angle2 ∧ b1 = b2 ∧ angle3 = angle4
def AAS (angle1 angle2 b1 angle3 angle4 b2 : ℝ) : Prop := angle1 = angle2 ∧ angle3 = angle4 ∧ b1 = b2
def HL (hyp1 leg1 hyp2 leg2 : ℝ) : Prop := hyp1 = hyp2 ∧ leg1 = leg2

-- Define the condition D, which states the equality of two corresponding sides and a non-included angle
def conditionD (a1 b1 angle1 a2 b2 angle2 : ℝ) : Prop := a1 = a2 ∧ b1 = b2 ∧ angle1 = angle2

-- The theorem to be proven
theorem cannot_determine_congruency (a1 b1 angle1 a2 b2 angle2 : ℝ) :
  conditionD a1 b1 angle1 a2 b2 angle2 → ¬(SSS a1 b1 0 a2 b2 0 ∨ SAS a1 b1 0 a2 b2 0 ∨ ASA 0 b1 0 0 b2 0 ∨ AAS 0 0 b1 0 0 b2 ∨ HL 0 0 0 0) :=
by
  sorry

end NUMINAMATH_GPT_cannot_determine_congruency_l947_94741


namespace NUMINAMATH_GPT_dinner_cost_l947_94721

theorem dinner_cost (tax_rate : ℝ) (tip_rate : ℝ) (total_amount : ℝ) : 
  tax_rate = 0.12 → 
  tip_rate = 0.18 → 
  total_amount = 30 → 
  (total_amount / (1 + tax_rate + tip_rate)) = 23.08 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_dinner_cost_l947_94721


namespace NUMINAMATH_GPT_complement_intersection_l947_94773

def A : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | x > 7} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l947_94773


namespace NUMINAMATH_GPT_probability_identical_cubes_l947_94750

-- Definitions translating given conditions
def total_ways_to_paint_single_cube : Nat := 3^6
def total_ways_to_paint_three_cubes : Nat := total_ways_to_paint_single_cube^3

-- Cases counting identical painting schemes
def identical_painting_schemes : Nat :=
  let case_A := 3
  let case_B := 90
  let case_C := 540
  case_A + case_B + case_C

-- The main theorem stating the desired probability
theorem probability_identical_cubes :
  let total_ways := (387420489 : ℚ) -- 729^3
  let favorable_ways := (633 : ℚ)  -- sum of all cases (3 + 90 + 540)
  favorable_ways / total_ways = (211 / 129140163 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_probability_identical_cubes_l947_94750


namespace NUMINAMATH_GPT_min_function_value_l947_94735

theorem min_function_value (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hxyz : x + y + z = 2) :
  (1/3 * x^3 + y^2 + z) = 13/12 :=
sorry

end NUMINAMATH_GPT_min_function_value_l947_94735


namespace NUMINAMATH_GPT_inequality_proof_l947_94777

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l947_94777


namespace NUMINAMATH_GPT_weight_12m_rod_l947_94743

-- Define the weight of a 6 meters long rod
def weight_of_6m_rod : ℕ := 7

-- Given the condition that the weight is proportional to the length
def weight_of_rod (length : ℕ) : ℕ := (length / 6) * weight_of_6m_rod

-- Prove the weight of a 12 meters long rod
theorem weight_12m_rod : weight_of_rod 12 = 14 := by
  -- Calculation skipped, proof required here
  sorry

end NUMINAMATH_GPT_weight_12m_rod_l947_94743


namespace NUMINAMATH_GPT_ball_bounces_less_than_two_meters_l947_94785

theorem ball_bounces_less_than_two_meters : ∀ k : ℕ, 500 * (1/3 : ℝ)^k < 2 → k ≥ 6 := by
  sorry

end NUMINAMATH_GPT_ball_bounces_less_than_two_meters_l947_94785


namespace NUMINAMATH_GPT_find_abc_l947_94789

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) : 
  abc_value a b c = 762 :=
sorry

end NUMINAMATH_GPT_find_abc_l947_94789


namespace NUMINAMATH_GPT_mean_greater_than_median_by_six_l947_94707

theorem mean_greater_than_median_by_six (x : ℕ) : 
  let mean := (x + (x + 2) + (x + 4) + (x + 7) + (x + 37)) / 5
  let median := x + 4
  mean - median = 6 :=
by
  sorry

end NUMINAMATH_GPT_mean_greater_than_median_by_six_l947_94707


namespace NUMINAMATH_GPT_det_abs_eq_one_l947_94742

variable {n : ℕ}
variable {A : Matrix (Fin n) (Fin n) ℤ}
variable {p q r : ℕ}
variable (hpq : p^2 = q^2 + r^2)
variable (hodd : Odd r)
variable (hA : p^2 • A ^ p^2 = q^2 • A ^ q^2 + r^2 • 1)

theorem det_abs_eq_one : |A.det| = 1 := by
  sorry

end NUMINAMATH_GPT_det_abs_eq_one_l947_94742


namespace NUMINAMATH_GPT_new_sequence_after_removal_is_geometric_l947_94751

theorem new_sequence_after_removal_is_geometric (a : ℕ → ℝ) (a₁ q : ℝ) (k : ℕ)
  (h_geo : ∀ n, a n = a₁ * q ^ n) :
  ∀ n, (a (n + k)) = a₁ * q ^ (n + k) :=
by
  sorry

end NUMINAMATH_GPT_new_sequence_after_removal_is_geometric_l947_94751


namespace NUMINAMATH_GPT_neg_p_l947_94765

theorem neg_p : ∀ (m : ℝ), ∀ (x : ℝ), (x^2 + m*x + 1 ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_neg_p_l947_94765


namespace NUMINAMATH_GPT_slices_leftover_l947_94739

def total_initial_slices : ℕ := 12 * 2
def bob_slices : ℕ := 12 / 2
def tom_slices : ℕ := 12 / 3
def sally_slices : ℕ := 12 / 6
def jerry_slices : ℕ := 12 / 4
def total_slices_eaten : ℕ := bob_slices + tom_slices + sally_slices + jerry_slices

theorem slices_leftover : total_initial_slices - total_slices_eaten = 9 := by
  sorry

end NUMINAMATH_GPT_slices_leftover_l947_94739


namespace NUMINAMATH_GPT_angle_terminal_side_on_non_negative_y_axis_l947_94761

theorem angle_terminal_side_on_non_negative_y_axis (P : ℝ × ℝ) (α : ℝ) (hP : P = (0, 3)) :
  α = some_angle_with_terminal_side_on_non_negative_y_axis := by
  sorry

end NUMINAMATH_GPT_angle_terminal_side_on_non_negative_y_axis_l947_94761


namespace NUMINAMATH_GPT_initial_marbles_l947_94763

theorem initial_marbles (M : ℕ) (h1 : M + 9 = 104) : M = 95 := by
  sorry

end NUMINAMATH_GPT_initial_marbles_l947_94763


namespace NUMINAMATH_GPT_certain_number_modulo_l947_94712

theorem certain_number_modulo (x : ℕ) : (57 * x) % 8 = 7 ↔ x = 1 := by
  sorry

end NUMINAMATH_GPT_certain_number_modulo_l947_94712


namespace NUMINAMATH_GPT_repeating_decimal_sum_l947_94768

theorem repeating_decimal_sum :
  (0.6666666666 : ℝ) + (0.7777777777 : ℝ) = (13 : ℚ) / 9 := by
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l947_94768


namespace NUMINAMATH_GPT_largest_of_five_consecutive_sum_180_l947_94729

theorem largest_of_five_consecutive_sum_180 (n : ℕ) (h : n + (n + 1) + (n + 2) + (n + 3) + (n + 4) = 180) :
  n + 4 = 38 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_five_consecutive_sum_180_l947_94729


namespace NUMINAMATH_GPT_hyperbola_asymptote_slope_proof_l947_94720

noncomputable def hyperbola_asymptote_slope : ℝ :=
  let foci_distance := Real.sqrt ((8 - 2)^2 + (3 - 3)^2)
  let c := foci_distance / 2
  let a := 2  -- Given that 2a = 4
  let b := Real.sqrt (c^2 - a^2)
  b / a

theorem hyperbola_asymptote_slope_proof :
  ∀ x y : ℝ, 
  (Real.sqrt ((x - 2)^2 + (y - 3)^2) - Real.sqrt ((x - 8)^2 + (y - 3)^2) = 4) →
  hyperbola_asymptote_slope = Real.sqrt 5 / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_slope_proof_l947_94720


namespace NUMINAMATH_GPT_validity_of_D_l947_94753

def binary_op (a b : ℕ) : ℕ := a^(b + 1)

theorem validity_of_D (a b n : ℕ) (ha : 0 < a) (hb : 0 < b) (hn : 0 < n) :
  binary_op (a^n) b = (binary_op a b)^n := 
by
  sorry

end NUMINAMATH_GPT_validity_of_D_l947_94753


namespace NUMINAMATH_GPT_pseudocode_output_l947_94744

theorem pseudocode_output :
  let s := 0
  let t := 1
  let (s, t) := (List.range 3).foldl (fun (s, t) i => (s + (i + 1), t * (i + 1))) (s, t)
  let r := s * t
  r = 36 :=
by
  sorry

end NUMINAMATH_GPT_pseudocode_output_l947_94744


namespace NUMINAMATH_GPT_tennis_tournament_rounds_l947_94797

/-- Defining the constants and conditions stated in the problem -/
def first_round_games : ℕ := 8
def second_round_games : ℕ := 4
def third_round_games : ℕ := 2
def finals_games : ℕ := 1
def cans_per_game : ℕ := 5
def balls_per_can : ℕ := 3
def total_balls_used : ℕ := 225

/-- Theorem stating the number of rounds in the tennis tournament -/
theorem tennis_tournament_rounds : 
  first_round_games + second_round_games + third_round_games + finals_games = 15 ∧
  15 * cans_per_game = 75 ∧
  75 * balls_per_can = total_balls_used →
  4 = 4 :=
by sorry

end NUMINAMATH_GPT_tennis_tournament_rounds_l947_94797


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l947_94771

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a < 0)
  (h2 : -1 + 2 = b / a) (h3 : -1 * 2 = c / a) :
  (b = a) ∧ (c = -2 * a) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l947_94771


namespace NUMINAMATH_GPT_parentheses_removal_correct_l947_94766

theorem parentheses_removal_correct (x y : ℝ) : -(x^2 + y^2) = -x^2 - y^2 :=
by
  sorry

end NUMINAMATH_GPT_parentheses_removal_correct_l947_94766


namespace NUMINAMATH_GPT_marble_problem_l947_94758

theorem marble_problem
  (x : ℕ) (h1 : 144 / x = 144 / (x + 2) + 1) :
  x = 16 :=
sorry

end NUMINAMATH_GPT_marble_problem_l947_94758


namespace NUMINAMATH_GPT_minimum_value_inequality_l947_94780

theorem minimum_value_inequality
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y - 3 = 0) :
  ∃ t : ℝ, (∀ (x y : ℝ), (2 * x + y = 3) → (0 < x) → (0 < y) → (t = (4 * y - x + 6) / (x * y)) → 9 ≤ t) ∧
          (∃ (x_ y_: ℝ), 2 * x_ + y_ = 3 ∧ 0 < x_ ∧ 0 < y_ ∧ (4 * y_ - x_ + 6) / (x_ * y_) = 9) :=
sorry

end NUMINAMATH_GPT_minimum_value_inequality_l947_94780


namespace NUMINAMATH_GPT_correct_oblique_projection_conclusions_l947_94705

def oblique_projection (shape : Type) : Type := shape

theorem correct_oblique_projection_conclusions :
  (oblique_projection Triangle = Triangle) ∧
  (oblique_projection Parallelogram = Parallelogram) ↔
  (oblique_projection Square ≠ Square) ∧
  (oblique_projection Rhombus ≠ Rhombus) :=
by
  sorry

end NUMINAMATH_GPT_correct_oblique_projection_conclusions_l947_94705


namespace NUMINAMATH_GPT_complement_of_M_l947_94769

def M : Set ℝ := {x | x^2 - 2 * x > 0}

def U : Set ℝ := Set.univ

theorem complement_of_M :
  (U \ M) = (Set.Icc 0 2) :=
by
  sorry

end NUMINAMATH_GPT_complement_of_M_l947_94769


namespace NUMINAMATH_GPT_expr_comparison_l947_94755

-- Define the given condition
def eight_pow_2001 : ℝ := 8 * (64 : ℝ) ^ 1000

-- State the theorem
theorem expr_comparison : (65 : ℝ) ^ 1000 > eight_pow_2001 := by
  sorry

end NUMINAMATH_GPT_expr_comparison_l947_94755


namespace NUMINAMATH_GPT_max_value_of_E_l947_94781

theorem max_value_of_E (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ^ 5 + b ^ 5 = a ^ 3 + b ^ 3) : 
  a^2 - a*b + b^2 ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_value_of_E_l947_94781


namespace NUMINAMATH_GPT_maximum_acute_triangles_from_four_points_l947_94787

-- Define a point in a plane
structure Point (α : Type) := (x : α) (y : α)

-- Definition of an acute triangle is intrinsic to the problem
def is_acute_triangle {α : Type} [LinearOrderedField α] (A B C : Point α) : Prop :=
  sorry -- Assume implementation for determining if a triangle is acute angles based

def maximum_number_acute_triangles {α : Type} [LinearOrderedField α] (A B C D : Point α) : ℕ :=
  sorry -- Assume implementation for verifying maximum number of acute triangles from four points

theorem maximum_acute_triangles_from_four_points {α : Type} [LinearOrderedField α] (A B C D : Point α) :
  maximum_number_acute_triangles A B C D = 4 :=
  sorry

end NUMINAMATH_GPT_maximum_acute_triangles_from_four_points_l947_94787


namespace NUMINAMATH_GPT_union_A_B_eq_real_subset_A_B_l947_94779

def A (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3 + a}
def B : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 1}

theorem union_A_B_eq_real (a : ℝ) : (A a ∪ B) = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 :=
by
  sorry

theorem subset_A_B (a : ℝ) : A a ⊆ B ↔ (a ≤ -4 ∨ a ≥ 1) :=
by
  sorry

end NUMINAMATH_GPT_union_A_B_eq_real_subset_A_B_l947_94779


namespace NUMINAMATH_GPT_optimal_washing_effect_l947_94770

noncomputable def optimal_laundry_addition (x y : ℝ) : Prop :=
  (5 + 0.02 * 2 + x + y = 20) ∧
  (0.02 * 2 + x = (20 - 5) * 0.004)

theorem optimal_washing_effect :
  ∃ x y : ℝ, optimal_laundry_addition x y ∧ x = 0.02 ∧ y = 14.94 :=
by
  sorry

end NUMINAMATH_GPT_optimal_washing_effect_l947_94770


namespace NUMINAMATH_GPT_intersection_sums_l947_94728

theorem intersection_sums :
  (∀ (x y : ℝ), (y = x^3 - 3 * x - 4) → (x + 3 * y = 3) → (∃ (x1 y1 x2 y2 x3 y3 : ℝ),
  (y1 = x1^3 - 3 * x1 - 4) ∧ (x1 + 3 * y1 = 3) ∧
  (y2 = x2^3 - 3 * x2 - 4) ∧ (x2 + 3 * y2 = 3) ∧
  (y3 = x3^3 - 3 * x3 - 4) ∧ (x3 + 3 * y3 = 3) ∧
  x1 + x2 + x3 = 8 / 3 ∧ y1 + y2 + y3 = 19 / 9)) :=
sorry

end NUMINAMATH_GPT_intersection_sums_l947_94728


namespace NUMINAMATH_GPT_mathe_matics_equals_2014_l947_94754

/-- 
Given the following mappings for characters in the word "MATHEMATICS":
M = 1, A = 8, T = 3, E = '+', I = 9, K = '-',
verify that the resulting numerical expression 183 + 1839 - 8 equals 2014.
-/
theorem mathe_matics_equals_2014 :
  183 + 1839 - 8 = 2014 :=
by
  sorry

end NUMINAMATH_GPT_mathe_matics_equals_2014_l947_94754


namespace NUMINAMATH_GPT_greatest_integer_floor_div_l947_94757

-- Define the parameters
def a : ℕ := 3^100 + 2^105
def b : ℕ := 3^96 + 2^101

-- Formulate the proof statement
theorem greatest_integer_floor_div (a b : ℕ) : 
  a = 3^100 + 2^105 →
  b = 3^96 + 2^101 →
  (a / b) = 16 := 
by
  intros ha hb
  sorry

end NUMINAMATH_GPT_greatest_integer_floor_div_l947_94757


namespace NUMINAMATH_GPT_parity_expression_l947_94786

theorem parity_expression
  (a b c : ℕ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_odd : a % 2 = 1)
  (h_b_odd : b % 2 = 1) :
  (5^a + (b + 1)^2 * c) % 2 = 1 :=
by
  sorry

end NUMINAMATH_GPT_parity_expression_l947_94786


namespace NUMINAMATH_GPT_andy_cavities_l947_94733

def candy_canes_from_parents : ℕ := 2
def candy_canes_per_teacher : ℕ := 3
def number_of_teachers : ℕ := 4
def fraction_to_buy : ℚ := 1 / 7
def cavities_per_candies : ℕ := 4

theorem andy_cavities : (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers 
                         + (candy_canes_from_parents 
                         + candy_canes_per_teacher * number_of_teachers) * fraction_to_buy)
                         / cavities_per_candies = 4 := by
  sorry

end NUMINAMATH_GPT_andy_cavities_l947_94733


namespace NUMINAMATH_GPT_interest_rate_bc_l947_94713

def interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * rate * time

def gain_b (interest_bc interest_ab : ℝ) : ℝ :=
  interest_bc - interest_ab

theorem interest_rate_bc :
  ∀ (principal : ℝ) (rate_ab rate_bc : ℝ) (time : ℕ) (gain : ℝ),
    principal = 3500 → rate_ab = 0.10 → time = 3 → gain = 525 →
    interest principal rate_ab time = 1050 →
    gain_b (interest principal rate_bc time) (interest principal rate_ab time) = gain →
    rate_bc = 0.15 :=
by
  intros principal rate_ab rate_bc time gain h_principal h_rate_ab h_time h_gain h_interest_ab h_gain_b
  sorry

end NUMINAMATH_GPT_interest_rate_bc_l947_94713


namespace NUMINAMATH_GPT_Liam_chapters_in_fourth_week_l947_94731

noncomputable def chapters_in_first_week (x : ℕ) : ℕ := x
noncomputable def chapters_in_second_week (x : ℕ) : ℕ := x + 3
noncomputable def chapters_in_third_week (x : ℕ) : ℕ := x + 6
noncomputable def chapters_in_fourth_week (x : ℕ) : ℕ := x + 9
noncomputable def total_chapters (x : ℕ) : ℕ := x + (x + 3) + (x + 6) + (x + 9)

theorem Liam_chapters_in_fourth_week : ∃ x : ℕ, total_chapters x = 50 → chapters_in_fourth_week x = 17 :=
by
  sorry

end NUMINAMATH_GPT_Liam_chapters_in_fourth_week_l947_94731


namespace NUMINAMATH_GPT_mr_johnson_needs_additional_volunteers_l947_94701

-- Definitions for the given conditions
def math_classes := 5
def students_per_class := 4
def total_students := math_classes * students_per_class

def total_teachers := 10
def carpentry_skilled_teachers := 3

def total_parents := 15
def lighting_sound_experienced_parents := 6

def total_volunteers_needed := 100
def carpentry_volunteers_needed := 8
def lighting_sound_volunteers_needed := 10

-- Total current volunteers
def current_volunteers := total_students + total_teachers + total_parents

-- Volunteers with specific skills
def current_carpentry_skilled := carpentry_skilled_teachers
def current_lighting_sound_experienced := lighting_sound_experienced_parents

-- Additional volunteers needed
def additional_carpentry_needed :=
  carpentry_volunteers_needed - current_carpentry_skilled
def additional_lighting_sound_needed :=
  lighting_sound_volunteers_needed - current_lighting_sound_experienced

-- Total additional volunteer needed
def additional_volunteers_needed :=
  additional_carpentry_needed + additional_lighting_sound_needed

-- The theorem we need to prove:
theorem mr_johnson_needs_additional_volunteers :
  additional_volunteers_needed = 9 := by
  sorry

end NUMINAMATH_GPT_mr_johnson_needs_additional_volunteers_l947_94701


namespace NUMINAMATH_GPT_problem_solution_l947_94709

variable {f : ℕ → ℕ}
variable (h_mul : ∀ a b : ℕ, f (a + b) = f a * f b)
variable (h_one : f 1 = 2)

theorem problem_solution : 
  (f 2 / f 1) + (f 4 / f 3) + (f 6 / f 5) + (f 8 / f 7) + (f 10 / f 9) = 10 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l947_94709


namespace NUMINAMATH_GPT_five_more_than_three_in_pages_l947_94717

def pages := (List.range 512).map (λ n => n + 1)

def count_digit (d : Nat) (n : Nat) : Nat :=
  if n = 0 then 0
  else if n % 10 = d then 1 + count_digit d (n / 10)
  else count_digit d (n / 10)

def total_digit_count (d : Nat) (l : List Nat) : Nat :=
  l.foldl (λ acc x => acc + count_digit d x) 0

theorem five_more_than_three_in_pages :
  total_digit_count 5 pages - total_digit_count 3 pages = 22 := 
by 
  sorry

end NUMINAMATH_GPT_five_more_than_three_in_pages_l947_94717


namespace NUMINAMATH_GPT_required_sand_volume_is_five_l947_94723

noncomputable def length : ℝ := 10
noncomputable def depth_cm : ℝ := 50
noncomputable def depth_m : ℝ := depth_cm / 100  -- converting cm to m
noncomputable def width : ℝ := 2
noncomputable def total_volume : ℝ := length * depth_m * width
noncomputable def current_volume : ℝ := total_volume / 2
noncomputable def additional_sand : ℝ := total_volume - current_volume

theorem required_sand_volume_is_five : additional_sand = 5 :=
by sorry

end NUMINAMATH_GPT_required_sand_volume_is_five_l947_94723


namespace NUMINAMATH_GPT_complex_modulus_squared_l947_94734

theorem complex_modulus_squared (z : ℂ) (h : z^2 + Complex.abs z^2 = 4 + 6 * Complex.I) : Complex.abs z^2 = 13 / 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_modulus_squared_l947_94734


namespace NUMINAMATH_GPT_period_and_monotonic_interval_range_of_f_l947_94732

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3 / 3) * Real.cos (2 * x) + Real.sin (x + Real.pi / 4) ^ 2

theorem period_and_monotonic_interval :
  (∃ T > 0, ∀ x, f (x + T) = f x) ∧ 
  (∃ k : ℤ, ∀ x, x ∈ Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12) →
    MonotoneOn f (Set.Icc (2 * k * Real.pi - Real.pi / 2) (2 * k * Real.pi + Real.pi / 2))) :=
sorry

theorem range_of_f (x : ℝ) (hx : x ∈ Set.Icc (-Real.pi / 12) (5 * Real.pi / 12)) :
  f x ∈ Set.Icc 0 (3 / 2) :=
sorry

end NUMINAMATH_GPT_period_and_monotonic_interval_range_of_f_l947_94732


namespace NUMINAMATH_GPT_jane_original_number_l947_94702

theorem jane_original_number (x : ℝ) (h : 5 * (3 * x + 16) = 250) : x = 34 / 3 := 
sorry

end NUMINAMATH_GPT_jane_original_number_l947_94702


namespace NUMINAMATH_GPT_invitation_methods_l947_94724

noncomputable def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem invitation_methods (A B : Type) (students : Finset Type) (h : students.card = 10) :
  (∃ s : Finset Type, s.card = 6 ∧ A ∉ s ∧ B ∉ s) ∧ 
  (∃ t : Finset Type, t.card = 6 ∧ (A ∈ t ∨ B ∉ t)) →
  (combination 10 6 - combination 8 4 = 140) :=
by
  sorry

end NUMINAMATH_GPT_invitation_methods_l947_94724


namespace NUMINAMATH_GPT_smallest_integer_inequality_l947_94788

theorem smallest_integer_inequality (x y z : ℝ) : 
  (x^3 + y^3 + z^3)^2 ≤ 3 * (x^6 + y^6 + z^6) ∧ 
  (∃ n : ℤ, (0 < n ∧ n < 3) → ∀ x y z : ℝ, ¬(x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_inequality_l947_94788


namespace NUMINAMATH_GPT_insurance_covers_80_percent_of_lenses_l947_94759

/--
James needs to get a new pair of glasses. 
His frames cost $200 and the lenses cost $500. 
Insurance will cover a certain percentage of the cost of lenses and he has a $50 off coupon for frames. 
Everything costs $250. 
Prove that the insurance covers 80% of the cost of the lenses.
-/

def frames_cost : ℕ := 200
def lenses_cost : ℕ := 500
def total_cost_after_discounts_and_insurance : ℕ := 250
def coupon : ℕ := 50

theorem insurance_covers_80_percent_of_lenses :
  ((frames_cost - coupon + lenses_cost - total_cost_after_discounts_and_insurance) * 100 / lenses_cost) = 80 := 
  sorry

end NUMINAMATH_GPT_insurance_covers_80_percent_of_lenses_l947_94759


namespace NUMINAMATH_GPT_find_sum_of_terms_l947_94722

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
∃ (r : ℝ), ∀ n : ℕ, a (n + 1) = r * a n

def given_conditions (a : ℕ → ℝ) : Prop :=
geometric_sequence a ∧ (a 4 + a 7 = 2) ∧ (a 5 * a 6 = -8)

theorem find_sum_of_terms (a : ℕ → ℝ) (h : given_conditions a) : a 1 + a 10 = -7 :=
sorry

end NUMINAMATH_GPT_find_sum_of_terms_l947_94722


namespace NUMINAMATH_GPT_simplify_expression_l947_94745

theorem simplify_expression (b : ℝ) : (1 * 3 * b * 4 * b^2 * 5 * b^3 * 6 * b^4) = 360 * b^10 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l947_94745


namespace NUMINAMATH_GPT_one_twentieth_of_eighty_l947_94799

/--
Given the conditions, to prove that \(\frac{1}{20}\) of 80 is equal to 4.
-/
theorem one_twentieth_of_eighty : (80 : ℚ) * (1 / 20) = 4 :=
by
  sorry

end NUMINAMATH_GPT_one_twentieth_of_eighty_l947_94799


namespace NUMINAMATH_GPT_supplement_of_complement_of_65_l947_94798

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end NUMINAMATH_GPT_supplement_of_complement_of_65_l947_94798


namespace NUMINAMATH_GPT_average_speed_monkey_l947_94738

def monkeyDistance : ℝ := 2160
def monkeyTimeMinutes : ℝ := 30
def monkeyTimeSeconds : ℝ := monkeyTimeMinutes * 60

theorem average_speed_monkey :
  (monkeyDistance / monkeyTimeSeconds) = 1.2 := 
sorry

end NUMINAMATH_GPT_average_speed_monkey_l947_94738


namespace NUMINAMATH_GPT_f_eq_zero_of_le_zero_l947_94708

variable {R : Type*} [LinearOrderedField R]
variable {f : R → R}
variable (cond : ∀ x y : R, f (x + y) ≤ y * f x + f (f x))

theorem f_eq_zero_of_le_zero (x : R) (h : x ≤ 0) : f x = 0 :=
sorry

end NUMINAMATH_GPT_f_eq_zero_of_le_zero_l947_94708


namespace NUMINAMATH_GPT_complement_union_M_N_l947_94756

universe u

namespace complement_union

def U : Set (ℝ × ℝ) := { p | true }

def M : Set (ℝ × ℝ) := { p | (p.2 - 3) = (p.1 - 2) }

def N : Set (ℝ × ℝ) := { p | p.2 ≠ (p.1 + 1) }

theorem complement_union_M_N : (U \ (M ∪ N)) = { (2, 3) } := 
by 
  sorry

end complement_union

end NUMINAMATH_GPT_complement_union_M_N_l947_94756


namespace NUMINAMATH_GPT_haley_more_than_josh_l947_94727

-- Definitions of the variables and conditions
variable (H : Nat) -- Number of necklaces Haley has
variable (J : Nat) -- Number of necklaces Jason has
variable (Jos : Nat) -- Number of necklaces Josh has

-- The conditions as assumptions
axiom h1 : H = 25
axiom h2 : H = J + 5
axiom h3 : Jos = J / 2

-- The theorem we want to prove based on these conditions
theorem haley_more_than_josh (H J Jos : Nat) (h1 : H = 25) (h2 : H = J + 5) (h3 : Jos = J / 2) : H - Jos = 15 := 
by 
  sorry

end NUMINAMATH_GPT_haley_more_than_josh_l947_94727


namespace NUMINAMATH_GPT_Lloyd_hourly_rate_is_3_5_l947_94736

/-!
Lloyd normally works 7.5 hours per day and earns a certain amount per hour.
For each hour he works in excess of 7.5 hours on a given day, he is paid 1.5 times his regular rate.
If Lloyd works 10.5 hours on a given day, he earns $42 for that day.
-/

variable (Lloyd_hourly_rate : ℝ)  -- regular hourly rate

def Lloyd_daily_earnings (total_hours : ℝ) (regular_hours : ℝ) (hourly_rate : ℝ) : ℝ :=
  let excess_hours := total_hours - regular_hours
  let excess_earnings := excess_hours * (1.5 * hourly_rate)
  let regular_earnings := regular_hours * hourly_rate
  excess_earnings + regular_earnings

-- Given conditions
axiom H1 : 7.5 = 7.5
axiom H2 : ∀ R : ℝ, Lloyd_hourly_rate = R
axiom H3 : ∀ R : ℝ, ∀ excess_hours : ℝ, Lloyd_hourly_rate + excess_hours = 1.5 * R
axiom H4 : Lloyd_daily_earnings 10.5 7.5 Lloyd_hourly_rate = 42

-- Prove Lloyd earns $3.50 per hour.
theorem Lloyd_hourly_rate_is_3_5 : Lloyd_hourly_rate = 3.5 :=
sorry

end NUMINAMATH_GPT_Lloyd_hourly_rate_is_3_5_l947_94736


namespace NUMINAMATH_GPT_find_z_l947_94790

theorem find_z (x : ℕ) (z : ℚ) (h1 : x = 103)
               (h2 : x^3 * z - 3 * x^2 * z + 2 * x * z = 208170) 
               : z = 5 / 265 := 
by 
  sorry

end NUMINAMATH_GPT_find_z_l947_94790
