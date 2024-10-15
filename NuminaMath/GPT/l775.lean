import Mathlib

namespace NUMINAMATH_GPT_range_a_for_increasing_f_l775_77503

theorem range_a_for_increasing_f :
  (∀ (x : ℝ), 1 ≤ x → (2 * x - 2 * a) ≥ 0) → a ≤ 1 := by
  intro h
  sorry

end NUMINAMATH_GPT_range_a_for_increasing_f_l775_77503


namespace NUMINAMATH_GPT_average_book_width_l775_77504

noncomputable def bookWidths : List ℝ := [5, 0.75, 1.5, 3, 12, 2, 7.5]

theorem average_book_width :
  (bookWidths.sum / bookWidths.length = 4.54) :=
by
  sorry

end NUMINAMATH_GPT_average_book_width_l775_77504


namespace NUMINAMATH_GPT_find_legs_of_triangle_l775_77513

-- Definition of the problem conditions
def right_triangle (x y : ℝ) := x * y = 200 ∧ 4 * (y - 4) = 8 * (x - 8)

-- Theorem we want to prove
theorem find_legs_of_triangle : 
  ∃ (x y : ℝ), right_triangle x y ∧ ((x = 40 ∧ y = 5) ∨ (x = 10 ∧ y = 20)) :=
by
  sorry

end NUMINAMATH_GPT_find_legs_of_triangle_l775_77513


namespace NUMINAMATH_GPT_fraction_of_field_planted_l775_77507

theorem fraction_of_field_planted (a b : ℕ) (d : ℝ) :
  a = 5 → b = 12 → d = 3 →
  let hypotenuse := Real.sqrt (a^2 + b^2)
  let side_square := (d * hypotenuse - d^2)/(a + b - 2 * d)
  let area_square := side_square^2
  let area_triangle : ℝ := 1/2 * a * b
  let planted_area := area_triangle - area_square
  let fraction_planted := planted_area / area_triangle
  fraction_planted = 9693/10140 := by
  sorry

end NUMINAMATH_GPT_fraction_of_field_planted_l775_77507


namespace NUMINAMATH_GPT_boat_stream_ratio_l775_77552

-- Conditions: A man takes twice as long to row a distance against the stream as to row the same distance in favor of the stream.
theorem boat_stream_ratio (B S : ℝ) (h : ∀ (d : ℝ), d / (B - S) = 2 * (d / (B + S))) : B / S = 3 :=
by
  sorry

end NUMINAMATH_GPT_boat_stream_ratio_l775_77552


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l775_77562

theorem arithmetic_sequence_a5 (a : ℕ → ℝ) (h1 : a 1 = 3) (h3 : a 3 = 5) (h_arith : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0) : 
  a 5 = 7 :=
by
  -- proof to be filled later
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l775_77562


namespace NUMINAMATH_GPT_variance_of_dataset_l775_77577

theorem variance_of_dataset (a : ℝ) 
  (h1 : (4 + a + 5 + 3 + 8) / 5 = a) :
  (1 / 5) * ((4 - a) ^ 2 + (a - a) ^ 2 + (5 - a) ^ 2 + (3 - a) ^ 2 + (8 - a) ^ 2) = 14 / 5 :=
by
  sorry

end NUMINAMATH_GPT_variance_of_dataset_l775_77577


namespace NUMINAMATH_GPT_fraction_transform_l775_77551

theorem fraction_transform (x : ℕ) (h : 9 * (537 - x) = 463 + x) : x = 437 :=
by
  sorry

end NUMINAMATH_GPT_fraction_transform_l775_77551


namespace NUMINAMATH_GPT_decreasing_intervals_l775_77566

noncomputable def f (x : ℝ) : ℝ := (x + 2) / (x + 1)

theorem decreasing_intervals : 
  (∀ x y : ℝ, x < y → ((y < -1 ∨ x > -1) → f y < f x)) ∧
  (∀ x y : ℝ, x < y → (y ≥ -1 ∧ x ≤ -1 → f y < f x)) :=
by 
  intros;
  sorry

end NUMINAMATH_GPT_decreasing_intervals_l775_77566


namespace NUMINAMATH_GPT_find_g_two_l775_77532

variable (g : ℝ → ℝ)

-- Condition 1: Functional equation
axiom g_eq : ∀ x y : ℝ, g (x - y) = g x * g y

-- Condition 2: Non-zero property
axiom g_ne_zero : ∀ x : ℝ, g x ≠ 0

-- Proof statement
theorem find_g_two : g 2 = 1 := 
by sorry

end NUMINAMATH_GPT_find_g_two_l775_77532


namespace NUMINAMATH_GPT_length_of_AD_l775_77549

-- Define the segment AD and points B, C, and M as given conditions
variable (x : ℝ) -- Assuming x is the length of segments AB, BC, CD
variable (AD : ℝ)
variable (MC : ℝ)

-- Conditions given in the problem statement
def trisect (AD : ℝ) : Prop :=
  ∃ (x : ℝ), AD = 3 * x ∧ 0 < x

def one_third_way (M AD : ℝ) : Prop :=
  M = AD / 3

def distance_MC (M C : ℝ) : ℝ :=
  C - M

noncomputable def D : Prop := sorry

-- The main theorem statement
theorem length_of_AD (AD : ℝ) (M : ℝ) (MC : ℝ) : trisect AD → one_third_way M AD → MC = M / 3 → AD = 15 :=
by
  intro H1 H2 H3
  -- sorry is added to skip the actual proof
  sorry

end NUMINAMATH_GPT_length_of_AD_l775_77549


namespace NUMINAMATH_GPT_cos_sum_sin_sum_cos_diff_sin_diff_l775_77546

section

variables (A B : ℝ)

-- Definition of cos and sin of angles
def cos (θ : ℝ) : ℝ := sorry
def sin (θ : ℝ) : ℝ := sorry

-- Cosine of the sum of angles
theorem cos_sum : cos (A + B) = cos A * cos B - sin A * sin B := sorry

-- Sine of the sum of angles
theorem sin_sum : sin (A + B) = sin A * cos B + cos A * sin B := sorry

-- Cosine of the difference of angles
theorem cos_diff : cos (A - B) = cos A * cos B + sin A * sin B := sorry

-- Sine of the difference of angles
theorem sin_diff : sin (A - B) = sin A * cos B - cos A * sin B := sorry

end

end NUMINAMATH_GPT_cos_sum_sin_sum_cos_diff_sin_diff_l775_77546


namespace NUMINAMATH_GPT_sqrt7_sub_m_div_n_gt_inv_mn_l775_77559

variables (m n : ℤ)
variables (h_m_nonneg : m ≥ 1) (h_n_nonneg : n ≥ 1)
variables (h_ineq : Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 0)

theorem sqrt7_sub_m_div_n_gt_inv_mn : 
  Real.sqrt 7 - (m : ℝ) / (n : ℝ) > 1 / ((m : ℝ) * (n : ℝ)) :=
by
  sorry

end NUMINAMATH_GPT_sqrt7_sub_m_div_n_gt_inv_mn_l775_77559


namespace NUMINAMATH_GPT_range_of_ab_l775_77586

def circle_eq (x y : ℝ) := x^2 + y^2 + 2 * x - 4 * y + 1 = 0
def line_eq (a b x y : ℝ) := 2 * a * x - b * y + 2 = 0

theorem range_of_ab (a b : ℝ) :
  (∃ x y : ℝ, circle_eq x y ∧ line_eq a b x y) ∧ (∃ x y : ℝ, x = -1 ∧ y = 2) →
  ab <= 1/4 := 
by
  sorry

end NUMINAMATH_GPT_range_of_ab_l775_77586


namespace NUMINAMATH_GPT_remainder_4015_div_32_l775_77536

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end NUMINAMATH_GPT_remainder_4015_div_32_l775_77536


namespace NUMINAMATH_GPT_lily_pads_cover_half_l775_77506

theorem lily_pads_cover_half (P D : ℕ) (cover_entire : P * (2 ^ 25) = D) : P * (2 ^ 24) = D / 2 :=
by sorry

end NUMINAMATH_GPT_lily_pads_cover_half_l775_77506


namespace NUMINAMATH_GPT_complement_union_l775_77533

open Set

def I : Set ℕ := {0, 1, 2, 3}
def M : Set ℕ := {0, 2}
def N : Set ℕ := {0, 2, 3}

theorem complement_union :
  compl (M ∪ N) = {1} :=
by
  sorry

end NUMINAMATH_GPT_complement_union_l775_77533


namespace NUMINAMATH_GPT_circle_radius_l775_77573

theorem circle_radius (M N : ℝ) (h1 : M / N = 20) :
  ∃ r : ℝ, M = π * r^2 ∧ N = 2 * π * r ∧ r = 40 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_l775_77573


namespace NUMINAMATH_GPT_divisible_by_27000_l775_77558

theorem divisible_by_27000 (k : ℕ) (h₁ : k = 30) : ∃ n : ℕ, k^3 = 27000 * n :=
by {
  sorry
}

end NUMINAMATH_GPT_divisible_by_27000_l775_77558


namespace NUMINAMATH_GPT_total_number_of_coins_l775_77502

theorem total_number_of_coins (x : ℕ) :
  5 * x + 10 * x + 25 * x = 120 → 3 * x = 9 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_total_number_of_coins_l775_77502


namespace NUMINAMATH_GPT_value_of_expression_l775_77585

theorem value_of_expression (a b : ℝ) (h1 : 3 * a^2 + 9 * a - 21 = 0) (h2 : 3 * b^2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (5 * b - 6) = -27 :=
by
  -- The proof is omitted, place 'sorry' to indicate it.
  sorry

end NUMINAMATH_GPT_value_of_expression_l775_77585


namespace NUMINAMATH_GPT_negation_of_proposition_l775_77550

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1)) ↔
  (∃ x₀ : ℝ, x₀ ≤ 0 ∧ (x₀ + 1) * Real.exp x₀ ≤ 1) := 
sorry

end NUMINAMATH_GPT_negation_of_proposition_l775_77550


namespace NUMINAMATH_GPT_largest_common_divisor_476_330_l775_77572

theorem largest_common_divisor_476_330 :
  ∀ (S₁ S₂ : Finset ℕ), 
    S₁ = {1, 2, 4, 7, 14, 28, 17, 34, 68, 119, 238, 476} → 
    S₂ = {1, 2, 3, 5, 6, 10, 11, 15, 22, 30, 33, 55, 66, 110, 165, 330} → 
    ∃ D, D ∈ S₁ ∧ D ∈ S₂ ∧ ∀ x, x ∈ S₁ ∧ x ∈ S₂ → x ≤ D ∧ D = 2 :=
by
  intros S₁ S₂ hS₁ hS₂
  use 2
  sorry

end NUMINAMATH_GPT_largest_common_divisor_476_330_l775_77572


namespace NUMINAMATH_GPT_inequality_reciprocal_l775_77547

theorem inequality_reciprocal (a b : ℝ) (h₀ : a < b) (h₁ : b < 0) : (1 / a) > (1 / b) :=
sorry

end NUMINAMATH_GPT_inequality_reciprocal_l775_77547


namespace NUMINAMATH_GPT_michael_total_weight_loss_l775_77570

def weight_loss_march := 3
def weight_loss_april := 4
def weight_loss_may := 3

theorem michael_total_weight_loss : weight_loss_march + weight_loss_april + weight_loss_may = 10 := by
  sorry

end NUMINAMATH_GPT_michael_total_weight_loss_l775_77570


namespace NUMINAMATH_GPT_strawberries_weight_before_l775_77594

variables (M D E B : ℝ)

noncomputable def total_weight_before (M D E : ℝ) := M + D - E

theorem strawberries_weight_before :
  ∀ (M D E : ℝ), M = 36 ∧ D = 16 ∧ E = 30 → total_weight_before M D E = 22 :=
by
  intros M D E h
  simp [total_weight_before, h]
  sorry

end NUMINAMATH_GPT_strawberries_weight_before_l775_77594


namespace NUMINAMATH_GPT_banana_cantaloupe_cost_l775_77509

theorem banana_cantaloupe_cost {a b c d : ℕ} 
  (h1 : a + b + c + d = 20) 
  (h2 : d = 2 * a)
  (h3 : c = a - b) : b + c = 5 :=
sorry

end NUMINAMATH_GPT_banana_cantaloupe_cost_l775_77509


namespace NUMINAMATH_GPT_Teresa_age_when_Michiko_born_l775_77576

theorem Teresa_age_when_Michiko_born 
  (Teresa_age : ℕ) (Morio_age : ℕ) (Michiko_born_age : ℕ) (Kenji_diff : ℕ)
  (Emiko_diff : ℕ) (Hideki_same_as_Kenji : Prop) (Ryuji_age_same_as_Morio : Prop)
  (h1 : Teresa_age = 59) 
  (h2 : Morio_age = 71) 
  (h3 : Morio_age = Michiko_born_age + 33)
  (h4 : Kenji_diff = 4)
  (h5 : Emiko_diff = 10)
  (h6 : Hideki_same_as_Kenji = True)
  (h7 : Ryuji_age_same_as_Morio = True) : 
  ∃ Michiko_age Hideki_age Michiko_Hideki_diff Teresa_birth_age,
    Michiko_age = 33 ∧ 
    Hideki_age = 29 ∧ 
    Michiko_Hideki_diff = 4 ∧ 
    Teresa_birth_age = 26 :=
sorry

end NUMINAMATH_GPT_Teresa_age_when_Michiko_born_l775_77576


namespace NUMINAMATH_GPT_joining_fee_per_person_l775_77535

variables (F : ℝ)
variables (family_members : ℕ) (monthly_cost_per_person : ℝ) (john_yearly_payment : ℝ)

def total_cost (F : ℝ) (family_members : ℕ) (monthly_cost_per_person : ℝ) : ℝ :=
  family_members * (F + 12 * monthly_cost_per_person)

theorem joining_fee_per_person :
  (family_members = 4) →
  (monthly_cost_per_person = 1000) →
  (john_yearly_payment = 32000) →
  john_yearly_payment = 0.5 * total_cost F family_members monthly_cost_per_person →
  F = 4000 :=
by
  intros h_family h_monthly_cost h_yearly_payment h_eq
  sorry

end NUMINAMATH_GPT_joining_fee_per_person_l775_77535


namespace NUMINAMATH_GPT_sum_with_extra_five_l775_77571

theorem sum_with_extra_five 
  (a b c : ℕ)
  (h1 : a + b = 31)
  (h2 : b + c = 48)
  (h3 : c + a = 55) : 
  a + b + c + 5 = 72 :=
by
  sorry

end NUMINAMATH_GPT_sum_with_extra_five_l775_77571


namespace NUMINAMATH_GPT_rate_second_year_l775_77544

/-- Define the principal amount at the start. -/
def P : ℝ := 4000

/-- Define the rate of interest for the first year. -/
def rate_first_year : ℝ := 0.04

/-- Define the final amount after 2 years. -/
def A : ℝ := 4368

/-- Define the amount after the first year. -/
def P1 : ℝ := P + P * rate_first_year

/-- Define the interest for the second year. -/
def Interest2 : ℝ := A - P1

/-- Define the principal amount for the second year, which is the amount after the first year. -/
def P2 : ℝ := P1

/-- Prove that the rate of interest for the second year is 5%. -/
theorem rate_second_year : (Interest2 / P2) * 100 = 5 :=
by
  sorry

end NUMINAMATH_GPT_rate_second_year_l775_77544


namespace NUMINAMATH_GPT_libby_quarters_left_l775_77561

theorem libby_quarters_left (initial_quarters : ℕ) (dress_cost_dollars : ℕ) (quarters_per_dollar : ℕ) 
  (h1 : initial_quarters = 160) (h2 : dress_cost_dollars = 35) (h3 : quarters_per_dollar = 4) : 
  initial_quarters - (dress_cost_dollars * quarters_per_dollar) = 20 := by
  sorry

end NUMINAMATH_GPT_libby_quarters_left_l775_77561


namespace NUMINAMATH_GPT_hall_paving_l775_77563

theorem hall_paving :
  ∀ (hall_length hall_breadth stone_length stone_breadth : ℕ),
    hall_length = 72 →
    hall_breadth = 30 →
    stone_length = 8 →
    stone_breadth = 10 →
    let Area_hall := hall_length * hall_breadth
    let Length_stone := stone_length / 10
    let Breadth_stone := stone_breadth / 10
    let Area_stone := Length_stone * Breadth_stone 
    (Area_hall / Area_stone) = 2700 :=
by
  intros hall_length hall_breadth stone_length stone_breadth
  intro h1 h2 h3 h4
  let Area_hall := hall_length * hall_breadth
  let Length_stone := stone_length / 10
  let Breadth_stone := stone_breadth / 10
  let Area_stone := Length_stone * Breadth_stone 
  have h5 : Area_hall / Area_stone = 2700 := sorry
  exact h5

end NUMINAMATH_GPT_hall_paving_l775_77563


namespace NUMINAMATH_GPT_find_real_roots_of_PQ_l775_77554

noncomputable def P (x b : ℝ) : ℝ := x^2 + x / 2 + b
noncomputable def Q (x c d : ℝ) : ℝ := x^2 + c * x + d

theorem find_real_roots_of_PQ (b c d : ℝ)
  (h: ∀ x : ℝ, P x b * Q x c d = Q (P x b) c d)
  (h_d_zero: d = 0) :
  ∃ x : ℝ, P (Q x c d) b = 0 → x = (-c + Real.sqrt (c^2 + 2)) / 2 ∨ x = (-c - Real.sqrt (c^2 + 2)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_find_real_roots_of_PQ_l775_77554


namespace NUMINAMATH_GPT_alcohol_water_ratio_l775_77527

theorem alcohol_water_ratio (alcohol water : ℝ) (h_alcohol : alcohol = 3 / 5) (h_water : water = 2 / 5) :
  alcohol / water = 3 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_alcohol_water_ratio_l775_77527


namespace NUMINAMATH_GPT_major_axis_length_l775_77579

theorem major_axis_length (radius : ℝ) (k : ℝ) (minor_axis : ℝ) (major_axis : ℝ)
  (cyl_radius : radius = 2)
  (minor_eq_diameter : minor_axis = 2 * radius)
  (major_longer : major_axis = minor_axis * (1 + k))
  (k_value : k = 0.25) :
  major_axis = 5 :=
by
  -- Proof omitted, using sorry
  sorry

end NUMINAMATH_GPT_major_axis_length_l775_77579


namespace NUMINAMATH_GPT_value_of_a_8_l775_77596

-- Definitions of the sequence and sum of first n terms
def sum_first_terms (S : ℕ → ℕ) := ∀ n : ℕ, n > 0 → S n = n^2

-- Definition of the term a_n
def a_n (S : ℕ → ℕ) (n : ℕ) := S n - S (n - 1)

-- The theorem we want to prove: a_8 = 15
theorem value_of_a_8 (S : ℕ → ℕ) (h_sum : sum_first_terms S) : a_n S 8 = 15 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_8_l775_77596


namespace NUMINAMATH_GPT_value_of_a_l775_77560

theorem value_of_a (a : ℝ) (h : 1 ∈ ({a, a ^ 2} : Set ℝ)) : a = -1 :=
sorry

end NUMINAMATH_GPT_value_of_a_l775_77560


namespace NUMINAMATH_GPT_fraction_simplification_l775_77529

theorem fraction_simplification :
  (3100 - 3037)^2 / 81 = 49 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l775_77529


namespace NUMINAMATH_GPT_smallest_m_l775_77518

theorem smallest_m (m : ℤ) :
  (∀ x : ℝ, (3 * x * (m * x - 5) - x^2 + 8) = 0) → (257 - 96 * m < 0) → (m = 3) :=
sorry

end NUMINAMATH_GPT_smallest_m_l775_77518


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l775_77591

theorem sum_of_consecutive_integers (n : ℕ) (a : ℕ) (h : n ≥ 1) (h_sum : n * (2 * a + n - 1) = 56) : n ≤ 7 := 
by
  sorry

theorem largest_set_of_consecutive_positive_integers : ∃ n a, n ≥ 1 ∧ n * (2 * a + n - 1) = 56 ∧ n = 7 := 
by
  use 7, 1
  repeat {split}
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_largest_set_of_consecutive_positive_integers_l775_77591


namespace NUMINAMATH_GPT_distances_from_median_l775_77592

theorem distances_from_median (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  ∃ (x y : ℝ), x = (b * c) / (a + b) ∧ y = (a * c) / (a + b) ∧ x + y = c :=
by
  sorry

end NUMINAMATH_GPT_distances_from_median_l775_77592


namespace NUMINAMATH_GPT_negated_proposition_false_l775_77578

theorem negated_proposition_false : ¬ ∀ x : ℝ, 2^x + x^2 > 1 :=
by 
sorry

end NUMINAMATH_GPT_negated_proposition_false_l775_77578


namespace NUMINAMATH_GPT_consecutive_integers_avg_l775_77526

theorem consecutive_integers_avg (n x : ℤ) (h_avg : (2*x + n - 1 : ℝ)/2 = 20.5) (h_10th : x + 9 = 25) :
  n = 10 :=
by
  sorry

end NUMINAMATH_GPT_consecutive_integers_avg_l775_77526


namespace NUMINAMATH_GPT_ratio_of_sum_l775_77574

theorem ratio_of_sum (x y : ℚ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_sum_l775_77574


namespace NUMINAMATH_GPT_average_speed_trip_l775_77534

-- Conditions: Definitions
def distance_north_feet : ℝ := 5280
def speed_north_mpm : ℝ := 2
def speed_south_mpm : ℝ := 1

-- Question and Equivalent Proof Problem
theorem average_speed_trip :
  let distance_north_miles := distance_north_feet / 5280
  let distance_south_miles := 2 * distance_north_miles
  let total_distance_miles := distance_north_miles + distance_south_miles + distance_south_miles
  let time_north_hours := distance_north_miles / speed_north_mpm / 60
  let time_south_hours := distance_south_miles / speed_south_mpm / 60
  let time_return_hours := distance_south_miles / speed_south_mpm / 60
  let total_time_hours := time_north_hours + time_south_hours + time_return_hours
  let average_speed_mph := total_distance_miles / total_time_hours
  average_speed_mph = 76.4 := by
    sorry

end NUMINAMATH_GPT_average_speed_trip_l775_77534


namespace NUMINAMATH_GPT_division_of_exponents_l775_77542

-- Define the conditions as constants and statements that we are concerned with
variables (x : ℝ)

-- The Lean 4 statement of the equivalent proof problem
theorem division_of_exponents (h₁ : x ≠ 0) : x^8 / x^2 = x^6 := 
sorry

end NUMINAMATH_GPT_division_of_exponents_l775_77542


namespace NUMINAMATH_GPT_area_of_BCD_l775_77583

theorem area_of_BCD (S_ABC : ℝ) (a_CD : ℝ) (h_ratio : ℝ) (h_ABC : ℝ) :
  S_ABC = 36 ∧ a_CD = 30 ∧ h_ratio = 0.5 ∧ h_ABC = 12 → 
  (1 / 2) * a_CD * (h_ratio * h_ABC) = 90 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_area_of_BCD_l775_77583


namespace NUMINAMATH_GPT_repeating_decimal_sum_l775_77537

noncomputable def x : ℚ := 2 / 9
noncomputable def y : ℚ := 1 / 33

theorem repeating_decimal_sum :
  x + y = 25 / 99 :=
by
  -- Note that Lean can automatically simplify rational expressions.
  sorry

end NUMINAMATH_GPT_repeating_decimal_sum_l775_77537


namespace NUMINAMATH_GPT_initial_people_count_25_l775_77595

-- Definition of the initial number of people (X) and the condition
def initial_people (X : ℕ) : Prop := X - 8 + 13 = 30

-- The theorem stating that the initial number of people is 25
theorem initial_people_count_25 : ∃ (X : ℕ), initial_people X ∧ X = 25 :=
by
  -- We add sorry here to skip the actual proof
  sorry

end NUMINAMATH_GPT_initial_people_count_25_l775_77595


namespace NUMINAMATH_GPT_sum_of_squares_iff_double_l775_77593

theorem sum_of_squares_iff_double (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_iff_double_l775_77593


namespace NUMINAMATH_GPT_beka_flew_more_l775_77505

def bekaMiles := 873
def jacksonMiles := 563

theorem beka_flew_more : bekaMiles - jacksonMiles = 310 := by
  -- proof here
  sorry

end NUMINAMATH_GPT_beka_flew_more_l775_77505


namespace NUMINAMATH_GPT_shaded_triangle_area_l775_77575

-- Definitions and conditions
def grid_width : ℕ := 15
def grid_height : ℕ := 5

def larger_triangle_base : ℕ := grid_width
def larger_triangle_height : ℕ := grid_height - 1

def smaller_triangle_base : ℕ := 12
def smaller_triangle_height : ℕ := 3

-- The proof problem stating that the area of the smaller shaded triangle is 18 units
theorem shaded_triangle_area :
  (smaller_triangle_base * smaller_triangle_height) / 2 = 18 :=
by
  sorry

end NUMINAMATH_GPT_shaded_triangle_area_l775_77575


namespace NUMINAMATH_GPT_trapezoid_perimeter_l775_77557

theorem trapezoid_perimeter (AB CD AD BC h : ℝ)
  (AB_eq : AB = 40)
  (CD_eq : CD = 70)
  (AD_eq_BC : AD = BC)
  (h_eq : h = 24)
  : AB + BC + CD + AD = 110 + 2 * Real.sqrt 801 :=
by
  -- Proof goes here, you can replace this comment with actual proof.
  sorry

end NUMINAMATH_GPT_trapezoid_perimeter_l775_77557


namespace NUMINAMATH_GPT_percentage_increase_after_decrease_l775_77556

variable (P : ℝ) (x : ℝ)

-- Conditions
def decreased_price : ℝ := 0.80 * P
def final_price_condition : Prop := 0.80 * P + (x / 100) * (0.80 * P) = 1.04 * P
def correct_answer : Prop := x = 30

-- The proof goal
theorem percentage_increase_after_decrease : final_price_condition P x → correct_answer x :=
by sorry

end NUMINAMATH_GPT_percentage_increase_after_decrease_l775_77556


namespace NUMINAMATH_GPT_intersection_A_B_l775_77520

open Set

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | ∃ a ∈ A, x = 2 * a}

theorem intersection_A_B : A ∩ B = {0, 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_l775_77520


namespace NUMINAMATH_GPT_net_effect_on_sale_value_l775_77524

theorem net_effect_on_sale_value (P Q : ℝ) (hP : P > 0) (hQ : Q > 0) :
  let original_sale_value := P * Q
  let new_price := 0.82 * P
  let new_quantity := 1.88 * Q
  let new_sale_value := new_price * new_quantity
  let net_effect := (new_sale_value / original_sale_value - 1) * 100
  net_effect = 54.16 :=
by
  sorry

end NUMINAMATH_GPT_net_effect_on_sale_value_l775_77524


namespace NUMINAMATH_GPT_perimeter_of_semi_circle_region_l775_77530

theorem perimeter_of_semi_circle_region (side_length : ℝ) (h : side_length = 1/π) : 
  let radius := side_length / 2
  let circumference_of_half_circle := (1 / 2) * π * side_length
  3 * circumference_of_half_circle = 3 / 2
  := by
  sorry

end NUMINAMATH_GPT_perimeter_of_semi_circle_region_l775_77530


namespace NUMINAMATH_GPT_leesburg_population_l775_77525

theorem leesburg_population (salem_population leesburg_population half_salem_population number_moved_out : ℕ)
  (h1 : half_salem_population * 2 = salem_population)
  (h2 : salem_population - number_moved_out = 754100)
  (h3 : salem_population = 15 * leesburg_population)
  (h4 : half_salem_population = 377050)
  (h5 : number_moved_out = 130000) :
  leesburg_population = 58940 :=
by
  sorry

end NUMINAMATH_GPT_leesburg_population_l775_77525


namespace NUMINAMATH_GPT_largest_int_square_3_digits_base_7_l775_77501

theorem largest_int_square_3_digits_base_7 :
  ∃ (N : ℕ), (7^2 ≤ N^2) ∧ (N^2 < 7^3) ∧ 
  ∃ k : ℕ, N = k ∧ k^2 ≥ 7^2 ∧ k^2 < 7^3 ∧
  N = 45 := sorry

end NUMINAMATH_GPT_largest_int_square_3_digits_base_7_l775_77501


namespace NUMINAMATH_GPT_find_C_l775_77598

theorem find_C (A B C : ℕ) (h1 : A + B + C = 900) (h2 : A + C = 400) (h3 : B + C = 750) : C = 250 :=
by
  sorry

end NUMINAMATH_GPT_find_C_l775_77598


namespace NUMINAMATH_GPT_vector_dot_product_value_l775_77512

def a : ℝ × ℝ := (1, -2)
def b : ℝ × ℝ := (-3, 4)
def c : ℝ × ℝ := (3, 2)

def scalar_mul (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ := (k * v.1, k * v.2)
def add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

theorem vector_dot_product_value : dot_product (add (scalar_mul 2 a) b) c = -3 := by
  sorry

end NUMINAMATH_GPT_vector_dot_product_value_l775_77512


namespace NUMINAMATH_GPT_power_function_k_values_l775_77567

theorem power_function_k_values (k : ℝ) : (∃ (a : ℝ), (k^2 - k - 5) = a) → (k = 3 ∨ k = -2) :=
by
  intro h
  have h1 : k^2 - k - 5 = 1 := sorry -- Using the condition that it is a power function
  have h2 : k^2 - k - 6 = 0 := by linarith -- Simplify the equation
  exact sorry -- Solve the quadratic equation

end NUMINAMATH_GPT_power_function_k_values_l775_77567


namespace NUMINAMATH_GPT_probability_of_multiples_of_4_l775_77514

def number_of_multiples_of_4 (n : ℕ) : ℕ :=
  n / 4

def number_not_multiples_of_4 (n : ℕ) (m : ℕ) : ℕ :=
  n - m

def probability_neither_multiples_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  (m / n : ℚ) * (m / n)

def probability_at_least_one_multiple_of_4 (n : ℕ) (m : ℕ) : ℚ :=
  1 - probability_neither_multiples_of_4 n m

theorem probability_of_multiples_of_4 :
  probability_at_least_one_multiple_of_4 60 45 = 7 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_multiples_of_4_l775_77514


namespace NUMINAMATH_GPT_complex_inverse_l775_77582

noncomputable def complex_expression (i : ℂ) (h_i : i ^ 2 = -1) : ℂ :=
  (3 * i - 3 * (1 / i))⁻¹

theorem complex_inverse (i : ℂ) (h_i : i^2 = -1) :
  complex_expression i h_i = -i / 6 :=
by
  -- the proof part is omitted
  sorry

end NUMINAMATH_GPT_complex_inverse_l775_77582


namespace NUMINAMATH_GPT_six_inch_cube_value_is_2700_l775_77555

noncomputable def value_of_six_inch_cube (value_four_inch_cube : ℕ) : ℕ :=
  let volume_four_inch_cube := 4^3
  let volume_six_inch_cube := 6^3
  let scaling_factor := volume_six_inch_cube / volume_four_inch_cube
  value_four_inch_cube * scaling_factor

theorem six_inch_cube_value_is_2700 : value_of_six_inch_cube 800 = 2700 := by
  sorry

end NUMINAMATH_GPT_six_inch_cube_value_is_2700_l775_77555


namespace NUMINAMATH_GPT_absolute_value_inequality_solution_l775_77519

theorem absolute_value_inequality_solution (x : ℝ) : abs (x - 3) < 2 ↔ 1 < x ∧ x < 5 :=
by
  sorry

end NUMINAMATH_GPT_absolute_value_inequality_solution_l775_77519


namespace NUMINAMATH_GPT_percentage_of_number_l775_77500

theorem percentage_of_number (X P : ℝ) (h1 : 0.20 * X = 80) (h2 : (P / 100) * X = 160) : P = 40 := by
  sorry

end NUMINAMATH_GPT_percentage_of_number_l775_77500


namespace NUMINAMATH_GPT_no_solution_equation_l775_77528

theorem no_solution_equation (x : ℝ) : (x + 1) / (x - 1) + 4 / (1 - x^2) ≠ 1 :=
  sorry

end NUMINAMATH_GPT_no_solution_equation_l775_77528


namespace NUMINAMATH_GPT_average_visitors_remaining_days_l775_77516

-- Definitions
def visitors_monday := 50
def visitors_tuesday := 2 * visitors_monday
def total_week_visitors := 250
def days_remaining := 5
def remaining_visitors := total_week_visitors - (visitors_monday + visitors_tuesday)
def average_remaining_visitors_per_day := remaining_visitors / days_remaining

-- Theorem statement
theorem average_visitors_remaining_days : average_remaining_visitors_per_day = 20 :=
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_average_visitors_remaining_days_l775_77516


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l775_77521

theorem solve_equation_1 (x : ℝ) : x^2 - 2 * x = 0 ↔ x = 0 ∨ x = 2 := 
sorry

theorem solve_equation_2 (x : ℝ) : (2 * x - 1)^2 = (3 - x)^2 ↔ x = -2 ∨ x = 4 / 3 := 
sorry

theorem solve_equation_3 (x : ℝ) : 3 * x * (x - 2) = x - 2 ↔ x = 2 ∨ x = 1 / 3 :=
sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_solve_equation_3_l775_77521


namespace NUMINAMATH_GPT_problem_solution_l775_77523

theorem problem_solution (x y : ℝ) (h₁ : (4 * y^2 + 1) * (x^4 + 2 * x^2 + 2) = 8 * |y| * (x^2 + 1))
  (h₂ : y ≠ 0) :
  (x = 0 ∧ (y = 1/2 ∨ y = -1/2)) :=
by {
  sorry -- Proof required
}

end NUMINAMATH_GPT_problem_solution_l775_77523


namespace NUMINAMATH_GPT_sum_of_decimals_l775_77584

theorem sum_of_decimals : 5.27 + 4.19 = 9.46 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_decimals_l775_77584


namespace NUMINAMATH_GPT_intersection_eq_l775_77599

noncomputable def U : Set ℝ := Set.univ
def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℝ := {x | 2 ≤ x ∧ x < 3}

-- Complement of B in U
def complement_B : Set ℝ := {x | x < 2 ∨ x ≥ 3}

-- Intersection of A and complement of B
def intersection : Set ℕ := {x ∈ A | ↑x < 2 ∨ ↑x ≥ 3}

theorem intersection_eq : intersection = {1, 3, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l775_77599


namespace NUMINAMATH_GPT_min_value_of_expression_l775_77538

theorem min_value_of_expression (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : 4 * x + 3 * y = 1) :
  1 / (2 * x - y) + 2 / (x + 2 * y) = 9 :=
sorry

end NUMINAMATH_GPT_min_value_of_expression_l775_77538


namespace NUMINAMATH_GPT_find_constants_l775_77597

theorem find_constants (a b : ℚ) (h1 : 3 * a + b = 7) (h2 : a + 4 * b = 5) :
  a = 61 / 33 ∧ b = 8 / 11 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_l775_77597


namespace NUMINAMATH_GPT_semicircle_area_l775_77548

theorem semicircle_area (x : ℝ) (y : ℝ) (r : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : x^2 + y^2 = (2*r)^2) :
  (1/2) * π * r^2 = (13 * π) / 8 :=
by
  sorry

end NUMINAMATH_GPT_semicircle_area_l775_77548


namespace NUMINAMATH_GPT_triangle_shape_l775_77522

theorem triangle_shape (a b : ℝ) (A B : ℝ) (hA : 0 < A) (hB : A < π) (h : a * Real.cos A = b * Real.cos B) :
  (A = B ∨ A + B = π / 2 ∨ a = b) :=
by
  sorry

end NUMINAMATH_GPT_triangle_shape_l775_77522


namespace NUMINAMATH_GPT_min_value_inequality_l775_77565

open Real

theorem min_value_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + 3 * y = 4) :
  ∃ z, z = (2 / x + 3 / y) ∧ z = 25 / 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l775_77565


namespace NUMINAMATH_GPT_messenger_speed_l775_77539

noncomputable def team_length : ℝ := 6

noncomputable def team_speed : ℝ := 5

noncomputable def total_time : ℝ := 0.5

theorem messenger_speed (x : ℝ) :
  (6 / (x + team_speed) + 6 / (x - team_speed) = total_time) →
  x = 25 := by
  sorry

end NUMINAMATH_GPT_messenger_speed_l775_77539


namespace NUMINAMATH_GPT_sum_of_three_numbers_l775_77545

theorem sum_of_three_numbers (a b c : ℕ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 12)
    (h4 : (a + b + c) / 3 = a + 8) (h5 : (a + b + c) / 3 = c - 18) : 
    a + b + c = 66 := 
sorry

end NUMINAMATH_GPT_sum_of_three_numbers_l775_77545


namespace NUMINAMATH_GPT_problem_l775_77517

theorem problem
  (x y : ℝ)
  (h1 : x + 3 * y = 9)
  (h2 : x * y = -27) :
  x^2 + 9 * y^2 = 243 :=
sorry

end NUMINAMATH_GPT_problem_l775_77517


namespace NUMINAMATH_GPT_natalie_height_l775_77589

variable (height_Natalie height_Harpreet height_Jiayin : ℝ)
variable (h1 : height_Natalie = height_Harpreet)
variable (h2 : height_Jiayin = 161)
variable (h3 : (height_Natalie + height_Harpreet + height_Jiayin) / 3 = 171)

theorem natalie_height : height_Natalie = 176 :=
by 
  sorry

end NUMINAMATH_GPT_natalie_height_l775_77589


namespace NUMINAMATH_GPT_range_of_a_l775_77569

def f (x a : ℝ) : ℝ := x^2 + 2 * (a - 2) * x + 5

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x ≥ 4 → f x a ≤ f (x+1) a) : a ≥ -2 := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_l775_77569


namespace NUMINAMATH_GPT_at_least_two_even_l775_77511

theorem at_least_two_even (x y z : ℤ) (u : ℤ)
  (h : x^2 + y^2 + z^2 = u^2) : (↑x % 2 = 0) ∨ (↑y % 2 = 0) → (↑x % 2 = 0) ∨ (↑z % 2 = 0) ∨ (↑y % 2 = 0) := 
by
  sorry

end NUMINAMATH_GPT_at_least_two_even_l775_77511


namespace NUMINAMATH_GPT_jill_draws_spade_probability_l775_77515

noncomputable def probability_jill_draws_spade : ℚ :=
  ∑' (k : ℕ), ((3 / 4) * (3 / 4))^k * ((3 / 4) * (1 / 4))

theorem jill_draws_spade_probability : probability_jill_draws_spade = 3 / 7 :=
sorry

end NUMINAMATH_GPT_jill_draws_spade_probability_l775_77515


namespace NUMINAMATH_GPT_xy_over_y_plus_x_l775_77564

theorem xy_over_y_plus_x {x y z : ℝ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : 1/x + 1/y = 1/z) : z = xy/(y+x) :=
sorry

end NUMINAMATH_GPT_xy_over_y_plus_x_l775_77564


namespace NUMINAMATH_GPT_arithmetic_sequence_integers_l775_77541

theorem arithmetic_sequence_integers (a3 a18 : ℝ) (d : ℝ) (n : ℕ)
  (h3 : a3 = 14) (h18 : a18 = 23) (hd : d = 0.6)
  (hn : n = 2010) : 
  (∃ (k : ℕ), n = 5 * (k + 1) - 2) ∧ (k ≤ 401) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_integers_l775_77541


namespace NUMINAMATH_GPT_problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l775_77510

/-- Lean statement for the math proof problem -/

/- First problem -/
theorem problem1_equation_of_line_intersection_perpendicular :
  ∃ k, 3 * k - 2 * ( - (5 - 3 * k) / 2) - 11 = 0 :=
sorry

/- Second problem -/
theorem problem2_equation_of_line_point_equal_intercepts :
  (∃ a, (1, 2) ∈ {(x, y) | x + y = a}) ∧ a = 3
  ∨ (∃ b, (1, 2) ∈ {(x, y) | y = b * x}) ∧ b = 2 :=
sorry

end NUMINAMATH_GPT_problem1_equation_of_line_intersection_perpendicular_problem2_equation_of_line_point_equal_intercepts_l775_77510


namespace NUMINAMATH_GPT_tank_full_volume_l775_77553

theorem tank_full_volume (x : ℝ) (h1 : 5 / 6 * x > 0) (h2 : 5 / 6 * x - 15 = 1 / 3 * x) : x = 30 :=
by
  -- The proof is omitted as per the requirement.
  sorry

end NUMINAMATH_GPT_tank_full_volume_l775_77553


namespace NUMINAMATH_GPT_plot_length_l775_77587

theorem plot_length (b : ℝ) (cost_per_meter cost_total : ℝ)
  (h1 : cost_per_meter = 26.5) 
  (h2 : cost_total = 5300) 
  (h3 : (2 * (b + (b + 20)) * cost_per_meter) = cost_total) : 
  b + 20 = 60 := 
by 
  -- Proof here
  sorry

end NUMINAMATH_GPT_plot_length_l775_77587


namespace NUMINAMATH_GPT_arithmetic_sequence_term_l775_77543

theorem arithmetic_sequence_term :
  (∀ (a_n : ℕ → ℚ) (S : ℕ → ℚ),
    (∀ n, a_n n = a_n 1 + (n - 1) * 1) → -- Arithmetic sequence with common difference of 1
    (∀ n, S n = n * a_n 1 + (n * (n - 1)) / 2) →  -- Sum of first n terms of sequence
    S 8 = 4 * S 4 →
    a_n 10 = 19 / 2) :=
by
  intros a_n S ha_n hSn hS8_eq
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_term_l775_77543


namespace NUMINAMATH_GPT_sum_of_reversed_base_digits_eq_zero_l775_77508

theorem sum_of_reversed_base_digits_eq_zero : ∃ n : ℕ, 
  (∀ a₁ a₀ : ℕ, n = 5 * a₁ + a₀ ∧ n = 12 * a₀ + a₁ ∧ 0 ≤ a₁ ∧ a₁ < 5 ∧ 0 ≤ a₀ ∧ a₀ < 12 
  ∧ n > 0 → n = 0)
:= sorry

end NUMINAMATH_GPT_sum_of_reversed_base_digits_eq_zero_l775_77508


namespace NUMINAMATH_GPT_propositions_A_and_D_true_l775_77531

theorem propositions_A_and_D_true :
  (∀ x : ℝ, x^2 - 4*x + 5 > 0) ∧ (∃ x : ℤ, 3*x^2 - 2*x - 1 = 0) :=
by
  sorry

end NUMINAMATH_GPT_propositions_A_and_D_true_l775_77531


namespace NUMINAMATH_GPT_elements_map_to_4_l775_77568

def f (x : ℝ) : ℝ := x^2

theorem elements_map_to_4 :
  { x : ℝ | f x = 4 } = {2, -2} :=
by
  sorry

end NUMINAMATH_GPT_elements_map_to_4_l775_77568


namespace NUMINAMATH_GPT_fruit_basket_combinations_l775_77581

theorem fruit_basket_combinations (apples oranges : ℕ) (ha : apples = 6) (ho : oranges = 12) : 
  (∃ (baskets : ℕ), 
    (∀ a, 1 ≤ a ∧ a ≤ apples → ∃ b, 2 ≤ b ∧ b ≤ oranges ∧ baskets = a * b) ∧ baskets = 66) :=
by {
  sorry
}

end NUMINAMATH_GPT_fruit_basket_combinations_l775_77581


namespace NUMINAMATH_GPT_hyperbola_min_sum_dist_l775_77580

open Real

theorem hyperbola_min_sum_dist (x y : ℝ) (F1 F2 A B : ℝ × ℝ) :
  -- Conditions for the hyperbola and the foci
  (∀ (x y : ℝ), x^2 / 9 - y^2 / 6 = 1) →
  F1 = (-c, 0) →
  F2 = (c, 0) →
  -- Minimum value of |AF2| + |BF2|
  ∃ (l : ℝ × ℝ → Prop), l F1 ∧ (∃ A B, l A ∧ l B ∧ A = (-3, y_A) ∧ B = (-3, y_B) ) →
  |dist A F2| + |dist B F2| = 16 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_min_sum_dist_l775_77580


namespace NUMINAMATH_GPT_worker_savings_multiple_l775_77588

variable (P : ℝ)

theorem worker_savings_multiple (h1 : P > 0) (h2 : 0.4 * P + 0.6 * P = P) : 
  (12 * 0.4 * P) / (0.6 * P) = 8 :=
by
  sorry

end NUMINAMATH_GPT_worker_savings_multiple_l775_77588


namespace NUMINAMATH_GPT_number_of_terms_in_sequence_l775_77540

theorem number_of_terms_in_sequence : ∃ n : ℕ, 6 + (n-1) * 4 = 154 ∧ n = 38 :=
by
  sorry

end NUMINAMATH_GPT_number_of_terms_in_sequence_l775_77540


namespace NUMINAMATH_GPT_problem_statement_l775_77590

noncomputable def lhs: ℝ := 8^6 * 27^6 * 8^27 * 27^8
noncomputable def rhs: ℝ := 216^14 * 8^19

theorem problem_statement : lhs = rhs :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l775_77590
