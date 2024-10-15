import Mathlib

namespace NUMINAMATH_GPT_arithmetic_seq_a7_constant_l821_82127

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

-- Definition of an arithmetic sequence
def is_arithmetic_seq (a : ℕ → α) : Prop :=
∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

-- Given arithmetic sequence {a_n}
variable (a : ℕ → α)
-- Given the property that a_2 + a_4 + a_{15} is a constant
variable (C : α)
variable (h : is_arithmetic_seq a)
variable (h_constant : a 2 + a 4 + a 15 = C)

-- Prove that a_7 is a constant
theorem arithmetic_seq_a7_constant (h : is_arithmetic_seq a) (h_constant : a 2 + a 4 + a 15 = C) : ∃ k : α, a 7 = k :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_seq_a7_constant_l821_82127


namespace NUMINAMATH_GPT_parabola_satisfies_given_condition_l821_82155

variable {p : ℝ}
variable {x1 x2 : ℝ}

-- Condition 1: The equation of the parabola is y^2 = 2px where p > 0.
def parabola_equation (p : ℝ) (x y : ℝ) : Prop :=
  y^2 = 2 * p * x

-- Condition 2: The parabola has a focus F.
-- Condition 3: A line passes through the focus F with an inclination angle of π/3.
def line_through_focus (p : ℝ) (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - p / 2)

-- Condition 4 & 5: The line intersects the parabola at points A and B with distance |AB| = 8.
def intersection_points (p : ℝ) (x1 x2 : ℝ) : Prop :=
  x1 ≠ x2 ∧ parabola_equation p x1 (Real.sqrt 3 * (x1 - p / 2)) ∧ parabola_equation p x2 (Real.sqrt 3 * (x2 - p / 2)) ∧
  abs (x1 - x2) * Real.sqrt (1 + 3) = 8

-- The proof statement
theorem parabola_satisfies_given_condition (hp : 0 < p) (hintersect : intersection_points p x1 x2) : 
  parabola_equation 3 x1 (Real.sqrt 3 * (x1 - 3 / 2)) ∧ parabola_equation 3 x2 (Real.sqrt 3 * (x2 - 3 / 2)) := sorry

end NUMINAMATH_GPT_parabola_satisfies_given_condition_l821_82155


namespace NUMINAMATH_GPT_avg_and_var_of_scaled_shifted_data_l821_82172

-- Definitions of average and variance
noncomputable def avg (l: List ℝ) : ℝ := (l.sum) / l.length
noncomputable def var (l: List ℝ) : ℝ := (l.map (λ x => (x - avg l) ^ 2)).sum / l.length

theorem avg_and_var_of_scaled_shifted_data
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_avg : avg (List.ofFn x) = 2)
  (h_var : var (List.ofFn x) = 3) :
  avg (List.ofFn (λ i => 2 * x i + 3)) = 7 ∧ var (List.ofFn (λ i => 2 * x i + 3)) = 12 := by
  sorry

end NUMINAMATH_GPT_avg_and_var_of_scaled_shifted_data_l821_82172


namespace NUMINAMATH_GPT_Jennifer_future_age_Jordana_future_age_Jordana_current_age_l821_82162

variable (Jennifer_age_now Jordana_age_now : ℕ)

-- Conditions
def age_in_ten_years (current_age : ℕ) : ℕ := current_age + 10
theorem Jennifer_future_age : age_in_ten_years Jennifer_age_now = 30 := sorry
theorem Jordana_future_age : age_in_ten_years Jordana_age_now = 3 * age_in_ten_years Jennifer_age_now := sorry

-- Question to prove
theorem Jordana_current_age : Jordana_age_now = 80 := sorry

end NUMINAMATH_GPT_Jennifer_future_age_Jordana_future_age_Jordana_current_age_l821_82162


namespace NUMINAMATH_GPT_online_textbooks_cost_l821_82179

theorem online_textbooks_cost (x : ℕ) :
  (5 * 10) + x + 3 * x = 210 → x = 40 :=
by
  sorry

end NUMINAMATH_GPT_online_textbooks_cost_l821_82179


namespace NUMINAMATH_GPT_compute_expression_l821_82175

theorem compute_expression : (-1) ^ 2014 + (π - 3.14) ^ 0 - (1 / 2) ^ (-2) = -2 := by
  sorry

end NUMINAMATH_GPT_compute_expression_l821_82175


namespace NUMINAMATH_GPT_fred_grew_38_cantaloupes_l821_82100

/-
  Fred grew some cantaloupes. Tim grew 44 cantaloupes.
  Together, they grew a total of 82 cantaloupes.
  Prove that Fred grew 38 cantaloupes.
-/

theorem fred_grew_38_cantaloupes (T F : ℕ) (h1 : T = 44) (h2 : T + F = 82) : F = 38 :=
by
  rw [h1] at h2
  linarith

end NUMINAMATH_GPT_fred_grew_38_cantaloupes_l821_82100


namespace NUMINAMATH_GPT_sum_of_palindromes_l821_82111

-- Define a three-digit palindrome predicate
def is_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ 0 ∧ b < 10 ∧ n = 100*a + 10*b + a

-- Define the product of the two palindromes equaling 436,995
theorem sum_of_palindromes (a b : ℕ) (h_a : is_palindrome a) (h_b : is_palindrome b) (h_prod : a * b = 436995) : 
  a + b = 1332 :=
sorry

end NUMINAMATH_GPT_sum_of_palindromes_l821_82111


namespace NUMINAMATH_GPT_num_whole_numbers_between_sqrt_50_and_sqrt_200_l821_82176

theorem num_whole_numbers_between_sqrt_50_and_sqrt_200 :
  let lower := Nat.ceil (Real.sqrt 50)
  let upper := Nat.floor (Real.sqrt 200)
  lower <= upper ∧ (upper - lower + 1) = 7 :=
by
  sorry

end NUMINAMATH_GPT_num_whole_numbers_between_sqrt_50_and_sqrt_200_l821_82176


namespace NUMINAMATH_GPT_greatest_integer_with_gcd_6_l821_82105

theorem greatest_integer_with_gcd_6 (x : ℕ) :
  x < 150 ∧ gcd x 12 = 6 → x = 138 :=
by
  sorry

end NUMINAMATH_GPT_greatest_integer_with_gcd_6_l821_82105


namespace NUMINAMATH_GPT_find_g_1_l821_82184

theorem find_g_1 (g : ℝ → ℝ) 
  (h : ∀ x : ℝ, g (2*x - 3) = 2*x^2 - x + 4) : 
  g 1 = 11.5 :=
sorry

end NUMINAMATH_GPT_find_g_1_l821_82184


namespace NUMINAMATH_GPT_avg_salary_officers_l821_82126

-- Definitions of the given conditions
def avg_salary_employees := 120
def avg_salary_non_officers := 110
def num_officers := 15
def num_non_officers := 495

-- The statement to be proven
theorem avg_salary_officers : (15 * (15 * X) / (15 + 495)) = 450 :=
by
  sorry

end NUMINAMATH_GPT_avg_salary_officers_l821_82126


namespace NUMINAMATH_GPT_mistaken_divisor_is_12_l821_82144

theorem mistaken_divisor_is_12 (dividend : ℕ) (mistaken_divisor : ℕ) (correct_divisor : ℕ) 
  (mistaken_quotient : ℕ) (correct_quotient : ℕ) (remainder : ℕ) :
  remainder = 0 ∧ correct_divisor = 21 ∧ mistaken_quotient = 42 ∧ correct_quotient = 24 ∧ 
  dividend = mistaken_quotient * mistaken_divisor ∧ dividend = correct_quotient * correct_divisor →
  mistaken_divisor = 12 :=
by 
  sorry

end NUMINAMATH_GPT_mistaken_divisor_is_12_l821_82144


namespace NUMINAMATH_GPT_sum_of_integers_eq_l821_82158

-- We define the conditions
variables (x y : ℕ)
-- The conditions specified in the problem
def diff_condition : Prop := x - y = 16
def prod_condition : Prop := x * y = 63

-- The theorem stating that given the conditions, the sum is 2*sqrt(127)
theorem sum_of_integers_eq : diff_condition x y → prod_condition x y → x + y = 2 * Real.sqrt 127 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_eq_l821_82158


namespace NUMINAMATH_GPT_complement_intersection_l821_82128

-- Definitions for the sets
def U : Set ℕ := {1, 2, 3, 4, 6}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Definition of the complement of a set with respect to a universal set
def complement (U A : Set ℕ) : Set ℕ := {x ∈ U | x ∉ A}

-- Theorem to prove
theorem complement_intersection :
  complement U (A ∩ B) = {1, 4, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l821_82128


namespace NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l821_82124

theorem tenth_term_arithmetic_sequence :
  ∀ (a₁ : ℚ) (d : ℚ), 
  (a₁ = 3/4) → (d = 1/2) →
  (a₁ + 9 * d) = 21/4 :=
by
  intro a₁ d ha₁ hd
  rw [ha₁, hd]
  sorry

end NUMINAMATH_GPT_tenth_term_arithmetic_sequence_l821_82124


namespace NUMINAMATH_GPT_a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l821_82167

theorem a_m_power_m_divides_a_n_power_n:
  ∀ (a : ℕ → ℕ) (m : ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) ∧ m > 1 → ∃ n > m, (a m) ^ m ∣ (a n) ^ n := by 
  sorry

theorem a1_does_not_divide_any_an_power_n:
  ∀ (a : ℕ → ℕ), (a 1).gcd (a 2) = 1 ∧ (∀ n, a (n + 2) = a (n + 1) * a n + 1) → ¬ ∃ n > 1, (a 1) ∣ (a n) ^ n := by
  sorry

end NUMINAMATH_GPT_a_m_power_m_divides_a_n_power_n_a1_does_not_divide_any_an_power_n_l821_82167


namespace NUMINAMATH_GPT_intersection_point_of_y_eq_4x_minus_2_with_x_axis_l821_82109

theorem intersection_point_of_y_eq_4x_minus_2_with_x_axis :
  ∃ x, (4 * x - 2 = 0 ∧ (x, 0) = (1 / 2, 0)) :=
by
  sorry

end NUMINAMATH_GPT_intersection_point_of_y_eq_4x_minus_2_with_x_axis_l821_82109


namespace NUMINAMATH_GPT_john_saves_money_l821_82136

theorem john_saves_money :
  let original_spending := 4 * 2
  let new_price_per_coffee := 2 + (2 * 0.5)
  let new_coffees := 4 / 2
  let new_spending := new_coffees * new_price_per_coffee
  original_spending - new_spending = 2 :=
by
  -- calculations omitted
  sorry

end NUMINAMATH_GPT_john_saves_money_l821_82136


namespace NUMINAMATH_GPT_motorcyclist_travel_time_l821_82196

-- Define the conditions and the proof goal:
theorem motorcyclist_travel_time :
  ∀ (z : ℝ) (t₁ t₂ t₃ : ℝ),
    t₂ = 60 →
    t₃ = 3240 →
    (t₃ - 5) / (z / 40 - z / t₁) = 10 →
    t₃ / (z / 40) = 10 + t₂ / (z / 60 - z / t₁) →
    t₁ = 80 :=
by
  intros z t₁ t₂ t₃ h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_motorcyclist_travel_time_l821_82196


namespace NUMINAMATH_GPT_subset_neg1_of_leq3_l821_82112

theorem subset_neg1_of_leq3 :
  {x | x = -1} ⊆ {x | x ≤ 3} :=
sorry

end NUMINAMATH_GPT_subset_neg1_of_leq3_l821_82112


namespace NUMINAMATH_GPT_geom_arith_seq_l821_82174

theorem geom_arith_seq (a : ℕ → ℝ) (q : ℝ) (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_arith : 2 * a 3 - (a 5 / 2) = (a 5 / 2) - 3 * a 1) (hq : q > 0) :
  (a 2 + a 5) / (a 9 + a 6) = 1 / 9 :=
by
  sorry

end NUMINAMATH_GPT_geom_arith_seq_l821_82174


namespace NUMINAMATH_GPT_sequence_bounded_l821_82145

theorem sequence_bounded (a : ℕ → ℕ) (a1 : ℕ) (h1 : a 0 = a1)
  (heven : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n) = a (2 * n - 1) - d)
  (hodd : ∀ n : ℕ, ∃ d : ℕ, 0 ≤ d ∧ d ≤ 9 ∧ a (2 * n + 1) = a (2 * n) + d) :
  ∀ n : ℕ, a n ≤ 10 * a1 := 
by
  sorry

end NUMINAMATH_GPT_sequence_bounded_l821_82145


namespace NUMINAMATH_GPT_subset_condition_for_A_B_l821_82107

open Set

theorem subset_condition_for_A_B {a : ℝ} (A B : Set ℝ) 
  (hA : A = {x | abs (x - 2) < a}) 
  (hB : B = {x | x^2 - 2 * x - 3 < 0}) :
  B ⊆ A ↔ 3 ≤ a :=
  sorry

end NUMINAMATH_GPT_subset_condition_for_A_B_l821_82107


namespace NUMINAMATH_GPT_boat_speed_in_still_water_l821_82181

-- Define the conditions
def speed_of_stream : ℝ := 3 -- (speed in km/h)
def time_downstream : ℝ := 1 -- (time in hours)
def time_upstream : ℝ := 1.5 -- (time in hours)

-- Define the goal by proving the speed of the boat in still water
theorem boat_speed_in_still_water : 
  ∃ V_b : ℝ, (V_b + speed_of_stream) * time_downstream = (V_b - speed_of_stream) * time_upstream ∧ V_b = 15 :=
by
  sorry -- (Proof will be provided here)

end NUMINAMATH_GPT_boat_speed_in_still_water_l821_82181


namespace NUMINAMATH_GPT_subtraction_correct_l821_82186

def x : ℝ := 5.75
def y : ℝ := 1.46
def result : ℝ := 4.29

theorem subtraction_correct : x - y = result := 
by
  sorry

end NUMINAMATH_GPT_subtraction_correct_l821_82186


namespace NUMINAMATH_GPT_part_one_part_two_l821_82146

noncomputable def a (n : ℕ) : ℚ := if n = 1 then 1 / 2 else 2 ^ (n - 1) / (1 + 2 ^ (n - 1))

noncomputable def b (n : ℕ) : ℚ := n / a n

noncomputable def S (n : ℕ) : ℚ :=
  (Finset.range n).sum (λ i => b (i + 1))

/-Theorem:
1. Prove that for all n > 0, a(n) = 2^(n-1) / (1 + 2^(n-1)).
2. Prove that for all n ≥ 3, S(n) > n^2 / 2 + 4.
-/
theorem part_one (n : ℕ) (h : n > 0) : a n = 2 ^ (n - 1) / (1 + 2 ^ (n - 1)) := sorry

theorem part_two (n : ℕ) (h : n ≥ 3) : S n > n ^ 2 / 2 + 4 := sorry

end NUMINAMATH_GPT_part_one_part_two_l821_82146


namespace NUMINAMATH_GPT_flower_shop_percentage_l821_82125

theorem flower_shop_percentage (C : ℕ) : 
  let V := (1/3 : ℝ) * C
  let T := (1/12 : ℝ) * C
  let R := T
  let total := C + V + T + R
  (C / total) * 100 = 66.67 := 
by
  sorry

end NUMINAMATH_GPT_flower_shop_percentage_l821_82125


namespace NUMINAMATH_GPT_quilt_shaded_fraction_l821_82151

theorem quilt_shaded_fraction :
  let total_squares := 16
  let shaded_full_square := 4
  let shaded_half_triangles_as_square := 2
  let total_area := total_squares
  let shaded_area := shaded_full_square + shaded_half_triangles_as_square
  shaded_area / total_area = 3 / 8 :=
by
  sorry

end NUMINAMATH_GPT_quilt_shaded_fraction_l821_82151


namespace NUMINAMATH_GPT_find_interest_rate_l821_82122

theorem find_interest_rate
  (P : ℝ)  -- Principal amount
  (A : ℝ)  -- Final amount
  (T : ℝ)  -- Time period in years
  (H1 : P = 1000)
  (H2 : A = 1120)
  (H3 : T = 2.4)
  : ∃ R : ℝ, (A - P) = (P * R * T) / 100 ∧ R = 5 :=
by
  -- Proof with calculations to be provided here
  sorry

end NUMINAMATH_GPT_find_interest_rate_l821_82122


namespace NUMINAMATH_GPT_circle_people_count_l821_82116

def num_people (n : ℕ) (a b : ℕ) : Prop :=
  a = 7 ∧ b = 18 ∧ (b = a + (n / 2))

theorem circle_people_count (n : ℕ) (a b : ℕ) (h : num_people n a b) : n = 24 :=
by
  sorry

end NUMINAMATH_GPT_circle_people_count_l821_82116


namespace NUMINAMATH_GPT_parabola_latus_rectum_l821_82160

theorem parabola_latus_rectum (p : ℝ) (H : ∀ y : ℝ, y^2 = 2 * p * -2) : p = 4 :=
sorry

end NUMINAMATH_GPT_parabola_latus_rectum_l821_82160


namespace NUMINAMATH_GPT_joe_paint_fraction_l821_82120

theorem joe_paint_fraction :
  let total_paint := 360
  let fraction_first_week := 1 / 9
  let used_first_week := (fraction_first_week * total_paint)
  let remaining_after_first_week := total_paint - used_first_week
  let total_used := 104
  let used_second_week := total_used - used_first_week
  let fraction_second_week := used_second_week / remaining_after_first_week
  fraction_second_week = 1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_joe_paint_fraction_l821_82120


namespace NUMINAMATH_GPT_quadratic_with_real_roots_l821_82129

theorem quadratic_with_real_roots: 
  ∀ k : ℝ, (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + 4 * x₁ + k = 0 ∧ x₂^2 + 4 * x₂ + k = 0) ↔ (k ≤ 4) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_with_real_roots_l821_82129


namespace NUMINAMATH_GPT_event_probability_l821_82187

noncomputable def probability_event : ℝ :=
  let a : ℝ := (1 : ℝ) / 2
  let b : ℝ := (3 : ℝ) / 2
  let interval_length : ℝ := 2
  (b - a) / interval_length

theorem event_probability :
  probability_event = (3 : ℝ) / 4 :=
by
  -- Proof step will be supplied here
  sorry

end NUMINAMATH_GPT_event_probability_l821_82187


namespace NUMINAMATH_GPT_trajectory_of_Q_l821_82103

-- Define Circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define Line l
def lineL (x y : ℝ) : Prop := x + y = 2

-- Define Conditions based on polar definitions
def polarCircle (ρ θ : ℝ) : Prop := ρ = 2

def polarLine (ρ θ : ℝ) : Prop := ρ * (Real.cos θ + Real.sin θ) = 2

-- Define points on ray OP
def pointP (ρ₁ θ : ℝ) : Prop := ρ₁ = 2 / (Real.cos θ + Real.sin θ)
def pointR (ρ₂ θ : ℝ) : Prop := ρ₂ = 2

-- Prove the trajectory of Q
theorem trajectory_of_Q (O P R Q : ℝ × ℝ)
  (ρ₁ θ ρ ρ₂ : ℝ)
  (h1: circleC O.1 O.2)
  (h2: lineL P.1 P.2)
  (h3: polarCircle ρ₂ θ)
  (h4: polarLine ρ₁ θ)
  (h5: ρ * ρ₁ = ρ₂^2) :
  ρ = 2 * (Real.cos θ + Real.sin θ) :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_Q_l821_82103


namespace NUMINAMATH_GPT_saree_stripes_l821_82152

theorem saree_stripes
  (G : ℕ) (B : ℕ) (Br : ℕ) (total_stripes : ℕ) (total_patterns : ℕ)
  (h1 : G = 3 * Br)
  (h2 : B = 5 * G)
  (h3 : Br = 4)
  (h4 : B + G + Br = 100)
  (h5 : total_stripes = 100)
  (h6 : total_patterns = total_stripes / 3) :
  B = 84 ∧ total_patterns = 33 := 
  by {
    sorry
  }

end NUMINAMATH_GPT_saree_stripes_l821_82152


namespace NUMINAMATH_GPT_find_phi_l821_82106

theorem find_phi 
  (f : ℝ → ℝ)
  (phi : ℝ)
  (y : ℝ → ℝ)
  (h1 : ∀ x, f x = Real.sin (2 * x + Real.pi / 6))
  (h2 : 0 < phi ∧ phi < Real.pi / 2)
  (h3 : ∀ x, y x = f (x - phi) ∧ y x = y (-x)) :
  phi = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_phi_l821_82106


namespace NUMINAMATH_GPT_expression_value_l821_82148

theorem expression_value : 2016 - 2017 + 2018 - 2019 + 2020 = 2018 := 
by 
  sorry

end NUMINAMATH_GPT_expression_value_l821_82148


namespace NUMINAMATH_GPT_arrange_polynomial_l821_82114

theorem arrange_polynomial :
  ∀ (x y : ℝ), 2 * x^3 * y - 4 * y^2 + 5 * x^2 = 5 * x^2 + 2 * x^3 * y - 4 * y^2 :=
by
  sorry

end NUMINAMATH_GPT_arrange_polynomial_l821_82114


namespace NUMINAMATH_GPT_fixed_monthly_charge_for_100_GB_l821_82137

theorem fixed_monthly_charge_for_100_GB
  (fixed_charge M : ℝ)
  (extra_charge_per_GB : ℝ := 0.25)
  (total_bill : ℝ := 65)
  (GB_over : ℝ := 80)
  (extra_charge : ℝ := GB_over * extra_charge_per_GB) :
  total_bill = M + extra_charge → M = 45 :=
by sorry

end NUMINAMATH_GPT_fixed_monthly_charge_for_100_GB_l821_82137


namespace NUMINAMATH_GPT_tins_of_beans_left_l821_82154

theorem tins_of_beans_left (cases : ℕ) (tins_per_case : ℕ) (damage_percentage : ℝ) (h_cases : cases = 15)
  (h_tins_per_case : tins_per_case = 24) (h_damage_percentage : damage_percentage = 0.05) :
  let total_tins := cases * tins_per_case
  let damaged_tins := total_tins * damage_percentage
  let tins_left := total_tins - damaged_tins
  tins_left = 342 :=
by
  sorry

end NUMINAMATH_GPT_tins_of_beans_left_l821_82154


namespace NUMINAMATH_GPT_dryer_cost_l821_82192

theorem dryer_cost (washer_dryer_total_cost washer_cost dryer_cost : ℝ) (h1 : washer_dryer_total_cost = 1200) (h2 : washer_cost = dryer_cost + 220) :
  dryer_cost = 490 :=
by
  sorry

end NUMINAMATH_GPT_dryer_cost_l821_82192


namespace NUMINAMATH_GPT_tea_garden_problem_pruned_to_wild_conversion_l821_82132

-- Definitions and conditions as per the problem statement
def total_area : ℕ := 16
def total_yield : ℕ := 660
def wild_yield_per_mu : ℕ := 30
def pruned_yield_per_mu : ℕ := 50

-- Lean 4 statement as per the proof problem
theorem tea_garden_problem :
  ∃ (x y : ℕ), (x + y = total_area) ∧ (wild_yield_per_mu * x + pruned_yield_per_mu * y = total_yield) ∧
  x = 7 ∧ y = 9 :=
sorry

-- Additional theorem for the conversion condition
theorem pruned_to_wild_conversion :
  ∀ (a : ℕ), (wild_yield_per_mu * (7 + a) ≥ pruned_yield_per_mu * (9 - a)) → a ≥ 3 :=
sorry

end NUMINAMATH_GPT_tea_garden_problem_pruned_to_wild_conversion_l821_82132


namespace NUMINAMATH_GPT_answer_one_answer_two_answer_three_l821_82110

def point_condition (A B : ℝ) (P : ℝ) (k : ℝ) : Prop := |A - P| = k * |B - P|

def question_one : Prop :=
  let A := -3
  let B := 6
  let k := 2
  let P := 3
  point_condition A B P k

def question_two : Prop :=
  ∀ x k : ℝ, |x + 2| + |x - 1| = 3 → point_condition (-3) 6 x k → (1 / 8 ≤ k ∧ k ≤ 4 / 5)

def question_three : Prop :=
  let A := -3
  let B := 6
  ∃ t : ℝ, t = 3 / 2 ∧ point_condition A (-3 + t) (6 - 2 * t) 3

theorem answer_one : question_one := by sorry

theorem answer_two : question_two := by sorry

theorem answer_three : question_three := by sorry

end NUMINAMATH_GPT_answer_one_answer_two_answer_three_l821_82110


namespace NUMINAMATH_GPT_pyramid_top_block_l821_82191

theorem pyramid_top_block (a b c d e : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
                         (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
                         (h : a * b ^ 4 * c ^ 6 * d ^ 4 * e = 140026320) : 
                         (a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 5) ∨ 
                         (a = 1 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 5) ∨ 
                         (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 7 ∧ e = 1) ∨ 
                         (a = 5 ∧ b = 7 ∧ c = 3 ∧ d = 2 ∧ e = 1) := 
sorry

end NUMINAMATH_GPT_pyramid_top_block_l821_82191


namespace NUMINAMATH_GPT_pies_sold_each_day_l821_82197

theorem pies_sold_each_day (total_pies : ℕ) (days_in_week : ℕ) (h1 : total_pies = 56) (h2 : days_in_week = 7) :
  (total_pies / days_in_week = 8) :=
by
exact sorry

end NUMINAMATH_GPT_pies_sold_each_day_l821_82197


namespace NUMINAMATH_GPT_percentage_profit_l821_82190

theorem percentage_profit 
  (C S : ℝ) 
  (h : 29 * C = 24 * S) : 
  ((S - C) / C) * 100 = 20.83 := 
by
  sorry

end NUMINAMATH_GPT_percentage_profit_l821_82190


namespace NUMINAMATH_GPT_determine_m_value_l821_82177

theorem determine_m_value
  (m : ℝ)
  (h : ∀ x : ℝ, -7 < x ∧ x < -1 ↔ mx^2 + 8 * m * x + 28 < 0) :
  m = 4 := by
  sorry

end NUMINAMATH_GPT_determine_m_value_l821_82177


namespace NUMINAMATH_GPT_tank_C_capacity_is_80_percent_of_tank_B_l821_82139

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := 
  Real.pi * r^2 * h

theorem tank_C_capacity_is_80_percent_of_tank_B :
  ∀ (h_C c_C h_B c_B : ℝ), 
    h_C = 10 ∧ c_C = 8 ∧ h_B = 8 ∧ c_B = 10 → 
    (volume_of_cylinder (c_C / (2 * Real.pi)) h_C) / 
    (volume_of_cylinder (c_B / (2 * Real.pi)) h_B) * 100 = 80 := 
by 
  intros h_C c_C h_B c_B h_conditions
  obtain ⟨h_C_10, c_C_8, h_B_8, c_B_10⟩ := h_conditions
  sorry

end NUMINAMATH_GPT_tank_C_capacity_is_80_percent_of_tank_B_l821_82139


namespace NUMINAMATH_GPT_count_total_wheels_l821_82133

theorem count_total_wheels (trucks : ℕ) (cars : ℕ) (truck_wheels : ℕ) (car_wheels : ℕ) :
  trucks = 12 → cars = 13 → truck_wheels = 4 → car_wheels = 4 →
  (trucks * truck_wheels + cars * car_wheels) = 100 :=
by
  intros h_trucks h_cars h_truck_wheels h_car_wheels
  sorry

end NUMINAMATH_GPT_count_total_wheels_l821_82133


namespace NUMINAMATH_GPT_brownies_pieces_count_l821_82123

theorem brownies_pieces_count:
  let pan_width := 24
  let pan_length := 15
  let piece_width := 3
  let piece_length := 2
  pan_width * pan_length / (piece_width * piece_length) = 60 := 
by
  sorry

end NUMINAMATH_GPT_brownies_pieces_count_l821_82123


namespace NUMINAMATH_GPT_no_solution_for_eq_l821_82141

theorem no_solution_for_eq (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -3) :
  (12 / (x^2 - 9) - 2 / (x - 3) = 1 / (x + 3)) → False :=
sorry

end NUMINAMATH_GPT_no_solution_for_eq_l821_82141


namespace NUMINAMATH_GPT_erin_trolls_count_l821_82171

def forest_trolls : ℕ := 6
def bridge_trolls : ℕ := 4 * forest_trolls - 6
def plains_trolls : ℕ := bridge_trolls / 2
def total_trolls : ℕ := forest_trolls + bridge_trolls + plains_trolls

theorem erin_trolls_count : total_trolls = 33 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_erin_trolls_count_l821_82171


namespace NUMINAMATH_GPT_largest_measureable_quantity_is_1_l821_82119

theorem largest_measureable_quantity_is_1 : 
  Nat.gcd (Nat.gcd (Nat.gcd (Nat.gcd 496 403) 713) 824) 1171 = 1 :=
  sorry

end NUMINAMATH_GPT_largest_measureable_quantity_is_1_l821_82119


namespace NUMINAMATH_GPT_solve_equation_l821_82134

theorem solve_equation (x : ℝ) : x*(x-3)^2*(5+x) = 0 ↔ x = 0 ∨ x = 3 ∨ x = -5 := 
by 
  sorry

end NUMINAMATH_GPT_solve_equation_l821_82134


namespace NUMINAMATH_GPT_sum_pqrs_eq_3150_l821_82185

theorem sum_pqrs_eq_3150
  (p q r s : ℝ)
  (h1 : p ≠ q) (h2 : p ≠ r) (h3 : p ≠ s) (h4 : q ≠ r) (h5 : q ≠ s) (h6 : r ≠ s)
  (hroots1 : ∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 → (x = r ∨ x = s))
  (hroots2 : ∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 → (x = p ∨ x = q)) :
  p + q + r + s = 3150 :=
by
  sorry

end NUMINAMATH_GPT_sum_pqrs_eq_3150_l821_82185


namespace NUMINAMATH_GPT_digit_making_527B_divisible_by_9_l821_82102

theorem digit_making_527B_divisible_by_9 (B : ℕ) : 14 + B ≡ 0 [MOD 9] → B = 4 :=
by
  intro h
  -- sorry is used in place of the actual proof.
  sorry

end NUMINAMATH_GPT_digit_making_527B_divisible_by_9_l821_82102


namespace NUMINAMATH_GPT_solve_quadratic_abs_l821_82189

theorem solve_quadratic_abs (x : ℝ) :
  x^2 - |x| - 1 = 0 ↔ x = (1 + Real.sqrt 5) / 2 ∨ x = (1 - Real.sqrt 5) / 2 ∨ 
                   x = (-1 + Real.sqrt 5) / 2 ∨ x = (-1 - Real.sqrt 5) / 2 := 
sorry

end NUMINAMATH_GPT_solve_quadratic_abs_l821_82189


namespace NUMINAMATH_GPT_abs_inequality_solution_l821_82115

theorem abs_inequality_solution (x : ℝ) : |x + 2| + |x - 1| ≥ 5 ↔ x ≤ -3 ∨ x ≥ 2 :=
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l821_82115


namespace NUMINAMATH_GPT_greatest_b_value_l821_82164

theorem greatest_b_value (b : ℤ) : 
  (∀ x : ℝ, x^2 + (b:ℝ) * x + 15 ≠ -9) ↔ b = 9 :=
sorry

end NUMINAMATH_GPT_greatest_b_value_l821_82164


namespace NUMINAMATH_GPT_volume_of_extended_parallelepiped_l821_82108

theorem volume_of_extended_parallelepiped :
  let main_box_volume := 3 * 3 * 6
  let external_boxes_volume := 2 * (3 * 3 * 1 + 3 * 6 * 1 + 3 * 6 * 1)
  let spheres_volume := 8 * (1 / 8) * (4 / 3) * Real.pi * (1 ^ 3)
  let cylinders_volume := 12 * (1 / 4) * Real.pi * 1^2 * 3 + 12 * (1 / 4) * Real.pi * 1^2 * 6
  main_box_volume + external_boxes_volume + spheres_volume + cylinders_volume = (432 + 52 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_extended_parallelepiped_l821_82108


namespace NUMINAMATH_GPT_arithmetic_sequence_check_l821_82142

theorem arithmetic_sequence_check 
  (a : ℕ → ℝ) 
  (d : ℝ)
  (h : ∀ n : ℕ, a (n+1) = a n + d) 
  : (∀ n : ℕ, (a n + 1) - (a (n - 1) + 1) = d) 
    ∧ (∀ n : ℕ, 2 * a (n + 1) - 2 * a n = 2 * d)
    ∧ (∀ n : ℕ, a (n + 1) - (a n + n) = d + 1) := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_check_l821_82142


namespace NUMINAMATH_GPT_prove_box_problem_l821_82149

noncomputable def boxProblem : Prop :=
  let height1 := 2
  let width1 := 4
  let length1 := 6
  let clay1 := 48
  let height2 := 3 * height1
  let width2 := 2 * width1
  let length2 := 1.5 * length1
  let volume1 := height1 * width1 * length1
  let volume2 := height2 * width2 * length2
  let n := (volume2 / volume1) * clay1
  n = 432

theorem prove_box_problem : boxProblem := by
  sorry

end NUMINAMATH_GPT_prove_box_problem_l821_82149


namespace NUMINAMATH_GPT_lauren_earnings_tuesday_l821_82165

def money_from_commercials (num_commercials : ℕ) (rate_per_commercial : ℝ) : ℝ :=
  num_commercials * rate_per_commercial

def money_from_subscriptions (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  num_subscriptions * rate_per_subscription

def total_money (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) : ℝ :=
  money_from_commercials num_commercials rate_per_commercial + money_from_subscriptions num_subscriptions rate_per_subscription

theorem lauren_earnings_tuesday (num_commercials : ℕ) (rate_per_commercial : ℝ) (num_subscriptions : ℕ) (rate_per_subscription : ℝ) :
  num_commercials = 100 → rate_per_commercial = 0.50 → num_subscriptions = 27 → rate_per_subscription = 1.00 → 
  total_money num_commercials rate_per_commercial num_subscriptions rate_per_subscription = 77 :=
by
  intros h1 h2 h3 h4
  simp [money_from_commercials, money_from_subscriptions, total_money, h1, h2, h3, h4]
  sorry

end NUMINAMATH_GPT_lauren_earnings_tuesday_l821_82165


namespace NUMINAMATH_GPT_acute_angle_ACD_l821_82143

theorem acute_angle_ACD (α : ℝ) (h : α ≤ 120) :
  ∃ (ACD : ℝ), ACD = Real.arcsin ((Real.tan (α / 2)) / Real.sqrt 3) :=
sorry

end NUMINAMATH_GPT_acute_angle_ACD_l821_82143


namespace NUMINAMATH_GPT_find_x3_y3_l821_82118

noncomputable def x_y_conditions (x y : ℝ) :=
  x - y = 3 ∧
  x^2 + y^2 = 27

theorem find_x3_y3 (x y : ℝ) (h : x_y_conditions x y) : x^3 - y^3 = 108 :=
  sorry

end NUMINAMATH_GPT_find_x3_y3_l821_82118


namespace NUMINAMATH_GPT_acute_triangle_l821_82163

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  ∃ area, 
    (area = (1 / 2) * a * b * Real.sin C) ∧
    (a / Real.sin A = 2 * c / Real.sqrt 3) ∧
    (c = Real.sqrt 7) ∧
    (area = (3 * Real.sqrt 3) / 2)

theorem acute_triangle (a b c A B C : ℝ) (h : triangle_ABC a b c A B C) :
  C = 60 ∧ a^2 + b^2 = 13 :=
by
  obtain ⟨_, h_area, h_sine, h_c, h_area_eq⟩ := h
  sorry

end NUMINAMATH_GPT_acute_triangle_l821_82163


namespace NUMINAMATH_GPT_tank_capacity_l821_82178

-- Definitions from conditions
def initial_fraction := (1 : ℚ) / 4  -- The tank is 1/4 full initially
def added_amount := 5  -- Adding 5 liters

-- The proof problem to show that the tank's total capacity c equals 60 liters
theorem tank_capacity
  (c : ℚ)  -- The total capacity of the tank in liters
  (h1 : c / 4 + added_amount = c / 3)  -- Adding 5 liters makes the tank 1/3 full
  : c = 60 := 
sorry

end NUMINAMATH_GPT_tank_capacity_l821_82178


namespace NUMINAMATH_GPT_shaded_region_perimeter_l821_82161

theorem shaded_region_perimeter (r : ℝ) (h : r = 12 / Real.pi) :
  3 * (24 / 6) = 12 := 
by
  sorry

end NUMINAMATH_GPT_shaded_region_perimeter_l821_82161


namespace NUMINAMATH_GPT_sum_of_first_60_digits_l821_82198

-- Define the repeating sequence and the number of repetitions
def repeating_sequence : List ℕ := [0, 0, 0, 1]
def repetitions : ℕ := 15

-- Define the sum of first n elements of a repeating sequence
def sum_repeating_sequence (seq : List ℕ) (n : ℕ) : ℕ :=
  let len := seq.length
  let complete_cycles := n / len
  let remaining_digits := n % len
  let sum_complete_cycles := complete_cycles * seq.sum
  let sum_remaining_digits := (seq.take remaining_digits).sum
  sum_complete_cycles + sum_remaining_digits

-- Prove the specific case for 60 digits
theorem sum_of_first_60_digits : sum_repeating_sequence repeating_sequence 60 = 15 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_first_60_digits_l821_82198


namespace NUMINAMATH_GPT_students_left_correct_l821_82130

-- Define the initial number of students
def initial_students : ℕ := 8

-- Define the number of new students
def new_students : ℕ := 8

-- Define the final number of students
def final_students : ℕ := 11

-- Define the number of students who left during the year
def students_who_left : ℕ :=
  (initial_students + new_students) - final_students

theorem students_left_correct : students_who_left = 5 :=
by
  -- Instantiating the definitions
  let initial := initial_students
  let new := new_students
  let final := final_students

  -- Calculation of students who left
  let L := (initial + new) - final

  -- Asserting the result
  show L = 5
  sorry

end NUMINAMATH_GPT_students_left_correct_l821_82130


namespace NUMINAMATH_GPT_max_alpha_beta_square_l821_82147

theorem max_alpha_beta_square (k : ℝ) (α β : ℝ)
  (h1 : α^2 - (k - 2) * α + (k^2 + 3 * k + 5) = 0)
  (h2 : β^2 - (k - 2) * β + (k^2 + 3 * k + 5) = 0)
  (h3 : α ≠ β) :
  (α^2 + β^2) ≤ 18 :=
sorry

end NUMINAMATH_GPT_max_alpha_beta_square_l821_82147


namespace NUMINAMATH_GPT_lucas_seq_units_digit_M47_l821_82168

def lucas_seq : ℕ → ℕ := 
  sorry -- skipped sequence generation for brevity

def M (n : ℕ) : ℕ :=
  if n = 0 then 3 else
  if n = 1 then 1 else
  lucas_seq n -- will call the lucas sequence generator

-- Helper function to get the units digit of a number
def units_digit (n: ℕ) : ℕ :=
  n % 10

theorem lucas_seq_units_digit_M47 : units_digit (M (M 6)) = 3 := 
sorry

end NUMINAMATH_GPT_lucas_seq_units_digit_M47_l821_82168


namespace NUMINAMATH_GPT_solution_set_of_inequality_l821_82113

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 3*x - 2 ≥ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l821_82113


namespace NUMINAMATH_GPT_polynomial_simplification_l821_82166

def A (x : ℝ) := 5 * x^2 + 4 * x - 1
def B (x : ℝ) := -x^2 - 3 * x + 3
def C (x : ℝ) := 8 - 7 * x - 6 * x^2

theorem polynomial_simplification (x : ℝ) : A x - B x + C x = 4 :=
by
  simp [A, B, C]
  sorry

end NUMINAMATH_GPT_polynomial_simplification_l821_82166


namespace NUMINAMATH_GPT_tan_pi_over_12_plus_tan_7pi_over_12_l821_82173

theorem tan_pi_over_12_plus_tan_7pi_over_12 : 
  (Real.tan (Real.pi / 12) + Real.tan (7 * Real.pi / 12)) = -4 * (3 - Real.sqrt 3) / 5 :=
by
  sorry

end NUMINAMATH_GPT_tan_pi_over_12_plus_tan_7pi_over_12_l821_82173


namespace NUMINAMATH_GPT_damaged_potatoes_l821_82104

theorem damaged_potatoes (initial_potatoes : ℕ) (weight_per_bag : ℕ) (price_per_bag : ℕ) (total_sales : ℕ) :
  initial_potatoes = 6500 →
  weight_per_bag = 50 →
  price_per_bag = 72 →
  total_sales = 9144 →
  ∃ damaged_potatoes : ℕ, damaged_potatoes = initial_potatoes - (total_sales / price_per_bag) * weight_per_bag ∧
                               damaged_potatoes = 150 :=
by
  intros _ _ _ _ 
  exact sorry

end NUMINAMATH_GPT_damaged_potatoes_l821_82104


namespace NUMINAMATH_GPT_verify_statements_l821_82131

def line1 (a x y : ℝ) : Prop := a * x - (a + 2) * y - 2 = 0
def line2 (a x y : ℝ) : Prop := (a - 2) * x + 3 * a * y + 2 = 0

theorem verify_statements (a : ℝ) :
  (∀ x y : ℝ, line1 a x y ↔ x = -1 ∧ y = -1) ∧
  (∀ x y : ℝ, (line1 a x y ∧ line2 a x y) → (a = 0 ∨ a = -4)) :=
by sorry

end NUMINAMATH_GPT_verify_statements_l821_82131


namespace NUMINAMATH_GPT_abs_quadratic_inequality_solution_l821_82150

theorem abs_quadratic_inequality_solution (x : ℝ) :
  |x^2 - 4 * x + 3| ≤ 3 ↔ 0 ≤ x ∧ x ≤ 4 :=
by sorry

end NUMINAMATH_GPT_abs_quadratic_inequality_solution_l821_82150


namespace NUMINAMATH_GPT_base_7_to_base_10_conversion_l821_82156

theorem base_7_to_base_10_conversion :
  (6 * 7^2 + 5 * 7^1 + 3 * 7^0) = 332 :=
by sorry

end NUMINAMATH_GPT_base_7_to_base_10_conversion_l821_82156


namespace NUMINAMATH_GPT_compound_interest_years_l821_82188

-- Define the parameters
def principal : ℝ := 7500
def future_value : ℝ := 8112
def annual_rate : ℝ := 0.04
def compounding_periods : ℕ := 1

-- Define the proof statement
theorem compound_interest_years :
  ∃ t : ℕ, future_value = principal * (1 + annual_rate / compounding_periods) ^ t ∧ t = 2 :=
by
  sorry

end NUMINAMATH_GPT_compound_interest_years_l821_82188


namespace NUMINAMATH_GPT_coordinates_of_A_l821_82121

theorem coordinates_of_A 
  (a : ℝ)
  (h1 : (a - 1) = 3 + (3 * a - 2)) :
  (a - 1, 3 * a - 2) = (-2, -5) :=
by
  sorry

end NUMINAMATH_GPT_coordinates_of_A_l821_82121


namespace NUMINAMATH_GPT_find_x_l821_82135

theorem find_x : ∃ (x : ℝ), x > 0 ∧ x + 17 = 60 * (1 / x) ∧ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_x_l821_82135


namespace NUMINAMATH_GPT_alberto_spent_more_l821_82183

-- Define the expenses of Alberto and Samara
def alberto_expenses : ℕ := 2457
def samara_oil_expense : ℕ := 25
def samara_tire_expense : ℕ := 467
def samara_detailing_expense : ℕ := 79
def samara_total_expenses : ℕ := samara_oil_expense + samara_tire_expense + samara_detailing_expense

-- State the theorem to prove the difference in expenses
theorem alberto_spent_more :
  alberto_expenses - samara_total_expenses = 1886 := by
  sorry

end NUMINAMATH_GPT_alberto_spent_more_l821_82183


namespace NUMINAMATH_GPT_proof_problem_l821_82180

theorem proof_problem (a b A B : ℝ) (f : ℝ → ℝ)
  (h_f : ∀ θ : ℝ, f θ ≥ 0)
  (h_f_def : ∀ θ : ℝ, f θ = 1 + a * Real.cos θ + b * Real.sin θ + A * Real.sin (2 * θ) + B * Real.cos (2 * θ)) :
  a^2 + b^2 ≤ 2 ∧ A^2 + B^2 ≤ 1 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l821_82180


namespace NUMINAMATH_GPT_function_is_monotonically_decreasing_l821_82159

noncomputable def f (x : ℝ) : ℝ := x^2 * (x - 3)

theorem function_is_monotonically_decreasing :
  ∀ x, 0 ≤ x ∧ x ≤ 2 → deriv f x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_function_is_monotonically_decreasing_l821_82159


namespace NUMINAMATH_GPT_minimum_value_of_expression_l821_82182

theorem minimum_value_of_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_eq : 1 / a + 1 / b = 1) :
  ∃ (x : ℝ), x = (1 / (a - 1) + 9 / (b - 1)) ∧ x = 6 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l821_82182


namespace NUMINAMATH_GPT_min_expr_value_l821_82157

theorem min_expr_value (a b c : ℝ) (h₀ : b > c) (h₁ : c > a) (h₂ : a > 0) (h₃ : b ≠ 0) :
  (∀ (a b c : ℝ), b > c → c > a → a > 0 → b ≠ 0 → 
   (2 + 6 * a^2 = (a+b)^3 / b^2 + (b-c)^2 / b^2 + (c-a)^3 / b^2) →
   2 <= (a + b)^3 / b^2 + (b - c)^2 / b^2 + (c - a)^3 / b^2) :=
by 
  sorry

end NUMINAMATH_GPT_min_expr_value_l821_82157


namespace NUMINAMATH_GPT_min_value_a_over_b_l821_82199

theorem min_value_a_over_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 2 * Real.sqrt a + b = 1) : ∃ c, c = 0 := 
by
  -- We need to show that the minimum value of a / b is 0 
  sorry

end NUMINAMATH_GPT_min_value_a_over_b_l821_82199


namespace NUMINAMATH_GPT_cost_of_each_ring_l821_82170

theorem cost_of_each_ring (R : ℝ) 
  (h1 : 4 * 12 + 8 * R = 80) : R = 4 :=
by 
  sorry

end NUMINAMATH_GPT_cost_of_each_ring_l821_82170


namespace NUMINAMATH_GPT_bottle_caps_per_person_l821_82195

noncomputable def initial_caps : Nat := 150
noncomputable def rebecca_caps : Nat := 42
noncomputable def alex_caps : Nat := 2 * rebecca_caps
noncomputable def total_caps : Nat := initial_caps + rebecca_caps + alex_caps
noncomputable def number_of_people : Nat := 6

theorem bottle_caps_per_person : total_caps / number_of_people = 46 := by
  sorry

end NUMINAMATH_GPT_bottle_caps_per_person_l821_82195


namespace NUMINAMATH_GPT_trigonometric_identity_proof_l821_82117

theorem trigonometric_identity_proof (θ : ℝ) 
  (h : Real.tan (θ + Real.pi / 4) = -3) : 
  2 * Real.sin θ ^ 2 - Real.cos θ ^ 2 = 7 / 5 :=
sorry

end NUMINAMATH_GPT_trigonometric_identity_proof_l821_82117


namespace NUMINAMATH_GPT_least_integer_solution_l821_82169

theorem least_integer_solution (x : ℤ) (h : x^2 = 2 * x + 98) : x = -7 :=
by {
  sorry
}

end NUMINAMATH_GPT_least_integer_solution_l821_82169


namespace NUMINAMATH_GPT_count_true_propositions_l821_82193

theorem count_true_propositions :
  let prop1 := false  -- Proposition ① is false
  let prop2 := true   -- Proposition ② is true
  let prop3 := true   -- Proposition ③ is true
  let prop4 := false  -- Proposition ④ is false
  (if prop1 then 1 else 0) + (if prop2 then 1 else 0) +
  (if prop3 then 1 else 0) + (if prop4 then 1 else 0) = 2 :=
by
  -- The theorem is expected to be proven here
  sorry

end NUMINAMATH_GPT_count_true_propositions_l821_82193


namespace NUMINAMATH_GPT_problem_statement_l821_82140

-- Definitions related to the given conditions
noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - (5 * Real.pi) / 6)

theorem problem_statement :
  (∀ x1 x2 : ℝ, (x1 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → (x2 ∈ Set.Ioo (Real.pi / 6) (2 * Real.pi / 3)) → x1 < x2 → f x1 < f x2) →
  (f (Real.pi / 6) = f (2 * Real.pi / 3)) →
  f (-((5 * Real.pi) / 12)) = (Real.sqrt 3) / 2 :=
by
  intros h_mono h_symm
  sorry

end NUMINAMATH_GPT_problem_statement_l821_82140


namespace NUMINAMATH_GPT_find_a_purely_imaginary_l821_82153

noncomputable def purely_imaginary_condition (a : ℝ) : Prop :=
    (2 * a - 1) / (a^2 + 1) = 0 ∧ (a + 2) / (a^2 + 1) ≠ 0

theorem find_a_purely_imaginary :
    ∀ (a : ℝ), purely_imaginary_condition a ↔ a = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_find_a_purely_imaginary_l821_82153


namespace NUMINAMATH_GPT_bowling_ball_weight_l821_82138

theorem bowling_ball_weight (b k : ℝ) (h1 : 5 * b = 3 * k) (h2 : 4 * k = 120) : b = 18 :=
by
  sorry

end NUMINAMATH_GPT_bowling_ball_weight_l821_82138


namespace NUMINAMATH_GPT_book_loss_percentage_l821_82101

theorem book_loss_percentage 
  (C S : ℝ) 
  (h : 15 * C = 20 * S) : 
  (C - S) / C * 100 = 25 := 
by 
  sorry

end NUMINAMATH_GPT_book_loss_percentage_l821_82101


namespace NUMINAMATH_GPT_singers_in_fifth_verse_l821_82194

theorem singers_in_fifth_verse (choir : ℕ) (absent : ℕ) (participating : ℕ) 
(half_first_verse : ℕ) (third_second_verse : ℕ) (quarter_third_verse : ℕ) 
(fifth_fourth_verse : ℕ) (late_singers : ℕ) :
  choir = 70 → 
  absent = 10 → 
  participating = choir - absent →
  half_first_verse = participating / 2 → 
  third_second_verse = (participating - half_first_verse) / 3 →
  quarter_third_verse = (participating - half_first_verse - third_second_verse) / 4 →
  fifth_fourth_verse = (participating - half_first_verse - third_second_verse - quarter_third_verse) / 5 →
  late_singers = 5 →
  participating = 60 :=
by sorry

end NUMINAMATH_GPT_singers_in_fifth_verse_l821_82194
