import Mathlib

namespace NUMINAMATH_GPT_Correct_Statement_l1152_115280

theorem Correct_Statement : 
  (∀ x : ℝ, 7 * x = 4 * x - 3 → 7 * x - 4 * x = -3) ∧
  (∀ x : ℝ, (2 * x - 1) / 3 = 1 + (x - 3) / 2 → 2 * (2 * x - 1) = 6 + 3 * (x - 3)) ∧
  (∀ x : ℝ, 2 * (2 * x - 1) - 3 * (x - 3) = 1 → 4 * x - 2 - 3 * x + 9 = 1) ∧
  (∀ x : ℝ, 2 * (x + 1) = x + 7 → x = 5) :=
by
  sorry

end NUMINAMATH_GPT_Correct_Statement_l1152_115280


namespace NUMINAMATH_GPT_vertex_of_parabola_l1152_115293

theorem vertex_of_parabola (x : ℝ) : 
  ∀ x y : ℝ, (y = x^2 - 6 * x + 1) → (∃ h k : ℝ, y = (x - h)^2 + k ∧ h = 3 ∧ k = -8) :=
by
  -- This is to state that given the parabola equation x^2 - 6x + 1, its vertex coordinates are (3, -8).
  sorry

end NUMINAMATH_GPT_vertex_of_parabola_l1152_115293


namespace NUMINAMATH_GPT_slope_range_of_tangent_line_l1152_115268

theorem slope_range_of_tangent_line (x : ℝ) (h : x ≠ 0) : (1 - 1/(x^2)) < 1 :=
by
  calc 
    1 - 1/(x^2) < 1 := sorry

end NUMINAMATH_GPT_slope_range_of_tangent_line_l1152_115268


namespace NUMINAMATH_GPT_max_value_k_l1152_115272

noncomputable def max_k (S : Finset ℕ) (A : ℕ → Finset ℕ) (k : ℕ) :=
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) ∧
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2)

theorem max_value_k : ∀ (S : Finset ℕ) (A : ℕ → Finset ℕ), 
  S = Finset.range 14 \{0} → 
  (∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ k → (A i ∩ A j).card ≤ 2) →
  (∀ i, 1 ≤ i ∧ i ≤ k → (A i).card = 6) →
  ∃ k, max_k S A k ∧ k = 4 :=
sorry

end NUMINAMATH_GPT_max_value_k_l1152_115272


namespace NUMINAMATH_GPT_total_cost_proof_l1152_115296

-- Define the prices of items
def price_coffee : ℕ := 4
def price_cake : ℕ := 7
def price_ice_cream : ℕ := 3

-- Define the number of items ordered by Mell and her friends
def mell_coffee : ℕ := 2
def mell_cake : ℕ := 1
def friend_coffee : ℕ := 2
def friend_cake : ℕ := 1
def friend_ice_cream : ℕ := 1
def number_of_friends : ℕ := 2

-- Calculate total cost for Mell
def total_mell : ℕ := (mell_coffee * price_coffee) + (mell_cake * price_cake)

-- Calculate total cost per friend
def total_friend : ℕ := (friend_coffee * price_coffee) + (friend_cake * price_cake) + (friend_ice_cream * price_ice_cream)

-- Calculate total cost for all friends
def total_friends : ℕ := number_of_friends * total_friend

-- Calculate total cost for Mell and her friends
def total_cost : ℕ := total_mell + total_friends

-- The theorem to prove
theorem total_cost_proof : total_cost = 51 := by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l1152_115296


namespace NUMINAMATH_GPT_correct_model_is_pakistan_traditional_l1152_115275

-- Given definitions
def hasPrimitiveModel (country : String) : Prop := country = "Nigeria"
def hasTraditionalModel (country : String) : Prop := country = "India" ∨ country = "Pakistan" ∨ country = "Nigeria"
def hasModernModel (country : String) : Prop := country = "China"

-- The proposition to prove
theorem correct_model_is_pakistan_traditional :
  (hasPrimitiveModel "Nigeria")
  ∧ (hasModernModel "China")
  ∧ (hasTraditionalModel "India")
  ∧ (hasTraditionalModel "Pakistan") →
  (hasTraditionalModel "Pakistan") := by
  intros h
  exact (h.right.right.right)

end NUMINAMATH_GPT_correct_model_is_pakistan_traditional_l1152_115275


namespace NUMINAMATH_GPT_ratio_of_intercepts_l1152_115239

variable (b1 b2 : ℝ)
variable (s t : ℝ)
variable (Hs : s = -b1 / 8)
variable (Ht : t = -b2 / 3)

theorem ratio_of_intercepts (hb1 : b1 ≠ 0) (hb2 : b2 ≠ 0) : s / t = 3 * b1 / (8 * b2) :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_intercepts_l1152_115239


namespace NUMINAMATH_GPT_remaining_sand_fraction_l1152_115292

theorem remaining_sand_fraction (total_weight : ℕ) (used_weight : ℕ) (h1 : total_weight = 50) (h2 : used_weight = 30) : 
  (total_weight - used_weight) / total_weight = 2 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_remaining_sand_fraction_l1152_115292


namespace NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1152_115253

theorem sum_of_consecutive_page_numbers (n : ℕ) (h : n * (n + 1) = 20412) : n + (n + 1) = 283 := 
sorry

end NUMINAMATH_GPT_sum_of_consecutive_page_numbers_l1152_115253


namespace NUMINAMATH_GPT_polygon_area_l1152_115283

theorem polygon_area (n : ℕ) (s : ℝ) (perimeter : ℝ) (area : ℝ) 
  (h1 : n = 24) 
  (h2 : n * s = perimeter) 
  (h3 : perimeter = 48) 
  (h4 : s = 2) 
  (h5 : area = n * s^2 / 2) : 
  area = 96 :=
by
  sorry

end NUMINAMATH_GPT_polygon_area_l1152_115283


namespace NUMINAMATH_GPT_largest_sum_pairs_l1152_115205

theorem largest_sum_pairs (a b c d : ℝ) (h₀ : a ≠ b) (h₁ : a ≠ c) (h₂ : a ≠ d) (h₃ : b ≠ c) (h₄ : b ≠ d) (h₅ : c ≠ d) (h₆ : a < b) (h₇ : b < c) (h₈ : c < d)
(h₉ : a + b = 9 ∨ a + b = 10) (h₁₀ : b + c = 9 ∨ b + c = 10)
(h₁₁ : b + d = 12) (h₁₂ : c + d = 13) :
d = 8 ∨ d = 7.5 :=
sorry

end NUMINAMATH_GPT_largest_sum_pairs_l1152_115205


namespace NUMINAMATH_GPT_a_value_for_even_function_l1152_115287

def f (x a : ℝ) := (x + 1) * (x + a)

theorem a_value_for_even_function (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = -1 :=
by
  sorry

end NUMINAMATH_GPT_a_value_for_even_function_l1152_115287


namespace NUMINAMATH_GPT_game_cost_proof_l1152_115243

variable (initial : ℕ) (allowance : ℕ) (final : ℕ) (cost : ℕ)

-- Initial amount
def initial_money : ℕ := 11
-- Allowance received
def allowance_money : ℕ := 14
-- Final amount of money
def final_money : ℕ := 22
-- Cost of the new game is to be proved
def game_cost : ℕ :=  initial_money - (final_money - allowance_money)

theorem game_cost_proof : game_cost = 3 := by
  sorry

end NUMINAMATH_GPT_game_cost_proof_l1152_115243


namespace NUMINAMATH_GPT_alcohol_percentage_in_mixed_solution_l1152_115218

theorem alcohol_percentage_in_mixed_solution :
  let vol1 := 8
  let perc1 := 0.25
  let vol2 := 2
  let perc2 := 0.12
  let total_alcohol := (vol1 * perc1) + (vol2 * perc2)
  let total_volume := vol1 + vol2
  (total_alcohol / total_volume) * 100 = 22.4 := by
  sorry

end NUMINAMATH_GPT_alcohol_percentage_in_mixed_solution_l1152_115218


namespace NUMINAMATH_GPT_greatest_value_inequality_l1152_115282

theorem greatest_value_inequality (x : ℝ) :
  x^2 - 6 * x + 8 ≤ 0 → x ≤ 4 := 
sorry

end NUMINAMATH_GPT_greatest_value_inequality_l1152_115282


namespace NUMINAMATH_GPT_combined_height_of_cylinders_l1152_115271

/-- Given three cylinders with perimeters 6 feet, 9 feet, and 11 feet respectively,
    and rolled out on a rectangular plate with a diagonal of 19 feet,
    the combined height of the cylinders is 26 feet. -/
theorem combined_height_of_cylinders
  (p1 p2 p3 : ℝ) (d : ℝ)
  (h_p1 : p1 = 6) (h_p2 : p2 = 9) (h_p3 : p3 = 11) (h_d : d = 19) :
  p1 + p2 + p3 = 26 :=
sorry

end NUMINAMATH_GPT_combined_height_of_cylinders_l1152_115271


namespace NUMINAMATH_GPT_cosQ_is_0_point_4_QP_is_12_prove_QR_30_l1152_115276

noncomputable def find_QR (Q : Real) (QP : Real) : Real :=
  let cosQ := 0.4
  let QR := QP / cosQ
  QR

theorem cosQ_is_0_point_4_QP_is_12_prove_QR_30 :
  find_QR 0.4 12 = 30 :=
by
  sorry

end NUMINAMATH_GPT_cosQ_is_0_point_4_QP_is_12_prove_QR_30_l1152_115276


namespace NUMINAMATH_GPT_awards_distribution_count_l1152_115294

-- Define the problem conditions
def num_awards : Nat := 5
def num_students : Nat := 3

-- Verify each student gets at least one award
def each_student_gets_at_least_one (distributions : List (List Nat)) : Prop :=
  ∀ (dist : List Nat), dist ∈ distributions → (∀ (d : Nat), d > 0)

-- Define the main theorem to be proved
theorem awards_distribution_count :
  ∃ (distributions : List (List Nat)), each_student_gets_at_least_one distributions ∧ distributions.length = 150 :=
sorry

end NUMINAMATH_GPT_awards_distribution_count_l1152_115294


namespace NUMINAMATH_GPT_find_larger_number_l1152_115256

-- Definitions based on the conditions
variables (x y : ℕ)

-- Main theorem
theorem find_larger_number (h1 : x + y = 50) (h2 : x - y = 10) : x = 30 :=
by
  sorry

end NUMINAMATH_GPT_find_larger_number_l1152_115256


namespace NUMINAMATH_GPT_probability_other_side_green_l1152_115213

-- Definitions based on the conditions
def Card : Type := ℕ
def num_cards : ℕ := 8
def blue_blue : ℕ := 4
def blue_green : ℕ := 2
def green_green : ℕ := 2

def total_green_sides : ℕ := (green_green * 2) + blue_green
def green_opposite_green_side : ℕ := green_green * 2

theorem probability_other_side_green (h_total_green_sides : total_green_sides = 6)
(h_green_opposite_green_side : green_opposite_green_side = 4) :
  (green_opposite_green_side / total_green_sides : ℚ) = 2 / 3 := 
by
  sorry

end NUMINAMATH_GPT_probability_other_side_green_l1152_115213


namespace NUMINAMATH_GPT_nathan_paintable_area_l1152_115269

def total_paintable_area (rooms : ℕ) (length width height : ℕ) (non_paintable_area : ℕ) : ℕ :=
  let wall_area := 2 * (length * height + width * height)
  rooms * (wall_area - non_paintable_area)

theorem nathan_paintable_area :
  total_paintable_area 4 15 12 9 75 = 1644 :=
by sorry

end NUMINAMATH_GPT_nathan_paintable_area_l1152_115269


namespace NUMINAMATH_GPT_range_of_a_l1152_115229

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x → 2 * x * log x ≥ -x^2 + a * x - 3) → a ≤ 4 :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l1152_115229


namespace NUMINAMATH_GPT_max_m_value_l1152_115261

noncomputable def f (x m : ℝ) : ℝ := x * Real.log x + x^2 - m * x + Real.exp (2 - x)

theorem max_m_value (m : ℝ) :
  (∀ x : ℝ, 0 < x → f x m ≥ 0) → m ≤ 3 :=
sorry

end NUMINAMATH_GPT_max_m_value_l1152_115261


namespace NUMINAMATH_GPT_product_of_perimeters_correct_l1152_115236

noncomputable def area (side_length : ℝ) : ℝ := side_length * side_length

theorem product_of_perimeters_correct (x y : ℝ)
  (h1 : area x + area y = 85)
  (h2 : area x - area y = 45) :
  4 * x * 4 * y = 32 * Real.sqrt 325 :=
by sorry

end NUMINAMATH_GPT_product_of_perimeters_correct_l1152_115236


namespace NUMINAMATH_GPT_area_inequalities_l1152_115291

noncomputable def f1 (x : ℝ) : ℝ := 1 - (1 / 2) * x
noncomputable def f2 (x : ℝ) : ℝ := 1 / (x + 1)
noncomputable def f3 (x : ℝ) : ℝ := 1 - (1 / 2) * x^2

noncomputable def S1 : ℝ := 1 - (1 / 4)
noncomputable def S2 : ℝ := Real.log 2
noncomputable def S3 : ℝ := (5 / 6)

theorem area_inequalities : S2 < S1 ∧ S1 < S3 := by
  sorry

end NUMINAMATH_GPT_area_inequalities_l1152_115291


namespace NUMINAMATH_GPT_carts_needed_each_day_last_two_days_l1152_115223

-- Define capacities as per conditions
def daily_capacity_large_truck : ℚ := 1 / (3 * 4)
def daily_capacity_small_truck : ℚ := 1 / (4 * 5)
def daily_capacity_cart : ℚ := 1 / (20 * 6)

-- Define the number of carts required each day in the last two days
def required_carts_last_two_days : ℚ :=
  let total_work_done_by_large_trucks := 2 * daily_capacity_large_truck * 2
  let total_work_done_by_small_trucks := 3 * daily_capacity_small_truck * 2
  let total_work_done_by_carts := 7 * daily_capacity_cart * 2
  let total_work_done := total_work_done_by_large_trucks + total_work_done_by_small_trucks + total_work_done_by_carts
  let remaining_work := 1 - total_work_done
  remaining_work / (2 * daily_capacity_cart)

-- Assertion of the number of carts required
theorem carts_needed_each_day_last_two_days :
  required_carts_last_two_days = 15 := by
  sorry

end NUMINAMATH_GPT_carts_needed_each_day_last_two_days_l1152_115223


namespace NUMINAMATH_GPT_find_a_l1152_115234

open Real

def are_perpendicular (l1 l2 : Real × Real × Real) : Prop :=
  let (a1, b1, c1) := l1
  let (a2, b2, c2) := l2
  a1 * a2 + b1 * b2 = 0

theorem find_a (a : Real) :
  let l1 := (a + 2, 1 - a, -1)
  let l2 := (a - 1, 2 * a + 3, 2)
  are_perpendicular l1 l2 → a = 1 ∨ a = -1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l1152_115234


namespace NUMINAMATH_GPT_nat_divisor_problem_l1152_115219

open Nat

theorem nat_divisor_problem (n : ℕ) (d : ℕ → ℕ) (k : ℕ)
    (h1 : 1 = d 1)
    (h2 : ∀ i, 1 < i → i ≤ k → d i < d (i + 1))
    (hk : d k = n)
    (hdiv : ∀ i, 1 ≤ i ∧ i ≤ k → d i ∣ n)
    (heq : n = d 2 * d 3 + d 2 * d 5 + d 3 * d 5) :
    k = 8 ∨ k = 9 :=
sorry

end NUMINAMATH_GPT_nat_divisor_problem_l1152_115219


namespace NUMINAMATH_GPT_conical_surface_radius_l1152_115286

theorem conical_surface_radius (r : ℝ) :
  (2 * Real.pi * r = 5 * Real.pi) → r = 2.5 :=
by
  sorry

end NUMINAMATH_GPT_conical_surface_radius_l1152_115286


namespace NUMINAMATH_GPT_α_plus_2β_eq_pi_div_2_l1152_115237

open Real

noncomputable def α : ℝ := sorry
noncomputable def β : ℝ := sorry

axiom h1 : 0 < α ∧ α < π / 2
axiom h2 : 0 < β ∧ β < π / 2
axiom h3 : 3 * sin α ^ 2 + 2 * sin β ^ 2 = 1
axiom h4 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0

theorem α_plus_2β_eq_pi_div_2 : α + 2 * β = π / 2 :=
by
  sorry

end NUMINAMATH_GPT_α_plus_2β_eq_pi_div_2_l1152_115237


namespace NUMINAMATH_GPT_machine_bottle_caps_l1152_115230

variable (A_rate : ℕ)
variable (A_time : ℕ)
variable (B_rate : ℕ)
variable (B_time : ℕ)
variable (C_rate : ℕ)
variable (C_time : ℕ)
variable (D_rate : ℕ)
variable (D_time : ℕ)
variable (E_rate : ℕ)
variable (E_time : ℕ)

def A_bottles := A_rate * A_time
def B_bottles := B_rate * B_time
def C_bottles := C_rate * C_time
def D_bottles := D_rate * D_time
def E_bottles := E_rate * E_time

theorem machine_bottle_caps (hA_rate : A_rate = 24)
                            (hA_time : A_time = 10)
                            (hB_rate : B_rate = A_rate - 3)
                            (hB_time : B_time = 12)
                            (hC_rate : C_rate = B_rate + 6)
                            (hC_time : C_time = 15)
                            (hD_rate : D_rate = C_rate - 4)
                            (hD_time : D_time = 8)
                            (hE_rate : E_rate = D_rate + 5)
                            (hE_time : E_time = 5) :
  A_bottles A_rate A_time = 240 ∧ 
  B_bottles B_rate B_time = 252 ∧ 
  C_bottles C_rate C_time = 405 ∧ 
  D_bottles D_rate D_time = 184 ∧ 
  E_bottles E_rate E_time = 140 := by
    sorry

end NUMINAMATH_GPT_machine_bottle_caps_l1152_115230


namespace NUMINAMATH_GPT_smallest_possible_perimeter_l1152_115226

-- Definitions for prime numbers and scalene triangles
def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_scalene_triangle (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions
def valid_sides (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧ a ≥ 5 ∧ b ≥ 5 ∧ c ≥ 5 ∧ is_scalene_triangle a b c

def valid_perimeter (a b c : ℕ) : Prop :=
  is_prime (a + b + c)

-- The goal statement
theorem smallest_possible_perimeter : ∃ a b c : ℕ, valid_sides a b c ∧ valid_perimeter a b c ∧ (a + b + c) = 23 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_perimeter_l1152_115226


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1152_115266

/-- Let {a_n} be an arithmetic sequence and S_n the sum of its first n terms.
   Given a_1 - a_5 - a_10 - a_15 + a_19 = 2, prove that S_19 = -38. --/
theorem arithmetic_sequence_sum (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h2 : a 1 - a 5 - a 10 - a 15 + a 19 = 2) :
  S 19 = -38 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1152_115266


namespace NUMINAMATH_GPT_index_cards_per_student_l1152_115228

theorem index_cards_per_student
    (periods_per_day : ℕ)
    (students_per_class : ℕ)
    (cost_per_pack : ℕ)
    (total_spent : ℕ)
    (cards_per_pack : ℕ)
    (total_packs : ℕ)
    (total_index_cards : ℕ)
    (total_students : ℕ)
    (index_cards_per_student : ℕ)
    (h1 : periods_per_day = 6)
    (h2 : students_per_class = 30)
    (h3 : cost_per_pack = 3)
    (h4 : total_spent = 108)
    (h5 : cards_per_pack = 50)
    (h6 : total_packs = total_spent / cost_per_pack)
    (h7 : total_index_cards = total_packs * cards_per_pack)
    (h8 : total_students = periods_per_day * students_per_class)
    (h9 : index_cards_per_student = total_index_cards / total_students) :
    index_cards_per_student = 10 := 
  by
    sorry

end NUMINAMATH_GPT_index_cards_per_student_l1152_115228


namespace NUMINAMATH_GPT_min_ab_l1152_115211

variable (a b : ℝ)

theorem min_ab (h1 : a > 1) (h2 : b > 2) (h3 : a * b = 2 * a + b) : a + b ≥ 3 + 2 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_min_ab_l1152_115211


namespace NUMINAMATH_GPT_megan_bottles_left_l1152_115255

-- Defining the initial conditions
def initial_bottles : Nat := 17
def bottles_drank : Nat := 3

-- Theorem stating that Megan has 14 bottles left
theorem megan_bottles_left : initial_bottles - bottles_drank = 14 := by
  sorry

end NUMINAMATH_GPT_megan_bottles_left_l1152_115255


namespace NUMINAMATH_GPT_evaluate_f_2010_times_l1152_115289

noncomputable def f (x : ℝ) : ℝ := 1 / (1 - x^2011)^(1/2011)

theorem evaluate_f_2010_times (x : ℝ) (h : x = 2011) :
  (f^[2010] x)^2011 = 2011^2011 :=
by
  rw [h]
  sorry

end NUMINAMATH_GPT_evaluate_f_2010_times_l1152_115289


namespace NUMINAMATH_GPT_beth_should_charge_42_cents_each_l1152_115238

theorem beth_should_charge_42_cents_each (n_alan_cookies : ℕ) (price_alan_cookie : ℕ) (n_beth_cookies : ℕ) (total_earnings : ℕ) (price_beth_cookie : ℕ):
  n_alan_cookies = 15 → 
  price_alan_cookie = 50 → 
  n_beth_cookies = 18 → 
  total_earnings = n_alan_cookies * price_alan_cookie → 
  price_beth_cookie = total_earnings / n_beth_cookies → 
  price_beth_cookie = 42 := 
by 
  intros h1 h2 h3 h4 h5 
  sorry

end NUMINAMATH_GPT_beth_should_charge_42_cents_each_l1152_115238


namespace NUMINAMATH_GPT_total_fuel_usage_is_250_l1152_115265

-- Define John's fuel consumption per km
def fuel_consumption_per_km : ℕ := 5

-- Define the distance of the first trip
def distance_trip1 : ℕ := 30

-- Define the distance of the second trip
def distance_trip2 : ℕ := 20

-- Define the fuel usage calculation
def fuel_usage_trip1 := distance_trip1 * fuel_consumption_per_km
def fuel_usage_trip2 := distance_trip2 * fuel_consumption_per_km
def total_fuel_usage := fuel_usage_trip1 + fuel_usage_trip2

-- Prove that the total fuel usage is 250 liters
theorem total_fuel_usage_is_250 : total_fuel_usage = 250 := by
  sorry

end NUMINAMATH_GPT_total_fuel_usage_is_250_l1152_115265


namespace NUMINAMATH_GPT_age_of_replaced_man_l1152_115258

-- Definitions based on conditions
def avg_age_men (A : ℝ) := A
def age_man1 := 10
def avg_age_women := 23
def total_age_women := 2 * avg_age_women
def new_avg_age_men (A : ℝ) := A + 2

-- Proposition stating that given conditions yield the age of the other replaced man
theorem age_of_replaced_man (A M : ℝ) :
  8 * avg_age_men A - age_man1 - M + total_age_women = 8 * new_avg_age_men A + 16 →
  M = 20 :=
by
  sorry

end NUMINAMATH_GPT_age_of_replaced_man_l1152_115258


namespace NUMINAMATH_GPT_johns_total_spending_l1152_115201

theorem johns_total_spending:
  ∀ (X : ℝ), (3/7 * X + 2/5 * X + 1/4 * X + 1/14 * X + 12 = X) → X = 80 :=
by
  intro X h
  sorry

end NUMINAMATH_GPT_johns_total_spending_l1152_115201


namespace NUMINAMATH_GPT_t_minus_d_l1152_115200

-- Define amounts paid by Tom, Dorothy, and Sammy
def tom_paid : ℕ := 140
def dorothy_paid : ℕ := 90
def sammy_paid : ℕ := 220

-- Define the total amount and required equal share
def total_paid : ℕ := tom_paid + dorothy_paid + sammy_paid
def equal_share : ℕ := total_paid / 3

-- Define the amounts t and d where Tom and Dorothy balance the costs by paying Sammy
def t : ℤ := equal_share - tom_paid -- Amount Tom gave to Sammy
def d : ℤ := equal_share - dorothy_paid -- Amount Dorothy gave to Sammy

-- Prove that t - d = -50
theorem t_minus_d : t - d = -50 := by
  sorry

end NUMINAMATH_GPT_t_minus_d_l1152_115200


namespace NUMINAMATH_GPT_bank1_more_advantageous_l1152_115262

-- Define the quarterly interest rate for Bank 1
def bank1_quarterly_rate : ℝ := 0.8

-- Define the annual interest rate for Bank 2
def bank2_annual_rate : ℝ := 9.0

-- Define the annual compounded interest rate for Bank 1
def bank1_annual_yield : ℝ :=
  (1 + bank1_quarterly_rate) ^ 4

-- Define the annual rate directly for Bank 2
def bank2_annual_yield : ℝ :=
  1 + bank2_annual_rate

-- The theorem stating that Bank 1 is more advantageous than Bank 2
theorem bank1_more_advantageous : bank1_annual_yield > bank2_annual_yield :=
  sorry

end NUMINAMATH_GPT_bank1_more_advantageous_l1152_115262


namespace NUMINAMATH_GPT_no_injective_function_l1152_115299

theorem no_injective_function (f : ℕ → ℕ) (h : ∀ m n : ℕ, f (m * n) = f m + f n) : ¬ Function.Injective f := 
sorry

end NUMINAMATH_GPT_no_injective_function_l1152_115299


namespace NUMINAMATH_GPT_different_algorithms_for_same_problem_l1152_115277

-- Define the basic concept of a problem
def Problem := Type

-- Define what it means for something to be an algorithm solving a problem
def Algorithm (P : Problem) := P -> Prop

-- Define the statement to be true: Different algorithms can solve the same problem
theorem different_algorithms_for_same_problem (P : Problem) (A1 A2 : Algorithm P) :
  P = P -> A1 ≠ A2 -> true :=
by
  sorry

end NUMINAMATH_GPT_different_algorithms_for_same_problem_l1152_115277


namespace NUMINAMATH_GPT_chess_group_players_l1152_115259

theorem chess_group_players (n : ℕ) (h : n * (n - 1) / 2 = 190) : n = 20 :=
sorry

end NUMINAMATH_GPT_chess_group_players_l1152_115259


namespace NUMINAMATH_GPT_probability_at_least_one_humanities_l1152_115244

theorem probability_at_least_one_humanities :
  let morning_classes := ["mathematics", "Chinese", "politics", "geography"]
  let afternoon_classes := ["English", "history", "physical_education"]
  let humanities := ["politics", "history", "geography"]
  let total_choices := List.length morning_classes * List.length afternoon_classes
  let favorable_morning := List.length (List.filter (fun x => x ∈ humanities) morning_classes)
  let favorable_afternoon := List.length (List.filter (fun x => x ∈ humanities) afternoon_classes)
  let favorable_choices := favorable_morning * List.length afternoon_classes + favorable_afternoon * (List.length morning_classes - favorable_morning)
  (favorable_choices / total_choices) = (2 / 3) := by sorry

end NUMINAMATH_GPT_probability_at_least_one_humanities_l1152_115244


namespace NUMINAMATH_GPT_darts_game_score_l1152_115288

variable (S1 S2 S3 : ℕ)
variable (n : ℕ)

theorem darts_game_score :
  n = 8 →
  S2 = 2 * S1 →
  S3 = (3 * S1) →
  S2 = 48 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_darts_game_score_l1152_115288


namespace NUMINAMATH_GPT_cube_volume_is_64_l1152_115214

theorem cube_volume_is_64 (a : ℕ) (h : (a - 2) * (a + 3) * a = a^3 + 12) : a^3 = 64 := 
  sorry

end NUMINAMATH_GPT_cube_volume_is_64_l1152_115214


namespace NUMINAMATH_GPT_Dima_claim_false_l1152_115216

theorem Dima_claim_false (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : (a*x^2 + b*x + c = 0) → ∃ α β, α < 0 ∧ β < 0 ∧ (α + β = -b/a) ∧ (α*β = c/a)) :
  ¬ ∃ α' β', α' > 0 ∧ β' > 0 ∧ (α' + β' = -c/b) ∧ (α'*β' = a/b) :=
sorry

end NUMINAMATH_GPT_Dima_claim_false_l1152_115216


namespace NUMINAMATH_GPT_geometric_sequence_a1_range_l1152_115245

theorem geometric_sequence_a1_range (a : ℕ → ℝ) (b : ℕ → ℝ) (a1 : ℝ) :
  (∀ n, a (n+1) = a n / 2) ∧ (∀ n, b n = n / 2) ∧ (∃! n : ℕ, a n > b n) →
  (6 < a1 ∧ a1 ≤ 16) :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_a1_range_l1152_115245


namespace NUMINAMATH_GPT_heat_capacity_at_100K_l1152_115251

noncomputable def heat_capacity (t : ℝ) : ℝ :=
  0.1054 + 0.000004 * t

theorem heat_capacity_at_100K :
  heat_capacity 100 = 0.1058 := 
by
  sorry

end NUMINAMATH_GPT_heat_capacity_at_100K_l1152_115251


namespace NUMINAMATH_GPT_find_constants_C_D_l1152_115298

theorem find_constants_C_D
  (C : ℚ) (D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32 / 9 ∧ D = 13 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_constants_C_D_l1152_115298


namespace NUMINAMATH_GPT_speed_in_still_water_l1152_115295

-- Define the velocities (speeds)
def speed_downstream (V_w V_s : ℝ) : ℝ := V_w + V_s
def speed_upstream (V_w V_s : ℝ) : ℝ := V_w - V_s

-- Define the given conditions
def downstream_condition (V_w V_s : ℝ) : Prop := speed_downstream V_w V_s = 9
def upstream_condition (V_w V_s : ℝ) : Prop := speed_upstream V_w V_s = 1

-- The main theorem to prove
theorem speed_in_still_water (V_s V_w : ℝ) (h1 : downstream_condition V_w V_s) (h2 : upstream_condition V_w V_s) : V_w = 5 :=
  sorry

end NUMINAMATH_GPT_speed_in_still_water_l1152_115295


namespace NUMINAMATH_GPT_max_cut_length_l1152_115260

theorem max_cut_length (board_size : ℕ) (total_pieces : ℕ) 
  (area_each : ℕ) 
  (total_area : ℕ)
  (total_perimeter : ℕ)
  (initial_perimeter : ℕ)
  (max_possible_length : ℕ)
  (h1 : board_size = 30) 
  (h2 : total_pieces = 225)
  (h3 : area_each = 4)
  (h4 : total_area = board_size * board_size)
  (h5 : total_perimeter = total_pieces * 10)
  (h6 : initial_perimeter = 4 * board_size)
  (h7 : max_possible_length = (total_perimeter - initial_perimeter) / 2) :
  max_possible_length = 1065 :=
by 
  -- Here, we do not include the proof as per the instructions
  sorry

end NUMINAMATH_GPT_max_cut_length_l1152_115260


namespace NUMINAMATH_GPT_max_area_of_triangle_on_parabola_l1152_115267

noncomputable def area_of_triangle_ABC (p : ℝ) : ℝ :=
  (1 / 2) * abs (3 * p^2 - 14 * p + 15)

theorem max_area_of_triangle_on_parabola :
  ∃ p : ℝ, 1 ≤ p ∧ p ≤ 3 ∧ area_of_triangle_ABC p = 2 := sorry

end NUMINAMATH_GPT_max_area_of_triangle_on_parabola_l1152_115267


namespace NUMINAMATH_GPT_alan_total_cost_is_84_l1152_115240

def num_dark_cds : ℕ := 2
def num_avn_cds : ℕ := 1
def num_90s_cds : ℕ := 5
def price_avn_cd : ℕ := 12 -- in dollars
def price_dark_cd : ℕ := price_avn_cd * 2
def total_cost_other_cds : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd
def price_90s_cds : ℕ := ((40 : ℕ) * total_cost_other_cds) / 100
def total_cost_all_products : ℕ := num_dark_cds * price_dark_cd + num_avn_cds * price_avn_cd + price_90s_cds

theorem alan_total_cost_is_84 : total_cost_all_products = 84 := by
  sorry

end NUMINAMATH_GPT_alan_total_cost_is_84_l1152_115240


namespace NUMINAMATH_GPT_mean_age_of_euler_family_children_l1152_115206

noncomputable def euler_family_children_ages : List ℕ := [9, 9, 9, 9, 18, 21, 21]

theorem mean_age_of_euler_family_children : 
  (List.sum euler_family_children_ages : ℚ) / (List.length euler_family_children_ages) = 96 / 7 := 
by
  sorry

end NUMINAMATH_GPT_mean_age_of_euler_family_children_l1152_115206


namespace NUMINAMATH_GPT_dog_age_64_human_years_l1152_115241

def dog_years (human_years : ℕ) : ℕ :=
if human_years = 0 then
  0
else if human_years = 1 then
  1
else if human_years = 2 then
  2
else
  2 + (human_years - 2) / 5

theorem dog_age_64_human_years : dog_years 64 = 10 :=
by 
    sorry

end NUMINAMATH_GPT_dog_age_64_human_years_l1152_115241


namespace NUMINAMATH_GPT_smaller_number_of_product_l1152_115273

theorem smaller_number_of_product :
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 5610 ∧ a = 34 :=
by
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_smaller_number_of_product_l1152_115273


namespace NUMINAMATH_GPT_min_abs_ab_perpendicular_lines_l1152_115284

theorem min_abs_ab_perpendicular_lines (a b : ℝ) (h : a * b = a ^ 2 + 1) : |a * b| = 1 :=
by sorry

end NUMINAMATH_GPT_min_abs_ab_perpendicular_lines_l1152_115284


namespace NUMINAMATH_GPT_solution_to_equation_l1152_115285

theorem solution_to_equation (x : ℝ) (h : (5 - x / 2)^(1/3) = 2) : x = -6 :=
sorry

end NUMINAMATH_GPT_solution_to_equation_l1152_115285


namespace NUMINAMATH_GPT_number_of_ants_in_section_correct_l1152_115208

noncomputable def ants_in_section := 
  let width_feet : ℝ := 600
  let length_feet : ℝ := 800
  let ants_per_square_inch : ℝ := 5
  let side_feet : ℝ := 200
  let feet_to_inches : ℝ := 12
  let side_inches := side_feet * feet_to_inches
  let area_section_square_inches := side_inches^2
  ants_per_square_inch * area_section_square_inches

theorem number_of_ants_in_section_correct :
  ants_in_section = 28800000 := 
by 
  unfold ants_in_section 
  sorry

end NUMINAMATH_GPT_number_of_ants_in_section_correct_l1152_115208


namespace NUMINAMATH_GPT_total_erasers_l1152_115233

def cases : ℕ := 7
def boxes_per_case : ℕ := 12
def erasers_per_box : ℕ := 25

theorem total_erasers : cases * boxes_per_case * erasers_per_box = 2100 := by
  sorry

end NUMINAMATH_GPT_total_erasers_l1152_115233


namespace NUMINAMATH_GPT_solutions_periodic_with_same_period_l1152_115246

variable {y z : ℝ → ℝ}
variable (f g : ℝ → ℝ)

-- defining the conditions
variable (h1 : ∀ x, deriv y x = - (z x)^3)
variable (h2 : ∀ x, deriv z x = (y x)^3)
variable (h3 : y 0 = 1)
variable (h4 : z 0 = 0)
variable (h5 : ∀ x, y x = f x)
variable (h6 : ∀ x, z x = g x)

-- proving periodicity
theorem solutions_periodic_with_same_period : ∃ k > 0, (∀ x, f (x + k) = f x ∧ g (x + k) = g x) := by
  sorry

end NUMINAMATH_GPT_solutions_periodic_with_same_period_l1152_115246


namespace NUMINAMATH_GPT_blake_change_l1152_115249

theorem blake_change :
  let lollipop_count := 4
  let chocolate_count := 6
  let lollipop_cost := 2
  let chocolate_cost := 4 * lollipop_cost
  let total_received := 6 * 10
  let total_cost := (lollipop_count * lollipop_cost) + (chocolate_count * chocolate_cost)
  let change := total_received - total_cost
  change = 4 :=
by
  sorry

end NUMINAMATH_GPT_blake_change_l1152_115249


namespace NUMINAMATH_GPT_curve_is_parabola_l1152_115215

theorem curve_is_parabola (t : ℝ) : 
  ∃ (x y : ℝ), (x = 3^t - 2) ∧ (y = 9^t - 4 * 3^t + 2 * t - 4) ∧ (∃ a b c : ℝ, y = a * x^2 + b * x + c) :=
by sorry

end NUMINAMATH_GPT_curve_is_parabola_l1152_115215


namespace NUMINAMATH_GPT_ratio_expression_l1152_115210

theorem ratio_expression 
  (m n r t : ℚ)
  (h1 : m / n = 5 / 2)
  (h2 : r / t = 7 / 15) : 
  (3 * m * r - n * t) / (4 * n * t - 7 * m * r) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_expression_l1152_115210


namespace NUMINAMATH_GPT_calculate_expression_l1152_115222

theorem calculate_expression : 
  (1007^2 - 995^2 - 1005^2 + 997^2) = 8008 := 
by {
  sorry
}

end NUMINAMATH_GPT_calculate_expression_l1152_115222


namespace NUMINAMATH_GPT_percent_of_part_l1152_115220

variable (Part : ℕ) (Whole : ℕ)

theorem percent_of_part (hPart : Part = 70) (hWhole : Whole = 280) :
  (Part / Whole) * 100 = 25 := by
  sorry

end NUMINAMATH_GPT_percent_of_part_l1152_115220


namespace NUMINAMATH_GPT_part_one_part_two_l1152_115224

noncomputable def problem_conditions (θ : ℝ) : Prop :=
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  ∃ m : ℝ, (∀ x : ℝ, x^2 - (Real.sqrt 3 - 1) * x + m = 0 → (x = sin_theta ∨ x = cos_theta))

theorem part_one (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let m := sin_theta * cos_theta
  m = (3 - 2 * Real.sqrt 3) / 2 :=
sorry

theorem part_two (θ: ℝ) (h: problem_conditions θ) : 
  let sin_theta := Real.sin θ
  let cos_theta := Real.cos θ
  let tan_theta := sin_theta / cos_theta
  (cos_theta - sin_theta * tan_theta) / (1 - tan_theta) = Real.sqrt 3 - 1 :=
sorry

end NUMINAMATH_GPT_part_one_part_two_l1152_115224


namespace NUMINAMATH_GPT_find_children_tickets_l1152_115248

variable (A C S : ℝ)

theorem find_children_tickets 
  (h1 : A + C + S = 600)
  (h2 : 6 * A + 4.5 * C + 5 * S = 3250) :
  C = (350 - S) / 1.5 := 
sorry

end NUMINAMATH_GPT_find_children_tickets_l1152_115248


namespace NUMINAMATH_GPT_product_xyz_one_l1152_115254

theorem product_xyz_one (x y z : ℝ) (h1 : x + 1/y = 2) (h2 : y + 1/z = 2) (h3 : z + 1/x = 2) : x * y * z = 1 := 
by {
    sorry
}

end NUMINAMATH_GPT_product_xyz_one_l1152_115254


namespace NUMINAMATH_GPT_ratio_of_selling_prices_l1152_115204

theorem ratio_of_selling_prices (C SP1 SP2 : ℝ)
  (h1 : SP1 = C + 0.20 * C)
  (h2 : SP2 = C + 1.40 * C) :
  SP2 / SP1 = 2 := by
  sorry

end NUMINAMATH_GPT_ratio_of_selling_prices_l1152_115204


namespace NUMINAMATH_GPT_pond_fish_approximation_l1152_115203

noncomputable def total_number_of_fish
  (tagged_first: ℕ) (total_caught_second: ℕ) (tagged_second: ℕ) : ℕ :=
  (tagged_first * total_caught_second) / tagged_second

theorem pond_fish_approximation :
  total_number_of_fish 60 50 2 = 1500 :=
by
  -- calculation of the total number of fish based on given conditions
  sorry

end NUMINAMATH_GPT_pond_fish_approximation_l1152_115203


namespace NUMINAMATH_GPT_volume_difference_is_867_25_l1152_115221

noncomputable def charlie_volume : ℝ :=
  let h_C := 9
  let circumference_C := 7
  let r_C := circumference_C / (2 * Real.pi)
  let v_C := Real.pi * r_C^2 * h_C
  v_C

noncomputable def dana_volume : ℝ :=
  let h_D := 5
  let circumference_D := 10
  let r_D := circumference_D / (2 * Real.pi)
  let v_D := Real.pi * r_D^2 * h_D
  v_D

noncomputable def volume_difference : ℝ :=
  Real.pi * (abs (charlie_volume - dana_volume))

theorem volume_difference_is_867_25 : volume_difference = 867.25 := by
  sorry

end NUMINAMATH_GPT_volume_difference_is_867_25_l1152_115221


namespace NUMINAMATH_GPT_angle_of_inclination_l1152_115264

theorem angle_of_inclination (θ : ℝ) : 
  (∀ x y : ℝ, x - y + 3 = 0 → ∃ θ : ℝ, Real.tan θ = 1 ∧ θ = Real.pi / 4) := by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l1152_115264


namespace NUMINAMATH_GPT_algebraic_expression_value_l1152_115242

theorem algebraic_expression_value (a : ℝ) (h : a^2 + a - 1 = 0) : a^2 + a + 1 = 2 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1152_115242


namespace NUMINAMATH_GPT_unique_solution_pair_l1152_115250

open Real

theorem unique_solution_pair :
  ∃! (x y : ℝ), y = (x-1)^2 ∧ x * y - y = -3 :=
sorry

end NUMINAMATH_GPT_unique_solution_pair_l1152_115250


namespace NUMINAMATH_GPT_start_of_range_l1152_115225

variable (x : ℕ)

theorem start_of_range (h : ∃ (n : ℕ), n ≤ 79 ∧ n % 11 = 0 ∧ x = 79 - 3 * 11) 
(h4 : ∀ (k : ℕ), 0 ≤ k ∧ k < 4 → ∃ (y : ℕ), y = 79 - (k * 11) ∧ y % 11 = 0) :
  x = 44 := by
  sorry

end NUMINAMATH_GPT_start_of_range_l1152_115225


namespace NUMINAMATH_GPT_solution_set_of_x_x_plus_2_lt_3_l1152_115274

theorem solution_set_of_x_x_plus_2_lt_3 :
  {x : ℝ | x*(x + 2) < 3} = {x : ℝ | -3 < x ∧ x < 1} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_x_x_plus_2_lt_3_l1152_115274


namespace NUMINAMATH_GPT_distance_light_travels_250_years_l1152_115207

def distance_light_travels_one_year : ℝ := 5.87 * 10^12
def years : ℝ := 250

theorem distance_light_travels_250_years :
  distance_light_travels_one_year * years = 1.4675 * 10^15 :=
by
  sorry

end NUMINAMATH_GPT_distance_light_travels_250_years_l1152_115207


namespace NUMINAMATH_GPT_product_polynomial_coeffs_l1152_115263

theorem product_polynomial_coeffs
  (g h : ℚ)
  (h1 : 7 * d^2 - 3 * d + g * (3 * d^2 + h * d - 5) = 21 * d^4 - 44 * d^3 - 35 * d^2 + 14 * d + 15) :
  g + h = -28/9 := 
  sorry

end NUMINAMATH_GPT_product_polynomial_coeffs_l1152_115263


namespace NUMINAMATH_GPT_bus_speed_with_stoppages_l1152_115235

theorem bus_speed_with_stoppages :
  ∀ (speed_excluding_stoppages : ℕ) (stop_minutes : ℕ) (total_minutes : ℕ)
  (speed_including_stoppages : ℕ),
  speed_excluding_stoppages = 80 →
  stop_minutes = 15 →
  total_minutes = 60 →
  speed_including_stoppages = (speed_excluding_stoppages * (total_minutes - stop_minutes) / total_minutes) →
  speed_including_stoppages = 60 := by
  sorry

end NUMINAMATH_GPT_bus_speed_with_stoppages_l1152_115235


namespace NUMINAMATH_GPT_setC_is_not_pythagorean_triple_l1152_115232

-- Define what it means to be a Pythagorean triple
def isPythagoreanTriple (a b c : ℤ) : Prop :=
  a^2 + b^2 = c^2

-- Define the sets of numbers
def setA := (3, 4, 5)
def setB := (5, 12, 13)
def setC := (7, 25, 26)
def setD := (6, 8, 10)

-- The theorem stating that setC is not a Pythagorean triple
theorem setC_is_not_pythagorean_triple : ¬isPythagoreanTriple 7 25 26 := 
by sorry

end NUMINAMATH_GPT_setC_is_not_pythagorean_triple_l1152_115232


namespace NUMINAMATH_GPT_find_a_for_max_y_l1152_115231

theorem find_a_for_max_y (a : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 2 * (x - 1)^2 - 3 ≤ 15) →
  (∃ x : ℝ, 1 ≤ x ∧ x ≤ a ∧ 2 * (x - 1)^2 - 3 = 15) →
  a = 4 :=
by sorry

end NUMINAMATH_GPT_find_a_for_max_y_l1152_115231


namespace NUMINAMATH_GPT_shane_chewed_pieces_l1152_115281

theorem shane_chewed_pieces :
  ∀ (Elyse Rick Shane: ℕ),
  Elyse = 100 →
  Rick = Elyse / 2 →
  Shane = Rick / 2 →
  Shane_left = 14 →
  (Shane - Shane_left) = 11 :=
by
  intros Elyse Rick Shane Elyse_def Rick_def Shane_def Shane_left_def
  sorry

end NUMINAMATH_GPT_shane_chewed_pieces_l1152_115281


namespace NUMINAMATH_GPT_athlete_with_most_stable_performance_l1152_115297

def variance_A : ℝ := 0.78
def variance_B : ℝ := 0.2
def variance_C : ℝ := 1.28

theorem athlete_with_most_stable_performance : variance_B < variance_A ∧ variance_B < variance_C :=
by {
  -- Variance comparisons:
  -- 0.2 < 0.78
  -- 0.2 < 1.28
  sorry
}

end NUMINAMATH_GPT_athlete_with_most_stable_performance_l1152_115297


namespace NUMINAMATH_GPT_num_ways_to_designated_face_l1152_115290

-- Define the structure of the dodecahedron
inductive Face
| Top
| Bottom
| TopRing (n : ℕ)   -- n ranges from 1 to 5
| BottomRing (n : ℕ)  -- n ranges from 1 to 5
deriving Repr, DecidableEq

-- Define adjacency relations on Faces (simplified)
def adjacent : Face → Face → Prop
| Face.Top, Face.TopRing n          => true
| Face.TopRing n, Face.TopRing m    => (m = (n % 5) + 1) ∨ (m = ((n + 3) % 5) + 1)
| Face.TopRing n, Face.BottomRing m => true
| Face.BottomRing n, Face.BottomRing m => true
| _, _ => false

-- Predicate for specific face on the bottom ring
def designated_bottom_face (f : Face) : Prop :=
  match f with
  | Face.BottomRing 1 => true
  | _ => false

-- Define the number of ways to move from top to the designated bottom face
noncomputable def num_ways : ℕ :=
  5 + 10

-- Lean statement that represents our equivalent proof problem
theorem num_ways_to_designated_face :
  num_ways = 15 := by
  sorry

end NUMINAMATH_GPT_num_ways_to_designated_face_l1152_115290


namespace NUMINAMATH_GPT_min_x_y_l1152_115279

noncomputable def min_value (x y : ℝ) : ℝ := x + y

theorem min_x_y (x y : ℝ) (h₁ : x > 0) (h₂ : y > 0) (h₃ : x + 16 * y = x * y) :
  min_value x y = 25 :=
sorry

end NUMINAMATH_GPT_min_x_y_l1152_115279


namespace NUMINAMATH_GPT_simplify_fraction_l1152_115217

theorem simplify_fraction (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : y - 1/x ≠ 0) :
  (x - 1/y) / (y - 1/x) = x / y :=
sorry

end NUMINAMATH_GPT_simplify_fraction_l1152_115217


namespace NUMINAMATH_GPT_kid_ticket_price_l1152_115257

theorem kid_ticket_price (adult_price kid_tickets tickets total_profit : ℕ) 
  (h_adult_price : adult_price = 6) 
  (h_kid_tickets : kid_tickets = 75) 
  (h_tickets : tickets = 175) 
  (h_total_profit : total_profit = 750) : 
  (total_profit - (tickets - kid_tickets) * adult_price) / kid_tickets = 2 :=
by
  sorry

end NUMINAMATH_GPT_kid_ticket_price_l1152_115257


namespace NUMINAMATH_GPT_remainder_of_base12_integer_divided_by_9_l1152_115252

-- Define the base-12 integer
def base12_integer := 2 * 12^3 + 7 * 12^2 + 4 * 12 + 3

-- Define the condition for our problem
def divisor := 9

-- State the theorem to be proved
theorem remainder_of_base12_integer_divided_by_9 :
  base12_integer % divisor = 0 :=
sorry

end NUMINAMATH_GPT_remainder_of_base12_integer_divided_by_9_l1152_115252


namespace NUMINAMATH_GPT_marys_age_l1152_115227

variable (M R : ℕ) -- Define M (Mary's current age) and R (Rahul's current age) as natural numbers

theorem marys_age
  (h1 : R = M + 40)       -- Rahul is 40 years older than Mary
  (h2 : R + 30 = 3 * (M + 30))  -- In 30 years, Rahul will be three times as old as Mary
  : M = 20 := 
sorry  -- The proof goes here

end NUMINAMATH_GPT_marys_age_l1152_115227


namespace NUMINAMATH_GPT_find_a_if_f_is_odd_l1152_115202

noncomputable def f (a x : ℝ) : ℝ := (a * 2^x + a - 2) / (2^x + 1)

theorem find_a_if_f_is_odd :
  (∀ x : ℝ, f 1 x = -f 1 (-x)) ↔ (1 = 1) :=
by
  sorry

end NUMINAMATH_GPT_find_a_if_f_is_odd_l1152_115202


namespace NUMINAMATH_GPT_min_value_frac_l1152_115270

open Real

theorem min_value_frac (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 4 * b = 1) :
  (1 / a + 2 / b) = 9 + 4 * sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_frac_l1152_115270


namespace NUMINAMATH_GPT_find_values_of_a_and_b_l1152_115247

theorem find_values_of_a_and_b (a b : ℚ) (h1 : 4 * a + 2 * b = 92) (h2 : 6 * a - 4 * b = 60) : 
  a = 122 / 7 ∧ b = 78 / 7 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_values_of_a_and_b_l1152_115247


namespace NUMINAMATH_GPT_value_of_a_l1152_115209

theorem value_of_a (a x y : ℤ) (h1 : x = 2) (h2 : y = 1) (h3 : a * x - 3 * y = 1) : a = 2 := by
  sorry

end NUMINAMATH_GPT_value_of_a_l1152_115209


namespace NUMINAMATH_GPT_percentage_increase_in_allowance_l1152_115212

def middle_school_allowance : ℕ := 8 + 2
def senior_year_allowance : ℕ := 2 * middle_school_allowance + 5

theorem percentage_increase_in_allowance : 
  (senior_year_allowance - middle_school_allowance) * 100 / middle_school_allowance = 150 := 
  by
    sorry

end NUMINAMATH_GPT_percentage_increase_in_allowance_l1152_115212


namespace NUMINAMATH_GPT_max_xy_l1152_115278

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry

-- Conditions given in the problem
axiom pos_x : 0 < x
axiom pos_y : 0 < y
axiom eq1 : x + 1/y = 3
axiom eq2 : y + 2/x = 3

theorem max_xy : ∃ (xy : ℝ), 
  xy = x * y ∧ xy = 3 + Real.sqrt 7 := sorry

end NUMINAMATH_GPT_max_xy_l1152_115278
