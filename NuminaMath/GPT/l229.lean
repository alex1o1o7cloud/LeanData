import Mathlib

namespace cosine_of_third_angle_l229_229519

theorem cosine_of_third_angle 
  (α β γ : ℝ) 
  (h1 : α < 40 * Real.pi / 180) 
  (h2 : β < 80 * Real.pi / 180) 
  (h3 : Real.sin γ = 5 / 8) :
  Real.cos γ = -Real.sqrt 39 / 8 := 
sorry

end cosine_of_third_angle_l229_229519


namespace donation_to_treetown_and_forest_reserve_l229_229559

noncomputable def donation_problem (x : ℕ) :=
  x + (x + 140) = 1000

theorem donation_to_treetown_and_forest_reserve :
  ∃ x : ℕ, donation_problem x ∧ (x + 140 = 570) := 
by
  sorry

end donation_to_treetown_and_forest_reserve_l229_229559


namespace toad_difference_l229_229647

variables (Tim_toads Jim_toads Sarah_toads : ℕ)

theorem toad_difference (h1 : Tim_toads = 30) 
                        (h2 : Jim_toads > Tim_toads) 
                        (h3 : Sarah_toads = 2 * Jim_toads) 
                        (h4 : Sarah_toads = 100) :
  Jim_toads - Tim_toads = 20 :=
by
  -- The next lines are placeholders for the logical steps which need to be proven
  sorry

end toad_difference_l229_229647


namespace find_n_in_arithmetic_sequence_l229_229168

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 4 then 7 else
  if n = 5 then 16 - 7 else sorry

-- Define the arithmetic sequence and the given conditions
theorem find_n_in_arithmetic_sequence (n : ℕ) (a : ℕ → ℕ) 
  (h1 : a 4 = 7) 
  (h2 : a 3 + a 6 = 16) 
  (h3 : a n = 31) :
  n = 16 :=
by
  sorry

end find_n_in_arithmetic_sequence_l229_229168


namespace multiple_of_sandy_age_l229_229365

theorem multiple_of_sandy_age
    (k_age : ℕ)
    (e : ℕ) 
    (s_current_age : ℕ) 
    (h1: k_age = 10) 
    (h2: e = 340) 
    (h3: s_current_age + 2 = 3 * (k_age + 2)) :
  e / s_current_age = 10 :=
by
  sorry

end multiple_of_sandy_age_l229_229365


namespace distinct_prime_factors_330_l229_229903

def num_prime_factors (n : ℕ) : ℕ :=
  if n = 330 then 4 else 0

theorem distinct_prime_factors_330 : num_prime_factors 330 = 4 :=
sorry

end distinct_prime_factors_330_l229_229903


namespace dan_present_age_l229_229165

-- Let x be Dan's present age
variable (x : ℤ)

-- Condition: Dan's age after 18 years will be 8 times his age 3 years ago
def condition (x : ℤ) : Prop :=
  x + 18 = 8 * (x - 3)

-- The goal is to prove that Dan's present age is 6
theorem dan_present_age (x : ℤ) (h : condition x) : x = 6 :=
by
  sorry

end dan_present_age_l229_229165


namespace fg_of_2_l229_229473

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x + 1)^2

theorem fg_of_2 : f (g 2) = 29 := by
  sorry

end fg_of_2_l229_229473


namespace area_of_triangle_ABC_l229_229351

theorem area_of_triangle_ABC (AB CD : ℝ) (height : ℝ) (h1 : CD = 3 * AB) (h2 : AB * height + CD * height = 48) :
  (1/2) * AB * height = 6 :=
by
  have trapezoid_area : AB * height + CD * height = 48 := h2
  have length_relation : CD = 3 * AB := h1
  have area_triangle_ABC := 6
  sorry

end area_of_triangle_ABC_l229_229351


namespace distribute_positions_l229_229668

structure DistributionProblem :=
  (volunteer_positions : ℕ)
  (schools : ℕ)
  (min_positions : ℕ)
  (distinct_allocations : ∀ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c)

noncomputable def count_ways (p : DistributionProblem) : ℕ :=
  if p.volunteer_positions = 7 ∧ p.schools = 3 ∧ p.min_positions = 1 then 6 else 0

theorem distribute_positions (p : DistributionProblem) :
  count_ways p = 6 :=
by
  sorry

end distribute_positions_l229_229668


namespace number_added_is_59_l229_229540

theorem number_added_is_59 (x : ℤ) (h1 : -2 < 0) (h2 : -3 < 0) (h3 : -2 * -3 + x = 65) : x = 59 :=
by sorry

end number_added_is_59_l229_229540


namespace sum_smallest_largest_eq_2z_l229_229655

theorem sum_smallest_largest_eq_2z (m b z : ℤ) (h1 : m > 0) (h2 : z = (b + (b + 2 * (m - 1))) / 2) :
  b + (b + 2 * (m - 1)) = 2 * z :=
sorry

end sum_smallest_largest_eq_2z_l229_229655


namespace concert_attendance_difference_l229_229623

noncomputable def first_concert : ℕ := 65899
noncomputable def second_concert : ℕ := 66018

theorem concert_attendance_difference :
  (second_concert - first_concert) = 119 :=
by
  sorry

end concert_attendance_difference_l229_229623


namespace expand_expression_l229_229508

variable (x y : ℝ)

theorem expand_expression :
  ((6 * x + 8 - 3 * y) * (4 * x - 5 * y)) = 
  (24 * x^2 - 42 * x * y + 32 * x - 40 * y + 15 * y^2) :=
by
  sorry

end expand_expression_l229_229508


namespace area_DEF_l229_229788

structure Point where
  x : ℝ
  y : ℝ

def D : Point := {x := -3, y := 4}
def E : Point := {x := 1, y := 7}
def F : Point := {x := 3, y := -1}

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * |(A.x * B.y + B.x * C.y + C.x * A.y - A.y * B.x - B.y * C.x - C.y * A.x)|

theorem area_DEF : area_of_triangle D E F = 16 := by
  sorry

end area_DEF_l229_229788


namespace find_c_l229_229879

-- Definitions for the conditions
def is_solution (x c : ℝ) : Prop := x^2 + c * x - 36 = 0

theorem find_c (c : ℝ) (h : is_solution (-9) c) : c = 5 :=
sorry

end find_c_l229_229879


namespace cole_time_to_work_is_90_minutes_l229_229127

noncomputable def cole_drive_time_to_work (D : ℝ) : ℝ := D / 30

def cole_trip_proof : Prop :=
  ∃ (D : ℝ), (D / 30) + (D / 90) = 2 ∧ cole_drive_time_to_work D * 60 = 90

theorem cole_time_to_work_is_90_minutes : cole_trip_proof :=
  sorry

end cole_time_to_work_is_90_minutes_l229_229127


namespace sallys_change_l229_229472

-- Define the total cost calculation:
def totalCost (numFrames : Nat) (costPerFrame : Nat) : Nat :=
  numFrames * costPerFrame

-- Define the change calculation:
def change (totalAmount : Nat) (amountPaid : Nat) : Nat :=
  amountPaid - totalAmount

-- Define the specific conditions in the problem:
def numFrames := 3
def costPerFrame := 3
def amountPaid := 20

-- Prove that the change Sally gets is $11:
theorem sallys_change : change (totalCost numFrames costPerFrame) amountPaid = 11 := by
  sorry

end sallys_change_l229_229472


namespace tangents_equal_l229_229980

theorem tangents_equal (α β γ : ℝ) (h1 : Real.sin α + Real.sin β + Real.sin γ = 0) (h2 : Real.cos α + Real.cos β + Real.cos γ = 0) :
  Real.tan (3 * α) = Real.tan (3 * β) ∧ Real.tan (3 * β) = Real.tan (3 * γ) := 
sorry

end tangents_equal_l229_229980


namespace total_fertilizer_used_l229_229764

def daily_fertilizer := 3
def num_days := 12
def extra_final_day := 6

theorem total_fertilizer_used : 
    (daily_fertilizer * num_days + (daily_fertilizer + extra_final_day)) = 45 :=
by
  sorry

end total_fertilizer_used_l229_229764


namespace prime_square_mod_24_l229_229909

theorem prime_square_mod_24 (p q : ℕ) (k : ℤ) 
  (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (hp_gt_5 : p > 5) (hq_gt_5 : q > 5) 
  (h_diff : p ≠ q)
  (h_eq : p^2 - q^2 = 6 * k) : (p^2 - q^2) % 24 = 0 := by
sorry

end prime_square_mod_24_l229_229909


namespace triangle_side_length_l229_229067

theorem triangle_side_length (a : ℝ) (h1 : 4 < a) (h2 : a < 8) : a = 6 :=
sorry

end triangle_side_length_l229_229067


namespace roots_g_eq_zero_l229_229572

noncomputable def g : ℝ → ℝ := sorry

theorem roots_g_eq_zero :
  (∀ x : ℝ, g (3 + x) = g (3 - x)) →
  (∀ x : ℝ, g (8 + x) = g (8 - x)) →
  (∀ x : ℝ, g (12 + x) = g (12 - x)) →
  g 0 = 0 →
  ∃ L : ℕ, 
  (∀ k, 0 ≤ k ∧ k ≤ L → g (k * 48) = 0) ∧ 
  (∀ k : ℤ, -1000 ≤ k ∧ k ≤ 1000 → (∃ n : ℕ, k = n * 48)) ∧ 
  L + 1 = 42 := 
by sorry

end roots_g_eq_zero_l229_229572


namespace minimum_P_ge_37_l229_229701

noncomputable def minimum_P (x y z : ℝ) : ℝ := 
  (x / y + y / z + z / x) * (y / x + z / y + x / z)

theorem minimum_P_ge_37 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (h : (x / y + y / z + z / x) + (y / x + z / y + x / z) = 10) : 
  minimum_P x y z ≥ 37 :=
sorry

end minimum_P_ge_37_l229_229701


namespace find_a5_from_geometric_sequence_l229_229034

def geo_seq (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = a n * q

noncomputable def geometric_sequence_property (a : ℕ → ℝ) (q : ℝ) :=
  geo_seq a q ∧ 0 < a 1 ∧ 0 < q ∧ 
  (a 4 = (a 2) ^ 2) ∧ 
  (a 2 + a 4 = 5 / 16)

theorem find_a5_from_geometric_sequence :
  ∀ (a : ℕ → ℝ) (q : ℝ), geometric_sequence_property a q → 
  a 5 = 1 / 32 :=
by 
  sorry

end find_a5_from_geometric_sequence_l229_229034


namespace total_amount_l229_229913

-- Definitions directly derived from the conditions in the problem
variable (you_spent friend_spent : ℕ)
variable (h1 : friend_spent = you_spent + 1)
variable (h2 : friend_spent = 8)

-- The goal is to prove that the total amount spent on lunch is $15
theorem total_amount : you_spent + friend_spent = 15 := by
  sorry

end total_amount_l229_229913


namespace length_of_FD_l229_229273

/-- In a square of side length 8 cm, point E is located on side AD,
2 cm from A and 6 cm from D. Point F lies on side CD such that folding
the square so that C coincides with E creates a crease along GF. 
Prove that the length of segment FD is 7/4 cm. -/
theorem length_of_FD (x : ℝ) (h_square : ∀ (A B C D : ℝ), A = 8 ∧ B = 8 ∧ C = 8 ∧ D = 8)
    (h_AE : ∀ (A E : ℝ), A - E = 2) (h_ED : ∀ (E D : ℝ), E - D = 6)
    (h_pythagorean : ∀ (x : ℝ), (8 - x)^2 = x^2 + 6^2) : x = 7/4 :=
by
  sorry

end length_of_FD_l229_229273


namespace intersection_M_complement_N_l229_229511

noncomputable def M := {y : ℝ | 1 ≤ y ∧ y ≤ 2}
noncomputable def N_complement := {x : ℝ | 1 ≤ x}

theorem intersection_M_complement_N : M ∩ N_complement = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end intersection_M_complement_N_l229_229511


namespace incorrect_counting_of_students_l229_229172

open Set

theorem incorrect_counting_of_students
  (total_students : ℕ)
  (english_only : ℕ)
  (german_only : ℕ)
  (french_only : ℕ)
  (english_german : ℕ)
  (english_french : ℕ)
  (german_french : ℕ)
  (all_three : ℕ)
  (reported_total : ℕ)
  (h_total_students : total_students = 100)
  (h_english_only : english_only = 30)
  (h_german_only : german_only = 23)
  (h_french_only : french_only = 50)
  (h_english_german : english_german = 10)
  (h_english_french : english_french = 8)
  (h_german_french : german_french = 20)
  (h_all_three : all_three = 5)
  (h_reported_total : reported_total = 100) :
  (english_only + german_only + french_only + english_german +
   english_french + german_french - 2 * all_three) ≠ reported_total :=
by
  sorry

end incorrect_counting_of_students_l229_229172


namespace total_eggs_collected_l229_229377

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l229_229377


namespace coordinates_of_point_B_l229_229884

theorem coordinates_of_point_B (A B : ℝ × ℝ) (AB : ℝ) :
  A = (-1, 2) ∧ B.1 = -1 ∧ AB = 3 ∧ (B.2 = 5 ∨ B.2 = -1) :=
by
  sorry

end coordinates_of_point_B_l229_229884


namespace beavers_working_l229_229784

theorem beavers_working (a b : ℝ) (h₁ : a = 2.0) (h₂ : b = 1.0) : a + b = 3.0 := 
by 
  rw [h₁, h₂]
  norm_num

end beavers_working_l229_229784


namespace range_of_a_l229_229924

noncomputable def f (a x : ℝ) := a / x - 1 + Real.log x

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 0 < x ∧ f a x ≤ 0) → a ≤ 1 := 
sorry

end range_of_a_l229_229924


namespace x_squared_minus_y_squared_l229_229203

theorem x_squared_minus_y_squared (x y : ℚ) (h₁ : x + y = 9 / 17) (h₂ : x - y = 1 / 51) : x^2 - y^2 = 1 / 289 :=
by
  sorry

end x_squared_minus_y_squared_l229_229203


namespace area_N1N2N3_relative_l229_229207

-- Definitions
variable (A B C D E F N1 N2 N3 : Type)
-- Assuming D, E, F are points on sides BC, CA, AB respectively such that CD, AE, BF are one-fourth of their respective sides.
variable (area_ABC : ℝ)  -- Total area of triangle ABC
variable (area_N1N2N3 : ℝ)  -- Area of triangle N1N2N3

-- Given conditions
variable (H1 : CD = 1 / 4 * BC)
variable (H2 : AE = 1 / 4 * CA)
variable (H3 : BF = 1 / 4 * AB)

-- The expected result
theorem area_N1N2N3_relative :
  area_N1N2N3 = 7 / 15 * area_ABC :=
sorry

end area_N1N2N3_relative_l229_229207


namespace ratio_equality_l229_229178

def op_def (a b : ℕ) : ℕ := a * b + b^2
def ot_def (a b : ℕ) : ℕ := a - b + a * b^2

theorem ratio_equality : (op_def 8 3 : ℚ) / (ot_def 8 3 : ℚ) = (33 : ℚ) / 77 := by
  sorry

end ratio_equality_l229_229178


namespace minimum_height_l229_229148

theorem minimum_height (x : ℝ) (h : ℝ) (A : ℝ) :
  (h = x + 4) →
  (A = 6*x^2 + 16*x) →
  (A ≥ 120) →
  (x ≥ 2) →
  h = 6 :=
by
  intros h_def A_def A_geq min_x
  sorry

end minimum_height_l229_229148


namespace inverse_proposition_is_false_l229_229154

theorem inverse_proposition_is_false (a : ℤ) (h : a = 6) : ¬ (|a| = 6 → a = 6) :=
sorry

end inverse_proposition_is_false_l229_229154


namespace geometric_sequence_value_of_b_l229_229187

-- Definitions
def is_geometric_sequence (a b c : ℝ) := 
  ∃ r : ℝ, a * r = b ∧ b * r = c

-- Theorem statement
theorem geometric_sequence_value_of_b (b : ℝ) (h : b > 0) 
  (h_seq : is_geometric_sequence 15 b 1) : b = Real.sqrt 15 :=
by
  sorry

end geometric_sequence_value_of_b_l229_229187


namespace angle_measure_l229_229665

theorem angle_measure (x y : ℝ) (h1 : x + y = 90) (h2 : x = 3 * y) : x = 67.5 :=
by
  -- sorry is used here to indicate that the proof is omitted
  sorry

end angle_measure_l229_229665


namespace brand_z_percentage_correct_l229_229769

noncomputable def percentage_of_brand_z (capacity : ℝ := 1) (brand_z1 : ℝ := 1) (brand_x1 : ℝ := 0) 
(brand_z2 : ℝ := 1/4) (brand_x2 : ℝ := 3/4) (brand_z3 : ℝ := 5/8) (brand_x3 : ℝ := 3/8) 
(brand_z4 : ℝ := 5/16) (brand_x4 : ℝ := 11/16) : ℝ :=
    (brand_z4 / (brand_z4 + brand_x4)) * 100

theorem brand_z_percentage_correct : percentage_of_brand_z = 31.25 := by
  sorry

end brand_z_percentage_correct_l229_229769


namespace find_b_l229_229306

theorem find_b (b : ℤ) (h₁ : b < 0) : (∃ n : ℤ, (x : ℤ) * x + b * x - 36 = (x + n) * (x + n) - 20) → b = -8 :=
by
  intro hX
  sorry

end find_b_l229_229306


namespace archery_competition_l229_229244

theorem archery_competition (points : Finset ℕ) (product : ℕ) : 
  points = {11, 7, 5, 2} ∧ product = 38500 → 
  ∃ n : ℕ, n = 7 := 
by
  intros h
  sorry

end archery_competition_l229_229244


namespace percentage_of_men_35_l229_229867

theorem percentage_of_men_35 (M W : ℝ) (hm1 : M + W = 100) 
  (hm2 : 0.6 * M + 0.2923 * W = 40)
  (hw : W = 100 - M) : 
  M = 35 :=
by
  sorry

end percentage_of_men_35_l229_229867


namespace reducible_iff_form_l229_229412

def isReducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem reducible_iff_form (a : ℕ) : isReducible a ↔ ∃ k : ℕ, a = 7 * k + 1 := by
  sorry

end reducible_iff_form_l229_229412


namespace total_selling_price_correct_l229_229354

-- Definitions of initial purchase prices in different currencies
def init_price_eur : ℕ := 600
def init_price_gbp : ℕ := 450
def init_price_usd : ℕ := 750

-- Definitions of initial exchange rates
def init_exchange_rate_eur_to_usd : ℝ := 1.1
def init_exchange_rate_gbp_to_usd : ℝ := 1.3

-- Definitions of profit percentages for each article
def profit_percent_eur : ℝ := 0.08
def profit_percent_gbp : ℝ := 0.1
def profit_percent_usd : ℝ := 0.15

-- Definitions of new exchange rates at the time of selling
def new_exchange_rate_eur_to_usd : ℝ := 1.15
def new_exchange_rate_gbp_to_usd : ℝ := 1.25

-- Calculation of purchase prices in USD
def purchase_price_in_usd₁ : ℝ := init_price_eur * init_exchange_rate_eur_to_usd
def purchase_price_in_usd₂ : ℝ := init_price_gbp * init_exchange_rate_gbp_to_usd
def purchase_price_in_usd₃ : ℝ := init_price_usd

-- Calculation of selling prices including profit in USD
def selling_price_in_usd₁ : ℝ := (init_price_eur + (init_price_eur * profit_percent_eur)) * new_exchange_rate_eur_to_usd
def selling_price_in_usd₂ : ℝ := (init_price_gbp + (init_price_gbp * profit_percent_gbp)) * new_exchange_rate_gbp_to_usd
def selling_price_in_usd₃ : ℝ := init_price_usd * (1 + profit_percent_usd)

-- Total selling price in USD
def total_selling_price_in_usd : ℝ :=
  selling_price_in_usd₁ + selling_price_in_usd₂ + selling_price_in_usd₃

-- Proof goal: total selling price should equal 2225.85 USD
theorem total_selling_price_correct :
  total_selling_price_in_usd = 2225.85 :=
by
  sorry

end total_selling_price_correct_l229_229354


namespace expected_winnings_l229_229281

theorem expected_winnings :
  let p_heads : ℚ := 1 / 4
  let p_tails : ℚ := 1 / 2
  let p_edge : ℚ := 1 / 4
  let win_heads : ℚ := 1
  let win_tails : ℚ := 3
  let loss_edge : ℚ := -8
  (p_heads * win_heads + p_tails * win_tails + p_edge * loss_edge) = -0.25 := 
by sorry

end expected_winnings_l229_229281


namespace bottles_per_case_l229_229101

theorem bottles_per_case (days: ℕ) (daily_intake: ℚ) (total_spent: ℚ) (case_cost: ℚ) (total_cases: ℕ) (total_bottles: ℕ) (B: ℕ) 
    (H1 : days = 240)
    (H2 : daily_intake = 1/2)
    (H3 : total_spent = 60)
    (H4 : case_cost = 12)
    (H5 : total_cases = total_spent / case_cost)
    (H6 : total_bottles = days * daily_intake)
    (H7 : B = total_bottles / total_cases) :
    B = 24 :=
by
    sorry

end bottles_per_case_l229_229101


namespace arithmetic_sequence_a1_l229_229145

theorem arithmetic_sequence_a1 (a : ℕ → ℝ) (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_inc : d > 0)
  (h_a3 : a 3 = 1)
  (h_a2a4 : a 2 * a 4 = 3 / 4) : 
  a 1 = 0 :=
sorry

end arithmetic_sequence_a1_l229_229145


namespace find_value_of_g1_l229_229258

variable (f : ℝ → ℝ)
variable (g : ℝ → ℝ)

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
∀ x : ℝ, g (-x) = g x

theorem find_value_of_g1 (h1 : is_odd f)
  (h2 : is_even g)
  (h3 : f (-1) + g 1 = 2)
  (h4 : f 1 + g (-1) = 4) : 
  g 1 = 3 :=
sorry

end find_value_of_g1_l229_229258


namespace complex_purely_imaginary_l229_229076

theorem complex_purely_imaginary (x : ℝ) :
  (x^2 - 1 = 0) → (x - 1 ≠ 0) → x = -1 :=
by
  intro h1 h2
  sorry

end complex_purely_imaginary_l229_229076


namespace perfect_square_trinomial_implies_value_of_a_l229_229387

theorem perfect_square_trinomial_implies_value_of_a (a : ℝ) :
  (∃ (b : ℝ), (∃ (x : ℝ), (x^2 - ax + 9 = 0) ∧ (x + b)^2 = x^2 - ax + 9)) ↔ a = 6 ∨ a = -6 :=
by
  sorry

end perfect_square_trinomial_implies_value_of_a_l229_229387


namespace swim_meet_time_l229_229461

theorem swim_meet_time {distance : ℕ} (d : distance = 50) (t : ℕ) 
  (meet_first : ∃ t1 : ℕ, t1 = 2 ∧ distance - 20 = 30) 
  (turn : ∀ t1, t1 = 2 → ∀ d1 : ℕ, d1 = 50 → t1 + t1 = 4) :
  t = 4 :=
by
  -- Placeholder proof
  sorry

end swim_meet_time_l229_229461


namespace find_M_base7_l229_229278

theorem find_M_base7 :
  ∃ M : ℕ, M = 48 ∧ (M^2).digits 7 = [6, 6] ∧ (∃ (m : ℕ), 49 ≤ m^2 ∧ m^2 < 343 ∧ M = m - 1) :=
sorry

end find_M_base7_l229_229278


namespace domain_f_2x_minus_1_l229_229663

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) :
  (∀ x, -2 ≤ x ∧ x ≤ 3 → ∃ y, f (x + 1) = y) →
  (∀ x, 0 ≤ x ∧ x ≤ 5 / 2 → ∃ y, f (2 * x - 1) = y) :=
by
  intro h
  sorry

end domain_f_2x_minus_1_l229_229663


namespace every_positive_integer_sum_of_distinct_powers_of_3_4_7_l229_229176

theorem every_positive_integer_sum_of_distinct_powers_of_3_4_7 :
  ∀ n : ℕ, n > 0 →
  ∃ (a b c : ℕ), a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  ∃ (i j k : ℕ), n = 3^i + 4^j + 7^k :=
by
  sorry

end every_positive_integer_sum_of_distinct_powers_of_3_4_7_l229_229176


namespace no_solution_exists_l229_229263

theorem no_solution_exists : ¬ ∃ n : ℕ, (n^2 ≡ 1 [MOD 5]) ∧ (n^3 ≡ 3 [MOD 5]) := 
sorry

end no_solution_exists_l229_229263


namespace projective_iff_fractional_linear_l229_229635

def projective_transformation (P : ℝ → ℝ) : Prop :=
  ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d))

theorem projective_iff_fractional_linear (P : ℝ → ℝ) : 
  projective_transformation P ↔ ∃ (a b c d : ℝ), (a * d - b * c ≠ 0) ∧ (∀ x : ℝ, P x = (a * x + b) / (c * x + d)) :=
by 
  sorry

end projective_iff_fractional_linear_l229_229635


namespace intersections_vary_with_A_l229_229928

theorem intersections_vary_with_A (A : ℝ) (hA : A > 0) :
  ∃ x y : ℝ, (y = A * x^2) ∧ (y^2 + 2 = x^2 + 6 * y) ∧ (y = 2 * x - 1) :=
sorry

end intersections_vary_with_A_l229_229928


namespace find_b_l229_229113

-- Variables representing the terms in the equations
variables (a b t : ℝ)

-- Conditions given in the problem
def cond1 : Prop := a - (t / 6) * b = 20
def cond2 : Prop := a - (t / 5) * b = -10
def t_value : Prop := t = 60

-- The theorem we need to prove
theorem find_b (H1 : cond1 a b t) (H2 : cond2 a b t) (H3 : t_value t) : b = 15 :=
by {
  -- Assuming the conditions are true
  sorry
}

end find_b_l229_229113


namespace distance_traveled_by_car_l229_229580

theorem distance_traveled_by_car (total_distance : ℕ) (fraction_foot : ℚ) (fraction_bus : ℚ)
  (h_total : total_distance = 40) (h_fraction_foot : fraction_foot = 1/4)
  (h_fraction_bus : fraction_bus = 1/2) :
  (total_distance * (1 - fraction_foot - fraction_bus)) = 10 :=
by
  sorry

end distance_traveled_by_car_l229_229580


namespace arithmetic_mean_of_fractions_l229_229492

theorem arithmetic_mean_of_fractions :
  (3 / 8 + 5 / 9 + 7 / 12) / 3 = 109 / 216 :=
by
  sorry

end arithmetic_mean_of_fractions_l229_229492


namespace sum_of_digits_of_N_plus_2021_is_10_l229_229319

-- The condition that N is the smallest positive integer whose digits add to 41.
def smallest_integer_with_digit_sum_41 (N : ℕ) : Prop :=
  (N > 0) ∧ ((N.digits 10).sum = 41)

-- The Lean 4 statement to prove the problem.
theorem sum_of_digits_of_N_plus_2021_is_10 :
  ∃ N : ℕ, smallest_integer_with_digit_sum_41 N ∧ ((N + 2021).digits 10).sum = 10 :=
by
  -- The proof would go here
  sorry

end sum_of_digits_of_N_plus_2021_is_10_l229_229319


namespace max_distance_proof_l229_229310

def highway_mpg : ℝ := 12.2
def city_mpg : ℝ := 7.6
def gasoline_gallons : ℝ := 21
def maximum_distance : ℝ := highway_mpg * gasoline_gallons

theorem max_distance_proof : maximum_distance = 256.2 := by
  sorry

end max_distance_proof_l229_229310


namespace increased_area_l229_229098

variable (r : ℝ)

theorem increased_area (r : ℝ) : 
  let initial_area : ℝ := π * r^2
  let final_area : ℝ := π * (r + 3)^2
  final_area - initial_area = 6 * π * r + 9 * π := by
sorry

end increased_area_l229_229098


namespace eden_stuffed_bears_l229_229637

theorem eden_stuffed_bears
  (initial_bears : ℕ)
  (favorite_bears : ℕ)
  (sisters : ℕ)
  (eden_initial_bears : ℕ)
  (remaining_bears := initial_bears - favorite_bears)
  (bears_per_sister := remaining_bears / sisters)
  (eden_bears_now := eden_initial_bears + bears_per_sister)
  (h1 : initial_bears = 20)
  (h2 : favorite_bears = 8)
  (h3 : sisters = 3)
  (h4 : eden_initial_bears = 10) :
  eden_bears_now = 14 := by
{
  sorry
}

end eden_stuffed_bears_l229_229637


namespace rope_total_in_inches_l229_229528

theorem rope_total_in_inches (feet_last_week feet_less_this_week feet_to_inch : ℕ) 
  (h1 : feet_last_week = 6)
  (h2 : feet_less_this_week = 4)
  (h3 : feet_to_inch = 12) :
  (feet_last_week + (feet_last_week - feet_less_this_week)) * feet_to_inch = 96 :=
by
  sorry

end rope_total_in_inches_l229_229528


namespace triangular_weight_60_grams_l229_229574

-- Define the weights as variables
variables {R T : ℝ} -- round weights and triangular weights are real numbers

-- Define the conditions as hypotheses
theorem triangular_weight_60_grams
  (h1 : R + T = 3 * R)
  (h2 : 4 * R + T = T + R + 90) :
  T = 60 :=
by
  -- indicate that the actual proof is omitted
  sorry

end triangular_weight_60_grams_l229_229574


namespace find_divisors_l229_229408

theorem find_divisors (N : ℕ) :
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ (N = 2013 ∨ N = 1006 ∨ N = 105 ∨ N = 52) := by
  sorry

end find_divisors_l229_229408


namespace range_of_a12_l229_229125

variable (a : ℕ → ℝ)
variable (a1 d : ℝ)

def arithmetic_seq (a : ℕ → ℝ) (a1 d : ℝ) : Prop :=
  ∀ n : ℕ, a n = a1 + n * d

variable (h_arith_seq : arithmetic_seq a a1 d)
variable (h_a8 : a 7 ≥ 15)
variable (h_a9 : a 8 ≤ 13)

theorem range_of_a12 : ∀ a1 d, (arithmetic_seq a a1 d) → (a 7 ≥ 15) → (a 8 ≤ 13) → (a 11 ≤ 7) :=
by
  intro a1 d h_arith_seq h_a8 h_a9
  sorry

end range_of_a12_l229_229125


namespace fill_cistern_time_l229_229950

-- Define the rates of the taps
def rateA := (1 : ℚ) / 3  -- Tap A fills 1 cistern in 3 hours (rate is 1/3 per hour)
def rateB := -(1 : ℚ) / 6  -- Tap B empties 1 cistern in 6 hours (rate is -1/6 per hour)
def rateC := (1 : ℚ) / 2  -- Tap C fills 1 cistern in 2 hours (rate is 1/2 per hour)

-- Define the combined rate
def combinedRate := rateA + rateB + rateC

-- The time to fill the cistern when all taps are opened simultaneously
def timeToFill := 1 / combinedRate

-- The theorem stating that the time to fill the cistern is 1.5 hours
theorem fill_cistern_time : timeToFill = (3 : ℚ) / 2 := by
  sorry  -- The proof is omitted as per the instructions

end fill_cistern_time_l229_229950


namespace Andy_is_late_l229_229666

def school_start_time : Nat := 8 * 60 -- in minutes (8:00 AM)
def normal_travel_time : Nat := 30 -- in minutes
def delay_red_lights : Nat := 4 * 3 -- in minutes (4 red lights * 3 minutes each)
def delay_construction : Nat := 10 -- in minutes
def delay_detour_accident : Nat := 7 -- in minutes
def delay_store_stop : Nat := 5 -- in minutes
def delay_searching_store : Nat := 2 -- in minutes
def delay_traffic : Nat := 15 -- in minutes
def delay_neighbor_help : Nat := 6 -- in minutes
def delay_closed_road : Nat := 8 -- in minutes
def all_delays : Nat := delay_red_lights + delay_construction + delay_detour_accident + delay_store_stop + delay_searching_store + delay_traffic + delay_neighbor_help + delay_closed_road
def departure_time : Nat := 7 * 60 + 15 -- in minutes (7:15 AM)

def arrival_time : Nat := departure_time + normal_travel_time + all_delays
def late_minutes : Nat := arrival_time - school_start_time

theorem Andy_is_late : late_minutes = 50 := by
  sorry

end Andy_is_late_l229_229666


namespace eval_sequence_l229_229632

noncomputable def b : ℕ → ℤ
| 1 => 1
| 2 => 4
| 3 => 9
| n => if h : n > 3 then b (n - 1) * (b (n - 1) - 1) + 1 else 0

theorem eval_sequence :
  b 1 * b 2 * b 3 * b 4 * b 5 * b 6 - (b 1 ^ 2 + b 2 ^ 2 + b 3 ^ 2 + b 4 ^ 2 + b 5 ^ 2 + b 6 ^ 2)
  = -3166598256 :=
by
  /- The proof steps are omitted. -/
  sorry

end eval_sequence_l229_229632


namespace math_problem_l229_229704

theorem math_problem (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^3 + y^3 = x - y) : x^2 + 4 * y^2 < 1 := 
sorry

end math_problem_l229_229704


namespace at_least_six_consecutive_heads_l229_229744

noncomputable def flip_probability : ℚ :=
  let total_outcomes := 2^8
  let successful_outcomes := 7
  successful_outcomes / total_outcomes

theorem at_least_six_consecutive_heads : 
  flip_probability = 7 / 256 :=
by
  sorry

end at_least_six_consecutive_heads_l229_229744


namespace max_marked_cells_100x100_board_l229_229876

theorem max_marked_cells_100x100_board : 
  ∃ n, (3 * n + 1 = 100) ∧ (2 * n + 1) * (n + 1) = 2278 :=
by
  sorry

end max_marked_cells_100x100_board_l229_229876


namespace spatial_quadrilateral_angle_sum_l229_229300

theorem spatial_quadrilateral_angle_sum (A B C D : ℝ) (ABD DBC ADB BDC : ℝ) :
  (A <= ABD + DBC) → (C <= ADB + BDC) → 
  (A + C + B + D <= 360) := 
by
  intros
  sorry

end spatial_quadrilateral_angle_sum_l229_229300


namespace father_three_times_marika_in_year_l229_229864

-- Define the given conditions as constants.
def marika_age_2004 : ℕ := 8
def father_age_2004 : ℕ := 32

-- Define the proof goal.
theorem father_three_times_marika_in_year :
  ∃ (x : ℕ), father_age_2004 + x = 3 * (marika_age_2004 + x) → 2004 + x = 2008 := 
by {
  sorry
}

end father_three_times_marika_in_year_l229_229864


namespace slices_per_person_l229_229827

namespace PizzaProblem

def pizzas : Nat := 3
def slices_per_pizza : Nat := 8
def coworkers : Nat := 12

theorem slices_per_person : (pizzas * slices_per_pizza) / coworkers = 2 := by
  sorry

end PizzaProblem

end slices_per_person_l229_229827


namespace line_through_circle_center_l229_229702

theorem line_through_circle_center
  (C : ℝ × ℝ)
  (hC : C = (-1, 0))
  (hCircle : ∀ (x y : ℝ), x^2 + 2 * x + y^2 = 0 → (x, y) = (-1, 0))
  (hPerpendicular : ∀ (m₁ m₂ : ℝ), (m₁ * m₂ = -1) → m₁ = -1 → m₂ = 1)
  (line_eq : ∀ (x y : ℝ), y = x + 1)
  : ∀ (x y : ℝ), x - y + 1 = 0 :=
sorry

end line_through_circle_center_l229_229702


namespace necessary_condition_l229_229549

theorem necessary_condition (m : ℝ) (h : ∀ x : ℝ, x^2 - x + m > 0) : m > 0 := 
sorry

end necessary_condition_l229_229549


namespace people_after_second_turn_l229_229121

noncomputable def number_of_people_in_front_after_second_turn (formation_size : ℕ) (initial_people : ℕ) (first_turn_people : ℕ) : ℕ := 
  if formation_size = 9 ∧ initial_people = 2 ∧ first_turn_people = 4 then 6 else 0

theorem people_after_second_turn :
  number_of_people_in_front_after_second_turn 9 2 4 = 6 :=
by
  -- Prove the theorem using the conditions and given data
  sorry

end people_after_second_turn_l229_229121


namespace arithmetic_geometric_sequence_ratio_l229_229958

theorem arithmetic_geometric_sequence_ratio (a : ℕ → ℕ) (d : ℕ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_positive_d : d > 0)
  (h_geometric : a 6 ^ 2 = a 2 * a 12) :
  (a 12) / (a 2) = 9 / 4 :=
sorry

end arithmetic_geometric_sequence_ratio_l229_229958


namespace towels_per_pack_l229_229537

open Nat

-- Define the given conditions
def packs : Nat := 9
def total_towels : Nat := 27

-- Define the property to prove
theorem towels_per_pack : total_towels / packs = 3 := by
  sorry

end towels_per_pack_l229_229537


namespace john_tour_days_l229_229149

noncomputable def numberOfDaysInTourProgram (d e : ℕ) : Prop :=
  d * e = 800 ∧ (d + 7) * (e - 5) = 800

theorem john_tour_days :
  ∃ (d e : ℕ), numberOfDaysInTourProgram d e ∧ d = 28 :=
by
  sorry

end john_tour_days_l229_229149


namespace graph_is_empty_l229_229566

/-- The given equation 3x² + 4y² - 12x - 16y + 36 = 0 has no real solutions. -/
theorem graph_is_empty : ∀ (x y : ℝ), 3 * x^2 + 4 * y^2 - 12 * x - 16 * y + 36 ≠ 0 :=
by
  intro x y
  sorry

end graph_is_empty_l229_229566


namespace new_barbell_cost_l229_229051

theorem new_barbell_cost (old_barbell_cost new_barbell_cost : ℝ) 
  (h1 : old_barbell_cost = 250)
  (h2 : new_barbell_cost = old_barbell_cost * 1.3) :
  new_barbell_cost = 325 := by
  sorry

end new_barbell_cost_l229_229051


namespace volume_of_prism_in_cubic_feet_l229_229344

theorem volume_of_prism_in_cubic_feet:
  let length_yd := 1
  let width_yd := 2
  let height_yd := 3
  let yard_to_feet := 3
  let length_ft := length_yd * yard_to_feet
  let width_ft := width_yd * yard_to_feet
  let height_ft := height_yd * yard_to_feet
  let volume := length_ft * width_ft * height_ft
  volume = 162 := by
  sorry

end volume_of_prism_in_cubic_feet_l229_229344


namespace candy_store_sampling_l229_229280

theorem candy_store_sampling (total_customers sampling_customers caught_customers not_caught_customers : ℝ)
    (h1 : caught_customers = 0.22 * total_customers)
    (h2 : not_caught_customers = 0.15 * sampling_customers)
    (h3 : sampling_customers = caught_customers + not_caught_customers):
    sampling_customers = 0.2588 * total_customers := by
  sorry

end candy_store_sampling_l229_229280


namespace solution_interval_for_x_l229_229111

theorem solution_interval_for_x (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (48 / 7 ≤ x ∧ x < 49 / 7) :=
by sorry

end solution_interval_for_x_l229_229111


namespace p_implies_q_and_not_converse_l229_229430

def p (a : ℝ) := a ≤ 1
def q (a : ℝ) := abs a ≤ 1

theorem p_implies_q_and_not_converse (a : ℝ) : (p a → q a) ∧ ¬(q a → p a) :=
by
  repeat { sorry }

end p_implies_q_and_not_converse_l229_229430


namespace alice_ate_more_l229_229032

theorem alice_ate_more (cookies : Fin 8 → ℕ) (h_alice : cookies 0 = 8) (h_tom : cookies 7 = 1) :
  cookies 0 - cookies 7 = 7 :=
by
  -- Placeholder for the actual proof, which is not required here
  sorry

end alice_ate_more_l229_229032


namespace GODOT_value_l229_229976

theorem GODOT_value (G O D I T : ℕ) (h1 : G ≠ 0) (h2 : D ≠ 0) 
  (eq1 : 1000 * G + 100 * O + 10 * G + O + 1000 * D + 100 * I + 10 * D + I = 10000 * G + 1000 * O + 100 * D + 10 * O + T) : 
  10000 * G + 1000 * O + 100 * D + 10 * O + T = 10908 :=
by {
  sorry
}

end GODOT_value_l229_229976


namespace calc_result_l229_229841

theorem calc_result : (-3)^2 - (-2)^3 = 17 := 
by
  sorry

end calc_result_l229_229841


namespace pipe_individual_empty_time_l229_229496

variable (a b c : ℝ)

noncomputable def timeToEmptyFirstPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * c + b * c - a * b)

noncomputable def timeToEmptySecondPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + b * c - a * c)

noncomputable def timeToEmptyThirdPipe (a b c : ℝ) : ℝ :=
  2 * a * b * c / (a * b + a * c - b * c)

theorem pipe_individual_empty_time
  (x y z : ℝ)
  (h1 : 1 / x + 1 / y = 1 / a)
  (h2 : 1 / x + 1 / z = 1 / b)
  (h3 : 1 / y + 1 / z = 1 / c) :
  x = timeToEmptyFirstPipe a b c ∧ y = timeToEmptySecondPipe a b c ∧ z = timeToEmptyThirdPipe a b c :=
sorry

end pipe_individual_empty_time_l229_229496


namespace f_negative_l229_229029

-- Let f be a function defined on the real numbers
variable (f : ℝ → ℝ)

-- Conditions: f is odd and given form for non-negative x
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom f_positive : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2 * x

theorem f_negative (x : ℝ) (hx : x < 0) : f x = -x^2 + 2 * x := by
  sorry

end f_negative_l229_229029


namespace alice_paid_percentage_l229_229998

theorem alice_paid_percentage {P : ℝ} (hP : P > 0)
  (hMP : ∀ P, MP = 0.60 * P)
  (hPrice_Alice_Paid : ∀ MP, Price_Alice_Paid = 0.40 * MP) :
  (Price_Alice_Paid / P) * 100 = 24 := by
  sorry

end alice_paid_percentage_l229_229998


namespace find_remaining_rectangle_area_l229_229577

-- Definitions of given areas
def S_DEIH : ℝ := 20
def S_HILK : ℝ := 40
def S_ABHG : ℝ := 126
def S_GHKJ : ℝ := 63
def S_DFMK : ℝ := 161

-- Definition of areas of the remaining rectangle
def S_EFML : ℝ := 101

-- Theorem statement to prove the area of the remaining rectangle
theorem find_remaining_rectangle_area :
  S_DFMK - S_DEIH - S_HILK = S_EFML :=
by
  -- This is where the proof would go
  sorry

end find_remaining_rectangle_area_l229_229577


namespace permutations_with_k_in_first_position_l229_229791

noncomputable def numberOfPermutationsWithKInFirstPosition (N k : ℕ) (h : k < N) : ℕ :=
  (2 : ℕ)^(N-1)

theorem permutations_with_k_in_first_position (N k : ℕ) (h : k < N) :
  numberOfPermutationsWithKInFirstPosition N k h = (2 : ℕ)^(N-1) :=
sorry

end permutations_with_k_in_first_position_l229_229791


namespace largest_even_integer_sum_l229_229922

theorem largest_even_integer_sum (x : ℤ) (h : (20 * (x + x + 38) / 2) = 6400) : 
  x + 38 = 339 :=
sorry

end largest_even_integer_sum_l229_229922


namespace no_integer_roots_of_polynomial_l229_229742

theorem no_integer_roots_of_polynomial :
  ¬ ∃ (x : ℤ), x^3 - 3 * x^2 - 10 * x + 20 = 0 :=
by
  sorry

end no_integer_roots_of_polynomial_l229_229742


namespace gcd_pow_sub_one_l229_229974

theorem gcd_pow_sub_one (a b : ℕ) 
  (h_a : a = 2^2004 - 1) 
  (h_b : b = 2^1995 - 1) : 
  Int.gcd a b = 511 :=
by
  sorry

end gcd_pow_sub_one_l229_229974


namespace rectangle_area_l229_229088

theorem rectangle_area :
  ∃ (x y : ℝ), (x + 3.5) * (y - 1.5) = x * y ∧
               (x - 3.5) * (y + 2.5) = x * y ∧
               2 * (x + 3.5) + 2 * (y - 3.5) = 2 * x + 2 * y ∧
               x * y = 196 :=
by
  sorry

end rectangle_area_l229_229088


namespace motorists_with_tickets_l229_229337

section SpeedingTickets

variables
  (total_motorists : ℕ)
  (percent_speeding : ℝ) -- percent_speeding is 25% (given)
  (percent_not_ticketed : ℝ) -- percent_not_ticketed is 60% (given)

noncomputable def percent_ticketed : ℝ :=
  let speeding_motorists := percent_speeding * total_motorists / 100
  let ticketed_motorists := speeding_motorists * ((100 - percent_not_ticketed) / 100)
  ticketed_motorists / total_motorists * 100

theorem motorists_with_tickets (total_motorists : ℕ) 
  (h1 : percent_speeding = 25)
  (h2 : percent_not_ticketed = 60) :
  percent_ticketed total_motorists percent_speeding percent_not_ticketed = 10 := 
by
  unfold percent_ticketed
  rw [h1, h2]
  sorry

end SpeedingTickets

end motorists_with_tickets_l229_229337


namespace isosceles_triangle_area_l229_229122

theorem isosceles_triangle_area 
  (x y : ℝ)
  (h_perimeter : 2*y + 2*x = 32)
  (h_height : ∃ h : ℝ, h = 8 ∧ y^2 = x^2 + h^2) :
  ∃ area : ℝ, area = 48 :=
by
  sorry

end isosceles_triangle_area_l229_229122


namespace palace_to_airport_distance_l229_229164

-- Let I be the distance from the palace to the airport
-- Let v be the speed of the Emir's car
-- Let t be the time taken to travel from the palace to the airport

theorem palace_to_airport_distance (v t I : ℝ) 
    (h1 : v = I / t) 
    (h2 : v + 20 = I / (t - 2 / 60)) 
    (h3 : v - 20 = I / (t + 3 / 60)) : 
    I = 20 := by
  sorry

end palace_to_airport_distance_l229_229164


namespace cube_vertices_probability_l229_229621

theorem cube_vertices_probability (totalVertices : ℕ) (selectedVertices : ℕ) 
   (totalCombinations : ℕ) (favorableOutcomes : ℕ) : 
   totalVertices = 8 ∧ selectedVertices = 4 ∧ totalCombinations = 70 ∧ favorableOutcomes = 12 → 
   (favorableOutcomes : ℚ) / totalCombinations = 6 / 35 := by
   sorry

end cube_vertices_probability_l229_229621


namespace total_questions_on_test_l229_229849

/-- A teacher grades students' tests by subtracting twice the number of incorrect responses
    from the number of correct responses. Given that a student received a score of 64
    and answered 88 questions correctly, prove that the total number of questions on the test is 100. -/
theorem total_questions_on_test (score correct_responses : ℕ) (grading_system : ℕ → ℕ → ℕ)
  (h1 : score = grading_system correct_responses (88 - 2 * 12))
  (h2 : correct_responses = 88)
  (h3 : score = 64) : correct_responses + (88 - 2 * 12) = 100 :=
by
  sorry

end total_questions_on_test_l229_229849


namespace solution_in_quadrant_II_l229_229821

theorem solution_in_quadrant_II (k x y : ℝ) (h1 : 2 * x + y = 6) (h2 : k * x - y = 4) : x < 0 ∧ y > 0 ↔ k < -2 :=
by
  sorry

end solution_in_quadrant_II_l229_229821


namespace tangent_line_parallel_to_given_line_l229_229703

theorem tangent_line_parallel_to_given_line 
  (x : ℝ) (y : ℝ) (tangent_line : ℝ → ℝ) :
  (tangent_line y = x^2 - 1) → 
  (tangent_line = 4) → 
  (4 * x - y - 5 = 0) :=
by 
  sorry

end tangent_line_parallel_to_given_line_l229_229703


namespace hyperbola_line_intersection_unique_l229_229857

theorem hyperbola_line_intersection_unique :
  ∀ (x y : ℝ), (x^2 / 9 - y^2 = 1) ∧ (y = 1/3 * (x + 1)) → ∃! p : ℝ × ℝ, p.1 = x ∧ p.2 = y :=
by
  sorry

end hyperbola_line_intersection_unique_l229_229857


namespace product_of_midpoint_coordinates_l229_229657

theorem product_of_midpoint_coordinates
  (x1 y1 x2 y2 : ℤ)
  (h1 : x1 = 4) (h2 : y1 = -3) (h3 : x2 = -8) (h4 : y2 = 7) :
  let mx := (x1 + x2) / 2
  let my := (y1 + y2) / 2
  (mx * my = -4) :=
by
  -- Here we would carry out the proof.
  sorry

end product_of_midpoint_coordinates_l229_229657


namespace tailor_trim_amount_l229_229722

variable (x : ℝ)

def original_side : ℝ := 22
def trimmed_side : ℝ := original_side - x
def fixed_trimmed_side : ℝ := original_side - 5
def remaining_area : ℝ := 120

theorem tailor_trim_amount :
  (original_side - x) * 17 = remaining_area → x = 15 :=
by
  intro h
  sorry

end tailor_trim_amount_l229_229722


namespace radius_large_circle_l229_229249

/-- Definitions for the problem context -/
structure Circle :=
(center : ℝ × ℝ)
(radius : ℝ)

noncomputable def tangent_circles (c1 c2 : Circle) : Prop :=
dist c1.center c2.center = c1.radius + c2.radius

/-- Theorem to prove the radius of the large circle -/
theorem radius_large_circle 
  (small_circle : Circle)
  (h_radius : small_circle.radius = 2)
  (large_circle : Circle)
  (h_tangency1 : tangent_circles small_circle large_circle)
  (small_circle2 : Circle)
  (small_circle3 : Circle)
  (h_tangency2 : tangent_circles small_circle small_circle2)
  (h_tangency3 : tangent_circles small_circle small_circle3)
  (h_tangency4 : tangent_circles small_circle2 large_circle)
  (h_tangency5 : tangent_circles small_circle3 large_circle)
  (h_tangency6 : tangent_circles small_circle2 small_circle3)
  : large_circle.radius = 2 * (Real.sqrt 3 + 1) :=
sorry

end radius_large_circle_l229_229249


namespace sum_of_number_is_8_l229_229059

theorem sum_of_number_is_8 (x v : ℝ) (h1 : 0.75 * x + 2 = v) (h2 : x = 8.0) : v = 8.0 :=
by
  sorry

end sum_of_number_is_8_l229_229059


namespace proof_statement_l229_229184

def convert_base_9_to_10 (n : Nat) : Nat :=
  3 * 9^2 + 2 * 9^1 + 4 * 9^0

def convert_base_6_to_10 (n : Nat) : Nat :=
  2 * 6^2 + 2 * 6^1 + 1 * 6^0

def problem_statement : Prop :=
  convert_base_9_to_10 324 - convert_base_6_to_10 221 = 180

theorem proof_statement : problem_statement := 
  by
    sorry

end proof_statement_l229_229184


namespace maximize_tetrahedron_volume_l229_229252

noncomputable def volume_of_tetrahedron (a : ℝ) : ℝ :=
  a / 6

theorem maximize_tetrahedron_volume (a : ℝ) (h_a : 0 < a) 
  (P Q X Y : ℝ × ℝ × ℝ) (h_PQ : dist P Q = 1) (h_XY : dist X Y = 1) :
  volume_of_tetrahedron a = a / 6 :=
by
  sorry

end maximize_tetrahedron_volume_l229_229252


namespace intersection_point_l229_229596

theorem intersection_point (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ d) :
  let x := (d - c) / (2 * b)
  let y := (a * (d - c)^2) / (4 * b^2) + (d + c) / 2
  (ax^2 + bx + c = y) ∧ (ax^2 - bx + d = y) :=
by
  sorry

end intersection_point_l229_229596


namespace problem_solution_l229_229670

theorem problem_solution : (275^2 - 245^2) / 30 = 520 := by
  sorry

end problem_solution_l229_229670


namespace top_layer_blocks_l229_229786

theorem top_layer_blocks (x : Nat) (h : x + 3 * x + 9 * x + 27 * x = 40) : x = 1 :=
by
  sorry

end top_layer_blocks_l229_229786


namespace product_of_solutions_l229_229374

theorem product_of_solutions : 
  ∀ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47) ∧ (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 :=
by
  sorry

end product_of_solutions_l229_229374


namespace required_sticks_l229_229078

variables (x y : ℕ)
variables (h1 : 2 * x + 3 * y = 96)
variables (h2 : x + y = 40)

theorem required_sticks (x y : ℕ) (h1 : 2 * x + 3 * y = 96) (h2 : x + y = 40) : 
  x = 24 ∧ y = 16 ∧ (96 - (x * 2 + y * 3) / 2) = 116 :=
by
  sorry

end required_sticks_l229_229078


namespace oscar_leap_longer_l229_229456

noncomputable def elmer_strides (poles : ℕ) (strides_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_strides := (poles - 1) * strides_per_gap
  total_distance / total_strides

noncomputable def oscar_leaps (poles : ℕ) (leaps_per_gap : ℕ) (distance_miles : ℝ) : ℝ :=
  let total_distance := distance_miles * 5280  -- convert miles to feet
  let total_leaps := (poles - 1) * leaps_per_gap
  total_distance / total_leaps

theorem oscar_leap_longer (poles : ℕ) (strides_per_gap leaps_per_gap : ℕ) (distance_miles : ℝ) :
  poles = 51 -> strides_per_gap = 50 -> leaps_per_gap = 15 -> distance_miles = 1.25 ->
  let elmer_stride := elmer_strides poles strides_per_gap distance_miles
  let oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  (oscar_leap - elmer_stride) * 12 = 74 :=
by
  intros h_poles h_strides h_leaps h_distance
  have elmer_stride := elmer_strides poles strides_per_gap distance_miles
  have oscar_leap := oscar_leaps poles leaps_per_gap distance_miles
  sorry

end oscar_leap_longer_l229_229456


namespace problem_solution_l229_229826

theorem problem_solution
  (x1 y1 x2 y2 x3 y3 : ℝ)
  (h1 : x1^3 - 3 * x1 * y1^2 = 2007)
  (h2 : y1^3 - 3 * x1^2 * y1 = 2006)
  (h3 : x2^3 - 3 * x2 * y2^2 = 2007)
  (h4 : y2^3 - 3 * x2^2 * y2 = 2006)
  (h5 : x3^3 - 3 * x3 * y3^2 = 2007)
  (h6 : y3^3 - 3 * x3^2 * y3 = 2006) :
  (1 - x1 / y1) * (1 - x2 / y2) * (1 - x3 / y3) = 1 / 1003 := 
sorry

end problem_solution_l229_229826


namespace num_four_letter_initials_sets_l229_229953

def num_initials_sets : ℕ := 8 ^ 4

theorem num_four_letter_initials_sets:
  num_initials_sets = 4096 :=
by
  rw [num_initials_sets]
  norm_num

end num_four_letter_initials_sets_l229_229953


namespace sum_five_consecutive_l229_229459

theorem sum_five_consecutive (n : ℤ) : (n + (n + 1) + (n + 2) + (n + 3) + (n + 4)) = 5 * n + 10 := by
  sorry

end sum_five_consecutive_l229_229459


namespace exists_pairs_angle_120_degrees_l229_229418

theorem exists_pairs_angle_120_degrees :
  ∃ a b : ℤ, a + b ≠ 0 ∧ a + b ≠ a ^ 2 - a * b + b ^ 2 ∧ (a + b) * 13 = 3 * (a ^ 2 - a * b + b ^ 2) :=
sorry

end exists_pairs_angle_120_degrees_l229_229418


namespace food_duration_l229_229314

theorem food_duration (mom_meals_per_day : ℕ) (mom_cups_per_meal : ℚ)
                      (puppy_count : ℕ) (puppy_meals_per_day : ℕ) (puppy_cups_per_meal : ℚ)
                      (total_food : ℚ)
                      (H_mom : mom_meals_per_day = 3) 
                      (H_mom_cups : mom_cups_per_meal = 3/2)
                      (H_puppies : puppy_count = 5) 
                      (H_puppy_meals : puppy_meals_per_day = 2) 
                      (H_puppy_cups : puppy_cups_per_meal = 1/2) 
                      (H_total_food : total_food = 57) : 
  (total_food / ((mom_meals_per_day * mom_cups_per_meal) + (puppy_count * puppy_meals_per_day * puppy_cups_per_meal))) = 6 := 
by
  sorry

end food_duration_l229_229314


namespace probability_two_even_multiples_of_five_drawn_l229_229901

-- Definition of conditions
def toys : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 
                      21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 
                      39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

def isEvenMultipleOfFive (n : ℕ) : Bool := n % 10 == 0

-- Collect all such numbers from the list
def evenMultiplesOfFive : List ℕ := toys.filter isEvenMultipleOfFive

-- Number of such even multiples of 5
def countEvenMultiplesOfFive : ℕ := evenMultiplesOfFive.length

theorem probability_two_even_multiples_of_five_drawn :
  (countEvenMultiplesOfFive / 50) * ((countEvenMultiplesOfFive - 1) / 49) = 2 / 245 :=
  by sorry

end probability_two_even_multiples_of_five_drawn_l229_229901


namespace smallest_portion_proof_l229_229675

theorem smallest_portion_proof :
  ∃ (a d : ℚ), 5 * a = 100 ∧ 3 * (a + d) = 2 * d + 7 * (a - 2 * d) ∧ a - 2 * d = 5 / 3 :=
by
  sorry

end smallest_portion_proof_l229_229675


namespace average_of_pqrs_l229_229158

variable (p q r s : ℝ)

theorem average_of_pqrs
  (h : (5 / 4) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 4 :=
by
  sorry

end average_of_pqrs_l229_229158


namespace minimize_sum_of_reciprocals_l229_229042

theorem minimize_sum_of_reciprocals (a b : ℕ) (h : 4 * a + b = 6) : 
  a = 1 ∧ b = 2 ∨ a = 2 ∧ b = 1 :=
by
  sorry

end minimize_sum_of_reciprocals_l229_229042


namespace by_how_much_were_the_numerator_and_denominator_increased_l229_229959

noncomputable def original_fraction_is_six_over_eleven (n : ℕ) : Prop :=
  n / (n + 5) = 6 / 11

noncomputable def resulting_fraction_is_seven_over_twelve (n x : ℕ) : Prop :=
  (n + x) / (n + 5 + x) = 7 / 12

theorem by_how_much_were_the_numerator_and_denominator_increased :
  ∃ (n x : ℕ), original_fraction_is_six_over_eleven n ∧ resulting_fraction_is_seven_over_twelve n x ∧ x = 1 :=
by
  sorry

end by_how_much_were_the_numerator_and_denominator_increased_l229_229959


namespace price_returns_to_initial_l229_229667

theorem price_returns_to_initial (x : ℝ) (h : 0.918 * (100 + x) = 100) : x = 9 := 
by
  sorry

end price_returns_to_initial_l229_229667


namespace two_polygons_sum_of_interior_angles_l229_229934

theorem two_polygons_sum_of_interior_angles
  (n1 n2 : ℕ) (h1 : Even n1) (h2 : Even n2) 
  (h_sum : (n1 - 2) * 180 + (n2 - 2) * 180 = 1800):
  (n1 = 4 ∧ n2 = 10) ∨ (n1 = 6 ∧ n2 = 8) :=
by
  sorry

end two_polygons_sum_of_interior_angles_l229_229934


namespace roman_numeral_sketching_l229_229877

/-- Roman numeral sketching problem. -/
theorem roman_numeral_sketching (n : ℕ) (k : ℕ) (students : ℕ) 
  (h1 : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n ∧ i / 1 = i) 
  (h2 : ∀ i : ℕ, i > n → i = n - (i - n)) 
  (h3 : k = 7) 
  (h4 : ∀ r : ℕ, r = (k * n)) : students = 350 :=
by
  sorry

end roman_numeral_sketching_l229_229877


namespace evaluate_expression_l229_229196

theorem evaluate_expression : ((5^2 + 3)^2 - (5^2 - 3)^2)^3 = 27000000 :=
by
  sorry

end evaluate_expression_l229_229196


namespace ariel_fish_l229_229033

theorem ariel_fish (total_fish : ℕ) (male_fraction female_fraction : ℚ) (h1 : total_fish = 45) (h2 : male_fraction = 2 / 3) (h3 : female_fraction = 1 - male_fraction) : total_fish * female_fraction = 15 :=
by
  sorry

end ariel_fish_l229_229033


namespace average_weight_increase_l229_229945

-- Define the initial conditions as given in the problem
def W_old : ℕ := 53
def W_new : ℕ := 71
def N : ℕ := 10

-- Average weight increase after replacing one oarsman
theorem average_weight_increase : (W_new - W_old : ℝ) / N = 1.8 := by
  sorry

end average_weight_increase_l229_229945


namespace college_students_freshmen_psych_majors_l229_229997

variable (T : ℕ)
variable (hT : T > 0)

def freshmen (T : ℕ) : ℕ := 40 * T / 100
def lib_arts (F : ℕ) : ℕ := 50 * F / 100
def psych_majors (L : ℕ) : ℕ := 50 * L / 100
def percent_freshmen_psych_majors (P : ℕ) (T : ℕ) : ℕ := 100 * P / T

theorem college_students_freshmen_psych_majors :
  percent_freshmen_psych_majors (psych_majors (lib_arts (freshmen T))) T = 10 := by
  sorry

end college_students_freshmen_psych_majors_l229_229997


namespace cannot_form_triangle_l229_229587

theorem cannot_form_triangle (a b c : ℕ) (h1 : a + b > c) (h2 : a + c > b) (h3 : b + c > a) : 
  ¬ ∃ a b c : ℕ, (a, b, c) = (1, 2, 3) := 
  sorry

end cannot_form_triangle_l229_229587


namespace pencils_left_l229_229009

-- Define the initial quantities
def MondayPencils := 35
def TuesdayPencils := 42
def WednesdayPencils := 3 * TuesdayPencils
def WednesdayLoss := 20
def ThursdayPencils := WednesdayPencils / 2
def FridayPencils := 2 * MondayPencils
def WeekendLoss := 50

-- Define the total number of pencils Sarah has at the end of each day
def TotalMonday := MondayPencils
def TotalTuesday := TotalMonday + TuesdayPencils
def TotalWednesday := TotalTuesday + WednesdayPencils - WednesdayLoss
def TotalThursday := TotalWednesday + ThursdayPencils
def TotalFriday := TotalThursday + FridayPencils
def TotalWeekend := TotalFriday - WeekendLoss

-- The proof statement
theorem pencils_left : TotalWeekend = 266 :=
by
  sorry

end pencils_left_l229_229009


namespace strips_area_coverage_l229_229404

-- Define paper strips and their properties
def length_strip : ℕ := 8
def width_strip : ℕ := 2
def number_of_strips : ℕ := 5

-- Total area without considering overlaps
def area_one_strip : ℕ := length_strip * width_strip
def total_area_without_overlap : ℕ := number_of_strips * area_one_strip

-- Overlapping areas
def area_center_overlap : ℕ := 4 * (2 * 2)
def area_additional_overlap : ℕ := 2 * (2 * 2)
def total_overlap_area : ℕ := area_center_overlap + area_additional_overlap

-- Actual area covered
def actual_area_covered : ℕ := total_area_without_overlap - total_overlap_area

-- Theorem stating the required proof
theorem strips_area_coverage : actual_area_covered = 56 :=
by sorry

end strips_area_coverage_l229_229404


namespace cinema_total_cost_l229_229512

theorem cinema_total_cost 
  (total_students : ℕ)
  (ticket_cost : ℕ)
  (half_price_interval : ℕ)
  (free_interval : ℕ)
  (half_price_cost : ℕ)
  (free_cost : ℕ)
  (total_cost : ℕ)
  (H_total_students : total_students = 84)
  (H_ticket_cost : ticket_cost = 50)
  (H_half_price_interval : half_price_interval = 12)
  (H_free_interval : free_interval = 35)
  (H_half_price_cost : half_price_cost = ticket_cost / 2)
  (H_free_cost : free_cost = 0)
  (H_total_cost : total_cost = 3925) :
  total_cost = ((total_students / half_price_interval) * half_price_cost +
                (total_students / free_interval) * free_cost +
                (total_students - (total_students / half_price_interval + total_students / free_interval)) * ticket_cost) :=
by 
  sorry

end cinema_total_cost_l229_229512


namespace rhombus_diagonal_l229_229352

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) (h1 : d1 = 20) (h2 : area = 170) :
  (area = (d1 * d2) / 2) → d2 = 17 :=
by
  sorry

end rhombus_diagonal_l229_229352


namespace linda_original_savings_l229_229173

theorem linda_original_savings (S : ℝ) (h1 : (2 / 3) * S + (1 / 3) * S = S) 
  (h2 : (1 / 3) * S = 250) : S = 750 :=
by sorry

end linda_original_savings_l229_229173


namespace domain_of_f_l229_229371

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f_l229_229371


namespace correct_statements_count_l229_229951

theorem correct_statements_count :
  (¬(1 = 1) ∧ ¬(1 = 0)) ∧
  (¬(1 = 11)) ∧
  ((1 - 2 + 1 / 2) = 3) ∧
  (2 = 2) →
  2 = ([false, false, true, true].count true) := 
sorry

end correct_statements_count_l229_229951


namespace smallest_units_C_union_D_l229_229163

-- Definitions for the sets C and D and their sizes
def C_units : ℝ := 25.5
def D_units : ℝ := 18.0

-- Definition stating the inclusion-exclusion principle for sets C and D
def C_union_D (C_units D_units C_intersection_units : ℝ) : ℝ :=
  C_units + D_units - C_intersection_units

-- Statement to prove the minimum units in C union D
theorem smallest_units_C_union_D : ∃ h, h ≤ C_union_D C_units D_units D_units ∧ h = 25.5 := by
  sorry

end smallest_units_C_union_D_l229_229163


namespace correct_option_B_l229_229747

variable {a b x y : ℤ}

def option_A (a : ℤ) : Prop := -a - a = 0
def option_B (x y : ℤ) : Prop := -(x + y) = -x - y
def option_C (b a : ℤ) : Prop := 3 * (b - 2 * a) = 3 * b - 2 * a
def option_D (a : ℤ) : Prop := 8 * a^4 - 6 * a^2 = 2 * a^2

theorem correct_option_B (x y : ℤ) : option_B x y := by
  -- The proof would go here
  sorry

end correct_option_B_l229_229747


namespace fraction_of_short_students_l229_229197

theorem fraction_of_short_students 
  (total_students tall_students average_students : ℕ) 
  (htotal : total_students = 400) 
  (htall : tall_students = 90) 
  (haverage : average_students = 150) : 
  (total_students - (tall_students + average_students)) / total_students = 2 / 5 :=
by
  sorry

end fraction_of_short_students_l229_229197


namespace find_a1_plus_a9_l229_229561

variable (a : ℕ → ℝ) (d : ℝ)

-- condition: arithmetic sequence
def is_arithmetic_seq : Prop := ∀ n, a (n + 1) = a n + d

-- condition: sum of specific terms
def sum_specific_terms : Prop := a 3 + a 4 + a 5 + a 6 + a 7 = 450

-- theorem: prove the desired sum
theorem find_a1_plus_a9 (h1 : is_arithmetic_seq a d) (h2 : sum_specific_terms a) : 
  a 1 + a 9 = 180 :=
  sorry

end find_a1_plus_a9_l229_229561


namespace negation_proof_l229_229276

theorem negation_proof :
  ¬ (∀ x : ℝ, 0 < x ∧ x < (π / 2) → x > Real.sin x) ↔ 
  ∃ x : ℝ, 0 < x ∧ x < (π / 2) ∧ x ≤ Real.sin x := 
sorry

end negation_proof_l229_229276


namespace represents_not_much_different_l229_229048

def not_much_different_from (x : ℝ) (c : ℝ) : Prop := x - c ≤ 0

theorem represents_not_much_different {x : ℝ} :
  (not_much_different_from x 2023) = (x - 2023 ≤ 0) :=
by
  sorry

end represents_not_much_different_l229_229048


namespace initial_pokemon_cards_l229_229476

theorem initial_pokemon_cards (x : ℕ) (h : x - 9 = 4) : x = 13 := by
  sorry

end initial_pokemon_cards_l229_229476


namespace solve_fraction_eq_l229_229363

theorem solve_fraction_eq (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 3) : (1 / (x - 1) = 3 / (x - 3)) ↔ x = 0 :=
by {
  sorry
}

end solve_fraction_eq_l229_229363


namespace sarah_trucks_l229_229413

-- Define the initial number of trucks denoted by T
def initial_trucks (T : ℝ) : Prop :=
  let left_after_jeff := T - 13.5
  let left_after_ashley := left_after_jeff - 0.25 * left_after_jeff
  left_after_ashley = 38

-- Theorem stating the initial number of trucks Sarah had is 64
theorem sarah_trucks : ∃ T : ℝ, initial_trucks T ∧ T = 64 :=
by
  sorry

end sarah_trucks_l229_229413


namespace sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l229_229699

-- Given conditions for the triangle ABC
variables {A B C a b c : ℝ}
axiom angle_C_eq_two_pi_over_three : C = 2 * Real.pi / 3
axiom c_squared_eq_five_a_squared_plus_ab : c^2 = 5 * a^2 + a * b

-- Proof statements
theorem sin_B_over_sin_A_eq_two (hAC: C = 2 * Real.pi / 3) (hCond: c^2 = 5 * a^2 + a * b) :
  Real.sin B / Real.sin A = 2 :=
sorry

theorem max_value_sin_A_sin_B (hAC: C = 2 * Real.pi / 3) :
  ∃ A B : ℝ, 0 < A ∧ A < Real.pi / 3 ∧ B = (Real.pi / 3 - A) ∧ Real.sin A * Real.sin B ≤ 1 / 4 :=
sorry

end sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l229_229699


namespace oil_bill_january_l229_229188

-- Define the problem in Lean
theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 := 
sorry

end oil_bill_january_l229_229188


namespace population_growth_rate_l229_229086

-- Define initial and final population
def initial_population : ℕ := 240
def final_population : ℕ := 264

-- Define the formula for calculating population increase rate
def population_increase_rate (P_i P_f : ℕ) : ℕ :=
  ((P_f - P_i) * 100) / P_i

-- State the theorem
theorem population_growth_rate :
  population_increase_rate initial_population final_population = 10 := by
  sorry

end population_growth_rate_l229_229086


namespace new_paint_intensity_l229_229194

def I1 : ℝ := 0.50
def I2 : ℝ := 0.25
def F : ℝ := 0.2

theorem new_paint_intensity : (1 - F) * I1 + F * I2 = 0.45 := by
  sorry

end new_paint_intensity_l229_229194


namespace slope_and_intercept_of_line_l229_229767

theorem slope_and_intercept_of_line :
  ∀ (x y : ℝ), 3 * x + 2 * y + 6 = 0 → y = - (3 / 2) * x - 3 :=
by
  intros x y h
  sorry

end slope_and_intercept_of_line_l229_229767


namespace value_of_x_when_z_is_32_l229_229008

variables {x y z k : ℝ}
variable (m n : ℝ)

def directly_proportional (x y : ℝ) (m : ℝ) := x = m * y^2
def inversely_proportional (y z : ℝ) (n : ℝ) := y = n / z^2

-- Our main proof goal
theorem value_of_x_when_z_is_32 (h1 : directly_proportional x y m) 
  (h2 : inversely_proportional y z n) (h3 : z = 8) (hx : x = 5) : 
  x = 5 / 256 :=
by
  let k := x * z^4
  have k_value : k = 20480 := by sorry
  have x_new : x = k / z^4 := by sorry
  have z_new : z = 32 := by sorry
  have x_final : x = 5 / 256 := by sorry
  exact x_final

end value_of_x_when_z_is_32_l229_229008


namespace MrFletcherPaymentPerHour_l229_229063

theorem MrFletcherPaymentPerHour :
  (2 * (10 + 8 + 15)) * x = 660 → x = 10 :=
by
  -- This is where you'd provide the proof, but we skip it as per instructions.
  sorry

end MrFletcherPaymentPerHour_l229_229063


namespace ratio_of_men_to_women_l229_229499

theorem ratio_of_men_to_women
  (M W : ℕ)
  (h1 : W = M + 6)
  (h2 : M + W = 16) :
  M * 11 = 5 * W :=
by
    -- We can explicitly construct the necessary proof here, but according to instructions we add sorry to bypass for now
    sorry

end ratio_of_men_to_women_l229_229499


namespace problem_solution_l229_229118

def base8_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 64 + ((n % 100) / 10) * 8 + (n % 10)

def base3_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 3 + (n % 10)

def base7_to_base10 (n : ℕ) : ℕ :=
  (n / 100) * 49 + ((n % 100) / 10) * 7 + (n % 10)

def base5_to_base10 (n : ℕ) : ℕ :=
  (n / 10) * 5 + (n % 10)

def expression_in_base10 : ℕ :=
  (base8_to_base10 254) / (base3_to_base10 13) + (base7_to_base10 232) / (base5_to_base10 32)

theorem problem_solution : expression_in_base10 = 35 :=
by
  sorry

end problem_solution_l229_229118


namespace cone_base_radius_l229_229800

/-- Given a semicircular piece of paper with a diameter of 2 cm is used to construct the 
  lateral surface of a cone, prove that the radius of the base of the cone is 0.5 cm. --/
theorem cone_base_radius (d : ℝ) (arc_length : ℝ) (circumference : ℝ) (r : ℝ)
  (h₀ : d = 2)
  (h₁ : arc_length = (1 / 2) * d * Real.pi)
  (h₂ : circumference = arc_length)
  (h₃ : r = circumference / (2 * Real.pi)) :
  r = 0.5 :=
by
  sorry

end cone_base_radius_l229_229800


namespace smallest_three_digit_divisible_by_4_and_5_l229_229357

theorem smallest_three_digit_divisible_by_4_and_5 : ∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, 100 ≤ m ∧ m < 1000 ∧ (m % 4 = 0) ∧ (m % 5 = 0) → m ≥ n →
n = 100 :=
sorry

end smallest_three_digit_divisible_by_4_and_5_l229_229357


namespace inequality1_inequality2_l229_229025

-- Problem 1
def f (x : ℝ) : ℝ := |2 * x + 1| + |x - 1|

theorem inequality1 (x : ℝ) : f x > 2 ↔ x < -2/3 ∨ x > 0 := sorry

-- Problem 2
def g (x : ℝ) : ℝ := f x + f (-x)

theorem inequality2 (k : ℝ) (h : ∀ x : ℝ, |k - 1| < g x) : -3 < k ∧ k < 5 := sorry

end inequality1_inequality2_l229_229025


namespace intercept_sum_equation_l229_229419

theorem intercept_sum_equation (c : ℝ) (h₀ : 3 * x + 4 * y + c = 0)
  (h₁ : (-(c / 3)) + (-(c / 4)) = 28) : c = -48 := 
by
  sorry

end intercept_sum_equation_l229_229419


namespace triangle_inequality_l229_229460

theorem triangle_inequality (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) : 
  a^2 * (b + c - a) + b^2 * (c + a - b) + c^2 * (a + b - c) ≤ 3 * a * b * c := 
by
  sorry

end triangle_inequality_l229_229460


namespace Tim_weekly_earnings_l229_229696

def number_of_tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def working_days_per_week : ℕ := 6

theorem Tim_weekly_earnings :
  (number_of_tasks_per_day * pay_per_task) * working_days_per_week = 720 := by
  sorry

end Tim_weekly_earnings_l229_229696


namespace probability_three_aligned_l229_229159

theorem probability_three_aligned (total_arrangements favorable_arrangements : ℕ) 
  (h1 : total_arrangements = 126)
  (h2 : favorable_arrangements = 48) :
  (favorable_arrangements : ℚ) / total_arrangements = 8 / 21 :=
by sorry

end probability_three_aligned_l229_229159


namespace find_angle_A_l229_229043

theorem find_angle_A (A B C : ℝ) (a b c : ℝ) :
  (a * Real.sin A = b * Real.sin B + (c - b) * Real.sin C)
  → (A = π / 3) :=
sorry

end find_angle_A_l229_229043


namespace find_m_for_parallel_lines_l229_229395

-- The given lines l1 and l2
def line1 (m: ℝ) : Prop := ∀ x y : ℝ, (3 + m) * x - 4 * y = 5 - 3 * m
def line2 : Prop := ∀ x y : ℝ, 2 * x - y = 8

-- Definition for parallel lines
def parallel_lines (l₁ l₂ : Prop) : Prop := 
  ∃ m : ℝ, (3 + m) / 4 = 2

-- The main theorem to prove
theorem find_m_for_parallel_lines (m: ℝ) (h: parallel_lines (line1 m) line2) : m = 5 :=
by sorry

end find_m_for_parallel_lines_l229_229395


namespace number_with_1_before_and_after_l229_229253

theorem number_with_1_before_and_after (n : ℕ) (hn : n < 10) : 100 * 1 + 10 * n + 1 = 101 + 10 * n := by
    sorry

end number_with_1_before_and_after_l229_229253


namespace kangaroo_mob_has_6_l229_229073

-- Define the problem conditions
def mob_of_kangaroos (W : ℝ) (k : ℕ) : Prop :=
  ∃ (two_lightest three_heaviest remaining : ℝ) (n_two n_three n_rem : ℕ),
    two_lightest = 0.25 * W ∧
    three_heaviest = 0.60 * W ∧
    remaining = 0.15 * W ∧
    n_two = 2 ∧
    n_three = 3 ∧
    n_rem = 1 ∧
    k = n_two + n_three + n_rem

-- The theorem to be proven
theorem kangaroo_mob_has_6 (W : ℝ) : ∃ k, mob_of_kangaroos W k ∧ k = 6 :=
by
  sorry

end kangaroo_mob_has_6_l229_229073


namespace find_m_value_l229_229532

noncomputable def x0 : ℝ := sorry

noncomputable def m : ℝ := x0^3 + 2 * x0^2 + 2

theorem find_m_value :
  (x0^2 + x0 - 1 = 0) → (m = 3) :=
by
  intro h
  have hx : x0 = sorry := sorry
  have hm : m = x0 ^ 3 + 2 * x0^2 + 2 := rfl
  rw [hx] at hm
  sorry

end find_m_value_l229_229532


namespace realize_ancient_dreams_only_C_l229_229678

-- Define the available options
inductive Options
| A : Options
| B : Options
| C : Options
| D : Options

-- Define the ancient dreams condition
def realize_ancient_dreams (o : Options) : Prop :=
  o = Options.C

-- The theorem states that only Geographic Information Technology (option C) can realize the ancient dreams
theorem realize_ancient_dreams_only_C :
  realize_ancient_dreams Options.C :=
by
  -- skip the exact proof
  sorry

end realize_ancient_dreams_only_C_l229_229678


namespace handshakes_meeting_l229_229174

theorem handshakes_meeting (x : ℕ) (h : x * (x - 1) / 2 = 66) : x = 12 := 
by 
  sorry

end handshakes_meeting_l229_229174


namespace area_of_trapezium_l229_229949

-- Defining the lengths of the sides and the distance
def a : ℝ := 12  -- 12 cm
def b : ℝ := 16  -- 16 cm
def h : ℝ := 14  -- 14 cm

-- Statement that the area of the trapezium is 196 cm²
theorem area_of_trapezium : (1 / 2) * (a + b) * h = 196 :=
by
  sorry

end area_of_trapezium_l229_229949


namespace Billys_age_l229_229957

variable (B J : ℕ)

theorem Billys_age :
  B = 2 * J ∧ B + J = 45 → B = 30 :=
by
  sorry

end Billys_age_l229_229957


namespace basketball_game_score_l229_229852

theorem basketball_game_score 
  (a r b d : ℕ)
  (H1 : a = b)
  (H2 : a + a * r + a * r^2 = b + (b + d) + (b + 2 * d))
  (H3 : a * (1 + r + r^2 + r^3) = 4 * b + 6 * d + 3)
  (H4 : r = 3)
  (H5 : a = 3)
  (H6 : d = 10)
  (H7 : a * (1 + r) = 12)
  (H8 : b * (1 + 3 + (b + d)) = 16) :
  a + a * r + b + (b + d) = 28 :=
by simp [H4, H5, H6, H7, H8]; linarith

end basketball_game_score_l229_229852


namespace sin_squared_plus_one_l229_229211

theorem sin_squared_plus_one (x : ℝ) (hx : Real.tan x = 2) : Real.sin x ^ 2 + 1 = 9 / 5 := 
by 
  sorry

end sin_squared_plus_one_l229_229211


namespace two_distinct_real_roots_of_modified_quadratic_l229_229372

theorem two_distinct_real_roots_of_modified_quadratic (a b k : ℝ) (h1 : a^2 - b > 0) (h2 : k > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + 2 * a * x₁ + b + k * (x₁ + a)^2 = 0) ∧ (x₂^2 + 2 * a * x₂ + b + k * (x₂ + a)^2 = 0) :=
by
  sorry

end two_distinct_real_roots_of_modified_quadratic_l229_229372


namespace lines_parallel_l229_229823

-- Definitions based on conditions
variable (line1 line2 : ℝ → ℝ → Prop) -- Assuming lines as relations for simplicity
variable (plane : ℝ → ℝ → ℝ → Prop) -- Assuming plane as a relation for simplicity

-- Condition: Both lines are perpendicular to the same plane
def perpendicular_to_plane (line : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop) : Prop :=
  ∀ (x y z : ℝ), plane x y z → line x y

axiom line1_perpendicular : perpendicular_to_plane line1 plane
axiom line2_perpendicular : perpendicular_to_plane line2 plane

-- Theorem: Both lines are parallel
theorem lines_parallel : ∀ (line1 line2 : ℝ → ℝ → Prop) (plane : ℝ → ℝ → ℝ → Prop),
  (perpendicular_to_plane line1 plane) →
  (perpendicular_to_plane line2 plane) →
  (∀ x y : ℝ, line1 x y → line2 x y) := sorry

end lines_parallel_l229_229823


namespace domain_of_function_l229_229485

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1) + 1 / Real.sqrt (2 - x^2)

theorem domain_of_function : 
  {x : ℝ | x > -1 ∧ x < Real.sqrt 2} = {x : ℝ | x ∈ Set.Ioo (-1) (Real.sqrt 2)} :=
by
  sorry

end domain_of_function_l229_229485


namespace inequality_1_inequality_2_inequality_4_l229_229080

variable {a b : ℝ}

def condition (a b : ℝ) : Prop := (1/a < 1/b) ∧ (1/b < 0)

theorem inequality_1 (ha : a < 0) (hb : b < 0) (hc : condition a b) : a + b < a * b :=
sorry

theorem inequality_2 (hc : condition a b) : |a| < |b| :=
sorry

theorem inequality_4 (hc : condition a b) : (b / a) + (a / b) > 2 :=
sorry

end inequality_1_inequality_2_inequality_4_l229_229080


namespace maximum_area_of_triangle_ABC_l229_229985

noncomputable def max_area_triangle_ABC (a b c : ℝ) (A B C : ℝ) : ℝ :=
  1/2 * b * c * Real.sin A

theorem maximum_area_of_triangle_ABC (a b c A B C : ℝ) 
  (h1: a = 4) 
  (h2: (4 + b) * (Real.sin A - Real.sin B) = (c - b) * Real.sin C) :
  max_area_triangle_ABC a b c A B C = 4 * Real.sqrt 3 := 
sorry

end maximum_area_of_triangle_ABC_l229_229985


namespace triangle_area_l229_229546

/-- Proof that the area of a triangle with side lengths 9 cm, 40 cm, and 41 cm is 180 square centimeters, 
    given that these lengths form a right triangle. -/
theorem triangle_area : ∀ (a b c : ℕ), a = 9 → b = 40 → c = 41 → a^2 + b^2 = c^2 → (a * b) / 2 = 180 := by
  intros a b c ha hb hc hpyth
  sorry

end triangle_area_l229_229546


namespace wine_age_problem_l229_229366

theorem wine_age_problem
  (C F T B Bo : ℕ)
  (h1 : F = 3 * C)
  (h2 : C = 4 * T)
  (h3 : B = (1 / 2 : ℝ) * T)
  (h4 : Bo = 2 * F)
  (h5 : C = 40) :
  F = 120 ∧ T = 10 ∧ B = 5 ∧ Bo = 240 := 
  by
    sorry

end wine_age_problem_l229_229366


namespace animals_on_stump_l229_229362

def possible_n_values (n : ℕ) : Prop :=
  n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 8 ∨ n = 9 ∨ n = 12 ∨ n = 15

theorem animals_on_stump (n : ℕ) (h1 : n ≥ 3) (h2 : n ≤ 20)
  (h3 : 11 ≥ (n + 1) / 3) (h4 : 9 ≥ n - (n + 1) / 3) : possible_n_values n :=
by {
  sorry
}

end animals_on_stump_l229_229362


namespace find_initial_amount_l229_229793

theorem find_initial_amount
  (A : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hA : A = 1050)
  (hR : R = 8)
  (hT : T = 5) :
  P = 750 :=
by
  have hSI : P * R * T / 100 = 1050 - P := sorry
  have hFormulaSimplified : P * 0.4 = 1050 - P := sorry
  have hFinal : P * 1.4 = 1050 := sorry
  exact sorry

end find_initial_amount_l229_229793


namespace max_area_triangle_max_area_quadrilateral_l229_229271

-- Define the terms and conditions

variables {A O : Point}
variables {r d : ℝ}
variables {C D B : Point}

-- Problem (a)
theorem max_area_triangle (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (3 / 4) * d) :=
sorry

-- Problem (b)
theorem max_area_quadrilateral (A O : Point) (d : ℝ) :
  (∃ x : ℝ, x = (1 / 2) * d) :=
sorry

end max_area_triangle_max_area_quadrilateral_l229_229271


namespace largest_angle_of_triangle_l229_229815

theorem largest_angle_of_triangle 
  (α β γ : ℝ) 
  (h1 : α = 60) 
  (h2 : β = 70) 
  (h3 : α + β + γ = 180) : 
  max α (max β γ) = 70 := 
by 
  sorry

end largest_angle_of_triangle_l229_229815


namespace am_gm_inequality_l229_229099

theorem am_gm_inequality {a b : ℝ} (n : ℕ) (h₁ : n ≠ 1) (h₂ : a > b) (h₃ : b > 0) : 
  ( (a + b) / 2 )^n < (a^n + b^n) / 2 := 
sorry

end am_gm_inequality_l229_229099


namespace sum_of_roots_of_cis_equation_l229_229380

theorem sum_of_roots_of_cis_equation 
  (cis : ℝ → ℂ)
  (phi : ℕ → ℝ)
  (h_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 0 ≤ phi k ∧ phi k < 360)
  (h_equation : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → (cis (phi k)) ^ 5 = (1 / Real.sqrt 2) + (Complex.I / Real.sqrt 2))
  : (phi 1 + phi 2 + phi 3 + phi 4 + phi 5) = 450 :=
by
  sorry

end sum_of_roots_of_cis_equation_l229_229380


namespace circular_sequence_zero_if_equidistant_l229_229603

noncomputable def circular_sequence_property (x y z : ℤ): Prop :=
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0

theorem circular_sequence_zero_if_equidistant {x y z : ℤ} :
  (x = 0 ∧ y = 0 ∧ dist x y = dist y z) → z = 0 :=
by sorry

end circular_sequence_zero_if_equidistant_l229_229603


namespace jerry_added_action_figures_l229_229780

theorem jerry_added_action_figures (x : ℕ) (h1 : 7 + x - 10 = 8) : x = 11 :=
by
  sorry

end jerry_added_action_figures_l229_229780


namespace toys_left_after_two_weeks_l229_229806

theorem toys_left_after_two_weeks
  (initial_stock : ℕ)
  (sold_first_week : ℕ)
  (sold_second_week : ℕ)
  (total_stock : initial_stock = 83)
  (first_week_sales : sold_first_week = 38)
  (second_week_sales : sold_second_week = 26) :
  initial_stock - (sold_first_week + sold_second_week) = 19 :=
by
  sorry

end toys_left_after_two_weeks_l229_229806


namespace total_number_of_flowers_l229_229438

theorem total_number_of_flowers : 
  let red_roses := 1491
  let yellow_carnations := 3025
  let white_roses := 1768
  let purple_tulips := 2150
  let pink_daisies := 3500
  let blue_irises := 2973
  let orange_marigolds := 4234
  red_roses + yellow_carnations + white_roses + purple_tulips + pink_daisies + blue_irises + orange_marigolds = 19141 :=
by 
  sorry

end total_number_of_flowers_l229_229438


namespace polygon_sides_l229_229737

theorem polygon_sides (n : ℕ) (h : 144 * n = 180 * (n - 2)) : n = 10 :=
by { sorry }

end polygon_sides_l229_229737


namespace triangle_area_rational_l229_229001

-- Define the conditions
def satisfies_eq (x y : ℤ) : Prop := x - y = 1

-- Define the points
variables (x1 y1 x2 y2 x3 y3 : ℤ)

-- Assume each point satisfies the equation
axiom point1 : satisfies_eq x1 y1
axiom point2 : satisfies_eq x2 y2
axiom point3 : satisfies_eq x3 y3

-- Statement that we need to prove
theorem triangle_area_rational :
  ∃ (area : ℚ), 
    ∃ (triangle_points : ∃ (x1 y1 x2 y2 x3 y3 : ℤ), satisfies_eq x1 y1 ∧ satisfies_eq x2 y2 ∧ satisfies_eq x3 y3), 
      true :=
sorry

end triangle_area_rational_l229_229001


namespace first_number_is_210_l229_229250

theorem first_number_is_210 (A B hcf lcm : ℕ) (h1 : lcm = 2310) (h2: hcf = 47) (h3 : B = 517) :
  A * B = lcm * hcf → A = 210 :=
by
  sorry

end first_number_is_210_l229_229250


namespace additional_carpet_needed_l229_229077

-- Definitions according to the given conditions
def length_feet := 18
def width_feet := 12
def covered_area := 4 -- in square yards
def feet_per_yard := 3

-- Prove that the additional square yards needed to cover the remaining part of the floor is 20
theorem additional_carpet_needed : 
  ((length_feet / feet_per_yard) * (width_feet / feet_per_yard) - covered_area) = 20 := 
by
  sorry

end additional_carpet_needed_l229_229077


namespace circles_tangent_dist_l229_229943

theorem circles_tangent_dist (t : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 4) ∧ 
  (∀ x y : ℝ, (x - t)^2 + y^2 = 1) ∧ 
  (∀ x1 y1 x2 y2 : ℝ, x1^2 + y1^2 = 4 → (x2 - t)^2 + y2^2 = 1 → 
    dist (x1, y1) (x2, y2) = 3) → 
  t = 3 ∨ t = -3 :=
by 
  sorry

end circles_tangent_dist_l229_229943


namespace number_of_elements_l229_229189

noncomputable def set_mean (S : Set ℝ) : ℝ := sorry

theorem number_of_elements (S : Set ℝ) (M : ℝ)
  (h1 : set_mean (S ∪ {15}) = M + 2)
  (h2 : set_mean (S ∪ {15, 1}) = M + 1) :
  ∃ k : ℕ, (M * k + 15 = (M + 2) * (k + 1)) ∧ (M * k + 16 = (M + 1) * (k + 2)) ∧ k = 4 := sorry

end number_of_elements_l229_229189


namespace Jakes_height_is_20_l229_229895

-- Define the conditions
def Sara_width : ℤ := 12
def Sara_height : ℤ := 24
def Sara_depth : ℤ := 24
def Jake_width : ℤ := 16
def Jake_depth : ℤ := 18
def volume_difference : ℤ := 1152

-- Volume calculation
def Sara_volume : ℤ := Sara_width * Sara_height * Sara_depth

-- Prove Jake's height is 20 inches
theorem Jakes_height_is_20 :
  ∃ h : ℤ, (Sara_volume - (Jake_width * h * Jake_depth) = volume_difference) ∧ h = 20 :=
by
  sorry

end Jakes_height_is_20_l229_229895


namespace problem_I_problem_II_l229_229979

-- Problem (I): Proving the inequality solution set
theorem problem_I (x : ℝ) : |x - 5| + |x + 6| ≤ 12 ↔ -13/2 ≤ x ∧ x ≤ 11/2 :=
by
  sorry

-- Problem (II): Proving the range of m
theorem problem_II (m : ℝ) : (∀ x : ℝ, |x - m| + |x + 6| ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end problem_I_problem_II_l229_229979


namespace sum_of_three_pentagons_l229_229324

variable (x y : ℚ)

axiom eq1 : 3 * x + 2 * y = 27
axiom eq2 : 2 * x + 3 * y = 25

theorem sum_of_three_pentagons : 3 * y = 63 / 5 := 
by {
  sorry -- No need to provide proof steps
}

end sum_of_three_pentagons_l229_229324


namespace greatest_number_divisible_by_11_and_3_l229_229343

namespace GreatestNumberDivisibility

theorem greatest_number_divisible_by_11_and_3 : 
  ∃ (A B C : ℕ), 
    A ≠ B ∧ A ≠ C ∧ B ≠ C ∧ 
    (2 * A - 2 * B + C) % 11 = 0 ∧ 
    (2 * A + 2 * C + B) % 3 = 0 ∧
    (10000 * A + 1000 * C + 100 * C + 10 * B + A) = 95695 :=
by
  -- The proof here is omitted.
  sorry

end GreatestNumberDivisibility

end greatest_number_divisible_by_11_and_3_l229_229343


namespace distance_from_tangency_to_tangent_l229_229243

theorem distance_from_tangency_to_tangent 
  (R r : ℝ)
  (hR : R = 3)
  (hr : r = 1)
  (externally_tangent : true) :
  ∃ d : ℝ, (d = 0 ∨ d = 7/3) :=
by
  sorry

end distance_from_tangency_to_tangent_l229_229243


namespace average_of_remaining_two_numbers_l229_229407

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
(h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
(h_avg_2_1 : (a + b) / 2 = 3.4)
(h_avg_2_2 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 := 
sorry

end average_of_remaining_two_numbers_l229_229407


namespace constant_term_expansion_l229_229013

theorem constant_term_expansion (x : ℝ) (hx : x ≠ 0) : 
  let term (r : ℕ) : ℝ := (1 / 2) ^ (9 - r) * (-1) ^ r * Nat.choose 9 r * x ^ (3 / 2 * r - 9)
  term 6 = 21 / 2 :=
by
  sorry

end constant_term_expansion_l229_229013


namespace michael_eggs_count_l229_229868

def initial_crates : List ℕ := [24, 28, 32, 36, 40, 44]
def wednesday_given : List ℕ := [28, 32, 40]
def thursday_purchases : List ℕ := [50, 45, 55, 60]
def friday_sold : List ℕ := [60, 55]

theorem michael_eggs_count :
  let total_tuesday := initial_crates.sum
  let total_given_wednesday := wednesday_given.sum
  let remaining_wednesday := total_tuesday - total_given_wednesday
  let total_thursday := thursday_purchases.sum
  let total_after_thursday := remaining_wednesday + total_thursday
  let total_sold_friday := friday_sold.sum
  total_after_thursday - total_sold_friday = 199 :=
by
  sorry

end michael_eggs_count_l229_229868


namespace extreme_values_f_range_of_a_l229_229752

noncomputable def f (x a : ℝ) : ℝ := x^3 - x^2 - x - a
noncomputable def df (x : ℝ) : ℝ := 3 * x^2 - 2 * x - 1

theorem extreme_values_f (a : ℝ) :
  ∃ (x₁ x₂ : ℝ), df x₁ = 0 ∧ df x₂ = 0 ∧ f x₁ a = (5 / 27) - a ∧ f x₂ a = -1 - a :=
sorry

theorem range_of_a (a : ℝ) :
  (∃ (a : ℝ), f (-1/3) a < 0 ∧ f 1 a > 0) ↔ (a < -1 ∨ a > 5 / 27) :=
sorry

end extreme_values_f_range_of_a_l229_229752


namespace plantingMethodsCalculation_l229_229944

noncomputable def numPlantingMethods : Nat :=
  let totalSeeds := 5
  let endChoices := 3 * 2 -- Choosing 2 seeds for the ends from 3 remaining types
  let middleChoices := 6 -- Permutations of (A, B, another type) = 3! = 6
  endChoices * middleChoices

theorem plantingMethodsCalculation : numPlantingMethods = 24 := by
  sorry

end plantingMethodsCalculation_l229_229944


namespace general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l229_229299

open Real

-- Definitions for the problem
variable (t : ℝ) (φ θ : ℝ) (x y P : ℝ)

-- Conditions
def line_parametric := x = t * sin φ ∧ y = 1 + t * cos φ
def curve_polar := P * (cos θ)^2 = 4 * sin θ
def curve_cartesian := x^2 = 4 * y
def line_general := x * cos φ - y * sin φ + sin φ = 0

-- Proof problem statements

-- 1. Prove the general equation of line l
theorem general_equation_of_line (h : line_parametric t φ x y) : line_general φ x y :=
sorry

-- 2. Prove the cartesian coordinate equation of curve C
theorem cartesian_equation_of_curve (h : curve_polar P θ) : curve_cartesian x y :=
sorry

-- 3. Prove the minimum |AB| where line l intersects curve C
theorem minimum_AB (h_line : line_parametric t φ x y) (h_curve : curve_cartesian x y) : ∃ (min_ab : ℝ), min_ab = 4 :=
sorry

end general_equation_of_line_cartesian_equation_of_curve_minimum_AB_l229_229299


namespace rowing_distance_l229_229453

theorem rowing_distance (D : ℝ) 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (total_time : ℝ) 
  (downstream_speed : ℝ := boat_speed + stream_speed) 
  (upstream_speed : ℝ := boat_speed - stream_speed)
  (downstream_time : ℝ := D / downstream_speed)
  (upstream_time : ℝ := D / upstream_speed)
  (round_trip_time : ℝ := downstream_time + upstream_time) 
  (h1 : boat_speed = 16) 
  (h2 : stream_speed = 2) 
  (h3 : total_time = 914.2857142857143)
  (h4 : round_trip_time = total_time) :
  D = 720 :=
by sorry

end rowing_distance_l229_229453


namespace cube_painting_l229_229633

-- Let's start with importing Mathlib for natural number operations

theorem cube_painting (n : ℕ) (h : 2 < n)
  (num_one_black_face : ℕ := 3 * (n - 2)^2)
  (num_unpainted : ℕ := (n - 2)^3) :
  num_one_black_face = num_unpainted → n = 5 :=
by
  sorry

end cube_painting_l229_229633


namespace unique_solution_m_n_eq_l229_229494

theorem unique_solution_m_n_eq (m n : ℕ) (h : m^2 = (10 * n + 1) * n + 2) : (m, n) = (11, 7) := by
  sorry

end unique_solution_m_n_eq_l229_229494


namespace no_integer_roots_l229_229322

  theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 4 * x + 24 ≠ 0 :=
  by
    sorry
  
end no_integer_roots_l229_229322


namespace binary_addition_l229_229942

-- Define the binary numbers as natural numbers
def b1 : ℕ := 0b101  -- 101_2
def b2 : ℕ := 0b11   -- 11_2
def b3 : ℕ := 0b1100 -- 1100_2
def b4 : ℕ := 0b11101 -- 11101_2
def sum_b : ℕ := 0b110001 -- 110001_2

theorem binary_addition :
  b1 + b2 + b3 + b4 = sum_b := 
by
  sorry

end binary_addition_l229_229942


namespace rectangle_width_length_ratio_l229_229210

theorem rectangle_width_length_ratio (w l P : ℕ) (h_l : l = 10) (h_P : P = 30) (h_perimeter : 2*w + 2*l = P) :
  w / l = 1 / 2 := 
by {
  sorry
}

end rectangle_width_length_ratio_l229_229210


namespace class_average_score_l229_229179

theorem class_average_score :
  let total_students := 40
  let absent_students := 2
  let present_students := total_students - absent_students
  let initial_avg := 92
  let absent_scores := [100, 100]
  let initial_total_score := initial_avg * present_students
  let total_final_score := initial_total_score + absent_scores.sum
  let final_avg := total_final_score / total_students
  final_avg = 92.4 := by
  sorry

end class_average_score_l229_229179


namespace students_taking_all_three_classes_l229_229479

variables (total_students Y B P N : ℕ)
variables (X₁ X₂ X₃ X₄ : ℕ)  -- variables representing students taking exactly two classes or all three

theorem students_taking_all_three_classes:
  total_students = 20 →
  Y = 10 →  -- Number of students taking yoga
  B = 13 →  -- Number of students taking bridge
  P = 9 →   -- Number of students taking painting
  N = 9 →   -- Number of students taking at least two classes
  X₂ + X₃ + X₄ = 9 →  -- This equation represents the total number of students taking at least two classes, where \( X₄ \) represents students taking all three (c).
  4 + X₃ + X₄ - (9 - X₃) + 1 + (9 - X₄ - X₂) + X₂ = 11 →
  X₄ = 3 :=                     -- Proving that the number of students taking all three classes is 3.
sorry

end students_taking_all_three_classes_l229_229479


namespace new_team_average_weight_l229_229488

theorem new_team_average_weight :
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  (new_total_weight / new_player_count) = 92 :=
by
  let original_team_weight := 7 * 94
  let new_players_weight := 110 + 60
  let new_total_weight := original_team_weight + new_players_weight
  let new_player_count := 9
  sorry

end new_team_average_weight_l229_229488


namespace sum_first_8_even_numbers_is_72_l229_229837

theorem sum_first_8_even_numbers_is_72 : (2 + 4 + 6 + 8 + 10 + 12 + 14 + 16) = 72 :=
by
  sorry

end sum_first_8_even_numbers_is_72_l229_229837


namespace sum_exponents_binary_3400_l229_229160

theorem sum_exponents_binary_3400 : 
  ∃ (a b c d e : ℕ), 
    3400 = 2^a + 2^b + 2^c + 2^d + 2^e ∧ 
    a > b ∧ b > c ∧ c > d ∧ d > e ∧ 
    a + b + c + d + e = 38 :=
sorry

end sum_exponents_binary_3400_l229_229160


namespace part_A_part_B_l229_229347

-- Definitions for the setup
variables (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0)

-- Part (A): Specific distance 5d
theorem part_A (d : ℝ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = 25 * d^2 ∧ |y - d| = 5 * d → 
  (x = 3 * d ∧ y = -4 * d) ∨ (x = -3 * d ∧ y = -4 * d)) :=
sorry

-- Part (B): General distance nd
theorem part_B (d : ℝ) (n : ℕ) (d_ne_0 : d ≠ 0) : 
  (∀ (x y : ℝ), x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d → ∃ x y, (x^2 + y^2 = (n * d)^2 ∧ |y - d| = n * d)) :=
sorry

end part_A_part_B_l229_229347


namespace area_of_inscribed_rectangle_l229_229878

variable (b h x : ℝ)

def is_isosceles_triangle (b h : ℝ) : Prop :=
  b > 0 ∧ h > 0

def is_inscribed_rectangle (b h x : ℝ) : Prop :=
  x > 0 ∧ x < h 

theorem area_of_inscribed_rectangle (h_pos : is_isosceles_triangle b h) 
                                    (rect_pos : is_inscribed_rectangle b h x) : 
                                    ∃ A : ℝ, A = (b / (2 * h)) * x ^ 2 :=
by
  sorry

end area_of_inscribed_rectangle_l229_229878


namespace mans_rate_in_still_water_l229_229223

theorem mans_rate_in_still_water (R S : ℝ) (h1 : R + S = 18) (h2 : R - S = 4) : R = 11 :=
by {
  sorry
}

end mans_rate_in_still_water_l229_229223


namespace weight_of_second_piece_l229_229327

-- Given conditions
def area (length : ℕ) (width : ℕ) : ℕ := length * width

def weight (density : ℚ) (area : ℕ) : ℚ := density * area

-- Given dimensions and weight of the first piece
def length1 : ℕ := 4
def width1 : ℕ := 3
def area1 : ℕ := area length1 width1
def weight1 : ℚ := 18

-- Given dimensions of the second piece
def length2 : ℕ := 6
def width2 : ℕ := 4
def area2 : ℕ := area length2 width2

-- Uniform density implies a proportional relationship between area and weight
def density1 : ℚ := weight1 / area1

-- The main theorem to prove
theorem weight_of_second_piece :
  weight density1 area2 = 36 :=
by
  -- use sorry to skip the proof
  sorry

end weight_of_second_piece_l229_229327


namespace min_tablets_to_ensure_three_each_l229_229660

theorem min_tablets_to_ensure_three_each (A B C : ℕ) (hA : A = 20) (hB : B = 25) (hC : C = 15) : 
  ∃ n, n = 48 ∧ (∀ x y z, x + y + z = n → x ≥ 3 ∧ y ≥ 3 ∧ z ≥ 3) :=
by
  -- proof goes here
  sorry

end min_tablets_to_ensure_three_each_l229_229660


namespace nina_basketball_cards_l229_229434

theorem nina_basketball_cards (cost_toy cost_shirt cost_card total_spent : ℕ) (n_toys n_shirts n_cards n_packs_result : ℕ)
  (h1 : cost_toy = 10)
  (h2 : cost_shirt = 6)
  (h3 : cost_card = 5)
  (h4 : n_toys = 3)
  (h5 : n_shirts = 5)
  (h6 : total_spent = 70)
  (h7 : n_packs_result =  2)
  : (3 * cost_toy + 5 * cost_shirt + n_cards * cost_card = total_spent) → n_cards = n_packs_result :=
by
  sorry

end nina_basketball_cards_l229_229434


namespace black_car_speed_l229_229620

theorem black_car_speed
  (red_speed black_speed : ℝ)
  (initial_distance time : ℝ)
  (red_speed_eq : red_speed = 10)
  (initial_distance_eq : initial_distance = 20)
  (time_eq : time = 0.5)
  (distance_eq : black_speed * time = initial_distance + red_speed * time) :
  black_speed = 50 := by
  rw [red_speed_eq, initial_distance_eq, time_eq] at distance_eq
  sorry

end black_car_speed_l229_229620


namespace h_of_j_of_3_l229_229014

def h (x : ℝ) : ℝ := 4 * x + 3
def j (x : ℝ) : ℝ := (x + 2) ^ 2

theorem h_of_j_of_3 : h (j 3) = 103 := by
  sorry

end h_of_j_of_3_l229_229014


namespace smallest_integer_y_l229_229370

theorem smallest_integer_y (y : ℤ) :
  (∃ y : ℤ, ((y / 4 : ℚ) + (3 / 7 : ℚ) > 2 / 3) ∧ (∀ z : ℤ, (z > 20 / 21) → y ≤ z)) :=
sorry

end smallest_integer_y_l229_229370


namespace equalities_imply_forth_l229_229303

variables {a b c d e f g h S1 S2 S3 O2 O3 : ℕ}

def S1_def := S1 = a + b + c
def S2_def := S2 = d + e + f
def S3_def := S3 = b + c + g + h - d
def O2_def := O2 = b + e + g
def O3_def := O3 = c + f + h

theorem equalities_imply_forth (h1 : S1 = S2) (h2 : S1 = S3) (h3 : S1 = O2) : S1 = O3 :=
  by sorry

end equalities_imply_forth_l229_229303


namespace range_of_a_l229_229331

-- Define the function f as given in the problem
def f (a x : ℝ) : ℝ := x^3 + a * x^2 + (a + 6) * x + 1

-- The mathematical statement to be proven in Lean
theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, ∃ m M : ℝ, m = (f a x) ∧ M = (f a y) ∧ (∀ z : ℝ, f a z ≥ m) ∧ (∀ z : ℝ, f a z ≤ M)) ↔ 
  (a < -3 ∨ a > 6) :=
sorry

end range_of_a_l229_229331


namespace graph_single_point_l229_229334

theorem graph_single_point (d : ℝ) :
  (∀ (x y : ℝ), 3 * x^2 + y^2 + 6 * x - 6 * y + d = 0 -> (x = -1 ∧ y = 3)) ↔ d = 12 :=
by 
  sorry

end graph_single_point_l229_229334


namespace min_ab_min_a_plus_b_l229_229917

theorem min_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : ab >= 8 :=
sorry

theorem min_a_plus_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) : a + b >= 3 + 2 * Real.sqrt 2 :=
sorry

end min_ab_min_a_plus_b_l229_229917


namespace part_I_part_II_l229_229251

def setA (x : ℝ) : Prop := 0 ≤ x - 1 ∧ x - 1 ≤ 2

def setB (x : ℝ) (a : ℝ) : Prop := 1 < x - a ∧ x - a < 2 * a + 3

def complement_R (x : ℝ) (a : ℝ) : Prop := x ≤ 2 ∨ x ≥ 6

theorem part_I (a : ℝ) (x : ℝ) (ha : a = 1) : 
  setA x ∨ setB x a ↔ (1 ≤ x ∧ x < 6) ∧ 
  (setA x ∧ complement_R x a ↔ 1 ≤ x ∧ x ≤ 2) := 
by
  sorry

theorem part_II (a : ℝ) : 
  (∃ x, setA x ∧ setB x a) ↔ -2/3 < a ∧ a < 2 := 
by
  sorry

end part_I_part_II_l229_229251


namespace prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l229_229215

noncomputable def prob_TeamA_wins_game : ℝ := 0.6
noncomputable def prob_TeamB_wins_game : ℝ := 0.4

-- Probability of Team A winning 2-1 in a best-of-three
noncomputable def prob_TeamA_wins_2_1 : ℝ := 2 * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game 

-- Probability of Team B winning in a best-of-three
noncomputable def prob_TeamB_wins_2_0 : ℝ := prob_TeamB_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins_2_1 : ℝ := 2 * prob_TeamB_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game
noncomputable def prob_TeamB_wins : ℝ := prob_TeamB_wins_2_0 + prob_TeamB_wins_2_1

-- Probability of Team A winning in a best-of-three
noncomputable def prob_TeamA_wins_best_of_three : ℝ := 1 - prob_TeamB_wins

-- Probability of Team A winning in a best-of-five
noncomputable def prob_TeamA_wins_3_0 : ℝ := prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamA_wins_game
noncomputable def prob_TeamA_wins_3_1 : ℝ := 3 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)
noncomputable def prob_TeamA_wins_3_2 : ℝ := 6 * (prob_TeamA_wins_game * prob_TeamA_wins_game * prob_TeamB_wins_game * prob_TeamB_wins_game * prob_TeamA_wins_game)

noncomputable def prob_TeamA_wins_best_of_five : ℝ := prob_TeamA_wins_3_0 + prob_TeamA_wins_3_1 + prob_TeamA_wins_3_2

theorem prob_TeamA_wins_2_1_proof :
  prob_TeamA_wins_2_1 = 0.288 :=
sorry

theorem prob_TeamB_wins_proof :
  prob_TeamB_wins = 0.352 :=
sorry

theorem best_of_five_increases_prob :
  prob_TeamA_wins_best_of_three < prob_TeamA_wins_best_of_five :=
sorry

end prob_TeamA_wins_2_1_proof_prob_TeamB_wins_proof_best_of_five_increases_prob_l229_229215


namespace abigail_lost_money_l229_229613

theorem abigail_lost_money (initial_amount spent_first_store spent_second_store remaining_amount_lost: ℝ) 
  (h_initial : initial_amount = 50) 
  (h_spent_first : spent_first_store = 15.25) 
  (h_spent_second : spent_second_store = 8.75) 
  (h_remaining : remaining_amount_lost = 16) : (initial_amount - spent_first_store - spent_second_store - remaining_amount_lost = 10) :=
by
  sorry

end abigail_lost_money_l229_229613


namespace ratio_lcm_gcf_l229_229004

theorem ratio_lcm_gcf (a b : ℕ) (h₁ : a = 2^2 * 3^2 * 7) (h₂ : b = 2 * 3^2 * 5 * 7) :
  (Nat.lcm a b) / (Nat.gcd a b) = 10 := by
  sorry

end ratio_lcm_gcf_l229_229004


namespace amount_first_set_correct_l229_229908

-- Define the amounts as constants
def total_amount : ℝ := 900.00
def amount_second_set : ℝ := 260.00
def amount_third_set : ℝ := 315.00

-- Define the amount given to the first set
def amount_first_set : ℝ :=
  total_amount - amount_second_set - amount_third_set

-- Statement: prove that the amount given to the first set of families equals $325.00
theorem amount_first_set_correct :
  amount_first_set = 325.00 :=
sorry

end amount_first_set_correct_l229_229908


namespace probability_of_successful_meeting_l229_229505

noncomputable def successful_meeting_probability : ℝ :=
  let volume_hypercube := 16.0
  let volume_pyramid := (1.0/3.0) * 2.0^3 * 2.0
  let volume_reduced_base := volume_pyramid / 4.0
  let successful_meeting_volume := volume_reduced_base
  successful_meeting_volume / volume_hypercube

theorem probability_of_successful_meeting : successful_meeting_probability = 1 / 12 :=
  sorry

end probability_of_successful_meeting_l229_229505


namespace sugar_cups_used_l229_229035

def ratio_sugar_water : ℕ × ℕ := (1, 2)
def total_cups : ℕ := 84

theorem sugar_cups_used (r : ℕ × ℕ) (tc : ℕ) (hsugar : r.1 = 1) (hwater : r.2 = 2) (htotal : tc = 84) :
  (tc * r.1) / (r.1 + r.2) = 28 :=
by
  sorry

end sugar_cups_used_l229_229035


namespace sum_expression_l229_229558

theorem sum_expression : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end sum_expression_l229_229558


namespace goods_train_speed_l229_229889

theorem goods_train_speed 
  (length_train : ℕ)
  (length_platform : ℕ)
  (time_to_cross : ℕ)
  (h_train : length_train = 270)
  (h_platform : length_platform = 250)
  (h_time : time_to_cross = 26) : 
  (length_train + length_platform) / time_to_cross = 20 := 
by
  sorry

end goods_train_speed_l229_229889


namespace find_range_a_l229_229378

-- Define the proposition p
def p (m : ℝ) : Prop :=
1 < m ∧ m < 3 / 2

-- Define the proposition q
def q (m a : ℝ) : Prop :=
(m - a) * (m - (a + 1)) < 0

-- Define the sufficient but not necessary condition
def sufficient (a : ℝ) : Prop :=
(a ≤ 1) ∧ (3 / 2 ≤ a + 1)

theorem find_range_a (a : ℝ) :
  (∀ m, p m → q m a) → sufficient a → (1 / 2 ≤ a ∧ a ≤ 1) :=
sorry

end find_range_a_l229_229378


namespace contrapositive_inequality_l229_229140

theorem contrapositive_inequality (a b : ℝ) :
  (a > b → a - 5 > b - 5) ↔ (a - 5 ≤ b - 5 → a ≤ b) := by
sorry

end contrapositive_inequality_l229_229140


namespace pentagon_ABEDF_area_l229_229175

theorem pentagon_ABEDF_area (BD_diagonal : ∀ (ABCD : Nat) (BD : Nat),
                            ABCD = BD^2 / 2 → BD = 20) 
                            (BDFE_is_rectangle : ∀ (BDFE : Nat), BDFE = 2 * BD) 
                            : ∃ (area : Nat), area = 300 :=
by
  -- Placeholder for the actual proof
  sorry

end pentagon_ABEDF_area_l229_229175


namespace inverse_proportion_function_neg_k_l229_229947

variable {k : ℝ}
variable {y1 y2 : ℝ}

theorem inverse_proportion_function_neg_k
  (h1 : k ≠ 0)
  (h2 : y1 > y2)
  (hA : y1 = k / (-2))
  (hB : y2 = k / 5) :
  k < 0 :=
sorry

end inverse_proportion_function_neg_k_l229_229947


namespace least_positive_three_digit_multiple_of_7_l229_229548

theorem least_positive_three_digit_multiple_of_7 : ∃ n : ℕ, n % 7 = 0 ∧ n ≥ 100 ∧ n < 1000 ∧ ∀ m : ℕ, (m % 7 = 0 ∧ m ≥ 100 ∧ m < 1000) → n ≤ m := 
by
  sorry

end least_positive_three_digit_multiple_of_7_l229_229548


namespace compare_f_neg_x1_neg_x2_l229_229955

noncomputable def f : ℝ → ℝ := sorry

theorem compare_f_neg_x1_neg_x2 
  (h1 : ∀ x : ℝ, f (1 + x) = f (1 - x)) 
  (h2 : ∀ x y : ℝ, 1 ≤ x → 1 ≤ y → x < y → f x < f y)
  (x1 x2 : ℝ)
  (hx1 : x1 < 0)
  (hx2 : x2 > 0)
  (hx1x2 : x1 + x2 < -2) :
  f (-x1) > f (-x2) :=
by sorry

end compare_f_neg_x1_neg_x2_l229_229955


namespace cost_of_remaining_ingredients_l229_229152

theorem cost_of_remaining_ingredients :
  let cocoa_required := 0.4
  let sugar_required := 0.6
  let cake_weight := 450
  let given_cocoa := 259
  let cost_per_lb_cocoa := 3.50
  let cost_per_lb_sugar := 0.80
  let total_cocoa_needed := cake_weight * cocoa_required
  let total_sugar_needed := cake_weight * sugar_required
  let remaining_cocoa := max 0 (total_cocoa_needed - given_cocoa)
  let remaining_sugar := total_sugar_needed
  let total_cost := remaining_cocoa * cost_per_lb_cocoa + remaining_sugar * cost_per_lb_sugar
  total_cost = 216 := by
  sorry

end cost_of_remaining_ingredients_l229_229152


namespace find_k_l229_229787

noncomputable def vec_a : ℝ × ℝ := (1, 2)
noncomputable def vec_b : ℝ × ℝ := (-3, 2)
noncomputable def vec_k_a_plus_b (k : ℝ) : ℝ × ℝ := (k - 3, 2 * k + 2)
noncomputable def vec_a_minus_3b : ℝ × ℝ := (10, -4)

theorem find_k :
  ∃! k : ℝ, (vec_k_a_plus_b k).1 * vec_a_minus_3b.2 = (vec_k_a_plus_b k).2 * vec_a_minus_3b.1 ∧ k = -1 / 3 :=
by
  sorry

end find_k_l229_229787


namespace simplify_expression_l229_229247

theorem simplify_expression : -Real.sqrt 4 + abs (Real.sqrt 2 - 2) - 2023^0 = -2 := 
by 
  sorry

end simplify_expression_l229_229247


namespace a1_a9_sum_l229_229570

noncomputable def arithmetic_sequence (a: ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a1_a9_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3_a7_roots : (a 3 = 3 ∧ a 7 = -1) ∨ (a 3 = -1 ∧ a 7 = 3)) :
  a 1 + a 9 = 2 :=
by
  sorry

end a1_a9_sum_l229_229570


namespace central_angle_of_sector_l229_229672

theorem central_angle_of_sector (r l : ℝ) (h1 : r = 1) (h2 : l = 4 - 2*r) : 
    ∃ α : ℝ, α = 2 :=
by
  use l / r
  have hr : r = 1 := h1
  have hl : l = 4 - 2*r := h2
  sorry

end central_angle_of_sector_l229_229672


namespace selling_price_of_book_l229_229055

theorem selling_price_of_book (cost_price : ℝ) (profit_percentage : ℝ) (profit : ℝ) (selling_price : ℝ) 
  (h₁ : cost_price = 60) 
  (h₂ : profit_percentage = 25) 
  (h₃ : profit = (profit_percentage / 100) * cost_price) 
  (h₄ : selling_price = cost_price + profit) : 
  selling_price = 75 := 
by
  sorry

end selling_price_of_book_l229_229055


namespace min_disks_to_store_files_l229_229624

open Nat

theorem min_disks_to_store_files :
  ∃ minimum_disks : ℕ,
    (minimum_disks = 24) ∧
    ∀ (files : ℕ) (disk_capacity : ℕ) (file_sizes : List ℕ),
      files = 36 →
      disk_capacity = 144 →
      (∃ (size_85 : ℕ) (size_75 : ℕ) (size_45 : ℕ),
         size_85 = 5 ∧
         size_75 = 15 ∧
         size_45 = 16 ∧
         (∀ (disks : ℕ), disks >= minimum_disks →
            ∃ (used_disks_85 : ℕ) (remaining_files_45 : ℕ) (used_disks_45 : ℕ) (used_disks_75 : ℕ),
              remaining_files_45 = size_45 - used_disks_85 ∧
              used_disks_85 = size_85 ∧
              (remaining_files_45 % 3 = 0 → used_disks_45 = remaining_files_45 / 3) ∧
              (remaining_files_45 % 3 ≠ 0 → used_disks_45 = remaining_files_45 / 3 + 1) ∧
              used_disks_75 = size_75 ∧
              disks = used_disks_85 + used_disks_45 + used_disks_75)) :=
by
  sorry

end min_disks_to_store_files_l229_229624


namespace gcd_779_209_589_l229_229682

theorem gcd_779_209_589 : Int.gcd (Int.gcd 779 209) 589 = 19 := 
by 
  sorry

end gcd_779_209_589_l229_229682


namespace perfect_square_adjacent_smaller_l229_229065

noncomputable def is_perfect_square (n : ℕ) : Prop := 
    ∃ k : ℕ, k * k = n

theorem perfect_square_adjacent_smaller (m : ℕ) (hm : is_perfect_square m) : 
    ∃ k : ℕ, (k * k = m ∧ (k - 1) * (k - 1) = m - 2 * k + 1) := 
by 
  sorry

end perfect_square_adjacent_smaller_l229_229065


namespace find_a_l229_229936

theorem find_a (x : ℝ) (n : ℕ) (hx : x > 0) (hn : n > 0) :
  x + n^n * (1 / (x^n)) ≥ n + 1 :=
sorry

end find_a_l229_229936


namespace incorrect_statement_l229_229719

theorem incorrect_statement : 
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → (¬p ∧ ¬q)) := 
    sorry

end incorrect_statement_l229_229719


namespace inequality_proof_l229_229833

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^2 / c) + (c^2 / a) ≥ a + b + c + 4 * (a - b)^2 / (a + b + c) :=
by
  sorry

end inequality_proof_l229_229833


namespace some_value_correct_l229_229064

theorem some_value_correct (w x y : ℝ) (some_value : ℝ)
  (h1 : 3 / w + some_value = 3 / y)
  (h2 : w * x = y)
  (h3 : (w + x) / 2 = 0.5) :
  some_value = 6 := by
  sorry

end some_value_correct_l229_229064


namespace distance_from_apex_to_larger_cross_section_l229_229275

noncomputable def area1 : ℝ := 324 * Real.sqrt 2
noncomputable def area2 : ℝ := 648 * Real.sqrt 2
def distance_between_planes : ℝ := 12

theorem distance_from_apex_to_larger_cross_section
  (area1 area2 : ℝ)
  (distance_between_planes : ℝ)
  (h_area1 : area1 = 324 * Real.sqrt 2)
  (h_area2 : area2 = 648 * Real.sqrt 2)
  (h_distance : distance_between_planes = 12) :
  ∃ (H : ℝ), H = 24 + 12 * Real.sqrt 2 :=
by sorry

end distance_from_apex_to_larger_cross_section_l229_229275


namespace profit_percentage_is_25_l229_229439

theorem profit_percentage_is_25 
  (selling_price : ℝ) (cost_price : ℝ) 
  (sp_val : selling_price = 600) 
  (cp_val : cost_price = 480) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_is_25_l229_229439


namespace janice_walk_dog_more_than_homework_l229_229090

theorem janice_walk_dog_more_than_homework 
  (H C T: Nat) 
  (W: Nat) 
  (total_time remaining_time spent_time: Nat) 
  (hw_time room_time trash_time extra_time: Nat)
  (H_eq : H = 30)
  (C_eq : C = H / 2)
  (T_eq : T = H / 6)
  (remaining_time_eq : remaining_time = 35)
  (total_time_eq : total_time = 120)
  (spent_time_eq : spent_time = total_time - remaining_time)
  (task_time_sum_eq : task_time_sum = H + C + T)
  (W_eq : W = spent_time - task_time_sum)
  : W - H = 5 := 
sorry

end janice_walk_dog_more_than_homework_l229_229090


namespace proportion_margin_l229_229259

theorem proportion_margin (S M C : ℝ) (n : ℝ) (hM : M = S / n) (hC : C = (1 - 1 / n) * S) :
  M / C = 1 / (n - 1) :=
by
  sorry

end proportion_margin_l229_229259


namespace cylindrical_to_rectangular_l229_229432

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 5 → θ = (3 * Real.pi) / 4 → z = 2 →
    (r * Real.cos θ, r * Real.sin θ, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  -- Proof steps would go here, but are omitted as they are not required.
  sorry

end cylindrical_to_rectangular_l229_229432


namespace even_n_condition_l229_229006

theorem even_n_condition (x : ℝ) (n : ℕ) (h : ∀ x, 3 * x^n + n * (x + 2) - 3 ≥ n * x^2) : n % 2 = 0 :=
sorry

end even_n_condition_l229_229006


namespace fill_grid_power_of_two_l229_229146

theorem fill_grid_power_of_two (n : ℕ) (h : ∃ m : ℕ, n = 2^m) :
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j : ℕ, i < n → j < n → 1 ≤ f i j ∧ f i j ≤ 2 * n - 1) ∧
    (∀ k, 1 ≤ k ∧ k ≤ n → (∀ i, i < n → ∀ j, j < n → i ≠ j → f i k ≠ f j k))
:= by
  sorry

end fill_grid_power_of_two_l229_229146


namespace length_of_segment_AB_l229_229721

variables (h : ℝ) (AB CD : ℝ)

-- Defining the conditions
def condition_one : Prop := (AB / CD = 5 / 2)
def condition_two : Prop := (AB + CD = 280)

-- The theorem to prove
theorem length_of_segment_AB (h : ℝ) (AB CD : ℝ) :
  condition_one AB CD ∧ condition_two AB CD → AB = 200 :=
by
  sorry

end length_of_segment_AB_l229_229721


namespace train_crosses_bridge_in_time_l229_229851

noncomputable def length_of_train : ℝ := 125
noncomputable def length_of_bridge : ℝ := 250.03
noncomputable def speed_of_train_kmh : ℝ := 45

noncomputable def speed_of_train_ms : ℝ := (speed_of_train_kmh * 1000) / 3600
noncomputable def total_distance : ℝ := length_of_train + length_of_bridge
noncomputable def time_to_cross_bridge : ℝ := total_distance / speed_of_train_ms

theorem train_crosses_bridge_in_time :
  time_to_cross_bridge = 30.0024 :=
  sorry

end train_crosses_bridge_in_time_l229_229851


namespace golf_problem_l229_229382

variable (D : ℝ)

theorem golf_problem (h1 : D / 2 + D = 270) : D = 180 :=
by
  sorry

end golf_problem_l229_229382


namespace determine_d_l229_229137

theorem determine_d (u v d c : ℝ) (p q : ℝ → ℝ)
  (hp : ∀ x, p x = x^3 + c * x + d)
  (hq : ∀ x, q x = x^3 + c * x + d + 300)
  (huv : p u = 0 ∧ p v = 0)  
  (hu5_v4 : q (u + 5) = 0 ∧ q (v - 4) = 0)
  (sum_roots_p : u + v + (-u - v) = 0)
  (sum_roots_q : (u + 5) + (v - 4) + (-u - v - 1) = 0)
  : d = -4 ∨ d = 6 :=
sorry

end determine_d_l229_229137


namespace inequality_holds_l229_229888

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def even_function : Prop := ∀ x : ℝ, f x = f (-x)
def decreasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f y ≤ f x

-- Proof goal
theorem inequality_holds (h_even : even_function f) (h_decreasing : decreasing_on_pos f) : 
  f (-3/4) ≥ f (a^2 - a + 1) := 
by
  sorry

end inequality_holds_l229_229888


namespace solve_for_n_l229_229359

theorem solve_for_n :
  ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 :=
by
  sorry

end solve_for_n_l229_229359


namespace trigonometric_identity_proof_l229_229718

noncomputable def m : ℝ := 2 * Real.sin (Real.pi / 10)
noncomputable def n : ℝ := 4 - m^2

theorem trigonometric_identity_proof :
  (m = 2 * Real.sin (Real.pi / 10)) →
  (m^2 + n = 4) →
  (m * Real.sqrt n) / (2 * Real.cos (3 * Real.pi / 20)^2 - 1) = 2 :=
by
  intros h1 h2
  sorry

end trigonometric_identity_proof_l229_229718


namespace arithmetic_sequence_a3_l229_229068

variable {a : ℕ → ℝ}  -- Define the sequence as a function from natural numbers to real numbers.

-- Definition that the sequence is arithmetic.
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- The given condition in the problem
axiom h1 : a 1 + a 5 = 6

-- The statement to prove
theorem arithmetic_sequence_a3 (h : is_arithmetic_sequence a) : a 3 = 3 :=
by {
  -- The proof is omitted.
  sorry
}

end arithmetic_sequence_a3_l229_229068


namespace value_decrease_proof_l229_229385

noncomputable def value_comparison (diana_usd : ℝ) (etienne_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let etienne_usd := etienne_eur * eur_to_usd
  let percentage_decrease := ((diana_usd - etienne_usd) / diana_usd) * 100
  percentage_decrease

theorem value_decrease_proof :
  value_comparison 700 300 1.5 = 35.71 :=
by
  sorry

end value_decrease_proof_l229_229385


namespace river_depth_mid_may_l229_229071

-- Definitions corresponding to the conditions
def depth_mid_june (D : ℕ) : ℕ := D + 10
def depth_mid_july (D : ℕ) : ℕ := 3 * (depth_mid_june D)

-- The theorem statement
theorem river_depth_mid_may (D : ℕ) (h : depth_mid_july D = 45) : D = 5 :=
by
  sorry

end river_depth_mid_may_l229_229071


namespace solve_for_square_solve_for_cube_l229_229132

variable (x : ℂ)

-- Given condition
def condition := x + 1/x = 8

-- Prove that x^2 + 1/x^2 = 62 given the condition
theorem solve_for_square (h : condition x) : x^2 + 1/x^2 = 62 := 
  sorry

-- Prove that x^3 + 1/x^3 = 488 given the condition
theorem solve_for_cube (h : condition x) : x^3 + 1/x^3 = 488 :=
  sorry

end solve_for_square_solve_for_cube_l229_229132


namespace trueConverseB_l229_229989

noncomputable def conditionA : Prop :=
  ∀ (x y : ℝ), -- "Vertical angles are equal"
  sorry -- Placeholder for vertical angles equality

noncomputable def conditionB : Prop :=
  ∀ (l₁ l₂ : ℝ), -- "If the consecutive interior angles are supplementary, then the two lines are parallel."
  sorry -- Placeholder for supplementary angles imply parallel lines

noncomputable def conditionC : Prop :=
  ∀ (a b : ℝ), -- "If \(a = b\), then \(a^2 = b^2\)"
  a = b → a^2 = b^2

noncomputable def conditionD : Prop :=
  ∀ (a b : ℝ), -- "If \(a > 0\) and \(b > 0\), then \(a^2 + b^2 > 0\)"
  a > 0 ∧ b > 0 → a^2 + b^2 > 0

theorem trueConverseB (hB: conditionB) : -- Proposition (B) has a true converse
  ∀ (l₁ l₂ : ℝ), 
  (∃ (a1 a2 : ℝ), -- Placeholder for angles
  sorry) → (l₁ = l₂) := -- Placeholder for consecutive interior angles are supplementary
  sorry

end trueConverseB_l229_229989


namespace seq_a_seq_b_l229_229598

theorem seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) (n : ℕ) :
  a 1 = 1 ∧ (∀ n, S (n + 1) = 3 * S n + 2) →
  (∀ n, a n = if n = 1 then 1 else 4 * 3 ^ (n - 2)) :=
by
  sorry

theorem seq_b (b : ℕ → ℕ) (a : ℕ → ℕ) (T : ℕ → ℕ) (n : ℕ) :
  (b n = 8 * n / (a (n + 1) - a n)) →
  (T n = 77 / 12 - (n / 2 + 3 / 4) * (1 / 3) ^ (n - 2)) :=
by
  sorry

end seq_a_seq_b_l229_229598


namespace remaining_tickets_equation_l229_229629

-- Define the constants and variables
variables (x y : ℕ)

-- Conditions from the problem
def tickets_whack_a_mole := 32
def tickets_skee_ball := 25
def tickets_space_invaders : ℕ := x

def spent_hat := 7
def spent_keychain := 10
def spent_toy := 15

-- Define the condition for the total number of tickets spent
def total_tickets_spent := spent_hat + spent_keychain + spent_toy
-- Prove the remaining tickets equation
theorem remaining_tickets_equation : y = (tickets_whack_a_mole + tickets_skee_ball + tickets_space_invaders) - total_tickets_spent ->
                                      y = 25 + x :=
by
  sorry

end remaining_tickets_equation_l229_229629


namespace arithmetic_seq_sum_l229_229898

theorem arithmetic_seq_sum (a : ℕ → ℤ) (S : ℤ → ℤ) 
  (h1 : ∀ n, a n = a 0 + n * (a 1 - a 0)) 
  (h2 : a 4 + a 6 + a 8 + a 10 + a 12 = 110) : 
  S 15 = 330 := 
by
  sorry

end arithmetic_seq_sum_l229_229898


namespace fraction_defined_range_l229_229612

theorem fraction_defined_range (x : ℝ) : 
  (∃ y, y = 3 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end fraction_defined_range_l229_229612


namespace probability_of_females_right_of_males_l229_229030

-- Defining the total and favorable outcomes
def total_outcomes : ℕ := Nat.factorial 5
def favorable_outcomes : ℕ := Nat.factorial 3 * Nat.factorial 2

-- Defining the probability as a rational number
def probability_all_females_right : ℚ := favorable_outcomes / total_outcomes

-- Stating the theorem
theorem probability_of_females_right_of_males :
  probability_all_females_right = 1 / 10 :=
by
  -- Proof to be filled in
  sorry

end probability_of_females_right_of_males_l229_229030


namespace scientific_notation_five_hundred_billion_l229_229757

theorem scientific_notation_five_hundred_billion :
  500000000000 = 5 * 10^11 := by
  sorry

end scientific_notation_five_hundred_billion_l229_229757


namespace households_3_houses_proportion_l229_229128

noncomputable def total_households : ℕ := 100000
noncomputable def ordinary_households : ℕ := 99000
noncomputable def high_income_households : ℕ := 1000

noncomputable def sampled_ordinary_households : ℕ := 990
noncomputable def sampled_high_income_households : ℕ := 100

noncomputable def sampled_ordinary_3_houses : ℕ := 40
noncomputable def sampled_high_income_3_houses : ℕ := 80

noncomputable def proportion_3_houses : ℝ := (sampled_ordinary_3_houses / sampled_ordinary_households * ordinary_households + sampled_high_income_3_houses / sampled_high_income_households * high_income_households) / total_households

theorem households_3_houses_proportion : proportion_3_houses = 0.048 := 
by
  sorry

end households_3_houses_proportion_l229_229128


namespace interval_length_of_solutions_l229_229730

theorem interval_length_of_solutions (a b : ℝ) :
  (∃ x : ℝ, a ≤ 3*x + 6 ∧ 3*x + 6 ≤ b) ∧ (∃ (l : ℝ), l = (b - a) / 3 ∧ l = 15) → b - a = 45 :=
by sorry

end interval_length_of_solutions_l229_229730


namespace range_of_t_l229_229353

theorem range_of_t (t : ℝ) : 
  (∀ x : ℝ, x ≥ 1 → (x^2 + 2*x + t) / x > 0) ↔ t > -3 := 
by
  sorry

end range_of_t_l229_229353


namespace time_saved_calculator_l229_229759

-- Define the conditions
def time_with_calculator (n : ℕ) : ℕ := 2 * n
def time_without_calculator (n : ℕ) : ℕ := 5 * n
def total_problems : ℕ := 20

-- State the theorem to prove the time saved is 60 minutes
theorem time_saved_calculator : 
  time_without_calculator total_problems - time_with_calculator total_problems = 60 :=
sorry

end time_saved_calculator_l229_229759


namespace number_of_valid_m_values_l229_229999

/--
In the coordinate plane, construct a right triangle with its legs parallel to the x and y axes, and with the medians on its legs lying on the lines y = 3x + 1 and y = mx + 2. 
Prove that the number of values for the constant m such that this triangle exists is 2.
-/
theorem number_of_valid_m_values : 
  ∃ (m : ℝ), 
    (∃ (a b : ℝ), 
      (∀ D E : ℝ × ℝ, D = (a / 2, 0) ∧ E = (0, b / 2) →
      D.2 = 3 * D.1 + 1 ∧ 
      E.2 = m * E.1 + 2)) → 
    (number_of_solutions_for_m = 2) 
  :=
sorry

end number_of_valid_m_values_l229_229999


namespace number_one_seventh_equals_five_l229_229333

theorem number_one_seventh_equals_five (n : ℕ) (h : n / 7 = 5) : n = 35 :=
sorry

end number_one_seventh_equals_five_l229_229333


namespace sum_of_undefined_fractions_l229_229208

theorem sum_of_undefined_fractions (x₁ x₂ : ℝ) (h₁ : x₁^2 - 7*x₁ + 12 = 0) (h₂ : x₂^2 - 7*x₂ + 12 = 0) :
  x₁ + x₂ = 7 :=
sorry

end sum_of_undefined_fractions_l229_229208


namespace platform_length_is_350_l229_229141

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l229_229141


namespace total_clothes_washed_l229_229768

def number_of_clothing_items (Cally Danny Emily shared_socks : ℕ) : ℕ :=
  Cally + Danny + Emily + shared_socks

theorem total_clothes_washed :
  let Cally_clothes := (10 + 5 + 7 + 6 + 3)
  let Danny_clothes := (6 + 8 + 10 + 6 + 4)
  let Emily_clothes := (8 + 6 + 9 + 5 + 2)
  let shared_socks := (3 + 2)
  number_of_clothing_items Cally_clothes Danny_clothes Emily_clothes shared_socks = 100 :=
by
  sorry

end total_clothes_washed_l229_229768


namespace rectangular_field_area_l229_229729

-- Given a rectangle with one side 4 meters and diagonal 5 meters, prove that its area is 12 square meters.
theorem rectangular_field_area
  (w l d : ℝ)
  (h_w : w = 4)
  (h_d : d = 5)
  (h_pythagoras : w^2 + l^2 = d^2) :
  w * l = 12 := 
by
  sorry

end rectangular_field_area_l229_229729


namespace cos_decreasing_intervals_l229_229644

open Real

def is_cos_decreasing_interval (k : ℤ) : Prop := 
  let f (x : ℝ) := cos (π / 4 - 2 * x)
  ∀ x y : ℝ, (k * π + π / 8 ≤ x) → (x ≤ k * π + 5 * π / 8) → 
             (k * π + π / 8 ≤ y) → (y ≤ k * π + 5 * π / 8) → 
             x < y → f x > f y

theorem cos_decreasing_intervals : ∀ k : ℤ, is_cos_decreasing_interval k :=
by
  sorry

end cos_decreasing_intervals_l229_229644


namespace increasing_inverse_relation_l229_229283

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry -- This is the inverse function f^-1

theorem increasing_inverse_relation {a b c : ℝ} 
  (h_inc_f : ∀ x y, x < y → f x < f y)
  (h_inc_f_inv : ∀ x y, x < y → f_inv x < f_inv y)
  (h_f3 : f 3 = 0)
  (h_f2 : f 2 = a)
  (h_f_inv2 : f_inv 2 = b)
  (h_f_inv0 : f_inv 0 = c) :
  b > c ∧ c > a := sorry

end increasing_inverse_relation_l229_229283


namespace lamplighter_monkey_distance_traveled_l229_229542

-- Define the parameters
def running_speed : ℕ := 15
def running_time : ℕ := 5
def swinging_speed : ℕ := 10
def swinging_time : ℕ := 10

-- Define the proof statement
theorem lamplighter_monkey_distance_traveled :
  (running_speed * running_time) + (swinging_speed * swinging_time) = 175 := by
  sorry

end lamplighter_monkey_distance_traveled_l229_229542


namespace third_neigh_uses_100_more_l229_229693

def total_water : Nat := 1200
def first_neigh_usage : Nat := 150
def second_neigh_usage : Nat := 2 * first_neigh_usage
def fourth_neigh_remaining : Nat := 350

def third_neigh_usage := total_water - (first_neigh_usage + second_neigh_usage + fourth_neigh_remaining)
def diff_third_second := third_neigh_usage - second_neigh_usage

theorem third_neigh_uses_100_more :
  diff_third_second = 100 := by
  sorry

end third_neigh_uses_100_more_l229_229693


namespace polynomial_divisibility_l229_229529

theorem polynomial_divisibility (a : ℤ) : 
  (∀x : ℤ, x^2 - x + a ∣ x^13 + x + 94) → a = 2 := 
by 
  sorry

end polynomial_divisibility_l229_229529


namespace proof_problem_l229_229996

-- Define the sum of the first n terms of the arithmetic sequence
def S (n : ℕ) : ℕ := (n^2 + n) / 2

-- Define the arithmetic sequence a_n based on S_n
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Define the geometric sequence b_n with initial conditions
def b (n : ℕ) : ℕ :=
  if n = 1 then a 1 + 1
  else if n = 2 then a 2 + 2
  else 2^n

-- Define the sum of the first n terms of the geometric sequence b_n
def T (n : ℕ) : ℕ := 2 * (2^n - 1)

-- Main theorem to prove
theorem proof_problem :
  (∀ n, a n = n) ∧
  (∀ n, n ≥ 1 → b n = 2^n) ∧
  (∃ n, T n + a n > 300 ∧ ∀ m < n, T m + a m ≤ 300) :=
by {
  sorry
}

end proof_problem_l229_229996


namespace kevin_hopping_distance_l229_229772

theorem kevin_hopping_distance :
  let hop_distance (n : Nat) : ℚ :=
    let factor : ℚ := (3/4 : ℚ)^n
    1/4 * factor
  let total_distance : ℚ :=
    (hop_distance 0 + hop_distance 1 + hop_distance 2 + hop_distance 3 + hop_distance 4 + hop_distance 5)
  total_distance = 39677 / 40960 :=
by
  sorry

end kevin_hopping_distance_l229_229772


namespace system_of_equations_solution_l229_229367

theorem system_of_equations_solution (x y z : ℕ) :
  x + y + z = 6 ∧ xy + yz + zx = 11 ∧ xyz = 6 ↔
  (x, y, z) = (1, 2, 3) ∨ (x, y, z) = (1, 3, 2) ∨ 
  (x, y, z) = (2, 1, 3) ∨ (x, y, z) = (2, 3, 1) ∨ 
  (x, y, z) = (3, 1, 2) ∨ (x, y, z) = (3, 2, 1) := by
  sorry

end system_of_equations_solution_l229_229367


namespace tan_product_l229_229770

theorem tan_product :
    (Real.tan (Real.pi / 9)) * (Real.tan (2 * Real.pi / 9)) * (Real.tan (4 * Real.pi / 9)) = 1 / 3 :=
by sorry

end tan_product_l229_229770


namespace kaleb_clothing_problem_l229_229839

theorem kaleb_clothing_problem 
  (initial_clothing : ℕ) 
  (one_load : ℕ) 
  (remaining_loads : ℕ) : 
  initial_clothing = 39 → one_load = 19 → remaining_loads = 5 → (initial_clothing - one_load) / remaining_loads = 4 :=
sorry

end kaleb_clothing_problem_l229_229839


namespace min_button_presses_l229_229157

theorem min_button_presses :
  ∃ (a b : ℤ), 9 * a - 20 * b = 13 ∧  a + b = 24 := 
by
  sorry

end min_button_presses_l229_229157


namespace combined_profit_is_14000_l229_229763

-- Define constants and conditions
def center1_daily_packages : ℕ := 10000
def daily_profit_per_package : ℝ := 0.05
def center2_multiplier : ℕ := 3
def days_per_week : ℕ := 7

-- Define the profit for the first center
def center1_daily_profit : ℝ := center1_daily_packages * daily_profit_per_package

-- Define the packages processed by the second center
def center2_daily_packages : ℕ := center1_daily_packages * center2_multiplier

-- Define the profit for the second center
def center2_daily_profit : ℝ := center2_daily_packages * daily_profit_per_package

-- Define the combined daily profit
def combined_daily_profit : ℝ := center1_daily_profit + center2_daily_profit

-- Define the combined weekly profit
def combined_weekly_profit : ℝ := combined_daily_profit * days_per_week

-- Prove that the combined weekly profit is $14,000
theorem combined_profit_is_14000 : combined_weekly_profit = 14000 := by
  -- You can replace sorry with the steps to solve the proof.
  sorry

end combined_profit_is_14000_l229_229763


namespace diagonals_in_polygon_with_150_sides_l229_229105

-- (a) Definitions for conditions
def sides : ℕ := 150

def diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- (c) Statement of the problem in Lean 4
theorem diagonals_in_polygon_with_150_sides :
  diagonals sides = 11025 :=
by
  sorry

end diagonals_in_polygon_with_150_sides_l229_229105


namespace work_completion_l229_229151

theorem work_completion (a b : ℝ) 
  (h1 : a + b = 6) 
  (h2 : a = 10) : 
  a + b = 6 :=
by sorry

end work_completion_l229_229151


namespace smallest_element_in_M_l229_229854

def f : ℝ → ℝ := sorry
axiom f1 (x y : ℝ) (h1 : x ≥ 1) (h2 : y = 3 * x) : f y = 3 * f x
axiom f2 (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : f x = 1 - abs (x - 2)
axiom f99_value : f 99 = 18

theorem smallest_element_in_M : ∃ x : ℝ, x = 45 ∧ f x = 18 := by
  -- proof will be provided later
  sorry

end smallest_element_in_M_l229_229854


namespace sales_not_books_magazines_stationery_l229_229906

variable (books_sales : ℕ := 45)
variable (magazines_sales : ℕ := 30)
variable (stationery_sales : ℕ := 10)
variable (total_sales : ℕ := 100)

theorem sales_not_books_magazines_stationery : 
  books_sales + magazines_sales + stationery_sales < total_sales → 
  total_sales - (books_sales + magazines_sales + stationery_sales) = 15 :=
by
  sorry

end sales_not_books_magazines_stationery_l229_229906


namespace find_value_of_A_l229_229202

theorem find_value_of_A (A ω φ c : ℝ)
  (a : ℕ+ → ℝ)
  (h_seq : ∀ n : ℕ+, a n * a (n + 1) * a (n + 2) = a n + a (n + 1) + a (n + 2))
  (h_neq : ∀ n : ℕ+, a n * a (n + 1) ≠ 1)
  (h_a1 : a 1 = 1)
  (h_a2 : a 2 = 2)
  (h_form : ∀ n : ℕ+, a n = A * Real.sin (ω * n + φ) + c)
  (h_ω_gt_0 : ω > 0)
  (h_phi_lt_pi_div_2 : |φ| < Real.pi / 2) :
  A = -2 * Real.sqrt 3 / 3 := 
sorry

end find_value_of_A_l229_229202


namespace jacket_initial_reduction_l229_229389

theorem jacket_initial_reduction (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.9 * 1.481481481481481 = P → x = 25 :=
by
  sorry

end jacket_initial_reduction_l229_229389


namespace dan_marbles_l229_229681

theorem dan_marbles (original_marbles : ℕ) (given_marbles : ℕ) (remaining_marbles : ℕ) : 
  original_marbles = 64 ∧ given_marbles = 14 → remaining_marbles = 50 := 
by 
  sorry

end dan_marbles_l229_229681


namespace smallest_integer_in_consecutive_set_l229_229614

theorem smallest_integer_in_consecutive_set :
  ∃ (n : ℤ), 2 < n ∧ ∀ m : ℤ, m < n → ¬ (m + 6 < 2 * (m + 3) - 2) :=
sorry

end smallest_integer_in_consecutive_set_l229_229614


namespace arithmetic_sequence_8th_term_l229_229041

theorem arithmetic_sequence_8th_term (a d : ℤ)
  (h4 : a + 3 * d = 23) 
  (h6 : a + 5 * d = 47) : 
  a + 7 * d = 71 := 
by
  sorry

end arithmetic_sequence_8th_term_l229_229041


namespace necessary_but_not_sufficient_l229_229992

theorem necessary_but_not_sufficient (p q : Prop) : 
  (p ∨ q) → (p ∧ q) → False :=
by
  sorry

end necessary_but_not_sufficient_l229_229992


namespace worst_player_is_nephew_l229_229619

-- Define the family members
inductive Player
| father : Player
| sister : Player
| son : Player
| nephew : Player

open Player

-- Define a twin relationship
def is_twin (p1 p2 : Player) : Prop :=
  (p1 = son ∧ p2 = nephew) ∨ (p1 = nephew ∧ p2 = son)

-- Define that two players are of opposite sex
def opposite_sex (p1 p2 : Player) : Prop :=
  (p1 = sister ∧ (p2 = father ∨ p2 = son ∨ p2 = nephew)) ∨
  (p2 = sister ∧ (p1 = father ∨ p1 = son ∨ p1 = nephew))

-- Predicate for the worst player
structure WorstPlayer (p : Player) : Prop :=
  (twin_exists : ∃ twin : Player, is_twin p twin)
  (opposite_sex_best : ∀ twin best, is_twin p twin → best ≠ twin → opposite_sex twin best)

-- The goal is to show that the worst player is the nephew
theorem worst_player_is_nephew : WorstPlayer nephew := sorry

end worst_player_is_nephew_l229_229619


namespace n_plus_d_is_155_l229_229592

noncomputable def n_and_d_sum : Nat :=
sorry

theorem n_plus_d_is_155 (n d : Nat) (hn : 0 < n) (hd : d < 10) 
  (h1 : 4 * n^2 + 2 * n + d = 305) 
  (h2 : 4 * n^3 + 2 * n^2 + d * n + 1 = 577 + 8 * d) : n + d = 155 := 
sorry

end n_plus_d_is_155_l229_229592


namespace fraction_subtraction_simplification_l229_229897

/-- Given that 57 equals 19 times 3, we want to prove that (8/19) - (5/57) equals 1/3. -/
theorem fraction_subtraction_simplification :
  8 / 19 - 5 / 57 = 1 / 3 := by
  sorry

end fraction_subtraction_simplification_l229_229897


namespace probability_all_quitters_same_tribe_l229_229873

-- Definitions of the problem conditions
def total_contestants : ℕ := 20
def tribe_size : ℕ := 10
def quitters : ℕ := 3

-- Definition of the binomial coefficient
def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Theorem statement
theorem probability_all_quitters_same_tribe :
  (choose tribe_size quitters + choose tribe_size quitters) * 
  (total_contestants.choose quitters) = 240 
  ∧ ((choose tribe_size quitters + choose tribe_size quitters) / (total_contestants.choose quitters)) = 20 / 95 :=
by
  sorry

end probability_all_quitters_same_tribe_l229_229873


namespace evaluate_x_squared_plus_y_squared_l229_229109

theorem evaluate_x_squared_plus_y_squared (x y : ℝ) (h₁ : 3 * x + y = 20) (h₂ : 4 * x + y = 25) :
  x^2 + y^2 = 50 :=
sorry

end evaluate_x_squared_plus_y_squared_l229_229109


namespace find_k_l229_229228

theorem find_k (k : ℕ) (h : 2 * 3 - k + 1 = 0) : k = 7 :=
sorry

end find_k_l229_229228


namespace cats_to_dogs_ratio_l229_229968

noncomputable def num_dogs : ℕ := 18
noncomputable def num_cats : ℕ := num_dogs - 6
noncomputable def ratio (a b : ℕ) : ℚ := a / b

theorem cats_to_dogs_ratio (h1 : num_dogs = 18) (h2 : num_cats = num_dogs - 6) : ratio num_cats num_dogs = 2 / 3 :=
by
  sorry

end cats_to_dogs_ratio_l229_229968


namespace john_money_left_l229_229705

theorem john_money_left 
  (start_amount : ℝ := 100) 
  (price_roast : ℝ := 17)
  (price_vegetables : ℝ := 11)
  (price_wine : ℝ := 12)
  (price_dessert : ℝ := 8)
  (price_bread : ℝ := 4)
  (price_milk : ℝ := 2)
  (discount_rate : ℝ := 0.15)
  (tax_rate : ℝ := 0.05)
  (total_cost := price_roast + price_vegetables + price_wine + price_dessert + price_bread + price_milk)
  (discount_amount := discount_rate * total_cost)
  (discounted_total := total_cost - discount_amount)
  (tax_amount := tax_rate * discounted_total)
  (final_amount := discounted_total + tax_amount)
  : start_amount - final_amount = 51.80 := sorry

end john_money_left_l229_229705


namespace xy_is_necessary_but_not_sufficient_l229_229893

theorem xy_is_necessary_but_not_sufficient (x y : ℝ) :
  (x^2 + y^2 = 0 → xy = 0) ∧ (xy = 0 → ¬(x^2 + y^2 ≠ 0)) := by
  sorry

end xy_is_necessary_but_not_sufficient_l229_229893


namespace quadratic_intersects_at_3_points_l229_229916

theorem quadratic_intersects_at_3_points (m : ℝ) : 
  (exists x : ℝ, x^2 + 2*x + m = 0) ∧ (m ≠ 0) → m < 1 :=
by
  sorry

end quadratic_intersects_at_3_points_l229_229916


namespace factorize_3a_squared_minus_6a_plus_3_l229_229778

theorem factorize_3a_squared_minus_6a_plus_3 (a : ℝ) : 
  3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 :=
by 
  sorry

end factorize_3a_squared_minus_6a_plus_3_l229_229778


namespace expected_digits_of_fair_icosahedral_die_l229_229481

noncomputable def expected_num_of_digits : ℚ :=
  (9 / 20) * 1 + (11 / 20) * 2

theorem expected_digits_of_fair_icosahedral_die :
  expected_num_of_digits = 1.55 := by
  sorry

end expected_digits_of_fair_icosahedral_die_l229_229481


namespace matrix_power_is_correct_l229_229568

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]
def A_cubed : Matrix (Fin 2) (Fin 2) ℤ := !![3, -6; 6, -3]

theorem matrix_power_is_correct : A ^ 3 = A_cubed := by 
  sorry

end matrix_power_is_correct_l229_229568


namespace smallest_sum_arith_geo_seq_l229_229860

theorem smallest_sum_arith_geo_seq (A B C D : ℕ) 
  (h1 : A + B + C + D > 0)
  (h2 : 2 * B = A + C)
  (h3 : 16 * C = 7 * B)
  (h4 : 16 * D = 49 * B) :
  A + B + C + D = 97 :=
sorry

end smallest_sum_arith_geo_seq_l229_229860


namespace david_marks_in_mathematics_l229_229036

-- Define marks in individual subjects and the average
def marks_in_english : ℝ := 70
def marks_in_physics : ℝ := 78
def marks_in_chemistry : ℝ := 60
def marks_in_biology : ℝ := 65
def average_marks : ℝ := 66.6
def number_of_subjects : ℕ := 5

-- Define a statement to be proven
theorem david_marks_in_mathematics : 
    average_marks * number_of_subjects 
    - (marks_in_english + marks_in_physics + marks_in_chemistry + marks_in_biology) = 60 := 
by simp [average_marks, number_of_subjects, marks_in_english, marks_in_physics, marks_in_chemistry, marks_in_biology]; sorry

end david_marks_in_mathematics_l229_229036


namespace find_original_price_l229_229198

theorem find_original_price (reduced_price : ℝ) (percent : ℝ) (original_price : ℝ) 
  (h1 : reduced_price = 6) (h2 : percent = 0.25) (h3 : reduced_price = percent * original_price) : 
  original_price = 24 :=
sorry

end find_original_price_l229_229198


namespace total_students_high_school_l229_229487

theorem total_students_high_school (s10 s11 s12 total_students sample: ℕ ) 
  (h1 : s10 = 600) 
  (h2 : sample = 45) 
  (h3 : s11 = 20) 
  (h4 : s12 = 10) 
  (h5 : sample = s10 + s11 + s12) : 
  total_students = 1800 :=
by 
  sorry

end total_students_high_school_l229_229487


namespace calc_ratio_of_d_to_s_l229_229277

theorem calc_ratio_of_d_to_s {n s d : ℝ} (h_n_eq_24 : n = 24)
    (h_tiles_area_64_pct : (576 * s^2) = 0.64 * (n * s + d)^2) : 
    d / s = 6 / 25 :=
by
  sorry

end calc_ratio_of_d_to_s_l229_229277


namespace recurrence_relation_l229_229007

noncomputable def p (n k : ℕ) : ℚ := sorry

theorem recurrence_relation (n k : ℕ) (h : k < n) :
  p n k = p (n - 1) k - (1 / 2^k) * p (n - k) k + (1 / 2^k) :=
by sorry

end recurrence_relation_l229_229007


namespace quadratic_solution_1_quadratic_solution_2_l229_229785

theorem quadratic_solution_1 (x : ℝ) :
  x^2 + 3 * x - 1 = 0 ↔ (x = (-3 + Real.sqrt 13) / 2) ∨ (x = (-3 - Real.sqrt 13) / 2) :=
by
  sorry

theorem quadratic_solution_2 (x : ℝ) :
  (x - 2)^2 = 2 * (x - 2) ↔ (x = 2) ∨ (x = 4) :=
by
  sorry

end quadratic_solution_1_quadratic_solution_2_l229_229785


namespace compare_logs_l229_229796

noncomputable def a := Real.log 6 / Real.log 3
noncomputable def b := Real.log 10 / Real.log 5
noncomputable def c := Real.log 14 / Real.log 7

theorem compare_logs : a > b ∧ b > c := by
  -- Proof will be written here, currently placeholder
  sorry

end compare_logs_l229_229796


namespace algebraic_expression_value_l229_229804

theorem algebraic_expression_value (a b : ℝ) (h1 : a + b = 8) (h2 : a * b = 9) : a^2 - 3 * a * b + b^2 = 19 :=
sorry

end algebraic_expression_value_l229_229804


namespace power_comparison_l229_229905

theorem power_comparison : (5 : ℕ) ^ 30 < (3 : ℕ) ^ 50 ∧ (3 : ℕ) ^ 50 < (4 : ℕ) ^ 40 := by
  sorry

end power_comparison_l229_229905


namespace part1_part2_l229_229177

noncomputable def is_monotonically_increasing (f' : ℝ → ℝ) := ∀ x, f' x ≥ 0

noncomputable def is_monotonically_decreasing (f' : ℝ → ℝ) (I : Set ℝ) := ∀ x ∈ I, f' x ≤ 0

def f' (a x : ℝ) : ℝ := 3 * x ^ 2 - a

theorem part1 (a : ℝ) : 
  is_monotonically_increasing (f' a) ↔ a ≤ 0 := sorry

theorem part2 (a : ℝ) : 
  is_monotonically_decreasing (f' a) (Set.Ioo (-1 : ℝ) (1 : ℝ)) ↔ a ≥ 3 := sorry

end part1_part2_l229_229177


namespace triangle_area_l229_229969

theorem triangle_area (P : ℝ × ℝ)
  (Q : ℝ × ℝ) (R : ℝ × ℝ)
  (P_eq : P = (3, 2))
  (Q_eq : ∃ b, Q = (7/3, 0) ∧ 2 = 3 * 3 + b ∧ 0 = 3 * (7/3) + b)
  (R_eq : ∃ b, R = (4, 0) ∧ 2 = -2 * 3 + b ∧ 0 = -2 * 4 + b) :
  (1/2) * abs (Q.1 - R.1) * abs (P.2) = 5/3 :=
by
  sorry

end triangle_area_l229_229969


namespace acute_angles_complementary_l229_229683

-- Given conditions
variables (α β : ℝ)
variables (α_acute : 0 < α ∧ α < π / 2) (β_acute : 0 < β ∧ β < π / 2)
variables (h : (sin α) ^ 2 + (sin β) ^ 2 = sin (α + β))

-- Statement we want to prove
theorem acute_angles_complementary : α + β = π / 2 :=
  sorry

end acute_angles_complementary_l229_229683


namespace max_marks_l229_229652

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 59 + 40) : M = 300 :=
by
  sorry

end max_marks_l229_229652


namespace difference_between_a_b_l229_229684

theorem difference_between_a_b (a b : ℝ) (d : ℝ) : 
  (a - b = d) → (a ^ 2 + b ^ 2 = 150) → (a * b = 25) → d = 10 :=
by
  sorry

end difference_between_a_b_l229_229684


namespace monotonically_increasing_interval_l229_229775

def f (x : ℝ) (a : ℝ) : ℝ := |2 * x + a| + 3

theorem monotonically_increasing_interval (a : ℝ) :
  (∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ a ≤ f x₂ a) → a ≥ -2 :=
by
  sorry

end monotonically_increasing_interval_l229_229775


namespace value_of_fraction_l229_229535

variables {a b c : ℝ}

-- Conditions
def quadratic_has_no_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c < 0

def person_A_roots (a' b c : ℝ) : Prop :=
  b = -6 * a' ∧ c = 8 * a'

def person_B_roots (a b' c : ℝ) : Prop :=
  b' = -3 * a ∧ c = -4 * a

-- Proof Statement
theorem value_of_fraction (a b c a' b' : ℝ)
  (hnr : quadratic_has_no_real_roots a b c)
  (hA : person_A_roots a' b c)
  (hB : person_B_roots a b' c) :
  (2 * b + 3 * c) / a = 6 :=
by
  sorry

end value_of_fraction_l229_229535


namespace arithmetic_sequence_problem_l229_229874

variables (a_n b_n : ℕ → ℚ)
variables (S_n T_n : ℕ → ℚ)
variable (n : ℕ)

axiom sum_a_terms : ∀ n : ℕ, S_n n = n / 2 * (a_n 1 + a_n n)
axiom sum_b_terms : ∀ n : ℕ, T_n n = n / 2 * (b_n 1 + b_n n)
axiom given_fraction : ∀ n : ℕ, n > 0 → S_n n / T_n n = (2 * n + 1) / (4 * n - 2)

theorem arithmetic_sequence_problem : 
  (a_n 10) / (b_n 3 + b_n 18) + (a_n 11) / (b_n 6 + b_n 15) = 41 / 78 :=
sorry

end arithmetic_sequence_problem_l229_229874


namespace square_side_is_8_l229_229182

-- Definitions based on problem conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 16
def rectangle_area : ℝ := rectangle_width * rectangle_length

def square_side_length (s : ℝ) : Prop := s^2 = rectangle_area

-- The theorem we need to prove
theorem square_side_is_8 (s : ℝ) : square_side_length s → s = 8 := by
  -- Proof to be filled in
  sorry

end square_side_is_8_l229_229182


namespace students_taking_art_l229_229341

def total_students : ℕ := 500
def students_taking_music : ℕ := 20
def students_taking_both : ℕ := 10
def students_taking_neither : ℕ := 470

theorem students_taking_art :
  ∃ (A : ℕ), A = 20 ∧ total_students = 
             (students_taking_music - students_taking_both) + (A - students_taking_both) + students_taking_both + students_taking_neither :=
by
  sorry

end students_taking_art_l229_229341


namespace deer_meat_distribution_l229_229723

theorem deer_meat_distribution :
  ∃ (a1 a2 a3 a4 a5 : ℕ), a1 > a2 ∧ a2 > a3 ∧ a3 > a4 ∧ a4 > a5 ∧
  (a1 + a2 + a3 + a4 + a5 = 500) ∧
  (a2 + a3 + a4 = 300) :=
sorry

end deer_meat_distribution_l229_229723


namespace soccer_season_length_l229_229605

def total_games : ℕ := 27
def games_per_month : ℕ := 9
def months_in_season : ℕ := total_games / games_per_month

theorem soccer_season_length : months_in_season = 3 := by
  unfold months_in_season
  unfold total_games
  unfold games_per_month
  sorry

end soccer_season_length_l229_229605


namespace expression_value_l229_229863

noncomputable def expression (x b : ℝ) : ℝ :=
  (x / (x + b) + b / (x - b)) / (b / (x + b) - x / (x - b))

theorem expression_value (b x : ℝ) (hb : b ≠ 0) (hx : x ≠ b ∧ x ≠ -b) :
  expression x b = -1 := 
by
  sorry

end expression_value_l229_229863


namespace croissant_price_l229_229919

theorem croissant_price (price_almond: ℝ) (total_expenditure: ℝ) (weeks: ℕ) (price_regular: ℝ) 
  (h1: price_almond = 5.50) (h2: total_expenditure = 468) (h3: weeks = 52) 
  (h4: weeks * price_regular + weeks * price_almond = total_expenditure) : price_regular = 3.50 :=
by 
  sorry

end croissant_price_l229_229919


namespace cost_per_dvd_l229_229589

theorem cost_per_dvd (total_cost : ℝ) (num_dvds : ℕ) 
  (h1 : total_cost = 4.8) (h2 : num_dvds = 4) : (total_cost / num_dvds) = 1.2 :=
by
  sorry

end cost_per_dvd_l229_229589


namespace surface_area_after_removal_l229_229134

theorem surface_area_after_removal :
  let cube_side := 4
  let corner_cube_side := 2
  let original_surface_area := 6 * (cube_side * cube_side)
  (original_surface_area = 96) ->
  (6 * (cube_side * cube_side) - 8 * 3 * (corner_cube_side * corner_cube_side) + 8 * 3 * (corner_cube_side * corner_cube_side) = 96) :=
by
  intros
  sorry

end surface_area_after_removal_l229_229134


namespace min_value_of_a_plus_b_l229_229138

-- Definitions based on the conditions
variables (a b : ℝ)
def roots_real (a b : ℝ) : Prop := a^2 ≥ 8 * b ∧ b^2 ≥ a
def positive_vars (a b : ℝ) : Prop := a > 0 ∧ b > 0
def min_a_plus_b (a b : ℝ) : Prop := a + b = 6

-- Lean theorem statement
theorem min_value_of_a_plus_b (a b : ℝ) (hr : roots_real a b) (pv : positive_vars a b) : min_a_plus_b a b :=
sorry

end min_value_of_a_plus_b_l229_229138


namespace negation_of_universal_l229_229880

theorem negation_of_universal (P : ∀ x : ℤ, x^3 < 1) : ¬ (∀ x : ℤ, x^3 < 1) ↔ ∃ x : ℤ, x^3 ≥ 1 :=
by
  sorry

end negation_of_universal_l229_229880


namespace difference_in_price_l229_229714

noncomputable def total_cost : ℝ := 70.93
noncomputable def pants_price : ℝ := 34.00

theorem difference_in_price (total_cost pants_price : ℝ) (h_total : total_cost = 70.93) (h_pants : pants_price = 34.00) :
  (total_cost - pants_price) - pants_price = 2.93 :=
by
  sorry

end difference_in_price_l229_229714


namespace sam_annual_income_l229_229383

theorem sam_annual_income
  (q : ℝ) (I : ℝ)
  (h1 : 30000 * 0.01 * q + 15000 * 0.01 * (q + 3) + (I - 45000) * 0.01 * (q + 5) = (q + 0.35) * 0.01 * I) :
  I = 48376 := 
sorry

end sam_annual_income_l229_229383


namespace S_ploughing_time_l229_229560

theorem S_ploughing_time (R S : ℝ) (hR_rate : R = 1 / 15) (h_combined_rate : R + S = 1 / 10) : S = 1 / 30 := sorry

end S_ploughing_time_l229_229560


namespace min_value_x_squared_plus_10x_l229_229960

theorem min_value_x_squared_plus_10x : ∃ x : ℝ, (x^2 + 10 * x) = -25 :=
by {
  sorry
}

end min_value_x_squared_plus_10x_l229_229960


namespace distance_to_directrix_l229_229039

noncomputable def ellipse (x y : ℝ) : Prop := x^2 / 64 + y^2 / 28 = 1

noncomputable def left_focus : ℝ × ℝ := (-6, 0)

noncomputable def right_focus : ℝ × ℝ := (6, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_to_directrix (P : ℝ × ℝ) (hP : ellipse P.1 P.2) (hPF1 : distance P left_focus = 4) :
  distance P right_focus * 4 / 3 = 16 :=
sorry

end distance_to_directrix_l229_229039


namespace football_team_lineup_count_l229_229012

theorem football_team_lineup_count :
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3

  offensive_lineman_choices * quarterback_choices * running_back_choices * wide_receiver_choices * tight_end_choices = 39600 :=
by
  let team_members := 12
  let offensive_lineman_choices := 5
  let remaining_choices := team_members - 1
  let quarterback_choices := remaining_choices
  let running_back_choices := remaining_choices - 1
  let wide_receiver_choices := remaining_choices - 2
  let tight_end_choices := remaining_choices - 3
  
  exact sorry

end football_team_lineup_count_l229_229012


namespace hotel_flat_fee_l229_229604

theorem hotel_flat_fee (f n : ℝ) (h1 : f + n = 120) (h2 : f + 6 * n = 330) : f = 78 :=
by
  sorry

end hotel_flat_fee_l229_229604


namespace magnitude_of_b_l229_229551

variable (a b : ℝ)

-- Defining the given conditions as hypotheses
def condition1 : Prop := (a - b) * (a - b) = 9
def condition2 : Prop := (a + 2 * b) * (a + 2 * b) = 36
def condition3 : Prop := a^2 + (a * b) - 2 * b^2 = -9

-- Defining the theorem to prove
theorem magnitude_of_b (ha : condition1 a b) (hb : condition2 a b) (hc : condition3 a b) : b^2 = 3 := 
sorry

end magnitude_of_b_l229_229551


namespace circle_through_points_l229_229538

-- Definitions of the points
def O : (ℝ × ℝ) := (0, 0)
def M1 : (ℝ × ℝ) := (1, 1)
def M2 : (ℝ × ℝ) := (4, 2)

-- Definition of the center and radius of the circle
def center : (ℝ × ℝ) := (4, -3)
def radius : ℝ := 5

-- The circle equation function
def circle_eq (x y : ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
  (x - c.1)^2 + (y + c.2)^2 = r^2

theorem circle_through_points :
  circle_eq 0 0 center radius ∧ circle_eq 1 1 center radius ∧ circle_eq 4 2 center radius :=
by
  -- This is where the proof would go
  sorry

end circle_through_points_l229_229538


namespace fraction_simplify_l229_229047

theorem fraction_simplify :
  (3 + 6 - 12 + 24 + 48 - 96 + 192) / (6 + 12 - 24 + 48 + 96 - 192 + 384) = 1 / 2 :=
by sorry

end fraction_simplify_l229_229047


namespace kathleen_allowance_l229_229836

theorem kathleen_allowance (x : ℝ) :
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  percentage_increase = 150 → x = 2 :=
by
  -- Definitions and conditions setup
  let middle_school_allowance := 10
  let senior_year_allowance := 10 * x + 5
  let percentage_increase := ((senior_year_allowance - middle_school_allowance) / middle_school_allowance) * 100
  intros h
  -- Skipping the proof
  sorry

end kathleen_allowance_l229_229836


namespace Don_poured_milk_correct_amount_l229_229204

theorem Don_poured_milk_correct_amount :
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  poured_milk = 5 / 16 :=
by
  let original_milk : ℚ := 3 / 8
  let portion_poured : ℚ := 5 / 6
  let poured_milk : ℚ := portion_poured * original_milk
  show poured_milk = 5 / 16
  sorry

end Don_poured_milk_correct_amount_l229_229204


namespace work_last_duration_l229_229636

theorem work_last_duration
  (work_rate_x : ℚ := 1 / 20)
  (work_rate_y : ℚ := 1 / 12)
  (days_x_worked_alone : ℚ := 4)
  (combined_work_rate : ℚ := work_rate_x + work_rate_y)
  (remaining_work : ℚ := 1 - days_x_worked_alone * work_rate_x) :
  (remaining_work / combined_work_rate + days_x_worked_alone = 10) :=
by
  sorry

end work_last_duration_l229_229636


namespace starting_number_unique_l229_229257

-- Definitions based on conditions
def has_two_threes (n : ℕ) : Prop :=
  (n / 10 = 3 ∧ n % 10 = 3)

def is_starting_number (n m : ℕ) : Prop :=
  ∃ k, n + k = m ∧ k < (m - n) ∧ has_two_threes m

-- Theorem stating the proof problem
theorem starting_number_unique : ∃ n, is_starting_number n 30 ∧ n = 32 := 
sorry

end starting_number_unique_l229_229257


namespace fn_conjecture_l229_229743

theorem fn_conjecture (f : ℕ → ℝ → ℝ) (x : ℝ) (h_pos : x > 0) :
  (f 1 x = x / (Real.sqrt (1 + x^2))) →
  (∀ n, f (n + 1) x = f 1 (f n x)) →
  (∀ n, f n x = x / (Real.sqrt (1 + n * x ^ 2))) := by
  sorry

end fn_conjecture_l229_229743


namespace addition_example_l229_229625

theorem addition_example : 0.4 + 56.7 = 57.1 := by
  -- Here we need to prove the main statement
  sorry

end addition_example_l229_229625


namespace geom_seq_sum_l229_229525

theorem geom_seq_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = 2) (h2 : a 1 * a 5 = 64) :
  (a 1 * (1 - q^n)) / (1 - q) = 2^(n+1) - 2 := 
sorry

end geom_seq_sum_l229_229525


namespace volume_of_sphere_inscribed_in_cube_of_edge_8_l229_229286

noncomputable def volume_of_inscribed_sphere (edge_length : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (edge_length / 2) ^ 3

theorem volume_of_sphere_inscribed_in_cube_of_edge_8 :
  volume_of_inscribed_sphere 8 = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_sphere_inscribed_in_cube_of_edge_8_l229_229286


namespace inequality_solution_l229_229424

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end inequality_solution_l229_229424


namespace minimize_f_l229_229119

noncomputable def f (x a : ℝ) : ℝ :=
  |7 * x - 3 * a + 8| + |5 * x + 4 * a - 6| + |x - a - 8| - 24

theorem minimize_f (a : ℝ) : a = 82 / 43 :=
by
  sorry

end minimize_f_l229_229119


namespace part1_part2_l229_229871

variable {f : ℝ → ℝ}

-- Condition 1: f is an odd function
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Condition 2: ∀ a b ∈ ℝ, (a + b ≠ 0) → (f(a) + f(b))/(a + b) > 0
def positiveQuotient (f : ℝ → ℝ) : Prop :=
  ∀ a b, a + b ≠ 0 → (f a + f b) / (a + b) > 0

-- Sub-problem (1): For any a, b ∈ ℝ, a > b ⟹ f(a) > f(b)
theorem part1 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) (a b : ℝ) (h : a > b) : f a > f b :=
  sorry

-- Sub-problem (2): If f(9^x - 2 * 3^x) + f(2 * 9^x - k) > 0 for any x ∈ [0, ∞), then k < 1
theorem part2 (h_odd : isOddFunction f) (h_posQuot : positiveQuotient f) :
  (∀ x : ℝ, 0 ≤ x → f (9^x - 2 * 3^x) + f (2 * 9^x - k) > 0) → k < 1 :=
  sorry

end part1_part2_l229_229871


namespace square_of_1023_l229_229986

def square_1023_eq_1046529 : Prop :=
  let x := 1023
  x * x = 1046529

theorem square_of_1023 : square_1023_eq_1046529 :=
by
  sorry

end square_of_1023_l229_229986


namespace target_hit_probability_l229_229881

-- Defining the probabilities for A, B, and C hitting the target.
def P_A_hit := 1 / 2
def P_B_hit := 1 / 3
def P_C_hit := 1 / 4

-- Defining the probability that A, B, and C miss the target.
def P_A_miss := 1 - P_A_hit
def P_B_miss := 1 - P_B_hit
def P_C_miss := 1 - P_C_hit

-- Calculating the combined probability that none of them hit the target.
def P_none_hit := P_A_miss * P_B_miss * P_C_miss

-- Now, calculating the probability that at least one of them hits the target.
def P_hit := 1 - P_none_hit

-- Statement of the theorem.
theorem target_hit_probability : P_hit = 3 / 4 := by
  sorry

end target_hit_probability_l229_229881


namespace crow_eats_nuts_l229_229186

theorem crow_eats_nuts (time_fifth_nuts : ℕ) (time_quarter_nuts : ℕ) (h : time_fifth_nuts = 8) :
  time_quarter_nuts = 10 :=
sorry

end crow_eats_nuts_l229_229186


namespace probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l229_229552

noncomputable def probability_k_gnomes_fall (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  p * (1 - p) ^ (n - k)

noncomputable def expected_number_of_fallen_gnomes (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : ℝ :=
  n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p)

theorem probability_k_gnomes_fall_correct (n k : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  probability_k_gnomes_fall n k p h = p * (1 - p) ^ (n - k) :=
by sorry

theorem expected_number_of_fallen_gnomes_correct (n : ℕ) (p : ℝ) (h : 0 < p ∧ p < 1) : 
  expected_number_of_fallen_gnomes n p h = n + 1 - (1 / p) + ((1 - p) ^ (n + 1) / p) :=
by sorry

end probability_k_gnomes_fall_correct_expected_number_of_fallen_gnomes_correct_l229_229552


namespace ellipse_with_foci_on_y_axis_l229_229123

theorem ellipse_with_foci_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  (∀ x y : ℝ, mx^2 + ny^2 = 1) ↔ (m > n ∧ n > 0) := 
sorry

end ellipse_with_foci_on_y_axis_l229_229123


namespace vec_a_squared_minus_vec_b_squared_l229_229754

variable (a b : ℝ × ℝ)
variable (h1 : a + b = (-3, 6))
variable (h2 : a - b = (-3, 2))

theorem vec_a_squared_minus_vec_b_squared : (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = 32 :=
sorry

end vec_a_squared_minus_vec_b_squared_l229_229754


namespace pattern_E_cannot_be_formed_l229_229143

-- Define the basic properties of the tile and the patterns
inductive Tile
| rhombus (diag_coloring : Bool) -- representing black-and-white diagonals

inductive Pattern
| optionA
| optionB
| optionC
| optionD
| optionE

-- The given tile is a rhombus with a certain coloring scheme
def given_tile : Tile := Tile.rhombus true

-- The statement to prove
theorem pattern_E_cannot_be_formed : 
  ¬ (∃ f : Pattern → Tile, f Pattern.optionE = given_tile) :=
sorry

end pattern_E_cannot_be_formed_l229_229143


namespace Buffy_whiskers_is_40_l229_229776

def number_of_whiskers (Puffy Scruffy Buffy Juniper : ℕ) : Prop :=
  Puffy = 3 * Juniper ∧
  Puffy = Scruffy / 2 ∧
  Buffy = (Puffy + Scruffy + Juniper) / 3 ∧
  Juniper = 12

theorem Buffy_whiskers_is_40 :
  ∃ (Puffy Scruffy Buffy Juniper : ℕ), 
    number_of_whiskers Puffy Scruffy Buffy Juniper ∧ Buffy = 40 := 
by
  sorry

end Buffy_whiskers_is_40_l229_229776


namespace six_digit_numbers_with_zero_l229_229799

-- Define the total number of 6-digit numbers
def total_six_digit_numbers : ℕ := 900000

-- Define the number of 6-digit numbers with no zero
def six_digit_numbers_no_zero : ℕ := 531441

-- Prove the number of 6-digit numbers with at least one zero
theorem six_digit_numbers_with_zero : 
  (total_six_digit_numbers - six_digit_numbers_no_zero) = 368559 :=
by
  sorry

end six_digit_numbers_with_zero_l229_229799


namespace system_solution_l229_229553

theorem system_solution :
  (∀ x y : ℝ, (2 * x + 3 * y = 19) ∧ (3 * x + 4 * y = 26) → x = 2 ∧ y = 5) →
  (∃ x y : ℝ, (2 * (2 * x + 4) + 3 * (y + 3) = 19) ∧ (3 * (2 * x + 4) + 4 * (y + 3) = 26) ∧ x = -1 ∧ y = 2) :=
by
  sorry

end system_solution_l229_229553


namespace intersection_of_A_and_B_l229_229741

noncomputable def U : Set ℝ := Set.univ
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 3}
def B : Set ℝ := {x : ℝ | x < -1 ∨ x ≥ 4}

theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} :=
by
  sorry

end intersection_of_A_and_B_l229_229741


namespace goldfish_count_15_weeks_l229_229309

def goldfish_count_after_weeks (initial : ℕ) (weeks : ℕ) : ℕ :=
  let deaths := λ n => 10 + 2 * (n - 1)
  let purchases := λ n => 5 + 2 * (n - 1)
  let rec update_goldfish (current : ℕ) (week : ℕ) :=
    if week = 0 then current
    else 
      let new_count := current - deaths week + purchases week
      update_goldfish new_count (week - 1)
  update_goldfish initial weeks

theorem goldfish_count_15_weeks : goldfish_count_after_weeks 35 15 = 15 :=
  by
  sorry

end goldfish_count_15_weeks_l229_229309


namespace arccos_one_eq_zero_l229_229046

theorem arccos_one_eq_zero : Real.arccos 1 = 0 := by
  sorry

end arccos_one_eq_zero_l229_229046


namespace bus_stops_for_28_minutes_per_hour_l229_229521

-- Definitions based on the conditions
def without_stoppages_speed : ℕ := 75
def with_stoppages_speed : ℕ := 40
def speed_difference : ℕ := without_stoppages_speed - with_stoppages_speed

-- Theorem statement
theorem bus_stops_for_28_minutes_per_hour : 
  ∀ (T : ℕ), (T = (speed_difference*60)/(without_stoppages_speed))  → 
  T = 28 := 
by
  sorry

end bus_stops_for_28_minutes_per_hour_l229_229521


namespace quadratic_intersects_x_axis_only_once_l229_229709

theorem quadratic_intersects_x_axis_only_once (a : ℝ) :
  (∀ x : ℝ, (a * x^2 - a * x + 3 * x + 1 = 0) → a = 1 ∨ a = 9) :=
sorry

end quadratic_intersects_x_axis_only_once_l229_229709


namespace problem_remainder_6_pow_83_add_8_pow_83_mod_49_l229_229963

-- Definitions based on the conditions.
def euler_totient_49 : ℕ := 42

theorem problem_remainder_6_pow_83_add_8_pow_83_mod_49 
  (h1 : 6 ^ euler_totient_49 ≡ 1 [MOD 49])
  (h2 : 8 ^ euler_totient_49 ≡ 1 [MOD 49]) :
  (6 ^ 83 + 8 ^ 83) % 49 = 35 :=
by
  sorry

end problem_remainder_6_pow_83_add_8_pow_83_mod_49_l229_229963


namespace sum_sequences_l229_229180

theorem sum_sequences : 
  (1 + 12 + 23 + 34 + 45) + (10 + 20 + 30 + 40 + 50) = 265 := by
  sorry

end sum_sequences_l229_229180


namespace solve_for_x_l229_229340

theorem solve_for_x (x : ℚ) :
  (4 * x - 12) / 3 = (3 * x + 6) / 5 → 
  x = 78 / 11 :=
sorry

end solve_for_x_l229_229340


namespace lcm_12_18_l229_229641

theorem lcm_12_18 : Nat.lcm 12 18 = 36 :=
by
  -- Definitions of the conditions
  have h12 : 12 = 2 * 2 * 3 := by norm_num
  have h18 : 18 = 2 * 3 * 3 := by norm_num
  
  -- Calculating LCM using the built-in Nat.lcm
  rw [Nat.lcm_comm]  -- Ordering doesn't matter for lcm
  rw [Nat.lcm, h12, h18]
  -- Prime factorizations checks are implicitly handled
  
  -- Calculate the LCM based on the highest powers from the factorizations
  have lcm_val : 4 * 9 = 36 := by norm_num
  
  -- So, the LCM of 12 and 18 is
  exact lcm_val

end lcm_12_18_l229_229641


namespace simplification_evaluation_l229_229911

-- Define the variables x and y
def x : ℕ := 2
def y : ℕ := 3

-- Define the expression
def expr := 5 * (3 * x^2 * y - x * y^2) - (x * y^2 + 3 * x^2 * y)

-- Lean 4 statement to prove the equivalence
theorem simplification_evaluation : expr = 36 :=
by
  -- Place the proof here when needed
  sorry

end simplification_evaluation_l229_229911


namespace arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l229_229628

theorem arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125 :
  (16 + 23 + 38 + 11.5) / 4 = 22.125 :=
by
  sorry

end arithmetic_mean_of_16_23_38_and_11_point_5_is_22_point_125_l229_229628


namespace product_ab_zero_l229_229369

theorem product_ab_zero (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 125) : a * b = 0 :=
by
  sorry

end product_ab_zero_l229_229369


namespace ellipse_standard_equation_l229_229834

theorem ellipse_standard_equation
  (a b : ℝ) (P : ℝ × ℝ) (h_center : P = (3, 0))
  (h_a_eq_3b : a = 3 * b) 
  (h1 : a = 3) 
  (h2 : b = 1) : 
  (∀ (x y : ℝ), (x = 3 → y = 0) → (x = 0 → y = 3)) → 
  ((x^2 / a^2) + y^2 = 1 ∨ (x^2 / b^2) + (y^2 / a^2) = 1) := 
by sorry

end ellipse_standard_equation_l229_229834


namespace remainder_17_plus_x_mod_31_l229_229583

theorem remainder_17_plus_x_mod_31 {x : ℕ} (h : 13 * x ≡ 3 [MOD 31]) : (17 + x) % 31 = 22 := 
sorry

end remainder_17_plus_x_mod_31_l229_229583


namespace inequality_comparison_l229_229381

theorem inequality_comparison (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3 * y + x * y^3 :=
  sorry

end inequality_comparison_l229_229381


namespace cs_share_l229_229040

-- Definitions for the conditions
def daily_work (days: ℕ) : ℚ := 1 / days

def total_work_contribution (a_days: ℕ) (b_days: ℕ) (c_days: ℕ): ℚ := 
  daily_work a_days + daily_work b_days + daily_work c_days

def total_payment (payment: ℕ) (work_contribution: ℚ) : ℚ := 
  payment * work_contribution

-- The mathematically equivalent proof problem
theorem cs_share (a_days: ℕ) (b_days: ℕ) (total_days : ℕ) (payment: ℕ) : 
  a_days = 6 → b_days = 8 → total_days = 3 → payment = 1200 →
  total_payment payment (daily_work total_days - (daily_work a_days + daily_work b_days)) = 50 :=
sorry

end cs_share_l229_229040


namespace percentage_by_which_x_more_than_y_l229_229887

theorem percentage_by_which_x_more_than_y
    (x y z : ℝ)
    (h1 : y = 1.20 * z)
    (h2 : z = 150)
    (h3 : x + y + z = 555) :
    ((x - y) / y) * 100 = 25 :=
by
  sorry

end percentage_by_which_x_more_than_y_l229_229887


namespace childSupportOwed_l229_229547

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l229_229547


namespace distance_between_home_and_retreat_l229_229020

theorem distance_between_home_and_retreat (D : ℝ) 
  (h1 : D / 50 + D / 75 = 10) : D = 300 :=
sorry

end distance_between_home_and_retreat_l229_229020


namespace find_k_l229_229687

noncomputable def series (k : ℝ) : ℝ := ∑' n, (7 * n - 2) / k^n

theorem find_k (k : ℝ) (h₁ : 1 < k) (h₂ : series k = 17 / 2) : k = 17 / 7 :=
by
  sorry

end find_k_l229_229687


namespace xiaoma_miscalculation_l229_229409

theorem xiaoma_miscalculation (x : ℤ) (h : 40 + x = 35) : 40 / x = -8 := by
  sorry

end xiaoma_miscalculation_l229_229409


namespace smallest_int_square_eq_3x_plus_72_l229_229816

theorem smallest_int_square_eq_3x_plus_72 :
  ∃ x : ℤ, x^2 = 3 * x + 72 ∧ (∀ y : ℤ, y^2 = 3 * y + 72 → x ≤ y) :=
sorry

end smallest_int_square_eq_3x_plus_72_l229_229816


namespace max_value_A_l229_229049

noncomputable def A (x y : ℝ) : ℝ :=
  ((x^2 - y) * Real.sqrt (y + x^3 - x * y) + (y^2 - x) * Real.sqrt (x + y^3 - x * y) + 1) /
  ((x - y)^2 + 1)

theorem max_value_A (x y : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) :
  A x y ≤ 1 :=
sorry

end max_value_A_l229_229049


namespace math_problem_l229_229284

theorem math_problem : 
  ∀ n : ℕ, 
  n = 5 * 96 → 
  ((n + 17) * 69) = 34293 := 
by
  intros n h
  sorry

end math_problem_l229_229284


namespace time_jran_l229_229255

variable (D : ℕ) (S : ℕ)

theorem time_jran (hD: D = 80) (hS : S = 10) : D / S = 8 := 
  sorry

end time_jran_l229_229255


namespace sum_of_numbers_l229_229691

theorem sum_of_numbers (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) 
  (h1 : x^2 + y^2 + z^2 = 52) (h2 : x * y + y * z + z * x = 24) : 
  x + y + z = 10 :=
by
  sorry

end sum_of_numbers_l229_229691


namespace tobias_mowed_four_lawns_l229_229442

-- Let’s define the conditions
def shoe_cost : ℕ := 95
def allowance_per_month : ℕ := 5
def savings_months : ℕ := 3
def lawn_mowing_charge : ℕ := 15
def shoveling_charge : ℕ := 7
def change_after_purchase : ℕ := 15
def num_driveways_shoveled : ℕ := 5

-- Total money Tobias had before buying the shoes
def total_money : ℕ := shoe_cost + change_after_purchase

-- Money saved from allowance
def money_from_allowance : ℕ := allowance_per_month * savings_months

-- Money earned from shoveling driveways
def money_from_shoveling : ℕ := shoveling_charge * num_driveways_shoveled

-- Money earned from mowing lawns
def money_from_mowing : ℕ := total_money - money_from_allowance - money_from_shoveling

-- Number of lawns mowed
def num_lawns_mowed : ℕ := money_from_mowing / lawn_mowing_charge

-- The theorem stating the number of lawns mowed is 4
theorem tobias_mowed_four_lawns : num_lawns_mowed = 4 :=
by
  sorry

end tobias_mowed_four_lawns_l229_229442


namespace alligators_not_hiding_l229_229662

theorem alligators_not_hiding (total_alligators hiding_alligators : ℕ) 
  (h1 : total_alligators = 75) 
  (h2 : hiding_alligators = 19) : 
  total_alligators - hiding_alligators = 56 :=
by
  -- The proof will go here, which is currently a placeholder.
  sorry

end alligators_not_hiding_l229_229662


namespace candy_groups_l229_229609

theorem candy_groups (total_candies group_size : Nat) (h1 : total_candies = 30) (h2 : group_size = 3) : total_candies / group_size = 10 := by
  sorry

end candy_groups_l229_229609


namespace tony_combined_lift_weight_l229_229326

theorem tony_combined_lift_weight :
  let curl_weight := 90
  let military_press_weight := 2 * curl_weight
  let squat_weight := 5 * military_press_weight
  let bench_press_weight := 1.5 * military_press_weight
  squat_weight + bench_press_weight = 1170 :=
by
  sorry

end tony_combined_lift_weight_l229_229326


namespace petes_original_number_l229_229541

theorem petes_original_number (x y z : ℝ) (h1 : y = 3 * x) (h2 : z = y - 5) (h3 : 3 * z = 96) :
  x = 12.33 :=
by
  -- Proof goes here
  sorry

end petes_original_number_l229_229541


namespace square_division_l229_229618

theorem square_division (n : ℕ) (h : n ≥ 6) :
  ∃ (sq_div : ℕ → Prop), sq_div 6 ∧ (∀ n, sq_div n → sq_div (n + 3)) :=
by
  sorry

end square_division_l229_229618


namespace cost_of_four_dozen_bananas_l229_229982

/-- Given that five dozen bananas cost $24.00,
    prove that the cost for four dozen bananas is $19.20. -/
theorem cost_of_four_dozen_bananas 
  (cost_five_dozen: ℝ)
  (rate: cost_five_dozen = 24) : 
  ∃ (cost_four_dozen: ℝ), cost_four_dozen = 19.2 := by
  sorry

end cost_of_four_dozen_bananas_l229_229982


namespace simplification_problem_l229_229726

theorem simplification_problem :
  (3^2015 - 3^2013 + 3^2011) / (3^2015 + 3^2013 - 3^2011) = 73 / 89 :=
  sorry

end simplification_problem_l229_229726


namespace David_fewer_crunches_l229_229838

-- Definitions as per conditions.
def Zachary_crunches := 62
def David_crunches := 45

-- Proof statement for how many fewer crunches David did compared to Zachary.
theorem David_fewer_crunches : Zachary_crunches - David_crunches = 17 := by
  -- Proof details would go here, but we skip them with 'sorry'.
  sorry

end David_fewer_crunches_l229_229838


namespace students_per_table_l229_229745

theorem students_per_table (total_students tables students_bathroom students_canteen added_students exchange_students : ℕ) 
  (h1 : total_students = 47)
  (h2 : tables = 6)
  (h3 : students_bathroom = 3)
  (h4 : students_canteen = 3 * students_bathroom)
  (h5 : added_students = 2 * 4)
  (h6 : exchange_students = 3 + 3 + 3) :
  (total_students - (students_bathroom + students_canteen + added_students + exchange_students)) / tables = 3 := 
by 
  sorry

end students_per_table_l229_229745


namespace oranges_count_l229_229658

theorem oranges_count (N : ℕ) (k : ℕ) (m : ℕ) (j : ℕ) :
  (N ≡ 2 [MOD 10]) ∧ (N ≡ 0 [MOD 12]) → N = 72 :=
by
  sorry

end oranges_count_l229_229658


namespace fraction_q_p_l229_229060

theorem fraction_q_p (k : ℝ) (c p q : ℝ) (h : 8 * k^2 - 12 * k + 20 = c * (k + p)^2 + q) :
  c = 8 ∧ p = -3/4 ∧ q = 31/2 → q / p = -62 / 3 :=
by
  intros hc_hp_hq
  sorry

end fraction_q_p_l229_229060


namespace sum_non_solution_values_l229_229941

theorem sum_non_solution_values (A B C : ℝ) (h : ∀ x : ℝ, (x+B) * (A*x+36) / ((x+C) * (x+9)) = 4) :
  ∃ M : ℝ, M = - (B + 9) := 
sorry

end sum_non_solution_values_l229_229941


namespace theta_in_second_quadrant_l229_229530

open Real

-- Definitions for conditions
def cond1 (θ : ℝ) : Prop := sin θ > cos θ
def cond2 (θ : ℝ) : Prop := tan θ < 0

-- Main theorem statement
theorem theta_in_second_quadrant (θ : ℝ) 
  (h1 : cond1 θ) 
  (h2 : cond2 θ) : 
  θ > π/2 ∧ θ < π :=
sorry

end theta_in_second_quadrant_l229_229530


namespace power_multiplication_l229_229856

theorem power_multiplication :
  2^4 * 5^4 = 10000 := 
by
  sorry

end power_multiplication_l229_229856


namespace answer_key_combinations_l229_229746

theorem answer_key_combinations : 
  (2^3 - 2) * 4^2 = 96 := 
by 
  -- Explanation about why it equals to this multi-step skipped, directly written as sorry.
  sorry

end answer_key_combinations_l229_229746


namespace bobby_pancakes_left_l229_229581

theorem bobby_pancakes_left (initial_pancakes : ℕ) (bobby_ate : ℕ) (dog_ate : ℕ) :
  initial_pancakes = 21 → bobby_ate = 5 → dog_ate = 7 → initial_pancakes - (bobby_ate + dog_ate) = 9 :=
by
  intros h1 h2 h3
  sorry

end bobby_pancakes_left_l229_229581


namespace initial_num_files_l229_229396

-- Define the conditions: number of files organized in the morning, files to organize in the afternoon, and missing files.
def num_files_organized_in_morning (X : ℕ) : ℕ := X / 2
def num_files_to_organize_in_afternoon : ℕ := 15
def num_files_missing : ℕ := 15

-- Theorem to prove the initial number of files is 60.
theorem initial_num_files (X : ℕ) 
  (h1 : num_files_organized_in_morning X = X / 2)
  (h2 : num_files_to_organize_in_afternoon = 15)
  (h3 : num_files_missing = 15) :
  X = 60 :=
by
  sorry

end initial_num_files_l229_229396


namespace mask_digits_l229_229673

theorem mask_digits : 
  ∃ (elephant mouse pig panda : ℕ), 
  (elephant ≠ mouse ∧ elephant ≠ pig ∧ elephant ≠ panda ∧ 
   mouse ≠ pig ∧ mouse ≠ panda ∧ pig ≠ panda) ∧
  (4 * 4 = 16) ∧ (7 * 7 = 49) ∧ (8 * 8 = 64) ∧ (9 * 9 = 81) ∧
  (elephant = 6) ∧ (mouse = 4) ∧ (pig = 8) ∧ (panda = 1) :=
by
  sorry

end mask_digits_l229_229673


namespace a_7_eq_64_l229_229933

-- Define the problem conditions using variables in Lean
variable {a : ℕ → ℝ} -- defining the sequence as a function from natural numbers to reals
variable {q : ℝ}  -- common ratio

-- The sequence is geometric
axiom geom_seq (n : ℕ) : a (n + 1) = a n * q

-- Conditions given in the problem
axiom condition1 : a 1 + a 2 = 3
axiom condition2 : a 2 + a 3 = 6

-- Target statement to prove
theorem a_7_eq_64 : a 7 = 64 := 
sorry

end a_7_eq_64_l229_229933


namespace pipes_fill_tank_in_10_hours_l229_229230

noncomputable def R_A := 1 / 70
noncomputable def R_B := 2 * R_A
noncomputable def R_C := 2 * R_B
noncomputable def R_total := R_A + R_B + R_C
noncomputable def T := 1 / R_total

theorem pipes_fill_tank_in_10_hours :
  T = 10 := 
sorry

end pipes_fill_tank_in_10_hours_l229_229230


namespace range_of_a_l229_229607

def p (a : ℝ) := ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0
def q (a : ℝ) := ∃ x₀ : ℝ, x₀^2 + 2 * a * x₀ + 2 - a = 0

theorem range_of_a (a : ℝ) (hp : p a) (hq : q a) : a ≤ -2 ∨ a = 1 :=
by
  sorry

end range_of_a_l229_229607


namespace range_of_m_l229_229711

theorem range_of_m (m : ℝ) : 
  (∀ x y : ℝ, (x - (m^2 - 2 * m + 4) * y - 6 > 0) ↔ (x, y) ≠ (-1, -1)) →
  -1 ≤ m ∧ m ≤ 3 :=
by
  intro h
  sorry

end range_of_m_l229_229711


namespace license_plates_count_l229_229069

theorem license_plates_count :
  let num_vowels := 5
  let num_letters := 26
  let num_odd_digits := 5
  let num_even_digits := 5
  num_vowels * num_letters * num_letters * num_odd_digits * num_even_digits = 84500 :=
by
  sorry

end license_plates_count_l229_229069


namespace beautifulEquations_1_find_n_l229_229024

def isBeautifulEquations (eq1 eq2 : ℝ → Prop) : Prop :=
  ∃ x y : ℝ, eq1 x ∧ eq2 y ∧ x + y = 1

def eq1a (x : ℝ) : Prop := 4 * x - (x + 5) = 1
def eq2a (y : ℝ) : Prop := -2 * y - y = 3

theorem beautifulEquations_1 : isBeautifulEquations eq1a eq2a :=
sorry

def eq1b (x : ℝ) (n : ℝ) : Prop := 2 * x - n + 3 = 0
def eq2b (x : ℝ) (n : ℝ) : Prop := x + 5 * n - 1 = 0

theorem find_n (n : ℝ) : (∀ x1 x2 : ℝ, eq1b x1 n ∧ eq2b x2 n ∧ x1 + x2 = 1) → n = -1 / 3 :=
sorry

end beautifulEquations_1_find_n_l229_229024


namespace sum_of_squares_of_four_consecutive_even_numbers_eq_344_l229_229712

theorem sum_of_squares_of_four_consecutive_even_numbers_eq_344 (n : ℤ) 
  (h : n + (n + 2) + (n + 4) + (n + 6) = 36) : 
  n^2 + (n + 2)^2 + (n + 4)^2 + (n + 6)^2 = 344 :=
by sorry

end sum_of_squares_of_four_consecutive_even_numbers_eq_344_l229_229712


namespace square_side_length_l229_229938

theorem square_side_length (s : ℝ) (h1 : 4 * s = 12) (h2 : s^2 = 9) : s = 3 :=
sorry

end square_side_length_l229_229938


namespace correct_probability_statement_l229_229539

-- Define the conditions
def impossible_event_has_no_probability : Prop := ∀ (P : ℝ), P < 0 ∨ P > 0
def every_event_has_probability : Prop := ∀ (P : ℝ), 0 ≤ P ∧ P ≤ 1
def not_all_random_events_have_probability : Prop := ∃ (P : ℝ), P < 0 ∨ P > 1
def certain_events_do_not_have_probability : Prop := (∀ (P : ℝ), P ≠ 1)

-- The main theorem asserting that every event has a probability
theorem correct_probability_statement : every_event_has_probability :=
by sorry

end correct_probability_statement_l229_229539


namespace calculation_proof_l229_229610

theorem calculation_proof
    (a : ℝ) (b : ℝ) (c : ℝ)
    (h1 : a = 3.6)
    (h2 : b = 0.25)
    (h3 : c = 0.5) :
    (a * b) / c = 1.8 := 
by
  sorry

end calculation_proof_l229_229610


namespace average_weight_of_Arun_l229_229242

def Arun_weight_opinion (w : ℝ) : Prop :=
  (66 < w) ∧ (w < 72)

def Brother_weight_opinion (w : ℝ) : Prop :=
  (60 < w) ∧ (w < 70)

def Mother_weight_opinion (w : ℝ) : Prop :=
  w ≤ 69

def Father_weight_opinion (w : ℝ) : Prop :=
  (65 ≤ w) ∧ (w ≤ 71)

def Sister_weight_opinion (w : ℝ) : Prop :=
  (62 < w) ∧ (w ≤ 68)

def All_opinions (w : ℝ) : Prop :=
  Arun_weight_opinion w ∧
  Brother_weight_opinion w ∧
  Mother_weight_opinion w ∧
  Father_weight_opinion w ∧
  Sister_weight_opinion w

theorem average_weight_of_Arun : ∃ avg : ℝ, avg = 67.5 ∧ (∀ w, All_opinions w → (w = 67 ∨ w = 68)) :=
by
  sorry

end average_weight_of_Arun_l229_229242


namespace redistribute_oil_l229_229859

def total_boxes (trucks1 trucks2 boxes1 boxes2 : Nat) :=
  (trucks1 * boxes1) + (trucks2 * boxes2)

def total_containers (boxes containers_per_box : Nat) :=
  boxes * containers_per_box

def containers_per_truck (total_containers trucks : Nat) :=
  total_containers / trucks

theorem redistribute_oil :
  ∀ (trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks : Nat),
  trucks1 = 7 →
  trucks2 = 5 →
  boxes1 = 20 →
  boxes2 = 12 →
  containers_per_box = 8 →
  total_trucks = 10 →
  containers_per_truck (total_containers (total_boxes trucks1 trucks2 boxes1 boxes2) containers_per_box) total_trucks = 160 :=
by
  intros trucks1 trucks2 boxes1 boxes2 containers_per_box total_trucks
  intros h_trucks1 h_trucks2 h_boxes1 h_boxes2 h_containers_per_box h_total_trucks
  sorry

end redistribute_oil_l229_229859


namespace children_tickets_count_l229_229089

theorem children_tickets_count (A C : ℕ) (h1 : 8 * A + 5 * C = 201) (h2 : A + C = 33) : C = 21 :=
by
  sorry

end children_tickets_count_l229_229089


namespace wall_area_l229_229870

-- Define the conditions
variables (R J D : ℕ) (L W : ℝ)
variable (area_regular_tiles : ℝ)
variables (ratio_regular : ℕ) (ratio_jumbo : ℕ) (ratio_diamond : ℕ)
variables (length_ratio_jumbo : ℝ) (width_ratio_jumbo : ℝ)
variables (length_ratio_diamond : ℝ) (width_ratio_diamond : ℝ)
variable (total_area : ℝ)

-- Assign values to the conditions
axiom ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1
axiom size_regular : area_regular_tiles = 80
axiom jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3
axiom diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5

-- Define the statement
theorem wall_area (ratio : ratio_regular = 4 ∧ ratio_jumbo = 2 ∧ ratio_diamond = 1)
    (size_regular : area_regular_tiles = 80)
    (jumbo_tile_ratio : length_ratio_jumbo = 3 ∧ width_ratio_jumbo = 3)
    (diamond_tile_ratio : length_ratio_diamond = 2 ∧ width_ratio_diamond = 0.5):
    total_area = 140 := 
sorry

end wall_area_l229_229870


namespace bicycle_discount_l229_229469

theorem bicycle_discount (original_price : ℝ) (discount : ℝ) (discounted_price : ℝ) :
  original_price = 760 ∧ discount = 0.75 ∧ discounted_price = 570 → 
  original_price * discount = discounted_price := by
  sorry

end bicycle_discount_l229_229469


namespace christopher_strolled_5_miles_l229_229713

theorem christopher_strolled_5_miles (s t : ℝ) (hs : s = 4) (ht : t = 1.25) : s * t = 5 :=
by
  rw [hs, ht]
  norm_num

end christopher_strolled_5_miles_l229_229713


namespace expand_expression_l229_229602

theorem expand_expression (x : ℝ) : (x - 3) * (x + 6) = x^2 + 3 * x - 18 :=
by
  sorry

end expand_expression_l229_229602


namespace quadratic_root_value_m_l229_229495

theorem quadratic_root_value_m (m : ℝ) : ∃ x, x = 1 ∧ x^2 + x - m = 0 → m = 2 := by
  sorry

end quadratic_root_value_m_l229_229495


namespace dad_steps_l229_229983

theorem dad_steps (D M Y : ℕ) (h1 : 3 * D = 5 * M)
                        (h2 : 3 * M = 5 * Y)
                        (h3 : M + Y = 400) : D = 90 :=
sorry

end dad_steps_l229_229983


namespace candle_height_relation_l229_229307

theorem candle_height_relation : 
  ∀ (h : ℝ) (t : ℝ), h = 1 → (∀ (h1_burn_rate : ℝ), h1_burn_rate = 1 / 5) → (∀ (h2_burn_rate : ℝ), h2_burn_rate = 1 / 6) →
  (1 - t * 1 / 5 = 3 * (1 - t * 1 / 6)) → t = 20 / 3 :=
by
  intros h t h_init h1_burn_rate h2_burn_rate height_eq
  sorry

end candle_height_relation_l229_229307


namespace simplify_fraction_l229_229777

theorem simplify_fraction :
  ((3^2008)^2 - (3^2006)^2) / ((3^2007)^2 - (3^2005)^2) = 9 :=
by
  sorry

end simplify_fraction_l229_229777


namespace intersection_complement_l229_229686

-- Definitions and conditions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 2, 3}
def B : Set ℕ := {1, 3, 4}
def C_U (B : Set ℕ) : Set ℕ := {x ∈ U | x ∉ B}

-- Theorem statement
theorem intersection_complement :
  (C_U B) ∩ A = {0, 2} := 
by
  -- Proof is not required, so we use sorry
  sorry

end intersection_complement_l229_229686


namespace fraction_green_after_tripling_l229_229332

theorem fraction_green_after_tripling 
  (x : ℕ)
  (h₁ : ∃ x, 0 < x) -- Total number of marbles is a positive integer
  (h₂ : ∀ g y, g + y = x ∧ g = 1/4 * x ∧ y = 3/4 * x) -- Initial distribution
  (h₃ : ∀ y : ℕ, g' = 3 * g ∧ y' = y) -- Triple the green marbles, yellow stays the same
  : (g' / (g' + y')) = 1/2 := 
sorry

end fraction_green_after_tripling_l229_229332


namespace right_triangle_legs_sum_l229_229451

theorem right_triangle_legs_sum : 
  ∃ (x : ℕ), (x^2 + (x + 1)^2 = 41^2) ∧ (x + (x + 1) = 57) :=
by
  sorry

end right_triangle_legs_sum_l229_229451


namespace prime_count_at_least_two_l229_229869

theorem prime_count_at_least_two :
  ∃ (n1 n2 : ℕ), n1 ≥ 2 ∧ n2 ≥ 2 ∧ (n1 ≠ n2) ∧ Prime (n1^3 + n1^2 + 1) ∧ Prime (n2^3 + n2^2 + 1) := 
by
  sorry

end prime_count_at_least_two_l229_229869


namespace flag_movement_distance_l229_229240

theorem flag_movement_distance 
  (flagpole_length : ℝ)
  (half_mast : ℝ)
  (top_to_halfmast : ℝ)
  (halfmast_to_top : ℝ)
  (top_to_bottom : ℝ)
  (H1 : flagpole_length = 60)
  (H2 : half_mast = flagpole_length / 2)
  (H3 : top_to_halfmast = half_mast)
  (H4 : halfmast_to_top = half_mast)
  (H5 : top_to_bottom = flagpole_length) :
  top_to_halfmast + halfmast_to_top + top_to_halfmast + top_to_bottom = 180 := 
sorry

end flag_movement_distance_l229_229240


namespace radius_ratio_l229_229368

noncomputable def ratio_of_radii (V1 V2 : ℝ) (R : ℝ) : ℝ := 
  (V2 / V1)^(1/3) * R 

theorem radius_ratio (V1 V2 : ℝ) (π : ℝ) (R r : ℝ) :
  V1 = 450 * π → 
  V2 = 36 * π → 
  (4 / 3) * π * R^3 = V1 →
  (4 / 3) * π * r^3 = V2 →
  r / R = 1 / (12.5)^(1/3) :=
by {
  sorry
}

end radius_ratio_l229_229368


namespace greatest_distance_between_centers_l229_229328

-- Define the conditions
noncomputable def circle_radius : ℝ := 4
noncomputable def rectangle_length : ℝ := 20
noncomputable def rectangle_width : ℝ := 16

-- Define the centers of the circles
noncomputable def circle_center1 : ℝ × ℝ := (4, circle_radius)
noncomputable def circle_center2 : ℝ × ℝ := (rectangle_length - 4, circle_radius)

-- Calculate the greatest possible distance
noncomputable def distance : ℝ := Real.sqrt ((8 ^ 2) + (rectangle_width ^ 2))

-- Statement to prove
theorem greatest_distance_between_centers :
  distance = 8 * Real.sqrt 5 :=
  sorry

end greatest_distance_between_centers_l229_229328


namespace prime_719_exists_l229_229567

theorem prime_719_exists (a b c : ℕ) (ha : Nat.Prime a) (hb : Nat.Prime b) (hc : Nat.Prime c) :
  (a^4 + b^4 + c^4 - 3 = 719) → Nat.Prime (a^4 + b^4 + c^4 - 3) := sorry

end prime_719_exists_l229_229567


namespace smallest_x_for_palindrome_l229_229588

-- Define the condition for a number to be a palindrome
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Mathematically equivalent proof problem statement
theorem smallest_x_for_palindrome : ∃ (x : ℕ), x > 0 ∧ is_palindrome (x + 2345) ∧ x = 97 :=
by sorry

end smallest_x_for_palindrome_l229_229588


namespace person_B_spheres_needed_l229_229031

-- Translate conditions to Lean definitions
def sum_squares (n : ℕ) : ℕ := (n * (n + 1) * (2 * n + 1)) / 6
def sum_triangulars (m : ℕ) : ℕ := (m * (m + 1) * (m + 2)) / 6

-- Define the main theorem
theorem person_B_spheres_needed (n m : ℕ) (hA : sum_squares n = 2109)
    (hB : m ≥ 25) : sum_triangulars m = 2925 :=
    sorry

end person_B_spheres_needed_l229_229031


namespace find_m_l229_229751

theorem find_m (m x1 x2 : ℝ) (h1 : x1^2 + m * x1 + 5 = 0) (h2 : x2^2 + m * x2 + 5 = 0) (h3 : x1 = 2 * |x2| - 3) : 
  m = -9 / 2 :=
sorry

end find_m_l229_229751


namespace james_fence_problem_l229_229405

theorem james_fence_problem (w : ℝ) (hw : 0 ≤ w) (h_area : w * (2 * w + 10) ≥ 120) : w = 5 :=
by
  sorry

end james_fence_problem_l229_229405


namespace tino_more_jellybeans_than_lee_l229_229645

-- Declare the conditions
variables (arnold_jellybeans lee_jellybeans tino_jellybeans : ℕ)
variables (arnold_jellybeans_half_lee : arnold_jellybeans = lee_jellybeans / 2)
variables (arnold_jellybean_count : arnold_jellybeans = 5)
variables (tino_jellybean_count : tino_jellybeans = 34)

-- The goal is to prove how many more jellybeans Tino has than Lee
theorem tino_more_jellybeans_than_lee : tino_jellybeans - lee_jellybeans = 24 :=
by
  sorry -- proof skipped

end tino_more_jellybeans_than_lee_l229_229645


namespace drink_cost_l229_229167

/-- Wade has called into a rest stop and decides to get food for the road. 
  He buys a sandwich to eat now, one for the road, and one for the evening. 
  He also buys 2 drinks. Wade spends a total of $26 and the sandwiches 
  each cost $6. Prove that the drinks each cost $4. -/
theorem drink_cost (cost_sandwich : ℕ) (num_sandwiches : ℕ) (cost_total : ℕ) (num_drinks : ℕ) :
  cost_sandwich = 6 → num_sandwiches = 3 → cost_total = 26 → num_drinks = 2 → 
  ∃ (cost_drink : ℕ), cost_drink = 4 :=
by
  intro h1 h2 h3 h4
  sorry

end drink_cost_l229_229167


namespace average_track_width_l229_229444

theorem average_track_width (r1 r2 s1 s2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * s1 - 2 * Real.pi * s2 = 30 * Real.pi) :
  (r1 - r2 + (s1 - s2)) / 2 = 12.5 := 
sorry

end average_track_width_l229_229444


namespace min_max_a_e_l229_229390

noncomputable def find_smallest_largest (a b c d e : ℝ) : ℝ × ℝ :=
  if a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e
    then (a, e)
    else (-1, -1) -- using -1 to indicate invalid input

theorem min_max_a_e (a b c d e : ℝ) : a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e → 
    find_smallest_largest a b c d e = (a, e) :=
  by
    -- Proof to be filled in by user
    sorry

end min_max_a_e_l229_229390


namespace quadratic_term_elimination_l229_229081

theorem quadratic_term_elimination (m : ℝ) :
  (3 * (x : ℝ) ^ 2 - 10 - 2 * x - 4 * x ^ 2 + m * x ^ 2) = -(x : ℝ) * (2 * x + 10) ↔ m = 1 := 
by sorry

end quadratic_term_elimination_l229_229081


namespace no_three_distinct_integers_solving_polynomial_l229_229886

theorem no_three_distinct_integers_solving_polynomial (p : ℤ → ℤ) (hp : ∀ x, ∃ k : ℕ, p x = k • x + p 0) :
  ∀ a b c : ℤ, a ≠ b → b ≠ c → c ≠ a → p a = b → p b = c → p c = a → false :=
by
  intros a b c hab hbc hca hpa_hp pb_pc_pc
  sorry

end no_three_distinct_integers_solving_polynomial_l229_229886


namespace proof_problem_l229_229110

theorem proof_problem (x : ℝ) : (0 < x ∧ x < 5) → (x^2 - 5 * x < 0) ∧ (|x - 2| < 3) :=
by
  sorry

end proof_problem_l229_229110


namespace ratio_of_age_differences_l229_229094

variable (R J K : ℕ)

-- conditions
axiom h1 : R = J + 6
axiom h2 : R + 2 = 2 * (J + 2)
axiom h3 : (R + 2) * (K + 2) = 108

-- statement to prove
theorem ratio_of_age_differences : (R - J) = 2 * (R - K) := 
sorry

end ratio_of_age_differences_l229_229094


namespace sum_of_primes_final_sum_l229_229755

theorem sum_of_primes (p : ℕ) (hp : Nat.Prime p) :
  (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) →
  p = 2 ∨ p = 5 :=
sorry

theorem final_sum :
  (∀ p : ℕ, Nat.Prime p → (¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 6 [ZMOD p]) → p = 2 ∨ p = 5) →
  (2 + 5 = 7) :=
sorry

end sum_of_primes_final_sum_l229_229755


namespace factorize_expression_l229_229116

theorem factorize_expression (x y : ℝ) : 4 * x^2 - 2 * x * y = 2 * x * (2 * x - y) := 
by
  sorry

end factorize_expression_l229_229116


namespace mod_inverse_5_221_l229_229708

theorem mod_inverse_5_221 : ∃ x : ℤ, 0 ≤ x ∧ x < 221 ∧ (5 * x) % 221 = 1 % 221 :=
by
  use 177
  sorry

end mod_inverse_5_221_l229_229708


namespace kolya_sheets_exceed_500_l229_229415

theorem kolya_sheets_exceed_500 :
  ∃ k : ℕ, (10 + k * (k + 1) / 2 > 500) :=
sorry

end kolya_sheets_exceed_500_l229_229415


namespace parabola_inequality_l229_229586

theorem parabola_inequality (k : ℝ) : 
  (∀ x : ℝ, 2 * x^2 - 2 * k * x + (k^2 + 2 * k + 2) > x^2 + 2 * k * x - 2 * k^2 - 1) ↔ (-1 < k ∧ k < 3) := 
sorry

end parabola_inequality_l229_229586


namespace number_of_real_roots_l229_229932

noncomputable def f (x : ℝ) : ℝ := x ^ 3 - 6 * x ^ 2 + 9 * x - 10

theorem number_of_real_roots : ∃! x : ℝ, f x = 0 :=
sorry

end number_of_real_roots_l229_229932


namespace solve_equation_l229_229293

theorem solve_equation (x y : ℕ) (hx : x > 0) (hy : y > 0) :
  x^(2 * y - 1) + (x + 1)^(2 * y - 1) = (x + 2)^(2 * y - 1) ↔ (x = 1 ∧ y = 1) := by
  sorry

end solve_equation_l229_229293


namespace positive_integer_solution_l229_229975

theorem positive_integer_solution (n : ℕ) (h1 : n + 2009 ∣ n^2 + 2009) (h2 : n + 2010 ∣ n^2 + 2010) : n = 1 := 
by
  -- The proof would go here.
  sorry

end positive_integer_solution_l229_229975


namespace area_of_absolute_value_sum_l229_229597

theorem area_of_absolute_value_sum :
  ∃ area : ℝ, (area = 80) ∧ (∀ x y : ℝ, |2 * x| + |5 * y| = 20 → area = 80) :=
by
  sorry

end area_of_absolute_value_sum_l229_229597


namespace teal_sales_revenue_l229_229939

theorem teal_sales_revenue :
  let pumpkin_pie_slices := 8
  let pumpkin_pie_price := 5
  let pumpkin_pies_sold := 4
  let custard_pie_slices := 6
  let custard_pie_price := 6
  let custard_pies_sold := 5
  let apple_pie_slices := 10
  let apple_pie_price := 4
  let apple_pies_sold := 3
  let pecan_pie_slices := 12
  let pecan_pie_price := 7
  let pecan_pies_sold := 2
  (pumpkin_pie_slices * pumpkin_pie_price * pumpkin_pies_sold) +
  (custard_pie_slices * custard_pie_price * custard_pies_sold) +
  (apple_pie_slices * apple_pie_price * apple_pies_sold) +
  (pecan_pie_slices * pecan_pie_price * pecan_pies_sold) = 
  628 := by
  sorry

end teal_sales_revenue_l229_229939


namespace abs_inequalities_imply_linear_relationship_l229_229646

theorem abs_inequalities_imply_linear_relationship (a b c : ℝ)
(h1 : |a - b| ≥ |c|)
(h2 : |b - c| ≥ |a|)
(h3 : |c - a| ≥ |b|) :
a = b + c ∨ b = c + a ∨ c = a + b :=
sorry

end abs_inequalities_imply_linear_relationship_l229_229646


namespace p_squared_plus_41_composite_for_all_primes_l229_229639

theorem p_squared_plus_41_composite_for_all_primes (p : ℕ) (hp : Prime p) : 
  ∃ d : ℕ, d > 1 ∧ d < p^2 + 41 ∧ d ∣ (p^2 + 41) :=
by
  sorry

end p_squared_plus_41_composite_for_all_primes_l229_229639


namespace intersection_is_A_l229_229995

-- Define the set M based on the given condition
def M : Set ℝ := {x | x / (x - 1) ≥ 0}

-- Define the set N based on the given condition
def N : Set ℝ := {x | ∃ y, y = 3 * x^2 + 1}

-- Define the set A as the intersection of M and N
def A : Set ℝ := {x | x > 1}

-- Prove that the intersection of M and N is equal to the set A
theorem intersection_is_A : (M ∩ N = A) :=
by {
  sorry
}

end intersection_is_A_l229_229995


namespace find_additional_fuel_per_person_l229_229732

def num_passengers : ℕ := 30
def num_crew : ℕ := 5
def num_people : ℕ := num_passengers + num_crew
def num_bags_per_person : ℕ := 2
def num_bags : ℕ := num_people * num_bags_per_person
def fuel_empty_plane : ℕ := 20
def fuel_per_bag : ℕ := 2
def total_trip_fuel : ℕ := 106000
def trip_distance : ℕ := 400
def fuel_per_mile : ℕ := total_trip_fuel / trip_distance

def additional_fuel_per_person (x : ℕ) : Prop :=
  fuel_empty_plane + num_people * x + num_bags * fuel_per_bag = fuel_per_mile

theorem find_additional_fuel_per_person : additional_fuel_per_person 3 :=
  sorry

end find_additional_fuel_per_person_l229_229732


namespace case1_DC_correct_case2_DC_correct_l229_229656

-- Case 1
theorem case1_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 10) (hAD : AD = 4)
  (hHM : HM = 6 / 5) (hBD : BD = 2 * Real.sqrt 21) (hDH : DH = 4 * Real.sqrt 21 / 5)
  (hMD : MD = 6 * (Real.sqrt 21 - 1) / 5):
  (BD - HM : ℝ) == (8 * Real.sqrt 21 - 12) / 5 :=
by {
  sorry
}

-- Case 2
theorem case2_DC_correct (AB AD : ℝ) (HM BD DH MD : ℝ) (hAB : AB = 8 * Real.sqrt 2) (hAD : AD = 4)
  (hHM : HM = Real.sqrt 2) (hBD : BD = 4 * Real.sqrt 7) (hDH : DH = Real.sqrt 14)
  (hMD : MD = Real.sqrt 14 - Real.sqrt 2):
  (BD - HM : ℝ) == 2 * Real.sqrt 14 - 2 * Real.sqrt 2 :=
by {
  sorry
}

end case1_DC_correct_case2_DC_correct_l229_229656


namespace value_of_expression_l229_229342

theorem value_of_expression : (4 * 3) + 2 = 14 := by
  sorry

end value_of_expression_l229_229342


namespace sequence_diff_l229_229268

theorem sequence_diff (a : ℕ → ℕ) (S : ℕ → ℕ)
  (hSn : ∀ n, S n = n^2)
  (hS1 : a 1 = S 1)
  (ha_n : ∀ n, a (n + 1) = S (n + 1) - S n) :
  a 3 - a 2 = 2 := sorry

end sequence_diff_l229_229268


namespace perfect_square_factors_450_l229_229315

def prime_factorization (n : ℕ) : Prop :=
  n = 2^1 * 3^2 * 5^2

theorem perfect_square_factors_450 : prime_factorization 450 → ∃ n : ℕ, n = 4 :=
by
  sorry

end perfect_square_factors_450_l229_229315


namespace fraction_simplification_l229_229965

theorem fraction_simplification : 
  (1877^2 - 1862^2) / (1880^2 - 1859^2) = 5 / 7 := 
by 
  sorry

end fraction_simplification_l229_229965


namespace arithmetic_expression_l229_229526

theorem arithmetic_expression : (-9) + 18 + 2 + (-1) = 10 :=
by 
  sorry

end arithmetic_expression_l229_229526


namespace probability_obtuse_triangle_is_one_fourth_l229_229288

-- Define the set of possible integers
def S : Set ℤ := {1, 2, 3, 4, 5, 6}

-- Condition for forming an obtuse triangle
def is_obtuse_triangle (a b c : ℤ) : Prop :=
  a < b + c ∧ b < a + c ∧ c < a + b ∧ 
  (a^2 + b^2 < c^2 ∨ a^2 + c^2 < b^2 ∨ b^2 + c^2 < a^2)

-- List of valid triples that can form an obtuse triangle
def valid_obtuse_triples : List (ℤ × ℤ × ℤ) :=
  [(2, 3, 4), (2, 4, 5), (2, 5, 6), (3, 4, 6), (3, 5, 6)]

-- Total number of combinations
def total_combinations : Nat := 20

-- Number of valid combinations for obtuse triangles
def valid_combinations : Nat := 5

-- Calculate the probability
def probability_obtuse_triangle : ℚ := valid_combinations / total_combinations

theorem probability_obtuse_triangle_is_one_fourth :
  probability_obtuse_triangle = 1 / 4 :=
by
  sorry

end probability_obtuse_triangle_is_one_fourth_l229_229288


namespace find_pairs_l229_229861

theorem find_pairs (x y : ℤ) (h : 19 / x + 96 / y = (19 * 96) / (x * y)) :
  ∃ m : ℤ, x = 19 * m ∧ y = 96 - 96 * m :=
by
  sorry

end find_pairs_l229_229861


namespace calculate_value_l229_229493

def a : ℤ := 3 * 4 * 5
def b : ℚ := 1/3 + 1/4 + 1/5

theorem calculate_value :
  (a : ℚ) * b = 47 := by
sorry

end calculate_value_l229_229493


namespace sum_of_integers_l229_229298

theorem sum_of_integers : (∀ (x y : ℤ), x = -4 ∧ y = -5 ∧ x - y = 1 → x + y = -9) := 
by 
  intros x y
  sorry

end sum_of_integers_l229_229298


namespace length_of_ae_l229_229904

-- Definition of points and lengths between them
variables (a b c d e : Type)
variables (bc cd de ab ac : ℝ)

-- Given conditions
axiom H1 : bc = 3 * cd
axiom H2 : de = 8
axiom H3 : ab = 5
axiom H4 : ac = 11
axiom H5 : bc = ac - ab
axiom H6 : cd = bc / 3

-- Theorem to prove
theorem length_of_ae : ∀ ab bc cd de : ℝ, ae = ab + bc + cd + de := by
  sorry

end length_of_ae_l229_229904


namespace solution_set_of_inequality_l229_229072

theorem solution_set_of_inequality (x : ℝ) :
  x * |x - 1| > 0 ↔ (0 < x ∧ x < 1) ∨ (x > 1) :=
sorry

end solution_set_of_inequality_l229_229072


namespace triangle_area_is_15_l229_229506

def Point := (ℝ × ℝ)

def A : Point := (2, 2)
def B : Point := (7, 2)
def C : Point := (4, 8)

def area_of_triangle (A B C : Point) : ℝ :=
  0.5 * (B.1 - A.1) * (C.2 - A.2)

theorem triangle_area_is_15 : area_of_triangle A B C = 15 :=
by
  -- The proof goes here
  sorry

end triangle_area_is_15_l229_229506


namespace inequality_proof_l229_229517

theorem inequality_proof (a b : ℝ) (h : a + b > 0) :
  (a / (b^2) + b / (a^2) ≥ 1 / a + 1 / b) :=
by
  sorry

end inequality_proof_l229_229517


namespace expression_evaluate_l229_229095

theorem expression_evaluate :
  50 * (50 - 5) - (50 * 50 - 5) = -245 :=
by
  sorry

end expression_evaluate_l229_229095


namespace scientific_notation_of_30067_l229_229318

theorem scientific_notation_of_30067 : ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 30067 = a * 10^n := by
  use 3.0067
  use 4
  sorry

end scientific_notation_of_30067_l229_229318


namespace chicken_price_reaches_81_in_2_years_l229_229019

theorem chicken_price_reaches_81_in_2_years :
  ∃ t : ℝ, (t / 12 = 2) ∧ (∃ n : ℕ, (3:ℝ)^(n / 6) = 81 ∧ n = t) :=
by
  sorry

end chicken_price_reaches_81_in_2_years_l229_229019


namespace sum_max_min_ratios_l229_229133

theorem sum_max_min_ratios
  (c d : ℚ)
  (h1 : ∀ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 → y / x = c ∨ y / x = d)
  (h2 : ∀ r : ℚ, (∃ x y : ℚ, 3*x^2 + 2*x*y + 4*y^2 - 13*x - 26*y + 53 = 0 ∧ y / x = r) → (r = c ∨ r = d))
  : c + d = 63 / 43 :=
sorry

end sum_max_min_ratios_l229_229133


namespace fourth_term_of_geometric_sequence_l229_229946

theorem fourth_term_of_geometric_sequence (a₁ : ℝ) (a₆ : ℝ) (a₄ : ℝ) (r : ℝ)
  (h₁ : a₁ = 1000)
  (h₂ : a₆ = a₁ * r^5)
  (h₃ : a₆ = 125)
  (h₄ : a₄ = a₁ * r^3) : 
  a₄ = 125 :=
sorry

end fourth_term_of_geometric_sequence_l229_229946


namespace price_per_pie_l229_229845

-- Define the relevant variables and conditions
def cost_pumpkin_pie : ℕ := 3
def num_pumpkin_pies : ℕ := 10
def cost_cherry_pie : ℕ := 5
def num_cherry_pies : ℕ := 12
def desired_profit : ℕ := 20

-- Total production and profit calculation
def total_cost : ℕ := (cost_pumpkin_pie * num_pumpkin_pies) + (cost_cherry_pie * num_cherry_pies)
def total_earnings_needed : ℕ := total_cost + desired_profit
def total_pies : ℕ := num_pumpkin_pies + num_cherry_pies

-- Proposition to prove that the price per pie should be $5
theorem price_per_pie : (total_earnings_needed / total_pies) = 5 := by
  sorry

end price_per_pie_l229_229845


namespace find_numbers_l229_229894

theorem find_numbers (x y z t : ℕ) 
  (h1 : x + t = 37) 
  (h2 : y + z = 36) 
  (h3 : x + z = 2 * y) 
  (h4 : y * t = z * z) : 
  x = 12 ∧ y = 16 ∧ z = 20 ∧ t = 25 :=
by
  sorry

end find_numbers_l229_229894


namespace positive_value_of_A_l229_229497

-- Define the relation
def hash (A B : ℝ) : ℝ := A^2 - B^2

-- State the main theorem
theorem positive_value_of_A (A : ℝ) : hash A 7 = 72 → A = 11 :=
by
  -- Placeholder for the proof
  sorry

end positive_value_of_A_l229_229497


namespace stella_weeks_l229_229191

-- Define the constants used in the conditions
def rolls_per_bathroom_per_day : ℕ := 1
def bathrooms : ℕ := 6
def days_per_week : ℕ := 7
def rolls_per_pack : ℕ := 12
def packs_bought : ℕ := 14

-- Define the total number of rolls Stella uses per day and per week
def rolls_per_day := rolls_per_bathroom_per_day * bathrooms
def rolls_per_week := rolls_per_day * days_per_week

-- Calculate the total number of rolls bought
def total_rolls_bought := packs_bought * rolls_per_pack

-- Calculate the number of weeks Stella bought toilet paper for
def weeks := total_rolls_bought / rolls_per_week

theorem stella_weeks : weeks = 4 := by
  sorry

end stella_weeks_l229_229191


namespace variance_of_data_l229_229285

theorem variance_of_data :
  let data := [3, 1, 0, -1, -3]
  let mean := (3 + 1 + 0 - 1 - 3) / (5:ℝ)
  let variance := (1 / 5:ℝ) * (3^2 + 1^2 + (-1)^2 + (-3)^2)
  variance = 4 := sorry

end variance_of_data_l229_229285


namespace largest_four_digit_number_property_l229_229107

theorem largest_four_digit_number_property : ∃ (a b c d : Nat), 
  (1000 * a + 100 * b + 10 * c + d = 9099) ∧ 
  (c = a + b) ∧ 
  (d = b + c) ∧ 
  1 ≤ a ∧ a ≤ 9 ∧ 
  0 ≤ b ∧ b ≤ 9 ∧ 
  0 ≤ c ∧ c ≤ 9 ∧ 
  0 ≤ d ∧ d ≤ 9 := 
sorry

end largest_four_digit_number_property_l229_229107


namespace triangle_side_length_l229_229865

theorem triangle_side_length 
  (A : ℝ) (a m n : ℝ) 
  (hA : A = 60) 
  (h1 : m + n = 7) 
  (h2 : m * n = 11) : a = 4 :=
by
  sorry

end triangle_side_length_l229_229865


namespace jackson_total_souvenirs_l229_229329

-- Define the conditions
def num_hermit_crabs : ℕ := 45
def spiral_shells_per_hermit_crab : ℕ := 3
def starfish_per_spiral_shell : ℕ := 2

-- Define the calculation based on the conditions
def num_spiral_shells := num_hermit_crabs * spiral_shells_per_hermit_crab
def num_starfish := num_spiral_shells * starfish_per_spiral_shell
def total_souvenirs := num_hermit_crabs + num_spiral_shells + num_starfish

-- Prove that the total number of souvenirs is 450
theorem jackson_total_souvenirs : total_souvenirs = 450 :=
by
  sorry

end jackson_total_souvenirs_l229_229329


namespace part1_part2_l229_229237

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 1

theorem part1 :
  ∀ x : ℝ, f (x + π) = f x :=
by sorry

theorem part2 :
  ∃ (max_x min_x : ℝ), max_x ∈ Set.Icc (π/12) (π/4) ∧ min_x ∈ Set.Icc (π/12) (π/4) ∧
    f max_x = 7 / 4 ∧ f min_x = (5 + Real.sqrt 3) / 4 ∧
    (max_x = π / 6) ∧ (min_x = π / 12 ∨ min_x = π / 4) :=
by sorry

end part1_part2_l229_229237


namespace solution_set_correct_l229_229446

noncomputable def solution_set (x : ℝ) : Prop :=
  x + 2 / (x + 1) > 2

theorem solution_set_correct :
  {x : ℝ | solution_set x} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_correct_l229_229446


namespace shenille_scores_points_l229_229706

theorem shenille_scores_points :
  ∀ (x y : ℕ), (x + y = 45) → (x = 2 * y) → 
  (25/100 * x + 40/100 * y) * 3 + (40/100 * y) * 2 = 33 :=
by 
  intros x y h1 h2
  sorry

end shenille_scores_points_l229_229706


namespace correct_operation_only_l229_229058

theorem correct_operation_only (a b x y : ℝ) : 
  (2 + Real.sqrt 2 ≠ 2 * Real.sqrt 2) ∧ 
  (4 * x^2 * y - x^2 * y ≠ 3) ∧ 
  ((a + b)^2 ≠ a^2 + b^2) ∧
  ((ab)^3 = a^3 * b^3) := 
by 
  sorry

end correct_operation_only_l229_229058


namespace seashell_count_l229_229320

variable (initial_seashells additional_seashells total_seashells : ℕ)

theorem seashell_count (h1 : initial_seashells = 19) (h2 : additional_seashells = 6) : 
  total_seashells = initial_seashells + additional_seashells → total_seashells = 25 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end seashell_count_l229_229320


namespace average_not_1380_l229_229643

-- Define the set of numbers
def numbers := [1200, 1400, 1510, 1520, 1530, 1200]

-- Define the claimed average
def claimed_avg := 1380

-- The sum of the numbers
def sumNumbers := numbers.sum

-- The number of items in the set
def countNumbers := numbers.length

-- The correct average calculation
def correct_avg : ℚ := sumNumbers / countNumbers

-- The proof problem: proving that the correct average is not equal to the claimed average
theorem average_not_1380 : correct_avg ≠ claimed_avg := by
  sorry

end average_not_1380_l229_229643


namespace focus_of_parabola_l229_229844

theorem focus_of_parabola (x y : ℝ) (h : x^2 = 16 * y) : (0, 4) = (0, 4) :=
by {
  sorry
}

end focus_of_parabola_l229_229844


namespace right_triangle_similarity_l229_229872

theorem right_triangle_similarity (y : ℝ) (h : 12 / y = 9 / 7) : y = 9.33 := 
by 
  sorry

end right_triangle_similarity_l229_229872


namespace sum_of_squares_of_consecutive_even_numbers_l229_229883

theorem sum_of_squares_of_consecutive_even_numbers :
  ∃ (x : ℤ), x + (x + 2) + (x + 4) + (x + 6) = 36 → (x ^ 2 + (x + 2) ^ 2 + (x + 4) ^ 2 + (x + 6) ^ 2 = 344) :=
by
  sorry

end sum_of_squares_of_consecutive_even_numbers_l229_229883


namespace sector_area_l229_229954

theorem sector_area (r : ℝ) (alpha : ℝ) (h_r : r = 2) (h_alpha : alpha = π / 4) : 
  (1 / 2) * r^2 * alpha = π / 2 :=
by
  rw [h_r, h_alpha]
  -- proof steps would go here
  sorry

end sector_area_l229_229954


namespace chess_tournament_distribution_l229_229828

theorem chess_tournament_distribution 
    (students : List String)
    (games_played : Nat)
    (scores : List ℝ)
    (points_per_game : List ℝ)
    (unique_scores : ∀ (x y : ℝ), x ≠ y → scores.contains x → scores.contains y → x ≠ y)
    (first_place : String)
    (second_place : String)
    (third_place : String)
    (fourth_place : String)
    (andrey_wins_equal_sasha : ℝ)
    (total_points : ℝ)
    : 
    students = ["Andrey", "Vanya", "Dima", "Sasha"] ∧
    games_played = 6 ∧
    points_per_game = [1, 0.5, 0] ∧
    first_place = "Andrey" ∧
    second_place = "Dima" ∧
    third_place = "Vanya" ∧
    fourth_place = "Sasha" ∧
    scores = [4, 3.5, 2.5, 2] ∧
    andrey_wins_equal_sasha = 2 ∧
    total_points = 12 := 
sorry

end chess_tournament_distribution_l229_229828


namespace range_of_f_l229_229397

noncomputable def f (x : ℤ) : ℤ := x ^ 2 + 1

def domain : Set ℤ := {-1, 0, 1, 2}

def range_f : Set ℤ := {1, 2, 5}

theorem range_of_f : Set.image f domain = range_f :=
by
  sorry

end range_of_f_l229_229397


namespace max_acute_triangles_l229_229236

theorem max_acute_triangles (n : ℕ) (hn : n ≥ 3) :
  (∃ k, k = if n % 2 = 0 then (n * (n-2) * (n+2)) / 24 else (n * (n-1) * (n+1)) / 24) :=
by 
  sorry

end max_acute_triangles_l229_229236


namespace unattainable_y_l229_229153

theorem unattainable_y (x : ℝ) (hx : x ≠ -2 / 3) : ¬ (∃ x, y = (x - 3) / (3 * x + 2) ∧ y = 1 / 3) := by
  sorry

end unattainable_y_l229_229153


namespace trigonometric_expression_proof_l229_229305

theorem trigonometric_expression_proof :
  (Real.cos (76 * Real.pi / 180) * Real.cos (16 * Real.pi / 180) +
   Real.cos (14 * Real.pi / 180) * Real.cos (74 * Real.pi / 180) -
   2 * Real.cos (75 * Real.pi / 180) * Real.cos (15 * Real.pi / 180)) = 0 :=
by
  sorry

end trigonometric_expression_proof_l229_229305


namespace marbles_per_friend_l229_229267

theorem marbles_per_friend (total_marbles friends : ℕ) (h1 : total_marbles = 5504) (h2 : friends = 64) :
  total_marbles / friends = 86 :=
by {
  -- Proof will be added here
  sorry
}

end marbles_per_friend_l229_229267


namespace intersection_A_B_l229_229027

def set_A : Set ℝ := { x | x ≥ 0 }
def set_B : Set ℝ := { x | -1 < x ∧ x < 1 }

theorem intersection_A_B : set_A ∩ set_B = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end intersection_A_B_l229_229027


namespace chloe_first_round_points_l229_229940

variable (P : ℤ)
variable (totalPoints : ℤ := 86)
variable (secondRoundPoints : ℤ := 50)
variable (lastRoundLoss : ℤ := 4)

theorem chloe_first_round_points 
  (h : P + secondRoundPoints - lastRoundLoss = totalPoints) : 
  P = 40 := by
  sorry

end chloe_first_round_points_l229_229940


namespace total_words_in_week_l229_229465

def typing_minutes_MWF : ℤ := 260
def typing_minutes_TTh : ℤ := 150
def typing_minutes_Sat : ℤ := 240
def typing_speed_MWF : ℤ := 50
def typing_speed_TTh : ℤ := 40
def typing_speed_Sat : ℤ := 60

def words_per_day_MWF : ℤ := typing_minutes_MWF * typing_speed_MWF
def words_per_day_TTh : ℤ := typing_minutes_TTh * typing_speed_TTh
def words_Sat : ℤ := typing_minutes_Sat * typing_speed_Sat

def total_words_week : ℤ :=
  (words_per_day_MWF * 3) + (words_per_day_TTh * 2) + words_Sat + 0

theorem total_words_in_week :
  total_words_week = 65400 :=
by
  sorry

end total_words_in_week_l229_229465


namespace intersection_A_B_l229_229388

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

-- Define the intersection we need to prove
def A_cap_B_target : Set ℝ := {-2, -1, 0, 1}

-- Prove the intersection of A and B equals the target set
theorem intersection_A_B :
  A ∩ B = A_cap_B_target := 
sorry

end intersection_A_B_l229_229388


namespace S15_constant_l229_229296

-- Define the arithmetic sequence a_n
noncomputable def a (n : ℕ) (a1 d : ℤ) : ℤ := a1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S (n : ℕ) (a1 d : ℤ) : ℤ := n * (2 * a1 + (n - 1) * d) / 2

-- Given condition: a_5 + a_8 + a_11 is constant
axiom const_sum : ∀ (a1 d : ℤ), a 5 a1 d + a 8 a1 d + a 11 a1 d = 3 * a1 + 21 * d

-- The equivalent proof problem
theorem S15_constant (a1 d : ℤ) : S 15 a1 d = 5 * (3 * a1 + 21 * d) :=
by
  sorry

end S15_constant_l229_229296


namespace initial_innings_l229_229927

/-- The number of innings a player played initially given the conditions described in the problem. -/
theorem initial_innings (n : ℕ)
  (average_runs : ℕ)
  (additional_runs : ℕ)
  (new_average_increase : ℕ)
  (h1 : average_runs = 42)
  (h2 : additional_runs = 86)
  (h3 : new_average_increase = 4) :
  42 * n + 86 = 46 * (n + 1) → n = 10 :=
by
  intros h
  linarith

end initial_innings_l229_229927


namespace proportion_is_equation_l229_229169

/-- A proportion containing unknowns is an equation -/
theorem proportion_is_equation (P : Prop) (contains_equality_sign: Prop)
  (indicates_equality : Prop)
  (contains_unknowns : Prop) : (contains_equality_sign ∧ indicates_equality ∧ contains_unknowns ↔ True) := by
  sorry

end proportion_is_equation_l229_229169


namespace intersection_AB_l229_229892

/-- Define the set A based on the given condition -/
def setA : Set ℝ := {x | 2 * x ^ 2 + x > 0}

/-- Define the set B based on the given condition -/
def setB : Set ℝ := {x | 2 * x + 1 > 0}

/-- Prove that A ∩ B = {x | x > 0} -/
theorem intersection_AB : (setA ∩ setB) = {x | x > 0} :=
sorry

end intersection_AB_l229_229892


namespace decreasing_interval_f_l229_229756

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem decreasing_interval_f :
  ∀ x : ℝ, x > 0 → (f (x) = (1 / 2) * x^2 - Real.log x) →
  (∃ a b : ℝ, 0 < a ∧ a ≤ b ∧ b = 1 ∧ ∀ y, a < y ∧ y ≤ b → f (y) ≤ f (y+1)) := sorry

end decreasing_interval_f_l229_229756


namespace distinct_factors_1320_l229_229669

theorem distinct_factors_1320 : ∃ (n : ℕ), (n = 24) ∧ (∀ d : ℕ, d ∣ 1320 → d.divisors.card = n) ∧ (1320 = 2^2 * 3 * 5 * 11) :=
sorry

end distinct_factors_1320_l229_229669


namespace equivalent_expression_l229_229900

-- Let a, b, c, d, e be real numbers
variables (a b c d e : ℝ)

-- Condition given in the problem
def condition : Prop := 81 * a - 27 * b + 9 * c - 3 * d + e = -5

-- Objective: Prove that 8 * a - 4 * b + 2 * c - d + e = -5 given the condition
theorem equivalent_expression (h : condition a b c d e) : 8 * a - 4 * b + 2 * c - d + e = -5 :=
sorry

end equivalent_expression_l229_229900


namespace probability_divisor_of_60_l229_229565

theorem probability_divisor_of_60 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 60 ∧ (∃ a b c : ℕ, n = 2 ^ a * 3 ^ b * 5 ^ c ∧ a ≤ 2 ∧ b ≤ 1 ∧ c ≤ 1)) → 
  ∃ p : ℚ, p = 1 / 5 :=
by
  sorry

end probability_divisor_of_60_l229_229565


namespace find_p_q_sum_l229_229885

variable (P Q x : ℝ)

theorem find_p_q_sum (h : (P / (x - 3)) +  Q * (x + 2) = (-5 * x^2 + 18 * x + 40) / (x - 3)) : P + Q = 20 :=
sorry

end find_p_q_sum_l229_229885


namespace find_d_in_triangle_ABC_l229_229221

theorem find_d_in_triangle_ABC (AB BC AC : ℝ) (P : Type) (d : ℝ) 
  (h_AB : AB = 480) (h_BC : BC = 500) (h_AC : AC = 550)
  (h_segments_equal : ∀ (D D' E E' F F' : Type), true) : 
  d = 132000 / 654 :=
sorry

end find_d_in_triangle_ABC_l229_229221


namespace a7_equals_21_l229_229028

-- Define the sequence {a_n} recursively
def seq (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | (n + 2) => seq n + seq (n + 1)

-- Statement to prove that a_7 = 21
theorem a7_equals_21 : seq 6 = 21 := 
  sorry

end a7_equals_21_l229_229028


namespace john_paid_correct_amount_l229_229557

def cost_bw : ℝ := 160
def markup_percentage : ℝ := 0.5

def cost_color : ℝ := cost_bw * (1 + markup_percentage)

theorem john_paid_correct_amount : 
  cost_color = 240 := 
by
  -- proof required here
  sorry

end john_paid_correct_amount_l229_229557


namespace fourth_game_water_correct_fourth_game_sports_drink_l229_229190

noncomputable def total_bottled_water_cases : ℕ := 10
noncomputable def total_sports_drink_cases : ℕ := 5
noncomputable def bottles_per_case_water : ℕ := 20
noncomputable def bottles_per_case_sports_drink : ℕ := 15
noncomputable def initial_bottled_water : ℕ := total_bottled_water_cases * bottles_per_case_water
noncomputable def initial_sports_drinks : ℕ := total_sports_drink_cases * bottles_per_case_sports_drink

noncomputable def first_game_water : ℕ := 70
noncomputable def first_game_sports_drink : ℕ := 30
noncomputable def second_game_water : ℕ := 40
noncomputable def second_game_sports_drink : ℕ := 20
noncomputable def third_game_water : ℕ := 50
noncomputable def third_game_sports_drink : ℕ := 25

noncomputable def total_consumed_water : ℕ := first_game_water + second_game_water + third_game_water
noncomputable def total_consumed_sports_drink : ℕ := first_game_sports_drink + second_game_sports_drink + third_game_sports_drink

noncomputable def remaining_water_before_fourth_game : ℕ := initial_bottled_water - total_consumed_water
noncomputable def remaining_sports_drink_before_fourth_game : ℕ := initial_sports_drinks - total_consumed_sports_drink

noncomputable def remaining_water_after_fourth_game : ℕ := 20
noncomputable def remaining_sports_drink_after_fourth_game : ℕ := 10

noncomputable def fourth_game_water_consumed : ℕ := remaining_water_before_fourth_game - remaining_water_after_fourth_game

theorem fourth_game_water_correct : fourth_game_water_consumed = 20 :=
by
  unfold fourth_game_water_consumed remaining_water_before_fourth_game
  sorry

theorem fourth_game_sports_drink : false :=
by
  sorry

end fourth_game_water_correct_fourth_game_sports_drink_l229_229190


namespace vera_first_place_l229_229507

noncomputable def placement (anna vera katya natasha : ℕ) : Prop :=
  (anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)

theorem vera_first_place :
  ∃ (anna vera katya natasha : ℕ),
    (placement anna vera katya natasha) ∧ 
    (vera = 1) ∧ 
    (1 ≠ 4) → 
    ((anna ≠ 1 ∧ anna ≠ 4) ∧ (vera ≠ 4) ∧ (katya = 1) ∧ (natasha = 4)) ∧ 
    (1 = 1) ∧ 
    (∃ i j k l : ℕ, (i ≠ 1 ∧ i ≠ 4) ∧ (j = 1) ∧ (k ≠ 1) ∧ (l = 4)) ∧ 
    (vera = 1) :=
sorry

end vera_first_place_l229_229507


namespace Jordana_current_age_is_80_l229_229882

-- Given conditions
def current_age_Jennifer := 20  -- since Jennifer will be 30 in ten years
def current_age_Jordana := 80  -- since the problem states we need to verify this

-- Prove that Jordana's current age is 80 years old given the conditions
theorem Jordana_current_age_is_80:
  (current_age_Jennifer + 10 = 30) →
  (current_age_Jordana + 10 = 3 * 30) →
  current_age_Jordana = 80 :=
by 
  intros h1 h2
  sorry

end Jordana_current_age_is_80_l229_229882


namespace total_respondents_l229_229474

theorem total_respondents (X Y : ℕ) (hX : X = 360) (h_ratio : 9 * Y = X) : X + Y = 400 := by
  sorry

end total_respondents_l229_229474


namespace problem1_problem2_l229_229238

-- Problem 1
theorem problem1 (a b : ℝ) : 
  a^2 * (2 * a * b - 1) + (a - 3 * b) * (a + b) = 2 * a^3 * b - 2 * a * b - 3 * b^2 :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : 
  (2 * x - 3)^2 - (x + 2)^2 = 3 * x^2 - 16 * x + 5 :=
by sorry

end problem1_problem2_l229_229238


namespace point_P_quadrant_l229_229731

theorem point_P_quadrant 
  (h1 : Real.sin (θ / 2) = 3 / 5) 
  (h2 : Real.cos (θ / 2) = -4 / 5) : 
  (0 < Real.cos θ) ∧ (Real.sin θ < 0) :=
by
  sorry

end point_P_quadrant_l229_229731


namespace race_head_start_l229_229500

theorem race_head_start (v_A v_B : ℕ) (h : v_A = 4 * v_B) (d : ℕ) : 
  100 / v_A = (100 - d) / v_B → d = 75 :=
by
  sorry

end race_head_start_l229_229500


namespace tanya_efficiency_greater_sakshi_l229_229155

theorem tanya_efficiency_greater_sakshi (S_e T_e : ℝ) (h1 : S_e = 1 / 20) (h2 : T_e = 1 / 16) :
  ((T_e - S_e) / S_e) * 100 = 25 := by
  sorry

end tanya_efficiency_greater_sakshi_l229_229155


namespace actual_tax_equals_600_l229_229066

-- Definition for the first condition: initial tax amount
variable (a : ℝ)

-- Define the first reduction: 25% reduction
def first_reduction (a : ℝ) : ℝ := 0.75 * a

-- Define the second reduction: further 20% reduction
def second_reduction (tax_after_first_reduction : ℝ) : ℝ := 0.80 * tax_after_first_reduction

-- Define the final reduction: combination of both reductions
def final_tax (a : ℝ) : ℝ := second_reduction (first_reduction a)

-- Proof that with a = 1000, the actual tax is 600 million euros
theorem actual_tax_equals_600 (a : ℝ) (h₁ : a = 1000) : final_tax a = 600 := by
    rw [h₁]
    simp [final_tax, first_reduction, second_reduction]
    sorry

end actual_tax_equals_600_l229_229066


namespace find_coefficient_of_x_in_expansion_l229_229842

noncomputable def coefficient_of_x_in_expansion (x : ℤ) : ℤ :=
  (1 / 2 * x - 1) * (2 * x - 1 / x) ^ 6

theorem find_coefficient_of_x_in_expansion :
  coefficient_of_x_in_expansion x = -80 :=
by {
  sorry
}

end find_coefficient_of_x_in_expansion_l229_229842


namespace area_PST_is_5_l229_229971

noncomputable def area_of_triangle_PST 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : ℝ := 
  5

theorem area_PST_is_5 
  (P Q R S T : Point)
  (PQ QR PR : ℝ) 
  (PS PT : ℝ)
  (hPQ : PQ = 8)
  (hQR : QR = 9)
  (hPR : PR = 10)
  (hPS : PS = 3)
  (hPT : PT = 5)
  : area_of_triangle_PST P Q R S T PQ QR PR PS PT hPQ hQR hPR hPS hPT = 5 :=
sorry

end area_PST_is_5_l229_229971


namespace largest_repeating_number_l229_229984

theorem largest_repeating_number :
  ∃ n, n * 365 = 273863 * 365 := sorry

end largest_repeating_number_l229_229984


namespace biscuits_afternoon_eq_40_l229_229044

-- Define the initial conditions given in the problem.
def butter_cookies_afternoon : Nat := 10
def additional_biscuits : Nat := 30

-- Define the number of biscuits based on the initial conditions.
def biscuits_afternoon : Nat := butter_cookies_afternoon + additional_biscuits

-- The statement to prove according to the problem.
theorem biscuits_afternoon_eq_40 : biscuits_afternoon = 40 := by
  -- The proof is to be done, hence we use 'sorry'.
  sorry

end biscuits_afternoon_eq_40_l229_229044


namespace right_angled_trapezoid_base_height_l229_229700

theorem right_angled_trapezoid_base_height {a b : ℝ} (h : a = b) :
  ∃ (base height : ℝ), base = a ∧ height = b := 
by
  sorry

end right_angled_trapezoid_base_height_l229_229700


namespace largest_n_arithmetic_sequences_l229_229290

theorem largest_n_arithmetic_sequences
  (a : ℕ → ℤ) (b : ℕ → ℤ) (x y : ℤ)
  (a_1 : a 1 = 2) (b_1 : b 1 = 3)
  (a_formula : ∀ n : ℕ, a n = 2 + (n - 1) * x)
  (b_formula : ∀ n : ℕ, b n = 3 + (n - 1) * y)
  (x_lt_y : x < y)
  (product_condition : ∃ n : ℕ, a n * b n = 1638) :
  ∃ n : ℕ, a n * b n = 1638 ∧ n = 35 := 
sorry

end largest_n_arithmetic_sequences_l229_229290


namespace shortest_segment_length_l229_229779

theorem shortest_segment_length :
  let total_length := 1
  let red_dot := 0.618
  let yellow_dot := total_length - red_dot  -- yellow_dot is at the same point after fold
  let first_cut := red_dot  -- Cut the strip at the red dot
  let remaining_strip := red_dot
  let distance_between_red_and_yellow := total_length - 2 * yellow_dot
  let second_cut := distance_between_red_and_yellow
  let shortest_segment := remaining_strip - 2 * distance_between_red_and_yellow
  shortest_segment = 0.146 :=
by
  sorry

end shortest_segment_length_l229_229779


namespace composite_number_iff_ge_2_l229_229490

theorem composite_number_iff_ge_2 (n : ℕ) : 
  ¬(Prime (3^(2*n+1) - 2^(2*n+1) - 6^n)) ↔ n ≥ 2 := by
  sorry

end composite_number_iff_ge_2_l229_229490


namespace total_blood_cells_correct_l229_229680

def first_sample : ℕ := 4221
def second_sample : ℕ := 3120
def total_blood_cells : ℕ := first_sample + second_sample

theorem total_blood_cells_correct : total_blood_cells = 7341 := by
  -- proof goes here
  sorry

end total_blood_cells_correct_l229_229680


namespace hotel_people_per_room_l229_229789

theorem hotel_people_per_room
  (total_rooms : ℕ := 10)
  (towels_per_person : ℕ := 2)
  (total_towels : ℕ := 60) :
  (total_towels / towels_per_person) / total_rooms = 3 :=
by
  sorry

end hotel_people_per_room_l229_229789


namespace count_negative_rationals_is_two_l229_229403

theorem count_negative_rationals_is_two :
  let a := (-1 : ℚ) ^ 2007
  let b := (|(-1 : ℚ)| ^ 3)
  let c := -(1 : ℚ) ^ 18
  let d := (18 : ℚ)
  (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) = 2 := by
  sorry

end count_negative_rationals_is_two_l229_229403


namespace speed_in_still_water_l229_229170

-- Definitions for the conditions
def upstream_speed : ℕ := 30
def downstream_speed : ℕ := 60

-- Prove that the speed of the man in still water is 45 kmph
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 45 := by
  sorry

end speed_in_still_water_l229_229170


namespace vector_on_line_l229_229468

noncomputable def k_value (a b : Vector ℝ 3) (m : ℝ) : ℝ :=
  if h : m = 5 / 7 then
    (5 / 7 : ℝ)
  else
    0 -- This branch will never be taken because we will assume m = 5 / 7 as a hypothesis.


theorem vector_on_line (a b : Vector ℝ 3) (m k : ℝ) (h : m = 5 / 7) :
  k = k_value a b m :=
by
  sorry

end vector_on_line_l229_229468


namespace find_number_l229_229584

theorem find_number (x : ℕ) (h : x * 48 = 173 * 240) : x = 865 :=
sorry

end find_number_l229_229584


namespace approx_cube_of_331_l229_229504

noncomputable def cube (x : ℝ) : ℝ := x * x * x

theorem approx_cube_of_331 : 
  ∃ ε > 0, abs (cube 0.331 - 0.037) < ε :=
by
  sorry

end approx_cube_of_331_l229_229504


namespace salt_percentage_in_first_solution_l229_229463

variable (S : ℚ)
variable (H : 0 ≤ S ∧ S ≤ 100)  -- percentage constraints

theorem salt_percentage_in_first_solution (h : 0.75 * S / 100 + 7 = 16) : S = 12 :=
by { sorry }

end salt_percentage_in_first_solution_l229_229463


namespace product_of_primes_l229_229425

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l229_229425


namespace inequality_proof_l229_229634

theorem inequality_proof (a b c : ℝ) 
    (ha : a > 1) (hb : b > 1) (hc : c > 1) :
    (a^2 / (b - 1)) + (b^2 / (c - 1)) + (c^2 / (a - 1)) ≥ 12 :=
by {
    sorry
}

end inequality_proof_l229_229634


namespace area_PQR_l229_229801

-- Define the point P
def P : ℝ × ℝ := (1, 6)

-- Define the functions for lines passing through P with slopes 1 and 3
def line1 (x : ℝ) : ℝ := x + 5
def line2 (x : ℝ) : ℝ := 3 * x + 3

-- Define the x-intercepts of the lines
def Q : ℝ × ℝ := (-5, 0)
def R : ℝ × ℝ := (-1, 0)

-- Calculate the distance QR
def distance_QR : ℝ := abs (-1 - (-5))

-- Calculate the height from P to the x-axis
def height_P : ℝ := 6

-- State and prove the area of the triangle PQR
theorem area_PQR : 1 / 2 * distance_QR * height_P = 12 := by
  sorry -- The actual proof would be provided here

end area_PQR_l229_229801


namespace grape_juice_problem_l229_229707

noncomputable def grape_juice_amount (initial_mixture_volume : ℕ) (initial_concentration : ℝ) (final_concentration : ℝ) : ℝ :=
  let initial_grape_juice := initial_mixture_volume * initial_concentration
  let total_volume := initial_mixture_volume + final_concentration * (final_concentration - initial_grape_juice) / (1 - final_concentration) -- Total volume after adding x gallons
  let added_grape_juice := total_volume - initial_mixture_volume -- x gallons added
  added_grape_juice

theorem grape_juice_problem :
  grape_juice_amount 40 0.20 0.36 = 10 := 
by
  sorry

end grape_juice_problem_l229_229707


namespace product_of_averages_is_125000_l229_229695

-- Define the problem from step a
def sum_1_to_99 : ℕ := (99 * (1 + 99)) / 2
def average_of_group (x : ℕ) : Prop := 3 * 33 * x = sum_1_to_99

-- Define the goal to prove
theorem product_of_averages_is_125000 (x : ℕ) (h : average_of_group x) : x^3 = 125000 :=
by
  sorry

end product_of_averages_is_125000_l229_229695


namespace p_more_than_q_l229_229483

def stamps (p q : ℕ) : Prop :=
  p / q = 7 / 4 ∧ (p - 8) / (q + 8) = 6 / 5

theorem p_more_than_q (p q : ℕ) (h : stamps p q) : p - 8 - (q + 8) = 8 :=
by {
  sorry
}

end p_more_than_q_l229_229483


namespace max_ab_value_l229_229891

theorem max_ab_value (a b : ℝ) (h : ∀ x : ℝ, x^2 - 2 * a * x - b^2 + 12 ≤ 0 → x = a) : ab = 6 := by
  sorry

end max_ab_value_l229_229891


namespace ratio_four_l229_229544

variable {x y : ℝ}

theorem ratio_four : y = 0.25 * x → x / y = 4 := by
  sorry

end ratio_four_l229_229544


namespace smallest_positive_integer_expr_2010m_44000n_l229_229219

theorem smallest_positive_integer_expr_2010m_44000n :
  ∃ (m n : ℤ), 10 = gcd 2010 44000 :=
by
  sorry

end smallest_positive_integer_expr_2010m_44000n_l229_229219


namespace larry_expression_correct_l229_229358

theorem larry_expression_correct (a b c d e : ℤ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : c = 2) (h₄ : d = 5) :
  (a - b + c - d + e = a - (b + (c - (d - e)))) → e = 3 :=
by
  sorry

end larry_expression_correct_l229_229358


namespace geometric_sequence_sum_9000_l229_229952

noncomputable def sum_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := 
  a * (1 - r^n) / (1 - r)

theorem geometric_sequence_sum_9000 (a r : ℝ) (h : r ≠ 1) 
  (h1 : sum_geometric_sequence a r 3000 = 1000)
  (h2 : sum_geometric_sequence a r 6000 = 1900) : 
  sum_geometric_sequence a r 9000 = 2710 :=
sorry

end geometric_sequence_sum_9000_l229_229952


namespace carnival_days_l229_229346

theorem carnival_days (d : ℕ) (h : 50 * d + 3 * (50 * d) - 30 * d - 75 = 895) : d = 5 :=
by
  sorry

end carnival_days_l229_229346


namespace fill_tank_with_leak_l229_229394

namespace TankFilling

-- Conditions
def pump_fill_rate (P : ℝ) : Prop := P = 1 / 4
def leak_drain_rate (L : ℝ) : Prop := L = 1 / 5
def net_fill_rate (P L R : ℝ) : Prop := P - L = R
def fill_time (R T : ℝ) : Prop := T = 1 / R

-- Statement
theorem fill_tank_with_leak (P L R T : ℝ) (hP : pump_fill_rate P) (hL : leak_drain_rate L) (hR : net_fill_rate P L R) (hT : fill_time R T) :
  T = 20 :=
  sorry

end TankFilling

end fill_tank_with_leak_l229_229394


namespace sequence_formula_l229_229166

noncomputable def seq (a : ℕ+ → ℚ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ+, (a n - 3) * a (n + 1) - a n + 4 = 0

theorem sequence_formula (a : ℕ+ → ℚ) (h : seq a) :
  ∀ n : ℕ+, a n = (2 * n - 1) / n :=
by
  sorry

end sequence_formula_l229_229166


namespace normal_distribution_test_l229_229866

noncomputable def normal_distribution_at_least_90 : Prop :=
  let μ := 78
  let σ := 4
  -- Given reference data
  let p_within_3_sigma := 0.9974
  -- Calculate P(X >= 90)
  let p_at_least_90 := (1 - p_within_3_sigma) / 2
  -- The expected answer 0.13% ⇒ 0.0013
  p_at_least_90 = 0.0013

theorem normal_distribution_test :
  normal_distribution_at_least_90 :=
by
  sorry

end normal_distribution_test_l229_229866


namespace ratio_closest_to_one_l229_229336

-- Define the entrance fee for adults and children.
def adult_fee : ℕ := 20
def child_fee : ℕ := 15

-- Define the total collected amount.
def total_collected : ℕ := 2400

-- Define the number of adults and children.
variables (a c : ℕ)

-- The main theorem to prove:
theorem ratio_closest_to_one 
  (h1 : a > 0) -- at least one adult
  (h2 : c > 0) -- at least one child
  (h3 : adult_fee * a + child_fee * c = total_collected) : 
  a / (c : ℚ) = 69 / 68 := 
sorry

end ratio_closest_to_one_l229_229336


namespace bike_ride_distance_l229_229556

theorem bike_ride_distance (D : ℝ) (h : D / 10 = D / 15 + 0.5) : D = 15 :=
  sorry

end bike_ride_distance_l229_229556


namespace ellipse_area_l229_229638

def ellipse_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x + 9 * y^2 - 36 * y + 36 = 0

theorem ellipse_area :
  (∀ x y : ℝ, ellipse_equation x y → true) →
  (π * 1 * (4/3) = 4 * π / 3) :=
by
  intro h
  norm_num
  sorry

end ellipse_area_l229_229638


namespace unique_solution_ffx_eq_27_l229_229440

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 27

-- Prove that there is exactly one solution for f(f(x)) = 27 in the domain -3 ≤ x ≤ 5
theorem unique_solution_ffx_eq_27 :
  (∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 27) :=
by
  sorry

end unique_solution_ffx_eq_27_l229_229440


namespace front_view_length_l229_229364

-- Define the conditions of the problem
variables (d_body : ℝ) (d_side : ℝ) (d_top : ℝ)
variables (d_front : ℝ)

-- The given conditions
def conditions :=
  d_body = 5 * Real.sqrt 2 ∧
  d_side = 5 ∧
  d_top = Real.sqrt 34

-- The theorem to be proved
theorem front_view_length : 
  conditions d_body d_side d_top →
  d_front = Real.sqrt 41 :=
sorry

end front_view_length_l229_229364


namespace factorization_correct_l229_229765

theorem factorization_correct (C D : ℤ) (h : 15 = C * D ∧ 48 = 8 * 6 ∧ -56 = -8 * D - 6 * C):
  C * D + C = 18 :=
  sorry

end factorization_correct_l229_229765


namespace intersection_of_A_and_B_l229_229480

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | 1 < x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = {x | 1 < x ∧ x < 2} := by
  sorry

end intersection_of_A_and_B_l229_229480


namespace complex_purely_imaginary_a_eq_3_l229_229514

theorem complex_purely_imaginary_a_eq_3 (a : ℝ) :
  (∀ (a : ℝ), (a^2 - 2*a - 3) + (a + 1)*I = 0 + (a + 1)*I → a = 3) :=
by
  sorry

end complex_purely_imaginary_a_eq_3_l229_229514


namespace Z_evaluation_l229_229825

def Z (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem Z_evaluation : Z 5 3 = 19 := by
  sorry

end Z_evaluation_l229_229825


namespace find_a_l229_229471

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 :=
sorry

end find_a_l229_229471


namespace bob_bakes_pie_in_6_minutes_l229_229226

theorem bob_bakes_pie_in_6_minutes (x : ℕ) (h_alice : 60 / 5 = 12)
  (h_condition : 12 - 2 = 60 / x) : x = 6 :=
sorry

end bob_bakes_pie_in_6_minutes_l229_229226


namespace ratio_area_rect_sq_l229_229131

/-- 
  Given:
  1. The longer side of rectangle R is 1.2 times the length of a side of square S.
  2. The shorter side of rectangle R is 0.85 times the length of a side of square S.
  Prove that the ratio of the area of rectangle R to the area of square S is 51/50.
-/
theorem ratio_area_rect_sq (s : ℝ) 
  (h1 : ∃ r1, r1 = 1.2 * s) 
  (h2 : ∃ r2, r2 = 0.85 * s) : 
  (1.2 * s * 0.85 * s) / (s * s) = 51 / 50 := 
by
  sorry

end ratio_area_rect_sq_l229_229131


namespace slope_of_line_l229_229981

theorem slope_of_line (x y : ℝ) : 
  3 * y + 9 = -6 * x - 15 → 
  ∃ m b, y = m * x + b ∧ m = -2 := 
by {
  sorry
}

end slope_of_line_l229_229981


namespace find_C_and_D_l229_229142

theorem find_C_and_D (C D : ℚ) (h1 : 5 * C + 3 * D - 4 = 47) (h2 : C = D + 2) : 
  C = 57 / 8 ∧ D = 41 / 8 :=
by 
  sorry

end find_C_and_D_l229_229142


namespace solution_mn_l229_229441

theorem solution_mn (m n : ℤ) (h1 : |m| = 4) (h2 : |n| = 5) (h3 : n < 0) : m + n = -1 ∨ m + n = -9 := 
by
  sorry

end solution_mn_l229_229441


namespace emily_sixth_quiz_score_l229_229225

theorem emily_sixth_quiz_score (a1 a2 a3 a4 a5 : ℕ) (target_mean : ℕ) (sixth_score : ℕ) :
  a1 = 94 ∧ a2 = 97 ∧ a3 = 88 ∧ a4 = 90 ∧ a5 = 102 ∧ target_mean = 95 →
  sixth_score = (target_mean * 6 - (a1 + a2 + a3 + a4 + a5)) →
  sixth_score = 99 :=
by
  sorry

end emily_sixth_quiz_score_l229_229225


namespace solve_for_b_l229_229212

theorem solve_for_b (b : ℚ) (h : b + b / 4 - 1 = 3 / 2) : b = 2 :=
sorry

end solve_for_b_l229_229212


namespace solve_inequality_l229_229393

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l229_229393


namespace gcd_654327_543216_is_1_l229_229848

-- Define the gcd function and relevant numbers
def gcd_problem : Prop :=
  gcd 654327 543216 = 1

-- The statement of the theorem, with a placeholder for the proof
theorem gcd_654327_543216_is_1 : gcd_problem :=
by {
  -- actual proof will go here
  sorry
}

end gcd_654327_543216_is_1_l229_229848


namespace simplify_expr_1_simplify_expr_2_l229_229902

-- The first problem
theorem simplify_expr_1 (a : ℝ) : 2 * a^2 - 3 * a - 5 * a^2 + 6 * a = -3 * a^2 + 3 * a := 
by
  sorry

-- The second problem
theorem simplify_expr_2 (a : ℝ) : 2 * (a - 1) - (2 * a - 3) + 3 = 4 :=
by
  sorry

end simplify_expr_1_simplify_expr_2_l229_229902


namespace tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l229_229379

variable (α : ℝ)
variable (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1 / 4)

theorem tan_alpha_eq_neg2 : Real.tan α = -2 :=
  sorry

theorem sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2 :
  (Real.sin (2 * α) + 1) / (1 + Real.sin (2 * α) + Real.cos (2 * α)) = -1 / 2 :=
  sorry

end tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l229_229379


namespace no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l229_229264

theorem no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014 :
  ¬ ∃ (m n : ℕ), m^2 = n^2 + 2014 :=
by 
  sorry

end no_nat_m_n_eq_m_squared_eq_n_squared_plus_2014_l229_229264


namespace songs_per_album_l229_229843

theorem songs_per_album (C P : ℕ) (h1 : 4 * C + 5 * P = 72) (h2 : C = P) : C = 8 :=
by
  sorry

end songs_per_album_l229_229843


namespace journey_time_proof_l229_229664

noncomputable def journey_time_on_wednesday (d s x : ℝ) : ℝ :=
  d / s

theorem journey_time_proof (d s x : ℝ) (usual_speed_nonzero : s ≠ 0) :
  (journey_time_on_wednesday d s x) = 11 * x :=
by
  have thursday_speed : ℝ := 1.1 * s
  have thursday_time : ℝ := d / thursday_speed
  have time_diff : ℝ := (d / s) - thursday_time
  have reduced_time_eq_x : time_diff = x := by sorry
  have journey_time_eq : (d / s) = 11 * x := by sorry
  exact journey_time_eq

end journey_time_proof_l229_229664


namespace measure_of_angle_A_l229_229926

-- Define the conditions as assumptions
variable (B : Real) (angle1 angle2 A : Real)
-- Angle B is 120 degrees
axiom h1 : B = 120
-- One of the angles formed by the dividing line is 50 degrees
axiom h2 : angle1 = 50
-- Angles formed sum up to 180 degrees as they are supplementary
axiom h3 : angle2 = 180 - angle1
-- Vertical angles are equal
axiom h4 : A = angle2

theorem measure_of_angle_A (B angle1 angle2 A : Real) 
    (h1 : B = 120) (h2 : angle1 = 50) (h3 : angle2 = 180 - angle1) (h4 : A = angle2) : A = 130 := 
by
    sorry

end measure_of_angle_A_l229_229926


namespace geom_sequence_sum_l229_229254

theorem geom_sequence_sum (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, a n > 0)
  (h_geom : ∀ m n p q : ℕ, m + n = p + q → a m * a n = a p * a q)
  (h_eq : a 4 * a 6 + 2 * a 5 * a 7 + a 6 * a 8 = 36) :
  a 5 + a 7 = 6 :=
sorry

end geom_sequence_sum_l229_229254


namespace candy_last_days_l229_229822

variable (candy_from_neighbors candy_from_sister candy_per_day : ℕ)

theorem candy_last_days
  (h_candy_from_neighbors : candy_from_neighbors = 66)
  (h_candy_from_sister : candy_from_sister = 15)
  (h_candy_per_day : candy_per_day = 9) :
  let total_candy := candy_from_neighbors + candy_from_sister  
  (total_candy / candy_per_day) = 9 := by
  sorry

end candy_last_days_l229_229822


namespace state_B_more_candidates_l229_229685

theorem state_B_more_candidates (appeared : ℕ) (selected_A_pct selected_B_pct : ℝ)
  (h1 : appeared = 8000)
  (h2 : selected_A_pct = 0.06)
  (h3 : selected_B_pct = 0.07) :
  (selected_B_pct * appeared - selected_A_pct * appeared = 80) :=
by
  sorry

end state_B_more_candidates_l229_229685


namespace set_of_x_satisfying_2f_less_than_x_plus_1_l229_229739

theorem set_of_x_satisfying_2f_less_than_x_plus_1 (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : ∀ x : ℝ, deriv f x > 1 / 2) :
  { x : ℝ | 2 * f x < x + 1 } = { x : ℝ | x < 1 } :=
by
  sorry

end set_of_x_satisfying_2f_less_than_x_plus_1_l229_229739


namespace grains_on_11th_more_than_1_to_9_l229_229311

theorem grains_on_11th_more_than_1_to_9 : 
  let grains_on_square (k : ℕ) := 3 ^ k
  let sum_first_n_squares (n : ℕ) := (3 * (3 ^ n - 1) / (3 - 1))
  grains_on_square 11 - sum_first_n_squares 9 = 147624 :=
by
  sorry

end grains_on_11th_more_than_1_to_9_l229_229311


namespace selected_numbers_in_range_l229_229671

noncomputable def systematic_sampling (n_students selected_students interval_num start_num n : ℕ) : ℕ :=
  start_num + interval_num * (n - 1)

theorem selected_numbers_in_range (x : ℕ) :
  (500 = 500) ∧ (50 = 50) ∧ (10 = 500 / 50) ∧ (6 ∈ {y : ℕ | 1 ≤ y ∧ y ≤ 10}) ∧ (125 ≤ x ∧ x ≤ 140) → 
  (x = systematic_sampling 500 50 10 6 13 ∨ x = systematic_sampling 500 50 10 6 14) :=
by
  sorry

end selected_numbers_in_range_l229_229671


namespace y_in_terms_of_x_l229_229112

theorem y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 4) : y = 4 - 3 * x := 
by
  sorry

end y_in_terms_of_x_l229_229112


namespace largest_3_digit_sum_l229_229199

-- Defining the condition that ensures X, Y, Z are different digits ranging from 0 to 9
def valid_digits (X Y Z : ℕ) : Prop :=
  X ≠ Y ∧ Y ≠ Z ∧ X ≠ Z ∧ X < 10 ∧ Y < 10 ∧ Z < 10

-- Problem statement: Proving the largest possible 3-digit sum is 994
theorem largest_3_digit_sum : ∃ (X Y Z : ℕ), valid_digits X Y Z ∧ 111 * X + 11 * Y + Z = 994 :=
by
  sorry

end largest_3_digit_sum_l229_229199


namespace crows_and_trees_l229_229725

theorem crows_and_trees : ∃ (x y : ℕ), 3 * y + 5 = x ∧ 5 * (y - 1) = x ∧ x = 20 ∧ y = 5 :=
by
  sorry

end crows_and_trees_l229_229725


namespace marbles_steve_now_l229_229964
-- Import necessary libraries

-- Define the initial conditions as given in a)
def initial_conditions (sam steve sally : ℕ) := sam = 2 * steve ∧ sally = sam - 5 ∧ sam - 6 = 8

-- Define the proof problem statement
theorem marbles_steve_now (sam steve sally : ℕ) (h : initial_conditions sam steve sally) : steve + 3 = 10 :=
sorry

end marbles_steve_now_l229_229964


namespace probability_no_intersecting_chords_l229_229209

open Nat

def double_factorial (n : Nat) : Nat :=
  if n = 0 ∨ n = 1 then 1 else n * double_factorial (n - 2)

def catalan_number (n : Nat) : Nat :=
  (factorial (2 * n)) / (factorial n * factorial (n + 1))

theorem probability_no_intersecting_chords (n : Nat) (h : n > 0) :
  (catalan_number n) / (double_factorial (2 * n - 1)) = 2^n / (factorial (n + 1)) :=
by
  sorry

end probability_no_intersecting_chords_l229_229209


namespace log_one_plus_two_x_lt_two_x_l229_229294
open Real

theorem log_one_plus_two_x_lt_two_x {x : ℝ} (hx : x > 0) : log (1 + 2 * x) < 2 * x :=
sorry

end log_one_plus_two_x_lt_two_x_l229_229294


namespace probability_three_draws_one_white_l229_229899

def num_white_balls : ℕ := 4
def num_black_balls : ℕ := 3
def total_balls : ℕ := num_white_balls + num_black_balls

def probability_one_white_three_draws : ℚ := 
  (num_white_balls / total_balls) * 
  ((num_black_balls - 1) / (total_balls - 1)) * 
  ((num_black_balls - 2) / (total_balls - 2)) * 3

theorem probability_three_draws_one_white :
  probability_one_white_three_draws = 12 / 35 := by sorry

end probability_three_draws_one_white_l229_229899


namespace oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l229_229896

-- Definitions for oil consumption per person
def oilConsumptionWest : ℝ := 55.084
def oilConsumptionNonWest : ℝ := 214.59
def oilConsumptionRussia : ℝ := 1038.33

-- Lean statements
theorem oilProductionPerPerson_west : oilConsumptionWest = 55.084 := by
  sorry

theorem oilProductionPerPerson_nonwest : oilConsumptionNonWest = 214.59 := by
  sorry

theorem oilProductionPerPerson_russia : oilConsumptionRussia = 1038.33 := by
  sorry

end oilProductionPerPerson_west_oilProductionPerPerson_nonwest_oilProductionPerPerson_russia_l229_229896


namespace train_speed_problem_l229_229272

open Real

/-- Given specific conditions about the speeds and lengths of trains, prove the speed of the third train is 99 kmph. -/
theorem train_speed_problem
  (man_train_speed_kmph : ℝ)
  (man_train_speed : ℝ)
  (goods_train_length : ℝ)
  (goods_train_time : ℝ)
  (third_train_length : ℝ)
  (third_train_time : ℝ) :
  man_train_speed_kmph = 45 →
  man_train_speed = 45 * 1000 / 3600 →
  goods_train_length = 340 →
  goods_train_time = 8 →
  third_train_length = 480 →
  third_train_time = 12 →
  (third_train_length / third_train_time - man_train_speed) * 3600 / 1000 = 99 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end train_speed_problem_l229_229272


namespace find_p_a_l229_229847

variables (p : ℕ → ℝ) (a b : ℕ)

-- Given conditions
axiom p_b : p b = 0.5
axiom p_b_given_a : p b / p a = 0.2 
axiom p_a_inter_b : p a * p b = 0.36

-- Problem statement
theorem find_p_a : p a = 1.8 :=
by
  sorry

end find_p_a_l229_229847


namespace combined_weight_of_boxes_l229_229513

-- Defining the weights of each box as constants
def weight1 : ℝ := 2.5
def weight2 : ℝ := 11.3
def weight3 : ℝ := 5.75
def weight4 : ℝ := 7.2
def weight5 : ℝ := 3.25

-- The main theorem statement
theorem combined_weight_of_boxes : weight1 + weight2 + weight3 + weight4 + weight5 = 30 := by
  sorry

end combined_weight_of_boxes_l229_229513


namespace fence_poles_count_l229_229750

def length_path : ℕ := 900
def length_bridge : ℕ := 42
def distance_between_poles : ℕ := 6

theorem fence_poles_count :
  2 * (length_path - length_bridge) / distance_between_poles = 286 :=
by
  sorry

end fence_poles_count_l229_229750


namespace find_xy_l229_229233

theorem find_xy (x y : ℝ) (π_ne_zero : Real.pi ≠ 0) (h1 : 4 * (x + 2) = 6 * x) (h2 : 6 * x = 2 * Real.pi * y) : x = 4 ∧ y = 12 / Real.pi :=
by
  sorry

end find_xy_l229_229233


namespace percent_women_non_union_employees_is_65_l229_229761

-- Definitions based on the conditions
variables {E : ℝ} -- Denoting the total number of employees as a real number

def percent_men (E : ℝ) : ℝ := 0.56 * E
def percent_union_employees (E : ℝ) : ℝ := 0.60 * E
def percent_non_union_employees (E : ℝ) : ℝ := 0.40 * E
def percent_women_non_union (percent_non_union_employees : ℝ) : ℝ := 0.65 * percent_non_union_employees

-- Theorem statement
theorem percent_women_non_union_employees_is_65 :
  percent_women_non_union (percent_non_union_employees E) / (percent_non_union_employees E) = 0.65 :=
by
  sorry

end percent_women_non_union_employees_is_65_l229_229761


namespace sam_initial_balloons_l229_229139

theorem sam_initial_balloons:
  ∀ (S : ℝ), (S - 5.0 + 7.0 = 8) → S = 6.0 :=
by
  intro S h
  sorry

end sam_initial_balloons_l229_229139


namespace required_run_rate_equivalence_l229_229321

-- Define the conditions
def run_rate_first_10_overs : ℝ := 3.5
def overs_first_phase : ℝ := 10
def total_target_runs : ℝ := 350
def remaining_overs : ℝ := 35
def total_overs : ℝ := 45

-- Define the already scored runs
def runs_scored_first_10_overs : ℝ := run_rate_first_10_overs * overs_first_phase

-- Define the required runs for the remaining overs
def runs_needed : ℝ := total_target_runs - runs_scored_first_10_overs

-- Theorem stating the required run rate in the remaining 35 overs
theorem required_run_rate_equivalence :
  runs_needed / remaining_overs = 9 :=
by
  sorry

end required_run_rate_equivalence_l229_229321


namespace order_large_pizzas_sufficient_l229_229003

def pizza_satisfaction (gluten_free_slices_per_large : ℕ) (medium_slices : ℕ) (small_slices : ℕ) 
                       (gluten_free_needed : ℕ) (dairy_free_needed : ℕ) :=
  let slices_gluten_free := small_slices
  let slices_dairy_free := 2 * medium_slices
  (slices_gluten_free < gluten_free_needed) → 
  let additional_slices_gluten_free := gluten_free_needed - slices_gluten_free
  let large_pizzas_gluten_free := (additional_slices_gluten_free + gluten_free_slices_per_large - 1) / gluten_free_slices_per_large
  large_pizzas_gluten_free = 1

theorem order_large_pizzas_sufficient :
  pizza_satisfaction 14 10 8 15 15 :=
by
  unfold pizza_satisfaction
  sorry

end order_large_pizzas_sufficient_l229_229003


namespace base_conversion_least_sum_l229_229400

theorem base_conversion_least_sum :
  ∃ (c d : ℕ), (5 * c + 8 = 8 * d + 5) ∧ c > 0 ∧ d > 0 ∧ (c + d = 15) := by
sorry

end base_conversion_least_sum_l229_229400


namespace rectangle_breadth_l229_229661

theorem rectangle_breadth (sq_area : ℝ) (rect_area : ℝ) (radius_rect_relation : ℝ → ℝ) 
  (rect_length_relation : ℝ → ℝ) (breadth_correct: ℝ) : 
  (sq_area = 3600) →
  (rect_area = 240) →
  (forall r, radius_rect_relation r = r) →
  (forall r, rect_length_relation r = (2/5) * r) →
  breadth_correct = 10 :=
by
  intros h_sq_area h_rect_area h_radius_rect h_rect_length
  sorry

end rectangle_breadth_l229_229661


namespace sum_of_ages_l229_229117

theorem sum_of_ages (A B C : ℕ)
  (h1 : A = C + 8)
  (h2 : A + 10 = 3 * (C - 6))
  (h3 : B = 2 * C) :
  A + B + C = 80 := 
by 
  sorry

end sum_of_ages_l229_229117


namespace compare_negative_positive_l229_229659

theorem compare_negative_positive : -897 < 0.01 := sorry

end compare_negative_positive_l229_229659


namespace sam_walked_distance_l229_229642

theorem sam_walked_distance
  (distance_apart : ℝ) (fred_speed : ℝ) (sam_speed : ℝ) (t : ℝ)
  (H1 : distance_apart = 35) (H2 : fred_speed = 2) (H3 : sam_speed = 5)
  (H4 : 2 * t + 5 * t = distance_apart) :
  5 * t = 25 :=
by
  -- Lean proof goes here
  sorry

end sam_walked_distance_l229_229642


namespace problem1_problem2_l229_229392

open Real

noncomputable def alpha (hα : 0 < α ∧ α < π / 3) :=
  α

noncomputable def vec_a (hα : 0 < α ∧ α < π / 3) :=
  (sqrt 6 * sin (alpha hα), sqrt 2)

noncomputable def vec_b (hα : 0 < α ∧ α < π / 3) :=
  (1, cos (alpha hα) - sqrt 6 / 2)

theorem problem1 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  tan (alpha hα + π / 6) = sqrt 15 / 5 :=
sorry

theorem problem2 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  cos (2 * alpha hα + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
sorry

end problem1_problem2_l229_229392


namespace trigonometric_identity_proof_l229_229991

theorem trigonometric_identity_proof 
  (α β γ : ℝ) (a b c : ℝ)
  (h1 : 0 < α ∧ α < π)
  (h2 : 0 < β ∧ β < π)
  (h3 : 0 < γ ∧ γ < π)
  (hc : 0 < c)
  (hb : b = (c * (Real.cos α + Real.cos β * Real.cos γ)) / (Real.sin γ)^2)
  (ha : a = (c * (Real.cos β + Real.cos α * Real.cos γ)) / (Real.sin γ)^2) :
  1 - (Real.cos α)^2 - (Real.cos β)^2 - (Real.cos γ)^2 - 2 * Real.cos α * Real.cos β * Real.cos γ = 0 :=
by
  sorry

end trigonometric_identity_proof_l229_229991


namespace problem_I_problem_II_l229_229466

def intervalA := { x : ℝ | -2 < x ∧ x < 5 }
def intervalB (m : ℝ) := { x : ℝ | m < x ∧ x < m + 3 }

theorem problem_I (m : ℝ) :
  (intervalB m ⊆ intervalA) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by sorry

theorem problem_II (m : ℝ) :
  (intervalA ∩ intervalB m ≠ ∅) ↔ (-5 < m ∧ m < 2) :=
by sorry

end problem_I_problem_II_l229_229466


namespace ratio_of_distances_l229_229758

-- Definitions based on conditions in a)
variables (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y) (h_nonneg_w : 0 ≤ w)
variables (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0) (h_eq_times : y / w = x / w + (x + y) / (9 * w))

-- The proof statement
theorem ratio_of_distances (x y w : ℝ) (h_nonneg_x : 0 ≤ x) (h_nonneg_y : 0 ≤ y)
  (h_nonneg_w : 0 ≤ w) (h_w_ne_zero : w ≠ 0) (h_y_ne_zero : y ≠ 0)
  (h_eq_times : y / w = x / w + (x + y) / (9 * w)) :
  x / y = 4 / 5 :=
sorry

end ratio_of_distances_l229_229758


namespace infinite_series_sum_l229_229222

theorem infinite_series_sum :
  (∑' n : Nat, (4 * n + 1) / ((4 * n - 1)^2 * (4 * n + 3)^2)) = 1 / 72 :=
by
  sorry

end infinite_series_sum_l229_229222


namespace determine_pairs_l229_229576

theorem determine_pairs (p : ℕ) (hp: Nat.Prime p) :
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ p^x - y^3 = 1 ∧ ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2)) := 
sorry

end determine_pairs_l229_229576


namespace dimes_paid_l229_229692

theorem dimes_paid (cost_in_dollars : ℕ) (dollars_to_dimes : ℕ) (h₁ : cost_in_dollars = 5) (h₂ : dollars_to_dimes = 10) :
  cost_in_dollars * dollars_to_dimes = 50 :=
by
  sorry

end dimes_paid_l229_229692


namespace edward_remaining_money_l229_229213

def initial_amount : ℕ := 19
def spent_amount : ℕ := 13
def remaining_amount : ℕ := initial_amount - spent_amount

theorem edward_remaining_money : remaining_amount = 6 := by
  sorry

end edward_remaining_money_l229_229213


namespace find_painted_stencils_l229_229531

variable (hourly_wage racquet_wage grommet_wage stencil_wage total_earnings hours_worked racquets_restrung grommets_changed : ℕ)
variable (painted_stencils: ℕ)

axiom condition_hourly_wage : hourly_wage = 9
axiom condition_racquet_wage : racquet_wage = 15
axiom condition_grommet_wage : grommet_wage = 10
axiom condition_stencil_wage : stencil_wage = 1
axiom condition_total_earnings : total_earnings = 202
axiom condition_hours_worked : hours_worked = 8
axiom condition_racquets_restrung : racquets_restrung = 7
axiom condition_grommets_changed : grommets_changed = 2

theorem find_painted_stencils :
  painted_stencils = 5 :=
by
  -- Given:
  -- hourly_wage = 9
  -- racquet_wage = 15
  -- grommet_wage = 10
  -- stencil_wage = 1
  -- total_earnings = 202
  -- hours_worked = 8
  -- racquets_restrung = 7
  -- grommets_changed = 2

  -- We need to prove:
  -- painted_stencils = 5
  
  sorry

end find_painted_stencils_l229_229531


namespace pirate_coins_total_l229_229527

def total_coins (y : ℕ) := 6 * y

theorem pirate_coins_total : 
  (∃ y : ℕ, y ≠ 0 ∧ y * (y + 1) / 2 = 5 * y) →
  total_coins 9 = 54 :=
by
  sorry

end pirate_coins_total_l229_229527


namespace investment_growth_theorem_l229_229771

variable (x : ℝ)

-- Defining the initial and final investments
def initial_investment : ℝ := 800
def final_investment : ℝ := 960

-- Defining the growth equation
def growth_equation (x : ℝ) : Prop := initial_investment * (1 + x) ^ 2 = final_investment

-- The theorem statement that needs to be proven
theorem investment_growth_theorem : growth_equation x := sorry

end investment_growth_theorem_l229_229771


namespace ellipse_equation_line_equation_l229_229925
-- Import the necessary libraries

-- Problem (I): The equation of the ellipse
theorem ellipse_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (hA : (1 : ℝ) / a^2 + (9 / 4 : ℝ) / b^2 = 1)
  (h_ecc : b^2 = (3 / 4 : ℝ) * a^2) : 
  (a^2 = 4 ∧ b^2 = 3) :=
by
  sorry

-- Problem (II): The equation of the line
theorem line_equation (k : ℝ) (h_area : (12 * Real.sqrt (2 : ℝ)) / 7 = 12 * abs k / (4 * k^2 + 3)) : 
  k = 1 ∨ k = -1 :=
by
  sorry

end ellipse_equation_line_equation_l229_229925


namespace remainder_when_divided_by_17_l229_229316

theorem remainder_when_divided_by_17 (N : ℤ) (k : ℤ) 
  (h : N = 221 * k + 43) : N % 17 = 9 := 
by
  sorry

end remainder_when_divided_by_17_l229_229316


namespace man_l229_229554

-- Given conditions
def V_m := 15 - 3.2
def V_c := 3.2
def man's_speed_with_current : Real := 15

-- Required to prove
def man's_speed_against_current := V_m - V_c

theorem man's_speed_against_current_is_correct : man's_speed_against_current = 8.6 := by
  sorry

end man_l229_229554


namespace commission_percentage_proof_l229_229915

-- Let's define the problem conditions in Lean

-- Condition 1: Commission on first Rs. 10,000
def commission_first_10000 (sales : ℕ) : ℕ :=
  if sales ≤ 10000 then
    5 * sales / 100
  else
    500

-- Condition 2: Amount remitted to company after commission
def amount_remitted (total_sales : ℕ) (commission : ℕ) : ℕ :=
  total_sales - commission

-- Condition 3: Function to calculate commission on exceeding amount
def commission_exceeding (sales : ℕ) (x : ℕ) : ℕ :=
  x * sales / 100

-- The main hypothesis as per the given problem
def correct_commission_percentage (total_sales : ℕ) (remitted : ℕ) (x : ℕ) :=
  commission_first_10000 10000 + commission_exceeding (total_sales - 10000) x
  = total_sales - remitted

-- Problem statement to prove the percentage of commission on exceeding Rs. 10,000 is 4%
theorem commission_percentage_proof : correct_commission_percentage 32500 31100 4 := 
  by sorry

end commission_percentage_proof_l229_229915


namespace hyperbola_ellipse_b_value_l229_229578

theorem hyperbola_ellipse_b_value (a c b : ℝ) (h1 : c = 5 * a / 4) (h2 : c^2 - a^2 = (9 * a^2) / 16) (h3 : 4 * (b^2 - 4) = 16 * b^2 / 25) :
  b = 6 / 5 ∨ b = 10 / 3 :=
by
  sorry

end hyperbola_ellipse_b_value_l229_229578


namespace apple_picking_ratio_l229_229819

theorem apple_picking_ratio (a b c : ℕ) 
  (h1 : a = 66) 
  (h2 : b = 2 * 66) 
  (h3 : a + b + c = 220) :
  c = 22 → a = 66 → c / a = 1 / 3 := by
    intros
    sorry

end apple_picking_ratio_l229_229819


namespace decreases_as_x_increases_graph_passes_through_origin_l229_229386

-- Proof Problem 1: Show that y decreases as x increases if and only if k > 2
theorem decreases_as_x_increases (k : ℝ) : (∀ x1 x2 : ℝ, (x1 < x2) → ((2 - k) * x1 - k^2 + 4) > ((2 - k) * x2 - k^2 + 4)) ↔ (k > 2) := 
  sorry

-- Proof Problem 2: Show that the graph passes through the origin if and only if k = -2
theorem graph_passes_through_origin (k : ℝ) : ((2 - k) * 0 - k^2 + 4 = 0) ↔ (k = -2) :=
  sorry

end decreases_as_x_increases_graph_passes_through_origin_l229_229386


namespace melissa_bonus_points_l229_229724

/-- Given that Melissa scored 109 points per game and a total of 15089 points in 79 games,
    prove that she got 82 bonus points per game. -/
theorem melissa_bonus_points (points_per_game : ℕ) (total_points : ℕ) (num_games : ℕ)
  (H1 : points_per_game = 109)
  (H2 : total_points = 15089)
  (H3 : num_games = 79) : 
  (total_points - points_per_game * num_games) / num_games = 82 := by
  sorry

end melissa_bonus_points_l229_229724


namespace scientific_notation_correct_l229_229084

def num : ℝ := 1040000000

theorem scientific_notation_correct : num = 1.04 * 10^9 := by
  sorry

end scientific_notation_correct_l229_229084


namespace initial_water_percentage_l229_229355

theorem initial_water_percentage (W : ℕ) (V1 V2 V3 W3 : ℕ) (h1 : V1 = 10) (h2 : V2 = 15) (h3 : V3 = V1 + V2) (h4 : V3 = 25) (h5 : W3 = 2) (h6 : (W * V1) / 100 = (W3 * V3) / 100) : W = 5 :=
by
  sorry

end initial_water_percentage_l229_229355


namespace gcd_multiple_less_than_120_l229_229256

theorem gcd_multiple_less_than_120 (n : ℕ) (h1 : n < 120) (h2 : n % 10 = 0) (h3 : n % 15 = 0) : n ≤ 90 :=
by {
  sorry
}

end gcd_multiple_less_than_120_l229_229256


namespace brown_gumdrops_count_l229_229445

def gumdrops_conditions (total : ℕ) (blue : ℕ) (brown : ℕ) (red : ℕ) (yellow : ℕ) (green : ℕ) : Prop :=
  total = blue + brown + red + yellow + green ∧
  blue = total * 25 / 100 ∧
  brown = total * 25 / 100 ∧
  red = total * 20 / 100 ∧
  yellow = total * 15 / 100 ∧
  green = 40 ∧
  green = total * 15 / 100

theorem brown_gumdrops_count: ∃ total blue brown red yellow green new_brown,
  gumdrops_conditions total blue brown red yellow green →
  new_brown = brown + blue / 3 →
  new_brown = 89 :=
by
  sorry

end brown_gumdrops_count_l229_229445


namespace downstream_rate_l229_229135

/--  
A man's rowing conditions and rates:
- The man's upstream rate is U = 12 kmph.
- The man's rate in still water is S = 7 kmph.
- We need to prove that the man's downstream rate D is 14 kmph.
-/
theorem downstream_rate (U S D : ℝ) (hU : U = 12) (hS : S = 7) : D = 14 :=
by
  -- Proof to be filled here
  sorry

end downstream_rate_l229_229135


namespace lights_on_fourth_tier_l229_229807

def number_lights_topmost_tier (total_lights : ℕ) : ℕ :=
  total_lights / 127

def number_lights_tier (tier : ℕ) (lights_topmost : ℕ) : ℕ :=
  2^(tier - 1) * lights_topmost

theorem lights_on_fourth_tier (total_lights : ℕ) (H : total_lights = 381) : number_lights_tier 4 (number_lights_topmost_tier total_lights) = 24 :=
by
  rw [H]
  sorry

end lights_on_fourth_tier_l229_229807


namespace consecutive_coeff_sum_l229_229171

theorem consecutive_coeff_sum (P : Polynomial ℕ) (hdeg : P.degree = 699)
  (hP : P.eval 1 ≤ 2022) :
  ∃ k : ℕ, k < 700 ∧ (P.coeff (k + 1) + P.coeff k) = 22 ∨
                    (P.coeff (k + 1) + P.coeff k) = 55 ∨
                    (P.coeff (k + 1) + P.coeff k) = 77 :=
by
  sorry

end consecutive_coeff_sum_l229_229171


namespace minimum_number_of_tiles_l229_229375

def tile_width_in_inches : ℕ := 6
def tile_height_in_inches : ℕ := 4
def region_width_in_feet : ℕ := 3
def region_height_in_feet : ℕ := 8

def inches_to_feet (i : ℕ) : ℚ :=
  i / 12

def tile_width_in_feet : ℚ :=
  inches_to_feet tile_width_in_inches

def tile_height_in_feet : ℚ :=
  inches_to_feet tile_height_in_inches

def tile_area_in_square_feet : ℚ :=
  tile_width_in_feet * tile_height_in_feet

def region_area_in_square_feet : ℚ :=
  region_width_in_feet * region_height_in_feet

def number_of_tiles : ℚ :=
  region_area_in_square_feet / tile_area_in_square_feet

theorem minimum_number_of_tiles :
  number_of_tiles = 144 := by
    sorry

end minimum_number_of_tiles_l229_229375


namespace remainder_six_pow_4032_mod_13_l229_229026

theorem remainder_six_pow_4032_mod_13 : (6 ^ 4032) % 13 = 1 := 
by
  sorry

end remainder_six_pow_4032_mod_13_l229_229026


namespace inequality_solution_set_l229_229486

theorem inequality_solution_set (x : ℝ) : (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 :=
by sorry

end inequality_solution_set_l229_229486


namespace factor_fraction_eq_l229_229448

theorem factor_fraction_eq (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) 
  / ((a + b)^3 + (b + c)^3 + (c + a)^3) = 
  ((a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2)) 
  / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end factor_fraction_eq_l229_229448


namespace find_n_l229_229455

def seq (a : ℕ → ℚ) : Prop :=
  a 1 = 3 ∧ ∀ n : ℕ, a (n + 1) = -1 / (a n + 1)

theorem find_n (a : ℕ → ℚ) (h : seq a) : ∃ n : ℕ, a n = 3 ∧ n = 16 :=
by
  sorry

end find_n_l229_229455


namespace negation_example_l229_229183

theorem negation_example :
  ¬ (∀ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ ∃ x₀ : ℝ, x₀^3 - x₀^2 + 1 > 0 :=
by
  sorry

end negation_example_l229_229183


namespace find_function_solution_l229_229414

def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (x y : ℝ), f (f (x * y)) = |x| * f y + 3 * f (x * y)

theorem find_function_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 4 * |x|) ∨ (∀ x : ℝ, f x = -4 * |x|) :=
by
  sorry

end find_function_solution_l229_229414


namespace sasha_skated_distance_l229_229812

theorem sasha_skated_distance (d total_distance v : ℝ)
  (h1 : total_distance = 3300)
  (h2 : v > 0)
  (h3 : d = 3 * v * (total_distance / (3 * v + 2 * v))) :
  d = 1100 :=
by
  sorry

end sasha_skated_distance_l229_229812


namespace no_integer_solution_for_Q_square_l229_229489

def Q (x : ℤ) : ℤ := x^4 + 5*x^3 + 10*x^2 + 5*x + 56

theorem no_integer_solution_for_Q_square :
  ∀ x : ℤ, ∃ k : ℤ, Q x = k^2 → false :=
by
  sorry

end no_integer_solution_for_Q_square_l229_229489


namespace burger_cost_l229_229061

theorem burger_cost {B : ℝ} (sandwich_cost : ℝ) (smoothies_cost : ℝ) (total_cost : ℝ)
  (H1 : sandwich_cost = 4)
  (H2 : smoothies_cost = 8)
  (H3 : total_cost = 17)
  (H4 : B + sandwich_cost + smoothies_cost = total_cost) :
  B = 5 :=
sorry

end burger_cost_l229_229061


namespace valentines_distribution_l229_229423

theorem valentines_distribution (valentines_initial : ℝ) (valentines_needed : ℝ) (students : ℕ) 
  (h_initial : valentines_initial = 58.0) (h_needed : valentines_needed = 16.0) (h_students : students = 74) : 
  (valentines_initial + valentines_needed) / students = 1 :=
by
  sorry

end valentines_distribution_l229_229423


namespace polar_to_cartesian_circle_l229_229037

theorem polar_to_cartesian_circle :
  ∀ (r : ℝ) (x y : ℝ), r = 3 → r = Real.sqrt (x^2 + y^2) → x^2 + y^2 = 9 :=
by
  intros r x y hr h
  sorry

end polar_to_cartesian_circle_l229_229037


namespace max_children_arrangement_l229_229050

theorem max_children_arrangement (n : ℕ) (h1 : n = 49) 
  (h2 : ∀ i j, i ≠ j → 1 ≤ i ∧ i ≤ 49 → 1 ≤ j ∧ j ≤ 49 → (i * j < 100)) : 
  ∃ k, k = 18 :=
by
  sorry

end max_children_arrangement_l229_229050


namespace problem_expression_value_l229_229853

theorem problem_expression_value :
  (1 * 2 * 3 * 4 * 5 * 6 * 7 * 8 * 9 : ℤ) / (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 : ℤ) = 6608 :=
by
  sorry

end problem_expression_value_l229_229853


namespace find_three_digit_number_l229_229626

theorem find_three_digit_number (A B C D : ℕ) 
  (h1 : A + C = 5) 
  (h2 : B = 3)
  (h3 : A * 100 + B * 10 + C + 124 = D * 111) 
  (h4 : A ≠ B ∧ A ≠ C ∧ B ≠ C) : 
  A * 100 + B * 10 + C = 431 := 
by 
  sorry

end find_three_digit_number_l229_229626


namespace profit_diff_is_560_l229_229600

-- Define the initial conditions
def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1400

-- Define the ratio parts
def ratio_A : ℕ := 4
def ratio_B : ℕ := 5
def ratio_C : ℕ := 6

-- Define the value of one part based on B's profit share and ratio part
def value_per_part : ℕ := profit_share_B / ratio_B

-- Define the profit shares of A and C
def profit_share_A : ℕ := ratio_A * value_per_part
def profit_share_C : ℕ := ratio_C * value_per_part

-- Define the difference between the profit shares of A and C
def profit_difference : ℕ := profit_share_C - profit_share_A

-- The theorem to prove
theorem profit_diff_is_560 : profit_difference = 560 := 
by sorry

end profit_diff_is_560_l229_229600


namespace max_value_problem1_l229_229087

theorem max_value_problem1 (x : ℝ) (h1 : 0 < x) (h2 : x < 1/2) : 
  ∃ t, t = (1 / 2) * x * (1 - 2 * x) ∧ t ≤ 1 / 16 := sorry

end max_value_problem1_l229_229087


namespace division_addition_problem_l229_229792

theorem division_addition_problem :
  3752 / (39 * 2) + 5030 / (39 * 10) = 61 := by
  sorry

end division_addition_problem_l229_229792


namespace probability_of_selection_of_Ram_l229_229239

noncomputable def P_Ravi : ℚ := 1 / 5
noncomputable def P_Ram_and_Ravi : ℚ := 57 / 1000  -- This is the exact form of 0.05714285714285714

axiom independent_selection : ∀ (P_Ram P_Ravi : ℚ), P_Ram_and_Ravi = P_Ram * P_Ravi

theorem probability_of_selection_of_Ram (P_Ram : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ram = 2 / 7 := by
  intro h
  have h1 : P_Ram = P_Ram_and_Ravi / P_Ravi := sorry
  rw [h1, P_Ram_and_Ravi, P_Ravi]
  norm_num
  exact sorry

end probability_of_selection_of_Ram_l229_229239


namespace directrix_of_parabola_l229_229571

theorem directrix_of_parabola (a : ℝ) (h : a = -4) : ∃ k : ℝ, k = 1/16 ∧ ∀ x : ℕ, y = ax ^ 2 → y = k := 
by 
  sorry

end directrix_of_parabola_l229_229571


namespace belle_stickers_l229_229948

theorem belle_stickers (c_stickers : ℕ) (diff : ℕ) (b_stickers : ℕ) (h1 : c_stickers = 79) (h2 : diff = 18) (h3 : c_stickers = b_stickers - diff) : b_stickers = 97 := 
by
  sorry

end belle_stickers_l229_229948


namespace volume_of_tetrahedron_l229_229144

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l229_229144


namespace sequence_nth_term_l229_229501

/-- The nth term of the sequence {a_n} defined by a_1 = 1 and
    the recurrence relation a_{n+1} = 2a_n + 2 for all n ∈ ℕ*,
    is given by the formula a_n = 3 * 2 ^ (n - 1) - 2. -/
theorem sequence_nth_term (n : ℕ) (h : n > 0) : 
  ∃ (a : ℕ → ℤ), a 1 = 1 ∧ (∀ n > 0, a (n + 1) = 2 * a n + 2) ∧ a n = 3 * 2 ^ (n - 1) - 2 :=
  sorry

end sequence_nth_term_l229_229501


namespace part1_solution_part2_no_solution_l229_229234

theorem part1_solution (x y : ℚ) :
  x + y = 5 ∧ 3 * x + 10 * y = 30 ↔ x = 20 / 7 ∧ y = 15 / 7 :=
by
  sorry

theorem part2_no_solution (x : ℚ) :
  (x + 7) / 2 < 4 ∧ (3 * x - 1) / 2 ≤ 2 * x - 3 ↔ False :=
by
  sorry

end part1_solution_part2_no_solution_l229_229234


namespace solution_set_empty_iff_a_in_range_l229_229795

theorem solution_set_empty_iff_a_in_range (a : ℝ) :
  (∀ x : ℝ, ¬ (2 * x^2 + a * x + 2 < 0)) ↔ (-4 ≤ a ∧ a ≤ 4) :=
by
  sorry

end solution_set_empty_iff_a_in_range_l229_229795


namespace find_larger_number_l229_229120

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1365)
  (h2 : L = 8 * S + 15) : L = 1557 := 
sorry

end find_larger_number_l229_229120


namespace smallest_perfect_square_divisible_by_5_and_7_l229_229229

theorem smallest_perfect_square_divisible_by_5_and_7 
  (n : ℕ) 
  (h1 : n > 0) 
  (h2 : ∃ k : ℕ, n = k^2)
  (h3 : 5 ∣ n)
  (h4 : 7 ∣ n) : 
  n = 1225 :=
sorry

end smallest_perfect_square_divisible_by_5_and_7_l229_229229


namespace farmer_initial_tomatoes_l229_229200

theorem farmer_initial_tomatoes 
  (T : ℕ) -- The initial number of tomatoes
  (picked : ℕ)   -- The number of tomatoes picked
  (diff : ℕ) -- The difference between initial number of tomatoes and picked
  (h1 : picked = 9) -- The farmer picked 9 tomatoes
  (h2 : diff = 8) -- The difference is 8
  (h3 : T - picked = diff) -- T - 9 = 8
  :
  T = 17 := sorry

end farmer_initial_tomatoes_l229_229200


namespace chess_tournament_no_804_games_l229_229000

/-- Statement of the problem: 
    Under the given conditions, prove that it is impossible for exactly 804 games to have been played in the chess tournament.
--/
theorem chess_tournament_no_804_games :
  ¬ ∃ n : ℕ, n * (n - 4) = 1608 :=
by
  sorry

end chess_tournament_no_804_games_l229_229000


namespace no_burial_needed_for_survivors_l229_229914

def isSurvivor (p : Person) : Bool := sorry
def isBuried (p : Person) : Bool := sorry
variable (p : Person) (accident : Bool)

theorem no_burial_needed_for_survivors (h : accident = true) (hsurvive : isSurvivor p = true) : isBuried p = false :=
sorry

end no_burial_needed_for_survivors_l229_229914


namespace fraction_reach_impossible_l229_229640

theorem fraction_reach_impossible :
  ¬ ∃ (a b : ℕ), (2 + 2013 * a) / (3 + 2014 * b) = 3 / 5 := by
  sorry

end fraction_reach_impossible_l229_229640


namespace two_absent_one_present_probability_l229_229162

-- Define the probabilities
def probability_absent_normal : ℚ := 1 / 15

-- Given that the absence rate on Monday increases by 10%
def monday_increase_factor : ℚ := 1.1

-- Calculate the probability of being absent on Monday
def probability_absent_monday : ℚ := probability_absent_normal * monday_increase_factor

-- Calculate the probability of being present on Monday
def probability_present_monday : ℚ := 1 - probability_absent_monday

-- Define the probability that exactly two students are absent and one present
def probability_two_absent_one_present : ℚ :=
  3 * (probability_absent_monday ^ 2) * probability_present_monday

-- Convert the probability to a percentage and round to the nearest tenth
def probability_as_percent : ℚ := round (probability_two_absent_one_present * 100 * 10) / 10

theorem two_absent_one_present_probability : probability_as_percent = 1.5 := by sorry

end two_absent_one_present_probability_l229_229162


namespace ellipse_hyperbola_foci_l229_229339

theorem ellipse_hyperbola_foci (a b : ℝ) 
    (h1 : b^2 - a^2 = 25) 
    (h2 : a^2 + b^2 = 49) : 
    |a * b| = 2 * Real.sqrt 111 := 
by 
  -- proof omitted 
  sorry

end ellipse_hyperbola_foci_l229_229339


namespace functions_equal_l229_229232

noncomputable def f (x : ℝ) : ℝ := x^0
noncomputable def g (x : ℝ) : ℝ := x / x

theorem functions_equal (x : ℝ) (hx : x ≠ 0) : f x = g x :=
by
  unfold f g
  sorry

end functions_equal_l229_229232


namespace surface_area_of_glued_cubes_l229_229520

noncomputable def calculate_surface_area (large_cube_edge_length : ℕ) : ℕ :=
sorry

theorem surface_area_of_glued_cubes :
  calculate_surface_area 4 = 136 :=
sorry

end surface_area_of_glued_cubes_l229_229520


namespace monotonic_decreasing_interval_l229_229935

noncomputable def f (x : ℝ) := Real.log x + x^2 - 3 * x

theorem monotonic_decreasing_interval :
  (∃ I : Set ℝ, I = Set.Ioo (1 / 2 : ℝ) 1 ∧ ∀ x ∈ I, ∀ y ∈ I, x < y → f x ≥ f y) := 
by
  sorry

end monotonic_decreasing_interval_l229_229935


namespace total_cost_l229_229411

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l229_229411


namespace diophantine_infinite_solutions_l229_229595

theorem diophantine_infinite_solutions :
  ∃ (a b c x y : ℤ), (a + b + c = x + y) ∧ (a^3 + b^3 + c^3 = x^3 + y^3) ∧ 
  ∃ (d : ℤ), (a = b - d) ∧ (c = b + d) :=
sorry

end diophantine_infinite_solutions_l229_229595


namespace Rogers_age_more_than_twice_Jills_age_l229_229611

/--
Jill is 20 years old.
Finley is 40 years old.
Roger's age is more than twice Jill's age.
In 15 years, the age difference between Roger and Jill will be 30 years less than Finley's age.
Prove that Roger's age is 5 years more than twice Jill's age.
-/
theorem Rogers_age_more_than_twice_Jills_age 
  (J F : ℕ) (hJ : J = 20) (hF : F = 40) (R x : ℕ)
  (hR : R = 2 * J + x) 
  (age_diff_condition : (R + 15) - (J + 15) = (F + 15) - 30) :
  x = 5 := 
sorry

end Rogers_age_more_than_twice_Jills_age_l229_229611


namespace range_of_3x_minus_y_l229_229052

-- Defining the conditions in Lean
variable (x y : ℝ)

-- Condition 1: -1 ≤ x + y ≤ 1
def cond1 : Prop := -1 ≤ x + y ∧ x + y ≤ 1

-- Condition 2: 1 ≤ x - y ≤ 3
def cond2 : Prop := 1 ≤ x - y ∧ x - y ≤ 3

-- The theorem statement to prove that the range of 3x - y is [1, 7]
theorem range_of_3x_minus_y (h1 : cond1 x y) (h2 : cond2 x y) : 1 ≤ 3 * x - y ∧ 3 * x - y ≤ 7 := by
  sorry

end range_of_3x_minus_y_l229_229052


namespace vertex_of_parabola_l229_229297

theorem vertex_of_parabola (a b : ℝ) (roots_condition : ∀ x, -x^2 + a * x + b ≤ 0 ↔ (x ≤ -3 ∨ x ≥ 5)) :
  ∃ v : ℝ × ℝ, v = (1, 16) :=
by
  sorry

end vertex_of_parabola_l229_229297


namespace x_lt_2_necessary_not_sufficient_x_sq_lt_4_l229_229820

theorem x_lt_2_necessary_not_sufficient_x_sq_lt_4 (x : ℝ) :
  (x < 2) → (x^2 < 4) ∧ ¬((x^2 < 4) → (x < 2)) :=
by
  sorry

end x_lt_2_necessary_not_sufficient_x_sq_lt_4_l229_229820


namespace upstream_distance_l229_229727

theorem upstream_distance (v : ℝ) 
  (H1 : ∀ d : ℝ, (10 + v) * 2 = 28) 
  (H2 : (10 - v) * 2 = d) : d = 12 := by
  sorry

end upstream_distance_l229_229727


namespace exists_consecutive_numbers_divisible_by_3_5_7_l229_229545

theorem exists_consecutive_numbers_divisible_by_3_5_7 :
  ∃ (a : ℕ), 100 ≤ a ∧ a ≤ 200 ∧
    a % 3 = 0 ∧ (a + 1) % 5 = 0 ∧ (a + 2) % 7 = 0 :=
by
  sorry

end exists_consecutive_numbers_divisible_by_3_5_7_l229_229545


namespace problem_1_problem_2_l229_229973

-- First Proof Problem
theorem problem_1 (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = 2 * x^2 + 1) : 
  f x = 2 * x^2 - 4 * x + 3 :=
sorry

-- Second Proof Problem
theorem problem_2 {a b : ℝ} (f : ℝ → ℝ) (hf : ∀ x, f x = x / (a * x + b))
  (h1 : f 2 = 1) (h2 : ∃! x, f x = x) : 
  f x = 2 * x / (x + 2) :=
sorry

end problem_1_problem_2_l229_229973


namespace find_h_l229_229018

theorem find_h (h : ℝ) (r s : ℝ) (h_eq : ∀ x : ℝ, x^2 - 4 * h * x - 8 = 0)
  (sum_of_squares : r^2 + s^2 = 20) (roots_eq : x^2 - 4 * h * x - 8 = (x - r) * (x - s)) :
  h = 1 / 2 ∨ h = -1 / 2 := 
sorry

end find_h_l229_229018


namespace cory_can_eat_fruits_in_105_ways_l229_229536

-- Define the number of apples, oranges, and bananas Cory has
def apples := 4
def oranges := 1
def bananas := 2

-- Define the total number of fruits Cory has
def total_fruits := apples + oranges + bananas

-- Calculate the number of distinct orders in which Cory can eat the fruits
theorem cory_can_eat_fruits_in_105_ways :
  (Nat.factorial total_fruits) / (Nat.factorial apples * Nat.factorial oranges * Nat.factorial bananas) = 105 :=
by
  -- Provide a sorry to skip the proof
  sorry

end cory_can_eat_fruits_in_105_ways_l229_229536


namespace problem1_problem2_l229_229773

noncomputable def f (x : ℝ) : ℝ := 1 / (1 + x^2)

theorem problem1 :
  f 1 + f 2 + f 3 + f (1 / 2) + f (1 / 3) = 5 / 2 :=
by
  sorry

theorem problem2 : ∀ x : ℝ, 0 < f x ∧ f x ≤ 1 :=
by
  intro x
  sorry

end problem1_problem2_l229_229773


namespace pond_field_area_ratio_l229_229022

theorem pond_field_area_ratio
  (l : ℝ) (w : ℝ) (A_field : ℝ) (A_pond : ℝ)
  (h1 : l = 2 * w)
  (h2 : l = 16)
  (h3 : A_field = l * w)
  (h4 : A_pond = 8 * 8) :
  A_pond / A_field = 1 / 2 :=
by
  sorry

end pond_field_area_ratio_l229_229022


namespace max_colors_4x4_grid_l229_229794

def cell := (Fin 4) × (Fin 4)
def color := Fin 8

def valid_coloring (f : cell → color) : Prop :=
∀ c1 c2 : color, (c1 ≠ c2) →
(∃ i : Fin 4, ∃ j1 j2 : Fin 4, j1 ≠ j2 ∧ f (i, j1) = c1 ∧ f (i, j2) = c2) ∨ 
(∃ j : Fin 4, ∃ i1 i2 : Fin 4, i1 ≠ i2 ∧ f (i1, j) = c1 ∧ f (i2, j) = c2)

theorem max_colors_4x4_grid : ∃ (f : cell → color), valid_coloring f :=
sorry

end max_colors_4x4_grid_l229_229794


namespace set_of_possible_values_l229_229956

-- Define the variables and the conditions as a Lean definition
noncomputable def problem (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) : Set ℝ :=
  {x : ℝ | x = (1 / a + 1 / b + 1 / c)}

-- Define the theorem to state that the set of all possible values is [9, ∞)
theorem set_of_possible_values (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (sum_eq_one : a + b + c = 1) :
  problem a b c ha hb hc sum_eq_one = {x : ℝ | 9 ≤ x} :=
sorry

end set_of_possible_values_l229_229956


namespace r_plus_s_value_l229_229313

theorem r_plus_s_value :
  (∃ (r s : ℝ) (line_intercepts : ∀ x y, y = -1/2 * x + 8 ∧ ((x = 16 ∧ y = 0) ∨ (x = 0 ∧ y = 8))), 
    s = -1/2 * r + 8 ∧ (16 * 8 / 2) = 2 * (16 * s / 2) ∧ r + s = 12) :=
sorry

end r_plus_s_value_l229_229313


namespace reciprocal_of_neg_four_l229_229002

def is_reciprocal (x y : ℚ) : Prop := x * y = 1

theorem reciprocal_of_neg_four : is_reciprocal (-4) (-1/4) :=
by
  sorry

end reciprocal_of_neg_four_l229_229002


namespace quadratic_root_l229_229083

theorem quadratic_root (a : ℝ) : (∃ x : ℝ, x = 1 ∧ a * x^2 + x - 2 = 0) → a = 1 := by
  sorry

end quadratic_root_l229_229083


namespace inequality_solution_l229_229057

theorem inequality_solution (x y : ℝ) (h : 5 * x > -5 * y) : x + y > 0 :=
sorry

end inequality_solution_l229_229057


namespace inequality_proof_l229_229516

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b + b^2 / c + c^2 / a) + (a + b + c) ≥ (6 * (a^2 + b^2 + c^2) / (a + b + c)) :=
by
  sorry

end inequality_proof_l229_229516


namespace length_of_platform_l229_229108

theorem length_of_platform
  (length_train : ℝ)
  (speed_train_kmph : ℝ)
  (time_seconds : ℝ)
  (distance_covered : ℝ)
  (conversion_factor : ℝ) :
  length_train = 250 →
  speed_train_kmph = 90 →
  time_seconds = 20 →
  distance_covered = (speed_train_kmph * 1000 / 3600) * time_seconds →
  conversion_factor = 1000 / 3600 →
  ∃ P : ℝ, distance_covered = length_train + P ∧ P = 250 :=
by
  sorry

end length_of_platform_l229_229108


namespace booksJuly_l229_229533

-- Definitions of the conditions
def booksMay : ℕ := 2
def booksJune : ℕ := 6
def booksTotal : ℕ := 18

-- Theorem statement proving how many books Tom read in July
theorem booksJuly : (booksTotal - (booksMay + booksJune)) = 10 :=
by
  sorry

end booksJuly_l229_229533


namespace johns_payment_l229_229987

-- Define the value of the camera
def camera_value : ℕ := 5000

-- Define the rental fee rate per week as a percentage
def rental_fee_rate : ℝ := 0.1

-- Define the rental period in weeks
def rental_period : ℕ := 4

-- Define the friend's contribution rate as a percentage
def friend_contribution_rate : ℝ := 0.4

-- Theorem: Calculate how much John pays for the camera rental
theorem johns_payment :
  let weekly_rental_fee := camera_value * rental_fee_rate
  let total_rental_fee := weekly_rental_fee * rental_period
  let friends_contribution := total_rental_fee * friend_contribution_rate
  let johns_payment := total_rental_fee - friends_contribution
  johns_payment = 1200 :=
by
  sorry

end johns_payment_l229_229987


namespace probability_same_color_socks_l229_229261

-- Define the total number of socks and the groups
def total_socks : ℕ := 30
def blue_socks : ℕ := 16
def green_socks : ℕ := 10
def red_socks : ℕ := 4

-- Define combinatorial functions to calculate combinations
def comb (n m : ℕ) : ℕ := n.choose m

-- Calculate the number of favorable outcomes
def favorable_outcomes : ℕ :=
  comb blue_socks 2 +
  comb green_socks 2 +
  comb red_socks 2

-- Calculate the total number of possible outcomes
def total_outcomes : ℕ := comb total_socks 2

-- Calculate the probability as a ratio of favorable outcomes to total outcomes
def probability := favorable_outcomes / total_outcomes

-- Prove the probability is 19/45
theorem probability_same_color_socks : probability = 19 / 45 := by
  sorry

end probability_same_color_socks_l229_229261


namespace arithmetic_sequence_sum_n_squared_l229_229097

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

end arithmetic_sequence_sum_n_squared_l229_229097


namespace area_covered_three_layers_l229_229417

noncomputable def auditorium_width : ℕ := 10
noncomputable def auditorium_height : ℕ := 10

noncomputable def first_rug_width : ℕ := 6
noncomputable def first_rug_height : ℕ := 8
noncomputable def second_rug_width : ℕ := 6
noncomputable def second_rug_height : ℕ := 6
noncomputable def third_rug_width : ℕ := 5
noncomputable def third_rug_height : ℕ := 7

-- Prove that the area of part of the auditorium covered with rugs in three layers is 6 square meters.
theorem area_covered_three_layers : 
  let horizontal_overlap_second_third := 5
  let vertical_overlap_second_third := 3
  let area_overlap_second_third := horizontal_overlap_second_third * vertical_overlap_second_third
  let horizontal_overlap_all := 3
  let vertical_overlap_all := 2
  let area_overlap_all := horizontal_overlap_all * vertical_overlap_all
  area_overlap_all = 6 := 
by
  sorry

end area_covered_three_layers_l229_229417


namespace arithmetic_sequence_nth_term_639_l229_229195

theorem arithmetic_sequence_nth_term_639 :
  ∀ (x n : ℕ) (a₁ a₂ a₃ aₙ : ℤ),
  a₁ = 3 * x - 5 →
  a₂ = 7 * x - 17 →
  a₃ = 4 * x + 3 →
  aₙ = a₁ + (n - 1) * (a₂ - a₁) →
  aₙ = 4018 →
  n = 639 :=
by
  intros x n a₁ a₂ a₃ aₙ h₁ h₂ h₃ hₙ hₙ_eq
  sorry

end arithmetic_sequence_nth_term_639_l229_229195


namespace perpendicular_tangents_sum_x1_x2_gt_4_l229_229467

noncomputable def f (x : ℝ) : ℝ := (1 / 6) * x^3 - (1 / 2) * x^2 + (1 / 3)
noncomputable def g (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def F (x : ℝ) : ℝ := (1 / 2) * x^2 - x - 2 * Real.log x

theorem perpendicular_tangents (a : ℝ) (b : ℝ) (c : ℝ) (h₁ : a = 1) (h₂ : b = 1 / 3) (h₃ : c = 0) :
  let f' x := (1 / 2) * x^2 - x
  let g' x := 2 / x
  f' 1 * g' 1 = -1 :=
by sorry

theorem sum_x1_x2_gt_4 (x1 x2 : ℝ) (h₁ : 0 < x1 ∧ x1 < 4) (h₂ : 0 < x2 ∧ x2 < 4) (h₃ : x1 ≠ x2) (h₄ : F x1 = F x2) :
  x1 + x2 > 4 :=
by sorry

end perpendicular_tangents_sum_x1_x2_gt_4_l229_229467


namespace annual_fixed_costs_l229_229653

theorem annual_fixed_costs
  (profit : ℝ := 30500000)
  (selling_price : ℝ := 9035)
  (variable_cost : ℝ := 5000)
  (units_sold : ℕ := 20000) :
  ∃ (fixed_costs : ℝ), profit = (selling_price * units_sold) - (variable_cost * units_sold) - fixed_costs :=
sorry

end annual_fixed_costs_l229_229653


namespace trigonometric_identity_l229_229287

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  (Real.sin α - Real.cos α) / (5 * Real.sin α + 3 * Real.cos α) = 1 / 13 := 
by
-- The proof goes here
sorry

end trigonometric_identity_l229_229287


namespace problem_1_problem_2_l229_229269

-- Problem (1)
theorem problem_1 (a c : ℝ) (h1 : ∀ x, -1/3 < x ∧ x < 1/2 → ax^2 + 2*x + c > 0) :
  ∃ s, s = { x | -2 < x ∧ x < 3 } ∧ (∀ x, x ∈ s → cx^2 - 2*x + a < 0) := 
sorry

-- Problem (2)
theorem problem_2 (m : ℝ) (h : ∀ x : ℝ, x > 0 → x^2 - m*x + 4 > 0) :
  m < 4 := 
sorry

end problem_1_problem_2_l229_229269


namespace percentage_of_males_l229_229115

noncomputable def total_employees : ℝ := 1800
noncomputable def males_below_50_years_old : ℝ := 756
noncomputable def percentage_below_50 : ℝ := 0.70

theorem percentage_of_males : (males_below_50_years_old / percentage_below_50 / total_employees) * 100 = 60 :=
by
  sorry

end percentage_of_males_l229_229115


namespace marie_socks_problem_l229_229260

theorem marie_socks_problem (x y z : ℕ) : 
  x + y + z = 15 → 
  2 * x + 3 * y + 5 * z = 36 → 
  1 ≤ x → 
  1 ≤ y → 
  1 ≤ z → 
  x = 11 :=
by
  sorry

end marie_socks_problem_l229_229260


namespace JulioHasMoreSoda_l229_229231

-- Define the number of bottles each person has
def JulioOrangeBottles : ℕ := 4
def JulioGrapeBottles : ℕ := 7
def MateoOrangeBottles : ℕ := 1
def MateoGrapeBottles : ℕ := 3

-- Define the volume of each bottle in liters
def BottleVolume : ℕ := 2

-- Define the total liters of soda each person has
def JulioTotalLiters : ℕ := JulioOrangeBottles * BottleVolume + JulioGrapeBottles * BottleVolume
def MateoTotalLiters : ℕ := MateoOrangeBottles * BottleVolume + MateoGrapeBottles * BottleVolume

-- Prove the difference in total liters of soda between Julio and Mateo
theorem JulioHasMoreSoda : JulioTotalLiters - MateoTotalLiters = 14 := by
  sorry

end JulioHasMoreSoda_l229_229231


namespace probability_of_square_product_l229_229398

theorem probability_of_square_product :
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9 -- (1,1), (1,4), (2,2), (4,1), (3,3), (9,1), (4,4), (5,5), (6,6)
  favorable_outcomes / total_outcomes = 1 / 8 :=
by
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9
  have h1 : favorable_outcomes / total_outcomes = 1 / 8 := sorry
  exact h1

end probability_of_square_product_l229_229398


namespace cricket_initial_overs_l229_229248

theorem cricket_initial_overs
  (x : ℕ)
  (hx1 : ∃ x : ℕ, 0 ≤ x)
  (initial_run_rate : ℝ)
  (remaining_run_rate : ℝ)
  (remaining_overs : ℕ)
  (target_runs : ℕ)
  (H1 : initial_run_rate = 3.2)
  (H2 : remaining_run_rate = 6.25)
  (H3 : remaining_overs = 40)
  (H4 : target_runs = 282) :
  3.2 * (x : ℝ) + 6.25 * 40 = 282 → x = 10 := 
by 
  simp only [H1, H2, H3, H4]
  sorry

end cricket_initial_overs_l229_229248


namespace b_2016_eq_neg_4_l229_229410

def b : ℕ → ℤ
| 0     => 1
| 1     => 5
| (n+2) => b (n+1) - b n

theorem b_2016_eq_neg_4 : b 2015 = -4 :=
sorry

end b_2016_eq_neg_4_l229_229410


namespace tan_theta_neq_2sqrt2_l229_229809

theorem tan_theta_neq_2sqrt2 (θ : ℝ) (h₀ : 0 < θ ∧ θ < Real.pi) (h₁ : Real.sin θ + Real.cos θ = (2 * Real.sqrt 2 - 1) / 3) : Real.tan θ = -2 * Real.sqrt 2 := by
  sorry

end tan_theta_neq_2sqrt2_l229_229809


namespace sin_neg_60_eq_neg_sqrt_3_div_2_l229_229348

theorem sin_neg_60_eq_neg_sqrt_3_div_2 : 
  Real.sin (-π / 3) = - (Real.sqrt 3) / 2 := 
by
  sorry

end sin_neg_60_eq_neg_sqrt_3_div_2_l229_229348


namespace find_factor_l229_229840

theorem find_factor (x f : ℕ) (h1 : x = 15) (h2 : (2 * x + 5) * f = 105) : f = 3 :=
sorry

end find_factor_l229_229840


namespace min_value_of_a_k_l229_229349

-- Define the conditions for our proof in Lean

-- a_n is a positive arithmetic sequence
def is_positive_arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ d, ∀ m, a (m + 1) = a m + d

-- Given inequality condition for the sequence
def inequality_condition (a : ℕ → ℝ) (k : ℕ) : Prop :=
  k ≥ 2 ∧ (1 / a 1 + 4 / a (2 * k - 1) ≤ 1)

-- Prove the minimum value of a_k
theorem min_value_of_a_k (a : ℕ → ℝ) (k : ℕ) (h_arith : is_positive_arithmetic_seq a) (h_ineq : inequality_condition a k) :
  a k = 9 / 2 :=
sorry

end min_value_of_a_k_l229_229349


namespace gcd_m_n_l229_229484

def m : ℕ := 3333333
def n : ℕ := 66666666

theorem gcd_m_n : Nat.gcd m n = 3 := by
  sorry

end gcd_m_n_l229_229484


namespace smallest_z_value_l229_229421

theorem smallest_z_value :
  ∃ (w x y z : ℕ), w < x ∧ x < y ∧ y < z ∧
  w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
  w^3 + x^3 + y^3 = z^3 ∧ z = 6 := by
  sorry

end smallest_z_value_l229_229421


namespace both_not_divisible_by_7_l229_229728

theorem both_not_divisible_by_7 {a b : ℝ} (h : ¬ (∃ k : ℤ, ab = 7 * k)) : ¬ (∃ m : ℤ, a = 7 * m) ∧ ¬ (∃ n : ℤ, b = 7 * n) :=
sorry

end both_not_divisible_by_7_l229_229728


namespace largest_multiple_of_12_neg_gt_neg_150_l229_229805

theorem largest_multiple_of_12_neg_gt_neg_150 : ∃ m : ℤ, (m % 12 = 0) ∧ (-m > -150) ∧ ∀ n : ℤ, (n % 12 = 0) ∧ (-n > -150) → n ≤ m := sorry

end largest_multiple_of_12_neg_gt_neg_150_l229_229805


namespace eq_has_positive_integer_solution_l229_229422

theorem eq_has_positive_integer_solution (a : ℤ) :
  (∃ x : ℕ+, (x : ℤ) - 4 - 2 * (a * x - 1) = 2) → a = 0 :=
by
  sorry

end eq_has_positive_integer_solution_l229_229422


namespace find_x_l229_229962

-- Definitions corresponding to conditions a)
def rectangle (AB CD BC AD x : ℝ) := AB = 2 ∧ CD = 2 ∧ BC = 1 ∧ AD = 1 ∧ x = 0

-- Define the main statement to be proven
theorem find_x (AB CD BC AD x k m: ℝ) (h: rectangle AB CD BC AD x) : 
  x = (0 : ℝ) ∧ k = 0 ∧ m = 0 ∧ x = (Real.sqrt k - m) ∧ k + m = 0 :=
by
  cases h
  sorry

end find_x_l229_229962


namespace bob_25_cent_coins_l229_229464

theorem bob_25_cent_coins (a b c : ℕ)
    (h₁ : a + b + c = 15)
    (h₂ : 15 + 4 * c = 27) : c = 3 := by
  sorry

end bob_25_cent_coins_l229_229464


namespace mary_time_l229_229679

-- Define the main entities for the problem
variables (mary_days : ℕ) (rosy_days : ℕ)
variable (rosy_efficiency_factor : ℝ) -- Rosy's efficiency factor compared to Mary

-- Given conditions
def rosy_efficient := rosy_efficiency_factor = 1.4
def rosy_time := rosy_days = 20

-- Problem Statement
theorem mary_time (h1 : rosy_efficient rosy_efficiency_factor) (h2 : rosy_time rosy_days) : mary_days = 28 :=
by
  sorry

end mary_time_l229_229679


namespace kickball_students_l229_229509

theorem kickball_students (w t : ℕ) (hw : w = 37) (ht : t = w - 9) : w + t = 65 :=
by
  sorry

end kickball_students_l229_229509


namespace difference_between_median_and_mean_is_five_l229_229150

noncomputable def mean_score : ℝ :=
  0.20 * 60 + 0.20 * 75 + 0.40 * 85 + 0.20 * 95

noncomputable def median_score : ℝ := 85

theorem difference_between_median_and_mean_is_five :
  abs (median_score - mean_score) = 5 :=
by
  unfold mean_score median_score
  -- median_score - mean_score = 85 - 80
  -- thus the absolute value of the difference is 5
  sorry

end difference_between_median_and_mean_is_five_l229_229150


namespace angle_of_inclination_vert_line_l229_229829

theorem angle_of_inclination_vert_line (x : ℝ) (h : x = -1) : 
  ∃ θ : ℝ, θ = 90 := 
by
  sorry

end angle_of_inclination_vert_line_l229_229829


namespace senior_citizen_ticket_cost_l229_229599

theorem senior_citizen_ticket_cost 
  (total_tickets : ℕ)
  (regular_ticket_cost : ℕ)
  (total_sales : ℕ)
  (sold_regular_tickets : ℕ)
  (x : ℕ)
  (h1 : total_tickets = 65)
  (h2 : regular_ticket_cost = 15)
  (h3 : total_sales = 855)
  (h4 : sold_regular_tickets = 41)
  (h5 : total_sales = (sold_regular_tickets * regular_ticket_cost) + ((total_tickets - sold_regular_tickets) * x)) :
  x = 10 :=
by
  sorry

end senior_citizen_ticket_cost_l229_229599


namespace sum_of_natural_numbers_l229_229543

theorem sum_of_natural_numbers :
  36 + 17 + 32 + 54 + 28 + 3 = 170 :=
by
  sorry

end sum_of_natural_numbers_l229_229543


namespace robbers_divide_and_choose_l229_229335

/-- A model of dividing loot between two robbers who do not trust each other -/
def divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) : Prop :=
  ∀ (B : ℕ → ℕ), B (max P1 P2) ≥ B P1 ∧ B (max P1 P2) ≥ B P2

theorem robbers_divide_and_choose (P1 P2 : ℕ) (A : P1 = P2) :
  divide_and_choose P1 P2 A :=
sorry

end robbers_divide_and_choose_l229_229335


namespace fraction_students_walk_home_l229_229627

theorem fraction_students_walk_home :
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  walk_home = 41/120 :=
by 
  let bus := 1/3
  let auto := 1/5
  let bicycle := 1/8
  let total_students := 1
  let other_transportation := bus + auto + bicycle
  let walk_home := total_students - other_transportation
  have h_bus : bus = 40 / 120 := by sorry
  have h_auto : auto = 24 / 120 := by sorry
  have h_bicycle : bicycle = 15 / 120 := by sorry
  have h_total_transportation : other_transportation = 40 / 120 + 24 / 120 + 15 / 120 := by sorry
  have h_other_transportation_sum : other_transportation = 79 / 120 := by sorry
  have h_walk_home : walk_home = 1 - 79 / 120 := by sorry
  have h_walk_home_simplified : walk_home = 41 / 120 := by sorry
  exact h_walk_home_simplified

end fraction_students_walk_home_l229_229627


namespace total_flour_correct_l229_229790

-- Define the quantities specified in the conditions
def cups_of_flour_already_added : ℕ := 2
def cups_of_flour_to_add : ℕ := 7

-- Define the total cups of flour required by the recipe as a sum of the quantities
def cups_of_flour_required : ℕ := cups_of_flour_already_added + cups_of_flour_to_add

-- Prove that the total cups of flour required is 9
theorem total_flour_correct : cups_of_flour_required = 9 := by
  -- use auto proof placeholder
  rfl

end total_flour_correct_l229_229790


namespace probability_of_winning_noughts_l229_229594

def ticTacToeProbability : ℚ :=
  let totalWays := Nat.choose 9 3
  let winningWays := 8
  winningWays / totalWays

theorem probability_of_winning_noughts :
  ticTacToeProbability = 2 / 21 := by
  sorry

end probability_of_winning_noughts_l229_229594


namespace third_number_is_60_l229_229811

theorem third_number_is_60 (x : ℤ) :
  (20 + 40 + x) / 3 = (10 + 80 + 15) / 3 + 5 → x = 60 :=
by
  intro h
  sorry

end third_number_is_60_l229_229811


namespace ratio_of_money_earned_l229_229590

variable (L T J : ℕ) 

theorem ratio_of_money_earned 
  (total_earned : L + T + J = 60)
  (lisa_earning : L = 30)
  (lisa_tommy_diff : L = T + 15) : 
  T / L = 1 / 2 := 
by
  sorry

end ratio_of_money_earned_l229_229590


namespace Kuwabara_class_girls_percentage_l229_229062

variable (num_girls num_boys : ℕ)

def total_students (num_girls num_boys : ℕ) : ℕ :=
  num_girls + num_boys

def girls_percentage (num_girls num_boys : ℕ) : ℚ :=
  (num_girls : ℚ) / (total_students num_girls num_boys : ℚ) * 100

theorem Kuwabara_class_girls_percentage (num_girls num_boys : ℕ) (h1: num_girls = 10) (h2: num_boys = 15) :
  girls_percentage num_girls num_boys = 40 := 
by
  sorry

end Kuwabara_class_girls_percentage_l229_229062


namespace market_value_correct_l229_229391

noncomputable def face_value : ℝ := 100
noncomputable def dividend_per_share : ℝ := 0.14 * face_value
noncomputable def yield : ℝ := 0.08

theorem market_value_correct :
  (dividend_per_share / yield) * 100 = 175 := by
  sorry

end market_value_correct_l229_229391


namespace length_of_purple_part_l229_229185

theorem length_of_purple_part (p : ℕ) (black : ℕ) (blue : ℕ) (total : ℕ) 
  (h1 : black = 2) (h2 : blue = 1) (h3 : total = 6) (h4 : p + black + blue = total) : 
  p = 3 :=
by
  sorry

end length_of_purple_part_l229_229185


namespace find_blue_balls_l229_229797

/-- 
Given the conditions that a bag contains:
- 5 red balls
- B blue balls
- 2 green balls
And the probability of picking 2 red balls at random is 0.1282051282051282,
prove that the number of blue balls (B) is 6.
--/

theorem find_blue_balls (B : ℕ) (h : 0.1282051282051282 = (10 : ℚ) / (↑((7 + B) * (6 + B)) / 2)) : B = 6 := 
by sorry

end find_blue_balls_l229_229797


namespace bridget_apples_l229_229443

theorem bridget_apples :
  ∃ x : ℕ, (x - x / 3 - 4) = 6 :=
by
  sorry

end bridget_apples_l229_229443


namespace john_tips_problem_l229_229426

theorem john_tips_problem
  (A M : ℝ)
  (H1 : ∀ (A : ℝ), M * A = 0.5 * (6 * A + M * A)) :
  M = 6 := 
by
  sorry

end john_tips_problem_l229_229426


namespace cannot_be_value_of_A_plus_P_l229_229266

theorem cannot_be_value_of_A_plus_P (a b : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (a_neq_b: a ≠ b) :
  let A : ℕ := a * b
  let P : ℕ := 2 * a + 2 * b
  A + P ≠ 102 :=
by
  sorry

end cannot_be_value_of_A_plus_P_l229_229266


namespace remainder_division_l229_229017

theorem remainder_division (k : ℤ) (N : ℤ) (h : N = 133 * k + 16) : N % 50 = 49 := by
  sorry

end remainder_division_l229_229017


namespace find_five_digit_number_l229_229291

theorem find_five_digit_number (a b c d e : ℕ) 
  (h : [ (10 * a + a), (10 * a + b), (10 * a + b), (10 * a + b), (10 * a + c), 
         (10 * b + c), (10 * b + b), (10 * b + c), (10 * c + b), (10 * c + b)] = 
         [33, 37, 37, 37, 38, 73, 77, 78, 83, 87]) :
  10000 * a + 1000 * b + 100 * c + 10 * d + e = 37837 :=
sorry

end find_five_digit_number_l229_229291


namespace number_of_children_l229_229295

theorem number_of_children (n m : ℕ) (h1 : 11 * (m + 6) + n * m = n^2 + 3 * n - 2) : n = 9 :=
sorry

end number_of_children_l229_229295


namespace min_reciprocal_sum_l229_229216

theorem min_reciprocal_sum (m n a b : ℝ) (h1 : m = 5) (h2 : n = 5) 
  (h3 : m * a + n * b = 1) (h4 : 0 < a) (h5 : 0 < b) : 
  (1 / a + 1 / b) = 20 :=
by 
  sorry

end min_reciprocal_sum_l229_229216


namespace not_prime_3999991_l229_229156

   theorem not_prime_3999991 : ¬ Nat.Prime 3999991 :=
   by
     -- Provide the factorization proof
     sorry
   
end not_prime_3999991_l229_229156


namespace roots_expression_l229_229715

theorem roots_expression (p q : ℝ) (hpq : (∀ x, 3*x^2 + 9*x - 21 = 0 → x = p ∨ x = q)) 
  (sum_roots : p + q = -3) 
  (prod_roots : p * q = -7) : (3*p - 4) * (6*q - 8) = 122 :=
by
  sorry

end roots_expression_l229_229715


namespace sum_of_a6_and_a7_l229_229045

theorem sum_of_a6_and_a7 (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_sum : a 2 + a 3 + a 10 + a 11 = 48) : a 6 + a 7 = 24 :=
by
  sorry

end sum_of_a6_and_a7_l229_229045


namespace third_discount_is_five_percent_l229_229360

theorem third_discount_is_five_percent (P F : ℝ) (D : ℝ)
  (h1: P = 9356.725146198829)
  (h2: F = 6400)
  (h3: F = (1 - D / 100) * (0.9 * (0.8 * P))) : 
  D = 5 := by
  sorry

end third_discount_is_five_percent_l229_229360


namespace parabola_line_unique_eq_l229_229808

noncomputable def parabola_line_equation : Prop :=
  ∃ (A B : ℝ × ℝ),
    (A.2^2 = 4 * A.1) ∧ (B.2^2 = 4 * B.1) ∧
    ((A.1 + B.1) / 2 = 2) ∧ ((A.2 + B.2) / 2 = 2) ∧
    ∀ x y, (y - 2 = 1 * (x - 2)) → (x - y = 0)

theorem parabola_line_unique_eq : parabola_line_equation :=
  sorry

end parabola_line_unique_eq_l229_229808


namespace solve_for_exponent_l229_229967

theorem solve_for_exponent (K : ℕ) (h1 : 32 = 2 ^ 5) (h2 : 64 = 2 ^ 6) 
    (h3 : 32 ^ 5 * 64 ^ 2 = 2 ^ K) : K = 37 := 
by 
    sorry

end solve_for_exponent_l229_229967


namespace g_84_value_l229_229564

-- Define the function g with the given conditions
def g (x : ℝ) : ℝ := sorry

-- Conditions given in the problem
axiom g_property1 : ∀ x y : ℝ, g (x * y) = y * g x
axiom g_property2 : g 2 = 48

-- Statement to prove
theorem g_84_value : g 84 = 2016 :=
by
  sorry

end g_84_value_l229_229564


namespace product_of_random_numbers_greater_zero_l229_229435

noncomputable def random_product_positive_probability : ℝ := 
  let interval_length := 45
  let neg_interval_length := 30
  let pos_interval_length := 15
  let prob_neg := (neg_interval_length : ℝ) / interval_length
  let prob_pos := (pos_interval_length : ℝ) / interval_length
  prob_pos * prob_pos + prob_neg * prob_neg

-- Prove that the probability that the product of two randomly selected numbers
-- from the interval [-30, 15] is greater than zero is 5/9.
theorem product_of_random_numbers_greater_zero : 
  random_product_positive_probability = 5 / 9 := by
  sorry

end product_of_random_numbers_greater_zero_l229_229435


namespace cake_slices_l229_229782

open Nat

theorem cake_slices (S : ℕ) (h1 : 2 * S - 12 = 10) : S = 8 := by
  sorry

end cake_slices_l229_229782


namespace lesser_solution_of_quadratic_eq_l229_229563

theorem lesser_solution_of_quadratic_eq : ∃ x ∈ {x | x^2 + 10*x - 24 = 0}, x = -12 :=
by 
  sorry

end lesser_solution_of_quadratic_eq_l229_229563


namespace logarithmic_inequality_and_integral_l229_229582

theorem logarithmic_inequality_and_integral :
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  a > b ∧ b > c :=
by
  let a := Real.log 3 / Real.log 2
  let b := Real.log 2 / Real.log 3
  let c := 2 / Real.pi^2
  sorry

end logarithmic_inequality_and_integral_l229_229582


namespace number_of_rabbits_l229_229015

theorem number_of_rabbits
  (dogs : ℕ) (cats : ℕ) (total_animals : ℕ)
  (joins_each_cat : ℕ → ℕ)
  (hares_per_rabbit : ℕ)
  (h_dogs : dogs = 1)
  (h_cats : cats = 4)
  (h_total : total_animals = 37)
  (h_hares_per_rabbit : hares_per_rabbit = 3)
  (H : total_animals = dogs + cats + 4 * joins_each_cat cats + 3 * 4 * joins_each_cat cats) :
  joins_each_cat cats = 2 :=
by
  sorry

end number_of_rabbits_l229_229015


namespace combined_cost_price_l229_229608

theorem combined_cost_price :
  let stock1_price := 100
  let stock1_discount := 5 / 100
  let stock1_brokerage := 1.5 / 100
  let stock2_price := 200
  let stock2_discount := 7 / 100
  let stock2_brokerage := 0.75 / 100
  let stock3_price := 300
  let stock3_discount := 3 / 100
  let stock3_brokerage := 1 / 100

  -- Calculated values
  let stock1_discounted_price := stock1_price * (1 - stock1_discount)
  let stock1_total_price := stock1_discounted_price * (1 + stock1_brokerage)
  
  let stock2_discounted_price := stock2_price * (1 - stock2_discount)
  let stock2_total_price := stock2_discounted_price * (1 + stock2_brokerage)
  
  let stock3_discounted_price := stock3_price * (1 - stock3_discount)
  let stock3_total_price := stock3_discounted_price * (1 + stock3_brokerage)
  
  let combined_cost := stock1_total_price + stock2_total_price + stock3_total_price
  combined_cost = 577.73 := sorry

end combined_cost_price_l229_229608


namespace jina_mascots_l229_229781

variables (x y z x_new Total : ℕ)

def mascots_problem :=
  (y = 3 * x) ∧
  (x_new = x + 2 * y) ∧
  (z = 2 * y) ∧
  (Total = x_new + y + z) →
  Total = 16 * x

-- The statement only, no proof is required
theorem jina_mascots : mascots_problem x y z x_new Total := sorry

end jina_mascots_l229_229781


namespace jar_filling_fraction_l229_229798

theorem jar_filling_fraction (C1 C2 C3 W : ℝ)
  (h1 : W = (1/7) * C1)
  (h2 : W = (2/9) * C2)
  (h3 : W = (3/11) * C3)
  (h4 : C3 > C1 ∧ C3 > C2) :
  (3 * W) = (9 / 11) * C3 :=
by sorry

end jar_filling_fraction_l229_229798


namespace fraction_of_males_l229_229966

theorem fraction_of_males (M F : ℝ) 
  (h1 : M + F = 1)
  (h2 : (7 / 8) * M + (4 / 5) * F = 0.845) :
  M = 0.6 :=
by
  sorry

end fraction_of_males_l229_229966


namespace geometric_sequence_alpha_5_l229_229760

theorem geometric_sequence_alpha_5 (α : ℕ → ℝ) (h1 : α 4 * α 5 * α 6 = 27) (h2 : α 4 * α 6 = (α 5) ^ 2) : α 5 = 3 := 
sorry

end geometric_sequence_alpha_5_l229_229760


namespace remainder_when_divided_l229_229079
-- First, import the necessary library.

-- Define the problem conditions and the goal.
theorem remainder_when_divided (P Q Q' R R' S T D D' D'' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + D'' * R' + R')
  (h3 : S = D'' * T)
  (h4 : R' = S + T) :
  P % (D * D' * D'') = D * R' + R := by
  sorry

end remainder_when_divided_l229_229079


namespace find_angle_l229_229593

theorem find_angle (θ : ℝ) (h : 180 - θ = 3 * (90 - θ)) : θ = 45 :=
by
  sorry

end find_angle_l229_229593


namespace find_dividend_l229_229575

theorem find_dividend (divisor quotient remainder : ℕ) (h_divisor : divisor = 38) (h_quotient : quotient = 19) (h_remainder : remainder = 7) :
  divisor * quotient + remainder = 729 := by
  sorry

end find_dividend_l229_229575


namespace building_height_l229_229462

theorem building_height (h : ℕ) (flagpole_height : ℕ) (flagpole_shadow : ℕ) (building_shadow : ℕ) :
  flagpole_height = 18 ∧ flagpole_shadow = 45 ∧ building_shadow = 60 → h = 24 :=
by
  intros
  sorry

end building_height_l229_229462


namespace train_speed_l229_229697

theorem train_speed (train_length bridge_length : ℕ) (time : ℝ)
  (h_train_length : train_length = 110)
  (h_bridge_length : bridge_length = 290)
  (h_time : time = 23.998080153587715) :
  (train_length + bridge_length) / time * 3.6 = 60 := 
by
  rw [h_train_length, h_bridge_length, h_time]
  sorry

end train_speed_l229_229697


namespace print_pages_l229_229136

theorem print_pages (pages_per_cost : ℕ) (cost_cents : ℕ) (dollars : ℕ)
                    (h1 : pages_per_cost = 7) (h2 : cost_cents = 9) (h3 : dollars = 50) :
  (dollars * 100 * pages_per_cost) / cost_cents = 3888 :=
by
  sorry

end print_pages_l229_229136


namespace slope_angle_of_vertical_line_l229_229524

theorem slope_angle_of_vertical_line :
  ∀ {θ : ℝ}, (∀ x, (x = 3) → x = 3) → θ = 90 := by
  sorry

end slope_angle_of_vertical_line_l229_229524


namespace find_the_number_l229_229890

theorem find_the_number (x k : ℕ) (h1 : x / k = 4) (h2 : k = 8) : x = 32 := by
  sorry

end find_the_number_l229_229890


namespace Lowella_score_l229_229475

theorem Lowella_score
  (Mandy_score : ℕ)
  (Pamela_score : ℕ)
  (Lowella_score : ℕ)
  (h1 : Mandy_score = 84) 
  (h2 : Mandy_score = 2 * Pamela_score)
  (h3 : Pamela_score = Lowella_score + 20) :
  Lowella_score = 22 := by
  sorry

end Lowella_score_l229_229475


namespace min_questions_any_three_cards_min_questions_consecutive_three_cards_l229_229817

-- Definitions for numbers on cards and necessary questions
variables (n : ℕ) (h_n : n > 3)
  (cards : Fin n → ℤ)
  (h_cards_range : ∀ i, cards i = 1 ∨ cards i = -1)

-- Case (a): Product of any three cards
theorem min_questions_any_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (∃ (k : ℕ), n = 3 * k + 1 ∧ p = k + 1) ∨
  (∃ (k : ℕ), n = 3 * k + 2 ∧ p = k + 2) :=
sorry
  
-- Case (b): Product of any three consecutive cards
theorem min_questions_consecutive_three_cards :
  (∃ (k : ℕ), n = 3 * k ∧ p = k) ∨
  (¬(∃ (k : ℕ), n = 3 * k) ∧ p = n) :=
sorry

end min_questions_any_three_cards_min_questions_consecutive_three_cards_l229_229817


namespace max_value_expression_l229_229735

open Real

theorem max_value_expression (x : ℝ) : 
  ∃ (y : ℝ), y ≤ (x^6 / (x^10 + 3 * x^8 - 5 * x^6 + 10 * x^4 + 25)) ∧
  y = 1 / (5 + 2 * sqrt 30) :=
sorry

end max_value_expression_l229_229735


namespace circle_equation_through_intersections_l229_229748

theorem circle_equation_through_intersections 
  (h₁ : ∀ x y : ℝ, x^2 + y^2 + 6 * x - 4 = 0 ↔ x^2 + y^2 + 6 * y - 28 = 0)
  (h₂ : ∀ x y : ℝ, x - y - 4 = 0) : 
  ∃ x y : ℝ, (x - 1/2) ^ 2 + (y + 7 / 2) ^ 2 = 89 / 2 :=
by sorry

end circle_equation_through_intersections_l229_229748


namespace total_jellybeans_needed_l229_229698

def large_glass_jellybeans : ℕ := 50
def small_glass_jellybeans : ℕ := large_glass_jellybeans / 2
def num_large_glasses : ℕ := 5
def num_small_glasses : ℕ := 3

theorem total_jellybeans_needed : 
  (num_large_glasses * large_glass_jellybeans) + (num_small_glasses * small_glass_jellybeans) = 325 := 
by
  sorry

end total_jellybeans_needed_l229_229698


namespace quadratic_equation_m_value_l229_229021

-- Definition of the quadratic equation having exactly one solution with the given parameters
def quadratic_equation_has_one_solution (a b c : ℚ) : Prop :=
  b^2 - 4 * a * c = 0

-- Given constants in the problem
def a : ℚ := 3
def b : ℚ := -7

-- The value of m we aim to prove
def m_correct : ℚ := 49 / 12

-- The theorem stating the problem
theorem quadratic_equation_m_value (m : ℚ) (h : quadratic_equation_has_one_solution a b m) : m = m_correct :=
  sorry

end quadratic_equation_m_value_l229_229021


namespace simplify_expression_l229_229562

-- Define the initial expression
def expr (q : ℚ) := (5 * q^4 - 4 * q^3 + 7 * q - 8) + (3 - 5 * q^2 + q^3 - 2 * q)

-- Define the simplified expression
def simplified_expr (q : ℚ) := 5 * q^4 - 3 * q^3 - 5 * q^2 + 5 * q - 5

-- The theorem stating that the two expressions are equal
theorem simplify_expression (q : ℚ) : expr q = simplified_expr q :=
by
  sorry

end simplify_expression_l229_229562


namespace smallest_x_satisfies_abs_eq_l229_229515

theorem smallest_x_satisfies_abs_eq (x : ℝ) :
  (|2 * x + 5| = 21) → (x = -13) :=
sorry

end smallest_x_satisfies_abs_eq_l229_229515


namespace extra_interest_amount_l229_229810

def principal : ℝ := 15000
def rate1 : ℝ := 0.15
def rate2 : ℝ := 0.12
def time : ℕ := 2

theorem extra_interest_amount :
  principal * (rate1 - rate2) * time = 900 := by
  sorry

end extra_interest_amount_l229_229810


namespace hyperbola_eccentricity_l229_229279

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
variables (c e : ℝ)

-- Define the eccentricy condition for hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_eccentricity :
  -- Conditions regarding the hyperbola and the distances
  (∀ x y : ℝ, hyperbola a b x y → 
    (∃ x y : ℝ, y = (2 / 3) * c ∧ x = 2 * a + (2 / 3) * c ∧
    ((2 / 3) * c)^2 + (2 * a + (2 / 3) * c)^2 = 4 * c^2 ∧
    (7 * e^2 - 6 * e - 9 = 0))) →
  -- Proving that the eccentricity e is as given
  e = (3 + Real.sqrt 6) / 7 :=
sorry

end hyperbola_eccentricity_l229_229279


namespace first_discount_is_10_l229_229858

def list_price : ℝ := 70
def final_price : ℝ := 59.85
def second_discount : ℝ := 0.05

theorem first_discount_is_10 :
  ∃ (x : ℝ), list_price * (1 - x/100) * (1 - second_discount) = final_price ∧ x = 10 :=
by
  sorry

end first_discount_is_10_l229_229858


namespace area_of_ABC_l229_229734

noncomputable def area_of_triangle (AB AC angleB : ℝ) : ℝ :=
  0.5 * AB * AC * Real.sin angleB

theorem area_of_ABC :
  area_of_triangle 5 3 (120 * Real.pi / 180) = (15 * Real.sqrt 3) / 4 :=
by
  sorry

end area_of_ABC_l229_229734


namespace change_in_expression_l229_229429

theorem change_in_expression (x b : ℝ) (hb : 0 < b) : 
    (2 * (x + b) ^ 2 + 5 - (2 * x ^ 2 + 5) = 4 * x * b + 2 * b ^ 2) ∨ 
    (2 * (x - b) ^ 2 + 5 - (2 * x ^ 2 + 5) = -4 * x * b + 2 * b ^ 2) := 
by
    sorry

end change_in_expression_l229_229429


namespace arithmetic_progression_contains_sixth_power_l229_229753

theorem arithmetic_progression_contains_sixth_power
  (a h : ℕ) (a_pos : 0 < a) (h_pos : 0 < h)
  (sq : ∃ n : ℕ, a + n * h = k^2)
  (cube : ∃ m : ℕ, a + m * h = l^3) :
  ∃ p : ℕ, ∃ q : ℕ, a + q * h = p^6 := sorry

end arithmetic_progression_contains_sixth_power_l229_229753


namespace neither_drinkers_eq_nine_l229_229676

-- Define the number of businessmen at the conference
def total_businessmen : Nat := 30

-- Define the number of businessmen who drank coffee
def coffee_drinkers : Nat := 15

-- Define the number of businessmen who drank tea
def tea_drinkers : Nat := 13

-- Define the number of businessmen who drank both coffee and tea
def both_drinkers : Nat := 7

-- Prove the number of businessmen who drank neither coffee nor tea
theorem neither_drinkers_eq_nine : 
  total_businessmen - ((coffee_drinkers + tea_drinkers) - both_drinkers) = 9 := 
by
  sorry

end neither_drinkers_eq_nine_l229_229676


namespace kiddie_scoop_cost_is_three_l229_229970

-- Define the parameters for the costs of different scoops and total payment
variable (k : ℕ)  -- cost of kiddie scoop
def cost_regular : ℕ := 4
def cost_double : ℕ := 6
def total_payment : ℕ := 32

-- Conditions: Mr. and Mrs. Martin each get a regular scoop
def regular_cost : ℕ := 2 * cost_regular

-- Their three teenage children each get double scoops
def double_cost : ℕ := 3 * cost_double

-- Total cost of regular and double scoops
def combined_cost : ℕ := regular_cost + double_cost

-- Total payment includes two kiddie scoops
def kiddie_total_cost : ℕ := total_payment - combined_cost

-- The cost of one kiddie scoop
def kiddie_cost : ℕ := kiddie_total_cost / 2

theorem kiddie_scoop_cost_is_three : kiddie_cost = 3 := by
  sorry

end kiddie_scoop_cost_is_three_l229_229970


namespace pattern_continues_for_max_8_years_l229_229053

def is_adult_age (age : ℕ) := 18 ≤ age ∧ age < 40

def fits_pattern (p1 p2 n : ℕ) : Prop := 
  is_adult_age p1 ∧
  is_adult_age p2 ∧ 
  (∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 
    (k % (p1 + k) = 0 ∨ k % (p2 + k) = 0) ∧ ¬ (k % (p1 + k) = 0 ∧ k % (p2 + k) = 0))

theorem pattern_continues_for_max_8_years (p1 p2 : ℕ) : 
  fits_pattern p1 p2 8 := 
sorry

end pattern_continues_for_max_8_years_l229_229053


namespace other_car_speed_l229_229356

-- Definitions of the conditions
def red_car_speed : ℕ := 30
def initial_gap : ℕ := 20
def overtaking_time : ℕ := 1

-- Assertion of what needs to be proved
theorem other_car_speed : (initial_gap + red_car_speed * overtaking_time) = 50 :=
  sorry

end other_car_speed_l229_229356


namespace justin_home_time_l229_229304

noncomputable def dinner_duration : ℕ := 45
noncomputable def homework_duration : ℕ := 30
noncomputable def cleaning_room_duration : ℕ := 30
noncomputable def taking_out_trash_duration : ℕ := 5
noncomputable def emptying_dishwasher_duration : ℕ := 10

noncomputable def total_time_required : ℕ :=
  dinner_duration + homework_duration + cleaning_room_duration + taking_out_trash_duration + emptying_dishwasher_duration

noncomputable def latest_start_time_hour : ℕ := 18 -- 6 pm in 24-hour format
noncomputable def total_time_required_hours : ℕ := 2
noncomputable def movie_time_hour : ℕ := 20 -- 8 pm in 24-hour format

theorem justin_home_time : latest_start_time_hour - total_time_required_hours = 16 := -- 4 pm in 24-hour format
by
  sorry

end justin_home_time_l229_229304


namespace find_b_l229_229308

theorem find_b (x : ℝ) (b : ℝ) :
  (∃ t u : ℝ, (bx^2 + 18 * x + 9) = (t * x + u)^2 ∧ u^2 = 9 ∧ 2 * t * u = 18 ∧ t^2 = b) →
  b = 9 :=
by
  sorry

end find_b_l229_229308


namespace min_value_is_neg_one_l229_229056

noncomputable def find_min_value (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : ℝ :=
  1 / a + 2 / b + 4 / c

theorem min_value_is_neg_one (a b c : ℝ) (h : 4 * a^2 - 2 * a * b + b^2 - c = 0) (h1 : 0 < c) (h2 : |2 * a + b| = sorry) : 
  find_min_value a b c h h1 h2 = -1 :=
sorry

end min_value_is_neg_one_l229_229056


namespace find_length_of_AB_l229_229569

theorem find_length_of_AB (x y : ℝ) (AP PB AQ QB PQ AB : ℝ) 
  (h1 : AP = 3 * x) 
  (h2 : PB = 4 * x) 
  (h3 : AQ = 4 * y) 
  (h4 : QB = 5 * y)
  (h5 : PQ = 5) 
  (h6 : AP + PB = AB)
  (h7 : AQ + QB = AB)
  (h8 : PQ = AQ - AP)
  (h9 : 7 * x = 9 * y) : 
  AB = 315 := 
by
  sorry

end find_length_of_AB_l229_229569


namespace find_fraction_l229_229161

noncomputable def fraction_of_third (F N : ℝ) : Prop := F * (1 / 3 * N) = 30

noncomputable def fraction_of_number (G N : ℝ) : Prop := G * N = 75

noncomputable def product_is_90 (F N : ℝ) : Prop := F * N = 90

theorem find_fraction (F G N : ℝ) (h1 : fraction_of_third F N) (h2 : fraction_of_number G N) (h3 : product_is_90 F N) :
  G = 5 / 6 :=
sorry

end find_fraction_l229_229161


namespace sum_of_digits_N_l229_229093

-- Define a function to compute the least common multiple (LCM) of a list of numbers
def lcm_list (xs : List ℕ) : ℕ :=
  xs.foldr Nat.lcm 1

-- The set of numbers less than 8
def nums : List ℕ := [1, 2, 3, 4, 5, 6, 7]

-- The LCM of numbers less than 8
def N_lcm : ℕ := lcm_list nums

-- The second smallest positive integer that is divisible by every positive integer less than 8
def N : ℕ := 2 * N_lcm

-- Function to compute the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Prove that the sum of the digits of N is 12
theorem sum_of_digits_N : sum_of_digits N = 12 :=
by
  -- Necessary proof steps will be filled here
  sorry

end sum_of_digits_N_l229_229093


namespace geometric_sequence_second_term_l229_229824

theorem geometric_sequence_second_term (a r : ℝ) (h1 : a * r ^ 2 = 5) (h2 : a * r ^ 4 = 45) :
  a * r = 5 / 3 :=
by
  sorry

end geometric_sequence_second_term_l229_229824


namespace parallel_lines_k_l229_229246

theorem parallel_lines_k (k : ℝ) 
  (h₁ : k ≠ 0)
  (h₂ : ∀ x y : ℝ, (x - k * y - k = 0) = (y = (1 / k) * x - 1))
  (h₃ : ∀ x : ℝ, (y = k * (x - 1))) :
  k = -1 :=
by
  sorry

end parallel_lines_k_l229_229246


namespace average_weight_increase_per_month_l229_229373

theorem average_weight_increase_per_month (w_initial w_final : ℝ) (t : ℝ) 
  (h_initial : w_initial = 3.25) (h_final : w_final = 7) (h_time : t = 3) :
  (w_final - w_initial) / t = 1.25 := 
by 
  sorry

end average_weight_increase_per_month_l229_229373


namespace inequality_of_positive_numbers_l229_229323

theorem inequality_of_positive_numbers (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := 
sorry

end inequality_of_positive_numbers_l229_229323


namespace einstein_needs_more_money_l229_229102

-- Definitions based on conditions
def pizza_price : ℝ := 12
def fries_price : ℝ := 0.3
def soda_price : ℝ := 2
def pizzas_sold : ℕ := 15
def fries_sold : ℕ := 40
def sodas_sold : ℕ := 25
def goal : ℝ := 500

-- Total amount raised calculation
def total_raised : ℝ :=
  (pizzas_sold * pizza_price) +
  (fries_sold * fries_price) +
  (sodas_sold * soda_price)

-- Proof statement
theorem einstein_needs_more_money : goal - total_raised = 258 :=
by
  sorry

end einstein_needs_more_money_l229_229102


namespace charged_amount_is_35_l229_229104

-- Definitions based on conditions
def annual_interest_rate : ℝ := 0.05
def owed_amount : ℝ := 36.75
def time_in_years : ℝ := 1

-- The amount charged on the account in January
def charged_amount (P : ℝ) : Prop :=
  owed_amount = P + (P * annual_interest_rate * time_in_years)

-- The proof statement
theorem charged_amount_is_35 : charged_amount 35 := by
  sorry

end charged_amount_is_35_l229_229104


namespace solve_inequality_l229_229802

theorem solve_inequality (a x : ℝ) :
  (a = 0 → x < 1) ∧
  (a ≠ 0 → ((a > 0 → (a-1)/a < x ∧ x < 1) ∧
            (a < 0 → (x < 1 ∨ x > (a-1)/a)))) :=
by
  sorry

end solve_inequality_l229_229802


namespace reflection_matrix_condition_l229_229245

noncomputable def reflection_matrix (a b : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![a, b], ![-(3/4 : ℝ), 1/4]]

noncomputable def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![1, 0], ![0, 1]]

theorem reflection_matrix_condition (a b : ℝ) :
  (reflection_matrix a b)^2 = identity_matrix ↔ a = -(1/4) ∧ b = -(3/4) :=
  by
  sorry

end reflection_matrix_condition_l229_229245


namespace A_finishes_job_in_12_days_l229_229977

variable (A B : ℝ)

noncomputable def work_rate_A_and_B := (1 / 40)
noncomputable def work_rate_A := (1 / A)
noncomputable def work_rate_B := (1 / B)

theorem A_finishes_job_in_12_days
  (h1 : work_rate_A + work_rate_B = work_rate_A_and_B)
  (h2 : 10 * work_rate_A_and_B = 1 / 4)
  (h3 : 9 * work_rate_A = 3 / 4) :
  A = 12 :=
  sorry

end A_finishes_job_in_12_days_l229_229977


namespace sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l229_229835

variable (α β : ℝ)
variable (hα : α < π/2) (hβ : β < π/2) -- acute angles
variable (h1 : Real.cos (α + π/6) = 3/5)
variable (h2 : Real.cos (α + β) = -Real.sqrt 5 / 5)

theorem sin_2alpha_plus_pi_over_3 :
  Real.sin (2 * α + π/3) = 24 / 25 :=
by
  sorry

theorem cos_beta_minus_pi_over_6 :
  Real.cos (β - π/6) = Real.sqrt 5 / 5 :=
by
  sorry

end sin_2alpha_plus_pi_over_3_cos_beta_minus_pi_over_6_l229_229835


namespace convince_jury_l229_229289

def not_guilty : Prop := sorry  -- definition indicating the defendant is not guilty
def not_liar : Prop := sorry    -- definition indicating the defendant is not a liar
def innocent_knight_statement : Prop := sorry  -- statement "I am an innocent knight"

theorem convince_jury (not_guilty : not_guilty) (not_liar : not_liar) : innocent_knight_statement :=
sorry

end convince_jury_l229_229289


namespace total_cost_of_shirt_and_coat_l229_229534

-- Definition of the conditions
def shirt_cost : ℕ := 150
def one_third_of_coat (coat_cost: ℕ) : Prop := shirt_cost = coat_cost / 3

-- Theorem stating the problem to prove
theorem total_cost_of_shirt_and_coat (coat_cost : ℕ) (h : one_third_of_coat coat_cost) : shirt_cost + coat_cost = 600 :=
by 
  -- Proof goes here, using sorry as placeholder
  sorry

end total_cost_of_shirt_and_coat_l229_229534


namespace final_amount_after_5_years_l229_229648

-- Define conditions as hypotheses
def principal := 200
def final_amount_after_2_years := 260
def time_2_years := 2

-- Define our final question and answer as a Lean theorem
theorem final_amount_after_5_years : 
  (final_amount_after_2_years - principal) = principal * (rate * time_2_years) →
  (rate * 3) = 90 →
  final_amount_after_2_years + (principal * rate * 3) = 350 :=
by
  intros h1 h2
  -- Proof skipped using sorry
  sorry

end final_amount_after_5_years_l229_229648


namespace randy_piggy_bank_l229_229631

theorem randy_piggy_bank : 
  ∀ (initial_amount trips_per_month cost_per_trip months_per_year total_spent_left : ℕ),
  initial_amount = 200 →
  cost_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  total_spent_left = initial_amount - (cost_per_trip * trips_per_month * months_per_year) →
  total_spent_left = 104 :=
by
  intros initial_amount trips_per_month cost_per_trip months_per_year total_spent_left
  sorry

end randy_piggy_bank_l229_229631


namespace sum_x_y_l229_229016

theorem sum_x_y (x y : ℤ) (h1 : x - y = 40) (h2 : x = 32) : x + y = 24 := by
  sorry

end sum_x_y_l229_229016


namespace biology_marks_correct_l229_229482

-- Define the known marks in other subjects
def math_marks : ℕ := 76
def science_marks : ℕ := 65
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 62

-- Define the total number of subjects
def total_subjects : ℕ := 5

-- Define the average marks
def average_marks : ℕ := 74

-- Calculate the total marks of the known four subjects
def total_known_marks : ℕ := math_marks + science_marks + social_studies_marks + english_marks

-- Define a variable to represent the marks in biology
def biology_marks : ℕ := 370 - total_known_marks

-- Statement to prove
theorem biology_marks_correct : biology_marks = 85 := by
  sorry

end biology_marks_correct_l229_229482


namespace initial_inventory_correct_l229_229601

-- Define the conditions as given in the problem
def bottles_sold_monday : ℕ := 2445
def bottles_sold_tuesday : ℕ := 900
def bottles_sold_per_day_wed_to_sun : ℕ := 50
def days_wed_to_sun : ℕ := 5
def bottles_delivered_saturday : ℕ := 650
def final_inventory : ℕ := 1555

-- Define the total number of bottles sold during the week
def total_bottles_sold : ℕ :=
  bottles_sold_monday + bottles_sold_tuesday + (bottles_sold_per_day_wed_to_sun * days_wed_to_sun)

-- Define the initial inventory calculation
def initial_inventory : ℕ :=
  final_inventory + total_bottles_sold - bottles_delivered_saturday

-- The theorem we want to prove
theorem initial_inventory_correct :
  initial_inventory = 4500 :=
by
  sorry

end initial_inventory_correct_l229_229601


namespace combined_resistance_parallel_l229_229803

theorem combined_resistance_parallel (R1 R2 R3 : ℝ) (r : ℝ) (h1 : R1 = 2) (h2 : R2 = 5) (h3 : R3 = 6) :
  (1 / r) = (1 / R1) + (1 / R2) + (1 / R3) → r = 15 / 13 :=
by
  sorry

end combined_resistance_parallel_l229_229803


namespace prime_even_intersection_l229_229740

-- Define P as the set of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def P : Set ℕ := { n | is_prime n }

-- Define Q as the set of even numbers
def Q : Set ℕ := { n | n % 2 = 0 }

-- Statement to prove
theorem prime_even_intersection : P ∩ Q = {2} :=
by
  sorry

end prime_even_intersection_l229_229740


namespace valid_odd_and_increasing_functions_l229_229201

   def is_odd_function (f : ℝ → ℝ) : Prop :=
     ∀ x, f (-x) = -f (x)

   def is_increasing_function (f : ℝ → ℝ) : Prop :=
     ∀ x y, x < y → f (x) < f (y)

   noncomputable def f1 (x : ℝ) : ℝ := 3 * x^2
   noncomputable def f2 (x : ℝ) : ℝ := 6 * x
   noncomputable def f3 (x : ℝ) : ℝ := x * abs x
   noncomputable def f4 (x : ℝ) : ℝ := x + 1 / x

   theorem valid_odd_and_increasing_functions :
     (is_odd_function f2 ∧ is_increasing_function f2) ∧
     (is_odd_function f3 ∧ is_increasing_function f3) :=
   by
     sorry -- Proof goes here
   
end valid_odd_and_increasing_functions_l229_229201


namespace find_base_tax_rate_l229_229130

noncomputable def income : ℝ := 10550
noncomputable def tax_paid : ℝ := 950
noncomputable def base_income : ℝ := 5000
noncomputable def excess_income : ℝ := income - base_income
noncomputable def excess_tax_rate : ℝ := 0.10

theorem find_base_tax_rate (base_tax_rate: ℝ) :
  base_tax_rate * base_income + excess_tax_rate * excess_income = tax_paid -> 
  base_tax_rate = 7.9 / 100 :=
by sorry

end find_base_tax_rate_l229_229130


namespace managers_participation_l229_229241

theorem managers_participation (teams : ℕ) (people_per_team : ℕ) (employees : ℕ) (total_people : teams * people_per_team = 6) (num_employees : employees = 3) :
  teams * people_per_team - employees = 3 :=
by
  sorry

end managers_participation_l229_229241


namespace malcolm_needs_more_lights_l229_229399

def red_lights := 12
def blue_lights := 3 * red_lights
def green_lights := 6
def white_lights := 59

def colored_lights := red_lights + blue_lights + green_lights
def need_more_lights := white_lights - colored_lights

theorem malcolm_needs_more_lights :
  need_more_lights = 5 :=
by
  sorry

end malcolm_needs_more_lights_l229_229399


namespace shared_friends_l229_229402

theorem shared_friends (crackers total_friends : ℕ) (each_friend_crackers : ℕ) 
  (h1 : crackers = 22) 
  (h2 : each_friend_crackers = 2)
  (h3 : crackers = each_friend_crackers * total_friends) 
  : total_friends = 11 := by 
  sorry

end shared_friends_l229_229402


namespace five_letter_words_start_end_same_l229_229606

def num_five_letter_words_start_end_same : ℕ :=
  26 ^ 4

theorem five_letter_words_start_end_same :
  num_five_letter_words_start_end_same = 456976 :=
by
  -- Sorry is used as a placeholder for the proof.
  sorry

end five_letter_words_start_end_same_l229_229606


namespace hulk_jump_distance_l229_229070

theorem hulk_jump_distance :
  ∃ n : ℕ, 3^n > 1500 ∧ ∀ m < n, 3^m ≤ 1500 := 
sorry

end hulk_jump_distance_l229_229070


namespace pages_left_to_read_l229_229420

theorem pages_left_to_read (total_pages : ℕ) (pages_read : ℕ) (pages_skipped : ℕ) : 
  total_pages = 372 → pages_read = 125 → pages_skipped = 16 → (total_pages - (pages_read + pages_skipped)) = 231 :=
by
  intros
  sorry

end pages_left_to_read_l229_229420


namespace gumballs_initial_count_l229_229428

noncomputable def initial_gumballs := (34.3 / (0.7 ^ 3))

theorem gumballs_initial_count :
  initial_gumballs = 100 :=
sorry

end gumballs_initial_count_l229_229428


namespace percentage_selected_in_state_B_l229_229005

theorem percentage_selected_in_state_B (appeared: ℕ) (selectedA: ℕ) (selected_diff: ℕ)
  (percentage_selectedA: ℝ)
  (h1: appeared = 8100)
  (h2: percentage_selectedA = 6.0)
  (h3: selectedA = appeared * (percentage_selectedA / 100))
  (h4: selected_diff = 81)
  (h5: selectedB = selectedA + selected_diff) :
  ((selectedB : ℝ) / appeared) * 100 = 7 := 
  sorry

end percentage_selected_in_state_B_l229_229005


namespace min_m_value_inequality_x2y2z_l229_229292

theorem min_m_value (a b : ℝ) (h1 : a * b > 0) (h2 : a^2 * b = 2) : 
  ∃ (m : ℝ), m = a * b + a^2 ∧ m = 3 :=
sorry

theorem inequality_x2y2z 
  (t : ℝ) (ht : t = 3) (x y z : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : z > 0) (h4 : x^2 + y^2 + z^2 = t / 3) : 
  |x + 2 * y + 2 * z| ≤ 3 :=
sorry

end min_m_value_inequality_x2y2z_l229_229292


namespace circle_tangent_to_x_axis_at_origin_l229_229217

theorem circle_tangent_to_x_axis_at_origin
  (D E F : ℝ)
  (h1 : ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 → x = 0 ∧ y = 0 ∨ y = -D/E ∧ x = 0 ∧ F = 0):
  D = 0 ∧ E ≠ 0 ∧ F = 0 :=
sorry

end circle_tangent_to_x_axis_at_origin_l229_229217


namespace converse_even_sum_l229_229262

variable (a b : ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem converse_even_sum (h : is_even (a + b)) : is_even a ∧ is_even b :=
sorry

end converse_even_sum_l229_229262


namespace pet_center_final_count_l229_229615

def initial_dogs : Nat := 36
def initial_cats : Nat := 29
def adopted_dogs : Nat := 20
def collected_cats : Nat := 12
def final_pets : Nat := 57

theorem pet_center_final_count :
  (initial_dogs - adopted_dogs) + (initial_cats + collected_cats) = final_pets := 
by
  sorry

end pet_center_final_count_l229_229615


namespace john_initial_pens_l229_229585

theorem john_initial_pens (P S C : ℝ) (n : ℕ) 
  (h1 : 20 * S = P) 
  (h2 : C = (2 / 3) * S) 
  (h3 : n * C = P)
  (h4 : P > 0) 
  (h5 : S > 0) 
  (h6 : C > 0)
  : n = 30 :=
by
  sorry

end john_initial_pens_l229_229585


namespace river_bank_depth_l229_229265

-- Definitions related to the problem
def is_trapezium (top_width bottom_width height area : ℝ) :=
  area = 1 / 2 * (top_width + bottom_width) * height

-- The theorem we want to prove
theorem river_bank_depth :
  ∀ (top_width bottom_width area : ℝ), 
    top_width = 12 → 
    bottom_width = 8 → 
    area = 500 → 
    ∃ h : ℝ, is_trapezium top_width bottom_width h area ∧ h = 50 :=
by
  intros top_width bottom_width area ht hb ha
  sorry

end river_bank_depth_l229_229265


namespace batsman_sixes_l229_229074

theorem batsman_sixes (total_runs : ℕ) (boundaries : ℕ) (running_percentage : ℝ) (score_per_boundary : ℕ) (score_per_six : ℕ)
  (h1 : total_runs = 150)
  (h2 : boundaries = 5)
  (h3 : running_percentage = 66.67)
  (h4 : score_per_boundary = 4)
  (h5 : score_per_six = 6) :
  ∃ (sixes : ℕ), sixes = 5 :=
by
  -- Calculations omitted
  existsi 5
  sorry

end batsman_sixes_l229_229074


namespace train_stops_time_l229_229096

/-- Given the speeds of a train excluding and including stoppages, 
calculate the stopping time in minutes per hour. --/
theorem train_stops_time
  (speed_excluding_stoppages : ℝ)
  (speed_including_stoppages : ℝ)
  (h1 : speed_excluding_stoppages = 48)
  (h2 : speed_including_stoppages = 40) :
  ∃ minutes_stopped : ℝ, minutes_stopped = 10 :=
by
  sorry

end train_stops_time_l229_229096


namespace add_to_divisible_l229_229301

theorem add_to_divisible (n d x : ℕ) (h : n = 987654) (h1 : d = 456) (h2 : x = 222) : 
  (n + x) % d = 0 := 
by {
  sorry
}

end add_to_divisible_l229_229301


namespace max_value_expression_l229_229674

theorem max_value_expression (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : 2 * a + 3 * b = 5) : 
  (∀ x y : ℝ, x = 2 * a + 2 → y = 3 * b + 1 → x * y ≤ 16) := by
  sorry

end max_value_expression_l229_229674


namespace circle_radius_tangent_to_ellipse_l229_229831

theorem circle_radius_tangent_to_ellipse (r : ℝ) :
  (∀ x y : ℝ, (x - r)^2 + y^2 = r^2 → x^2 + 4*y^2 = 8) ↔ r = (Real.sqrt 6) / 2 :=
by
  sorry

end circle_radius_tangent_to_ellipse_l229_229831


namespace greg_total_earnings_l229_229930

-- Define the charges and walking times as given
def charge_per_dog : ℕ := 20
def charge_per_minute : ℕ := 1
def one_dog_minutes : ℕ := 10
def two_dogs_minutes : ℕ := 7
def three_dogs_minutes : ℕ := 9
def total_dogs_one : ℕ := 1
def total_dogs_two : ℕ := 2
def total_dogs_three : ℕ := 3

-- Total earnings computation
def earnings_one_dog : ℕ := charge_per_dog + charge_per_minute * one_dog_minutes
def earnings_two_dogs : ℕ := total_dogs_two * charge_per_dog + total_dogs_two * charge_per_minute * two_dogs_minutes
def earnings_three_dogs : ℕ := total_dogs_three * charge_per_dog + total_dogs_three * charge_per_minute * three_dogs_minutes
def total_earnings : ℕ := earnings_one_dog + earnings_two_dogs + earnings_three_dogs

-- The proof: Greg's total earnings should be $171
theorem greg_total_earnings : total_earnings = 171 := by
  -- Placeholder for the proof (not required as per the instructions)
  sorry

end greg_total_earnings_l229_229930


namespace inequality_product_lt_zero_l229_229814

theorem inequality_product_lt_zero (a b c : ℝ) (h1 : a > b) (h2 : c < 1) : (a - b) * (c - 1) < 0 :=
  sorry

end inequality_product_lt_zero_l229_229814


namespace odd_integer_divisibility_l229_229470

theorem odd_integer_divisibility (n : ℕ) (hodd : n % 2 = 1) (hpos : n > 0) : ∃ k : ℕ, n^4 - n^2 - n = n * k := 
sorry

end odd_integer_divisibility_l229_229470


namespace hem_dress_time_l229_229738

theorem hem_dress_time
  (hem_length_feet : ℕ)
  (stitch_length_inches : ℝ)
  (stitches_per_minute : ℕ)
  (hem_length_inches : ℝ)
  (total_stitches : ℕ)
  (time_minutes : ℝ)
  (h1 : hem_length_feet = 3)
  (h2 : stitch_length_inches = 1 / 4)
  (h3 : stitches_per_minute = 24)
  (h4 : hem_length_inches = 12 * hem_length_feet)
  (h5 : total_stitches = hem_length_inches / stitch_length_inches)
  (h6 : time_minutes = total_stitches / stitches_per_minute) :
  time_minutes = 6 := 
sorry

end hem_dress_time_l229_229738


namespace quadratic_real_roots_iff_l229_229690

-- Define the statement of the problem in Lean
theorem quadratic_real_roots_iff (m : ℝ) :
  (∃ x : ℂ, m * x^2 + 2 * x - 1 = 0) ↔ (m ≥ -1 ∧ m ≠ 0) := 
by
  sorry

end quadratic_real_roots_iff_l229_229690


namespace greater_savings_on_hat_l229_229181

theorem greater_savings_on_hat (savings_shoes spent_shoes savings_hat sale_price_hat : ℝ) 
  (h1 : savings_shoes = 3.75)
  (h2 : spent_shoes = 42.25)
  (h3 : savings_hat = 1.80)
  (h4 : sale_price_hat = 18.20) :
  ((savings_hat / (sale_price_hat + savings_hat)) * 100) > ((savings_shoes / (spent_shoes + savings_shoes)) * 100) :=
by
  sorry

end greater_savings_on_hat_l229_229181


namespace triangle_area_is_correct_l229_229085

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_area_is_correct :
  area_of_triangle (0, 3) (4, -2) (9, 6) = 16.5 :=
by
  sorry

end triangle_area_is_correct_l229_229085


namespace interest_rate_condition_l229_229376

theorem interest_rate_condition 
    (P1 P2 : ℝ) 
    (R2 : ℝ) 
    (T1 T2 : ℝ) 
    (SI500 SI160 : ℝ) 
    (H1: SI500 = (P1 * R2 * T1) / 100) 
    (H2: SI160 = (P2 * (25 / 100))):
  25 * (160 / 100) / 12.5  = 6.4 :=
by
  sorry

end interest_rate_condition_l229_229376


namespace expressions_equal_iff_conditions_l229_229193

theorem expressions_equal_iff_conditions (a b c : ℝ) :
  (2 * a + 3 * b * c = (a + 2 * b) * (2 * a + 3 * c)) ↔ (a = 0 ∨ a + 2 * b + 1.5 * c = 0) :=
by
  sorry

end expressions_equal_iff_conditions_l229_229193


namespace compute_expression_l229_229220

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l229_229220


namespace abcd_product_l229_229990

theorem abcd_product :
  let A := (Real.sqrt 3003 + Real.sqrt 3004)
  let B := (-Real.sqrt 3003 - Real.sqrt 3004)
  let C := (Real.sqrt 3003 - Real.sqrt 3004)
  let D := (Real.sqrt 3004 - Real.sqrt 3003)
  A * B * C * D = 1 := 
by
  sorry

end abcd_product_l229_229990


namespace appropriate_sampling_method_l229_229406

def total_families := 500
def high_income_families := 125
def middle_income_families := 280
def low_income_families := 95
def sample_size := 100
def influenced_by_income := True

theorem appropriate_sampling_method
  (htotal : total_families = 500)
  (hhigh : high_income_families = 125)
  (hmiddle : middle_income_families = 280)
  (hlow : low_income_families = 95)
  (hsample : sample_size = 100)
  (hinfluence : influenced_by_income = True) :
  ∃ method, method = "Stratified sampling method" :=
sorry

end appropriate_sampling_method_l229_229406


namespace inverse_solution_correct_l229_229452

noncomputable def f (a b c x : ℝ) : ℝ :=
  1 / (a * x^2 + b * x + c)

theorem inverse_solution_correct (a b c x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c x = 1 ↔ x = (-b + Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) ∨
               x = (-b - Real.sqrt (b^2 - 4 * a * (c - 1))) / (2 * a) :=
by
  sorry

end inverse_solution_correct_l229_229452


namespace count_elements_in_A_l229_229733

variables (a b : ℕ)

def condition1 : Prop := a = 3 * b / 2
def condition2 : Prop := a + b - 1200 = 4500

theorem count_elements_in_A (h1 : condition1 a b) (h2 : condition2 a b) : a = 3420 :=
by sorry

end count_elements_in_A_l229_229733


namespace maximum_bottles_l229_229436

-- Definitions for the number of bottles each shop sells
def bottles_from_shop_A : ℕ := 150
def bottles_from_shop_B : ℕ := 180
def bottles_from_shop_C : ℕ := 220

-- The main statement to prove
theorem maximum_bottles : bottles_from_shop_A + bottles_from_shop_B + bottles_from_shop_C = 550 := 
by 
  sorry

end maximum_bottles_l229_229436


namespace computation_result_l229_229317

theorem computation_result :
  let a := -6
  let b := 25
  let c := -39
  let d := 40
  9 * a + 3 * b + 6 * c + d = -173 := by
  sorry

end computation_result_l229_229317


namespace pasha_game_solvable_l229_229205

def pasha_game : Prop :=
∃ (a : Fin 2017 → ℕ), 
  (∀ i, a i > 0) ∧
  (∃ (moves : ℕ), moves = 43 ∧
   (∀ (box_contents : Fin 2017 → ℕ), 
    (∀ j, box_contents j = 0) →
    (∃ (equal_count : ℕ),
      (∀ j, box_contents j = equal_count)
      ∧
      (∀ m < 43,
        ∃ j, box_contents j ≠ equal_count))))

theorem pasha_game_solvable : pasha_game :=
by
  sorry

end pasha_game_solvable_l229_229205


namespace reciprocal_proof_l229_229850

theorem reciprocal_proof :
  (-2) * (-(1 / 2)) = 1 := 
by 
  sorry

end reciprocal_proof_l229_229850


namespace david_initial_money_l229_229092

-- Given conditions as definitions
def spent (S : ℝ) : Prop := S - 800 = 500
def has_left (H : ℝ) : Prop := H = 500

-- The main theorem to prove
theorem david_initial_money (S : ℝ) (X : ℝ) (H : ℝ)
  (h1 : spent S) 
  (h2 : has_left H) 
  : X = S + H → X = 1800 :=
by
  sorry

end david_initial_money_l229_229092


namespace simplify_fraction_1_210_plus_17_35_l229_229650

theorem simplify_fraction_1_210_plus_17_35 :
  1 / 210 + 17 / 35 = 103 / 210 :=
by sorry

end simplify_fraction_1_210_plus_17_35_l229_229650


namespace intersection_A_B_l229_229555

def A : Set ℤ := {-2, -1, 1, 2}

def B : Set ℤ := {x | x^2 - x - 2 ≥ 0}

theorem intersection_A_B : (A ∩ B) = {-2, -1, 2} := by
  sorry

end intersection_A_B_l229_229555


namespace expression_not_defined_l229_229783

-- Define the quadratic equation
def quadratic_eq (x : ℝ) : ℝ := x^2 - 25*x + 125

-- Theorem statement that the expression is not defined for specific values of x
theorem expression_not_defined (x : ℝ) : quadratic_eq x = 0 ↔ (x = 5 ∨ x = 20) :=
by
  sorry

end expression_not_defined_l229_229783


namespace polynomial_roots_sum_l229_229651

theorem polynomial_roots_sum (p q : ℂ) (hp : p + q = 5) (hq : p * q = 7) : 
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 559 := 
by 
  sorry

end polynomial_roots_sum_l229_229651


namespace remaining_paint_fraction_l229_229075

theorem remaining_paint_fraction :
  ∀ (initial_paint : ℝ) (half_usage : ℕ → ℝ → ℝ),
    initial_paint = 2 →
    half_usage 0 (2 : ℝ) = 1 →
    half_usage 1 (1 : ℝ) = 0.5 →
    half_usage 2 (0.5 : ℝ) = 0.25 →
    half_usage 3 (0.25 : ℝ) = (0.25 / initial_paint) := by
  sorry

end remaining_paint_fraction_l229_229075


namespace monitor_width_l229_229457

theorem monitor_width (d w h : ℝ) (h_ratio : w / h = 16 / 9) (h_diag : d = 24) :
  w = 384 / Real.sqrt 337 :=
by
  sorry

end monitor_width_l229_229457


namespace quadratic_has_distinct_real_roots_l229_229478

theorem quadratic_has_distinct_real_roots (m : ℝ) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = m - 1 ∧ (b^2 - 4 * a * c > 0) → (m < 2) :=
by
  sorry

end quadratic_has_distinct_real_roots_l229_229478


namespace find_number_l229_229689

theorem find_number (X : ℝ) (h : 0.8 * X = 0.7 * 60.00000000000001 + 30) : X = 90.00000000000001 :=
sorry

end find_number_l229_229689


namespace find_C_l229_229361

theorem find_C (A B C D : ℕ) (h_diff : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_eq : 4000 + 100 * A + 50 + B + (1000 * C + 200 + 10 * D + 7) = 7070) : C = 2 :=
sorry

end find_C_l229_229361


namespace range_of_a_l229_229147

-- Lean statement that represents the proof problem
theorem range_of_a 
  (h1 : ∀ x y : ℝ, x^2 - 2 * x + Real.log (2 * y^2 - y) = 0 → x > 0 ∧ y < 0 ∨ x < 0 ∧ y > 0)
  (h2 : ∀ b : ℝ, 2 * b^2 - b > 0) :
  (∀ a : ℝ, x^2 - 2 * x + Real.log (2 * a^2 - a) = 0 → (- (1:ℝ) / 2) < a ∧ a < 0 ∨ (1 / 2) < a ∧ a < 1) :=
sorry

end range_of_a_l229_229147


namespace asia_discount_problem_l229_229023

theorem asia_discount_problem
  (originalPrice : ℝ)
  (storeDiscount : ℝ)
  (memberDiscount : ℝ)
  (finalPriceUSD : ℝ)
  (exchangeRate : ℝ)
  (finalDiscountPercentage : ℝ) :
  originalPrice = 300 →
  storeDiscount = 0.20 →
  memberDiscount = 0.10 →
  finalPriceUSD = 224 →
  exchangeRate = 1.10 →
  finalDiscountPercentage = 28 :=
by
  sorry

end asia_discount_problem_l229_229023


namespace woman_waits_time_until_man_catches_up_l229_229114

theorem woman_waits_time_until_man_catches_up
  (woman_speed : ℝ)
  (man_speed : ℝ)
  (wait_time : ℝ)
  (woman_slows_after : ℝ)
  (h_man_speed : man_speed = 5 / 60) -- man's speed in miles per minute
  (h_woman_speed : woman_speed = 25 / 60) -- woman's speed in miles per minute
  (h_wait_time : woman_slows_after = 5) -- the time in minutes after which the woman waits for man
  (h_woman_waits : wait_time = 25) : wait_time = (woman_slows_after * woman_speed) / man_speed :=
sorry

end woman_waits_time_until_man_catches_up_l229_229114


namespace pounds_in_one_ton_is_2600_l229_229907

variable (pounds_in_one_ton : ℕ)
variable (ounces_in_one_pound : ℕ := 16)
variable (packets : ℕ := 2080)
variable (weight_per_packet_pounds : ℕ := 16)
variable (weight_per_packet_ounces : ℕ := 4)
variable (gunny_bag_capacity_tons : ℕ := 13)

theorem pounds_in_one_ton_is_2600 :
  (packets * (weight_per_packet_pounds + weight_per_packet_ounces / ounces_in_one_pound)) = (gunny_bag_capacity_tons * pounds_in_one_ton) →
  pounds_in_one_ton = 2600 :=
sorry

end pounds_in_one_ton_is_2600_l229_229907


namespace area_of_rectangle_l229_229437

noncomputable def leanProblem : Prop :=
  let E := 8
  let F := 2.67
  let BE := E -- length from B to E on AB
  let AF := F -- length from A to F on AD
  let BC := E * (Real.sqrt 3) -- from triangle properties CB is BE * sqrt(3)
  let FD := BC - F -- length from F to D on AD
  let CD := FD * (Real.sqrt 3) -- applying the triangle properties again
  (BC * CD = 192 * (Real.sqrt 3) - 64.08)

theorem area_of_rectangle (E : ℝ) (F : ℝ) 
  (hE : E = 8) 
  (hF : F = 2.67) 
  (BC : ℝ) (CD : ℝ) :
  leanProblem :=
by 
  sorry

end area_of_rectangle_l229_229437


namespace compositeQuotientCorrect_l229_229454

namespace CompositeNumbersProof

def firstFiveCompositesProduct : ℕ :=
  21 * 22 * 24 * 25 * 26

def subsequentFiveCompositesProduct : ℕ :=
  27 * 28 * 30 * 32 * 33

def compositeQuotient : ℚ :=
  firstFiveCompositesProduct / subsequentFiveCompositesProduct

theorem compositeQuotientCorrect : compositeQuotient = 1 / 1964 := by sorry

end CompositeNumbersProof

end compositeQuotientCorrect_l229_229454


namespace perpendicular_angles_l229_229091

theorem perpendicular_angles (α : ℝ) 
  (h1 : 4 * Real.pi < α) 
  (h2 : α < 6 * Real.pi)
  (h3 : ∃ (k : ℤ), α = -2 * Real.pi / 3 + Real.pi / 2 + k * Real.pi) :
  α = 29 * Real.pi / 6 ∨ α = 35 * Real.pi / 6 :=
by
  sorry

end perpendicular_angles_l229_229091


namespace initial_milk_water_ratio_l229_229688

theorem initial_milk_water_ratio
  (M W : ℕ)
  (h1 : M + W = 40000)
  (h2 : (M : ℚ) / (W + 1600) = 3 / 1) :
  (M : ℚ) / W = 3.55 :=
by
  sorry

end initial_milk_water_ratio_l229_229688


namespace consecutive_integers_product_sum_l229_229522

theorem consecutive_integers_product_sum (a b c d : ℕ) :
  a * b * c * d = 3024 ∧ b = a + 1 ∧ c = b + 1 ∧ d = c + 1 → a + b + c + d = 30 :=
by
  sorry

end consecutive_integers_product_sum_l229_229522


namespace sqrt_expression_l229_229330

theorem sqrt_expression (x : ℝ) : 2 - x ≥ 0 ↔ x ≤ 2 := sorry

end sqrt_expression_l229_229330


namespace john_total_skateboarded_miles_l229_229325

-- Definitions
def distance_skateboard_to_park := 16
def distance_walk := 8
def distance_bike := 6
def distance_skateboard_home := distance_skateboard_to_park

-- Statement to prove
theorem john_total_skateboarded_miles : 
  distance_skateboard_to_park + distance_skateboard_home = 32 := 
by
  sorry

end john_total_skateboarded_miles_l229_229325


namespace f_g_minus_g_f_l229_229994

-- Defining the functions f and g
def f (x : ℝ) : ℝ := x^2 + 3
def g (x : ℝ) : ℝ := 3 * x^2 + 5

-- Proving the given math problem
theorem f_g_minus_g_f :
  f (g 2) - g (f 2) = 140 := by
sorry

end f_g_minus_g_f_l229_229994


namespace obtuse_angle_probability_l229_229502

noncomputable def probability_obtuse_angle : ℝ :=
  let F : ℝ × ℝ := (0, 3)
  let G : ℝ × ℝ := (5, 0)
  let H : ℝ × ℝ := (2 * Real.pi + 2, 0)
  let I : ℝ × ℝ := (2 * Real.pi + 2, 3)
  let rectangle_area : ℝ := (2 * Real.pi + 2) * 3
  let semicircle_radius : ℝ := Real.sqrt (2.5^2 + 1.5^2)
  let semicircle_area : ℝ := (1 / 2) * Real.pi * semicircle_radius^2
  semicircle_area / rectangle_area

theorem obtuse_angle_probability :
  probability_obtuse_angle = 17 / (24 + 4 * Real.pi) :=
by
  sorry

end obtuse_angle_probability_l229_229502


namespace solution_set_of_inequality_l229_229766

open Set

theorem solution_set_of_inequality :
  {x : ℝ | - x ^ 2 - 4 * x + 5 > 0} = {x : ℝ | -5 < x ∧ x < 1} :=
sorry

end solution_set_of_inequality_l229_229766


namespace distance_A_B_l229_229716

variable (x : ℚ)

def pointA := x
def pointB := 1
def pointC := -1

theorem distance_A_B : |pointA x - pointB| = |x - 1| := by
  sorry

end distance_A_B_l229_229716


namespace divisor_of_p_l229_229616

theorem divisor_of_p (p q r s : ℕ) (h₁ : Nat.gcd p q = 30) (h₂ : Nat.gcd q r = 45) (h₃ : Nat.gcd r s = 75) (h₄ : 120 < Nat.gcd s p) (h₅ : Nat.gcd s p < 180) : 5 ∣ p := 
sorry

end divisor_of_p_l229_229616


namespace probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l229_229384

noncomputable def total_cassettes : ℕ := 30
noncomputable def disco_cassettes : ℕ := 12
noncomputable def classical_cassettes : ℕ := 18

-- Part (a): DJ returns the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_returned :
  (disco_cassettes / total_cassettes) * (disco_cassettes / total_cassettes) = 4 / 25 :=
by
  sorry

-- Part (b): DJ does not return the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_not_returned :
  (disco_cassettes / total_cassettes) * ((disco_cassettes - 1) / (total_cassettes - 1)) = 22 / 145 :=
by
  sorry

end probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l229_229384


namespace sandy_comic_books_l229_229206

-- Problem definition
def initial_comic_books := 14
def sold_comic_books := initial_comic_books / 2
def remaining_comic_books := initial_comic_books - sold_comic_books
def bought_comic_books := 6
def final_comic_books := remaining_comic_books + bought_comic_books

-- Proof statement
theorem sandy_comic_books : final_comic_books = 13 := by
  sorry

end sandy_comic_books_l229_229206


namespace find_xy_l229_229988

theorem find_xy (x y : ℝ) (h : (x - 13)^2 + (y - 14)^2 + (x - y)^2 = 1/3) : 
  x = 40/3 ∧ y = 41/3 :=
sorry

end find_xy_l229_229988


namespace inversely_proportional_find_p_l229_229855

theorem inversely_proportional_find_p (p q : ℕ) (h1 : p * 8 = 160) (h2 : q = 10) : p * q = 160 → p = 16 :=
by
  intro h
  sorry

end inversely_proportional_find_p_l229_229855


namespace find_a_extreme_value_l229_229710

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - x - a * x

theorem find_a_extreme_value :
  (∃ a : ℝ, ∀ x, f x a = Real.log (x + 1) - x - a * x ∧ (∃ m : ℝ, ∀ y : ℝ, f y a ≤ m)) ↔ a = -1 / 2 :=
by
  sorry

end find_a_extreme_value_l229_229710


namespace evaluate_i_powers_sum_l229_229338

-- Given conditions: i is the imaginary unit
def i : ℂ := Complex.I

-- Proof problem: Prove that i^2023 + i^2024 + i^2025 + i^2026 = 0
theorem evaluate_i_powers_sum : i^2023 + i^2024 + i^2025 + i^2026 = 0 := 
by sorry

end evaluate_i_powers_sum_l229_229338


namespace shaded_areas_are_different_l229_229762

theorem shaded_areas_are_different :
  let shaded_area_I := 3 / 8
  let shaded_area_II := 1 / 3
  let shaded_area_III := 1 / 2
  (shaded_area_I ≠ shaded_area_II) ∧ (shaded_area_I ≠ shaded_area_III) ∧ (shaded_area_II ≠ shaded_area_III) :=
by
  sorry

end shaded_areas_are_different_l229_229762


namespace foreign_students_next_sem_eq_740_l229_229622

def total_students : ℕ := 1800
def percentage_foreign : ℕ := 30
def new_foreign_students : ℕ := 200

def initial_foreign_students : ℕ := total_students * percentage_foreign / 100
def total_foreign_students_next_semester : ℕ :=
  initial_foreign_students + new_foreign_students

theorem foreign_students_next_sem_eq_740 :
  total_foreign_students_next_semester = 740 :=
by
  sorry

end foreign_students_next_sem_eq_740_l229_229622


namespace triangle_angle_B_l229_229450

theorem triangle_angle_B {A B C : ℝ} (h1 : A = 60) (h2 : B = 2 * C) (h3 : A + B + C = 180) : B = 80 :=
sorry

end triangle_angle_B_l229_229450


namespace find_n_in_range_l229_229875

theorem find_n_in_range :
  ∃ n : ℕ, n > 1 ∧ 
           n % 3 = 2 ∧ 
           n % 5 = 2 ∧ 
           n % 7 = 2 ∧ 
           101 ≤ n ∧ n ≤ 134 :=
by sorry

end find_n_in_range_l229_229875


namespace floor_width_l229_229498

theorem floor_width (tile_length tile_width floor_length max_tiles : ℕ) (h1 : tile_length = 25) (h2 : tile_width = 65) (h3 : floor_length = 150) (h4 : max_tiles = 36) :
  ∃ floor_width : ℕ, floor_width = 450 :=
by
  sorry

end floor_width_l229_229498


namespace max_distance_between_circle_and_ellipse_l229_229573

noncomputable def max_distance_PQ : ℝ :=
  1 + (3 * Real.sqrt 6) / 2

theorem max_distance_between_circle_and_ellipse :
  ∀ (P Q : ℝ × ℝ), (P.1^2 + (P.2 - 2)^2 = 1) → 
                   (Q.1^2 / 9 + Q.2^2 = 1) →
                   dist P Q ≤ max_distance_PQ :=
by
  intros P Q hP hQ
  sorry

end max_distance_between_circle_and_ellipse_l229_229573


namespace number_of_numbers_is_ten_l229_229503

open Nat

-- Define the conditions as given
variable (n : ℕ) -- Total number of numbers
variable (incorrect_average correct_average incorrect_value correct_value : ℤ)
variable (h1 : incorrect_average = 16)
variable (h2 : correct_average = 17)
variable (h3 : incorrect_value = 25)
variable (h4 : correct_value = 35)

-- Define the proof problem
theorem number_of_numbers_is_ten
  (h1 : incorrect_average = 16)
  (h2 : correct_average = 17)
  (h3 : incorrect_value = 25)
  (h4 : correct_value = 35)
  (h5 : ∀ (x : ℤ), x ≠ incorrect_value → incorrect_average * (n : ℤ) + x = correct_average * (n : ℤ) + correct_value - incorrect_value)
  : n = 10 := 
sorry

end number_of_numbers_is_ten_l229_229503


namespace total_pairs_of_jeans_purchased_l229_229736

-- Definitions based on the problem conditions
def price_fox : ℝ := 15
def price_pony : ℝ := 18
def discount_save : ℝ := 8.64
def pairs_fox : ℕ := 3
def pairs_pony : ℕ := 2
def sum_discount_rate : ℝ := 0.22
def discount_rate_pony : ℝ := 0.13999999999999993

-- Lean 4 statement to prove the total number of pairs of jeans purchased
theorem total_pairs_of_jeans_purchased :
  pairs_fox + pairs_pony = 5 :=
by
  sorry

end total_pairs_of_jeans_purchased_l229_229736


namespace smallest_integer_consecutive_set_l229_229929

theorem smallest_integer_consecutive_set :
  ∃ m : ℤ, (m+3 < 3*m - 5) ∧ (∀ n : ℤ, (n+3 < 3*n - 5) → n ≥ m) ∧ m = 5 :=
by
  sorry

end smallest_integer_consecutive_set_l229_229929


namespace parabola_focus_coordinates_l229_229862

open Real

theorem parabola_focus_coordinates (x y : ℝ) (h : y^2 = 6 * x) : (x, y) = (3 / 2, 0) :=
  sorry

end parabola_focus_coordinates_l229_229862


namespace complement_of_A_in_U_l229_229106

open Set

def U : Set ℕ := {x | x < 8}
def A : Set ℕ := {x | (x - 1) * (x - 3) * (x - 4) * (x - 7) = 0}

theorem complement_of_A_in_U : (U \ A) = {0, 2, 5, 6} := by
  sorry

end complement_of_A_in_U_l229_229106


namespace problem1_problem2_l229_229918

-- Problem 1: Prove the expression evaluates to 8
theorem problem1 : (1:ℝ) * (- (1 / 2)⁻¹) + (3 - Real.pi)^0 + (-3)^2 = 8 := 
by
  sorry

-- Problem 2: Prove the expression simplifies to 9a^6 - 2a^2
theorem problem2 (a : ℝ) : a^2 * a^4 - (-2 * a^2)^3 - 3 * a^2 + a^2 = 9 * a^6 - 2 * a^2 := 
by
  sorry

end problem1_problem2_l229_229918


namespace cost_of_fencing_each_side_l229_229401

theorem cost_of_fencing_each_side (total_cost : ℕ) (num_sides : ℕ) (h1 : total_cost = 288) (h2 : num_sides = 4) : (total_cost / num_sides) = 72 := by
  sorry

end cost_of_fencing_each_side_l229_229401


namespace book_total_pages_l229_229103

theorem book_total_pages (P : ℕ) (days_read : ℕ) (pages_per_day : ℕ) (fraction_read : ℚ) 
  (total_pages_read : ℕ) :
  (days_read = 15 ∧ pages_per_day = 12 ∧ fraction_read = 3 / 4 ∧ total_pages_read = 180 ∧ 
    total_pages_read = days_read * pages_per_day ∧ total_pages_read = fraction_read * P) → 
    P = 240 :=
by
  intros h
  sorry

end book_total_pages_l229_229103


namespace inequality_solution_set_non_empty_l229_229978

theorem inequality_solution_set_non_empty (a : ℝ) :
  (∃ x : ℝ, a * x > -1 ∧ x + a > 0) ↔ a > -1 :=
sorry

end inequality_solution_set_non_empty_l229_229978


namespace sqrt_five_minus_one_range_l229_229282

theorem sqrt_five_minus_one_range (h : 2 < Real.sqrt 5 ∧ Real.sqrt 5 < 3) : 
  1 < Real.sqrt 5 - 1 ∧ Real.sqrt 5 - 1 < 2 := 
by 
  sorry

end sqrt_five_minus_one_range_l229_229282


namespace quadratic_root_shift_c_value_l229_229447

theorem quadratic_root_shift_c_value
  (r s : ℝ)
  (h1 : r + s = 2)
  (h2 : r * s = -5) :
  ∃ b : ℝ, x^2 + b * x - 2 = 0 :=
by
  sorry

end quadratic_root_shift_c_value_l229_229447


namespace nine_wolves_nine_sheep_seven_days_l229_229510

theorem nine_wolves_nine_sheep_seven_days
    (wolves_sheep_seven_days : ∀ {n : ℕ}, 7 * n / 7 = n) :
    9 * 9 / 9 = 7 := by
  sorry

end nine_wolves_nine_sheep_seven_days_l229_229510


namespace a4_is_5_l229_229218

-- Definitions based on the given conditions in the problem
def sum_arith_seq (n a1 d : ℤ) : ℤ := n * a1 + (n * (n-1)) / 2 * d

def S6 : ℤ := 24
def S9 : ℤ := 63

-- The proof problem: we need to prove that a4 = 5 given the conditions
theorem a4_is_5 (a1 d : ℤ) (h_S6 : sum_arith_seq 6 a1 d = S6) (h_S9 : sum_arith_seq 9 a1 d = S9) : 
  a1 + 3 * d = 5 :=
sorry

end a4_is_5_l229_229218


namespace jina_has_1_koala_bear_l229_229477

theorem jina_has_1_koala_bear:
  let teddies := 5
  let bunnies := 3 * teddies
  let additional_teddies := 2 * bunnies
  let total_teddies := teddies + additional_teddies
  let total_bunnies_and_teddies := total_teddies + bunnies
  let total_mascots := 51
  let koala_bears := total_mascots - total_bunnies_and_teddies
  koala_bears = 1 :=
by
  sorry

end jina_has_1_koala_bear_l229_229477


namespace line_passes_through_circle_center_l229_229920

theorem line_passes_through_circle_center (a : ℝ) : 
  ∀ x y : ℝ, (x, y) = (a, 2*a) → (x - a)^2 + (y - 2*a)^2 = 1 → 2*x - y = 0 :=
by
  sorry

end line_passes_through_circle_center_l229_229920


namespace problem_y_eq_l229_229345

theorem problem_y_eq (y : ℝ) (h : y^3 - 3*y = 9) : y^5 - 10*y^2 = -y^2 + 9*y + 27 := by
  sorry

end problem_y_eq_l229_229345


namespace arithmetic_expression_evaluation_l229_229235

theorem arithmetic_expression_evaluation : 
  -6 * 3 - (-8 * -2) + (-7 * -5) - 10 = -9 := 
by
  sorry

end arithmetic_expression_evaluation_l229_229235


namespace frustum_midsection_area_l229_229302

theorem frustum_midsection_area (r1 r2 : ℝ) (h1 : r1 = 2) (h2 : r2 = 3) :
  let r_mid := (r1 + r2) / 2
  let area_mid := Real.pi * r_mid^2
  area_mid = 25 * Real.pi / 4 := by
  sorry

end frustum_midsection_area_l229_229302


namespace volume_less_than_1000_l229_229458

noncomputable def volume (x : ℕ) : ℤ :=
(x + 3) * (x - 1) * (x^3 - 20)

theorem volume_less_than_1000 : ∃ (n : ℕ), n = 2 ∧ 
  ∃ x1 x2, x1 ≠ x2 ∧ 0 < x1 ∧ 
  0 < x2 ∧
  volume x1 < 1000 ∧
  volume x2 < 1000 ∧
  ∀ x, 0 < x → volume x < 1000 → (x = x1 ∨ x = x2) :=
by
  sorry

end volume_less_than_1000_l229_229458


namespace part_I_part_II_l229_229129

def f (x : ℝ) : ℝ := x ^ 3 - 3 * x

theorem part_I (m : ℝ) (h : ∀ x : ℝ, f x ≠ m) : m < -2 ∨ m > 2 :=
sorry

theorem part_II (P : ℝ × ℝ) (hP : P = (2, -6)) :
  (∃ m b : ℝ, ∀ x : ℝ, (m * x + b = 0 ∧ m = 3 ∧ b = 0) ∨ 
                 (m * x + b = 24 * x - 54 ∧ P.2 = 24 * P.1 - 54)) :=
sorry

end part_I_part_II_l229_229129


namespace equidistant_point_x_coord_l229_229649

theorem equidistant_point_x_coord :
  ∃ x y : ℝ, y = x ∧ dist (x, y) (x, 0) = dist (x, y) (0, y) ∧ dist (x, y) (0, y) = dist (x, y) (x, 5 - x)
    → x = 5 / 2 :=
by sorry

end equidistant_point_x_coord_l229_229649


namespace Randy_initial_money_l229_229010

theorem Randy_initial_money (M : ℝ) (r1 : M + 200 - 1200 = 2000) : M = 3000 :=
by
  sorry

end Randy_initial_money_l229_229010


namespace rubles_exchange_l229_229774

theorem rubles_exchange (x : ℕ) : 
  (3000 * x - 7000 = 2950 * x) → x = 140 := by
  sorry

end rubles_exchange_l229_229774


namespace prove_avg_mark_of_batch3_l229_229054

noncomputable def avg_mark_of_batch3 (A1 A2 A3 : ℕ) (Marks1 Marks2 Marks3 : ℚ) : Prop :=
  A1 = 40 ∧ A2 = 50 ∧ A3 = 60 ∧ Marks1 = 45 ∧ Marks2 = 55 ∧ 
  (A1 * Marks1 + A2 * Marks2 + A3 * Marks3) / (A1 + A2 + A3) = 56.333333333333336 → 
  Marks3 = 65

theorem prove_avg_mark_of_batch3 : avg_mark_of_batch3 40 50 60 45 55 65 :=
by
  unfold avg_mark_of_batch3
  sorry

end prove_avg_mark_of_batch3_l229_229054


namespace ratio_to_percent_l229_229227

theorem ratio_to_percent (a b : ℕ) (h : a = 6) (h2 : b = 3) :
  ((a / b : ℚ) * 100 = 200) :=
by
  have h3 : a = 6 := h
  have h4 : b = 3 := h2
  sorry

end ratio_to_percent_l229_229227


namespace simplify_expression_solve_fractional_eq_l229_229350

-- Problem 1
theorem simplify_expression (x : ℝ) :
  (12 * x^4 + 6 * x^2) / (3 * x) - (-2 * x)^2 * (x + 1) = 2 * x - 4 * x^2 :=
by {
  sorry
}

-- Problem 2
theorem solve_fractional_eq (x : ℝ) (h : x ≠ 0) (h' : x ≠ 1) (h'' : x ≠ -1) :
  (5 / (x^2 + x)) - (1 / (x^2 - x)) = 0 ↔ x = 3 / 2 :=
by {
  sorry
}

end simplify_expression_solve_fractional_eq_l229_229350


namespace todd_initial_money_l229_229591

-- Definitions of the conditions
def cost_per_candy_bar : ℕ := 2
def number_of_candy_bars : ℕ := 4
def money_left : ℕ := 12
def total_money_spent := number_of_candy_bars * cost_per_candy_bar

-- The statement proving the initial amount of money Todd had
theorem todd_initial_money : 
  (total_money_spent + money_left) = 20 :=
by
  sorry

end todd_initial_money_l229_229591


namespace find_m_l229_229491

noncomputable def f (x m : ℝ) : ℝ := x ^ 2 + m
noncomputable def g (x : ℝ) : ℝ := 6 * Real.log x - 4 * x

theorem find_m (m : ℝ) : 
  ∃ a b : ℝ, (0 < a) ∧ (f a m = b) ∧ (g a = b) ∧ (2 * a = (6 / a) - 4) → m = -5 := 
by
  sorry

end find_m_l229_229491


namespace percentage_greater_than_88_l229_229972

theorem percentage_greater_than_88 (x : ℝ) (percentage : ℝ) (h1 : x = 110) (h2 : x = 88 + (percentage * 88)) : percentage = 0.25 :=
by
  sorry

end percentage_greater_than_88_l229_229972


namespace find_x_value_l229_229124

theorem find_x_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 5 * x^2 + 15 * x * y = x^3 + 2 * x^2 * y + 3 * x * y^2) : x = 5 :=
sorry

end find_x_value_l229_229124


namespace cos_double_angle_l229_229993

theorem cos_double_angle (α : ℝ) (h : Real.sin (π + α) = 2 / 3) : Real.cos (2 * α) = 1 / 9 := 
by
  sorry

end cos_double_angle_l229_229993


namespace inequality_and_equality_conditions_l229_229923

theorem inequality_and_equality_conditions
  (x y a b : ℝ)
  (h1 : a + b = 1)
  (h2 : a ≥ 0)
  (h3 : b ≥ 0) :
  (a * x + b * y)^2 ≤ a * x^2 + b * y^2 ∧ ((a * b = 0) ∨ (x = y)) :=
by
  sorry

end inequality_and_equality_conditions_l229_229923


namespace ajay_total_gain_l229_229910

theorem ajay_total_gain:
  let dal_A_kg := 15
  let dal_B_kg := 10
  let dal_C_kg := 12
  let dal_D_kg := 8
  let rate_A := 14.50
  let rate_B := 13
  let rate_C := 16
  let rate_D := 18
  let selling_rate := 17.50
  let cost_A := dal_A_kg * rate_A
  let cost_B := dal_B_kg * rate_B
  let cost_C := dal_C_kg * rate_C
  let cost_D := dal_D_kg * rate_D
  let total_cost := cost_A + cost_B + cost_C + cost_D
  let total_weight := dal_A_kg + dal_B_kg + dal_C_kg + dal_D_kg
  let total_selling_price := total_weight * selling_rate
  let gain := total_selling_price - total_cost
  gain = 104 := by
    sorry

end ajay_total_gain_l229_229910


namespace find_k_l229_229813

theorem find_k (σ μ : ℝ) (hσ : σ = 2) (hμ : μ = 55) :
  ∃ k : ℝ, μ - k * σ > 48 ∧ k = 3 :=
by
  sorry

end find_k_l229_229813


namespace smartphone_price_l229_229192

/-
Question: What is the sticker price of the smartphone, given the following conditions?
Conditions:
1: Store A offers a 20% discount on the sticker price, followed by a $120 rebate. Prices include an 8% sales tax applied after all discounts and fees.
2: Store B offers a 30% discount on the sticker price but adds a $50 handling fee. Prices include an 8% sales tax applied after all discounts and fees.
3: Natalie saves $27 by purchasing the smartphone at store A instead of store B.

Proof Problem:
Prove that given the above conditions, the sticker price of the smartphone is $1450.
-/

theorem smartphone_price (p : ℝ) :
  (1.08 * (0.7 * p + 50) - 1.08 * (0.8 * p - 120)) = 27 ->
  p = 1450 :=
by
  sorry

end smartphone_price_l229_229192


namespace Rebecca_tent_stakes_l229_229654

theorem Rebecca_tent_stakes : 
  ∃ T D W : ℕ, 
    D = 3 * T ∧ 
    W = T + 2 ∧ 
    T + D + W = 22 ∧ 
    T = 4 := 
by
  sorry

end Rebecca_tent_stakes_l229_229654


namespace tom_made_washing_cars_l229_229431

-- Definitions of the conditions
def initial_amount : ℕ := 74
def final_amount : ℕ := 86

-- Statement to be proved
theorem tom_made_washing_cars : final_amount - initial_amount = 12 := by
  sorry

end tom_made_washing_cars_l229_229431


namespace simplify_exponents_l229_229270

theorem simplify_exponents (x : ℝ) : x^5 * x^3 = x^8 :=
by
  sorry

end simplify_exponents_l229_229270


namespace arithmetic_sequence_length_l229_229224

theorem arithmetic_sequence_length :
  ∃ n : ℕ, ∀ (a d l : ℕ), a = 2 → d = 5 → l = 3007 → l = a + (n-1) * d → n = 602 :=
by
  sorry

end arithmetic_sequence_length_l229_229224


namespace total_order_cost_is_correct_l229_229312

noncomputable def totalOrderCost : ℝ :=
  let costGeography := 35 * 10.5
  let costEnglish := 35 * 7.5
  let costMath := 20 * 12.0
  let costScience := 30 * 9.5
  let costHistory := 25 * 11.25
  let costArt := 15 * 6.75
  let discount c := c * 0.10
  let netGeography := if 35 >= 30 then costGeography - discount costGeography else costGeography
  let netEnglish := if 35 >= 30 then costEnglish - discount costEnglish else costEnglish
  let netScience := if 30 >= 30 then costScience - discount costScience else costScience
  let netMath := costMath
  let netHistory := costHistory
  let netArt := costArt
  netGeography + netEnglish + netMath + netScience + netHistory + netArt

theorem total_order_cost_is_correct : totalOrderCost = 1446.00 := by
  sorry

end total_order_cost_is_correct_l229_229312


namespace tessa_owes_30_l229_229617

-- Definitions based on given conditions
def initial_debt : ℕ := 40
def paid_back : ℕ := initial_debt / 2
def remaining_debt_after_payment : ℕ := initial_debt - paid_back
def additional_borrowing : ℕ := 10
def total_debt : ℕ := remaining_debt_after_payment + additional_borrowing

-- Theorem to be proved
theorem tessa_owes_30 : total_debt = 30 :=
by
  sorry

end tessa_owes_30_l229_229617


namespace money_equation_l229_229931

variables (a b: ℝ)

theorem money_equation (h1: 8 * a + b > 160) (h2: 4 * a + b = 120) : a > 10 ∧ ∀ (a1 a2 : ℝ), a1 > a2 → b = 120 - 4 * a → b = 120 - 4 * a1 ∧ 120 - 4 * a1 < 120 - 4 * a2 :=
by 
  sorry

end money_equation_l229_229931


namespace correct_total_annual_cost_l229_229274

def cost_after_coverage (cost: ℕ) (coverage: ℕ) : ℕ :=
  cost - (cost * coverage / 100)

def epiPen_costs : ℕ :=
  (cost_after_coverage 500 75) +
  (cost_after_coverage 550 60) +
  (cost_after_coverage 480 70) +
  (cost_after_coverage 520 65)

def monthly_medical_expenses : ℕ :=
  (cost_after_coverage 250 80) +
  (cost_after_coverage 180 70) +
  (cost_after_coverage 300 75) +
  (cost_after_coverage 350 60) +
  (cost_after_coverage 200 70) +
  (cost_after_coverage 400 80) +
  (cost_after_coverage 150 90) +
  (cost_after_coverage 100 100) +
  (cost_after_coverage 300 60) +
  (cost_after_coverage 350 90) +
  (cost_after_coverage 450 85) +
  (cost_after_coverage 500 65)

def total_annual_cost : ℕ :=
  epiPen_costs + monthly_medical_expenses

theorem correct_total_annual_cost :
  total_annual_cost = 1542 :=
  by sorry

end correct_total_annual_cost_l229_229274


namespace max_intersections_intersections_ge_n_special_case_l229_229518

variable {n m : ℕ}

-- Conditions: n points on a circumference, m and n are positive integers, relatively prime, 6 ≤ 2m < n
def valid_conditions (n m : ℕ) : Prop := Nat.gcd m n = 1 ∧ 6 ≤ 2 * m ∧ 2 * m < n

-- Maximum intersections I = (m-1)n
theorem max_intersections (h : valid_conditions n m) : ∃ I, I = (m - 1) * n :=
by
  sorry

-- Prove I ≥ n
theorem intersections_ge_n (h : valid_conditions n m) : ∃ I, I ≥ n :=
by
  sorry

-- Special case: m = 3 and n is even
theorem special_case (h : valid_conditions n 3) (hn : Even n) : ∃ I, I = n :=
by
  sorry

end max_intersections_intersections_ge_n_special_case_l229_229518


namespace ducks_in_marsh_l229_229694

theorem ducks_in_marsh (total_birds geese : ℕ) (h1 : total_birds = 95) (h2 : geese = 58) : total_birds - geese = 37 := by
  sorry

end ducks_in_marsh_l229_229694


namespace martha_initial_marbles_l229_229100

-- Definition of the conditions
def initial_marbles_dilan : ℕ := 14
def initial_marbles_phillip : ℕ := 19
def initial_marbles_veronica : ℕ := 7
def marbles_after_redistribution_each : ℕ := 15
def number_of_people : ℕ := 4

-- Total marbles after redistribution
def total_marbles_after_redistribution : ℕ := marbles_after_redistribution_each * number_of_people

-- Total initial marbles of Dilan, Phillip, and Veronica
def total_initial_marbles_dilan_phillip_veronica : ℕ := initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica

-- Prove the number of marbles Martha initially had
theorem martha_initial_marbles : initial_marbles_dilan + initial_marbles_phillip + initial_marbles_veronica + x = number_of_people * marbles_after_redistribution →
  x = 20 := by
  sorry

end martha_initial_marbles_l229_229100


namespace beef_not_used_l229_229921

-- Define the context and necessary variables
variable (totalBeef : ℕ) (usedVegetables : ℕ)
variable (beefUsed : ℕ)

-- The conditions given in the problem
def initial_beef : Prop := totalBeef = 4
def used_vegetables : Prop := usedVegetables = 6
def relation_vegetables_beef : Prop := usedVegetables = 2 * beefUsed

-- The statement we need to prove
theorem beef_not_used
  (h1 : initial_beef totalBeef)
  (h2 : used_vegetables usedVegetables)
  (h3 : relation_vegetables_beef usedVegetables beefUsed) :
  (totalBeef - beefUsed) = 1 := by
  sorry

end beef_not_used_l229_229921


namespace magic_square_S_divisible_by_3_l229_229961

-- Definitions of the 3x3 magic square conditions
def is_magic_square (a : ℕ → ℕ → ℤ) (S : ℤ) : Prop :=
  (a 0 0 + a 0 1 + a 0 2 = S) ∧
  (a 1 0 + a 1 1 + a 1 2 = S) ∧
  (a 2 0 + a 2 1 + a 2 2 = S) ∧
  (a 0 0 + a 1 0 + a 2 0 = S) ∧
  (a 0 1 + a 1 1 + a 2 1 = S) ∧
  (a 0 2 + a 1 2 + a 2 2 = S) ∧
  (a 0 0 + a 1 1 + a 2 2 = S) ∧
  (a 0 2 + a 1 1 + a 2 0 = S)

-- Main theorem statement
theorem magic_square_S_divisible_by_3 :
  ∀ (a : ℕ → ℕ → ℤ) (S : ℤ),
    is_magic_square a S →
    S % 3 = 0 :=
by
  -- Here we assume the existence of the proof
  sorry

end magic_square_S_divisible_by_3_l229_229961


namespace kelsey_total_distance_l229_229677

-- Define the constants and variables involved
def total_distance (total_time : ℕ) (speed1 speed2 half_dist1 half_dist2 : ℕ) : ℕ :=
  let T1 := half_dist1 / speed1
  let T2 := half_dist2 / speed2
  let T := T1 + T2
  total_time

-- Prove the equivalency given the conditions
theorem kelsey_total_distance (total_time : ℕ) (speed1 speed2 : ℕ) : 
  (total_time = 10) ∧ (speed1 = 25) ∧ (speed2 = 40)  →
  ∃ D, D = 307 ∧ (10 = D / 50 + D / 80) :=
by 
  intro h
  have h_total_time := h.1
  have h_speed1 := h.2.1
  have h_speed2 := h.2.2
  -- Need to prove the statement using provided conditions
  let D := 307
  sorry

end kelsey_total_distance_l229_229677


namespace goldie_earnings_l229_229011

theorem goldie_earnings
  (hourly_wage : ℕ := 5)
  (hours_last_week : ℕ := 20)
  (hours_this_week : ℕ := 30) :
  hourly_wage * hours_last_week + hourly_wage * hours_this_week = 250 :=
by
  sorry

end goldie_earnings_l229_229011


namespace five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l229_229427

theorem five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand :
  5.8 / 0.001 = 5.8 * 1000 :=
by
  -- This is where the proof would go
  sorry

end five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l229_229427


namespace problem_l229_229579

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ :=
if h : (-1 : ℝ) ≤ x ∧ x < 0 then a*x + 1
else if h : (0 : ℝ) ≤ x ∧ x ≤ 1 then (b*x + 2) / (x + 1)
else 0 -- This should not matter as we only care about the given ranges

theorem problem (a b : ℝ) (h₁ : f 0.5 a b = f 1.5 a b) : a + 3 * b = -10 :=
by
  -- We'll derive equations from given conditions and prove the result.
  sorry

end problem_l229_229579


namespace polynomial_equation_example_l229_229937

theorem polynomial_equation_example (a0 a1 a2 a3 a4 a5 a6 a7 a8 : ℤ)
  (h : x^5 * (x + 3)^3 = a8 * (x + 1)^8 + a7 * (x + 1)^7 + a6 * (x + 1)^6 + a5 * (x + 1)^5 + a4 * (x + 1)^4 + a3 * (x + 1)^3 + a2 * (x + 1)^2 + a1 * (x + 1) + a0) :
  7 * a7 + 5 * a5 + 3 * a3 + a1 = -8 :=
sorry

end polynomial_equation_example_l229_229937


namespace line_parallel_plane_l229_229416

axiom line (m : Type) : Prop
axiom plane (α : Type) : Prop
axiom has_no_common_points (m : Type) (α : Type) : Prop
axiom parallel (m : Type) (α : Type) : Prop

theorem line_parallel_plane
  (m : Type) (α : Type)
  (h : has_no_common_points m α) : parallel m α := sorry

end line_parallel_plane_l229_229416


namespace problem_statement_l229_229523

theorem problem_statement 
  (w x y z : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z = 945) :
  2 * w + 3 * x + 5 * y + 7 * z = 21 :=
by
  sorry

end problem_statement_l229_229523


namespace common_tangent_lines_l229_229449

theorem common_tangent_lines (m : ℝ) (hm : 0 < m) :
  (∀ x y : ℝ, x^2 + y^2 - (4 * m + 2) * x - 2 * m * y + 4 * m^2 + 4 * m + 1 = 0 →
     (y = 0 ∨ y = 4 / 3 * x - 4 / 3)) :=
by sorry

end common_tangent_lines_l229_229449


namespace mean_of_combined_set_is_52_over_3_l229_229830

noncomputable def mean_combined_set : ℚ := 
  let mean_set1 := 10
  let size_set1 := 4
  let mean_set2 := 21
  let size_set2 := 8
  let sum_set1 := mean_set1 * size_set1
  let sum_set2 := mean_set2 * size_set2
  let total_sum := sum_set1 + sum_set2
  let combined_size := size_set1 + size_set2
  let combined_mean := total_sum / combined_size
  combined_mean

theorem mean_of_combined_set_is_52_over_3 :
  mean_combined_set = 52 / 3 :=
by
  sorry

end mean_of_combined_set_is_52_over_3_l229_229830


namespace find_difference_l229_229082

theorem find_difference (m n : ℕ) (hm : ∃ x, m = 111 * x) (hn : ∃ y, n = 31 * y) (h_sum : m + n = 2017) :
  n - m = 463 :=
sorry

end find_difference_l229_229082


namespace inequality_proof_l229_229433

theorem inequality_proof
  (a b c d : ℝ)
  (ha : abs a > 1)
  (hb : abs b > 1)
  (hc : abs c > 1)
  (hd : abs d > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end inequality_proof_l229_229433


namespace units_digit_div_product_l229_229720

theorem units_digit_div_product :
  (30 * 31 * 32 * 33 * 34 * 35) / 14000 % 10 = 2 :=
by
  sorry

end units_digit_div_product_l229_229720


namespace weight_difference_l229_229038

theorem weight_difference :
  let Box_A := 2.4
  let Box_B := 5.3
  let Box_C := 13.7
  let Box_D := 7.1
  let Box_E := 10.2
  let Box_F := 3.6
  let Box_G := 9.5
  max Box_A (max Box_B (max Box_C (max Box_D (max Box_E (max Box_F Box_G))))) -
  min Box_A (min Box_B (min Box_C (min Box_D (min Box_E (min Box_F Box_G))))) = 11.3 :=
by
  sorry

end weight_difference_l229_229038


namespace calculate_speed_l229_229832

variable (time : ℝ) (distance : ℝ)

theorem calculate_speed (h_time : time = 5) (h_distance : distance = 500) : 
  distance / time = 100 := 
by 
  sorry

end calculate_speed_l229_229832


namespace worker_saves_one_third_l229_229717

variable {P : ℝ} 
variable {f : ℝ}

theorem worker_saves_one_third (h : P ≠ 0) (h_eq : 12 * f * P = 6 * (1 - f) * P) : 
  f = 1 / 3 :=
sorry

end worker_saves_one_third_l229_229717


namespace tan_alpha_eq_one_l229_229214

open Real

theorem tan_alpha_eq_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 := 
by
  sorry

end tan_alpha_eq_one_l229_229214


namespace cost_per_millisecond_l229_229749

theorem cost_per_millisecond
  (C : ℝ)
  (h1 : 1.07 + (C * 1500) + 5.35 = 40.92) :
  C = 0.023 :=
sorry

end cost_per_millisecond_l229_229749


namespace find_G_14_l229_229126

noncomputable def G (x : ℝ) : ℝ := sorry

lemma G_at_7 : G 7 = 20 := sorry

lemma functional_equation (x : ℝ) (hx: x ^ 2 + 8 * x + 16 ≠ 0) : 
  G (4 * x) / G (x + 4) = 16 - (96 * x + 128) / (x^2 + 8 * x + 16) := sorry

theorem find_G_14 : G 14 = 96 := sorry

end find_G_14_l229_229126


namespace christopher_avg_speed_l229_229550

-- Definition of a palindrome (not required for this proof, but helpful for context)
def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

-- Given conditions
def initial_reading : ℕ := 12321
def final_reading : ℕ := 12421
def duration : ℕ := 4

-- Definition of average speed calculation
def average_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- Main theorem to prove
theorem christopher_avg_speed : average_speed (final_reading - initial_reading) duration = 25 :=
by
  sorry

end christopher_avg_speed_l229_229550


namespace marciaHairLengthProof_l229_229846

noncomputable def marciaHairLengthAtEndOfSchoolYear : Float :=
  let L0 := 24.0                           -- initial length
  let L1 := L0 - 0.3 * L0                  -- length after September cut
  let L2 := L1 + 3.0 * 1.5                 -- length after three months of growth (Sept - Dec)
  let L3 := L2 - 0.2 * L2                  -- length after January cut
  let L4 := L3 + 5.0 * 1.8                 -- length after five months of growth (Jan - May)
  let L5 := L4 - 4.0                       -- length after June cut
  L5

theorem marciaHairLengthProof : marciaHairLengthAtEndOfSchoolYear = 22.04 :=
by
  sorry

end marciaHairLengthProof_l229_229846


namespace rahul_deepak_present_ages_l229_229818

theorem rahul_deepak_present_ages (R D : ℕ) 
  (h1 : R / D = 4 / 3)
  (h2 : R + 6 = 26)
  (h3 : D + 6 = 1/2 * (R + (R + 6)))
  (h4 : (R + 11) + (D + 11) = 59) 
  : R = 20 ∧ D = 17 :=
sorry

end rahul_deepak_present_ages_l229_229818


namespace quadratic_point_inequality_l229_229912

theorem quadratic_point_inequality 
  (m y1 y2 : ℝ)
  (hA : y1 = (m - 1)^2)
  (hB : y2 = (m + 1 - 1)^2)
  (hy1_lt_y2 : y1 < y2) :
  m > 1 / 2 :=
by 
  sorry

end quadratic_point_inequality_l229_229912


namespace number_of_players_l229_229630

-- Definitions of the conditions
def initial_bottles : ℕ := 4 * 12
def bottles_remaining : ℕ := 15
def bottles_taken_per_player : ℕ := 2 + 1

-- Total number of bottles taken
def bottles_taken := initial_bottles - bottles_remaining

-- The main theorem stating that the number of players is 11.
theorem number_of_players : (bottles_taken / bottles_taken_per_player) = 11 :=
by
  sorry

end number_of_players_l229_229630
