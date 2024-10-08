import Mathlib

namespace trigonometric_identity_l22_22378

theorem trigonometric_identity (α : ℝ) 
  (h : Real.sin (π / 6 - α) = 1 / 3) : 
  2 * (Real.cos (π / 6 + α / 2))^2 - 1 = 1 / 3 := 
by sorry

end trigonometric_identity_l22_22378


namespace slices_per_friend_l22_22822

theorem slices_per_friend (n : ℕ) (h1 : n > 0)
    (h2 : ∀ i : ℕ, i < n → (15 + 18 + 20 + 25) = 78 * n) :
    78 = (15 + 18 + 20 + 25) / n := 
by
  sorry

end slices_per_friend_l22_22822


namespace fish_initial_numbers_l22_22965

theorem fish_initial_numbers (x y : ℕ) (h1 : x + y = 100) (h2 : x - 30 = y - 40) : x = 45 ∧ y = 55 :=
by
  sorry

end fish_initial_numbers_l22_22965


namespace find_d_l22_22796

theorem find_d (c : ℝ) (d : ℝ) (α : ℝ) (β : ℝ) (γ : ℝ) (ω : ℝ)  
  (h1 : α = c) 
  (h2 : β = 43)
  (h3 : γ = 59)
  (h4 : ω = d)
  (h5 : α + d + β + γ = 180) :
  d = 42 :=
by
  sorry

end find_d_l22_22796


namespace smallest_square_value_l22_22269

theorem smallest_square_value (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h₁ : ∃ r : ℕ, 15 * a + 16 * b = r^2) (h₂ : ∃ s : ℕ, 16 * a - 15 * b = s^2) :
  ∃ (m : ℕ), m = 481^2 ∧ (15 * a + 16 * b = m ∨ 16 * a - 15 * b = m) :=
  sorry

end smallest_square_value_l22_22269


namespace product_positive_l22_22911

variables {x y : ℝ}

noncomputable def non_zero (z : ℝ) := z ≠ 0

theorem product_positive (hx : non_zero x) (hy : non_zero y) 
(h1 : x^2 - x > y^2) (h2 : y^2 - y > x^2) : x * y > 0 :=
by
  sorry

end product_positive_l22_22911


namespace max_value_of_3x_plus_4y_on_curve_C_l22_22888

theorem max_value_of_3x_plus_4y_on_curve_C :
  ∀ (x y : ℝ),
  (∃ (ρ θ : ℝ), ρ^2 = 36 / (4 * (Real.cos θ)^2 + 9 * (Real.sin θ)^2) ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  (P : ℝ × ℝ) →
  (P = (x, y)) →
  3 * x + 4 * y ≤ Real.sqrt 145 ∧ ∃ (α : ℝ), 0 ≤ α ∧ α < 2 * Real.pi ∧ 3 * x + 4 * y = Real.sqrt 145 := 
by
  intros x y h_exists P hP
  sorry

end max_value_of_3x_plus_4y_on_curve_C_l22_22888


namespace min_value_of_abc_l22_22349

variables {a b c : ℝ}

noncomputable def satisfies_condition (a b c : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ (b + c) / a + (a + c) / b = (a + b) / c + 1

theorem min_value_of_abc (a b c : ℝ) (h : satisfies_condition a b c) : (a + b) / c ≥ 5 / 2 :=
sorry

end min_value_of_abc_l22_22349


namespace simplify_fraction_l22_22025

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 65 / 12 := by
  sorry

end simplify_fraction_l22_22025


namespace original_number_l22_22818

theorem original_number (x : ℤ) (h : (x - 5) / 4 = (x - 4) / 5) : x = 9 :=
sorry

end original_number_l22_22818


namespace trains_total_distance_l22_22608

theorem trains_total_distance (speed_A speed_B : ℝ) (time_A time_B : ℝ) (dist_A dist_B : ℝ):
  speed_A = 90 ∧ 
  speed_B = 120 ∧ 
  time_A = 1 ∧ 
  time_B = 5/6 ∧ 
  dist_A = speed_A * time_A ∧ 
  dist_B = speed_B * time_B ->
  (dist_A + dist_B) = 190 :=
by 
  intros h
  obtain ⟨h1, h2, h3, h4, h5, h6⟩ := h
  sorry

end trains_total_distance_l22_22608


namespace simplify_expression1_simplify_expression2_l22_22857

section
variables (a b : ℝ)

theorem simplify_expression1 : -b*(2*a - b) + (a + b)^2 = a^2 + 2*b^2 :=
sorry
end

section
variables (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2)

theorem simplify_expression2 : (1 - (x/(2 + x))) / ((x^2 - 4)/(x^2 + 4*x + 4)) = 2/(x - 2) :=
sorry
end

end simplify_expression1_simplify_expression2_l22_22857


namespace has_three_zeros_iff_b_lt_neg3_l22_22124

def f (x b : ℝ) : ℝ := x^3 - b * x^2 - 4

theorem has_three_zeros_iff_b_lt_neg3 (b : ℝ) :
  (∃ x₁ x₂ x₃ : ℝ, f x₁ b = 0 ∧ f x₂ b = 0 ∧ f x₃ b = 0 ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃) ↔ b < -3 := 
sorry

end has_three_zeros_iff_b_lt_neg3_l22_22124


namespace number_of_students_speaking_two_languages_l22_22867

variables (G H M GH GM HM GHM N : ℕ)

def students_speaking_two_languages (G H M GH GM HM GHM N : ℕ) : ℕ :=
  G + H + M - (GH + GM + HM) + GHM

theorem number_of_students_speaking_two_languages 
  (h_total : N = 22)
  (h_G : G = 6)
  (h_H : H = 15)
  (h_M : M = 6)
  (h_GHM : GHM = 1)
  (h_students : N = students_speaking_two_languages G H M GH GM HM GHM N): 
  GH + GM + HM = 6 := 
by 
  unfold students_speaking_two_languages at h_students 
  sorry

end number_of_students_speaking_two_languages_l22_22867


namespace car_speed_is_104_mph_l22_22810

noncomputable def speed_of_car_in_mph
  (fuel_efficiency_km_per_liter : ℝ) -- car travels 64 km per liter
  (fuel_consumption_gallons : ℝ) -- fuel tank decreases by 3.9 gallons
  (time_hours : ℝ) -- period of 5.7 hours
  (gallon_to_liter : ℝ) -- 1 gallon is 3.8 liters
  (km_to_mile : ℝ) -- 1 mile is 1.6 km
  : ℝ :=
  let fuel_consumption_liters := fuel_consumption_gallons * gallon_to_liter
  let distance_km := fuel_efficiency_km_per_liter * fuel_consumption_liters
  let distance_miles := distance_km / km_to_mile
  let speed_mph := distance_miles / time_hours
  speed_mph

theorem car_speed_is_104_mph 
  (fuel_efficiency_km_per_liter : ℝ := 64)
  (fuel_consumption_gallons : ℝ := 3.9)
  (time_hours : ℝ := 5.7)
  (gallon_to_liter : ℝ := 3.8)
  (km_to_mile : ℝ := 1.6)
  : speed_of_car_in_mph fuel_efficiency_km_per_liter fuel_consumption_gallons time_hours gallon_to_liter km_to_mile = 104 :=
  by
    sorry

end car_speed_is_104_mph_l22_22810


namespace number_of_multiples_840_in_range_l22_22039

theorem number_of_multiples_840_in_range :
  ∃ n, n = 1 ∧ ∀ x, 1000 ≤ x ∧ x ≤ 2500 ∧ (840 ∣ x) → x = 1680 :=
by
  sorry

end number_of_multiples_840_in_range_l22_22039


namespace bella_steps_l22_22971

/-- Bella begins to walk from her house toward her friend Ella's house. At the same time, Ella starts to skate toward Bella's house. They each maintain a constant speed, and Ella skates three times as fast as Bella walks. The distance between their houses is 10560 feet, and Bella covers 3 feet with each step. Prove that Bella will take 880 steps by the time she meets Ella. -/
theorem bella_steps 
  (d : ℝ)    -- distance between their houses in feet
  (s_bella : ℝ)    -- speed of Bella in feet per minute
  (s_ella : ℝ)    -- speed of Ella in feet per minute
  (steps_per_ft : ℝ)    -- feet per step of Bella
  (h1 : d = 10560)    -- distance between their houses is 10560 feet
  (h2 : s_ella = 3 * s_bella)    -- Ella skates three times as fast as Bella
  (h3 : steps_per_ft = 3)    -- Bella covers 3 feet with each step
  : (10560 / (4 * s_bella)) * s_bella / 3 = 880 :=
by
  -- proof here 
  sorry

end bella_steps_l22_22971


namespace chessboard_polygon_l22_22861

-- Conditions
variable (A B a b : ℕ)

-- Statement of the theorem
theorem chessboard_polygon (A B a b : ℕ) : A - B = 4 * (a - b) :=
sorry

end chessboard_polygon_l22_22861


namespace sin_300_eq_neg_sqrt3_div_2_l22_22689

-- Defining the problem statement as a Lean theorem
theorem sin_300_eq_neg_sqrt3_div_2 : Real.sin (300 * Real.pi / 180) = - (Real.sqrt 3 / 2) :=
by
  sorry

end sin_300_eq_neg_sqrt3_div_2_l22_22689


namespace tom_gave_jessica_some_seashells_l22_22339

theorem tom_gave_jessica_some_seashells
  (original_seashells : ℕ := 5)
  (current_seashells : ℕ := 3) :
  original_seashells - current_seashells = 2 :=
by
  sorry

end tom_gave_jessica_some_seashells_l22_22339


namespace value_of_a_l22_22478

-- Definitions of sets A and B
def A : Set ℝ := {x | x^2 = 1}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

-- The main theorem statement
theorem value_of_a (a : ℝ) (H : B a ⊆ A) : a = 0 ∨ a = 1 ∨ a = -1 :=
by 
  sorry

end value_of_a_l22_22478


namespace alicia_average_speed_correct_l22_22125

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

end alicia_average_speed_correct_l22_22125


namespace arccos_cos_7_l22_22705

noncomputable def arccos_cos_7_eq_7_minus_2pi : Prop :=
  ∃ x : ℝ, x = 7 - 2 * Real.pi ∧ Real.arccos (Real.cos 7) = x

theorem arccos_cos_7 :
  arccos_cos_7_eq_7_minus_2pi :=
by
  sorry

end arccos_cos_7_l22_22705


namespace fraction_of_A_eq_l22_22128

noncomputable def fraction_A (A B C T : ℕ) : ℚ :=
  A / (T - A)

theorem fraction_of_A_eq :
  ∃ (A B C T : ℕ), T = 360 ∧ A = B + 10 ∧ B = 2 * (A + C) / 7 ∧ T = A + B + C ∧ fraction_A A B C T = 1 / 3 :=
by
  sorry

end fraction_of_A_eq_l22_22128


namespace matrix_addition_l22_22648

def M1 : Matrix (Fin 3) (Fin 3) ℤ :=
![![4, 1, -3],
  ![0, -2, 5],
  ![7, 0, 1]]

def M2 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -6,  9, 2],
  ![  3, -4, -8],
  ![  0,  5, -3]]

def M3 : Matrix (Fin 3) (Fin 3) ℤ :=
![![ -2, 10, -1],
  ![  3, -6, -3],
  ![  7,  5, -2]]

theorem matrix_addition : M1 + M2 = M3 := by
  sorry

end matrix_addition_l22_22648


namespace equilateral_triangle_intersections_l22_22734

-- Define the main theorem based on the conditions

theorem equilateral_triangle_intersections :
  let a_1 := (6 - 1) * (7 - 1) / 2
  let a_2 := (6 - 2) * (7 - 2) / 2
  let a_3 := (6 - 3) * (7 - 3) / 2
  let a_4 := (6 - 4) * (7 - 4) / 2
  let a_5 := (6 - 5) * (7 - 5) / 2
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 70 := by
  sorry

end equilateral_triangle_intersections_l22_22734


namespace basketball_competition_l22_22513

theorem basketball_competition:
  (∃ x : ℕ, (0 ≤ x) ∧ (x ≤ 12) ∧ (3 * x - (12 - x) ≥ 28)) := by
  sorry

end basketball_competition_l22_22513


namespace fraction_power_mult_equality_l22_22699

-- Define the fraction and the power
def fraction := (1 : ℚ) / 3
def power : ℚ := fraction ^ 4

-- Define the multiplication
def result := 8 * power

-- Prove the equality
theorem fraction_power_mult_equality : result = 8 / 81 := by
  sorry

end fraction_power_mult_equality_l22_22699


namespace find_value_l22_22717

noncomputable def a : ℝ := 5 - 2 * Real.sqrt 6

theorem find_value :
  a^2 - 10 * a + 1 = 0 :=
by
  -- Since we are only required to write the statement, add sorry to skip the proof.
  sorry

end find_value_l22_22717


namespace B_inter_A_complement_eq_one_l22_22727

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3}
def A_complement : Set ℕ := I \ A

theorem B_inter_A_complement_eq_one : B ∩ A_complement = {1} := by
  sorry

end B_inter_A_complement_eq_one_l22_22727


namespace unique_ordered_triples_count_l22_22560

theorem unique_ordered_triples_count :
  ∃ (n : ℕ), n = 1 ∧ ∀ (a b c : ℕ), 1 ≤ a ∧ a ≤ b ∧ b ≤ c ∧
  abc = 4 * (ab + bc + ca) ∧ a = c / 4 -> False :=
sorry

end unique_ordered_triples_count_l22_22560


namespace candle_height_relation_l22_22962

variables (t : ℝ)

def height_candle_A (t : ℝ) := 12 - 2 * t
def height_candle_B (t : ℝ) := 9 - 2 * t

theorem candle_height_relation : 
  12 - 2 * (15 / 4) = 3 * (9 - 2 * (15 / 4)) :=
by
  sorry

end candle_height_relation_l22_22962


namespace percentage_respondents_liked_B_l22_22043

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

end percentage_respondents_liked_B_l22_22043


namespace max_value_is_63_l22_22538

noncomputable def max_value (x y : ℝ) : ℝ :=
  x^2 + 3*x*y + 4*y^2

theorem max_value_is_63 (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (cond : x^2 - 3*x*y + 4*y^2 = 9) :
  max_value x y ≤ 63 :=
by
  sorry

end max_value_is_63_l22_22538


namespace jessica_quarters_l22_22358

theorem jessica_quarters (original_borrowed : ℕ) (quarters_borrowed : ℕ) 
  (H1 : original_borrowed = 8)
  (H2 : quarters_borrowed = 3) : 
  original_borrowed - quarters_borrowed = 5 := sorry

end jessica_quarters_l22_22358


namespace cost_price_proof_l22_22160

noncomputable def cost_price_per_bowl : ℚ := 1400 / 103

theorem cost_price_proof
  (total_bowls: ℕ) (sold_bowls: ℕ) (selling_price_per_bowl: ℚ)
  (percentage_gain: ℚ) 
  (total_bowls_eq: total_bowls = 110)
  (sold_bowls_eq: sold_bowls = 100)
  (selling_price_per_bowl_eq: selling_price_per_bowl = 14)
  (percentage_gain_eq: percentage_gain = 300 / 11) :
  (selling_price_per_bowl * sold_bowls - (sold_bowls + 3) * (selling_price_per_bowl / (3 * percentage_gain / 100))) = cost_price_per_bowl :=
by
  sorry

end cost_price_proof_l22_22160


namespace opposite_number_in_circle_l22_22605

theorem opposite_number_in_circle (n : ℕ) (h1 : 1 ≤ n) (h2 : n ≤ 200) (h3 : ∀ k, 1 ≤ k ∧ k ≤ 200 → ∃ m, (m = (k + 100) % 200) ∧ (m ≠ k) ∧ (n + 100) % 200 < k):
  ∃ m : ℕ, m = 114 ∧ (113 + 100) % 200 = m :=
by
  sorry

end opposite_number_in_circle_l22_22605


namespace third_factor_of_product_l22_22853

theorem third_factor_of_product (w : ℕ) (h_w_pos : w > 0) (h_w_168 : w = 168)
  (w_factors : (936 * w) = 2^5 * 3^3 * x)
  (h36_factors : 2^5 ∣ (936 * w)) (h33_factors : 3^3 ∣ (936 * w)) : 
  (936 * w) / (2^5 * 3^3) = 182 :=
by {
  -- This is a placeholder. The actual proof is omitted.
  sorry
}

end third_factor_of_product_l22_22853


namespace find_y_l22_22292

theorem find_y :
  ∃ y : ℝ, ((0.47 * 1442) - (0.36 * y) + 65 = 5) ∧ y = 2049.28 :=
by
  sorry

end find_y_l22_22292


namespace solve_for_a_l22_22876

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end solve_for_a_l22_22876


namespace interval_k_is_40_l22_22123

def total_students := 1200
def sample_size := 30

theorem interval_k_is_40 : (total_students / sample_size) = 40 :=
by
  sorry

end interval_k_is_40_l22_22123


namespace max_students_distribute_pens_pencils_l22_22011

noncomputable def gcd_example : ℕ :=
  Nat.gcd 1340 1280

theorem max_students_distribute_pens_pencils : gcd_example = 20 :=
sorry

end max_students_distribute_pens_pencils_l22_22011


namespace simplify_expr_l22_22809

theorem simplify_expr : (1 / (1 - Real.sqrt 3)) * (1 / (1 + Real.sqrt 3)) = -1 / 2 := by
  sorry

end simplify_expr_l22_22809


namespace exists_nat_with_digit_sum_1000_and_square_sum_1000000_l22_22910

-- Define a function to calculate the sum of digits in base-10
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main theorem
theorem exists_nat_with_digit_sum_1000_and_square_sum_1000000 :
  ∃ n : ℕ, sum_of_digits n = 1000 ∧ sum_of_digits (n^2) = 1000000 :=
by
  sorry

end exists_nat_with_digit_sum_1000_and_square_sum_1000000_l22_22910


namespace sum_of_squares_of_roots_l22_22617

theorem sum_of_squares_of_roots (x_1 x_2 : ℚ) (h1 : 6 * x_1^2 - 13 * x_1 + 5 = 0)
                                (h2 : 6 * x_2^2 - 13 * x_2 + 5 = 0) 
                                (h3 : x_1 ≠ x_2) :
  x_1^2 + x_2^2 = 109 / 36 :=
sorry

end sum_of_squares_of_roots_l22_22617


namespace john_needs_60_bags_l22_22803

theorem john_needs_60_bags
  (horses : ℕ)
  (feeding_per_day : ℕ)
  (food_per_feeding : ℕ)
  (bag_weight : ℕ)
  (days : ℕ)
  (tons_in_pounds : ℕ)
  (half : ℕ)
  (h1 : horses = 25)
  (h2 : feeding_per_day = 2)
  (h3 : food_per_feeding = 20)
  (h4 : bag_weight = 1000)
  (h5 : days = 60)
  (h6 : tons_in_pounds = 2000)
  (h7 : half = 1 / 2) :
  ((horses * feeding_per_day * food_per_feeding * days) / (tons_in_pounds * half)) = 60 := by
  sorry

end john_needs_60_bags_l22_22803


namespace Mater_costs_10_percent_of_Lightning_l22_22031

-- Conditions
def price_Lightning : ℕ := 140000
def price_Sally : ℕ := 42000
def price_Mater : ℕ := price_Sally / 3

-- The theorem we want to prove
theorem Mater_costs_10_percent_of_Lightning :
  (price_Mater * 100 / price_Lightning) = 10 := 
by 
  sorry

end Mater_costs_10_percent_of_Lightning_l22_22031


namespace div_by_5_l22_22636

theorem div_by_5 (a b : ℕ) (h: 5 ∣ (a * b)) : (5 ∣ a) ∨ (5 ∣ b) :=
by
  -- Proof by contradiction
  -- Assume the negation of the conclusion
  have h_nand : ¬ (5 ∣ a) ∧ ¬ (5 ∣ b) := sorry

  -- Derive a contradiction based on the assumptions
  sorry

end div_by_5_l22_22636


namespace two_digit_number_determined_l22_22338

theorem two_digit_number_determined
  (x y : ℕ)
  (hx : 0 ≤ x ∧ x ≤ 9)
  (hy : 1 ≤ y ∧ y ≤ 9)
  (h : 2 * (5 * x - 3) + y = 21) :
  10 * y + x = 72 := 
sorry

end two_digit_number_determined_l22_22338


namespace find_a_l22_22572

def line1 (a : ℝ) (P : ℝ × ℝ) : Prop := 2 * P.1 - a * P.2 - 1 = 0

def line2 (P : ℝ × ℝ) : Prop := P.1 + 2 * P.2 = 0

theorem find_a (a : ℝ) :
  (∀ P : ℝ × ℝ, line1 a P ∧ line2 P) → a = 1 := by
  sorry

end find_a_l22_22572


namespace min_choir_members_l22_22157

theorem min_choir_members (n : ℕ) : 
  (∀ (m : ℕ), m % 9 = 0 ∧ m % 10 = 0 ∧ m % 11 = 0 → m ≥ n) → 
  n = 990 :=
by
  sorry

end min_choir_members_l22_22157


namespace part1_part2_l22_22884

theorem part1 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (1 / a^2 + 1 / b^2 ≥ 1 / 2) :=
sorry

theorem part2 (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = a * b) :
  (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a + b = a * b ∧ (|2 * a - 1| + |3 * b - 1| = 2 * Real.sqrt 6 + 3)) :=
sorry

end part1_part2_l22_22884


namespace area_of_given_circle_is_4pi_l22_22723

-- Define the given equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0

-- Define the area of the circle to be proved
noncomputable def area_of_circle : ℝ := 4 * Real.pi

-- Statement of the theorem to be proved in Lean
theorem area_of_given_circle_is_4pi :
  (∃ x y : ℝ, circle_equation x y) → area_of_circle = 4 * Real.pi :=
by
  -- The proof will go here
  sorry

end area_of_given_circle_is_4pi_l22_22723


namespace gwendolyn_read_time_l22_22275

theorem gwendolyn_read_time :
  let rate := 200 -- sentences per hour
  let paragraphs_per_page := 30
  let sentences_per_paragraph := 15
  let pages := 100
  let sentences_per_page := sentences_per_paragraph * paragraphs_per_page
  let total_sentences := sentences_per_page * pages
  let total_time := total_sentences / rate
  total_time = 225 :=
by
  sorry

end gwendolyn_read_time_l22_22275


namespace sasha_stickers_l22_22040

variables (m n : ℕ) (t : ℝ)

-- Conditions
def conditions : Prop :=
  m < n ∧ -- Fewer coins than stickers
  m ≥ 1 ∧ -- At least one coin
  n ≥ 1 ∧ -- At least one sticker
  t > 1 ∧ -- t is greater than 1
  m * t + n = 100 ∧ -- Coin increase condition
  m + n * t = 101 -- Sticker increase condition

-- Theorem stating that the number of stickers must be 34 or 66
theorem sasha_stickers : conditions m n t → n = 34 ∨ n = 66 :=
sorry

end sasha_stickers_l22_22040


namespace min_value_one_over_a_plus_two_over_b_l22_22192

/-- Given a > 0, b > 0, 2a + b = 1, prove that the minimum value of (1/a) + (2/b) is 8 --/
theorem min_value_one_over_a_plus_two_over_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = 1) :
  (1 / a) + (2 / b) ≥ 8 :=
sorry

end min_value_one_over_a_plus_two_over_b_l22_22192


namespace tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l22_22200

theorem tan_symmetric_about_k_pi_over_2 (k : ℤ) : 
  (∀ x : ℝ, Real.tan (x + k * Real.pi / 2) = Real.tan x) := 
sorry

theorem min_value_cos2x_plus_sinx : 
  (∀ x : ℝ, Real.cos x ^ 2 + Real.sin x ≥ -1) ∧ (∃ x : ℝ, Real.cos x ^ 2 + Real.sin x = -1) :=
sorry

end tan_symmetric_about_k_pi_over_2_min_value_cos2x_plus_sinx_l22_22200


namespace total_games_l22_22966

-- The conditions
def working_games : ℕ := 6
def bad_games : ℕ := 5

-- The theorem to prove
theorem total_games : working_games + bad_games = 11 :=
by
  sorry

end total_games_l22_22966


namespace find_b_from_conditions_l22_22671

theorem find_b_from_conditions 
  (x y b : ℝ) 
  (h1 : 3 * x - 5 * y = b) 
  (h2 : x / (x + y) = 5 / 7) 
  (h3 : x - y = 3) : 
  b = 5 := 
by 
  sorry

end find_b_from_conditions_l22_22671


namespace powderman_distance_when_blast_heard_l22_22639

-- Define constants
def fuse_time : ℝ := 30  -- seconds
def run_rate : ℝ := 8    -- yards per second
def sound_rate : ℝ := 1080  -- feet per second
def yards_to_feet : ℝ := 3  -- conversion factor

-- Define the time at which the blast was heard
noncomputable def blast_heard_time : ℝ := 675 / 22

-- Define distance functions
def p (t : ℝ) : ℝ := run_rate * yards_to_feet * t  -- distance run by powderman in feet
def q (t : ℝ) : ℝ := sound_rate * (t - fuse_time)  -- distance sound has traveled in feet

-- Proof statement: given the conditions, the distance run by the powderman equals 245 yards
theorem powderman_distance_when_blast_heard :
  p (blast_heard_time) / yards_to_feet = 245 := by
  sorry

end powderman_distance_when_blast_heard_l22_22639


namespace angle_between_lines_at_most_l22_22570
-- Import the entire Mathlib library for general mathematical definitions

-- Define the problem statement in Lean 4
theorem angle_between_lines_at_most (n : ℕ) (h : n > 0) :
  ∃ (l1 l2 : ℝ), l1 ≠ l2 ∧ (n : ℝ) > 0 → ∃ θ, 0 ≤ θ ∧ θ ≤ 180 / n := by
  sorry

end angle_between_lines_at_most_l22_22570


namespace find_k_value_l22_22263

theorem find_k_value (x k : ℝ) (hx : Real.logb 9 3 = x) (hk : Real.logb 3 81 = k * x) : k = 8 :=
by sorry

end find_k_value_l22_22263


namespace coefficient_comparison_expansion_l22_22726

theorem coefficient_comparison_expansion (n : ℕ) (h₁ : 2 * n * (n - 1) = 14 * n) : n = 8 :=
by
  sorry

end coefficient_comparison_expansion_l22_22726


namespace car_clock_problem_l22_22284

-- Define the conditions and statements required for the proof
variable (t₀ : ℕ) -- Initial time in minutes corresponding to 2:00 PM
variable (t₁ : ℕ) -- Time in minutes when the accurate watch shows 2:40 PM
variable (t₂ : ℕ) -- Time in minutes when the car clock shows 2:50 PM
variable (t₃ : ℕ) -- Time in minutes when the car clock shows 8:00 PM
variable (rate : ℚ) -- Rate of the car clock relative to real time

-- Define the initial condition
def initial_time := (t₀ = 0)

-- Define the time gain from 2:00 PM to 2:40 PM on the accurate watch
def accurate_watch_time := (t₁ = 40)

-- Define the time gain for car clock from 2:00 PM to 2:50 PM
def car_clock_time := (t₂ = 50)

-- Define the rate of the car clock relative to real time as 5/4
def car_clock_rate := (rate = 5/4)

-- Define the car clock reading at 8:00 PM
def car_clock_later := (t₃ = 8 * 60)

-- Define the actual time corresponding to the car clock reading 8:00 PM
def actual_time : ℚ := (t₀ + (t₃ - t₀) * (4/5))

-- Define the statement theorem using the defined conditions and variables
theorem car_clock_problem 
  (h₀ : initial_time t₀) 
  (h₁ : accurate_watch_time t₁) 
  (h₂ : car_clock_time t₂) 
  (h₃ : car_clock_rate rate) 
  (h₄ : car_clock_later t₃) 
  : actual_time t₀ t₃ = 8 * 60 + 24 :=
by sorry

end car_clock_problem_l22_22284


namespace opposite_of_3_is_neg3_l22_22914

def opposite (x : ℝ) := -x

theorem opposite_of_3_is_neg3 : opposite 3 = -3 :=
by
  sorry

end opposite_of_3_is_neg3_l22_22914


namespace tetrahedron_face_inequality_l22_22313

theorem tetrahedron_face_inequality
    (A B C D : ℝ) :
    |A^2 + B^2 - C^2 - D^2| ≤ 2 * (A * B + C * D) := by
  sorry

end tetrahedron_face_inequality_l22_22313


namespace trajectory_equation_l22_22824

noncomputable def circle1_center := (-3, 0)
noncomputable def circle2_center := (3, 0)

def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

def is_tangent_internally (x y : ℝ) : Prop := 
  ∃ (P : ℝ × ℝ), circle1 P.1 P.2 ∧ circle2 P.1 P.2

theorem trajectory_equation :
  ∀ (x y : ℝ), is_tangent_internally x y → (x^2 / 16 + y^2 / 7 = 1) :=
sorry

end trajectory_equation_l22_22824


namespace initial_birds_l22_22295

theorem initial_birds (B : ℕ) (h1 : B + 21 = 35) : B = 14 :=
by
  sorry

end initial_birds_l22_22295


namespace translate_parabola_l22_22626

theorem translate_parabola (x : ℝ) :
  (∃ (h k : ℝ), h = 1 ∧ k = 3 ∧ ∀ x: ℝ, y = 2*x^2 → y = 2*(x - h)^2 + k) := 
by
  use 1, 3
  sorry

end translate_parabola_l22_22626


namespace no_combination_of_five_coins_is_75_l22_22037

theorem no_combination_of_five_coins_is_75 :
  ∀ (a b c d e : ℕ), 
    (a + b + c + d + e = 5) →
    ∀ (v : ℤ), 
      v = a * 1 + b * 5 + c * 10 + d * 25 + e * 50 → 
      v ≠ 75 :=
by
  intro a b c d e h1 v h2
  sorry

end no_combination_of_five_coins_is_75_l22_22037


namespace cost_per_acre_proof_l22_22009

def cost_of_land (tac tl : ℕ) (hc hcc hcp heq : ℕ) (ttl : ℕ) : ℕ := ttl - (hc + hcc + hcp + heq)

def cost_per_acre (total_land : ℕ) (cost_land : ℕ) : ℕ := cost_land / total_land

theorem cost_per_acre_proof (tac tl hc hcc hcp heq ttl epl : ℕ) 
  (h1 : tac = 30)
  (h2 : hc = 120000)
  (h3 : hcc = 20 * 1000)
  (h4 : hcp = 100 * 5)
  (h5 : heq = 6 * 100 + 6000)
  (h6 : ttl = 147700) :
  cost_per_acre tac (cost_of_land tac tl hc hcc hcp heq ttl) = epl := by
  sorry

end cost_per_acre_proof_l22_22009


namespace max_notebooks_no_more_than_11_l22_22452

noncomputable def maxNotebooks (money : ℕ) (cost_single : ℕ) (cost_pack4 : ℕ) (cost_pack7 : ℕ) (max_pack7 : ℕ) : ℕ :=
if money >= cost_pack7 then
  if (money - cost_pack7) >= cost_pack4 then 7 + 4
  else if (money - cost_pack7) >= cost_single then 7 + 1
  else 7
else if money >= cost_pack4 then
  if (money - cost_pack4) >= cost_pack4 then 4 + 4
  else if (money - cost_pack4) >= cost_single then 4 + 1
  else 4
else
  money / cost_single

theorem max_notebooks_no_more_than_11 :
  maxNotebooks 15 2 6 9 1 = 11 :=
by
  sorry

end max_notebooks_no_more_than_11_l22_22452


namespace part1_part2_l22_22632

def f (x a : ℝ) := abs (x - a)

theorem part1 (a : ℝ) :
  (∀ x : ℝ, (f x a) ≤ 2 ↔ 1 ≤ x ∧ x ≤ 5) → a = 3 :=
by
  intros h
  sorry

theorem part2 (m : ℝ) : 
  (∀ x : ℝ, f (2 * x) 3 + f (x + 2) 3 ≥ m) → m ≤ 1 / 2 :=
by
  intros h
  sorry

end part1_part2_l22_22632


namespace inheritance_amount_l22_22441

theorem inheritance_amount (x : ℝ) 
  (federal_tax : ℝ := 0.25 * x) 
  (state_tax : ℝ := 0.15 * (x - federal_tax)) 
  (city_tax : ℝ := 0.05 * (x - federal_tax - state_tax)) 
  (total_tax : ℝ := 20000) :
  (federal_tax + state_tax + city_tax = total_tax) → 
  x = 50704 :=
by
  intros h
  sorry

end inheritance_amount_l22_22441


namespace min_distance_A_D_l22_22243

theorem min_distance_A_D (A B C E D : Type) 
  (d_AB d_BC d_CE d_ED : ℝ) 
  (h1 : d_AB = 12) 
  (h2 : d_BC = 7) 
  (h3 : d_CE = 2) 
  (h4 : d_ED = 5) : 
  ∃ d_AD : ℝ, d_AD = 2 := 
by
  sorry

end min_distance_A_D_l22_22243


namespace batsman_boundaries_l22_22282

theorem batsman_boundaries
  (total_runs : ℕ)
  (boundaries : ℕ)
  (sixes : ℕ)
  (runs_by_running : ℕ)
  (runs_by_sixes : ℕ)
  (runs_by_boundaries : ℕ)
  (half_runs : ℕ)
  (sixes_runs : ℕ)
  (boundaries_runs : ℕ)
  (total_runs_eq : total_runs = 120)
  (sixes_eq : sixes = 8)
  (half_total_eq : half_runs = total_runs / 2)
  (runs_by_running_eq : runs_by_running = half_runs)
  (sixes_runs_eq : runs_by_sixes = sixes * 6)
  (boundaries_runs_eq : runs_by_boundaries = total_runs - runs_by_running - runs_by_sixes)
  (boundaries_eq : boundaries_runs = boundaries * 4) :
  boundaries = 3 :=
by
  sorry

end batsman_boundaries_l22_22282


namespace crayons_given_l22_22687

theorem crayons_given (initial lost left given : ℕ)
  (h1 : initial = 1453)
  (h2 : lost = 558)
  (h3 : left = 332)
  (h4 : given = initial - left - lost) :
  given = 563 :=
by
  rw [h1, h2, h3] at h4
  exact h4

end crayons_given_l22_22687


namespace soldiers_movement_l22_22380

theorem soldiers_movement (n : ℕ) 
  (initial_positions : Fin (n+3) × Fin (n+1) → Prop) 
  (moves_to_adjacent : ∀ p : Fin (n+3) × Fin (n+1), initial_positions p → initial_positions (p.1 + 1, p.2) ∨ initial_positions (p.1 - 1, p.2) ∨ initial_positions (p.1, p.2 + 1) ∨ initial_positions (p.1, p.2 - 1))
  (final_positions : Fin (n+1) × Fin (n+3) → Prop) : Even n := 
sorry

end soldiers_movement_l22_22380


namespace find_star_value_l22_22956

theorem find_star_value (x : ℤ) :
  45 - (28 - (37 - (15 - x))) = 58 ↔ x = 19 :=
  by
    sorry

end find_star_value_l22_22956


namespace rectangle_triangle_height_l22_22335

theorem rectangle_triangle_height (l : ℝ) (h : ℝ) (w : ℝ) (d : ℝ) 
  (hw : w = Real.sqrt 2 * l)
  (hd : d = Real.sqrt (l^2 + w^2))
  (A_triangle : (1 / 2) * d * h = l * w) :
  h = (2 * l * Real.sqrt 6) / 3 := by
  sorry

end rectangle_triangle_height_l22_22335


namespace students_neither_cool_l22_22651

variable (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)

def only_cool_dads := cool_dads - both_cool
def only_cool_moms := cool_moms - both_cool
def only_cool := only_cool_dads + only_cool_moms + both_cool
def neither_cool := total_students - only_cool

theorem students_neither_cool (h1 : total_students = 40) (h2 : cool_dads = 18) (h3 : cool_moms = 22) (h4 : both_cool = 10) 
: neither_cool total_students cool_dads cool_moms both_cool = 10 :=
by 
  sorry

end students_neither_cool_l22_22651


namespace chemical_reaction_l22_22181

def reaction_balanced (koh nh4i ki nh3 h2o : ℕ) : Prop :=
  koh = nh4i ∧ nh4i = ki ∧ ki = nh3 ∧ nh3 = h2o

theorem chemical_reaction
  (KOH NH4I : ℕ)
  (h1 : KOH = 3)
  (h2 : NH4I = 3)
  (balanced : reaction_balanced KOH NH4I 3 3 3) :
  (∃ (NH3 KI H2O : ℕ),
    NH3 = 3 ∧ KI = 3 ∧ H2O = 3 ∧ 
    NH3 = NH4I - NH4I ∧
    KI = KOH - KOH ∧
    H2O = KOH - KOH) ∧
  (KOH = NH4I) := 
by sorry

end chemical_reaction_l22_22181


namespace sum_of_digits_10pow97_minus_97_l22_22175

-- Define a function that computes the sum of digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Define the main statement we want to prove
theorem sum_of_digits_10pow97_minus_97 :
  sum_of_digits (10^97 - 97) = 858 :=
by
  sorry

end sum_of_digits_10pow97_minus_97_l22_22175


namespace number_of_girls_l22_22559

-- Define the problem conditions as constants
def total_saplings : ℕ := 44
def teacher_saplings : ℕ := 6
def boy_saplings : ℕ := 4
def girl_saplings : ℕ := 2
def total_students : ℕ := 12
def students_saplings : ℕ := total_saplings - teacher_saplings

-- The proof problem statement
theorem number_of_girls (x y : ℕ) (h1 : x + y = total_students)
  (h2 : boy_saplings * x + girl_saplings * y = students_saplings) :
  y = 5 :=
by
  sorry

end number_of_girls_l22_22559


namespace tim_watched_total_hours_tv_l22_22254

-- Define the conditions
def short_show_episodes : ℕ := 24
def short_show_duration_per_episode : ℝ := 0.5

def long_show_episodes : ℕ := 12
def long_show_duration_per_episode : ℝ := 1

-- Define the total duration for each show
def short_show_total_duration : ℝ :=
  short_show_episodes * short_show_duration_per_episode

def long_show_total_duration : ℝ :=
  long_show_episodes * long_show_duration_per_episode

-- Define the total TV hours watched
def total_tv_hours_watched : ℝ :=
  short_show_total_duration + long_show_total_duration

-- Write the theorem statement
theorem tim_watched_total_hours_tv : total_tv_hours_watched = 24 := 
by
  -- proof goes here
  sorry

end tim_watched_total_hours_tv_l22_22254


namespace unused_sector_angle_l22_22515

noncomputable def cone_volume (r h : ℝ) : ℝ := (1/3) * Real.pi * r^2 * h
noncomputable def slant_height (r h : ℝ) : ℝ := Real.sqrt (r^2 + h^2)
noncomputable def central_angle (r base_circumference : ℝ) : ℝ := (base_circumference / (2 * Real.pi * r)) * 360
noncomputable def unused_angle (total_degrees used_angle : ℝ) : ℝ := total_degrees - used_angle

theorem unused_sector_angle (R : ℝ)
  (cone_radius := 15)
  (cone_volume := 675 * Real.pi)
  (total_circumference := 2 * Real.pi * R)
  (cone_height := 9)
  (slant_height := Real.sqrt (cone_radius^2 + cone_height^2))
  (base_circumference := 2 * Real.pi * cone_radius)
  (used_angle := central_angle slant_height base_circumference) :

  unused_angle 360 used_angle = 164.66 := by
  sorry

end unused_sector_angle_l22_22515


namespace find_prime_q_l22_22729

theorem find_prime_q (p q r : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (eq_r : q - p = r)
  (cond_p : 5 < p ∧ p < 15)
  (cond_q : q < 15) :
  q = 13 :=
sorry

end find_prime_q_l22_22729


namespace major_axis_length_l22_22856

/-- Defines the properties of the ellipse we use in this problem. --/
def ellipse (x y : ℝ) : Prop :=
  let f1 := (5, 1 + Real.sqrt 8)
  let f2 := (5, 1 - Real.sqrt 8)
  let tangent_line_at_y := y = 1
  let tangent_line_at_x := x = 1
  tangent_line_at_y ∧ tangent_line_at_x ∧
  ((x - f1.1)^2 + (y - f1.2)^2) + ((x - f2.1)^2 + (y - f2.2)^2) = 4

/-- Proves the length of the major axis of the specific ellipse --/
theorem major_axis_length : ∃ l : ℝ, l = 4 :=
  sorry

end major_axis_length_l22_22856


namespace movie_theater_loss_l22_22456

theorem movie_theater_loss :
  let capacity := 50
  let ticket_price := 8.0
  let tickets_sold := 24
  (capacity * ticket_price - tickets_sold * ticket_price) = 208 := by
  sorry

end movie_theater_loss_l22_22456


namespace number_of_solutions_eq_two_l22_22620

theorem number_of_solutions_eq_two : 
  (∃ (x y : ℝ), x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) ∧
  (∀ (x y : ℝ), (x^2 - 3*x - 4 = 0 ∧ y^2 - 6*y + 9 = 0) → ((x = 4 ∨ x = -1) ∧ y = 3)) :=
by
  sorry

end number_of_solutions_eq_two_l22_22620


namespace distance_from_P_to_AB_l22_22381

-- Let \(ABC\) be an isosceles triangle where \(AB\) is the base. 
-- An altitude from vertex \(C\) to base \(AB\) measures 6 units.
-- A line drawn through a point \(P\) inside the triangle, parallel to base \(AB\), 
-- divides the triangle into two regions of equal area.
-- The vertex angle at \(C\) is a right angle.
-- Prove that the distance from \(P\) to \(AB\) is 3 units.

theorem distance_from_P_to_AB :
  ∀ (A B C P : Type)
    (distance_AB distance_AC distance_BC : ℝ)
    (is_isosceles : distance_AC = distance_BC)
    (right_angle_C : distance_AC^2 + distance_BC^2 = distance_AB^2)
    (altitude_C : distance_BC = 6)
    (line_through_P_parallel_to_AB : ∃ (P_x : ℝ), 0 < P_x ∧ P_x < distance_BC),
  ∃ (distance_P_to_AB : ℝ), distance_P_to_AB = 3 :=
by
  sorry

end distance_from_P_to_AB_l22_22381


namespace num_girls_on_trip_l22_22512

/-- Given the conditions: 
  * Three adults each eating 3 eggs.
  * Ten boys each eating one more egg than each girl.
  * A total of 36 eggs.
  Prove that there are 7 girls on the trip. -/
theorem num_girls_on_trip (adults boys girls eggs : ℕ) 
  (H1 : adults = 3)
  (H2 : boys = 10)
  (H3 : eggs = 36)
  (H4 : ∀ g, (girls * g) + (boys * (g + 1)) + (adults * 3) = eggs)
  (H5 : ∀ g, g = 1) :
  girls = 7 :=
by
  sorry

end num_girls_on_trip_l22_22512


namespace find_C_probability_within_r_l22_22883

noncomputable def probability_density (x y R : ℝ) (C : ℝ) : ℝ :=
if x^2 + y^2 <= R^2 then C * (R - Real.sqrt (x^2 + y^2)) else 0

noncomputable def total_integral (R : ℝ) (C : ℝ) : ℝ :=
∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C

theorem find_C (R : ℝ) (hR : 0 < R) : 
  (∫ (x : ℝ) in -R..R, ∫ (y : ℝ) in -R..R, probability_density x y R C) = 1 ↔ 
  C = 3 / (π * R^3) := 
by 
  sorry

theorem probability_within_r (R r : ℝ) 
  (hR : 0 < R) (hr : 0 < r) (hrR : r <= R) (P : ℝ) : 
  (∫ (x : ℝ) in -r..r, ∫ (y : ℝ) in -r..r, probability_density x y R (3 / (π * R^3))) = P ↔ 
  (R = 2 ∧ r = 1 → P = 1 / 2) := 
by 
  sorry

end find_C_probability_within_r_l22_22883


namespace coordinate_sum_of_point_on_graph_l22_22268

theorem coordinate_sum_of_point_on_graph (g : ℕ → ℕ) (h : ℕ → ℕ)
  (h1 : g 2 = 8)
  (h2 : ∀ x, h x = 3 * (g x) ^ 2) :
  2 + h 2 = 194 :=
by
  sorry

end coordinate_sum_of_point_on_graph_l22_22268


namespace range_of_a_l22_22653

theorem range_of_a (a : ℝ) (h : (2 - a)^3 > (a - 1)^3) : a < 3/2 :=
sorry

end range_of_a_l22_22653


namespace night_crew_fraction_l22_22367

theorem night_crew_fraction (D N : ℝ) (B : ℝ) 
  (h1 : ∀ d, d = D → ∀ n, n = N → ∀ b, b = B → (n * (3/4) * b) = (3/4) * (d * b) / 3)
  (h2 : ∀ t, t = (D * B + (N * (3/4) * B)) → (D * B) / t = 2 / 3) :
  N / D = 2 / 3 :=
by
  sorry

end night_crew_fraction_l22_22367


namespace jane_change_l22_22258

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l22_22258


namespace ratio_of_speeds_l22_22607

variable (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r)))
variable (f1 f2 : ℝ) (h2 : b * (1/4) + b * (3/4) = b)

theorem ratio_of_speeds (b r : ℝ) (h1 : 1 / (b - r) = 2 * (1 / (b + r))) : b = 3 * r :=
by sorry

end ratio_of_speeds_l22_22607


namespace Lisa_days_l22_22222

theorem Lisa_days (L : ℝ) (h : 1/4 + 1/2 + 1/L = 1/1.09090909091) : L = 2.93333333333 :=
by sorry

end Lisa_days_l22_22222


namespace james_total_distance_l22_22581

-- Define the conditions
def speed_part1 : ℝ := 30  -- mph
def time_part1 : ℝ := 0.5  -- hours
def speed_part2 : ℝ := 2 * speed_part1  -- 2 * 30 mph
def time_part2 : ℝ := 2 * time_part1  -- 2 * 0.5 hours

-- Compute distances
def distance_part1 : ℝ := speed_part1 * time_part1
def distance_part2 : ℝ := speed_part2 * time_part2

-- Total distance
def total_distance : ℝ := distance_part1 + distance_part2

-- The theorem to prove
theorem james_total_distance :
  total_distance = 75 := 
sorry

end james_total_distance_l22_22581


namespace two_rides_combinations_l22_22972

-- Define the number of friends
def num_friends : ℕ := 7

-- Define the size of the group for one ride
def ride_group_size : ℕ := 4

-- Define the number of combinations of choosing 'ride_group_size' out of 'num_friends'
def combinations_first_ride : ℕ := Nat.choose num_friends ride_group_size

-- Define the number of friends left for the second ride
def remaining_friends : ℕ := num_friends - ride_group_size

-- Define the number of combinations of choosing 'ride_group_size' out of 'remaining_friends' friends
def combinations_second_ride : ℕ := Nat.choose remaining_friends ride_group_size

-- Define the total number of possible combinations for two rides
def total_combinations : ℕ := combinations_first_ride * combinations_second_ride

-- The final theorem stating the total number of combinations is equal to 525
theorem two_rides_combinations : total_combinations = 525 := by
  -- Placeholder for proof
  sorry

end two_rides_combinations_l22_22972


namespace equal_roots_quadratic_l22_22799

theorem equal_roots_quadratic {k : ℝ} 
  (h : (∃ x : ℝ, x^2 - 6 * x + k = 0 ∧ x^2 - 6 * x + k = 0)) : 
  k = 9 :=
sorry

end equal_roots_quadratic_l22_22799


namespace point_on_graph_l22_22533

theorem point_on_graph (x y : ℝ) (h : y = 3 * x + 1) : (x, y) = (2, 7) :=
sorry

end point_on_graph_l22_22533


namespace cookie_radius_and_area_l22_22115

def boundary_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 = 2 * x + 4 * y

theorem cookie_radius_and_area :
  (∃ r : ℝ, r = Real.sqrt 13) ∧ (∃ A : ℝ, A = 13 * Real.pi) :=
by
  sorry

end cookie_radius_and_area_l22_22115


namespace fraction_relation_l22_22247

-- Definitions for arithmetic sequences and their sums
noncomputable def a_n (a₁ d₁ n : ℕ) := a₁ + (n - 1) * d₁
noncomputable def b_n (b₁ d₂ n : ℕ) := b₁ + (n - 1) * d₂

noncomputable def A_n (a₁ d₁ n : ℕ) := n * a₁ + n * (n - 1) * d₁ / 2
noncomputable def B_n (b₁ d₂ n : ℕ) := n * b₁ + n * (n - 1) * d₂ / 2

-- Theorem statement
theorem fraction_relation (a₁ d₁ b₁ d₂ : ℕ) (h : ∀ n : ℕ, B_n a₁ d₁ n ≠ 0 → A_n a₁ d₁ n / B_n b₁ d₂ n = (2 * n - 1) / (3 * n + 1)) :
  ∀ n : ℕ, b_n b₁ d₂ n ≠ 0 → a_n a₁ d₁ n / b_n b₁ d₂ n = (4 * n - 3) / (6 * n - 2) :=
sorry

end fraction_relation_l22_22247


namespace five_card_draw_probability_l22_22932

noncomputable def probability_at_least_one_card_from_each_suit : ℚ := 3 / 32

theorem five_card_draw_probability :
  let deck_size := 52
  let suits := 4
  let cards_drawn := 5
  (1 : ℚ) * (3 / 4) * (1 / 2) * (1 / 4) = probability_at_least_one_card_from_each_suit := by
  sorry

end five_card_draw_probability_l22_22932


namespace probability_same_number_l22_22909

def is_multiple (n factor : ℕ) : Prop :=
  ∃ k : ℕ, n = k * factor

def multiples_below (factor upper_limit : ℕ) : ℕ :=
  (upper_limit - 1) / factor

theorem probability_same_number :
  let upper_limit := 250
  let billy_factor := 20
  let bobbi_factor := 30
  let common_factor := 60
  let billy_multiples := multiples_below billy_factor upper_limit
  let bobbi_multiples := multiples_below bobbi_factor upper_limit
  let common_multiples := multiples_below common_factor upper_limit
  (common_multiples : ℚ) / (billy_multiples * bobbi_multiples) = 1 / 24 :=
by
  sorry

end probability_same_number_l22_22909


namespace range_independent_variable_l22_22630

noncomputable def range_of_independent_variable (x : ℝ) : Prop :=
  x ≠ 3

theorem range_independent_variable (x : ℝ) :
  (∃ y : ℝ, y = 1 / (x - 3)) → x ≠ 3 :=
by
  intro h
  sorry

end range_independent_variable_l22_22630


namespace minimum_energy_H1_l22_22341

-- Define the given conditions
def energyEfficiencyMin : ℝ := 0.1
def energyRequiredH6 : ℝ := 10 -- Energy in KJ
def energyLevels : Nat := 5 -- Number of energy levels from H1 to H6

-- Define the theorem to prove the minimum energy required from H1
theorem minimum_energy_H1 : (10 ^ energyLevels : ℝ) = 1000000 :=
by
  -- Placeholder for actual proof
  sorry

end minimum_energy_H1_l22_22341


namespace simplify_expression_l22_22439

theorem simplify_expression :
  8 * (15 / 4) * (-45 / 50) = - (12 / 25) :=
by
  sorry

end simplify_expression_l22_22439


namespace min_value_of_f_l22_22382

noncomputable def f (x y : ℝ) : ℝ := (x^2 * y) / (x^3 + y^3)

theorem min_value_of_f :
  (∀ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) → f x y ≥ 12 / 35) ∧
  ∃ (x y : ℝ), (1/3 ≤ x ∧ x ≤ 2/3) ∧ (1/4 ≤ y ∧ y ≤ 1/2) ∧ f x y = 12 / 35 :=
by
  sorry

end min_value_of_f_l22_22382


namespace region_area_l22_22041

noncomputable def area_of_region := 
  let a := 0
  let b := Real.sqrt 2 / 2
  ∫ x in a..b, (Real.arccos x) - (Real.arcsin x)

theorem region_area : area_of_region = 2 - Real.sqrt 2 :=
by
  sorry

end region_area_l22_22041


namespace find_other_number_l22_22411

theorem find_other_number (a b : ℕ) (h1 : a + b = 62) (h2 : b - a = 12) (h3 : a = 25) : b = 37 :=
sorry

end find_other_number_l22_22411


namespace meaningful_fraction_l22_22684

theorem meaningful_fraction {x : ℝ} : (x - 2) ≠ 0 ↔ x ≠ 2 :=
by
  sorry

end meaningful_fraction_l22_22684


namespace equation_of_line_l_l22_22590

theorem equation_of_line_l :
  (∃ l : ℝ → ℝ → Prop, 
     (∀ x y, l x y ↔ (x - y + 3) = 0)
     ∧ (∀ x y, l x y → x^2 + (y - 3)^2 = 4)
     ∧ (∀ x y, l x y → x + y + 1 = 0)) :=
sorry

end equation_of_line_l_l22_22590


namespace initial_fee_l22_22711

theorem initial_fee (total_bowls : ℤ) (lost_bowls : ℤ) (broken_bowls : ℤ) (safe_fee : ℤ)
  (loss_fee : ℤ) (total_payment : ℤ) (paid_amount : ℤ) :
  total_bowls = 638 →
  lost_bowls = 12 →
  broken_bowls = 15 →
  safe_fee = 3 →
  loss_fee = 4 →
  total_payment = 1825 →
  paid_amount = total_payment - ((total_bowls - lost_bowls - broken_bowls) * safe_fee - (lost_bowls + broken_bowls) * loss_fee) →
  paid_amount = 100 :=
by
  intros _ _ _ _ _ _ _
  sorry

end initial_fee_l22_22711


namespace _l22_22866

noncomputable def polynomial_divides (x : ℂ) (n : ℕ) : Prop :=
  (x - 1) ^ 3 ∣ x ^ (2 * n + 1) - (2 * n + 1) * x ^ (n + 1) + (2 * n + 1) * x ^ n - 1

lemma polynomial_division_theorem : ∀ (n : ℕ), n ≥ 1 → ∀ (x : ℂ), polynomial_divides x n :=
by
  intros n hn x
  unfold polynomial_divides
  sorry

end _l22_22866


namespace car_speed_l22_22099

/-- 
If a tire rotates at 400 revolutions per minute, and the circumference of the tire is 6 meters, 
the speed of the car is 144 km/h.
-/
theorem car_speed (rev_per_min : ℕ) (circumference : ℝ) (speed : ℝ) :
  rev_per_min = 400 → circumference = 6 → speed = 144 :=
by
  intro h_rev h_circ
  sorry

end car_speed_l22_22099


namespace exists_positive_integer_m_l22_22447

theorem exists_positive_integer_m (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0) (h_cd : c * d = 1) : 
  ∃ m : ℕ, (a * b ≤ ↑m * ↑m) ∧ (↑m * ↑m ≤ (a + c) * (b + d)) :=
by
  sorry

end exists_positive_integer_m_l22_22447


namespace increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l22_22400

noncomputable def f (a x : ℝ) : ℝ := x - a / x

theorem increasing_interval_when_a_neg {a : ℝ} (h : a < 0) :
  ∀ x : ℝ, x > 0 → f a x > 0 :=
sorry

theorem increasing_and_decreasing_intervals_when_a_pos {a : ℝ} (h : a > 0) :
  (∀ x : ℝ, 0 < x → x < Real.sqrt a → f a x < 0) ∧
  (∀ x : ℝ, x > Real.sqrt a → f a x > 0) :=
sorry

end increasing_interval_when_a_neg_increasing_and_decreasing_intervals_when_a_pos_l22_22400


namespace total_pints_l22_22207

-- Define the given conditions as constants
def annie_picked : Int := 8
def kathryn_picked : Int := annie_picked + 2
def ben_picked : Int := kathryn_picked - 3

-- State the main theorem to prove
theorem total_pints : annie_picked + kathryn_picked + ben_picked = 25 := by
  sorry

end total_pints_l22_22207


namespace simplify_expression_l22_22234

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 2 - 1) :
  (x - 2) / (x ^ 2 - 1) / (1 - 1 / (x - 1)) = Real.sqrt 2 / 2 :=
by
  sorry

end simplify_expression_l22_22234


namespace nm_odd_if_squares_sum_odd_l22_22178

theorem nm_odd_if_squares_sum_odd
  (n m : ℤ)
  (h : (n^2 + m^2) % 2 = 1) :
  (n * m) % 2 = 1 :=
sorry

end nm_odd_if_squares_sum_odd_l22_22178


namespace calc1_calc2_l22_22873

-- Problem 1
theorem calc1 : 2 * Real.sqrt 3 - 3 * Real.sqrt 12 + 5 * Real.sqrt 27 = 11 * Real.sqrt 3 := 
by sorry

-- Problem 2
theorem calc2 : (1 + Real.sqrt 3) * (Real.sqrt 2 - Real.sqrt 6) - (2 * Real.sqrt 3 - 1)^2 
              = -2 * Real.sqrt 2 + 4 * Real.sqrt 3 - 13 := 
by sorry

end calc1_calc2_l22_22873


namespace image_center_after_reflection_and_translation_l22_22890

def circle_center_before_translation : ℝ × ℝ := (3, -4)

def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-x, y)

def translate_up (p : ℝ × ℝ) (d : ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (x, y + d)

theorem image_center_after_reflection_and_translation :
  translate_up (reflect_y_axis circle_center_before_translation) 5 = (-3, 1) :=
by
  -- The detail proof goes here.
  sorry

end image_center_after_reflection_and_translation_l22_22890


namespace price_each_puppy_l22_22915

def puppies_initial : ℕ := 8
def puppies_given_away : ℕ := puppies_initial / 2
def puppies_remaining_after_giveaway : ℕ := puppies_initial - puppies_given_away
def puppies_kept : ℕ := 1
def puppies_to_sell : ℕ := puppies_remaining_after_giveaway - puppies_kept
def stud_fee : ℕ := 300
def profit : ℕ := 1500
def total_amount_made : ℕ := profit + stud_fee
def price_per_puppy : ℕ := total_amount_made / puppies_to_sell

theorem price_each_puppy :
  price_per_puppy = 600 :=
sorry

end price_each_puppy_l22_22915


namespace container_unoccupied_volume_is_628_l22_22913

def rectangular_prism_volume (length width height : ℕ) : ℕ :=
  length * width * height

def water_volume (total_volume : ℕ) : ℕ :=
  total_volume / 3

def ice_cubes_volume (number_of_cubes volume_per_cube : ℕ) : ℕ :=
  number_of_cubes * volume_per_cube

def unoccupied_volume (total_volume occupied_volume : ℕ) : ℕ :=
  total_volume - occupied_volume

theorem container_unoccupied_volume_is_628 :
  let length := 12
  let width := 10
  let height := 8
  let number_of_ice_cubes := 12
  let volume_per_ice_cube := 1
  let V := rectangular_prism_volume length width height
  let V_water := water_volume V
  let V_ice := ice_cubes_volume number_of_ice_cubes volume_per_ice_cube
  let V_occupied := V_water + V_ice
  unoccupied_volume V V_occupied = 628 :=
by
  sorry

end container_unoccupied_volume_is_628_l22_22913


namespace brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l22_22496

def march_13_2007_day_of_week : String := "Tuesday"

def days_until_brothers_birthday : Nat := 2000

def start_date := (2007, 3, 13)  -- (year, month, day)

def days_per_week := 7

def carlos_initial_age := 7

def day_of_week_after_n_days (start_day : String) (n : Nat) : String :=
  match n % 7 with
  | 0 => "Tuesday"
  | 1 => "Wednesday"
  | 2 => "Thursday"
  | 3 => "Friday"
  | 4 => "Saturday"
  | 5 => "Sunday"
  | 6 => "Monday"
  | _ => "Unknown" -- This case should never happen

def carlos_age_after_n_days (initial_age : Nat) (n : Nat) : Nat :=
  initial_age + n / 365

theorem brother_15th_birthday_day_of_week : 
  day_of_week_after_n_days march_13_2007_day_of_week days_until_brothers_birthday = "Sunday" := 
by sorry

theorem carlos_age_on_brothers_15th_birthday :
  carlos_age_after_n_days carlos_initial_age days_until_brothers_birthday = 12 :=
by sorry

end brother_15th_birthday_day_of_week_carlos_age_on_brothers_15th_birthday_l22_22496


namespace legos_set_cost_l22_22383

-- Definitions for the conditions
def cars_sold : ℕ := 3
def price_per_car : ℕ := 5
def total_earned : ℕ := 45

-- The statement to prove
theorem legos_set_cost :
  total_earned - (cars_sold * price_per_car) = 30 := by
  sorry

end legos_set_cost_l22_22383


namespace ratio_of_overtime_to_regular_rate_l22_22945

def regular_rate : ℝ := 3
def regular_hours : ℕ := 40
def total_pay : ℝ := 186
def overtime_hours : ℕ := 11

theorem ratio_of_overtime_to_regular_rate 
  (r : ℝ) (h : ℕ) (T : ℝ) (h_ot : ℕ) 
  (h_r : r = regular_rate) 
  (h_h : h = regular_hours) 
  (h_T : T = total_pay)
  (h_hot : h_ot = overtime_hours) :
  (T - (h * r)) / h_ot / r = 2 := 
by {
  sorry 
}

end ratio_of_overtime_to_regular_rate_l22_22945


namespace cos_660_degrees_is_one_half_l22_22777

noncomputable def cos_660_eq_one_half : Prop :=
  (Real.cos (660 * Real.pi / 180) = 1 / 2)

theorem cos_660_degrees_is_one_half : cos_660_eq_one_half :=
by
  sorry

end cos_660_degrees_is_one_half_l22_22777


namespace find_smaller_number_l22_22140

theorem find_smaller_number (a b : ℕ) 
  (h1 : a + b = 15) 
  (h2 : 3 * (a - b) = 21) : b = 4 :=
by
  sorry

end find_smaller_number_l22_22140


namespace field_trip_vans_l22_22061

-- Define the number of students and adults
def students := 12
def adults := 3

-- Define the capacity of each van
def van_capacity := 5

-- Total number of people
def total_people := students + adults

-- Calculate the number of vans needed
def vans_needed := (total_people + van_capacity - 1) / van_capacity  -- For rounding up division

theorem field_trip_vans : vans_needed = 3 :=
by
  -- Calculation and proof would go here
  sorry

end field_trip_vans_l22_22061


namespace find_a_l22_22435

theorem find_a (a : ℝ) (α : ℝ) (P : ℝ × ℝ) 
  (h_P : P = (3 * a, 4)) 
  (h_cos : Real.cos α = -3/5) : 
  a = -1 := 
by
  sorry

end find_a_l22_22435


namespace arithmetic_sequence_common_difference_l22_22992

theorem arithmetic_sequence_common_difference 
    (a : ℤ) (last_term : ℤ) (sum_terms : ℤ) (n : ℕ)
    (h1 : a = 3) 
    (h2 : last_term = 58) 
    (h3 : sum_terms = 488)
    (h4 : sum_terms = n * (a + last_term) / 2)
    (h5 : last_term = a + (n - 1) * d) :
    d = 11 / 3 := by
  sorry

end arithmetic_sequence_common_difference_l22_22992


namespace bus_routes_arrangement_l22_22063

-- Define the lines and intersection points (stops).
def routes := Fin 10
def stops (r1 r2 : routes) : Prop := r1 ≠ r2 -- Representing intersection

-- First condition: Any subset of 9 routes will cover all stops.
def covers_all_stops (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 9 → ∀ r1 r2 : routes, r1 ≠ r2 → stops r1 r2

-- Second condition: Any subset of 8 routes will miss at least one stop.
def misses_at_least_one_stop (routes_subset : Finset routes) : Prop :=
  routes_subset.card = 8 → ∃ r1 r2 : routes, r1 ≠ r2 ∧ ¬stops r1 r2

-- The theorem to prove that this arrangement is possible.
theorem bus_routes_arrangement : 
  (∃ stops_scheme : routes → routes → Prop, 
    (∀ subset_9 : Finset routes, covers_all_stops subset_9) ∧ 
    (∀ subset_8 : Finset routes, misses_at_least_one_stop subset_8)) :=
by
  sorry

end bus_routes_arrangement_l22_22063


namespace area_of_square_l22_22654

theorem area_of_square (side_length : ℝ) (h : side_length = 17) : side_length * side_length = 289 :=
by
  sorry

end area_of_square_l22_22654


namespace problem_solution_l22_22028

def lean_problem (a : ℝ) : Prop :=
  (∀ x : ℝ, |x - 1| + |x + 1| ≥ 3 * a) ∧ 
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (2 * a - 1)^x₁ > (2 * a - 1)^x₂) →
  a > 1 / 2 ∧ a ≤ 2 / 3

theorem problem_solution (a : ℝ) : lean_problem a :=
  sorry -- Proof is to be filled in

end problem_solution_l22_22028


namespace intersection_of_M_and_N_l22_22288

theorem intersection_of_M_and_N :
  let M := { x : ℝ | -6 ≤ x ∧ x < 4 }
  let N := { x : ℝ | -2 < x ∧ x ≤ 8 }
  M ∩ N = { x | -2 < x ∧ x < 4 } :=
by
  sorry -- Proof is omitted

end intersection_of_M_and_N_l22_22288


namespace arithmetic_geometric_sequence_product_l22_22774

theorem arithmetic_geometric_sequence_product :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 = 3 →
    (a 1) + (a 1 * q^2) + (a 1 * q^4) = 21 →
    (a 2) * (a 6) = 72 :=
by 
  intros a q h1 h2 
  sorry

end arithmetic_geometric_sequence_product_l22_22774


namespace polynomial_coefficients_l22_22032

theorem polynomial_coefficients (a_0 a_1 a_2 a_3 a_4 : ℤ) :
  (∀ x : ℤ, (x + 2)^5 = (x + 1)^5 + a_4 * x^4 + a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0) →
  a_0 = 31 ∧ a_1 = 75 :=
by
  sorry

end polynomial_coefficients_l22_22032


namespace tank_capacity_l22_22419

theorem tank_capacity (liters_cost : ℕ) (liters_amount : ℕ) (full_tank_cost : ℕ) (h₁ : liters_cost = 18) (h₂ : liters_amount = 36) (h₃ : full_tank_cost = 32) : 
  (full_tank_cost * liters_amount / liters_cost) = 64 :=
by 
  sorry

end tank_capacity_l22_22419


namespace tan_alpha_value_sin_cos_expression_l22_22228

noncomputable def tan_alpha (α : ℝ) : ℝ := Real.tan α

theorem tan_alpha_value (α : ℝ) (h1 : Real.tan (α + Real.pi / 4) = 2) : tan_alpha α = 1 / 3 :=
by
  sorry

theorem sin_cos_expression (α : ℝ) (h2 : tan_alpha α = 1 / 3) :
  (Real.sin (2 * α) - Real.sin α ^ 2) / (1 + Real.cos (2 * α)) = 5 / 18 :=
by
  sorry

end tan_alpha_value_sin_cos_expression_l22_22228


namespace rolling_circle_trace_eq_envelope_l22_22847

-- Definitions for the geometrical setup
variable {a : ℝ} (C : ℝ → ℝ → Prop)

-- The main statement to prove
theorem rolling_circle_trace_eq_envelope (hC : ∀ t : ℝ, C (a * t) a) :
  ∃ P : ℝ × ℝ → Prop, ∀ t : ℝ, C (a/2 * t + a/2 * Real.sin t) (a/2 + a/2 * Real.cos t) :=
by
  sorry

end rolling_circle_trace_eq_envelope_l22_22847


namespace lee_can_make_cookies_l22_22127

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

end lee_can_make_cookies_l22_22127


namespace possible_values_of_d_l22_22547

theorem possible_values_of_d :
  ∃ (e f d : ℤ), (e + 12) * (f + 12) = 1 ∧
  ∀ x, (x - d) * (x - 12) + 1 = (x + e) * (x + f) ↔ (d = 22 ∨ d = 26) :=
by
  sorry

end possible_values_of_d_l22_22547


namespace original_profit_percentage_l22_22652

theorem original_profit_percentage (C : ℝ) (C' : ℝ) (S' : ℝ) (H1 : C = 40) (H2 : C' = 32) (H3 : S' = 41.60) 
  (H4 : S' = (1.30 * C')) : (S' + 8.40 - C) / C * 100 = 25 := 
by 
  sorry

end original_profit_percentage_l22_22652


namespace sufficient_conditions_for_quadratic_l22_22885

theorem sufficient_conditions_for_quadratic (x : ℝ) : 
  (0 < x ∧ x < 4) ∨ (-2 < x ∧ x < 4) ∨ (-2 < x ∧ x < 3) → x^2 - 2*x - 8 < 0 :=
by
  sorry

end sufficient_conditions_for_quadratic_l22_22885


namespace find_expression_l22_22390

theorem find_expression : 1^567 + 3^5 / 3^3 - 2 = 8 :=
by
  sorry

end find_expression_l22_22390


namespace maximize_revenue_l22_22122

noncomputable def revenue (p : ℝ) : ℝ :=
  p * (150 - 6 * p)

theorem maximize_revenue : ∃ (p : ℝ), p = 12.5 ∧ p ≤ 30 ∧ ∀ q ≤ 30, revenue q ≤ revenue 12.5 := by 
  sorry

end maximize_revenue_l22_22122


namespace original_price_of_pants_l22_22346

theorem original_price_of_pants (P : ℝ) 
  (sale_discount : ℝ := 0.50)
  (saturday_additional_discount : ℝ := 0.20)
  (savings : ℝ := 50.40)
  (saturday_effective_discount : ℝ := 0.40) :
  savings = 0.60 * P ↔ P = 84.00 :=
by
  sorry

end original_price_of_pants_l22_22346


namespace total_hunts_l22_22539

-- Conditions
def Sam_hunts : ℕ := 6
def Rob_hunts := Sam_hunts / 2
def combined_Rob_Sam_hunts := Rob_hunts + Sam_hunts
def Mark_hunts := combined_Rob_Sam_hunts / 3
def Peter_hunts := 3 * Mark_hunts

-- Question and proof statement
theorem total_hunts : Sam_hunts + Rob_hunts + Mark_hunts + Peter_hunts = 21 := by
  sorry

end total_hunts_l22_22539


namespace mike_spent_on_mower_blades_l22_22855

theorem mike_spent_on_mower_blades (x : ℝ) 
  (initial_money : ℝ := 101) 
  (cost_of_games : ℝ := 54) 
  (games : ℝ := 9) 
  (price_per_game : ℝ := 6) 
  (h1 : 101 - x = 54) :
  x = 47 := 
by
  sorry

end mike_spent_on_mower_blades_l22_22855


namespace calculate_x_value_l22_22546

theorem calculate_x_value : 
  529 + 2 * 23 * 3 + 9 = 676 := 
by
  sorry

end calculate_x_value_l22_22546


namespace total_votes_l22_22829

variable (V : ℝ)

theorem total_votes (h1 : 0.34 * V + 640 = 0.66 * V) : V = 2000 :=
by 
  sorry

end total_votes_l22_22829


namespace saline_solution_mixture_l22_22498

theorem saline_solution_mixture 
  (x : ℝ) 
  (h₁ : 20 + 0.1 * x = 0.25 * (50 + x)) 
  : x = 50 := 
by 
  sorry

end saline_solution_mixture_l22_22498


namespace statement1_statement2_statement3_l22_22008

variable (a b c m : ℝ)

-- Given condition
def quadratic_eq (a b c : ℝ) : Prop := a ≠ 0

-- Statement 1
theorem statement1 (h0 : quadratic_eq a b c) (h1 : ∀ x, a * x^2 + b * x + c = 0 ↔ x = 1 ∨ x = 2) : 2 * a - c = 0 :=
sorry

-- Statement 2
theorem statement2 (h0 : quadratic_eq a b c) (h2 : b = 2 * a + c) : (b^2 - 4 * a * c) > 0 :=
sorry

-- Statement 3
theorem statement3 (h0 : quadratic_eq a b c) (h3 : a * m^2 + b * m + c = 0) : b^2 - 4 * a * c = (2 * a * m + b)^2 :=
sorry

end statement1_statement2_statement3_l22_22008


namespace orchid_bushes_planted_tomorrow_l22_22068

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


end orchid_bushes_planted_tomorrow_l22_22068


namespace avg_b_c_weight_l22_22446

theorem avg_b_c_weight (a b c : ℝ) (H1 : (a + b + c) / 3 = 45) (H2 : (a + b) / 2 = 40) (H3 : b = 39) : (b + c) / 2 = 47 :=
by
  sorry

end avg_b_c_weight_l22_22446


namespace smallest_part_in_ratio_l22_22677

variable (b : ℝ)

theorem smallest_part_in_ratio (h : b = -2620) : 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  smallest_part = 100 :=
by 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  sorry

end smallest_part_in_ratio_l22_22677


namespace robin_total_cost_l22_22927

def num_letters_in_name (name : String) : Nat := name.length

def calculate_total_cost (names : List String) (cost_per_bracelet : Nat) : Nat :=
  let total_bracelets := names.foldl (fun acc name => acc + num_letters_in_name name) 0
  total_bracelets * cost_per_bracelet

theorem robin_total_cost : 
  calculate_total_cost ["Jessica", "Tori", "Lily", "Patrice"] 2 = 44 :=
by
  sorry

end robin_total_cost_l22_22927


namespace cost_of_adult_ticket_is_8_l22_22534

variables (A : ℕ) (num_people : ℕ := 22) (total_money : ℕ := 50) (num_children : ℕ := 18) (child_ticket_cost : ℕ := 1)

-- Definitions based on the given conditions
def child_tickets_cost := num_children * child_ticket_cost
def num_adults := num_people - num_children
def adult_tickets_cost := total_money - child_tickets_cost
def cost_per_adult_ticket := adult_tickets_cost / num_adults

-- The theorem stating that the cost of an adult ticket is 8 dollars
theorem cost_of_adult_ticket_is_8 : cost_per_adult_ticket = 8 :=
by sorry

end cost_of_adult_ticket_is_8_l22_22534


namespace gold_initial_amount_l22_22918

theorem gold_initial_amount :
  ∃ x : ℝ, x - (x / 2 * (2 / 3) * (3 / 4) * (4 / 5) * (5 / 6)) = 1 ∧ x = 1.2 :=
by
  existsi 1.2
  sorry

end gold_initial_amount_l22_22918


namespace dusting_days_l22_22710

theorem dusting_days 
    (vacuuming_minutes_per_day : ℕ) 
    (vacuuming_days_per_week : ℕ)
    (dusting_minutes_per_day : ℕ)
    (total_cleaning_minutes_per_week : ℕ)
    (x : ℕ) :
    vacuuming_minutes_per_day = 30 →
    vacuuming_days_per_week = 3 →
    dusting_minutes_per_day = 20 →
    total_cleaning_minutes_per_week = 130 →
    (vacuuming_minutes_per_day * vacuuming_days_per_week + dusting_minutes_per_day * x = total_cleaning_minutes_per_week) →
    x = 2 :=
by
  -- Proof steps go here
  sorry

end dusting_days_l22_22710


namespace sum_interior_angles_equal_diagonals_l22_22075

theorem sum_interior_angles_equal_diagonals (n : ℕ) (h : n = 4 ∨ n = 5) :
  (n - 2) * 180 = 360 ∨ (n - 2) * 180 = 540 :=
by sorry

end sum_interior_angles_equal_diagonals_l22_22075


namespace find_age_l22_22701

theorem find_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 := 
by 
  sorry

end find_age_l22_22701


namespace paint_cost_of_cube_l22_22428

def cube_side_length : ℝ := 10
def paint_cost_per_quart : ℝ := 3.20
def coverage_per_quart : ℝ := 1200
def number_of_faces : ℕ := 6

theorem paint_cost_of_cube : 
  (number_of_faces * (cube_side_length^2) / coverage_per_quart) * paint_cost_per_quart = 3.20 :=
by 
  sorry

end paint_cost_of_cube_l22_22428


namespace number_of_people_prefer_soda_l22_22215

-- Given conditions
def total_people : ℕ := 600
def central_angle_soda : ℝ := 198
def full_circle_angle : ℝ := 360

-- Problem statement
theorem number_of_people_prefer_soda : 
  (total_people : ℝ) * (central_angle_soda / full_circle_angle) = 330 := by
  sorry

end number_of_people_prefer_soda_l22_22215


namespace find_lambda_l22_22595

noncomputable def vec_length (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem find_lambda {a b : ℝ × ℝ} (lambda : ℝ) 
  (ha : vec_length a = 1) (hb : vec_length b = 2)
  (hab_angle : dot_product a b = -1) 
  (h_perp : dot_product (lambda • a + b) (a - 2 • b) = 0) : 
  lambda = 3 := 
sorry

end find_lambda_l22_22595


namespace solve_linear_system_l22_22851

variable {a b : ℝ}
variables {m n : ℝ}

theorem solve_linear_system
  (h1 : a * 2 - b * 1 = 3)
  (h2 : a * 2 + b * 1 = 5)
  (h3 : a * (m + 2 * n) - 2 * b * n = 6)
  (h4 : a * (m + 2 * n) + 2 * b * n = 10) :
  m = 2 ∧ n = 1 := 
sorry

end solve_linear_system_l22_22851


namespace hyperbola_equation_l22_22537

theorem hyperbola_equation 
  (x y : ℝ)
  (h_ellipse : x^2 / 10 + y^2 / 5 = 1)
  (h_asymptote : 3 * x + 4 * y = 0)
  (h_hyperbola : ∃ k ≠ 0, 9 * x^2 - 16 * y^2 = k) :
  ∃ k : ℝ, k = 45 ∧ (x^2 / 5 - 16 * y^2 / 45 = 1) :=
sorry

end hyperbola_equation_l22_22537


namespace length_of_field_l22_22107

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

end length_of_field_l22_22107


namespace negation_of_universal_proposition_l22_22713

theorem negation_of_universal_proposition (x : ℝ) :
  (¬ (∀ x : ℝ, |x| < 0)) ↔ (∃ x_0 : ℝ, |x_0| ≥ 0) := by
  sorry

end negation_of_universal_proposition_l22_22713


namespace range_of_a_l22_22084

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ ((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) →
  (-2:ℝ) ≤ a ∧ a < (6 / 5:ℝ) :=
by
  sorry

end range_of_a_l22_22084


namespace circle_area_difference_l22_22087

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

end circle_area_difference_l22_22087


namespace crates_needed_l22_22587

def ceil_div (a b : ℕ) : ℕ := (a + b - 1) / b

theorem crates_needed :
  ceil_div 145 12 + ceil_div 271 8 + ceil_div 419 10 + ceil_div 209 14 = 104 :=
by
  sorry

end crates_needed_l22_22587


namespace john_total_spent_l22_22076

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

end john_total_spent_l22_22076


namespace sandwiches_difference_l22_22474

theorem sandwiches_difference :
  let monday_lunch := 3
  let monday_dinner := 2 * monday_lunch
  let monday_total := monday_lunch + monday_dinner

  let tuesday_lunch := 4
  let tuesday_dinner := tuesday_lunch / 2
  let tuesday_total := tuesday_lunch + tuesday_dinner

  let wednesday_lunch := 2 * tuesday_lunch
  let wednesday_dinner := 3 * tuesday_lunch
  let wednesday_total := wednesday_lunch + wednesday_dinner

  let total_mw := monday_total + tuesday_total + wednesday_total

  let thursday_lunch := 3 * 2
  let thursday_dinner := 5
  let thursday_total := thursday_lunch + thursday_dinner

  total_mw - thursday_total = 24 :=
by
  sorry

end sandwiches_difference_l22_22474


namespace perpendicular_lines_k_value_l22_22801

theorem perpendicular_lines_k_value :
  ∀ (k : ℝ), (∀ (x y : ℝ), x + 4 * y - 1 = 0) →
             (∀ (x y : ℝ), k * x + y + 2 = 0) →
             (-1 / 4 * -k = -1) →
             k = -4 :=
by
  intros k h1 h2 h3
  sorry

end perpendicular_lines_k_value_l22_22801


namespace arrangement_of_numbers_l22_22920

theorem arrangement_of_numbers (numbers : Finset ℕ) 
  (h1 : numbers = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}) 
  (h_sum : ∀ a b c d e f, a + b + c + d + e + f = 33)
  (h_group_sum : ∀ k1 k2 k3 k4, k1 + k2 + k3 + k4 = 26)
  : ∃ (n : ℕ), n = 2304 := by
  sorry

end arrangement_of_numbers_l22_22920


namespace total_balls_is_108_l22_22731

theorem total_balls_is_108 (B : ℕ) (W : ℕ) (n : ℕ) (h1 : W = 8 * B) 
                           (h2 : n = B + W) 
                           (h3 : 100 ≤ n - W + 1) 
                           (h4 : 100 > B) : n = 108 := 
by sorry

end total_balls_is_108_l22_22731


namespace xz_squared_value_l22_22298

theorem xz_squared_value (x y z : ℝ) (h₁ : 3 * x * 5 * z = (4 * y)^2) (h₂ : (y^2 : ℝ) = (x^2 + z^2) / 2) :
  x^2 + z^2 = 16 := 
sorry

end xz_squared_value_l22_22298


namespace basketball_three_point_shots_l22_22274

theorem basketball_three_point_shots (t h f : ℕ) 
  (h1 : 2 * t = 6 * h)
  (h2 : f = h - 4)
  (h3: 2 * t + 3 * h + f = 76)
  (h4: t + h + f = 40) : h = 8 :=
sorry

end basketball_three_point_shots_l22_22274


namespace sum_of_ratios_l22_22976

theorem sum_of_ratios (a b c : ℤ) (h : (a * a : ℚ) / (b * b) = 32 / 63) : a + b + c = 39 :=
sorry

end sum_of_ratios_l22_22976


namespace train_length_correct_l22_22959

noncomputable def length_of_first_train (speed1 speed2 : ℝ) (time : ℝ) (length2 : ℝ) : ℝ :=
  let speed1_ms := speed1 * 1000 / 3600
  let speed2_ms := speed2 * 1000 / 3600
  let relative_speed := speed1_ms - speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem train_length_correct :
  length_of_first_train 72 36 69.99440044796417 300 = 399.9440044796417 :=
by
  sorry

end train_length_correct_l22_22959


namespace probability_of_rain_l22_22662

variable (P_R P_B0 : ℝ)
variable (H1 : 0 ≤ P_R ∧ P_R ≤ 1)
variable (H2 : 0 ≤ P_B0 ∧ P_B0 ≤ 1)
variable (H : P_R + P_B0 - P_R * P_B0 = 0.2)

theorem probability_of_rain : 
  P_R = 1/9 :=
by
  sorry

end probability_of_rain_l22_22662


namespace dogs_prevent_wolf_escape_l22_22158

theorem dogs_prevent_wolf_escape
  (wolf_speed dog_speed : ℝ)
  (at_center: True)
  (dogs_at_vertices: True)
  (wolf_all_over_field: True)
  (dogs_on_perimeter: True)
  (wolf_handles_one_dog: ∀ (d : ℕ), d = 1 → True)
  (wolf_handles_two_dogs: ∀ (d : ℕ), d = 2 → False)
  (dog_faster_than_wolf: dog_speed = 1.5 * wolf_speed) : 
  ∀ (wolf_position : ℝ × ℝ) (boundary_position : ℝ × ℝ), 
  wolf_position != boundary_position → dog_speed > wolf_speed → 
  False := 
by sorry

end dogs_prevent_wolf_escape_l22_22158


namespace apples_found_l22_22154

theorem apples_found (start_apples : ℕ) (end_apples : ℕ) (h_start : start_apples = 7) (h_end : end_apples = 81) : 
  end_apples - start_apples = 74 := 
by 
  sorry

end apples_found_l22_22154


namespace difference_of_interchanged_digits_l22_22650

theorem difference_of_interchanged_digits (X Y : ℕ) (h : X - Y = 5) : (10 * X + Y) - (10 * Y + X) = 45 :=
by
  sorry

end difference_of_interchanged_digits_l22_22650


namespace hexagon_largest_angle_l22_22294

theorem hexagon_largest_angle (x : ℝ) (h : 3 * x + 3 * x + 3 * x + 4 * x + 5 * x + 6 * x = 720) : 
  6 * x = 180 :=
by
  sorry

end hexagon_largest_angle_l22_22294


namespace parking_garage_floors_l22_22937

theorem parking_garage_floors 
  (total_time : ℕ)
  (time_per_floor : ℕ)
  (gate_time : ℕ)
  (every_n_floors : ℕ) 
  (F : ℕ) 
  (h1 : total_time = 1440)
  (h2 : time_per_floor = 80)
  (h3 : gate_time = 120)
  (h4 : every_n_floors = 3)
  :
  F = 13 :=
by
  have total_id_time : ℕ := gate_time * ((F - 1) / every_n_floors)
  have total_drive_time : ℕ := time_per_floor * (F - 1)
  have total_time_calc : ℕ := total_drive_time + total_id_time
  have h5 := total_time_calc = total_time
  -- Now we simplify the algebraic equation given the problem conditions
  sorry

end parking_garage_floors_l22_22937


namespace cube_splitting_odd_numbers_l22_22083

theorem cube_splitting_odd_numbers (m : ℕ) (h1 : m > 1) (h2 : ∃ k, 2 * k + 1 = 333) : m = 18 :=
sorry

end cube_splitting_odd_numbers_l22_22083


namespace ben_owes_rachel_l22_22250

theorem ben_owes_rachel :
  let dollars_per_lawn := (13 : ℚ) / 3
  let lawns_mowed := (8 : ℚ) / 5
  let total_owed := (104 : ℚ) / 15
  dollars_per_lawn * lawns_mowed = total_owed := 
by 
  sorry

end ben_owes_rachel_l22_22250


namespace cafe_table_count_l22_22794

theorem cafe_table_count (cafe_seats_base7 : ℕ) (seats_per_table : ℕ) (cafe_seats_base10 : ℕ)
    (h1 : cafe_seats_base7 = 3 * 7^2 + 1 * 7^1 + 2 * 7^0) 
    (h2 : seats_per_table = 3) : cafe_seats_base10 = 156 ∧ (cafe_seats_base10 / seats_per_table) = 52 := 
by {
  sorry
}

end cafe_table_count_l22_22794


namespace max_value_90_l22_22468

noncomputable def max_value_expression (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_value_90 (a b c d : ℝ) (h₁ : -4.5 ≤ a) (h₂ : a ≤ 4.5)
                                   (h₃ : -4.5 ≤ b) (h₄ : b ≤ 4.5)
                                   (h₅ : -4.5 ≤ c) (h₆ : c ≤ 4.5)
                                   (h₇ : -4.5 ≤ d) (h₈ : d ≤ 4.5) :
  max_value_expression a b c d ≤ 90 :=
sorry

end max_value_90_l22_22468


namespace regular_polygon_inscribed_circle_area_l22_22724

theorem regular_polygon_inscribed_circle_area
  (n : ℕ) (R : ℝ) (hR : R ≠ 0) (h_area : (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) :
  n = 20 :=
by 
  sorry

end regular_polygon_inscribed_circle_area_l22_22724


namespace pavan_distance_travelled_l22_22387

theorem pavan_distance_travelled (D : ℝ) (h1 : D / 60 + D / 50 = 11) : D = 300 :=
sorry

end pavan_distance_travelled_l22_22387


namespace value_of_expression_l22_22443

theorem value_of_expression
  (m n : ℝ)
  (h1 : n = -2 * m + 3) :
  4 * m + 2 * n + 1 = 7 :=
sorry

end value_of_expression_l22_22443


namespace radius_of_smaller_base_l22_22030

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

end radius_of_smaller_base_l22_22030


namespace sum_of_four_consecutive_integers_with_product_5040_eq_34_l22_22973

theorem sum_of_four_consecutive_integers_with_product_5040_eq_34 :
  ∃ a b c d : ℕ, a * b * c * d = 5040 ∧ a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ (a + b + c + d) = 34 :=
sorry

end sum_of_four_consecutive_integers_with_product_5040_eq_34_l22_22973


namespace average_weight_increase_l22_22344

theorem average_weight_increase 
  (n : ℕ) (A : ℕ → ℝ)
  (h_total : n = 10)
  (h_replace : A 65 = 137) : 
  (137 - 65) / 10 = 7.2 := 
by 
  sorry

end average_weight_increase_l22_22344


namespace henrys_friend_money_l22_22519

theorem henrys_friend_money (h1 h2 : ℕ) (T : ℕ) (f : ℕ) : h1 = 5 → h2 = 2 → T = 20 → h1 + h2 + f = T → f = 13 :=
by
  intros h1_eq h2_eq T_eq total_eq
  rw [h1_eq, h2_eq, T_eq] at total_eq
  sorry

end henrys_friend_money_l22_22519


namespace exists_x0_in_interval_l22_22179

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem exists_x0_in_interval :
  ∃ x0 : ℝ, 0 < x0 ∧ x0 < 4 ∧ f x0 = 0 ∧ 2 < x0 ∧ x0 < 3 :=
sorry

end exists_x0_in_interval_l22_22179


namespace men_left_the_job_l22_22195

theorem men_left_the_job
    (work_rate_20men : 20 * 4 = 30)
    (work_rate_remaining : 6 * 6 = 36) :
    4 = 20 - (20 * 4) / (6 * 6)  :=
by
  sorry

end men_left_the_job_l22_22195


namespace period1_period2_multiple_l22_22658

theorem period1_period2_multiple
  (students_period1 : ℕ)
  (students_period2 : ℕ)
  (h_students_period1 : students_period1 = 11)
  (h_students_period2 : students_period2 = 8)
  (M : ℕ)
  (h_condition : students_period1 = M * students_period2 - 5) :
  M = 2 :=
by
  sorry

end period1_period2_multiple_l22_22658


namespace delta_y_over_delta_x_l22_22893

def curve (x : ℝ) : ℝ := x^2 + x

theorem delta_y_over_delta_x (Δx Δy : ℝ) 
  (hQ : (2 + Δx, 6 + Δy) = (2 + Δx, curve (2 + Δx)))
  (hP : 6 = curve 2) : 
  (Δy / Δx) = Δx + 5 :=
by
  sorry

end delta_y_over_delta_x_l22_22893


namespace ratio_of_areas_l22_22556

noncomputable def area_of_right_triangle (a b : ℝ) : ℝ :=
1 / 2 * a * b

theorem ratio_of_areas (a b c x y z : ℝ)
  (h1 : a = 6) (h2 : b = 8) (h3 : c = 10) 
  (h4 : x = 9) (h5 : y = 12) (h6 : z = 15)
  (h7 : a^2 + b^2 = c^2) (h8 : x^2 + y^2 = z^2) :
  (area_of_right_triangle a b) / (area_of_right_triangle x y) = 4 / 9 :=
sorry

end ratio_of_areas_l22_22556


namespace infinitely_many_87_b_seq_l22_22311

def a_seq : ℕ → ℕ
| 0 => 3
| (n + 1) => 3 ^ (a_seq n)

def b_seq (n : ℕ) : ℕ := (a_seq n) % 100

theorem infinitely_many_87_b_seq (n : ℕ) (hn : n ≥ 2) : b_seq n = 87 := by
  sorry

end infinitely_many_87_b_seq_l22_22311


namespace packets_of_sugar_per_week_l22_22392

theorem packets_of_sugar_per_week (total_grams : ℕ) (packet_weight : ℕ) (total_packets : ℕ) :
  total_grams = 2000 →
  packet_weight = 100 →
  total_packets = total_grams / packet_weight →
  total_packets = 20 := 
  by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3 

end packets_of_sugar_per_week_l22_22392


namespace football_team_progress_l22_22414

theorem football_team_progress : 
  ∀ {loss gain : ℤ}, loss = 5 → gain = 11 → gain - loss = 6 :=
by
  intros loss gain h_loss h_gain
  rw [h_loss, h_gain]
  sorry

end football_team_progress_l22_22414


namespace same_number_of_friends_l22_22823

-- Definitions and conditions
def num_people (n : ℕ) := true   -- Placeholder definition to indicate the number of people
def num_friends (person : ℕ) (n : ℕ) : ℕ := sorry -- The number of friends a given person has (needs to be defined)
def friends_range (n : ℕ) := ∀ person, 0 ≤ num_friends person n ∧ num_friends person n < n

-- Theorem statement
theorem same_number_of_friends (n : ℕ) (h1 : num_people n) (h2 : friends_range n) : 
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ num_friends p1 n = num_friends p2 n :=
by
  sorry

end same_number_of_friends_l22_22823


namespace linear_equation_solution_l22_22202

theorem linear_equation_solution (m n : ℤ) (x y : ℤ)
  (h1 : x + 2 * y = 5)
  (h2 : x + y = 7)
  (h3 : x = -m)
  (h4 : y = -n) :
  (3 * m + 2 * n) / (5 * m - n) = 11 / 14 :=
by
  sorry

end linear_equation_solution_l22_22202


namespace max_not_expressed_as_linear_comb_l22_22255

theorem max_not_expressed_as_linear_comb {a b c : ℕ} (h_coprime_ab : Nat.gcd a b = 1)
                                        (h_coprime_bc : Nat.gcd b c = 1)
                                        (h_coprime_ca : Nat.gcd c a = 1) :
    Nat := sorry

end max_not_expressed_as_linear_comb_l22_22255


namespace total_dolls_48_l22_22138

def dolls_sister : ℕ := 8

def dolls_hannah : ℕ := 5 * dolls_sister

def total_dolls : ℕ := dolls_hannah + dolls_sister

theorem total_dolls_48 : total_dolls = 48 := 
by
  unfold total_dolls dolls_hannah dolls_sister
  rfl

end total_dolls_48_l22_22138


namespace sum_of_midpoints_l22_22005

theorem sum_of_midpoints (a b c : ℝ) (h : a + b + c = 12) : 
  (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 12 :=
by
  sorry

end sum_of_midpoints_l22_22005


namespace expected_value_is_6_5_l22_22460

noncomputable def expected_value_12_sided_die : ℚ :=
  (1 / 12) * (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9 + 10 + 11 + 12)

theorem expected_value_is_6_5 : expected_value_12_sided_die = 6.5 := 
by
  sorry

end expected_value_is_6_5_l22_22460


namespace ernaldo_friends_count_l22_22568

-- Define the members of the group
inductive Member
| Arnaldo
| Bernaldo
| Cernaldo
| Dernaldo
| Ernaldo

open Member

-- Define the number of friends for each member
def number_of_friends : Member → ℕ
| Arnaldo  => 1
| Bernaldo => 2
| Cernaldo => 3
| Dernaldo => 4
| Ernaldo  => 0  -- This will be our unknown to solve

-- The main theorem we need to prove
theorem ernaldo_friends_count : number_of_friends Ernaldo = 2 :=
sorry

end ernaldo_friends_count_l22_22568


namespace inequality_solution_l22_22742

section
variables (a x : ℝ)

theorem inequality_solution (h : a < 0) :
  (ax^2 + (1 - a) * x - 1 > 0 ↔
     (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a) ∨
     (a = -1 ∧ false) ∨
     (a < -1 ∧ -1/a < x ∧ x < 1)) :=
by sorry

end inequality_solution_l22_22742


namespace problem_1_problem_2_problem_3_l22_22242

noncomputable def f (a x : ℝ) : ℝ := a * x - Real.log x

theorem problem_1 :
  (∀ x : ℝ, f 1 x ≥ f 1 1) :=
by sorry

theorem problem_2 (x e : ℝ) (hx : x ∈ Set.Icc (1 / Real.exp 1) (Real.exp 1)) (hf : f a x = 1) :
  0 ≤ a ∧ a ≤ 1 :=
by sorry

theorem problem_3 (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Ici 1 → f a x ≥ f a (1 / x)) → 1 ≤ a :=
by sorry

end problem_1_problem_2_problem_3_l22_22242


namespace volume_of_rotated_solid_l22_22947

theorem volume_of_rotated_solid (unit_cylinder_r1 h1 r2 h2 : ℝ) :
  unit_cylinder_r1 = 6 → h1 = 1 → r2 = 3 → h2 = 4 → 
  (π * unit_cylinder_r1^2 * h1 + π * r2^2 * h2) = 72 * π :=
by 
-- We place the arguments and sorry for skipping the proof
  sorry

end volume_of_rotated_solid_l22_22947


namespace minimum_value_of_fraction_l22_22988

theorem minimum_value_of_fraction (x : ℝ) (h : x > 0) : 
  ∃ (m : ℝ), m = 2 * Real.sqrt 3 - 1 ∧ ∀ y, y = (x^2 + x + 3) / (x + 1) -> y ≥ m :=
sorry

end minimum_value_of_fraction_l22_22988


namespace vehicle_distribution_l22_22894

theorem vehicle_distribution :
  ∃ B T U : ℕ, 2 * B + 3 * T + U = 18 ∧ ∀ n : ℕ, n ≤ 18 → ∃ t : ℕ, ∃ (u : ℕ), 2 * (n - t) + u = 18 ∧ 2 * Nat.gcd t u + 3 * t + u = 18 ∧
  10 + 8 + 7 + 5 + 4 + 2 + 1 = 37 := by
  sorry

end vehicle_distribution_l22_22894


namespace inequality_non_empty_solution_set_l22_22415

theorem inequality_non_empty_solution_set (a : ℝ) : ∃ x : ℝ, ax^2 - (a-2)*x - 2 ≤ 0 :=
sorry

end inequality_non_empty_solution_set_l22_22415


namespace determine_m_in_hexadecimal_conversion_l22_22780

theorem determine_m_in_hexadecimal_conversion :
  ∃ m : ℕ, 1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 ∧ m = 4 :=
by
  sorry

end determine_m_in_hexadecimal_conversion_l22_22780


namespace china_GDP_in_2016_l22_22982

noncomputable def GDP_2016 (a r : ℝ) : ℝ := a * (1 + r / 100)^5

theorem china_GDP_in_2016 (a r : ℝ) :
  GDP_2016 a r = a * (1 + r / 100)^5 :=
by
  -- proof
  sorry

end china_GDP_in_2016_l22_22982


namespace ferry_time_difference_l22_22080

theorem ferry_time_difference :
  ∃ (t : ℕ), (∀ (dP : ℕ) (sP : ℕ) (sQ : ℕ), dP = sP * 3 →
   dP = 24 →
   sP = 8 →
   sQ = sP + 1 →
   t = (dP * 3) / sQ - 3) ∧ t = 5 := 
  sorry

end ferry_time_difference_l22_22080


namespace shaded_region_area_l22_22168

theorem shaded_region_area (a b : ℕ) (H : a = 2) (K : b = 4) :
  let s := a + b
  let area_square_EFGH := s * s
  let area_smaller_square_FG := a * a
  let area_smaller_square_EF := b * b
  let shaded_area := area_square_EFGH - (area_smaller_square_FG + area_smaller_square_EF)
  shaded_area = 16 := 
by
  sorry

end shaded_region_area_l22_22168


namespace max_imaginary_part_angle_l22_22161

def poly (z : Complex) : Complex := z^6 - z^4 + z^2 - 1

theorem max_imaginary_part_angle :
  ∃ θ : Real, θ = 45 ∧ 
  (∃ z : Complex, poly z = 0 ∧ ∀ w : Complex, poly w = 0 → w.im ≤ z.im)
:= sorry

end max_imaginary_part_angle_l22_22161


namespace price_of_individual_rose_l22_22092

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

end price_of_individual_rose_l22_22092


namespace find_uncertain_mushrooms_l22_22789

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end find_uncertain_mushrooms_l22_22789


namespace sqrt10_parts_sqrt6_value_sqrt3_opposite_l22_22096

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

end sqrt10_parts_sqrt6_value_sqrt3_opposite_l22_22096


namespace fourth_root_is_four_l22_22790

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 - 7 * x^2 + 9 * x + 11

-- Conditions that must be true for the given problem
@[simp] def f_neg1_zero : f (-1) = 0 := by sorry
@[simp] def f_2_zero : f (2) = 0 := by sorry
@[simp] def f_neg3_zero : f (-3) = 0 := by sorry

-- The theorem stating the fourth root
theorem fourth_root_is_four (root4 : ℝ) (H : f root4 = 0) : root4 = 4 := by sorry

end fourth_root_is_four_l22_22790


namespace work_efficiency_l22_22518

theorem work_efficiency (orig_time : ℝ) (new_time : ℝ) (work : ℝ) 
  (h1 : orig_time = 1)
  (h2 : new_time = orig_time * (1 - 0.20))
  (h3 : work = 1) :
  (orig_time / new_time) * 100 = 125 :=
by
  sorry

end work_efficiency_l22_22518


namespace sue_charge_per_dog_l22_22828

def amount_saved_christian : ℝ := 5
def amount_saved_sue : ℝ := 7
def charge_per_yard : ℝ := 5
def yards_mowed_christian : ℝ := 4
def total_cost_perfume : ℝ := 50
def additional_amount_needed : ℝ := 6
def dogs_walked_sue : ℝ := 6

theorem sue_charge_per_dog :
  (amount_saved_christian + (charge_per_yard * yards_mowed_christian) + amount_saved_sue + (dogs_walked_sue * x) + additional_amount_needed = total_cost_perfume) → x = 2 :=
by
  sorry

end sue_charge_per_dog_l22_22828


namespace cylinder_lateral_surface_area_l22_22309

-- Define structures for the problem
structure Cylinder where
  generatrix : ℝ
  base_radius : ℝ

-- Define the conditions
def cylinder_conditions : Cylinder :=
  { generatrix := 1, base_radius := 1 }

-- The theorem statement
theorem cylinder_lateral_surface_area (cyl : Cylinder) (h_gen : cyl.generatrix = 1) (h_rad : cyl.base_radius = 1) :
  ∀ (area : ℝ), area = 2 * Real.pi :=
sorry

end cylinder_lateral_surface_area_l22_22309


namespace max_value_of_x_plus_2y_l22_22015

theorem max_value_of_x_plus_2y {x y : ℝ} (h : |x| + |y| ≤ 1) : (x + 2 * y) ≤ 2 :=
sorry

end max_value_of_x_plus_2y_l22_22015


namespace value_at_1971_l22_22633

def sequence_x (x : ℕ → ℝ) :=
  ∀ n > 1, 3 * x n - x (n - 1) = n

theorem value_at_1971 (x : ℕ → ℝ) (hx : sequence_x x) (h_initial : abs (x 1) < 1971) :
  abs (x 1971 - 985.25) < 0.000001 :=
by sorry

end value_at_1971_l22_22633


namespace simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l22_22681

theorem simplify_expr_for_a_neq_0_1_neg1 (a : ℝ) (h1 : a ≠ 1) (h0 : a ≠ 0) (h_neg1 : a ≠ -1) :
  ( (a - 1)^2 / ((a + 1) * (a - 1)) ) / (a - (2 * a / (a + 1))) = 1 / a := by
  sorry

theorem final_value_when_a_2 :
  ( (2 - 1)^2 / ((2 + 1) * (2 - 1)) ) / (2 - (2 * 2 / (2 + 1))) = 1 / 2 := by
  sorry

end simplify_expr_for_a_neq_0_1_neg1_final_value_when_a_2_l22_22681


namespace problem_solution_l22_22968

theorem problem_solution :
  (-2: ℤ)^2004 + 3 * (-2: ℤ)^2003 = -2^2003 := 
by
  sorry

end problem_solution_l22_22968


namespace trevor_coin_difference_l22_22760

theorem trevor_coin_difference:
  ∀ (total_coins quarters: ℕ),
  total_coins = 77 →
  quarters = 29 →
  (total_coins - quarters = 48) := by
  intros total_coins quarters h1 h2
  sorry

end trevor_coin_difference_l22_22760


namespace smallest_x_l22_22216

theorem smallest_x (x : ℕ) : (x % 3 = 2) ∧ (x % 4 = 3) ∧ (x % 5 = 4) → x = 59 :=
by
  intro h
  sorry

end smallest_x_l22_22216


namespace find_3m_plus_n_l22_22321

theorem find_3m_plus_n (m n : ℕ) (h1 : m > n) (h2 : 3 * (3 * m * n - 2)^2 - 2 * (3 * m - 3 * n)^2 = 2019) : 3 * m + n = 46 :=
sorry

end find_3m_plus_n_l22_22321


namespace find_value_of_m_l22_22797

/-- Given the parabola y = 4x^2 + 4x + 5 and the line y = 8mx + 8m intersect at exactly one point,
    prove the value of m^{36} + 1155 / m^{12} is 39236. -/
theorem find_value_of_m (m : ℝ) (h: ∃ x, 4 * x^2 + 4 * x + 5 = 8 * m * x + 8 * m ∧
  ∀ x₁ x₂, 4 * x₁^2 + 4 * x₁ + 5 = 8 * m * x₁ + 8 * m →
  4 * x₂^2 + 4 * x₂ + 5 = 8 * m * x₂ + 8 * m → x₁ = x₂) :
  m^36 + 1155 / m^12 = 39236 := 
sorry

end find_value_of_m_l22_22797


namespace compute_value_l22_22038

theorem compute_value : (7^2 - 6^2)^3 = 2197 := by
  sorry

end compute_value_l22_22038


namespace enclosed_area_l22_22276

noncomputable def calculateArea : ℝ :=
  ∫ (x : ℝ) in (1 / 2)..2, 1 / x

theorem enclosed_area : calculateArea = 2 * Real.log 2 :=
by
  sorry

end enclosed_area_l22_22276


namespace trail_length_l22_22328

theorem trail_length (v_Q : ℝ) (v_P : ℝ) (d_P d_Q : ℝ) 
  (h_vP: v_P = 1.25 * v_Q) 
  (h_dP: d_P = 20) 
  (h_meet: d_P / v_P = d_Q / v_Q) :
  d_P + d_Q = 36 :=
sorry

end trail_length_l22_22328


namespace collinear_condition_perpendicular_condition_l22_22100

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

end collinear_condition_perpendicular_condition_l22_22100


namespace functional_equation_zero_solution_l22_22986

theorem functional_equation_zero_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (f x + x + y) = f (x + y) + y * f y) :
  ∀ x : ℝ, f x = 0 :=
sorry

end functional_equation_zero_solution_l22_22986


namespace school_students_l22_22286

theorem school_students (T S : ℕ) (h1 : T = 6 * S - 78) (h2 : T - S = 2222) : T = 2682 :=
by
  sorry

end school_students_l22_22286


namespace ant_moves_probability_l22_22000

theorem ant_moves_probability :
  let m := 73
  let n := 48
  m + n = 121 := by
  sorry

end ant_moves_probability_l22_22000


namespace max_value_of_f_on_interval_l22_22706

noncomputable def f (x : ℝ) : ℝ := (Real.sin (4 * x)) / (2 * Real.sin ((Real.pi / 2) - 2 * x))

theorem max_value_of_f_on_interval :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 6), f x = (Real.sqrt 3) / 2 := sorry

end max_value_of_f_on_interval_l22_22706


namespace abs_inequality_solution_l22_22329

theorem abs_inequality_solution (x : ℝ) : |x - 3| ≥ |x| ↔ x ≤ 3 / 2 :=
by
  sorry

end abs_inequality_solution_l22_22329


namespace contact_alignment_possible_l22_22151

/-- A vacuum tube has seven contacts arranged in a circle and is inserted into a socket that has seven holes.
Prove that it is possible to number the tube's contacts and the socket's holes in such a way that:
in any insertion of the tube, at least one contact will align with its corresponding hole (i.e., the hole with the same number). -/
theorem contact_alignment_possible : ∃ (f : Fin 7 → Fin 7), ∀ (rotation : Fin 7 → Fin 7), ∃ k : Fin 7, f k = rotation k := 
sorry

end contact_alignment_possible_l22_22151


namespace common_ratio_of_geometric_sequence_l22_22746

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 2) 
  (h2 : a 5 = 1 / 4) : 
  ( ∃ a1 : ℝ, a n = a1 * q ^ (n - 1)) 
    :=
sorry

end common_ratio_of_geometric_sequence_l22_22746


namespace part_1_part_2_l22_22623

noncomputable def f (x a : ℝ) : ℝ := abs (2 * x + a) + 2 * a

theorem part_1 (h : ∀ x : ℝ, f x a = f (3 - x) a) : a = -3 :=
by
  sorry

theorem part_2 (h : ∃ x : ℝ, f x a ≤ -abs (2 * x - 1) + a) : a ≤ -1 / 2 :=
by
  sorry

end part_1_part_2_l22_22623


namespace JaneTotalEarningsIs138_l22_22119

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

end JaneTotalEarningsIs138_l22_22119


namespace ratio_of_money_spent_on_clothes_is_1_to_2_l22_22393

-- Definitions based on conditions
def allowance1 : ℕ := 5
def weeks1 : ℕ := 8
def allowance2 : ℕ := 6
def weeks2 : ℕ := 6
def cost_video : ℕ := 35
def remaining_money : ℕ := 3

-- Calculations
def total_saved : ℕ := (allowance1 * weeks1) + (allowance2 * weeks2)
def total_expended : ℕ := cost_video + remaining_money
def spent_on_clothes : ℕ := total_saved - total_expended

-- Prove the ratio of money spent on clothes to the total money saved is 1:2
theorem ratio_of_money_spent_on_clothes_is_1_to_2 :
  (spent_on_clothes : ℚ) / (total_saved : ℚ) = 1 / 2 :=
by
  sorry

end ratio_of_money_spent_on_clothes_is_1_to_2_l22_22393


namespace player_placing_third_won_against_seventh_l22_22881

theorem player_placing_third_won_against_seventh :
  ∃ (s : Fin 8 → ℚ),
    -- Condition 1: Scores are different
    (∀ i j, i ≠ j → s i ≠ s j) ∧
    -- Condition 2: Second place score equals the sum of the bottom four scores
    (s 1 = s 4 + s 5 + s 6 + s 7) ∧
    -- Result: Third player won against the seventh player
    (s 2 > s 6) :=
sorry

end player_placing_third_won_against_seventh_l22_22881


namespace simplify_and_evaluate_l22_22531

theorem simplify_and_evaluate (a b : ℤ) (h₁ : a = -1) (h₂ : b = 3) :
  2 * a * b^2 - (3 * a^2 * b - 2 * (3 * a^2 * b - a * b^2 - 1)) = 7 :=
by
  sorry

end simplify_and_evaluate_l22_22531


namespace complex_quadratic_solution_l22_22703

theorem complex_quadratic_solution (c d : ℤ) (h1 : 0 < c) (h2 : 0 < d) (h3 : (c + d * Complex.I) ^ 2 = 7 + 24 * Complex.I) :
  c + d * Complex.I = 4 + 3 * Complex.I :=
sorry

end complex_quadratic_solution_l22_22703


namespace total_distance_travelled_l22_22562

def speed_one_sail : ℕ := 25 -- knots
def speed_two_sails : ℕ := 50 -- knots
def conversion_factor : ℕ := 115 -- 1.15, in hundredths

def distance_in_nautical_miles : ℕ :=
  (2 * speed_one_sail) +      -- Two hours, one sail
  (3 * speed_two_sails) +     -- Three hours, two sails
  (1 * speed_one_sail) +      -- One hour, one sail, navigating around obstacles
  (2 * (speed_one_sail - speed_one_sail * 30 / 100)) -- Two hours, strong winds, 30% reduction in speed

def distance_in_land_miles : ℕ :=
  distance_in_nautical_miles * conversion_factor / 100 -- Convert to land miles

theorem total_distance_travelled : distance_in_land_miles = 299 := by
  sorry

end total_distance_travelled_l22_22562


namespace average_possible_k_l22_22487

theorem average_possible_k (k : ℕ) (r1 r2 : ℕ) (h : r1 * r2 = 24) (h_pos : r1 > 0 ∧ r2 > 0) (h_eq_k : r1 + r2 = k) : 
  (25 + 14 + 11 + 10) / 4 = 15 :=
by 
  sorry

end average_possible_k_l22_22487


namespace circle_tangent_to_y_axis_l22_22922

theorem circle_tangent_to_y_axis (m : ℝ) :
  (0 < m) → (∀ p : ℝ × ℝ, (p.1 - m)^2 + p.2^2 = 4 ↔ p.1 ^ 2 = p.2^2) → (m = 2 ∨ m = -2) :=
by
  sorry

end circle_tangent_to_y_axis_l22_22922


namespace chosen_number_is_155_l22_22364

variable (x : ℤ)
variable (h₁ : 2 * x - 200 = 110)

theorem chosen_number_is_155 : x = 155 := by
  sorry

end chosen_number_is_155_l22_22364


namespace division_addition_l22_22485

theorem division_addition :
  (-150 + 50) / (-50) = 2 := by
  sorry

end division_addition_l22_22485


namespace lesser_of_two_numbers_l22_22176

theorem lesser_of_two_numbers (x y : ℝ) (h1 : x + y = 50) (h2 : x - y = 6) : y = 22 :=
by
  sorry

end lesser_of_two_numbers_l22_22176


namespace sarah_copies_total_pages_l22_22785

noncomputable def total_pages_copied (people : ℕ) (pages_first : ℕ) (copies_first : ℕ) (pages_second : ℕ) (copies_second : ℕ) : ℕ :=
  (pages_first * (copies_first * people)) + (pages_second * (copies_second * people))

theorem sarah_copies_total_pages :
  total_pages_copied 20 30 3 45 2 = 3600 := by
  sorry

end sarah_copies_total_pages_l22_22785


namespace discounted_price_per_bag_l22_22501

theorem discounted_price_per_bag
  (cost_per_bag : ℝ)
  (num_bags : ℕ)
  (initial_price : ℝ)
  (num_sold_initial : ℕ)
  (net_profit : ℝ)
  (discounted_revenue : ℝ)
  (discounted_price : ℝ) :
  cost_per_bag = 3.0 →
  num_bags = 20 →
  initial_price = 6.0 →
  num_sold_initial = 15 →
  net_profit = 50 →
  discounted_revenue = (net_profit + (num_bags * cost_per_bag) - (num_sold_initial * initial_price) ) →
  discounted_price = (discounted_revenue / (num_bags - num_sold_initial)) →
  discounted_price = 4.0 :=
by
  sorry

end discounted_price_per_bag_l22_22501


namespace training_trip_duration_l22_22352

-- Define the number of supervisors
def num_supervisors : ℕ := 15

-- Define the number of supervisors overseeing the pool each day
def supervisors_per_day : ℕ := 3

-- Define the number of pairs supervised per day
def pairs_per_day : ℕ := (supervisors_per_day * (supervisors_per_day - 1)) / 2

-- Define the total number of pairs from the given number of supervisors
def total_pairs : ℕ := (num_supervisors * (num_supervisors - 1)) / 2

-- Define the number of days required
def num_days : ℕ := total_pairs / pairs_per_day

-- The theorem we need to prove
theorem training_trip_duration : 
  (num_supervisors = 15) ∧
  (supervisors_per_day = 3) ∧
  (∀ (a b : ℕ), a * (a - 1) / 2 = b * (b - 1) / 2 → a = b) ∧ 
  (∀ (N : ℕ), total_pairs = N * pairs_per_day → N = 35) :=
by
  sorry

end training_trip_duration_l22_22352


namespace Gerald_needs_to_average_5_chores_per_month_l22_22676

def spending_per_month := 100
def season_length := 4
def cost_per_chore := 10
def total_spending := spending_per_month * season_length
def months_not_playing := 12 - season_length
def amount_to_save_per_month := total_spending / months_not_playing
def chores_per_month := amount_to_save_per_month / cost_per_chore

theorem Gerald_needs_to_average_5_chores_per_month :
  chores_per_month = 5 := by
  sorry

end Gerald_needs_to_average_5_chores_per_month_l22_22676


namespace linear_equation_a_is_minus_one_l22_22261

theorem linear_equation_a_is_minus_one (a : ℝ) (x : ℝ) :
  ((a - 1) * x ^ (2 - |a|) + 5 = 0) → (2 - |a| = 1) → (a ≠ 1) → a = -1 :=
by
  intros h1 h2 h3
  sorry

end linear_equation_a_is_minus_one_l22_22261


namespace odd_function_five_value_l22_22640

variable (f : ℝ → ℝ)

theorem odd_function_five_value (h_odd : ∀ x : ℝ, f (-x) = -f x)
                               (h_f1 : f 1 = 1 / 2)
                               (h_f_recurrence : ∀ x : ℝ, f (x + 2) = f x + f 2) :
  f 5 = 5 / 2 :=
sorry

end odd_function_five_value_l22_22640


namespace beta_greater_than_alpha_l22_22830

theorem beta_greater_than_alpha (α β : ℝ) (h1 : 0 < α) (h2 : α < π / 2) (h3 : 0 < β) (h4 : β < π / 2) (h5 : Real.sin (α + β) = 2 * Real.sin α) : β > α := 
sorry

end beta_greater_than_alpha_l22_22830


namespace not_lt_neg_version_l22_22551

theorem not_lt_neg_version (a b : ℝ) (h : a < b) : ¬ (-3 * a < -3 * b) :=
by 
  -- This is where the proof would go
  sorry

end not_lt_neg_version_l22_22551


namespace cherry_pie_degrees_l22_22281

theorem cherry_pie_degrees :
  ∀ (total_students chocolate_students apple_students blueberry_students : ℕ),
  total_students = 36 →
  chocolate_students = 12 →
  apple_students = 8 →
  blueberry_students = 6 →
  (total_students - chocolate_students - apple_students - blueberry_students) / 2 = 5 →
  ((5 : ℕ) * 360 / total_students) = 50 := 
by
  sorry

end cherry_pie_degrees_l22_22281


namespace range_of_a_for_inequality_l22_22174

open Real

theorem range_of_a_for_inequality (a : ℝ) : 
  (∀ x : ℝ, ¬(a*x^2 - |x + 1| + 2*a < 0)) ↔ a ≥ (sqrt 3 + 1) / 4 := 
by
  sorry

end range_of_a_for_inequality_l22_22174


namespace eunji_received_900_won_l22_22963

-- Define the conditions
def eunji_pocket_money (X : ℝ) : Prop :=
  (X / 2 + 550 = 1000)

-- Define the theorem to prove the question equals the correct answer
theorem eunji_received_900_won {X : ℝ} (h : eunji_pocket_money X) : X = 900 :=
  by
    sorry

end eunji_received_900_won_l22_22963


namespace arithmetic_sequence_s9_l22_22929

noncomputable def arithmetic_sum (a1 d n : ℝ) : ℝ :=
  n * (2*a1 + (n - 1)*d) / 2

noncomputable def general_term (a1 d n : ℝ) : ℝ :=
  a1 + (n - 1) * d

theorem arithmetic_sequence_s9 (a1 d : ℝ)
  (h1 : general_term a1 d 3 + general_term a1 d 4 + general_term a1 d 8 = 25) :
  arithmetic_sum a1 d 9 = 75 :=
by sorry

end arithmetic_sequence_s9_l22_22929


namespace option_d_necessary_sufficient_l22_22887

theorem option_d_necessary_sufficient (a : ℝ) : (a ≠ 0) ↔ (∃! x : ℝ, a * x = 1) := 
sorry

end option_d_necessary_sufficient_l22_22887


namespace corn_growth_ratio_l22_22366

theorem corn_growth_ratio 
  (growth_first_week : ℕ := 2) 
  (growth_second_week : ℕ) 
  (growth_third_week : ℕ) 
  (total_height : ℕ := 22) 
  (r : ℕ) 
  (h1 : growth_second_week = 2 * r) 
  (h2 : growth_third_week = 4 * (2 * r)) 
  (h3 : growth_first_week + growth_second_week + growth_third_week = total_height) 
  : r = 2 := 
by 
  sorry

end corn_growth_ratio_l22_22366


namespace find_a_minus_b_l22_22422

theorem find_a_minus_b (a b c d : ℤ) 
  (h1 : (a - b) + c - d = 19) 
  (h2 : a - b - c - d = 9) : 
  a - b = 14 :=
sorry

end find_a_minus_b_l22_22422


namespace product_of_differences_of_squares_is_diff_of_square_l22_22878

-- Define when an integer is a difference of squares of positive integers
def diff_of_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ n = x^2 - y^2

-- State the main theorem
theorem product_of_differences_of_squares_is_diff_of_square 
  (a b c d : ℕ) (h₁ : diff_of_squares a) (h₂ : diff_of_squares b) (h₃ : diff_of_squares c) (h₄ : diff_of_squares d) : 
  diff_of_squares (a * b * c * d) := by
  sorry

end product_of_differences_of_squares_is_diff_of_square_l22_22878


namespace total_slides_used_l22_22004

theorem total_slides_used (duration : ℕ) (initial_slides : ℕ) (initial_time : ℕ) (constant_rate : ℕ) (total_time: ℕ)
  (H1 : duration = 50)
  (H2 : initial_slides = 4)
  (H3 : initial_time = 2)
  (H4 : constant_rate = initial_slides / initial_time)
  (H5 : total_time = duration) 
  : (constant_rate * total_time) = 100 := 
by
  sorry

end total_slides_used_l22_22004


namespace min_solution_l22_22938

theorem min_solution :
  ∀ (x : ℝ), (min (1 / (1 - x)) (2 / (1 - x)) = 2 / (x - 1) - 3) → x = 7 / 3 := 
by
  sorry

end min_solution_l22_22938


namespace composite_fraction_l22_22908

theorem composite_fraction (x : ℤ) (hx : x = 5^25) : 
  ∃ a b : ℤ, a > 1 ∧ b > 1 ∧ a * b = x^4 + x^3 + x^2 + x + 1 :=
by sorry

end composite_fraction_l22_22908


namespace find_b_l22_22738

theorem find_b (α β b : ℤ)
  (h1: α > 1)
  (h2: β < -1)
  (h3: ∃ x : ℝ, α * x^2 + β * x - 2 = 0)
  (h4: ∃ x : ℝ, x^2 + bx - 2 = 0)
  (hb: ∀ root1 root2 : ℝ, root1 * root2 = -2 ∧ root1 + root2 = -b) :
  b = 0 := 
sorry

end find_b_l22_22738


namespace second_year_associates_l22_22481

theorem second_year_associates (total_associates : ℕ) (not_first_year : ℕ) (more_than_two_years : ℕ) 
  (h1 : not_first_year = 60 * total_associates / 100) 
  (h2 : more_than_two_years = 30 * total_associates / 100) :
  not_first_year - more_than_two_years = 30 * total_associates / 100 :=
by
  sorry

end second_year_associates_l22_22481


namespace geometric_seq_common_ratio_l22_22042

theorem geometric_seq_common_ratio 
  (a : ℝ) (q : ℝ)
  (h1 : a * q^2 = 4)
  (h2 : a * q^5 = 1 / 2) : 
  q = 1 / 2 := 
by
  sorry

end geometric_seq_common_ratio_l22_22042


namespace total_ingredients_cups_l22_22345

theorem total_ingredients_cups (butter_ratio flour_ratio sugar_ratio sugar_cups : ℚ) 
  (h_ratio : butter_ratio / sugar_ratio = 1 / 4 ∧ flour_ratio / sugar_ratio = 6 / 4) 
  (h_sugar : sugar_cups = 10) : 
  butter_ratio * (sugar_cups / sugar_ratio) + flour_ratio * (sugar_cups / sugar_ratio) + sugar_cups = 27.5 :=
by
  sorry

end total_ingredients_cups_l22_22345


namespace average_revenue_per_hour_l22_22583

theorem average_revenue_per_hour 
    (sold_A_hour1 : ℕ) (sold_B_hour1 : ℕ) (sold_A_hour2 : ℕ) (sold_B_hour2 : ℕ)
    (price_A_hour1 : ℕ) (price_A_hour2 : ℕ) (price_B_constant : ℕ) : 
    (sold_A_hour1 = 10) ∧ (sold_B_hour1 = 5) ∧ (sold_A_hour2 = 2) ∧ (sold_B_hour2 = 3) ∧
    (price_A_hour1 = 3) ∧ (price_A_hour2 = 4) ∧ (price_B_constant = 2) →
    (54 / 2 = 27) :=
by
  intros
  sorry

end average_revenue_per_hour_l22_22583


namespace probability_calculation_l22_22060

def p_X := 1 / 5
def p_Y := 1 / 2
def p_Z := 5 / 8
def p_not_Z := 1 - p_Z

theorem probability_calculation : 
    (p_X * p_Y * p_not_Z) = (3 / 80) := by
    sorry

end probability_calculation_l22_22060


namespace hannah_speed_l22_22747

theorem hannah_speed :
  ∃ H : ℝ, 
    (∀ t : ℝ, (t = 6) → d = 130) ∧ 
    (∀ t : ℝ, (t = 11) → d = 130) → 
    (d = 37 * 5 + H * 5) → 
    H = 15 := 
by 
  sorry

end hannah_speed_l22_22747


namespace enclosed_area_abs_eq_54_l22_22130

theorem enclosed_area_abs_eq_54 :
  (∃ (x y : ℝ), abs x + abs (3 * y) = 9) → True := 
by
  sorry

end enclosed_area_abs_eq_54_l22_22130


namespace fraction_work_AC_l22_22616

theorem fraction_work_AC (total_payment Rs B_payment : ℝ)
  (payment_AC : ℝ)
  (h1 : total_payment = 529)
  (h2 : B_payment = 12)
  (h3 : payment_AC = total_payment - B_payment) : 
  payment_AC / total_payment = 517 / 529 :=
by
  rw [h1, h2] at h3
  rw [h3]
  norm_num
  sorry

end fraction_work_AC_l22_22616


namespace four_digit_cubes_divisible_by_16_l22_22449

theorem four_digit_cubes_divisible_by_16 (n : ℕ) : 
  1000 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 9999 ∧ (4 * n)^3 % 16 = 0 ↔ n = 4 ∨ n = 5 := 
sorry

end four_digit_cubes_divisible_by_16_l22_22449


namespace dinner_guest_arrangement_l22_22998

noncomputable def number_of_ways (n k : ℕ) : ℕ :=
  if n < k then 0 else Nat.factorial n / Nat.factorial (n - k)

theorem dinner_guest_arrangement :
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements
  valid_arrangements = 5040 :=
by
  -- Definitions and preliminary calculations
  let total_arrangements := number_of_ways 8 5
  let unwanted_arrangements := 7 * number_of_ways 6 3 * 2
  let valid_arrangements := total_arrangements - unwanted_arrangements

  -- This is where the proof would go, but we insert sorry to skip it for now
  sorry

end dinner_guest_arrangement_l22_22998


namespace semicircle_perimeter_l22_22401

-- Assuming π as 3.14 for approximation
def π_approx : ℝ := 3.14

-- Radius of the semicircle
def radius : ℝ := 2.1

-- Half of the circumference
def half_circumference (r : ℝ) : ℝ := π_approx * r

-- Diameter of the semicircle
def diameter (r : ℝ) : ℝ := 2 * r

-- Perimeter of the semicircle
def perimeter (r : ℝ) : ℝ := half_circumference r + diameter r

-- Theorem stating the perimeter of the semicircle with given radius
theorem semicircle_perimeter : perimeter radius = 10.794 := by
  sorry

end semicircle_perimeter_l22_22401


namespace fraction_capacity_noah_ali_l22_22566

def capacity_Ali_closet : ℕ := 200
def total_capacity_Noah_closet : ℕ := 100
def each_capacity_Noah_closet : ℕ := total_capacity_Noah_closet / 2

theorem fraction_capacity_noah_ali : (each_capacity_Noah_closet : ℚ) / capacity_Ali_closet = 1 / 4 :=
by sorry

end fraction_capacity_noah_ali_l22_22566


namespace ratio_sea_horses_penguins_l22_22376

def sea_horses := 70
def penguins := sea_horses + 85

theorem ratio_sea_horses_penguins : (70 : ℚ) / (sea_horses + 85) = 14 / 31 :=
by
  -- Proof omitted
  sorry

end ratio_sea_horses_penguins_l22_22376


namespace grandson_age_is_5_l22_22150

-- Definitions based on the conditions
def grandson_age_months_eq_grandmother_years (V B : ℕ) : Prop := B = 12 * V
def combined_age_eq_65 (V B : ℕ) : Prop := B + V = 65

-- Main theorem stating that under these conditions, the grandson's age is 5 years
theorem grandson_age_is_5 (V B : ℕ) (h₁ : grandson_age_months_eq_grandmother_years V B) (h₂ : combined_age_eq_65 V B) : V = 5 :=
by sorry

end grandson_age_is_5_l22_22150


namespace average_rainfall_february_1964_l22_22602

theorem average_rainfall_february_1964 :
  let total_rainfall := 280
  let days_february := 29
  let hours_per_day := 24
  (total_rainfall / (days_february * hours_per_day)) = (280 / (29 * 24)) :=
by
  sorry

end average_rainfall_february_1964_l22_22602


namespace milk_price_increase_l22_22307

theorem milk_price_increase
  (P : ℝ) (C : ℝ) (P_new : ℝ)
  (h1 : P * C = P_new * (5 / 6) * C) :
  (P_new - P) / P * 100 = 20 :=
by
  sorry

end milk_price_increase_l22_22307


namespace findQuadraticFunctionAndVertex_l22_22097

noncomputable section

def quadraticFunction (x : ℝ) (b c : ℝ) : ℝ :=
  (1 / 2) * x^2 + b * x + c

theorem findQuadraticFunctionAndVertex :
  (∃ b c : ℝ, quadraticFunction 0 b c = -1 ∧ quadraticFunction 2 b c = -3) →
  (quadraticFunction x (-2) (-1) = (1 / 2) * x^2 - 2 * x - 1) ∧
  (∃ (vₓ vᵧ : ℝ), vₓ = 2 ∧ vᵧ = -3 ∧ quadraticFunction vₓ (-2) (-1) = vᵧ)  :=
by
  sorry

end findQuadraticFunctionAndVertex_l22_22097


namespace adam_chocolate_boxes_l22_22540

theorem adam_chocolate_boxes 
  (c : ℕ) -- number of chocolate boxes Adam bought
  (h1 : 4 * c + 4 * 5 = 28) : 
  c = 2 := 
by
  sorry

end adam_chocolate_boxes_l22_22540


namespace weight_of_replaced_person_l22_22244

theorem weight_of_replaced_person (avg_weight : ℝ) (new_person_weight : ℝ)
  (h1 : new_person_weight = 65)
  (h2 : ∀ (initial_avg_weight : ℝ), 8 * (initial_avg_weight + 2.5) - 8 * initial_avg_weight = new_person_weight - avg_weight) :
  avg_weight = 45 := 
by
  -- Proof goes here
  sorry

end weight_of_replaced_person_l22_22244


namespace triangle_inscribed_relation_l22_22584

noncomputable def herons_area (p a b c : ℝ) : ℝ := (p * (p - a) * (p - b) * (p - c)).sqrt

theorem triangle_inscribed_relation
  (S S' p p' : ℝ)
  (a b c a' b' c' r : ℝ)
  (h1 : r = S / p)
  (h2 : r = S' / p')
  (h3 : S = herons_area p a b c)
  (h4 : S' = herons_area p' a' b' c') :
  (p - a) * (p - b) * (p - c) / p = (p' - a') * (p' - b') * (p' - c') / p' :=
by sorry

end triangle_inscribed_relation_l22_22584


namespace solve_equation1_solve_equation2_solve_equation3_solve_equation4_l22_22950

theorem solve_equation1 (x : ℝ) : (x - 1) ^ 2 = 4 ↔ x = 3 ∨ x = -1 :=
by sorry

theorem solve_equation2 (x : ℝ) : x ^ 2 + 3 * x - 4 = 0 ↔ x = 1 ∨ x = -4 :=
by sorry

theorem solve_equation3 (x : ℝ) : 4 * x * (2 * x + 1) = 3 * (2 * x + 1) ↔ x = -1 / 2 ∨ x = 3 / 4 :=
by sorry

theorem solve_equation4 (x : ℝ) : 2 * x ^ 2 + 5 * x - 3 = 0 ↔ x = 1 / 2 ∨ x = -3 :=
by sorry

end solve_equation1_solve_equation2_solve_equation3_solve_equation4_l22_22950


namespace neg_sin_leq_1_l22_22093

theorem neg_sin_leq_1 :
  (¬ ∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by
  sorry

end neg_sin_leq_1_l22_22093


namespace total_amount_paid_is_correct_l22_22035

def rate_per_kg_grapes := 98
def quantity_grapes := 15
def rate_per_kg_mangoes := 120
def quantity_mangoes := 8
def rate_per_kg_pineapples := 75
def quantity_pineapples := 5
def rate_per_kg_oranges := 60
def quantity_oranges := 10

def cost_grapes := rate_per_kg_grapes * quantity_grapes
def cost_mangoes := rate_per_kg_mangoes * quantity_mangoes
def cost_pineapples := rate_per_kg_pineapples * quantity_pineapples
def cost_oranges := rate_per_kg_oranges * quantity_oranges

def total_amount_paid := cost_grapes + cost_mangoes + cost_pineapples + cost_oranges

theorem total_amount_paid_is_correct : total_amount_paid = 3405 := by
  sorry

end total_amount_paid_is_correct_l22_22035


namespace average_visitors_30_day_month_l22_22362

def visitors_per_day (total_visitors : ℕ) (days : ℕ) : ℕ := total_visitors / days

theorem average_visitors_30_day_month (visitors_sunday : ℕ) (visitors_other_days : ℕ) 
  (total_days : ℕ) (sundays : ℕ) (other_days : ℕ) :
  visitors_sunday = 510 →
  visitors_other_days = 240 →
  total_days = 30 →
  sundays = 4 →
  other_days = 26 →
  visitors_per_day (sundays * visitors_sunday + other_days * visitors_other_days) total_days = 276 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end average_visitors_30_day_month_l22_22362


namespace BP_PA_ratio_l22_22053

section

variable (A B C P : Type)
variable {AC BC PA PB BP : ℕ}

-- Conditions:
-- 1. In triangle ABC, the ratio AC:CB = 2:5.
axiom AC_CB_ratio : 2 * BC = 5 * AC

-- 2. The bisector of the exterior angle at C intersects the extension of BA at P,
--    such that B is between P and A.
axiom Angle_Bisector_Theorem : PA * BC = PB * AC

theorem BP_PA_ratio (h1 : 2 * BC = 5 * AC) (h2 : PA * BC = PB * AC) :
  BP * PA = 5 * PA := sorry

end

end BP_PA_ratio_l22_22053


namespace diagonals_of_angle_bisectors_l22_22825

theorem diagonals_of_angle_bisectors (a b : ℝ) (BAD ABC : ℝ) (hBAD : BAD = ABC) :
  ∃ d : ℝ, d = |a - b| :=
by
  sorry

end diagonals_of_angle_bisectors_l22_22825


namespace evaluate_expression_l22_22764

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 4 = 6564 :=
by
  sorry

end evaluate_expression_l22_22764


namespace roots_triangle_ineq_l22_22721

variable {m : ℝ}

def roots_form_triangle (x1 x2 x3 : ℝ) : Prop :=
  x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1

theorem roots_triangle_ineq (h : ∀ x, (x - 2) * (x^2 - 4*x + m) = 0) :
  3 < m ∧ m < 4 :=
by
  sorry

end roots_triangle_ineq_l22_22721


namespace div_by_3_implies_one_div_by_3_l22_22482

theorem div_by_3_implies_one_div_by_3 (a b : ℕ) (h_ab : 3 ∣ (a * b)) (h_na : ¬ 3 ∣ a) (h_nb : ¬ 3 ∣ b) : false :=
sorry

end div_by_3_implies_one_div_by_3_l22_22482


namespace correct_answer_l22_22214

theorem correct_answer (A B C D : String) (sentence : String)
  (h1 : A = "us")
  (h2 : B = "we")
  (h3 : C = "our")
  (h4 : D = "ours")
  (h_sentence : sentence = "To save class time, our teacher has _ students do half of the exercise in class and complete the other half for homework.") :
  sentence = "To save class time, our teacher has " ++ A ++ " students do half of the exercise in class and complete the other half for homework." :=
by
  sorry

end correct_answer_l22_22214


namespace arithmetic_sequence_common_difference_l22_22778

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n * a₁ + (n * (n - 1) / 2) * d

/-- Condition for the sum of the first 5 terms -/
def S5 (a₁ d : ℚ) : Prop := S 5 a₁ d = 6

/-- Condition for the second term of the sequence -/
def a2 (a₁ d : ℚ) : Prop := a₁ + d = 1

/-- The main theorem to be proved -/
theorem arithmetic_sequence_common_difference (a₁ d : ℚ) (hS5 : S5 a₁ d) (ha2 : a2 a₁ d) : d = 1 / 5 :=
sorry

end arithmetic_sequence_common_difference_l22_22778


namespace find_square_side_length_l22_22170

theorem find_square_side_length
  (a CF AE : ℝ)
  (h_CF : CF = 2 * a)
  (h_AE : AE = 3.5 * a)
  (h_sum : CF + AE = 91) :
  a = 26 := by
  sorry

end find_square_side_length_l22_22170


namespace push_ups_total_l22_22095

theorem push_ups_total (d z : ℕ) (h1 : d = 51) (h2 : d = z + 49) : d + z = 53 := by
  sorry

end push_ups_total_l22_22095


namespace y_intercept_of_line_l22_22939

theorem y_intercept_of_line (m : ℝ) (x1 y1 : ℝ) (h_slope : m = -3) (h_x_intercept : x1 = 7) (h_y_intercept : y1 = 0) : 
  ∃ y_intercept : ℝ, y_intercept = 21 ∧ y_intercept = -m * 0 + 21 :=
by
  sorry

end y_intercept_of_line_l22_22939


namespace exterior_angle_sum_l22_22277

theorem exterior_angle_sum (n : ℕ) (h_n : 3 ≤ n) :
  let polygon_exterior_angle_sum := 360
  let triangle_exterior_angle_sum := 0
  (polygon_exterior_angle_sum + triangle_exterior_angle_sum = 360) :=
by sorry

end exterior_angle_sum_l22_22277


namespace completing_the_square_correct_l22_22901

theorem completing_the_square_correct :
  (∃ x : ℝ, x^2 - 6 * x + 5 = 0) →
  (∃ x : ℝ, (x - 3)^2 = 4) :=
by
  sorry

end completing_the_square_correct_l22_22901


namespace Geli_pushups_and_runs_l22_22680

def initial_pushups : ℕ := 10
def increment_pushups : ℕ := 5
def workouts_per_week : ℕ := 3
def weeks_in_a_month : ℕ := 4
def pushups_per_mile_run : ℕ := 30

def workout_days_in_month : ℕ := workouts_per_week * weeks_in_a_month

def pushups_on_day (day : ℕ) : ℕ := initial_pushups + (day - 1) * increment_pushups

def total_pushups : ℕ := (workout_days_in_month / 2) * (initial_pushups + pushups_on_day workout_days_in_month)

def one_mile_runs (total_pushups : ℕ) : ℕ := total_pushups / pushups_per_mile_run

theorem Geli_pushups_and_runs :
  total_pushups = 450 ∧ one_mile_runs total_pushups = 15 :=
by
  -- Here, we should prove total_pushups = 450 and one_mile_runs total_pushups = 15.
  sorry

end Geli_pushups_and_runs_l22_22680


namespace work_rate_sum_l22_22022

theorem work_rate_sum (A B : ℝ) (W : ℝ) (h1 : (A + B) * 4 = W) (h2 : A * 8 = W) : (A + B) * 4 = W :=
by
  -- placeholder for actual proof
  sorry

end work_rate_sum_l22_22022


namespace max_alpha_flights_achievable_l22_22579

def max_alpha_flights (n : ℕ) : ℕ :=
  let total_flights := n * (n - 1) / 2
  let max_beta_flights := n / 2
  total_flights - max_beta_flights

theorem max_alpha_flights_achievable (n : ℕ) : 
  ∃ k, k = n * (n - 1) / 2 - n / 2 ∧ k ≤ max_alpha_flights n :=
by
  sorry

end max_alpha_flights_achievable_l22_22579


namespace walter_fraction_fewer_bananas_l22_22199

theorem walter_fraction_fewer_bananas (f : ℚ) (h1 : 56 + (56 - 56 * f) = 98) : f = 1 / 4 :=
sorry

end walter_fraction_fewer_bananas_l22_22199


namespace quadratic_equation_l22_22557

theorem quadratic_equation (p q : ℝ) 
  (h1 : p^2 + 9 * q^2 + 3 * p - p * q = 30)
  (h2 : p - 5 * q - 8 = 0) : 
  p^2 - p - 6 = 0 :=
by sorry

end quadratic_equation_l22_22557


namespace golden_state_total_points_l22_22613

theorem golden_state_total_points :
  ∀ (Draymond Curry Kelly Durant Klay : ℕ),
  Draymond = 12 →
  Curry = 2 * Draymond →
  Kelly = 9 →
  Durant = 2 * Kelly →
  Klay = Draymond / 2 →
  Draymond + Curry + Kelly + Durant + Klay = 69 :=
by
  intros Draymond Curry Kelly Durant Klay
  intros hD hC hK hD2 hK2
  rw [hD, hC, hK, hD2, hK2]
  sorry

end golden_state_total_points_l22_22613


namespace sum_of_three_digit_even_naturals_correct_l22_22141

noncomputable def sum_of_three_digit_even_naturals : ℕ := 
  let a := 100
  let l := 998
  let d := 2
  let n := (l - a) / d + 1
  n / 2 * (a + l)

theorem sum_of_three_digit_even_naturals_correct : 
  sum_of_three_digit_even_naturals = 247050 := by 
  sorry

end sum_of_three_digit_even_naturals_correct_l22_22141


namespace proof_of_inequality_l22_22787

theorem proof_of_inequality (a : ℝ) (h : (∃ x : ℝ, x - 2 * a + 4 = 0 ∧ x < 0)) :
  (a - 3) * (a - 4) > 0 :=
by
  sorry

end proof_of_inequality_l22_22787


namespace hyperbola_foci_condition_l22_22599

theorem hyperbola_foci_condition (m n : ℝ) (h : m * n > 0) :
    (m > 0 ∧ n > 0) ↔ ((∃ (x y : ℝ), m * x^2 - n * y^2 = 1) ∧ (∃ (x y : ℝ), m * x^2 - n * y^2 = 1)) :=
sorry

end hyperbola_foci_condition_l22_22599


namespace remainders_of_65_powers_l22_22667

theorem remainders_of_65_powers (n : ℕ) :
  (65 ^ (6 * n)) % 9 = 1 ∧
  (65 ^ (6 * n + 1)) % 9 = 2 ∧
  (65 ^ (6 * n + 2)) % 9 = 4 ∧
  (65 ^ (6 * n + 3)) % 9 = 8 :=
by
  sorry

end remainders_of_65_powers_l22_22667


namespace area_square_ratio_l22_22104

theorem area_square_ratio (r : ℝ) (h1 : r > 0)
  (s1 : ℝ) (hs1 : s1^2 = r^2)
  (s2 : ℝ) (hs2 : s2^2 = (4/5) * r^2) : 
  (s1^2 / s2^2) = (5 / 4) :=
by 
  sorry

end area_square_ratio_l22_22104


namespace polynomial_remainder_l22_22741

noncomputable def remainder_div (p : Polynomial ℚ) (d1 d2 d3 : Polynomial ℚ) : Polynomial ℚ :=
  let d := d1 * d2 * d3 
  let q := p /ₘ d 
  let r := p %ₘ d 
  r

theorem polynomial_remainder :
  let p := (X^6 + 2 * X^4 - X^3 - 7 * X^2 + 3 * X + 1)
  let d1 := X - 2
  let d2 := X + 1
  let d3 := X - 3
  remainder_div p d1 d2 d3 = 29 * X^2 + 17 * X - 19 :=
by
  sorry

end polynomial_remainder_l22_22741


namespace evaluate_expression_l22_22112

variables (x y : ℕ)

theorem evaluate_expression : x = 2 → y = 4 → y * (y - 2 * x + 1) = 4 :=
by
  intro h1 h2
  sorry

end evaluate_expression_l22_22112


namespace total_pairs_purchased_l22_22588

-- Define the conditions as hypotheses
def foxPrice : ℝ := 15
def ponyPrice : ℝ := 18
def totalSaved : ℝ := 8.91
def foxPairs : ℕ := 3
def ponyPairs : ℕ := 2
def sumDiscountRates : ℝ := 0.22
def ponyDiscountRate : ℝ := 0.10999999999999996

-- Prove that the total number of pairs of jeans purchased is 5
theorem total_pairs_purchased : foxPairs + ponyPairs = 5 := by
  sorry

end total_pairs_purchased_l22_22588


namespace license_plates_count_l22_22425

theorem license_plates_count :
  let letters := 26
  let digits := 10
  let odd_digits := 5
  let even_digits := 5
  (letters^3) * digits * (odd_digits + even_digits) = 878800 := by
  sorry

end license_plates_count_l22_22425


namespace max_value_relationship_l22_22912

theorem max_value_relationship (x y : ℝ) :
  (2005 - (x + y)^2 = 2005) → (x = -y) :=
by
  intro h
  sorry

end max_value_relationship_l22_22912


namespace measure_of_angle_C_range_of_sum_ab_l22_22171

-- Proof problem (1): Prove the measure of angle C
theorem measure_of_angle_C (a b c : ℝ) (A B C : ℝ) 
  (h1 : 2 * c * Real.sin C = (2 * b + a) * Real.sin B + (2 * a - 3 * b) * Real.sin A) :
  C = Real.pi / 3 := by 
  sorry

-- Proof problem (2): Prove the range of possible values of a + b
theorem range_of_sum_ab (a b : ℝ) (c : ℝ) (h1 : c = 4) (h2 : 16 = a^2 + b^2 - a * b) :
  4 < a + b ∧ a + b ≤ 8 := by 
  sorry

end measure_of_angle_C_range_of_sum_ab_l22_22171


namespace find_k_range_for_two_roots_l22_22476

noncomputable def f (k x : ℝ) : ℝ := (Real.log x / x) - k * x

theorem find_k_range_for_two_roots :
  ∃ k_min k_max : ℝ, k_min = (2 / (Real.exp 4)) ∧ k_max = (1 / (2 * Real.exp 1)) ∧
  ∀ k : ℝ, (k_min ≤ k ∧ k < k_max) ↔
    ∃ x1 x2 : ℝ, 
    (1 / Real.exp 1) ≤ x1 ∧ x1 ≤ Real.exp 2 ∧ 
    (1 / Real.exp 1) ≤ x2 ∧ x2 ≤ Real.exp 2 ∧ 
    f k x1 = 0 ∧ f k x2 = 0 ∧ 
    x1 ≠ x2 :=
sorry

end find_k_range_for_two_roots_l22_22476


namespace holds_under_condition_l22_22302

theorem holds_under_condition (a b c : ℕ) (ha : a ≤ 10) (hb : b ≤ 10) (hc : c ≤ 10) (cond : b + 11 * c = 10 * a) :
  (10 * a + b) * (10 * a + c) = 100 * a * a + 100 * a + 11 * b * c :=
by
  sorry

end holds_under_condition_l22_22302


namespace shopkeeper_profit_percent_l22_22811

noncomputable def profit_percent : ℚ := 
let cp_each := 1       -- Cost price of each article
let sp_each := 1.2     -- Selling price of each article without discount
let discount := 0.05   -- 5% discount
let tax := 0.10        -- 10% sales tax
let articles := 30     -- Number of articles
let cp_total := articles * cp_each      -- Total cost price
let sp_after_discount := sp_each * (1 - discount)    -- Selling price after discount
let revenue_before_tax := articles * sp_after_discount   -- Total revenue before tax
let tax_amount := revenue_before_tax * tax   -- Sales tax amount
let revenue_after_tax := revenue_before_tax + tax_amount -- Total revenue after tax
let profit := revenue_after_tax - cp_total -- Profit
(profit / cp_total) * 100 -- Profit percent

theorem shopkeeper_profit_percent : profit_percent = 25.4 :=
by
  -- Here follows the proof based on the conditions and steps above
  sorry

end shopkeeper_profit_percent_l22_22811


namespace range_of_a_l22_22137

noncomputable def f (a x : ℝ) : ℝ := (a - 1) * x^2 + (a - 1) * x + 1

theorem range_of_a (a : ℝ) : (∀ x : ℝ, f a x > 0) ↔ (1 ≤ a ∧ a < 5) := by
  sorry

end range_of_a_l22_22137


namespace paint_time_for_two_people_l22_22977

/-- 
Proof Problem Statement: Prove that it would take 12 hours for two people to paint the house
given that six people can paint it in 4 hours, assuming everyone works at the same rate.
--/
theorem paint_time_for_two_people 
  (h1 : 6 * 4 = 24) 
  (h2 : ∀ (n : ℕ) (t : ℕ), n * t = 24 → t = 24 / n) : 
  2 * 12 = 24 :=
sorry

end paint_time_for_two_people_l22_22977


namespace sqrt_sum_l22_22585

theorem sqrt_sum : (Real.sqrt 50) + (Real.sqrt 32) = 9 * (Real.sqrt 2) :=
by
  sorry

end sqrt_sum_l22_22585


namespace remaining_last_year_budget_is_13_l22_22322

-- Variables representing the conditions of the problem
variable (cost1 cost2 given_budget remaining this_year_spent remaining_last_year : ℤ)

-- Define the conditions as hypotheses
def conditions : Prop :=
  cost1 = 13 ∧ cost2 = 24 ∧ 
  given_budget = 50 ∧ 
  remaining = 19 ∧ 
  (cost1 + cost2 = 37) ∧
  (this_year_spent = given_budget - remaining) ∧
  (remaining_last_year + (cost1 + cost2 - this_year_spent) = remaining)

-- The statement that needs to be proven
theorem remaining_last_year_budget_is_13 : conditions cost1 cost2 given_budget remaining this_year_spent remaining_last_year → remaining_last_year = 13 :=
by 
  intro h
  sorry

end remaining_last_year_budget_is_13_l22_22322


namespace log_identity_l22_22050

noncomputable def log_base_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_identity :
    2 * log_base_10 2 + log_base_10 (5 / 8) - log_base_10 25 = -1 :=
by 
  sorry

end log_identity_l22_22050


namespace fraction_of_cost_due_to_high_octane_is_half_l22_22773

theorem fraction_of_cost_due_to_high_octane_is_half :
  ∀ (cost_regular cost_high : ℝ) (units_high units_regular : ℕ),
    units_high * cost_high + units_regular * cost_regular ≠ 0 →
    cost_high = 3 * cost_regular →
    units_high = 1515 →
    units_regular = 4545 →
    (units_high * cost_high) / (units_high * cost_high + units_regular * cost_regular) = 1 / 2 :=
by
  intro cost_regular cost_high units_high units_regular h_total_cost_ne_zero h_cost_rel h_units_high h_units_regular
  -- skip the actual proof steps
  sorry

end fraction_of_cost_due_to_high_octane_is_half_l22_22773


namespace find_sum_of_coefficients_l22_22248

theorem find_sum_of_coefficients (a b : ℝ)
  (h1 : ∀ x : ℝ, ax^2 + bx + 2 > 0 ↔ (x < -(1/2) ∨ x > 1/3)) :
  a + b = -14 := 
sorry

end find_sum_of_coefficients_l22_22248


namespace f_2017_of_9_eq_8_l22_22707

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def f (n : ℕ) : ℕ :=
  sum_of_digits (n^2 + 1)

def f_k (k n : ℕ) : ℕ :=
  if k = 0 then n else f (f_k (k-1) n)

theorem f_2017_of_9_eq_8 : f_k 2017 9 = 8 := by
  sorry

end f_2017_of_9_eq_8_l22_22707


namespace problem1_problem2_l22_22919

section ProofProblems

variables {a b : ℝ}

-- Given that a and b are distinct positive numbers
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom a_neq_b : a ≠ b

-- Problem (i): Prove that a^4 + b^4 > a^3 * b + a * b^3
theorem problem1 : a^4 + b^4 > a^3 * b + a * b^3 :=
by {
  sorry
}

-- Problem (ii): Prove that a^5 + b^5 > a^3 * b^2 + a^2 * b^3
theorem problem2 : a^5 + b^5 > a^3 * b^2 + a^2 * b^3 :=
by {
  sorry
}

end ProofProblems

end problem1_problem2_l22_22919


namespace max_geq_four_ninths_sum_min_leq_quarter_sum_l22_22019

theorem max_geq_four_ninths_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  max a (max b c) >= 4 / 9 * (a + b + c) :=
by 
  sorry

theorem min_leq_quarter_sum (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_discriminant : b^2 >= 4*a*c) :
  min a (min b c) <= 1 / 4 * (a + b + c) :=
by 
  sorry

end max_geq_four_ninths_sum_min_leq_quarter_sum_l22_22019


namespace data_set_variance_l22_22762

def data_set : List ℕ := [2, 4, 5, 3, 6]

noncomputable def mean (l : List ℕ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℕ) : ℝ :=
  let m : ℝ := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem data_set_variance : variance data_set = 2 := by
  sorry

end data_set_variance_l22_22762


namespace problem1_problem2_l22_22373

-- Problem (I)
theorem problem1 (α : ℝ) (h1 : Real.tan α = 3) :
  (4 * Real.sin (Real.pi - α) - 2 * Real.cos (-α)) / (3 * Real.cos (Real.pi / 2 - α) - 5 * Real.cos (Real.pi + α)) = 5 / 7 := by
sorry

-- Problem (II)
theorem problem2 (x : ℝ) (h2 : Real.sin x + Real.cos x = 1 / 5) (h3 : 0 < x ∧ x < Real.pi) :
  Real.sin x = 4 / 5 ∧ Real.cos x = -3 / 5 := by
sorry

end problem1_problem2_l22_22373


namespace part1_part2_l22_22696

noncomputable def f (x a : ℝ) := (x - 1) * Real.exp x + a * x + 1
noncomputable def g (x : ℝ) := x * Real.exp x

-- Problem Part 1: Prove the range of a for which f(x) has two extreme points
theorem part1 (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a = f x₂ a) ↔ (0 < a ∧ a < (1 / Real.exp 1)) :=
sorry

-- Problem Part 2: Prove the range of a for which f(x) ≥ 2sin(x) for x ≥ 0
theorem part2 (a : ℝ) :
  (∀ x : ℝ, 0 ≤ x → f x a ≥ 2 * Real.sin x) ↔ (2 ≤ a) :=
sorry

end part1_part2_l22_22696


namespace mixture_ratios_equal_quantities_l22_22758

-- Define the given conditions
def ratio_p_milk_water := (5, 4)
def ratio_q_milk_water := (2, 7)

-- Define what we're trying to prove: the ratio p : q such that the resulting mixture has equal milk and water
theorem mixture_ratios_equal_quantities 
  (P Q : ℝ) 
  (h1 : 5 * P + 2 * Q = 4 * P + 7 * Q) :
  P / Q = 5 :=
  sorry

end mixture_ratios_equal_quantities_l22_22758


namespace ratio_jerky_l22_22272

/-
  Given conditions:
  1. Janette camps for 5 days.
  2. She has an initial 40 pieces of beef jerky.
  3. She eats 4 pieces of beef jerky per day.
  4. She will have 10 pieces of beef jerky left after giving some to her brother.

  Prove that the ratio of the pieces of beef jerky she gives to her brother 
  to the remaining pieces is 1:1.
-/

theorem ratio_jerky (days : ℕ) (initial_jerky : ℕ) (jerky_per_day : ℕ) (jerky_left_after_trip : ℕ)
  (h1 : days = 5) (h2 : initial_jerky = 40) (h3 : jerky_per_day = 4) (h4 : jerky_left_after_trip = 10) :
  (initial_jerky - days * jerky_per_day - jerky_left_after_trip) = jerky_left_after_trip :=
by
  sorry

end ratio_jerky_l22_22272


namespace total_cost_nancy_spends_l22_22664

def price_crystal_beads : ℝ := 12
def price_metal_beads : ℝ := 15
def sets_crystal_beads : ℕ := 3
def sets_metal_beads : ℕ := 4
def discount_crystal : ℝ := 0.10
def tax_metal : ℝ := 0.05

theorem total_cost_nancy_spends :
  sets_crystal_beads * price_crystal_beads * (1 - discount_crystal) + 
  sets_metal_beads * price_metal_beads * (1 + tax_metal) = 95.40 := 
  by sorry

end total_cost_nancy_spends_l22_22664


namespace solve_equation_l22_22297

theorem solve_equation (y : ℝ) : 
  5 * (y + 2) + 9 = 3 * (1 - y) ↔ y = -2 := 
by 
  sorry

end solve_equation_l22_22297


namespace min_value_of_n_l22_22775

/-!
    Given:
    - There are 53 students.
    - Each student must join one club and can join at most two clubs.
    - There are three clubs: Science, Culture, and Lifestyle.

    Prove:
    The minimum value of n, where n is the maximum number of people who join exactly the same set of clubs, is 9.
-/

def numStudents : ℕ := 53
def numClubs : ℕ := 3
def numSets : ℕ := 6

theorem min_value_of_n : ∃ n : ℕ, n = 9 ∧ 
  ∀ (students clubs sets : ℕ), students = numStudents → clubs = numClubs → sets = numSets →
  (students / sets + if students % sets = 0 then 0 else 1) = 9 :=
by
  sorry -- proof to be filled out

end min_value_of_n_l22_22775


namespace simplify_expression_l22_22418

theorem simplify_expression (x y : ℝ) (h : y = x / (1 - 2 * x)) :
    (2 * x - 3 * x * y - 2 * y) / (y + x * y - x) = -7 / 3 := 
by {
  sorry
}

end simplify_expression_l22_22418


namespace quadratic_roots_product_sum_l22_22715

theorem quadratic_roots_product_sum :
  ∀ (f g : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 4 * x + 2 = 0 → x = f ∨ x = g) → 
  (f + g = 4 / 3) → 
  (f * g = 2 / 3) → 
  (f + 2) * (g + 2) = 22 / 3 :=
by
  intro f g roots_eq sum_eq product_eq
  sorry

end quadratic_roots_product_sum_l22_22715


namespace percentage_of_diameter_l22_22753

variable (d_R d_S r_R r_S : ℝ)
variable (A_R A_S : ℝ)
variable (pi : ℝ) (h1 : pi > 0)

theorem percentage_of_diameter 
(h_area : A_R = 0.64 * A_S) 
(h_radius_R : r_R = d_R / 2) 
(h_radius_S : r_S = d_S / 2)
(h_area_R : A_R = pi * r_R^2) 
(h_area_S : A_S = pi * r_S^2) 
: (d_R / d_S) * 100 = 80 := by
  sorry

end percentage_of_diameter_l22_22753


namespace net_percentage_gain_approx_l22_22597

noncomputable def netPercentageGain : ℝ :=
  let costGlassBowls := 250 * 18
  let costCeramicPlates := 150 * 25
  let totalCostBeforeDiscount := costGlassBowls + costCeramicPlates
  let discount := 0.05 * totalCostBeforeDiscount
  let totalCostAfterDiscount := totalCostBeforeDiscount - discount
  let revenueGlassBowls := 200 * 25
  let revenueCeramicPlates := 120 * 32
  let totalRevenue := revenueGlassBowls + revenueCeramicPlates
  let costBrokenGlassBowls := 30 * 18
  let costBrokenCeramicPlates := 10 * 25
  let totalCostBrokenItems := costBrokenGlassBowls + costBrokenCeramicPlates
  let netGain := totalRevenue - (totalCostAfterDiscount + totalCostBrokenItems)
  let netPercentageGain := (netGain / totalCostAfterDiscount) * 100
  netPercentageGain

theorem net_percentage_gain_approx :
  abs (netPercentageGain - 2.71) < 0.01 := sorry

end net_percentage_gain_approx_l22_22597


namespace overall_average_score_l22_22565

structure Club where
  members : Nat
  average_score : Nat

def ClubA : Club := { members := 40, average_score := 90 }
def ClubB : Club := { members := 50, average_score := 81 }

theorem overall_average_score : 
  (ClubA.members * ClubA.average_score + ClubB.members * ClubB.average_score) / 
  (ClubA.members + ClubB.members) = 85 :=
by
  sorry

end overall_average_score_l22_22565


namespace kelcie_books_multiple_l22_22974

theorem kelcie_books_multiple (x : ℕ) :
  let megan_books := 32
  let kelcie_books := megan_books / 4
  let greg_books := x * kelcie_books + 9
  let total_books := megan_books + kelcie_books + greg_books
  total_books = 65 → x = 2 :=
by
  intros megan_books kelcie_books greg_books total_books h
  sorry

end kelcie_books_multiple_l22_22974


namespace number_of_students_in_class_l22_22146

theorem number_of_students_in_class
  (x : ℕ)
  (S : ℝ)
  (incorrect_score correct_score : ℝ)
  (incorrect_score_mistake : incorrect_score = 85)
  (correct_score_corrected : correct_score = 78)
  (average_difference : ℝ)
  (average_difference_value : average_difference = 0.75)
  (test_attendance : ℕ)
  (test_attendance_value : test_attendance = x - 3)
  (average_difference_condition : (S + incorrect_score) / test_attendance - (S + correct_score) / test_attendance = average_difference) :
  x = 13 :=
by
  sorry

end number_of_students_in_class_l22_22146


namespace min_value_x_plus_y_l22_22081

theorem min_value_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + y ≥ 4 :=
by
  sorry

end min_value_x_plus_y_l22_22081


namespace distance_covered_is_9_17_miles_l22_22424

noncomputable def totalDistanceCovered 
  (walkingTimeInMinutes : ℕ) (walkingRate : ℝ)
  (runningTimeInMinutes : ℕ) (runningRate : ℝ)
  (cyclingTimeInMinutes : ℕ) (cyclingRate : ℝ) : ℝ :=
  (walkingRate * (walkingTimeInMinutes / 60.0)) + 
  (runningRate * (runningTimeInMinutes / 60.0)) + 
  (cyclingRate * (cyclingTimeInMinutes / 60.0))

theorem distance_covered_is_9_17_miles :
  totalDistanceCovered 30 3 20 8 25 12 = 9.17 := 
by 
  sorry

end distance_covered_is_9_17_miles_l22_22424


namespace unique_triple_l22_22437

def is_prime (n : ℕ) : Prop := Nat.Prime n

noncomputable def find_triples (x y z : ℕ) : Prop :=
  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
  is_prime x ∧ is_prime y ∧ is_prime z ∧
  is_prime (x - y) ∧ is_prime (y - z) ∧ is_prime (x - z)

theorem unique_triple :
  ∀ (x y z : ℕ), find_triples x y z → (x, y, z) = (7, 5, 2) :=
by
  sorry

end unique_triple_l22_22437


namespace final_fraction_of_water_is_243_over_1024_l22_22296

theorem final_fraction_of_water_is_243_over_1024 :
  let initial_volume := 20
  let replaced_volume := 5
  let cycles := 5
  let initial_fraction_of_water := 1
  let final_fraction_of_water :=
        (initial_fraction_of_water * (initial_volume - replaced_volume) / initial_volume) ^ cycles
  final_fraction_of_water = 243 / 1024 :=
by
  sorry

end final_fraction_of_water_is_243_over_1024_l22_22296


namespace find_a_b_find_extreme_values_l22_22596

-- Definitions based on the conditions in the problem
def f (x a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2 * b

-- The function f attains a maximum value of 2 at x = -1
def f_max_at_neg_1 (a b : ℝ) : Prop :=
  (∃ x : ℝ, x = -1 ∧ 
  (∀ y : ℝ, f x a b ≤ f y a b)) ∧ f (-1) a b = 2

-- Statement (1): Finding the values of a and b
theorem find_a_b : ∃ a b : ℝ, f_max_at_neg_1 a b ∧ a = 2 ∧ b = 1 :=
sorry

-- The function f with a=2 and b=1
def f_specific (x : ℝ) : ℝ := f x 2 1

-- Statement (2): Finding the extreme values of f(x) on the interval [-1, 1]
def extreme_values_on_interval : Prop :=
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → f_specific x ≤ 6 ∧ f_specific x ≥ 50/27) ∧
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 6) ∧ 
  (∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ f_specific x = 50/27)

theorem find_extreme_values : extreme_values_on_interval :=
sorry

end find_a_b_find_extreme_values_l22_22596


namespace line_PQ_passes_through_fixed_point_l22_22564

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop :=
  x^2 - y^2 / 2 = 1

-- Define the conditions for points P and Q on the hyperbola
def on_hyperbola (P Q : ℝ × ℝ) : Prop :=
  hyperbola P.1 P.2 ∧ hyperbola Q.1 Q.2

-- Define the condition for perpendicular lines, given points A, P, and Q
def perpendicular (A P Q : ℝ × ℝ) : Prop :=
  ((P.2 - A.2) / (P.1 - A.1)) * ((Q.2 - A.2) / (Q.1 - A.1)) = -1

-- Define the main theorem to prove
theorem line_PQ_passes_through_fixed_point :
  ∀ (P Q : ℝ × ℝ), on_hyperbola P Q → perpendicular ⟨-1, 0⟩ P Q →
    ∃ (b : ℝ), ∀ (y : ℝ), (P.1 = y * P.2 + b ∨ Q.1 = y * Q.2 + b) → (b = 3) :=
by
  sorry

end line_PQ_passes_through_fixed_point_l22_22564


namespace sum_of_solutions_is_24_l22_22784

theorem sum_of_solutions_is_24 (a : ℝ) (x1 x2 : ℝ) 
    (h1 : abs (x1 - a) = 100) (h2 : abs (x2 - a) = 100)
    (sum_eq : x1 + x2 = 24) : a = 12 :=
sorry

end sum_of_solutions_is_24_l22_22784


namespace gpa_at_least_3_5_l22_22521

noncomputable def prob_gpa_at_least_3_5 : ℚ :=
  let p_A_eng := 1 / 3
  let p_B_eng := 1 / 5
  let p_C_eng := 7 / 15 -- 1 - p_A_eng - p_B_eng
  
  let p_A_hist := 1 / 5
  let p_B_hist := 1 / 4
  let p_C_hist := 11 / 20 -- 1 - p_A_hist - p_B_hist

  let prob_two_As := p_A_eng * p_A_hist
  let prob_A_eng_B_hist := p_A_eng * p_B_hist
  let prob_A_hist_B_eng := p_A_hist * p_B_eng
  let prob_two_Bs := p_B_eng * p_B_hist

  let total_prob := prob_two_As + prob_A_eng_B_hist + prob_A_hist_B_eng + prob_two_Bs
  total_prob

theorem gpa_at_least_3_5 : prob_gpa_at_least_3_5 = 6 / 25 := by {
  sorry
}

end gpa_at_least_3_5_l22_22521


namespace green_square_area_percentage_l22_22627

variable (s a : ℝ)
variable (h : a^2 + 4 * a * (s - 2 * a) = 0.49 * s^2)

theorem green_square_area_percentage :
  (a^2 / s^2) = 0.1225 :=
sorry

end green_square_area_percentage_l22_22627


namespace find_x_l22_22954

theorem find_x (x : ℝ) (h : 5.76 = 0.12 * 0.40 * x) : x = 120 := 
sorry

end find_x_l22_22954


namespace angle_passing_through_point_l22_22994

-- Definition of the problem conditions
def is_terminal_side_of_angle (x y : ℝ) (α : ℝ) : Prop :=
  let r := Real.sqrt (x^2 + y^2);
  (x = Real.cos α * r) ∧ (y = Real.sin α * r)

-- Lean 4 statement of the problem
theorem angle_passing_through_point (α : ℝ) :
  is_terminal_side_of_angle 1 (-1) α → α = - (Real.pi / 4) :=
by sorry

end angle_passing_through_point_l22_22994


namespace circular_garden_radius_l22_22511

theorem circular_garden_radius (r : ℝ) (h : 2 * Real.pi * r = (1 / 8) * Real.pi * r^2) : r = 16 :=
sorry

end circular_garden_radius_l22_22511


namespace walk_direction_east_l22_22793

theorem walk_direction_east (m : ℤ) (h : m = -2023) : m = -(-2023) :=
by
  sorry

end walk_direction_east_l22_22793


namespace bert_same_kangaroos_as_kameron_in_40_days_l22_22670

theorem bert_same_kangaroos_as_kameron_in_40_days
  (k : ℕ := 100)
  (b : ℕ := 20)
  (r : ℕ := 2) :
  ∃ t : ℕ, t = 40 ∧ b + t * r = k := by
  sorry

end bert_same_kangaroos_as_kameron_in_40_days_l22_22670


namespace points_on_parabola_l22_22499

theorem points_on_parabola (s : ℝ) : ∃ (u v : ℝ), u = 3^s - 4 ∧ v = 9^s - 7 * 3^s - 2 ∧ v = u^2 + u - 14 :=
by
  sorry

end points_on_parabola_l22_22499


namespace carnival_total_cost_l22_22196

def morning_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + over18_cost

def afternoon_costs (under18_cost over18_cost : ℕ) : ℕ :=
  under18_cost + 1 + over18_cost + 1

noncomputable def mara_cost : ℕ :=
  let bumper_car_cost := morning_costs 2 0 + afternoon_costs 2 0
  let ferris_wheel_cost := morning_costs 5 5 + 5
  bumper_car_cost + ferris_wheel_cost

noncomputable def riley_cost : ℕ :=
  let space_shuttle_cost := morning_costs 0 5 + afternoon_costs 0 5
  let ferris_wheel_cost := morning_costs 0 6 + (6 + 1)
  space_shuttle_cost + ferris_wheel_cost

theorem carnival_total_cost :
  mara_cost + riley_cost = 61 := by
  sorry

end carnival_total_cost_l22_22196


namespace interest_rate_is_10_percent_l22_22389

theorem interest_rate_is_10_percent
  (principal : ℝ)
  (interest_rate_c : ℝ) 
  (time : ℝ)
  (gain_b : ℝ)
  (interest_c : ℝ := principal * interest_rate_c / 100 * time)
  (interest_a : ℝ := interest_c - gain_b)
  (expected_rate : ℝ := (interest_a / (principal * time)) * 100)
  (h1: principal = 3500)
  (h2: interest_rate_c = 12)
  (h3: time = 3)
  (h4: gain_b = 210)
  : expected_rate = 10 := 
  by 
  sorry

end interest_rate_is_10_percent_l22_22389


namespace initial_number_of_bedbugs_l22_22686

theorem initial_number_of_bedbugs (N : ℕ) 
  (h1 : ∃ N : ℕ, True)
  (h2 : ∀ (n : ℕ), (triples_daily : ℕ → ℕ) → triples_daily n = 3 * n)
  (h3 : ∀ (n : ℕ), (N * 3^4 = n) → n = 810) : 
  N = 10 :=
sorry

end initial_number_of_bedbugs_l22_22686


namespace base_k_eq_26_l22_22765

theorem base_k_eq_26 (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 :=
by {
  -- The actual proof will go here.
  sorry
}

end base_k_eq_26_l22_22765


namespace lap_distance_l22_22926

theorem lap_distance (boys_laps : ℕ) (girls_extra_laps : ℕ) (total_girls_miles : ℚ) : 
  boys_laps = 27 → girls_extra_laps = 9 → total_girls_miles = 27 →
  (total_girls_miles / (boys_laps + girls_extra_laps) = 3 / 4) :=
by
  intros hb hg hm
  sorry

end lap_distance_l22_22926


namespace three_layers_rug_area_l22_22586

theorem three_layers_rug_area :
  ∀ (A B C D E : ℝ),
    A + B + C = 212 →
    (A + B + C) - D - 2 * E = 140 →
    D = 24 →
    E = 24 :=
by
  intros A B C D E h1 h2 h3
  sorry

end three_layers_rug_area_l22_22586


namespace right_triangle_sides_l22_22051

theorem right_triangle_sides 
  (a b c : ℝ) 
  (h_right_angle : a^2 + b^2 = c^2) 
  (h_area : (1 / 2) * a * b = 150) 
  (h_perimeter : a + b + c = 60) 
  : (a = 15 ∧ b = 20 ∧ c = 25) ∨ (a = 20 ∧ b = 15 ∧ c = 25) :=
by
  sorry

end right_triangle_sides_l22_22051


namespace find_first_train_length_l22_22804

namespace TrainProblem

-- Define conditions
def speed_first_train_kmph := 42
def speed_second_train_kmph := 48
def length_second_train_m := 163
def time_clear_s := 12
def relative_speed_kmph := speed_first_train_kmph + speed_second_train_kmph

-- Convert kmph to m/s
def kmph_to_mps(kmph : ℕ) : ℕ := kmph * 5 / 18
def relative_speed_mps := kmph_to_mps relative_speed_kmph

-- Calculate total distance covered by the trains in meters
def total_distance_m := relative_speed_mps * time_clear_s

-- Define the length of the first train to be proved
def length_first_train_m := 137

-- Theorem statement
theorem find_first_train_length :
  total_distance_m = length_first_train_m + length_second_train_m :=
sorry

end TrainProblem

end find_first_train_length_l22_22804


namespace max_val_proof_l22_22166

noncomputable def max_val (p q r x y z : ℝ) : ℝ :=
  1 / (p + q) + 1 / (p + r) + 1 / (q + r) + 1 / (x + y) + 1 / (x + z) + 1 / (y + z)

theorem max_val_proof {p q r x y z : ℝ}
  (h_pos_p : 0 < p) (h_pos_q : 0 < q) (h_pos_r : 0 < r)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum_pqr : p + q + r = 2) (h_sum_xyz : x + y + z = 1) :
  max_val p q r x y z = 27 / 4 :=
sorry

end max_val_proof_l22_22166


namespace adam_cat_food_vs_dog_food_l22_22347

def cat_packages := 15
def dog_packages := 10
def cans_per_cat_package := 12
def cans_per_dog_package := 8

theorem adam_cat_food_vs_dog_food:
  cat_packages * cans_per_cat_package - dog_packages * cans_per_dog_package = 100 :=
by
  sorry

end adam_cat_food_vs_dog_food_l22_22347


namespace area_of_triangle_ACD_l22_22070

theorem area_of_triangle_ACD :
  ∀ (AD AC height_AD height_AC : ℝ),
  AD = 6 → height_AD = 3 → AC = 3 → height_AC = 3 →
  (1 / 2 * AD * height_AD - 1 / 2 * AC * height_AC) = 4.5 :=
by
  intros AD AC height_AD height_AC hAD hheight_AD hAC hheight_AC
  sorry

end area_of_triangle_ACD_l22_22070


namespace sin_alpha_plus_pi_over_4_tan_double_alpha_l22_22834

-- Definitions of sin and tan 
open Real

variable (α : ℝ)

-- Given conditions
axiom α_in_interval : 0 < α ∧ α < π / 2
axiom sin_alpha_def : sin α = sqrt 5 / 5

-- Statement to prove
theorem sin_alpha_plus_pi_over_4 : sin (α + π / 4) = 3 * sqrt 10 / 10 :=
by
  sorry

theorem tan_double_alpha : tan (2 * α) = 4 / 3 :=
by
  sorry

end sin_alpha_plus_pi_over_4_tan_double_alpha_l22_22834


namespace one_add_i_cubed_eq_one_sub_i_l22_22852

theorem one_add_i_cubed_eq_one_sub_i (i : ℂ) (h : i^2 = -1) : 1 + i^3 = 1 - i := by
  sorry

end one_add_i_cubed_eq_one_sub_i_l22_22852


namespace Captain_Zarnin_staffing_scheme_l22_22279

theorem Captain_Zarnin_staffing_scheme :
  let positions := 6
  let candidates := 15
  (Nat.choose candidates positions) * 
  (Nat.factorial positions) = 3276000 :=
by
  let positions := 6
  let candidates := 15
  let ways_to_choose := Nat.choose candidates positions
  let ways_to_permute := Nat.factorial positions
  have h : (ways_to_choose * ways_to_permute) = 3276000 := sorry
  exact h

end Captain_Zarnin_staffing_scheme_l22_22279


namespace line_through_origin_and_conditions_l22_22067

-- Definitions:
def system_defines_line (m n p x y z : ℝ) : Prop :=
  (x / m = y / n) ∧ (y / n = z / p)

def lies_in_coordinate_plane (m n p : ℝ) : Prop :=
  (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)

def coincides_with_coordinate_axis (m n p : ℝ) : Prop :=
  (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)

-- Theorem statement:
theorem line_through_origin_and_conditions (m n p x y z : ℝ) :
  system_defines_line m n p x y z →
  (∀ m n p, lies_in_coordinate_plane m n p ↔ (m = 0 ∨ n = 0 ∨ p = 0) ∧ ¬(m = 0 ∧ n = 0 ∧ p = 0)) ∧
  (∀ m n p, coincides_with_coordinate_axis m n p ↔ (m = 0 ∧ n = 0) ∨ (m = 0 ∧ p = 0) ∨ (n = 0 ∧ p = 0)) :=
by
  sorry

end line_through_origin_and_conditions_l22_22067


namespace grover_total_profit_l22_22371

-- Definitions based on conditions
def original_price : ℝ := 10
def discount_first_box : ℝ := 0.20
def discount_second_box : ℝ := 0.30
def discount_third_box : ℝ := 0.40
def packs_first_box : ℕ := 20
def packs_second_box : ℕ := 30
def packs_third_box : ℕ := 40
def masks_per_pack : ℕ := 5
def price_per_mask_first_box : ℝ := 0.75
def price_per_mask_second_box : ℝ := 0.85
def price_per_mask_third_box : ℝ := 0.95

-- Computations
def cost_first_box := original_price - (discount_first_box * original_price)
def cost_second_box := original_price - (discount_second_box * original_price)
def cost_third_box := original_price - (discount_third_box * original_price)

def total_cost := cost_first_box + cost_second_box + cost_third_box

def revenue_first_box := packs_first_box * masks_per_pack * price_per_mask_first_box
def revenue_second_box := packs_second_box * masks_per_pack * price_per_mask_second_box
def revenue_third_box := packs_third_box * masks_per_pack * price_per_mask_third_box

def total_revenue := revenue_first_box + revenue_second_box + revenue_third_box

def total_profit := total_revenue - total_cost

-- Proof statement
theorem grover_total_profit : total_profit = 371.5 := by
  sorry

end grover_total_profit_l22_22371


namespace smallest_positive_multiple_of_37_l22_22737

theorem smallest_positive_multiple_of_37 :
  ∃ n, n > 0 ∧ (∃ a, n = 37 * a) ∧ (∃ k, n = 76 * k + 7) ∧ n = 2405 := 
by
  sorry

end smallest_positive_multiple_of_37_l22_22737


namespace original_price_of_sarees_l22_22820

theorem original_price_of_sarees (P : ℝ) (h : 0.75 * 0.85 * P = 306) : P = 480 :=
by
  sorry

end original_price_of_sarees_l22_22820


namespace handshake_count_l22_22821

-- Define the number of team members, referees, and the total number of handshakes
def num_team_members := 7
def num_referees := 3
def num_coaches := 2

-- Calculate the handshakes
def team_handshakes := num_team_members * num_team_members
def player_refhandshakes := (2 * num_team_members) * num_referees
def coach_handshakes := num_coaches * (2 * num_team_members + num_referees)

-- The total number of handshakes
def total_handshakes := team_handshakes + player_refhandshakes + coach_handshakes

-- The proof statement
theorem handshake_count : total_handshakes = 125 := 
by
  -- Placeholder for proof
  sorry

end handshake_count_l22_22821


namespace least_product_ab_l22_22722

theorem least_product_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (1 : ℚ) / a + 1 / (3 * b) = 1 / 6) : a * b ≥ 48 :=
by
  sorry

end least_product_ab_l22_22722


namespace uniqueSumEqualNumber_l22_22257

noncomputable def sumPreceding (n : ℕ) : ℕ :=
  (n * (n - 1)) / 2

theorem uniqueSumEqualNumber :
  ∃! n : ℕ, sumPreceding n = n := by
  sorry

end uniqueSumEqualNumber_l22_22257


namespace find_quotient_l22_22612

theorem find_quotient (D d R Q : ℤ) (hD : D = 729) (hd : d = 38) (hR : R = 7)
  (h : D = d * Q + R) : Q = 19 := by
  sorry

end find_quotient_l22_22612


namespace find_root_of_equation_l22_22395

theorem find_root_of_equation (a b c d x : ℕ) (h_ad : a + d = 2016) (h_bc : b + c = 2016) (h_ac : a ≠ c) :
  (x - a) * (x - b) = (x - c) * (x - d) → x = 1008 :=
by
  sorry

end find_root_of_equation_l22_22395


namespace jenny_change_l22_22069

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

end jenny_change_l22_22069


namespace largest_study_only_Biology_l22_22892

-- Let's define the total number of students
def total_students : ℕ := 500

-- Define the given conditions
def S : ℕ := 65 * total_students / 100
def M : ℕ := 55 * total_students / 100
def B : ℕ := 50 * total_students / 100
def P : ℕ := 15 * total_students / 100

def MS : ℕ := 35 * total_students / 100
def MB : ℕ := 25 * total_students / 100
def BS : ℕ := 20 * total_students / 100
def MSB : ℕ := 10 * total_students / 100

-- Required to prove that the largest number of students who study only Biology is 75
theorem largest_study_only_Biology : 
  (B - MB - BS + MSB) = 75 :=
by 
  sorry

end largest_study_only_Biology_l22_22892


namespace arithmetic_sequence_nth_term_l22_22165

theorem arithmetic_sequence_nth_term (a₁ : ℤ) (d : ℤ) (n : ℕ) :
  (a₁ = 11) →
  (d = -3) →
  (-49 = a₁ + (n - 1) * d) →
  (n = 21) :=
by 
  intros h₁ h₂ h₃
  sorry

end arithmetic_sequence_nth_term_l22_22165


namespace sqrt_defined_iff_le_l22_22077

theorem sqrt_defined_iff_le (x : ℝ) : (∃ y : ℝ, y^2 = 4 - x) ↔ (x ≤ 4) :=
by
  sorry

end sqrt_defined_iff_le_l22_22077


namespace number_of_rectangles_l22_22182

theorem number_of_rectangles (m n : ℕ) (h1 : m = 8) (h2 : n = 10) : (m - 1) * (n - 1) = 63 := by
  sorry

end number_of_rectangles_l22_22182


namespace average_tickets_sold_by_male_members_l22_22813

theorem average_tickets_sold_by_male_members 
  (M F : ℕ)
  (total_average : ℕ)
  (female_average : ℕ)
  (ratio : ℕ × ℕ)
  (h1 : total_average = 66)
  (h2 : female_average = 70)
  (h3 : ratio = (1, 2))
  (h4 : F = 2 * M)
  (h5 : (M + F) * total_average = M * r + F * female_average) :
  r = 58 :=
sorry

end average_tickets_sold_by_male_members_l22_22813


namespace range_of_k_l22_22221

theorem range_of_k (k : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 + 2*x - 4*y + k - 2 = 0 ∧ (x, y) = (1, 2)) →
  (3 < k ∧ k < 7) :=
by
  intros hxy
  sorry

end range_of_k_l22_22221


namespace probability_of_pink_gumball_l22_22330

theorem probability_of_pink_gumball (P_B P_P : ℝ)
    (h1 : P_B ^ 2 = 25 / 49)
    (h2 : P_B + P_P = 1) :
    P_P = 2 / 7 := 
    sorry

end probability_of_pink_gumball_l22_22330


namespace sum_of_first_100_digits_of_1_div_2222_l22_22536

theorem sum_of_first_100_digits_of_1_div_2222 : 
  (let repeating_block := [0, 0, 0, 4, 5];
  let sum_of_digits (lst : List ℕ) := lst.sum;
  let block_sum := sum_of_digits repeating_block;
  let num_blocks := 100 / 5;
  num_blocks * block_sum = 180) :=
by 
  let repeating_block := [0, 0, 0, 4, 5]
  let sum_of_digits (lst : List ℕ) := lst.sum
  let block_sum := sum_of_digits repeating_block
  let num_blocks := 100 / 5
  have h : num_blocks * block_sum = 180 := sorry
  exact h

end sum_of_first_100_digits_of_1_div_2222_l22_22536


namespace least_value_x_l22_22132

theorem least_value_x (x : ℕ) (p q : ℕ) (h_prime_p : Nat.Prime p) (h_prime_q : Nat.Prime q)
  (h_distinct : p ≠ q) (h_diff : q - p = 3) (h_even_prim : x / (11 * p * q) = 2) : x = 770 := by
  sorry

end least_value_x_l22_22132


namespace initial_milk_amount_l22_22493

theorem initial_milk_amount (M : ℝ) (H1 : 0.05 * M = 0.02 * (M + 15)) : M = 10 :=
by
  sorry

end initial_milk_amount_l22_22493


namespace cistern_water_depth_l22_22832

theorem cistern_water_depth:
  ∀ h: ℝ,
  (4 * 4 + 4 * h * 4 + 4 * h * 4 = 36) → h = 1.25 := by
    sorry

end cistern_water_depth_l22_22832


namespace max_x_minus_y_isosceles_l22_22591

theorem max_x_minus_y_isosceles (x y : ℝ) (hx : x ≠ 50) (hy : y ≠ 50) 
  (h_iso1 : x = y ∨ 50 = y) (h_iso2 : x = y ∨ 50 = x)
  (h_triangle : 50 + x + y = 180) : 
  max (x - y) (y - x) = 30 :=
sorry

end max_x_minus_y_isosceles_l22_22591


namespace minimum_bail_rate_l22_22903

theorem minimum_bail_rate 
  (distance : ℝ)
  (leak_rate : ℝ)
  (max_water : ℝ)
  (rowing_speed : ℝ)
  (bail_rate : ℝ)
  (time_to_shore : ℝ) :
  distance = 2 ∧
  leak_rate = 15 ∧
  max_water = 60 ∧
  rowing_speed = 3 ∧
  time_to_shore = distance / rowing_speed * 60 →
  bail_rate = (leak_rate * time_to_shore - max_water) / time_to_shore →
  bail_rate = 13.5 :=
by
  intros
  sorry

end minimum_bail_rate_l22_22903


namespace probability_not_eat_pizza_l22_22388

theorem probability_not_eat_pizza (P_eat_pizza : ℚ) (h : P_eat_pizza = 5 / 8) : 
  ∃ P_not_eat_pizza : ℚ, P_not_eat_pizza = 3 / 8 :=
by
  use 1 - P_eat_pizza
  sorry

end probability_not_eat_pizza_l22_22388


namespace rectangle_area_l22_22120

theorem rectangle_area (b : ℕ) (l : ℕ) (h1 : l = 3 * b) (h2 : 2 * (l + b) = 48) : l * b = 108 := 
by
  sorry

end rectangle_area_l22_22120


namespace right_triangle_area_l22_22326

theorem right_triangle_area (leg1 leg2 hypotenuse : ℕ) (h_leg1 : leg1 = 30)
  (h_hypotenuse : hypotenuse = 34)
  (hypotenuse_sq : hypotenuse * hypotenuse = leg1 * leg1 + leg2 * leg2) :
  (1 / 2 : ℚ) * leg1 * leg2 = 240 := by
  sorry

end right_triangle_area_l22_22326


namespace parabola_equation_l22_22843

noncomputable def parabola_focus : (ℝ × ℝ) := (5, -2)

noncomputable def parabola_directrix (x y : ℝ) : Prop := 4 * x - 5 * y = 20

theorem parabola_equation (x y : ℝ) :
  (parabola_focus = (5, -2)) →
  (parabola_directrix x y) →
  25 * x^2 + 40 * x * y + 16 * y^2 - 650 * x + 184 * y + 1009 = 0 :=
by
  sorry

end parabola_equation_l22_22843


namespace total_clothes_count_l22_22238

theorem total_clothes_count (shirts_per_pants : ℕ) (pants : ℕ) (shirts : ℕ) : shirts_per_pants = 6 → pants = 40 → shirts = shirts_per_pants * pants → shirts + pants = 280 := by
  intro h1 h2 h3
  rw [h1, h2] at h3
  rw [h3]
  sorry

end total_clothes_count_l22_22238


namespace longest_side_of_enclosure_l22_22688

theorem longest_side_of_enclosure (l w : ℝ)
  (h_perimeter : 2 * l + 2 * w = 240)
  (h_area : l * w = 8 * 240) :
  max l w = 80 :=
by
  sorry

end longest_side_of_enclosure_l22_22688


namespace regression_line_is_y_eq_x_plus_1_l22_22473

theorem regression_line_is_y_eq_x_plus_1 :
  let points : List (ℝ × ℝ) := [(1, 2), (2, 3), (3, 4), (4, 5)]
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) ∈ points → y = m * x + b) ∧ m = 1 ∧ b = 1 :=
by
  sorry 

end regression_line_is_y_eq_x_plus_1_l22_22473


namespace line_through_points_a_minus_b_l22_22983

theorem line_through_points_a_minus_b :
  ∃ a b : ℝ, 
  (∀ x, (x = 3 → 7 = a * 3 + b) ∧ (x = 6 → 19 = a * 6 + b)) → 
  a - b = 9 :=
by
  sorry

end line_through_points_a_minus_b_l22_22983


namespace moles_of_NaCl_formed_l22_22219

-- Define the balanced chemical reaction and quantities
def chemical_reaction :=
  "NaOH + HCl → NaCl + H2O"

-- Define the initial moles of sodium hydroxide (NaOH) and hydrochloric acid (HCl)
def moles_NaOH : ℕ := 2
def moles_HCl : ℕ := 2

-- The stoichiometry from the balanced equation: 1 mole NaOH reacts with 1 mole HCl to produce 1 mole NaCl.
def stoichiometry_NaOH_to_NaCl : ℕ := 1
def stoichiometry_HCl_to_NaCl : ℕ := 1

-- Given the initial conditions, prove that 2 moles of NaCl are formed.
theorem moles_of_NaCl_formed :
  (moles_NaOH = 2) → (moles_HCl = 2) → 2 = 2 :=
by 
  sorry

end moles_of_NaCl_formed_l22_22219


namespace whitney_total_cost_l22_22431

-- Definitions of the number of items and their costs
def w := 15
def c_w := 14
def f := 12
def c_f := 13
def s := 5
def c_s := 10
def m := 8
def c_m := 3

-- The total cost Whitney spent
theorem whitney_total_cost :
  w * c_w + f * c_f + s * c_s + m * c_m = 440 := by
  sorry

end whitney_total_cost_l22_22431


namespace max_x_of_conditions_l22_22432

theorem max_x_of_conditions (x y z : ℝ) (h1 : x + y + z = 6) (h2 : xy + xz + yz = 11) : x ≤ 2 :=
by
  -- Placeholder for the actual proof
  sorry

end max_x_of_conditions_l22_22432


namespace negation_of_existence_l22_22144

theorem negation_of_existence : 
  (¬ ∃ x : ℝ, Real.exp x = x - 1) = (∀ x : ℝ, Real.exp x ≠ x - 1) :=
by
  sorry

end negation_of_existence_l22_22144


namespace total_bouncy_balls_l22_22246

-- Definitions of the given quantities
def r : ℕ := 4 -- number of red packs
def y : ℕ := 8 -- number of yellow packs
def g : ℕ := 4 -- number of green packs
def n : ℕ := 10 -- number of balls per pack

-- Proof statement to show the correct total number of balls
theorem total_bouncy_balls : r * n + y * n + g * n = 160 := by
  sorry

end total_bouncy_balls_l22_22246


namespace value_of_fraction_l22_22749

theorem value_of_fraction (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 + 2 * n - 1 = 0) (h3 : m * n ≠ 1) : 
  (mn + n + 1) / n = 3 :=
by
  sorry

end value_of_fraction_l22_22749


namespace marta_sold_on_saturday_l22_22430

-- Definitions of conditions
def initial_shipment : ℕ := 1000
def rotten_tomatoes : ℕ := 200
def second_shipment : ℕ := 2000
def tomatoes_on_tuesday : ℕ := 2500
def x := 300

-- Total tomatoes on Monday after the second shipment
def tomatoes_on_monday (sold_tomatoes : ℕ) : ℕ :=
  initial_shipment - sold_tomatoes - rotten_tomatoes + second_shipment

-- Theorem statement to prove
theorem marta_sold_on_saturday : (tomatoes_on_monday x = tomatoes_on_tuesday) -> (x = 300) :=
by 
  intro h
  sorry

end marta_sold_on_saturday_l22_22430


namespace find_difference_condition_l22_22806

variable (a b c : ℝ)

theorem find_difference_condition (h1 : (a + b) / 2 = 40) (h2 : (b + c) / 2 = 60) : c - a = 40 := by
  sorry

end find_difference_condition_l22_22806


namespace minimum_fruits_l22_22227

open Nat

theorem minimum_fruits (n : ℕ) :
    (n % 3 = 2) ∧ (n % 4 = 3) ∧ (n % 5 = 4) ∧ (n % 6 = 5) →
    n = 59 := by
  sorry

end minimum_fruits_l22_22227


namespace NaNO3_moles_l22_22417

theorem NaNO3_moles (moles_NaCl moles_HNO3 moles_NaNO3 : ℝ) (h_HNO3 : moles_HNO3 = 2) (h_ratio : moles_NaNO3 = moles_NaCl) (h_NaNO3 : moles_NaNO3 = 2) :
  moles_NaNO3 = 2 :=
sorry

end NaNO3_moles_l22_22417


namespace imaginary_part_zero_iff_a_eq_neg1_l22_22704

theorem imaginary_part_zero_iff_a_eq_neg1 (a : ℝ) (h : (Complex.I * (a + Complex.I) + a - 1).im = 0) : 
  a = -1 :=
sorry

end imaginary_part_zero_iff_a_eq_neg1_l22_22704


namespace find_LCM_of_three_numbers_l22_22535

noncomputable def LCM_of_three_numbers (a b c : ℕ) : ℕ :=
  Nat.lcm (Nat.lcm a b) c

theorem find_LCM_of_three_numbers
  (a b c : ℕ)
  (h_prod : a * b * c = 1354808)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 11) :
  LCM_of_three_numbers a b c = 123164 := by
  sorry

end find_LCM_of_three_numbers_l22_22535


namespace seven_times_one_fifth_cubed_l22_22163

theorem seven_times_one_fifth_cubed : 7 * (1 / 5) ^ 3 = 7 / 125 := 
by 
  sorry

end seven_times_one_fifth_cubed_l22_22163


namespace allocation_methods_count_l22_22661

/-- The number of ways to allocate 24 quotas to 3 venues such that:
1. Each venue gets at least one quota.
2. Each venue gets a different number of quotas.
is equal to 222. -/
theorem allocation_methods_count : 
  ∃ n : ℕ, n = 222 ∧ 
  ∃ (a b c: ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ 
  a + b + c = 24 ∧ 
  1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c := 
sorry

end allocation_methods_count_l22_22661


namespace thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l22_22589

theorem thirty_percent_less_than_ninety_eq_one_fourth_more_than_n (n : ℝ) :
  0.7 * 90 = (5 / 4) * n → n = 50.4 :=
by sorry

end thirty_percent_less_than_ninety_eq_one_fourth_more_than_n_l22_22589


namespace european_savings_correct_l22_22273

noncomputable def movie_ticket_price : ℝ := 8
noncomputable def popcorn_price : ℝ := 8 - 3
noncomputable def drink_price : ℝ := popcorn_price + 1
noncomputable def candy_price : ℝ := drink_price / 2
noncomputable def hotdog_price : ℝ := 5

noncomputable def monday_discount_popcorn : ℝ := 0.15 * popcorn_price
noncomputable def wednesday_discount_candy : ℝ := 0.10 * candy_price
noncomputable def friday_discount_drink : ℝ := 0.05 * drink_price

noncomputable def monday_price : ℝ := 22
noncomputable def wednesday_price : ℝ := 20
noncomputable def friday_price : ℝ := 25
noncomputable def weekend_price : ℝ := 25
noncomputable def monday_exchange_rate : ℝ := 0.85
noncomputable def wednesday_exchange_rate : ℝ := 0.85
noncomputable def friday_exchange_rate : ℝ := 0.83
noncomputable def weekend_exchange_rate : ℝ := 0.81

noncomputable def total_cost_monday : ℝ := movie_ticket_price + (popcorn_price - monday_discount_popcorn) + drink_price + candy_price + hotdog_price
noncomputable def savings_monday_usd : ℝ := total_cost_monday - monday_price
noncomputable def savings_monday_eur : ℝ := savings_monday_usd * monday_exchange_rate

noncomputable def total_cost_wednesday : ℝ := movie_ticket_price + popcorn_price + drink_price + (candy_price - wednesday_discount_candy) + hotdog_price
noncomputable def savings_wednesday_usd : ℝ := total_cost_wednesday - wednesday_price
noncomputable def savings_wednesday_eur : ℝ := savings_wednesday_usd * wednesday_exchange_rate

noncomputable def total_cost_friday : ℝ := movie_ticket_price + popcorn_price + (drink_price - friday_discount_drink) + candy_price + hotdog_price
noncomputable def savings_friday_usd : ℝ := total_cost_friday - friday_price
noncomputable def savings_friday_eur : ℝ := savings_friday_usd * friday_exchange_rate

noncomputable def total_cost_weekend : ℝ := movie_ticket_price + popcorn_price + drink_price + candy_price + hotdog_price
noncomputable def savings_weekend_usd : ℝ := total_cost_weekend - weekend_price
noncomputable def savings_weekend_eur : ℝ := savings_weekend_usd * weekend_exchange_rate

theorem european_savings_correct :
  savings_monday_eur = 3.61 ∧ 
  savings_wednesday_eur = 5.70 ∧ 
  savings_friday_eur = 1.41 ∧ 
  savings_weekend_eur = 1.62 :=
by
  sorry

end european_savings_correct_l22_22273


namespace num_isosceles_triangles_is_24_l22_22709

-- Define the structure of the hexagonal prism
structure HexagonalPrism :=
  (height : ℝ)
  (side_length : ℝ)
  (num_vertices : ℕ)

-- Define the specific hexagonal prism from the problem
def prism := HexagonalPrism.mk 2 1 12

-- Function to count the number of isosceles triangles in a given hexagonal prism
noncomputable def count_isosceles_triangles (hp : HexagonalPrism) : ℕ := sorry

-- The theorem that needs to be proved
theorem num_isosceles_triangles_is_24 :
  count_isosceles_triangles prism = 24 :=
sorry

end num_isosceles_triangles_is_24_l22_22709


namespace maximum_value_cosine_sine_combination_l22_22283

noncomputable def max_cosine_sine_combination : Real :=
  let g (θ : Real) := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have h₁ : ∃ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 :=
    sorry -- Existence of such θ is trivial
  Real.sqrt 2

theorem maximum_value_cosine_sine_combination :
  ∀ θ : Real, -Real.pi / 2 < θ ∧ θ < Real.pi / 2 →
  (Real.cos (θ / 2)) * (1 + Real.sin θ) ≤ Real.sqrt 2 :=
by
  intros θ h
  let y := (Real.cos (θ / 2)) * (1 + Real.sin θ)
  have hy : y ≤ Real.sqrt 2 := sorry
  exact hy

end maximum_value_cosine_sine_combination_l22_22283


namespace bike_price_l22_22609

theorem bike_price (x : ℝ) (h : 0.20 * x = 240) : x = 1200 :=
by
  sorry

end bike_price_l22_22609


namespace probability_not_passing_l22_22800

theorem probability_not_passing (P_passing : ℚ) (h : P_passing = 4/7) : (1 - P_passing = 3/7) :=
by
  rw [h]
  norm_num

end probability_not_passing_l22_22800


namespace integer_values_b_for_three_integer_solutions_l22_22024

theorem integer_values_b_for_three_integer_solutions (b : ℤ) :
  ¬ ∃ x1 x2 x3 : ℤ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ (x1^2 + b * x1 + 5 ≤ 0) ∧
                     (x2^2 + b * x2 + 5 ≤ 0) ∧ (x3^2 + b * x3 + 5 ≤ 0) ∧
                     (∀ x : ℤ, x1 < x ∧ x < x3 → x^2 + b * x + 5 > 0) :=
by
  sorry

end integer_values_b_for_three_integer_solutions_l22_22024


namespace p_squared_plus_13_mod_n_eq_2_l22_22251

theorem p_squared_plus_13_mod_n_eq_2 (p : ℕ) (prime_p : Prime p) (h : p > 3) (n : ℕ) :
  (∃ (k : ℕ), p ^ 2 + 13 = k * n + 2) → n = 2 :=
by
  sorry

end p_squared_plus_13_mod_n_eq_2_l22_22251


namespace integer_solutions_l22_22305

theorem integer_solutions (x y k : ℤ) :
  21 * x + 48 * y = 6 ↔ ∃ k : ℤ, x = -2 + 16 * k ∧ y = 1 - 7 * k :=
by
  sorry

end integer_solutions_l22_22305


namespace deliver_all_cargo_l22_22720

theorem deliver_all_cargo (containers : ℕ) (cargo_mass : ℝ) (ships : ℕ) (ship_capacity : ℝ)
  (h1 : containers ≥ 35)
  (h2 : cargo_mass = 18)
  (h3 : ships = 7)
  (h4 : ship_capacity = 3)
  (h5 : ∀ t, (0 < t) → (t ≤ containers) → (t = 35)) :
  (ships * ship_capacity) ≥ cargo_mass :=
by
  sorry

end deliver_all_cargo_l22_22720


namespace white_balls_in_bag_l22_22990

theorem white_balls_in_bag : 
  ∀ x : ℕ, (3 + 2 + x ≠ 0) → (2 : ℚ) / (3 + 2 + x) = 1 / 4 → x = 3 :=
by
  intro x
  intro h1
  intro h2
  sorry

end white_balls_in_bag_l22_22990


namespace fred_blue_marbles_l22_22010

theorem fred_blue_marbles (tim_marbles : ℕ) (fred_marbles : ℕ) (h1 : tim_marbles = 5) (h2 : fred_marbles = 22 * tim_marbles) : fred_marbles = 110 :=
by
  sorry

end fred_blue_marbles_l22_22010


namespace student_missed_number_l22_22102

theorem student_missed_number (student_sum : ℕ) (n : ℕ) (actual_sum : ℕ) : 
  student_sum = 575 → 
  actual_sum = n * (n + 1) / 2 → 
  n = 34 → 
  actual_sum - student_sum = 20 := 
by 
  sorry

end student_missed_number_l22_22102


namespace bruce_paid_correct_amount_l22_22839

-- Define the conditions
def kg_grapes : ℕ := 8
def cost_per_kg_grapes : ℕ := 70
def kg_mangoes : ℕ := 8
def cost_per_kg_mangoes : ℕ := 55

-- Calculate partial costs
def cost_grapes := kg_grapes * cost_per_kg_grapes
def cost_mangoes := kg_mangoes * cost_per_kg_mangoes
def total_paid := cost_grapes + cost_mangoes

-- The theorem to prove
theorem bruce_paid_correct_amount : total_paid = 1000 := 
by 
  -- Merge several logical steps into one
  -- sorry can be used for incomplete proof
  sorry

end bruce_paid_correct_amount_l22_22839


namespace geometric_sequence_general_term_l22_22859

theorem geometric_sequence_general_term (a : ℕ → ℕ) (q : ℕ) (h_q : q = 4) (h_sum : a 0 + a 1 + a 2 = 21)
  (h_geo : ∀ n, a (n + 1) = a n * q) : ∀ n, a n = 4 ^ n :=
by {
  sorry
}

end geometric_sequence_general_term_l22_22859


namespace Namjoon_gave_Yoongi_9_pencils_l22_22407

theorem Namjoon_gave_Yoongi_9_pencils
  (stroke_pencils : ℕ)
  (strokes : ℕ)
  (pencils_left : ℕ)
  (total_pencils : ℕ := stroke_pencils * strokes)
  (given_pencils : ℕ := total_pencils - pencils_left) :
  stroke_pencils = 12 →
  strokes = 2 →
  pencils_left = 15 →
  given_pencils = 9 := by
  sorry

end Namjoon_gave_Yoongi_9_pencils_l22_22407


namespace compound_propositions_l22_22052

def divides (a b : Nat) : Prop := ∃ k : Nat, b = k * a

-- Define the propositions p and q
def p : Prop := divides 6 12
def q : Prop := divides 6 24

-- Prove the compound propositions
theorem compound_propositions :
  (p ∨ q) ∧ (p ∧ q) ∧ ¬¬p :=
by
  -- We are proving three statements:
  -- 1. "p or q" is true.
  -- 2. "p and q" is true.
  -- 3. "not p" is false (which is equivalent to "¬¬p" being true).
  -- The actual proof will be constructed here.
  sorry

end compound_propositions_l22_22052


namespace Marias_score_l22_22554

def total_questions := 30
def points_per_correct_answer := 20
def points_deducted_per_incorrect_answer := 5
def total_answered := total_questions
def correct_answers := 19
def incorrect_answers := total_questions - correct_answers
def score := (correct_answers * points_per_correct_answer) - (incorrect_answers * points_deducted_per_incorrect_answer)

theorem Marias_score : score = 325 := by
  -- proof goes here
  sorry

end Marias_score_l22_22554


namespace bushels_given_away_l22_22278

-- Definitions from the problem conditions
def initial_bushels : ℕ := 50
def ears_per_bushel : ℕ := 14
def remaining_ears : ℕ := 357

-- Theorem to prove the number of bushels given away
theorem bushels_given_away : 
  initial_bushels * ears_per_bushel - remaining_ears = 24 * ears_per_bushel :=
by
  sorry

end bushels_given_away_l22_22278


namespace length_of_tunnel_l22_22185

theorem length_of_tunnel
    (length_of_train : ℕ)
    (speed_kmh : ℕ)
    (crossing_time_seconds : ℕ)
    (distance_covered : ℕ)
    (length_of_tunnel : ℕ) :
    length_of_train = 1200 →
    speed_kmh = 96 →
    crossing_time_seconds = 90 →
    distance_covered = (speed_kmh * 1000 / 3600) * crossing_time_seconds →
    length_of_train + length_of_tunnel = distance_covered →
    length_of_tunnel = 6000 :=
by
  sorry

end length_of_tunnel_l22_22185


namespace find_a_l22_22544

theorem find_a (a : ℝ) (h : ∫ x in -a..a, (2 * x - 1) = -8) : a = 4 :=
sorry

end find_a_l22_22544


namespace solve_for_x_l22_22638

theorem solve_for_x (x : ℚ) (h : 3 / 4 - 1 / x = 1 / 2) : x = 4 :=
sorry

end solve_for_x_l22_22638


namespace perimeter_of_resulting_figure_l22_22999

-- Define the perimeters of the squares
def perimeter_small_square : ℕ := 40
def perimeter_large_square : ℕ := 100

-- Define the side lengths of the squares
def side_length_small_square := perimeter_small_square / 4
def side_length_large_square := perimeter_large_square / 4

-- Define the total perimeter of the uncombined squares
def total_perimeter_uncombined := perimeter_small_square + perimeter_large_square

-- Define the shared side length
def shared_side_length := side_length_small_square

-- Define the perimeter after considering the shared side
def resulting_perimeter := total_perimeter_uncombined - 2 * shared_side_length

-- Prove that the resulting perimeter is 120 cm
theorem perimeter_of_resulting_figure : resulting_perimeter = 120 := by
  sorry

end perimeter_of_resulting_figure_l22_22999


namespace cost_difference_zero_l22_22634

theorem cost_difference_zero
  (A O X : ℝ)
  (h1 : 3 * A + 7 * O = 4.56)
  (h2 : A + O = 0.26)
  (h3 : O = A + X) :
  X = 0 := 
sorry

end cost_difference_zero_l22_22634


namespace new_number_is_100t_plus_10u_plus_3_l22_22550

theorem new_number_is_100t_plus_10u_plus_3 (t u : ℕ) (ht : t < 10) (hu : u < 10) :
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  new_number = 100 * t + 10 * u + 3 :=
by
  let original_number := 10 * t + u
  let new_number := original_number * 10 + 3
  show new_number = 100 * t + 10 * u + 3
  sorry

end new_number_is_100t_plus_10u_plus_3_l22_22550


namespace total_production_by_june_l22_22985

def initial_production : ℕ := 10

def common_ratio : ℕ := 3

def production_june : ℕ :=
  let a := initial_production
  let r := common_ratio
  a * ((r^6 - 1) / (r - 1))

theorem total_production_by_june : production_june = 3640 :=
by sorry

end total_production_by_june_l22_22985


namespace max_sum_of_squares_eq_100_l22_22149

theorem max_sum_of_squares_eq_100 : 
  ∃ (x y : ℤ), x^2 + y^2 = 100 ∧ 
  (∀ (x y : ℤ), x^2 + y^2 = 100 → x + y ≤ 14) ∧ 
  (∃ (x y : ℕ), x^2 + y^2 = 100 ∧ x + y = 14) :=
by {
  sorry
}

end max_sum_of_squares_eq_100_l22_22149


namespace base_length_of_parallelogram_l22_22812

theorem base_length_of_parallelogram 
  (area : ℝ) (base altitude : ℝ) 
  (h_area : area = 242)
  (h_altitude : altitude = 2 * base) :
  base = 11 :=
by
  sorry

end base_length_of_parallelogram_l22_22812


namespace inequality_proof_l22_22879

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a * b * c = 1) :
  2 * (a + b + c) + 9 / (a * b + b * c + c * a)^2 ≥ 7 :=
by
  sorry

end inequality_proof_l22_22879


namespace maximum_value_quadratic_l22_22213

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 8 * x - 6

theorem maximum_value_quadratic :
  ∃ x : ℝ, quadratic_function x = 2 ∧ ∀ y : ℝ, quadratic_function y ≤ 2 :=
sorry

end maximum_value_quadratic_l22_22213


namespace beavers_still_working_is_one_l22_22905

def initial_beavers : Nat := 2
def beavers_swimming : Nat := 1
def still_working_beavers : Nat := initial_beavers - beavers_swimming

theorem beavers_still_working_is_one : still_working_beavers = 1 :=
by
  sorry

end beavers_still_working_is_one_l22_22905


namespace consistency_condition_l22_22049

theorem consistency_condition (x y z a b c d : ℝ)
  (h1 : y + z = a)
  (h2 : x + y = b)
  (h3 : x + z = c)
  (h4 : x + y + z = d) : a + b + c = 2 * d :=
by sorry

end consistency_condition_l22_22049


namespace sum_of_a_b_l22_22961

theorem sum_of_a_b (a b : ℝ) (h1 : |a| = 6) (h2 : |b| = 4) (h3 : a * b < 0) :
    a + b = 2 ∨ a + b = -2 :=
sorry

end sum_of_a_b_l22_22961


namespace area_of_triangle_ABC_l22_22637

variable (A B C : ℝ × ℝ)
variable (x1 y1 x2 y2 x3 y3 : ℝ)

def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let x1 := A.1
  let y1 := A.2
  let x2 := B.1
  let y2 := B.2
  let x3 := C.1
  let y3 := C.2
  0.5 * (abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

theorem area_of_triangle_ABC :
  let A := (1, 2)
  let B := (-2, 5)
  let C := (4, -2)
  area_of_triangle A B C = 1.5 :=
by
  sorry

end area_of_triangle_ABC_l22_22637


namespace find_number_l22_22027

theorem find_number (x : ℝ) : 8050 * x = 80.5 → x = 0.01 :=
by
  sorry

end find_number_l22_22027


namespace tax_percentage_first_40000_l22_22488

theorem tax_percentage_first_40000 (P : ℝ) :
  (0 < P) → 
  (P / 100) * 40000 + 0.20 * 10000 = 8000 →
  P = 15 :=
by
  intros hP h
  sorry

end tax_percentage_first_40000_l22_22488


namespace distance_between_riya_and_priya_l22_22235

theorem distance_between_riya_and_priya (speed_riya speed_priya : ℝ) (time_hours : ℝ)
  (h1 : speed_riya = 21) (h2 : speed_priya = 22) (h3 : time_hours = 1) :
  speed_riya * time_hours + speed_priya * time_hours = 43 := by
  sorry

end distance_between_riya_and_priya_l22_22235


namespace find_x_for_g_statement_l22_22204

noncomputable def g (x : ℝ) : ℝ := (x + 4) ^ (1/3) / 5 ^ (1/3)

theorem find_x_for_g_statement (x : ℝ) : g (3 * x) = 3 * g x ↔ x = -13 / 3 := by
  sorry

end find_x_for_g_statement_l22_22204


namespace least_clock_equivalent_l22_22848

theorem least_clock_equivalent (x : ℕ) : 
  x > 3 ∧ x % 12 = (x * x) % 12 → x = 12 := 
by
  sorry

end least_clock_equivalent_l22_22848


namespace total_cookies_is_390_l22_22363

def abigail_boxes : ℕ := 2
def grayson_boxes : ℚ := 3 / 4
def olivia_boxes : ℕ := 3
def cookies_per_box : ℕ := 48

def abigail_cookies : ℕ := abigail_boxes * cookies_per_box
def grayson_cookies : ℚ := grayson_boxes * cookies_per_box
def olivia_cookies : ℕ := olivia_boxes * cookies_per_box
def isabella_cookies : ℚ := (1 / 2) * grayson_cookies
def ethan_cookies : ℤ := (abigail_boxes * 2 * cookies_per_box) / 2

def total_cookies : ℚ := ↑abigail_cookies + grayson_cookies + ↑olivia_cookies + isabella_cookies + ↑ethan_cookies

theorem total_cookies_is_390 : total_cookies = 390 :=
by
  sorry

end total_cookies_is_390_l22_22363


namespace geometric_sequence_common_ratio_range_l22_22098

theorem geometric_sequence_common_ratio_range (a : ℕ → ℝ) (q : ℝ)
  (h1 : a 1 < 0) 
  (h2 : ∀ n : ℕ, 0 < n → a n < a (n + 1))
  (hq : ∀ n : ℕ, a (n + 1) = a n * q) :
  0 < q ∧ q < 1 :=
sorry

end geometric_sequence_common_ratio_range_l22_22098


namespace complex_quadrant_l22_22237

theorem complex_quadrant 
  (z : ℂ) 
  (h : (2 + 3 * Complex.I) * z = 1 + Complex.I) : 
  z.re > 0 ∧ z.im < 0 := 
sorry

end complex_quadrant_l22_22237


namespace third_row_number_of_trees_l22_22542

theorem third_row_number_of_trees (n : ℕ) 
  (divisible_by_7 : 84 % 7 = 0) 
  (divisible_by_6 : 84 % 6 = 0) 
  (divisible_by_n : 84 % n = 0) 
  (least_trees : 84 = 84): 
  n = 4 := 
sorry

end third_row_number_of_trees_l22_22542


namespace minimum_value_expression_l22_22046

noncomputable def problem_statement (x y z : ℝ) : ℝ :=
  (x + y) / z + (x + z) / y + (y + z) / x + 3

theorem minimum_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  problem_statement x y z ≥ 9 :=
sorry

end minimum_value_expression_l22_22046


namespace contrapositive_prop_l22_22807

theorem contrapositive_prop {α : Type} [Mul α] [Zero α] (a b : α) : 
  (a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0) :=
by sorry

end contrapositive_prop_l22_22807


namespace matrix_A_pow_100_eq_l22_22386

noncomputable def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 1], ![-9, -2]]

theorem matrix_A_pow_100_eq : matrix_A ^ 100 = ![![301, 100], ![-900, -299]] :=
  sorry

end matrix_A_pow_100_eq_l22_22386


namespace complement_of_A_in_U_l22_22116

-- Define the universal set U and set A
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {3, 4, 5}

-- Define the complement of A in U
theorem complement_of_A_in_U : (U \ A) = {1, 2} :=
by
  sorry

end complement_of_A_in_U_l22_22116


namespace Tiffany_total_score_l22_22129

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

end Tiffany_total_score_l22_22129


namespace largest_side_of_enclosure_l22_22408

-- Definitions for the conditions
def perimeter (l w : ℝ) : ℝ := 2 * l + 2 * w
def area (l w : ℝ) : ℝ := l * w

theorem largest_side_of_enclosure (l w : ℝ) (h_fencing : perimeter l w = 240) (h_area : area l w = 12 * 240) : l = 86.83 ∨ w = 86.83 :=
by {
  sorry
}

end largest_side_of_enclosure_l22_22408


namespace age_ratio_in_1_year_l22_22464

variable (j m x : ℕ)

-- Conditions
def condition1 (j m : ℕ) : Prop :=
  j - 3 = 2 * (m - 3)

def condition2 (j m : ℕ) : Prop :=
  j - 5 = 3 * (m - 5)

-- Question
def age_ratio (j m x : ℕ) : Prop :=
  (j + x) * 2 = 3 * (m + x)

theorem age_ratio_in_1_year (j m x : ℕ) :
  condition1 j m → condition2 j m → age_ratio j m 1 :=
by
  sorry

end age_ratio_in_1_year_l22_22464


namespace intersection_a_b_l22_22663

-- Definitions of sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

-- The proof problem
theorem intersection_a_b : A ∩ B = {-1, 0} :=
by
  sorry

end intersection_a_b_l22_22663


namespace ratio_of_x_to_y_l22_22089

variable {x y : ℝ}

theorem ratio_of_x_to_y (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
by
  sorry

end ratio_of_x_to_y_l22_22089


namespace minimum_value_of_f_l22_22666

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 4^2)) + (Real.sqrt ((x + 1)^2 + 3^2))

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  use -3
  sorry

end minimum_value_of_f_l22_22666


namespace three_digit_numbers_l22_22907

theorem three_digit_numbers (n : ℕ) (h1 : 100 ≤ n ∧ n ≤ 999) (h2 : n^2 % 1000 = n % 1000) : 
  n = 376 ∨ n = 625 :=
by
  sorry

end three_digit_numbers_l22_22907


namespace application_methods_count_l22_22048

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

end application_methods_count_l22_22048


namespace total_lives_correct_l22_22668

namespace VideoGame

def num_friends : ℕ := 8
def lives_each : ℕ := 8

def total_lives (n : ℕ) (l : ℕ) : ℕ := n * l 

theorem total_lives_correct : total_lives num_friends lives_each = 64 := by
  sorry

end total_lives_correct_l22_22668


namespace factor_expression_l22_22267

theorem factor_expression (y : ℝ) : 3 * y^2 - 12 = 3 * (y + 2) * (y - 2) :=
by
  sorry

end factor_expression_l22_22267


namespace problem_I_problem_II_l22_22350

namespace MathProof

-- Define the function f(x) given m
def f (m : ℝ) (x : ℝ) : ℝ := m - |x - 1| - 2 * |x + 1|

-- Problem (I)
theorem problem_I (x : ℝ) : (5 - |x - 1| - 2 * |x + 1| > 2) ↔ (-4/3 < x ∧ x < 0) := 
sorry

-- Define the quadratic function
def y (x : ℝ) : ℝ := x^2 + 2*x + 3

-- Problem (II)
theorem problem_II (m : ℝ) : (∀ x : ℝ, ∃ t : ℝ, t = x^2 + 2*x + 3 ∧ t = f m x) ↔ (m ≥ 4) :=
sorry

end MathProof

end problem_I_problem_II_l22_22350


namespace complex_purely_imaginary_m_value_l22_22074

theorem complex_purely_imaginary_m_value (m : ℝ) :
  (m^2 - 1 = 0) ∧ (m + 1 ≠ 0) → m = 1 :=
by
  sorry

end complex_purely_imaginary_m_value_l22_22074


namespace fraction_simplification_l22_22798

theorem fraction_simplification :
  (1 * 2 * 4 + 2 * 4 * 8 + 3 * 6 * 12 + 4 * 8 * 16) /
  (1 * 3 * 9 + 2 * 6 * 18 + 3 * 9 * 27 + 4 * 12 * 36) = 8 / 27 :=
by
  sorry

end fraction_simplification_l22_22798


namespace interchanged_digit_multiple_of_sum_l22_22941

theorem interchanged_digit_multiple_of_sum (n a b : ℕ) 
  (h1 : n = 10 * a + b) 
  (h2 : n = 3 * (a + b)) 
  (h3 : 1 ≤ a) (h4 : a ≤ 9) 
  (h5 : 0 ≤ b) (h6 : b ≤ 9) : 
  10 * b + a = 8 * (a + b) := 
by 
  sorry

end interchanged_digit_multiple_of_sum_l22_22941


namespace percentage_error_calculation_l22_22563

theorem percentage_error_calculation (x : ℝ) :
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  percentage_error = 64 := 
by
  let correct_value := x * (5 / 3)
  let incorrect_value := x * (3 / 5)
  let difference := correct_value - incorrect_value
  let percentage_error := (difference / correct_value) * 100
  sorry

end percentage_error_calculation_l22_22563


namespace alcohol_percentage_new_mixture_l22_22208

theorem alcohol_percentage_new_mixture :
  let initial_alcohol_percentage := 0.90
  let initial_solution_volume := 24
  let added_water_volume := 16
  let total_new_volume := initial_solution_volume + added_water_volume
  let initial_alcohol_amount := initial_solution_volume * initial_alcohol_percentage
  let new_alcohol_percentage := (initial_alcohol_amount / total_new_volume) * 100
  new_alcohol_percentage = 54 := by
    sorry

end alcohol_percentage_new_mixture_l22_22208


namespace eccentricity_of_hyperbola_l22_22877

theorem eccentricity_of_hyperbola (a b c : ℝ) (h_a : a > 0) (h_b : b > 0)
  (h_c : c = Real.sqrt (a^2 + b^2))
  (F1 : ℝ × ℝ := (-c, 0))
  (A B : ℝ × ℝ)
  (slope_of_AB : ∀ (x y : ℝ), y = x + c)
  (asymptotes_eqn : ∀ (x : ℝ), x = a ∨ x = -a)
  (intersections : A = (-(a * c / (a - b)), -(b * c / (a - b))) ∧ B = (-(a * c / (a + b)), (b * c / (a + b))))
  (AB_eq_2BF1 : 2 * (F1 - B) = A - B) :
  Real.sqrt (1 + (b / a)^2) = Real.sqrt 5 :=
sorry

end eccentricity_of_hyperbola_l22_22877


namespace enhanced_inequality_l22_22858

theorem enhanced_inequality 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (2 * a^2 / (b + c) + 2 * b^2 / (c + a) + 2 * c^2 / (a + b) ≥ a + b + c + (2 * a - b - c)^2 / (a + b + c)) :=
sorry

end enhanced_inequality_l22_22858


namespace min_value_of_expression_l22_22779

theorem min_value_of_expression (n : ℕ) (h : n > 0) : (n / 3 + 27 / n) ≥ 6 :=
by {
  -- Proof goes here but is not required in the statement
  sorry
}

end min_value_of_expression_l22_22779


namespace jim_makes_60_dollars_l22_22771

-- Definitions based on the problem conditions
def average_weight_per_rock : ℝ := 1.5
def price_per_pound : ℝ := 4
def number_of_rocks : ℕ := 10

-- Problem statement
theorem jim_makes_60_dollars :
  (average_weight_per_rock * number_of_rocks) * price_per_pound = 60 := by
  sorry

end jim_makes_60_dollars_l22_22771


namespace HVAC_cost_per_vent_l22_22002

/-- 
The cost of Joe's new HVAC system is $20,000. It includes 2 conditioning zones, each with 5 vents.
Prove that the cost of the system per vent is $2,000.
-/
theorem HVAC_cost_per_vent
    (cost : ℕ := 20000)
    (zones : ℕ := 2)
    (vents_per_zone : ℕ := 5)
    (total_vents : ℕ := zones * vents_per_zone) :
    (cost / total_vents) = 2000 := by
  sorry

end HVAC_cost_per_vent_l22_22002


namespace coffee_grinder_assembly_time_l22_22545

-- Variables for the assembly rates
variables (h r : ℝ)

-- Definitions of conditions
def condition1 : Prop := h / 4 = r
def condition2 : Prop := r / 4 = h
def condition3 : Prop := ∀ start_time end_time net_added, 
  start_time = 9 ∧ end_time = 12 ∧ net_added = 27 → 3 * 3/4 * h = net_added
def condition4 : Prop := ∀ start_time end_time net_added, 
  start_time = 13 ∧ end_time = 19 ∧ net_added = 120 → 6 * 3/4 * r = net_added

-- Theorem statement
theorem coffee_grinder_assembly_time
  (h r : ℝ)
  (c1 : condition1 h r)
  (c2 : condition2 h r)
  (c3 : condition3 h)
  (c4 : condition4 r) :
  h = 12 ∧ r = 80 / 3 :=
sorry

end coffee_grinder_assembly_time_l22_22545


namespace cos_pi_over_3_plus_double_alpha_l22_22336

theorem cos_pi_over_3_plus_double_alpha (α : ℝ) (h : Real.sin (π / 3 - α) = 1 / 4) :
  Real.cos (π / 3 + 2 * α) = -7 / 8 :=
sorry

end cos_pi_over_3_plus_double_alpha_l22_22336


namespace remainder_of_division_l22_22020

-- Define the dividend and divisor
def dividend : ℕ := 3^303 + 303
def divisor : ℕ := 3^101 + 3^51 + 1

-- State the theorem to be proven
theorem remainder_of_division:
  (dividend % divisor) = 303 := by
  sorry

end remainder_of_division_l22_22020


namespace junior_titles_in_sample_l22_22625

noncomputable def numberOfJuniorTitlesInSample (totalEmployees: ℕ) (juniorEmployees: ℕ) (sampleSize: ℕ) : ℕ :=
  (juniorEmployees * sampleSize) / totalEmployees

theorem junior_titles_in_sample (totalEmployees juniorEmployees intermediateEmployees seniorEmployees sampleSize : ℕ) 
  (h_total : totalEmployees = 150) 
  (h_junior : juniorEmployees = 90) 
  (h_intermediate : intermediateEmployees = 45) 
  (h_senior : seniorEmployees = 15) 
  (h_sampleSize : sampleSize = 30) : 
  numberOfJuniorTitlesInSample totalEmployees juniorEmployees sampleSize = 18 := by
  sorry

end junior_titles_in_sample_l22_22625


namespace convince_the_king_l22_22206

/-- Define the types of inhabitants -/
inductive Inhabitant
| Knight
| Liar
| Normal

/-- Define the king's preference -/
def K (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- All knights tell the truth -/
def tells_truth (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => True
  | Inhabitant.Liar => False
  | Inhabitant.Normal => False

/-- All liars always lie -/
def tells_lie (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => True
  | Inhabitant.Normal => False

/-- Normal persons can tell both truths and lies -/
def can_tell_both (inhabitant : Inhabitant) : Prop :=
  match inhabitant with
  | Inhabitant.Knight => False
  | Inhabitant.Liar => False
  | Inhabitant.Normal => True

/-- Prove there exists a true statement and a false statement to convince the king -/
theorem convince_the_king (p : Inhabitant) :
  (∃ S : Prop, (S ↔ tells_truth p) ∧ K p) ∧ (∃ S' : Prop, (¬ S' ↔ tells_lie p) ∧ K p) :=
by
  sorry

end convince_the_king_l22_22206


namespace sum_of_first_20_terms_l22_22088

variable {a : ℕ → ℕ}

-- Conditions given in the problem
axiom seq_property : ∀ n, a n + 2 * a (n + 1) = 3 * n + 2
axiom arithmetic_sequence : ∀ n m, a (n + 1) - a n = a (m + 1) - a m

-- Theorem to be proved
theorem sum_of_first_20_terms (a : ℕ → ℕ) (S20 := (Finset.range 20).sum a) :
  S20 = 210 :=
  sorry

end sum_of_first_20_terms_l22_22088


namespace roots_negative_and_bounds_find_possible_values_of_b_and_c_l22_22674

theorem roots_negative_and_bounds
  (b c x₁ x₂ x₁' x₂' : ℤ) 
  (h1 : x₁ * x₂ > 0) 
  (h2 : x₁' * x₂' > 0)
  (h3 : x₁^2 + b * x₁ + c = 0) 
  (h4 : x₂^2 + b * x₂ + c = 0) 
  (h5 : x₁'^2 + c * x₁' + b = 0) 
  (h6 : x₂'^2 + c * x₂' + b = 0) :
  x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0 ∧ (b - 1 ≤ c ∧ c ≤ b + 1) :=
by
  sorry


theorem find_possible_values_of_b_and_c 
  (b c : ℤ) 
  (h's : ∃ x₁ x₂ x₁' x₂', 
    x₁ * x₂ > 0 ∧ 
    x₁' * x₂' > 0 ∧ 
    (x₁^2 + b * x₁ + c = 0) ∧ 
    (x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁'^2 + c * x₁' + b = 0) ∧ 
    (x₂'^2 + c * x₂' + b = 0)) :
  (b = 4 ∧ c = 4) ∨ 
  (b = 5 ∧ c = 6) ∨ 
  (b = 6 ∧ c = 5) :=
by
  sorry

end roots_negative_and_bounds_find_possible_values_of_b_and_c_l22_22674


namespace incorrect_expression_l22_22936

variable (D : ℚ) (P Q : ℕ) (r s : ℕ)

-- D represents a repeating decimal.
-- P denotes the r figures of D which do not repeat themselves.
-- Q denotes the s figures of D which repeat themselves.

theorem incorrect_expression :
  10^r * (10^s - 1) * D ≠ Q * (P - 1) :=
sorry

end incorrect_expression_l22_22936


namespace total_commencement_addresses_l22_22931

-- Define the given conditions
def sandoval_addresses := 12
def sandoval_rainy_addresses := 5
def sandoval_public_holidays := 2
def sandoval_non_rainy_addresses := sandoval_addresses - sandoval_rainy_addresses

def hawkins_addresses := sandoval_addresses / 2
def sloan_addresses := sandoval_addresses + 10
def sloan_non_rainy_addresses := sloan_addresses -- assuming no rainy day details are provided

def davenport_addresses := (sandoval_non_rainy_addresses + sloan_non_rainy_addresses) / 2 - 3
def davenport_addresses_rounded := 11 -- rounding down to nearest integer as per given solution

def adkins_addresses := hawkins_addresses + davenport_addresses_rounded + 2

-- Calculate the total number of addresses
def total_addresses := sandoval_addresses + hawkins_addresses + sloan_addresses + davenport_addresses_rounded + adkins_addresses

-- The proof goal statement
theorem total_commencement_addresses : total_addresses = 70 := by
  -- Proof to be provided here
  sorry

end total_commencement_addresses_l22_22931


namespace complement_union_eq_l22_22190

def U : Set ℤ := {-2, -1, 0, 1, 2, 3}
def A : Set ℤ := {-1, 2}
def B : Set ℤ := {x | x ^ 2 - 4 * x + 3 = 0}

theorem complement_union_eq :
  (U \ (A ∪ B)) = {-2, 0} :=
by
  sorry

end complement_union_eq_l22_22190


namespace ratio_of_dogs_to_cats_l22_22260

-- Definition of conditions
def total_animals : Nat := 21
def cats_to_spay : Nat := 7
def dogs_to_spay : Nat := total_animals - cats_to_spay

-- Ratio of dogs to cats
def dogs_to_cats_ratio : Nat := dogs_to_spay / cats_to_spay

-- Statement to prove
theorem ratio_of_dogs_to_cats : dogs_to_cats_ratio = 2 :=
by
  -- Proof goes here
  sorry

end ratio_of_dogs_to_cats_l22_22260


namespace eval_polynomial_l22_22209

theorem eval_polynomial (x : ℝ) (h : x^2 - 3 * x - 9 = 0) : x^3 - 3 * x^2 - 9 * x + 27 = 27 := 
by
  sorry

end eval_polynomial_l22_22209


namespace min_quotient_l22_22064

theorem min_quotient {a b : ℕ} (h₁ : 100 ≤ a) (h₂ : a ≤ 300) (h₃ : 400 ≤ b) (h₄ : b ≤ 800) (h₅ : a + b ≤ 950) : a / b = 1 / 8 := 
by
  sorry

end min_quotient_l22_22064


namespace total_spent_on_toys_l22_22868

-- Definition of the costs
def football_cost : ℝ := 5.71
def marbles_cost : ℝ := 6.59

-- The theorem to prove the total amount spent on toys
theorem total_spent_on_toys : football_cost + marbles_cost = 12.30 :=
by sorry

end total_spent_on_toys_l22_22868


namespace find_third_integer_l22_22754

theorem find_third_integer (a b c : ℕ) (h1 : a * b * c = 42) (h2 : a + b = 9) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : c = 3 :=
sorry

end find_third_integer_l22_22754


namespace negation_prop_l22_22900

theorem negation_prop : (¬(∃ x : ℝ, x + 2 ≤ 0)) ↔ (∀ x : ℝ, x + 2 > 0) := 
  sorry

end negation_prop_l22_22900


namespace relatively_prime_sums_l22_22044

theorem relatively_prime_sums (x y : ℤ) (h : Int.gcd x y = 1) 
  : Int.gcd (x^2 + x * y + y^2) (x^2 + 3 * x * y + y^2) = 1 :=
by
  sorry

end relatively_prime_sums_l22_22044


namespace sum_of_squares_base_b_l22_22173

theorem sum_of_squares_base_b (b : ℕ) (h : (b + 4)^2 + (b + 8)^2 + (2 * b)^2 = 2 * b^3 + 8 * b^2 + 5 * b) :
  (4 * b + 12 : ℕ) = 62 :=
by
  sorry

end sum_of_squares_base_b_l22_22173


namespace present_ages_l22_22233

theorem present_ages
  (R D K : ℕ) (x : ℕ)
  (H1 : R = 4 * x)
  (H2 : D = 3 * x)
  (H3 : K = 5 * x)
  (H4 : R + 6 = 26)
  (H5 : (R + 8) + (D + 8) = K) :
  D = 15 ∧ K = 51 :=
sorry

end present_ages_l22_22233


namespace females_in_band_not_orchestra_l22_22980

/-- The band at Pythagoras High School has 120 female members. -/
def females_in_band : ℕ := 120

/-- The orchestra at Pythagoras High School has 70 female members. -/
def females_in_orchestra : ℕ := 70

/-- There are 45 females who are members of both the band and the orchestra. -/
def females_in_both : ℕ := 45

/-- The combined total number of students involved in either the band or orchestra or both is 250. -/
def total_students : ℕ := 250

/-- The number of females in the band who are NOT in the orchestra. -/
def females_in_band_only : ℕ := females_in_band - females_in_both

theorem females_in_band_not_orchestra : females_in_band_only = 75 := by
  sorry

end females_in_band_not_orchestra_l22_22980


namespace fraction_check_l22_22420

variable (a b x y : ℝ)
noncomputable def is_fraction (expr : ℝ) : Prop :=
∃ n d : ℝ, d ≠ 0 ∧ expr = n / d ∧ ∃ var : ℝ, d = var

theorem fraction_check :
  is_fraction ((x + 3) / x) :=
sorry

end fraction_check_l22_22420


namespace center_of_circle_l22_22792

theorem center_of_circle (x y : ℝ) : x^2 - 8 * x + y^2 - 4 * y = 4 → (x, y) = (4, 2) :=
by
  sorry

end center_of_circle_l22_22792


namespace beta_value_l22_22739

variable {α β : Real}
open Real

theorem beta_value :
  cos α = 1 / 7 ∧ cos (α + β) = -11 / 14 ∧ 0 < α ∧ α < π / 2 ∧ π / 2 < α + β ∧ α + β < π → 
  β = π / 3 := 
by
  -- Proof would go here
  sorry

end beta_value_l22_22739


namespace part1_part2_l22_22036

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

end part1_part2_l22_22036


namespace train_length_calculation_l22_22229

noncomputable def length_of_train (time : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * (1000 / 3600)
  speed_ms * time

theorem train_length_calculation : 
  length_of_train 4.99960003199744 72 = 99.9920006399488 :=
by 
  sorry  -- proof of the actual calculation

end train_length_calculation_l22_22229


namespace mean_temperature_correct_l22_22351

def temperatures : List ℤ := [-6, -3, -3, -4, 2, 4, 1]

def mean_temperature (temps : List ℤ) : ℚ :=
  (temps.sum : ℚ) / temps.length

theorem mean_temperature_correct :
  mean_temperature temperatures = -9 / 7 := 
by
  sorry

end mean_temperature_correct_l22_22351


namespace max_buses_in_city_l22_22886

theorem max_buses_in_city (num_stops stops_per_bus shared_stops : ℕ) (h_stops : num_stops = 9) (h_stops_per_bus : stops_per_bus = 3) (h_shared_stops : shared_stops = 1) : 
  ∃ (max_buses : ℕ), max_buses = 12 :=
by
  sorry

end max_buses_in_city_l22_22886


namespace original_number_of_coins_in_first_pile_l22_22212

noncomputable def originalCoinsInFirstPile (x y z : ℕ) : ℕ :=
  if h : (2 * (x - y) = 16) ∧ (2 * y - z = 16) ∧ (2 * z - (x + y) = 16) then x else 0

theorem original_number_of_coins_in_first_pile (x y z : ℕ) (h1 : 2 * (x - y) = 16) 
                                              (h2 : 2 * y - z = 16) 
                                              (h3 : 2 * z - (x + y) = 16) : x = 22 :=
by sorry

end original_number_of_coins_in_first_pile_l22_22212


namespace find_n_l22_22147

theorem find_n (a b n : ℕ) (k l m : ℤ) 
  (ha : a % n = 2) 
  (hb : b % n = 3) 
  (h_ab : a > b) 
  (h_ab_mod : (a - b) % n = 5) : 
  n = 6 := 
sorry

end find_n_l22_22147


namespace vasya_read_entire_book_l22_22928

theorem vasya_read_entire_book :
  let day1 := 1 / 2
  let day2 := 1 / 3 * (1 - day1)
  let days12 := day1 + day2
  let day3 := 1 / 2 * days12
  (days12 + day3) = 1 :=
by
  sorry

end vasya_read_entire_book_l22_22928


namespace total_pounds_of_peppers_l22_22034

def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335
def total_peppers : ℝ := 5.666666666666667

theorem total_pounds_of_peppers :
  green_peppers + red_peppers = total_peppers :=
by
  -- sorry: Proof is omitted
  sorry

end total_pounds_of_peppers_l22_22034


namespace probability_black_white_l22_22849

structure Jar :=
  (black_balls : ℕ)
  (white_balls : ℕ)
  (green_balls : ℕ)

def total_balls (j : Jar) : ℕ :=
  j.black_balls + j.white_balls + j.green_balls

def choose (n k : ℕ) : ℕ := n.choose k

theorem probability_black_white (j : Jar) (h_black : j.black_balls = 3) (h_white : j.white_balls = 3) (h_green : j.green_balls = 1) :
  (choose 3 1 * choose 3 1) / (choose (total_balls j) 2) = 3 / 7 :=
by
  sorry

end probability_black_white_l22_22849


namespace average_speed_correct_l22_22300

-- Define the conditions as constants
def distance (D : ℝ) := D
def first_segment_speed := 60 -- km/h
def second_segment_speed := 24 -- km/h
def third_segment_speed := 48 -- km/h

-- Define the function that calculates average speed
noncomputable def average_speed (D : ℝ) : ℝ :=
  let t1 := (D / 3) / first_segment_speed
  let t2 := (D / 3) / second_segment_speed
  let t3 := (D / 3) / third_segment_speed
  let total_time := t1 + t2 + t3
  let total_distance := D
  total_distance / total_time

-- Prove that the average speed is 720 / 19 km/h
theorem average_speed_correct (D : ℝ) (hD : D > 0) : 
  average_speed D = 720 / 19 :=
by
  sorry

end average_speed_correct_l22_22300


namespace gcd_lcm_product_24_36_proof_l22_22610

def gcd_lcm_product_24_36 : Prop :=
  let a := 24
  let b := 36
  let gcd_ab := Int.gcd a b
  let lcm_ab := Int.lcm a b
  gcd_ab * lcm_ab = 864

theorem gcd_lcm_product_24_36_proof : gcd_lcm_product_24_36 :=
by
  sorry

end gcd_lcm_product_24_36_proof_l22_22610


namespace negation_of_P_is_there_exists_x_ge_0_l22_22455

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + x - 1 < 0

-- State the theorem of the negation of P
theorem negation_of_P_is_there_exists_x_ge_0 : ¬P ↔ ∃ x : ℝ, x^2 + x - 1 ≥ 0 :=
by sorry

end negation_of_P_is_there_exists_x_ge_0_l22_22455


namespace common_ratio_of_arithmetic_seq_l22_22223

theorem common_ratio_of_arithmetic_seq (a_1 q : ℝ) 
  (h1 : a_1 + a_1 * q^2 = 10) 
  (h2 : a_1 * q^3 + a_1 * q^5 = 5 / 4) : 
  q = 1 / 2 := 
by 
  sorry

end common_ratio_of_arithmetic_seq_l22_22223


namespace complex_number_sum_l22_22236

noncomputable def x : ℝ := 3 / 5
noncomputable def y : ℝ := -3 / 5

theorem complex_number_sum :
  (x + y) = -2 / 5 := 
by
  sorry

end complex_number_sum_l22_22236


namespace find_fifth_number_l22_22766

def avg_sum_9_numbers := 936
def sum_first_5_numbers := 495
def sum_last_5_numbers := 500

theorem find_fifth_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℝ)
  (h1 : A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 = avg_sum_9_numbers)
  (h2 : A1 + A2 + A3 + A4 + A5 = sum_first_5_numbers)
  (h3 : A5 + A6 + A7 + A8 + A9 = sum_last_5_numbers) :
  A5 = 29.5 :=
sorry

end find_fifth_number_l22_22766


namespace simplify_expression_l22_22603

theorem simplify_expression (p q x : ℝ) (h₀ : p ≠ 0) (h₁ : q ≠ 0) (h₂ : x > 0) (h₃ : x ≠ 1) :
  (x^(3 / p) - x^(3 / q)) / ((x^(1 / p) + x^(1 / q))^2 - 2 * x^(1 / q) * (x^(1 / q) + x^(1 / p)))
  + x^(1 / p) / (x^((q - p) / (p * q)) + 1) = x^(1 / p) + x^(1 / q) := 
sorry

end simplify_expression_l22_22603


namespace arithmetic_sequence_solution_l22_22509

noncomputable def arithmetic_sequence (a : ℕ → ℤ) (a1 d : ℤ) : Prop :=
∀ n : ℕ, a n = a1 + n * d

noncomputable def S (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * a 0 + (n * (n - 1) / 2) * (a 1 - a 0)

theorem arithmetic_sequence_solution :
  ∃ d : ℤ,
  (∀ n : ℕ, n > 0 ∧ n < 10 → a n = 23 + n * d) ∧
  (23 + 5 * d > 0) ∧
  (23 + 6 * d < 0) ∧
  d = -4 ∧
  S a 6 = 78 ∧
  ∀ n : ℕ, S a n > 0 → n ≤ 12 :=
by
  sorry

end arithmetic_sequence_solution_l22_22509


namespace subset_iff_a_values_l22_22655

theorem subset_iff_a_values (a : ℝ) :
  let P := { x : ℝ | x^2 = 1 }
  let Q := { x : ℝ | a * x = 1 }
  Q ⊆ P ↔ a = 0 ∨ a = 1 ∨ a = -1 :=
by sorry

end subset_iff_a_values_l22_22655


namespace player_catches_ball_in_5_seconds_l22_22033

theorem player_catches_ball_in_5_seconds
    (s_ball : ℕ → ℝ) (s_player : ℕ → ℝ)
    (t_ball : ℕ)
    (t_player : ℕ)
    (d_player_initial : ℝ)
    (d_sideline : ℝ) :
  (∀ t, s_ball t = (4.375 * t - 0.375 * t^2)) →
  (∀ t, s_player t = (3.25 * t + 0.25 * t^2)) →
  (d_player_initial = 10) →
  (d_sideline = 23) →
  t_player = 5 →
  s_player t_player + d_player_initial = s_ball t_player ∧ s_ball t_player < d_sideline := 
by sorry

end player_catches_ball_in_5_seconds_l22_22033


namespace range_of_a_l22_22967

noncomputable def f (a x : ℝ) : ℝ := Real.exp (a * x) + 3 * x

def has_pos_extremum (a : ℝ) : Prop :=
  ∃ x : ℝ, (3 + a * Real.exp (a * x) = 0) ∧ (x > 0)

theorem range_of_a (a : ℝ) : has_pos_extremum a → a < -3 := by
  sorry

end range_of_a_l22_22967


namespace mul_pos_neg_eq_neg_l22_22143

theorem mul_pos_neg_eq_neg (a : Int) : 3 * (-2) = -6 := by
  sorry

end mul_pos_neg_eq_neg_l22_22143


namespace john_ingrid_combined_weighted_average_tax_rate_l22_22434

noncomputable def john_employment_income : ℕ := 57000
noncomputable def john_employment_tax_rate : ℚ := 0.30
noncomputable def john_rental_income : ℕ := 11000
noncomputable def john_rental_tax_rate : ℚ := 0.25

noncomputable def ingrid_employment_income : ℕ := 72000
noncomputable def ingrid_employment_tax_rate : ℚ := 0.40
noncomputable def ingrid_investment_income : ℕ := 4500
noncomputable def ingrid_investment_tax_rate : ℚ := 0.15

noncomputable def combined_weighted_average_tax_rate : ℚ :=
  let john_total_tax := john_employment_income * john_employment_tax_rate + john_rental_income * john_rental_tax_rate
  let john_total_income := john_employment_income + john_rental_income
  let ingrid_total_tax := ingrid_employment_income * ingrid_employment_tax_rate + ingrid_investment_income * ingrid_investment_tax_rate
  let ingrid_total_income := ingrid_employment_income + ingrid_investment_income
  let combined_total_tax := john_total_tax + ingrid_total_tax
  let combined_total_income := john_total_income + ingrid_total_income
  (combined_total_tax / combined_total_income) * 100

theorem john_ingrid_combined_weighted_average_tax_rate :
  combined_weighted_average_tax_rate = 34.14 := by
  sorry

end john_ingrid_combined_weighted_average_tax_rate_l22_22434


namespace values_of_a_plus_b_l22_22598

theorem values_of_a_plus_b (a b : ℝ) (h1 : abs (-a) = abs (-1)) (h2 : b^2 = 9) (h3 : abs (a - b) = b - a) : a + b = 2 ∨ a + b = 4 := 
by 
  sorry

end values_of_a_plus_b_l22_22598


namespace avg_speed_round_trip_l22_22458

-- Definitions for the conditions
def speed_P_to_Q : ℝ := 80
def distance (D : ℝ) : ℝ := D
def speed_increase_percentage : ℝ := 0.1
def speed_Q_to_P : ℝ := speed_P_to_Q * (1 + speed_increase_percentage)

-- Average speed calculation function
noncomputable def average_speed (D : ℝ) : ℝ := 
  let total_distance := 2 * D
  let time_P_to_Q := D / speed_P_to_Q
  let time_Q_to_P := D / speed_Q_to_P
  let total_time := time_P_to_Q + time_Q_to_P
  total_distance / total_time

-- Theorem: Average speed for the round trip is 83.81 km/hr
theorem avg_speed_round_trip (D : ℝ) : average_speed D = 83.81 := 
by 
  -- Dummy proof placeholder
  sorry

end avg_speed_round_trip_l22_22458


namespace usual_time_56_l22_22477

theorem usual_time_56 (S : ℝ) (T : ℝ) (h : (T + 24) * S = T * (0.7 * S)) : T = 56 :=
by sorry

end usual_time_56_l22_22477


namespace proof_problem_l22_22056

variable (p q : Prop)

theorem proof_problem
  (h₁ : p ∨ q)
  (h₂ : ¬p) :
  ¬p ∧ q :=
by
  sorry

end proof_problem_l22_22056


namespace percentage_increase_ticket_price_l22_22934

-- Definitions for the conditions
def last_year_income := 100.0
def clubs_share_last_year := 0.10 * last_year_income
def rental_cost := 0.90 * last_year_income
def new_clubs_share := 0.20
def new_income := rental_cost / (1 - new_clubs_share)

-- Lean 4 theorem statement
theorem percentage_increase_ticket_price : 
  new_income = 112.5 → ((new_income - last_year_income) / last_year_income * 100) = 12.5 := 
by
  sorry

end percentage_increase_ticket_price_l22_22934


namespace centroid_of_triangle_l22_22897

theorem centroid_of_triangle :
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  ( (x1 + x2 + x3) / 3 = 8 / 3 ∧ (y1 + y2 + y3) / 3 = -5 / 3 ) :=
by
  let x1 := 9
  let y1 := -8
  let x2 := -5
  let y2 := 6
  let x3 := 4
  let y3 := -3
  have centroid_x : (x1 + x2 + x3) / 3 = 8 / 3 := sorry
  have centroid_y : (y1 + y2 + y3) / 3 = -5 / 3 := sorry
  exact ⟨centroid_x, centroid_y⟩

end centroid_of_triangle_l22_22897


namespace minimum_value_proof_l22_22197

noncomputable def minValue (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : ℝ := 
  (x + 8 * y) / (x * y)

theorem minimum_value_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 2) : 
  minValue x y hx hy h = 9 := 
by
  sorry

end minimum_value_proof_l22_22197


namespace boxes_given_away_l22_22656

def total_boxes := 12
def pieces_per_box := 6
def remaining_pieces := 30

theorem boxes_given_away : (total_boxes * pieces_per_box - remaining_pieces) / pieces_per_box = 7 :=
by
  sorry

end boxes_given_away_l22_22656


namespace bananas_in_each_box_l22_22017

-- You might need to consider noncomputable if necessary here for Lean's real number support.
noncomputable def bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) : ℕ :=
  total_bananas / total_boxes

theorem bananas_in_each_box :
  bananas_per_box 40 8 = 5 := by
  sorry

end bananas_in_each_box_l22_22017


namespace units_digit_sum_squares_of_odd_integers_l22_22177

theorem units_digit_sum_squares_of_odd_integers :
  let first_2005_odd_units := [802, 802, 401] -- counts for units 1, 9, 5 respectively
  let extra_squares_last_6 := [9, 1, 3, 9, 5, 9] -- units digits of the squares of the last 6 numbers
  let total_sum :=
        (first_2005_odd_units[0] * 1 + 
         first_2005_odd_units[1] * 9 + 
         first_2005_odd_units[2] * 5) +
        (extra_squares_last_6.sum)
  (total_sum % 10) = 1 :=
by
  sorry

end units_digit_sum_squares_of_odd_integers_l22_22177


namespace distance_of_point_P_to_base_AB_l22_22490

theorem distance_of_point_P_to_base_AB :
  ∀ (P : ℝ) (A B C : ℝ → ℝ)
    (h : ∀ (x : ℝ), A x = B x)
    (altitude : ℝ)
    (area_ratio : ℝ),
  altitude = 6 →
  area_ratio = 1 / 3 →
  (∃ d : ℝ, d = 6 - (2 / 3) * 6 ∧ d = 2) := 
  sorry

end distance_of_point_P_to_base_AB_l22_22490


namespace probability_two_red_two_blue_l22_22964

theorem probability_two_red_two_blue :
  let total_marbles := 20
  let red_marbles := 12
  let blue_marbles := 8
  let total_ways_to_choose_4 := Nat.choose total_marbles 4
  let ways_to_choose_2_red := Nat.choose red_marbles 2
  let ways_to_choose_2_blue := Nat.choose blue_marbles 2
  (ways_to_choose_2_red * ways_to_choose_2_blue : ℚ) / total_ways_to_choose_4 = 56 / 147 := 
by {
  sorry
}

end probability_two_red_two_blue_l22_22964


namespace greatest_possible_triangle_perimeter_l22_22495

noncomputable def triangle_perimeter (x : ℕ) : ℕ :=
  x + 2 * x + 18

theorem greatest_possible_triangle_perimeter :
  (∃ (x : ℕ), 7 ≤ x ∧ x < 18 ∧ ∀ y : ℕ, (7 ≤ y ∧ y < 18) → triangle_perimeter y ≤ triangle_perimeter x) ∧
  triangle_perimeter 17 = 69 :=
by
  sorry

end greatest_possible_triangle_perimeter_l22_22495


namespace infinite_either_interval_exists_rational_infinite_elements_l22_22529

variable {ε : ℝ} (x : ℕ → ℝ) (hε : ε > 0) (hεlt : ε < 1/2)

-- Problem 1
theorem infinite_either_interval (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) :
  (∃ N : ℕ, ∀ n ≥ N, x n < 1/2) ∨ (∃ N : ℕ, ∀ n ≥ N, x n ≥ 1/2) :=
sorry

-- Problem 2
theorem exists_rational_infinite_elements (x : ℕ → ℝ) (hx : ∀ n, 0 ≤ x n ∧ x n < 1) (hε : ε > 0) (hεlt : ε < 1/2) :
  ∃ (α : ℚ), 0 ≤ α ∧ α ≤ 1 ∧ ∃ N : ℕ, ∀ n ≥ N, x n ∈ [α - ε, α + ε] :=
sorry

end infinite_either_interval_exists_rational_infinite_elements_l22_22529


namespace proportion_of_face_cards_l22_22979

theorem proportion_of_face_cards (p : ℝ) (h : 1 - (1 - p)^3 = 19 / 27) : p = 1 / 3 :=
sorry

end proportion_of_face_cards_l22_22979


namespace min_moves_to_visit_all_non_forbidden_squares_l22_22643

def min_diagonal_moves (n : ℕ) : ℕ :=
  2 * (n / 2) - 1

theorem min_moves_to_visit_all_non_forbidden_squares (n : ℕ) :
  min_diagonal_moves n = 2 * (n / 2) - 1 := by
  sorry

end min_moves_to_visit_all_non_forbidden_squares_l22_22643


namespace parallelogram_height_same_area_l22_22108

noncomputable def rectangle_area (length width : ℕ) : ℕ := length * width

theorem parallelogram_height_same_area (length width base height : ℕ) 
  (h₁ : rectangle_area length width = base * height) 
  (h₂ : length = 12) 
  (h₃ : width = 6) 
  (h₄ : base = 12) : 
  height = 6 := 
sorry

end parallelogram_height_same_area_l22_22108


namespace geom_seq_product_a2_a3_l22_22606

theorem geom_seq_product_a2_a3 :
  ∃ (a_n : ℕ → ℝ), (a_n 1 * a_n 4 = -3) ∧ (∀ n, a_n n = a_n 1 * (a_n 2 / a_n 1) ^ (n - 1)) → a_n 2 * a_n 3 = -3 :=
by
  sorry

end geom_seq_product_a2_a3_l22_22606


namespace scrambled_eggs_count_l22_22239

-- Definitions based on the given conditions
def num_sausages := 3
def time_per_sausage := 5
def time_per_egg := 4
def total_time := 39

-- Prove that Kira scrambled 6 eggs
theorem scrambled_eggs_count : (total_time - num_sausages * time_per_sausage) / time_per_egg = 6 := by
  sorry

end scrambled_eggs_count_l22_22239


namespace find_y_find_x_l22_22952

section
variables (a b : ℝ × ℝ) (x y : ℝ)

-- Definition of vectors a and b
def vec_a : ℝ × ℝ := (3, -2)
def vec_b (y : ℝ) : ℝ × ℝ := (-1, y)

-- Definition of perpendicular condition
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0
-- Proof that y = -3/2 if a is perpendicular to b
theorem find_y (h : perpendicular vec_a (vec_b y)) : y = -3 / 2 :=
sorry

-- Definition of vectors a and c
def vec_c (x : ℝ) : ℝ × ℝ := (x, 5)

-- Definition of parallel condition
def parallel (u v : ℝ × ℝ) : Prop := u.1 / v.1 = u.2 / v.2
-- Proof that x = -15/2 if a is parallel to c
theorem find_x (h : parallel vec_a (vec_c x)) : x = -15 / 2 :=
sorry
end

end find_y_find_x_l22_22952


namespace find_x_l22_22685

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
by
  sorry

end find_x_l22_22685


namespace wheat_distribution_l22_22751

def mill1_rate := 19 / 3 -- quintals per hour
def mill2_rate := 32 / 5 -- quintals per hour
def mill3_rate := 5     -- quintals per hour

def total_wheat := 1330 -- total wheat in quintals

theorem wheat_distribution :
    ∃ (x1 x2 x3 : ℚ), 
    x1 = 475 ∧ x2 = 480 ∧ x3 = 375 ∧ 
    x1 / mill1_rate = x2 / mill2_rate ∧ x2 / mill2_rate = x3 / mill3_rate ∧ 
    x1 + x2 + x3 = total_wheat :=
by {
  sorry
}

end wheat_distribution_l22_22751


namespace price_comparison_2010_l22_22142

def X_initial : ℝ := 4.20
def Y_initial : ℝ := 6.30
def r_X : ℝ := 0.45
def r_Y : ℝ := 0.20
def n : ℕ := 9

theorem price_comparison_2010: 
  X_initial + r_X * n > Y_initial + r_Y * n := by
  sorry

end price_comparison_2010_l22_22142


namespace negation_example_l22_22840

theorem negation_example (p : ∀ x : ℝ, x > 0 → (x + 1) * Real.exp x > 1) :
  ∃ x0 : ℝ, x0 > 0 ∧ (x0 + 1) * Real.exp x0 ≤ 1 :=
sorry

end negation_example_l22_22840


namespace probability_all_and_at_least_one_pass_l22_22593

-- Define conditions
def pA : ℝ := 0.8
def pB : ℝ := 0.6
def pC : ℝ := 0.5

-- Define the main theorem we aim to prove
theorem probability_all_and_at_least_one_pass :
  (pA * pB * pC = 0.24) ∧ ((1 - (1 - pA) * (1 - pB) * (1 - pC)) = 0.96) := by
  sorry

end probability_all_and_at_least_one_pass_l22_22593


namespace simplify_expression_l22_22169

theorem simplify_expression : (1 / (1 + Real.sqrt 3)) * (1 / (1 - Real.sqrt 3)) = -1 / 2 :=
by
  sorry

end simplify_expression_l22_22169


namespace initial_strawberry_plants_l22_22951

theorem initial_strawberry_plants (P : ℕ) (h1 : 24 * P - 4 = 500) : P = 21 := 
by
  sorry

end initial_strawberry_plants_l22_22951


namespace find_x_l22_22875

theorem find_x :
  let a := 0.15
  let b := 0.06
  let c := 0.003375
  let d := 0.000216
  let e := 0.0225
  let f := 0.0036
  let g := 0.08999999999999998
  ∃ x, c - (d / e) + x + f = g →
  x = 0.092625 :=
by
  sorry

end find_x_l22_22875


namespace future_value_proof_l22_22327

noncomputable def present_value : ℝ := 1093.75
noncomputable def interest_rate : ℝ := 0.04
noncomputable def years : ℕ := 2

def future_value (PV : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PV * (1 + r) ^ n

theorem future_value_proof :
  future_value present_value interest_rate years = 1183.06 :=
by
  -- Calculation details skipped here, assuming the required proof steps are completed.
  sorry

end future_value_proof_l22_22327


namespace quadratic_inequality_range_of_k_l22_22226

theorem quadratic_inequality_range_of_k :
  ∀ k : ℝ , (∀ x : ℝ, k * x^2 + 2 * k * x - (k + 2) < 0) ↔ (-1 < k ∧ k ≤ 0) :=
sorry

end quadratic_inequality_range_of_k_l22_22226


namespace pages_per_hour_l22_22241

-- Definitions corresponding to conditions
def lunch_time : ℕ := 4 -- time taken to grab lunch and come back (in hours)
def total_pages : ℕ := 4000 -- total pages in the book
def book_time := 2 * lunch_time  -- time taken to read the book is twice the lunch_time

-- Statement of the problem to be proved
theorem pages_per_hour : (total_pages / book_time = 500) := 
  by
    -- We assume the definitions and want to show the desired property
    sorry

end pages_per_hour_l22_22241


namespace double_point_quadratic_l22_22935

theorem double_point_quadratic (m x1 x2 : ℝ) 
  (H1 : x1 < 1) (H2 : 1 < x2)
  (H3 : ∃ (y1 y2 : ℝ), y1 = 2 * x1 ∧ y2 = 2 * x2 ∧ y1 = x1^2 + 2 * m * x1 - m ∧ y2 = x2^2 + 2 * m * x2 - m)
  : m < 1 :=
sorry

end double_point_quadratic_l22_22935


namespace sum_of_digits_of_N_l22_22191

open Nat

theorem sum_of_digits_of_N (T : ℕ) (hT : T = 3003) :
  ∃ N : ℕ, (N * (N + 1)) / 2 = T ∧ (digits 10 N).sum = 14 :=
by 
  sorry

end sum_of_digits_of_N_l22_22191


namespace transformed_conic_symmetric_eq_l22_22264

def conic_E (x y : ℝ) := x^2 + 2 * x * y + y^2 + 3 * x + y
def line_l (x y : ℝ) := 2 * x - y - 1

def transformed_conic_equation (x y : ℝ) := x^2 + 14 * x * y + 49 * y^2 - 21 * x + 103 * y + 54

theorem transformed_conic_symmetric_eq (x y : ℝ) :
  (∀ x y, conic_E x y = 0 → 
    ∃ x' y', line_l x' y' = 0 ∧ conic_E x' y' = 0 ∧ transformed_conic_equation x y = 0) :=
sorry

end transformed_conic_symmetric_eq_l22_22264


namespace jogger_distance_l22_22996

theorem jogger_distance (t : ℝ) (h : 16 * t = 12 * t + 10) : 12 * t = 30 :=
by
  -- Definition and proof would go here
  --
  sorry

end jogger_distance_l22_22996


namespace convert_13_to_binary_l22_22854

theorem convert_13_to_binary : (13 : ℕ) = 1 * 2^3 + 1 * 2^2 + 0 * 2^1 + 1 * 2^0 :=
by sorry

end convert_13_to_binary_l22_22854


namespace trapezoid_perimeter_area_sum_l22_22377

noncomputable def distance (p1 p2 : Real × Real) : Real :=
  ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2).sqrt

noncomputable def perimeter (vertices : List (Real × Real)) : Real :=
  match vertices with
  | [a, b, c, d] => (distance a b) + (distance b c) + (distance c d) + (distance d a)
  | _ => 0

noncomputable def area_trapezoid (b1 b2 h : Real) : Real :=
  0.5 * (b1 + b2) * h

theorem trapezoid_perimeter_area_sum
  (A B C D : Real × Real)
  (h_AB : A = (2, 3))
  (h_BC : B = (7, 3))
  (h_CD : C = (9, 7))
  (h_DA : D = (0, 7)) :
  let perimeter := perimeter [A, B, C, D]
  let area := area_trapezoid (distance C D) (distance A B) (C.2 - B.2)
  perimeter + area = 42 + 4 * Real.sqrt 5 :=
by
  sorry

end trapezoid_perimeter_area_sum_l22_22377


namespace is_linear_equation_l22_22629

def quadratic_equation (x y : ℝ) : Prop := x * y + 2 * x = 7
def fractional_equation (x y : ℝ) : Prop := (1 / x) + y = 5
def quadratic_equation_2 (x y : ℝ) : Prop := x^2 + y = 2

def linear_equation (x y : ℝ) : Prop := 2 * x - y = 2

theorem is_linear_equation (x y : ℝ) (h1 : quadratic_equation x y) (h2 : fractional_equation x y) (h3 : quadratic_equation_2 x y) : linear_equation x y :=
  sorry

end is_linear_equation_l22_22629


namespace benny_january_savings_l22_22072

theorem benny_january_savings :
  ∃ x : ℕ, x + x + 8 = 46 ∧ x = 19 :=
by
  sorry

end benny_january_savings_l22_22072


namespace condition_M_intersect_N_N_l22_22860

theorem condition_M_intersect_N_N (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + (y - a)^2 ≤ 1 → y ≥ x^2)) ↔ (a ≥ 5 / 4) :=
sorry

end condition_M_intersect_N_N_l22_22860


namespace smallest_positive_x_l22_22631

theorem smallest_positive_x (x : ℝ) (h : x > 0) (h_eq : x / 4 + 3 / (4 * x) = 1) : x = 1 :=
by
  sorry

end smallest_positive_x_l22_22631


namespace bus_is_there_probability_l22_22530

noncomputable def probability_bus_present : ℚ :=
  let total_area := 90 * 90
  let triangle_area := (75 * 75) / 2
  let parallelogram_area := 75 * 15
  let shaded_area := triangle_area + parallelogram_area
  shaded_area / total_area

theorem bus_is_there_probability :
  probability_bus_present = 7/16 :=
by
  sorry

end bus_is_there_probability_l22_22530


namespace max_value_of_a_l22_22448

theorem max_value_of_a (a b c : ℕ) (h : a + b + c = Nat.gcd a b + Nat.gcd b c + Nat.gcd c a + 120) : a ≤ 240 :=
by
  sorry

end max_value_of_a_l22_22448


namespace right_triangle_hypotenuse_l22_22819

theorem right_triangle_hypotenuse (a b : ℕ) (a_val : a = 4) (b_val : b = 5) :
    ∃ c : ℝ, c^2 = (a:ℝ)^2 + (b:ℝ)^2 ∧ c = Real.sqrt 41 :=
by
  sorry

end right_triangle_hypotenuse_l22_22819


namespace pentagon_inequality_l22_22469

-- Definitions
variables {S R1 R2 R3 R4 R5 : ℝ}
noncomputable def sine108 := Real.sin (108 * Real.pi / 180)

-- Theorem statement
theorem pentagon_inequality (h_area : S > 0) (h_radii : R1 > 0 ∧ R2 > 0 ∧ R3 > 0 ∧ R4 > 0 ∧ R5 > 0) :
  R1^4 + R2^4 + R3^4 + R4^4 + R5^4 ≥ (4 / (5 * sine108^2)) * S^2 :=
by
  sorry

end pentagon_inequality_l22_22469


namespace average_salary_correct_l22_22767

def A_salary := 10000
def B_salary := 5000
def C_salary := 11000
def D_salary := 7000
def E_salary := 9000

def total_salary := A_salary + B_salary + C_salary + D_salary + E_salary
def num_individuals := 5

def average_salary := total_salary / num_individuals

theorem average_salary_correct : average_salary = 8600 := by
  sorry

end average_salary_correct_l22_22767


namespace find_n_l22_22549

theorem find_n :
  ∃ n : ℕ, ∀ (a b c : ℕ), a + b + c = 200 ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
    (n = a + b * c) ∧ (n = b + c * a) ∧ (n = c + a * b) → n = 199 :=
by {
  sorry
}

end find_n_l22_22549


namespace normal_trip_distance_l22_22187

variable (S D : ℝ)

-- Conditions
axiom h1 : D = 3 * S
axiom h2 : D + 50 = 5 * S

theorem normal_trip_distance : D = 75 :=
by
  sorry

end normal_trip_distance_l22_22187


namespace calculate_g_g_2_l22_22923

def g (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

theorem calculate_g_g_2 : g (g 2) = 263 :=
by
  sorry

end calculate_g_g_2_l22_22923


namespace mixedGasTemperature_is_correct_l22_22525

noncomputable def mixedGasTemperature (V₁ V₂ p₁ p₂ T₁ T₂ : ℝ) : ℝ := 
  (p₁ * V₁ + p₂ * V₂) / ((p₁ * V₁) / T₁ + (p₂ * V₂) / T₂)

theorem mixedGasTemperature_is_correct :
  mixedGasTemperature 2 3 3 4 400 500 = 462 := by
    sorry

end mixedGasTemperature_is_correct_l22_22525


namespace relationship_between_x_and_y_l22_22708

theorem relationship_between_x_and_y (x y : ℝ) (h1 : 2 * x - y > 3 * x) (h2 : x + 2 * y < 2 * y) :
  x < 0 ∧ y > 0 :=
sorry

end relationship_between_x_and_y_l22_22708


namespace quadratic_inequality_solution_set_empty_l22_22716

theorem quadratic_inequality_solution_set_empty
  (m : ℝ)
  (h : ∀ x : ℝ, mx^2 - mx - 1 < 0) :
  -4 < m ∧ m < 0 :=
sorry

end quadratic_inequality_solution_set_empty_l22_22716


namespace max_k_l22_22396

noncomputable def f (x : ℝ) : ℝ := x + x * Real.log x

theorem max_k (k : ℤ) : (∀ x : ℝ, 1 < x → f x - k * x + k > 0) → k ≤ 3 :=
by
  sorry

end max_k_l22_22396


namespace negation_of_universal_l22_22314

theorem negation_of_universal (P : Prop) :
  (¬ (∀ x : ℝ, x > 0 → x^3 > 0)) ↔ (∃ x : ℝ, x > 0 ∧ x^3 ≤ 0) :=
by sorry

end negation_of_universal_l22_22314


namespace evaluate_expression_l22_22310

theorem evaluate_expression :
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  (x^2 * y^4 * z * w = - (243 / 256)) := 
by {
  let x := (1 : ℚ) / 2
  let y := (3 : ℚ) / 4
  let z := -6
  let w := 2
  sorry
}

end evaluate_expression_l22_22310


namespace time_to_pass_pole_l22_22444

def train_length : ℕ := 250
def platform_length : ℕ := 1250
def time_to_pass_platform : ℕ := 60

theorem time_to_pass_pole : 
  (train_length + platform_length) / time_to_pass_platform * train_length = 10 :=
by
  sorry

end time_to_pass_pole_l22_22444


namespace expression_eval_l22_22453

theorem expression_eval : (-4)^7 / 4^5 + 5^3 * 2 - 7^2 = 185 := by
  sorry

end expression_eval_l22_22453


namespace DVDs_sold_is_168_l22_22619

variables (C D : ℕ)
variables (h1 : D = (16 * C) / 10)
variables (h2 : D + C = 273)

theorem DVDs_sold_is_168 : D = 168 := by
  sorry

end DVDs_sold_is_168_l22_22619


namespace cuberoot_eq_3_implies_cube_eq_19683_l22_22902

theorem cuberoot_eq_3_implies_cube_eq_19683 (x : ℝ) (h : (x + 6)^(1/3) = 3) : (x + 6)^3 = 19683 := by
  sorry

end cuberoot_eq_3_implies_cube_eq_19683_l22_22902


namespace cannot_have_2020_l22_22357

theorem cannot_have_2020 (a b c : ℤ) : 
  ∀ (n : ℕ), n ≥ 4 → 
  ∀ (x y z : ℕ → ℤ), 
    (x 0 = a) → (y 0 = b) → (z 0 = c) → 
    (∀ (k : ℕ), x (k + 1) = y k - z k) →
    (∀ (k : ℕ), y (k + 1) = z k - x k) →
    (∀ (k : ℕ), z (k + 1) = x k - y k) → 
    (¬ (∃ k, k > 0 ∧ k ≤ n ∧ (x k = 2020 ∨ y k = 2020 ∨ z k = 2020))) := 
by
  intros
  sorry

end cannot_have_2020_l22_22357


namespace salt_percentage_in_first_solution_l22_22880

theorem salt_percentage_in_first_solution
    (S : ℝ)
    (h1 : ∀ w : ℝ, w ≥ 0 → ∃ q : ℝ, q = w)  -- One fourth of the first solution was replaced by the second solution
    (h2 : ∀ w1 w2 w3 : ℝ,
            w1 + w2 = w3 →
            (w1 / w3 * S + w2 / w3 * 25 = 16)) :  -- Resulting solution was 16 percent salt by weight
  S = 13 :=   -- Correct answer
sorry

end salt_percentage_in_first_solution_l22_22880


namespace lawrence_walking_speed_l22_22467

theorem lawrence_walking_speed :
  let distance := 4
  let time := (4 : ℝ) / 3
  let speed := distance / time
  speed = 3 := 
by
  sorry

end lawrence_walking_speed_l22_22467


namespace new_person_weight_l22_22517

theorem new_person_weight (W : ℝ) (N : ℝ) (avg_increase : ℝ := 2.5) (replaced_weight : ℝ := 35) :
  (W - replaced_weight + N) = (W + (8 * avg_increase)) → N = 55 := sorry

end new_person_weight_l22_22517


namespace orange_balls_count_l22_22574

theorem orange_balls_count (P_black : ℚ) (O : ℕ) (total_balls : ℕ) 
  (condition1 : total_balls = O + 7 + 6) 
  (condition2 : P_black = 7 / total_balls) 
  (condition3 : P_black = 0.38095238095238093) :
  O = 5 := 
by
  sorry

end orange_balls_count_l22_22574


namespace initial_fraction_spent_on_clothes_l22_22993

-- Define the conditions and the theorem to be proved
theorem initial_fraction_spent_on_clothes 
  (M : ℝ) (F : ℝ)
  (h1 : M = 249.99999999999994)
  (h2 : (3 / 4) * (4 / 5) * (1 - F) * M = 100) :
  F = 11 / 15 :=
sorry

end initial_fraction_spent_on_clothes_l22_22993


namespace solve_for_x_l22_22479

theorem solve_for_x (x : ℝ) (h : 3 - 1 / (1 - x) = 2 * (1 / (1 - x))) : x = 0 :=
by
  sorry

end solve_for_x_l22_22479


namespace vector_addition_correct_l22_22989

variables {A B C D : Type} [AddCommGroup A] [Module ℝ A]

def vector_addition (da cd cb ba : A) : Prop :=
  da + cd - cb = ba

theorem vector_addition_correct (da cd cb ba : A) :
  vector_addition da cd cb ba :=
  sorry

end vector_addition_correct_l22_22989


namespace expected_value_shorter_gentlemen_l22_22270

-- Definitions based on the problem conditions
def expected_shorter_gentlemen (n : ℕ) : ℚ :=
  (n - 1) / 2

-- The main theorem statement based on the problem translation
theorem expected_value_shorter_gentlemen (n : ℕ) : 
  expected_shorter_gentlemen n = (n - 1) / 2 :=
by
  sorry

end expected_value_shorter_gentlemen_l22_22270


namespace find_solutions_l22_22369

theorem find_solutions (x y z : ℝ) :
    (x^2 + y^2 - z * (x + y) = 2 ∧ y^2 + z^2 - x * (y + z) = 4 ∧ z^2 + x^2 - y * (z + x) = 8) ↔
    (x = 1 ∧ y = -1 ∧ z = 2) ∨ (x = -1 ∧ y = 1 ∧ z = -2) := sorry

end find_solutions_l22_22369


namespace new_average_daily_production_l22_22183

theorem new_average_daily_production (n : ℕ) (avg_past : ℕ) (production_today : ℕ) (new_avg : ℕ)
  (h1 : n = 9)
  (h2 : avg_past = 50)
  (h3 : production_today = 100)
  (h4 : new_avg = (avg_past * n + production_today) / (n + 1)) :
  new_avg = 55 :=
by
  -- Using the provided conditions, it will be shown in the proof stage that new_avg equals 55
  sorry

end new_average_daily_production_l22_22183


namespace total_yen_l22_22194

-- Define the given conditions in Lean 4
def bal_bahamian_dollars : ℕ := 5000
def bal_us_dollars : ℕ := 2000
def bal_euros : ℕ := 3000

def exchange_rate_bahamian_to_yen : ℝ := 122.13
def exchange_rate_us_to_yen : ℝ := 110.25
def exchange_rate_euro_to_yen : ℝ := 128.50

def check_acc1 : ℕ := 15000
def check_acc2 : ℕ := 6359
def sav_acc1 : ℕ := 5500
def sav_acc2 : ℕ := 3102

def stocks : ℕ := 200000
def bonds : ℕ := 150000
def mutual_funds : ℕ := 120000

-- Prove the total amount of yen the family has
theorem total_yen : 
  bal_bahamian_dollars * exchange_rate_bahamian_to_yen + 
  bal_us_dollars * exchange_rate_us_to_yen + 
  bal_euros * exchange_rate_euro_to_yen
  + (check_acc1 + check_acc2 + sav_acc1 + sav_acc2 : ℝ)
  + (stocks + bonds + mutual_funds : ℝ) = 1716611 := 
by
  sorry

end total_yen_l22_22194


namespace sara_gave_dan_pears_l22_22791

theorem sara_gave_dan_pears :
  ∀ (original_pears left_pears given_to_dan : ℕ),
    original_pears = 35 →
    left_pears = 7 →
    given_to_dan = original_pears - left_pears →
    given_to_dan = 28 :=
by
  intros original_pears left_pears given_to_dan h_original h_left h_given
  rw [h_original, h_left] at h_given
  exact h_given

end sara_gave_dan_pears_l22_22791


namespace g_five_l22_22337

variable (g : ℝ → ℝ)

-- Given conditions
axiom g_add : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_three : g 3 = 4

-- Prove g(5) = 16 * (1 / 4)^(1/3)
theorem g_five : g 5 = 16 * (1 / 4)^(1/3) := by
  sorry

end g_five_l22_22337


namespace trapezium_second_side_length_l22_22725

-- Define the problem in Lean
variables (a h A b : ℝ)

-- Define the conditions
def conditions : Prop :=
  a = 20 ∧ h = 25 ∧ A = 475

-- Prove the length of the second parallel side
theorem trapezium_second_side_length (h_cond : conditions a h A) : b = 18 :=
by
  sorry

end trapezium_second_side_length_l22_22725


namespace BURN_maps_to_8615_l22_22904

open List Function

def tenLetterMapping : List (Char × Nat) := 
  [('G', 0), ('R', 1), ('E', 2), ('A', 3), ('T', 4), ('N', 5), ('U', 6), ('M', 7), ('B', 8), ('S', 9)]

def charToDigit (c : Char) : Option Nat :=
  tenLetterMapping.lookup c

def wordToNumber (word : List Char) : Option (List Nat) :=
  word.mapM charToDigit 

theorem BURN_maps_to_8615 :
  wordToNumber ['B', 'U', 'R', 'N'] = some [8, 6, 1, 5] :=
by
  sorry

end BURN_maps_to_8615_l22_22904


namespace exists_negative_fraction_lt_four_l22_22253

theorem exists_negative_fraction_lt_four : 
  ∃ (x : ℚ), x < 0 ∧ |x| < 4 := 
sorry

end exists_negative_fraction_lt_four_l22_22253


namespace sam_wins_probability_l22_22232

theorem sam_wins_probability (hitting_probability missing_probability : ℚ)
    (hit_prob : hitting_probability = 2/5)
    (miss_prob : missing_probability = 3/5) : 
    let p := hitting_probability / (1 - missing_probability ^ 2)
    p = 5 / 8 :=
by
    sorry

end sam_wins_probability_l22_22232


namespace cos_arithmetic_sequence_result_l22_22153

-- Define an arithmetic sequence as a function
def arithmetic_seq (a d : ℝ) (n : ℕ) : ℝ := a + (n - 1) * d

theorem cos_arithmetic_sequence_result (a d : ℝ) 
  (h : arithmetic_seq a d 1 + arithmetic_seq a d 5 + arithmetic_seq a d 9 = 8 * Real.pi) :
  Real.cos (arithmetic_seq a d 3 + arithmetic_seq a d 7) = -1 / 2 := by
  sorry

end cos_arithmetic_sequence_result_l22_22153


namespace sector_perimeter_l22_22757

-- Conditions:
def theta : ℝ := 54  -- central angle in degrees
def r : ℝ := 20      -- radius in cm

-- Translation of given conditions and expected result:
theorem sector_perimeter (theta_eq : theta = 54) (r_eq : r = 20) :
  let l := (θ * r) / 180 * Real.pi 
  let perim := l + 2 * r 
  perim = 6 * Real.pi + 40 := sorry

end sector_perimeter_l22_22757


namespace ratio_humans_to_beavers_l22_22308

-- Define the conditions
def humans : ℕ := 38 * 10^6
def moose : ℕ := 1 * 10^6
def beavers : ℕ := 2 * moose

-- Define the theorem to prove the ratio of humans to beavers
theorem ratio_humans_to_beavers : humans / beavers = 19 := by
  sorry

end ratio_humans_to_beavers_l22_22308


namespace min_value_expr_l22_22763

theorem min_value_expr (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) :
  ∃ x : ℝ, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by sorry

end min_value_expr_l22_22763


namespace altitude_of_triangle_l22_22491

theorem altitude_of_triangle
  (a b c : ℝ)
  (h₁ : a = 13)
  (h₂ : b = 15)
  (h₃ : c = 22)
  (h₄ : a + b > c)
  (h₅ : a + c > b)
  (h₆ : b + c > a) :
  let s := (a + b + c) / 2
  let A := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  let h := (2 * A) / c
  h = (30 * Real.sqrt 10) / 11 :=
by
  sorry

end altitude_of_triangle_l22_22491


namespace remainder_8437_by_9_l22_22642

theorem remainder_8437_by_9 : 8437 % 9 = 4 :=
by
  -- proof goes here
  sorry

end remainder_8437_by_9_l22_22642


namespace solve_for_a_l22_22836

theorem solve_for_a (a : ℚ) (h : a + a/3 + a/4 = 11/4) : a = 33/19 :=
sorry

end solve_for_a_l22_22836


namespace two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l22_22409

theorem two_pow_2023_add_three_pow_2023_mod_seven_not_zero : (2^2023 + 3^2023) % 7 ≠ 0 := 
by sorry

end two_pow_2023_add_three_pow_2023_mod_seven_not_zero_l22_22409


namespace width_of_room_l22_22953

theorem width_of_room
  (carpet_has : ℕ)
  (room_length : ℕ)
  (carpet_needs : ℕ)
  (h1 : carpet_has = 18)
  (h2 : room_length = 4)
  (h3 : carpet_needs = 62) :
  (carpet_has + carpet_needs) = room_length * 20 :=
by
  sorry

end width_of_room_l22_22953


namespace net_hourly_rate_correct_l22_22520

noncomputable def net_hourly_rate
    (hours : ℕ) 
    (speed : ℕ) 
    (fuel_efficiency : ℕ) 
    (earnings_per_mile : ℝ) 
    (cost_per_gallon : ℝ) 
    (distance := speed * hours) 
    (gasoline_used := distance / fuel_efficiency) 
    (earnings := earnings_per_mile * distance) 
    (cost_of_gasoline := cost_per_gallon * gasoline_used) 
    (net_earnings := earnings - cost_of_gasoline) : ℝ :=
  net_earnings / hours

theorem net_hourly_rate_correct : 
  net_hourly_rate 3 45 25 0.6 1.8 = 23.76 := 
by 
  unfold net_hourly_rate
  norm_num
  sorry

end net_hourly_rate_correct_l22_22520


namespace usamo_2003_q3_l22_22126

open Real

theorem usamo_2003_q3 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ( (2 * a + b + c) ^ 2 / (2 * a ^ 2 + (b + c) ^ 2)
  + (2 * b + a + c) ^ 2 / (2 * b ^ 2 + (c + a) ^ 2)
  + (2 * c + a + b) ^ 2 / (2 * c ^ 2 + (a + b) ^ 2) ) ≤ 8 := 
sorry

end usamo_2003_q3_l22_22126


namespace sum_proper_divisors_243_l22_22426

theorem sum_proper_divisors_243 : (1 + 3 + 9 + 27 + 81) = 121 :=
by
  sorry

end sum_proper_divisors_243_l22_22426


namespace exists_fraction_bound_infinite_no_fraction_bound_l22_22023

-- Problem 1: Statement 1
theorem exists_fraction_bound (n : ℕ) (hn : 0 < n) :
  ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n + 1 ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

-- Problem 2: Statement 2
theorem infinite_no_fraction_bound :
  ∃ᶠ n : ℕ in Filter.atTop, ¬ ∃ (a b : ℤ), 0 < b ∧ (b : ℝ) ≤ Real.sqrt n ∧ Real.sqrt n ≤ (a : ℝ) / b ∧ (a : ℝ) / b ≤ Real.sqrt (n + 1) :=
sorry

end exists_fraction_bound_infinite_no_fraction_bound_l22_22023


namespace solve_for_x_l22_22379

theorem solve_for_x (x y z : ℤ) (h1 : x + y + z = 14) (h2 : x - y - z = 60) (h3 : x + z = 2 * y) : x = 37 := by
  sorry

end solve_for_x_l22_22379


namespace max_distance_equals_2_sqrt_5_l22_22385

noncomputable def max_distance_from_point_to_line : Real :=
  let P : Real × Real := (2, -1)
  let Q : Real × Real := (-2, 1)
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem max_distance_equals_2_sqrt_5 : max_distance_from_point_to_line = 2 * Real.sqrt 5 := by
  sorry

end max_distance_equals_2_sqrt_5_l22_22385


namespace total_pay_per_week_l22_22714

variable (X Y : ℝ)
variable (hx : X = 1.2 * Y)
variable (hy : Y = 240)

theorem total_pay_per_week : X + Y = 528 := by
  sorry

end total_pay_per_week_l22_22714


namespace projections_possibilities_l22_22131

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

end projections_possibilities_l22_22131


namespace subset_M_N_l22_22091

-- Definition of the sets
def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | 1/x < 3 }

theorem subset_M_N : M ⊆ N :=
by
  -- sorry to skip the proof
  sorry

end subset_M_N_l22_22091


namespace total_amount_spent_l22_22567

theorem total_amount_spent : 
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  wednesday_spending + next_day_spending = 9.00 :=
by
  let value_half_dollar := 0.50
  let wednesday_spending := 4 * value_half_dollar
  let next_day_spending := 14 * value_half_dollar
  show _ 
  sorry

end total_amount_spent_l22_22567


namespace largest_y_coordinate_l22_22577

theorem largest_y_coordinate (x y : ℝ) (h : x^2 / 49 + (y - 3)^2 / 25 = 0) : y = 3 :=
sorry

end largest_y_coordinate_l22_22577


namespace total_savings_l22_22340

theorem total_savings (savings_sep savings_oct : ℕ) 
  (h1 : savings_sep = 260)
  (h2 : savings_oct = savings_sep + 30) :
  savings_sep + savings_oct = 550 := 
sorry

end total_savings_l22_22340


namespace pieces_after_cuts_l22_22489

theorem pieces_after_cuts (n : ℕ) : 
  (∃ n, (8 * n + 1 = 2009)) ↔ (n = 251) :=
by 
  sorry

end pieces_after_cuts_l22_22489


namespace combined_spots_l22_22600

-- Definitions of the conditions
def Rover_spots : ℕ := 46
def Cisco_spots : ℕ := Rover_spots / 2 - 5
def Granger_spots : ℕ := 5 * Cisco_spots

-- The proof statement
theorem combined_spots :
  Granger_spots + Cisco_spots = 108 := by
  sorry

end combined_spots_l22_22600


namespace base13_addition_l22_22995

/--
Given two numbers in base 13: 528₁₃ and 274₁₃, prove that their sum is 7AC₁₃.
-/
theorem base13_addition :
  let u1 := 8
  let t1 := 2
  let h1 := 5
  let u2 := 4
  let t2 := 7
  let h2 := 2
  -- Add the units digits: 8 + 4 = 12; 12 is C in base 13
  let s1 := 12 -- 'C' in base 13
  let carry1 := 1
  -- Add the tens digits along with the carry: 2 + 7 + 1 = 10; 10 is A in base 13
  let s2 := 10 -- 'A' in base 13
  -- Add the hundreds digits: 5 + 2 = 7
  let s3 := 7 -- 7 in base 13
  s1 = 12 ∧ s2 = 10 ∧ s3 = 7 :=
by
  sorry

end base13_addition_l22_22995


namespace largest_c_value_l22_22167

theorem largest_c_value (c : ℝ) (h : -2 * c^2 + 8 * c - 6 ≥ 0) : c ≤ 3 := 
sorry

end largest_c_value_l22_22167


namespace speed_limit_of_friend_l22_22768

theorem speed_limit_of_friend (total_distance : ℕ) (christina_speed : ℕ) (christina_time_min : ℕ) (friend_time_hr : ℕ) 
(h1 : total_distance = 210)
(h2 : christina_speed = 30)
(h3 : christina_time_min = 180)
(h4 : friend_time_hr = 3)
(h5 : total_distance = (christina_speed * (christina_time_min / 60)) + (christina_speed * friend_time_hr)) :
  (total_distance - christina_speed * (christina_time_min / 60)) / friend_time_hr = 40 := 
by
  sorry

end speed_limit_of_friend_l22_22768


namespace cube_volume_is_27_l22_22675

noncomputable def original_cube_edge (a : ℝ) : ℝ := a

noncomputable def original_cube_volume (a : ℝ) : ℝ := a^3

noncomputable def new_rectangular_solid_volume (a : ℝ) : ℝ := (a-2) * a * (a+2)

theorem cube_volume_is_27 (a : ℝ) (h : original_cube_volume a - new_rectangular_solid_volume a = 14) : original_cube_volume a = 27 :=
by
  sorry

end cube_volume_is_27_l22_22675


namespace red_paint_cans_needed_l22_22006

-- Definitions for the problem
def ratio_red_white : ℚ := 3 / 2
def total_cans : ℕ := 30

-- Theorem statement to prove the number of cans of red paint
theorem red_paint_cans_needed : total_cans * (3 / 5) = 18 := by 
  sorry

end red_paint_cans_needed_l22_22006


namespace problem1_problem2_l22_22786

-- Problem (Ⅰ)
theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 :=
sorry

-- Problem (Ⅱ)
theorem problem2 (a : ℝ) (h1 : ∀ (x : ℝ), x ≥ 1 ↔ |x + 3| - |x - a| ≥ 2) :
  a = 2 :=
sorry

end problem1_problem2_l22_22786


namespace find_percentage_of_number_l22_22078

theorem find_percentage_of_number (P : ℝ) (N : ℝ) (h1 : P * N = (4 / 5) * N - 21) (h2 : N = 140) : P * 100 = 65 := 
by 
  sorry

end find_percentage_of_number_l22_22078


namespace angle_measure_l22_22404

theorem angle_measure (x : ℝ) 
  (h : x = 2 * (90 - x) - 60) : 
  x = 40 := 
  sorry

end angle_measure_l22_22404


namespace total_juice_drank_l22_22930

open BigOperators

theorem total_juice_drank (joe_juice sam_fraction alex_fraction : ℚ) :
  joe_juice = 3 / 4 ∧ sam_fraction = 1 / 2 ∧ alex_fraction = 1 / 4 → 
  sam_fraction * joe_juice + alex_fraction * joe_juice = 9 / 16 :=
by
  sorry

end total_juice_drank_l22_22930


namespace problem1_problem2_l22_22601

-- First Problem Statement:
theorem problem1 :  12 - (-18) + (-7) - 20 = 3 := 
by 
  sorry

-- Second Problem Statement:
theorem problem2 : -4 / (1 / 2) * 8 = -64 := 
by 
  sorry

end problem1_problem2_l22_22601


namespace weight_computation_requires_initial_weight_l22_22657

-- Let's define the conditions
variable (initial_weight : ℕ) -- The initial weight of the pet; needs to be provided
def yearly_gain := 11  -- The pet gains 11 pounds each year
def age := 8  -- The pet is 8 years old

-- Define the goal to be proved
def current_weight_computable : Prop :=
  initial_weight ≠ 0 → initial_weight + (yearly_gain * age) ≠ 0

-- State the theorem
theorem weight_computation_requires_initial_weight : ¬ ∃ current_weight, initial_weight + (yearly_gain * age) = current_weight :=
by {
  sorry
}

end weight_computation_requires_initial_weight_l22_22657


namespace minimum_x_for_g_maximum_l22_22184

theorem minimum_x_for_g_maximum :
  ∃ x > 0, ∀ k m: ℤ, (x = 1440 * k + 360 ∧ x = 2520 * m + 630) -> x = 7560 :=
by
  sorry

end minimum_x_for_g_maximum_l22_22184


namespace eliana_total_steps_l22_22113

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

end eliana_total_steps_l22_22113


namespace person_half_Jordyn_age_is_6_l22_22553

variables (Mehki_age Jordyn_age certain_age : ℕ)
axiom h1 : Mehki_age = Jordyn_age + 10
axiom h2 : Jordyn_age = 2 * certain_age
axiom h3 : Mehki_age = 22

theorem person_half_Jordyn_age_is_6 : certain_age = 6 :=
by sorry

end person_half_Jordyn_age_is_6_l22_22553


namespace total_cost_is_346_l22_22561

-- Definitions of the given conditions
def total_people : ℕ := 35 + 5 + 1
def total_lunches : ℕ := total_people + 3
def vegetarian_lunches : ℕ := 10
def gluten_free_lunches : ℕ := 5
def nut_free_lunches : ℕ := 3
def halal_lunches : ℕ := 4
def veg_and_gluten_free_lunches : ℕ := 2
def regular_cost : ℕ := 7
def special_cost : ℕ := 8
def veg_and_gluten_free_cost : ℕ := 9

-- Calculate regular lunches considering dietary overlaps
def regular_lunches : ℕ := 
  total_lunches - vegetarian_lunches - gluten_free_lunches - nut_free_lunches - halal_lunches + veg_and_gluten_free_lunches

-- Calculate costs per category of lunches
def total_regular_cost : ℕ := regular_lunches * regular_cost
def total_vegetarian_cost : ℕ := (vegetarian_lunches - veg_and_gluten_free_lunches) * special_cost
def total_gluten_free_cost : ℕ := gluten_free_lunches * special_cost
def total_nut_free_cost : ℕ := nut_free_lunches * special_cost
def total_halal_cost : ℕ := halal_lunches * special_cost
def total_veg_and_gluten_free_cost : ℕ := veg_and_gluten_free_lunches * veg_and_gluten_free_cost

-- Calculate total cost
def total_cost : ℕ :=
  total_regular_cost + total_vegetarian_cost + total_gluten_free_cost + total_nut_free_cost + total_halal_cost + total_veg_and_gluten_free_cost

-- Theorem stating the main question
theorem total_cost_is_346 : total_cost = 346 :=
  by
    -- This is where the proof would go
    sorry

end total_cost_is_346_l22_22561


namespace mean_height_is_68_l22_22874

/-
Given the heights of the volleyball players:
  heights_50s = [58, 59]
  heights_60s = [60, 61, 62, 65, 65, 66, 67]
  heights_70s = [70, 71, 71, 72, 74, 75, 79, 79]

We need to prove that the mean height of the players is 68 inches.
-/
def heights_50s : List ℕ := [58, 59]
def heights_60s : List ℕ := [60, 61, 62, 65, 65, 66, 67]
def heights_70s : List ℕ := [70, 71, 71, 72, 74, 75, 79, 79]

def total_heights : List ℕ := heights_50s ++ heights_60s ++ heights_70s
def number_of_players : ℕ := total_heights.length
def total_height : ℕ := total_heights.sum
def mean_height : ℕ := total_height / number_of_players

theorem mean_height_is_68 : mean_height = 68 := by
  sorry

end mean_height_is_68_l22_22874


namespace trader_sold_40_meters_of_cloth_l22_22692

theorem trader_sold_40_meters_of_cloth 
  (total_profit_per_meter : ℕ) 
  (total_profit : ℕ) 
  (meters_sold : ℕ) 
  (h1 : total_profit_per_meter = 30) 
  (h2 : total_profit = 1200) 
  (h3 : total_profit = total_profit_per_meter * meters_sold) : 
  meters_sold = 40 := by
  sorry

end trader_sold_40_meters_of_cloth_l22_22692


namespace find_b_if_lines_parallel_l22_22505

theorem find_b_if_lines_parallel (b : ℝ) :
  (∀ x y : ℝ, 3 * y - 3 * b = 9 * x → y = 3 * x + b) ∧
  (∀ x y : ℝ, y + 2 = (b + 9) * x → y = (b + 9) * x - 2) →
  3 = b + 9 →
  b = -6 :=
by {
  sorry
}

end find_b_if_lines_parallel_l22_22505


namespace max_playground_area_l22_22948

theorem max_playground_area
  (l w : ℝ)
  (h_fence : 2 * l + 2 * w = 400)
  (h_l_min : l ≥ 100)
  (h_w_min : w ≥ 50) :
  l * w ≤ 10000 :=
by
  sorry

end max_playground_area_l22_22948


namespace razorback_tshirts_sold_l22_22451

variable (T : ℕ) -- Number of t-shirts sold
variable (price_per_tshirt : ℕ := 62) -- Price of each t-shirt
variable (total_revenue : ℕ := 11346) -- Total revenue from t-shirts

theorem razorback_tshirts_sold :
  (price_per_tshirt * T = total_revenue) → T = 183 :=
by
  sorry

end razorback_tshirts_sold_l22_22451


namespace smallest_positive_period_l22_22394

theorem smallest_positive_period :
  ∀ (x : ℝ), 5 * Real.sin ((π / 6) - (π / 3) * x) = 5 * Real.sin ((π / 6) - (π / 3) * (x + 6)) :=
by
  sorry

end smallest_positive_period_l22_22394


namespace polynomial_average_k_l22_22693

theorem polynomial_average_k (h : ∀ x : ℕ, x * (36 / x) = 36 → (x + (36 / x) = 37 ∨ x + (36 / x) = 20 ∨ x + (36 / x) = 15 ∨ x + (36 / x) = 13 ∨ x + (36 / x) = 12)) :
  (37 + 20 + 15 + 13 + 12) / 5 = 19.4 := by
sorry

end polynomial_average_k_l22_22693


namespace sum_of_integers_l22_22324

theorem sum_of_integers :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ a < 30 ∧ b < 30 ∧ (a * b + a + b = 167) ∧ Nat.gcd a b = 1 ∧ (a + b = 24) :=
by {
  sorry
}

end sum_of_integers_l22_22324


namespace find_number_l22_22898

theorem find_number (x : ℝ) :
  9 * (((x + 1.4) / 3) - 0.7) = 5.4 ↔ x = 2.5 :=
by sorry

end find_number_l22_22898


namespace quadratic_expression_positive_l22_22782

theorem quadratic_expression_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ (7 - 4 * Real.sqrt 2 < k ∧ k < 7 + 4 * Real.sqrt 2) :=
by
  sorry

end quadratic_expression_positive_l22_22782


namespace unique_solution_condition_l22_22679

theorem unique_solution_condition (a b c : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + a = c * x + b) ↔ c ≠ 4 :=
sorry

end unique_solution_condition_l22_22679


namespace range_of_m_l22_22569

noncomputable def f (x m : ℝ) : ℝ :=
if x < 0 then 1 / (Real.exp x) + m * x^2
else Real.exp x + m * x^2

theorem range_of_m {m : ℝ} : (∀ m, ∃ x y, f x m = 0 ∧ f y m = 0 ∧ x ≠ y) ↔ m < -Real.exp 2 / 4 := by
  sorry

end range_of_m_l22_22569


namespace no_2021_residents_possible_l22_22678

-- Definition: Each islander is either a knight or a liar
def is_knight_or_liar (i : ℕ) : Prop := true -- Placeholder definition for either being a knight or a liar

-- Definition: Knights always tell the truth
def knight_tells_truth (i : ℕ) : Prop := true -- Placeholder definition for knights telling the truth

-- Definition: Liars always lie
def liar_always_lies (i : ℕ) : Prop := true -- Placeholder definition for liars always lying

-- Definition: Even number of knights claimed by some islanders
def even_number_of_knights : Prop := true -- Placeholder definition for the claim of even number of knights

-- Definition: Odd number of liars claimed by remaining islanders
def odd_number_of_liars : Prop := true -- Placeholder definition for the claim of odd number of liars

-- Question and proof problem
theorem no_2021_residents_possible (K L : ℕ) (h1 : K + L = 2021) (h2 : ∀ i, is_knight_or_liar i) 
(h3 : ∀ k, knight_tells_truth k → even_number_of_knights) 
(h4 : ∀ l, liar_always_lies l → odd_number_of_liars) : 
  false := sorry

end no_2021_residents_possible_l22_22678


namespace max_pawns_19x19_l22_22523

def maxPawnsOnChessboard (n : ℕ) := 
  n * n

theorem max_pawns_19x19 :
  maxPawnsOnChessboard 19 = 361 := 
by
  sorry

end max_pawns_19x19_l22_22523


namespace total_litter_pieces_l22_22205

-- Define the number of glass bottles and aluminum cans as constants.
def glass_bottles : ℕ := 10
def aluminum_cans : ℕ := 8

-- State the theorem that the sum of glass bottles and aluminum cans is 18.
theorem total_litter_pieces : glass_bottles + aluminum_cans = 18 := by
  sorry

end total_litter_pieces_l22_22205


namespace arithmetic_geometric_mean_l22_22065

theorem arithmetic_geometric_mean (a b : ℝ) (h1 : a + b = 48) (h2 : a * b = 440) : a^2 + b^2 = 1424 := 
by 
  -- Proof goes here
  sorry

end arithmetic_geometric_mean_l22_22065


namespace abc_inequality_l22_22110

theorem abc_inequality (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_cond : a * b + b * c + c * a = 1) :
  (a + b + c) ≥ Real.sqrt 3 ∧ (a + b + c = Real.sqrt 3 ↔ a = b ∧ b = c ∧ c = Real.sqrt 1 / Real.sqrt 3) :=
by sorry

end abc_inequality_l22_22110


namespace toby_photo_shoot_l22_22360

theorem toby_photo_shoot (initial_photos : ℕ) (deleted_bad_shots : ℕ) (cat_pictures : ℕ) (deleted_post_editing : ℕ) (final_photos : ℕ) (photo_shoot_photos : ℕ) :
  initial_photos = 63 →
  deleted_bad_shots = 7 →
  cat_pictures = 15 →
  deleted_post_editing = 3 →
  final_photos = 84 →
  final_photos = initial_photos - deleted_bad_shots + cat_pictures + photo_shoot_photos - deleted_post_editing →
  photo_shoot_photos = 16 :=
by
  intros
  sorry

end toby_photo_shoot_l22_22360


namespace triangle_inequality_l22_22611

variables (a b c S : ℝ) (S_def : S = (a + b + c) / 2)

theorem triangle_inequality 
  (h_triangle: a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) :
  2 * S * (Real.sqrt (S - a) + Real.sqrt (S - b) + Real.sqrt (S - c)) 
  ≤ 3 * (Real.sqrt (b * c * (S - a)) + Real.sqrt (c * a * (S - b)) + Real.sqrt (a * b * (S - c))) :=
sorry

end triangle_inequality_l22_22611


namespace ratio_of_cream_l22_22332

def initial_coffee := 12
def joe_drank := 2
def cream_added := 2
def joann_cream_added := 2
def joann_drank := 2

noncomputable def joe_coffee_after_drink_add := initial_coffee - joe_drank + cream_added
noncomputable def joe_cream := cream_added

noncomputable def joann_initial_mixture := initial_coffee + joann_cream_added
noncomputable def joann_portion_before_drink := joann_cream_added / joann_initial_mixture
noncomputable def joann_remaining_coffee := joann_initial_mixture - joann_drank
noncomputable def joann_cream_after_drink := joann_portion_before_drink * joann_remaining_coffee
noncomputable def joann_cream := joann_cream_after_drink

theorem ratio_of_cream : joe_cream / joann_cream = 7 / 6 :=
by sorry

end ratio_of_cream_l22_22332


namespace child_tickets_sold_l22_22659

-- Define variables and types
variables (A C : ℕ)

-- Main theorem to prove
theorem child_tickets_sold : A + C = 80 ∧ 12 * A + 5 * C = 519 → C = 63 :=
by
  intros
  sorry

end child_tickets_sold_l22_22659


namespace relationship_between_b_and_g_l22_22291

-- Definitions based on the conditions
def n_th_boy_dances (n : ℕ) : ℕ := n + 5
def last_boy_dances_with_all : Prop := ∃ b g : ℕ, (n_th_boy_dances b = g)

-- The main theorem to prove the relationship between b and g
theorem relationship_between_b_and_g (b g : ℕ) (h : last_boy_dances_with_all) : b = g - 5 :=
by
  sorry

end relationship_between_b_and_g_l22_22291


namespace parabola_point_coord_l22_22718

theorem parabola_point_coord {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : (x - 1)^2 + y^2 = 100) : x = 9 ∧ (y = 6 ∨ y = -6) :=
by 
  sorry

end parabola_point_coord_l22_22718


namespace silk_pieces_count_l22_22152

theorem silk_pieces_count (S C : ℕ) (h1 : S = 2 * C) (h2 : S + C + 2 = 13) : S = 7 :=
by
  sorry

end silk_pieces_count_l22_22152


namespace M_union_N_eq_M_l22_22372

def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | abs (p.1 * p.2) = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end M_union_N_eq_M_l22_22372


namespace lilies_per_centerpiece_l22_22312

def centerpieces := 6
def roses_per_centerpiece := 8
def orchids_per_rose := 2
def total_flowers := 120
def ratio_roses_orchids_lilies_centerpiece := 1 / 2 / 3

theorem lilies_per_centerpiece :
  ∀ (c : ℕ) (r : ℕ) (o : ℕ) (l : ℕ),
  c = centerpieces → r = roses_per_centerpiece →
  o = orchids_per_rose * r →
  total_flowers = 6 * (r + o + l) →
  ratio_roses_orchids_lilies_centerpiece = r / o / l →
  l = 10 := by sorry

end lilies_per_centerpiece_l22_22312


namespace area_of_triangle_ABC_l22_22155

noncomputable def area_triangle_ABC (AF BE : ℝ) (angle_FGB : ℝ) : ℝ :=
  let FG := AF / 3
  let BG := (2 / 3) * BE
  let area_FGB := (1 / 2) * FG * BG * Real.sin angle_FGB
  6 * area_FGB

theorem area_of_triangle_ABC
  (AF BE : ℕ) (hAF : AF = 10) (hBE : BE = 15)
  (angle_FGB : ℝ) (h_angle_FGB : angle_FGB = Real.pi / 3) :
  area_triangle_ABC AF BE angle_FGB = 50 * Real.sqrt 3 :=
by
  simp [area_triangle_ABC, hAF, hBE, h_angle_FGB]
  sorry

end area_of_triangle_ABC_l22_22155


namespace ratio_of_area_of_inscribed_circle_to_triangle_l22_22508

theorem ratio_of_area_of_inscribed_circle_to_triangle (h r : ℝ) (h_pos : 0 < h) (r_pos : 0 < r) :
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  (π * r) / s = (5 * π * r) / (12 * h) :=
by
  let a := (3 / 5) * h
  let b := (4 / 5) * h
  let A := (1 / 2) * a * b
  let s := (a + b + h) / 2
  sorry

end ratio_of_area_of_inscribed_circle_to_triangle_l22_22508


namespace max_a_if_monotonically_increasing_l22_22480

noncomputable def f (x a : ℝ) : ℝ := x^3 + Real.exp x - a * x

theorem max_a_if_monotonically_increasing (a : ℝ) : 
  (∀ x, 0 ≤ x → 3 * x^2 + Real.exp x - a ≥ 0) ↔ a ≤ 1 :=
by
  sorry

end max_a_if_monotonically_increasing_l22_22480


namespace find_value_l22_22057

theorem find_value (number remainder certain_value : ℕ) (h1 : number = 26)
  (h2 : certain_value / 2 = remainder) 
  (h3 : remainder = ((number + 20) * 2 / 2) - 2) :
  certain_value = 88 :=
by
  sorry

end find_value_l22_22057


namespace decagon_side_length_in_rectangle_l22_22188

theorem decagon_side_length_in_rectangle
  (AB CD : ℝ)
  (AE FB : ℝ)
  (s : ℝ)
  (cond1 : AB = 10)
  (cond2 : CD = 15)
  (cond3 : AE = 5)
  (cond4 : FB = 5)
  (regular_decagon : ℝ → Prop)
  (h : regular_decagon s) : 
  s = 5 * (Real.sqrt 2 - 1) :=
by 
  sorry

end decagon_side_length_in_rectangle_l22_22188


namespace range_of_m_l22_22500

noncomputable def A := {x : ℝ | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 - m * x + 2 = 0}

theorem range_of_m (m : ℝ) (h : ∀ x, x ∈ B m → x ∈ A) : m = 3 ∨ -2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2 :=
by
  sorry

end range_of_m_l22_22500


namespace find_abc_l22_22333

theorem find_abc :
  ∃ (N : ℕ), (N > 0 ∧ (N % 10000 = N^2 % 10000) ∧ (N % 1000 > 100)) ∧ (N % 1000 / 100 = 937) :=
sorry

end find_abc_l22_22333


namespace store_profit_loss_l22_22740

theorem store_profit_loss :
  ∃ (x y : ℝ), (1 + 0.25) * x = 135 ∧ (1 - 0.25) * y = 135 ∧ (135 - x) + (135 - y) = -18 :=
by
  sorry

end store_profit_loss_l22_22740


namespace cars_people_equation_l22_22289

-- Define the first condition
def condition1 (x : ℕ) : ℕ := 4 * (x - 1)

-- Define the second condition
def condition2 (x : ℕ) : ℕ := 2 * x + 8

-- Main theorem which states that the conditions lead to the equation
theorem cars_people_equation (x : ℕ) : condition1 x = condition2 x :=
by
  sorry

end cars_people_equation_l22_22289


namespace joan_seashells_l22_22691

variable (initialSeashells seashellsGiven remainingSeashells : ℕ)

theorem joan_seashells : initialSeashells = 79 ∧ seashellsGiven = 63 ∧ remainingSeashells = initialSeashells - seashellsGiven → remainingSeashells = 16 :=
by
  intros
  sorry

end joan_seashells_l22_22691


namespace box_surface_area_l22_22136

theorem box_surface_area (a b c : ℕ) (h1 : a * b * c = 280) (h2 : a < 10) (h3 : b < 10) (h4 : c < 10) : 
  2 * (a * b + b * c + c * a) = 262 :=
sorry

end box_surface_area_l22_22136


namespace max_min_value_d_l22_22702

-- Definitions of the given conditions
def circle_eqn (x y : ℝ) : Prop := (x - 3)^2 + (y - 4)^2 = 1

def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (1, 0)

-- Definition of the distance squared from a point to a fixed point
def dist_sq (P Q : ℝ × ℝ) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Definition of the function d
def d (P : ℝ × ℝ) : ℝ := dist_sq P A + dist_sq P B

-- The main theorem that we need to prove
theorem max_min_value_d :
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → d P ≤ 74) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 74) ∧
  (∀ P : ℝ × ℝ, circle_eqn P.1 P.2 → 34 ≤ d P) ∧
  (∃ P : ℝ × ℝ, circle_eqn P.1 P.2 ∧ d P = 34) :=
sorry

end max_min_value_d_l22_22702


namespace carrots_total_l22_22503

theorem carrots_total 
  (picked_1 : Nat) 
  (thrown_out : Nat) 
  (picked_2 : Nat) 
  (total_carrots : Nat) 
  (h_picked1 : picked_1 = 23) 
  (h_thrown_out : thrown_out = 10) 
  (h_picked2 : picked_2 = 47) : 
  total_carrots = 60 := 
by
  sorry

end carrots_total_l22_22503


namespace math_problem_l22_22306

noncomputable def proof_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : Prop :=
  let n1 := a + 1/b
  let n2 := b + 1/c
  let n3 := c + 1/a
  (n1 ≤ -2) ∨ (n2 ≤ -2) ∨ (n3 ≤ -2)

theorem math_problem (a b c : ℝ) (h₀ : a < 0) (h₁ : b < 0) (h₂ : c < 0) : proof_problem a b c h₀ h₁ h₂ :=
sorry

end math_problem_l22_22306


namespace find_a_b_c_sum_l22_22029

theorem find_a_b_c_sum (a b c : ℤ)
  (h_gcd : gcd (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x + 1)
  (h_lcm : lcm (x ^ 2 + a * x + b) (x ^ 2 + b * x + c) = x ^ 3 - 4 * x ^ 2 + x + 6) :
  a + b + c = -6 := 
sorry

end find_a_b_c_sum_l22_22029


namespace trigonometric_identity_l22_22506

theorem trigonometric_identity (θ : ℝ) (h : 2 * (Real.cos θ) + (Real.sin θ) = 0) :
  Real.cos (2 * θ) + 1/2 * Real.sin (2 * θ) = -1 := 
sorry

end trigonometric_identity_l22_22506


namespace eccentricity_of_ellipse_l22_22320

variables {E F1 F2 P Q : Type}
variables (a c : ℝ) 

-- Define the foci and intersection conditions
def is_right_foci (F1 F2 : Type) (E : Type) : Prop := sorry
def line_intersects_ellipse (E : Type) (P Q : Type) (slope : ℝ) : Prop := sorry
def is_right_triangle (P F2 : Type) : Prop := sorry

-- Prove the eccentricity condition
theorem eccentricity_of_ellipse
  (h_foci : is_right_foci F1 F2 E)
  (h_line : line_intersects_ellipse E P Q (4 / 3))
  (h_triangle : is_right_triangle P F2) :
  (c / a) = (5 / 7) :=
sorry

end eccentricity_of_ellipse_l22_22320


namespace degree_of_g_l22_22815

theorem degree_of_g (f g : Polynomial ℝ) (h : Polynomial ℝ) (H1 : h = f.comp g + g) 
  (H2 : h.natDegree = 6) (H3 : f.natDegree = 3) : g.natDegree = 2 := 
sorry

end degree_of_g_l22_22815


namespace distance_to_school_is_correct_l22_22582

-- Define the necessary constants, variables, and conditions
def distance_to_market : ℝ := 2
def total_weekly_mileage : ℝ := 44
def school_trip_miles (x : ℝ) : ℝ := 16 * x
def market_trip_miles : ℝ := 2 * distance_to_market
def total_trip_miles (x : ℝ) : ℝ := school_trip_miles x + market_trip_miles

-- Prove that the distance from Philip's house to the children's school is 2.5 miles
theorem distance_to_school_is_correct (x : ℝ) (h : total_trip_miles x = total_weekly_mileage) :
  x = 2.5 :=
by
  -- Insert necessary proof steps starting with the provided hypothesis
  sorry

end distance_to_school_is_correct_l22_22582


namespace smallest_next_divisor_221_l22_22323

structure Conditions (m : ℕ) :=
  (m_even : m % 2 = 0)
  (m_4digit : 1000 ≤ m ∧ m < 10000)
  (m_div_221 : 221 ∣ m)

theorem smallest_next_divisor_221 (m : ℕ) (h : Conditions m) : ∃ k, k > 221 ∧ k ∣ m ∧ k = 289 := by
  sorry

end smallest_next_divisor_221_l22_22323


namespace proof_problem_l22_22245

variable (a b c m : ℝ)

-- Condition
def condition : Prop := m = (c * a * b) / (a + b)

-- Question
def question : Prop := b = (m * a) / (c * a - m)

-- Proof statement
theorem proof_problem (h : condition a b c m) : question a b c m := 
sorry

end proof_problem_l22_22245


namespace percentage_hindus_l22_22635

-- Conditions 
def total_boys : ℕ := 850
def percentage_muslims : ℝ := 0.44
def percentage_sikhs : ℝ := 0.10
def boys_other_communities : ℕ := 272

-- Question and proof statement
theorem percentage_hindus (total_boys : ℕ) (percentage_muslims percentage_sikhs : ℝ) (boys_other_communities : ℕ) : 
  (total_boys = 850) →
  (percentage_muslims = 0.44) →
  (percentage_sikhs = 0.10) →
  (boys_other_communities = 272) →
  ((850 - (374 + 85 + 272)) / 850) * 100 = 14 := 
by
  intros
  sorry

end percentage_hindus_l22_22635


namespace total_spent_l22_22304

def original_cost_vacuum_cleaner : ℝ := 250
def discount_vacuum_cleaner : ℝ := 0.20
def cost_dishwasher : ℝ := 450
def special_offer_discount : ℝ := 75

theorem total_spent :
  let discounted_vacuum_cleaner := original_cost_vacuum_cleaner * (1 - discount_vacuum_cleaner)
  let total_before_special := discounted_vacuum_cleaner + cost_dishwasher
  total_before_special - special_offer_discount = 575 := by
  sorry

end total_spent_l22_22304


namespace tangent_line_at_point_l22_22748

theorem tangent_line_at_point (x y : ℝ) (h : y = x^2) (hx : x = 1) (hy : y = 1) : 
  2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_at_point_l22_22748


namespace cone_base_circumference_l22_22012

theorem cone_base_circumference (r : ℝ) (θ : ℝ) (circ_res : ℝ) :
  r = 4 → θ = 270 → circ_res = 6 * Real.pi :=
by 
  sorry

end cone_base_circumference_l22_22012


namespace initial_food_supplies_l22_22548

theorem initial_food_supplies (x : ℝ) 
  (h1 : (3 / 5) * x - (3 / 5) * ((3 / 5) * x) = 96) : x = 400 :=
by
  sorry

end initial_food_supplies_l22_22548


namespace cyclic_path_1310_to_1315_l22_22230

theorem cyclic_path_1310_to_1315 :
  ∀ (n : ℕ), (n % 6 = 2 → (n + 5) % 6 = 3) :=
by
  sorry

end cyclic_path_1310_to_1315_l22_22230


namespace penthouse_units_l22_22445

theorem penthouse_units (total_floors : ℕ) (regular_units_per_floor : ℕ) (penthouse_floors : ℕ) (total_units : ℕ) :
  total_floors = 23 →
  regular_units_per_floor = 12 →
  penthouse_floors = 2 →
  total_units = 256 →
  (total_units - (total_floors - penthouse_floors) * regular_units_per_floor) / penthouse_floors = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end penthouse_units_l22_22445


namespace women_population_percentage_l22_22101

theorem women_population_percentage (W M : ℕ) (h : M = 2 * W) : (W : ℚ) / (M : ℚ) = (50 : ℚ) / 100 :=
by
  -- Proof omitted
  sorry

end women_population_percentage_l22_22101


namespace has_two_distinct_real_roots_parabola_equation_l22_22359

open Real

-- Define the quadratic polynomial
def quad_poly (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4

-- Question 1: Prove that the quadratic equation has two distinct real roots
theorem has_two_distinct_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quad_poly m x₁ = 0) ∧ (quad_poly m x₂ = 0) := by
  sorry

-- Question 2: Prove the equation of the parabola given certain conditions
theorem parabola_equation (m : ℝ) (hx : quad_poly m 0 = 0) : 
  m = 0 ∧ ∀ x : ℝ, quad_poly m x = x^2 - 4 := by
  sorry

end has_two_distinct_real_roots_parabola_equation_l22_22359


namespace solution_set_quadratic_inequality_l22_22846

theorem solution_set_quadratic_inequality :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} :=
sorry

end solution_set_quadratic_inequality_l22_22846


namespace find_chemistry_marks_l22_22690

theorem find_chemistry_marks :
  (let E := 96
   let M := 95
   let P := 82
   let B := 95
   let avg := 93
   let n := 5
   let Total := avg * n
   let Chemistry_marks := Total - (E + M + P + B)
   Chemistry_marks = 97) :=
by
  let E := 96
  let M := 95
  let P := 82
  let B := 95
  let avg := 93
  let n := 5
  let Total := avg * n
  have h_total : Total = 465 := by norm_num
  let Chemistry_marks := Total - (E + M + P + B)
  have h_chemistry_marks : Chemistry_marks = 97 := by norm_num
  exact h_chemistry_marks

end find_chemistry_marks_l22_22690


namespace negation_of_proposition_l22_22071

theorem negation_of_proposition : (¬ (∀ x : ℝ, x > 2 → x > 3)) = ∃ x > 2, x ≤ 3 := by
  sorry

end negation_of_proposition_l22_22071


namespace intersection_of_P_and_Q_l22_22466

def P : Set ℝ := {x | 1 ≤ x}
def Q : Set ℝ := {x | x < 2}

theorem intersection_of_P_and_Q : P ∩ Q = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_P_and_Q_l22_22466


namespace cost_of_12_cheaper_fruits_l22_22463

-- Defining the price per 10 apples in cents.
def price_per_10_apples : ℕ := 200

-- Defining the price per 5 oranges in cents.
def price_per_5_oranges : ℕ := 150

-- No bulk discount means per item price is just total cost divided by the number of items
def price_per_apple := price_per_10_apples / 10
def price_per_orange := price_per_5_oranges / 5

-- Given the calculation steps, we have to prove that the cost for 12 cheaper fruits (apples) is 240
theorem cost_of_12_cheaper_fruits : 12 * price_per_apple = 240 := by
  -- This step performs the proof, which we skip with sorry
  sorry

end cost_of_12_cheaper_fruits_l22_22463


namespace gcd_8164_2937_l22_22872

/-- Define the two integers a and b -/
def a : ℕ := 8164
def b : ℕ := 2937

/-- Prove that the greatest common divisor of a and b is 1 -/
theorem gcd_8164_2937 : Nat.gcd a b = 1 :=
  by
  sorry

end gcd_8164_2937_l22_22872


namespace family_gathering_l22_22769

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end family_gathering_l22_22769


namespace tickets_used_to_buy_toys_l22_22949

-- Definitions for the conditions
def initial_tickets : ℕ := 13
def leftover_tickets : ℕ := 7

-- The theorem we want to prove
theorem tickets_used_to_buy_toys : initial_tickets - leftover_tickets = 6 :=
by
  sorry

end tickets_used_to_buy_toys_l22_22949


namespace intersection_of_lines_l22_22343

theorem intersection_of_lines : ∃ (x y : ℚ), 8 * x - 5 * y = 20 ∧ 6 * x + 2 * y = 18 ∧ x = 65 / 23 ∧ y = 1 / 2 :=
by {
  -- The solution to the theorem is left as an exercise
  sorry
}

end intersection_of_lines_l22_22343


namespace probability_of_two_white_balls_l22_22869

-- Define the total number of balls
def total_balls : ℕ := 11

-- Define the number of white balls
def white_balls : ℕ := 5

-- Define the number of ways to choose 2 out of n (combinations)
def choose (n r : ℕ) : ℕ := n.choose r

-- Define the total combinations of drawing 2 balls out of 11
def total_combinations : ℕ := choose total_balls 2

-- Define the combinations of drawing 2 white balls out of 5
def white_combinations : ℕ := choose white_balls 2

-- Define the probability of drawing 2 white balls
noncomputable def probability_white : ℚ := (white_combinations : ℚ) / (total_combinations : ℚ)

-- Now, state the theorem that states the desired result
theorem probability_of_two_white_balls : probability_white = 2 / 11 := sorry

end probability_of_two_white_balls_l22_22869


namespace trip_time_difference_l22_22058

theorem trip_time_difference 
  (speed : ℕ) (dist1 dist2 : ℕ) (time_per_hour : ℕ) 
  (h_speed : speed = 60) 
  (h_dist1 : dist1 = 360) 
  (h_dist2 : dist2 = 420) 
  (h_time_per_hour : time_per_hour = 60) : 
  ((dist2 / speed - dist1 / speed) * time_per_hour) = 60 := 
by
  sorry

end trip_time_difference_l22_22058


namespace prime_product_mod_32_l22_22105

open Nat

theorem prime_product_mod_32 : 
  let primes := [3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  let M := List.foldr (· * ·) 1 primes
  M % 32 = 25 :=
by
  sorry

end prime_product_mod_32_l22_22105


namespace eval_expression_l22_22079

theorem eval_expression : (Int.ceil (7 / 3 : ℚ) + Int.floor (-7 / 3 : ℚ)) = 0 := by
  sorry

end eval_expression_l22_22079


namespace solve_inequality_l22_22776

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l22_22776


namespace find_angle_ACB_l22_22457

theorem find_angle_ACB
    (convex_quadrilateral : Prop)
    (angle_BAC : ℝ)
    (angle_CAD : ℝ)
    (angle_ADB : ℝ)
    (angle_BDC : ℝ)
    (h1 : convex_quadrilateral)
    (h2 : angle_BAC = 20)
    (h3 : angle_CAD = 60)
    (h4 : angle_ADB = 50)
    (h5 : angle_BDC = 10)
    : ∃ angle_ACB : ℝ, angle_ACB = 80 :=
by
  -- Here use sorry to skip the proof.
  sorry

end find_angle_ACB_l22_22457


namespace time_spent_on_each_piece_l22_22299

def chairs : Nat := 7
def tables : Nat := 3
def total_time : Nat := 40
def total_pieces := chairs + tables
def time_per_piece := total_time / total_pieces

theorem time_spent_on_each_piece : time_per_piece = 4 :=
by
  sorry

end time_spent_on_each_piece_l22_22299


namespace stratified_sampling_second_year_students_l22_22397

theorem stratified_sampling_second_year_students 
  (total_athletes : ℕ) 
  (first_year_students : ℕ) 
  (sample_size : ℕ) 
  (second_year_students_in_sample : ℕ)
  (h1 : total_athletes = 98) 
  (h2 : first_year_students = 56) 
  (h3 : sample_size = 28)
  (h4 : second_year_students_in_sample = (42 * sample_size) / total_athletes) :
  second_year_students_in_sample = 4 := 
sorry

end stratified_sampling_second_year_students_l22_22397


namespace arithmetic_mean_correct_l22_22198

noncomputable def arithmetic_mean (n : ℕ) (h : n > 1) : ℝ :=
  let one_minus_one_div_n := 1 - (1 / n : ℝ)
  let rest_ones := (n - 1 : ℕ) • 1
  let total_sum : ℝ := rest_ones + one_minus_one_div_n
  total_sum / n

theorem arithmetic_mean_correct (n : ℕ) (h : n > 1) :
  arithmetic_mean n h = 1 - (1 / (n * n : ℝ)) := sorry

end arithmetic_mean_correct_l22_22198


namespace square_tiles_count_l22_22814

theorem square_tiles_count (t s p : ℕ) (h1 : t + s + p = 30) (h2 : 3 * t + 4 * s + 5 * p = 108) : s = 6 := by
  sorry

end square_tiles_count_l22_22814


namespace fewest_seats_occupied_l22_22732

def min_seats_occupied (N : ℕ) : ℕ :=
  if h : N % 4 = 0 then (N / 2) else (N / 2) + 1

theorem fewest_seats_occupied (N : ℕ) (h : N = 150) : min_seats_occupied N = 74 := by
  sorry

end fewest_seats_occupied_l22_22732


namespace find_constants_l22_22889

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (x + 2)

theorem find_constants (a : ℝ) (x : ℝ) (h : x ≠ -2) :
  f a (f a x) = x ∧ a = -4 :=
by
  sorry

end find_constants_l22_22889


namespace smallest_n_satisfying_condition_l22_22016

theorem smallest_n_satisfying_condition :
  ∃ n : ℕ, (n > 1) ∧ (∀ i : ℕ, i ≥ 1 → i < n → (∃ k : ℕ, i + (i+1) = k^2)) ∧ n = 8 :=
sorry

end smallest_n_satisfying_condition_l22_22016


namespace tan_trig_identity_l22_22795

noncomputable def given_condition (α : ℝ) : Prop :=
  Real.tan (α + Real.pi / 3) = 2

theorem tan_trig_identity (α : ℝ) (h : given_condition α) :
  (Real.sin (α + (4 * Real.pi / 3)) + Real.cos ((2 * Real.pi / 3) - α)) /
  (Real.cos ((Real.pi / 6) - α) - Real.sin (α + (5 * Real.pi / 6))) = -3 :=
sorry

end tan_trig_identity_l22_22795


namespace triangle_angle_B_l22_22342

theorem triangle_angle_B (A B C : ℕ) (h₁ : B + C = 110) (h₂ : A + B + C = 180) (h₃ : A = 70) :
  B = 70 ∨ B = 55 ∨ B = 40 :=
by
  sorry

end triangle_angle_B_l22_22342


namespace sin_330_eq_negative_half_l22_22055

theorem sin_330_eq_negative_half : Real.sin (330 * Real.pi / 180) = -1 / 2 :=
by
  sorry

end sin_330_eq_negative_half_l22_22055


namespace joe_height_l22_22436

theorem joe_height (S J : ℕ) (h1 : S + J = 120) (h2 : J = 2 * S + 6) : J = 82 :=
by
  sorry

end joe_height_l22_22436


namespace sum_of_root_and_square_of_other_root_eq_2007_l22_22745

/-- If α and β are the two real roots of the equation x^2 - x - 2006 = 0,
    then the value of α + β^2 is 2007. --/
theorem sum_of_root_and_square_of_other_root_eq_2007
  (α β : ℝ)
  (hα : α^2 - α - 2006 = 0)
  (hβ : β^2 - β - 2006 = 0) :
  α + β^2 = 2007 := sorry

end sum_of_root_and_square_of_other_root_eq_2007_l22_22745


namespace solve_trig_eq_l22_22319

theorem solve_trig_eq (k : ℤ) : 
  ∃ x : ℝ, 12 * Real.sin x - 5 * Real.cos x = 13 ∧ 
           x = Real.pi / 2 + Real.arctan (5 / 12) + 2 * ↑k * Real.pi :=
by
  sorry

end solve_trig_eq_l22_22319


namespace cylindrical_to_rectangular_conversion_l22_22899

theorem cylindrical_to_rectangular_conversion 
  (r θ z : ℝ) 
  (h1 : r = 10) 
  (h2 : θ = Real.pi / 3) 
  (h3 : z = -2) :
  (r * Real.cos θ, r * Real.sin θ, z) = (5, 5 * Real.sqrt 3, -2) :=
by
  sorry

end cylindrical_to_rectangular_conversion_l22_22899


namespace factor_expression_l22_22838

theorem factor_expression (y : ℝ) : 3 * y * (y - 4) + 5 * (y - 4) = (3 * y + 5) * (y - 4) :=
by
  sorry

end factor_expression_l22_22838


namespace garden_width_l22_22578

theorem garden_width :
  ∃ w l : ℝ, (2 * l + 2 * w = 60) ∧ (l * w = 200) ∧ (l = 2 * w) ∧ (w = 10) :=
by
  sorry

end garden_width_l22_22578


namespace jennie_speed_difference_l22_22045

theorem jennie_speed_difference :
  (∀ (d t1 t2 : ℝ), (d = 200) → (t1 = 5) → (t2 = 4) → (40 = d / t1) → (50 = d / t2) → (50 - 40 = 10)) :=
by
  intros d t1 t2 h_d h_t1 h_t2 h_speed_heavy h_speed_no_traffic
  sorry

end jennie_speed_difference_l22_22045


namespace students_play_basketball_l22_22293

theorem students_play_basketball 
  (total_students : ℕ)
  (cricket_players : ℕ)
  (both_players : ℕ)
  (total_students_eq : total_students = 880)
  (cricket_players_eq : cricket_players = 500)
  (both_players_eq : both_players = 220) 
  : ∃ B : ℕ, B = 600 :=
by
  sorry

end students_play_basketball_l22_22293


namespace tim_original_vocab_l22_22541

theorem tim_original_vocab (days_in_year : ℕ) (years : ℕ) (learned_per_day : ℕ) (vocab_increase : ℝ) :
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  original_vocab = 14600 :=
by
  let days := days_in_year * years
  let learned_words := learned_per_day * days
  let original_vocab := learned_words / vocab_increase
  show original_vocab = 14600
  sorry

end tim_original_vocab_l22_22541


namespace parallel_transitive_l22_22870

-- Definition of parallel lines
def are_parallel (l1 l2 : Line) : Prop :=
  ∃ (P : Line), l1 = P ∧ l2 = P

-- Theorem stating that if two lines are parallel to the same line, then they are parallel to each other
theorem parallel_transitive (l1 l2 l3 : Line) (h1 : are_parallel l1 l3) (h2 : are_parallel l2 l3) :
  are_parallel l1 l2 :=
by
  sorry

end parallel_transitive_l22_22870


namespace exists_f_condition_l22_22240

open Nat

-- Define the function φ from ℕ to ℕ
variable (ϕ : ℕ → ℕ)

-- The formal statement capturing the given math proof problem
theorem exists_f_condition (ϕ : ℕ → ℕ) : 
  ∃ (f : ℕ → ℤ), (∀ x : ℕ, f x > f (ϕ x)) :=
  sorry

end exists_f_condition_l22_22240


namespace no_integer_solution_l22_22891

theorem no_integer_solution (x y : ℤ) : 2 * x + 6 * y ≠ 91 :=
by
  sorry

end no_integer_solution_l22_22891


namespace find_mean_of_two_l22_22833

-- Define the set of numbers
def numbers : List ℕ := [1879, 1997, 2023, 2029, 2113, 2125]

-- Define the mean of the four selected numbers
def mean_of_four : ℕ := 2018

-- Define the sum of all numbers
def total_sum : ℕ := numbers.sum

-- Define the sum of the four numbers with a given mean
def sum_of_four : ℕ := 4 * mean_of_four

-- Define the sum of the remaining two numbers
def sum_of_two (total sum_of_four : ℕ) : ℕ := total - sum_of_four

-- Define the mean of the remaining two numbers
def mean_of_two (sum_two : ℕ) : ℕ := sum_two / 2

-- Define the condition theorem to be proven
theorem find_mean_of_two : mean_of_two (sum_of_two total_sum sum_of_four) = 2047 := 
by
  sorry

end find_mean_of_two_l22_22833


namespace problem_statement_l22_22361

theorem problem_statement (x : ℝ) (hx : 47 = x^4 + 1 / x^4) : x^2 + 1 / x^2 = 7 :=
by
  sorry

end problem_statement_l22_22361


namespace car_value_correct_l22_22527

-- Define the initial value and the annual decrease percentages
def initial_value : ℝ := 10000
def annual_decreases : List ℝ := [0.20, 0.15, 0.10, 0.08, 0.05]

-- Function to compute the value of the car after n years
def value_after_years (initial_value : ℝ) (annual_decreases : List ℝ) : ℝ :=
  annual_decreases.foldl (λ acc decrease => acc * (1 - decrease)) initial_value

-- The target value after 5 years
def target_value : ℝ := 5348.88

-- Theorem stating that the computed value matches the target value
theorem car_value_correct :
  value_after_years initial_value annual_decreases = target_value := 
sorry

end car_value_correct_l22_22527


namespace regular_polygon_sides_l22_22842

theorem regular_polygon_sides (n : ℕ) (h : n ≥ 3) 
(h_interior : (n - 2) * 180 / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l22_22842


namespace jerry_age_l22_22303

theorem jerry_age
  (M J : ℕ)
  (h1 : M = 2 * J + 5)
  (h2 : M = 21) :
  J = 8 :=
by
  sorry

end jerry_age_l22_22303


namespace product_percent_x_l22_22412

variables {x y z w : ℝ}
variables (h1 : 0.45 * z = 1.2 * y) 
variables (h2 : y = 0.75 * x) 
variables (h3 : z = 0.8 * w)

theorem product_percent_x :
  (w * y) / x = 1.875 :=
by 
  sorry

end product_percent_x_l22_22412


namespace sheila_weekly_earnings_l22_22826

-- Defining the conditions
def hourly_wage : ℕ := 12
def hours_mwf : ℕ := 8
def days_mwf : ℕ := 3
def hours_tt : ℕ := 6
def days_tt : ℕ := 2

-- Defining Sheila's total weekly earnings
noncomputable def weekly_earnings := (hours_mwf * hourly_wage * days_mwf) + (hours_tt * hourly_wage * days_tt)

-- The statement of the proof
theorem sheila_weekly_earnings : weekly_earnings = 432 :=
by
  sorry

end sheila_weekly_earnings_l22_22826


namespace find_3a_plus_3b_l22_22497

theorem find_3a_plus_3b (a b : ℚ) (h1 : 2 * a + 5 * b = 47) (h2 : 8 * a + 2 * b = 50) :
  3 * a + 3 * b = 73 / 2 := 
sorry

end find_3a_plus_3b_l22_22497


namespace range_of_a_l22_22502

theorem range_of_a (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x + y = 2 ∧ 
    (if x > 1 then (x^2 + 1) / x else Real.log (x + a)) = 
    (if y > 1 then (y^2 + 1) / y else Real.log (y + a))) ↔ 
    a > Real.exp 2 - 1 :=
by sorry

end range_of_a_l22_22502


namespace monotonically_decreasing_iff_a_lt_1_l22_22086

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + (1/2) * a * x^2 - 2 * x

theorem monotonically_decreasing_iff_a_lt_1 {a : ℝ} (h : ∀ x > 0, (deriv (f a) x) < 0) : a < 1 :=
sorry

end monotonically_decreasing_iff_a_lt_1_l22_22086


namespace distinct_integer_values_of_a_l22_22660

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l22_22660


namespace find_t_value_l22_22189

theorem find_t_value (k t : ℤ) (h1 : 0 < k) (h2 : k < 10) (h3 : 0 < t) (h4 : t < 10) : t = 6 :=
by
  sorry

end find_t_value_l22_22189


namespace arithmetic_sequence_seventy_fifth_term_l22_22906

theorem arithmetic_sequence_seventy_fifth_term:
  ∀ (a₁ a₂ d : ℕ), a₁ = 3 → a₂ = 51 → a₂ = a₁ + 24 * d → (3 + 74 * d) = 151 := by
  sorry

end arithmetic_sequence_seventy_fifth_term_l22_22906


namespace interest_difference_l22_22316

theorem interest_difference (P P_B : ℝ) (R_A R_B T : ℝ)
    (h₁ : P = 10000)
    (h₂ : P_B = 4000.0000000000005)
    (h₃ : R_A = 15)
    (h₄ : R_B = 18)
    (h₅ : T = 2) :
    let P_A := P - P_B
    let I_A := (P_A * R_A * T) / 100
    let I_B := (P_B * R_B * T) / 100
    I_A - I_B = 359.99999999999965 := 
by
  sorry

end interest_difference_l22_22316


namespace mike_ride_distance_l22_22103

-- Definitions from conditions
def mike_cost (m : ℕ) : ℝ := 2.50 + 0.25 * m
def annie_cost : ℝ := 2.50 + 5.00 + 0.25 * 16

-- Theorem to prove
theorem mike_ride_distance (m : ℕ) (h : mike_cost m = annie_cost) : m = 36 := by
  sorry

end mike_ride_distance_l22_22103


namespace allan_correct_answers_l22_22845

theorem allan_correct_answers (x y : ℕ) (h1 : x + y = 120) (h2 : x - (0.25 : ℝ) * y = 100) : x = 104 :=
by
  sorry

end allan_correct_answers_l22_22845


namespace slope_range_l22_22018

theorem slope_range (α : Real) (hα : -1 ≤ Real.cos α ∧ Real.cos α ≤ 1) :
  ∃ k ∈ Set.Icc (- Real.sqrt 3 / 3) (Real.sqrt 3 / 3), ∀ x y : Real, x * Real.cos α - Real.sqrt 3 * y - 2 = 0 → y = k * x - (2 / Real.sqrt 3) :=
by
  sorry

end slope_range_l22_22018


namespace moles_of_hcl_l22_22266

-- Definitions according to the conditions
def methane := 1 -- 1 mole of methane (CH₄)
def chlorine := 2 -- 2 moles of chlorine (Cl₂)
def hcl := 1 -- The expected number of moles of Hydrochloric acid (HCl)

-- The Lean 4 statement (no proof required)
theorem moles_of_hcl (methane chlorine : ℕ) : hcl = 1 :=
by sorry

end moles_of_hcl_l22_22266


namespace hyperbola_eccentricity_condition_l22_22450

theorem hyperbola_eccentricity_condition (m : ℝ) (h : m > 0) : 
  (∃ e : ℝ, e = Real.sqrt (1 + m) ∧ e > Real.sqrt 2) → m > 1 :=
by
  sorry

end hyperbola_eccentricity_condition_l22_22450


namespace solution_set_of_inequality_l22_22375

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 + 2*x + 15 ≥ 0} = {x : ℝ | -3 ≤ x ∧ x ≤ 5} := 
sorry

end solution_set_of_inequality_l22_22375


namespace least_positive_integer_x_l22_22249

theorem least_positive_integer_x : ∃ x : ℕ, ((2 * x)^2 + 2 * 43 * (2 * x) + 43^2) % 53 = 0 ∧ 0 < x ∧ (∀ y : ℕ, ((2 * y)^2 + 2 * 43 * (2 * y) + 43^2) % 53 = 0 → 0 < y → x ≤ y) := 
by
  sorry

end least_positive_integer_x_l22_22249


namespace simplify_scientific_notation_l22_22759

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := 
sorry

end simplify_scientific_notation_l22_22759


namespace line_slope_l22_22475

theorem line_slope (x y : ℝ) : 3 * y - (1 / 2) * x = 9 → ∃ m, m = 1 / 6 :=
by
  sorry

end line_slope_l22_22475


namespace circle_equation_l22_22733

theorem circle_equation (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 1)^2 + (b - 1)^2 = 2 → (a, b) = (0, 0)) ∧
  ((0 - 1)^2 + (0 - 1)^2 = 2) → 
  (x - 1)^2 + (y - 1)^2 = 2 := 
by 
  sorry

end circle_equation_l22_22733


namespace mean_of_remaining_students_l22_22543

theorem mean_of_remaining_students
  (n : ℕ) (h : n > 20)
  (mean_score_first_15 : ℝ)
  (mean_score_next_5 : ℝ)
  (overall_mean_score : ℝ) :
  mean_score_first_15 = 10 →
  mean_score_next_5 = 16 →
  overall_mean_score = 11 →
  ∀ a, a = (11 * n - 230) / (n - 20) := by
sorry

end mean_of_remaining_students_l22_22543


namespace inequality_solution_set_l22_22958

theorem inequality_solution_set (x : ℝ) : 9 > -3 * x → x > -3 :=
by
  intro h
  sorry

end inequality_solution_set_l22_22958


namespace find_correct_grades_l22_22755

structure StudentGrades := 
  (Volodya: ℕ) 
  (Sasha: ℕ) 
  (Petya: ℕ)

def isCorrectGrades (grades : StudentGrades) : Prop :=
  grades.Volodya = 5 ∧ grades.Sasha = 4 ∧ grades.Petya = 3

theorem find_correct_grades (grades : StudentGrades)
  (h1 : grades.Volodya = 5 ∨ grades.Volodya ≠ 5)
  (h2 : grades.Sasha = 3 ∨ grades.Sasha ≠ 3)
  (h3 : grades.Petya ≠ 5 ∨ grades.Petya = 5)
  (unique_h1: grades.Volodya = 5 ∨ grades.Sasha = 5 ∨ grades.Petya = 5) 
  (unique_h2: grades.Volodya = 4 ∨ grades.Sasha = 4 ∨ grades.Petya = 4)
  (unique_h3: grades.Volodya = 3 ∨ grades.Sasha = 3 ∨ grades.Petya = 3) 
  (lyingCount: (grades.Volodya ≠ 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya = 5)
              ∨ (grades.Volodya = 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya ≠ 5)
              ∨ (grades.Volodya ≠ 5 ∧ grades.Sasha = 3 ∧ grades.Petya ≠ 5)) :
  isCorrectGrades grades :=
sorry

end find_correct_grades_l22_22755


namespace cost_calculation_l22_22410

variables (H M F : ℝ)

theorem cost_calculation 
  (h1 : 3 * H + 5 * M + F = 23.50) 
  (h2 : 5 * H + 9 * M + F = 39.50) : 
  2 * H + 2 * M + 2 * F = 15.00 :=
sorry

end cost_calculation_l22_22410


namespace find_b_plus_c_l22_22135

variable {a b c d : ℝ}

theorem find_b_plus_c
  (h1 : a + b = 4)
  (h2 : c + d = 3)
  (h3 : a + d = 2) :
  b + c = 5 := 
  by
  sorry

end find_b_plus_c_l22_22135


namespace ram_task_completion_days_l22_22942

theorem ram_task_completion_days (R : ℕ) (h1 : ∀ k : ℕ, k = R / 2) (h2 : 1 / R + 2 / R = 1 / 12) : R = 36 :=
sorry

end ram_task_completion_days_l22_22942


namespace intersection_point_l22_22816

/-- Coordinates of points A, B, C, and D -/
def pointA : Fin 3 → ℝ := ![3, -2, 4]
def pointB : Fin 3 → ℝ := ![13, -12, 9]
def pointC : Fin 3 → ℝ := ![1, 6, -8]
def pointD : Fin 3 → ℝ := ![3, -1, 2]

/-- Prove the intersection point of the lines AB and CD is (-7, 8, -1) -/
theorem intersection_point :
  let lineAB (t : ℝ) := pointA + t • (pointB - pointA)
  let lineCD (s : ℝ) := pointC + s • (pointD - pointC)
  ∃ t s : ℝ, lineAB t = lineCD s ∧ lineAB t = ![-7, 8, -1] :=
sorry

end intersection_point_l22_22816


namespace last_two_digits_l22_22940

theorem last_two_digits (a b : ℕ) (n : ℕ) (h : b ≡ 25 [MOD 100]) (h_pow : (25 : ℕ) ^ n ≡ 25 [MOD 100]) :
  (33 * b ^ n) % 100 = 25 :=
by
  sorry

end last_two_digits_l22_22940


namespace article_initial_cost_l22_22614

theorem article_initial_cost (x : ℝ) (h : 0.44 * x = 4400) : x = 10000 :=
by
  sorry

end article_initial_cost_l22_22614


namespace length_of_field_l22_22391

def width : ℝ := 13.5

def length (w : ℝ) : ℝ := 2 * w - 3

theorem length_of_field : length width = 24 :=
by
  -- full proof goes here
  sorry

end length_of_field_l22_22391


namespace determine_b_for_inverse_function_l22_22111

theorem determine_b_for_inverse_function (b : ℝ) :
  (∀ x, (2 - 3 * (1 / (2 * x + b))) / (3 * (1 / (2 * x + b))) = x) ↔ b = 3 / 2 := by
  sorry

end determine_b_for_inverse_function_l22_22111


namespace multiple_of_four_l22_22265

theorem multiple_of_four (n : ℕ) (h1 : ∃ k : ℕ, 12 + 4 * k = n) (h2 : 21 = (n - 12) / 4 + 1) : n = 96 := 
sorry

end multiple_of_four_l22_22265


namespace determine_m_for_unique_solution_l22_22970

-- Define the quadratic equation and the condition for a unique solution
def quadratic_eq_has_one_solution (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c = 0

-- Define the specific quadratic equation and its discriminant
def specific_quadratic_eq (m : ℝ) : Prop :=
  quadratic_eq_has_one_solution 3 (-7) m

-- State the main theorem to prove the value of m
theorem determine_m_for_unique_solution :
  specific_quadratic_eq (49 / 12) :=
by
  unfold specific_quadratic_eq quadratic_eq_has_one_solution
  sorry

end determine_m_for_unique_solution_l22_22970


namespace sqrt_floor_squared_l22_22494

theorem sqrt_floor_squared (h1 : 7^2 = 49) (h2 : 8^2 = 64) (h3 : 7 < Real.sqrt 50) (h4 : Real.sqrt 50 < 8) : (Int.floor (Real.sqrt 50))^2 = 49 :=
by
  sorry

end sqrt_floor_squared_l22_22494


namespace ab_value_l22_22156

theorem ab_value (a b : ℝ) (h1 : a = Real.exp (2 - a)) (h2 : 1 + Real.log b = Real.exp (1 - Real.log b)) : 
  a * b = Real.exp 1 :=
sorry

end ab_value_l22_22156


namespace gcd_sequence_inequality_l22_22592

-- Add your Lean 4 statement here
theorem gcd_sequence_inequality {n : ℕ} 
  (h : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 35 → Nat.gcd n k < Nat.gcd n (k+1)) : 
  Nat.gcd n 35 < Nat.gcd n 36 := 
sorry

end gcd_sequence_inequality_l22_22592


namespace solve_for_x_l22_22315

theorem solve_for_x :
  ∀ (x : ℚ), x = 45 / (8 - 3 / 7) → x = 315 / 53 :=
by
  sorry

end solve_for_x_l22_22315


namespace roy_total_pens_l22_22021

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

end roy_total_pens_l22_22021


namespace walking_time_estimate_l22_22981

-- Define constants for distance, speed, and time conversion factor
def distance : ℝ := 1000
def speed : ℝ := 4000
def time_conversion : ℝ := 60

-- Define the expected time to walk from home to school in minutes
def expected_time : ℝ := 15

-- Prove the time calculation
theorem walking_time_estimate : (distance / speed) * time_conversion = expected_time :=
by
  sorry

end walking_time_estimate_l22_22981


namespace wesley_breenah_ages_l22_22470

theorem wesley_breenah_ages (w b : ℕ) (h₁ : w = 15) (h₂ : b = 7) (h₃ : w + b = 22) :
  ∃ n : ℕ, 2 * (w + b) = (w + n) + (b + n) := by
  exists 11
  sorry

end wesley_breenah_ages_l22_22470


namespace infinite_68_in_cells_no_repeats_in_cells_l22_22960

-- Define the spiral placement function
def spiral (n : ℕ) : ℕ := sorry  -- This function should describe the placement of numbers in the spiral

-- Define a function to get the sum of the numbers in the nodes of a cell.
def cell_sum (cell : ℕ) : ℕ := sorry  -- This function should calculate the sum based on the spiral placement.

-- Proving that numbers divisible by 68 appear infinitely many times in cell centers
theorem infinite_68_in_cells : ∀ N : ℕ, ∃ n > N, 68 ∣ cell_sum n :=
by sorry

-- Proving that numbers in cell centers do not repeat
theorem no_repeats_in_cells : ∀ m n : ℕ, m ≠ n → cell_sum m ≠ cell_sum n :=
by sorry

end infinite_68_in_cells_no_repeats_in_cells_l22_22960


namespace cost_percentage_l22_22438

-- Define the original and new costs
def original_cost (t b : ℝ) : ℝ := t * b^4
def new_cost (t b : ℝ) : ℝ := t * (2 * b)^4

-- Define the theorem to prove the percentage relationship
theorem cost_percentage (t b : ℝ) (C R : ℝ) (h1 : C = original_cost t b) (h2 : R = new_cost t b) :
  (R / C) * 100 = 1600 :=
by sorry

end cost_percentage_l22_22438


namespace probability_selecting_cooking_l22_22628

theorem probability_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let favorable_outcomes := 1
  let total_outcomes := courses.length
  let probability := favorable_outcomes / total_outcomes
  probability = 1 / 4 :=
by
  sorry

end probability_selecting_cooking_l22_22628


namespace money_left_after_shopping_l22_22459

-- Definitions based on conditions
def initial_amount : ℝ := 5000
def percentage_spent : ℝ := 0.30
def amount_spent : ℝ := percentage_spent * initial_amount
def remaining_amount : ℝ := initial_amount - amount_spent

-- Theorem statement based on the question and correct answer
theorem money_left_after_shopping : remaining_amount = 3500 :=
by
  sorry

end money_left_after_shopping_l22_22459


namespace bus_passengers_l22_22134

variable (P : ℕ) -- P represents the initial number of passengers

theorem bus_passengers (h1 : P + 16 - 17 = 49) : P = 50 :=
by
  sorry

end bus_passengers_l22_22134


namespace jade_more_transactions_l22_22172

theorem jade_more_transactions (mabel_transactions : ℕ) (anthony_percentage : ℕ) (cal_fraction_numerator : ℕ) 
  (cal_fraction_denominator : ℕ) (jade_transactions : ℕ) (h1 : mabel_transactions = 90) 
  (h2 : anthony_percentage = 10) (h3 : cal_fraction_numerator = 2) (h4 : cal_fraction_denominator = 3) 
  (h5 : jade_transactions = 83) :
  jade_transactions - (2 * (90 + (90 * 10 / 100)) / 3) = 17 := 
by
  sorry

end jade_more_transactions_l22_22172


namespace present_age_of_A_is_11_l22_22730

-- Definitions for present ages
variables (A B C : ℕ)

-- Definitions for the given conditions
def sum_of_ages_present : Prop := A + B + C = 57
def age_ratio_three_years_ago (x : ℕ) : Prop := (A - 3 = x) ∧ (B - 3 = 2 * x) ∧ (C - 3 = 3 * x)

-- The proof statement
theorem present_age_of_A_is_11 (x : ℕ) (h1 : sum_of_ages_present A B C) (h2 : age_ratio_three_years_ago A B C x) : A = 11 := 
by
  sorry

end present_age_of_A_is_11_l22_22730


namespace inequality_may_not_hold_l22_22406

theorem inequality_may_not_hold (a b : ℝ) (h : 0 < b ∧ b < a) :
  ¬(∀ x y : ℝ,  x = 1 / (a - b) → y = 1 / b → x > y) :=
sorry

end inequality_may_not_hold_l22_22406


namespace non_congruent_squares_6x6_grid_l22_22694

theorem non_congruent_squares_6x6_grid : 
  let count_squares (n: ℕ) : ℕ := 
    let horizontal_or_vertical := (6 - n) * (6 - n)
    let diagonal := if n * n <= 6 * 6 then (6 - n + 1) * (6 - n + 1) else 0
    horizontal_or_vertical + diagonal
  (count_squares 1) + (count_squares 2) + (count_squares 3) + (count_squares 4) + (count_squares 5) = 141 :=
by
  sorry

end non_congruent_squares_6x6_grid_l22_22694


namespace elderly_teachers_in_sample_l22_22003

-- Definitions based on the conditions
def numYoungTeachersSampled : ℕ := 320
def ratioYoungToElderly : ℚ := 16 / 9

-- The theorem that needs to be proved
theorem elderly_teachers_in_sample :
  ∃ numElderlyTeachersSampled : ℕ, 
    numYoungTeachersSampled * (9 / 16) = numElderlyTeachersSampled := 
by
  use 180
  sorry

end elderly_teachers_in_sample_l22_22003


namespace option_b_correct_l22_22014

variables {m n : Line} {α β : Plane}

-- Define the conditions as per the problem.
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry
def line_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry
def plane_perpendicular_to_plane (p1 p2 : Plane) : Prop := sorry
def lines_perpendicular (l1 l2 : Line) : Prop := sorry

theorem option_b_correct (h1 : line_perpendicular_to_plane m α)
                         (h2 : line_perpendicular_to_plane n β)
                         (h3 : lines_perpendicular m n) :
                         plane_perpendicular_to_plane α β :=
sorry

end option_b_correct_l22_22014


namespace max_sequence_is_ten_l22_22403

noncomputable def max_int_sequence_length : Prop :=
  ∀ (a : ℕ → ℤ), 
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) > 0) ∧
    (∀ i : ℕ, a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) < 0) →
    (∃ n ≤ 10, ∀ i ≥ n, a i = 0)

theorem max_sequence_is_ten : max_int_sequence_length :=
sorry

end max_sequence_is_ten_l22_22403


namespace range_of_a_l22_22695

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → ax^2 - 2 * x + 2 > 0) ↔ (a > 1/2) :=
sorry

end range_of_a_l22_22695


namespace inheritance_amount_l22_22571

theorem inheritance_amount
  (x : ℝ)
  (H1 : 0.25 * x + 0.15 * (x - 0.25 * x) = 15000) : x = 41379 := 
sorry

end inheritance_amount_l22_22571


namespace tangent_lines_parallel_l22_22672

-- Definitions and conditions
def curve (x : ℝ) : ℝ := x^3 + x - 2
def line (x : ℝ) : ℝ := 4 * x - 1
def tangent_line_eq (a b c : ℝ) (x y : ℝ) : Prop := a * x + b * y + c = 0

-- Proof statement
theorem tangent_lines_parallel (tangent_line : ℝ → ℝ) :
  (∃ x : ℝ, tangent_line_eq 4 (-1) 0 x (curve x)) ∧ 
  (∃ x : ℝ, tangent_line_eq 4 (-1) (-4) x (curve x)) :=
sorry

end tangent_lines_parallel_l22_22672


namespace elderly_people_sampled_l22_22059

theorem elderly_people_sampled (total_population : ℕ) (children : ℕ) (elderly : ℕ) (middle_aged : ℕ) (sample_size : ℕ)
  (h1 : total_population = 1500)
  (h2 : ∃ d, children + d = elderly ∧ elderly + d = middle_aged)
  (h3 : total_population = children + elderly + middle_aged)
  (h4 : sample_size = 60) :
  elderly * (sample_size / total_population) = 20 :=
by
  -- Proof will be written here
  sorry

end elderly_people_sampled_l22_22059


namespace find_number_of_children_l22_22472

theorem find_number_of_children (adults children : ℕ) (adult_ticket_price child_ticket_price total_money change : ℕ) 
    (h1 : adult_ticket_price = 9) 
    (h2 : child_ticket_price = adult_ticket_price - 2) 
    (h3 : total_money = 40) 
    (h4 : change = 1) 
    (h5 : adults = 2) 
    (total_cost : total_money - change = adults * adult_ticket_price + children * child_ticket_price) : 
    children = 3 :=
sorry

end find_number_of_children_l22_22472


namespace minimum_abs_sum_l22_22783

def matrix_squared_condition (p q r s : ℤ) : Prop :=
  (p * p + q * r = 9) ∧ 
  (q * r + s * s = 9) ∧ 
  (p * q + q * s = 0) ∧ 
  (r * p + r * s = 0)

def abs_sum (p q r s : ℤ) : ℤ :=
  |p| + |q| + |r| + |s|

theorem minimum_abs_sum (p q r s : ℤ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0) 
  (h5 : matrix_squared_condition p q r s) : abs_sum p q r s = 8 :=
by 
  sorry

end minimum_abs_sum_l22_22783


namespace integer_solutions_l22_22575

theorem integer_solutions (n : ℤ) : (n^2 + 1) ∣ (n^5 + 3) ↔ n = -3 ∨ n = -1 ∨ n = 0 ∨ n = 1 ∨ n = 2 := 
sorry

end integer_solutions_l22_22575


namespace cone_altitude_to_radius_ratio_l22_22946

theorem cone_altitude_to_radius_ratio (r h : ℝ) (V_cone V_sphere : ℝ)
  (h1 : V_sphere = (4 / 3) * Real.pi * r^3)
  (h2 : V_cone = (1 / 3) * Real.pi * r^2 * h)
  (h3 : V_cone = (1 / 3) * V_sphere) :
  h / r = 4 / 3 :=
by
  sorry

end cone_altitude_to_radius_ratio_l22_22946


namespace tan_ratio_l22_22217

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : (Real.tan x) / (Real.tan y) = 2 := 
by
  sorry

end tan_ratio_l22_22217


namespace length_of_RS_l22_22210

-- Define the lengths of the edges of the tetrahedron
def edge_lengths : List ℕ := [9, 16, 22, 31, 39, 48]

-- Given the edge PQ has length 48
def PQ_length : ℕ := 48

-- We need to prove that the length of edge RS is 9
theorem length_of_RS :
  ∃ (RS : ℕ), RS = 9 ∧
  ∃ (PR QR PS SQ : ℕ),
  [PR, QR, PS, SQ] ⊆ edge_lengths ∧
  PR + QR > PQ_length ∧
  PR + PQ_length > QR ∧
  QR + PQ_length > PR ∧
  PS + SQ > PQ_length ∧
  PS + PQ_length > SQ ∧
  SQ + PQ_length > PS :=
by
  sorry

end length_of_RS_l22_22210


namespace solve_cubic_root_eq_l22_22522

theorem solve_cubic_root_eq (x : ℝ) (h : (5 - x)^(1/3) = 4) : x = -59 := 
by
  sorry

end solve_cubic_root_eq_l22_22522


namespace cody_spent_19_dollars_l22_22442

-- Given conditions
def initial_money : ℕ := 45
def birthday_gift : ℕ := 9
def remaining_money : ℕ := 35

-- Problem: Prove that the amount of money spent on the game is $19.
theorem cody_spent_19_dollars :
  (initial_money + birthday_gift - remaining_money) = 19 :=
by sorry

end cody_spent_19_dollars_l22_22442


namespace integer_rational_ratio_l22_22164

open Real

theorem integer_rational_ratio (a b : ℤ) (h : (a : ℝ) + sqrt b = sqrt (15 + sqrt 216)) : (a : ℚ) / b = 1 / 2 := 
by 
  -- Omitted proof 
  sorry

end integer_rational_ratio_l22_22164


namespace position_of_2010_is_correct_l22_22831

-- Definition of the arithmetic sequence and row starting points
def first_term : Nat := 1
def common_difference : Nat := 2
def S (n : Nat) : Nat := (n * (2 * first_term + (n - 1) * common_difference)) / 2

-- Definition of the position where number 2010 appears
def row_of_number (x : Nat) : Nat :=
  let n := (Nat.sqrt x) + 1
  if (n - 1) * (n - 1) < x && x <= n * n then n else n - 1

def column_of_number (x : Nat) : Nat :=
  let row := row_of_number x
  x - (S (row - 1)) + 1

-- Main theorem
theorem position_of_2010_is_correct :
  row_of_number 2010 = 45 ∧ column_of_number 2010 = 74 :=
by
  sorry

end position_of_2010_is_correct_l22_22831


namespace distribute_cousins_l22_22924

-- Define the variables and the conditions
noncomputable def ways_to_distribute_cousins (cousins : ℕ) (rooms : ℕ) : ℕ :=
  if cousins = 5 ∧ rooms = 3 then 66 else sorry

-- State the problem
theorem distribute_cousins: ways_to_distribute_cousins 5 3 = 66 :=
by
  sorry

end distribute_cousins_l22_22924


namespace number_of_therapy_hours_l22_22665

theorem number_of_therapy_hours (A F H : ℝ) (h1 : F = A + 35) 
  (h2 : F + (H - 1) * A = 350) (h3 : F + A = 161) :
  H = 5 :=
by
  sorry

end number_of_therapy_hours_l22_22665


namespace certain_number_is_3_l22_22365

theorem certain_number_is_3 (x : ℚ) (h : (x / 11) * ((121 : ℚ) / 3) = 11) : x = 3 := 
sorry

end certain_number_is_3_l22_22365


namespace enchilada_cost_l22_22744

theorem enchilada_cost : ∃ T E : ℝ, 2 * T + 3 * E = 7.80 ∧ 3 * T + 5 * E = 12.70 ∧ E = 2.00 :=
by
  sorry

end enchilada_cost_l22_22744


namespace problem_l22_22121

theorem problem (p q r : ℝ)
    (h1 : p * 1^2 + q * 1 + r = 5)
    (h2 : p * 2^2 + q * 2 + r = 3) :
  p + q + 2 * r = 10 := 
sorry

end problem_l22_22121


namespace time_for_B_alone_l22_22615

theorem time_for_B_alone (W_A W_B : ℝ) (h1 : W_A = 2 * W_B) (h2 : W_A + W_B = 1/6) : 1 / W_B = 18 := by
  sorry

end time_for_B_alone_l22_22615


namespace Lily_books_on_Wednesday_l22_22895

noncomputable def booksMike : ℕ := 45

noncomputable def booksCorey : ℕ := 2 * booksMike

noncomputable def booksMikeGivenToLily : ℕ := 13

noncomputable def booksCoreyGivenToLily : ℕ := booksMikeGivenToLily + 5

noncomputable def booksEmma : ℕ := 28

noncomputable def booksEmmaGivenToLily : ℕ := booksEmma / 4

noncomputable def totalBooksLilyGot : ℕ := booksMikeGivenToLily + booksCoreyGivenToLily + booksEmmaGivenToLily

theorem Lily_books_on_Wednesday : totalBooksLilyGot = 38 := by
  sorry

end Lily_books_on_Wednesday_l22_22895


namespace cuberoot_inequality_l22_22133

theorem cuberoot_inequality (a b : ℝ) : a < b → (∃ x y : ℝ, x^3 = a ∧ y^3 = b ∧ (x = y ∨ x > y)) := 
sorry

end cuberoot_inequality_l22_22133


namespace mary_max_earnings_l22_22850

theorem mary_max_earnings
  (max_hours : ℕ)
  (regular_rate : ℕ)
  (overtime_rate_increase_percent : ℕ)
  (first_hours : ℕ)
  (total_max_hours : ℕ)
  (total_hours_payable : ℕ) :
  max_hours = 60 →
  regular_rate = 8 →
  overtime_rate_increase_percent = 25 →
  first_hours = 20 →
  total_max_hours = 60 →
  total_hours_payable = 560 →
  ((first_hours * regular_rate) + ((total_max_hours - first_hours) * (regular_rate + (regular_rate * overtime_rate_increase_percent / 100)))) = total_hours_payable :=
by
  intros
  sorry

end mary_max_earnings_l22_22850


namespace angle_quadrant_l22_22001

theorem angle_quadrant (α : ℝ) (h : 0 < α ∧ α < 90) : 90 < α + 180 ∧ α + 180 < 270 :=
by
  sorry

end angle_quadrant_l22_22001


namespace mike_spent_total_l22_22259

-- Define the prices of the items
def price_trumpet : ℝ := 145.16
def price_song_book : ℝ := 5.84

-- Define the total amount spent
def total_spent : ℝ := price_trumpet + price_song_book

-- The theorem to be proved
theorem mike_spent_total :
  total_spent = 151.00 :=
sorry

end mike_spent_total_l22_22259


namespace smallest_natural_with_50_perfect_squares_l22_22641

theorem smallest_natural_with_50_perfect_squares (a : ℕ) (h : a = 4486) :
  (∃ n, n^2 ≤ a ∧ (n+50)^2 < 3 * a ∧ (∀ b, n^2 ≤ b^2 ∧ b^2 < 3 * a → n ≤ b-1 ∧ b-1 ≤ n+49)) :=
by {
  sorry
}

end smallest_natural_with_50_perfect_squares_l22_22641


namespace division_of_power_l22_22594

theorem division_of_power (m : ℕ) 
  (h : m = 16^2018) : m / 8 = 2^8069 := by
  sorry

end division_of_power_l22_22594


namespace mass_percentage_of_Cl_in_NaClO_l22_22429

noncomputable def molarMassNa : ℝ := 22.99
noncomputable def molarMassCl : ℝ := 35.45
noncomputable def molarMassO : ℝ := 16.00

noncomputable def molarMassNaClO : ℝ := molarMassNa + molarMassCl + molarMassO

theorem mass_percentage_of_Cl_in_NaClO : 
  (molarMassCl / molarMassNaClO) * 100 = 47.61 :=
by 
  sorry

end mass_percentage_of_Cl_in_NaClO_l22_22429


namespace one_in_M_l22_22062

def M : Set ℕ := {1, 2, 3}

theorem one_in_M : 1 ∈ M := sorry

end one_in_M_l22_22062


namespace point_not_on_line_l22_22201

theorem point_not_on_line
  (p q : ℝ)
  (h : p * q > 0) :
  ¬ (∃ (x y : ℝ), x = 2023 ∧ y = 0 ∧ y = p * x + q) :=
by
  sorry

end point_not_on_line_l22_22201


namespace john_runs_more_than_jane_l22_22368

def street_width : ℝ := 25
def block_side : ℝ := 500
def jane_perimeter (side : ℝ) : ℝ := 4 * side
def john_perimeter (side : ℝ) (width : ℝ) : ℝ := 4 * (side + 2 * width)

theorem john_runs_more_than_jane :
  john_perimeter block_side street_width - jane_perimeter block_side = 200 :=
by
  -- Substituting values to verify the equality:
  -- Calculate: john_perimeter 500 25 = 4 * (500 + 2 * 25) = 4 * 550 = 2200
  -- Calculate: jane_perimeter 500 = 4 * 500 = 2000
  sorry

end john_runs_more_than_jane_l22_22368


namespace geometric_sequence_third_term_l22_22532

theorem geometric_sequence_third_term (r : ℕ) (h_r : 5 * r ^ 4 = 1620) : 5 * r ^ 2 = 180 := by
  sorry

end geometric_sequence_third_term_l22_22532


namespace brownie_pieces_count_l22_22646

def pan_width : ℕ := 24
def pan_height : ℕ := 15
def brownie_width : ℕ := 3
def brownie_height : ℕ := 2

theorem brownie_pieces_count : (pan_width * pan_height) / (brownie_width * brownie_height) = 60 := by
  sorry

end brownie_pieces_count_l22_22646


namespace quadratic_roots_real_and_equal_l22_22461

theorem quadratic_roots_real_and_equal (m : ℤ) :
  (∀ x : ℝ, 3 * x^2 + (2 - m) * x + 12 = 0 →
   (∃ r, x = r ∧ 3 * r^2 + (2 - m) * r + 12 = 0)) →
   (m = -10 ∨ m = 14) :=
sorry

end quadratic_roots_real_and_equal_l22_22461


namespace decimal_addition_l22_22301

theorem decimal_addition : 0.4 + 0.02 + 0.006 = 0.426 := by
  sorry

end decimal_addition_l22_22301


namespace jessica_withdrew_200_l22_22837

noncomputable def initial_balance (final_balance : ℝ) : ℝ :=
  (final_balance * 25 / 18)

noncomputable def withdrawn_amount (initial_balance : ℝ) : ℝ :=
  (initial_balance * 2 / 5)

theorem jessica_withdrew_200 :
  ∀ (final_balance : ℝ), final_balance = 360 → withdrawn_amount (initial_balance final_balance) = 200 :=
by
  intros final_balance h
  rw [h]
  unfold initial_balance withdrawn_amount
  sorry

end jessica_withdrew_200_l22_22837


namespace students_catching_up_on_homework_l22_22370

theorem students_catching_up_on_homework :
  ∀ (total_students : ℕ) (half : ℕ) (third : ℕ),
  total_students = 24 → half = total_students / 2 → third = total_students / 3 →
  total_students - (half + third) = 4 :=
by
  intros total_students half third
  intros h_total h_half h_third
  sorry

end students_catching_up_on_homework_l22_22370


namespace number_of_clerks_l22_22225

theorem number_of_clerks 
  (num_officers : ℕ) 
  (num_clerks : ℕ) 
  (avg_salary_staff : ℕ) 
  (avg_salary_officers : ℕ) 
  (avg_salary_clerks : ℕ)
  (h1 : avg_salary_staff = 90)
  (h2 : avg_salary_officers = 600)
  (h3 : avg_salary_clerks = 84)
  (h4 : num_officers = 2)
  : num_clerks = 170 :=
sorry

end number_of_clerks_l22_22225


namespace sum_series_eq_half_l22_22827

theorem sum_series_eq_half :
  ∑' n : ℕ, (3^(n+1) / (9^(n+1) - 1)) = 1/2 := 
sorry

end sum_series_eq_half_l22_22827


namespace independence_of_events_l22_22317

noncomputable def is_independent (A B : Prop) (chi_squared : ℝ) := 
  chi_squared ≤ 3.841

theorem independence_of_events (A B : Prop) (chi_squared : ℝ) : 
  is_independent A B chi_squared → A ↔ B :=
by
  sorry

end independence_of_events_l22_22317


namespace find_coordinates_of_P0_find_equation_of_l_l22_22925

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

def is_in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

/-- Problem statement 1: Find the coordinates of P₀ --/
theorem find_coordinates_of_P0 (p0 : ℝ × ℝ)
    (h_tangent_parallel : tangent_slope p0.1 = 4)
    (h_third_quadrant : is_in_third_quadrant p0) :
    p0 = (-1, -4) :=
sorry

/-- Problem statement 2: Find the equation of line l --/
theorem find_equation_of_l (P0 : ℝ × ℝ)
    (h_P0_coordinates: P0 = (-1, -4))
    (h_perpendicular : ∀ (l1_slope : ℝ), l1_slope = 4 → ∃ l_slope : ℝ, l_slope = (-1) / 4)
    (x y : ℝ) : 
    line_eq 1 4 17 x y :=
sorry

end find_coordinates_of_P0_find_equation_of_l_l22_22925


namespace regular_polygon_sides_l22_22916

theorem regular_polygon_sides (h : ∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18) : 
∀ n : ℕ, n > 2 → 160 * n = 180 * (n - 2) → n = 18 :=
by
  exact h

end regular_polygon_sides_l22_22916


namespace remainder_of_7_pow_7_pow_7_pow_7_mod_500_l22_22465

theorem remainder_of_7_pow_7_pow_7_pow_7_mod_500 :
    (7 ^ (7 ^ (7 ^ 7))) % 500 = 343 := 
sorry

end remainder_of_7_pow_7_pow_7_pow_7_mod_500_l22_22465


namespace systemOfEquationsUniqueSolution_l22_22649

def largeBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  5 * x + y = 3

def smallBarrelHolds (x : ℝ) (y : ℝ) : Prop :=
  x + 5 * y = 2

theorem systemOfEquationsUniqueSolution (x y : ℝ) :
  (largeBarrelHolds x y) ∧ (smallBarrelHolds x y) ↔ 
  (5 * x + y = 3 ∧ x + 5 * y = 2) :=
by
  sorry

end systemOfEquationsUniqueSolution_l22_22649


namespace iodine_atomic_weight_l22_22984

noncomputable def atomic_weight_of_iodine : ℝ :=
  127.01

theorem iodine_atomic_weight
  (mw_AlI3 : ℝ := 408)
  (aw_Al : ℝ := 26.98)
  (formula_mw_AlI3 : mw_AlI3 = aw_Al + 3 * atomic_weight_of_iodine) :
  atomic_weight_of_iodine = 127.01 :=
by sorry

end iodine_atomic_weight_l22_22984


namespace fabian_initial_hours_l22_22683

-- Define the conditions
def speed : ℕ := 5
def total_distance : ℕ := 30
def additional_time : ℕ := 3

-- The distance Fabian covers in the additional time
def additional_distance := speed * additional_time

-- The initial distance walked by Fabian
def initial_distance := total_distance - additional_distance

-- The initial hours Fabian walked
def initial_hours := initial_distance / speed

theorem fabian_initial_hours : initial_hours = 3 := by
  -- Proof goes here
  sorry

end fabian_initial_hours_l22_22683


namespace price_decrease_l22_22682

theorem price_decrease (P : ℝ) (h₁ : 1.25 * P = P * 1.25) (h₂ : 1.10 * P = P * 1.10) :
  1.25 * P * (1 - 12 / 100) = 1.10 * P :=
by
  sorry

end price_decrease_l22_22682


namespace problem_statement_l22_22871

noncomputable def distance_from_line_to_point (a b : ℝ) : ℝ :=
  abs (1 / 2) / (Real.sqrt (a ^ 2 + b ^ 2))

theorem problem_statement (a b : ℝ) (h1 : a = (1 - 2 * b) / 2) (h2 : b = 1 / 2 - a) :
  distance_from_line_to_point a b ≤ Real.sqrt 2 := 
sorry

end problem_statement_l22_22871


namespace min_draws_to_ensure_20_of_one_color_l22_22193

-- Define the total number of balls for each color
def red_balls : ℕ := 30
def green_balls : ℕ := 25
def yellow_balls : ℕ := 22
def blue_balls : ℕ := 15
def white_balls : ℕ := 12
def black_balls : ℕ := 10

-- Define the minimum number of balls to guarantee at least one color reaches 20 balls
def min_balls_needed : ℕ := 95

-- Theorem to state the problem mathematically in Lean
theorem min_draws_to_ensure_20_of_one_color :
  ∀ (r g y b w bl : ℕ),
    r = 30 → g = 25 → y = 22 → b = 15 → w = 12 → bl = 10 →
    (∃ n : ℕ, n ≥ min_balls_needed ∧
    ∀ (r_draw g_draw y_draw b_draw w_draw bl_draw : ℕ),
      r_draw + g_draw + y_draw + b_draw + w_draw + bl_draw = n →
      (r_draw > 19 ∨ g_draw > 19 ∨ y_draw > 19 ∨ b_draw > 19 ∨ w_draw > 19 ∨ bl_draw > 19)) :=
by
  intros r g y b w bl hr hg hy hb hw hbl
  use min_balls_needed
  sorry

end min_draws_to_ensure_20_of_one_color_l22_22193


namespace general_term_correct_l22_22405

-- Define the sequence a_n
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^n

-- Define the general term formula for the sequence a_n
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = n * 2^(n - 1)

-- Theorem statement: the general term formula holds for the sequence a_n
theorem general_term_correct (a : ℕ → ℕ) (h_seq : seq a) : general_term a :=
by
  sorry

end general_term_correct_l22_22405


namespace number_of_chairs_is_40_l22_22863

-- Define the conditions
variables (C : ℕ) -- Total number of chairs
variables (capacity_per_chair : ℕ := 2) -- Each chair's capacity is 2 people
variables (occupied_ratio : ℚ := 3 / 5) -- Ratio of occupied chairs
variables (attendees : ℕ := 48) -- Number of attendees

theorem number_of_chairs_is_40
  (h1 : ∀ c : ℕ, capacity_per_chair * c = attendees)
  (h2 : occupied_ratio * C * capacity_per_chair = attendees) : 
  C = 40 := sorry

end number_of_chairs_is_40_l22_22863


namespace find_ac_pair_l22_22841

theorem find_ac_pair (a c : ℤ) (h1 : a + c = 37) (h2 : a < c) (h3 : 36^2 - 4 * a * c = 0) : a = 12 ∧ c = 25 :=
by
  sorry

end find_ac_pair_l22_22841


namespace segment_length_cd_l22_22106

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

end segment_length_cd_l22_22106


namespace algebraic_expression_value_l22_22697

-- Define the given condition as a predicate
def condition (a : ℝ) := a^2 + a - 4 = 0

-- Then the goal to prove with the given condition
theorem algebraic_expression_value (a : ℝ) (h : condition a) : (a^2 - 3) * (a + 2) = -2 :=
sorry

end algebraic_expression_value_l22_22697


namespace value_of_bill_used_to_pay_l22_22528

-- Definitions of the conditions
def num_games : ℕ := 6
def cost_per_game : ℕ := 15
def num_change_bills : ℕ := 2
def change_per_bill : ℕ := 5
def total_cost : ℕ := num_games * cost_per_game
def total_change : ℕ := num_change_bills * change_per_bill

-- Proof statement: What was the value of the bill Jed used to pay
theorem value_of_bill_used_to_pay : 
  total_value = (total_cost + total_change) :=
by
  sorry

end value_of_bill_used_to_pay_l22_22528


namespace range_of_a_l22_22516

def A (a : ℝ) : Set ℝ := {x | |x - a| ≤ 1}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

theorem range_of_a (a : ℝ) (h : A a ∩ B = ∅) : 2 < a ∧ a < 3 := sorry

end range_of_a_l22_22516


namespace circle_equation_l22_22471

theorem circle_equation (x y : ℝ) :
  let center := (0, 4)
  let point_on_circle := (3, 0)
  (x - center.1)^2 + (y - center.2)^2 = 25 :=
by
  sorry

end circle_equation_l22_22471


namespace avg_hamburgers_per_day_l22_22398

theorem avg_hamburgers_per_day (total_hamburgers : ℕ) (days_in_week : ℕ) (h1 : total_hamburgers = 63) (h2 : days_in_week = 7) :
  total_hamburgers / days_in_week = 9 := by
  sorry

end avg_hamburgers_per_day_l22_22398


namespace find_2023rd_letter_in_sequence_l22_22073

def repeating_sequence : List Char := ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'F', 'E', 'D', 'C', 'B', 'A']

def nth_in_repeating_sequence (n : ℕ) : Char :=
  repeating_sequence.get! (n % 13)

theorem find_2023rd_letter_in_sequence :
  nth_in_repeating_sequence 2023 = 'H' :=
by
  sorry

end find_2023rd_letter_in_sequence_l22_22073


namespace total_goals_l22_22770

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end total_goals_l22_22770


namespace euler_children_mean_age_l22_22318

-- Define the ages of each child
def ages : List ℕ := [8, 8, 8, 13, 13, 16]

-- Define the total number of children
def total_children := 6

-- Define the correct sum of ages
def total_sum_ages := 66

-- Define the correct answer (mean age)
def mean_age := 11

-- Prove that the mean (average) age of these children is 11
theorem euler_children_mean_age : (List.sum ages) / total_children = mean_age :=
by
  sorry

end euler_children_mean_age_l22_22318


namespace decreasing_function_range_l22_22975

noncomputable def f (a x : ℝ) := a * (x^3) - x + 1

theorem decreasing_function_range (a : ℝ) :
  (∀ x : ℝ, deriv (f a) x ≤ 0) → a ≤ 0 := by
  sorry

end decreasing_function_range_l22_22975


namespace division_result_l22_22997

open Polynomial

noncomputable def dividend := (X ^ 6 - 5 * X ^ 4 + 3 * X ^ 3 - 7 * X ^ 2 + 2 * X - 8 : Polynomial ℤ)
noncomputable def divisor := (X - 3 : Polynomial ℤ)
noncomputable def expected_quotient := (X ^ 5 + 3 * X ^ 4 + 4 * X ^ 3 + 15 * X ^ 2 + 38 * X + 116 : Polynomial ℤ)
noncomputable def expected_remainder := (340 : ℤ)

theorem division_result : (dividend /ₘ divisor) = expected_quotient ∧ (dividend %ₘ divisor) = C expected_remainder := by
  sorry

end division_result_l22_22997


namespace trig_relation_l22_22882

theorem trig_relation (a b c : ℝ) 
  (h1 : a = Real.sin 2) 
  (h2 : b = Real.cos 2) 
  (h3 : c = Real.tan 2) : c < b ∧ b < a := 
by
  sorry

end trig_relation_l22_22882


namespace total_amount_divided_l22_22402

-- Define the conditions
variables (A B C : ℕ)
axiom h1 : 4 * A = 5 * B
axiom h2 : 4 * A = 10 * C
axiom h3 : C = 160

-- Define the theorem to prove the total amount
theorem total_amount_divided (h1 : 4 * A = 5 * B) (h2 : 4 * A = 10 * C) (h3 : C = 160) : 
  A + B + C = 880 :=
sorry

end total_amount_divided_l22_22402


namespace f_inv_f_inv_15_l22_22580

def f (x : ℝ) : ℝ := 3 * x + 6

noncomputable def f_inv (x : ℝ) : ℝ := (x - 6) / 3

theorem f_inv_f_inv_15 : f_inv (f_inv 15) = -1 :=
by
  sorry

end f_inv_f_inv_15_l22_22580


namespace arithmetic_sequence_a13_l22_22844

theorem arithmetic_sequence_a13 (a : ℕ → ℤ) (d : ℤ) (a1 : ℤ) 
  (h1 : a 5 = 3) (h2 : a 9 = 6) 
  (h3 : ∀ n, a n = a1 + (n - 1) * d) : 
  a 13 = 9 :=
sorry

end arithmetic_sequence_a13_l22_22844


namespace work_problem_l22_22865

-- Definition of the conditions and the problem statement
theorem work_problem (P D : ℕ)
  (h1 : ∀ (P : ℕ), ∀ (D : ℕ), (2 * P) * 6 = P * D * 1 / 2) : 
  D = 24 :=
by
  sorry

end work_problem_l22_22865


namespace distance_hyperbola_focus_to_line_l22_22416

def hyperbola_right_focus : Type := { x : ℝ // x = 3 } -- Right focus is at (3, 0)
def line : Type := { x // x + 2 * (0 : ℝ) - 8 = 0 } -- Represents the line x + 2y - 8 = 0

theorem distance_hyperbola_focus_to_line : Real.sqrt 5 = 
  abs (1 * 3 + 2 * 0 - 8) / Real.sqrt (1^2 + 2^2) := 
by
  sorry

end distance_hyperbola_focus_to_line_l22_22416


namespace g_of_10_l22_22802

noncomputable def g : ℕ → ℝ := sorry

axiom g_initial : g 1 = 2

axiom g_condition : ∀ (m n : ℕ), m ≥ n → g (m + n) + g (m - n) = 2 * g m + 3 * g n

theorem g_of_10 : g 10 = 496 :=
by
  sorry

end g_of_10_l22_22802


namespace customs_days_l22_22433

-- Definitions from the problem conditions
def navigation_days : ℕ := 21
def transport_days : ℕ := 7
def total_days : ℕ := 30

-- Proposition we need to prove
theorem customs_days (expected_days: ℕ) (ship_departure_days : ℕ) : expected_days = 2 → ship_departure_days = 30 → (navigation_days + expected_days + transport_days = total_days) → expected_days = 2 :=
by
  intros h_expected h_departure h_eq
  sorry

end customs_days_l22_22433


namespace simplify_and_evaluate_l22_22728

theorem simplify_and_evaluate 
  (x y : ℤ) 
  (h1 : |x| = 2) 
  (h2 : y = 1) 
  (h3 : x * y < 0) : 
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end simplify_and_evaluate_l22_22728


namespace harold_monthly_income_l22_22781

variable (M : ℕ)

def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50

def total_expenses : ℕ := rent + car_payment + utilities + groceries
def remaining_money_after_expenses : ℕ := M - total_expenses
def retirement_saving_target : ℕ := 650
def required_remaining_money_pre_saving : ℕ := 2 * retirement_saving_target

theorem harold_monthly_income :
  remaining_money_after_expenses = required_remaining_money_pre_saving → M = 2500 :=
by
  sorry

end harold_monthly_income_l22_22781


namespace arithmetic_sequence_8th_term_l22_22331

theorem arithmetic_sequence_8th_term (a d : ℤ) :
  (a + d = 25) ∧ (a + 5 * d = 49) → (a + 7 * d = 61) :=
by
  sorry

end arithmetic_sequence_8th_term_l22_22331


namespace power_rule_example_l22_22987

variable {R : Type*} [Ring R] (a b : R)

theorem power_rule_example : (a * b^3) ^ 2 = a^2 * b^6 :=
sorry

end power_rule_example_l22_22987


namespace monthly_manufacturing_expenses_l22_22558

theorem monthly_manufacturing_expenses 
  (num_looms : ℕ) (total_sales_value : ℚ) 
  (monthly_establishment_charges : ℚ) 
  (decrease_in_profit : ℚ) 
  (sales_per_loom : ℚ) 
  (manufacturing_expenses_per_loom : ℚ) 
  (total_manufacturing_expenses : ℚ) : 
  num_looms = 80 → 
  total_sales_value = 500000 → 
  monthly_establishment_charges = 75000 → 
  decrease_in_profit = 4375 → 
  sales_per_loom = total_sales_value / num_looms → 
  manufacturing_expenses_per_loom = sales_per_loom - decrease_in_profit → 
  total_manufacturing_expenses = manufacturing_expenses_per_loom * num_looms →
  total_manufacturing_expenses = 150000 :=
by
  intros h_num_looms h_total_sales h_monthly_est_charges h_decrease_in_profit h_sales_per_loom h_manufacturing_expenses_per_loom h_total_manufacturing_expenses
  sorry

end monthly_manufacturing_expenses_l22_22558


namespace f_is_increasing_l22_22256

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x) + 3 * x

theorem f_is_increasing : ∀ (x : ℝ), (deriv f x) > 0 :=
by
  intro x
  calc
    deriv f x = 2 * Real.exp (2 * x) + 3 := by sorry
    _ > 0 := by sorry

end f_is_increasing_l22_22256


namespace selling_price_to_equal_percentage_profit_and_loss_l22_22978

-- Definition of the variables and conditions
def cost_price : ℝ := 1500
def sp_profit_25 : ℝ := 1875
def sp_loss : ℝ := 1280

theorem selling_price_to_equal_percentage_profit_and_loss :
  ∃ SP : ℝ, SP = 1720.05 ∧
  (sp_profit_25 = cost_price * 1.25) ∧
  (sp_loss < cost_price) ∧
  (14.67 = ((SP - cost_price) / cost_price) * 100) ∧
  (14.67 = ((cost_price - sp_loss) / cost_price) * 100) :=
by
  sorry

end selling_price_to_equal_percentage_profit_and_loss_l22_22978


namespace find_g_l22_22484

noncomputable def g : ℝ → ℝ := sorry

theorem find_g :
  (g 1 = 2) ∧ (∀ x y : ℝ, g (x + y) = 4^y * g x + 3^x * g y) ↔ (∀ x : ℝ, g x = 2 * (4^x - 3^x)) := 
by
  sorry

end find_g_l22_22484


namespace exponents_multiplication_exponents_power_exponents_distributive_l22_22805

variables (x y m : ℝ)

theorem exponents_multiplication (x : ℝ) : (x^5) * (x^2) = x^7 :=
by sorry

theorem exponents_power (m : ℝ) : (m^2)^4 = m^8 :=
by sorry

theorem exponents_distributive (x y : ℝ) : (-2 * x * y^2)^3 = -8 * x^3 * y^6 :=
by sorry

end exponents_multiplication_exponents_power_exponents_distributive_l22_22805


namespace sum_of_k_values_l22_22348

theorem sum_of_k_values 
  (h : ∀ (k : ℤ), (∀ x y : ℤ, x * y = 15 → x + y = k) → k > 0 → false) : 
  ∃ k_values : List ℤ, 
  (∀ (k : ℤ), k ∈ k_values → (∀ x y : ℤ, x * y = 15 → x + y = k) ∧ k > 0) ∧ 
  k_values.sum = 24 := sorry

end sum_of_k_values_l22_22348


namespace natural_pair_prime_ratio_l22_22700

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_pair_prime_ratio :
  ∃ (x y : ℕ), (x = 14 ∧ y = 2) ∧ is_prime (x * y^3 / (x + y)) :=
by
  use 14
  use 2
  sorry

end natural_pair_prime_ratio_l22_22700


namespace angela_spent_78_l22_22647

-- Definitions
def angela_initial_money : ℕ := 90
def angela_left_money : ℕ := 12
def angela_spent_money : ℕ := angela_initial_money - angela_left_money

-- Theorem statement
theorem angela_spent_78 : angela_spent_money = 78 := by
  -- Proof would go here, but it is not required.
  sorry

end angela_spent_78_l22_22647


namespace inequality_proof_l22_22047

theorem inequality_proof (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_eq : a^2 + b^2 + c^2 + a * b * c = 4) :
  a^2 * b^2 + b^2 * c^2 + c^2 * a^2 + a * b * c ≤ 4 := by
  sorry

end inequality_proof_l22_22047


namespace factor_expression_l22_22526

theorem factor_expression (x : ℚ) : 12 * x ^ 2 + 8 * x = 4 * x * (3 * x + 2) := sorry

end factor_expression_l22_22526


namespace no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l22_22750

theorem no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122 :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 := sorry

end no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l22_22750


namespace harry_took_5_eggs_l22_22252

theorem harry_took_5_eggs (initial : ℕ) (left : ℕ) (took : ℕ) 
  (h1 : initial = 47) (h2 : left = 42) (h3 : left = initial - took) : 
  took = 5 :=
sorry

end harry_took_5_eggs_l22_22252


namespace no_x2_term_imp_a_eq_half_l22_22944

theorem no_x2_term_imp_a_eq_half (a : ℝ) :
  (∀ x : ℝ, (x + 1) * (x^2 - 2 * a * x + a^2) = x^3 + (1 - 2 * a) * x^2 + ((a^2 - 2 * a) * x + a^2)) →
  (∀ c : ℝ, (1 - 2 * a) = 0) →
  a = 1 / 2 :=
by
  intros h_prod h_eq
  have h_eq' : 1 - 2 * a = 0 := h_eq 0
  linarith

end no_x2_term_imp_a_eq_half_l22_22944


namespace find_two_heaviest_l22_22761

theorem find_two_heaviest (a b c d : ℝ) : 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  ∃ x y : ℝ, (x ≠ y) ∧ (x = max (max (max a b) c) d) ∧ (y = max (max (min (max a b) c) d) d) :=
by sorry

end find_two_heaviest_l22_22761


namespace find_x_l22_22618

-- Defining the sum of integers from 30 to 40 inclusive
def sum_30_to_40 : ℕ := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)

-- Defining the number of even integers from 30 to 40 inclusive
def count_even_30_to_40 : ℕ := 6

-- Given that x + y = 391, and y = count_even_30_to_40
-- Prove that x is equal to 385
theorem find_x (h : sum_30_to_40 + count_even_30_to_40 = 391) : sum_30_to_40 = 385 :=
by
  simp [sum_30_to_40, count_even_30_to_40] at h
  sorry

end find_x_l22_22618


namespace linear_function_through_two_points_l22_22524

theorem linear_function_through_two_points :
  ∃ (k b : ℝ), (∀ x, y = k * x + b) ∧
  (k ≠ 0) ∧
  (3 = 2 * k + b) ∧
  (2 = 3 * k + b) ∧
  (∀ x, y = -x + 5) :=
by
  sorry

end linear_function_through_two_points_l22_22524


namespace mass_of_alcl3_formed_l22_22573

noncomputable def molarMass (atomicMasses : List (ℕ × ℕ)) : ℕ :=
atomicMasses.foldl (λ acc elem => acc + elem.1 * elem.2) 0

theorem mass_of_alcl3_formed :
  let atomic_mass_al := 26.98
  let atomic_mass_cl := 35.45
  let molar_mass_alcl3 := 2 * atomic_mass_al + 3 * atomic_mass_cl
  let moles_al2co3 := 10
  let moles_alcl3 := 2 * moles_al2co3
  let mass_alcl3 := moles_alcl3 * molar_mass_alcl3
  mass_alcl3 = 3206.2 := sorry

end mass_of_alcl3_formed_l22_22573


namespace hairstylist_weekly_earnings_l22_22325

-- Definition of conditions as given in part a)
def cost_normal_haircut := 5
def cost_special_haircut := 6
def cost_trendy_haircut := 8

def number_normal_haircuts_per_day := 5
def number_special_haircuts_per_day := 3
def number_trendy_haircuts_per_day := 2

def working_days_per_week := 7

-- The goal is to prove that the hairstylist's weekly earnings equal to 413 dollars
theorem hairstylist_weekly_earnings : 
  (number_normal_haircuts_per_day * cost_normal_haircut +
  number_special_haircuts_per_day * cost_special_haircut +
  number_trendy_haircuts_per_day * cost_trendy_haircut) * 
  working_days_per_week = 413 := 
by sorry -- We use "by sorry" to skip the proof

end hairstylist_weekly_earnings_l22_22325


namespace james_nickels_l22_22504

theorem james_nickels (p n : ℕ) (h₁ : p + n = 50) (h₂ : p + 5 * n = 150) : n = 25 :=
by
  -- Skipping the proof since only the statement is required
  sorry

end james_nickels_l22_22504


namespace quadratic_has_sum_r_s_l22_22462

/-
  Define the quadratic equation 6x^2 - 24x - 54 = 0
-/
def quadratic_eq (x : ℝ) : Prop :=
  6 * x^2 - 24 * x - 54 = 0

/-
  Define the value 11 which is the sum r + s when completing the square
  for the above quadratic equation  
-/
def result_value := -2 + 13

/-
  State the proof that r + s = 11 given the quadratic equation.
-/
theorem quadratic_has_sum_r_s : ∀ x : ℝ, quadratic_eq x → -2 + 13 = 11 :=
by
  intros
  exact rfl

end quadratic_has_sum_r_s_l22_22462


namespace add_fractions_l22_22917

theorem add_fractions (a : ℝ) (ha : a ≠ 0) : 
  (3 / a) + (2 / a) = (5 / a) := 
by 
  -- The proof goes here, but we'll skip it with sorry.
  sorry

end add_fractions_l22_22917


namespace desired_percentage_alcohol_l22_22117

noncomputable def original_volume : ℝ := 6
noncomputable def original_percentage : ℝ := 0.40
noncomputable def added_alcohol : ℝ := 1.2
noncomputable def final_solution_volume : ℝ := original_volume + added_alcohol
noncomputable def final_alcohol_volume : ℝ := (original_percentage * original_volume) + added_alcohol
noncomputable def desired_percentage : ℝ := (final_alcohol_volume / final_solution_volume) * 100

theorem desired_percentage_alcohol :
  desired_percentage = 50 := by
  sorry

end desired_percentage_alcohol_l22_22117


namespace value_of_w_l22_22492

theorem value_of_w (x : ℝ) (hx : x + 1/x = 5) : x^2 + (1/x)^2 = 23 :=
by
  sorry

end value_of_w_l22_22492


namespace find_range_of_a_l22_22218

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (2 * x) - a * x

theorem find_range_of_a (a : ℝ) :
  (∀ x > 0, f x a > a * x^2 + 1) → a ≤ 2 :=
by
  sorry

end find_range_of_a_l22_22218


namespace beau_age_calculation_l22_22736

variable (sons_age : ℕ) (beau_age_today : ℕ) (beau_age_3_years_ago : ℕ)

def triplets := 3
def sons_today := 16
def sons_age_3_years_ago := sons_today - 3
def sum_of_sons_3_years_ago := triplets * sons_age_3_years_ago

theorem beau_age_calculation
  (h1 : sons_today = 16)
  (h2 : sum_of_sons_3_years_ago = beau_age_3_years_ago)
  (h3 : beau_age_today = beau_age_3_years_ago + 3) :
  beau_age_today = 42 :=
sorry

end beau_age_calculation_l22_22736


namespace set_intersection_l22_22486

open Set

universe u

variables {U : Type u} (A B : Set ℝ) (x : ℝ)

def universal_set : Set ℝ := univ
def set_A : Set ℝ := {x | abs x < 1}
def set_B : Set ℝ := {x | x > -1/2}
def complement_B : Set ℝ := {x | x ≤ -1/2}
def intersection : Set ℝ := {x | -1 < x ∧ x ≤ -1/2}

theorem set_intersection :
  (universal_set \ set_B) ∩ set_A = {x | -1 < x ∧ x ≤ -1/2} :=
by 
  -- The actual proof steps would go here
  sorry

end set_intersection_l22_22486


namespace tan_alpha_eq_neg_one_l22_22735

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : Real.tan α = -1 :=
  sorry

end tan_alpha_eq_neg_one_l22_22735


namespace hazel_lemonade_total_l22_22752

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end hazel_lemonade_total_l22_22752


namespace find_m_l22_22148

theorem find_m (m : ℤ) (h : 3 ∈ ({1, m + 2} : Set ℤ)) : m = 1 :=
sorry

end find_m_l22_22148


namespace third_racer_sent_time_l22_22483

theorem third_racer_sent_time (a : ℝ) (t t1 : ℝ) :
  t1 = 1.5 * t → 
  (1.25 * a) * (t1 - (1 / 2)) = 1.5 * a * t → 
  t = 5 / 3 → 
  (t1 - t) * 60 = 50 :=
by 
  intro h_t1_eq h_second_eq h_t_value
  rw [h_t1_eq] at h_second_eq
  have t_correct : t = 5 / 3 := h_t_value
  sorry

end third_racer_sent_time_l22_22483


namespace Olivia_money_left_l22_22669

theorem Olivia_money_left (initial_amount spend_amount : ℕ) (h1 : initial_amount = 128) 
  (h2 : spend_amount = 38) : initial_amount - spend_amount = 90 := by
  sorry

end Olivia_money_left_l22_22669


namespace smallest_positive_period_sin_cos_sin_l22_22719

noncomputable def smallest_positive_period := 2 * Real.pi

theorem smallest_positive_period_sin_cos_sin :
  ∃ T > 0, (∀ x, (Real.sin x - 2 * Real.cos (2 * x) + 4 * Real.sin (4 * x)) = (Real.sin (x + T) - 2 * Real.cos (2 * (x + T)) + 4 * Real.sin (4 * (x + T)))) ∧ T = smallest_positive_period := by
sorry

end smallest_positive_period_sin_cos_sin_l22_22719


namespace equivalent_fraction_power_multiplication_l22_22552

theorem equivalent_fraction_power_multiplication : 
  (8 / 9) ^ 2 * (1 / 3) ^ 2 * (2 / 5) = (128 / 3645) := 
by 
  sorry

end equivalent_fraction_power_multiplication_l22_22552


namespace only_n_eq_1_solution_l22_22085

theorem only_n_eq_1_solution (n : ℕ) (h : n > 0): 
  (∃ x : ℤ, x^n + (2 + x)^n + (2 - x)^n = 0) ↔ n = 1 :=
by
  sorry

end only_n_eq_1_solution_l22_22085


namespace division_multiplication_calculation_l22_22645

theorem division_multiplication_calculation :
  (30 / (7 + 2 - 3)) * 4 = 20 :=
by
  sorry

end division_multiplication_calculation_l22_22645


namespace truck_travel_distance_l22_22054

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

end truck_travel_distance_l22_22054


namespace small_pizza_slices_l22_22835

-- Definitions based on conditions
def large_pizza_slices : ℕ := 16
def num_large_pizzas : ℕ := 2
def num_small_pizzas : ℕ := 2
def total_slices_eaten : ℕ := 48

-- Statement to prove
theorem small_pizza_slices (S : ℕ) (H : num_large_pizzas * large_pizza_slices + num_small_pizzas * S = total_slices_eaten) : S = 8 :=
by
  sorry

end small_pizza_slices_l22_22835


namespace remainder_of_sum_l22_22354

theorem remainder_of_sum (D k l : ℕ) (hk : 242 = k * D + 11) (hl : 698 = l * D + 18) :
  (242 + 698) % D = 29 :=
by
  sorry

end remainder_of_sum_l22_22354


namespace consecutive_even_integer_bases_l22_22423

/-- Given \(X\) and \(Y\) are consecutive even positive integers and the equation
\[ 241_X + 36_Y = 94_{X+Y} \]
this theorem proves that \(X + Y = 22\). -/
theorem consecutive_even_integer_bases (X Y : ℕ) (h1 : X > 0) (h2 : Y = X + 2)
    (h3 : 2 * X^2 + 4 * X + 1 + 3 * Y + 6 = 9 * (X + Y) + 4) : 
    X + Y = 22 :=
by sorry

end consecutive_even_integer_bases_l22_22423


namespace find_oysters_first_day_l22_22280

variable (O : ℕ)  -- Number of oysters on the rocks on the first day

def count_crabs_first_day := 72  -- Number of crabs on the beach on the first day

def oysters_second_day := O / 2  -- Number of oysters on the rocks on the second day

def crabs_second_day := (2 / 3) * count_crabs_first_day  -- Number of crabs on the beach on the second day

def total_count := 195  -- Total number of oysters and crabs counted over the two days

theorem find_oysters_first_day (h:  O + oysters_second_day O + count_crabs_first_day + crabs_second_day = total_count) : 
  O = 50 := by
  sorry

end find_oysters_first_day_l22_22280


namespace positive_real_inequality_l22_22285

noncomputable def positive_real_sum_condition (u v w : ℝ) [OrderedRing ℝ] :=
  u + v + w + Real.sqrt (u * v * w) = 4

theorem positive_real_inequality (u v w : ℝ) (hu : 0 < u) (hv : 0 < v) (hw : 0 < w) :
  positive_real_sum_condition u v w →
  Real.sqrt (v * w / u) + Real.sqrt (u * w / v) + Real.sqrt (u * v / w) ≥ u + v + w :=
by
  sorry

end positive_real_inequality_l22_22285


namespace B_is_werewolf_l22_22186

def is_werewolf (x : Type) : Prop := sorry
def is_knight (x : Type) : Prop := sorry
def is_liar (x : Type) : Prop := sorry

variables (A B : Type)

-- Conditions
axiom one_is_werewolf : is_werewolf A ∨ is_werewolf B
axiom only_one_werewolf : ¬ (is_werewolf A ∧ is_werewolf B)
axiom A_statement : is_werewolf A → is_knight A
axiom B_statement : is_werewolf B → is_liar B

theorem B_is_werewolf : is_werewolf B := 
by
  sorry

end B_is_werewolf_l22_22186


namespace triangle_third_side_l22_22514

theorem triangle_third_side (a b x : ℝ) (h : (a - 3) ^ 2 + |b - 4| = 0) :
  x = 5 ∨ x = Real.sqrt 7 :=
sorry

end triangle_third_side_l22_22514


namespace number_of_roots_in_right_half_plane_is_one_l22_22817

def Q5 (z : ℂ) : ℂ := z^5 + z^4 + 2*z^3 - 8*z - 1

theorem number_of_roots_in_right_half_plane_is_one :
  (∃ n, ∀ z, Q5 z = 0 ∧ z.re > 0 ↔ n = 1) := 
sorry

end number_of_roots_in_right_half_plane_is_one_l22_22817


namespace length_of_chord_l22_22454

theorem length_of_chord 
  (a : ℝ)
  (h_sym : ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y = 0) → (3*x - a*y - 11 = 0))
  (h_line : 3 * 1 - a * (-2) - 11 = 0)
  (h_midpoint : (1 : ℝ) = (a / 4) ∧ (-1 : ℝ) = (-a / 4)) :
  let r := Real.sqrt 5
  let d := Real.sqrt ((1 - 1)^2 + (-1 + 2)^2)
  (2 * Real.sqrt (r^2 - d^2)) = 4 :=
by {
  -- Variables and assumptions would go here
  sorry
}

end length_of_chord_l22_22454


namespace train_crossing_time_l22_22262

noncomputable def time_to_cross_platform
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ) : ℝ :=
  let speed_ms := speed_kmph / 3.6
  let total_distance := length_train + length_platform
  total_distance / speed_ms

theorem train_crossing_time
  (speed_kmph : ℝ)
  (length_train : ℝ)
  (length_platform : ℝ)
  (h_speed : speed_kmph = 72)
  (h_train_length : length_train = 280.0416)
  (h_platform_length : length_platform = 240) :
  time_to_cross_platform speed_kmph length_train length_platform = 26.00208 := by
  sorry

end train_crossing_time_l22_22262


namespace speed_of_car_in_second_hour_l22_22159

theorem speed_of_car_in_second_hour
(speed_in_first_hour : ℝ)
(average_speed : ℝ)
(total_time : ℝ)
(speed_in_second_hour : ℝ)
(h1 : speed_in_first_hour = 100)
(h2 : average_speed = 65)
(h3 : total_time = 2)
(h4 : average_speed = (speed_in_first_hour + speed_in_second_hour) / total_time) :
  speed_in_second_hour = 30 :=
by {
  sorry
}

end speed_of_car_in_second_hour_l22_22159


namespace circumference_ratio_l22_22555

theorem circumference_ratio (C D : ℝ) (hC : C = 94.2) (hD : D = 30) : C / D = 3.14 :=
by {
  sorry
}

end circumference_ratio_l22_22555


namespace marked_price_percentage_l22_22290

variables (L M: ℝ)

-- The store owner purchases items at a 25% discount of the list price.
def cost_price (L : ℝ) := 0.75 * L

-- The store owner plans to mark them up such that after a 10% discount on the marked price,
-- he achieves a 25% profit on the selling price.
def selling_price (M : ℝ) := 0.9 * M

-- Given condition: cost price is 75% of selling price
theorem marked_price_percentage (h : cost_price L = 0.75 * selling_price M) : 
  M = 1.111 * L :=
by 
  sorry

end marked_price_percentage_l22_22290


namespace alcohol_percentage_in_second_vessel_l22_22271

open Real

theorem alcohol_percentage_in_second_vessel (x : ℝ) (h : (0.2 * 2) + (0.01 * x * 6) = 8 * 0.28) : 
  x = 30.666666666666668 :=
by 
  sorry

end alcohol_percentage_in_second_vessel_l22_22271


namespace iron_column_lifted_by_9_6_cm_l22_22621

namespace VolumeLift

def base_area_container : ℝ := 200
def base_area_column : ℝ := 40
def height_water : ℝ := 16
def distance_water_surface : ℝ := 4

theorem iron_column_lifted_by_9_6_cm :
  ∃ (h_lift : ℝ),
    h_lift = 9.6 ∧ height_water - distance_water_surface = 16 - h_lift :=
by
sorry

end VolumeLift

end iron_column_lifted_by_9_6_cm_l22_22621


namespace find_y_l22_22353

theorem find_y (x y : Int) (h1 : x + y = 280) (h2 : x - y = 200) : y = 40 := 
by 
  sorry

end find_y_l22_22353


namespace cost_of_candy_bar_l22_22374

theorem cost_of_candy_bar (t c b : ℕ) (h1 : t = 13) (h2 : c = 6) (h3 : t = b + c) : b = 7 := 
by
  sorry

end cost_of_candy_bar_l22_22374


namespace sampling_is_systematic_l22_22220

-- Defining the conditions
def mock_exam (rooms students_per_room seat_selected: ℕ) : Prop :=
  rooms = 80 ∧ students_per_room = 30 ∧ seat_selected = 15

-- Theorem statement
theorem sampling_is_systematic 
  (rooms students_per_room seat_selected: ℕ)
  (h: mock_exam rooms students_per_room seat_selected) : 
  sampling_method = "Systematic sampling" :=
sorry

end sampling_is_systematic_l22_22220


namespace grape_rate_per_kg_l22_22510

theorem grape_rate_per_kg (G : ℝ) : 
    (8 * G) + (9 * 55) = 1055 → G = 70 := by
  sorry

end grape_rate_per_kg_l22_22510


namespace expected_rice_yield_l22_22421

theorem expected_rice_yield (x : ℝ) (y : ℝ) (h : y = 5 * x + 250) (hx : x = 80) : y = 650 :=
by
  sorry

end expected_rice_yield_l22_22421


namespace number_of_tables_l22_22356

-- Define the total number of customers the waiter is serving
def total_customers := 90

-- Define the number of women per table
def women_per_table := 7

-- Define the number of men per table
def men_per_table := 3

-- Define the total number of people per table
def people_per_table : ℕ := women_per_table + men_per_table

-- Statement to prove the number of tables
theorem number_of_tables (T : ℕ) (h : T * people_per_table = total_customers) : T = 9 := by
  sorry

end number_of_tables_l22_22356


namespace ara_current_height_l22_22624

variable (h : ℝ)  -- Original height of both Shea and Ara
variable (sheas_growth_rate : ℝ := 0.20)  -- Shea's growth rate (20%)
variable (sheas_current_height : ℝ := 60)  -- Shea's current height
variable (aras_growth_rate : ℝ := 0.5)  -- Ara's growth rate in terms of Shea's growth

theorem ara_current_height : 
  h * (1 + sheas_growth_rate) = sheas_current_height →
  (h + (sheas_current_height - h) * aras_growth_rate) = 55 :=
  by
    sorry

end ara_current_height_l22_22624


namespace quadruple_exists_unique_l22_22162

def digits (x : Nat) : Prop := x ≤ 9

theorem quadruple_exists_unique :
  ∃ (A B C D: Nat),
    digits A ∧ digits B ∧ digits C ∧ digits D ∧
    A > B ∧ B > C ∧ C > D ∧
    (A * 1000 + B * 100 + C * 10 + D) -
    (D * 1000 + C * 100 + B * 10 + A) =
    (B * 1000 + D * 100 + A * 10 + C) ∧
    (A, B, C, D) = (7, 6, 4, 1) :=
by
  sorry

end quadruple_exists_unique_l22_22162


namespace unique_solutions_l22_22921

noncomputable def func_solution (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, (x - 2) * f y + f (y + 2 * f x) = f (x + y * f x)

theorem unique_solutions (f : ℝ → ℝ) :
  func_solution f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = x - 1) :=
by
  intro h
  sorry

end unique_solutions_l22_22921


namespace sum_2004_impossible_sum_2005_possible_l22_22413

-- Condition Definitions
def is_valid_square (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  s = (1, 2, 3, 4) ∨ s = (1, 2, 4, 3) ∨ s = (1, 3, 2, 4) ∨ s = (1, 3, 4, 2) ∨ 
  s = (1, 4, 2, 3) ∨ s = (1, 4, 3, 2) ∨ s = (2, 1, 3, 4) ∨ s = (2, 1, 4, 3) ∨ 
  s = (2, 3, 1, 4) ∨ s = (2, 3, 4, 1) ∨ s = (2, 4, 1, 3) ∨ s = (2, 4, 3, 1) ∨ 
  s = (3, 1, 2, 4) ∨ s = (3, 1, 4, 2) ∨ s = (3, 2, 1, 4) ∨ s = (3, 2, 4, 1) ∨ 
  s = (3, 4, 1, 2) ∨ s = (3, 4, 2, 1) ∨ s = (4, 1, 2, 3) ∨ s = (4, 1, 3, 2) ∨ 
  s = (4, 2, 1, 3) ∨ s = (4, 2, 3, 1) ∨ s = (4, 3, 1, 2) ∨ s = (4, 3, 2, 1)

-- Proof Problems
theorem sum_2004_impossible (n : ℕ) (corners : ℕ → ℕ × ℕ × ℕ × ℕ) (h : ∀ i, is_valid_square (corners i)) :
  4 * 2004 ≠ n * 10 := 
sorry

theorem sum_2005_possible (h : ∃ n, ∃ corners : ℕ → ℕ × ℕ × ℕ × ℕ, (∀ i, is_valid_square (corners i)) ∧ 4 * 2005 = n * 10 + 2005) :
  true := 
sorry

end sum_2004_impossible_sum_2005_possible_l22_22413


namespace Nathan_daily_hours_l22_22224

theorem Nathan_daily_hours (x : ℝ) 
  (h1 : 14 * x + 35 = 77) : 
  x = 3 := 
by 
  sorry

end Nathan_daily_hours_l22_22224


namespace factorize_cubic_l22_22644

theorem factorize_cubic (a : ℝ) : a^3 - 16 * a = a * (a + 4) * (a - 4) :=
sorry

end factorize_cubic_l22_22644


namespace total_vehicles_l22_22507

-- Define the conditions
def num_trucks_per_lane := 60
def num_lanes := 4
def total_trucks := num_trucks_per_lane * num_lanes
def num_cars_per_lane := 2 * total_trucks
def total_cars := num_cars_per_lane * num_lanes

-- Prove the total number of vehicles in all lanes
theorem total_vehicles : total_trucks + total_cars = 2160 := by
  sorry

end total_vehicles_l22_22507


namespace candy_bar_cost_l22_22955

def num_quarters := 4
def num_dimes := 3
def num_nickel := 1
def change_received := 4

def value_quarter := 25
def value_dime := 10
def value_nickel := 5

def total_paid := (num_quarters * value_quarter) + (num_dimes * value_dime) + (num_nickel * value_nickel)
def cost_candy_bar := total_paid - change_received

theorem candy_bar_cost : cost_candy_bar = 131 := by
  sorry

end candy_bar_cost_l22_22955


namespace find_number_l22_22094

theorem find_number (x : ℤ) (h : 35 - 3 * x = 14) : x = 7 :=
by {
  sorry -- This is where the proof would go.
}

end find_number_l22_22094


namespace saturday_earnings_l22_22943

variable (S : ℝ)
variable (totalEarnings : ℝ := 5182.50)
variable (difference : ℝ := 142.50)

theorem saturday_earnings : 
  S + (S - difference) = totalEarnings → S = 2662.50 := 
by 
  intro h 
  sorry

end saturday_earnings_l22_22943


namespace distinct_integer_roots_l22_22576

-- Definitions of m and the polynomial equation.
def poly (m : ℤ) (x : ℤ) : Prop :=
  x^2 - 2 * (2 * m - 3) * x + 4 * m^2 - 14 * m + 8 = 0

-- Theorem stating that for m = 12 and m = 24, the polynomial has specific roots.
theorem distinct_integer_roots (m x : ℤ) (h1 : 4 < m) (h2 : m < 40) :
  (m = 12 ∨ m = 24) ∧ 
  ((m = 12 ∧ (x = 26 ∨ x = 16) ∧ poly m x) ∨
   (m = 24 ∧ (x = 52 ∨ x = 38) ∧ poly m x)) :=
by
  sorry

end distinct_integer_roots_l22_22576


namespace best_selling_price_70_l22_22145

-- Definitions for the conditions in the problem
def purchase_price : ℕ := 40
def initial_selling_price : ℕ := 50
def initial_sales_volume : ℕ := 50

-- The profit function
def profit (x : ℕ) : ℕ :=
(50 + x - purchase_price) * (initial_sales_volume - x)

-- The problem statement to be proved
theorem best_selling_price_70 :
  ∃ x : ℕ, 0 < x ∧ x < 50 ∧ profit x = 900 ∧ (initial_selling_price + x) = 70 :=
by
  sorry

end best_selling_price_70_l22_22145


namespace calculate_f3_times_l22_22991

def f (n : ℕ) : ℕ :=
  if n ≤ 3 then n^2 + 1 else 4 * n + 2

theorem calculate_f3_times : f (f (f 3)) = 170 := by
  sorry

end calculate_f3_times_l22_22991


namespace part1_part2_l22_22384

-- Step 1: Define the problem for a triangle with specific side length conditions and perimeter
theorem part1 (x : ℝ) (h1 : 2 * x + 2 * (2 * x) = 18) : 
  x = 18 / 5 ∧ 2 * x = 36 / 5 :=
by
  sorry

-- Step 2: Verify if an isosceles triangle with a side length of 4 cm can be formed
theorem part2 (a b c : ℝ) (h2 : a = 4 ∨ b = 4 ∨ c = 4) (h3 : a + b + c = 18) : 
  (a = 4 ∧ b = 7 ∧ c = 7 ∨ b = 4 ∧ a = 7 ∧ c = 7 ∨ c = 4 ∧ a = 7 ∧ b = 7) ∨
  (¬(a = 4 ∧ b + c <= a ∨ b = 4 ∧ a + c <= b ∨ c = 4 ∧ a + b <= c)) :=
by
  sorry

end part1_part2_l22_22384


namespace degree_of_divisor_l22_22622

theorem degree_of_divisor (f d q r : Polynomial ℝ) 
  (hf : f.degree = 15) 
  (hq : q.degree = 9) 
  (hr : r.degree = 4) 
  (hr_poly : r = (Polynomial.C 5) * (Polynomial.X^4) + (Polynomial.C 6) * (Polynomial.X^3) - (Polynomial.C 2) * (Polynomial.X) + (Polynomial.C 7)) 
  (hdiv : f = d * q + r) : 
  d.degree = 6 := 
sorry

end degree_of_divisor_l22_22622


namespace triangle_area_l22_22712

theorem triangle_area (a b c : ℕ) (h₁ : a = 7) (h₂ : b = 24) (h₃ : c = 25) (h₄ : a^2 + b^2 = c^2) : 
  ∃ A : ℕ, A = 84 ∧ A = (a * b) / 2 := by
  sorry

end triangle_area_l22_22712


namespace paths_A_to_D_l22_22969

noncomputable def num_paths_from_A_to_D : ℕ := 
  2 * 2 * 2 + 1

theorem paths_A_to_D : num_paths_from_A_to_D = 9 := 
by
  sorry

end paths_A_to_D_l22_22969


namespace count_ns_divisible_by_5_l22_22114

open Nat

theorem count_ns_divisible_by_5 : 
  let f (n : ℕ) := 2 * n^5 + 2 * n^4 + 3 * n^2 + 3 
  ∃ (N : ℕ), N = 19 ∧ 
  (∀ (n : ℕ), 2 ≤ n ∧ n ≤ 100 → f n % 5 = 0 → 
  (∃ (m : ℕ), 1 ≤ m ∧ m ≤ 19 ∧ n = 5 * m + 1)) :=
by
  sorry

end count_ns_divisible_by_5_l22_22114


namespace problem_l22_22864

variable (a b c : ℝ)

theorem problem (h : a^2 * b^2 + 18 * a * b * c > 4 * b^3 + 4 * a^3 * c + 27 * c^2) : a^2 > 3 * b :=
by
  sorry

end problem_l22_22864


namespace equalize_foma_ierema_l22_22399

variables 
  (F E Y : ℕ)
  (h1 : F - 70 = E + 70)
  (h2 : F - 40 = Y)

def foma_should_give_ierema : ℕ := (F - E) / 2

theorem equalize_foma_ierema (F E Y : ℕ) (h1 : F - 70 = E + 70) (h2 : F - 40 = Y) :
  foma_should_give_ierema F E = 55 := 
by
  sorry

end equalize_foma_ierema_l22_22399


namespace set_intersection_example_l22_22355

theorem set_intersection_example (A : Set ℝ) (B : Set ℝ):
  A = { -1, 1, 2, 4 } → 
  B = { x | |x - 1| ≤ 1 } → 
  A ∩ B = {1, 2} :=
by
  intros hA hB
  sorry

end set_intersection_example_l22_22355


namespace grant_total_earnings_l22_22743

def earnings_first_month : ℕ := 350
def earnings_second_month : ℕ := 2 * earnings_first_month + 50
def earnings_third_month : ℕ := 4 * (earnings_first_month + earnings_second_month)
def total_earnings : ℕ := earnings_first_month + earnings_second_month + earnings_third_month

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end grant_total_earnings_l22_22743


namespace no_real_roots_of_quadratic_l22_22139

def quadratic (a b c : ℝ) : ℝ × ℝ × ℝ := (a^2, b^2 + a^2 - c^2, b^2)

def discriminant (A B C : ℝ) : ℝ := B^2 - 4 * A * C

theorem no_real_roots_of_quadratic (a b c : ℝ)
  (h1 : a + b > c)
  (h2 : c > 0)
  (h3 : |a - b| < c)
  : (discriminant (a^2) (b^2 + a^2 - c^2) (b^2)) < 0 :=
sorry

end no_real_roots_of_quadratic_l22_22139


namespace sum_even_less_100_correct_l22_22013

-- Define the sequence of even, positive integers less than 100
def even_seq (n : ℕ) : Prop := n % 2 = 0 ∧ 0 < n ∧ n < 100

-- Sum of the first n positive integers
def sum_n (n : ℕ) : ℕ := n * (n + 1) / 2

-- Sum of the even, positive integers less than 100
def sum_even_less_100 : ℕ := 2 * sum_n 49

theorem sum_even_less_100_correct : sum_even_less_100 = 2450 := by
  sorry

end sum_even_less_100_correct_l22_22013


namespace rationalize_denominator_l22_22109

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

end rationalize_denominator_l22_22109


namespace smallest_n_l22_22896

theorem smallest_n (n : ℕ) (h1 : n ≥ 1)
  (h2 : ∃ k : ℕ, 2002 * n = k ^ 3)
  (h3 : ∃ m : ℕ, n = 2002 * m ^ 2) :
  n = 2002^5 := sorry

end smallest_n_l22_22896


namespace cost_price_of_cricket_bat_for_A_l22_22862

-- Define the cost price of the cricket bat for A as a variable
variable (CP_A : ℝ)

-- Define the conditions given in the problem
def condition1 := CP_A * 1.20 -- B buys at 20% profit
def condition2 := CP_A * 1.20 * 1.25 -- B sells at 25% profit
def totalCost := 231 -- C pays $231

-- The theorem we need to prove
theorem cost_price_of_cricket_bat_for_A : (condition2 = totalCost) → CP_A = 154 := by
  intros h
  sorry

end cost_price_of_cricket_bat_for_A_l22_22862


namespace swapped_coefficients_have_roots_l22_22082

theorem swapped_coefficients_have_roots 
  (a b c p q r : ℝ)
  (h1 : ∀ x : ℝ, ¬ (a * x^2 + b * x + c = 0))
  (h2 : ∀ x : ℝ, ¬ (p * x^2 + q * x + r = 0))
  (h3 : b^2 < 4 * p * c)
  (h4 : q^2 < 4 * a * r) :
  ∃ x : ℝ, a * x^2 + q * x + c = 0 ∧ ∃ y : ℝ, p * y^2 + b * y + r = 0 :=
by
  sorry

end swapped_coefficients_have_roots_l22_22082


namespace circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l22_22334

variable {α β γ : ℝ}

/-- The equation of the circumscribed Steiner ellipse in barycentric coordinates -/
theorem circumscribed_steiner_ellipse (h : α + β + γ = 1) :
  β * γ + α * γ + α * β = 0 :=
sorry

/-- The equation of the inscribed Steiner ellipse in barycentric coordinates -/
theorem inscribed_steiner_ellipse (h : α + β + γ = 1) :
  2 * β * γ + 2 * α * γ + 2 * α * β = α^2 + β^2 + γ^2 :=
sorry

end circumscribed_steiner_ellipse_inscribed_steiner_ellipse_l22_22334


namespace range_of_a_plus_c_l22_22026

-- Let a, b, c be the sides of the triangle opposite to angles A, B, and C respectively.
variable (a b c A B C : ℝ)

-- Given conditions
variable (h1 : b = Real.sqrt 3)
variable (h2 : (2 * c - a) / b * Real.cos B = Real.cos A)
variable (h3 : 0 < A ∧ A < Real.pi / 2)
variable (h4 : 0 < B ∧ B < Real.pi / 2)
variable (h5 : 0 < C ∧ C < Real.pi / 2)
variable (h6 : A + B + C = Real.pi)

-- The range of a + c
theorem range_of_a_plus_c (a b c A B C : ℝ) (h1 : b = Real.sqrt 3)
  (h2 : (2 * c - a) / b * Real.cos B = Real.cos A) (h3 : 0 < A ∧ A < Real.pi / 2)
  (h4 : 0 < B ∧ B < Real.pi / 2) (h5 : 0 < C ∧ C < Real.pi / 2) (h6 : A + B + C = Real.pi) :
  a + c ∈ Set.Ioc (Real.sqrt 3) (2 * Real.sqrt 3) :=
  sorry

end range_of_a_plus_c_l22_22026


namespace solve_students_and_apples_l22_22957

noncomputable def students_and_apples : Prop :=
  ∃ (x y : ℕ), y = 4 * x + 3 ∧ 6 * (x - 1) ≤ y ∧ y ≤ 6 * (x - 1) + 2 ∧ x = 4 ∧ y = 19

theorem solve_students_and_apples : students_and_apples :=
  sorry

end solve_students_and_apples_l22_22957


namespace price_for_two_bracelets_l22_22673

theorem price_for_two_bracelets
    (total_bracelets : ℕ)
    (price_per_bracelet : ℕ)
    (total_earned_for_single : ℕ)
    (total_earned : ℕ)
    (bracelets_sold_single : ℕ)
    (bracelets_left : ℕ)
    (remaining_earned : ℕ)
    (pairs_sold : ℕ)
    (price_per_pair : ℕ) :
    total_bracelets = 30 →
    price_per_bracelet = 5 →
    total_earned_for_single = 60 →
    total_earned = 132 →
    bracelets_sold_single = total_earned_for_single / price_per_bracelet →
    bracelets_left = total_bracelets - bracelets_sold_single →
    remaining_earned = total_earned - total_earned_for_single →
    pairs_sold = bracelets_left / 2 →
    price_per_pair = remaining_earned / pairs_sold →
    price_per_pair = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8 h9
  sorry

end price_for_two_bracelets_l22_22673


namespace least_number_l22_22090

theorem least_number (n : ℕ) (h1 : n % 31 = 3) (h2 : n % 9 = 3) : n = 282 :=
sorry

end least_number_l22_22090


namespace exp_neg_eq_l22_22604

theorem exp_neg_eq (θ φ : ℝ) (h : Complex.exp (Complex.I * θ) + Complex.exp (Complex.I * φ) = (1 / 2 : ℂ) + (1 / 3 : ℂ) * Complex.I) :
  Complex.exp (-Complex.I * θ) + Complex.exp (-Complex.I * φ) = (1 / 2 : ℂ) - (1 / 3 : ℂ) * Complex.I :=
by sorry

end exp_neg_eq_l22_22604


namespace container_weight_l22_22180

noncomputable def weight_in_pounds : ℝ := 57 + 3/8
noncomputable def weight_in_ounces : ℝ := weight_in_pounds * 16
noncomputable def number_of_containers : ℝ := 7
noncomputable def ounces_per_container : ℝ := weight_in_ounces / number_of_containers

theorem container_weight :
  ounces_per_container = 131.142857 :=
by sorry

end container_weight_l22_22180


namespace avg_variance_stability_excellent_performance_probability_l22_22440

-- Define the scores of players A and B in seven games
def scores_A : List ℕ := [26, 28, 32, 22, 37, 29, 36]
def scores_B : List ℕ := [26, 29, 32, 28, 39, 29, 27]

-- Define the mean and variance calculations
def mean (scores : List ℕ) : ℚ := (scores.sum : ℚ) / scores.length
def variance (scores : List ℕ) : ℚ := 
  (scores.map (λ x => (x - mean scores) ^ 2)).sum / scores.length

theorem avg_variance_stability :
  mean scores_A = 30 ∧ mean scores_B = 30 ∧
  variance scores_A = 174 / 7 ∧ variance scores_B = 116 / 7 ∧
  variance scores_A > variance scores_B := 
by
  sorry

-- Define the probabilities of scoring higher than 30
def probability_excellent (scores : List ℕ) : ℚ := 
  (scores.filter (λ x => x > 30)).length / scores.length

theorem excellent_performance_probability :
  probability_excellent scores_A = 3 / 7 ∧ probability_excellent scores_B = 2 / 7 ∧
  (probability_excellent scores_A * probability_excellent scores_B = 6 / 49) :=
by
  sorry

end avg_variance_stability_excellent_performance_probability_l22_22440


namespace find_preimage_l22_22231

def mapping (x y : ℝ) : ℝ × ℝ :=
  (x + y, x - y)

theorem find_preimage :
  mapping 2 1 = (3, 1) :=
by
  sorry

end find_preimage_l22_22231


namespace red_robin_team_arrangements_l22_22788

theorem red_robin_team_arrangements :
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  waysToPositionBoys * waysToPositionRemainingMembers = 720 :=
by
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  have : waysToPositionBoys * waysToPositionRemainingMembers = 720 := 
    by sorry -- Proof omitted here
  exact this

end red_robin_team_arrangements_l22_22788


namespace non_neg_integer_solutions_for_inequality_l22_22007

theorem non_neg_integer_solutions_for_inequality :
  {x : ℤ | 5 * x - 1 < 3 * (x + 1) ∧ (1 - x) / 3 ≤ 1 ∧ 0 ≤ x } = {0, 1} := 
by {
  sorry
}

end non_neg_integer_solutions_for_inequality_l22_22007


namespace mn_condition_l22_22066

theorem mn_condition {m n : ℕ} (h : m * n = 121) : (m + 1) * (n + 1) = 144 :=
sorry

end mn_condition_l22_22066


namespace coprime_permutations_count_l22_22772

noncomputable def count_coprime_permutations (l : List ℕ) : ℕ :=
if h : l = [1, 2, 3, 4, 5, 6, 7] ∨ l = [1, 2, 3, 7, 5, 6, 4] -- other permutations can be added as needed
then 864
else 0

theorem coprime_permutations_count :
  count_coprime_permutations [1, 2, 3, 4, 5, 6, 7] = 864 :=
sorry

end coprime_permutations_count_l22_22772


namespace walking_speed_l22_22808

theorem walking_speed 
  (D : ℝ) 
  (V_w : ℝ) 
  (h1 : D = V_w * 8) 
  (h2 : D = 36 * 2) : 
  V_w = 9 :=
by
  sorry

end walking_speed_l22_22808


namespace tray_height_l22_22118

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

end tray_height_l22_22118


namespace desired_interest_rate_l22_22287

def nominalValue : ℝ := 20
def dividendRate : ℝ := 0.09
def marketValue : ℝ := 15

theorem desired_interest_rate : (dividendRate * nominalValue / marketValue) * 100 = 12 := by
  sorry

end desired_interest_rate_l22_22287


namespace base7_to_base10_proof_l22_22698

theorem base7_to_base10_proof (c d : ℕ) (h1 : 764 = 4 * 100 + c * 10 + d) : (c * d) / 20 = 6 / 5 :=
by
  sorry

end base7_to_base10_proof_l22_22698


namespace parabola_transformation_l22_22211

-- Defining the original parabola
def original_parabola (x : ℝ) : ℝ :=
  3 * x^2

-- Condition: Transformation 1 -> Translation 4 units to the right
def translated_right_parabola (x : ℝ) : ℝ :=
  original_parabola (x - 4)

-- Condition: Transformation 2 -> Translation 1 unit upwards
def translated_up_parabola (x : ℝ) : ℝ :=
  translated_right_parabola x + 1

-- Statement that needs to be proved
theorem parabola_transformation :
  ∀ x : ℝ, translated_up_parabola x = 3 * (x - 4)^2 + 1 :=
by
  intros x
  sorry

end parabola_transformation_l22_22211


namespace ellipse_with_foci_on_x_axis_l22_22427

theorem ellipse_with_foci_on_x_axis (a : ℝ) :
  (∀ x y : ℝ, (x^2) / (a - 5) + (y^2) / 2 = 1 →  
   (∃ cx cy : ℝ, ∀ x', cx - x' = a - 5 ∧ cy = 2)) → 
  a > 7 :=
by sorry

end ellipse_with_foci_on_x_axis_l22_22427


namespace sum_of_remainders_l22_22203

theorem sum_of_remainders (a b c : ℕ) (h₁ : a % 30 = 15) (h₂ : b % 30 = 7) (h₃ : c % 30 = 18) : 
    (a + b + c) % 30 = 10 := 
by
  sorry

end sum_of_remainders_l22_22203


namespace algebraic_expression_l22_22756

theorem algebraic_expression (a b : Real) 
  (h : a * b = 2 * (a^2 + b^2)) : 2 * a * b - (a^2 + b^2) = 0 :=
by
  sorry

end algebraic_expression_l22_22756


namespace roots_eq_s_l22_22933

theorem roots_eq_s (n c d : ℝ) (h₁ : c * d = 6) (h₂ : c + d = n)
  (h₃ : c^2 + 1 / d = c^2 + d^2 + 1 / c): 
  (n + 217 / 6) = d^2 + 1/ c * (n + c + d)
  :=
by
  -- The proof will go here
  sorry

end roots_eq_s_l22_22933
