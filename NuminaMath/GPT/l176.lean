import Mathlib

namespace gcd_of_459_and_357_l176_176528

theorem gcd_of_459_and_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_of_459_and_357_l176_176528


namespace area_of_one_postcard_is_150_cm2_l176_176513

/-- Define the conditions of the problem. -/
def perimeter_of_stitched_postcard : ℕ := 70
def vertical_length_of_postcard : ℕ := 15

/-- Definition stating that postcards are attached horizontally and do not overlap. 
    This logically implies that the horizontal length gets doubled and perimeter is 2V + 4H. -/
def attached_horizontally (V H : ℕ) (P : ℕ) : Prop :=
  2 * V + 4 * H = P

/-- Main theorem stating the question and the derived answer,
    proving that the area of one postcard is 150 square centimeters. -/
theorem area_of_one_postcard_is_150_cm2 :
  ∃ (H : ℕ), attached_horizontally vertical_length_of_postcard H perimeter_of_stitched_postcard ∧
  (vertical_length_of_postcard * H = 150) :=
by 
  sorry -- the proof is omitted

end area_of_one_postcard_is_150_cm2_l176_176513


namespace tangent_parallel_x_axis_coordinates_l176_176995

theorem tangent_parallel_x_axis_coordinates :
  ∃ (x y : ℝ), (y = x^2 - 3 * x) ∧ (2 * x - 3 = 0) ∧ (x = 3 / 2) ∧ (y = -9 / 4) :=
by
  use (3 / 2)
  use (-9 / 4)
  sorry

end tangent_parallel_x_axis_coordinates_l176_176995


namespace max_marks_l176_176269

theorem max_marks (M: ℝ) (h1: 0.95 * M = 285):
  M = 300 :=
by
  sorry

end max_marks_l176_176269


namespace infinite_tame_pairs_l176_176915

def s (n : ℕ) : ℕ :=
  (n.digits 10).sum_sq

def is_tame (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate s k n = 1)

theorem infinite_tame_pairs :
  ∃ (a : ℕ → ℕ) (b : ℕ → ℕ), (∀ i, a i + 1 = b i) ∧ (∀ i, is_tame (a i)) ∧ (∀ i, is_tame (b i)) :=
by
  sorry

end infinite_tame_pairs_l176_176915


namespace sum_of_three_largest_l176_176038

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176038


namespace find_a_l176_176352

theorem find_a (x a : ℝ) (h₁ : x = 2) (h₂ : (4 - x) / 2 + a = 4) : a = 3 :=
by
  -- Proof steps will go here
  sorry

end find_a_l176_176352


namespace complex_number_z_l176_176557

theorem complex_number_z (z : ℂ) (i : ℂ) (hz : i^2 = -1) (h : (1 - i)^2 / z = 1 + i) : z = -1 - i :=
by
  sorry

end complex_number_z_l176_176557


namespace count_valid_tuples_l176_176747

variable {b_0 b_1 b_2 b_3 : ℕ}

theorem count_valid_tuples : 
  (∃ b_0 b_1 b_2 b_3 : ℕ, 
    0 ≤ b_0 ∧ b_0 ≤ 99 ∧ 
    0 ≤ b_1 ∧ b_1 ≤ 99 ∧ 
    0 ≤ b_2 ∧ b_2 ≤ 99 ∧ 
    0 ≤ b_3 ∧ b_3 ≤ 99 ∧ 
    5040 = b_3 * 10^3 + b_2 * 10^2 + b_1 * 10 + b_0) ∧ 
    ∃ (M : ℕ), 
    M = 504 :=
sorry

end count_valid_tuples_l176_176747


namespace tiling_scenarios_unique_l176_176578

theorem tiling_scenarios_unique (m n : ℕ) 
  (h1 : 60 * m + 150 * n = 360) : m = 1 ∧ n = 2 :=
by {
  -- The proof will be provided here
  sorry
}

end tiling_scenarios_unique_l176_176578


namespace sequence_is_arithmetic_not_geometric_l176_176563

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 6 / Real.log 2
noncomputable def c := Real.log 12 / Real.log 2

theorem sequence_is_arithmetic_not_geometric : 
  (b - a = c - b) ∧ (b / a ≠ c / b) := 
by
  sorry

end sequence_is_arithmetic_not_geometric_l176_176563


namespace total_journey_length_l176_176781

theorem total_journey_length (y : ℚ)
  (h1 : y * 1 / 4 + 30 + y * 1 / 7 = y) : 
  y = 840 / 17 :=
by 
  sorry

end total_journey_length_l176_176781


namespace total_number_of_bees_is_fifteen_l176_176463

noncomputable def totalBees (B : ℝ) : Prop :=
  (1/5) * B + (1/3) * B + (2/5) * B + 1 = B

theorem total_number_of_bees_is_fifteen : ∃ B : ℝ, totalBees B ∧ B = 15 :=
by
  sorry

end total_number_of_bees_is_fifteen_l176_176463


namespace average_speed_of_trip_is_correct_l176_176283

-- Definitions
def total_distance : ℕ := 450
def distance_part1 : ℕ := 300
def speed_part1 : ℕ := 20
def distance_part2 : ℕ := 150
def speed_part2 : ℕ := 15

-- The average speed problem
theorem average_speed_of_trip_is_correct :
  (total_distance : ℤ) / (distance_part1 / speed_part1 + distance_part2 / speed_part2 : ℤ) = 18 := by
  sorry

end average_speed_of_trip_is_correct_l176_176283


namespace bricklayer_hours_l176_176290

theorem bricklayer_hours
  (B E : ℝ)
  (h1 : B + E = 90)
  (h2 : 12 * B + 16 * E = 1350) :
  B = 22.5 :=
by
  sorry

end bricklayer_hours_l176_176290


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176896

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176896


namespace dice_product_divisibility_probability_l176_176280

theorem dice_product_divisibility_probability :
  let p := 1 - ((5 / 18)^6 : ℚ)
  p = (33996599 / 34012224 : ℚ) :=
by
  -- This is the condition where the probability p is computed as the complementary probability.
  sorry

end dice_product_divisibility_probability_l176_176280


namespace polynomial_g_l176_176788

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l176_176788


namespace both_girls_given_at_least_one_girl_l176_176663

open Probability

theorem both_girls_given_at_least_one_girl :
  let events := ["GG", "GB", "BG", "BB"]
  in P {x | x ∈ {"GG"}} = 1 / 3 :=
by
  have h_conditions := {"GG", "GB", "BG"}
  have h_probability_set := P (event h_conditions)
  have h_single_event := P {x | x ∈ {"GG"}}
  sorry

end both_girls_given_at_least_one_girl_l176_176663


namespace range_of_m_plus_n_l176_176079

theorem range_of_m_plus_n (f : ℝ → ℝ) (n m : ℝ)
  (h_f_def : ∀ x, f x = x^2 + n * x + m)
  (h_non_empty : ∃ x, f x = 0 ∧ f (f x) = 0)
  (h_condition : ∀ x, f x = 0 ↔ f (f x) = 0) :
  0 < m + n ∧ m + n < 4 :=
by {
  -- Proof needed here; currently skipped
  sorry
}

end range_of_m_plus_n_l176_176079


namespace six_digit_palindromes_count_l176_176006

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l176_176006


namespace sum_of_integers_satisfying_l176_176801

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l176_176801


namespace hyperbola_no_common_point_l176_176470

theorem hyperbola_no_common_point (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (y_line : ∀ x : ℝ, y = 2 * x) : 
  ∃ e : ℝ, e = (Real.sqrt (a^2 + b^2)) / a ∧ 1 < e ∧ e ≤ Real.sqrt 5 :=
by
  sorry

end hyperbola_no_common_point_l176_176470


namespace find_interest_rate_l176_176538

theorem find_interest_rate (initial_investment : ℚ) (duration_months : ℚ) 
  (first_rate : ℚ) (final_value : ℚ) (s : ℚ) :
  initial_investment = 15000 →
  duration_months = 9 →
  first_rate = 0.09 →
  final_value = 17218.50 →
  (∃ s : ℚ, 16012.50 * (1 + (s * 0.75) / 100) = final_value) →
  s = 10 := 
by
  sorry

end find_interest_rate_l176_176538


namespace distance_AB_l176_176174

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l176_176174


namespace find_k_l176_176738

-- Definitions for the given conditions
def slope_of_first_line : ℝ := 2
def alpha : ℝ := slope_of_first_line
def slope_of_second_line : ℝ := 2 * alpha

-- The proof goal
theorem find_k (k : ℝ) : slope_of_second_line = k ↔ k = 4 := by
  sorry

end find_k_l176_176738


namespace six_digit_palindromes_count_l176_176004

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l176_176004


namespace sum_of_coordinates_of_A_l176_176951

theorem sum_of_coordinates_of_A
  (A B C : ℝ × ℝ)
  (AC AB BC : ℝ)
  (h1 : AC / AB = 1 / 3)
  (h2 : BC / AB = 2 / 3)
  (hB : B = (2, 5))
  (hC : C = (5, 8)) :
  (A.1 + A.2) = 16 :=
sorry

end sum_of_coordinates_of_A_l176_176951


namespace calculate_tough_week_sales_l176_176760

-- Define the conditions
variables (G T : ℝ)
def condition1 := T = G / 2
def condition2 := 5 * G + 3 * T = 10400

-- By substituting and proving
theorem calculate_tough_week_sales (G T : ℝ) (h1 : condition1 G T) (h2 : condition2 G T) : T = 800 := 
by {
  sorry 
}

end calculate_tough_week_sales_l176_176760


namespace regular_tiles_area_l176_176415

theorem regular_tiles_area (L W : ℝ) (T : ℝ) (h₁ : 1/3 * T * (3 * L * W) + 2/3 * T * (L * W) = 385) : 
  (2/3 * T * (L * W) = 154) :=
by
  sorry

end regular_tiles_area_l176_176415


namespace ship_B_has_highest_rt_no_cars_l176_176766

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l176_176766


namespace domain_eq_l176_176390

def domain_of_function :
    Set ℝ := {x | (x - 1 ≥ 0) ∧ (x + 1 > 0)}

theorem domain_eq :
    domain_of_function = {x | x ≥ 1} :=
by
  sorry

end domain_eq_l176_176390


namespace ship_with_highest_no_car_round_trip_percentage_l176_176762

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l176_176762


namespace arithmetic_mean_reciprocals_primes_l176_176865

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176865


namespace solution_set_abs_inequality_l176_176800

theorem solution_set_abs_inequality (x : ℝ) :
  |2 * x + 1| < 3 ↔ -2 < x ∧ x < 1 :=
by
  sorry

end solution_set_abs_inequality_l176_176800


namespace distance_AB_l176_176173

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l176_176173


namespace colleen_paid_more_l176_176949

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l176_176949


namespace distance_AB_l176_176179

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176179


namespace jackson_entertainment_cost_l176_176107

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l176_176107


namespace pow_product_l176_176315

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l176_176315


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176774

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176774


namespace determine_value_of_m_l176_176610

noncomputable def conics_same_foci (m : ℝ) : Prop :=
  let c1 := Real.sqrt (4 - m^2)
  let c2 := Real.sqrt (m + 2)
  (∀ (x y : ℝ),
    (x^2 / 4 + y^2 / m^2 = 1) → (x^2 / m - y^2 / 2 = 1) → c1 = c2) → 
  m = 1

theorem determine_value_of_m : ∃ (m : ℝ), conics_same_foci m :=
sorry

end determine_value_of_m_l176_176610


namespace inequality_proof_l176_176338

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem inequality_proof (a b : ℝ) (ha : 1 < a) (hb : 0 < b) : 
  f a (a + b) > f a 1 → g (a / b) < g 0 → 1 / (a + b) < Real.log (a + b) / b ∧ Real.log (a + b) / b < a / b := 
by
  sorry

end inequality_proof_l176_176338


namespace find_integer_cosine_l176_176697

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l176_176697


namespace ab_distance_l176_176161

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l176_176161


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176775

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176775


namespace total_cantaloupes_l176_176067

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l176_176067


namespace find_original_price_l176_176522

-- Define the given conditions
def decreased_price : ℝ := 836
def decrease_percentage : ℝ := 0.24
def remaining_percentage : ℝ := 1 - decrease_percentage -- 76% in decimal

-- Define the original price as a variable
variable (x : ℝ)

-- State the theorem
theorem find_original_price (h : remaining_percentage * x = decreased_price) : x = 1100 :=
by
  sorry

end find_original_price_l176_176522


namespace find_x_l176_176289

theorem find_x :
  ∃ x : ℝ, 8 * 5.4 - (x * 10) / 1.2 = 31.000000000000004 ∧ x = 1.464 :=
by
  sorry

end find_x_l176_176289


namespace find_first_term_arithmetic_progression_l176_176545

theorem find_first_term_arithmetic_progression
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 12)
  (h2 : a1 * a2 * a3 = 48)
  (h3 : a2 = a1 + d)
  (h4 : a3 = a1 + 2 * d)
  (h5 : a1 < a2 ∧ a2 < a3) :
  a1 = 2 :=
by
  sorry

end find_first_term_arithmetic_progression_l176_176545


namespace find_a_l176_176959

def A : Set ℝ := {x | x^2 - 2 * x - 3 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}

theorem find_a (a : ℝ) (h : A ∩ B a = B a) : a = 0 ∨ a = -1 ∨ a = 1/3 := by
  sorry

end find_a_l176_176959


namespace simplify_frac_l176_176383

theorem simplify_frac (b : ℤ) (hb : b = 2) : (15 * b^4) / (45 * b^3) = 2 / 3 :=
by {
  sorry
}

end simplify_frac_l176_176383


namespace express_2011_with_digit_1_l176_176743

theorem express_2011_with_digit_1 :
  ∃ (a b c d e: ℕ), 2011 = a * b - c * d + e - f + g ∧
  (a = 1111 ∧ b = 1111) ∧ (c = 111 ∧ d = 11111) ∧ (e = 1111) ∧ (f = 111) ∧ (g = 11) ∧
  (a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f ∧ f ≠ g) :=
sorry

end express_2011_with_digit_1_l176_176743


namespace abs_m_minus_n_l176_176349

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l176_176349


namespace fourth_power_sum_l176_176731

theorem fourth_power_sum
  (a b c : ℝ)
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 19.5 := 
sorry

end fourth_power_sum_l176_176731


namespace bell_rings_count_l176_176205

def classes : List String := ["Maths", "English", "History", "Geography", "Chemistry", "Physics", "Literature", "Music"]

def total_classes : Nat := classes.length

def rings_per_class : Nat := 2

def classes_before_music : Nat := total_classes - 1

def rings_before_music : Nat := classes_before_music * rings_per_class

def current_class_rings : Nat := 1

def total_rings_by_now : Nat := rings_before_music + current_class_rings

theorem bell_rings_count :
  total_rings_by_now = 15 := by
  sorry

end bell_rings_count_l176_176205


namespace mean_of_reciprocals_first_four_primes_l176_176859

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176859


namespace coffee_mix_price_per_pound_l176_176588

-- Definitions based on conditions
def total_weight : ℝ := 100
def columbian_price_per_pound : ℝ := 8.75
def brazilian_price_per_pound : ℝ := 3.75
def columbian_weight : ℝ := 52
def brazilian_weight : ℝ := total_weight - columbian_weight

-- Goal to prove
theorem coffee_mix_price_per_pound :
  (columbian_weight * columbian_price_per_pound + brazilian_weight * brazilian_price_per_pound) / total_weight = 6.35 :=
by
  sorry

end coffee_mix_price_per_pound_l176_176588


namespace find_m_l176_176457

theorem find_m (m : ℝ) : (Real.tan (20 * Real.pi / 180) + m * Real.sin (20 * Real.pi / 180) = Real.sqrt 3) → m = 4 :=
by
  sorry

end find_m_l176_176457


namespace tetrahedron_circumsphere_surface_area_eq_five_pi_l176_176554

noncomputable def rectangle_diagonal (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 + b^2)

noncomputable def circumscribed_sphere_radius (a b : ℝ) : ℝ :=
  rectangle_diagonal a b / 2

noncomputable def circumscribed_sphere_surface_area (a b : ℝ) : ℝ :=
  4 * Real.pi * (circumscribed_sphere_radius a b)^2

theorem tetrahedron_circumsphere_surface_area_eq_five_pi :
  circumscribed_sphere_surface_area 2 1 = 5 * Real.pi := by
  sorry

end tetrahedron_circumsphere_surface_area_eq_five_pi_l176_176554


namespace sqrt_product_simplification_l176_176681

theorem sqrt_product_simplification (q : ℝ) : 
  Real.sqrt (15 * q) * Real.sqrt (10 * q^3) * Real.sqrt (14 * q^5) = 10 * q^4 * Real.sqrt (21 * q) := 
by 
  sorry

end sqrt_product_simplification_l176_176681


namespace triangle_area_problem_l176_176085

theorem triangle_area_problem (c d : ℝ) (hc : c > 0) (hd : d > 0) 
  (h_area : (∃ t : ℝ, t > 0 ∧ (2 * c * t + 3 * d * (12 / (2 * c)) = 12) ∧ (∃ s : ℝ, s > 0 ∧ 2 * c * (12 / (3 * d)) + 3 * d * s = 12)) ∧ 
    ((1 / 2) * (12 / (2 * c)) * (12 / (3 * d)) = 12)) : c * d = 1 := 
by 
  sorry

end triangle_area_problem_l176_176085


namespace calculate_T6_l176_176596

noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + 1 / y^m

theorem calculate_T6 (y : ℝ) (h : y + 1 / y = 5) : T y 6 = 12098 := 
by
  sorry

end calculate_T6_l176_176596


namespace cone_diameter_l176_176931

theorem cone_diameter (S : ℝ) (hS : S = 3 * Real.pi) (unfold_semicircle : ∃ (r l : ℝ), l = 2 * r ∧ S = π * r^2 + (1 / 2) * π * l^2) : 
∃ d : ℝ, d = Real.sqrt 6 := 
by
  sorry

end cone_diameter_l176_176931


namespace g_six_l176_176500

theorem g_six (g : ℝ → ℝ) (H1 : ∀ x y : ℝ, g (x + y) = g x * g y) (H2 : g 2 = 4) : g 6 = 64 :=
by
  sorry

end g_six_l176_176500


namespace convert_to_rectangular_form_l176_176686

noncomputable def rectangular_form (z : ℂ) : ℂ :=
  let e := Complex.exp (13 * Real.pi * Complex.I / 6)
  3 * e

theorem convert_to_rectangular_form :
  rectangular_form (3 * Complex.exp (13 * Real.pi * Complex.I / 6)) = (3 * (Complex.cos (Real.pi / 6)) + 3 * Complex.I * (Complex.sin (Real.pi / 6))) :=
by
  sorry

end convert_to_rectangular_form_l176_176686


namespace find_n_l176_176703

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l176_176703


namespace melissa_trip_total_time_l176_176376

theorem melissa_trip_total_time :
  ∀ (freeway_dist rural_dist : ℕ) (freeway_speed_factor : ℕ) 
  (rural_time : ℕ),
  freeway_dist = 80 →
  rural_dist = 20 →
  freeway_speed_factor = 4 →
  rural_time = 40 →
  (rural_dist * freeway_speed_factor / rural_time + freeway_dist / (rural_dist * freeway_speed_factor / rural_time)) = 80 :=
by
  intros freeway_dist rural_dist freeway_speed_factor rural_time hd1 hd2 hd3 hd4
  sorry

end melissa_trip_total_time_l176_176376


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176887

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176887


namespace Marla_colors_green_squares_l176_176209

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l176_176209


namespace certain_number_existence_l176_176826

theorem certain_number_existence : ∃ x : ℝ, (102 * 102) + (x * x) = 19808 ∧ x = 97 := by
  sorry

end certain_number_existence_l176_176826


namespace sum_of_integers_satisfying_l176_176803

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l176_176803


namespace lola_dora_allowance_l176_176597

variable (total_cost deck_cost sticker_cost sticker_count packs_each : ℕ)
variable (allowance : ℕ)

theorem lola_dora_allowance 
  (h1 : deck_cost = 10)
  (h2 : sticker_cost = 2)
  (h3 : packs_each = 2)
  (h4 : sticker_count = 2 * packs_each)
  (h5 : total_cost = deck_cost + sticker_count * sticker_cost)
  (h6 : total_cost = 18) :
  allowance = 9 :=
sorry

end lola_dora_allowance_l176_176597


namespace factorize_problem_1_factorize_problem_2_l176_176692

theorem factorize_problem_1 (x : ℝ) : 4 * x^2 - 16 = 4 * (x + 2) * (x - 2) := 
by sorry

theorem factorize_problem_2 (x y : ℝ) : 2 * x^3 - 12 * x^2 * y + 18 * x * y^2 = 2 * x * (x - 3 * y)^2 :=
by sorry

end factorize_problem_1_factorize_problem_2_l176_176692


namespace existence_of_positive_numbers_l176_176382

open Real

theorem existence_of_positive_numbers {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^2 + b^2 + c^2 > 2 ∧ a^3 + b^3 + c^3 < 2 ∧ a^4 + b^4 + c^4 > 2 :=
sorry

end existence_of_positive_numbers_l176_176382


namespace solve_equation_l176_176285

noncomputable def f (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs x - 8) - 4) - 2) - 1)

noncomputable def g (x : ℝ) : ℝ :=
  abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs (abs x - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1) - 1)

theorem solve_equation : ∀ (x : ℝ), f x = g x :=
by
  sorry -- The proof will be inserted here

end solve_equation_l176_176285


namespace prob_first_three_heads_all_heads_l176_176642

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l176_176642


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176901

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176901


namespace Chris_had_before_birthday_l176_176683

-- Define the given amounts
def grandmother_money : ℕ := 25
def aunt_uncle_money : ℕ := 20
def parents_money : ℕ := 75
def total_money_now : ℕ := 279

-- Define the total birthday money received
def birthday_money : ℕ := grandmother_money + aunt_uncle_money + parents_money

-- Define the amount of money Chris had before his birthday
def money_before_birthday (total_now birthday_money : ℕ) : ℕ := total_now - birthday_money

-- Proposition to prove
theorem Chris_had_before_birthday : money_before_birthday total_money_now birthday_money = 159 := by
  sorry

end Chris_had_before_birthday_l176_176683


namespace simplify_abs_expression_l176_176558

theorem simplify_abs_expression (a b c : ℝ) (h1 : a + c > b) (h2 : b + c > a) (h3 : a + b > c) :
  |a - b + c| - |a - b - c| = 2 * a - 2 * b :=
by
  sorry

end simplify_abs_expression_l176_176558


namespace difference_of_squares_l176_176088

variable (x y : ℚ)

theorem difference_of_squares (h1 : x + y = 3 / 8) (h2 : x - y = 1 / 8) : x^2 - y^2 = 3 / 64 := 
by
  sorry

end difference_of_squares_l176_176088


namespace quadratic_even_coeff_l176_176378

theorem quadratic_even_coeff (a b c : ℤ) (h : a ≠ 0) (hq : ∃ x : ℚ, a * x^2 + b * x + c = 0) : ¬ (∀ x : ℤ, (x ≠ 0 → (x % 2 = 1))) := 
sorry

end quadratic_even_coeff_l176_176378


namespace rabbit_speed_l176_176549

theorem rabbit_speed (dog_speed : ℝ) (head_start : ℝ) (catch_time_minutes : ℝ) 
  (H1 : dog_speed = 24) (H2 : head_start = 0.6) (H3 : catch_time_minutes = 4) :
  let catch_time_hours := catch_time_minutes / 60
  let distance_dog_runs := dog_speed * catch_time_hours
  let distance_rabbit_runs := distance_dog_runs - head_start
  let rabbit_speed := distance_rabbit_runs / catch_time_hours
  rabbit_speed = 15 :=
  sorry

end rabbit_speed_l176_176549


namespace books_purchased_with_grant_l176_176231

-- Define the conditions
def total_books_now : ℕ := 8582
def books_before_grant : ℕ := 5935

-- State the theorem that we need to prove
theorem books_purchased_with_grant : (total_books_now - books_before_grant) = 2647 := by
  sorry

end books_purchased_with_grant_l176_176231


namespace value_of_a_b_l176_176560

theorem value_of_a_b (a b : ℕ) (ha : 2 * 100 + a * 10 + 3 + 326 = 5 * 100 + b * 10 + 9) (hb : (5 + b + 9) % 9 = 0): 
  a + b = 6 := 
sorry

end value_of_a_b_l176_176560


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176903

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176903


namespace distance_MD_geq_half_AB_l176_176221

open_locale real

noncomputable section

variables {A B C D M F : Type}
variables [metric_space A]
variables (ABC : simplex ℝ A)

def midpoint (x y : Point ℝ) : Point ℝ := (x + y) / 2

def triangle_midpoints (ABC : simplex ℝ A) : Point ℝ × Point ℝ × Point ℝ :=
  let (a, b, c) := ABC.vertices in
  (midpoint b c, midpoint a b, midpoint_of_arc a b c)

theorem distance_MD_geq_half_AB (ABC : simplex ℝ A) :
  let (D, F, M) := triangle_midpoints ABC in
  dist M D ≥ dist (ABC.vertices.1.1.1) (ABC.vertices.1.1.2) / 2 := 
sorry

end distance_MD_geq_half_AB_l176_176221


namespace determine_g_l176_176790

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l176_176790


namespace ratio_arithmetic_seq_a2019_a2017_eq_l176_176845

def ratio_arithmetic_seq (a : ℕ → ℝ) : Prop := 
  ∀ n : ℕ, n ≥ 1 → a (n+2) / a (n+1) - a (n+1) / a n = 2

theorem ratio_arithmetic_seq_a2019_a2017_eq (a : ℕ → ℝ) 
  (h : ratio_arithmetic_seq a) 
  (ha1 : a 1 = 1) 
  (ha2 : a 2 = 1) 
  (ha3 : a 3 = 3) : 
  a 2019 / a 2017 = 4 * 2017^2 - 1 :=
sorry

end ratio_arithmetic_seq_a2019_a2017_eq_l176_176845


namespace greatest_possible_x_l176_176282

-- Define the conditions and the target proof in Lean 4
theorem greatest_possible_x 
  (x : ℤ)  -- x is an integer
  (h : 2.134 * (10:ℝ)^x < 21000) : 
  x ≤ 3 :=
sorry

end greatest_possible_x_l176_176282


namespace number_division_reduction_l176_176525

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 24) : x = 36 := sorry

end number_division_reduction_l176_176525


namespace symmetry_sum_zero_l176_176839

theorem symmetry_sum_zero (v : ℝ → ℝ) 
  (h_sym : ∀ x : ℝ, v (-x) = -v x) : 
  v (-2.00) + v (-1.00) + v (1.00) + v (2.00) = 0 := 
by 
  sorry

end symmetry_sum_zero_l176_176839


namespace parabola_problem_l176_176156

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176156


namespace normal_distribution_probability_l176_176071

noncomputable def normal_distribution_3_1_4 := measure_theory.measure_space.ProbabilityMeasure (measure_theory.measure.gaussian 3 (real.sqrt (1 / 4)))
def X : random_variable normal_distribution_3_1_4 real := sorry

theorem normal_distribution_probability :
  (∀ x, P(X > 7 / 2) = 0.1587) → P(5 / 2 ≤ X ∧ X ≤ 7 / 2) = 0.6826 :=
begin
  sorry
end

end normal_distribution_probability_l176_176071


namespace turnover_threshold_l176_176709

-- Definitions based on the problem conditions
def valid_domain (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2
def daily_turnover (x : ℝ) : ℝ := 20 * (10 - x) * (50 + 8 * x)

-- Lean 4 statement equivalent to mathematical proof problem
theorem turnover_threshold (x : ℝ) (hx : valid_domain x) (h_turnover : daily_turnover x ≥ 10260) :
  x ≥ 1 / 2 ∧ x ≤ 2 :=
sorry

end turnover_threshold_l176_176709


namespace parabola_distance_l176_176131

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l176_176131


namespace union_of_A_and_B_l176_176719

def A : Set ℝ := {x | 2 ≤ x ∧ x ≤ 8}
def B : Set ℝ := {x | 1 < x ∧ x < 6}
def union_AB := {x : ℝ | 1 < x ∧ x ≤ 8}

theorem union_of_A_and_B : A ∪ B = union_AB :=
sorry

end union_of_A_and_B_l176_176719


namespace correct_product_of_0_035_and_3_84_l176_176840

theorem correct_product_of_0_035_and_3_84 : 
  (0.035 * 3.84 = 0.1344) := sorry

end correct_product_of_0_035_and_3_84_l176_176840


namespace sun_volume_exceeds_moon_volume_by_387_cubed_l176_176389

/-- Given Sun's distance to Earth is 387 times greater than Moon's distance to Earth. 
Given diameters:
- Sun's diameter: D_s
- Moon's diameter: D_m
Formula for volume of a sphere: V = (4/3) * pi * R^3
Derive that the Sun's volume exceeds the Moon's volume by 387^3 times. -/
theorem sun_volume_exceeds_moon_volume_by_387_cubed
  (D_s D_m : ℝ)
  (h : D_s = 387 * D_m) :
  (4/3) * Real.pi * (D_s / 2)^3 = 387^3 * (4/3) * Real.pi * (D_m / 2)^3 := by
  sorry

end sun_volume_exceeds_moon_volume_by_387_cubed_l176_176389


namespace parabola_distance_l176_176171

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l176_176171


namespace complement_of_A_in_B_l176_176073

def set_A : Set ℤ := {x | 2 * x = x^2}
def set_B : Set ℤ := {x | -x^2 + x + 2 ≥ 0}

theorem complement_of_A_in_B :
  (set_B \ set_A) = {-1, 1} :=
by
  sorry

end complement_of_A_in_B_l176_176073


namespace three_heads_in_a_row_l176_176650

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l176_176650


namespace find_n_cos_eq_l176_176701

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l176_176701


namespace snow_prob_correct_l176_176969

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l176_176969


namespace arithmetic_mean_reciprocals_primes_l176_176866

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176866


namespace necessary_but_not_sufficient_l176_176333

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * Real.sin (ω * x - (Real.pi / 3))

theorem necessary_but_not_sufficient (ω : ℝ) :
  (∀ x : ℝ, f ω (x + Real.pi) = f ω x) ↔ (ω = 2) ∨ (∃ ω ≠ 2, ∀ x : ℝ, f ω (x + Real.pi) = f ω x) :=
by
  sorry

end necessary_but_not_sufficient_l176_176333


namespace inradius_of_triangle_area_three_times_perimeter_l176_176741

theorem inradius_of_triangle_area_three_times_perimeter (A p s r : ℝ) (h1 : A = 3 * p) (h2 : p = 2 * s) (h3 : A = r * s) (h4 : s ≠ 0) :
  r = 6 :=
sorry

end inradius_of_triangle_area_three_times_perimeter_l176_176741


namespace mr_brown_final_price_is_correct_l176_176212

noncomputable def mr_brown_final_purchase_price :
  Float :=
  let initial_price : Float := 100000
  let mr_brown_price  := initial_price * 1.12
  let improvement := mr_brown_price * 0.05
  let mr_brown_total_investment := mr_brown_price + improvement
  let mr_green_purchase_price := mr_brown_total_investment * 1.04
  let market_decline := mr_green_purchase_price * 0.03
  let value_after_decline := mr_green_purchase_price - market_decline
  let loss := value_after_decline * 0.10
  let ms_white_purchase_price := value_after_decline - loss
  let market_increase := ms_white_purchase_price * 0.08
  let value_after_increase := ms_white_purchase_price + market_increase
  let profit := value_after_increase * 0.05
  let final_price := value_after_increase + profit
  final_price

theorem mr_brown_final_price_is_correct :
  mr_brown_final_purchase_price = 121078.76 := by
  sorry

end mr_brown_final_price_is_correct_l176_176212


namespace find_a3_a4_a5_l176_176936

open Real

variables {a : ℕ → ℝ} (q : ℝ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

noncomputable def a_1 : ℝ := 3

def sum_of_first_three (a : ℕ → ℝ) : Prop :=
  a 0 + a 1 + a 2 = 21

def all_terms_positive (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 0 < a n

theorem find_a3_a4_a5 (h1 : is_geometric_sequence a) (h2 : a 0 = a_1) (h3 : sum_of_first_three a) (h4 : all_terms_positive a) :
  a 2 + a 3 + a 4 = 84 :=
sorry

end find_a3_a4_a5_l176_176936


namespace sum_of_largest_three_l176_176044

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176044


namespace sum_inequality_l176_176476

theorem sum_inequality 
  {a b c : ℝ}
  (h : a + b + c = 3) : 
  (1 / (a^2 - a + 2) + 1 / (b^2 - b + 2) + 1 / (c^2 - c + 2)) ≤ 3 / 2 := 
sorry

end sum_inequality_l176_176476


namespace reduced_rate_fraction_l176_176653

-- Definitions
def hours_in_a_week := 7 * 24
def hours_with_reduced_rates_on_weekdays := (12 * 5)
def hours_with_reduced_rates_on_weekends := (24 * 2)

-- Question in form of theorem
theorem reduced_rate_fraction :
  (hours_with_reduced_rates_on_weekdays + hours_with_reduced_rates_on_weekends) / hours_in_a_week = 9 / 14 := 
by
  sorry

end reduced_rate_fraction_l176_176653


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176858

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176858


namespace all_possible_triples_l176_176441

theorem all_possible_triples (x y : ℕ) (z : ℤ) (hz : z % 2 = 1)
                            (h : x.factorial + y.factorial = 8 * z + 2017) :
                            (x = 1 ∧ y = 4 ∧ z = -249) ∨
                            (x = 4 ∧ y = 1 ∧ z = -249) ∨
                            (x = 1 ∧ y = 5 ∧ z = -237) ∨
                            (x = 5 ∧ y = 1 ∧ z = -237) := 
  sorry

end all_possible_triples_l176_176441


namespace total_cantaloupes_l176_176065

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l176_176065


namespace cube_sqrt_three_eq_three_sqrt_three_l176_176394

theorem cube_sqrt_three_eq_three_sqrt_three : (Real.sqrt 3) ^ 3 = 3 * Real.sqrt 3 := 
by 
  sorry

end cube_sqrt_three_eq_three_sqrt_three_l176_176394


namespace sum_of_integers_square_greater_272_l176_176808

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l176_176808


namespace seven_pow_k_eq_two_l176_176459

theorem seven_pow_k_eq_two {k : ℕ} (h : 7 ^ (4 * k + 2) = 784) : 7 ^ k = 2 := 
by 
  sorry

end seven_pow_k_eq_two_l176_176459


namespace cos_value_l176_176084

theorem cos_value (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 3) : Real.cos (2 * π / 3 + 2 * α) = -7 / 9 :=
by
  sorry

end cos_value_l176_176084


namespace part1_part2_l176_176339

noncomputable def f (a x : ℝ) : ℝ := a * x + x * Real.log x

theorem part1 (a : ℝ) :
  (∀ x, x ≥ Real.exp 1 → (a + 1 + Real.log x) ≥ 0) →
  a ≥ -2 :=
by
  sorry

theorem part2 (k : ℤ) :
  (∀ x, 1 < x → (k : ℝ) * (x - 1) < f 1 x) →
  k ≤ 3 :=
by
  sorry

end part1_part2_l176_176339


namespace fivefold_composition_l176_176575

def f (x : ℚ) : ℚ := -2 / x

theorem fivefold_composition :
  f (f (f (f (f (3))))) = -2 / 3 := 
by
  -- Proof goes here
  sorry

end fivefold_composition_l176_176575


namespace rectangle_problem_l176_176937

noncomputable def calculate_width (L P : ℕ) : ℕ :=
  (P - 2 * L) / 2

theorem rectangle_problem :
  ∀ (L P : ℕ), L = 12 → P = 36 → (calculate_width L P = 6) ∧ ((calculate_width L P) / L = 1 / 2) :=
by
  intros L P hL hP
  have hw : calculate_width L P = 6 := by
    sorry
  have hr : ((calculate_width L P) / L) = 1 / 2 := by
    sorry
  exact ⟨hw, hr⟩

end rectangle_problem_l176_176937


namespace m_gt_p_l176_176366

theorem m_gt_p (p m n : ℕ) (prime_p : Nat.Prime p) (pos_m : 0 < m) (pos_n : 0 < n) (h : p^2 + m^2 = n^2) : m > p :=
sorry

end m_gt_p_l176_176366


namespace algebraic_identity_l176_176220

theorem algebraic_identity 
  (p q r a b c : ℝ)
  (h₁ : p + q + r = 1)
  (h₂ : 1 / p + 1 / q + 1 / r = 0) :
  a^2 + b^2 + c^2 = (p * a + q * b + r * c)^2 + (q * a + r * b + p * c)^2 + (r * a + p * b + q * c)^2 := by
  sorry

end algebraic_identity_l176_176220


namespace q_at_2_l176_176080

noncomputable def q (x : ℝ) : ℝ :=
  Real.sign (3 * x - 2) * |3 * x - 2|^(1/4) +
  2 * Real.sign (3 * x - 2) * |3 * x - 2|^(1/6) +
  |3 * x - 2|^(1/8)

theorem q_at_2 : q 2 = 4 := by
  -- Proof attempt needed
  sorry

end q_at_2_l176_176080


namespace prob_first_three_heads_all_heads_l176_176640

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l176_176640


namespace remaining_amount_is_12_l176_176836

-- Define initial amount and amount spent
def initial_amount : ℕ := 90
def amount_spent : ℕ := 78

-- Define the remaining amount after spending
def remaining_amount : ℕ := initial_amount - amount_spent

-- Theorem asserting the remaining amount is 12
theorem remaining_amount_is_12 : remaining_amount = 12 :=
by
  -- Proof omitted
  sorry

end remaining_amount_is_12_l176_176836


namespace find_radius_of_cylinder_l176_176986

theorem find_radius_of_cylinder :
  (∃ R : ℝ, 
    R = 5 / (Real.sqrt 3) ∨ 
    R = 4 * Real.sqrt (2 / 3) ∨ 
    R = (20 / 3) * Real.sqrt (2 / 11)) :=
by
  -- Definitions of the constants and geometric conditions:
  let a := 4   -- side length of square ABCD
  let h := 6   -- height of the parallelepiped
  let R1 := 5 / (Real.sqrt 3)
  let R2 := 4 * Real.sqrt (2 / 3)
  let R3 := (20 / 3) * Real.sqrt (2 / 11)
  
  -- Statement that at least one of these R is correct
  use R1
  left
  rfl
  sorry

end find_radius_of_cylinder_l176_176986


namespace equalize_costs_l176_176113

theorem equalize_costs (X Y Z : ℝ) (hXY : X < Y) (hYZ : Y < Z) : (Y + Z - 2 * X) / 3 = (X + Y + Z) / 3 - X := by
  sorry

end equalize_costs_l176_176113


namespace square_plot_area_l176_176665

theorem square_plot_area
  (cost_per_foot : ℕ)
  (total_cost : ℕ)
  (s : ℕ)
  (area : ℕ)
  (h1 : cost_per_foot = 55)
  (h3 : total_cost = 3740)
  (h4 : total_cost = 4 * s * cost_per_foot)
  (h5 : area = s * s) :
  area = 289 := sorry

end square_plot_area_l176_176665


namespace max_competitors_l176_176669

theorem max_competitors (P1 P2 P3 : ℕ → ℕ → ℕ)
(hP1 : ∀ i, 0 ≤ P1 i ∧ P1 i ≤ 7)
(hP2 : ∀ i, 0 ≤ P2 i ∧ P2 i ≤ 7)
(hP3 : ∀ i, 0 ≤ P3 i ∧ P3 i ≤ 7)
(hDistinct : ∀ i j, i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :
  ∃ n, n ≤ 64 ∧ ∀ k, k < n → (∀ i j, i < k → j < k → i ≠ j → (P1 i ≠ P1 j ∨ P2 i ≠ P2 j ∨ P3 i ≠ P3 j)) :=
sorry

end max_competitors_l176_176669


namespace boys_count_l176_176409

/-
Conditions:
1. The total number of members in the chess team is 26.
2. 18 members were present at the last session.
3. One-third of the girls attended the session.
4. All of the boys attended the session.
-/
def TotalMembers : Nat := 26
def LastSessionAttendance : Nat := 18
def GirlsAttendance (G : Nat) : Nat := G / 3
def BoysAttendance (B : Nat) : Nat := B

/-
Main theorem statement:
Prove that the number of boys in the chess team is 14.
-/
theorem boys_count (B G : Nat) (h1 : B + G = TotalMembers) (h2 : GirlsAttendance G + BoysAttendance B = LastSessionAttendance) : B = 14 :=
by
  sorry

end boys_count_l176_176409


namespace bumper_cars_number_of_tickets_l176_176745

theorem bumper_cars_number_of_tickets (Ferris_Wheel Roller_Coaster Jeanne_Has Jeanne_Buys : ℕ)
  (h1 : Ferris_Wheel = 5)
  (h2 : Roller_Coaster = 4)
  (h3 : Jeanne_Has = 5)
  (h4 : Jeanne_Buys = 8) :
  Ferris_Wheel + Roller_Coaster + (13 - (Ferris_Wheel + Roller_Coaster)) = 13 - (Ferris_Wheel + Roller_Coaster) :=
by
  sorry

end bumper_cars_number_of_tickets_l176_176745


namespace total_outfits_l176_176230

-- Define the number of shirts, pants, ties (including no-tie option), and shoes as given in the conditions.
def num_shirts : ℕ := 5
def num_pants : ℕ := 4
def num_ties : ℕ := 6 -- 5 ties + 1 no-tie option
def num_shoes : ℕ := 2

-- Proof statement: The total number of different outfits is 240.
theorem total_outfits : num_shirts * num_pants * num_ties * num_shoes = 240 :=
by
  sorry

end total_outfits_l176_176230


namespace solve_quadratic_eq_l176_176256

theorem solve_quadratic_eq (x : ℝ) : x^2 - 4 = 0 ↔ x = 2 ∨ x = -2 :=
by
  sorry

end solve_quadratic_eq_l176_176256


namespace tan_sum_formula_eq_l176_176341

theorem tan_sum_formula_eq {θ : ℝ} (h1 : ∃θ, θ ∈ Set.Ico 0 (2 * Real.pi) 
  ∧ ∃P, P = (Real.sin (3 * Real.pi / 4), Real.cos (3 * Real.pi / 4)) 
  ∧ θ = (3 * Real.pi / 4)) : 
  Real.tan (θ + Real.pi / 3) = 2 - Real.sqrt 3 := 
sorry

end tan_sum_formula_eq_l176_176341


namespace product_of_sequence_is_256_l176_176544

-- Definitions for conditions
def seq : List ℚ := [1 / 4, 16 / 1, 1 / 64, 256 / 1, 1 / 1024, 4096 / 1, 1 / 16384, 65536 / 1]

-- The main theorem
theorem product_of_sequence_is_256 : (seq.prod = 256) :=
by
  sorry

end product_of_sequence_is_256_l176_176544


namespace probability_first_three_heads_l176_176631

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l176_176631


namespace sum_of_three_largest_consecutive_numbers_l176_176026

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176026


namespace cubicsum_eq_neg36_l176_176203

noncomputable def roots (p q r : ℝ) := 
  ∃ l : ℝ, (p^3 - 12) / p = l ∧ (q^3 - 12) / q = l ∧ (r^3 - 12) / r = l

theorem cubicsum_eq_neg36 {p q r : ℝ} (h : p ≠ q ∧ q ≠ r ∧ p ≠ r) 
  (hl : roots p q r) :
  p^3 + q^3 + r^3 = -36 :=
sorry

end cubicsum_eq_neg36_l176_176203


namespace probability_first_three_heads_l176_176633

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l176_176633


namespace complement_intersection_l176_176477

-- Definitions to set the universal set and other sets
def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 3, 4}
def N : Set ℕ := {2, 4, 5}

-- Complement of M with respect to U
def CU_M : Set ℕ := U \ M

-- Intersection of (CU_M) and N
def intersection_CU_M_N : Set ℕ := CU_M ∩ N

-- The proof problem statement
theorem complement_intersection :
  intersection_CU_M_N = {2, 5} :=
sorry

end complement_intersection_l176_176477


namespace football_championship_l176_176096

noncomputable def exists_trio_did_not_play_each_other 
  (teams : Finset ℕ) (rounds : ℕ) (matches : Finset (ℕ × ℕ)) : Prop :=
  ∃ (T : Finset ℕ), T.card = 3 ∧
    (∀ (a b : ℕ), a ∈ T → b ∈ T → a ≠ b → (a, b) ∉ matches ∧ (b, a) ∉ matches)

theorem football_championship (teams : Finset ℕ) (h_teams : teams.card = 18)
  (rounds : ℕ) (h_rounds : rounds = 8) 
  (matches_per_round : Finset (Finset (ℕ × ℕ))) 
  (h_matches_per_round : ∀ round ∈ matches_per_round, round.card = 9)
  (unique_pairs : ∀ (r₁ r₂ ∈ matches_per_round) (p : ℕ × ℕ), p ∈ r₁ → p ∈ r₂ → r₁ = r₂) :
  exists_trio_did_not_play_each_other teams rounds (matches_per_round.bUnion id) :=
by 
  sorry

end football_championship_l176_176096


namespace AB_distance_l176_176145

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l176_176145


namespace sum_of_three_largest_l176_176021

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176021


namespace inclination_angle_range_l176_176351

theorem inclination_angle_range (k : ℝ) (h : |k| ≤ 1) :
    ∃ α : ℝ, (k = Real.tan α) ∧ (0 ≤ α ∧ α ≤ Real.pi / 4 ∨ 3 * Real.pi / 4 ≤ α ∧ α < Real.pi) :=
by
  sorry

end inclination_angle_range_l176_176351


namespace chess_tournament_no_804_games_l176_176097

/-- Statement of the problem: 
    Under the given conditions, prove that it is impossible for exactly 804 games to have been played in the chess tournament.
--/
theorem chess_tournament_no_804_games :
  ¬ ∃ n : ℕ, n * (n - 4) = 1608 :=
by
  sorry

end chess_tournament_no_804_games_l176_176097


namespace inequality_proof_inequality_equality_conditions_l176_176655

theorem inequality_proof
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  (x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 ≤ (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2)) :=
sorry

theorem inequality_equality_conditions
  (x1 x2 y1 y2 z1 z2 : ℝ)
  (hx1 : x1 > 0) (hx2 : x2 > 0)
  (hy1 : y1 > 0) (hy2 : y2 > 0)
  (hxy1 : x1 * y1 - z1 ^ 2 > 0) (hxy2 : x2 * y2 - z2 ^ 2 > 0) :
  ((x1 + x2) * (y1 + y2) - (z1 + z2) ^ 2 = (1 / (x1 * y1 - z1 ^ 2) + 1 / (x2 * y2 - z2 ^ 2))
  ↔ (x1 = x2 ∧ y1 = y2 ∧ z1 = z2)) :=
sorry

end inequality_proof_inequality_equality_conditions_l176_176655


namespace sum_of_three_largest_l176_176035

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176035


namespace ratio_of_unit_prices_is_17_over_25_l176_176426

def vol_B (v_B : ℝ) := v_B
def price_B (p_B : ℝ) := p_B

def vol_A (v_B : ℝ) := 1.25 * v_B
def price_A (p_B : ℝ) := 0.85 * p_B

def unit_price_A (p_B v_B : ℝ) := price_A p_B / vol_A v_B
def unit_price_B (p_B v_B : ℝ) := price_B p_B / vol_B v_B

def ratio (p_B v_B : ℝ) := unit_price_A p_B v_B / unit_price_B p_B v_B

theorem ratio_of_unit_prices_is_17_over_25 (p_B v_B : ℝ) (h_vB : v_B ≠ 0) (h_pB : p_B ≠ 0) :
  ratio p_B v_B = 17 / 25 := by
  sorry

end ratio_of_unit_prices_is_17_over_25_l176_176426


namespace compute_difference_of_squares_l176_176319

theorem compute_difference_of_squares :
    75^2 - 25^2 = 5000 :=
by
  sorry

end compute_difference_of_squares_l176_176319


namespace triangle_inequality_l176_176526

variable {a b c : ℝ}
variable {x y z : ℝ}

theorem triangle_inequality (ha : a ≥ b) (hb : b ≥ c) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (hx_yz_sum : x + y + z = π) :
  bc + ca - ab < bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ∧
  bc * Real.cos x + ca * Real.cos y + ab * Real.cos z ≤ (a^2 + b^2 + c^2) / 2 := sorry

end triangle_inequality_l176_176526


namespace solution_to_inequality_l176_176257

theorem solution_to_inequality (x : ℝ) (hx : 0 < x ∧ x < 1) : 1 / x > 1 :=
by
  sorry

end solution_to_inequality_l176_176257


namespace sum_of_three_largest_l176_176053

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176053


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176897

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176897


namespace line_intersects_ellipse_l176_176561

theorem line_intersects_ellipse
  (m : ℝ) :
  ∃ P : ℝ × ℝ, P = (3, 2) ∧ ((m + 2) * P.1 - (m + 4) * P.2 + 2 - m = 0) ∧ 
  (P.1^2 / 25 + P.2^2 / 9 < 1) :=
by 
  sorry

end line_intersects_ellipse_l176_176561


namespace sum_of_fractions_correct_l176_176684

def sum_of_fractions : ℚ := (4 / 3) + (8 / 9) + (18 / 27) + (40 / 81) + (88 / 243) - 5

theorem sum_of_fractions_correct : sum_of_fractions = -305 / 243 := by
  sorry -- proof to be provided

end sum_of_fractions_correct_l176_176684


namespace heather_aprons_l176_176924

variable {totalAprons : Nat} (apronsSewnBeforeToday apronsSewnToday apronsSewnTomorrow apronsSewnSoFar apronsRemaining : Nat)

theorem heather_aprons (h_total : totalAprons = 150)
                       (h_today : apronsSewnToday = 3 * apronsSewnBeforeToday)
                       (h_sewnSoFar : apronsSewnSoFar = apronsSewnBeforeToday + apronsSewnToday)
                       (h_tomorrow : apronsSewnTomorrow = 49)
                       (h_remaining : apronsRemaining = totalAprons - apronsSewnSoFar)
                       (h_halfRemaining : 2 * apronsSewnTomorrow = apronsRemaining) :
  apronsSewnBeforeToday = 13 :=
by
  sorry

end heather_aprons_l176_176924


namespace sin_alpha_cos_squared_beta_range_l176_176331

theorem sin_alpha_cos_squared_beta_range (α β : ℝ) 
  (h : Real.sin α + Real.sin β = 1) : 
  ∃ y, y = Real.sin α - Real.cos β ^ 2 ∧ (-1/4 ≤ y ∧ y ≤ 0) :=
sorry

end sin_alpha_cos_squared_beta_range_l176_176331


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176878

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176878


namespace intersection_distance_l176_176960

noncomputable def distance_between_intersections (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) : Prop :=
  ∃ A B : ℝ × ℝ, 
    l A.1 A.2 ∧ C A.1 A.2 ∧ l B.1 B.2 ∧ C B.1 B.2 ∧ 
    dist A B = Real.sqrt 6

def line_l (x y : ℝ) : Prop :=
  x - y + 1 = 0

def curve_C (x y : ℝ) : Prop :=
  ∃ θ : ℝ, x = Real.sqrt 2 * Real.cos θ ∧ y = Real.sqrt 2 * Real.sin θ

theorem intersection_distance :
  distance_between_intersections line_l curve_C :=
sorry

end intersection_distance_l176_176960


namespace minutes_in_3_5_hours_l176_176345

theorem minutes_in_3_5_hours : 3.5 * 60 = 210 := 
by
  sorry

end minutes_in_3_5_hours_l176_176345


namespace range_of_a_l176_176461

theorem range_of_a :
  (∃ x : ℝ, 0 ≤ x ∧ x ≤ 1 ∧ 2 * x * (3 * x + a) < 1) → a < 1 :=
by
  sorry

end range_of_a_l176_176461


namespace sum_of_three_largest_consecutive_numbers_l176_176025

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176025


namespace sequences_get_arbitrarily_close_l176_176487

noncomputable def a_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^n
noncomputable def b_n (n : ℕ) : ℝ := (1 + (1 / n : ℝ))^(n + 1)

theorem sequences_get_arbitrarily_close (n : ℕ) : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |b_n n - a_n n| < ε :=
sorry

end sequences_get_arbitrarily_close_l176_176487


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176854

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176854


namespace sum_of_largest_three_consecutive_numbers_l176_176055

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176055


namespace difference_of_numbers_l176_176988

theorem difference_of_numbers 
  (L S : ℤ) (hL : L = 1636) (hdiv : L = 6 * S + 10) : 
  L - S = 1365 :=
sorry

end difference_of_numbers_l176_176988


namespace width_decreased_by_28_6_percent_l176_176245

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l176_176245


namespace rectangular_field_length_l176_176402

noncomputable def area_triangle (base height : ℝ) : ℝ :=
  (base * height) / 2

noncomputable def length_rectangle (area width : ℝ) : ℝ :=
  area / width

theorem rectangular_field_length (base height width : ℝ) (h_base : base = 7.2) (h_height : height = 7) (h_width : width = 4) :
  length_rectangle (area_triangle base height) width = 6.3 :=
by
  -- sorry would be replaced by the actual proof.
  sorry

end rectangular_field_length_l176_176402


namespace inequality_inequality_l176_176602

theorem inequality_inequality (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
  (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 :=
by sorry

end inequality_inequality_l176_176602


namespace maximize_revenue_at_175_l176_176410

def price (x : ℕ) : ℕ :=
  if x ≤ 150 then 200 else 200 - (x - 150)

def revenue (x : ℕ) : ℕ :=
  price x * x

theorem maximize_revenue_at_175 :
  ∀ x : ℕ, revenue 175 ≥ revenue x := 
sorry

end maximize_revenue_at_175_l176_176410


namespace AB_distance_l176_176192

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l176_176192


namespace imaginary_part_of_complex_num_l176_176624

-- Define the complex number and the imaginary part condition
def complex_num : ℂ := ⟨1, 2⟩

-- Define the theorem to prove the imaginary part is 2
theorem imaginary_part_of_complex_num : complex_num.im = 2 :=
by
  -- The proof steps would go here
  sorry

end imaginary_part_of_complex_num_l176_176624


namespace point_in_region_l176_176761

theorem point_in_region (x y : ℝ) (h : x * y ≥ 0) : (real.sqrt (x * y) ≥ x - 2 * y) ↔ 
  ((x ≥ 0 ∧ y ≥ 0 ∧ y ≥ x / 2) ∨ (x ≤ 0 ∧ y ≤ 0 ∧ y ≤ x / 2)) :=
by
  sorry

end point_in_region_l176_176761


namespace some_zen_not_cen_l176_176081

variable {Zen Ben Cen : Type}
variables (P Q R : Zen → Prop)

theorem some_zen_not_cen (h1 : ∀ x, P x → Q x)
                        (h2 : ∃ x, Q x ∧ ¬ (R x)) :
  ∃ x, P x ∧ ¬ (R x) :=
  sorry

end some_zen_not_cen_l176_176081


namespace unicorn_witch_ratio_l176_176103

theorem unicorn_witch_ratio (W D U : ℕ) (h1 : W = 7) (h2 : D = W + 25) (h3 : U + W + D = 60) :
  U / W = 3 := by
  sorry

end unicorn_witch_ratio_l176_176103


namespace parabola_distance_l176_176191

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l176_176191


namespace pond_capacity_l176_176363

theorem pond_capacity :
  let normal_rate := 6 -- gallons per minute
  let restriction_rate := (2/3 : ℝ) * normal_rate -- gallons per minute
  let time := 50 -- minutes
  let capacity := restriction_rate * time -- total capacity in gallons
  capacity = 200 := sorry

end pond_capacity_l176_176363


namespace distance_AB_l176_176122

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l176_176122


namespace rectangle_width_decrease_proof_l176_176239

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l176_176239


namespace probability_three_heads_l176_176638

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l176_176638


namespace find_vector_b_l176_176450

structure Vec2 where
  x : ℝ
  y : ℝ

def is_parallel (a b : Vec2) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ b.x = k * a.x ∧ b.y = k * a.y

def vec_a : Vec2 := { x := 2, y := 3 }
def vec_b : Vec2 := { x := -2, y := -3 }

theorem find_vector_b :
  is_parallel vec_a vec_b := 
sorry

end find_vector_b_l176_176450


namespace colleen_paid_more_l176_176948

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l176_176948


namespace equal_circumradii_of_inscribed_quadrilateral_l176_176072

theorem equal_circumradii_of_inscribed_quadrilateral
  {A B C D M N P Q O : Point}
  (h1 : is_inscribed_quadrilateral A B C D)
  (h2 : is_midpoint A B M)
  (h3 : is_midpoint B C N)
  (h4 : is_midpoint C D P)
  (h5 : is_midpoint D A Q)
  (h6 : is_intersection AC BD O) :
  circumradius (triangle O M N) =
  circumradius (triangle O N P) ∧
  circumradius (triangle O N P) =
  circumradius (triangle O P Q) ∧
  circumradius (triangle O P Q) =
  circumradius (triangle O Q M) :=
sorry

end equal_circumradii_of_inscribed_quadrilateral_l176_176072


namespace prob_multiple_of_3_l176_176362

-- Define the possible start points
def start_points := Fin 15

-- Define the spinner movement instructions
inductive Spinner
| right1 | right2 | left1 | left2

-- Define the transitions for each spinner result
def move (pos : ℤ) : Spinner → ℤ
| Spinner.right1 := pos + 1
| Spinner.right2 := pos + 2
| Spinner.left1 := pos - 1
| Spinner.left2 := pos - 2

-- What we need to prove
theorem prob_multiple_of_3 : (∃ p : ℚ, p = 17 / 80) :=
sorry

end prob_multiple_of_3_l176_176362


namespace probability_of_green_l176_176828

theorem probability_of_green : 
  ∀ (P_red P_orange P_yellow P_green : ℝ), 
    P_red = 0.25 → P_orange = 0.35 → P_yellow = 0.1 → 
    P_red + P_orange + P_yellow + P_green = 1 →
    P_green = 0.3 :=
by
  intros P_red P_orange P_yellow P_green h_red h_orange h_yellow h_total
  sorry

end probability_of_green_l176_176828


namespace parabola_distance_l176_176118

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l176_176118


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176857

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176857


namespace solve_inequality_l176_176604

theorem solve_inequality :
  {x : ℝ | x ∈ { y | (y^2 - 5*y + 6) / (y - 3)^2 > 0 }} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end solve_inequality_l176_176604


namespace area_of_rectangle_l176_176211

theorem area_of_rectangle (a b : ℝ) (area : ℝ) 
(h1 : a = 5.9) 
(h2 : b = 3) 
(h3 : area = a * b) : 
area = 17.7 := 
by 
  -- proof goes here
  sorry

-- Definitions and conditions alignment:
-- a represents one side of the rectangle.
-- b represents the other side of the rectangle.
-- area represents the area of the rectangle.
-- h1: a = 5.9 corresponds to the first condition.
-- h2: b = 3 corresponds to the second condition.
-- h3: area = a * b connects the conditions to the formula to find the area.
-- The goal is to show that area = 17.7, which matches the correct answer.

end area_of_rectangle_l176_176211


namespace cans_to_paint_35_rooms_l176_176976

/-- Paula the painter initially had enough paint for 45 identically sized rooms.
    Unfortunately, she lost five cans of paint, leaving her with only enough paint for 35 rooms.
    Prove that she now uses 18 cans of paint to paint the 35 rooms. -/
theorem cans_to_paint_35_rooms :
  ∀ (cans_per_room : ℕ) (total_cans : ℕ) (lost_cans : ℕ) (rooms_before : ℕ) (rooms_after : ℕ),
  rooms_before = 45 →
  lost_cans = 5 →
  rooms_after = 35 →
  rooms_before - rooms_after = cans_per_room * lost_cans →
  (cans_per_room * rooms_after) / rooms_after = 18 :=
by
  intros
  sorry

end cans_to_paint_35_rooms_l176_176976


namespace line_contains_point_l176_176442

theorem line_contains_point (k : ℝ) : 
  let x := (1 : ℝ) / 3
  let y := -2 
  let line_eq := (3 : ℝ) - 3 * k * x = 4 * y
  line_eq → k = 11 :=
by
  intro h
  sorry

end line_contains_point_l176_176442


namespace coins_to_rubles_l176_176407

theorem coins_to_rubles (a1 a2 a3 a4 a5 a6 a7 k m : ℕ)
  (h1 : a1 + 2 * a2 + 5 * a3 + 10 * a4 + 20 * a5 + 50 * a6 + 100 * a7 = m)
  (h2 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = k) :
  m * 100 = k :=
by sorry

end coins_to_rubles_l176_176407


namespace largest_possible_value_of_p_l176_176834

theorem largest_possible_value_of_p (m n p : ℕ) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * m * n * p = (m + 2) * (n + 2) * (p + 2)) : p ≤ 130 :=
by
  sorry

end largest_possible_value_of_p_l176_176834


namespace max_value_sin_transform_l176_176622

open Real

theorem max_value_sin_transform (φ : ℝ) (hφ : |φ| < π / 2) :
  let f (x : ℝ) := sin (2 * x + φ)
  ∃ x ∈ Icc (0 : ℝ) (π / 2), f x = 1 := 
begin
  let g (x : ℝ) := sin (2 * x + 2 * π / 3 + φ),
  have hg_symm : ∀ x, g x = - g (-x), from sorry, -- Using symmetry about the origin
  have hφ_eq : 2 * π / 3 + φ = π, from sorry, -- Derived from symmetry condition
  let φ := π / 3,
  have f_def : f = (λ x, sin (2 * x + π / 3)), 
  { unfold f, rw [hφ_eq], sorry },
  
  have max_value_interval : ∃ x ∈ Icc (0 : ℝ) (π / 2), sin (2 * x + π / 3) = 1,
  { use π / 6,
    split,
    { split,
      { linarith },
      { linarith }},
    { rw [sin_add],
      have : sin (π / 2) = 1 := sin_pi_div_two,
      rw [this, mul_div_cancel_left π (show 2 ≠ 0, by linarith)], 
      rw [sin_pi, cos_pi_div_two, zero_mul, add_zero],
      exact this }},
  exact max_value_interval
end

end max_value_sin_transform_l176_176622


namespace seashells_broken_l176_176962

theorem seashells_broken (total_seashells : ℕ) (unbroken_seashells : ℕ) (broken_seashells : ℕ) : 
  total_seashells = 6 → unbroken_seashells = 2 → broken_seashells = total_seashells - unbroken_seashells → broken_seashells = 4 :=
by
  intros ht hu hb
  rw [ht, hu] at hb
  exact hb

end seashells_broken_l176_176962


namespace marla_colors_green_squares_l176_176207

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l176_176207


namespace distance_AB_l176_176197

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176197


namespace mean_of_reciprocals_of_first_four_primes_l176_176890

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176890


namespace second_smallest_N_prevent_Bananastasia_win_l176_176469

-- Definition of the set S, as positive integers not divisible by any p^4.
def S : Set ℕ := {n | ∀ p : ℕ, Prime p → ¬ (p ^ 4 ∣ n)}

-- Definition of the game rules and the condition for Anastasia to prevent Bananastasia from winning.
-- N is a value such that for all a in S, it is not possible for Bananastasia to directly win.

theorem second_smallest_N_prevent_Bananastasia_win :
  ∃ N : ℕ, N = 625 ∧ (∀ a ∈ S, N - a ≠ 0 ∧ N - a ≠ 1) :=
by
  sorry

end second_smallest_N_prevent_Bananastasia_win_l176_176469


namespace AB_distance_l176_176143

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l176_176143


namespace parabola_distance_l176_176121

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l176_176121


namespace chess_tournament_games_l176_176579

def stage1_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage2_games (players : ℕ) : ℕ := (players * (players - 1) * 2) / 2
def stage3_games : ℕ := 4

def total_games (stage1 stage2 stage3 : ℕ) : ℕ := stage1 + stage2 + stage3

theorem chess_tournament_games : total_games (stage1_games 20) (stage2_games 10) stage3_games = 474 :=
by
  unfold stage1_games
  unfold stage2_games
  unfold total_games
  simp
  sorry

end chess_tournament_games_l176_176579


namespace totalGamesPlayed_l176_176284

def numPlayers : ℕ := 30

def numGames (n : ℕ) : ℕ := (n * (n - 1)) / 2

theorem totalGamesPlayed :
  numGames numPlayers = 435 :=
by
  sorry

end totalGamesPlayed_l176_176284


namespace sum_of_three_largest_l176_176031

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176031


namespace points_per_enemy_l176_176465

theorem points_per_enemy (total_enemies destroyed_enemies points_earned points_per_enemy : ℕ)
  (h1 : total_enemies = 8)
  (h2 : destroyed_enemies = total_enemies - 6)
  (h3 : points_earned = 10)
  (h4 : points_per_enemy = points_earned / destroyed_enemies) : 
  points_per_enemy = 5 := 
by
  sorry

end points_per_enemy_l176_176465


namespace find_height_of_cuboid_l176_176999

variable (A : ℝ) (V : ℝ) (h : ℝ)

theorem find_height_of_cuboid (h_eq : h = V / A) (A_eq : A = 36) (V_eq : V = 252) : h = 7 :=
by
  sorry

end find_height_of_cuboid_l176_176999


namespace sum_of_three_largest_l176_176016

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176016


namespace six_digit_palindrome_count_l176_176001

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l176_176001


namespace smallest_k_exists_l176_176438

theorem smallest_k_exists :
  ∃ (k : ℕ), k = 1001 ∧ (∃ (a : ℕ), 500000 < a ∧ ∃ (b : ℕ), (1 / (a : ℝ) + 1 / (a + k : ℝ) = 1 / (b : ℝ))) :=
by
  sorry

end smallest_k_exists_l176_176438


namespace inequality_x4_y4_l176_176779

theorem inequality_x4_y4 (x y : ℝ) : x^4 + y^4 + 8 ≥ 8 * x * y := 
by {
  sorry
}

end inequality_x4_y4_l176_176779


namespace sum_of_three_largest_l176_176017

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176017


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176881

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176881


namespace quadratic_has_two_distinct_roots_l176_176222

theorem quadratic_has_two_distinct_roots (a b c α : ℝ) (h : a * (a * α^2 + b * α + c) < 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a*x1^2 + b*x1 + c = 0) ∧ (a*x2^2 + b*x2 + c = 0) ∧ x1 < α ∧ x2 > α :=
sorry

end quadratic_has_two_distinct_roots_l176_176222


namespace moles_CO2_is_one_l176_176707

noncomputable def moles_CO2_formed (moles_HNO3 moles_NaHCO3 : ℕ) : ℕ :=
  if moles_HNO3 = 1 ∧ moles_NaHCO3 = 1 then 1 else 0

theorem moles_CO2_is_one :
  moles_CO2_formed 1 1 = 1 :=
by
  sorry

end moles_CO2_is_one_l176_176707


namespace speed_first_hour_l176_176258

variable (x : ℕ)

-- Definitions based on conditions
def total_distance (x : ℕ) : ℕ := x + 50
def average_speed (x : ℕ) : Prop := (total_distance x) / 2 = 70

-- Theorem statement
theorem speed_first_hour : ∃ x, average_speed x ∧ x = 90 := by
  sorry

end speed_first_hour_l176_176258


namespace total_cost_of_groceries_l176_176233

noncomputable def M (R : ℝ) : ℝ := 24 * R / 10
noncomputable def F : ℝ := 22

theorem total_cost_of_groceries (R : ℝ) (hR : 2 * R = 22) :
  10 * M R = 24 * R ∧ F = 2 * R ∧ F = 22 →
  4 * M R + 3 * R + 5 * F = 248.6 := by
  sorry

end total_cost_of_groceries_l176_176233


namespace number_decomposition_l176_176799

theorem number_decomposition : 10101 = 10000 + 100 + 1 :=
by
  sorry

end number_decomposition_l176_176799


namespace range_of_z_l176_176444

theorem range_of_z (x y : ℝ) (h : x^2 + 2*x*y + 4*y^2 = 6) :
  12 ≤ x^2 + 4*y^2 ∧ x^2 + 4*y^2 ≤ 20 :=
by
  sorry

end range_of_z_l176_176444


namespace weighted_average_salary_l176_176662

theorem weighted_average_salary :
  let num_managers := 9
  let salary_managers := 4500
  let num_associates := 18
  let salary_associates := 3500
  let num_lead_cashiers := 6
  let salary_lead_cashiers := 3000
  let num_sales_representatives := 45
  let salary_sales_representatives := 2500
  let total_salaries := 
    (num_managers * salary_managers) +
    (num_associates * salary_associates) +
    (num_lead_cashiers * salary_lead_cashiers) +
    (num_sales_representatives * salary_sales_representatives)
  let total_employees := 
    num_managers + num_associates + num_lead_cashiers + num_sales_representatives
  let weighted_avg_salary := total_salaries / total_employees
  weighted_avg_salary = 3000 := 
by
  sorry

end weighted_average_salary_l176_176662


namespace prob_three_heads_is_one_eighth_l176_176634

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l176_176634


namespace larry_result_is_correct_l176_176757

theorem larry_result_is_correct (a b c d e : ℤ) 
  (h1: a = 2) (h2: b = 4) (h3: c = 3) (h4: d = 5) (h5: e = -15) :
  a - (b - (c * (d + e))) = (-17 + e) :=
by 
  rw [h1, h2, h3, h4, h5]
  sorry

end larry_result_is_correct_l176_176757


namespace non_defective_probability_l176_176297

theorem non_defective_probability :
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  p_non_def = 0.96 :=
by
  let p_B := 0.03
  let p_C := 0.01
  let p_def := p_B + p_C
  let p_non_def := 1 - p_def
  sorry

end non_defective_probability_l176_176297


namespace parabola_distance_l176_176128

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l176_176128


namespace remaining_regular_toenails_l176_176925

def big_toenail_space := 2
def total_capacity := 100
def big_toenails_count := 20
def regular_toenails_count := 40

theorem remaining_regular_toenails : 
  total_capacity - (big_toenails_count * big_toenail_space + regular_toenails_count) = 20 := by
  sorry

end remaining_regular_toenails_l176_176925


namespace total_outcomes_l176_176288

-- Define the number of students
def num_students : ℕ := 5

-- Define the number of events
def num_events : ℕ := 3

-- Theorem statement: asserting the total number of different outcomes
theorem total_outcomes : num_students ^ num_events = 125 :=
by
  sorry

end total_outcomes_l176_176288


namespace routes_on_3x3_grid_are_20_l176_176532

def number_of_routes_3x3 : ℕ := 
  Nat.choose 6 3

theorem routes_on_3x3_grid_are_20 : number_of_routes_3x3 = 20 :=
sorry

end routes_on_3x3_grid_are_20_l176_176532


namespace sum_of_three_largest_consecutive_numbers_l176_176030

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176030


namespace remainder_of_five_consecutive_odds_mod_12_l176_176518

/-- Let x be an odd integer. Prove that (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 
    when x ≡ 5 (mod 12). -/
theorem remainder_of_five_consecutive_odds_mod_12 {x : ℤ} (h : x % 12 = 5) :
  (x + (x + 2) + (x + 4) + (x + 6) + (x + 8)) % 12 = 9 :=
sorry

end remainder_of_five_consecutive_odds_mod_12_l176_176518


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176904

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176904


namespace sum_of_largest_three_consecutive_numbers_l176_176057

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176057


namespace parabola_distance_l176_176190

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l176_176190


namespace root_division_simplification_l176_176849

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l176_176849


namespace beyonce_total_songs_l176_176542

theorem beyonce_total_songs (s a b t : ℕ) (h_s : s = 5) (h_a : a = 2 * 15) (h_b : b = 20) (h_t : t = s + a + b) : t = 55 := by
  rw [h_s, h_a, h_b] at h_t
  exact h_t

end beyonce_total_songs_l176_176542


namespace sequence_a_n_l176_176843

theorem sequence_a_n (a : ℕ → ℕ) (h₁ : a 1 = 1)
(h₂ : ∀ n : ℕ, n > 0 → a (n + 1) = a (n / 2) + a ((n + 1) / 2)) :
∀ n : ℕ, a n = n :=
by
  -- skip the proof with sorry
  sorry

end sequence_a_n_l176_176843


namespace difference_in_price_l176_176660

-- Definitions based on the given conditions
def price_with_cork : ℝ := 2.10
def price_cork : ℝ := 0.05
def price_without_cork : ℝ := price_with_cork - price_cork

-- The theorem proving the given question and correct answer
theorem difference_in_price : price_with_cork - price_without_cork = price_cork :=
by
  -- Proof can be omitted
  sorry

end difference_in_price_l176_176660


namespace min_value_of_sum_l176_176574

theorem min_value_of_sum (a b : ℝ) (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 2 = 6) :
  a + b ≥ 16 :=
sorry

end min_value_of_sum_l176_176574


namespace g_max_value_l176_176689

def g (n : ℕ) : ℕ :=
if n < 15 then n + 15 else g (n - 7)

theorem g_max_value : ∃ N : ℕ, ∀ n : ℕ, g n ≤ N ∧ N = 29 := 
by 
  sorry

end g_max_value_l176_176689


namespace integer_cubed_fraction_l176_176440

theorem integer_cubed_fraction
  (a b : ℕ)
  (hab : 0 < b ∧ 0 < a)
  (h : (a^2 + b^2) % (a - b)^2 = 0) :
  (a^3 + b^3) % (a - b)^3 = 0 :=
by sorry

end integer_cubed_fraction_l176_176440


namespace relation_among_a_b_c_l176_176721

theorem relation_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = (3 / 5)^4)
  (h2 : b = (3 / 5)^3)
  (h3 : c = Real.log (3 / 5) / Real.log 3) :
  c < a ∧ a < b :=
by
  sorry

end relation_among_a_b_c_l176_176721


namespace initial_student_count_l176_176495

theorem initial_student_count
  (n : ℕ)
  (T : ℝ)
  (h1 : T = 60.5 * (n : ℝ))
  (h2 : T - 8 = 64 * ((n - 1) : ℝ))
  : n = 16 :=
sorry

end initial_student_count_l176_176495


namespace tan_sin_cos_identity_l176_176912

theorem tan_sin_cos_identity {x : ℝ} (htan : Real.tan x = 1 / 3) : Real.sin x * Real.cos x + 1 = 13 / 10 :=
by
  sorry

end tan_sin_cos_identity_l176_176912


namespace remainder_4015_div_32_l176_176627

theorem remainder_4015_div_32 : 4015 % 32 = 15 := by
  sorry

end remainder_4015_div_32_l176_176627


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176905

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176905


namespace AB_distance_l176_176142

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l176_176142


namespace sum_q_p_eq_zero_l176_176369

def p (x : Int) : Int := x^2 - 4

def q (x : Int) : Int := 
  if x ≥ 0 then -x
  else x

def q_p (x : Int) : Int := q (p x)

#eval List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0

theorem sum_q_p_eq_zero :
  List.sum (List.map q_p [-3, -2, -1, 0, 1, 2, 3]) = 0 :=
sorry

end sum_q_p_eq_zero_l176_176369


namespace total_shelves_needed_l176_176585

def regular_shelf_capacity : Nat := 45
def large_shelf_capacity : Nat := 30
def regular_books : Nat := 240
def large_books : Nat := 75

def shelves_needed (book_count : Nat) (shelf_capacity : Nat) : Nat :=
  (book_count + shelf_capacity - 1) / shelf_capacity

theorem total_shelves_needed :
  shelves_needed regular_books regular_shelf_capacity +
  shelves_needed large_books large_shelf_capacity = 9 := by
sorry

end total_shelves_needed_l176_176585


namespace teams_in_double_round_robin_l176_176739
-- Import the standard math library

-- Lean statement for the proof problem
theorem teams_in_double_round_robin (m n : ℤ) 
  (h : 9 * n^2 + 6 * n + 32 = m * (m - 1) / 2) : 
  m = 8 ∨ m = 32 :=
sorry

end teams_in_double_round_robin_l176_176739


namespace jake_snake_length_l176_176943

theorem jake_snake_length (j p : ℕ) (h1 : j = p + 12) (h2 : j + p = 70) : j = 41 := by
  sorry

end jake_snake_length_l176_176943


namespace distance_inequality_l176_176590

theorem distance_inequality
  (n : ℕ)
  (P : Fin (n+1) → ℝ × ℝ)
  (d : ℝ)
  (h0 : 0 < d)
  (h1 : ∀ i j : Fin (n+1), i ≠ j → dist (P i) (P j) ≥ d) :
  ∏ i in Finset.filter (≠ 0) Finset.univ, dist (P 0) (P i) > (d / 3)^n * real.sqrt (nat.factorial (n+1)) :=
by
  sorry

end distance_inequality_l176_176590


namespace school_fee_l176_176371

theorem school_fee (a b c d e f g h i j k l : ℕ) (h1 : a = 2) (h2 : b = 100) (h3 : c = 1) (h4 : d = 50) (h5 : e = 5) (h6 : f = 20) (h7 : g = 3) (h8 : h = 10) (h9 : i = 4) (h10 : j = 5) (h11 : k = 4 ) (h12 : l = 50) :
  a * b + c * d + e * f + g * h + i * j + 3 * b + k * d + 2 * f + l * h + 6 * j = 980 := sorry

end school_fee_l176_176371


namespace tom_total_dimes_l176_176818

-- Define the original and additional dimes Tom received.
def original_dimes : ℕ := 15
def additional_dimes : ℕ := 33

-- Define the total number of dimes Tom has now.
def total_dimes : ℕ := original_dimes + additional_dimes

-- Statement to prove that the total number of dimes Tom has is 48.
theorem tom_total_dimes : total_dimes = 48 := by
  sorry

end tom_total_dimes_l176_176818


namespace product_of_cubes_eq_l176_176429

theorem product_of_cubes_eq :
  ( (3^3 - 1) / (3^3 + 1) ) * 
  ( (4^3 - 1) / (4^3 + 1) ) * 
  ( (5^3 - 1) / (5^3 + 1) ) * 
  ( (6^3 - 1) / (6^3 + 1) ) * 
  ( (7^3 - 1) / (7^3 + 1) ) * 
  ( (8^3 - 1) / (8^3 + 1) ) 
  = 73 / 256 :=
by
  sorry

end product_of_cubes_eq_l176_176429


namespace largest_of_consecutive_even_integers_l176_176810

theorem largest_of_consecutive_even_integers (x : ℤ) (h : 25 * (x + 24) = 10000) : x + 48 = 424 :=
sorry

end largest_of_consecutive_even_integers_l176_176810


namespace find_m_l176_176917

open Set

theorem find_m (m : ℝ) (A B : Set ℝ)
  (h1 : A = {-1, 3, 2 * m - 1})
  (h2 : B = {3, m})
  (h3 : B ⊆ A) : m = 1 ∨ m = -1 :=
by
  sorry

end find_m_l176_176917


namespace prob_first_three_heads_all_heads_l176_176641

-- Define the probability of a single flip resulting in heads
def prob_head : ℚ := 1 / 2

-- Define the probability of three consecutive heads for an independent and fair coin
def prob_three_heads (p : ℚ) : ℚ := p * p * p

theorem prob_first_three_heads_all_heads : prob_three_heads prob_head = 1 / 8 := 
sorry

end prob_first_three_heads_all_heads_l176_176641


namespace problem_statement_l176_176711

theorem problem_statement (m n : ℤ) (h : 3 * m - n = 1) : 9 * m ^ 2 - n ^ 2 - 2 * n = 1 := 
by sorry

end problem_statement_l176_176711


namespace total_roses_tom_sent_l176_176265

theorem total_roses_tom_sent
  (roses_in_dozen : ℕ := 12)
  (dozens_per_day : ℕ := 2)
  (days_in_week : ℕ := 7) :
  7 * (2 * 12) = 168 := by
  sorry

end total_roses_tom_sent_l176_176265


namespace angle_C_in_triangle_l176_176105

theorem angle_C_in_triangle {A B C : ℝ} 
  (h1 : A - B = 10) 
  (h2 : B = 0.5 * A) : 
  C = 150 :=
by
  -- Placeholder for proof
  sorry

end angle_C_in_triangle_l176_176105


namespace total_sessions_l176_176298

theorem total_sessions (p1 p2 p3 p4 : ℕ) 
(h1 : p1 = 6) 
(h2 : p2 = p1 + 5) 
(h3 : p3 = 8) 
(h4 : p4 = 8) : 
p1 + p2 + p3 + p4 = 33 := 
by
  sorry

end total_sessions_l176_176298


namespace number_of_dogs_on_tuesday_l176_176344

variable (T : ℕ)
variable (H1 : 7 + T + 7 + 7 + 9 = 42)

theorem number_of_dogs_on_tuesday : T = 12 := by
  sorry

end number_of_dogs_on_tuesday_l176_176344


namespace smallest_integer_in_set_of_seven_l176_176580

theorem smallest_integer_in_set_of_seven (n : ℤ) (h : n + 6 < 3 * (n + 3)) : n = -1 :=
sorry

end smallest_integer_in_set_of_seven_l176_176580


namespace increase_in_length_and_breadth_is_4_l176_176235

-- Define the variables for the original length and breadth of the room
variables (L B x : ℕ)

-- Define the original perimeter
def P_original : ℕ := 2 * (L + B)

-- Define the new perimeter after the increase
def P_new : ℕ := 2 * ((L + x) + (B + x))

-- Define the condition that the perimeter increases by 16 feet
axiom increase_perimeter : P_new L B x - P_original L B = 16

-- State the theorem that \(x = 4\)
theorem increase_in_length_and_breadth_is_4 : x = 4 :=
by
  -- Proof would be filled in here using the axioms and definitions
  sorry

end increase_in_length_and_breadth_is_4_l176_176235


namespace sum_quotient_product_diff_l176_176259

theorem sum_quotient_product_diff (x y : ℚ) (h₁ : x + y = 6) (h₂ : x / y = 6) : 
  (x * y) - (x - y) = 6 / 49 :=
  sorry

end sum_quotient_product_diff_l176_176259


namespace parabola_distance_l176_176130

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l176_176130


namespace simplify_expression_l176_176488

theorem simplify_expression :
  5 * (18 / 7) * (21 / -45) = -6 / 5 := 
sorry

end simplify_expression_l176_176488


namespace bottles_needed_to_fill_large_bottle_l176_176294

def medium_bottle_ml : ℕ := 150
def large_bottle_ml : ℕ := 1200

theorem bottles_needed_to_fill_large_bottle : large_bottle_ml / medium_bottle_ml = 8 :=
by
  sorry

end bottles_needed_to_fill_large_bottle_l176_176294


namespace abs_diff_l176_176346

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l176_176346


namespace haley_marble_distribution_l176_176455

theorem haley_marble_distribution (total_marbles : ℕ) (num_boys : ℕ) (h1 : total_marbles = 20) (h2 : num_boys = 2) : (total_marbles / num_boys) = 10 := 
by 
  sorry

end haley_marble_distribution_l176_176455


namespace pond_depth_l176_176359

theorem pond_depth (L W V D : ℝ) (hL : L = 20) (hW : W = 10) (hV : V = 1000) :
    V = L * W * D ↔ D = 5 := 
by
  rw [hL, hW, hV]
  constructor
  · intro h1
    linarith
  · intro h2
    rw [h2]
    linarith

#check pond_depth

end pond_depth_l176_176359


namespace find_n_l176_176474

def x := 3
def y := 1
def n := x - 3 * y^(x - y) + 1

theorem find_n : n = 1 :=
by
  unfold n x y
  sorry

end find_n_l176_176474


namespace find_a1_l176_176942

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a1 (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_sequence a →
  a 6 = 9 →
  a 3 = 3 * a 2 →
  a 1 = -1 :=
by
  sorry

end find_a1_l176_176942


namespace range_of_a_l176_176819

noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x

theorem range_of_a (a : ℝ) (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → a < g x) : a < 0 := 
by sorry

end range_of_a_l176_176819


namespace no_such_triplets_of_positive_reals_l176_176572

-- Define the conditions that the problem states.
def satisfies_conditions (a b c : ℝ) : Prop :=
  a = b + c ∧ b = c + a ∧ c = a + b

-- The main theorem to prove.
theorem no_such_triplets_of_positive_reals :
  ∀ (a b c : ℝ), (0 < a) → (0 < b) → (0 < c) → satisfies_conditions a b c → false :=
by
  intro a b c
  intro ha hb hc
  intro habc
  sorry

end no_such_triplets_of_positive_reals_l176_176572


namespace chocolate_pieces_l176_176659

theorem chocolate_pieces (total_pieces : ℕ) (michael_portion : ℕ) (paige_portion : ℕ) (mandy_portion : ℕ) 
  (h_total : total_pieces = 60) 
  (h_michael : michael_portion = total_pieces / 2) 
  (h_paige : paige_portion = (total_pieces - michael_portion) / 2) 
  (h_mandy : mandy_portion = total_pieces - (michael_portion + paige_portion)) : 
  mandy_portion = 15 :=
by
  sorry

end chocolate_pieces_l176_176659


namespace distance_AB_l176_176176

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l176_176176


namespace probability_first_three_heads_l176_176632

noncomputable def fair_coin : ProbabilityMassFunction ℕ :=
{ prob := {
    | 0 := 1/2, -- heads
    | 1 := 1/2, -- tails
    },
  prob_sum := by norm_num,
  prob_nonneg := by dec_trivial }

theorem probability_first_three_heads :
  (fair_coin.prob 0 * fair_coin.prob 0 * fair_coin.prob 0) = 1/8 :=
by {
  unfold fair_coin,
  norm_num,
  sorry
}

end probability_first_three_heads_l176_176632


namespace al_sandwiches_count_l176_176796

noncomputable def total_sandwiches (bread meat cheese : ℕ) : ℕ :=
  bread * meat * cheese

noncomputable def prohibited_combinations (bread_forbidden_combination cheese_forbidden_combination : ℕ) : ℕ := 
  bread_forbidden_combination + cheese_forbidden_combination

theorem al_sandwiches_count (bread meat cheese : ℕ) 
  (bread_forbidden_combination cheese_forbidden_combination : ℕ) 
  (h1 : bread = 5) 
  (h2 : meat = 7) 
  (h3 : cheese = 6) 
  (h4 : bread_forbidden_combination = 5) 
  (h5 : cheese_forbidden_combination = 6) : 
  total_sandwiches bread meat cheese - prohibited_combinations bread_forbidden_combination cheese_forbidden_combination = 199 :=
by
  sorry

end al_sandwiches_count_l176_176796


namespace sum_of_three_largest_l176_176048

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176048


namespace angle_AM_BN_eq_60_area_ABP_eq_area_MDNP_l176_176958

variables (A B C D E F M N P : Point)
variables (hexagon ABCDEF : RegularHexagon ABCDEF)
variables (M_mid : midpoint C D M)
variables (N_mid : midpoint D E N)
variables (P_intersect : intersection_point (line_through A M) (line_through B N) P)

-- Angle between AM and BN is 60 degrees
theorem angle_AM_BN_eq_60 (hex : RegularHexagon ABCDEF) (M : Point) (N : Point) (P : Point)
  (hM : midpoint C D M) (hN : midpoint D E N) (hP : intersection_point (line_through A M) (line_through B N) P) :
  angle (line_through A M) (line_through B N) = 60 :=
sorry

-- Area of triangle ABP is equal to the area of quadrilateral MDNP
theorem area_ABP_eq_area_MDNP (hex : RegularHexagon ABCDEF) (M : Point) (N : Point) (P : Point)
  (hM : midpoint C D M) (hN : midpoint D E N) (hP : intersection_point (line_through A M) (line_through B N) P) :
  area (triangle A B P) = area (quadrilateral M D N P) :=
sorry

end angle_AM_BN_eq_60_area_ABP_eq_area_MDNP_l176_176958


namespace sum_of_three_largest_of_consecutive_numbers_l176_176010

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176010


namespace pow_product_l176_176314

theorem pow_product (a b : ℝ) : (2 * a * b^2)^3 = 8 * a^3 * b^6 := 
by {
  sorry
}

end pow_product_l176_176314


namespace number_of_other_workers_l176_176944

theorem number_of_other_workers (N : ℕ) (h1 : N ≥ 2) (h2 : 1 / ((N * (N - 1)) / 2) = 1 / 6) : N - 2 = 2 :=
by
  sorry

end number_of_other_workers_l176_176944


namespace bound_on_k_l176_176116

variables {n k : ℕ}
variables (a : ℕ → ℕ) (h1 : 1 ≤ k) (h2 : ∀ i j, 1 ≤ i → j ≤ k → i < j → a i < a j)
variables (h3 : ∀ i, a i ≤ n) (h4 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → a i ≠ a j))
variables (h5 : (∀ i j : ℕ, i ≤ j → i ≤ k → j ≤ k → ∀ m p, m ≤ p → m ≤ k → p ≤ k → a i + a j ≠ a m + a p))

theorem bound_on_k : k ≤ Nat.floor (Real.sqrt (2 * n) + 1) :=
sorry

end bound_on_k_l176_176116


namespace find_g_l176_176791

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l176_176791


namespace july_savings_l176_176678

theorem july_savings (january: ℕ := 100) (total_savings: ℕ := 12700) :
  let february := 2 * january
  let march := 2 * february
  let april := 2 * march
  let may := 2 * april
  let june := 2 * may
  let july := 2 * june
  let total := january + february + march + april + may + june + july
  total = total_savings → july = 6400 := 
by
  sorry

end july_savings_l176_176678


namespace pages_after_break_correct_l176_176586

-- Definitions based on conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.7
def pages_before_break : ℕ := (total_pages : ℝ * break_percentage).to_nat
def pages_after_break : ℕ := total_pages - pages_before_break

-- Theorem statement
theorem pages_after_break_correct : pages_after_break = 9 :=
by
  -- The proof is unnecessary as per instructions
  sorry

end pages_after_break_correct_l176_176586


namespace subsets_union_l176_176114

theorem subsets_union (n m : ℕ) (h1 : n ≥ 3) (h2 : m ≥ 2^(n-1) + 1) 
  (A : Fin m → Finset (Fin n)) (hA : ∀ i j, i ≠ j → A i ≠ A j) 
  (hB : ∀ i, A i ≠ ∅) : 
  ∃ i j k, i ≠ j ∧ A i ∪ A j = A k := 
sorry

end subsets_union_l176_176114


namespace parabola_distance_l176_176137

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l176_176137


namespace simplify_expression_l176_176983

theorem simplify_expression (x y : ℝ) (h1 : x = 10) (h2 : y = -1/25) :
  ((xy + 2) * (xy - 2) - 2 * x^2 * y^2 + 4) / (xy) = 2 / 5 := 
by
  sorry

end simplify_expression_l176_176983


namespace train_length_l176_176832

theorem train_length (L V : ℝ) 
  (h1 : L = V * 110) 
  (h2 : L + 700 = V * 180) : 
  L = 1100 :=
by
  sorry

end train_length_l176_176832


namespace g_is_even_l176_176467

noncomputable def g (x : ℝ) : ℝ := 4^(x^2 - 3) - 2 * |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x := by
  sorry

end g_is_even_l176_176467


namespace parabola_distance_problem_l176_176151

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l176_176151


namespace ajay_walks_distance_l176_176307

theorem ajay_walks_distance (speed : ℝ) (time : ℝ) (distance : ℝ) 
  (h_speed : speed = 3) 
  (h_time : time = 16.666666666666668) : 
  distance = speed * time :=
by
  sorry

end ajay_walks_distance_l176_176307


namespace petes_average_speed_l176_176312

-- Definitions of the conditions
def map_distance : ℝ := 5 -- in inches
def driving_time : ℝ := 6.5 -- in hours
def map_scale : ℝ := 0.01282051282051282 -- in inches per mile

-- Theorem statement: If the conditions are given, then the average speed is 60 miles per hour
theorem petes_average_speed :
  (map_distance / map_scale) / driving_time = 60 :=
by
  -- The proof will go here
  sorry

end petes_average_speed_l176_176312


namespace parabola_distance_l176_176119

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l176_176119


namespace find_abs_xyz_l176_176957

noncomputable def conditions_and_question (x y z : ℝ) : Prop :=
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ z ≠ x ∧
  (x + 1/y = y + 1/z ∧ y + 1/z = z + 1/x + 1)

theorem find_abs_xyz (x y z : ℝ) (h : conditions_and_question x y z) : |x * y * z| = 1 :=
  sorry

end find_abs_xyz_l176_176957


namespace find_k_value_l176_176592

variable (S : ℕ → ℤ) (n : ℕ)

-- Conditions
def is_arithmetic_sum (S : ℕ → ℤ) : Prop :=
  ∃ (a d : ℤ), ∀ n : ℕ, S n = n * (2 * a + (n - 1) * d) / 2

axiom S3_eq_S8 (S : ℕ → ℤ) (hS : is_arithmetic_sum S) : S 3 = S 8
axiom Sk_eq_S7 (S : ℕ → ℤ) (k : ℕ) (hS: is_arithmetic_sum S)  : S 7 = S k

theorem find_k_value (S : ℕ → ℤ) (hS: is_arithmetic_sum S) :  S 3 = S 8 → S 7 = S 4 :=
by
  sorry

end find_k_value_l176_176592


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176875

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176875


namespace find_n_cosine_l176_176705

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l176_176705


namespace parabola_problem_l176_176132

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176132


namespace grade_students_difference_condition_l176_176395

variables (G1 G2 G5 : ℕ)

theorem grade_students_difference_condition (h : G1 + G2 = G2 + G5 + 30) : G1 - G5 = 30 :=
sorry

end grade_students_difference_condition_l176_176395


namespace johns_father_age_l176_176746

variable {Age : Type} [OrderedRing Age]
variables (J M F : Age)

def john_age := J
def mother_age := M
def father_age := F

def john_younger_than_father (F J : Age) : Prop := F = 2 * J
def father_older_than_mother (F M : Age) : Prop := F = M + 4
def age_difference_between_john_and_mother (M J : Age) : Prop := M = J + 16

-- The question to be proved in Lean:
theorem johns_father_age :
  john_younger_than_father F J →
  father_older_than_mother F M →
  age_difference_between_john_and_mother M J →
  F = 40 := 
by
  intros h1 h2 h3
  sorry

end johns_father_age_l176_176746


namespace problem_statement_l176_176453

noncomputable def A : Set ℝ := {x | x^2 - 3*x + 2 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | x^2 - m*x + 2 = 0}

theorem problem_statement (m : ℝ) : (A ∩ (B m) = B m) → (m = 3 ∨ (-2 * Real.sqrt 2 < m ∧ m < 2 * Real.sqrt 2)) :=
by
  sorry

end problem_statement_l176_176453


namespace completing_the_square_l176_176514

theorem completing_the_square {x : ℝ} : x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := 
sorry

end completing_the_square_l176_176514


namespace rem_fraction_l176_176841

theorem rem_fraction : 
  let rem (x y : ℚ) : ℚ := x - y * ⌊x / y⌋;
  rem (5/7) (-3/4) = -1/28 := 
by
  sorry

end rem_fraction_l176_176841


namespace sum_of_numbers_l176_176264

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def tens_digit_zero (n : ℕ) : Prop := (n / 10) % 10 = 0
def units_digit_nonzero (n : ℕ) : Prop := n % 10 ≠ 0
def same_units_digits (m n : ℕ) : Prop := m % 10 = n % 10

theorem sum_of_numbers (a b c : ℕ)
  (h1 : is_perfect_square a) (h2 : is_perfect_square b) (h3 : is_perfect_square c)
  (h4 : tens_digit_zero a) (h5 : tens_digit_zero b) (h6 : tens_digit_zero c)
  (h7 : units_digit_nonzero a) (h8 : units_digit_nonzero b) (h9 : units_digit_nonzero c)
  (h10 : same_units_digits b c)
  (h11 : a % 10 % 2 = 0) :
  a + b + c = 14612 :=
sorry

end sum_of_numbers_l176_176264


namespace right_triangle_third_side_l176_176938

/-- In a right triangle, given the lengths of two sides are 4 and 5, prove that the length of the
third side is either sqrt 41 or 3. -/
theorem right_triangle_third_side (a b : ℕ) (h1 : a = 4 ∨ a = 5) (h2 : b = 4 ∨ b = 5) (h3 : a ≠ b) :
  ∃ c, c = Real.sqrt 41 ∨ c = 3 :=
by
  sorry

end right_triangle_third_side_l176_176938


namespace sum_of_solutions_eq_one_l176_176806

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l176_176806


namespace captain_age_l176_176934

-- Definitions
def num_team_members : ℕ := 11
def total_team_age : ℕ := 11 * 24
def total_age_remainder := 9 * (24 - 1)
def combined_age_of_captain_and_keeper := total_team_age - total_age_remainder

-- The actual proof statement
theorem captain_age (C : ℕ) (W : ℕ) 
  (hW : W = C + 5)
  (h_total_team : total_team_age = 264)
  (h_total_remainders : total_age_remainder = 207)
  (h_combined_age : combined_age_of_captain_and_keeper = 57) :
  C = 26 :=
by sorry

end captain_age_l176_176934


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176885

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176885


namespace inequal_min_value_l176_176082

theorem inequal_min_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 4) : 
  (1/x + 4/y) ≥ 9/4 :=
sorry

end inequal_min_value_l176_176082


namespace find_m_plus_n_l176_176666

def num_fir_trees : ℕ := 4
def num_pine_trees : ℕ := 5
def num_acacia_trees : ℕ := 6

def num_non_acacia_trees : ℕ := num_fir_trees + num_pine_trees
def total_trees : ℕ := num_fir_trees + num_pine_trees + num_acacia_trees

def prob_no_two_acacia_adj : ℚ :=
  (Nat.choose (num_non_acacia_trees + 1) num_acacia_trees * Nat.choose num_non_acacia_trees num_fir_trees : ℚ) /
  Nat.choose total_trees num_acacia_trees

theorem find_m_plus_n : (prob_no_two_acacia_adj = 84/159) -> (84 + 159 = 243) :=
by {
  admit
}

end find_m_plus_n_l176_176666


namespace ship_with_highest_no_car_round_trip_percentage_l176_176764

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l176_176764


namespace smallest_n_probability_l176_176848

theorem smallest_n_probability (n : ℕ) : (1 / (n * (n + 1)) < 1 / 2023) → (n ≥ 45) :=
by
  sorry

end smallest_n_probability_l176_176848


namespace tim_total_spending_l176_176460

def lunch_cost : ℝ := 50.50
def dessert_cost : ℝ := 8.25
def beverage_cost : ℝ := 3.75
def lunch_discount : ℝ := 0.10
def dessert_tax : ℝ := 0.07
def beverage_tax : ℝ := 0.05
def lunch_tip_rate : ℝ := 0.20
def other_items_tip_rate : ℝ := 0.15

def total_spending : ℝ := 
  let lunch_after_discount := lunch_cost * (1 - lunch_discount)
  let dessert_after_tax := dessert_cost * (1 + dessert_tax)
  let beverage_after_tax := beverage_cost * (1 + beverage_tax)
  let tip_on_lunch := lunch_after_discount * lunch_tip_rate
  let combined_other_items := dessert_after_tax + beverage_after_tax
  let tip_on_other_items := combined_other_items * other_items_tip_rate
  lunch_after_discount + dessert_after_tax + beverage_after_tax + tip_on_lunch + tip_on_other_items

theorem tim_total_spending :
  total_spending = 69.23 :=
by
  sorry

end tim_total_spending_l176_176460


namespace trigonometric_inequalities_l176_176332

theorem trigonometric_inequalities (θ : ℝ) (h1 : Real.sin (θ + Real.pi) < 0) (h2 : Real.cos (θ - Real.pi) > 0) : 
  Real.sin θ > 0 ∧ Real.cos θ < 0 :=
sorry

end trigonometric_inequalities_l176_176332


namespace rectangle_width_decrease_l176_176244

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l176_176244


namespace patsy_deviled_eggs_l176_176482

-- Definitions based on given problem conditions
def guests : ℕ := 30
def appetizers_per_guest : ℕ := 6
def total_appetizers_needed : ℕ := appetizers_per_guest * guests
def pigs_in_blanket : ℕ := 2
def kebabs : ℕ := 2
def additional_appetizers_needed (already_planned : ℕ) : ℕ := 8 + already_planned
def already_planned_appetizers : ℕ := pigs_in_blanket + kebabs
def total_appetizers_planned : ℕ := additional_appetizers_needed already_planned_appetizers

-- The proof problem statement
theorem patsy_deviled_eggs : total_appetizers_needed = total_appetizers_planned * 12 → 
                            total_appetizers_planned = already_planned_appetizers + 8 →
                            (total_appetizers_planned - already_planned_appetizers) = 8 :=
by
  sorry

end patsy_deviled_eggs_l176_176482


namespace parabola_problem_l176_176162

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l176_176162


namespace train_length_proof_l176_176413

-- Definitions for conditions
def jogger_speed_kmh : ℕ := 9
def train_speed_kmh : ℕ := 45
def initial_distance_ahead_m : ℕ := 280
def time_to_pass_s : ℕ := 40

-- Conversion factors
def km_per_hr_to_m_per_s (speed_kmh : ℕ) : ℕ := speed_kmh * 1000 / 3600

-- Converted speeds
def jogger_speed_m_per_s : ℕ := km_per_hr_to_m_per_s jogger_speed_kmh
def train_speed_m_per_s : ℕ := km_per_hr_to_m_per_s train_speed_kmh

-- Relative speed
def relative_speed_m_per_s : ℕ := train_speed_m_per_s - jogger_speed_m_per_s

-- Distance covered relative to the jogger
def distance_covered_relative_m : ℕ := relative_speed_m_per_s * time_to_pass_s

-- Length of the train
def length_of_train_m : ℕ := distance_covered_relative_m + initial_distance_ahead_m

-- Theorem to prove 
theorem train_length_proof : length_of_train_m = 680 := 
by
   sorry

end train_length_proof_l176_176413


namespace least_positive_integer_x_l176_176516

theorem least_positive_integer_x :
  ∃ x : ℕ, (x > 0) ∧ (∃ k : ℕ, (2 * x + 51) = k * 59) ∧ x = 4 :=
by
  -- Lean statement
  sorry

end least_positive_integer_x_l176_176516


namespace mean_of_reciprocals_first_four_primes_l176_176861

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176861


namespace least_number_to_add_l176_176630

theorem least_number_to_add (n : ℕ) : 
  (∀ k : ℕ, n = 1 + k * 425 ↔ n + 1019 % 425 = 0) → n = 256 := 
sorry

end least_number_to_add_l176_176630


namespace sum_max_min_eq_four_l176_176340

noncomputable def f (x : ℝ) : ℝ :=
  (|2 * x| + x^3 + 2) / (|x| + 1)

-- Define the maximum value M and minimum value m
noncomputable def M : ℝ := sorry -- The maximum value of the function f(x)
noncomputable def m : ℝ := sorry -- The minimum value of the function f(x)

theorem sum_max_min_eq_four : M + m = 4 := by
  sorry

end sum_max_min_eq_four_l176_176340


namespace probability_four_lights_needed_expected_replacements_needed_l176_176507

/- Part (a) -/
def probability_four_replacements (n : ℕ) (k : ℕ) : ℝ := 
  if n = 9 ∧ k = 4 then (25:ℝ) / 84 else 0

theorem probability_four_lights_needed : 
  probability_four_replacements 9 4 = 25 / 84 :=
sorry

/- Part (b) -/
def expected_replacements (n : ℕ) : ℝ := 
  if n = 9 then 837 / 252 else 0

theorem expected_replacements_needed : 
  expected_replacements 9 = 837 / 252 :=
sorry

end probability_four_lights_needed_expected_replacements_needed_l176_176507


namespace number_of_classes_l176_176354

theorem number_of_classes (x : ℕ) (h : x * (x - 1) = 20) : x = 5 :=
by
  sorry

end number_of_classes_l176_176354


namespace find_k_l176_176456

theorem find_k 
  (x k : ℚ)
  (h1 : (x^2 - 3*k)*(x + 3*k) = x^3 + 3*k*(x^2 - x - 7))
  (h2 : k ≠ 0) : k = 7 / 3 := 
sorry

end find_k_l176_176456


namespace rectangle_width_decrease_l176_176238

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l176_176238


namespace quadratic_rewrite_h_l176_176932

theorem quadratic_rewrite_h (a k h x : ℝ) :
  (3 * x^2 + 9 * x + 17) = a * (x - h)^2 + k ↔ h = -3/2 :=
by sorry

end quadratic_rewrite_h_l176_176932


namespace rectangle_width_decrease_l176_176242

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l176_176242


namespace no_pairs_probability_l176_176935

-- Define the number of socks and initial conditions
def pairs_of_socks : ℕ := 3
def total_socks : ℕ := pairs_of_socks * 2

-- Probabilistic outcome space for no pairs in first three draws
def probability_no_pairs_in_first_three_draws : ℚ :=
  (4/5) * (1/2)

-- Theorem stating that probability of no matching pairs in the first three draws is 2/5
theorem no_pairs_probability : probability_no_pairs_in_first_three_draws = 2/5 := by
  sorry

end no_pairs_probability_l176_176935


namespace smaller_angle_at_9_15_l176_176400

theorem smaller_angle_at_9_15 (h_degree : ℝ) (m_degree : ℝ) (smaller_angle : ℝ) :
  (h_degree = 277.5) → (m_degree = 90) → (smaller_angle = 172.5) :=
by
  sorry

end smaller_angle_at_9_15_l176_176400


namespace maximize_ab2c3_l176_176573

def positive_numbers (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 

def sum_constant (a b c A : ℝ) : Prop :=
  a + b + c = A

noncomputable def maximize_expression (a b c : ℝ) : ℝ :=
  a * b^2 * c^3

theorem maximize_ab2c3 (a b c A : ℝ) (h1 : positive_numbers a b c)
  (h2 : sum_constant a b c A) : 
  maximize_expression a b c ≤ maximize_expression (A / 6) (A / 3) (A / 2) :=
sorry

end maximize_ab2c3_l176_176573


namespace incorrect_statement_B_l176_176401

open Set

-- Define the relevant events as described in the problem
def event_subscribe_at_least_one (ω : Type) (A B : Set ω) : Set ω := A ∪ B
def event_subscribe_at_most_one (ω : Type) (A B : Set ω) : Set ω := (A ∩ B)ᶜ

-- Define the problem statement
theorem incorrect_statement_B (ω : Type) (A B : Set ω) :
  ¬ (event_subscribe_at_least_one ω A B) = (event_subscribe_at_most_one ω A B)ᶜ :=
sorry

end incorrect_statement_B_l176_176401


namespace number_of_possible_values_of_s_l176_176254

noncomputable def s := {s : ℚ | ∃ w x y z : ℕ, s = w / 1000 + x / 10000 + y / 100000 + z / 1000000 ∧ w < 10 ∧ x < 10 ∧ y < 10 ∧ z < 10}

theorem number_of_possible_values_of_s (s_approx : s → ℚ → Prop) (h_s_approx : ∀ s, s_approx s (3 / 11)) :
  ∃ n : ℕ, n = 266 :=
by
  sorry

end number_of_possible_values_of_s_l176_176254


namespace sum_of_a_and_c_l176_176990

variable {R : Type} [LinearOrderedField R]

theorem sum_of_a_and_c
    (ha hb hc hd : R) 
    (h_intersect : (1, 7) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (1, 7) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}
                 ∧ (9, 1) ∈ {p | p.2 = -2 * abs (p.1 - ha) + hb} ∧ (9, 1) ∈ {p | p.2 = 2 * abs (p.1 - hc) + hd}) :
  ha + hc = 10 :=
by
  sorry

end sum_of_a_and_c_l176_176990


namespace sam_collected_42_cans_l176_176287

noncomputable def total_cans_collected (bags_saturday : ℕ) (bags_sunday : ℕ) (cans_per_bag : ℕ) : ℕ :=
  bags_saturday + bags_sunday * cans_per_bag

theorem sam_collected_42_cans :
  total_cans_collected 4 3 6 = 42 :=
by
  sorry

end sam_collected_42_cans_l176_176287


namespace alice_height_after_growth_l176_176680

/-- Conditions: Bob and Alice were initially the same height. Bob has grown by 25%, Alice 
has grown by one third as many inches as Bob, and Bob is now 75 inches tall. --/
theorem alice_height_after_growth (initial_height : ℕ)
  (bob_growth_rate : ℚ)
  (alice_growth_ratio : ℚ)
  (bob_final_height : ℕ) :
  bob_growth_rate = 0.25 →
  alice_growth_ratio = 1 / 3 →
  bob_final_height = 75 →
  initial_height + (bob_final_height - initial_height) / 3 = 65 :=
by
  sorry

end alice_height_after_growth_l176_176680


namespace part1_solution_l176_176570

theorem part1_solution (x : ℝ) (f : ℝ → ℝ) (hf : ∀ x, f x = |x|) :
  f x + f (x - 1) ≤ 2 ↔ x ∈ set.Icc (-1 / 2 : ℝ) (3 / 2 : ℝ) :=
sorry

end part1_solution_l176_176570


namespace choir_members_max_l176_176661

theorem choir_members_max (s x : ℕ) (h1 : s * x < 147) (h2 : s * x + 3 = (s - 3) * (x + 2)) : s * x = 84 :=
sorry

end choir_members_max_l176_176661


namespace sum_of_ages_l176_176214

-- Defining the ages of Nathan and his twin sisters.
variables (n t : ℕ)

-- Nathan has two twin younger sisters, and the product of their ages equals 72.
def valid_ages (n t : ℕ) : Prop := t < n ∧ n * t * t = 72

-- Prove that the sum of the ages of Nathan and his twin sisters is 14.
theorem sum_of_ages (n t : ℕ) (h : valid_ages n t) : 2 * t + n = 14 :=
sorry

end sum_of_ages_l176_176214


namespace expected_value_sum_path_find_p_plus_q_l176_176589

noncomputable def expected_sum_path : ℚ := (100850 + 201700 + 2000)

theorem expected_value_sum_path 
    (a : Finset ℕ) (b : Finset ℕ)
    (H₁ : a.card = 100) 
    (H₂ : b.card = 200) 
    (H₃ : ∀ x ∈ a, 1 ≤ x ∧ x ≤ 2016) 
    (H₄ : ∀ y ∈ b, 1 ≤ y ∧ y ≤ 2016)
    (H₅ : a.val.nodup) (H₆ : b.val.nodup) : 
    (∑ i in a, i + ∑ j in b, j + 100 * (a.min' (by linarith))) = 304550 := 
begin
  sorry
end

theorem find_p_plus_q : 304550 + 1 = 304551 := 
begin
  norm_num,
end

end expected_value_sum_path_find_p_plus_q_l176_176589


namespace snow_probability_l176_176966

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l176_176966


namespace race_distance_l176_176740

theorem race_distance 
  (D : ℝ) 
  (A_time : ℝ) (B_time : ℝ) 
  (A_beats_B_by : ℝ) 
  (A_time_eq : A_time = 36)
  (B_time_eq : B_time = 45)
  (A_beats_B_by_eq : A_beats_B_by = 24) :
  ((D / A_time) * B_time = D + A_beats_B_by) -> D = 24 := 
by 
  sorry

end race_distance_l176_176740


namespace find_b_l176_176615

theorem find_b (a b : ℝ) (h_inv_var : a^2 * Real.sqrt b = k) (h_ab : a * b = 72) (ha3 : a = 3) (hb64 : b = 64) : b = 18 :=
sorry

end find_b_l176_176615


namespace percentage_solution_P_mixture_l176_176384

-- Define constants for volumes and percentages
variables (P Q : ℝ)

-- Define given conditions
def percentage_lemonade_P : ℝ := 0.2
def percentage_carbonated_P : ℝ := 0.8
def percentage_lemonade_Q : ℝ := 0.45
def percentage_carbonated_Q : ℝ := 0.55
def percentage_carbonated_mixture : ℝ := 0.72

-- Prove that the percentage of the volume of the mixture that is Solution P is 68%
theorem percentage_solution_P_mixture : 
  (percentage_carbonated_P * P + percentage_carbonated_Q * Q = percentage_carbonated_mixture * (P + Q)) → 
  ((P / (P + Q)) * 100 = 68) :=
by
  -- proof skipped
  sorry

end percentage_solution_P_mixture_l176_176384


namespace parabola_distance_l176_176117

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l176_176117


namespace abs_diff_l176_176347

theorem abs_diff (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : ((m^2 + n^2 + 81 + 64 + 100) / 5) - 81 = 2) :
  |m - n| = 4 := by
  sorry

end abs_diff_l176_176347


namespace fourth_pentagon_has_31_dots_l176_176940

-- Conditions representing the sequence of pentagons
def first_pentagon_dots : ℕ := 1

def second_pentagon_dots : ℕ := first_pentagon_dots + 5

def nth_layer_dots (n : ℕ) : ℕ := 5 * (n - 1)

def nth_pentagon_dots (n : ℕ) : ℕ :=
  (List.range n).foldl (λ acc k => acc + nth_layer_dots (k+1)) first_pentagon_dots

-- Question and proof statement
theorem fourth_pentagon_has_31_dots : nth_pentagon_dots 4 = 31 :=
  sorry

end fourth_pentagon_has_31_dots_l176_176940


namespace cos_eq_43_l176_176699

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l176_176699


namespace mean_of_reciprocals_first_four_primes_l176_176864

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176864


namespace sum_of_three_largest_l176_176019

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176019


namespace soda_preference_count_eq_243_l176_176941

def total_respondents : ℕ := 540
def soda_angle : ℕ := 162
def total_circle_angle : ℕ := 360

theorem soda_preference_count_eq_243 :
  (total_respondents * soda_angle / total_circle_angle) = 243 := 
by 
  sorry

end soda_preference_count_eq_243_l176_176941


namespace minimum_omega_l176_176568

/-- Given function f and its properties, determine the minimum valid ω. -/
theorem minimum_omega {f : ℝ → ℝ} 
  (Hf : ∀ x : ℝ, f x = (1 / 2) * Real.cos (ω * x + φ) + 1)
  (Hsymmetry : ∃ k : ℤ, ω * (π / 3) + φ = k * π)
  (Hvalue : ∃ n : ℤ, f (π / 12) = 1 ∧ ω * (π / 12) + φ = n * π + π / 2)
  (Hpos : ω > 0) : ω = 2 := 
sorry

end minimum_omega_l176_176568


namespace find_flag_count_l176_176430

-- Definitions of conditions
inductive Color
| purple
| gold
| silver

-- Function to count valid flags
def countValidFlags : Nat :=
  let first_stripe_choices := 3
  let second_stripe_choices := 2
  let third_stripe_choices := 2
  first_stripe_choices * second_stripe_choices * third_stripe_choices

-- Statement to prove
theorem find_flag_count : countValidFlags = 12 := by
  sorry

end find_flag_count_l176_176430


namespace ship_B_has_highest_rt_no_cars_l176_176768

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l176_176768


namespace distance_AB_l176_176126

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l176_176126


namespace average_rate_of_change_l176_176387

noncomputable def f (x : ℝ) : ℝ := x^2 + x

theorem average_rate_of_change :
  (f 2 - f 1) / (2 - 1) = 4 :=
by
  sorry

end average_rate_of_change_l176_176387


namespace solve_eq1_solve_eq2_l176_176385

theorem solve_eq1 : (∃ x : ℚ, (5 * x - 1) / 4 = (3 * x + 1) / 2 - (2 - x) / 3) ↔ x = -1 / 7 :=
sorry

theorem solve_eq2 : (∃ x : ℚ, (3 * x + 2) / 2 - 1 = (2 * x - 1) / 4 - (2 * x + 1) / 5) ↔ x = -9 / 28 :=
sorry

end solve_eq1_solve_eq2_l176_176385


namespace two_bags_remainder_l176_176550

-- Given conditions
variables (n : ℕ)

-- Assume n ≡ 8 (mod 11)
def satisfied_mod_condition : Prop := n % 11 = 8

-- Prove that 2n ≡ 5 (mod 11)
theorem two_bags_remainder (h : satisfied_mod_condition n) : (2 * n) % 11 = 5 :=
by 
  unfold satisfied_mod_condition at h
  sorry

end two_bags_remainder_l176_176550


namespace third_root_of_polynomial_l176_176329

variable (a b x : ℝ)
noncomputable def polynomial := a * x^3 + (a + 3 * b) * x^2 + (b - 4 * a) * x + (10 - a)

theorem third_root_of_polynomial (h1 : polynomial a b (-3) = 0) (h2 : polynomial a b 4 = 0) :
  ∃ r : ℝ, r = -17 / 10 ∧ polynomial a b r = 0 :=
by
  sorry

end third_root_of_polynomial_l176_176329


namespace tangent_values_l176_176449

theorem tangent_values (A : ℝ) (h : A < π) (cos_A : Real.cos A = 3 / 5) :
  Real.tan A = 4 / 3 ∧ Real.tan (A + π / 4) = -7 := 
by
  sorry

end tangent_values_l176_176449


namespace tripodasaurus_flock_l176_176996

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end tripodasaurus_flock_l176_176996


namespace max_ab_condition_l176_176454

-- Define the circles and the tangency condition
def circle1 (a : ℝ) : Set (ℝ × ℝ) := {p | (p.1 - a)^2 + (p.2 + 2)^2 = 4}
def circle2 (b : ℝ) : Set (ℝ × ℝ) := {p | (p.1 + b)^2 + (p.2 + 2)^2 = 1}
def internally_tangent (a b : ℝ) : Prop := (a + b) ^ 2 = 1

-- Define the maximum value condition
def max_ab (a b : ℝ) : ℝ := a * b

-- Main theorem
theorem max_ab_condition {a b : ℝ} (h_tangent : internally_tangent a b) : max_ab a b ≤ 1 / 4 :=
by
  -- Proof steps are not necessary, so we use sorry to end the proof.
  sorry

end max_ab_condition_l176_176454


namespace sum_of_solutions_eq_one_l176_176805

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l176_176805


namespace f_neg_two_l176_176070

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_neg_two (a b c : ℝ) (h : f a b c 2 = 4) : f a b c (-2) = 0 :=
sorry

end f_neg_two_l176_176070


namespace probability_of_first_three_heads_l176_176644

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l176_176644


namespace xy_equals_252_l176_176564

-- Definitions and conditions
variables (x y : ℕ) -- positive integers
variable (h1 : x + y = 36)
variable (h2 : 4 * x * y + 12 * x = 5 * y + 390)

-- Statement of the problem
theorem xy_equals_252 (h1 : x + y = 36) (h2 : 4 * x * y + 12 * x = 5 * y + 390) : x * y = 252 := by 
  sorry

end xy_equals_252_l176_176564


namespace min_employees_needed_l176_176675

theorem min_employees_needed
  (W A S : Finset ℕ)
  (hW : W.card = 120)
  (hA : A.card = 150)
  (hS : S.card = 100)
  (hWA : (W ∩ A).card = 50)
  (hAS : (A ∩ S).card = 30)
  (hWS : (W ∩ S).card = 20)
  (hWAS : (W ∩ A ∩ S).card = 10) :
  (W ∪ A ∪ S).card = 280 :=
by
  sorry

end min_employees_needed_l176_176675


namespace polynomial_g_l176_176787

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end polynomial_g_l176_176787


namespace mean_of_reciprocals_of_first_four_primes_l176_176889

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176889


namespace probability_of_snow_at_least_once_first_week_l176_176975

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l176_176975


namespace probability_three_heads_l176_176637

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l176_176637


namespace find_percentage_of_number_l176_176827

theorem find_percentage_of_number (P : ℝ) (N : ℝ) (h1 : P * N = (4 / 5) * N - 21) (h2 : N = 140) : P * 100 = 65 := 
by 
  sorry

end find_percentage_of_number_l176_176827


namespace sum_of_largest_three_l176_176040

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176040


namespace parabola_equation_l176_176501

variables (x y : ℝ)

def parabola_passes_through_point (x y : ℝ) : Prop :=
(x = 2 ∧ y = 7)

def focus_x_coord_five (x : ℝ) : Prop :=
(x = 5)

def axis_of_symmetry_parallel_to_y : Prop := True

def vertex_lies_on_x_axis (x y : ℝ) : Prop :=
(x = 5 ∧ y = 0)

theorem parabola_equation
  (h1 : parabola_passes_through_point x y)
  (h2 : focus_x_coord_five x)
  (h3 : axis_of_symmetry_parallel_to_y)
  (h4 : vertex_lies_on_x_axis x y) :
  49 * x + 3 * y^2 - 245 = 0
:= sorry

end parabola_equation_l176_176501


namespace parabola_distance_problem_l176_176150

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l176_176150


namespace AB_distance_l176_176146

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l176_176146


namespace geometric_sequence_general_term_l176_176726

variable (a : ℕ → ℝ)
variable (n : ℕ)

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n
  
theorem geometric_sequence_general_term 
  (h_geo : is_geometric_sequence a)
  (h_a3 : a 3 = 3)
  (h_a10 : a 10 = 384) :
  a n = 3 * 2^(n-3) :=
by sorry

end geometric_sequence_general_term_l176_176726


namespace smallest_n_satisfying_ratio_l176_176685

-- Definitions and conditions from problem
def sum_first_n_odd_numbers_starting_from_3 (n : ℕ) : ℕ := n^2 + 2 * n
def sum_first_n_even_numbers (n : ℕ) : ℕ := n * (n + 1)

theorem smallest_n_satisfying_ratio :
  ∃ n : ℕ, n > 0 ∧ (sum_first_n_odd_numbers_starting_from_3 n : ℚ) / (sum_first_n_even_numbers n : ℚ) = 49 / 50 ∧ n = 51 :=
by
  use 51
  exact sorry

end smallest_n_satisfying_ratio_l176_176685


namespace ratio_of_fusilli_to_penne_l176_176464

def number_of_students := 800
def preferred_pasta_types := ["penne", "tortellini", "fusilli", "spaghetti"]
def students_prefer_fusilli := 320
def students_prefer_penne := 160

theorem ratio_of_fusilli_to_penne : (students_prefer_fusilli / students_prefer_penne) = 2 := by
  -- Here we would provide the proof, but since it's a statement, we use sorry
  sorry

end ratio_of_fusilli_to_penne_l176_176464


namespace probability_distribution_correct_l176_176612

noncomputable def probability_of_hit : ℝ := 0.1
noncomputable def probability_of_miss : ℝ := 1 - probability_of_hit

def X_distribution : Fin 4 → ℝ
| ⟨3, _⟩ => probability_of_hit
| ⟨2, _⟩ => probability_of_miss * probability_of_hit
| ⟨1, _⟩ => probability_of_miss^2 * probability_of_hit
| ⟨0, _⟩ => probability_of_miss^3 * probability_of_hit + probability_of_miss^4

theorem probability_distribution_correct :
  X_distribution ⟨0, by simp⟩ = 0.729 ∧
  X_distribution ⟨1, by simp⟩ = 0.081 ∧
  X_distribution ⟨2, by simp⟩ = 0.09 ∧
  X_distribution ⟨3, by simp⟩ = 0.1 :=
by
  sorry

end probability_distribution_correct_l176_176612


namespace probability_of_at_least_one_vowel_is_799_over_1024_l176_176227

def Set1 : Set Char := {'a', 'e', 'i', 'b', 'c', 'd', 'f', 'g'}
def Set2 : Set Char := {'u', 'o', 'y', 'k', 'l', 'm', 'n', 'p'}
def Set3 : Set Char := {'e', 'u', 'v', 'r', 's', 't', 'w', 'x'}
def Set4 : Set Char := {'a', 'i', 'o', 'z', 'h', 'j', 'q', 'r'}

noncomputable def probability_of_at_least_one_vowel : ℚ :=
  1 - (5/8 : ℚ) * (3/4 : ℚ) * (3/4 : ℚ) * (5/8 : ℚ)

theorem probability_of_at_least_one_vowel_is_799_over_1024 :
  probability_of_at_least_one_vowel = 799 / 1024 :=
by
  sorry

end probability_of_at_least_one_vowel_is_799_over_1024_l176_176227


namespace fifth_equation_pattern_l176_176375

theorem fifth_equation_pattern :
  1^3 + 2^3 + 3^3 + 4^3 + 5^3 + 6^3 = 21^2 := 
by sorry

end fifth_equation_pattern_l176_176375


namespace group_size_increase_by_4_l176_176232

theorem group_size_increase_by_4
    (N : ℕ)
    (weight_old : ℕ)
    (weight_new : ℕ)
    (average_increase : ℕ)
    (weight_increase_diff : ℕ)
    (h1 : weight_old = 55)
    (h2 : weight_new = 87)
    (h3 : average_increase = 4)
    (h4 : weight_increase_diff = weight_new - weight_old)
    (h5 : average_increase * N = weight_increase_diff) :
    N = 8 :=
by
  sorry

end group_size_increase_by_4_l176_176232


namespace sum_of_three_largest_l176_176022

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176022


namespace sally_more_2s_than_5s_l176_176086

open BigOperators

-- Define the probability of rolling more 2's than 5's out of 6 rolls of an 8-sided die
def dice_rolls (prob : ℚ) : Prop :=
  (prob = (86684 : ℚ) / 262144)

noncomputable def probability_of_more_2s_than_5s : ℚ :=
  -- Here should go the full proof calculation skipped with sorry
  sorry

theorem sally_more_2s_than_5s :
  dice_rolls probability_of_more_2s_than_5s :=
begin
  -- The intended proof steps should go here, skipped with sorry
  sorry
end

end sally_more_2s_than_5s_l176_176086


namespace solve_x_l176_176984

theorem solve_x (x : ℝ) (h : (30 * x + 15)^(1/3) = 15) : x = 112 := by
  sorry

end solve_x_l176_176984


namespace find_x_in_interval_l176_176436

theorem find_x_in_interval (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  2 * Real.cos x ≤ abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ∧
  abs (Real.sqrt (1 + Real.sin (2 * x)) - Real.sqrt (1 - Real.sin (2 * x))) ≤ Real.sqrt 2 → 
  Real.pi / 4 ≤ x ∧ x ≤ 7 * Real.pi / 4 :=
by 
  sorry

end find_x_in_interval_l176_176436


namespace smallest_AAB_value_l176_176303

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 110 * A + B = 8 * (10 * A + B) ∧ ¬ (A = B) ∧ 110 * A + B = 773 :=
by sorry

end smallest_AAB_value_l176_176303


namespace remaining_regular_toenails_l176_176926

theorem remaining_regular_toenails :
  ∀ (total_capacity big_toenail_space big_toenails regular_toenails : ℕ),
    total_capacity = 100 →
    big_toenail_space = 2 →
    big_toenails = 20 →
    regular_toenails = 40 →
    let occupied_space := big_toenails * big_toenail_space + regular_toenails in
    let remaining_space := total_capacity - occupied_space in
    remaining_space = 20 :=
by
  intros total_capacity big_toenail_space big_toenails regular_toenails htcs hbs hbt hrt
  let occupied_space := big_toenails * big_toenail_space + regular_toenails
  have h1 : occupied_space = 40 * 2 + 40 := rfl
  let remaining_space := total_capacity - occupied_space
  have h2 : remaining_space = 100 - 80 := rfl
  have h3 : 20 = 20 := rfl
  exact h3

end remaining_regular_toenails_l176_176926


namespace sum_distinct_integers_l176_176751

theorem sum_distinct_integers (a b c d e : ℤ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) (h4 : a ≠ e)
    (h5 : b ≠ c) (h6 : b ≠ d) (h7 : b ≠ e) (h8 : c ≠ d) (h9 : c ≠ e) (h10 : d ≠ e)
    (h : (5 - a) * (5 - b) * (5 - c) * (5 - d) * (5 - e) = 120) :
    a + b + c + d + e = 13 := by
  sorry

end sum_distinct_integers_l176_176751


namespace position_of_z_l176_176964

theorem position_of_z (total_distance : ℕ) (total_steps : ℕ) (steps_taken : ℕ) (distance_covered : ℕ) (h1 : total_distance = 30) (h2 : total_steps = 6) (h3 : steps_taken = 4) (h4 : distance_covered = total_distance / total_steps) : 
  steps_taken * distance_covered = 20 :=
by
  sorry

end position_of_z_l176_176964


namespace sum_of_three_largest_l176_176047

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176047


namespace sum_of_first_30_terms_l176_176360

variable (a : Nat → ℤ)
variable (d : ℤ)
variable (S_30 : ℤ)

-- Conditions from part a)
def condition1 := a 1 + a 2 + a 3 = 3
def condition2 := a 28 + a 29 + a 30 = 165

-- Question translated to Lean 4 statement
theorem sum_of_first_30_terms 
  (h1 : condition1 a)
  (h2 : condition2 a) :
  S_30 = 840 := 
sorry

end sum_of_first_30_terms_l176_176360


namespace difference_of_squares_example_l176_176318

theorem difference_of_squares_example : (75^2 - 25^2) = 5000 := by
  let a := 75
  let b := 25
  have step1 : a + b = 100 := by
    rw [a, b]
    norm_num
  have step2 : a - b = 50 := by
    rw [a, b]
    norm_num
  have result : (a + b) * (a - b) = 5000 := by
    rw [step1, step2]
    norm_num
  rw [pow_two, pow_two, mul_sub, ← result]
  norm_num

end difference_of_squares_example_l176_176318


namespace carrots_total_l176_176980
-- import the necessary library

-- define the conditions as given
def sandy_carrots : Nat := 6
def sam_carrots : Nat := 3

-- state the problem as a theorem to be proven
theorem carrots_total : sandy_carrots + sam_carrots = 9 := by
  sorry

end carrots_total_l176_176980


namespace solve_x_for_equation_l176_176439

theorem solve_x_for_equation (x : ℝ) (h : 2 / (x + 3) + 3 * x / (x + 3) - 4 / (x + 3) = 4) : x = -14 :=
by 
  sorry

end solve_x_for_equation_l176_176439


namespace sum_of_three_largest_of_consecutive_numbers_l176_176012

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176012


namespace inequality_integral_ln_bounds_l176_176657

-- Define the conditions
variables (x a : ℝ)
variables (hx : 0 < x) (ha : x < a)

-- First part: inequality involving integral
theorem inequality_integral (hx : 0 < x) (ha : x < a) :
  (2 * x / a) < (∫ t in a - x..a + x, 1 / t) ∧ (∫ t in a - x..a + x, 1 / t) < x * (1 / (a + x) + 1 / (a - x)) :=
sorry

-- Second part: to prove 0.68 < ln(2) < 0.71 using the result of the first part
theorem ln_bounds :
  0.68 < Real.log 2 ∧ Real.log 2 < 0.71 :=
sorry

end inequality_integral_ln_bounds_l176_176657


namespace dancer_count_l176_176668

theorem dancer_count (n : ℕ) : 
  ((n + 5) % 12 = 0) ∧ ((n + 5) % 10 = 0) ∧ (200 ≤ n) ∧ (n ≤ 300) → (n = 235 ∨ n = 295) := 
by
  sorry

end dancer_count_l176_176668


namespace minimum_value_frac_l176_176091

theorem minimum_value_frac (a b : ℝ) (h₁ : 2 * a - b + 2 * 0 = 0) 
  (h₂ : a > 0) (h₃ : b > 0) (h₄ : a + b = 1) : 
  (1 / a) + (1 / b) = 4 :=
sorry

end minimum_value_frac_l176_176091


namespace necessary_but_not_sufficient_l176_176260

theorem necessary_but_not_sufficient :
    (∀ (x y : ℝ), x > 2 ∧ y > 3 → x + y > 5 ∧ x * y > 6) ∧ 
    ¬(∀ (x y : ℝ), x + y > 5 ∧ x * y > 6 → x > 2 ∧ y > 3) := by
  sorry

end necessary_but_not_sufficient_l176_176260


namespace bipartite_graph_acyclic_orientations_not_divisible_by_3_l176_176491

open Polynomial

noncomputable def χ_G (G : Type*) [fintype G] [decidable_eq G] : Polynomial ℤ := sorry

theorem bipartite_graph_acyclic_orientations_not_divisible_by_3 (G : Type*) [fintype G] [decidable_eq G] (hG : ∀ v₁ v₂ : G, v₁ ≠ v₂ → connected v₁ v₂) :
  ¬ (χ_G(G).eval (-1) % 3 = 0) :=
sorry

end bipartite_graph_acyclic_orientations_not_divisible_by_3_l176_176491


namespace unfair_coin_probability_l176_176540

theorem unfair_coin_probability (P : ℕ → ℝ) :
  let heads := 3/4
  let initial_condition := P 0 = 1
  let recurrence_relation := ∀n, P (n + 1) = 3 / 4 * (1 - P n) + 1 / 4 * P n
  recurrence_relation →
  initial_condition →
  P 40 = 1 / 2 * (1 + (1 / 2) ^ 40) :=
by
  sorry

end unfair_coin_probability_l176_176540


namespace hexagonal_pyramid_volume_l176_176505

theorem hexagonal_pyramid_volume (a : ℝ) (h : a > 0) (lateral_surface_area : ℝ) (base_area : ℝ)
  (H_base_area : base_area = (3 * Real.sqrt 3 / 2) * a^2)
  (H_lateral_surface_area : lateral_surface_area = 10 * base_area) :
  (1 / 3) * base_area * (a * Real.sqrt 3 / 2) * 3 * Real.sqrt 11 = (9 * a^3 * Real.sqrt 11) / 4 :=
by sorry

end hexagonal_pyramid_volume_l176_176505


namespace curve_in_second_quadrant_range_l176_176089

theorem curve_in_second_quadrant_range (a : ℝ) :
  (∀ (x y : ℝ), (x^2 + y^2 + 2*a*x - 4*a*y + 5*a^2 - 4 = 0 → x < 0 ∧ y > 0)) → a > 2 :=
by
  sorry

end curve_in_second_quadrant_range_l176_176089


namespace students_in_class_l176_176356

theorem students_in_class (b g : ℕ) 
  (h1 : b + g = 20)
  (h2 : (b : ℚ) / 20 = (3 : ℚ) / 4 * (g : ℚ) / 20) : 
  b = 12 ∧ g = 8 :=
by
  sorry

end students_in_class_l176_176356


namespace tourism_revenue_scientific_notation_l176_176278

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l176_176278


namespace kenneth_initial_money_l176_176364

-- Define the costs of the items
def cost_baguette := 2
def cost_water := 1

-- Define the quantities bought
def baguettes_bought := 2
def water_bought := 2

-- Define the amount left after buying the items
def money_left := 44

-- Calculate the total cost
def total_cost := (baguettes_bought * cost_baguette) + (water_bought * cost_water)

-- Define the initial money Kenneth had
def initial_money := total_cost + money_left

-- Prove the initial money is $50
theorem kenneth_initial_money : initial_money = 50 := 
by 
  -- The proof part is omitted because it is not required.
  sorry

end kenneth_initial_money_l176_176364


namespace mike_peaches_l176_176210

theorem mike_peaches (initial_peaches picked_peaches : ℝ) (h1 : initial_peaches = 34.0) (h2 : picked_peaches = 86.0) : initial_peaches + picked_peaches = 120.0 :=
by
  rw [h1, h2]
  norm_num

end mike_peaches_l176_176210


namespace distance_AB_l176_176201

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176201


namespace highest_percentage_without_car_l176_176771

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l176_176771


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176873

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176873


namespace parabola_distance_l176_176120

theorem parabola_distance 
  (F : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA : A.1^2 = 4 * A.2) -- A lies on the parabola
  (hAF_BF : dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
by
  have hAB : dist A B = 2 * Real.sqrt 2, from sorry
  exact hAB

end parabola_distance_l176_176120


namespace width_decrease_percentage_l176_176250

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l176_176250


namespace new_volume_of_cylinder_l176_176393

theorem new_volume_of_cylinder (r h : ℝ) (π : ℝ := Real.pi) (V : ℝ := π * r^2 * h) (hV : V = 15) :
  let r_new := 3 * r
  let h_new := 4 * h
  let V_new := π * (r_new)^2 * h_new
  V_new = 540 :=
by
  sorry

end new_volume_of_cylinder_l176_176393


namespace find_value_of_expression_l176_176445

theorem find_value_of_expression (a b : ℝ) (h : |a + 1| + (b - 2)^2 = 0) :
  (a + b)^9 + a^6 = 2 :=
sorry

end find_value_of_expression_l176_176445


namespace fruit_basket_count_l176_176083

theorem fruit_basket_count :
  let pears := 8
  let bananas := 12
  let total_baskets := (pears + 1) * (bananas + 1) - 1
  total_baskets = 116 :=
by
  sorry

end fruit_basket_count_l176_176083


namespace parabola_distance_l176_176129

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l176_176129


namespace cosine_periodicity_l176_176696

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l176_176696


namespace monotonic_increase_interval_l176_176991

noncomputable def interval_of_monotonic_increase (k : ℤ) : Set ℝ :=
  {x : ℝ | k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12}

theorem monotonic_increase_interval 
    (ω : ℝ)
    (hω : 0 < ω)
    (hperiod : Real.pi = 2 * Real.pi / ω) :
    ∀ k : ℤ, ∃ I : Set ℝ, I = interval_of_monotonic_increase k := 
by
  sorry

end monotonic_increase_interval_l176_176991


namespace time_worked_on_thursday_l176_176213

/-
  Given:
  - Monday: 3/4 hour
  - Tuesday: 1/2 hour
  - Wednesday: 2/3 hour
  - Friday: 75 minutes
  - Total (Monday to Friday): 4 hours = 240 minutes
  
  The time Mr. Willson worked on Thursday is 50 minutes.
-/

noncomputable def time_worked_monday : ℝ := (3 / 4) * 60
noncomputable def time_worked_tuesday : ℝ := (1 / 2) * 60
noncomputable def time_worked_wednesday : ℝ := (2 / 3) * 60
noncomputable def time_worked_friday : ℝ := 75
noncomputable def total_time_worked : ℝ := 4 * 60

theorem time_worked_on_thursday :
  time_worked_monday + time_worked_tuesday + time_worked_wednesday + time_worked_friday + 50 = total_time_worked :=
by
  sorry

end time_worked_on_thursday_l176_176213


namespace sum_of_three_largest_l176_176034

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176034


namespace factorization_sum_l176_176325

variable {a b c : ℤ}

theorem factorization_sum 
  (h1 : ∀ x : ℤ, x^2 + 17 * x + 52 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x^2 + 7 * x - 60 = (x + b) * (x - c)) : 
  a + b + c = 27 :=
sorry

end factorization_sum_l176_176325


namespace tree_heights_l176_176217

theorem tree_heights (h : ℕ) (ratio : 5 / 7 = (h - 20) / h) : h = 70 :=
sorry

end tree_heights_l176_176217


namespace parabola_problem_l176_176164

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l176_176164


namespace solve_equation_l176_176783

theorem solve_equation (x : ℝ) : 
  (x - 1)^2 + 2 * x * (x - 1) = 0 ↔ x = 1 ∨ x = 1 / 3 :=
by sorry

end solve_equation_l176_176783


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176886

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176886


namespace problem_equiv_proof_l176_176320

theorem problem_equiv_proof : ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h
  sorry

end problem_equiv_proof_l176_176320


namespace game_end_probability_l176_176492

noncomputable def prob_game_ends_on_fifth_toss : ℚ := 1 / 4

theorem game_end_probability :
  let coin_space := {0, 1} -- 0 for Tails, 1 for Heads
  let tosses := list coin_space -- a list of toss outcomes
  let fifth_toss_ends_game (xs : tosses) := (xs.length = 5) ∧ (xs.head = 1) ∧ (xs.take 4).count 1 = 1 ∨ (xs.take 4).count 0 = 1
  let favorable_outcomes := {xs | fifth_toss_ends_game xs}
  Finset.card favorable_outcomes = 8 →
  Finset.card (finset.univ : finset tosses) = 32 →
  (favorable_outcomes.card : ℚ) / (finset.univ.card : ℚ) = prob_game_ends_on_fifth_toss :=
by assumption

end game_end_probability_l176_176492


namespace no_intersection_points_of_polar_graphs_l176_176847

theorem no_intersection_points_of_polar_graphs :
  let c1_center := (3 / 2, 0)
  let r1 := 3 / 2
  let c2_center := (0, 3)
  let r2 := 3
  let distance_between_centers := Real.sqrt ((3 / 2 - 0) ^ 2 + (0 - 3) ^ 2)
  distance_between_centers > r1 + r2 :=
by
  sorry

end no_intersection_points_of_polar_graphs_l176_176847


namespace train_stoppage_time_l176_176523

theorem train_stoppage_time (speed_excluding_stoppages speed_including_stoppages : ℝ) 
(H1 : speed_excluding_stoppages = 54) 
(H2 : speed_including_stoppages = 36) : (18 / (54 / 60)) = 20 :=
by
  sorry

end train_stoppage_time_l176_176523


namespace six_digit_palindromes_l176_176000

theorem six_digit_palindromes : 
  let digits := {x : ℕ | x < 10} in
  let a_choices := {a : ℕ | 0 < a ∧ a < 10} in
  let b_choices := digits in
  let c_choices := digits in
  let d_choices := digits in
  (a_choices.card * b_choices.card * c_choices.card * d_choices.card = 9000) :=
by
  sorry

end six_digit_palindromes_l176_176000


namespace parabola_problem_l176_176133

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176133


namespace max_min_value_of_product_l176_176092

theorem max_min_value_of_product (x y : ℝ) (h : x ^ 2 + y ^ 2 = 1) :
  (1 + x * y) * (1 - x * y) ≤ 1 ∧ (1 + x * y) * (1 - x * y) ≥ 3 / 4 :=
by sorry

end max_min_value_of_product_l176_176092


namespace distance_travelled_by_gavril_l176_176710

noncomputable def smartphoneFullyDischargesInVideoWatching : ℝ := 3
noncomputable def smartphoneFullyDischargesInPlayingTetris : ℝ := 5
noncomputable def speedForHalfDistanceFirst : ℝ := 80
noncomputable def speedForHalfDistanceSecond : ℝ := 60
noncomputable def averageSpeed (distance speed time : ℝ) :=
  distance / time = speed

theorem distance_travelled_by_gavril : 
  ∃ S : ℝ, 
    (∃ t : ℝ, 
      (t / 2 / smartphoneFullyDischargesInVideoWatching + t / 2 / smartphoneFullyDischargesInPlayingTetris = 1) ∧ 
      (S / 2 / t / 2 = speedForHalfDistanceFirst) ∧
      (S / 2 / t / 2 = speedForHalfDistanceSecond)) ∧
     S = 257 := 
sorry

end distance_travelled_by_gavril_l176_176710


namespace find_g_l176_176792

noncomputable def f (x : ℝ) : ℝ := x^2

def is_solution (g : ℝ → ℝ) : Prop :=
  ∀ x, f (g x) = 9 * x^2 - 6 * x + 1

theorem find_g (g : ℝ → ℝ) : is_solution g → g = (λ x, 3 * x - 1) ∨ g = (λ x, -3 * x + 1) :=
by
  intro h
  sorry

end find_g_l176_176792


namespace compute_fraction_l176_176629

theorem compute_fraction :
  ((5 * 4) + 6) / 10 = 2.6 :=
by
  sorry

end compute_fraction_l176_176629


namespace ship_B_has_highest_rt_no_cars_l176_176767

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l176_176767


namespace find_x_l176_176929

theorem find_x (n x q p : ℕ) (h1 : n = q * x + 2) (h2 : 2 * n = p * x + 4) : x = 6 :=
sorry

end find_x_l176_176929


namespace polar_radius_tangent_to_circle_l176_176504

theorem polar_radius_tangent_to_circle :
  ∀ (r : ℝ), (r > 0) → 
    (∀ t : ℝ, let x := 8 * t^2 in let y := 8 * t in
     y^2 = 8 * x) →
    (∀ x y : ℝ, (y - x = -2) ∧ x^2 + y^2 = r^2 →
      r = ℚ.sqrt(2)) :=
by
  sorry


end polar_radius_tangent_to_circle_l176_176504


namespace probability_horizontal_distance_at_least_one_lemma_l176_176748

noncomputable def probability_horizontal_distance_at_least_one (T : set (ℝ × ℝ)) (side_length : ℝ) : ℝ :=
if (side_length = 2) ∧ (∀ x ∈ T, (0 ≤ x.1 ∧ x.1 ≤ 2) ∧ (0 ≤ x.2 ∧ x.2 ≤ 2)) then
  1 / 2
else
  0

theorem probability_horizontal_distance_at_least_one_lemma :
  let T := { p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2 } in
  probability_horizontal_distance_at_least_one T 2 = 1 / 2 :=
by {
  sorry
}

end probability_horizontal_distance_at_least_one_lemma_l176_176748


namespace power_add_one_eq_twice_l176_176330

theorem power_add_one_eq_twice (a b : ℕ) (h : 2^a = b) : 2^(a + 1) = 2 * b := by
  sorry

end power_add_one_eq_twice_l176_176330


namespace sum_of_largest_three_l176_176043

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176043


namespace sum_is_five_or_negative_five_l176_176724

theorem sum_is_five_or_negative_five (a b c d : ℤ) 
  (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d) 
  (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d)
  (h7 : a * b * c * d = 14) : 
  (a + b + c + d = 5) ∨ (a + b + c + d = -5) :=
by
  sorry

end sum_is_five_or_negative_five_l176_176724


namespace ab_distance_l176_176159

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l176_176159


namespace jacob_hours_l176_176778

theorem jacob_hours (J : ℕ) (H1 : ∃ (G P : ℕ),
    G = J - 6 ∧
    P = 2 * G - 4 ∧
    J + G + P = 50) : J = 18 :=
by
  sorry

end jacob_hours_l176_176778


namespace difference_in_payment_l176_176945

theorem difference_in_payment (joy_pencils : ℕ) (colleen_pencils : ℕ) (price_per_pencil : ℕ) (H1 : joy_pencils = 30) (H2 : colleen_pencils = 50) (H3 : price_per_pencil = 4) :
  (colleen_pencils * price_per_pencil) - (joy_pencils * price_per_pencil) = 80 :=
by
  rw [H1, H2, H3]
  simp
  norm_num
  sorry

end difference_in_payment_l176_176945


namespace arithmetic_sequence_sum_l176_176744

variable (a : ℕ → ℝ) (d : ℝ)

-- Condition: The sequence {a_n} is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
axiom a1 : a 1 = 2
axiom a2_a3_sum : a 2 + a 3 = 13

-- The theorem to be proved
theorem arithmetic_sequence_sum (h : is_arithmetic_sequence a d) : a (4) + a (5) + a (6) = 42 :=
sorry

end arithmetic_sequence_sum_l176_176744


namespace arianna_sleeping_hours_l176_176837

def hours_in_day : ℕ := 24
def hours_at_work : ℕ := 6
def hours_on_chores : ℕ := 5
def hours_sleeping : ℕ := hours_in_day - (hours_at_work + hours_on_chores)

theorem arianna_sleeping_hours : hours_sleeping = 13 := by
  sorry

end arianna_sleeping_hours_l176_176837


namespace width_decreased_by_28_6_percent_l176_176246

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l176_176246


namespace find_m_from_arithmetic_sequence_l176_176093

theorem find_m_from_arithmetic_sequence (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -4) (h2 : S m = 0) (h3 : S (m + 1) = 6) : m = 5 := by
  sorry

end find_m_from_arithmetic_sequence_l176_176093


namespace initial_eggs_proof_l176_176225

noncomputable def initial_eggs (total_cost : ℝ) (price_per_egg : ℝ) (leftover_eggs : ℝ) : ℝ :=
  let eggs_sold := total_cost / price_per_egg
  eggs_sold + leftover_eggs

theorem initial_eggs_proof : initial_eggs 5 0.20 5 = 30 := by
  sorry

end initial_eggs_proof_l176_176225


namespace greatest_possible_gcd_value_l176_176688

noncomputable def sn (n : ℕ) := n ^ 2
noncomputable def expression (n : ℕ) := 2 * sn n + 10 * n
noncomputable def gcd_value (a b : ℕ) := Nat.gcd a b 

theorem greatest_possible_gcd_value :
  ∃ n : ℕ, gcd_value (expression n) (n - 3) = 42 :=
sorry

end greatest_possible_gcd_value_l176_176688


namespace find_general_term_l176_176930

-- Definition of sequence sum condition
def seq_sum_condition (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (2/3) * a n + 1/3

-- Statement of the proof problem
theorem find_general_term (a S : ℕ → ℝ) 
  (h : seq_sum_condition a S) : 
  ∀ n, a n = (-2)^(n-1) := 
by
  sorry

end find_general_term_l176_176930


namespace width_decrease_percentage_l176_176249

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l176_176249


namespace mean_of_reciprocals_of_first_four_primes_l176_176892

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176892


namespace equation_has_real_root_l176_176421

theorem equation_has_real_root (x : ℝ) : (x^3 + 3 = 0) ↔ (x = -((3:ℝ)^(1/3))) :=
sorry

end equation_has_real_root_l176_176421


namespace solution_is_13_l176_176111

def marbles_in_jars : Prop :=
  let jar1 := (5, 3, 1)  -- (red, blue, green)
  let jar2 := (1, 5, 3)  -- (red, blue, green)
  let jar3 := (3, 1, 5)  -- (red, blue, green)
  let total_ways := 125 + 15 + 15 + 3 + 27 + 15
  let favorable_ways := 125
  let probability := favorable_ways / total_ways
  let simplified_probability := 5 / 8
  let m := 5
  let n := 8
  m + n = 13

theorem solution_is_13 : marbles_in_jars :=
by {
  sorry
}

end solution_is_13_l176_176111


namespace work_rate_l176_176412

theorem work_rate (x : ℕ) (hx : 2 * x = 30) : x = 15 := by
  -- We assume the prerequisite 2 * x = 30
  sorry

end work_rate_l176_176412


namespace parabola_problem_l176_176136

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176136


namespace length_of_place_mat_l176_176299

noncomputable def length_of_mat
  (R : ℝ)
  (w : ℝ)
  (n : ℕ)
  (θ : ℝ) : ℝ :=
  2 * R * Real.sin (θ / 2)

theorem length_of_place_mat :
  ∃ y : ℝ, y = length_of_mat 5 1 7 (360 / 7) := by
  use 4.38
  sorry

end length_of_place_mat_l176_176299


namespace ab_distance_l176_176160

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l176_176160


namespace division_and_multiplication_result_l176_176520

theorem division_and_multiplication_result :
  let num : ℝ := 6.5
  let divisor : ℝ := 6
  let multiplier : ℝ := 12
  num / divisor * multiplier = 13 :=
by
  sorry

end division_and_multiplication_result_l176_176520


namespace proof_problem_l176_176350

theorem proof_problem (x : ℤ) (h : (x - 34) / 10 = 2) : (x - 5) / 7 = 7 :=
  sorry

end proof_problem_l176_176350


namespace tourism_revenue_scientific_notation_l176_176277

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l176_176277


namespace larger_volume_of_rotated_rectangle_l176_176261

-- Definitions based on the conditions
def length : ℝ := 4
def width : ℝ := 3

-- Problem statement: Proving the volume of the larger geometric solid
theorem larger_volume_of_rotated_rectangle :
  max (Real.pi * (width ^ 2) * length) (Real.pi * (length ^ 2) * width) = 48 * Real.pi :=
by
  sorry

end larger_volume_of_rotated_rectangle_l176_176261


namespace ratio_unit_price_l176_176428

theorem ratio_unit_price
  (v : ℝ) (p : ℝ) (h_v : v > 0) (h_p : p > 0)
  (vol_A : ℝ := 1.25 * v)
  (price_A : ℝ := 0.85 * p) :
  (price_A / vol_A) / (p / v) = 17 / 25 :=
by
  sorry

end ratio_unit_price_l176_176428


namespace ratio_volume_surface_area_l176_176534

noncomputable def volume : ℕ := 10
noncomputable def surface_area : ℕ := 45

theorem ratio_volume_surface_area : volume / surface_area = 2 / 9 := by
  sorry

end ratio_volume_surface_area_l176_176534


namespace text_message_costs_equal_l176_176652

theorem text_message_costs_equal (x : ℝ) : 
  (0.25 * x + 9 = 0.40 * x) ∧ (0.25 * x + 9 = 0.20 * x + 12) → x = 60 :=
by 
  sorry

end text_message_costs_equal_l176_176652


namespace kim_distance_traveled_l176_176539

-- Definitions based on the problem conditions:
def infantry_column_length : ℝ := 1  -- The length of the infantry column in km.
def distance_inf_covered : ℝ := 2.4  -- Distance the infantrymen covered in km.

-- Theorem statement:
theorem kim_distance_traveled (column_length : ℝ) (inf_covered : ℝ) :
  column_length = 1 →
  inf_covered = 2.4 →
  ∃ d : ℝ, d = 3.6 :=
by
  sorry

end kim_distance_traveled_l176_176539


namespace parabola_problem_l176_176134

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176134


namespace rectangle_width_decrease_l176_176243

theorem rectangle_width_decrease (L W : ℝ) (hL : L > 0) (hW : W > 0) :
  let L' := 1.4 * L in
  let W' := (L * W) / L' in
  let percent_decrease := (1 - W' / W) * 100 in
  percent_decrease ≈ 28.57 :=
by
  let L' := 1.4 * L
  let W' := (L * W) / L'
  let percent_decrease := (1 - W' / W) * 100
  sorry

end rectangle_width_decrease_l176_176243


namespace total_stamps_in_collection_l176_176101

-- Definitions reflecting the problem conditions
def foreign_stamps : ℕ := 90
def old_stamps : ℕ := 60
def both_foreign_and_old_stamps : ℕ := 20
def neither_foreign_nor_old_stamps : ℕ := 70

-- The expected total number of stamps in the collection
def total_stamps : ℕ :=
  (foreign_stamps + old_stamps - both_foreign_and_old_stamps) + neither_foreign_nor_old_stamps

-- Statement to prove the total number of stamps is 200
theorem total_stamps_in_collection : total_stamps = 200 := by
  -- Proof omitted
  sorry

end total_stamps_in_collection_l176_176101


namespace ship_B_has_highest_rt_no_cars_l176_176769

def ship_percentage_with_no_cars (total_rt: ℕ) (percent_with_cars: ℕ) : ℕ :=
  total_rt - (percent_with_cars * total_rt) / 100

theorem ship_B_has_highest_rt_no_cars :
  let A_rt := 30
  let A_with_cars := 25
  let B_rt := 50
  let B_with_cars := 15
  let C_rt := 20
  let C_with_cars := 35
  let A_no_cars := ship_percentage_with_no_cars A_rt A_with_cars
  let B_no_cars := ship_percentage_with_no_cars B_rt B_with_cars
  let C_no_cars := ship_percentage_with_no_cars C_rt C_with_cars
  A_no_cars < B_no_cars ∧ C_no_cars < B_no_cars := by
  sorry

end ship_B_has_highest_rt_no_cars_l176_176769


namespace injective_functions_count_l176_176716

theorem injective_functions_count (m n : ℕ) (h_mn : m ≥ n) (h_n2 : n ≥ 2) :
  ∃ k, k = Nat.choose m n * (2^n - n - 1) :=
sorry

end injective_functions_count_l176_176716


namespace problem_1_problem_2_l176_176336

variables (α : ℝ) (h : Real.tan α = 3)

theorem problem_1 : (Real.sin α + 3 * Real.cos α) / (2 * Real.sin α + 5 * Real.cos α) = 6 / 11 :=
by
  -- Proof is skipped
  sorry

theorem problem_2 : Real.sin α * Real.sin α + Real.sin α * Real.cos α + 3 * Real.cos α * Real.cos α = 3 / 2 :=
by
  -- Proof is skipped
  sorry

end problem_1_problem_2_l176_176336


namespace arithmetic_mean_reciprocals_primes_l176_176870

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176870


namespace ab_distance_l176_176157

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l176_176157


namespace fraction_eval_l176_176908

theorem fraction_eval : 1 / (3 + 1 / (3 + 1 / (3 - 1 / 3))) = 27 / 89 :=
by
  sorry

end fraction_eval_l176_176908


namespace distance_between_cities_l176_176497

-- Definitions
def map_distance : ℝ := 120 -- Distance on the map in cm
def scale_factor : ℝ := 10  -- Scale factor in km per cm

-- Theorem statement
theorem distance_between_cities :
  map_distance * scale_factor = 1200 :=
by
  sorry

end distance_between_cities_l176_176497


namespace least_integer_solution_l176_176625

theorem least_integer_solution :
  ∃ x : ℤ, (abs (3 * x - 4) ≤ 25) ∧ (∀ y : ℤ, (abs (3 * y - 4) ≤ 25) → x ≤ y) :=
sorry

end least_integer_solution_l176_176625


namespace remainder_div2_l176_176993

   theorem remainder_div2 :
     ∀ z x : ℕ, (∃ k : ℕ, z = 4 * k) → (∃ n : ℕ, x = 2 * n) → (z + x + 4 + z + 3) % 2 = 1 :=
   by
     intros z x h1 h2
     sorry
   
end remainder_div2_l176_176993


namespace sum_of_consecutive_integers_product_336_l176_176255

theorem sum_of_consecutive_integers_product_336 :
  ∃ (x y : ℕ), x * (x + 1) = 336 ∧ (y - 1) * y * (y + 1) = 336 ∧ x + (x + 1) + (y - 1) + y + (y + 1) = 54 :=
by
  -- The formal proof would go here
  sorry

end sum_of_consecutive_integers_product_336_l176_176255


namespace find_question_mark_l176_176656

noncomputable def c1 : ℝ := (5568 / 87)^(1/3)
noncomputable def c2 : ℝ := (72 * 2)^(1/2)
noncomputable def sum_c1_c2 : ℝ := c1 + c2

theorem find_question_mark : sum_c1_c2 = 16 → 256 = 16^2 :=
by
  sorry

end find_question_mark_l176_176656


namespace problem1_problem2_l176_176451

-- Problem 1: Prove that the minimum value of f(x) is at least m for all x ∈ ℝ when k = 0
theorem problem1 (f : ℝ → ℝ) (m : ℝ) (h : ∀ x : ℝ, f x = Real.exp x - x) : m ≤ 1 := 
sorry

-- Problem 2: Prove that there exists exactly one zero of f(x) in the interval (k, 2k) when k > 1
theorem problem2 (f : ℝ → ℝ) (k : ℝ) (hk : k > 1) (h : ∀ x : ℝ, f x = Real.exp (x - k) - x) :
  ∃! (x : ℝ), x ∈ Set.Ioo k (2 * k) ∧ f x = 0 := 
sorry

end problem1_problem2_l176_176451


namespace snow_probability_l176_176967

theorem snow_probability :
  let p_first_four_days := 1 / 4
  let p_next_three_days := 1 / 3
  let p_no_snow_first_four := (3 / 4) ^ 4
  let p_no_snow_next_three := (2 / 3) ^ 3
  let p_no_snow_all_week := p_no_snow_first_four * p_no_snow_next_three
  let p_snow_at_least_once := 1 - p_no_snow_all_week
  in
  p_snow_at_least_once = 29 / 32 :=
sorry

end snow_probability_l176_176967


namespace isosceles_triangle_area_l176_176335

theorem isosceles_triangle_area (x : ℤ) (h1 : x > 2) (h2 : x < 4) 
  (h3 : ∃ (a b : ℤ), a = x ∧ b = 8 - 2 * x ∧ a = b) :
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end isosceles_triangle_area_l176_176335


namespace rate_per_kg_for_fruits_l176_176398

-- Definitions and conditions
def total_cost (rate_per_kg : ℝ) : ℝ := 8 * rate_per_kg + 9 * rate_per_kg

def total_paid : ℝ := 1190

theorem rate_per_kg_for_fruits : ∃ R : ℝ, total_cost R = total_paid ∧ R = 70 :=
by
  sorry

end rate_per_kg_for_fruits_l176_176398


namespace sum_of_coefficients_l176_176555

theorem sum_of_coefficients (a b c d : ℤ)
  (h1 : a + c = 2)
  (h2 : a * c + b + d = -3)
  (h3 : a * d + b * c = 7)
  (h4 : b * d = -6) :
  a + b + c + d = 7 :=
sorry

end sum_of_coefficients_l176_176555


namespace negation_proposition_false_l176_176613

variable (a : ℝ)

theorem negation_proposition_false : ¬ (∃ a : ℝ, a ≤ 2 ∧ a^2 ≥ 4) :=
sorry

end negation_proposition_false_l176_176613


namespace sandwich_bread_consumption_l176_176485

theorem sandwich_bread_consumption :
  ∀ (num_bread_per_sandwich : ℕ),
  (2 * num_bread_per_sandwich) + num_bread_per_sandwich = 6 →
  num_bread_per_sandwich = 2 := by
    intros num_bread_per_sandwich h
    sorry

end sandwich_bread_consumption_l176_176485


namespace number_solution_l176_176458

-- Statement based on identified conditions and answer
theorem number_solution (x : ℝ) (h : 0.10 * 0.30 * 0.50 * x = 90) : x = 6000 :=
by
  -- Skip the proof
  sorry

end number_solution_l176_176458


namespace solution_set_x_l176_176337

theorem solution_set_x (x : ℝ) : 
  (|x^2 - x - 2| + |1 / x| = |x^2 - x - 2 + 1 / x|) ↔ 
  (x ∈ {y : ℝ | -1 ≤ y ∧ y < 0} ∨ x ≥ 2) :=
sorry

end solution_set_x_l176_176337


namespace sum_of_three_largest_consecutive_numbers_l176_176029

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176029


namespace gcd_of_all_elements_in_B_is_2_l176_176367

-- Define the set B as the set of all numbers that can be represented as the sum of four consecutive positive integers.
def B : Set ℕ := {n | ∃ x : ℕ, n = 4 * x + 2 ∧ x > 0}

-- Translate the question to a Lean statement.
theorem gcd_of_all_elements_in_B_is_2 : ∀ n ∈ B, gcd n 2 = 2 := 
by
  sorry

end gcd_of_all_elements_in_B_is_2_l176_176367


namespace correct_vector_equation_l176_176820

variables {V : Type*} [AddCommGroup V]

variables (A B C: V)

theorem correct_vector_equation : 
  (A - B) - (B - C) = A - C :=
sorry

end correct_vector_equation_l176_176820


namespace sum_of_three_largest_consecutive_numbers_l176_176027

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176027


namespace relationship_between_mode_median_mean_l176_176916

def data_set : List ℕ := [20, 30, 40, 50, 60, 60, 70]

def mode : ℕ := 60 -- derived from the problem conditions
def median : ℕ := 50 -- derived from the problem conditions
def mean : ℚ := 330 / 7 -- derived from the problem conditions

theorem relationship_between_mode_median_mean :
  mode > median ∧ median > mean :=
by
  sorry

end relationship_between_mode_median_mean_l176_176916


namespace sum_of_three_largest_l176_176054

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176054


namespace ariel_age_l176_176537

theorem ariel_age : ∃ A : ℕ, (A + 15 = 4 * A) ∧ A = 5 :=
by
  -- Here we skip the proof
  sorry

end ariel_age_l176_176537


namespace sum_of_three_largest_l176_176051

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176051


namespace find_A_for_diamond_l176_176844

def diamond (A B : ℕ) : ℕ := 4 * A + 3 * B + 7

theorem find_A_for_diamond (A : ℕ) (h : diamond A 7 = 76) : A = 12 :=
by
  sorry

end find_A_for_diamond_l176_176844


namespace arrange_abc_l176_176471

open Real

noncomputable def a := log 4 / log 5
noncomputable def b := (log 3 / log 5)^2
noncomputable def c := 1 / (log 4 / log 5)

theorem arrange_abc : b < a ∧ a < c :=
by
  -- Mathematical translations as Lean proof obligations
  have a_lt_one : a < 1 := by sorry
  have c_gt_one : c > 1 := by sorry
  have b_lt_a : b < a := by sorry
  have a_lt_c : a < c := by sorry
  exact ⟨b_lt_a, a_lt_c⟩

end arrange_abc_l176_176471


namespace alice_bob_same_point_after_3_turns_l176_176674

noncomputable def alice_position (t : ℕ) : ℕ := (15 + 4 * t) % 15

noncomputable def bob_position (t : ℕ) : ℕ :=
  if t < 2 then 15
  else (15 - 11 * (t - 2)) % 15

theorem alice_bob_same_point_after_3_turns :
  ∃ t, t = 3 ∧ alice_position t = bob_position t :=
by
  exists 3
  simp only [alice_position, bob_position]
  norm_num
  -- Alice's position after 3 turns
  -- alice_position 3 = (15 + 4 * 3) % 15
  -- bob_position 3 = (15 - 11 * (3 - 2)) % 15
  -- Therefore,
  -- alice_position 3 = 12
  -- bob_position 3 = 12
  sorry

end alice_bob_same_point_after_3_turns_l176_176674


namespace remainder_of_n_l176_176911

theorem remainder_of_n (n : ℕ) (h1 : n^2 ≡ 9 [MOD 11]) (h2 : n^3 ≡ 5 [MOD 11]) : n ≡ 3 [MOD 11] :=
sorry

end remainder_of_n_l176_176911


namespace probability_of_first_three_heads_l176_176645

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l176_176645


namespace light_off_combinations_l176_176814

theorem light_off_combinations (k n m : ℕ) (h1 : k = 300) (h2 : n = 2020) (h3 : m = 1710):
  ∃ ways : ℕ, ways = nat.choose 1710 300 :=
by {
  have h := nat.choose_eq_formula m k,
  rw [h1, h2, h3],
  exact h
}

end light_off_combinations_l176_176814


namespace find_AC_find_area_l176_176933

theorem find_AC (BC : ℝ) (angleA : ℝ) (cosB : ℝ) 
(hBC : BC = Real.sqrt 7) (hAngleA : angleA = 60) (hCosB : cosB = Real.sqrt 6 / 3) :
  (AC : ℝ) → (hAC : AC = 2 * Real.sqrt 7 / 3) → Prop :=
by
  sorry

theorem find_area (BC AB : ℝ) (angleA : ℝ) 
(hBC : BC = Real.sqrt 7) (hAB : AB = 2) (hAngleA : angleA = 60) :
  (area : ℝ) → (hArea : area = 3 * Real.sqrt 3 / 2) → Prop :=
by
  sorry

end find_AC_find_area_l176_176933


namespace symmetric_point_l176_176715

theorem symmetric_point (x y : ℝ) (h1 : x < 0) (h2 : y > 0) (h3 : |x| = 2) (h4 : |y| = 3) : 
  (2, -3) = (-x, -y) :=
sorry

end symmetric_point_l176_176715


namespace parabola_distance_problem_l176_176149

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l176_176149


namespace fresh_pineapples_left_l176_176508

namespace PineappleStore

def initial := 86
def sold := 48
def rotten := 9

theorem fresh_pineapples_left (initial sold rotten : ℕ) (h_initial : initial = 86) (h_sold : sold = 48) (h_rotten : rotten = 9) :
  initial - sold - rotten = 29 :=
by sorry

end PineappleStore

end fresh_pineapples_left_l176_176508


namespace sum_of_largest_three_l176_176042

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176042


namespace systematic_sampling_correct_l176_176816

-- Definitions for the conditions
def total_products := 60
def group_count := 5
def products_per_group := total_products / group_count

-- systematic sampling condition: numbers are in increments of products_per_group
def systematic_sample (start : ℕ) (count : ℕ) : List ℕ := List.range' start products_per_group count

-- Given sequences
def A : List ℕ := [5, 10, 15, 20, 25]
def B : List ℕ := [5, 12, 31, 39, 57]
def C : List ℕ := [5, 17, 29, 41, 53]
def D : List ℕ := [5, 15, 25, 35, 45]

-- Correct solution defined
def correct_solution := [5, 17, 29, 41, 53]

-- Problem Statement
theorem systematic_sampling_correct :
  systematic_sample 5 group_count = correct_solution :=
by
  sorry

end systematic_sampling_correct_l176_176816


namespace discount_rate_l176_176531

theorem discount_rate (cost_shoes cost_socks cost_bag paid_price total_cost discount_amount amount_subject_to_discount discount_rate: ℝ)
  (h1 : cost_shoes = 74)
  (h2 : cost_socks = 2 * 2)
  (h3 : cost_bag = 42)
  (h4 : paid_price = 118)
  (h5 : total_cost = cost_shoes + cost_socks + cost_bag)
  (h6 : discount_amount = total_cost - paid_price)
  (h7 : amount_subject_to_discount = total_cost - 100)
  (h8 : discount_rate = (discount_amount / amount_subject_to_discount) * 100) :
  discount_rate = 10 := sorry

end discount_rate_l176_176531


namespace orchids_cut_l176_176817

-- defining the initial conditions
def initial_orchids : ℕ := 3
def final_orchids : ℕ := 7

-- the question: prove the number of orchids cut
theorem orchids_cut : final_orchids - initial_orchids = 4 := by
  sorry

end orchids_cut_l176_176817


namespace mean_of_reciprocals_first_four_primes_l176_176862

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176862


namespace sum_of_three_largest_consecutive_numbers_l176_176028

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176028


namespace sum_possible_x_values_in_isosceles_triangle_l176_176309

def isosceles_triangle (A B C : ℝ) : Prop :=
  A = B ∨ B = C ∨ C = A

def valid_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180

theorem sum_possible_x_values_in_isosceles_triangle :
  ∃ (x1 x2 x3 : ℝ), isosceles_triangle 80 x1 x1 ∧ isosceles_triangle x2 80 80 ∧ isosceles_triangle 80 x3 x3 ∧ 
  valid_triangle 80 x1 x1 ∧ valid_triangle x2 80 80 ∧ valid_triangle 80 x3 x3 ∧ 
  x1 + x2 + x3 = 150 :=
by
  sorry

end sum_possible_x_values_in_isosceles_triangle_l176_176309


namespace third_week_cases_l176_176758

-- Define the conditions as Lean definitions
def first_week_cases : ℕ := 5000
def second_week_cases : ℕ := first_week_cases / 2
def total_cases_after_three_weeks : ℕ := 9500

-- The statement to be proven
theorem third_week_cases :
  first_week_cases + second_week_cases + 2000 = total_cases_after_three_weeks :=
by
  sorry

end third_week_cases_l176_176758


namespace correct_statements_l176_176753

def f (x : ℝ) (b : ℝ) (c : ℝ) (d : ℝ) : ℝ := x^3 + b*x^2 + c*x + d
def f_prime (x : ℝ) (b : ℝ) (c : ℝ) : ℝ := 3*x^2 + 2*b*x + c

theorem correct_statements (b c d : ℝ) :
  (∃ x : ℝ, f x b c d = 4 ∧ f_prime x b c = 0) ∧
  (∃ x : ℝ, f x b c d = 0 ∧ f_prime x b c = 0) :=
by
  sorry

end correct_statements_l176_176753


namespace intersection_first_quadrant_l176_176576

theorem intersection_first_quadrant (a : ℝ) : 
  (∃ x y : ℝ, (ax + y = 4) ∧ (x - y = 2) ∧ (0 < x) ∧ (0 < y)) ↔ (-1 < a ∧ a < 2) :=
by
  sorry

end intersection_first_quadrant_l176_176576


namespace parabola_distance_l176_176138

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l176_176138


namespace parabola_problem_l176_176155

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176155


namespace fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l176_176825

variable (p : ℕ) [Fact (Nat.Prime p)]

theorem fermats_little_theorem_poly (X : ℤ) :
  (X + 1) ^ p = X ^ p + 1 := by
    sorry

theorem binom_coeff_divisible_by_prime {k : ℕ} (hkp : 1 ≤ k ∧ k < p) :
  p ∣ Nat.choose p k := by
    sorry

end fermats_little_theorem_poly_binom_coeff_divisible_by_prime_l176_176825


namespace rectangle_width_decrease_l176_176236

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l176_176236


namespace winnie_keeps_balloons_l176_176821

theorem winnie_keeps_balloons (red white green chartreuse friends total remainder : ℕ) (hRed : red = 17) (hWhite : white = 33) (hGreen : green = 65) (hChartreuse : chartreuse = 83) (hFriends : friends = 10) (hTotal : total = red + white + green + chartreuse) (hDiv : total % friends = remainder) : remainder = 8 :=
by
  have hTotal_eq : total = 198 := by
    sorry -- This would be the computation of 17 + 33 + 65 + 83
  have hRemainder_eq : 198 % 10 = remainder := by
    sorry -- This would involve the computation of the remainder
  exact sorry -- This would be the final proof that remainder = 8, tying all parts together

end winnie_keeps_balloons_l176_176821


namespace divideDogs_l176_176607

def waysToDivideDogs : ℕ :=
  (10.choose 2) * (8.choose 4)

theorem divideDogs : waysToDivideDogs = 3150 := by
  sorry

end divideDogs_l176_176607


namespace cost_of_adult_ticket_l176_176536

theorem cost_of_adult_ticket (x : ℕ) (total_persons : ℕ) (total_collected : ℕ) (adult_tickets : ℕ) (child_ticket_cost : ℕ) (amount_from_children : ℕ) :
  total_persons = 280 →
  total_collected = 14000 →
  adult_tickets = 200 →
  child_ticket_cost = 25 →
  amount_from_children = 2000 →
  200 * x + amount_from_children = total_collected →
  x = 60 :=
by
  intros h_persons h_total h_adults h_child_cost h_children_amount h_eq
  sorry

end cost_of_adult_ticket_l176_176536


namespace parabola_distance_l176_176187

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l176_176187


namespace Monica_books_next_year_l176_176374

-- Definitions for conditions
def books_last_year : ℕ := 25
def books_this_year (bl_year: ℕ) : ℕ := 3 * bl_year
def books_next_year (bt_year: ℕ) : ℕ := 3 * bt_year + 7

-- Theorem statement
theorem Monica_books_next_year : books_next_year (books_this_year books_last_year) = 232 :=
by
  sorry

end Monica_books_next_year_l176_176374


namespace scientific_notation_of_virus_diameter_l176_176234

theorem scientific_notation_of_virus_diameter :
  0.00000012 = 1.2 * 10 ^ (-7) :=
by
  sorry

end scientific_notation_of_virus_diameter_l176_176234


namespace AB_distance_l176_176144

open Real

noncomputable def focus_of_parabola : ℝ × ℝ := (1, 0)

def lies_on_parabola (A : ℝ × ℝ) : Prop :=
  (A.2 ^ 2) = 4 * A.1

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

def problem_statement (A B : ℝ × ℝ) :=
  A ≠ B →
  lies_on_parabola A →
  B = (3, 0) →
  focus_of_parabola = (1, 0) →
  distance A (1, 0) = 2 →
  distance (3, 0) (1, 0) = 2 →
  distance A B = 2 * sqrt 2

-- Now we would want to place the theorem that we need to prove:

theorem AB_distance (A B : ℝ × ℝ) :
  problem_statement A B :=
sorry

end AB_distance_l176_176144


namespace find_y_of_pentagon_l176_176483

def y_coordinate (y : ℝ) : Prop :=
  let area_ABDE := 12
  let area_BCD := 2 * (y - 3)
  let total_area := area_ABDE + area_BCD
  total_area = 35

theorem find_y_of_pentagon :
  ∃ y : ℝ, y_coordinate y ∧ y = 14.5 :=
by
  sorry

end find_y_of_pentagon_l176_176483


namespace gcd_of_B_is_2_l176_176368

def B : Set ℕ :=
    { m | ∃ n : ℕ, m = n + (n + 1) + (n + 2) + (n + 3) }

theorem gcd_of_B_is_2 :
    gcd (Set.toFinset B).val = 2 := 
begin
    -- Formalization of problem's given conditions and required proofs
    sorry
end

end gcd_of_B_is_2_l176_176368


namespace probability_of_snow_at_least_once_first_week_l176_176974

theorem probability_of_snow_at_least_once_first_week :
  let p_first4 := 1 / 4
  let p_next3 := 1 / 3
  let p_no_snow_first4 := (1 - p_first4) ^ 4
  let p_no_snow_next3 := (1 - p_next3) ^ 3
  let p_no_snow_week := p_no_snow_first4 * p_no_snow_next3
  1 - p_no_snow_week = 29 / 32 :=
by
  sorry

end probability_of_snow_at_least_once_first_week_l176_176974


namespace remainder_division_l176_176404

theorem remainder_division (x : ℤ) (hx : x % 82 = 5) : (x + 7) % 41 = 12 := 
by 
  sorry

end remainder_division_l176_176404


namespace solve_system_of_equations_l176_176490

def system_of_equations (x y z : ℤ) : Prop :=
  x^2 + 25*y + 19*z = -471 ∧
  y^2 + 23*x + 21*z = -397 ∧
  z^2 + 21*x + 21*y = -545

theorem solve_system_of_equations :
  system_of_equations (-22) (-23) (-20) :=
by
  unfold system_of_equations
  split
  -- Equation 1
  calc
    (-22:ℤ)^2 + 25*(-23) + 19*(-20)
      = 484 - 575 - 380 : by norm_num
      = -471 : by norm_num
  split
  -- Equation 2
  calc
    (-23:ℤ)^2 + 23*(-22) + 21*(-20)
      = 529 - 506 - 420 : by norm_num
      = -397 : by norm_num
  -- Equation 3
  calc
    (-20:ℤ)^2 + 21*(-22) + 21*(-23)
      = 400 - 462 - 483 : by norm_num
      = -545 : by norm_num

end solve_system_of_equations_l176_176490


namespace mean_of_reciprocals_of_first_four_primes_l176_176891

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176891


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176853

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176853


namespace highest_percentage_without_car_l176_176772

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l176_176772


namespace distance_AB_l176_176175

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l176_176175


namespace circle_radius_is_7_5_l176_176918

noncomputable def radius_of_circle (side_length : ℝ) : ℝ := sorry

theorem circle_radius_is_7_5 :
  radius_of_circle 12 = 7.5 := sorry

end circle_radius_is_7_5_l176_176918


namespace geometric_sequence_formula_l176_176334

noncomputable def a_n (q : ℝ) (n : ℕ) : ℝ := if n = 0 then 0 else 2^(n - 1)

theorem geometric_sequence_formula (q : ℝ) (S : ℕ → ℝ) (n : ℕ) (hn : n > 0) :
  a_n q n = 2^(n - 1) :=
sorry

end geometric_sequence_formula_l176_176334


namespace n_calculation_l176_176074

theorem n_calculation (n : ℕ) (hn : 0 < n)
  (h1 : Int.lcm 24 n = 72)
  (h2 : Int.lcm n 27 = 108) :
  n = 36 :=
sorry

end n_calculation_l176_176074


namespace cupcakes_per_package_l176_176372

theorem cupcakes_per_package
  (packages : ℕ) (total_left : ℕ) (cupcakes_eaten : ℕ) (initial_packages : ℕ) (cupcakes_per_package : ℕ)
  (h1 : initial_packages = 3)
  (h2 : cupcakes_eaten = 5)
  (h3 : total_left = 7)
  (h4 : packages = initial_packages * cupcakes_per_package - cupcakes_eaten)
  (h5 : packages = total_left) : 
  cupcakes_per_package = 4 := 
by
  sorry

end cupcakes_per_package_l176_176372


namespace probability_snow_at_least_once_first_week_l176_176973

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l176_176973


namespace distance_AB_l176_176198

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176198


namespace solve_system_l176_176489

variable (x y z : ℝ)

def equation1 : Prop := x^2 + 25 * y + 19 * z = -471
def equation2 : Prop := y^2 + 23 * x + 21 * z = -397
def equation3 : Prop := z^2 + 21 * x + 21 * y = -545

theorem solve_system : equation1 (-22) (-23) (-20) ∧ equation2 (-22) (-23) (-20) ∧ equation3 (-22) (-23) (-20) := by
  sorry

end solve_system_l176_176489


namespace parabola_distance_l176_176139

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l176_176139


namespace sum_of_three_largest_consecutive_numbers_l176_176023

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176023


namespace tourism_revenue_scientific_notation_l176_176279

-- Define the conditions given in the problem.
def total_tourism_revenue := 12.41 * 10^9

-- Prove the scientific notation of the total tourism revenue.
theorem tourism_revenue_scientific_notation :
  total_tourism_revenue = 1.241 * 10^9 :=
sorry

end tourism_revenue_scientific_notation_l176_176279


namespace batches_of_muffins_l176_176679

-- Definitions of the costs and savings
def cost_blueberries_6oz : ℝ := 5
def cost_raspberries_12oz : ℝ := 3
def ounces_per_batch : ℝ := 12
def total_savings : ℝ := 22

-- The proof problem is to show the number of batches Bill plans to make
theorem batches_of_muffins : (total_savings / (2 * cost_blueberries_6oz - cost_raspberries_12oz)) = 3 := 
by 
  sorry  -- Proof goes here

end batches_of_muffins_l176_176679


namespace find_range_a_l176_176712

def setA (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0
def setB (x a : ℝ) : Prop := |x - a| < 5
def real_line (x : ℝ) : Prop := True

theorem find_range_a (a : ℝ) :
  (∀ x, setA x ∨ setB x a) ↔ (-3:ℝ) ≤ a ∧ a ≤ 1 := by
sorry

end find_range_a_l176_176712


namespace sequence_solution_l176_176551

theorem sequence_solution (a : ℕ → ℝ) :
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m + n) = a m + a n - m * n) ∧ 
  (∀ m n : ℕ, 1 ≤ m → 1 ≤ n → a (m * n) = m^2 * a n + n^2 * a m + 2 * a m * a n) →
    (∀ n, a n = -n*(n-1)/2) ∨ (∀ n, a n = -n^2/2) :=
  by
  sorry

end sequence_solution_l176_176551


namespace sum_of_largest_three_l176_176046

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176046


namespace problem_l176_176752

noncomputable def f (A B x : ℝ) : ℝ := A * x^2 + B
noncomputable def g (A B x : ℝ) : ℝ := B * x^2 + A

theorem problem (A B x : ℝ) (h : A ≠ B) 
  (h1 : f A B (g A B x) - g A B (f A B x) = B^2 - A^2) : 
  A + B = 0 := 
  sorry

end problem_l176_176752


namespace total_amount_l176_176419

theorem total_amount (A B C : ℤ) (S : ℤ) (h_ratio : 100 * B = 45 * A ∧ 100 * C = 30 * A) (h_B : B = 6300) : S = 24500 := by
  sorry

end total_amount_l176_176419


namespace second_term_deposit_interest_rate_l176_176424

theorem second_term_deposit_interest_rate
  (initial_deposit : ℝ)
  (first_term_annual_rate : ℝ)
  (first_term_months : ℝ)
  (second_term_initial_value : ℝ)
  (second_term_final_value : ℝ)
  (s : ℝ)
  (first_term_value : initial_deposit * (1 + first_term_annual_rate / 100 / 12 * first_term_months) = second_term_initial_value)
  (second_term_value : second_term_initial_value * (1 + s / 100 / 12 * first_term_months) = second_term_final_value) :
  s = 11.36 :=
by
  sorry

end second_term_deposit_interest_rate_l176_176424


namespace farmer_turkeys_l176_176664

variable (n c : ℝ)

theorem farmer_turkeys (h1 : n * c = 60) (h2 : (c + 0.10) * (n - 15) = 54) : n = 75 :=
sorry

end farmer_turkeys_l176_176664


namespace find_n_cosine_l176_176706

theorem find_n_cosine :
  ∃ (n : ℕ), 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) ∧ n = 43 :=
by
  sorry

end find_n_cosine_l176_176706


namespace parabola_problem_l176_176154

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176154


namespace sum_of_largest_three_l176_176039

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176039


namespace copy_pages_l176_176106

theorem copy_pages
  (total_cents : ℕ)
  (cost_per_page : ℚ)
  (h_total : total_cents = 2000)
  (h_cost : cost_per_page = 2.5) :
  (total_cents / cost_per_page) = 800 :=
by
  -- This is where the proof would go
  sorry

end copy_pages_l176_176106


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176777

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176777


namespace limit_of_P_n_l176_176480

noncomputable def A_n (n : ℕ) : ℝ × ℝ := (n / (n + 1), (n + 1) / n)

noncomputable def B_n (n : ℕ) : ℝ × ℝ := ((n + 1) / n, n / (n + 1))

def M : ℝ × ℝ := (1, 1)

noncomputable def P_n (n : ℕ) : ℝ × ℝ := 
  let x_n := ((2 * n + 1) ^ 2) / (2 * n * (n + 1))
  let y_n := ((2 * n + 1) ^ 2) / (2 * n * (n + 1))
  (x_n, y_n)

theorem limit_of_P_n (a b : ℝ) :
  (a = 2) ∧ (b = 2) ↔
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, dist (P_n n) (2, 2) < ε := 
begin
  sorry
end

end limit_of_P_n_l176_176480


namespace probability_snow_at_least_once_first_week_l176_176972

noncomputable def probability_no_snow_first_4_days : ℚ := (3/4)^4
noncomputable def probability_no_snow_last_3_days : ℚ := (2/3)^3
noncomputable def probability_no_snow_entire_week : ℚ := probability_no_snow_first_4_days * probability_no_snow_last_3_days
noncomputable def probability_snow_at_least_once : ℚ := 1 - probability_no_snow_entire_week

theorem probability_snow_at_least_once_first_week : probability_snow_at_least_once = 125/128 :=
by
  unfold probability_no_snow_first_4_days
  unfold probability_no_snow_last_3_days
  unfold probability_no_snow_entire_week
  unfold probability_snow_at_least_once
  sorry

end probability_snow_at_least_once_first_week_l176_176972


namespace fx_leq_one_l176_176569

noncomputable def f (x : ℝ) : ℝ := (x + 1) / Real.exp x

theorem fx_leq_one : ∀ x : ℝ, f x ≤ 1 := by
  sorry

end fx_leq_one_l176_176569


namespace train_length_l176_176302

noncomputable def length_of_train (time_sec : ℕ) (speed_kmh : ℝ) : ℝ :=
  (speed_kmh * 1000 / 3600) * time_sec

theorem train_length (h_time : 21 = 21) (h_speed : 75.6 = 75.6) :
  length_of_train 21 75.6 = 441 :=
by
  sorry

end train_length_l176_176302


namespace median_isosceles_right_triangle_leg_length_l176_176310

theorem median_isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ)
  (H1 : m = 15)
  (H2 : m = h / 2)
  (H3 : 2 * x * x = h * h) : x = 15 * Real.sqrt 2 :=
by
  sorry

end median_isosceles_right_triangle_leg_length_l176_176310


namespace pears_equivalence_l176_176605

theorem pears_equivalence :
  (3 / 4 : ℚ) * 16 * (5 / 6) = 10 → 
  (2 / 5 : ℚ) * 20 * (5 / 6) = 20 / 3 := 
by
  intros h
  sorry

end pears_equivalence_l176_176605


namespace molecular_weight_CaO_is_56_l176_176626

def atomic_weight_Ca : ℕ := 40
def atomic_weight_O : ℕ := 16
def molecular_weight_CaO : ℕ := atomic_weight_Ca + atomic_weight_O

theorem molecular_weight_CaO_is_56 :
  molecular_weight_CaO = 56 := by
  sorry

end molecular_weight_CaO_is_56_l176_176626


namespace prob_three_heads_is_one_eighth_l176_176636

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l176_176636


namespace square_of_larger_number_is_1156_l176_176616

theorem square_of_larger_number_is_1156
  (x y : ℕ)
  (h1 : x + y = 60)
  (h2 : x - y = 8) :
  x^2 = 1156 := by
  sorry

end square_of_larger_number_is_1156_l176_176616


namespace sum_of_three_largest_of_consecutive_numbers_l176_176009

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176009


namespace pow_mod_cycle_l176_176270

theorem pow_mod_cycle (n : ℕ) : 3^250 % 13 = 3 := 
by
  sorry

end pow_mod_cycle_l176_176270


namespace six_digit_palindromes_count_l176_176003

theorem six_digit_palindromes_count : 
  let valid_digit_a := {x : ℕ | 1 ≤ x ∧ x ≤ 9}
  let valid_digit_bc := {x : ℕ | 0 ≤ x ∧ x ≤ 9}
  (set.card valid_digit_a * set.card valid_digit_bc * set.card valid_digit_bc) = 900 :=
by
  sorry

end six_digit_palindromes_count_l176_176003


namespace ac_bc_ratios_l176_176600

theorem ac_bc_ratios (A B C : ℝ) (m n : ℕ) (h : AC / BC = m / n) : 
  if m ≠ n then
    ((AC / AB = m / (m+n) ∧ BC / AB = n / (m+n)) ∨ 
     (AC / AB = m / (n-m) ∧ BC / AB = n / (n-m)))
  else 
    (AC / AB = 1 / 2 ∧ BC / AB = 1 / 2) := sorry

end ac_bc_ratios_l176_176600


namespace total_distance_hiked_east_l176_176541

-- Define Annika's constant rate of hiking
def constant_rate : ℝ := 10 -- minutes per kilometer

-- Define already hiked distance
def distance_hiked : ℝ := 2.75 -- kilometers

-- Define total available time to return
def total_time : ℝ := 45 -- minutes

-- Prove that the total distance hiked east is 4.5 kilometers
theorem total_distance_hiked_east : distance_hiked + (total_time - distance_hiked * constant_rate) / constant_rate = 4.5 :=
by
  sorry

end total_distance_hiked_east_l176_176541


namespace sequence_formula_general_formula_l176_176559

open BigOperators

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2 * n + 2

def S_n (n : ℕ) : ℕ :=
  n^2 + 3 * n + 1

theorem sequence_formula :
  ∀ n, a_n n =
    if n = 1 then 5 else 2 * n + 2 := by
  sorry

theorem general_formula (n : ℕ) :
  a_n n =
    if n = 1 then S_n 1 else S_n n - S_n (n - 1) := by
  sorry

end sequence_formula_general_formula_l176_176559


namespace fourth_student_guess_l176_176793

theorem fourth_student_guess :
    let guess1 := 100
    let guess2 := 8 * guess1
    let guess3 := guess2 - 200
    let avg := (guess1 + guess2 + guess3) / 3
    let guess4 := avg + 25
    guess4 = 525 := by
    intros guess1 guess2 guess3 avg guess4
    have h1 : guess1 = 100 := rfl
    have h2 : guess2 = 8 * guess1 := rfl
    have h3 : guess3 = guess2 - 200 := rfl
    have h4 : avg = (guess1 + guess2 + guess3) / 3 := rfl
    have h5 : guess4 = avg + 25 := rfl
    simp [h1, h2, h3, h4, h5]
    sorry

end fourth_student_guess_l176_176793


namespace interest_rate_l176_176979

theorem interest_rate (part1_amount part2_amount total_amount total_income : ℝ) (interest_rate1 interest_rate2 : ℝ) :
  part1_amount = 2000 →
  part2_amount = total_amount - part1_amount →
  interest_rate2 = 6 →
  total_income = (part1_amount * interest_rate1 / 100) + (part2_amount * interest_rate2 / 100) →
  total_amount = 2500 →
  total_income = 130 →
  interest_rate1 = 5 :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end interest_rate_l176_176979


namespace problem1_solution_problem2_solution_l176_176784

-- Statement for Problem 1
theorem problem1_solution (x : ℝ) : (1 / 2 * (x - 3) ^ 2 = 18) ↔ (x = 9 ∨ x = -3) :=
by sorry

-- Statement for Problem 2
theorem problem2_solution (x : ℝ) : (x ^ 2 + 6 * x = 5) ↔ (x = -3 + Real.sqrt 14 ∨ x = -3 - Real.sqrt 14) :=
by sorry

end problem1_solution_problem2_solution_l176_176784


namespace width_decreased_by_28_6_percent_l176_176247

theorem width_decreased_by_28_6_percent (L W : ℝ) (A : ℝ) 
    (hA : A = L * W) (hL : 1.4 * L * (W / 1.4) = A) :
    (1 - (W / 1.4 / W)) * 100 = 28.6 :=
by 
  sorry

end width_decreased_by_28_6_percent_l176_176247


namespace find_initial_number_l176_176907

-- Define the initial equation
def initial_equation (x : ℤ) : Prop := x - 12 * 3 * 2 = 9938

-- Prove that the initial number x is equal to 10010 given initial_equation
theorem find_initial_number (x : ℤ) (h : initial_equation x) : x = 10010 :=
sorry

end find_initial_number_l176_176907


namespace Sara_lunch_bill_l176_176981

theorem Sara_lunch_bill :
  let hotdog := 5.36
  let salad := 5.10
  let drink := 2.50
  let side_item := 3.75
  hotdog + salad + drink + side_item = 16.71 :=
by
  sorry

end Sara_lunch_bill_l176_176981


namespace distance_AB_l176_176123

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l176_176123


namespace ryan_finished_time_l176_176311

theorem ryan_finished_time :
  ∀ (r : ℝ), (∀ {t : ℝ}, 0 ≤ t ≤ 9 → r * t = 1/2) ∧
             (∀ {t : ℝ}, 9 ≤ t ≤ 10 → r * t = 7/8) ∧
             (∀ {t₀ t₁ : ℝ}, 9 ≤ t₀ ∧ t₀ ≤ t₁ ∧ t₁ ≤ 10 → ((7/8) - (1/2)) / (t₁ - t₀) = r) →
             (r * 20/60 = 1/8) →
             (10 + 20/60 = 10.3333) :=
sorry

end ryan_finished_time_l176_176311


namespace youseff_blocks_l176_176403

theorem youseff_blocks (x : ℕ) 
  (H1 : (1 : ℚ) * x = (1/3 : ℚ) * x + 8) : 
  x = 12 := 
sorry

end youseff_blocks_l176_176403


namespace remainder_of_large_power_l176_176328

def powerMod (base exp mod_ : ℕ) : ℕ := (base ^ exp) % mod_

theorem remainder_of_large_power :
  powerMod 2 (2^(2^2)) 500 = 536 :=
sorry

end remainder_of_large_power_l176_176328


namespace sum_eq_two_l176_176223

theorem sum_eq_two (x y : ℝ) (hx : x^3 - 3 * x^2 + 5 * x = 1) (hy : y^3 - 3 * y^2 + 5 * y = 5) : x + y = 2 := 
sorry

end sum_eq_two_l176_176223


namespace silverware_probability_l176_176358

/-- In a drawer containing 8 forks, 8 spoons, and 8 knives, the probability of randomly
selecting six pieces of silverware and retrieving exactly two forks, two spoons, and two knives
is equal to 2744 / 16825. -/
theorem silverware_probability :
  let total_pieces := 24
  let select_count := 6
  let forks := 8
  let spoons := 8
  let knives := 8
  let desired_forks := 2
  let desired_spoons := 2
  let desired_knives := 2
  let prob := (choose forks desired_forks) * (choose spoons desired_spoons) * (choose knives desired_knives) / ((choose total_pieces select_count) : ℚ)
  in prob = 2744 / 16825 :=
by
  sorry

end silverware_probability_l176_176358


namespace product_of_numbers_l176_176708

theorem product_of_numbers :
  ∃ (a b c : ℚ), a + b + c = 30 ∧
                 a = 2 * (b + c) ∧
                 b = 5 * c ∧
                 a + c = 22 ∧
                 a * b * c = 2500 / 9 :=
by
  sorry

end product_of_numbers_l176_176708


namespace parabola_problem_l176_176165

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l176_176165


namespace largest_n_satisfying_inequality_l176_176326

theorem largest_n_satisfying_inequality :
  ∃ n : ℕ, n ≥ 1 ∧ n^(6033) < 2011^(2011) ∧ ∀ m : ℕ, m > n → m^(6033) ≥ 2011^(2011) :=
sorry

end largest_n_satisfying_inequality_l176_176326


namespace distance_AB_l176_176172

theorem distance_AB (A B F : ℝ × ℝ) (h : |A - F| = |B - F|) : 
  A ∈ {p : ℝ × ℝ | p.2^2 = 4 * p.1} → B = (3, 0) → F = (1, 0) → |A - B| = 2 * Real.sqrt 2 :=
by
  intro hA hB hF
  sorry

end distance_AB_l176_176172


namespace difference_between_numbers_l176_176811

theorem difference_between_numbers (x y : ℕ) 
  (h1 : x + y = 20000) 
  (h2 : y = 7 * x) : y - x = 15000 :=
by
  sorry

end difference_between_numbers_l176_176811


namespace crocus_bulbs_count_l176_176822

theorem crocus_bulbs_count (C D : ℕ) 
  (h1 : C + D = 55) 
  (h2 : 0.35 * (C : ℝ) + 0.65 * (D : ℝ) = 29.15) :
  C = 22 :=
sorry

end crocus_bulbs_count_l176_176822


namespace hypotenuse_length_l176_176939

theorem hypotenuse_length (a b c : ℝ) (h₀ : a^2 + b^2 + c^2 = 1800) (h₁ : c^2 = a^2 + b^2) : c = 30 :=
by
  sorry

end hypotenuse_length_l176_176939


namespace value_of_fg_neg_one_l176_176473

def f (x : ℝ) : ℝ := x - 2

def g (x : ℝ) : ℝ := x^2 + 4 * x + 3

theorem value_of_fg_neg_one : f (g (-1)) = -2 :=
by
  sorry

end value_of_fg_neg_one_l176_176473


namespace total_chickens_l176_176397

theorem total_chickens (ducks geese : ℕ) (hens roosters chickens: ℕ) :
  ducks = 45 → geese = 28 →
  hens = ducks - 13 → roosters = geese + 9 →
  chickens = hens + roosters →
  chickens = 69 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_chickens_l176_176397


namespace total_prime_dates_in_non_leap_year_l176_176581

def prime_dates_in_non_leap_year (days_in_months : List (Nat × Nat)) : Nat :=
  let prime_numbers := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
  days_in_months.foldl 
    (λ acc (month, days) => 
      acc + (prime_numbers.filter (λ day => day ≤ days)).length) 
    0

def month_days : List (Nat × Nat) :=
  [(2, 28), (3, 31), (5, 31), (7, 31), (11,30)]

theorem total_prime_dates_in_non_leap_year : prime_dates_in_non_leap_year month_days = 52 :=
  sorry

end total_prime_dates_in_non_leap_year_l176_176581


namespace circle_center_radius_l176_176670

theorem circle_center_radius (x y : ℝ) :
  (x^2 + y^2 + 4 * x - 6 * y = 11) →
  ∃ (h k r : ℝ), h = -2 ∧ k = 3 ∧ r = 2 * Real.sqrt 6 ∧
  (x+h)^2 + (y+k)^2 = r^2 :=
by
  sorry

end circle_center_radius_l176_176670


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176884

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176884


namespace sum_of_three_largest_of_consecutive_numbers_l176_176014

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176014


namespace black_pieces_more_than_white_l176_176617

theorem black_pieces_more_than_white (B W : ℕ) 
  (h₁ : (B - 1) * 7 = 9 * W)
  (h₂ : B * 5 = 7 * (W - 1)) :
  B - W = 7 :=
sorry

end black_pieces_more_than_white_l176_176617


namespace observations_count_l176_176252

theorem observations_count (n : ℕ) 
  (original_mean : ℚ) (wrong_value_corrected : ℚ) (corrected_mean : ℚ)
  (h1 : original_mean = 36)
  (h2 : wrong_value_corrected = 1)
  (h3 : corrected_mean = 36.02) :
  n = 50 :=
by
  sorry

end observations_count_l176_176252


namespace max_profit_at_80_l176_176831

-- Definitions based on conditions
def cost_price : ℝ := 40
def functional_relationship (x : ℝ) : ℝ := -x + 140
def profit (x : ℝ) : ℝ := (x - cost_price) * functional_relationship x

-- Statement to prove that maximum profit is achieved at x = 80
theorem max_profit_at_80 : (40 ≤ 80) → (80 ≤ 80) → profit 80 = 2400 := by
  sorry

end max_profit_at_80_l176_176831


namespace quadratic_root_one_is_minus_one_l176_176736

theorem quadratic_root_one_is_minus_one (m : ℝ) (h : ∃ x : ℝ, x = -1 ∧ m * x^2 + x - m^2 + 1 = 0) : m = 1 :=
by
  sorry

end quadratic_root_one_is_minus_one_l176_176736


namespace total_cantaloupes_l176_176064

def Fred_grew_38 : ℕ := 38
def Tim_grew_44 : ℕ := 44

theorem total_cantaloupes : Fred_grew_38 + Tim_grew_44 = 82 := by
  sorry

end total_cantaloupes_l176_176064


namespace sum_of_three_largest_of_consecutive_numbers_l176_176013

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176013


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176876

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176876


namespace determine_m_j_l176_176502

open Matrix

noncomputable def B (m : ℚ) : Matrix (Fin 2) (Fin 2) ℚ := ![![4, 5], ![3, m]]

theorem determine_m_j (m j : ℚ) (h : (B m)⁻¹ = j • B m) : m = -4 ∧ j = (1 : ℚ) / 31 := by
  -- The proof is omitted.
  sorry

end determine_m_j_l176_176502


namespace range_of_a_l176_176611

theorem range_of_a {a : ℝ} : 
  (∃ x : ℝ, (1 / 2 < x ∧ x < 3) ∧ (x ^ 2 - a * x + 1 = 0)) ↔ (2 ≤ a ∧ a < 10 / 3) :=
by
  sorry

end range_of_a_l176_176611


namespace sum_of_three_largest_l176_176049

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176049


namespace colleen_paid_more_l176_176947

def pencils_joy : ℕ := 30
def pencils_colleen : ℕ := 50
def cost_per_pencil : ℕ := 4

theorem colleen_paid_more : 
  (pencils_colleen - pencils_joy) * cost_per_pencil = 80 :=
by
  sorry

end colleen_paid_more_l176_176947


namespace pen_cost_l176_176690

def pencil_cost : ℝ := 1.60
def elizabeth_money : ℝ := 20.00
def num_pencils : ℕ := 5
def num_pens : ℕ := 6

theorem pen_cost (pen_cost : ℝ) : 
  elizabeth_money - (num_pencils * pencil_cost) = num_pens * pen_cost → 
  pen_cost = 2 :=
by 
  sorry

end pen_cost_l176_176690


namespace solution_set_of_inequality_l176_176506

theorem solution_set_of_inequality (x : ℝ) : 3 * x - 7 ≤ 2 → x ≤ 3 :=
by
  intro h
  sorry

end solution_set_of_inequality_l176_176506


namespace max_value_expr_l176_176955

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l176_176955


namespace probability_of_first_three_heads_l176_176643

noncomputable def problem : ℚ := 
  if (prob_heads = 1 / 2 ∧ independent_flips ∧ first_three_all_heads) then 1 / 8 else 0

theorem probability_of_first_three_heads :
  (∀ (coin : Type), (fair_coin : coin → ℚ) (flip : ℕ → coin) (indep : ∀ (n : ℕ), independent (λ _, flip n) (λ _, flip (n + 1))), 
  fair_coin(heads) = 1 / 2 ∧
  (∀ n, indep n) ∧
  let prob_heads := fair_coin(heads) in
  let first_three_all_heads := prob_heads * prob_heads * prob_heads
  ) → problem = 1 / 8 :=
by
  sorry

end probability_of_first_three_heads_l176_176643


namespace probability_C_l176_176919

noncomputable def probability_A : ℝ := 0.3
noncomputable def probability_B : ℝ := 0.2

axiom mutually_exclusive (A B : set ω) : P(A ∩ B) = 0
axiom complementary (A C : set ω) : P(A ∪ C) = 1 ∧ P(A ∩ C) = 0
axiom prob_A_union_B (A B : set ω) : P(A ∪ B) = 0.5
axiom prob_B (B : set ω) : P(B) = 0.2

theorem probability_C (A B C : set ω) (h_me : mutually_exclusive A B) (h_compl : complementary A C) (h_PAuB : prob_A_union_B A B) (h_PB : prob_B B) : P(C) = 0.7 := 
sorry

end probability_C_l176_176919


namespace parallel_lines_no_intersection_l176_176433

theorem parallel_lines_no_intersection (k : ℝ) :
  (∀ t s : ℝ, 
    ∃ (a b : ℝ), (a, b) = (1, -3) + t • (2, 5) ∧ (a, b) = (-4, 2) + s • (3, k)) → 
  k = 15 / 2 :=
by
  sorry

end parallel_lines_no_intersection_l176_176433


namespace sum_of_largest_three_consecutive_numbers_l176_176056

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176056


namespace rectangle_width_decrease_l176_176237

theorem rectangle_width_decrease (L W : ℝ) (h : L * W = A) (h_new_length : 1.40 * L = L') 
    (h_area_unchanged : L' * W' = L * W) : 
    (W - W') / W = 0.285714 :=
begin
  sorry
end

end rectangle_width_decrease_l176_176237


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176871

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176871


namespace sum_of_three_largest_l176_176036

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176036


namespace point_below_line_l176_176090

theorem point_below_line {a : ℝ} (h : 2 * a - 3 < 3) : a < 3 :=
by {
  sorry
}

end point_below_line_l176_176090


namespace find_A_l176_176069

axiom power_eq_A (A : ℝ) (x y : ℝ) : 2^x = A ∧ 7^(2*y) = A
axiom reciprocal_sum_eq_2 (x y : ℝ) : (1/x) + (1/y) = 2

theorem find_A (A x y : ℝ) : 
  (2^x = A) ∧ (7^(2*y) = A) ∧ ((1/x) + (1/y) = 2) -> A = 7*Real.sqrt 2 :=
by 
  sorry

end find_A_l176_176069


namespace range_of_x_range_of_a_l176_176718

variable (a x : ℝ)

-- Define proposition p: x^2 - 3ax + 2a^2 < 0
def p (a x : ℝ) : Prop := x^2 - 3 * a * x + 2 * a^2 < 0

-- Define proposition q: x^2 - x - 6 ≤ 0 and x^2 + 2x - 8 > 0
def q (x : ℝ) : Prop := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

-- First theorem: Prove the range of x when a = 2 and p ∨ q is true
theorem range_of_x (h : p 2 x ∨ q x) : 2 < x ∧ x < 4 := 
by sorry

-- Second theorem: Prove the range of a when ¬p is necessary but not sufficient for ¬q
theorem range_of_a (h : ∀ x, q x → p a x) : 3/2 ≤ a ∧ a ≤ 2 := 
by sorry

end range_of_x_range_of_a_l176_176718


namespace xyz_product_condition_l176_176509

theorem xyz_product_condition (x y z : ℝ) (h : x^2 + y^2 = x * y * (z + 1 / z)) : 
  x = y * z ∨ y = x * z :=
sorry

end xyz_product_condition_l176_176509


namespace rectangle_width_decrease_proof_l176_176241

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l176_176241


namespace car_distance_ratio_l176_176824

theorem car_distance_ratio (speed_A time_A speed_B time_B : ℕ) 
  (hA : speed_A = 70) (hTA : time_A = 10) 
  (hB : speed_B = 35) (hTB : time_B = 10) : 
  (speed_A * time_A) / gcd (speed_A * time_A) (speed_B * time_B) = 2 :=
by
  sorry

end car_distance_ratio_l176_176824


namespace six_digit_palindrome_count_l176_176002

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end six_digit_palindrome_count_l176_176002


namespace gcd_a2011_a2012_l176_176406

open Int

noncomputable def sequence : ℕ → ℤ
| 0       := 5
| 1       := 8
| (n + 2) := sequence (n + 1) + 3 * sequence n

theorem gcd_a2011_a2012 : gcd (sequence 2011) (sequence 2012) = 1 :=
by
  -- Proof skipped
  sorry

end gcd_a2011_a2012_l176_176406


namespace system_of_equations_solution_l176_176547

theorem system_of_equations_solution (x y z : ℝ) (hx : x = Real.exp (Real.log y))
(hy : y = Real.exp (Real.log z)) (hz : z = Real.exp (Real.log x)) : x = y ∧ y = z ∧ z = x ∧ x = Real.exp 1 :=
by
  sorry

end system_of_equations_solution_l176_176547


namespace range_of_a_l176_176727

def A := {x : ℝ | |x| >= 3}
def B (a : ℝ) := {x : ℝ | x >= a}

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a <= -3 :=
sorry

end range_of_a_l176_176727


namespace parabola_distance_l176_176189

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l176_176189


namespace arithmetic_sequence_a6_l176_176583

-- Definitions representing the conditions
def arithmetic_sequence (a_n : ℕ → ℝ) : Prop :=
∀ n m : ℕ, a_n (n + m) = a_n n + a_n m + n

def sum_of_first_n_terms (S : ℕ → ℝ) (a_n : ℕ → ℝ) : Prop :=
∀ n : ℕ, S n = (n / 2) * (2 * a_n 1 + (n - 1) * (a_n 2 - a_n 1))

theorem arithmetic_sequence_a6 (S : ℕ → ℝ) (a_n : ℕ → ℝ) 
  (h_seq : arithmetic_sequence a_n)
  (h_sum : sum_of_first_n_terms S a_n)
  (h_cond : S 9 - S 2 = 35) : 
  a_n 6 = 5 :=
by
  sorry

end arithmetic_sequence_a6_l176_176583


namespace fourth_student_guess_l176_176795

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let total := first_guess + second_guess + third_guess
  let average := total / 3
  let fourth_guess := average + 25
  fourth_guess = 525 :=
by
  sorry

end fourth_student_guess_l176_176795


namespace parabola_distance_l176_176167

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l176_176167


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176895

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176895


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176877

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176877


namespace Teena_speed_is_55_l176_176608

def Teena_speed (Roe_speed T : ℝ) (initial_gap final_gap time : ℝ) : Prop :=
  Roe_speed * time + initial_gap + final_gap = T * time

theorem Teena_speed_is_55 :
  Teena_speed 40 55 7.5 15 1.5 :=
by 
  sorry

end Teena_speed_is_55_l176_176608


namespace tommy_house_price_l176_176621

variable (P : ℝ)

theorem tommy_house_price 
  (h1 : 1.25 * P = 125000) : 
  P = 100000 :=
by
  sorry

end tommy_house_price_l176_176621


namespace sum_of_base_areas_eq_5_l176_176812

-- Define the surface area, lateral area, and the sum of the areas of the two base faces.
def surface_area : ℝ := 30
def lateral_area : ℝ := 25
def sum_base_areas : ℝ := surface_area - lateral_area

-- The theorem statement.
theorem sum_of_base_areas_eq_5 : sum_base_areas = 5 := 
by 
  sorry

end sum_of_base_areas_eq_5_l176_176812


namespace ship_with_highest_no_car_round_trip_percentage_l176_176763

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l176_176763


namespace Robert_salary_loss_l176_176224

-- Define the conditions as hypotheses
variable (S : ℝ) (decrease_percent increase_percent : ℝ)
variable (decrease_percent_eq : decrease_percent = 0.6)
variable (increase_percent_eq : increase_percent = 0.6)

-- Define the problem statement to prove that Robert loses 36% of his salary.
theorem Robert_salary_loss (S : ℝ) (decrease_percent increase_percent : ℝ) 
  (decrease_percent_eq : decrease_percent = 0.6) 
  (increase_percent_eq : increase_percent = 0.6) :
  let new_salary := S * (1 - decrease_percent)
  let increased_salary := new_salary * (1 + increase_percent)
  let loss_percentage := (S - increased_salary) / S * 100 
  loss_percentage = 36 := 
by
  sorry

end Robert_salary_loss_l176_176224


namespace greatest_n_for_xy_le_0_l176_176591

theorem greatest_n_for_xy_le_0
  (a b : ℕ) (coprime_ab : Nat.gcd a b = 1) :
  ∃ n : ℕ, (n = a * b ∧ ∃ x y : ℤ, n = a * x + b * y ∧ x * y ≤ 0) :=
sorry

end greatest_n_for_xy_le_0_l176_176591


namespace guy_has_sixty_cents_l176_176112

-- Definitions for the problem conditions
def lance_has (lance_cents : ℕ) : Prop := lance_cents = 70
def margaret_has (margaret_cents : ℕ) : Prop := margaret_cents = 75
def bill_has (bill_cents : ℕ) : Prop := bill_cents = 60
def total_has (total_cents : ℕ) : Prop := total_cents = 265

-- Problem Statement in Lean format
theorem guy_has_sixty_cents (lance_cents margaret_cents bill_cents total_cents guy_cents : ℕ) 
    (h_lance : lance_has lance_cents)
    (h_margaret : margaret_has margaret_cents)
    (h_bill : bill_has bill_cents)
    (h_total : total_has total_cents) :
    guy_cents = total_cents - (lance_cents + margaret_cents + bill_cents) → guy_cents = 60 :=
by
  intros h
  simp [lance_has, margaret_has, bill_has, total_has] at *
  rw [h_lance, h_margaret, h_bill, h_total] at h
  exact h

end guy_has_sixty_cents_l176_176112


namespace plane_through_point_contains_line_l176_176846

-- Definitions from conditions
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)

def passes_through (p : Point) (plane : Point → Prop) : Prop :=
  plane p

def contains_line (line : ℝ → Point) (plane : Point → Prop) : Prop :=
  ∀ t, plane (line t)

def line_eq (t : ℝ) : Point :=
  ⟨4 * t + 2, -6 * t - 3, 2 * t + 4⟩

def plane_eq (A B C D : ℝ) (p : Point) : Prop :=
  A * p.x + B * p.y + C * p.z + D = 0

theorem plane_through_point_contains_line :
  ∃ (A B C D : ℝ), 1 < A ∧ gcd (abs A) (gcd (abs B) (gcd (abs C) (abs D))) = 1 ∧
  passes_through ⟨1, 2, -3⟩ (plane_eq A B C D) ∧
  contains_line line_eq (plane_eq A B C D) ∧ 
  (∃ (k : ℝ), 3 * k = A ∧ k = 1 / 3 ∧ B = k * 1 ∧ C = k * (-3) ∧ D = k * 2) :=
sorry

end plane_through_point_contains_line_l176_176846


namespace pythagorean_triplet_unique_solution_l176_176552

-- Define the conditions given in the problem
def is_solution (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∧
  Nat.gcd a (Nat.gcd b c) = 1 ∧
  2000 ≤ a ∧ a ≤ 3000 ∧
  2000 ≤ b ∧ b ≤ 3000 ∧
  2000 ≤ c ∧ c ≤ 3000

-- Prove that the only set of integers (a, b, c) meeting the conditions
-- equals the specific tuple (2100, 2059, 2941)
theorem pythagorean_triplet_unique_solution : 
  ∀ a b c : ℕ, is_solution a b c ↔ (a = 2100 ∧ b = 2059 ∧ c = 2941) :=
by
  sorry

end pythagorean_triplet_unique_solution_l176_176552


namespace complex_fraction_value_l176_176842

theorem complex_fraction_value :
  1 + 1 / (1 + 1 / (1 + 1 / (1 + 2))) = 7 / 4 :=
sorry

end complex_fraction_value_l176_176842


namespace compute_fg_l176_176733

def f (x : ℤ) : ℤ := x * x
def g (x : ℤ) : ℤ := 3 * x + 4

theorem compute_fg : f (g (-3)) = 25 := by
  sorry

end compute_fg_l176_176733


namespace probability_player_A_wins_first_B_wins_second_l176_176218

theorem probability_player_A_wins_first_B_wins_second :
  (1 / 2) * (4 / 5) * (2 / 3) + (1 / 2) * (1 / 3) * (2 / 3) = 17 / 45 :=
by
  sorry

end probability_player_A_wins_first_B_wins_second_l176_176218


namespace maximum_rectangle_area_l176_176268

theorem maximum_rectangle_area (P : ℝ) (hP : P = 36) :
  ∃ (A : ℝ), A = (P / 4) * (P / 4) :=
by
  use 81
  sorry

end maximum_rectangle_area_l176_176268


namespace mystical_swamp_l176_176099

/-- 
In a mystical swamp, there are two species of talking amphibians: toads, whose statements are always true, and frogs, whose statements are always false. 
Five amphibians: Adam, Ben, Cara, Dan, and Eva make the following statements:
Adam: "Eva and I are different species."
Ben: "Cara is a frog."
Cara: "Dan is a frog."
Dan: "Of the five of us, at least three are toads."
Eva: "Adam is a toad."
Given these statements, prove that the number of frogs is 3.
-/
theorem mystical_swamp :
  (∀ α β : Prop, α ∨ ¬β) ∧ -- Adam's statement: "Eva and I are different species."
  (Cara = "frog") ∧          -- Ben's statement: "Cara is a frog."
  (Dan = "frog") ∧         -- Cara's statement: "Dan is a frog."
  (∃ t, t = nat → t ≥ 3) ∧ -- Dan's statement: "Of the five of us, at least three are toads."
  (Adam = "toad")               -- Eva's statement: "Adam is a toad."
  → num_frogs = 3 := sorry       -- Number of frogs is 3.

end mystical_swamp_l176_176099


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176872

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176872


namespace fraction_of_unoccupied_chairs_is_two_fifths_l176_176462

noncomputable def fraction_unoccupied_chairs (total_chairs : ℕ) (chair_capacity : ℕ) (attended_board_members : ℕ) : ℚ :=
  let total_capacity := total_chairs * chair_capacity
  let total_board_members := total_capacity
  let unoccupied_members := total_board_members - attended_board_members
  let unoccupied_chairs := unoccupied_members / chair_capacity
  unoccupied_chairs / total_chairs

theorem fraction_of_unoccupied_chairs_is_two_fifths :
  fraction_unoccupied_chairs 40 2 48 = 2 / 5 :=
by
  sorry

end fraction_of_unoccupied_chairs_is_two_fifths_l176_176462


namespace g_50_unique_l176_176754

namespace Proof

-- Define the function g and the condition it should satisfy
variable (g : ℕ → ℕ)
variable (h : ∀ (a b : ℕ), 3 * g (a^2 + b^2) = g a * g b + 2 * (g a + g b))

theorem g_50_unique : ∃ (m t : ℕ), m * t = 0 := by
  -- Existence of m and t fulfilling the condition
  -- Placeholder for the proof
  sorry

end Proof

end g_50_unique_l176_176754


namespace geometric_progression_a5_value_l176_176077

theorem geometric_progression_a5_value
  (a : ℕ → ℝ)
  (h_geom : ∃ r : ℝ, ∀ n, a (n + 1) = a n * r)
  (h_roots : ∃ x y, x^2 - 5*x + 4 = 0 ∧ y^2 - 5*y + 4 = 0 ∧ x = a 3 ∧ y = a 7) :
  a 5 = 2 :=
by
  sorry

end geometric_progression_a5_value_l176_176077


namespace max_knights_on_island_l176_176599

theorem max_knights_on_island :
  ∃ n x, (n * (n - 1) = 90) ∧ (x * (10 - x) = 24) ∧ (x ≤ n) ∧ (∀ y, y * (10 - y) = 24 → y ≤ x) := sorry

end max_knights_on_island_l176_176599


namespace Steve_pencils_left_l176_176229

-- Define the initial number of boxes and pencils per box
def boxes := 2
def pencils_per_box := 12
def initial_pencils := boxes * pencils_per_box

-- Define the number of pencils given to Lauren and the additional pencils given to Matt
def pencils_to_Lauren := 6
def diff_Lauren_Matt := 3
def pencils_to_Matt := pencils_to_Lauren + diff_Lauren_Matt

-- Calculate the total pencils given away
def pencils_given_away := pencils_to_Lauren + pencils_to_Matt

-- Number of pencils left with Steve
def pencils_left := initial_pencils - pencils_given_away

-- The statement to prove
theorem Steve_pencils_left : pencils_left = 9 := by
  sorry

end Steve_pencils_left_l176_176229


namespace line_circle_intersection_common_points_l176_176078

noncomputable def radius (d : ℝ) := d / 2

theorem line_circle_intersection_common_points 
  (diameter : ℝ) (distance_from_center_to_line : ℝ) 
  (h_dlt_r : distance_from_center_to_line < radius diameter) :
  ∃ common_points : ℕ, common_points = 2 :=
by
  sorry

end line_circle_intersection_common_points_l176_176078


namespace sum_of_three_largest_of_consecutive_numbers_l176_176011

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176011


namespace sum_of_three_largest_consecutive_numbers_l176_176024

theorem sum_of_three_largest_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  -- proof using Lean tactics to be added here
  sorry

end sum_of_three_largest_consecutive_numbers_l176_176024


namespace no_integer_solutions_l176_176730

theorem no_integer_solutions :
  ¬ ∃ (x y : ℤ), x^4 + y^2 = 6 * y - 3 :=
by
  sorry

end no_integer_solutions_l176_176730


namespace length_of_DC_l176_176584

theorem length_of_DC (AB : ℝ) (angle_ADB : ℝ) (sin_A : ℝ) (sin_C : ℝ)
  (h1 : AB = 30) (h2 : angle_ADB = pi / 2) (h3 : sin_A = 3 / 5) (h4 : sin_C = 1 / 4) :
  ∃ DC : ℝ, DC = 18 * Real.sqrt 15 :=
by
  sorry

end length_of_DC_l176_176584


namespace geom_seq_root_product_l176_176104

theorem geom_seq_root_product
  (a : ℕ → ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * a 1)
  (h_root1 : 3 * (a 1)^2 + 7 * a 1 - 9 = 0)
  (h_root10 : 3 * (a 10)^2 + 7 * a 10 - 9 = 0) :
  a 4 * a 7 = -3 := 
by
  sorry

end geom_seq_root_product_l176_176104


namespace solution_l176_176437

noncomputable def problem (x : ℝ) : Prop :=
  0 < x ∧ (1/2 * (4 * x^2 - 1) = (x^2 - 50 * x - 20) * (x^2 + 25 * x + 10))

theorem solution (x : ℝ) (h : problem x) : x = 26 + Real.sqrt 677 :=
by
  sorry

end solution_l176_176437


namespace ball_bounce_height_l176_176408

theorem ball_bounce_height :
  ∃ k : ℕ, (500 * (2 / 3:ℝ)^k < 10) ∧ (∀ m : ℕ, m < k → ¬(500 * (2 / 3:ℝ)^m < 10)) :=
sorry

end ball_bounce_height_l176_176408


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176899

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176899


namespace three_heads_in_a_row_l176_176651

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l176_176651


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176898

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176898


namespace sum_of_integers_satisfying_l176_176802

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l176_176802


namespace f_4_1981_eq_l176_176798

def f : ℕ → ℕ → ℕ
| 0, y     => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem f_4_1981_eq : f 4 1981 = 2 ^ 16 - 3 := sorry

end f_4_1981_eq_l176_176798


namespace bucket_capacity_l176_176529

theorem bucket_capacity (x : ℝ) (h1 : 24 * x = 36 * 9) : x = 13.5 :=
by 
  sorry

end bucket_capacity_l176_176529


namespace initial_cost_of_article_correct_l176_176423

noncomputable def initial_cost_of_article (final_cost : ℝ) : ℝ :=
  final_cost / (0.75 * 0.85 * 1.10 * 1.05)

theorem initial_cost_of_article_correct (final_cost : ℝ) (h : final_cost = 1226.25) :
  initial_cost_of_article final_cost = 1843.75 :=
by
  rw [h]
  norm_num
  rw [initial_cost_of_article]
  simp [initial_cost_of_article]
  norm_num
  sorry

end initial_cost_of_article_correct_l176_176423


namespace possible_polynomials_l176_176786

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l176_176786


namespace solve_inequality_l176_176677

theorem solve_inequality (x : ℝ) : (1 + x) / 3 < x / 2 → x > 2 := 
by {
  sorry
}

end solve_inequality_l176_176677


namespace two_digit_number_digits_34_l176_176418

theorem two_digit_number_digits_34 :
  let x := (34 / 99.0)
  ∃ n : ℕ, n = 34 ∧ (48 * x - 48 * 0.34 = 0.2) := 
by
  let x := (34.0 / 99.0)
  use 34
  sorry

end two_digit_number_digits_34_l176_176418


namespace find_triples_of_positive_integers_l176_176693

theorem find_triples_of_positive_integers (p q n : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hn_pos : 0 < n) 
  (equation : p * (p + 3) + q * (q + 3) = n * (n + 3)) : 
  (p = 3 ∧ q = 2 ∧ n = 4) :=
sorry

end find_triples_of_positive_integers_l176_176693


namespace Ralph_TV_hours_l176_176380

theorem Ralph_TV_hours :
  let hoursWeekdays := 4 * 5,
  let hoursWeekends := 6 * 2,
  let totalHours := hoursWeekdays + hoursWeekends
  in totalHours = 32 := 
by
  sorry

end Ralph_TV_hours_l176_176380


namespace prob_three_heads_is_one_eighth_l176_176635

-- Define the probability of heads in a fair coin
def fair_coin_prob_heads : ℚ := 1 / 2

-- Define the probability of three consecutive heads
def prob_three_heads (p : ℚ) : ℚ := p * p * p

-- Theorem statement
theorem prob_three_heads_is_one_eighth :
  prob_three_heads fair_coin_prob_heads = 1 / 8 := 
sorry

end prob_three_heads_is_one_eighth_l176_176635


namespace half_sum_squares_ge_product_l176_176219

theorem half_sum_squares_ge_product (x y : ℝ) : 
  1 / 2 * (x^2 + y^2) ≥ x * y := 
by 
  sorry

end half_sum_squares_ge_product_l176_176219


namespace finite_parabolas_do_not_cover_plane_l176_176468

theorem finite_parabolas_do_not_cover_plane (parabolas : Finset (ℝ → ℝ)) :
  ¬ (∀ x y : ℝ, ∃ p ∈ parabolas, y < p x) :=
by sorry

end finite_parabolas_do_not_cover_plane_l176_176468


namespace range_of_eccentricity_l176_176922

theorem range_of_eccentricity
  (a b c : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c^2 = a^2 - b^2)
  (h4 : c^2 - b^2 + a * c < 0) :
  0 < c / a ∧ c / a < 1 / 2 :=
sorry

end range_of_eccentricity_l176_176922


namespace correct_product_exists_l176_176098

variable (a b : ℕ)

theorem correct_product_exists
  (h1 : a < 100)
  (h2 : 10 * (a % 10) + a / 10 = 14)
  (h3 : 14 * b = 182) : a * b = 533 := sorry

end correct_product_exists_l176_176098


namespace mean_of_reciprocals_of_first_four_primes_l176_176893

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176893


namespace hoseok_basketballs_l176_176815

theorem hoseok_basketballs (v s b : ℕ) (h₁ : v = 40) (h₂ : s = v + 18) (h₃ : b = s - 23) : b = 35 := by
  sorry

end hoseok_basketballs_l176_176815


namespace range_of_x_for_f_ln_x_gt_f_1_l176_176075

noncomputable def is_even (f : ℝ → ℝ) := ∀ x, f x = f (-x)

noncomputable def is_decreasing_on_nonneg (f : ℝ → ℝ) := ∀ ⦃x y⦄, 0 ≤ x → x ≤ y → f y ≤ f x

theorem range_of_x_for_f_ln_x_gt_f_1
  (f : ℝ → ℝ)
  (hf_even : is_even f)
  (hf_dec : is_decreasing_on_nonneg f)
  (hf_condition : ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e) :
  ∀ x : ℝ, f (Real.log x) > f 1 ↔ e⁻¹ < x ∧ x < e := sorry

end range_of_x_for_f_ln_x_gt_f_1_l176_176075


namespace marcus_percentage_of_team_points_l176_176479

theorem marcus_percentage_of_team_points
  (three_point_goals : ℕ)
  (two_point_goals : ℕ)
  (team_points : ℕ)
  (h1 : three_point_goals = 5)
  (h2 : two_point_goals = 10)
  (h3 : team_points = 70) :
  ((three_point_goals * 3 + two_point_goals * 2) / team_points : ℚ) * 100 = 50 := by
sorry

end marcus_percentage_of_team_points_l176_176479


namespace solution_l176_176296

/-- Original cost price, original selling price, and daily sales at original price -/
def original_cost : ℝ := 80
def original_price : ℝ := 120
def daily_sales : ℝ := 20

/-- Conditions: price reduction per unit and increased sales -/
def price_reduction_per_piece (x : ℝ) : ℝ := x
def daily_sales_increase (x : ℝ) : ℝ := 2 * x

/-- Profit per piece given price reduction x -/
def profit_per_piece (x : ℝ) : ℝ := 40 - x

/-- Daily sales volume given price reduction x -/
def sales_volume (x : ℝ) : ℝ := 20 + 2 * x

/-- Daily profit as a function of price reduction x -/
def daily_profit (x : ℝ) : ℝ := (40 - x) * (20 + 2 * x)

/-- Problem: find price reduction x for a daily profit of 1200 yuan -/
def price_reduction_for_target_profit (target_profit : ℝ) : ℝ := 
  if (solver : ∃ x : ℝ, (40 - x) * (20 + 2 * x) = target_profit) then
    classical.some solver
  else 
    0

/-- Check if a daily profit of 1800 yuan can be achieved -/
def can_achieve_daily_profit_1800 : Prop :=
  ¬ ∃ x : ℝ, (40 - x) * (20 + 2 * x) = 1800

/--Theorem stating the solution to the problem -/
theorem solution : can_achieve_daily_profit_1800 := by
  sorry

end solution_l176_176296


namespace meal_order_probability_l176_176399

noncomputable def probability_two_people_get_correct_meal 
  (total_people : ℕ) (pasta_orders : ℕ) (salad_orders : ℕ) 
  (favorable_outcomes : ℕ) (total_outcomes : ℕ) : ℚ := 
  favorable_outcomes / total_outcomes

theorem meal_order_probability :
  let total_people := 12
  let pasta_orders := 5
  let salad_orders := 7
  let favorable_outcomes := 157410
  let total_outcomes := 12.factorial
  probability_two_people_get_correct_meal total_people pasta_orders salad_orders favorable_outcomes total_outcomes = 
  (157410 : ℚ) / (479001600 : ℚ) :=
by 
  have total_outcomes_fact : total_outcomes = 12.factorial := rfl
  rw [total_outcomes_fact, show 12.factorial = 479001600, by norm_num]
  exact rfl

end meal_order_probability_l176_176399


namespace goldfish_to_pretzels_ratio_l176_176373

theorem goldfish_to_pretzels_ratio :
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := kids * items_per_baggie
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  ratio = 4 :=
by
  let pretzels := 64
  let suckers := 32
  let kids := 16
  let items_per_baggie := 22
  let total_items := 16 * 22 -- or kids * items_per_baggie for clarity
  let goldfish := total_items - pretzels - suckers
  let ratio := goldfish / pretzels
  show ratio = 4
  · sorry

end goldfish_to_pretzels_ratio_l176_176373


namespace distance_AB_l176_176178

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176178


namespace salt_added_correct_l176_176420

theorem salt_added_correct (x : ℝ)
  (hx : x = 119.99999999999996)
  (initial_salt : ℝ := 0.20 * x)
  (evaporation_volume : ℝ := x - (1/4) * x)
  (additional_water : ℝ := 8)
  (final_volume : ℝ := evaporation_volume + additional_water)
  (final_concentration : ℝ := 1 / 3)
  (final_salt : ℝ := final_concentration * final_volume)
  (salt_added : ℝ := final_salt - initial_salt) :
  salt_added = 8.67 :=
sorry

end salt_added_correct_l176_176420


namespace infinite_solutions_eq_a_l176_176571

variable (a x y: ℝ)

-- Define the two equations
def eq1 : Prop := a * x + y - 1 = 0
def eq2 : Prop := 4 * x + a * y - 2 = 0

theorem infinite_solutions_eq_a (h : ∃ x y, eq1 a x y ∧ eq2 a x y) :
  a = 2 := 
sorry

end infinite_solutions_eq_a_l176_176571


namespace root_division_simplification_l176_176850

theorem root_division_simplification (a : ℝ) (h1 : a = (7 : ℝ)^(1/4)) (h2 : a = (7 : ℝ)^(1/7)) :
  ((7 : ℝ)^(1/4) / (7 : ℝ)^(1/7)) = (7 : ℝ)^(3/28) :=
sorry

end root_division_simplification_l176_176850


namespace tan_half_angle_rational_iff_sin_cos_rational_l176_176478

noncomputable def is_not_odd_multiple_pi (α : ℝ) : Prop :=
  ∃ n : ℤ, α ≠ (2 * n + 1) * π

theorem tan_half_angle_rational_iff_sin_cos_rational 
  (α : ℝ) (h : is_not_odd_multiple_pi α) :
  (∃ t : ℚ, t = Real.tan (α / 2)) ↔ (∃ r s : ℚ, r = Real.cos α ∧ s = Real.sin α) :=
by sorry

end tan_half_angle_rational_iff_sin_cos_rational_l176_176478


namespace solve_for_x_l176_176228

theorem solve_for_x :
  (16^x * 16^x * 16^x * 4^(3 * x) = 64^(4 * x)) → x = 0 := by
  sorry

end solve_for_x_l176_176228


namespace symmetric_circle_eq_l176_176797

theorem symmetric_circle_eq (x y : ℝ) :
  (x^2 + y^2 - 4 * x = 0) ↔ (x^2 + y^2 - 4 * y = 0) :=
sorry

end symmetric_circle_eq_l176_176797


namespace no_infinite_seq_pos_int_l176_176405

theorem no_infinite_seq_pos_int : 
  ¬∃ (a : ℕ → ℕ), 
  (∀ n : ℕ, 0 < a n) ∧ 
  ∀ n : ℕ, a (n+1) ^ 2 ≥ 2 * a n * a (n+2) :=
by
  sorry

end no_infinite_seq_pos_int_l176_176405


namespace distance_AB_l176_176124

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l176_176124


namespace solve_inequality_system_l176_176386

theorem solve_inequality_system (x : ℝ) :
  (x + 2 > -1) ∧ (x - 5 < 3 * (x - 1)) ↔ (x > -1) :=
by
  sorry

end solve_inequality_system_l176_176386


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176880

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176880


namespace triangle_value_l176_176322

-- Define the operation \(\triangle\)
def triangle (m n p q : ℕ) : ℕ := (m * m) * p * q / n

-- Define the problem statement
theorem triangle_value : triangle 5 6 9 4 = 150 := by
  sorry

end triangle_value_l176_176322


namespace quadratic_expression_value_l176_176722

theorem quadratic_expression_value (a : ℝ) (h : a^2 - 2 * a - 3 = 0) : a^2 - 2 * a + 1 = 4 :=
by 
  -- Proof omitted for clarity in this part
  sorry 

end quadratic_expression_value_l176_176722


namespace determine_x_l176_176324

theorem determine_x (y : ℚ) (h : y = (36 + 249 / 999) / 100) :
  ∃ x : ℕ, y = x / 99900 ∧ x = 36189 :=
by
  sorry

end determine_x_l176_176324


namespace chickens_and_rabbits_l176_176355

theorem chickens_and_rabbits (c r : ℕ) (h1 : c + r = 15) (h2 : 2 * c + 4 * r = 40) : c = 10 ∧ r = 5 :=
sorry

end chickens_and_rabbits_l176_176355


namespace residue_11_pow_1234_mod_19_l176_176628

theorem residue_11_pow_1234_mod_19 : 
  (11 ^ 1234) % 19 = 11 := 
by
  sorry

end residue_11_pow_1234_mod_19_l176_176628


namespace find_n_cos_eq_l176_176702

theorem find_n_cos_eq : ∃ (n : ℕ), (0 ≤ n ∧ n ≤ 180) ∧ (n = 43) ∧ (cos (n * real.pi / 180) = cos (317 * real.pi / 180)) :=
by
  use 43
  split
  { split
    { exact dec_trivial }
    { exact dec_trivial } }
  split
  { exact rfl }
  { sorry }

end find_n_cos_eq_l176_176702


namespace sum_of_solutions_eq_one_l176_176804

theorem sum_of_solutions_eq_one :
  let solutions := {x : ℤ | x^2 = 272 + x} in
  ∑ x in solutions, x = 1 := by
  sorry

end sum_of_solutions_eq_one_l176_176804


namespace distance_AB_l176_176180

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176180


namespace max_value_is_one_l176_176954

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l176_176954


namespace trivia_game_probability_l176_176673

/-
Theorem: The probability that the player wins the trivia game by guessing at least three 
out of four questions correctly is 13/256.
Conditions:
1. Each game consists of 4 multiple-choice questions.
2. Each question has 4 choices.
3. A player wins if they answer at least 3 out of 4 questions correctly.
4. The player guesses on each question.

We need to prove that the probability of winning is 13/256.
-/
theorem trivia_game_probability : 
  let p : ℚ := 1 / 4 in
  let prob_all_correct : ℚ := p^4 in
  let prob_three_correct : ℚ := (p^3) * (3 / 4) * 4 in
  prob_all_correct + prob_three_correct = 13 / 256 :=
by
  sorry

end trivia_game_probability_l176_176673


namespace expected_heads_for_100_coins_l176_176961

noncomputable def expected_heads (n : ℕ) (p_heads : ℚ) : ℚ :=
  n * p_heads

theorem expected_heads_for_100_coins :
  expected_heads 100 (15 / 16) = 93.75 :=
by
  sorry

end expected_heads_for_100_coins_l176_176961


namespace number_division_l176_176965

theorem number_division (n q r d : ℕ) (h1 : d = 18) (h2 : q = 11) (h3 : r = 1) (h4 : n = (d * q) + r) : n = 199 := 
by 
  sorry

end number_division_l176_176965


namespace probability_of_correct_digit_in_two_attempts_l176_176493

theorem probability_of_correct_digit_in_two_attempts : 
  let num_possible_digits := 10
  let num_attempts := 2
  let total_possible_outcomes := num_possible_digits * (num_possible_digits - 1)
  let total_favorable_outcomes := (num_possible_digits - 1) + (num_possible_digits - 1)
  let probability := (total_favorable_outcomes : ℚ) / (total_possible_outcomes : ℚ)
  probability = (1 / 5 : ℚ) :=
by
  sorry

end probability_of_correct_digit_in_two_attempts_l176_176493


namespace fraction_of_grid_covered_by_triangle_l176_176377

-- Define the vertices of the triangle
def A : ℝ × ℝ := (2, 5)
def B : ℝ × ℝ := (7, 2)
def C : ℝ × ℝ := (6, 6)

-- Define the dimensions of the grid
def gridLength : ℝ := 8
def gridWidth : ℝ := 6

-- Calculate the area of the triangle using the Shoelace theorem
def triangleArea : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

-- Calculate the area of the grid
def gridArea : ℝ := gridLength * gridWidth

-- Calculate the fraction of the grid covered by the triangle
def fractionCovered : ℚ := triangleArea / gridArea

-- Proof statement
theorem fraction_of_grid_covered_by_triangle : fractionCovered = 17 / 96 := by
  sorry

end fraction_of_grid_covered_by_triangle_l176_176377


namespace distance_AB_l176_176181

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176181


namespace average_speed_before_increase_l176_176306

-- Definitions for the conditions
def t_before := 12   -- Travel time before the speed increase in hours
def t_after := 10    -- Travel time after the speed increase in hours
def speed_diff := 20 -- Speed difference between before and after in km/h

-- Variable for the speed before increase
variable (s_before : ℕ) -- Average speed before the speed increase in km/h

-- Definitions for the speeds
def s_after := s_before + speed_diff -- Average speed after the speed increase in km/h

-- Equations derived from the problem conditions
def dist_eqn_before := s_before * t_before
def dist_eqn_after := s_after * t_after

-- The proof problem stated in Lean
theorem average_speed_before_increase : dist_eqn_before = dist_eqn_after → s_before = 100 := by
  sorry

end average_speed_before_increase_l176_176306


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176856

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176856


namespace soap_box_missing_dimension_l176_176667

theorem soap_box_missing_dimension
  (x : ℕ) -- The missing dimension of the soap box
  (Volume_carton : ℕ := 25 * 48 * 60)
  (Volume_soap_box : ℕ := 8 * x * 5)
  (Max_soap_boxes : ℕ := 300)
  (condition : Max_soap_boxes * Volume_soap_box ≤ Volume_carton) :
  x ≤ 6 := by
sorry

end soap_box_missing_dimension_l176_176667


namespace M_geq_N_l176_176913

variable (x y : ℝ)
def M : ℝ := x^2 + y^2 + 1
def N : ℝ := x + y + x * y

theorem M_geq_N (x y : ℝ) : M x y ≥ N x y :=
by
sorry

end M_geq_N_l176_176913


namespace intersection_of_M_and_N_is_N_l176_176920

def M := {x : ℝ | x ≥ -1}
def N := {y : ℝ | y ≥ 0}

theorem intersection_of_M_and_N_is_N : M ∩ N = N := sorry

end intersection_of_M_and_N_is_N_l176_176920


namespace arithmetic_mean_reciprocals_primes_l176_176869

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176869


namespace polygon_P_properties_l176_176115

-- Definitions of points A, B, and C
def A : (ℝ × ℝ × ℝ) := (0, 0, 0)
def B : (ℝ × ℝ × ℝ) := (1, 0.5, 0)
def C : (ℝ × ℝ × ℝ) := (0, 0.5, 1)

-- Condition of cube intersection and plane containing A, B, and C
def is_corner_of_cube (p : ℝ × ℝ × ℝ) : Prop :=
  p = A

def are_midpoints_of_cube_edges (p₁ p₂ : ℝ × ℝ × ℝ) : Prop :=
  (p₁ = B ∧ p₂ = C)

-- Polygon P resulting from the plane containing A, B, and C intersecting the cube
def num_sides_of_polygon (p : ℝ × ℝ × ℝ) : ℕ := 5 -- Given the polygon is a pentagon

-- Area of triangle ABC
noncomputable def area_triangle_ABC : ℝ :=
  (1/2) * (Real.sqrt 1.5)

-- Area of polygon P
noncomputable def area_polygon_P : ℝ :=
  (11/6) * area_triangle_ABC

-- Theorem stating that polygon P has 5 sides and the ratio of its area to the area of triangle ABC is 11/6
theorem polygon_P_properties (A B C : (ℝ × ℝ × ℝ))
  (hA : is_corner_of_cube A) (hB : are_midpoints_of_cube_edges B C) :
  num_sides_of_polygon A = 5 ∧ area_polygon_P / area_triangle_ABC = (11/6) :=
by sorry

end polygon_P_properties_l176_176115


namespace rectangle_width_decrease_proof_l176_176240

def rectangle_width_decreased_percentage (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : ℝ := 
  28.57

theorem rectangle_width_decrease_proof (L W : ℝ) (h : L * W = (1.4 * L) * (W / 1.4)) : 
  rectangle_width_decreased_percentage L W h = 28.57 := 
by
  sorry

end rectangle_width_decrease_proof_l176_176240


namespace hyperbola_asymptotes_l176_176989

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := y^2 - (x^2 / 4) = 1

-- The statement to prove: The equation of the asymptotes of the hyperbola is as follows
theorem hyperbola_asymptotes :
  (∀ x y : ℝ, hyperbola_eq x y → (y = (1/2) * x ∨ y = -(1/2) * x)) :=
sorry

end hyperbola_asymptotes_l176_176989


namespace estimate_students_above_110_l176_176292

noncomputable def students : ℕ := 50
noncomputable def mean : ℝ := 100
noncomputable def std_dev : ℝ := 10
noncomputable def normal_dist (x : ℝ) : ℝ := (Real.exp (-((x - mean)^2 / (2 * std_dev^2)))) / (std_dev * (Real.sqrt (2 * Real.pi)))
noncomputable def prob_90_to_100 : ℝ := 0.3
noncomputable def desired_prob : ℝ := 0.5 - (prob_90_to_100 + prob_90_to_100)
noncomputable def estimated_students_above_110 : ℝ := desired_prob * students

theorem estimate_students_above_110 : estimated_students_above_110 = 10 := by 
  sorry

end estimate_students_above_110_l176_176292


namespace feasible_test_for_rhombus_l176_176511

def is_rhombus (paper : Type) : Prop :=
  true -- Placeholder for the actual definition of a rhombus

def method_A (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the four internal angles are equal"
  true

def method_B (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the two diagonals are equal"
  true

def method_C (paper : Type) : Prop :=
  -- Placeholder for the condition "Measure if the distance from the intersection of the two diagonals to the four vertices is equal"
  true

def method_D (paper : Type) : Prop :=
  -- Placeholder for the condition "Fold the paper along the two diagonals separately and see if the parts on both sides of the diagonals coincide completely each time"
  true

theorem feasible_test_for_rhombus (paper : Type) : is_rhombus paper → method_D paper :=
by
  intro h_rhombus
  sorry

end feasible_test_for_rhombus_l176_176511


namespace polynomial_evaluation_l176_176682

def polynomial_at (x : ℝ) : ℝ :=
  let f := (7 : ℝ) * x^5 + 12 * x^4 - 5 * x^3 - 6 * x^2 + 3 * x - 5
  f

theorem polynomial_evaluation : polynomial_at 3 = 2488 :=
by
  sorry

end polynomial_evaluation_l176_176682


namespace snow_prob_correct_l176_176968

variable (P : ℕ → ℚ)

-- Conditions
def prob_snow_first_four_days (i : ℕ) (h : i ∈ {1, 2, 3, 4}) : ℚ := 1 / 4
def prob_snow_next_three_days (i : ℕ) (h : i ∈ {5, 6, 7}) : ℚ := 1 / 3

-- Definition of no snow on a single day
def prob_no_snow_day (i : ℕ) (h : i ∈ {1, 2, 3, 4} ∪ {5, 6, 7}) : ℚ := 
  if h1 : i ∈ {1, 2, 3, 4} then 1 - prob_snow_first_four_days i h1
  else if h2 : i ∈ {5, 6, 7} then 1 - prob_snow_next_three_days i h2
  else 1

-- No snow all week
def prob_no_snow_all_week : ℚ := 
  (prob_no_snow_day 1 (by simp)) * (prob_no_snow_day 2 (by simp)) *
  (prob_no_snow_day 3 (by simp)) * (prob_no_snow_day 4 (by simp)) *
  (prob_no_snow_day 5 (by simp)) * (prob_no_snow_day 6 (by simp)) *
  (prob_no_snow_day 7 (by simp))

-- Probability of at least one snow day
def prob_at_least_one_snow_day : ℚ := 1 - prob_no_snow_all_week

-- Theorem
theorem snow_prob_correct : prob_at_least_one_snow_day = 29 / 32 := by
  -- Proof omitted, as requested
  sorry

end snow_prob_correct_l176_176968


namespace possible_to_select_three_numbers_l176_176452

theorem possible_to_select_three_numbers (n : ℕ) (a : ℕ → ℕ) (h : ∀ i j, i < j → a i < a j) (h_bound : ∀ i, a i < 2 * n) :
  ∃ i j k, i ≠ j ∧ i ≠ k ∧ j ≠ k ∧ a i + a j = a k := sorry

end possible_to_select_three_numbers_l176_176452


namespace lift_time_15_minutes_l176_176994

theorem lift_time_15_minutes (t : ℕ) (h₁ : 5 = 5) (h₂ : 6 * (t + 5) = 120) : t = 15 :=
by {
  sorry
}

end lift_time_15_minutes_l176_176994


namespace solve_equation_l176_176553

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 48 ↔ x = 6 ∨ x = 8 := 
by
  sorry

end solve_equation_l176_176553


namespace AB_distance_l176_176193

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l176_176193


namespace computer_operations_in_three_hours_l176_176293

theorem computer_operations_in_three_hours :
  let additions_per_second := 12000
  let multiplications_per_second := 2 * additions_per_second
  let seconds_in_three_hours := 3 * 3600
  (additions_per_second + multiplications_per_second) * seconds_in_three_hours = 388800000 :=
by
  sorry

end computer_operations_in_three_hours_l176_176293


namespace parabola_distance_l176_176188

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  sqrt ((q.1 - p.1) ^ 2 + (q.2 - p.2) ^ 2)

theorem parabola_distance (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hA_on_C : A.1 = (A.2)^2 / 4)
  (hAF_eq_BF : distance A F = distance B F):
  distance A B = 2 * sqrt 2 :=
by
  sorry

end parabola_distance_l176_176188


namespace max_volume_prism_l176_176100

theorem max_volume_prism (V : ℝ) (h l w : ℝ) 
  (h_eq_2h : l = 2 * h ∧ w = 2 * h) 
  (surface_area_eq : l * h + w * h + l * w = 36) : 
  V = 27 * Real.sqrt 2 := 
  sorry

end max_volume_prism_l176_176100


namespace intersection_M_complement_N_l176_176343

def U : Set ℝ := Set.univ

def M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

def N : Set ℝ := {x | ∃ y : ℝ, y = 3*x^2 + 1 }

def complement_N : Set ℝ := {x | ¬ ∃ y : ℝ, y = 3*x^2 + 1}

theorem intersection_M_complement_N :
  (M ∩ complement_N) = {x | -1 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_M_complement_N_l176_176343


namespace ab_distance_l176_176158

def parabola (x y : ℝ) : Prop := y^2 = 4 * x

def focus (F : ℝ × ℝ) : Prop := F = (1, 0)

def on_parabola (A : ℝ × ℝ) : Prop := parabola A.1 A.2

def distance (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

def equal_distance_from_focus (A B F : ℝ × ℝ) : Prop := distance A F = distance B F

theorem ab_distance:
  ∀ (A B F : ℝ × ℝ), 
    focus F →
    on_parabola A →
    B = (3, 0) →
    equal_distance_from_focus A B F →
    distance A B = 2 * Real.sqrt 2 :=
by
  intros A B F hF hA hB hEqual
  sorry

end ab_distance_l176_176158


namespace octahedron_parallel_edge_pairs_count_l176_176323

-- defining a regular octahedron structure
structure RegularOctahedron where
  vertices : Fin 8
  edges : Fin 12
  faces : Fin 8

noncomputable def numberOfStrictlyParallelEdgePairs (O : RegularOctahedron) : Nat :=
  12 -- Given the symmetry and structure.

theorem octahedron_parallel_edge_pairs_count (O : RegularOctahedron) : 
  numberOfStrictlyParallelEdgePairs O = 12 :=
by
  sorry

end octahedron_parallel_edge_pairs_count_l176_176323


namespace value_of_f_of_x_minus_3_l176_176593

theorem value_of_f_of_x_minus_3 (x : ℝ) (f : ℝ → ℝ) (h : ∀ y : ℝ, f y = y^2) : f (x - 3) = x^2 - 6*x + 9 :=
by
  sorry

end value_of_f_of_x_minus_3_l176_176593


namespace smallest_AAB_value_l176_176304

noncomputable def AAB_value (A B : ℕ) : ℕ := 100 * A + 10 * A + B

theorem smallest_AAB_value : ∃ (A B : ℕ), 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ A ≠ B ∧ 30 * A = 7 * B ∧ AAB_value A B = 773 :=
by
  use 7
  use 3
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  sorry

end smallest_AAB_value_l176_176304


namespace geometric_sequence_problem_l176_176475

noncomputable def geometric_sum (a q : ℕ) (n : ℕ) : ℕ :=
  a * (1 - q ^ n) / (1 - q)

theorem geometric_sequence_problem (a : ℕ) (q : ℕ) (n : ℕ) (h_q : q = 2) (h_n : n = 4) :
  (geometric_sum a q 4) / (a * q) = 15 / 2 :=
by
  sorry

end geometric_sequence_problem_l176_176475


namespace parabola_distance_l176_176170

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l176_176170


namespace parabola_distance_problem_l176_176148

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l176_176148


namespace ratio_Binkie_Frankie_eq_4_l176_176431

-- Definitions based on given conditions
def SpaatzGems : ℕ := 1
def BinkieGems : ℕ := 24

-- Assume the number of gemstones on Frankie's collar
variable (FrankieGems : ℕ)

-- Given condition about the gemstones on Spaatz's collar
axiom SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2

-- The theorem to be proved
theorem ratio_Binkie_Frankie_eq_4 
    (FrankieGems : ℕ) 
    (SpaatzCondition : SpaatzGems = (FrankieGems / 2) - 2) 
    (BinkieGems_eq : BinkieGems = 24) 
    (SpaatzGems_eq : SpaatzGems = 1) 
    (f_nonzero : FrankieGems ≠ 0) :
    BinkieGems / FrankieGems = 4 :=
by
  sorry  -- We're only writing the statement, not the proof.

end ratio_Binkie_Frankie_eq_4_l176_176431


namespace problem_l176_176182

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l176_176182


namespace sum_of_integers_square_greater_272_l176_176809

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l176_176809


namespace parabola_distance_problem_l176_176147

noncomputable def focus_of_parabola (a : ℝ) : (ℝ × ℝ) := (a, 0)

def distance (p q : ℝ × ℝ) : ℝ := 
Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem parabola_distance_problem : 
  let F := focus_of_parabola 1,
  (A : (ℝ × ℝ)) := (1, 2),
  (B : (ℝ × ℝ)) := (3, 0) in
  distance A F = distance B F → distance A B = 2 * Real.sqrt 2 :=
begin
  intros F A B h,
  sorry
end

end parabola_distance_problem_l176_176147


namespace range_of_sum_abs_l176_176927

variable {x y z : ℝ}

theorem range_of_sum_abs : 
  x^2 + y^2 + z = 15 → 
  x + y + z^2 = 27 → 
  xy + yz + zx = 7 → 
  7 ≤ |x + y + z| ∧ |x + y + z| ≤ 8 := by
  sorry

end range_of_sum_abs_l176_176927


namespace arithmetic_mean_reciprocals_primes_l176_176867

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176867


namespace temperature_at_80_degrees_l176_176215

theorem temperature_at_80_degrees (t : ℝ) :
  (-t^2 + 10 * t + 60 = 80) ↔ (t = 5 + 3 * Real.sqrt 5 ∨ t = 5 - 3 * Real.sqrt 5) := by
  sorry

end temperature_at_80_degrees_l176_176215


namespace room_width_l176_176251

theorem room_width (length : ℝ) (total_cost : ℝ) (rate_per_sqm : ℝ) (width : ℝ)
  (h_length : length = 5.5)
  (h_total_cost : total_cost = 15400)
  (h_rate_per_sqm : rate_per_sqm = 700)
  (h_area : total_cost = rate_per_sqm * (length * width)) :
  width = 4 := 
sorry

end room_width_l176_176251


namespace f_3_1_plus_f_3_4_l176_176365

def f (a b : ℕ) : ℚ :=
  if a + b < 5 then (a * b - a + 4) / (2 * a)
  else (a * b - b - 5) / (-2 * b)

theorem f_3_1_plus_f_3_4 :
  f 3 1 + f 3 4 = 7 / 24 :=
by
  sorry

end f_3_1_plus_f_3_4_l176_176365


namespace snow_probability_first_week_l176_176971

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l176_176971


namespace three_heads_in_a_row_l176_176649

theorem three_heads_in_a_row (h : 1 / 2) : (1 / 2) ^ 3 = 1 / 8 :=
by
  have fair_coin_probability : 1 / 2 = h := sorry
  have independent_events : ∀ a b : ℝ, a * b = h * b := sorry
  rw [fair_coin_probability]
  calc
    (1 / 2) ^ 3 = (1 / 2) * (1 / 2) * (1 / 2) : sorry
    ... = 1 / 8 : sorry

end three_heads_in_a_row_l176_176649


namespace scientific_notation_l176_176567

-- Given radius of a water molecule
def radius_of_water_molecule := 0.00000000192

-- Required scientific notation
theorem scientific_notation : radius_of_water_molecule = 1.92 * 10 ^ (-9) :=
by
  sorry

end scientific_notation_l176_176567


namespace sum_of_three_largest_l176_176052

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176052


namespace find_a_l176_176723

theorem find_a (a : ℝ) : 
  (∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
  ∃ y, y ≤ 6 ∧ 
  (∀ x, x^2 + y^2 = a^2 ∧ 
  x^2 + y^2 + a * y - 6 = 0)) → 
  a = 2 ∨ a = -2 :=
by sorry

end find_a_l176_176723


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176874

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1 / 2 + 1 / 3 + 1 / 5 + 1 / 7) / 4 = 247 / 840 := 
by 
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176874


namespace parabola_distance_l176_176168

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l176_176168


namespace parabola_problem_l176_176135

open Real

noncomputable def parabola_focus : ℝ × ℝ := (1, 0)

noncomputable def point_B : ℝ × ℝ := (3, 0)

noncomputable def point_on_parabola (x : ℝ) : ℝ × ℝ := (x, (4*x)^(1/2))

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem parabola_problem (x : ℝ)
  (hA : point_on_parabola x = (1, 2))
  (hAF_eq_BF : distance (1, 0) (1, 2) = distance (1, 0) (3, 0)) :
  distance (1, 2) (3, 0) = 2 * real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176135


namespace maximum_xy_l176_176448

theorem maximum_xy (x y : ℝ) (h : x^2 + 2 * y^2 - 2 * x * y = 4) : 
  xy ≤ 2 * (Float.sqrt 2) + 2 :=
sorry

end maximum_xy_l176_176448


namespace ralph_tv_hours_l176_176381

def hours_per_day_mf : ℕ := 4 -- 4 hours per day from Monday to Friday
def days_mf : ℕ := 5         -- 5 days from Monday to Friday
def hours_per_day_ss : ℕ := 6 -- 6 hours per day on Saturday and Sunday
def days_ss : ℕ := 2          -- 2 days, Saturday and Sunday

def total_hours_mf : ℕ := hours_per_day_mf * days_mf
def total_hours_ss : ℕ := hours_per_day_ss * days_ss
def total_hours_in_week : ℕ := total_hours_mf + total_hours_ss

theorem ralph_tv_hours : total_hours_in_week = 32 := 
by
sory -- proof will be written here

end ralph_tv_hours_l176_176381


namespace max_product_is_2331_l176_176623

open Nat

noncomputable def max_product (a b : ℕ) : ℕ :=
  if a + b = 100 ∧ a % 5 = 2 ∧ b % 6 = 3 then a * b else 0

theorem max_product_is_2331 (a b : ℕ) (h_sum : a + b = 100) (h_mod_a : a % 5 = 2) (h_mod_b : b % 6 = 3) :
  max_product a b = 2331 :=
  sorry

end max_product_is_2331_l176_176623


namespace limit_of_fraction_l176_176434

open Real

theorem limit_of_fraction (f : ℝ → ℝ) :
  (∀ x, f x = (x^3 - 1) / (x - 1)) → filter.tendsto f (𝓝 1) (𝓝 3) :=
by
  intros h
  have h₁ : ∀ x, x ≠ 1 → f x = x^2 + x + 1 := by
    intro x hx
    rw h
    field_simp [hx]
    ring
  have h₂ : ∀ x, f x = (x^3 - 1)/(x - 1) := h
  have h₃ : ∀ x, x ≠ 1 → f x = x^2 + x + 1 := by
    intro x hx
    rw [h₂, div_eq_iff (sub_ne_zero.mpr hx)]
    ring
  exact tendsto_congr' (eventually_nhds_iff.mpr ⟨{y | y ≠ 1}, is_open_ne, λ x hx, h₃ x hx⟩)
  simp
  exact tendsto_const_nhds.add (tendsto_const_nhds.add tendsto_id)


end limit_of_fraction_l176_176434


namespace total_spent_on_entertainment_l176_176110

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l176_176110


namespace domain_of_f_l176_176498

open Real

noncomputable def f (x : ℝ) : ℝ := (log (2 * x - x^2)) / (x - 1)

theorem domain_of_f (x : ℝ) : (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) ↔ (2 * x - x^2 > 0 ∧ x ≠ 1) := by
  sorry

end domain_of_f_l176_176498


namespace sum_of_three_largest_l176_176050

variable {n : ℕ}

def five_consecutive_numbers_sum (n : ℕ) := n + (n + 1) + (n + 2) = 60

theorem sum_of_three_largest (n : ℕ) (h : five_consecutive_numbers_sum n) : (n + 2) + (n + 3) + (n + 4) = 66 := by
  sorry

end sum_of_three_largest_l176_176050


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176888

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176888


namespace problem_l176_176185

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l176_176185


namespace part1_part2_l176_176512

-- Step 1: Define necessary probabilities
def P_A1 : ℚ := 5 / 6
def P_A2 : ℚ := 2 / 3
def P_B1 : ℚ := 3 / 5
def P_B2 : ℚ := 3 / 4

-- Step 2: Winning event probabilities for both participants
def P_A_wins := P_A1 * P_A2
def P_B_wins := P_B1 * P_B2

-- Step 3: Problem statement: Comparing probabilities
theorem part1 (P_A_wins P_A_wins : ℚ) : P_A_wins > P_B_wins := 
  by sorry

-- Step 4: Complement probabilities for not winning the competition
def P_not_A_wins := 1 - P_A_wins
def P_not_B_wins := 1 - P_B_wins

-- Step 5: Probability at least one wins
def P_at_least_one_wins := 1 - (P_not_A_wins * P_not_B_wins)

-- Step 6: Problem statement: At least one wins
theorem part2 : P_at_least_one_wins = 34 / 45 := 
  by sorry

end part1_part2_l176_176512


namespace mean_of_reciprocals_first_four_primes_l176_176860

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176860


namespace length_of_platform_l176_176535

theorem length_of_platform 
  (speed_kmph : ℕ)
  (time_cross_platform : ℕ)
  (time_cross_man : ℕ)
  (speed_mps : ℕ)
  (length_of_train : ℕ)
  (distance_platform : ℕ)
  (length_of_platform : ℕ) :
  speed_kmph = 72 →
  time_cross_platform = 30 →
  time_cross_man = 16 →
  speed_mps = speed_kmph * 1000 / 3600 →
  length_of_train = speed_mps * time_cross_man →
  distance_platform = speed_mps * time_cross_platform →
  length_of_platform = distance_platform - length_of_train →
  length_of_platform = 280 := by
  sorry

end length_of_platform_l176_176535


namespace sum_of_three_largest_l176_176032

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176032


namespace sequence_sum_l176_176755

def arithmetic_seq (a₀ : ℕ) (d : ℕ) : ℕ → ℕ
  | n => a₀ + n * d

def geometric_seq (b₀ : ℕ) (r : ℕ) : ℕ → ℕ
  | n => b₀ * r^(n)

theorem sequence_sum :
  let a : ℕ → ℕ := arithmetic_seq 3 1
  let b : ℕ → ℕ := geometric_seq 1 2
  b (a 0) + b (a 1) + b (a 2) + b (a 3) = 60 :=
  by
    let a : ℕ → ℕ := arithmetic_seq 3 1
    let b : ℕ → ℕ := geometric_seq 1 2
    have h₀ : a 0 = 3 := by rfl
    have h₁ : a 1 = 4 := by rfl
    have h₂ : a 2 = 5 := by rfl
    have h₃ : a 3 = 6 := by rfl
    have hsum : b 3 + b 4 + b 5 + b 6 = 60 := by sorry
    exact hsum

end sequence_sum_l176_176755


namespace evaluate_expression_l176_176852

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l176_176852


namespace rate_2nd_and_3rd_hours_equals_10_l176_176316

-- Define the conditions as given in the problem
def total_gallons_after_5_hours := 34 
def rate_1st_hour := 8 
def rate_4th_hour := 14 
def water_lost_5th_hour := 8 

-- Problem statement: Prove the rate during 2nd and 3rd hours is 10 gallons/hour
theorem rate_2nd_and_3rd_hours_equals_10 (R : ℕ) :
  total_gallons_after_5_hours = rate_1st_hour + 2 * R + rate_4th_hour - water_lost_5th_hour →
  R = 10 :=
by sorry

end rate_2nd_and_3rd_hours_equals_10_l176_176316


namespace negation_of_p_l176_176923

open Classical

variable (p : Prop)

theorem negation_of_p (h : ∀ x : ℝ, x^3 + 2 < 0) : 
  ∃ x : ℝ, x^3 + 2 ≥ 0 :=
by
  sorry

end negation_of_p_l176_176923


namespace average_speed_of_car_l176_176281

-- Definitions of the given conditions
def uphill_speed : ℝ := 30  -- km/hr
def downhill_speed : ℝ := 70  -- km/hr
def uphill_distance : ℝ := 100  -- km
def downhill_distance : ℝ := 50  -- km

-- Required proof statement (with the correct answer derived from the conditions)
theorem average_speed_of_car :
  (uphill_distance + downhill_distance) / 
  ((uphill_distance / uphill_speed) + (downhill_distance / downhill_speed)) = 37.04 := by
  sorry

end average_speed_of_car_l176_176281


namespace mean_of_reciprocals_of_first_four_primes_l176_176894

theorem mean_of_reciprocals_of_first_four_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let r1 := 1 / (p1 : ℚ)
  let r2 := 1 / (p2 : ℚ)
  let r3 := 1 / (p3 : ℚ)
  let r4 := 1 / (p4 : ℚ)
  (r1 + r2 + r3 + r4) / 4 = 247 / 840 :=
by
  sorry

end mean_of_reciprocals_of_first_four_primes_l176_176894


namespace tourism_revenue_scientific_notation_l176_176276

theorem tourism_revenue_scientific_notation:
  (12.41 * 10^9) = (1.241 * 10^9) := 
sorry

end tourism_revenue_scientific_notation_l176_176276


namespace polynomial_divisibility_l176_176978

theorem polynomial_divisibility (n : ℕ) : 120 ∣ (n^5 - 5*n^3 + 4*n) :=
sorry

end polynomial_divisibility_l176_176978


namespace sub_mixed_fraction_eq_l176_176271

theorem sub_mixed_fraction_eq :
  (2 + 1 / 4 : ℚ) - (2 / 3) = (1 + 7 / 12) := by
  sorry

end sub_mixed_fraction_eq_l176_176271


namespace cosine_periodicity_l176_176695

theorem cosine_periodicity (n : ℕ) (h_range : 0 ≤ n ∧ n ≤ 180) (h_cos : Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180)) :
  n = 43 :=
by
  sorry

end cosine_periodicity_l176_176695


namespace probability_three_heads_l176_176646

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l176_176646


namespace parabola_problem_l176_176152

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176152


namespace negation_equiv_l176_176503
variable (x : ℝ)

theorem negation_equiv :
  (¬ ∃ x : ℝ, x^2 + 1 > 3 * x) ↔ (∀ x : ℝ, x^2 + 1 ≤ 3 * x) :=
by 
  sorry

end negation_equiv_l176_176503


namespace OHara_triple_example_l176_176620

def is_OHara_triple (a b x : ℕ) : Prop :=
  (Real.sqrt a + Real.sqrt b = x)

theorem OHara_triple_example : is_OHara_triple 36 25 11 :=
by {
  sorry
}

end OHara_triple_example_l176_176620


namespace cash_calculation_l176_176411

theorem cash_calculation 
  (value_gold_coin : ℕ) (value_silver_coin : ℕ) 
  (num_gold_coins : ℕ) (num_silver_coins : ℕ) 
  (total_money : ℕ) : 
  value_gold_coin = 50 → 
  value_silver_coin = 25 → 
  num_gold_coins = 3 → 
  num_silver_coins = 5 → 
  total_money = 305 → 
  (total_money - (num_gold_coins * value_gold_coin + num_silver_coins * value_silver_coin) = 30) := 
by
  intros h1 h2 h3 h4 h5
  sorry

end cash_calculation_l176_176411


namespace value_of_a_l176_176353

noncomputable def f (a x : ℝ) : ℝ := a^x + Real.logb a (x^2 + 1)

theorem value_of_a (a : ℝ) (h : f a 1 + f a 2 = a^2 + a + 2) : a = Real.sqrt 10 :=
by
  sorry

end value_of_a_l176_176353


namespace chocolate_cookies_initial_count_l176_176835

theorem chocolate_cookies_initial_count
  (andy_ate : ℕ) (brother : ℕ) (friends_each : ℕ) (num_friends : ℕ)
  (team_members : ℕ) (first_share : ℕ) (common_diff : ℕ)
  (last_member_share : ℕ) (total_sum_team : ℕ)
  (total_cookies : ℕ) :
  andy_ate = 4 →
  brother = 6 →
  friends_each = 2 →
  num_friends = 3 →
  team_members = 10 →
  first_share = 2 →
  common_diff = 2 →
  last_member_share = first_share + (team_members - 1) * common_diff →
  total_sum_team = team_members / 2 * (first_share + last_member_share) →
  total_cookies = andy_ate + brother + (friends_each * num_friends) + total_sum_team →
  total_cookies = 126 :=
by
  intros ha hb hf hn ht hf1 hc hl hs ht
  sorry

end chocolate_cookies_initial_count_l176_176835


namespace daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l176_176295

-- Part (1)
theorem daily_sales_volume_and_profit (x : ℝ) :
  let increase_in_sales := 2 * x
  let profit_per_piece := 40 - x
  increase_in_sales = 2 * x ∧ profit_per_piece = 40 - x :=
by
  sorry

-- Part (2)
theorem profit_for_1200_yuan (x : ℝ) (h1 : (40 - x) * (20 + 2 * x) = 1200) :
  x = 10 ∨ x = 20 :=
by
  sorry

-- Part (3)
theorem profit_impossible_for_1800_yuan :
  ¬ ∃ y : ℝ, (40 - y) * (20 + 2 * y) = 1800 :=
by
  sorry

end daily_sales_volume_and_profit_profit_for_1200_yuan_profit_impossible_for_1800_yuan_l176_176295


namespace bicycle_meets_light_vehicle_l176_176618

noncomputable def meeting_time (v_1 v_2 v_3 v_4 : ℚ) : ℚ :=
  let x := 2 * (v_1 + v_4)
  let y := 6 * (v_2 - v_4)
  (x + y) / (v_3 + v_4) + 12

theorem bicycle_meets_light_vehicle (v_1 v_2 v_3 v_4 : ℚ) (h1 : 2 * (v_1 + v_4) = x)
  (h2 : x + y = 4 * (v_1 + v_2))
  (h3 : x + y = 5 * (v_2 + v_3))
  (h4 : 6 * (v_2 - v_4) = y) :
  meeting_time v_1 v_2 v_3 v_4 = 15 + 1/3 :=
by
  sorry

end bicycle_meets_light_vehicle_l176_176618


namespace difference_in_payment_l176_176946

theorem difference_in_payment (joy_pencils : ℕ) (colleen_pencils : ℕ) (price_per_pencil : ℕ) (H1 : joy_pencils = 30) (H2 : colleen_pencils = 50) (H3 : price_per_pencil = 4) :
  (colleen_pencils * price_per_pencil) - (joy_pencils * price_per_pencil) = 80 :=
by
  rw [H1, H2, H3]
  simp
  norm_num
  sorry

end difference_in_payment_l176_176946


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176900

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176900


namespace AB_distance_l176_176195

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l176_176195


namespace count_valid_three_digit_numbers_l176_176068

open Function

-- Definitions of the conditions
def valid_number_of_digits := {1, 2, 3, 4, 5}
def no_repeating_digits : ∀ (d1 d2 d3 : ℕ), d1 ≠ d2 ∧ d1 ≠ d3 ∧ d2 ≠ d3
def two_precedes_three : ∀ (d1 d2 d3 : ℕ), d1 = 2 → d2 = 3 → False

-- Main Theorem Statement
theorem count_valid_three_digit_numbers : 
  (finset.card {num | 
    ∃ (d1 d2 d3 : ℕ), 
      d1 ∈ valid_number_of_digits ∧ 
      d2 ∈ valid_number_of_digits ∧ 
      d3 ∈ valid_number_of_digits ∧ 
      d1 ≠ d2 ∧ 
      d1 ≠ d3 ∧ 
      d2 ≠ d3 ∧ 
      ¬(two_precedes_three d1 d2 d3) 
    } = 51) :=
sorry

end count_valid_three_digit_numbers_l176_176068


namespace roy_cat_finishes_food_on_wednesday_l176_176676

-- Define the conditions
def morning_consumption := (1 : ℚ) / 5
def evening_consumption := (1 : ℚ) / 6
def total_cans := 10

-- Define the daily consumption calculation
def daily_consumption := morning_consumption + evening_consumption

-- Define the day calculation function
def day_cat_finishes_food : String :=
  let total_days := total_cans / daily_consumption
  if total_days ≤ 7 then "certain day within a week"
  else if total_days ≤ 14 then "Wednesday next week"
  else "later"

-- The main theorem to prove
theorem roy_cat_finishes_food_on_wednesday : day_cat_finishes_food = "Wednesday next week" := sorry

end roy_cat_finishes_food_on_wednesday_l176_176676


namespace ratio_unit_price_l176_176427

theorem ratio_unit_price (v p : ℝ) (hv : v > 0) (hp : p > 0) :
  let vA := 1.25 * v
  let pA := 0.85 * p
  (pA / vA) / (p / v) = 17 / 25 :=
by
  let vA := 1.25 * v
  let pA := 0.85 * p
  have unit_price_B := p / v
  have unit_price_A := pA / vA
  have ratio := unit_price_A / unit_price_B
  have h_pA_vA : pA / vA = 17 / 25 * (p / v) := by
    sorry
  exact calc
    (pA / vA) / (p / v) = 17 / 25 : by
      rw [← h_pA_vA]
      exact (div_div_eq_div_mul _ _ _).symm

end ratio_unit_price_l176_176427


namespace quadratic_no_real_solution_l176_176921

theorem quadratic_no_real_solution 
  (a b c : ℝ) 
  (h1 : (2 * a)^2 - 4 * b^2 > 0) 
  (h2 : (2 * b)^2 - 4 * c^2 > 0) : 
  (2 * c)^2 - 4 * a^2 < 0 :=
sorry

end quadratic_no_real_solution_l176_176921


namespace probability_three_heads_l176_176647

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l176_176647


namespace simplify_expression_l176_176982

variable (a b : ℤ)

theorem simplify_expression : 
  (50 * a + 130 * b) + (21 * a + 64 * b) - (30 * a + 115 * b) - 2 * (10 * a - 25 * b) = 21 * a + 129 * b := 
by
  sorry

end simplify_expression_l176_176982


namespace symmetric_line_eq_l176_176391

theorem symmetric_line_eq (x y : ℝ) :
    3 * x - 4 * y + 5 = 0 ↔ 3 * x + 4 * (-y) + 5 = 0 :=
sorry

end symmetric_line_eq_l176_176391


namespace swimming_pool_min_cost_l176_176416

theorem swimming_pool_min_cost (a : ℝ) (x : ℝ) (y : ℝ) :
  (∀ (x : ℝ), x > 0 → y = 2400 * a + 6 * (x + 1600 / x) * a) →
  (∃ (x : ℝ), x > 0 ∧ y = 2880 * a) :=
by
  sorry

end swimming_pool_min_cost_l176_176416


namespace possible_polynomials_l176_176785

noncomputable def f (x : ℝ) : ℝ := x^2

theorem possible_polynomials (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x^2 - 6 * x + 1) → 
  (∀ x, (g x = 3 * x - 1) ∨ (g x = -(3 * x - 1))) := 
by
  intros h x
  sorry

end possible_polynomials_l176_176785


namespace smaller_consecutive_number_divisibility_l176_176308

theorem smaller_consecutive_number_divisibility :
  ∃ (m : ℕ), (m < m + 1) ∧ (1 ≤ m ∧ m ≤ 200) ∧ (1 ≤ m + 1 ∧ m + 1 ≤ 200) ∧
              (∀ n, (1 ≤ n ∧ n ≤ 200 ∧ n ≠ m ∧ n ≠ m + 1) → ∃ k, chosen_num = k * n) ∧
              (128 = m) :=
sorry

end smaller_consecutive_number_divisibility_l176_176308


namespace annulus_area_l176_176422

variables {R r d : ℝ}
variables (h1 : R > r) (h2 : d < R)

theorem annulus_area :
  π * (R^2 - r^2 - d^2 / (R - r)) = π * ((R - r)^2 - d^2) :=
sorry

end annulus_area_l176_176422


namespace sum_of_octal_numbers_l176_176305

theorem sum_of_octal_numbers :
  (176 : ℕ) + 725 + 63 = 1066 := by
sorry

end sum_of_octal_numbers_l176_176305


namespace correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176776

-- Define the given percentages for each ship
def P_A : ℝ := 0.30
def C_A : ℝ := 0.25
def P_B : ℝ := 0.50
def C_B : ℝ := 0.15
def P_C : ℝ := 0.20
def C_C : ℝ := 0.35

-- Define the derived non-car round-trip percentages 
def NR_A : ℝ := P_A - (P_A * C_A)
def NR_B : ℝ := P_B - (P_B * C_B)
def NR_C : ℝ := P_C - (P_C * C_C)

-- Statements to be proved
theorem correct_NR_A : NR_A = 0.225 := sorry
theorem correct_NR_B : NR_B = 0.425 := sorry
theorem correct_NR_C : NR_C = 0.13 := sorry

-- Proof that NR_B is the highest percentage
theorem NR_B_highest : NR_B > NR_A ∧ NR_B > NR_C := sorry

end correct_NR_A_correct_NR_B_correct_NR_C_NR_B_highest_l176_176776


namespace determine_constants_l176_176548

theorem determine_constants (k a b : ℝ) :
  (3*x^2 - 4*x + 5)*(5*x^2 + k*x + 8) = 15*x^4 - 47*x^3 + a*x^2 - b*x + 40 →
  k = -9 ∧ a = 15 ∧ b = 72 :=
by
  sorry

end determine_constants_l176_176548


namespace sum_of_largest_three_consecutive_numbers_l176_176058

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176058


namespace parabola_problem_l176_176163

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l176_176163


namespace sum_of_largest_three_consecutive_numbers_l176_176061

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176061


namespace right_triangle_area_l176_176992

theorem right_triangle_area (a b c : ℝ) (h1 : a + b + c = 90) (h2 : a^2 + b^2 + c^2 = 3362) (h3 : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 180 :=
by
  sorry

end right_triangle_area_l176_176992


namespace robotics_club_neither_l176_176759

theorem robotics_club_neither (total students programming electronics both: ℕ) 
  (h1: total = 120)
  (h2: programming = 80)
  (h3: electronics = 50)
  (h4: both = 15) : 
  total - ((programming - both) + (electronics - both) + both) = 5 :=
by
  sorry

end robotics_club_neither_l176_176759


namespace number_of_paths_from_A_to_D_l176_176910

-- Definitions based on conditions
def paths_A_to_B : ℕ := 2
def paths_B_to_C : ℕ := 2
def paths_A_to_C : ℕ := 1
def paths_C_to_D : ℕ := 2
def paths_B_to_D : ℕ := 2

-- Theorem statement
theorem number_of_paths_from_A_to_D : 
  paths_A_to_B * paths_B_to_C * paths_C_to_D + 
  paths_A_to_C * paths_C_to_D + 
  paths_A_to_B * paths_B_to_D = 14 :=
by {
  -- proof steps will go here
  sorry
}

end number_of_paths_from_A_to_D_l176_176910


namespace nurse_distribution_l176_176425

theorem nurse_distribution (nurses hospitals : ℕ) (h1 : nurses = 3) (h2 : hospitals = 6) 
  (h3 : ∀ (a b c : ℕ), a = b → b = c → a = c → a ≤ 2) : 
  (hospitals^nurses - hospitals) = 210 := 
by 
  sorry

end nurse_distribution_l176_176425


namespace parabola_distance_l176_176141

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l176_176141


namespace train_A_reaches_destination_in_6_hours_l176_176266

noncomputable def t : ℕ := 
  let tA := 110
  let tB := 165
  let tB_time := 4
  (tB * tB_time) / tA

theorem train_A_reaches_destination_in_6_hours :
  t = 6 := by
  sorry

end train_A_reaches_destination_in_6_hours_l176_176266


namespace rug_area_is_24_l176_176671

def length_floor : ℕ := 12
def width_floor : ℕ := 10
def strip_width : ℕ := 3

theorem rug_area_is_24 :
  (length_floor - 2 * strip_width) * (width_floor - 2 * strip_width) = 24 := 
by
  sorry

end rug_area_is_24_l176_176671


namespace mean_of_reciprocals_first_four_primes_l176_176863

theorem mean_of_reciprocals_first_four_primes :
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  mean = (247 / 840 : ℚ) :=
by 
  let p1 := (2 : ℕ)
  let p2 := (3 : ℕ)
  let p3 := (5 : ℕ)
  let p4 := (7 : ℕ)
  let rec1 := 1 / (p1 : ℚ)
  let rec2 := 1 / (p2 : ℚ)
  let rec3 := 1 / (p3 : ℚ)
  let rec4 := 1 / (p4 : ℚ)
  let mean := (rec1 + rec2 + rec3 + rec4) / 4
  show mean = (247 / 840 : ℚ), from
  sorry

end mean_of_reciprocals_first_four_primes_l176_176863


namespace intersection_with_y_axis_l176_176987

theorem intersection_with_y_axis :
  ∃ (x y : ℝ), x = 0 ∧ y = 5 * x - 6 ∧ (x, y) = (0, -6) := 
sorry

end intersection_with_y_axis_l176_176987


namespace a_profit_share_l176_176823

/-- Definitions for the shares of capital -/
def a_share : ℚ := 1 / 3
def b_share : ℚ := 1 / 4
def c_share : ℚ := 1 / 5
def d_share : ℚ := 1 - (a_share + b_share + c_share)
def total_profit : ℚ := 2415

/-- The profit share for A, given the conditions on capital subscriptions -/
theorem a_profit_share : a_share * total_profit = 805 := by
  sorry

end a_profit_share_l176_176823


namespace ralph_tv_hours_l176_176379

theorem ralph_tv_hours :
  (4 * 5 + 6 * 2) = 32 :=
by
  sorry

end ralph_tv_hours_l176_176379


namespace determine_plane_by_trapezoid_legs_l176_176267

-- Defining basic objects
structure Point := (x : ℝ) (y : ℝ) (z : ℝ)
structure Line := (p1 : Point) (p2 : Point)
structure Plane := (l1 : Line) (l2 : Line)

-- Theorem statement for the problem
theorem determine_plane_by_trapezoid_legs (trapezoid_legs : Line) :
  ∃ (pl : Plane), ∀ (l1 l2 : Line), (l1 = trapezoid_legs) ∧ (l2 = trapezoid_legs) → (pl = Plane.mk l1 l2) :=
sorry

end determine_plane_by_trapezoid_legs_l176_176267


namespace general_formula_an_l176_176466

theorem general_formula_an {a : ℕ → ℝ} (S : ℕ → ℝ) (d : ℝ) (hS : ∀ n, S n = (n / 2) * (a 1 + a n)) (hd : d = a 2 - a 1) : 
  ∀ n, a n = a 1 + (n - 1) * d :=
sorry

end general_formula_an_l176_176466


namespace maximum_watchman_demand_l176_176813

theorem maximum_watchman_demand (bet_loss : ℕ) (bet_win : ℕ) (x : ℕ) 
  (cond_bet_loss : bet_loss = 100)
  (cond_bet_win : bet_win = 100) :
  x < 200 :=
by
  have h₁ : bet_loss = 100 := cond_bet_loss
  have h₂ : bet_win = 100 := cond_bet_win
  sorry

end maximum_watchman_demand_l176_176813


namespace sum_of_first_17_terms_l176_176577

variable {α : Type*} [LinearOrderedField α] 

-- conditions
def arithmetic_sequence (a : ℕ → α) : Prop := 
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → α) (S : ℕ → α) : Prop :=
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

variable {a : ℕ → α}
variable {S : ℕ → α}

-- main theorem
theorem sum_of_first_17_terms (h_arith : arithmetic_sequence a)
  (h_S : sum_of_first_n_terms a S)
  (h_condition : a 7 + a 12 = 12 - a 8) :
  S 17 = 68 := sorry

end sum_of_first_17_terms_l176_176577


namespace tank_inflow_rate_l176_176619

/-- 
  Tanks A and B have the same capacity of 20 liters. Tank A has
  an inflow rate of 2 liters per hour and takes 5 hours longer to
  fill than tank B. Show that the inflow rate in tank B is 4 liters 
  per hour.
-/
theorem tank_inflow_rate (capacity : ℕ) (rate_A : ℕ) (extra_time : ℕ) (rate_B : ℕ) 
  (h1 : capacity = 20) (h2 : rate_A = 2) (h3 : extra_time = 5) (h4 : capacity / rate_A = (capacity / rate_B) + extra_time) :
  rate_B = 4 :=
sorry

end tank_inflow_rate_l176_176619


namespace midpoint_product_l176_176594

theorem midpoint_product (x' y' : ℤ) 
  (h1 : (0 + x') / 2 = 2) 
  (h2 : (9 + y') / 2 = 4) : 
  (x' * y') = -4 :=
by
  sorry

end midpoint_product_l176_176594


namespace boat_cost_per_foot_l176_176598

theorem boat_cost_per_foot (total_savings : ℝ) (license_cost : ℝ) (docking_fee_multiplier : ℝ) (max_boat_length : ℝ) 
  (h1 : total_savings = 20000) 
  (h2 : license_cost = 500) 
  (h3 : docking_fee_multiplier = 3) 
  (h4 : max_boat_length = 12) 
  : (total_savings - (license_cost + docking_fee_multiplier * license_cost)) / max_boat_length = 1500 :=
by
  sorry

end boat_cost_per_foot_l176_176598


namespace find_integer_cosine_l176_176698

theorem find_integer_cosine :
  ∃ n: ℤ, 0 ≤ n ∧ n ≤ 180 ∧ real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180) :=
begin
  use 43,
  split,
  { norm_num },
  split,
  { norm_num },
  { sorry }
end

end find_integer_cosine_l176_176698


namespace cos_eq_43_l176_176700

theorem cos_eq_43 (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) (h3 : cos (n * pi / 180) = cos (317 * pi / 180)) : n = 43 :=
sorry

end cos_eq_43_l176_176700


namespace pages_after_break_l176_176587

-- Formalize the conditions
def total_pages : ℕ := 30
def break_percentage : ℝ := 0.70

-- Define the proof problem
theorem pages_after_break : 
  let pages_read_before_break := (break_percentage * total_pages)
  let pages_remaining := total_pages - pages_read_before_break
  pages_remaining = 9 :=
by
  sorry

end pages_after_break_l176_176587


namespace total_cantaloupes_l176_176066

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l176_176066


namespace solve_equation_l176_176603

theorem solve_equation (x : ℝ) : 
  (9 - 3 * x) * (3 ^ x) - (x - 2) * (x ^ 2 - 5 * x + 6) = 0 ↔ x = 3 :=
by sorry

end solve_equation_l176_176603


namespace min_distance_eq_sqrt2_l176_176566

open Real

variables {P Q : ℝ × ℝ}
variables {x y : ℝ}

/-- Given that point P is on the curve y = e^x and point Q is on the curve y = ln x, prove that the minimum value of the distance |PQ| is sqrt(2). -/
theorem min_distance_eq_sqrt2 : 
  (P.2 = exp P.1) ∧ (Q.2 = log Q.1) → (dist P Q) = sqrt 2 :=
by
  sorry

end min_distance_eq_sqrt2_l176_176566


namespace Linda_total_sales_l176_176756

theorem Linda_total_sales (necklaces_sold : ℕ) (rings_sold : ℕ) 
    (necklace_price : ℕ) (ring_price : ℕ) 
    (total_sales : ℕ) : 
    necklaces_sold = 4 → 
    rings_sold = 8 → 
    necklace_price = 12 → 
    ring_price = 4 → 
    total_sales = necklaces_sold * necklace_price + rings_sold * ring_price → 
    total_sales = 80 :=
by
  intros H1 H2 H3 H4 H5
  sorry

end Linda_total_sales_l176_176756


namespace one_fifth_of_5_times_7_l176_176694

theorem one_fifth_of_5_times_7 : (1 / 5) * (5 * 7) = 7 := by
  sorry

end one_fifth_of_5_times_7_l176_176694


namespace flowers_per_row_correct_l176_176275

/-- Definition for the number of each type of flower. -/
def num_yellow_flowers : ℕ := 12
def num_green_flowers : ℕ := 2 * num_yellow_flowers -- Given that green flowers are twice the yellow flowers.
def num_red_flowers : ℕ := 42

/-- Total number of flowers. -/
def total_flowers : ℕ := num_yellow_flowers + num_green_flowers + num_red_flowers

/-- Number of rows in the garden. -/
def num_rows : ℕ := 6

/-- The number of flowers per row Wilma's garden has. -/
def flowers_per_row : ℕ := total_flowers / num_rows

/-- Proof statement: flowers per row should be 13. -/
theorem flowers_per_row_correct : flowers_per_row = 13 :=
by
  -- The proof will go here.
  sorry

end flowers_per_row_correct_l176_176275


namespace relationship_between_A_and_B_l176_176833

def A : Set ℤ := {-2, 0, 2}
def B : Set ℤ := {x | x^2 + 2 * x = 0}

theorem relationship_between_A_and_B : B ⊆ A :=
sorry

end relationship_between_A_and_B_l176_176833


namespace inequality_holds_l176_176595

theorem inequality_holds (a b c : ℝ) 
  (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : a * b * c = 1) : 
  1 / (a^3 * (b + c)) + 1 / (b^3 * (a + c)) + 1 / (c^3 * (a + b)) ≥ 3 / 2 :=
by
  sorry

end inequality_holds_l176_176595


namespace question_inequality_l176_176750

theorem question_inequality
  (a b : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (cond : a + b ≤ 4) :
  (1 / a + 1 / b) ≥ 1 := 
sorry

end question_inequality_l176_176750


namespace radius_of_inscribed_circle_l176_176658

theorem radius_of_inscribed_circle (a b c : ℝ) (r : ℝ) 
  (ha : a = 5) (hb : b = 10) (hc : c = 20)
  (h : 1 / r = 1 / a + 1 / b + 1 / c + 2 * Real.sqrt ((1 / (a * b)) + (1 / (a * c)) + (1 / (b * c)))) :
  r = 20 * (7 - Real.sqrt 10) / 39 :=
by
  -- Statements and conditions are setup, but the proof is omitted.
  sorry

end radius_of_inscribed_circle_l176_176658


namespace problem_l176_176184

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l176_176184


namespace max_value_expr_l176_176956

theorem max_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  (∀ x : ℝ, (a + b)^2 / (a^2 + 2 * a * b + b^2) ≤ x) → 1 ≤ x :=
sorry

end max_value_expr_l176_176956


namespace train_length_proof_l176_176672

noncomputable def train_length (speed_kmh : ℕ) (time_s : ℕ) : ℕ :=
  let speed_ms := speed_kmh * 5 / 18
  speed_ms * time_s

theorem train_length_proof : train_length 144 16 = 640 := by
  sorry

end train_length_proof_l176_176672


namespace width_decrease_percentage_l176_176248

theorem width_decrease_percentage {L W W' : ℝ} 
  (h1 : W' = W / 1.40)
  (h2 : 1.40 * L * W' = L * W) : 
  W' = 0.7143 * W → (1 - W' / W) * 100 = 28.57 := 
by
  sorry

end width_decrease_percentage_l176_176248


namespace mean_noon_temperature_l176_176253

def temperatures : List ℕ := [82, 80, 83, 88, 90, 92, 90, 95]

def mean_temperature (temps : List ℕ) : ℚ :=
  (temps.foldr (λ a b => a + b) 0 : ℚ) / temps.length

theorem mean_noon_temperature :
  mean_temperature temperatures = 87.5 := by
  sorry

end mean_noon_temperature_l176_176253


namespace inequality_proof_l176_176446

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b + b * c + c * d + d * a = 1) : 
  (a^3 / (b + c + d) + b^3 / (c + d + a) + c^3 / (d + a + b) + d^3 / (a + b + c)) ≥ 1 / 3 := 
sorry

end inequality_proof_l176_176446


namespace sum_of_three_largest_l176_176033

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176033


namespace neg_p_iff_forall_l176_176447

-- Define the proposition p
def p : Prop := ∃ (x : ℝ), x > 1 ∧ x^2 - 1 > 0

-- State the negation of p as a theorem
theorem neg_p_iff_forall : ¬ p ↔ ∀ (x : ℝ), x > 1 → x^2 - 1 ≤ 0 :=
by sorry

end neg_p_iff_forall_l176_176447


namespace rod_length_is_38_point_25_l176_176443

noncomputable def length_of_rod (n : ℕ) (l : ℕ) (conversion_factor : ℕ) : ℝ :=
  (n * l : ℝ) / conversion_factor

theorem rod_length_is_38_point_25 :
  length_of_rod 45 85 100 = 38.25 :=
by
  sorry

end rod_length_is_38_point_25_l176_176443


namespace find_a_l176_176728

def setA : Set ℤ := {-1, 0, 1}

def setB (a : ℤ) : Set ℤ := {a, a ^ 2}

theorem find_a (a : ℤ) (h : setA ∪ setB a = setA) : a = -1 :=
sorry

end find_a_l176_176728


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176902

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176902


namespace profit_percentage_is_40_l176_176601

-- Define the given conditions
def total_cost : ℚ := 44 * 150 + 36 * 125  -- Rs 11100
def total_weight : ℚ := 44 + 36            -- 80 kg
def selling_price_per_kg : ℚ := 194.25     -- Rs 194.25
def total_selling_price : ℚ := total_weight * selling_price_per_kg  -- Rs 15540
def profit : ℚ := total_selling_price - total_cost  -- Rs 4440

-- Define the statement about the profit percentage
def profit_percentage : ℚ := (profit / total_cost) * 100

-- State the theorem
theorem profit_percentage_is_40 :
  profit_percentage = 40 := by
  -- This is where the proof would go
  sorry

end profit_percentage_is_40_l176_176601


namespace find_AD_length_l176_176742

variables (A B C D O : Point)
variables (BO OD AO OC AB AD : ℝ)

def quadrilateral_properties (BO OD AO OC AB : ℝ) (O : Point) : Prop :=
  BO = 3 ∧ OD = 9 ∧ AO = 5 ∧ OC = 2 ∧ AB = 7

theorem find_AD_length (h : quadrilateral_properties BO OD AO OC AB O) : AD = Real.sqrt 151 :=
by
  sorry

end find_AD_length_l176_176742


namespace Marla_colors_green_squares_l176_176208

theorem Marla_colors_green_squares :
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  green_squares = 66 :=
by
  let total_squares := 10 * 15
  let red_squares := 4 * 6
  let blue_squares := 4 * 15
  let green_squares := total_squares - red_squares - blue_squares
  show green_squares = 66
  sorry

end Marla_colors_green_squares_l176_176208


namespace find_son_age_l176_176521

variable {S F : ℕ}

theorem find_son_age (h1 : F = S + 35) (h2 : F + 2 = 2 * (S + 2)) : S = 33 :=
sorry

end find_son_age_l176_176521


namespace six_digit_palindromes_count_l176_176005

noncomputable def count_six_digit_palindromes : ℕ :=
  let a_choices := 9
  let bcd_choices := 10 * 10 * 10
  a_choices * bcd_choices

theorem six_digit_palindromes_count : count_six_digit_palindromes = 9000 := by
  unfold count_six_digit_palindromes
  simp
  sorry

end six_digit_palindromes_count_l176_176005


namespace point_on_inverse_graph_and_sum_l176_176606

-- Definitions
variable (f : ℝ → ℝ)
variable (h : f 2 = 6)

-- Theorem statement
theorem point_on_inverse_graph_and_sum (hf : ∀ x, x = 2 → 3 = (f x) / 2) :
  (6, 1 / 2) ∈ {p : ℝ × ℝ | ∃ x, p = (x, (f⁻¹ x) / 2)} ∧
  (6 + (1 / 2) = 13 / 2) :=
by
  sorry

end point_on_inverse_graph_and_sum_l176_176606


namespace highest_percentage_without_car_l176_176770

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l176_176770


namespace number_of_erasers_l176_176998

theorem number_of_erasers (P E : ℕ) (h1 : P + E = 240) (h2 : P = E - 2) : E = 121 := by
  sorry

end number_of_erasers_l176_176998


namespace find_2008_star_2010_l176_176321

-- Define the operation
def operation_star (x y : ℕ) : ℕ := sorry  -- We insert a sorry here because the precise definition is given by the conditions

-- The properties given in the problem
axiom property1 : operation_star 2 2010 = 1
axiom property2 : ∀ n : ℕ, operation_star (2 * (n + 1)) 2010 = 3 * operation_star (2 * n) 2010

-- The main proof statement
theorem find_2008_star_2010 : operation_star 2008 2010 = 3 ^ 1003 :=
by
  -- Here we would provide the proof, but it's omitted.
  sorry

end find_2008_star_2010_l176_176321


namespace evaluate_fraction_l176_176691

theorem evaluate_fraction : (3 / (1 - 3 / 4) = 12) := by
  have h : (1 - 3 / 4) = 1 / 4 := by
    sorry
  rw [h]
  sorry

end evaluate_fraction_l176_176691


namespace ship_with_highest_no_car_round_trip_percentage_l176_176765

theorem ship_with_highest_no_car_round_trip_percentage
    (pA : ℝ)
    (cA_r : ℝ)
    (pB : ℝ)
    (cB_r : ℝ)
    (pC : ℝ)
    (cC_r : ℝ)
    (hA : pA = 0.30)
    (hA_car : cA_r = 0.25)
    (hB : pB = 0.50)
    (hB_car : cB_r = 0.15)
    (hC : pC = 0.20)
    (hC_car : cC_r = 0.35) :
    let percentA := pA - (cA_r * pA)
    let percentB := pB - (cB_r * pB)
    let percentC := pC - (cC_r * pC)
    percentB > percentA ∧ percentB > percentC :=
by
  sorry

end ship_with_highest_no_car_round_trip_percentage_l176_176765


namespace exists_p_for_q_l176_176782

noncomputable def sqrt_56 : ℝ := Real.sqrt 56
noncomputable def sqrt_58 : ℝ := Real.sqrt 58

theorem exists_p_for_q (q : ℕ) (hq : q > 0) (hq_ne_1 : q ≠ 1) (hq_ne_3 : q ≠ 3) :
  ∃ p : ℤ, sqrt_56 < (p : ℝ) / q ∧ (p : ℝ) / q < sqrt_58 :=
by sorry

end exists_p_for_q_l176_176782


namespace problem_l176_176183

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l176_176183


namespace linear_function_quadrants_l176_176737

theorem linear_function_quadrants (a b : ℝ) (h1 : a < 0) (h2 : b > 0) : ¬ ∃ x : ℝ, ∃ y : ℝ, x > 0 ∧ y < 0 ∧ y = b * x - a :=
sorry

end linear_function_quadrants_l176_176737


namespace polynomial_evaluation_l176_176687

-- Define operations using Lean syntax
def star (a b : ℚ) := a + b
def otimes (a b : ℚ) := a - b

-- Define a function to represent the polynomial expression
def expression (a b : ℚ) := star (a^2 * b) (3 * a * b) + otimes (5 * a^2 * b) (4 * a * b)

theorem polynomial_evaluation (a b : ℚ) (ha : a = 5) (hb : b = 3) : expression a b = 435 := by
  sorry

end polynomial_evaluation_l176_176687


namespace find_n_l176_176704

theorem find_n (n : ℕ) (h₁ : 0 ≤ n) (h₂ : n ≤ 180) (h₃ : real.cos (n * real.pi / 180) = real.cos (317 * real.pi / 180)) : n = 43 := 
sorry

end find_n_l176_176704


namespace probability_three_heads_l176_176639

theorem probability_three_heads (p : ℝ) (h : ∀ n : ℕ, n < 3 → p = 1 / 2):
  (p * p * p) = 1 / 8 :=
by {
  -- p must be 1/2 for each flip
  have hp : p = 1 / 2 := by obtain ⟨m, hm⟩ := h 0 (by norm_num); exact hm,
  rw hp,
  norm_num,
  sorry -- This would be where a more detailed proof goes.
}

end probability_three_heads_l176_176639


namespace intersection_complement_M_and_N_l176_176749
open Set

def U := @univ ℝ
def M := {x : ℝ | x^2 + 2*x - 8 ≤ 0}
def N := {x : ℝ | -1 < x ∧ x < 3}
def complement_M := {x : ℝ | ¬ (x ∈ M)}

theorem intersection_complement_M_and_N :
  (complement_M ∩ N) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_complement_M_and_N_l176_176749


namespace colleen_paid_more_l176_176950

-- Define the number of pencils Joy has
def joy_pencils : ℕ := 30

-- Define the number of pencils Colleen has
def colleen_pencils : ℕ := 50

-- Define the cost per pencil
def pencil_cost : ℕ := 4

-- The proof problem: Colleen paid $80 more for her pencils than Joy
theorem colleen_paid_more : 
  (colleen_pencils - joy_pencils) * pencil_cost = 80 := by
  sorry

end colleen_paid_more_l176_176950


namespace fourth_student_guess_l176_176794

theorem fourth_student_guess :
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  fourth_guess = 525 := 
by
  let first_guess := 100
  let second_guess := 8 * first_guess
  let third_guess := second_guess - 200
  let avg_three_guesses := (first_guess + second_guess + third_guess) / 3
  let fourth_guess := avg_three_guesses + 25
  show fourth_guess = 525 from sorry

end fourth_student_guess_l176_176794


namespace target_more_tools_than_walmart_target_to_walmart_tools_ratio_l176_176515

def walmart_screwdrivers : ℕ := 2
def walmart_knives : ℕ := 4
def walmart_can_opener : ℕ := 1
def walmart_pliers : ℕ := 1
def walmart_bottle_opener : ℕ := 1

def target_screwdrivers : ℕ := 3
def target_knives : ℕ := 12
def target_can_openers : ℕ := 2
def target_scissors : ℕ := 1
def target_pliers : ℕ := 1
def target_saws : ℕ := 2

def walmart_tools : ℕ := walmart_screwdrivers + walmart_knives + walmart_can_opener + walmart_pliers + walmart_bottle_opener
def target_tools : ℕ := target_screwdrivers + target_knives + target_can_openers + target_scissors + target_pliers + target_saws

theorem target_more_tools_than_walmart : target_tools - walmart_tools = 12 :=
by sorry

theorem target_to_walmart_tools_ratio : (target_tools : ℚ) / walmart_tools = 7 / 3 :=
by sorry

end target_more_tools_than_walmart_target_to_walmart_tools_ratio_l176_176515


namespace babylon_game_proof_l176_176286

section BabylonGame

-- Defining the number of holes on the sphere
def number_of_holes : Nat := 26

-- The number of 45° angles formed by the pairs of rays
def num_45_degree_angles : Nat := 40

-- The number of 60° angles formed by the pairs of rays
def num_60_degree_angles : Nat := 48

-- The other angles that can occur between pairs of rays
def other_angles : List Real := [31.4, 81.6, 90]

-- Constructs possible given the conditions
def constructible (shape : String) : Bool :=
  shape = "regular tetrahedron" ∨ shape = "regular octahedron"

-- Constructs not possible given the conditions
def non_constructible (shape : String) : Bool :=
  shape = "joined regular tetrahedrons"

-- Proof problem statement
theorem babylon_game_proof :
  (number_of_holes = 26) →
  (num_45_degree_angles = 40) →
  (num_60_degree_angles = 48) →
  (other_angles = [31.4, 81.6, 90]) →
  (constructible "regular tetrahedron" = True) →
  (constructible "regular octahedron" = True) →
  (non_constructible "joined regular tetrahedrons" = True) :=
  by
    sorry

end BabylonGame

end babylon_game_proof_l176_176286


namespace find_number_l176_176530

theorem find_number (x : ℕ) (h : x / 4 + 3 = 5) : x = 8 :=
by sorry

end find_number_l176_176530


namespace sprint_team_total_miles_l176_176262

-- Define the number of people and miles per person as constants
def numberOfPeople : ℕ := 250
def milesPerPerson : ℝ := 7.5

-- Assertion to prove the total miles
def totalMilesRun : ℝ := numberOfPeople * milesPerPerson

-- Proof statement
theorem sprint_team_total_miles : totalMilesRun = 1875 := 
by 
  -- Proof to be filled in
  sorry

end sprint_team_total_miles_l176_176262


namespace parabola_problem_l176_176153

theorem parabola_problem :
  let F := (1 : ℝ, 0 : ℝ) in
  let B := (3 : ℝ, 0 : ℝ) in
  ∃ A : ℝ × ℝ,
    (A.fst * A.fst = A.snd * 4) ∧
    dist A F = dist B F ∧
    dist A B = 2 * Real.sqrt 2 :=
by
  sorry

end parabola_problem_l176_176153


namespace largest_number_of_stores_visited_l176_176102

-- Definitions of the conditions
def num_stores := 7
def total_visits := 21
def num_shoppers := 11
def two_stores_visitors := 7
def at_least_one_store (n : ℕ) : Prop := n ≥ 1

-- The goal statement
theorem largest_number_of_stores_visited :
  ∃ n, n ≤ num_shoppers ∧ 
       at_least_one_store n ∧ 
       (n * 2 + (num_shoppers - n)) <= total_visits ∧ 
       (num_shoppers - n) ≥ 3 → 
       n = 4 :=
sorry

end largest_number_of_stores_visited_l176_176102


namespace sum_a5_a6_a7_l176_176720

variable (a : ℕ → ℝ) (q : ℝ)

-- Assumptions
axiom geometric_sequence : ∀ n, a (n + 1) = a n * q

axiom sum_a1_a2_a3 : a 1 + a 2 + a 3 = 1
axiom sum_a2_a3_a4 : a 2 + a 3 + a 4 = 2

-- The theorem we want to prove
theorem sum_a5_a6_a7 : a 5 + a 6 + a 7 = 16 := sorry

end sum_a5_a6_a7_l176_176720


namespace sum_gn_eq_one_third_l176_176063

noncomputable def g (n : ℕ) : ℝ :=
  ∑' i : ℕ, if i ≥ 3 then 1 / (i ^ n) else 0

theorem sum_gn_eq_one_third :
  (∑' n : ℕ, if n ≥ 3 then g n else 0) = 1 / 3 := 
by sorry

end sum_gn_eq_one_third_l176_176063


namespace algebra_ineq_a2_b2_geq_2_l176_176392

theorem algebra_ineq_a2_b2_geq_2
  (a b : ℝ)
  (h1 : a^3 - b^3 = 2)
  (h2 : a^5 - b^5 ≥ 4) :
  a^2 + b^2 ≥ 2 :=
by
  sorry

end algebra_ineq_a2_b2_geq_2_l176_176392


namespace sum_of_largest_three_l176_176041

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176041


namespace snow_probability_first_week_l176_176970

theorem snow_probability_first_week :
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3) = 3 / 32 →
  (1 - (p_no_snow_first_four_days ^ 4) * (p_no_snow_next_three_days ^ 3)) = 29 / 32 :=
by
  let p_snow_first_four_days := 1 / 4
  let p_no_snow_first_four_days := 1 - p_snow_first_four_days
  let p_snow_next_three_days := 1 / 3
  let p_no_snow_next_three_days := 1 - p_snow_next_three_days
  sorry

end snow_probability_first_week_l176_176970


namespace determine_pairs_l176_176432

noncomputable def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem determine_pairs (n p : ℕ) (hn_pos : 0 < n) (hp_prime : is_prime p) (hn_le_2p : n ≤ 2 * p) (divisibility : n^p - 1 ∣ (p - 1)^n + 1):
  (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) ∨ (n = 1 ∧ is_prime p) :=
by
  sorry

end determine_pairs_l176_176432


namespace johnnys_age_l176_176519

theorem johnnys_age (x : ℤ) (h : x + 2 = 2 * (x - 3)) : x = 8 := sorry

end johnnys_age_l176_176519


namespace paintable_wall_area_l176_176226

theorem paintable_wall_area :
  let bedrooms := 4
  let length := 14
  let width := 11
  let height := 9
  let doorway_window_area := 70
  let area_one_bedroom := 
    2 * (length * height) + 2 * (width * height) - doorway_window_area
  let total_paintable_area := bedrooms * area_one_bedroom
  total_paintable_area = 1520 := by
  sorry

end paintable_wall_area_l176_176226


namespace primary_schools_to_be_selected_l176_176094

noncomputable def total_schools : ℕ := 150 + 75 + 25
noncomputable def proportion_primary : ℚ := 150 / total_schools
noncomputable def selected_primary : ℚ := proportion_primary * 30

theorem primary_schools_to_be_selected : selected_primary = 18 :=
by sorry

end primary_schools_to_be_selected_l176_176094


namespace expectedValueProof_l176_176313

-- Definition of the problem conditions
def veryNormalCoin {n : ℕ} : Prop :=
  ∀ t : ℕ, (5 < t → (t - 5) = n → (t+1 = t + 1)) ∧ (t ≤ 5 ∨ n = t)

-- Definition of the expected value calculation
def expectedValue (n : ℕ) : ℚ :=
  if n > 0 then (1/2)^n else 0

-- Expected value for the given problem
def expectedValueProblem : ℚ := 
  let a1 := -2/683
  let expectedFirstFlip := 1/2 - 1/(2 * 683)
  100 * 341 + 683

-- Main statement to prove
theorem expectedValueProof : expectedValueProblem = 34783 := 
  sorry -- Proof omitted

end expectedValueProof_l176_176313


namespace eccentricity_range_l176_176562

noncomputable def ellipse_eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) (e : ℝ) : Prop :=
  ∃ c : ℝ, c^2 = a^2 - b^2 ∧ e = c / a ∧ (2 * ((-a) * (c + a / 2) - (b / 2) * b) + b^2 + c^2 ≥ 0)

theorem eccentricity_range (a b : ℝ) (h : a > b ∧ b > 0) :
  ∃ e : ℝ, ellipse_eccentricity_range a b h e ∧ (0 < e ∧ e ≤ -1 + Real.sqrt 3) :=
sorry

end eccentricity_range_l176_176562


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176879

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176879


namespace pauline_convertibles_l176_176977

theorem pauline_convertibles : 
  ∀ (total_cars regular_percentage truck_percentage sedan_percentage sports_percentage suv_percentage : ℕ),
  total_cars = 125 →
  regular_percentage = 38 →
  truck_percentage = 12 →
  sedan_percentage = 17 →
  sports_percentage = 22 →
  suv_percentage = 6 →
  (total_cars - (regular_percentage * total_cars / 100 + truck_percentage * total_cars / 100 + sedan_percentage * total_cars / 100 + sports_percentage * total_cars / 100 + suv_percentage * total_cars / 100)) = 8 :=
by
  intros
  sorry

end pauline_convertibles_l176_176977


namespace distance_AB_l176_176125

-- Declare the main entities involved
variable (A B F : Point)
variable (parabola : Point → Prop)

-- Define what it means to be on the parabola C: y^2 = 4x
def on_parabola (P : Point) : Prop := P.2^2 = 4 * P.1

-- Declare the specific points of interest
def point_A := A
def point_B : Point := ⟨3, 0⟩
def focus_F : Point := ⟨1, 0⟩

-- Define the distance function
def dist (P Q : Point) : ℝ := (P.1 - Q.1)^2 + (P.2 - Q.2)^2

-- Problem statement in Lean
theorem distance_AB (hA_on_C : on_parabola A) (h_AF_eq_BF : dist A F = dist B F) : dist A B = 8 :=
by sorry

end distance_AB_l176_176125


namespace max_band_members_l176_176997

theorem max_band_members (n : ℤ) (h1 : 20 * n % 31 = 11) (h2 : 20 * n < 1200) : 20 * n = 1100 :=
sorry

end max_band_members_l176_176997


namespace cyclic_inequality_l176_176565

theorem cyclic_inequality
    (x1 x2 x3 x4 x5 : ℝ)
    (h1 : 0 < x1)
    (h2 : 0 < x2)
    (h3 : 0 < x3)
    (h4 : 0 < x4)
    (h5 : 0 < x5) :
    (x1 + x2 + x3 + x4 + x5)^2 > 4 * (x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x1) :=
by
  sorry

end cyclic_inequality_l176_176565


namespace radian_measure_of_sector_l176_176076

-- Lean statement for the proof problem
theorem radian_measure_of_sector (R : ℝ) (hR : 0 < R) (h_area : (1 / 2) * (2 : ℝ) * R^2 = R^2) : 
  (2 : ℝ) = 2 :=
by 
  sorry
 
end radian_measure_of_sector_l176_176076


namespace intersection_M_N_l176_176729

def M : Set ℝ := {x | |x| ≤ 2}
def N : Set ℝ := {x | x^2 + 2 * x - 3 ≤ 0}
def intersection : Set ℝ := {x | -2 ≤ x ∧ x ≤ 1}

theorem intersection_M_N : M ∩ N = intersection := by
  sorry

end intersection_M_N_l176_176729


namespace inequality_true_l176_176274

theorem inequality_true (x : ℝ) : x^2 + 1 ≥ 2 * |x| :=
by
  sorry

end inequality_true_l176_176274


namespace largest_expression_l176_176952

def U := 2 * 2004^2005
def V := 2004^2005
def W := 2003 * 2004^2004
def X := 2 * 2004^2004
def Y := 2004^2004
def Z := 2004^2003

theorem largest_expression :
  U - V > V - W ∧
  U - V > W - X ∧
  U - V > X - Y ∧
  U - V > Y - Z :=
by
  sorry

end largest_expression_l176_176952


namespace prove_inequalities_l176_176732

theorem prove_inequalities (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^3 * b > a * b^3 ∧ a - b / a > b - a / b :=
by
  sorry

end prove_inequalities_l176_176732


namespace probability_three_heads_l176_176648

theorem probability_three_heads : 
  let p := (1/2 : ℝ) in
  (p * p * p) = (1/8 : ℝ) :=
by
  sorry

end probability_three_heads_l176_176648


namespace AB_distance_l176_176194

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l176_176194


namespace student_count_before_new_student_l176_176609

variable {W : ℝ} -- total weight of students before the new student joined
variable {n : ℕ} -- number of students before the new student joined
variable {W_new : ℝ} -- total weight including the new student
variable {n_new : ℕ} -- number of students including the new student

theorem student_count_before_new_student 
  (h1 : W = n * 28) 
  (h2 : W_new = W + 7) 
  (h3 : n_new = n + 1) 
  (h4 : W_new / n_new = 27.3) : n = 29 := 
by
  sorry

end student_count_before_new_student_l176_176609


namespace chocolate_bars_in_large_box_l176_176829

theorem chocolate_bars_in_large_box
  (number_of_small_boxes : ℕ)
  (chocolate_bars_per_box : ℕ)
  (h1 : number_of_small_boxes = 21)
  (h2 : chocolate_bars_per_box = 25) :
  number_of_small_boxes * chocolate_bars_per_box = 525 :=
by {
  sorry
}

end chocolate_bars_in_large_box_l176_176829


namespace smallest_integer_k_distinct_real_roots_l176_176556

theorem smallest_integer_k_distinct_real_roots :
  ∃ k : ℤ, (∀ x : ℝ, x^2 - x + 2 - k = 0 → x ≠ 0) ∧ k = 2 :=
by
  sorry

end smallest_integer_k_distinct_real_roots_l176_176556


namespace abs_m_minus_n_l176_176348

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l176_176348


namespace find_a_l176_176909

noncomputable def binomial_coefficient (n k : ℕ) : ℤ :=
if k ≤ n then nat.choose n k else 0

theorem find_a (a : ℝ) (h : binomial_coefficient 5 2 * a^3 = 80) : a = 2 :=
by sorry

end find_a_l176_176909


namespace proof_problem_l176_176985

theorem proof_problem 
  (A a B b : ℝ) 
  (h1 : |A - 3 * a| ≤ 1 - a) 
  (h2 : |B - 3 * b| ≤ 1 - b) 
  (h3 : 0 < a) 
  (h4 : 0 < b) :
  (|((A * B) / 3) - 3 * (a * b)|) - 3 * (a * b) ≤ 1 - (a * b) :=
sorry

end proof_problem_l176_176985


namespace arithmetic_mean_reciprocals_primes_l176_176868

theorem arithmetic_mean_reciprocals_primes :
  let p1 := 2
  let p2 := 3
  let p3 := 5
  let p4 := 7
  let rec1 := (1:ℚ) / p1
  let rec2 := (1:ℚ) / p2
  let rec3 := (1:ℚ) / p3
  let rec4 := (1:ℚ) / p4
  (rec1 + rec2 + rec3 + rec4) / 4 = 247 / 840 := by
  sorry

end arithmetic_mean_reciprocals_primes_l176_176868


namespace percent_owning_only_cats_l176_176582

theorem percent_owning_only_cats (total_students dogs cats both : ℕ) (h1 : total_students = 500)
  (h2 : dogs = 150) (h3 : cats = 80) (h4 : both = 25) : (cats - both) / total_students * 100 = 11 :=
by
  sorry

end percent_owning_only_cats_l176_176582


namespace sum_of_three_largest_of_consecutive_numbers_l176_176007

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176007


namespace length_of_bridge_l176_176301

/-- Prove the length of the bridge -/
theorem length_of_bridge (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 120 →
  train_speed_kmph = 70 →
  crossing_time_sec = 13.884603517432893 →
  (70 * (1000 / 3600) * 13.884603517432893 - 120 = 150) :=
by
  intros h1 h2 h3
  sorry

end length_of_bridge_l176_176301


namespace evaluate_expression_l176_176851

theorem evaluate_expression : (7^(1/4) / 7^(1/7)) = 7^(3/28) := 
by sorry

end evaluate_expression_l176_176851


namespace find_g1_l176_176714

open Function

-- Definitions based on the conditions
def f (g : ℝ → ℝ) (x : ℝ) : ℝ := g x + x^2

theorem find_g1 (g : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, f g (-x) + f g x = 0) 
  (h2 : g (-1) = 1) 
  : g 1 = -3 :=
sorry

end find_g1_l176_176714


namespace cubic_polynomial_solution_l176_176546

noncomputable def q (x : ℚ) : ℚ := (51/13) * x^3 + (-31/13) * x^2 + (16/13) * x + (3/13)

theorem cubic_polynomial_solution : 
  q 1 = 3 ∧ q 2 = 23 ∧ q 3 = 81 ∧ q 5 = 399 :=
by {
  sorry
}

end cubic_polynomial_solution_l176_176546


namespace triangle_min_area_l176_176527

theorem triangle_min_area :
  ∃ (p q : ℤ), (p, q).fst = 3 ∧ (p, q).snd = 3 ∧ 1/2 * |18 * p - 30 * q| = 3 := 
sorry

end triangle_min_area_l176_176527


namespace average_class_score_l176_176357

theorem average_class_score : 
  ∀ (n total score_per_100 score_per_0 avg_rest : ℕ), 
  n = 20 → 
  total = 800 → 
  score_per_100 = 2 → 
  score_per_0 = 3 → 
  avg_rest = 40 → 
  ((score_per_100 * 100 + score_per_0 * 0 + (n - (score_per_100 + score_per_0)) * avg_rest) / n = 40)
:= by
  intros n total score_per_100 score_per_0 avg_rest h_n h_total h_100 h_0 h_rest
  sorry

end average_class_score_l176_176357


namespace least_pos_integer_with_8_factors_l176_176517

theorem least_pos_integer_with_8_factors : 
  ∃ k : ℕ, (k > 0 ∧ ((∃ m n p q : ℕ, k = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, k = p^7 ∧ Prime p)) ∧ 
            ∀ l : ℕ, (l > 0 ∧ ((∃ m n p q : ℕ, l = p^m * q^n ∧ p ≠ q ∧ Prime p ∧ Prime q ∧ m + 1 = 4 ∧ n + 1 = 2) 
                     ∨ (∃ p : ℕ, l = p^7 ∧ Prime p)) → k ≤ l)) ∧ k = 24 :=
sorry

end least_pos_integer_with_8_factors_l176_176517


namespace parabola_distance_l176_176127

theorem parabola_distance
  (A B F : ℝ × ℝ)
  (hF : F = (1, 0))
  (hB : B = (3, 0))
  (hC : A.1 * A.1 = A.2 * 4)
  (hDist : Real.sqrt ((A.1 - F.1)^2 + (A.2 - F.2)^2) = Real.sqrt ((B.1 - F.1)^2 + (B.2 - F.2)^2)):
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 2 := 
by
  sorry

end parabola_distance_l176_176127


namespace product_neg_int_add_five_l176_176734

theorem product_neg_int_add_five:
  let x := -11 
  let y := -8 
  x * y + 5 = 93 :=
by
  -- Proof omitted
  sorry

end product_neg_int_add_five_l176_176734


namespace sum_of_integers_square_greater_272_l176_176807

theorem sum_of_integers_square_greater_272 (x : ℤ) (h : x^2 = x + 272) :
  ∃ (roots : List ℤ), (roots = [17, -16]) ∧ (roots.sum = 1) :=
sorry

end sum_of_integers_square_greater_272_l176_176807


namespace inequality_am_gm_l176_176780

theorem inequality_am_gm 
  (a b c : ℝ) : a^2 + b^2 + c^2 ≥ a*b + a*c + b*c := 
by 
  sorry

end inequality_am_gm_l176_176780


namespace product_of_ys_l176_176435

theorem product_of_ys (x y : ℤ) (h1 : x^3 + y^2 - 3 * y + 1 < 0)
                                     (h2 : 3 * x^3 - y^2 + 3 * y > 0) : 
  (y = 1 ∨ y = 2) → (1 * 2 = 2) :=
by {
  sorry
}

end product_of_ys_l176_176435


namespace exponential_inequality_l176_176472

-- Define the problem conditions and the proof goal
theorem exponential_inequality (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_eq : Real.exp a + 2 * a = Real.exp b + 3 * b) : a > b := 
sorry

end exponential_inequality_l176_176472


namespace correct_calculation_l176_176273

variable (a : ℝ) -- assuming a ∈ ℝ

theorem correct_calculation : (a ^ 3) ^ 2 = a ^ 6 :=
by {
  sorry
}

end correct_calculation_l176_176273


namespace sum_of_largest_three_consecutive_numbers_l176_176059

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176059


namespace arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176882

theorem arithmetic_mean_of_reciprocals_of_first_four_primes :
  arithmetic_mean ([2, 3, 5, 7].map (λ p, 1 / (p : ℚ))) = 247 / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_of_first_four_primes_l176_176882


namespace sum_of_largest_three_consecutive_numbers_l176_176062

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176062


namespace C_gets_more_than_D_l176_176300

-- Define the conditions
def proportion_B := 3
def share_B : ℕ := 3000
def proportion_C := 5
def proportion_D := 4

-- Define the parts based on B's share
def part_value := share_B / proportion_B

-- Define the shares based on the proportions
def share_C := proportion_C * part_value
def share_D := proportion_D * part_value

-- Prove the final statement about the difference
theorem C_gets_more_than_D : share_C - share_D = 1000 :=
by
  -- Proof goes here
  sorry

end C_gets_more_than_D_l176_176300


namespace arithmetic_mean_of_reciprocals_is_correct_l176_176883

/-- The first four prime numbers -/
def first_four_primes : List ℕ := [2, 3, 5, 7]

/-- Taking reciprocals and summing them up  -/
def reciprocals_sum : ℚ :=
  (1/2) + (1/3) + (1/5) + (1/7)

/-- The arithmetic mean of the reciprocals  -/
def arithmetic_mean_of_reciprocals :=
  reciprocals_sum / 4

/-- The result of the arithmetic mean of the reciprocals  -/
theorem arithmetic_mean_of_reciprocals_is_correct :
  arithmetic_mean_of_reciprocals = 247/840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_is_correct_l176_176883


namespace total_number_of_cards_l176_176263

/-- There are 9 playing cards and 4 ID cards initially.
If you add 6 more playing cards and 3 more ID cards,
then the total number of playing cards and ID cards will be 22. -/
theorem total_number_of_cards :
  let initial_playing_cards := 9
  let initial_id_cards := 4
  let additional_playing_cards := 6
  let additional_id_cards := 3
  let total_playing_cards := initial_playing_cards + additional_playing_cards
  let total_id_cards := initial_id_cards + additional_id_cards
  let total_cards := total_playing_cards + total_id_cards
  total_cards = 22 :=
by
  sorry

end total_number_of_cards_l176_176263


namespace distance_AB_l176_176199

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176199


namespace binary_divisible_by_136_l176_176484

theorem binary_divisible_by_136 :
  let N := 2^139 + 2^105 + 2^15 + 2^13
  N % 136 = 0 :=
by {
  let N := 2^139 + 2^105 + 2^15 + 2^13;
  sorry
}

end binary_divisible_by_136_l176_176484


namespace sum_of_three_largest_of_consecutive_numbers_l176_176008

theorem sum_of_three_largest_of_consecutive_numbers (n : ℕ) :
  (n + (n + 1) + (n + 2) = 60) → ((n + 2) + (n + 3) + (n + 4) = 66) :=
by
  -- Given the conditions and expected result, we can break down the proof as follows:
  intros h1
  sorry

end sum_of_three_largest_of_consecutive_numbers_l176_176008


namespace percentage_difference_l176_176524

theorem percentage_difference :
  let a := 0.80 * 40
  let b := (4 / 5) * 15
  a - b = 20 := by
sorry

end percentage_difference_l176_176524


namespace hyperbola_condition_l176_176499

theorem hyperbola_condition (m : ℝ) : 
  (∀ x y : ℝ, (m-2) * (m+3) < 0 → (x^2) / (m-2) + (y^2) / (m+3) = 1) ↔ -3 < m ∧ m < 2 :=
by
  sorry

end hyperbola_condition_l176_176499


namespace minimum_containers_needed_l176_176481

-- Definition of the problem conditions
def container_sizes := [5, 10, 20]
def target_units := 85

-- Proposition stating the minimum number of containers required
theorem minimum_containers_needed : 
  ∃ (x y z : ℕ), 
    5 * x + 10 * y + 20 * z = target_units ∧ 
    x + y + z = 5 :=
sorry

end minimum_containers_needed_l176_176481


namespace ratio_meerkats_to_lion_cubs_l176_176533

-- Defining the initial conditions 
def initial_animals : ℕ := 68
def gorillas_sent : ℕ := 6
def hippo_adopted : ℕ := 1
def rhinos_rescued : ℕ := 3
def lion_cubs : ℕ := 8
def final_animal_count : ℕ := 90

-- Calculating the number of meerkats
def animals_before_meerkats : ℕ := initial_animals - gorillas_sent + hippo_adopted + rhinos_rescued + lion_cubs
def meerkats : ℕ := final_animal_count - animals_before_meerkats

-- Proving the ratio of meerkats to lion cubs is 2:1
theorem ratio_meerkats_to_lion_cubs : meerkats / lion_cubs = 2 := by
  -- Placeholder for the proof
  sorry

end ratio_meerkats_to_lion_cubs_l176_176533


namespace sarahs_brother_apples_l176_176486

theorem sarahs_brother_apples (x : ℝ) (hx : 5 * x = 45.0) : x = 9.0 :=
by
  sorry

end sarahs_brother_apples_l176_176486


namespace problem_l176_176186

noncomputable theory
open_locale real

def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def point_B : ℝ × ℝ := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem problem 
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (h_dist : distance A focus = distance point_B focus) : 
  distance A point_B = 2 * real.sqrt 2 :=
sorry

end problem_l176_176186


namespace hyperbola_foci_eccentricity_l176_176388

-- Definitions and conditions
def hyperbola_eq := (x y : ℝ) → (x^2 / 4) - (y^2 / 12) = 1

-- Proof goals: Coordinates of the foci and eccentricity
theorem hyperbola_foci_eccentricity (x y : ℝ) : 
  (∃ c : ℝ, (x^2 / 4) - (y^2 / 12) = 1 ∧ (x = 4 ∧ y = 0) ∨ (x = -4 ∧ y = 0)) ∧ 
  (∃ e : ℝ, e = 2) :=
sorry

end hyperbola_foci_eccentricity_l176_176388


namespace additional_investment_l176_176830

-- Given the conditions
variables (x y : ℝ)
def interest_rate_1 := 0.02
def interest_rate_2 := 0.04
def invested_amount := 1000
def total_interest := 92

-- Theorem to prove
theorem additional_investment : 
  0.02 * invested_amount + 0.04 * (invested_amount + y) = total_interest → 
  y = 800 :=
by
  sorry

end additional_investment_l176_176830


namespace exist_three_not_played_l176_176095

noncomputable def football_championship (teams : Finset ℕ) (rounds : ℕ) : Prop :=
  let pairs_per_round := teams.card / 2 in
  let total_pairs_constraint := rounds * pairs_per_round in
  let constraint_matrix := (teams.card - 1) - pairs_per_round in
  (teams.card = 18) ∧
  (rounds = 8) ∧
  (pairs_per_round * rounds < constraint_matrix * (constraint_matrix - 1) / 2) ->
  ∃ (A B C : ℕ) (hA : A ∈ teams) (hB : B ∈ teams) (hC : C ∈ teams),
    ¬ (⊢ (A, B) ∈ teams * teams) ∧ ¬ (⊢ (B, C) ∈ teams * teams) ∧ ¬ (⊢ (A, C) ∈ teams * teams)

theorem exist_three_not_played :
  ∃ (football_championship (Finset.range 18) 8) :=
begin
  sorry,
end

end exist_three_not_played_l176_176095


namespace midpoint_trace_quarter_circle_l176_176417

theorem midpoint_trace_quarter_circle (L : ℝ) (hL : 0 < L):
  ∃ (C : ℝ) (M : ℝ × ℝ → ℝ), 
    (∀ (x y : ℝ), x^2 + y^2 = L^2 → M (x, y) = C) ∧ 
    (C = (1/2) * L) ∧ 
    (∀ (x y : ℝ), M (x, y) = (x/2)^2 + (y/2)^2) → 
    ∀ (x y : ℝ), x^2 + y^2 = L^2 → (x/2)^2 + (y/2)^2 = (1/2 * L)^2 := 
by
  sorry

end midpoint_trace_quarter_circle_l176_176417


namespace jackson_entertainment_cost_l176_176108

def price_computer_game : ℕ := 66
def price_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3
def total_entertainment_cost : ℕ := price_computer_game + number_of_movie_tickets * price_movie_ticket

theorem jackson_entertainment_cost : total_entertainment_cost = 102 := by
  sorry

end jackson_entertainment_cost_l176_176108


namespace matrix_unique_solution_l176_176327

-- Definitions for the conditions given in the problem
def vec_i : Fin 3 → ℤ := ![1, 0, 0]
def vec_j : Fin 3 → ℤ := ![0, 1, 0]
def vec_k : Fin 3 → ℤ := ![0, 0, 1]

def matrix_M : Matrix (Fin 3) (Fin 3) ℤ := ![
  ![5, -3, 8],
  ![4, 6, -2],
  ![-9, 0, 5]
]

-- Define the target vectors
def target_i : Fin 3 → ℤ := ![5, 4, -9]
def target_j : Fin 3 → ℤ := ![-3, 6, 0]
def target_k : Fin 3 → ℤ := ![8, -2, 5]

-- The statement of the proof
theorem matrix_unique_solution : 
  (matrix_M.mulVec vec_i = target_i) ∧
  (matrix_M.mulVec vec_j = target_j) ∧
  (matrix_M.mulVec vec_k = target_k) :=
  by {
    sorry
  }

end matrix_unique_solution_l176_176327


namespace max_value_is_one_l176_176953

noncomputable def max_expression (a b : ℝ) : ℝ :=
(a + b) ^ 2 / (a ^ 2 + 2 * a * b + b ^ 2)

theorem max_value_is_one {a b : ℝ} (ha : 0 < a) (hb : 0 < b) :
  max_expression a b ≤ 1 :=
sorry

end max_value_is_one_l176_176953


namespace AB_distance_l176_176196

noncomputable def parabola_focus : (ℝ × ℝ) := (1, 0)

def parabola_equation (p : ℝ × ℝ) : Prop :=
  p.2^2 = 4 * p.1

def B : (ℝ × ℝ) := (3, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def A_condition (A : ℝ × ℝ) : Prop :=
  parabola_equation A ∧ distance A parabola_focus = 2

theorem AB_distance (A : ℝ × ℝ) (hA : A_condition A) : distance A B = 2 * real.sqrt 2 :=
  sorry

end AB_distance_l176_176196


namespace sum_S_17_33_50_l176_176087

def S (n : ℕ) : ℤ :=
  if n % 2 = 0 then - (n / 2)
  else (n / 2) + 1

theorem sum_S_17_33_50 : (S 17) + (S 33) + (S 50) = 1 := by
  sorry

end sum_S_17_33_50_l176_176087


namespace perp_lines_iff_m_values_l176_176713

section
variables (m x y : ℝ)

def l1 := (m * x + y - 2 = 0)
def l2 := ((m + 1) * x - 2 * m * y + 1 = 0)

theorem perp_lines_iff_m_values (h1 : l1 m x y) (h2 : l2 m x y) (h_perp : (m * (m + 1) + (-2 * m) = 0)) : m = 0 ∨ m = 1 :=
by {
  sorry
}
end

end perp_lines_iff_m_values_l176_176713


namespace vowel_initial_probability_is_correct_l176_176963

-- Given conditions as definitions
def total_students : ℕ := 34
def vowels : List Char := ['A', 'E', 'I', 'O', 'U', 'Y']
def vowels_count_per_vowel : ℕ := 2
def total_vowels_count := vowels.length * vowels_count_per_vowel

-- The probabilistic statement we want to prove
def vowel_probability : ℚ := total_vowels_count / total_students

-- The final statement to prove
theorem vowel_initial_probability_is_correct :
  vowel_probability = 6 / 17 :=
by
  unfold vowel_probability total_vowels_count
  -- Simplification to verify our statement.
  sorry

end vowel_initial_probability_is_correct_l176_176963


namespace certain_number_is_7000_l176_176272

theorem certain_number_is_7000 (x : ℕ) (h1 : 1 / 10 * (1 / 100 * x) = x / 1000)
    (h2 : 1 / 10 * x = x / 10)
    (h3 : x / 10 - x / 1000 = 693) : 
  x = 7000 := 
sorry

end certain_number_is_7000_l176_176272


namespace find_common_ratio_limit_SN_over_TN_l176_176914

noncomputable def S (q : ℚ) (n : ℕ) : ℚ := (1 - q^n) / (1 - q)
noncomputable def T (q : ℚ) (n : ℕ) : ℚ := (1 - q^(2 * n)) / (1 - q^2)

theorem find_common_ratio
  (S3 : S q 3 = 3)
  (S6 : S q 6 = -21) :
  q = -2 :=
sorry

theorem limit_SN_over_TN
  (q_pos : 0 < q)
  (Tn_def : ∀ n, T q n = 1) :
  (q > 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 0| < ε) ∧
  (0 < q ∧ q < 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - (1 + q)| < ε) ∧
  (q = 1 → ∀ ε > 0, ∃ N, ∀ n ≥ N, |S q n / T q n - 1| < ε) :=
sorry

end find_common_ratio_limit_SN_over_TN_l176_176914


namespace greatest_of_consecutive_even_numbers_l176_176654

theorem greatest_of_consecutive_even_numbers (n : ℤ) (h : ((n - 4) + (n - 2) + n + (n + 2) + (n + 4)) / 5 = 35) : n + 4 = 39 :=
by
  sorry

end greatest_of_consecutive_even_numbers_l176_176654


namespace carrey_fixed_amount_l176_176317

theorem carrey_fixed_amount :
  ∃ C : ℝ, 
    (C + 0.25 * 44.44444444444444 = 24 + 0.16 * 44.44444444444444) →
    C = 20 :=
by
  sorry

end carrey_fixed_amount_l176_176317


namespace find_sin_theta_l176_176202

noncomputable def direction_vector : ℝ^3 := ![4, 5, 7]
noncomputable def normal_vector : ℝ^3 := ![5, -3, 9]

noncomputable def dot_product (u v : ℝ^3) : ℝ :=
u 0 * v 0 + u 1 * v 1 + u 2 * v 2

noncomputable def magnitude (v : ℝ^3) : ℝ :=
real.sqrt (v 0 ^ 2 + v 1 ^ 2 + v 2 ^ 2)

noncomputable def angle_sin := 
dot_product direction_vector normal_vector / 
(magnitude direction_vector * magnitude normal_vector)

theorem find_sin_theta : angle_sin = 68 / real.sqrt 10350 :=
by
  sorry

end find_sin_theta_l176_176202


namespace parabola_problem_l176_176166

noncomputable def parabola_focus := (1 : ℝ, 0 : ℝ)

def parabola (x : ℝ) (y : ℝ) : Prop := y^2 = 4 * x

def point_B := (3 : ℝ, 0 : ℝ)

def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_problem
  (A : ℝ × ℝ)
  (hA : parabola A.1 A.2)
  (hAF_BF : distance A parabola_focus = distance point_B parabola_focus) :
  distance A point_B = 2 * Real.sqrt 2 :=
sorry

end parabola_problem_l176_176166


namespace distance_AB_l176_176200

def focus_of_parabola (a : ℝ) : ℝ × ℝ :=
  (a / 4, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def is_on_parabola (A : ℝ × ℝ) (a : ℝ) : Prop :=
  A.2 ^ 2 = 4 * A.1

noncomputable def point_B : ℝ × ℝ := (3, 0)

theorem distance_AB {A : ℝ × ℝ} (cond_A : is_on_parabola A 4) 
  (h : distance A (focus_of_parabola 4) = distance point_B (focus_of_parabola 4)) : 
  distance A point_B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176200


namespace sum_of_three_largest_l176_176037

theorem sum_of_three_largest (n : ℕ) 
  (h1 : n + (n + 1) + (n + 2) = 60) : 
  (n + 2) + (n + 3) + (n + 4) = 66 :=
by
  sorry

end sum_of_three_largest_l176_176037


namespace total_spent_on_entertainment_l176_176109

def cost_of_computer_game : ℕ := 66
def cost_of_one_movie_ticket : ℕ := 12
def number_of_movie_tickets : ℕ := 3

theorem total_spent_on_entertainment : cost_of_computer_game + cost_of_one_movie_ticket * number_of_movie_tickets = 102 := 
by sorry

end total_spent_on_entertainment_l176_176109


namespace find_quotient_l176_176216

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h1 : dividend = 23) (h2 : divisor = 4) (h3 : remainder = 3)
  (h4 : dividend = (divisor * quotient) + remainder) : quotient = 5 :=
sorry

end find_quotient_l176_176216


namespace coefficient_x2_expansion_l176_176496

theorem coefficient_x2_expansion : 
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  (expansion_coeff 1 (-2) 4 2) = 24 :=
by
  let binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k
  let expansion_coeff (a b : ℤ) (n k : ℕ) : ℤ := (b ^ k) * (binomial_coeff n k) * (a ^ (n - k))
  have coeff : ℤ := expansion_coeff 1 (-2) 4 2
  sorry -- Proof goes here

end coefficient_x2_expansion_l176_176496


namespace parabola_distance_l176_176140

theorem parabola_distance
    (F : ℝ × ℝ)
    (A B : ℝ × ℝ)
    (hC : ∀ (x y : ℝ), (x, y)  ∈ A → y^2 = 4 * x)
    (hF : F = (1, 0))
    (hB : B = (3, 0))
    (hAF_eq_BF : dist F A = dist F B) :
    dist A B = 2 * Real.sqrt 2 := 
sorry

end parabola_distance_l176_176140


namespace number_of_ways_to_take_pieces_l176_176396

theorem number_of_ways_to_take_pieces : 
  (Nat.choose 6 4) = 15 := 
by
  sorry

end number_of_ways_to_take_pieces_l176_176396


namespace arithmetic_mean_reciprocals_first_four_primes_l176_176906

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l176_176906


namespace rowing_upstream_speed_l176_176414

theorem rowing_upstream_speed (V_down V_m : ℝ) (h_down : V_down = 35) (h_still : V_m = 31) : ∃ V_up, V_up = V_m - (V_down - V_m) ∧ V_up = 27 := by
  sorry

end rowing_upstream_speed_l176_176414


namespace parabola_distance_l176_176169

theorem parabola_distance (A B: ℝ × ℝ)
  (hA_on_parabola : A.2^2 = 4 * A.1)
  (hB_coord : B = (3, 0))
  (hF_focus : ∃ F : ℝ × ℝ, F = (1, 0) ∧ dist A F = dist B F) :
  dist A B = 2 * Real.sqrt 2 :=
sorry

end parabola_distance_l176_176169


namespace decreasing_power_function_on_interval_l176_176342

noncomputable def power_function (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(m^2 + m - 3)

theorem decreasing_power_function_on_interval (m : ℝ) :
  (∀ x : ℝ, (0 < x) -> power_function m x < 0) ↔ m = -1 := 
by 
  sorry

end decreasing_power_function_on_interval_l176_176342


namespace ratio_of_girls_participated_to_total_l176_176614

noncomputable def ratio_participating_girls {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ℚ :=
    girls_participated / (girls_participated + boys_participated)

theorem ratio_of_girls_participated_to_total {a : ℕ} (h1 : a > 0)
    (equal_boys_girls : ∀ (b g : ℕ), b = a ∧ g = a)
    (girls_participated : ℕ := (3 * a) / 4)
    (boys_participated : ℕ := (2 * a) / 3) :
    ratio_participating_girls h1 equal_boys_girls girls_participated boys_participated = 9 / 17 :=
by
    sorry

end ratio_of_girls_participated_to_total_l176_176614


namespace sum_of_largest_three_l176_176045

theorem sum_of_largest_three (n : ℕ) (h : n + (n+1) + (n+2) = 60) : 
  (n+2) + (n+3) + (n+4) = 66 :=
sorry

end sum_of_largest_three_l176_176045


namespace minimum_possible_value_of_BC_l176_176725

def triangle_ABC_side_lengths_are_integers (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0

def angle_A_is_twice_angle_B (A B C : ℝ) : Prop :=
  A = 2 * B

def CA_is_nine (CA : ℕ) : Prop :=
  CA = 9

theorem minimum_possible_value_of_BC
  (a b c : ℕ) (A B C : ℝ) (CA : ℕ)
  (h1 : triangle_ABC_side_lengths_are_integers a b c)
  (h2 : angle_A_is_twice_angle_B A B C)
  (h3 : CA_is_nine CA) :
  ∃ (BC : ℕ), BC = 12 := 
sorry

end minimum_possible_value_of_BC_l176_176725


namespace bubble_gum_cost_l176_176928

theorem bubble_gum_cost (n_pieces : ℕ) (total_cost : ℕ) (cost_per_piece : ℕ) 
  (h1 : n_pieces = 136) (h2 : total_cost = 2448) : cost_per_piece = 18 :=
by
  sorry

end bubble_gum_cost_l176_176928


namespace distance_AB_l176_176177

open Classical Real

theorem distance_AB (A B F : ℝ × ℝ) (hA : A.2 ^ 2 = 4 * A.1) (B_def : B = (3, 0)) (F_def : F = (1, 0)) (AF_eq_BF : dist A F = dist B F) : dist A B = 2 * Real.sqrt 2 := by
  sorry

end distance_AB_l176_176177


namespace cost_to_feed_treats_for_a_week_l176_176838

theorem cost_to_feed_treats_for_a_week :
  (let dog_biscuits_cost := 4 * 0.25 in
   let rawhide_bones_cost := 2 * 1 in
   let daily_cost := dog_biscuits_cost + rawhide_bones_cost in
   7 * daily_cost = 21) :=
by
  sorry

end cost_to_feed_treats_for_a_week_l176_176838


namespace true_proposition_is_A_l176_176717

-- Define the propositions
def l1 := ∀ (x y : ℝ), x - 2 * y + 3 = 0
def l2 := ∀ (x y : ℝ), 2 * x + y + 3 = 0
def p : Prop := ¬(l1 ∧ l2 ∧ ¬(∃ (x y : ℝ), x - 2 * y + 3 = 0 ∧ 2 * x + y + 3 = 0 ∧ (1 * 2 + (-2) * 1 ≠ 0)))
def q : Prop := ∃ x₀ : ℝ, (0 < x₀) ∧ (x₀ + 2 > Real.exp x₀)

-- The proof problem statement
theorem true_proposition_is_A : (¬p) ∧ q :=
by
  sorry

end true_proposition_is_A_l176_176717


namespace initial_printing_presses_l176_176361

theorem initial_printing_presses (P : ℕ) 
  (h1 : 500000 / (9 * P) = 500000 / (12 * 30)) : 
  P = 40 :=
by
  sorry

end initial_printing_presses_l176_176361


namespace sum_of_three_largest_l176_176015

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176015


namespace montoya_family_budget_on_food_l176_176494

def spending_on_groceries : ℝ := 0.6
def spending_on_eating_out : ℝ := 0.2

theorem montoya_family_budget_on_food :
  spending_on_groceries + spending_on_eating_out = 0.8 :=
  by
  sorry

end montoya_family_budget_on_food_l176_176494


namespace sum_of_three_largest_l176_176020

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176020


namespace radius_of_tangent_circle_l176_176291

def is_tangent_coor_axes_and_leg (r : ℝ) : Prop :=
  -- Circle with radius r is tangent to coordinate axes and one leg of the triangle
  ∃ O B C : ℝ × ℝ, 
  -- Conditions: centers and tangency
  O = (r, r) ∧ 
  B = (0, 2) ∧ 
  C = (2, 0) ∧ 
  r = 1

theorem radius_of_tangent_circle :
  ∀ r : ℝ, is_tangent_coor_axes_and_leg r → r = 1 :=
by
  sorry

end radius_of_tangent_circle_l176_176291


namespace proof_f_g_f3_l176_176204

def f (x: ℤ) : ℤ := 2*x + 5
def g (x: ℤ) : ℤ := 5*x + 2

theorem proof_f_g_f3 :
  f (g (f 3)) = 119 := by
  sorry

end proof_f_g_f3_l176_176204


namespace remaining_balance_is_correct_l176_176735

def total_price (deposit amount sales_tax_rate discount_rate service_charge P : ℝ) :=
  let sales_tax := sales_tax_rate * P
  let price_after_tax := P + sales_tax
  let discount := discount_rate * price_after_tax
  let price_after_discount := price_after_tax - discount
  let total_price := price_after_discount + service_charge
  total_price

theorem remaining_balance_is_correct (deposit : ℝ) (amount_paid : ℝ) (sales_tax_rate : ℝ) (discount_rate : ℝ) (service_charge : ℝ)
  (P : ℝ) : deposit = 0.10 * P →
         amount_paid = 110 →
         sales_tax_rate = 0.15 →
         discount_rate = 0.05 →
         service_charge = 50 →
         total_price deposit amount_paid sales_tax_rate discount_rate service_charge P - amount_paid = 1141.75 :=
by
  sorry

end remaining_balance_is_correct_l176_176735


namespace sum_of_largest_three_consecutive_numbers_l176_176060

theorem sum_of_largest_three_consecutive_numbers (n : ℕ) 
  (h : n + (n + 1) + (n + 2) = 60) : (n + 2) + (n + 3) + (n + 4) = 66 := 
by
  sorry

end sum_of_largest_three_consecutive_numbers_l176_176060


namespace marla_colors_green_squares_l176_176206

-- Condition 1: Grid dimensions
def num_rows : ℕ := 10
def num_cols : ℕ := 15

-- Condition 2: Red squares
def red_rows : ℕ := 4
def red_squares_per_row : ℕ := 6
def red_squares : ℕ := red_rows * red_squares_per_row

-- Condition 3: Blue rows (first 2 and last 2)
def blue_rows : ℕ := 2 + 2
def blue_squares_per_row : ℕ := num_cols
def blue_squares : ℕ := blue_rows * blue_squares_per_row

-- Derived information
def total_squares : ℕ := num_rows * num_cols
def non_green_squares : ℕ := red_squares + blue_squares

-- The Lemma to prove
theorem marla_colors_green_squares : total_squares - non_green_squares = 66 := by
  sorry

end marla_colors_green_squares_l176_176206


namespace area_of_set_U_l176_176370

noncomputable def five_presentable (z : ℂ) : Prop :=
  ∃ (w : ℂ), abs w = 5 ∧ z = w^2 - (1 / w^2)

def set_U : Set ℂ := {z | five_presentable z}

theorem area_of_set_U : measure_theory.measure.measure2.area (set_U) = 48 * real.pi := by
  sorry

end area_of_set_U_l176_176370


namespace beyonce_total_songs_l176_176543

-- Define the conditions as given in the problem
def singles : ℕ := 5
def albums_15_songs : ℕ := 2
def songs_per_15_album : ℕ := 15
def albums_20_songs : ℕ := 1
def songs_per_20_album : ℕ := 20

-- Define a function to calculate the total number of songs released by Beyonce
def total_songs_released : ℕ :=
  singles + (albums_15_songs * songs_per_15_album) + (albums_20_songs * songs_per_20_album)

-- Theorem statement for the total number of songs released
theorem beyonce_total_songs {singles albums_15_songs songs_per_15_album albums_20_songs songs_per_20_album : ℕ} :
  singles = 5 →
  albums_15_songs = 2 →
  songs_per_15_album = 15 →
  albums_20_songs = 1 →
  songs_per_20_album = 20 →
  total_songs_released = 55 :=
by {
  intros h_singles h_albums_15_songs h_songs_per_15_album h_albums_20_songs h_songs_per_20_album,
  -- replace with the proven result
  sorry
}

end beyonce_total_songs_l176_176543


namespace determine_g_l176_176789

-- Definitions of the given conditions
def f (x : ℝ) := x^2
def h1 (g : ℝ → ℝ) : Prop := f (g x) = 9 * x^2 - 6 * x + 1

-- The statement that needs to be proven
theorem determine_g (g : ℝ → ℝ) (H1 : h1 g) :
  g = (fun x => 3 * x - 1) ∨ g = (fun x => -3 * x + 1) :=
sorry

end determine_g_l176_176789


namespace oranges_in_bag_l176_176510

variables (O : ℕ)

def initial_oranges (O : ℕ) := O
def initial_tangerines := 17
def oranges_left_after_taking_away := O - 2
def tangerines_left_after_taking_away := 7
def tangerines_and_oranges_condition (O : ℕ) := 7 = (O - 2) + 4

theorem oranges_in_bag (O : ℕ) (h₀ : tangerines_and_oranges_condition O) : O = 5 :=
by
  sorry

end oranges_in_bag_l176_176510


namespace highest_percentage_without_car_l176_176773

noncomputable def percentage_without_car (total_percentage : ℝ) (car_percentage : ℝ) : ℝ :=
  total_percentage - total_percentage * car_percentage / 100

theorem highest_percentage_without_car :
  let A_total := 30
  let A_with_car := 25
  let B_total := 50
  let B_with_car := 15
  let C_total := 20
  let C_with_car := 35

  percentage_without_car A_total A_with_car = 22.5 /\
  percentage_without_car B_total B_with_car = 42.5 /\
  percentage_without_car C_total C_with_car = 13 /\
  percentage_without_car B_total B_with_car = max (percentage_without_car A_total A_with_car) (max (percentage_without_car B_total B_with_car) (percentage_without_car C_total C_with_car)) :=
by
  sorry

end highest_percentage_without_car_l176_176773


namespace arithmetic_mean_of_reciprocals_first_four_primes_l176_176855

theorem arithmetic_mean_of_reciprocals_first_four_primes : 
  let primes := [2, 3, 5, 7]
  let reciprocals := primes.map (λ p, 1 / (p:ℚ))
  let sum_reciprocals := reciprocals.sum
  let mean_reciprocals := sum_reciprocals / 4
  mean_reciprocals = (247:ℚ) / 840 :=
by
  sorry

end arithmetic_mean_of_reciprocals_first_four_primes_l176_176855


namespace sum_of_three_largest_l176_176018

theorem sum_of_three_largest :
  ∃ n : ℕ, (n + n.succ + n.succ.succ = 60) → ((n.succ.succ + n.succ.succ.succ + n.succ.succ.succ.succ) = 66) :=
by
  sorry

end sum_of_three_largest_l176_176018
