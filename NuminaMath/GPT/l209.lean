import Mathlib

namespace discriminant_of_quadratic_equation_l209_209906

noncomputable def discriminant (a b c : ℤ) : ℤ :=
  b^2 - 4 * a * c

theorem discriminant_of_quadratic_equation : discriminant 5 (-11) (-18) = 481 := by
  sorry

end discriminant_of_quadratic_equation_l209_209906


namespace range_of_a_l209_209636

variable (a : ℝ)

def p : Prop := ∃ x₀ : ℝ, x₀^2 - a * x₀ + 1 = 0

def q : Prop := ∀ x : ℝ, x ≥ 0 → x^2 - 2 * a * x + a^2 + 1 ≥ 1

theorem range_of_a : ¬(p a ∨ q a) → -2 < a ∧ a < 0 := by
  sorry

end range_of_a_l209_209636


namespace smallest_integer_is_nine_l209_209895

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end smallest_integer_is_nine_l209_209895


namespace distance_from_wall_l209_209970

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l209_209970


namespace identify_perfect_square_is_689_l209_209153

-- Definitions of the conditions
def natural_numbers (n : ℕ) : Prop := True -- All natural numbers are accepted
def digits_in_result (n m : ℕ) (d : ℕ) : Prop := (n * m) % 1000 = d

-- Theorem to be proved
theorem identify_perfect_square_is_689 (n : ℕ) :
  (∀ m, natural_numbers m → digits_in_result m m 689 ∨ digits_in_result m m 759) →
  ∃ m, natural_numbers m ∧ digits_in_result m m 689 :=
sorry

end identify_perfect_square_is_689_l209_209153


namespace coconut_transport_l209_209615

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l209_209615


namespace total_pies_sold_l209_209065

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l209_209065


namespace part_cost_l209_209399

theorem part_cost (hours : ℕ) (hourly_rate total_paid : ℕ) 
  (h1 : hours = 2)
  (h2 : hourly_rate = 75)
  (h3 : total_paid = 300) : 
  total_paid - (hours * hourly_rate) = 150 := 
by
  sorry

end part_cost_l209_209399


namespace exists_same_color_ratios_l209_209606

-- Definition of coloring function.
def coloring : ℕ → Fin 2 := sorry

-- Definition of the problem: there exist A, B, C such that A : C = C : B,
-- and A, B, C are of same color.
theorem exists_same_color_ratios :
  ∃ A B C : ℕ, coloring A = coloring B ∧ coloring B = coloring C ∧ 
  (A : ℚ) / C = (C : ℚ) / B := 
sorry

end exists_same_color_ratios_l209_209606


namespace percentage_increase_l209_209119

theorem percentage_increase (D1 D2 : ℕ) (total_days : ℕ) (H1 : D1 = 4) (H2 : total_days = 9) (H3 : D1 + D2 = total_days) : 
  (D2 - D1) / D1 * 100 = 25 := 
sorry

end percentage_increase_l209_209119


namespace three_digit_sum_27_l209_209355

theorem three_digit_sum_27 {a b c : ℕ} (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 0 ≤ c ∧ c ≤ 9) :
  a + b + c = 27 → (a, b, c) = (9, 9, 9) :=
by
  sorry

end three_digit_sum_27_l209_209355


namespace ball_hits_ground_time_l209_209276

theorem ball_hits_ground_time (t : ℝ) : 
  (∃ t : ℝ, -10 * t^2 + 40 * t + 50 = 0 ∧ t ≥ 0) → t = 5 := 
by
  -- placeholder for proof
  sorry

end ball_hits_ground_time_l209_209276


namespace ladder_base_distance_l209_209937

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l209_209937


namespace base_from_wall_l209_209966

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l209_209966


namespace coefficient_x3y5_in_expansion_l209_209722

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l209_209722


namespace value_of_f_neg_a_l209_209088

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin x + x^3 + 1

theorem value_of_f_neg_a (a : ℝ) (h : f a = 3) : f (-a) = -1 := 
by
  sorry

end value_of_f_neg_a_l209_209088


namespace consecutive_even_sum_l209_209284

theorem consecutive_even_sum (n : ℤ) (h : (n - 2) + (n + 2) = 156) : n = 78 :=
by
  sorry

end consecutive_even_sum_l209_209284


namespace power_mod_equiv_l209_209910

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l209_209910


namespace ladder_distance_l209_209947

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l209_209947


namespace cost_of_bananas_l209_209444

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l209_209444


namespace age_ratio_proof_l209_209762

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l209_209762


namespace coeff_x3y5_in_expansion_l209_209736

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l209_209736


namespace pete_miles_walked_l209_209416

-- Define the conditions
def maxSteps := 99999
def numFlips := 50
def finalReading := 25000
def stepsPerMile := 1500

-- Proof statement that Pete walked 3350 miles
theorem pete_miles_walked : 
  (numFlips * (maxSteps + 1) + finalReading) / stepsPerMile = 3350 := 
by 
  sorry

end pete_miles_walked_l209_209416


namespace maximum_bags_of_milk_l209_209455

theorem maximum_bags_of_milk (bag_cost : ℚ) (promotion : ℕ → ℕ) (total_money : ℚ) 
  (h1 : bag_cost = 2.5) 
  (h2 : promotion 2 = 3) 
  (h3 : total_money = 30) : 
  ∃ n, n = 18 ∧ (total_money >= n * bag_cost - (n / 3) * bag_cost) :=
by
  sorry

end maximum_bags_of_milk_l209_209455


namespace hundredth_number_is_524_l209_209610


open Nat

-- Define the set as described
def exp_triple_set := { n : ℕ | ∃ (x y z : ℕ), x < y ∧ y < z ∧ n = 2^x + 2^y + 2^z }

-- The theorem proving the 100th element in the ordered set is 524
theorem hundredth_number_is_524 : 
  finset.val (finset.sort (≤) (finset.filter (λ n, n ∈ exp_triple_set) (finset.range 9999))) 99 = 524 := 
begin
  sorry
end

end hundredth_number_is_524_l209_209610


namespace water_tank_capacity_l209_209300

theorem water_tank_capacity (C : ℝ) :
  (0.40 * C - 0.25 * C = 36) → C = 240 :=
  sorry

end water_tank_capacity_l209_209300


namespace sum_of_primes_no_solution_congruence_l209_209343

theorem sum_of_primes_no_solution_congruence :
  2 + 5 = 7 :=
by
  sorry

end sum_of_primes_no_solution_congruence_l209_209343


namespace red_card_value_l209_209547

theorem red_card_value (credits : ℕ) (total_cards : ℕ) (blue_card_value : ℕ) (red_cards : ℕ) (blue_cards : ℕ) 
    (condition1 : blue_card_value = 5)
    (condition2 : total_cards = 20)
    (condition3 : credits = 84)
    (condition4 : red_cards = 8)
    (condition5 : blue_cards = total_cards - red_cards) :
  (credits - blue_cards * blue_card_value) / red_cards = 3 :=
by
  sorry

end red_card_value_l209_209547


namespace treasure_value_l209_209633

theorem treasure_value
    (fonzie_paid : ℕ) (auntbee_paid : ℕ) (lapis_paid : ℕ)
    (lapis_share : ℚ) (lapis_received : ℕ) (total_value : ℚ)
    (h1 : fonzie_paid = 7000) 
    (h2 : auntbee_paid = 8000) 
    (h3 : lapis_paid = 9000) 
    (h4 : fonzie_paid + auntbee_paid + lapis_paid = 24000) 
    (h5 : lapis_share = lapis_paid / (fonzie_paid + auntbee_paid + lapis_paid)) 
    (h6 : lapis_received = 337500) 
    (h7 : lapis_share * total_value = lapis_received) :
  total_value = 1125000 := by
  sorry

end treasure_value_l209_209633


namespace probability_of_selecting_one_marble_each_color_l209_209164

theorem probability_of_selecting_one_marble_each_color
  (total_red_marbles : ℕ) (total_blue_marbles : ℕ) (total_green_marbles : ℕ) (total_selected_marbles : ℕ) 
  (total_marble_count : ℕ) : 
  total_red_marbles = 3 → total_blue_marbles = 3 → total_green_marbles = 3 → total_selected_marbles = 3 → total_marble_count = 9 →
  (27 / 84) = 9 / 28 :=
by
  intros h_red h_blue h_green h_selected h_total
  sorry

end probability_of_selecting_one_marble_each_color_l209_209164


namespace fraction_sum_divided_by_2_equals_decimal_l209_209336

theorem fraction_sum_divided_by_2_equals_decimal :
  let f1 := (3 : ℚ) / 20
  let f2 := (5 : ℚ) / 200
  let f3 := (7 : ℚ) / 2000
  let sum := f1 + f2 + f3
  let result := sum / 2
  result = 0.08925 := 
by
  sorry

end fraction_sum_divided_by_2_equals_decimal_l209_209336


namespace johns_original_number_l209_209666

def switch_digits (n : ℕ) : ℕ :=
  let tens := n / 10
  let units := n % 10
  10 * units + tens

theorem johns_original_number :
  ∃ x : ℕ, (10 ≤ x ∧ x < 100) ∧ (∃ y : ℕ, y = 5 * x + 13 ∧ 82 ≤ switch_digits y ∧ switch_digits y ≤ 86 ∧ x = 11) :=
by
  sorry

end johns_original_number_l209_209666


namespace books_leftover_l209_209230

theorem books_leftover (boxes : ℕ) (books_per_box : ℕ) (new_box_capacity : ℕ) 
  (h1 : boxes = 1575) (h2 : books_per_box = 45) (h3 : new_box_capacity = 50) :
  ((boxes * books_per_box) % new_box_capacity) = 25 :=
by
  sorry

end books_leftover_l209_209230


namespace compare_a_b_c_l209_209199

def a := 2^12
def b := 3^8
def c := 7^4

theorem compare_a_b_c : b > a ∧ a > c :=
by {
  unfold a b c, 
  -- comparision of exponents
  have h1 : b = 9^4 := by sorry,
  have h2 : a = 8^4 := by sorry,
  have h3 : c = 7^4 := by sorry,
  -- comparison of bases
  have h4 : 9 > 8 := by exact nat.succ_pos 8,
  have h5 : 8 > 7 := by exact nat.succ_pos 7,
  exact ⟨pow_lt_pow_of_lt_left h4 (by linarith) zero_lt_four, pow_lt_pow_of_lt_left h5 (by linarith) zero_lt_four⟩
}

end compare_a_b_c_l209_209199


namespace probability_two_congestion_days_is_correct_l209_209902

-- Define the condition for traffic congestion probability
def prob_traffic_congestion_per_day := 0.4

-- Define the mapping of integers representing traffic congestion
def is_traffic (x : ℕ) : Prop := x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4

-- Define the list of simulated data
def simulated_data := [[8, 0, 7], [0, 6, 6], [1, 2, 3], [9, 2, 3], [4, 7, 1], 
                       [5, 3, 2], [7, 1, 2], [2, 6, 9], [5, 0, 7], [7, 5, 2], 
                       [4, 4, 3], [2, 7, 7], [3, 0, 3], [9, 2, 7], [7, 5, 6], 
                       [3, 6, 8], [8, 4, 0], [4, 1, 3], [7, 3, 0], [0, 8, 6]]

-- Function to check if exactly two days have traffic congestion
def exactly_two_traffic_days (days : List ℕ) : Prop :=
  (countp is_traffic days) = 2

-- Calculate the probability
def prob_exactly_two_traffic_days (data : List (List ℕ)) : ℝ :=
  (countp exactly_two_traffic_days data).to_real / data.length.to_real

-- Statement to be proved
theorem probability_two_congestion_days_is_correct :
  prob_exactly_two_traffic_days simulated_data = 0.25 :=
by sorry

end probability_two_congestion_days_is_correct_l209_209902


namespace age_difference_l209_209858

variable (K_age L_d_age L_s_age : ℕ)

def condition1 : Prop := L_d_age = K_age - 10
def condition2 : Prop := L_s_age = 2 * K_age
def condition3 : Prop := K_age = 12

theorem age_difference : condition1 → condition2 → condition3 → L_s_age - L_d_age = 22 := by
  intros h1 h2 h3
  rw [h3] at h1
  rw [h3] at h2
  simp at h1
  simp at h2
  rw [h1, h2]
  norm_num
  sorry

end age_difference_l209_209858


namespace largest_divisor_8_l209_209395

theorem largest_divisor_8 (p q : ℤ) (hp : p % 2 = 1) (hq : q % 2 = 1) (h : q < p) : 
  8 ∣ (p^2 - q^2 + 2*p - 2*q) := 
sorry

end largest_divisor_8_l209_209395


namespace solve_system_l209_209161

theorem solve_system (a b c x y z : ℝ) (h₀ : a = (a * x + c * y) / (b * z + 1))
  (h₁ : b = (b * x + y) / (b * z + 1)) 
  (h₂ : c = (a * z + c) / (b * z + 1)) 
  (h₃ : ¬ a = b * c) :
  x = 1 ∧ y = 0 ∧ z = 0 :=
sorry

end solve_system_l209_209161


namespace problem1_l209_209586

theorem problem1 :
  (Real.sqrt (3/2)) * (Real.sqrt (21/4)) / (Real.sqrt (7/2)) = 3/2 :=
sorry

end problem1_l209_209586


namespace point_P_on_line_l_intersection_of_line_l_and_curve_C_l209_209234

noncomputable def pointP : ℝ × ℝ := (0, Real.sqrt 3)
def line_l_polar (ρ θ : ℝ) : Prop := ρ = (Real.sqrt 3) / (2 * Real.cos (θ - π / 6))
def cartesian_of_line_l (x y : ℝ) : Prop := Real.sqrt 3 * x + y = Real.sqrt 3
def parametric_C (φ : ℝ) : ℝ × ℝ := (Real.sqrt 2 * Real.cos φ, 2 * Real.sin φ)
def curve_C (x y : ℝ) : Prop := (x^2 / 2) + (y^2 / 4) = 1

open Real

theorem point_P_on_line_l (ρ θ : ℝ) : cartesian_of_line_l (0) (sqrt 3) :=
by sorry

theorem intersection_of_line_l_and_curve_C (t : ℝ) : 
  let A := parametric_C t, B := parametric_C (-t) in
  ∃ t₁ t₂ : ℝ, 
  t₁ + t₂ = -(12/5) ∧ t₁ * t₂ = -(4/5) ∧ 
  (sqrt 14 : ℝ) = (1 / dist pointP A) + (1 / dist pointP B) :=
by sorry

end point_P_on_line_l_intersection_of_line_l_and_curve_C_l209_209234


namespace exists_special_integer_l209_209500

-- Define the mathematical conditions and the proof
theorem exists_special_integer (P : Finset ℕ) (hP : ∀ p ∈ P, Nat.Prime p) : 
  ∃ x : ℕ, 
    (∀ p ∈ P, ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) ∧
    (∀ p ∉ P, ¬∃ a b : ℕ, 0 < a ∧ 0 < b ∧ x = a^p + b^p) :=
sorry

end exists_special_integer_l209_209500


namespace initial_momentum_eq_2Fx_div_v_l209_209327

variable (m v F x t : ℝ)
variable (H_initial_conditions : v ≠ 0)
variable (H_force : F > 0)
variable (H_distance : x > 0)
variable (H_time : t > 0)
variable (H_stopping_distance : x = (m * v^2) / (2 * F))
variable (H_stopping_time : t = (m * v) / F)

theorem initial_momentum_eq_2Fx_div_v :
  m * v = (2 * F * x) / v :=
sorry

end initial_momentum_eq_2Fx_div_v_l209_209327


namespace construct_polygon_with_area_l209_209010

theorem construct_polygon_with_area 
  (n : ℕ) (l : ℝ) (a : ℝ) 
  (matchsticks : n = 12) 
  (matchstick_length : l = 2) 
  (area_target : a = 16) : 
  ∃ (polygon : EuclideanGeometry.Polygon ℝ) (sides : list ℝ),
    sides.length = n ∧ ∀ side ∈ sides, side = l ∧ polygon.area = a := 
sorry

end construct_polygon_with_area_l209_209010


namespace n_minus_m_eq_200_l209_209691

-- Define the parameters
variable (m n x : ℝ)

-- State the conditions
def condition1 : Prop := m ≤ 8 * x - 1 ∧ 8 * x - 1 ≤ n 
def condition2 : Prop := (n + 1)/8 - (m + 1)/8 = 25

-- State the theorem to prove
theorem n_minus_m_eq_200 (h1 : condition1 m n x) (h2 : condition2 m n) : n - m = 200 := 
by 
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end n_minus_m_eq_200_l209_209691


namespace greatest_integer_third_side_l209_209575

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l209_209575


namespace crayons_remaining_l209_209528

def initial_crayons : ℕ := 87
def eaten_crayons : ℕ := 7

theorem crayons_remaining : (initial_crayons - eaten_crayons) = 80 := by
  sorry

end crayons_remaining_l209_209528


namespace journey_duration_is_9_hours_l209_209152

noncomputable def journey_time : ℝ :=
  let d1 := 90 -- Distance traveled by Tom and Dick by car before Tom got off
  let d2 := 60 -- Distance Dick backtracked to pick up Harry
  let T := (d1 / 30) + ((120 - d1) / 5) -- Time taken for Tom's journey
  T

theorem journey_duration_is_9_hours : journey_time = 9 := 
by 
  sorry

end journey_duration_is_9_hours_l209_209152


namespace impossibility_of_quadratic_conditions_l209_209345

open Real

theorem impossibility_of_quadratic_conditions :
  ∀ (a b c t : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ t ∧ b ≠ t ∧ c ≠ t →
  (b * t) ^ 2 - 4 * a * c > 0 →
  c ^ 2 - 4 * b * a > 0 →
  (a * t) ^ 2 - 4 * b * c > 0 →
  false :=
by sorry

end impossibility_of_quadratic_conditions_l209_209345


namespace police_speed_l209_209325

/-- 
A thief runs away from a location with a speed of 20 km/hr.
A police officer starts chasing him from a location 60 km away after 1 hour.
The police officer catches the thief after 4 hours.
Prove that the speed of the police officer is 40 km/hr.
-/
theorem police_speed
  (thief_speed : ℝ)
  (police_start_distance : ℝ)
  (police_chase_time : ℝ)
  (time_head_start : ℝ)
  (police_distance_to_thief : ℝ)
  (thief_distance_after_time : ℝ)
  (total_distance_police_officer : ℝ) :
  thief_speed = 20 ∧
  police_start_distance = 60 ∧
  police_chase_time = 4 ∧
  time_head_start = 1 ∧
  police_distance_to_thief = police_start_distance + 100 ∧
  thief_distance_after_time = thief_speed * police_chase_time + thief_speed * time_head_start ∧
  total_distance_police_officer = police_start_distance + (thief_speed * (police_chase_time + time_head_start)) →
  (total_distance_police_officer / police_chase_time) = 40 := by
  sorry

end police_speed_l209_209325


namespace sqrt_sum_eval_l209_209489

theorem sqrt_sum_eval : 
  (Real.sqrt 50 + Real.sqrt 72) = 11 * Real.sqrt 2 := 
by 
  sorry

end sqrt_sum_eval_l209_209489


namespace coefficient_x3y5_in_expansion_l209_209721

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l209_209721


namespace r_daily_earnings_l209_209032

def earnings_problem (P Q R : ℝ) : Prop :=
  (9 * (P + Q + R) = 1890) ∧ 
  (5 * (P + R) = 600) ∧ 
  (7 * (Q + R) = 910)

theorem r_daily_earnings :
  ∃ P Q R : ℝ, earnings_problem P Q R ∧ R = 40 := sorry

end r_daily_earnings_l209_209032


namespace data_properties_l209_209692

-- Defining the list of numbers
def data : List ℝ := [-5, 3, 2, -3, 3]

-- Defining the mean function
noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

-- Defining the mode function
noncomputable def mode (l : List ℝ) : ℝ :=
  l.mode

-- Defining the median function
noncomputable def median (l : List ℝ) : ℝ :=
  l.median

-- Defining the variance function
noncomputable def variance (l : List ℝ) : ℝ :=
  let μ := mean l
  (l.map (λ x => (x - μ)^2)).sum / l.length

-- Define the theorem to be proven
theorem data_properties :
  mean data = 0 ∧
  mode data = 3 ∧
  median data = 2 ∧
  variance data = 11.2 :=
by
  sorry

end data_properties_l209_209692


namespace intersection_of_A_and_B_l209_209503

noncomputable def A : Set ℕ := {x | x > 0 ∧ x ≤ 3}
def B : Set ℕ := {x | 0 < x ∧ x < 4}

theorem intersection_of_A_and_B : 
  A ∩ B = {1, 2, 3} :=
by {
  sorry
}

end intersection_of_A_and_B_l209_209503


namespace inequality_minus_x_plus_3_l209_209106

variable (x y : ℝ)

theorem inequality_minus_x_plus_3 (h : x < y) : -x + 3 > -y + 3 :=
by {
  sorry
}

end inequality_minus_x_plus_3_l209_209106


namespace smallest_solution_l209_209829

def equation (x : ℝ) : ℝ := (3*x)/(x-3) + (3*x^2 - 27)/x

theorem smallest_solution : ∃ x : ℝ, equation x = 14 ∧ x = (7 - Real.sqrt 76) / 3 := 
by {
  -- proof steps go here
  sorry
}

end smallest_solution_l209_209829


namespace coefficient_of_x3_y5_in_binomial_expansion_l209_209716

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l209_209716


namespace scientific_notation_of_18M_l209_209695

theorem scientific_notation_of_18M : 18000000 = 1.8 * 10^7 :=
by
  sorry

end scientific_notation_of_18M_l209_209695


namespace product_correlation_function_l209_209417

open ProbabilityTheory

/-
Theorem: Given two centered and uncorrelated random functions \( \dot{X}(t) \) and \( \dot{Y}(t) \),
the correlation function of their product \( Z(t) = \dot{X}(t) \dot{Y}(t) \) is the product of their correlation functions.
-/
theorem product_correlation_function 
  (X Y : ℝ → ℝ)
  (hX_centered : ∀ t, (∫ x, X t ∂x) = 0) 
  (hY_centered : ∀ t, (∫ y, Y t ∂y) = 0)
  (h_uncorrelated : ∀ t1 t2, ∫ x, X t1 * Y t2 ∂x = (∫ x, X t1 ∂x) * (∫ y, Y t2 ∂y)) :
  ∀ t1 t2, 
  (∫ x, (X t1 * Y t1) * (X t2 * Y t2) ∂x) = 
  (∫ x, X t1 * X t2 ∂x) * (∫ y, Y t1 * Y t2 ∂y) :=
by
  sorry

end product_correlation_function_l209_209417


namespace ladder_base_distance_l209_209934

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l209_209934


namespace james_lifting_ratio_correct_l209_209996

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l209_209996


namespace book_surface_area_l209_209131

variables (L : ℕ) (T : ℕ) (A1 : ℕ) (A2 : ℕ) (W : ℕ) (S : ℕ)

theorem book_surface_area (hL : L = 5) (hT : T = 2) 
                         (hA1 : A1 = L * W) (hA1_val : A1 = 50)
                         (hA2 : A2 = T * W) (hA2_val : A2 = 10) :
  S = 2 * A1 + A2 + 2 * (L * T) :=
sorry

end book_surface_area_l209_209131


namespace equal_roots_a_l209_209555

theorem equal_roots_a {a : ℕ} :
  (a * a - 4 * (a + 3) = 0) → a = 6 := 
sorry

end equal_roots_a_l209_209555


namespace compound_interest_calculation_l209_209601

-- Define the variables used in the problem
def principal : ℝ := 8000
def annual_rate : ℝ := 0.05
def compound_frequency : ℕ := 1
def final_amount : ℝ := 9261
def years : ℝ := 3

-- Statement we need to prove
theorem compound_interest_calculation :
  final_amount = principal * (1 + annual_rate / compound_frequency) ^ (compound_frequency * years) :=
by 
  sorry

end compound_interest_calculation_l209_209601


namespace ratio_new_circumference_to_original_diameter_l209_209847

-- Define the problem conditions
variables (r k : ℝ) (hk : k > 0)

-- Define the Lean theorem to express the proof problem
theorem ratio_new_circumference_to_original_diameter (r k : ℝ) (hk : k > 0) :
  (π * (1 + k / r)) = (2 * π * (r + k)) / (2 * r) :=
by {
  -- Placeholder proof, to be filled in
  sorry
}

end ratio_new_circumference_to_original_diameter_l209_209847


namespace arithmetic_sequence_common_difference_l209_209361

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h1 : a 1 = 1)
  (h3 : a 3 = 11)
  (h_arith : ∀ n : ℕ, a (n + 1) - a n = d) : d = 5 :=
sorry

end arithmetic_sequence_common_difference_l209_209361


namespace more_boys_than_girls_l209_209566

theorem more_boys_than_girls (total_people : ℕ) (num_girls : ℕ) (num_boys : ℕ) (more_boys : ℕ) : 
  total_people = 133 ∧ num_girls = 50 ∧ num_boys = total_people - num_girls ∧ more_boys = num_boys - num_girls → more_boys = 33 :=
by 
  sorry

end more_boys_than_girls_l209_209566


namespace total_pies_sold_l209_209066

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l209_209066


namespace union_sets_intersection_complement_l209_209511

open Set

noncomputable def U := (univ : Set ℝ)
def A := { x : ℝ | x ≥ 2 }
def B := { x : ℝ | x < 5 }

theorem union_sets : A ∪ B = univ := by
  sorry

theorem intersection_complement : (U \ A) ∩ B = { x : ℝ | x < 2 } := by
  sorry

end union_sets_intersection_complement_l209_209511


namespace cost_of_pastrami_l209_209259

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l209_209259


namespace total_pies_sold_l209_209064

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l209_209064


namespace correct_quotient_l209_209917

theorem correct_quotient (D : ℕ) (Q : ℕ) (h1 : D = 21 * Q) (h2 : D = 12 * 49) : Q = 28 := 
by
  sorry

end correct_quotient_l209_209917


namespace neg_p_l209_209693

-- Let's define the original proposition p
def p : Prop := ∃ x : ℝ, x ≥ 2 ∧ x^2 - 2 * x - 2 > 0

-- Now, we state the problem in Lean as requiring the proof of the negation of p
theorem neg_p : ¬p ↔ ∀ x : ℝ, x ≥ 2 → x^2 - 2 * x - 2 ≤ 0 :=
by
  sorry

end neg_p_l209_209693


namespace _l209_209126

noncomputable def sin_cos_ratio_theorem (u v : ℝ)
  (h1 : sin u / sin v = 4) 
  (h2 : cos u / cos v = 1/3) : 
  sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v) = 19 / 381 :=
sorry

end _l209_209126


namespace sec_150_eq_neg_2_sqrt3_over_3_l209_209800

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l209_209800


namespace emily_num_dresses_l209_209484

theorem emily_num_dresses (M : ℕ) (D : ℕ) (E : ℕ) 
  (h1 : D = M + 12) 
  (h2 : M = E / 2) 
  (h3 : M + D + E = 44) : 
  E = 16 := 
by 
  sorry

end emily_num_dresses_l209_209484


namespace ladder_distance_l209_209931

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l209_209931


namespace total_money_spent_l209_209753

noncomputable def total_expenditure (A : ℝ) : ℝ :=
  let person1_8_expenditure := 8 * 12
  let person9_expenditure := A + 8
  person1_8_expenditure + person9_expenditure

theorem total_money_spent :
  (∃ A : ℝ, total_expenditure A = 9 * A ∧ A = 13) →
  total_expenditure 13 = 117 :=
by
  intro h
  sorry

end total_money_spent_l209_209753


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209785

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209785


namespace remainder_of_sum_div_8_l209_209622

theorem remainder_of_sum_div_8 :
  let a := 2356789
  let b := 211
  (a + b) % 8 = 0 := 
by 
  sorry

end remainder_of_sum_div_8_l209_209622


namespace sum_of_remainders_l209_209742

theorem sum_of_remainders (a b c : ℕ) (h₁ : a % 30 = 15) (h₂ : b % 30 = 7) (h₃ : c % 30 = 18) : 
    (a + b + c) % 30 = 10 := 
by
  sorry

end sum_of_remainders_l209_209742


namespace eccentricity_ratio_l209_209205

noncomputable def ellipse_eccentricity (m n : ℝ) : ℝ := (1 - (1 / n) / (1 / m))^(1/2)

theorem eccentricity_ratio (m n : ℝ) (h : ellipse_eccentricity m n = 1 / 2) :
  m / n = 3 / 4 :=
by
  sorry

end eccentricity_ratio_l209_209205


namespace geometric_series_six_terms_l209_209453

theorem geometric_series_six_terms :
  (1/4 - 1/16 + 1/64 - 1/256 + 1/1024 - 1/4096 : ℚ) = 4095 / 20480 :=
by
  sorry

end geometric_series_six_terms_l209_209453


namespace volleyball_club_members_l209_209468

variables (B G : ℝ)

theorem volleyball_club_members (h1 : B + G = 30) (h2 : 1 / 3 * G + B = 18) : B = 12 := by
  -- Mathematical steps and transformations done here to show B = 12
  sorry

end volleyball_club_members_l209_209468


namespace initial_treasure_amount_l209_209878

theorem initial_treasure_amount 
  (T : ℚ)
  (h₁ : T * (1 - 1/13) * (1 - 1/17) = 150) : 
  T = 172 + 21/32 :=
sorry

end initial_treasure_amount_l209_209878


namespace shifted_parabola_equation_l209_209003

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Proposition to prove that the given parabola equation is correct after transformations
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = -2 * (x + 1)^2 + 3 :=
by
  sorry

end shifted_parabola_equation_l209_209003


namespace no_point_in_common_l209_209658

theorem no_point_in_common (b : ℝ) :
  (∀ (x y : ℝ), y = 2 * x + b → (x^2 / 4) + y^2 ≠ 1) ↔ (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) :=
by
  sorry

end no_point_in_common_l209_209658


namespace sum_of_midpoint_coords_is_seven_l209_209024

-- Define coordinates of the endpoints
def endpoint1 : ℝ × ℝ := (8, 16)
def endpoint2 : ℝ × ℝ := (-2, -8)

-- Define the midpoint coordinates
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the sum of the coordinates of the midpoint
def sum_of_midpoint_coords (p : ℝ × ℝ) : ℝ :=
  p.1 + p.2

-- Theorem stating that the sum of the coordinates of the midpoint is 7
theorem sum_of_midpoint_coords_is_seven : 
  sum_of_midpoint_coords (midpoint endpoint1 endpoint2) = 7 :=
by
  -- Proof would go here
  sorry

end sum_of_midpoint_coords_is_seven_l209_209024


namespace product_of_two_consecutive_even_numbers_is_divisible_by_8_l209_209158

theorem product_of_two_consecutive_even_numbers_is_divisible_by_8 (n : ℤ) : (4 * n * (n + 1)) % 8 = 0 :=
sorry

end product_of_two_consecutive_even_numbers_is_divisible_by_8_l209_209158


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209803

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209803


namespace sec_150_eq_l209_209822

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l209_209822


namespace shaded_square_cover_columns_l209_209321

def triangular_number (n : Nat) : Nat := n * (n + 1) / 2

theorem shaded_square_cover_columns :
  ∃ n : Nat, 
    triangular_number n = 136 ∧ 
    ∀ i : Fin 10, ∃ k ≤ n, (triangular_number k) % 10 = i.val :=
sorry

end shaded_square_cover_columns_l209_209321


namespace factor_64_minus_16y_squared_l209_209077

theorem factor_64_minus_16y_squared (y : ℝ) : 
  64 - 16 * y^2 = 16 * (2 - y) * (2 + y) :=
by
  -- skipping the actual proof steps
  sorry

end factor_64_minus_16y_squared_l209_209077


namespace eccentricity_of_ellipse_equation_of_ellipse_l209_209363

variable {a b : ℝ}
variable {x y : ℝ}

/-- Problem 1: Eccentricity of the given ellipse --/
theorem eccentricity_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ e : ℝ, e = Real.sqrt (1 - (b / a) ^ 2) ∧ e = Real.sqrt 3 / 2 := by
  sorry

/-- Problem 2: Equation of the ellipse with respect to maximizing the area of triangle OMN --/
theorem equation_of_ellipse (ha : a = 2 * b) (hb0 : 0 < b) :
  ∃ l : ℝ → ℝ, (∃ k : ℝ, ∀ x, l x = k * x + 2) →
  ∀ x y : ℝ, (x^2 / (a^2) + y^2 / (b^2) = 1) →
  (∀ x' y' : ℝ, (x'^2 + 4 * y'^2 = 4 * b^2) ∧ y' = k * x' + 2) →
  (∃ a b : ℝ, a = 8 ∧ b = 2 ∧ x^2 / a + y^2 / b = 1) := by
  sorry

end eccentricity_of_ellipse_equation_of_ellipse_l209_209363


namespace solutions_of_equation_l209_209741

theorem solutions_of_equation :
  ∀ x : ℝ, x * (x - 3) = x - 3 ↔ x = 1 ∨ x = 3 :=
by sorry

end solutions_of_equation_l209_209741


namespace solve_inequality_l209_209140

open Set Real

noncomputable def inequality_solution_set : Set ℝ :=
  {x | -2 ≤ x ∧ x ≤ 2} ∪ {6}

theorem solve_inequality (x : ℝ) :
  (x ^ 2 - 4) * (x - 6) ^ 2 ≤ 0 ↔ x ∈ inequality_solution_set := 
sorry

end solve_inequality_l209_209140


namespace a_oxen_count_l209_209302

-- Define the conditions from the problem
def total_rent : ℝ := 210
def c_share_rent : ℝ := 54
def oxen_b : ℝ := 12
def oxen_c : ℝ := 15
def months_b : ℝ := 5
def months_c : ℝ := 3
def months_a : ℝ := 7
def oxen_c_months : ℝ := oxen_c * months_c
def total_ox_months (oxen_a : ℝ) : ℝ := (oxen_a * months_a) + (oxen_b * months_b) + oxen_c_months

-- The theorem we want to prove
theorem a_oxen_count (oxen_a : ℝ) (h : c_share_rent / total_rent = oxen_c_months / total_ox_months oxen_a) :
  oxen_a = 10 := by sorry

end a_oxen_count_l209_209302


namespace mileage_per_gallon_l209_209870

-- Define the conditions
def miles_driven : ℝ := 100
def gallons_used : ℝ := 5

-- Define the question as a theorem to be proven
theorem mileage_per_gallon : (miles_driven / gallons_used) = 20 := by
  sorry

end mileage_per_gallon_l209_209870


namespace integer_pairs_satisfy_equation_l209_209079

theorem integer_pairs_satisfy_equation :
  ∀ (x y : ℤ), (x^2 * y + y^2 = x^3) → (x = 0 ∧ y = 0) ∨ (x = -4 ∧ y = -8) :=
by
  sorry

end integer_pairs_satisfy_equation_l209_209079


namespace tiled_floor_area_correct_garden_area_correct_seating_area_correct_l209_209767

noncomputable def length_room : ℝ := 20
noncomputable def width_room : ℝ := 12
noncomputable def width_veranda : ℝ := 2
noncomputable def length_pool : ℝ := 15
noncomputable def width_pool : ℝ := 6

noncomputable def area (length width : ℝ) : ℝ := length * width

noncomputable def area_room : ℝ := area length_room width_room
noncomputable def area_pool : ℝ := area length_pool width_pool
noncomputable def area_tiled_floor : ℝ := area_room - area_pool

noncomputable def total_length : ℝ := length_room + 2 * width_veranda
noncomputable def total_width : ℝ := width_room + 2 * width_veranda
noncomputable def area_total : ℝ := area total_length total_width
noncomputable def area_veranda : ℝ := area_total - area_room
noncomputable def area_garden : ℝ := area_veranda / 2
noncomputable def area_seating : ℝ := area_veranda / 2

theorem tiled_floor_area_correct : area_tiled_floor = 150 := by
  sorry

theorem garden_area_correct : area_garden = 72 := by
  sorry

theorem seating_area_correct : area_seating = 72 := by
  sorry

end tiled_floor_area_correct_garden_area_correct_seating_area_correct_l209_209767


namespace finger_cycle_2004th_l209_209396

def finger_sequence : List String :=
  ["Little finger", "Ring finger", "Middle finger", "Index finger", "Thumb", "Index finger", "Middle finger", "Ring finger"]

theorem finger_cycle_2004th : 
  finger_sequence.get! ((2004 - 1) % finger_sequence.length) = "Index finger" :=
by
  -- The proof is not required, so we use sorry
  sorry

end finger_cycle_2004th_l209_209396


namespace find_b_l209_209224

theorem find_b (a b c : ℝ) (h1 : a = 6) (h2 : c = 3) (h3 : a * b * c = (Real.sqrt ((a + 2) * (b + 3))) / (c + 1)) : b = 15 :=
by
  rw [h1, h2] at h3
  sorry

end find_b_l209_209224


namespace triangle_problem_l209_209749

theorem triangle_problem (perimeter_XYZ : ℝ) (angle_XZY : ℝ) (radius_O : ℝ)
  (O_on_XZ : ∃ O : ℝ, O ∈ set.interval 0 120) (tangent_ZY_YX : ∃ O : ℝ, ∀ ZY YX : ℝ, tangent ZY O ∧ tangent YX O)
  (h1 : perimeter_XYZ = 120)
  (h2 : angle_XZY = 90)
  (h3 : radius_O = 15) :
  let OY := 5 / 2 in ∃ p q : ℕ, nat.gcd p q = 1 ∧ OY = p / q ∧ p + q = 7 := by
  sorry

end triangle_problem_l209_209749


namespace sequence_formula_l209_209838

theorem sequence_formula (a : ℕ → ℤ)
  (h₁ : a 1 = 1)
  (h₂ : a 2 = -3)
  (h₃ : a 3 = 5)
  (h₄ : a 4 = -7)
  (h₅ : a 5 = 9) :
  ∀ n : ℕ, a n = (-1)^(n+1) * (2 * n - 1) :=
by
  sorry

end sequence_formula_l209_209838


namespace angle_measure_is_fifty_l209_209426

theorem angle_measure_is_fifty (x : ℝ) :
  (90 - x = (1 / 2) * (180 - x) - 25) → x = 50 := by
  intro h
  sorry

end angle_measure_is_fifty_l209_209426


namespace sarees_with_6_shirts_l209_209694

-- Define the prices of sarees, shirts and the equation parameters
variables (S T : ℕ) (X : ℕ)

-- Define the conditions as hypotheses
def condition1 := 2 * S + 4 * T = 1600
def condition2 := 12 * T = 2400
def condition3 := X * S + 6 * T = 1600

-- Define the theorem to prove X = 1 under these conditions
theorem sarees_with_6_shirts
  (h1 : condition1 S T)
  (h2 : condition2 T)
  (h3 : condition3 S T X) : 
  X = 1 :=
sorry

end sarees_with_6_shirts_l209_209694


namespace ladder_base_distance_l209_209954

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l209_209954


namespace sec_150_eq_l209_209810

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l209_209810


namespace product_of_intersection_coordinates_l209_209740

theorem product_of_intersection_coordinates :
  let circle1 := {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 4)^2 = 4}
  let circle2 := {p : ℝ × ℝ | (p.1 - 5)^2 + (p.2 - 4)^2 = 9}
  ∃ p : ℝ × ℝ, p ∈ circle1 ∧ p ∈ circle2 ∧ p.1 * p.2 = 16 :=
by
  sorry

end product_of_intersection_coordinates_l209_209740


namespace power_cycle_i_pow_2012_l209_209750

-- Define the imaginary unit i as a complex number
def i : ℂ := Complex.I

-- Define the periodic properties of i
theorem power_cycle (n : ℕ) : Complex := 
  match n % 4 with
  | 0 => 1
  | 1 => i
  | 2 => -1
  | 3 => -i
  | _ => 0 -- this case should never happen

-- Using the periodic properties
theorem i_pow_2012 : (i ^ 2012) = 1 := by
  sorry

end power_cycle_i_pow_2012_l209_209750


namespace smallest_n_exists_l209_209625

theorem smallest_n_exists :
  ∃ n : ℕ, n > 0 ∧ 3^(3^(n + 1)) ≥ 3001 :=
by
  sorry

end smallest_n_exists_l209_209625


namespace min_abs_y1_minus_4y2_l209_209559

-- Definitions based on conditions
def parabola (x y : ℝ) : Prop := y^2 = 4 * x
def focus : (ℝ × ℝ) := (1, 0)

noncomputable def equation_of_line (k y : ℝ) : ℝ := k * y + 1

-- The Lean theorem statement
theorem min_abs_y1_minus_4y2 {x1 y1 x2 y2 : ℝ} (H1 : parabola x1 y1) (H2 : parabola x2 y2)
    (A_in_first_quadrant : 0 < x1 ∧ 0 < y1)
    (line_through_focus : ∃ k : ℝ, x1 = equation_of_line k y1 ∧ x2 = equation_of_line k y2)
    : |y1 - 4 * y2| = 8 :=
sorry

end min_abs_y1_minus_4y2_l209_209559


namespace driver_a_driven_more_distance_l209_209584

-- Definitions based on conditions
def initial_distance : ℕ := 787
def speed_a : ℕ := 90
def speed_b : ℕ := 80
def start_difference : ℕ := 1

-- Statement of the problem
theorem driver_a_driven_more_distance :
  let distance_a := speed_a * (start_difference + (initial_distance - speed_a) / (speed_a + speed_b))
  let distance_b := speed_b * ((initial_distance - speed_a) / (speed_a + speed_b))
  distance_a - distance_b = 131 := by
sorry

end driver_a_driven_more_distance_l209_209584


namespace largest_common_value_l209_209425

/-- The largest value less than 300 that appears in both sequences 
    {7, 14, 21, 28, ...} and {5, 15, 25, 35, ...} -/
theorem largest_common_value (a : ℕ) (n m k : ℕ) :
  (a = 7 * (1 + n)) ∧ (a = 5 + 10 * m) ∧ (a < 300) ∧ (∀ k, (55 + 70 * k < 300) → (55 + 70 * k) ≤ a) 
  → a = 265 :=
by
  sorry

end largest_common_value_l209_209425


namespace mason_father_age_l209_209541

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l209_209541


namespace probability_A1_selected_probability_neither_A2_B2_selected_l209_209177

-- Define the set of male members and female members
def male_members : set ℕ := {1, 2, 3, 4}
def female_members : set ℕ := {1, 2, 3}

-- Define the universal set of all possible outcomes
def outcomes : set (ℕ × ℕ) := { (m, f) | m ∈ male_members ∧ f ∈ female_members }

-- Define event M: "A₁ is selected"
def event_M : set (ℕ × ℕ) := { (m, _) | m = 1 }

-- Define event N: "Neither A₂ nor B₂ is selected"
def event_N : set (ℕ × ℕ) := { (m, f) | m ≠ 2 ∧ f ≠ 2 }

-- Probability space
noncomputable def probability_space : MeasureSpace (ℕ × ℕ) := sorry

-- Statement 1: Probability of A₁ being selected
theorem probability_A1_selected : probability_space.measure (event_M) = 1 / 4 := sorry

-- Statement 2: Probability of neither A₂ nor B₂ being selected
theorem probability_neither_A2_B2_selected : probability_space.measure (event_N) = 11 / 12 := sorry

end probability_A1_selected_probability_neither_A2_B2_selected_l209_209177


namespace ladder_base_distance_l209_209924

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l209_209924


namespace original_price_of_dish_l209_209306

variable (P : ℝ)

def john_paid (P : ℝ) : ℝ := 0.9 * P + 0.15 * P
def jane_paid (P : ℝ) : ℝ := 0.9 * P + 0.135 * P

theorem original_price_of_dish (h : john_paid P = jane_paid P + 1.26) : P = 84 := by
  sorry

end original_price_of_dish_l209_209306


namespace andy_questions_wrong_l209_209773

variables (a b c d : ℕ)

-- Given conditions
def condition1 : Prop := a + b = c + d
def condition2 : Prop := a + d = b + c + 6
def condition3 : Prop := c = 7

-- The theorem to prove
theorem andy_questions_wrong (h1 : condition1 a b c d) (h2 : condition2 a b c d) (h3 : condition3 c) : a = 10 :=
by
  sorry

end andy_questions_wrong_l209_209773


namespace find_b_l209_209830

theorem find_b 
  (b : ℝ)
  (h_pos : 0 < b)
  (h_geom_sequence : ∃ r : ℝ, 10 * r = b ∧ b * r = 2 / 3) :
  b = 2 * Real.sqrt 15 / 3 :=
by
  sorry

end find_b_l209_209830


namespace proof_problem_l209_209097

open Set Real

def M : Set ℝ := { x : ℝ | ∃ y : ℝ, y = log (1 - 2 / x) }
def N : Set ℝ := { x : ℝ | ∃ y : ℝ, y = sqrt (x - 1) }

theorem proof_problem : N ∩ (U \ M) = Icc 1 2 := by
  sorry

end proof_problem_l209_209097


namespace math_problem_l209_209072

theorem math_problem (x : ℤ) (h : x = 9) :
  (x^6 - 27*x^3 + 729) / (x^3 - 27) = 702 :=
by
  sorry

end math_problem_l209_209072


namespace big_white_toys_l209_209323

/-- A store has two types of toys, Big White and Little Yellow, with a total of 60 toys.
    The price ratio of Big White to Little Yellow is 6:5.
    Selling all of them results in a total of 2016 yuan.
    We want to determine how many Big Whites there are. -/
theorem big_white_toys (x k : ℕ) (h1 : 6 * x + 5 * (60 - x) = 2016) (h2 : k = 6) : x = 36 :=
by
  sorry

end big_white_toys_l209_209323


namespace runners_speed_ratio_l209_209904

/-- Two runners, 20 miles apart, start at the same time, aiming to meet. 
    If they run in the same direction, they meet in 5 hours. 
    If they run towards each other, they meet in 1 hour.
    Prove that the ratio of the speed of the faster runner to the slower runner is 3/2. -/
theorem runners_speed_ratio (v1 v2 : ℝ) (h1 : v1 > v2)
  (h2 : 20 = 5 * (v1 - v2)) 
  (h3 : 20 = (v1 + v2)) : 
  v1 / v2 = 3 / 2 :=
sorry

end runners_speed_ratio_l209_209904


namespace foundation_cost_l209_209665

theorem foundation_cost (volume_per_house : ℝ)
    (density : ℝ)
    (cost_per_pound : ℝ)
    (num_houses : ℕ) 
    (dimension_len : ℝ)
    (dimension_wid : ℝ)
    (dimension_height : ℝ)
    : cost_per_pound = 0.02 → density = 150 → dimension_len = 100 → dimension_wid = 100 → dimension_height = 0.5 → num_houses = 3
    → volume_per_house = dimension_len * dimension_wid * dimension_height 
    → (num_houses : ℝ) * (volume_per_house * density * cost_per_pound) = 45000 := 
by 
  sorry

end foundation_cost_l209_209665


namespace magic_shop_purchase_l209_209197

theorem magic_shop_purchase :
  let deck_price := 7
  let frank_decks := 3
  let friend_decks := 2
  let discount_rate := 0.1
  let tax_rate := 0.05
  let total_cost := (frank_decks + friend_decks) * deck_price
  let discount := discount_rate * total_cost
  let discounted_total := total_cost - discount
  let sales_tax := tax_rate * discounted_total
  let rounded_sales_tax := (sales_tax * 100).round / 100
  let final_amount := discounted_total + rounded_sales_tax
  final_amount = 33.08 :=
by
  sorry

end magic_shop_purchase_l209_209197


namespace union_of_sets_l209_209092

def A : Set ℤ := {0, 1}
def B : Set ℤ := {1, 2}

theorem union_of_sets :
  A ∪ B = {0, 1, 2} :=
by
  sorry

end union_of_sets_l209_209092


namespace grasshopper_jumps_rational_angle_l209_209570

noncomputable def alpha_is_rational (α : ℝ) (jump : ℕ → ℝ × ℝ) : Prop :=
  ∃ k n : ℕ, (n ≠ 0) ∧ (jump n = (0, 0)) ∧ (α = (k : ℝ) / (n : ℝ) * 360)

theorem grasshopper_jumps_rational_angle :
  ∀ (α : ℝ) (jump : ℕ → ℝ × ℝ),
    (∀ n : ℕ, dist (jump (n + 1)) (jump n) = 1) →
    (jump 0 = (0, 0)) →
    (∃ n : ℕ, n ≠ 0 ∧ jump n = (0, 0)) →
    alpha_is_rational α jump :=
by
  intros α jump jumps_eq_1 start_exists returns_to_start
  sorry

end grasshopper_jumps_rational_angle_l209_209570


namespace set_operations_l209_209309

def A : Set ℝ := { x | x < 2 }
def B : Set ℝ := { x | 0 < x ∧ x < 5 }
def U : Set ℝ := Set.univ  -- Universal set ℝ
def complement (s : Set ℝ) : Set ℝ := { x | x ∉ s }

theorem set_operations :
  (A ∩ B = { x | 0 < x ∧ x < 2 }) ∧ 
  (complement A ∪ B = { x | 0 < x }) :=
by {
  sorry
}

end set_operations_l209_209309


namespace smallest_positive_integer_n_l209_209626

theorem smallest_positive_integer_n :
  ∃ (n : ℕ), (n > 0) ∧ 
  (∑ k in finset.range (n + 1), real.logb 3 (1 + 1 / 3^(3^k))) ≥ 1 + real.logb 3 (3000 / 3001) ∧
  (∀ m : ℕ, m > 0 ∧ m < n → (∑ k in finset.range (m + 1), real.logb 3 (1 + 1 / 3^(3^k))) < 1 + real.logb 3 (3000 / 3001)) := 
sorry

end smallest_positive_integer_n_l209_209626


namespace ellipse_intersection_l209_209331

open Real

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem ellipse_intersection (f1 f2 : ℝ × ℝ)
    (h1 : f1 = (0, 5))
    (h2 : f2 = (4, 0))
    (origin_intersection : distance (0, 0) f1 + distance (0, 0) f2 = 5) :
    ∃ x : ℝ, (distance (x, 0) f1 + distance (x, 0) f2 = 5 ∧ x > 0 ∧ x ≠ 0 → x = 28 / 9) :=
by 
  sorry

end ellipse_intersection_l209_209331


namespace square_area_4900_l209_209225

/-- If one side of a square is increased by 3.5 times and the other side is decreased by 30 cm, resulting in a rectangle that has twice the area of the square, then the area of the square is 4900 square centimeters. -/
theorem square_area_4900 (x : ℝ) (h1 : 3.5 * x * (x - 30) = 2 * x^2) : x^2 = 4900 :=
sorry

end square_area_4900_l209_209225


namespace probability_one_unit_apart_l209_209141

/-
We define the points on the 3x3 grid as a Finset.
We will define the total number of points and the property that two points are one unit apart.
-/

/-- A 3x3 grid with 10 points spaced around at intervals of one unit. -/
def points_on_grid : Finset (ℕ × ℕ) := { -- Here we define the set of 10 points
  (0, 0), (0, 2), (2, 0), (2, 2), -- Corners
  (1, 0), (1, 2), (0, 1), (2, 1), -- Mid-points of sides
  (1, 1) -- Center
}

-- Function to determine if two points are one unit apart
def one_unit_apart (a b : ℕ × ℕ) : Prop :=
  (abs (a.1 - b.1) + abs (a.2 - b.2) = 1)

/-- The probability calculation proof statement -/
theorem probability_one_unit_apart :
  (∃! (p1 p2 : (ℕ × ℕ)), ((p1 ∈ points_on_grid) ∧ (p2 ∈ points_on_grid) ∧ (one_unit_apart p1 p2))) →
  ∃! (ratio : ℚ), ratio = 16 / 45 :=
by
  sorry

end probability_one_unit_apart_l209_209141


namespace find_f_2010_l209_209639

noncomputable def f (a b α β x : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem find_f_2010 (a b α β : ℝ) (h : f a b α β 2009 = 3) : f a b α β 2010 = -3 :=
  sorry

end find_f_2010_l209_209639


namespace basketball_shots_l209_209229

variable (x y : ℕ)

theorem basketball_shots : 3 * x + 2 * y = 26 ∧ x + y = 11 → x = 4 :=
by
  intros h
  sorry

end basketball_shots_l209_209229


namespace smallest_integer_is_nine_l209_209894

theorem smallest_integer_is_nine 
  (a b c : ℕ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0) 
  (h2 : a + b + c = 90) 
  (h3 : (a:ℝ)/b = 2/3) 
  (h4 : (b:ℝ)/c = 3/5) : 
  a = 9 :=
by 
  sorry

end smallest_integer_is_nine_l209_209894


namespace age_difference_l209_209052

variables (O N A : ℕ)

theorem age_difference (avg_age_stable : 10 * A = 10 * A + 50 - O + N) :
  O - N = 50 :=
by
  -- proof would go here
  sorry

end age_difference_l209_209052


namespace barbie_bruno_trips_l209_209619

theorem barbie_bruno_trips (coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : 
  coconuts = 144 → barbie_capacity = 4 → bruno_capacity = 8 → (coconuts / (barbie_capacity + bruno_capacity) = 12) :=
by 
  intros h_coconuts h_barbie h_bruno
  rw [h_coconuts, h_barbie, h_bruno]
  norm_num
  sorry

end barbie_bruno_trips_l209_209619


namespace staplers_left_is_correct_l209_209149

-- Define the initial conditions as constants
def initial_staplers : ℕ := 450
def stacie_reports : ℕ := 8 * 12 -- Stacie's reports in dozens converted to actual number
def jack_reports : ℕ := 9 * 12   -- Jack's reports in dozens converted to actual number
def laura_reports : ℕ := 50      -- Laura's individual reports

-- Define the stapler usage rates
def stacie_usage_rate : ℕ := 1                  -- Stacie's stapler usage rate (1 stapler per report)
def jack_usage_rate : ℕ := stacie_usage_rate / 2  -- Jack's stapler usage rate (half of Stacie's)
def laura_usage_rate : ℕ := stacie_usage_rate * 2 -- Laura's stapler usage rate (twice of Stacie's)

-- Define the usage calculations
def stacie_usage : ℕ := stacie_reports * stacie_usage_rate
def jack_usage : ℕ := jack_reports * jack_usage_rate
def laura_usage : ℕ := laura_reports * laura_usage_rate

-- Define total staplers used
def total_usage : ℕ := stacie_usage + jack_usage + laura_usage

-- Define the number of staplers left
def staplers_left : ℕ := initial_staplers - total_usage

-- Prove that the staplers left is 200
theorem staplers_left_is_correct : staplers_left = 200 := by
  unfold staplers_left initial_staplers total_usage stacie_usage jack_usage laura_usage
  unfold stacie_reports jack_reports laura_reports
  unfold stacie_usage_rate jack_usage_rate laura_usage_rate
  sorry   -- Place proof here

end staplers_left_is_correct_l209_209149


namespace max_square_plots_l209_209598
-- Lean 4 statement for the equivalent math problem

theorem max_square_plots (w l f s : ℕ) (h₁ : w = 40) (h₂ : l = 60) 
                         (h₃ : f = 2400) (h₄ : s ≠ 0) (h₅ : 2400 - 100 * s ≤ 2400)
                         (h₆ : w % s = 0) (h₇ : l % s = 0) :
  (w * l) / (s * s) = 6 :=
by {
  sorry
}

end max_square_plots_l209_209598


namespace length_of_platform_l209_209035

theorem length_of_platform
  (length_of_train time_crossing_platform time_crossing_pole : ℝ) 
  (length_of_train_eq : length_of_train = 400)
  (time_crossing_platform_eq : time_crossing_platform = 45)
  (time_crossing_pole_eq : time_crossing_pole = 30) :
  ∃ (L : ℝ), (400 + L) / time_crossing_platform = length_of_train / time_crossing_pole :=
by {
  use 200,
  sorry
}

end length_of_platform_l209_209035


namespace greatest_integer_third_side_of_triangle_l209_209573

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l209_209573


namespace mike_total_rose_bushes_l209_209135

-- Definitions based on the conditions
def costPerRoseBush : ℕ := 75
def costPerTigerToothAloe : ℕ := 100
def numberOfRoseBushesForFriend : ℕ := 2
def totalExpenseByMike : ℕ := 500
def numberOfTigerToothAloe : ℕ := 2

-- The total number of rose bushes Mike bought
noncomputable def totalNumberOfRoseBushes : ℕ :=
  let totalSpentOnAloes := numberOfTigerToothAloe * costPerTigerToothAloe
  let amountSpentOnRoseBushes := totalExpenseByMike - totalSpentOnAloes
  let numberOfRoseBushesForMike := amountSpentOnRoseBushes / costPerRoseBush
  numberOfRoseBushesForMike + numberOfRoseBushesForFriend

-- The theorem to prove
theorem mike_total_rose_bushes : totalNumberOfRoseBushes = 6 :=
  by
    sorry

end mike_total_rose_bushes_l209_209135


namespace top_quality_soccer_balls_l209_209594

theorem top_quality_soccer_balls (N : ℕ) (f : ℝ) (hN : N = 10000) (hf : f = 0.975) : N * f = 9750 := by
  sorry

end top_quality_soccer_balls_l209_209594


namespace fraction_of_remaining_birds_left_l209_209288

theorem fraction_of_remaining_birds_left :
  ∀ (total_birds initial_fraction next_fraction x : ℚ), 
    total_birds = 60 ∧ 
    initial_fraction = 1 / 3 ∧ 
    next_fraction = 2 / 5 ∧ 
    8 = (total_birds * (1 - initial_fraction)) * (1 - next_fraction) * (1 - x) →
    x = 2 / 3 :=
by
  intros total_birds initial_fraction next_fraction x h
  obtain ⟨hb, hi, hn, he⟩ := h
  sorry

end fraction_of_remaining_birds_left_l209_209288


namespace range_of_k_l209_209846

def f (x : ℝ) : ℝ := x^3 - 12*x

def not_monotonic_on_I (k : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), k - 1 < x₁ ∧ x₁ < k + 1 ∧ k - 1 < x₂ ∧ x₂ < k + 1 ∧ x₁ ≠ x₂ ∧ (f x₁ - f x₂) * (x₁ - x₂) < 0

theorem range_of_k (k : ℝ) : not_monotonic_on_I k ↔ (k > -3 ∧ k < -1) ∨ (k > 1 ∧ k < 3) :=
sorry

end range_of_k_l209_209846


namespace arithmetic_sequence_sum_l209_209235

variable (a : ℕ → ℕ)
variable (h_arith_seq : ∀ n : ℕ, a (n+1) - a n = a 2 - a 1)

theorem arithmetic_sequence_sum (h : a 2 + a 8 = 6) : 
  1 / 2 * 9 * (a 1 + a 9) = 27 :=
by 
  sorry

end arithmetic_sequence_sum_l209_209235


namespace exponent_equality_l209_209651

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l209_209651


namespace total_sides_of_cookie_cutters_l209_209488

theorem total_sides_of_cookie_cutters :
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  total_sides = 75 :=
by
  let top_layer := 6 * 3
  let middle_layer := 4 * 4 + 2 * 6
  let bottom_layer := 3 * 8 + 5 * 0 + 1 * 5
  let total_sides := top_layer + middle_layer + bottom_layer
  show total_sides = 75
  sorry

end total_sides_of_cookie_cutters_l209_209488


namespace problem_l209_209223

theorem problem (m : ℝ) (h : m^2 + 3 * m = -1) : m - 1 / (m + 1) = -2 :=
by
  sorry

end problem_l209_209223


namespace population_difference_is_16_l209_209660

def total_birds : ℕ := 250

def pigeons_percent : ℕ := 30
def sparrows_percent : ℕ := 25
def crows_percent : ℕ := 20
def swans_percent : ℕ := 15
def parrots_percent : ℕ := 10

def black_pigeons_percent : ℕ := 60
def white_pigeons_percent : ℕ := 40
def black_male_pigeons_percent : ℕ := 20
def white_female_pigeons_percent : ℕ := 50

def female_sparrows_percent : ℕ := 60
def male_sparrows_percent : ℕ := 40

def female_crows_percent : ℕ := 30
def male_crows_percent : ℕ := 70

def male_parrots_percent : ℕ := 65
def female_parrots_percent : ℕ := 35

noncomputable
def black_male_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (black_pigeons_percent * (black_male_pigeons_percent / 100)) / 100
noncomputable
def white_female_pigeons : ℕ := (pigeons_percent * total_birds / 100) * (white_pigeons_percent * (white_female_pigeons_percent / 100)) / 100
noncomputable
def male_sparrows : ℕ := (sparrows_percent * total_birds / 100) * (male_sparrows_percent / 100)
noncomputable
def female_crows : ℕ := (crows_percent * total_birds / 100) * (female_crows_percent / 100)
noncomputable
def male_parrots : ℕ := (parrots_percent * total_birds / 100) * (male_parrots_percent / 100)

noncomputable
def max_population : ℕ := max (max (max (max black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots
noncomputable
def min_population : ℕ := min (min (min (min black_male_pigeons white_female_pigeons) male_sparrows) female_crows) male_parrots

noncomputable
def population_difference : ℕ := max_population - min_population

theorem population_difference_is_16 : population_difference = 16 :=
sorry

end population_difference_is_16_l209_209660


namespace triangle_area_transform_l209_209186

-- Define the concept of a triangle with integer coordinates
structure Triangle :=
  (A : ℤ × ℤ)
  (B : ℤ × ℤ)
  (C : ℤ × ℤ)

-- Define the area of a triangle using determinant
def triangle_area (T : Triangle) : ℤ :=
  let ⟨(x1, y1), (x2, y2), (x3, y3)⟩ := (T.A, T.B, T.C)
  abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

-- Define a legal transformation for triangles
def legal_transform (T : Triangle) : Set Triangle :=
  { T' : Triangle |
    (∃ c : ℤ, 
      (T'.A = (T.A.1 + c * (T.B.1 - T.C.1), T.A.2 + c * (T.B.2 - T.C.2)) ∧ T'.B = T.B ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = (T.B.1 + c * (T.A.1 - T.C.1), T.B.2 + c * (T.A.2 - T.C.2)) ∧ T'.C = T.C) ∨
      (T'.A = T.A ∧ T'.B = T.B ∧ T'.C = (T.C.1 + c * (T.A.1 - T.B.1), T.C.2 + c * (T.A.2 - T.B.2)))) }

-- Proposition that any two triangles with equal area can be legally transformed into each other
theorem triangle_area_transform (T1 T2 : Triangle) (h : triangle_area T1 = triangle_area T2) :
  ∃ (T' : Triangle), T' ∈ legal_transform T1 ∧ triangle_area T' = triangle_area T2 :=
sorry

end triangle_area_transform_l209_209186


namespace boxes_left_l209_209420

theorem boxes_left (received : ℕ) (brother : ℕ) (sister : ℕ) (cousin : ℕ)
  (h_received : received = 45)
  (h_brother : brother = 12)
  (h_sister : sister = 9)
  (h_cousin : cousin = 7) :
  received - (brother + sister + cousin) = 17 :=
by
  rw [h_received, h_brother, h_sister, h_cousin]
  norm_num
  sorry

end boxes_left_l209_209420


namespace incorrect_formula_l209_209757

-- The conditions of the problem
def students : ℕ := 60
def class_president : ℕ := 1
def vice_president : ℕ := 1
def selected_students : ℕ := 5

-- The question: Which formula is incorrect given the conditions?
theorem incorrect_formula :
  let A := choose 2 1 * choose 59 4 in
  let B := choose 60 5 - choose 58 5 in
  let C := choose 2 1 * choose 59 4 - choose 2 2 * choose 58 3 in
  let D := choose 2 1 * choose 58 4 + choose 2 2 * choose 58 3 in
  A ≠ (B ∨ C ∨ D) :=
sorry -- Prove that A is incorrect under the given conditions

end incorrect_formula_l209_209757


namespace option_one_better_than_option_two_l209_209973

/-- Define the probability of winning in the first lottery option (drawing two red balls from a box
containing 4 red balls and 2 white balls). -/
def probability_option_one : ℚ := 2 / 5

/-- Define the probability of winning in the second lottery option (rolling two dice and having at least one die show a four). -/
def probability_option_two : ℚ := 11 / 36

/-- Prove that the probability of winning in the first lottery option is greater than the probability of winning in the second lottery option. -/
theorem option_one_better_than_option_two : probability_option_one > probability_option_two :=
by sorry

end option_one_better_than_option_two_l209_209973


namespace min_expression_min_expression_achieve_l209_209825

theorem min_expression (x : ℝ) (hx : 0 < x) : 
  (x^2 + 8 * x + 64 / x^3) ≥ 28 :=
sorry

theorem min_expression_achieve (x : ℝ) (hx : x = 2): 
  (x^2 + 8 * x + 64 / x^3) = 28 :=
sorry

end min_expression_min_expression_achieve_l209_209825


namespace total_interest_l209_209613

def P : ℝ := 1000
def r : ℝ := 0.1
def n : ℕ := 3

theorem total_interest : (P * (1 + r)^n) - P = 331 := by
  sorry

end total_interest_l209_209613


namespace triangle_angles_l209_209389

theorem triangle_angles (A B C : ℝ) 
  (h1 : B = 4 * A)
  (h2 : C - B = 27)
  (h3 : A + B + C = 180) : 
  A = 17 ∧ B = 68 ∧ C = 95 :=
by {
  -- Sorry will be replaced once the actual proof is provided
  sorry 
}

end triangle_angles_l209_209389


namespace pastrami_sandwich_cost_l209_209262

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l209_209262


namespace current_speed_correct_l209_209319

noncomputable def speed_of_current : ℝ :=
  let rowing_speed_still_water := 10 -- speed of rowing in still water in kmph
  let distance_meters := 60 -- distance covered in meters
  let time_seconds := 17.998560115190788 -- time taken in seconds
  let distance_km := distance_meters / 1000 -- converting distance to kilometers
  let time_hours := time_seconds / 3600 -- converting time to hours
  let downstream_speed := distance_km / time_hours -- calculating downstream speed
  downstream_speed - rowing_speed_still_water -- calculating and returning the speed of the current

theorem current_speed_correct : speed_of_current = 2.00048 := by
  -- The proof is not provided in this statement as per the requirements.
  sorry

end current_speed_correct_l209_209319


namespace min_value_alpha_beta_gamma_l209_209640

def is_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def is_cube (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = n

def is_fifth_power (n : ℕ) : Prop :=
  ∃ k : ℕ, k ^ 5 = n

def A (α β γ : ℕ) : ℕ := 2 ^ α * 3 ^ β * 5 ^ γ

def condition_1 (α β γ : ℕ) : Prop :=
  is_square (A α β γ / 2)

def condition_2 (α β γ : ℕ) : Prop :=
  is_cube (A α β γ / 3)

def condition_3 (α β γ : ℕ) : Prop :=
  is_fifth_power (A α β γ / 5)

theorem min_value_alpha_beta_gamma (α β γ : ℕ) :
  condition_1 α β γ → condition_2 α β γ → condition_3 α β γ →
  α + β + γ = 31 :=
sorry

end min_value_alpha_beta_gamma_l209_209640


namespace remainder_142_to_14_l209_209042

theorem remainder_142_to_14 (N k : ℤ) 
  (h : N = 142 * k + 110) : N % 14 = 8 :=
sorry

end remainder_142_to_14_l209_209042


namespace min_value_of_expr_l209_209836

theorem min_value_of_expr {a b : ℝ} (ha : a > 0) (hb : b > 0) (h : a + b = (1 / a) + (1 / b)) :
  ∃ x : ℝ, x = (1 / a) + (2 / b) ∧ x = 2 * Real.sqrt 2 :=
sorry

end min_value_of_expr_l209_209836


namespace compound_weight_distribution_l209_209624

theorem compound_weight_distribution :
  let total_weight := 500
  let ratio_A := 2
  let ratio_B := 10
  let ratio_C := 5
  let total_parts := ratio_A + ratio_B + ratio_C
  let weight_per_part := (total_weight : ℝ) / total_parts
in 
  (ratio_A * weight_per_part = 58.82) ∧ 
  (ratio_B * weight_per_part = 294.12) ∧ 
  (ratio_C * weight_per_part = 147.06) :=
by
  sorry

end compound_weight_distribution_l209_209624


namespace gcd_f_101_102_l209_209248

def f (x : ℕ) : ℕ := x^2 + x + 2010

theorem gcd_f_101_102 : Nat.gcd (f 101) (f 102) = 12 := 
by sorry

end gcd_f_101_102_l209_209248


namespace contractor_realized_work_done_after_20_days_l209_209169

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l209_209169


namespace Mary_paid_on_Tuesday_l209_209256

theorem Mary_paid_on_Tuesday 
  (credit_limit total_spent paid_on_thursday remaining_payment paid_on_tuesday : ℝ)
  (h1 : credit_limit = 100)
  (h2 : total_spent = credit_limit)
  (h3 : paid_on_thursday = 23)
  (h4 : remaining_payment = 62)
  (h5 : total_spent = paid_on_thursday + remaining_payment + paid_on_tuesday) :
  paid_on_tuesday = 15 :=
sorry

end Mary_paid_on_Tuesday_l209_209256


namespace algebraic_identity_specific_case_l209_209682

theorem algebraic_identity (a b : ℝ) : (a - b)^2 = a^2 + b^2 - 2 * a * b :=
by sorry

theorem specific_case : 2021^2 - 2021 * 4034 + 2017^2 = 16 :=
by sorry

end algebraic_identity_specific_case_l209_209682


namespace bananas_to_oranges_l209_209421

theorem bananas_to_oranges :
  (3 / 4) * 12 * b = 9 * o →
  ((3 / 5) * 15 * b) = 9 * o := 
by
  sorry

end bananas_to_oranges_l209_209421


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209814

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209814


namespace calculate_m_l209_209650

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l209_209650


namespace g_g2_is_394_l209_209222

def g (x : ℝ) : ℝ :=
  4 * x^2 - 6

theorem g_g2_is_394 : g(g(2)) = 394 :=
by
  -- Proof is omitted by using sorry
  sorry

end g_g2_is_394_l209_209222


namespace chess_team_arrangements_l209_209879

def num_arrangements (boys girls : Nat) (alice_end : Bool) (boys_adjacent : Bool) : Nat :=
  if alice_end && boys_adjacent then 
    2 * 4 * 2 * 6
  else 
    0

theorem chess_team_arrangements :
  num_arrangements 2 4 true true = 96 :=
by 
  simp [num_arrangements]
  sorry

end chess_team_arrangements_l209_209879


namespace total_cows_in_ranch_l209_209297

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l209_209297


namespace sec_150_eq_neg_two_div_sqrt_three_l209_209805

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l209_209805


namespace base7_addition_l209_209243

variable (A B C : ℕ)

def distinct_nonzero_digits_lt_eq_six := (A ≠ B ∧ A ≠ C ∧ B ≠ C) ∧ 
                                          (A > 0 ∧ B > 0 ∧ C > 0) ∧ 
                                          (A ≤ 6 ∧ B ≤ 6 ∧ C ≤ 6)

-- Prove the given base 7 addition problem
theorem base7_addition (h : distinct_nonzero_digits_lt_eq_six A B C) 
  (h_add : ∀ base7_add_prop : 
    let base7_add_prop : (A * 49 + B * 7 + C) + (B * 7 + C) = A * 49 + C * 7 + A := by sorry) :
  (A + B + C : ℕ) ≡ 12 [MOD 7] :=
begin
  sorry,
end

end base7_addition_l209_209243


namespace sum_of_perimeters_geq_4400_l209_209985

theorem sum_of_perimeters_geq_4400 (side original_side : ℕ) 
  (h_side_le_10 : ∀ s, s ≤ side → s ≤ 10) 
  (h_original_square : original_side = 100) 
  (h_cut_condition : side ≤ 10) : 
  ∃ (small_squares : ℕ → ℕ × ℕ), (original_side / side = n) → 4 * n * side ≥ 4400 :=
by
  sorry

end sum_of_perimeters_geq_4400_l209_209985


namespace range_of_x_l209_209579

theorem range_of_x (x : ℤ) : x^2 < 3 * x → x = 1 ∨ x = 2 :=
by
  sorry

end range_of_x_l209_209579


namespace find_y_from_expression_l209_209777

theorem find_y_from_expression :
  ∀ y : ℕ, 2^10 + 2^10 + 2^10 + 2^10 = 4^y → y = 6 :=
by
  sorry

end find_y_from_expression_l209_209777


namespace cylinder_surface_area_l209_209424

noncomputable def total_surface_area_cylinder (r h : ℝ) : ℝ :=
  let base_area := 64 * Real.pi
  let lateral_surface_area := 2 * Real.pi * r * h
  let total_surface_area := 2 * base_area + lateral_surface_area
  total_surface_area

theorem cylinder_surface_area (r h : ℝ) (hr : Real.pi * r^2 = 64 * Real.pi) (hh : h = 2 * r) : 
  total_surface_area_cylinder r h = 384 * Real.pi := by
  sorry

end cylinder_surface_area_l209_209424


namespace yoojeong_initial_correct_l209_209031

variable (yoojeong_initial yoojeong_after marbles_given : ℕ)

-- Given conditions
axiom marbles_given_cond : marbles_given = 8
axiom yoojeong_after_cond : yoojeong_after = 24

-- Equation relating initial, given marbles, and marbles left
theorem yoojeong_initial_correct : 
  yoojeong_initial = yoojeong_after + marbles_given := by
  -- Proof skipped
  sorry

end yoojeong_initial_correct_l209_209031


namespace sec_150_l209_209819

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l209_209819


namespace determine_k_l209_209486

def f(x : ℝ) : ℝ := 5 * x^2 - 3 * x + 8
def g(x k : ℝ) : ℝ := x^3 - k * x - 10

theorem determine_k : 
  (f (-5) - g (-5) k = -24) → k = 61 := 
by 
-- Begin the proof script here
sorry

end determine_k_l209_209486


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209718

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209718


namespace solve_system_of_equations_l209_209685

theorem solve_system_of_equations (x y : ℤ) (h1 : x + y = 8) (h2 : x - 3 * y = 4) : x = 7 ∧ y = 1 :=
by {
    -- Proof would go here
    sorry
}

end solve_system_of_equations_l209_209685


namespace odd_and_increasing_f1_odd_and_increasing_f2_l209_209581

-- Define the functions
def f1 (x : ℝ) : ℝ := x * |x|
def f2 (x : ℝ) : ℝ := x^3

-- Define the odd function property
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)

-- Define the increasing function property
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → f x1 < f x2

-- Lean statement to prove
theorem odd_and_increasing_f1 : is_odd f1 ∧ is_increasing f1 := by
  sorry

theorem odd_and_increasing_f2 : is_odd f2 ∧ is_increasing f2 := by
  sorry

end odd_and_increasing_f1_odd_and_increasing_f2_l209_209581


namespace ladder_distance_from_wall_l209_209939

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l209_209939


namespace intersection_of_sets_l209_209642

def A := {1, 6, 8, 10}
def B := {2, 4, 8, 10}

theorem intersection_of_sets : A ∩ B = {8, 10} :=
by
  sorry

end intersection_of_sets_l209_209642


namespace puzzle_permutations_l209_209101

/--
The number of distinct arrangements of the letters in the word "puzzle",
where the letter "z" appears twice, is 360.
-/
theorem puzzle_permutations : 
  ∀ (word : list Char),
  (word = ['p', 'u', 'z', 'z', 'l', 'e']) →
  (Nat.factorial 6) / (Nat.factorial 2) = 360 :=
by
  intros word h_word
  sorry

end puzzle_permutations_l209_209101


namespace ladder_distance_l209_209930

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l209_209930


namespace value_of_expression_l209_209635

theorem value_of_expression (x y z : ℝ) (h : (x * y * z) / (|x * y * z|) = 1) :
  (|x| / x + y / |y| + |z| / z) = 3 ∨ (|x| / x + y / |y| + |z| / z) = -1 :=
sorry

end value_of_expression_l209_209635


namespace constant_term_expansion_l209_209274

noncomputable def binomial_expansion_term (r : ℕ) : ℤ :=
  (-1)^r * (2 : ℤ)^(6 - r) * (Nat.choose 6 r : ℤ) 

theorem constant_term_expansion : (x : ℤ) = 60 :=
by
  have term_4 := binomial_expansion_term 4
  -- We know r = 4 gives the constant term
  -- Calculating the specific term
  have : term_4 = (-1)^4 * 2^(6-4) * Nat.choose 6 4 := rfl
  have : term_4 = 1 * 2^2 * 15 := rfl
  have : term_4 = 4 * 15 := rfl
  have : 4 * 15 = 60 := by ring
  exact this

end constant_term_expansion_l209_209274


namespace escalator_steps_l209_209021

theorem escalator_steps
  (x : ℕ)
  (time_me : ℕ := 60)
  (steps_me : ℕ := 20)
  (time_wife : ℕ := 72)
  (steps_wife : ℕ := 16)
  (escalator_speed_me : x - steps_me = 60 * (x - 20) / 72)
  (escalator_speed_wife : x - steps_wife = 72 * (x - 16) / 60) :
  x = 40 := by
  sorry

end escalator_steps_l209_209021


namespace probability_same_number_l209_209413

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l209_209413


namespace symmetric_about_z_correct_l209_209237

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_z (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_about_z_correct (p : Point3D) :
  p = {x := 3, y := 4, z := 5} → symmetric_about_z p = {x := -3, y := -4, z := 5} :=
by
  sorry

end symmetric_about_z_correct_l209_209237


namespace club_membership_l209_209522

def total_people_in_club (T B TB N : ℕ) : ℕ :=
  T + B - TB + N

theorem club_membership : total_people_in_club 138 255 94 11 = 310 := by
  sorry

end club_membership_l209_209522


namespace find_y_coordinate_l209_209662

theorem find_y_coordinate (m n : ℝ) 
  (h₁ : m = 2 * n + 5) 
  (h₂ : m + 5 = 2 * (n + 2.5) + 5) : 
  n = (m - 5) / 2 := 
sorry

end find_y_coordinate_l209_209662


namespace negation_of_universal_proposition_l209_209002

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, |x - 1| - |x + 1| ≤ 3) ↔ ∃ x : ℝ, |x - 1| - |x + 1| > 3 :=
by
  sorry

end negation_of_universal_proposition_l209_209002


namespace ladder_distance_l209_209943

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l209_209943


namespace total_soccer_balls_donated_l209_209983

-- Defining the conditions
def elementary_classes_per_school := 4
def middle_classes_per_school := 5
def schools := 2
def soccer_balls_per_class := 5

-- Proving the total number of soccer balls donated
theorem total_soccer_balls_donated : 
  (soccer_balls_per_class * (schools * (elementary_classes_per_school + middle_classes_per_school))) = 90 := 
by
  -- Using the given conditions to compute the numbers
  let total_classes_per_school := elementary_classes_per_school + middle_classes_per_school
  let total_classes := total_classes_per_school * schools
  let total_soccer_balls := soccer_balls_per_class * total_classes
  -- Prove the equivalence
  show total_soccer_balls = 90
  from sorry

end total_soccer_balls_donated_l209_209983


namespace gcd_1234_2047_l209_209081

theorem gcd_1234_2047 : Nat.gcd 1234 2047 = 1 :=
by sorry

end gcd_1234_2047_l209_209081


namespace f_at_7_l209_209367

-- Define the function f and its properties
axiom f : ℝ → ℝ
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom periodic_f : ∀ x : ℝ, f (x + 4) = f x
axiom values_f : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = 2 * x^2

-- Prove that f(7) = -2
theorem f_at_7 : f 7 = -2 :=
by
  sorry

end f_at_7_l209_209367


namespace tenth_number_in_twentieth_row_l209_209270

def arrangement : ∀ n : ℕ, ℕ := -- A function defining the nth number in the sequence.
  sorry

-- A function to get the nth number in the mth row, respecting the arithmetic sequence property.
def number_in_row (m n : ℕ) : ℕ := 
  sorry

theorem tenth_number_in_twentieth_row : number_in_row 20 10 = 426 :=
  sorry

end tenth_number_in_twentieth_row_l209_209270


namespace james_lifting_ratio_correct_l209_209997

theorem james_lifting_ratio_correct :
  let lt_initial := 2200
  let bw_initial := 245
  let lt_gain_percentage := 0.15
  let bw_gain := 8
  let lt_final := lt_initial + lt_initial * lt_gain_percentage
  let bw_final := bw_initial + bw_gain
  (lt_final / bw_final) = 10 :=
by
  sorry

end james_lifting_ratio_correct_l209_209997


namespace diorama_factor_l209_209774

theorem diorama_factor (P B factor : ℕ) (h1 : P + B = 67) (h2 : B = P * factor - 5) (h3 : B = 49) : factor = 3 :=
by
  sorry

end diorama_factor_l209_209774


namespace martha_black_butterflies_l209_209404

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l209_209404


namespace find_a_conditions_l209_209496

theorem find_a_conditions (a : ℝ) : 
    (∃ m : ℤ, a = m + 1/2) ∨ (∃ m : ℤ, a = m + 1/3) ∨ (∃ m : ℤ, a = m - 1/3) ↔ 
    (∃ n : ℤ, a = n + 1/2 ∨ a = n + 1/3 ∨ a = n - 1/3) :=
by
  sorry

end find_a_conditions_l209_209496


namespace solve_y_determinant_l209_209087

theorem solve_y_determinant (b y : ℝ) (hb : b ≠ 0) :
  Matrix.det ![
    ![y + b, y, y], 
    ![y, y + b, y], 
    ![y, y, y + b]
  ] = 0 ↔ y = -b / 3 :=
by
  sorry

end solve_y_determinant_l209_209087


namespace decreasing_power_function_l209_209357

open Nat

/-- For the power function y = x^(m^2 + 2*m - 3) (where m : ℕ) 
    to be a decreasing function in the interval (0, +∞), prove that m = 0. -/
theorem decreasing_power_function (m : ℕ) (h : m^2 + 2 * m - 3 < 0) : m = 0 := 
by
  sorry

end decreasing_power_function_l209_209357


namespace tap_B_fills_remaining_pool_l209_209990

theorem tap_B_fills_remaining_pool :
  ∀ (flow_A flow_B : ℝ) (t_A t_B : ℕ),
  flow_A = 7.5 / 100 →  -- A fills 7.5% of the pool per hour
  flow_B = 5 / 100 →    -- B fills 5% of the pool per hour
  t_A = 2 →             -- A is open for 2 hours during the second phase
  t_A * flow_A = 15 / 100 →  -- A fills 15% of the pool in 2 hours
  4 * (flow_A + flow_B) = 50 / 100 →  -- A and B together fill 50% of the pool in 4 hours
  (100 / 100 - 50 / 100 - 15 / 100) / flow_B = t_B →  -- remaining pool filled only by B
  t_B = 7 := sorry    -- Prove that t_B is 7

end tap_B_fills_remaining_pool_l209_209990


namespace sec_150_eq_neg_two_div_sqrt_three_l209_209806

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l209_209806


namespace carpet_length_is_9_l209_209772

noncomputable def carpet_length (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) : ℝ :=
  living_room_area * coverage / width

theorem carpet_length_is_9 (width : ℝ) (living_room_area : ℝ) (coverage : ℝ) (length := carpet_length width living_room_area coverage) :
    width = 4 → living_room_area = 48 → coverage = 0.75 → length = 9 := by
  intros
  sorry

end carpet_length_is_9_l209_209772


namespace Emily_subtract_59_l209_209899

theorem Emily_subtract_59 : (30 - 1) ^ 2 = 30 ^ 2 - 59 := by
  sorry

end Emily_subtract_59_l209_209899


namespace count_integers_abs_inequality_l209_209102

theorem count_integers_abs_inequality : 
  ∃ n : ℕ, n = 15 ∧ ∀ x : ℤ, |(x: ℝ) - 3| ≤ 7.2 ↔ x ∈ {i : ℤ | -4 ≤ i ∧ i ≤ 10} := 
by 
  sorry

end count_integers_abs_inequality_l209_209102


namespace students_at_end_l209_209113

def initial_students : ℝ := 42.0
def students_left : ℝ := 4.0
def students_transferred : ℝ := 10.0

theorem students_at_end : initial_students - students_left - students_transferred = 28.0 :=
by
  -- Proof omitted
  sorry

end students_at_end_l209_209113


namespace Barbier_theorem_for_delta_curves_l209_209155

def delta_curve (h : ℝ) : Type := sorry 
def can_rotate_freely_in_3gon (K : delta_curve h) : Prop := sorry
def length_of_curve (K : delta_curve h) : ℝ := sorry

theorem Barbier_theorem_for_delta_curves
  (K : delta_curve h)
  (h : ℝ)
  (H : can_rotate_freely_in_3gon K)
  : length_of_curve K = (2 * Real.pi * h) / 3 := 
sorry

end Barbier_theorem_for_delta_curves_l209_209155


namespace ladder_base_distance_l209_209922

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l209_209922


namespace binomial_coefficient_x3y5_in_expansion_l209_209729

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209729


namespace john_remaining_money_l209_209392

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end john_remaining_money_l209_209392


namespace base_from_wall_l209_209964

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l209_209964


namespace part1_part2_l209_209099

variables (a b : ℝ → E)
variables (k : ℝ)
variables [inner_product_space ℝ E]

-- Given conditions
def norm_a : real := 2
def norm_b : real := 1
def angle_ab : real := 60
def dot_ab : real := norm_a * norm_b * real.cos(angle_ab * real.pi / 180)

def c : E := 2 • a + 3 • b
def d : E := 3 • a + k • b

-- Part 1: Proving k for orthogonality
theorem part1 (h1 : inner c d = 0) : k = -33/5 :=
by sorry

-- Part 2: Proving k for magnitude of d
theorem part2 (h2 : ∥d∥ = 2 * sqrt 13) : k = 2 ∨ k = -8 :=
by sorry

end part1_part2_l209_209099


namespace solution_set_transformation_l209_209839

variables (a b c α β : ℝ) (h_root : (α : ℝ) > 0)

open Set

def quadratic_inequality (x : ℝ) : Prop :=
  a * x^2 + b * x + c > 0

def transformed_inequality (x : ℝ) : Prop :=
  c * x^2 + b * x + a < 0

theorem solution_set_transformation :
  (∀ x, quadratic_inequality a b c x ↔ (α < x ∧ x < β)) →
  (∃ α β : ℝ, α > 0 ∧ (∀ x, transformed_inequality c b a x ↔ (x < 1/β ∨ x > 1/α))) :=
by
  sorry

end solution_set_transformation_l209_209839


namespace total_pies_sold_l209_209063

theorem total_pies_sold :
  let shepherd_slices := 52
  let chicken_slices := 80
  let shepherd_pieces_per_pie := 4
  let chicken_pieces_per_pie := 5
  let shepherd_pies := shepherd_slices / shepherd_pieces_per_pie
  let chicken_pies := chicken_slices / chicken_pieces_per_pie
  shepherd_pies + chicken_pies = 29 :=
by
  sorry

end total_pies_sold_l209_209063


namespace base_from_wall_l209_209963

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l209_209963


namespace compare_abc_l209_209216

theorem compare_abc (a b c : ℝ)
  (h1 : a = Real.log 0.9 / Real.log 2)
  (h2 : b = 3 ^ (-1 / 3 : ℝ))
  (h3 : c = (1 / 3 : ℝ) ^ (1 / 2 : ℝ)) :
  a < c ∧ c < b := by
  sorry

end compare_abc_l209_209216


namespace liam_total_money_l209_209253

-- Define the conditions as noncomputable since they involve monetary calculations
noncomputable def liam_money (initial_bottles : ℕ) (price_per_bottle : ℕ) (bottles_sold : ℕ) (extra_money : ℕ) : ℚ :=
  let cost := initial_bottles * price_per_bottle
  let money_after_selling_part := cost + extra_money
  let selling_price_per_bottle := money_after_selling_part / bottles_sold
  let total_revenue := initial_bottles * selling_price_per_bottle
  total_revenue

-- State the theorem with the given problem
theorem liam_total_money :
  let initial_bottles := 50
  let price_per_bottle := 1
  let bottles_sold := 40
  let extra_money := 10
  liam_money initial_bottles price_per_bottle bottles_sold extra_money = 75 := 
sorry

end liam_total_money_l209_209253


namespace nature_of_roots_l209_209356

noncomputable def custom_operation (a b : ℝ) : ℝ := a^2 - a * b

theorem nature_of_roots :
  ∀ x : ℝ, custom_operation (x + 1) 3 = -2 → ∃ x1 x2 : ℝ, x1 ≠ x2 :=
by
  intro x h
  sorry

end nature_of_roots_l209_209356


namespace train_speed_l209_209467

theorem train_speed (length_train length_bridge time : ℝ) (h_train : length_train = 125) (h_bridge : length_bridge = 250) (h_time : time = 30) :
    (length_train + length_bridge) / time * 3.6 = 45 := by
  sorry

end train_speed_l209_209467


namespace selling_price_l209_209330

/-- 
Prove that the selling price (S) of an article with a cost price (C) of 180 sold at a 15% profit (P) is 207.
-/
theorem selling_price (C P S : ℝ) (hC : C = 180) (hP : P = 15) (hS : S = 207) :
  S = C + (P / 100 * C) :=
by
  -- here we rely on sorry to skip the proof details
  sorry

end selling_price_l209_209330


namespace inequality_solution_range_l209_209509

theorem inequality_solution_range (a : ℝ) : (∃ x : ℝ, |x+2| + |x-3| < a) ↔ a > 5 :=
by
  sorry

end inequality_solution_range_l209_209509


namespace coefficient_x3y5_in_expansion_l209_209720

theorem coefficient_x3y5_in_expansion :
  (∑ k in Finset.range 9,
      Nat.choose 8 k * (x ^ k) * (y ^ (8 - k))) =
  56 := by
  sorry

end coefficient_x3y5_in_expansion_l209_209720


namespace king_middle_school_teachers_l209_209855

theorem king_middle_school_teachers 
    (students : ℕ)
    (classes_per_student : ℕ)
    (normal_class_size : ℕ)
    (special_classes : ℕ)
    (special_class_size : ℕ)
    (classes_per_teacher : ℕ)
    (H1 : students = 1500)
    (H2 : classes_per_student = 5)
    (H3 : normal_class_size = 30)
    (H4 : special_classes = 10)
    (H5 : special_class_size = 15)
    (H6 : classes_per_teacher = 3) : 
    ∃ teachers : ℕ, teachers = 85 :=
by
  sorry

end king_middle_school_teachers_l209_209855


namespace exists_within_distance_l209_209534

theorem exists_within_distance (a : ℝ) (n : ℕ) (h₁ : a > 0) (h₂ : n > 0) :
  ∃ k : ℕ, k < n ∧ ∃ m : ℤ, |k * a - m| < 1 / n :=
by
  sorry

end exists_within_distance_l209_209534


namespace coloring_possible_l209_209251

theorem coloring_possible (n k : ℕ) (h1 : 2 ≤ k ∧ k ≤ n) (h2 : n ≥ 2) :
  (n ≥ k ∧ k ≥ 3) ∨ (2 ≤ k ∧ k ≤ n ∧ n ≤ 3) :=
sorry

end coloring_possible_l209_209251


namespace manuscript_typing_cost_l209_209582

theorem manuscript_typing_cost :
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 :=
by
  let total_pages := 100
  let revised_once_pages := 30
  let revised_twice_pages := 20
  let cost_first_time := 10
  let cost_revision := 5
  let cost_first_typing := total_pages * cost_first_time
  let cost_revisions_once := revised_once_pages * cost_revision
  let cost_revisions_twice := revised_twice_pages * (2 * cost_revision)
  have : cost_first_typing + cost_revisions_once + cost_revisions_twice = 1350 := sorry
  exact this

end manuscript_typing_cost_l209_209582


namespace coefficient_x3y5_in_expansion_l209_209708

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l209_209708


namespace final_result_is_110_l209_209603

def chosen_number : ℕ := 63
def multiplier : ℕ := 4
def subtracted_value : ℕ := 142

def final_result : ℕ := (chosen_number * multiplier) - subtracted_value

theorem final_result_is_110 : final_result = 110 := by
  sorry

end final_result_is_110_l209_209603


namespace g_g_2_eq_394_l209_209218

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l209_209218


namespace school_fee_correct_l209_209865

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l209_209865


namespace min_value_of_a_for_inverse_l209_209900

theorem min_value_of_a_for_inverse (a : ℝ) : 
  (∀ x y : ℝ, x ≥ a → y ≥ a → (x^2 + 4*x ≤ y^2 + 4*y ↔ x ≤ y)) → a = -2 :=
by
  sorry

end min_value_of_a_for_inverse_l209_209900


namespace adults_wearing_hats_l209_209311

theorem adults_wearing_hats (total_adults : ℕ) (percent_men : ℝ) (percent_men_hats : ℝ) 
  (percent_women_hats : ℝ) (num_hats : ℕ) 
  (h1 : total_adults = 3600) 
  (h2 : percent_men = 0.40) 
  (h3 : percent_men_hats = 0.15) 
  (h4 : percent_women_hats = 0.25) 
  (h5 : num_hats = 756) : 
  (percent_men * total_adults) * percent_men_hats + (total_adults - (percent_men * total_adults)) * percent_women_hats = num_hats := 
sorry

end adults_wearing_hats_l209_209311


namespace opposite_sides_of_line_l209_209845

theorem opposite_sides_of_line (m : ℝ) 
  (ha : (m + 0 - 1) * (2 + m - 1) < 0): 
  -1 < m ∧ m < 1 :=
sorry

end opposite_sides_of_line_l209_209845


namespace prove_ratio_l209_209000

def P_and_Q (P Q : ℤ) : Prop :=
  ∀ x : ℝ, x ≠ -5 → x ≠ 0 → x ≠ 6 →
  P * x * (x - 6) + Q * (x + 5) = x^2 - 4 * x + 20

theorem prove_ratio (P Q : ℤ) (h : P_and_Q P Q) : Q / P = 4 :=
by
  sorry

end prove_ratio_l209_209000


namespace coconut_transport_l209_209614

theorem coconut_transport (coconuts total_coconuts barbie_capacity bruno_capacity combined_capacity trips : ℕ)
  (h1 : total_coconuts = 144)
  (h2 : barbie_capacity = 4)
  (h3 : bruno_capacity = 8)
  (h4 : combined_capacity = barbie_capacity + bruno_capacity)
  (h5 : combined_capacity = 12)
  (h6 : trips = total_coconuts / combined_capacity) :
  trips = 12 :=
by
  sorry

end coconut_transport_l209_209614


namespace find_d_l209_209648

theorem find_d (d : ℤ) :
  (∀ x : ℤ, 6 * x^3 + 19 * x^2 + d * x - 15 = 0) ->
  d = -32 :=
by
  sorry

end find_d_l209_209648


namespace fraction_doubled_l209_209384

variable (x y : ℝ)

theorem fraction_doubled (x y : ℝ) : 
  (x + y) ≠ 0 → (2 * x * 2 * y) / (2 * x + 2 * y) = 2 * (x * y / (x + y)) := 
by
  intro h
  sorry

end fraction_doubled_l209_209384


namespace at_least_two_dice_same_number_probability_l209_209409

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l209_209409


namespace number_of_scenarios_l209_209059

theorem number_of_scenarios :
  ∃ (count : ℕ), count = 42244 ∧
  (∃ (x1 x2 x3 x4 x5 x6 x7 : ℕ),
    x1 % 7 = 0 ∧ x2 % 7 = 0 ∧ x3 % 7 = 0 ∧ x4 % 7 = 0 ∧
    x5 % 13 = 0 ∧ x6 % 13 = 0 ∧ x7 % 13 = 0 ∧
    x1 + x2 + x3 + x4 + x5 + x6 + x7 = 270) :=
sorry

end number_of_scenarios_l209_209059


namespace problem_1_problem_2_l209_209833

variable (x y : ℝ)
noncomputable def x_val : ℝ := 2 + Real.sqrt 3
noncomputable def y_val : ℝ := 2 - Real.sqrt 3

theorem problem_1 :
  3 * x_val^2 + 5 * x_val * y_val + 3 * y_val^2 = 47 := sorry

theorem problem_2 :
  Real.sqrt (x_val / y_val) + Real.sqrt (y_val / x_val) = 4 := sorry

end problem_1_problem_2_l209_209833


namespace initial_erasers_in_box_l209_209017

-- Definitions based on the conditions
def erasers_in_bag_jane := 15
def erasers_taken_out_doris := 54
def erasers_left_in_box := 15

-- Theorem statement
theorem initial_erasers_in_box : ∃ B_i : ℕ, B_i = erasers_taken_out_doris + erasers_left_in_box ∧ B_i = 69 :=
by
  use 69
  -- omitted proof steps
  sorry

end initial_erasers_in_box_l209_209017


namespace sum_first_six_geom_seq_l209_209477

-- Definitions based on the conditions given in the problem
def a : ℚ := 1 / 6
def r : ℚ := 1 / 2
def n : ℕ := 6

-- Statement to prove the desired result
theorem sum_first_six_geom_seq : 
  geom_series a r n = 21 / 64 := by
  sorry

end sum_first_six_geom_seq_l209_209477


namespace inequality_a4_b4_c4_geq_l209_209162

theorem inequality_a4_b4_c4_geq (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
by
  sorry

end inequality_a4_b4_c4_geq_l209_209162


namespace sam_travel_time_l209_209583

theorem sam_travel_time (d_AC d_CB : ℕ) (v_sam : ℕ) 
  (h1 : d_AC = 600) (h2 : d_CB = 400) (h3 : v_sam = 50) : 
  (d_AC + d_CB) / v_sam = 20 := 
by
  sorry

end sam_travel_time_l209_209583


namespace greatest_third_side_of_triangle_l209_209571

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : ∃ x : ℕ, x < a + b ∧ x = 16 := by
  use 16
  rw [h1, h2]
  split
  · linarith
  · rfl

end greatest_third_side_of_triangle_l209_209571


namespace first_term_of_arithmetic_sequence_l209_209245

theorem first_term_of_arithmetic_sequence (T : ℕ → ℝ) (b : ℝ) 
  (h1 : ∀ n : ℕ, T n = (n * (2 * b + (n - 1) * 4)) / 2) 
  (h2 : ∃ d : ℝ, ∀ n : ℕ, T (4 * n) / T n = d) :
  b = 2 :=
by
  sorry

end first_term_of_arithmetic_sequence_l209_209245


namespace license_plate_configurations_l209_209602

theorem license_plate_configurations :
  (3 * 10^4 = 30000) :=
by
  sorry

end license_plate_configurations_l209_209602


namespace evaluate_expression_at_2_l209_209116

noncomputable def replace_and_evaluate (x : ℝ) : ℝ :=
  (3 * x - 2) / (-x + 6)

theorem evaluate_expression_at_2 :
  replace_and_evaluate 2 = -2 :=
by
  -- evaluation and computation would go here, skipped with sorry
  sorry

end evaluate_expression_at_2_l209_209116


namespace find_fx_l209_209201

theorem find_fx (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = x^2 - 2 * x) : ∀ x, f x = x^2 - 1 :=
by
  intro x
  sorry

end find_fx_l209_209201


namespace letters_received_per_day_l209_209560

-- Define the conditions
def packages_per_day := 20
def total_pieces_in_six_months := 14400
def days_in_month := 30
def months := 6

-- Calculate total days in six months
def total_days := months * days_in_month

-- Calculate pieces of mail per day
def pieces_per_day := total_pieces_in_six_months / total_days

-- Define the number of letters per day
def letters_per_day := pieces_per_day - packages_per_day

-- Prove that the number of letters per day is 60
theorem letters_received_per_day : letters_per_day = 60 := sorry

end letters_received_per_day_l209_209560


namespace shakes_sold_l209_209318

variable (s : ℕ) -- the number of shakes sold

-- conditions
def shakes_ounces := 4 * s
def cone_ounces := 6
def total_ounces := 14

-- the theorem to prove
theorem shakes_sold : shakes_ounces + cone_ounces = total_ounces → s = 2 := by
  intros h
  -- proof can be filled in here
  sorry

end shakes_sold_l209_209318


namespace greatest_integer_third_side_of_triangle_l209_209574

theorem greatest_integer_third_side_of_triangle (x : ℕ) (h1 : 7 + 10 > x) (h2 : x > 3) : x = 16 :=
by
  sorry

end greatest_integer_third_side_of_triangle_l209_209574


namespace domain_of_f_l209_209275

-- Given conditions
def f (x : ℝ) : ℝ := 1 / real.sqrt (2 * x - 3)
def dom := { x : ℝ | 2 * x - 3 > 0 }

-- Prove that the domain of the function f is (3/2, +∞)
theorem domain_of_f :
  ∀ x : ℝ, (x ∈ dom ↔ x > 3/2) :=
by sorry

end domain_of_f_l209_209275


namespace probability_same_number_l209_209414

theorem probability_same_number (n k : ℕ) (h₁ : n = 8) (h₂ : k = 6) : 
  (∃ m : ℝ, 0 ≤ m ∧ m ≤ 1 ∧ m = 1) := by
  sorry

end probability_same_number_l209_209414


namespace ladder_distance_from_wall_l209_209938

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l209_209938


namespace abs_expression_not_positive_l209_209452

theorem abs_expression_not_positive (x : ℝ) (h : |2 * x - 7| = 0) : x = 7 / 2 :=
by
  sorry

end abs_expression_not_positive_l209_209452


namespace ladder_base_distance_l209_209948

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l209_209948


namespace worker_total_amount_l209_209608

-- Definitions of the conditions
def pay_per_day := 20
def deduction_per_idle_day := 3
def total_days := 60
def idle_days := 40
def worked_days := total_days - idle_days
def earnings := worked_days * pay_per_day
def deductions := idle_days * deduction_per_idle_day

-- Statement of the problem
theorem worker_total_amount : earnings - deductions = 280 := by
  sorry

end worker_total_amount_l209_209608


namespace ladder_base_distance_l209_209957

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l209_209957


namespace number_of_male_rabbits_l209_209015

-- Definitions based on the conditions
def white_rabbits : ℕ := 12
def black_rabbits : ℕ := 9
def female_rabbits : ℕ := 8

-- The question and proof goal
theorem number_of_male_rabbits : 
  (white_rabbits + black_rabbits - female_rabbits) = 13 :=
by
  sorry

end number_of_male_rabbits_l209_209015


namespace increase_percentage_when_selfcheckout_broken_l209_209687

-- The problem conditions as variable definitions and declarations
def normal_complaints : ℕ := 120
def short_staffed_increase : ℚ := 1 / 3
def short_staffed_complaints : ℕ := normal_complaints + (normal_complaints / 3)
def total_complaints_three_days : ℕ := 576
def days : ℕ := 3
def both_conditions_complaints : ℕ := total_complaints_three_days / days

-- The theorem that we need to prove
theorem increase_percentage_when_selfcheckout_broken : 
  (both_conditions_complaints - short_staffed_complaints) * 100 / short_staffed_complaints = 20 := 
by
  -- This line sets up that the conclusion is true
  sorry

end increase_percentage_when_selfcheckout_broken_l209_209687


namespace speed_of_stream_l209_209593

theorem speed_of_stream (b s : ℝ) 
  (H1 : b + s = 10)
  (H2 : b - s = 4) : 
  s = 3 :=
sorry

end speed_of_stream_l209_209593


namespace inverse_log_value_l209_209506

noncomputable def f : ℝ → ℝ := sorry

-- Given that f is the inverse function of y = log2(x)
theorem inverse_log_value :
  (∀ x, f (log 2 x) = x) → 
  (∀ x, log 2 (f x) = x) → 
  f 3 = log 3 2 :=
by
  intros h1 h2
  -- Proof goes here
  sorry

end inverse_log_value_l209_209506


namespace sum_cis_angles_l209_209888

noncomputable def complex.cis (θ : ℝ) := Complex.exp (Complex.I * θ)

theorem sum_cis_angles :
  (complex.cis (80 * Real.pi / 180) + complex.cis (88 * Real.pi / 180) + complex.cis (96 * Real.pi / 180) + 
  complex.cis (104 * Real.pi / 180) + complex.cis (112 * Real.pi / 180) + complex.cis (120 * Real.pi / 180) + 
  complex.cis (128 * Real.pi / 180)) = r * complex.cis (104 * Real.pi / 180) := 
sorry

end sum_cis_angles_l209_209888


namespace win_sector_area_l209_209184

noncomputable def radius : ℝ := 8
noncomputable def probability : ℝ := 1 / 4
noncomputable def total_area : ℝ := Real.pi * radius^2

theorem win_sector_area :
  ∃ (W : ℝ), W = probability * total_area ∧ W = 16 * Real.pi :=
by
  -- Proof skipped
  sorry

end win_sector_area_l209_209184


namespace complex_magnitude_equality_l209_209347

open Complex Real

theorem complex_magnitude_equality :
  abs ((Complex.mk (5 * sqrt 2) (-5)) * (Complex.mk (2 * sqrt 3) 6)) = 60 :=
by
  sorry

end complex_magnitude_equality_l209_209347


namespace smaller_root_of_equation_l209_209074

theorem smaller_root_of_equation :
  ∀ x : ℚ, (x - 7 / 8)^2 + (x - 1/4) * (x - 7 / 8) = 0 → x = 9 / 16 :=
by
  intro x
  intro h
  sorry

end smaller_root_of_equation_l209_209074


namespace united_airlines_discount_l209_209623

theorem united_airlines_discount :
  ∀ (delta_price original_price_u discount_delta discount_u saved_amount cheapest_price: ℝ),
    delta_price = 850 →
    original_price_u = 1100 →
    discount_delta = 0.20 →
    saved_amount = 90 →
    cheapest_price = delta_price * (1 - discount_delta) - saved_amount →
    discount_u = (original_price_u - cheapest_price) / original_price_u →
    discount_u = 0.4636363636 :=
by
  intros delta_price original_price_u discount_delta discount_u saved_amount cheapest_price δeq ueq deq saeq cpeq dueq
  -- Placeholder for the actual proof steps
  sorry

end united_airlines_discount_l209_209623


namespace red_flower_ratio_l209_209039

theorem red_flower_ratio
  (total : ℕ)
  (O : ℕ)
  (P Pu : ℕ)
  (R Y : ℕ)
  (h_total : total = 105)
  (h_orange : O = 10)
  (h_pink_purple : P + Pu = 30)
  (h_equal_pink_purple : P = Pu)
  (h_yellow : Y = R - 5)
  (h_sum : R + Y + O + P + Pu = total) :
  (R / O) = 7 / 2 :=
by
  sorry

end red_flower_ratio_l209_209039


namespace exists_zero_in_interval_l209_209439

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem exists_zero_in_interval : 
  (f 2) * (f 3) < 0 := by
  sorry

end exists_zero_in_interval_l209_209439


namespace sec_150_eq_neg_two_sqrt_three_div_three_l209_209789

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l209_209789


namespace min_S6_minus_S4_l209_209859

variable {a₁ a₂ q : ℝ} (h1 : q > 1) (h2 : (q^2 - 1) * (a₁ + a₂) = 3)

theorem min_S6_minus_S4 : 
  ∃ (a₁ a₂ q : ℝ), q > 1 ∧ (q^2 - 1) * (a₁ + a₂) = 3 ∧ (q^4 * (a₁ + a₂) - (a₁ + a₂ + a₂ * q + a₂ * q^2) = 12) := sorry

end min_S6_minus_S4_l209_209859


namespace solution_correct_l209_209823

noncomputable def solution_set : Set ℝ :=
  {x | (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x)}

theorem solution_correct (x : ℝ) :
  (1 / (x * (x + 1)) - 1 / ((x + 1) * (x + 2))) < 1 / 4 ↔ (x < -2) ∨ (-1 < x ∧ x < 0) ∨ (1 < x) :=
by sorry

end solution_correct_l209_209823


namespace tangent_line_equation_l209_209556

open Real
open TopologicalSpace
open Filter
open Asymptotics

def curve (x : ℝ) : ℝ := x * exp x + 2 * x - 1

def point := (0 : ℝ, -1 : ℝ)

theorem tangent_line_equation : ∃ m b, 
  (∀ x, curve x - (-1) = m * (x - 0)) ∧ m = 3 ∧ b = -1 :=
sorry

end tangent_line_equation_l209_209556


namespace distance_between_cities_l209_209030

def distance_thing 
  (d_A d_B : ℝ) 
  (v_A v_B : ℝ) 
  (t_diff : ℝ) : Prop :=
d_A = (3 / 5) * d_B ∧
v_A = 72 ∧
v_B = 108 ∧
t_diff = (1 / 4) ∧
(d_A + d_B) = 432

theorem distance_between_cities
  (d_A d_B : ℝ)
  (v_A v_B : ℝ)
  (t_diff : ℝ)
  (h : distance_thing d_A d_B v_A v_B t_diff)
  : d_A + d_B = 432 := by
  sorry

end distance_between_cities_l209_209030


namespace sec_150_eq_l209_209821

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l209_209821


namespace min_value_a_l209_209382

theorem min_value_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x ≤ (Real.sqrt 2) / 2 → x^3 - 2 * x * Real.log x / Real.log a ≤ 0) ↔ a ≥ 1 / 4 := 
sorry

end min_value_a_l209_209382


namespace base_distance_l209_209962

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l209_209962


namespace multiple_of_Mel_weight_l209_209621

/-- Given that Brenda weighs 10 pounds more than a certain multiple of Mel's weight,
    and given that Brenda weighs 220 pounds and Mel's weight is 70 pounds,
    show that the multiple is 3. -/
theorem multiple_of_Mel_weight 
    (Brenda_weight Mel_weight certain_multiple : ℝ) 
    (h1 : Brenda_weight = Mel_weight * certain_multiple + 10)
    (h2 : Brenda_weight = 220)
    (h3 : Mel_weight = 70) :
  certain_multiple = 3 :=
by 
  sorry

end multiple_of_Mel_weight_l209_209621


namespace manny_has_more_10_bills_than_mandy_l209_209538

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l209_209538


namespace probability_no_shaded_square_l209_209588

/-
  Given:
  A 2 by 2001 rectangle consists of unit squares.
  The middle unit square of each row is shaded.
  
  Prove:
  The probability that a randomly chosen rectangle from the figure does not include a shaded square is 1001 / 2001.
-/
theorem probability_no_shaded_square (rect_width : ℕ) (rect_height : ℕ) (mid_shaded : ℕ) :
  rect_width = 2001 → rect_height = 2 → mid_shaded = 1001 →
  let n := ((rect_width + 1) * rect_width) / 2 in
  let m := mid_shaded * (rect_width - mid_shaded) in
  let total_no_shaded_rectangles := 1 - (m / n : ℚ) in
  total_no_shaded_rectangles = 1001 / 2001 :=
by
  intros h_width h_height h_mid_shaded
  let n := ((2001 + 1) * 2001) / 2
  let m := 1001 * (2001 - 1001)
  let total_no_shaded_rectangles : ℚ := 1 - (m / n)
  show total_no_shaded_rectangles = 1001 / 2001
  sorry

end probability_no_shaded_square_l209_209588


namespace valid_call_time_at_15_l209_209437

def time_difference := 5 -- Beijing is 5 hours ahead of Moscow

def beijing_start_time := 14 -- Start time in Beijing corresponding to 9:00 in Moscow
def beijing_end_time := 17  -- End time in Beijing corresponding to 17:00 in Beijing

-- Define the call time in Beijing
def call_time_beijing := 15

-- The time window during which they can start the call in Beijing
def valid_call_time (t : ℕ) : Prop :=
  beijing_start_time <= t ∧ t <= beijing_end_time

-- The theorem to prove that 15:00 is a valid call time in Beijing
theorem valid_call_time_at_15 : valid_call_time call_time_beijing :=
by
  sorry

end valid_call_time_at_15_l209_209437


namespace flower_pots_on_path_count_l209_209136

theorem flower_pots_on_path_count (L d : ℕ) (hL : L = 15) (hd : d = 3) : 
  (L / d) + 1 = 6 :=
by
  sorry

end flower_pots_on_path_count_l209_209136


namespace larger_number_of_two_l209_209278

theorem larger_number_of_two (A B : ℕ) (hcf : ℕ) (factor1 factor2 : ℕ) 
  (h_hcf : hcf = 28) (h_factors : A % hcf = 0 ∧ B % hcf = 0) 
  (h_f1 : factor1 = 12) (h_f2 : factor2 = 15)
  (h_lcm : Nat.lcm A B = hcf * factor1 * factor2)
  (h_coprime : Nat.gcd (A / hcf) (B / hcf) = 1)
  : max A B = 420 := 
sorry

end larger_number_of_two_l209_209278


namespace indistinguishable_balls_boxes_l209_209104

theorem indistinguishable_balls_boxes (n m : ℕ) (h : n = 6) (k : m = 2) : 
  (finset.card (finset.filter (λ x : finset ℕ, x.card ≤ n / 2) 
    (finset.powerset (finset.range (n + 1)))) = 4) :=
by
  sorry

end indistinguishable_balls_boxes_l209_209104


namespace boxes_amount_l209_209317

/-- 
  A food company has 777 kilograms of food to put into boxes. 
  If each box gets a certain amount of kilograms, they will have 388 full boxes.
  Prove that each box gets 2 kilograms of food.
-/
theorem boxes_amount (total_food : ℕ) (boxes : ℕ) (kilograms_per_box : ℕ) 
  (h_total : total_food = 777)
  (h_boxes : boxes = 388) :
  total_food / boxes = kilograms_per_box :=
by {
  -- Skipped proof
  sorry 
}

end boxes_amount_l209_209317


namespace proportion_solution_l209_209304

theorem proportion_solution (x : ℚ) (h : 0.75 / x = 7 / 8) : x = 6 / 7 :=
by sorry

end proportion_solution_l209_209304


namespace distance_from_wall_l209_209968

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l209_209968


namespace rooms_equation_l209_209661

theorem rooms_equation (x : ℕ) (h₁ : ∃ n, n = 6 * (x - 1)) (h₂ : ∃ m, m = 5 * x + 4) :
  6 * (x - 1) = 5 * x + 4 :=
sorry

end rooms_equation_l209_209661


namespace customers_served_total_l209_209991

theorem customers_served_total :
  let Ann_hours := 8
  let Ann_rate := 7
  let Becky_hours := 7
  let Becky_rate := 8
  let Julia_hours := 6
  let Julia_rate := 6
  let lunch_break := 0.5
  let Ann_customers := (Ann_hours - lunch_break) * Ann_rate
  let Becky_customers := (Becky_hours - lunch_break) * Becky_rate
  let Julia_customers := (Julia_hours - lunch_break) * Julia_rate
  Ann_customers + Becky_customers + Julia_customers = 137 := by
  sorry

end customers_served_total_l209_209991


namespace tom_speed_RB_l209_209442

/-- Let d be the distance between B and C (in miles).
    Let 2d be the distance between R and B (in miles).
    Let v be Tom’s speed driving from R to B (in mph).
    Given conditions:
    1. Tom's speed from B to C = 20 mph.
    2. Total average speed of the whole journey = 36 mph.
    Prove that Tom's speed driving from R to B is 60 mph. -/
theorem tom_speed_RB
  (d : ℝ) (v : ℝ)
  (h1 : 20 ≠ 0)
  (h2 : 36 ≠ 0)
  (avg_speed : 3 * d / (2 * d / v + d / 20) = 36) :
  v = 60 := 
sorry

end tom_speed_RB_l209_209442


namespace can_construct_polygon_l209_209014

def match_length : ℕ := 2
def number_of_matches : ℕ := 12
def total_length : ℕ := number_of_matches * match_length
def required_area : ℝ := 16

theorem can_construct_polygon : 
  (∃ (P : Polygon), P.perimeter = total_length ∧ P.area = required_area) := 
sorry

end can_construct_polygon_l209_209014


namespace product_b1_b13_l209_209641

variable {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Conditions for the arithmetic sequence
def is_arithmetic_seq (a : ℕ → ℝ) := ∀ n m k : ℕ, m > 0 → k > 0 → a (n + m) - a n = a (n + k) - a (n + k - m)

-- Conditions for the geometric sequence
def is_geometric_seq (b : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = r * b n

-- Given conditions
def conditions (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  (a 3 - (a 7 ^ 2) / 2 + a 11 = 0) ∧ (b 7 = a 7)

theorem product_b1_b13 
  (ha : is_arithmetic_seq a)
  (hb : is_geometric_seq b)
  (h : conditions a b) :
  b 1 * b 13 = 16 :=
sorry

end product_b1_b13_l209_209641


namespace sum_of_ages_l209_209993

/--
Given:
- Beckett's age is 12.
- Olaf is 3 years older than Beckett.
- Shannen is 2 years younger than Olaf.
- Jack is 5 more than twice as old as Shannen.

Prove that the sum of the ages of Beckett, Olaf, Shannen, and Jack is 71 years.
-/
theorem sum_of_ages :
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  Beckett + Olaf + Shannen + Jack = 71 :=
by
  let Beckett := 12
  let Olaf := Beckett + 3
  let Shannen := Olaf - 2
  let Jack := 2 * Shannen + 5
  show Beckett + Olaf + Shannen + Jack = 71
  sorry

end sum_of_ages_l209_209993


namespace ladder_base_distance_l209_209918

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l209_209918


namespace total_cows_in_ranch_l209_209296

def WeThePeopleCows : ℕ := 17
def HappyGoodHealthyFamilyCows : ℕ := 3 * WeThePeopleCows + 2

theorem total_cows_in_ranch : WeThePeopleCows + HappyGoodHealthyFamilyCows = 70 := by
  sorry

end total_cows_in_ranch_l209_209296


namespace cauchy_problem_solution_l209_209683

noncomputable def solution (y : ℝ → ℝ) (x : ℝ) : Prop :=
  y x = (x^2) / 2 + (x^3) / 6 + (x^4) / 12 + (x^5) / 20 + x + 1

theorem cauchy_problem_solution (y : ℝ → ℝ) (x : ℝ) 
  (h1: ∀ x, (deriv^[2] y) x = 1 + x + x^2 + x^3)
  (h2: y 0 = 1)
  (h3: deriv y 0 = 1) : 
  solution y x := 
by
  -- Proof Steps
  sorry

end cauchy_problem_solution_l209_209683


namespace age_ratio_proof_l209_209763

-- Define the ages
def sonAge := 22
def manAge := sonAge + 24

-- Define the ratio computation statement
def ageRatioInTwoYears : ℚ := 
  let sonAgeInTwoYears := sonAge + 2
  let manAgeInTwoYears := manAge + 2
  manAgeInTwoYears / sonAgeInTwoYears

-- The theorem to prove
theorem age_ratio_proof : ageRatioInTwoYears = 2 :=
by
  sorry

end age_ratio_proof_l209_209763


namespace votes_cast_46800_l209_209111

-- Define the election context
noncomputable def total_votes (v : ℕ) : Prop :=
  let percentage_a := 0.35
  let percentage_b := 0.40
  let vote_diff := 2340
  (percentage_b - percentage_a) * (v : ℝ) = (vote_diff : ℝ)

-- Theorem stating the total number of votes cast in the election
theorem votes_cast_46800 : total_votes 46800 :=
by
  sorry

end votes_cast_46800_l209_209111


namespace pizza_left_for_Wally_l209_209578

theorem pizza_left_for_Wally (a b c : ℚ) (ha : a = 1/3) (hb : b = 1/6) (hc : c = 1/4) :
  1 - (a + b + c) = 1/4 :=
by
  sorry

end pizza_left_for_Wally_l209_209578


namespace sum_of_midpoint_coordinates_l209_209025

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end sum_of_midpoint_coordinates_l209_209025


namespace polygon_possible_with_area_sixteen_l209_209011

theorem polygon_possible_with_area_sixteen :
  ∃ (P : polygon) (matches : list (side P)), (length(matches) = 12 ∧ (∀ m ∈ matches, m.length = 2) ∧ P.area = 16) := 
sorry

end polygon_possible_with_area_sixteen_l209_209011


namespace rhombus_area_l209_209423

theorem rhombus_area (x y : ℝ) (h : |x - 1| + |y - 1| = 1) : 
  ∃ (area : ℝ), area = 2 :=
by
  sorry

end rhombus_area_l209_209423


namespace solutions_equiv_cond_l209_209283

theorem solutions_equiv_cond (a : ℝ) :
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x + 1 / (x - 1) = a + 1 / (x - 1)) ↔ 
  (∀ x : ℝ, x ≠ 1 → x^2 + 3 * x = a) ∧ (∃ x : ℝ, x = 1 → a ≠ 4)  :=
sorry

end solutions_equiv_cond_l209_209283


namespace f_one_eq_zero_l209_209339

-- Define the function f on ℝ
variable (f : ℝ → ℝ)

-- Conditions for the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f (x)
axiom periodic_function : ∀ x : ℝ, f (x + 2) = f (x)

-- Goal: Prove that f(1) = 0
theorem f_one_eq_zero : f 1 = 0 :=
by
  sorry

end f_one_eq_zero_l209_209339


namespace cost_of_football_correct_l209_209869

-- We define the variables for the costs
def total_amount_spent : ℝ := 20.52
def cost_of_marbles : ℝ := 9.05
def cost_of_baseball : ℝ := 6.52
def cost_of_football : ℝ := total_amount_spent - cost_of_marbles - cost_of_baseball

-- We now state what needs to be proven: that Mike spent $4.95 on the football.
theorem cost_of_football_correct : cost_of_football = 4.95 := by
  sorry

end cost_of_football_correct_l209_209869


namespace processing_time_600_parts_l209_209212

theorem processing_time_600_parts :
  ∀ (x: ℕ), x = 600 → (∃ y : ℝ, y = 0.01 * x + 0.5 ∧ y = 6.5) :=
by
  sorry

end processing_time_600_parts_l209_209212


namespace sum_first_8_terms_l209_209835

variable {α : Type*} [LinearOrderedField α]

-- Define the arithmetic sequence
def arithmetic_sequence (a_1 d : α) (n : ℕ) : α := a_1 + (n - 1) * d

-- Define the sum of the first n terms of the arithmetic sequence
def sum_arithmetic_sequence (a_1 d : α) (n : ℕ) : α :=
  (n * (2 * a_1 + (n - 1) * d)) / 2

-- Define the given condition
variable (a_1 d : α)
variable (h : arithmetic_sequence a_1 d 3 = 20 - arithmetic_sequence a_1 d 6)

-- Statement of the problem
theorem sum_first_8_terms : sum_arithmetic_sequence a_1 d 8 = 80 :=
by
  sorry

end sum_first_8_terms_l209_209835


namespace martha_black_butterflies_l209_209400

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l209_209400


namespace no_integer_solution_k_range_l209_209521

theorem no_integer_solution_k_range (k : ℝ) :
  (∀ x : ℤ, ¬ ((k * x - k^2 - 4) * (x - 4) < 0)) → (1 ≤ k ∧ k ≤ 4) :=
by
  sorry

end no_integer_solution_k_range_l209_209521


namespace tomatoes_multiplier_l209_209525

theorem tomatoes_multiplier (before_vacation : ℕ) (grown_during_vacation : ℕ)
  (h1 : before_vacation = 36)
  (h2 : grown_during_vacation = 3564) :
  (before_vacation + grown_during_vacation) / before_vacation = 100 :=
by
  -- Insert proof here later
  sorry

end tomatoes_multiplier_l209_209525


namespace polynomial_equality_l209_209085

theorem polynomial_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 3) + 5) + 1 = -x^4 + 3*x^3 - 3*x^2 + 5*x + 1 :=
by 
  sorry

end polynomial_equality_l209_209085


namespace a_n_general_term_b_n_general_term_l209_209627

noncomputable def seq_a (n : ℕ) : ℕ :=
  2 * n - 1

theorem a_n_general_term (n : ℕ) (Sn : ℕ → ℕ) (S_property : ∀ n : ℕ, 4 * Sn n = (seq_a n) ^ 2 + 2 * seq_a n + 1) :
  seq_a n = 2 * n - 1 :=
sorry

noncomputable def geom_seq (q : ℕ) (n : ℕ) : ℕ :=
  q ^ (n - 1)

theorem b_n_general_term (n m q : ℕ) (a1 am am3 : ℕ) (b_property : ∀ n : ℕ, geom_seq q n = q ^ (n - 1))
  (a_property : ∀ n : ℕ, seq_a n = 2 * n - 1)
  (b1_condition : geom_seq q 1 = seq_a 1) (bm_condition : geom_seq q m = seq_a m)
  (bm1_condition : geom_seq q (m + 1) = seq_a (m + 3)) :
  q = 3 ∨ q = 7 ∧ (∀ n : ℕ, geom_seq q n = 3 ^ (n - 1) ∨ geom_seq q n = 7 ^ (n - 1)) :=
sorry

end a_n_general_term_b_n_general_term_l209_209627


namespace gcd_calculation_l209_209448

def gcd_36_45_495 : ℕ :=
  Int.gcd 36 (Int.gcd 45 495)

theorem gcd_calculation : gcd_36_45_495 = 9 := by
  sorry

end gcd_calculation_l209_209448


namespace find_solutions_l209_209492

def is_solution (a b c d : ℕ) : Prop :=
  Nat.gcd (Nat.gcd (Nat.gcd a b) c) d = 1 ∧
  a ∣ (b + c) ∧
  b ∣ (c + d) ∧
  c ∣ (d + a) ∧
  d ∣ (a + b)

theorem find_solutions : ∀ (a b c d : ℕ),
  is_solution a b c d →
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 2 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 3 ∧ b = 2 ∧ c = 1 ∧ d = 1) ∨
  (a = 5 ∧ b = 3 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 4 ∧ c = 1 ∧ d = 3) ∨
  (a = 7 ∧ b = 5 ∧ c = 2 ∧ d = 3) ∨
  (a = 3 ∧ b = 1 ∧ c = 2 ∧ d = 1) ∨
  (a = 5 ∧ b = 1 ∧ c = 4 ∧ d = 3) ∨
  (a = 5 ∧ b = 2 ∧ c = 3 ∧ d = 1) ∨
  (a = 7 ∧ b = 2 ∧ c = 5 ∧ d = 3) ∨
  (a = 7 ∧ b = 3 ∧ c = 4 ∧ d = 5) :=
by
  intros a b c d h
  sorry

end find_solutions_l209_209492


namespace expected_value_decisive_games_l209_209472

theorem expected_value_decisive_games :
  let X : ℕ → ℕ := -- Random variable representing the number of decisive games
    -- Expected value calculation for random variable X
    have h : ∃ e : ℕ, e = (2 * 1/2 + (2 + e) * 1/2), from sorry, 
    -- Extracting the expected value from the equation
    let ⟨E_X, h_ex⟩ := Classical.indefinite_description (λ e, e = (2 * 1/2 + (2 + e) * 1/2)) h in
    E_X = 4 :=
begin
  sorry,
end

end expected_value_decisive_games_l209_209472


namespace banana_production_total_l209_209592

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l209_209592


namespace Elise_paid_23_dollars_l209_209022

-- Definitions and conditions
def base_price := 3
def cost_per_mile := 4
def distance := 5

-- Desired conclusion (total cost)
def total_cost := base_price + cost_per_mile * distance

-- Theorem statement
theorem Elise_paid_23_dollars : total_cost = 23 := by
  sorry

end Elise_paid_23_dollars_l209_209022


namespace trapezoid_area_l209_209609

-- Define the problem context
variables (b : ℝ) (theta : ℝ)
def is_isosceles (a c : ℝ) := 
  a = c

def is_inscribed (b : ℝ) := 
  ∀ (r : ℝ), r > 0

theorem trapezoid_area
  (h1 : is_isosceles 18 18)
  (h2 : is_inscribed 18)
  (h3 : \arccos(0.6) = theta)
  : trapezoid_area 18 theta = 101.25 :=
sorry

end trapezoid_area_l209_209609


namespace ratio_proof_l209_209656

variable (x y z : ℝ)
variable (h1 : y / z = 1 / 2)
variable (h2 : z / x = 2 / 3)
variable (h3 : x / y = 3 / 1)

theorem ratio_proof : (x / (y * z)) / (y / (z * x)) = 4 / 1 := 
  sorry

end ratio_proof_l209_209656


namespace donut_selection_l209_209872

theorem donut_selection :
  ∃ (ways : ℕ), ways = Nat.choose 8 3 ∧ ways = 56 :=
by
  sorry

end donut_selection_l209_209872


namespace polynomial_real_root_l209_209080

theorem polynomial_real_root (a : ℝ) :
  (∃ x : ℝ, x^5 + a * x^4 - x^3 + a * x^2 + x + 1 = 0) ↔
  (a ∈ (Set.Iic (-1/2)) ∨ a ∈ (Set.Ici (1/2))) :=
by
  sorry

end polynomial_real_root_l209_209080


namespace height_difference_percentage_l209_209328

theorem height_difference_percentage (H_A H_B : ℝ) (h : H_B = H_A * 1.8181818181818183) :
  (H_A < H_B) → ((H_B - H_A) / H_B) * 100 = 45 := 
by 
  sorry

end height_difference_percentage_l209_209328


namespace watermelon_cost_is_100_rubles_l209_209979

theorem watermelon_cost_is_100_rubles :
  (∀ (x y k m n : ℕ) (a b : ℝ),
    x + y = k →
    n * a = m * b →
    n * a + m * b = 24000 →
    n = 120 →
    m = 30 →
    k = 150 →
    a = 100) :=
by
  intros x y k m n a b
  intros h1 h2 h3 h4 h5 h6
  have h7 : 120 * a = 30 * b, from h2
  have h8 : 120 * a + 30 * b = 24000, from h3
  have h9 : 120 * a = 12000, from sorry
  have h10 : a = 100, from sorry
  exact h10

end watermelon_cost_is_100_rubles_l209_209979


namespace sec_150_eq_neg_two_sqrt_three_div_three_l209_209787

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l209_209787


namespace max_real_solution_under_100_l209_209545

theorem max_real_solution_under_100 (k a b c r : ℕ) (h0 : ∃ (m n p : ℕ), a = k^m ∧ b = k^n ∧ c = k^p)
  (h1 : r < 100) (h2 : b^2 = 4 * a * c) (h3 : r = b / (2 * a)) : r ≤ 64 :=
sorry

end max_real_solution_under_100_l209_209545


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209804

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209804


namespace not_p_and_not_p_and_q_implies_not_p_or_q_l209_209507

theorem not_p_and_not_p_and_q_implies_not_p_or_q (p q : Prop) (h1 : ¬p) (h2 : ¬(p ∧ q)) : ¬(p ∨ q) :=
sorry

end not_p_and_not_p_and_q_implies_not_p_or_q_l209_209507


namespace ladder_base_distance_l209_209926

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l209_209926


namespace base_distance_l209_209958

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l209_209958


namespace compare_powers_l209_209748

theorem compare_powers :
  let a := 5 ^ 140
  let b := 3 ^ 210
  let c := 2 ^ 280
  c < a ∧ a < b := by
  -- Proof omitted
  sorry

end compare_powers_l209_209748


namespace fraction_of_occupied_student_chairs_is_4_over_5_l209_209009

-- Definitions based on the conditions provided
def total_chairs : ℕ := 10 * 15
def awardees_chairs : ℕ := 15
def admin_teachers_chairs : ℕ := 2 * 15
def parents_chairs : ℕ := 2 * 15
def student_chairs : ℕ := total_chairs - (awardees_chairs + admin_teachers_chairs + parents_chairs)
def vacant_student_chairs_given_to_parents : ℕ := 15
def occupied_student_chairs : ℕ := student_chairs - vacant_student_chairs_given_to_parents

-- Theorem statement based on the problem
theorem fraction_of_occupied_student_chairs_is_4_over_5 :
    (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 :=
by
    sorry

end fraction_of_occupied_student_chairs_is_4_over_5_l209_209009


namespace ladder_distance_l209_209945

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l209_209945


namespace spaceship_distance_l209_209047

-- Define the distance variables and conditions
variables (D : ℝ) -- Distance from Earth to Planet X
variable (T : ℝ) -- Total distance traveled by the spaceship

-- Conditions
variables (hx : T = 0.7) -- Total distance traveled is 0.7 light-years
variables (hy : D + 0.1 + 0.1 = T) -- Sum of distances along the path

-- Theorem statement to prove the distance from Earth to Planet X
theorem spaceship_distance (h1 : T = 0.7) (h2 : D + 0.1 + 0.1 = T) : D = 0.5 :=
by
  -- Proof steps would go here
  sorry

end spaceship_distance_l209_209047


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209725

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209725


namespace ladder_base_distance_l209_209951

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l209_209951


namespace common_ratio_geometric_sequence_l209_209523

theorem common_ratio_geometric_sequence (q : ℝ) (a : ℕ → ℝ)
  (h₀ : a 1 = 1)
  (h₁ : a 2 = q)
  (h₂ : a 3 = q^2)
  (h₃ : (4 * a 1 + a 3 = 2 * 2 * a 2)) :
  q = 2 :=
by sorry

end common_ratio_geometric_sequence_l209_209523


namespace guinea_pig_food_ratio_l209_209268

-- Definitions of amounts of food consumed by each guinea pig
def first_guinea_pig_food : ℕ := 2
variable (x : ℕ)
def second_guinea_pig_food : ℕ := x
def third_guinea_pig_food : ℕ := x + 3

-- Total food requirement condition
def total_food_required := first_guinea_pig_food + second_guinea_pig_food x + third_guinea_pig_food x = 13

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- The goal is to prove this ratio given the conditions
theorem guinea_pig_food_ratio (h : total_food_required x) : ratio (second_guinea_pig_food x) first_guinea_pig_food = 2 := by
  sorry

end guinea_pig_food_ratio_l209_209268


namespace watermelon_cost_100_l209_209975

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l209_209975


namespace find_sin_cos_of_perpendicular_vectors_l209_209512

theorem find_sin_cos_of_perpendicular_vectors 
  (θ : ℝ) 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (h_a : a = (Real.sin θ, -2)) 
  (h_b : b = (1, Real.cos θ)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) 
  (h_theta_range : 0 < θ ∧ θ < Real.pi / 2) : 
  Real.sin θ = 2 * Real.sqrt 5 / 5 ∧ Real.cos θ = Real.sqrt 5 / 5 := 
by 
  sorry

end find_sin_cos_of_perpendicular_vectors_l209_209512


namespace geometric_sequence_conditions_l209_209301

variable (a : ℕ → ℝ) (q : ℝ)

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = q * a n

theorem geometric_sequence_conditions (a : ℕ → ℝ) (q : ℝ)
  (h1 : geometric_sequence a q)
  (h2 : -1 < q)
  (h3 : q < 0) :
  (∀ n, a n * a (n + 1) < 0) ∧ (∀ n, |a n| > |a (n + 1)|) :=
by
  sorry

end geometric_sequence_conditions_l209_209301


namespace sec_150_eq_l209_209809

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l209_209809


namespace part_1_part_2_part_3_l209_209332

-- Problem conditions and requirements as Lean definitions and statements
def y_A (x : ℝ) : ℝ := (2 / 5) * x
def y_B (x : ℝ) : ℝ := -(1 / 5) * x^2 + 2 * x

-- Part (1)
theorem part_1 : y_A 10 = 4 := sorry

-- Part (2)
theorem part_2 (m : ℝ) (h : m > 0) : y_A m = y_B m → m = 3 := sorry

-- Part (3)
def W (x_A x_B : ℝ) : ℝ := y_A x_A + y_B x_B

theorem part_3 (x_A x_B : ℝ) (h : x_A + x_B = 32) : 
  (∀ x_A' x_B', x_A' + x_B' = 32 → W x_A x_B ≥ W x_A' x_B') ∧ W x_A x_B = 16 :=
begin
  sorry
end

end part_1_part_2_part_3_l209_209332


namespace triangle_hypotenuse_and_area_l209_209657

theorem triangle_hypotenuse_and_area 
  (A B C D : Type) 
  (CD : ℝ) 
  (angle_A : ℝ) 
  (hypotenuse_AC : ℝ) 
  (area_ABC : ℝ) 
  (h1 : CD = 1) 
  (h2 : angle_A = 45) : 
  hypotenuse_AC = Real.sqrt 2 
  ∧ 
  area_ABC = 1 / 2 := 
by
  sorry

end triangle_hypotenuse_and_area_l209_209657


namespace farmer_initial_productivity_l209_209463

theorem farmer_initial_productivity (x : ℝ) (d : ℝ)
  (hx1 : d = 1440 / x)
  (hx2 : 2 * x + (d - 4) * 1.25 * x = 1440) :
  x = 120 :=
by
  sorry

end farmer_initial_productivity_l209_209463


namespace pastrami_sandwich_cost_l209_209264

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l209_209264


namespace sum_of_real_values_l209_209228

theorem sum_of_real_values (x : ℝ) (h : |3 * x - 15| + |x - 5| = 92) : (x = 28 ∨ x = -18) → x + 10 = 0 := by
  sorry

end sum_of_real_values_l209_209228


namespace find_total_cows_l209_209851

-- Define the conditions given in the problem
def ducks_legs (D : ℕ) : ℕ := 2 * D
def cows_legs (C : ℕ) : ℕ := 4 * C
def total_legs (D C : ℕ) : ℕ := ducks_legs D + cows_legs C
def total_heads (D C : ℕ) : ℕ := D + C

-- State the problem in Lean 4
theorem find_total_cows (D C : ℕ) (h : total_legs D C = 2 * total_heads D C + 32) : C = 16 :=
sorry

end find_total_cows_l209_209851


namespace emails_received_afternoon_is_one_l209_209239

-- Define the number of emails received by Jack in the morning
def emails_received_morning : ℕ := 4

-- Define the total number of emails received by Jack in a day
def total_emails_received : ℕ := 5

-- Define the number of emails received by Jack in the afternoon
def emails_received_afternoon : ℕ := total_emails_received - emails_received_morning

-- Prove the number of emails received by Jack in the afternoon
theorem emails_received_afternoon_is_one : emails_received_afternoon = 1 :=
by 
  -- Proof is neglected as per instructions.
  sorry

end emails_received_afternoon_is_one_l209_209239


namespace framed_painting_ratio_l209_209176

/-- A rectangular painting measuring 20" by 30" is to be framed, with the longer dimension vertical.
The width of the frame at the top and bottom is three times the width of the frame on the sides.
Given that the total area of the frame equals the area of the painting, the ratio of the smaller to the 
larger dimension of the framed painting is 4:7. -/
theorem framed_painting_ratio : 
  ∀ (w h : ℝ) (side_frame_width : ℝ), 
    w = 20 ∧ h = 30 ∧ 3 * side_frame_width * (2 * (w + 2 * side_frame_width) + 2 * (h + 6 * side_frame_width) - w * h) = w * h 
    → side_frame_width = 2 
    → (w + 2 * side_frame_width) / (h + 6 * side_frame_width) = 4 / 7 :=
sorry

end framed_painting_ratio_l209_209176


namespace odd_if_and_only_if_m_even_l209_209863

variables (o n m : ℕ)

theorem odd_if_and_only_if_m_even
  (h_o_odd : o % 2 = 1) :
  ((o^3 + n*o + m) % 2 = 1) ↔ (m % 2 = 0) :=
sorry

end odd_if_and_only_if_m_even_l209_209863


namespace inequality_holds_for_all_x_l209_209196

theorem inequality_holds_for_all_x (m : ℝ) :
  (∀ x : ℝ, m * x^2 - (m + 3) * x - 1 < 0) ↔ -9 < m ∧ m < -1 :=
by
  sorry

end inequality_holds_for_all_x_l209_209196


namespace number_of_participants_eq_14_l209_209112

theorem number_of_participants_eq_14 (n : ℕ) (h : n * (n - 1) / 2 = 91) : n = 14 :=
by
  sorry

end number_of_participants_eq_14_l209_209112


namespace westbound_cyclist_speed_increase_l209_209569

def eastbound_speed : ℕ := 18
def travel_time : ℕ := 6
def total_distance : ℕ := 246

theorem westbound_cyclist_speed_increase (x : ℕ) :
  eastbound_speed * travel_time + (eastbound_speed + x) * travel_time = total_distance →
  x = 5 :=
by
  sorry

end westbound_cyclist_speed_increase_l209_209569


namespace ammonia_moles_l209_209826

-- Definitions corresponding to the given conditions
def moles_KOH : ℚ := 3
def moles_NH4I : ℚ := 3

def balanced_equation (n_KOH n_NH4I : ℚ) : ℚ :=
  if n_KOH = n_NH4I then n_KOH else 0

-- Proof problem: Prove that the reaction produces 3 moles of NH3
theorem ammonia_moles (n_KOH n_NH4I : ℚ) (h1 : n_KOH = moles_KOH) (h2 : n_NH4I = moles_NH4I) :
  balanced_equation n_KOH n_NH4I = 3 :=
by 
  -- proof here 
  sorry

end ammonia_moles_l209_209826


namespace base_distance_l209_209960

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l209_209960


namespace base_distance_l209_209961

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l209_209961


namespace fraction_expression_value_l209_209479

theorem fraction_expression_value:
  (1/4 - 1/5) / (1/3 - 1/6) = 3/10 :=
by
  sorry

end fraction_expression_value_l209_209479


namespace remainder_of_50_pow_2019_plus_1_mod_7_l209_209696

theorem remainder_of_50_pow_2019_plus_1_mod_7 :
  (50 ^ 2019 + 1) % 7 = 2 :=
by
  sorry

end remainder_of_50_pow_2019_plus_1_mod_7_l209_209696


namespace contractor_realized_after_20_days_l209_209167

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end contractor_realized_after_20_days_l209_209167


namespace fraction_of_students_with_partner_l209_209231

theorem fraction_of_students_with_partner (s t : ℕ) 
  (h : t = (4 * s) / 3) :
  (t / 4 + s / 3) / (t + s) = 2 / 7 :=
by
  -- Proof omitted
  sorry

end fraction_of_students_with_partner_l209_209231


namespace a_completes_in_12_days_l209_209458

def work_rate_a_b (r_A r_B : ℝ) := r_A + r_B = 1 / 3
def work_rate_b_c (r_B r_C : ℝ) := r_B + r_C = 1 / 2
def work_rate_a_c (r_A r_C : ℝ) := r_A + r_C = 1 / 3

theorem a_completes_in_12_days (r_A r_B r_C : ℝ) 
  (h1 : work_rate_a_b r_A r_B)
  (h2 : work_rate_b_c r_B r_C)
  (h3 : work_rate_a_c r_A r_C) : 
  1 / r_A = 12 :=
by
  sorry

end a_completes_in_12_days_l209_209458


namespace average_of_a_and_b_l209_209881

theorem average_of_a_and_b (a b c : ℝ) 
  (h₁ : (b + c) / 2 = 90)
  (h₂ : c - a = 90) :
  (a + b) / 2 = 45 :=
sorry

end average_of_a_and_b_l209_209881


namespace g_g_2_eq_394_l209_209217

def g (x : ℝ) : ℝ := 4 * x ^ 2 - 6

theorem g_g_2_eq_394 : g (g 2) = 394 :=
by
  sorry

end g_g_2_eq_394_l209_209217


namespace ladder_base_distance_l209_209935

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l209_209935


namespace math_problem_proof_l209_209438

def eight_to_zero : ℝ := 1
def log_base_10_of_100 : ℝ := 2

theorem math_problem_proof : eight_to_zero - log_base_10_of_100 = -1 :=
by sorry

end math_problem_proof_l209_209438


namespace percentage_of_women_in_study_group_l209_209460

variable (W : ℝ) -- W is the percentage of women in the study group in decimal form

-- Given conditions as hypotheses
axiom h1 : 0 < W ∧ W <= 1         -- W represents a percentage, so it must be between 0 and 1.
axiom h2 : 0.40 * W = 0.28         -- 40 percent of women are lawyers, and the probability of selecting a woman lawyer is 0.28.

-- The statement to prove
theorem percentage_of_women_in_study_group : W = 0.7 :=
by
  sorry

end percentage_of_women_in_study_group_l209_209460


namespace evaluations_total_l209_209138

theorem evaluations_total :
    let class_A_students := 30
    let class_A_mc := 12
    let class_A_essay := 3
    let class_A_presentation := 1

    let class_B_students := 25
    let class_B_mc := 15
    let class_B_short_answer := 5
    let class_B_essay := 2

    let class_C_students := 35
    let class_C_mc := 10
    let class_C_essay := 3
    let class_C_presentation_groups := class_C_students / 5 -- groups of 5

    let class_D_students := 40
    let class_D_mc := 11
    let class_D_short_answer := 4
    let class_D_essay := 3

    let class_E_students := 20
    let class_E_mc := 14
    let class_E_short_answer := 5
    let class_E_essay := 2

    let total_mc := (class_A_students * class_A_mc) +
                    (class_B_students * class_B_mc) +
                    (class_C_students * class_C_mc) +
                    (class_D_students * class_D_mc) +
                    (class_E_students * class_E_mc)

    let total_short_answer := (class_B_students * class_B_short_answer) +
                              (class_D_students * class_D_short_answer) +
                              (class_E_students * class_E_short_answer)

    let total_essay := (class_A_students * class_A_essay) +
                       (class_B_students * class_B_essay) +
                       (class_C_students * class_C_essay) +
                       (class_D_students * class_D_essay) +
                       (class_E_students * class_E_essay)

    let total_presentation := (class_A_students * class_A_presentation) +
                              class_C_presentation_groups

    total_mc + total_short_answer + total_essay + total_presentation = 2632 := by
    sorry

end evaluations_total_l209_209138


namespace mari_buttons_l209_209398

/-- 
Given that:
1. Sue made 6 buttons
2. Sue made half as many buttons as Kendra.
3. Mari made 4 more than five times as many buttons as Kendra.

We are to prove that Mari made 64 buttons.
-/
theorem mari_buttons (sue_buttons : ℕ) (kendra_buttons : ℕ) (mari_buttons : ℕ) 
  (h1 : sue_buttons = 6)
  (h2 : sue_buttons = kendra_buttons / 2)
  (h3 : mari_buttons = 5 * kendra_buttons + 4) :
  mari_buttons = 64 :=
  sorry

end mari_buttons_l209_209398


namespace ladder_base_distance_l209_209953

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l209_209953


namespace odd_function_inequality_l209_209130

-- Define f as an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

theorem odd_function_inequality
  (f : ℝ → ℝ) (h1 : is_odd_function f)
  (a b : ℝ) (h2 : f a > f b) :
  f (-a) < f (-b) :=
by
  sorry

end odd_function_inequality_l209_209130


namespace cloth_sale_total_amount_l209_209046

theorem cloth_sale_total_amount :
  let CP := 70 -- Cost Price per metre in Rs.
  let Loss := 10 -- Loss per metre in Rs.
  let SP := CP - Loss -- Selling Price per metre in Rs.
  let total_metres := 600 -- Total metres sold
  let total_amount := SP * total_metres -- Total amount from the sale
  total_amount = 36000 := by
  sorry

end cloth_sale_total_amount_l209_209046


namespace hostel_provisions_l209_209173

theorem hostel_provisions (x : ℕ) :
  (250 * x = 200 * 60) → x = 48 :=
by
  sorry

end hostel_provisions_l209_209173


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209792

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209792


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209812

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209812


namespace ladder_distance_l209_209944

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l209_209944


namespace monster_ratio_l209_209316

theorem monster_ratio (r : ℝ) :
  (121 + 121 * r + 121 * r^2 = 847) → r = 2 :=
by
  intros h
  sorry

end monster_ratio_l209_209316


namespace fourth_root_sum_of_square_roots_eq_l209_209676

theorem fourth_root_sum_of_square_roots_eq :
  (1 + Real.sqrt 2 + Real.sqrt 3) = 
    Real.sqrt (Real.sqrt 6400 + Real.sqrt 6144 + Real.sqrt 4800 + Real.sqrt 4608) ^ 4 :=
by
  sorry

end fourth_root_sum_of_square_roots_eq_l209_209676


namespace parabola_directrix_l209_209884

theorem parabola_directrix (x y : ℝ) (h : y = 8 * x^2) : y = -1 / 32 :=
sorry

end parabola_directrix_l209_209884


namespace barbie_bruno_trips_l209_209618

theorem barbie_bruno_trips (coconuts : ℕ) (barbie_capacity : ℕ) (bruno_capacity : ℕ) : 
  coconuts = 144 → barbie_capacity = 4 → bruno_capacity = 8 → (coconuts / (barbie_capacity + bruno_capacity) = 12) :=
by 
  intros h_coconuts h_barbie h_bruno
  rw [h_coconuts, h_barbie, h_bruno]
  norm_num
  sorry

end barbie_bruno_trips_l209_209618


namespace find_PQ_length_l209_209697

-- Define the lengths of the sides of the triangles and the angle
def PQ_length : ℝ := 9
def QR_length : ℝ := 20
def PR_length : ℝ := 15
def ST_length : ℝ := 4.5
def TU_length : ℝ := 7.5
def SU_length : ℝ := 15
def angle_PQR : ℝ := 135
def angle_STU : ℝ := 135

-- Define the similarity condition
def triangles_similar (PQ QR PR ST TU SU angle_PQR angle_STU : ℝ) : Prop :=
  angle_PQR = angle_STU ∧ PQ / QR = ST / TU

-- Theorem statement
theorem find_PQ_length (PQ QR PR ST TU SU angle_PQR angle_STU: ℝ) 
  (H : triangles_similar PQ QR PR ST TU SU angle_PQR angle_STU) : PQ = 20 :=
by
  sorry

end find_PQ_length_l209_209697


namespace bill_difference_l209_209537

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l209_209537


namespace polygon_possible_l209_209013

-- Definition: a polygon with matches without breaking them
structure MatchPolygon (n : ℕ) (length : ℝ) where
  num_matches : ℕ
  len_matches : ℝ
  area : ℝ
  notequalzero : len_matches ≠ 0
  notequalzero2 : area ≠ 0
  perimeter_eq: num_matches * len_matches = length * real.of_nat n
  all_matches_used : n = 12
  no_breaking : (length / real.of_nat n) = 2 

theorem polygon_possible : 
  ∃ P : MatchPolygon 12 2, P.area = 16 :=
sorry

end polygon_possible_l209_209013


namespace percentage_of_boy_scouts_with_signed_permission_slips_l209_209040

noncomputable def total_scouts : ℕ := 100 -- assume 100 scouts
noncomputable def total_signed_permission_slips : ℕ := 70 -- 70% of 100
noncomputable def boy_scouts : ℕ := 60 -- 60% of 100
noncomputable def girl_scouts : ℕ := 40 -- total_scouts - boy_scouts 

noncomputable def girl_scouts_signed_permission_slips : ℕ := girl_scouts * 625 / 1000 

theorem percentage_of_boy_scouts_with_signed_permission_slips :
  (boy_scouts * 75 / 100) = (total_signed_permission_slips - girl_scouts_signed_permission_slips) :=
by
  sorry

end percentage_of_boy_scouts_with_signed_permission_slips_l209_209040


namespace boat_speed_in_still_water_l209_209561

theorem boat_speed_in_still_water 
  (rate_of_current : ℝ) 
  (time_in_hours : ℝ) 
  (distance_downstream : ℝ)
  (h_rate : rate_of_current = 5) 
  (h_time : time_in_hours = 15 / 60) 
  (h_distance : distance_downstream = 6.25) : 
  ∃ x : ℝ, (distance_downstream = (x + rate_of_current) * time_in_hours) ∧ x = 20 :=
by 
  -- Main theorem statement, proof omitted for brevity.
  sorry

end boat_speed_in_still_water_l209_209561


namespace probability_two_dice_same_number_l209_209407

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l209_209407


namespace bryden_receives_10_dollars_l209_209758

theorem bryden_receives_10_dollars 
  (collector_rate : ℝ := 5)
  (num_quarters : ℝ := 4)
  (face_value_per_quarter : ℝ := 0.50) :
  collector_rate * num_quarters * face_value_per_quarter = 10 :=
by
  sorry

end bryden_receives_10_dollars_l209_209758


namespace man_son_age_ratio_l209_209764

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l209_209764


namespace circle_radius_l209_209226
open Real

theorem circle_radius (d : ℝ) (h_diam : d = 24) : d / 2 = 12 :=
by
  -- The proof will be here
  sorry

end circle_radius_l209_209226


namespace ladder_base_distance_l209_209919

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l209_209919


namespace sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l209_209563

-- Definitions for vertices of pyramids
variables (A B C D E : ℝ)

-- Assuming E is inside pyramid ABCD
variable (inside : E ∈ convex_hull ℝ {A, B, C, D})

-- Assertion 1
theorem sum_of_edges_not_always_smaller
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A) (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D) (h7 : D ≠ E):
  ¬ (abs A - E + abs B - E + abs C - E < abs A - D + abs B - D + abs C - D) :=
sorry

-- Assertion 2
theorem at_least_one_edge_shorter
  (h1 : A ≠ B) (h2 : B ≠ C) (h3 : C ≠ A)
  (h4 : A ≠ D) (h5 : B ≠ D) (h6 : C ≠ D)
  (h7 : D ≠ E):
  abs A - E < abs A - D ∨ abs B - E < abs B - D ∨ abs C - E < abs C - D :=
sorry

end sum_of_edges_not_always_smaller_at_least_one_edge_shorter_l209_209563


namespace composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l209_209751

theorem composite_10201_base_gt_2 (x : ℕ) (hx : x > 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + 2*x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base (x : ℕ) (hx : x ≥ 2) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

theorem composite_10101_any_base_any_x (x : ℕ) (hx : x ≥ 1) : ∃ a b, a > 1 ∧ b > 1 ∧ x^4 + x^2 + 1 = a * b := by
  sorry

end composite_10201_base_gt_2_composite_10101_any_base_composite_10101_any_base_any_x_l209_209751


namespace pastrami_sandwich_cost_l209_209263

variable (X : ℕ)

theorem pastrami_sandwich_cost
  (h1 : 10 * X + 5 * (X + 2) = 55) :
  X + 2 = 5 := 
by
  sorry

end pastrami_sandwich_cost_l209_209263


namespace building_height_l209_209240

theorem building_height
  (num_stories_1 : ℕ)
  (height_story_1 : ℕ)
  (num_stories_2 : ℕ)
  (height_story_2 : ℕ)
  (h1 : num_stories_1 = 10)
  (h2 : height_story_1 = 12)
  (h3 : num_stories_2 = 10)
  (h4 : height_story_2 = 15)
  :
  num_stories_1 * height_story_1 + num_stories_2 * height_story_2 = 270 :=
by
  sorry

end building_height_l209_209240


namespace simplify_and_evaluate_l209_209876

theorem simplify_and_evaluate (a : ℝ) (h : a = Real.sqrt 2 + 1) :
  (1 + 1 / a) / ((a^2 - 1) / a) = (Real.sqrt 2 / 2) :=
by
  sorry

end simplify_and_evaluate_l209_209876


namespace sum_primes_no_solution_congruence_l209_209344

theorem sum_primes_no_solution_congruence :
  ∑ p in {p | Nat.Prime p ∧ ¬ (∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [ZMOD p])}, p = 7 :=
sorry

end sum_primes_no_solution_congruence_l209_209344


namespace intelligent_test_failure_prob_maximize_failure_prob_production_improvement_needed_l209_209842

-- Definitions to formalize the conditions
def prob_safe : ℚ := 49 / 50
def prob_energy : ℚ := 48 / 49
def prob_perf : ℚ := 47 / 48
def prob_fail_manual (p : ℚ) : Prop := 0 < p ∧ p < 1

-- Definitions for the main probabilities
def prob_pass_intelligent : ℚ := prob_safe * prob_energy * prob_perf
def prob_fail_intelligent : ℚ := 1 - prob_pass_intelligent

-- Formalize Question 1
theorem intelligent_test_failure_prob :
  prob_fail_intelligent = 3 / 50 :=
begin
  -- proof omitted
  sorry
end

-- Formalize Question 2
def f (p : ℚ) : ℚ := 50 * p * (1 - p) ^ 49

theorem maximize_failure_prob (p : ℚ) (h : prob_fail_manual p) :
  argmax f p = 1 / 50 :=
begin
  -- proof omitted
  sorry
end

-- Formalize Question 3
def overall_pass_rate (p : ℚ) : ℚ :=
  prob_pass_intelligent * (1 - p)

theorem production_improvement_needed (p : ℚ) (h : prob_fail_manual p) :
  p = 1 / 50 → overall_pass_rate p < 93 / 100 :=
begin
  -- proof omitted
  sorry
end

end intelligent_test_failure_prob_maximize_failure_prob_production_improvement_needed_l209_209842


namespace chef_michel_total_pies_l209_209069

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l209_209069


namespace tenth_graders_science_only_l209_209057

theorem tenth_graders_science_only (total_students science_students art_students : ℕ) 
  (h1 : total_students = 140) 
  (h2 : science_students = 100) 
  (h3 : art_students = 75) : 
  (science_students - (science_students + art_students - total_students)) = 65 :=
by
  sorry

end tenth_graders_science_only_l209_209057


namespace ladder_base_distance_l209_209936

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l209_209936


namespace total_number_of_cows_l209_209110

variable (D C : ℕ) -- D is the number of ducks and C is the number of cows

-- Define the condition given in the problem
def legs_eq : Prop := 2 * D + 4 * C = 2 * (D + C) + 28

theorem total_number_of_cows (h : legs_eq D C) : C = 14 := by
  sorry

end total_number_of_cows_l209_209110


namespace initial_nickels_eq_l209_209257

variable (quarters : ℕ) (initial_nickels : ℕ) (nickels_borrowed : ℕ) (nickels_left : ℕ)

-- Assumptions based on the problem
axiom quarters_had : quarters = 33
axiom nickels_left_axiom : nickels_left = 12
axiom nickels_borrowed_axiom : nickels_borrowed = 75

-- Theorem to prove: initial number of nickels
theorem initial_nickels_eq :
  initial_nickels = nickels_left + nickels_borrowed :=
by
  sorry

end initial_nickels_eq_l209_209257


namespace find_y_for_two_thirds_l209_209481

theorem find_y_for_two_thirds (x y : ℝ) (h₁ : (2 / 3) * x + y = 10) (h₂ : x = 6) : y = 6 :=
by
  sorry

end find_y_for_two_thirds_l209_209481


namespace ladder_distance_l209_209929

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l209_209929


namespace train_speed_without_stoppages_l209_209326

theorem train_speed_without_stoppages 
  (distance_with_stoppages : ℝ)
  (avg_speed_with_stoppages : ℝ)
  (stoppage_time_per_hour : ℝ)
  (distance_without_stoppages : ℝ)
  (avg_speed_without_stoppages : ℝ) :
  avg_speed_with_stoppages = 200 → 
  stoppage_time_per_hour = 20 / 60 →
  distance_without_stoppages = distance_with_stoppages * avg_speed_without_stoppages →
  distance_with_stoppages = avg_speed_with_stoppages →
  avg_speed_without_stoppages == 300 := 
by
  intros
  sorry

end train_speed_without_stoppages_l209_209326


namespace solve_system_l209_209551

theorem solve_system (a b c : ℝ) (h₁ : a^2 + 3 * a + 1 = (b + c) / 2)
                                (h₂ : b^2 + 3 * b + 1 = (a + c) / 2)
                                (h₃ : c^2 + 3 * c + 1 = (a + b) / 2) : 
  a = -1 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end solve_system_l209_209551


namespace therapy_hours_l209_209461

variable (F A n : ℕ)
variable (h1 : F = A + 20)
variable (h2 : F + 2 * A = 188)
variable (h3 : F + A * (n - 1) = 300)

theorem therapy_hours : n = 5 := by
  sorry

end therapy_hours_l209_209461


namespace find_base_k_l209_209638

theorem find_base_k (k : ℕ) (h1 : 1 + 3 * k + 2 * k^2 = 30) : k = 4 :=
by sorry

end find_base_k_l209_209638


namespace sec_150_l209_209817

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l209_209817


namespace grantRooms_is_2_l209_209483

/-- Danielle's apartment has 6 rooms. -/
def danielleRooms : ℕ := 6

/-- Heidi's apartment has 3 times as many rooms as Danielle's apartment. -/
def heidiRooms : ℕ := 3 * danielleRooms

/-- Grant's apartment has 1/9 as many rooms as Heidi's apartment. -/
def grantRooms : ℕ := heidiRooms / 9

/-- Prove that Grant's apartment has 2 rooms. -/
theorem grantRooms_is_2 : grantRooms = 2 := by
  sorry

end grantRooms_is_2_l209_209483


namespace find_ratio_of_radii_l209_209480

noncomputable def ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) : Prop :=
  a / b = Real.sqrt 5 / 5

theorem find_ratio_of_radii (a b : ℝ) (h1 : π * b ^ 2 - π * a ^ 2 = 4 * π * a ^ 2) :
  ratio_of_radii a b h1 :=
sorry

end find_ratio_of_radii_l209_209480


namespace increasing_function_range_l209_209095

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (2 * a - 1) * x - 1 else x + 1

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x ≤ f a y) ↔ (1 / 2 < a ∧ a ≤ 2) :=
sorry

end increasing_function_range_l209_209095


namespace sec_150_eq_neg_2_sqrt3_over_3_l209_209799

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l209_209799


namespace expected_value_decisive_games_l209_209470

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l209_209470


namespace jonathan_tax_per_hour_l209_209393

-- Given conditions
def wage : ℝ := 25          -- wage in dollars per hour
def tax_rate : ℝ := 0.024    -- tax rate in decimal

-- Prove statement
theorem jonathan_tax_per_hour :
  (wage * 100) * tax_rate = 60 :=
sorry

end jonathan_tax_per_hour_l209_209393


namespace solve_equation_l209_209352

theorem solve_equation (x : ℝ) :
  (15 * x - x^2) / (x + 2) * (x + (15 - x) / (x + 2)) = 60 → x = 12 :=
by
  sorry

end solve_equation_l209_209352


namespace cars_given_by_mum_and_dad_l209_209871

-- Define the conditions given in the problem
def initial_cars : ℕ := 150
def final_cars : ℕ := 196
def cars_by_auntie : ℕ := 6
def cars_more_than_uncle : ℕ := 1
def cars_given_by_family (uncle : ℕ) (grandpa : ℕ) (auntie : ℕ) : ℕ :=
  uncle + grandpa + auntie

-- Prove the required statement
theorem cars_given_by_mum_and_dad :
  ∃ (uncle grandpa : ℕ), grandpa = 2 * uncle ∧ auntie = uncle + cars_more_than_uncle ∧ 
    auntie = cars_by_auntie ∧
    final_cars - initial_cars - cars_given_by_family uncle grandpa auntie = 25 :=
by
  -- Placeholder for the actual proof
  sorry

end cars_given_by_mum_and_dad_l209_209871


namespace total_pies_sold_l209_209067

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l209_209067


namespace cone_base_circumference_l209_209322

theorem cone_base_circumference (V : ℝ) (h : ℝ) (C : ℝ) (r : ℝ) :
  V = 18 * Real.pi →
  h = 6 →
  (V = (1 / 3) * Real.pi * r^2 * h) →
  C = 2 * Real.pi * r →
  C = 6 * Real.pi :=
by
  intros h1 h2 h3 h4
  sorry

end cone_base_circumference_l209_209322


namespace reciprocal_sum_fractions_l209_209023

theorem reciprocal_sum_fractions:
  let a := (3: ℚ) / 4
  let b := (5: ℚ) / 6
  let c := (1: ℚ) / 2
  (a + b + c)⁻¹ = 12 / 25 :=
by
  sorry

end reciprocal_sum_fractions_l209_209023


namespace total_cows_in_ranch_l209_209298

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l209_209298


namespace ladder_distance_from_wall_l209_209940

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l209_209940


namespace two_x_equals_y_l209_209505

theorem two_x_equals_y (x y : ℝ) (h1 : (x + y) / 3 = 1) (h2 : x + 2 * y = 5) : 2 * x = y := 
by
  sorry

end two_x_equals_y_l209_209505


namespace trig_identity_nec_but_not_suff_l209_209188

open Real

theorem trig_identity_nec_but_not_suff (α β : ℝ) (k : ℤ) :
  (α + β = 2 * k * π + π / 6) → (sin α * cos β + cos α * sin β = 1 / 2) := by
  sorry

end trig_identity_nec_but_not_suff_l209_209188


namespace smaller_number_is_four_l209_209891

theorem smaller_number_is_four (x y : ℝ) (h1 : x + y = 18) (h2 : x - y = 10) : y = 4 :=
by
  sorry

end smaller_number_is_four_l209_209891


namespace basketball_team_total_players_l209_209754

theorem basketball_team_total_players (total_points : ℕ) (min_points : ℕ) (max_points : ℕ) (team_size : ℕ)
  (h1 : total_points = 100)
  (h2 : min_points = 7)
  (h3 : max_points = 23)
  (h4 : ∀ (n : ℕ), n ≥ min_points)
  (h5 : max_points = 23)
  : team_size = 12 :=
sorry

end basketball_team_total_players_l209_209754


namespace probability_same_color_l209_209016

theorem probability_same_color :
  let bagA_white := 8
  let bagA_red := 4
  let bagB_white := 6
  let bagB_red := 6
  let totalA := bagA_white + bagA_red
  let totalB := bagB_white + bagB_red
  let prob_white_white := (bagA_white / totalA) * (bagB_white / totalB)
  let prob_red_red := (bagA_red / totalA) * (bagB_red / totalB)
  let total_prob := prob_white_white + prob_red_red
  total_prob = 1 / 2 := 
by 
  sorry

end probability_same_color_l209_209016


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209706

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209706


namespace constant_k_for_linear_function_l209_209689

theorem constant_k_for_linear_function (k : ℝ) (h : ∀ (x : ℝ), y = x^(k-1) + 2 → y = a * x + b) : k = 2 :=
sorry

end constant_k_for_linear_function_l209_209689


namespace sin_double_angle_l209_209115

noncomputable def r := Real.sqrt 5
noncomputable def sin_α := -2 / r
noncomputable def cos_α := 1 / r
noncomputable def sin_2α := 2 * sin_α * cos_α

theorem sin_double_angle (α : ℝ) :
  (∃ P : ℝ × ℝ, P = (1, -2) ∧ ∃ α : ℝ, true) → sin_2α = -4 / 5 :=
by
  sorry

end sin_double_angle_l209_209115


namespace intersection_of_P_and_Q_l209_209213

def P : Set (ℝ × ℝ) := {p | p.fst + p.snd = 0}
def Q : Set (ℝ × ℝ) := {p | p.fst - p.snd = 2}

theorem intersection_of_P_and_Q :
  P ∩ Q = {(1, -1)} :=
by
  sorry

end intersection_of_P_and_Q_l209_209213


namespace area_correct_l209_209690

-- Define the conditions provided in the problem
def width (w : ℝ) := True
def length (l : ℝ) := True
def perimeter (p : ℝ) := True

-- Add the conditions about the playground
axiom length_exceeds_width_by : ∃ l w, l = 3 * w + 30
axiom perimeter_is_given : ∃ l w, 2 * (l + w) = 730

-- Define the area of the playground and state the theorem
noncomputable def area_of_playground : ℝ := 83.75 * 281.25

theorem area_correct :
  (∃ l w, l = 3 * w + 30 ∧ 2 * (l + w) = 730) →
  area_of_playground = 23554.6875 :=
by
  sorry

end area_correct_l209_209690


namespace simplify_and_evaluate_expr_l209_209875

noncomputable def a : ℝ := 3 + Real.sqrt 5
noncomputable def b : ℝ := 3 - Real.sqrt 5

theorem simplify_and_evaluate_expr : 
  (a^2 - 2 * a * b + b^2) / (a^2 - b^2) * (a * b) / (a - b) = 2 / 3 := by
  sorry

end simplify_and_evaluate_expr_l209_209875


namespace gcd_3pow600_minus_1_3pow612_minus_1_l209_209156

theorem gcd_3pow600_minus_1_3pow612_minus_1 :
  Nat.gcd (3^600 - 1) (3^612 - 1) = 531440 :=
by
  sorry

end gcd_3pow600_minus_1_3pow612_minus_1_l209_209156


namespace largest_angle_measure_l209_209279

theorem largest_angle_measure (v : ℝ) (h : v > 3/2) :
  ∃ θ, θ = Real.arccos ((4 * v - 4) / (2 * Real.sqrt ((2 * v - 3) * (4 * v - 4)))) ∧
       θ = π - θ ∧
       θ = Real.arccos ((2 * v - 3) / (2 * Real.sqrt ((2 * v + 3) * (4 * v - 4)))) := 
sorry

end largest_angle_measure_l209_209279


namespace divisor_of_12401_76_13_l209_209519

theorem divisor_of_12401_76_13 (D : ℕ) (h1: 12401 = (D * 76) + 13) : D = 163 :=
sorry

end divisor_of_12401_76_13_l209_209519


namespace right_triangle_side_length_l209_209236

theorem right_triangle_side_length 
  (a b c : ℝ) 
  (h1 : a = 5) 
  (h2 : c = 12) 
  (h_right : a^2 + b^2 = c^2) : 
  b = Real.sqrt 119 :=
by
  sorry

end right_triangle_side_length_l209_209236


namespace sec_150_eq_neg_two_div_sqrt_three_l209_209807

noncomputable def sec (θ : Real) : Real := 1 / Real.cos θ

theorem sec_150_eq_neg_two_div_sqrt_three :
  sec 150 = -2 / Real.sqrt 3 := by
  sorry

end sec_150_eq_neg_two_div_sqrt_three_l209_209807


namespace inequality_solution_l209_209281

theorem inequality_solution (x : ℝ) :
  (2 * x - 1 > 0 ∧ x + 1 ≤ 3) ↔ (1 / 2 < x ∧ x ≤ 2) :=
by
  sorry

end inequality_solution_l209_209281


namespace fireworks_display_l209_209989

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l209_209989


namespace first_company_managers_percentage_l209_209462

-- Definitions from the conditions
variable (F M : ℝ) -- total workforce of first company and merged company
variable (x : ℝ) -- percentage of managers in the first company
variable (cond1 : 0.25 * M = F) -- 25% of merged company's workforce originated from the first company
variable (cond2 : 0.25 * M / M = 0.25) -- resulting merged company's workforce consists of 25% managers

-- The statement to prove
theorem first_company_managers_percentage : x = 25 :=
by
  sorry

end first_company_managers_percentage_l209_209462


namespace initial_shares_bought_l209_209903

variable (x : ℕ) -- x is the number of shares Tom initially bought

-- Conditions:
def initial_cost_per_share : ℕ := 3
def num_shares_sold : ℕ := 10
def selling_price_per_share : ℕ := 4
def doubled_value_per_remaining_share : ℕ := 2 * initial_cost_per_share
def total_profit : ℤ := 40

-- Proving the number of shares initially bought
theorem initial_shares_bought (h : num_shares_sold * selling_price_per_share - x * initial_cost_per_share = total_profit) :
  x = 10 := by sorry

end initial_shares_bought_l209_209903


namespace sec_150_l209_209818

-- Define the conditions
def sec (θ : ℝ) : ℝ := 1 / (Real.cos θ)
def cos_150 := Real.cos (Real.pi - Real.pi / 6)
def cos_30 := Real.sqrt 3 / 2

-- The main statement to prove
theorem sec_150 : sec (5 * Real.pi / 6) = -2 * Real.sqrt 3 / 3 :=
by
  have h1 : Real.cos (5 * Real.pi / 6) = -cos_30 :=
    by rw [cos_150, cos_30]; sorry
  have h2 : sec (5 * Real.pi / 6) = 1 / (-cos_30) :=
    by rw [sec, h1]; sorry
  have h3 : 1 / (- (Real.sqrt 3 / 2)) = -2 / Real.sqrt 3 :=
    by sorry
  have h4 : -2 / Real.sqrt 3 = -2 * Real.sqrt 3 / 3 :=
    by nth_rewrite 1 [div_mul_eq_mul_div]; nth_rewrite 1 [mul_div_cancel (Real.sqrt 3) (ne_of_gt (Real.sqrt_pos_of_pos three_pos))]; sorry
  rw [h2, h3, h4]; sorry

end sec_150_l209_209818


namespace ladder_base_distance_l209_209933

theorem ladder_base_distance
  (ladder_length : ℝ)
  (wall_height : ℝ)
  (base_distance : ℝ)
  (h1 : ladder_length = 13)
  (h2 : wall_height = 12)
  (h3 : ladder_length^2 = wall_height^2 + base_distance^2) :
  base_distance = 5 :=
sorry

end ladder_base_distance_l209_209933


namespace watermelon_cost_l209_209976

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l209_209976


namespace ratio_of_areas_of_similar_triangles_l209_209546

theorem ratio_of_areas_of_similar_triangles (a b a1 b1 S S1 : ℝ) (α k : ℝ) :
  S = (1/2) * a * b * (Real.sin α) →
  S1 = (1/2) * a1 * b1 * (Real.sin α) →
  a1 = k * a →
  b1 = k * b →
  S1 / S = k^2 := by
  intros h1 h2 h3 h4
  sorry

end ratio_of_areas_of_similar_triangles_l209_209546


namespace joey_return_speed_l209_209390

theorem joey_return_speed
    (h1: 1 = (2 : ℝ) / u)
    (h2: (4 : ℝ) / (1 + t) = 3)
    (h3: u = 2)
    (h4: t = 1 / 3) :
    (2 : ℝ) / t = 6 :=
by
  sorry

end joey_return_speed_l209_209390


namespace bob_hair_length_l209_209060

-- Define the current length of Bob's hair
def current_length : ℝ := 36

-- Define the growth rate in inches per month
def growth_rate : ℝ := 0.5

-- Define the duration in years
def duration_years : ℕ := 5

-- Define the total growth over the duration in years
def total_growth : ℝ := growth_rate * 12 * duration_years

-- Define the length of Bob's hair when he last cut it
def initial_length : ℝ := current_length - total_growth

-- Theorem stating that the length of Bob's hair when he last cut it was 6 inches
theorem bob_hair_length :
  initial_length = 6 :=
by
  -- Proof omitted
  sorry

end bob_hair_length_l209_209060


namespace find_a1_over_1_minus_q_l209_209206

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_a1_over_1_minus_q 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 5 + a 6 + a 7 + a 8 = 48) :
  (a 1) / (1 - q) = -1 / 5 :=
sorry

end find_a1_over_1_minus_q_l209_209206


namespace gcd_140_396_is_4_l209_209157

def gcd_140_396 : ℕ := Nat.gcd 140 396

theorem gcd_140_396_is_4 : gcd_140_396 = 4 :=
by
  unfold gcd_140_396
  sorry

end gcd_140_396_is_4_l209_209157


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209705

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209705


namespace angle_of_inclination_range_l209_209501

noncomputable def curve (x : ℝ) : ℝ := 4 / (Real.exp x + 1)

noncomputable def tangent_slope (x : ℝ) : ℝ := 
  -4 * Real.exp x / (Real.exp x + 1) ^ 2

theorem angle_of_inclination_range (x : ℝ) (a : ℝ) 
  (hx : tangent_slope x = Real.tan a) : 
  (3 * Real.pi / 4 ≤ a ∧ a < Real.pi) :=
by 
  sorry

end angle_of_inclination_range_l209_209501


namespace booth_makes_50_per_day_on_popcorn_l209_209755

-- Define the conditions as provided
def daily_popcorn_revenue (P : ℝ) : Prop :=
  let cotton_candy_revenue := 3 * P
  let total_days := 5
  let rent := 30
  let ingredients := 75
  let total_expenses := rent + ingredients
  let profit := 895
  let total_revenue_before_expenses := profit + total_expenses
  total_revenue_before_expenses = 20 * P 

theorem booth_makes_50_per_day_on_popcorn : daily_popcorn_revenue 50 :=
  by sorry

end booth_makes_50_per_day_on_popcorn_l209_209755


namespace solve_for_b_l209_209380

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l209_209380


namespace three_distinct_roots_condition_l209_209185

noncomputable def k_condition (k : ℝ) : Prop :=
  ∀ (x : ℝ), (x / (x - 1) + x / (x - 3)) = k * x → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3)

theorem three_distinct_roots_condition (k : ℝ) : k ≠ 0 ↔ k_condition k :=
by
  sorry

end three_distinct_roots_condition_l209_209185


namespace find_range_of_num_on_die_l209_209315

-- Defining the conditions
def coin_toss {α : Type*} [Fintype α] :=
  (8 : Fin 8 → α)

axiom first_toss_tail {α : Type*} [Fintype α] :
  Π (s : set α), (coin_toss s).head = False

axiom equally_likely_events {α : Type*} [Fintype α] :
  ∀ s : set α, card s = 8

axiom die_roll_prob {α : Type*} [Fintype α] [Nonempty α] (r : fin 6 → Prop) :
  (1/3 : ℝ)

-- Defining the problem in Lean statement
theorem find_range_of_num_on_die {α : Type*} [Fintype α] [decidable_eq (fin 6)] :
  ∃ ranges : list (finset (fin 6)), 
    (∀ range : finset (fin 6), range.card = 2 → range ∈ ranges ∧ 
    (die_roll_prob (λ x, x ∈ range) = 1/3)) :=
sorry

end find_range_of_num_on_die_l209_209315


namespace ladder_base_distance_l209_209921

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l209_209921


namespace smallest_N_constant_l209_209827

-- Define the property to be proven
theorem smallest_N_constant (a b c : ℝ) 
  (h₁ : a + b > c) (h₂ : a + c > b) (h₃ : b + c > a) (h₄ : k = 0):
  (a^2 + b^2 + k) / c^2 > 1 / 2 :=
by
  sorry

end smallest_N_constant_l209_209827


namespace solutions_of_quadratic_l209_209282

theorem solutions_of_quadratic (x : ℝ) : x^2 - x = 0 ↔ (x = 0 ∨ x = 1) :=
by
  sorry

end solutions_of_quadratic_l209_209282


namespace integer_points_inequality_l209_209249

theorem integer_points_inequality
  (a b : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b + a - b - 5 = 0)
  (M := max ((a : ℤ)^2 + (b : ℤ)^2)) :
  (3 * x^2 + 2 * y^2 <= M) → ∃ (n : ℕ), n = 51 :=
by sorry

end integer_points_inequality_l209_209249


namespace num_black_circles_in_first_120_circles_l209_209049

theorem num_black_circles_in_first_120_circles : 
  let S := λ n : ℕ, n * (n + 1) / 2 in
  ∃ n : ℕ, S n < 120 ∧ 120 ≤ S (n + 1) := 
by
  sorry

end num_black_circles_in_first_120_circles_l209_209049


namespace lcm_of_denominators_l209_209580

theorem lcm_of_denominators : Nat.lcm (List.foldr Nat.lcm 1 [2, 3, 4, 5, 6, 7]) = 420 :=
by 
  sorry

end lcm_of_denominators_l209_209580


namespace rearrange_circles_sums13_l209_209889

def isSum13 (a b c d x y z w : ℕ) : Prop :=
  (a + 4 + b = 13) ∧ (b + 2 + d = 13) ∧ (d + 1 + c = 13) ∧ (c + 3 + a = 13)

theorem rearrange_circles_sums13 : 
  ∃ (a b c d x y z w : ℕ), 
  a = 4 ∧ b = 5 ∧ c = 6 ∧ d = 6 ∧ 
  a + b = 9 ∧ b + z = 11 ∧ z + c = 12 ∧ c + a = 10 ∧ 
  isSum13 a b c d x y z w :=
by {
  sorry
}

end rearrange_circles_sums13_l209_209889


namespace count_1988_in_S_1988_eq_phi_1988_l209_209886

def seq (n : ℕ) : List ℕ :=
  match n with
  | 1 => [1, 1]
  | 2 => [1, 2, 1]
  | m + 3 =>
    let prevSeq := seq (m + 2)
    List.bind (List.zip prevSeq (List.tail prevSeq)) (fun p => [p.1, p.1 + p.2]) ++ [List.last prevSeq 1]

def count_occurrences (n a : ℕ) : ℕ :=
  List.count a (seq n)

def phi_1988 : ℕ :=
  Nat.totient 1988

theorem count_1988_in_S_1988_eq_phi_1988 :
  count_occurrences 1988 1988 = phi_1988 := by
  sorry

end count_1988_in_S_1988_eq_phi_1988_l209_209886


namespace find_total_kids_l209_209441

-- Given conditions
def total_kids_in_camp (X : ℕ) : Prop :=
  let soccer_kids := X / 2
  let morning_soccer_kids := soccer_kids / 4
  let afternoon_soccer_kids := soccer_kids - morning_soccer_kids
  afternoon_soccer_kids = 750

-- Theorem statement
theorem find_total_kids (X : ℕ) (h : total_kids_in_camp X) : X = 2000 :=
by
  sorry

end find_total_kids_l209_209441


namespace more_movies_than_books_l209_209564

-- Conditions
def books_read := 15
def movies_watched := 29

-- Question: How many more movies than books have you watched?
theorem more_movies_than_books : (movies_watched - books_read) = 14 := sorry

end more_movies_than_books_l209_209564


namespace expected_value_decisive_games_l209_209469

/-- According to the rules of a chess match, the winner is the one who gains two victories over the opponent. -/
def winner_conditions (a b : Nat) : Prop :=
  a = 2 ∨ b = 2

/-- A game match where the probabilities of winning for the opponents are equal.-/
def probabilities_equal : Prop :=
  true

/-- Define X as the random variable representing the number of decisive games in the match. -/
def X (a b : Nat) : Nat :=
  a + b

/-- The expected value of the number of decisive games given equal probabilities of winning. -/
theorem expected_value_decisive_games (a b : Nat) (h1 : winner_conditions a b) (h2 : probabilities_equal) : 
  (X a b) / 2 = 4 :=
sorry

end expected_value_decisive_games_l209_209469


namespace probability_at_least_two_same_l209_209411

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l209_209411


namespace area_of_triangle_is_2_l209_209848

-- Define the conditions of the problem
variable (a b c : ℝ)
variable (A B C : ℝ)  -- Angles in radians

-- Conditions for the triangle ABC
variable (sin_A : ℝ) (sin_C : ℝ)
variable (c2sinA_eq_5sinC : c^2 * sin_A = 5 * sin_C)
variable (a_plus_c_squared_eq_16_plus_b_squared : (a + c)^2 = 16 + b^2)
variable (ac_eq_5 : a * c = 5)
variable (cos_B : ℝ)
variable (sin_B : ℝ)

-- Sine and Cosine law results
variable (cos_B_def : cos_B = (a^2 + c^2 - b^2) / (2 * a * c))
variable (sin_B_def : sin_B = Real.sqrt (1 - cos_B^2))

-- Area of the triangle
noncomputable def area_triangle_ABC := (1/2) * a * c * sin_B

-- Theorem to prove the area
theorem area_of_triangle_is_2 :
  area_triangle_ABC a c sin_B = 2 :=
by
  rw [area_triangle_ABC]
  sorry

end area_of_triangle_is_2_l209_209848


namespace xy_product_eq_two_l209_209203

theorem xy_product_eq_two (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 2 / x = y + 2 / y) : x * y = 2 := 
sorry

end xy_product_eq_two_l209_209203


namespace max_side_parallel_to_barn_correct_l209_209038

noncomputable def maximizeArea : Real :=
let cost_per_foot := 5
let total_cost := 1500
let total_length := total_cost / cost_per_foot
let area (x : ℝ) := x * (total_length - 2 * x)
let x := deriv area x
let critical_point := 75
let side_parallel_to_barn := total_length - 2 * critical_point
side_parallel_to_barn

theorem max_side_parallel_to_barn_correct :
  maximizeArea = 150 :=
by
  sorry

end max_side_parallel_to_barn_correct_l209_209038


namespace xy_product_solution_l209_209456

theorem xy_product_solution (x y : ℝ)
  (h1 : x / (x^2 * y^2 - 1) - 1 / x = 4)
  (h2 : (x^2 * y) / (x^2 * y^2 - 1) + y = 2) :
  x * y = 1 / Real.sqrt 2 ∨ x * y = -1 / Real.sqrt 2 :=
sorry

end xy_product_solution_l209_209456


namespace only_powers_of_2_satisfy_condition_l209_209628

theorem only_powers_of_2_satisfy_condition:
  ∀ (n : ℕ), n ≥ 2 →
  (∃ (x : ℕ → ℕ), 
    ∀ (i j : ℕ), 
      0 < i ∧ i < n → 0 < j ∧ j < n → i ≠ j ∧ (n ∣ (2 * i + j)) → x i < x j) ↔
      ∃ (s : ℕ), n = 2^s ∧ s ≥ 1 :=
by
  sorry

end only_powers_of_2_satisfy_condition_l209_209628


namespace find_number_l209_209084

-- Let's define the condition
def condition (x : ℝ) : Prop := x * 99999 = 58293485180

-- Statement to be proved
theorem find_number : ∃ x : ℝ, condition x ∧ x = 582.935 := 
by
  sorry

end find_number_l209_209084


namespace poly_coeff_difference_l209_209646

theorem poly_coeff_difference :
  ∀ (a a_1 a_2 a_3 a_4 : ℝ),
  (2 + x)^4 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 →
  a = 16 →
  1 = a - a_1 + a_2 - a_3 + a_4 →
  a_2 - a_1 + a_4 - a_3 = -15 :=
by
  intros a a_1 a_2 a_3 a_4 h_poly h_a h_eq
  sorry

end poly_coeff_difference_l209_209646


namespace equivalent_single_reduction_l209_209004

theorem equivalent_single_reduction :
  ∀ (P : ℝ), P * (1 - 0.25) * (1 - 0.20) = P * (1 - 0.40) :=
by
  intros P
  -- Proof will be skipped
  sorry

end equivalent_single_reduction_l209_209004


namespace ladder_base_distance_l209_209956

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l209_209956


namespace price_of_pastries_is_5_l209_209266

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l209_209266


namespace ellipse_range_of_k_l209_209508

theorem ellipse_range_of_k (k : ℝ) :
  (1 - k > 0) ∧ (1 + k > 0) ∧ (1 - k ≠ 1 + k) ↔ (-1 < k ∧ k < 0) ∨ (0 < k ∧ k < 1) :=
by
  sorry

end ellipse_range_of_k_l209_209508


namespace determinant_calculation_l209_209631

variable {R : Type*} [CommRing R]

def matrix_example (a b c : R) : Matrix (Fin 3) (Fin 3) R :=
  ![![1, a, b], ![1, a + b, b + c], ![1, a, a + c]]

theorem determinant_calculation (a b c : R) :
  (matrix_example a b c).det = ab + b^2 + bc :=
by sorry

end determinant_calculation_l209_209631


namespace lines_through_point_l209_209497

theorem lines_through_point {a b c : ℝ} :
  (3 = a + b) ∧ (3 = b + c) ∧ (3 = c + a) → (a = 1.5 ∧ b = 1.5 ∧ c = 1.5) :=
by
  intros h
  sorry

end lines_through_point_l209_209497


namespace water_for_1200ml_flour_l209_209703

-- Define the condition of how much water is mixed with a specific amount of flour
def water_per_flour (flour water : ℕ) : Prop :=
  water = (flour / 400) * 100

-- Given condition: Maria uses 100 mL of water for every 400 mL of flour
def condition : Prop := water_per_flour 400 100

-- Problem Statement: How many mL of water for 1200 mL of flour?
theorem water_for_1200ml_flour (h : condition) : water_per_flour 1200 300 :=
sorry

end water_for_1200ml_flour_l209_209703


namespace fraction_phone_numbers_9_ending_even_l209_209612

def isValidPhoneNumber (n : Nat) : Bool :=
  n / 10^6 != 0 && n / 10^6 != 1 && n / 10^6 != 2

def isValidEndEven (n : Nat) : Bool :=
  let lastDigit := n % 10
  lastDigit == 0 || lastDigit == 2 || lastDigit == 4 || lastDigit == 6 || lastDigit == 8

def countValidPhoneNumbers : Nat :=
  7 * 10^6

def countValidStarting9EndingEven : Nat :=
  5 * 10^5

theorem fraction_phone_numbers_9_ending_even :
  (countValidStarting9EndingEven : ℚ) / (countValidPhoneNumbers : ℚ) = 1 / 14 :=
by 
  sorry

end fraction_phone_numbers_9_ending_even_l209_209612


namespace sec_150_eq_neg_2_sqrt3_div_3_l209_209781

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l209_209781


namespace find_g_l209_209747

theorem find_g (f g : ℝ → ℝ) (h1 : ∀ x : ℝ, f (x+1) = 3 - 2 * x) (h2 : ∀ x : ℝ, f (g x) = 6 * x - 3) : 
  ∀ x : ℝ, g x = 4 - 3 * x := 
by
  sorry

end find_g_l209_209747


namespace indistinguishable_distributions_l209_209103

def ways_to_distribute_balls (balls : ℕ) (boxes : ℕ) : ℕ :=
  if boxes = 2 && balls = 6 then 4 else 0

theorem indistinguishable_distributions : ways_to_distribute_balls 6 2 = 4 :=
by sorry

end indistinguishable_distributions_l209_209103


namespace expected_profit_l209_209893

namespace DailyLottery

/-- Definitions for the problem -/

def ticket_cost : ℝ := 2
def first_prize : ℝ := 100
def second_prize : ℝ := 10
def prob_first_prize : ℝ := 0.001
def prob_second_prize : ℝ := 0.1
def prob_no_prize : ℝ := 1 - prob_first_prize - prob_second_prize

/-- Expected profit calculation as a theorem -/

theorem expected_profit :
  (first_prize * prob_first_prize + second_prize * prob_second_prize + 0 * prob_no_prize) - ticket_cost = -0.9 :=
by
  sorry

end DailyLottery

end expected_profit_l209_209893


namespace difference_between_twice_smaller_and_larger_is_three_l209_209544

theorem difference_between_twice_smaller_and_larger_is_three
(S L x : ℕ) 
(h1 : L = 2 * S - x) 
(h2 : S + L = 39)
(h3 : S = 14) : 
2 * S - L = 3 := 
sorry

end difference_between_twice_smaller_and_larger_is_three_l209_209544


namespace board_cut_ratio_l209_209034

theorem board_cut_ratio (L S : ℝ) (h1 : S + L = 20) (h2 : S = L + 4) (h3 : S = 8.0) : S / L = 1 := by
  sorry

end board_cut_ratio_l209_209034


namespace determine_x_l209_209075

theorem determine_x (x : ℚ) (h : ∀ y : ℚ, 10 * x * y - 15 * y + 3 * x - (9 / 2) = 0) : x = 3 / 2 :=
sorry

end determine_x_l209_209075


namespace boxes_A_B_cost_condition_boxes_B_profit_condition_l209_209595

/-
Part 1: Prove the number of brand A boxes is 60 and number of brand B boxes is 40 given the cost condition.
-/
theorem boxes_A_B_cost_condition (x : ℕ) (y : ℕ) :
  80 * x + 130 * y = 10000 ∧ x + y = 100 → x = 60 ∧ y = 40 :=
by sorry

/-
Part 2: Prove the number of brand B boxes should be at least 54 given the profit condition.
-/
theorem boxes_B_profit_condition (y : ℕ) :
  40 * (100 - y) + 70 * y ≥ 5600 → y ≥ 54 :=
by sorry

end boxes_A_B_cost_condition_boxes_B_profit_condition_l209_209595


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209813

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209813


namespace probability_red_or_white_l209_209744

-- Definitions based on the conditions
def total_marbles := 20
def blue_marbles := 5
def red_marbles := 9
def white_marbles := total_marbles - (blue_marbles + red_marbles)

-- Prove that the probability of selecting a red or white marble is 3/4
theorem probability_red_or_white : (red_marbles + white_marbles : ℚ) / total_marbles = 3 / 4 :=
by sorry

end probability_red_or_white_l209_209744


namespace smaller_rectangle_perimeter_l209_209175

def perimeter_original_rectangle (a b : ℝ) : Prop := 2 * (a + b) = 100
def number_of_cuts (vertical_cuts horizontal_cuts : ℕ) : Prop := vertical_cuts = 7 ∧ horizontal_cuts = 10
def total_length_of_cuts (a b : ℝ) : Prop := 7 * b + 10 * a = 434

theorem smaller_rectangle_perimeter (a b : ℝ) (vertical_cuts horizontal_cuts : ℕ) (m n : ℕ) :
  perimeter_original_rectangle a b →
  number_of_cuts vertical_cuts horizontal_cuts →
  total_length_of_cuts a b →
  (m = 8) →
  (n = 11) →
  (a / 8 + b / 11) * 2 = 11 :=
by
  sorry

end smaller_rectangle_perimeter_l209_209175


namespace intersection_point_ordinate_interval_l209_209001

theorem intersection_point_ordinate_interval:
  ∃ m : ℤ, ∀ x : ℝ, e ^ x = 5 - x → 3 < x ∧ x < 4 :=
by sorry

end intersection_point_ordinate_interval_l209_209001


namespace value_of_b_cannot_form_arithmetic_sequence_l209_209699

theorem value_of_b 
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b > 0) :
  b = 5 * Real.sqrt 10 := 
sorry

theorem cannot_form_arithmetic_sequence 
  (d : ℝ)
  (a1 : ℝ) (a2 : ℝ) (a3 : ℝ) 
  (h1 : a1 = 150)
  (h2 : a2 = b)
  (h3 : a3 = 60 / 36)
  (h4 : b = 5 * Real.sqrt 10) :
  ¬(∃ d, a1 + d = a2 ∧ a2 + d = a3) := 
sorry

end value_of_b_cannot_form_arithmetic_sequence_l209_209699


namespace books_sold_on_monday_75_l209_209532

namespace Bookstore

variables (total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold : ℕ)
variable (percent_not_sold : ℝ)

def given_conditions : Prop :=
  total_books = 1200 ∧
  percent_not_sold = 0.665 ∧
  sold_Tuesday = 50 ∧
  sold_Wednesday = 64 ∧
  sold_Thursday = 78 ∧
  sold_Friday = 135 ∧
  books_not_sold = (percent_not_sold * total_books) ∧
  (total_books - books_not_sold) = (sold_Monday + sold_Tuesday + sold_Wednesday + sold_Thursday + sold_Friday)

theorem books_sold_on_monday_75 (h : given_conditions total_books sold_Monday sold_Tuesday sold_Wednesday sold_Thursday sold_Friday books_not_sold percent_not_sold) :
  sold_Monday = 75 :=
sorry

end Bookstore

end books_sold_on_monday_75_l209_209532


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209719

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209719


namespace max_cookies_Andy_could_have_eaten_l209_209154

theorem max_cookies_Andy_could_have_eaten (cookies : ℕ) (Andy Alexa : ℕ) 
  (h1 : cookies = 24) 
  (h2 : Alexa = k * Andy) 
  (h3 : k > 0) 
  (h4 : Andy + Alexa = cookies) 
  : Andy ≤ 12 := 
sorry

end max_cookies_Andy_could_have_eaten_l209_209154


namespace additive_inverse_commutativity_l209_209849

section
  variable {R : Type} [Ring R] (h : ∀ x : R, x ^ 2 = x)

  theorem additive_inverse (x : R) : -x = x := by
    sorry

  theorem commutativity (x y : R) : x * y = y * x := by
    sorry
end

end additive_inverse_commutativity_l209_209849


namespace julia_tuesday_l209_209533

variable (M : ℕ) -- The number of kids Julia played with on Monday
variable (T : ℕ) -- The number of kids Julia played with on Tuesday

-- Conditions
def condition1 : Prop := M = T + 8
def condition2 : Prop := M = 22

-- Theorem to prove
theorem julia_tuesday : condition1 M T → condition2 M → T = 14 := by
  sorry

end julia_tuesday_l209_209533


namespace sandy_spent_on_shorts_l209_209548

variable (amount_on_shirt amount_on_jacket total_amount amount_on_shorts : ℝ)

theorem sandy_spent_on_shorts :
  amount_on_shirt = 12.14 →
  amount_on_jacket = 7.43 →
  total_amount = 33.56 →
  amount_on_shorts = total_amount - amount_on_shirt - amount_on_jacket →
  amount_on_shorts = 13.99 :=
by
  intros h_shirt h_jacket h_total h_computation
  sorry

end sandy_spent_on_shorts_l209_209548


namespace initial_brownies_l209_209675

theorem initial_brownies (B : ℕ) (eaten_by_father : ℕ) (eaten_by_mooney : ℕ) (new_brownies : ℕ) (total_brownies : ℕ) :
  eaten_by_father = 8 →
  eaten_by_mooney = 4 →
  new_brownies = 24 →
  total_brownies = 36 →
  (B - (eaten_by_father + eaten_by_mooney) + new_brownies = total_brownies) →
  B = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end initial_brownies_l209_209675


namespace sec_150_eq_neg_2_sqrt3_div_3_l209_209782

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l209_209782


namespace simplify_expression_l209_209912

theorem simplify_expression (x : ℝ) (h : x ≤ 2) : 
  (Real.sqrt (x^2 - 4*x + 4) - Real.sqrt (x^2 - 6*x + 9)) = -1 :=
by 
  sorry

end simplify_expression_l209_209912


namespace find_EQ_l209_209568

open Real

noncomputable def Trapezoid_EFGH (EF FG GH HE EQ QF : ℝ) : Prop :=
  EF = 110 ∧
  FG = 60 ∧
  GH = 23 ∧
  HE = 75 ∧
  EQ + QF = EF ∧
  EQ = 250 / 3

theorem find_EQ (EF FG GH HE EQ QF : ℝ) (h : Trapezoid_EFGH EF FG GH HE EQ QF) :
  EQ = 250 / 3 :=
by
  sorry

end find_EQ_l209_209568


namespace math_problem_l209_209098

noncomputable def alpha_condition (α : ℝ) : Prop :=
  4 * Real.cos α - 2 * Real.sin α = 0

theorem math_problem (α : ℝ) (h : alpha_condition α) :
  (Real.sin α)^3 + (Real.cos α)^3 / (Real.sin α - Real.cos α) = 9 / 5 :=
  sorry

end math_problem_l209_209098


namespace hyperbrick_hyperbox_probability_l209_209353

theorem hyperbrick_hyperbox_probability :
  let a_nums := {1, 2, 3, ... 500}.to_finset
  let a_sample := a_nums.sample_without_replacement 5
  let b_nums := a_nums \ a_sample
  let b_sample := b_nums.sample_without_replacement 4
  let q := (32 : ℚ) / 126
  let reduced_q := q.num.gcd q.denom
  ((32/126).num / (32/126).denom).num + ((32/126).num / (32/126).denom).denom = 79 :=
sorry

end hyperbrick_hyperbox_probability_l209_209353


namespace emerson_rowed_last_part_l209_209487

-- Define the given conditions
def emerson_initial_distance: ℝ := 6
def emerson_continued_distance: ℝ := 15
def total_trip_distance: ℝ := 39

-- Define the distance Emerson covered before the last part
def distance_before_last_part := emerson_initial_distance + emerson_continued_distance

-- Define the distance Emerson rowed in the last part of his trip
def distance_last_part := total_trip_distance - distance_before_last_part

-- The theorem we need to prove
theorem emerson_rowed_last_part : distance_last_part = 18 := by
  sorry

end emerson_rowed_last_part_l209_209487


namespace evaluate_f_3_minus_f_neg3_l209_209073

def f (x : ℝ) : ℝ := x^6 + x^4 + 3*x^3 + 4*x^2 + 8*x

theorem evaluate_f_3_minus_f_neg3 : f 3 - f (-3) = 210 := by
  sorry

end evaluate_f_3_minus_f_neg3_l209_209073


namespace cuboid_first_edge_length_l209_209427

theorem cuboid_first_edge_length (x : ℝ) (hx : 180 = x * 5 * 6) : x = 6 :=
by
  sorry

end cuboid_first_edge_length_l209_209427


namespace graph_passes_through_point_l209_209198

theorem graph_passes_through_point (a : ℝ) (h : a < 0) : (0, 0) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p = (x, (1 - a)^x - 1)} :=
by
  sorry

end graph_passes_through_point_l209_209198


namespace dave_paid_more_l209_209778

-- Definitions based on conditions in the problem statement
def total_pizza_cost : ℕ := 11  -- Total cost of the pizza in dollars
def num_slices : ℕ := 8  -- Total number of slices in the pizza
def plain_pizza_cost : ℕ := 8  -- Cost of the plain pizza in dollars
def anchovies_cost : ℕ := 2  -- Extra cost of adding anchovies in dollars
def mushrooms_cost : ℕ := 1  -- Extra cost of adding mushrooms in dollars
def dave_slices : ℕ := 7  -- Number of slices Dave ate
def doug_slices : ℕ := 1  -- Number of slices Doug ate
def doug_payment : ℕ := 1  -- Amount Doug paid in dollars
def dave_payment : ℕ := total_pizza_cost - doug_payment  -- Amount Dave paid in dollars

-- Prove that Dave paid 9 dollars more than Doug
theorem dave_paid_more : dave_payment - doug_payment = 9 := by
  -- Proof to be filled in
  sorry

end dave_paid_more_l209_209778


namespace find_h_l209_209430

theorem find_h (j k h : ℕ) (h₁ : 2013 = 3 * h^2 + j) (h₂ : 2014 = 2 * h^2 + k)
  (pos_int_x_intercepts_1 : ∃ x1 x2 : ℕ, x1 ≠ x2 ∧ x1 > 0 ∧ x2 > 0 ∧ (3 * (x1 - h)^2 + j = 0 ∧ 3 * (x2 - h)^2 + j = 0))
  (pos_int_x_intercepts_2 : ∃ y1 y2 : ℕ, y1 ≠ y2 ∧ y1 > 0 ∧ y2 > 0 ∧ (2 * (y1 - h)^2 + k = 0 ∧ 2 * (y2 - h)^2 + k = 0)):
  h = 36 :=
by
  sorry

end find_h_l209_209430


namespace intersection_of_A_and_B_l209_209643

-- Define the sets A and B
def A : Set ℕ := {1, 6, 8, 10}
def B : Set ℕ := {2, 4, 8, 10}

-- Prove that the intersection of A and B is {8, 10}
theorem intersection_of_A_and_B : A ∩ B = {8, 10} :=
by
  -- Proof will be filled here
  sorry

end intersection_of_A_and_B_l209_209643


namespace average_price_blankets_l209_209464

theorem average_price_blankets :
  let cost_blankets1 := 3 * 100
  let cost_blankets2 := 5 * 150
  let cost_blankets3 := 550
  let total_cost := cost_blankets1 + cost_blankets2 + cost_blankets3
  let total_blankets := 3 + 5 + 2
  total_cost / total_blankets = 160 :=
by
  sorry

end average_price_blankets_l209_209464


namespace vector_subtraction_proof_l209_209100

theorem vector_subtraction_proof (a b : ℝ × ℝ) (ha : a = (3, 2)) (hb : b = (0, -1)) :
    3 • b - a = (-3, -5) := by
  sorry

end vector_subtraction_proof_l209_209100


namespace remainder_of_7_pow_145_mod_12_l209_209907

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l209_209907


namespace sum_proof_l209_209562

-- Define the context and assumptions
variables (F S T : ℕ)
axiom sum_of_numbers : F + S + T = 264
axiom first_number_twice_second : F = 2 * S
axiom third_number_one_third_first : T = F / 3
axiom second_number_given : S = 72

-- The theorem to prove the sum is 264 given the conditions
theorem sum_proof : F + S + T = 264 :=
by
  -- Given conditions already imply the theorem, the actual proof follows from these
  sorry

end sum_proof_l209_209562


namespace cost_of_pastrami_l209_209261

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l209_209261


namespace f_properties_l209_209485

theorem f_properties (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x^2) - f (y^2) ≤ (f x + y) * (x - f y)) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = -x) :=
sorry

end f_properties_l209_209485


namespace value_of_expression_l209_209669

open Polynomial

theorem value_of_expression (a b : ℚ) (h1 : (3 : ℚ) * a ^ 2 + 9 * a - 21 = 0) (h2 : (3 : ℚ) * b ^ 2 + 9 * b - 21 = 0) :
  (3 * a - 4) * (2 * b - 2) = -4 :=
by sorry

end value_of_expression_l209_209669


namespace books_read_so_far_l209_209148

/-- There are 22 different books in the 'crazy silly school' series -/
def total_books : Nat := 22

/-- You still have to read 10 more books -/
def books_left_to_read : Nat := 10

theorem books_read_so_far :
  total_books - books_left_to_read = 12 :=
by
  sorry

end books_read_so_far_l209_209148


namespace sum_of_primes_no_solution_l209_209341

def is_prime (n : ℕ) : Prop := Nat.Prime n

def no_solution (p : ℕ) : Prop :=
  is_prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]

def gcd_condition (p : ℕ) : Prop :=
  p = 2 ∨ p = 5

theorem sum_of_primes_no_solution : (∑ p in {p | is_prime p ∧ gcd_condition p}, p) = 7 :=
by
  sorry

end sum_of_primes_no_solution_l209_209341


namespace mason_father_age_l209_209540

theorem mason_father_age
  (Mason_age : ℕ) 
  (Sydney_age : ℕ) 
  (Father_age : ℕ)
  (h1 : Mason_age = 20)
  (h2 : Sydney_age = 3 * Mason_age)
  (h3 : Father_age = Sydney_age + 6) :
  Father_age = 66 :=
by
  sorry

end mason_father_age_l209_209540


namespace contractor_realized_after_20_days_l209_209168

-- Defining the conditions as assumptions
variables {W : ℝ} {r : ℝ} {x : ℝ} -- Total work, rate per person per day, and unknown number of days

-- Condition 1: 10 people to complete W work in x days results in one fourth completed
axiom one_fourth_work_done (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4

-- Condition 2: After firing 2 people, 8 people complete three fourths of work in 75 days
axiom remaining_three_fourths_work_done (W : ℝ) (r : ℝ) :
  8 * r * 75 = 3 * (W / 4)

-- Theorem: The contractor realized that one fourth of the work was done after 20 days
theorem contractor_realized_after_20_days (W : ℝ) (r : ℝ) (x : ℝ) :
  10 * r * x = W / 4 → (8 * r * 75 = 3 * (W / 4)) → x = 20 := 
sorry

end contractor_realized_after_20_days_l209_209168


namespace hyperbola_equation_l209_209852

theorem hyperbola_equation 
  (h k a c : ℝ)
  (center_cond : (h, k) = (3, -1))
  (vertex_cond : a = abs (2 - (-1)))
  (focus_cond : c = abs (7 - (-1)))
  (b : ℝ)
  (b_square : c^2 = a^2 + b^2) :
  h + k + a + b = 5 + Real.sqrt 55 := 
by
  -- Prove that given the conditions, the value of h + k + a + b is 5 + √55.
  sorry

end hyperbola_equation_l209_209852


namespace ratio_won_to_lost_l209_209242

-- Define the total number of games and the number of games won
def total_games : Nat := 30
def games_won : Nat := 18

-- Define the number of games lost
def games_lost : Nat := total_games - games_won

-- Define the ratio of games won to games lost as a pair
def ratio : Nat × Nat := (games_won / Nat.gcd games_won games_lost, games_lost / Nat.gcd games_won games_lost)

-- The theorem to be proved
theorem ratio_won_to_lost : ratio = (3, 2) :=
  by
    -- Skipping the proof here
    sorry

end ratio_won_to_lost_l209_209242


namespace oldest_bride_age_l209_209189

theorem oldest_bride_age (G B : ℕ) (h1 : B = G + 19) (h2 : B + G = 185) : B = 102 :=
by
  sorry

end oldest_bride_age_l209_209189


namespace ratio_of_screams_to_hours_l209_209543

-- Definitions from conditions
def hours_hired : ℕ := 6
def current_babysitter_rate : ℕ := 16
def new_babysitter_rate : ℕ := 12
def extra_charge_per_scream : ℕ := 3
def cost_difference : ℕ := 18

-- Calculate necessary costs
def current_babysitter_cost : ℕ := current_babysitter_rate * hours_hired
def new_babysitter_base_cost : ℕ := new_babysitter_rate * hours_hired
def new_babysitter_total_cost : ℕ := current_babysitter_cost - cost_difference
def screams_cost : ℕ := new_babysitter_total_cost - new_babysitter_base_cost
def number_of_screams : ℕ := screams_cost / extra_charge_per_scream

-- Theorem to prove the ratio
theorem ratio_of_screams_to_hours : number_of_screams / hours_hired = 1 := by
  sorry

end ratio_of_screams_to_hours_l209_209543


namespace tiffany_daily_miles_l209_209334

-- Definitions for running schedule
def billy_sunday_miles := 1
def billy_monday_miles := 1
def billy_tuesday_miles := 1
def billy_wednesday_miles := 1
def billy_thursday_miles := 1
def billy_friday_miles := 1
def billy_saturday_miles := 1

def tiffany_wednesday_miles := 1 / 3
def tiffany_thursday_miles := 1 / 3
def tiffany_friday_miles := 1 / 3

-- Total miles is the sum of miles for the week
def billy_total_miles := billy_sunday_miles + billy_monday_miles + billy_tuesday_miles +
                         billy_wednesday_miles + billy_thursday_miles + billy_friday_miles +
                         billy_saturday_miles

def tiffany_total_miles (T : ℝ) := T * 3 + 
                                   tiffany_wednesday_miles + tiffany_thursday_miles + tiffany_friday_miles

-- Proof problem: show that Tiffany runs 2 miles each day on Sunday, Monday, and Tuesday
theorem tiffany_daily_miles : ∃ T : ℝ, (tiffany_total_miles T = billy_total_miles) ∧ T = 2 :=
by
  sorry

end tiffany_daily_miles_l209_209334


namespace sec_150_eq_neg_two_sqrt_three_div_three_l209_209788

theorem sec_150_eq_neg_two_sqrt_three_div_three : 
  real.sec (150 * real.pi / 180) = -2 * real.sqrt 3 / 3 := 
by 
sorry

end sec_150_eq_neg_two_sqrt_three_div_three_l209_209788


namespace balls_in_boxes_l209_209514

theorem balls_in_boxes : 
  (number_of_ways : ℕ) = 52 :=
by
  let number_of_balls := 5
  let number_of_boxes := 4
  let balls_indistinguishable := true
  let boxes_distinguishable := true
  let max_balls_per_box := 3
  
  -- Proof omitted
  sorry

end balls_in_boxes_l209_209514


namespace other_solution_of_quadratic_l209_209365

theorem other_solution_of_quadratic (x : ℚ) 
  (hx1 : 77 * x^2 - 125 * x + 49 = 0) (hx2 : x = 8/11) : 
  77 * (1 : ℚ)^2 - 125 * (1 : ℚ) + 49 = 0 :=
by sorry

end other_solution_of_quadratic_l209_209365


namespace range_of_a_l209_209466

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, -2 < x ∧ x < -1 → (a + x) * (1 + x) < 0) → a > 2 :=
by
  sorry

end range_of_a_l209_209466


namespace number_added_l209_209320

def initial_number : ℕ := 9
def final_resultant : ℕ := 93

theorem number_added : ∃ x : ℕ, 3 * (2 * initial_number + x) = final_resultant ∧ x = 13 := by
  sorry

end number_added_l209_209320


namespace base_from_wall_l209_209967

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l209_209967


namespace total_crayons_l209_209159

-- Define relevant conditions
def crayons_per_child : ℕ := 8
def number_of_children : ℕ := 7

-- Define the Lean statement to prove the total number of crayons
theorem total_crayons : crayons_per_child * number_of_children = 56 :=
by
  sorry

end total_crayons_l209_209159


namespace find_period_l209_209211

noncomputable theory

def period_of_sine (x : ℝ) : ℝ := Real.sin (π * x + π / 3)

theorem find_period : ∃ T, ∀ x, period_of_sine (x + T) = period_of_sine x ∧ T = 2 :=
by
  sorry

end find_period_l209_209211


namespace price_of_pastries_is_5_l209_209265

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l209_209265


namespace optimal_garden_area_l209_209019

variable (l w : ℕ)

/-- Tiffany is building a fence around a rectangular garden. Determine the optimal area, 
    in square feet, that can be enclosed under the conditions. -/
theorem optimal_garden_area 
  (h1 : l >= 100)
  (h2 : w >= 50)
  (h3 : 2 * l + 2 * w = 400) : (l * w) ≤ 7500 := 
sorry

end optimal_garden_area_l209_209019


namespace probability_not_finishing_on_time_l209_209005

-- Definitions based on the conditions
def P_finishing_on_time : ℚ := 5 / 8

-- Theorem to prove the required probability
theorem probability_not_finishing_on_time :
  (1 - P_finishing_on_time) = 3 / 8 := by
  sorry

end probability_not_finishing_on_time_l209_209005


namespace seq_sum_difference_l209_209061

-- Define the sequences
def seq1 : List ℕ := List.range 93 |> List.map (λ n => 2001 + n)
def seq2 : List ℕ := List.range 93 |> List.map (λ n => 301 + n)

-- Define the sum of the sequences
def sum_seq1 : ℕ := seq1.sum
def sum_seq2 : ℕ := seq2.sum

-- Define the difference between the sums of the sequences
def diff_seq_sum : ℕ := sum_seq1 - sum_seq2

-- Lean statement to prove the difference equals 158100
theorem seq_sum_difference : diff_seq_sum = 158100 := by
  sorry

end seq_sum_difference_l209_209061


namespace binomial_coefficient_x3y5_in_expansion_l209_209731

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209731


namespace simplify_expression_l209_209269

variable (a : ℝ)

theorem simplify_expression : 2 * a * (2 * a ^ 2 + a) - a ^ 2 = 4 * a ^ 3 + a ^ 2 := 
  sorry

end simplify_expression_l209_209269


namespace fair_coin_three_flips_l209_209027

open ProbabilityTheory

/-- When flipping a fair coin three times, the probability that the first flip is heads and 
    the last two flips are tails is 1/8. -/
theorem fair_coin_three_flips (p : Real) (H : p = 1/2) :
  P (λ (s : Fin 3 → Bool), s 0 = tt ∧ s 1 = ff ∧ s 2 = ff) = 1/8 := 
sorry

end fair_coin_three_flips_l209_209027


namespace expected_value_decisive_games_l209_209471

theorem expected_value_decisive_games :
  let X : ℕ → ℕ := -- Random variable representing the number of decisive games
    -- Expected value calculation for random variable X
    have h : ∃ e : ℕ, e = (2 * 1/2 + (2 + e) * 1/2), from sorry, 
    -- Extracting the expected value from the equation
    let ⟨E_X, h_ex⟩ := Classical.indefinite_description (λ e, e = (2 * 1/2 + (2 + e) * 1/2)) h in
    E_X = 4 :=
begin
  sorry,
end

end expected_value_decisive_games_l209_209471


namespace number_of_children_l209_209118

-- Definitions based on conditions
def numDogs : ℕ := 2
def numCats : ℕ := 1
def numLegsTotal : ℕ := 22
def numLegsDog : ℕ := 4
def numLegsCat : ℕ := 4
def numLegsHuman : ℕ := 2

-- Main theorem proving the number of children
theorem number_of_children :
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  numChildren = 4 :=
by
  let totalPetLegs := (numDogs * numLegsDog) + (numCats * numLegsCat)
  let totalLegsAccounted := totalPetLegs + numLegsHuman
  let numLegsRemaining := numLegsTotal - totalLegsAccounted
  let numChildren := numLegsRemaining / numLegsHuman
  exact sorry

end number_of_children_l209_209118


namespace f_m_eq_five_l209_209644

def f (x : ℝ) (a : ℝ) : ℝ :=
  x^3 + a * x + 3

axiom f_neg_m : ∀ (m a : ℝ), f (-m) a = 1

theorem f_m_eq_five (m a : ℝ) (h : f (-m) a = 1) : f m a = 5 :=
  by sorry

end f_m_eq_five_l209_209644


namespace total_accessories_correct_l209_209663

-- Definitions
def dresses_first_period := 10 * 4
def dresses_second_period := 3 * 5
def total_dresses := dresses_first_period + dresses_second_period
def accessories_per_dress := 3 + 2 + 1
def total_accessories := total_dresses * accessories_per_dress

-- Theorem statement
theorem total_accessories_correct : total_accessories = 330 := by
  sorry

end total_accessories_correct_l209_209663


namespace find_n_l209_209701

theorem find_n (n k : ℕ) (h_pos : k > 0) (h_calls : ∀ (s : Finset (Fin n)), s.card = n-2 → (∃ (f : Finset (Fin n × Fin n)), f.card = 3^k ∧ ∀ (x y : Fin n), (x, y) ∈ f → x ≠ y)) : n = 5 := 
sorry

end find_n_l209_209701


namespace possible_values_of_g_zero_l209_209277

variable {g : ℝ → ℝ}

theorem possible_values_of_g_zero (h : ∀ x : ℝ, g (2 * x) = g x ^ 2) : g 0 = 0 ∨ g 0 = 1 := 
sorry

end possible_values_of_g_zero_l209_209277


namespace max_sum_value_l209_209394

noncomputable def max_sum (x y : ℝ) (h : 3 * (x^2 + y^2) = x - y) : ℝ :=
  x + y

theorem max_sum_value :
  ∃ x y : ℝ, ∃ h : 3 * (x^2 + y^2) = x - y, max_sum x y h = 1/3 :=
sorry

end max_sum_value_l209_209394


namespace circumference_base_of_cone_l209_209312

-- Define the given conditions
def radius_circle : ℝ := 6
def angle_sector : ℝ := 300

-- Define the problem to prove the circumference of the base of the resulting cone in terms of π
theorem circumference_base_of_cone :
  (angle_sector / 360) * (2 * π * radius_circle) = 10 * π := by
sorry

end circumference_base_of_cone_l209_209312


namespace reciprocal_of_neg3_l209_209007

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l209_209007


namespace binomial_coefficient_x3y5_in_expansion_l209_209712

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209712


namespace find_prime_n_l209_209093

def is_prime (p : ℕ) : Prop := 
  p > 1 ∧ (∀ n, n ∣ p → n = 1 ∨ n = p)

def prime_candidates : List ℕ := [11, 17, 23, 29, 41, 47, 53, 59, 61, 71, 83, 89]

theorem find_prime_n (n : ℕ) 
  (h1 : n ∈ prime_candidates) 
  (h2 : is_prime (n)) 
  (h3 : is_prime (n + 20180500)) : 
  n = 61 :=
by sorry

end find_prime_n_l209_209093


namespace zongzi_unit_prices_max_type_A_zongzi_l209_209630

theorem zongzi_unit_prices (x : ℝ) : 
  (800 / x - 1200 / (2 * x) = 50) → 
  (x = 4 ∧ 2 * x = 8) :=
by
  intro h
  sorry

theorem max_type_A_zongzi (m : ℕ) : 
  (m ≤ 200) → 
  (8 * m + 4 * (200 - m) ≤ 1150) → 
  (m ≤ 87) :=
by
  intros h1 h2
  sorry

end zongzi_unit_prices_max_type_A_zongzi_l209_209630


namespace sequence_term_is_square_l209_209874

noncomputable def sequence_term (n : ℕ) : ℕ :=
  let part1 := (10 ^ (n + 1) - 1) / 9
  let part2 := (10 ^ (2 * n + 2) - 10 ^ (n + 1)) / 9
  1 + 4 * part1 + 4 * part2

theorem sequence_term_is_square (n : ℕ) : ∃ k : ℕ, k^2 = sequence_term n :=
by
  sorry

end sequence_term_is_square_l209_209874


namespace bobbit_worm_days_l209_209567

variable (initial_fish : ℕ)
variable (fish_added : ℕ)
variable (fish_eaten_per_day : ℕ)
variable (week_days : ℕ)
variable (final_fish : ℕ)
variable (d : ℕ)

theorem bobbit_worm_days (h1 : initial_fish = 60)
                         (h2 : fish_added = 8)
                         (h3 : fish_eaten_per_day = 2)
                         (h4 : week_days = 7)
                         (h5 : final_fish = 26) :
  60 - 2 * d + 8 - 2 * week_days = 26 → d = 14 :=
by {
  sorry
}

end bobbit_worm_days_l209_209567


namespace greatest_integer_third_side_l209_209576

-- Given two sides of a triangle measure 7 cm and 10 cm,
-- we need to prove that the greatest integer number of
-- centimeters that could be the third side is 16 cm.

theorem greatest_integer_third_side (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : 
    ∃ c : ℕ, c < a + b ∧ (∀ d : ℕ, d < a + b → d ≤ c) ∧ c = 16 := 
by
  sorry

end greatest_integer_third_side_l209_209576


namespace find_f_neg3_l209_209861

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x * (1 - x) else -x * (1 + x)

theorem find_f_neg3 :
  is_odd_function f →
  (∀ x, x > 0 → f x = x * (1 - x)) →
  f (-3) = 6 :=
by
  intros h_odd h_condition
  sorry

end find_f_neg3_l209_209861


namespace meeting_point_l209_209053

/-- Along a straight alley with 400 streetlights placed at equal intervals, numbered consecutively from 1 to 400,
    Alla and Boris set out towards each other from opposite ends of the alley with different constant speeds.
    Alla starts at streetlight number 1 and Boris starts at streetlight number 400. When Alla is at the 55th streetlight,
    Boris is at the 321st streetlight. The goal is to prove that they will meet at the 163rd streetlight.
-/
theorem meeting_point (n : ℕ) (h1 : n = 400) (h2 : ∀ i j k l : ℕ, i = 55 → j = 321 → k = 1 → l = 400) : 
  ∃ m, m = 163 := 
by
  sorry

end meeting_point_l209_209053


namespace ladder_base_distance_l209_209925

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l209_209925


namespace sec_150_eq_neg_2_sqrt3_over_3_l209_209801

theorem sec_150_eq_neg_2_sqrt3_over_3 : 
    Real.sec (150 * Real.pi / 180) = - (2 * Real.sqrt 3 / 3) := 
by 
  -- Statement of all conditions used
  have h1 : Real.sec x = 1 / Real.cos x := sorry
  have h2 : Real.cos (150 * Real.pi / 180) = Real.cos (180 * Real.pi / 180 - 30 * Real.pi / 180) := sorry
  have h3 : Real.cos (30 * Real.pi / 180) = Real.sqrt 3 / 2 := sorry
  -- Final goal statement
  sorry

end sec_150_eq_neg_2_sqrt3_over_3_l209_209801


namespace find_k_l209_209107

theorem find_k (x y k : ℝ) (hx1 : x - 4 * y + 3 ≤ 0) (hx2 : 3 * x + 5 * y - 25 ≤ 0) (hx3 : x ≥ 1)
  (hmax : ∃ (z : ℝ), z = 12 ∧ z = k * x + y) (hmin : ∃ (z : ℝ), z = 3 ∧ z = k * x + y) :
  k = 2 :=
sorry

end find_k_l209_209107


namespace distinct_solutions_difference_eq_sqrt29_l209_209494

theorem distinct_solutions_difference_eq_sqrt29 :
  (∃ a b : ℝ, a > b ∧
    (∀ x : ℝ, (5 * x - 20) / (x^2 + 3 * x - 18) = x + 3 ↔ 
      x = a ∨ x = b) ∧ 
    a - b = Real.sqrt 29) :=
sorry

end distinct_solutions_difference_eq_sqrt29_l209_209494


namespace mary_mileage_l209_209020

def base9_to_base10 : Nat :=
  let d0 := 6 * 9^0
  let d1 := 5 * 9^1
  let d2 := 9 * 9^2
  let d3 := 3 * 9^3
  d0 + d1 + d2 + d3 

theorem mary_mileage :
  base9_to_base10 = 2967 :=
by 
  -- Calculation steps are skipped using sorry
  sorry

end mary_mileage_l209_209020


namespace number_of_merchants_l209_209033

theorem number_of_merchants (x : ℕ) (h : 2 * x^3 = 2662) : x = 11 :=
  sorry

end number_of_merchants_l209_209033


namespace max_rectangle_area_l209_209607

theorem max_rectangle_area (P : ℝ) (hP : 0 < P) : 
  ∃ (x y : ℝ), (2*x + 2*y = P) ∧ (x * y = P ^ 2 / 16) :=
by
  sorry

end max_rectangle_area_l209_209607


namespace evaluate_expression_l209_209348

theorem evaluate_expression :
    123 - (45 * (9 - 6) - 78) + (0 / 1994) = 66 :=
by
  sorry

end evaluate_expression_l209_209348


namespace binary_calculation_l209_209491

-- Binary arithmetic definition
def binary_mul (a b : Nat) : Nat := a * b
def binary_div (a b : Nat) : Nat := a / b

-- Binary numbers in Nat (representing binary literals by their decimal equivalent)
def b110010 := 50   -- 110010_2 in decimal
def b101000 := 40   -- 101000_2 in decimal
def b100 := 4       -- 100_2 in decimal
def b10 := 2        -- 10_2 in decimal
def b10111000 := 184-- 10111000_2 in decimal

theorem binary_calculation :
  binary_div (binary_div (binary_mul b110010 b101000) b100) b10 = b10111000 :=
by
  sorry

end binary_calculation_l209_209491


namespace ladder_distance_l209_209932

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l209_209932


namespace john_children_l209_209121

def total_notebooks (john_notebooks : ℕ) (wife_notebooks : ℕ) (children : ℕ) := 
  2 * children + 5 * children

theorem john_children (c : ℕ) (h : total_notebooks 2 5 c = 21) :
  c = 3 :=
sorry

end john_children_l209_209121


namespace product_sequence_equals_8_l209_209335

theorem product_sequence_equals_8 :
  (1 + 1/2) * (1 + 1/3) * (1 + 1/4) * (1 + 1/5) * (1 + 1/6) * (1 + 1/7) = 8 := 
by
  sorry

end product_sequence_equals_8_l209_209335


namespace question1_effective_purification_16days_question2_min_mass_optimal_purification_l209_209760

noncomputable def f (x : ℝ) : ℝ :=
if 0 < x ∧ x ≤ 4 then x^2 / 16 + 2
else if x > 4 then (x + 14) / (2 * x - 2)
else 0

-- Effective Purification Conditions
def effective_purification (m : ℝ) (x : ℝ) : Prop := m * f x ≥ 4

-- Optimal Purification Conditions
def optimal_purification (m : ℝ) (x : ℝ) : Prop := 4 ≤ m * f x ∧ m * f x ≤ 10

-- Proof for Question 1
theorem question1_effective_purification_16days (x : ℝ) (hx : 0 < x ∧ x ≤ 16) :
  effective_purification 4 x :=
by sorry

-- Finding Minimum m for Optimal Purification within 7 days
theorem question2_min_mass_optimal_purification :
  ∃ m : ℝ, (16 / 7 ≤ m ∧ m ≤ 10 / 3) ∧ ∀ (x : ℝ), (0 < x ∧ x ≤ 7) → optimal_purification m x :=
by sorry

end question1_effective_purification_16days_question2_min_mass_optimal_purification_l209_209760


namespace lcm_18_60_is_180_l209_209911

theorem lcm_18_60_is_180 : Nat.lcm 18 60 = 180 := 
  sorry

end lcm_18_60_is_180_l209_209911


namespace relationship_l209_209200

noncomputable def a : ℝ := 3^(-1/3 : ℝ)
noncomputable def b : ℝ := Real.log 3 / Real.log 2⁻¹
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship (a_def : a = 3^(-1/3 : ℝ)) 
                     (b_def : b = Real.log 3 / Real.log 2⁻¹) 
                     (c_def : c = Real.log 3 / Real.log 2) : 
  b < a ∧ a < c :=
  sorry

end relationship_l209_209200


namespace customerPaidPercentGreater_l209_209759

-- Definitions for the conditions
def costOfManufacture (C : ℝ) : ℝ := C
def designerPrice (C : ℝ) : ℝ := C * 1.40
def retailerTaxedPrice (C : ℝ) : ℝ := (C * 1.40) * 1.05
def customerInitialPrice (C : ℝ) : ℝ := ((C * 1.40) * 1.05) * 1.10
def customerFinalPrice (C : ℝ) : ℝ := (((C * 1.40) * 1.05) * 1.10) * 0.90

-- The theorem statement
theorem customerPaidPercentGreater (C : ℝ) (hC : 0 < C) : 
    (customerFinalPrice C - costOfManufacture C) / costOfManufacture C * 100 = 45.53 := by 
  sorry

end customerPaidPercentGreater_l209_209759


namespace louie_took_home_pie_l209_209992

theorem louie_took_home_pie (h_pie_left: ℚ) (h_people: ℕ) (h_equal_split: h_pie_left = 12 / 13 ∧ h_people = 4):
  ∃ (x : ℚ), x = 3 / 13 :=
begin
  rcases h_equal_split with ⟨h_pie_left_eq, h_people_eq⟩,
  use (h_pie_left / h_people),
  have h_pie_left_val : h_pie_left = 12 / 13 := h_pie_left_eq,
  have h_people_val : h_people = 4 := h_people_eq,
  rw [←h_pie_left_val, ←h_people_val],
  norm_num,
  apply rat.ext,
  norm_num,
end

end louie_took_home_pie_l209_209992


namespace train_length_l209_209980

theorem train_length (v_kmph : ℝ) (t_s : ℝ) (L_p : ℝ) (L_t : ℝ) : 
  (v_kmph = 72) ∧ (t_s = 15) ∧ (L_p = 250) →
  L_t = 50 :=
by
  intro h
  sorry

end train_length_l209_209980


namespace extremum_points_of_f_l209_209143

noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then (x + 1)^3 * Real.exp (x + 1) - Real.exp 1
  else -((if -x < 0 then (-x + 1)^3 * Real.exp (-x + 1) - Real.exp 1 else 0))

theorem extremum_points_of_f : ∃! (a b : ℝ), 
  (∀ x < 0, f x = (x + 1)^3 * Real.exp (x + 1) - Real.exp 1) ∧ (f a = f b) ∧ a ≠ b :=
sorry

end extremum_points_of_f_l209_209143


namespace current_year_2021_l209_209475

variables (Y : ℤ)

def parents_moved_to_America := 1982
def Aziz_age := 36
def years_before_born := 3

theorem current_year_2021
  (h1 : parents_moved_to_America = 1982)
  (h2 : Aziz_age = 36)
  (h3 : years_before_born = 3)
  (h4 : Y - (Aziz_age) - (years_before_born) = 1982) : 
  Y = 2021 :=
by {
  sorry
}

end current_year_2021_l209_209475


namespace total_number_of_squares_l209_209885

theorem total_number_of_squares (n : ℕ) (h : n = 12) : 
  ∃ t, t = 17 :=
by
  -- The proof is omitted here
  sorry

end total_number_of_squares_l209_209885


namespace martha_black_butterflies_l209_209403

-- Define the hypotheses
variables (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ)

-- Given conditions
def martha_collection_conditions : Prop :=
  total_butterflies = 19 ∧
  blue_butterflies = 6 ∧
  blue_butterflies = 2 * yellow_butterflies

-- The statement we want to prove
theorem martha_black_butterflies : martha_collection_conditions total_butterflies blue_butterflies yellow_butterflies black_butterflies →
  black_butterflies = 10 :=
sorry

end martha_black_butterflies_l209_209403


namespace column_sum_correct_l209_209192

theorem column_sum_correct : 
  -- Define x to be the sum of the first column (which is also the minuend of the second column)
  ∃ x : ℕ, 
  -- x should match the expected valid sum provided:
  (x = 1001) := 
sorry

end column_sum_correct_l209_209192


namespace simplify_expression_l209_209549
-- Import the entire Mathlib library to ensure all necessary lemmas and theorems are available

-- Define the main problem as a theorem
theorem simplify_expression (t : ℝ) : 
  (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end simplify_expression_l209_209549


namespace melanie_plums_count_l209_209134

theorem melanie_plums_count (dan_plums sally_plums total_plums melanie_plums : ℕ)
    (h1 : dan_plums = 9)
    (h2 : sally_plums = 3)
    (h3 : total_plums = 16)
    (h4 : melanie_plums = total_plums - (dan_plums + sally_plums)) :
    melanie_plums = 4 := by
  -- Proof will be filled here
  sorry

end melanie_plums_count_l209_209134


namespace exponent_equality_l209_209652

theorem exponent_equality (m : ℕ) (h : 9^4 = 3^m) : m = 8 := 
  sorry

end exponent_equality_l209_209652


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209816

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209816


namespace sec_150_eq_l209_209820

theorem sec_150_eq : real.sec (150 * real.pi / 180) = - (2 * real.sqrt 3) / 3 :=
by
  -- We first convert degrees to radians, 150 degrees = 150 * π / 180 radians.
  have h : 150 * real.pi / 180 = 5 * real.pi / 6 := by sorry 
  rw h,
  -- Use the definition of secant.
  -- sec θ = 1 / cos θ
  rw [real.sec, real.cos_pi_div_six],
  -- Cosine of 5π/6 is the negation of cos π/6.
  rw real.cos_arg_neg_pi_div_six,
  -- Evaluate the cos π/6
  have hcos : real.cos (real.pi / 6) = real.sqrt 3 / 2 := real.cos_pi_div_six,
  rw hcos,
  -- Simplify the expression -2/(sqrt(3)) == -2√3/3
  norm_num,
  field_simp,
  norm_num,
  sorry

end sec_150_eq_l209_209820


namespace greatest_area_difference_l209_209292

theorem greatest_area_difference :
  ∃ (l1 w1 l2 w2 : ℕ), 2 * l1 + 2 * w1 = 200 ∧ 2 * l2 + 2 * w2 = 200 ∧
  (l1 * w1 - l2 * w2 = 2401) :=
by
  sorry

end greatest_area_difference_l209_209292


namespace find_n_l209_209843

theorem find_n (n : ℕ) (h : 7^(2*n) = (1/7)^(n-12)) : n = 4 :=
sorry

end find_n_l209_209843


namespace total_matches_l209_209045

noncomputable def matches_in_tournament (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem total_matches :
  matches_in_tournament 5 + matches_in_tournament 7 + matches_in_tournament 4 = 37 := 
by 
  sorry

end total_matches_l209_209045


namespace misha_scored_48_in_second_attempt_l209_209674

theorem misha_scored_48_in_second_attempt (P1 P2 P3 : ℕ)
  (h1 : P2 = 2 * P1)
  (h2 : P3 = (3 / 2) * P2)
  (h3 : 24 ≤ P1)
  (h4 : (3 / 2) * 2 * P1 = 72) : P2 = 48 :=
by sorry

end misha_scored_48_in_second_attempt_l209_209674


namespace probability_two_dice_same_number_l209_209408

theorem probability_two_dice_same_number (n : ℕ) (sides : ℕ) (h_n : n = 8) (h_sides : sides = 6):
  (∃ (prob : ℝ), prob = 1) :=
by
  sorry

end probability_two_dice_same_number_l209_209408


namespace ball_distribution_l209_209668

theorem ball_distribution (N a b : ℕ) (h1 : N = 6912) (h2 : N = 100 * a + b) (h3 : a < 100) (h4 : b < 100) : a + b = 81 :=
by
  sorry

end ball_distribution_l209_209668


namespace simplify_and_evaluate_l209_209681

variable (x y : ℤ)

noncomputable def given_expr := (x + y) ^ 2 - 3 * x * (x + y) + (x + 2 * y) * (x - 2 * y)

theorem simplify_and_evaluate : given_expr 1 (-1) = -3 :=
by
  -- The proof is to be completed here
  sorry

end simplify_and_evaluate_l209_209681


namespace min_of_quadratic_l209_209454

theorem min_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, x^2 + 7 * x + 3 ≤ y^2 + 7 * y + 3) ∧ x = -7 / 2 :=
by
  sorry

end min_of_quadratic_l209_209454


namespace total_soccer_balls_donated_l209_209984

def num_elementary_classes_per_school := 4
def num_middle_classes_per_school := 5
def num_schools := 2
def soccer_balls_per_class := 5

theorem total_soccer_balls_donated : 
  (num_elementary_classes_per_school + num_middle_classes_per_school) * num_schools * soccer_balls_per_class = 90 :=
by
  sorry

end total_soccer_balls_donated_l209_209984


namespace binomial_coefficient_x3y5_in_expansion_l209_209711

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209711


namespace circles_intersect_l209_209364

def C1 (x y : ℝ) : Prop := x^2 + (y-1)^2 = 1
def C2 (x y a : ℝ) : Prop := (x-a)^2 + (y-1)^2 = 16

theorem circles_intersect (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, C1 x y → ∃ x' y' : ℝ, C2 x' y' a) ↔ 3 < a ∧ a < 4 :=
sorry

end circles_intersect_l209_209364


namespace cyclist_speed_l209_209171

theorem cyclist_speed
  (V : ℝ)
  (H1 : ∃ t_p : ℝ, V * t_p = 96 ∧ t_p = (96 / (V - 1)) - 2)
  (H2 : V > 1.25 * (V - 1)) :
  V = 16 :=
by
  sorry

end cyclist_speed_l209_209171


namespace apple_cost_l209_209056

theorem apple_cost (l q : ℕ)
  (h1 : 30 * l + 6 * q = 366)
  (h2 : 15 * l = 150)
  (h3 : 30 * l + (333 - 30 * l) / q * q = 333) :
  30 + (333 - 30 * l) / q = 33 := 
sorry

end apple_cost_l209_209056


namespace original_recipe_pasta_l209_209482

noncomputable def pasta_per_person (total_pasta : ℕ) (total_people : ℕ) : ℚ :=
  total_pasta / total_people

noncomputable def original_pasta (pasta_per_person : ℚ) (people_served : ℕ) : ℚ :=
  pasta_per_person * people_served

theorem original_recipe_pasta (total_pasta : ℕ) (total_people : ℕ) (people_served : ℕ) (required_pasta : ℚ) :
  total_pasta = 10 → total_people = 35 → people_served = 7 → required_pasta = 2 →
  pasta_per_person total_pasta total_people * people_served = required_pasta :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end original_recipe_pasta_l209_209482


namespace feb_03_2013_nine_day_l209_209880

-- Definitions of the main dates involved
def dec_21_2012 : Nat := 0  -- Assuming day 0 is Dec 21, 2012
def feb_03_2013 : Nat := 45  -- 45 days after Dec 21, 2012

-- Definition to determine the Nine-day period
def nine_day_period (x : Nat) : (Nat × Nat) :=
  let q := x / 9
  let r := x % 9
  (q + 1, r + 1)

-- Theorem we want to prove
theorem feb_03_2013_nine_day : nine_day_period feb_03_2013 = (5, 9) :=
by
  sorry

end feb_03_2013_nine_day_l209_209880


namespace value_of_g_g_2_l209_209219

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l209_209219


namespace probability_each_team_loses_and_wins_at_least_one_l209_209346

theorem probability_each_team_loses_and_wins_at_least_one (n : ℕ) (p : ℚ)
  (h1 : n = 8)
  (h2 : p = 1 / 2) : 
  (∃ k : ℚ, k = 903 / 1024 ∧
  k = 1 - (8 * (2^22 - 7 * 2^15) / 2^28)) :=
by {
  use 903 / 1024,
  split;
  sorry
}

end probability_each_team_loses_and_wins_at_least_one_l209_209346


namespace democrats_republicans_circular_arrangement_l209_209163

open Finset

noncomputable def circular_arrangements_no_adjacent (d r : ℕ) : ℕ := 
  (factorial (r - 1)) * choose r d * factorial d

theorem democrats_republicans_circular_arrangement :
  circular_arrangements_no_adjacent 4 6 = 43200 := 
by 
  simp [circular_arrangements_no_adjacent, factorial, choose]
  sorry

end democrats_republicans_circular_arrangement_l209_209163


namespace four_pow_sub_divisible_iff_l209_209139

open Nat

theorem four_pow_sub_divisible_iff (m n k : ℕ) (h₁ : m > n) : 
  (3^(k + 1)) ∣ (4^m - 4^n) ↔ (3^k) ∣ (m - n) := 
by sorry

end four_pow_sub_divisible_iff_l209_209139


namespace power_mod_equiv_l209_209909

theorem power_mod_equiv :
  7 ^ 145 % 12 = 7 % 12 :=
by
  -- Here the solution would go
  sorry

end power_mod_equiv_l209_209909


namespace remainder_of_x_plus_3uy_l209_209026

-- Given conditions
variables (x y u v : ℕ)
variable (Hdiv : x = u * y + v)
variable (H0_le_v : 0 ≤ v)
variable (Hv_lt_y : v < y)

-- Statement to prove
theorem remainder_of_x_plus_3uy (x y u v : ℕ) (Hdiv : x = u * y + v) (H0_le_v : 0 ≤ v) (Hv_lt_y : v < y) :
  (x + 3 * u * y) % y = v :=
sorry

end remainder_of_x_plus_3uy_l209_209026


namespace remainder_of_7_pow_145_mod_12_l209_209908

theorem remainder_of_7_pow_145_mod_12 : (7 ^ 145) % 12 = 7 :=
by
  sorry

end remainder_of_7_pow_145_mod_12_l209_209908


namespace smallest_integer_is_10_l209_209897

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l209_209897


namespace hannah_spent_on_dessert_l209_209376

theorem hannah_spent_on_dessert
  (initial_money : ℕ)
  (money_left : ℕ)
  (half_spent_on_rides : ℕ)
  (total_spent : ℕ)
  (spent_on_dessert : ℕ)
  (H1 : initial_money = 30)
  (H2 : money_left = 10)
  (H3 : half_spent_on_rides = initial_money / 2)
  (H4 : total_spent = initial_money - money_left)
  (H5 : spent_on_dessert = total_spent - half_spent_on_rides) : spent_on_dessert = 5 :=
by
  sorry

end hannah_spent_on_dessert_l209_209376


namespace martha_black_butterflies_l209_209405

theorem martha_black_butterflies (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ) 
    (h1 : total_butterflies = 19)
    (h2 : blue_butterflies = 2 * yellow_butterflies)
    (h3 : blue_butterflies = 6) :
    black_butterflies = 10 :=
by
  -- Prove the theorem assuming the conditions are met
  sorry

end martha_black_butterflies_l209_209405


namespace hyperbola_s_squared_zero_l209_209761

open Real

theorem hyperbola_s_squared_zero :
  ∃ s : ℝ, (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (∀ (x y : ℝ), 
  ((x, y) = (-2, 3) ∨ (x, y) = (0, -1) ∨ (x, y) = (s, 1)) → (y^2 / a^2 - x^2 / b^2 = 1))
  ) → s ^ 2 = 0 :=
by
  sorry

end hyperbola_s_squared_zero_l209_209761


namespace carbon_neutrality_l209_209585

theorem carbon_neutrality (a b : ℝ) (t : ℕ) (ha : a > 0)
  (h1 : S = a * b ^ t)
  (h2 : a * b ^ 7 = 4 * a / 5)
  (h3 : a / 4 = S) :
  t = 42 := 
sorry

end carbon_neutrality_l209_209585


namespace minimum_questions_to_find_number_l209_209294

theorem minimum_questions_to_find_number (n : ℕ) (h : n ≤ 2020) :
  ∃ m, m = 64 ∧ (∀ (strategy : ℕ → ℕ), ∃ questions : ℕ, questions ≤ m ∧ (strategy questions = n)) :=
sorry

end minimum_questions_to_find_number_l209_209294


namespace sale_in_first_month_is_5420_l209_209599

-- Definitions of the sales in months 2 to 6
def sale_month2 : ℕ := 5660
def sale_month3 : ℕ := 6200
def sale_month4 : ℕ := 6350
def sale_month5 : ℕ := 6500
def sale_month6 : ℕ := 6470

-- Definition of the average sale goal
def average_sale_goal : ℕ := 6100

-- Calculating the total needed sales to achieve the average sale goal
def total_required_sales := 6 * average_sale_goal

-- Known sales for months 2 to 6
def known_sales := sale_month2 + sale_month3 + sale_month4 + sale_month5 + sale_month6

-- Definition of the sale in the first month
def sale_month1 := total_required_sales - known_sales

-- The proof statement that the sale in the first month is 5420
theorem sale_in_first_month_is_5420 : sale_month1 = 5420 := by
  sorry

end sale_in_first_month_is_5420_l209_209599


namespace quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l209_209090

/-- Proof that the quadratic equation x^2 - (2k + 1)x + k^2 + k = 0 has two distinct real roots -/
theorem quadratic_eq_two_distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ 
  (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1) :=
by
  sorry

/-- For triangle ΔABC with sides AB, AC as roots of x^2 - (2k + 1)x + k^2 + k = 0 and BC = 4, find k when ΔABC is isosceles -/
theorem isosceles_triangle_value_of_k (k : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 * x2 = k^2 + k ∧ x1 + x2 = 2*k + 1 ∧ 
    ((x1 = 4 ∨ x2 = 4) ∧ (x1 + x2 - 4 isosceles))) →
  (k = 3 ∨ k = 4) :=
by
  sorry

end quadratic_eq_two_distinct_real_roots_isosceles_triangle_value_of_k_l209_209090


namespace total_earnings_from_selling_working_games_l209_209542

-- Conditions definition
def total_games : ℕ := 16
def broken_games : ℕ := 8
def working_games : ℕ := total_games - broken_games
def game_prices : List ℕ := [6, 7, 9, 5, 8, 10, 12, 11]

-- Proof problem statement
theorem total_earnings_from_selling_working_games : List.sum game_prices = 68 := by
  sorry

end total_earnings_from_selling_working_games_l209_209542


namespace initial_leaves_l209_209258

theorem initial_leaves (l_0 : ℕ) (blown_away : ℕ) (leaves_left : ℕ) (h1 : blown_away = 244) (h2 : leaves_left = 112) (h3 : l_0 - blown_away = leaves_left) : l_0 = 356 :=
by
  sorry

end initial_leaves_l209_209258


namespace sum_of_special_primes_l209_209342

theorem sum_of_special_primes : 
  let primes_with_no_solution := {p : ℕ | prime p ∧ ¬ ∃ x : ℤ, 5 * (10 * x + 2) ≡ 3 [MOD p]} in
  ∃ p1 p2, p1 ∈ primes_with_no_solution ∧ p2 ∈ primes_with_no_solution ∧ p1 ≠ p2 ∧ p1 + p2 = 7 :=
by
  sorry

end sum_of_special_primes_l209_209342


namespace distance_from_wall_l209_209972

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l209_209972


namespace coefficient_of_term_in_binomial_expansion_l209_209727

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l209_209727


namespace pure_imaginary_iff_real_part_zero_l209_209037

theorem pure_imaginary_iff_real_part_zero (a b : ℝ) : (∃ z : ℂ, z = a + bi ∧ z.im ≠ 0) ↔ (a = 0 ∧ b ≠ 0) :=
sorry

end pure_imaginary_iff_real_part_zero_l209_209037


namespace modulus_sum_complex_l209_209190

theorem modulus_sum_complex :
  let z1 : Complex := Complex.mk 3 (-8)
  let z2 : Complex := Complex.mk 4 6
  Complex.abs (z1 + z2) = Real.sqrt 53 := by
  sorry

end modulus_sum_complex_l209_209190


namespace sec_150_eq_l209_209808

noncomputable def sec_150 (cos : ℝ → ℝ) (sec : ℝ → ℝ) : ℝ :=
  sec 150

theorem sec_150_eq :
  let cos_30 := (√3) / 2 in
  let cos := λ x, if x = 150 then -cos_30 else sorry in
  let sec := λ x, 1 / cos x in
  sec_150 cos sec = -2 * (√3) / 3 :=
by
  let cos_30 := (√3) / 2
  let cos := λ x, if x = 150 then -cos_30 else sorry
  let sec := λ x, 1 / cos x
  have h_cos_150 : cos 150 = -cos_30, from sorry
  have h_sec_150 : sec 150 = 1 / cos 150, from sorry
  simp [sec_150, cos, sec, h_cos_150, h_sec_150]
  sorry

end sec_150_eq_l209_209808


namespace max_area_difference_l209_209293

theorem max_area_difference (l1 l2 w1 w2 : ℤ) (h1 : 2*l1 + 2*w1 = 200) (h2 : 2*l2 + 2*w2 = 200) :
  let A := λ l w, l * w in
  (max {A l w | l + w = 100} - min {A l w | l + w = 100}) = 2401 :=
sorry

end max_area_difference_l209_209293


namespace medal_allocation_l209_209853

-- Define the participants
inductive Participant
| Jiri
| Vit
| Ota

open Participant

-- Define the medals
inductive Medal
| Gold
| Silver
| Bronze

open Medal

-- Define a structure to capture each person's statement
structure Statements :=
  (Jiri : Prop)
  (Vit : Prop)
  (Ota : Prop)

-- Define the condition based on their statements
def statements (m : Participant → Medal) : Statements :=
  {
    Jiri := m Ota = Gold,
    Vit := m Ota = Silver,
    Ota := (m Ota ≠ Gold ∧ m Ota ≠ Silver)
  }

-- Define the condition for truth-telling and lying based on medals
def truths_and_lies (m : Participant → Medal) (s : Statements) : Prop :=
  (m Jiri = Gold → s.Jiri) ∧ (m Jiri = Bronze → ¬ s.Jiri) ∧
  (m Vit = Gold → s.Vit) ∧ (m Vit = Bronze → ¬ s.Vit) ∧
  (m Ota = Gold → s.Ota) ∧ (m Ota = Bronze → ¬ s.Ota)

-- Define the final theorem to be proven
theorem medal_allocation : 
  ∃ (m : Participant → Medal), 
    truths_and_lies m (statements m) ∧ 
    m Vit = Gold ∧ 
    m Ota = Silver ∧ 
    m Jiri = Bronze := 
sorry

end medal_allocation_l209_209853


namespace sqrt_pi_decimal_expansion_l209_209338

-- Statement of the problem: Compute the first 23 digits of the decimal expansion of sqrt(pi)
theorem sqrt_pi_decimal_expansion : 
  ( ∀ n, n ≤ 22 → 
    (digits : List ℕ) = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, d15, d16, d17, d18, d19, d20, d21, d22, d23] →
      (d1 = 1 ∧ d2 = 7 ∧ d3 = 7 ∧ d4 = 2 ∧ d5 = 4 ∧ d6 = 5 ∧ d7 = 3 ∧ d8 = 8 ∧ d9 = 5 ∧ d10 = 0 ∧ d11 = 9 ∧ d12 = 0 ∧ d13 = 5 ∧ d14 = 5 ∧ d15 = 1 ∧ d16 = 6 ∧ d17 = 0 ∧ d18 = 2 ∧ d19 = 7 ∧ d20 = 2 ∧ d21 = 9 ∧ d22 = 8 ∧ d23 = 1)) → 
  True :=
by
  sorry
  -- Actual proof to be filled, this is just the statement showing that we expected the digits 
  -- of the decimal expansion of sqrt(pi) match the specified values up to the 23rd place.

end sqrt_pi_decimal_expansion_l209_209338


namespace range_of_m_l209_209637

theorem range_of_m (x y m : ℝ) (hx : x > 0) (hy : y > 0) (h : (2 / x) + (1 / y) = 1) (h2 : x + 2 * y > m^2 + 2 * m) : -4 < m ∧ m < 2 :=
by
  sorry

end range_of_m_l209_209637


namespace find_monotonic_function_l209_209193

-- Define Jensen's functional equation property
def jensens_eq (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ) (t : ℝ), 0 ≤ t ∧ t ≤ 1 → f (t * x + (1 - t) * y) = t * f x + (1 - t) * f y

-- Define monotonicity property
def monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

-- The main theorem stating the equivalence
theorem find_monotonic_function (f : ℝ → ℝ) (h₁ : jensens_eq f) (h₂ : monotonic f) : 
  ∃ a b : ℝ, ∀ x : ℝ, f x = a * x + b := 
sorry

end find_monotonic_function_l209_209193


namespace distance_from_wall_l209_209971

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l209_209971


namespace total_cows_in_ranch_l209_209299

theorem total_cows_in_ranch :
  ∀ (WTP_cows : ℕ) (HGHF_cows : ℕ), WTP_cows = 17 → HGHF_cows = 3 * WTP_cows + 2 → (HGHF_cows + WTP_cows) = 70 :=
by 
  intros WTP_cows HGHF_cows WTP_cows_def HGHF_cows_def
  rw [WTP_cows_def, HGHF_cows_def]
  sorry

end total_cows_in_ranch_l209_209299


namespace total_fireworks_l209_209986

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l209_209986


namespace common_difference_of_arithmetic_sequence_l209_209698

variable (S_n : ℕ → ℝ) (a_n : ℕ → ℝ) (d : ℝ)
variable (h₁ : S_n 5 = -15) (h₂ : a_n 2 + a_n 5 = -2)

theorem common_difference_of_arithmetic_sequence :
  d = 4 :=
by
  sorry

end common_difference_of_arithmetic_sequence_l209_209698


namespace four_digit_number_exists_l209_209160

theorem four_digit_number_exists :
  ∃ (A B C D : ℕ), A = B / 3 ∧ C = A + B ∧ D = 3 * B ∧
  A ≠ 0 ∧ A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (A * 1000 + B * 100 + C * 10 + D = 1349) :=
by
  sorry

end four_digit_number_exists_l209_209160


namespace geometric_progression_nonzero_k_l209_209493

theorem geometric_progression_nonzero_k (k : ℝ) : k ≠ 0 ↔ (40*k)^2 = (10*k) * (160*k) := by sorry

end geometric_progression_nonzero_k_l209_209493


namespace distance_Owlford_Highcastle_l209_209058

open Complex

theorem distance_Owlford_Highcastle :
  let Highcastle := (0 : ℂ)
  let Owlford := (900 + 1200 * I : ℂ)
  dist Highcastle Owlford = 1500 := by
  sorry

end distance_Owlford_Highcastle_l209_209058


namespace total_pencils_crayons_l209_209078

theorem total_pencils_crayons (r : ℕ) (p : ℕ) (c : ℕ) 
  (hp : p = 31) (hc : c = 27) (hr : r = 11) : 
  r * p + r * c = 638 := 
  by
  sorry

end total_pencils_crayons_l209_209078


namespace binomial_defective_products_l209_209054

-- Define the conditions
def total_products : ℕ := 100
def defective_products : ℕ := 5
def selection_count : ℕ := 10
def p_defective : ℝ := defective_products / total_products

-- Define the random variable X
def X : ProbDistrib ℝ := Distrib.binomial selection_count p_defective

-- State the theorem
theorem binomial_defective_products :
  X = Distrib.binomial selection_count p_defective :=
by sorry

end binomial_defective_products_l209_209054


namespace total_fireworks_l209_209987

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end total_fireworks_l209_209987


namespace coeff_of_x_pow_4_in_expansion_l209_209273

theorem coeff_of_x_pow_4_in_expansion : 
  (∃ c : ℤ, c = (-1)^3 * Nat.choose 8 3 ∧ c = -56) :=
by
  sorry

end coeff_of_x_pow_4_in_expansion_l209_209273


namespace triangle_sides_square_perfect_l209_209179

theorem triangle_sides_square_perfect (x y z : ℕ) (h : ∃ h_x h_y h_z, 
  h_x = h_y + h_z ∧ 
  2 * h_x * x = 2 * h_y * y ∧ 
  2 * h_x * x = 2 * h_z * z ) :
  ∃ k : ℕ, x^2 + y^2 + z^2 = k^2 :=
by
  sorry

end triangle_sides_square_perfect_l209_209179


namespace sum_of_three_integers_with_product_of_5_cubed_l209_209432

theorem sum_of_three_integers_with_product_of_5_cubed :
  ∃ (a b c : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ 
  a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 
  a * b * c = 5^3 ∧ 
  a + b + c = 31 :=
by
  sorry

end sum_of_three_integers_with_product_of_5_cubed_l209_209432


namespace davids_profit_l209_209768

-- Definitions of conditions
def weight_of_rice : ℝ := 50
def cost_of_rice : ℝ := 50
def selling_price_per_kg : ℝ := 1.20

-- Theorem stating the expected profit
theorem davids_profit : 
  (selling_price_per_kg * weight_of_rice) - cost_of_rice = 10 := 
by 
  -- Proofs are omitted.
  sorry

end davids_profit_l209_209768


namespace D_won_zero_matches_l209_209232

-- Define the players
inductive Player
| A | B | C | D deriving DecidableEq

-- Function to determine the winner of a match
def match_winner (p1 p2 : Player) : Option Player :=
  if p1 = Player.A ∧ p2 = Player.D then 
    some Player.A
  else if p2 = Player.A ∧ p1 = Player.D then 
    some Player.A
  else 
    none -- This represents that we do not know the outcome for matches not given

-- Assuming A, B, and C have won the same number of matches
def same_wins (w_A w_B w_C : Nat) : Prop := 
  w_A = w_B ∧ w_B = w_C

-- Define the problem statement
theorem D_won_zero_matches (w_D : Nat) (h_winner_AD: match_winner Player.A Player.D = some Player.A)
  (h_same_wins : ∃ w_A w_B w_C : Nat, same_wins w_A w_B w_C) : w_D = 0 :=
sorry

end D_won_zero_matches_l209_209232


namespace solve_for_b_l209_209379

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l209_209379


namespace average_temperature_l209_209051

theorem average_temperature (temps : List ℕ) (temps_eq : temps = [40, 47, 45, 41, 39]) :
  (temps.sum : ℚ) / temps.length = 42.4 :=
by
  sorry

end average_temperature_l209_209051


namespace parabola_slopes_l209_209360

theorem parabola_slopes (k : ℝ) (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hC : C = (0, -2)) (hA : A.1^2 = 2 * A.2) (hB : B.1^2 = 2 * B.2) 
    (hA_eq : A.2 = k * A.1 + 2) (hB_eq : B.2 = k * B.1 + 2) :
  ((C.2 - A.2) / (C.1 - A.1))^2 + ((C.2 - B.2) / (C.1 - B.1))^2 - 2 * k^2 = 8 := 
sorry

end parabola_slopes_l209_209360


namespace minimum_value_Q_l209_209535

theorem minimum_value_Q (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : (a / b + b / c + c / a) + (b / a + c / b + a / c) = 10) :
  (a / b + b / c + c / a) * (b / a + c / b + a / c) = 47 := 
  sorry

end minimum_value_Q_l209_209535


namespace coins_ratio_l209_209620

-- Conditions
def initial_coins : Nat := 125
def gift_coins : Nat := 35
def sold_coins : Nat := 80

-- Total coins after receiving the gift
def total_coins := initial_coins + gift_coins

-- Statement to prove the ratio simplifies to 1:2
theorem coins_ratio : (sold_coins : ℚ) / total_coins = 1 / 2 := by
  sorry

end coins_ratio_l209_209620


namespace johns_out_of_pocket_expense_l209_209529

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l209_209529


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209791

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209791


namespace repeating_decimal_product_l209_209191

-- Define the repeating decimal 0.\overline{137} as a fraction
def repeating_decimal_137 : ℚ := 137 / 999

-- Define the repeating decimal 0.\overline{6} as a fraction
def repeating_decimal_6 : ℚ := 2 / 3

-- The problem is to prove that the product of these fractions is 274 / 2997
theorem repeating_decimal_product : repeating_decimal_137 * repeating_decimal_6 = 274 / 2997 := by
  sorry

end repeating_decimal_product_l209_209191


namespace m_not_in_P_l209_209252

noncomputable def m : ℝ := Real.sqrt 3
def P : Set ℝ := { x | x^2 - Real.sqrt 2 * x ≤ 0 }

theorem m_not_in_P : m ∉ P := by
  sorry

end m_not_in_P_l209_209252


namespace shelves_used_l209_209769

def initial_books : Nat := 87
def sold_books : Nat := 33
def books_per_shelf : Nat := 6

theorem shelves_used :
  (initial_books - sold_books) / books_per_shelf = 9 := by
  sorry

end shelves_used_l209_209769


namespace num_factors_2012_l209_209378

theorem num_factors_2012 : (Nat.factors 2012).length = 6 := by
  sorry

end num_factors_2012_l209_209378


namespace trigonometric_identity_l209_209832

theorem trigonometric_identity
  (x : ℝ)
  (h_tan : Real.tan x = -1/2) :
  Real.sin x ^ 2 + 3 * Real.sin x * Real.cos x - 1 = -2 :=
sorry

end trigonometric_identity_l209_209832


namespace smallest_possible_x_l209_209914

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l209_209914


namespace cannot_finish_third_l209_209550

variable (P Q R S T U : ℕ)
variable (beats : ℕ → ℕ → Prop)
variable (finishes_after : ℕ → ℕ → Prop)
variable (finishes_before : ℕ → ℕ → Prop)

noncomputable def race_conditions (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) : Prop :=
  beats P Q ∧
  beats P R ∧
  beats Q S ∧
  finishes_after T P ∧
  finishes_before T Q ∧
  finishes_after U R ∧
  beats U T

theorem cannot_finish_third (P Q R S T U : ℕ) (beats finishes_after finishes_before : ℕ → ℕ → Prop) :
  race_conditions P Q R S T U beats finishes_after finishes_before →
  ¬ (finishes_before P T ∧ finishes_before T S ∧ finishes_after P R ∧ finishes_after P S) ∧ ¬ (finishes_before S T ∧ finishes_before T P) :=
sorry

end cannot_finish_third_l209_209550


namespace total_investment_with_interest_l209_209295

theorem total_investment_with_interest
  (total_investment : ℝ)
  (amount_at_3_percent : ℝ)
  (interest_rate_3 : ℝ)
  (interest_rate_5 : ℝ)
  (remaining_amount : ℝ)
  (interest_3 : ℝ)
  (interest_5 : ℝ) :
  total_investment = 1000 →
  amount_at_3_percent = 199.99999999999983 →
  interest_rate_3 = 0.03 →
  interest_rate_5 = 0.05 →
  remaining_amount = total_investment - amount_at_3_percent →
  interest_3 = amount_at_3_percent * interest_rate_3 →
  interest_5 = remaining_amount * interest_rate_5 →
  total_investment + interest_3 + remaining_amount + interest_5 = 1046 :=
by
  intros H1 H2 H3 H4 H5 H6 H7
  sorry

end total_investment_with_interest_l209_209295


namespace difference_between_two_numbers_l209_209890

theorem difference_between_two_numbers : 
  ∃ (a b : ℕ),
    (a + b = 21780) ∧
    (a % 5 = 0) ∧
    ((a / 10) = b) ∧
    (a - b = 17825) :=
sorry

end difference_between_two_numbers_l209_209890


namespace fraction_transformation_half_l209_209844

theorem fraction_transformation_half (a b : ℝ) (a_ne_zero : a ≠ 0) (b_ne_zero : b ≠ 0) :
  ((2 * a + 2 * b) / (4 * a^2 + 4 * b^2)) = (1 / 2) * ((a + b) / (a^2 + b^2)) :=
by sorry

end fraction_transformation_half_l209_209844


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209790

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209790


namespace proof_d_e_f_value_l209_209366

theorem proof_d_e_f_value
  (a b c d e f : ℝ)
  (h1 : a * b * c = 195)
  (h2 : b * c * d = 65)
  (h3 : c * d * e = 1000)
  (h4 : (a * f) / (c * d) = 0.75) :
  d * e * f = 250 :=
sorry

end proof_d_e_f_value_l209_209366


namespace five_digit_integers_count_l209_209377
open BigOperators

noncomputable def permutations_with_repetition (n : ℕ) (reps : List ℕ) : ℕ :=
  n.factorial / ((reps.map (λ x => x.factorial)).prod)

theorem five_digit_integers_count :
  permutations_with_repetition 5 [2, 2] = 30 :=
by
  sorry

end five_digit_integers_count_l209_209377


namespace watermelon_cost_is_100_rubles_l209_209978

theorem watermelon_cost_is_100_rubles :
  (∀ (x y k m n : ℕ) (a b : ℝ),
    x + y = k →
    n * a = m * b →
    n * a + m * b = 24000 →
    n = 120 →
    m = 30 →
    k = 150 →
    a = 100) :=
by
  intros x y k m n a b
  intros h1 h2 h3 h4 h5 h6
  have h7 : 120 * a = 30 * b, from h2
  have h8 : 120 * a + 30 * b = 24000, from h3
  have h9 : 120 * a = 12000, from sorry
  have h10 : a = 100, from sorry
  exact h10

end watermelon_cost_is_100_rubles_l209_209978


namespace fractional_inspection_l209_209856

theorem fractional_inspection:
  ∃ (J E A : ℝ),
  J + E + A = 1 ∧
  0.005 * J + 0.007 * E + 0.012 * A = 0.01 :=
by
  sorry

end fractional_inspection_l209_209856


namespace expression_evaluation_l209_209981

theorem expression_evaluation : 
  (1 : ℝ)^(6 * z - 3) / (7⁻¹ + 4⁻¹) = 28 / 11 :=
by
  sorry

end expression_evaluation_l209_209981


namespace charge_difference_is_51_l209_209634

-- Define the charges and calculations for print shop X
def print_shop_x_cost (n : ℕ) : ℝ :=
  if n ≤ 50 then n * 1.20 else 50 * 1.20 + (n - 50) * 0.90

-- Define the charges and calculations for print shop Y
def print_shop_y_cost (n : ℕ) : ℝ :=
  10 + n * 1.70

-- Define the difference in charges for 70 copies
def charge_difference : ℝ :=
  print_shop_y_cost 70 - print_shop_x_cost 70

-- The proof statement
theorem charge_difference_is_51 : charge_difference = 51 :=
by
  sorry

end charge_difference_is_51_l209_209634


namespace inequality_non_empty_solution_set_l209_209887

theorem inequality_non_empty_solution_set (a : ℝ) : ∃ x : ℝ, ax^2 - (a-2)*x - 2 ≤ 0 :=
sorry

end inequality_non_empty_solution_set_l209_209887


namespace total_banana_produce_correct_l209_209589

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l209_209589


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209797

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209797


namespace ladder_base_distance_l209_209955

theorem ladder_base_distance (h c₁ c₂ : ℝ) (lwall : c₁ = 12) (lladder : h = 13) :
  ∃ b : ℝ, b = 5 := by
  sorry

end ladder_base_distance_l209_209955


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209796

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209796


namespace roots_of_quadratic_l209_209434

theorem roots_of_quadratic (x : ℝ) : (x * (x - 2) = 2 - x) ↔ (x = 2 ∨ x = -1) :=
by
  sorry

end roots_of_quadratic_l209_209434


namespace number_of_possible_schedules_l209_209877

-- Define the six teams
inductive Team : Type
| A | B | C | D | E | F

open Team

-- Define the function to get the number of different schedules possible
noncomputable def number_of_schedules : ℕ := 70

-- Define the theorem statement
theorem number_of_possible_schedules (teams : Finset Team) (play_games : Team → Finset Team) (h : teams.card = 6) 
  (h2 : ∀ t ∈ teams, (play_games t).card = 3 ∧ ∀ t' ∈ (play_games t), t ≠ t') : 
  number_of_schedules = 70 :=
by sorry

end number_of_possible_schedules_l209_209877


namespace sector_angle_l209_209433

theorem sector_angle (r L : ℝ) (h1 : r = 1) (h2 : L = 4) : abs (L - 2 * r) = 2 :=
by 
  -- This is the statement of our proof problem
  -- and does not include the proof itself.
  sorry

end sector_angle_l209_209433


namespace strictly_increasing_0_to_e_l209_209340

noncomputable def ln (x : ℝ) : ℝ := Real.log x

noncomputable def f (x : ℝ) : ℝ := ln x / x

theorem strictly_increasing_0_to_e :
  ∀ x : ℝ, 0 < x ∧ x < Real.exp 1 → 0 < (1 - ln x) / (x^2) :=
by
  sorry

end strictly_increasing_0_to_e_l209_209340


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209793

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209793


namespace calculate_fraction_l209_209478

theorem calculate_fraction : (10^20 / 50^10) = 2^10 := by
  sorry

end calculate_fraction_l209_209478


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209802

open Real

theorem sec_150_eq_neg_two_sqrt_three_over_three :
  sec (150 * pi / 180) = - (2 * sqrt 3 / 3) :=
by
  -- definitions
  have h1: sec (x:ℝ) = 1 / cos x := sec_eq_inverse_cos x
  have h2: cos (150 * pi / 180) = - cos (30 * pi / 180) := by sorry
  have h3: cos (30 * pi / 180) = sqrt 3 / 2 := by sorry
  -- the actual proof
  sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209802


namespace students_received_B_l209_209385

/-!
# Problem Statement

Given:
1. In Mr. Johnson's class, 18 out of 30 students received a B.
2. Ms. Smith has 45 students in total, and the ratio of students receiving a B is the same as in Mr. Johnson's class.
Prove:
27 students in Ms. Smith's class received a B.
-/

theorem students_received_B (s1 s2 b1 : ℕ) (r1 : ℚ) (r2 : ℕ) (h₁ : s1 = 30) (h₂ : b1 = 18) (h₃ : s2 = 45) (h₄ : r1 = 3/5) 
(H : (b1 : ℚ) / s1 = r1) : r2 = 27 :=
by
  -- Conditions provided
  -- h₁ : s1 = 30
  -- h₂ : b1 = 18
  -- h₃ : s2 = 45
  -- h₄ : r1 = 3/5
  -- H : (b1 : ℚ) / s1 = r1
  sorry

end students_received_B_l209_209385


namespace range_of_k_roots_for_neg_k_l209_209142

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x ≠ y ∧ (x^2 + (2*k + 1)*x + (k^2 - 1) = 0 ∧ y^2 + (2*k + 1)*y + (k^2 - 1) = 0)) ↔ k > -5 / 4 :=
by sorry

theorem roots_for_neg_k (k : ℤ) (h1 : k < 0) (h2 : k > -5 / 4) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + (2*k + 1)*x1 + (k^2 - 1) = 0 ∧ x2^2 + (2*k + 1)*x2 + (k^2 - 1) = 0 ∧ x1 = 0 ∧ x2 = 1)) :=
by sorry

end range_of_k_roots_for_neg_k_l209_209142


namespace share_sheets_equally_l209_209241

theorem share_sheets_equally (sheets friends : ℕ) (h_sheets : sheets = 15) (h_friends : friends = 3) : sheets / friends = 5 := by
  sorry

end share_sheets_equally_l209_209241


namespace repeating_block_length_five_sevenths_l209_209449

theorem repeating_block_length_five_sevenths : 
  ∃ n : ℕ, (∃ k : ℕ, (5 * 10^k - 5) % 7 = 0) ∧ n = 6 :=
sorry

end repeating_block_length_five_sevenths_l209_209449


namespace custom_op_4_3_l209_209446

-- Define the custom operation a * b
def custom_op (a b : ℤ) : ℤ := a^2 + a * b - b^2

-- State the theorem to be proven
theorem custom_op_4_3 : custom_op 4 3 = 19 := 
by
sorry

end custom_op_4_3_l209_209446


namespace find_a_l209_209374

-- Definitions for conditions
def line_equation (a : ℝ) (x y : ℝ) := a * x - y - 1 = 0
def angle_of_inclination (θ : ℝ) := θ = Real.pi / 3

-- The main theorem statement
theorem find_a (a : ℝ) (θ : ℝ) (h1 : angle_of_inclination θ) (h2 : a = Real.tan θ) : a = Real.sqrt 3 :=
 by
   -- skipping the proof
   sorry

end find_a_l209_209374


namespace two_largest_divisors_difference_l209_209181

theorem two_largest_divisors_difference (N : ℕ) (h : N > 1) (a : ℕ) (ha : a ∣ N) (h6a : 6 * a ∣ N) :
  (N / 2 : ℚ) / (N / 3 : ℚ) = 1.5 := by
  sorry

end two_largest_divisors_difference_l209_209181


namespace proposal_spreading_problem_l209_209901

theorem proposal_spreading_problem (n : ℕ) : 1 + n + n^2 = 1641 := 
sorry

end proposal_spreading_problem_l209_209901


namespace slope_of_line_l209_209700

-- Definitions of the conditions in the problem
def line_eq (a : ℝ) (x y : ℝ) : Prop := x + a * y + 1 = 0

def y_intercept (l : ℝ → ℝ → Prop) (b : ℝ) : Prop :=
  l 0 b

-- The statement of the proof problem
theorem slope_of_line (a : ℝ) (h : y_intercept (line_eq a) (-2)) : 
  ∃ (m : ℝ), m = -2 :=
sorry

end slope_of_line_l209_209700


namespace quad_eq_sum_ab_l209_209645

theorem quad_eq_sum_ab {a b : ℝ} (h1 : a < 0)
  (h2 : ∀ x : ℝ, (x = -1 / 2 ∨ x = 1 / 3) ↔ ax^2 + bx + 2 = 0) :
  a + b = -14 :=
by
  sorry

end quad_eq_sum_ab_l209_209645


namespace product_identity_l209_209779

variable (x y : ℝ)

theorem product_identity :
  (3 * x^2 - 4 * y^3) * (9 * x^4 + 12 * x^2 * y^3 + 16 * y^6) = 27 * x^6 - 64 * y^9 :=
by
  sorry

end product_identity_l209_209779


namespace length_of_stone_slab_l209_209752

theorem length_of_stone_slab 
  (num_slabs : ℕ) 
  (total_area : ℝ) 
  (h_num_slabs : num_slabs = 30) 
  (h_total_area : total_area = 50.7): 
  ∃ l : ℝ, l = 1.3 ∧ l * l * num_slabs = total_area := 
by 
  sorry

end length_of_stone_slab_l209_209752


namespace find_number_l209_209520

theorem find_number (x : ℝ) (h : x - x / 3 = x - 24) : x = 72 := 
by 
  sorry

end find_number_l209_209520


namespace smallest_solution_l209_209828

def equation (x : ℝ) := (3 * x) / (x - 3) + (3 * x^2 - 27) / x = 14

theorem smallest_solution :
  ∀ x : ℝ, equation x → x = (3 - Real.sqrt 333) / 6 :=
sorry

end smallest_solution_l209_209828


namespace eighth_term_l209_209436

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ := (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ} {d : ℤ}

-- Conditions
axiom sum_of_first_n_terms : ∀ n : ℕ, S n a = (n * (a 1 + a n)) / 2
axiom second_term : a 2 = 3
axiom sum_of_first_five_terms : S 5 a = 25

-- Question
theorem eighth_term : a 8 = 15 :=
sorry

end eighth_term_l209_209436


namespace common_ratio_geometric_series_l209_209146

theorem common_ratio_geometric_series {a r S : ℝ} (h₁ : S = (a / (1 - r))) (h₂ : (ar^4 / (1 - r)) = S / 64) (h₃ : S ≠ 0) : r = 1 / 2 :=
sorry

end common_ratio_geometric_series_l209_209146


namespace find_k_l209_209358

noncomputable section

variables {a b k : ℝ}

theorem find_k 
  (h1 : 4^a = k) 
  (h2 : 9^b = k)
  (h3 : 1 / a + 1 / b = 2) : 
  k = 6 :=
sorry

end find_k_l209_209358


namespace martha_black_butterflies_l209_209401

theorem martha_black_butterflies
    (total_butterflies : ℕ)
    (total_blue_butterflies : ℕ)
    (total_yellow_butterflies : ℕ)
    (total_black_butterflies : ℕ)
    (h1 : total_butterflies = 19)
    (h2 : total_blue_butterflies = 6)
    (h3 : total_blue_butterflies = 2 * total_yellow_butterflies)
    (h4 : total_black_butterflies = total_butterflies - (total_blue_butterflies + total_yellow_butterflies))
    : total_black_butterflies = 10 :=
  sorry

end martha_black_butterflies_l209_209401


namespace min_value_expression_l209_209504

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) : 
  (x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) >= 1 / 4) ∧ (x = 1/3 ∧ y = 1/3 ∧ z = 1/3 → x^2 / (1 + x) + y^2 / (1 + y) + z^2 / (1 + z) = 1 / 4) :=
sorry

end min_value_expression_l209_209504


namespace continuous_of_compact_and_connected_image_l209_209123

open Set Filter Topology

variable {n m : ℕ} (f : ℝ^n → ℝ^m)

theorem continuous_of_compact_and_connected_image (h1 : ∀ K : Set ℝ^n, IsCompact K → IsCompact (f '' K))
  (h2 : ∀ C : Set ℝ^n, IsConnected C → IsConnected (f '' C)) :
  Continuous f :=
begin
  sorry
end

end continuous_of_compact_and_connected_image_l209_209123


namespace chef_michel_total_pies_l209_209071

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l209_209071


namespace line_through_point_parallel_to_line_l209_209688

theorem line_through_point_parallel_to_line {x y : ℝ} 
  (point : x = 1 ∧ y = 0) 
  (parallel_line : ∃ c : ℝ, ∀ x y : ℝ, x - 2 * y + c = 0) :
  x - 2 * y - 1 = 0 := 
by
  sorry

end line_through_point_parallel_to_line_l209_209688


namespace sec_150_eq_neg_two_sqrt_three_over_three_l209_209811

-- Definitions to match the problem conditions
noncomputable def sec (θ : ℝ) : ℝ := 1 / Real.cos θ

-- Stating the proposition to be proved
theorem sec_150_eq_neg_two_sqrt_three_over_three : sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
sorry

end sec_150_eq_neg_two_sqrt_three_over_three_l209_209811


namespace student_19_in_sample_l209_209771

-- Definitions based on conditions
def total_students := 52
def sample_size := 4
def sampling_interval := 13

def selected_students := [6, 32, 45]

-- The theorem to prove
theorem student_19_in_sample : 19 ∈ selected_students ∨ ∃ k : ℕ, 13 * k + 6 = 19 :=
by
  sorry

end student_19_in_sample_l209_209771


namespace total_value_of_coins_l209_209704

theorem total_value_of_coins (q d : ℕ) (total_value original_value swapped_value : ℚ)
  (h1 : q + d = 30)
  (h2 : total_value = 4.50)
  (h3 : original_value = 25 * q + 10 * d)
  (h4 : swapped_value = 10 * q + 25 * d)
  (h5 : swapped_value = original_value + 1.50) :
  total_value = original_value / 100 :=
sorry

end total_value_of_coins_l209_209704


namespace train_length_l209_209604

theorem train_length (L : ℝ) (v : ℝ)
  (h1 : L = v * 36)
  (h2 : L + 25 = v * 39) :
  L = 300 :=
by
  sorry

end train_length_l209_209604


namespace Anita_should_buy_more_cartons_l209_209055

def Anita_needs (total_needed : ℕ) : Prop :=
total_needed = 26

def Anita_has (strawberries blueberries : ℕ) : Prop :=
strawberries = 10 ∧ blueberries = 9

def additional_cartons (total_needed strawberries blueberries : ℕ) : ℕ :=
total_needed - (strawberries + blueberries)

theorem Anita_should_buy_more_cartons :
  ∀ (total_needed strawberries blueberries : ℕ),
    Anita_needs total_needed →
    Anita_has strawberries blueberries →
    additional_cartons total_needed strawberries blueberries = 7 :=
by
  intros total_needed strawberries blueberries Hneeds Hhas
  sorry

end Anita_should_buy_more_cartons_l209_209055


namespace no_such_abc_exists_l209_209429

theorem no_such_abc_exists :
  ¬ ∃ (a b c : ℝ), 
      ((a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0) ∨
       (a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∧ c < 0 ∧ a > 0)) ∧
      ((a < 0 ∧ b < 0 ∧ c > 0) ∨ (a < 0 ∧ c < 0 ∧ b > 0) ∨ (b < 0 ∨ c < 0 ∧ a > 0) ∨
       (a > 0 ∧ b > 0 ∧ c < 0) ∨ (a > 0 ∧ c > 0 ∧ b < 0) ∨ (b > 0 ∧ c > 0 ∧ a < 0)) :=
by {
  sorry
}

end no_such_abc_exists_l209_209429


namespace point_P_below_line_l209_209510

def line_equation (x y : ℝ) : ℝ := 2 * x - y + 3

def point_below_line (P : ℝ × ℝ) : Prop :=
  let (x, y) := P
  2 * x - y + 3 > 0

theorem point_P_below_line :
  point_below_line (1, -1) :=
by
  sorry

end point_P_below_line_l209_209510


namespace ladder_distance_from_wall_l209_209942

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l209_209942


namespace find_sum_l209_209145

theorem find_sum (a b : ℝ) (ha : a^3 - 3 * a^2 + 5 * a - 17 = 0) (hb : b^3 - 3 * b^2 + 5 * b + 11 = 0) :
  a + b = 2 :=
sorry

end find_sum_l209_209145


namespace coefficient_x3y5_in_expansion_l209_209709

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l209_209709


namespace football_game_initial_population_l209_209290

theorem football_game_initial_population (B G : ℕ) (h1 : G = 240)
  (h2 : (3 / 4 : ℚ) * B + (7 / 8 : ℚ) * G = 480) : B + G = 600 :=
sorry

end football_game_initial_population_l209_209290


namespace largest_base6_five_digit_l209_209739

def base6_to_base10 : ℕ → ℕ
| n :=
  let digits := [5, 5, 5, 5, 5] in
  digits.zipWith (λ digit idx, digit * (6 ^ idx)) [4, 3, 2, 1, 0]
  |> List.sum

theorem largest_base6_five_digit : base6_to_base10 55555 = 7775 :=
by
  sorry

end largest_base6_five_digit_l209_209739


namespace coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209707

theorem coefficient_x3y5_in_expansion_of_x_plus_y_8 :
  (Nat.choose 8 3) = 56 := 
by
  sorry

end coefficient_x3y5_in_expansion_of_x_plus_y_8_l209_209707


namespace find_f_21_l209_209359

def f : ℝ → ℝ := sorry

lemma f_condition (x : ℝ) : f (2 / x + 1) = Real.log x := sorry

theorem find_f_21 : f 21 = -1 := sorry

end find_f_21_l209_209359


namespace martha_black_butterflies_l209_209402

-- Define the hypotheses
variables (total_butterflies blue_butterflies yellow_butterflies black_butterflies : ℕ)

-- Given conditions
def martha_collection_conditions : Prop :=
  total_butterflies = 19 ∧
  blue_butterflies = 6 ∧
  blue_butterflies = 2 * yellow_butterflies

-- The statement we want to prove
theorem martha_black_butterflies : martha_collection_conditions total_butterflies blue_butterflies yellow_butterflies black_butterflies →
  black_butterflies = 10 :=
sorry

end martha_black_butterflies_l209_209402


namespace coefficient_of_x3_y5_in_binomial_expansion_l209_209715

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l209_209715


namespace coefficient_x3_y5_in_binomial_expansion_l209_209733

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l209_209733


namespace extreme_points_inequality_l209_209209

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1 / 2) * x^2 + a * Real.log (1 - x)

theorem extreme_points_inequality (a x1 x2 : ℝ) (h_a : 0 < a ∧ a < 1 / 4) 
  (h_sum : x1 + x2 = 1) (h_prod : x1 * x2 = a) (h_order : x1 < x2) :
  f x2 a - x1 > -(3 + Real.log 4) / 8 := 
by
  -- proof needed
  sorry

end extreme_points_inequality_l209_209209


namespace exists_unique_i_l209_209862

theorem exists_unique_i (p : ℕ) (hp : Nat.Prime p) (hp2 : p % 2 = 1) 
  (a : ℤ) (ha1 : 2 ≤ a) (ha2 : a ≤ p - 2) : 
  ∃! (i : ℤ), 2 ≤ i ∧ i ≤ p - 2 ∧ (i * a) % p = 1 ∧ Nat.gcd (i.natAbs) (a.natAbs) = 1 :=
sorry

end exists_unique_i_l209_209862


namespace total_quarters_l209_209678

def Sara_initial_quarters : Nat := 21
def quarters_given_by_dad : Nat := 49

theorem total_quarters : Sara_initial_quarters + quarters_given_by_dad = 70 := 
by
  sorry

end total_quarters_l209_209678


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209794

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209794


namespace age_problem_l209_209982

theorem age_problem
  (D M : ℕ)
  (h1 : M = D + 45)
  (h2 : M - 5 = 6 * (D - 5)) :
  D = 14 ∧ M = 59 := by
  sorry

end age_problem_l209_209982


namespace mass_percentage_of_C_in_CCl4_l209_209083

theorem mass_percentage_of_C_in_CCl4 :
  let mass_carbon : ℝ := 12.01
  let mass_chlorine : ℝ := 35.45
  let molar_mass_CCl4 : ℝ := mass_carbon + 4 * mass_chlorine
  let mass_percentage_C : ℝ := (mass_carbon / molar_mass_CCl4) * 100
  mass_percentage_C = 7.81 := 
by
  sorry

end mass_percentage_of_C_in_CCl4_l209_209083


namespace cost_of_bananas_l209_209443

theorem cost_of_bananas (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + B = 5) : B = 3 :=
by
  sorry

end cost_of_bananas_l209_209443


namespace largest_value_in_interval_l209_209518

theorem largest_value_in_interval (x : ℝ) (h : 0 < x ∧ x < 1) : 
  (∀ y ∈ ({x, x^3, 3*x, x^(1/3), 1/x} : Set ℝ), y ≤ 1/x) :=
sorry

end largest_value_in_interval_l209_209518


namespace max_sum_of_distinct_integers_l209_209117

theorem max_sum_of_distinct_integers (A B C : ℕ) (hABC_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (hProduct : A * B * C = 1638) :
  A + B + C ≤ 126 :=
sorry

end max_sum_of_distinct_integers_l209_209117


namespace division_remainder_l209_209109

theorem division_remainder (dividend divisor quotient remainder : ℕ)
  (h₁ : dividend = 689)
  (h₂ : divisor = 36)
  (h₃ : quotient = 19)
  (h₄ : dividend = divisor * quotient + remainder) :
  remainder = 5 :=
by
  sorry

end division_remainder_l209_209109


namespace greatest_third_side_of_triangle_l209_209572

theorem greatest_third_side_of_triangle (a b : ℕ) (h1 : a = 7) (h2 : b = 10) : ∃ x : ℕ, x < a + b ∧ x = 16 := by
  use 16
  rw [h1, h2]
  split
  · linarith
  · rfl

end greatest_third_side_of_triangle_l209_209572


namespace number_of_violinists_l209_209873

open Nat

/-- There are 3 violinists in the orchestra, based on given conditions. -/
theorem number_of_violinists
  (total : ℕ)
  (percussion : ℕ)
  (brass : ℕ)
  (cellist : ℕ)
  (contrabassist : ℕ)
  (woodwinds : ℕ)
  (maestro : ℕ)
  (total_eq : total = 21)
  (percussion_eq : percussion = 1)
  (brass_eq : brass = 7)
  (strings_excluding_violinists : ℕ)
  (cellist_eq : cellist = 1)
  (contrabassist_eq : contrabassist = 1)
  (woodwinds_eq : woodwinds = 7)
  (maestro_eq : maestro = 1) :
  (total - (percussion + brass + (cellist + contrabassist) + woodwinds + maestro)) = 3 := 
by
  sorry

end number_of_violinists_l209_209873


namespace rectangle_area_l209_209756

-- Conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def ratio_length_to_width : ℝ := 3

-- Given the ratio of the length to the width is 3:1
def length : ℝ := ratio_length_to_width * width

-- Theorem stating the area of the rectangle
theorem rectangle_area :
  let area := length * width
  area = 432 := by
    sorry

end rectangle_area_l209_209756


namespace circumscribed_triangle_area_relationship_l209_209166

theorem circumscribed_triangle_area_relationship (X Y Z : ℝ) :
  let a := 15
  let b := 20
  let c := 25
  let triangle_area := (1/2) * a * b
  let diameter := c
  let radius := diameter / 2
  let circle_area := Real.pi * radius^2
  let Z := circle_area / 2
  (X + Y + triangle_area = Z) :=
sorry

end circumscribed_triangle_area_relationship_l209_209166


namespace line_tangent_to_circle_l209_209383

theorem line_tangent_to_circle (k : ℝ) :
  (∀ x y : ℝ, k * x - y - 2 * k + 3 = 0 → x^2 + (y + 1)^2 = 4) → k = 3 / 4 :=
by 
  intro h
  sorry

end line_tangent_to_circle_l209_209383


namespace ratio_rectangle_to_semicircles_area_l209_209597

theorem ratio_rectangle_to_semicircles_area (AB AD : ℝ) (h1 : AB = 40) (h2 : AD / AB = 3 / 2) : 
  (AB * AD) / (2 * (π * (AB / 2)^2)) = 6 / π :=
by
  -- here we process the proof
  sorry

end ratio_rectangle_to_semicircles_area_l209_209597


namespace probability_all_calls_same_probability_two_calls_for_A_l209_209151

theorem probability_all_calls_same (pA pB pC : ℚ) (hA : pA = 1/6) (hB : pB = 1/3) (hC : pC = 1/2) :
  (pA^3 + pB^3 + pC^3) = 1/6 :=
by
  sorry

theorem probability_two_calls_for_A (pA : ℚ) (hA : pA = 1/6) :
  (3 * (pA^2) * (5/6)) = 5/72 :=
by
  sorry

end probability_all_calls_same_probability_two_calls_for_A_l209_209151


namespace students_not_yet_pictured_l209_209892

def students_in_class : ℕ := 24
def students_before_lunch : ℕ := students_in_class / 3
def students_after_lunch_before_gym : ℕ := 10
def total_students_pictures_taken : ℕ := students_before_lunch + students_after_lunch_before_gym

theorem students_not_yet_pictured : total_students_pictures_taken = 18 → students_in_class - total_students_pictures_taken = 6 := by
  intros h
  rw [h]
  rfl

end students_not_yet_pictured_l209_209892


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209798

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (sec 150 = - (2 * sqrt 3) / 3) :=
by
  -- Use the known conditions as definitions within the Lean proof.
  have h1 : sec θ = 1 / cos θ := sorry
  have h2 : cos (180 - θ) = -cos θ := sorry
  have h3 : cos 30 = sqrt 3 / 2 := sorry
  -- Proof statements to show sec 150 = - (2 * sqrt 3) / 3
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209798


namespace lavinias_son_older_than_daughter_l209_209857

def katies_daughter_age := 12
def lavinias_daughter_age := katies_daughter_age - 10
def lavinias_son_age := 2 * katies_daughter_age

theorem lavinias_son_older_than_daughter :
  lavinias_son_age - lavinias_daughter_age = 22 :=
by
  sorry

end lavinias_son_older_than_daughter_l209_209857


namespace margaret_score_l209_209271

theorem margaret_score (average_score marco_score margaret_score : ℝ)
  (h1: average_score = 90)
  (h2: marco_score = average_score - 0.10 * average_score)
  (h3: margaret_score = marco_score + 5) : 
  margaret_score = 86 := 
by
  sorry

end margaret_score_l209_209271


namespace trips_needed_l209_209617

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l209_209617


namespace find_k_l209_209108

theorem find_k (x y k : ℝ) 
  (h1 : 4 * x + 2 * y = 5 * k - 4) 
  (h2 : 2 * x + 4 * y = -1) 
  (h3 : x - y = 1) : 
  k = 1 := 
by sorry

end find_k_l209_209108


namespace find_m_l209_209834

theorem find_m (m : ℝ) (α : ℝ) (h1 : P(m, 2) is_on_terminal_side_of α) (h2 : sin α = 1/3) : 
  m = 4 * real.sqrt 2 ∨ m = -4 * real.sqrt 2 := 
sorry

end find_m_l209_209834


namespace opponent_final_score_l209_209147

theorem opponent_final_score (x : ℕ) (h : x + 29 = 39) : x = 10 :=
by {
  sorry
}

end opponent_final_score_l209_209147


namespace sheila_attends_picnic_l209_209418

theorem sheila_attends_picnic :
  let probRain := 0.30
  let probSunny := 0.50
  let probCloudy := 0.20
  let probAttendIfRain := 0.15
  let probAttendIfSunny := 0.85
  let probAttendIfCloudy := 0.40
  (probRain * probAttendIfRain + probSunny * probAttendIfSunny + probCloudy * probAttendIfCloudy) = 0.55 :=
by sorry

end sheila_attends_picnic_l209_209418


namespace solution_system_linear_eqns_l209_209362

theorem solution_system_linear_eqns
    (a1 b1 c1 a2 b2 c2 : ℝ)
    (h1: a1 * 6 + b1 * 3 = c1)
    (h2: a2 * 6 + b2 * 3 = c2) :
    (4 * a1 * 22 + 3 * b1 * 33 = 11 * c1) ∧
    (4 * a2 * 22 + 3 * b2 * 33 = 11 * c2) :=
by
    sorry

end solution_system_linear_eqns_l209_209362


namespace cost_of_pastrami_l209_209260

-- Definitions based on the problem conditions
def cost_of_reuben (R : ℝ) : Prop :=
  ∃ P : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55

-- Theorem stating the solution to the problem
theorem cost_of_pastrami : ∃ P : ℝ, ∃ R : ℝ, P = R + 2 ∧ 10 * R + 5 * P = 55 ∧ P = 5 :=
by 
  sorry

end cost_of_pastrami_l209_209260


namespace coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209717

theorem coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8 :
  (nat.choose 8 5) = 56 :=
by
  sorry

end coefficient_of_x3y5_in_expansion_of_x_plus_y_pow_8_l209_209717


namespace reciprocal_of_neg3_l209_209006

theorem reciprocal_of_neg3 : (1 : ℚ) / (-3 : ℚ) = -1 / 3 := 
by
  sorry

end reciprocal_of_neg3_l209_209006


namespace value_of_g_g_2_l209_209220

def g (x : ℝ) : ℝ := 4 * x^2 - 6

theorem value_of_g_g_2 :
  g (g 2) = 394 :=
sorry

end value_of_g_g_2_l209_209220


namespace coefficient_x3y5_in_expansion_l209_209710

theorem coefficient_x3y5_in_expansion (x y : ℤ) :
  (↑(nat.choose 8 3) : ℤ) = 56 :=
by
  sorry

end coefficient_x3y5_in_expansion_l209_209710


namespace six_digit_number_under_5_lakh_with_digit_sum_43_l209_209303

def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000
def under_500000 (n : ℕ) : Prop := n < 500000
def digit_sum (n : ℕ) : ℕ := (n / 100000) + (n / 10000 % 10) + (n / 1000 % 10) + (n / 100 % 10) + (n / 10 % 10) + (n % 10)

theorem six_digit_number_under_5_lakh_with_digit_sum_43 :
  is_6_digit 499993 ∧ under_500000 499993 ∧ digit_sum 499993 = 43 :=
by 
  sorry

end six_digit_number_under_5_lakh_with_digit_sum_43_l209_209303


namespace johns_remaining_money_l209_209391

theorem johns_remaining_money (q : ℝ) : 
  let cost_of_drinks := 4 * q,
      cost_of_small_pizzas := 2 * q,
      cost_of_large_pizza := 4 * q,
      total_cost := cost_of_drinks + cost_of_small_pizzas + cost_of_large_pizza,
      initial_money := 50 in
  initial_money - total_cost = 50 - 10 * q :=
by
  sorry

end johns_remaining_money_l209_209391


namespace price_of_pastries_is_5_l209_209267

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end price_of_pastries_is_5_l209_209267


namespace trigonometric_identity_l209_209659

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  Real.sin α ^ 2 - Real.cos α ^ 2 + Real.sin α * Real.cos α = 1 := 
by {
  sorry
}

end trigonometric_identity_l209_209659


namespace final_color_all_blue_l209_209406

-- Definitions based on the problem's initial conditions
def initial_blue_sheep : ℕ := 22
def initial_red_sheep : ℕ := 18
def initial_green_sheep : ℕ := 15

-- The final problem statement: prove that all sheep end up being blue
theorem final_color_all_blue (B R G : ℕ) 
  (hB : B = initial_blue_sheep) 
  (hR : R = initial_red_sheep) 
  (hG : G = initial_green_sheep) 
  (interaction : ∀ (B R G : ℕ), (B > 0 ∨ R > 0 ∨ G > 0) → (R ≡ G [MOD 3])) :
  ∃ b, b = B + R + G ∧ R = 0 ∧ G = 0 ∧ b % 3 = 1 ∧ B = b :=
by
  -- Proof to be provided
  sorry

end final_color_all_blue_l209_209406


namespace tailor_cut_difference_l209_209178

theorem tailor_cut_difference :
  (7 / 8 + 11 / 12) - (5 / 6 + 3 / 4) = 5 / 24 :=
by
  sorry

end tailor_cut_difference_l209_209178


namespace larger_of_two_numbers_l209_209308

theorem larger_of_two_numbers (A B : ℕ) (hcf : A.gcd B = 47) (lcm_factors : A.lcm B = 47 * 49 * 11 * 13 * 4913) : max A B = 123800939 :=
sorry

end larger_of_two_numbers_l209_209308


namespace dresser_clothing_capacity_l209_209137

theorem dresser_clothing_capacity (pieces_per_drawer : ℕ) (number_of_drawers : ℕ) (total_pieces : ℕ) 
  (h1 : pieces_per_drawer = 5)
  (h2 : number_of_drawers = 8)
  (h3 : total_pieces = 40) :
  pieces_per_drawer * number_of_drawers = total_pieces :=
by {
  sorry
}

end dresser_clothing_capacity_l209_209137


namespace number_of_ways_to_place_letters_l209_209415

-- Define the number of letters and mailboxes
def num_letters : Nat := 3
def num_mailboxes : Nat := 5

-- Define the function to calculate the number of ways to place the letters into mailboxes
def count_ways (n : Nat) (m : Nat) : Nat := m ^ n

-- The theorem to prove
theorem number_of_ways_to_place_letters :
  count_ways num_letters num_mailboxes = 5 ^ 3 :=
by
  sorry

end number_of_ways_to_place_letters_l209_209415


namespace sum_of_two_consecutive_negative_integers_l209_209280

theorem sum_of_two_consecutive_negative_integers (n : ℤ) (h : n * (n + 1) = 2210) (hn : n < 0) : n + (n + 1) = -95 := 
sorry

end sum_of_two_consecutive_negative_integers_l209_209280


namespace oranges_needed_l209_209133

theorem oranges_needed 
  (total_fruit_needed : ℕ := 12) 
  (apples : ℕ := 3) 
  (bananas : ℕ := 4) : 
  total_fruit_needed - (apples + bananas) = 5 :=
by 
  sorry

end oranges_needed_l209_209133


namespace min_value_condition_solve_inequality_l209_209372

open Real

-- Define the function f(x) = |x - a| + |x + 2|
def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 2)

-- Part I: Proving the values of a for f(x) having minimum value of 2
theorem min_value_condition (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 2) → (∃ x : ℝ, f x a = 2) → (a = 0 ∨ a = -4) :=
by
  sorry

-- Part II: Solving inequality f(x) ≤ 6 when a = 2
theorem solve_inequality : 
  ∀ x : ℝ, f x 2 ≤ 6 ↔ (x ≥ -3 ∧ x ≤ 3) :=
by
  sorry

end min_value_condition_solve_inequality_l209_209372


namespace probability_of_same_type_is_correct_l209_209780

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def ways_to_pick_any_3_socks : ℕ := Nat.choose total_socks 3
noncomputable def ways_to_pick_3_black_socks : ℕ := Nat.choose 12 3
noncomputable def ways_to_pick_3_white_socks : ℕ := Nat.choose 10 3
noncomputable def ways_to_pick_3_striped_socks : ℕ := Nat.choose 6 3
noncomputable def ways_to_pick_3_same_type : ℕ := ways_to_pick_3_black_socks + ways_to_pick_3_white_socks + ways_to_pick_3_striped_socks
noncomputable def probability_same_type : ℚ := ways_to_pick_3_same_type / ways_to_pick_any_3_socks

theorem probability_of_same_type_is_correct :
  probability_same_type = 60 / 546 :=
by
  sorry

end probability_of_same_type_is_correct_l209_209780


namespace neg_p_false_sufficient_but_not_necessary_for_p_or_q_l209_209096

variable (p q : Prop)

theorem neg_p_false_sufficient_but_not_necessary_for_p_or_q :
  (¬ p = false) → (p ∨ q) ∧ ¬((p ∨ q) → (¬ p = false)) :=
by
  sorry

end neg_p_false_sufficient_but_not_necessary_for_p_or_q_l209_209096


namespace ounces_per_container_l209_209307

def weight_pounds : ℝ := 3.75
def num_containers : ℕ := 4
def pound_to_ounces : ℕ := 16

theorem ounces_per_container :
  (weight_pounds * pound_to_ounces) / num_containers = 15 :=
by
  sorry

end ounces_per_container_l209_209307


namespace people_got_on_at_third_stop_l209_209180

theorem people_got_on_at_third_stop :
  let people_1st_stop := 10
  let people_off_2nd_stop := 3
  let twice_people_1st_stop := 2 * people_1st_stop
  let people_off_3rd_stop := 18
  let people_after_3rd_stop := 12

  let people_after_1st_stop := people_1st_stop
  let people_after_2nd_stop := (people_after_1st_stop - people_off_2nd_stop) + twice_people_1st_stop
  let people_after_3rd_stop_but_before_new_ones := people_after_2nd_stop - people_off_3rd_stop
  let people_on_at_3rd_stop := people_after_3rd_stop - people_after_3rd_stop_but_before_new_ones

  people_on_at_3rd_stop = 3 := 
by
  sorry

end people_got_on_at_third_stop_l209_209180


namespace at_least_two_dice_same_number_probability_l209_209410

theorem at_least_two_dice_same_number_probability :
  let total_outcomes := 6^8
  let favorable_outcomes := 28 * 6! * 6^2
  let probability_all_different := favorable_outcomes / total_outcomes
  let required_probability := 1 - probability_all_different
  required_probability = 191 / 336
:= by
  sorry

end at_least_two_dice_same_number_probability_l209_209410


namespace sin_gamma_delta_l209_209105

theorem sin_gamma_delta (γ δ : ℝ)
  (hγ : Complex.exp (Complex.I * γ) = Complex.ofReal 4 / 5 + Complex.I * (3 / 5))
  (hδ : Complex.exp (Complex.I * δ) = Complex.ofReal (-5 / 13) + Complex.I * (12 / 13)) :
  Real.sin (γ + δ) = 21 / 65 :=
by
  sorry

end sin_gamma_delta_l209_209105


namespace correct_operation_l209_209915

theorem correct_operation (a m : ℝ) :
  ¬(a^5 / a^10 = a^2) ∧ 
  (-2 * a^3)^2 = 4 * a^6 ∧ 
  ¬((1 / (2 * m)) - (1 / m) = (1 / m)) ∧ 
  ¬(a^4 + a^3 = a^7) :=
by
  sorry

end correct_operation_l209_209915


namespace sequence_general_formula_l209_209227

-- Define conditions: The sum of the first n terms of the sequence is Sn = an - 3
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom condition (n : ℕ) : S n = a n - 3

-- Define the main theorem to prove
theorem sequence_general_formula (n : ℕ) (hn : 0 < n) : a n = 2 * 3 ^ n :=
sorry

end sequence_general_formula_l209_209227


namespace river_current_speed_l209_209313

variable (c : ℝ)

def boat_speed_still_water : ℝ := 20
def round_trip_distance : ℝ := 182
def round_trip_time : ℝ := 10

theorem river_current_speed (h : (91 / (boat_speed_still_water - c)) + (91 / (boat_speed_still_water + c)) = round_trip_time) : c = 6 :=
sorry

end river_current_speed_l209_209313


namespace calculate_expression_l209_209999

theorem calculate_expression : 7 * (12 + 2 / 5) - 3 = 83.8 :=
by
  sorry

end calculate_expression_l209_209999


namespace smallest_7_heavy_three_digit_number_l209_209050

def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

theorem smallest_7_heavy_three_digit_number :
  ∃ n : ℕ, is_three_digit n ∧ is_7_heavy n ∧ (∀ m : ℕ, is_three_digit m ∧ is_7_heavy m → n ≤ m) ∧
  n = 103 := 
by
  sorry

end smallest_7_heavy_three_digit_number_l209_209050


namespace fibonacci_series_sum_l209_209244

-- Definition of the Fibonacci sequence
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib (n + 1) + fib n

-- Theorem to prove that the infinite series sum is 2
theorem fibonacci_series_sum : (∑' n : ℕ, (fib n : ℝ) / (2 ^ n : ℝ)) = 2 :=
sorry

end fibonacci_series_sum_l209_209244


namespace people_in_line_l209_209183

theorem people_in_line (initially_in_line : ℕ) (left_line : ℕ) (after_joined_line : ℕ) 
  (h1 : initially_in_line = 12) (h2 : left_line = 10) (h3 : after_joined_line = 17) : 
  initially_in_line - left_line + 15 = after_joined_line := by
  sorry

end people_in_line_l209_209183


namespace contrapositive_proof_l209_209677

-- Defining the necessary variables and the hypothesis
variables (a b : ℝ)

theorem contrapositive_proof (h : a^2 - b^2 + 2 * a - 4 * b - 3 ≠ 0) : a - b ≠ 1 :=
sorry

end contrapositive_proof_l209_209677


namespace find_e_l209_209128

noncomputable def f (x : ℝ) (c : ℝ) := 5 * x + 2 * c

noncomputable def g (x : ℝ) (c : ℝ) := c * x^2 + 3

noncomputable def fg (x : ℝ) (c : ℝ) := f (g x c) c

theorem find_e (c : ℝ) (e : ℝ) (h1 : f (g x c) c = 15 * x^2 + e) (h2 : 5 * c = 15) : e = 21 :=
by
  sorry

end find_e_l209_209128


namespace frequency_of_middle_group_l209_209850

theorem frequency_of_middle_group
    (num_rectangles : ℕ)
    (middle_area : ℝ)
    (other_areas_sum : ℝ)
    (sample_size : ℕ)
    (total_area_norm : ℝ)
    (h1 : num_rectangles = 11)
    (h2 : middle_area = other_areas_sum)
    (h3 : sample_size = 160)
    (h4 : middle_area + other_areas_sum = total_area_norm)
    (h5 : total_area_norm = 1):
    160 * (middle_area / total_area_norm) = 80 :=
by
  sorry

end frequency_of_middle_group_l209_209850


namespace bill_difference_l209_209536

theorem bill_difference (mandy_bills : ℕ) (manny_bills : ℕ) 
  (mandy_bill_value : ℕ) (manny_bill_value : ℕ) (target_bill_value : ℕ) 
  (h_mandy : mandy_bills = 3) (h_mandy_val : mandy_bill_value = 20) 
  (h_manny : manny_bills = 2) (h_manny_val : manny_bill_value = 50)
  (h_target : target_bill_value = 10) :
  (manny_bills * manny_bill_value / target_bill_value) - (mandy_bills * mandy_bill_value / target_bill_value) = 4 :=
by
  sorry

end bill_difference_l209_209536


namespace possible_values_of_A_l209_209565

theorem possible_values_of_A :
  ∃ (A : ℕ), (A ≤ 4 ∧ A < 10) ∧ A = 5 :=
sorry

end possible_values_of_A_l209_209565


namespace sec_150_eq_neg_2_sqrt3_div_3_l209_209783

theorem sec_150_eq_neg_2_sqrt3_div_3 : Real.sec (150 * Real.pi / 180) = -2 * Real.sqrt 3 / 3 := 
by
  -- Conversion of degrees to radians: 150° -> 150 * π / 180 radians
  -- Assertion of the correct answer.
  sorry

end sec_150_eq_neg_2_sqrt3_div_3_l209_209783


namespace unique_three_digit_sum_27_l209_209354

/--
There is exactly one three-digit whole number such that the sum of its digits is 27.
-/
theorem unique_three_digit_sum_27 :
  ∃! (n : ℕ), 
    100 ≤ n ∧ n < 1000 ∧ 
    let a := n / 100, b := (n / 10) % 10, c := n % 10
    in a + b + c = 27 := sorry

end unique_three_digit_sum_27_l209_209354


namespace distance_between_A_and_B_l209_209174

theorem distance_between_A_and_B (x : ℝ) (boat_speed : ℝ) (flow_speed : ℝ) (dist_AC : ℝ) (total_time : ℝ) :
  (boat_speed = 8) →
  (flow_speed = 2) →
  (dist_AC = 2) →
  (total_time = 3) →
  (x = 10 ∨ x = 12.5) :=
by {
  sorry
}

end distance_between_A_and_B_l209_209174


namespace revenue_highest_visitors_is_48_thousand_l209_209314

-- Define the frequencies for each day
def freq_Oct_1 : ℝ := 0.05
def freq_Oct_2 : ℝ := 0.08
def freq_Oct_3 : ℝ := 0.09
def freq_Oct_4 : ℝ := 0.13
def freq_Oct_5 : ℝ := 0.30
def freq_Oct_6 : ℝ := 0.15
def freq_Oct_7 : ℝ := 0.20

-- Define the revenue on October 1st
def revenue_Oct_1 : ℝ := 80000

-- Define the revenue is directly proportional to the frequency of visitors
def avg_daily_visitor_spending_is_constant := true

-- The goal is to prove that the revenue on the day with the highest frequency is 48 thousand yuan
theorem revenue_highest_visitors_is_48_thousand :
  avg_daily_visitor_spending_is_constant →
  revenue_Oct_1 / freq_Oct_1 = x / freq_Oct_5 →
  x = 48000 :=
by
  sorry

end revenue_highest_visitors_is_48_thousand_l209_209314


namespace coefficient_x3_y5_in_binomial_expansion_l209_209732

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l209_209732


namespace nat_prime_p_and_5p_plus_1_is_prime_l209_209349

theorem nat_prime_p_and_5p_plus_1_is_prime (p : ℕ) (hp : Nat.Prime p) (h5p1 : Nat.Prime (5 * p + 1)) : p = 2 := 
by 
  -- Sorry is added to skip the proof
  sorry 

end nat_prime_p_and_5p_plus_1_is_prime_l209_209349


namespace smallest_nine_digit_times_smallest_seven_digit_l209_209435

theorem smallest_nine_digit_times_smallest_seven_digit :
  let smallest_nine_digit := 100000000
  let smallest_seven_digit := 1000000
  smallest_nine_digit = 100 * smallest_seven_digit :=
by
  sorry

end smallest_nine_digit_times_smallest_seven_digit_l209_209435


namespace sum_of_adjacent_cells_multiple_of_4_l209_209672

theorem sum_of_adjacent_cells_multiple_of_4 :
  ∃ (i j : ℕ) (a b : ℕ) (H₁ : i < 22) (H₂ : j < 22),
    let grid (i j : ℕ) : ℕ := -- define the function for grid indexing
      ((i * 22) + j + 1 : ℕ)
    ∃ (i1 j1 : ℕ) (H₁₁ : i1 = i ∨ i1 = i + 1 ∨ i1 = i - 1)
                   (H₁₂ : j1 = j ∨ j1 = j + 1 ∨ j1 = j - 1),
      a = grid i j ∧ b = grid i1 j1 ∧ (a + b) % 4 = 0 := sorry

end sum_of_adjacent_cells_multiple_of_4_l209_209672


namespace matchstick_polygon_area_l209_209012

-- Given conditions
def number_of_matches := 12
def length_of_each_match := 2 -- in cm

-- Question: Is it possible to construct a polygon with an area of 16 cm^2 using all the matches?
def polygon_possible : Prop :=
  ∃ (p : Polygon), 
    (p.edges = number_of_matches) ∧ 
    (∃ (match_length : ℝ), match_length = length_of_each_match ∧ by 
      -- Form the polygon using all matches without breaking
      sorry) ∧ 
    (polygon_area p = 16)

-- Proof statement
theorem matchstick_polygon_area :
  polygon_possible :=
  sorry

end matchstick_polygon_area_l209_209012


namespace mean_of_squares_of_first_four_odd_numbers_l209_209476

theorem mean_of_squares_of_first_four_odd_numbers :
  (1^2 + 3^2 + 5^2 + 7^2) / 4 = 21 := 
by
  sorry

end mean_of_squares_of_first_four_odd_numbers_l209_209476


namespace number_of_correct_statements_l209_209431

def statement1_condition : Prop :=
∀ a b : ℝ, (a - b > 0) → (a > 0 ∧ b > 0)

def statement2_condition : Prop :=
∀ a b : ℝ, a - b = a + (-b)

def statement3_condition : Prop :=
∀ a : ℝ, (a - (-a) = 0)

def statement4_condition : Prop :=
∀ a : ℝ, 0 - a = -a

theorem number_of_correct_statements : 
  (¬ statement1_condition ∧ statement2_condition ∧ ¬ statement3_condition ∧ statement4_condition) →
  (2 = 2) :=
by
  intros
  trivial

end number_of_correct_statements_l209_209431


namespace x_plus_y_value_l209_209745

def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def count_even_integers (a b : ℕ) : ℕ :=
  (b - a) / 2 + 1

theorem x_plus_y_value :
  let x := sum_of_integers 50 70
  let y := count_even_integers 50 70
  x + y = 1271 := by
    let x := sum_of_integers 50 70
    let y := count_even_integers 50 70
    sorry

end x_plus_y_value_l209_209745


namespace percent_area_square_in_rectangle_l209_209044

theorem percent_area_square_in_rectangle
  (s : ℝ) 
  (w : ℝ) 
  (l : ℝ)
  (h1 : w = 3 * s) 
  (h2 : l = (9 / 2) * s) 
  : (s^2 / (l * w)) * 100 = 7.41 :=
by
  sorry

end percent_area_square_in_rectangle_l209_209044


namespace lifting_ratio_after_gain_l209_209994

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l209_209994


namespace clothing_factory_exceeded_tasks_l209_209596

theorem clothing_factory_exceeded_tasks :
  let first_half := (2 : ℚ) / 3
  let second_half := (3 : ℚ) / 5
  first_half + second_half - 1 = (4 : ℚ) / 15 :=
by
  sorry

end clothing_factory_exceeded_tasks_l209_209596


namespace problem1_problem2_l209_209210

-- Definition of the function
def f (a x : ℝ) := x^2 + a * x + 3

-- Problem statement 1: Prove that if f(x) ≥ a for all x ∈ ℜ, then a ≤ 3.
theorem problem1 (a : ℝ) : (∀ x : ℝ, f a x ≥ a) → a ≤ 3 := sorry

-- Problem statement 2: Prove that if f(x) ≥ a for all x ∈ [-2, 2], then -6 ≤ a ≤ 2.
theorem problem2 (a : ℝ) : (∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x ≥ a) → -6 ≤ a ∧ a ≤ 2 := sorry

end problem1_problem2_l209_209210


namespace arithmetic_sequence_a13_l209_209387

variable (a1 d : ℤ)

theorem arithmetic_sequence_a13 (h : a1 + 2 * d + a1 + 8 * d + a1 + 26 * d = 12) : a1 + 12 * d = 4 :=
by
  sorry

end arithmetic_sequence_a13_l209_209387


namespace immortal_flea_can_visit_every_natural_l209_209558

theorem immortal_flea_can_visit_every_natural :
  ∀ (k : ℕ), ∃ (jumps : ℕ → ℤ), (∀ n : ℕ, ∃ m : ℕ, jumps m = n) :=
by
  -- proof goes here
  sorry

end immortal_flea_can_visit_every_natural_l209_209558


namespace arithmetic_sequence_length_l209_209094

theorem arithmetic_sequence_length 
  (a₁ : ℕ) (d : ℤ) (x : ℤ) (n : ℕ) 
  (h_start : a₁ = 20)
  (h_diff : d = -2)
  (h_eq : x = 10)
  (h_term : x = a₁ + (n - 1) * d) :
  n = 6 :=
by
  sorry

end arithmetic_sequence_length_l209_209094


namespace film_radius_l209_209397

theorem film_radius 
  (thickness : ℝ)
  (container_volume : ℝ)
  (r : ℝ)
  (H1 : thickness = 0.25)
  (H2 : container_volume = 128) :
  r = Real.sqrt (512 / Real.pi) :=
by
  -- Placeholder for proof
  sorry

end film_radius_l209_209397


namespace initial_dogwood_trees_in_park_l209_209150

def num_added_trees := 5 + 4
def final_num_trees := 16
def initial_num_trees (x : ℕ) := x

theorem initial_dogwood_trees_in_park (x : ℕ) 
  (h1 : num_added_trees = 9) 
  (h2 : final_num_trees = 16) : 
  initial_num_trees x + num_added_trees = final_num_trees → 
  x = 7 := 
by 
  intro h3
  rw [initial_num_trees, num_added_trees] at h3
  linarith

end initial_dogwood_trees_in_park_l209_209150


namespace numBoysInClassroom_l209_209287

-- Definitions based on the problem conditions
def numGirls : ℕ := 10
def girlsToBoysRatio : ℝ := 0.5

-- The statement to prove
theorem numBoysInClassroom : ∃ B : ℕ, girlsToBoysRatio * B = numGirls ∧ B = 20 :=
by
  -- Proof goes here
  sorry

end numBoysInClassroom_l209_209287


namespace boxes_left_for_Sonny_l209_209419

def initial_boxes : ℕ := 45
def boxes_given_to_brother : ℕ := 12
def boxes_given_to_sister : ℕ := 9
def boxes_given_to_cousin : ℕ := 7

def total_given_away : ℕ := boxes_given_to_brother + boxes_given_to_sister + boxes_given_to_cousin

def remaining_boxes : ℕ := initial_boxes - total_given_away

theorem boxes_left_for_Sonny : remaining_boxes = 17 := by
  sorry

end boxes_left_for_Sonny_l209_209419


namespace g_g2_is_394_l209_209221

def g (x : ℝ) : ℝ :=
  4 * x^2 - 6

theorem g_g2_is_394 : g(g(2)) = 394 :=
by
  -- Proof is omitted by using sorry
  sorry

end g_g2_is_394_l209_209221


namespace total_pies_sold_l209_209068

def shepherds_pie_slices_per_pie : Nat := 4
def chicken_pot_pie_slices_per_pie : Nat := 5
def shepherds_pie_slices_ordered : Nat := 52
def chicken_pot_pie_slices_ordered : Nat := 80

theorem total_pies_sold :
  shepherds_pie_slices_ordered / shepherds_pie_slices_per_pie +
  chicken_pot_pie_slices_ordered / chicken_pot_pie_slices_per_pie = 29 := by
sorry

end total_pies_sold_l209_209068


namespace area_of_region_l209_209905

theorem area_of_region : ∃ A, (∀ x y : ℝ, x^2 + y^2 + 6*x - 4*y = 12 → A = 25 * Real.pi) :=
by
  -- Completing the square and identifying the circle
  -- We verify that the given equation represents a circle
  existsi (25 * Real.pi)
  intros x y h
  sorry

end area_of_region_l209_209905


namespace ladder_distance_l209_209946

-- Define the distance function
def distance_from_base_to_wall (ladder_length wall_height : ℝ) : ℝ :=
  Real.sqrt (ladder_length^2 - wall_height^2)

-- State the theorem proving the distance
theorem ladder_distance (l : ℝ) (h : ℝ) (d : ℝ) 
  (hl: l = 13) (hh: h = 12) (hd: d = distance_from_base_to_wall l h) : d = 5 :=
by
  -- use the given conditions
  rw [hl, hh, distance_from_base_to_wall]
  -- calculate the distance using the Pythagorean theorem
  rw [Real.sqrt]
  sorry

end ladder_distance_l209_209946


namespace movie_replay_count_l209_209766

def movie_length_hours : ℝ := 1.5
def advertisement_length_minutes : ℝ := 20
def theater_operating_hours : ℝ := 11

theorem movie_replay_count :
  let movie_length_minutes := movie_length_hours * 60
  let total_showing_time_minutes := movie_length_minutes + advertisement_length_minutes
  let operating_time_minutes := theater_operating_hours * 60
  (operating_time_minutes / total_showing_time_minutes) = 6 :=
by
  sorry

end movie_replay_count_l209_209766


namespace total_cost_l209_209214

def num_of_rings : ℕ := 2

def cost_per_ring : ℕ := 12

theorem total_cost : num_of_rings * cost_per_ring = 24 :=
by sorry

end total_cost_l209_209214


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209724

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209724


namespace minimize_fraction_l209_209286

theorem minimize_fraction (n : ℕ) (h : 0 < n) : 
  (n = 9) ↔ (∀ m : ℕ, 0 < m → (n / 3 + 27 / n) ≤ (m / 3 + 27 / m)) :=
by
  sorry

end minimize_fraction_l209_209286


namespace multiplication_factor_correct_l209_209041

theorem multiplication_factor_correct (N X : ℝ) (h1 : 98 = abs ((N * X - N / 10) / (N * X)) * 100) : X = 5 := by
  sorry

end multiplication_factor_correct_l209_209041


namespace sum_of_decimals_as_fraction_l209_209632

theorem sum_of_decimals_as_fraction :
  let x := (0 : ℝ) + 1 / 3;
  let y := (0 : ℝ) + 2 / 3;
  let z := (0 : ℝ) + 2 / 5;
  x + y + z = 7 / 5 :=
by
  let x := (0 : ℝ) + 1 / 3
  let y := (0 : ℝ) + 2 / 3
  let z := (0 : ℝ) + 2 / 5
  show x + y + z = 7 / 5
  sorry

end sum_of_decimals_as_fraction_l209_209632


namespace union_A_B_inter_compl_A_B_subset_intersection_l209_209375

open Set

variable {R : Set Real}
def A : Set ℝ := {x : ℝ | -5 < x ∧ x < 1}
def B : Set ℝ := {x : ℝ | -2 < x ∧ x < 8}
def C (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def A_c : Set ℝ := compl A

-- First part: Prove A ∪ B = {x | -5 < x < 8}
theorem union_A_B : A ∪ B = {x : ℝ | -5 < x ∧ x < 8} :=
sorry

-- Second part: Prove Aᶜ ∩ B = {x | 1 ≤ x ∧ x < 8}
theorem inter_compl_A_B : A_c ∩ B = {x : ℝ | 1 ≤ x ∧ x < 8} :=
sorry

-- Third part: Given A ∩ B ⊆ C, prove a ≥ 1
theorem subset_intersection (a : ℝ) (h : A ∩ B ⊆ C a) : a ≥ 1 :=
sorry

end union_A_B_inter_compl_A_B_subset_intersection_l209_209375


namespace polynomial_expansion_sum_is_21_l209_209647

theorem polynomial_expansion_sum_is_21 :
  ∃ (A B C D : ℤ), (∀ (x : ℤ), (x + 2) * (3 * x^2 - x + 5) = A * x^3 + B * x^2 + C * x + D) ∧
  A + B + C + D = 21 :=
by
  sorry

end polynomial_expansion_sum_is_21_l209_209647


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209784

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209784


namespace quadractic_roots_value_l209_209247

theorem quadractic_roots_value (c d : ℝ) (h₁ : 3*c^2 + 9*c - 21 = 0) (h₂ : 3*d^2 + 9*d - 21 = 0) :
  (3*c - 4) * (6*d - 8) = -22 := by
  sorry

end quadractic_roots_value_l209_209247


namespace bananas_added_l209_209440

variable (initial_bananas final_bananas added_bananas : ℕ)

-- Initial condition: There are 2 bananas initially
def initial_bananas_def : Prop := initial_bananas = 2

-- Final condition: There are 9 bananas finally
def final_bananas_def : Prop := final_bananas = 9

-- The number of bananas added to the pile
def added_bananas_def : Prop := final_bananas = initial_bananas + added_bananas

-- Proof statement: Prove that the number of bananas added is 7
theorem bananas_added (h1 : initial_bananas = 2) (h2 : final_bananas = 9) : added_bananas = 7 := by
  sorry

end bananas_added_l209_209440


namespace find_a_l209_209124

theorem find_a (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a > b) (h4 : a * b - a - b = 4) : a = 6 :=
sorry

end find_a_l209_209124


namespace ladder_distance_l209_209928

theorem ladder_distance
  (h : ∀ (a b c : ℕ), a^2 + b^2 = c^2 → a = 5 ∧ b = 12 ∧ c = 13) :
  ∃ x : ℕ, (x^2 + 12^2 = 13^2) ∧ x = 5 :=
by
  use 5
  split
  · calc
      5^2 + 12^2 = 25 + 12^2 : by rfl
      _ = 25 + 144 : by rfl
      _ = 169 : by rfl
      _ = 13^2 : by rfl
  · rfl

end ladder_distance_l209_209928


namespace tape_mounting_cost_correct_l209_209882

-- Define the given conditions as Lean definitions
def os_overhead_cost : ℝ := 1.07
def cost_per_millisecond : ℝ := 0.023
def total_cost : ℝ := 40.92
def runtime_seconds : ℝ := 1.5

-- Define the required target cost for mounting a data tape
def cost_of_data_tape : ℝ := 5.35

-- Prove that the cost of mounting a data tape is correct given the conditions
theorem tape_mounting_cost_correct :
  let computer_time_cost := cost_per_millisecond * (runtime_seconds * 1000)
  let total_cost_computed := os_overhead_cost + computer_time_cost
  cost_of_data_tape = total_cost - total_cost_computed := by
{
  sorry
}

end tape_mounting_cost_correct_l209_209882


namespace x_is_integer_l209_209310

theorem x_is_integer
  (x : ℝ)
  (h_pos : 0 < x)
  (h1 : ∃ k1 : ℤ, x^2012 = x^2001 + k1)
  (h2 : ∃ k2 : ℤ, x^2001 = x^1990 + k2) : 
  ∃ n : ℤ, x = n :=
sorry

end x_is_integer_l209_209310


namespace problem1_problem2_problem3_l209_209333

-- Problem 1
theorem problem1 (x : ℝ) (h : x = 10) : (2 / 5) * x = 4 :=
by sorry

-- Problem 2
theorem problem2 (m : ℝ) (h1 : m > 0) (h2 : (2 / 5) * m = (- (1 / 5) * m^2) + 2 * m) : m = 8 :=
by sorry

-- Problem 3
theorem problem3 (t : ℝ) (h1 : ∃ t, (2 / 5) * (32 - t) + (- (1 / 5) * t^2) + 2 * t = 16) : true :=
by sorry

end problem1_problem2_problem3_l209_209333


namespace geometric_sequence_sum_product_l209_209018

theorem geometric_sequence_sum_product {a b c : ℝ} : 
  a + b + c = 14 → 
  a * b * c = 64 → 
  (a = 8 ∧ b = 4 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 4 ∧ c = 8) :=
by
  sorry

end geometric_sequence_sum_product_l209_209018


namespace coefficient_of_term_in_binomial_expansion_l209_209728

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l209_209728


namespace mars_mission_cost_per_person_l209_209422

theorem mars_mission_cost_per_person
  (total_cost : ℕ) (number_of_people : ℕ)
  (h1 : total_cost = 50000000000) (h2 : number_of_people = 500000000) :
  (total_cost / number_of_people) = 100 := 
by
  sorry

end mars_mission_cost_per_person_l209_209422


namespace soda_cost_l209_209577

theorem soda_cost (b s f : ℕ) (h1 : 3 * b + 2 * s + 2 * f = 590) (h2 : 2 * b + 3 * s + f = 610) : s = 140 :=
sorry

end soda_cost_l209_209577


namespace man_son_age_ratio_l209_209765

-- Define the present age of the son
def son_age_present : ℕ := 22

-- Define the present age of the man based on the son's age
def man_age_present : ℕ := son_age_present + 24

-- Define the son's age in two years
def son_age_future : ℕ := son_age_present + 2

-- Define the man's age in two years
def man_age_future : ℕ := man_age_present + 2

-- Prove the ratio of the man's age to the son's age in two years is 2:1
theorem man_son_age_ratio : man_age_future / son_age_future = 2 := by
  sorry

end man_son_age_ratio_l209_209765


namespace sequence_terms_are_integers_l209_209122

theorem sequence_terms_are_integers (a : ℕ → ℕ)
  (h0 : a 0 = 1) 
  (h1 : a 1 = 2) 
  (h_recurrence : ∀ n : ℕ, (n + 3) * a (n + 2) = (6 * n + 9) * a (n + 1) - n * a n) :
  ∀ n : ℕ, ∃ k : ℤ, a n = k := 
by
  -- Initialize the proof
  sorry

end sequence_terms_are_integers_l209_209122


namespace g_at_negative_two_l209_209670

-- Function definition
def g (x : ℝ) : ℝ := 3*x^5 - 4*x^4 + 2*x^3 - 5*x^2 - x + 8

-- Theorem statement
theorem g_at_negative_two : g (-2) = -186 :=
by
  -- Proof will go here, but it is skipped with sorry
  sorry

end g_at_negative_two_l209_209670


namespace probability_at_least_two_same_l209_209412

theorem probability_at_least_two_same (n : ℕ) (H : n = 8) : 
  (∃ i j, (1 ≤ i ∧ i ≤ n) ∧ (1 ≤ j ∧ j ≤ n) ∧ i ≠ j ∧ ∀ (x : ℕ), x ∈ {i, j}) :=
by
  sorry

end probability_at_least_two_same_l209_209412


namespace number_of_dogs_l209_209305

variable {C D : ℕ}

def ratio_of_dogs_to_cats (D C : ℕ) : Prop := D = (15/7) * C

def ratio_after_additional_cats (D C : ℕ) : Prop :=
  D = 15 * (C + 8) / 11

theorem number_of_dogs (h1 : ratio_of_dogs_to_cats D C) (h2 : ratio_after_additional_cats D C) :
  D = 30 :=
by
  sorry

end number_of_dogs_l209_209305


namespace sum_of_common_ratios_l209_209250

noncomputable def geom_sequences_common_ratios (k a2 a3 b2 b3 : ℝ) (p r : ℝ) : Prop :=
  a3 = k * p^2 ∧ b3 = k * r^2 ∧ a2 = k * p ∧ b2 = k * r ∧ a3 - b3 = 5 * (a2 - b2)

theorem sum_of_common_ratios (k a2 a3 b2 b3 p r : ℝ) (h : geom_sequences_common_ratios k a2 a3 b2 b3 p r) :
  p + r = 5 :=
by
  sorry

end sum_of_common_ratios_l209_209250


namespace cost_of_cheaper_feed_l209_209898

theorem cost_of_cheaper_feed (C : ℝ) 
  (h1 : 35 * 0.36 = 12.6)
  (h2 : 18 * 0.53 = 9.54)
  (h3 : 17 * C + 9.54 = 12.6) :
  C = 0.18 := sorry

end cost_of_cheaper_feed_l209_209898


namespace fencers_count_l209_209552

theorem fencers_count (n : ℕ) (h : n * (n - 1) = 72) : n = 9 :=
sorry

end fencers_count_l209_209552


namespace number_of_games_can_buy_l209_209457

-- Definitions based on the conditions
def initial_money : ℕ := 42
def spent_money : ℕ := 10
def game_cost : ℕ := 8

-- The statement we need to prove: Mike can buy 4 games given the conditions
theorem number_of_games_can_buy : (initial_money - spent_money) / game_cost = 4 :=
by
  sorry

end number_of_games_can_buy_l209_209457


namespace base_distance_l209_209959

-- defining the lengths given in the problem
def ladder_length : ℝ := 13
def height_on_wall : ℝ := 12

-- stating that the ladder length, height on wall, and base form a right triangle using Pythagorean theorem
def base_from_wall (l : ℝ) (h : ℝ) : ℝ := Real.sqrt (l^2 - h^2)

theorem base_distance (l : ℝ) (h : ℝ) (h₀ : l = ladder_length) (h₁ : h = height_on_wall) : 
  base_from_wall l h = 5 :=
by
  sorry

end base_distance_l209_209959


namespace find_alpha_and_sin_beta_l209_209513

variable (x α β : ℝ)

def vec_a : ℝ × ℝ := (2 * Real.sin x, Real.sin x + Real.cos x)
def vec_b : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * (Real.sin x - Real.cos x))
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem find_alpha_and_sin_beta
  (hα : 0 < α ∧ α < Real.pi / 2)
  (h1 : f (α / 2) = -1)
  (hβ : 0 < β ∧ β < Real.pi / 2)
  (h2 : Real.cos (α + β) = -1 / 3) :
  α = Real.pi / 6 ∧ Real.sin β = (2 * Real.sqrt 6 + 1) / 6 :=
sorry

end find_alpha_and_sin_beta_l209_209513


namespace train_length_l209_209605

theorem train_length (T : ℕ) (S : ℕ) (conversion_factor : ℚ) (L : ℕ) 
  (hT : T = 16)
  (hS : S = 108)
  (hconv : conversion_factor = 5 / 18)
  (hL : L = 480) :
  L = ((S * conversion_factor : ℚ) * T : ℚ) :=
sorry

end train_length_l209_209605


namespace xy_sum_is_one_l209_209515

theorem xy_sum_is_one (x y : ℤ) (h1 : 2021 * x + 2025 * y = 2029) (h2 : 2023 * x + 2027 * y = 2031) : 
  x + y = 1 :=
by sorry

end xy_sum_is_one_l209_209515


namespace meetings_percentage_l209_209667

-- Define all the conditions given in the problem
def first_meeting := 60 -- duration of first meeting in minutes
def second_meeting := 2 * first_meeting -- duration of second meeting in minutes
def third_meeting := first_meeting / 2 -- duration of third meeting in minutes
def total_meeting_time := first_meeting + second_meeting + third_meeting -- total meeting time
def total_workday := 10 * 60 -- total workday time in minutes

-- Statement to prove that the percentage of workday spent in meetings is 35%
def percent_meetings : Prop := (total_meeting_time / total_workday) * 100 = 35

theorem meetings_percentage :
  percent_meetings :=
by
  sorry

end meetings_percentage_l209_209667


namespace union_of_sets_l209_209840

open Set

theorem union_of_sets :
  ∀ (P Q : Set ℕ), P = {1, 2} → Q = {2, 3} → P ∪ Q = {1, 2, 3} :=
by
  intros P Q hP hQ
  rw [hP, hQ]
  exact sorry

end union_of_sets_l209_209840


namespace coefficient_of_term_in_binomial_expansion_l209_209726

theorem coefficient_of_term_in_binomial_expansion :
  ( ∀ (x y : ℕ), (x = 3) → (y = 5) → (8.choose y) = 56 ) :=
by
  intros x y hx hy
  rw [hx, hy]
  sorry

end coefficient_of_term_in_binomial_expansion_l209_209726


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209815

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  (real.sec 150) = - (2 * real.sqrt 3) / 3 :=
by
  sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209815


namespace problem_statement_l209_209386

-- Define the arithmetic sequence and required terms
def arithmetic_seq (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Conditions
variables (a : ℕ → ℝ) (d : ℝ)
axiom seq_is_arithmetic : arithmetic_seq a d
axiom sum_of_a2_a4_a6_is_3 : a 2 + a 4 + a 6 = 3

-- Goal: Prove a1 + a3 + a5 + a7 = 4
theorem problem_statement : a 1 + a 3 + a 5 + a 7 = 4 :=
by 
  sorry

end problem_statement_l209_209386


namespace simplify_expression_l209_209450

variable (w : ℝ)

theorem simplify_expression : 3 * w + 5 - 6 * w^2 + 4 * w - 7 + 9 * w^2 = 3 * w^2 + 7 * w - 2 := by
  sorry

end simplify_expression_l209_209450


namespace valid_arrangements_count_l209_209114

-- Define the names of the individuals and the problem constraints
def names : List String := ["Wilma", "Paul", "Adam", "Betty", "Charlie", "D", "E", "F"]

-- The final number of valid arrangements
theorem valid_arrangements_count : 
  let total := (8!).toNat
  let WilmaPaulTogether := (7!).toNat * (2!).toNat
  let ABCtogether := (6!).toNat * (3!).toNat
  let bothTogether := (5!).toNat * (2!).toNat * (3!).toNat
  total - (WilmaPaulTogether + ABCtogether) + bothTogether = 25360 :=
by
  -- Decompose the problem into subcalculations
  let total := (8!).toNat
  let WilmaPaulTogether := (7!).toNat * (2!).toNat
  let ABCtogether := (6!).toNat * (3!).toNat
  let bothTogether := (5!).toNat * (2!).toNat * (3!).toNat
  sorry

end valid_arrangements_count_l209_209114


namespace chef_michel_total_pies_l209_209070

theorem chef_michel_total_pies 
  (shepherd_pie_pieces : ℕ) 
  (chicken_pot_pie_pieces : ℕ)
  (shepherd_pie_customers : ℕ) 
  (chicken_pot_pie_customers : ℕ) 
  (h1 : shepherd_pie_pieces = 4)
  (h2 : chicken_pot_pie_pieces = 5)
  (h3 : shepherd_pie_customers = 52)
  (h4 : chicken_pot_pie_customers = 80) :
  (shepherd_pie_customers / shepherd_pie_pieces) +
  (chicken_pot_pie_customers / chicken_pot_pie_pieces) = 29 :=
by {
  sorry
}

end chef_michel_total_pies_l209_209070


namespace solve_for_b_l209_209381

theorem solve_for_b (b : ℚ) (h : b + b / 4 = 5 / 2) : b = 2 := 
sorry

end solve_for_b_l209_209381


namespace raghu_investment_l209_209746

noncomputable def investment_problem (R T V : ℝ) : Prop :=
  V = 1.1 * T ∧
  T = 0.9 * R ∧
  R + T + V = 6358 ∧
  R = 2200

theorem raghu_investment
  (R T V : ℝ)
  (h1 : V = 1.1 * T)
  (h2 : T = 0.9 * R)
  (h3 : R + T + V = 6358) :
  R = 2200 :=
sorry

end raghu_investment_l209_209746


namespace frac_series_simplification_l209_209337

theorem frac_series_simplification :
  (1 - 2 + 4 - 8 + 16 - 32 + 64 - 128 : ℚ) / (1^2 + 2^2 - 4^2 + 8^2 + 16^2 - 32^2 + 64^2 - 128^2 : ℚ) = 1 / 113 := 
by
  sorry

end frac_series_simplification_l209_209337


namespace f_1986_eq_one_l209_209125

def f : ℕ → ℤ := sorry

axiom f_def (a b : ℕ) : f (a + b) = f a + f b - 2 * f (a * b) + 1
axiom f_one : f 1 = 1

theorem f_1986_eq_one : f 1986 = 1 :=
sorry

end f_1986_eq_one_l209_209125


namespace distance_from_wall_l209_209969

theorem distance_from_wall (hypotenuse height : ℝ) 
  (h1 : hypotenuse = 13) (h2 : height = 12) :
  let base := real.sqrt (hypotenuse^2 - height^2) 
  in base = 5 :=
by
  sorry

end distance_from_wall_l209_209969


namespace coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209723

noncomputable def binomial_coefficient(n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coefficient_of_x3_y5_in_expansion_of_x_plus_y_8 :
  (binomial_coefficient 8 3 = 56) :=
by
  sorry

end coefficient_of_x3_y5_in_expansion_of_x_plus_y_8_l209_209723


namespace supermarket_sales_l209_209324

theorem supermarket_sales (S_Dec : ℝ) (S_Jan : ℝ) (S_Feb : ℝ) (S_Jan_eq : S_Jan = S_Dec * (1 + x))
  (S_Feb_eq : S_Feb = S_Jan * (1 + x))
  (inc_eq : S_Feb = S_Dec + 0.24 * S_Dec) :
  x = 0.2 ∧ S_Feb = S_Dec * (1 + 0.2)^2 := by
sorry

end supermarket_sales_l209_209324


namespace integer_solution_n_l209_209447

theorem integer_solution_n 
  (n : Int) 
  (h1 : n + 13 > 15) 
  (h2 : -6 * n > -18) : 
  n = 2 := 
sorry

end integer_solution_n_l209_209447


namespace coefficient_x3_y5_in_binomial_expansion_l209_209734

theorem coefficient_x3_y5_in_binomial_expansion :
  (∃ (f : ℕ → ℕ) (k n : ℕ), 8 = n ∧ 3 = k ∧ f k = nat.choose n k ∧ 
   (nat.choose 8 3 = 56)) → true :=
by
  sorry

end coefficient_x3_y5_in_binomial_expansion_l209_209734


namespace condition_for_equation_l209_209653

theorem condition_for_equation (a b c d : ℝ) 
  (h : (a^2 + b) / (b + c^2) = (c^2 + d) / (d + a^2)) : 
  a = c ∨ a^2 + d + 2 * b = 0 :=
by
  sorry

end condition_for_equation_l209_209653


namespace expected_flips_is_four_l209_209254

noncomputable def expected_flips_to_second_tails : ℕ :=
  let E_Y := 2 in
  E_Y + E_Y

theorem expected_flips_is_four : expected_flips_to_second_tails = 4 :=
  by
  -- sorry is added to skip proof
  sorry

end expected_flips_is_four_l209_209254


namespace miki_sandcastle_height_correct_l209_209474

namespace SandcastleHeight

def sister_sandcastle_height := 0.5
def difference_in_height := 0.3333333333333333
def miki_sandcastle_height := sister_sandcastle_height + difference_in_height

theorem miki_sandcastle_height_correct : miki_sandcastle_height = 0.8333333333333333 := by
  unfold miki_sandcastle_height sister_sandcastle_height difference_in_height
  simp
  sorry

end SandcastleHeight

end miki_sandcastle_height_correct_l209_209474


namespace watermelon_cost_100_l209_209974

variable (a b : ℕ) -- costs of one watermelon and one melon respectively
variable (x : ℕ) -- number of watermelons in the container

theorem watermelon_cost_100 :
  (∀ x, (1 : ℚ) = x / 160 + (150 - x) / 120 ∧ 120 * a = 30 * b ∧ 120 * a + 30 * b = 24000 ∧ x = 120) →
  a = 100 :=
by
  intro h
  sorry

end watermelon_cost_100_l209_209974


namespace greatest_negative_root_l209_209495

noncomputable def sine (x : ℝ) : ℝ := Real.sin (Real.pi * x)
noncomputable def cosine (x : ℝ) : ℝ := Real.cos (2 * Real.pi * x)

theorem greatest_negative_root :
  ∀ (x : ℝ), (x < 0 ∧ (sine x - cosine x) / ((sine x + 1)^2 + (Real.cos (Real.pi * x))^2) = 0) → 
    x ≤ -7/6 :=
by
  sorry

end greatest_negative_root_l209_209495


namespace perpendicular_bisector_AC_circumcircle_eqn_l209_209369

/-- Given vertices of triangle ABC, prove the equation of the perpendicular bisector of side AC --/
theorem perpendicular_bisector_AC (A B C D : ℝ×ℝ) (hA: A = (0, 2)) (hC: C = (4, 0)) (hD: D = (2, 1)) :
  ∃ k b, (k = 2) ∧ (b = -3) ∧ (∀ x y, y = k * x + b ↔ 2 * x - y - 3 = 0) :=
sorry

/-- Given vertices of triangle ABC, prove the equation of the circumcircle --/
theorem circumcircle_eqn (A B C D E F : ℝ×ℝ) (hA: A = (0, 2)) (hB: B = (6, 4)) (hC: C = (4, 0)) :
  ∃ k, k = 10 ∧ 
  (∀ x y, (x - 3) ^ 2 + (y - 3) ^ 2 = k ↔ x ^ 2 + y ^ 2 - 6 * x - 2 * y + 8 = 0) :=
sorry

end perpendicular_bisector_AC_circumcircle_eqn_l209_209369


namespace ladder_base_distance_l209_209949

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l209_209949


namespace ladder_base_distance_l209_209920

theorem ladder_base_distance (h l : ℕ) (ladder_hypotenuse : h = 13) (ladder_height : l = 12) : 
  (13^2 - 12^2) = 5^2 :=
by
  sorry

end ladder_base_distance_l209_209920


namespace ladder_base_distance_l209_209927

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l209_209927


namespace slope_of_line_through_points_l209_209451

theorem slope_of_line_through_points :
  let x1 := 1
  let y1 := 3
  let x2 := 5
  let y2 := 7
  let m := (y2 - y1) / (x2 - x1)
  m = 1 := by
  sorry

end slope_of_line_through_points_l209_209451


namespace ladder_base_distance_l209_209950

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l209_209950


namespace lifting_ratio_after_gain_l209_209995

def intial_lifting_total : ℕ := 2200
def initial_bodyweight : ℕ := 245
def percentage_gain_total : ℕ := 15
def weight_gain : ℕ := 8

theorem lifting_ratio_after_gain :
  (intial_lifting_total * (100 + percentage_gain_total) / 100) / (initial_bodyweight + weight_gain) = 10 := by
  sorry

end lifting_ratio_after_gain_l209_209995


namespace term_containing_1x2_is_10_find_n_equality_l209_209208

-- Definitions based on the problem's conditions
def general_term (a b : ℤ) (x : ℤ) (n r : ℕ) : ℤ := (Nat.choose n r) * a ^ (n - r) * b ^ r * x ^ (n - 2 * r)

axiom sum_binom_lt_coeff (sum_binom : ℕ) (coeff3 : ℕ) : sum_binom + 28 = coeff3

-- Axioms (the correct answers derived from the given problems)
axiom term_containing_1x2_expansion : ∀ (x : ℤ), general_term 2 (1 / x) x 5 4 = 10 / x ^ 2
axiom coeff_third_term_expansion : ∀ (x : ℤ) (n : ℕ), general_term (sqrt x) (2 / x) x n 2 = 4 * (Nat.choose n 2) * x ^ (n / 2 - 3)

-- The two proofs as Lean statements
theorem term_containing_1x2_is_10 (x : ℤ) : general_term 2 (1 / x) x 5 4 = 10 / x ^ 2 := 
  term_containing_1x2_expansion x

theorem find_n_equality (n : ℕ) (sum_binom : ℕ) : 
  let coeff3 := 4 * (Nat.choose n 2) * 1
  sum_binom + 28 = coeff3 → Nat.choose n 2 = 15 → n = 6 := sorry

end term_containing_1x2_is_10_find_n_equality_l209_209208


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209786

open Real

theorem sec_150_eq_neg_2_sqrt_3_div_3 : sec 150 = - (2 * sqrt 3) / 3 := sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209786


namespace school_fee_correct_l209_209866

-- Definitions
def mother_fifty_bills : ℕ := 1
def mother_twenty_bills : ℕ := 2
def mother_ten_bills : ℕ := 3

def father_fifty_bills : ℕ := 4
def father_twenty_bills : ℕ := 1
def father_ten_bills : ℕ := 1

def total_fifty_bills : ℕ := mother_fifty_bills + father_fifty_bills
def total_twenty_bills : ℕ := mother_twenty_bills + father_twenty_bills
def total_ten_bills : ℕ := mother_ten_bills + father_ten_bills

def value_fifty_bills : ℕ := 50 * total_fifty_bills
def value_twenty_bills : ℕ := 20 * total_twenty_bills
def value_ten_bills : ℕ := 10 * total_ten_bills

-- Theorem
theorem school_fee_correct :
  value_fifty_bills + value_twenty_bills + value_ten_bills = 350 :=
by
  sorry

end school_fee_correct_l209_209866


namespace manny_has_more_10_bills_than_mandy_l209_209539

theorem manny_has_more_10_bills_than_mandy :
  let mandy_bills_20 := 3
  let manny_bills_50 := 2
  let mandy_total_money := 20 * mandy_bills_20
  let manny_total_money := 50 * manny_bills_50
  let mandy_10_bills := mandy_total_money / 10
  let manny_10_bills := manny_total_money / 10
  mandy_10_bills < manny_10_bills →
  manny_10_bills - mandy_10_bills = 4 := sorry

end manny_has_more_10_bills_than_mandy_l209_209539


namespace fireworks_display_l209_209988

def year_fireworks : Nat := 4 * 6
def letters_fireworks : Nat := 12 * 5
def boxes_fireworks : Nat := 50 * 8

theorem fireworks_display : year_fireworks + letters_fireworks + boxes_fireworks = 484 := by
  have h1 : year_fireworks = 24 := rfl
  have h2 : letters_fireworks = 60 := rfl
  have h3 : boxes_fireworks = 400 := rfl
  calc
    year_fireworks + letters_fireworks + boxes_fireworks 
        = 24 + 60 + 400 := by rw [h1, h2, h3]
    _ = 484 := rfl

end fireworks_display_l209_209988


namespace negation_necessary_not_sufficient_l209_209091

theorem negation_necessary_not_sufficient (p q : Prop) : 
  ((¬ p) → ¬ (p ∨ q)) := 
sorry

end negation_necessary_not_sufficient_l209_209091


namespace johns_out_of_pocket_expense_l209_209531

theorem johns_out_of_pocket_expense
  (computer_cost : ℕ)
  (accessories_cost : ℕ)
  (playstation_value : ℕ)
  (playstation_sold_percent_less : ℕ) :
  computer_cost = 700 →
  accessories_cost = 200 →
  playstation_value = 400 →
  playstation_sold_percent_less = 20 →
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100) in
  let total_cost := computer_cost + accessories_cost in
  let pocket_expense := total_cost - playstation_sold_price in
  pocket_expense = 580 :=
by
  intros h1 h2 h3 h4
  let playstation_sold_price := playstation_value - (playstation_sold_percent_less * playstation_value / 100)
  let total_cost := computer_cost + accessories_cost
  let pocket_expense := total_cost - playstation_sold_price
  sorry

end johns_out_of_pocket_expense_l209_209531


namespace trapezoid_extensions_meet_at_acute_angle_l209_209272

theorem trapezoid_extensions_meet_at_acute_angle
  (A B C D : Point ℝ)
  (h_isosceles : IsIsoscelesTrapezoid A B C D)
  (h_base_AB : distance A B = 2)
  (h_base_CD : distance C D = 11) :
  IsAcuteAngle (LineThrough A B).extension (LineThrough C D).extension :=
sorry

end trapezoid_extensions_meet_at_acute_angle_l209_209272


namespace range_of_n_l209_209686

theorem range_of_n (m n : ℝ) (h : (m^2 - 2 * m)^2 + 4 * m^2 - 8 * m + 6 - n = 0) : n ≥ 3 :=
sorry

end range_of_n_l209_209686


namespace additional_people_to_halve_speed_l209_209195

variables (s : ℕ → ℝ)
variables (x : ℕ)

-- Given conditions
axiom speed_with_200_people : s 200 = 500
axiom speed_with_400_people : s 400 = 125
axiom speed_halved : ∀ n, s (n + x) = s n / 2

theorem additional_people_to_halve_speed : x = 100 :=
by
  sorry

end additional_people_to_halve_speed_l209_209195


namespace total_pupils_in_school_l209_209524

theorem total_pupils_in_school (girls boys : ℕ) (h_girls : girls = 542) (h_boys : boys = 387) : girls + boys = 929 := by
  sorry

end total_pupils_in_school_l209_209524


namespace lambs_traded_for_goat_l209_209867

-- Definitions for the given conditions
def initial_lambs : ℕ := 6
def babies_per_lamb : ℕ := 2 -- each of 2 lambs had 2 babies
def extra_babies : ℕ := 2 * babies_per_lamb
def extra_lambs : ℕ := 7
def current_lambs : ℕ := 14

-- Proof statement for the number of lambs traded
theorem lambs_traded_for_goat : initial_lambs + extra_babies + extra_lambs - current_lambs = 3 :=
by
  sorry

end lambs_traded_for_goat_l209_209867


namespace window_width_correct_l209_209233

def total_width_window (x : ℝ) : ℝ :=
  let pane_width := 4 * x
  let num_panes_per_row := 4
  let num_borders := 5
  num_panes_per_row * pane_width + num_borders * 3

theorem window_width_correct (x : ℝ) :
  total_width_window x = 16 * x + 15 := sorry

end window_width_correct_l209_209233


namespace sequence_identity_l209_209207

noncomputable def a_n (n : ℕ) : ℝ := n + 1
noncomputable def b_n (n : ℕ) : ℝ := 2 * 3^(n-1)
noncomputable def S_n (n : ℕ) : ℝ := (n * (n+1)) / 2  -- Sum of first n terms of arithmetic sequence
noncomputable def T_n (n : ℕ) : ℝ := 2 * (3^n - 1) / 2  -- Sum of first n terms of geometric sequence
noncomputable def c_n (n : ℕ) : ℝ := 2 * a_n n / b_n n
noncomputable def C_n (n : ℕ) : ℝ := (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))

theorem sequence_identity :
  a_n 1 = b_n 1 ∧
  2 * a_n 2 = b_n 2 ∧
  S_n 2 + T_n 2 = 13 ∧
  2 * S_n 3 = b_n 3 →
  (∀ n : ℕ, a_n n = n + 1 ∧ b_n n = 2 * 3^(n-1)) ∧
  (∀ n : ℕ, C_n n = (15/2) - ((2 * n + 5) / (2 * 3^(n-1)))) :=
sorry

end sequence_identity_l209_209207


namespace cyclists_meet_after_24_minutes_l209_209498

noncomputable def meet_time (D : ℝ) (vm vb : ℝ) : ℝ :=
  D / (2.5 * D - 12)

theorem cyclists_meet_after_24_minutes
  (D vm vb : ℝ)
  (h_vm : 1/3 * vm + 2 = D/2)
  (h_vb : 1/2 * vb = D/2 - 3) :
  meet_time D vm vb = 24 :=
by
  sorry

end cyclists_meet_after_24_minutes_l209_209498


namespace cannot_be_covered_by_dominoes_l209_209076

-- Definitions for each board
def board_3x4_squares : ℕ := 3 * 4
def board_3x5_squares : ℕ := 3 * 5
def board_4x4_one_removed_squares : ℕ := 4 * 4 - 1
def board_5x5_squares : ℕ := 5 * 5
def board_6x3_squares : ℕ := 6 * 3

-- Parity check
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Mathematical proof problem statement
theorem cannot_be_covered_by_dominoes :
  ¬ is_even board_3x5_squares ∧
  ¬ is_even board_4x4_one_removed_squares ∧
  ¬ is_even board_5x5_squares :=
by
  -- Checking the conditions that must hold
  sorry

end cannot_be_covered_by_dominoes_l209_209076


namespace calculate_m_l209_209649

theorem calculate_m (m : ℕ) : 9^4 = 3^m → m = 8 :=
by
  sorry

end calculate_m_l209_209649


namespace find_third_number_l209_209351
open BigOperators

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

def LCM_of_three (a b c : ℕ) : ℕ := LCM (LCM a b) c

theorem find_third_number (n : ℕ) (h₁ : LCM 15 25 = 75) (h₂ : LCM_of_three 15 25 n = 525) : n = 7 :=
by 
  sorry

end find_third_number_l209_209351


namespace intersection_of_M_and_N_l209_209864

def M : Set ℝ := {x | (x + 3) * (x - 2) < 0}
def N : Set ℝ := {x | 1 ≤ x ∧ x ≤ 3}

theorem intersection_of_M_and_N : M ∩ N = {x | 1 ≤ x ∧ x < 2} :=
by
  sorry

end intersection_of_M_and_N_l209_209864


namespace zacks_friends_l209_209916

theorem zacks_friends (initial_marbles : ℕ) (marbles_kept : ℕ) (marbles_per_friend : ℕ) 
  (h_initial : initial_marbles = 65) (h_kept : marbles_kept = 5) 
  (h_per_friend : marbles_per_friend = 20) : (initial_marbles - marbles_kept) / marbles_per_friend = 3 :=
by
  sorry

end zacks_friends_l209_209916


namespace largest_of_decimals_l209_209743

theorem largest_of_decimals :
  let a := 0.993
  let b := 0.9899
  let c := 0.990
  let d := 0.989
  let e := 0.9909
  a > b ∧ a > c ∧ a > d ∧ a > e :=
by
  sorry

end largest_of_decimals_l209_209743


namespace binomial_coefficient_x3y5_in_expansion_l209_209730

theorem binomial_coefficient_x3y5_in_expansion : 
  (∑ k in Finset.range 9, (Nat.choose 8 k) * (x^k) * (y^(8-k))) = 56 :=
by
  -- Proof omitted
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209730


namespace exists_multiple_with_digits_0_or_1_l209_209679

theorem exists_multiple_with_digits_0_or_1 (n : ℕ) (hn : 0 < n) :
  ∃ k : ℕ, (k % n = 0) ∧ (∀ digit ∈ k.digits 10, digit = 0 ∨ digit = 1) ∧ (k.digits 10).length ≤ n :=
sorry

end exists_multiple_with_digits_0_or_1_l209_209679


namespace percentage_paid_X_vs_Y_l209_209445

theorem percentage_paid_X_vs_Y (X Y : ℝ) (h1 : X + Y = 528) (h2 : Y = 240) :
  ((X / Y) * 100) = 120 :=
by
  sorry

end percentage_paid_X_vs_Y_l209_209445


namespace same_remainder_division_l209_209086

theorem same_remainder_division (k r a b c d : ℕ) 
  (h_k_pos : 0 < k)
  (h_nonzero_r : 0 < r)
  (h_r_lt_k : r < k)
  (a_def : a = 2613)
  (b_def : b = 2243)
  (c_def : c = 1503)
  (d_def : d = 985)
  (h_a : a % k = r)
  (h_b : b % k = r)
  (h_c : c % k = r)
  (h_d : d % k = r) : 
  k = 74 ∧ r = 23 := 
by
  sorry

end same_remainder_division_l209_209086


namespace rotation_phenomena_l209_209557

/-- 
The rotation of the hour hand fits the definition of rotation since it turns around 
the center of the clock, covering specific angles as time passes.
-/
def is_rotation_of_hour_hand : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The rotation of the Ferris wheel fits the definition of rotation since it turns around 
its central axis, making a complete circle.
-/
def is_rotation_of_ferris_wheel : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The annual decline of the groundwater level does not fit the definition of rotation 
since it is a vertical movement (translation).
-/
def is_not_rotation_of_groundwater_level : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
The movement of the robots on the conveyor belt does not fit the definition of rotation 
since it is a linear/translational movement.
-/
def is_not_rotation_of_robots_on_conveyor : Prop :=
  true -- we assume this is true based on the problem condition

/-- 
Proof that the phenomena which belong to rotation are exactly the rotation of the hour hand 
and the rotation of the Ferris wheel.
-/
theorem rotation_phenomena :
  is_rotation_of_hour_hand ∧ 
  is_rotation_of_ferris_wheel ∧ 
  is_not_rotation_of_groundwater_level ∧ 
  is_not_rotation_of_robots_on_conveyor →
  "①②" = "①②" :=
by
  intro h
  sorry

end rotation_phenomena_l209_209557


namespace triangle_angle_extension_l209_209527

theorem triangle_angle_extension :
  ∀ (BAC ABC BCA CDB DBC : ℝ),
  180 = BAC + ABC + BCA →
  CDB = BAC + ABC →
  DBC = BAC + BCA →
  (CDB + DBC) / (BAC + ABC) = 2 :=
by
  intros BAC ABC BCA CDB DBC h1 h2 h3
  sorry

end triangle_angle_extension_l209_209527


namespace tan_alpha_eq_one_l209_209516

open Real

theorem tan_alpha_eq_one (α : ℝ) (h : (sin α + cos α) / (2 * sin α - cos α) = 2) : tan α = 1 := 
by
  sorry

end tan_alpha_eq_one_l209_209516


namespace evaluate_expression_l209_209680

theorem evaluate_expression (m n : ℤ) (hm : m = 2) (hn : n = -3) : (m + n) ^ 2 - 2 * m * (m + n) = 5 := by
  -- Proof skipped
  sorry

end evaluate_expression_l209_209680


namespace boy_and_girl_roles_l209_209165

-- Definitions of the conditions
def Sasha_says_boy : Prop := True
def Zhenya_says_girl : Prop := True
def at_least_one_lying (sasha_boy zhenya_girl : Prop) : Prop := 
  (sasha_boy = False) ∨ (zhenya_girl = False)

-- Theorem statement
theorem boy_and_girl_roles (sasha_boy : Prop) (zhenya_girl : Prop) 
  (H1 : Sasha_says_boy) (H2 : Zhenya_says_girl) (H3 : at_least_one_lying sasha_boy zhenya_girl) :
  sasha_boy = False ∧ zhenya_girl = True :=
sorry

end boy_and_girl_roles_l209_209165


namespace constant_term_binomial_expansion_l209_209350

theorem constant_term_binomial_expansion : ∃ T, (∀ x : ℝ, T = (2 * x - 1 / (2 * x)) ^ 6) ∧ T = -20 := 
by
  sorry

end constant_term_binomial_expansion_l209_209350


namespace perfect_squares_example_l209_209587

def isPerfectSquare (n: ℕ) : Prop := ∃ m: ℕ, m * m = n

theorem perfect_squares_example :
  let a := 10430
  let b := 3970
  let c := 2114
  let d := 386
  isPerfectSquare (a + b) ∧
  isPerfectSquare (a + c) ∧
  isPerfectSquare (a + d) ∧
  isPerfectSquare (b + c) ∧
  isPerfectSquare (b + d) ∧
  isPerfectSquare (c + d) ∧
  isPerfectSquare (a + b + c + d) :=
by
  -- Proof steps go here
  sorry

end perfect_squares_example_l209_209587


namespace banana_production_total_l209_209591

def banana_production (nearby_island_production : ℕ) (jakies_multiplier : ℕ) : ℕ :=
  nearby_island_production + (jakies_multiplier * nearby_island_production)

theorem banana_production_total
  (nearby_island_production : ℕ)
  (jakies_multiplier : ℕ)
  (h1 : nearby_island_production = 9000)
  (h2 : jakies_multiplier = 10)
  : banana_production nearby_island_production jakies_multiplier = 99000 :=
by
  sorry

end banana_production_total_l209_209591


namespace Shawn_scored_6_points_l209_209868

theorem Shawn_scored_6_points
  (points_per_basket : ℤ)
  (matthew_points : ℤ)
  (total_baskets : ℤ)
  (h1 : points_per_basket = 3)
  (h2 : matthew_points = 9)
  (h3 : total_baskets = 5)
  : (∃ shawn_points : ℤ, shawn_points = 6) :=
by
  sorry

end Shawn_scored_6_points_l209_209868


namespace prime_numbers_solution_l209_209629

theorem prime_numbers_solution (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q)
  (h1 : Nat.Prime (p + q)) (h2 : Nat.Prime (p^2 + q^2 - q)) : p = 3 ∧ q = 2 :=
by
  sorry

end prime_numbers_solution_l209_209629


namespace identity_map_a_plus_b_l209_209860

theorem identity_map_a_plus_b (a b : ℝ) (h : ∀ x ∈ ({-1, b / a, 1} : Set ℝ), x ∈ ({a, b, b - a} : Set ℝ)) : a + b = -1 ∨ a + b = 1 :=
by
  sorry

end identity_map_a_plus_b_l209_209860


namespace sum_of_arithmetic_sequence_l209_209526

theorem sum_of_arithmetic_sequence (S : ℕ → ℕ) 
  (h₁ : S 4 = 2) 
  (h₂ : S 8 = 6) 
  : S 12 = 12 := 
by
  sorry

end sum_of_arithmetic_sequence_l209_209526


namespace equivalent_angle_in_radians_l209_209499

noncomputable def degrees_to_radians (d : ℝ) : ℝ := d * π / 180

theorem equivalent_angle_in_radians (α : ℝ) (h₁ : α = 2022) (h₂ : 0 < degrees_to_radians 222 ∧ degrees_to_radians 222 < 2 * π) :
  ∃ β, β = degrees_to_radians 222 ∧ β ∈ (0, 2 * π) :=
by {
  use degrees_to_radians 222,
  split,
  { refl },
  { exact h₂ },
}

end equivalent_angle_in_radians_l209_209499


namespace parametric_curve_intersects_l209_209611

noncomputable def curve_crosses_itself : Prop :=
  let t1 := Real.sqrt 11
  let t2 := -Real.sqrt 11
  let x (t : ℝ) := t^3 - t + 1
  let y (t : ℝ) := t^3 - 11*t + 11
  (x t1 = 10 * Real.sqrt 11 + 1) ∧ (y t1 = 11) ∧
  (x t2 = 10 * Real.sqrt 11 + 1) ∧ (y t2 = 11)

theorem parametric_curve_intersects : curve_crosses_itself :=
by
  sorry

end parametric_curve_intersects_l209_209611


namespace find_third_number_l209_209008

theorem find_third_number (x y z : ℝ) 
  (h1 : y = 3 * x - 7)
  (h2 : z = 2 * x + 2)
  (h3 : x + y + z = 168) : z = 60 :=
sorry

end find_third_number_l209_209008


namespace trips_needed_l209_209616

def barbieCapacity : Nat := 4
def brunoCapacity : Nat := 8
def totalCoconuts : Nat := 144

theorem trips_needed : (totalCoconuts / (barbieCapacity + brunoCapacity)) = 12 := by
  sorry

end trips_needed_l209_209616


namespace calculate_fraction_l209_209062

theorem calculate_fraction : 
  let a := 0.3
  let b := 0.03
  a = 3 * 10^(-1) ∧ b = 3 * 10^(-2)
  → (a^4 / b^3) = 300 :=
by
  intros a b ha hb
  rw [ha, hb]
  sorry

end calculate_fraction_l209_209062


namespace trigonometric_ratio_l209_209127

open Real

theorem trigonometric_ratio (u v : ℝ) (h1 : sin u / sin v = 4) (h2 : cos u / cos v = 1 / 3) :
  (sin (2 * u) / sin (2 * v) + cos (2 * u) / cos (2 * v)) = 389 / 381 :=
by
  sorry

end trigonometric_ratio_l209_209127


namespace solve_for_x_l209_209459

theorem solve_for_x (x : ℝ) (h : 0.009 / x = 0.1) : x = 0.09 :=
sorry

end solve_for_x_l209_209459


namespace total_banana_produce_correct_l209_209590

-- Defining the conditions as variables and constants
def B_nearby : ℕ := 9000
def B_Jakies : ℕ := 10 * B_nearby
def T : ℕ := B_nearby + B_Jakies

-- Theorem statement
theorem total_banana_produce_correct : T = 99000 := by
  sorry  -- Proof placeholder

end total_banana_produce_correct_l209_209590


namespace expression_evaluation_l209_209490

def e1 : ℤ := 72 + (120 / 15) + (15 * 12) - 250 - (480 / 8)

theorem expression_evaluation : e1 = -50 :=
by
  sorry

end expression_evaluation_l209_209490


namespace original_selling_price_l209_209776

variable (P : ℝ)

def SP1 := 1.10 * P
def P_new := 0.90 * P
def SP2 := 1.17 * P
def price_diff := SP2 - SP1

theorem original_selling_price : price_diff = 49 → SP1 = 770 :=
by
  sorry

end original_selling_price_l209_209776


namespace probability_of_region_F_l209_209770

theorem probability_of_region_F
  (pD pE pG pF : ℚ)
  (hD : pD = 3/8)
  (hE : pE = 1/4)
  (hG : pG = 1/8)
  (hSum : pD + pE + pF + pG = 1) : pF = 1/4 :=
by
  -- we can perform the steps as mentioned in the solution without actually executing them
  sorry

end probability_of_region_F_l209_209770


namespace positive_diff_of_supplementary_angles_l209_209144

theorem positive_diff_of_supplementary_angles (x : ℝ) (h : 5 * x + 3 * x = 180) : 
  abs ((5 * x - 3 * x)) = 45 := by
  sorry

end positive_diff_of_supplementary_angles_l209_209144


namespace purchases_per_customer_l209_209702

noncomputable def number_of_customers_in_cars (num_cars : ℕ) (customers_per_car : ℕ) : ℕ :=
  num_cars * customers_per_car

def total_sales (sports_store_sales : ℕ) (music_store_sales : ℕ) : ℕ :=
  sports_store_sales + music_store_sales

theorem purchases_per_customer {num_cars : ℕ} {customers_per_car : ℕ} {sports_store_sales : ℕ} {music_store_sales : ℕ}
    (h1 : num_cars = 10)
    (h2 : customers_per_car = 5)
    (h3 : sports_store_sales = 20)
    (h4: music_store_sales = 30) :
    (total_sales sports_store_sales music_store_sales / number_of_customers_in_cars num_cars customers_per_car) = 1 :=
by
  sorry

end purchases_per_customer_l209_209702


namespace equation_one_solution_equation_two_solution_l209_209684

theorem equation_one_solution (x : ℝ) (h : 2 * (2 - x) - 5 * (2 - x) = 9) : x = 5 :=
sorry

theorem equation_two_solution (x : ℝ) (h : x / 3 - (3 * x - 1) / 6 = 1) : x = -5 :=
sorry

end equation_one_solution_equation_two_solution_l209_209684


namespace contractor_realized_work_done_after_20_days_l209_209170

-- Definitions based on conditions
variable (W w : ℝ)  -- W is total work, w is work per person per day
variable (d : ℝ)  -- d is the number of days we want to find

-- Conditions transformation into Lean definitions
def initial_work_done_in_d_days := 10 * w * d = (1 / 4) * W
def remaining_work_done_in_75_days := 8 * w * 75 = (3 / 4) * W
def total_work := (10 * w * d) + (8 * w * 75) = W

-- Proof statement we need to prove
theorem contractor_realized_work_done_after_20_days :
  initial_work_done_in_d_days W w d ∧ 
  remaining_work_done_in_75_days W w → 
  total_work W w d →
  d = 20 := by
  sorry

end contractor_realized_work_done_after_20_days_l209_209170


namespace minimize_x_2y_l209_209671

noncomputable def minimum_value_x_2y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 3 / (x + 2) + 3 / (y + 2) = 1) : ℝ :=
  x + 2 * y

theorem minimize_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 3 / (x + 2) + 3 / (y + 2) = 1) :
  minimum_value_x_2y x y hx hy h = 3 + 6 * Real.sqrt 2 :=
sorry

end minimize_x_2y_l209_209671


namespace probability_divisible_by_4_l209_209172

theorem probability_divisible_by_4:
  let M : ℕ :=
    sorry -- Define a four-digit positive integer with ones digit 6
  in (M % 10 = 6 ∧ (M / 1000 ≥ 1) ∧ (M / 1000 < 10)) →
  ∃ (p q : ℕ), p/q = 2/5 ∧ 
  ∀ x y z : ℕ, 0 ≤ x ∧ x < 10 ∧ 0 ≤ y ∧ y < 10 ∧ 0 ≤ z ∧ z < 10 →
  (10*z + 6) % 4 = 0 →
  p/q = 2/5 :=
sorry

end probability_divisible_by_4_l209_209172


namespace martha_butterflies_total_l209_209255

theorem martha_butterflies_total
  (B : ℕ) (Y : ℕ) (black : ℕ)
  (h1 : B = 4)
  (h2 : Y = B / 2)
  (h3 : black = 5) :
  B + Y + black = 11 :=
by {
  -- skip proof 
  sorry 
}

end martha_butterflies_total_l209_209255


namespace john_out_of_pocket_l209_209530

-- Define the conditions
def computer_cost : ℕ := 700
def accessories_cost : ℕ := 200
def playstation_value : ℕ := 400
def sale_discount : ℚ := 0.2

-- Define the total cost of the computer and accessories
def total_cost : ℕ := computer_cost + accessories_cost

-- Define the selling price of the PlayStation
def selling_price : ℕ := playstation_value - (playstation_value * sale_discount).to_nat

-- Define the amount out of John's pocket
def out_of_pocket : ℕ := total_cost - selling_price

-- The proof goal
theorem john_out_of_pocket : out_of_pocket = 580 :=
by
  sorry

end john_out_of_pocket_l209_209530


namespace largest_base_6_five_digits_l209_209738

-- Define the base-6 number 55555 in base 10
def base_6_to_base_10 (n : Nat) : Nat :=
  let d0 := (n % 10)
  let d1 := (n / 10) % 10
  let d2 := (n / 100) % 10
  let d3 := (n / 1000) % 10
  let d4 := (n / 10000) % 10
  d4 * 6^4 + d3 * 6^3 + d2 * 6^2 + d1 * 6^1 + d0 * 6^0

theorem largest_base_6_five_digits : base_6_to_base_10 55555 = 7775 := by
  sorry

end largest_base_6_five_digits_l209_209738


namespace rearrange_expression_l209_209238

theorem rearrange_expression :
  1 - 2 - 3 - 4 - (5 - 6 - 7) = 0 :=
by
  sorry

end rearrange_expression_l209_209238


namespace watermelon_cost_l209_209977

-- Define the problem conditions
def container_full_conditions (w m : ℕ) : Prop :=
  w + m = 150 ∧ (w / 160) + (m / 120) = 1

def equal_total_values (w m w_value m_value : ℕ) : Prop :=
  w * w_value = m * m_value ∧ w * w_value + m * m_value = 24000

-- Define the proof problem
theorem watermelon_cost (w m w_value m_value : ℕ) (hw : container_full_conditions w m) (hv : equal_total_values w m w_value m_value) :
  w_value = 100 :=
by
  -- precise proof goes here
  sorry

end watermelon_cost_l209_209977


namespace find_values_of_x_and_y_l209_209655

-- Define the conditions
def first_condition (x : ℝ) : Prop := 0.75 / x = 5 / 7
def second_condition (y : ℝ) : Prop := y / 19 = 11 / 3

-- Define the main theorem to prove
theorem find_values_of_x_and_y (x y : ℝ) (h1 : first_condition x) (h2 : second_condition y) :
  x = 1.05 ∧ y = 209 / 3 := 
by 
  sorry

end find_values_of_x_and_y_l209_209655


namespace part1_part2_l209_209371

noncomputable def f (x m : ℝ) : ℝ := x^2 - 2*m*x + 2 - m

theorem part1 (m : ℝ) : (∀ x : ℝ, f x m ≥ x - m*x) → -7 ≤ m ∧ m ≤ 1 :=
by
  sorry

theorem part2 (m : ℝ) : (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x m) → m ≤ 1 :=
by
  sorry

end part1_part2_l209_209371


namespace proposition_4_l209_209182

theorem proposition_4 (x y ε : ℝ) (h1 : |x - 2| < ε) (h2 : |y - 2| < ε) : |x - y| < 2 * ε :=
by
  sorry

end proposition_4_l209_209182


namespace find_nearest_integer_x_minus_y_l209_209517

variable (x y : ℝ)

theorem find_nearest_integer_x_minus_y
  (h1 : abs x + y = 5)
  (h2 : abs x * y - x^3 = 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0) :
  |x - y| = 5 := sorry

end find_nearest_integer_x_minus_y_l209_209517


namespace number_of_dots_in_120_circles_l209_209048

theorem number_of_dots_in_120_circles :
  ∃ n : ℕ, (n = 14) ∧ (∀ m : ℕ, m * (m + 1) / 2 + m ≤ 120 → m ≤ n) :=
by
  sorry

end number_of_dots_in_120_circles_l209_209048


namespace count_int_values_not_satisfying_ineq_l209_209831

theorem count_int_values_not_satisfying_ineq :
  ∃ (s : Finset ℤ), (∀ x ∈ s, 3 * x^2 + 14 * x + 8 ≤ 17) ∧ (s.card = 10) :=
by
  sorry

end count_int_values_not_satisfying_ineq_l209_209831


namespace ladder_distance_from_wall_l209_209941

theorem ladder_distance_from_wall (h : ℝ) (d : ℝ) (x : ℝ) 
  (hypotenuse_len : h = 13) 
  (height_on_wall : d = 12) :
  x = 5 :=
by
  -- Definitions for the equation
  have h_squared := h * h
  have d_squared := d * d 
  have x_squared := x * x
  -- Pythagorean theorem: h^2 = d^2 + x^2
  have pythagorean_theorem : h_squared = d_squared + x_squared

  -- Given: h = 13 and d = 12
  have h_val : h_squared = 13 * 13 from calc
    h_squared = (13 : ℝ) * 13                    : by rw [hypotenuse_len]
              ... = (169 : ℝ)                    : by norm_num
              
  have d_val : d_squared = 12 * 12 from calc 
    d_squared = (12 : ℝ) * 12                    : by rw [height_on_wall]
              ... = (144 : ℝ)                    : by norm_num
              
  have equation : 169 = 144 + x_squared := by
    rw [h_val, d_val, pythagorean_theorem]
    sorry

  -- Solving for x
  suffices : x * x = 25 from this
  -- Then |x| = 5, so x = 5
  sorry

end ladder_distance_from_wall_l209_209941


namespace find_a_div_b_l209_209246

theorem find_a_div_b (a b : ℝ) (h : a ≠ b) (h_eq : a / b + (a + 6 * b) / (b + 6 * a) = 3) : 
  a / b = (8 + Real.sqrt 46) / 6 ∨ a / b = (8 - Real.sqrt 46) / 6 :=
by 
  sorry

end find_a_div_b_l209_209246


namespace stationery_sales_other_l209_209553

theorem stationery_sales_other (p e n : ℝ) (h_p : p = 25) (h_e : e = 30) (h_n : n = 20) :
    100 - (p + e + n) = 25 :=
by
  sorry

end stationery_sales_other_l209_209553


namespace solve_for_z_l209_209654

theorem solve_for_z (x y : ℝ) (z : ℝ) (h : 2 / x - 1 / y = 3 / z) : 
  z = (2 * y - x) / 3 :=
by
  sorry

end solve_for_z_l209_209654


namespace inequality_proof_l209_209204

variable (a b c : ℝ)

-- Conditions
def conditions : Prop := a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 + c^2 = 14

-- Statement to prove
theorem inequality_proof (h : conditions a b c) : 
  a^5 + (1/8) * b^5 + (1/27) * c^5 ≥ 14 := 
sorry

end inequality_proof_l209_209204


namespace task_completion_time_l209_209120

variable (x : Real) (y : Real)

theorem task_completion_time :
  (1 / 16) * y + (1 / 12) * x = 1 ∧ y + 5 = 8 → x = 3 ∧ y = 3 :=
  by {
    sorry 
  }

end task_completion_time_l209_209120


namespace unique_integer_solution_m_l209_209373

theorem unique_integer_solution_m {m : ℤ} (h : ∀ x : ℤ, |2 * x - m| ≤ 1 → x = 2) : m = 4 := 
sorry

end unique_integer_solution_m_l209_209373


namespace ladder_base_distance_l209_209952

theorem ladder_base_distance (h l : ℝ) (h_eq : h = 13) (l_eq : l = 12) :
  ∃ b : ℝ, (h^2 = l^2 + b^2) ∧ b = 5 :=
by
  sorry

end ladder_base_distance_l209_209952


namespace arithmetic_sequence_find_m_l209_209132

theorem arithmetic_sequence_find_m (S : ℕ → ℤ) (m : ℕ)
  (h1 : S (m - 1) = -2)
  (h2 : S m = 0)
  (h3 : S (m + 1) = 3) :
  m = 5 :=
sorry

end arithmetic_sequence_find_m_l209_209132


namespace value_of_P_2017_l209_209129

theorem value_of_P_2017 (a b c : ℝ) (h_distinct: a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ 0 < a ∧ 0 < b ∧ 0 < c)
    (p : ℝ → ℝ) :
    (∀ x, p x = (c * (x - a) * (x - b) / ((c - a) * (c - b))) + (a * (x - b) * (x - c) / ((a - b) * (a - c))) + (b * (x - c) * (x - a) / ((b - c) * (b - a))) + 1) →
    p 2017 = 2 :=
sorry

end value_of_P_2017_l209_209129


namespace dave_apps_problem_l209_209187

theorem dave_apps_problem 
  (initial_apps : ℕ)
  (added_apps : ℕ)
  (final_apps : ℕ)
  (total_apps := initial_apps + added_apps)
  (deleted_apps := total_apps - final_apps) :
  initial_apps = 21 →
  added_apps = 89 →
  final_apps = 24 →
  (added_apps - deleted_apps = 3) :=
by
  intros
  sorry

end dave_apps_problem_l209_209187


namespace parabola_properties_l209_209202

theorem parabola_properties (p m k1 k2 k3 : ℝ)
  (parabola_eq : ∀ x y, y^2 = 2 * p * x ↔ y = m)
  (parabola_passes_through : m^2 = 2 * p)
  (point_distance : ((1 + p / 2)^2 + m^2 = 8) ∨ ((1 + p / 2)^2 + m^2 = 8))
  (p_gt_zero : p > 0)
  (point_P : (1, 2) ∈ { (x, y) | y^2 = 4 * x })
  (slope_eq : k3 = (k1 * k2) / (k1 + k2 - k1 * k2)) :
  (y^2 = 4 * x) ∧ (1/k1 + 1/k2 - 1/k3 = 1) := sorry

end parabola_properties_l209_209202


namespace cistern_wet_surface_area_l209_209036

theorem cistern_wet_surface_area
  (length : ℝ) (width : ℝ) (breadth : ℝ)
  (h_length : length = 9)
  (h_width : width = 6)
  (h_breadth : breadth = 2.25) :
  (length * width + 2 * (length * breadth) + 2 * (width * breadth)) = 121.5 :=
by
  -- Proof goes here
  sorry

end cistern_wet_surface_area_l209_209036


namespace smallest_possible_x_l209_209913

/-- Proof problem: When x is divided by 6, 7, and 8, remainders of 5, 6, and 7 (respectively) are obtained. 
We need to show that the smallest possible positive integer value of x is 167. -/
theorem smallest_possible_x (x : ℕ) (h1 : x % 6 = 5) (h2 : x % 7 = 6) (h3 : x % 8 = 7) : x = 167 :=
by 
  sorry

end smallest_possible_x_l209_209913


namespace foundation_cost_calculation_l209_209664

section FoundationCost

-- Define the constants given in the conditions
def length : ℝ := 100
def width : ℝ := 100
def height : ℝ := 0.5
def density : ℝ := 150  -- in pounds per cubic foot
def cost_per_pound : ℝ := 0.02
def number_of_houses : ℕ := 3

-- Define the problem using these conditions
theorem foundation_cost_calculation :
  let volume := length * width * height in
  let weight := volume * density in
  let cost_one_house := weight * cost_per_pound in
  let total_cost := cost_one_house * (number_of_houses:ℝ) in
  total_cost = 45000 := 
by {
  -- The proof goes here
  sorry
}

end FoundationCost

end foundation_cost_calculation_l209_209664


namespace area_ratio_l209_209388

variables (A B C D E F P N1 N2 N3 : Type)
variable [field A] -- Assuming this represents the areas for simplicity

-- Given conditions
def is_division_ratio (X Y : Type) (r : ℝ) : Prop := sorry

def ratios (D E F P : Type) :=
  (is_division_ratio D A 0.25) ∧
  (is_division_ratio E A 0.25) ∧
  (is_division_ratio F A 0.25) ∧
  (is_division_ratio P B (1/4))

-- The question we want to answer and the proof statement
theorem area_ratio (h : ratios D E F P) : 
  let K := (A B C) in
    (N1 N2 N3) = (14/25) * K :=
sorry

end area_ratio_l209_209388


namespace glass_bowls_sold_l209_209043

theorem glass_bowls_sold
  (BowlsBought : ℕ) (CostPricePerBowl SellingPricePerBowl : ℝ) (PercentageGain : ℝ)
  (CostPrice := BowlsBought * CostPricePerBowl)
  (SellingPrice : ℝ := (102 : ℝ) * SellingPricePerBowl)
  (gain := (SellingPrice - CostPrice) / CostPrice * 100) :
  PercentageGain = 8.050847457627118 →
  BowlsBought = 118 →
  CostPricePerBowl = 12 →
  SellingPricePerBowl = 15 →
  PercentageGain = gain →
  102 = 102 := by
  intro h1 h2 h3 h4 h5
  sorry

end glass_bowls_sold_l209_209043


namespace sec_150_eq_neg_2_sqrt_3_div_3_l209_209795

theorem sec_150_eq_neg_2_sqrt_3_div_3 :
  ∃ (sec : ℝ → ℝ),
    (∀ θ, sec θ = 1 / Real.cos θ) →
    sec 150 = - (2 * Real.sqrt 3) / 3 :=
by
  assume sec : ℝ → ℝ
  assume h_sec : ∀ θ, sec θ = 1 / Real.cos θ
  have h_cos_150 : Real.cos 150 = -Real.cos 30 := by sorry
  have h_cos_30 : Real.cos 30 = Real.sqrt 3 / 2 := by sorry
  show sec 150 = - (2 * Real.sqrt 3) / 3 := by sorry

end sec_150_eq_neg_2_sqrt_3_div_3_l209_209795


namespace two_x_plus_two_y_value_l209_209089

theorem two_x_plus_two_y_value (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2 * x + 2 * y = 8 / 3 := 
by sorry

end two_x_plus_two_y_value_l209_209089


namespace problems_per_page_l209_209673

theorem problems_per_page (total_problems : ℕ) (percent_solved : ℝ) (pages_left : ℕ)
  (h_total : total_problems = 550)
  (h_percent : percent_solved = 0.65)
  (h_pages : pages_left = 3) :
  (total_problems - Nat.ceil (percent_solved * total_problems)) / pages_left = 64 := by
  sorry

end problems_per_page_l209_209673


namespace constants_sum_l209_209194

theorem constants_sum (c d : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = if x ≤ 5 then c * x + d else 10 - 2 * x) 
  (h₂ : ∀ x : ℝ, f (f x) = x) : c + d = 6.5 := 
by sorry

end constants_sum_l209_209194


namespace base_from_wall_l209_209965

-- Define the given constants
def ladder_length : ℝ := 13
def wall_height : ℝ := 12

-- Define the base of the ladder being sought
def base_of_ladder (x : ℝ) : Prop :=
  x^2 + wall_height^2 = ladder_length^2

-- Prove that the base of the ladder is 5 meters
theorem base_from_wall : ∃ x : ℝ, base_of_ladder x ∧ x = 5 := by
  have h_ladder : ladder_length = 13 := rfl
  have h_wall : wall_height = 12 := rfl
  use 5
  split
  unfold base_of_ladder
  rw [h_wall, h_ladder]
  norm_num
  sorry

end base_from_wall_l209_209965


namespace coeff_x3y5_in_expansion_l209_209737

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l209_209737


namespace number_of_juniors_twice_seniors_l209_209473

variable (j s : ℕ)

theorem number_of_juniors_twice_seniors
  (h1 : (3 / 7 : ℝ) * j = (6 / 7 : ℝ) * s) : j = 2 * s := 
sorry

end number_of_juniors_twice_seniors_l209_209473


namespace not_less_than_x3_y5_for_x2y_l209_209029

theorem not_less_than_x3_y5_for_x2y (x y : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) : x^2 * y ≥ x^3 + y^5 :=
sorry

end not_less_than_x3_y5_for_x2y_l209_209029


namespace value_expression_l209_209465

noncomputable def g (p q r s t : ℝ) (x : ℝ) : ℝ := p * x^4 + q * x^3 + r * x^2 + s * x + t

theorem value_expression (p q r s t : ℝ) (h : g p q r s t (-3) = 9) : 
  16 * p - 8 * q + 4 * r - 2 * s + t = -9 := 
by
  sorry

end value_expression_l209_209465


namespace smallest_integer_is_10_l209_209896

noncomputable def smallest_integer (a b c : ℕ) : ℕ :=
  if h : (a + b + c = 90) ∧ (2 * b = 3 * a) ∧ (5 * a = 2 * c)
  then a
  else 0

theorem smallest_integer_is_10 (a b c : ℕ) (h₁ : a + b + c = 90) (h₂ : 2 * b = 3 * a) (h₃ : 5 * a = 2 * c) : 
  smallest_integer a b c = 10 :=
sorry

end smallest_integer_is_10_l209_209896


namespace min_value_of_a_b_c_l209_209837

variable (a b c : ℕ)
variable (x1 x2 : ℝ)

axiom h1 : a > 0
axiom h2 : b > 0
axiom h3 : c > 0
axiom h4 : a * x1^2 + b * x1 + c = 0
axiom h5 : a * x2^2 + b * x2 + c = 0
axiom h6 : |x1| < 1/3
axiom h7 : |x2| < 1/3

theorem min_value_of_a_b_c : a + b + c = 25 :=
by
  sorry

end min_value_of_a_b_c_l209_209837


namespace ratio_celeste_bianca_l209_209775

-- Definitions based on given conditions
def bianca_hours : ℝ := 12.5
def celest_hours (x : ℝ) : ℝ := 12.5 * x
def mcclain_hours (x : ℝ) : ℝ := 12.5 * x - 8.5

-- The total time worked in hours
def total_hours : ℝ := 54

-- The ratio to prove
def celeste_bianca_ratio : ℝ := 2

-- The proof statement
theorem ratio_celeste_bianca (x : ℝ) (hx :  12.5 + 12.5 * x + (12.5 * x - 8.5) = total_hours) :
  celest_hours 2 / bianca_hours = celeste_bianca_ratio :=
by
  sorry

end ratio_celeste_bianca_l209_209775


namespace mural_lunch_break_duration_l209_209600

variable (a t L : ℝ)

theorem mural_lunch_break_duration
  (h1 : (8 - L) * (a + t) = 0.6)
  (h2 : (6.5 - L) * t = 0.3)
  (h3 : (11 - L) * a = 0.1) :
  L = 40 :=
by
  sorry

end mural_lunch_break_duration_l209_209600


namespace joe_spent_on_fruits_l209_209854

theorem joe_spent_on_fruits (total_money amount_left : ℝ) (spent_on_chocolates : ℝ)
  (h1 : total_money = 450)
  (h2 : spent_on_chocolates = (1/9) * total_money)
  (h3 : amount_left = 220)
  : (total_money - spent_on_chocolates - amount_left) / total_money = 2 / 5 :=
by
  sorry

end joe_spent_on_fruits_l209_209854


namespace range_of_m_l209_209370

variable {x y m : ℝ}

theorem range_of_m (hx : 0 < x) (hy : 0 < y) (h : 4 / x + 1 / y = 1) :
  x + y ≥ m^2 + m + 3 ↔ -3 ≤ m ∧ m ≤ 2 := sorry

end range_of_m_l209_209370


namespace axis_of_symmetry_sine_function_l209_209428

theorem axis_of_symmetry_sine_function :
  ∃ k : ℤ, x = k * (π / 2) := sorry

end axis_of_symmetry_sine_function_l209_209428


namespace coeff_x3y5_in_expansion_l209_209735

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose n k

theorem coeff_x3y5_in_expansion (x y : ℕ) :
  (binomial_coefficient 8 5) = 56 := by
  sorry

end coeff_x3y5_in_expansion_l209_209735


namespace arithmetic_seq_condition_l209_209368

def sum_first_n_terms (a d : ℕ) (n : ℕ) : ℕ := 
  n * a + (n * (n - 1) / 2) * d

theorem arithmetic_seq_condition (a2 : ℕ) (S3 S9 : ℕ) :
  a2 = 1 → 
  (∃ d, (d > 4 ∧ S3 = 3 * a2 + (3 * (3 - 1) / 2) * d ∧ S9 = 9 * a2 + (9 * (9 - 1) / 2) * d) → (S3 + S9) > 93) ↔ 
  (∃ d, (S3 + S9 = sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9 ∧ (sum_first_n_terms a2 d 3 + sum_first_n_terms a2 d 9) > 93 → d > 3 ∧ a2 + d > 5)) :=
by 
  sorry

end arithmetic_seq_condition_l209_209368


namespace range_of_x_l209_209502

-- Define the necessary properties and functions.
variable (f : ℝ → ℝ)
variable (hf_even : ∀ x : ℝ, f (-x) = f x)
variable (hf_monotonic : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y)

-- Define the statement to be proved.
theorem range_of_x (f : ℝ → ℝ) (hf_even : ∀ x, f (-x) = f x) (hf_monotonic : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  { x : ℝ | f (2 * x - 1) ≤ f 3 } = { x | -1 ≤ x ∧ x ≤ 2 } :=
by
  sorry

end range_of_x_l209_209502


namespace largest_consecutive_odd_number_sum_is_27_l209_209285

theorem largest_consecutive_odd_number_sum_is_27
  (a b c : ℤ)
  (h1 : a + b + c = 75)
  (h2 : c - a = 4)
  (h3 : a % 2 = 1)
  (h4 : b % 2 = 1)
  (h5 : c % 2 = 1) :
  c = 27 := 
sorry

end largest_consecutive_odd_number_sum_is_27_l209_209285


namespace probability_three_primes_in_seven_dice_l209_209998

def prime_probability (n : ℕ) : ℚ :=
  if n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7 then (2 / 5) else (3 / 5)

def primes_in_dice (dice : List ℕ) : ℚ :=
  let num_primes := dice.count (λ x => x = 2 ∨ x = 3 ∨ x = 5 ∨ x = 7)
  if num_primes = 3 then (35 : ℚ) * (2 / 5) ^ 3 * (3 / 5) ^ 4 else 0

theorem probability_three_primes_in_seven_dice :
  primes_in_dice [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1, 2, 3, 4, 5, 6, 7] = (9072 / 31250) :=
sorry

end probability_three_primes_in_seven_dice_l209_209998


namespace tim_buys_loaves_l209_209291

theorem tim_buys_loaves (slices_per_loaf : ℕ) (paid : ℕ) (change : ℕ) (price_per_slice_cents : ℕ) 
    (h1 : slices_per_loaf = 20) 
    (h2 : paid = 2 * 20) 
    (h3 : change = 16) 
    (h4 : price_per_slice_cents = 40) : 
    (paid - change) / (slices_per_loaf * price_per_slice_cents / 100) = 3 := 
by 
  -- proof omitted 
  sorry

end tim_buys_loaves_l209_209291


namespace find_n_l209_209082

theorem find_n : ∃ n : ℤ, 100 ≤ n ∧ n ≤ 280 ∧ Real.cos (n * Real.pi / 180) = Real.cos (317 * Real.pi / 180) ∧ n = 317 := 
by
  sorry

end find_n_l209_209082


namespace binomial_coefficient_x3y5_in_expansion_l209_209713

theorem binomial_coefficient_x3y5_in_expansion :
  (binomial.coeff 8 5 = 56) := by
  sorry

end binomial_coefficient_x3y5_in_expansion_l209_209713


namespace coefficient_of_x3_y5_in_binomial_expansion_l209_209714

theorem coefficient_of_x3_y5_in_binomial_expansion : 
  let n := 8
  let k := 3
  (finset.sum (finset.range (n + 1)) (λ i, (nat.choose n i) * (x ^ i) * (y ^ (n - i)))) = 56 := 
by
  sorry

end coefficient_of_x3_y5_in_binomial_expansion_l209_209714


namespace ladder_base_distance_l209_209923

variable (x : ℕ)
variable (ladder_length : ℕ := 13)
variable (height : ℕ := 12)

theorem ladder_base_distance : ∃ x, (x^2 + height^2 = ladder_length^2) ∧ x = 5 := by
  sorry

end ladder_base_distance_l209_209923


namespace number_of_geese_is_correct_l209_209289

noncomputable def number_of_ducks := 37
noncomputable def total_number_of_birds := 95
noncomputable def number_of_geese := total_number_of_birds - number_of_ducks

theorem number_of_geese_is_correct : number_of_geese = 58 := by
  sorry

end number_of_geese_is_correct_l209_209289


namespace vector_proj_problems_l209_209841

noncomputable theory
open_locale big_operators
open Finset

structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

def vector_sub (a b : Point3D) : Point3D :=
⟨a.x - b.x, a.y - b.y, a.z - b.z⟩

def dot_product (v1 v2 : Point3D) : ℝ :=
(v1.x * v2.x) + (v1.y * v2.y) + (v1.z * v2.z)

def magnitude (v : Point3D) : ℝ :=
real.sqrt ((v.x ^ 2) + (v.y ^ 2) + (v.z ^ 2))

def cos_angle (v1 v2 : Point3D) : ℝ :=
(dot_product v1 v2) / ((magnitude v1) * (magnitude v2))

def distance_point_to_line (a b c : Point3D) : ℝ :=
let ab := vector_sub b a,
    ac := vector_sub c a,
    bc := vector_sub c b,
    height := magnitude ac * real.sqrt (1 - (cos_angle ac bc) ^ 2) in
    height / magnitude bc

theorem vector_proj_problems :
  let A := Point3D.mk (-1) 2 1,
      B := Point3D.mk 1 3 1,
      C := Point3D.mk (-2) 4 2 in
  (dot_product (vector_sub B A) (vector_sub C A) = 0) ∧
  (dot_product (Point3D.mk 1 (-2) (-5)) (vector_sub C A) ≠ 0) ∧
  (cos_angle (vector_sub C A) (vector_sub C B) = real.sqrt 66 / 11) ∧
  (distance_point_to_line A B C = real.sqrt 330 / 11) :=
by
  sorry

end vector_proj_problems_l209_209841


namespace domain_of_sqrt_function_l209_209554

theorem domain_of_sqrt_function :
  {x : ℝ | (1 / (Real.log x / Real.log 2) - 2 ≥ 0) ∧ (x > 0) ∧ (x ≠ 1)} 
  = {x : ℝ | 1 < x ∧ x ≤ Real.sqrt 10} :=
sorry

end domain_of_sqrt_function_l209_209554


namespace license_plate_count_l209_209215

/-- 
Proof statement: 
A license plate consists of 4 characters where:
1. The first character is a letter.
2. The second and third characters can either be a letter or a digit.
3. The fourth character is a digit.
4. There must be two characters on the license plate which are the same.

Prove that the number of ways to choose such a license plate equals 56,520.
-/
theorem license_plate_count :
  ∃ (n : ℕ), 
    n = 56520 ∧  
    (∃ f : fin 4 → char, 
      (f 0 ∈ alphabet ∧ 
       (f 1 ∈ alphabet ∪ digits) ∧ 
       (f 2 ∈ alphabet ∪ digits) ∧ 
       (f 3 ∈ digits) ∧ 
       (∃ i j : fin 4, i ≠ j ∧ f i = f j))) := 
sorry

end license_plate_count_l209_209215


namespace digits_satisfy_sqrt_l209_209824

theorem digits_satisfy_sqrt (a b : ℕ) (ha : a ≤ 9) (hb : b ≤ 9) :
  (b = 0 ∧ a = 0) ∨ (b = 3 ∧ a = 1) ∨ (b = 6 ∧ a = 4) ∨ (b = 9 ∧ a = 9) ↔ b^2 = 9 * a :=
by
  sorry

end digits_satisfy_sqrt_l209_209824


namespace real_root_fraction_l209_209883

theorem real_root_fraction (a b : ℝ) 
  (h_cond_a : a^4 - 7 * a - 3 = 0) 
  (h_cond_b : b^4 - 7 * b - 3 = 0)
  (h_order : a > b) : 
  (a - b) / (a^4 - b^4) = 1 / 7 := 
sorry

end real_root_fraction_l209_209883


namespace pyramid_transport_volume_l209_209329

-- Define the conditions of the problem
def pyramid_height : ℝ := 15
def pyramid_base_side_length : ℝ := 8
def box_length : ℝ := 10
def box_width : ℝ := 10
def box_height : ℝ := 15

-- Define the volume of the box
def box_volume : ℝ := box_length * box_width * box_height

-- State the theorem
theorem pyramid_transport_volume : box_volume = 1500 := by
  sorry

end pyramid_transport_volume_l209_209329


namespace converse_even_sum_l209_209028

variable (a b : ℤ)

def is_even (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k

theorem converse_even_sum (h : is_even (a + b)) : is_even a ∧ is_even b :=
sorry

end converse_even_sum_l209_209028
