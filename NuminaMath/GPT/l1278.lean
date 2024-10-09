import Mathlib

namespace D_cows_grazed_l1278_127823

-- Defining the given conditions:
def A_cows := 24
def A_months := 3
def A_rent := 1440

def B_cows := 10
def B_months := 5

def C_cows := 35
def C_months := 4

def D_months := 3

def total_rent := 6500

-- Calculate the cost per cow per month (CPCM)
def CPCM := A_rent / (A_cows * A_months)

-- Proving the number of cows D grazed
theorem D_cows_grazed : ∃ x : ℕ, (x * D_months * CPCM + A_rent + (B_cows * B_months * CPCM) + (C_cows * C_months * CPCM) = total_rent) ∧ x = 21 := by
  sorry

end D_cows_grazed_l1278_127823


namespace dorms_and_students_l1278_127872

theorem dorms_and_students (x : ℕ) :
  (4 * x + 19) % 6 ≠ 0 → ∃ s : ℕ, (x = 10 ∧ s = 59) ∨ (x = 11 ∧ s = 63) ∨ (x = 12 ∧ s = 67) :=
by
  sorry

end dorms_and_students_l1278_127872


namespace renovation_cost_distribution_l1278_127815

/-- A mathematical proof that if Team A works alone for 3 weeks, followed by both Team A and Team B working together, and the total renovation cost is 4000 yuan, then the payment should be distributed equally between Team A and Team B, each receiving 2000 yuan. -/
theorem renovation_cost_distribution :
  let time_A := 18
  let time_B := 12
  let initial_time_A := 3
  let total_cost := 4000
  ∃ x, (1 / time_A * (x + initial_time_A) + 1 / time_B * x = 1) ∧
       let work_A := 1 / time_A * (x + initial_time_A)
       let work_B := 1 / time_B * x
       work_A = work_B ∧
       total_cost / 2 = 2000 :=
by
  sorry

end renovation_cost_distribution_l1278_127815


namespace crescent_perimeter_l1278_127831

def radius_outer : ℝ := 10.5
def radius_inner : ℝ := 6.7

theorem crescent_perimeter : (radius_outer + radius_inner) * Real.pi = 54.037 :=
by
  sorry

end crescent_perimeter_l1278_127831


namespace katie_needs_more_sugar_l1278_127867

-- Let total_cups be the total cups of sugar required according to the recipe
def total_cups : ℝ := 3

-- Let already_put_in be the cups of sugar Katie has already put in
def already_put_in : ℝ := 0.5

-- Define the amount of sugar Katie still needs to put in
def remaining_cups : ℝ := total_cups - already_put_in 

-- Prove that remaining_cups is 2.5
theorem katie_needs_more_sugar : remaining_cups = 2.5 := 
by 
  -- substitute total_cups and already_put_in
  dsimp [remaining_cups, total_cups, already_put_in]
  -- calculate the difference
  norm_num

end katie_needs_more_sugar_l1278_127867


namespace linear_function_iff_l1278_127896

variable {x : ℝ} (m : ℝ)

def f (m : ℝ) (x : ℝ) : ℝ := (m + 2) * x + 4 * x - 5

theorem linear_function_iff (m : ℝ) : 
  (∃ c d, ∀ x, f m x = c * x + d) ↔ m ≠ -6 :=
by 
  sorry

end linear_function_iff_l1278_127896


namespace min_age_of_youngest_person_l1278_127878

theorem min_age_of_youngest_person
  {a b c d e : ℕ}
  (h_sum : a + b + c + d + e = 256)
  (h_order : a ≤ b ∧ b ≤ c ∧ c ≤ d ∧ d ≤ e)
  (h_diff : 2 ≤ (b - a) ∧ (b - a) ≤ 10 ∧ 
            2 ≤ (c - b) ∧ (c - b) ≤ 10 ∧ 
            2 ≤ (d - c) ∧ (d - c) ≤ 10 ∧ 
            2 ≤ (e - d) ∧ (e - d) ≤ 10) : 
  a = 32 :=
sorry

end min_age_of_youngest_person_l1278_127878


namespace radio_show_songs_duration_l1278_127898

-- Definitions of the conditions
def hours_per_day := 3
def minutes_per_hour := 60
def talking_segments := 3
def talking_segment_duration := 10
def ad_breaks := 5
def ad_break_duration := 5

-- The main statement translating the conditions and questions to Lean
theorem radio_show_songs_duration :
  (hours_per_day * minutes_per_hour) - (talking_segments * talking_segment_duration + ad_breaks * ad_break_duration) = 125 := by
  sorry

end radio_show_songs_duration_l1278_127898


namespace island_width_l1278_127866

theorem island_width (area length width : ℕ) (h₁ : area = 50) (h₂ : length = 10) : width = area / length := by 
  sorry

end island_width_l1278_127866


namespace find_m_through_point_l1278_127899

theorem find_m_through_point :
  ∃ m : ℝ, ∀ (x y : ℝ), ((y = (m - 1) * x - 4) ∧ (x = 2) ∧ (y = 4)) → m = 5 :=
by 
  -- Sorry can be used here to skip the proof as instructed
  sorry

end find_m_through_point_l1278_127899


namespace shorter_side_length_l1278_127845

theorem shorter_side_length (L W : ℝ) (h1 : L * W = 91) (h2 : 2 * L + 2 * W = 40) :
  min L W = 7 :=
by
  sorry

end shorter_side_length_l1278_127845


namespace seats_per_bus_l1278_127875

theorem seats_per_bus (students buses : ℕ) (h1 : students = 14) (h2 : buses = 7) : students / buses = 2 := by
  sorry

end seats_per_bus_l1278_127875


namespace shoes_count_l1278_127873

def numberOfShoes (numPairs : Nat) (matchingPairProbability : ℚ) : Nat :=
  let S := numPairs * 2
  if (matchingPairProbability = 1 / (S - 1))
  then S
  else 0

theorem shoes_count 
(numPairs : Nat)
(matchingPairProbability : ℚ)
(hp : numPairs = 9)
(hq : matchingPairProbability = 0.058823529411764705) :
numberOfShoes numPairs matchingPairProbability = 18 := 
by
  -- definition only, the proof is not required
  sorry

end shoes_count_l1278_127873


namespace tom_has_hours_to_spare_l1278_127813

-- Conditions as definitions
def numberOfWalls : Nat := 5
def wallWidth : Nat := 2 -- in meters
def wallHeight : Nat := 3 -- in meters
def paintingRate : Nat := 10 -- in minutes per square meter
def totalAvailableTime : Nat := 10 -- in hours

-- Lean 4 statement of the problem
theorem tom_has_hours_to_spare :
  let areaOfOneWall := wallWidth * wallHeight -- 2 * 3
  let totalArea := numberOfWalls * areaOfOneWall -- 5 * (2 * 3)
  let totalTimeToPaint := (totalArea * paintingRate) / 60 -- (30 * 10) / 60
  totalAvailableTime - totalTimeToPaint = 5 :=
by
  sorry

end tom_has_hours_to_spare_l1278_127813


namespace problem_f8_f2018_l1278_127861

theorem problem_f8_f2018 (f : ℕ → ℝ) (h₀ : ∀ n, f (n + 3) = (f n - 1) / (f n + 1)) 
  (h₁ : f 1 ≠ 0) (h₂ : f 1 ≠ 1) (h₃ : f 1 ≠ -1) : 
  f 8 * f 2018 = -1 :=
sorry

end problem_f8_f2018_l1278_127861


namespace correct_table_count_l1278_127809

def stools_per_table : ℕ := 8
def chairs_per_table : ℕ := 2
def legs_per_stool : ℕ := 3
def legs_per_chair : ℕ := 4
def legs_per_table : ℕ := 4
def total_legs : ℕ := 656

theorem correct_table_count (t : ℕ) :
  stools_per_table * legs_per_stool * t +
  chairs_per_table * legs_per_chair * t +
  legs_per_table * t = total_legs → t = 18 :=
by
  intros h
  sorry

end correct_table_count_l1278_127809


namespace digit_problem_l1278_127807

theorem digit_problem (A B C D E F : ℕ) (hABC : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ D ≠ E ∧ D ≠ F ∧ E ≠ F) 
    (h1 : 100 * A + 10 * B + C = D * 100000 + A * 10000 + E * 1000 + C * 100 + F * 10 + B)
    (h2 : 100 * C + 10 * B + A = E * 100000 + D * 10000 + C * 1000 + A * 100 + B * 10 + F) : 
    A = 3 ∧ B = 6 ∧ C = 4 ∧ D = 1 ∧ E = 2 ∧ F = 9 := 
sorry

end digit_problem_l1278_127807


namespace rectangle_parallelepiped_angles_l1278_127835

theorem rectangle_parallelepiped_angles 
  (a b c d : ℝ) 
  (α β : ℝ) 
  (h_a : a = d * Real.sin β)
  (h_b : b = d * Real.sin α)
  (h_d : d^2 = (d * Real.sin β)^2 + c^2 + (d * Real.sin α)^2) :
  (α > 0 ∧ β > 0 ∧ α + β < 90) := sorry

end rectangle_parallelepiped_angles_l1278_127835


namespace cows_now_l1278_127826

-- Defining all conditions
def initial_cows : ℕ := 39
def cows_died : ℕ := 25
def cows_sold : ℕ := 6
def cows_increase : ℕ := 24
def cows_bought : ℕ := 43
def cows_gift : ℕ := 8

-- Lean statement for the equivalent proof problem
theorem cows_now :
  let cows_left := initial_cows - cows_died
  let cows_after_selling := cows_left - cows_sold
  let cows_this_year_increased := cows_after_selling + cows_increase
  let cows_with_purchase := cows_this_year_increased + cows_bought
  let total_cows := cows_with_purchase + cows_gift
  total_cows = 83 :=
by
  sorry

end cows_now_l1278_127826


namespace geom_seq_sum_problem_l1278_127892

noncomputable def geom_sum_first_n_terms (a₁ q : ℕ) (n : ℕ) : ℕ :=
  a₁ * (1 - q^n) / (1 - q)

noncomputable def geom_sum_specific_terms (a₃ q : ℕ) (n m : ℕ) : ℕ :=
  a₃ * ((1 - (q^m) ^ n) / (1 - q^m))

theorem geom_seq_sum_problem :
  ∀ (a₁ q S₈₇ : ℕ),
  q = 2 →
  S₈₇ = 140 →
  geom_sum_first_n_terms a₁ q 87 = S₈₇ →
  ∃ a₃, a₃ = ((q * q) * a₁) →
  geom_sum_specific_terms a₃ q 29 3 = 80 := 
by
  intros a₁ q S₈₇ hq₁ hS₈₇ hsum
  -- Further proof would go here
  sorry

end geom_seq_sum_problem_l1278_127892


namespace find_x_l1278_127895

theorem find_x (y : ℝ) (x : ℝ) (h : x / (x - 1) = (y^2 + 2 * y + 3) / (y^2 + 2 * y - 2)) :
  x = (y^2 + 2 * y + 3) / 5 := by
  sorry

end find_x_l1278_127895


namespace distance_to_place_l1278_127852

-- Define the conditions
def speed_boat_standing_water : ℝ := 16
def speed_stream : ℝ := 2
def total_time_taken : ℝ := 891.4285714285714

-- Define the calculated speeds
def downstream_speed : ℝ := speed_boat_standing_water + speed_stream
def upstream_speed : ℝ := speed_boat_standing_water - speed_stream

-- Define the variable for the distance
variable (D : ℝ)

-- State the theorem to prove
theorem distance_to_place :
  D / downstream_speed + D / upstream_speed = total_time_taken →
  D = 7020 :=
by
  intro h
  sorry

end distance_to_place_l1278_127852


namespace red_car_speed_is_10mph_l1278_127858

noncomputable def speed_of_red_car (speed_black : ℝ) (initial_distance : ℝ) (time_to_overtake : ℝ) : ℝ :=
  (speed_black * time_to_overtake - initial_distance) / time_to_overtake

theorem red_car_speed_is_10mph :
  ∀ (speed_black initial_distance time_to_overtake : ℝ),
  speed_black = 50 →
  initial_distance = 20 →
  time_to_overtake = 0.5 →
  speed_of_red_car speed_black initial_distance time_to_overtake = 10 :=
by
  intros speed_black initial_distance time_to_overtake hb hd ht
  rw [hb, hd, ht]
  norm_num
  sorry

end red_car_speed_is_10mph_l1278_127858


namespace still_water_speed_l1278_127884

-- The conditions as given in the problem
variables (V_m V_r V'_r : ℝ)
axiom upstream_speed : V_m - V_r = 20
axiom downstream_increased_speed : V_m + V_r = 30
axiom downstream_reduced_speed : V_m + V'_r = 26

-- Prove that the man's speed in still water is 25 km/h
theorem still_water_speed : V_m = 25 :=
by
  sorry

end still_water_speed_l1278_127884


namespace largest_prime_factor_sum_of_four_digit_numbers_l1278_127805

theorem largest_prime_factor_sum_of_four_digit_numbers 
  (a b c d : ℕ) (h1 : 1 ≤ a) (h2 : a ≤ 9) 
  (h3 : 1 ≤ b) (h4 : b ≤ 9) 
  (h5 : 1 ≤ c) (h6 : c ≤ 9) 
  (h7 : 1 ≤ d) (h8 : d ≤ 9) 
  (h_diff : (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d))
  : Nat.gcd 6666 (a + b + c + d) = 101 :=
sorry

end largest_prime_factor_sum_of_four_digit_numbers_l1278_127805


namespace petya_wins_l1278_127863

theorem petya_wins (n : ℕ) : n = 111 → (∀ k : ℕ, 1 ≤ k ∧ k ≤ 9 → ∃ x : ℕ, 1 ≤ x ∧ x ≤ 9 ∧ (n - k - x) % 10 = 0) → wins_optimal_play := sorry

end petya_wins_l1278_127863


namespace projectile_height_time_l1278_127820

theorem projectile_height_time :
  ∃ t, t ≥ 0 ∧ -16 * t^2 + 80 * t = 72 ↔ t = 1 := 
by sorry

end projectile_height_time_l1278_127820


namespace investment_percentage_change_l1278_127843

/-- 
Isabel's investment problem statement:
Given an initial investment, and percentage changes over three years,
prove that the overall percentage change in Isabel's investment is 1.2% gain.
-/
theorem investment_percentage_change (initial_investment : ℝ) (gain1 : ℝ) (loss2 : ℝ) (gain3 : ℝ) 
    (final_investment : ℝ) :
    initial_investment = 500 →
    gain1 = 0.10 →
    loss2 = 0.20 →
    gain3 = 0.15 →
    final_investment = initial_investment * (1 + gain1) * (1 - loss2) * (1 + gain3) →
    ((final_investment - initial_investment) / initial_investment) * 100 = 1.2 :=
by
  intros h_init h_gain1 h_loss2 h_gain3 h_final
  sorry

end investment_percentage_change_l1278_127843


namespace sum_of_logs_l1278_127864

open Real

noncomputable def log_base (b a : ℝ) : ℝ := log a / log b

theorem sum_of_logs (x y z : ℝ)
  (h1 : log_base 2 (log_base 4 (log_base 5 x)) = 0)
  (h2 : log_base 3 (log_base 5 (log_base 2 y)) = 0)
  (h3 : log_base 4 (log_base 2 (log_base 3 z)) = 0) :
  x + y + z = 666 := sorry

end sum_of_logs_l1278_127864


namespace johns_out_of_pocket_expense_l1278_127838

theorem johns_out_of_pocket_expense :
  let computer_cost := 700
  let accessories_cost := 200
  let playstation_value := 400
  let playstation_loss_percent := 0.2
  (computer_cost + accessories_cost - playstation_value * (1 - playstation_loss_percent) = 580) :=
by {
  sorry
}

end johns_out_of_pocket_expense_l1278_127838


namespace range_of_a_l1278_127862

theorem range_of_a (a : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 4 → |x - 1| < a) ↔ a ≥ 3 := by
  sorry

end range_of_a_l1278_127862


namespace four_r_eq_sum_abcd_l1278_127889

theorem four_r_eq_sum_abcd (a b c d r : ℤ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d :=
by 
  sorry

end four_r_eq_sum_abcd_l1278_127889


namespace complex_quadrant_l1278_127881

theorem complex_quadrant (a b : ℝ) (h : (a + Complex.I) / (b - Complex.I) = 2 - Complex.I) :
  (a < 0 ∧ b < 0) :=
by
  sorry

end complex_quadrant_l1278_127881


namespace martin_distance_l1278_127806

def speed : ℝ := 12.0  -- Speed in miles per hour
def time : ℝ := 6.0    -- Time in hours

theorem martin_distance : (speed * time) = 72.0 :=
by
  sorry

end martin_distance_l1278_127806


namespace parabola_point_distance_eq_l1278_127876

open Real

theorem parabola_point_distance_eq (P : ℝ × ℝ) (V : ℝ × ℝ) (F : ℝ × ℝ)
    (hV: V = (0, 0)) (hF : F = (0, 2)) (P_on_parabola : P.1 ^ 2 = 8 * P.2) 
    (hPf : dist P F = 150) (P_in_first_quadrant : 0 ≤ P.1 ∧ 0 ≤ P.2) :
    P = (sqrt 1184, 148) :=
sorry

end parabola_point_distance_eq_l1278_127876


namespace max_min_PA_l1278_127855

open Classical

variables (A B P : Type) [MetricSpace A] [MetricSpace B] [MetricSpace P]
          (dist_AB : ℝ) (dist_PA_PB : ℝ)

noncomputable def max_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry
noncomputable def min_PA (A B : Type) [MetricSpace A] [MetricSpace B] (dist_AB : ℝ) : ℝ := sorry

theorem max_min_PA (A B : Type) [MetricSpace A] [MetricSpace B] [Inhabited P]
                   (dist_AB : ℝ) (dist_PA_PB : ℝ) :
  dist_AB = 4 → dist_PA_PB = 6 → max_PA A B 4 = 5 ∧ min_PA A B 4 = 1 :=
by
  intros h_AB h_PA_PB
  sorry

end max_min_PA_l1278_127855


namespace complement_U_P_l1278_127891

def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

theorem complement_U_P :
  (U \ P) = Set.Ici (1 / 2) := 
by
  sorry

end complement_U_P_l1278_127891


namespace fraction_increases_l1278_127847

theorem fraction_increases (a : ℝ) (h : ℝ) (ha : a > -1) (hh : h > 0) : 
  (a + h) / (a + h + 1) > a / (a + 1) := 
by 
  sorry

end fraction_increases_l1278_127847


namespace Ofelia_savings_l1278_127822

theorem Ofelia_savings (X : ℝ) (h : 16 * X = 160) : X = 10 :=
by
  sorry

end Ofelia_savings_l1278_127822


namespace largest_number_of_gold_coins_l1278_127869

theorem largest_number_of_gold_coins (n : ℕ) :
  (∃ k : ℕ, n = 13 * k + 3 ∧ n < 150) → n ≤ 146 :=
by
  sorry

end largest_number_of_gold_coins_l1278_127869


namespace find_x_value_l1278_127830

theorem find_x_value (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x < 2 * Real.pi) (h₂ : Real.sin x - Real.cos x = Real.sqrt 2) : x = 3 * Real.pi / 4 :=
sorry

end find_x_value_l1278_127830


namespace complement_union_A_B_is_correct_l1278_127816

-- Define the set of real numbers R
def R : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := { x | ∃ (y : ℝ), y = Real.log (x + 3) }

-- Simplified definition for A to reflect x > -3
def A_simplified : Set ℝ := { x | x > -3 }

-- Define set B
def B : Set ℝ := { x | x ≥ 2 }

-- Define the union of A and B
def union_A_B : Set ℝ := A_simplified ∪ B

-- Define the complement of the union in R
def complement_R_union_A_B : Set ℝ := R \ union_A_B

-- State the theorem
theorem complement_union_A_B_is_correct :
  complement_R_union_A_B = { x | x ≤ -3 } := by
  sorry

end complement_union_A_B_is_correct_l1278_127816


namespace evaluate_g_3_times_l1278_127833

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 2 * n - 1
  else 2 * n + 3

theorem evaluate_g_3_times : g (g (g 3)) = 65 := by
  sorry

end evaluate_g_3_times_l1278_127833


namespace expression_value_l1278_127877

theorem expression_value (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) 
  (h₃ : (a^2 * b^2) / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1/3 := 
by
  sorry

end expression_value_l1278_127877


namespace evaluate_expression_l1278_127870

-- Definition of variables a, b, c as given in conditions
def a : ℕ := 7
def b : ℕ := 11
def c : ℕ := 13

-- The theorem to prove the given expression equals 31
theorem evaluate_expression : 
  (a^2 * (1 / b - 1 / c) + b^2 * (1 / c - 1 / a) + c^2 * (1 / a - 1 / b)) / 
  (a * (1 / b - 1 / c) + b * (1 / c - 1 / a) + c * (1 / a - 1 / b)) = 31 :=
by
  sorry

end evaluate_expression_l1278_127870


namespace volume_of_pyramid_correct_l1278_127879

noncomputable def volume_of_pyramid (lateral_surface_area base_area inscribed_circle_area radius : ℝ) : ℝ :=
  if lateral_surface_area = 3 * base_area ∧ inscribed_circle_area = radius then
    (2 * Real.sqrt 6) / (Real.pi ^ 3)
  else
    0

theorem volume_of_pyramid_correct
  (lateral_surface_area base_area inscribed_circle_area radius : ℝ)
  (h1 : lateral_surface_area = 3 * base_area)
  (h2 : inscribed_circle_area = radius) :
  volume_of_pyramid lateral_surface_area base_area inscribed_circle_area radius = (2 * Real.sqrt 6) / (Real.pi ^ 3) :=
by {
  sorry
}

end volume_of_pyramid_correct_l1278_127879


namespace sum_two_numbers_eq_twelve_l1278_127890

theorem sum_two_numbers_eq_twelve (x y : ℕ) (h1 : x^2 + y^2 = 90) (h2 : x * y = 27) : x + y = 12 :=
by
  sorry

end sum_two_numbers_eq_twelve_l1278_127890


namespace sum_of_points_probabilities_l1278_127874

-- Define probabilities for the sums of 2, 3, and 4
def P_A : ℚ := 1 / 36
def P_B : ℚ := 2 / 36
def P_C : ℚ := 3 / 36

-- Theorem statement
theorem sum_of_points_probabilities :
  (P_A < P_B) ∧ (P_B < P_C) :=
  sorry

end sum_of_points_probabilities_l1278_127874


namespace plane_through_A_perpendicular_to_BC_l1278_127888

-- Define points A, B, and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A : Point3D := ⟨-3, 6, 4⟩
def B : Point3D := ⟨8, -3, 5⟩
def C : Point3D := ⟨10, -3, 7⟩

-- Define the vector BC
def vectorBC (B C : Point3D) : Point3D :=
  ⟨C.x - B.x, C.y - B.y, C.z - B.z⟩

-- Equation of the plane
def planeEquation (p : Point3D) (n : Point3D) (x y z : ℝ) : ℝ :=
  n.x * (x - p.x) + n.y * (y - p.y) + n.z * (z - p.z)

theorem plane_through_A_perpendicular_to_BC : 
  planeEquation A (vectorBC B C) x y z = 0 ↔ x + z - 1 = 0 :=
by
  sorry

end plane_through_A_perpendicular_to_BC_l1278_127888


namespace mean_of_four_numbers_l1278_127849

theorem mean_of_four_numbers (a b c d : ℝ) (h : (a + b + c + d + 130) / 5 = 90) : (a + b + c + d) / 4 = 80 := by
  sorry

end mean_of_four_numbers_l1278_127849


namespace number_is_4_less_than_opposite_l1278_127808

-- Define the number and its opposite relationship
def opposite_relation (x : ℤ) : Prop := x = -x + (-4)

-- Theorem stating that the given number is 4 less than its opposite
theorem number_is_4_less_than_opposite (x : ℤ) : opposite_relation x :=
sorry

end number_is_4_less_than_opposite_l1278_127808


namespace solve_x_l1278_127840

theorem solve_x (x y : ℝ) (h1 : 3 * x - y = 7) (h2 : x + 3 * y = 16) : x = 16 := by
  sorry

end solve_x_l1278_127840


namespace students_in_class_l1278_127841

-- Define the relevant variables and conditions
variables (P H W T A S : ℕ)

-- Given conditions
axiom poetry_club : P = 22
axiom history_club : H = 27
axiom writing_club : W = 28
axiom two_clubs : T = 6
axiom all_clubs : A = 6

-- Statement to prove
theorem students_in_class
  (poetry_club : P = 22)
  (history_club : H = 27)
  (writing_club : W = 28)
  (two_clubs : T = 6)
  (all_clubs : A = 6) :
  S = P + H + W - T - 2 * A :=
sorry

end students_in_class_l1278_127841


namespace remainder_of_4n_minus_6_l1278_127825

theorem remainder_of_4n_minus_6 (n : ℕ) (h : n % 9 = 5) : (4 * n - 6) % 9 = 5 :=
sorry

end remainder_of_4n_minus_6_l1278_127825


namespace chinese_medicine_excess_purchased_l1278_127893

-- Define the conditions of the problem

def total_plan : ℕ := 1500

def first_half_percentage : ℝ := 0.55
def second_half_percentage : ℝ := 0.65

-- State the theorem to prove the amount purchased in excess
theorem chinese_medicine_excess_purchased :
    first_half_percentage * total_plan + second_half_percentage * total_plan - total_plan = 300 :=
by 
  sorry

end chinese_medicine_excess_purchased_l1278_127893


namespace max_value_expression_l1278_127868

theorem max_value_expression (x k : ℕ) (h₀ : 0 < x) (h₁ : 0 < k) (y := k * x) : 
  (∀ x k : ℕ, 0 < x → 0 < k → y = k * x → ∃ m : ℝ, m = 2 ∧ 
    ∀ x k : ℕ, 0 < x → 0 < k → y = k * x → (x + y)^2 / (x^2 + y^2) ≤ 2) :=
sorry

end max_value_expression_l1278_127868


namespace k_is_even_set_l1278_127856

open Set -- using Set from Lean library

noncomputable def kSet (s : Set ℤ) :=
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0)

theorem k_is_even_set (s : Set ℤ) :
  (∀ g ∈ ({5, 8, 7, 1} : Set ℤ), ∀ k ∈ s, (g * k) % 2 = 0) →
  ∀ k ∈ s, k % 2 = 0 :=
by
  intro h
  sorry

end k_is_even_set_l1278_127856


namespace mike_picked_64_peaches_l1278_127846

theorem mike_picked_64_peaches :
  ∀ (initial peaches_given total final_picked : ℕ),
    initial = 34 →
    peaches_given = 12 →
    total = 86 →
    final_picked = total - (initial - peaches_given) →
    final_picked = 64 :=
by
  intros
  sorry

end mike_picked_64_peaches_l1278_127846


namespace factorized_polynomial_sum_of_squares_l1278_127829

theorem factorized_polynomial_sum_of_squares :
  ∃ a b c d e f : ℤ, 
    (729 * x^3 + 64 = (a * x^2 + b * x + c) * (d * x^2 + e * x + f)) →
    (a^2 + b^2 + c^2 + d^2 + e^2 + f^2 = 8210) :=
sorry

end factorized_polynomial_sum_of_squares_l1278_127829


namespace factor_diff_of_squares_l1278_127824

theorem factor_diff_of_squares (t : ℝ) : 
  t^2 - 64 = (t - 8) * (t + 8) :=
by
  -- The purpose here is to state the theorem only, without the proof.
  sorry

end factor_diff_of_squares_l1278_127824


namespace equation_of_line_l1278_127803

theorem equation_of_line {x y : ℝ} (b : ℝ) (h1 : ∀ x y, (3 * x + 4 * y - 7 = 0) → (y = -3/4 * x))
  (h2 : (1 / 2) * |b| * |(4 / 3) * b| = 24) : 
  ∃ b : ℝ, ∀ x, y = -3/4 * x + b := 
sorry

end equation_of_line_l1278_127803


namespace parabola_area_l1278_127883

theorem parabola_area (m p : ℝ) (h1 : p > 0) (h2 : (1:ℝ)^2 = 2 * p * m)
    (h3 : (1/2) * (m + p / 2) = 1/2) : p = 1 :=
  by
    sorry

end parabola_area_l1278_127883


namespace solve_system_of_equations_l1278_127818

theorem solve_system_of_equations (x y z t : ℝ) :
  xy - t^2 = 9 ∧ x^2 + y^2 + z^2 = 18 ↔ (x = 3 ∧ y = 3 ∧ z = 0 ∧ t = 0) ∨ (x = -3 ∧ y = -3 ∧ z = 0 ∨ t = 0) :=
sorry

end solve_system_of_equations_l1278_127818


namespace merchant_printer_count_l1278_127851

theorem merchant_printer_count (P : ℕ) 
  (cost_keyboards : 15 * 20 = 300)
  (total_cost : 300 + 70 * P = 2050) :
  P = 25 := 
by
  sorry

end merchant_printer_count_l1278_127851


namespace truckToCarRatio_l1278_127848

-- Conditions
def liftsCar (C : ℕ) : Prop := C = 5
def peopleNeeded (C T : ℕ) : Prop := 6 * C + 3 * T = 60

-- Theorem statement
theorem truckToCarRatio (C T : ℕ) (hc : liftsCar C) (hp : peopleNeeded C T) : T / C = 2 :=
by
  sorry

end truckToCarRatio_l1278_127848


namespace cookies_with_new_flour_l1278_127886

-- Definitions for the conditions
def cookies_per_cup (total_cookies : ℕ) (total_flour : ℕ) : ℕ :=
  total_cookies / total_flour

noncomputable def cookies_from_flour (cookies_per_cup : ℕ) (flour : ℕ) : ℕ :=
  cookies_per_cup * flour

-- Given data
def total_cookies := 24
def total_flour := 4
def new_flour := 3

-- Theorem (problem statement)
theorem cookies_with_new_flour : cookies_from_flour (cookies_per_cup total_cookies total_flour) new_flour = 18 :=
by
  sorry

end cookies_with_new_flour_l1278_127886


namespace inradius_of_right_triangle_l1278_127801

theorem inradius_of_right_triangle (a b c r : ℝ) (h : a^2 + b^2 = c^2) :
  r = (1/2) * (a + b - c) :=
sorry

end inradius_of_right_triangle_l1278_127801


namespace smallest_four_digit_number_divisible_by_4_l1278_127880

theorem smallest_four_digit_number_divisible_by_4 : 
  ∃ n : ℕ, (1000 ≤ n ∧ n < 10000) ∧ (n % 4 = 0) ∧ n = 1000 := by
  sorry

end smallest_four_digit_number_divisible_by_4_l1278_127880


namespace bus_passing_time_l1278_127887

noncomputable def time_for_bus_to_pass (bus_length : ℝ) (bus_speed_kph : ℝ) (man_speed_kph : ℝ) : ℝ :=
  let relative_speed_kph := bus_speed_kph + man_speed_kph
  let relative_speed_mps := (relative_speed_kph * (1000/3600))
  bus_length / relative_speed_mps

theorem bus_passing_time :
  time_for_bus_to_pass 15 40 8 = 1.125 :=
by
  sorry

end bus_passing_time_l1278_127887


namespace seats_per_table_l1278_127837

-- Definitions based on conditions
def tables := 4
def total_people := 32

-- Statement to prove
theorem seats_per_table : (total_people / tables) = 8 :=
by 
  sorry

end seats_per_table_l1278_127837


namespace cone_height_correct_l1278_127871

noncomputable def cone_height (radius : ℝ) (central_angle : ℝ) : ℝ := 
  let base_radius := (central_angle * radius) / (2 * Real.pi)
  let height := Real.sqrt (radius ^ 2 - base_radius ^ 2)
  height

theorem cone_height_correct:
  cone_height 3 (2 * Real.pi / 3) = 2 * Real.sqrt 2 := 
by
  sorry

end cone_height_correct_l1278_127871


namespace cos_double_angle_of_parallel_vectors_l1278_127844

variables {α : Type*}

/-- Given vectors a and b specified by the problem, if they are parallel, then cos 2α = 7/9. -/
theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (a b : ℝ × ℝ) 
  (ha : a = (1/3, Real.tan α)) 
  (hb : b = (Real.cos α, 1)) 
  (parallel : a.1 * b.2 = a.2 * b.1) : 
  Real.cos (2 * α) = 7/9 := 
by 
  sorry

end cos_double_angle_of_parallel_vectors_l1278_127844


namespace compare_neg_fractions_l1278_127857

theorem compare_neg_fractions : (-5 / 4) < (-4 / 5) := sorry

end compare_neg_fractions_l1278_127857


namespace rectangle_perimeter_ratio_l1278_127810

theorem rectangle_perimeter_ratio (side_length : ℝ) (h : side_length = 4) :
  let small_rectangle_perimeter := 2 * (side_length + (side_length / 4))
  let large_rectangle_perimeter := 2 * (side_length + (side_length / 2))
  small_rectangle_perimeter / large_rectangle_perimeter = 5 / 6 :=
by
  sorry

end rectangle_perimeter_ratio_l1278_127810


namespace least_integer_l1278_127832

theorem least_integer (x p d n : ℕ) (hx : x = 10^p * d + n) (hn : n = 10^p * d / 18) : x = 950 :=
by sorry

end least_integer_l1278_127832


namespace factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l1278_127894

-- Given condition and question, prove equality for the first expression
theorem factorize_x4_minus_16y4 (x y : ℝ) :
  x^4 - 16 * y^4 = (x^2 + 4 * y^2) * (x + 2 * y) * (x - 2 * y) := 
by sorry

-- Given condition and question, prove equality for the second expression
theorem factorize_minus_2a3_plus_12a2_minus_16a (a : ℝ) :
  -2 * a^3 + 12 * a^2 - 16 * a = -2 * a * (a - 2) * (a - 4) := 
by sorry

end factorize_x4_minus_16y4_factorize_minus_2a3_plus_12a2_minus_16a_l1278_127894


namespace luncheon_cost_l1278_127854

theorem luncheon_cost (s c p : ℝ) (h1 : 5 * s + 9 * c + 2 * p = 5.95)
  (h2 : 7 * s + 12 * c + 2 * p = 7.90) (h3 : 3 * s + 5 * c + p = 3.50) :
  s + c + p = 1.05 :=
sorry

end luncheon_cost_l1278_127854


namespace distinct_positive_integers_mod_1998_l1278_127839

theorem distinct_positive_integers_mod_1998
  (a : Fin 93 → ℕ)
  (h_distinct : Function.Injective a) :
  ∃ m n p q : Fin 93, (m ≠ n ∧ p ≠ q) ∧ (a m - a n) * (a p - a q) % 1998 = 0 :=
by
  sorry

end distinct_positive_integers_mod_1998_l1278_127839


namespace compatible_polynomial_count_l1278_127850

theorem compatible_polynomial_count (n : ℕ) : 
  ∃ num_polynomials : ℕ, num_polynomials = (n / 2) + 1 :=
by
  sorry

end compatible_polynomial_count_l1278_127850


namespace sum_tens_units_11_pow_2010_l1278_127827

def tens_digit (n : ℕ) : ℕ :=
  (n / 10) % 10

def units_digit (n : ℕ) : ℕ :=
  n % 10

def sum_tens_units_digits (n : ℕ) : ℕ :=
  tens_digit n + units_digit n

theorem sum_tens_units_11_pow_2010 :
  sum_tens_units_digits (11 ^ 2010) = 1 :=
sorry

end sum_tens_units_11_pow_2010_l1278_127827


namespace part1_part2_l1278_127821

-- Part 1: Prove values of m and n.
theorem part1 (m n : ℝ) :
  (∀ x : ℝ, |x - m| ≤ n ↔ 0 ≤ x ∧ x ≤ 4) → m = 2 ∧ n = 2 :=
by
  intro h
  -- Proof omitted
  sorry

-- Part 2: Prove the minimum value of a + b.
theorem part2 (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : a * b = 2) :
  a + b = (2 / a) + (2 / b) → a + b ≥ 2 * Real.sqrt 2 :=
by
  intro h
  -- Proof omitted
  sorry

end part1_part2_l1278_127821


namespace x1_x2_product_l1278_127842

theorem x1_x2_product (x1 x2 : ℝ) (h1 : x1 ≠ x2) (h2 : x1^2 - 2006 * x1 = 1) (h3 : x2^2 - 2006 * x2 = 1) : x1 * x2 = -1 := 
by
  sorry

end x1_x2_product_l1278_127842


namespace men_with_6_boys_work_l1278_127811

theorem men_with_6_boys_work (m b : ℚ) (x : ℕ) :
  2 * m + 4 * b = 1 / 4 →
  x * m + 6 * b = 1 / 3 →
  2 * b = 5 * m →
  x = 1 :=
by
  intros h1 h2 h3
  sorry

end men_with_6_boys_work_l1278_127811


namespace max_value_expression_l1278_127885

theorem max_value_expression (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) (hsum : x + y + z = 3) :
  (x^2 - x*y + y^2) * (y^2 - y*z + z^2) * (z^2 - z*x + x^2) ≤ 27 / 8 :=
sorry

end max_value_expression_l1278_127885


namespace largest_integer_x_l1278_127802

theorem largest_integer_x (x : ℕ) : (1 / 4 : ℚ) + (x / 8 : ℚ) < 1 ↔ x <= 5 := sorry

end largest_integer_x_l1278_127802


namespace cos_135_eq_neg_sqrt2_div_2_l1278_127865

theorem cos_135_eq_neg_sqrt2_div_2 : Real.cos (135 * Real.pi / 180) = -Real.sqrt 2 / 2 := by
  -- Conditions incorporated directly into the requirement due to the nature of trigonometric evaluations.
  sorry

end cos_135_eq_neg_sqrt2_div_2_l1278_127865


namespace packs_sold_by_Robyn_l1278_127800

theorem packs_sold_by_Robyn (total_packs : ℕ) (lucy_packs : ℕ) (robyn_packs : ℕ) 
  (h1 : total_packs = 98) (h2 : lucy_packs = 43) (h3 : robyn_packs = total_packs - lucy_packs) :
  robyn_packs = 55 :=
by
  rw [h1, h2] at h3
  exact h3

end packs_sold_by_Robyn_l1278_127800


namespace original_quantity_of_ghee_l1278_127819

theorem original_quantity_of_ghee
  (Q : ℝ) 
  (H1 : (0.5 * Q) = (0.3 * (Q + 20))) : 
  Q = 30 := 
by
  -- proof goes here
  sorry

end original_quantity_of_ghee_l1278_127819


namespace houses_without_features_l1278_127817

-- Definitions for the given conditions
def N : ℕ := 70
def G : ℕ := 50
def P : ℕ := 40
def GP : ℕ := 35

-- The statement of the proof problem
theorem houses_without_features : N - (G + P - GP) = 15 := by
  sorry

end houses_without_features_l1278_127817


namespace fewest_posts_l1278_127897

def grazingAreaPosts (length width post_interval rock_wall_length : ℕ) : ℕ :=
  let side1 := width / post_interval + 1
  let side2 := length / post_interval
  side1 + 2 * side2

theorem fewest_posts (length width post_interval rock_wall_length posts : ℕ) :
  length = 70 ∧ width = 50 ∧ post_interval = 10 ∧ rock_wall_length = 150 ∧ posts = 18 →
  grazingAreaPosts length width post_interval rock_wall_length = posts := 
by
  intros h
  obtain ⟨hl, hw, hp, hr, ht⟩ := h
  simp [grazingAreaPosts, hl, hw, hp, hr]
  sorry

end fewest_posts_l1278_127897


namespace length_gh_parallel_lines_l1278_127814

theorem length_gh_parallel_lines (
    AB CD EF GH : ℝ
) (
    h1 : AB = 300
) (
    h2 : CD = 200
) (
    h3 : EF = (AB + CD) / 2 * (1 / 2)
) (
    h4 : GH = EF * (1 - 1 / 4)
) :
    GH = 93.75 :=
by
    sorry

end length_gh_parallel_lines_l1278_127814


namespace paper_needed_l1278_127812

theorem paper_needed : 26 + 26 + 10 = 62 := by
  sorry

end paper_needed_l1278_127812


namespace no_solutions_xyz_l1278_127804

theorem no_solutions_xyz :
  ¬ ∃ (x y z : ℝ), x + y = 3 ∧ xy - z^2 = 4 :=
by
  sorry

end no_solutions_xyz_l1278_127804


namespace tables_count_l1278_127834

def total_tables (four_legged_tables three_legged_tables : Nat) : Nat :=
  four_legged_tables + three_legged_tables

theorem tables_count
  (four_legged_tables three_legged_tables : Nat)
  (total_legs : Nat)
  (h1 : four_legged_tables = 16)
  (h2 : total_legs = 124)
  (h3 : 4 * four_legged_tables + 3 * three_legged_tables = total_legs) :
  total_tables four_legged_tables three_legged_tables = 36 :=
by
  sorry

end tables_count_l1278_127834


namespace required_circle_properties_l1278_127828

-- Define the two given circles' equations
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*x - 4 = 0

def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 + 6*y - 28 = 0

-- Define the line on which the center of the required circle lies
def line (x y : ℝ) : Prop :=
  x - y - 4 = 0

-- The equation of the required circle
def required_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - x + 7*y - 32 = 0

-- Prove that the required circle satisfies the conditions
theorem required_circle_properties (x y : ℝ) (hx : required_circle x y) :
  (∃ x y, circle1 x y ∧ circle2 x y ∧ required_circle x y) ∧
  (∃ x y, required_circle x y ∧ line x y) :=
by
  sorry

end required_circle_properties_l1278_127828


namespace alice_bob_speed_l1278_127853

theorem alice_bob_speed (x : ℝ) (h : x = 3 + 2 * Real.sqrt 7) :
  x^2 - 5 * x - 14 = 8 + 2 * Real.sqrt 7 - 5 := by
sorry

end alice_bob_speed_l1278_127853


namespace range_of_a_l1278_127882

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, -2 ≤ x ∧ x ≤ 1 ∧ 2 * a * x + 4 = 0) ↔ (-2 ≤ a ∧ a ≤ 1) :=
by
  sorry

end range_of_a_l1278_127882


namespace kth_term_in_sequence_l1278_127836

theorem kth_term_in_sequence (k : ℕ) (hk : 0 < k) : ℚ :=
  (2 * k) / (2 * k + 1)

end kth_term_in_sequence_l1278_127836


namespace size_of_first_type_package_is_5_l1278_127860

noncomputable def size_of_first_type_package (total_coffee : ℕ) (num_first_type : ℕ) (num_second_type : ℕ) (size_second_type : ℕ) : ℕ :=
  (total_coffee - num_second_type * size_second_type) / num_first_type

theorem size_of_first_type_package_is_5 :
  size_of_first_type_package 70 (4 + 2) 4 10 = 5 :=
by
  sorry

end size_of_first_type_package_is_5_l1278_127860


namespace parallel_condition_l1278_127859

def are_parallel (a b : ℝ × ℝ) : Prop := a.1 * b.2 = a.2 * b.1

theorem parallel_condition (x : ℝ) : 
  let a := (2, 1)
  let b := (3 * x ^ 2 - 1, x)
  (x = 1 → are_parallel a b) ∧ 
  ∃ x', x' ≠ 1 ∧ are_parallel a (3 * x' ^ 2 - 1, x') :=
by
  sorry

end parallel_condition_l1278_127859
