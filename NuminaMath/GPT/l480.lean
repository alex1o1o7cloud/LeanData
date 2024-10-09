import Mathlib

namespace volume_removed_percentage_l480_48084

noncomputable def volume_rect_prism (l w h : ℝ) : ℝ :=
  l * w * h

noncomputable def volume_cube (s : ℝ) : ℝ :=
  s * s * s

noncomputable def percent_removed (original_volume removed_volume : ℝ) : ℝ :=
  (removed_volume / original_volume) * 100

theorem volume_removed_percentage :
  let l := 18
  let w := 12
  let h := 10
  let cube_side := 4
  let num_cubes := 8
  let original_volume := volume_rect_prism l w h
  let removed_volume := num_cubes * volume_cube cube_side
  percent_removed original_volume removed_volume = 23.7 := 
sorry

end volume_removed_percentage_l480_48084


namespace frog_problem_l480_48001

theorem frog_problem 
  (N : ℕ) 
  (h1 : N < 50) 
  (h2 : N % 2 = 1) 
  (h3 : N % 3 = 1) 
  (h4 : N % 4 = 1) 
  (h5 : N % 5 = 0) : 
  N = 25 := 
  sorry

end frog_problem_l480_48001


namespace sum_of_number_and_reverse_l480_48097

theorem sum_of_number_and_reverse (a b : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9)
  (h3 : (10 * a + b) - (10 * b + a) = 7 * (a + b)) :
  (10 * a + b) + (10 * b + a) = 99 :=
by
  sorry

end sum_of_number_and_reverse_l480_48097


namespace roots_reciprocal_l480_48026

theorem roots_reciprocal (a b c r s : ℝ) (h_eqn : a ≠ 0) (h_roots : a * r^2 + b * r + c = 0 ∧ a * s^2 + b * s + c = 0) (h_cond : b^2 = 4 * a * c) : r * s = 1 :=
by
  -- Proof goes here
  sorry

end roots_reciprocal_l480_48026


namespace evaluate_expression_l480_48074

theorem evaluate_expression : (125^(1/3 : ℝ)) * (81^(-1/4 : ℝ)) * (32^(1/5 : ℝ)) = (10 / 3 : ℝ) :=
by
  sorry

end evaluate_expression_l480_48074


namespace pens_difference_proof_l480_48056

variables (A B M N X Y : ℕ)

-- Initial number of pens for Alex and Jane
def Alex_initial (A : ℕ) := A
def Jane_initial (B : ℕ) := B

-- Weekly multiplication factors for Alex and Jane
def Alex_weekly_growth (X : ℕ) := X
def Jane_weekly_growth (Y : ℕ) := Y

-- Number of pens after 4 weeks
def Alex_after_4_weeks (A X : ℕ) := A * X^4
def Jane_after_4_weeks (B Y : ℕ) := B * Y^4

-- Proving the difference in the number of pens
theorem pens_difference_proof (hM : M = A * X^4) (hN : N = B * Y^4) :
  M - N = (A * X^4) - (B * Y^4) :=
by sorry

end pens_difference_proof_l480_48056


namespace standard_lamp_probability_l480_48076

-- Define the given probabilities
def P_A1 : ℝ := 0.45
def P_A2 : ℝ := 0.40
def P_A3 : ℝ := 0.15

def P_B_given_A1 : ℝ := 0.70
def P_B_given_A2 : ℝ := 0.80
def P_B_given_A3 : ℝ := 0.81

-- Define the calculation for the total probability of B
def P_B : ℝ := P_A1 * P_B_given_A1 + P_A2 * P_B_given_A2 + P_A3 * P_B_given_A3

-- The statement to prove
theorem standard_lamp_probability : P_B = 0.7565 := by sorry

end standard_lamp_probability_l480_48076


namespace solve_for_s_l480_48039

theorem solve_for_s (k s : ℝ) 
  (h1 : 7 = k * 3^s) 
  (h2 : 126 = k * 9^s) : 
  s = 2 + Real.log 2 / Real.log 3 := by
  sorry

end solve_for_s_l480_48039


namespace vet_appointments_cost_l480_48070

variable (x : ℝ)

def JohnVetAppointments (x : ℝ) : Prop := 
  (x + 0.20 * x + 0.20 * x + 100 = 660)

theorem vet_appointments_cost :
  (∃ x : ℝ, JohnVetAppointments x) → x = 400 :=
by
  intro h
  obtain ⟨x, hx⟩ := h
  simp [JohnVetAppointments] at hx
  sorry

end vet_appointments_cost_l480_48070


namespace mod_product_l480_48002

theorem mod_product : (198 * 955) % 50 = 40 :=
by sorry

end mod_product_l480_48002


namespace sophia_collection_value_l480_48067

-- Define the conditions
def stamps_count : ℕ := 24
def partial_stamps_count : ℕ := 8
def partial_value : ℤ := 40
def stamp_value_per_each : ℤ := partial_value / partial_stamps_count
def total_value : ℤ := stamps_count * stamp_value_per_each

-- Statement of the conclusion that needs proving
theorem sophia_collection_value :
  total_value = 120 := by
  sorry

end sophia_collection_value_l480_48067


namespace similar_triangles_area_ratio_l480_48041

theorem similar_triangles_area_ratio (ratio_angles : ℕ) (area_larger : ℕ) (h_ratio : ratio_angles = 3) (h_area_larger : area_larger = 400) :
  ∃ area_smaller : ℕ, area_smaller = 36 :=
by
  sorry

end similar_triangles_area_ratio_l480_48041


namespace kaashish_problem_l480_48093

theorem kaashish_problem (x y : ℤ) (h : 2 * x + 3 * y = 100) (k : x = 28 ∨ y = 28) : x = 8 ∨ y = 8 :=
by
  sorry

end kaashish_problem_l480_48093


namespace string_length_l480_48045

theorem string_length (cylinder_circumference : ℝ)
  (total_loops : ℕ) (post_height : ℝ)
  (height_per_loop : ℝ := post_height / total_loops)
  (hypotenuse_per_loop : ℝ := Real.sqrt (height_per_loop ^ 2 + cylinder_circumference ^ 2))
  : total_loops = 5 → cylinder_circumference = 4 → post_height = 15 → hypotenuse_per_loop * total_loops = 25 :=
by 
  intros h1 h2 h3
  sorry

end string_length_l480_48045


namespace number_of_possible_lists_l480_48073

/-- 
Define the basic conditions: 
- 18 balls, numbered 1 through 18
- Selection process is repeated 4 times 
- Each selection is independent
- After each selection, the ball is replaced 
- We need to prove the total number of possible lists of four numbers 
--/
def number_of_balls : ℕ := 18
def selections : ℕ := 4

theorem number_of_possible_lists : (number_of_balls ^ selections) = 104976 := by
  sorry

end number_of_possible_lists_l480_48073


namespace mikes_earnings_l480_48025

-- Definitions based on the conditions:
def blade_cost : ℕ := 47
def game_count : ℕ := 9
def game_cost : ℕ := 6

-- The total money Mike made:
def total_money (M : ℕ) : Prop :=
  M - (blade_cost + game_count * game_cost) = 0

theorem mikes_earnings (M : ℕ) : total_money M → M = 101 :=
by
  sorry

end mikes_earnings_l480_48025


namespace find_coordinates_of_B_find_equation_of_BC_l480_48055

-- Problem 1: Prove that the coordinates of B are (10, 5)
theorem find_coordinates_of_B (A B : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0) :
  B = (10, 5) :=
sorry

-- Problem 2: Prove that the equation of line BC is 2x + 9y - 65 = 0
theorem find_equation_of_BC (A B C : ℝ × ℝ)
  (eq_med_C : ∀ (M : ℝ × ℝ), (M = ((B.1+3)/2, (B.2-1)/2) → 6 * M.1 + 10 * M.2 - 59 = 0))
  (eq_angle_bisector : B.1 - 4 * B.2 + 10 = 0)
  (coordinates_B : B = (10, 5)) :
  ∃ k : ℝ, ∀ P : ℝ × ℝ, (P.1 - C.1) / (P.2 - C.2) = k → 2 * P.1 + 9 * P.2 - 65 = 0 :=
sorry

end find_coordinates_of_B_find_equation_of_BC_l480_48055


namespace mindy_mork_earnings_ratio_l480_48040

theorem mindy_mork_earnings_ratio (M K : ℝ) (h1 : 0.20 * M + 0.30 * K = 0.225 * (M + K)) : M / K = 3 :=
by
  sorry

end mindy_mork_earnings_ratio_l480_48040


namespace A_and_C_work_together_in_2_hours_l480_48034

theorem A_and_C_work_together_in_2_hours
  (A_rate : ℚ)
  (B_rate : ℚ)
  (C_rate : ℚ)
  (A_4_hours : A_rate = 1 / 4)
  (B_12_hours : B_rate = 1 / 12)
  (B_and_C_3_hours : B_rate + C_rate = 1 / 3) :
  (A_rate + C_rate = 1 / 2) :=
by
  sorry

end A_and_C_work_together_in_2_hours_l480_48034


namespace avg_visitors_per_day_l480_48054

theorem avg_visitors_per_day 
  (avg_visitors_sundays : ℕ) 
  (avg_visitors_other_days : ℕ) 
  (total_days : ℕ) 
  (sundays : ℕ) 
  (other_days : ℕ)
  (hs : avg_visitors_sundays = 630)
  (ho : avg_visitors_other_days = 240)
  (td : total_days = 30)
  (sd : sundays = 4)
  (od : other_days = 26)
  : (4 * avg_visitors_sundays + 26 * avg_visitors_other_days) / 30 = 292 := 
by
  sorry

end avg_visitors_per_day_l480_48054


namespace triangle_altitudes_perfect_square_l480_48000

theorem triangle_altitudes_perfect_square
  (a b c : ℤ)
  (h : (2 * (↑a * ↑b * ↑c )) = (2 * (↑a * ↑c ) + 2 * (↑a * ↑b))) :
  ∃ k : ℤ, a^2 + b^2 + c^2 = k^2 :=
by
  sorry

end triangle_altitudes_perfect_square_l480_48000


namespace find_m_range_l480_48060

-- Defining the function and conditions
variable {f : ℝ → ℝ}
variable {m : ℝ}

-- Prove if given the conditions, then the range of m is as specified
theorem find_m_range (h1 : ∀ x, f (-x) = -f x) 
                     (h2 : ∀ x, -2 < x ∧ x < 2 → f (x) > f (x+1)) 
                     (h3 : -2 < m - 1 ∧ m - 1 < 2) 
                     (h4 : -2 < 2 * m - 1 ∧ 2 * m - 1 < 2) 
                     (h5 : f (m - 1) + f (2 * m - 1) > 0) :
  -1/2 < m ∧ m < 2/3 :=
sorry

end find_m_range_l480_48060


namespace yellow_surface_area_min_fraction_l480_48019

/-- 
  Given a larger cube with 4-inch edges, constructed from 64 smaller cubes (each with 1-inch edge),
  where 50 cubes are colored blue, and 14 cubes are colored yellow. 
  If the large cube is crafted to display the minimum possible yellow surface area externally,
  then the fraction of the surface area of the large cube that is yellow is 7/48.
-/
theorem yellow_surface_area_min_fraction (n_smaller_cubes blue_cubes yellow_cubes : ℕ) 
  (edge_small edge_large : ℕ) (surface_area_larger_cube yellow_surface_min : ℕ) :
  edge_small = 1 → edge_large = 4 → n_smaller_cubes = 64 → 
  blue_cubes = 50 → yellow_cubes = 14 →
  surface_area_larger_cube = 96 → yellow_surface_min = 14 → 
  (yellow_surface_min : ℚ) / (surface_area_larger_cube : ℚ) = 7 / 48 := 
by 
  intros h_edge_small h_edge_large h_n h_blue h_yellow h_surface_area h_yellow_surface
  sorry

end yellow_surface_area_min_fraction_l480_48019


namespace greatest_integer_with_gcf_5_l480_48047

theorem greatest_integer_with_gcf_5 :
  ∃ n, n < 200 ∧ gcd n 30 = 5 ∧ ∀ m, m < 200 ∧ gcd m 30 = 5 → m ≤ n :=
by
  sorry

end greatest_integer_with_gcf_5_l480_48047


namespace theresa_hours_l480_48027

theorem theresa_hours (h1 h2 h3 h4 h5 h6 : ℕ) (avg : ℕ) (x : ℕ) 
  (H_cond : h1 = 10 ∧ h2 = 8 ∧ h3 = 9 ∧ h4 = 11 ∧ h5 = 6 ∧ h6 = 8)
  (H_avg : avg = 9) : 
  (h1 + h2 + h3 + h4 + h5 + h6 + x) / 7 = avg ↔ x = 11 :=
by
  sorry

end theresa_hours_l480_48027


namespace water_volume_per_minute_l480_48095

theorem water_volume_per_minute 
  (depth : ℝ) (width : ℝ) (flow_kmph : ℝ)
  (h_depth : depth = 8) (h_width : width = 25) (h_flow_rate : flow_kmph = 8) :
  (width * depth * (flow_kmph * 1000 / 60)) = 26666.67 :=
by 
  have flow_m_per_min := flow_kmph * 1000 / 60
  have area := width * depth
  have volume_per_minute := area * flow_m_per_min
  sorry

end water_volume_per_minute_l480_48095


namespace lamp_turn_off_ways_l480_48042

theorem lamp_turn_off_ways : 
  ∃ (ways : ℕ), ways = 10 ∧
  (∃ (n : ℕ) (m : ℕ), 
    n = 6 ∧  -- 6 lamps in a row
    m = 2 ∧  -- turn off 2 of them
    ways = Nat.choose (n - m + 1) m) := -- 2 adjacent lamps cannot be turned off
by
  -- Proof will be provided here.
  sorry

end lamp_turn_off_ways_l480_48042


namespace smallest_four_digit_solution_l480_48014

theorem smallest_four_digit_solution :
  ∃ x : ℕ, 1000 ≤ x ∧ x < 10000 ∧
  (3 * x ≡ 6 [MOD 12]) ∧
  (5 * x + 20 ≡ 25 [MOD 15]) ∧
  (3 * x - 2 ≡ 2 * x [MOD 35]) ∧
  x = 1274 :=
by
  sorry

end smallest_four_digit_solution_l480_48014


namespace c_investment_l480_48037

theorem c_investment (x : ℝ) (h1 : 5000 / (5000 + 8000 + x) * 88000 = 36000) : 
  x = 20454.5 :=
by
  sorry

end c_investment_l480_48037


namespace parallel_slope_l480_48046

theorem parallel_slope {x1 y1 x2 y2 : ℝ} (h : x1 = 3 ∧ y1 = -2 ∧ x2 = 1 ∧ y2 = 5) :
    let slope := (y2 - y1) / (x2 - x1)
    slope = -7 / 2 := 
by 
    sorry

end parallel_slope_l480_48046


namespace range_of_j_l480_48032

def h (x: ℝ) : ℝ := 2 * x + 1
def j (x: ℝ) : ℝ := h (h (h (h (h x))))

theorem range_of_j :
  ∀ x, -1 ≤ x ∧ x ≤ 3 → -1 ≤ j x ∧ j x ≤ 127 :=
by 
  intros x hx
  sorry

end range_of_j_l480_48032


namespace amplitude_of_resultant_wave_l480_48044

noncomputable def y1 (t : ℝ) := 3 * Real.sqrt 2 * Real.sin (100 * Real.pi * t)
noncomputable def y2 (t : ℝ) := 3 * Real.sin (100 * Real.pi * t - Real.pi / 4)
noncomputable def y (t : ℝ) := y1 t + y2 t

theorem amplitude_of_resultant_wave :
  ∃ R : ℝ, R = 3 * Real.sqrt 5 ∧ ∀ t : ℝ, y t = R * Real.sin (100 * Real.pi * t - θ) :=
by
  let y_combined := y
  use 3 * Real.sqrt 5
  sorry

end amplitude_of_resultant_wave_l480_48044


namespace katie_sold_4_bead_necklaces_l480_48061

theorem katie_sold_4_bead_necklaces :
  ∃ (B : ℕ), 
    (∃ (G : ℕ), G = 3) ∧ 
    (∃ (C : ℕ), C = 3) ∧ 
    (∃ (T : ℕ), T = 21) ∧ 
    B * 3 + 3 * 3 = 21 :=
sorry

end katie_sold_4_bead_necklaces_l480_48061


namespace a_sufficient_but_not_necessary_l480_48059

theorem a_sufficient_but_not_necessary (a : ℝ) : 
  (a = 1 → |a| = 1) ∧ (¬ (|a| = 1 → a = 1)) :=
by 
  sorry

end a_sufficient_but_not_necessary_l480_48059


namespace chosen_number_is_120_l480_48003

theorem chosen_number_is_120 (x : ℤ) (h : 2 * x - 138 = 102) : x = 120 :=
sorry

end chosen_number_is_120_l480_48003


namespace domain_of_sqrt_function_l480_48048

theorem domain_of_sqrt_function :
  {x : ℝ | 0 ≤ x + 1} = {x : ℝ | -1 ≤ x} :=
by {
  sorry
}

end domain_of_sqrt_function_l480_48048


namespace investment_time_R_l480_48053

theorem investment_time_R (x t : ℝ) 
  (h1 : 7 * 5 * x / (5 * 7 * x) = 7 / 9)
  (h2 : 3 * t * x / (5 * 7 * x) = 4 / 9) : 
  t = 140 / 27 :=
by
  -- Placeholder for the proof, which is not required in this step.
  sorry

end investment_time_R_l480_48053


namespace tens_digit_of_19_pow_2023_l480_48094

theorem tens_digit_of_19_pow_2023 :
  (19^2023 % 100) / 10 % 10 = 5 := 
sorry

end tens_digit_of_19_pow_2023_l480_48094


namespace probability_of_distinct_divisors_l480_48091

theorem probability_of_distinct_divisors :
  ∃ (m n : ℕ), (m.gcd n = 1) ∧ (m / n) = 125 / 158081 :=
by
  sorry

end probability_of_distinct_divisors_l480_48091


namespace problem_1_problem_2_l480_48050

-- Definition f
def f (a x : ℝ) : ℝ := a * x + 3 - abs (2 * x - 1)

-- Problem 1: If a = 1, prove ∀ x, f(1, x) ≤ 2
theorem problem_1 : (∀ x : ℝ, f 1 x ≤ 2) :=
sorry

-- Problem 2: The range of a for which f has a maximum value is -2 ≤ a ≤ 2
theorem problem_2 : (∀ a : ℝ, (∀ x : ℝ, (2 * x - 1 > 0 -> (f a x) ≤ (f a ((4 - a) / (2 * (4 - a))))) 
                        ∧ (2 * x - 1 ≤ 0 -> (f a x) ≤ (f a (1 - 2 / (1 - a))))) 
                        ↔ -2 ≤ a ∧ a ≤ 2) :=
sorry

end problem_1_problem_2_l480_48050


namespace circle_radius_c_eq_32_l480_48089

theorem circle_radius_c_eq_32 :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x-4)^2 + (y+5)^2 = 9) :=
by
  use 32
  sorry

end circle_radius_c_eq_32_l480_48089


namespace girls_additional_laps_l480_48098

def distance_per_lap : ℚ := 1 / 6
def boys_laps : ℕ := 34
def boys_distance : ℚ := boys_laps * distance_per_lap
def girls_distance : ℚ := 9
def additional_distance : ℚ := girls_distance - boys_distance
def additional_laps (distance : ℚ) (lap_distance : ℚ) : ℚ := distance / lap_distance

theorem girls_additional_laps :
  additional_laps additional_distance distance_per_lap = 20 := 
by
  sorry

end girls_additional_laps_l480_48098


namespace sophomores_in_program_l480_48004

theorem sophomores_in_program (total_students : ℕ) (not_sophomores_nor_juniors : ℕ) 
    (percentage_sophomores_debate : ℚ) (percentage_juniors_debate : ℚ) 
    (eq_debate_team : ℚ) (total_students := 40) 
    (not_sophomores_nor_juniors := 5) 
    (percentage_sophomores_debate := 0.20) 
    (percentage_juniors_debate := 0.25) 
    (eq_debate_team := (percentage_sophomores_debate * S = percentage_juniors_debate * J)) :
    ∀ (S J : ℚ), S + J = total_students - not_sophomores_nor_juniors → 
    (S = 5 * J / 4) → S = 175 / 9 := 
by 
  sorry

end sophomores_in_program_l480_48004


namespace ratio_sum_l480_48065

theorem ratio_sum {x y : ℚ} (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_sum_l480_48065


namespace no_perfect_square_for_nnplus1_l480_48011

theorem no_perfect_square_for_nnplus1 :
  ¬ ∃ (n : ℕ), 0 < n ∧ ∃ (k : ℕ), n * (n + 1) = k * k :=
sorry

end no_perfect_square_for_nnplus1_l480_48011


namespace min_value_proof_l480_48005

noncomputable def min_value_of_expression (a b c d e f g h : ℝ) : ℝ :=
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2

theorem min_value_proof (a b c d e f g h : ℝ)
  (h1 : a * b * c * d = 8)
  (h2 : e * f * g * h = 16) :
  ∃ (x : ℝ), x = 32 ∧ min_value_of_expression a b c d e f g h = x :=
by
  use 32
  sorry

end min_value_proof_l480_48005


namespace find_x_l480_48063

theorem find_x (x y : ℤ) (h1 : x + y = 4) (h2 : x - y = 36) : x = 20 :=
by
  sorry

end find_x_l480_48063


namespace factorize_expression_l480_48085

theorem factorize_expression (m x : ℝ) : 
  m^3 * (x - 2) - m * (x - 2) = m * (x - 2) * (m + 1) * (m - 1) := 
by sorry

end factorize_expression_l480_48085


namespace rounding_addition_to_tenth_l480_48016

def number1 : Float := 96.23
def number2 : Float := 47.849

theorem rounding_addition_to_tenth (sum : Float) : 
    sum = number1 + number2 →
    Float.round (sum * 10) / 10 = 144.1 :=
by
  intro h
  rw [h]
  norm_num
  sorry -- Skipping the actual rounding proof

end rounding_addition_to_tenth_l480_48016


namespace ratio_of_teenagers_to_toddlers_l480_48043

theorem ratio_of_teenagers_to_toddlers
  (total_children : ℕ)
  (number_of_toddlers : ℕ)
  (number_of_newborns : ℕ)
  (h1 : total_children = 40)
  (h2 : number_of_toddlers = 6)
  (h3 : number_of_newborns = 4)
  : (total_children - number_of_toddlers - number_of_newborns) / number_of_toddlers = 5 :=
by
  sorry

end ratio_of_teenagers_to_toddlers_l480_48043


namespace difference_between_x_and_y_l480_48030

theorem difference_between_x_and_y 
  (x y : ℕ) 
  (h1 : 3 ^ x * 4 ^ y = 531441) 
  (h2 : x = 12) : x - y = 12 := 
by 
  sorry

end difference_between_x_and_y_l480_48030


namespace sum_of_digits_is_3_l480_48038

-- We introduce variables for the digits a and b, and the number
variables (a b : ℕ)

-- Conditions: a and b must be digits, and the number must satisfy the given equation
-- One half of (10a + b) exceeds its one fourth by 3
def valid_digits (a b : ℕ) : Prop := a < 10 ∧ b < 10
def equation_condition (a b : ℕ) : Prop := 2 * (10 * a + b) = (10 * a + b) + 12

-- The number is two digits number
def two_digits_number (a b : ℕ) : ℕ := 10 * a + b

-- Final statement combining all conditions and proving the desired sum of digits
theorem sum_of_digits_is_3 : 
  ∀ (a b : ℕ), valid_digits a b → equation_condition a b → a + b = 3 := 
by
  intros a b h1 h2
  sorry

end sum_of_digits_is_3_l480_48038


namespace selena_taco_packages_l480_48068

-- Define the problem conditions
def tacos_per_package : ℕ := 4
def shells_per_package : ℕ := 6
def min_tacos : ℕ := 60
def min_shells : ℕ := 60

-- Lean statement to prove the smallest number of taco packages needed
theorem selena_taco_packages :
  ∃ n : ℕ, (n * tacos_per_package ≥ min_tacos) ∧ (∃ m : ℕ, (m * shells_per_package ≥ min_shells) ∧ (n * tacos_per_package = m * shells_per_package) ∧ n = 15) := 
by {
  sorry
}

end selena_taco_packages_l480_48068


namespace cos_sin_inequality_inequality_l480_48033

noncomputable def proof_cos_sin_inequality (a b : ℝ) (cos_x sin_x: ℝ) : Prop :=
  (cos_x ^ 2 = a) → (sin_x ^ 2 = b) → (a + b = 1) → (1 / 4 ≤ a ^ 3 + b ^ 3 ∧ a ^ 3 + b ^ 3 ≤ 1)

theorem cos_sin_inequality_inequality (a b : ℝ) (cos_x sin_x : ℝ) :
  proof_cos_sin_inequality a b cos_x sin_x :=
  by { sorry }

end cos_sin_inequality_inequality_l480_48033


namespace largest_prime_factor_problem_l480_48075

def largest_prime_factor (n : ℕ) : ℕ :=
  -- This function calculates the largest prime factor of n
  sorry

theorem largest_prime_factor_problem :
  largest_prime_factor 57 = 19 ∧
  largest_prime_factor 133 = 19 ∧
  ∀ n, n = 63 ∨ n = 85 ∨ n = 143 → largest_prime_factor n < 19 :=
by
  sorry

end largest_prime_factor_problem_l480_48075


namespace polygon_edges_l480_48024

theorem polygon_edges :
  ∃ a b : ℕ, a + b = 2014 ∧
              (a * (a - 3) / 2 + b * (b - 3) / 2 = 1014053) ∧
              a ≤ b ∧
              a = 952 :=
by
  sorry

end polygon_edges_l480_48024


namespace number_of_ways_to_choose_officers_l480_48096

open Nat

theorem number_of_ways_to_choose_officers (n : ℕ) (h : n = 8) : 
  n * (n - 1) * (n - 2) = 336 := by
  sorry

end number_of_ways_to_choose_officers_l480_48096


namespace total_charge_for_2_hours_l480_48052

theorem total_charge_for_2_hours (A F : ℕ) (h1 : F = A + 35) (h2 : F + 4 * A = 350) : 
  F + A = 161 := 
by 
  sorry

end total_charge_for_2_hours_l480_48052


namespace find_two_primes_l480_48092

-- Define what it means for a number to be prime
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 ∧ m ≠ n → n % m ≠ 0

-- Prove the existence of two specific prime numbers with the desired properties
theorem find_two_primes :
  ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p = 2 ∧ q = 5 ∧ is_prime (p + q) ∧ is_prime (q - p) :=
by
  exists 2
  exists 5
  repeat {split}
  sorry

end find_two_primes_l480_48092


namespace distance_from_highest_point_of_sphere_to_bottom_of_glass_l480_48009

theorem distance_from_highest_point_of_sphere_to_bottom_of_glass :
  ∀ (x y : ℝ),
  x^2 = 2 * y →
  0 ≤ y ∧ y < 15 →
  ∃ b : ℝ, (x^2 + (y - b)^2 = 9) ∧ b = 5 ∧ (b + 3 = 8) :=
by
  sorry

end distance_from_highest_point_of_sphere_to_bottom_of_glass_l480_48009


namespace monotonic_power_function_l480_48035

noncomputable def power_function (a : ℝ) (x : ℝ) : ℝ := (a^2 - 2 * a - 2) * x^a

theorem monotonic_power_function (a : ℝ) (h1 : ∀ x : ℝ, ( ∀ x1 x2 : ℝ, x1 < x2 → power_function a x1 < power_function a x2 ) )
  (h2 : a^2 - 2 * a - 2 = 1) (h3 : a > 0) : a = 3 :=
by
  sorry

end monotonic_power_function_l480_48035


namespace initial_number_is_31_l480_48007

theorem initial_number_is_31 (N : ℕ) (h : ∃ k : ℕ, N - 10 = 21 * k) : N = 31 :=
sorry

end initial_number_is_31_l480_48007


namespace find_a₁_l480_48069

noncomputable def geometric_sequence (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * q ^ n

noncomputable def sequence_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q ^ n) / (1 - q)

variables (a₁ q : ℝ)
-- Condition: The common ratio should not be 1.
axiom hq : q ≠ 1
-- Condition: Second term of the sequence a₂ = 1
axiom ha₂ : geometric_sequence a₁ q 1 = 1
-- Condition: 9S₃ = S₆
axiom hsum : 9 * sequence_sum a₁ q 3 = sequence_sum a₁ q 6

theorem find_a₁ : a₁ = 1 / 2 :=
  sorry

end find_a₁_l480_48069


namespace no_solution_to_system_l480_48086

theorem no_solution_to_system :
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 12 ∧ 9 * x - 12 * y = 15) :=
by
  sorry

end no_solution_to_system_l480_48086


namespace meeting_point_2015_is_C_l480_48029

-- Given definitions based on conditions
variable (x y : ℝ) -- Speeds of the motorcycle and the cyclist
variable (A B C D : Point) -- Points on segment AB
variable (meetings : ℕ → Point) -- Function representing the meeting point sequence

-- Conditions stating the alternating meeting pattern
axiom start_at_A (n : ℕ) : meetings (2 * n + 1) = C
axiom start_at_B (n : ℕ) : meetings (2 * n + 2) = D

-- The theorem statement to be proved
theorem meeting_point_2015_is_C : meetings 2015 = C := sorry

end meeting_point_2015_is_C_l480_48029


namespace value_range_sin_neg_l480_48088

theorem value_range_sin_neg (x : ℝ) (h : x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) : 
  Set.Icc (-1) (Real.sqrt 2 / 2) ( - (Real.sin x) ) :=
sorry

end value_range_sin_neg_l480_48088


namespace prob_exceeds_175_l480_48021

-- Definitions from the conditions
def prob_less_than_160 (p : ℝ) : Prop := p = 0.2
def prob_160_to_175 (p : ℝ) : Prop := p = 0.5

-- The mathematical equivalence proof we need
theorem prob_exceeds_175 (p₁ p₂ p₃ : ℝ) 
  (h₁ : prob_less_than_160 p₁) 
  (h₂ : prob_160_to_175 p₂) 
  (H : p₃ = 1 - (p₁ + p₂)) :
  p₃ = 0.3 := 
by
  -- Placeholder for proof
  sorry

end prob_exceeds_175_l480_48021


namespace triangle_obtuse_of_cos_relation_l480_48017

theorem triangle_obtuse_of_cos_relation
  (a b c : ℝ)
  (A B C : ℝ)
  (hTriangle : A + B + C = Real.pi)
  (hSides : a^2 = b^2 + c^2 - 2*b*c*Real.cos A)
  (hSides' : b^2 = a^2 + c^2 - 2*a*c*Real.cos B)
  (hSides'' : c^2 = a^2 + b^2 - 2*a*b*Real.cos C)
  (hRelation : a * Real.cos C = b + 2/3 * c) :
 ∃ (A' : ℝ), A' = A ∧ A > (Real.pi / 2) := 
sorry

end triangle_obtuse_of_cos_relation_l480_48017


namespace train_speed_120_kmph_l480_48036

theorem train_speed_120_kmph (t : ℝ) (d : ℝ) (h_t : t = 9) (h_d : d = 300) : 
    (d / t) * 3.6 = 120 :=
by
  sorry

end train_speed_120_kmph_l480_48036


namespace value_of_xyz_l480_48051

open Real

theorem value_of_xyz (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 37)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 11) :
  x * y * z = 26 / 3 := 
  sorry

end value_of_xyz_l480_48051


namespace geometric_progression_l480_48006

theorem geometric_progression (p : ℝ) 
  (a b c : ℝ)
  (h1 : a = p - 2)
  (h2 : b = 2 * Real.sqrt p)
  (h3 : c = -3 - p)
  (h4 : b ^ 2 = a * c) : 
  p = 1 := 
by 
  sorry

end geometric_progression_l480_48006


namespace impossible_tiling_conditions_l480_48057

theorem impossible_tiling_conditions (m n : ℕ) :
  ¬ (∃ (a b : ℕ), (a - 1) * 4 + (b + 1) * 4 = m * n ∧ a * 4 % 4 = 2 ∧ b * 4 % 4 = 0) :=
sorry

end impossible_tiling_conditions_l480_48057


namespace larger_solution_quadratic_l480_48049

theorem larger_solution_quadratic :
  (∃ a b : ℝ, a ≠ b ∧ (a = 9) ∧ (b = -2) ∧
              (∀ x : ℝ, x^2 - 7 * x - 18 = 0 → (x = a ∨ x = b))) →
  9 = max a b :=
by
  sorry

end larger_solution_quadratic_l480_48049


namespace exponent_problem_proof_l480_48090

theorem exponent_problem_proof :
  3 * 3^4 - 27^60 / 27^58 = -486 :=
by
  sorry

end exponent_problem_proof_l480_48090


namespace solve_for_c_l480_48081

theorem solve_for_c (a b c : ℝ) (h : 1/a - 1/b = 2/c) : c = (a * b * (b - a)) / 2 := by
  sorry

end solve_for_c_l480_48081


namespace area_triangle_ABC_l480_48064

theorem area_triangle_ABC (AB CD height : ℝ) 
  (h_parallel : AB + CD = 20)
  (h_ratio : CD = 3 * AB)
  (h_height : height = (2 * 20) / (AB + CD)) :
  (1 / 2) * AB * height = 5 := sorry

end area_triangle_ABC_l480_48064


namespace find_a1_l480_48012

theorem find_a1 (f : ℝ → ℝ) (a : ℕ → ℝ) (h₀ : ∀ x, f x = (x - 1)^3 + x + 2)
(h₁ : ∀ n, a (n + 1) = a n + 1/2)
(h₂ : f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 18) :
a 1 = -1 / 4 :=
by
  sorry

end find_a1_l480_48012


namespace negation_of_exists_inequality_l480_48022

theorem negation_of_exists_inequality :
  ¬ (∃ x : ℝ, x * x + 4 * x + 5 ≤ 0) ↔ ∀ x : ℝ, x * x + 4 * x + 5 > 0 :=
by
  sorry

end negation_of_exists_inequality_l480_48022


namespace largest_exponent_l480_48008

theorem largest_exponent : 
  ∀ (a b c d e : ℕ), a = 2^5000 → b = 3^4000 → c = 4^3000 → d = 5^2000 → e = 6^1000 → b > a ∧ b > c ∧ b > d ∧ b > e :=
by
  sorry

end largest_exponent_l480_48008


namespace least_n_exceeds_product_l480_48015

def product_exceeds (n : ℕ) : Prop :=
  10^(n * (n + 1) / 18) > 10^6

theorem least_n_exceeds_product (n : ℕ) (h : n = 12) : product_exceeds n :=
by
  rw [h]
  sorry

end least_n_exceeds_product_l480_48015


namespace least_three_digit_multiple_13_l480_48013

theorem least_three_digit_multiple_13 : 
  ∃ n : ℕ, (n ≥ 100) ∧ (∃ k : ℕ, n = 13 * k) ∧ ∀ m, m < n → (m < 100 ∨ ¬∃ k : ℕ, m = 13 * k) :=
by
  sorry

end least_three_digit_multiple_13_l480_48013


namespace chromium_alloy_l480_48066

theorem chromium_alloy (x : ℝ) (h1 : 0.12 * x + 0.10 * 35 = 0.106 * (x + 35)) : x = 15 := 
by 
  -- statement only, no proof required.
  sorry

end chromium_alloy_l480_48066


namespace entrance_exam_correct_answers_l480_48082

theorem entrance_exam_correct_answers (c w : ℕ) 
  (h1 : c + w = 70) 
  (h2 : 3 * c - w = 38) : 
  c = 27 := 
sorry

end entrance_exam_correct_answers_l480_48082


namespace playground_area_l480_48023

noncomputable def length (w : ℝ) := 2 * w + 30
noncomputable def perimeter (l w : ℝ) := 2 * (l + w)
noncomputable def area (l w : ℝ) := l * w

theorem playground_area :
  ∃ (w l : ℝ), length w = l ∧ perimeter l w = 700 ∧ area l w = 25955.56 :=
by {
  sorry
}

end playground_area_l480_48023


namespace polynomial_irreducible_if_not_divisible_by_5_l480_48087

theorem polynomial_irreducible_if_not_divisible_by_5 (k : ℤ) (h1 : ¬ ∃ m : ℤ, k = 5 * m) :
    ¬ ∃ (f g : Polynomial ℤ), (f.degree < 5) ∧ (f * g = x^5 - x + Polynomial.C k) :=
  sorry

end polynomial_irreducible_if_not_divisible_by_5_l480_48087


namespace value_of_a_plus_b_l480_48079

noncomputable def f (x : ℝ) := abs (Real.log (x + 1))

theorem value_of_a_plus_b (a b : ℝ) (h1 : a < b) 
  (h2 : f a = f (- (b + 1) / (b + 2))) 
  (h3 : f (10 * a + 6 * b + 21) = 4 * Real.log 2) : 
  a + b = -11 / 15 := 
by 
  sorry

end value_of_a_plus_b_l480_48079


namespace inequality_proof_l480_48099

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
(h : Real.sqrt a + Real.sqrt b + Real.sqrt c = 3) :
  (a + b) / (2 + a + b) + (b + c) / (2 + b + c) + (c + a) / (2 + c + a) ≥ 3 / 2 := 
sorry

end inequality_proof_l480_48099


namespace sum_consecutive_not_power_of_two_l480_48018

theorem sum_consecutive_not_power_of_two :
  ∀ n k : ℕ, ∀ x : ℕ, n > 0 → k > 0 → (n * (n + 2 * k - 1)) / 2 ≠ 2 ^ x := by
  sorry

end sum_consecutive_not_power_of_two_l480_48018


namespace evaluate_five_applications_of_f_l480_48077

def f (x : ℤ) : ℤ :=
  if x ≥ 0 then x + 5 else -x^2 - 3

theorem evaluate_five_applications_of_f :
  f (f (f (f (f (-1))))) = -17554795004 :=
by
  sorry

end evaluate_five_applications_of_f_l480_48077


namespace steve_family_time_l480_48080

theorem steve_family_time :
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  hours_per_day - (hours_sleeping + hours_school + hours_assignments) = 10 :=
by
  let hours_per_day := 24
  let hours_sleeping := hours_per_day * (1/3)
  let hours_school := hours_per_day * (1/6)
  let hours_assignments := hours_per_day * (1/12)
  sorry

end steve_family_time_l480_48080


namespace equivalent_angle_l480_48062

theorem equivalent_angle (θ : ℝ) : 
  (∃ k : ℤ, θ = k * 360 + 257) ↔ θ = -463 ∨ (∃ k : ℤ, θ = k * 360 + 257) :=
by
  sorry

end equivalent_angle_l480_48062


namespace usable_area_l480_48071

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def pond_side : ℕ := 4

theorem usable_area :
  garden_length * garden_width - pond_side * pond_side = 344 :=
by
  sorry

end usable_area_l480_48071


namespace stickers_distribution_l480_48058

-- Definitions for initial sticker quantities and stickers given to first four friends
def initial_space_stickers : ℕ := 120
def initial_cat_stickers : ℕ := 80
def initial_dinosaur_stickers : ℕ := 150
def initial_superhero_stickers : ℕ := 45

def given_space_stickers : ℕ := 25
def given_cat_stickers : ℕ := 13
def given_dinosaur_stickers : ℕ := 33
def given_superhero_stickers : ℕ := 29

-- Definitions for remaining stickers calculation
def remaining_space_stickers : ℕ := initial_space_stickers - given_space_stickers
def remaining_cat_stickers : ℕ := initial_cat_stickers - given_cat_stickers
def remaining_dinosaur_stickers : ℕ := initial_dinosaur_stickers - given_dinosaur_stickers
def remaining_superhero_stickers : ℕ := initial_superhero_stickers - given_superhero_stickers

def total_remaining_stickers : ℕ := remaining_space_stickers + remaining_cat_stickers + remaining_dinosaur_stickers + remaining_superhero_stickers

-- Definition for number of each type of new sticker
def each_new_type_stickers : ℕ := total_remaining_stickers / 4
def remainder_stickers : ℕ := total_remaining_stickers % 4

-- Statement to be proved
theorem stickers_distribution :
  ∃ X : ℕ, X = 3 ∧ each_new_type_stickers = 73 :=
by
  sorry

end stickers_distribution_l480_48058


namespace fraction_of_Bs_l480_48072

theorem fraction_of_Bs 
  (num_students : ℕ)
  (As_fraction : ℚ)
  (Cs_fraction : ℚ)
  (Ds_number : ℕ)
  (total_students : ℕ) 
  (h1 : As_fraction = 1 / 5) 
  (h2 : Cs_fraction = 1 / 2) 
  (h3 : Ds_number = 40) 
  (h4 : total_students = 800) : 
  num_students / total_students = 1 / 4 :=
by
sorry

end fraction_of_Bs_l480_48072


namespace mary_needs_6_cups_l480_48078
-- We import the whole Mathlib library first.

-- We define the conditions and the question.
def total_cups : ℕ := 8
def cups_added : ℕ := 2
def cups_needed : ℕ := total_cups - cups_added

-- We state the theorem we need to prove.
theorem mary_needs_6_cups : cups_needed = 6 :=
by
  -- We use a placeholder for the proof.
  sorry

end mary_needs_6_cups_l480_48078


namespace ball_hits_ground_at_time_l480_48020

theorem ball_hits_ground_at_time :
  ∃ t : ℚ, -9.8 * t^2 + 5.6 * t + 10 = 0 ∧ t = 131 / 98 :=
by
  sorry

end ball_hits_ground_at_time_l480_48020


namespace integral_equality_l480_48031

theorem integral_equality :
  ∫ x in (-1 : ℝ)..(1 : ℝ), (Real.tan x) ^ 11 + (Real.cos x) ^ 21
  = 2 * ∫ x in (0 : ℝ)..(1 : ℝ), (Real.cos x) ^ 21 :=
by
  sorry

end integral_equality_l480_48031


namespace fraction_negative_iff_x_lt_2_l480_48010

theorem fraction_negative_iff_x_lt_2 (x : ℝ) :
  (-5) / (2 - x) < 0 ↔ x < 2 := by
  sorry

end fraction_negative_iff_x_lt_2_l480_48010


namespace prime_pairs_solution_l480_48083

def is_prime (n : ℕ) : Prop := Nat.Prime n

def conditions (p q : ℕ) : Prop := 
  p^2 ∣ q^3 + 1 ∧ q^2 ∣ p^6 - 1

theorem prime_pairs_solution :
  ({(p, q) | is_prime p ∧ is_prime q ∧ conditions p q} = {(3, 2), (2, 3)}) :=
by
  sorry

end prime_pairs_solution_l480_48083


namespace combined_garden_area_l480_48028

def garden_area (length width : ℕ) : ℕ :=
  length * width

def total_area (count length width : ℕ) : ℕ :=
  count * garden_area length width

theorem combined_garden_area :
  let M_length := 16
  let M_width := 5
  let M_count := 3
  let Ma_length := 8
  let Ma_width := 4
  let Ma_count := 2
  total_area M_count M_length M_width + total_area Ma_count Ma_length Ma_width = 304 :=
by
  sorry

end combined_garden_area_l480_48028
