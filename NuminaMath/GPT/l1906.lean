import Mathlib

namespace smallest_integer_in_odd_set_l1906_190640

theorem smallest_integer_in_odd_set (is_odd: ℤ → Prop)
  (median: ℤ) (greatest: ℤ) (smallest: ℤ) 
  (h1: median = 126)
  (h2: greatest = 153) 
  (h3: ∀ x, is_odd x ↔ ∃ k: ℤ, x = 2*k + 1)
  (h4: ∀ a b c, median = (a+b) / 2 → c = a → a ≤ b)
  : 
  smallest = 100 :=
sorry

end smallest_integer_in_odd_set_l1906_190640


namespace estimate_total_fish_l1906_190674

theorem estimate_total_fish (marked : ℕ) (sample_size : ℕ) (marked_in_sample : ℕ) (x : ℝ) 
  (h1 : marked = 50) 
  (h2 : sample_size = 168) 
  (h3 : marked_in_sample = 8) 
  (h4 : sample_size * 50 = marked_in_sample * x) : 
  x = 1050 := 
sorry

end estimate_total_fish_l1906_190674


namespace cost_of_paving_is_correct_l1906_190607

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate_per_sq_metre : ℝ := 400
def area_of_rectangle (l: ℝ) (w: ℝ) : ℝ := l * w
def cost_of_paving_floor (area: ℝ) (rate: ℝ) : ℝ := area * rate

theorem cost_of_paving_is_correct
  (h_length: length = 5.5)
  (h_width: width = 3.75)
  (h_rate: rate_per_sq_metre = 400):
  cost_of_paving_floor (area_of_rectangle length width) rate_per_sq_metre = 8250 :=
  by {
    sorry
  }

end cost_of_paving_is_correct_l1906_190607


namespace correct_proposition_l1906_190665

-- Define the propositions as Lean 4 statements.
def PropA (a : ℝ) : Prop := a^4 + a^2 = a^6
def PropB (a : ℝ) : Prop := (-2 * a^2)^3 = -6 * a^8
def PropC (a : ℝ) : Prop := 6 * a - a = 5
def PropD (a : ℝ) : Prop := a^2 * a^3 = a^5

-- The main theorem statement that only PropD is true.
theorem correct_proposition (a : ℝ) : ¬ PropA a ∧ ¬ PropB a ∧ ¬ PropC a ∧ PropD a :=
by
  sorry

end correct_proposition_l1906_190665


namespace notebook_cost_l1906_190620

theorem notebook_cost (s n c : ℕ) (h1 : s ≥ 19) (h2 : n > 2) (h3 : c > n) (h4 : s * c * n = 3969) : c = 27 :=
sorry

end notebook_cost_l1906_190620


namespace quadratic_root_sum_l1906_190622

theorem quadratic_root_sum (k : ℝ) (h : k ≤ 1 / 2) : 
  ∃ (α β : ℝ), (α + β = 2 - 2 * k) ∧ (α^2 - 2 * (1 - k) * α + k^2 = 0) ∧ (β^2 - 2 * (1 - k) * β + k^2 = 0) ∧ (α + β ≥ 1) :=
sorry

end quadratic_root_sum_l1906_190622


namespace find_value_of_fraction_l1906_190681

open Real

theorem find_value_of_fraction (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : x / y + y / x = 8) : 
  (x + y) / (x - y) = -sqrt (5 / 3) :=
sorry

end find_value_of_fraction_l1906_190681


namespace car_truck_ratio_l1906_190613

theorem car_truck_ratio (total_vehicles trucks cars : ℕ)
  (h1 : total_vehicles = 300)
  (h2 : trucks = 100)
  (h3 : cars + trucks = total_vehicles)
  (h4 : ∃ (k : ℕ), cars = k * trucks) : 
  cars / trucks = 2 :=
by
  sorry

end car_truck_ratio_l1906_190613


namespace correct_calculation_l1906_190664

theorem correct_calculation (x a b : ℝ) : 
  (x^4 * x^4 = x^8) ∧ ((a^3)^2 = a^6) ∧ ((a * (b^2))^3 = a^3 * b^6) → (a + 2*a = 3*a) := 
by 
  sorry

end correct_calculation_l1906_190664


namespace route_length_l1906_190637

theorem route_length (D : ℝ) (T : ℝ) 
  (hx : T = 400 / D) 
  (hy : 80 = (D / 5) * T) 
  (hz : 80 + (D / 4) * T = D) : 
  D = 180 :=
by
  sorry

end route_length_l1906_190637


namespace transformer_minimum_load_l1906_190698

-- Define the conditions as hypotheses
def running_current_1 := 40
def running_current_2 := 60
def running_current_3 := 25

def start_multiplier_1 := 2
def start_multiplier_2 := 3
def start_multiplier_3 := 4

def units_1 := 3
def units_2 := 2
def units_3 := 1

def starting_current_1 := running_current_1 * start_multiplier_1
def starting_current_2 := running_current_2 * start_multiplier_2
def starting_current_3 := running_current_3 * start_multiplier_3

def total_starting_current_1 := starting_current_1 * units_1
def total_starting_current_2 := starting_current_2 * units_2
def total_starting_current_3 := starting_current_3 * units_3

def total_combined_minimum_current_load := 
  total_starting_current_1 + total_starting_current_2 + total_starting_current_3

-- The theorem to prove that the total combined minimum current load is 700A
theorem transformer_minimum_load : total_combined_minimum_current_load = 700 := by
  sorry

end transformer_minimum_load_l1906_190698


namespace find_f_of_1_div_8_l1906_190667

noncomputable def f (x : ℝ) (a : ℝ) := (a^2 + a - 5) * Real.logb a x

theorem find_f_of_1_div_8 (a : ℝ) (hx1 : x = 1 / 8) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a^2 + a - 5 = 1) :
  f x a = -3 :=
by
  sorry

end find_f_of_1_div_8_l1906_190667


namespace unique_real_root_t_l1906_190634

theorem unique_real_root_t (t : ℝ) :
  (∃ x : ℝ, 3 * x + 7 * t - 2 + (2 * t * x^2 + 7 * t^2 - 9) / (x - t) = 0 ∧ 
  ∀ y : ℝ, 3 * y + 7 * t - 2 + (2 * t * y^2 + 7 * t^2 - 9) / (y - t) = 0 ∧ x ≠ y → false) →
  t = -3 ∨ t = -7 / 2 ∨ t = 1 :=
by
  sorry

end unique_real_root_t_l1906_190634


namespace area_of_square_with_diagonal_30_l1906_190600

theorem area_of_square_with_diagonal_30 :
  ∀ (d : ℝ), d = 30 → (d * d / 2) = 450 := 
by
  intros d h
  rw [h]
  sorry

end area_of_square_with_diagonal_30_l1906_190600


namespace find_a_l1906_190644

-- Definitions and conditions from the problem
def M (a : ℝ) : Set ℝ := {1, 2, a^2 - 3*a - 1}
def N (a : ℝ) : Set ℝ := {-1, a, 3}
def intersection_is_three (a : ℝ) : Prop := M a ∩ N a = {3}

-- The theorem we want to prove
theorem find_a (a : ℝ) (h : intersection_is_three a) : a = 4 :=
by
  sorry

end find_a_l1906_190644


namespace sequence_properties_l1906_190638

theorem sequence_properties (a : ℕ → ℝ)
  (h1 : a 1 = 1 / 5)
  (h2 : ∀ n : ℕ, n > 1 → a (n - 1) / a n = (2 * a (n - 1) + 1) / (1 - 2 * a n)) :
  (∀ n : ℕ, n > 0 → (1 / a n) - (1 / a (n - 1)) = 4) ∧
  (∀ m k : ℕ, m > 0 ∧ k > 0 → a m * a k = a (4 * m * k + m + k)) :=
by
  sorry

end sequence_properties_l1906_190638


namespace ratio_6_3_to_percent_l1906_190621

theorem ratio_6_3_to_percent : (6 / 3) * 100 = 200 := by
  sorry

end ratio_6_3_to_percent_l1906_190621


namespace gcd_f_of_x_and_x_l1906_190632

theorem gcd_f_of_x_and_x (x : ℕ) (hx : 7200 ∣ x) :
  Nat.gcd ((5 * x + 6) * (8 * x + 3) * (11 * x + 9) * (4 * x + 12)) x = 72 :=
sorry

end gcd_f_of_x_and_x_l1906_190632


namespace combined_basketballs_l1906_190666

-- Conditions as definitions
def spursPlayers := 22
def rocketsPlayers := 18
def basketballsPerPlayer := 11

-- Math Proof Problem statement
theorem combined_basketballs : 
  (spursPlayers * basketballsPerPlayer) + (rocketsPlayers * basketballsPerPlayer) = 440 :=
by
  sorry

end combined_basketballs_l1906_190666


namespace votes_for_candidate_a_l1906_190602

theorem votes_for_candidate_a :
  let total_votes : ℝ := 560000
  let percentage_invalid : ℝ := 0.15
  let percentage_candidate_a : ℝ := 0.85
  let valid_votes := (1 - percentage_invalid) * total_votes
  let votes_candidate_a := percentage_candidate_a * valid_votes
  votes_candidate_a = 404600 :=
by
  sorry

end votes_for_candidate_a_l1906_190602


namespace SusanBooks_l1906_190668

-- Definitions based on the conditions of the problem
def Lidia (S : ℕ) : ℕ := 4 * S
def TotalBooks (S : ℕ) : ℕ := S + Lidia S

-- The proof statement
theorem SusanBooks (S : ℕ) (h : TotalBooks S = 3000) : S = 600 :=
by
  sorry

end SusanBooks_l1906_190668


namespace cara_younger_than_mom_l1906_190651

noncomputable def cara_grandmothers_age : ℤ := 75
noncomputable def cara_moms_age := cara_grandmothers_age - 15
noncomputable def cara_age : ℤ := 40

theorem cara_younger_than_mom :
  cara_moms_age - cara_age = 20 := by
  sorry

end cara_younger_than_mom_l1906_190651


namespace intersection_of_squares_perimeter_l1906_190652

noncomputable def perimeter_of_rectangle (side1 side2 : ℝ) : ℝ :=
2 * (side1 + side2)

theorem intersection_of_squares_perimeter
  (side_length : ℝ)
  (diagonal : ℝ)
  (distance_between_centers : ℝ)
  (h1 : 4 * side_length = 8) 
  (h2 : (side1^2 + side2^2) = diagonal^2)
  (h3 : (2 - side1)^2 + (2 - side2)^2 = distance_between_centers^2) : 
10 * (perimeter_of_rectangle side1 side2) = 25 :=
sorry

end intersection_of_squares_perimeter_l1906_190652


namespace factorize_expression_l1906_190683

variable {a b : ℕ}

theorem factorize_expression (a b : ℕ) : 9 * a - 6 * b = 3 * (3 * a - 2 * b) :=
by
  sorry

end factorize_expression_l1906_190683


namespace xy_value_l1906_190616

theorem xy_value (x y : ℝ) (h : x * (x + y) = x ^ 2 + 12) : x * y = 12 :=
by {
  sorry
}

end xy_value_l1906_190616


namespace mrs_choi_profit_percentage_l1906_190653

theorem mrs_choi_profit_percentage :
  ∀ (original_price selling_price : ℝ) (broker_percentage : ℝ),
    original_price = 80000 →
    selling_price = 100000 →
    broker_percentage = 0.05 →
    (selling_price - (broker_percentage * original_price) - original_price) / original_price * 100 = 20 :=
by
  intros original_price selling_price broker_percentage h1 h2 h3
  sorry

end mrs_choi_profit_percentage_l1906_190653


namespace tobys_friends_boys_count_l1906_190678

theorem tobys_friends_boys_count (total_friends : ℕ) (girls : ℕ) (boys_percentage : ℕ) 
    (h1 : girls = 27) (h2 : boys_percentage = 55) (total_friends_calc : total_friends = 60) : 
    (total_friends * boys_percentage / 100) = 33 :=
by
  -- Proof is deferred
  sorry

end tobys_friends_boys_count_l1906_190678


namespace empty_cistern_time_l1906_190642

variable (t_fill : ℝ) (t_empty₁ : ℝ) (t_empty₂ : ℝ) (t_empty₃ : ℝ)

theorem empty_cistern_time
  (h_fill : t_fill = 3.5)
  (h_empty₁ : t_empty₁ = 14)
  (h_empty₂ : t_empty₂ = 16)
  (h_empty₃ : t_empty₃ = 18) :
  1008 / (1/t_empty₁ + 1/t_empty₂ + 1/t_empty₃) = 1.31979 := by
  sorry

end empty_cistern_time_l1906_190642


namespace certain_number_division_l1906_190646

theorem certain_number_division (x : ℝ) (h : x / 3 + x + 3 = 63) : x = 45 :=
by
  sorry

end certain_number_division_l1906_190646


namespace real_part_of_z_is_neg3_l1906_190612

noncomputable def z : ℂ := (1 + 2 * Complex.I) ^ 2

theorem real_part_of_z_is_neg3 : z.re = -3 := by
  sorry

end real_part_of_z_is_neg3_l1906_190612


namespace card_draw_sequential_same_suit_l1906_190691

theorem card_draw_sequential_same_suit : 
  let hearts := 13
  let diamonds := 13
  let total_suits := hearts + diamonds
  ∃ ways : ℕ, ways = total_suits * (hearts - 1) :=
by
  sorry

end card_draw_sequential_same_suit_l1906_190691


namespace g_triple_composition_l1906_190679

def g (n : ℕ) : ℕ :=
if n < 5 then n^2 + 1 else 2 * n + 3

theorem g_triple_composition : g (g (g 3)) = 49 :=
by
  sorry

end g_triple_composition_l1906_190679


namespace observation_count_l1906_190623

theorem observation_count (mean_before mean_after : ℝ) 
  (wrong_value : ℝ) (correct_value : ℝ) (n : ℝ) :
  mean_before = 36 →
  correct_value = 60 →
  wrong_value = 23 →
  mean_after = 36.5 →
  n = 74 :=
by
  intros h_mean_before h_correct_value h_wrong_value h_mean_after
  sorry

end observation_count_l1906_190623


namespace no_perfect_squares_xy_zt_l1906_190660

theorem no_perfect_squares_xy_zt
    (x y z t : ℕ) 
    (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : 0 < t)
    (h_eq1 : x + y = z + t) 
    (h_eq2 : xy - zt = x + y) : ¬(∃ a b : ℕ, xy = a^2 ∧ zt = b^2) :=
by
  sorry

end no_perfect_squares_xy_zt_l1906_190660


namespace gardener_cabbages_increased_by_197_l1906_190609

theorem gardener_cabbages_increased_by_197 (x : ℕ) (last_year_cabbages : ℕ := x^2) (increase : ℕ := 197) :
  (x + 1)^2 = x^2 + increase → (x + 1)^2 = 9801 :=
by
  intros h
  sorry

end gardener_cabbages_increased_by_197_l1906_190609


namespace heather_start_time_later_than_stacy_l1906_190611

theorem heather_start_time_later_than_stacy :
  ∀ (distance_initial : ℝ) (H_speed : ℝ) (S_speed : ℝ) (H_distance_when_meet : ℝ),
    distance_initial = 5 ∧
    H_speed = 5 ∧
    S_speed = 6 ∧
    H_distance_when_meet = 1.1818181818181817 →
    ∃ (Δt : ℝ), Δt = 24 / 60 :=
by
  sorry

end heather_start_time_later_than_stacy_l1906_190611


namespace solve_cyclist_return_speed_l1906_190648

noncomputable def cyclist_return_speed (D : ℝ) (V : ℝ) : Prop :=
  let avg_speed := 9.5
  let out_speed := 10
  let T_out := D / out_speed
  let T_back := D / V
  2 * D / (T_out + T_back) = avg_speed

theorem solve_cyclist_return_speed : ∀ (D : ℝ), cyclist_return_speed D (20 / 2.1) :=
by
  intro D
  sorry

end solve_cyclist_return_speed_l1906_190648


namespace rectangle_y_value_l1906_190699

theorem rectangle_y_value
  (E : (ℝ × ℝ)) (F : (ℝ × ℝ)) (G : (ℝ × ℝ)) (H : (ℝ × ℝ))
  (hE : E = (0, 0)) (hF : F = (0, 5)) (hG : ∃ y : ℝ, G = (y, 5))
  (hH : ∃ y : ℝ, H = (y, 0)) (area : ℝ) (h_area : area = 35)
  (hy_pos : ∃ y : ℝ, y > 0)
  : ∃ y : ℝ, y = 7 :=
by
  sorry

end rectangle_y_value_l1906_190699


namespace terminal_side_of_610_deg_is_250_deg_l1906_190633

theorem terminal_side_of_610_deg_is_250_deg:
  ∃ k : ℤ, 610 % 360 = 250 := by
  sorry

end terminal_side_of_610_deg_is_250_deg_l1906_190633


namespace largest_number_l1906_190603

theorem largest_number (a b c : ℤ) 
  (h_sum : a + b + c = 67)
  (h_diff1 : c - b = 7)
  (h_diff2 : b - a = 3)
  : c = 28 :=
sorry

end largest_number_l1906_190603


namespace instantaneous_velocity_at_3_l1906_190614

noncomputable def displacement (t : ℝ) : ℝ := 
  - (1 / 3) * t^3 + 2 * t^2 - 5

theorem instantaneous_velocity_at_3 : 
  (deriv displacement 3 = 3) :=
by
  sorry

end instantaneous_velocity_at_3_l1906_190614


namespace find_polynomials_g_l1906_190631

-- Assume f(x) = x^2
def f (x : ℝ) : ℝ := x ^ 2

-- Define the condition that f(g(x)) = 9x^2 - 6x + 1
def condition (g : ℝ → ℝ) : Prop := ∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1

-- Prove that the possible polynomials for g(x) are 3x - 1 or -3x + 1
theorem find_polynomials_g (g : ℝ → ℝ) (h : condition g) :
  (∀ x, g x = 3 * x - 1) ∨ (∀ x, g x = -3 * x + 1) :=
sorry

end find_polynomials_g_l1906_190631


namespace sufficient_not_necessary_condition_l1906_190604

variable (a : ℝ)

theorem sufficient_not_necessary_condition (h1 : a > 2) : (1 / a < 1 / 2) ↔ (a > 2 ∨ a < 0) :=
by
  sorry

end sufficient_not_necessary_condition_l1906_190604


namespace parabola_vertex_l1906_190659

theorem parabola_vertex (c d : ℝ) (h : ∀ (x : ℝ), (-x^2 + c * x + d ≤ 0) ↔ (x ≤ -5 ∨ x ≥ 3)) :
  (∃ a b : ℝ, a = 4 ∧ b = 1 ∧ (-x^2 + c * x + d = -x^2 + 8 * x - 15)) :=
by
  sorry

end parabola_vertex_l1906_190659


namespace friend_decks_l1906_190657

-- Definitions for conditions
def price_per_deck : ℕ := 8
def victor_decks : ℕ := 6
def total_spent : ℕ := 64

-- Conclusion based on the conditions
theorem friend_decks : (64 - (6 * 8)) / 8 = 2 := by
  sorry

end friend_decks_l1906_190657


namespace equivalent_after_eliminating_denominators_l1906_190606

theorem equivalent_after_eliminating_denominators (x : ℝ) (h : 1 + 2 / (x - 1) = (x - 5) / (x - 3)) :
  (x - 1) * (x - 3) + 2 * (x - 3) = (x - 5) * (x - 1) :=
sorry

end equivalent_after_eliminating_denominators_l1906_190606


namespace sara_lunch_total_cost_l1906_190675

noncomputable def cost_hotdog : ℝ := 5.36
noncomputable def cost_salad : ℝ := 5.10
noncomputable def cost_soda : ℝ := 2.75
noncomputable def cost_fries : ℝ := 3.20
noncomputable def discount_rate : ℝ := 0.15
noncomputable def tax_rate : ℝ := 0.08

noncomputable def total_cost_before_discount_tax : ℝ :=
  cost_hotdog + cost_salad + cost_soda + cost_fries

noncomputable def discount : ℝ :=
  discount_rate * total_cost_before_discount_tax

noncomputable def discounted_total : ℝ :=
  total_cost_before_discount_tax - discount

noncomputable def tax : ℝ := 
  tax_rate * discounted_total

noncomputable def final_total : ℝ :=
  discounted_total + tax

theorem sara_lunch_total_cost : final_total = 15.07 :=
by
  sorry

end sara_lunch_total_cost_l1906_190675


namespace total_students_l1906_190601

theorem total_students (initial_candies leftover_candies girls boys : ℕ) (h1 : initial_candies = 484)
  (h2 : leftover_candies = 4) (h3 : boys = girls + 3) (h4 : (2 * girls + boys) * (2 * girls + boys) = initial_candies - leftover_candies) :
  2 * girls + boys = 43 :=
  sorry

end total_students_l1906_190601


namespace evaluate_f_diff_l1906_190669

def f (x : ℝ) := x^5 + 2*x^3 + 7*x

theorem evaluate_f_diff : f 3 - f (-3) = 636 := by
  sorry

end evaluate_f_diff_l1906_190669


namespace gray_region_correct_b_l1906_190639

-- Define the basic conditions
def square_side_length : ℝ := 3
def small_square_side_length : ℝ := 1

-- Define the triangles resulting from cutting a square
def triangle_area : ℝ := 0.5 * square_side_length * square_side_length

-- Define the gray region area for the second figure (b)
def gray_region_area_b : ℝ := 0.25

-- Lean statement to prove the area of the gray region
theorem gray_region_correct_b : gray_region_area_b = 0.25 := by
  -- Proof is omitted
  sorry

end gray_region_correct_b_l1906_190639


namespace vasya_filling_time_l1906_190696

-- Definition of conditions
def hose_filling_time (x : ℝ) : Prop :=
  ∀ (first_hose_mult second_hose_mult : ℝ), 
    first_hose_mult = x ∧
    second_hose_mult = 5 * x ∧
    (5 * second_hose_mult - 5 * first_hose_mult) = 1

-- Conclusion
theorem vasya_filling_time (x : ℝ) (first_hose_mult second_hose_mult : ℝ) :
  hose_filling_time x → 25 * x = 1 * (60 + 15) := sorry

end vasya_filling_time_l1906_190696


namespace real_solutions_system_l1906_190647

theorem real_solutions_system (x y z : ℝ) : 
  (x = 4 * z^2 / (1 + 4 * z^2) ∧ y = 4 * x^2 / (1 + 4 * x^2) ∧ z = 4 * y^2 / (1 + 4 * y^2)) ↔ 
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end real_solutions_system_l1906_190647


namespace cost_price_of_radio_l1906_190697

theorem cost_price_of_radio (C : ℝ) 
  (overhead_expenses : ℝ := 20) 
  (selling_price : ℝ := 300) 
  (profit_percent : ℝ := 22.448979591836732) :
  C = 228.57 :=
by
  sorry

end cost_price_of_radio_l1906_190697


namespace vector_n_value_l1906_190688

theorem vector_n_value {n : ℤ} (hAB : (2, 4) = (2, 4)) (hBC : (-2, n) = (-2, n)) (hAC : (0, 2) = (2 + -2, 4 + n)) : n = -2 :=
by
  sorry

end vector_n_value_l1906_190688


namespace pat_kate_mark_ratio_l1906_190687

variables (P K M r : ℚ) 

theorem pat_kate_mark_ratio (h1 : P + K + M = 189) 
                            (h2 : P = r * K) 
                            (h3 : P = (1 / 3) * M) 
                            (h4 : M = K + 105) :
  r = 4 / 3 :=
sorry

end pat_kate_mark_ratio_l1906_190687


namespace average_after_12th_inning_revised_average_not_out_l1906_190655

theorem average_after_12th_inning (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) : (A + 2) = 70 :=
by
  -- Calculation steps are skipped
  sorry

theorem revised_average_not_out (A : ℝ) (H_innings : 11 * A + 92 = 12 * (A + 2)) (H_not_out : 11 * A + 92 = 840) :
  (11 * A + 92) / 9 = 93.33 :=
by
  -- Calculation steps are skipped
  sorry

end average_after_12th_inning_revised_average_not_out_l1906_190655


namespace mean_of_combined_set_l1906_190608

theorem mean_of_combined_set
  (mean1 : ℕ → ℝ)
  (n1 : ℕ)
  (mean2 : ℕ → ℝ)
  (n2 : ℕ)
  (h1 : ∀ n1, mean1 n1 = 15)
  (h2 : ∀ n2, mean2 n2 = 26) :
  (n1 + n2) = 15 → 
  ((n1 * 15 + n2 * 26) / (n1 + n2)) = (313/15) :=
by
  sorry

end mean_of_combined_set_l1906_190608


namespace Jeffs_donuts_l1906_190629

theorem Jeffs_donuts (D : ℕ) (h1 : ∀ n, n = 12 * D - 20) (h2 : n = 100) : D = 10 :=
by
  sorry

end Jeffs_donuts_l1906_190629


namespace geometric_sequence_ratio_l1906_190628

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end geometric_sequence_ratio_l1906_190628


namespace apples_rate_per_kg_l1906_190656

variable (A : ℝ)

theorem apples_rate_per_kg (h : 8 * A + 9 * 65 = 1145) : A = 70 :=
sorry

end apples_rate_per_kg_l1906_190656


namespace equilateral_triangle_perimeter_l1906_190680

-- Definitions based on conditions
def equilateral_triangle_side : ℕ := 8

-- The statement we need to prove
theorem equilateral_triangle_perimeter : 3 * equilateral_triangle_side = 24 := by
  sorry

end equilateral_triangle_perimeter_l1906_190680


namespace problem_inequality_l1906_190617

theorem problem_inequality 
  (a b c d : ℝ)
  (h1 : d > 0)
  (h2 : a ≥ b)
  (h3 : b ≥ c)
  (h4 : c ≥ d)
  (h5 : a * b * c * d = 1) : 
  (1 / (1 + a)) + (1 / (1 + b)) + (1 / (1 + c)) ≥ 3 / (1 + (a * b * c) ^ (1 / 3)) :=
sorry

end problem_inequality_l1906_190617


namespace tan_double_angle_difference_l1906_190624

variable {α β : Real}

theorem tan_double_angle_difference (h1 : Real.tan α = 1 / 2) (h2 : Real.tan (α - β) = 1 / 5) :
  Real.tan (2 * α - β) = 7 / 9 := 
sorry

end tan_double_angle_difference_l1906_190624


namespace megatek_manufacturing_percentage_l1906_190671

theorem megatek_manufacturing_percentage (total_degrees manufacturing_degrees : ℝ)
    (h_proportional : total_degrees = 360)
    (h_manufacturing_degrees : manufacturing_degrees = 180) :
    (manufacturing_degrees / total_degrees) * 100 = 50 := by
  -- The proof will go here.
  sorry

end megatek_manufacturing_percentage_l1906_190671


namespace binomial_product_result_l1906_190693

-- Defining the combination (binomial coefficient) formula
def combination (n k : Nat) : Nat := n.factorial / (k.factorial * (n - k).factorial)

-- Lean theorem statement to prove the problem
theorem binomial_product_result : combination 10 3 * combination 8 3 = 6720 := by
  sorry

end binomial_product_result_l1906_190693


namespace intersection_eq_l1906_190627

def M : Set ℝ := { x | -2 ≤ x ∧ x ≤ 0 }
def N : Set ℝ := { -1, 0, 1 }

theorem intersection_eq : M ∩ N = { -1, 0 } := by
  sorry

end intersection_eq_l1906_190627


namespace centroid_plane_distance_l1906_190694

theorem centroid_plane_distance :
  ∀ (α β γ : ℝ) (p q r : ℝ),
    (1 / α^2 + 1 / β^2 + 1 / γ^2 = 1 / 4) →
    (p = α / 3) →
    (q = β / 3) →
    (r = γ / 3) →
    (1 / p^2 + 1 / q^2 + 1 / r^2 = 9 / 4) :=
by sorry

end centroid_plane_distance_l1906_190694


namespace common_difference_arithmetic_sequence_l1906_190645

theorem common_difference_arithmetic_sequence 
    (a : ℕ → ℝ) 
    (S₅ : ℝ)
    (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h₁ : a 4 + a 6 = 6)
    (h₂ : S₅ = (a 1 + a 2 + a 3 + a 4 + a 5))
    (h_S₅_val : S₅ = 10) :
  ∃ d : ℝ, d = (a 5 - a 1) / 4 ∧ d = 1/2 := 
by
  sorry

end common_difference_arithmetic_sequence_l1906_190645


namespace sally_initial_peaches_l1906_190685

section
variables 
  (peaches_after : ℕ)
  (peaches_picked : ℕ)
  (initial_peaches : ℕ)

theorem sally_initial_peaches 
    (h1 : peaches_picked = 42)
    (h2 : peaches_after = 55)
    (h3 : peaches_after = initial_peaches + peaches_picked) : 
    initial_peaches = 13 := 
by 
  sorry
end

end sally_initial_peaches_l1906_190685


namespace percentage_volume_taken_by_cubes_l1906_190686

def volume_of_box (l w h : ℝ) : ℝ := l * w * h

def volume_of_cube (side : ℝ) : ℝ := side ^ 3

noncomputable def total_cubes_fit (l w h side : ℝ) : ℝ := 
  (l / side) * (w / side) * (h / side)

theorem percentage_volume_taken_by_cubes (l w h side : ℝ) (hl : l = 12) (hw : w = 6) (hh : h = 9) (hside : side = 3) :
  volume_of_box l w h ≠ 0 → 
  (total_cubes_fit l w h side * volume_of_cube side / volume_of_box l w h) * 100 = 100 :=
by
  intros
  rw [hl, hw, hh, hside]
  simp only [volume_of_box, volume_of_cube, total_cubes_fit]
  sorry

end percentage_volume_taken_by_cubes_l1906_190686


namespace derivative_at_one_is_three_l1906_190672

-- Definition of the function
def f (x : ℝ) := (x - 1)^2 + 3 * (x - 1)

-- The statement of the problem
theorem derivative_at_one_is_three : deriv f 1 = 3 := 
  sorry

end derivative_at_one_is_three_l1906_190672


namespace evaluate_expression_l1906_190641

theorem evaluate_expression : 1 - (-2) * 2 - 3 - (-4) * 2 - 5 - (-6) * 2 = 17 := 
by
  sorry

end evaluate_expression_l1906_190641


namespace forty_percent_of_thirty_percent_l1906_190677

theorem forty_percent_of_thirty_percent (x : ℝ) 
  (h : 0.3 * 0.4 * x = 48) : 0.4 * 0.3 * x = 48 :=
by
  sorry

end forty_percent_of_thirty_percent_l1906_190677


namespace who_is_next_to_Boris_l1906_190610

noncomputable def arrangement := ℕ → ℕ

-- Definitions for positions
def position (n : ℕ) := n % 6 + 1 -- Modulo to ensure circular arrangement

-- Names mapped to numbers for simplicity
def Arkady := 1
def Boris := 2
def Vera := 3
def Galya := 4
def Danya := 5
def Egor := 6

-- Conditions:
-- 1. Danya stands next to Vera, on her right side.
def cond1 (a : arrangement) := ∃ n, a n = Vera ∧ a (position (n + 1)) = Danya

-- 2. Galya stands opposite Egor.
def cond2 (a : arrangement) := ∃ n, a n = Egor ∧ a (position (n + 3)) = Galya

-- 3. Egor stands next to Danya.
def cond3 (a : arrangement) := ∃ n, a n = Danya ∧ (a (position (n - 1)) = Egor ∨ a (position (n + 1)) = Egor)

-- 4. Arkady and Galya do not want to stand next to each other.
def cond4 (a : arrangement) := ∀ n, ¬(a n = Arkady ∧ (a (position (n - 1)) = Galya ∨ a (position (n + 1)) = Galya))

-- Conclusion: Arkady and Galya are standing next to Boris.
theorem who_is_next_to_Boris (a : arrangement) :
  cond1 a ∧ cond2 a ∧ cond3 a ∧ cond4 a → 
  (∃ n, a n = Boris ∧ ((a (position (n - 1)) = Arkady ∧ a (position (n + 1)) = Galya) ∨ (a (position (n + 1)) = Arkady ∧ a (position (n - 1)) = Galya))) :=
sorry

end who_is_next_to_Boris_l1906_190610


namespace precious_stones_l1906_190605

variable (total_amount : ℕ) (price_per_stone : ℕ) (number_of_stones : ℕ)

theorem precious_stones (h1 : total_amount = 14280) (h2 : price_per_stone = 1785) : number_of_stones = 8 :=
by
  sorry

end precious_stones_l1906_190605


namespace right_triangle_area_l1906_190654

theorem right_triangle_area (a b c: ℝ) (h1: c = 2) (h2: a + b + c = 2 + Real.sqrt 6) (h3: (a * b) / 2 = 1 / 2) :
  (1 / 2) * (a * b) = 1 / 2 :=
by
  -- Sorry is used to skip the proof
  sorry

end right_triangle_area_l1906_190654


namespace sqrt_subtraction_l1906_190661

theorem sqrt_subtraction : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end sqrt_subtraction_l1906_190661


namespace remainder_of_19_pow_60_mod_7_l1906_190658

theorem remainder_of_19_pow_60_mod_7 : (19 ^ 60) % 7 = 1 := 
by {
  sorry
}

end remainder_of_19_pow_60_mod_7_l1906_190658


namespace prime_ge_7_div_30_l1906_190663

theorem prime_ge_7_div_30 (p : ℕ) (hp : Nat.Prime p) (h7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end prime_ge_7_div_30_l1906_190663


namespace original_couch_price_l1906_190619

def chair_price : ℝ := sorry
def table_price := 3 * chair_price
def couch_price := 5 * table_price
def bookshelf_price := 0.5 * couch_price

def discounted_chair_price := 0.8 * chair_price
def discounted_couch_price := 0.9 * couch_price
def total_price_before_tax := discounted_chair_price + table_price + discounted_couch_price + bookshelf_price
def total_price_after_tax := total_price_before_tax * 1.08

theorem original_couch_price (budget : ℝ) (h_budget : budget = 900) : 
  total_price_after_tax = budget → couch_price = 503.85 :=
by
  sorry

end original_couch_price_l1906_190619


namespace cheese_cut_indefinite_l1906_190643

theorem cheese_cut_indefinite (w : ℝ) (R : ℝ) (h : ℝ) :
  R = 0.5 →
  (∀ a b c d : ℝ, a > b → b > c → c > d →
    (∃ h, h < min (a - d) (d - c) ∧
     (d + h < a ∧ d - h > c))) →
  ∃ l1 l2 : ℕ → ℝ, (∀ n, l1 (n + 1) > l2 (n) ∧ l1 n > R * l2 (n)) :=
sorry

end cheese_cut_indefinite_l1906_190643


namespace profit_share_difference_l1906_190695

theorem profit_share_difference
    (P_A P_B P_C P_D : ℕ) (R_A R_B R_C R_D parts_A parts_B parts_C parts_D : ℕ) (profit_B : ℕ)
    (h1 : P_A = 8000) (h2 : P_B = 10000) (h3 : P_C = 12000) (h4 : P_D = 15000)
    (h5 : R_A = 3) (h6 : R_B = 5) (h7 : R_C = 6) (h8 : R_D = 7)
    (h9: profit_B = 2000) :
    profit_B / R_B = 400 ∧ P_C * R_C / R_B - P_A * R_A / R_B = 1200 :=
by
  sorry

end profit_share_difference_l1906_190695


namespace equivalent_conditions_l1906_190692

open Real

theorem equivalent_conditions (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (1 / x + 1 / y + 1 / z ≤ 1) ↔
  (∀ a b c d : ℝ, a + b + c > d → a^2 * x + b^2 * y + c^2 * z > d^2) :=
by
  sorry

end equivalent_conditions_l1906_190692


namespace line_intersects_parabola_at_one_point_l1906_190690
   
   theorem line_intersects_parabola_at_one_point (k : ℝ) :
     (∃ y : ℝ, (x = 3 * y^2 - 7 * y + 2 ∧ x = k) → x = k) ↔ k = (-25 / 12) :=
   by
     -- your proof goes here
     sorry
   
end line_intersects_parabola_at_one_point_l1906_190690


namespace arccos_gt_arctan_l1906_190625

theorem arccos_gt_arctan (x : ℝ) (h : -1 ≤ x ∧ x < 1/2) : Real.arccos x > Real.arctan x :=
sorry

end arccos_gt_arctan_l1906_190625


namespace sequence_ratio_l1906_190689

variable (a : ℕ → ℝ) -- Define the sequence a_n
variable (q : ℝ) (h_q : q > 0) -- q is the common ratio and it is positive

-- Define the conditions
axiom geom_seq_pos : ∀ n : ℕ, 0 < a n
axiom geom_seq_def : ∀ n : ℕ, a (n + 1) = q * a n
axiom arith_seq_def : 2 * (1 / 2 * a 3) = 3 * a 1 + 2 * a 2

theorem sequence_ratio : (a 11 + a 13) / (a 8 + a 10) = 27 := 
by
  sorry

end sequence_ratio_l1906_190689


namespace tribe_leadership_choices_l1906_190626

theorem tribe_leadership_choices :
  let members := 15
  let ways_to_choose_chief := members
  let remaining_after_chief := members - 1
  let ways_to_choose_supporting_chiefs := Nat.choose remaining_after_chief 2
  let remaining_after_supporting_chiefs := remaining_after_chief - 2
  let ways_to_choose_officers_A := Nat.choose remaining_after_supporting_chiefs 2
  let remaining_for_assistants_A := remaining_after_supporting_chiefs - 2
  let ways_to_choose_assistants_A := Nat.choose remaining_for_assistants_A 2 * Nat.choose (remaining_for_assistants_A - 2) 2
  let remaining_after_A := remaining_for_assistants_A - 2
  let ways_to_choose_officers_B := Nat.choose remaining_after_A 2
  let remaining_for_assistants_B := remaining_after_A - 2
  let ways_to_choose_assistants_B := Nat.choose remaining_for_assistants_B 2 * Nat.choose (remaining_for_assistants_B - 2) 2
  (ways_to_choose_chief * ways_to_choose_supporting_chiefs *
  ways_to_choose_officers_A * ways_to_choose_assistants_A *
  ways_to_choose_officers_B * ways_to_choose_assistants_B = 400762320000) := by
  sorry

end tribe_leadership_choices_l1906_190626


namespace greatest_whole_number_satisfying_inequality_l1906_190650

theorem greatest_whole_number_satisfying_inequality :
  ∀ (x : ℤ), 3 * x + 2 < 5 - 2 * x → x <= 0 :=
by
  sorry

end greatest_whole_number_satisfying_inequality_l1906_190650


namespace math_problem_l1906_190682

theorem math_problem (a b n r : ℕ) (h₁ : 1853 ≡ 53 [MOD 600]) (h₂ : 2101 ≡ 101 [MOD 600]) :
  (1853 * 2101) ≡ 553 [MOD 600] := by
  sorry

end math_problem_l1906_190682


namespace abs_e_pi_minus_six_l1906_190670

noncomputable def e : ℝ := 2.718
noncomputable def pi : ℝ := 3.14159

theorem abs_e_pi_minus_six : |e + pi - 6| = 0.14041 := by
  sorry

end abs_e_pi_minus_six_l1906_190670


namespace relationship_abc_l1906_190676

noncomputable def a : ℝ := (0.7 : ℝ) ^ (0.6 : ℝ)
noncomputable def b : ℝ := (0.6 : ℝ) ^ (-0.6 : ℝ)
noncomputable def c : ℝ := (0.6 : ℝ) ^ (0.7 : ℝ)

theorem relationship_abc : b > a ∧ a > c :=
by
  -- Proof will go here
  sorry

end relationship_abc_l1906_190676


namespace voice_of_china_signup_ways_l1906_190618

theorem voice_of_china_signup_ways : 
  (2 * 2 * 2 = 8) :=
by {
  sorry
}

end voice_of_china_signup_ways_l1906_190618


namespace total_number_of_seats_l1906_190684

theorem total_number_of_seats (n : ℕ) (h1 : ∀ i, 1 ≤ i ∧ i ≤ n) 
                               (h2 : (10 : ℕ) < (29 : ℕ)) 
                               (h3 : (29 - 10) % (n / 2) = 0) : n = 38 :=
by sorry

end total_number_of_seats_l1906_190684


namespace sequence_formula_l1906_190630

theorem sequence_formula (a : ℕ → ℤ) (S : ℕ → ℤ)
  (h1 : a 1 = 1)
  (h2 : a 2 = 4)
  (h3 : ∀ n : ℕ, n > 0 → a (n + 2) + 2 * a n = 3 * a (n + 1)) :
  (∀ n, a n = 3 * 2^(n-1) - 2) ∧ (S 4 > 21 - 2 * 4) :=
by
  sorry

end sequence_formula_l1906_190630


namespace fraction_pizza_covered_by_pepperoni_l1906_190673

/--
Given that six pepperoni circles fit exactly across the diameter of a 12-inch pizza
and a total of 24 circles of pepperoni are placed on the pizza without overlap,
prove that the fraction of the pizza covered by pepperoni is 2/3.
-/
theorem fraction_pizza_covered_by_pepperoni : 
  (∃ d r : ℝ, 6 * r = d ∧ d = 12 ∧ (r * r * π * 24) / (6 * 6 * π) = 2 / 3) := 
sorry

end fraction_pizza_covered_by_pepperoni_l1906_190673


namespace find_a_l1906_190636

theorem find_a (a b : ℝ) (h1 : 0 < a ∧ 0 < b) (h2 : a^b = b^a) (h3 : b = 4 * a) : 
  a = (4 : ℝ)^(1 / 3) :=
by
  sorry

end find_a_l1906_190636


namespace spherical_to_rectangular_correct_l1906_190662

noncomputable def sphericalToRectangular (ρ θ φ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin φ * Real.cos θ, ρ * Real.sin φ * Real.sin θ, ρ * Real.cos φ)

theorem spherical_to_rectangular_correct :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  sphericalToRectangular ρ θ φ = (5 * Real.sqrt 6 / 4, 5 * Real.sqrt 6 / 4, 5 / 2) :=
by
  sorry

end spherical_to_rectangular_correct_l1906_190662


namespace complement_of_intersection_l1906_190615

-- Definitions of the sets M and N
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | x < 3 }

-- Definition of the intersection of M and N
def M_inter_N : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

-- Definition of the complement of M ∩ N in ℝ
def complement_M_inter_N : Set ℝ := { x | x < 2 ∨ x ≥ 3 }

-- The theorem to be proved
theorem complement_of_intersection :
  (M_inter_Nᶜ) = complement_M_inter_N :=
by sorry

end complement_of_intersection_l1906_190615


namespace Mahesh_completes_in_60_days_l1906_190649

noncomputable def MaheshWork (W : ℝ) : ℝ :=
    W / 60

variables (W : ℝ)
variables (M R : ℝ)
variables (daysMahesh daysRajesh daysFullRajesh : ℝ)

theorem Mahesh_completes_in_60_days
  (h1 : daysMahesh = 20)
  (h2 : daysRajesh = 30)
  (h3 : daysFullRajesh = 45)
  (hR : R = W / daysFullRajesh)
  (hM : M = (W - R * daysRajesh) / daysMahesh) :
  W / M = 60 :=
by
  sorry

end Mahesh_completes_in_60_days_l1906_190649


namespace no_integer_pairs_satisfy_equation_l1906_190635

theorem no_integer_pairs_satisfy_equation :
  ∀ (m n : ℤ), ¬(m^3 + 10 * m^2 + 11 * m + 2 = 81 * n^3 + 27 * n^2 + 3 * n - 8) :=
by
  sorry

end no_integer_pairs_satisfy_equation_l1906_190635
