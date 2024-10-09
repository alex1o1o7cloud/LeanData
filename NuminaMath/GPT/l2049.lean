import Mathlib

namespace range_of_m_solve_inequality_l2049_204962

open Real Set

noncomputable def f (x: ℝ) := -abs (x - 2)
noncomputable def g (x: ℝ) (m: ℝ) := -abs (x - 3) + m

-- Problem 1: Prove the range of m given the condition
theorem range_of_m (h : ∀ x : ℝ, f x > g x m) : m < 1 :=
  sorry

-- Problem 2: Prove the set of solutions for f(x) + a - 1 > 0
theorem solve_inequality (a : ℝ) :
  (if a = 1 then {x : ℝ | x ≠ 2}
   else if a > 1 then univ
   else {x : ℝ | x < 1 + a} ∪ {x : ℝ | x > 3 - a}) = {x : ℝ | f x + a - 1 > 0} :=
  sorry

end range_of_m_solve_inequality_l2049_204962


namespace functional_expression_y_l2049_204995

theorem functional_expression_y (x y : ℝ) (k : ℝ) 
  (h1 : ∀ x, y + 2 = k * x) 
  (h2 : y = 7) 
  (h3 : x = 3) : 
  y = 3 * x - 2 := 
by 
  sorry

end functional_expression_y_l2049_204995


namespace beetle_total_distance_l2049_204989

theorem beetle_total_distance 
  (r_outer : ℝ) (r_middle : ℝ) (r_inner : ℝ)
  (r_outer_eq : r_outer = 25)
  (r_middle_eq : r_middle = 15)
  (r_inner_eq : r_inner = 5)
  : (1/3 * 2 * Real.pi * r_middle + (r_outer - r_middle) + 1/2 * 2 * Real.pi * r_inner + 2 * r_outer + (r_middle - r_inner)) = (15 * Real.pi + 70) :=
by
  rw [r_outer_eq, r_middle_eq, r_inner_eq]
  have := Real.pi
  sorry

end beetle_total_distance_l2049_204989


namespace estimate_red_balls_l2049_204977

theorem estimate_red_balls (x : ℕ) (drawn_black_balls : ℕ) (total_draws : ℕ) (black_balls : ℕ) 
  (h1 : black_balls = 4) 
  (h2 : total_draws = 100) 
  (h3 : drawn_black_balls = 40) 
  (h4 : (black_balls : ℚ) / (black_balls + x) = drawn_black_balls / total_draws) : 
  x = 6 := 
sorry

end estimate_red_balls_l2049_204977


namespace pyramid_edge_length_correct_l2049_204939

-- Definitions for the conditions
def total_length (sum_of_edges : ℝ) := sum_of_edges = 14.8
def edges_count (num_of_edges : ℕ) := num_of_edges = 8

-- Definition for the question and corresponding answer to prove
def length_of_one_edge (sum_of_edges : ℝ) (num_of_edges : ℕ) (one_edge_length : ℝ) :=
  sum_of_edges / num_of_edges = one_edge_length

-- The statement that needs to be proven
theorem pyramid_edge_length_correct : total_length 14.8 → edges_count 8 → length_of_one_edge 14.8 8 1.85 :=
by
  intros h1 h2
  sorry

end pyramid_edge_length_correct_l2049_204939


namespace sum_of_values_l2049_204944

def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_values (z₁ z₂ : ℝ) (h₁ : f (3 * z₁) = 10) (h₂ : f (3 * z₂) = 10) :
  z₁ + z₂ = - (2 / 9) :=
by
  sorry

end sum_of_values_l2049_204944


namespace odd_integer_divisibility_l2049_204901

theorem odd_integer_divisibility (n : ℕ) (hodd : n % 2 = 1) (hpos : n > 0) : ∃ k : ℕ, n^4 - n^2 - n = n * k := 
sorry

end odd_integer_divisibility_l2049_204901


namespace sine_double_angle_l2049_204942

theorem sine_double_angle (theta : ℝ) (h : Real.tan (theta + Real.pi / 4) = 2) : Real.sin (2 * theta) = 3 / 5 :=
sorry

end sine_double_angle_l2049_204942


namespace fraction_proof_l2049_204958

theorem fraction_proof (w x y z : ℚ) 
  (h1 : w / x = 1 / 3)
  (h2 : w / y = 3 / 4)
  (h3 : x / z = 2 / 5) : 
  (x + y) / (y + z) = 26 / 53 := 
by
  sorry

end fraction_proof_l2049_204958


namespace quadratic_root_shift_c_value_l2049_204911

theorem quadratic_root_shift_c_value
  (r s : ℝ)
  (h1 : r + s = 2)
  (h2 : r * s = -5) :
  ∃ b : ℝ, x^2 + b * x - 2 = 0 :=
by
  sorry

end quadratic_root_shift_c_value_l2049_204911


namespace sector_arc_length_l2049_204949

noncomputable def arc_length (R : ℝ) (θ : ℝ) : ℝ :=
  θ / 180 * Real.pi * R

theorem sector_arc_length
  (central_angle : ℝ) (area : ℝ) (arc_length_answer : ℝ)
  (h1 : central_angle = 120)
  (h2 : area = 300 * Real.pi) :
  arc_length_answer = 20 * Real.pi :=
by
  sorry

end sector_arc_length_l2049_204949


namespace woman_born_1892_l2049_204979

theorem woman_born_1892 (y : ℕ) (hy : 1850 ≤ y^2 - y ∧ y^2 - y < 1900) : y = 44 :=
by
  sorry

end woman_born_1892_l2049_204979


namespace total_animals_on_farm_l2049_204965

theorem total_animals_on_farm :
  let coop1 := 60
  let coop2 := 45
  let coop3 := 55
  let coop4 := 40
  let coop5 := 35
  let coop6 := 20
  let coop7 := 50
  let coop8 := 10
  let coop9 := 10
  let first_shed := 2 * 10
  let second_shed := 10
  let third_shed := 6
  let section1 := 15
  let section2 := 25
  let section3 := 2 * 15
  coop1 + coop2 + coop3 + coop4 + coop5 + coop6 + coop7 + coop8 + coop9 + first_shed + second_shed + third_shed + section1 + section2 + section3 = 431 :=
by
  sorry

end total_animals_on_farm_l2049_204965


namespace total_savings_correct_l2049_204972

-- Define the savings of Sam, Victory and Alex according to the given conditions
def sam_savings : ℕ := 1200
def victory_savings : ℕ := sam_savings - 200
def alex_savings : ℕ := 2 * victory_savings

-- Define the total savings
def total_savings : ℕ := sam_savings + victory_savings + alex_savings

-- The theorem to prove the total savings
theorem total_savings_correct : total_savings = 4200 :=
by
  sorry

end total_savings_correct_l2049_204972


namespace find_particular_number_l2049_204936

theorem find_particular_number (A B : ℤ) (x : ℤ) (hA : A = 14) (hB : B = 24)
  (h : (((A + x) * A - B) / B = 13)) : x = 10 :=
by {
  -- You can add an appropriate lemma or proof here if necessary
  sorry
}

end find_particular_number_l2049_204936


namespace hexagon_arrangements_eq_144_l2049_204996

def is_valid_arrangement (arr : (Fin 7 → ℕ)) : Prop :=
  ∀ (i j k : Fin 7),
    (i.val + j.val + k.val = 18) → -- 18 being a derived constant factor (since 3x = 28 + 2G where G ∈ {1, 4, 7} and hence x = 30,34,38/3 respectively make it divisible by 3 sum is 18 always)
    arr i + arr j + arr k = arr ⟨3, sorry⟩ -- arr[3] is the position of G

noncomputable def count_valid_arrangements : ℕ :=
  sorry -- Calculation of 3*48 goes here and respective pairing and permutations.

theorem hexagon_arrangements_eq_144 :
  count_valid_arrangements = 144 :=
sorry

end hexagon_arrangements_eq_144_l2049_204996


namespace least_non_lucky_multiple_of_10_l2049_204964

def is_lucky (n : ℕ) : Prop :=
  n % (n.digits 10).sum = 0

theorem least_non_lucky_multiple_of_10 : 
  ∃ n : ℕ, n % 10 = 0 ∧ ¬is_lucky n ∧ (∀ m : ℕ, m % 10 = 0 ∧ ¬is_lucky m → m ≥ n) ∧ n = 110 :=
by
  sorry

end least_non_lucky_multiple_of_10_l2049_204964


namespace photos_per_album_l2049_204982

theorem photos_per_album
  (n : ℕ) -- number of pages in each album
  (x y : ℕ) -- album numbers
  (h1 : 4 * n * (x - 1) + 17 ≤ 81 ∧ 81 ≤ 4 * n * (x - 1) + 20)
  (h2 : 4 * n * (y - 1) + 9 ≤ 171 ∧ 171 ≤ 4 * n * (y - 1) + 12) :
  4 * n = 32 :=
by 
  sorry

end photos_per_album_l2049_204982


namespace david_overall_average_l2049_204993

open Real

noncomputable def english_weighted_average := (74 * 0.20) + (80 * 0.25) + (77 * 0.55)
noncomputable def english_modified := english_weighted_average * 1.5

noncomputable def math_weighted_average := (65 * 0.15) + (75 * 0.25) + (90 * 0.60)
noncomputable def math_modified := math_weighted_average * 2.0

noncomputable def physics_weighted_average := (82 * 0.40) + (85 * 0.60)
noncomputable def physics_modified := physics_weighted_average * 1.2

noncomputable def chemistry_weighted_average := (67 * 0.35) + (89 * 0.65)
noncomputable def chemistry_modified := chemistry_weighted_average * 1.0

noncomputable def biology_weighted_average := (90 * 0.30) + (95 * 0.70)
noncomputable def biology_modified := biology_weighted_average * 1.5

noncomputable def overall_average := (english_modified + math_modified + physics_modified + chemistry_modified + biology_modified) / 5

theorem david_overall_average :
  overall_average = 120.567 :=
by
  -- Proof to be filled in
  sorry

end david_overall_average_l2049_204993


namespace alex_pen_difference_l2049_204970

theorem alex_pen_difference 
  (alex_initial_pens : Nat) 
  (doubling_rate : Nat) 
  (weeks : Nat) 
  (jane_pens_month : Nat) :
  alex_initial_pens = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  jane_pens_month = 16 →
  (alex_initial_pens * doubling_rate ^ weeks) - jane_pens_month = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end alex_pen_difference_l2049_204970


namespace length_of_segment_l2049_204998

theorem length_of_segment (x : ℝ) (h₀ : 0 < x ∧ x < Real.pi / 2)
  (h₁ : 6 * Real.cos x = 5 * Real.tan x) :
  ∃ P_1 P_2 : ℝ, P_1 = 0 ∧ P_2 = (1 / 2) * Real.sin x ∧ abs (P_2 - P_1) = 1 / 3 :=
by
  sorry

end length_of_segment_l2049_204998


namespace function_minimum_value_no_maximum_l2049_204940

noncomputable def f (a x : ℝ) : ℝ :=
  (Real.sin x + a) / Real.sin x

theorem function_minimum_value_no_maximum (a : ℝ) (h_a : 0 < a) : 
  ∃ x_min, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≥ x_min ∧ 
           (∀ x ∈ Set.Ioo 0 Real.pi, f a x ≠ x_min) ∧ 
           ¬ (∃ x_max, ∀ x ∈ Set.Ioo 0 Real.pi, f a x ≤ x_max) :=
by
  let t := Real.sin
  have h : ∀ x ∈ Set.Ioo 0 Real.pi, t x ∈ Set.Ioo 0 1 := sorry -- Simple property of sine function in (0, π)
  -- Exact details skipped to align with the conditions from the problem, leveraging the property
  sorry -- Full proof not required as per instructions

end function_minimum_value_no_maximum_l2049_204940


namespace fraction_simplification_l2049_204963

theorem fraction_simplification : 
  (2222 - 2123) ^ 2 / 121 = 81 :=
by
  sorry

end fraction_simplification_l2049_204963


namespace expected_digits_of_fair_icosahedral_die_l2049_204921

noncomputable def expected_num_of_digits : ℚ :=
  (9 / 20) * 1 + (11 / 20) * 2

theorem expected_digits_of_fair_icosahedral_die :
  expected_num_of_digits = 1.55 := by
  sorry

end expected_digits_of_fair_icosahedral_die_l2049_204921


namespace find_k_l2049_204973

theorem find_k (a : ℕ → ℕ) (S : ℕ → ℕ) (k : ℕ) 
  (h_nz : ∀ n, S n = n ^ 2 - a n) 
  (hSk : 1 < S k ∧ S k < 9) :
  k = 2 := 
sorry

end find_k_l2049_204973


namespace distance_from_dormitory_to_city_l2049_204983

theorem distance_from_dormitory_to_city (D : ℝ) :
  (1 / 4) * D + (1 / 2) * D + 10 = D → D = 40 :=
by
  intro h
  sorry

end distance_from_dormitory_to_city_l2049_204983


namespace range_of_m_for_basis_l2049_204955

open Real

def vector_a : ℝ × ℝ := (1, 2)
def vector_b (m : ℝ) : ℝ × ℝ := (m, 3 * m - 2)

theorem range_of_m_for_basis (m : ℝ) :
  vector_a ≠ vector_b m → m ≠ 2 :=
sorry

end range_of_m_for_basis_l2049_204955


namespace largest_last_digit_in_string_l2049_204980

theorem largest_last_digit_in_string :
  ∃ (s : Nat → Fin 10), 
    (s 0 = 1) ∧ 
    (∀ k, k < 99 → (∃ m, (s k * 10 + s (k + 1)) = 17 * m ∨ (s k * 10 + s (k + 1)) = 23 * m)) ∧
    (∃ l, l < 10 ∧ (s 99 = l)) ∧
    (forall last, (last < 10 ∧ (s 99 = last))) ∧
    (∀ m n, s 99 = m → s 99 = n → m ≤ n → n = 9) :=
sorry

end largest_last_digit_in_string_l2049_204980


namespace oscar_leap_longer_l2049_204900

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

end oscar_leap_longer_l2049_204900


namespace range_of_t_l2049_204954

theorem range_of_t (t : ℝ) : 
  (∃ x : ℝ, x^2 - 3 * x + t ≤ 0 ∧ x ≤ t) ↔ (0 ≤ t ∧ t ≤ 9 / 4) := 
sorry

end range_of_t_l2049_204954


namespace total_number_of_flowers_l2049_204919

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

end total_number_of_flowers_l2049_204919


namespace profit_percentage_is_25_l2049_204920

theorem profit_percentage_is_25 
  (selling_price : ℝ) (cost_price : ℝ) 
  (sp_val : selling_price = 600) 
  (cp_val : cost_price = 480) : 
  (selling_price - cost_price) / cost_price * 100 = 25 := by
  sorry

end profit_percentage_is_25_l2049_204920


namespace evaluate_expression_l2049_204986

-- Define the integers a and b
def a := 2019
def b := 2020

-- The main theorem stating the equivalence
theorem evaluate_expression :
  (a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3 + 6) / (a * b) = 5 / (a * b) := 
by
  sorry

end evaluate_expression_l2049_204986


namespace sallys_change_l2049_204913

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

end sallys_change_l2049_204913


namespace volume_less_than_1000_l2049_204931

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

end volume_less_than_1000_l2049_204931


namespace bryan_travel_hours_per_year_l2049_204994

-- Definitions based on the conditions
def minutes_walk_to_bus_station := 5
def minutes_ride_bus := 20
def minutes_walk_to_job := 5
def days_per_year := 365

-- Total time for one-way travel in minutes
def one_way_travel_minutes := minutes_walk_to_bus_station + minutes_ride_bus + minutes_walk_to_job

-- Total daily travel time in minutes
def daily_travel_minutes := one_way_travel_minutes * 2

-- Convert daily travel time from minutes to hours
def daily_travel_hours := daily_travel_minutes / 60

-- Total yearly travel time in hours
def yearly_travel_hours := daily_travel_hours * days_per_year

-- The theorem to prove
theorem bryan_travel_hours_per_year : yearly_travel_hours = 365 :=
by {
  -- The preliminary arithmetic is not the core of the theorem
  sorry
}

end bryan_travel_hours_per_year_l2049_204994


namespace maximum_bottles_l2049_204909

-- Definitions for the number of bottles each shop sells
def bottles_from_shop_A : ℕ := 150
def bottles_from_shop_B : ℕ := 180
def bottles_from_shop_C : ℕ := 220

-- The main statement to prove
theorem maximum_bottles : bottles_from_shop_A + bottles_from_shop_B + bottles_from_shop_C = 550 := 
by 
  sorry

end maximum_bottles_l2049_204909


namespace monitor_width_l2049_204905

theorem monitor_width (d w h : ℝ) (h_ratio : w / h = 16 / 9) (h_diag : d = 24) :
  w = 384 / Real.sqrt 337 :=
by
  sorry

end monitor_width_l2049_204905


namespace jina_has_1_koala_bear_l2049_204902

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

end jina_has_1_koala_bear_l2049_204902


namespace factor_fraction_eq_l2049_204910

theorem factor_fraction_eq (a b c : ℝ) :
  ((a^2 + b^2)^3 + (b^2 + c^2)^3 + (c^2 + a^2)^3) 
  / ((a + b)^3 + (b + c)^3 + (c + a)^3) = 
  ((a^2 + b^2) * (b^2 + c^2) * (c^2 + a^2)) 
  / ((a + b) * (b + c) * (c + a)) :=
by
  sorry

end factor_fraction_eq_l2049_204910


namespace gcd_1043_2295_eq_1_l2049_204997

theorem gcd_1043_2295_eq_1 : Nat.gcd 1043 2295 = 1 := by
  sorry

end gcd_1043_2295_eq_1_l2049_204997


namespace bicycle_discount_l2049_204925

theorem bicycle_discount (original_price : ℝ) (discount : ℝ) (discounted_price : ℝ) :
  original_price = 760 ∧ discount = 0.75 ∧ discounted_price = 570 → 
  original_price * discount = discounted_price := by
  sorry

end bicycle_discount_l2049_204925


namespace find_m_l2049_204968

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 40 m = 120) (h2 : Nat.lcm m 45 = 180) : m = 12 := sorry

end find_m_l2049_204968


namespace solution_set_correct_l2049_204912

noncomputable def solution_set (x : ℝ) : Prop :=
  x + 2 / (x + 1) > 2

theorem solution_set_correct :
  {x : ℝ | solution_set x} = {x : ℝ | -1 < x ∧ x < 0} ∪ {x : ℝ | x > 1} :=
by sorry

end solution_set_correct_l2049_204912


namespace Soyun_distance_l2049_204969

theorem Soyun_distance
  (perimeter : ℕ)
  (Soyun_speed : ℕ)
  (Jia_speed : ℕ)
  (meeting_time : ℕ)
  (time_to_meet : perimeter = (Soyun_speed + Jia_speed) * meeting_time) :
  Soyun_speed * meeting_time = 10 :=
by
  sorry

end Soyun_distance_l2049_204969


namespace initial_pokemon_cards_l2049_204918

theorem initial_pokemon_cards (x : ℕ) (h : x - 9 = 4) : x = 13 := by
  sorry

end initial_pokemon_cards_l2049_204918


namespace total_respondents_l2049_204928

theorem total_respondents (X Y : ℕ) (hX : X = 360) (h_ratio : 9 * Y = X) : X + Y = 400 := by
  sorry

end total_respondents_l2049_204928


namespace find_n_l2049_204937

noncomputable def f (n : ℝ) : ℝ :=
  n ^ (n / 2)

example : f 2 = 2 := sorry

theorem find_n : ∃ n : ℝ, f n = 12 ∧ abs (n - 3.4641) < 0.0001 := sorry

end find_n_l2049_204937


namespace problem_I_problem_II_l2049_204926

def intervalA := { x : ℝ | -2 < x ∧ x < 5 }
def intervalB (m : ℝ) := { x : ℝ | m < x ∧ x < m + 3 }

theorem problem_I (m : ℝ) :
  (intervalB m ⊆ intervalA) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by sorry

theorem problem_II (m : ℝ) :
  (intervalA ∩ intervalB m ≠ ∅) ↔ (-5 < m ∧ m < 2) :=
by sorry

end problem_I_problem_II_l2049_204926


namespace sum_of_roots_l2049_204985

theorem sum_of_roots (x1 x2 k c : ℝ) (h1 : 2 * x1^2 - k * x1 = 2 * c) 
  (h2 : 2 * x2^2 - k * x2 = 2 * c) (h3 : x1 ≠ x2) : x1 + x2 = k / 2 := 
sorry

end sum_of_roots_l2049_204985


namespace michael_and_emma_dig_time_correct_l2049_204981

noncomputable def michael_and_emma_digging_time : ℝ :=
let father_rate := 4
let father_time := 450
let father_depth := father_rate * father_time
let mother_rate := 5
let mother_time := 300
let mother_depth := mother_rate * mother_time
let michael_desired_depth := 3 * father_depth - 600
let emma_desired_depth := 2 * mother_depth + 300
let desired_depth := max michael_desired_depth emma_desired_depth
let michael_rate := 3
let emma_rate := 6
let combined_rate := michael_rate + emma_rate
desired_depth / combined_rate

theorem michael_and_emma_dig_time_correct :
  michael_and_emma_digging_time = 533.33 := 
sorry

end michael_and_emma_dig_time_correct_l2049_204981


namespace inequality_solution_set_l2049_204927

theorem inequality_solution_set (x : ℝ) : (x - 4) * (x + 1) > 0 ↔ x > 4 ∨ x < -1 :=
by sorry

end inequality_solution_set_l2049_204927


namespace find_x_l2049_204953

noncomputable def x (n : ℕ) := 6^n + 1

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_three_prime_divisors (x : ℕ) : Prop :=
  ∃ a b c : ℕ, (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ Prime a ∧ Prime b ∧ Prime c ∧ a * b * c ∣ x ∧ ∀ d, Prime d ∧ d ∣ x → d = a ∨ d = b ∨ d = c

theorem find_x (n : ℕ) (hodd : is_odd n) (hdiv : has_three_prime_divisors (x n)) (hprime : 11 ∣ (x n)) : x n = 7777 :=
by 
  sorry

end find_x_l2049_204953


namespace irreducible_fraction_l2049_204978

theorem irreducible_fraction {n : ℕ} : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 :=
by
  sorry

end irreducible_fraction_l2049_204978


namespace unique_solution_ffx_eq_27_l2049_204932

-- Definition of the function f(x)
def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 27

-- Prove that there is exactly one solution for f(f(x)) = 27 in the domain -3 ≤ x ≤ 5
theorem unique_solution_ffx_eq_27 :
  (∃! x : ℝ, -3 ≤ x ∧ x ≤ 5 ∧ f (f x) = 27) :=
by
  sorry

end unique_solution_ffx_eq_27_l2049_204932


namespace average_track_width_l2049_204916

theorem average_track_width (r1 r2 s1 s2 : ℝ) 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : 2 * Real.pi * s1 - 2 * Real.pi * s2 = 30 * Real.pi) :
  (r1 - r2 + (s1 - s2)) / 2 = 12.5 := 
sorry

end average_track_width_l2049_204916


namespace brown_gumdrops_count_l2049_204917

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

end brown_gumdrops_count_l2049_204917


namespace area_of_bounded_region_l2049_204945

theorem area_of_bounded_region (x y : ℝ) (h : y^2 + 2 * x * y + 50 * abs x = 500) : 
  ∃ A, A = 1250 :=
sorry

end area_of_bounded_region_l2049_204945


namespace determine_coefficients_l2049_204974

variable {α : Type} [Field α]
variables (a a1 a2 a3 : α)

theorem determine_coefficients (h : ∀ x : α, a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 = x^3) :
  a = 1 ∧ a2 = 3 :=
by
  -- To be proven
  sorry

end determine_coefficients_l2049_204974


namespace quadratic_has_distinct_real_roots_l2049_204903

theorem quadratic_has_distinct_real_roots (m : ℝ) :
  ∃ (a b c : ℝ), a = 1 ∧ b = -2 ∧ c = m - 1 ∧ (b^2 - 4 * a * c > 0) → (m < 2) :=
by
  sorry

end quadratic_has_distinct_real_roots_l2049_204903


namespace travel_time_l2049_204951

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l2049_204951


namespace part1_solution_part2_solution_part3_solution_l2049_204966

-- Define the basic conditions
variables (x y m : ℕ)

-- Part 1: Number of pieces of each type purchased (Proof for 10 pieces of A, 20 pieces of B)
theorem part1_solution (h1 : x + y = 30) (h2 : 28 * x + 22 * y = 720) :
  (x = 10) ∧ (y = 20) :=
sorry

-- Part 2: Maximize sales profit for the second purchase
theorem part2_solution (h1 : 28 * m + 22 * (80 - m) ≤ 2000) :
  m = 40 ∧ (max_profit = 1040) :=
sorry

-- Variables for Part 3
variables (a : ℕ)
-- Profit equation for type B apples with adjusted selling price
theorem part3_solution (h : (4 + 2 * a) * (34 - a - 22) = 90) :
  (a = 7) ∧ (selling_price = 27) :=
sorry

end part1_solution_part2_solution_part3_solution_l2049_204966


namespace inequality_holds_l2049_204941

-- Define the function f
variable (f : ℝ → ℝ)

-- Given conditions
axiom symmetric_property : ∀ x : ℝ, f (1 - x) = f (1 + x)
axiom increasing_property : ∀ x y : ℝ, (1 ≤ x) → (x ≤ y) → f x ≤ f y

-- The statement of the theorem
theorem inequality_holds (m : ℝ) (h : m < 1 / 2) : f (1 - m) < f m :=
by sorry

end inequality_holds_l2049_204941


namespace option_C_correct_l2049_204957

theorem option_C_correct (a b : ℝ) : 
  (1 / (b / a) * (a / b) = a^2 / b^2) :=
sorry

end option_C_correct_l2049_204957


namespace swim_meet_time_l2049_204908

theorem swim_meet_time {distance : ℕ} (d : distance = 50) (t : ℕ) 
  (meet_first : ∃ t1 : ℕ, t1 = 2 ∧ distance - 20 = 30) 
  (turn : ∀ t1, t1 = 2 → ∀ d1 : ℕ, d1 = 50 → t1 + t1 = 4) :
  t = 4 :=
by
  -- Placeholder proof
  sorry

end swim_meet_time_l2049_204908


namespace problem_statement_l2049_204959

def permutations (n r : ℕ) : ℕ := n.factorial / (n - r).factorial
def combinations (n r : ℕ) : ℕ := n.factorial / (r.factorial * (n - r).factorial)

theorem problem_statement : permutations 4 2 - combinations 4 3 = 8 := 
by 
  sorry

end problem_statement_l2049_204959


namespace mark_money_l2049_204943

theorem mark_money (M : ℝ) (h1 : M / 2 + 14 ≤ M) (h2 : M / 3 + 16 ≤ M) :
  M - (M / 2 + 14) - (M / 3 + 16) = 0 → M = 180 := by
  sorry

end mark_money_l2049_204943


namespace lesser_solution_quadratic_l2049_204975

theorem lesser_solution_quadratic (x : ℝ) :
  x^2 + 9 * x - 22 = 0 → x = -11 ∨ x = 2 :=
sorry

end lesser_solution_quadratic_l2049_204975


namespace p_of_neg3_equals_14_l2049_204984

-- Functions definitions
def u (x : ℝ) : ℝ := 4 * x + 5
def p (y : ℝ) : ℝ := y^2 - 2 * y + 6

-- Theorem statement
theorem p_of_neg3_equals_14 : p (-3) = 14 := by
  sorry

end p_of_neg3_equals_14_l2049_204984


namespace find_pairs_l2049_204971

theorem find_pairs (m n : ℕ) : 
  ∃ x : ℤ, x * x = 2^m * 3^n + 1 ↔ (m = 3 ∧ n = 1) ∨ (m = 4 ∧ n = 1) ∨ (m = 5 ∧ n = 2) :=
by
  sorry

end find_pairs_l2049_204971


namespace nonstudent_ticket_cost_l2049_204948

theorem nonstudent_ticket_cost :
  ∃ x : ℝ, (530 * 2 + (821 - 530) * x = 1933) ∧ x = 3 :=
by 
  sorry

end nonstudent_ticket_cost_l2049_204948


namespace total_combined_rainfall_l2049_204967

def mondayRainfall := 7 * 1
def tuesdayRainfall := 4 * 2
def wednesdayRate := 2 * 2
def wednesdayRainfall := 2 * wednesdayRate
def totalRainfall := mondayRainfall + tuesdayRainfall + wednesdayRainfall

theorem total_combined_rainfall : totalRainfall = 23 :=
by
  unfold totalRainfall mondayRainfall tuesdayRainfall wednesdayRainfall wednesdayRate
  sorry

end total_combined_rainfall_l2049_204967


namespace area_of_rectangle_l2049_204923

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

end area_of_rectangle_l2049_204923


namespace vector_on_line_l2049_204924

noncomputable def k_value (a b : Vector ℝ 3) (m : ℝ) : ℝ :=
  if h : m = 5 / 7 then
    (5 / 7 : ℝ)
  else
    0 -- This branch will never be taken because we will assume m = 5 / 7 as a hypothesis.


theorem vector_on_line (a b : Vector ℝ 3) (m k : ℝ) (h : m = 5 / 7) :
  k = k_value a b m :=
by
  sorry

end vector_on_line_l2049_204924


namespace anthony_initial_pencils_l2049_204935

def initial_pencils (given_pencils : ℝ) (remaining_pencils : ℝ) : ℝ :=
  given_pencils + remaining_pencils

theorem anthony_initial_pencils :
  initial_pencils 9.0 47.0 = 56.0 :=
by
  sorry

end anthony_initial_pencils_l2049_204935


namespace building_height_l2049_204914

theorem building_height (h : ℕ) (flagpole_height : ℕ) (flagpole_shadow : ℕ) (building_shadow : ℕ) :
  flagpole_height = 18 ∧ flagpole_shadow = 45 ∧ building_shadow = 60 → h = 24 :=
by
  intros
  sorry

end building_height_l2049_204914


namespace not_a_solution_set4_l2049_204952

def set1 : ℝ × ℝ := (1, 2)
def set2 : ℝ × ℝ := (2, 0)
def set3 : ℝ × ℝ := (0.5, 3)
def set4 : ℝ × ℝ := (-2, 4)

noncomputable def is_solution (p : ℝ × ℝ) : Prop := 2 * p.1 + p.2 = 4

theorem not_a_solution_set4 : ¬ is_solution set4 := 
by 
  sorry

end not_a_solution_set4_l2049_204952


namespace inequality_solution_min_value_of_a2_b2_c2_min_achieved_l2049_204946

noncomputable def f (x : ℝ) : ℝ := abs (2 * x + 1) + abs (x - 1)

theorem inequality_solution :
  ∀ x : ℝ, (f x ≥ 3) ↔ (x ≤ -1 ∨ x ≥ 1) :=
by sorry

theorem min_value_of_a2_b2_c2 (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  a^2 + b^2 + c^2 ≥ 3/7 :=
by sorry

theorem min_achieved (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : (1/2)*a + b + 2*c = 3/2) :
  (2*a = b) ∧ (b = c/2) ∧ (a^2 + b^2 + c^2 = 3/7) :=
by sorry

end inequality_solution_min_value_of_a2_b2_c2_min_achieved_l2049_204946


namespace initial_population_l2049_204990

theorem initial_population (P : ℝ) : 
  (P * 1.2 * 0.8 = 9600) → P = 10000 :=
by
  sorry

end initial_population_l2049_204990


namespace students_taking_all_three_classes_l2049_204904

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

end students_taking_all_three_classes_l2049_204904


namespace find_a_l2049_204929

theorem find_a (a b : ℤ) (h1 : a > b) (h2 : b > 0) (h3 : a + b + a * b = 119) : a = 59 :=
sorry

end find_a_l2049_204929


namespace law_firm_more_than_two_years_l2049_204992

theorem law_firm_more_than_two_years (p_second p_not_first : ℝ) : 
  p_second = 0.30 →
  p_not_first = 0.60 →
  ∃ p_more_than_two_years : ℝ, p_more_than_two_years = 0.30 :=
by
  intros h1 h2
  use (p_not_first - p_second)
  rw [h1, h2]
  norm_num
  done

end law_firm_more_than_two_years_l2049_204992


namespace olivia_wallet_l2049_204956

theorem olivia_wallet (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 78)
  (h2 : spent_amount = 15):
  remaining_amount = initial_amount - spent_amount →
  remaining_amount = 63 :=
sorry

end olivia_wallet_l2049_204956


namespace total_sales_l2049_204976

noncomputable def sales_in_june : ℕ := 96
noncomputable def sales_in_july : ℕ := sales_in_june * 4 / 3

theorem total_sales (june_sales : ℕ) (july_sales : ℕ) (h1 : june_sales = 96)
                    (h2 : july_sales = june_sales * 4 / 3) :
                    june_sales + july_sales = 224 :=
by
  rw [h1, h2]
  norm_num
  sorry

end total_sales_l2049_204976


namespace bob_25_cent_coins_l2049_204907

theorem bob_25_cent_coins (a b c : ℕ)
    (h₁ : a + b + c = 15)
    (h₂ : 15 + 4 * c = 27) : c = 3 := by
  sorry

end bob_25_cent_coins_l2049_204907


namespace total_students_high_school_l2049_204906

theorem total_students_high_school (s10 s11 s12 total_students sample: ℕ ) 
  (h1 : s10 = 600) 
  (h2 : sample = 45) 
  (h3 : s11 = 20) 
  (h4 : s12 = 10) 
  (h5 : sample = s10 + s11 + s12) : 
  total_students = 1800 :=
by 
  sorry

end total_students_high_school_l2049_204906


namespace salt_percentage_in_first_solution_l2049_204933

variable (S : ℚ)
variable (H : 0 ≤ S ∧ S ≤ 100)  -- percentage constraints

theorem salt_percentage_in_first_solution (h : 0.75 * S / 100 + 7 = 16) : S = 12 :=
by { sorry }

end salt_percentage_in_first_solution_l2049_204933


namespace fg_of_2_l2049_204934

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x + 1)^2

theorem fg_of_2 : f (g 2) = 29 := by
  sorry

end fg_of_2_l2049_204934


namespace find_complementary_angle_l2049_204947

theorem find_complementary_angle (angle1 angle2 : ℝ) (h1 : angle1 + angle2 = 90) (h2 : angle1 = 25) : angle2 = 65 := 
by 
  sorry

end find_complementary_angle_l2049_204947


namespace perpendicular_tangents_sum_x1_x2_gt_4_l2049_204930

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

end perpendicular_tangents_sum_x1_x2_gt_4_l2049_204930


namespace find_playground_side_length_l2049_204991

-- Define the conditions
def playground_side_length (x : ℝ) : Prop :=
  let perimeter_square := 4 * x
  let perimeter_garden := 2 * (12 + 9)
  let total_perimeter := perimeter_square + perimeter_garden
  total_perimeter = 150

-- State the main theorem to prove that the side length of the square fence around the playground is 27 yards
theorem find_playground_side_length : ∃ x : ℝ, playground_side_length x ∧ x = 27 :=
by
  exists 27
  sorry

end find_playground_side_length_l2049_204991


namespace common_chord_eq_l2049_204960

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 10*x - 10*y = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 6*x - 2*y - 40 = 0

-- Define the statement to prove
theorem common_chord_eq (x y : ℝ) : circle1 x y ∧ circle2 x y → 2*x + y - 5 = 0 :=
sorry

end common_chord_eq_l2049_204960


namespace sufficient_balance_after_29_months_l2049_204961

noncomputable def accumulated_sum (S0 : ℕ) (D : ℕ) (r : ℚ) (n : ℕ) : ℚ :=
  S0 * (1 + r)^n + D * ((1 + r)^n - 1) / r

theorem sufficient_balance_after_29_months :
  let S0 := 300000
  let D := 15000
  let r := (1 / 100 : ℚ) -- interest rate of 1%
  accumulated_sum S0 D r 29 ≥ 900000 :=
by
  sorry -- The proof will be elaborated later

end sufficient_balance_after_29_months_l2049_204961


namespace composite_number_iff_ge_2_l2049_204922

theorem composite_number_iff_ge_2 (n : ℕ) : 
  ¬(Prime (3^(2*n+1) - 2^(2*n+1) - 6^n)) ↔ n ≥ 2 := by
  sorry

end composite_number_iff_ge_2_l2049_204922


namespace max_sum_of_factors_l2049_204988

theorem max_sum_of_factors (heartsuit spadesuit : ℕ) (h : heartsuit * spadesuit = 24) :
  heartsuit + spadesuit ≤ 25 :=
sorry

end max_sum_of_factors_l2049_204988


namespace Mike_can_play_300_minutes_l2049_204938

-- Define the weekly earnings, spending, and costs as conditions
def weekly_earnings : ℕ := 100
def half_spent_at_arcade : ℕ := weekly_earnings / 2
def food_cost : ℕ := 10
def token_cost_per_hour : ℕ := 8
def hour_in_minutes : ℕ := 60

-- Define the remaining money after buying food
def money_for_tokens : ℕ := half_spent_at_arcade - food_cost

-- Define the hours he can play
def hours_playable : ℕ := money_for_tokens / token_cost_per_hour

-- Define the total minutes he can play
def total_minutes_playable : ℕ := hours_playable * hour_in_minutes

-- Prove that with his expenditure, Mike can play for 300 minutes
theorem Mike_can_play_300_minutes : total_minutes_playable = 300 := 
by
  sorry -- Proof will be filled here

end Mike_can_play_300_minutes_l2049_204938


namespace greatest_integer_l2049_204987

theorem greatest_integer (n : ℕ) (h1 : n < 150) (h2 : ∃ k : ℤ, n = 9 * k - 2) (h3 : ∃ l : ℤ, n = 8 * l - 4) : n = 124 := 
sorry

end greatest_integer_l2049_204987


namespace largest_sum_is_1173_l2049_204950

def largest_sum_of_two_3digit_numbers : Prop :=
  ∃ a b c d e f : ℕ, 
  (a = 6 ∧ b = 5 ∧ c = 4 ∧ d = 3 ∧ e = 2 ∧ f = 1) ∧
  100 * (a + b) + 10 * (c + d) + (e + f) = 1173

theorem largest_sum_is_1173 : largest_sum_of_two_3digit_numbers :=
  by
  sorry

end largest_sum_is_1173_l2049_204950


namespace total_words_in_week_l2049_204915

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

end total_words_in_week_l2049_204915


namespace range_of_m_l2049_204999

noncomputable def A := {x : ℝ | -3 ≤ x ∧ x ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ m + 1}

theorem range_of_m (m : ℝ) (h : B m ⊆ A) : -2 ≤ m ∧ m ≤ 3 :=
sorry

end range_of_m_l2049_204999
