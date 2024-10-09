import Mathlib

namespace truthful_dwarfs_count_l1336_133618

theorem truthful_dwarfs_count (x y: ℕ) (h_sum: x + y = 10) 
                              (h_hands: x + 2 * y = 16) : x = 4 := 
by
  sorry

end truthful_dwarfs_count_l1336_133618


namespace min_value_d_l1336_133619

theorem min_value_d (a b c d : ℕ) (h1 : a < b) (h2 : b < c) (h3 : c < d)
  (unique_solution : ∃! x y : ℤ, 2 * x + y = 2007 ∧ y = (abs (x - a) + abs (x - b) + abs (x - c) + abs (x - d))) :
  d = 504 :=
sorry

end min_value_d_l1336_133619


namespace find_a_l1336_133683

theorem find_a (a : ℝ) (h : (1 / Real.log 3 / Real.log a) + (1 / Real.log 5 / Real.log a) + (1 / Real.log 7 / Real.log a) = 1) : 
  a = 105 := 
sorry

end find_a_l1336_133683


namespace quadratic_solutions_l1336_133657

theorem quadratic_solutions (x : ℝ) :
  (4 * x^2 - 6 * x = 0) ↔ (x = 0) ∨ (x = 3 / 2) :=
sorry

end quadratic_solutions_l1336_133657


namespace length_fraction_of_radius_l1336_133685

noncomputable def side_of_square_area (A : ℕ) : ℕ := Nat.sqrt A
noncomputable def radius_of_circle_from_square_area (A : ℕ) : ℕ := side_of_square_area A

noncomputable def length_of_rectangle_from_area_breadth (A b : ℕ) : ℕ := A / b
noncomputable def fraction_of_radius (len rad : ℕ) : ℚ := len / rad

theorem length_fraction_of_radius 
  (A_square A_rect breadth : ℕ) 
  (h_square_area : A_square = 1296)
  (h_rect_area : A_rect = 360)
  (h_breadth : breadth = 10) : 
  fraction_of_radius 
    (length_of_rectangle_from_area_breadth A_rect breadth)
    (radius_of_circle_from_square_area A_square) = 1 := 
by
  sorry

end length_fraction_of_radius_l1336_133685


namespace remainder_division_l1336_133669

theorem remainder_division (x : ℝ) :
  (x ^ 2021 + 1) % (x ^ 12 - x ^ 9 + x ^ 6 - x ^ 3 + 1) = -x ^ 4 + 1 :=
sorry

end remainder_division_l1336_133669


namespace negation_of_universal_prop_l1336_133607

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, x^3 + 3^x > 0) ↔ (∃ x : ℝ, x^3 + 3^x ≤ 0) :=
by sorry

end negation_of_universal_prop_l1336_133607


namespace sum_of_consecutive_integers_l1336_133676

theorem sum_of_consecutive_integers (n : ℕ) (h : n * (n + 1) * (n + 2) * (n + 3) = 358800) : 
  n + (n + 1) + (n + 2) + (n + 3) = 98 :=
sorry

end sum_of_consecutive_integers_l1336_133676


namespace jenna_peeled_potatoes_l1336_133674

-- Definitions of constants
def initial_potatoes : ℕ := 60
def homer_rate : ℕ := 4
def jenna_rate : ℕ := 6
def combined_rate : ℕ := homer_rate + jenna_rate
def homer_time : ℕ := 6
def remaining_potatoes : ℕ := initial_potatoes - (homer_rate * homer_time)
def combined_time : ℕ := 4 -- Rounded from 3.6

-- Statement to prove
theorem jenna_peeled_potatoes : remaining_potatoes / combined_rate * jenna_rate = 24 :=
by
  sorry

end jenna_peeled_potatoes_l1336_133674


namespace product_implication_l1336_133631

theorem product_implication (a b : ℝ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hab : a * b > 1) : a > 1 ∨ b > 1 :=
sorry

end product_implication_l1336_133631


namespace solution_set_of_abs_inequality_is_real_l1336_133659

theorem solution_set_of_abs_inequality_is_real (m : ℝ) :
  (∀ x : ℝ, |x + 1| + |x - 2| + m - 7 > 0) ↔ m > 4 :=
by
  sorry

end solution_set_of_abs_inequality_is_real_l1336_133659


namespace pendulum_faster_17_seconds_winter_l1336_133691

noncomputable def pendulum_period (l g : ℝ) : ℝ :=
  2 * Real.pi * Real.sqrt (l / g)

noncomputable def pendulum_seconds_faster_in_winter (T : ℝ) (l : ℝ) (g : ℝ) (shorten : ℝ) (hours : ℝ) : ℝ :=
  let summer_period := T
  let winter_length := l - shorten
  let winter_period := pendulum_period winter_length g
  let summer_cycles := (hours * 60 * 60) / summer_period
  let winter_cycles := (hours * 60 * 60) / winter_period
  winter_cycles - summer_cycles

theorem pendulum_faster_17_seconds_winter :
  let T := 1
  let l := 980 * (1 / (4 * Real.pi ^ 2))
  let g := 980
  let shorten := 0.01 / 100
  let hours := 24
  pendulum_seconds_faster_in_winter T l g shorten hours = 17 :=
by
  sorry

end pendulum_faster_17_seconds_winter_l1336_133691


namespace neces_not_suff_cond_l1336_133610

theorem neces_not_suff_cond (a : ℝ) (h : a ≠ 0) : (1 / a < 1) → (a > 1) :=
sorry

end neces_not_suff_cond_l1336_133610


namespace parametric_plane_equiv_l1336_133608

/-- Define the parametric form of the plane -/
def parametric_plane (s t : ℝ) : ℝ × ℝ × ℝ :=
  (1 + s - t, 2 - s, 3 - 2*s + 2*t)

/-- Define the equation of the plane in standard form -/
def plane_equation (x y z : ℝ) : Prop :=
  2 * x + z - 5 = 0

/-- The theorem stating that the parametric form corresponds to the given plane equation -/
theorem parametric_plane_equiv :
  ∃ x y z s t,
    (x, y, z) = parametric_plane s t ∧ plane_equation x y z :=
by
  sorry

end parametric_plane_equiv_l1336_133608


namespace ball_distribution_ways_l1336_133695

theorem ball_distribution_ways :
  let R := 5
  let W := 3
  let G := 2
  let total_balls := 10
  let balls_in_first_box := 4
  ∃ (distributions : ℕ), distributions = (Nat.choose total_balls balls_in_first_box) ∧ distributions = 210 :=
by
  sorry

end ball_distribution_ways_l1336_133695


namespace quadratic_properties_l1336_133675

open Real

noncomputable section

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ := x^2 - 4 * x + 3

-- Vertex form of the quadratic
def vertexForm (x : ℝ) : ℝ := (x - 2)^2 - 1

-- Axis of symmetry
def axisOfSymmetry : ℝ := 2

-- Vertex of the quadratic
def vertex : ℝ × ℝ := (2, -1)

-- Minimum value of the quadratic
def minimumValue : ℝ := -1

-- Interval where the function decreases
def decreasingInterval (x : ℝ) : Prop := -1 ≤ x ∧ x < 2

-- Range of y in the interval -1 <= x < 3
def rangeOfY (y : ℝ) : Prop := -1 ≤ y ∧ y ≤ 8

-- Main statement
theorem quadratic_properties :
  (∀ x, quadratic x = vertexForm x) ∧
  (∃ x, axisOfSymmetry = x) ∧
  (∃ v, vertex = v) ∧
  (minimumValue = -1) ∧
  (∀ x, -1 ≤ x ∧ x < 2 → quadratic x > quadratic (x + 1)) ∧
  (∀ y, (∃ x, -1 ≤ x ∧ x < 3 ∧ y = quadratic x) → rangeOfY y) :=
sorry

end quadratic_properties_l1336_133675


namespace manager_hourly_wage_l1336_133626

open Real

theorem manager_hourly_wage (M D C : ℝ) 
  (hD : D = M / 2)
  (hC : C = 1.20 * D)
  (hC_manager : C = M - 3.40) :
  M = 8.50 :=
by
  sorry

end manager_hourly_wage_l1336_133626


namespace power_addition_l1336_133648

theorem power_addition :
  (-2 : ℤ) ^ 2009 + (-2 : ℤ) ^ 2010 = 2 ^ 2009 :=
by
  sorry

end power_addition_l1336_133648


namespace M_inter_N_is_01_l1336_133687

variable (x : ℝ)

def M := { x : ℝ | Real.log (1 - x) < 0 }
def N := { x : ℝ | -1 ≤ x ∧ x ≤ 1 }

theorem M_inter_N_is_01 : M ∩ N = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  -- Proof will go here
  sorry

end M_inter_N_is_01_l1336_133687


namespace not_associative_star_l1336_133662

def star (x y : ℝ) : ℝ := (x + 2) * (y + 2) - x - y

theorem not_associative_star : ¬ (∀ x y z : ℝ, star (star x y) z = star x (star y z)) :=
by
  sorry

end not_associative_star_l1336_133662


namespace medal_award_count_l1336_133628

theorem medal_award_count :
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  no_canadians_get_medals + one_canadian_gets_medal = 480 :=
by
  let total_runners := 10
  let canadian_runners := 4
  let medals := 3
  let non_canadian_runners := total_runners - canadian_runners
  let no_canadians_get_medals := Nat.choose non_canadian_runners medals * Nat.factorial medals
  let one_canadian_gets_medal := canadian_runners * medals * (Nat.choose non_canadian_runners (medals - 1) * Nat.factorial (medals - 1))
  show no_canadians_get_medals + one_canadian_gets_medal = 480
  -- here should be the steps skipped
  sorry

end medal_award_count_l1336_133628


namespace perpendicular_line_through_point_l1336_133636

def point : ℝ × ℝ := (1, 0)

def given_line (x y : ℝ) : Prop := x - y + 2 = 0

def is_perpendicular_to (l1 l2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, l1 x y → l2 (y - x) (-x - y + 2)

def target_line (x y : ℝ) : Prop := x + y - 1 = 0

theorem perpendicular_line_through_point (l1 : ℝ → ℝ → Prop) (p : ℝ × ℝ) :
  given_line = l1 ∧ p = point →
  (∃ l2 : ℝ → ℝ → Prop, is_perpendicular_to l1 l2 ∧ l2 p.1 p.2) →
  target_line p.1 p.2 :=
by
  intro hp hl2
  sorry

end perpendicular_line_through_point_l1336_133636


namespace break_even_point_l1336_133637

/-- Conditions of the problem -/
def fixed_costs : ℝ := 10410
def variable_cost_per_unit : ℝ := 2.65
def selling_price_per_unit : ℝ := 20

/-- The mathematically equivalent proof problem / statement -/
theorem break_even_point :
  fixed_costs / (selling_price_per_unit - variable_cost_per_unit) = 600 := 
by
  -- Proof to be filled in
  sorry

end break_even_point_l1336_133637


namespace sum_of_prism_features_l1336_133638

theorem sum_of_prism_features : (12 + 8 + 6 = 26) := by
  sorry

end sum_of_prism_features_l1336_133638


namespace f_at_2018_l1336_133694

noncomputable def f : ℝ → ℝ := sorry

axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom f_periodic : ∀ x : ℝ, f (x + 6) = f x
axiom f_at_4 : f 4 = 5

theorem f_at_2018 : f 2018 = 5 :=
by
  -- Proof goes here
  sorry

end f_at_2018_l1336_133694


namespace num_pos_cubes_ending_in_5_lt_5000_l1336_133666

theorem num_pos_cubes_ending_in_5_lt_5000 : 
  (∃ (n1 n2 : ℕ), (n1 ≤ 5000 ∧ n2 ≤ 5000) ∧ (n1^3 % 10 = 5 ∧ n2^3 % 10 = 5) ∧ (n1^3 < 5000 ∧ n2^3 < 5000) ∧ n1 ≠ n2 ∧ 
  ∀ n, (n^3 < 5000 ∧ n^3 % 10 = 5) → (n = n1 ∨ n = n2)) :=
sorry

end num_pos_cubes_ending_in_5_lt_5000_l1336_133666


namespace total_weight_correct_l1336_133646

variable (c1 c2 w2 c : Float)

def total_weight (c1 c2 w2 c : Float) (W x : Float) :=
  (c1 * x + c2 * w2) / (x + w2) = c ∧ W = x + w2

theorem total_weight_correct :
  total_weight 9 8 12 8.40 20 8 :=
by sorry

end total_weight_correct_l1336_133646


namespace base_conversion_subtraction_l1336_133681

theorem base_conversion_subtraction :
  (4 * 6^4 + 3 * 6^3 + 2 * 6^2 + 1 * 6^1 + 0 * 6^0) - (3 * 7^3 + 2 * 7^2 + 1 * 7^1 + 0 * 7^0) = 4776 :=
by {
  sorry
}

end base_conversion_subtraction_l1336_133681


namespace count_students_with_green_eyes_l1336_133699

-- Definitions for the given conditions
def total_students := 50
def students_with_both := 10
def students_with_neither := 5

-- Let the number of students with green eyes be y
variable (y : ℕ) 

-- There are twice as many students with brown hair as with green eyes
def students_with_brown := 2 * y

-- There are y - 10 students with green eyes only
def students_with_green_only := y - students_with_both

-- There are 2y - 10 students with brown hair only
def students_with_brown_only := students_with_brown - students_with_both

-- Proof statement
theorem count_students_with_green_eyes (y : ℕ) 
  (h1 : (students_with_green_only) + (students_with_brown_only) + students_with_both + students_with_neither = total_students) : y = 15 := 
by
  -- sorry to skip the proof
  sorry

end count_students_with_green_eyes_l1336_133699


namespace gcd_lcm_sum_l1336_133651

-- Define the numbers and their prime factorizations
def a := 120
def b := 4620
def a_prime_factors := (2, 3) -- 2^3
def b_prime_factors := (2, 2) -- 2^2

-- Define gcd and lcm based on the problem statement
def gcd_ab := 60
def lcm_ab := 4620

-- The statement to be proved
theorem gcd_lcm_sum : gcd a b + lcm a b = 4680 :=
by sorry

end gcd_lcm_sum_l1336_133651


namespace original_price_l1336_133621

theorem original_price (P : ℝ) (h : 0.75 * (0.75 * P) = 17) : P = 30.22 :=
by
  sorry

end original_price_l1336_133621


namespace smallest_number_of_weights_l1336_133677

/-- The smallest number of weights in a set that can be divided into 4, 5, and 6 equal piles is 11. -/
theorem smallest_number_of_weights (n : ℕ) (M : ℕ) : (∀ k : ℕ, (k = 4 ∨ k = 5 ∨ k = 6) → M % k = 0) → n = 11 :=
sorry

end smallest_number_of_weights_l1336_133677


namespace total_players_on_ground_l1336_133663

def cricket_players : ℕ := 15
def hockey_players : ℕ := 12
def football_players : ℕ := 13
def softball_players : ℕ := 15

theorem total_players_on_ground : 
  cricket_players + hockey_players + football_players + softball_players = 55 := 
by
  sorry

end total_players_on_ground_l1336_133663


namespace residue_class_equivalence_l1336_133678

variable {a m : ℤ}
variable {b : ℤ}

def residue_class (a m b : ℤ) : Prop := ∃ t : ℤ, b = m * t + a

theorem residue_class_equivalence (m a b : ℤ) :
  (∃ t : ℤ, b = m * t + a) ↔ b % m = a % m :=
by sorry

end residue_class_equivalence_l1336_133678


namespace second_solution_volume_l1336_133632

theorem second_solution_volume
  (V : ℝ)
  (h1 : 0.20 * 6 + 0.60 * V = 0.36 * (6 + V)) : 
  V = 4 :=
sorry

end second_solution_volume_l1336_133632


namespace salary_increase_l1336_133696

theorem salary_increase (x : ℝ) 
  (h : ∀ s : ℕ, 1 ≤ s ∧ s ≤ 5 → ∃ p : ℝ, p = 7.50 + x * (s - 1))
  (h₁ : ∃ p₁ p₅ : ℝ, 1 ≤ 1 ∧ 5 ≤ 5 ∧ p₅ = p₁ + 1.25) :
  x = 0.3125 := sorry

end salary_increase_l1336_133696


namespace fuelA_amount_l1336_133612

def tankCapacity : ℝ := 200
def ethanolInFuelA : ℝ := 0.12
def ethanolInFuelB : ℝ := 0.16
def totalEthanol : ℝ := 30
def limitedFuelA : ℝ := 100
def limitedFuelB : ℝ := 150

theorem fuelA_amount : ∃ (x : ℝ), 
  (x ≤ limitedFuelA ∧ x ≥ 0) ∧ 
  ((tankCapacity - x) ≤ limitedFuelB ∧ (tankCapacity - x) ≥ 0) ∧ 
  (ethanolInFuelA * x + ethanolInFuelB * (tankCapacity - x)) = totalEthanol ∧ 
  x = 50 := 
by
  sorry

end fuelA_amount_l1336_133612


namespace faces_painted_morning_l1336_133661

def faces_of_cuboid : ℕ := 6
def faces_painted_evening : ℕ := 3

theorem faces_painted_morning : faces_of_cuboid - faces_painted_evening = 3 := 
by 
  sorry

end faces_painted_morning_l1336_133661


namespace min_tablets_to_extract_l1336_133601

noncomputable def min_tablets_needed : ℕ :=
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  worst_case + required_A -- 14 + 18 + 20 + 3 = 55

theorem min_tablets_to_extract : min_tablets_needed = 55 :=
by {
  let A := 10
  let B := 14
  let C := 18
  let D := 20
  let required_A := 3
  let required_B := 4
  let required_C := 3
  let required_D := 2
  let worst_case := B + C + D
  have h : worst_case + required_A = 55 := by decide
  exact h
}

end min_tablets_to_extract_l1336_133601


namespace probability_three_specific_cards_l1336_133622

noncomputable def deck_size : ℕ := 52
noncomputable def num_suits : ℕ := 4
noncomputable def cards_per_suit : ℕ := 13
noncomputable def p_king_spades : ℚ := 1 / deck_size
noncomputable def p_10_hearts : ℚ := 1 / (deck_size - 1)
noncomputable def p_queen : ℚ := 4 / (deck_size - 2)

theorem probability_three_specific_cards :
  (p_king_spades * p_10_hearts * p_queen) = 1 / 33150 := 
sorry

end probability_three_specific_cards_l1336_133622


namespace part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l1336_133605

noncomputable def a (n : ℕ) : ℚ := 1 / (n : ℚ)

noncomputable def S (n : ℕ) : ℚ := (Finset.range (n+1)).sum (λ k => a (k + 1))

noncomputable def f (n : ℕ) : ℚ :=
  if n = 1 then S 2
  else S (2 * n) - S (n - 1)

theorem part1_f1 : f 1 = 3 / 2 := by sorry

theorem part1_f2 : f 2 = 13 / 12 := by sorry

theorem part1_f3 : f 3 = 19 / 20 := by sorry

theorem part2_f_gt_1_for_n_1_2 (n : ℕ) (h₁ : n = 1 ∨ n = 2) : f n > 1 := by sorry

theorem part2_f_lt_1_for_n_ge_3 (n : ℕ) (h₁ : n ≥ 3) : f n < 1 := by sorry

end part1_f1_part1_f2_part1_f3_part2_f_gt_1_for_n_1_2_part2_f_lt_1_for_n_ge_3_l1336_133605


namespace sum_numbers_l1336_133665

theorem sum_numbers : 3456 + 4563 + 5634 + 6345 = 19998 := by
  sorry

end sum_numbers_l1336_133665


namespace cubic_sum_l1336_133660

theorem cubic_sum (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 13) : x^3 + y^3 = 35 := by
  -- Using the provided conditions x + y = 5 and x^2 + y^2 = 13
  sorry

end cubic_sum_l1336_133660


namespace least_possible_square_area_l1336_133668

theorem least_possible_square_area (s : ℝ) (h1 : 4.5 ≤ s) (h2 : s < 5.5) : s * s ≥ 20.25 := by
  sorry

end least_possible_square_area_l1336_133668


namespace bus_people_difference_l1336_133698

theorem bus_people_difference 
  (initial : ℕ) (got_off : ℕ) (got_on : ℕ) (current : ℕ) 
  (h_initial : initial = 35)
  (h_got_off : got_off = 18)
  (h_got_on : got_on = 15)
  (h_current : current = initial - got_off + got_on) :
  initial - current = 3 := by
  sorry

end bus_people_difference_l1336_133698


namespace value_of_y_l1336_133654

theorem value_of_y (x y : ℝ) (h1 : y = (x^2 - 9) / (x - 3)) (h2 : y = 3 * x) : y = 9 / 2 :=
sorry

end value_of_y_l1336_133654


namespace truck_total_distance_l1336_133658

noncomputable def truck_distance (b t : ℝ) : ℝ :=
  let acceleration := b / 3
  let time_seconds := 300 + t
  let distance_feet := (1 / 2) * (acceleration / t) * time_seconds^2
  distance_feet / 5280

theorem truck_total_distance (b t : ℝ) : 
  truck_distance b t = b * (90000 + 600 * t + t ^ 2) / (31680 * t) :=
by
  sorry

end truck_total_distance_l1336_133658


namespace ball_radius_l1336_133609

theorem ball_radius (x r : ℝ) (h1 : x^2 + 256 = r^2) (h2 : r = x + 16) : r = 16 :=
by
  sorry

end ball_radius_l1336_133609


namespace production_rate_l1336_133655

theorem production_rate (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (H : x * x * 2 * x = 2 * x^3) :
  y * y * 3 * y = 3 * y^3 := by
  sorry

end production_rate_l1336_133655


namespace pamela_spilled_sugar_l1336_133615

theorem pamela_spilled_sugar 
  (original_amount : ℝ)
  (amount_left : ℝ)
  (h1 : original_amount = 9.8)
  (h2 : amount_left = 4.6)
  : original_amount - amount_left = 5.2 :=
by 
  sorry

end pamela_spilled_sugar_l1336_133615


namespace intersection_M_N_l1336_133671

-- Definition of the sets M and N
def M : Set ℝ := {x | 4 < x ∧ x < 8}
def N : Set ℝ := {x | x^2 - 6 * x < 0}

-- Intersection of M and N
def intersection : Set ℝ := {x | 4 < x ∧ x < 6}

-- Theorem statement asserting the equality between the intersection and the desired set
theorem intersection_M_N : ∀ (x : ℝ), x ∈ M ∩ N ↔ x ∈ intersection := by
  sorry

end intersection_M_N_l1336_133671


namespace Mike_books_l1336_133604

theorem Mike_books
  (initial_books : ℝ)
  (books_sold : ℝ)
  (books_gifts : ℝ) 
  (books_bought : ℝ)
  (h_initial : initial_books = 51.5)
  (h_sold : books_sold = 45.75)
  (h_gifts : books_gifts = 12.25)
  (h_bought : books_bought = 3.5):
  initial_books - books_sold + books_gifts + books_bought = 21.5 := 
sorry

end Mike_books_l1336_133604


namespace line_through_P_with_opposite_sign_intercepts_l1336_133690

theorem line_through_P_with_opposite_sign_intercepts 
  (P : ℝ × ℝ) (hP : P = (3, -2)) 
  (h : ∀ (A B : ℝ), A ≠ 0 → B ≠ 0 → A * B < 0) : 
  (∀ (x y : ℝ), (x = 5 ∧ y = -5) → (5 * x - 5 * y - 25 = 0)) ∨ (∀ (x y : ℝ), (3 * y = -2) → (y = - (2 / 3) * x)) :=
sorry

end line_through_P_with_opposite_sign_intercepts_l1336_133690


namespace car_speed_l1336_133617

theorem car_speed (distance: ℚ) (hours minutes: ℚ) (h_distance: distance = 360) (h_hours: hours = 4) (h_minutes: minutes = 30) : 
  (distance / (hours + (minutes / 60))) = 80 := by
  sorry

end car_speed_l1336_133617


namespace genevieve_initial_amount_l1336_133602

def cost_per_kg : ℕ := 8
def kg_bought : ℕ := 250
def short_amount : ℕ := 400
def total_cost : ℕ := kg_bought * cost_per_kg
def initial_amount := total_cost - short_amount

theorem genevieve_initial_amount : initial_amount = 1600 := by
  unfold initial_amount total_cost cost_per_kg kg_bought short_amount
  sorry

end genevieve_initial_amount_l1336_133602


namespace true_proposition_l1336_133649

def p : Prop := ∃ x₀ : ℝ, x₀^2 < x₀
def q : Prop := ∀ x : ℝ, x^2 - x + 1 > 0

theorem true_proposition : p ∧ q :=
by 
  sorry

end true_proposition_l1336_133649


namespace complex_magnitude_condition_l1336_133684

noncomputable def magnitude_of_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem complex_magnitude_condition (z : ℂ) (i : ℂ) (h : i * i = -1) (h1 : z - 2 * i = 1 + z * i) :
  magnitude_of_z z = Real.sqrt (10) / 2 :=
by
  -- proof goes here
  sorry

end complex_magnitude_condition_l1336_133684


namespace school_days_per_week_l1336_133652

-- Definitions based on the conditions given
def paper_per_class_per_day : ℕ := 200
def total_paper_per_week : ℕ := 9000
def number_of_classes : ℕ := 9

-- The theorem stating the main claim to prove
theorem school_days_per_week :
  total_paper_per_week / (paper_per_class_per_day * number_of_classes) = 5 :=
  by
  sorry

end school_days_per_week_l1336_133652


namespace martin_ratio_of_fruits_eaten_l1336_133624

theorem martin_ratio_of_fruits_eaten
    (initial_fruits : ℕ)
    (current_oranges : ℕ)
    (current_oranges_twice_limes : current_oranges = 2 * (current_oranges / 2))
    (initial_fruits_count : initial_fruits = 150)
    (current_oranges_count : current_oranges = 50) :
    (initial_fruits - (current_oranges + (current_oranges / 2))) / initial_fruits = 1 / 2 := 
by
    sorry

end martin_ratio_of_fruits_eaten_l1336_133624


namespace probability_x_lt_2y_l1336_133689

noncomputable def probability_x_lt_2y_in_rectangle : ℚ :=
  let area_triangle : ℚ := (1/2) * 4 * 2
  let area_rectangle : ℚ := 4 * 2
  (area_triangle / area_rectangle)

theorem probability_x_lt_2y (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ 4) (h3 : 0 ≤ y) (h4 : y ≤ 2) :
  probability_x_lt_2y_in_rectangle = 1/2 := by
  sorry

end probability_x_lt_2y_l1336_133689


namespace paint_proof_l1336_133664

/-- 
Suppose Jack's room has 27 square meters of wall and ceiling area. He has three choices for paint:
- Using 1 can of paint leaves 1 liter of paint left over,
- Using 5 gallons of paint leaves 1 liter of paint left over,
- Using 4 gallons and 2.8 liters of paint.

1. Prove: The ratio between the volume of a can and the volume of a gallon is 1:5.
2. Prove: The volume of a gallon is 3.8 liters.
3. Prove: The paint's coverage is 1.5 square meters per liter.
-/
theorem paint_proof (A : ℝ) (C G : ℝ) (R : ℝ):
  ∀ (H1: A = 27) (H2: C - 1 = 27) (H3: 5 * G - 1 = 27) (H4: 4 * G + 2.8 = 27), 
  (C / G = 1 / 5) ∧ (G = 3.8) ∧ ((A / (5 * G - 1)) = 1.5) :=
by
  sorry

end paint_proof_l1336_133664


namespace expand_expression_l1336_133640

theorem expand_expression (x y : ℝ) : 
  (2 * x + 3) * (5 * y + 7) = 10 * x * y + 14 * x + 15 * y + 21 := 
by sorry

end expand_expression_l1336_133640


namespace max_intersections_circle_pentagon_l1336_133639

theorem max_intersections_circle_pentagon : 
  ∃ (circle : Set Point) (pentagon : List (Set Point)),
    (∀ (side : Set Point), side ∈ pentagon → ∃ p1 p2 : Point, p1 ∈ circle ∧ p2 ∈ circle ∧ p1 ≠ p2) ∧
    pentagon.length = 5 →
    (∃ n : ℕ, n = 10) :=
by
  sorry

end max_intersections_circle_pentagon_l1336_133639


namespace Haley_initial_trees_l1336_133650

theorem Haley_initial_trees (T : ℕ) (h1 : T - 4 ≥ 0) (h2 : (T - 4) + 5 = 10): T = 9 :=
by
  -- proof goes here
  sorry

end Haley_initial_trees_l1336_133650


namespace segment_length_l1336_133600

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end segment_length_l1336_133600


namespace amount_for_gifts_and_charitable_causes_l1336_133672

namespace JillExpenses

def net_monthly_salary : ℝ := 3700
def discretionary_income : ℝ := 0.20 * net_monthly_salary -- 1/5 * 3700
def vacation_fund : ℝ := 0.30 * discretionary_income
def savings : ℝ := 0.20 * discretionary_income
def eating_out_and_socializing : ℝ := 0.35 * discretionary_income
def gifts_and_charitable_causes : ℝ := discretionary_income - (vacation_fund + savings + eating_out_and_socializing)

theorem amount_for_gifts_and_charitable_causes : gifts_and_charitable_causes = 111 := sorry

end JillExpenses

end amount_for_gifts_and_charitable_causes_l1336_133672


namespace harry_fish_count_l1336_133680

theorem harry_fish_count
  (sam_fish : ℕ) (joe_fish : ℕ) (harry_fish : ℕ)
  (h1 : sam_fish = 7)
  (h2 : joe_fish = 8 * sam_fish)
  (h3 : harry_fish = 4 * joe_fish) :
  harry_fish = 224 :=
by
  sorry

end harry_fish_count_l1336_133680


namespace vinny_final_weight_l1336_133627

theorem vinny_final_weight :
  let initial_weight := 300
  let first_month_loss := 20
  let second_month_loss := first_month_loss / 2
  let third_month_loss := second_month_loss / 2
  let fourth_month_loss := third_month_loss / 2
  let fifth_month_loss := 12
  let total_loss := first_month_loss + second_month_loss + third_month_loss + fourth_month_loss + fifth_month_loss
  let final_weight := initial_weight - total_loss
  final_weight = 250.5 :=
by
  sorry

end vinny_final_weight_l1336_133627


namespace function_monotonically_increasing_l1336_133673

-- The function y = x^2 - 2x + 8
def f (x : ℝ) : ℝ := x^2 - 2 * x + 8

-- The theorem stating the function is monotonically increasing on (1, +∞)
theorem function_monotonically_increasing : ∀ x y : ℝ, (1 < x) → (x < y) → (f x < f y) :=
by
  -- Proof is omitted
  sorry

end function_monotonically_increasing_l1336_133673


namespace wine_problem_l1336_133633

theorem wine_problem (x y : ℕ) (h1 : x + y = 19) (h2 : 3 * x + (1 / 3) * y = 33) : x + y = 19 ∧ 3 * x + (1 / 3) * y = 33 :=
by
  sorry

end wine_problem_l1336_133633


namespace compute_div_square_of_negatives_l1336_133653

theorem compute_div_square_of_negatives : (-128)^2 / (-64)^2 = 4 := by
  sorry

end compute_div_square_of_negatives_l1336_133653


namespace bah_to_yah_conversion_l1336_133606

theorem bah_to_yah_conversion :
  (10 : ℝ) * (1500 * (3/5) * (10/16)) / 16 = 562.5 := by
sorry

end bah_to_yah_conversion_l1336_133606


namespace right_triangle_ratio_l1336_133656

theorem right_triangle_ratio (a b c : ℝ) (h1 : a / b = 3 / 4) (h2 : a^2 + b^2 = c^2) (r s : ℝ) (h3 : r = a^2 / c) (h4 : s = b^2 / c) : 
  r / s = 9 / 16 := by
 sorry

end right_triangle_ratio_l1336_133656


namespace percentage_of_third_number_l1336_133616

theorem percentage_of_third_number (A B C : ℝ) 
  (h1 : A = 0.06 * C) 
  (h2 : B = 0.18 * C) 
  (h3 : A = 0.3333333333333333 * B) : 
  A / C = 0.06 := 
by
  sorry

end percentage_of_third_number_l1336_133616


namespace village_population_l1336_133642

theorem village_population (initial_population: ℕ) (died_percent left_percent: ℕ) (remaining_population current_population: ℕ)
    (h1: initial_population = 6324)
    (h2: died_percent = 10)
    (h3: left_percent = 20)
    (h4: remaining_population = initial_population - (initial_population * died_percent / 100))
    (h5: current_population = remaining_population - (remaining_population * left_percent / 100)):
  current_population = 4554 :=
  by
    sorry

end village_population_l1336_133642


namespace marbles_difference_l1336_133635

theorem marbles_difference {red_marbles blue_marbles : ℕ} 
  (h₁ : red_marbles = 288) (bags_red : ℕ) (h₂ : bags_red = 12) 
  (h₃ : blue_marbles = 243) (bags_blue : ℕ) (h₄ : bags_blue = 9) :
  (blue_marbles / bags_blue) - (red_marbles / bags_red) = 3 :=
by
  sorry

end marbles_difference_l1336_133635


namespace general_term_of_seq_l1336_133682

open Nat

noncomputable def seq (a : ℕ → ℕ) :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 3 * 2^n

theorem general_term_of_seq (a : ℕ → ℕ) :
  seq a → ∀ n, a n = (3 * n - 1) * 2^(n-1) :=
by
  sorry

end general_term_of_seq_l1336_133682


namespace part1_part2_l1336_133645

def f (x : ℝ) : ℝ := |x - 1| + |x + 1| - 2

theorem part1 (x : ℝ) : f x ≥ 1 ↔ (x ≤ -5/2 ∨ x ≥ 3/2) :=
sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x ≥ a^2 - a - 2) ↔ (-1 ≤ a ∧ a ≤ 2) :=
sorry

end part1_part2_l1336_133645


namespace number_of_adults_l1336_133629

theorem number_of_adults (total_apples : ℕ) (children : ℕ) (apples_per_child : ℕ) (apples_per_adult : ℕ) (h : total_apples = 450) (h1 : children = 33) (h2 : apples_per_child = 10) (h3 : apples_per_adult = 3) :
  total_apples - (children * apples_per_child) = 120 →
  (total_apples - (children * apples_per_child)) / apples_per_adult = 40 :=
by
  intros
  sorry

end number_of_adults_l1336_133629


namespace sum_of_possible_values_of_x_l1336_133644

theorem sum_of_possible_values_of_x (x : ℝ) (h : (x + 3) * (x - 4) = 22) : ∃ (x1 x2 : ℝ), x^2 - x - 34 = 0 ∧ x1 + x2 = 1 :=
by
  sorry

end sum_of_possible_values_of_x_l1336_133644


namespace proof_problem_l1336_133630

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3, 4}

-- Define the set M
def M : Set Nat := {2, 4}

-- Define the set N
def N : Set Nat := {0, 4}

-- Define the union of sets M and N
def M_union_N : Set Nat := M ∪ N

-- Define the complement of M ∪ N in U
def complement_U (s : Set Nat) : Set Nat := U \ s

-- State the theorem
theorem proof_problem : complement_U M_union_N = {1, 3} := by
  sorry

end proof_problem_l1336_133630


namespace crayons_count_l1336_133697

theorem crayons_count 
  (initial_crayons erasers : ℕ) 
  (erasers_count end_crayons : ℕ) 
  (initial_erasers : erasers = 38) 
  (end_crayons_more_erasers : end_crayons = erasers + 353) : 
  initial_crayons = end_crayons := 
by 
  sorry

end crayons_count_l1336_133697


namespace xy_equals_nine_l1336_133643

theorem xy_equals_nine (x y : ℝ) (h : (|x + 3| > 0 ∧ (y - 2)^2 = 0) ∨ (|x + 3| = 0 ∧ (y - 2)^2 > 0)) : x^y = 9 :=
sorry

end xy_equals_nine_l1336_133643


namespace range_of_g_l1336_133625

noncomputable def g (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_g : Set.Icc (-1.1071) 1.1071 = Set.image g (Set.Icc (-1:ℝ) 1) := by
  sorry

end range_of_g_l1336_133625


namespace correct_statement_l1336_133623

theorem correct_statement : -3 > -5 := 
by {
  sorry
}

end correct_statement_l1336_133623


namespace sequence_k_value_l1336_133641

theorem sequence_k_value {k : ℕ} (h : 9 < (2 * k - 8) ∧ (2 * k - 8) < 12) 
  (Sn : ℕ → ℤ) (hSn : ∀ n, Sn n = n^2 - 7*n) 
  : k = 9 :=
by
  sorry

end sequence_k_value_l1336_133641


namespace journey_distance_l1336_133634

theorem journey_distance :
  ∃ D T : ℝ,
    D = 100 * T ∧
    D = 80 * (T + 1/3) ∧
    D = 400 / 3 :=
by
  sorry

end journey_distance_l1336_133634


namespace fraction_numerator_l1336_133693

theorem fraction_numerator (x : ℚ) : 
  (∃ y : ℚ, y = 4 * x + 4 ∧ x / y = 3 / 7) → x = -12 / 5 :=
by
  sorry

end fraction_numerator_l1336_133693


namespace defective_probability_l1336_133647

theorem defective_probability {total_switches checked_switches defective_checked : ℕ}
  (h1 : total_switches = 2000)
  (h2 : checked_switches = 100)
  (h3 : defective_checked = 10) :
  (defective_checked : ℚ) / checked_switches = 1 / 10 :=
sorry

end defective_probability_l1336_133647


namespace car_B_speed_90_l1336_133679

def car_speed_problem (distance : ℝ) (ratio_A : ℕ) (ratio_B : ℕ) (time_minutes : ℝ) : Prop :=
  let x := distance / (ratio_A + ratio_B) * (60 / time_minutes)
  (ratio_B * x = 90)

theorem car_B_speed_90 
  (distance : ℝ := 88)
  (ratio_A : ℕ := 5)
  (ratio_B : ℕ := 6)
  (time_minutes : ℝ := 32)
  : car_speed_problem distance ratio_A ratio_B time_minutes :=
by
  sorry

end car_B_speed_90_l1336_133679


namespace rewrite_subtraction_rewrite_division_l1336_133611

theorem rewrite_subtraction : -8 - 5 = -8 + (-5) :=
by sorry

theorem rewrite_division : (1/2) / (-2) = (1/2) * (-1/2) :=
by sorry

end rewrite_subtraction_rewrite_division_l1336_133611


namespace charles_picked_50_pears_l1336_133614

variable (P B S : ℕ)

theorem charles_picked_50_pears 
  (cond1 : S = B + 10)
  (cond2 : B = 3 * P)
  (cond3 : S = 160) : 
  P = 50 := by
  sorry

end charles_picked_50_pears_l1336_133614


namespace sin_double_angle_given_condition_l1336_133670

open Real

variable (x : ℝ)

theorem sin_double_angle_given_condition :
  sin (π / 4 - x) = 3 / 5 → sin (2 * x) = 7 / 25 :=
by
  intro h
  sorry

end sin_double_angle_given_condition_l1336_133670


namespace proof_equivalence_l1336_133692

noncomputable def compute_expression (N : ℕ) (M : ℕ) : ℚ :=
  ((N - 3)^3 + (N - 2)^3 + (N - 1)^3 + N^3 + (N + 1)^3 + (N + 2)^3 + (N + 3)^3) /
  ((M - 3) * (M - 2) + (M - 1) * M + M * (M + 1) + (M + 2) * (M + 3))

theorem proof_equivalence:
  let N := 65536
  let M := 32768
  compute_expression N M = 229376 := 
  by
    sorry

end proof_equivalence_l1336_133692


namespace base_8_addition_l1336_133603

-- Definitions
def five_base_8 : ℕ := 5
def thirteen_base_8 : ℕ := 1 * 8 + 3 -- equivalent of (13)_8 in base 10

-- Theorem statement
theorem base_8_addition :
  (five_base_8 + thirteen_base_8) = 2 * 8 + 0 :=
sorry

end base_8_addition_l1336_133603


namespace fraction_equivalence_l1336_133620

variable {m n p q : ℚ}

theorem fraction_equivalence
  (h1 : m / n = 20)
  (h2 : p / n = 4)
  (h3 : p / q = 1 / 5) :
  m / q = 1 :=
by {
  sorry
}

end fraction_equivalence_l1336_133620


namespace numberOfBoys_is_50_l1336_133686

-- Define the number of boys and the conditions given.
def numberOfBoys (B G : ℕ) : Prop :=
  B / G = 5 / 13 ∧ G = B + 80

-- The theorem that we need to prove.
theorem numberOfBoys_is_50 (B G : ℕ) (h : numberOfBoys B G) : B = 50 :=
  sorry

end numberOfBoys_is_50_l1336_133686


namespace number_of_blue_balloons_l1336_133688

def total_balloons : ℕ := 37
def red_balloons : ℕ := 14
def green_balloons : ℕ := 10

theorem number_of_blue_balloons : (total_balloons - red_balloons - green_balloons) = 13 := 
by
  -- Placeholder for the proof
  sorry

end number_of_blue_balloons_l1336_133688


namespace quadratic_function_range_l1336_133667

theorem quadratic_function_range (x : ℝ) (h : x ≥ 0) : 
  3 ≤ x^2 + 2 * x + 3 :=
by {
  sorry
}

end quadratic_function_range_l1336_133667


namespace sum_of_other_endpoint_coordinates_l1336_133613

theorem sum_of_other_endpoint_coordinates
  (x₁ y₁ x₂ y₂ : ℝ)
  (hx : (x₁ + x₂) / 2 = 5)
  (hy : (y₁ + y₂) / 2 = -8)
  (endpt1 : x₁ = 7)
  (endpt2 : y₁ = -2) :
  x₂ + y₂ = -11 :=
sorry

end sum_of_other_endpoint_coordinates_l1336_133613
