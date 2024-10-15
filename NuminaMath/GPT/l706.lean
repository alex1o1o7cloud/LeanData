import Mathlib

namespace NUMINAMATH_GPT_complement_intersection_l706_70692

-- Conditions
def U : Set Int := {-1, 0, 1, 2, 3}
def A : Set Int := {-1, 0}
def B : Set Int := {0, 1, 2}

-- Theorem statement (proof not included)
theorem complement_intersection :
  let C_UA : Set Int := U \ A
  (C_UA ∩ B) = {1, 2} := 
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l706_70692


namespace NUMINAMATH_GPT_arithmetic_progression_product_l706_70634

theorem arithmetic_progression_product (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ (b : ℕ), (a * (a + d) * (a + 2 * d) * (a + 3 * d) * (a + 4 * d) = b ^ 2008) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_progression_product_l706_70634


namespace NUMINAMATH_GPT_total_wood_gathered_l706_70683

def pieces_per_sack := 20
def number_of_sacks := 4

theorem total_wood_gathered : pieces_per_sack * number_of_sacks = 80 := 
by 
  sorry

end NUMINAMATH_GPT_total_wood_gathered_l706_70683


namespace NUMINAMATH_GPT_most_likely_units_digit_is_5_l706_70678

-- Define the problem conditions
def in_range (n : ℕ) := 1 ≤ n ∧ n ≤ 8
def Jack_pick (J : ℕ) := in_range J
def Jill_pick (J K : ℕ) := in_range K ∧ J ≠ K

-- Define the function to get the units digit of the sum
def units_digit (J K : ℕ) := (J + K) % 10

-- Define the proposition stating the most likely units digit is 5
theorem most_likely_units_digit_is_5 :
  ∃ (d : ℕ), d = 5 ∧
    (∃ (J K : ℕ), Jack_pick J → Jill_pick J K → units_digit J K = d) :=
sorry

end NUMINAMATH_GPT_most_likely_units_digit_is_5_l706_70678


namespace NUMINAMATH_GPT_least_n_condition_l706_70689

theorem least_n_condition (n : ℕ) (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n + 1 → (k ∣ n * (n - 1) → k ≠ n + 1)) : n = 4 :=
sorry

end NUMINAMATH_GPT_least_n_condition_l706_70689


namespace NUMINAMATH_GPT_tennis_tournament_l706_70635

noncomputable def tennis_tournament_n (k : ℕ) : ℕ := 8 * k + 1

theorem tennis_tournament (n : ℕ) :
  (∃ k : ℕ, n = tennis_tournament_n k) ↔
  (∃ k : ℕ, n = 8 * k + 1) :=
by sorry

end NUMINAMATH_GPT_tennis_tournament_l706_70635


namespace NUMINAMATH_GPT_find_a_l706_70639

theorem find_a (a : ℝ) :
  (∃! x : ℝ, (a^2 - 1) * x^2 + (a + 1) * x + 1 = 0) ↔ a = 1 ∨ a = 5/3 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l706_70639


namespace NUMINAMATH_GPT_range_of_a_zero_value_of_a_minimum_l706_70605

noncomputable def f (x a : ℝ) : ℝ := Real.log x + (7 * a) / x

-- Problem 1: Range of a where f(x) has exactly one zero in its domain
theorem range_of_a_zero (a : ℝ) : 
  (∃! x : ℝ, (0 < x) ∧ f x a = 0) ↔ (a ∈ Set.Iic 0 ∪ {1 / (7 * Real.exp 1)}) := sorry

-- Problem 2: Value of a such that the minimum value of f(x) on [e, e^2] is 3
theorem value_of_a_minimum (a : ℝ) : 
  (∃ x : ℝ, (Real.exp 1 ≤ x ∧ x ≤ Real.exp 2) ∧ f x a = 3) ↔ (a = (Real.exp 2)^2 / 7) := sorry

end NUMINAMATH_GPT_range_of_a_zero_value_of_a_minimum_l706_70605


namespace NUMINAMATH_GPT_total_team_cost_correct_l706_70644

variable (jerseyCost shortsCost socksCost cleatsCost waterBottleCost : ℝ)
variable (numPlayers : ℕ)
variable (discountThreshold discountRate salesTaxRate : ℝ)

noncomputable def totalTeamCost : ℝ :=
  let totalCostPerPlayer := jerseyCost + shortsCost + socksCost + cleatsCost + waterBottleCost
  let totalCost := totalCostPerPlayer * numPlayers
  let discount := if totalCost > discountThreshold then totalCost * discountRate else 0
  let discountedTotal := totalCost - discount
  let tax := discountedTotal * salesTaxRate
  let finalCost := discountedTotal + tax
  finalCost

theorem total_team_cost_correct :
  totalTeamCost 25 15.20 6.80 40 12 25 500 0.10 0.07 = 2383.43 := by
  sorry

end NUMINAMATH_GPT_total_team_cost_correct_l706_70644


namespace NUMINAMATH_GPT_c_divisible_by_a_l706_70603

theorem c_divisible_by_a {a b c : ℤ} (h1 : a ∣ b * c) (h2 : Int.gcd a b = 1) : a ∣ c :=
by
  sorry

end NUMINAMATH_GPT_c_divisible_by_a_l706_70603


namespace NUMINAMATH_GPT_best_model_is_model4_l706_70668

-- Define the R^2 values for each model
def R_squared_model1 : ℝ := 0.25
def R_squared_model2 : ℝ := 0.80
def R_squared_model3 : ℝ := 0.50
def R_squared_model4 : ℝ := 0.98

-- Define the highest R^2 value and which model it belongs to
theorem best_model_is_model4 (R1 R2 R3 R4 : ℝ) (h1 : R1 = R_squared_model1) (h2 : R2 = R_squared_model2) (h3 : R3 = R_squared_model3) (h4 : R4 = R_squared_model4) : 
  (R4 = 0.98) ∧ (R4 > R1) ∧ (R4 > R2) ∧ (R4 > R3) :=
by
  sorry

end NUMINAMATH_GPT_best_model_is_model4_l706_70668


namespace NUMINAMATH_GPT_zoe_candy_bars_needed_l706_70641

def total_cost : ℝ := 485
def grandma_contribution : ℝ := 250
def per_candy_earning : ℝ := 1.25
def required_candy_bars : ℕ := 188

theorem zoe_candy_bars_needed :
  (total_cost - grandma_contribution) / per_candy_earning = required_candy_bars :=
by
  sorry

end NUMINAMATH_GPT_zoe_candy_bars_needed_l706_70641


namespace NUMINAMATH_GPT_cards_dealt_to_people_l706_70681

theorem cards_dealt_to_people (total_cards : ℕ) (total_people : ℕ) (h1 : total_cards = 60) (h2 : total_people = 9) :
  (∃ k, k = total_people - (total_cards % total_people) ∧ k = 3) := 
by
  sorry

end NUMINAMATH_GPT_cards_dealt_to_people_l706_70681


namespace NUMINAMATH_GPT_steve_speed_on_way_back_l706_70622

-- Let's define the variables and constants used in the problem.
def distance_to_work : ℝ := 30 -- in km
def total_time_on_road : ℝ := 6 -- in hours
def back_speed_ratio : ℝ := 2 -- Steve drives twice as fast on the way back

theorem steve_speed_on_way_back :
  ∃ v : ℝ, v > 0 ∧ (30 / v + 15 / v = 6) ∧ (2 * v = 15) := by
  sorry

end NUMINAMATH_GPT_steve_speed_on_way_back_l706_70622


namespace NUMINAMATH_GPT_perfect_square_append_100_digits_l706_70680

-- Define the number X consisting of 99 nines

def X : ℕ := (10^99 - 1)

theorem perfect_square_append_100_digits :
  ∃ n : ℕ, X * 10^100 ≤ n^2 ∧ n^2 < X * 10^100 + 10^100 :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_append_100_digits_l706_70680


namespace NUMINAMATH_GPT_volume_of_blue_tetrahedron_in_cube_l706_70697

theorem volume_of_blue_tetrahedron_in_cube (side_length : ℝ) (h : side_length = 8) :
  let cube_volume := side_length^3
  let tetrahedra_volume := 4 * (1/3 * (1/2 * side_length * side_length) * side_length)
  cube_volume - tetrahedra_volume = 512/3 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_blue_tetrahedron_in_cube_l706_70697


namespace NUMINAMATH_GPT_remaining_hair_length_is_1_l706_70611

-- Variables to represent the inches of hair
variable (initial_length cut_length : ℕ)

-- Given initial length and cut length
def initial_length_is_14 (initial_length : ℕ) := initial_length = 14
def cut_length_is_13 (cut_length : ℕ) := cut_length = 13

-- Definition of the remaining hair length
def remaining_length (initial_length cut_length : ℕ) := initial_length - cut_length

-- Main theorem: Proving the remaining hair length is 1 inch
theorem remaining_hair_length_is_1 : initial_length_is_14 initial_length → cut_length_is_13 cut_length → remaining_length initial_length cut_length = 1 := by
  intros h1 h2
  rw [initial_length_is_14, cut_length_is_13] at *
  simp [remaining_length]
  sorry

end NUMINAMATH_GPT_remaining_hair_length_is_1_l706_70611


namespace NUMINAMATH_GPT_classA_classC_ratio_l706_70600

-- Defining the sizes of classes B and C as given in conditions
def classB_size : ℕ := 20
def classC_size : ℕ := 120

-- Defining the size of class A based on the condition that it is twice as big as class B
def classA_size : ℕ := 2 * classB_size

-- Theorem to prove that the ratio of the size of class A to class C is 1:3
theorem classA_classC_ratio : classA_size / classC_size = 1 / 3 := 
sorry

end NUMINAMATH_GPT_classA_classC_ratio_l706_70600


namespace NUMINAMATH_GPT_ratio_of_sides_of_rectangles_l706_70602

theorem ratio_of_sides_of_rectangles (s x y : ℝ) 
  (hsx : x + s = 2 * s) 
  (hsy : s + 2 * y = 2 * s)
  (houter_inner_area : (2 * s) ^ 2 = 4 * s ^ 2) : 
  x / y = 2 :=
by
  -- Assuming the conditions hold, we are interested in proving that the ratio x / y = 2
  -- The proof will be provided here
  sorry

end NUMINAMATH_GPT_ratio_of_sides_of_rectangles_l706_70602


namespace NUMINAMATH_GPT_john_bought_3_reels_l706_70693

theorem john_bought_3_reels (reel_length section_length : ℕ) (n_sections : ℕ)
  (h1 : reel_length = 100) (h2 : section_length = 10) (h3 : n_sections = 30) :
  n_sections * section_length / reel_length = 3 :=
by
  sorry

end NUMINAMATH_GPT_john_bought_3_reels_l706_70693


namespace NUMINAMATH_GPT_positive_value_of_A_l706_70674

theorem positive_value_of_A (A : ℝ) :
  (A ^ 2 + 7 ^ 2 = 200) → A = Real.sqrt 151 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_positive_value_of_A_l706_70674


namespace NUMINAMATH_GPT_find_a_10_l706_70665

-- We define the arithmetic sequence and sum properties
def arithmetic_seq (a_1 d : ℚ) (a_n : ℕ → ℚ) :=
  ∀ n, a_n n = a_1 + d * n

def sum_arithmetic_seq (a : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  ∀ n, S_n n = n * (a 1 + a n) / 2

-- Conditions given in the problem
def given_conditions (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) :=
  arithmetic_seq a_1 1 a_n ∧ sum_arithmetic_seq a_n S_n ∧ S_n 6 = 4 * S_n 3

-- The theorem to prove
theorem find_a_10 (a_1 : ℚ) (a_n : ℕ → ℚ) (S_n : ℕ → ℚ) 
  (h : given_conditions a_1 a_n S_n) : a_n 10 = 19 / 2 :=
by sorry

end NUMINAMATH_GPT_find_a_10_l706_70665


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l706_70661

-- Define the hyperbola and the condition of the asymptote passing through (2,1)
def hyperbola (a b : ℝ) : Prop := 
  ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1 ∧
               (a ≠ 0 ∧ b ≠ 0) ∧
               (x, y) = (2, 1)

-- Define the eccentricity of the hyperbola
def eccentricity (a b e : ℝ) : Prop :=
  a^2 + b^2 = (b * e)^2

theorem hyperbola_eccentricity (a b e : ℝ) 
  (hx : hyperbola a b)
  (ha : a = 2 * b)
  (ggt: (a^2 = 4 * b^2)) :
  eccentricity a b e → e = (Real.sqrt 5) / 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l706_70661


namespace NUMINAMATH_GPT_roots_of_cubic_eq_l706_70637

theorem roots_of_cubic_eq (r s t p q : ℝ) (h1 : r + s + t = p) (h2 : r * s + r * t + s * t = q) 
(h3 : r * s * t = r) : r^2 + s^2 + t^2 = p^2 - 2 * q := 
by 
  sorry

end NUMINAMATH_GPT_roots_of_cubic_eq_l706_70637


namespace NUMINAMATH_GPT_lines_intersect_l706_70699

-- Define the coefficients of the lines
def A1 : ℝ := 3
def B1 : ℝ := -2
def C1 : ℝ := 5

def A2 : ℝ := 1
def B2 : ℝ := 3
def C2 : ℝ := 10

-- Define the equations of the lines
def line1 (x y : ℝ) : Prop := A1 * x + B1 * y + C1 = 0
def line2 (x y : ℝ) : Prop := A2 * x + B2 * y + C2 = 0

-- Mathematical problem to prove
theorem lines_intersect : ∃ (x y : ℝ), line1 x y ∧ line2 x y :=
by
  sorry

end NUMINAMATH_GPT_lines_intersect_l706_70699


namespace NUMINAMATH_GPT_largest_possible_value_of_s_l706_70646

theorem largest_possible_value_of_s (r s : Nat) (h1 : r ≥ s) (h2 : s ≥ 3)
  (h3 : (r - 2) * s * 61 = (s - 2) * r * 60) : s = 121 :=
sorry

end NUMINAMATH_GPT_largest_possible_value_of_s_l706_70646


namespace NUMINAMATH_GPT_T_sum_correct_l706_70643

-- Defining the sequence T_n
def T (n : ℕ) : ℤ := 
(-1)^n * 2 * n + (-1)^(n + 1) * n

-- Values to compute
def n1 : ℕ := 27
def n2 : ℕ := 43
def n3 : ℕ := 60

-- Sum of particular values
def T_sum : ℤ := T n1 + T n2 + T n3

-- Placeholder value until actual calculation
def expected_sum : ℤ := -42 -- Replace with the correct calculated result

theorem T_sum_correct : T_sum = expected_sum := sorry

end NUMINAMATH_GPT_T_sum_correct_l706_70643


namespace NUMINAMATH_GPT_probability_r25_to_r35_l706_70607

theorem probability_r25_to_r35 (n : ℕ) (r : Fin n → ℕ) (h : n = 50) 
  (distinct : ∀ i j : Fin n, i ≠ j → r i ≠ r j) : 1 + 1260 = 1261 :=
by
  sorry

end NUMINAMATH_GPT_probability_r25_to_r35_l706_70607


namespace NUMINAMATH_GPT_ted_worked_hours_l706_70647

variable (t : ℝ)
variable (julie_rate ted_rate combined_rate : ℝ)
variable (julie_alone_time : ℝ)
variable (job_done : ℝ)

theorem ted_worked_hours :
  julie_rate = 1 / 10 →
  ted_rate = 1 / 8 →
  combined_rate = julie_rate + ted_rate →
  julie_alone_time = 0.9999999999999998 →
  job_done = combined_rate * t + julie_rate * julie_alone_time →
  t = 4 :=
by
  sorry

end NUMINAMATH_GPT_ted_worked_hours_l706_70647


namespace NUMINAMATH_GPT_inequality_proof_l706_70601

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : b + d < a + c :=
sorry

end NUMINAMATH_GPT_inequality_proof_l706_70601


namespace NUMINAMATH_GPT_largest_4_digit_congruent_15_mod_22_l706_70616

theorem largest_4_digit_congruent_15_mod_22 :
  ∃ (x : ℤ), x < 10000 ∧ x % 22 = 15 ∧ (∀ (y : ℤ), y < 10000 ∧ y % 22 = 15 → y ≤ x) → x = 9981 :=
sorry

end NUMINAMATH_GPT_largest_4_digit_congruent_15_mod_22_l706_70616


namespace NUMINAMATH_GPT_smallest_number_of_eggs_l706_70633

-- Define the conditions given in the problem
def total_containers (c : ℕ) : ℕ := 15 * c - 3

-- Prove that given the conditions, the smallest number of eggs you could have is 162
theorem smallest_number_of_eggs (h : ∃ c : ℕ, total_containers c > 150) : ∃ c : ℕ, total_containers c = 162 :=
by
  sorry

end NUMINAMATH_GPT_smallest_number_of_eggs_l706_70633


namespace NUMINAMATH_GPT_number_of_integer_solutions_l706_70648

theorem number_of_integer_solutions : ∃ (n : ℕ), n = 120 ∧ ∀ (x y z : ℤ), x * y * z = 2008 → n = 120 :=
by
  sorry

end NUMINAMATH_GPT_number_of_integer_solutions_l706_70648


namespace NUMINAMATH_GPT_certain_number_value_l706_70658

variable {t b c x : ℕ}

theorem certain_number_value 
  (h1 : (t + b + c + 14 + x) / 5 = 12) 
  (h2 : (t + b + c + 29) / 4 = 15) : 
  x = 15 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_value_l706_70658


namespace NUMINAMATH_GPT_height_large_cylinder_is_10_l706_70682

noncomputable def height_large_cylinder : ℝ :=
  let V_small := 13.5 * Real.pi
  let factor := 74.07407407407408
  let V_large := 100 * Real.pi
  factor * V_small / V_large

theorem height_large_cylinder_is_10 :
  height_large_cylinder = 10 :=
by
  sorry

end NUMINAMATH_GPT_height_large_cylinder_is_10_l706_70682


namespace NUMINAMATH_GPT_distance_traveled_l706_70626

def velocity (t : ℝ) : ℝ := t^2 + 1

theorem distance_traveled :
  (∫ t in (0:ℝ)..(3:ℝ), velocity t) = 12 :=
by
  simp [velocity]
  sorry

end NUMINAMATH_GPT_distance_traveled_l706_70626


namespace NUMINAMATH_GPT_no_valid_n_exists_l706_70657

theorem no_valid_n_exists :
  ¬ ∃ n : ℕ, 219 ≤ n ∧ n ≤ 2019 ∧ ∃ x y : ℕ, 
    1 ≤ x ∧ x < n ∧ n < y ∧ (∀ k : ℕ, k ≤ n → k ≠ x ∧ k ≠ x+1 → y % k = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_no_valid_n_exists_l706_70657


namespace NUMINAMATH_GPT_sugar_mixture_problem_l706_70612

theorem sugar_mixture_problem :
  ∃ x : ℝ, (9 * x + 7 * (63 - x) = 0.9 * (9.24 * 63)) ∧ x = 41.724 :=
by
  sorry

end NUMINAMATH_GPT_sugar_mixture_problem_l706_70612


namespace NUMINAMATH_GPT_candy_problem_l706_70694

theorem candy_problem
  (G : Nat := 7) -- Gwen got 7 pounds of candy
  (C : Nat := 17) -- Combined weight of candy
  (F : Nat) -- Pounds of candy Frank got
  (h : F + G = C) -- Condition: Combined weight
  : F = 10 := 
by
  sorry

end NUMINAMATH_GPT_candy_problem_l706_70694


namespace NUMINAMATH_GPT_work_time_relation_l706_70631

theorem work_time_relation (m n k x y z : ℝ) 
    (h1 : 1 / x = m / (y + z)) 
    (h2 : 1 / y = n / (x + z)) 
    (h3 : 1 / z = k / (x + y)) : 
    k = (m + n + 2) / (m * n - 1) :=
by
  sorry

end NUMINAMATH_GPT_work_time_relation_l706_70631


namespace NUMINAMATH_GPT_trip_time_is_approximate_l706_70669

noncomputable def total_distance : ℝ := 620
noncomputable def half_distance : ℝ := total_distance / 2
noncomputable def speed1 : ℝ := 70
noncomputable def speed2 : ℝ := 85
noncomputable def time1 : ℝ := half_distance / speed1
noncomputable def time2 : ℝ := half_distance / speed2
noncomputable def total_time : ℝ := time1 + time2

theorem trip_time_is_approximate :
  abs (total_time - 8.0757) < 0.0001 :=
sorry

end NUMINAMATH_GPT_trip_time_is_approximate_l706_70669


namespace NUMINAMATH_GPT_distance_from_point_to_line_l706_70624

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

def cartesian_distance_to_line (point : ℝ × ℝ) (y_line : ℝ) : ℝ :=
  abs (point.snd - y_line)

theorem distance_from_point_to_line
  (ρ θ : ℝ)
  (h_point : ρ = 2 ∧ θ = Real.pi / 6)
  (h_line : ∀ θ, (3 : ℝ) = ρ * Real.sin θ) :
  cartesian_distance_to_line (polar_to_cartesian ρ θ) 3 = 2 :=
  sorry

end NUMINAMATH_GPT_distance_from_point_to_line_l706_70624


namespace NUMINAMATH_GPT_box_cost_coffee_pods_l706_70676

theorem box_cost_coffee_pods :
  ∀ (days : ℕ) (cups_per_day : ℕ) (pods_per_box : ℕ) (total_cost : ℕ), 
  days = 40 → cups_per_day = 3 → pods_per_box = 30 → total_cost = 32 → 
  total_cost / ((days * cups_per_day) / pods_per_box) = 8 := 
by
  intros days cups_per_day pods_per_box total_cost hday hcup hpod hcost
  sorry

end NUMINAMATH_GPT_box_cost_coffee_pods_l706_70676


namespace NUMINAMATH_GPT_find_number_l706_70630

theorem find_number (N : ℝ) (h : (0.47 * N - 0.36 * 1412) + 66 = 6) : N = 953.87 :=
  sorry

end NUMINAMATH_GPT_find_number_l706_70630


namespace NUMINAMATH_GPT_min_moves_to_reset_counters_l706_70651

theorem min_moves_to_reset_counters (f : Fin 28 -> Nat) (h_initial : ∀ i, 1 ≤ f i ∧ f i ≤ 2017) :
  ∃ k, k = 11 ∧ ∀ g : Fin 28 -> Nat, (∀ i, f i = 0) :=
by
  sorry

end NUMINAMATH_GPT_min_moves_to_reset_counters_l706_70651


namespace NUMINAMATH_GPT_consecutive_integers_sum_l706_70649

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 380) : x + (x + 1) = 39 := by
  sorry

end NUMINAMATH_GPT_consecutive_integers_sum_l706_70649


namespace NUMINAMATH_GPT_find_k_l706_70654

def f (n : ℤ) : ℤ :=
if n % 2 = 0 then n / 2 else n + 3

theorem find_k (k : ℤ) (h_odd : k % 2 = 1) (h_f_f_f_k : f (f (f k)) = 27) : k = 105 := by
  sorry

end NUMINAMATH_GPT_find_k_l706_70654


namespace NUMINAMATH_GPT_coefficient_a6_l706_70621

def expand_equation (x a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 : ℝ) : Prop :=
  x * (x - 2) ^ 8 =
    a0 + a1 * (x - 1) + a2 * (x - 1) ^ 2 + a3 * (x - 1) ^ 3 + a4 * (x - 1) ^ 4 +
    a5 * (x - 1) ^ 5 + a6 * (x - 1) ^ 6 + a7 * (x - 1) ^ 7 + a8 * (x - 1) ^ 8 + 
    a9 * (x - 1) ^ 9

theorem coefficient_a6 (x a0 a1 a2 a3 a4 a5 a7 a8 a9 : ℝ) (h : expand_equation x a0 a1 a2 a3 a4 a5 (-28) a7 a8 a9) :
  a6 = -28 :=
sorry

end NUMINAMATH_GPT_coefficient_a6_l706_70621


namespace NUMINAMATH_GPT_abc_sum_square_identity_l706_70620

theorem abc_sum_square_identity (a b c : ℝ) (h1 : a^2 + b^2 + c^2 = 941) (h2 : a + b + c = 31) :
  ab + bc + ca = 10 :=
by
  sorry

end NUMINAMATH_GPT_abc_sum_square_identity_l706_70620


namespace NUMINAMATH_GPT_remainder_of_n_mod_9_eq_5_l706_70636

-- Definitions of the variables and conditions
variables (a b c n : ℕ)

-- The given conditions as assumptions
def conditions : Prop :=
  a + b + c = 63 ∧
  a = c + 22 ∧
  n = 2 * a + 3 * b + 4 * c

-- The proof statement that needs to be proven
theorem remainder_of_n_mod_9_eq_5 (h : conditions a b c n) : n % 9 = 5 := 
  sorry

end NUMINAMATH_GPT_remainder_of_n_mod_9_eq_5_l706_70636


namespace NUMINAMATH_GPT_domain_of_f_l706_70638

noncomputable def f (x : ℝ) : ℝ := (x^3 + 8) / (x - 8)

theorem domain_of_f : ∀ x : ℝ, x ≠ 8 ↔ ∃ y : ℝ, f x = y :=
  by admit

end NUMINAMATH_GPT_domain_of_f_l706_70638


namespace NUMINAMATH_GPT_find_N_mod_inverse_l706_70650

-- Definitions based on given conditions
def A := 111112
def B := 142858
def M := 1000003
def AB : Nat := (A * B) % M
def N := 513487

-- Statement to prove
theorem find_N_mod_inverse : (711812 * N) % M = 1 := by
  -- Proof skipped as per instruction
  sorry

end NUMINAMATH_GPT_find_N_mod_inverse_l706_70650


namespace NUMINAMATH_GPT_range_of_a_l706_70659

def A : Set ℝ := { x | -2 ≤ x ∧ x ≤ 2 }
def B (a : ℝ) : Set ℝ := { x | x ≥ a }

theorem range_of_a (a : ℝ) (h : A ⊆ B a) : a ≤ -2 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l706_70659


namespace NUMINAMATH_GPT_wire_length_l706_70623

theorem wire_length (S L W : ℝ) (h1 : S = 20) (h2 : S = (2 / 7) * L) (h3 : W = S + L) : W = 90 :=
by sorry

end NUMINAMATH_GPT_wire_length_l706_70623


namespace NUMINAMATH_GPT_solution_set_inequality_l706_70688

theorem solution_set_inequality (x : ℝ) : (x-3) * (x-1) > 0 → (x < 1 ∨ x > 3) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l706_70688


namespace NUMINAMATH_GPT_smaller_number_is_neg_five_l706_70698

theorem smaller_number_is_neg_five (x y : ℤ) (h1 : x + y = 30) (h2 : x - y = 40) : y = -5 :=
by
  sorry

end NUMINAMATH_GPT_smaller_number_is_neg_five_l706_70698


namespace NUMINAMATH_GPT_nathan_ate_total_gumballs_l706_70686

-- Define the constants and variables based on the conditions
def gumballs_small : Nat := 5
def gumballs_medium : Nat := 12
def gumballs_large : Nat := 20
def small_packages : Nat := 4
def medium_packages : Nat := 3
def large_packages : Nat := 2

-- The total number of gumballs Nathan ate
def total_gumballs : Nat := (small_packages * gumballs_small) + (medium_packages * gumballs_medium) + (large_packages * gumballs_large)

-- The theorem to prove
theorem nathan_ate_total_gumballs : total_gumballs = 96 :=
by
  unfold total_gumballs
  sorry

end NUMINAMATH_GPT_nathan_ate_total_gumballs_l706_70686


namespace NUMINAMATH_GPT_billy_boxes_of_candy_l706_70606

theorem billy_boxes_of_candy (pieces_per_box total_pieces : ℕ) (h1 : pieces_per_box = 3) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 7 := 
by
  sorry

end NUMINAMATH_GPT_billy_boxes_of_candy_l706_70606


namespace NUMINAMATH_GPT_solve_inequality_l706_70629

theorem solve_inequality : { x : ℝ | 0 ≤ x^2 - x - 2 ∧ x^2 - x - 2 ≤ 4 } = { x | (-2 ≤ x ∧ x ≤ -1) ∨ (2 ≤ x ∧ x ≤ 3) } :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l706_70629


namespace NUMINAMATH_GPT_calculate_area_ADC_l706_70664

def area_AD (BD DC : ℕ) (area_ABD : ℕ) := 
  area_ABD * DC / BD

theorem calculate_area_ADC
  (BD DC : ℕ) 
  (h_ratio : BD = 5 * DC / 2)
  (area_ABD : ℕ)
  (h_area_ABD : area_ABD = 35) :
  area_AD BD DC area_ABD = 14 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_area_ADC_l706_70664


namespace NUMINAMATH_GPT_meeting_time_l706_70662

-- Define the conditions
def distance : ℕ := 600  -- distance between A and B
def speed_A_to_B : ℕ := 70  -- speed of the first person
def speed_B_to_A : ℕ := 80  -- speed of the second person
def start_time : ℕ := 10  -- start time in hours

-- State the problem formally in Lean 4
theorem meeting_time : (distance / (speed_A_to_B + speed_B_to_A)) + start_time = 14 := 
by
  sorry

end NUMINAMATH_GPT_meeting_time_l706_70662


namespace NUMINAMATH_GPT_seating_arrangements_l706_70615

theorem seating_arrangements {n k : ℕ} (h1 : n = 8) (h2 : k = 6) :
  ∃ c : ℕ, c = (n - 1) * Nat.factorial k ∧ c = 20160 :=
by
  sorry

end NUMINAMATH_GPT_seating_arrangements_l706_70615


namespace NUMINAMATH_GPT_trapezium_height_l706_70684

-- Defining the lengths of the parallel sides and the area of the trapezium
def a : ℝ := 28
def b : ℝ := 18
def area : ℝ := 345

-- Defining the distance between the parallel sides to be proven
def h : ℝ := 15

-- The theorem that proves the distance between the parallel sides
theorem trapezium_height :
  (1 / 2) * (a + b) * h = area :=
by
  sorry

end NUMINAMATH_GPT_trapezium_height_l706_70684


namespace NUMINAMATH_GPT_jorge_acres_l706_70663

theorem jorge_acres (A : ℕ) (H1 : A = 60) 
    (H2 : ∀ acres, acres / 3 = 60 / 3 ∧ 2 * (acres / 3) = 2 * (60 / 3)) 
    (H3 : ∀ good_yield_per_acre, good_yield_per_acre = 400) 
    (H4 : ∀ clay_yield_per_acre, clay_yield_per_acre = 200) 
    (H5 : ∀ total_yield, total_yield = (2 * (A / 3) * 400 + (A / 3) * 200)) 
    : total_yield = 20000 :=
by 
  sorry

end NUMINAMATH_GPT_jorge_acres_l706_70663


namespace NUMINAMATH_GPT_surface_area_of_modified_structure_l706_70696

-- Define the given conditions
def initial_cube_side_length : ℕ := 12
def smaller_cube_side_length : ℕ := 2
def smaller_cubes_count : ℕ := 72
def face_center_cubes_count : ℕ := 6

-- Define the calculation of the surface area
def single_smaller_cube_surface_area : ℕ := 6 * (smaller_cube_side_length ^ 2)
def added_surface_from_removed_center_cube : ℕ := 4 * (smaller_cube_side_length ^ 2)
def modified_smaller_cube_surface_area : ℕ := single_smaller_cube_surface_area + added_surface_from_removed_center_cube
def unaffected_smaller_cubes : ℕ := smaller_cubes_count - face_center_cubes_count

-- Define the given surface area according to the problem
def correct_surface_area : ℕ := 1824

-- The equivalent proof problem statement
theorem surface_area_of_modified_structure : 
    66 * single_smaller_cube_surface_area + 6 * modified_smaller_cube_surface_area = correct_surface_area := 
by
    -- placeholders for the actual proof
    sorry

end NUMINAMATH_GPT_surface_area_of_modified_structure_l706_70696


namespace NUMINAMATH_GPT_mountain_climbing_time_proof_l706_70625

noncomputable def mountain_climbing_time (x : ℝ) : ℝ := (x + 2) / 4

theorem mountain_climbing_time_proof (x : ℝ) (h1 : (x / 3 + (x + 2) / 4 = 4)) : mountain_climbing_time x = 2 := by
  -- assume the given conditions and proof steps explicitly
  sorry

end NUMINAMATH_GPT_mountain_climbing_time_proof_l706_70625


namespace NUMINAMATH_GPT_angle_z_value_l706_70673

theorem angle_z_value
  (ABC BAC : ℝ)
  (h1 : ABC = 70)
  (h2 : BAC = 50)
  (h3 : ∀ BCA : ℝ, BCA + ABC + BAC = 180) :
  ∃ z : ℝ, z = 30 :=
by
  sorry

end NUMINAMATH_GPT_angle_z_value_l706_70673


namespace NUMINAMATH_GPT_custom_op_evaluation_l706_70672

def custom_op (a b : ℝ) : ℝ := 4 * a + 5 * b

theorem custom_op_evaluation : custom_op 4 2 = 26 := 
by 
  sorry

end NUMINAMATH_GPT_custom_op_evaluation_l706_70672


namespace NUMINAMATH_GPT_total_cost_proof_l706_70645

def tuition_fee : ℕ := 1644
def room_and_board_cost : ℕ := tuition_fee - 704
def total_cost : ℕ := tuition_fee + room_and_board_cost

theorem total_cost_proof : total_cost = 2584 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_proof_l706_70645


namespace NUMINAMATH_GPT_cyclic_sum_inequality_l706_70670

theorem cyclic_sum_inequality (a b c : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0) :
  ( (b + c - a)^2 / (a^2 + (b + c)^2) +
    (c + a - b)^2 / (b^2 + (c + a)^2) +
    (a + b - c)^2 / (c^2 + (a + b)^2) ) ≥ 3 / 5 :=
  sorry

end NUMINAMATH_GPT_cyclic_sum_inequality_l706_70670


namespace NUMINAMATH_GPT_no_perfect_power_l706_70677

theorem no_perfect_power (n m : ℕ) (hn : 0 < n) (hm : 1 < m) : 102 ^ 1991 + 103 ^ 1991 ≠ n ^ m := 
sorry

end NUMINAMATH_GPT_no_perfect_power_l706_70677


namespace NUMINAMATH_GPT_sum_of_x_and_y_l706_70687

theorem sum_of_x_and_y (x y : ℕ) (hxpos : 0 < x) (hypos : 1 < y) (hxy : x^y < 500) (hmax : ∀ (a b : ℕ), 0 < a → 1 < b → a^b < 500 → a^b ≤ x^y) : x + y = 24 := 
sorry

end NUMINAMATH_GPT_sum_of_x_and_y_l706_70687


namespace NUMINAMATH_GPT_range_of_a_l706_70653

noncomputable def f (x : ℝ) (a : ℝ) := x * Real.log x + a / x + 3
noncomputable def g (x : ℝ) := x^3 - x^2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc (1/2) 2 → x2 ∈ Set.Icc (1/2) 2 → f x1 a - g x2 ≥ 0) →
  1 ≤ a :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l706_70653


namespace NUMINAMATH_GPT_remainder_of_3n_mod_9_l706_70690

theorem remainder_of_3n_mod_9 (n : ℕ) (h : n % 9 = 7) : (3 * n) % 9 = 3 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_3n_mod_9_l706_70690


namespace NUMINAMATH_GPT_rect_area_162_l706_70618

def rectangle_field_area (w l : ℝ) (A : ℝ) : Prop :=
  w = (1/2) * l ∧ 2 * (w + l) = 54 ∧ A = w * l

theorem rect_area_162 {w l A : ℝ} :
  rectangle_field_area w l A → A = 162 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_rect_area_162_l706_70618


namespace NUMINAMATH_GPT_Emily_spent_28_dollars_l706_70652

theorem Emily_spent_28_dollars :
  let roses_cost := 4
  let daisies_cost := 3
  let tulips_cost := 5
  let lilies_cost := 6
  let roses_qty := 2
  let daisies_qty := 3
  let tulips_qty := 1
  let lilies_qty := 1
  (roses_qty * roses_cost) + (daisies_qty * daisies_cost) + (tulips_qty * tulips_cost) + (lilies_qty * lilies_cost) = 28 :=
by
  sorry

end NUMINAMATH_GPT_Emily_spent_28_dollars_l706_70652


namespace NUMINAMATH_GPT_two_a_sq_minus_six_b_plus_one_l706_70685

theorem two_a_sq_minus_six_b_plus_one (a b : ℝ) (h : a^2 - 3 * b = 5) : 2 * a^2 - 6 * b + 1 = 11 := by
  sorry

end NUMINAMATH_GPT_two_a_sq_minus_six_b_plus_one_l706_70685


namespace NUMINAMATH_GPT_minimum_value_l706_70679

noncomputable def minimum_y_over_2x_plus_1_over_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) : ℝ :=
  (y / (2 * x)) + (1 / y)

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 2 * y = 1) :
  minimum_y_over_2x_plus_1_over_y x y hx hy h = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_l706_70679


namespace NUMINAMATH_GPT_speed_of_river_l706_70617

-- Definitions of the conditions
def rowing_speed_still_water := 9 -- kmph in still water
def total_time := 1 -- hour for a round trip
def total_distance := 8.84 -- km

-- Distance to the place the man rows to
def d := total_distance / 2

-- Problem statement in Lean 4
theorem speed_of_river (v : ℝ) : 
  rowing_speed_still_water = 9 ∧
  total_time = 1 ∧
  total_distance = 8.84 →
  (4.42 / (rowing_speed_still_water + v) + 4.42 / (rowing_speed_still_water - v) = 1) →
  v = 1.2 := 
by
  sorry

end NUMINAMATH_GPT_speed_of_river_l706_70617


namespace NUMINAMATH_GPT_rectangular_field_area_l706_70691

theorem rectangular_field_area (w l : ℝ) (h1 : l = 3 * w) (h2 : 2 * (w + l) = 72) :
  w * l = 243 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_rectangular_field_area_l706_70691


namespace NUMINAMATH_GPT_pq_sum_eight_l706_70609

theorem pq_sum_eight
  (p q : ℤ)
  (hp1 : p > 1)
  (hq1 : q > 1)
  (hs1 : (2 * q - 1) % p = 0)
  (hs2 : (2 * p - 1) % q = 0) : p + q = 8 := 
sorry

end NUMINAMATH_GPT_pq_sum_eight_l706_70609


namespace NUMINAMATH_GPT_exists_square_divisible_by_12_between_100_and_200_l706_70632

theorem exists_square_divisible_by_12_between_100_and_200 : 
  ∃ x : ℕ, (∃ y : ℕ, x = y * y) ∧ (12 ∣ x) ∧ (100 ≤ x ∧ x ≤ 200) ∧ x = 144 :=
by
  sorry

end NUMINAMATH_GPT_exists_square_divisible_by_12_between_100_and_200_l706_70632


namespace NUMINAMATH_GPT_total_apartment_units_l706_70610

-- Define the number of apartment units on different floors
def units_first_floor := 2
def units_other_floors := 5
def num_other_floors := 3
def num_buildings := 2

-- Calculation of total units in one building
def units_one_building := units_first_floor + num_other_floors * units_other_floors

-- Calculation of total units in all buildings
def total_units := num_buildings * units_one_building

-- The theorem to prove
theorem total_apartment_units : total_units = 34 :=
by
  sorry

end NUMINAMATH_GPT_total_apartment_units_l706_70610


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_eq_6_l706_70655

variable {a_n : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem arithmetic_sequence_a5_eq_6 (h_arith : is_arithmetic_sequence a_n) (h_sum : a_n 2 + a_n 8 = 12) : a_n 5 = 6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_eq_6_l706_70655


namespace NUMINAMATH_GPT_problem_statement_l706_70675

-- Define the problem
theorem problem_statement (a b : ℝ) (h : a - b = 1 / 2) : -3 * (b - a) = 3 / 2 := 
  sorry

end NUMINAMATH_GPT_problem_statement_l706_70675


namespace NUMINAMATH_GPT_total_fruits_picked_l706_70604

theorem total_fruits_picked (g_oranges g_apples a_oranges a_apples o_oranges o_apples : ℕ) :
  g_oranges = 45 →
  g_apples = a_apples + 5 →
  a_oranges = g_oranges - 18 →
  a_apples = 15 →
  o_oranges = 6 * 3 →
  o_apples = 6 * 2 →
  g_oranges + g_apples + a_oranges + a_apples + o_oranges + o_apples = 137 :=
by
  intros
  sorry

end NUMINAMATH_GPT_total_fruits_picked_l706_70604


namespace NUMINAMATH_GPT_complementary_angle_ratio_l706_70642

noncomputable def smaller_angle_measure (x : ℝ) : ℝ := 
  3 * (90 / 7)

theorem complementary_angle_ratio :
  ∀ (A B : ℝ), (B = 4 * (90 / 7)) → (A = 3 * (90 / 7)) → 
  (A + B = 90) → A = 38.57142857142857 :=
by
  intros A B hB hA hSum
  sorry

end NUMINAMATH_GPT_complementary_angle_ratio_l706_70642


namespace NUMINAMATH_GPT_max_cubes_fit_in_box_l706_70614

theorem max_cubes_fit_in_box :
  ∀ (h w l : ℕ) (cube_vol box_max_cubes : ℕ),
    h = 12 → w = 8 → l = 9 → cube_vol = 27 → 
    box_max_cubes = (h * w * l) / cube_vol → box_max_cubes = 32 :=
by
  intros h w l cube_vol box_max_cubes h_def w_def l_def cube_vol_def box_max_cubes_def
  sorry

end NUMINAMATH_GPT_max_cubes_fit_in_box_l706_70614


namespace NUMINAMATH_GPT_line_intersection_l706_70666

-- Definitions for the parametric lines
def line1 (t : ℝ) : ℝ × ℝ := (3 + t, 2 * t)
def line2 (u : ℝ) : ℝ × ℝ := (-1 + 3 * u, 4 - u)

-- Statement that expresses the intersection point condition
theorem line_intersection :
  ∃ t u : ℝ, line1 t = line2 u ∧ line1 t = (30 / 7, 18 / 7) :=
by
  sorry

end NUMINAMATH_GPT_line_intersection_l706_70666


namespace NUMINAMATH_GPT_car_A_overtakes_car_B_l706_70695

theorem car_A_overtakes_car_B (z : ℕ) :
  let y := (5 * z) / 4
  let x := (13 * z) / 10
  10 * y / (x - y) = 250 := 
by
  sorry

end NUMINAMATH_GPT_car_A_overtakes_car_B_l706_70695


namespace NUMINAMATH_GPT_increase_in_votes_l706_70613

noncomputable def initial_vote_for (y : ℝ) : ℝ := 500 - y
noncomputable def revote_for (y : ℝ) : ℝ := (10 / 9) * y

theorem increase_in_votes {x x' y m : ℝ}
  (H1 : x + y = 500)
  (H2 : y - x = m)
  (H3 : x' - y = 2 * m)
  (H4 : x' + y = 500)
  (H5 : x' = (10 / 9) * y)
  (H6 : y = 282) :
  revote_for y - initial_vote_for y = 95 :=
by sorry

end NUMINAMATH_GPT_increase_in_votes_l706_70613


namespace NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_l706_70627

theorem smallest_four_digit_multiple_of_17 : ∃ n, n ≥ 1000 ∧ n < 10000 ∧ 17 ∣ n ∧ ∀ m, m ≥ 1000 ∧ m < 10000 ∧ 17 ∣ m → n ≤ m := 
by
  use 1003
  sorry

end NUMINAMATH_GPT_smallest_four_digit_multiple_of_17_l706_70627


namespace NUMINAMATH_GPT_greatest_of_three_consecutive_integers_with_sum_21_l706_70628

theorem greatest_of_three_consecutive_integers_with_sum_21 :
  ∃ (x : ℤ), (x + (x + 1) + (x + 2) = 21) ∧ ((x + 2) = 8) :=
by
  sorry

end NUMINAMATH_GPT_greatest_of_three_consecutive_integers_with_sum_21_l706_70628


namespace NUMINAMATH_GPT_arithmetic_sequence_a6_value_l706_70640

theorem arithmetic_sequence_a6_value
  (a : ℕ → ℤ)
  (d : ℤ)
  (h_arithmetic : ∀ n, a (n + 1) = a n + d)
  (h_sum : a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 14) :
  a 6 = 2 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_a6_value_l706_70640


namespace NUMINAMATH_GPT_sufficient_not_necessary_l706_70619

theorem sufficient_not_necessary (a : ℝ) (h : a ≠ 0) : 
  (a > 1 → a > 1 / a) ∧ (¬ (a > 1) → a > 1 / a → -1 < a ∧ a < 0) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l706_70619


namespace NUMINAMATH_GPT_max_and_next_max_values_l706_70608

noncomputable def log_base (a b : ℝ) : ℝ := (Real.log a) / b

theorem max_and_next_max_values :
  let values := [4.0^(1/4), 5.0^(1/5), 16.0^(1/16), 25.0^(1/25)]
  ∃ max2 max1, 
    max1 = 4.0^(1/4) ∧ max2 = 5.0^(1/5) ∧ 
    (∀ x ∈ values, x <= max1) ∧ 
    (∀ x ∈ values, x < max1 → x <= max2) :=
by
  sorry

end NUMINAMATH_GPT_max_and_next_max_values_l706_70608


namespace NUMINAMATH_GPT_length_sawed_off_l706_70667

-- Define the lengths as constants
def original_length : ℝ := 8.9
def final_length : ℝ := 6.6

-- State the property to be proven
theorem length_sawed_off : original_length - final_length = 2.3 := by
  sorry

end NUMINAMATH_GPT_length_sawed_off_l706_70667


namespace NUMINAMATH_GPT_find_length_of_train_l706_70671

noncomputable def speed_kmhr : ℝ := 30
noncomputable def time_seconds : ℝ := 9
noncomputable def conversion_factor : ℝ := 5 / 18
noncomputable def speed_ms : ℝ := speed_kmhr * conversion_factor
noncomputable def length_train : ℝ := speed_ms * time_seconds

theorem find_length_of_train : length_train = 74.97 := 
by
  sorry

end NUMINAMATH_GPT_find_length_of_train_l706_70671


namespace NUMINAMATH_GPT_angle_sum_straight_line_l706_70660

  theorem angle_sum_straight_line (x : ℝ) (h : 90 + x + 20 = 180) : x = 70 :=
  by
    sorry
  
end NUMINAMATH_GPT_angle_sum_straight_line_l706_70660


namespace NUMINAMATH_GPT_matrix_B_pow48_l706_70656

open Matrix

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 0, 0], ![0, 0, 2], ![0, -2, 0]]

theorem matrix_B_pow48 :
  B ^ 48 = ![![0, 0, 0], ![0, 16^12, 0], ![0, 0, 16^12]] :=
by sorry

end NUMINAMATH_GPT_matrix_B_pow48_l706_70656
