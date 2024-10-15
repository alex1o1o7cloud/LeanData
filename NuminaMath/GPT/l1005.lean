import Mathlib

namespace NUMINAMATH_GPT_C_pays_228_for_cricket_bat_l1005_100535

def CostPriceA : ℝ := 152

def ProfitA (price : ℝ) : ℝ := 0.20 * price

def SellingPriceA (price : ℝ) : ℝ := price + ProfitA price

def ProfitB (price : ℝ) : ℝ := 0.25 * price

def SellingPriceB (price : ℝ) : ℝ := price + ProfitB price

theorem C_pays_228_for_cricket_bat :
  SellingPriceB (SellingPriceA CostPriceA) = 228 :=
by
  sorry

end NUMINAMATH_GPT_C_pays_228_for_cricket_bat_l1005_100535


namespace NUMINAMATH_GPT_bell_rings_count_l1005_100556

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

end NUMINAMATH_GPT_bell_rings_count_l1005_100556


namespace NUMINAMATH_GPT_granddaughter_age_is_12_l1005_100559

/-
Conditions:
- Betty is 60 years old.
- Her daughter is 40 percent younger than Betty.
- Her granddaughter is one-third her mother's age.

Question:
- Prove that the granddaughter is 12 years old.
-/

def age_of_Betty := 60

def age_of_daughter (age_of_Betty : ℕ) : ℕ :=
  age_of_Betty - age_of_Betty * 40 / 100

def age_of_granddaughter (age_of_daughter : ℕ) : ℕ :=
  age_of_daughter / 3

theorem granddaughter_age_is_12 (h1 : age_of_Betty = 60) : age_of_granddaughter (age_of_daughter age_of_Betty) = 12 := by
  sorry

end NUMINAMATH_GPT_granddaughter_age_is_12_l1005_100559


namespace NUMINAMATH_GPT_length_of_wooden_block_l1005_100578

theorem length_of_wooden_block (cm_to_m : ℝ := 30 / 100) (base_length : ℝ := 31) :
  base_length + cm_to_m = 31.3 :=
by
  sorry

end NUMINAMATH_GPT_length_of_wooden_block_l1005_100578


namespace NUMINAMATH_GPT_fgf_3_equals_108_l1005_100524

def f (x : ℕ) : ℕ := 2 * x + 4
def g (x : ℕ) : ℕ := 5 * x + 2

theorem fgf_3_equals_108 : f (g (f 3)) = 108 := 
by
  sorry

end NUMINAMATH_GPT_fgf_3_equals_108_l1005_100524


namespace NUMINAMATH_GPT_solution_set_f_derivative_l1005_100579

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 - 9 * x + 1

theorem solution_set_f_derivative :
  { x : ℝ | (deriv f x) < 0 } = { x : ℝ | -1 < x ∧ x < 3 } :=
by
  sorry

end NUMINAMATH_GPT_solution_set_f_derivative_l1005_100579


namespace NUMINAMATH_GPT_solve_system_l1005_100528

theorem solve_system (x y z a b c : ℝ)
  (h1 : x * (x + y + z) = a^2)
  (h2 : y * (x + y + z) = b^2)
  (h3 : z * (x + y + z) = c^2) :
  (x = a^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ x = -a^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (y = b^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ y = -b^2 / Real.sqrt (a^2 + b^2 + c^2)) ∧
  (z = c^2 / Real.sqrt (a^2 + b^2 + c^2) ∨ z = -c^2 / Real.sqrt (a^2 + b^2 + c^2)) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l1005_100528


namespace NUMINAMATH_GPT_charlie_fraction_l1005_100542

theorem charlie_fraction (J B C : ℕ) (f : ℚ) (hJ : J = 12) (hB : B = 10) 
  (h1 : B = (2 / 3) * C) (h2 : C = f * J + 9) : f = (1 / 2) := by
  sorry

end NUMINAMATH_GPT_charlie_fraction_l1005_100542


namespace NUMINAMATH_GPT_chips_needed_per_console_l1005_100515

-- Definitions based on the conditions
def chips_per_day : ℕ := 467
def consoles_per_day : ℕ := 93

-- The goal is to prove that each video game console needs 5 computer chips
theorem chips_needed_per_console : chips_per_day / consoles_per_day = 5 :=
by sorry

end NUMINAMATH_GPT_chips_needed_per_console_l1005_100515


namespace NUMINAMATH_GPT_solution_set_l1005_100599

-- Given conditions
variable (x : ℝ)

def inequality1 := 2 * x + 1 > 0
def inequality2 := (x + 1) / 3 > x - 1

-- The proof statement
theorem solution_set (h1 : inequality1 x) (h2 : inequality2 x) :
  -1 / 2 < x ∧ x < 2 :=
sorry

end NUMINAMATH_GPT_solution_set_l1005_100599


namespace NUMINAMATH_GPT_beggars_society_votes_l1005_100558

def total_voting_members (votes_for votes_against additional_against : ℕ) :=
  let majority := additional_against / 4
  let initial_difference := votes_for - votes_against
  let updated_against := votes_against + additional_against
  let updated_for := votes_for - additional_against
  updated_for + updated_against

theorem beggars_society_votes :
  total_voting_members 115 92 12 = 207 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_beggars_society_votes_l1005_100558


namespace NUMINAMATH_GPT_pyramid_base_edge_length_l1005_100509

theorem pyramid_base_edge_length 
(radius_hemisphere height_pyramid : ℝ)
(h_radius : radius_hemisphere = 4)
(h_height : height_pyramid = 10)
(h_tangent : ∀ face : ℝ, True) : 
∃ s : ℝ, s = 2 * Real.sqrt 42 :=
by
  sorry

end NUMINAMATH_GPT_pyramid_base_edge_length_l1005_100509


namespace NUMINAMATH_GPT_problem1_problem2_l1005_100551

open Real

-- Proof problem 1: Given condition and the required result.
theorem problem1 (x y : ℝ) (h : (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7) :
  x^2 + y^2 = 5 :=
sorry

-- Proof problem 2: Solve the polynomial equation.
theorem problem2 (x : ℝ) :
  (x = sqrt 2 ∨ x = -sqrt 2 ∨ x = 2 ∨ x = -2) ↔ (x^4 - 6 * x^2 + 8 = 0) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l1005_100551


namespace NUMINAMATH_GPT_find_abs_xyz_l1005_100586

variables {x y z : ℝ}

def distinct (a b c : ℝ) : Prop := a ≠ b ∧ b ≠ c ∧ c ≠ a

theorem find_abs_xyz
  (h1 : x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0)
  (h2 : distinct x y z)
  (h3 : x + 1 / y = 2)
  (h4 : y + 1 / z = 2)
  (h5 : z + 1 / x = 2) :
  |x * y * z| = 1 :=
by sorry

end NUMINAMATH_GPT_find_abs_xyz_l1005_100586


namespace NUMINAMATH_GPT_range_of_a_for_monotonic_f_l1005_100572

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x - a^2 * x^2 + a * x

theorem range_of_a_for_monotonic_f (a : ℝ) : 
  (∀ x, 1 < x → f a x ≤ f a (1 : ℝ)) ↔ (a ≤ -1 / 2 ∨ 1 ≤ a) := 
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_monotonic_f_l1005_100572


namespace NUMINAMATH_GPT_rectangle_area_l1005_100500

-- Definitions
def perimeter (l w : ℝ) : ℝ := 2 * (l + w)
def length (w : ℝ) : ℝ := 2 * w
def area (l w : ℝ) : ℝ := l * w

-- Main Statement
theorem rectangle_area (w l : ℝ) (h_p : perimeter l w = 120) (h_l : l = length w) :
  area l w = 800 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_l1005_100500


namespace NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1993_l1005_100597

theorem rightmost_three_digits_of_7_pow_1993 :
  7^1993 % 1000 = 407 := 
sorry

end NUMINAMATH_GPT_rightmost_three_digits_of_7_pow_1993_l1005_100597


namespace NUMINAMATH_GPT_river_bend_students_more_than_pets_l1005_100561

theorem river_bend_students_more_than_pets 
  (students_per_classroom : ℕ)
  (rabbits_per_classroom : ℕ)
  (hamsters_per_classroom : ℕ)
  (number_of_classrooms : ℕ)
  (total_students : ℕ := students_per_classroom * number_of_classrooms)
  (total_rabbits : ℕ := rabbits_per_classroom * number_of_classrooms)
  (total_hamsters : ℕ := hamsters_per_classroom * number_of_classrooms)
  (total_pets : ℕ := total_rabbits + total_hamsters) :
  students_per_classroom = 24 ∧ rabbits_per_classroom = 2 ∧ hamsters_per_classroom = 3 ∧ number_of_classrooms = 5 →
  total_students - total_pets = 95 :=
by
  sorry

end NUMINAMATH_GPT_river_bend_students_more_than_pets_l1005_100561


namespace NUMINAMATH_GPT_sum_of_sides_le_twice_third_side_l1005_100577

theorem sum_of_sides_le_twice_third_side 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : A + C = 2 * B) 
  (h2 : A + B + C = 180)
  (h3 : a / (Real.sin A) = b / (Real.sin B))
  (h4 : a / (Real.sin A) = c / (Real.sin C))
  (h5 : b / (Real.sin B) = c / (Real.sin C)) : 
  a + c ≤ 2 * b := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_sides_le_twice_third_side_l1005_100577


namespace NUMINAMATH_GPT_probability_heads_and_multiple_of_five_l1005_100594

def is_multiple_of_five (n : ℕ) : Prop := n % 5 = 0

def coin_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

def die_is_fair : Prop := true -- since given in conditions, it’s fair, no need to reprove; assume true

theorem probability_heads_and_multiple_of_five :
  coin_is_fair ∧ die_is_fair →
  (1 / 2) * (1 / 6) = (1 / 12) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_probability_heads_and_multiple_of_five_l1005_100594


namespace NUMINAMATH_GPT_average_minutes_per_player_l1005_100569

theorem average_minutes_per_player
  (pg sg sf pf c : ℕ)
  (total_players : ℕ)
  (hp_pg : pg = 130)
  (hp_sg : sg = 145)
  (hp_sf : sf = 85)
  (hp_pf : pf = 60)
  (hp_c : c = 180)
  (hp_total_players : total_players = 5) :
  (pg + sg + sf + pf + c) / total_players / 60 = 2 :=
by
  sorry

end NUMINAMATH_GPT_average_minutes_per_player_l1005_100569


namespace NUMINAMATH_GPT_y_intercept_of_line_l1005_100514

def line_equation (x y : ℝ) : Prop := x - 2 * y + 4 = 0

theorem y_intercept_of_line : ∀ y : ℝ, line_equation 0 y → y = 2 :=
by 
  intro y h
  unfold line_equation at h
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l1005_100514


namespace NUMINAMATH_GPT_triangle_area_ratio_l1005_100596

noncomputable def area_ratio (a b c d e f : ℕ) : ℚ :=
  (a * b) / (d * e)

theorem triangle_area_ratio : area_ratio 6 8 10 9 12 15 = 4 / 9 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_ratio_l1005_100596


namespace NUMINAMATH_GPT_arithmetic_sequence_a5_l1005_100570

variable (a : ℕ → ℝ)

-- Conditions translated to Lean definitions
def cond1 : Prop := a 3 = 7
def cond2 : Prop := a 9 = 19

-- Theorem statement that needs to be proved
theorem arithmetic_sequence_a5 (h1 : cond1 a) (h2 : cond2 a) : a 5 = 11 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_a5_l1005_100570


namespace NUMINAMATH_GPT_hyperbola_eqn_correct_l1005_100530

def parabola_focus : ℝ × ℝ := (1, 0)

def hyperbola_vertex := parabola_focus

def hyperbola_eccentricity : ℝ := 2

def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 - (y^2 / 3) = 1

theorem hyperbola_eqn_correct (x y : ℝ) :
  hyperbola_equation x y :=
sorry

end NUMINAMATH_GPT_hyperbola_eqn_correct_l1005_100530


namespace NUMINAMATH_GPT_symmetric_circle_equation_l1005_100537

theorem symmetric_circle_equation :
  (∀ x y : ℝ, (x - 1) ^ 2 + y ^ 2 = 1 ↔ x ^ 2 + (y + 1) ^ 2 = 1) :=
by sorry

end NUMINAMATH_GPT_symmetric_circle_equation_l1005_100537


namespace NUMINAMATH_GPT_initial_matchsticks_l1005_100536

-- Define the problem conditions
def matchsticks_elvis := 4
def squares_elvis := 5
def matchsticks_ralph := 8
def squares_ralph := 3
def matchsticks_left := 6

-- Calculate the total matchsticks used by Elvis and Ralph
def total_used_elvis := matchsticks_elvis * squares_elvis
def total_used_ralph := matchsticks_ralph * squares_ralph
def total_used := total_used_elvis + total_used_ralph

-- The proof statement
theorem initial_matchsticks (matchsticks_elvis squares_elvis matchsticks_ralph squares_ralph matchsticks_left : ℕ) : total_used + matchsticks_left = 50 := 
by
  sorry

end NUMINAMATH_GPT_initial_matchsticks_l1005_100536


namespace NUMINAMATH_GPT_ducks_remaining_after_three_nights_l1005_100538

def initial_ducks : ℕ := 320
def first_night_ducks (initial_ducks : ℕ) : ℕ := initial_ducks - (initial_ducks / 4)
def second_night_ducks (first_night_ducks : ℕ) : ℕ := first_night_ducks - (first_night_ducks / 6)
def third_night_ducks (second_night_ducks : ℕ) : ℕ := second_night_ducks - (second_night_ducks * 30 / 100)

theorem ducks_remaining_after_three_nights : 
  third_night_ducks (second_night_ducks (first_night_ducks initial_ducks)) = 140 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_ducks_remaining_after_three_nights_l1005_100538


namespace NUMINAMATH_GPT_find_k_series_sum_l1005_100574

theorem find_k_series_sum (k : ℝ) :
  (2 + ∑' n : ℕ, (2 + (n + 1) * k) / 2 ^ (n + 1)) = 6 -> k = 1 :=
by 
  sorry

end NUMINAMATH_GPT_find_k_series_sum_l1005_100574


namespace NUMINAMATH_GPT_number_of_children_l1005_100504

-- Definitions of given conditions
def total_passengers := 170
def men := 90
def women := men / 2
def adults := men + women
def children := total_passengers - adults

-- Theorem statement
theorem number_of_children : children = 35 :=
by
  sorry

end NUMINAMATH_GPT_number_of_children_l1005_100504


namespace NUMINAMATH_GPT_usual_time_to_catch_bus_l1005_100573

variables (S T T' : ℝ)

theorem usual_time_to_catch_bus
  (h1 : T' = (5 / 4) * T)
  (h2 : T' - T = 6) : T = 24 :=
sorry

end NUMINAMATH_GPT_usual_time_to_catch_bus_l1005_100573


namespace NUMINAMATH_GPT_gino_initial_sticks_l1005_100566

-- Definitions based on the conditions
def given_sticks : ℕ := 50
def remaining_sticks : ℕ := 13
def initial_sticks (x y : ℕ) : ℕ := x + y

-- Theorem statement based on the mathematically equivalent proof problem
theorem gino_initial_sticks :
  initial_sticks given_sticks remaining_sticks = 63 :=
by
  sorry

end NUMINAMATH_GPT_gino_initial_sticks_l1005_100566


namespace NUMINAMATH_GPT_quadratic_roots_sum_product_l1005_100576

noncomputable def quadratic_sum (a b c : ℝ) : ℝ := -b / a
noncomputable def quadratic_product (a b c : ℝ) : ℝ := c / a

theorem quadratic_roots_sum_product :
  let a := 9
  let b := -45
  let c := 50
  quadratic_sum a b c = 5 ∧ quadratic_product a b c = 50 / 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_sum_product_l1005_100576


namespace NUMINAMATH_GPT_C_investment_value_is_correct_l1005_100534

noncomputable def C_investment_contribution 
  (A_investment B_investment total_profit A_profit_share : ℝ) : ℝ :=
  let C_investment := 
    (A_profit_share * (A_investment + B_investment) - A_investment * total_profit) / 
    (total_profit - A_profit_share)
  C_investment

theorem C_investment_value_is_correct : 
  C_investment_contribution 6300 4200 13600 4080 = 10500 := 
by
  unfold C_investment_contribution
  norm_num
  sorry

end NUMINAMATH_GPT_C_investment_value_is_correct_l1005_100534


namespace NUMINAMATH_GPT_probability_of_selecting_cooking_l1005_100521

theorem probability_of_selecting_cooking :
  let courses := ["planting", "cooking", "pottery", "carpentry"]
  let total_courses := courses.length
  (1 / total_courses) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_selecting_cooking_l1005_100521


namespace NUMINAMATH_GPT_greatest_x_integer_l1005_100506

theorem greatest_x_integer (x : ℤ) : 
  (∃ k : ℤ, (x^2 + 4 * x + 9) = k * (x - 4)) ↔ x ≤ 5 :=
by
  sorry

end NUMINAMATH_GPT_greatest_x_integer_l1005_100506


namespace NUMINAMATH_GPT_original_time_between_maintenance_checks_l1005_100554

theorem original_time_between_maintenance_checks (x : ℝ) 
  (h1 : 2 * x = 60) : x = 30 := sorry

end NUMINAMATH_GPT_original_time_between_maintenance_checks_l1005_100554


namespace NUMINAMATH_GPT_max_value_7a_9b_l1005_100526

theorem max_value_7a_9b 
    (r_1 r_2 r_3 a b : ℝ) 
    (h_eq : ∀ x, x^3 - x^2 + a * x - b = 0 → (x = r_1 ∨ x = r_2 ∨ x = r_3))
    (h_root_sum : r_1 + r_2 + r_3 = 1)
    (h_root_prod : r_1 * r_2 * r_3 = b)
    (h_root_sumprod : r_1 * r_2 + r_2 * r_3 + r_3 * r_1 = a)
    (h_bounds : ∀ i, i = r_1 ∨ i = r_2 ∨ i = r_3 → 0 < i ∧ i < 1) :
        7 * a - 9 * b ≤ 2 := 
sorry

end NUMINAMATH_GPT_max_value_7a_9b_l1005_100526


namespace NUMINAMATH_GPT_jellybean_probability_l1005_100588

theorem jellybean_probability :
  let total_jellybeans := 12
  let red_jellybeans := 5
  let blue_jellybeans := 2
  let yellow_jellybeans := 5
  let total_picks := 4
  let successful_outcomes := 10 * 7 
  let total_outcomes := Nat.choose 12 4 
  let required_probability := 14 / 99 
  successful_outcomes = 70 ∧ total_outcomes = 495 → 
  successful_outcomes / total_outcomes = required_probability := 
by 
  intros
  sorry

end NUMINAMATH_GPT_jellybean_probability_l1005_100588


namespace NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1005_100555

theorem min_value_of_reciprocal_sum {a b : ℝ} (ha : a > 0) (hb : b > 0)
  (hgeom : 3 = Real.sqrt (3^a * 3^b)) : (1 / a + 1 / b) = 2 :=
sorry  -- Proof not required, only the statement is needed.

end NUMINAMATH_GPT_min_value_of_reciprocal_sum_l1005_100555


namespace NUMINAMATH_GPT_chessboard_number_determination_l1005_100560

theorem chessboard_number_determination (d_n : ℤ) (a_n b_n a_1 b_1 c_0 d_0 : ℤ) :
  (∀ i j : ℤ, d_n + a_n = b_n + a_1 + b_1 - (c_0 + d_0) → 
   a_n + b_n = c_0 + d_0 + d_n) →
  ∃ x : ℤ, x = a_1 + b_1 - d_n ∧ 
  x = d_n + (a_1 - c_0) + (b_1 - d_0) :=
by
  sorry

end NUMINAMATH_GPT_chessboard_number_determination_l1005_100560


namespace NUMINAMATH_GPT_number_of_rings_l1005_100552

def is_number_ring (A : Set ℝ) : Prop :=
  ∀ (a b : ℝ), a ∈ A → b ∈ A → (a + b) ∈ A ∧ (a - b) ∈ A ∧ (a * b) ∈ A

def Z := { n : ℝ | ∃ k : ℤ, n = k }
def N := { n : ℝ | ∃ k : ℕ, n = k }
def Q := { n : ℝ | ∃ (a b : ℤ), b ≠ 0 ∧ n = a / b }
def R := { n : ℝ | True }
def M := { x : ℝ | ∃ (n m : ℤ), x = n + m * Real.sqrt 2 }
def P := { x : ℝ | ∃ (m n : ℕ), n ≠ 0 ∧ x = m / (2 * n) }

theorem number_of_rings :
  (is_number_ring Z) ∧ ¬(is_number_ring N) ∧ (is_number_ring Q) ∧ 
  (is_number_ring R) ∧ (is_number_ring M) ∧ ¬(is_number_ring P) :=
by sorry

end NUMINAMATH_GPT_number_of_rings_l1005_100552


namespace NUMINAMATH_GPT_odd_positive_int_divisible_by_24_l1005_100543

theorem odd_positive_int_divisible_by_24 (n : ℕ) (hn : n % 2 = 1 ∧ n > 0) : 24 ∣ (n ^ n - n) :=
sorry

end NUMINAMATH_GPT_odd_positive_int_divisible_by_24_l1005_100543


namespace NUMINAMATH_GPT_pizza_slices_correct_l1005_100580

-- Definitions based on conditions
def john_slices : Nat := 3
def sam_slices : Nat := 2 * john_slices
def eaten_slices : Nat := john_slices + sam_slices
def remaining_slices : Nat := 3
def total_slices : Nat := eaten_slices + remaining_slices

-- The statement to be proven.
theorem pizza_slices_correct : total_slices = 12 := by
  sorry

end NUMINAMATH_GPT_pizza_slices_correct_l1005_100580


namespace NUMINAMATH_GPT_dartboard_central_angle_l1005_100593

-- Define the conditions
variables {A : ℝ} {x : ℝ}

-- State the theorem
theorem dartboard_central_angle (h₁ : A > 0) (h₂ : (1/4 : ℝ) = ((x / 360) * A) / A) : x = 90 := 
by sorry

end NUMINAMATH_GPT_dartboard_central_angle_l1005_100593


namespace NUMINAMATH_GPT_trig_identity_l1005_100510

theorem trig_identity : 
  ( 4 * Real.sin (40 * Real.pi / 180) * Real.cos (40 * Real.pi / 180) / Real.cos (20 * Real.pi / 180) 
   - Real.tan (20 * Real.pi / 180) ) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_trig_identity_l1005_100510


namespace NUMINAMATH_GPT_k_of_neg7_l1005_100545

noncomputable def h (x : ℝ) : ℝ := 4 * x - 9
noncomputable def k (x : ℝ) : ℝ := 3 * x^2 + 4 * x - 2

theorem k_of_neg7 : k (-7) = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_k_of_neg7_l1005_100545


namespace NUMINAMATH_GPT_function_three_distinct_zeros_l1005_100567

theorem function_three_distinct_zeros (a : ℝ) (f : ℝ → ℝ) :
  (∀ x : ℝ, f x = x^3 - 3 * a * x + a) ∧ (∀ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ f x = 0 ∧ f y = 0 ∧ f z = 0) →
  a > 1/4 :=
by
  sorry

end NUMINAMATH_GPT_function_three_distinct_zeros_l1005_100567


namespace NUMINAMATH_GPT_fox_jeans_price_l1005_100583

theorem fox_jeans_price (F : ℝ) (P : ℝ) 
  (pony_price : P = 18) 
  (total_savings : 3 * F * 0.08 + 2 * P * 0.14 = 8.64)
  (total_discount_rate : 0.08 + 0.14 = 0.22)
  (pony_discount_rate : 0.14 = 13.999999999999993 / 100) 
  : F = 15 :=
by
  sorry

end NUMINAMATH_GPT_fox_jeans_price_l1005_100583


namespace NUMINAMATH_GPT_smallest_even_piece_to_stop_triangle_l1005_100517

-- Define a predicate to check if an integer is even
def even (x : ℕ) : Prop := x % 2 = 0

-- Define the conditions for triangle inequality to hold
def triangle_inequality_violated (a b c : ℕ) : Prop :=
  a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

-- Define the main theorem
theorem smallest_even_piece_to_stop_triangle
  (x : ℕ) (hx : even x) (len1 len2 len3 : ℕ)
  (h_len1 : len1 = 7) (h_len2 : len2 = 24) (h_len3 : len3 = 25) :
  6 ≤ x → triangle_inequality_violated (len1 - x) (len2 - x) (len3 - x) :=
by
  sorry

end NUMINAMATH_GPT_smallest_even_piece_to_stop_triangle_l1005_100517


namespace NUMINAMATH_GPT_min_value_of_f_l1005_100565

noncomputable def f (x : ℝ) : ℝ :=
  x^2 / (x - 3)

theorem min_value_of_f : ∀ x > 3, f x ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_f_l1005_100565


namespace NUMINAMATH_GPT_g_26_equals_125_l1005_100590

noncomputable def g : ℕ → ℕ := sorry

axiom g_property : ∀ x, g (x + g x) = 5 * g x
axiom g_initial : g 1 = 5

theorem g_26_equals_125 : g 26 = 125 :=
by
  sorry

end NUMINAMATH_GPT_g_26_equals_125_l1005_100590


namespace NUMINAMATH_GPT_find_x_l1005_100575

theorem find_x (x : ℕ) (hx : x > 0 ∧ x <= 100) 
    (mean_twice_mode : (40 + 57 + 76 + 90 + x + x) / 6 = 2 * x) : 
    x = 26 :=
sorry

end NUMINAMATH_GPT_find_x_l1005_100575


namespace NUMINAMATH_GPT_fraction_orange_juice_in_large_container_l1005_100549

-- Definitions according to the conditions
def pitcher1_capacity : ℕ := 800
def pitcher2_capacity : ℕ := 500
def pitcher1_fraction_orange_juice : ℚ := 1 / 4
def pitcher2_fraction_orange_juice : ℚ := 3 / 5

-- Prove the fraction of orange juice
theorem fraction_orange_juice_in_large_container :
  ( (pitcher1_capacity * pitcher1_fraction_orange_juice + pitcher2_capacity * pitcher2_fraction_orange_juice) / 
    (pitcher1_capacity + pitcher2_capacity) ) = 5 / 13 :=
by
  sorry

end NUMINAMATH_GPT_fraction_orange_juice_in_large_container_l1005_100549


namespace NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1005_100502

-- Define the terms in the arithmetic sequence
def sequence_term (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Given conditions
def a1 : ℚ := 2 / 3
def a17 : ℚ := 5 / 6
def d : ℚ := 1 / 96 -- Calculated common difference

-- Prove the ninth term is 3/4
theorem arithmetic_sequence_ninth_term :
  sequence_term a1 d 9 = 3 / 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_ninth_term_l1005_100502


namespace NUMINAMATH_GPT_common_ratio_of_geometric_series_l1005_100501

theorem common_ratio_of_geometric_series (a S r : ℝ) (h1 : a = 500) (h2 : S = 2500) (h3 : a / (1 - r) = S) : r = 4 / 5 :=
by
  rw [h1, h2] at h3
  sorry

end NUMINAMATH_GPT_common_ratio_of_geometric_series_l1005_100501


namespace NUMINAMATH_GPT_total_apples_proof_l1005_100512

-- Define the quantities Adam bought each day
def apples_monday := 15
def apples_tuesday := apples_monday * 3
def apples_wednesday := apples_tuesday * 4

-- The total quantity of apples Adam bought over these three days
def total_apples := apples_monday + apples_tuesday + apples_wednesday

-- Theorem stating that the total quantity of apples bought is 240
theorem total_apples_proof : total_apples = 240 := by
  sorry

end NUMINAMATH_GPT_total_apples_proof_l1005_100512


namespace NUMINAMATH_GPT_sin_45_deg_l1005_100546

theorem sin_45_deg : Real.sin (Real.pi / 4) = Real.sqrt 2 / 2 := 
by 
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sin_45_deg_l1005_100546


namespace NUMINAMATH_GPT_number_triangle_value_of_n_l1005_100507

theorem number_triangle_value_of_n:
  ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = 2022 ∧ (∃ n : ℕ, n > 0 ∧ n^2 ∣ 2022 ∧ n = 1) :=
by sorry

end NUMINAMATH_GPT_number_triangle_value_of_n_l1005_100507


namespace NUMINAMATH_GPT_hcl_formed_l1005_100541

-- Define the balanced chemical equation as a relationship between reactants and products
def balanced_equation (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 + 4 * m_Cl2 = m_CCl4 + 6 * m_HCl

-- Define the problem-specific values
def reaction_given (m_C2H6 m_Cl2 m_CCl4 m_HCl : ℝ) :=
  m_C2H6 = 3 ∧ m_Cl2 = 21 ∧ m_CCl4 = 6 ∧ balanced_equation m_C2H6 m_Cl2 m_CCl4 m_HCl

-- Prove the number of moles of HCl formed
theorem hcl_formed : ∃ (m_HCl : ℝ), reaction_given 3 21 6 m_HCl ∧ m_HCl = 18 :=
by
  sorry

end NUMINAMATH_GPT_hcl_formed_l1005_100541


namespace NUMINAMATH_GPT_man_speed_proof_l1005_100564

noncomputable def man_speed_to_post_office (v : ℝ) : Prop :=
  let distance := 19.999999999999996
  let time_back := distance / 4
  let total_time := 5 + 48 / 60
  v > 0 ∧ distance / v + time_back = total_time

theorem man_speed_proof : ∃ v : ℝ, man_speed_to_post_office v ∧ v = 25 := by
  sorry

end NUMINAMATH_GPT_man_speed_proof_l1005_100564


namespace NUMINAMATH_GPT_find_x_y_l1005_100571

theorem find_x_y (x y : ℝ) : 
    (3 * x + 2 * y + 5 * x + 7 * x = 360) →
    (x = y) →
    (x = 360 / 17) ∧ (y = 360 / 17) := by
  intros h₁ h₂
  sorry

end NUMINAMATH_GPT_find_x_y_l1005_100571


namespace NUMINAMATH_GPT_find_k_of_sequence_l1005_100544

theorem find_k_of_sequence (S : ℕ → ℤ) (a : ℕ → ℤ) (hS : ∀ n, S n = n^2 - 9 * n)
  (hS_recurr : ∀ n ≥ 2, a n = S n - S (n-1)) (h_a_k : ∃ k, 5 < a k ∧ a k < 8) : ∃ k, k = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_k_of_sequence_l1005_100544


namespace NUMINAMATH_GPT_quadratic_roots_ratio_l1005_100562

theorem quadratic_roots_ratio {m n p : ℤ} (h₀ : m ≠ 0) (h₁ : n ≠ 0) (h₂ : p ≠ 0)
  (h₃ : ∃ r1 r2 : ℤ, r1 * r2 = m ∧ n = 9 * r1 * r2 ∧ p = -(r1 + r2) ∧ m = -3 * (r1 + r2)) :
  n / p = -27 := by
  sorry

end NUMINAMATH_GPT_quadratic_roots_ratio_l1005_100562


namespace NUMINAMATH_GPT_projection_sum_of_squares_l1005_100584

theorem projection_sum_of_squares (a : ℝ) (α β γ : ℝ) 
    (h1 : (Real.cos α)^2 + (Real.cos β)^2 + (Real.cos γ)^2 = 1) 
    (h2 : (Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2 = 2) :
    4 * a^2 * ((Real.sin α)^2 + (Real.sin β)^2 + (Real.sin γ)^2) = 8 * a^2 := 
by
  sorry

end NUMINAMATH_GPT_projection_sum_of_squares_l1005_100584


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1005_100532

theorem sufficient_but_not_necessary_condition (x : ℝ) (p : -1 < x ∧ x < 3) (q : x^2 - 5 * x - 6 < 0) : 
  (-1 < x ∧ x < 3) → (x^2 - 5 * x - 6 < 0) ∧ ¬((x^2 - 5 * x - 6 < 0) → (-1 < x ∧ x < 3)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_condition_l1005_100532


namespace NUMINAMATH_GPT_f_at_3_l1005_100529

theorem f_at_3 (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x + 2) = x + 3) : f 3 = 4 := 
sorry

end NUMINAMATH_GPT_f_at_3_l1005_100529


namespace NUMINAMATH_GPT_birds_on_fence_l1005_100539

theorem birds_on_fence (B : ℕ) : ∃ B, (∃ S, S = 6 ∧ S = (B + 3) + 1) → B = 2 :=
by
  sorry

end NUMINAMATH_GPT_birds_on_fence_l1005_100539


namespace NUMINAMATH_GPT_logical_inconsistency_in_dihedral_angle_def_l1005_100557

-- Define the given incorrect definition
def incorrect_dihedral_angle_def : String :=
  "A dihedral angle is an angle formed by two half-planes originating from one straight line."

-- Define the correct definition
def correct_dihedral_angle_def : String :=
  "A dihedral angle is a spatial figure consisting of two half-planes that share a common edge."

-- Define the logical inconsistency
theorem logical_inconsistency_in_dihedral_angle_def :
  incorrect_dihedral_angle_def ≠ correct_dihedral_angle_def := by
  sorry

end NUMINAMATH_GPT_logical_inconsistency_in_dihedral_angle_def_l1005_100557


namespace NUMINAMATH_GPT_leif_fruit_weight_difference_l1005_100533

theorem leif_fruit_weight_difference :
  let apples_ounces := 27.5
  let grams_per_ounce := 28.35
  let apples_grams := apples_ounces * grams_per_ounce
  let dozens_oranges := 5.5
  let oranges_per_dozen := 12
  let total_oranges := dozens_oranges * oranges_per_dozen
  let weight_per_orange := 45
  let oranges_grams := total_oranges * weight_per_orange
  let weight_difference := oranges_grams - apples_grams
  weight_difference = 2190.375 := by
{
  sorry
}

end NUMINAMATH_GPT_leif_fruit_weight_difference_l1005_100533


namespace NUMINAMATH_GPT_total_children_l1005_100516

theorem total_children {x y : ℕ} (h₁ : x = 18) (h₂ : y = 12) 
  (h₃ : x + y = 30) (h₄ : x = 18) (h₅ : y = 12) : 2 * x + 3 * y = 72 := 
by
  sorry

end NUMINAMATH_GPT_total_children_l1005_100516


namespace NUMINAMATH_GPT_vector_on_plane_l1005_100511

-- Define the vectors w and the condition for proj_w v
def w : ℝ × ℝ × ℝ := (3, -3, 3)
def v (x y z : ℝ) : ℝ × ℝ × ℝ := (x, y, z)
def projection_condition (x y z : ℝ) : Prop :=
  ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * (-3) = -6 ∧ ((3 * x - 3 * y + 3 * z) / 27) * 3 = 6

-- Define the plane equation
def plane_eq (x y z : ℝ) : Prop := x - y + z - 18 = 0

-- Prove that the set of vectors v lies on the plane
theorem vector_on_plane (x y z : ℝ) (h : projection_condition x y z) : plane_eq x y z :=
  sorry

end NUMINAMATH_GPT_vector_on_plane_l1005_100511


namespace NUMINAMATH_GPT_trivia_team_missing_members_l1005_100548

theorem trivia_team_missing_members 
  (total_members : ℕ)
  (points_per_member : ℕ)
  (total_points : ℕ)
  (showed_up_members : ℕ)
  (missing_members : ℕ) 
  (h1 : total_members = 15) 
  (h2 : points_per_member = 3) 
  (h3 : total_points = 27) 
  (h4 : showed_up_members = total_points / points_per_member) 
  (h5 : missing_members = total_members - showed_up_members) : 
  missing_members = 6 :=
by
  sorry

end NUMINAMATH_GPT_trivia_team_missing_members_l1005_100548


namespace NUMINAMATH_GPT_same_last_k_digits_pow_l1005_100547

theorem same_last_k_digits_pow (A B : ℤ) (k n : ℕ) 
  (h : A % 10^k = B % 10^k) : 
  (A^n % 10^k = B^n % 10^k) := 
by
  sorry

end NUMINAMATH_GPT_same_last_k_digits_pow_l1005_100547


namespace NUMINAMATH_GPT_lasagna_ground_mince_l1005_100598

theorem lasagna_ground_mince (total_ground_mince : ℕ) (num_cottage_pies : ℕ) (ground_mince_per_cottage_pie : ℕ) 
  (num_lasagnas : ℕ) (L : ℕ) : 
  total_ground_mince = 500 ∧ num_cottage_pies = 100 ∧ ground_mince_per_cottage_pie = 3 
  ∧ num_lasagnas = 100 ∧ total_ground_mince - num_cottage_pies * ground_mince_per_cottage_pie = num_lasagnas * L 
  → L = 2 := 
by sorry

end NUMINAMATH_GPT_lasagna_ground_mince_l1005_100598


namespace NUMINAMATH_GPT_midpoint_quadrilateral_inequality_l1005_100525

theorem midpoint_quadrilateral_inequality 
  (A B C D E F G H : ℝ) 
  (S_ABCD : ℝ)
  (midpoints_A : E = (A + B) / 2)
  (midpoints_B : F = (B + C) / 2)
  (midpoints_C : G = (C + D) / 2)
  (midpoints_D : H = (D + A) / 2)
  (EG : ℝ)
  (HF : ℝ) :
  S_ABCD ≤ EG * HF ∧ EG * HF ≤ (B + D) * (A + C) / 4 := by
  sorry

end NUMINAMATH_GPT_midpoint_quadrilateral_inequality_l1005_100525


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1005_100518

theorem quadratic_two_distinct_real_roots
  (a1 a2 a3 a4 : ℝ)
  (h : a1 > a2 ∧ a2 > a3 ∧ a3 > a4) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 - (a1 + a2 + a3 + a4) * x1 + (a1 * a3 + a2 * a4) = 0)
  ∧ (x2^2 - (a1 + a2 + a3 + a4) * x2 + (a1 * a3 + a2 * a4) = 0) :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1005_100518


namespace NUMINAMATH_GPT_find_whole_number_l1005_100582

theorem find_whole_number (N : ℕ) : 9.25 < (N : ℝ) / 4 ∧ (N : ℝ) / 4 < 9.75 → N = 38 := by
  intros h
  have hN : 37 < (N : ℝ) ∧ (N : ℝ) < 39 := by
    -- This part follows directly from multiplying the inequality by 4.
    sorry

  -- Convert to integer comparison
  have h1 : 38 ≤ N := by
    -- Since 37 < N, N must be at least 38 as N is an integer.
    sorry
    
  have h2 : N < 39 := by
    sorry

  -- Conclude that N = 38 as it is the single whole number within the range.
  sorry

end NUMINAMATH_GPT_find_whole_number_l1005_100582


namespace NUMINAMATH_GPT_average_eq_35_implies_y_eq_50_l1005_100553

theorem average_eq_35_implies_y_eq_50 (y : ℤ) (h : (15 + 30 + 45 + y) / 4 = 35) : y = 50 :=
by
  sorry

end NUMINAMATH_GPT_average_eq_35_implies_y_eq_50_l1005_100553


namespace NUMINAMATH_GPT_pups_more_than_adults_l1005_100531

-- Define the counts of dogs
def H := 5  -- number of huskies
def P := 2  -- number of pitbulls
def G := 4  -- number of golden retrievers

-- Define the number of pups each type of dog had
def pups_per_husky_and_pitbull := 3
def additional_pups_per_golden_retriever := 2
def pups_per_golden_retriever := pups_per_husky_and_pitbull + additional_pups_per_golden_retriever

-- Calculate the total number of pups
def total_pups := H * pups_per_husky_and_pitbull + P * pups_per_husky_and_pitbull + G * pups_per_golden_retriever

-- Calculate the total number of adult dogs
def total_adult_dogs := H + P + G

-- Prove that the number of pups is 30 more than the number of adult dogs
theorem pups_more_than_adults : total_pups - total_adult_dogs = 30 :=
by
  -- fill in the proof later
  sorry

end NUMINAMATH_GPT_pups_more_than_adults_l1005_100531


namespace NUMINAMATH_GPT_square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l1005_100527

variable {a b : ℝ}

theorem square_inequality_not_sufficient_nor_necessary_for_cube_inequality (a b : ℝ) :
  (a^2 > b^2) ↔ (a^3 > b^3) = false :=
sorry

end NUMINAMATH_GPT_square_inequality_not_sufficient_nor_necessary_for_cube_inequality_l1005_100527


namespace NUMINAMATH_GPT_primes_p_plus_10_plus_14_l1005_100581

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem primes_p_plus_10_plus_14 (p : ℕ) 
  (h1 : is_prime p) 
  (h2 : is_prime (p + 10)) 
  (h3 : is_prime (p + 14)) 
  : p = 3 := sorry

end NUMINAMATH_GPT_primes_p_plus_10_plus_14_l1005_100581


namespace NUMINAMATH_GPT_solution_set_l1005_100568

def within_bounds (x : ℝ) : Prop := |2 * x + 1| < 1

theorem solution_set : {x : ℝ | within_bounds x} = {x : ℝ | -1 < x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_l1005_100568


namespace NUMINAMATH_GPT_smallest_x_for_M_cube_l1005_100591

theorem smallest_x_for_M_cube (x M : ℤ) (h1 : 1890 * x = M^3) : x = 4900 :=
sorry

end NUMINAMATH_GPT_smallest_x_for_M_cube_l1005_100591


namespace NUMINAMATH_GPT_factorize_x_squared_minus_1_l1005_100505

theorem factorize_x_squared_minus_1 (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x_squared_minus_1_l1005_100505


namespace NUMINAMATH_GPT_five_fourths_of_twelve_fifths_eq_three_l1005_100522

theorem five_fourths_of_twelve_fifths_eq_three : (5 : ℝ) / 4 * (12 / 5) = 3 := 
by 
  sorry

end NUMINAMATH_GPT_five_fourths_of_twelve_fifths_eq_three_l1005_100522


namespace NUMINAMATH_GPT_minimum_red_chips_l1005_100508

theorem minimum_red_chips (w b r : ℕ) (h1 : b ≥ (1 / 3) * w) (h2 : b ≤ (1 / 4) * r) (h3 : w + b ≥ 70) : r ≥ 72 := by
  sorry

end NUMINAMATH_GPT_minimum_red_chips_l1005_100508


namespace NUMINAMATH_GPT_quadratic_sum_l1005_100550

theorem quadratic_sum (b c : ℤ) : 
  (∃ b c : ℤ, (x^2 - 10*x + 15 = 0) ↔ ((x + b)^2 = c)) → b + c = 5 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_quadratic_sum_l1005_100550


namespace NUMINAMATH_GPT_dvd_cd_ratio_l1005_100595

theorem dvd_cd_ratio (total_sales : ℕ) (dvd_sales : ℕ) (cd_sales : ℕ) (h1 : total_sales = 273) (h2 : dvd_sales = 168) (h3 : cd_sales = total_sales - dvd_sales) : (dvd_sales / Nat.gcd dvd_sales cd_sales) = 8 ∧ (cd_sales / Nat.gcd dvd_sales cd_sales) = 5 :=
by
  sorry

end NUMINAMATH_GPT_dvd_cd_ratio_l1005_100595


namespace NUMINAMATH_GPT_people_from_second_row_joined_l1005_100503

theorem people_from_second_row_joined
  (initial_first_row : ℕ) (initial_second_row : ℕ) (initial_third_row : ℕ) (people_waded : ℕ) (remaining_people : ℕ)
  (H1 : initial_first_row = 24)
  (H2 : initial_second_row = 20)
  (H3 : initial_third_row = 18)
  (H4 : people_waded = 3)
  (H5 : remaining_people = 54) :
  initial_second_row - (initial_first_row + initial_second_row + initial_third_row - initial_first_row - people_waded - remaining_people) = 5 :=
by
  sorry

end NUMINAMATH_GPT_people_from_second_row_joined_l1005_100503


namespace NUMINAMATH_GPT_two_integers_difference_l1005_100592

theorem two_integers_difference
  (x y : ℕ)
  (h_sum : x + y = 5)
  (h_cube_diff : x^3 - y^3 = 63)
  (h_gt : x > y) :
  x - y = 3 := 
sorry

end NUMINAMATH_GPT_two_integers_difference_l1005_100592


namespace NUMINAMATH_GPT_total_revenue_full_price_tickets_l1005_100523

theorem total_revenue_full_price_tickets (f q : ℕ) (p : ℝ) :
  f + q = 170 ∧ f * p + q * (p / 4) = 2917 → f * p = 1748 := by
  sorry

end NUMINAMATH_GPT_total_revenue_full_price_tickets_l1005_100523


namespace NUMINAMATH_GPT_find_f_neg_a_l1005_100563

noncomputable def f (x : ℝ) : ℝ := (Real.exp x - Real.exp (-x)) / 2 + 3 * Real.sin x + 2

theorem find_f_neg_a (a : ℝ) (h : f a = 1) : f (-a) = 3 := by
  sorry

end NUMINAMATH_GPT_find_f_neg_a_l1005_100563


namespace NUMINAMATH_GPT_count_ordered_pairs_l1005_100540

theorem count_ordered_pairs (x y : ℕ) (px : 0 < x) (py : 0 < y) (h : 2310 = 2 * 3 * 5 * 7 * 11) :
  (x * y = 2310 → ∃ n : ℕ, n = 32) :=
by
  sorry

end NUMINAMATH_GPT_count_ordered_pairs_l1005_100540


namespace NUMINAMATH_GPT_oil_in_Tank_C_is_982_l1005_100519

-- Definitions of tank capacities and oil amounts
def capacity_A := 80
def capacity_B := 120
def capacity_C := 160
def capacity_D := 240

def total_oil_bought := 1387

def oil_in_A := 70
def oil_in_B := 95
def oil_in_D := capacity_D  -- Since Tank D is 100% full

-- Statement of the problem
theorem oil_in_Tank_C_is_982 :
  oil_in_A + oil_in_B + oil_in_D + (total_oil_bought - (oil_in_A + oil_in_B + oil_in_D)) = total_oil_bought :=
by
  sorry

end NUMINAMATH_GPT_oil_in_Tank_C_is_982_l1005_100519


namespace NUMINAMATH_GPT_janet_dresses_total_pockets_l1005_100585

theorem janet_dresses_total_pockets :
  ∃ dresses pockets pocket_2 pocket_3,
  dresses = 24 ∧ 
  pockets = dresses / 2 ∧ 
  pocket_2 = pockets / 3 ∧ 
  pocket_3 = pockets - pocket_2 ∧ 
  (pocket_2 * 2 + pocket_3 * 3) = 32 := by
    sorry

end NUMINAMATH_GPT_janet_dresses_total_pockets_l1005_100585


namespace NUMINAMATH_GPT_no_solution_for_x4_plus_y4_eq_z4_l1005_100520

theorem no_solution_for_x4_plus_y4_eq_z4 :
  ∀ (x y z : ℤ), x ≠ 0 → y ≠ 0 → z ≠ 0 → gcd (gcd x y) z = 1 → x^4 + y^4 ≠ z^4 :=
sorry

end NUMINAMATH_GPT_no_solution_for_x4_plus_y4_eq_z4_l1005_100520


namespace NUMINAMATH_GPT_range_of_m_common_tangents_with_opposite_abscissas_l1005_100587

section part1
variable {x : ℝ}

noncomputable def f (x : ℝ) := Real.exp x
noncomputable def h (m : ℝ) (x : ℝ) := m * f x / Real.sin x

theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 0 Real.pi, h m x ≥ Real.sqrt 2) ↔ m ∈ Set.Ici (Real.sqrt 2 / Real.exp (Real.pi / 4)) := 
by
  sorry
end part1

section part2
variable {x : ℝ}

noncomputable def g (x : ℝ) := Real.log x
noncomputable def f_tangent_line_at (x₁ : ℝ) (x : ℝ) := Real.exp x₁ * x + (1 - x₁) * Real.exp x₁
noncomputable def g_tangent_line_at (x₂ : ℝ) (x : ℝ) := x / x₂ + Real.log x₂ - 1

theorem common_tangents_with_opposite_abscissas :
  ∃ x₁ x₂ : ℝ, (f_tangent_line_at x₁ = g_tangent_line_at (Real.exp (-x₁))) ∧ (x₁ = -x₂) :=
by
  sorry
end part2

end NUMINAMATH_GPT_range_of_m_common_tangents_with_opposite_abscissas_l1005_100587


namespace NUMINAMATH_GPT_ellipse_equation_hyperbola_equation_l1005_100513

/-- Ellipse problem -/
def ellipse_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_equation (e a c b : ℝ) (h_c : c = 3) (h_e : e = 0.5) (h_a : a = 6) (h_b : b^2 = 27) :
  ellipse_eq x y a b := 
sorry

/-- Hyperbola problem -/
def hyperbola_eq (x y a b : ℝ) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

theorem hyperbola_equation (a b c : ℝ) 
  (h_c : c = 6) 
  (h_A : ∀ (x y : ℝ), (x, y) = (-5, 2) → hyperbola_eq x y a b) 
  (h_eq1 : a^2 + b^2 = 36) 
  (h_eq2 : 25 / (a^2) - 4 / (b^2) = 1) :
  hyperbola_eq x y a b :=
sorry

end NUMINAMATH_GPT_ellipse_equation_hyperbola_equation_l1005_100513


namespace NUMINAMATH_GPT_green_beans_weight_l1005_100589

/-- 
    Mary uses plastic grocery bags that can hold a maximum of twenty pounds. 
    She buys some green beans, 6 pounds milk, and twice the amount of carrots as green beans. 
    She can fit 2 more pounds of groceries in that bag. 
    Prove that the weight of green beans she bought is equal to 4 pounds.
-/
theorem green_beans_weight (G : ℕ) (H1 : ∀ g : ℕ, g + 6 + 2 * g ≤ 20 - 2) : G = 4 :=
by 
  have H := H1 4
  sorry

end NUMINAMATH_GPT_green_beans_weight_l1005_100589
