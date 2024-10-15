import Mathlib

namespace NUMINAMATH_GPT_total_wicks_l2216_221630

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end NUMINAMATH_GPT_total_wicks_l2216_221630


namespace NUMINAMATH_GPT_repeating_prime_exists_l2216_221607

open Nat

theorem repeating_prime_exists (p : Fin 2021 → ℕ) 
  (prime_seq : ∀ i : Fin 2021, Nat.Prime (p i))
  (diff_condition : ∀ i : Fin 2019, (p (i + 1) - p i = 6 ∨ p (i + 1) - p i = 12) ∧ (p (i + 2) - p (i + 1) = 6 ∨ p (i + 2) - p (i + 1) = 12)) : 
  ∃ i j : Fin 2021, i ≠ j ∧ p i = p j := by
  sorry

end NUMINAMATH_GPT_repeating_prime_exists_l2216_221607


namespace NUMINAMATH_GPT_find_intersection_l2216_221643

def A : Set ℝ := {x | abs (x + 1) = x + 1}

def B : Set ℝ := {x | x^2 + x < 0}

def intersection (A B : Set ℝ) : Set ℝ := {x | x ∈ A ∧ x ∈ B}

theorem find_intersection : intersection A B = {x | -1 < x ∧ x < 0} :=
by
  sorry

end NUMINAMATH_GPT_find_intersection_l2216_221643


namespace NUMINAMATH_GPT_sum_of_7a_and_3b_l2216_221620

theorem sum_of_7a_and_3b (a b : ℤ) (h : a + b = 1998) : 7 * a + 3 * b ≠ 6799 :=
by sorry

end NUMINAMATH_GPT_sum_of_7a_and_3b_l2216_221620


namespace NUMINAMATH_GPT_common_point_geometric_lines_l2216_221663

-- Define that a, b, c form a geometric progression given common ratio r
def geometric_prog (a b c r : ℝ) : Prop := b = a * r ∧ c = a * r^2

-- Prove that all lines with the equation ax + by = c pass through the point (-1, 1)
theorem common_point_geometric_lines (a b c r x y : ℝ) (h : geometric_prog a b c r) :
  a * x + b * y = c → (x, y) = (-1, 1) :=
by
  sorry

end NUMINAMATH_GPT_common_point_geometric_lines_l2216_221663


namespace NUMINAMATH_GPT_binom_13_11_eq_78_l2216_221602

theorem binom_13_11_eq_78 : Nat.choose 13 11 = 78 := by
  sorry

end NUMINAMATH_GPT_binom_13_11_eq_78_l2216_221602


namespace NUMINAMATH_GPT_percentage_selected_B_l2216_221692

-- Definitions for the given conditions
def candidates := 7900
def selected_A := (6 / 100) * candidates
def selected_B := selected_A + 79

-- The question to be answered
def P_B := (selected_B / candidates) * 100

-- Proof statement
theorem percentage_selected_B : P_B = 7 := 
by
  -- Canonical statement placeholder 
  sorry

end NUMINAMATH_GPT_percentage_selected_B_l2216_221692


namespace NUMINAMATH_GPT_tan_240_eq_sqrt3_l2216_221612

theorem tan_240_eq_sqrt3 : Real.tan (240 * Real.pi / 180) = Real.sqrt 3 :=
by sorry

end NUMINAMATH_GPT_tan_240_eq_sqrt3_l2216_221612


namespace NUMINAMATH_GPT_calculate_expression_l2216_221644

theorem calculate_expression
  (x y : ℚ)
  (D E : ℚ × ℚ)
  (hx : x = (D.1 + E.1) / 2)
  (hy : y = (D.2 + E.2) / 2)
  (hD : D = (15, -3))
  (hE : E = (-4, 12)) :
  3 * x - 5 * y = -6 :=
by
  subst hD
  subst hE
  subst hx
  subst hy
  sorry

end NUMINAMATH_GPT_calculate_expression_l2216_221644


namespace NUMINAMATH_GPT_no_friendly_triplet_in_range_l2216_221671

open Nat

def isFriendly (a b c : ℕ) : Prop :=
  (a ∣ (b * c) ∨ b ∣ (a * c) ∨ c ∣ (a * b))

theorem no_friendly_triplet_in_range (n : ℕ) (a b c : ℕ) :
  n^2 < a ∧ a < n^2 + n → n^2 < b ∧ b < n^2 + n → n^2 < c ∧ c < n^2 + n → a ≠ b → b ≠ c → a ≠ c →
  ¬ isFriendly a b c :=
by sorry

end NUMINAMATH_GPT_no_friendly_triplet_in_range_l2216_221671


namespace NUMINAMATH_GPT_find_a_l2216_221645

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end NUMINAMATH_GPT_find_a_l2216_221645


namespace NUMINAMATH_GPT_repeating_decmials_sum_is_fraction_l2216_221673

noncomputable def x : ℚ := 2/9
noncomputable def y : ℚ := 2/99
noncomputable def z : ℚ := 2/9999

theorem repeating_decmials_sum_is_fraction :
  (x + y + z) = 2426 / 9999 := by
  sorry

end NUMINAMATH_GPT_repeating_decmials_sum_is_fraction_l2216_221673


namespace NUMINAMATH_GPT_game_show_prize_guess_l2216_221609

noncomputable def total_possible_guesses : ℕ :=
  (Nat.choose 8 3) * (Nat.choose 5 3) * (Nat.choose 2 2) * (Nat.choose 7 3)

theorem game_show_prize_guess :
  total_possible_guesses = 19600 :=
by
  -- Omitted proof steps
  sorry

end NUMINAMATH_GPT_game_show_prize_guess_l2216_221609


namespace NUMINAMATH_GPT_bakery_total_items_l2216_221694

theorem bakery_total_items (total_money : ℝ) (cupcake_cost : ℝ) (pastry_cost : ℝ) (max_cupcakes : ℕ) (remaining_money : ℝ) (total_items : ℕ) :
  total_money = 50 ∧ cupcake_cost = 3 ∧ pastry_cost = 2.5 ∧ max_cupcakes = 16 ∧ remaining_money = 2 ∧ total_items = max_cupcakes + 0 → total_items = 16 :=
by
  sorry

end NUMINAMATH_GPT_bakery_total_items_l2216_221694


namespace NUMINAMATH_GPT_maxEccentricity_l2216_221682

noncomputable def majorAxisLength := 4
noncomputable def majorSemiAxis := 2
noncomputable def leftVertexParabolaEq (y : ℝ) := y^2 = -3
noncomputable def distanceCondition (c : ℝ) := 2^2 / c - 2 ≥ 1

theorem maxEccentricity : ∃ c : ℝ, distanceCondition c ∧ (c ≤ 4 / 3) ∧ (c / majorSemiAxis = 2 / 3) :=
by
  sorry

end NUMINAMATH_GPT_maxEccentricity_l2216_221682


namespace NUMINAMATH_GPT_total_journey_distance_l2216_221614

-- Definitions of the conditions

def journey_time : ℝ := 40
def first_half_speed : ℝ := 20
def second_half_speed : ℝ := 30

-- Proof statement
theorem total_journey_distance : ∃ D : ℝ, (D / first_half_speed + D / second_half_speed = journey_time) ∧ (D = 960) :=
by 
  sorry

end NUMINAMATH_GPT_total_journey_distance_l2216_221614


namespace NUMINAMATH_GPT_find_least_q_l2216_221660

theorem find_least_q : 
  ∃ q : ℕ, 
    (q ≡ 0 [MOD 7]) ∧ 
    (q ≥ 1000) ∧ 
    (q ≡ 1 [MOD 3]) ∧ 
    (q ≡ 1 [MOD 4]) ∧ 
    (q ≡ 1 [MOD 5]) ∧ 
    (q = 1141) :=
by
  sorry

end NUMINAMATH_GPT_find_least_q_l2216_221660


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2216_221635

theorem solution_set_of_inequality:
  {x : ℝ | |x - 5| + |x + 1| < 8} = {x : ℝ | -2 < x ∧ x < 6} :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2216_221635


namespace NUMINAMATH_GPT_children_per_block_l2216_221657

theorem children_per_block {children total_blocks : ℕ} 
  (h_total_blocks : total_blocks = 9) 
  (h_total_children : children = 54) : 
  (children / total_blocks = 6) :=
by
  -- Definitions from conditions
  have h1 : total_blocks = 9 := h_total_blocks
  have h2 : children = 54 := h_total_children

  -- Goal to prove
  -- children / total_blocks = 6
  sorry

end NUMINAMATH_GPT_children_per_block_l2216_221657


namespace NUMINAMATH_GPT_heroes_can_reduce_heads_to_zero_l2216_221646

-- Definition of the Hero strikes
def IlyaMurometsStrikes (H : ℕ) : ℕ := H / 2 - 1
def DobrynyaNikitichStrikes (H : ℕ) : ℕ := 2 * H / 3 - 2
def AlyoshaPopovichStrikes (H : ℕ) : ℕ := 3 * H / 4 - 3

-- The ultimate goal is proving this theorem
theorem heroes_can_reduce_heads_to_zero (H : ℕ) : 
  ∃ (n : ℕ), ∀ i ≤ n, 
  (if i % 3 = 0 then H = 0 
   else if i % 3 = 1 then IlyaMurometsStrikes H = 0 
   else if i % 3 = 2 then DobrynyaNikitichStrikes H = 0 
   else AlyoshaPopovichStrikes H = 0)
:= sorry

end NUMINAMATH_GPT_heroes_can_reduce_heads_to_zero_l2216_221646


namespace NUMINAMATH_GPT_range_of_a_plus_b_l2216_221637

theorem range_of_a_plus_b 
  (a b : ℝ)
  (h : ∀ x : ℝ, a * Real.cos x + b * Real.cos (2 * x) ≥ -1) : 
  -1 ≤ a + b ∧ a + b ≤ 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_plus_b_l2216_221637


namespace NUMINAMATH_GPT_no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l2216_221699

theorem no_solution_for_x_y_z (a : ℕ) : 
  ¬ ∃ (x y z : ℚ), x^2 + y^2 + z^2 = 8 * a + 7 :=
by
  sorry

theorem seven_n_plus_eight_is_perfect_square (n : ℕ) :
  ∃ x : ℕ, 7^n + 8 = x^2 ↔ n = 0 :=
by
  sorry

end NUMINAMATH_GPT_no_solution_for_x_y_z_seven_n_plus_eight_is_perfect_square_l2216_221699


namespace NUMINAMATH_GPT_total_days_off_l2216_221640

-- Definitions for the problem conditions
def days_off_personal (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_professional (months_in_year : ℕ) (days_per_month : ℕ) : ℕ :=
  days_per_month * months_in_year

def days_off_teambuilding (quarters_in_year : ℕ) (days_per_quarter : ℕ) : ℕ :=
  days_per_quarter * quarters_in_year

-- Main theorem to prove
theorem total_days_off
  (months_in_year : ℕ) (quarters_in_year : ℕ)
  (days_per_month_personal : ℕ) (days_per_month_professional : ℕ) (days_per_quarter_teambuilding: ℕ)
  (h_months : months_in_year = 12) (h_quarters : quarters_in_year = 4) 
  (h_days_personal : days_per_month_personal = 4) (h_days_professional : days_per_month_professional = 2) (h_days_teambuilding : days_per_quarter_teambuilding = 1) :
  days_off_personal months_in_year days_per_month_personal
  + days_off_professional months_in_year days_per_month_professional
  + days_off_teambuilding quarters_in_year days_per_quarter_teambuilding
  = 76 := 
by {
  -- Calculation
  sorry
}

end NUMINAMATH_GPT_total_days_off_l2216_221640


namespace NUMINAMATH_GPT_max_value_f1_l2216_221691

-- Definitions for the conditions
def f (x a b : ℝ) : ℝ := x^2 + a * b * x + a + 2 * b

-- Lean theorem statements
theorem max_value_f1 (a b : ℝ) (h : a + 2 * b = 4) :
  f 0 a b = 4 → f 1 a b ≤ 7 :=
sorry

end NUMINAMATH_GPT_max_value_f1_l2216_221691


namespace NUMINAMATH_GPT_vote_ratio_l2216_221648

theorem vote_ratio (X Y Z : ℕ) (hZ : Z = 25000) (hX : X = 22500) (hX_Y : X = Y + (1/2 : ℚ) * Y) 
    : Y / (Z - Y) = 2 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_vote_ratio_l2216_221648


namespace NUMINAMATH_GPT_deepak_present_age_l2216_221684

-- Define the variables R and D
variables (R D : ℕ)

-- The conditions:
-- 1. After 4 years, Rahul's age will be 32 years.
-- 2. The ratio between Rahul and Deepak's ages is 4:3.
def rahul_age_after_4 : Prop := R + 4 = 32
def age_ratio : Prop := R / D = 4 / 3

-- The statement we want to prove:
theorem deepak_present_age (h1 : rahul_age_after_4 R) (h2 : age_ratio R D) : D = 21 :=
by sorry

end NUMINAMATH_GPT_deepak_present_age_l2216_221684


namespace NUMINAMATH_GPT_solve_inequalities_l2216_221628

theorem solve_inequalities (x : ℤ) :
  (1 ≤ x ∧ x < 3) ↔ 
  ((↑x - 1) / 2 < (↑x : ℝ) / 3 ∧ 2 * (↑x : ℝ) - 5 ≤ 3 * (↑x : ℝ) - 6) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequalities_l2216_221628


namespace NUMINAMATH_GPT_top_triangle_is_multiple_of_5_l2216_221610

-- Definitions of the conditions given in the problem

def lower_left_triangle := 12
def lower_right_triangle := 3

-- Let a, b, c, d be the four remaining numbers in the bottom row
variables (a b c d : ℤ)

-- Conditions that the sums of triangles must be congruent to multiples of 5
def second_lowest_row : Prop :=
  (3 - a) % 5 = 0 ∧
  (-a - b) % 5 = 0 ∧
  (-b - c) % 5 = 0 ∧
  (-c - d) % 5 = 0 ∧
  (2 - d) % 5 = 0

def third_lowest_row : Prop :=
  (2 + 2*a + b) % 5 = 0 ∧
  (a + 2*b + c) % 5 = 0 ∧
  (b + 2*c + d) % 5 = 0 ∧
  (3 + c + 2*d) % 5 = 0

def fourth_lowest_row : Prop :=
  (3 + 2*a + 2*b - c) % 5 = 0 ∧
  (-a + 2*b + 2*c - d) % 5 = 0 ∧
  (2 - b + 2*c + 2*d) % 5 = 0

def second_highest_row : Prop :=
  (2 - a + b - c + d) % 5 = 0 ∧
  (3 + a - b + c - d) % 5 = 0

def top_triangle : Prop :=
  (2 - a + b - c + d + 3 + a - b + c - d) % 5 = 0

theorem top_triangle_is_multiple_of_5 (a b c d : ℤ) :
  second_lowest_row a b c d →
  third_lowest_row a b c d →
  fourth_lowest_row a b c d →
  second_highest_row a b c d →
  top_triangle a b c d →
  ∃ k : ℤ, (2 - a + b - c + d + 3 + a - b + c - d) = 5 * k :=
by sorry

end NUMINAMATH_GPT_top_triangle_is_multiple_of_5_l2216_221610


namespace NUMINAMATH_GPT_jinhee_pages_per_day_l2216_221653

noncomputable def pages_per_day (total_pages : ℕ) (days : ℕ) : ℕ :=
  (total_pages + days - 1) / days

theorem jinhee_pages_per_day : 
  ∀ (total_pages : ℕ) (days : ℕ), total_pages = 220 → days = 7 → pages_per_day total_pages days = 32 :=
by 
  intros total_pages days hp hd
  rw [hp, hd]
  -- the computation of the function
  show pages_per_day 220 7 = 32
  sorry

end NUMINAMATH_GPT_jinhee_pages_per_day_l2216_221653


namespace NUMINAMATH_GPT_similar_triangles_PQ_length_l2216_221668

theorem similar_triangles_PQ_length (XY YZ QR : ℝ) (hXY : XY = 8) (hYZ : YZ = 16) (hQR : QR = 24)
  (hSimilar : ∃ (k : ℝ), XY = k * 8 ∧ YZ = k * 16 ∧ QR = k * 24) : (∃ (PQ : ℝ), PQ = 12) :=
by 
  -- Here we need to prove the theorem using similarity and given equalities
  sorry

end NUMINAMATH_GPT_similar_triangles_PQ_length_l2216_221668


namespace NUMINAMATH_GPT_bridge_length_is_219_l2216_221664

noncomputable def length_of_bridge (train_length : ℕ) (train_speed_kmh : ℤ) (time_seconds : ℕ) : ℝ :=
  let train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
  let total_distance : ℝ := train_speed_ms * time_seconds
  total_distance - train_length

theorem bridge_length_is_219 :
  length_of_bridge 156 45 30 = 219 :=
by
  sorry

end NUMINAMATH_GPT_bridge_length_is_219_l2216_221664


namespace NUMINAMATH_GPT_part1_part2_l2216_221662

-- Define the quadratic equation and its discriminant
def quadratic_discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

-- Define the conditions
def quadratic_equation (m : ℝ) : ℝ :=
  quadratic_discriminant 1 (-2) (-3 * m^2)

-- Part 1: Prove the quadratic equation always has two distinct real roots
theorem part1 (m : ℝ) : 
  quadratic_equation m > 0 :=
by
  sorry

-- Part 2: Find the value of m given the roots satisfy the equation α + 2β = 5
theorem part2 (α β m : ℝ) (h1 : α + β = 2) (h2 : α + 2 * β = 5) : 
  m = 1 ∨ m = -1 :=
by
  sorry


end NUMINAMATH_GPT_part1_part2_l2216_221662


namespace NUMINAMATH_GPT_fraction_of_full_tank_used_l2216_221618

-- Define the initial conditions as per the problem statement
def speed : ℝ := 50 -- miles per hour
def time : ℝ := 5   -- hours
def miles_per_gallon : ℝ := 30
def full_tank_capacity : ℝ := 15 -- gallons

-- We need to prove that the fraction of gasoline used is 5/9
theorem fraction_of_full_tank_used : 
  ((speed * time) / miles_per_gallon) / full_tank_capacity = 5 / 9 := by
sorry

end NUMINAMATH_GPT_fraction_of_full_tank_used_l2216_221618


namespace NUMINAMATH_GPT_crackers_shared_equally_l2216_221631

theorem crackers_shared_equally : ∀ (matthew_crackers friends_crackers left_crackers friends : ℕ),
  matthew_crackers = 23 →
  left_crackers = 11 →
  friends = 2 →
  matthew_crackers - left_crackers = friends_crackers →
  friends_crackers = friends * 6 :=
by
  intro matthew_crackers friends_crackers left_crackers friends
  sorry

end NUMINAMATH_GPT_crackers_shared_equally_l2216_221631


namespace NUMINAMATH_GPT_calculate_AH_l2216_221676

def square (a : ℝ) := a ^ 2
def area_square (s : ℝ) := s ^ 2
def area_triangle (b h : ℝ) := 0.5 * b * h

theorem calculate_AH (s DG DH AH : ℝ) 
  (h_square : area_square s = 144) 
  (h_area_triangle : area_triangle DG DH = 63)
  (h_perpendicular : DG = DH)
  (h_hypotenuse : square AH = square s + square DH) :
  AH = 3 * Real.sqrt 30 :=
by
  -- Proof would be provided here
  sorry

end NUMINAMATH_GPT_calculate_AH_l2216_221676


namespace NUMINAMATH_GPT_university_students_l2216_221627

theorem university_students (total_students students_both math_students physics_students : ℕ) 
  (h1 : total_students = 75) 
  (h2 : total_students = (math_students - students_both) + (physics_students - students_both) + students_both)
  (h3 : math_students = 2 * physics_students) 
  (h4 : students_both = 10) : 
  math_students = 56 := by
  sorry

end NUMINAMATH_GPT_university_students_l2216_221627


namespace NUMINAMATH_GPT_solve_for_y_l2216_221688

theorem solve_for_y (y : ℚ) (h : 1 / 3 + 1 / y = 7 / 9) : y = 9 / 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_y_l2216_221688


namespace NUMINAMATH_GPT_odd_function_m_zero_l2216_221679

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := 2 * x^3 + m

theorem odd_function_m_zero (m : ℝ) : (∀ x : ℝ, f (-x) m = -f x m) → m = 0 :=
by
  sorry

end NUMINAMATH_GPT_odd_function_m_zero_l2216_221679


namespace NUMINAMATH_GPT_football_club_initial_balance_l2216_221624

noncomputable def initial_balance (final_balance income expense : ℕ) : ℕ :=
  final_balance + income - expense

theorem football_club_initial_balance :
  initial_balance 60 (2 * 10) (4 * 15) = 20 := by
sorry

end NUMINAMATH_GPT_football_club_initial_balance_l2216_221624


namespace NUMINAMATH_GPT_fraction_water_by_volume_l2216_221656

theorem fraction_water_by_volume
  (A W : ℝ) 
  (h1 : A / W = 0.5)
  (h2 : A / (A + W) = 1/7) : 
  W / (A + W) = 2/7 :=
by
  sorry

end NUMINAMATH_GPT_fraction_water_by_volume_l2216_221656


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l2216_221696

theorem perpendicular_lines_condition (A1 B1 C1 A2 B2 C2 : ℝ) :
  (A1 * A2 + B1 * B2 = 0) ↔ (A1 * A2) / (B1 * B2) = -1 := sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l2216_221696


namespace NUMINAMATH_GPT_one_corresponds_to_36_l2216_221622

-- Define the given conditions
def corresponds (n : Nat) (s : String) : Prop :=
match n with
| 2  => s = "36"
| 3  => s = "363"
| 4  => s = "364"
| 5  => s = "365"
| 36 => s = "2"
| _  => False

-- Statement for the proof problem: Prove that 1 corresponds to 36
theorem one_corresponds_to_36 : corresponds 1 "36" :=
by
  sorry

end NUMINAMATH_GPT_one_corresponds_to_36_l2216_221622


namespace NUMINAMATH_GPT_time_to_meet_l2216_221604

variable (distance : ℕ)
variable (speed1 speed2 time : ℕ)

-- Given conditions
def distanceAB := 480
def speedPassengerCar := 65
def speedCargoTruck := 55

-- Sum of the speeds of the two vehicles
def sumSpeeds := speedPassengerCar + speedCargoTruck

-- Prove that the time it takes for the two vehicles to meet is 4 hours
theorem time_to_meet : sumSpeeds * time = distanceAB → time = 4 :=
by
  sorry

end NUMINAMATH_GPT_time_to_meet_l2216_221604


namespace NUMINAMATH_GPT_unobserved_planet_exists_l2216_221658

theorem unobserved_planet_exists
  (n : ℕ) (h_n_eq : n = 15)
  (planets : Fin n → Type)
  (dist : ∀ (i j : Fin n), ℝ)
  (h_distinct : ∀ (i j : Fin n), i ≠ j → dist i j ≠ dist j i)
  (nearest : ∀ i : Fin n, Fin n)
  (h_nearest : ∀ i : Fin n, nearest i ≠ i)
  : ∃ i : Fin n, ∀ j : Fin n, nearest j ≠ i := by
  sorry

end NUMINAMATH_GPT_unobserved_planet_exists_l2216_221658


namespace NUMINAMATH_GPT_dividend_calculation_l2216_221652

theorem dividend_calculation (divisor quotient remainder dividend : ℕ)
  (h1 : divisor = 36)
  (h2 : quotient = 20)
  (h3 : remainder = 5)
  (h4 : dividend = (divisor * quotient) + remainder)
  : dividend = 725 := 
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_dividend_calculation_l2216_221652


namespace NUMINAMATH_GPT_brianne_january_savings_l2216_221605

theorem brianne_january_savings (S : ℝ) (h : 16 * S = 160) : S = 10 :=
sorry

end NUMINAMATH_GPT_brianne_january_savings_l2216_221605


namespace NUMINAMATH_GPT_shelves_full_percentage_l2216_221693

-- Define the conditions as constants
def ridges_per_record : Nat := 60
def cases : Nat := 4
def shelves_per_case : Nat := 3
def records_per_shelf : Nat := 20
def total_ridges : Nat := 8640

-- Define the total number of records
def total_records := total_ridges / ridges_per_record

-- Define the total capacity of the shelves
def total_capacity := cases * shelves_per_case * records_per_shelf

-- Define the percentage of shelves that are full
def percentage_full := (total_records * 100) / total_capacity

-- State the theorem that the percentage of the shelves that are full is 60%
theorem shelves_full_percentage : percentage_full = 60 := 
by
  sorry

end NUMINAMATH_GPT_shelves_full_percentage_l2216_221693


namespace NUMINAMATH_GPT_range_of_m_l2216_221636

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x^2 + 2 * x + m > 0) → m > 1 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_range_of_m_l2216_221636


namespace NUMINAMATH_GPT_percentage_increase_each_job_l2216_221650

-- Definitions of original and new amounts for each job as given conditions
def original_first_job : ℝ := 65
def new_first_job : ℝ := 70

def original_second_job : ℝ := 240
def new_second_job : ℝ := 315

def original_third_job : ℝ := 800
def new_third_job : ℝ := 880

-- Proof problem statement
theorem percentage_increase_each_job :
  (new_first_job - original_first_job) / original_first_job * 100 = 7.69 ∧
  (new_second_job - original_second_job) / original_second_job * 100 = 31.25 ∧
  (new_third_job - original_third_job) / original_third_job * 100 = 10 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_each_job_l2216_221650


namespace NUMINAMATH_GPT_sum_of_interior_angles_n_plus_3_l2216_221687

-- Define the condition that the sum of the interior angles of a convex polygon with n sides is 1260 degrees
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Prove that given the above condition for n, the sum of the interior angles of a convex polygon with n + 3 sides is 1800 degrees
theorem sum_of_interior_angles_n_plus_3 (n : ℕ) (h : sum_of_interior_angles n = 1260) : 
  sum_of_interior_angles (n + 3) = 1800 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_n_plus_3_l2216_221687


namespace NUMINAMATH_GPT_distance_they_both_run_l2216_221686

theorem distance_they_both_run
  (time_A time_B : ℕ)
  (distance_advantage: ℝ)
  (speed_A speed_B : ℝ)
  (D : ℝ) :
  time_A = 198 →
  time_B = 220 →
  distance_advantage = 300 →
  speed_A = D / time_A →
  speed_B = D / time_B →
  speed_A * time_B = D + distance_advantage →
  D = 2700 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_distance_they_both_run_l2216_221686


namespace NUMINAMATH_GPT_similar_triangles_x_value_l2216_221647

theorem similar_triangles_x_value : ∃ (x : ℝ), (12 / x = 9 / 6) ∧ x = 8 := by
  use 8
  constructor
  · sorry
  · rfl

end NUMINAMATH_GPT_similar_triangles_x_value_l2216_221647


namespace NUMINAMATH_GPT_initial_coins_l2216_221659

-- Define the condition for the initial number of coins
variable (x : Nat) -- x represents the initial number of coins

-- The main statement theorem that needs proof
theorem initial_coins (h : x + 8 = 29) : x = 21 := 
by { sorry } -- placeholder for the proof

end NUMINAMATH_GPT_initial_coins_l2216_221659


namespace NUMINAMATH_GPT_tan_75_degrees_eq_l2216_221632

noncomputable def tan_75_degrees : ℝ := Real.tan (75 * Real.pi / 180)

theorem tan_75_degrees_eq : tan_75_degrees = 2 + Real.sqrt 3 := by
  sorry

end NUMINAMATH_GPT_tan_75_degrees_eq_l2216_221632


namespace NUMINAMATH_GPT_ap_number_of_terms_l2216_221600

theorem ap_number_of_terms (a d : ℕ) (n : ℕ) (ha1 : (n - 1) * d = 12) (ha2 : a + 2 * d = 6)
  (h_odd_sum : (n / 2) * (2 * a + (n - 2) * d) = 36) (h_even_sum : (n / 2) * (2 * a + n * d) = 42) :
    n = 12 :=
by
  sorry

end NUMINAMATH_GPT_ap_number_of_terms_l2216_221600


namespace NUMINAMATH_GPT_num_even_multiple_5_perfect_squares_lt_1000_l2216_221639

theorem num_even_multiple_5_perfect_squares_lt_1000 : 
  ∃ n, n = 3 ∧ ∀ x, (x < 1000) ∧ (x > 0) ∧ (∃ k, x = 100 * k^2) → (n = 3) := by 
  sorry

end NUMINAMATH_GPT_num_even_multiple_5_perfect_squares_lt_1000_l2216_221639


namespace NUMINAMATH_GPT_remainder_when_divided_by_7_l2216_221616

theorem remainder_when_divided_by_7 :
  let a := -1234
  let b := 1984
  let c := -1460
  let d := 2008
  (a * b * c * d) % 7 = 0 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_7_l2216_221616


namespace NUMINAMATH_GPT_three_digit_number_value_l2216_221641

theorem three_digit_number_value (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) 
    (h4 : a > b) (h5 : b > c)
    (h6 : (10 * a + b) + (10 * b + a) = 55)  
    (h7 : 1300 < 222 * (a + b + c) ∧ 222 * (a + b + c) < 1400) : 
    (100 * a + 10 * b + c) = 321 := 
sorry

end NUMINAMATH_GPT_three_digit_number_value_l2216_221641


namespace NUMINAMATH_GPT_volume_of_snow_l2216_221601

theorem volume_of_snow (L W H : ℝ) (hL : L = 30) (hW : W = 3) (hH : H = 0.75) :
  L * W * H = 67.5 := by
  sorry

end NUMINAMATH_GPT_volume_of_snow_l2216_221601


namespace NUMINAMATH_GPT_sum_of_roots_l2216_221608

theorem sum_of_roots 
  (a b c : ℝ)
  (h1 : 1^2 + a * 1 + 2 = 0)
  (h2 : (∀ x : ℝ, x^2 + 5 * x + c = 0 → (x = a ∨ x = b))) :
  a + b + c = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_roots_l2216_221608


namespace NUMINAMATH_GPT_polynomial_evaluation_l2216_221675

theorem polynomial_evaluation (y : ℝ) (hy : y^2 - 3 * y - 9 = 0) : y^3 - 3 * y^2 - 9 * y + 7 = 7 := 
  sorry

end NUMINAMATH_GPT_polynomial_evaluation_l2216_221675


namespace NUMINAMATH_GPT_percent_yz_of_x_l2216_221617

theorem percent_yz_of_x (x y z : ℝ) 
  (h₁ : 0.6 * (x - y) = 0.3 * (x + y))
  (h₂ : 0.4 * (x + z) = 0.2 * (y + z))
  (h₃ : 0.5 * (x - z) = 0.25 * (x + y + z)) :
  y + z = 0.0 * x :=
sorry

end NUMINAMATH_GPT_percent_yz_of_x_l2216_221617


namespace NUMINAMATH_GPT_max_sum_length_le_98306_l2216_221633

noncomputable def L (k : ℕ) : ℕ := sorry

theorem max_sum_length_le_98306 (x y : ℕ) (hx : x > 1) (hy : y > 1) (hl : L x + L y = 16) : x + 3 * y < 98306 :=
sorry

end NUMINAMATH_GPT_max_sum_length_le_98306_l2216_221633


namespace NUMINAMATH_GPT_part_I_part_II_l2216_221670

noncomputable def f (x : ℝ) (a : ℝ) := Real.log x + x^2 - a * x

theorem part_I (x : ℝ) (a : ℝ) (h_inc : ∀ x > 0, (1/x + 2*x - a) ≥ 0) : a ≤ 2 * Real.sqrt 2 :=
sorry

noncomputable def g (x : ℝ) (a : ℝ) := f x a + 2 * Real.log ((a * x + 2) / (6 * Real.sqrt x))

theorem part_II (a : ℝ) (k : ℝ) (h_a : 2 < a ∧ a < 4) (h_ex : ∃ x : ℝ, (3/2) ≤ x ∧ x ≤ 2 ∧ g x a > k * (4 - a^2)) : k ≥ 1/3 :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l2216_221670


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l2216_221651

theorem geometric_sequence_common_ratio (a q : ℝ) (h : a = a * q / (1 - q)) : q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l2216_221651


namespace NUMINAMATH_GPT_car_R_average_speed_l2216_221615

theorem car_R_average_speed :
  ∃ (v : ℕ), (600 / v) - 2 = 600 / (v + 10) ∧ v = 50 :=
by sorry

end NUMINAMATH_GPT_car_R_average_speed_l2216_221615


namespace NUMINAMATH_GPT_total_vessels_l2216_221680

theorem total_vessels (C G S F : ℕ) (h1 : C = 4) (h2 : G = 2 * C) (h3 : S = G + 6) (h4 : S = 7 * F) : 
  C + G + S + F = 28 :=
by
  sorry

end NUMINAMATH_GPT_total_vessels_l2216_221680


namespace NUMINAMATH_GPT_smaller_angle_at_3_45_l2216_221603

def minute_hand_angle : ℝ := 270
def hour_hand_angle : ℝ := 90 + 0.75 * 30

theorem smaller_angle_at_3_45 :
  min (|minute_hand_angle - hour_hand_angle|) (360 - |minute_hand_angle - hour_hand_angle|) = 202.5 := 
by
  sorry

end NUMINAMATH_GPT_smaller_angle_at_3_45_l2216_221603


namespace NUMINAMATH_GPT_complement_intersection_l2216_221672

noncomputable def M : Set ℝ := {x | 2 / x < 1}
noncomputable def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

theorem complement_intersection : 
  ((Set.univ \ M) ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l2216_221672


namespace NUMINAMATH_GPT_square_minus_self_divisible_by_2_l2216_221654

theorem square_minus_self_divisible_by_2 (a : ℕ) : 2 ∣ (a^2 - a) :=
by sorry

end NUMINAMATH_GPT_square_minus_self_divisible_by_2_l2216_221654


namespace NUMINAMATH_GPT_tan_sub_theta_cos_double_theta_l2216_221638

variables (θ : ℝ)

-- Condition: given tan θ = 2
axiom tan_theta_eq_two : Real.tan θ = 2

-- Proof problem 1: Prove tan (π/4 - θ) = -1/3
theorem tan_sub_theta (h : Real.tan θ = 2) : Real.tan (Real.pi / 4 - θ) = -1/3 :=
by sorry

-- Proof problem 2: Prove cos 2θ = -3/5
theorem cos_double_theta (h : Real.tan θ = 2) : Real.cos (2 * θ) = -3/5 :=
by sorry

end NUMINAMATH_GPT_tan_sub_theta_cos_double_theta_l2216_221638


namespace NUMINAMATH_GPT_LCM_4_6_15_is_60_l2216_221623

def prime_factors (n : ℕ) : List ℕ :=
  [] -- placeholder, definition of prime_factor is not necessary for the problem statement, so we leave it abstract

def LCM (a b : ℕ) : ℕ := 
  sorry -- placeholder, definition of LCM not directly necessary for the statement

theorem LCM_4_6_15_is_60 : LCM (LCM 4 6) 15 = 60 := 
  sorry

end NUMINAMATH_GPT_LCM_4_6_15_is_60_l2216_221623


namespace NUMINAMATH_GPT_find_b_l2216_221626

variable (x : ℝ)

noncomputable def d : ℝ := 3

theorem find_b (b c : ℝ) :
  (7 * x^2 - 5 * x + 11 / 4) * (d * x^2 + b * x + c) = 21 * x^4 - 26 * x^3 + 34 * x^2 - 55 / 4 * x + 33 / 4 →
  b = -11 / 7 :=
by
  sorry

end NUMINAMATH_GPT_find_b_l2216_221626


namespace NUMINAMATH_GPT_octagon_perimeter_l2216_221625

-- Definitions based on conditions
def is_octagon (n : ℕ) : Prop := n = 8
def side_length : ℕ := 12

-- The proof problem statement
theorem octagon_perimeter (n : ℕ) (h : is_octagon n) : n * side_length = 96 := by
  sorry

end NUMINAMATH_GPT_octagon_perimeter_l2216_221625


namespace NUMINAMATH_GPT_minimum_value_of_f_l2216_221674

noncomputable def f (x m : ℝ) := (1 / 3) * x^3 - x + m

theorem minimum_value_of_f (m : ℝ) (h_max : f (-1) m = 1) : 
  f 1 m = -1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_f_l2216_221674


namespace NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l2216_221677

theorem solve_quadratic_1 : ∀ x : ℝ, x^2 - 5 * x + 4 = 0 ↔ x = 4 ∨ x = 1 :=
by sorry

theorem solve_quadratic_2 : ∀ x : ℝ, x^2 = 4 - 2 * x ↔ x = -1 + Real.sqrt 5 ∨ x = -1 - Real.sqrt 5 :=
by sorry

end NUMINAMATH_GPT_solve_quadratic_1_solve_quadratic_2_l2216_221677


namespace NUMINAMATH_GPT_cost_of_whistle_l2216_221685

theorem cost_of_whistle (cost_yoyo : ℕ) (total_spent : ℕ) (cost_yoyo_equals : cost_yoyo = 24) (total_spent_equals : total_spent = 38) : (total_spent - cost_yoyo) = 14 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_whistle_l2216_221685


namespace NUMINAMATH_GPT_school_growth_difference_l2216_221613

theorem school_growth_difference (X Y : ℕ) (H₁ : Y = 2400)
  (H₂ : X + Y = 4000) : (X + 7 * X / 100 - X) - (Y + 3 * Y / 100 - Y) = 40 :=
by
  sorry

end NUMINAMATH_GPT_school_growth_difference_l2216_221613


namespace NUMINAMATH_GPT_paul_mowing_lawns_l2216_221629

theorem paul_mowing_lawns : 
  ∃ M : ℕ, 
    (∃ money_made_weeating : ℕ, money_made_weeating = 13) ∧
    (∃ spending_per_week : ℕ, spending_per_week = 9) ∧
    (∃ weeks_last : ℕ, weeks_last = 9) ∧
    (M + 13 = 9 * 9) → 
    M = 68 := by
sorry

end NUMINAMATH_GPT_paul_mowing_lawns_l2216_221629


namespace NUMINAMATH_GPT_ratio_of_price_l2216_221683

-- Definitions from conditions
def original_price : ℝ := 3.00
def tom_pay_price : ℝ := 9.00

-- Theorem stating the ratio
theorem ratio_of_price : tom_pay_price / original_price = 3 := by
  sorry

end NUMINAMATH_GPT_ratio_of_price_l2216_221683


namespace NUMINAMATH_GPT_sum_of_three_squares_l2216_221655

-- Using the given conditions to define the problem.
variable (square triangle : ℝ)

-- Conditions
axiom h1 : square + triangle + 2 * square + triangle = 34
axiom h2 : triangle + square + triangle + 3 * square = 40

-- Statement to prove
theorem sum_of_three_squares : square + square + square = 66 / 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_squares_l2216_221655


namespace NUMINAMATH_GPT_valid_seating_arrangements_l2216_221665

def num_people : Nat := 10
def total_arrangements : Nat := Nat.factorial num_people
def restricted_group_arrangements : Nat := Nat.factorial 7 * Nat.factorial 4
def valid_arrangements : Nat := total_arrangements - restricted_group_arrangements

theorem valid_seating_arrangements : valid_arrangements = 3507840 := by
  sorry

end NUMINAMATH_GPT_valid_seating_arrangements_l2216_221665


namespace NUMINAMATH_GPT_total_gray_area_trees_l2216_221698

/-- 
Three aerial photos were taken by the drone, each capturing the same number of trees.
First rectangle has 100 trees in total and 82 trees in the white area.
Second rectangle has 90 trees in total and 82 trees in the white area.
Prove that the number of trees in gray areas in both rectangles is 26.
-/
theorem total_gray_area_trees : (100 - 82) + (90 - 82) = 26 := 
by sorry

end NUMINAMATH_GPT_total_gray_area_trees_l2216_221698


namespace NUMINAMATH_GPT_pie_eaten_after_four_trips_l2216_221689

theorem pie_eaten_after_four_trips : 
  let trip1 := (1 / 3 : ℝ)
  let trip2 := (1 / 3^2 : ℝ)
  let trip3 := (1 / 3^3 : ℝ)
  let trip4 := (1 / 3^4 : ℝ)
  trip1 + trip2 + trip3 + trip4 = (40 / 81 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_pie_eaten_after_four_trips_l2216_221689


namespace NUMINAMATH_GPT_evan_books_l2216_221661

theorem evan_books (B M : ℕ) (h1 : B = 200 - 40) (h2 : M * B + 60 = 860) : M = 5 :=
by {
  sorry  -- proof is omitted as per instructions
}

end NUMINAMATH_GPT_evan_books_l2216_221661


namespace NUMINAMATH_GPT_probability_even_sum_l2216_221678

theorem probability_even_sum (x y : ℕ) (h : x + y ≤ 10) : 
  (∃ (p : ℚ), p = 6 / 11 ∧ (x + y) % 2 = 0) :=
sorry

end NUMINAMATH_GPT_probability_even_sum_l2216_221678


namespace NUMINAMATH_GPT_probability_of_losing_l2216_221690

noncomputable def odds_of_winning : ℕ := 5
noncomputable def odds_of_losing : ℕ := 3
noncomputable def total_outcomes : ℕ := odds_of_winning + odds_of_losing

theorem probability_of_losing : 
  (odds_of_losing : ℚ) / (total_outcomes : ℚ) = 3 / 8 := 
by
  sorry

end NUMINAMATH_GPT_probability_of_losing_l2216_221690


namespace NUMINAMATH_GPT_xyz_zero_if_equation_zero_l2216_221695

theorem xyz_zero_if_equation_zero (x y z : ℚ) 
  (h : x^3 + 3 * y^3 + 9 * z^3 - 9 * x * y * z = 0) : 
  x = 0 ∧ y = 0 ∧ z = 0 := 
by 
  sorry

end NUMINAMATH_GPT_xyz_zero_if_equation_zero_l2216_221695


namespace NUMINAMATH_GPT_transformed_sum_l2216_221606

open BigOperators -- Open namespace to use big operators like summation

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ) 
  (h_sum : ∑ i, x i = s) : 
  ∑ i, ((3 * (x i + 10)) - 10) = 3 * s + 20 * n :=
by
  sorry

end NUMINAMATH_GPT_transformed_sum_l2216_221606


namespace NUMINAMATH_GPT_series_sum_l2216_221669

theorem series_sum :
  ∑' n : ℕ, (3 * (n + 1) - 2) / ((n + 1) * (n + 2) * (n + 4)) = (55 / 12) :=
sorry

end NUMINAMATH_GPT_series_sum_l2216_221669


namespace NUMINAMATH_GPT_sum_of_24_consecutive_integers_is_square_l2216_221666

theorem sum_of_24_consecutive_integers_is_square : ∃ n : ℕ, ∃ k : ℕ, (n > 0) ∧ (24 * (2 * n + 23)) = k * k ∧ k * k = 324 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_24_consecutive_integers_is_square_l2216_221666


namespace NUMINAMATH_GPT_heights_inscribed_circle_inequality_l2216_221667

theorem heights_inscribed_circle_inequality
  {h₁ h₂ r : ℝ} (h₁_pos : 0 < h₁) (h₂_pos : 0 < h₂) (r_pos : 0 < r)
  (triangle_heights : ∃ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c ∧ a * h₁ = b * h₂ ∧ 
                                       a + b > c ∧ h₁ = 2 * r * (a + b + c) / (a * b)):
  (1 / (2 * r) < 1 / h₁ + 1 / h₂ ∧ 1 / h₁ + 1 / h₂ < 1 / r) :=
sorry

end NUMINAMATH_GPT_heights_inscribed_circle_inequality_l2216_221667


namespace NUMINAMATH_GPT_mary_needs_to_add_l2216_221697

-- Define the conditions
def total_flour_required : ℕ := 7
def flour_already_added : ℕ := 2

-- Define the statement that corresponds to the mathematical equivalent proof problem
theorem mary_needs_to_add :
  total_flour_required - flour_already_added = 5 :=
by
  sorry

end NUMINAMATH_GPT_mary_needs_to_add_l2216_221697


namespace NUMINAMATH_GPT_sum_of_squares_l2216_221649

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 0)
  (h2 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5) (h3 : a^3 + b^3 + c^3 = a^7 + b^7 + c^7) :
  a^2 + b^2 + c^2 = 6 / 5 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l2216_221649


namespace NUMINAMATH_GPT_range_of_m_l2216_221681

theorem range_of_m (a b c : ℝ) (m : ℝ) (h1 : a > b) (h2 : b > c) (h3 : 1 / (a - b) + m / (b - c) ≥ 9 / (a - c)) :
  m ≥ 4 :=
sorry

end NUMINAMATH_GPT_range_of_m_l2216_221681


namespace NUMINAMATH_GPT_average_temperature_l2216_221619

theorem average_temperature (T_NY T_Miami T_SD : ℝ) (h1 : T_NY = 80) (h2 : T_Miami = T_NY + 10) (h3 : T_SD = T_Miami + 25) :
  (T_NY + T_Miami + T_SD) / 3 = 95 :=
by
  sorry

end NUMINAMATH_GPT_average_temperature_l2216_221619


namespace NUMINAMATH_GPT_range_of_half_alpha_minus_beta_l2216_221621

theorem range_of_half_alpha_minus_beta (α β : ℝ) (hα : 1 < α ∧ α < 3) (hβ : -4 < β ∧ β < 2) :
  -3 / 2 < (1 / 2) * α - β ∧ (1 / 2) * α - β < 11 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_half_alpha_minus_beta_l2216_221621


namespace NUMINAMATH_GPT_average_book_width_is_3_point_9375_l2216_221611

def book_widths : List ℚ := [3, 4, 3/4, 1.5, 7, 2, 5.25, 8]
def number_of_books : ℚ := 8
def total_width : ℚ := List.sum book_widths
def average_width : ℚ := total_width / number_of_books

theorem average_book_width_is_3_point_9375 :
  average_width = 3.9375 := by
  sorry

end NUMINAMATH_GPT_average_book_width_is_3_point_9375_l2216_221611


namespace NUMINAMATH_GPT_total_distance_traveled_l2216_221634

noncomputable def totalDistance
  (d1 d2 : ℝ) (s1 s2 : ℝ) (average_speed : ℝ) (total_time : ℝ) : ℝ := 
  average_speed * total_time

theorem total_distance_traveled :
  let d1 := 160
  let s1 := 64
  let d2 := 160
  let s2 := 80
  let average_speed := 71.11111111111111
  let total_time := d1 / s1 + d2 / s2
  totalDistance d1 d2 s1 s2 average_speed total_time = 320 :=
by
  -- This is the main statement theorem
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l2216_221634


namespace NUMINAMATH_GPT_wifi_cost_per_hour_l2216_221642

-- Define the conditions as hypotheses
def ticket_cost : ℝ := 11
def snacks_cost : ℝ := 3
def headphones_cost : ℝ := 16
def hourly_income : ℝ := 12
def trip_duration : ℝ := 3
def total_expenses : ℝ := ticket_cost + snacks_cost + headphones_cost
def total_earnings : ℝ := hourly_income * trip_duration

-- Translate the proof problem to Lean 4 statement
theorem wifi_cost_per_hour: 
  (total_earnings - total_expenses) / trip_duration = 2 :=
by sorry

end NUMINAMATH_GPT_wifi_cost_per_hour_l2216_221642
