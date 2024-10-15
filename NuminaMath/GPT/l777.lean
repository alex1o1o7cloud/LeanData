import Mathlib

namespace NUMINAMATH_GPT_base12_addition_l777_77722

theorem base12_addition : ∀ a b : ℕ, a = 956 ∧ b = 273 → (a + b) = 1009 := by
  sorry

end NUMINAMATH_GPT_base12_addition_l777_77722


namespace NUMINAMATH_GPT_test_total_points_l777_77796

theorem test_total_points (computation_points_per_problem : ℕ) (word_points_per_problem : ℕ) (total_problems : ℕ) (computation_problems : ℕ) :
  computation_points_per_problem = 3 →
  word_points_per_problem = 5 →
  total_problems = 30 →
  computation_problems = 20 →
  (computation_problems * computation_points_per_problem + 
  (total_problems - computation_problems) * word_points_per_problem) = 110 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_test_total_points_l777_77796


namespace NUMINAMATH_GPT_cliff_total_rocks_l777_77729

theorem cliff_total_rocks (I S : ℕ) (h1 : S = 2 * I) (h2 : I / 3 = 30) :
  I + S = 270 :=
sorry

end NUMINAMATH_GPT_cliff_total_rocks_l777_77729


namespace NUMINAMATH_GPT_smallest_portion_l777_77768

theorem smallest_portion
    (a_1 d : ℚ)
    (h1 : 5 * a_1 + 10 * d = 10)
    (h2 : (a_1 + 2 * d + a_1 + 3 * d + a_1 + 4 * d) / 7 = a_1 + a_1 + d) :
  a_1 = 1 / 6 := 
sorry

end NUMINAMATH_GPT_smallest_portion_l777_77768


namespace NUMINAMATH_GPT_animal_shelter_cats_l777_77700

theorem animal_shelter_cats (D C x : ℕ) (h1 : 15 * C = 7 * D) (h2 : 15 * (C + x) = 11 * D) (h3 : D = 60) : x = 16 :=
by
  sorry

end NUMINAMATH_GPT_animal_shelter_cats_l777_77700


namespace NUMINAMATH_GPT_six_digit_palindrome_count_l777_77734

def num_six_digit_palindromes : Nat :=
  let a_choices := 9
  let b_choices := 10
  let c_choices := 10
  let d_choices := 10
  a_choices * b_choices * c_choices * d_choices

theorem six_digit_palindrome_count : num_six_digit_palindromes = 9000 := by
  sorry

end NUMINAMATH_GPT_six_digit_palindrome_count_l777_77734


namespace NUMINAMATH_GPT_nests_count_l777_77741

theorem nests_count (birds nests : ℕ) (h1 : birds = 6) (h2 : birds - nests = 3) : nests = 3 := by
  sorry

end NUMINAMATH_GPT_nests_count_l777_77741


namespace NUMINAMATH_GPT_max_area_of_region_S_l777_77730

-- Define the radii of the circles
def radii : List ℕ := [2, 4, 6, 8]

-- Define the function for the maximum area of region S given the conditions
def max_area_region_S : ℕ := 75

-- Prove the maximum area of region S is 75π
theorem max_area_of_region_S {radii : List ℕ} (h : radii = [2, 4, 6, 8]) 
: max_area_region_S = 75 := by 
  sorry

end NUMINAMATH_GPT_max_area_of_region_S_l777_77730


namespace NUMINAMATH_GPT_blue_tshirt_count_per_pack_l777_77746

theorem blue_tshirt_count_per_pack :
  ∀ (total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack : ℕ), 
    white_packs = 3 →
    blue_packs = 2 → 
    tshirts_per_white_pack = 6 → 
    total_tshirts = 26 →
    total_tshirts = white_packs * tshirts_per_white_pack + blue_packs * tshirts_per_blue_pack →
  tshirts_per_blue_pack = 4 :=
by
  intros total_tshirts white_packs blue_packs tshirts_per_white_pack tshirts_per_blue_pack
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_blue_tshirt_count_per_pack_l777_77746


namespace NUMINAMATH_GPT_percent_increase_lines_l777_77718

theorem percent_increase_lines (final_lines increase : ℕ) (h1 : final_lines = 5600) (h2 : increase = 1600) :
  (increase * 100) / (final_lines - increase) = 40 := 
sorry

end NUMINAMATH_GPT_percent_increase_lines_l777_77718


namespace NUMINAMATH_GPT_no_two_digit_factorization_1729_l777_77726

noncomputable def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem no_two_digit_factorization_1729 :
  ¬ ∃ (a b : ℕ), a * b = 1729 ∧ is_two_digit a ∧ is_two_digit b :=
by
  sorry

end NUMINAMATH_GPT_no_two_digit_factorization_1729_l777_77726


namespace NUMINAMATH_GPT_friends_Sarah_brought_l777_77708

def total_people_in_house : Nat := 15
def in_bedroom : Nat := 2
def living_room : Nat := 8
def Sarah : Nat := 1

theorem friends_Sarah_brought :
  total_people_in_house - (in_bedroom + Sarah + living_room) = 4 := by
  sorry

end NUMINAMATH_GPT_friends_Sarah_brought_l777_77708


namespace NUMINAMATH_GPT_select_at_least_8_sticks_l777_77774

theorem select_at_least_8_sticks (S : Finset ℕ) (hS : S = (Finset.range 92 \ {0})) :
  ∃ (sticks : Finset ℕ) (h_sticks : sticks.card = 8),
    ∃ (a b c : ℕ) (h_a : a ∈ sticks) (h_b : b ∈ sticks) (h_c : c ∈ sticks),
    (a + b > c) ∧ (b + c > a) ∧ (c + a > b) :=
by
  -- Proof required here
  sorry

end NUMINAMATH_GPT_select_at_least_8_sticks_l777_77774


namespace NUMINAMATH_GPT_minimize_fractions_sum_l777_77764

theorem minimize_fractions_sum {A B C D E : ℕ}
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : A ≠ D) (h4 : A ≠ E)
  (h5 : B ≠ C) (h6 : B ≠ D) (h7 : B ≠ E)
  (h8 : C ≠ D) (h9 : C ≠ E) (h10 : D ≠ E)
  (h11 : A ≠ 9) (h12 : B ≠ 9) (h13 : C ≠ 9) (h14 : D ≠ 9) (h15 : E ≠ 9)
  (hA : 1 ≤ A) (hB : 1 ≤ B) (hC : 1 ≤ C) (hD : 1 ≤ D) (hE : 1 ≤ E)
  (hA' : A ≤ 9) (hB' : B ≤ 9) (hC' : C ≤ 9) (hD' : D ≤ 9) (hE' : E ≤ 9) :
  A / B + C / D + E / 9 = 125 / 168 :=
sorry

end NUMINAMATH_GPT_minimize_fractions_sum_l777_77764


namespace NUMINAMATH_GPT_martin_speed_l777_77720

theorem martin_speed (distance : ℝ) (time : ℝ) (h₁ : distance = 12) (h₂ : time = 6) : (distance / time = 2) :=
by 
  -- Note: The proof is not required as per instructions, so we use 'sorry'
  sorry

end NUMINAMATH_GPT_martin_speed_l777_77720


namespace NUMINAMATH_GPT_range_of_function_l777_77787

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_function : Set.Icc (-2 : ℝ) 7 = Set.image f (Set.Icc (-3 : ℝ) 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_function_l777_77787


namespace NUMINAMATH_GPT_total_rain_duration_l777_77745

theorem total_rain_duration:
  let first_day_duration := 10
  let second_day_duration := first_day_duration + 2
  let third_day_duration := 2 * second_day_duration
  first_day_duration + second_day_duration + third_day_duration = 46 :=
by
  sorry

end NUMINAMATH_GPT_total_rain_duration_l777_77745


namespace NUMINAMATH_GPT_spadesuit_evaluation_l777_77785

def spadesuit (a b : ℤ) : ℤ := Int.natAbs (a - b)

theorem spadesuit_evaluation :
  spadesuit 5 (spadesuit 3 9) = 1 := 
by 
  sorry

end NUMINAMATH_GPT_spadesuit_evaluation_l777_77785


namespace NUMINAMATH_GPT_bad_carrots_l777_77719

theorem bad_carrots (carol_carrots : ℕ) (mom_carrots : ℕ) (good_carrots : ℕ) (total_carrots : ℕ) (bad_carrots : ℕ) 
  (h1 : carol_carrots = 29)
  (h2 : mom_carrots = 16)
  (h3 : good_carrots = 38)
  (h4 : total_carrots = carol_carrots + mom_carrots)
  (h5 : bad_carrots = total_carrots - good_carrots) :
  bad_carrots = 7 := by
  sorry

end NUMINAMATH_GPT_bad_carrots_l777_77719


namespace NUMINAMATH_GPT_find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l777_77713

-- Define the nature of a "cool" triple.
def is_cool_triple (x y z : ℕ) : Prop :=
  x > 0 ∧ y > 1 ∧ z > 0 ∧ x^2 - 3 * y^2 = z^2 - 3

-- Part (a) i: For x = 5.
theorem find_cool_triple_x_eq_5 : ∃ (y z : ℕ), is_cool_triple 5 y z := sorry

-- Part (a) ii: For x = 7.
theorem find_cool_triple_x_eq_7 : ∃ (y z : ℕ), is_cool_triple 7 y z := sorry

-- Part (b): For every x ≥ 5 and odd, there are at least two distinct cool triples.
theorem two_distinct_cool_triples_for_odd_x (x : ℕ) (h1 : x ≥ 5) (h2 : x % 2 = 1) : 
  ∃ (y₁ z₁ y₂ z₂ : ℕ), is_cool_triple x y₁ z₁ ∧ is_cool_triple x y₂ z₂ ∧ (y₁, z₁) ≠ (y₂, z₂) := sorry

-- Part (c): Find a cool type triple with x even.
theorem find_cool_triple_x_even : ∃ (x y z : ℕ), x % 2 = 0 ∧ is_cool_triple x y z := sorry

end NUMINAMATH_GPT_find_cool_triple_x_eq_5_find_cool_triple_x_eq_7_two_distinct_cool_triples_for_odd_x_find_cool_triple_x_even_l777_77713


namespace NUMINAMATH_GPT_arithmetic_progression_12th_term_l777_77707

theorem arithmetic_progression_12th_term (a d n : ℤ) (h_a : a = 2) (h_d : d = 8) (h_n : n = 12) :
  a + (n - 1) * d = 90 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_arithmetic_progression_12th_term_l777_77707


namespace NUMINAMATH_GPT_max_value_of_a_plus_b_l777_77770

def max_possible_sum (a b : ℝ) (h1 : 4 * a + 3 * b ≤ 10) (h2 : a + 2 * b ≤ 4) : ℝ :=
  a + b

theorem max_value_of_a_plus_b :
  ∃a b : ℝ, (4 * a + 3 * b ≤ 10) ∧ (a + 2 * b ≤ 4) ∧ (a + b = 14 / 5) :=
by {
  sorry
}

end NUMINAMATH_GPT_max_value_of_a_plus_b_l777_77770


namespace NUMINAMATH_GPT_f_decreasing_increasing_find_b_range_l777_77776

-- Define the function f(x) and prove its properties for x > 0 and x < 0
noncomputable def f (x a : ℝ) : ℝ := x + a / x

theorem f_decreasing_increasing (a : ℝ) (h : a > 0):
  (∀ x : ℝ, 0 < x → x ≤ Real.sqrt a → ∀ x1 x2 : ℝ, (0 < x1 ∧ x1 < x2 ∧ x2 ≤ Real.sqrt a) → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, 0 < Real.sqrt a → Real.sqrt a ≤ x → ∀ x1 x2 : ℝ, (Real.sqrt a ≤ x1 ∧ x1 < x2) → f x1 a < f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → -Real.sqrt a ≤ x ∧ x < 0 → f x1 a > f x2 a) ∧ 
  (∀ x : ℝ, x < 0 → x < -Real.sqrt a → f x1 a < f x2 a)
:= sorry

-- Define the function h(x) and find the range of b
noncomputable def h (x : ℝ) : ℝ := x + 4 / x - 8
noncomputable def g (x b : ℝ) : ℝ := -x - 2 * b

theorem find_b_range:
  (∀ x1 : ℝ, 1 ≤ x1 ∧ x1 ≤ 3 → ∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 3 ∧ g x2 b = h x1) ↔
  1/2 ≤ b ∧ b ≤ 1
:= sorry

end NUMINAMATH_GPT_f_decreasing_increasing_find_b_range_l777_77776


namespace NUMINAMATH_GPT_combined_rate_l777_77773

theorem combined_rate
  (earl_rate : ℕ)
  (ellen_time : ℚ)
  (total_envelopes : ℕ)
  (total_time : ℕ)
  (combined_total_envelopes : ℕ)
  (combined_total_time : ℕ) :
  earl_rate = 36 →
  ellen_time = 1.5 →
  total_envelopes = 36 →
  total_time = 1 →
  combined_total_envelopes = 180 →
  combined_total_time = 3 →
  (earl_rate + (total_envelopes / ellen_time)) = 60 :=
by
  sorry

end NUMINAMATH_GPT_combined_rate_l777_77773


namespace NUMINAMATH_GPT_baker_cakes_total_l777_77771

-- Conditions
def initial_cakes : ℕ := 121
def cakes_sold : ℕ := 105
def cakes_bought : ℕ := 170

-- Proof Problem
theorem baker_cakes_total :
  initial_cakes - cakes_sold + cakes_bought = 186 :=
by
  sorry

end NUMINAMATH_GPT_baker_cakes_total_l777_77771


namespace NUMINAMATH_GPT_max_ratio_of_sequence_l777_77788

theorem max_ratio_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ)
  (hS : ∀ n : ℕ, S n = (n + 2) / 3 * a n) :
  ∃ n : ℕ, ∀ m : ℕ, (n = 2 → m ≠ 1) → (a n / a (n - 1)) ≤ (a m / a (m - 1)) :=
by
  sorry

end NUMINAMATH_GPT_max_ratio_of_sequence_l777_77788


namespace NUMINAMATH_GPT_player_jump_height_to_dunk_l777_77799

/-- Definitions given in the conditions -/
def rim_height : ℕ := 120
def player_height : ℕ := 72
def player_reach_above_head : ℕ := 22

/-- The statement to be proven -/
theorem player_jump_height_to_dunk :
  rim_height - (player_height + player_reach_above_head) = 26 :=
by
  sorry

end NUMINAMATH_GPT_player_jump_height_to_dunk_l777_77799


namespace NUMINAMATH_GPT_sum_of_squares_l777_77775

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 14) (h2 : a * b + b * c + a * c = 72) : 
  a^2 + b^2 + c^2 = 52 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l777_77775


namespace NUMINAMATH_GPT_total_weight_l777_77733

axiom D : ℕ -- Daughter's weight
axiom C : ℕ -- Grandchild's weight
axiom M : ℕ -- Mother's weight

-- Given conditions from the problem
axiom h1 : D + C = 60
axiom h2 : C = M / 5
axiom h3 : D = 50

-- The statement to be proven
theorem total_weight : M + D + C = 110 :=
by sorry

end NUMINAMATH_GPT_total_weight_l777_77733


namespace NUMINAMATH_GPT_marie_stamps_giveaway_l777_77750

theorem marie_stamps_giveaway :
  let notebooks := 4
  let stamps_per_notebook := 20
  let binders := 2
  let stamps_per_binder := 50
  let fraction_to_keep := 1/4
  let total_stamps := notebooks * stamps_per_notebook + binders * stamps_per_binder
  let stamps_to_keep := fraction_to_keep * total_stamps
  let stamps_to_give_away := total_stamps - stamps_to_keep
  stamps_to_give_away = 135 :=
by
  sorry

end NUMINAMATH_GPT_marie_stamps_giveaway_l777_77750


namespace NUMINAMATH_GPT_roots_of_polynomial_l777_77753

def poly (x : ℝ) : ℝ := x^3 - 3 * x^2 - 4 * x + 12

theorem roots_of_polynomial : 
  (poly 2 = 0) ∧ (poly (-2) = 0) ∧ (poly 3 = 0) ∧ 
  (∀ x, poly x = 0 → x = 2 ∨ x = -2 ∨ x = 3) :=
by
  sorry

end NUMINAMATH_GPT_roots_of_polynomial_l777_77753


namespace NUMINAMATH_GPT_max_value_y_l777_77755

theorem max_value_y (x : ℝ) (h : x < -1) : x + 1/(x + 1) ≤ -3 :=
by sorry

end NUMINAMATH_GPT_max_value_y_l777_77755


namespace NUMINAMATH_GPT_total_number_of_flags_is_12_l777_77756

def number_of_flags : Nat :=
  3 * 2 * 2

theorem total_number_of_flags_is_12 : number_of_flags = 12 := by
  sorry

end NUMINAMATH_GPT_total_number_of_flags_is_12_l777_77756


namespace NUMINAMATH_GPT_money_allocation_l777_77705

theorem money_allocation (x y : ℝ) (h1 : x + 1/2 * y = 50) (h2 : y + 2/3 * x = 50) : 
  x + 1/2 * y = 50 ∧ y + 2/3 * x = 50 :=
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_money_allocation_l777_77705


namespace NUMINAMATH_GPT_employees_females_l777_77710

theorem employees_females
  (total_employees : ℕ)
  (adv_deg_employees : ℕ)
  (coll_deg_employees : ℕ)
  (males_coll_deg : ℕ)
  (females_adv_deg : ℕ)
  (females_coll_deg : ℕ)
  (h1 : total_employees = 180)
  (h2 : adv_deg_employees = 90)
  (h3 : coll_deg_employees = 180 - 90)
  (h4 : males_coll_deg = 35)
  (h5 : females_adv_deg = 55)
  (h6 : females_coll_deg = 90 - 35) :
  females_coll_deg + females_adv_deg = 110 :=
by
  sorry

end NUMINAMATH_GPT_employees_females_l777_77710


namespace NUMINAMATH_GPT_total_triangles_l777_77786

theorem total_triangles (small_triangles : ℕ)
    (triangles_4_small : ℕ)
    (triangles_9_small : ℕ)
    (triangles_16_small : ℕ)
    (number_small_triangles : small_triangles = 20)
    (number_triangles_4_small : triangles_4_small = 5)
    (number_triangles_9_small : triangles_9_small = 1)
    (number_triangles_16_small : triangles_16_small = 1) :
    small_triangles + triangles_4_small + triangles_9_small + triangles_16_small = 27 := 
by 
    -- proof omitted
    sorry

end NUMINAMATH_GPT_total_triangles_l777_77786


namespace NUMINAMATH_GPT_slope_angle_at_point_l777_77784

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 4 * x + 8

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := 3 * x^2 - 4

-- State the problem: Prove the slope angle at point (1, 5) is 135 degrees
theorem slope_angle_at_point (θ : ℝ) (h : θ = 135) :
    f' 1 = -1 := 
by 
    sorry

end NUMINAMATH_GPT_slope_angle_at_point_l777_77784


namespace NUMINAMATH_GPT_intersection_eq_l777_77724

-- Define Set A based on the given condition
def setA : Set ℝ := {x | 1 < (3:ℝ)^x ∧ (3:ℝ)^x ≤ 9}

-- Define Set B based on the given condition
def setB : Set ℝ := {x | (x + 2) / (x - 1) ≤ 0}

-- Define the intersection of Set A and Set B
def intersection : Set ℝ := {x | x > 0 ∧ x < 1}

-- Prove that the intersection of setA and setB equals (0, 1)
theorem intersection_eq : {x | x > 0 ∧ x < 1} = {x | x ∈ setA ∧ x ∈ setB} :=
by
  sorry

end NUMINAMATH_GPT_intersection_eq_l777_77724


namespace NUMINAMATH_GPT_divide_8_friends_among_4_teams_l777_77716

def num_ways_to_divide_friends (n : ℕ) (teams : ℕ) :=
  teams ^ n

theorem divide_8_friends_among_4_teams :
  num_ways_to_divide_friends 8 4 = 65536 :=
by sorry

end NUMINAMATH_GPT_divide_8_friends_among_4_teams_l777_77716


namespace NUMINAMATH_GPT_percentage_discount_l777_77715

def cost_per_ball : ℝ := 0.1
def number_of_balls : ℕ := 10000
def amount_paid : ℝ := 700

theorem percentage_discount : (number_of_balls * cost_per_ball - amount_paid) / (number_of_balls * cost_per_ball) * 100 = 30 :=
by
  sorry

end NUMINAMATH_GPT_percentage_discount_l777_77715


namespace NUMINAMATH_GPT_wenlock_olympian_games_first_held_year_difference_l777_77791

theorem wenlock_olympian_games_first_held_year_difference :
  2012 - 1850 = 162 :=
sorry

end NUMINAMATH_GPT_wenlock_olympian_games_first_held_year_difference_l777_77791


namespace NUMINAMATH_GPT_part1_part2_part3_l777_77748

-- Definitions from the problem
def initial_cost_per_bottle := 16
def initial_selling_price := 20
def initial_sales_volume := 60
def sales_decrease_per_yuan_increase := 5

def daily_sales_volume (x : ℕ) : ℕ :=
  initial_sales_volume - sales_decrease_per_yuan_increase * x

def profit_per_bottle (x : ℕ) : ℕ :=
  (initial_selling_price - initial_cost_per_bottle) + x

def daily_profit (x : ℕ) : ℕ :=
  daily_sales_volume x * profit_per_bottle x

-- The proofs we need to establish
theorem part1 (x : ℕ) : 
  daily_sales_volume x = 60 - 5 * x ∧ profit_per_bottle x = 4 + x :=
sorry

theorem part2 (x : ℕ) : 
  daily_profit x = 300 → x = 6 ∨ x = 2 :=
sorry

theorem part3 : 
  ∃ x : ℕ, ∀ y : ℕ, (daily_profit x < daily_profit y) → 
              (daily_profit x = 320 ∧ x = 4) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l777_77748


namespace NUMINAMATH_GPT_largest_fraction_among_list_l777_77754

theorem largest_fraction_among_list :
  ∃ (f : ℚ), f = 105 / 209 ∧ 
  (f > 5 / 11) ∧ 
  (f > 9 / 20) ∧ 
  (f > 23 / 47) ∧ 
  (f > 205 / 409) := 
by
  sorry

end NUMINAMATH_GPT_largest_fraction_among_list_l777_77754


namespace NUMINAMATH_GPT_three_digit_numbers_with_repeats_l777_77725

theorem three_digit_numbers_with_repeats :
  (let total_numbers := 9 * 10 * 10
   let non_repeating_numbers := 9 * 9 * 8
   total_numbers - non_repeating_numbers = 252) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_numbers_with_repeats_l777_77725


namespace NUMINAMATH_GPT_maximize_profit_l777_77717

noncomputable def profit (t : ℝ) : ℝ :=
  27 - (18 / t) - t

theorem maximize_profit : ∀ t > 0, profit t ≤ 27 - 6 * Real.sqrt 2 ∧ profit (3 * Real.sqrt 2) = 27 - 6 * Real.sqrt 2 := by {
  sorry
}

end NUMINAMATH_GPT_maximize_profit_l777_77717


namespace NUMINAMATH_GPT_non_adjacent_boys_arrangements_l777_77751

-- We define the number of boys and girls
def boys := 4
def girls := 6

-- The function to compute combinations C(n, k)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- The function to compute permutations P(n, k)
def permutations (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

-- The total arrangements where 2 selected boys are not adjacent
def total_non_adjacent_arrangements : ℕ :=
  (combinations boys 2) * (combinations girls 3) * (permutations 3 3) * (permutations (3 + 1) 2)

theorem non_adjacent_boys_arrangements :
  total_non_adjacent_arrangements = 8640 := by
  sorry

end NUMINAMATH_GPT_non_adjacent_boys_arrangements_l777_77751


namespace NUMINAMATH_GPT_jenny_ate_65_chocolates_l777_77782

-- Define the number of chocolate squares Mike ate
def MikeChoc := 20

-- Define the function that calculates the chocolates Jenny ate
def JennyChoc (mikeChoc : ℕ) := 3 * mikeChoc + 5

-- The theorem stating the solution
theorem jenny_ate_65_chocolates (h : MikeChoc = 20) : JennyChoc MikeChoc = 65 := by
  -- Automatic proof step
  sorry

end NUMINAMATH_GPT_jenny_ate_65_chocolates_l777_77782


namespace NUMINAMATH_GPT_even_sum_of_digits_residue_l777_77738

def sum_of_digits (n : ℕ) : ℕ :=
  if n = 0 then 0 else (n % 10) + sum_of_digits (n / 10)

theorem even_sum_of_digits_residue (k : ℕ) (h : 2 ≤ k) (r : ℕ) (hr : r < k) :
  ∃ n : ℕ, sum_of_digits n % 2 = 0 ∧ n % k = r := 
sorry

end NUMINAMATH_GPT_even_sum_of_digits_residue_l777_77738


namespace NUMINAMATH_GPT_ab_value_l777_77783

theorem ab_value (a b : ℤ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 50) : a * b = 7 := by
  sorry

end NUMINAMATH_GPT_ab_value_l777_77783


namespace NUMINAMATH_GPT_scientific_notation_219400_l777_77793

def scientific_notation (n : ℝ) (m : ℝ) : Prop := n = m * 10^5

theorem scientific_notation_219400 : scientific_notation 219400 2.194 := 
by
  sorry

end NUMINAMATH_GPT_scientific_notation_219400_l777_77793


namespace NUMINAMATH_GPT_solve_for_x_l777_77766

theorem solve_for_x (x: ℝ) (h: (x-3)^4 = 16): x = 5 := 
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l777_77766


namespace NUMINAMATH_GPT_driver_actual_speed_l777_77723

theorem driver_actual_speed (v t : ℝ) 
  (h1 : t > 0) 
  (h2 : v > 0) 
  (cond : v * t = (v + 18) * (2 / 3 * t)) : 
  v = 36 :=
by 
  sorry

end NUMINAMATH_GPT_driver_actual_speed_l777_77723


namespace NUMINAMATH_GPT_area_of_square_l777_77706

-- Define the problem setting and the conditions
def square (side_length : ℝ) : Prop :=
  ∃ (width height : ℝ), width * height = side_length^2
    ∧ width = 5
    ∧ side_length / height = 5 / height

-- State the theorem to be proven
theorem area_of_square (side_length : ℝ) (width height : ℝ) (h1 : width = 5) (h2: side_length = 5 + 2 * height): 
  square side_length → side_length^2 = 400 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_area_of_square_l777_77706


namespace NUMINAMATH_GPT_optimal_garden_dimensions_l777_77758

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l ≥ w + 20 ∧ l * w = 9600 :=
by
  sorry

end NUMINAMATH_GPT_optimal_garden_dimensions_l777_77758


namespace NUMINAMATH_GPT_find_omega_l777_77798

noncomputable def f (ω x : ℝ) : ℝ := 3 * Real.sin (ω * x) - Real.sqrt 3 * Real.cos (ω * x)

theorem find_omega (ω : ℝ) (h₁ : ∀ x₁ x₂, (-ω < x₁ ∧ x₁ < x₂ ∧ x₂ < 2 * ω) → f ω x₁ < f ω x₂)
  (h₂ : ∀ x, f ω x = f ω (-2 * ω - x)) :
  ω = Real.sqrt (3 * Real.pi) / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_omega_l777_77798


namespace NUMINAMATH_GPT_us_supermarkets_count_l777_77760

-- Definition of variables and conditions
def total_supermarkets : ℕ := 84
def difference_us_canada : ℕ := 10

-- Proof statement
theorem us_supermarkets_count (C : ℕ) (H : 2 * C + difference_us_canada = total_supermarkets) :
  C + difference_us_canada = 47 :=
sorry

end NUMINAMATH_GPT_us_supermarkets_count_l777_77760


namespace NUMINAMATH_GPT_mod_product_prob_l777_77744

def prob_mod_product (a b : ℕ) : ℚ :=
  let quotient := a * b % 4
  if quotient = 0 then 1/2
  else if quotient = 1 then 1/8
  else if quotient = 2 then 1/4
  else if quotient = 3 then 1/8
  else 0

theorem mod_product_prob (a b : ℕ) :
  (∃ n : ℚ, n = prob_mod_product a b) :=
by
  sorry

end NUMINAMATH_GPT_mod_product_prob_l777_77744


namespace NUMINAMATH_GPT_number_of_sweaters_l777_77794

theorem number_of_sweaters 
(total_price_shirts : ℝ)
(total_shirts : ℕ)
(total_price_sweaters : ℝ)
(price_difference : ℝ) :
total_price_shirts = 400 ∧ total_shirts = 25 ∧ total_price_sweaters = 1500 ∧ price_difference = 4 →
(total_price_sweaters / ((total_price_shirts / total_shirts) + price_difference) = 75) :=
by
  intros
  sorry

end NUMINAMATH_GPT_number_of_sweaters_l777_77794


namespace NUMINAMATH_GPT_great_dane_more_than_triple_pitbull_l777_77731

variables (C P G : ℕ)
variables (h1 : G = 307) (h2 : P = 3 * C) (h3 : C + P + G = 439)

theorem great_dane_more_than_triple_pitbull
  : G - 3 * P = 10 :=
by
  sorry

end NUMINAMATH_GPT_great_dane_more_than_triple_pitbull_l777_77731


namespace NUMINAMATH_GPT_is_opposite_if_differ_in_sign_l777_77749

-- Define opposite numbers based on the given condition in the problem:
def opposite_numbers (a b : ℝ) : Prop := a = -b

-- State the theorem based on the translation in c)
theorem is_opposite_if_differ_in_sign (a b : ℝ) (h : a = -b) : opposite_numbers a b := by
  sorry

end NUMINAMATH_GPT_is_opposite_if_differ_in_sign_l777_77749


namespace NUMINAMATH_GPT_shaded_area_T_shape_l777_77789

theorem shaded_area_T_shape (a b c d e: ℕ) (square_side_length rect_length rect_width: ℕ)
  (h_side_lengths: ∀ x, x = 2 ∨ x = 4) (h_square: square_side_length = 6) 
  (h_rect_dim: rect_length = 4 ∧ rect_width = 2)
  (h_areas: [a, b, c, d, e] = [4, 4, 4, 8, 4]) :
  a + b + d + e = 20 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_T_shape_l777_77789


namespace NUMINAMATH_GPT_all_flowers_bloom_simultaneously_l777_77752

-- Define days of the week
inductive Day : Type
| Monday | Tuesday | Wednesday | Thursday | Friday | Saturday | Sunday
deriving DecidableEq

open Day

-- Define bloom conditions for the flowers
def sunflowers_bloom (d : Day) : Prop :=
  d ≠ Tuesday ∧ d ≠ Thursday ∧ d ≠ Sunday

def lilies_bloom (d : Day) : Prop :=
  d ≠ Thursday ∧ d ≠ Saturday

def peonies_bloom (d : Day) : Prop :=
  d ≠ Sunday

-- Define the main theorem
theorem all_flowers_bloom_simultaneously : ∃ d : Day, 
  sunflowers_bloom d ∧ lilies_bloom d ∧ peonies_bloom d ∧
  (∀ d', d' ≠ d → ¬ (sunflowers_bloom d' ∧ lilies_bloom d' ∧ peonies_bloom d')) :=
by
  sorry

end NUMINAMATH_GPT_all_flowers_bloom_simultaneously_l777_77752


namespace NUMINAMATH_GPT_find_m_l777_77709

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : m = -3 := by
  sorry

end NUMINAMATH_GPT_find_m_l777_77709


namespace NUMINAMATH_GPT_tangent_sum_problem_l777_77742

theorem tangent_sum_problem
  (α β : ℝ)
  (h_eq_root : ∃ (x y : ℝ), (x = Real.tan α) ∧ (y = Real.tan β) ∧ (6*x^2 - 5*x + 1 = 0) ∧ (6*y^2 - 5*y + 1 = 0))
  (h_range_α : 0 < α ∧ α < π/2)
  (h_range_β : π < β ∧ β < 3*π/2) :
  (Real.tan (α + β) = 1) ∧ (α + β = 5*π/4) := 
sorry

end NUMINAMATH_GPT_tangent_sum_problem_l777_77742


namespace NUMINAMATH_GPT_number_of_nurses_l777_77780

variables (D N : ℕ)

-- Condition: The total number of doctors and nurses is 250
def total_staff := D + N = 250

-- Condition: The ratio of doctors to nurses is 2 to 3
def ratio_doctors_to_nurses := D = (2 * N) / 3

-- Proof: The number of nurses is 150
theorem number_of_nurses (h1 : total_staff D N) (h2 : ratio_doctors_to_nurses D N) : N = 150 :=
sorry

end NUMINAMATH_GPT_number_of_nurses_l777_77780


namespace NUMINAMATH_GPT_husk_estimation_l777_77703

-- Define the conditions: total rice, sample size, and number of husks in the sample
def total_rice : ℕ := 1520
def sample_size : ℕ := 144
def husks_in_sample : ℕ := 18

-- Define the expected amount of husks in the total batch of rice
def expected_husks : ℕ := 190

-- The theorem stating the problem
theorem husk_estimation 
  (h : (husks_in_sample / sample_size) * total_rice = expected_husks) :
  (18 / 144) * 1520 = 190 := 
sorry

end NUMINAMATH_GPT_husk_estimation_l777_77703


namespace NUMINAMATH_GPT_solution_count_l777_77736

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem solution_count (a : ℝ) : 
  (∃ x : ℝ, f x = a) ↔ 
  ((a > 2 ∨ a < -2 ∧ ∃! x₁, f x₁ = a) ∨ 
   ((a = 2 ∨ a = -2) ∧ ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = a ∧ f x₂ = a) ∨ 
   (-2 < a ∧ a < 2 ∧ ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ f x₁ = a ∧ f x₂ = a ∧ f x₃ = a)) := 
by sorry

end NUMINAMATH_GPT_solution_count_l777_77736


namespace NUMINAMATH_GPT_decrease_in_profit_due_to_idle_loom_correct_l777_77712

def loom_count : ℕ := 80
def total_sales_value : ℕ := 500000
def monthly_manufacturing_expenses : ℕ := 150000
def establishment_charges : ℕ := 75000
def efficiency_level_idle_loom : ℕ := 100
def sales_per_loom : ℕ := total_sales_value / loom_count
def expenses_per_loom : ℕ := monthly_manufacturing_expenses / loom_count
def profit_contribution_idle_loom : ℕ := sales_per_loom - expenses_per_loom

def decrease_in_profit_due_to_idle_loom : ℕ := 4375

theorem decrease_in_profit_due_to_idle_loom_correct :
  profit_contribution_idle_loom = decrease_in_profit_due_to_idle_loom :=
by sorry

end NUMINAMATH_GPT_decrease_in_profit_due_to_idle_loom_correct_l777_77712


namespace NUMINAMATH_GPT_abs_x_minus_y_zero_l777_77763

theorem abs_x_minus_y_zero (x y : ℝ) 
  (h_avg : (x + y + 30 + 29 + 31) / 5 = 30)
  (h_var : ((x - 30)^2 + (y - 30)^2 + (30 - 30)^2 + (29 - 30)^2 + (31 - 30)^2) / 5 = 2) : 
  |x - y| = 0 :=
  sorry

end NUMINAMATH_GPT_abs_x_minus_y_zero_l777_77763


namespace NUMINAMATH_GPT_find_certain_number_l777_77790

theorem find_certain_number (x certain_number : ℕ) (h1 : certain_number + x = 13200) (h2 : x = 3327) : certain_number = 9873 :=
by
  sorry

end NUMINAMATH_GPT_find_certain_number_l777_77790


namespace NUMINAMATH_GPT_cos_expression_range_l777_77702

theorem cos_expression_range (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  -25 / 16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end NUMINAMATH_GPT_cos_expression_range_l777_77702


namespace NUMINAMATH_GPT_y_intercept_of_line_l777_77795

theorem y_intercept_of_line (m : ℝ) (x₀ : ℝ) (y₀ : ℝ) (h_slope : m = -3) (h_intercept : (x₀, y₀) = (7, 0)) : (0, 21) = (0, (y₀ - m * x₀)) :=
by
  sorry

end NUMINAMATH_GPT_y_intercept_of_line_l777_77795


namespace NUMINAMATH_GPT_gcd_sequence_condition_l777_77732

theorem gcd_sequence_condition (p q : ℕ) (hp : 0 < p) (hq : 0 < q)
  (a : ℕ → ℕ)
  (ha1 : a 1 = 1) (ha2 : a 2 = 1) 
  (ha_rec : ∀ n, a (n + 2) = p * a (n + 1) + q * a n) 
  (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  (gcd (a m) (a n) = a (gcd m n)) ↔ (p = 1) := 
sorry

end NUMINAMATH_GPT_gcd_sequence_condition_l777_77732


namespace NUMINAMATH_GPT_quadratic_function_through_point_l777_77701

theorem quadratic_function_through_point : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = a * x ^ 2 ∧ ((x, y) = (-1, 4)) → y = 4 * x ^ 2) :=
sorry

end NUMINAMATH_GPT_quadratic_function_through_point_l777_77701


namespace NUMINAMATH_GPT_interest_rate_is_20_percent_l777_77781

theorem interest_rate_is_20_percent (P A : ℝ) (t : ℝ) (r : ℝ) 
  (h1 : P = 500) (h2 : A = 1000) (h3 : t = 5) :
  A = P * (1 + r * t) → r = 0.20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_interest_rate_is_20_percent_l777_77781


namespace NUMINAMATH_GPT_values_of_m_l777_77769

def A : Set ℝ := { -1, 2 }
def B (m : ℝ) : Set ℝ := { x | m * x + 1 = 0 }

theorem values_of_m (m : ℝ) : (A ∪ B m = A) ↔ (m = -1/2 ∨ m = 0 ∨ m = 1) := by
  sorry

end NUMINAMATH_GPT_values_of_m_l777_77769


namespace NUMINAMATH_GPT_median_isosceles_right_triangle_leg_length_l777_77765

theorem median_isosceles_right_triangle_leg_length (m : ℝ) (h : ℝ) (x : ℝ)
  (H1 : m = 15)
  (H2 : m = h / 2)
  (H3 : 2 * x * x = h * h) : x = 15 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_median_isosceles_right_triangle_leg_length_l777_77765


namespace NUMINAMATH_GPT_solution_set_is_circle_with_exclusion_l777_77711

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end NUMINAMATH_GPT_solution_set_is_circle_with_exclusion_l777_77711


namespace NUMINAMATH_GPT_parabola_vertex_coordinates_l777_77778

theorem parabola_vertex_coordinates :
  ∃ (x y : ℝ), (∀ x : ℝ, y = 3 * x^2 + 2) ∧ x = 0 ∧ y = 2 :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_coordinates_l777_77778


namespace NUMINAMATH_GPT_film_finishes_earlier_on_first_channel_l777_77772

-- Definitions based on conditions
def DurationSegmentFirstChannel (n : ℕ) : ℝ := n * 22
def DurationSegmentSecondChannel (k : ℕ) : ℝ := k * 11

-- The time when first channel starts the n-th segment
def StartNthSegmentFirstChannel (n : ℕ) : ℝ := (n - 1) * 22

-- The number of segments second channel shows by the time first channel starts the n-th segment
def SegmentsShownSecondChannel (n : ℕ) : ℕ := ((n - 1) * 22) / 11

-- If first channel finishes earlier than second channel
theorem film_finishes_earlier_on_first_channel (n : ℕ) (hn : 1 < n) :
  DurationSegmentFirstChannel n < DurationSegmentSecondChannel (SegmentsShownSecondChannel n + 1) :=
sorry

end NUMINAMATH_GPT_film_finishes_earlier_on_first_channel_l777_77772


namespace NUMINAMATH_GPT_good_numbers_100_2010_ex_good_and_not_good_x_y_l777_77721

-- Definition of a good number
def is_good_number (n : ℤ) : Prop := ∃ a b : ℤ, n = a^2 + 161 * b^2

-- (1) Prove 100 and 2010 are good numbers
theorem good_numbers_100_2010 : is_good_number 100 ∧ is_good_number 2010 :=
by sorry

-- (2) Prove there exist positive integers x and y such that x^161 + y^161 is a good number, 
-- but x + y is not a good number
theorem ex_good_and_not_good_x_y : ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ is_good_number (x^161 + y^161) ∧ ¬ is_good_number (x + y) :=
by sorry

end NUMINAMATH_GPT_good_numbers_100_2010_ex_good_and_not_good_x_y_l777_77721


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_over_2_l777_77740

theorem cos_alpha_minus_pi_over_2 (α : ℝ) 
  (h1 : ∃ k : ℤ, α = k * (2 * Real.pi) ∨ α = k * (2 * Real.pi) + Real.pi / 2 ∨ α = k * (2 * Real.pi) + Real.pi ∨ α = k * (2 * Real.pi) + 3 * Real.pi / 2)
  (h2 : Real.cos α = 4 / 5)
  (h3 : Real.sin α = -3 / 5) : 
  Real.cos (α - Real.pi / 2) = -3 / 5 := 
by 
  sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_over_2_l777_77740


namespace NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l777_77704

noncomputable def radius_of_enclosing_sphere (r : ℝ) : ℝ :=
  let s := 6 -- side length of the cube
  let d := s * Real.sqrt 3 -- space diagonal of the cube
  (d + 2 * r) / 2

theorem smallest_enclosing_sphere_radius :
  radius_of_enclosing_sphere 2 = 3 * Real.sqrt 3 + 2 :=
by
  -- skipping the proof with sorry
  sorry

end NUMINAMATH_GPT_smallest_enclosing_sphere_radius_l777_77704


namespace NUMINAMATH_GPT_arithmetic_expression_equality_l777_77728

theorem arithmetic_expression_equality : 
  (1/4 : ℝ) * 8 * (1/16) * 32 * (1/64) * 128 * (1/256) * 512 * (1/1024) * 2048 * (1/4096) * 8192 = 64 := 
by
  sorry

end NUMINAMATH_GPT_arithmetic_expression_equality_l777_77728


namespace NUMINAMATH_GPT_f_neg_a_l777_77797

noncomputable def f (x : ℝ) : ℝ := x^3 + Real.sin x + 2

theorem f_neg_a (a : ℝ) (h : f a = 2) : f (-a) = 2 := by
  sorry

end NUMINAMATH_GPT_f_neg_a_l777_77797


namespace NUMINAMATH_GPT_gcd_of_items_l777_77743

theorem gcd_of_items :
  ∀ (plates spoons glasses bowls : ℕ),
  plates = 3219 →
  spoons = 5641 →
  glasses = 1509 →
  bowls = 2387 →
  Nat.gcd (Nat.gcd (Nat.gcd plates spoons) glasses) bowls = 1 :=
by
  intros plates spoons glasses bowls
  intros Hplates Hspoons Hglasses Hbowls
  rw [Hplates, Hspoons, Hglasses, Hbowls]
  sorry

end NUMINAMATH_GPT_gcd_of_items_l777_77743


namespace NUMINAMATH_GPT_problem_1_problem_2_l777_77757

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log (x + 1) / Real.log 2 else 2^(-x) - 1

theorem problem_1 : f (f (-2)) = 2 := by 
  sorry

theorem problem_2 (x_0 : ℝ) (h : f x_0 < 3) : -2 < x_0 ∧ x_0 < 7 := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l777_77757


namespace NUMINAMATH_GPT_price_restoration_percentage_l777_77727

noncomputable def original_price := 100
def reduced_price (P : ℝ) := 0.8 * P
def restored_price (P : ℝ) (x : ℝ) := P = x * reduced_price P

theorem price_restoration_percentage (P : ℝ) (x : ℝ) (h : restored_price P x) : x = 1.25 :=
by
  sorry

end NUMINAMATH_GPT_price_restoration_percentage_l777_77727


namespace NUMINAMATH_GPT_ratio_of_x_intercepts_l777_77747

theorem ratio_of_x_intercepts (c : ℝ) (u v : ℝ) (h1 : c ≠ 0) 
  (h2 : u = -c / 8) (h3 : v = -c / 4) : u / v = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_ratio_of_x_intercepts_l777_77747


namespace NUMINAMATH_GPT_pentagon_largest_angle_l777_77779

theorem pentagon_largest_angle
  (F G H I J : ℝ)
  (hF : F = 90)
  (hG : G = 70)
  (hH_eq_I : H = I)
  (hJ : J = 2 * H + 20)
  (sum_angles : F + G + H + I + J = 540) :
  max F (max G (max H (max I J))) = 200 :=
by
  sorry

end NUMINAMATH_GPT_pentagon_largest_angle_l777_77779


namespace NUMINAMATH_GPT_cricket_player_innings_l777_77739

theorem cricket_player_innings (n : ℕ) (T : ℕ) 
  (h1 : T = n * 48) 
  (h2 : T + 178 = (n + 1) * 58) : 
  n = 12 :=
by
  sorry

end NUMINAMATH_GPT_cricket_player_innings_l777_77739


namespace NUMINAMATH_GPT_base4_to_base10_conversion_l777_77735

theorem base4_to_base10_conversion :
  2 * 4^4 + 0 * 4^3 + 3 * 4^2 + 1 * 4^1 + 2 * 4^0 = 566 :=
by
  sorry

end NUMINAMATH_GPT_base4_to_base10_conversion_l777_77735


namespace NUMINAMATH_GPT_remi_water_bottle_capacity_l777_77767

-- Let's define the problem conditions
def daily_refills : ℕ := 3
def days : ℕ := 7
def total_spilled : ℕ := 5 + 8 -- Total spilled water in ounces
def total_intake : ℕ := 407 -- Total amount of water drunk in 7 days

-- The capacity of Remi's water bottle is the quantity we need to prove
def bottle_capacity (x : ℕ) : Prop :=
  daily_refills * days * x - total_spilled = total_intake

-- Statement of the proof problem
theorem remi_water_bottle_capacity : bottle_capacity 20 :=
by
  sorry

end NUMINAMATH_GPT_remi_water_bottle_capacity_l777_77767


namespace NUMINAMATH_GPT_perfect_square_fraction_l777_77759

open Nat

theorem perfect_square_fraction (a b : ℕ) (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : (ab + 1) ∣ (a^2 + b^2)) : ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
by 
  sorry

end NUMINAMATH_GPT_perfect_square_fraction_l777_77759


namespace NUMINAMATH_GPT_probability_of_specific_roll_l777_77761

noncomputable def probability_event : ℚ :=
  let favorable_outcomes_first_die := 3 -- 1, 2, 3
  let total_outcomes_die := 8
  let probability_first_die := favorable_outcomes_first_die / total_outcomes_die
  
  let favorable_outcomes_second_die := 4 -- 5, 6, 7, 8
  let probability_second_die := favorable_outcomes_second_die / total_outcomes_die
  
  probability_first_die * probability_second_die

theorem probability_of_specific_roll :
  probability_event = 3 / 16 := 
  by
    sorry

end NUMINAMATH_GPT_probability_of_specific_roll_l777_77761


namespace NUMINAMATH_GPT_intersection_M_N_l777_77737

open Set

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := sorry

end NUMINAMATH_GPT_intersection_M_N_l777_77737


namespace NUMINAMATH_GPT_neg_p_l777_77714

variable {x : ℝ}

def p := ∀ x > 0, Real.sin x ≤ 1

theorem neg_p : ¬ p ↔ ∃ x > 0, Real.sin x > 1 :=
by
  sorry

end NUMINAMATH_GPT_neg_p_l777_77714


namespace NUMINAMATH_GPT_distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l777_77777

noncomputable section

variables (a b c : ℝ) (ha : a ≠ b) (hb : b ≠ c) (hc : c ≠ a) (h1 : 0 < a) 
(h2 : 0 < b) (h3 : 0 < c)

theorem distinct_pos_numbers_implies_not_zero :
  (a - b) ^ 2 + (b - c) ^ 2 + (c - a) ^ 2 ≠ 0 :=
sorry

theorem at_least_one_of_abc :
  a > b ∨ a < b ∨ a = b :=
sorry

theorem impossible_for_all_neq :
  ¬(a ≠ c ∧ b ≠ c ∧ a ≠ b) :=
sorry

end NUMINAMATH_GPT_distinct_pos_numbers_implies_not_zero_at_least_one_of_abc_impossible_for_all_neq_l777_77777


namespace NUMINAMATH_GPT_coal_extraction_in_four_months_l777_77762

theorem coal_extraction_in_four_months
  (x1 x2 x3 x4 : ℝ)
  (h1 : 4 * x1 + x2 + 2 * x3 + 5 * x4 = 10)
  (h2 : 2 * x1 + 3 * x2 + 2 * x3 + x4 = 7)
  (h3 : 5 * x1 + 2 * x2 + x3 + 4 * x4 = 14) :
  4 * (x1 + x2 + x3 + x4) = 12 :=
by
  sorry

end NUMINAMATH_GPT_coal_extraction_in_four_months_l777_77762


namespace NUMINAMATH_GPT_polynomial_sum_eq_l777_77792

-- Definitions of the given polynomials
def p (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def q (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def r (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2
def s (x : ℝ) : ℝ := 3 * x^2 - 2 * x + 1

-- The theorem to prove
theorem polynomial_sum_eq (x : ℝ) : 
  p x + q x + r x + s x = -x^2 + 10 * x - 11 :=
by 
  -- Proof steps are omitted here
  sorry

end NUMINAMATH_GPT_polynomial_sum_eq_l777_77792
