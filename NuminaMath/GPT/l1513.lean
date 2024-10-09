import Mathlib

namespace average_student_headcount_l1513_151399

def student_headcount_fall_0203 : ℕ := 11700
def student_headcount_fall_0304 : ℕ := 11500
def student_headcount_fall_0405 : ℕ := 11600

theorem average_student_headcount : 
  (student_headcount_fall_0203 + student_headcount_fall_0304 + student_headcount_fall_0405) / 3 = 11600 := by
  sorry

end average_student_headcount_l1513_151399


namespace find_rate_per_kg_mangoes_l1513_151329

noncomputable def rate_per_kg_mangoes
  (cost_grapes_rate : ℕ)
  (quantity_grapes : ℕ)
  (quantity_mangoes : ℕ)
  (total_paid : ℕ)
  (rate_grapes : ℕ)
  (rate_mangoes : ℕ) :=
  total_paid = (rate_grapes * quantity_grapes) + (rate_mangoes * quantity_mangoes)

theorem find_rate_per_kg_mangoes :
  rate_per_kg_mangoes 70 8 11 1165 70 55 :=
by
  sorry

end find_rate_per_kg_mangoes_l1513_151329


namespace fenced_area_l1513_151326

theorem fenced_area (length_large : ℕ) (width_large : ℕ) 
                    (length_cutout : ℕ) (width_cutout : ℕ) 
                    (h_large : length_large = 20 ∧ width_large = 15)
                    (h_cutout : length_cutout = 4 ∧ width_cutout = 2) : 
                    ((length_large * width_large) - (length_cutout * width_cutout) = 292) := 
by
  sorry

end fenced_area_l1513_151326


namespace derek_dogs_count_l1513_151358

theorem derek_dogs_count
  (initial_dogs : ℕ)
  (initial_cars : ℕ)
  (cars_after_10_years : ℕ)
  (dogs_after_10_years : ℕ)
  (h1 : initial_dogs = 90)
  (h2 : initial_dogs = 3 * initial_cars)
  (h3 : cars_after_10_years = initial_cars + 210)
  (h4 : cars_after_10_years = 2 * dogs_after_10_years) :
  dogs_after_10_years = 120 :=
by
  sorry

end derek_dogs_count_l1513_151358


namespace uniquely_determine_T_l1513_151337

theorem uniquely_determine_T'_n (b e : ℤ) (S' T' : ℕ → ℤ)
  (hb : ∀ n, S' n = n * (2 * b + (n - 1) * e) / 2)
  (ht : ∀ n, T' n = n * (n + 1) * (3 * b + (n - 1) * e) / 6)
  (h3028 : S' 3028 = 3028 * (b + 1514 * e)) :
  T' 4543 = (4543 * (4543 + 1) * (3 * b + 4542 * e)) / 6 :=
by
  sorry

end uniquely_determine_T_l1513_151337


namespace math_problem_l1513_151364

theorem math_problem : 2357 + 3572 + 5723 + 2 * 7235 = 26122 :=
  by sorry

end math_problem_l1513_151364


namespace range_of_m_l1513_151349

theorem range_of_m (m : ℝ) : 
  ((∀ x : ℝ, (m + 1) * x^2 - 2 * (m - 1) * x + 3 * (m - 1) < 0) ↔ (m < -1)) :=
sorry

end range_of_m_l1513_151349


namespace mark_team_free_throws_l1513_151317

theorem mark_team_free_throws (F : ℕ) : 
  let mark_2_pointers := 25
  let mark_3_pointers := 8
  let opp_2_pointers := 2 * mark_2_pointers
  let opp_3_pointers := 1 / 2 * mark_3_pointers
  let total_points := 201
  2 * mark_2_pointers + 3 * mark_3_pointers + F + 2 * mark_2_pointers + 3 / 2 * mark_3_pointers + F / 2 = total_points →
  F = 10 := by
  sorry

end mark_team_free_throws_l1513_151317


namespace sol_sells_more_candy_each_day_l1513_151387

variable {x : ℕ}

-- Definition of the conditions
def sells_candy (first_day : ℕ) (rate : ℕ) (days : ℕ) : ℕ :=
  first_day + rate * (days - 1) * days / 2

def earns (bars_sold : ℕ) (price_cents : ℕ) : ℕ :=
  bars_sold * price_cents

-- Problem statement in Lean:
theorem sol_sells_more_candy_each_day
  (first_day_sales : ℕ := 10)
  (days : ℕ := 6)
  (price_cents : ℕ := 10)
  (total_earnings : ℕ := 1200) :
  earns (sells_candy first_day_sales x days) price_cents = total_earnings → x = 76 :=
sorry

end sol_sells_more_candy_each_day_l1513_151387


namespace min_det_is_neg_six_l1513_151373

-- Define the set of possible values for a, b, c, d
def values : List ℤ := [-1, 1, 2]

-- Define the determinant function for a 2x2 matrix
def det (a b c d : ℤ) : ℤ := a * d - b * c

-- State the theorem that the minimum value of the determinant is -6
theorem min_det_is_neg_six :
  ∃ (a b c d : ℤ), a ∈ values ∧ b ∈ values ∧ c ∈ values ∧ d ∈ values ∧ 
  (∀ (a' b' c' d' : ℤ), a' ∈ values → b' ∈ values → c' ∈ values → d' ∈ values → det a b c d ≤ det a' b' c' d') ∧ det a b c d = -6 :=
by
  sorry

end min_det_is_neg_six_l1513_151373


namespace common_area_of_triangles_is_25_l1513_151380

-- Define basic properties and conditions of an isosceles right triangle with hypotenuse = 10 units
def hypotenuse (a b : ℝ) : Prop := a^2 + b^2 = 10^2
def is_isosceles_right_triangle (a b : ℝ) : Prop := a = b ∧ hypotenuse a b

-- Definitions representing the triangls
noncomputable def triangle1 := ∃ a b : ℝ, is_isosceles_right_triangle a b
noncomputable def triangle2 := ∃ a b : ℝ, is_isosceles_right_triangle a b

-- The area common to both triangles is the focus
theorem common_area_of_triangles_is_25 : 
  triangle1 ∧ triangle2 → 
  ∃ area : ℝ, area = 25 
  := 
sorry

end common_area_of_triangles_is_25_l1513_151380


namespace cows_gift_by_friend_l1513_151391

-- Define the base conditions
def initial_cows : Nat := 39
def cows_died : Nat := 25
def cows_sold : Nat := 6
def cows_increase : Nat := 24
def cows_bought : Nat := 43
def final_cows : Nat := 83

-- Define the computation to get the number of cows after each event
def cows_after_died : Nat := initial_cows - cows_died
def cows_after_sold : Nat := cows_after_died - cows_sold
def cows_after_increase : Nat := cows_after_sold + cows_increase
def cows_after_bought : Nat := cows_after_increase + cows_bought

-- Define the proof problem
theorem cows_gift_by_friend : (final_cows - cows_after_bought) = 8 := by
  sorry

end cows_gift_by_friend_l1513_151391


namespace initial_sodium_chloride_percentage_l1513_151355

theorem initial_sodium_chloride_percentage :
  ∀ (P : ℝ),
  (∃ (C : ℝ), C = 24) → -- Tank capacity
  (∃ (E_rate : ℝ), E_rate = 0.4) → -- Evaporation rate per hour
  (∃ (time : ℝ), time = 6) → -- Time in hours
  (1 / 4 * C = 6) → -- Volume of mixture
  (6 * P / 100 + (6 - 6 * P / 100 - E_rate * time) = 3.6) → -- Concentration condition
  P = 30 :=
by
  intros P hC hE_rate htime hvolume hconcentration
  rcases hC with ⟨C, hC⟩
  rcases hE_rate with ⟨E_rate, hE_rate⟩
  rcases htime with ⟨time, htime⟩
  rw [hC, hE_rate, htime] at *
  sorry

end initial_sodium_chloride_percentage_l1513_151355


namespace stack_glasses_opacity_l1513_151306

-- Define the main problem's parameters and conditions
def num_glass_pieces : Nat := 5
def rotations := [0, 90, 180, 270] -- Possible rotations

-- Define the main theorem to state the problem in Lean
theorem stack_glasses_opacity :
  (∃ count : Nat, count = 7200 ∧
   -- There are 5 glass pieces
   ∀ (g : Fin num_glass_pieces), 
     -- Each piece is divided into 4 triangles
     ∀ (parts : Fin 4),
     -- There exists a unique painting configuration for each piece, can one prove it is exactly 7200 ways
     True
  ) :=
  sorry

end stack_glasses_opacity_l1513_151306


namespace students_walk_fraction_l1513_151347

theorem students_walk_fraction (h1 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/3))
                               (h2 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/5))
                               (h3 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/8))
                               (h4 : ∀ (students : ℕ), (∃ num : ℕ, num / students = 1/10)) :
  ∃ (students : ℕ), (students - num1 - num2 - num3 - num4) / students = 29 / 120 :=
by
  sorry

end students_walk_fraction_l1513_151347


namespace Vincent_sells_8_literature_books_per_day_l1513_151333

theorem Vincent_sells_8_literature_books_per_day
  (fantasy_book_cost : ℕ)
  (literature_book_cost : ℕ)
  (fantasy_books_sold_per_day : ℕ)
  (total_earnings_5_days : ℕ)
  (H_fantasy_book_cost : fantasy_book_cost = 4)
  (H_literature_book_cost : literature_book_cost = 2)
  (H_fantasy_books_sold_per_day : fantasy_books_sold_per_day = 5)
  (H_total_earnings_5_days : total_earnings_5_days = 180) :
  ∃ L : ℕ, L = 8 :=
by
  sorry

end Vincent_sells_8_literature_books_per_day_l1513_151333


namespace match_scheduling_ways_l1513_151302

def different_ways_to_schedule_match (num_players : Nat) (num_rounds : Nat) : Nat :=
  (num_rounds.factorial * num_rounds.factorial)

theorem match_scheduling_ways : different_ways_to_schedule_match 4 4 = 576 :=
by
  sorry

end match_scheduling_ways_l1513_151302


namespace room_width_l1513_151318

theorem room_width (length : ℕ) (total_cost : ℕ) (cost_per_sqm : ℕ) : ℚ :=
  let area := total_cost / cost_per_sqm
  let width := area / length
  width

example : room_width 9 38475 900 = 4.75 := by
  sorry

end room_width_l1513_151318


namespace move_line_down_eq_l1513_151322

theorem move_line_down_eq (x y : ℝ) : (y = 2 * x) → (y - 3 = 2 * x - 3) :=
by
  sorry

end move_line_down_eq_l1513_151322


namespace problem1_problem2_l1513_151390

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Prove that for a = 1/2, A ∩ B = { x | 0 < x ∧ x < 1 }
theorem problem1 : setA (1/2) ∩ setB = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

-- Prove that if A ∩ B = ∅, then a ≤ -1/2 or a ≥ 2
theorem problem2 (a : ℝ) (h : setA a ∩ setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by
  sorry

end problem1_problem2_l1513_151390


namespace yoongi_caught_frogs_l1513_151381

theorem yoongi_caught_frogs (initial_frogs caught_later : ℕ) (h1 : initial_frogs = 5) (h2 : caught_later = 2) : (initial_frogs + caught_later = 7) :=
by
  sorry

end yoongi_caught_frogs_l1513_151381


namespace line_point_coordinates_l1513_151301

theorem line_point_coordinates (t : ℝ) (x y z : ℝ) : 
  (x, y, z) = (5, 0, 3) + t • (0, 3, 0) →
  t = 1/2 →
  (x, y, z) = (5, 3/2, 3) :=
by
  intros h1 h2
  sorry

end line_point_coordinates_l1513_151301


namespace Wendy_runs_farther_l1513_151357

-- Define the distances Wendy ran and walked
def distance_ran : ℝ := 19.83
def distance_walked : ℝ := 9.17

-- Define the difference in distances
def difference : ℝ := distance_ran - distance_walked

-- The theorem to prove
theorem Wendy_runs_farther : difference = 10.66 := by
  sorry

end Wendy_runs_farther_l1513_151357


namespace suff_and_not_necessary_l1513_151319

theorem suff_and_not_necessary (a b : ℝ) (h : a > b ∧ b > 0) :
  (|a| > |b|) ∧ (¬(∀ x y : ℝ, (|x| > |y|) → (x > y ∧ y > 0))) :=
by
  sorry

end suff_and_not_necessary_l1513_151319


namespace river_depth_mid_may_l1513_151340

variable (DepthMidMay DepthMidJune DepthMidJuly : ℕ)

theorem river_depth_mid_may :
  (DepthMidJune = DepthMidMay + 10) →
  (DepthMidJuly = 3 * DepthMidJune) →
  (DepthMidJuly = 45) →
  DepthMidMay = 5 :=
by
  intros h1 h2 h3
  sorry

end river_depth_mid_may_l1513_151340


namespace green_balloons_correct_l1513_151396

-- Defining the quantities
def total_balloons : ℕ := 67
def red_balloons : ℕ := 29
def blue_balloons : ℕ := 21

-- Calculating the green balloons
def green_balloons : ℕ := total_balloons - red_balloons - blue_balloons

-- The theorem we want to prove
theorem green_balloons_correct : green_balloons = 17 :=
by
  -- proof goes here
  sorry

end green_balloons_correct_l1513_151396


namespace first_route_red_lights_longer_l1513_151376

-- Conditions
def first_route_base_time : ℕ := 10
def red_light_time : ℕ := 3
def num_stoplights : ℕ := 3
def second_route_time : ℕ := 14

-- Question to Answer
theorem first_route_red_lights_longer : (first_route_base_time + num_stoplights * red_light_time - second_route_time) = 5 := by
  sorry

end first_route_red_lights_longer_l1513_151376


namespace customer_paid_correct_amount_l1513_151324

theorem customer_paid_correct_amount (cost_price : ℕ) (markup_percentage : ℕ) (total_price : ℕ) :
  cost_price = 6500 → 
  markup_percentage = 30 → 
  total_price = cost_price + (cost_price * markup_percentage / 100) → 
  total_price = 8450 :=
by
  intros h_cost_price h_markup_percentage h_total_price
  sorry

end customer_paid_correct_amount_l1513_151324


namespace volume_of_sphere_from_cube_surface_area_l1513_151316

theorem volume_of_sphere_from_cube_surface_area (S : ℝ) (h : S = 24) : 
  ∃ V : ℝ, V = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_sphere_from_cube_surface_area_l1513_151316


namespace odd_number_adjacent_product_diff_l1513_151362

variable (x : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem odd_number_adjacent_product_diff (h : is_odd x)
  (adjacent_diff : x * (x + 2) - x * (x - 2) = 44) : x = 11 :=
by
  sorry

end odd_number_adjacent_product_diff_l1513_151362


namespace sqrt_floor_19992000_l1513_151350

theorem sqrt_floor_19992000 : (Int.floor (Real.sqrt 19992000)) = 4471 := by
  sorry

end sqrt_floor_19992000_l1513_151350


namespace quadratic_always_positive_l1513_151339

theorem quadratic_always_positive (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 3) * x - 2 * k + 12 > 0) ↔ -7 < k ∧ k < 5 :=
sorry

end quadratic_always_positive_l1513_151339


namespace marbles_with_at_least_one_blue_l1513_151382

theorem marbles_with_at_least_one_blue :
  (Nat.choose 10 4) - (Nat.choose 8 4) = 140 :=
by
  sorry

end marbles_with_at_least_one_blue_l1513_151382


namespace find_triples_l1513_151372

theorem find_triples (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxy : x ≤ y) (hyz : y ≤ z) 
  (h_eq : x * y + y * z + z * x - x * y * z = 2) : (x = 1 ∧ y = 1 ∧ z = 1) ∨ (x = 2 ∧ y = 3 ∧ z = 4) := 
by 
  sorry

end find_triples_l1513_151372


namespace scientific_notation_826M_l1513_151331

theorem scientific_notation_826M : 826000000 = 8.26 * 10^8 :=
by
  sorry

end scientific_notation_826M_l1513_151331


namespace number_of_articles_l1513_151346

-- Define main conditions
variable (N : ℕ) -- Number of articles
variable (CP SP : ℝ) -- Cost price and Selling price per article

-- Condition 1: Cost price of N articles equals the selling price of 15 articles
def condition1 : Prop := N * CP = 15 * SP

-- Condition 2: Selling price includes a 33.33% profit on cost price
def condition2 : Prop := SP = CP * 1.3333

-- Prove that the number of articles N equals 20
theorem number_of_articles (h1 : condition1 N CP SP) (h2 : condition2 CP SP) : N = 20 :=
by sorry

end number_of_articles_l1513_151346


namespace some_number_is_five_l1513_151395

theorem some_number_is_five (x : ℕ) (some_number : ℕ) (h1 : x = 5) (h2 : x / some_number + 3 = 4) : some_number = 5 := by
  sorry

end some_number_is_five_l1513_151395


namespace final_score_correct_l1513_151388

def innovation_score : ℕ := 88
def comprehensive_score : ℕ := 80
def language_score : ℕ := 75

def weight_innovation : ℕ := 5
def weight_comprehensive : ℕ := 3
def weight_language : ℕ := 2

def final_score : ℕ :=
  (innovation_score * weight_innovation + comprehensive_score * weight_comprehensive +
   language_score * weight_language) /
  (weight_innovation + weight_comprehensive + weight_language)

theorem final_score_correct :
  final_score = 83 :=
by
  -- proof goes here
  sorry

end final_score_correct_l1513_151388


namespace sum_of_squares_l1513_151359

theorem sum_of_squares (a b : ℝ) (h1 : (a + b) / 2 = 8) (h2 : Real.sqrt (a * b) = 2 * Real.sqrt 5) :
  a^2 + b^2 = 216 :=
by
  sorry

end sum_of_squares_l1513_151359


namespace greatest_integer_l1513_151348

-- Define the conditions for the problem
def isMultiple4 (n : ℕ) : Prop := n % 4 = 0
def notMultiple8 (n : ℕ) : Prop := n % 8 ≠ 0
def notMultiple12 (n : ℕ) : Prop := n % 12 ≠ 0
def gcf4 (n : ℕ) : Prop := Nat.gcd n 24 = 4
def lessThan200 (n : ℕ) : Prop := n < 200

-- State the main theorem
theorem greatest_integer : ∃ n : ℕ, lessThan200 n ∧ gcf4 n ∧ n = 196 :=
by
  sorry

end greatest_integer_l1513_151348


namespace line_contains_diameter_of_circle_l1513_151332

noncomputable def equation_of_circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 8 = 0

noncomputable def equation_of_line (x y : ℝ) : Prop :=
  2*x - y - 1 = 0

theorem line_contains_diameter_of_circle :
  (∃ x y : ℝ, equation_of_circle x y ∧ equation_of_line x y) :=
sorry

end line_contains_diameter_of_circle_l1513_151332


namespace y_equals_4_if_abs_diff_eq_l1513_151371

theorem y_equals_4_if_abs_diff_eq (y : ℝ) (h : |y - 3| = |y - 5|) : y = 4 :=
sorry

end y_equals_4_if_abs_diff_eq_l1513_151371


namespace big_dogs_count_l1513_151312

theorem big_dogs_count (B S : ℕ) (h_ratio : 3 * S = 17 * B) (h_total : B + S = 80) :
  B = 12 :=
by
  sorry

end big_dogs_count_l1513_151312


namespace candy_remainder_l1513_151389

theorem candy_remainder :
  38759863 % 6 = 1 :=
by
  sorry

end candy_remainder_l1513_151389


namespace fraction_solution_l1513_151366

theorem fraction_solution (N : ℝ) (h : N = 12.0) : (0.6667 * N + 1) = (3/4) * N := by 
  sorry

end fraction_solution_l1513_151366


namespace arcade_game_monster_perimeter_l1513_151342

theorem arcade_game_monster_perimeter :
  let r := 1 -- radius of the circle in cm
  let theta := 60 -- central angle of the missing sector in degrees
  let circumference := 2 * Real.pi * r -- circumference of the full circle
  let arc_fraction := (360 - theta) / 360 -- fraction of the circle forming the arc
  let arc_length := arc_fraction * circumference -- length of the arc
  let perimeter := arc_length + 2 * r -- total perimeter (arc + two radii)
  perimeter = (5 / 3) * Real.pi + 2 :=
by
  sorry

end arcade_game_monster_perimeter_l1513_151342


namespace min_value_expression_l1513_151386

theorem min_value_expression : ∃ x : ℝ, x = 300 ∧ ∀ y : ℝ, (y^2 - 600*y + 369) ≥ (300^2 - 600*300 + 369) := by
  use 300
  sorry

end min_value_expression_l1513_151386


namespace problem1_solution_set_problem2_inequality_l1513_151368

theorem problem1_solution_set (x : ℝ) : (-1 < x) ∧ (x < 9) ↔ (|x| + |x - 3| < x + 6) :=
by sorry

theorem problem2_inequality (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hn : 9 * x + y = 1) : x + y ≥ 16 * x * y :=
by sorry

end problem1_solution_set_problem2_inequality_l1513_151368


namespace slope_ratio_l1513_151352

theorem slope_ratio (s t k b : ℝ) 
  (h1: b = -12 * s)
  (h2: b = k - 7) 
  (ht: t = (7 - k) / 7) 
  (hs: s = (7 - k) / 12): 
  s / t = 7 / 12 := 
  sorry

end slope_ratio_l1513_151352


namespace max_area_square_l1513_151308

theorem max_area_square (P : ℝ) : 
  ∀ x y : ℝ, 2 * x + 2 * y = P → (x * y ≤ (P / 4) ^ 2) :=
by
  sorry

end max_area_square_l1513_151308


namespace smallest_possible_third_term_l1513_151345

theorem smallest_possible_third_term :
  ∃ (d : ℝ), (d = -3 + Real.sqrt 134 ∨ d = -3 - Real.sqrt 134) ∧ 
  (7, 7 + d + 3, 7 + 2 * d + 18) = (7, 10 + d, 25 + 2 * d) ∧ 
  min (25 + 2 * (-3 + Real.sqrt 134)) (25 + 2 * (-3 - Real.sqrt 134)) = 19 + 2 * Real.sqrt 134 :=
by
  sorry

end smallest_possible_third_term_l1513_151345


namespace combined_salaries_of_ABCD_l1513_151320

theorem combined_salaries_of_ABCD 
  (A B C D E : ℝ)
  (h1 : E = 9000)
  (h2 : (A + B + C + D + E) / 5 = 8600) :
  A + B + C + D = 34000 := 
sorry

end combined_salaries_of_ABCD_l1513_151320


namespace ring_toss_total_amount_l1513_151385

-- Defining the amounts made in the two periods
def amount_first_period : Nat := 382
def amount_second_period : Nat := 374

-- The total amount made
def total_amount : Nat := amount_first_period + amount_second_period

-- Statement that the total amount calculated is equal to the given answer
theorem ring_toss_total_amount :
  total_amount = 756 := by
  sorry

end ring_toss_total_amount_l1513_151385


namespace max_value_neg_expr_l1513_151307

theorem max_value_neg_expr (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_sum : a + b = 1) :
  - (1 / (2 * a)) - (2 / b) ≤ - (9 / 2) :=
by 
  sorry

end max_value_neg_expr_l1513_151307


namespace pf1_pf2_range_l1513_151315

noncomputable def ellipse_point (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 = 1

noncomputable def dot_product (x y : ℝ) : ℝ :=
  (x ^ 2 + y ^ 2 - 3)

theorem pf1_pf2_range (x y : ℝ) (h : ellipse_point x y) :
  -2 ≤ dot_product x y ∧ dot_product x y ≤ 1 :=
by
  sorry

end pf1_pf2_range_l1513_151315


namespace tan_positive_implies_sin_cos_positive_l1513_151303

variables {α : ℝ}

theorem tan_positive_implies_sin_cos_positive (h : Real.tan α > 0) : Real.sin α * Real.cos α > 0 :=
sorry

end tan_positive_implies_sin_cos_positive_l1513_151303


namespace polynomial_factorization_l1513_151398

-- Define the polynomial and its factorized form
def polynomial (x : ℝ) : ℝ := x^2 - 4*x + 4
def factorized_form (x : ℝ) : ℝ := (x - 2)^2

-- The theorem stating that the polynomial equals its factorized form
theorem polynomial_factorization (x : ℝ) : polynomial x = factorized_form x :=
by {
  sorry -- Proof skipped
}

end polynomial_factorization_l1513_151398


namespace coefficient_x3_expansion_l1513_151309

/--
Prove that the coefficient of \(x^{3}\) in the expansion of \(( \frac{x}{\sqrt{y}} - \frac{y}{\sqrt{x}})^{6}\) is \(15\).
-/
theorem coefficient_x3_expansion (x y : ℝ) : 
  (∃ c : ℝ, c = 15 ∧ (x / y.sqrt - y / x.sqrt) ^ 6 = c * x ^ 3) :=
sorry

end coefficient_x3_expansion_l1513_151309


namespace equilateral_triangle_in_ellipse_l1513_151360

-- Given
def ellipse (x y : ℝ) : Prop := x^2 + 4 * y^2 = 4
def altitude_on_y_axis (v : ℝ × ℝ := (0, 1)) : Prop := 
  v.1 = 0 ∧ v.2 = 1

-- The problem statement translated into a Lean proof goal
theorem equilateral_triangle_in_ellipse :
  ∃ (m n : ℕ), 
    (∀ (x y : ℝ), ellipse x y) →
    altitude_on_y_axis (0,1) →
    m.gcd n = 1 ∧ m + n = 937 :=
sorry

end equilateral_triangle_in_ellipse_l1513_151360


namespace find_a_minus_inv_a_l1513_151353

variable (a : ℝ)
variable (h : a + 1 / a = Real.sqrt 13)

theorem find_a_minus_inv_a : a - 1 / a = 3 ∨ a - 1 / a = -3 := by
  sorry

end find_a_minus_inv_a_l1513_151353


namespace find_z_coordinate_of_point_on_line_passing_through_l1513_151330

theorem find_z_coordinate_of_point_on_line_passing_through
  (p1 p2 : ℝ × ℝ × ℝ)
  (x_value : ℝ)
  (z_value : ℝ)
  (h1 : p1 = (1, 3, 2))
  (h2 : p2 = (4, 2, -1))
  (h3 : x_value = 3)
  (param : ℝ)
  (h4 : x_value = (1 + 3 * param))
  (h5 : z_value = (2 - 3 * param)) :
  z_value = 0 := by
  sorry

end find_z_coordinate_of_point_on_line_passing_through_l1513_151330


namespace shift_left_by_pi_over_six_l1513_151313

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 3) * Real.sin x + Real.cos x
noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin x

theorem shift_left_by_pi_over_six : f = λ x => g (x + Real.pi / 6) := by
  sorry

end shift_left_by_pi_over_six_l1513_151313


namespace tank_capacity_l1513_151311

theorem tank_capacity (C : ℕ) (h₁ : C = 785) :
  360 - C / 4 - C / 8 = C / 12 :=
by 
  -- Assuming h₁: C = 785
  have h₁: C = 785 := by exact h₁
  -- Provide proof steps here (not required for the task)
  sorry

end tank_capacity_l1513_151311


namespace planes_parallel_if_any_line_parallel_l1513_151397

axiom Plane : Type
axiom Line : Type
axiom contains : Plane → Line → Prop
axiom parallel : Plane → Plane → Prop
axiom parallel_lines : Line → Plane → Prop

theorem planes_parallel_if_any_line_parallel (α β : Plane)
  (h₁ : ∀ l, contains α l → parallel_lines l β) :
  parallel α β :=
sorry

end planes_parallel_if_any_line_parallel_l1513_151397


namespace total_cases_after_three_weeks_l1513_151377

-- Definitions and conditions directly from the problem
def week1_cases : ℕ := 5000
def week2_cases : ℕ := week1_cases / 2
def week3_cases : ℕ := week2_cases + 2000
def total_cases : ℕ := week1_cases + week2_cases + week3_cases

-- The theorem to prove
theorem total_cases_after_three_weeks :
  total_cases = 12000 := 
by
  -- Sorry allows us to skip the actual proof
  sorry

end total_cases_after_three_weeks_l1513_151377


namespace sophia_book_length_l1513_151375

variables {P : ℕ}

def total_pages (P : ℕ) : Prop :=
  (2 / 3 : ℝ) * P = (1 / 3 : ℝ) * P + 90

theorem sophia_book_length 
  (h1 : total_pages P) :
  P = 270 :=
sorry

end sophia_book_length_l1513_151375


namespace trainer_voice_radius_l1513_151370

noncomputable def area_of_heard_voice (r : ℝ) : ℝ := (1/4) * Real.pi * r^2

theorem trainer_voice_radius :
  ∃ r : ℝ, abs (r - 140) < 1 ∧ area_of_heard_voice r = 15393.804002589986 :=
by
  sorry

end trainer_voice_radius_l1513_151370


namespace unit_fraction_decomposition_l1513_151354

theorem unit_fraction_decomposition (n : ℕ) (hn : 0 < n): 
  (1 : ℚ) / n = (1 : ℚ) / (2 * n) + (1 : ℚ) / (3 * n) + (1 : ℚ) / (6 * n) :=
by
  sorry

end unit_fraction_decomposition_l1513_151354


namespace problem_x_l1513_151327

theorem problem_x (f : ℝ → ℝ) (m : ℝ) 
  (h1 : ∀ x : ℝ, f (1/2 * x - 1) = 2 * x + 3) 
  (h2 : f m = 6) : 
  m = -1/4 :=
sorry

end problem_x_l1513_151327


namespace max_four_digit_prime_product_l1513_151304

theorem max_four_digit_prime_product :
  ∃ (x y : ℕ) (n : ℕ), x < 5 ∧ y < 5 ∧ x ≠ y ∧ Prime x ∧ Prime y ∧ Prime (10 * x + y) ∧ n = x * y * (10 * x + y) ∧ n = 138 :=
by
  sorry

end max_four_digit_prime_product_l1513_151304


namespace fixed_point_l1513_151344

noncomputable def fixed_point_function (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : (ℝ × ℝ) :=
  (1, a^(1 - (1 : ℝ)) + 5)

theorem fixed_point (a : ℝ) (h₀ : a > 0) (h₁ : a ≠ 1) : fixed_point_function a h₀ h₁ = (1, 6) :=
by 
  sorry

end fixed_point_l1513_151344


namespace length_of_metallic_sheet_l1513_151305

variable (L : ℝ) (width side volume : ℝ)

theorem length_of_metallic_sheet (h1 : width = 36) (h2 : side = 8) (h3 : volume = 5120) :
  ((L - 2 * side) * (width - 2 * side) * side = volume) → L = 48 := 
by
  intros h_eq
  sorry

end length_of_metallic_sheet_l1513_151305


namespace swimming_pool_length_l1513_151369

theorem swimming_pool_length :
  ∀ (w d1 d2 V : ℝ), w = 9 → d1 = 1 → d2 = 4 → V = 270 → 
  (((V = (1 / 2) * (d1 + d2) * w * l) → l = 12)) :=
by
  intros w d1 d2 V hw hd1 hd2 hV hv
  simp only [hw, hd1, hd2, hV] at hv
  sorry

end swimming_pool_length_l1513_151369


namespace seventh_term_of_arithmetic_sequence_l1513_151394

theorem seventh_term_of_arithmetic_sequence (a d : ℤ) 
  (h1 : 5 * a + 10 * d = 15) 
  (h2 : a + 5 * d = 6) : 
  a + 6 * d = 7 :=
by
  -- Proof omitted
  sorry

end seventh_term_of_arithmetic_sequence_l1513_151394


namespace probability_diamond_first_and_ace_or_king_second_l1513_151343

-- Define the condition of the combined deck consisting of two standard decks (104 cards total)
def two_standard_decks := 104

-- Define the number of diamonds, aces, and kings in the combined deck
def number_of_diamonds := 26
def number_of_aces := 8
def number_of_kings := 8

-- Define the events for drawing cards
def first_card_is_diamond := (number_of_diamonds : ℕ) / (two_standard_decks : ℕ)
def second_card_is_ace_or_king_if_first_is_not_ace_or_king :=
  (16 / 103 : ℚ) -- 16 = 8 (aces) + 8 (kings)
def second_card_is_ace_or_king_if_first_is_ace_or_king :=
  (15 / 103 : ℚ) -- 15 = 7 (remaining aces) + 7 (remaining kings) + 1 (remaining ace or king of the same suit)

-- Define the probabilities of the combined event
def probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king :=
  (22 / 104) * (16 / 103)
def probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king :=
  (4 / 104) * (15 / 103)

-- Define the total probability combining both events
noncomputable def total_probability :=
  probability_first_is_non_ace_king_diamond_and_second_is_ace_or_king +
  probability_first_is_ace_or_king_diamond_and_second_is_ace_or_king

-- Theorem stating the desired probability result
theorem probability_diamond_first_and_ace_or_king_second :
  total_probability = (103 / 2678 : ℚ) :=
sorry

end probability_diamond_first_and_ace_or_king_second_l1513_151343


namespace unique_root_conditions_l1513_151379

theorem unique_root_conditions (m : ℝ) (x y : ℝ) :
  (x^2 = 2 * abs x ∧ abs x - y - m = 1 - y^2) ↔ m = 3 / 4 := sorry

end unique_root_conditions_l1513_151379


namespace min_staff_members_l1513_151393

theorem min_staff_members
  (num_male_students : ℕ)
  (num_benches_3_students : ℕ)
  (num_benches_4_students : ℕ)
  (num_female_students : ℕ)
  (total_students : ℕ)
  (total_seating_capacity : ℕ)
  (additional_seats_required : ℕ)
  (num_staff_members : ℕ)
  (h1 : num_female_students = 4 * num_male_students)
  (h2 : num_male_students = 29)
  (h3 : num_benches_3_students = 15)
  (h4 : num_benches_4_students = 14)
  (h5 : total_seating_capacity = 3 * num_benches_3_students + 4 * num_benches_4_students)
  (h6 : total_students = num_male_students + num_female_students)
  (h7 : additional_seats_required = total_students - total_seating_capacity)
  (h8 : num_staff_members = additional_seats_required)
  : num_staff_members = 44 := 
sorry

end min_staff_members_l1513_151393


namespace incorrect_transformation_D_l1513_151335

theorem incorrect_transformation_D (x y m : ℝ) (hxy: x = y) : m = 0 → ¬ (x / m = y / m) :=
by
  intro hm
  simp [hm]
  -- Lean's simp tactic simplifies known equalities
  -- The simp tactic will handle the contradiction case directly when m = 0.
  sorry

end incorrect_transformation_D_l1513_151335


namespace sum_of_ages_l1513_151321

-- Problem statement:
-- Given: The product of their ages is 144.
-- Prove: The sum of their ages is 16.
theorem sum_of_ages (k t : ℕ) (htwins : t > k) (hprod : 2 * t * k = 144) : 2 * t + k = 16 := 
sorry

end sum_of_ages_l1513_151321


namespace total_books_proof_l1513_151328

def initial_books : ℝ := 41.0
def added_books_first : ℝ := 33.0
def added_books_next : ℝ := 2.0

theorem total_books_proof : initial_books + added_books_first + added_books_next = 76.0 :=
by
  sorry

end total_books_proof_l1513_151328


namespace relationship_of_abc_l1513_151334

theorem relationship_of_abc (a b c : ℕ) (ha : a = 2) (hb : b = 3) (hc : c = 4) : c > b ∧ b > a := by
  sorry

end relationship_of_abc_l1513_151334


namespace find_z_l1513_151323

-- Condition: there exists a constant k such that z = k * w
def direct_variation (z w : ℝ): Prop := ∃ k, z = k * w

-- We set up the conditions given in the problem.
theorem find_z (k : ℝ) (hw1 : 10 = k * 5) (hw2 : w = -15) : direct_variation z w → z = -30 :=
by
  sorry

end find_z_l1513_151323


namespace tan_of_negative_7pi_over_4_l1513_151383

theorem tan_of_negative_7pi_over_4 : Real.tan (-7 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_of_negative_7pi_over_4_l1513_151383


namespace students_left_in_final_year_l1513_151336

variable (s10 s_next s_final x : Nat)

-- Conditions
def initial_students : Prop := s10 = 150
def students_after_joining : Prop := s_next = s10 + 30
def students_final_year : Prop := s_final = s_next - x
def final_year_students : Prop := s_final = 165

-- Theorem to prove
theorem students_left_in_final_year (h1 : initial_students s10)
                                     (h2 : students_after_joining s10 s_next)
                                     (h3 : students_final_year s_next s_final x)
                                     (h4 : final_year_students s_final) :
  x = 15 :=
by
  sorry

end students_left_in_final_year_l1513_151336


namespace odd_periodic_function_l1513_151310

noncomputable def f : ℤ → ℤ := sorry

theorem odd_periodic_function (f_odd : ∀ x : ℤ, f (-x) = -f x)
  (period_f_3x1 : ∀ x : ℤ, f (3 * x + 1) = f (3 * (x + 3) + 1))
  (f_one : f 1 = -1) : f 2006 = 1 :=
sorry

end odd_periodic_function_l1513_151310


namespace max_value_of_m_l1513_151300

theorem max_value_of_m
  (a b : ℝ)
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : (2 / a) + (1 / b) = 1 / 4)
  (h4 : ∀ a b, 2 * a + b ≥ 9 * m) :
  m = 4 := 
sorry

end max_value_of_m_l1513_151300


namespace lcm_9_12_15_l1513_151356

theorem lcm_9_12_15 :
  let n := 9
  let m := 12
  let p := 15
  let prime_factors_n := (3, 2)  -- 9 = 3^2
  let prime_factors_m := ((2, 2), (3, 1))  -- 12 = 2^2 * 3
  let prime_factors_p := ((3, 1), (5, 1))  -- 15 = 3 * 5
  lcm n (lcm m p) = 180 := sorry

end lcm_9_12_15_l1513_151356


namespace at_least_one_did_not_land_stably_l1513_151392

-- Define the propositions p and q
variables (p q : Prop)

-- Define the theorem to prove
theorem at_least_one_did_not_land_stably :
  (¬p ∨ ¬q) ↔ ¬(p ∧ q) :=
by
  sorry

end at_least_one_did_not_land_stably_l1513_151392


namespace find_C_coordinates_l1513_151374

open Real

noncomputable def pointC_coordinates (A B : ℝ × ℝ) (hA : A = (-1, 0)) (hB : B = (3, 8)) (hdist : dist A C = 2 * dist C B) : ℝ × ℝ :=
  (⟨7 / 3, 20 / 3⟩)

theorem find_C_coordinates :
  ∀ (A B C : ℝ × ℝ), 
  A = (-1, 0) → B = (3, 8) → dist A C = 2 * dist C B →
  C = (7 / 3, 20 / 3) :=
by 
  intros A B C hA hB hdist
  -- We will use the given conditions and definitions to find the coordinates of C
  sorry

end find_C_coordinates_l1513_151374


namespace greatest_possible_mean_BC_l1513_151384

-- Mean weights for piles A, B
def mean_weight_A : ℝ := 60
def mean_weight_B : ℝ := 70

-- Combined mean weight for piles A and B
def mean_weight_AB : ℝ := 64

-- Combined mean weight for piles A and C
def mean_weight_AC : ℝ := 66

-- Prove that the greatest possible integer value for the mean weight of
-- the rocks in the combined piles B and C
theorem greatest_possible_mean_BC : ∃ (w : ℝ), (⌊w⌋ = 75) :=
by
  -- Definitions and assumptions based on problem conditions
  have h1 : mean_weight_A = 60 := rfl
  have h2 : mean_weight_B = 70 := rfl
  have h3 : mean_weight_AB = 64 := rfl
  have h4 : mean_weight_AC = 66 := rfl
  sorry

end greatest_possible_mean_BC_l1513_151384


namespace problem_part1_problem_part2_l1513_151351

theorem problem_part1 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  2 * ((1 / x) + (1 / y) + (1 / z)) ≤ (1 / p) + (1 / q) + (1 / r) :=
sorry

theorem problem_part2 
  (x y z p q r : ℝ)
  (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) 
  (h4 : 0 < p) (h5 : 0 < q) (h6 : 0 < r) :
  x * y + y * z + z * x ≥ 2 * (p * x + q * y + r * z) :=
sorry

end problem_part1_problem_part2_l1513_151351


namespace cars_needed_to_double_march_earnings_l1513_151341

-- Definition of given conditions
def base_salary : Nat := 1000
def commission_per_car : Nat := 200
def march_earnings : Nat := 2000

-- Question to prove
theorem cars_needed_to_double_march_earnings : 
  (2 * march_earnings - base_salary) / commission_per_car = 15 := 
by sorry

end cars_needed_to_double_march_earnings_l1513_151341


namespace fever_above_threshold_l1513_151365

-- Definitions as per conditions
def normal_temp : ℤ := 95
def temp_increase : ℤ := 10
def fever_threshold : ℤ := 100

-- Calculated new temperature
def new_temp := normal_temp + temp_increase

-- The proof statement, asserting the correct answer
theorem fever_above_threshold : new_temp - fever_threshold = 5 := 
by 
  sorry

end fever_above_threshold_l1513_151365


namespace wire_goes_around_field_l1513_151325

theorem wire_goes_around_field :
  (7348 / (4 * Real.sqrt 27889)) = 11 :=
by
  sorry

end wire_goes_around_field_l1513_151325


namespace forty_ab_l1513_151314

theorem forty_ab (a b : ℝ) (h₁ : 4 * a = 30) (h₂ : 5 * b = 30) : 40 * a * b = 1800 :=
by
  sorry

end forty_ab_l1513_151314


namespace geometric_seq_seventh_term_l1513_151378

theorem geometric_seq_seventh_term (a r : ℕ) (r_pos : r > 0) (first_term : a = 3)
    (fifth_term : a * r^4 = 243) : a * r^6 = 2187 := by
  sorry

end geometric_seq_seventh_term_l1513_151378


namespace trillion_in_scientific_notation_l1513_151363

theorem trillion_in_scientific_notation :
  (10^4) * (10^4) * (10^4) = 10^(12) := 
by sorry

end trillion_in_scientific_notation_l1513_151363


namespace exists_n_consecutive_composites_l1513_151367

theorem exists_n_consecutive_composites (n : ℕ) (h : n ≥ 1) (a r : ℕ) :
  ∃ K : ℕ, ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → ¬(Nat.Prime (a + (K + i) * r)) := 
sorry

end exists_n_consecutive_composites_l1513_151367


namespace angle_between_vectors_acute_l1513_151338

def isAcuteAngle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 > 0

def notCollinear (a b : ℝ × ℝ) : Prop :=
  ¬ ∃ k : ℝ, a = (k * b.1, k * b.2)

theorem angle_between_vectors_acute (m : ℝ) :
  let a := (-1, 1)
  let b := (2 * m, m + 3)
  isAcuteAngle a b ∧ notCollinear a b ↔ m < 3 ∧ m ≠ -1 :=
by
  sorry

end angle_between_vectors_acute_l1513_151338


namespace bob_clean_time_l1513_151361

-- Definitions for the problem conditions
def alice_time : ℕ := 30
def bob_time := (1 / 3 : ℚ) * alice_time

-- The proof problem statement (only) in Lean 4
theorem bob_clean_time : bob_time = 10 := by
  sorry

end bob_clean_time_l1513_151361
