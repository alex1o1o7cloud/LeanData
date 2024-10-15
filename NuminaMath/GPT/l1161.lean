import Mathlib

namespace NUMINAMATH_GPT_decrease_hours_worked_l1161_116139

theorem decrease_hours_worked (initial_hourly_wage : ℝ) (initial_hours_worked : ℝ) :
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  initial_hours_worked > 0 → 
  initial_hourly_wage > 0 → 
  new_hours_worked < initial_hours_worked :=
by
  intros initial_hours_worked_pos initial_hourly_wage_pos
  let new_hourly_wage := initial_hourly_wage * 1.25
  let new_hours_worked := (initial_hourly_wage * initial_hours_worked) / new_hourly_wage
  sorry

end NUMINAMATH_GPT_decrease_hours_worked_l1161_116139


namespace NUMINAMATH_GPT_at_least_three_points_in_circle_l1161_116149

noncomputable def point_in_circle (p : ℝ × ℝ) (c : ℝ × ℝ) (r : ℝ) : Prop :=
(dist p c) ≤ r

theorem at_least_three_points_in_circle (points : Fin 51 → (ℝ × ℝ)) (side_length : ℝ) (circle_radius : ℝ)
  (h_side_length : side_length = 1) (h_circle_radius : circle_radius = 1 / 7) : 
  ∃ (c : ℝ × ℝ), ∃ (p1 p2 p3 : Fin 51), 
    p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ 
    point_in_circle (points p1) c circle_radius ∧ 
    point_in_circle (points p2) c circle_radius ∧ 
    point_in_circle (points p3) c circle_radius :=
sorry

end NUMINAMATH_GPT_at_least_three_points_in_circle_l1161_116149


namespace NUMINAMATH_GPT_sum_of_integers_l1161_116176

theorem sum_of_integers (a b c d : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1)
    (h_prod : a * b * c * d = 1000000)
    (h_gcd1 : Nat.gcd a b = 1) (h_gcd2 : Nat.gcd a c = 1) (h_gcd3 : Nat.gcd a d = 1)
    (h_gcd4 : Nat.gcd b c = 1) (h_gcd5 : Nat.gcd b d = 1) (h_gcd6 : Nat.gcd c d = 1) : 
    a + b + c + d = 15698 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l1161_116176


namespace NUMINAMATH_GPT_exceeding_fraction_l1161_116161

def repeatingDecimal : ℚ := 8 / 33
def decimalFraction : ℚ := 6 / 25
def difference : ℚ := repeatingDecimal - decimalFraction

theorem exceeding_fraction :
  difference = 2 / 825 := by
  sorry

end NUMINAMATH_GPT_exceeding_fraction_l1161_116161


namespace NUMINAMATH_GPT_jean_total_jail_time_l1161_116173

def arson_counts := 3
def burglary_counts := 2
def petty_larceny_multiplier := 6
def arson_sentence_per_count := 36
def burglary_sentence_per_count := 18
def petty_larceny_fraction := 1/3

def total_jail_time :=
  arson_counts * arson_sentence_per_count +
  burglary_counts * burglary_sentence_per_count +
  (petty_larceny_multiplier * burglary_counts) * (petty_larceny_fraction * burglary_sentence_per_count)

theorem jean_total_jail_time : total_jail_time = 216 :=
by
  sorry

end NUMINAMATH_GPT_jean_total_jail_time_l1161_116173


namespace NUMINAMATH_GPT_hyperbola_asymptote_b_value_l1161_116189

theorem hyperbola_asymptote_b_value (b : ℝ) (hb : 0 < b) : 
  (∀ x y, x^2 - y^2 / b^2 = 1 → y = 3 * x ∨ y = -3 * x) → b = 3 := 
by
  sorry

end NUMINAMATH_GPT_hyperbola_asymptote_b_value_l1161_116189


namespace NUMINAMATH_GPT_stream_speed_l1161_116188

variable (D : ℝ) -- Distance rowed

theorem stream_speed (v : ℝ) (h : D / (60 - v) = 2 * (D / (60 + v))) : v = 20 :=
by
  sorry

end NUMINAMATH_GPT_stream_speed_l1161_116188


namespace NUMINAMATH_GPT_dishonest_dealer_uses_correct_weight_l1161_116180

noncomputable def dishonest_dealer_weight (profit_percent : ℝ) (true_weight : ℝ) : ℝ :=
  true_weight - (profit_percent / 100 * true_weight)

theorem dishonest_dealer_uses_correct_weight :
  dishonest_dealer_weight 11.607142857142861 1 = 0.8839285714285714 :=
by
  -- We skip the proof here
  sorry

end NUMINAMATH_GPT_dishonest_dealer_uses_correct_weight_l1161_116180


namespace NUMINAMATH_GPT_red_light_adds_3_minutes_l1161_116115

-- Definitions (conditions)
def first_route_time_if_all_green := 10
def second_route_time := 14
def additional_time_if_all_red := 5

-- Given that the first route is 5 minutes longer when all stoplights are red
def first_route_time_if_all_red := second_route_time + additional_time_if_all_red

-- Define red_light_time as the time each stoplight adds if it is red
def red_light_time := (first_route_time_if_all_red - first_route_time_if_all_green) / 3

-- Theorem (question == answer)
theorem red_light_adds_3_minutes :
  red_light_time = 3 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_red_light_adds_3_minutes_l1161_116115


namespace NUMINAMATH_GPT_hexagon_largest_angle_l1161_116183

theorem hexagon_largest_angle (x : ℝ) 
    (h_sum : (x + 2) + (2*x + 4) + (3*x - 6) + (4*x + 8) + (5*x - 10) + (6*x + 12) = 720) :
    (6*x + 12) = 215 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_largest_angle_l1161_116183


namespace NUMINAMATH_GPT_relationship_among_log_sin_exp_l1161_116130

theorem relationship_among_log_sin_exp (x : ℝ) (h₁ : 0 < x) (h₂ : x < 1) (a b c : ℝ) 
(h₃ : a = Real.log 3 / Real.log x) (h₄ : b = Real.sin x)
(h₅ : c = 2 ^ x) : a < b ∧ b < c := 
sorry

end NUMINAMATH_GPT_relationship_among_log_sin_exp_l1161_116130


namespace NUMINAMATH_GPT_quadratic_real_roots_m_l1161_116120

theorem quadratic_real_roots_m (m : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * x1 + 4 * x1 + m = 0 ∧ x2 * x2 + 4 * x2 + m = 0) →
  m ≤ 4 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_real_roots_m_l1161_116120


namespace NUMINAMATH_GPT_anne_trip_shorter_l1161_116171

noncomputable def john_walk_distance : ℝ := 2 + 1

noncomputable def anne_walk_distance : ℝ := Real.sqrt (2^2 + 1^2)

noncomputable def distance_difference : ℝ := john_walk_distance - anne_walk_distance

noncomputable def percentage_reduction : ℝ := (distance_difference / john_walk_distance) * 100

theorem anne_trip_shorter :
  20 ≤ percentage_reduction ∧ percentage_reduction < 30 :=
by
  sorry

end NUMINAMATH_GPT_anne_trip_shorter_l1161_116171


namespace NUMINAMATH_GPT_find_number_l1161_116108

def sum : ℕ := 2468 + 1375
def diff : ℕ := 2468 - 1375
def first_quotient : ℕ := 3 * diff
def second_quotient : ℕ := 5 * diff
def remainder : ℕ := 150

theorem find_number (N : ℕ) (h1 : sum = 3843) (h2 : diff = 1093) 
                    (h3 : first_quotient = 3279) (h4 : second_quotient = 5465)
                    (h5 : remainder = 150) (h6 : N = sum * first_quotient + remainder)
                    (h7 : N = sum * second_quotient + remainder) :
  N = 12609027 := 
by 
  sorry

end NUMINAMATH_GPT_find_number_l1161_116108


namespace NUMINAMATH_GPT_inequality_proof_l1161_116184

theorem inequality_proof
  (x y z : ℝ)
  (h_x : x ≥ 0)
  (h_y : y ≥ 0)
  (h_z : z > 0)
  (h_xy : x ≥ y)
  (h_yz : y ≥ z) :
  (x + y + z) * (x + y - z) * (x - y + z) / (x * y * z) ≥ 3 := by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1161_116184


namespace NUMINAMATH_GPT_evaluate_expression_1_evaluate_expression_2_l1161_116193

-- Problem 1
def expression_1 (a b : Int) : Int :=
  2 * a + 3 * b - 2 * a * b - a - 4 * b - a * b

theorem evaluate_expression_1 : expression_1 6 (-1) = 25 :=
by
  sorry

-- Problem 2
def expression_2 (m n : Int) : Int :=
  m^2 + 2 * m * n + n^2

theorem evaluate_expression_2 (m n : Int) (hm : |m| = 3) (hn : |n| = 2) (hmn : m < n) : expression_2 m n = 1 :=
by
  sorry

end NUMINAMATH_GPT_evaluate_expression_1_evaluate_expression_2_l1161_116193


namespace NUMINAMATH_GPT_area_of_garden_l1161_116127

theorem area_of_garden (L P : ℝ) (H1 : 1500 = 30 * L) (H2 : 1500 = 12 * P) (H3 : P = 2 * L + 2 * (P / 2 - L)) : 
  (L * (P/2 - L)) = 625 :=
by
  sorry

end NUMINAMATH_GPT_area_of_garden_l1161_116127


namespace NUMINAMATH_GPT_man_climbing_out_of_well_l1161_116116

theorem man_climbing_out_of_well (depth climb slip : ℕ) (h1 : depth = 30) (h2 : climb = 4) (h3 : slip = 3) : 
  let effective_climb_per_day := climb - slip
  let total_days := if depth % effective_climb_per_day = 0 then depth / effective_climb_per_day else depth / effective_climb_per_day + 1
  total_days = 30 :=
by
  sorry

end NUMINAMATH_GPT_man_climbing_out_of_well_l1161_116116


namespace NUMINAMATH_GPT_original_price_is_100_l1161_116117

variable (P : ℝ) -- Declare the original price P as a real number
variable (h : 0.10 * P = 10) -- The condition given in the problem

theorem original_price_is_100 (P : ℝ) (h : 0.10 * P = 10) : P = 100 := by
  sorry

end NUMINAMATH_GPT_original_price_is_100_l1161_116117


namespace NUMINAMATH_GPT_no_positive_integer_exists_l1161_116182

theorem no_positive_integer_exists
  (P1 P2 : ℤ → ℤ)
  (a : ℤ)
  (h_a_neg : a < 0)
  (h_common_root : P1 a = 0 ∧ P2 a = 0) :
  ¬ ∃ b : ℤ, b > 0 ∧ P1 b = 2007 ∧ P2 b = 2008 :=
sorry

end NUMINAMATH_GPT_no_positive_integer_exists_l1161_116182


namespace NUMINAMATH_GPT_total_weekly_allowance_l1161_116105

theorem total_weekly_allowance
  (total_students : ℕ)
  (students_6dollar : ℕ)
  (students_4dollar : ℕ)
  (students_7dollar : ℕ)
  (allowance_6dollar : ℕ)
  (allowance_4dollar : ℕ)
  (allowance_7dollar : ℕ)
  (days_in_week : ℕ) :
  total_students = 100 →
  students_6dollar = 60 →
  students_4dollar = 25 →
  students_7dollar = 15 →
  allowance_6dollar = 6 →
  allowance_4dollar = 4 →
  allowance_7dollar = 7 →
  days_in_week = 7 →
  (students_6dollar * allowance_6dollar + students_4dollar * allowance_4dollar + students_7dollar * allowance_7dollar) * days_in_week = 3955 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end NUMINAMATH_GPT_total_weekly_allowance_l1161_116105


namespace NUMINAMATH_GPT_min_value_of_expression_l1161_116196

theorem min_value_of_expression (a b c : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_abc : a * b * c = 4) :
  (3 * a + b) * (2 * b + 3 * c) * (a * c + 4) ≥ 384 := 
by sorry

end NUMINAMATH_GPT_min_value_of_expression_l1161_116196


namespace NUMINAMATH_GPT_worker_b_time_l1161_116164

theorem worker_b_time (T_B : ℝ) : 
  (1 / 10) + (1 / T_B) = 1 / 6 → T_B = 15 := by
  intro h
  sorry

end NUMINAMATH_GPT_worker_b_time_l1161_116164


namespace NUMINAMATH_GPT_complement_intersection_l1161_116174

open Set

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_intersection (hU : U = {2, 3, 6, 8}) (hA : A = {2, 3}) (hB : B = {2, 6, 8}) :
  ((U \ A) ∩ B) = {6, 8} := 
by
  sorry

end NUMINAMATH_GPT_complement_intersection_l1161_116174


namespace NUMINAMATH_GPT_linda_winning_probability_l1161_116129

noncomputable def probability_linda_wins : ℝ :=
  (1 / 16 : ℝ) / (1 - (1 / 32 : ℝ))

theorem linda_winning_probability :
  probability_linda_wins = 2 / 31 :=
sorry

end NUMINAMATH_GPT_linda_winning_probability_l1161_116129


namespace NUMINAMATH_GPT_find_interesting_numbers_l1161_116102

def is_interesting (A B : ℕ) : Prop :=
  A > B ∧ (∃ p : ℕ, Nat.Prime p ∧ A - B = p) ∧ ∃ n : ℕ, A * B = n ^ 2

theorem find_interesting_numbers :
  {A | (∃ B : ℕ, is_interesting A B) ∧ 200 < A ∧ A < 400} = {225, 256, 361} :=
by
  sorry

end NUMINAMATH_GPT_find_interesting_numbers_l1161_116102


namespace NUMINAMATH_GPT_evaluate_expression_l1161_116155

theorem evaluate_expression : 
  (1 - (2 / 5)) / (1 - (1 / 4)) = (4 / 5) := 
by 
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1161_116155


namespace NUMINAMATH_GPT_six_digit_number_l1161_116114

/-- 
Find a six-digit number that starts with the digit 1 and such that if this digit is moved to the end, the resulting number is three times the original number.
-/
theorem six_digit_number (N : ℕ) (h₁ : 100000 ≤ N ∧ N < 1000000) (h₂ : ∃ x : ℕ, N = 1 * 10^5 + x ∧ 10 * x + 1 = 3 * N) : N = 142857 :=
by sorry

end NUMINAMATH_GPT_six_digit_number_l1161_116114


namespace NUMINAMATH_GPT_square_area_with_circles_l1161_116109

theorem square_area_with_circles
  (radius : ℝ) 
  (side_length : ℝ)
  (area : ℝ)
  (h_radius : radius = 7) 
  (h_side_length : side_length = 2 * (2 * radius)) 
  (h_area : area = side_length ^ 2) : 
  area = 784 := by
  sorry

end NUMINAMATH_GPT_square_area_with_circles_l1161_116109


namespace NUMINAMATH_GPT_lunch_break_duration_l1161_116126

/-- Paula and her two helpers start at 7:00 AM and paint 60% of a house together,
    finishing at 5:00 PM. The next day, only the helpers paint and manage to
    paint 30% of another house, finishing at 3:00 PM. On the third day, Paula
    paints alone and paints the remaining 40% of the house, finishing at 4:00 PM.
    Prove that the length of their lunch break each day is 1 hour (60 minutes). -/
theorem lunch_break_duration :
  ∃ (L : ℝ), 
    (0 < L) ∧ 
    (L < 10) ∧
    (∃ (p h : ℝ), 
       (10 - L) * (p + h) = 0.6 ∧
       (8 - L) * h = 0.3 ∧
       (9 - L) * p = 0.4) ∧  
    L = 1 :=
by
  sorry

end NUMINAMATH_GPT_lunch_break_duration_l1161_116126


namespace NUMINAMATH_GPT_Margo_paired_with_Irma_probability_l1161_116177

noncomputable def probability_Margo_paired_with_Irma : ℚ :=
  1 / 29

theorem Margo_paired_with_Irma_probability :
  let total_students := 30
  let number_of_pairings := total_students - 1
  probability_Margo_paired_with_Irma = 1 / number_of_pairings := 
by
  sorry

end NUMINAMATH_GPT_Margo_paired_with_Irma_probability_l1161_116177


namespace NUMINAMATH_GPT_maximum_profit_is_achieved_at_14_yuan_l1161_116156

-- Define the initial conditions
def cost_per_unit : ℕ := 8
def initial_selling_price : ℕ := 10
def initial_selling_quantity : ℕ := 100

-- Define the sales volume decrease per price increase
def decrease_per_yuan_increase : ℕ := 10

-- Define the profit function
def profit (price_increase : ℕ) : ℕ :=
  let new_selling_price := initial_selling_price + price_increase
  let new_selling_quantity := initial_selling_quantity - (decrease_per_yuan_increase * price_increase)
  (new_selling_price - cost_per_unit) * new_selling_quantity

-- Define the statement to be proved
theorem maximum_profit_is_achieved_at_14_yuan :
  ∃ price_increase : ℕ, price_increase = 4 ∧ profit price_increase = profit 4 := by
  sorry

end NUMINAMATH_GPT_maximum_profit_is_achieved_at_14_yuan_l1161_116156


namespace NUMINAMATH_GPT_gcd_45_75_l1161_116158

theorem gcd_45_75 : Nat.gcd 45 75 = 15 :=
by
  sorry

end NUMINAMATH_GPT_gcd_45_75_l1161_116158


namespace NUMINAMATH_GPT_ratio_area_II_to_III_l1161_116112

-- Define the properties of the squares as given in the conditions
def perimeter_region_I : ℕ := 16
def perimeter_region_II : ℕ := 32
def side_length_region_I := perimeter_region_I / 4
def side_length_region_II := perimeter_region_II / 4
def side_length_region_III := 2 * side_length_region_II
def area_region_II := side_length_region_II ^ 2
def area_region_III := side_length_region_III ^ 2

-- Prove that the ratio of the area of region II to the area of region III is 1/4
theorem ratio_area_II_to_III : (area_region_II : ℚ) / (area_region_III : ℚ) = 1 / 4 := 
by sorry

end NUMINAMATH_GPT_ratio_area_II_to_III_l1161_116112


namespace NUMINAMATH_GPT_tangent_perpendicular_intersection_x_4_l1161_116165

noncomputable def f (x : ℝ) := (x^2 / 4) - (4 * Real.log x)
noncomputable def f' (x : ℝ) := (1/2 : ℝ) * x - 4 / x

theorem tangent_perpendicular_intersection_x_4 :
  ∀ x : ℝ, (0 < x) → (f' x = 1) → (x = 4) :=
by {
  sorry
}

end NUMINAMATH_GPT_tangent_perpendicular_intersection_x_4_l1161_116165


namespace NUMINAMATH_GPT_find_a_l1161_116103

noncomputable def A (a : ℝ) : Set ℝ := { x | a * x - 1 = 0 }
def B : Set ℝ := { x | x^2 - 3 * x + 2 = 0 }

theorem find_a (a : ℝ) :
  A a ∪ B = B ↔ a = 0 ∨ a = 1 ∨ a = 1 / 2 :=
sorry

end NUMINAMATH_GPT_find_a_l1161_116103


namespace NUMINAMATH_GPT_gillian_more_than_three_times_sandi_l1161_116119

-- Definitions of the conditions
def sandi_initial : ℕ := 600
def sandi_spent : ℕ := sandi_initial / 2
def gillian_spent : ℕ := 1050
def three_times_sandi_spent : ℕ := 3 * sandi_spent

-- Theorem statement with the proof to be added
theorem gillian_more_than_three_times_sandi :
  gillian_spent - three_times_sandi_spent = 150 := 
sorry

end NUMINAMATH_GPT_gillian_more_than_three_times_sandi_l1161_116119


namespace NUMINAMATH_GPT_dave_bought_packs_l1161_116100

def packs_of_white_shirts (bought_total : ℕ) (white_per_pack : ℕ) (blue_packs : ℕ) (blue_per_pack : ℕ) : ℕ :=
  (bought_total - blue_packs * blue_per_pack) / white_per_pack

theorem dave_bought_packs : packs_of_white_shirts 26 6 2 4 = 3 :=
by
  sorry

end NUMINAMATH_GPT_dave_bought_packs_l1161_116100


namespace NUMINAMATH_GPT_students_on_bleachers_l1161_116199

theorem students_on_bleachers (F B : ℕ) (h1 : F + B = 26) (h2 : F / (F + B) = 11 / 13) : B = 4 :=
by sorry

end NUMINAMATH_GPT_students_on_bleachers_l1161_116199


namespace NUMINAMATH_GPT_village_population_rate_l1161_116123

theorem village_population_rate (R : ℕ) :
  (76000 - 17 * R = 42000 + 17 * 800) → R = 1200 :=
by
  intro h
  -- The actual proof is omitted.
  sorry

end NUMINAMATH_GPT_village_population_rate_l1161_116123


namespace NUMINAMATH_GPT_area_triangle_QXY_l1161_116148

-- Definition of the problem
def length_rectangle (PQ PS : ℝ) : Prop :=
  PQ = 8 ∧ PS = 6

def diagonal_division (PR : ℝ) (X Y : ℝ) : Prop :=
  PR = 10 ∧ X = 2.5 ∧ Y = 2.5

-- The statement we need to prove
theorem area_triangle_QXY
  (PQ PS PR X Y : ℝ)
  (h1 : length_rectangle PQ PS)
  (h2 : diagonal_division PR X Y)
  : ∃ (A : ℝ), A = 6 := by
  sorry

end NUMINAMATH_GPT_area_triangle_QXY_l1161_116148


namespace NUMINAMATH_GPT_range_of_m_l1161_116191

theorem range_of_m (m : ℝ) (h : 9 > m^2 ∧ m ≠ 0) : m ∈ Set.Ioo (-3) 0 ∨ m ∈ Set.Ioo 0 3 := 
sorry

end NUMINAMATH_GPT_range_of_m_l1161_116191


namespace NUMINAMATH_GPT_smallest_n_inequality_l1161_116131

theorem smallest_n_inequality:
  ∃ n : ℤ, (∀ x y z : ℝ, (x^2 + 2 * y^2 + z^2)^2 ≤ n * (x^4 + 3 * y^4 + z^4)) ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_inequality_l1161_116131


namespace NUMINAMATH_GPT_rooms_needed_l1161_116192

/-
  We are given that there are 30 students and each hotel room accommodates 5 students.
  Prove that the number of rooms required to accommodate all students is 6.
-/
theorem rooms_needed (total_students : ℕ) (students_per_room : ℕ) (h1 : total_students = 30) (h2 : students_per_room = 5) : total_students / students_per_room = 6 := by
  -- proof
  sorry

end NUMINAMATH_GPT_rooms_needed_l1161_116192


namespace NUMINAMATH_GPT_travel_distance_l1161_116153

noncomputable def distance_traveled (AB BC : ℝ) : ℝ :=
  let BD := Real.sqrt (AB^2 + BC^2)
  let arc1 := (2 * Real.pi * BD) / 4
  let arc2 := (2 * Real.pi * AB) / 4
  arc1 + arc2

theorem travel_distance (hAB : AB = 3) (hBC : BC = 4) : 
  distance_traveled AB BC = 4 * Real.pi := by
    sorry

end NUMINAMATH_GPT_travel_distance_l1161_116153


namespace NUMINAMATH_GPT_distance_between_sets_is_zero_l1161_116133

noncomputable def A (x : ℝ) : ℝ := 2 * x - 1
noncomputable def B (x : ℝ) : ℝ := x^2 + 1

theorem distance_between_sets_is_zero : 
  ∃ (a b : ℝ), (∃ x₀ : ℝ, a = A x₀) ∧ (∃ y₀ : ℝ, b = B y₀) ∧ abs (a - b) = 0 := 
sorry

end NUMINAMATH_GPT_distance_between_sets_is_zero_l1161_116133


namespace NUMINAMATH_GPT_sin_x1_x2_value_l1161_116197

open Real

theorem sin_x1_x2_value (m x1 x2 : ℝ) :
  (2 * sin (2 * x1) + cos (2 * x1) = m) →
  (2 * sin (2 * x2) + cos (2 * x2) = m) →
  (0 ≤ x1 ∧ x1 ≤ π / 2) →
  (0 ≤ x2 ∧ x2 ≤ π / 2) →
  sin (x1 + x2) = 2 * sqrt 5 / 5 := 
by
  sorry

end NUMINAMATH_GPT_sin_x1_x2_value_l1161_116197


namespace NUMINAMATH_GPT_smallest_number_of_slices_l1161_116178

def cheddar_slices : ℕ := 12
def swiss_slices : ℕ := 28
def gouda_slices : ℕ := 18

theorem smallest_number_of_slices : Nat.lcm (Nat.lcm cheddar_slices swiss_slices) gouda_slices = 252 :=
by 
  sorry

end NUMINAMATH_GPT_smallest_number_of_slices_l1161_116178


namespace NUMINAMATH_GPT_find_value_of_x_l1161_116186

theorem find_value_of_x :
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := 
by
  sorry

end NUMINAMATH_GPT_find_value_of_x_l1161_116186


namespace NUMINAMATH_GPT_find_point_B_coordinates_l1161_116140

theorem find_point_B_coordinates : 
  ∃ B : ℝ × ℝ, 
    (∀ A C B : ℝ × ℝ, A = (2, 3) ∧ C = (0, 1) ∧ 
    (B.1 - A.1, B.2 - A.2) = (-2) • (C.1 - B.1, C.2 - B.2)) → B = (-2, -1) :=
by 
  sorry

end NUMINAMATH_GPT_find_point_B_coordinates_l1161_116140


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l1161_116166

theorem smallest_x_for_multiple (x : ℕ) (h720 : 720 = 2^4 * 3^2 * 5) (h1250 : 1250 = 2 * 5^4) : 
  (∃ x, (x > 0) ∧ (1250 ∣ (720 * x))) → x = 125 :=
by
  sorry

end NUMINAMATH_GPT_smallest_x_for_multiple_l1161_116166


namespace NUMINAMATH_GPT_infinite_perfect_squares_of_form_l1161_116132

theorem infinite_perfect_squares_of_form (k : ℕ) (hk : 0 < k) : 
  ∃ (n : ℕ), ∀ m : ℕ, ∃ a : ℕ, (n + m) * 2^k - 7 = a^2 :=
sorry

end NUMINAMATH_GPT_infinite_perfect_squares_of_form_l1161_116132


namespace NUMINAMATH_GPT_simplify_expression_l1161_116146

theorem simplify_expression (b : ℝ) : 3 * b * (3 * b ^ 2 + 2 * b) - 4 * b ^ 2 = 9 * b ^ 3 + 2 * b ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1161_116146


namespace NUMINAMATH_GPT_probability_same_color_l1161_116101

-- Definitions with conditions
def totalPlates := 11
def redPlates := 6
def bluePlates := 5

-- Calculate combinations
def choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The main statement
theorem probability_same_color (totalPlates redPlates bluePlates : ℕ) (h1 : totalPlates = 11) 
(h2 : redPlates = 6) (h3 : bluePlates = 5) : 
  (2 / 11 : ℚ) = ((choose redPlates 3 + choose bluePlates 3) : ℚ) / (choose totalPlates 3) :=
by
  -- Proof steps will be inserted here
  sorry

end NUMINAMATH_GPT_probability_same_color_l1161_116101


namespace NUMINAMATH_GPT_percentage_increase_mario_salary_is_zero_l1161_116104

variable (M : ℝ) -- Mario's salary last year
variable (P : ℝ) -- Percentage increase in Mario's salary

-- Condition 1: Mario's salary increased to $4000 this year
def mario_salary_increase (M P : ℝ) : Prop :=
  M * (1 + P / 100) = 4000 

-- Condition 2: Bob's salary last year was 3 times Mario's salary this year
def bob_salary_last_year (M : ℝ) : Prop :=
  3 * 4000 = 12000 

-- Condition 3: Bob's current salary is 20% more than his salary last year
def bob_current_salary : Prop :=
  12000 * 1.2 = 14400

-- Theorem : The percentage increase in Mario's salary is 0%
theorem percentage_increase_mario_salary_is_zero
  (h1 : mario_salary_increase M P)
  (h2 : bob_salary_last_year M)
  (h3 : bob_current_salary) : 
  P = 0 := 
sorry

end NUMINAMATH_GPT_percentage_increase_mario_salary_is_zero_l1161_116104


namespace NUMINAMATH_GPT_compute_value_l1161_116124

theorem compute_value
  (x y z : ℝ)
  (h1 : (xz / (x + y)) + (yx / (y + z)) + (zy / (z + x)) = -9)
  (h2 : (yz / (x + y)) + (zx / (y + z)) + (xy / (z + x)) = 15) :
  (y / (x + y)) + (z / (y + z)) + (x / (z + x)) = 13.5 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l1161_116124


namespace NUMINAMATH_GPT_arccos_cos_11_l1161_116195

theorem arccos_cos_11 : Real.arccos (Real.cos 11) = 1.425 :=
by
  sorry

end NUMINAMATH_GPT_arccos_cos_11_l1161_116195


namespace NUMINAMATH_GPT_inequality_reciprocal_of_negative_l1161_116172

variable {a b : ℝ}

theorem inequality_reciprocal_of_negative (h : a < b) (h_neg_a : a < 0) (h_neg_b : b < 0) : 
  (1 / a) > (1 / b) := by
  sorry

end NUMINAMATH_GPT_inequality_reciprocal_of_negative_l1161_116172


namespace NUMINAMATH_GPT_square_area_with_tangent_circles_l1161_116159

theorem square_area_with_tangent_circles :
  let r := 3 -- radius of each circle in inches
  let d := 2 * r -- diameter of each circle in inches
  let side_length := 2 * d -- side length of the square in inches
  let area := side_length * side_length -- area of the square in square inches
  side_length = 12 ∧ area = 144 :=
by
  let r := 3
  let d := 2 * r
  let side_length := 2 * d
  let area := side_length * side_length
  sorry

end NUMINAMATH_GPT_square_area_with_tangent_circles_l1161_116159


namespace NUMINAMATH_GPT_part1_part2_part3_l1161_116134

-- Part 1
theorem part1 (a b : ℝ) : 
    3 * (a - b) ^ 2 - 6 * (a - b) ^ 2 + 2 * (a - b) ^ 2 = - (a - b) ^ 2 := 
    sorry

-- Part 2
theorem part2 (x y : ℝ) (h : x ^ 2 - 2 * y = 4) : 
    3 * x ^ 2 - 6 * y - 21 = -9 := 
    sorry

-- Part 3
theorem part3 (a b c d : ℝ) (h1 : a - 5 * b = 3) (h2 : 5 * b - 3 * c = -5) (h3 : 3 * c - d = 10) : 
    (a - 3 * c) + (5 * b - d) - (5 * b - 3 * c) = 8 := 
    sorry

end NUMINAMATH_GPT_part1_part2_part3_l1161_116134


namespace NUMINAMATH_GPT_quadratic_solution_l1161_116190

theorem quadratic_solution (x : ℝ) : -x^2 + 4 * x + 5 < 0 ↔ x > 5 ∨ x < -1 :=
sorry

end NUMINAMATH_GPT_quadratic_solution_l1161_116190


namespace NUMINAMATH_GPT_equivalent_statements_l1161_116110

variable (P Q : Prop)

theorem equivalent_statements (h : P → Q) :
  (¬Q → ¬P) ∧ (¬P ∨ Q) :=
by 
  sorry

end NUMINAMATH_GPT_equivalent_statements_l1161_116110


namespace NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1161_116179

theorem smallest_angle_of_trapezoid (a d : ℝ) (h1 : a + 3 * d = 120) (h2 : 4 * a + 6 * d = 360) :
  a = 60 := by
  sorry

end NUMINAMATH_GPT_smallest_angle_of_trapezoid_l1161_116179


namespace NUMINAMATH_GPT_smallest_product_not_factor_60_l1161_116137

theorem smallest_product_not_factor_60 : ∃ (a b : ℕ), a ≠ b ∧ a ∣ 60 ∧ b ∣ 60 ∧ ¬ (a * b) ∣ 60 ∧ a * b = 8 := sorry

end NUMINAMATH_GPT_smallest_product_not_factor_60_l1161_116137


namespace NUMINAMATH_GPT_value_of_algebraic_expression_l1161_116169

noncomputable def quadratic_expression (m : ℝ) : ℝ :=
  3 * m * (2 * m - 3) - 1

theorem value_of_algebraic_expression (m : ℝ) (h : 2 * m^2 - 3 * m - 1 = 0) : quadratic_expression m = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_value_of_algebraic_expression_l1161_116169


namespace NUMINAMATH_GPT_positive_multiples_of_6_l1161_116160

theorem positive_multiples_of_6 (k a b : ℕ) (h₁ : a = (3 + 3 * k))
  (h₂ : b = 24) (h₃ : a^2 - b^2 = 0) : k = 7 :=
sorry

end NUMINAMATH_GPT_positive_multiples_of_6_l1161_116160


namespace NUMINAMATH_GPT_arithmetic_sequence_condition_l1161_116198

theorem arithmetic_sequence_condition (a b c d : ℝ) :
  (∃ k : ℝ, b = a + k ∧ c = a + 2*k ∧ d = a + 3*k) ↔ (a + d = b + c) :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_condition_l1161_116198


namespace NUMINAMATH_GPT_extreme_value_at_one_symmetric_points_range_l1161_116141

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
if x < 0 then
  x^2 + 3 * a * x
else
  2 * Real.exp x - x^2 + 2 * a * x

theorem extreme_value_at_one (a : ℝ) :
  (∀ x > 0, f x a = 2 * Real.exp x - x^2 + 2 * a * x) →
  (∀ x < 0, f x a = x^2 + 3 * a * x) →
  (∀ x > 0, deriv (fun x => f x a) x = 2 * Real.exp x - 2 * x + 2 * a) →
  deriv (fun x => f x a) 1 = 0 →
  a = 1 - Real.exp 1 :=
  sorry

theorem symmetric_points_range (a : ℝ) :
  (∃ x0 > 0, (∃ y0 : ℝ, 
  (f x0 a = y0 ∧ f (-x0) a = -y0))) →
  a ≥ 2 * Real.exp 1 :=
  sorry

end NUMINAMATH_GPT_extreme_value_at_one_symmetric_points_range_l1161_116141


namespace NUMINAMATH_GPT_total_apple_trees_is_800_l1161_116113

variable (T P A : ℕ) -- Total number of trees, peach trees, and apple trees respectively
variable (samples_peach samples_apple : ℕ) -- Sampled peach trees and apple trees respectively
variable (sampled_percentage : ℕ) -- Percentage of total trees sampled

-- Given conditions
axiom H1 : sampled_percentage = 10
axiom H2 : samples_peach = 50
axiom H3 : samples_apple = 80

-- Theorem to prove the number of apple trees
theorem total_apple_trees_is_800 : A = 800 :=
by sorry

end NUMINAMATH_GPT_total_apple_trees_is_800_l1161_116113


namespace NUMINAMATH_GPT_no_such_point_exists_l1161_116151

theorem no_such_point_exists 
  (side_length : ℝ)
  (original_area : ℝ)
  (total_area_after_first_rotation : ℝ)
  (total_area_after_second_rotation : ℝ)
  (no_overlapping_exists : Prop) :
  side_length = 12 → 
  original_area = 144 → 
  total_area_after_first_rotation = 211 → 
  total_area_after_second_rotation = 287 →
  no_overlapping_exists := sorry

end NUMINAMATH_GPT_no_such_point_exists_l1161_116151


namespace NUMINAMATH_GPT_find_x_l1161_116106

theorem find_x (x y : ℚ) (h1 : 3 * x - y = 9) (h2 : x + 4 * y = 11) : x = 47 / 13 :=
sorry

end NUMINAMATH_GPT_find_x_l1161_116106


namespace NUMINAMATH_GPT_carlson_total_land_l1161_116185

open Real

theorem carlson_total_land 
  (initial_land : ℝ)
  (cost_additional_land1 : ℝ)
  (cost_additional_land2 : ℝ)
  (cost_per_square_meter : ℝ) :
  initial_land = 300 →
  cost_additional_land1 = 8000 →
  cost_additional_land2 = 4000 →
  cost_per_square_meter = 20 →
  (initial_land + (cost_additional_land1 + cost_additional_land2) / cost_per_square_meter) = 900 :=
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_num
  done

end NUMINAMATH_GPT_carlson_total_land_l1161_116185


namespace NUMINAMATH_GPT_cube_volume_l1161_116122

theorem cube_volume (d_AF : Real) (h : d_AF = 6 * Real.sqrt 2) : ∃ (V : Real), V = 216 :=
by {
  sorry
}

end NUMINAMATH_GPT_cube_volume_l1161_116122


namespace NUMINAMATH_GPT_reciprocal_of_lcm_24_221_l1161_116157

theorem reciprocal_of_lcm_24_221 : (1 / Nat.lcm 24 221) = (1 / 5304) :=
by 
  sorry

end NUMINAMATH_GPT_reciprocal_of_lcm_24_221_l1161_116157


namespace NUMINAMATH_GPT_calculate_years_l1161_116118

variable {P R T SI : ℕ}

-- Conditions translations
def simple_interest_one_fifth (P SI : ℕ) : Prop :=
  SI = P / 5

def rate_of_interest (R : ℕ) : Prop :=
  R = 4

-- Proof of the number of years T
theorem calculate_years (h1 : simple_interest_one_fifth P SI)
                        (h2 : rate_of_interest R)
                        (h3 : SI = (P * R * T) / 100) : T = 5 :=
by
  sorry

end NUMINAMATH_GPT_calculate_years_l1161_116118


namespace NUMINAMATH_GPT_tangent_line_through_point_l1161_116136

theorem tangent_line_through_point (t : ℝ) :
    (∃ l : ℝ → ℝ, (∃ m : ℝ, (∀ x, l x = 2 * m * x - m^2) ∧ (t = m - 2 * m + 2 * m * m) ∧ m = 1/2) ∧ l t = 0)
    → t = 1/4 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_through_point_l1161_116136


namespace NUMINAMATH_GPT_max_distance_sum_l1161_116138

theorem max_distance_sum {P : ℝ × ℝ} 
  (C : Set (ℝ × ℝ)) 
  (hC : ∀ (P : ℝ × ℝ), P ∈ C ↔ (P.1 - 3)^2 + (P.2 - 4)^2 = 1)
  (A : ℝ × ℝ := (0, -1))
  (B : ℝ × ℝ := (0, 1)) :
  ∃ P : ℝ × ℝ, 
    P ∈ C ∧ (P = (18 / 5, 24 / 5)) :=
by
  sorry

end NUMINAMATH_GPT_max_distance_sum_l1161_116138


namespace NUMINAMATH_GPT_inequality_sqrt_three_l1161_116163

theorem inequality_sqrt_three (a b : ℤ) (h1 : a > b) (h2 : b > 1)
  (h3 : (a + b) ∣ (a * b + 1))
  (h4 : (a - b) ∣ (a * b - 1)) : a < Real.sqrt 3 * b := by
  sorry

end NUMINAMATH_GPT_inequality_sqrt_three_l1161_116163


namespace NUMINAMATH_GPT_number_of_white_balls_l1161_116175

theorem number_of_white_balls (total : ℕ) (freq_red freq_black : ℚ) (h1 : total = 120) 
                              (h2 : freq_red = 0.15) (h3 : freq_black = 0.45) : 
                              (total - total * freq_red - total * freq_black = 48) :=
by sorry

end NUMINAMATH_GPT_number_of_white_balls_l1161_116175


namespace NUMINAMATH_GPT_rectangular_garden_shorter_side_length_l1161_116145

theorem rectangular_garden_shorter_side_length
  (a b : ℕ)
  (h1 : 2 * a + 2 * b = 46)
  (h2 : a * b = 108) :
  b = 9 :=
by 
  sorry

end NUMINAMATH_GPT_rectangular_garden_shorter_side_length_l1161_116145


namespace NUMINAMATH_GPT_find_second_sum_l1161_116154

def sum : ℕ := 2717
def interest_rate_first : ℚ := 3 / 100
def interest_rate_second : ℚ := 5 / 100
def time_first : ℕ := 8
def time_second : ℕ := 3

theorem find_second_sum (x : ℚ) (h : x * interest_rate_first * time_first = (sum - x) * interest_rate_second * time_second) : 
  sum - x = 2449 :=
by
  sorry

end NUMINAMATH_GPT_find_second_sum_l1161_116154


namespace NUMINAMATH_GPT_find_g3_l1161_116194

-- Define a function g from ℝ to ℝ
variable (g : ℝ → ℝ)

-- Condition: ∀ x, g(3^x) + 2 * x * g(3^(-x)) = 3
axiom condition : ∀ x : ℝ, g (3^x) + 2 * x * g (3^(-x)) = 3

-- The theorem we need to prove
theorem find_g3 : g 3 = -3 := 
by 
  sorry

end NUMINAMATH_GPT_find_g3_l1161_116194


namespace NUMINAMATH_GPT_equiangular_hexagon_sides_l1161_116162

variable {a b c d e f : ℝ}

-- Definition of the equiangular hexagon condition
def equiangular_hexagon (a b c d e f : ℝ) := true

theorem equiangular_hexagon_sides (h : equiangular_hexagon a b c d e f) :
  a - d = e - b ∧ e - b = c - f :=
by
  sorry

end NUMINAMATH_GPT_equiangular_hexagon_sides_l1161_116162


namespace NUMINAMATH_GPT_polynomial_divisible_by_5040_l1161_116121

theorem polynomial_divisible_by_5040 (n : ℤ) (hn : n > 3) :
  5040 ∣ (n^7 - 14 * n^5 + 49 * n^3 - 36 * n) :=
sorry

end NUMINAMATH_GPT_polynomial_divisible_by_5040_l1161_116121


namespace NUMINAMATH_GPT_largest_N_cannot_pay_exactly_without_change_l1161_116128

theorem largest_N_cannot_pay_exactly_without_change
  (N : ℕ) (hN : N ≤ 50) :
  (¬ ∃ a b : ℕ, N = 5 * a + 6 * b) ↔ N = 19 := by
  sorry

end NUMINAMATH_GPT_largest_N_cannot_pay_exactly_without_change_l1161_116128


namespace NUMINAMATH_GPT_unit_digit_div_l1161_116147

theorem unit_digit_div (n : ℕ) : (33 * 10) % (2 ^ 1984) = n % 10 :=
by
  have h := 2 ^ 1984
  have u_digit_2_1984 := 6 -- Since 1984 % 4 = 0, last digit in the cycle of 2^n for n ≡ 0 [4] is 6
  sorry
  
example : (33 * 10) / (2 ^ 1984) % 10 = 6 :=
by sorry

end NUMINAMATH_GPT_unit_digit_div_l1161_116147


namespace NUMINAMATH_GPT_ellipse_equation_l1161_116107

-- Definitions of the tangents given as conditions
def tangent1 (x y : ℝ) : Prop := 4 * x + 5 * y = 25
def tangent2 (x y : ℝ) : Prop := 9 * x + 20 * y = 75

-- The statement we need to prove
theorem ellipse_equation :
  (∀ (x y : ℝ), tangent1 x y → tangent2 x y → 
  (∃ (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0), a = 5 ∧ b = 3 ∧ 
  (x^2 / a^2 + y^2 / b^2 = 1))) :=
sorry

end NUMINAMATH_GPT_ellipse_equation_l1161_116107


namespace NUMINAMATH_GPT_days_for_30_men_to_build_wall_l1161_116152

theorem days_for_30_men_to_build_wall 
  (men1 days1 men2 k : ℕ)
  (h1 : men1 = 18)
  (h2 : days1 = 5)
  (h3 : men2 = 30)
  (h_k : men1 * days1 = k)
  : (men2 * 3 = k) := by 
sorry

end NUMINAMATH_GPT_days_for_30_men_to_build_wall_l1161_116152


namespace NUMINAMATH_GPT_red_pieces_count_l1161_116125

-- Define the conditions
def total_pieces : ℕ := 3409
def blue_pieces : ℕ := 3264

-- Prove the number of red pieces
theorem red_pieces_count : total_pieces - blue_pieces = 145 :=
by sorry

end NUMINAMATH_GPT_red_pieces_count_l1161_116125


namespace NUMINAMATH_GPT_smallest_d_l1161_116170

-- Constants and conditions
variables (c d : ℝ)
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Conditions involving c and d
def conditions (c d : ℝ) : Prop :=
  2 < c ∧ c < d ∧ ¬triangle_inequality 2 c d ∧ ¬triangle_inequality (1/d) (1/c) 2

-- Goal statement: the smallest possible value of d
theorem smallest_d (c d : ℝ) (h : conditions c d) : d = 2 + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_smallest_d_l1161_116170


namespace NUMINAMATH_GPT_contradiction_even_odd_l1161_116168

theorem contradiction_even_odd (a b c : ℕ) (h1 : (a % 2 = 1 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ c % 2 = 1) ∨ (b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 1)) :
  (a % 2 = 0 ∧ b % 2 = 1 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0 ∧ c % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 1 ∧ c % 2 = 0) :=
by
  -- proof by contradiction
  sorry

end NUMINAMATH_GPT_contradiction_even_odd_l1161_116168


namespace NUMINAMATH_GPT_number_of_special_three_digit_numbers_l1161_116142

noncomputable def count_special_three_digit_numbers : ℕ :=
  Nat.choose 9 3

theorem number_of_special_three_digit_numbers : count_special_three_digit_numbers = 84 := by
  sorry

end NUMINAMATH_GPT_number_of_special_three_digit_numbers_l1161_116142


namespace NUMINAMATH_GPT_exist_non_negative_product_l1161_116143

theorem exist_non_negative_product (a1 a2 a3 a4 a5 a6 a7 a8 : ℝ) :
  0 ≤ a1 * a3 + a2 * a4 ∨
  0 ≤ a1 * a5 + a2 * a6 ∨
  0 ≤ a1 * a7 + a2 * a8 ∨
  0 ≤ a3 * a5 + a4 * a6 ∨
  0 ≤ a3 * a7 + a4 * a8 ∨
  0 ≤ a5 * a7 + a6 * a8 :=
sorry

end NUMINAMATH_GPT_exist_non_negative_product_l1161_116143


namespace NUMINAMATH_GPT_greatest_number_of_matching_pairs_l1161_116150

theorem greatest_number_of_matching_pairs 
  (original_pairs : ℕ := 27)
  (lost_shoes : ℕ := 9) 
  (remaining_pairs : ℕ := original_pairs - (lost_shoes / 1))
  : remaining_pairs = 18 := by
  sorry

end NUMINAMATH_GPT_greatest_number_of_matching_pairs_l1161_116150


namespace NUMINAMATH_GPT_eighth_term_l1161_116167

noncomputable def S (n : ℕ) (a : ℕ → ℤ) : ℤ := (n * (a 1 + a n)) / 2

variables {a : ℕ → ℤ} {d : ℤ}

-- Conditions
axiom sum_of_first_n_terms : ∀ n : ℕ, S n a = (n * (a 1 + a n)) / 2
axiom second_term : a 2 = 3
axiom sum_of_first_five_terms : S 5 a = 25

-- Question
theorem eighth_term : a 8 = 15 :=
sorry

end NUMINAMATH_GPT_eighth_term_l1161_116167


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l1161_116144

noncomputable def repeating_decimal := 4 + 36 / 99

theorem repeating_decimal_fraction : 
  repeating_decimal = 144 / 33 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l1161_116144


namespace NUMINAMATH_GPT_sequence_formula_l1161_116111

theorem sequence_formula (a : ℕ → ℚ) (n : ℕ) (h1 : a 1 = 1) (h2 : a 2 = -1/2) (h3 : a 3 = 1/3) (h4 : a 4 = -1/4) :
  a n = (-1)^(n+1) * (1/n) :=
sorry

end NUMINAMATH_GPT_sequence_formula_l1161_116111


namespace NUMINAMATH_GPT_sum_of_star_tip_angles_l1161_116181

noncomputable def sum_star_tip_angles : ℝ :=
  let segment_angle := 360 / 8
  let subtended_arc := 3 * segment_angle
  let theta := subtended_arc / 2
  8 * theta

theorem sum_of_star_tip_angles:
  sum_star_tip_angles = 540 := by
  sorry

end NUMINAMATH_GPT_sum_of_star_tip_angles_l1161_116181


namespace NUMINAMATH_GPT_value_of_stamp_collection_l1161_116135

theorem value_of_stamp_collection 
  (n m : ℕ) (v_m : ℝ)
  (hn : n = 18) 
  (hm : m = 6)
  (hv_m : v_m = 15)
  (uniform_value : ∀ (k : ℕ), k ≤ m → v_m / m = v_m / k):
  ∃ v_total : ℝ, v_total = 45 :=
by 
  sorry

end NUMINAMATH_GPT_value_of_stamp_collection_l1161_116135


namespace NUMINAMATH_GPT_algebra_eq_iff_sum_eq_one_l1161_116187

-- Definitions from conditions
def expr1 (a b c : ℝ) : ℝ := a + b * c
def expr2 (a b c : ℝ) : ℝ := (a + b) * (a + c)

-- Lean statement for the proof problem
theorem algebra_eq_iff_sum_eq_one (a b c : ℝ) : expr1 a b c = expr2 a b c ↔ a + b + c = 1 :=
by
  sorry

end NUMINAMATH_GPT_algebra_eq_iff_sum_eq_one_l1161_116187
