import Mathlib

namespace NUMINAMATH_CALUDE_quadratic_function_k_value_l3279_327963

theorem quadratic_function_k_value (a b c k : ℤ) :
  let g := fun (x : ℤ) => a * x^2 + b * x + c
  g 1 = 0 ∧
  20 < g 5 ∧ g 5 < 30 ∧
  40 < g 6 ∧ g 6 < 50 ∧
  3000 * k < g 100 ∧ g 100 < 3000 * (k + 1) →
  k = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_k_value_l3279_327963


namespace NUMINAMATH_CALUDE_children_without_candy_l3279_327919

/-- Represents the number of children in the circle -/
def num_children : ℕ := 73

/-- Represents the total number of candies distributed -/
def total_candies : ℕ := 2020

/-- Calculates the position of the nth candy distribution -/
def candy_position (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Represents the number of unique positions reached -/
def unique_positions : ℕ := 37

theorem children_without_candy :
  num_children - unique_positions = 36 :=
sorry

end NUMINAMATH_CALUDE_children_without_candy_l3279_327919


namespace NUMINAMATH_CALUDE_sampling_probability_theorem_l3279_327942

/-- Represents the probability of a student being selected in a sampling process -/
def sampling_probability (total_students : ℕ) (selected_students : ℕ) : ℚ :=
  selected_students / total_students

/-- The sampling method described in the problem -/
structure SamplingMethod where
  total_students : ℕ
  selected_students : ℕ
  eliminated_students : ℕ

/-- Theorem stating that the probability of each student being selected is equal and is 25/1002 -/
theorem sampling_probability_theorem (method : SamplingMethod)
  (h1 : method.total_students = 2004)
  (h2 : method.selected_students = 50)
  (h3 : method.eliminated_students = 4) :
  sampling_probability method.total_students method.selected_students = 25 / 1002 :=
sorry

end NUMINAMATH_CALUDE_sampling_probability_theorem_l3279_327942


namespace NUMINAMATH_CALUDE_modified_full_house_probability_l3279_327970

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Represents the number of ranks in a standard deck -/
def NumberOfRanks : ℕ := 13

/-- Represents the number of cards per rank -/
def CardsPerRank : ℕ := 4

/-- Represents the number of cards drawn -/
def CardsDrawn : ℕ := 6

/-- Represents a modified full house -/
structure ModifiedFullHouse :=
  (rank1 : Fin NumberOfRanks)
  (rank2 : Fin NumberOfRanks)
  (rank3 : Fin NumberOfRanks)
  (h1 : rank1 ≠ rank2)
  (h2 : rank1 ≠ rank3)
  (h3 : rank2 ≠ rank3)

/-- The probability of drawing a modified full house -/
def probabilityModifiedFullHouse : ℚ :=
  24 / 2977

theorem modified_full_house_probability :
  probabilityModifiedFullHouse = (NumberOfRanks * (CardsPerRank.choose 3) * (NumberOfRanks - 1) * (CardsPerRank.choose 2) * ((NumberOfRanks - 2) * CardsPerRank)) / (StandardDeck.choose CardsDrawn) :=
by sorry

end NUMINAMATH_CALUDE_modified_full_house_probability_l3279_327970


namespace NUMINAMATH_CALUDE_inverse_f_sum_l3279_327928

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 2*x - x^2 + 1

theorem inverse_f_sum :
  let f_inv := Function.invFun f
  f_inv (-1) + f_inv 1 + f_inv 5 = 4 + Real.sqrt 3 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_sum_l3279_327928


namespace NUMINAMATH_CALUDE_sphere_radius_from_hole_l3279_327934

theorem sphere_radius_from_hole (hole_diameter : ℝ) (hole_depth : ℝ) (sphere_radius : ℝ) :
  hole_diameter = 30 →
  hole_depth = 12 →
  sphere_radius = (27 / 8 + 12) →
  sphere_radius = 15.375 := by
sorry

end NUMINAMATH_CALUDE_sphere_radius_from_hole_l3279_327934


namespace NUMINAMATH_CALUDE_evaluate_expression_l3279_327962

theorem evaluate_expression : 7 - 5 * (9 - 4^2) * 3 = 112 := by sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3279_327962


namespace NUMINAMATH_CALUDE_missing_data_point_l3279_327977

def linear_regression (x y : ℝ) := 0.28 * x + 0.16 = y

def data_points : List (ℝ × ℝ) := [(1, 0.5), (3, 1), (4, 1.4), (5, 1.5)]

theorem missing_data_point : 
  ∀ (a : ℝ), 
  (∀ (point : ℝ × ℝ), point ∈ data_points → linear_regression point.1 point.2) →
  linear_regression 2 a →
  linear_regression 3 ((0.5 + a + 1 + 1.4 + 1.5) / 5) →
  a = 0.6 := by sorry

end NUMINAMATH_CALUDE_missing_data_point_l3279_327977


namespace NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l3279_327993

theorem power_of_seven_mod_twelve : 7^145 % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_power_of_seven_mod_twelve_l3279_327993


namespace NUMINAMATH_CALUDE_soccer_tournament_games_l3279_327976

def soccer_tournament (n : ℕ) (m : ℕ) (tie_breaker : ℕ) : ℕ :=
  let first_stage := n * (n - 1) / 2
  let second_stage := 2 * (m * (m - 1) / 2)
  first_stage + second_stage + tie_breaker

theorem soccer_tournament_games :
  soccer_tournament 25 10 1 = 391 := by
  sorry

end NUMINAMATH_CALUDE_soccer_tournament_games_l3279_327976


namespace NUMINAMATH_CALUDE_wage_increase_proof_l3279_327907

theorem wage_increase_proof (original_wage new_wage : ℝ) 
  (h1 : new_wage = 70)
  (h2 : new_wage = original_wage * (1 + 0.16666666666666664)) :
  original_wage = 60 := by
  sorry

end NUMINAMATH_CALUDE_wage_increase_proof_l3279_327907


namespace NUMINAMATH_CALUDE_gathering_handshakes_l3279_327926

/-- Represents the number of handshakes in a gathering with specific conditions -/
def handshakes_in_gathering (total_people : ℕ) (group_a : ℕ) (group_b : ℕ) 
  (group_b_connected : ℕ) (connections : ℕ) : ℕ :=
  let group_b_isolated := group_b - group_b_connected
  let handshakes_isolated_to_a := group_b_isolated * group_a
  let handshakes_connected_to_a := group_b_connected * (group_a - connections)
  let handshakes_within_b := (group_b * (group_b - 1)) / 2
  handshakes_isolated_to_a + handshakes_connected_to_a + handshakes_within_b

theorem gathering_handshakes : 
  handshakes_in_gathering 40 30 10 3 5 = 330 :=
sorry

end NUMINAMATH_CALUDE_gathering_handshakes_l3279_327926


namespace NUMINAMATH_CALUDE_min_product_of_prime_sum_l3279_327984

theorem min_product_of_prime_sum (m n p : ℕ) : 
  Prime m → Prime n → Prime p → 
  m ≠ n → m ≠ p → n ≠ p → 
  m + n = p → 
  (∀ m' n' p' : ℕ, Prime m' → Prime n' → Prime p' → 
    m' ≠ n' → m' ≠ p' → n' ≠ p' → 
    m' + n' = p' → m' * n' * p' ≥ m * n * p) → 
  m * n * p = 30 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_prime_sum_l3279_327984


namespace NUMINAMATH_CALUDE_smallest_multiple_of_seven_l3279_327987

theorem smallest_multiple_of_seven (x : ℕ) : 
  (∃ k : ℕ, x = 7 * k) ∧ 
  (x^2 > 150) ∧ 
  (x < 40) → 
  x = 14 :=
sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_seven_l3279_327987


namespace NUMINAMATH_CALUDE_fraction_problem_l3279_327906

theorem fraction_problem (n : ℕ) : 
  (n : ℚ) / (2 * n + 4) = 3 / 7 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l3279_327906


namespace NUMINAMATH_CALUDE_jerrys_average_increase_l3279_327995

theorem jerrys_average_increase (initial_average : ℝ) (fourth_test_score : ℝ) : 
  initial_average = 85 → fourth_test_score = 93 → 
  (4 * (initial_average + 2) = 3 * initial_average + fourth_test_score) := by
sorry

end NUMINAMATH_CALUDE_jerrys_average_increase_l3279_327995


namespace NUMINAMATH_CALUDE_log_inequality_l3279_327911

theorem log_inequality (a b c : ℝ) : 
  a = Real.log 6 / Real.log 3 →
  b = Real.log 10 / Real.log 5 →
  c = Real.log 14 / Real.log 7 →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_log_inequality_l3279_327911


namespace NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l3279_327968

theorem x_gt_2_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 2 → x^2 > 4) ∧ 
  ¬(∀ x : ℝ, x^2 > 4 → x > 2) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_2_sufficient_not_necessary_for_x_sq_gt_4_l3279_327968


namespace NUMINAMATH_CALUDE_triangle_area_two_solutions_l3279_327901

theorem triangle_area_two_solutions (A B C : ℝ) (AB AC : ℝ) :
  B = π / 6 →  -- 30 degrees in radians
  AB = 2 * Real.sqrt 3 →
  AC = 2 →
  let area := (1 / 2) * AB * AC * Real.sin A
  area = 2 * Real.sqrt 3 ∨ area = Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_two_solutions_l3279_327901


namespace NUMINAMATH_CALUDE_expected_value_S_l3279_327980

def num_boys : ℕ := 7
def num_girls : ℕ := 13
def total_people : ℕ := num_boys + num_girls

def prob_boy_girl : ℚ := (num_boys : ℚ) / total_people * (num_girls : ℚ) / (total_people - 1)
def prob_girl_boy : ℚ := (num_girls : ℚ) / total_people * (num_boys : ℚ) / (total_people - 1)

def prob_adjacent_pair : ℚ := prob_boy_girl + prob_girl_boy
def num_adjacent_pairs : ℕ := total_people - 1

theorem expected_value_S : (num_adjacent_pairs : ℚ) * prob_adjacent_pair = 91 / 10 := by
  sorry

end NUMINAMATH_CALUDE_expected_value_S_l3279_327980


namespace NUMINAMATH_CALUDE_least_non_lucky_multiple_of_10_l3279_327905

def sumOfDigits (n : ℕ) : ℕ := 
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

def isLucky (n : ℕ) : Prop := 
  n > 0 ∧ n % sumOfDigits n = 0

def isMultipleOf10 (n : ℕ) : Prop := 
  ∃ k : ℕ, n = 10 * k

theorem least_non_lucky_multiple_of_10 : 
  (∀ m : ℕ, m < 110 → isMultipleOf10 m → isLucky m) ∧ 
  isMultipleOf10 110 ∧ 
  ¬isLucky 110 := by
sorry

end NUMINAMATH_CALUDE_least_non_lucky_multiple_of_10_l3279_327905


namespace NUMINAMATH_CALUDE_cos_double_angle_problem_l3279_327951

theorem cos_double_angle_problem (α : ℝ) (h : Real.cos (π + α) = 2/5) :
  Real.cos (2 * α) = -17/25 := by sorry

end NUMINAMATH_CALUDE_cos_double_angle_problem_l3279_327951


namespace NUMINAMATH_CALUDE_eight_digit_integers_count_l3279_327936

theorem eight_digit_integers_count : 
  (Finset.range 9).card * (Finset.range 10).card ^ 7 = 90000000 := by
  sorry

end NUMINAMATH_CALUDE_eight_digit_integers_count_l3279_327936


namespace NUMINAMATH_CALUDE_lesser_fraction_l3279_327900

theorem lesser_fraction (x y : ℝ) (sum_eq : x + y = 5/6) (prod_eq : x * y = 1/8) :
  min x y = (5 - Real.sqrt 7) / 12 := by sorry

end NUMINAMATH_CALUDE_lesser_fraction_l3279_327900


namespace NUMINAMATH_CALUDE_animals_on_shore_l3279_327950

/-- Proves that given the initial numbers of animals and the conditions about drowning,
    the total number of animals that made it to shore is 35. -/
theorem animals_on_shore (initial_sheep initial_cows initial_dogs : ℕ)
                         (drowned_sheep : ℕ)
                         (h1 : initial_sheep = 20)
                         (h2 : initial_cows = 10)
                         (h3 : initial_dogs = 14)
                         (h4 : drowned_sheep = 3)
                         (h5 : drowned_sheep * 2 = initial_cows - (initial_cows - drowned_sheep * 2)) :
  initial_sheep - drowned_sheep + (initial_cows - drowned_sheep * 2) + initial_dogs = 35 := by
  sorry

#check animals_on_shore

end NUMINAMATH_CALUDE_animals_on_shore_l3279_327950


namespace NUMINAMATH_CALUDE_tournament_prize_orderings_l3279_327996

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of games played in the tournament -/
def num_games : ℕ := 5

/-- Represents the number of possible outcomes for each game -/
def outcomes_per_game : ℕ := 2

/-- Theorem stating the number of possible prize orderings in the tournament -/
theorem tournament_prize_orderings :
  (outcomes_per_game ^ num_games : ℕ) = 32 := by sorry

end NUMINAMATH_CALUDE_tournament_prize_orderings_l3279_327996


namespace NUMINAMATH_CALUDE_four_carpenters_in_five_hours_l3279_327944

/-- Represents the number of desks built by a given number of carpenters in a specific time -/
def desks_built (carpenters : ℕ) (hours : ℚ) : ℚ :=
  sorry

/-- Two carpenters can build 2 desks in 2.5 hours -/
axiom two_carpenters_rate : desks_built 2 (5/2) = 2

/-- All carpenters work at the same pace -/
axiom same_pace (c₁ c₂ : ℕ) (h : ℚ) :
  c₁ * desks_built c₂ h = c₂ * desks_built c₁ h

theorem four_carpenters_in_five_hours :
  desks_built 4 5 = 8 := by
  sorry

end NUMINAMATH_CALUDE_four_carpenters_in_five_hours_l3279_327944


namespace NUMINAMATH_CALUDE_area_XYZA_is_four_thirds_l3279_327975

/-- Right trapezoid PQRS with the given properties -/
structure RightTrapezoid where
  PQ : ℝ
  RS : ℝ
  PR : ℝ
  trisectPQ : ℝ → ℝ → ℝ  -- Function to represent trisection points on PQ
  trisectRS : ℝ → ℝ → ℝ  -- Function to represent trisection points on RS
  midpoint : ℝ → ℝ → ℝ   -- Function to calculate midpoint

/-- The area of quadrilateral XYZA in the right trapezoid -/
def areaXYZA (t : RightTrapezoid) : ℝ :=
  let X := t.midpoint 0 (t.trisectPQ 0 1)
  let Y := t.midpoint (t.trisectPQ 0 1) (t.trisectPQ 1 2)
  let Z := t.midpoint (t.trisectRS 1 2) (t.trisectRS 0 1)
  let A := t.midpoint (t.trisectRS 0 1) t.RS
  -- Area calculation would go here
  sorry

/-- Theorem stating that the area of XYZA is 4/3 -/
theorem area_XYZA_is_four_thirds (t : RightTrapezoid) 
    (h1 : t.PQ = 2) 
    (h2 : t.RS = 6) 
    (h3 : t.PR = 4) : 
  areaXYZA t = 4/3 := by
  sorry

end NUMINAMATH_CALUDE_area_XYZA_is_four_thirds_l3279_327975


namespace NUMINAMATH_CALUDE_first_term_formula_l3279_327952

theorem first_term_formula (p q : ℕ) (hp : p ≥ 2) (hq : q ≥ 2) :
  ∃ (rest : ℕ), p^q = (p^(q-1) - p + 1) + rest :=
sorry

end NUMINAMATH_CALUDE_first_term_formula_l3279_327952


namespace NUMINAMATH_CALUDE_wasted_meat_price_l3279_327969

def minimum_wage : ℝ := 8
def fruit_veg_price : ℝ := 4
def bread_price : ℝ := 1.5
def janitorial_wage : ℝ := 10
def fruit_veg_weight : ℝ := 15
def bread_weight : ℝ := 60
def meat_weight : ℝ := 20
def overtime_hours : ℝ := 10
def james_work_hours : ℝ := 50

theorem wasted_meat_price (meat_price : ℝ) : meat_price = 5 := by
  sorry

end NUMINAMATH_CALUDE_wasted_meat_price_l3279_327969


namespace NUMINAMATH_CALUDE_second_train_length_second_train_length_is_100_l3279_327933

/-- Calculates the length of the second train given the speeds of two trains moving in opposite directions, the length of the first train, and the time it takes for them to pass each other completely. -/
theorem second_train_length 
  (speed1 : ℝ) 
  (speed2 : ℝ) 
  (length1 : ℝ) 
  (pass_time : ℝ) : ℝ :=
  let relative_speed := speed1 + speed2
  let relative_speed_mps := relative_speed * 1000 / 3600
  let total_distance := relative_speed_mps * pass_time
  total_distance - length1

/-- Proves that the length of the second train is 100 meters under the given conditions. -/
theorem second_train_length_is_100 : 
  second_train_length 80 70 150 5.999520038396928 = 100 := by
  sorry

end NUMINAMATH_CALUDE_second_train_length_second_train_length_is_100_l3279_327933


namespace NUMINAMATH_CALUDE_sleep_increase_l3279_327960

theorem sleep_increase (initial_sleep : ℝ) (increase_fraction : ℝ) (final_sleep : ℝ) :
  initial_sleep = 6 →
  increase_fraction = 1/3 →
  final_sleep = initial_sleep + initial_sleep * increase_fraction →
  final_sleep = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sleep_increase_l3279_327960


namespace NUMINAMATH_CALUDE_expressions_equal_thirty_l3279_327961

theorem expressions_equal_thirty : 
  (6 * 6 - 6 = 30) ∧ 
  (5 * 5 + 5 = 30) ∧ 
  (33 - 3 = 30) ∧ 
  (3^3 + 3 = 30) := by
  sorry

#check expressions_equal_thirty

end NUMINAMATH_CALUDE_expressions_equal_thirty_l3279_327961


namespace NUMINAMATH_CALUDE_min_value_theorem_l3279_327918

theorem min_value_theorem (a b c d e f g h : ℝ) 
  (h1 : a * b * c * d = 16) 
  (h2 : e * f * g * h = 25) : 
  (a * e)^2 + (b * f)^2 + (c * g)^2 + (d * h)^2 ≥ 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3279_327918


namespace NUMINAMATH_CALUDE_square_perimeter_proof_l3279_327910

theorem square_perimeter_proof (p1 p2 p3 : ℝ) (h1 : p1 = 40) (h2 : p2 = 32) (h3 : p3 = 24)
  (h4 : (p3 / 4) ^ 2 = ((p1 / 4) ^ 2) - ((p2 / 4) ^ 2)) : p1 = 40 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_proof_l3279_327910


namespace NUMINAMATH_CALUDE_perfect_squares_between_200_and_600_l3279_327938

theorem perfect_squares_between_200_and_600 :
  (Finset.filter (fun n => 200 < n^2 ∧ n^2 < 600) (Finset.range 25)).card = 10 :=
by sorry

end NUMINAMATH_CALUDE_perfect_squares_between_200_and_600_l3279_327938


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3279_327947

/-- Arithmetic sequence type -/
structure ArithmeticSequence (α : Type*) [Add α] [Mul α] where
  first : α
  diff : α

/-- Sum of first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  n * (2 * seq.first + (n - 1) * seq.diff) / 2

/-- n-th term of an arithmetic sequence -/
def nth_term (seq : ArithmeticSequence ℚ) (n : ℕ) : ℚ :=
  seq.first + (n - 1) * seq.diff

theorem arithmetic_sequence_ratio 
  (a b : ArithmeticSequence ℚ) 
  (h : ∀ n : ℕ, sum_n a n / sum_n b n = (3 * n - 1) / (n + 3)) :
  nth_term a 8 / (nth_term b 5 + nth_term b 11) = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_ratio_l3279_327947


namespace NUMINAMATH_CALUDE_max_sin_x_sin_2x_l3279_327949

theorem max_sin_x_sin_2x (x : Real) (h : 0 < x ∧ x < π / 2) :
  (∀ y : Real, y = Real.sin x * Real.sin (2 * x) → y ≤ 4 * Real.sqrt 3 / 9) ∧
  (∃ y : Real, y = Real.sin x * Real.sin (2 * x) ∧ y = 4 * Real.sqrt 3 / 9) :=
sorry

end NUMINAMATH_CALUDE_max_sin_x_sin_2x_l3279_327949


namespace NUMINAMATH_CALUDE_minimum_fourth_exam_score_l3279_327965

def exam1 : ℕ := 86
def exam2 : ℕ := 82
def exam3 : ℕ := 89
def required_increase : ℚ := 2

def average (a b c d : ℕ) : ℚ := (a + b + c + d : ℚ) / 4

theorem minimum_fourth_exam_score :
  ∀ x : ℕ,
    (average exam1 exam2 exam3 x ≥ (exam1 + exam2 + exam3 : ℚ) / 3 + required_increase) ↔
    x ≥ 94 :=
by sorry

end NUMINAMATH_CALUDE_minimum_fourth_exam_score_l3279_327965


namespace NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l3279_327902

def f (x : ℝ) := abs x

theorem abs_is_even_and_decreasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y ∧ y ≤ 0 → f y < f x) :=
by sorry

end NUMINAMATH_CALUDE_abs_is_even_and_decreasing_l3279_327902


namespace NUMINAMATH_CALUDE_max_triangle_side_length_l3279_327992

theorem max_triangle_side_length (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →  -- Three different side lengths
  a + b + c = 24 →         -- Perimeter is 24
  a < b + c ∧ b < a + c ∧ c < a + b →  -- Triangle inequality
  a ≤ 11 ∧ b ≤ 11 ∧ c ≤ 11 :=
by sorry

end NUMINAMATH_CALUDE_max_triangle_side_length_l3279_327992


namespace NUMINAMATH_CALUDE_methane_hydrate_density_scientific_notation_l3279_327914

theorem methane_hydrate_density_scientific_notation :
  0.00092 = 9.2 * 10^(-4) := by sorry

end NUMINAMATH_CALUDE_methane_hydrate_density_scientific_notation_l3279_327914


namespace NUMINAMATH_CALUDE_max_value_of_expression_l3279_327958

theorem max_value_of_expression (x y : ℝ) (h : x + y = 5) :
  x^2 + 2*x*y ≤ 25 ∧ ∃ x y : ℝ, x + y = 5 ∧ x^2 + 2*x*y = 25 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l3279_327958


namespace NUMINAMATH_CALUDE_t_minus_s_eq_negative_19_583_l3279_327904

/-- The number of students in the school -/
def num_students : ℕ := 120

/-- The number of teachers in the school -/
def num_teachers : ℕ := 6

/-- The list of class enrollments -/
def class_enrollments : List ℕ := [60, 30, 10, 10, 5, 5]

/-- The average number of students per teacher -/
def t : ℚ := (num_students : ℚ) / num_teachers

/-- The average number of students per student -/
noncomputable def s : ℚ := (class_enrollments.map (λ x => x * x)).sum / num_students

/-- The difference between t and s -/
theorem t_minus_s_eq_negative_19_583 : t - s = -19583 / 1000 := by sorry

end NUMINAMATH_CALUDE_t_minus_s_eq_negative_19_583_l3279_327904


namespace NUMINAMATH_CALUDE_tim_dan_balloon_ratio_l3279_327931

theorem tim_dan_balloon_ratio :
  let dan_balloons : ℕ := 29
  let tim_balloons : ℕ := 203
  (tim_balloons / dan_balloons : ℚ) = 7 := by sorry

end NUMINAMATH_CALUDE_tim_dan_balloon_ratio_l3279_327931


namespace NUMINAMATH_CALUDE_iains_old_pennies_l3279_327935

/-- The number of pennies older than Iain -/
def oldPennies : ℕ := 30

/-- The initial number of pennies Iain has -/
def initialPennies : ℕ := 200

/-- The number of pennies Iain has left after removing old pennies and 20% of the remaining -/
def remainingPennies : ℕ := 136

/-- The percentage of remaining pennies Iain throws out -/
def throwOutPercentage : ℚ := 1/5

theorem iains_old_pennies :
  oldPennies = initialPennies - (remainingPennies / (1 - throwOutPercentage)) := by
  sorry

end NUMINAMATH_CALUDE_iains_old_pennies_l3279_327935


namespace NUMINAMATH_CALUDE_joey_sneaker_purchase_l3279_327971

/-- The number of collectible figures Joey needs to sell to buy sneakers -/
def figures_to_sell (sneaker_cost lawn_count lawn_pay job_hours job_pay figure_price : ℕ) : ℕ :=
  let lawn_earnings := lawn_count * lawn_pay
  let job_earnings := job_hours * job_pay
  let total_earnings := lawn_earnings + job_earnings
  let remaining_amount := sneaker_cost - total_earnings
  (remaining_amount + figure_price - 1) / figure_price

theorem joey_sneaker_purchase :
  figures_to_sell 92 3 8 10 5 9 = 2 := by
  sorry

end NUMINAMATH_CALUDE_joey_sneaker_purchase_l3279_327971


namespace NUMINAMATH_CALUDE_expression_value_l3279_327964

theorem expression_value (a : ℝ) (h1 : 1 < a) (h2 : a < 2) :
  Real.sqrt ((a - 2)^2) + |1 - a| = 1 := by sorry

end NUMINAMATH_CALUDE_expression_value_l3279_327964


namespace NUMINAMATH_CALUDE_zongzi_probability_theorem_l3279_327927

/-- Given a set of 6 items where 2 are of type A and 4 are of type B -/
def total_items : ℕ := 6
def type_A_items : ℕ := 2
def type_B_items : ℕ := 4
def selected_items : ℕ := 3

/-- Probability of selecting at least one item of type A -/
def prob_at_least_one_A : ℚ := 4/5

/-- Probability distribution of X (number of type A items selected) -/
def prob_dist_X : List (ℕ × ℚ) := [(0, 1/5), (1, 3/5), (2, 1/5)]

/-- Mathematical expectation of X -/
def expectation_X : ℚ := 1

/-- Main theorem -/
theorem zongzi_probability_theorem :
  (total_items = type_A_items + type_B_items) →
  (prob_at_least_one_A = 4/5) ∧
  (prob_dist_X = [(0, 1/5), (1, 3/5), (2, 1/5)]) ∧
  (expectation_X = 1) := by
  sorry


end NUMINAMATH_CALUDE_zongzi_probability_theorem_l3279_327927


namespace NUMINAMATH_CALUDE_q_div_p_equals_225_l3279_327921

/-- The number of cards in the box -/
def total_cards : ℕ := 50

/-- The number of distinct numbers on the cards -/
def distinct_numbers : ℕ := 10

/-- The number of cards for each number -/
def cards_per_number : ℕ := 5

/-- The number of cards drawn -/
def cards_drawn : ℕ := 5

/-- The probability of drawing 5 cards with the same number -/
def p : ℚ := (distinct_numbers : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The probability of drawing 4 cards of one number and 1 card of a different number -/
def q : ℚ := (distinct_numbers * (distinct_numbers - 1) * Nat.choose cards_per_number 4 * Nat.choose cards_per_number 1 : ℚ) / (Nat.choose total_cards cards_drawn)

/-- The theorem stating that q/p = 225 -/
theorem q_div_p_equals_225 : q / p = 225 := by sorry

end NUMINAMATH_CALUDE_q_div_p_equals_225_l3279_327921


namespace NUMINAMATH_CALUDE_quadratic_solution_value_l3279_327974

theorem quadratic_solution_value (a b : ℝ) : 
  (1 : ℝ)^2 + a * (1 : ℝ) + 2 * b = 0 → 2023 - a - 2 * b = 2024 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_solution_value_l3279_327974


namespace NUMINAMATH_CALUDE_b_current_age_l3279_327954

/-- Given two people A and B, where:
    1) In 10 years, A will be twice as old as B was 10 years ago.
    2) A is now 5 years older than B.
    Prove that B's current age is 35 years. -/
theorem b_current_age (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))
  (h2 : a = b + 5) : 
  b = 35 := by
  sorry

end NUMINAMATH_CALUDE_b_current_age_l3279_327954


namespace NUMINAMATH_CALUDE_inscribed_cube_volume_l3279_327908

/-- A pyramid with an equilateral triangular base and isosceles lateral faces -/
structure Pyramid :=
  (base_side : ℝ)
  (height : ℝ)
  (is_equilateral_base : base_side > 0)
  (is_isosceles_lateral : height^2 + (base_side/2)^2 = (base_side * Real.sqrt 3 / 2)^2)

/-- A cube inscribed in the pyramid -/
structure InscribedCube (p : Pyramid) :=
  (side_length : ℝ)
  (touches_base_center : side_length ≤ p.base_side * Real.sqrt 3 / 3)
  (touches_apex : 2 * side_length = p.height)

/-- The volume of the inscribed cube is 1/64 -/
theorem inscribed_cube_volume (p : Pyramid) (c : InscribedCube p) 
    (h1 : p.base_side = 1) : c.side_length^3 = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_cube_volume_l3279_327908


namespace NUMINAMATH_CALUDE_fraction_equals_five_l3279_327979

theorem fraction_equals_five (a b k : ℕ+) : 
  (a.val^2 + b.val^2) / (a.val * b.val - 1) = k.val → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equals_five_l3279_327979


namespace NUMINAMATH_CALUDE_group_five_frequency_l3279_327973

theorem group_five_frequency (total : ℕ) (group1 group2 group3 group4 : ℕ) 
  (h_total : total = 50)
  (h_group1 : group1 = 2)
  (h_group2 : group2 = 8)
  (h_group3 : group3 = 15)
  (h_group4 : group4 = 5) :
  (total - group1 - group2 - group3 - group4 : ℚ) / total = 0.4 := by
sorry

end NUMINAMATH_CALUDE_group_five_frequency_l3279_327973


namespace NUMINAMATH_CALUDE_tammy_earnings_l3279_327978

/-- Calculates Tammy's earnings from selling oranges over a period of time. -/
def orange_earnings (num_trees : ℕ) (oranges_per_tree : ℕ) (oranges_per_pack : ℕ) 
  (price_per_pack : ℕ) (num_days : ℕ) : ℕ :=
  let oranges_per_day := num_trees * oranges_per_tree
  let packs_per_day := oranges_per_day / oranges_per_pack
  let total_packs := packs_per_day * num_days
  total_packs * price_per_pack

/-- Proves that Tammy's earnings after 3 weeks equal $840. -/
theorem tammy_earnings : 
  orange_earnings 10 12 6 2 (3 * 7) = 840 := by
  sorry

end NUMINAMATH_CALUDE_tammy_earnings_l3279_327978


namespace NUMINAMATH_CALUDE_fill_fraction_in_three_minutes_l3279_327932

/-- Represents the fraction of a cistern filled in a given time -/
def fractionFilled (totalTime minutes : ℚ) : ℚ :=
  minutes / totalTime

theorem fill_fraction_in_three_minutes :
  let totalTime : ℚ := 33
  let minutes : ℚ := 3
  fractionFilled totalTime minutes = 1 / 11 := by
  sorry

end NUMINAMATH_CALUDE_fill_fraction_in_three_minutes_l3279_327932


namespace NUMINAMATH_CALUDE_max_sum_consecutive_integers_l3279_327929

/-- Given consecutive integers x, y, and z satisfying 1/x + 1/y + 1/z > 1/45,
    the maximum value of x + y + z is 402. -/
theorem max_sum_consecutive_integers (x y z : ℤ) :
  (y = x + 1) →
  (z = y + 1) →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z > (1 : ℚ) / 45 →
  ∀ a b c : ℤ, (b = a + 1) → (c = b + 1) →
    (1 : ℚ) / a + (1 : ℚ) / b + (1 : ℚ) / c > (1 : ℚ) / 45 →
    x + y + z ≥ a + b + c →
  x + y + z = 402 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_consecutive_integers_l3279_327929


namespace NUMINAMATH_CALUDE_richards_day2_distance_l3279_327985

/-- Richard's journey from Cincinnati to New York City -/
def richards_journey (day2_distance : ℝ) : Prop :=
  let total_distance : ℝ := 70
  let day1_distance : ℝ := 20
  let day3_distance : ℝ := 10
  let remaining_distance : ℝ := 36
  (day2_distance < day1_distance / 2) ∧
  (day1_distance + day2_distance + day3_distance + remaining_distance = total_distance)

theorem richards_day2_distance :
  ∃ (day2_distance : ℝ), richards_journey day2_distance ∧ day2_distance = 4 := by
  sorry

end NUMINAMATH_CALUDE_richards_day2_distance_l3279_327985


namespace NUMINAMATH_CALUDE_typing_time_calculation_l3279_327999

theorem typing_time_calculation (original_speed : ℕ) (speed_reduction : ℕ) (document_length : ℕ) :
  original_speed = 212 →
  speed_reduction = 40 →
  document_length = 3440 →
  (document_length : ℚ) / ((original_speed - speed_reduction) : ℚ) = 20 := by
  sorry

end NUMINAMATH_CALUDE_typing_time_calculation_l3279_327999


namespace NUMINAMATH_CALUDE_integral_x_squared_l3279_327953

theorem integral_x_squared : ∫ x in (0:ℝ)..1, x^2 = (1/3 : ℝ) := by sorry

end NUMINAMATH_CALUDE_integral_x_squared_l3279_327953


namespace NUMINAMATH_CALUDE_xyz_value_l3279_327956

theorem xyz_value (a b c x y z : ℂ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (eq1 : a = (b + c) / (x - 3))
  (eq2 : b = (a + c) / (y - 3))
  (eq3 : c = (a + b) / (z - 3))
  (eq4 : x * y + x * z + y * z = 9)
  (eq5 : x + y + z = 6) :
  x * y * z = -10 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l3279_327956


namespace NUMINAMATH_CALUDE_cubic_root_in_interval_l3279_327959

/-- Given a cubic equation with three real roots and a condition on its coefficients,
    prove that at least one root belongs to the interval [0, 2]. -/
theorem cubic_root_in_interval
  (a b c : ℝ)
  (has_three_real_roots : ∃ x y z : ℝ, ∀ t : ℝ, t^3 + a*t^2 + b*t + c = 0 ↔ t = x ∨ t = y ∨ t = z)
  (coef_sum_bound : 2 ≤ a + b + c ∧ a + b + c ≤ 0) :
  ∃ r : ℝ, r^3 + a*r^2 + b*r + c = 0 ∧ 0 ≤ r ∧ r ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_cubic_root_in_interval_l3279_327959


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3279_327909

/-- Given an ellipse C with equation x²/a² + y²/b² = 1 (where a > b > 0),
    with foci F₁ and F₂, and points A and B on the ellipse satisfying
    AF₁ = 3F₁B and ∠BAF₂ = 90°, prove that the eccentricity of the ellipse
    is √2/2. -/
theorem ellipse_eccentricity (a b : ℝ) (A B F₁ F₂ : ℝ × ℝ) :
  a > b ∧ b > 0 ∧
  (A.1^2 / a^2 + A.2^2 / b^2 = 1) ∧
  (B.1^2 / a^2 + B.2^2 / b^2 = 1) ∧
  (F₁.1^2 + F₁.2^2 = (a^2 - b^2)) ∧
  (F₂.1^2 + F₂.2^2 = (a^2 - b^2)) ∧
  (A - F₁ = 3 • (F₁ - B)) ∧
  ((A - B) • (A - F₂) = 0) →
  Real.sqrt ((a^2 - b^2) / a^2) = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3279_327909


namespace NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l3279_327939

theorem arctan_sum_equals_pi_fourth (a b : ℝ) : 
  a = (1 : ℝ) / 2 → 
  (a + 1) * (b + 1) = 2 → 
  Real.arctan a + Real.arctan b = π / 4 := by
  sorry

end NUMINAMATH_CALUDE_arctan_sum_equals_pi_fourth_l3279_327939


namespace NUMINAMATH_CALUDE_solve_equation_l3279_327925

/-- Custom operation for pairs of real numbers -/
def pairOp (a b c d : ℝ) : ℝ := a * c + b * d

theorem solve_equation (x : ℝ) (h : pairOp (2 * x) 3 3 (-1) = 3) : x = 1 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3279_327925


namespace NUMINAMATH_CALUDE_words_with_a_count_l3279_327955

def alphabet : Finset Char := {'A', 'B', 'C', 'D', 'E'}

def word_length : Nat := 3

def words_with_a (s : Finset Char) (n : Nat) : Nat :=
  s.card ^ n - (s.erase 'A').card ^ n

theorem words_with_a_count :
  words_with_a alphabet word_length = 61 := by
  sorry

end NUMINAMATH_CALUDE_words_with_a_count_l3279_327955


namespace NUMINAMATH_CALUDE_min_value_P_l3279_327989

theorem min_value_P (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^2 + 1/x + 1/y = 27/4) : 
  ∀ (P : ℝ), P = 15/x - 3/(4*y) → P ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_min_value_P_l3279_327989


namespace NUMINAMATH_CALUDE_quadratic_inequality_problem_l3279_327924

/-- Given that the inequality ax^2 + 5x - 2 > 0 has the solution set {x|1/2 < x < 2},
    prove the value of a and the solution set of ax^2 - 5x + a^2 - 1 > 0 -/
theorem quadratic_inequality_problem (a : ℝ) :
  (∀ x : ℝ, ax^2 + 5*x - 2 > 0 ↔ 1/2 < x ∧ x < 2) →
  (a = -2 ∧
   ∀ x : ℝ, a*x^2 - 5*x + a^2 - 1 > 0 ↔ -3 < x ∧ x < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_problem_l3279_327924


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l3279_327917

theorem sqrt_difference_equals_five_sixths :
  Real.sqrt (9 / 4) - Real.sqrt (4 / 9) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_five_sixths_l3279_327917


namespace NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_legs_l3279_327988

/-- Represents a right triangle with an inscribed circle -/
structure RightTriangleWithInscribedCircle where
  /-- Length of one leg -/
  a : ℝ
  /-- Length of the other leg -/
  b : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- Distance from center of inscribed circle to one acute angle vertex -/
  d1 : ℝ
  /-- Distance from center of inscribed circle to other acute angle vertex -/
  d2 : ℝ
  /-- a and b are positive -/
  ha : 0 < a
  hb : 0 < b
  /-- r is positive -/
  hr : 0 < r
  /-- d1 and d2 are positive -/
  hd1 : 0 < d1
  hd2 : 0 < d2
  /-- Relationship between leg length and distance to vertex -/
  h1 : d1^2 = r^2 + (a - r)^2
  h2 : d2^2 = r^2 + (b - r)^2

/-- The main theorem -/
theorem right_triangle_with_inscribed_circle_legs
  (t : RightTriangleWithInscribedCircle)
  (h1 : t.d1 = Real.sqrt 5)
  (h2 : t.d2 = Real.sqrt 10) :
  t.a = 4 ∧ t.b = 3 := by
  sorry


end NUMINAMATH_CALUDE_right_triangle_with_inscribed_circle_legs_l3279_327988


namespace NUMINAMATH_CALUDE_sine_symmetry_l3279_327912

/-- Given a sinusoidal function y = 2sin(3x + φ) with |φ| < π/2,
    if the line of symmetry is x = π/12, then φ = π/4 -/
theorem sine_symmetry (φ : Real) : 
  (|φ| < π/2) →
  (∀ x : Real, 2 * Real.sin (3*x + φ) = 2 * Real.sin (3*(π/6 - x) + φ)) →
  φ = π/4 := by
  sorry

end NUMINAMATH_CALUDE_sine_symmetry_l3279_327912


namespace NUMINAMATH_CALUDE_condition_relationship_l3279_327957

theorem condition_relationship (x : ℝ) :
  ¬(∀ x, (1 / x ≤ 1 → (1/3)^x ≥ (1/2)^x)) ∧
  ¬(∀ x, ((1/3)^x ≥ (1/2)^x → 1 / x ≤ 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l3279_327957


namespace NUMINAMATH_CALUDE_largest_absolute_value_l3279_327983

theorem largest_absolute_value : 
  let numbers : List ℤ := [4, -5, 0, -1]
  ∀ x ∈ numbers, |x| ≤ |-5| :=
by sorry

end NUMINAMATH_CALUDE_largest_absolute_value_l3279_327983


namespace NUMINAMATH_CALUDE_factorization_problems_l3279_327903

theorem factorization_problems (x y : ℝ) : 
  ((x^2 + y^2)^2 - 4*x^2*y^2 = (x + y)^2 * (x - y)^2) ∧ 
  (3*x^3 - 12*x^2*y + 12*x*y^2 = 3*x*(x - 2*y)^2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_problems_l3279_327903


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3279_327922

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 2 → x^2 - 1 > 0)) ↔ (∃ x : ℝ, x > 2 ∧ x^2 - 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3279_327922


namespace NUMINAMATH_CALUDE_mini_quiz_multiple_choice_count_l3279_327994

/-- The number of ways to answer 3 true-false questions where all answers cannot be the same -/
def truefalse_combinations : ℕ := 6

/-- The number of answer choices for each multiple-choice question -/
def multiple_choice_options : ℕ := 4

/-- The total number of ways to write the answer key -/
def total_combinations : ℕ := 96

/-- Proves that the number of multiple-choice questions is 2 -/
theorem mini_quiz_multiple_choice_count :
  ∃ (n : ℕ), truefalse_combinations * multiple_choice_options ^ n = total_combinations ∧ n = 2 := by
sorry

end NUMINAMATH_CALUDE_mini_quiz_multiple_choice_count_l3279_327994


namespace NUMINAMATH_CALUDE_root_sum_ratio_l3279_327937

theorem root_sum_ratio (k : ℝ) (a b : ℝ) : 
  (k * (a^2 - 2*a) + 3*a + 7 = 0) →
  (k * (b^2 - 2*b) + 3*b + 7 = 0) →
  (a / b + b / a = 3 / 4) →
  ∃ (k₁ k₂ : ℝ), k₁ / k₂ + k₂ / k₁ = 433.42 := by sorry

end NUMINAMATH_CALUDE_root_sum_ratio_l3279_327937


namespace NUMINAMATH_CALUDE_male_non_listeners_l3279_327972

/-- Radio station survey data -/
structure SurveyData where
  total_listeners : ℕ
  total_non_listeners : ℕ
  female_listeners : ℕ
  male_non_listeners : ℕ

/-- Theorem: The number of males who do not listen to the radio station is 105 -/
theorem male_non_listeners (data : SurveyData)
  (h1 : data.total_listeners = 160)
  (h2 : data.total_non_listeners = 200)
  (h3 : data.female_listeners = 75)
  (h4 : data.male_non_listeners = 105) :
  data.male_non_listeners = 105 := by
  sorry


end NUMINAMATH_CALUDE_male_non_listeners_l3279_327972


namespace NUMINAMATH_CALUDE_bananas_per_box_l3279_327997

/-- Given 40 bananas and 10 boxes, prove that the number of bananas per box is 4. -/
theorem bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) (h1 : total_bananas = 40) (h2 : total_boxes = 10) :
  total_bananas / total_boxes = 4 := by
  sorry

end NUMINAMATH_CALUDE_bananas_per_box_l3279_327997


namespace NUMINAMATH_CALUDE_sandy_molly_age_ratio_l3279_327967

/-- The ratio of Sandy's current age to Molly's current age is 4:3, given that Sandy will be 38 years old in 6 years and Molly is currently 24 years old. -/
theorem sandy_molly_age_ratio :
  let sandy_future_age : ℕ := 38
  let years_until_future : ℕ := 6
  let molly_current_age : ℕ := 24
  let sandy_current_age : ℕ := sandy_future_age - years_until_future
  (sandy_current_age : ℚ) / molly_current_age = 4 / 3 := by
sorry

end NUMINAMATH_CALUDE_sandy_molly_age_ratio_l3279_327967


namespace NUMINAMATH_CALUDE_expression_value_l3279_327940

theorem expression_value : (2023 : ℚ) / 2022 - 2022 / 2023 + 1 = 4098551 / (2022 * 2023) := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3279_327940


namespace NUMINAMATH_CALUDE_contingency_fund_amount_l3279_327966

def total_donation : ℚ := 240

def community_pantry_ratio : ℚ := 1/3
def crisis_fund_ratio : ℚ := 1/2
def livelihood_ratio : ℚ := 1/4

def community_pantry : ℚ := total_donation * community_pantry_ratio
def crisis_fund : ℚ := total_donation * crisis_fund_ratio

def remaining_after_main : ℚ := total_donation - community_pantry - crisis_fund
def livelihood_fund : ℚ := remaining_after_main * livelihood_ratio

def contingency_fund : ℚ := remaining_after_main - livelihood_fund

theorem contingency_fund_amount : contingency_fund = 30 := by
  sorry

end NUMINAMATH_CALUDE_contingency_fund_amount_l3279_327966


namespace NUMINAMATH_CALUDE_volume_of_S_l3279_327982

/-- A line in ℝ³ -/
structure Line3D where
  point : ℝ × ℝ × ℝ
  direction : ℝ × ℝ × ℝ

/-- Distance from a point to a line in ℝ³ -/
def distPointToLine (p : ℝ × ℝ × ℝ) (l : Line3D) : ℝ :=
  sorry

/-- Distance between two points in ℝ³ -/
def distBetweenPoints (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The set S as described in the problem -/
def S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) : Set (ℝ × ℝ × ℝ) :=
  {X | distPointToLine X ℓ ≥ 2 * distBetweenPoints X P}

/-- The volume of a set in ℝ³ -/
noncomputable def volume (s : Set (ℝ × ℝ × ℝ)) : ℝ :=
  sorry

theorem volume_of_S (ℓ : Line3D) (P : ℝ × ℝ × ℝ) (d : ℝ) 
    (h_d : d > 0) (h_dist : distPointToLine P ℓ = d) :
    volume (S ℓ P) = (16 * Real.pi * d^3) / (27 * Real.sqrt 3) :=
  sorry

end NUMINAMATH_CALUDE_volume_of_S_l3279_327982


namespace NUMINAMATH_CALUDE_inverse_A_times_B_l3279_327930

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℚ := !![0, 1; 2, 3]
def B : Matrix (Fin 2) (Fin 2) ℚ := !![2, 0; 1, 8]

theorem inverse_A_times_B :
  A⁻¹ * B = !![-(5/2), 4; 2, 0] := by sorry

end NUMINAMATH_CALUDE_inverse_A_times_B_l3279_327930


namespace NUMINAMATH_CALUDE_mystery_books_ratio_l3279_327913

def total_books : ℕ := 46
def top_section_books : ℕ := 12 + 8 + 4
def bottom_section_books : ℕ := total_books - top_section_books
def known_bottom_books : ℕ := 5 + 6
def mystery_books : ℕ := bottom_section_books - known_bottom_books

theorem mystery_books_ratio :
  (mystery_books : ℚ) / bottom_section_books = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_mystery_books_ratio_l3279_327913


namespace NUMINAMATH_CALUDE_min_same_score_competition_l3279_327920

/-- Represents a math competition with fill-in-the-blank and short-answer questions. -/
structure MathCompetition where
  fill_in_blank_count : Nat
  fill_in_blank_points : Nat
  short_answer_count : Nat
  short_answer_points : Nat
  participant_count : Nat

/-- Calculates the minimum number of participants with the same score. -/
def min_same_score (comp : MathCompetition) : Nat :=
  let max_score := comp.fill_in_blank_count * comp.fill_in_blank_points +
                   comp.short_answer_count * comp.short_answer_points
  let distinct_scores := (comp.fill_in_blank_count + 1) * (comp.short_answer_count + 1)
  (comp.participant_count + distinct_scores - 1) / distinct_scores

/-- Theorem stating the minimum number of participants with the same score
    in the given competition configuration. -/
theorem min_same_score_competition :
  let comp := MathCompetition.mk 8 4 6 7 400
  min_same_score comp = 8 := by
  sorry


end NUMINAMATH_CALUDE_min_same_score_competition_l3279_327920


namespace NUMINAMATH_CALUDE_center_coordinate_sum_l3279_327998

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 12*y - 39

/-- The center of a circle given by its equation -/
def CenterOfCircle (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 1

/-- Theorem: The sum of coordinates of the center of the given circle is 8 -/
theorem center_coordinate_sum :
  ∃ h k : ℝ, CenterOfCircle h k ∧ h + k = 8 := by sorry

end NUMINAMATH_CALUDE_center_coordinate_sum_l3279_327998


namespace NUMINAMATH_CALUDE_correct_additional_muffins_l3279_327948

/-- Calculates the additional muffins needed for a charity event -/
def additional_muffins_needed (target : ℕ) (arthur_baked : ℕ) (beatrice_baked : ℕ) (charles_baked : ℕ) : ℕ :=
  target - (arthur_baked + beatrice_baked + charles_baked)

/-- Proves the correctness of additional muffins calculations for three charity events -/
theorem correct_additional_muffins :
  (additional_muffins_needed 200 35 48 29 = 88) ∧
  (additional_muffins_needed 150 20 35 25 = 70) ∧
  (additional_muffins_needed 250 45 60 30 = 115) := by
  sorry

#eval additional_muffins_needed 200 35 48 29
#eval additional_muffins_needed 150 20 35 25
#eval additional_muffins_needed 250 45 60 30

end NUMINAMATH_CALUDE_correct_additional_muffins_l3279_327948


namespace NUMINAMATH_CALUDE_height_ratio_equals_similarity_ratio_l3279_327981

/-- Two similar triangles with heights h₁ and h₂ and similarity ratio 1:4 -/
structure SimilarTriangles where
  h₁ : ℝ
  h₂ : ℝ
  h₁_pos : h₁ > 0
  h₂_pos : h₂ > 0
  similarity_ratio : h₁ / h₂ = 1 / 4

/-- The ratio of heights of similar triangles with similarity ratio 1:4 is 1:4 -/
theorem height_ratio_equals_similarity_ratio (t : SimilarTriangles) :
  t.h₁ / t.h₂ = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_height_ratio_equals_similarity_ratio_l3279_327981


namespace NUMINAMATH_CALUDE_parabola_equation_correct_l3279_327986

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the left focus of the ellipse
def left_focus : ℝ × ℝ := (-1, 0)

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (0, 0)

-- Define the parabola equation
def parabola_eq (x y : ℝ) : Prop := y^2 = -4*x

-- Theorem statement
theorem parabola_equation_correct :
  ∀ x y : ℝ,
  ellipse x y →
  parabola_eq x y →
  (left_focus.1 < 0 ∧ left_focus.2 = 0) →
  (vertex.1 = 0 ∧ vertex.2 = 0) →
  ∃ p : ℝ, p = 2 ∧ y^2 = -2*p*x :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_correct_l3279_327986


namespace NUMINAMATH_CALUDE_quadratic_function_bounds_l3279_327916

/-- Given a quadratic function f and its derivative g, 
    prove bounds on c and g(x) when f is bounded on [-1, 1] -/
theorem quadratic_function_bounds 
  (a b c : ℝ) 
  (f : ℝ → ℝ) 
  (g : ℝ → ℝ) 
  (hf : ∀ x, f x = a * x^2 + b * x + c) 
  (hg : ∀ x, g x = a * x + b) 
  (hbound : ∀ x ∈ Set.Icc (-1) 1, |f x| ≤ 1) : 
  (|c| ≤ 1) ∧ (∀ x ∈ Set.Icc (-1) 1, |g x| ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bounds_l3279_327916


namespace NUMINAMATH_CALUDE_remainder_theorem_l3279_327945

theorem remainder_theorem (P D Q R D' Q' R' : ℕ) 
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : R < D) :
  P % (D + D') = R :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3279_327945


namespace NUMINAMATH_CALUDE_factor_x9_minus_512_l3279_327923

theorem factor_x9_minus_512 (x : ℝ) : x^9 - 512 = (x^3 - 2) * (x^6 + 2*x^3 + 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_x9_minus_512_l3279_327923


namespace NUMINAMATH_CALUDE_domino_rearrangement_l3279_327991

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_covered : Bool)
  (empty_corner : Nat × Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a given position is a corner of the chessboard -/
def is_corner (board : Chessboard) (pos : Nat × Nat) : Prop :=
  (pos.1 = 1 ∨ pos.1 = board.size) ∧ (pos.2 = 1 ∨ pos.2 = board.size)

/-- Main theorem statement -/
theorem domino_rearrangement 
  (board : Chessboard) 
  (domino : Domino) 
  (h1 : board.size = 9)
  (h2 : domino.length = 1 ∧ domino.width = 2)
  (h3 : board.is_covered = true)
  (h4 : is_corner board board.empty_corner) :
  ∀ (corner : Nat × Nat), is_corner board corner → 
  ∃ (new_board : Chessboard), 
    new_board.size = board.size ∧ 
    new_board.is_covered = true ∧ 
    new_board.empty_corner = corner :=
sorry

end NUMINAMATH_CALUDE_domino_rearrangement_l3279_327991


namespace NUMINAMATH_CALUDE_nine_digit_sum_exists_l3279_327943

def is_nine_digit_permutation (n : ℕ) : Prop :=
  (n ≥ 100000000 ∧ n < 1000000000) ∧
  ∀ d : ℕ, d ∈ [1, 2, 3, 4, 5, 6, 7, 8, 9] → ∃ k : ℕ, n / 10^k % 10 = d

theorem nine_digit_sum_exists : ∃ a b : ℕ, 
  is_nine_digit_permutation a ∧ 
  is_nine_digit_permutation b ∧ 
  a + b = 987654321 :=
sorry

end NUMINAMATH_CALUDE_nine_digit_sum_exists_l3279_327943


namespace NUMINAMATH_CALUDE_student_weight_difference_l3279_327915

/-- Proves the difference in average weights given specific conditions about a group of students --/
theorem student_weight_difference (n : ℕ) (initial_avg : ℝ) (joe_weight : ℝ) (new_avg : ℝ) :
  initial_avg = 30 →
  joe_weight = 42 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  ((n + 1) * new_avg - 2 * initial_avg) / (n - 1) = initial_avg →
  abs (((n + 1) * new_avg - n * initial_avg) / 2 - joe_weight) = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_weight_difference_l3279_327915


namespace NUMINAMATH_CALUDE_f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l3279_327990

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem 1: When c = 0, f(-x) = -f(x) for all x
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f b 0 (-x) = -f b 0 x := by sorry

-- Theorem 2: When b = 0 and c > 0, f(x) = 0 has exactly one real root
theorem f_unique_root_when_b_zero_c_pos (c : ℝ) (hc : c > 0) :
  ∃! x, f 0 c x = 0 := by sorry

-- Theorem 3: The graph of y = f(x) is symmetric about (0, c)
theorem f_symmetric_about_0_c (b c : ℝ) :
  ∀ x, f b c (-x) + f b c x = 2 * c := by sorry

-- Theorem 4: There exists a case where f(x) = 0 has more than two real roots
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 := by sorry

end NUMINAMATH_CALUDE_f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l3279_327990


namespace NUMINAMATH_CALUDE_shirts_per_day_l3279_327941

theorem shirts_per_day (total_shirts : ℕ) (reused_shirts : ℕ) (vacation_days : ℕ) : 
  total_shirts = 11 → reused_shirts = 1 → vacation_days = 7 → 
  (total_shirts - reused_shirts) / (vacation_days - 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_shirts_per_day_l3279_327941


namespace NUMINAMATH_CALUDE_negation_equivalence_l3279_327946

theorem negation_equivalence (x y : ℝ) :
  ¬(x^2 + y^2 > 2 → |x| > 1 ∨ |y| > 1) ↔ (x^2 + y^2 ≤ 2 → |x| ≤ 1 ∧ |y| ≤ 1) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3279_327946
