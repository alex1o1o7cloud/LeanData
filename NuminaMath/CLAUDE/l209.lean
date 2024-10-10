import Mathlib

namespace fraction_integer_iff_q_in_set_l209_20942

theorem fraction_integer_iff_q_in_set (q : ℕ+) :
  (∃ (k : ℕ+), (5 * q + 35 : ℤ) = k * (3 * q - 7)) ↔ 
  q ∈ ({3, 4, 5, 7, 9, 15, 21, 31} : Set ℕ+) := by
  sorry

end fraction_integer_iff_q_in_set_l209_20942


namespace complex_modulus_problem_l209_20909

theorem complex_modulus_problem (a : ℝ) (i : ℂ) : 
  i ^ 2 = -1 →
  (Complex.I : ℂ) ^ 2 = -1 →
  ((a - Real.sqrt 2 + i) / i).im = 0 →
  Complex.abs (2 * a + Complex.I * Real.sqrt 2) = Real.sqrt 10 := by
  sorry

end complex_modulus_problem_l209_20909


namespace bernie_chocolate_savings_l209_20994

/-- Calculates the savings over a given number of weeks when buying chocolates at a discounted price --/
def chocolate_savings (chocolates_per_week : ℕ) (regular_price discount_price : ℚ) (weeks : ℕ) : ℚ :=
  (chocolates_per_week * (regular_price - discount_price)) * weeks

/-- The savings over three weeks when buying two chocolates per week at a store with a $2 price instead of a store with a $3 price is equal to $6 --/
theorem bernie_chocolate_savings :
  chocolate_savings 2 3 2 3 = 6 := by
  sorry

end bernie_chocolate_savings_l209_20994


namespace carols_pool_water_carols_pool_water_proof_l209_20904

/-- Calculates the amount of water left in Carol's pool after five hours of filling and a leak -/
theorem carols_pool_water (first_hour_rate : ℕ) (second_third_hour_rate : ℕ) (fourth_hour_rate : ℕ) (leak_amount : ℕ) : ℕ :=
  let total_added := first_hour_rate + 2 * second_third_hour_rate + fourth_hour_rate
  total_added - leak_amount

/-- Proves that the amount of water left in Carol's pool after five hours is 34 gallons -/
theorem carols_pool_water_proof :
  carols_pool_water 8 10 14 8 = 34 := by
  sorry

end carols_pool_water_carols_pool_water_proof_l209_20904


namespace binomial_seven_four_l209_20951

theorem binomial_seven_four : Nat.choose 7 4 = 35 := by
  sorry

end binomial_seven_four_l209_20951


namespace fishing_problem_l209_20999

theorem fishing_problem (blaine_catch : ℕ) (keith_catch : ℕ) : 
  blaine_catch = 5 → 
  keith_catch = 2 * blaine_catch → 
  blaine_catch + keith_catch = 15 := by
sorry

end fishing_problem_l209_20999


namespace sum_with_gap_l209_20957

theorem sum_with_gap (x : ℝ) (h1 : |x - 5.46| = 3.97) (h2 : x < 5.46) : x + 5.46 = 6.95 := by
  sorry

end sum_with_gap_l209_20957


namespace rectangular_hall_dimension_difference_l209_20929

theorem rectangular_hall_dimension_difference 
  (length width : ℝ) 
  (width_half_length : width = length / 2)
  (area_constraint : length * width = 578) :
  length - width = 17 := by
sorry

end rectangular_hall_dimension_difference_l209_20929


namespace exponential_equation_solution_l209_20981

theorem exponential_equation_solution :
  ∃ x : ℝ, (16 : ℝ) ^ x * (16 : ℝ) ^ x * (16 : ℝ) ^ x = (256 : ℝ) ^ 3 ∧ x = 2 := by
  sorry

end exponential_equation_solution_l209_20981


namespace retail_markup_percentage_l209_20958

theorem retail_markup_percentage 
  (wholesale : ℝ) 
  (retail : ℝ) 
  (h1 : retail > 0) 
  (h2 : wholesale > 0) 
  (h3 : retail * 0.75 = wholesale * 1.3500000000000001) : 
  (retail / wholesale - 1) * 100 = 80.00000000000002 := by
sorry

end retail_markup_percentage_l209_20958


namespace elephant_weighing_l209_20907

/-- The weight of a stone block in catties -/
def stone_weight : ℕ := 240

/-- The number of stone blocks initially on the boat -/
def initial_stones : ℕ := 20

/-- The number of workers initially on the boat -/
def initial_workers : ℕ := 3

/-- The number of stone blocks after adjustment -/
def adjusted_stones : ℕ := 21

/-- The number of workers after adjustment -/
def adjusted_workers : ℕ := 1

/-- The weight of the elephant in catties -/
def elephant_weight : ℕ := 5160

theorem elephant_weighing :
  ∃ (worker_weight : ℕ),
    (initial_stones * stone_weight + initial_workers * worker_weight =
     adjusted_stones * stone_weight + adjusted_workers * worker_weight) ∧
    (elephant_weight = initial_stones * stone_weight + initial_workers * worker_weight) :=
by sorry

end elephant_weighing_l209_20907


namespace product_mod_fifty_l209_20963

theorem product_mod_fifty : ∃ m : ℕ, 0 ≤ m ∧ m < 50 ∧ (289 * 673) % 50 = m ∧ m = 47 := by
  sorry

end product_mod_fifty_l209_20963


namespace log_625_squared_base_5_l209_20908

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- State the theorem
theorem log_625_squared_base_5 : log 5 (625^2) = 8 := by
  sorry

end log_625_squared_base_5_l209_20908


namespace purely_imaginary_complex_number_l209_20983

theorem purely_imaginary_complex_number (m : ℝ) : 
  (Complex.I * Complex.I = -1) →
  (∃ (z : ℂ), z = m * (m + 1) + Complex.I * (m^2 - 1) ∧ z.re = 0 ∧ z.im ≠ 0) →
  m = 0 := by sorry

end purely_imaginary_complex_number_l209_20983


namespace intersection_complement_equality_l209_20970

universe u

def U : Set Nat := {1, 2, 3, 4, 5, 6, 7}
def M : Set Nat := {2, 3, 4, 5}
def N : Set Nat := {1, 4, 5, 7}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {2, 3} := by sorry

end intersection_complement_equality_l209_20970


namespace triangle_side_range_l209_20947

theorem triangle_side_range (x : ℝ) : 
  (x > 0) →  -- Ensure positive side lengths
  (x + (x + 1) > (x + 2)) →  -- Triangle inequality
  (x + (x + 1) + (x + 2) ≤ 12) →  -- Perimeter condition
  (1 < x ∧ x ≤ 3) :=
by sorry

end triangle_side_range_l209_20947


namespace binomial_20_17_l209_20916

theorem binomial_20_17 : (Nat.choose 20 17) = 1140 := by
  sorry

end binomial_20_17_l209_20916


namespace james_beef_purchase_l209_20926

/-- Proves that James bought 20 pounds of beef given the problem conditions -/
theorem james_beef_purchase :
  ∀ (beef pork : ℝ) (meals : ℕ),
    pork = beef / 2 →
    meals * 1.5 = beef + pork →
    meals * 20 = 400 →
    beef = 20 :=
by
  sorry

end james_beef_purchase_l209_20926


namespace mack_writes_sixteen_pages_l209_20939

/-- Calculates the total number of pages Mack writes from Monday to Thursday -/
def total_pages (T1 R1 T2 R2 P3 T4 T5 R3 R4 : ℕ) : ℕ :=
  let monday_pages := T1 / R1
  let tuesday_pages := T2 / R2
  let wednesday_pages := P3
  let thursday_first_part := T5 / R3
  let thursday_second_part := (T4 - T5) / R4
  let thursday_pages := thursday_first_part + thursday_second_part
  monday_pages + tuesday_pages + wednesday_pages + thursday_pages

/-- Theorem stating that given the specified conditions, Mack writes 16 pages in total -/
theorem mack_writes_sixteen_pages :
  total_pages 60 30 45 15 5 90 30 10 20 = 16 := by
  sorry

end mack_writes_sixteen_pages_l209_20939


namespace play_attendance_l209_20986

theorem play_attendance (total_people : ℕ) (adult_price child_price : ℚ) (total_receipts : ℚ) :
  total_people = 610 →
  adult_price = 2 →
  child_price = 1 →
  total_receipts = 960 →
  ∃ (adults children : ℕ),
    adults + children = total_people ∧
    adult_price * adults + child_price * children = total_receipts ∧
    children = 260 :=
by sorry

end play_attendance_l209_20986


namespace unique_digit_for_divisibility_l209_20965

def is_divisible_by_9 (n : ℕ) : Prop := n % 9 = 0

def four_digit_number (B : ℕ) : ℕ := 5000 + 100 * B + 10 * B + 3

theorem unique_digit_for_divisibility :
  ∃! B : ℕ, B ≤ 9 ∧ is_divisible_by_9 (four_digit_number B) :=
by
  sorry

end unique_digit_for_divisibility_l209_20965


namespace mike_weekly_pullups_l209_20998

/-- The number of pull-ups Mike does each time he enters his office -/
def pullups_per_entry : ℕ := 2

/-- The number of times Mike enters his office per day -/
def office_entries_per_day : ℕ := 5

/-- The number of days in a week -/
def days_in_week : ℕ := 7

/-- The total number of pull-ups Mike does in a week -/
def total_pullups_per_week : ℕ := pullups_per_entry * office_entries_per_day * days_in_week

/-- Theorem stating that Mike does 70 pull-ups in a week -/
theorem mike_weekly_pullups : total_pullups_per_week = 70 := by
  sorry

end mike_weekly_pullups_l209_20998


namespace new_shoes_duration_l209_20990

/-- The duration of new shoes given repair and purchase costs -/
theorem new_shoes_duration (repair_cost : ℝ) (repair_duration : ℝ) (new_cost : ℝ) (cost_increase_percentage : ℝ) :
  repair_cost = 11.50 →
  repair_duration = 1 →
  new_cost = 28.00 →
  cost_increase_percentage = 0.2173913043478261 →
  ∃ (new_duration : ℝ),
    new_duration = 2 ∧
    (new_cost / new_duration) = (repair_cost / repair_duration) * (1 + cost_increase_percentage) :=
by
  sorry

end new_shoes_duration_l209_20990


namespace final_probability_l209_20972

/-- Represents the number of operations performed -/
def num_operations : ℕ := 5

/-- Represents the initial number of red balls -/
def initial_red : ℕ := 2

/-- Represents the initial number of blue balls -/
def initial_blue : ℕ := 1

/-- Represents the final number of red balls -/
def final_red : ℕ := 4

/-- Represents the final number of blue balls -/
def final_blue : ℕ := 4

/-- Calculates the probability of drawing a specific sequence of balls -/
def sequence_probability (red_draws blue_draws : ℕ) : ℚ := sorry

/-- Calculates the number of possible sequences -/
def num_sequences : ℕ := sorry

/-- The main theorem stating the probability of the final outcome -/
theorem final_probability : 
  sequence_probability (final_red - initial_red) (final_blue - initial_blue) * num_sequences = 2/7 := by sorry

end final_probability_l209_20972


namespace function_property_result_l209_20932

theorem function_property_result (g : ℝ → ℝ) 
    (h : ∀ a c : ℝ, c^3 * g a = a^3 * g c) 
    (h_nonzero : g 3 ≠ 0) : 
  (g 6 - g 2) / g 3 = 208/27 := by
sorry

end function_property_result_l209_20932


namespace total_students_correct_l209_20938

/-- The number of students who tried out for the trivia teams -/
def total_students : ℕ := 17

/-- The number of students who didn't get picked -/
def not_picked : ℕ := 5

/-- The number of groups formed -/
def num_groups : ℕ := 3

/-- The number of students in each group -/
def students_per_group : ℕ := 4

/-- Theorem stating that the total number of students who tried out is correct -/
theorem total_students_correct : 
  total_students = not_picked + num_groups * students_per_group := by
  sorry

end total_students_correct_l209_20938


namespace min_value_of_fraction_sum_l209_20934

theorem min_value_of_fraction_sum (m n : ℝ) (hm : m > 0) (hn : n > 0) (h_sum : 2*m + n = 1) :
  (1/m + 2/n) ≥ 8 ∧ ∃ (m₀ n₀ : ℝ), m₀ > 0 ∧ n₀ > 0 ∧ 2*m₀ + n₀ = 1 ∧ 1/m₀ + 2/n₀ = 8 :=
sorry

end min_value_of_fraction_sum_l209_20934


namespace trig_identity_l209_20941

theorem trig_identity (α : Real) (h : 3 * Real.sin α + Real.cos α = 0) : 
  1 / (Real.cos (2 * α) + Real.sin (2 * α)) = 5 := by
  sorry

end trig_identity_l209_20941


namespace poem_distribution_theorem_l209_20974

def distribute_poems (n : ℕ) (k : ℕ) (min_poems : ℕ) : ℕ :=
  let case1 := (n.choose 2) * ((n - 2).choose 2) * 3
  let case2 := (n.choose 2) * ((n - 2).choose 3) * 3
  case1 + case2

theorem poem_distribution_theorem :
  distribute_poems 8 3 2 = 2940 := by
  sorry

end poem_distribution_theorem_l209_20974


namespace inequality_solution_l209_20902

theorem inequality_solution : 
  ∃! a : ℝ, a > 0 ∧ ∀ x > 0, (2 * x - 2 * a + Real.log (x / a)) * (-2 * x^2 + a * x + 5) ≤ 0 := by
  sorry

end inequality_solution_l209_20902


namespace sum_of_product_of_roots_l209_20900

theorem sum_of_product_of_roots (p q r : ℂ) : 
  (4 * p^3 - 8 * p^2 + 16 * p - 12 = 0) ∧ 
  (4 * q^3 - 8 * q^2 + 16 * q - 12 = 0) ∧ 
  (4 * r^3 - 8 * r^2 + 16 * r - 12 = 0) →
  p * q + q * r + r * p = 4 := by
sorry

end sum_of_product_of_roots_l209_20900


namespace saras_basketball_games_l209_20993

theorem saras_basketball_games (won_games lost_games : ℕ) 
  (h1 : won_games = 12) 
  (h2 : lost_games = 4) : 
  won_games + lost_games = 16 := by
  sorry

end saras_basketball_games_l209_20993


namespace red_mailbox_houses_l209_20962

/-- Proves the number of houses with red mailboxes given the total junk mail,
    total houses, houses with white mailboxes, and junk mail per house. -/
theorem red_mailbox_houses
  (total_junk_mail : ℕ)
  (total_houses : ℕ)
  (white_mailbox_houses : ℕ)
  (junk_mail_per_house : ℕ)
  (h1 : total_junk_mail = 48)
  (h2 : total_houses = 8)
  (h3 : white_mailbox_houses = 2)
  (h4 : junk_mail_per_house = 6)
  : total_houses - white_mailbox_houses = 6 := by
  sorry

#check red_mailbox_houses

end red_mailbox_houses_l209_20962


namespace square_sum_geq_neg_double_product_l209_20935

theorem square_sum_geq_neg_double_product (a b : ℝ) : a^2 + b^2 ≥ -2*a*b := by
  sorry

end square_sum_geq_neg_double_product_l209_20935


namespace smallest_of_three_l209_20950

theorem smallest_of_three : 
  ∀ (x y z : ℝ), x = -Real.sqrt 2 ∧ y = 0 ∧ z = -1 → 
  x < y ∧ x < z := by
  sorry

end smallest_of_three_l209_20950


namespace arithmetic_sequence_sum_l209_20952

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (a 2 + a 16 = 6) →
  (a 2 * a 16 = 1) →
  a 7 + a 8 + a 9 + a 10 + a 11 = 15 := by
  sorry

end arithmetic_sequence_sum_l209_20952


namespace hex_351_equals_849_l209_20920

/-- Converts a hexadecimal digit to its decimal value -/
def hex_to_dec (c : Char) : ℕ :=
  match c with
  | '0' => 0 | '1' => 1 | '2' => 2 | '3' => 3
  | '4' => 4 | '5' => 5 | '6' => 6 | '7' => 7
  | '8' => 8 | '9' => 9 | 'A' => 10 | 'B' => 11
  | 'C' => 12 | 'D' => 13 | 'E' => 14 | 'F' => 15
  | _ => 0

/-- Converts a hexadecimal string to its decimal value -/
def hex_string_to_dec (s : String) : ℕ :=
  s.foldr (fun c acc => 16 * acc + hex_to_dec c) 0

/-- Theorem: The hexadecimal number 351 is equal to 849 in decimal -/
theorem hex_351_equals_849 : hex_string_to_dec "351" = 849 := by
  sorry

end hex_351_equals_849_l209_20920


namespace shortest_distance_circle_to_line_l209_20915

/-- The shortest distance from a point on a circle to a line -/
theorem shortest_distance_circle_to_line :
  let center : ℝ × ℝ := (3, -3)
  let radius : ℝ := 3
  let line := {p : ℝ × ℝ | p.1 = p.2}
  ∃ (shortest : ℝ), 
    shortest = 3 * (Real.sqrt 2 - 1) ∧
    ∀ (p : ℝ × ℝ), (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 →
      shortest ≤ Real.sqrt ((p.1 - p.2)^2 + (p.2 - p.2)^2) :=
by sorry

end shortest_distance_circle_to_line_l209_20915


namespace victor_initial_books_l209_20961

/-- The number of books Victor had initially -/
def initial_books : ℕ := sorry

/-- The number of books Victor bought during the book fair -/
def bought_books : ℕ := 3

/-- The total number of books Victor had after buying more -/
def total_books : ℕ := 12

/-- Theorem stating that Victor initially had 9 books -/
theorem victor_initial_books : 
  initial_books + bought_books = total_books → initial_books = 9 := by
  sorry

end victor_initial_books_l209_20961


namespace side_view_area_is_four_l209_20945

/-- Represents a triangular prism -/
structure TriangularPrism where
  lateral_edge_length : ℝ
  base_side_length : ℝ
  main_view_side_length : ℝ

/-- The area of the side view of a triangular prism -/
def side_view_area (prism : TriangularPrism) : ℝ :=
  prism.lateral_edge_length * prism.base_side_length

/-- Theorem: The area of the side view of a specific triangular prism is 4 -/
theorem side_view_area_is_four :
  ∀ (prism : TriangularPrism),
    prism.lateral_edge_length = 2 →
    prism.base_side_length = 2 →
    prism.main_view_side_length = 2 →
    side_view_area prism = 4 := by
  sorry

end side_view_area_is_four_l209_20945


namespace a_value_l209_20905

def U : Set ℤ := {3, 4, 5}

def M (a : ℤ) : Set ℤ := {|a - 3|, 3}

theorem a_value (a : ℤ) (h : (U \ M a) = {5}) : a = -1 ∨ a = 7 := by
  sorry

end a_value_l209_20905


namespace magnitude_a_plus_2b_eq_sqrt_2_l209_20949

def a : Fin 2 → ℝ := ![(-1 : ℝ), 3]
def b : Fin 2 → ℝ := ![1, -2]

theorem magnitude_a_plus_2b_eq_sqrt_2 :
  Real.sqrt ((a 0 + 2 * b 0)^2 + (a 1 + 2 * b 1)^2) = Real.sqrt 2 := by
  sorry

end magnitude_a_plus_2b_eq_sqrt_2_l209_20949


namespace horses_added_correct_horses_added_l209_20943

theorem horses_added (initial_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) 
  (total_days : ℕ) (total_water : ℕ) : ℕ :=
  let water_per_horse := drinking_water + bathing_water
  let initial_daily_water := initial_horses * water_per_horse
  let initial_total_water := initial_daily_water * total_days
  let new_horses_water := total_water - initial_total_water
  let new_horses_daily_water := new_horses_water / total_days
  new_horses_daily_water / water_per_horse

theorem correct_horses_added :
  horses_added 3 5 2 28 1568 = 5 := by
  sorry

end horses_added_correct_horses_added_l209_20943


namespace binomial_expansion_degree_l209_20936

theorem binomial_expansion_degree (n : ℕ) :
  (∀ x, (1 + x)^n = 1 + 6*x + 15*x^2 + 20*x^3 + 15*x^4 + 6*x^5 + x^6) →
  n = 6 := by
  sorry

end binomial_expansion_degree_l209_20936


namespace regular_soda_bottles_l209_20984

theorem regular_soda_bottles (total_bottles : ℕ) (diet_bottles : ℕ) 
  (h1 : total_bottles = 17) 
  (h2 : diet_bottles = 8) : 
  total_bottles - diet_bottles = 9 := by
  sorry

end regular_soda_bottles_l209_20984


namespace spending_vs_earning_difference_l209_20976

def initial_amount : Int := 153
def part_time_earnings : Int := 65
def atm_collection : Int := 195
def supermarket_spending : Int := 87
def electronics_spending : Int := 134
def clothes_spending : Int := 78

theorem spending_vs_earning_difference :
  (supermarket_spending + electronics_spending + clothes_spending) -
  (part_time_earnings + atm_collection) = -39 :=
by sorry

end spending_vs_earning_difference_l209_20976


namespace simplify_square_root_sum_l209_20933

theorem simplify_square_root_sum : 
  (Real.sqrt 450 / Real.sqrt 200) + (Real.sqrt 98 / Real.sqrt 56) = 13/4 := by
  sorry

end simplify_square_root_sum_l209_20933


namespace speed_in_still_water_l209_20954

/-- Theorem: Given a man's upstream and downstream speeds, his speed in still water
    is the average of these two speeds. -/
theorem speed_in_still_water (upstream_speed downstream_speed : ℝ) :
  upstream_speed = 55 →
  downstream_speed = 65 →
  (upstream_speed + downstream_speed) / 2 = 60 := by
sorry

end speed_in_still_water_l209_20954


namespace player_one_points_l209_20977

/-- Represents the sectors on the rotating table -/
def sectors : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1]

/-- The number of players -/
def num_players : ℕ := 16

/-- The number of rotations -/
def num_rotations : ℕ := 13

/-- Calculate the points for a player after given number of rotations -/
def player_points (player : ℕ) (rotations : ℕ) : ℕ := sorry

theorem player_one_points :
  player_points 5 num_rotations = 72 →
  player_points 9 num_rotations = 84 →
  player_points 1 num_rotations = 20 := by sorry

end player_one_points_l209_20977


namespace min_coefficient_value_l209_20980

theorem min_coefficient_value (c d box : ℤ) : 
  (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + box * x + 29) →
  c ≠ d ∧ c ≠ box ∧ d ≠ box →
  ∀ b : ℤ, (∀ x : ℝ, (c * x + d) * (d * x + c) = 29 * x^2 + b * x + 29) → box ≤ b →
  box = 842 :=
by sorry

end min_coefficient_value_l209_20980


namespace sqrt_25_l209_20966

theorem sqrt_25 : {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end sqrt_25_l209_20966


namespace tangent_lines_with_equal_intercepts_l209_20921

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 3 = 0

-- Define a tangent line
def is_tangent_line (a b c : ℝ) : Prop :=
  ∃ (x y : ℝ), circle_C x y ∧ a*x + b*y + c = 0 ∧
  ∀ (x' y' : ℝ), circle_C x' y' → a*x' + b*y' + c ≥ 0

-- Define the condition for equal absolute intercepts
def equal_abs_intercepts (a b c : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ |c/a| = |c/b|

-- Theorem statement
theorem tangent_lines_with_equal_intercepts :
  ∀ (a b c : ℝ),
    is_tangent_line a b c ∧ equal_abs_intercepts a b c →
    ((a = 1 ∧ b = 1 ∧ c = -3) ∨
     (a = 1 ∧ b = 1 ∧ c = 1) ∨
     (a = 1 ∧ b = -1 ∧ c = -5) ∨
     (a = 1 ∧ b = -1 ∧ c = -1) ∨
     (∃ k : ℝ, k^2 = 10 ∧ a = k ∧ b = -1 ∧ c = 0)) :=
by sorry

end tangent_lines_with_equal_intercepts_l209_20921


namespace engine_system_theorems_l209_20960

/-- Engine connecting rod and crank system -/
structure EngineSystem where
  a : ℝ  -- length of crank OA
  b : ℝ  -- length of connecting rod AP
  α : ℝ  -- angle AOP
  β : ℝ  -- angle APO
  x : ℝ  -- length of PQ

/-- Theorems about the engine connecting rod and crank system -/
theorem engine_system_theorems (sys : EngineSystem) :
  -- 1. Sine rule relation
  sys.a * Real.sin sys.α = sys.b * Real.sin sys.β ∧
  -- 2. Maximum value of sin β
  (∃ (max_sin_β : ℝ), max_sin_β = sys.a / sys.b ∧
    ∀ β', Real.sin β' ≤ max_sin_β) ∧
  -- 3. Relation for x
  sys.x = sys.a * (1 - Real.cos sys.α) + sys.b * (1 - Real.cos sys.β) :=
by sorry

end engine_system_theorems_l209_20960


namespace exam_pass_probability_l209_20975

theorem exam_pass_probability (p_A p_B p_C : ℚ) 
  (h_A : p_A = 2/3) 
  (h_B : p_B = 3/4) 
  (h_C : p_C = 2/5) : 
  p_A * p_B * (1 - p_C) + p_A * (1 - p_B) * p_C + (1 - p_A) * p_B * p_C = 7/15 := by
  sorry

end exam_pass_probability_l209_20975


namespace rectangle_side_difference_l209_20918

theorem rectangle_side_difference (A d : ℝ) (h_A : A > 0) (h_d : d > 0) :
  ∃ x y : ℝ, x > y ∧ x * y = A ∧ x^2 + y^2 = d^2 ∧ x - y = Real.sqrt (d^2 - 4 * A) :=
sorry

end rectangle_side_difference_l209_20918


namespace stans_paper_words_per_page_l209_20913

/-- Calculates the number of words per page in Stan's paper. -/
theorem stans_paper_words_per_page 
  (typing_speed : ℕ)        -- Stan's typing speed in words per minute
  (pages : ℕ)               -- Number of pages in the paper
  (water_per_hour : ℕ)      -- Water consumption rate in ounces per hour
  (total_water : ℕ)         -- Total water consumed while writing the paper
  (h1 : typing_speed = 50)  -- Stan types 50 words per minute
  (h2 : pages = 5)          -- The paper is 5 pages long
  (h3 : water_per_hour = 15) -- Stan drinks 15 ounces of water per hour while typing
  (h4 : total_water = 10)   -- Stan drinks 10 ounces of water while writing his paper
  : (typing_speed * (total_water * 60 / water_per_hour)) / pages = 400 := by
  sorry

#check stans_paper_words_per_page

end stans_paper_words_per_page_l209_20913


namespace smallest_coverage_l209_20911

/-- Represents a checkerboard configuration -/
structure CheckerBoard :=
  (rows : Nat)
  (cols : Nat)
  (checkers : Nat)
  (at_most_one_per_square : checkers ≤ rows * cols)

/-- Defines the coverage property for a given k -/
def covers (board : CheckerBoard) (k : Nat) : Prop :=
  ∀ (arrangement : Fin board.checkers → Fin board.rows × Fin board.cols),
    ∃ (rows : Fin k → Fin board.rows) (cols : Fin k → Fin board.cols),
      ∀ (c : Fin board.checkers),
        (arrangement c).1 ∈ Set.range rows ∨ (arrangement c).2 ∈ Set.range cols

/-- The main theorem statement -/
theorem smallest_coverage (board : CheckerBoard) 
  (h_rows : board.rows = 2011)
  (h_cols : board.cols = 2011)
  (h_checkers : board.checkers = 3000) :
  (covers board 1006 ∧ ∀ k < 1006, ¬covers board k) := by
  sorry

end smallest_coverage_l209_20911


namespace quadratic_factorization_l209_20978

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l209_20978


namespace delay_calculation_cottage_to_station_delay_l209_20937

theorem delay_calculation (usual_time : ℝ) (speed_increase : ℝ) (lateness : ℝ) : ℝ :=
  let normal_distance := usual_time
  let increased_speed_time := normal_distance / speed_increase
  let total_time := increased_speed_time - lateness
  usual_time - total_time

theorem cottage_to_station_delay : delay_calculation 18 1.2 2 = 5 := by
  sorry

end delay_calculation_cottage_to_station_delay_l209_20937


namespace hyperbola_eccentricity_sqrt_2_l209_20928

/-- Hyperbola eccentricity theorem -/
theorem hyperbola_eccentricity_sqrt_2 
  (a b c : ℝ) 
  (ha : a > 0) 
  (hb : b > 0) 
  (hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
  (asymptote_eq : ∀ x, b / a * x = x)
  (F2 : ℝ × ℝ)
  (hF2 : F2 = (c, 0))
  (M : ℝ × ℝ)
  (hM : M.1 = 0)
  (N : ℝ × ℝ)
  (perpendicular : (M.2 - N.2) * (b / a) = -(M.1 - N.1))
  (midpoint : N = ((F2.1 + M.1) / 2, (F2.2 + M.2) / 2))
  : c / a = Real.sqrt 2 := by
  sorry

#check hyperbola_eccentricity_sqrt_2

end hyperbola_eccentricity_sqrt_2_l209_20928


namespace m_plus_3_interpretation_l209_20948

/-- Represents the possible interpretations of an assignment statement -/
inductive AssignmentInterpretation
  | AssignToSum
  | AddAndReassign
  | Equality
  | None

/-- Defines the meaning of an assignment statement -/
def assignmentMeaning (left : String) (right : String) : AssignmentInterpretation :=
  if left = right.take (right.length - 2) && right.takeRight 2 = "+3" then
    AssignmentInterpretation.AddAndReassign
  else
    AssignmentInterpretation.None

/-- Theorem stating the correct interpretation of M=M+3 -/
theorem m_plus_3_interpretation :
  assignmentMeaning "M" "M+3" = AssignmentInterpretation.AddAndReassign :=
by sorry

end m_plus_3_interpretation_l209_20948


namespace sin_cos_15_deg_l209_20969

theorem sin_cos_15_deg : 
  Real.sin (15 * π / 180) ^ 4 - Real.cos (15 * π / 180) ^ 4 = -Real.sqrt 3 / 2 := by
  sorry

end sin_cos_15_deg_l209_20969


namespace prime_divisors_theorem_l209_20924

def f (p : ℕ) : ℕ := 3^p + 4^p + 5^p + 9^p - 98

theorem prime_divisors_theorem (p : ℕ) :
  Prime p ↔ (Nat.card (Nat.divisors (f p)) ≤ 6 ↔ p = 2 ∨ p = 3) := by sorry

end prime_divisors_theorem_l209_20924


namespace fraction_evaluation_l209_20992

theorem fraction_evaluation : 
  let x : ℚ := 5
  (x^6 - 16*x^3 + x^2 + 64) / (x^3 - 8) = 4571 / 39 := by
  sorry

end fraction_evaluation_l209_20992


namespace boys_age_l209_20997

theorem boys_age (boy daughter wife father : ℕ) : 
  boy = 5 * daughter →
  wife = 5 * boy →
  father = 2 * wife →
  boy + daughter + wife + father = 81 →
  boy = 5 :=
by sorry

end boys_age_l209_20997


namespace solve_equation_l209_20996

theorem solve_equation (n : ℚ) : 
  (1/(n+2)) + (3/(n+2)) + (2*n/(n+2)) = 4 → n = -2 := by
sorry

end solve_equation_l209_20996


namespace min_reciprocal_sum_l209_20995

theorem min_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 12) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y = 12 → 1/x + 1/y ≥ 1/a + 1/b) → 1/a + 1/b = 1/3 :=
by sorry

end min_reciprocal_sum_l209_20995


namespace factor_calculation_l209_20903

theorem factor_calculation (initial_number : ℕ) (factor : ℚ) : 
  initial_number = 5 → 
  factor * (2 * initial_number + 15) = 75 → 
  factor = 3 := by sorry

end factor_calculation_l209_20903


namespace perimeter_after_cuts_l209_20985

/-- The perimeter of a square after cutting out shapes --/
theorem perimeter_after_cuts (initial_side : ℝ) (green_side : ℝ) : 
  initial_side = 10 → green_side = 2 → 
  (4 * initial_side) + (4 * green_side) = 44 := by
  sorry

#check perimeter_after_cuts

end perimeter_after_cuts_l209_20985


namespace total_matchsticks_l209_20940

def boxes : ℕ := 4
def matchboxes_per_box : ℕ := 20
def sticks_per_matchbox : ℕ := 300

theorem total_matchsticks :
  boxes * matchboxes_per_box * sticks_per_matchbox = 24000 :=
by sorry

end total_matchsticks_l209_20940


namespace longest_side_of_triangle_l209_20914

theorem longest_side_of_triangle (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  Real.tan A = 1/4 ∧
  Real.tan B = 3/5 ∧
  a = min a (min b c) ∧
  a = Real.sqrt 2 ∧
  c = max a (max b c) →
  c = Real.sqrt 17 := by
sorry

end longest_side_of_triangle_l209_20914


namespace quadratic_equation_solution_difference_l209_20931

theorem quadratic_equation_solution_difference : ∃ (x₁ x₂ : ℝ),
  (2 * x₁^2 - 7 * x₁ + 1 = x₁ + 31) ∧
  (2 * x₂^2 - 7 * x₂ + 1 = x₂ + 31) ∧
  x₁ ≠ x₂ ∧
  |x₁ - x₂| = 2 * Real.sqrt 19 :=
by sorry

end quadratic_equation_solution_difference_l209_20931


namespace quadratic_minimum_l209_20988

/-- The quadratic function f(x) = x^2 + 14x + 24 -/
def f (x : ℝ) : ℝ := x^2 + 14*x + 24

theorem quadratic_minimum :
  (∃ (x : ℝ), f x = -25) ∧ (∀ (y : ℝ), f y ≥ -25) ∧ (f (-7) = -25) := by
  sorry

end quadratic_minimum_l209_20988


namespace perpendicular_vectors_a_equals_two_l209_20944

-- Define the vectors m and n
def m : ℝ × ℝ := (1, 2)
def n (a : ℝ) : ℝ × ℝ := (a, -1)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem perpendicular_vectors_a_equals_two :
  ∀ a : ℝ, dot_product m (n a) = 0 → a = 2 := by
  sorry


end perpendicular_vectors_a_equals_two_l209_20944


namespace smaller_number_proof_l209_20989

theorem smaller_number_proof (x y : ℝ) (sum_eq : x + y = 79) (diff_eq : x - y = 15) :
  y = 32 := by
  sorry

end smaller_number_proof_l209_20989


namespace smallest_possible_a_l209_20964

def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem smallest_possible_a :
  ∀ (a b c : ℝ),
  a > 0 →
  parabola a b c (-1/3) = -4/3 →
  (∃ n : ℤ, a + b + c = n) →
  (∀ a' : ℝ, a' > 0 ∧ 
    (∃ b' c' : ℝ, parabola a' b' c' (-1/3) = -4/3 ∧ 
    (∃ n : ℤ, a' + b' + c' = n)) → 
  a' ≥ 3/16) →
  a = 3/16 := by
sorry

end smallest_possible_a_l209_20964


namespace union_of_A_and_complement_of_B_l209_20982

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3, 4}

theorem union_of_A_and_complement_of_B :
  A ∪ (U \ B) = {1, 2, 5} := by sorry

end union_of_A_and_complement_of_B_l209_20982


namespace r_daily_earning_l209_20973

/-- The daily earnings of p, q, and r satisfy the given conditions and r earns 70 per day -/
theorem r_daily_earning (p q r : ℚ) : 
  (9 * (p + q + r) = 1620) → 
  (5 * (p + r) = 600) → 
  (7 * (q + r) = 910) → 
  r = 70 := by
  sorry

end r_daily_earning_l209_20973


namespace total_students_l209_20953

/-- Given a student's position from right and left in a line, calculate the total number of students -/
theorem total_students (rank_from_right rank_from_left : ℕ) 
  (h1 : rank_from_right = 6)
  (h2 : rank_from_left = 5) :
  rank_from_right + rank_from_left - 1 = 10 := by
  sorry

#check total_students

end total_students_l209_20953


namespace exam_score_problem_l209_20959

theorem exam_score_problem (total_questions : ℕ) (correct_score : ℤ) (wrong_score : ℤ) (total_score : ℤ) 
  (h1 : total_questions = 50)
  (h2 : correct_score = 4)
  (h3 : wrong_score = -1)
  (h4 : total_score = 130) :
  ∃ (correct_answers : ℕ),
    correct_answers ≤ total_questions ∧
    correct_score * correct_answers + wrong_score * (total_questions - correct_answers) = total_score ∧
    correct_answers = 36 :=
by sorry

end exam_score_problem_l209_20959


namespace decimal_to_fraction_l209_20979

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end decimal_to_fraction_l209_20979


namespace cos_210_degrees_l209_20968

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end cos_210_degrees_l209_20968


namespace rohan_farm_size_l209_20922

/-- Represents the characteristics of Rohan's coconut farm and its earnings -/
structure CoconutFarm where
  trees_per_sqm : ℕ := 2
  coconuts_per_tree : ℕ := 6
  harvest_period_months : ℕ := 3
  coconut_price : ℚ := 1/2
  total_earnings : ℚ := 240
  total_period_months : ℕ := 6

/-- Calculates the size of the coconut farm based on given parameters -/
def farm_size (farm : CoconutFarm) : ℚ :=
  farm.total_earnings / (farm.trees_per_sqm * farm.coconuts_per_tree * farm.coconut_price * (farm.total_period_months / farm.harvest_period_months))

/-- Theorem stating that Rohan's coconut farm size is 20 square meters -/
theorem rohan_farm_size (farm : CoconutFarm) : farm_size farm = 20 := by
  sorry

end rohan_farm_size_l209_20922


namespace shells_added_correct_l209_20919

/-- Given an initial amount of shells and a final amount of shells,
    calculate the amount of shells added. -/
def shells_added (initial final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given the initial amount of 5 pounds and
    final amount of 28 pounds, the amount of shells added is 23 pounds. -/
theorem shells_added_correct :
  shells_added 5 28 = 23 := by
  sorry

end shells_added_correct_l209_20919


namespace elevator_unreachable_l209_20971

def is_valid_floor (n : ℤ) : Prop := 1 ≤ n ∧ n ≤ 15

def elevator_move (n : ℤ) : ℤ → ℤ
  | 0 => n  -- base case: no moves
  | 1 => n + 7  -- move up 7 floors
  | -1 => n - 9  -- move down 9 floors
  | _ => n  -- invalid move, stay on the same floor

def can_reach (start finish : ℤ) : Prop :=
  ∃ (moves : List ℤ), 
    (∀ m ∈ moves, m = 1 ∨ m = -1) ∧
    (List.foldl elevator_move start moves = finish) ∧
    (∀ i, is_valid_floor (List.foldl elevator_move start (moves.take i)))

theorem elevator_unreachable :
  ¬(can_reach 3 12) :=
sorry

end elevator_unreachable_l209_20971


namespace rational_function_sum_l209_20910

/-- A rational function with specific properties -/
structure RationalFunction where
  p : ℝ → ℝ
  q : ℝ → ℝ
  p_quadratic : ∃ a b c : ℝ, ∀ x, p x = a * x^2 + b * x + c
  q_cubic : ∃ a b c d : ℝ, ∀ x, q x = a * x^3 + b * x^2 + c * x + d
  p_cond : p 4 = 4
  q_cond1 : q 1 = 0
  q_cond2 : q 3 = 3
  q_factor : ∃ r : ℝ → ℝ, ∀ x, q x = (x - 2) * r x

/-- The main theorem -/
theorem rational_function_sum (f : RationalFunction) :
  ∃ p q : ℝ → ℝ, (∀ x, f.p x = p x ∧ f.q x = q x) ∧
  (∀ x, p x + q x = (1/2) * x^3 - (5/4) * x^2 + (17/4) * x) := by
  sorry

end rational_function_sum_l209_20910


namespace equation_solution_l209_20967

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 5 * (x^(1/4))^2 - (3*x)/(x^(3/4)) = 10 + 2 * x^(1/4) ∧ x = 16 := by
  sorry

end equation_solution_l209_20967


namespace arbitrary_across_classes_most_representative_l209_20917

/-- Represents a sampling method for a student survey --/
inductive SamplingMethod
  | GradeSpecific
  | GenderSpecific
  | ActivitySpecific
  | ArbitraryAcrossClasses

/-- Determines if a sampling method is representative of the entire student population --/
def is_representative (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.ArbitraryAcrossClasses => true
  | _ => false

/-- Theorem stating that the arbitrary across classes method is the most representative --/
theorem arbitrary_across_classes_most_representative :
  ∀ (method : SamplingMethod),
    is_representative method →
    method = SamplingMethod.ArbitraryAcrossClasses :=
by
  sorry

#check arbitrary_across_classes_most_representative

end arbitrary_across_classes_most_representative_l209_20917


namespace expression_evaluation_l209_20930

theorem expression_evaluation :
  let x : ℝ := -1
  let y : ℝ := Real.sqrt 2
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 5 := by
sorry

end expression_evaluation_l209_20930


namespace algebraic_expression_equality_l209_20955

theorem algebraic_expression_equality (y : ℝ) : 
  2 * y^2 + 3 * y + 7 = 8 → 4 * y^2 + 6 * y - 9 = -7 := by
  sorry

end algebraic_expression_equality_l209_20955


namespace intersection_of_M_and_N_l209_20991

def M : Set ℤ := {0, 1, 2}
def N : Set ℤ := {x | -1 ≤ x ∧ x ≤ 1}

theorem intersection_of_M_and_N : M ∩ N = {0, 1} := by sorry

end intersection_of_M_and_N_l209_20991


namespace total_students_surveyed_l209_20901

/-- Represents the number of students speaking different combinations of languages --/
structure LanguageCounts where
  french : ℕ
  english : ℕ
  spanish : ℕ
  frenchEnglish : ℕ
  frenchSpanish : ℕ
  englishSpanish : ℕ
  allThree : ℕ
  none : ℕ

/-- The conditions given in the problem --/
def languageConditions (counts : LanguageCounts) : Prop :=
  -- 230 students speak only one language
  counts.french + counts.english + counts.spanish = 230 ∧
  -- 190 students speak exactly two languages
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish = 190 ∧
  -- 40 students speak all three languages
  counts.allThree = 40 ∧
  -- 60 students do not speak any of the three languages
  counts.none = 60 ∧
  -- Among French speakers, 25% speak English, 15% speak Spanish, and 10% speak both English and Spanish
  4 * (counts.frenchEnglish + counts.allThree) = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  20 * (counts.frenchSpanish + counts.allThree) = 3 * (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  10 * counts.allThree = (counts.french + counts.frenchEnglish + counts.frenchSpanish + counts.allThree) ∧
  -- Among English speakers, 20% also speak Spanish
  5 * (counts.englishSpanish + counts.allThree) = (counts.english + counts.frenchEnglish + counts.englishSpanish + counts.allThree)

/-- The theorem to be proved --/
theorem total_students_surveyed (counts : LanguageCounts) :
  languageConditions counts →
  counts.french + counts.english + counts.spanish +
  counts.frenchEnglish + counts.frenchSpanish + counts.englishSpanish +
  counts.allThree + counts.none = 520 :=
by sorry

end total_students_surveyed_l209_20901


namespace platform_length_l209_20912

/-- Given a train crossing a platform, calculate the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) 
  (h1 : train_length = 225) 
  (h2 : train_speed_kmph = 90) 
  (h3 : crossing_time = 25) : 
  ℝ := by
  
  -- Convert train speed from km/h to m/s
  let train_speed_ms := train_speed_kmph * 1000 / 3600

  -- Calculate total distance covered (train + platform)
  let total_distance := train_speed_ms * crossing_time

  -- Calculate platform length
  let platform_length := total_distance - train_length

  -- Prove that the platform length is 400 meters
  have : platform_length = 400 := by sorry

  -- Return the platform length
  exact platform_length


end platform_length_l209_20912


namespace range_of_a_for_solution_a_value_for_minimum_l209_20906

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Part 1
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end range_of_a_for_solution_a_value_for_minimum_l209_20906


namespace dance_event_relationship_l209_20927

/-- Represents a dance event with boys and girls. -/
structure DanceEvent where
  boys : ℕ
  girls : ℕ
  first_boy_dances : ℕ
  increment : ℕ

/-- The relationship between boys and girls in a specific dance event. -/
def dance_relationship (event : DanceEvent) : Prop :=
  event.boys = (event.girls - 4) / 2

/-- Theorem stating the relationship between boys and girls in the dance event. -/
theorem dance_event_relationship :
  ∀ (event : DanceEvent),
  event.first_boy_dances = 6 →
  event.increment = 2 →
  (∀ n : ℕ, n < event.boys → event.first_boy_dances + n * event.increment ≤ event.girls) →
  event.first_boy_dances + (event.boys - 1) * event.increment = event.girls →
  dance_relationship event :=
sorry

end dance_event_relationship_l209_20927


namespace complex_fraction_value_l209_20946

theorem complex_fraction_value (a : ℝ) (z : ℂ) :
  z = (a^2 - 1 : ℂ) + (a + 1 : ℂ) * I →
  z.re = 0 →
  (a + I^2016) / (1 + I) = 1 - I :=
by sorry

end complex_fraction_value_l209_20946


namespace min_value_theorem_l209_20956

theorem min_value_theorem (a b : ℝ) (ha : a > 0) 
  (h : ∀ x > 0, (a * x - 1) * (x^2 + b * x - 4) ≥ 0) : 
  (∀ c, b + 2 / a ≥ c) → c = 4 :=
sorry

end min_value_theorem_l209_20956


namespace programmers_remote_work_cycle_l209_20925

def alex_cycle : ℕ := 5
def brooke_cycle : ℕ := 3
def charlie_cycle : ℕ := 8
def dana_cycle : ℕ := 9

theorem programmers_remote_work_cycle : 
  Nat.lcm alex_cycle (Nat.lcm brooke_cycle (Nat.lcm charlie_cycle dana_cycle)) = 360 := by
  sorry

end programmers_remote_work_cycle_l209_20925


namespace novel_contest_first_prize_l209_20987

/-- The first place prize in a novel contest --/
def first_place_prize (total_prize : ℕ) (num_winners : ℕ) (second_prize : ℕ) (third_prize : ℕ) (other_prize : ℕ) : ℕ :=
  total_prize - (second_prize + third_prize + (num_winners - 3) * other_prize)

/-- Theorem stating the first place prize is $200 given the contest conditions --/
theorem novel_contest_first_prize :
  first_place_prize 800 18 150 120 22 = 200 := by
  sorry

end novel_contest_first_prize_l209_20987


namespace railway_theorem_l209_20923

structure City where
  id : Nat

structure DirectedGraph where
  cities : Set City
  connections : City → City → Prop

def reachable (g : DirectedGraph) (a b : City) : Prop :=
  g.connections a b ∨ ∃ c, g.connections a c ∧ g.connections c b

theorem railway_theorem (g : DirectedGraph) 
  (h₁ : ∀ a b : City, a ∈ g.cities → b ∈ g.cities → a ≠ b → (g.connections a b ∨ g.connections b a)) :
  ∃ n : City, n ∈ g.cities ∧ ∀ m : City, m ∈ g.cities → m ≠ n → reachable g m n :=
sorry

end railway_theorem_l209_20923
