import Mathlib

namespace greater_number_is_64_l1356_135624

theorem greater_number_is_64
  (x y : ℕ)
  (h1 : x * y = 2048)
  (h2 : (x + y) - (x - y) = 64)
  (h3 : x > y) :
  x = 64 :=
by
  -- proof to be filled in
  sorry

end greater_number_is_64_l1356_135624


namespace apples_in_market_l1356_135642

theorem apples_in_market (A O : ℕ) 
    (h1 : A = O + 27) 
    (h2 : A + O = 301) : 
    A = 164 :=
by
  sorry

end apples_in_market_l1356_135642


namespace largest_number_is_A_l1356_135641

theorem largest_number_is_A (x y z w: ℕ):
  x = (8 * 9 + 5) → -- 85 in base 9 to decimal
  y = (2 * 6 * 6) → -- 200 in base 6 to decimal
  z = ((6 * 11) + 8) → -- 68 in base 11 to decimal
  w = 70 → -- 70 in base 10 remains 70
  max (max x y) (max z w) = x := -- 77 is the maximum
by
  sorry

end largest_number_is_A_l1356_135641


namespace bob_cookie_price_same_as_jane_l1356_135606

theorem bob_cookie_price_same_as_jane
  (r_jane : ℝ)
  (s_bob : ℝ)
  (dough_jane : ℝ)
  (num_jane_cookies : ℕ)
  (price_jane_cookie : ℝ)
  (total_earning_jane : ℝ)
  (num_cookies_bob : ℝ)
  (price_bob_cookie : ℝ) :
  r_jane = 4 ∧
  s_bob = 6 ∧
  dough_jane = 18 * (Real.pi * r_jane^2) ∧
  price_jane_cookie = 0.50 ∧
  total_earning_jane = 18 * 50 ∧
  num_cookies_bob = dough_jane / s_bob^2 ∧
  total_earning_jane = num_cookies_bob * price_bob_cookie →
  price_bob_cookie = 36 :=
by
  intros
  sorry

end bob_cookie_price_same_as_jane_l1356_135606


namespace square_must_rotate_at_least_5_turns_l1356_135649

-- Define the square and pentagon as having equal side lengths
def square_sides : Nat := 4
def pentagon_sides : Nat := 5

-- The problem requires us to prove that the square needs to rotate at least 5 full turns
theorem square_must_rotate_at_least_5_turns :
  let lcm := Nat.lcm square_sides pentagon_sides
  lcm / square_sides = 5 :=
by
  -- Proof to be provided
  sorry

end square_must_rotate_at_least_5_turns_l1356_135649


namespace carrots_left_over_l1356_135651

theorem carrots_left_over (c g : ℕ) (h₁ : c = 47) (h₂ : g = 4) : c % g = 3 :=
by
  sorry

end carrots_left_over_l1356_135651


namespace triangle_is_obtuse_l1356_135602

-- Definitions based on given conditions
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  if a ≥ b ∧ a ≥ c then a^2 > b^2 + c^2
  else if b ≥ a ∧ b ≥ c then b^2 > a^2 + c^2
  else c^2 > a^2 + b^2

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove
theorem triangle_is_obtuse : is_triangle 4 6 8 ∧ is_obtuse_triangle 4 6 8 :=
by
  sorry

end triangle_is_obtuse_l1356_135602


namespace anglet_angle_measurement_l1356_135687

-- Definitions based on conditions
def anglet_measurement := 1
def sixth_circle_degrees := 360 / 6
def anglets_in_sixth_circle := 6000

-- Lean theorem statement proving the implied angle measurement
theorem anglet_angle_measurement (one_percent : Real := 0.01) :
  (anglets_in_sixth_circle * one_percent * sixth_circle_degrees) = anglet_measurement * 60 := 
  sorry

end anglet_angle_measurement_l1356_135687


namespace alex_shirts_count_l1356_135620

theorem alex_shirts_count (j a b : ℕ) (h1 : j = a + 3) (h2 : b = j + 8) (h3 : b = 15) : a = 4 :=
by
  sorry

end alex_shirts_count_l1356_135620


namespace inequality_am_gm_l1356_135619

theorem inequality_am_gm (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x / (x^4 + y^2) + y / (x^2 + y^4)) ≤ (1 / (x * y)) :=
by
  sorry

end inequality_am_gm_l1356_135619


namespace exist_unique_xy_solution_l1356_135681

theorem exist_unique_xy_solution :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 ∧ x = 1 / 3 ∧ y = 2 / 3 :=
by
  sorry

end exist_unique_xy_solution_l1356_135681


namespace fraction_not_going_l1356_135622

theorem fraction_not_going (S J : ℕ) (h1 : J = (2:ℕ)/3 * S) 
  (h_not_junior : 3/4 * J = 3/4 * (2/3 * S)) 
  (h_not_senior : 1/3 * S = (1:ℕ)/3 * S) :
  3/4 * (2/3 * S) + 1/3 * S = 5/6 * S :=
by 
  sorry

end fraction_not_going_l1356_135622


namespace deviation_interpretation_l1356_135645

variable (average_score : ℝ)
variable (x : ℝ)

-- Given condition
def higher_than_average : Prop := x = average_score + 5

-- To prove
def lower_than_average : Prop := x = average_score - 9

theorem deviation_interpretation (x : ℝ) (h : x = average_score + 5) : x - 14 = average_score - 9 :=
by
  sorry

end deviation_interpretation_l1356_135645


namespace y_value_l1356_135663

def star (a b c d : ℤ) : ℤ × ℤ := (a - c, b + d)

theorem y_value (x y : ℤ) (h1 : star 5 0 2 (-2) = (3, -2)) (h2 : star x y 0 3 = (3, -2)) :
  y = -5 :=
sorry

end y_value_l1356_135663


namespace parabola_intersection_difference_l1356_135679

noncomputable def parabola1 (x : ℝ) := 3 * x^2 - 6 * x + 6
noncomputable def parabola2 (x : ℝ) := -2 * x^2 + 2 * x + 6

theorem parabola_intersection_difference :
  let a := 0
  let c := 8 / 5
  c - a = 8 / 5 := by
  sorry

end parabola_intersection_difference_l1356_135679


namespace consultation_session_probability_l1356_135666

noncomputable def consultation_probability : ℝ :=
  let volume_cube := 3 * 3 * 3
  let volume_valid := 9 - 2 * (1/3 * 2.25 * 1.5)
  volume_valid / volume_cube

theorem consultation_session_probability : consultation_probability = 1 / 4 :=
by
  sorry

end consultation_session_probability_l1356_135666


namespace find_f1_increasing_on_positive_solve_inequality_l1356_135611

-- Given conditions
axiom f : ℝ → ℝ
axiom domain : ∀ x, 0 < x → true
axiom f4 : f 4 = 1
axiom multiplicative : ∀ x y, 0 < x → 0 < y → f (x * y) = f x + f y
axiom less_than_zero : ∀ x, 0 < x ∧ x < 1 → f x < 0

-- Required proofs
theorem find_f1 : f 1 = 0 := sorry

theorem increasing_on_positive : ∀ x y, 0 < x → 0 < y → x < y → f x < f y := sorry

theorem solve_inequality : {x : ℝ // 3 < x ∧ x ≤ 5} := sorry

end find_f1_increasing_on_positive_solve_inequality_l1356_135611


namespace sequence_eq_third_term_l1356_135630

theorem sequence_eq_third_term 
  (p : ℤ → ℤ)
  (a : ℕ → ℤ)
  (n : ℕ) (h₁ : n > 2)
  (h₂ : a 2 = p (a 1))
  (h₃ : a 3 = p (a 2))
  (h₄ : ∀ k, 4 ≤ k ∧ k ≤ n → a k = p (a (k - 1)))
  (h₅ : a 1 = p (a n))
  : a 1 = a 3 :=
sorry

end sequence_eq_third_term_l1356_135630


namespace range_of_a_range_of_f_diff_l1356_135697

noncomputable def f (a x : ℝ) : ℝ := x^3 + a * x^2 + x + 1
noncomputable def f' (a x : ℝ) : ℝ := 3 * x^2 + 2 * a * x + 1

theorem range_of_a (a : ℝ) : (∃ x1 x2 : ℝ, f' a x1 = 0 ∧ f' a x2 = 0 ∧ x1 ≠ x2) ↔ (a < -Real.sqrt 3 ∨ a > Real.sqrt 3) :=
by
  sorry

theorem range_of_f_diff (a x1 x2 : ℝ) (h1 : f' a x1 = 0) (h2 : f' a x2 = 0) (h12 : x1 ≠ x2) : 
  0 < f a x1 - f a x2 :=
by
  sorry

end range_of_a_range_of_f_diff_l1356_135697


namespace ratio_of_squares_l1356_135669

theorem ratio_of_squares (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a / b = 1 / 3) :
  (4 * a / (4 * b) = 1 / 3) ∧ (a * a / (b * b) = 1 / 9) :=
by
  sorry

end ratio_of_squares_l1356_135669


namespace wall_clock_ring_interval_l1356_135612

theorem wall_clock_ring_interval 
  (n : ℕ)                -- Number of rings in a day
  (total_minutes : ℕ)    -- Total minutes in a day
  (intervals : ℕ) :       -- Number of intervals
  n = 6 ∧ total_minutes = 1440 ∧ intervals = n - 1 ∧ intervals = 5
    → (1440 / intervals = 288 ∧ 288 / 60 = 4∧ 288 % 60 = 48) := sorry

end wall_clock_ring_interval_l1356_135612


namespace toys_produced_each_day_l1356_135694

-- Define the conditions
def total_weekly_production : ℕ := 8000
def days_worked_per_week : ℕ := 4
def daily_production : ℕ := total_weekly_production / days_worked_per_week

-- The statement to be proved
theorem toys_produced_each_day : daily_production = 2000 := sorry

end toys_produced_each_day_l1356_135694


namespace mms_pack_count_l1356_135614

def mms_per_pack (sundaes_monday : Nat) (mms_monday : Nat) (sundaes_tuesday : Nat) (mms_tuesday : Nat) (packs : Nat) : Nat :=
  (sundaes_monday * mms_monday + sundaes_tuesday * mms_tuesday) / packs

theorem mms_pack_count 
  (sundaes_monday : Nat)
  (mms_monday : Nat)
  (sundaes_tuesday : Nat)
  (mms_tuesday : Nat)
  (packs : Nat)
  (monday_total_mms : sundaes_monday * mms_monday = 240)
  (tuesday_total_mms : sundaes_tuesday * mms_tuesday = 200)
  (total_packs : packs = 11)
  : mms_per_pack sundaes_monday mms_monday sundaes_tuesday mms_tuesday packs = 40 := by
  sorry

end mms_pack_count_l1356_135614


namespace find_p_of_probability_l1356_135644

-- Define the conditions and the problem statement
theorem find_p_of_probability
  (A_red_prob : ℚ := 1/3) -- probability of drawing a red ball from bag A
  (A_to_B_ratio : ℚ := 1/2) -- ratio of number of balls in bag A to bag B
  (combined_red_prob : ℚ := 2/5) -- total probability of drawing a red ball after combining balls
  : p = 13 / 30 := by
  sorry

end find_p_of_probability_l1356_135644


namespace last_three_digits_of_7_to_50_l1356_135692

theorem last_three_digits_of_7_to_50 : (7^50) % 1000 = 991 := 
by 
  sorry

end last_three_digits_of_7_to_50_l1356_135692


namespace solution_of_system_l1356_135632

theorem solution_of_system : ∃ x y : ℝ, (2 * x + y = 2) ∧ (x - y = 1) ∧ (x = 1) ∧ (y = 0) := 
by
  sorry

end solution_of_system_l1356_135632


namespace solution_y_amount_l1356_135656

-- Definitions based on the conditions
def alcohol_content_x : ℝ := 0.10
def alcohol_content_y : ℝ := 0.30
def initial_volume_x : ℝ := 50
def final_alcohol_percent : ℝ := 0.25

-- Function to calculate the amount of solution y needed
def required_solution_y (y : ℝ) : Prop :=
  (alcohol_content_x * initial_volume_x + alcohol_content_y * y) / (initial_volume_x + y) = final_alcohol_percent

theorem solution_y_amount : ∃ y : ℝ, required_solution_y y ∧ y = 150 := by
  sorry

end solution_y_amount_l1356_135656


namespace parallelogram_fourth_vertex_distance_l1356_135600

theorem parallelogram_fourth_vertex_distance (d1 d2 d3 d4 : ℝ) (h1 : d1 = 1) (h2 : d2 = 3) (h3 : d3 = 5) :
    d4 = 7 :=
sorry

end parallelogram_fourth_vertex_distance_l1356_135600


namespace divides_y_l1356_135661

theorem divides_y
  (x y : ℤ)
  (h1 : 2 * x + 1 ∣ 8 * y) : 
  2 * x + 1 ∣ y :=
sorry

end divides_y_l1356_135661


namespace proof_of_problem_l1356_135688

variable (f : ℝ → ℝ)
variable (h_nonzero : ∀ x, f x ≠ 0)
variable (h_equation : ∀ x y, f (x * y) = y * f x + x * f y)

theorem proof_of_problem :
  f 1 = 0 ∧ f (-1) = 0 ∧ (∀ x, f (-x) = -f x) :=
by
  sorry

end proof_of_problem_l1356_135688


namespace ice_cream_melt_time_l1356_135662

theorem ice_cream_melt_time :
  let blocks := 16
  let block_length := 1.0/8.0 -- miles per block
  let distance := blocks * block_length -- in miles
  let speed := 12.0 -- miles per hour
  let time := distance / speed -- in hours
  let time_in_minutes := time * 60 -- converted to minutes
  time_in_minutes = 10 := by sorry

end ice_cream_melt_time_l1356_135662


namespace distinct_numbers_div_sum_diff_l1356_135699

theorem distinct_numbers_div_sum_diff (n : ℕ) : 
  ∃ (numbers : Fin n → ℕ), 
    ∀ i j, i ≠ j → (numbers i + numbers j) % (numbers i - numbers j) = 0 := 
by
  sorry

end distinct_numbers_div_sum_diff_l1356_135699


namespace pages_left_to_be_read_l1356_135610

def total_pages : ℕ := 381
def pages_read : ℕ := 149
def pages_per_day : ℕ := 20
def days_in_week : ℕ := 7

theorem pages_left_to_be_read :
  total_pages - pages_read - (pages_per_day * days_in_week) = 92 := by
  sorry

end pages_left_to_be_read_l1356_135610


namespace find_third_triangle_angles_l1356_135617

-- Define the problem context
variables {A B C : ℝ} -- angles of the original triangle

-- Condition: The sum of the angles in a triangle is 180 degrees
axiom sum_of_angles (a b c : ℝ) : a + b + c = 180

-- Given conditions about the triangle and inscribed circles
def original_triangle (a b c : ℝ) : Prop :=
a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 180

def inscribed_circle (a b c : ℝ) : Prop :=
original_triangle a b c

def second_triangle (a b c : ℝ) : Prop :=
inscribed_circle a b c

def third_triangle (a b c : ℝ) : Prop :=
second_triangle a b c

-- Goal: Prove that the angles in the third triangle are 60 degrees each
theorem find_third_triangle_angles (a b c : ℝ) (ha : original_triangle a b c)
  (h_inscribed : inscribed_circle a b c)
  (h_second : second_triangle a b c)
  (h_third : third_triangle a b c) : a = 60 ∧ b = 60 ∧ c = 60 := by
sorry

end find_third_triangle_angles_l1356_135617


namespace frac_abs_div_a_plus_one_l1356_135648

theorem frac_abs_div_a_plus_one (a : ℝ) (h : a ≠ 0) : abs a / a + 1 = 0 ∨ abs a / a + 1 = 2 :=
by sorry

end frac_abs_div_a_plus_one_l1356_135648


namespace total_goals_scored_l1356_135604

theorem total_goals_scored (g1 t1 g2 t2 : ℕ)
  (h1 : g1 = 2)
  (h2 : g1 = t1 - 3)
  (h3 : t2 = 6)
  (h4 : g2 = t2 - 2) :
  g1 + t1 + g2 + t2 = 17 :=
by
  sorry

end total_goals_scored_l1356_135604


namespace number_of_girls_l1356_135652

variable (N n g : ℕ)
variable (h1 : N = 1600)
variable (h2 : n = 200)
variable (h3 : g = 95)

theorem number_of_girls (G : ℕ) (h : g * N = G * n) : G = 760 :=
by sorry

end number_of_girls_l1356_135652


namespace solution_interval_l1356_135640

noncomputable def set_of_solutions : Set ℝ :=
  {x : ℝ | 4 * x - 3 < (x - 2) ^ 2 ∧ (x - 2) ^ 2 < 6 * x - 5}

theorem solution_interval :
  set_of_solutions = {x : ℝ | 7 < x ∧ x < 9} := by
  sorry

end solution_interval_l1356_135640


namespace min_pairs_with_same_sum_l1356_135654

theorem min_pairs_with_same_sum (n : ℕ) (h1 : n > 0) :
  (∀ weights : Fin n → ℕ, (∀ i, weights i ≤ 21) → (∃ i j k l : Fin n,
    i ≠ j ∧ i ≠ k ∧ i ≠ l ∧ j ≠ k ∧ j ≠ l ∧ k ≠ l ∧
    weights i + weights j = weights k + weights l)) ↔ n ≥ 8 :=
by
  sorry

end min_pairs_with_same_sum_l1356_135654


namespace john_collects_crabs_l1356_135667

-- Definitions for the conditions
def baskets_per_week : ℕ := 3
def crabs_per_basket : ℕ := 4
def price_per_crab : ℕ := 3
def total_income : ℕ := 72

-- Definition for the question
def times_per_week_to_collect (baskets_per_week crabs_per_basket price_per_crab total_income : ℕ) : ℕ :=
  (total_income / price_per_crab) / (baskets_per_week * crabs_per_basket)

-- The theorem statement
theorem john_collects_crabs (h1 : baskets_per_week = 3) (h2 : crabs_per_basket = 4) (h3 : price_per_crab = 3) (h4 : total_income = 72) :
  times_per_week_to_collect baskets_per_week crabs_per_basket price_per_crab total_income = 2 :=
by
  sorry

end john_collects_crabs_l1356_135667


namespace base9_sum_correct_l1356_135637

def base9_addition (a b c : ℕ) : ℕ :=
  a + b + c

theorem base9_sum_correct :
  base9_addition (263) (452) (247) = 1073 :=
by sorry

end base9_sum_correct_l1356_135637


namespace part1_equation_part2_equation_l1356_135696

-- Part (Ⅰ)
theorem part1_equation :
  (- ((-1) ^ 1000) - 2.45 * 8 + 2.55 * (-8) = -41) :=
by
  sorry

-- Part (Ⅱ)
theorem part2_equation :
  ((1 / 6 - 1 / 3 + 0.25) / (- (1 / 12)) = -1) :=
by
  sorry

end part1_equation_part2_equation_l1356_135696


namespace sum_of_common_ratios_is_five_l1356_135695

theorem sum_of_common_ratios_is_five {k p r : ℝ} 
  (h1 : p ≠ r)                       -- different common ratios
  (h2 : k ≠ 0)                       -- non-zero k
  (a2 : ℝ := k * p)                  -- term a2
  (a3 : ℝ := k * p^2)                -- term a3
  (b2 : ℝ := k * r)                  -- term b2
  (b3 : ℝ := k * r^2)                -- term b3
  (h3 : a3 - b3 = 5 * (a2 - b2))     -- given condition
  : p + r = 5 := 
by
  sorry

end sum_of_common_ratios_is_five_l1356_135695


namespace degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l1356_135672

-- Definition of the "isValidGraph" function based on degree sequences
-- Placeholder for the actual definition
def isValidGraph (degrees : List ℕ) : Prop :=
  sorry

-- Degree sequences given in the problem
def d_a := [8, 6, 5, 4, 4, 3, 2, 2]
def d_b := [7, 7, 6, 5, 4, 2, 2, 1]
def d_c := [6, 6, 6, 5, 5, 3, 2, 2]

-- Statement that proves none of these sequences can form a valid graph
theorem degree_sequence_a_invalid : ¬ isValidGraph d_a :=
  sorry

theorem degree_sequence_b_invalid : ¬ isValidGraph d_b :=
  sorry

theorem degree_sequence_c_invalid : ¬ isValidGraph d_c :=
  sorry

-- Final statement combining all individual proofs
theorem all_sequences_invalid :
  ¬ isValidGraph d_a ∧ ¬ isValidGraph d_b ∧ ¬ isValidGraph d_c :=
  ⟨degree_sequence_a_invalid, degree_sequence_b_invalid, degree_sequence_c_invalid⟩

end degree_sequence_a_invalid_degree_sequence_b_invalid_degree_sequence_c_invalid_all_sequences_invalid_l1356_135672


namespace true_propositions_l1356_135639

theorem true_propositions :
  (∀ x y, (x * y = 1 → x * y = (x * y))) ∧
  (¬ (∀ (a b : ℝ), (∀ (A B : ℝ), a = b → A = B) ∧ (A = B → a ≠ b))) ∧
  (∀ m : ℝ, (m ≤ 1 → ∃ x : ℝ, x^2 - 2 * x + m = 0)) ↔
    (true ∧ true ∧ true) :=
by sorry

end true_propositions_l1356_135639


namespace hyperbola_focal_length_l1356_135615

theorem hyperbola_focal_length : 
  (∃ (f : ℝ) (x y : ℝ), (3 * x^2 - y^2 = 3) ∧ (f = 4)) :=
by {
  sorry
}

end hyperbola_focal_length_l1356_135615


namespace jaden_toy_cars_left_l1356_135626

-- Definitions for each condition
def initial_toys : ℕ := 14
def purchased_toys : ℕ := 28
def birthday_toys : ℕ := 12
def given_to_sister : ℕ := 8
def given_to_vinnie : ℕ := 3
def traded_lost : ℕ := 5
def traded_received : ℕ := 7

-- The final number of toy cars Jaden has
def final_toys : ℕ :=
  initial_toys + purchased_toys + birthday_toys - given_to_sister - given_to_vinnie + (traded_received - traded_lost)

theorem jaden_toy_cars_left : final_toys = 45 :=
by
  -- The proof will be filled in here 
  sorry

end jaden_toy_cars_left_l1356_135626


namespace max_value_expr_l1356_135643

theorem max_value_expr (a b c d : ℝ) (ha : -4 ≤ a ∧ a ≤ 4) (hb : -4 ≤ b ∧ b ≤ 4) (hc : -4 ≤ c ∧ c ≤ 4) (hd : -4 ≤ d ∧ d ≤ 4) :
  (a + 2*b + c + 2*d - a*b - b*c - c*d - d*a) ≤ 72 :=
sorry

end max_value_expr_l1356_135643


namespace initial_winnings_l1356_135686

theorem initial_winnings (X : ℝ) 
  (h1 : X - 0.25 * X = 0.75 * X)
  (h2 : 0.75 * X - 0.10 * (0.75 * X) = 0.675 * X)
  (h3 : 0.675 * X - 0.15 * (0.675 * X) = 0.57375 * X)
  (h4 : 0.57375 * X = 240) :
  X = 418 := by
  sorry

end initial_winnings_l1356_135686


namespace expression_value_l1356_135625

theorem expression_value (a b c : ℚ) (h₁ : b = 8) (h₂ : c = 5) (h₃ : a * b * c = 2 * (a + b + c) + 14) : 
  (c - a) ^ 2 + b = 8513 / 361 := by 
  sorry

end expression_value_l1356_135625


namespace people_remaining_at_end_l1356_135647

def total_people_start : ℕ := 600
def girls_start : ℕ := 240
def boys_start : ℕ := total_people_start - girls_start
def boys_left_early : ℕ := boys_start / 4
def girls_left_early : ℕ := girls_start / 8
def total_left_early : ℕ := boys_left_early + girls_left_early
def people_remaining : ℕ := total_people_start - total_left_early

theorem people_remaining_at_end : people_remaining = 480 := by
  sorry

end people_remaining_at_end_l1356_135647


namespace original_price_of_shoes_l1356_135633

theorem original_price_of_shoes (P : ℝ) (h1 : 0.80 * P = 480) : P = 600 := 
by
  sorry

end original_price_of_shoes_l1356_135633


namespace range_of_a_l1356_135616

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
sorry

end range_of_a_l1356_135616


namespace jinsu_work_per_hour_l1356_135659

theorem jinsu_work_per_hour (t : ℝ) (h : t = 4) : (1 / t = 1 / 4) :=
by {
    sorry
}

end jinsu_work_per_hour_l1356_135659


namespace sum_of_digits_of_A15B94_multiple_of_99_l1356_135653

theorem sum_of_digits_of_A15B94_multiple_of_99 (A B : ℕ) 
  (hA : A < 10) (hB : B < 10)
  (h_mult_99 : ∃ n : ℕ, (100000 * A + 10000 + 5000 + 100 * B + 90 + 4) = 99 * n) :
  A + B = 8 := 
by
  sorry

end sum_of_digits_of_A15B94_multiple_of_99_l1356_135653


namespace circumscribed_sphere_radius_l1356_135693

noncomputable def radius_of_circumscribed_sphere (a : ℝ) (α : ℝ) : ℝ :=
  a / (3 * Real.sin α)

theorem circumscribed_sphere_radius (a α : ℝ) :
  radius_of_circumscribed_sphere a α = a / (3 * Real.sin α) :=
by
  sorry

end circumscribed_sphere_radius_l1356_135693


namespace sum_of_consecutive_even_integers_l1356_135607

theorem sum_of_consecutive_even_integers (n : ℕ) (h1 : (n - 2) + (n + 2) = 162) (h2 : ∃ k : ℕ, n = k^2) :
  (n - 2) + n + (n + 2) = 243 :=
by
  -- no proof required
  sorry

end sum_of_consecutive_even_integers_l1356_135607


namespace platform_length_proof_l1356_135638

noncomputable def train_length : ℝ := 1200
noncomputable def time_to_cross_tree : ℝ := 120
noncomputable def time_to_cross_platform : ℝ := 240
noncomputable def speed_of_train : ℝ := train_length / time_to_cross_tree
noncomputable def platform_length : ℝ := 2400 - train_length

theorem platform_length_proof (h1 : train_length = 1200) (h2 : time_to_cross_tree = 120) (h3 : time_to_cross_platform = 240) :
  platform_length = 1200 := by
  sorry

end platform_length_proof_l1356_135638


namespace mechanic_hourly_rate_l1356_135605

-- Definitions and conditions
def total_bill : ℕ := 450
def parts_charge : ℕ := 225
def hours_worked : ℕ := 5

-- The main theorem to prove
theorem mechanic_hourly_rate : (total_bill - parts_charge) / hours_worked = 45 := by
  sorry

end mechanic_hourly_rate_l1356_135605


namespace range_of_m_l1356_135660

variable (m : ℝ)

def prop_p : Prop := ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + m*x1 + 1 = 0) ∧ (x2^2 + m*x2 + 1 = 0)

def prop_q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 > 0

theorem range_of_m (h₁ : prop_p m) (h₂ : ¬prop_q m) : m < -2 ∨ m ≥ 3 :=
sorry

end range_of_m_l1356_135660


namespace equation_of_tangent_line_l1356_135684

-- Definitions for the given conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 2 * x
def P : ℝ × ℝ := (-1, 4)
def slope_of_tangent (a : ℝ) (x : ℝ) : ℝ := -6 * x^2 - 2

-- The main theorem to prove the equation of the tangent line
theorem equation_of_tangent_line (a : ℝ) (ha : f a (-1) = 4) :
  8 * x + y + 4 = 0 := by
  sorry

end equation_of_tangent_line_l1356_135684


namespace solution_set_l1356_135608

-- Define determinant operation on 2x2 matrices
def determinant (a b c d : ℝ) := a * d - b * c

-- Define the condition inequality
def condition (x : ℝ) : Prop :=
  determinant x 3 (-x) x < determinant 2 0 1 2

-- Prove that the solution to the condition is -4 < x < 1
theorem solution_set : {x : ℝ | condition x} = {x : ℝ | -4 < x ∧ x < 1} :=
by
  sorry

end solution_set_l1356_135608


namespace four_digit_square_l1356_135678

/-- A four-digit square number that satisfies the given conditions -/
theorem four_digit_square (a b c d : ℕ) (h₁ : b + c = a) (h₂ : a + c = 10 * d) :
  1000 * a + 100 * b + 10 * c + d = 6241 :=
sorry

end four_digit_square_l1356_135678


namespace correct_statement_D_l1356_135634

theorem correct_statement_D : (- 3 / 5 : ℚ) < (- 4 / 7 : ℚ) :=
  by
  -- The proof step is omitted as per the instruction
  sorry

end correct_statement_D_l1356_135634


namespace train_length_l1356_135623

theorem train_length (speed_kph : ℝ) (time_sec : ℝ) (speed_mps : ℝ) (length_m : ℝ) 
  (h1 : speed_kph = 60) 
  (h2 : time_sec = 42) 
  (h3 : speed_mps = speed_kph * 1000 / 3600) 
  (h4 : length_m = speed_mps * time_sec) :
  length_m = 700.14 :=
by
  sorry

end train_length_l1356_135623


namespace no_polygon_with_1974_diagonals_l1356_135601

theorem no_polygon_with_1974_diagonals :
  ¬ ∃ N : ℕ, N * (N - 3) / 2 = 1974 :=
sorry

end no_polygon_with_1974_diagonals_l1356_135601


namespace download_time_l1356_135685

def file_size : ℕ := 90
def rate_first_part : ℕ := 5
def rate_second_part : ℕ := 10
def size_first_part : ℕ := 60

def time_first_part : ℕ := size_first_part / rate_first_part
def size_second_part : ℕ := file_size - size_first_part
def time_second_part : ℕ := size_second_part / rate_second_part
def total_time : ℕ := time_first_part + time_second_part

theorem download_time :
  total_time = 15 := by
  -- sorry can be replaced with the actual proof if needed
  sorry

end download_time_l1356_135685


namespace regular_polygon_sides_l1356_135682

theorem regular_polygon_sides (n : ℕ) (h : (n - 2) * 180 / n = 135) : n = 8 := 
by
  sorry

end regular_polygon_sides_l1356_135682


namespace equal_share_expense_l1356_135635

theorem equal_share_expense (L B C X : ℝ) : 
  let T := L + B + C - X
  let share := T / 3 
  L + (share - L) == (B + C - X - 2 * L) / 3 := 
by
  sorry

end equal_share_expense_l1356_135635


namespace center_of_circle_l1356_135670

theorem center_of_circle (x y : ℝ) : 
    (∃ x y : ℝ, x^2 + y^2 = 4*x - 6*y + 9) → (x, y) = (2, -3) := 
by sorry

end center_of_circle_l1356_135670


namespace real_values_of_x_l1356_135628

theorem real_values_of_x (x : ℝ) (h : x ≠ 4) :
  (x * (x + 1) / (x - 4)^2 ≥ 15) ↔ (x ≤ 3 ∨ (40/7 < x ∧ x < 4) ∨ x > 4) :=
by sorry

end real_values_of_x_l1356_135628


namespace total_votes_l1356_135664

theorem total_votes (emma_votes : ℕ) (vote_fraction : ℚ) (h_emma : emma_votes = 45) (h_fraction : vote_fraction = 3/7) :
  emma_votes = vote_fraction * 105 :=
by {
  sorry
}

end total_votes_l1356_135664


namespace infinite_triangle_area_sum_l1356_135629

noncomputable def rectangle_area_sum : ℝ :=
  let AB := 2
  let BC := 1
  let Q₁ := 0.5
  let base_area := (1/2) * Q₁ * (1/4)
  base_area * (1/(1 - 1/4))

theorem infinite_triangle_area_sum :
  rectangle_area_sum = 1/12 :=
by
  sorry

end infinite_triangle_area_sum_l1356_135629


namespace num_positive_four_digit_integers_of_form_xx75_l1356_135689

theorem num_positive_four_digit_integers_of_form_xx75 : 
  ∃ n : ℕ, n = 90 ∧ ∀ a b : ℕ, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 → (∃ x: ℕ, x = 1000 * a + 100 * b + 75 ∧ 1000 ≤ x ∧ x < 10000) → n = 90 :=
sorry

end num_positive_four_digit_integers_of_form_xx75_l1356_135689


namespace find_extrema_on_interval_l1356_135636

noncomputable def y (x : ℝ) := (10 * x + 10) / (x^2 + 2 * x + 2)

theorem find_extrema_on_interval :
  ∃ (min_val max_val : ℝ) (min_x max_x : ℝ), 
    min_val = 0 ∧ min_x = -1 ∧ max_val = 5 ∧ max_x = 0 ∧ 
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≥ min_val) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 2, y x ≤ max_val) :=
by
  sorry

end find_extrema_on_interval_l1356_135636


namespace circle_arc_and_circumference_l1356_135613

theorem circle_arc_and_circumference (C_X : ℝ) (θ_YOZ : ℝ) (C_D : ℝ) (r_X r_D : ℝ) :
  C_X = 100 ∧ θ_YOZ = 150 ∧ r_X = 50 / π ∧ r_D = 25 / π ∧ C_D = 50 →
  (θ_YOZ / 360) * C_X = 500 / 12 ∧ 2 * π * r_D = C_D :=
by sorry

end circle_arc_and_circumference_l1356_135613


namespace union_of_A_and_B_l1356_135691

-- Condition definitions
def A : Set ℝ := {x : ℝ | abs (x - 3) < 2}
def B : Set ℝ := {x : ℝ | (x + 1) / (x - 2) ≤ 0}

-- The theorem we need to prove
theorem union_of_A_and_B : A ∪ B = {x : ℝ | -1 ≤ x ∧ x < 5} :=
by
  -- This is where the proof would go if it were required
  sorry

end union_of_A_and_B_l1356_135691


namespace gymnastics_team_l1356_135621

def number_of_rows (n m k : ℕ) : Prop :=
  n = k * (2 * m + k - 1) / 2

def members_in_first_row (n m k : ℕ) : Prop :=
  number_of_rows n m k ∧ 16 < k

theorem gymnastics_team : ∃ m k : ℕ, members_in_first_row 1000 m k ∧ k = 25 ∧ m = 28 :=
by
  sorry

end gymnastics_team_l1356_135621


namespace convert_to_rectangular_form_l1356_135674

noncomputable def θ : ℝ := 15 * Real.pi / 2

noncomputable def EulerFormula (θ : ℝ) : ℂ := Complex.exp (Complex.I * θ)

theorem convert_to_rectangular_form : EulerFormula θ = Complex.I := by
  sorry

end convert_to_rectangular_form_l1356_135674


namespace watch_loss_percentage_l1356_135676

noncomputable def initial_loss_percentage : ℝ :=
  let CP := 350
  let SP_new := 364
  let delta_SP := 140
  show ℝ from 
  sorry

theorem watch_loss_percentage (CP SP_new delta_SP : ℝ) (h₁ : CP = 350)
  (h₂ : SP_new = 364) (h₃ : delta_SP = 140) : 
  initial_loss_percentage = 36 :=
by
  -- Use the hypothesis and solve the corresponding problem
  sorry

end watch_loss_percentage_l1356_135676


namespace dozen_chocolate_bars_cost_l1356_135683

theorem dozen_chocolate_bars_cost
  (cost_mag : ℕ → ℝ) (cost_choco_bar : ℕ → ℝ)
  (H1 : cost_mag 1 = 1)
  (H2 : 4 * (cost_choco_bar 1) = 8 * (cost_mag 1)) :
  12 * (cost_choco_bar 1) = 24 := 
sorry

end dozen_chocolate_bars_cost_l1356_135683


namespace degree_of_minus_5x4y_l1356_135658

def degree_of_monomial (coeff : Int) (x_exp y_exp : Nat) : Nat :=
  x_exp + y_exp

theorem degree_of_minus_5x4y : degree_of_monomial (-5) 4 1 = 5 :=
by
  sorry

end degree_of_minus_5x4y_l1356_135658


namespace calculate_expression_l1356_135657

theorem calculate_expression (p q : ℝ) (hp : p + q = 7) (hq : p * q = 12) :
  p^3 + p^4 * q^2 + p^2 * q^4 + q^3 = 3691 := 
by sorry

end calculate_expression_l1356_135657


namespace exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l1356_135631

theorem exists_five_integers_sum_fifth_powers (A B C D E : ℤ) : 
  ∃ (A B C D E : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 + E^5 :=
  by
    sorry

theorem no_four_integers_sum_fifth_powers (A B C D : ℤ) : 
  ¬ ∃ (A B C D : ℤ), 2018 = A^5 + B^5 + C^5 + D^5 :=
  by
    sorry

end exists_five_integers_sum_fifth_powers_no_four_integers_sum_fifth_powers_l1356_135631


namespace original_amount_of_money_l1356_135698

-- Define the conditions
variables (x : ℕ) -- daily allowance

-- Spending details
def spend_10_days := 6 * 10 - 6 * x
def spend_15_days := 15 * 3 - 3 * x

-- Lean proof statement
theorem original_amount_of_money (h : spend_10_days = spend_15_days) : (6 * 10 - 6 * x) = 30 :=
by
  sorry

end original_amount_of_money_l1356_135698


namespace ratio_equivalence_l1356_135671

theorem ratio_equivalence (x y z : ℝ) (h1 : x ≠ y) (h2 : y ≠ z) (h3 : x ≠ z)
  (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h : y / (x - z) = (x + 2 * y) / z ∧ (x + 2 * y) / z = x / (y + z)) :
  x / (y + z) = (2 * y - z) / (y + z) :=
by
  sorry

end ratio_equivalence_l1356_135671


namespace triangle_AB_length_correct_l1356_135609

theorem triangle_AB_length_correct (BC AC : Real) (A : Real) 
  (hBC : BC = Real.sqrt 7) 
  (hAC : AC = 2 * Real.sqrt 3) 
  (hA : A = Real.pi / 6) :
  ∃ (AB : Real), (AB = 5 ∨ AB = 1) :=
by
  sorry

end triangle_AB_length_correct_l1356_135609


namespace quadratic_has_solution_zero_l1356_135668

theorem quadratic_has_solution_zero (k : ℝ) : 
  (∃ x : ℝ, (k - 2) * x^2 + 3 * x + k^2 - 4 = 0) →
  ((k - 2) ≠ 0) → k = -2 := 
by 
  sorry

end quadratic_has_solution_zero_l1356_135668


namespace polynomial_factorization_l1356_135646

theorem polynomial_factorization :
  ∀ (a b c : ℝ),
    a * (b - c) ^ 4 + b * (c - a) ^ 4 + c * (a - b) ^ 4 =
    (a - b) * (b - c) * (c - a) * (a + b + c) :=
  by
    intro a b c
    sorry

end polynomial_factorization_l1356_135646


namespace average_expenditure_week_l1356_135665

theorem average_expenditure_week (avg_3_days: ℝ) (avg_4_days: ℝ) (total_days: ℝ):
  avg_3_days = 350 → avg_4_days = 420 → total_days = 7 → 
  ((3 * avg_3_days + 4 * avg_4_days) / total_days = 390) :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_expenditure_week_l1356_135665


namespace largest_prime_factor_of_891_l1356_135690

theorem largest_prime_factor_of_891 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 891 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 891 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_891_l1356_135690


namespace find_z_l1356_135650

theorem find_z 
  {x y z : ℕ}
  (hx : x = 4)
  (hy : y = 7)
  (h_least : x - y - z = 17) : 
  z = 14 :=
by
  sorry

end find_z_l1356_135650


namespace total_earnings_l1356_135627

theorem total_earnings (L A J M : ℝ) 
  (hL : L = 2000) 
  (hA : A = 0.70 * L) 
  (hJ : J = 1.50 * A) 
  (hM : M = 0.40 * J) 
  : L + A + J + M = 6340 := 
  by 
    sorry

end total_earnings_l1356_135627


namespace min_abs_sum_l1356_135603

theorem min_abs_sum (x y z : ℝ) (hx : 0 ≤ x) (hxy : x ≤ y) (hyz : y ≤ z) (hz : z ≤ 4) 
  (hy_eq : y^2 = x^2 + 2) (hz_eq : z^2 = y^2 + 2) : 
  |x - y| + |y - z| = 4 - 2 * Real.sqrt 3 :=
sorry

end min_abs_sum_l1356_135603


namespace find_volume_of_12_percent_solution_l1356_135680

variable (x y : ℝ)

theorem find_volume_of_12_percent_solution
  (h1 : x + y = 60)
  (h2 : 0.02 * x + 0.12 * y = 3) :
  y = 18 := 
sorry

end find_volume_of_12_percent_solution_l1356_135680


namespace inequality_proof_equality_condition_l1356_135618

variable {x y z : ℝ}

def positive_reals (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0

theorem inequality_proof (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
sorry -- Proof goes here

theorem equality_condition (hxyz : positive_reals x y z) (h : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y = 7 ↔ x = 2 * z ∧ y = z :=
sorry -- Proof goes here

end inequality_proof_equality_condition_l1356_135618


namespace infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l1356_135673

theorem infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017 :
  ∀ n : ℕ, ∃ m : ℕ, (m ∈ {x | ∀ d ∈ Nat.digits 10 x, d = 0 ∨ d = 1}) ∧ 2017 ∣ m :=
by
  sorry

end infinitely_many_numbers_composed_of_0_and_1_divisible_by_2017_l1356_135673


namespace divide_and_add_l1356_135675

variable (number : ℝ)

theorem divide_and_add (h : 4 * number = 166.08) : number / 4 + 0.48 = 10.86 := by
  -- assume the proof follows accurately
  sorry

end divide_and_add_l1356_135675


namespace sqrt_164_between_12_and_13_l1356_135677

theorem sqrt_164_between_12_and_13 : 12 < Real.sqrt 164 ∧ Real.sqrt 164 < 13 :=
sorry

end sqrt_164_between_12_and_13_l1356_135677


namespace correct_sequence_is_A_l1356_135655

def Step := String
def Sequence := List Step

def correct_sequence : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]

def option_A : Sequence :=
  ["Buy a ticket", "Wait for the train", "Check the ticket", "Board the train"]
def option_B : Sequence :=
  ["Wait for the train", "Buy a ticket", "Board the train", "Check the ticket"]
def option_C : Sequence :=
  ["Buy a ticket", "Wait for the train", "Board the train", "Check the ticket"]
def option_D : Sequence :=
  ["Repair the train", "Buy a ticket", "Check the ticket", "Board the train"]

theorem correct_sequence_is_A :
  correct_sequence = option_A :=
sorry

end correct_sequence_is_A_l1356_135655
