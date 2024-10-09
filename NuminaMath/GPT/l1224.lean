import Mathlib

namespace probability_z_l1224_122400

variable (p q x y z : ℝ)

-- Conditions
def condition1 : Prop := z = p * y + q * x
def condition2 : Prop := x = p + q * x^2
def condition3 : Prop := y = q + p * y^2
def condition4 : Prop := x ≠ y

-- Theorem Statement
theorem probability_z : condition1 p q x y z ∧ condition2 p q x ∧ condition3 p q y ∧ condition4 x y → z = 2 * q := by
  sorry

end probability_z_l1224_122400


namespace trapezoid_area_l1224_122430

theorem trapezoid_area (u l h : ℕ) (hu : u = 12) (hl : l = u + 4) (hh : h = 10) : 
  (1 / 2 : ℚ) * (u + l) * h = 140 := by
  sorry

end trapezoid_area_l1224_122430


namespace ratio_of_boys_to_total_students_l1224_122497

theorem ratio_of_boys_to_total_students
  (p : ℝ)
  (h : p = (3/4) * (1 - p)) :
  p = 3 / 7 :=
by
  sorry

end ratio_of_boys_to_total_students_l1224_122497


namespace min_value_at_zero_max_value_a_l1224_122471

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log (x + 1) - (a * x / (x + 1))

-- Part (I)
theorem min_value_at_zero {a : ℝ} (h : ∀ x, f x a ≥ f 0 a) : a = 1 :=
sorry

-- Part (II)
theorem max_value_a (h : ∀ x > 0, f x a > 0) : a ≤ 1 :=
sorry

end min_value_at_zero_max_value_a_l1224_122471


namespace max_intersections_l1224_122455

theorem max_intersections (X Y : Type) [Fintype X] [Fintype Y]
  (hX : Fintype.card X = 20) (hY : Fintype.card Y = 10) : 
  ∃ (m : ℕ), m = 8550 := by
  sorry

end max_intersections_l1224_122455


namespace distance_between_points_l1224_122479

theorem distance_between_points :
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  (Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 5 * Real.sqrt 2) :=
by
  let x1 := 3
  let y1 := 3
  let x2 := -2
  let y2 := -2
  sorry

end distance_between_points_l1224_122479


namespace max_value_of_expression_l1224_122469

noncomputable def max_expression_value (a b : ℝ) := a * b * (100 - 5 * a - 2 * b)

theorem max_value_of_expression :
  ∀ (a b : ℝ), 0 < a → 0 < b → 5 * a + 2 * b < 100 →
  max_expression_value a b ≤ 78125 / 36 := by
  intros a b ha hb h
  sorry

end max_value_of_expression_l1224_122469


namespace opposite_of_neg_sqrt_two_l1224_122483

theorem opposite_of_neg_sqrt_two : -(-Real.sqrt 2) = Real.sqrt 2 := 
by {
  sorry
}

end opposite_of_neg_sqrt_two_l1224_122483


namespace mean_of_remaining_two_l1224_122418

def seven_numbers := [1865, 1990, 2015, 2023, 2105, 2120, 2135]

def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem mean_of_remaining_two
  (h : mean (seven_numbers.take 5) = 2043) :
  mean (seven_numbers.drop 5) = 969 :=
by
  sorry

end mean_of_remaining_two_l1224_122418


namespace find_k_l1224_122465

noncomputable section

variables {a b k : ℝ}

theorem find_k 
  (h1 : 4^a = k) 
  (h2 : 9^b = k)
  (h3 : 1 / a + 1 / b = 2) : 
  k = 6 :=
sorry

end find_k_l1224_122465


namespace not_jog_probability_eq_l1224_122413

def P_jog : ℚ := 5 / 8

theorem not_jog_probability_eq :
  1 - P_jog = 3 / 8 :=
by
  sorry

end not_jog_probability_eq_l1224_122413


namespace baker_sold_cakes_l1224_122443

def initialCakes : Nat := 110
def additionalCakes : Nat := 76
def remainingCakes : Nat := 111
def cakesSold : Nat := 75

theorem baker_sold_cakes :
  initialCakes + additionalCakes - remainingCakes = cakesSold := by
  sorry

end baker_sold_cakes_l1224_122443


namespace maria_remaining_towels_l1224_122487

def total_towels_initial := 40 + 44
def towels_given_away := 65

theorem maria_remaining_towels : (total_towels_initial - towels_given_away) = 19 := by
  sorry

end maria_remaining_towels_l1224_122487


namespace solve_for_difference_l1224_122485

variable (a b : ℝ)

theorem solve_for_difference (h1 : a^3 - b^3 = 4) (h2 : a^2 + ab + b^2 + a - b = 4) : a - b = 2 :=
sorry

end solve_for_difference_l1224_122485


namespace find_n_l1224_122434

theorem find_n (n : ℤ) 
  (h : (3 + 16 + 33 + (n + 1)) / 4 = 20) : n = 27 := 
by
  sorry

end find_n_l1224_122434


namespace find_f_2014_l1224_122402

noncomputable def f (x : ℝ) : ℝ := sorry

axiom functional_eqn : ∀ x : ℝ, f x = f (x + 1) - f (x + 2)
axiom interval_def : ∀ x, 0 < x ∧ x < 3 → f x = x^2

theorem find_f_2014 : f 2014 = -1 := sorry

end find_f_2014_l1224_122402


namespace brownies_each_l1224_122453

theorem brownies_each (num_columns : ℕ) (num_rows : ℕ) (total_people : ℕ) (total_brownies : ℕ) 
(h1 : num_columns = 6) (h2 : num_rows = 3) (h3 : total_people = 6) 
(h4 : total_brownies = num_columns * num_rows) : 
total_brownies / total_people = 3 := 
by
  -- Placeholder for the actual proof
  sorry

end brownies_each_l1224_122453


namespace problem_equivalent_final_answer_l1224_122481

noncomputable def a := 12
noncomputable def b := 27
noncomputable def c := 6

theorem problem_equivalent :
  2 * Real.sqrt 3 + (2 / Real.sqrt 3) + 3 * Real.sqrt 2 + (3 / Real.sqrt 2) = (a * Real.sqrt 3 + b * Real.sqrt 2) / c :=
  sorry

theorem final_answer :
  a + b + c = 45 :=
  by
    unfold a b c
    simp
    done

end problem_equivalent_final_answer_l1224_122481


namespace ratio_of_larger_to_smaller_l1224_122492

theorem ratio_of_larger_to_smaller (x y : ℝ) (h1 : x > y) (h2 : x + y = 7 * (x - y)) (h3 : 0 < x) (h4 : 0 < y) : x / y = 4 / 3 := by
  sorry

end ratio_of_larger_to_smaller_l1224_122492


namespace Oliver_has_9_dollars_left_l1224_122464

def initial_amount := 9
def saved := 5
def earned := 6
def spent_frisbee := 4
def spent_puzzle := 3
def spent_stickers := 2
def spent_movie_ticket := 7
def spent_snack := 3
def gift := 8

def final_amount (initial_amount : ℕ) (saved : ℕ) (earned : ℕ) (spent_frisbee : ℕ)
                 (spent_puzzle : ℕ) (spent_stickers : ℕ) (spent_movie_ticket : ℕ)
                 (spent_snack : ℕ) (gift : ℕ) : ℕ :=
  initial_amount + saved + earned - spent_frisbee - spent_puzzle - spent_stickers - 
  spent_movie_ticket - spent_snack + gift

theorem Oliver_has_9_dollars_left :
  final_amount initial_amount saved earned spent_frisbee 
               spent_puzzle spent_stickers spent_movie_ticket 
               spent_snack gift = 9 :=
  by
  sorry

end Oliver_has_9_dollars_left_l1224_122464


namespace no_intersecting_axes_l1224_122499

theorem no_intersecting_axes (m : ℝ) : (m^2 + 2 * m - 7 = 0) → m = -4 :=
sorry

end no_intersecting_axes_l1224_122499


namespace cube_volume_doubled_l1224_122440

theorem cube_volume_doubled (a : ℝ) (h : a > 0) : 
  ((2 * a)^3 - a^3) / a^3 = 7 :=
by
  sorry

end cube_volume_doubled_l1224_122440


namespace simplify_eval_expression_l1224_122480

theorem simplify_eval_expression (x y : ℝ) (hx : x = -2) (hy : y = -1) :
  3 * (2 * x^2 + x * y + 1 / 3) - (3 * x^2 + 4 * x * y - y^2) = 11 :=
by
  rw [hx, hy]
  sorry

end simplify_eval_expression_l1224_122480


namespace coefficient_x_squared_in_expansion_l1224_122432

theorem coefficient_x_squared_in_expansion :
  (∃ c : ℤ, (1 + x)^6 * (1 - x) = c * x^2 + b * x + a) → c = 9 :=
by
  sorry

end coefficient_x_squared_in_expansion_l1224_122432


namespace eggs_not_eaten_per_week_l1224_122459

theorem eggs_not_eaten_per_week : 
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2 -- 2 eggs each by 2 children
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2  -- Re-calculated trays
  total_eggs_bought - total_eggs_eaten_per_week = 40 :=
by
  let trays_bought := 2
  let eggs_per_tray := 24
  let days_per_week := 7
  let eggs_eaten_by_children_per_day := 2 * 2
  let eggs_eaten_by_parents_per_day := 4
  let total_eggs_eaten_per_week := (eggs_eaten_by_children_per_day + eggs_eaten_by_parents_per_day) * days_per_week
  let total_eggs_bought := trays_bought * eggs_per_tray * 2
  show total_eggs_bought - total_eggs_eaten_per_week = 40
  sorry

end eggs_not_eaten_per_week_l1224_122459


namespace xena_head_start_l1224_122444

theorem xena_head_start
  (xena_speed : ℝ) (dragon_speed : ℝ) (time : ℝ) (burn_distance : ℝ) 
  (xena_speed_eq : xena_speed = 15) 
  (dragon_speed_eq : dragon_speed = 30) 
  (time_eq : time = 32) 
  (burn_distance_eq : burn_distance = 120) :
  (dragon_speed * time - burn_distance) - (xena_speed * time) = 360 := 
  by 
  sorry

end xena_head_start_l1224_122444


namespace y_x_cubed_monotonic_increasing_l1224_122407

theorem y_x_cubed_monotonic_increasing : 
  ∀ x1 x2 : ℝ, (x1 ≤ x2) → (x1^3 ≤ x2^3) :=
by
  intros x1 x2 h
  sorry

end y_x_cubed_monotonic_increasing_l1224_122407


namespace directrix_of_parabola_l1224_122426

-- Define the given condition:
def parabola_eq (x : ℝ) : ℝ := 8 * x^2 + 4 * x + 2

-- State the theorem:
theorem directrix_of_parabola :
  (∀ x : ℝ, parabola_eq x = 8 * (x + 1/4)^2 + 1) → (y = 31 / 32) :=
by
  -- We'll prove this later
  sorry

end directrix_of_parabola_l1224_122426


namespace favorable_probability_l1224_122439

noncomputable def probability_favorable_events (L : ℝ) : ℝ :=
  1 - (0.5 * (5 / 12 * L)^2 / (0.5 * L^2))

theorem favorable_probability (L : ℝ) (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ L)
  (h3 : 0 ≤ y) (h4 : y ≤ L)
  (h5 : 0 ≤ x + y) (h6 : x + y ≤ L)
  (h7 : x ≤ 5 / 12 * L) (h8 : y ≤ 5 / 12 * L)
  (h9 : x + y ≥ 7 / 12 * L) :
  probability_favorable_events L = 15 / 16 :=
by sorry

end favorable_probability_l1224_122439


namespace volume_of_prism_l1224_122460

theorem volume_of_prism (x y z : ℝ) (h1 : x * y = 100) (h2 : z = 10) (h3 : x * z = 50) (h4 : y * z = 40):
  x * y * z = 200 :=
by
  sorry

end volume_of_prism_l1224_122460


namespace feifei_sheep_count_l1224_122458

noncomputable def sheep_number (x y : ℕ) : Prop :=
  (y = 3 * x + 15) ∧ (x = y - y / 3)

theorem feifei_sheep_count :
  ∃ x y : ℕ, sheep_number x y ∧ x = 5 :=
sorry

end feifei_sheep_count_l1224_122458


namespace jade_handled_80_transactions_l1224_122431

variable (mabel anthony cal jade : ℕ)

-- Conditions
def mabel_transactions : mabel = 90 :=
by sorry

def anthony_transactions : anthony = mabel + (10 * mabel / 100) :=
by sorry

def cal_transactions : cal = 2 * anthony / 3 :=
by sorry

def jade_transactions : jade = cal + 14 :=
by sorry

-- Proof problem
theorem jade_handled_80_transactions :
  mabel = 90 →
  anthony = mabel + (10 * mabel / 100) →
  cal = 2 * anthony / 3 →
  jade = cal + 14 →
  jade = 80 :=
by
  intros
  subst_vars
  -- The proof steps would normally go here, but we leave it with sorry.
  sorry

end jade_handled_80_transactions_l1224_122431


namespace jerry_age_l1224_122484

theorem jerry_age (M J : ℕ) (h1 : M = 20) (h2 : M = 2 * J - 8) : J = 14 := 
by
  sorry

end jerry_age_l1224_122484


namespace even_function_a_eq_4_l1224_122498

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 4)

theorem even_function_a_eq_4 (a : ℝ) (h : ∀ x : ℝ, f (-x) a = f x a) : a = 4 := by
  sorry

end even_function_a_eq_4_l1224_122498


namespace range_of_m_l1224_122495

def proposition_p (m : ℝ) : Prop :=
  ∀ x > 0, m^2 + 2 * m - 1 ≤ x + 1 / x

def proposition_q (m : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (5 - m^2) ^ x > (5 - m^2) ^ (x - 1)

theorem range_of_m (m : ℝ) : (proposition_p m ∨ proposition_q m) ∧ ¬ (proposition_p m ∧ proposition_q m) ↔ (-3 ≤ m ∧ m ≤ -2) ∨ (1 < m ∧ m < 2) :=
sorry

end range_of_m_l1224_122495


namespace fraction_evaluation_l1224_122475

theorem fraction_evaluation :
  (2 + 4 - 8 + 16 + 32 - 64 + 128 : ℚ) / (4 + 8 - 16 + 32 + 64 - 128 + 256) = 1 / 2 :=
by
  sorry

end fraction_evaluation_l1224_122475


namespace families_received_boxes_l1224_122474

theorem families_received_boxes (F : ℕ) (box_decorations total_decorations : ℕ)
  (h_box_decorations : box_decorations = 10)
  (h_total_decorations : total_decorations = 120)
  (h_eq : box_decorations * (F + 1) = total_decorations) :
  F = 11 :=
by
  sorry

end families_received_boxes_l1224_122474


namespace compute_expression_l1224_122427

theorem compute_expression : 10 * (3 / 27) * 36 = 40 := 
by 
  sorry

end compute_expression_l1224_122427


namespace mixture_weight_l1224_122450

theorem mixture_weight (C : ℚ) (W : ℚ)
  (H1: C > 0) -- C represents the cost per pound of milk powder and coffee in June, and is a positive number
  (H2: C * 0.2 = 0.2) -- The price per pound of milk powder in July
  (H3: (W / 2) * 0.2 + (W / 2) * 4 * C = 6.30) -- The cost of the mixture in July

  : W = 3 := 
sorry

end mixture_weight_l1224_122450


namespace diagonals_of_seven_sided_polygon_l1224_122433

-- Define the number of sides of the polygon
def n : ℕ := 7

-- Calculate the number of diagonals in a polygon with n sides
def numberOfDiagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- The statement to prove
theorem diagonals_of_seven_sided_polygon : numberOfDiagonals n = 14 := by
  -- Here we will write the proof steps, but they're not needed now.
  sorry

end diagonals_of_seven_sided_polygon_l1224_122433


namespace translated_circle_eq_l1224_122445

theorem translated_circle_eq (x y : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 16) →
  (x + 5) ^ 2 + (y + 3) ^ 2 = 16 :=
by
  sorry

end translated_circle_eq_l1224_122445


namespace xyz_sum_is_22_l1224_122421

theorem xyz_sum_is_22 (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h1 : x * y = 24) (h2 : x * z = 48) (h3 : y * z = 72) : 
  x + y + z = 22 :=
sorry

end xyz_sum_is_22_l1224_122421


namespace tan_a6_of_arithmetic_sequence_l1224_122466

noncomputable def arithmetic_sequence (a : ℕ → ℝ) := 
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := 
  ∀ n : ℕ, S n = (n * (a 1 + a n)) / 2

theorem tan_a6_of_arithmetic_sequence
  (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (H1 : arithmetic_sequence a)
  (H2 : sum_of_first_n_terms a S)
  (H3 : S 11 = 22 * Real.pi / 3) : 
  Real.tan (a 6) = -Real.sqrt 3 :=
sorry

end tan_a6_of_arithmetic_sequence_l1224_122466


namespace sqrt_of_36_is_6_l1224_122451

-- Define the naturals
def arithmetic_square_root (x : ℕ) : ℕ := Nat.sqrt x

theorem sqrt_of_36_is_6 : arithmetic_square_root 36 = 6 :=
by
  -- The proof goes here, but we use sorry to skip it as per instructions.
  sorry

end sqrt_of_36_is_6_l1224_122451


namespace necklaces_caught_l1224_122496

theorem necklaces_caught
  (LatchNecklaces RhondaNecklaces BoudreauxNecklaces: ℕ)
  (h1 : LatchNecklaces = 3 * RhondaNecklaces - 4)
  (h2 : RhondaNecklaces = BoudreauxNecklaces / 2)
  (h3 : BoudreauxNecklaces = 12) :
  LatchNecklaces = 14 := by
  sorry

end necklaces_caught_l1224_122496


namespace saeyoung_yen_value_l1224_122457

-- Define the exchange rate
def exchange_rate : ℝ := 17.25

-- Define Saeyoung's total yuan
def total_yuan : ℝ := 1000 + 10

-- Define the total yen based on the exchange rate
def total_yen : ℝ := total_yuan * exchange_rate

-- State the theorem
theorem saeyoung_yen_value : total_yen = 17422.5 :=
by
  sorry

end saeyoung_yen_value_l1224_122457


namespace quadratic_vertex_coordinates_l1224_122489

-- Define the quadratic function
def quadratic (x : ℝ) : ℝ :=
  -2 * (x + 1)^2 - 4

-- State the main theorem to be proved: The vertex of the quadratic function is at (-1, -4)
theorem quadratic_vertex_coordinates : 
  ∃ h k : ℝ, ∀ x : ℝ, quadratic x = -2 * (x + h)^2 + k ∧ h = -1 ∧ k = -4 := 
by
  -- proof required here
  sorry

end quadratic_vertex_coordinates_l1224_122489


namespace factor_expression_l1224_122473

theorem factor_expression (x : ℝ) : 
  ((4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8)) = 2 * x^2 * (5 * x + 31) := 
by sorry

end factor_expression_l1224_122473


namespace sum_of_money_l1224_122491

-- Conditions
def mass_record_coin_kg : ℝ := 100  -- 100 kg
def mass_one_pound_coin_g : ℝ := 10  -- 10 g

-- Conversion factor
def kg_to_g : ℝ := 1000

-- Question: Prove the sum of money in £1 coins that weighs the same as the record-breaking coin is £10,000.
theorem sum_of_money 
  (mass_record_coin_g := mass_record_coin_kg * kg_to_g)
  (number_of_coins := mass_record_coin_g / mass_one_pound_coin_g) 
  (sum_of_money := number_of_coins) : 
  sum_of_money = 10000 :=
  sorry

end sum_of_money_l1224_122491


namespace yogurt_combinations_l1224_122456

-- Define the conditions from a)
def num_flavors : ℕ := 5
def num_toppings : ℕ := 8
def num_sizes : ℕ := 3

-- Define the problem in a theorem statement
theorem yogurt_combinations : num_flavors * ((num_toppings * (num_toppings - 1)) / 2) * num_sizes = 420 :=
by
  -- sorry is used here to skip the proof
  sorry

end yogurt_combinations_l1224_122456


namespace picking_ball_is_random_event_l1224_122462

-- Definitions based on problem conditions
def total_balls := 201
def black_balls := 200
def white_balls := 1

-- The goal to prove
theorem picking_ball_is_random_event : 
  (total_balls = black_balls + white_balls) ∧ 
  (black_balls > 0) ∧ 
  (white_balls > 0) → 
  random_event :=
by sorry

end picking_ball_is_random_event_l1224_122462


namespace find_ratio_l1224_122490

noncomputable def p (x : ℝ) : ℝ := 3 * x * (x - 5)
noncomputable def q (x : ℝ) : ℝ := (x + 2) * (x - 5)

theorem find_ratio : (p 3) / (q 3) = 9 / 5 := by
  sorry

end find_ratio_l1224_122490


namespace find_x_value_l1224_122477

theorem find_x_value (x : ℝ) (h₀ : 0 < x ∧ x < 180) (h₁ : Real.tan (150 - x) = (Real.sin 150 - Real.sin x) / (Real.cos 150 - Real.cos x)) : x = 130 := 
by
  sorry

end find_x_value_l1224_122477


namespace value_of_a_l1224_122454

def f (a x : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 + 2

def f_prime (a x : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x

theorem value_of_a (a : ℝ) (h : f_prime a (-1) = 4) : a = 10 / 3 :=
by
  -- Proof goes here
  sorry

end value_of_a_l1224_122454


namespace woman_wait_time_to_be_caught_l1224_122408

theorem woman_wait_time_to_be_caught 
  (man_speed_mph : ℝ) (woman_speed_mph : ℝ) (wait_time_minutes : ℝ) 
  (conversion_factor : ℝ) (distance_apart_miles : ℝ) :
  man_speed_mph = 6 →
  woman_speed_mph = 12 →
  wait_time_minutes = 10 →
  conversion_factor = 1 / 60 →
  distance_apart_miles = (woman_speed_mph * conversion_factor) * wait_time_minutes →
  ∃ minutes_to_catch_up : ℝ, minutes_to_catch_up = distance_apart_miles / (man_speed_mph * conversion_factor) ∧ minutes_to_catch_up = 20 := sorry

end woman_wait_time_to_be_caught_l1224_122408


namespace hyperbola_condition_l1224_122424

theorem hyperbola_condition (k : ℝ) : (3 - k) * (k - 2) < 0 ↔ k < 2 ∨ k > 3 := by
  sorry

end hyperbola_condition_l1224_122424


namespace cube_root_of_nine_irrational_l1224_122412

theorem cube_root_of_nine_irrational : ¬ ∃ (r : ℚ), r^3 = 9 :=
by sorry

end cube_root_of_nine_irrational_l1224_122412


namespace total_number_of_elements_l1224_122405

theorem total_number_of_elements (a b c : ℕ) : 
  (a = 2 ∧ b = 2 ∧ c = 2) ∧ 
  (3.95 = ((4.4 * 2 + 3.85 * 2 + 3.6000000000000014 * 2) / 6)) ->
  a + b + c = 6 := 
by
  sorry

end total_number_of_elements_l1224_122405


namespace find_m_of_line_with_slope_l1224_122461

theorem find_m_of_line_with_slope (m : ℝ) (h_pos : m > 0)
(h_slope : (m - 4) / (2 - m) = m^2) : m = 2 := by
  sorry

end find_m_of_line_with_slope_l1224_122461


namespace area_ratio_l1224_122404

noncomputable def initial_areas (a b c : ℝ) :=
  a > 0 ∧ b > 0 ∧ c > 0

noncomputable def misallocated_areas (a b : ℝ) :=
  let b' := b + 0.1 * a - 0.5 * b
  b' = 0.4 * (a + b)

noncomputable def final_ratios (a b c : ℝ) :=
  let a' := 0.9 * a + 0.5 * b
  let b' := b + 0.1 * a - 0.5 * b
  let c' := 0.5 * c
  a' + b' + c' = a + b + c ∧ a' / b' = 2 ∧ b' / c' = 1 

theorem area_ratio (a b c m : ℝ) (h1 : initial_areas a b c) 
  (h2 : misallocated_areas a b)
  (h3 : final_ratios a b c) : 
  (m = 0.4 * a) → (m / (a + b + c) = 1 / 20) :=
sorry

end area_ratio_l1224_122404


namespace probability_prime_sum_is_correct_l1224_122435

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def cube_rolls_prob_prime_sum : ℚ :=
  let possible_outcomes := 36
  let prime_sums_count := 15
  prime_sums_count / possible_outcomes

theorem probability_prime_sum_is_correct :
  cube_rolls_prob_prime_sum = 5 / 12 :=
by
  -- The problem statement verifies that we have to show the calculation is correct
  sorry

end probability_prime_sum_is_correct_l1224_122435


namespace gcd_of_X_and_Y_l1224_122409

theorem gcd_of_X_and_Y (X Y : ℕ) (h1 : Nat.lcm X Y = 180) (h2 : 5 * X = 4 * Y) :
  Nat.gcd X Y = 9 := 
sorry

end gcd_of_X_and_Y_l1224_122409


namespace find_a_if_even_function_l1224_122493

-- Problem statement in Lean 4
theorem find_a_if_even_function (a : ℝ) (f : ℝ → ℝ) 
  (h : ∀ x, f x = x^2 - 2 * (a + 1) * x + 1) 
  (hf_even : ∀ x, f x = f (-x)) : a = -1 :=
sorry

end find_a_if_even_function_l1224_122493


namespace saline_solution_concentration_l1224_122463

theorem saline_solution_concentration
  (C : ℝ) -- concentration of the first saline solution
  (h1 : 3.6 * C + 1.4 * 9 = 5 * 3.24) : -- condition based on the total salt content
  C = 1 := 
sorry

end saline_solution_concentration_l1224_122463


namespace evaluate_neg2012_l1224_122428

def func (a b c x : ℝ) : ℝ := a * x^5 + b * x^3 + c * x + 1

theorem evaluate_neg2012 (a b c : ℝ) (h : func a b c 2012 = 3) : func a b c (-2012) = -1 :=
by
  sorry

end evaluate_neg2012_l1224_122428


namespace handshake_even_acquaintance_l1224_122468

theorem handshake_even_acquaintance (n : ℕ) (hn : n = 225) : 
  ∃ (k : ℕ), k < n ∧ (∀ m < n, k ≠ m) :=
by sorry

end handshake_even_acquaintance_l1224_122468


namespace price_of_water_margin_comics_l1224_122425

-- Define the conditions
variables (x : ℕ) (y : ℕ)

-- Condition 1: Price relationship
def price_relationship : Prop := y = x + 60

-- Condition 2: Total expenditure on Romance of the Three Kingdoms comic books
def total_expenditure_romance_three_kingdoms : Prop := 60 * (y / 60) = 3600

-- Condition 3: Total expenditure on Water Margin comic books
def total_expenditure_water_margin : Prop := 120 * (x / 120) = 4800

-- Condition 4: Number of sets relationship
def number_of_sets_relationship : Prop := y = (4800 / x) / 2

-- The main statement to prove
theorem price_of_water_margin_comics (x : ℕ) (h1: price_relationship x (x + 60))
  (h2: total_expenditure_romance_three_kingdoms x)
  (h3: total_expenditure_water_margin x)
  (h4: number_of_sets_relationship x (x + 60)) : x = 120 :=
sorry

end price_of_water_margin_comics_l1224_122425


namespace scalar_mult_l1224_122449

variables {α : Type*} [AddCommGroup α] [Module ℝ α]

theorem scalar_mult (a : α) (h : a ≠ 0) : (-4) • (3 • a) = -12 • a :=
  sorry

end scalar_mult_l1224_122449


namespace radius_increase_l1224_122416

theorem radius_increase (ΔC : ℝ) (ΔC_eq : ΔC = 0.628) : Δr = 0.1 :=
by
  sorry

end radius_increase_l1224_122416


namespace triangle_area_upper_bound_l1224_122414

variable {α : Type u}
variable [LinearOrderedField α]
variable {A B C : α} -- Points A, B, C as elements of some field.

-- Definitions for the lengths of the sides, interpreted as scalar distances.
variable (AB AC : α)

-- Assume that AB and AC are lengths of sides of the triangle
-- Assume the area of the triangle is non-negative and does not exceed the specified bound.
theorem triangle_area_upper_bound (S : α) (habc : S = (1 / 2) * AB * AC) :
  S ≤ (1 / 2) * AB * AC := 
sorry

end triangle_area_upper_bound_l1224_122414


namespace max_difference_y_coords_intersection_l1224_122447

def f (x : ℝ) : ℝ := 4 - x^2 + x^3
def g (x : ℝ) : ℝ := x^2 + x^4

theorem max_difference_y_coords_intersection : ∀ x : ℝ, 
  (f x = g x) → 
  (∀ x₁ x₂ : ℝ, f x₁ = g x₁ ∧ f x₂ = g x₂ → |f x₁ - f x₂| = 0) := 
by
  sorry

end max_difference_y_coords_intersection_l1224_122447


namespace total_shoes_tried_on_l1224_122436

variable (T : Type)
variable (store1 store2 store3 store4 : T)
variable (pair_of_shoes : T → ℕ)
variable (c1 : pair_of_shoes store1 = 7)
variable (c2 : pair_of_shoes store2 = pair_of_shoes store1 + 2)
variable (c3 : pair_of_shoes store3 = 0)
variable (c4 : pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3))

theorem total_shoes_tried_on (store1 store2 store3 store4 : T) (pair_of_shoes : T → ℕ) : 
  pair_of_shoes store1 = 7 →
  pair_of_shoes store2 = pair_of_shoes store1 + 2 →
  pair_of_shoes store3 = 0 →
  pair_of_shoes store4 = 2 * (pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3) →
  pair_of_shoes store1 + pair_of_shoes store2 + pair_of_shoes store3 + pair_of_shoes store4 = 48 := by
  intro c1 c2 c3 c4
  sorry

end total_shoes_tried_on_l1224_122436


namespace overlapping_squares_area_l1224_122482

theorem overlapping_squares_area :
  let s : ℝ := 5
  let total_area := 3 * s^2
  let redundant_area := s^2 / 8 * 4
  total_area - redundant_area = 62.5 := by
  sorry

end overlapping_squares_area_l1224_122482


namespace expression_of_24ab_in_P_and_Q_l1224_122470

theorem expression_of_24ab_in_P_and_Q (a b : ℕ) (P Q : ℝ)
  (hP : P = 2^a) (hQ : Q = 5^b) : 24^(a*b) = P^(3*b) * 3^(a*b) := 
  by
  sorry

end expression_of_24ab_in_P_and_Q_l1224_122470


namespace value_of_a_b_l1224_122419

theorem value_of_a_b:
  ∃ (a b : ℕ), a = 3 ∧ b = 2 ∧ (a + 6 * 10^3 + 7 * 10^2 + 9 * 10 + b) % 72 = 0 :=
by
  sorry

end value_of_a_b_l1224_122419


namespace calculate_expression_evaluate_expression_l1224_122422

theorem calculate_expression (a : ℕ) (h : a = 2020) :
  (a^4 - 3*a^3*(a+1) + 4*a*(a+1)^3 - (a+1)^4 + 1) / (a*(a+1)) = a^2 + 4*a + 6 :=
by sorry

theorem evaluate_expression :
  (2020^2 + 4 * 2020 + 6) = 4096046 :=
by sorry

end calculate_expression_evaluate_expression_l1224_122422


namespace number_of_perfect_square_factors_l1224_122442

theorem number_of_perfect_square_factors (a b c d : ℕ) :
  (∀ a b c d, 
    (0 ≤ a ∧ a ≤ 4) ∧ 
    (0 ≤ b ∧ b ≤ 2) ∧ 
    (0 ≤ c ∧ c ≤ 1) ∧ 
    (0 ≤ d ∧ d ≤ 1) ∧ 
    (a % 2 = 0) ∧ 
    (b % 2 = 0) ∧ 
    (c = 0) ∧ 
    (d = 0)
  → 3 * 2 * 1 * 1 = 6) := by
  sorry

end number_of_perfect_square_factors_l1224_122442


namespace sum_11_terms_l1224_122437

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop := 
  ∃ d, ∀ n, a (n + 1) = a n + d

def sum_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n / 2) * (a 1 + a n)

def condition (a : ℕ → ℝ) : Prop :=
  a 5 + a 7 = 14

-- Proof Problem
theorem sum_11_terms (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h_arith_seq : arithmetic_sequence a)
  (h_sum_formula : sum_first_n_terms S a)
  (h_condition : condition a) :
  S 11 = 77 := 
sorry

end sum_11_terms_l1224_122437


namespace rectangle_area_l1224_122476

noncomputable def width := 14
noncomputable def length := width + 6
noncomputable def perimeter := 2 * width + 2 * length
noncomputable def area := width * length

theorem rectangle_area (h1 : length = width + 6) (h2 : perimeter = 68) : area = 280 := 
by 
  have hw : width = 14 := by sorry 
  have hl : length = 20 := by sorry 
  have harea : area = 280 := by sorry
  exact harea

end rectangle_area_l1224_122476


namespace thirty_sixty_ninety_triangle_area_l1224_122486

theorem thirty_sixty_ninety_triangle_area (hypotenuse : ℝ) (angle : ℝ) (area : ℝ)
  (h_hypotenuse : hypotenuse = 12)
  (h_angle : angle = 30)
  (h_area : area = 18 * Real.sqrt 3) :
  ∃ (base height : ℝ), 
    base = hypotenuse / 2 ∧ 
    height = (hypotenuse / 2) * Real.sqrt 3 ∧ 
    area = (1 / 2) * base * height :=
by {
  sorry
}

end thirty_sixty_ninety_triangle_area_l1224_122486


namespace coeff_x3_in_expansion_l1224_122420

theorem coeff_x3_in_expansion : 
  ∃ c : ℕ, (c = 80) ∧ (∃ r : ℕ, r = 1 ∧ (2 * x + 1 / x) ^ 5 = (2 * x) ^ (5 - r) * (1 / x) ^ r)
:= sorry

end coeff_x3_in_expansion_l1224_122420


namespace x_must_be_even_l1224_122441

theorem x_must_be_even (x : ℤ) (h : ∃ (n : ℤ), (2 * x / 3 - x / 6) = n) : ∃ (k : ℤ), x = 2 * k :=
by
  sorry

end x_must_be_even_l1224_122441


namespace scaled_triangle_height_l1224_122415

theorem scaled_triangle_height (h b₁ h₁ b₂ h₂ : ℝ)
  (h₁_eq : h₁ = 6) (b₁_eq : b₁ = 12) (b₂_eq : b₂ = 8) :
  (b₁ / h₁ = b₂ / h₂) → h₂ = 4 :=
by
  -- Given conditions
  have h₁_eq : h₁ = 6 := h₁_eq
  have b₁_eq : b₁ = 12 := b₁_eq
  have b₂_eq : b₂ = 8 := b₂_eq
  -- Proof will go here
  sorry

end scaled_triangle_height_l1224_122415


namespace micah_total_strawberries_l1224_122494

theorem micah_total_strawberries (eaten saved total : ℕ) 
  (h1 : eaten = 6) 
  (h2 : saved = 18) 
  (h3 : total = eaten + saved) : 
  total = 24 := 
by
  sorry

end micah_total_strawberries_l1224_122494


namespace total_team_formation_plans_l1224_122423

def numberOfWaysToChooseDoctors (m f : ℕ) (k : ℕ) : ℕ :=
  (Nat.choose m (k - 1) * Nat.choose f 1) +
  (Nat.choose m 1 * Nat.choose f (k - 1))

theorem total_team_formation_plans :
  let m := 5
  let f := 4
  let total := 3
  numberOfWaysToChooseDoctors m f total = 70 :=
by
  let m := 5
  let f := 4
  let total := 3
  unfold numberOfWaysToChooseDoctors
  sorry

end total_team_formation_plans_l1224_122423


namespace max_terms_in_arithmetic_seq_l1224_122452

variable (a n : ℝ)

def arithmetic_seq_max_terms (a n : ℝ) : Prop :=
  let d := 4
  a^2 + (n - 1) * (a + d) + (n - 1) * n / 2 * d ≤ 100

theorem max_terms_in_arithmetic_seq (a n : ℝ) (h : arithmetic_seq_max_terms a n) : n ≤ 8 :=
sorry

end max_terms_in_arithmetic_seq_l1224_122452


namespace divide_rope_into_parts_l1224_122467

theorem divide_rope_into_parts:
  (∀ rope_length : ℝ, rope_length = 5 -> ∀ parts : ℕ, parts = 4 -> (∀ i : ℕ, i < parts -> ((rope_length / parts) = (5 / 4)))) :=
by sorry

end divide_rope_into_parts_l1224_122467


namespace rhombus_locus_l1224_122406

-- Define the coordinates of the vertices of the rhombus
structure Point :=
(x : ℝ)
(y : ℝ)

def A (e : ℝ) : Point := ⟨e, 0⟩
def B (f : ℝ) : Point := ⟨0, f⟩
def C (e : ℝ) : Point := ⟨-e, 0⟩
def D (f : ℝ) : Point := ⟨0, -f⟩

-- Define the distance squared from a point P to a point Q
def dist_sq (P Q : Point) : ℝ := (P.x - Q.x)^2 + (P.y - Q.y)^2

-- Define the geometric locus problem
theorem rhombus_locus (P : Point) (e f : ℝ) :
  dist_sq P (A e) = dist_sq P (B f) + dist_sq P (C e) + dist_sq P (D f) ↔
  (if e > f then
    (dist_sq P (A e) = (e^2 - f^2) ∨ dist_sq P (C e) = (e^2 - f^2))
   else if e = f then
    (P = A e ∨ P = B f ∨ P = C e ∨ P = D f)
   else
    false) :=
sorry

end rhombus_locus_l1224_122406


namespace differential_equation_approx_solution_l1224_122401

open Real

noncomputable def approximate_solution (x : ℝ) : ℝ := 0.1 * exp (x ^ 2 / 2)

theorem differential_equation_approx_solution :
  ∀ (x : ℝ), -1/2 ≤ x ∧ x ≤ 1/2 →
  ∀ (y : ℝ), -1/2 ≤ y ∧ y ≤ 1/2 →
  abs (approximate_solution x - y) < 1 / 650 :=
sorry

end differential_equation_approx_solution_l1224_122401


namespace arithmetic_sum_calculation_l1224_122410

theorem arithmetic_sum_calculation :
  3 * (71 + 75 + 79 + 83 + 87 + 91) = 1458 :=
by
  sorry

end arithmetic_sum_calculation_l1224_122410


namespace num_carnations_l1224_122411

-- Define the conditions
def num_roses : ℕ := 5
def total_flowers : ℕ := 10

-- Define the statement we want to prove
theorem num_carnations : total_flowers - num_roses = 5 :=
by {
  -- The proof itself is not required, so we use 'sorry' to indicate incomplete proof
  sorry
}

end num_carnations_l1224_122411


namespace Tom_allowance_leftover_l1224_122417

theorem Tom_allowance_leftover :
  let initial_allowance := 12
  let first_week_spending := (1/3) * initial_allowance
  let remaining_after_first_week := initial_allowance - first_week_spending
  let second_week_spending := (1/4) * remaining_after_first_week
  let final_amount := remaining_after_first_week - second_week_spending
  final_amount = 6 :=
by
  sorry

end Tom_allowance_leftover_l1224_122417


namespace minimum_value_l1224_122403

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem minimum_value :
  ∃ x₀ : ℝ, (∀ x : ℝ, f x₀ ≤ f x) ∧ f x₀ = -2 := by
  sorry

end minimum_value_l1224_122403


namespace sum_of_roots_l1224_122446

theorem sum_of_roots {x1 x2 x3 k m : ℝ} (h1 : x1 ≠ x2) (h2 : x2 ≠ x3) (h3 : x1 ≠ x3)
  (h4 : 2 * x1^3 - k * x1 = m) (h5 : 2 * x2^3 - k * x2 = m) (h6 : 2 * x3^3 - k * x3 = m) :
  x1 + x2 + x3 = 0 :=
sorry

end sum_of_roots_l1224_122446


namespace min_y_value_l1224_122488

theorem min_y_value (x : ℝ) : 
  (∀ y : ℝ, y = 4 * x^2 + 8 * x + 16 → y ≥ 12 ∧ (y = 12 ↔ x = -1)) :=
sorry

end min_y_value_l1224_122488


namespace integer_triples_condition_l1224_122429

theorem integer_triples_condition (p q r : ℤ) (h1 : 1 < p) (h2 : p < q) (h3 : q < r) 
  (h4 : ((p - 1) * (q - 1) * (r - 1)) ∣ (p * q * r - 1)) : (p = 2 ∧ q = 4 ∧ r = 8) ∨ (p = 3 ∧ q = 5 ∧ r = 15) :=
sorry

end integer_triples_condition_l1224_122429


namespace initial_quantity_of_milk_l1224_122472

-- Define initial condition for the quantity of milk in container A
noncomputable def container_A : ℝ := 1184

-- Define the quantities of milk in containers B and C
def container_B (A : ℝ) : ℝ := 0.375 * A
def container_C (A : ℝ) : ℝ := 0.625 * A

-- Define the final equal quantities of milk after transfer
def equal_quantity (A : ℝ) : ℝ := container_B A + 148

-- The proof statement that must be true
theorem initial_quantity_of_milk :
  ∀ (A : ℝ), container_B A + 148 = equal_quantity A → A = container_A :=
by
  intros A h
  rw [equal_quantity] at h
  sorry

end initial_quantity_of_milk_l1224_122472


namespace rain_ratio_l1224_122438

def monday_rain := 2 + 1 -- inches of rain on Monday
def wednesday_rain := 0 -- inches of rain on Wednesday
def thursday_rain := 1 -- inches of rain on Thursday
def average_rain_per_day := 4 -- daily average rain total
def days_in_week := 5 -- days in a week
def weekly_total_rain := average_rain_per_day * days_in_week

-- Theorem statement
theorem rain_ratio (tuesday_rain : ℝ) (friday_rain : ℝ) 
  (h1 : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h2 : monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = weekly_total_rain) :
  tuesday_rain / monday_rain = 2 := 
sorry

end rain_ratio_l1224_122438


namespace universal_friendship_l1224_122448

-- Define the inhabitants and their relationships
def inhabitants (n : ℕ) : Type := Fin n

-- Condition for friends and enemies
inductive Relationship (n : ℕ) : inhabitants n → inhabitants n → Prop
| friend (A B : inhabitants n) : Relationship n A B
| enemy (A B : inhabitants n) : Relationship n A B

-- Transitivity condition
axiom transitivity {n : ℕ} {A B C : inhabitants n} :
  Relationship n A B = Relationship n B C → Relationship n A C = Relationship n A B

-- At least two friends among any three inhabitants
axiom at_least_two_friends {n : ℕ} (A B C : inhabitants n) :
  ∃ X Y : inhabitants n, X ≠ Y ∧ Relationship n X Y = Relationship n X Y

-- Inhabitants can start a new life switching relationships
axiom start_new_life {n : ℕ} (A : inhabitants n) :
  ∀ B : inhabitants n, Relationship n A B = Relationship n B A

-- The main theorem we need to prove
theorem universal_friendship (n : ℕ) : 
  ∀ A B : inhabitants n, ∃ C : inhabitants n, Relationship n A C = Relationship n B C :=
sorry

end universal_friendship_l1224_122448


namespace solve_eq1_solve_eq2_l1224_122478

theorem solve_eq1 (x : ℝ):
  (x - 1) * (x + 3) = x - 1 ↔ x = 1 ∨ x = -2 :=
by 
  sorry

theorem solve_eq2 (x : ℝ):
  2 * x^2 - 6 * x = -3 ↔ x = (3 + Real.sqrt 3) / 2 ∨ x = (3 - Real.sqrt 3) / 2 :=
by 
  sorry

end solve_eq1_solve_eq2_l1224_122478
