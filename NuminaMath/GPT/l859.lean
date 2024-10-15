import Mathlib

namespace NUMINAMATH_GPT_geometric_sequence_term_l859_85905

theorem geometric_sequence_term (a_n : ℕ → ℝ) (S_n : ℕ → ℝ) (n : ℕ) :
  (∀ n, S_n n = 3^n - 1) →
  (a_n n = S_n n - S_n (n - 1)) →
  (a_n n = 2 * 3^(n - 1)) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_geometric_sequence_term_l859_85905


namespace NUMINAMATH_GPT_janet_dresses_total_pockets_l859_85941

theorem janet_dresses_total_pockets :
  let dresses := 24
  let with_pockets := dresses / 2
  let with_two_pockets := with_pockets / 3
  let with_three_pockets := with_pockets - with_two_pockets
  let total_two_pockets := with_two_pockets * 2
  let total_three_pockets := with_three_pockets * 3
  total_two_pockets + total_three_pockets = 32 := by
  sorry

end NUMINAMATH_GPT_janet_dresses_total_pockets_l859_85941


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l859_85989

theorem arithmetic_sequence_common_difference 
  (d : ℝ) (h : d ≠ 0) (a : ℕ → ℝ)
  (h1 : a 1 = 9 * d)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (k : ℕ) :
  (a k)^2 = (a 1) * (a (2 * k)) → k = 4 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l859_85989


namespace NUMINAMATH_GPT_algebraic_expression_value_l859_85915

theorem algebraic_expression_value (x y : ℝ) (h : x + 2 * y = 2) : 2 * x + 4 * y - 1 = 3 :=
sorry

end NUMINAMATH_GPT_algebraic_expression_value_l859_85915


namespace NUMINAMATH_GPT_rakesh_salary_l859_85952

variable (S : ℝ) -- The salary S is a real number
variable (h : 0.595 * S = 2380) -- Condition derived from the problem

theorem rakesh_salary : S = 4000 :=
by
  sorry

end NUMINAMATH_GPT_rakesh_salary_l859_85952


namespace NUMINAMATH_GPT_value_of_r_when_m_eq_3_l859_85996

theorem value_of_r_when_m_eq_3 :
  ∀ (r t m : ℕ),
  r = 5^t - 2*t →
  t = 3^m + 2 →
  m = 3 →
  r = 5^29 - 58 :=
by
  intros r t m h1 h2 h3
  rw [h3] at h2
  rw [Nat.pow_succ] at h2
  sorry

end NUMINAMATH_GPT_value_of_r_when_m_eq_3_l859_85996


namespace NUMINAMATH_GPT_orange_weight_l859_85907

variable (A O : ℕ)

theorem orange_weight (h1 : O = 5 * A) (h2 : A + O = 12) : O = 10 :=
  sorry

end NUMINAMATH_GPT_orange_weight_l859_85907


namespace NUMINAMATH_GPT_arcsin_one_half_l859_85965

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_one_half_l859_85965


namespace NUMINAMATH_GPT_trip_to_office_duration_l859_85922

noncomputable def distance (D : ℝ) : Prop :=
  let T1 := D / 58
  let T2 := D / 62
  T1 + T2 = 3

theorem trip_to_office_duration (D : ℝ) (h : distance D) : D / 58 = 1.55 :=
by sorry

end NUMINAMATH_GPT_trip_to_office_duration_l859_85922


namespace NUMINAMATH_GPT_original_salary_l859_85920

-- Given conditions as definitions
def salaryAfterRaise (x : ℝ) : ℝ := 1.10 * x
def salaryAfterReduction (x : ℝ) : ℝ := salaryAfterRaise x * 0.95
def finalSalary : ℝ := 1045

-- Statement to prove
theorem original_salary (x : ℝ) (h : salaryAfterReduction x = finalSalary) : x = 1000 :=
by
  sorry

end NUMINAMATH_GPT_original_salary_l859_85920


namespace NUMINAMATH_GPT_lattice_points_in_bounded_region_l859_85970

def isLatticePoint (p : ℤ × ℤ) : Prop :=
  true  -- All (n, m) ∈ ℤ × ℤ are lattice points

def boundedRegion (x y : ℤ) : Prop :=
  y = x ^ 2 ∨ y = 8 - x ^ 2
  
theorem lattice_points_in_bounded_region :
  ∃ S : Finset (ℤ × ℤ), 
    (∀ p ∈ S, isLatticePoint p ∧ boundedRegion p.1 p.2) ∧ S.card = 17 :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_in_bounded_region_l859_85970


namespace NUMINAMATH_GPT_closed_path_even_length_l859_85978

def is_closed_path (steps : List Char) : Bool :=
  let net_vertical := steps.count 'U' - steps.count 'D'
  let net_horizontal := steps.count 'R' - steps.count 'L'
  net_vertical = 0 ∧ net_horizontal = 0

def move_length (steps : List Char) : Nat :=
  steps.length

theorem closed_path_even_length (steps : List Char) :
  is_closed_path steps = true → move_length steps % 2 = 0 :=
by
  -- Conditions extracted as definitions
  intros h
  -- The proof will handle showing that the length of the closed path is even
  sorry

end NUMINAMATH_GPT_closed_path_even_length_l859_85978


namespace NUMINAMATH_GPT_train_crossing_time_l859_85918

-- Defining a structure for our problem context
structure TrainCrossing where
  length : Real -- length of the train in meters
  speed_kmh : Real -- speed of the train in km/h
  conversion_factor : Real -- conversion factor from km/h to m/s

-- Given the conditions in the problem
def trainData : TrainCrossing :=
  ⟨ 280, 50.4, 0.27778 ⟩

-- The main theorem statement:
theorem train_crossing_time (data : TrainCrossing) : 
  data.length / (data.speed_kmh * data.conversion_factor) = 20 := 
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l859_85918


namespace NUMINAMATH_GPT_original_earnings_l859_85960

variable (x : ℝ) -- John's original weekly earnings

theorem original_earnings:
  (1.20 * x = 72) → 
  (x = 60) :=
by
  intro h
  sorry

end NUMINAMATH_GPT_original_earnings_l859_85960


namespace NUMINAMATH_GPT_solve_integer_pairs_l859_85959

-- Definition of the predicate that (m, n) satisfies the given equation
def satisfies_equation (m n : ℤ) : Prop :=
  m * n^2 = 2009 * (n + 1)

-- Theorem stating that the only solutions are (4018, 1) and (0, -1)
theorem solve_integer_pairs :
  ∀ (m n : ℤ), satisfies_equation m n ↔ (m = 4018 ∧ n = 1) ∨ (m = 0 ∧ n = -1) :=
by
  sorry

end NUMINAMATH_GPT_solve_integer_pairs_l859_85959


namespace NUMINAMATH_GPT_least_x_for_inequality_l859_85982

theorem least_x_for_inequality : 
  ∃ (x : ℝ), (-x^2 + 9 * x - 20 ≤ 0) ∧ ∀ y, (-y^2 + 9 * y - 20 ≤ 0) → x ≤ y ∧ x = 4 := 
by
  sorry

end NUMINAMATH_GPT_least_x_for_inequality_l859_85982


namespace NUMINAMATH_GPT_chef_sold_12_meals_l859_85980

theorem chef_sold_12_meals
  (initial_meals_lunch : ℕ)
  (additional_meals_dinner : ℕ)
  (meals_left_after_lunch : ℕ)
  (meals_for_dinner : ℕ)
  (H1 : initial_meals_lunch = 17)
  (H2 : additional_meals_dinner = 5)
  (H3 : meals_for_dinner = 10) :
  ∃ (meals_sold_lunch : ℕ), meals_sold_lunch = 12 := by
  sorry

end NUMINAMATH_GPT_chef_sold_12_meals_l859_85980


namespace NUMINAMATH_GPT_no_simultaneous_negative_values_l859_85979

theorem no_simultaneous_negative_values (m n : ℝ) :
  ¬ ((3*m^2 + 4*m*n - 2*n^2 < 0) ∧ (-m^2 - 4*m*n + 3*n^2 < 0)) :=
by
  sorry

end NUMINAMATH_GPT_no_simultaneous_negative_values_l859_85979


namespace NUMINAMATH_GPT_james_total_time_l859_85985

def time_to_play_main_game : ℕ := 
  let download_time := 10
  let install_time := download_time / 2
  let update_time := download_time * 2
  let account_time := 5
  let internet_issues_time := 15
  let before_tutorial_time := download_time + install_time + update_time + account_time + internet_issues_time
  let tutorial_time := before_tutorial_time * 3
  before_tutorial_time + tutorial_time

theorem james_total_time : time_to_play_main_game = 220 := by
  sorry

end NUMINAMATH_GPT_james_total_time_l859_85985


namespace NUMINAMATH_GPT_moles_of_C2H6_are_1_l859_85933

def moles_of_C2H6_reacted (n_C2H6: ℕ) (n_Cl2: ℕ) (n_C2Cl6: ℕ): Prop :=
  n_Cl2 = 6 ∧ n_C2Cl6 = 1 ∧ (n_C2H6 + 6 * (n_Cl2 - 1) = n_C2Cl6 + 6 * (n_Cl2 - 1))

theorem moles_of_C2H6_are_1:
  ∀ (n_C2H6 n_Cl2 n_C2Cl6: ℕ), moles_of_C2H6_reacted n_C2H6 n_Cl2 n_C2Cl6 → n_C2H6 = 1 :=
by
  intros n_C2H6 n_Cl2 n_C2Cl6 h
  sorry

end NUMINAMATH_GPT_moles_of_C2H6_are_1_l859_85933


namespace NUMINAMATH_GPT_third_roll_six_probability_l859_85954

noncomputable def Die_A_six_prob : ℚ := 1 / 6
noncomputable def Die_B_six_prob : ℚ := 1 / 2
noncomputable def Die_C_one_prob : ℚ := 3 / 5
noncomputable def Die_B_not_six_prob : ℚ := 1 / 10
noncomputable def Die_C_not_one_prob : ℚ := 1 / 15

noncomputable def prob_two_sixes_die_A : ℚ := Die_A_six_prob ^ 2
noncomputable def prob_two_sixes_die_B : ℚ := Die_B_six_prob ^ 2
noncomputable def prob_two_sixes_die_C : ℚ := Die_C_not_one_prob ^ 2

noncomputable def total_prob_two_sixes : ℚ := 
  (1 / 3) * (prob_two_sixes_die_A + prob_two_sixes_die_B + prob_two_sixes_die_C)

noncomputable def cond_prob_die_A_given_two_sixes : ℚ := prob_two_sixes_die_A / total_prob_two_sixes
noncomputable def cond_prob_die_B_given_two_sixes : ℚ := prob_two_sixes_die_B / total_prob_two_sixes
noncomputable def cond_prob_die_C_given_two_sixes : ℚ := prob_two_sixes_die_C / total_prob_two_sixes

noncomputable def prob_third_six : ℚ := 
  cond_prob_die_A_given_two_sixes * Die_A_six_prob + 
  cond_prob_die_B_given_two_sixes * Die_B_six_prob + 
  cond_prob_die_C_given_two_sixes * Die_C_not_one_prob

theorem third_roll_six_probability : 
  prob_third_six = sorry := 
  sorry

end NUMINAMATH_GPT_third_roll_six_probability_l859_85954


namespace NUMINAMATH_GPT_no_empty_boxes_prob_l859_85968

def P (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem no_empty_boxes_prob :
  let num_balls := 3
  let num_boxes := 3
  let total_outcomes := num_boxes ^ num_balls
  let favorable_outcomes := P num_balls num_boxes
  let probability := favorable_outcomes / total_outcomes
  probability = 2 / 9 :=
by
  sorry

end NUMINAMATH_GPT_no_empty_boxes_prob_l859_85968


namespace NUMINAMATH_GPT_angle_C_45_l859_85932

theorem angle_C_45 (A B C : ℝ) 
(h : (Real.cos A + Real.sin A) * (Real.cos B + Real.sin B) = 2) 
(HA : 0 ≤ A) (HB : 0 ≤ B) (HC : 0 ≤ C):
A + B + C = π → 
A = B →
C = π / 2 - B →
C = π / 4 := 
by
  intros;
  sorry

end NUMINAMATH_GPT_angle_C_45_l859_85932


namespace NUMINAMATH_GPT_smallest_tournament_with_ordered_group_l859_85937

-- Define the concept of a tennis tournament with n players
def tennis_tournament (n : ℕ) := 
  ∀ (i j : ℕ), (i < n) → (j < n) → (i ≠ j) → (i < j) ∨ (j < i)

-- Define what it means for a group of four players to be "ordered"
def ordered_group (p1 p2 p3 p4 : ℕ) : Prop := 
  ∃ (winner : ℕ), ∃ (loser : ℕ), 
    (winner ≠ loser) ∧ (winner = p1 ∨ winner = p2 ∨ winner = p3 ∨ winner = p4) ∧ 
    (loser = p1 ∨ loser = p2 ∨ loser = p3 ∨ loser = p4)

-- Prove that any tennis tournament with 8 players has an ordered group
theorem smallest_tournament_with_ordered_group : 
  ∀ (n : ℕ), ∀ (tournament : tennis_tournament n), 
    (n ≥ 8) → 
    (∃ (p1 p2 p3 p4 : ℕ), ordered_group p1 p2 p3 p4) :=
  by
  -- proof omitted
  sorry

end NUMINAMATH_GPT_smallest_tournament_with_ordered_group_l859_85937


namespace NUMINAMATH_GPT_min_value_of_quadratic_l859_85935

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 2000 ≤ 3 * y^2 - 18 * y + 2000) ∧ (3 * x^2 - 18 * x + 2000 = 1973) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_quadratic_l859_85935


namespace NUMINAMATH_GPT_determine_a_minus_b_l859_85986

theorem determine_a_minus_b (a b : ℤ) 
  (h1 : 2009 * a + 2013 * b = 2021) 
  (h2 : 2011 * a + 2015 * b = 2023) : 
  a - b = -5 :=
sorry

end NUMINAMATH_GPT_determine_a_minus_b_l859_85986


namespace NUMINAMATH_GPT_smallest_number_is_42_l859_85981

theorem smallest_number_is_42 (x : ℤ) 
  (h1 : x + (x + 1) + (x + 2) + (x + 3) + (x + 4) = 225)
  (h2 : x % 7 = 0) : 
  x = 42 := 
sorry

end NUMINAMATH_GPT_smallest_number_is_42_l859_85981


namespace NUMINAMATH_GPT_red_black_ball_ratio_l859_85957

theorem red_black_ball_ratio (R B x : ℕ) (h1 : 3 * R = B + x) (h2 : 2 * R + x = B) :
  R / B = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_red_black_ball_ratio_l859_85957


namespace NUMINAMATH_GPT_azalea_paid_shearer_l859_85953

noncomputable def amount_paid_to_shearer (number_of_sheep wool_per_sheep price_per_pound profit : ℕ) : ℕ :=
  let total_wool := number_of_sheep * wool_per_sheep
  let total_revenue := total_wool * price_per_pound
  total_revenue - profit

theorem azalea_paid_shearer :
  let number_of_sheep := 200
  let wool_per_sheep := 10
  let price_per_pound := 20
  let profit := 38000
  amount_paid_to_shearer number_of_sheep wool_per_sheep price_per_pound profit = 2000 := 
by
  sorry

end NUMINAMATH_GPT_azalea_paid_shearer_l859_85953


namespace NUMINAMATH_GPT_range_of_a_l859_85950

def p (x : ℝ) : Prop := abs (2 * x - 1) ≤ 3

def q (x a : ℝ) : Prop := x^2 - (2*a + 1) * x + a*(a + 1) ≤ 0

theorem range_of_a : 
  (∀ x a, (¬ q x a) → (¬ p x))
  ∧ (∃ x a, (¬ q x a) ∧ (¬ p x))
  → (-1 : ℝ) ≤ a ∧ a ≤ (1 : ℝ) :=
sorry

end NUMINAMATH_GPT_range_of_a_l859_85950


namespace NUMINAMATH_GPT_constant_term_in_quadratic_eq_l859_85904

theorem constant_term_in_quadratic_eq : 
  ∀ (x : ℝ), (x^2 - 5 * x = 2) → (∃ a b c : ℝ, a = 1 ∧ a * x^2 + b * x + c = 0 ∧ c = -2) :=
by
  sorry

end NUMINAMATH_GPT_constant_term_in_quadratic_eq_l859_85904


namespace NUMINAMATH_GPT_minimum_value_of_quadratic_l859_85945

theorem minimum_value_of_quadratic : ∀ x : ℝ, (∃ y : ℝ, y = (x-2)^2 - 3) → ∃ m : ℝ, (∀ x : ℝ, (x-2)^2 - 3 ≥ m) ∧ m = -3 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_quadratic_l859_85945


namespace NUMINAMATH_GPT_george_reels_per_day_l859_85999

theorem george_reels_per_day
  (days : ℕ := 5)
  (jackson_per_day : ℕ := 6)
  (jonah_per_day : ℕ := 4)
  (total_fishes : ℕ := 90) :
  (∃ george_per_day : ℕ, george_per_day = 8) :=
by
  -- Calculation steps are skipped here; they would need to be filled in for a complete proof.
  sorry

end NUMINAMATH_GPT_george_reels_per_day_l859_85999


namespace NUMINAMATH_GPT_find_parabola_l859_85958

variable (P : ℝ × ℝ)
variable (a b : ℝ)

def parabola1 (P : ℝ × ℝ) (a : ℝ) := P.2^2 = 4 * a * P.1
def parabola2 (P : ℝ × ℝ) (b : ℝ) := P.1^2 = 4 * b * P.2

theorem find_parabola (hP : P = (-2, 4)) :
  (∃ a, parabola1 P a ∧ P.2^2 = -8 * P.1) ∨ 
  (∃ b, parabola2 P b ∧ P.1^2 = P.2) := by
  sorry

end NUMINAMATH_GPT_find_parabola_l859_85958


namespace NUMINAMATH_GPT_points_per_win_is_5_l859_85971

-- Definitions based on conditions
def rounds_played : ℕ := 30
def vlad_points : ℕ := 64
def taro_points (T : ℕ) : ℕ := (3 * T) / 5 - 4
def total_points (T : ℕ) : ℕ := taro_points T + vlad_points

-- Theorem statement to prove the number of points per win
theorem points_per_win_is_5 (T : ℕ) (H : total_points T = T) : T / rounds_played = 5 := sorry

end NUMINAMATH_GPT_points_per_win_is_5_l859_85971


namespace NUMINAMATH_GPT_infinite_squares_in_ap_l859_85983

theorem infinite_squares_in_ap
    (a d : ℤ)
    (h : ∃ n : ℤ, a^2 = a + n * d) :
    ∀ N : ℕ, ∃ m : ℤ, ∃ k : ℕ, k > N ∧ m^2 = a + k * d :=
by
  sorry

end NUMINAMATH_GPT_infinite_squares_in_ap_l859_85983


namespace NUMINAMATH_GPT_increase_by_40_percent_l859_85990

theorem increase_by_40_percent (initial_number : ℕ) (increase_rate : ℕ) :
  initial_number = 150 → increase_rate = 40 →
  initial_number + (increase_rate / 100 * initial_number) = 210 := by
  sorry

end NUMINAMATH_GPT_increase_by_40_percent_l859_85990


namespace NUMINAMATH_GPT_f_bounds_l859_85912

noncomputable def f (x1 x2 x3 x4 : ℝ) := 1 - (x1^3 + x2^3 + x3^3 + x4^3) - 6 * (x1 * x2 * x3 + x1 * x2 * x4 + x1 * x3 * x4 + x2 * x3 * x4)

theorem f_bounds (x1 x2 x3 x4 : ℝ) (h : x1 + x2 + x3 + x4 = 1) :
  0 < f x1 x2 x3 x4 ∧ f x1 x2 x3 x4 ≤ 3 / 4 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_f_bounds_l859_85912


namespace NUMINAMATH_GPT_fraction_sum_equals_l859_85928

theorem fraction_sum_equals :
  (1 / 20 : ℝ) + (2 / 10 : ℝ) + (4 / 40 : ℝ) = 0.35 :=
by
  sorry

end NUMINAMATH_GPT_fraction_sum_equals_l859_85928


namespace NUMINAMATH_GPT_parabola_tangents_intersection_y_coord_l859_85949

theorem parabola_tangents_intersection_y_coord
  (a b : ℝ)
  (ha : A = (a, a^2 + 1))
  (hb : B = (b, b^2 + 1))
  (tangent_perpendicular : ∀ t1 t2 : ℝ, t1 * t2 = -1):
  ∃ y : ℝ, y = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_tangents_intersection_y_coord_l859_85949


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l859_85916

theorem quadratic_inequality_solution (m : ℝ) :
    (∃ x : ℝ, x^2 - m * x + 1 ≤ 0) ↔ m ≥ 2 ∨ m ≤ -2 := by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l859_85916


namespace NUMINAMATH_GPT_relation_among_a_b_c_l859_85908

theorem relation_among_a_b_c
  (a : ℝ) (b : ℝ) (c : ℝ)
  (h1 : a = (3 / 5)^4)
  (h2 : b = (3 / 5)^3)
  (h3 : c = Real.log (3 / 5) / Real.log 3) :
  c < a ∧ a < b :=
by
  sorry

end NUMINAMATH_GPT_relation_among_a_b_c_l859_85908


namespace NUMINAMATH_GPT_parabola_equation_l859_85977

theorem parabola_equation (h_axis : ∃ p > 0, x = p / 2) :
  ∃ p > 0, y^2 = -2 * p * x :=
by 
  -- proof steps will be added here
  sorry

end NUMINAMATH_GPT_parabola_equation_l859_85977


namespace NUMINAMATH_GPT_mean_of_remaining_two_numbers_l859_85955

theorem mean_of_remaining_two_numbers :
  let n1 := 1871
  let n2 := 1997
  let n3 := 2023
  let n4 := 2029
  let n5 := 2113
  let n6 := 2125
  let n7 := 2137
  let total_sum := n1 + n2 + n3 + n4 + n5 + n6 + n7
  let known_mean := 2100
  let mean_of_other_two := 1397.5
  total_sum = 13295 →
  5 * known_mean = 10500 →
  total_sum - 10500 = 2795 →
  2795 / 2 = mean_of_other_two :=
by
  intros
  sorry

end NUMINAMATH_GPT_mean_of_remaining_two_numbers_l859_85955


namespace NUMINAMATH_GPT_average_weight_l859_85938

variable (A B C : ℝ) 

theorem average_weight (h1 : (A + B) / 2 = 48) (h2 : (B + C) / 2 = 42) (h3 : B = 51) :
  (A + B + C) / 3 = 43 := by
  sorry

end NUMINAMATH_GPT_average_weight_l859_85938


namespace NUMINAMATH_GPT_minimum_value_of_expression_l859_85948

theorem minimum_value_of_expression (a b : ℝ) (h : 1 / a + 2 / b = 1) : 4 * a^2 + b^2 ≥ 32 :=
by sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l859_85948


namespace NUMINAMATH_GPT_frac_eq_l859_85976

theorem frac_eq (x : ℝ) (h : 3 - 9 / x + 6 / x^2 = 0) : 2 / x = 1 ∨ 2 / x = 2 := 
by 
  sorry

end NUMINAMATH_GPT_frac_eq_l859_85976


namespace NUMINAMATH_GPT_jessica_quarters_l859_85923

theorem jessica_quarters (quarters_initial quarters_given : Nat) (h_initial : quarters_initial = 8) (h_given : quarters_given = 3) :
  quarters_initial + quarters_given = 11 := by
  sorry

end NUMINAMATH_GPT_jessica_quarters_l859_85923


namespace NUMINAMATH_GPT_income_before_taxes_l859_85966

/-- Define given conditions -/
def net_income (x : ℝ) : ℝ := x - 0.10 * (x - 3000)

/-- Prove that the income before taxes must have been 13000 given the conditions. -/
theorem income_before_taxes (x : ℝ) (hx : net_income x = 12000) : x = 13000 :=
by sorry

end NUMINAMATH_GPT_income_before_taxes_l859_85966


namespace NUMINAMATH_GPT_boat_distance_downstream_l859_85942

-- Definitions
def boat_speed_in_still_water : ℝ := 24
def stream_speed : ℝ := 4
def time_downstream : ℝ := 3

-- Effective speed downstream
def speed_downstream := boat_speed_in_still_water + stream_speed

-- Distance calculation
def distance_downstream := speed_downstream * time_downstream

-- Proof statement
theorem boat_distance_downstream : distance_downstream = 84 := 
by
  -- This is where the proof would go, but we use sorry for now
  sorry

end NUMINAMATH_GPT_boat_distance_downstream_l859_85942


namespace NUMINAMATH_GPT_simplify_and_evaluate_l859_85939

def my_expression (x : ℝ) := (x + 2) * (x - 2) + 3 * (1 - x)

theorem simplify_and_evaluate : 
  my_expression (Real.sqrt 2) = 1 - 3 * Real.sqrt 2 := by
    sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l859_85939


namespace NUMINAMATH_GPT_max_sum_of_radii_in_prism_l859_85975

noncomputable def sum_of_radii (AB AD AA1 : ℝ) : ℝ :=
  let r (t : ℝ) := 2 - 2 * t
  let R (t : ℝ) := 3 * t / (1 + t)
  let f (t : ℝ) := R t + r t
  let t_max := 1 / 2
  f t_max

theorem max_sum_of_radii_in_prism :
  let AB := 5
  let AD := 3
  let AA1 := 4
  sum_of_radii AB AD AA1 = 21 / 10 := by
sorry

end NUMINAMATH_GPT_max_sum_of_radii_in_prism_l859_85975


namespace NUMINAMATH_GPT_find_flour_amount_l859_85902

variables (F S C : ℕ)

-- Condition 1: Proportions must remain constant
axiom proportion : 11 * S = 7 * F ∧ 7 * C = 5 * S

-- Condition 2: Mary needs 2 more cups of flour than sugar
axiom flour_sugar : F = S + 2

-- Condition 3: Mary needs 1 more cup of sugar than cocoa powder
axiom sugar_cocoa : S = C + 1

-- Question: How many cups of flour did she put in?
theorem find_flour_amount : F = 8 :=
by
  sorry

end NUMINAMATH_GPT_find_flour_amount_l859_85902


namespace NUMINAMATH_GPT_xy_relationship_l859_85963

theorem xy_relationship (x y : ℤ) (h1 : 2 * x - y > x + 1) (h2 : x + 2 * y < 2 * y - 3) :
  x < -3 ∧ y < -4 ∧ x > y + 1 :=
sorry

end NUMINAMATH_GPT_xy_relationship_l859_85963


namespace NUMINAMATH_GPT_discount_is_100_l859_85998

-- Define the constants for the problem conditions
def suit_cost : ℕ := 430
def shoes_cost : ℕ := 190
def amount_paid : ℕ := 520

-- Total cost before discount
def total_cost_before_discount (a b : ℕ) : ℕ := a + b

-- Discount amount
def discount_amount (total paid : ℕ) : ℕ := total - paid

-- Main theorem statement
theorem discount_is_100 : discount_amount (total_cost_before_discount suit_cost shoes_cost) amount_paid = 100 := 
by
sorry

end NUMINAMATH_GPT_discount_is_100_l859_85998


namespace NUMINAMATH_GPT_correct_average_marks_l859_85903

theorem correct_average_marks 
  (n : ℕ) (average initial_wrong new_correct : ℕ) 
  (h_num_students : n = 30)
  (h_average_marks : average = 100)
  (h_initial_wrong : initial_wrong = 70)
  (h_new_correct : new_correct = 10) :
  (average * n - (initial_wrong - new_correct)) / n = 98 := 
by
  sorry

end NUMINAMATH_GPT_correct_average_marks_l859_85903


namespace NUMINAMATH_GPT_probability_class_4_drawn_first_second_l859_85924

noncomputable def P_1 : ℝ := 1 / 10
noncomputable def P_2 : ℝ := 9 / 100

theorem probability_class_4_drawn_first_second :
  P_1 = 1 / 10 ∧ P_2 = 9 / 100 := by
  sorry

end NUMINAMATH_GPT_probability_class_4_drawn_first_second_l859_85924


namespace NUMINAMATH_GPT_ladder_base_distance_l859_85967

theorem ladder_base_distance
  (c : ℕ) (b : ℕ) (hypotenuse : c = 13) (wall_height : b = 12) :
  ∃ x : ℕ, x^2 + b^2 = c^2 ∧ x = 5 := by
  sorry

end NUMINAMATH_GPT_ladder_base_distance_l859_85967


namespace NUMINAMATH_GPT_trigonometric_signs_l859_85919

noncomputable def terminal_side (θ α : ℝ) : Prop :=
  ∃ k : ℤ, θ = α + 2 * k * Real.pi

theorem trigonometric_signs :
  ∀ (α θ : ℝ), 
    (∃ k : ℤ, α = 2 * k * Real.pi - Real.pi / 5) ∧ terminal_side θ α →
    (Real.sin θ < 0) ∧ (Real.cos θ > 0) ∧ (Real.tan θ < 0) →
    (Real.sin θ / abs (Real.sin θ) + Real.cos θ / abs (Real.cos θ) + Real.tan θ / abs (Real.tan θ) = -1) :=
by intros
   sorry

end NUMINAMATH_GPT_trigonometric_signs_l859_85919


namespace NUMINAMATH_GPT_largest_circle_area_l859_85931

theorem largest_circle_area (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 200) :
  ∃ r : ℝ, (2 * π * r = 60) ∧ (π * r ^ 2 = 900 / π) := 
sorry

end NUMINAMATH_GPT_largest_circle_area_l859_85931


namespace NUMINAMATH_GPT_total_price_of_basic_computer_and_printer_l859_85921

-- Definitions for the conditions
def basic_computer_price := 2000
def enhanced_computer_price (C : ℕ) := C + 500
def printer_price (C : ℕ) (P : ℕ) := 1/6 * (C + 500 + P)

-- The proof problem statement
theorem total_price_of_basic_computer_and_printer (C P : ℕ) 
  (h1 : C = 2000)
  (h2 : printer_price C P = P) : 
  C + P = 2500 :=
sorry

end NUMINAMATH_GPT_total_price_of_basic_computer_and_printer_l859_85921


namespace NUMINAMATH_GPT_repeating_decimal_fraction_l859_85964

theorem repeating_decimal_fraction : (0.363636363636 : ℚ) = 4 / 11 := 
sorry

end NUMINAMATH_GPT_repeating_decimal_fraction_l859_85964


namespace NUMINAMATH_GPT_value_of_nested_custom_div_l859_85969

def custom_div (x y z : ℕ) (hz : z ≠ 0) : ℕ :=
  (x + y) / z

theorem value_of_nested_custom_div : custom_div (custom_div 45 15 60 (by decide)) (custom_div 3 3 6 (by decide)) (custom_div 20 10 30 (by decide)) (by decide) = 2 :=
sorry

end NUMINAMATH_GPT_value_of_nested_custom_div_l859_85969


namespace NUMINAMATH_GPT_cost_difference_l859_85974

def TMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 50
  let additional_line_cost := 16
  let discount := 0.1
  let data_charge := 3
  let monthly_cost_before_discount := base_cost + (additional_line_cost * (num_lines - 2))
  let total_monthly_cost := monthly_cost_before_discount + (data_charge * num_lines)
  (total_monthly_cost * (1 - discount)) * 12

def MMobile_cost (num_lines : ℕ) : ℝ :=
  let base_cost := 45
  let additional_line_cost := 14
  let activation_fee := 20
  let monthly_cost := base_cost + (additional_line_cost * (num_lines - 2))
  (monthly_cost * 12) + (activation_fee * num_lines)

theorem cost_difference (num_lines : ℕ) (h : num_lines = 5) :
  TMobile_cost num_lines - MMobile_cost num_lines = 76.40 :=
  sorry

end NUMINAMATH_GPT_cost_difference_l859_85974


namespace NUMINAMATH_GPT_find_function_f_l859_85944

-- Define the problem in Lean 4
theorem find_function_f (f : ℝ → ℝ) : 
  (f 0 = 1) → 
  ((∀ x y : ℝ, f (x * y + 1) = f x * f y - f y - x + 2)) → 
  (∀ x : ℝ, f x = x + 1) :=
  by
    intros h₁ h₂
    sorry

end NUMINAMATH_GPT_find_function_f_l859_85944


namespace NUMINAMATH_GPT_proof_sum_of_ab_l859_85946

theorem proof_sum_of_ab :
  ∃ (a b : ℕ), a ≤ b ∧ 0 < a ∧ 0 < b ∧ a ^ 2 + b ^ 2 + 8 * a * b = 2010 ∧ a + b = 42 :=
sorry

end NUMINAMATH_GPT_proof_sum_of_ab_l859_85946


namespace NUMINAMATH_GPT_cost_of_adult_ticket_is_10_l859_85927

-- Definitions based on the problem's conditions
def num_adults : ℕ := 5
def num_children : ℕ := 2
def cost_concessions : ℝ := 12
def total_cost : ℝ := 76
def cost_child_ticket : ℝ := 7

-- Statement to prove the cost of an adult ticket being $10
theorem cost_of_adult_ticket_is_10 :
  ∃ A : ℝ, (num_adults * A + num_children * cost_child_ticket + cost_concessions = total_cost) ∧ A = 10 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_adult_ticket_is_10_l859_85927


namespace NUMINAMATH_GPT_new_students_count_l859_85947

-- Define the conditions as given in the problem statement.
def original_average_age := 40
def original_number_students := 17
def new_students_average_age := 32
def decreased_age := 36  -- Since the average decreases by 4 years from 40 to 36

-- Let x be the number of new students, the proof problem is to find x.
def find_new_students (x : ℕ) : Prop :=
  original_average_age * original_number_students + new_students_average_age * x = decreased_age * (original_number_students + x)

-- Prove that find_new_students(x) holds for x = 17
theorem new_students_count : find_new_students 17 :=
by
  sorry -- the proof goes here

end NUMINAMATH_GPT_new_students_count_l859_85947


namespace NUMINAMATH_GPT_max_friendly_groups_19_max_friendly_groups_20_l859_85943

def friendly_group {Team : Type} (beat : Team → Team → Prop) (A B C : Team) : Prop :=
  beat A B ∧ beat B C ∧ beat C A

def max_friendly_groups_19_teams : ℕ := 285
def max_friendly_groups_20_teams : ℕ := 330

theorem max_friendly_groups_19 {Team : Type} (n : ℕ) (h : n = 19) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_19_teams := sorry

theorem max_friendly_groups_20 {Team : Type} (n : ℕ) (h : n = 20) (beat : Team → Team → Prop) :
  ∃ (G : ℕ), G = max_friendly_groups_20_teams := sorry

end NUMINAMATH_GPT_max_friendly_groups_19_max_friendly_groups_20_l859_85943


namespace NUMINAMATH_GPT_rosa_calls_pages_l859_85962

theorem rosa_calls_pages (pages_last_week : ℝ) (pages_this_week : ℝ) (h_last_week : pages_last_week = 10.2) (h_this_week : pages_this_week = 8.6) : pages_last_week + pages_this_week = 18.8 :=
by sorry

end NUMINAMATH_GPT_rosa_calls_pages_l859_85962


namespace NUMINAMATH_GPT_find_missing_number_l859_85900

theorem find_missing_number
  (a b c d e : ℝ) (mean : ℝ) (f : ℝ)
  (h1 : a = 13) 
  (h2 : b = 8)
  (h3 : c = 13)
  (h4 : d = 7)
  (h5 : e = 23)
  (hmean : mean = 14.2) :
  (a + b + c + d + e + f) / 6 = mean → f = 21.2 :=
by
  sorry

end NUMINAMATH_GPT_find_missing_number_l859_85900


namespace NUMINAMATH_GPT_ball_hits_ground_l859_85914

theorem ball_hits_ground : 
  ∃ t : ℚ, -4.9 * t^2 + 4 * t + 10 = 0 ∧ t = 10 / 7 :=
by sorry

end NUMINAMATH_GPT_ball_hits_ground_l859_85914


namespace NUMINAMATH_GPT_problem1_problem2_problem3_l859_85988

-- Problem 1
theorem problem1 (x : ℝ) (h : 0 < x ∧ x < 1/2) : 
  (1/2 * x * (1 - 2 * x) ≤ 1/16) := sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : 0 < x) : 
  (2 - x - 4 / x ≤ -2) := sorry

-- Problem 3
theorem problem3 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 4) : 
  (1 / x + 3 / y ≥ 1 + Real.sqrt 3 / 2) := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_l859_85988


namespace NUMINAMATH_GPT_molecular_weight_CO_l859_85940

theorem molecular_weight_CO :
  let atomic_weight_C := 12.01
  let atomic_weight_O := 16.00
  let molecular_weight := atomic_weight_C + atomic_weight_O
  molecular_weight = 28.01 := 
by
  sorry

end NUMINAMATH_GPT_molecular_weight_CO_l859_85940


namespace NUMINAMATH_GPT_fraction_of_white_roses_l859_85992

open Nat

def rows : ℕ := 10
def roses_per_row : ℕ := 20
def total_roses : ℕ := rows * roses_per_row
def red_roses : ℕ := total_roses / 2
def pink_roses : ℕ := 40
def white_roses : ℕ := total_roses - red_roses - pink_roses
def remaining_roses : ℕ := white_roses + pink_roses
def fraction_white_roses : ℚ := white_roses / remaining_roses

theorem fraction_of_white_roses :
  fraction_white_roses = 3 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_white_roses_l859_85992


namespace NUMINAMATH_GPT_math_problem_l859_85909

/-- Given a function definition f(x) = 2 * x * f''(1) + x^2,
    Prove that the second derivative f''(0) is equal to -4. -/
theorem math_problem (f : ℝ → ℝ) (h1 : ∀ x, f x = 2 * x * (deriv^[2] (f) 1) + x^2) :
  (deriv^[2] f) 0 = -4 :=
  sorry

end NUMINAMATH_GPT_math_problem_l859_85909


namespace NUMINAMATH_GPT_N_subseteq_M_l859_85930

/--
Let M = { x | ∃ n ∈ ℤ, x = n / 2 + 1 } and
N = { y | ∃ m ∈ ℤ, y = m + 0.5 }.
Prove that N is a subset of M.
-/
theorem N_subseteq_M : 
  let M := { x : ℝ | ∃ n : ℤ, x = n / 2 + 1 }
  let N := { y : ℝ | ∃ m : ℤ, y = m + 0.5 }
  N ⊆ M := sorry

end NUMINAMATH_GPT_N_subseteq_M_l859_85930


namespace NUMINAMATH_GPT_alcohol_quantity_in_mixture_l859_85951

theorem alcohol_quantity_in_mixture 
  (A W : ℝ)
  (h1 : A / W = 4 / 3)
  (h2 : A / (W + 4) = 4 / 5)
  : A = 8 :=
sorry

end NUMINAMATH_GPT_alcohol_quantity_in_mixture_l859_85951


namespace NUMINAMATH_GPT_find_SSE_l859_85995

theorem find_SSE (SST SSR : ℝ) (h1 : SST = 13) (h2 : SSR = 10) : SST - SSR = 3 :=
by
  sorry

end NUMINAMATH_GPT_find_SSE_l859_85995


namespace NUMINAMATH_GPT_part_a_l859_85972

theorem part_a (x α : ℝ) (hα : 0 < α ∧ α < 1) (hx : x ≥ 0) : x^α - α * x ≤ 1 - α :=
sorry

end NUMINAMATH_GPT_part_a_l859_85972


namespace NUMINAMATH_GPT_intersection_of_sets_l859_85984

-- Conditions as Lean definitions
def A : Set Int := {-2, -1}
def B : Set Int := {-1, 2, 3}

-- Stating the proof problem in Lean 4
theorem intersection_of_sets : A ∩ B = {-1} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_sets_l859_85984


namespace NUMINAMATH_GPT_handshake_problem_7_boys_21_l859_85911

theorem handshake_problem_7_boys_21 :
  let n := 7
  let total_handshakes := n * (n - 1) / 2
  total_handshakes = 21 → (n - 1) = 6 :=
by
  -- Let n be the number of boys (7 in this case)
  let n := 7
  
  -- Define the total number of handshakes equation
  let total_handshakes := n * (n - 1) / 2
  
  -- Assume the total number of handshakes is 21
  intro h
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_handshake_problem_7_boys_21_l859_85911


namespace NUMINAMATH_GPT_train_B_speed_l859_85929

-- Given conditions
def speed_train_A := 70 -- km/h
def time_after_meet_A := 9 -- hours
def time_after_meet_B := 4 -- hours

-- Proof statement
theorem train_B_speed : 
  ∃ (V_b : ℕ),
    V_b * time_after_meet_B + V_b * s = speed_train_A * time_after_meet_A + speed_train_A * s ∧
    V_b = speed_train_A := 
sorry

end NUMINAMATH_GPT_train_B_speed_l859_85929


namespace NUMINAMATH_GPT_distance_covered_by_center_of_circle_l859_85901

-- Definition of the sides of the triangle
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Definition of the circle's radius
def radius : ℕ := 2

-- Define a function that calculates the perimeter of the smaller triangle
noncomputable def smallerTrianglePerimeter (s1 s2 hyp r : ℕ) : ℕ :=
  (s1 - 2 * r) + (s2 - 2 * r) + (hyp - 2 * r)

-- Main theorem statement
theorem distance_covered_by_center_of_circle :
  smallerTrianglePerimeter side1 side2 hypotenuse radius = 18 :=
by
  sorry

end NUMINAMATH_GPT_distance_covered_by_center_of_circle_l859_85901


namespace NUMINAMATH_GPT_evaluate_expression_correct_l859_85906

noncomputable def evaluate_expression : ℤ :=
  6 - 8 * (9 - 4 ^ 2) * 5 + 2

theorem evaluate_expression_correct : evaluate_expression = 288 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_correct_l859_85906


namespace NUMINAMATH_GPT_find_f_2023_l859_85936

def is_odd_function (g : ℝ → ℝ) := ∀ x, g x = -g (-x)

def condition1 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)

def condition2 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (1 - x) = f (3 + x)

theorem find_f_2023 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, (f (x + 1)) - 2 = -((f (1 - x)) - 2)) 
  (h2 : ∀ x : ℝ, f (1 - x) = f (3 + x)) : 
  f 2023 = 2 :=
sorry

end NUMINAMATH_GPT_find_f_2023_l859_85936


namespace NUMINAMATH_GPT_compute_expression_l859_85910

-- Given Conditions
variables (a b c : ℕ)
variable (h : 2^a * 3^b * 5^c = 36000)

-- Proof Statement
theorem compute_expression (h : 2^a * 3^b * 5^c = 36000) : 3 * a + 4 * b + 6 * c = 41 :=
sorry

end NUMINAMATH_GPT_compute_expression_l859_85910


namespace NUMINAMATH_GPT_total_assembly_time_l859_85913

def chairs := 2
def tables := 2
def bookshelf := 1
def tv_stand := 1

def time_per_chair := 8
def time_per_table := 12
def time_per_bookshelf := 25
def time_per_tv_stand := 35

theorem total_assembly_time : (chairs * time_per_chair) + (tables * time_per_table) + (bookshelf * time_per_bookshelf) + (tv_stand * time_per_tv_stand) = 100 := by
  sorry

end NUMINAMATH_GPT_total_assembly_time_l859_85913


namespace NUMINAMATH_GPT_find_a_for_parallel_lines_l859_85917

def direction_vector_1 (a : ℝ) : ℝ × ℝ × ℝ :=
  (2 * a, 3, 2)

def direction_vector_2 : ℝ × ℝ × ℝ :=
  (2, 3, 2)

theorem find_a_for_parallel_lines : ∃ a : ℝ, direction_vector_1 a = direction_vector_2 :=
by
  use 1
  unfold direction_vector_1
  sorry  -- proof omitted

end NUMINAMATH_GPT_find_a_for_parallel_lines_l859_85917


namespace NUMINAMATH_GPT_base_conversion_subtraction_l859_85973

def base6_to_base10 (n : Nat) : Nat :=
  n / 100000 * 6^5 +
  (n / 10000 % 10) * 6^4 +
  (n / 1000 % 10) * 6^3 +
  (n / 100 % 10) * 6^2 +
  (n / 10 % 10) * 6^1 +
  (n % 10) * 6^0

def base7_to_base10 (n : Nat) : Nat :=
  n / 10000 * 7^4 +
  (n / 1000 % 10) * 7^3 +
  (n / 100 % 10) * 7^2 +
  (n / 10 % 10) * 7^1 +
  (n % 10) * 7^0

theorem base_conversion_subtraction :
  base6_to_base10 543210 - base7_to_base10 43210 = 34052 := by
  sorry

end NUMINAMATH_GPT_base_conversion_subtraction_l859_85973


namespace NUMINAMATH_GPT_notebooks_per_child_if_half_l859_85926

theorem notebooks_per_child_if_half (C N : ℕ) 
    (h1 : N = C / 8) 
    (h2 : C * N = 512) : 
    512 / (C / 2) = 16 :=
by
    sorry

end NUMINAMATH_GPT_notebooks_per_child_if_half_l859_85926


namespace NUMINAMATH_GPT_downstream_speed_l859_85961

-- Define the given conditions as constants
def V_u : ℝ := 25 -- upstream speed in kmph
def V_m : ℝ := 40 -- speed of the man in still water in kmph

-- Define the speed of the stream
def V_s := V_m - V_u

-- Define the downstream speed
def V_d := V_m + V_s

-- Assertion we need to prove
theorem downstream_speed : V_d = 55 := by
  sorry

end NUMINAMATH_GPT_downstream_speed_l859_85961


namespace NUMINAMATH_GPT_medal_winners_combinations_l859_85997

theorem medal_winners_combinations:
  ∀ n k : ℕ, (n = 6) → (k = 3) → (n.choose k = 20) :=
by
  intros n k hn hk
  simp [hn, hk]
  -- We can continue the proof using additional math concepts if necessary.
  sorry

end NUMINAMATH_GPT_medal_winners_combinations_l859_85997


namespace NUMINAMATH_GPT_amount_invested_l859_85994

theorem amount_invested (P : ℝ) :
  P * (1.03)^2 - P = 0.08 * P + 6 → P = 314.136 := by
  sorry

end NUMINAMATH_GPT_amount_invested_l859_85994


namespace NUMINAMATH_GPT_camila_weeks_to_goal_l859_85956

open Nat

noncomputable def camila_hikes : ℕ := 7
noncomputable def amanda_hikes : ℕ := 8 * camila_hikes
noncomputable def steven_hikes : ℕ := amanda_hikes + 15
noncomputable def additional_hikes_needed : ℕ := steven_hikes - camila_hikes
noncomputable def hikes_per_week : ℕ := 4
noncomputable def weeks_to_goal : ℕ := additional_hikes_needed / hikes_per_week

theorem camila_weeks_to_goal : weeks_to_goal = 16 :=
  by sorry

end NUMINAMATH_GPT_camila_weeks_to_goal_l859_85956


namespace NUMINAMATH_GPT_circle_radius_eq_one_l859_85993

theorem circle_radius_eq_one (x y : ℝ) : (16 * x^2 - 32 * x + 16 * y^2 + 64 * y + 64 = 0) → (1 = 1) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_circle_radius_eq_one_l859_85993


namespace NUMINAMATH_GPT_quadratic_even_coeff_l859_85987

theorem quadratic_even_coeff (a b c : ℤ) (h : a ≠ 0) (hq : ∃ x : ℚ, a * x^2 + b * x + c = 0) : ¬ (∀ x : ℤ, (x ≠ 0 → (x % 2 = 1))) := 
sorry

end NUMINAMATH_GPT_quadratic_even_coeff_l859_85987


namespace NUMINAMATH_GPT_ratio_of_water_level_increase_l859_85925

noncomputable def volume_narrow_cone (h₁ : ℝ) : ℝ := (16 / 3) * Real.pi * h₁
noncomputable def volume_wide_cone (h₂ : ℝ) : ℝ := (64 / 3) * Real.pi * h₂
noncomputable def volume_marble_narrow : ℝ := (32 / 3) * Real.pi
noncomputable def volume_marble_wide : ℝ := (4 / 3) * Real.pi

theorem ratio_of_water_level_increase :
  ∀ (h₁ h₂ h₁' h₂' : ℝ),
  h₁ = 4 * h₂ →
  h₁' = h₁ + 2 →
  h₂' = h₂ + (1 / 16) →
  volume_narrow_cone h₁ = volume_wide_cone h₂ →
  volume_narrow_cone h₁ + volume_marble_narrow = volume_narrow_cone h₁' →
  volume_wide_cone h₂ + volume_marble_wide = volume_wide_cone h₂' →
  (h₁' - h₁) / (h₂' - h₂) = 32 :=
by
  intros h₁ h₂ h₁' h₂' h₁_eq_4h₂ h₁'_eq_h₁_add_2 h₂'_eq_h₂_add_1_div_16 vol_h₁_eq_vol_h₂ vol_nar_eq vol_wid_eq
  sorry

end NUMINAMATH_GPT_ratio_of_water_level_increase_l859_85925


namespace NUMINAMATH_GPT_rs_value_l859_85991

theorem rs_value (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 2) (h4 : r^4 + s^4 = 15 / 8) :
  r * s = (Real.sqrt 17) / 4 := 
sorry

end NUMINAMATH_GPT_rs_value_l859_85991


namespace NUMINAMATH_GPT_min_correct_answers_l859_85934

/-- 
Given:
1. There are 25 questions in the preliminary round.
2. Scoring rules: 
   - 4 points for each correct answer,
   - -1 point for each incorrect or unanswered question.
3. A score of at least 60 points is required to advance to the next round.

Prove that the minimum number of correct answers needed to advance is 17.
-/
theorem min_correct_answers (x : ℕ) (h1 : 0 ≤ x) (h2 : x ≤ 25) (h3 : 4 * x - (25 - x) ≥ 60) : x ≥ 17 :=
sorry

end NUMINAMATH_GPT_min_correct_answers_l859_85934
