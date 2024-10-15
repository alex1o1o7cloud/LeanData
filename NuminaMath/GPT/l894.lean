import Mathlib

namespace NUMINAMATH_GPT_nth_number_in_S_l894_89459

def S : Set ℕ := {n | ∃ k : ℕ, n = 15 * k + 11}

theorem nth_number_in_S (n : ℕ) (hn : n = 127) : ∃ k, 15 * k + 11 = 1901 :=
by
  sorry

end NUMINAMATH_GPT_nth_number_in_S_l894_89459


namespace NUMINAMATH_GPT_intersection_eq_one_l894_89485

def M : Set ℕ := {0, 1}
def N : Set ℕ := {y | ∃ x ∈ M, y = x^2 + 1}

theorem intersection_eq_one : M ∩ N = {1} := 
by
  sorry

end NUMINAMATH_GPT_intersection_eq_one_l894_89485


namespace NUMINAMATH_GPT_track_width_l894_89436

theorem track_width (r1 r2 : ℝ) (h : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi) : r1 - r2 = 10 := by
  sorry

end NUMINAMATH_GPT_track_width_l894_89436


namespace NUMINAMATH_GPT_units_digit_of_m_squared_plus_3_to_the_m_l894_89428

def m : ℕ := 2021^3 + 3^2021

theorem units_digit_of_m_squared_plus_3_to_the_m 
  (hm : m = 2021^3 + 3^2021) : 
  ((m^2 + 3^m) % 10) = 7 := 
by 
  -- Here you would input the proof steps, however, we skip it now with sorry.
  sorry

end NUMINAMATH_GPT_units_digit_of_m_squared_plus_3_to_the_m_l894_89428


namespace NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_squares_l894_89404

def a : ℕ := 35
def b : ℕ := 84

theorem largest_prime_divisor_of_sum_of_squares : 
  ∃ p : ℕ, Prime p ∧ p = 13 ∧ (a^2 + b^2) % p = 0 := by
  sorry

end NUMINAMATH_GPT_largest_prime_divisor_of_sum_of_squares_l894_89404


namespace NUMINAMATH_GPT_smallest_two_digit_multiple_of_3_l894_89458

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n <= 99
def is_multiple_of_3 (n : ℕ) : Prop := ∃ k : ℕ, n = 3 * k

theorem smallest_two_digit_multiple_of_3 : ∃ n : ℕ, is_two_digit n ∧ is_multiple_of_3 n ∧ ∀ m : ℕ, is_two_digit m ∧ is_multiple_of_3 m → n <= m :=
sorry

end NUMINAMATH_GPT_smallest_two_digit_multiple_of_3_l894_89458


namespace NUMINAMATH_GPT_solve_for_b_l894_89464

theorem solve_for_b (a b : ℤ) (h1 : 3 * a + 2 = 5) (h2 : b - 4 * a = 2) : b = 6 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_solve_for_b_l894_89464


namespace NUMINAMATH_GPT_max_statements_true_l894_89456

theorem max_statements_true : ∃ x : ℝ, 
  (0 < x^2 ∧ x^2 < 1 ∨ x^2 > 1) ∧ 
  (-1 < x ∧ x < 0 ∨ 0 < x ∧ x < 1) ∧ 
  (0 < (x - x^3) ∧ (x - x^3) < 1) :=
  sorry

end NUMINAMATH_GPT_max_statements_true_l894_89456


namespace NUMINAMATH_GPT_solve_system_l894_89416

theorem solve_system:
  ∃ (x y : ℝ), (26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧ 10 * x^2 + 18 * x * y + 8 * y^2 = 6) ↔
  (x = -1 ∧ y = 2) ∨ (x = -11 ∧ y = 14) ∨ (x = 11 ∧ y = -14) ∨ (x = 1 ∧ y = -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l894_89416


namespace NUMINAMATH_GPT_sum_of_reciprocals_l894_89415

-- We state that for all non-zero real numbers x and y, if x + y = xy,
-- then the sum of their reciprocals equals 1.
theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y) :
  1/x + 1/y = 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_l894_89415


namespace NUMINAMATH_GPT_miller_rabin_probability_at_least_half_l894_89434

theorem miller_rabin_probability_at_least_half
  {n : ℕ} (hcomp : ¬Nat.Prime n) (s d : ℕ) (hd_odd : d % 2 = 1) (h_decomp : n - 1 = 2^s * d)
  (a : ℤ) (ha_range : 2 ≤ a ∧ a ≤ n - 2) :
  ∃ P : ℝ, P ≥ 1 / 2 ∧ ∀ a, (2 ≤ a ∧ a ≤ n - 2) → ¬(a^(d * 2^s) % n = 1)
  :=
sorry

end NUMINAMATH_GPT_miller_rabin_probability_at_least_half_l894_89434


namespace NUMINAMATH_GPT_player_A_wins_if_n_equals_9_l894_89411

-- Define the conditions
def drawing_game (n : ℕ) : Prop :=
  ∃ strategy : ℕ → ℕ,
    strategy 0 = 1 ∧ -- Player A always starts by drawing 1 ball
    (∀ k, 1 ≤ strategy k ∧ strategy k ≤ 3) ∧ -- Players draw between 1 and 3 balls
    ∀ b, 1 ≤ b → b ≤ 3 → (n - 1 - strategy (b - 1)) ≤ 3 → (strategy (n - 1 - (b - 1)) = n - (b - 1) - 1)

-- State the problem to prove Player A has a winning strategy if n = 9
theorem player_A_wins_if_n_equals_9 : drawing_game 9 :=
sorry

end NUMINAMATH_GPT_player_A_wins_if_n_equals_9_l894_89411


namespace NUMINAMATH_GPT_inequality_solution_l894_89473

theorem inequality_solution (x : ℝ) :
  (6 * (x ^ 3 - 8) * (Real.sqrt (x ^ 2 + 6 * x + 9)) / ((x ^ 2 + 2 * x + 4) * (x ^ 2 + x - 6)) ≥ x - 2) ↔
  (x ∈ Set.Iic (-4) ∪ Set.Ioo (-3) 2 ∪ Set.Ioo 2 8) := sorry

end NUMINAMATH_GPT_inequality_solution_l894_89473


namespace NUMINAMATH_GPT_more_tvs_sold_l894_89471

variable (T x : ℕ)

theorem more_tvs_sold (h1 : T + x = 327) (h2 : T + 3 * x = 477) : x = 75 := by
  sorry

end NUMINAMATH_GPT_more_tvs_sold_l894_89471


namespace NUMINAMATH_GPT_cost_of_10_pound_bag_is_correct_l894_89403

noncomputable def cost_of_5_pound_bag : ℝ := 13.80
noncomputable def cost_of_25_pound_bag : ℝ := 32.25
noncomputable def min_pounds_needed : ℝ := 65
noncomputable def max_pounds_allowed : ℝ := 80
noncomputable def least_possible_cost : ℝ := 98.73

def min_cost_10_pound_bag : ℝ := 1.98

theorem cost_of_10_pound_bag_is_correct :
  ∀ (x : ℝ), (x >= min_pounds_needed / cost_of_25_pound_bag ∧ x <= max_pounds_allowed / cost_of_5_pound_bag ∧ least_possible_cost = (3 * cost_of_25_pound_bag + x)) → x = min_cost_10_pound_bag :=
by
  sorry

end NUMINAMATH_GPT_cost_of_10_pound_bag_is_correct_l894_89403


namespace NUMINAMATH_GPT_recycle_cans_l894_89462

theorem recycle_cans (initial_cans : ℕ) (recycle_rate : ℕ) (n1 n2 n3 : ℕ)
  (h1 : initial_cans = 450)
  (h2 : recycle_rate = 5)
  (h3 : n1 = initial_cans / recycle_rate)
  (h4 : n2 = n1 / recycle_rate)
  (h5 : n3 = n2 / recycle_rate)
  (h6 : n3 / recycle_rate = 0) : 
  n1 + n2 + n3 = 111 :=
by
  sorry

end NUMINAMATH_GPT_recycle_cans_l894_89462


namespace NUMINAMATH_GPT_x_sq_y_sq_value_l894_89441

theorem x_sq_y_sq_value (x y : ℝ) 
  (h1 : x + y = 25) 
  (h2 : x^2 + y^2 = 169) 
  (h3 : x^3 * y^3 + y^3 * x^3 = 243) :
  x^2 * y^2 = 51984 := 
by 
  -- Proof to be added
  sorry

end NUMINAMATH_GPT_x_sq_y_sq_value_l894_89441


namespace NUMINAMATH_GPT_employed_males_percentage_l894_89438

theorem employed_males_percentage (total_population employed employed_as_percent employed_females female_as_percent employed_males employed_males_percentage : ℕ) 
(total_population_eq : total_population = 100)
(employed_eq : employed = employed_as_percent * total_population / 100)
(employed_as_percent_eq : employed_as_percent = 60)
(employed_females_eq : employed_females = female_as_percent * employed / 100)
(female_as_percent_eq : female_as_percent = 25)
(employed_males_eq : employed_males = employed - employed_females)
(employed_males_percentage_eq : employed_males_percentage = employed_males * 100 / total_population) :
employed_males_percentage = 45 :=
sorry

end NUMINAMATH_GPT_employed_males_percentage_l894_89438


namespace NUMINAMATH_GPT_brush_length_percentage_increase_l894_89492

-- Define the length of Carla's brush in inches
def carla_brush_length_in_inches : ℝ := 12

-- Define the conversion factor from inches to centimeters
def inch_to_cm : ℝ := 2.54

-- Define the length of Carmen's brush in centimeters
def carmen_brush_length_in_cm : ℝ := 45

-- Noncomputable definition to calculate the percentage increase
noncomputable def percentage_increase : ℝ :=
  let carla_brush_length_in_cm := carla_brush_length_in_inches * inch_to_cm
  (carmen_brush_length_in_cm - carla_brush_length_in_cm) / carla_brush_length_in_cm * 100

-- Statement to prove the percentage increase is 47.6%
theorem brush_length_percentage_increase :
  percentage_increase = 47.6 :=
sorry

end NUMINAMATH_GPT_brush_length_percentage_increase_l894_89492


namespace NUMINAMATH_GPT_science_club_officers_l894_89466

-- Definitions of the problem conditions
def num_members : ℕ := 25
def num_officers : ℕ := 3
def alice : ℕ := 1 -- unique identifier for Alice
def bob : ℕ := 2 -- unique identifier for Bob

-- Main theorem statement
theorem science_club_officers :
  ∃ (ways_to_choose_officers : ℕ), ways_to_choose_officers = 10764 :=
  sorry

end NUMINAMATH_GPT_science_club_officers_l894_89466


namespace NUMINAMATH_GPT_max_value_of_y_l894_89483

theorem max_value_of_y (x : ℝ) (h₁ : 0 < x) (h₂ : x < 4) : 
  ∃ y : ℝ, (y = x * (8 - 2 * x)) ∧ (∀ z : ℝ, z = x * (8 - 2 * x) → z ≤ 8) :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l894_89483


namespace NUMINAMATH_GPT_union_complement_U_B_l894_89498

def U : Set ℤ := { x | -3 < x ∧ x < 3 }
def A : Set ℤ := { 1, 2 }
def B : Set ℤ := { -2, -1, 2 }

theorem union_complement_U_B : A ∪ (U \ B) = { 0, 1, 2 } := by
  sorry

end NUMINAMATH_GPT_union_complement_U_B_l894_89498


namespace NUMINAMATH_GPT_solve_for_a_l894_89479

theorem solve_for_a (S P Q R : Type) (a b c d : ℝ) 
  (h1 : a + b + c + d = 360)
  (h2 : ∀ (PSQ : Type), d = 90) :
  a = 270 - b - c :=
by
  sorry

end NUMINAMATH_GPT_solve_for_a_l894_89479


namespace NUMINAMATH_GPT_length_of_courtyard_l894_89493

-- Define the dimensions and properties of the courtyard and paving stones
def width := 33 / 2
def numPavingStones := 132
def pavingStoneLength := 5 / 2
def pavingStoneWidth := 2

-- Total area covered by paving stones
def totalArea := numPavingStones * (pavingStoneLength * pavingStoneWidth)

-- To prove: Length of the courtyard
theorem length_of_courtyard : totalArea / width = 40 := by
  sorry

end NUMINAMATH_GPT_length_of_courtyard_l894_89493


namespace NUMINAMATH_GPT_find_a1_l894_89402

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n : ℕ, a n = a 0 + n * d

def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  ∀ n : ℕ, S n = n / 2 * (a 1 + a n)

theorem find_a1 (d : ℝ) (h1 : a 13 = 13) (h2 : S 13 = 13) : a 0 = -11 :=
by
  sorry

end NUMINAMATH_GPT_find_a1_l894_89402


namespace NUMINAMATH_GPT_parabola_intersections_l894_89430

theorem parabola_intersections :
  (∀ x y, (y = 4 * x^2 + 4 * x - 7) ↔ (y = x^2 + 5)) →
  (∃ (points : List (ℝ × ℝ)),
    (points = [(-2, 9), (2, 9)]) ∧
    (∀ p ∈ points, ∃ x, p = (x, x^2 + 5) ∧ y = 4 * x^2 + 4 * x - 7)) :=
by sorry

end NUMINAMATH_GPT_parabola_intersections_l894_89430


namespace NUMINAMATH_GPT_problem_statement_l894_89444

noncomputable def a : ℕ := by
  -- The smallest positive two-digit multiple of 3
  let a := Finset.range 100 \ Finset.range 10
  let multiples := a.filter (λ n => n % 3 = 0)
  exact multiples.min' ⟨12, sorry⟩

noncomputable def b : ℕ := by
  -- The smallest positive three-digit multiple of 4
  let b := Finset.range 1000 \ Finset.range 100
  let multiples := b.filter (λ n => n % 4 = 0)
  exact multiples.min' ⟨100, sorry⟩

theorem problem_statement : a + b = 112 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l894_89444


namespace NUMINAMATH_GPT_shaded_area_l894_89433

noncomputable def squareArea (a : ℝ) : ℝ := a * a

theorem shaded_area {s : ℝ} (h1 : squareArea s = 1) (h2 : s / s = 2) : 
  ∃ (shaded : ℝ), shaded = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l894_89433


namespace NUMINAMATH_GPT_sally_cards_l894_89475

theorem sally_cards (x : ℕ) (h1 : 27 + x + 20 = 88) : x = 41 := by
  sorry

end NUMINAMATH_GPT_sally_cards_l894_89475


namespace NUMINAMATH_GPT_part1_part2_part3_part3_expectation_l894_89463

/-- Conditions setup -/
noncomputable def gameCondition (Aacc Bacc : ℝ) :=
  (Aacc = 0.5) ∧ (Bacc = 0.6)

def scoreDist (X:ℤ) : ℝ :=
  if X = -1 then 0.3
  else if X = 0 then 0.5
  else if X = 1 then 0.2
  else 0

def tieProbability : ℝ := 0.2569

def roundDist (Y:ℤ) : ℝ :=
  if Y = 2 then 0.13
  else if Y = 3 then 0.13
  else if Y = 4 then 0.74
  else 0

def roundExpectation : ℝ := 3.61

/-- Proof Statements -/
theorem part1 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (X : ℤ), scoreDist X = if X = -1 then 0.3 else if X = 0 then 0.5 else if X = 1 then 0.2 else 0 :=
by sorry

theorem part2 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  tieProbability = 0.2569 :=
by sorry

theorem part3 (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) : 
  ∀ (Y : ℤ), roundDist Y = if Y = 2 then 0.13 else if Y = 3 then 0.13 else if Y = 4 then 0.74 else 0 :=
by sorry

theorem part3_expectation (Aacc Bacc : ℝ) (h : gameCondition Aacc Bacc) :
  roundExpectation = 3.61 :=
by sorry

end NUMINAMATH_GPT_part1_part2_part3_part3_expectation_l894_89463


namespace NUMINAMATH_GPT_part1_part2_l894_89422

/-- Part (1) -/
theorem part1 (a : ℝ) (p : ∀ x : ℝ, x^2 - a*x + 4 > 0) (q : ∀ x y : ℝ, (0 < x ∧ x < y) → x^a < y^a) : 
  0 < a ∧ a < 4 :=
sorry

/-- Part (2) -/
theorem part2 (a : ℝ) (p_iff: ∀ x : ℝ, x^2 - a*x + 4 > 0 ↔ -4 < a ∧ a < 4)
  (q_iff: ∀ x y : ℝ, (0 < x ∧ x < y) ↔ x^a < y^a ∧ a > 0) (hp : ∃ x : ℝ, ¬(x^2 - a*x + 4 > 0))
  (hq : ∀ x y : ℝ, (x^a < y^a) → (0 < x ∧ x < y)) : 
  (a >= 4) ∨ (-4 < a ∧ a <= 0) :=
sorry

end NUMINAMATH_GPT_part1_part2_l894_89422


namespace NUMINAMATH_GPT_find_value_of_a_l894_89423

theorem find_value_of_a (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
(h_eq : 7 * a^2 + 14 * a * b = a^3 + 2 * a^2 * b) : a = 7 := 
sorry

end NUMINAMATH_GPT_find_value_of_a_l894_89423


namespace NUMINAMATH_GPT_rationalize_sqrt_5_over_12_l894_89424

theorem rationalize_sqrt_5_over_12 : Real.sqrt (5 / 12) = (Real.sqrt 15) / 6 :=
sorry

end NUMINAMATH_GPT_rationalize_sqrt_5_over_12_l894_89424


namespace NUMINAMATH_GPT_no_real_roots_ffx_l894_89489

noncomputable def quadratic_f (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem no_real_roots_ffx (a b c : ℝ) (h : (b - 1)^2 < 4 * a * c) :
  ∀ x : ℝ, quadratic_f a b c (quadratic_f a b c x) ≠ x :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_ffx_l894_89489


namespace NUMINAMATH_GPT_toys_sold_week2_l894_89437

-- Define the given conditions
def original_stock := 83
def toys_sold_week1 := 38
def toys_left := 19

-- Define the statement we want to prove
theorem toys_sold_week2 : (original_stock - toys_left) - toys_sold_week1 = 26 :=
by
  sorry

end NUMINAMATH_GPT_toys_sold_week2_l894_89437


namespace NUMINAMATH_GPT_length_of_GH_l894_89431

theorem length_of_GH (AB FE CD : ℕ) (side_large side_second side_third side_small : ℕ) 
  (h1 : AB = 11) (h2 : FE = 13) (h3 : CD = 5)
  (h4 : side_large = side_second + AB)
  (h5 : side_second = side_third + CD)
  (h6 : side_third = side_small + FE) :
  GH = 29 :=
by
  -- Proof steps would follow here based on the problem's solution
  -- Using the given conditions and transformations.
  sorry

end NUMINAMATH_GPT_length_of_GH_l894_89431


namespace NUMINAMATH_GPT_intersection_A_compB_l894_89457

def setA : Set ℤ := {x | (abs (x - 1) < 3)}
def setB : Set ℝ := {x | x^2 + 2 * x - 3 ≥ 0}
def setCompB : Set ℝ := {x | ¬(x^2 + 2 * x - 3 ≥ 0)}

theorem intersection_A_compB :
  { x : ℤ | x ∈ setA ∧ (x:ℝ) ∈ setCompB } = {-1, 0} :=
sorry

end NUMINAMATH_GPT_intersection_A_compB_l894_89457


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l894_89470

variable (N : ℕ) (r : Fin N → Fin N → ℝ)

-- Part (a)
theorem part_a (h : ∀ (s : Finset (Fin N)), s.card = 5 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

-- Part (b)
theorem part_b (h : ∀ (s : Finset (Fin N)), s.card = 4 → (exists pts : s → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ¬ (∃ pts : Fin N → ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j) :=
sorry

-- Part (c)
theorem part_c (h : ∀ (s : Finset (Fin N)), s.card = 6 → (exists pts : s → ℝ × ℝ × ℝ, ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j)) :
  ∃ (pts : Fin N → ℝ × ℝ × ℝ), ∀ i j, i ≠ j → dist (pts i) (pts j) = r i j :=
sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l894_89470


namespace NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l894_89409

variable {a b x y : ℝ}

theorem simplify_expression1 : 3 * a - 5 * b - 2 * a + b = a - 4 * b :=
by sorry

theorem simplify_expression2 : 4 * x^2 + 5 * x * y - 2 * (2 * x^2 - x * y) = 7 * x * y :=
by sorry

end NUMINAMATH_GPT_simplify_expression1_simplify_expression2_l894_89409


namespace NUMINAMATH_GPT_find_number_of_women_in_first_group_l894_89484

variables (W : ℕ)

-- Conditions
def women_coloring_rate := 10
def total_cloth_colored_in_3_days := 180
def women_in_first_group := total_cloth_colored_in_3_days / 3

theorem find_number_of_women_in_first_group
  (h1 : 5 * women_coloring_rate * 4 = 200)
  (h2 : W * women_coloring_rate = women_in_first_group) :
  W = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_number_of_women_in_first_group_l894_89484


namespace NUMINAMATH_GPT_red_balls_count_after_game_l894_89487

structure BagState :=
  (red : Nat)         -- Number of red balls
  (green : Nat)       -- Number of green balls
  (blue : Nat)        -- Number of blue balls
  (yellow : Nat)      -- Number of yellow balls
  (black : Nat)       -- Number of black balls
  (white : Nat)       -- Number of white balls)

def initialBallCount (totalBalls : Nat) : BagState :=
  let totalRatio := 15 + 13 + 17 + 9 + 7 + 23
  { red := totalBalls * 15 / totalRatio
  , green := totalBalls * 13 / totalRatio
  , blue := totalBalls * 17 / totalRatio
  , yellow := totalBalls * 9 / totalRatio
  , black := totalBalls * 7 / totalRatio
  , white := totalBalls * 23 / totalRatio
  }

def finalBallCount (initialState : BagState) : BagState :=
  { red := initialState.red + 400
  , green := initialState.green - 250
  , blue := initialState.blue
  , yellow := initialState.yellow - 100
  , black := initialState.black + 200
  , white := initialState.white - 500
  }

theorem red_balls_count_after_game :
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  final.red = 2185 :=
by
  let initial := initialBallCount 10000
  let final := finalBallCount initial
  sorry

end NUMINAMATH_GPT_red_balls_count_after_game_l894_89487


namespace NUMINAMATH_GPT_min_side_length_is_isosceles_l894_89450

-- Let a denote the side length BC
-- Let b denote the side length AB
-- Let c denote the side length AC

theorem min_side_length_is_isosceles (α : ℝ) (S : ℝ) (a b c : ℝ) :
  (a^2 = b^2 + c^2 - 2 * b * c * Real.cos α ∧ S = 0.5 * b * c * Real.sin α) →
  a = Real.sqrt (((b - c)^2 + (4 * S * (1 - Real.cos α)) / Real.sin α)) →
  b = c :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_min_side_length_is_isosceles_l894_89450


namespace NUMINAMATH_GPT_one_cow_one_bag_l894_89400

-- Definitions based on the conditions provided.
def cows : ℕ := 45
def bags : ℕ := 45
def days : ℕ := 45

-- Problem statement: Prove that one cow will eat one bag of husk in 45 days.
theorem one_cow_one_bag (h : cows * bags = bags * days) : days = 45 :=
by
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_l894_89400


namespace NUMINAMATH_GPT_trisha_money_left_l894_89460

theorem trisha_money_left
    (meat cost: ℕ) (chicken_cost: ℕ) (veggies_cost: ℕ) (eggs_cost: ℕ) (dog_food_cost: ℕ) 
    (initial_money: ℕ) (total_spent: ℕ) (money_left: ℕ) :
    meat_cost = 17 →
    chicken_cost = 22 →
    veggies_cost = 43 →
    eggs_cost = 5 →
    dog_food_cost = 45 →
    initial_money = 167 →
    total_spent = meat_cost + chicken_cost + veggies_cost + eggs_cost + dog_food_cost →
    money_left = initial_money - total_spent →
    money_left = 35 :=
by
    intros
    sorry

end NUMINAMATH_GPT_trisha_money_left_l894_89460


namespace NUMINAMATH_GPT_min_number_of_gennadys_l894_89476

theorem min_number_of_gennadys (a b v g : ℕ) (h_a : a = 45) (h_b : b = 122) (h_v : v = 27)
    (h_needed_g : g = 49) :
    (b - 1) - (a + v) = g :=
by
  -- We include sorry because we are focusing on the statement, not the proof itself.
  sorry

end NUMINAMATH_GPT_min_number_of_gennadys_l894_89476


namespace NUMINAMATH_GPT_find_cost_price_l894_89426

theorem find_cost_price (SP PP : ℝ) (hSP : SP = 600) (hPP : PP = 25) : 
  ∃ CP : ℝ, CP = 480 := 
by
  sorry

end NUMINAMATH_GPT_find_cost_price_l894_89426


namespace NUMINAMATH_GPT_domain_of_function_l894_89440

theorem domain_of_function (x : ℝ) : 4 - x ≥ 0 ∧ x ≠ 2 ↔ (x ≤ 4 ∧ x ≠ 2) :=
sorry

end NUMINAMATH_GPT_domain_of_function_l894_89440


namespace NUMINAMATH_GPT_fraction_zero_solution_l894_89488

theorem fraction_zero_solution (x : ℝ) (h1 : (x - 2) / (x + 3) = 0) (h2 : x + 3 ≠ 0) : x = 2 := 
by sorry

end NUMINAMATH_GPT_fraction_zero_solution_l894_89488


namespace NUMINAMATH_GPT_sphere_touches_pyramid_edges_l894_89496

theorem sphere_touches_pyramid_edges :
  ∃ (KL : ℝ), 
  ∃ (K L M N : ℝ) (MN LN NK : ℝ) (AC: ℝ) (BC: ℝ), 
  MN = 7 ∧ 
  NK = 5 ∧ 
  LN = 2 * Real.sqrt 29 ∧ 
  KL = L ∧ 
  KL = M ∧ 
  KL = 9 :=
sorry

end NUMINAMATH_GPT_sphere_touches_pyramid_edges_l894_89496


namespace NUMINAMATH_GPT_weight_of_11m_rebar_l894_89461

theorem weight_of_11m_rebar (w5m : ℝ) (l5m : ℝ) (l11m : ℝ) 
  (h_w5m : w5m = 15.3) (h_l5m : l5m = 5) (h_l11m : l11m = 11) : 
  (w5m / l5m) * l11m = 33.66 := 
by {
  sorry
}

end NUMINAMATH_GPT_weight_of_11m_rebar_l894_89461


namespace NUMINAMATH_GPT_bronson_yellow_leaves_l894_89497

-- Bronson collects 12 leaves on Thursday
def leaves_thursday : ℕ := 12

-- Bronson collects 13 leaves on Friday
def leaves_friday : ℕ := 13

-- 20% of the leaves are Brown (as a fraction)
def percent_brown : ℚ := 0.2

-- 20% of the leaves are Green (as a fraction)
def percent_green : ℚ := 0.2

theorem bronson_yellow_leaves : 
  (leaves_thursday + leaves_friday) * (1 - percent_brown - percent_green) = 15 := by
sorry

end NUMINAMATH_GPT_bronson_yellow_leaves_l894_89497


namespace NUMINAMATH_GPT_max_total_balls_l894_89495

theorem max_total_balls
  (r₁ : ℕ := 89)
  (t₁ : ℕ := 90)
  (r₂ : ℕ := 8)
  (t₂ : ℕ := 9)
  (y : ℕ)
  (h₁ : t₁ > 0)
  (h₂ : t₂ > 0)
  (h₃ : 92 ≤ (r₁ + r₂ * y) * 100 / (t₁ + t₂ * y))
  : y ≤ 22 → 90 + 9 * y = 288 :=
by sorry

end NUMINAMATH_GPT_max_total_balls_l894_89495


namespace NUMINAMATH_GPT_largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l894_89445

theorem largest_square_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ s, s = (a * b) / (a + b) := 
sorry

theorem largest_rectangle_in_right_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : 
  ∃ x y, x = a / 2 ∧ y = b / 2 :=
sorry

end NUMINAMATH_GPT_largest_square_in_right_triangle_largest_rectangle_in_right_triangle_l894_89445


namespace NUMINAMATH_GPT_nonagon_diagonals_l894_89499

-- Define nonagon and its properties
def is_nonagon (n : ℕ) : Prop := n = 9
def has_parallel_sides (n : ℕ) : Prop := n = 9 ∧ true

-- Define the formula for calculating diagonals in a convex polygon
def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- The main theorem statement
theorem nonagon_diagonals :
  ∀ (n : ℕ), is_nonagon n → has_parallel_sides n → diagonals n = 27 :=  by 
  intros n hn _ 
  rw [is_nonagon] at hn
  rw [hn]
  sorry

end NUMINAMATH_GPT_nonagon_diagonals_l894_89499


namespace NUMINAMATH_GPT_twenty_fifty_yuan_bills_unique_l894_89469

noncomputable def twenty_fifty_yuan_bills (x y : ℕ) : Prop :=
  x + y = 260 ∧ 20 * x + 50 * y = 100 * 100

theorem twenty_fifty_yuan_bills_unique (x y : ℕ) (h : twenty_fifty_yuan_bills x y) :
  x = 100 ∧ y = 160 :=
by
  sorry

end NUMINAMATH_GPT_twenty_fifty_yuan_bills_unique_l894_89469


namespace NUMINAMATH_GPT_rectangle_length_l894_89474

-- Define the area and width of the rectangle as given
def width : ℝ := 4
def area  : ℝ := 28

-- Prove that the length is 7 cm given the conditions
theorem rectangle_length : ∃ length : ℝ, length = 7 ∧ area = length * width :=
sorry

end NUMINAMATH_GPT_rectangle_length_l894_89474


namespace NUMINAMATH_GPT_square_area_l894_89421

theorem square_area (P : ℝ) (hP : P = 32) : ∃ A : ℝ, A = 64 ∧ A = (P / 4) ^ 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_square_area_l894_89421


namespace NUMINAMATH_GPT_choir_row_lengths_l894_89435

theorem choir_row_lengths : 
  ∃ s : Finset ℕ, (∀ d ∈ s, d ∣ 90 ∧ 6 ≤ d ∧ d ≤ 15) ∧ s.card = 4 := by
  sorry

end NUMINAMATH_GPT_choir_row_lengths_l894_89435


namespace NUMINAMATH_GPT_solve_equation_l894_89472

theorem solve_equation (x : ℝ) (h : 2 / x = 1 / (x + 1)) : x = -2 :=
sorry

end NUMINAMATH_GPT_solve_equation_l894_89472


namespace NUMINAMATH_GPT_chocolates_for_sister_l894_89467
-- Importing necessary library

-- Lean 4 statement of the problem
theorem chocolates_for_sister (S : ℕ) 
  (herself_chocolates_per_saturday : ℕ := 2)
  (birthday_gift_chocolates : ℕ := 10)
  (saturdays_in_month : ℕ := 4)
  (total_chocolates : ℕ := 22) 
  (monthly_chocolates_herself := saturdays_in_month * herself_chocolates_per_saturday) 
  (equation : saturdays_in_month * S + monthly_chocolates_herself + birthday_gift_chocolates = total_chocolates) : 
  S = 1 :=
  sorry

end NUMINAMATH_GPT_chocolates_for_sister_l894_89467


namespace NUMINAMATH_GPT_sum_first_12_terms_l894_89446

def arithmetic_sequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

def geometric_mean {α : Type} [Field α] (a b c : α) : Prop :=
b^2 = a * c

def sum_arithmetic_sequence (a : ℕ → ℚ) (n : ℕ) : ℚ :=
n * (a 1 + a n) / 2

theorem sum_first_12_terms 
  (a : ℕ → ℚ)
  (d : ℚ)
  (h1 : arithmetic_sequence a 1)
  (h2 : geometric_mean (a 3) (a 6) (a 11)) :
  sum_arithmetic_sequence a 12 = 96 :=
sorry

end NUMINAMATH_GPT_sum_first_12_terms_l894_89446


namespace NUMINAMATH_GPT_distance_between_points_l894_89452

theorem distance_between_points :
  let x1 := 2
  let y1 := -2
  let x2 := 8
  let y2 := 8
  let dist := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  dist = Real.sqrt 136 :=
by
  -- Proof to be filled in here.
  sorry

end NUMINAMATH_GPT_distance_between_points_l894_89452


namespace NUMINAMATH_GPT_remainder_17_pow_49_mod_5_l894_89425

theorem remainder_17_pow_49_mod_5 : (17^49) % 5 = 2 :=
by
  sorry

end NUMINAMATH_GPT_remainder_17_pow_49_mod_5_l894_89425


namespace NUMINAMATH_GPT_reflection_line_equation_l894_89453

-- Given condition 1: Original line equation
def original_line (x : ℝ) : ℝ := -2 * x + 7

-- Given condition 2: Reflection line
def reflection_line_x : ℝ := 3

-- Proving statement
theorem reflection_line_equation
  (a b : ℝ)
  (h₁ : a = -(-2))
  (h₂ : original_line 3 = 1)
  (h₃ : 1 = a * 3 + b) :
  2 * a + b = -1 :=
  sorry

end NUMINAMATH_GPT_reflection_line_equation_l894_89453


namespace NUMINAMATH_GPT_determine_true_propositions_l894_89448

def p (x y : ℝ) := x > y → -x < -y
def q (x y : ℝ) := (1/x > 1/y) → x < y

theorem determine_true_propositions (x y : ℝ) :
  (p x y ∨ q x y) ∧ (p x y ∧ ¬ q x y) :=
by
  sorry

end NUMINAMATH_GPT_determine_true_propositions_l894_89448


namespace NUMINAMATH_GPT_chord_length_on_parabola_eq_five_l894_89494

theorem chord_length_on_parabola_eq_five
  (A B : ℝ × ℝ)
  (hA : A.snd ^ 2 = 4 * A.fst)
  (hB : B.snd ^ 2 = 4 * B.fst)
  (hM : A.fst + B.fst = 3 ∧ A.snd + B.snd = 2 
     ∧ A.fst - B.fst = 0 ∧ A.snd - B.snd = 0) :
  dist A B = 5 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_chord_length_on_parabola_eq_five_l894_89494


namespace NUMINAMATH_GPT_max_g_value_l894_89429

def g : Nat → Nat
| n => if n < 15 then n + 15 else g (n - 6)

theorem max_g_value : ∀ n, g n ≤ 29 := by
  sorry

end NUMINAMATH_GPT_max_g_value_l894_89429


namespace NUMINAMATH_GPT_lcm_of_denominators_l894_89478

theorem lcm_of_denominators : Nat.lcm (List.foldr Nat.lcm 1 [2, 3, 4, 5, 6, 7]) = 420 :=
by 
  sorry

end NUMINAMATH_GPT_lcm_of_denominators_l894_89478


namespace NUMINAMATH_GPT_find_value_l894_89413

theorem find_value (a : ℝ) (h : a^2 - 2*a = -1) : 3*a^2 - 6*a + 2027 = 2024 :=
sorry

end NUMINAMATH_GPT_find_value_l894_89413


namespace NUMINAMATH_GPT_simplify_polynomial_l894_89468

theorem simplify_polynomial :
  (2 * x * (4 * x ^ 3 - 3 * x + 1) - 4 * (2 * x ^ 3 - x ^ 2 + 3 * x - 5)) =
  8 * x ^ 4 - 8 * x ^ 3 - 2 * x ^ 2 - 10 * x + 20 :=
by
  sorry

end NUMINAMATH_GPT_simplify_polynomial_l894_89468


namespace NUMINAMATH_GPT_p_necessary_but_not_sufficient_for_q_l894_89410

noncomputable def p (x : ℝ) : Prop := abs x ≤ 2
noncomputable def q (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2

theorem p_necessary_but_not_sufficient_for_q :
  (∀ x, q x → p x) ∧ ¬ (∀ x, p x → q x) := 
by 
  sorry

end NUMINAMATH_GPT_p_necessary_but_not_sufficient_for_q_l894_89410


namespace NUMINAMATH_GPT_volume_of_right_square_prism_l894_89480

theorem volume_of_right_square_prism (length width : ℕ) (H1 : length = 12) (H2 : width = 8) :
    ∃ V, (V = 72 ∨ V = 48) :=
by
  sorry

end NUMINAMATH_GPT_volume_of_right_square_prism_l894_89480


namespace NUMINAMATH_GPT_constant_sum_l894_89481

noncomputable def arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  a1 + (n - 1) * d

noncomputable def sum_arithmetic_sequence (a1 d : ℝ) (n : ℕ) : ℝ :=
  n / 2 * (2 * a1 + (n - 1) * d)

theorem constant_sum (a1 d : ℝ) (h : 3 * arithmetic_sequence a1 d 8 = k) :
  ∃ k : ℝ, sum_arithmetic_sequence a1 d 15 = k :=
sorry

end NUMINAMATH_GPT_constant_sum_l894_89481


namespace NUMINAMATH_GPT_solutions_diff_squared_l894_89442

theorem solutions_diff_squared (a b : ℝ) (h : 5 * a^2 - 6 * a - 55 = 0 ∧ 5 * b^2 - 6 * b - 55 = 0) :
  (a - b)^2 = 1296 / 25 := by
  sorry

end NUMINAMATH_GPT_solutions_diff_squared_l894_89442


namespace NUMINAMATH_GPT_solve_inequality_l894_89427

theorem solve_inequality (a : ℝ) :
  (a > 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ -a / 4 < x ∧ x < a / 3)) ∧
  (a = 0 → ∀ x : ℝ, ¬ (12 * x^2 - a * x - a^2 < 0)) ∧ 
  (a < 0 → ∀ x : ℝ, (12 * x^2 - a * x - a^2 < 0 ↔ a / 3 < x ∧ x < -a / 4)) :=
by
  sorry

end NUMINAMATH_GPT_solve_inequality_l894_89427


namespace NUMINAMATH_GPT_speed_of_second_train_l894_89401

-- Define the given values
def length_train1 := 290.0 -- in meters
def speed_train1 := 120.0 -- in km/h
def length_train2 := 210.04 -- in meters
def crossing_time := 9.0 -- in seconds

-- Define the conversion factors and useful calculations
def meters_per_second_to_kmph (v : Float) : Float := v * 3.6
def total_distance := length_train1 + length_train2
def relative_speed_ms := total_distance / crossing_time
def relative_speed_kmph := meters_per_second_to_kmph relative_speed_ms

-- Define the proof statement
theorem speed_of_second_train : relative_speed_kmph - speed_train1 = 80.0 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_second_train_l894_89401


namespace NUMINAMATH_GPT_probability_of_yellow_l894_89447

-- Definitions of the given conditions
def red_jelly_beans := 4
def green_jelly_beans := 8
def yellow_jelly_beans := 9
def blue_jelly_beans := 5
def total_jelly_beans := red_jelly_beans + green_jelly_beans + yellow_jelly_beans + blue_jelly_beans

-- Theorem statement
theorem probability_of_yellow :
  (yellow_jelly_beans : ℚ) / total_jelly_beans = 9 / 26 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_yellow_l894_89447


namespace NUMINAMATH_GPT_smaller_area_l894_89419

theorem smaller_area (x y : ℝ) 
  (h1 : x + y = 900)
  (h2 : y - x = (1 / 5) * (x + y) / 2) :
  x = 405 :=
sorry

end NUMINAMATH_GPT_smaller_area_l894_89419


namespace NUMINAMATH_GPT_harvey_sold_17_steaks_l894_89418

variable (initial_steaks : ℕ) (steaks_left_after_first_sale : ℕ) (steaks_sold_in_second_sale : ℕ)

noncomputable def total_steaks_sold (initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale : ℕ) : ℕ :=
  (initial_steaks - steaks_left_after_first_sale) + steaks_sold_in_second_sale

theorem harvey_sold_17_steaks :
  initial_steaks = 25 →
  steaks_left_after_first_sale = 12 →
  steaks_sold_in_second_sale = 4 →
  total_steaks_sold initial_steaks steaks_left_after_first_sale steaks_sold_in_second_sale = 17 :=
by
  intros
  sorry

end NUMINAMATH_GPT_harvey_sold_17_steaks_l894_89418


namespace NUMINAMATH_GPT_alice_bob_task_l894_89432

theorem alice_bob_task (t : ℝ) (h₁ : 1/4 + 1/6 = 5/12) (h₂ : t - 1/2 ≠ 0) :
    (5/12) * (t - 1/2) = 1 :=
sorry

end NUMINAMATH_GPT_alice_bob_task_l894_89432


namespace NUMINAMATH_GPT_increase_in_green_chameleons_is_11_l894_89420

-- Definitions to encode the problem conditions
def num_green_chameleons_increase : Nat :=
  let sunny_days := 18
  let cloudy_days := 12
  let deltaB := 5
  let delta_A_minus_B := sunny_days - cloudy_days
  delta_A_minus_B + deltaB

-- Assertion to prove
theorem increase_in_green_chameleons_is_11 : num_green_chameleons_increase = 11 := by 
  sorry

end NUMINAMATH_GPT_increase_in_green_chameleons_is_11_l894_89420


namespace NUMINAMATH_GPT_root_expression_value_l894_89482

theorem root_expression_value (p m n : ℝ) 
  (h1 : m^2 + (p - 2) * m + 1 = 0) 
  (h2 : n^2 + (p - 2) * n + 1 = 0) : 
  (m^2 + p * m + 1) * (n^2 + p * n + 1) - 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_root_expression_value_l894_89482


namespace NUMINAMATH_GPT_solve_for_M_plus_N_l894_89412

theorem solve_for_M_plus_N (M N : ℕ) (h1 : 4 * N = 588) (h2 : 4 * 63 = 7 * M) : M + N = 183 := by
  sorry

end NUMINAMATH_GPT_solve_for_M_plus_N_l894_89412


namespace NUMINAMATH_GPT_functional_equation_solution_l894_89449

open Function

theorem functional_equation_solution :
  ∀ (f g : ℚ → ℚ), 
    (∀ x y : ℚ, f (g x + g y) = f (g x) + y ∧ g (f x + f y) = g (f x) + y) →
    (∃ a b : ℚ, (ab = 1) ∧ (∀ x : ℚ, f x = a * x) ∧ (∀ x : ℚ, g x = b * x)) :=
by
  intros f g h
  sorry

end NUMINAMATH_GPT_functional_equation_solution_l894_89449


namespace NUMINAMATH_GPT_number_of_zeros_of_f_l894_89407

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then 4 * Real.exp x - 2
else abs (2 - Real.log x / Real.log 2)

theorem number_of_zeros_of_f :
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ = Real.log (1 / 2) ∧ x₂ = 4 :=
by
  sorry

end NUMINAMATH_GPT_number_of_zeros_of_f_l894_89407


namespace NUMINAMATH_GPT_slices_with_both_toppings_l894_89414

-- Definitions and conditions directly from the problem statement
def total_slices : ℕ := 24
def pepperoni_slices : ℕ := 15
def mushroom_slices : ℕ := 14

-- Theorem proving the number of slices with both toppings
theorem slices_with_both_toppings :
  (∃ n : ℕ, n + (pepperoni_slices - n) + (mushroom_slices - n) = total_slices) → ∃ n : ℕ, n = 5 := 
by 
  sorry

end NUMINAMATH_GPT_slices_with_both_toppings_l894_89414


namespace NUMINAMATH_GPT_weng_hourly_rate_l894_89455

theorem weng_hourly_rate (minutes_worked : ℝ) (earnings : ℝ) (fraction_of_hour : ℝ) 
  (conversion_rate : ℝ) (hourly_rate : ℝ) : 
  minutes_worked = 50 → earnings = 10 → 
  fraction_of_hour = minutes_worked / conversion_rate → 
  conversion_rate = 60 → 
  hourly_rate = earnings / fraction_of_hour → 
  hourly_rate = 12 := by
    sorry

end NUMINAMATH_GPT_weng_hourly_rate_l894_89455


namespace NUMINAMATH_GPT_blue_die_prime_yellow_die_power_2_probability_l894_89439

def prime_numbers : Finset ℕ := {2, 3, 5, 7}

def powers_of_2 : Finset ℕ := {1, 2, 4, 8}

def total_outcomes : ℕ := 8 * 8

def successful_outcomes : ℕ := prime_numbers.card * powers_of_2.card

def probability (x y : Finset ℕ) : ℚ := (x.card * y.card) / (total_outcomes : ℚ)

theorem blue_die_prime_yellow_die_power_2_probability :
  probability prime_numbers powers_of_2 = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_blue_die_prime_yellow_die_power_2_probability_l894_89439


namespace NUMINAMATH_GPT_calories_burned_per_week_l894_89486

-- Definitions from conditions
def classes_per_week : ℕ := 3
def hours_per_class : ℚ := 1.5
def calories_per_minute : ℕ := 7

-- Prove the total calories burned per week
theorem calories_burned_per_week : 
  (classes_per_week * (hours_per_class * 60) * calories_per_minute) = 1890 := by
    sorry

end NUMINAMATH_GPT_calories_burned_per_week_l894_89486


namespace NUMINAMATH_GPT_largest_three_digit_in_pascal_triangle_l894_89477

-- Define Pascal's triangle and binomial coefficient
def pascal (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem about the first appearance of the number 999 in Pascal's triangle
theorem largest_three_digit_in_pascal_triangle :
  ∃ (n : ℕ), n = 1000 ∧ ∃ (k : ℕ), pascal n k = 999 :=
sorry

end NUMINAMATH_GPT_largest_three_digit_in_pascal_triangle_l894_89477


namespace NUMINAMATH_GPT_rectangle_dimensions_l894_89454

theorem rectangle_dimensions (x y : ℝ) (h1 : y = 2 * x) (h2 : 2 * (x + y) = 2 * (x * y)) :
  (x = 3 / 2) ∧ (y = 3) := by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l894_89454


namespace NUMINAMATH_GPT_boat_crossing_time_l894_89451

theorem boat_crossing_time :
  ∀ (width_of_river speed_of_current speed_of_boat : ℝ),
  width_of_river = 1.5 →
  speed_of_current = 8 →
  speed_of_boat = 10 →
  (width_of_river / (Real.sqrt (speed_of_boat ^ 2 - speed_of_current ^ 2)) * 60) = 15 :=
by
  intros width_of_river speed_of_current speed_of_boat h1 h2 h3
  sorry

end NUMINAMATH_GPT_boat_crossing_time_l894_89451


namespace NUMINAMATH_GPT_rearrange_pairs_l894_89465

theorem rearrange_pairs {a b : ℕ} (hb: b = (2 / 3 : ℚ) * a) (boys_way_museum boys_way_back : ℕ) :
  boys_way_museum = 3 * a ∧ boys_way_back = 4 * b → 
  ∃ c : ℕ, boys_way_museum = 7 * c ∧ b = c := sorry

end NUMINAMATH_GPT_rearrange_pairs_l894_89465


namespace NUMINAMATH_GPT_ninth_graders_only_math_l894_89491

theorem ninth_graders_only_math 
  (total_students : ℕ)
  (math_students : ℕ)
  (foreign_language_students : ℕ)
  (science_only_students : ℕ)
  (math_and_foreign_language_no_science : ℕ)
  (h1 : total_students = 120)
  (h2 : math_students = 85)
  (h3 : foreign_language_students = 75)
  (h4 : science_only_students = 20)
  (h5 : math_and_foreign_language_no_science = 40) :
  math_students - math_and_foreign_language_no_science = 45 :=
by 
  sorry

end NUMINAMATH_GPT_ninth_graders_only_math_l894_89491


namespace NUMINAMATH_GPT_leila_cakes_monday_l894_89443

def number_of_cakes_monday (m : ℕ) : Prop :=
  let cakes_friday := 9
  let cakes_saturday := 3 * m
  let total_cakes := m + cakes_friday + cakes_saturday
  total_cakes = 33

theorem leila_cakes_monday : ∃ m : ℕ, number_of_cakes_monday m ∧ m = 6 :=
by 
  -- We propose that the number of cakes she ate on Monday, denoted as m, is 6.
  -- We need to prove that this satisfies the given conditions.
  -- This line is a placeholder for the proof.
  sorry

end NUMINAMATH_GPT_leila_cakes_monday_l894_89443


namespace NUMINAMATH_GPT_project_completion_days_l894_89408

-- Define the work rates and the total number of days to complete the project
variables (a_rate b_rate : ℝ) (days_to_complete : ℝ)
variable (a_quit_before_completion : ℝ)

-- Define the conditions
def A_rate := 1 / 20
def B_rate := 1 / 20
def quit_before_completion := 10 

-- The total work done in the project as 1 project 
def total_work := 1

-- Define the equation representing the amount of work done by A and B
def total_days := 
  A_rate * (days_to_complete - a_quit_before_completion) + B_rate * days_to_complete

-- The theorem statement
theorem project_completion_days :
  A_rate = a_rate → 
  B_rate = b_rate → 
  quit_before_completion = a_quit_before_completion → 
  total_days = total_work → 
  days_to_complete = 15 :=
by 
  -- placeholders for the conditions
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_project_completion_days_l894_89408


namespace NUMINAMATH_GPT_t_le_s_l894_89490

theorem t_le_s (a b : ℝ) (t s : ℝ) (h1 : t = a + 2 * b) (h2 : s = a + b^2 + 1) : t ≤ s :=
by
  sorry

end NUMINAMATH_GPT_t_le_s_l894_89490


namespace NUMINAMATH_GPT_gallon_of_water_weighs_eight_pounds_l894_89406

theorem gallon_of_water_weighs_eight_pounds
  (pounds_per_tablespoon : ℝ := 1.5)
  (cubic_feet_per_gallon : ℝ := 7.5)
  (cost_per_tablespoon : ℝ := 0.50)
  (total_cost : ℝ := 270)
  (bathtub_capacity_cubic_feet : ℝ := 6)
  : (6 * 7.5) * pounds_per_tablespoon = 270 / cost_per_tablespoon / 1.5 :=
by
  sorry

end NUMINAMATH_GPT_gallon_of_water_weighs_eight_pounds_l894_89406


namespace NUMINAMATH_GPT_probability_not_passing_l894_89417

noncomputable def probability_of_passing : ℚ := 4 / 7

theorem probability_not_passing (h : probability_of_passing = 4 / 7) : 1 - probability_of_passing = 3 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_not_passing_l894_89417


namespace NUMINAMATH_GPT_two_digit_number_l894_89405

theorem two_digit_number (x y : ℕ) (h1 : x + y = 11) (h2 : 10 * y + x = 10 * x + y + 63) : 10 * x + y = 29 := 
by 
  sorry

end NUMINAMATH_GPT_two_digit_number_l894_89405
