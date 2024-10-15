import Mathlib

namespace NUMINAMATH_GPT_no_valid_pairs_l1917_191755

theorem no_valid_pairs : ∀ (x y : ℕ), x > 0 → y > 0 → x^2 + y^2 + 1 = x^3 → false := 
by
  intros x y hx hy h
  sorry

end NUMINAMATH_GPT_no_valid_pairs_l1917_191755


namespace NUMINAMATH_GPT_arithmetic_sequence_proof_l1917_191732

variable (n : ℕ)
variable (a_n S_n : ℕ → ℤ)

noncomputable def a : ℕ → ℤ := 48 - 8 * n
noncomputable def S : ℕ → ℤ := -4 * (n ^ 2) + 44 * n

axiom a_3 : a 3 = 24
axiom S_11 : S 11 = 0

theorem arithmetic_sequence_proof :
  a n = 48 - 8 * n ∧
  S n = -4 * n ^ 2 + 44 * n ∧
  ∃ n, S n = 120 ∧ (n = 5 ∨ n = 6) :=
by
  unfold a S
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_proof_l1917_191732


namespace NUMINAMATH_GPT_positive_value_of_m_l1917_191791

variable {m : ℝ}

theorem positive_value_of_m (h : ∃ x : ℝ, (3 * x^2 + m * x + 36) = 0 ∧ (∀ y : ℝ, (3 * y^2 + m * y + 36) = 0 → y = x)) :
  m = 12 * Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_positive_value_of_m_l1917_191791


namespace NUMINAMATH_GPT_spaces_per_tray_l1917_191761

-- Conditions
def num_ice_cubes_glass : ℕ := 8
def num_ice_cubes_pitcher : ℕ := 2 * num_ice_cubes_glass
def total_ice_cubes_used : ℕ := num_ice_cubes_glass + num_ice_cubes_pitcher
def num_trays : ℕ := 2

-- Proof statement
theorem spaces_per_tray : total_ice_cubes_used / num_trays = 12 :=
by
  sorry

end NUMINAMATH_GPT_spaces_per_tray_l1917_191761


namespace NUMINAMATH_GPT_largest_tile_size_l1917_191718

def length_cm : ℕ := 378
def width_cm : ℕ := 525

theorem largest_tile_size :
  Nat.gcd length_cm width_cm = 21 := by
  sorry

end NUMINAMATH_GPT_largest_tile_size_l1917_191718


namespace NUMINAMATH_GPT_chinese_horses_problem_l1917_191758

variables (x y : ℕ)

theorem chinese_horses_problem (h1 : x + y = 100) (h2 : 3 * x + (y / 3) = 100) :
  (x + y = 100) ∧ (3 * x + (y / 3) = 100) :=
by
  sorry

end NUMINAMATH_GPT_chinese_horses_problem_l1917_191758


namespace NUMINAMATH_GPT_cannot_be_n_plus_2_l1917_191725

theorem cannot_be_n_plus_2 (n : ℕ) : 
  ¬(∃ Y, (Y = n + 2) ∧ 
         ((Y = n - 3) ∨ (Y = n - 1) ∨ (Y = n + 5))) := 
by {
  sorry
}

end NUMINAMATH_GPT_cannot_be_n_plus_2_l1917_191725


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1917_191746

theorem rhombus_diagonal_length (d2 : ℝ) (area : ℝ) (d1 : ℝ) (h1 : d2 = 80) (h2 : area = 2480) (h3 : area = (d1 * d2) / 2) : d1 = 62 :=
by sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1917_191746


namespace NUMINAMATH_GPT_tall_wins_min_voters_l1917_191727

structure VotingSetup where
  total_voters : ℕ
  districts : ℕ
  sections_per_district : ℕ
  voters_per_section : ℕ
  voters_majority_in_section : ℕ
  districts_to_win : ℕ
  sections_to_win_district : ℕ

def contest_victory (setup : VotingSetup) (min_voters : ℕ) : Prop :=
  setup.total_voters = 105 ∧
  setup.districts = 5 ∧
  setup.sections_per_district = 7 ∧
  setup.voters_per_section = 3 ∧
  setup.voters_majority_in_section = 2 ∧
  setup.districts_to_win = 3 ∧
  setup.sections_to_win_district = 4 ∧
  min_voters = 24

theorem tall_wins_min_voters : ∃ min_voters, contest_victory ⟨105, 5, 7, 3, 2, 3, 4⟩ min_voters :=
by { use 24, sorry }

end NUMINAMATH_GPT_tall_wins_min_voters_l1917_191727


namespace NUMINAMATH_GPT_negation_prop_l1917_191740

theorem negation_prop (x : ℝ) : (¬ (∀ x : ℝ, Real.exp x > x^2)) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) :=
by
  sorry

end NUMINAMATH_GPT_negation_prop_l1917_191740


namespace NUMINAMATH_GPT_calculate_expression_l1917_191742

theorem calculate_expression :
    (2^(1/2) * 4^(1/2)) + (18 / 3 * 3) - 8^(3/2) = 18 - 14 * Real.sqrt 2 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l1917_191742


namespace NUMINAMATH_GPT_distance_last_pair_of_trees_l1917_191762

theorem distance_last_pair_of_trees 
  (yard_length : ℝ := 1200)
  (num_trees : ℕ := 117)
  (initial_distance : ℝ := 5)
  (distance_increment : ℝ := 2) :
  let num_distances := num_trees - 1
  let last_distance := initial_distance + (num_distances - 1) * distance_increment
  last_distance = 235 := by 
  sorry

end NUMINAMATH_GPT_distance_last_pair_of_trees_l1917_191762


namespace NUMINAMATH_GPT_positive_difference_is_329_l1917_191720

-- Definitions of the fractions involved
def fraction1 : ℚ := (7^2 + 7^2) / 7
def fraction2 : ℚ := (7^2 * 7^2) / 7

-- Statement of the positive difference proof
theorem positive_difference_is_329 : abs (fraction2 - fraction1) = 329 := by
  -- Skipping the proof here
  sorry

end NUMINAMATH_GPT_positive_difference_is_329_l1917_191720


namespace NUMINAMATH_GPT_steel_scrap_problem_l1917_191729

theorem steel_scrap_problem 
  (x y : ℝ)
  (h1 : x + y = 140)
  (h2 : 0.05 * x + 0.40 * y = 42) :
  x = 40 ∧ y = 100 :=
by
  -- Solution steps are not required here
  sorry

end NUMINAMATH_GPT_steel_scrap_problem_l1917_191729


namespace NUMINAMATH_GPT_smaller_number_l1917_191706

theorem smaller_number (a b : ℕ) (h1 : 10 ≤ a ∧ a < 100) (h2 : 10 ≤ b ∧ b < 100) (h3 : a * b = 4851) : min a b = 53 :=
sorry

end NUMINAMATH_GPT_smaller_number_l1917_191706


namespace NUMINAMATH_GPT_p_is_sufficient_not_necessary_for_q_l1917_191780

-- Definitions for conditions p and q
def p (x : ℝ) := x^2 - x - 20 > 0
def q (x : ℝ) := 1 - x^2 < 0

-- The main statement
theorem p_is_sufficient_not_necessary_for_q:
  (∀ x, p x → q x) ∧ ¬(∀ x, q x → p x) :=
by
  sorry

end NUMINAMATH_GPT_p_is_sufficient_not_necessary_for_q_l1917_191780


namespace NUMINAMATH_GPT_sin_double_angle_l1917_191712

open Real

theorem sin_double_angle (α : ℝ) (h1 : α ∈ Set.Ioc (π / 2) π) (h2 : sin α = 4 / 5) :
  sin (2 * α) = -24 / 25 :=
by
  sorry

end NUMINAMATH_GPT_sin_double_angle_l1917_191712


namespace NUMINAMATH_GPT_find_fifth_term_l1917_191717

noncomputable def geometric_sequence_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : ℝ :=
  a * r^4

theorem find_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : geometric_sequence_fifth_term a r h₁ h₂ = 2 := sorry

end NUMINAMATH_GPT_find_fifth_term_l1917_191717


namespace NUMINAMATH_GPT_size_of_third_file_l1917_191754

theorem size_of_third_file 
  (s : ℝ) (t : ℝ) (f1 : ℝ) (f2 : ℝ) (f3 : ℝ) 
  (h1 : s = 2) (h2 : t = 120) (h3 : f1 = 80) (h4 : f2 = 90) : 
  f3 = s * t - (f1 + f2) :=
by
  sorry

end NUMINAMATH_GPT_size_of_third_file_l1917_191754


namespace NUMINAMATH_GPT_chuck_total_time_on_trip_l1917_191794

def distance_into_country : ℝ := 28.8
def rate_out : ℝ := 16
def rate_back : ℝ := 24

theorem chuck_total_time_on_trip : (distance_into_country / rate_out) + (distance_into_country / rate_back) = 3 := 
by sorry

end NUMINAMATH_GPT_chuck_total_time_on_trip_l1917_191794


namespace NUMINAMATH_GPT_chess_tournament_green_teams_l1917_191768

theorem chess_tournament_green_teams :
  ∀ (R G total_teams : ℕ)
  (red_team_count : ℕ → ℕ)
  (green_team_count : ℕ → ℕ)
  (mixed_team_count : ℕ → ℕ),
  R = 64 → G = 68 → total_teams = 66 →
  red_team_count R = 20 →
  (R + G = 132) →
  -- Details derived from mixed_team_count and green_team_count
  -- are inferred from the conditions provided
  mixed_team_count R + red_team_count R = 32 → 
  -- Total teams by definition including mixed teams 
  mixed_team_count G = G - (2 * red_team_count R) - green_team_count G →
  green_team_count (G - (mixed_team_count R)) = 2 → 
  2 * (green_team_count G) = 22 :=
by sorry

end NUMINAMATH_GPT_chess_tournament_green_teams_l1917_191768


namespace NUMINAMATH_GPT_solve_identity_l1917_191775

theorem solve_identity (x : ℝ) (a b p q : ℝ)
  (h : (6 * x + 1) / (6 * x ^ 2 + 19 * x + 15) = a / (x - p) + b / (x - q)) :
  a = -1 ∧ b = 2 ∧ p = -3/4 ∧ q = -5/3 :=
by
  sorry

end NUMINAMATH_GPT_solve_identity_l1917_191775


namespace NUMINAMATH_GPT_toys_left_l1917_191774

-- Given conditions
def initial_toys := 7
def sold_toys := 3

-- Proven statement
theorem toys_left : initial_toys - sold_toys = 4 := by
  sorry

end NUMINAMATH_GPT_toys_left_l1917_191774


namespace NUMINAMATH_GPT_sum_powers_is_76_l1917_191789

theorem sum_powers_is_76 (m n : ℕ) (h1 : m + n = 1) (h2 : m^2 + n^2 = 3)
                         (h3 : m^3 + n^3 = 4) (h4 : m^4 + n^4 = 7)
                         (h5 : m^5 + n^5 = 11) : m^9 + n^9 = 76 :=
sorry

end NUMINAMATH_GPT_sum_powers_is_76_l1917_191789


namespace NUMINAMATH_GPT_highway_length_is_105_l1917_191772

-- Define the speeds of the two cars
def speed_car1 : ℝ := 15
def speed_car2 : ℝ := 20

-- Define the time they travel for
def time_travelled : ℝ := 3

-- Define the distances covered by the cars
def distance_car1 : ℝ := speed_car1 * time_travelled
def distance_car2 : ℝ := speed_car2 * time_travelled

-- Define the total length of the highway
def length_highway : ℝ := distance_car1 + distance_car2

-- The theorem statement
theorem highway_length_is_105 : length_highway = 105 :=
by
  -- Skipping the proof for now
  sorry

end NUMINAMATH_GPT_highway_length_is_105_l1917_191772


namespace NUMINAMATH_GPT_ways_to_make_30_cents_is_17_l1917_191778

-- Define the value of each type of coin
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25

-- Define the function that counts the number of ways to make 30 cents
def count_ways_to_make_30_cents : ℕ :=
  let ways_with_1_quarter := (if 30 - quarter_value == 5 then 2 else 0)
  let ways_with_0_quarters :=
    let ways_with_2_dimes := (if 30 - 2 * dime_value == 10 then 3 else 0)
    let ways_with_1_dime := (if 30 - dime_value == 20 then 5 else 0)
    let ways_with_0_dimes := (if 30 == 30 then 7 else 0)
    ways_with_2_dimes + ways_with_1_dime + ways_with_0_dimes
  2 + ways_with_1_quarter + ways_with_0_quarters

-- Proof statement
theorem ways_to_make_30_cents_is_17 : count_ways_to_make_30_cents = 17 := sorry

end NUMINAMATH_GPT_ways_to_make_30_cents_is_17_l1917_191778


namespace NUMINAMATH_GPT_perfect_square_for_n_l1917_191741

theorem perfect_square_for_n 
  (a b : ℕ)
  (h1 : ∃ x : ℕ, ab = x^2)
  (h2 : ∃ y : ℕ, (a + 1) * (b + 1) = y^2) 
  : ∃ n : ℕ, n > 1 ∧ ∃ z : ℕ, (a + n) * (b + n) = z^2 :=
by
  let n := ab
  have h3 : n > 1 := sorry
  have h4 : ∃ z : ℕ, (a + n) * (b + n) = z^2 := sorry
  exact ⟨n, h3, h4⟩

end NUMINAMATH_GPT_perfect_square_for_n_l1917_191741


namespace NUMINAMATH_GPT_petya_vasya_same_sum_l1917_191701

theorem petya_vasya_same_sum :
  ∃ n : ℕ, (n * (n + 1)) / 2 = 2^99 * (2^100 - 1) :=
by
  sorry

end NUMINAMATH_GPT_petya_vasya_same_sum_l1917_191701


namespace NUMINAMATH_GPT_min_value_of_square_sum_l1917_191738

theorem min_value_of_square_sum (x y : ℝ) 
  (h1 : (x + 5) ^ 2 + (y - 12) ^ 2 = 14 ^ 2) : 
  x^2 + y^2 = 1 := 
sorry

end NUMINAMATH_GPT_min_value_of_square_sum_l1917_191738


namespace NUMINAMATH_GPT_simplify_expression_eq_square_l1917_191749

theorem simplify_expression_eq_square (a : ℤ) : a * (a + 2) - 2 * a = a^2 :=
by sorry

end NUMINAMATH_GPT_simplify_expression_eq_square_l1917_191749


namespace NUMINAMATH_GPT_strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l1917_191767

def f (x : ℝ) : ℝ := -x * abs x + 2 * x

theorem strictly_increasing : ∃ A : Set ℝ, A = (Set.Ioo 0 1) ∧ (∀ x y, x ∈ A → y ∈ A → x < y → f x < f y) :=
  sorry

theorem not_gamma_interval : ¬(Set.Icc (1/2) (3/2) ⊆ Set.Ioo 0 1 ∧ 
  (∀ x ∈ Set.Icc (1/2) (3/2), f x ∈ Set.Icc (1/(3/2)) (1/(1/2)))) :=
  sorry

theorem gamma_interval_within_one_inf : ∃ m n : ℝ, 1 ≤ m ∧ m < n ∧ 
  Set.Icc m n = Set.Icc 1 ((1 + Real.sqrt 5) / 2) ∧ 
  (∀ x ∈ Set.Icc m n, f x ∈ Set.Icc (1/n) (1/m)) :=
  sorry

end NUMINAMATH_GPT_strictly_increasing_not_gamma_interval_gamma_interval_within_one_inf_l1917_191767


namespace NUMINAMATH_GPT_verify_solution_l1917_191737

variable (x y : ℝ)

-- Conditions
def condition1 : Prop := x - y = 9
def condition2 : Prop := 4 * x + 3 * y = 1

-- Proof problem statement
theorem verify_solution
  (h1 : condition1 x y)
  (h2 : condition2 x y) :
  x = 4 ∧ y = -5 :=
sorry

end NUMINAMATH_GPT_verify_solution_l1917_191737


namespace NUMINAMATH_GPT_find_f_at_6_5_l1917_191711

noncomputable def f : ℝ → ℝ := sorry

axiom even_function (x : ℝ) : f (-x) = f x
axiom functional_equation (x : ℝ) : f (x + 2) = - (1 / f x)
axiom initial_condition (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) : f x = x - 2

theorem find_f_at_6_5 : f 6.5 = -0.5 := by
  sorry

end NUMINAMATH_GPT_find_f_at_6_5_l1917_191711


namespace NUMINAMATH_GPT_school_children_count_l1917_191702

theorem school_children_count (C B : ℕ) (h1 : B = 2 * C) (h2 : B = 4 * (C - 370)) : C = 740 :=
by sorry

end NUMINAMATH_GPT_school_children_count_l1917_191702


namespace NUMINAMATH_GPT_car_win_probability_l1917_191795

noncomputable def P (n : ℕ) : ℚ := 1 / n

theorem car_win_probability :
  let P_x := 1 / 7
  let P_y := 1 / 3
  let P_z := 1 / 5
  P_x + P_y + P_z = 71 / 105 :=
by
  sorry

end NUMINAMATH_GPT_car_win_probability_l1917_191795


namespace NUMINAMATH_GPT_find_2xy2_l1917_191731

theorem find_2xy2 (x y : ℤ) (h : y^2 + 2 * x^2 * y^2 = 20 * x^2 + 412) : 2 * x * y^2 = 288 :=
sorry

end NUMINAMATH_GPT_find_2xy2_l1917_191731


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1917_191760

section

variables (x y : Real)

-- Given conditions
def x_def : x = 3 + 2 * Real.sqrt 2 := sorry
def y_def : y = 3 - 2 * Real.sqrt 2 := sorry

-- Problem 1: Prove x + y = 6
theorem problem1 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x + y = 6 := 
by sorry

-- Problem 2: Prove x - y = 4 * sqrt 2
theorem problem2 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x - y = 4 * Real.sqrt 2 :=
by sorry

-- Problem 3: Prove xy = 1
theorem problem3 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x * y = 1 := 
by sorry

-- Problem 4: Prove x^2 - 3xy + y^2 - x - y = 25
theorem problem4 (h₁ : x = 3 + 2 * Real.sqrt 2) (h₂ : y = 3 - 2 * Real.sqrt 2) : x^2 - 3 * x * y + y^2 - x - y = 25 :=
by sorry

end

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1917_191760


namespace NUMINAMATH_GPT_find_norm_b_projection_of_b_on_a_l1917_191769

open Real EuclideanSpace

noncomputable def a : ℝ := 4

noncomputable def angle_ab : ℝ := π / 4  -- 45 degrees in radians

noncomputable def inner_prod_condition (b : ℝ) : ℝ := 
  (1 / 2 * a) * (2 * a) + 
  (1 / 2 * a) * (-3 * b) + 
  b * (2 * a) + 
  b * (-3 * b) - 12

theorem find_norm_b (b : ℝ) (hb : inner_prod_condition b = 0) : b = sqrt 2 :=
  sorry

theorem projection_of_b_on_a (b : ℝ) (hb : inner_prod_condition b = 0) : 
  (b * cos angle_ab) = 1 :=
  sorry

end NUMINAMATH_GPT_find_norm_b_projection_of_b_on_a_l1917_191769


namespace NUMINAMATH_GPT_units_digit_of_n_l1917_191715

def units_digit (x : ℕ) : ℕ := x % 10

theorem units_digit_of_n 
  (m n : ℕ) 
  (h1 : m * n = 21 ^ 6) 
  (h2 : units_digit m = 7) : 
  units_digit n = 3 := 
sorry

end NUMINAMATH_GPT_units_digit_of_n_l1917_191715


namespace NUMINAMATH_GPT_correct_choice_of_f_l1917_191751

def f1 (x : ℝ) : ℝ := (x - 1)^2 + 3 * (x - 1)
def f2 (x : ℝ) : ℝ := 2 * (x - 1)
def f3 (x : ℝ) : ℝ := 2 * (x - 1)^2
def f4 (x : ℝ) : ℝ := x - 1

theorem correct_choice_of_f (h : (deriv f1 1 = 3) ∧ (deriv f2 1 ≠ 3) ∧ (deriv f3 1 ≠ 3) ∧ (deriv f4 1 ≠ 3)) : 
  ∀ f, (f = f1 ∨ f = f2 ∨ f = f3 ∨ f = f4) → (deriv f 1 = 3 → f = f1) :=
by sorry

end NUMINAMATH_GPT_correct_choice_of_f_l1917_191751


namespace NUMINAMATH_GPT_proposition_D_l1917_191719

theorem proposition_D (a b c d : ℝ) (h1 : a < b) (h2 : c < d) : a + c < b + d :=
sorry

end NUMINAMATH_GPT_proposition_D_l1917_191719


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1917_191781

-- Define the lengths of the sides
def side1 := 2 -- 2 cm
def side2 := 4 -- 4 cm

-- Define the condition of being isosceles
def is_isosceles (a b c : ℝ) : Prop := (a = b) ∨ (a = c) ∨ (b = c)

-- Define the triangle inequality
def triangle_inequality (a b c : ℝ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

-- Define the triangle perimeter
def perimeter (a b c : ℝ) : ℝ := a + b + c

-- Define the main theorem to prove
theorem isosceles_triangle_perimeter {a b : ℝ} (ha : a = side1) (hb : b = side2)
    (h1 : is_isosceles a b c) (h2 : triangle_inequality a b c) : perimeter a b c = 10 :=
sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1917_191781


namespace NUMINAMATH_GPT_ratio_perimeter_pentagon_to_square_l1917_191703

theorem ratio_perimeter_pentagon_to_square
  (a : ℝ) -- Let a be the length of each side of the square
  (T_perimeter S_perimeter : ℝ) 
  (h1 : T_perimeter = S_perimeter) -- Given the perimeter of the triangle equals the perimeter of the square
  (h2 : S_perimeter = 4 * a) -- Given the perimeter of the square is 4 times the length of its side
  (P_perimeter : ℝ)
  (h3 : P_perimeter = (T_perimeter + S_perimeter) - 2 * a) -- Perimeter of the pentagon considering shared edge
  :
  P_perimeter / S_perimeter = 3 / 2 := 
sorry

end NUMINAMATH_GPT_ratio_perimeter_pentagon_to_square_l1917_191703


namespace NUMINAMATH_GPT_correct_inequality_l1917_191744

theorem correct_inequality (a b c d : ℝ)
    (hab : a > b) (hb0 : b > 0)
    (hcd : c > d) (hd0 : d > 0) :
    Real.sqrt (a / d) > Real.sqrt (b / c) :=
by
    sorry

end NUMINAMATH_GPT_correct_inequality_l1917_191744


namespace NUMINAMATH_GPT_arccos_one_eq_zero_l1917_191750

theorem arccos_one_eq_zero : Real.arccos 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_arccos_one_eq_zero_l1917_191750


namespace NUMINAMATH_GPT_total_volume_of_five_boxes_l1917_191779

-- Define the edge length of each cube
def edge_length : ℕ := 5

-- Define the volume of one cube
def volume_of_cube (s : ℕ) : ℕ := s ^ 3

-- Define the number of cubes
def number_of_cubes : ℕ := 5

-- Define the total volume
def total_volume (s : ℕ) (n : ℕ) : ℕ := n * (volume_of_cube s)

-- The theorem to prove
theorem total_volume_of_five_boxes :
  total_volume edge_length number_of_cubes = 625 := 
by
  -- Proof is skipped
  sorry

end NUMINAMATH_GPT_total_volume_of_five_boxes_l1917_191779


namespace NUMINAMATH_GPT_prove_m_eq_n_l1917_191747

variable (m n : ℕ)

noncomputable def p := m + n + 1

theorem prove_m_eq_n 
  (is_prime : Prime p) 
  (divides : p ∣ 2 * (m^2 + n^2) - 1) : 
  m = n :=
by
  sorry

end NUMINAMATH_GPT_prove_m_eq_n_l1917_191747


namespace NUMINAMATH_GPT_rational_solves_abs_eq_l1917_191785

theorem rational_solves_abs_eq (x : ℚ) : |6 + x| = |6| + |x| → 0 ≤ x := 
sorry

end NUMINAMATH_GPT_rational_solves_abs_eq_l1917_191785


namespace NUMINAMATH_GPT_product_of_sequence_is_243_l1917_191736

theorem product_of_sequence_is_243 : 
  (1/3 * 9 * 1/27 * 81 * 1/243 * 729 * 1/2187 * 6561 * 1/19683 * 59049) = 243 := 
by
  sorry

end NUMINAMATH_GPT_product_of_sequence_is_243_l1917_191736


namespace NUMINAMATH_GPT_total_teachers_in_all_departments_is_637_l1917_191799

noncomputable def total_teachers : ℕ :=
  let major_departments := 9
  let minor_departments := 8
  let teachers_per_major := 45
  let teachers_per_minor := 29
  (major_departments * teachers_per_major) + (minor_departments * teachers_per_minor)

theorem total_teachers_in_all_departments_is_637 : total_teachers = 637 := 
  by
  sorry

end NUMINAMATH_GPT_total_teachers_in_all_departments_is_637_l1917_191799


namespace NUMINAMATH_GPT_bucket_water_total_l1917_191753

theorem bucket_water_total (initial_gallons : ℝ) (added_gallons : ℝ) (total_gallons : ℝ) : 
  initial_gallons = 3 ∧ added_gallons = 6.8 → total_gallons = 9.8 :=
by
  { sorry }

end NUMINAMATH_GPT_bucket_water_total_l1917_191753


namespace NUMINAMATH_GPT_cream_ratio_l1917_191790

theorem cream_ratio (joe_initial_coffee joann_initial_coffee : ℝ)
                    (joe_drank_ounces joann_drank_ounces joe_added_cream joann_added_cream : ℝ) :
  joe_initial_coffee = 20 →
  joann_initial_coffee = 20 →
  joe_drank_ounces = 3 →
  joann_drank_ounces = 3 →
  joe_added_cream = 4 →
  joann_added_cream = 4 →
  (4 : ℝ) / ((21 / 24) * 24 - 3) = (8 : ℝ) / 7 :=
by
  intros h_ji h_ji h_jd h_jd h_jc h_jc
  sorry

end NUMINAMATH_GPT_cream_ratio_l1917_191790


namespace NUMINAMATH_GPT_exponent_problem_l1917_191764

variable {a m n : ℝ}

theorem exponent_problem (h1 : a^m = 2) (h2 : a^n = 3) : a^(3*m + 2*n) = 72 := 
  sorry

end NUMINAMATH_GPT_exponent_problem_l1917_191764


namespace NUMINAMATH_GPT_parabola_directrix_l1917_191788

theorem parabola_directrix (x : ℝ) (y : ℝ) (h : y = -4 * x ^ 2 - 3) : y = - 49 / 16 := sorry

end NUMINAMATH_GPT_parabola_directrix_l1917_191788


namespace NUMINAMATH_GPT_y_value_for_equations_l1917_191766

theorem y_value_for_equations (x y : ℝ) (h1 : x^2 + y^2 = 25) (h2 : x^2 + y = 10) :
  y = (1 - Real.sqrt 61) / 2 := by
  sorry

end NUMINAMATH_GPT_y_value_for_equations_l1917_191766


namespace NUMINAMATH_GPT_profit_percentage_correct_l1917_191728

noncomputable def CP : ℝ := 460
noncomputable def SP : ℝ := 542.8
noncomputable def profit : ℝ := SP - CP
noncomputable def profit_percentage : ℝ := (profit / CP) * 100

theorem profit_percentage_correct :
  profit_percentage = 18 := by
  sorry

end NUMINAMATH_GPT_profit_percentage_correct_l1917_191728


namespace NUMINAMATH_GPT_sum_possible_x_eq_16_5_l1917_191763

open Real

noncomputable def sum_of_possible_x : Real :=
  let a := 2
  let b := -33
  let c := 87
  (-b) / (2 * a)

theorem sum_possible_x_eq_16_5 : sum_of_possible_x = 16.5 :=
  by
    -- The actual proof goes here
    sorry

end NUMINAMATH_GPT_sum_possible_x_eq_16_5_l1917_191763


namespace NUMINAMATH_GPT_algebraic_expression_evaluation_l1917_191745

-- Given condition and goal statement
theorem algebraic_expression_evaluation (a b : ℝ) (h : a - 2 * b + 3 = 0) : 5 + 2 * b - a = 8 :=
by sorry

end NUMINAMATH_GPT_algebraic_expression_evaluation_l1917_191745


namespace NUMINAMATH_GPT_time_to_pass_jogger_l1917_191783

noncomputable def jogger_speed_kmh := 9 -- in km/hr
noncomputable def train_speed_kmh := 45 -- in km/hr
noncomputable def jogger_headstart_m := 240 -- in meters
noncomputable def train_length_m := 100 -- in meters

noncomputable def kmh_to_mps (speed_kmh : ℝ) : ℝ := speed_kmh * 1000 / 3600

noncomputable def jogger_speed_mps := kmh_to_mps jogger_speed_kmh
noncomputable def train_speed_mps := kmh_to_mps train_speed_kmh
noncomputable def relative_speed := train_speed_mps - jogger_speed_mps
noncomputable def distance_to_be_covered := jogger_headstart_m + train_length_m

theorem time_to_pass_jogger : distance_to_be_covered / relative_speed = 34 := by
  sorry

end NUMINAMATH_GPT_time_to_pass_jogger_l1917_191783


namespace NUMINAMATH_GPT_valid_license_plates_count_l1917_191776

theorem valid_license_plates_count :
  let letters := 26 * 26 * 26
  let digits := 9 * 10 * 10
  letters * digits = 15818400 :=
by
  sorry

end NUMINAMATH_GPT_valid_license_plates_count_l1917_191776


namespace NUMINAMATH_GPT_function_relationship_profit_1200_max_profit_l1917_191797

namespace SalesProblem

-- Define the linear relationship between sales quantity y and selling price x
def sales_quantity (x : ℝ) : ℝ := -2 * x + 160

-- Define the cost per item
def cost_per_item := 30

-- Define the profit given selling price x and quantity y
def profit (x : ℝ) (y : ℝ) : ℝ := (x - cost_per_item) * y

-- The given data points and conditions
def data_point_1 : (ℝ × ℝ) := (35, 90)
def data_point_2 : (ℝ × ℝ) := (40, 80)

-- Prove the linear relationship between y and x
theorem function_relationship : 
  sales_quantity data_point_1.1 = data_point_1.2 ∧ 
  sales_quantity data_point_2.1 = data_point_2.2 := 
  by sorry

-- Given daily profit of 1200, proves selling price should be 50 yuan
theorem profit_1200 (x : ℝ) (h₁ : 30 ≤ x ∧ x ≤ 54) 
  (h₂ : profit x (sales_quantity x) = 1200) : 
  x = 50 := 
  by sorry

-- Prove the maximum daily profit and corresponding selling price
theorem max_profit : 
  ∃ x, 30 ≤ x ∧ x ≤ 54 ∧ (∀ y, 30 ≤ y ∧ y ≤ 54 → profit y (sales_quantity y) ≤ profit x (sales_quantity x)) ∧ 
  profit x (sales_quantity x) = 1248 := 
  by sorry

end SalesProblem

end NUMINAMATH_GPT_function_relationship_profit_1200_max_profit_l1917_191797


namespace NUMINAMATH_GPT_only_solution_is_2_3_7_l1917_191793

theorem only_solution_is_2_3_7 (a b c : ℕ) (h1 : 1 < a) (h2 : 1 < b) (h3 : 1 < c)
  (h4 : c ∣ (a * b + 1)) (h5 : a ∣ (b * c + 1)) (h6 : b ∣ (c * a + 1)) :
  (a = 2 ∧ b = 3 ∧ c = 7) ∨ (a = 3 ∧ b = 7 ∧ c = 2) ∨ (a = 7 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 7 ∧ c = 3) ∨ (a = 7 ∧ b = 3 ∧ c = 2) ∨ (a = 3 ∧ b = 2 ∧ c = 7) :=
  sorry

end NUMINAMATH_GPT_only_solution_is_2_3_7_l1917_191793


namespace NUMINAMATH_GPT_line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l1917_191705

-- Problem 1: The line passes through a fixed point
theorem line_passes_through_fixed_point (k : ℝ) : ∃ P : ℝ × ℝ, P = (1, -2) ∧ (∀ x y, k * x - y - 2 - k = 0 → P = (x, y)) :=
by
  sorry

-- Problem 2: Range of values for k if the line does not pass through the second quadrant
theorem range_of_k_no_second_quadrant (k : ℝ) : ¬ (∃ x y : ℝ, x < 0 ∧ y > 0 ∧ k * x - y - 2 - k = 0) → k ∈ Set.Ici (0) :=
by
  sorry

-- Problem 3: Minimum area of triangle AOB
theorem min_area_triangle (k : ℝ) :
  let A := (2 + k) / k
  let B := -2 - k
  (∀ x y, k * x - y - 2 - k = 0 ↔ (x = A ∧ y = 0) ∨ (x = 0 ∧ y = B)) →
  ∃ S : ℝ, S = 4 ∧ (∀ x y : ℝ, (k = 2 ∧ k * x - y - 4 = 0) → S = 4) :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_fixed_point_range_of_k_no_second_quadrant_min_area_triangle_l1917_191705


namespace NUMINAMATH_GPT_expression_evaluation_l1917_191722

theorem expression_evaluation (p q : ℝ) (h : p / q = 4 / 5) : (25 / 7 + (2 * q - p) / (2 * q + p)) = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_expression_evaluation_l1917_191722


namespace NUMINAMATH_GPT_sum_mod_13_l1917_191798

theorem sum_mod_13 :
  (9023 % 13 = 5) → 
  (9024 % 13 = 6) → 
  (9025 % 13 = 7) → 
  (9026 % 13 = 8) → 
  ((9023 + 9024 + 9025 + 9026) % 13 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_sum_mod_13_l1917_191798


namespace NUMINAMATH_GPT_meeting_time_l1917_191756

noncomputable def combined_speed : ℕ := 10 -- km/h
noncomputable def distance_to_cover : ℕ := 50 -- km
noncomputable def start_time : ℕ := 6 -- pm (in hours)
noncomputable def speed_a : ℕ := 6 -- km/h
noncomputable def speed_b : ℕ := 4 -- km/h

theorem meeting_time : start_time + (distance_to_cover / combined_speed) = 11 :=
by
  sorry

end NUMINAMATH_GPT_meeting_time_l1917_191756


namespace NUMINAMATH_GPT_triangle_inequality_l1917_191735

variables (a b c : ℝ)

theorem triangle_inequality (h₀ : a > 0) (h₁ : b > 0) (h₂ : c > 0)
  (h₃ : a + b > c) (h₄ : b + c > a) (h₅ : c + a > b) :
  (|a^2 - b^2| / c) + (|b^2 - c^2| / a) ≥ (|c^2 - a^2| / b) :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l1917_191735


namespace NUMINAMATH_GPT_swap_numbers_l1917_191721

-- Define the initial state
variables (a b c : ℕ)
axiom initial_state : a = 8 ∧ b = 17

-- Define the assignment sequence
axiom swap_statement1 : c = b 
axiom swap_statement2 : b = a
axiom swap_statement3 : a = c

-- Define the theorem to be proved
theorem swap_numbers (a b c : ℕ) (initial_state : a = 8 ∧ b = 17)
  (swap_statement1 : c = b) (swap_statement2 : b = a) (swap_statement3 : a = c) :
  (a = 17 ∧ b = 8) :=
sorry

end NUMINAMATH_GPT_swap_numbers_l1917_191721


namespace NUMINAMATH_GPT_find_page_number_l1917_191777

theorem find_page_number (n p : ℕ) (h1 : (n * (n + 1)) / 2 + 2 * p = 2046) : p = 15 :=
sorry

end NUMINAMATH_GPT_find_page_number_l1917_191777


namespace NUMINAMATH_GPT_investment_time_P_l1917_191765

-- Variables and conditions
variables {x : ℕ} {time_P : ℕ}

-- Conditions as seen from the mathematical problem
def investment_P (x : ℕ) := 7 * x
def investment_Q (x : ℕ) := 5 * x
def profit_ratio := 1 / 2
def time_Q := 14

-- Statement of the problem
theorem investment_time_P : 
  (profit_ratio = (investment_P x * time_P) / (investment_Q x * time_Q)) → 
  time_P = 5 := 
sorry

end NUMINAMATH_GPT_investment_time_P_l1917_191765


namespace NUMINAMATH_GPT_union_M_N_l1917_191743

def M : Set ℕ := {1, 2}
def N : Set ℕ := {b | ∃ a ∈ M, b = 2 * a - 1}

theorem union_M_N : M ∪ N = {1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_M_N_l1917_191743


namespace NUMINAMATH_GPT_ravi_overall_profit_l1917_191786

-- Define the purchase prices
def refrigerator_purchase_price := 15000
def mobile_phone_purchase_price := 8000

-- Define the percentages
def refrigerator_loss_percent := 2
def mobile_phone_profit_percent := 10

-- Define the calculations for selling prices
def refrigerator_loss_amount := (refrigerator_loss_percent / 100) * refrigerator_purchase_price
def refrigerator_selling_price := refrigerator_purchase_price - refrigerator_loss_amount

def mobile_phone_profit_amount := (mobile_phone_profit_percent / 100) * mobile_phone_purchase_price
def mobile_phone_selling_price := mobile_phone_purchase_price + mobile_phone_profit_amount

-- Define the total purchase and selling prices
def total_purchase_price := refrigerator_purchase_price + mobile_phone_purchase_price
def total_selling_price := refrigerator_selling_price + mobile_phone_selling_price

-- Define the overall profit calculation
def overall_profit := total_selling_price - total_purchase_price

-- Statement of the theorem
theorem ravi_overall_profit :
  overall_profit = 500 := by
  sorry

end NUMINAMATH_GPT_ravi_overall_profit_l1917_191786


namespace NUMINAMATH_GPT_steve_final_height_l1917_191748

-- Define the initial height and growth in inches
def initial_height_feet := 5
def initial_height_inches := 6
def growth_inches := 6

-- Define the conversion factors and total height after growth
def feet_to_inches (feet: Nat) := feet * 12

theorem steve_final_height : feet_to_inches initial_height_feet + initial_height_inches + growth_inches = 72 := by
  sorry

end NUMINAMATH_GPT_steve_final_height_l1917_191748


namespace NUMINAMATH_GPT_smallest_possible_N_l1917_191759

theorem smallest_possible_N (table_size N : ℕ) (h_table_size : table_size = 72) :
  (∀ seating : Finset ℕ, (seating.card = N) → (seating ⊆ Finset.range table_size) →
    ∃ i ∈ Finset.range table_size, (seating = ∅ ∨ ∃ j, (j ∈ seating) ∧ (i = (j + 1) % table_size ∨ i = (j - 1) % table_size)))
  → N = 18 :=
by sorry

end NUMINAMATH_GPT_smallest_possible_N_l1917_191759


namespace NUMINAMATH_GPT_race_distance_100_l1917_191771

noncomputable def race_distance (a b c d : ℝ) :=
  (d / a = (d - 20) / b) ∧
  (d / b = (d - 10) / c) ∧
  (d / a = (d - 28) / c) 

theorem race_distance_100 (a b c d : ℝ) (h1 : d / a = (d - 20) / b) (h2 : d / b = (d - 10) / c) (h3 : d / a = (d - 28) / c) : 
  d = 100 :=
  sorry

end NUMINAMATH_GPT_race_distance_100_l1917_191771


namespace NUMINAMATH_GPT_rebecca_has_more_eggs_than_marbles_l1917_191716

-- Given conditions
def eggs : Int := 20
def marbles : Int := 6

-- Mathematically equivalent statement to prove
theorem rebecca_has_more_eggs_than_marbles :
    eggs - marbles = 14 :=
by
    sorry

end NUMINAMATH_GPT_rebecca_has_more_eggs_than_marbles_l1917_191716


namespace NUMINAMATH_GPT_expand_expression_l1917_191757

theorem expand_expression (x : ℝ) : (7 * x + 5) * (3 * x^2) = 21 * x^3 + 15 * x^2 :=
by
  sorry

end NUMINAMATH_GPT_expand_expression_l1917_191757


namespace NUMINAMATH_GPT_final_selling_price_l1917_191734

-- Define the conditions in Lean
def cost_price_A : ℝ := 150
def profit_A_rate : ℝ := 0.20
def profit_B_rate : ℝ := 0.25

-- Define the function to calculate selling price based on cost price and profit rate
def selling_price (cost_price : ℝ) (profit_rate : ℝ) : ℝ :=
  cost_price + (profit_rate * cost_price)

-- The theorem to be proved
theorem final_selling_price :
  selling_price (selling_price cost_price_A profit_A_rate) profit_B_rate = 225 :=
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_final_selling_price_l1917_191734


namespace NUMINAMATH_GPT_total_triangles_in_grid_l1917_191726

-- Conditions
def bottom_row_triangles : Nat := 3
def next_row_triangles : Nat := 2
def top_row_triangles : Nat := 1
def additional_triangle : Nat := 1

def small_triangles := bottom_row_triangles + next_row_triangles + top_row_triangles + additional_triangle

-- Combining the triangles into larger triangles
def larger_triangles := 1 -- Formed by combining 4 small triangles
def largest_triangle := 1 -- Formed by combining all 7 small triangles

-- Math proof problem
theorem total_triangles_in_grid : small_triangles + larger_triangles + largest_triangle = 9 :=
by
  sorry

end NUMINAMATH_GPT_total_triangles_in_grid_l1917_191726


namespace NUMINAMATH_GPT_quadratic_roots_one_is_twice_l1917_191733

theorem quadratic_roots_one_is_twice (a b c : ℝ) (m : ℝ) :
  (∃ x1 x2 : ℝ, 2 * x1^2 - (2 * m + 1) * x1 + m^2 - 9 * m + 39 = 0 ∧ x2 = 2 * x1) ↔ m = 10 ∨ m = 7 :=
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_one_is_twice_l1917_191733


namespace NUMINAMATH_GPT_boat_speed_of_stream_l1917_191709

theorem boat_speed_of_stream :
  ∀ (x : ℝ), 
    (∀ s_b : ℝ, s_b = 18) → 
    (∀ d1 d2 : ℝ, d1 = 48 → d2 = 32 → d1 / (18 + x) = d2 / (18 - x)) → 
    x = 3.6 :=
by 
  intros x h_speed h_distance
  sorry

end NUMINAMATH_GPT_boat_speed_of_stream_l1917_191709


namespace NUMINAMATH_GPT_eliot_account_balance_l1917_191773

theorem eliot_account_balance 
  (A E : ℝ) 
  (h1 : A > E)
  (h2 : A - E = (1 / 12) * (A + E))
  (h3 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 :=
by 
  sorry

end NUMINAMATH_GPT_eliot_account_balance_l1917_191773


namespace NUMINAMATH_GPT_width_of_carpet_is_1000_cm_l1917_191713

noncomputable def width_of_carpet_in_cm (total_cost : ℝ) (cost_per_meter : ℝ) (length_of_room : ℝ) : ℝ :=
  let total_length_of_carpet := total_cost / cost_per_meter
  let width_of_carpet_in_meters := total_length_of_carpet / length_of_room
  width_of_carpet_in_meters * 100

theorem width_of_carpet_is_1000_cm :
  width_of_carpet_in_cm 810 4.50 18 = 1000 :=
by sorry

end NUMINAMATH_GPT_width_of_carpet_is_1000_cm_l1917_191713


namespace NUMINAMATH_GPT_no_natural_numbers_for_squares_l1917_191710

theorem no_natural_numbers_for_squares :
  ∀ x y : ℕ, ¬(∃ k m : ℕ, k^2 = x^2 + y ∧ m^2 = y^2 + x) :=
by sorry

end NUMINAMATH_GPT_no_natural_numbers_for_squares_l1917_191710


namespace NUMINAMATH_GPT_complex_div_eq_l1917_191730

open Complex

def z := 4 - 2 * I

theorem complex_div_eq :
  (z + I = 4 - I) →
  (z / (4 + 2 * I) = (3 - 4 * I) / 5) :=
by
  sorry

end NUMINAMATH_GPT_complex_div_eq_l1917_191730


namespace NUMINAMATH_GPT_determinant_of_non_right_triangle_l1917_191787

theorem determinant_of_non_right_triangle (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
(h_sum_ABC : A + B + C = π) :
  Matrix.det ![
    ![2 * Real.sin A, 1, 1],
    ![1, 2 * Real.sin B, 1],
    ![1, 1, 2 * Real.sin C]
  ] = 2 := by
  sorry

end NUMINAMATH_GPT_determinant_of_non_right_triangle_l1917_191787


namespace NUMINAMATH_GPT_inequality_proof_l1917_191752

theorem inequality_proof (x y z : ℝ) (n : ℕ) (hx_pos : 0 < x) (hy_pos : 0 < y) (hz_pos : 0 < z) (h_sum : x + y + z = 1) :
  (x^4 / (y * (1 - y^n)) + y^4 / (z * (1 - z^n)) + z^4 / (x * (1 - x^n))) ≥ (3^n) / (3^(n+2) - 9) :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l1917_191752


namespace NUMINAMATH_GPT_find_equation_l1917_191708

theorem find_equation (x : ℝ) : 
  (3 + x < 1 → false) ∧
  ((x - 67 + 63 = x - 4) → false) ∧
  ((4.8 + x = x + 4.8) → false) ∧
  (x + 0.7 = 12 → true) := 
sorry

end NUMINAMATH_GPT_find_equation_l1917_191708


namespace NUMINAMATH_GPT_range_of_m_in_inverse_proportion_function_l1917_191796

theorem range_of_m_in_inverse_proportion_function (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (1 - m) / x > 0) ∧ (x < 0 → (1 - m) / x < 0))) ↔ m < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_in_inverse_proportion_function_l1917_191796


namespace NUMINAMATH_GPT_solve_coffee_problem_l1917_191739

variables (initial_stock new_purchase : ℕ)
           (initial_decaf_percentage new_decaf_percentage : ℚ)
           (total_stock total_decaf weight_percentage_decaf : ℚ)

def coffee_problem :=
  initial_stock = 400 ∧
  initial_decaf_percentage = 0.20 ∧
  new_purchase = 100 ∧
  new_decaf_percentage = 0.50 ∧
  total_stock = initial_stock + new_purchase ∧
  total_decaf = initial_stock * initial_decaf_percentage + new_purchase * new_decaf_percentage ∧
  weight_percentage_decaf = (total_decaf / total_stock) * 100

theorem solve_coffee_problem : coffee_problem 400 100 0.20 0.50 500 130 26 :=
by {
  sorry
}

end NUMINAMATH_GPT_solve_coffee_problem_l1917_191739


namespace NUMINAMATH_GPT_polar_to_cartesian_eq_polar_circle_area_l1917_191724

theorem polar_to_cartesian_eq (p θ x y : ℝ) (h : p = 2 * Real.cos θ)
  (hx : x = p * Real.cos θ) (hy : y = p * Real.sin θ) :
  x^2 - 2 * x + y^2 = 0 := sorry

theorem polar_circle_area (p θ : ℝ) (h : p = 2 * Real.cos θ) :
  Real.pi = Real.pi := (by ring)


end NUMINAMATH_GPT_polar_to_cartesian_eq_polar_circle_area_l1917_191724


namespace NUMINAMATH_GPT_votes_cast_l1917_191714

theorem votes_cast (V : ℝ) (hv1 : 0.35 * V + (0.35 * V + 1800) = V) : V = 6000 :=
sorry

end NUMINAMATH_GPT_votes_cast_l1917_191714


namespace NUMINAMATH_GPT_perpendicular_planes_implies_perpendicular_line_l1917_191707

-- Definitions of lines and planes and their properties in space
variable {Space : Type}
variable (m n l : Line Space) -- Lines in space
variable (α β γ : Plane Space) -- Planes in space

-- Conditions: m, n, and l are non-intersecting lines, α, β, and γ are non-intersecting planes
axiom non_intersecting_lines : ¬ (m = n) ∧ ¬ (m = l) ∧ ¬ (n = l)
axiom non_intersecting_planes : ¬ (α = β) ∧ ¬ (α = γ) ∧ ¬ (β = γ)

-- To prove: if α ⊥ γ, β ⊥ γ, and α ∩ β = l, then l ⊥ γ
theorem perpendicular_planes_implies_perpendicular_line
  (h1 : α ⊥ γ) 
  (h2 : β ⊥ γ)
  (h3 : α ∩ β = l) : l ⊥ γ := 
  sorry

end NUMINAMATH_GPT_perpendicular_planes_implies_perpendicular_line_l1917_191707


namespace NUMINAMATH_GPT_hyperbolas_same_asymptotes_l1917_191784

theorem hyperbolas_same_asymptotes :
  (∀ x y, (x^2 / 4 - y^2 / 9 = 1) → (∃ k, y = k * x)) →
  (∀ x y, (y^2 / 18 - x^2 / N = 1) → (∃ k, y = k * x)) →
  N = 8 :=
by sorry

end NUMINAMATH_GPT_hyperbolas_same_asymptotes_l1917_191784


namespace NUMINAMATH_GPT_ashton_remaining_items_l1917_191700

variables (pencil_boxes : ℕ) (pens_boxes : ℕ) (pencils_per_box : ℕ) (pens_per_box : ℕ)
          (given_pencils_brother : ℕ) (distributed_pencils_friends : ℕ)
          (distributed_pens_friends : ℕ)

def total_initial_pencils := 3 * 14
def total_initial_pens := 2 * 10

def remaining_pencils := total_initial_pencils - 6 - 12
def remaining_pens := total_initial_pens - 8
def remaining_items := remaining_pencils + remaining_pens

theorem ashton_remaining_items : remaining_items = 36 :=
sorry

end NUMINAMATH_GPT_ashton_remaining_items_l1917_191700


namespace NUMINAMATH_GPT_part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l1917_191792

theorem part1_condition_represents_line (m : ℝ) :
  (m^2 - 2 * m - 3 ≠ 0) ∧ (2 * m^2 + m - 1 ≠ 0) ↔ m ≠ -1 :=
sorry

theorem part2_slope_does_not_exist (m : ℝ) :
  (m = 1 / 2) ↔ (m^2 - 2 * m - 3 = 0 ∧ (2 * m^2 + m - 1 = 0) ∧ ((1 * x = (4 / 3)))) :=
sorry

theorem part3_x_intercept (m : ℝ) :
  (2 * m - 6) / (m^2 - 2 * m - 3) = -3 ↔ m = -5 / 3 :=
sorry

theorem part4_angle_condition (m : ℝ) :
  -((m^2 - 2 * m - 3) / (2 * m^2 + m - 1)) = 1 ↔ m = 4 / 3 :=
sorry

end NUMINAMATH_GPT_part1_condition_represents_line_part2_slope_does_not_exist_part3_x_intercept_part4_angle_condition_l1917_191792


namespace NUMINAMATH_GPT_time_spent_washing_car_l1917_191782

theorem time_spent_washing_car (x : ℝ) 
  (h1 : x + (1/4) * x = 100) : x = 80 := 
sorry  

end NUMINAMATH_GPT_time_spent_washing_car_l1917_191782


namespace NUMINAMATH_GPT_fencing_required_l1917_191770

theorem fencing_required (L : ℝ) (W : ℝ) (A : ℝ) (H1 : L = 20) (H2 : A = 720) 
  (H3 : A = L * W) : L + 2 * W = 92 := by 
{
  sorry
}

end NUMINAMATH_GPT_fencing_required_l1917_191770


namespace NUMINAMATH_GPT_incorrect_weight_estimation_l1917_191723

variables (x y : ℝ)

/-- Conditions -/
def regression_equation (x : ℝ) : ℝ := 0.85 * x - 85.71

/-- Incorrect conclusion -/
theorem incorrect_weight_estimation : regression_equation 160 ≠ 50.29 :=
by 
  sorry

end NUMINAMATH_GPT_incorrect_weight_estimation_l1917_191723


namespace NUMINAMATH_GPT_probability_of_two_boys_given_one_boy_l1917_191704

-- Define the events and probabilities
def P_BB : ℚ := 1/4
def P_BG : ℚ := 1/4
def P_GB : ℚ := 1/4
def P_GG : ℚ := 1/4

def P_at_least_one_boy : ℚ := 1 - P_GG

def P_two_boys_given_at_least_one_boy : ℚ := P_BB / P_at_least_one_boy

-- Statement to be proven
theorem probability_of_two_boys_given_one_boy : P_two_boys_given_at_least_one_boy = 1/3 :=
by sorry

end NUMINAMATH_GPT_probability_of_two_boys_given_one_boy_l1917_191704
