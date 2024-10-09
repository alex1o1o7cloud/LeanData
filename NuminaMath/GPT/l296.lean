import Mathlib

namespace find_g6_minus_g2_div_g3_l296_29688

noncomputable def g : ℝ → ℝ := sorry

axiom g_condition (a c : ℝ) : c^3 * g a = a^3 * g c
axiom g_nonzero : g 3 ≠ 0

theorem find_g6_minus_g2_div_g3 : (g 6 - g 2) / g 3 = 208 / 27 := by
  sorry

end find_g6_minus_g2_div_g3_l296_29688


namespace intersection_A_B_l296_29621

def set_A (x : ℝ) : Prop := (x + 1 / 2 ≥ 3 / 2) ∨ (x + 1 / 2 ≤ -3 / 2)
def set_B (x : ℝ) : Prop := x^2 + x < 6
def A_cap_B := { x : ℝ | set_A x ∧ set_B x }

theorem intersection_A_B : A_cap_B = { x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (1 ≤ x ∧ x < 2) } :=
sorry

end intersection_A_B_l296_29621


namespace triangle_area_l296_29604

theorem triangle_area :
  let line1 (x : ℝ) := 2 * x + 1
  let line2 (x : ℝ) := (16 + x) / 4
  ∃ (base height : ℝ), height = (16 + 2 * base) / 7 ∧ base * height / 2 = 18 / 7 :=
  by
    sorry

end triangle_area_l296_29604


namespace decimal_to_binary_51_l296_29627

theorem decimal_to_binary_51 : (51 : ℕ) = 0b110011 := by sorry

end decimal_to_binary_51_l296_29627


namespace poll_total_l296_29610

-- Define the conditions
variables (men women : ℕ)
variables (pct_favor : ℝ := 35) (women_opposed : ℕ := 39)
noncomputable def total_people (men women : ℕ) : ℕ := men + women

-- We need to prove the total number of people polled, given the conditions
theorem poll_total (h1 : men = women)
  (h2 : (pct_favor / 100) * women + (39 : ℝ) / (65 / 100) = (women: ℝ)) :
  total_people men women = 120 :=
sorry

end poll_total_l296_29610


namespace tax_rate_correct_l296_29617

noncomputable def tax_rate (total_payroll : ℕ) (tax_free_payroll : ℕ) (tax_paid : ℕ) : ℚ :=
  if total_payroll > tax_free_payroll 
  then (tax_paid : ℚ) / (total_payroll - tax_free_payroll) * 100
  else 0

theorem tax_rate_correct :
  tax_rate 400000 200000 400 = 0.2 :=
by
  sorry

end tax_rate_correct_l296_29617


namespace each_player_gets_seven_l296_29629

-- Define the total number of dominoes and players
def total_dominoes : Nat := 28
def total_players : Nat := 4

-- Define the question for how many dominoes each player would receive
def dominoes_per_player (dominoes players : Nat) : Nat := dominoes / players

-- The theorem to prove each player gets 7 dominoes
theorem each_player_gets_seven : dominoes_per_player total_dominoes total_players = 7 :=
by
  sorry

end each_player_gets_seven_l296_29629


namespace min_value_expression_l296_29654

theorem min_value_expression : ∃ x y z : ℝ, (3 * x^2 + 3 * x * y + y^2 - 3 * x + 3 * y + z^2 + 6 * z + 10) = -7 / 2 :=
by sorry

end min_value_expression_l296_29654


namespace scientific_notation_570_million_l296_29692

theorem scientific_notation_570_million:
  (570 * 10^6 : ℝ) = (5.7 * 10^8 : ℝ) :=
sorry

end scientific_notation_570_million_l296_29692


namespace find_b_when_a_equals_neg10_l296_29602

theorem find_b_when_a_equals_neg10 
  (ab_k : ∀ a b : ℝ, (a * b) = 675) 
  (sum_60 : ∀ a b : ℝ, (a + b = 60 → a = 3 * b)) 
  (a_eq_neg10 : ∀ a : ℝ, a = -10) : 
  ∃ b : ℝ, b = -67.5 := 
by 
  sorry

end find_b_when_a_equals_neg10_l296_29602


namespace units_digit_N_l296_29695

def P (n : ℕ) : ℕ := (n / 10) * (n % 10)
def S (n : ℕ) : ℕ := (n / 10) + (n % 10)

theorem units_digit_N (N : ℕ) (h1 : 10 ≤ N ∧ N ≤ 99) (h2 : N = P N + S N) : N % 10 = 9 :=
by
  sorry

end units_digit_N_l296_29695


namespace greendale_points_l296_29605

theorem greendale_points : 
  let roosevelt_game1 := 30 
  let roosevelt_game2 := roosevelt_game1 / 2
  let roosevelt_game3 := roosevelt_game2 * 3
  let roosevelt_bonus := 50
  let greendale_diff := 10
  let roosevelt_total := roosevelt_game1 + roosevelt_game2 + roosevelt_game3 + roosevelt_bonus
  let greendale_total := roosevelt_total - greendale_diff
  greendale_total = 130 :=
by
  sorry

end greendale_points_l296_29605


namespace choose_president_and_committee_l296_29678

-- Define the condition of the problem
def total_people := 10
def committee_size := 3

-- Define the function to calculate the number of combinations
def comb (n k : ℕ) : ℕ := Nat.choose n k

-- Proving the number of ways to choose the president and the committee
theorem choose_president_and_committee :
  (total_people * comb (total_people - 1) committee_size) = 840 :=
by
  sorry

end choose_president_and_committee_l296_29678


namespace problem1_l296_29657

theorem problem1 : 2 * Real.sin (Real.pi / 3) - 3 * Real.tan (Real.pi / 6) = 0 := by
  sorry

end problem1_l296_29657


namespace sin4x_eq_sin2x_solution_set_l296_29636

noncomputable def solution_set (x : ℝ) : Prop :=
  0 < x ∧ x < (3 / 2) * Real.pi ∧ Real.sin (4 * x) = Real.sin (2 * x)

theorem sin4x_eq_sin2x_solution_set :
  { x : ℝ | solution_set x } =
  { (Real.pi / 6), (Real.pi / 2), Real.pi, (5 * Real.pi / 6), (7 * Real.pi / 6) } :=
by
  sorry

end sin4x_eq_sin2x_solution_set_l296_29636


namespace probability_red_is_two_fifths_l296_29631

-- Define the durations
def red_light_duration : ℕ := 30
def yellow_light_duration : ℕ := 5
def green_light_duration : ℕ := 40

-- Define total cycle duration
def total_cycle_duration : ℕ :=
  red_light_duration + yellow_light_duration + green_light_duration

-- Define the probability function
def probability_of_red_light : ℚ :=
  red_light_duration / total_cycle_duration

-- The theorem statement to prove
theorem probability_red_is_two_fifths :
  probability_of_red_light = 2/5 := sorry

end probability_red_is_two_fifths_l296_29631


namespace hh3_eq_6582_l296_29666

def h (x : ℤ) : ℤ := 3 * x^2 + 5 * x + 4

theorem hh3_eq_6582 : h (h 3) = 6582 :=
by
  sorry

end hh3_eq_6582_l296_29666


namespace square_side_length_l296_29677

theorem square_side_length (x : ℝ) (h : 4 * x = 2 * x^2) : x = 2 :=
by 
  sorry

end square_side_length_l296_29677


namespace solve_for_x_l296_29689

-- Define the given equation as a hypothesis
def equation (x : ℝ) : Prop :=
  0.05 * x - 0.09 * (25 - x) = 5.4

-- State the theorem that x = 54.6428571 satisfies the given equation
theorem solve_for_x : (x : ℝ) → equation x → x = 54.6428571 :=
by
  sorry

end solve_for_x_l296_29689


namespace max_value_of_seq_diff_l296_29639

theorem max_value_of_seq_diff :
  ∀ (a : Fin 2017 → ℝ),
    a 0 = a 2016 →
    (∀ i : Fin 2015, |a i + a (i+2) - 2 * a (i+1)| ≤ 1) →
    ∃ b : ℝ, b = 508032 ∧ ∀ i j, 1 ≤ i → i < j → j ≤ 2017 → |a i - a j| ≤ b :=
  sorry

end max_value_of_seq_diff_l296_29639


namespace solve_equation_in_nat_l296_29611

theorem solve_equation_in_nat {x y : ℕ} :
  (x - 1) / (1 + (x - 1) * y) + (y - 1) / (2 * y - 1) = x / (x + 1) →
  x = 2 ∧ y = 2 :=
by
  sorry

end solve_equation_in_nat_l296_29611


namespace find_f1_l296_29646

variable {R : Type*} [LinearOrderedField R]

-- Define function f of the form px + q
def f (p q x : R) : R := p * x + q

-- Given conditions
variables (p q : R)

-- Define the equations from given conditions
def cond1 : Prop := (f p q 3) = 5
def cond2 : Prop := (f p q 5) = 9

theorem find_f1 (hpq1 : cond1 p q) (hpq2 : cond2 p q) : f p q 1 = 1 := sorry

end find_f1_l296_29646


namespace point_distance_l296_29649

theorem point_distance (x y n : ℝ) 
    (h1 : abs x = 8) 
    (h2 : (x - 3)^2 + (y - 10)^2 = 225) 
    (h3 : y > 10) 
    (hn : n = Real.sqrt (x^2 + y^2)) : 
    n = Real.sqrt (364 + 200 * Real.sqrt 2) := 
sorry

end point_distance_l296_29649


namespace simplify_expression_l296_29606

variable (x : ℝ)

theorem simplify_expression : 
  (3 * x + 6 - 5 * x) / 3 = (-2 / 3) * x + 2 := by
  sorry

end simplify_expression_l296_29606


namespace unique_chair_arrangement_l296_29619

theorem unique_chair_arrangement (n : ℕ) (h : n = 49)
  (h1 : ∀ i j : ℕ, (n = i * j) → (i ≥ 2) ∧ (j ≥ 2)) :
  ∃! i j : ℕ, (n = i * j) ∧ (i ≥ 2) ∧ (j ≥ 2) :=
by
  sorry

end unique_chair_arrangement_l296_29619


namespace find_k_value_l296_29633

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 5 * x^2 + 3 * x + 7
noncomputable def g (x : ℝ) (k : ℝ) : ℝ := 3 * x^3 - k * x^2 + 4

theorem find_k_value : (f 5 - g 5 k = 45) → k = 27 / 25 :=
by
  intro h
  sorry

end find_k_value_l296_29633


namespace problem_statement_l296_29641

theorem problem_statement (a b c x : ℝ) (h1 : a + x^2 = 2015) (h2 : b + x^2 = 2016)
    (h3 : c + x^2 = 2017) (h4 : a * b * c = 24) :
    (a / (b * c) + b / (a * c) + c / (a * b) - (1 / a) - (1 / b) - (1 / c) = 1 / 8) :=
by
  sorry

end problem_statement_l296_29641


namespace paving_cost_l296_29653

theorem paving_cost (length width rate : ℝ) (h_length : length = 8) (h_width : width = 4.75) (h_rate : rate = 900) :
  length * width * rate = 34200 :=
by
  rw [h_length, h_width, h_rate]
  norm_num

end paving_cost_l296_29653


namespace profit_share_difference_l296_29669

noncomputable def ratio (x y : ℕ) : ℕ := x / Nat.gcd x y

def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1900
def total_parts : ℕ := 15  -- Sum of the ratio parts (4 for A, 5 for B, 6 for C)
def part_amount : ℕ := profit_share_B / 5  -- 5 parts of B

def profit_share_A : ℕ := 4 * part_amount
def profit_share_C : ℕ := 6 * part_amount

theorem profit_share_difference :
  (profit_share_C - profit_share_A) = 760 := by
  sorry

end profit_share_difference_l296_29669


namespace otto_knives_l296_29690

theorem otto_knives (n : ℕ) (cost : ℕ) : 
  cost = 32 → 
  (n ≥ 1 → cost = 5 + ((min (n - 1) 3) * 4) + ((max 0 (n - 4)) * 3)) → 
  n = 9 :=
by
  intros h_cost h_structure
  sorry

end otto_knives_l296_29690


namespace percentage_of_pushups_l296_29642

-- Problem conditions as definitions
def jumpingJacks := 12
def pushups := 8
def situps := 20
def totalExercises := jumpingJacks + pushups + situps

-- Question and the proof goal
theorem percentage_of_pushups : 
  (pushups / totalExercises : ℝ) * 100 = 20 := by
  sorry

end percentage_of_pushups_l296_29642


namespace average_age_of_town_population_l296_29652

theorem average_age_of_town_population
  (children adults : ℕ)
  (ratio_condition : 3 * adults = 2 * children)
  (avg_age_children : ℕ := 10)
  (avg_age_adults : ℕ := 40) :
  ((10 * children + 40 * adults) / (children + adults) = 22) :=
by
  sorry

end average_age_of_town_population_l296_29652


namespace right_angle_triangle_iff_arithmetic_progression_l296_29671

noncomputable def exists_right_angle_triangle_with_rational_sides_and_area (d : ℤ) : Prop :=
  ∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)

noncomputable def rational_squares_in_arithmetic_progression (x y z : ℚ) : Prop :=
  2 * y^2 = x^2 + z^2

theorem right_angle_triangle_iff_arithmetic_progression (d : ℤ) :
  (∃ (a b c : ℚ), (a^2 + b^2 = c^2) ∧ (a * b = 2 * d)) ↔ ∃ (x y z : ℚ), rational_squares_in_arithmetic_progression x y z :=
sorry

end right_angle_triangle_iff_arithmetic_progression_l296_29671


namespace express_y_in_terms_of_x_l296_29697

-- Defining the parameters and assumptions
variables (x y : ℝ)
variables (h : x * y = 30)

-- Stating the theorem
theorem express_y_in_terms_of_x (h : x * y = 30) : y = 30 / x :=
sorry

end express_y_in_terms_of_x_l296_29697


namespace pencils_total_l296_29667

/-- The students in class 5A had a total of 2015 pencils. One of them lost a box containing five pencils and replaced it with a box containing 50 pencils. Prove the final number of pencils is 2060. -/
theorem pencils_total {initial_pencils lost_pencils gained_pencils final_pencils : ℕ} 
  (h1 : initial_pencils = 2015) 
  (h2 : lost_pencils = 5) 
  (h3 : gained_pencils = 50) 
  (h4 : final_pencils = (initial_pencils - lost_pencils + gained_pencils)) 
  : final_pencils = 2060 :=
sorry

end pencils_total_l296_29667


namespace technicians_count_l296_29683

def avg_salary_all := 9500
def avg_salary_technicians := 12000
def avg_salary_rest := 6000
def total_workers := 12

theorem technicians_count : 
  ∃ (T R : ℕ), 
  (T + R = total_workers) ∧ 
  ((T * avg_salary_technicians + R * avg_salary_rest) / total_workers = avg_salary_all) ∧ 
  (T = 7) :=
by sorry

end technicians_count_l296_29683


namespace dihedral_angle_of_equilateral_triangle_l296_29601

theorem dihedral_angle_of_equilateral_triangle (a : ℝ) 
(ABC_eq : ∀ {A B C : ℝ}, (B - A) ^ 2 + (C - A) ^ 2 = a^2 ∧ (C - B) ^ 2 + (A - B) ^ 2 = a^2 ∧ (A - C) ^ 2 + (B - C) ^ 2 = a^2) 
(perpendicular : ∀ A B C D : ℝ, D = (B + C)/2 ∧ (B - D) * (C - D) = 0) : 
∃ θ : ℝ, θ = 60 := 
  sorry

end dihedral_angle_of_equilateral_triangle_l296_29601


namespace divide_square_into_equal_parts_l296_29625

-- Given a square with four shaded smaller squares inside
structure SquareWithShaded (n : ℕ) :=
  (squares : Fin n → Fin n → Prop) -- this models the presence of shaded squares
  (shaded : (Fin 2) → (Fin 2) → Prop)

-- To prove: we can divide the square into four equal parts with each containing one shaded square
theorem divide_square_into_equal_parts :
  ∀ (sq : SquareWithShaded 4),
  ∃ (parts : Fin 2 → Fin 2 → Prop),
  (∀ i j, parts i j ↔ 
    ((i = 0 ∧ j = 0) ∨ (i = 1 ∧ j = 0) ∨ 
    (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 1)) ∧
    (∃! k l, sq.shaded k l ∧ parts i j)) :=
sorry

end divide_square_into_equal_parts_l296_29625


namespace average_gpa_difference_2_l296_29676

def avg_gpa_6th_grader := 93
def avg_gpa_8th_grader := 91
def school_avg_gpa := 93

noncomputable def gpa_diff (gpa_7th_grader diff : ℝ) (avg6 avg8 school_avg : ℝ) := 
  gpa_7th_grader = avg6 + diff ∧ 
  (avg6 + gpa_7th_grader + avg8) / 3 = school_avg

theorem average_gpa_difference_2 (x : ℝ) : 
  (∃ G : ℝ, gpa_diff G x avg_gpa_6th_grader avg_gpa_8th_grader school_avg_gpa) → x = 2 :=
by
  sorry

end average_gpa_difference_2_l296_29676


namespace pencils_bought_l296_29658

theorem pencils_bought (cindi_spent : ℕ) (cost_per_pencil : ℕ) 
  (cindi_pencils : ℕ) 
  (marcia_pencils : ℕ) 
  (donna_pencils : ℕ) :
  cindi_spent = 30 → 
  cost_per_pencil = 1/2 → 
  cindi_pencils = cindi_spent / cost_per_pencil → 
  marcia_pencils = 2 * cindi_pencils → 
  donna_pencils = 3 * marcia_pencils → 
  donna_pencils + marcia_pencils = 480 := 
by
  sorry

end pencils_bought_l296_29658


namespace log_2_bounds_l296_29640

theorem log_2_bounds:
  (2^9 = 512) → (2^8 = 256) → (10^2 = 100) → (10^3 = 1000) → 
  (2 / 9 < Real.log 2 / Real.log 10) ∧ (Real.log 2 / Real.log 10 < 3 / 8) :=
by
  intros h1 h2 h3 h4
  sorry

end log_2_bounds_l296_29640


namespace mike_average_points_per_game_l296_29603

theorem mike_average_points_per_game (total_points games_played points_per_game : ℕ) 
  (h1 : games_played = 6) 
  (h2 : total_points = 24) 
  (h3 : total_points = games_played * points_per_game) : 
  points_per_game = 4 :=
by
  rw [h1, h2] at h3  -- Substitute conditions h1 and h2 into the equation
  sorry  -- the proof goes here

end mike_average_points_per_game_l296_29603


namespace num_bikes_l296_29607

variable (C B : ℕ)

-- The given conditions
def num_cars : ℕ := 10
def num_wheels_total : ℕ := 44
def wheels_per_car : ℕ := 4
def wheels_per_bike : ℕ := 2

-- The mathematical proof problem statement
theorem num_bikes :
  C = num_cars →
  B = ((num_wheels_total - (C * wheels_per_car)) / wheels_per_bike) →
  B = 2 :=
by
  intros hC hB
  rw [hC] at hB
  sorry

end num_bikes_l296_29607


namespace derivative_of_exp_sin_l296_29650

theorem derivative_of_exp_sin (x : ℝ) : 
  (deriv (fun x => Real.exp x * Real.sin x)) x = Real.exp x * Real.sin x + Real.exp x * Real.cos x :=
sorry

end derivative_of_exp_sin_l296_29650


namespace min_flight_routes_l296_29643

-- Defining a problem of connecting cities with flight routes such that 
-- every city can be reached from any other city with no more than two layovers.
theorem min_flight_routes (n : ℕ) (h : n = 50) : ∃ (r : ℕ), (r = 49) ∧
  (∀ (c1 c2 : ℕ), c1 ≠ c2 → c1 < n → c2 < n → ∃ (a b : ℕ),
    a < n ∧ b < n ∧ (a = c1 ∨ a = c2) ∧ (b = c1 ∨ b = c2) ∧
    ((c1 = a ∧ c2 = b) ∨ (c1 = a ∧ b = c2) ∨ (a = c2 ∧ b = c1))) :=
by {
  sorry
}

end min_flight_routes_l296_29643


namespace arithmetic_sum_S8_proof_l296_29684

-- Definitions of variables and constants
variables (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
def a1_condition : a 1 = -40 := sorry
def a6_a10_condition : a 6 + a 10 = -10 := sorry

-- Theorem to prove
theorem arithmetic_sum_S8_proof (a : ℕ → ℝ) (S : ℕ → ℝ)
  (a1 : a 1 = -40)
  (a6a10 : a 6 + a 10 = -10)
  : S 8 = -180 := 
sorry

end arithmetic_sum_S8_proof_l296_29684


namespace find_c_plus_d_l296_29672

theorem find_c_plus_d (c d : ℝ) :
  (∀ x y, (x = (1 / 3) * y + c) → (y = (1 / 3) * x + d) → (x, y) = (3, 3)) → 
  c + d = 4 :=
by
  -- ahead declaration to meet the context requirements in Lean 4
  intros h
  -- Proof steps would go here, but they are omitted
  sorry

end find_c_plus_d_l296_29672


namespace cards_given_to_Jeff_l296_29674

theorem cards_given_to_Jeff
  (initial_cards : ℕ)
  (cards_given_to_John : ℕ)
  (remaining_cards : ℕ)
  (cards_left : ℕ)
  (h_initial : initial_cards = 573)
  (h_given_John : cards_given_to_John = 195)
  (h_left_before_Jeff : remaining_cards = initial_cards - cards_given_to_John)
  (h_final : cards_left = 210)
  (h_given_Jeff : remaining_cards - cards_left = 168) :
  (initial_cards - cards_given_to_John - cards_left = 168) :=
by
  sorry

end cards_given_to_Jeff_l296_29674


namespace positive_integer_solution_eq_l296_29694

theorem positive_integer_solution_eq :
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ (xyz + 2 * x + 3 * y + 6 * z = xy + 2 * xz + 3 * yz) ∧ (x, y, z) = (4, 3, 1) := 
by
  sorry

end positive_integer_solution_eq_l296_29694


namespace max_value_of_expression_l296_29614

theorem max_value_of_expression (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 8) : 
  (1 + x) * (1 + y) ≤ 25 :=
by
  sorry

end max_value_of_expression_l296_29614


namespace sum_of_squares_l296_29637

theorem sum_of_squares (n : ℕ) (h : n * (n + 1) * (n + 2) = 12 * (3 * n + 3)) :
  n^2 + (n + 1)^2 + (n + 2)^2 = 29 := 
sorry

end sum_of_squares_l296_29637


namespace interest_rate_is_12_percent_l296_29655

-- Definitions
def SI : ℝ := 5400
def P : ℝ := 15000
def T : ℝ := 3

-- Theorem to prove the interest rate
theorem interest_rate_is_12_percent :
  SI = (P * 12 * T) / 100 :=
by
  sorry

end interest_rate_is_12_percent_l296_29655


namespace cube_volume_l296_29682

theorem cube_volume (A : ℝ) (s : ℝ) (V : ℝ) (hA : A = 864) (hA_def : A = 6 * s^2) (hs : s = 12) :
  V = 12^3 :=
by
  -- Given the conditions
  sorry

end cube_volume_l296_29682


namespace parallel_lines_l296_29687

-- Definitions based on the conditions
def line1 (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y - 2 = 0
def line2 (m : ℝ) (x y : ℝ) : Prop := 5 * x + (m + 3) * y - 5 = 0
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), line1 m x y → line2 m x y

-- The theorem to be proved
theorem parallel_lines (m : ℝ) (h : parallel m) : m = -5 := 
by
  sorry

end parallel_lines_l296_29687


namespace range_of_a_l296_29622

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |x + 3| - |x + 1| ≤ a) → a ≥ 2 :=
by 
  intro h
  sorry

end range_of_a_l296_29622


namespace range_of_a_l296_29612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 - x^2 + x + 2
noncomputable def g (x : ℝ) : ℝ := (Real.exp 1 * Real.log x) / x

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, (0 < x1 ∧ x1 ≤ 1 ∧ 0 < x2 ∧ x2 ≤ 1) → f a x1 ≥ g x2) →
  a ≥ -2 :=
sorry

end range_of_a_l296_29612


namespace part1_monotonic_intervals_part2_range_of_a_l296_29693

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x * Real.log x - x + 0.5

theorem part1_monotonic_intervals (x : ℝ) : 
  (f 1 x < (f 1 (x + 1)) ↔ x < 1) ∧ 
  (f 1 x > (f 1 (x - 1)) ↔ x > 1) :=
by sorry

theorem part2_range_of_a (a : ℝ) (x : ℝ) (hx : 1 < x ∧ x ≤ Real.exp 1) 
  (h : (f a x / x) + (1 / (2 * x)) < 0) : 
  a < 1 - (1 / Real.exp 1) :=
by sorry

end part1_monotonic_intervals_part2_range_of_a_l296_29693


namespace correct_number_for_question_mark_l296_29645

def first_row := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 200]
def second_row_no_quest := [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]
def question_mark (x : ℕ) := first_row.sum = second_row_no_quest.sum + x

theorem correct_number_for_question_mark : question_mark 155 := 
by sorry -- proof to be completed

end correct_number_for_question_mark_l296_29645


namespace opposite_of_neg_two_is_two_l296_29623

def is_opposite (a b : Int) : Prop := a + b = 0

theorem opposite_of_neg_two_is_two : is_opposite (-2) 2 :=
by
  sorry

end opposite_of_neg_two_is_two_l296_29623


namespace find_time_interval_l296_29665

-- Definitions for conditions
def birthRate : ℕ := 4
def deathRate : ℕ := 2
def netIncreaseInPopulationPerInterval (T : ℕ) : ℕ := birthRate - deathRate
def totalTimeInOneDay : ℕ := 86400
def netIncreaseInOneDay (T : ℕ) : ℕ := (totalTimeInOneDay / T) * (netIncreaseInPopulationPerInterval T)

-- Theorem statement
theorem find_time_interval (T : ℕ) (h1 : netIncreaseInPopulationPerInterval T = 2) (h2 : netIncreaseInOneDay T = 86400) : T = 2 :=
sorry

end find_time_interval_l296_29665


namespace determine_age_l296_29656

def David_age (D Y : ℕ) : Prop := Y = 2 * D ∧ Y = D + 7

theorem determine_age (D : ℕ) (h : David_age D (D + 7)) : D = 7 :=
by
  sorry

end determine_age_l296_29656


namespace length_of_chord_AB_l296_29618

noncomputable def right_focus : ℝ × ℝ := (Real.sqrt 3, 0)
noncomputable def line_eq (x : ℝ) := x - Real.sqrt 3
noncomputable def ellipse_eq (x y : ℝ)  := x^2 / 4 + y^2 = 1

theorem length_of_chord_AB :
  ∀ (A B : ℝ × ℝ), 
  (line_eq A.1 = A.2) → 
  (line_eq B.1 = B.2) → 
  (ellipse_eq A.1 A.2) → 
  (ellipse_eq B.1 B.2) → 
  ∃ d : ℝ, d = 8 / 5 ∧ 
  dist A B = d := 
sorry

end length_of_chord_AB_l296_29618


namespace gazprom_rd_expense_l296_29661

theorem gazprom_rd_expense
  (R_and_D_t : ℝ) (ΔAPL_t_plus_1 : ℝ)
  (h1 : R_and_D_t = 3289.31)
  (h2 : ΔAPL_t_plus_1 = 1.55) :
  R_and_D_t / ΔAPL_t_plus_1 = 2122 := 
by
  sorry

end gazprom_rd_expense_l296_29661


namespace ellipse_hyperbola_foci_l296_29699

theorem ellipse_hyperbola_foci {a b : ℝ} (h1 : b^2 - a^2 = 25) (h2 : a^2 + b^2 = 49) :
  a = 2 * Real.sqrt 3 ∧ b = Real.sqrt 37 :=
by sorry

end ellipse_hyperbola_foci_l296_29699


namespace license_plates_count_l296_29615

theorem license_plates_count :
  let vowels := 5 -- choices for the first vowel
  let other_letters := 25 -- choices for the second and third letters
  let digits := 10 -- choices for each digit
  (vowels * other_letters * other_letters * (digits * digits * digits)) = 3125000 :=
by
  -- proof steps will go here
  sorry

end license_plates_count_l296_29615


namespace correct_answers_l296_29647

-- Definitions
variable (C W : ℕ)
variable (h1 : C + W = 120)
variable (h2 : 3 * C - W = 180)

-- Goal statement
theorem correct_answers : C = 75 :=
by
  sorry

end correct_answers_l296_29647


namespace quadratic_has_real_roots_l296_29673

theorem quadratic_has_real_roots (k : ℝ) : (∃ x : ℝ, k * x^2 - 6 * x + 9 = 0) ↔ (k ≤ 1 ∧ k ≠ 0) :=
by
  sorry

end quadratic_has_real_roots_l296_29673


namespace Δy_over_Δx_l296_29681

-- Conditions
def f (x : ℝ) : ℝ := 2 * x^2 - 4
def y1 : ℝ := f 1
def y2 (Δx : ℝ) : ℝ := f (1 + Δx)
def Δy (Δx : ℝ) : ℝ := y2 Δx - y1

-- Theorem statement
theorem Δy_over_Δx (Δx : ℝ) : Δy Δx / Δx = 4 + 2 * Δx := by
  sorry

end Δy_over_Δx_l296_29681


namespace trig_identity_l296_29632

theorem trig_identity (α : ℝ) :
  1 - Real.cos (2 * α - Real.pi) + Real.cos (4 * α - 2 * Real.pi) =
  4 * Real.cos (2 * α) * Real.cos (Real.pi / 6 + α) * Real.cos (Real.pi / 6 - α) :=
by
  sorry

end trig_identity_l296_29632


namespace tom_marbles_l296_29628

def jason_marbles := 44
def marbles_difference := 20

theorem tom_marbles : (jason_marbles - marbles_difference = 24) :=
by
  sorry

end tom_marbles_l296_29628


namespace students_only_one_activity_l296_29608

theorem students_only_one_activity 
  (total : ℕ) (both : ℕ) (neither : ℕ)
  (h_total : total = 317) 
  (h_both : both = 30) 
  (h_neither : neither = 20) : 
  (total - both - neither) = 267 :=
by 
  sorry

end students_only_one_activity_l296_29608


namespace purchase_price_is_600_l296_29670

open Real

def daily_food_cost : ℝ := 20
def num_days : ℝ := 40
def vaccination_cost : ℝ := 500
def selling_price : ℝ := 2500
def profit : ℝ := 600

def total_food_cost : ℝ := daily_food_cost * num_days
def total_expenses : ℝ := total_food_cost + vaccination_cost
def total_cost : ℝ := selling_price - profit
def purchase_price : ℝ := total_cost - total_expenses

theorem purchase_price_is_600 : purchase_price = 600 := by
  sorry

end purchase_price_is_600_l296_29670


namespace peter_son_is_nikolay_l296_29635

variable (x y : ℕ)

/-- Within the stated scenarios of Nikolai/Peter paired fishes caught -/
theorem peter_son_is_nikolay :
  (∀ n p ns ps : ℕ, (
    n = ns ∧              -- Nikolai caught as many fish as his son
    p = 3 * ps ∧          -- Peter caught three times more fish than his son
    n + ns + p + ps = 25  -- A total of 25 fish were caught
  ) → ("Nikolay" = "Peter's son")) := 
sorry

end peter_son_is_nikolay_l296_29635


namespace altitudes_sum_eq_l296_29660

variables {α : Type*} [LinearOrderedField α]

structure Triangle (α) :=
(A B C : α)
(R : α)   -- circumradius
(r : α)   -- inradius

variables (T : Triangle α)
(A B C : α)
(m n p : α)  -- points on respective arcs
(h1 h2 h3 : α)  -- altitudes of the segments

theorem altitudes_sum_eq (T : Triangle α) (A B C m n p h1 h2 h3 : α) :
  h1 + h2 + h3 = 2 * T.R - T.r :=
sorry

end altitudes_sum_eq_l296_29660


namespace number_satisfies_equation_l296_29626

theorem number_satisfies_equation :
  ∃ x : ℝ, (0.6667 * x - 10 = 0.25 * x) ∧ (x = 23.9936) :=
by
  sorry

end number_satisfies_equation_l296_29626


namespace no_point_in_common_l296_29659

theorem no_point_in_common (b : ℝ) :
  (∀ (x y : ℝ), y = 2 * x + b → (x^2 / 4) + y^2 ≠ 1) ↔ (b < -2 * Real.sqrt 2 ∨ b > 2 * Real.sqrt 2) :=
by
  sorry

end no_point_in_common_l296_29659


namespace min_words_to_learn_l296_29662

theorem min_words_to_learn (n : ℕ) (p_guess : ℝ) (required_score : ℝ)
  (h_n : n = 600) (h_p : p_guess = 0.1) (h_score : required_score = 0.9) :
  ∃ x : ℕ, (x + p_guess * (n - x)) / n ≥ required_score ∧ x = 534 :=
by
  sorry

end min_words_to_learn_l296_29662


namespace race_time_A_l296_29648

theorem race_time_A (v t : ℝ) (h1 : 1000 = v * t) (h2 : 950 = v * (t - 10)) : t = 200 :=
by
  sorry

end race_time_A_l296_29648


namespace evaluate_expression_zero_l296_29698

-- Define the variables and conditions
def x : ℕ := 4
def z : ℕ := 0

-- State the property to be proved
theorem evaluate_expression_zero : z * (2 * z - 5 * x) = 0 := by
  sorry

end evaluate_expression_zero_l296_29698


namespace train_speed_is_correct_l296_29664

-- Definitions of the given conditions.
def train_length : ℕ := 250
def bridge_length : ℕ := 150
def time_taken : ℕ := 20

-- Definition of the total distance covered by the train.
def total_distance : ℕ := train_length + bridge_length

-- The speed calculation.
def speed : ℕ := total_distance / time_taken

-- The theorem that we need to prove.
theorem train_speed_is_correct : speed = 20 := by
  -- proof steps go here
  sorry

end train_speed_is_correct_l296_29664


namespace number_added_at_end_l296_29651

theorem number_added_at_end :
  (26.3 * 12 * 20) / 3 + 125 = 2229 := sorry

end number_added_at_end_l296_29651


namespace simplify_expression_l296_29609

theorem simplify_expression :
  (Real.sqrt 600 / Real.sqrt 75 - Real.sqrt 243 / Real.sqrt 108) = (4 * Real.sqrt 2 - 3 * Real.sqrt 3) / 2 := by
  sorry

end simplify_expression_l296_29609


namespace radius_of_circle_l296_29620

theorem radius_of_circle (a b : ℝ) (h1 : a ≥ 0) (h2 : b ≥ a) : 
  ∃ R, R = (b - a) / 2 ∨ R = (b + a) / 2 :=
by {
  sorry
}

end radius_of_circle_l296_29620


namespace three_digit_numbers_not_multiples_of_3_or_11_l296_29696

def count_multiples (a b : ℕ) (lower upper : ℕ) : ℕ :=
  (upper - lower) / b + 1

theorem three_digit_numbers_not_multiples_of_3_or_11 : 
  let total := 900
  let multiples_3 := count_multiples 3 3 102 999
  let multiples_11 := count_multiples 11 11 110 990
  let multiples_33 := count_multiples 33 33 132 990
  let multiples_3_or_11 := multiples_3 + multiples_11 - multiples_33
  total - multiples_3_or_11 = 546 := 
by 
  sorry

end three_digit_numbers_not_multiples_of_3_or_11_l296_29696


namespace no_such_function_l296_29675

theorem no_such_function : ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, |f (x + y) + Real.sin x + Real.sin y| < 2 := 
sorry

end no_such_function_l296_29675


namespace min_third_side_length_l296_29668

theorem min_third_side_length (a b : ℝ) (ha : a = 7) (hb : b = 24) : 
  ∃ c : ℝ, (a^2 + b^2 = c^2 ∨ b^2 = a^2 + c^2 ∨  a^2 = b^2 + c^2) ∧ c = 7 :=
sorry

end min_third_side_length_l296_29668


namespace average_marks_is_75_l296_29638

-- Define the scores for the four tests based on the given conditions.
def first_test : ℕ := 80
def second_test : ℕ := first_test + 10
def third_test : ℕ := 65
def fourth_test : ℕ := third_test

-- Define the total marks scored in the four tests.
def total_marks : ℕ := first_test + second_test + third_test + fourth_test

-- Number of tests.
def num_tests : ℕ := 4

-- Define the average marks scored in the four tests.
def average_marks : ℕ := total_marks / num_tests

-- Prove that the average marks scored in the four tests is 75.
theorem average_marks_is_75 : average_marks = 75 :=
by
  sorry

end average_marks_is_75_l296_29638


namespace total_squares_in_4x4_grid_l296_29616

-- Define the grid size
def grid_size : ℕ := 4

-- Define a function to count the number of k x k squares in an n x n grid
def count_squares (n k : ℕ) : ℕ :=
  (n - k + 1) * (n - k + 1)

-- Total number of squares in a 4 x 4 grid
def total_squares (n : ℕ) : ℕ :=
  count_squares n 1 + count_squares n 2 + count_squares n 3 + count_squares n 4

-- The main theorem asserting the total number of squares in a 4 x 4 grid is 30
theorem total_squares_in_4x4_grid : total_squares grid_size = 30 := by
  sorry

end total_squares_in_4x4_grid_l296_29616


namespace smallest_sum_of_squares_l296_29679

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 145) : x^2 + y^2 = 433 :=
sorry

end smallest_sum_of_squares_l296_29679


namespace part_I_part_II_l296_29630

-- Define the sets A and B for the given conditions
def setA : Set ℝ := {x | -3 ≤ x - 2 ∧ x - 2 ≤ 1}
def setB (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ a + 2}

-- Part (Ⅰ) When a = 1, find A ∩ B
theorem part_I (a : ℝ) (ha : a = 1) :
  (setA ∩ setB a) = {x | 0 ≤ x ∧ x ≤ 3} :=
by
  sorry

-- Part (Ⅱ) If A ∪ B = A, find the range of real number a
theorem part_II : 
  (∀ a : ℝ, setA ∪ setB a = setA → 0 ≤ a ∧ a ≤ 1) :=
by
  sorry

end part_I_part_II_l296_29630


namespace blake_bought_six_chocolate_packs_l296_29613

-- Defining the conditions as hypotheses
variables (lollipops : ℕ) (lollipopCost : ℕ) (packCost : ℕ)
          (cashGiven : ℕ) (changeReceived : ℕ)
          (totalSpent : ℕ) (totalLollipopCost : ℕ) (amountSpentOnChocolates : ℕ)

-- Assertion of the values based on the conditions
axiom h1 : lollipops = 4
axiom h2 : lollipopCost = 2
axiom h3 : packCost = lollipops * lollipopCost
axiom h4 : cashGiven = 6 * 10
axiom h5 : changeReceived = 4
axiom h6 : totalSpent = cashGiven - changeReceived
axiom h7 : totalLollipopCost = lollipops * lollipopCost
axiom h8 : amountSpentOnChocolates = totalSpent - totalLollipopCost
axiom chocolatePacks : ℕ
axiom h9 : chocolatePacks = amountSpentOnChocolates / packCost

-- The statement to be proved
theorem blake_bought_six_chocolate_packs :
    chocolatePacks = 6 :=
by
  subst_vars
  sorry

end blake_bought_six_chocolate_packs_l296_29613


namespace combined_work_rate_l296_29686

theorem combined_work_rate (x_rate y_rate : ℚ) (h1 : x_rate = 1 / 15) (h2 : y_rate = 1 / 45) :
    1 / (x_rate + y_rate) = 11.25 :=
by
  -- Proof goes here
  sorry

end combined_work_rate_l296_29686


namespace base4_to_base10_conversion_l296_29634

theorem base4_to_base10_conversion : 
  (1 * 4^3 + 2 * 4^2 + 1 * 4^1 + 2 * 4^0) = 102 :=
by
  sorry

end base4_to_base10_conversion_l296_29634


namespace total_pears_picked_is_correct_l296_29680

-- Define the number of pears picked by Sara and Sally
def pears_picked_by_Sara : ℕ := 45
def pears_picked_by_Sally : ℕ := 11

-- The total number of pears picked
def total_pears_picked := pears_picked_by_Sara + pears_picked_by_Sally

-- The theorem statement: prove that the total number of pears picked is 56
theorem total_pears_picked_is_correct : total_pears_picked = 56 := by
  sorry

end total_pears_picked_is_correct_l296_29680


namespace total_percentage_failed_exam_l296_29600

theorem total_percentage_failed_exam :
  let total_candidates := 2000
  let general_candidates := 1000
  let obc_candidates := 600
  let sc_candidates := 300
  let st_candidates := total_candidates - (general_candidates + obc_candidates + sc_candidates)
  let general_pass_percentage := 0.35
  let obc_pass_percentage := 0.50
  let sc_pass_percentage := 0.25
  let st_pass_percentage := 0.30
  let general_failed := general_candidates - (general_candidates * general_pass_percentage)
  let obc_failed := obc_candidates - (obc_candidates * obc_pass_percentage)
  let sc_failed := sc_candidates - (sc_candidates * sc_pass_percentage)
  let st_failed := st_candidates - (st_candidates * st_pass_percentage)
  let total_failed := general_failed + obc_failed + sc_failed + st_failed
  let failed_percentage := (total_failed / total_candidates) * 100
  failed_percentage = 62.25 :=
by
  sorry

end total_percentage_failed_exam_l296_29600


namespace find_n_modulo_l296_29644

theorem find_n_modulo (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 11) (h3 : n ≡ 15827 [ZMOD 12]) : n = 11 :=
by
  sorry

end find_n_modulo_l296_29644


namespace sum_of_squares_of_two_numbers_l296_29685

theorem sum_of_squares_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 30) :
  x^2 + y^2 = 840 :=
by
  sorry

end sum_of_squares_of_two_numbers_l296_29685


namespace time_per_page_l296_29624

theorem time_per_page 
    (planning_time : ℝ := 3) 
    (fraction : ℝ := 3/4) 
    (pages_read : ℕ := 9) 
    (minutes_per_hour : ℕ := 60) : 
    (fraction * planning_time * minutes_per_hour) / pages_read = 15 := 
by
  sorry

end time_per_page_l296_29624


namespace vehicle_speeds_l296_29663

theorem vehicle_speeds (V_A V_B V_C : ℝ) (d_AB d_AC : ℝ) (decel_A : ℝ)
  (V_A_eff : ℝ) (delta_V_A : ℝ) :
  V_A = 70 → V_B = 50 → V_C = 65 →
  decel_A = 5 → V_A_eff = V_A - decel_A → 
  d_AB = 40 → d_AC = 250 →
  delta_V_A = 10 →
  (d_AB / (V_A_eff + delta_V_A - V_B) < d_AC / (V_A_eff + delta_V_A + V_C)) :=
by
  intros hVA hVB hVC hdecel hV_A_eff hdAB hdAC hdelta_V_A
  -- the proof would be filled in here
  sorry

end vehicle_speeds_l296_29663


namespace balance_increase_second_year_l296_29691

variable (initial_deposit : ℝ) (balance_first_year : ℝ) 
variable (total_percentage_increase : ℝ)

theorem balance_increase_second_year
  (h1 : initial_deposit = 1000)
  (h2 : balance_first_year = 1100)
  (h3 : total_percentage_increase = 0.32) : 
  (balance_first_year + (initial_deposit * total_percentage_increase) - balance_first_year) / balance_first_year * 100 = 20 :=
by
  sorry

end balance_increase_second_year_l296_29691
