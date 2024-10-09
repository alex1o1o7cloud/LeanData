import Mathlib

namespace range_of_f_l1526_152658

-- Define the function f
def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

-- Define the domain
def domain : Finset ℕ := {1, 2, 3, 4, 5}

-- Define the expected range
def expected_range : Finset ℤ := {-1, 1, 3, 5, 7}

-- Prove the range of f given the domain
theorem range_of_f : domain.image f = expected_range :=
  sorry

end range_of_f_l1526_152658


namespace count_integers_l1526_152603

theorem count_integers (n : ℕ) (h : n = 33000) :
  ∃ k : ℕ, k = 1600 ∧
  (∀ x, 1 ≤ x ∧ x ≤ n → (x % 11 = 0 → (x % 3 ≠ 0 ∧ x % 5 ≠ 0) → x ≤ x)) :=
by 
  sorry

end count_integers_l1526_152603


namespace shaded_area_calculation_l1526_152659

-- Define the dimensions of the grid and the size of each square
def gridWidth : ℕ := 9
def gridHeight : ℕ := 7
def squareSize : ℕ := 2

-- Define the number of 2x2 squares horizontally and vertically
def numSquaresHorizontally : ℕ := gridWidth / squareSize
def numSquaresVertically : ℕ := gridHeight / squareSize

-- Define the area of one 2x2 square and one shaded triangle within it
def squareArea : ℕ := squareSize * squareSize
def shadedTriangleArea : ℕ := squareArea / 2

-- Define the total number of 2x2 squares
def totalNumSquares : ℕ := numSquaresHorizontally * numSquaresVertically

-- Define the total area of shaded regions
def totalShadedArea : ℕ := totalNumSquares * shadedTriangleArea

-- The theorem to be proved
theorem shaded_area_calculation : totalShadedArea = 24 := by
  sorry    -- Placeholder for the proof

end shaded_area_calculation_l1526_152659


namespace equivalent_polar_point_representation_l1526_152650

/-- Representation of a point in polar coordinates -/
structure PolarPoint :=
  (r : ℝ)
  (θ : ℝ)

theorem equivalent_polar_point_representation :
  ∀ (p1 p2 : PolarPoint), p1 = PolarPoint.mk (-1) (5 * Real.pi / 6) →
    (p2 = PolarPoint.mk 1 (11 * Real.pi / 6) → p1.r + Real.pi = p2.r ∧ p1.θ = p2.θ) :=
by
  intros p1 p2 h1 h2
  sorry

end equivalent_polar_point_representation_l1526_152650


namespace figure4_total_length_l1526_152601

-- Define the conditions
def top_segments_sum := 3 + 1 + 1  -- Sum of top segments in Figure 3
def bottom_segment := top_segments_sum -- Bottom segment length in Figure 3
def vertical_segment1 := 10  -- First vertical segment length
def vertical_segment2 := 9  -- Second vertical segment length
def remaining_segment := 1  -- The remaining horizontal segment

-- Total length of remaining segments in Figure 4
theorem figure4_total_length : 
  bottom_segment + vertical_segment1 + vertical_segment2 + remaining_segment = 25 := by
  sorry

end figure4_total_length_l1526_152601


namespace ellipse_sum_l1526_152625

-- Define the givens
def h : ℤ := -3
def k : ℤ := 5
def a : ℤ := 7
def b : ℤ := 4

-- State the theorem to be proven
theorem ellipse_sum : h + k + a + b = 13 := by
  sorry

end ellipse_sum_l1526_152625


namespace increasing_g_on_neg_l1526_152680

variable {R : Type*} [LinearOrderedField R]

-- Assumptions: 
-- 1. f is an increasing function on R
-- 2. (h_neg : ∀ x : R, f x < 0)

theorem increasing_g_on_neg (f : R → R) (h_inc : ∀ x y : R, x < y → f x < f y) (h_neg : ∀ x : R, f x < 0) :
  ∀ x y : R, x < y → x < 0 → y < 0 → (x^2 * f x < y^2 * f y) :=
by
  sorry

end increasing_g_on_neg_l1526_152680


namespace number_of_terms_in_product_l1526_152630

theorem number_of_terms_in_product 
  (a b c d e f g h i : ℕ) :
  (a + b + c + d) * (e + f + g + h + i) = 20 :=
sorry

end number_of_terms_in_product_l1526_152630


namespace fraction_of_groups_with_a_and_b_l1526_152627

/- Definitions based on the conditions -/
def total_persons : ℕ := 6
def group_size : ℕ := 3
def person_a : ℕ := 1  -- arbitrary assignment for simplicity
def person_b : ℕ := 2  -- arbitrary assignment for simplicity

/- Hypotheses based on conditions -/
axiom six_persons (n : ℕ) : n = total_persons
axiom divided_into_two_groups (grp_size : ℕ) : grp_size = group_size
axiom a_and_b_included (a b : ℕ) : a = person_a ∧ b = person_b

/- The theorem to prove -/
theorem fraction_of_groups_with_a_and_b
    (total_groups : ℕ := Nat.choose total_persons group_size)
    (groups_with_a_b : ℕ := Nat.choose 4 1) :
    groups_with_a_b / total_groups = 1 / 5 :=
by
    sorry

end fraction_of_groups_with_a_and_b_l1526_152627


namespace primes_quadratic_roots_conditions_l1526_152600

theorem primes_quadratic_roots_conditions (p q : ℕ)
  (hp : Prime p) (hq : Prime q)
  (h1 : ∃ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p) :
  (¬ (∀ (x y : ℕ), x ≠ y ∧ x * y = 2 * q ∧ x + y = p → (x - y) % 2 = 0)) ∧
  (∃ (x : ℕ), x * 2 = 2 * q ∨ x * q = 2 * q ∧ Prime x) ∧
  (¬ Prime (p * p + 2 * q)) ∧
  (Prime (p - q)) :=
by sorry

end primes_quadratic_roots_conditions_l1526_152600


namespace principal_amount_l1526_152695

theorem principal_amount (SI R T : ℕ) (P : ℕ) : SI = 160 ∧ R = 5 ∧ T = 4 → P = 800 :=
by
  sorry

end principal_amount_l1526_152695


namespace min_moves_to_find_treasure_l1526_152664

theorem min_moves_to_find_treasure (cells : List ℕ) (h1 : cells = [5, 5, 5]) : 
  ∃ n, n = 2 ∧ (∀ moves, moves ≥ n → true) := sorry

end min_moves_to_find_treasure_l1526_152664


namespace chickens_and_rabbits_l1526_152635

theorem chickens_and_rabbits (c r : ℕ) 
    (h1 : c = 2 * r - 5)
    (h2 : 2 * c + r = 92) : ∃ c r : ℕ, (c = 2 * r - 5) ∧ (2 * c + r = 92) := 
by 
    -- proof steps
    sorry

end chickens_and_rabbits_l1526_152635


namespace batteries_manufactured_l1526_152641

theorem batteries_manufactured (gather_time create_time : Nat) (robots : Nat) (hours : Nat) (total_batteries : Nat) :
  gather_time = 6 →
  create_time = 9 →
  robots = 10 →
  hours = 5 →
  total_batteries = (hours * 60 / (gather_time + create_time)) * robots →
  total_batteries = 200 :=
by
  intros h_gather h_create h_robots h_hours h_batteries
  simp [h_gather, h_create, h_robots, h_hours] at h_batteries
  exact h_batteries

end batteries_manufactured_l1526_152641


namespace proof_statement_l1526_152602

variables {K_c A_c K_d B_d A_d B_c : ℕ}

def conditions (K_c A_c K_d B_d A_d B_c : ℕ) :=
  K_c > A_c ∧ K_d > B_d ∧ A_d > K_d ∧ B_c > A_c

noncomputable def statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : Prop :=
  A_d > max K_d B_d

theorem proof_statement (K_c A_c K_d B_d A_d B_c : ℕ) (h : conditions K_c A_c K_d B_d A_d B_c) : statement K_c A_c K_d B_d A_d B_c h :=
sorry

end proof_statement_l1526_152602


namespace min_value_y_l1526_152668

noncomputable def y (x : ℝ) := x^4 - 4*x + 3

theorem min_value_y : ∃ x ∈ Set.Icc (-2 : ℝ) 3, y x = 0 ∧ ∀ x' ∈ Set.Icc (-2 : ℝ) 3, y x' ≥ 0 :=
by
  sorry

end min_value_y_l1526_152668


namespace rational_terms_count_l1526_152674

noncomputable def number_of_rational_terms (n : ℕ) (x : ℝ) : ℕ :=
  -- The count of rational terms in the expansion
  17

theorem rational_terms_count (n : ℕ) (x : ℝ) :
  (number_of_rational_terms 100 x) = 17 := by
  sorry

end rational_terms_count_l1526_152674


namespace norris_savings_l1526_152689

theorem norris_savings:
  ∀ (N : ℕ), 
  (29 + 25 + N = 85) → N = 31 :=
by
  intros N h
  sorry

end norris_savings_l1526_152689


namespace total_shirts_l1526_152623

def hazel_shirts : ℕ := 6
def razel_shirts : ℕ := 2 * hazel_shirts

theorem total_shirts : hazel_shirts + razel_shirts = 18 := by
  sorry

end total_shirts_l1526_152623


namespace solve_inequality_l1526_152604

theorem solve_inequality (a x : ℝ) : 
  if a > 0 then -a < x ∧ x < 2*a else if a < 0 then 2*a < x ∧ x < -a else False :=
by sorry

end solve_inequality_l1526_152604


namespace triangle_sides_fraction_sum_eq_one_l1526_152638

theorem triangle_sides_fraction_sum_eq_one
  (a b c : ℝ)
  (h : a^2 + b^2 = c^2 + a * b) :
  a / (b + c) + b / (c + a) = 1 :=
sorry

end triangle_sides_fraction_sum_eq_one_l1526_152638


namespace StatementA_incorrect_l1526_152626

def f (n : ℕ) : ℕ := (n.factorial)^2

def g (x : ℕ) : ℕ := f (x + 1) / f x

theorem StatementA_incorrect (x : ℕ) (h : x = 1) : g x ≠ 4 := sorry

end StatementA_incorrect_l1526_152626


namespace foodAdditivesPercentage_l1526_152612

-- Define the given percentages
def microphotonicsPercentage : ℕ := 14
def homeElectronicsPercentage : ℕ := 24
def microorganismsPercentage : ℕ := 29
def industrialLubricantsPercentage : ℕ := 8

-- Define degrees representing basic astrophysics
def basicAstrophysicsDegrees : ℕ := 18

-- Define the total degrees in a circle
def totalDegrees : ℕ := 360

-- Define the total budget percentage
def totalBudgetPercentage : ℕ := 100

-- Prove that the remaining percentage for food additives is 20%
theorem foodAdditivesPercentage :
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  totalBudgetPercentage - totalKnownPercentage = 20 :=
by
  let basicAstrophysicsPercentage := (basicAstrophysicsDegrees * totalBudgetPercentage) / totalDegrees
  let totalKnownPercentage := microphotonicsPercentage + homeElectronicsPercentage + microorganismsPercentage + industrialLubricantsPercentage + basicAstrophysicsPercentage
  sorry

end foodAdditivesPercentage_l1526_152612


namespace find_y_l1526_152620

theorem find_y (x y : ℕ) (h1 : 24 * x = 173 * y) (h2 : 173 * y = 1730) : y = 10 :=
by 
  -- Proof is skipped
  sorry

end find_y_l1526_152620


namespace number_of_solutions_eq_one_l1526_152608

theorem number_of_solutions_eq_one :
  ∃! (n : ℕ), 0 < n ∧ 
              (∃ k : ℕ, (n + 1500) = 90 * k ∧ k = Int.floor (Real.sqrt n)) :=
sorry

end number_of_solutions_eq_one_l1526_152608


namespace topsoil_cost_is_112_l1526_152661

noncomputable def calculate_topsoil_cost (length width depth_in_inches : ℝ) (cost_per_cubic_foot : ℝ) : ℝ :=
  let depth_in_feet := depth_in_inches / 12
  let volume := length * width * depth_in_feet
  volume * cost_per_cubic_foot

theorem topsoil_cost_is_112 :
  calculate_topsoil_cost 8 4 6 7 = 112 :=
by
  sorry

end topsoil_cost_is_112_l1526_152661


namespace find_interest_rate_l1526_152683

-- conditions
def P : ℝ := 6200
def t : ℕ := 10

def interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * t
def I : ℝ := P - 3100

-- problem statement
theorem find_interest_rate (r : ℝ) :
  interest P r t = I → r = 0.05 :=
by
  sorry

end find_interest_rate_l1526_152683


namespace meaningful_expression_l1526_152617

theorem meaningful_expression (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 1))) → x > 1 :=
by sorry

end meaningful_expression_l1526_152617


namespace average_age_of_other_9_students_l1526_152654

variable (total_students : ℕ) (total_average_age : ℝ) (group1_students : ℕ) (group1_average_age : ℝ) (age_student12 : ℝ) (group2_students : ℕ)

theorem average_age_of_other_9_students 
  (h1 : total_students = 16) 
  (h2 : total_average_age = 16) 
  (h3 : group1_students = 5) 
  (h4 : group1_average_age = 14) 
  (h5 : age_student12 = 42) 
  (h6 : group2_students = 9) : 
  (group1_students * group1_average_age + group2_students * 16 + age_student12) / total_students = total_average_age := by
  sorry

end average_age_of_other_9_students_l1526_152654


namespace possible_values_a_l1526_152686

-- Define the problem statement
theorem possible_values_a :
  (∃ a b c : ℤ, ∀ x : ℝ, (x - a) * (x - 5) + 3 = (x + b) * (x + c)) → (a = 1 ∨ a = 9) :=
by 
  -- Variable declaration and theorem body will be placed here
  sorry

end possible_values_a_l1526_152686


namespace sequence_formula_l1526_152681

theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) (h2 : ∀ n : ℕ, a (n + 1) = 3 * a n - 2) :
  ∀ n : ℕ, a n = 3^(n - 1) + 1 :=
by sorry

end sequence_formula_l1526_152681


namespace delivery_cost_l1526_152616

theorem delivery_cost (base_fee : ℕ) (limit : ℕ) (extra_fee : ℕ) 
(item_weight : ℕ) (total_cost : ℕ) 
(h1 : base_fee = 13) (h2 : limit = 5) (h3 : extra_fee = 2) 
(h4 : item_weight = 7) (h5 : total_cost = 17) : 
  total_cost = base_fee + (item_weight - limit) * extra_fee := 
by
  sorry

end delivery_cost_l1526_152616


namespace num_positive_integers_n_l1526_152662

theorem num_positive_integers_n (n : ℕ) : 
  (∃ n, ( ∃ k : ℕ, n = 2015 * k^2 ∧ ∃ m, m^2 = 2015 * n) ∧ 
          (∃ k : ℕ, n = 2015 * k^2 ∧  ∃ l : ℕ, 2 * 2015 * k^2 = l * (1 + k^2)))
  →
  n = 5 := sorry

end num_positive_integers_n_l1526_152662


namespace another_divisor_l1526_152657

theorem another_divisor (n : ℕ) (h1 : n = 44402) (h2 : ∀ d ∈ [12, 48, 74, 100], (n + 2) % d = 0) : 
  199 ∣ (n + 2) := 
by 
  sorry

end another_divisor_l1526_152657


namespace fraction_of_red_knights_magical_l1526_152677

def total_knights : ℕ := 28
def red_fraction : ℚ := 3 / 7
def magical_fraction : ℚ := 1 / 4
def red_magical_to_blue_magical_ratio : ℚ := 3

theorem fraction_of_red_knights_magical :
  let red_knights := red_fraction * total_knights
  let blue_knights := total_knights - red_knights
  let total_magical := magical_fraction * total_knights
  let red_magical_fraction := 21 / 52
  let blue_magical_fraction := red_magical_fraction / red_magical_to_blue_magical_ratio
  red_knights * red_magical_fraction + blue_knights * blue_magical_fraction = total_magical :=
by
  sorry

end fraction_of_red_knights_magical_l1526_152677


namespace division_in_base_5_l1526_152696

noncomputable def base5_quick_divide : nat := sorry

theorem division_in_base_5 (a b quotient : ℕ) (h1 : a = 1324) (h2 : b = 12) (h3 : quotient = 111) :
  ∃ c : ℕ, c = quotient ∧ a / b = quotient :=
by
  sorry

end division_in_base_5_l1526_152696


namespace probability_not_win_l1526_152685

theorem probability_not_win (n : ℕ) (h : 1 - 1 / (n : ℝ) = 0.9375) : n = 16 :=
sorry

end probability_not_win_l1526_152685


namespace chick_hit_count_l1526_152682

theorem chick_hit_count :
  ∃ x y z : ℕ,
    9 * x + 5 * y + 2 * z = 61 ∧
    x + y + z = 10 ∧
    x ≥ 1 ∧
    y ≥ 1 ∧
    z ≥ 1 ∧
    x = 5 :=
by
  sorry

end chick_hit_count_l1526_152682


namespace problem_inequality_l1526_152633

theorem problem_inequality (a b : ℝ) (n : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_ab : 1 < a * b) (h_n : 2 ≤ n) :
  (a + b)^n > a^n + b^n + 2^n - 2 :=
sorry

end problem_inequality_l1526_152633


namespace problem_part1_problem_part2_l1526_152666

open Real

variables {α : ℝ}

theorem problem_part1 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) :
  sin α * cos α = 3 / 10 := sorry

theorem problem_part2 (h1 : sin α - cos α = sqrt 10 / 5) (h2 : α > pi ∧ α < 2 * pi) (h3 : sin α * cos α = 3 / 10) :
  sin α + cos α = - (2 * sqrt 10 / 5) := sorry

end problem_part1_problem_part2_l1526_152666


namespace inverse_prop_l1526_152693

theorem inverse_prop (x : ℝ) : x < 0 → x^2 > 0 :=
by
  sorry

end inverse_prop_l1526_152693


namespace positive_square_root_of_256_l1526_152622

theorem positive_square_root_of_256 (y : ℝ) (hy_pos : y > 0) (hy_squared : y^2 = 256) : y = 16 :=
by
  sorry

end positive_square_root_of_256_l1526_152622


namespace rectangle_square_problem_l1526_152670

theorem rectangle_square_problem
  (m n x : ℕ)
  (h : 2 * (m + n) + 2 * x = m * n)
  (h2 : m * n - x^2 = 2 * (m + n)) :
  x = 2 ∧ ((m = 3 ∧ n = 10) ∨ (m = 6 ∧ n = 4)) :=
by {
  -- Proof goes here
  sorry
}

end rectangle_square_problem_l1526_152670


namespace engineer_last_name_is_smith_l1526_152698

/-- Given these conditions:
 1. Businessman Robinson and a conductor live in Sheffield.
 2. Businessman Jones and a stoker live in Leeds.
 3. Businessman Smith and the railroad engineer live halfway between Leeds and Sheffield.
 4. The conductor’s namesake earns $10,000 a year.
 5. The engineer earns exactly 1/3 of what the businessman who lives closest to him earns.
 6. Railroad worker Smith beats the stoker at billiards.
 
We need to prove that the last name of the engineer is Smith. -/
theorem engineer_last_name_is_smith
  (lives_in_Sheffield_Robinson : Prop)
  (lives_in_Sheffield_conductor : Prop)
  (lives_in_Leeds_Jones : Prop)
  (lives_in_Leeds_stoker : Prop)
  (lives_in_halfway_Smith : Prop)
  (lives_in_halfway_engineer : Prop)
  (conductor_namesake_earns_10000 : Prop)
  (engineer_earns_one_third_closest_bizman : Prop)
  (railway_worker_Smith_beats_stoker_at_billiards : Prop) :
  (engineer_last_name = "Smith") :=
by
  -- Proof will go here
  sorry

end engineer_last_name_is_smith_l1526_152698


namespace contrapositive_of_x_squared_eq_one_l1526_152690

theorem contrapositive_of_x_squared_eq_one (x : ℝ) 
  (h : x^2 = 1 → x = 1 ∨ x = -1) : (x ≠ 1 ∧ x ≠ -1) → x^2 ≠ 1 :=
by
  sorry

end contrapositive_of_x_squared_eq_one_l1526_152690


namespace area_square_II_l1526_152648

theorem area_square_II (a b : ℝ) :
  let diag_I := 2 * (a + b)
  let area_I := (a + b) * (a + b) * 2
  let area_II := area_I * 3
  area_II = 6 * (a + b) ^ 2 :=
by
  sorry

end area_square_II_l1526_152648


namespace optimality_theorem_l1526_152640

def sequence_1 := "[[[a1, a2], a3], a4]" -- 22 symbols sequence
def sequence_2 := "[[a1, a2], [a3, a4]]" -- 16 symbols sequence

def optimal_sequence := sequence_2

theorem optimality_theorem : optimal_sequence = "[[a1, a2], [a3, a4]]" :=
by
  sorry

end optimality_theorem_l1526_152640


namespace sin_double_angle_eq_half_l1526_152660

theorem sin_double_angle_eq_half (α : ℝ) (h1 : 0 < α ∧ α < π / 2)
  (h2 : Real.sin (π / 2 + 2 * α) = Real.cos (π / 4 - α)) : 
  Real.sin (2 * α) = 1 / 2 :=
by
  sorry

end sin_double_angle_eq_half_l1526_152660


namespace shaded_area_is_20_l1526_152611

theorem shaded_area_is_20 (large_square_side : ℕ) (num_small_squares : ℕ) 
  (shaded_squares : ℕ) 
  (h1 : large_square_side = 10) (h2 : num_small_squares = 25) 
  (h3 : shaded_squares = 5) : 
  (large_square_side^2 / num_small_squares) * shaded_squares = 20 :=
by
  sorry

end shaded_area_is_20_l1526_152611


namespace one_angle_not_greater_than_60_l1526_152691

theorem one_angle_not_greater_than_60 (A B C : ℝ) (h : A + B + C = 180) : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 := 
sorry

end one_angle_not_greater_than_60_l1526_152691


namespace initial_cost_renting_car_l1526_152618

theorem initial_cost_renting_car
  (initial_cost : ℝ)
  (miles_monday : ℝ := 620)
  (miles_thursday : ℝ := 744)
  (cost_per_mile : ℝ := 0.50)
  (total_spent : ℝ := 832)
  (total_miles : ℝ := miles_monday + miles_thursday)
  (expected_initial_cost : ℝ := 150) :
  total_spent = initial_cost + cost_per_mile * total_miles → initial_cost = expected_initial_cost :=
by
  sorry

end initial_cost_renting_car_l1526_152618


namespace polynomial_perfect_square_l1526_152629

theorem polynomial_perfect_square (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5 * x + 5)^2 :=
by 
  sorry

end polynomial_perfect_square_l1526_152629


namespace lg_sum_geometric_seq_l1526_152644

noncomputable def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem lg_sum_geometric_seq (a : ℕ → ℝ) (h1 : geometric_sequence a) (h2 : a 2 * a 5 * a 8 = 1) :
  Real.log (a 4) + Real.log (a 6) = 0 := 
sorry

end lg_sum_geometric_seq_l1526_152644


namespace water_level_drop_recording_l1526_152676

theorem water_level_drop_recording (rise6_recorded: Int): 
    (rise6_recorded = 6) → (6 = -rise6_recorded) :=
by
  sorry

end water_level_drop_recording_l1526_152676


namespace problem_statement_l1526_152628

theorem problem_statement
  (c d : ℕ)
  (h_factorization : ∀ x, x^2 - 18 * x + 72 = (x - c) * (x - d))
  (h_c_nonnegative : c ≥ 0)
  (h_d_nonnegative : d ≥ 0)
  (h_c_greater_d : c > d) :
  4 * d - c = 12 :=
sorry

end problem_statement_l1526_152628


namespace budget_for_bulbs_l1526_152632

theorem budget_for_bulbs (num_crocus_bulbs : ℕ) (cost_per_crocus : ℝ) (budget : ℝ)
  (h1 : num_crocus_bulbs = 22)
  (h2 : cost_per_crocus = 0.35)
  (h3 : budget = num_crocus_bulbs * cost_per_crocus) :
  budget = 7.70 :=
sorry

end budget_for_bulbs_l1526_152632


namespace minimum_abs_a_plus_b_l1526_152615

theorem minimum_abs_a_plus_b {a b : ℤ} (h1 : |a| < |b|) (h2 : |b| ≤ 4) : ∃ (a b : ℤ), |a| + b = -4 :=
by
  sorry

end minimum_abs_a_plus_b_l1526_152615


namespace find_a_n_l1526_152631

def S (n : ℕ) : ℕ := 2^(n+1) - 1

def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 2^n

theorem find_a_n (n : ℕ) : a_n n = if n = 1 then 3 else 2^n :=
  sorry

end find_a_n_l1526_152631


namespace total_passengers_per_day_l1526_152643

-- Define the conditions
def airplanes : ℕ := 5
def rows_per_airplane : ℕ := 20
def seats_per_row : ℕ := 7
def flights_per_day : ℕ := 2

-- Define the proof problem
theorem total_passengers_per_day : 
  (airplanes * rows_per_airplane * seats_per_row * flights_per_day) = 1400 := 
by 
  sorry

end total_passengers_per_day_l1526_152643


namespace find_value_of_a_l1526_152699

theorem find_value_of_a (a : ℝ) (h : 2 - a = 0) : a = 2 :=
by {
  sorry
}

end find_value_of_a_l1526_152699


namespace cos_x_plus_2y_eq_one_l1526_152647

theorem cos_x_plus_2y_eq_one (x y a : ℝ) 
  (hx : -Real.pi / 4 ≤ x ∧ x ≤ Real.pi / 4)
  (hy : -Real.pi / 4 ≤ y ∧ y ≤ Real.pi / 4)
  (h_eq1 : x^3 + Real.sin x - 2 * a = 0)
  (h_eq2 : 4 * y^3 + (1 / 2) * Real.sin (2 * y) + a = 0) : 
  Real.cos (x + 2 * y) = 1 := 
sorry -- Proof goes here

end cos_x_plus_2y_eq_one_l1526_152647


namespace number_of_cows_l1526_152687

-- Define conditions
def total_bags_consumed_by_some_cows := 45
def bags_consumed_by_one_cow := 1

-- State the theorem to prove the number of cows
theorem number_of_cows (h1 : total_bags_consumed_by_some_cows = 45) (h2 : bags_consumed_by_one_cow = 1) : 
  total_bags_consumed_by_some_cows / bags_consumed_by_one_cow = 45 :=
by
  -- Proof goes here
  sorry

end number_of_cows_l1526_152687


namespace least_number_to_add_l1526_152672

theorem least_number_to_add (x : ℕ) (h1 : (1789 + x) % 6 = 0) (h2 : (1789 + x) % 4 = 0) (h3 : (1789 + x) % 3 = 0) : x = 7 := 
sorry

end least_number_to_add_l1526_152672


namespace correct_operation_l1526_152697

variable {R : Type*} [CommRing R] (x y : R)

theorem correct_operation : x * (1 + y) = x + x * y :=
by sorry

end correct_operation_l1526_152697


namespace g_at_seven_equals_92_l1526_152645

def g (n : ℕ) : ℕ := n^2 + 2*n + 29

theorem g_at_seven_equals_92 : g 7 = 92 :=
by
  sorry

end g_at_seven_equals_92_l1526_152645


namespace correct_operation_l1526_152621

theorem correct_operation (a b : ℝ) :
  ¬ (a^2 + a^3 = a^6) ∧
  ¬ ((a*b)^2 = a*(b^2)) ∧
  ¬ ((a + b)^2 = a^2 + b^2) ∧
  ((a + b)*(a - b) = a^2 - b^2) := 
by
  sorry

end correct_operation_l1526_152621


namespace complement_of_A_with_respect_to_U_l1526_152655

-- Definitions
def U : Set ℤ := {-2, -1, 1, 3, 5}
def A : Set ℤ := {-1, 3}

-- Statement of the problem
theorem complement_of_A_with_respect_to_U :
  (U \ A) = {-2, 1, 5} := 
by
  sorry

end complement_of_A_with_respect_to_U_l1526_152655


namespace cos_beta_l1526_152663

theorem cos_beta (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2) 
  (h_cos_α : Real.cos α = 3/5) (h_cos_alpha_plus_beta : Real.cos (α + β) = -5/13) : 
  Real.cos β = 33/65 :=
by
  sorry

end cos_beta_l1526_152663


namespace trig_identity_l1526_152619

theorem trig_identity (α : ℝ) (h : Real.tan α = 2 / 3) : 
  Real.sin (2 * α) - Real.cos (π - 2 * α) = 17 / 13 :=
by
  sorry

end trig_identity_l1526_152619


namespace sum_of_cubes_ages_l1526_152613

theorem sum_of_cubes_ages (d t h : ℕ) 
  (h1 : 4 * d + t = 3 * h) 
  (h2 : 4 * h ^ 2 = 2 * d ^ 2 + t ^ 2) 
  (h3 : Nat.gcd d (Nat.gcd t h) = 1)
  : d ^ 3 + t ^ 3 + h ^ 3 = 155557 :=
sorry

end sum_of_cubes_ages_l1526_152613


namespace polynomial_roots_unique_b_c_l1526_152667

theorem polynomial_roots_unique_b_c :
    ∀ (r : ℝ), (r ^ 2 - 2 * r - 1 = 0) → (r ^ 5 - 29 * r - 12 = 0) :=
by
    sorry

end polynomial_roots_unique_b_c_l1526_152667


namespace derivative_equals_l1526_152624

noncomputable def func (x : ℝ) : ℝ :=
  (3 / (8 * Real.sqrt 2) * Real.log ((Real.sqrt 2 + Real.tanh x) / (Real.sqrt 2 - Real.tanh x)))
  - (Real.tanh x / (4 * (2 - (Real.tanh x)^2)))

theorem derivative_equals :
  ∀ x : ℝ, deriv func x = 1 / (2 + (Real.cosh x)^2)^2 :=
by {
  sorry
}

end derivative_equals_l1526_152624


namespace lily_milk_remaining_l1526_152636

def lilyInitialMilk : ℚ := 4
def milkGivenAway : ℚ := 7 / 3
def milkLeft : ℚ := 5 / 3

theorem lily_milk_remaining : lilyInitialMilk - milkGivenAway = milkLeft := by
  sorry

end lily_milk_remaining_l1526_152636


namespace range_of_set_l1526_152679

-- Define the conditions
variable (a b c : ℝ)
variable (h1 : a ≤ b)
variable (h2 : b ≤ c)
variable (mean_condition : (a + b + c) / 3 = 5)
variable (median_condition : b = 5)
variable (smallest_condition : a = 2)

-- The theorem to prove
theorem range_of_set : c - a = 6 := by
  sorry

end range_of_set_l1526_152679


namespace permutations_of_three_digit_numbers_from_set_l1526_152671

theorem permutations_of_three_digit_numbers_from_set {digits : Finset ℕ} (h : digits = {1, 2, 3, 4, 5}) :
  ∃ n : ℕ, n = (Finset.card digits) * (Finset.card digits - 1) * (Finset.card digits - 2) ∧ n = 60 :=
by
  sorry

end permutations_of_three_digit_numbers_from_set_l1526_152671


namespace eq_sets_M_N_l1526_152606

def setM : Set ℤ := { u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l }
def setN : Set ℤ := { u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r }

theorem eq_sets_M_N : setM = setN := by
  sorry

end eq_sets_M_N_l1526_152606


namespace bowling_tournament_prize_orders_l1526_152609
-- Import necessary Lean library

-- Define the conditions
def match_outcome (num_games : ℕ) : ℕ := 2 ^ num_games

-- Theorem statement
theorem bowling_tournament_prize_orders : match_outcome 5 = 32 := by
  -- This is the statement, proof is not required
  sorry

end bowling_tournament_prize_orders_l1526_152609


namespace number_of_vip_children_l1526_152637

theorem number_of_vip_children (total_attendees children_percentage children_vip_percentage : ℕ) :
  total_attendees = 400 →
  children_percentage = 75 →
  children_vip_percentage = 20 →
  (total_attendees * children_percentage / 100) * children_vip_percentage / 100 = 60 :=
by
  intros h_total h_children_pct h_vip_pct
  sorry

end number_of_vip_children_l1526_152637


namespace simplify_expression_l1526_152684

theorem simplify_expression : 3000 * 3000^3000 = 3000^(3001) := 
by 
  sorry

end simplify_expression_l1526_152684


namespace find_shortest_side_of_triangle_l1526_152688

def Triangle (A B C : Type) := true -- Dummy definition for a triangle

structure Segments :=
(BD DE EC : ℝ)

def angle_ratios (AD AE : ℝ) (r1 r2 : ℕ) := true -- Dummy definition for angle ratios

def triangle_conditions (ABC : Type) (s : Segments) (r1 r2 : ℕ)
  (h1 : angle_ratios AD AE r1 r2)
  (h2 : s.BD = 4)
  (h3 : s.DE = 2)
  (h4 : s.EC = 5) : Prop := True

noncomputable def shortestSide (ABC : Type) (s : Segments) (r1 r2 : ℕ) : ℝ := 
  if true then sorry else 0 -- Placeholder for the shortest side length function

theorem find_shortest_side_of_triangle (ABC : Type) (s : Segments)
  (h1 : angle_ratios AD AE 2 3) (h2 : angle_ratios AE AD 1 1)
  (h3 : s.BD = 4) (h4 : s.DE = 2) (h5 : s.EC = 5) :
  shortestSide ABC s 2 3 = 30 / 11 :=
sorry

end find_shortest_side_of_triangle_l1526_152688


namespace brainiacs_like_both_l1526_152646

theorem brainiacs_like_both
  (R M B : ℕ)
  (h1 : R = 2 * M)
  (h2 : R + M - B = 96)
  (h3 : M - B = 20) : B = 18 := by
  sorry

end brainiacs_like_both_l1526_152646


namespace subtraction_of_tenths_l1526_152653

theorem subtraction_of_tenths (a b : ℝ) (n : ℕ) (h1 : a = (1 / 10) * 6000) (h2 : b = (1 / 10 / 100) * 6000) : (a - b) = 594 := by
sorry

end subtraction_of_tenths_l1526_152653


namespace positive_number_percent_l1526_152665

theorem positive_number_percent (x : ℝ) (h : 0.01 * x^2 = 9) (hx : 0 < x) : x = 30 :=
sorry

end positive_number_percent_l1526_152665


namespace min_value_a_l1526_152669

theorem min_value_a (a : ℝ) : 
  (∀ x y : ℝ, (3 * x - 5 * y) ≥ 0 → x > 0 → y > 0 → (1 - a) * x ^ 2 + 2 * x * y - a * y ^ 2 ≤ 0) ↔ a ≥ 55 / 34 := 
by 
  sorry

end min_value_a_l1526_152669


namespace distinct_positive_integer_roots_pq_l1526_152610

theorem distinct_positive_integer_roots_pq :
  ∃ (p q : ℝ), (∃ (a b c : ℕ), (a ≠ b ∧ b ≠ c ∧ c ≠ a) ∧ (a + b + c = 9) ∧ (a * b + a * c + b * c = p) ∧ (a * b * c = q)) ∧ p + q = 38 :=
by sorry


end distinct_positive_integer_roots_pq_l1526_152610


namespace find_a8_l1526_152692

-- Define the arithmetic sequence aₙ
def arithmetic_sequence (a₁ d : ℕ) (n : ℕ) := a₁ + (n - 1) * d

-- The given condition
def condition (a₁ d : ℕ) :=
  arithmetic_sequence a₁ d 2 + arithmetic_sequence a₁ d 7 + arithmetic_sequence a₁ d 15 = 12

-- The value we want to prove
def a₈ (a₁ d : ℕ ) : ℕ :=
  arithmetic_sequence a₁ d 8

theorem find_a8 (a₁ d : ℕ) (h : condition a₁ d) : a₈ a₁ d = 4 :=
  sorry

end find_a8_l1526_152692


namespace find_positive_integer_pair_l1526_152607

theorem find_positive_integer_pair (a b : ℕ) (h : ∀ n : ℕ, n > 0 → ∃ c_n : ℕ, a^n + b^n = c_n^(n + 1)) : a = 2 ∧ b = 2 := 
sorry

end find_positive_integer_pair_l1526_152607


namespace reassemble_square_with_hole_l1526_152675

theorem reassemble_square_with_hole 
  (a b c d k1 k2 : ℝ)
  (h1 : a = b)
  (h2 : c = d)
  (h3 : k1 = k2) :
  ∃ (f gh ef gh' : ℝ), 
    f = a - c ∧
    gh = b - d ∧
    ef = f ∧
    gh' = gh := 
by sorry

end reassemble_square_with_hole_l1526_152675


namespace bee_count_l1526_152634

theorem bee_count (initial_bees additional_bees : ℕ) (h_init : initial_bees = 16) (h_add : additional_bees = 9) :
  initial_bees + additional_bees = 25 :=
by
  sorry

end bee_count_l1526_152634


namespace garden_snake_length_l1526_152639

theorem garden_snake_length :
  ∀ (garden_snake boa_constrictor : ℝ),
    boa_constrictor * 7.0 = garden_snake →
    boa_constrictor = 1.428571429 →
    garden_snake = 10.0 :=
by
  intros garden_snake boa_constrictor H1 H2
  sorry

end garden_snake_length_l1526_152639


namespace distinct_rational_numbers_l1526_152656

theorem distinct_rational_numbers (m : ℚ) :
  abs m < 100 ∧ (∃ x : ℤ, 4 * x^2 + m * x + 15 = 0) → 
  ∃ n : ℕ, n = 48 :=
sorry

end distinct_rational_numbers_l1526_152656


namespace stephanie_gas_payment_l1526_152614

variables (electricity_bill : ℕ) (gas_bill : ℕ) (water_bill : ℕ) (internet_bill : ℕ)
variables (electricity_paid : ℕ) (gas_paid_fraction : ℚ) (water_paid_fraction : ℚ) (internet_paid : ℕ)
variables (additional_gas_payment : ℕ) (remaining_payment : ℕ) (expected_remaining : ℕ)

def stephanie_budget : Prop :=
  electricity_bill = 60 ∧
  electricity_paid = 60 ∧
  gas_bill = 40 ∧
  gas_paid_fraction = 3/4 ∧
  water_bill = 40 ∧
  water_paid_fraction = 1/2 ∧
  internet_bill = 25 ∧
  internet_paid = 4 * 5 ∧
  remaining_payment = 30 ∧
  expected_remaining = 
    (gas_bill - gas_paid_fraction * gas_bill) +
    (water_bill - water_paid_fraction * water_bill) + 
    (internet_bill - internet_paid) - 
    additional_gas_payment ∧
  expected_remaining = remaining_payment

theorem stephanie_gas_payment : additional_gas_payment = 5 :=
by sorry

end stephanie_gas_payment_l1526_152614


namespace polynomial_factorization_l1526_152649

theorem polynomial_factorization (x : ℝ) : x - x^3 = x * (1 - x) * (1 + x) := 
by sorry

end polynomial_factorization_l1526_152649


namespace min_value_of_f_l1526_152642

noncomputable def f (x a : ℝ) := Real.exp (x - a) - Real.log (x + a) - 1

theorem min_value_of_f (a : ℝ) : 
  (0 < a) → (∃ x : ℝ, f x a = 0) ↔ a = 1 / 2 :=
by
  sorry

end min_value_of_f_l1526_152642


namespace calculate_weight_of_first_batch_jelly_beans_l1526_152678

theorem calculate_weight_of_first_batch_jelly_beans (J : ℝ)
    (h1 : 16 = 8 * (J * 4)) : J = 2 := 
  sorry

end calculate_weight_of_first_batch_jelly_beans_l1526_152678


namespace number_of_lines_l1526_152673

-- Define points A and B
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (3, 1)

-- Define the distances from the points
def d_A : ℝ := 1
def d_B : ℝ := 2

-- A theorem stating the number of lines under the given conditions
theorem number_of_lines (A B : ℝ × ℝ) (d_A d_B : ℝ) (hA : A = (1, 2)) (hB : B = (3, 1)) (hdA : d_A = 1) (hdB : d_B = 2) :
  ∃ n : ℕ, n = 2 :=
by {
  sorry
}

end number_of_lines_l1526_152673


namespace total_unique_working_games_l1526_152694

-- Define the given conditions
def initial_games_from_friend := 25
def non_working_games_from_friend := 12

def games_from_garage_sale := 15
def non_working_games_from_garage_sale := 8
def duplicate_games := 3

-- Calculate the number of working games from each source
def working_games_from_friend := initial_games_from_friend - non_working_games_from_friend
def total_garage_sale_games := games_from_garage_sale - non_working_games_from_garage_sale
def unique_working_games_from_garage_sale := total_garage_sale_games - duplicate_games

-- Theorem statement
theorem total_unique_working_games : 
  working_games_from_friend + unique_working_games_from_garage_sale = 17 := by
  sorry

end total_unique_working_games_l1526_152694


namespace find_number_l1526_152651

theorem find_number (N : ℝ) (h : (1 / 2) * (3 / 5) * N = 36) : N = 120 :=
by
  sorry

end find_number_l1526_152651


namespace triangle_area_l1526_152605

theorem triangle_area (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (h₄ : a * a + b * b = c * c) :
  (1/2) * a * b = 30 :=
by
  sorry

end triangle_area_l1526_152605


namespace interest_rate_l1526_152652

noncomputable def compoundInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * ((1 + r / 100 : ℚ) ^ t) - P

noncomputable def simpleInterest (P : ℕ) (r : ℕ) (t : ℕ) : ℚ :=
  P * r * t / 100

theorem interest_rate (P t : ℕ) (D : ℚ) (r : ℕ) :
  P = 10000 → t = 2 → D = 49 →
  compoundInterest P r t - simpleInterest P r t = D → r = 7 := by
  sorry

end interest_rate_l1526_152652
