import Mathlib

namespace NUMINAMATH_GPT_problem_l689_68963

noncomputable def f (ω x : ℝ) : ℝ := (Real.sin (ω * x / 2))^2 + (1 / 2) * Real.sin (ω * x) - 1 / 2

theorem problem (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Set.Ioo (Real.pi : ℝ) (2 * Real.pi) → f ω x ≠ 0) →
  ω ∈ Set.Icc 0 (1 / 8) ∪ Set.Icc (1 / 4) (5 / 8) :=
by
  sorry

end NUMINAMATH_GPT_problem_l689_68963


namespace NUMINAMATH_GPT_tan_alpha_l689_68907

theorem tan_alpha {α : ℝ} (h : Real.tan (α + π / 4) = 9) : Real.tan α = 4 / 5 :=
sorry

end NUMINAMATH_GPT_tan_alpha_l689_68907


namespace NUMINAMATH_GPT_robin_initial_gum_l689_68983

theorem robin_initial_gum (x : ℕ) (h1 : x + 26 = 44) : x = 18 := 
by 
  sorry

end NUMINAMATH_GPT_robin_initial_gum_l689_68983


namespace NUMINAMATH_GPT_EdProblem_l689_68994

/- Define the conditions -/
def EdConditions := 
  ∃ (m : ℕ) (N : ℕ), 
    m = 16 ∧ 
    N = Nat.choose 15 5 ∧
    N % 1000 = 3

/- The statement to be proven -/
theorem EdProblem : EdConditions :=
  sorry

end NUMINAMATH_GPT_EdProblem_l689_68994


namespace NUMINAMATH_GPT_cos_alpha_minus_pi_six_l689_68969

theorem cos_alpha_minus_pi_six (α : ℝ) (h : Real.sin (α + Real.pi / 3) = 4 / 5) : 
  Real.cos (α - Real.pi / 6) = 4 / 5 :=
sorry

end NUMINAMATH_GPT_cos_alpha_minus_pi_six_l689_68969


namespace NUMINAMATH_GPT_total_distance_covered_l689_68958

theorem total_distance_covered (d : ℝ) :
  (d / 5 + d / 10 + d / 15 + d / 20 + d / 25 = 15 / 60) → (5 * d = 375 / 137) :=
by
  intro h
  -- proof will go here
  sorry

end NUMINAMATH_GPT_total_distance_covered_l689_68958


namespace NUMINAMATH_GPT_spinner_prob_l689_68924

theorem spinner_prob (PD PE PF_PG : ℚ) (hD : PD = 1/4) (hE : PE = 1/3) 
  (hTotal : PD + PE + PF_PG = 1) : PF_PG = 5/12 := by
  sorry

end NUMINAMATH_GPT_spinner_prob_l689_68924


namespace NUMINAMATH_GPT_number_of_ways_to_sign_up_probability_student_A_online_journalists_l689_68967

-- Definitions for the conditions
def students : Finset String := {"A", "B", "C", "D", "E"}
def projects : Finset String := {"Online Journalists", "Robot Action", "Sounds of Music"}

-- Function to calculate combinations (nCr)
def combinations (n k : ℕ) : ℕ := Nat.choose n k

-- Function to calculate arrangements
def arrangements (n : ℕ) : ℕ := Nat.factorial n

-- Proof opportunity for part 1
theorem number_of_ways_to_sign_up : 
  (combinations 5 3 * arrangements 3) + ((combinations 5 2 * combinations 3 2) / arrangements 2 * arrangements 3) = 150 :=
sorry

-- Proof opportunity for part 2
theorem probability_student_A_online_journalists
  (h : (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 243) : 
  ((combinations 4 3 * arrangements 2) * projects.card ^ 3) / 
  (combinations 5 3 * arrangements 3 + combinations 5 3 * combinations 3 2 * arrangements 2 * arrangements 3) = 1 / 15 :=
sorry

end NUMINAMATH_GPT_number_of_ways_to_sign_up_probability_student_A_online_journalists_l689_68967


namespace NUMINAMATH_GPT_initial_money_l689_68948

theorem initial_money (M : ℝ) (h1 : M - (1/4 * M) - (1/3 * (M - (1/4 * M))) = 1600) : M = 3200 :=
sorry

end NUMINAMATH_GPT_initial_money_l689_68948


namespace NUMINAMATH_GPT_find_x_l689_68919

noncomputable def x : ℝ := 10.3

theorem find_x (h1 : x + (⌈x⌉ : ℝ) = 21.3) (h2 : x > 0) : x = 10.3 :=
sorry

end NUMINAMATH_GPT_find_x_l689_68919


namespace NUMINAMATH_GPT_total_cleaning_time_l689_68950

def hose_time : ℕ := 10
def shampoos : ℕ := 3
def shampoo_time : ℕ := 15

theorem total_cleaning_time : hose_time + shampoos * shampoo_time = 55 := by
  sorry

end NUMINAMATH_GPT_total_cleaning_time_l689_68950


namespace NUMINAMATH_GPT_initial_dimes_l689_68923

theorem initial_dimes (x : ℕ) (h1 : x + 7 = 16) : x = 9 := by
  sorry

end NUMINAMATH_GPT_initial_dimes_l689_68923


namespace NUMINAMATH_GPT_parametric_equation_correct_max_min_x_plus_y_l689_68996

noncomputable def parametric_equation (φ : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ)

theorem parametric_equation_correct (ρ θ : ℝ) (h : ρ^2 - 4 * Real.sqrt 2 * Real.cos (θ - π/4) + 6 = 0) :
  ∃ (φ : ℝ), parametric_equation φ = ( 2 + Real.sqrt 2 * Real.cos φ, 2 + Real.sqrt 2 * Real.sin φ) := 
sorry

theorem max_min_x_plus_y (P : ℝ × ℝ) (hP : ∃ (φ : ℝ), P = parametric_equation φ) :
  ∃ f : ℝ, (P.fst + P.snd) = f ∧ (f = 6 ∨ f = 2) :=
sorry

end NUMINAMATH_GPT_parametric_equation_correct_max_min_x_plus_y_l689_68996


namespace NUMINAMATH_GPT_ratio_2006_to_2005_l689_68952

-- Conditions
def kids_in_2004 : ℕ := 60
def kids_in_2005 : ℕ := kids_in_2004 / 2
def kids_in_2006 : ℕ := 20

-- The statement to prove
theorem ratio_2006_to_2005 : 
  (kids_in_2006 : ℚ) / kids_in_2005 = 2 / 3 :=
sorry

end NUMINAMATH_GPT_ratio_2006_to_2005_l689_68952


namespace NUMINAMATH_GPT_frictional_force_is_correct_l689_68902

-- Definitions
def m1 := 2.0 -- mass of the tank in kg
def m2 := 10.0 -- mass of the cart in kg
def a := 5.0 -- acceleration of the cart in m/s^2
def mu := 0.6 -- coefficient of friction between the tank and the cart
def g := 9.8 -- acceleration due to gravity in m/s^2

-- Frictional force acting on the tank
def frictional_force := mu * (m1 * g)

-- Required force to accelerate the tank with the cart
def required_force := m1 * a

-- Proof statement
theorem frictional_force_is_correct : required_force = 10 := 
by
  -- skipping the proof as specified
  sorry

end NUMINAMATH_GPT_frictional_force_is_correct_l689_68902


namespace NUMINAMATH_GPT_domain_of_k_l689_68939

noncomputable def k (x : ℝ) := (1 / (x + 9)) + (1 / (x^2 + 9)) + (1 / (x^5 + 9)) + (1 / (x - 9))

theorem domain_of_k :
  ∀ x : ℝ, x ≠ -9 ∧ x ≠ -1.551 ∧ x ≠ 9 → ∃ y, y = k x := 
by
  sorry

end NUMINAMATH_GPT_domain_of_k_l689_68939


namespace NUMINAMATH_GPT_polynomial_one_negative_root_iff_l689_68906

noncomputable def polynomial_has_one_negative_real_root (p : ℝ) : Prop :=
  ∃ (x : ℝ), (x^4 + 3*p*x^3 + 6*x^2 + 3*p*x + 1 = 0) ∧
  ∀ (y : ℝ), y < x → y^4 + 3*p*y^3 + 6*y^2 + 3*p*y + 1 ≠ 0

theorem polynomial_one_negative_root_iff (p : ℝ) :
  polynomial_has_one_negative_real_root p ↔ p ≥ 4 / 3 :=
sorry

end NUMINAMATH_GPT_polynomial_one_negative_root_iff_l689_68906


namespace NUMINAMATH_GPT_g_increasing_on_interval_l689_68936

noncomputable def f (x : ℝ) : ℝ := Real.sin ((1/5) * x + 13 * Real.pi / 6)
noncomputable def g (x : ℝ) : ℝ := Real.sin ((1/5) * (x - 10 * Real.pi / 3) + 13 * Real.pi / 6)

theorem g_increasing_on_interval : ∀ x y : ℝ, (π ≤ x ∧ x < y ∧ y ≤ 2 * π) → g x < g y :=
by
  intro x y h
  -- Mathematical steps to prove this
  sorry

end NUMINAMATH_GPT_g_increasing_on_interval_l689_68936


namespace NUMINAMATH_GPT_find_ab_l689_68937

theorem find_ab (a b : ℝ) (h1 : a + b = 5) (h2 : a^3 + b^3 = 35) : a * b = 6 :=
by
  sorry

end NUMINAMATH_GPT_find_ab_l689_68937


namespace NUMINAMATH_GPT_find_z_l689_68926

theorem find_z (x y z : ℚ) (h1 : x / (y + 1) = 4 / 5) (h2 : 3 * z = 2 * x + y) (h3 : y = 10) : 
  z = 46 / 5 := 
sorry

end NUMINAMATH_GPT_find_z_l689_68926


namespace NUMINAMATH_GPT_distance_travelled_is_960_l689_68968

-- Definitions based on conditions
def speed_slower := 60 -- Speed of slower bike in km/h
def speed_faster := 64 -- Speed of faster bike in km/h
def time_diff := 1 -- Time difference in hours

-- Problem statement: Prove that the distance covered by both bikes is 960 km.
theorem distance_travelled_is_960 (T : ℝ) (D : ℝ) 
  (h1 : D = speed_slower * T)
  (h2 : D = speed_faster * (T - time_diff)) :
  D = 960 := 
sorry

end NUMINAMATH_GPT_distance_travelled_is_960_l689_68968


namespace NUMINAMATH_GPT_total_cost_898_8_l689_68912

theorem total_cost_898_8 :
  ∀ (M R F : ℕ → ℝ), 
    (10 * M 1 = 24 * R 1) →
    (6 * F 1 = 2 * R 1) →
    (F 1 = 21) →
    (4 * M 1 + 3 * R 1 + 5 * F 1 = 898.8) :=
by
  intros M R F h1 h2 h3
  sorry

end NUMINAMATH_GPT_total_cost_898_8_l689_68912


namespace NUMINAMATH_GPT_mixing_ratios_l689_68965

theorem mixing_ratios (V : ℝ) (hV : 0 < V) :
  (4 * V / 5 + 7 * V / 10) / (V / 5 + 3 * V / 10) = 3 :=
by
  sorry

end NUMINAMATH_GPT_mixing_ratios_l689_68965


namespace NUMINAMATH_GPT_prove_b_is_neg_two_l689_68964

-- Define the conditions
variables (b : ℝ)

-- Hypothesis: The real and imaginary parts of the complex number (2 - b * I) * I are opposites
def complex_opposite_parts (b : ℝ) : Prop :=
  b = -2

-- The theorem statement
theorem prove_b_is_neg_two : complex_opposite_parts b :=
sorry

end NUMINAMATH_GPT_prove_b_is_neg_two_l689_68964


namespace NUMINAMATH_GPT_sum_of_xi_l689_68987

theorem sum_of_xi {x1 x2 x3 x4 : ℝ} (h1: (x1 - 3) * Real.sin (π * x1) = 1)
  (h2: (x2 - 3) * Real.sin (π * x2) = 1)
  (h3: (x3 - 3) * Real.sin (π * x3) = 1)
  (h4: (x4 - 3) * Real.sin (π * x4) = 1)
  (hx1 : x1 > 0) (hx2: x2 > 0) (hx3 : x3 > 0) (hx4: x4 > 0) :
  x1 + x2 + x3 + x4 = 12 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_xi_l689_68987


namespace NUMINAMATH_GPT_avg_cans_used_per_game_l689_68920

theorem avg_cans_used_per_game (total_rounds : ℕ) (games_first_round : ℕ) (games_second_round : ℕ)
  (games_third_round : ℕ) (games_finals : ℕ) (total_tennis_balls : ℕ) (balls_per_can : ℕ)
  (h1 : total_rounds = 4) (h2 : games_first_round = 8) (h3 : games_second_round = 4) 
  (h4 : games_third_round = 2) (h5 : games_finals = 1) (h6 : total_tennis_balls = 225) 
  (h7 : balls_per_can = 3) :
  let total_games := games_first_round + games_second_round + games_third_round + games_finals
  let total_cans_used := total_tennis_balls / balls_per_can
  let avg_cans_per_game := total_cans_used / total_games
  avg_cans_per_game = 5 :=
by {
  -- proof steps here
  sorry
}

end NUMINAMATH_GPT_avg_cans_used_per_game_l689_68920


namespace NUMINAMATH_GPT_polynomial_expansion_l689_68986

variable (t : ℝ)

theorem polynomial_expansion :
  (3 * t^3 + 2 * t^2 - 4 * t + 3) * (-4 * t^3 + 3 * t - 5) = -12 * t^6 - 8 * t^5 + 25 * t^4 - 21 * t^3 - 22 * t^2 + 29 * t - 15 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_expansion_l689_68986


namespace NUMINAMATH_GPT_range_of_ab_l689_68991

theorem range_of_ab (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : |2 - a^2| = |2 - b^2|) : 0 < a * b ∧ a * b < 2 := by
  sorry

end NUMINAMATH_GPT_range_of_ab_l689_68991


namespace NUMINAMATH_GPT_daniel_utility_equation_solution_l689_68988

theorem daniel_utility_equation_solution (t : ℚ) :
  t * (10 - t) = (4 - t) * (t + 4) → t = 8 / 5 := by
  sorry

end NUMINAMATH_GPT_daniel_utility_equation_solution_l689_68988


namespace NUMINAMATH_GPT_exactly_two_overlap_l689_68982

-- Define the concept of rectangles
structure Rectangle :=
  (width : ℕ)
  (height : ℕ)

-- Define the given rectangles
def rect1 : Rectangle := ⟨4, 6⟩
def rect2 : Rectangle := ⟨4, 6⟩
def rect3 : Rectangle := ⟨4, 6⟩

-- Hypothesis defining the overlapping areas
def overlap1_2 : ℕ := 4 * 2 -- first and second rectangles overlap in 8 cells
def overlap2_3 : ℕ := 2 * 6 -- second and third rectangles overlap in 12 cells
def overlap1_3 : ℕ := 0    -- first and third rectangles do not directly overlap

-- Total overlap calculation
def total_exactly_two_overlap : ℕ := (overlap1_2 + overlap2_3)

-- The theorem we need to prove
theorem exactly_two_overlap (rect1 rect2 rect3 : Rectangle) : total_exactly_two_overlap = 14 := sorry

end NUMINAMATH_GPT_exactly_two_overlap_l689_68982


namespace NUMINAMATH_GPT_initial_customers_l689_68944

theorem initial_customers (S : ℕ) (initial : ℕ) (H1 : initial = S + (S + 5)) (H2 : S = 3) : initial = 11 := 
by
  sorry

end NUMINAMATH_GPT_initial_customers_l689_68944


namespace NUMINAMATH_GPT_geometric_sequence_common_ratio_l689_68927

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (r : ℝ) (h_geometric : ∀ n, a (n + 1) = r * a n)
  (h_relation : ∀ n, a n = (1 / 2) * (a (n + 1) + a (n + 2))) (h_positive : ∀ n, a n > 0) : r = 1 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_common_ratio_l689_68927


namespace NUMINAMATH_GPT_intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l689_68943

variables (x y z : ℕ)

-- Conditions
axiom condition_1 : ∀ (t: ℕ), t = (6 : ℕ) → y * z = 6 * (y - x)
axiom condition_2 : ∀ (t: ℕ), t = (3 : ℕ) → y * z = 3 * (y + x)

-- Proof statements
theorem intervals_between_trolleybuses : z = 4 :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

theorem sportsman_slower_than_trolleybus : y = 3 * x :=
by {
  -- Assuming the axioms as proof would involve using them
  sorry
}

end NUMINAMATH_GPT_intervals_between_trolleybuses_sportsman_slower_than_trolleybus_l689_68943


namespace NUMINAMATH_GPT_abs_diff_of_two_numbers_l689_68916

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 160) : |x - y| = 2 * Real.sqrt 65 :=
by
  sorry

end NUMINAMATH_GPT_abs_diff_of_two_numbers_l689_68916


namespace NUMINAMATH_GPT_sum_of_consecutive_integers_345_l689_68945

-- Definition of the conditions
def is_consecutive_sum (n : ℕ) (k : ℕ) (s : ℕ) : Prop :=
  s = k * n + k * (k - 1) / 2

-- Problem statement
theorem sum_of_consecutive_integers_345 :
  ∃ k_set : Finset ℕ, (∀ k ∈ k_set, k ≥ 2 ∧ ∃ n : ℕ, is_consecutive_sum n k 345) ∧ k_set.card = 6 :=
sorry

end NUMINAMATH_GPT_sum_of_consecutive_integers_345_l689_68945


namespace NUMINAMATH_GPT_ratio_of_boys_l689_68938

variables {b g o : ℝ}

theorem ratio_of_boys (h1 : b = (1/2) * o)
  (h2 : g = o - b)
  (h3 : b + g + o = 1) :
  b = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_boys_l689_68938


namespace NUMINAMATH_GPT_distance_from_point_to_origin_l689_68905

theorem distance_from_point_to_origin (x y : ℝ) (h : x = -3 ∧ y = 4) : 
  (Real.sqrt (x^2 + y^2)) = 5 := by
  sorry

end NUMINAMATH_GPT_distance_from_point_to_origin_l689_68905


namespace NUMINAMATH_GPT_solution_set_inequality_l689_68985

theorem solution_set_inequality : {x : ℝ | (x-1)*(x-2) ≤ 0} = {x : ℝ | 1 ≤ x ∧ x ≤ 2} :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l689_68985


namespace NUMINAMATH_GPT_exists_nonneg_integers_l689_68961

theorem exists_nonneg_integers (p : ℕ) (hp : Nat.Prime p) (hp_odd : p % 2 = 1) :
  ∃ (x y z t : ℕ), (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0 ∨ t ≠ 0) ∧ t < p ∧ x^2 + y^2 + z^2 = t * p :=
sorry

end NUMINAMATH_GPT_exists_nonneg_integers_l689_68961


namespace NUMINAMATH_GPT_conclusion_2_conclusion_3_conclusion_4_l689_68959

variable (b : ℝ)

def f (x : ℝ) : ℝ := x^2 - |b| * x - 3

theorem conclusion_2 (h_min : ∃ x, f b x = -3) : b = 0 :=
  sorry

theorem conclusion_3 (h_b : b = -2) (x : ℝ) (hx : -2 < x ∧ x < 2) :
    -4 ≤ f b x ∧ f b x ≤ -3 :=
  sorry

theorem conclusion_4 (hb_ne : b ≠ 0) (m : ℝ) (h_roots : ∃ x1 x2, f b x1 = m ∧ f b x2 = m ∧ x1 ≠ x2) :
    m > -3 ∨ b^2 = -4 * m - 12 :=
  sorry

end NUMINAMATH_GPT_conclusion_2_conclusion_3_conclusion_4_l689_68959


namespace NUMINAMATH_GPT_total_students_is_correct_l689_68984

-- Define the number of students in each class based on the conditions
def number_of_students_finley := 24
def number_of_students_johnson := (number_of_students_finley / 2) + 10
def number_of_students_garcia := 2 * number_of_students_johnson
def number_of_students_smith := number_of_students_finley / 3
def number_of_students_patel := (3 / 4) * (number_of_students_finley + number_of_students_johnson + number_of_students_garcia)

-- Define the total number of students in all five classes combined
def total_number_of_students := 
  number_of_students_finley + 
  number_of_students_johnson + 
  number_of_students_garcia +
  number_of_students_smith + 
  number_of_students_patel

-- The theorem statement to prove
theorem total_students_is_correct : total_number_of_students = 166 := by
  sorry

end NUMINAMATH_GPT_total_students_is_correct_l689_68984


namespace NUMINAMATH_GPT_solve_system_l689_68925

theorem solve_system :
  ∃ (x y z : ℝ), 
    (x + y - z = 4 ∧ x^2 - y^2 + z^2 = -4 ∧ xyz = 6) ↔ 
    (x, y, z) = (2, 3, 1) ∨ (x, y, z) = (-1, 3, -2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l689_68925


namespace NUMINAMATH_GPT_square_perimeter_of_N_l689_68928

theorem square_perimeter_of_N (area_M : ℝ) (area_N : ℝ) (side_N : ℝ) (perimeter_N : ℝ)
  (h1 : area_M = 100)
  (h2 : area_N = 4 * area_M)
  (h3 : area_N = side_N * side_N)
  (h4 : perimeter_N = 4 * side_N) :
  perimeter_N = 80 := 
sorry

end NUMINAMATH_GPT_square_perimeter_of_N_l689_68928


namespace NUMINAMATH_GPT_negative_number_among_options_l689_68913

theorem negative_number_among_options :
  let A := abs (-1)
  let B := -(2^2)
  let C := (-(Real.sqrt 3))^2
  let D := (-3)^0
  B < 0 ∧ A > 0 ∧ C > 0 ∧ D > 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_number_among_options_l689_68913


namespace NUMINAMATH_GPT_solve_for_x_l689_68993

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 6 = 13) : x = 35.5 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_solve_for_x_l689_68993


namespace NUMINAMATH_GPT_greatest_integer_solution_l689_68922

theorem greatest_integer_solution :
  ∃ x : ℤ, (∃ (k : ℤ), (8 : ℚ) / 11 > k / 15 ∧ k = 10) ∧ x = 10 :=
by {
  sorry
}

end NUMINAMATH_GPT_greatest_integer_solution_l689_68922


namespace NUMINAMATH_GPT_calculate_si_l689_68971

section SimpleInterest

def Principal : ℝ := 10000
def Rate : ℝ := 0.04
def Time : ℝ := 1
def SimpleInterest : ℝ := Principal * Rate * Time

theorem calculate_si : SimpleInterest = 400 := by
  -- Proof goes here.
  sorry

end SimpleInterest

end NUMINAMATH_GPT_calculate_si_l689_68971


namespace NUMINAMATH_GPT_card_draw_probability_l689_68997

-- Define a function to compute the probability of a sequence of draws
noncomputable def probability_of_event : Rat :=
  (4 / 52) * (4 / 51) * (1 / 50)

theorem card_draw_probability :
  probability_of_event = 4 / 33150 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_card_draw_probability_l689_68997


namespace NUMINAMATH_GPT_set_intersection_l689_68935

-- defining universal set U
def U : Set ℕ := {1, 3, 5, 7, 9}

-- defining set A
def A : Set ℕ := {1, 5, 9}

-- defining set B
def B : Set ℕ := {3, 7, 9}

-- complement of A in U
def complU (s : Set ℕ) := {x ∈ U | x ∉ s}

-- defining the intersection of complement of A with B
def intersection := complU A ∩ B

-- statement to be proved
theorem set_intersection : intersection = {3, 7} :=
by
  sorry

end NUMINAMATH_GPT_set_intersection_l689_68935


namespace NUMINAMATH_GPT_volleyball_team_geography_l689_68946

theorem volleyball_team_geography (total_players history_players both_subjects : ℕ) 
  (H1 : total_players = 15) 
  (H2 : history_players = 9) 
  (H3 : both_subjects = 4) : 
  ∃ (geography_players : ℕ), geography_players = 10 :=
by
  -- Definitions / Calculations
  -- Using conditions to derive the number of geography players
  let only_geography_players : ℕ := total_players - history_players
  let geography_players : ℕ := only_geography_players + both_subjects

  -- Prove the statement
  use geography_players
  sorry

end NUMINAMATH_GPT_volleyball_team_geography_l689_68946


namespace NUMINAMATH_GPT_smallest_four_digit_palindrome_divisible_by_8_l689_68962

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def divisible_by_8 (n : ℕ) : Prop :=
  n % 8 = 0

theorem smallest_four_digit_palindrome_divisible_by_8 : ∃ (n : ℕ), is_palindrome n ∧ is_four_digit n ∧ divisible_by_8 n ∧ n = 4004 := by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_palindrome_divisible_by_8_l689_68962


namespace NUMINAMATH_GPT_ratio_of_a_to_c_l689_68932

theorem ratio_of_a_to_c
  {a b c : ℕ}
  (h1 : a / b = 11 / 3)
  (h2 : b / c = 1 / 5) :
  a / c = 11 / 15 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_of_a_to_c_l689_68932


namespace NUMINAMATH_GPT_find_m_plus_n_l689_68974

theorem find_m_plus_n (a m n : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) (h3 : a^m = n) (h4 : a^0 = 1) : m + n = 1 :=
sorry

end NUMINAMATH_GPT_find_m_plus_n_l689_68974


namespace NUMINAMATH_GPT_income_in_scientific_notation_l689_68979

theorem income_in_scientific_notation :
  10870 = 1.087 * 10^4 := 
sorry

end NUMINAMATH_GPT_income_in_scientific_notation_l689_68979


namespace NUMINAMATH_GPT_exponentiation_problem_l689_68918

variable (x : ℝ) (m n : ℝ)

theorem exponentiation_problem (h1 : x ^ m = 5) (h2 : x ^ n = 1 / 4) :
  x ^ (2 * m - n) = 100 :=
sorry

end NUMINAMATH_GPT_exponentiation_problem_l689_68918


namespace NUMINAMATH_GPT_lcm_72_108_2100_l689_68953

theorem lcm_72_108_2100 : Nat.lcm (Nat.lcm 72 108) 2100 = 37800 := by
  sorry

end NUMINAMATH_GPT_lcm_72_108_2100_l689_68953


namespace NUMINAMATH_GPT_distance_between_two_cars_l689_68970

theorem distance_between_two_cars 
    (initial_distance : ℝ) 
    (first_car_distance1 : ℝ) 
    (first_car_distance2 : ℝ)
    (second_car_distance : ℝ) 
    (final_distance : ℝ) :
    initial_distance = 150 →
    first_car_distance1 = 25 →
    first_car_distance2 = 25 →
    second_car_distance = 35 →
    final_distance = initial_distance - (first_car_distance1 + first_car_distance2 + second_car_distance) →
    final_distance = 65 :=
by
  intros h_initial h_first1 h_first2 h_second h_final
  sorry

end NUMINAMATH_GPT_distance_between_two_cars_l689_68970


namespace NUMINAMATH_GPT_marbles_distribution_l689_68955

theorem marbles_distribution (marbles children : ℕ) (h1 : marbles = 60) (h2 : children = 7) :
  ∃ k, k = 3 → (∀ i < children, marbles / children + (if i < marbles % children then 1 else 0) < 9) → k = 3 :=
by
  sorry

end NUMINAMATH_GPT_marbles_distribution_l689_68955


namespace NUMINAMATH_GPT_infinitely_many_n_divisible_by_prime_l689_68903

theorem infinitely_many_n_divisible_by_prime (p : ℕ) (hp : Prime p) : 
  ∃ᶠ n in at_top, p ∣ (2^n - n) :=
by {
  sorry
}

end NUMINAMATH_GPT_infinitely_many_n_divisible_by_prime_l689_68903


namespace NUMINAMATH_GPT_employee_payment_correct_l689_68911

-- Define the wholesale cost
def wholesale_cost : ℝ := 200

-- Define the retail price increase percentage
def retail_increase_percentage : ℝ := 0.20

-- Define the employee discount percentage
def employee_discount_percentage : ℝ := 0.30

-- Define the retail price as wholesale cost increased by the retail increase percentage
def retail_price : ℝ := wholesale_cost * (1 + retail_increase_percentage)

-- Define the discount amount as the retail price multiplied by the discount percentage
def discount_amount : ℝ := retail_price * employee_discount_percentage

-- Define the final employee payment as retail price minus the discount amount
def employee_final_payment : ℝ := retail_price - discount_amount

-- Theorem statement: Prove that the employee final payment equals $168
theorem employee_payment_correct : employee_final_payment = 168 := by
  sorry

end NUMINAMATH_GPT_employee_payment_correct_l689_68911


namespace NUMINAMATH_GPT_part1_inequality_part2_range_of_a_l689_68931

noncomputable def f (x a : ℝ) : ℝ := abs (x - a) + abs (x + 1)

-- Part (1)
theorem part1_inequality (x : ℝ) (h : f x 2 < 5) : -2 < x ∧ x < 3 := sorry

-- Part (2)
theorem part2_range_of_a (x a : ℝ) (h : ∀ x, f x a ≥ 4 - abs (a - 1)) : a ≤ -2 ∨ a ≥ 2 := sorry

end NUMINAMATH_GPT_part1_inequality_part2_range_of_a_l689_68931


namespace NUMINAMATH_GPT_max_value_of_expr_l689_68981

theorem max_value_of_expr : ∃ t : ℝ, (∀ u : ℝ, (3^u - 2*u) * u / 9^u ≤ (3^t - 2*t) * t / 9^t) ∧ (3^t - 2*t) * t / 9^t = 1/8 :=
by sorry

end NUMINAMATH_GPT_max_value_of_expr_l689_68981


namespace NUMINAMATH_GPT_hyperbola_parabola_shared_focus_l689_68947

theorem hyperbola_parabola_shared_focus (a : ℝ) (h : a > 0) :
  (∃ b c : ℝ, b^2 = 3 ∧ c = 2 ∧ a^2 = c^2 - b^2 ∧ b ≠ 0) →
  a = 1 :=
by
  intro h_shared_focus
  sorry

end NUMINAMATH_GPT_hyperbola_parabola_shared_focus_l689_68947


namespace NUMINAMATH_GPT_construct_segment_eq_abc_div_de_l689_68909

theorem construct_segment_eq_abc_div_de 
(a b c d e : ℝ) (h_a : 0 < a) (h_b : 0 < b) (h_c : 0 < c) (h_d : 0 < d) (h_e : 0 < e) :
  ∃ x : ℝ, x = (a * b * c) / (d * e) :=
by sorry

end NUMINAMATH_GPT_construct_segment_eq_abc_div_de_l689_68909


namespace NUMINAMATH_GPT_remainder_of_2n_divided_by_11_l689_68949

theorem remainder_of_2n_divided_by_11
  (n k : ℤ)
  (h : n = 22 * k + 12) :
  (2 * n) % 11 = 2 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_remainder_of_2n_divided_by_11_l689_68949


namespace NUMINAMATH_GPT_value_of_a9_l689_68929

variables (a : ℕ → ℤ) (d : ℤ)
noncomputable def arithmetic_sequence : Prop :=
(a 1 + (a 1 + 10 * d)) / 2 = 15 ∧
a 1 + (a 1 + d) + (a 1 + 2 * d) = 9

theorem value_of_a9 (h : arithmetic_sequence a d) : a 9 = 24 :=
by sorry

end NUMINAMATH_GPT_value_of_a9_l689_68929


namespace NUMINAMATH_GPT_george_monthly_income_l689_68914

theorem george_monthly_income (I : ℝ) (h : I / 2 - 20 = 100) : I = 240 := 
by sorry

end NUMINAMATH_GPT_george_monthly_income_l689_68914


namespace NUMINAMATH_GPT_problem_solution_l689_68989

/-- Let f be an even function on ℝ such that f(x + 2) = f(x) and f(x) = x - 2 for x ∈ [3, 4]. 
    Then f(sin 1) < f(cos 1). -/
theorem problem_solution (f : ℝ → ℝ) 
  (h1 : ∀ x, f (-x) = f x)
  (h2 : ∀ x, f (x + 2) = f x)
  (h3 : ∀ x, 3 ≤ x ∧ x ≤ 4 → f x = x - 2) :
  f (Real.sin 1) < f (Real.cos 1) :=
sorry

end NUMINAMATH_GPT_problem_solution_l689_68989


namespace NUMINAMATH_GPT_smallest_even_number_l689_68942

theorem smallest_even_number (n1 n2 n3 n4 n5 n6 n7 : ℤ) 
  (h_sum_seven : n1 + n2 + n3 + n4 + n5 + n6 + n7 = 700)
  (h_sum_first_three : n1 + n2 + n3 > 200)
  (h_consecutive : n2 = n1 + 2 ∧ n3 = n2 + 2 ∧ n4 = n3 + 2 ∧ n5 = n4 + 2 ∧ n6 = n5 + 2 ∧ n7 = n6 + 2) :
  n1 = 94 := 
sorry

end NUMINAMATH_GPT_smallest_even_number_l689_68942


namespace NUMINAMATH_GPT_turnip_difference_l689_68995

theorem turnip_difference (melanie_turnips benny_turnips : ℕ) (h1 : melanie_turnips = 139) (h2 : benny_turnips = 113) : melanie_turnips - benny_turnips = 26 := by
  sorry

end NUMINAMATH_GPT_turnip_difference_l689_68995


namespace NUMINAMATH_GPT_jenny_improvements_value_l689_68972

-- Definitions based on the conditions provided
def property_tax_rate : ℝ := 0.02
def initial_house_value : ℝ := 400000
def rail_project_increase : ℝ := 0.25
def affordable_property_tax : ℝ := 15000

-- Statement of the theorem
theorem jenny_improvements_value :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_affordable_house_value := affordable_property_tax / property_tax_rate
  let value_of_improvements := max_affordable_house_value - new_house_value
  value_of_improvements = 250000 := 
by
  sorry

end NUMINAMATH_GPT_jenny_improvements_value_l689_68972


namespace NUMINAMATH_GPT_min_correct_answers_l689_68978

theorem min_correct_answers (total_questions correct_points incorrect_points target_score : ℕ)
                            (h_total : total_questions = 22)
                            (h_correct_points : correct_points = 4)
                            (h_incorrect_points : incorrect_points = 2)
                            (h_target : target_score = 81) :
  ∃ x : ℕ, 4 * x - 2 * (22 - x) > 81 ∧ x ≥ 21 :=
by {
  sorry
}

end NUMINAMATH_GPT_min_correct_answers_l689_68978


namespace NUMINAMATH_GPT_smallest_possible_other_integer_l689_68930

theorem smallest_possible_other_integer (n : ℕ) (h1 : Nat.lcm 60 n / Nat.gcd 60 n = 84) : n = 35 :=
sorry

end NUMINAMATH_GPT_smallest_possible_other_integer_l689_68930


namespace NUMINAMATH_GPT_betta_fish_count_l689_68910

theorem betta_fish_count 
  (total_guppies_per_day : ℕ) 
  (moray_eel_consumption : ℕ) 
  (betta_fish_consumption : ℕ) 
  (betta_fish_count : ℕ) 
  (h_total : total_guppies_per_day = 55)
  (h_eel : moray_eel_consumption = 20)
  (h_betta : betta_fish_consumption = 7) 
  (h_eq : total_guppies_per_day - moray_eel_consumption = betta_fish_consumption * betta_fish_count) : 
  betta_fish_count = 5 :=
by 
  sorry

end NUMINAMATH_GPT_betta_fish_count_l689_68910


namespace NUMINAMATH_GPT_total_minutes_exercised_l689_68999

-- Defining the conditions
def Javier_minutes_per_day : Nat := 50
def Javier_days : Nat := 10

def Sanda_minutes_day_90 : Nat := 90
def Sanda_days_90 : Nat := 3

def Sanda_minutes_day_75 : Nat := 75
def Sanda_days_75 : Nat := 2

def Sanda_minutes_day_45 : Nat := 45
def Sanda_days_45 : Nat := 4

-- Main statement to prove
theorem total_minutes_exercised : 
  (Javier_minutes_per_day * Javier_days) + 
  (Sanda_minutes_day_90 * Sanda_days_90) +
  (Sanda_minutes_day_75 * Sanda_days_75) +
  (Sanda_minutes_day_45 * Sanda_days_45) = 1100 := by
  sorry

end NUMINAMATH_GPT_total_minutes_exercised_l689_68999


namespace NUMINAMATH_GPT_number_of_distinct_real_roots_l689_68956

theorem number_of_distinct_real_roots (f : ℝ → ℝ) (h : ∀ x, f x = |x| - (4 / x) - (3 * |x| / x)) : ∃ k, k = 1 :=
by
  sorry

end NUMINAMATH_GPT_number_of_distinct_real_roots_l689_68956


namespace NUMINAMATH_GPT_parametric_function_f_l689_68998

theorem parametric_function_f (f : ℚ → ℚ)
  (x y : ℝ) (t : ℚ) :
  y = 20 * t - 10 →
  y = (3 / 4 : ℝ) * x - 15 →
  x = f t →
  f t = (80 / 3) * t + 20 / 3 :=
by
  sorry

end NUMINAMATH_GPT_parametric_function_f_l689_68998


namespace NUMINAMATH_GPT_no_nondegenerate_triangle_l689_68990

def distinct_positive_integers (a b c : ℕ) : Prop :=
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧ (a ≠ b) ∧ (b ≠ c) ∧ (c ≠ a)

def nondegenerate_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem no_nondegenerate_triangle (a b c : ℕ)
  (h_distinct : distinct_positive_integers a b c)
  (h_gcd : Nat.gcd (Nat.gcd a b) c = 1)
  (h1 : a ∣ (b - c) ^ 2)
  (h2 : b ∣ (c - a) ^ 2)
  (h3 : c ∣ (a - b) ^ 2) :
  ¬nondegenerate_triangle a b c :=
sorry

end NUMINAMATH_GPT_no_nondegenerate_triangle_l689_68990


namespace NUMINAMATH_GPT_nines_appear_600_times_l689_68934

-- Define a function to count the number of nines in a single digit place
def count_nines_in_place (place_value : Nat) (range_start : Nat) (range_end : Nat) : Nat :=
  (range_end - range_start + 1) / 10 * place_value

-- Define a function to count the total number of nines in all places from 1 to 1000
def total_nines_1_to_1000 : Nat :=
  let counts_per_place := 3 * count_nines_in_place 100 1 1000
  counts_per_place

theorem nines_appear_600_times :
  total_nines_1_to_1000 = 600 :=
by
  -- Using specific step counts calculated in the problem
  have units_place := count_nines_in_place 100 1 1000
  have tens_place := units_place
  have hundreds_place := units_place
  have thousands_place := 0
  have total_nines := units_place + tens_place + hundreds_place + thousands_place
  sorry

end NUMINAMATH_GPT_nines_appear_600_times_l689_68934


namespace NUMINAMATH_GPT_Yan_ratio_distance_l689_68960

theorem Yan_ratio_distance (w x y : ℕ) (h : w > 0) (h_eq : y/w = x/w + (x + y)/(5 * w)) : x/y = 2/3 := by
  sorry

end NUMINAMATH_GPT_Yan_ratio_distance_l689_68960


namespace NUMINAMATH_GPT_max_magnitude_vector_sub_l689_68977

open Real

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
sqrt (v.1^2 + v.2^2)

noncomputable def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
(v1.1 - v2.1, v1.2 - v2.2)

theorem max_magnitude_vector_sub (a b : ℝ × ℝ)
  (ha : vector_magnitude a = 2)
  (hb : vector_magnitude b = 1) :
  ∃ θ : ℝ, |vector_magnitude (vector_sub a b)| = 3 :=
by
  use π  -- θ = π to minimize cos θ to be -1
  sorry

end NUMINAMATH_GPT_max_magnitude_vector_sub_l689_68977


namespace NUMINAMATH_GPT_grisha_cross_coloring_l689_68900

open Nat

theorem grisha_cross_coloring :
  let grid_size := 40
  let cutout_rect_width := 36
  let cutout_rect_height := 37
  let total_cells := grid_size * grid_size
  let cutout_cells := cutout_rect_width * cutout_rect_height
  let remaining_cells := total_cells - cutout_cells
  let cross_cells := 5
  -- the result we need to prove is 113
  (remaining_cells - cross_cells - ((cutout_rect_width + cutout_rect_height - 1) - 1)) = 113 := by
  sorry

end NUMINAMATH_GPT_grisha_cross_coloring_l689_68900


namespace NUMINAMATH_GPT_angle_at_3_40_pm_is_130_degrees_l689_68908

def position_minute_hand (minutes : ℕ) : ℝ :=
  6 * minutes

def position_hour_hand (hours minutes : ℕ) : ℝ :=
  30 * hours + 0.5 * minutes

def angle_between_hands (hours minutes : ℕ) : ℝ :=
  abs (position_minute_hand minutes - position_hour_hand hours minutes)

theorem angle_at_3_40_pm_is_130_degrees : angle_between_hands 3 40 = 130 :=
by
  sorry

end NUMINAMATH_GPT_angle_at_3_40_pm_is_130_degrees_l689_68908


namespace NUMINAMATH_GPT_factorize_poly_l689_68941

theorem factorize_poly :
  (x : ℤ) → x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) :=
by
  sorry

end NUMINAMATH_GPT_factorize_poly_l689_68941


namespace NUMINAMATH_GPT_pentagon_total_area_l689_68940

-- Conditions definition
variables {a b c d e : ℕ}
variables {side1 side2 side3 side4 side5 : ℕ} 
variables {h : ℕ}
variables {triangle_area : ℕ}
variables {trapezoid_area : ℕ}
variables {total_area : ℕ}

-- Specific conditions given in the problem
def pentagon_sides (a b c d e : ℕ) : Prop :=
  a = 18 ∧ b = 25 ∧ c = 30 ∧ d = 28 ∧ e = 25

def can_be_divided (triangle_area trapezoid_area total_area : ℕ) : Prop :=
  triangle_area = 225 ∧ trapezoid_area = 770 ∧ total_area = 995

-- Total area of the pentagon under given conditions
theorem pentagon_total_area 
  (h_div: can_be_divided triangle_area trapezoid_area total_area) 
  (h_sides: pentagon_sides a b c d e)
  (h: triangle_area + trapezoid_area = total_area) :
  total_area = 995 := 
by
  sorry

end NUMINAMATH_GPT_pentagon_total_area_l689_68940


namespace NUMINAMATH_GPT_quadratic_function_value_l689_68921

theorem quadratic_function_value (x1 x2 a b : ℝ) (h1 : a ≠ 0)
  (h2 : 2012 = a * x1^2 + b * x1 + 2009)
  (h3 : 2012 = a * x2^2 + b * x2 + 2009) :
  (a * (x1 + x2)^2 + b * (x1 + x2) + 2009) = 2009 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_function_value_l689_68921


namespace NUMINAMATH_GPT_diamond_problem_l689_68975

def diamond (a b : ℚ) : ℚ := a - 1 / b

theorem diamond_problem : (diamond (diamond 1 2) 3) - (diamond 1 (diamond 2 3)) = -7 / 30 := by
  sorry

end NUMINAMATH_GPT_diamond_problem_l689_68975


namespace NUMINAMATH_GPT_find_x_l689_68933

theorem find_x (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h1 : x = 2 + 1/y) (h2 : y = 2 + 1/x) (h3 : x + y = 5) : 
  x = (7 + Real.sqrt 5) / 2 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l689_68933


namespace NUMINAMATH_GPT_transformed_parabola_equation_l689_68973

-- Conditions
def original_parabola (x : ℝ) : ℝ := 3 * x^2
def translate_downwards (y : ℝ) : ℝ := y - 3

-- Translations
def translate_to_right (x : ℝ) : ℝ := x - 2
def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 2)^2 - 3

-- Assertion
theorem transformed_parabola_equation :
  (∀ x : ℝ, translate_downwards (original_parabola x) = 3 * (translate_to_right x)^2 - 3) := by
  sorry

end NUMINAMATH_GPT_transformed_parabola_equation_l689_68973


namespace NUMINAMATH_GPT_way_to_cut_grid_l689_68966

def grid_ways : ℕ := 17

def rectangles (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 2) ∧ count = 8

def square (size : ℕ × ℕ) (count : ℕ) := 
  size = (1, 1) ∧ count = 1

theorem way_to_cut_grid :
  (∃ ways : ℕ, ways = 10) ↔ 
  ∀ g ways, g = grid_ways → 
  (rectangles (1, 2) 8 ∧ square (1, 1) 1 → ways = 10) :=
by 
  sorry

end NUMINAMATH_GPT_way_to_cut_grid_l689_68966


namespace NUMINAMATH_GPT_multiple_of_3_iff_has_odd_cycle_l689_68954

-- Define the undirected simple graph G
variable {V : Type} (G : SimpleGraph V)

-- Define the function f(G) which counts the number of acyclic orientations
def f (G : SimpleGraph V) : ℕ := sorry

-- Define what it means for a graph to have an odd-length cycle
def has_odd_cycle (G : SimpleGraph V) : Prop := sorry

-- The theorem statement
theorem multiple_of_3_iff_has_odd_cycle (G : SimpleGraph V) : 
  (f G) % 3 = 0 ↔ has_odd_cycle G := 
sorry

end NUMINAMATH_GPT_multiple_of_3_iff_has_odd_cycle_l689_68954


namespace NUMINAMATH_GPT_number_of_bottle_caps_l689_68957

def total_cost : ℝ := 25
def cost_per_bottle_cap : ℝ := 5

theorem number_of_bottle_caps : total_cost / cost_per_bottle_cap = 5 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_bottle_caps_l689_68957


namespace NUMINAMATH_GPT_weekend_price_of_coat_l689_68992

-- Definitions based on conditions
def original_price : ℝ := 250
def sale_price_discount : ℝ := 0.4
def weekend_additional_discount : ℝ := 0.3

-- To prove the final weekend price
theorem weekend_price_of_coat :
  (original_price * (1 - sale_price_discount) * (1 - weekend_additional_discount)) = 105 := by
  sorry

end NUMINAMATH_GPT_weekend_price_of_coat_l689_68992


namespace NUMINAMATH_GPT_rate_of_current_l689_68951

variable (c : ℝ)
def effective_speed_downstream (c : ℝ) : ℝ := 4.5 + c
def effective_speed_upstream (c : ℝ) : ℝ := 4.5 - c

theorem rate_of_current
  (h1 : ∀ d : ℝ, d / (4.5 - c) = 2 * (d / (4.5 + c)))
  : c = 1.5 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_current_l689_68951


namespace NUMINAMATH_GPT_rectangle_area_l689_68904

-- Definitions of conditions
def width : ℝ := 5
def length : ℝ := 2 * width

-- The goal is to prove the area is 50 square inches given the length and width
theorem rectangle_area : length * width = 50 := by
  have h_length : length = 2 * width := by rfl
  have h_width : width = 5 := by rfl
  sorry

end NUMINAMATH_GPT_rectangle_area_l689_68904


namespace NUMINAMATH_GPT_cost_of_article_l689_68917

theorem cost_of_article (C: ℝ) (G: ℝ) (h1: 380 = C + G) (h2: 420 = C + G + 0.05 * C) : C = 800 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_l689_68917


namespace NUMINAMATH_GPT_number_of_boxes_needed_l689_68915

def total_bananas : ℕ := 40
def bananas_per_box : ℕ := 5

theorem number_of_boxes_needed : (total_bananas / bananas_per_box) = 8 := by
  sorry

end NUMINAMATH_GPT_number_of_boxes_needed_l689_68915


namespace NUMINAMATH_GPT_birches_planted_l689_68976

variable 
  (G B X : ℕ) -- G: number of girls, B: number of boys, X: number of birches

-- Conditions:
variable
  (h1 : G + B = 24) -- Total number of students
  (h2 : 3 * G + X = 24) -- Total number of plants
  (h3 : X = B / 3) -- Birches planted by boys

-- Proof statement:
theorem birches_planted : X = 6 :=
by 
  sorry

end NUMINAMATH_GPT_birches_planted_l689_68976


namespace NUMINAMATH_GPT_proof_problem_l689_68980

-- Definitions of the conditions
def domain_R (f : ℝ → ℝ) : Prop := ∀ x : ℝ, true

def symmetric_graph_pt (f : ℝ → ℝ) (a : ℝ) (b : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = 2 * b - f (a + x)

def symmetric (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = -f (x)

def symmetric_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (2*a - x) = f (x)

-- Definitions of the statements to prove
def statement_1 (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (y = f (x - 1) → y = f (1 - x) → x = 1)

def statement_2 (f : ℝ → ℝ) : Prop :=
  symmetric_line f (3 / 2)

def statement_3 (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 3) = -f (x)

-- Main proof problem
theorem proof_problem (f : ℝ → ℝ) 
  (h_domain : domain_R f)
  (h_symmetric_pt : symmetric_graph_pt f (-3 / 4) 0)
  (h_symmetric : ∀ x : ℝ, f (x + 3 / 2) = -f (x))
  (h_property : ∀ x : ℝ, f (x + 2) = -f (-x + 4)) :
  statement_1 f ∧ statement_2 f ∧ statement_3 f :=
sorry

end NUMINAMATH_GPT_proof_problem_l689_68980


namespace NUMINAMATH_GPT_smallest_cost_l689_68901

def gift1_choc := 3
def gift1_caramel := 15
def price1 := 350

def gift2_choc := 20
def gift2_caramel := 5
def price2 := 500

def equal_candies (m n : ℕ) : Prop :=
  gift1_choc * m + gift2_choc * n = gift1_caramel * m + gift2_caramel * n

def total_cost (m n : ℕ) : ℕ :=
  price1 * m + price2 * n

theorem smallest_cost :
  ∃ m n : ℕ, equal_candies m n ∧ total_cost m n = 3750 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_cost_l689_68901
