import Mathlib

namespace NUMINAMATH_GPT_range_of_x_plus_2y_minus_2z_l9_942

theorem range_of_x_plus_2y_minus_2z (x y z : ℝ) (h : x^2 + y^2 + z^2 = 4) : -6 ≤ x + 2 * y - 2 * z ∧ x + 2 * y - 2 * z ≤ 6 :=
sorry

end NUMINAMATH_GPT_range_of_x_plus_2y_minus_2z_l9_942


namespace NUMINAMATH_GPT_radius_tangent_circle_l9_981

theorem radius_tangent_circle (r r1 r2 : ℝ) (h_r1 : r1 = 3) (h_r2 : r2 = 5)
    (h_concentric : true) : r = 1 := by
  -- Definitions are given as conditions
  have h1 := r1 -- radius of smaller concentric circle
  have h2 := r2 -- radius of larger concentric circle
  have h3 := h_concentric -- the circles are concentric
  have h4 := h_r1 -- r1 = 3
  have h5 := h_r2 -- r2 = 5
  sorry

end NUMINAMATH_GPT_radius_tangent_circle_l9_981


namespace NUMINAMATH_GPT_zombie_count_today_l9_982

theorem zombie_count_today (Z : ℕ) (h : Z < 50) : 16 * Z = 48 :=
by
  -- Assume Z, h conditions from a)
  -- Proof will go here, for now replaced with sorry
  sorry

end NUMINAMATH_GPT_zombie_count_today_l9_982


namespace NUMINAMATH_GPT_polynomial_positive_for_all_reals_l9_930

theorem polynomial_positive_for_all_reals (m : ℝ) : m^6 - m^5 + m^4 + m^2 - m + 1 > 0 :=
by
  sorry

end NUMINAMATH_GPT_polynomial_positive_for_all_reals_l9_930


namespace NUMINAMATH_GPT_prove_equation_1_prove_equation_2_l9_903

theorem prove_equation_1 : 
  ∀ x, (x - 3) / (x - 2) - 1 = 3 / x ↔ x = 3 / 2 :=
by
  sorry

theorem prove_equation_2 :
  ¬∃ x, (x + 1) / (x - 1) - 4 / (x^2 - 1) = 1 :=
by
  sorry

end NUMINAMATH_GPT_prove_equation_1_prove_equation_2_l9_903


namespace NUMINAMATH_GPT_total_number_of_birds_l9_929

def bird_cages : Nat := 9
def parrots_per_cage : Nat := 2
def parakeets_per_cage : Nat := 6
def birds_per_cage : Nat := parrots_per_cage + parakeets_per_cage
def total_birds : Nat := bird_cages * birds_per_cage

theorem total_number_of_birds : total_birds = 72 := by
  sorry

end NUMINAMATH_GPT_total_number_of_birds_l9_929


namespace NUMINAMATH_GPT_derivative_at_zero_l9_937

-- Define the function f
def f (x : ℝ) : ℝ := x * (1 + x)

-- Statement of the problem: The derivative of f at 0 is 1
theorem derivative_at_zero : deriv f 0 = 1 := 
  sorry

end NUMINAMATH_GPT_derivative_at_zero_l9_937


namespace NUMINAMATH_GPT_number_of_truthful_dwarfs_is_correct_l9_946

-- Definitions and assumptions based on the given conditions
def x : ℕ := 4 -- number of truthful dwarfs
def y : ℕ := 6 -- number of lying dwarfs

-- Conditions
axiom total_dwarfs : x + y = 10
axiom total_hands_raised : x + 2 * y = 16

-- The proof statement
theorem number_of_truthful_dwarfs_is_correct : x = 4 := by
  have h1 : x + y = 10 := total_dwarfs
  have h2 : x + 2 * y = 16 := total_hands_raised
  sorry -- The proof follows from solving the system of equations


end NUMINAMATH_GPT_number_of_truthful_dwarfs_is_correct_l9_946


namespace NUMINAMATH_GPT_sandwich_cost_l9_915

-- Defining the cost of each sandwich and the known conditions
variable (S : ℕ) -- Cost of each sandwich in dollars

-- Conditions as hypotheses
def buys_three_sandwiches (S : ℕ) : ℕ := 3 * S
def buys_two_drinks (drink_cost : ℕ) : ℕ := 2 * drink_cost
def total_cost (sandwich_cost drink_cost total_amount : ℕ) : Prop := buys_three_sandwiches sandwich_cost + buys_two_drinks drink_cost = total_amount

-- Given conditions in the problem
def given_conditions : Prop :=
  (buys_two_drinks 4 = 8) ∧ -- Each drink costs $4
  (total_cost S 4 26)       -- Total spending is $26

-- Theorem to prove the cost of each sandwich
theorem sandwich_cost : given_conditions S → S = 6 :=
by sorry

end NUMINAMATH_GPT_sandwich_cost_l9_915


namespace NUMINAMATH_GPT_find_m_l9_961

theorem find_m (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : (4 / x) + (9 / y) = m) (h4 : ∃ x y , x + y = 5/6) : m = 30 :=
sorry

end NUMINAMATH_GPT_find_m_l9_961


namespace NUMINAMATH_GPT_quadratic_inequality_range_l9_945

theorem quadratic_inequality_range (a : ℝ) :
  (∀ x : ℝ, (1/2) * a * x^2 - a * x + 2 > 0) ↔ a ∈ Set.Ico 0 4 := 
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_range_l9_945


namespace NUMINAMATH_GPT_total_participants_l9_927

-- Define the number of indoor and outdoor participants
variables (x y : ℕ)

-- First condition: number of outdoor participants is 480 more than indoor participants
def condition1 : Prop := y = x + 480

-- Second condition: moving 50 participants results in outdoor participants being 5 times the indoor participants
def condition2 : Prop := y + 50 = 5 * (x - 50)

-- Theorem statement: the total number of participants is 870
theorem total_participants (h1 : condition1 x y) (h2 : condition2 x y) : x + y = 870 :=
sorry

end NUMINAMATH_GPT_total_participants_l9_927


namespace NUMINAMATH_GPT_total_rabbits_and_chickens_l9_991

theorem total_rabbits_and_chickens (r c : ℕ) (h₁ : r = 64) (h₂ : r = c + 17) : r + c = 111 :=
by {
  sorry
}

end NUMINAMATH_GPT_total_rabbits_and_chickens_l9_991


namespace NUMINAMATH_GPT_counties_percentage_l9_943

theorem counties_percentage (a b c : ℝ) (ha : a = 0.2) (hb : b = 0.35) (hc : c = 0.25) :
  a + b + c = 0.8 :=
by
  rw [ha, hb, hc]
  sorry

end NUMINAMATH_GPT_counties_percentage_l9_943


namespace NUMINAMATH_GPT_calculate_product_value_l9_934

theorem calculate_product_value :
    (1 / 4) * 8 * (1 / 16) * 32 * (1 / 64) * 128 * (1 / 256) * 512 * (1 / 1024) * 2048 = 32 :=
by
  sorry

end NUMINAMATH_GPT_calculate_product_value_l9_934


namespace NUMINAMATH_GPT_number_square_25_l9_948

theorem number_square_25 (x : ℝ) : x^2 = 25 ↔ x = 5 ∨ x = -5 := 
sorry

end NUMINAMATH_GPT_number_square_25_l9_948


namespace NUMINAMATH_GPT_evaluate_expression_at_two_l9_992

theorem evaluate_expression_at_two : (2 * (2:ℝ)^2 - 3 * 2 + 4) = 6 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_at_two_l9_992


namespace NUMINAMATH_GPT_locus_of_point_P_l9_923

-- Definitions and conditions
def circle_M (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 4
def A_point : ℝ × ℝ := (2, 1)
def chord_BC (x y x₀ y₀ : ℝ) : Prop := (x₀ - 1) * x + y₀ * y - x₀ - 3 = 0
def point_P_locus (x₀ y₀ : ℝ) : Prop := ∃ x y, (chord_BC x y x₀ y₀) ∧ x = 2 ∧ y = 1

-- Lean 4 statement to be proved
theorem locus_of_point_P (x₀ y₀ : ℝ) (h : point_P_locus x₀ y₀) : x₀ + y₀ - 5 = 0 :=
  by
  sorry

end NUMINAMATH_GPT_locus_of_point_P_l9_923


namespace NUMINAMATH_GPT_probability_of_pink_gumball_l9_931

theorem probability_of_pink_gumball 
  (P B : ℕ) 
  (total_gumballs : P + B > 0)
  (prob_blue_blue : ((B : ℚ) / (B + P))^2 = 16 / 49) : 
  (B + P > 0) → ((P : ℚ) / (B + P) = 3 / 7) :=
by
  sorry

end NUMINAMATH_GPT_probability_of_pink_gumball_l9_931


namespace NUMINAMATH_GPT_douglas_vote_percentage_is_66_l9_901

noncomputable def percentDouglasVotes (v : ℝ) : ℝ :=
  let votesX := 0.74 * (2 * v)
  let votesY := 0.5000000000000002 * v
  let totalVotes := 3 * v
  let totalDouglasVotes := votesX + votesY
  (totalDouglasVotes / totalVotes) * 100

theorem douglas_vote_percentage_is_66 :
  ∀ v : ℝ, percentDouglasVotes v = 66 := 
by
  intros v
  unfold percentDouglasVotes
  sorry

end NUMINAMATH_GPT_douglas_vote_percentage_is_66_l9_901


namespace NUMINAMATH_GPT_rhombus_perimeter_l9_951

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 10) : 
  (4 * (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))) = 52 := by
  sorry

end NUMINAMATH_GPT_rhombus_perimeter_l9_951


namespace NUMINAMATH_GPT_fuel_consumption_gallons_l9_997

theorem fuel_consumption_gallons
  (distance_per_liter : ℝ)
  (speed_mph : ℝ)
  (time_hours : ℝ)
  (mile_to_km : ℝ)
  (gallon_to_liters : ℝ)
  (fuel_consumption : ℝ) :
  distance_per_liter = 56 →
  speed_mph = 91 →
  time_hours = 5.7 →
  mile_to_km = 1.6 →
  gallon_to_liters = 3.8 →
  fuel_consumption = 3.9 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_fuel_consumption_gallons_l9_997


namespace NUMINAMATH_GPT_area_of_sandbox_is_correct_l9_966

-- Define the length and width of the sandbox
def length_sandbox : ℕ := 312
def width_sandbox : ℕ := 146

-- Define the area calculation
def area_sandbox (length width : ℕ) : ℕ := length * width

-- The theorem stating that the area of the sandbox is 45552 cm²
theorem area_of_sandbox_is_correct : area_sandbox length_sandbox width_sandbox = 45552 := sorry

end NUMINAMATH_GPT_area_of_sandbox_is_correct_l9_966


namespace NUMINAMATH_GPT_sum_2019_l9_974

noncomputable def a : ℕ → ℝ := sorry
def S (n : ℕ) : ℝ := sorry

axiom prop_1 : (a 2 - 1)^3 + (a 2 - 1) = 2019
axiom prop_2 : (a 2018 - 1)^3 + (a 2018 - 1) = -2019
axiom arithmetic_sequence : ∀ n, a (n + 1) - a n = a 2 - a 1
axiom sum_formula : S 2019 = (2019 * (a 1 + a 2019)) / 2

theorem sum_2019 : S 2019 = 2019 :=
by sorry

end NUMINAMATH_GPT_sum_2019_l9_974


namespace NUMINAMATH_GPT_find_d_squared_l9_977

noncomputable def g (z : ℂ) (c d : ℝ) : ℂ := (c + d * Complex.I) * z

theorem find_d_squared (c d : ℝ) (z : ℂ) (h1 : ∀ z : ℂ, Complex.abs (g z c d - z) = 2 * Complex.abs (g z c d)) (h2 : Complex.abs (c + d * Complex.I) = 6) : d^2 = 11305 / 4 := 
sorry

end NUMINAMATH_GPT_find_d_squared_l9_977


namespace NUMINAMATH_GPT_girls_joined_school_l9_968

theorem girls_joined_school
  (initial_girls : ℕ)
  (initial_boys : ℕ)
  (total_pupils_after : ℕ)
  (computed_new_girls : ℕ) :
  initial_girls = 706 →
  initial_boys = 222 →
  total_pupils_after = 1346 →
  computed_new_girls = total_pupils_after - (initial_girls + initial_boys) →
  computed_new_girls = 418 :=
by
  intros h_initial_girls h_initial_boys h_total_pupils_after h_computed_new_girls
  sorry

end NUMINAMATH_GPT_girls_joined_school_l9_968


namespace NUMINAMATH_GPT_initial_mean_corrected_observations_l9_925

theorem initial_mean_corrected_observations:
  ∃ M : ℝ, 
  (∀ (Sum_initial Sum_corrected : ℝ), 
    Sum_initial = 50 * M ∧ 
    Sum_corrected = Sum_initial + (48 - 23) → 
    Sum_corrected / 50 = 41.5) →
  M = 41 :=
by
  sorry

end NUMINAMATH_GPT_initial_mean_corrected_observations_l9_925


namespace NUMINAMATH_GPT_ratio_equivalence_l9_972

theorem ratio_equivalence (m n s u : ℚ) (h1 : m / n = 5 / 4) (h2 : s / u = 8 / 15) :
  (5 * m * s - 2 * n * u) / (7 * n * u - 10 * m * s) = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l9_972


namespace NUMINAMATH_GPT_max_actors_chess_tournament_l9_933

-- Definitions based on conditions
variable {α : Type} [Fintype α] [DecidableEq α]

-- Each actor played with every other actor exactly once.
def played_with_everyone (R : α → α → ℝ) : Prop :=
  ∀ a b, a ≠ b → (R a b = 1 ∨ R a b = 0.5 ∨ R a b = 0)

-- Among every three participants, one earned exactly 1.5 solidus in matches against the other two.
def condition_1_5_solidi (R : α → α → ℝ) : Prop :=
  ∀ a b c, a ≠ b → b ≠ c → a ≠ c → 
   (R a b + R a c = 1.5 ∨ R b a + R b c = 1.5 ∨ R c a + R c b = 1.5)

-- Prove the maximum number of such participants is 5
theorem max_actors_chess_tournament (actors : Finset α) (R : α → α → ℝ) 
  (h_played : played_with_everyone R) (h_condition : condition_1_5_solidi R) :
  actors.card ≤ 5 :=
  sorry

end NUMINAMATH_GPT_max_actors_chess_tournament_l9_933


namespace NUMINAMATH_GPT_number_of_dogs_l9_939

theorem number_of_dogs 
  (d c b : Nat) 
  (ratio : d / c / b = 3 / 7 / 12) 
  (total_dogs_and_bunnies : d + b = 375) :
  d = 75 :=
by
  -- Using the hypothesis and given conditions to prove d = 75.
  sorry

end NUMINAMATH_GPT_number_of_dogs_l9_939


namespace NUMINAMATH_GPT_find_S2019_l9_956

-- Conditions given in the problem
variable (a : ℕ → ℝ)
variable (S : ℕ → ℝ)

-- Definitions and conditions extracted: conditions for sum of arithmetic sequence
axiom arithmetic_sum (n : ℕ) : S n = n * a (n / 2)
axiom OB_condition : a 3 + a 2017 = 1

-- Lean statement to prove S2019
theorem find_S2019 : S 2019 = 2019 / 2 := by
  sorry

end NUMINAMATH_GPT_find_S2019_l9_956


namespace NUMINAMATH_GPT_ratio_mark_days_used_l9_975

-- Defining the conditions
def num_sick_days : ℕ := 10
def num_vacation_days : ℕ := 10
def total_hours_left : ℕ := 80
def hours_per_workday : ℕ := 8

-- Total days allotted
def total_days_allotted : ℕ :=
  num_sick_days + num_vacation_days

-- Days left for Mark
def days_left : ℕ :=
  total_hours_left / hours_per_workday

-- Days used by Mark
def days_used : ℕ :=
  total_days_allotted - days_left

-- The ratio of days used to total days allotted (expected to be 1:2)
def ratio_used_to_allotted : ℚ :=
  days_used / total_days_allotted

theorem ratio_mark_days_used :
  ratio_used_to_allotted = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_mark_days_used_l9_975


namespace NUMINAMATH_GPT_consecutive_sum_divisible_by_12_l9_965

theorem consecutive_sum_divisible_by_12 
  (b : ℤ) 
  (a : ℤ := b - 1) 
  (c : ℤ := b + 1) 
  (d : ℤ := b + 2) :
  ∃ k : ℤ, ab + ac + ad + bc + bd + cd + 1 = 12 * k := by
  sorry

end NUMINAMATH_GPT_consecutive_sum_divisible_by_12_l9_965


namespace NUMINAMATH_GPT_marbles_lost_l9_947

theorem marbles_lost (m_initial m_current : ℕ) (h_initial : m_initial = 19) (h_current : m_current = 8) : m_initial - m_current = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_marbles_lost_l9_947


namespace NUMINAMATH_GPT_common_value_of_4a_and_5b_l9_941

theorem common_value_of_4a_and_5b (a b C : ℝ) (h1 : 4 * a = C) (h2 : 5 * b = C) (h3 : 40 * a * b = 1800) :
  C = 60 :=
sorry

end NUMINAMATH_GPT_common_value_of_4a_and_5b_l9_941


namespace NUMINAMATH_GPT_calculate_square_of_complex_l9_949

theorem calculate_square_of_complex (i : ℂ) (h : i^2 = -1) : (1 - i)^2 = -2 * i :=
by
  sorry

end NUMINAMATH_GPT_calculate_square_of_complex_l9_949


namespace NUMINAMATH_GPT_only_sqrt_three_is_irrational_l9_911

-- Definitions based on conditions
def zero_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (0 : ℝ) = p / q
def neg_three_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (-3 : ℝ) = p / q
def one_third_rational : Prop := ∃ p q : ℤ, q ≠ 0 ∧ (1/3 : ℝ) = p / q
def sqrt_three_irrational : Prop := ¬ ∃ p q : ℤ, q ≠ 0 ∧ (Real.sqrt 3) = p / q

-- The proof problem statement
theorem only_sqrt_three_is_irrational :
  zero_rational ∧
  neg_three_rational ∧
  one_third_rational ∧
  sqrt_three_irrational :=
by sorry

end NUMINAMATH_GPT_only_sqrt_three_is_irrational_l9_911


namespace NUMINAMATH_GPT_function_decreasing_odd_function_m_zero_l9_962

-- First part: Prove that the function is decreasing
theorem function_decreasing (m : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
    let f := fun x => -2 * x + m
    f x1 > f x2 :=
by
    sorry

-- Second part: Find the value of m when the function is odd
theorem odd_function_m_zero (m : ℝ) :
    (∀ x : ℝ, let f := fun x => -2 * x + m
              f (-x) = -f x) → m = 0 :=
by
    sorry

end NUMINAMATH_GPT_function_decreasing_odd_function_m_zero_l9_962


namespace NUMINAMATH_GPT_B_work_rate_l9_998

theorem B_work_rate (A B C : ℕ) (combined_work_rate_A_B_C : ℕ)
  (A_work_days B_work_days C_work_days : ℕ)
  (combined_abc : combined_work_rate_A_B_C = 4)
  (a_work_rate : A_work_days = 6)
  (c_work_rate : C_work_days = 36) :
  B = 18 :=
by
  sorry

end NUMINAMATH_GPT_B_work_rate_l9_998


namespace NUMINAMATH_GPT_quadratic_with_roots_1_and_2_l9_994

theorem quadratic_with_roots_1_and_2 : ∃ (a b c : ℝ), (a = 1 ∧ b = 2) ∧ (∀ x : ℝ, x ≠ 1 → x ≠ 2 → a * x^2 + b * x + c = 0) ∧ (a * x^2 + b * x + c = x^2 - 3 * x + 2) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_with_roots_1_and_2_l9_994


namespace NUMINAMATH_GPT_max_expression_value_l9_960

theorem max_expression_value {x y : ℝ} (h1 : |x - y| ≤ 2) (h2 : |3 * x + y| ≤ 6) :
  x^2 + y^2 ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_expression_value_l9_960


namespace NUMINAMATH_GPT_sum_of_excluded_solutions_l9_932

noncomputable def P : ℚ := 3
noncomputable def Q : ℚ := 5 / 3
noncomputable def R : ℚ := 25 / 3

theorem sum_of_excluded_solutions :
    (P = 3) ∧
    (Q = 5 / 3) ∧
    (R = 25 / 3) ∧
    (∀ x, (x ≠ -R ∧ x ≠ -10) →
    ((x + Q) * (P * x + 50) / ((x + R) * (x + 10)) = 3)) →
    (-R + -10 = -55 / 3) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_excluded_solutions_l9_932


namespace NUMINAMATH_GPT_prob_two_red_two_blue_is_3_over_14_l9_900

def red_marbles : ℕ := 15
def blue_marbles : ℕ := 10
def total_marbles : ℕ := red_marbles + blue_marbles
def chosen_marbles : ℕ := 4

noncomputable def prob_two_red_two_blue : ℚ :=
  let total_ways := (Nat.choose total_marbles chosen_marbles : ℚ)
  let ways_two_red := (Nat.choose red_marbles 2)
  let ways_two_blue := (Nat.choose blue_marbles 2)
  let favorable_outcomes := 6 * ways_two_red * ways_two_blue
  favorable_outcomes / total_ways

theorem prob_two_red_two_blue_is_3_over_14 : prob_two_red_two_blue = 3 / 14 :=
  sorry

end NUMINAMATH_GPT_prob_two_red_two_blue_is_3_over_14_l9_900


namespace NUMINAMATH_GPT_inequality_solution_set_l9_919

theorem inequality_solution_set (x : ℝ) : (-2 < x ∧ x ≤ 3) ↔ (x - 3) / (x + 2) ≤ 0 := 
sorry

end NUMINAMATH_GPT_inequality_solution_set_l9_919


namespace NUMINAMATH_GPT_possible_values_for_p_t_l9_953

theorem possible_values_for_p_t (p q r s t : ℝ)
(h₁ : |p - q| = 3)
(h₂ : |q - r| = 4)
(h₃ : |r - s| = 5)
(h₄ : |s - t| = 6) :
  ∃ (v : Finset ℝ), v = {0, 2, 4, 6, 8, 10, 12, 18} ∧ |p - t| ∈ v :=
sorry

end NUMINAMATH_GPT_possible_values_for_p_t_l9_953


namespace NUMINAMATH_GPT_initial_boys_down_slide_l9_971

variable (B : Int)

theorem initial_boys_down_slide:
  B + 13 = 35 → B = 22 := by
  sorry

end NUMINAMATH_GPT_initial_boys_down_slide_l9_971


namespace NUMINAMATH_GPT_simplify_to_linear_form_l9_906

theorem simplify_to_linear_form (p : ℤ) : 
  ((7 * p + 3) - 3 * p * 6) * 5 + (5 - 2 / 4) * (8 * p - 12) = -19 * p - 39 := 
by 
  sorry

end NUMINAMATH_GPT_simplify_to_linear_form_l9_906


namespace NUMINAMATH_GPT_simple_interest_rate_l9_985

theorem simple_interest_rate (P R: ℝ) (T: ℝ) (H: T = 5) (H1: P * (1/6) = P * (R * T / 100)) : R = 10/3 :=
by {
  sorry
}

end NUMINAMATH_GPT_simple_interest_rate_l9_985


namespace NUMINAMATH_GPT_truncatedPyramidVolume_l9_916

noncomputable def volumeOfTruncatedPyramid (R : ℝ) : ℝ :=
  let h := R * Real.sqrt 3 / 2
  let S_lower := 3 * R^2 * Real.sqrt 3 / 2
  let S_upper := 3 * R^2 * Real.sqrt 3 / 8
  let sqrt_term := Real.sqrt (S_lower * S_upper)
  (1/3) * h * (S_lower + S_upper + sqrt_term)

theorem truncatedPyramidVolume (R : ℝ) (h := R * Real.sqrt 3 / 2)
  (S_lower := 3 * R^2 * Real.sqrt 3 / 2)
  (S_upper := 3 * R^2 * Real.sqrt 3 / 8)
  (V := (1/3) * h * (S_lower + S_upper + Real.sqrt (S_lower * S_upper))) :
  volumeOfTruncatedPyramid R = 21 * R^3 / 16 := by
  sorry

end NUMINAMATH_GPT_truncatedPyramidVolume_l9_916


namespace NUMINAMATH_GPT_units_digit_of_square_l9_970

theorem units_digit_of_square (n : ℤ) (h : (n^2 / 10) % 10 = 7) : (n^2 % 10) = 6 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_of_square_l9_970


namespace NUMINAMATH_GPT_sin_2theta_plus_pi_div_2_l9_914

theorem sin_2theta_plus_pi_div_2 (θ : ℝ) (hθ1 : 0 < θ) (hθ2 : θ < π / 4)
    (h_tan2θ : Real.tan (2 * θ) = Real.cos θ / (2 - Real.sin θ)) :
    Real.sin (2 * θ + π / 2) = 7 / 8 :=
sorry

end NUMINAMATH_GPT_sin_2theta_plus_pi_div_2_l9_914


namespace NUMINAMATH_GPT_min_value_proof_l9_928

noncomputable def min_expr_value (x y : ℝ) : ℝ :=
  (1 / (2 * x)) + (1 / y)

theorem min_value_proof (x y : ℝ) (h1 : x + y = 1) (h2 : y > 0) (h3 : x > 0) :
  min_expr_value x y = (3 / 2) + Real.sqrt 2 :=
sorry

end NUMINAMATH_GPT_min_value_proof_l9_928


namespace NUMINAMATH_GPT_smallest_x_for_multiple_l9_913

theorem smallest_x_for_multiple (x : ℕ) : (450 * x) % 720 = 0 ↔ x = 8 := 
by {
  sorry
}

end NUMINAMATH_GPT_smallest_x_for_multiple_l9_913


namespace NUMINAMATH_GPT_factor_of_lcm_l9_909

theorem factor_of_lcm (A B hcf : ℕ) (h_gcd : Nat.gcd A B = hcf) (hcf_eq : hcf = 16) (A_eq : A = 224) :
  ∃ X : ℕ, X = 14 := by
  sorry

end NUMINAMATH_GPT_factor_of_lcm_l9_909


namespace NUMINAMATH_GPT_equal_costs_at_60_minutes_l9_955

-- Define the base rates and the per minute rates for each company
def base_rate_united : ℝ := 9.00
def rate_per_minute_united : ℝ := 0.25
def base_rate_atlantic : ℝ := 12.00
def rate_per_minute_atlantic : ℝ := 0.20

-- Define the total cost functions
def cost_united (m : ℝ) : ℝ := base_rate_united + rate_per_minute_united * m
def cost_atlantic (m : ℝ) : ℝ := base_rate_atlantic + rate_per_minute_atlantic * m

-- State the theorem to be proved
theorem equal_costs_at_60_minutes : 
  ∃ (m : ℝ), cost_united m = cost_atlantic m ∧ m = 60 :=
by
  -- Pending proof
  sorry

end NUMINAMATH_GPT_equal_costs_at_60_minutes_l9_955


namespace NUMINAMATH_GPT_Mary_more_than_Tim_l9_967

-- Define the incomes
variables (J T M : ℝ)

-- Conditions
def Tim_income : Prop := T = 0.80 * J
def Mary_income : Prop := M = 1.28 * J

-- Theorem statement to prove
theorem Mary_more_than_Tim (J T M : ℝ) (h1 : Tim_income J T)
  (h2 : Mary_income J M) : ((M - T) / T) * 100 = 60 :=
by
  -- Including sorry to skip the proof
  sorry

end NUMINAMATH_GPT_Mary_more_than_Tim_l9_967


namespace NUMINAMATH_GPT_solution_to_problem_l9_920

def problem_statement : Prop :=
  (3^202 + 7^203)^2 - (3^202 - 7^203)^2 = 59 * 10^202

theorem solution_to_problem : problem_statement := 
  by sorry

end NUMINAMATH_GPT_solution_to_problem_l9_920


namespace NUMINAMATH_GPT_find_function_expression_point_on_function_graph_l9_935

-- Problem setup
def y_minus_2_is_directly_proportional_to_x (y x : ℝ) : Prop :=
  ∃ k : ℝ, y - 2 = k * x

-- Conditions
def specific_condition : Prop :=
  y_minus_2_is_directly_proportional_to_x 6 1

-- Function expression derivation
theorem find_function_expression : ∃ k, ∀ x, 6 - 2 = k * 1 ∧ ∀ y, y = k * x + 2 :=
sorry

-- Given point P belongs to the function graph
theorem point_on_function_graph (a : ℝ) : (∀ x y, y = 4 * x + 2) → ∃ a, 4 * a + 2 = -1 :=
sorry

end NUMINAMATH_GPT_find_function_expression_point_on_function_graph_l9_935


namespace NUMINAMATH_GPT_work_finished_earlier_due_to_additional_men_l9_993

-- Define the conditions as given facts in Lean
def original_men := 10
def original_days := 12
def additional_men := 10

-- State the theorem to be proved
theorem work_finished_earlier_due_to_additional_men :
  let total_men := original_men + additional_men
  let original_work := original_men * original_days
  let days_earlier := original_days - x
  original_work = total_men * days_earlier → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_work_finished_earlier_due_to_additional_men_l9_993


namespace NUMINAMATH_GPT_wall_building_time_l9_990

variable (r : ℝ) -- rate at which one worker can build the wall
variable (W : ℝ) -- the wall in units, let’s denote one whole wall as 1 unit

theorem wall_building_time:
  (∀ (w t : ℝ), W = (60 * r) * t → W = (30 * r) * 6) :=
by
  sorry

end NUMINAMATH_GPT_wall_building_time_l9_990


namespace NUMINAMATH_GPT_greatest_int_less_than_50_satisfying_conditions_l9_938

def satisfies_conditions (n : ℕ) : Prop :=
  n < 50 ∧ Int.gcd n 18 = 6

theorem greatest_int_less_than_50_satisfying_conditions :
  ∃ n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → m ≤ n ∧ n = 42 :=
by
  sorry

end NUMINAMATH_GPT_greatest_int_less_than_50_satisfying_conditions_l9_938


namespace NUMINAMATH_GPT_sequence_divisible_by_11_l9_979

theorem sequence_divisible_by_11 
  (a : ℕ → ℕ) 
  (h₁ : a 1 = 1) 
  (h₂ : a 2 = 3) 
  (h₃ : ∀ n : ℕ, a (n + 2) = (n + 3) * a (n + 1) - (n + 2) * a n) :
  (∀ n, n = 4 ∨ n = 8 ∨ n ≥ 10 → 11 ∣ a n) := sorry

end NUMINAMATH_GPT_sequence_divisible_by_11_l9_979


namespace NUMINAMATH_GPT_trigonometric_expression_value_l9_964

theorem trigonometric_expression_value :
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  (1 - 1 / cos_30) * (1 + 1 / sin_60) * (1 - 1 / sin_30) * (1 + 1 / cos_60) = 3 / 4 :=
by
  let cos_30 := Real.sqrt 3 / 2
  let sin_60 := Real.sqrt 3 / 2
  let sin_30 := 1 / 2
  let cos_60 := 1 / 2
  sorry

end NUMINAMATH_GPT_trigonometric_expression_value_l9_964


namespace NUMINAMATH_GPT_average_of_other_half_l9_980

theorem average_of_other_half (avg : ℝ) (sum_half : ℝ) (n : ℕ) (n_half : ℕ)
    (h_avg : avg = 43.1)
    (h_sum_half : sum_half = 158.4)
    (h_n : n = 8)
    (h_n_half : n_half = n / 2) :
    ((n * avg - sum_half) / n_half) = 46.6 :=
by
  -- The proof steps would be given here. We're omitting them as the prompt instructs.
  sorry

end NUMINAMATH_GPT_average_of_other_half_l9_980


namespace NUMINAMATH_GPT_spots_combined_l9_908

def Rover : ℕ := 46
def Cisco : ℕ := Rover / 2 - 5
def Granger : ℕ := 5 * Cisco

theorem spots_combined : Granger + Cisco = 108 := by
  sorry

end NUMINAMATH_GPT_spots_combined_l9_908


namespace NUMINAMATH_GPT_largest_C_inequality_l9_963

theorem largest_C_inequality :
  ∃ C : ℝ, C = Real.sqrt (8 / 3) ∧ ∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z) :=
by
  sorry

end NUMINAMATH_GPT_largest_C_inequality_l9_963


namespace NUMINAMATH_GPT_tricycles_count_l9_912

theorem tricycles_count (b t : ℕ) 
  (hyp1 : b + t = 10)
  (hyp2 : 2 * b + 3 * t = 26) : 
  t = 6 := 
by 
  sorry

end NUMINAMATH_GPT_tricycles_count_l9_912


namespace NUMINAMATH_GPT_parallelogram_area_l9_987

variable (base height : ℝ) (tripled_area_factor original_area new_area : ℝ)

theorem parallelogram_area (h_base : base = 6) (h_height : height = 20)
    (h_tripled_area_factor : tripled_area_factor = 9)
    (h_original_area_calc : original_area = base * height)
    (h_new_area_calc : new_area = original_area * tripled_area_factor) :
    original_area = 120 ∧ tripled_area_factor = 9 ∧ new_area = 1080 := by
  sorry

end NUMINAMATH_GPT_parallelogram_area_l9_987


namespace NUMINAMATH_GPT_Cody_games_l9_905

/-- Cody had nine old video games he wanted to get rid of.
He decided to give four of the games to his friend Jake,
three games to his friend Sarah, and one game to his friend Luke.
On Saturday he bought five new games.
How many games does Cody have now? -/
theorem Cody_games (nine_games initially: ℕ) (jake_games: ℕ) (sarah_games: ℕ) (luke_games: ℕ) (saturday_games: ℕ)
  (h_initial: initially = 9)
  (h_jake: jake_games = 4)
  (h_sarah: sarah_games = 3)
  (h_luke: luke_games = 1)
  (h_saturday: saturday_games = 5) :
  ((initially - (jake_games + sarah_games + luke_games)) + saturday_games) = 6 :=
by
  sorry

end NUMINAMATH_GPT_Cody_games_l9_905


namespace NUMINAMATH_GPT_number_of_triangles_and_squares_l9_999

theorem number_of_triangles_and_squares (x y : ℕ) (h1 : x + y = 13) (h2 : 3 * x + 4 * y = 47) : 
  x = 5 ∧ y = 8 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_and_squares_l9_999


namespace NUMINAMATH_GPT_box_max_volume_l9_910

theorem box_max_volume (x : ℝ) (h1 : 0 < x) (h2 : x < 5) :
    (10 - 2 * x) * (16 - 2 * x) * x ≤ 144 :=
by
  -- The proof will be filled here
  sorry

end NUMINAMATH_GPT_box_max_volume_l9_910


namespace NUMINAMATH_GPT_tonya_needs_to_eat_more_l9_922

-- Define the conditions in the problem
def ounces_per_hamburger : ℕ := 4
def ounces_eaten_last_year : ℕ := 84

-- Define a function to calculate hamburgers given ounces
def hamburgers_eaten (ounces : ℕ) (ounce_per_hamburger : ℕ) : ℕ :=
  ounces / ounce_per_hamburger

-- State the theorem
theorem tonya_needs_to_eat_more (ounces_per_hamburger ounces_eaten_last_year : ℕ) :
  hamburgers_eaten ounces_eaten_last_year ounces_per_hamburger + 1 = 22 := by
  sorry

end NUMINAMATH_GPT_tonya_needs_to_eat_more_l9_922


namespace NUMINAMATH_GPT_right_triangle_ratio_l9_940

theorem right_triangle_ratio (x y : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : (x - y)^2 + x^2 = (x + y)^2) : x / y = 4 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_ratio_l9_940


namespace NUMINAMATH_GPT_noah_left_lights_on_2_hours_l9_926

-- Define the conditions
def bedroom_light_usage : ℕ := 6
def office_light_usage : ℕ := 3 * bedroom_light_usage
def living_room_light_usage : ℕ := 4 * bedroom_light_usage
def total_energy_used : ℕ := 96
def total_energy_per_hour := bedroom_light_usage + office_light_usage + living_room_light_usage

-- Define the main theorem to prove
theorem noah_left_lights_on_2_hours : total_energy_used / total_energy_per_hour = 2 := by
  sorry

end NUMINAMATH_GPT_noah_left_lights_on_2_hours_l9_926


namespace NUMINAMATH_GPT_assignment_plans_proof_l9_921

noncomputable def total_assignment_plans : ℕ :=
  let volunteers := ["Xiao Zhang", "Xiao Zhao", "Xiao Li", "Xiao Luo", "Xiao Wang"]
  let positions := ["translation", "tour guide", "etiquette", "driver"]
  -- Definitions for eligible volunteers for the first two positions
  let first_positions := ["Xiao Zhang", "Xiao Zhao"]
  let remaining_positions := ["Xiao Li", "Xiao Luo", "Xiao Wang"]
  -- Assume the computation for the exact number which results in 36
  36

theorem assignment_plans_proof : total_assignment_plans = 36 := 
  by 
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_assignment_plans_proof_l9_921


namespace NUMINAMATH_GPT_shorter_side_of_rectangular_room_l9_983

theorem shorter_side_of_rectangular_room 
  (a b : ℕ) 
  (h1 : 2 * a + 2 * b = 52) 
  (h2 : a * b = 168) : 
  min a b = 12 := 
  sorry

end NUMINAMATH_GPT_shorter_side_of_rectangular_room_l9_983


namespace NUMINAMATH_GPT_infinite_solutions_abs_eq_ax_minus_2_l9_952

theorem infinite_solutions_abs_eq_ax_minus_2 (a : ℝ) :
  (∀ x : ℝ, |x - 2| = ax - 2) ↔ a = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_infinite_solutions_abs_eq_ax_minus_2_l9_952


namespace NUMINAMATH_GPT_partition_weights_l9_917

theorem partition_weights :
  ∃ A B C : Finset ℕ,
    (∀ x ∈ A, x ≤ 552) ∧
    (∀ x ∈ B, x ≤ 552) ∧
    (∀ x ∈ C, x ≤ 552) ∧
    ∀ x, (x ∈ A ∨ x ∈ B ∨ x ∈ C) ↔ 1 ≤ x ∧ x ≤ 552 ∧
    A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
    A.sum id = 50876 ∧ B.sum id = 50876 ∧ C.sum id = 50876 :=
by
  sorry

end NUMINAMATH_GPT_partition_weights_l9_917


namespace NUMINAMATH_GPT_simplify_expression_l9_976

theorem simplify_expression (a b : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_condition : a^3 + b^3 = 3 * (a + b)) : 
  (a / b + b / a + 1 / (a * b) = 4 / (a * b) + 1) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l9_976


namespace NUMINAMATH_GPT_expression_is_correct_l9_959

theorem expression_is_correct (a : ℝ) : 2 * (a + 1) = 2 * a + 1 := 
sorry

end NUMINAMATH_GPT_expression_is_correct_l9_959


namespace NUMINAMATH_GPT_integer_solution_l9_996

theorem integer_solution (n : ℤ) (hneq : n ≠ -2) :
  ∃ (m : ℤ), (n^3 + 8) = m * (n^2 - 4) ↔ n = 0 ∨ n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 6 := 
sorry

end NUMINAMATH_GPT_integer_solution_l9_996


namespace NUMINAMATH_GPT_curve_is_hyperbola_l9_958

theorem curve_is_hyperbola (m n x y : ℝ) (h_eq : m * x^2 - m * y^2 = n) (h_mn : m * n < 0) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ b^2/a^2 - x^2/a^2 = 1 := 
sorry

end NUMINAMATH_GPT_curve_is_hyperbola_l9_958


namespace NUMINAMATH_GPT_range_of_m_l9_989

def P (m : ℝ) : Prop :=
  9 - m > 2 * m ∧ 2 * m > 0

def Q (m : ℝ) : Prop :=
  m > 0 ∧ (Real.sqrt (6) / 2 < Real.sqrt (5 + m) / Real.sqrt (5)) ∧ (Real.sqrt (5 + m) / Real.sqrt (5) < Real.sqrt (2))

theorem range_of_m (m : ℝ) : ¬(P m ∧ Q m) ∧ (P m ∨ Q m) → (0 < m ∧ m ≤ 5 / 2) ∨ (3 ≤ m ∧ m < 5) :=
sorry

end NUMINAMATH_GPT_range_of_m_l9_989


namespace NUMINAMATH_GPT_equation_contains_2020_l9_950

def first_term (n : Nat) : Nat :=
  2 * n^2

theorem equation_contains_2020 :
  ∃ n, first_term n = 2020 :=
by
  use 31
  sorry

end NUMINAMATH_GPT_equation_contains_2020_l9_950


namespace NUMINAMATH_GPT_remaining_gallons_to_fill_tank_l9_902

-- Define the conditions as constants
def tank_capacity : ℕ := 50
def rate_seconds_per_gallon : ℕ := 20
def time_poured_minutes : ℕ := 6

-- Define the number of gallons poured per minute
def gallons_per_minute : ℕ := 60 / rate_seconds_per_gallon

def gallons_poured (minutes : ℕ) : ℕ :=
  minutes * gallons_per_minute

-- The main statement to prove the remaining gallons needed
theorem remaining_gallons_to_fill_tank : 
  tank_capacity - gallons_poured time_poured_minutes = 32 :=
by
  sorry

end NUMINAMATH_GPT_remaining_gallons_to_fill_tank_l9_902


namespace NUMINAMATH_GPT_ShielaDrawingsPerNeighbor_l9_969

-- Defining our problem using the given conditions:
def ShielaTotalDrawings : ℕ := 54
def ShielaNeighbors : ℕ := 6

-- Mathematically restating the problem:
theorem ShielaDrawingsPerNeighbor : (ShielaTotalDrawings / ShielaNeighbors) = 9 := by
  sorry

end NUMINAMATH_GPT_ShielaDrawingsPerNeighbor_l9_969


namespace NUMINAMATH_GPT_find_n_for_sum_l9_918

theorem find_n_for_sum (n : ℕ) : ∃ n, n * (2 * n - 1) = 2009 ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_for_sum_l9_918


namespace NUMINAMATH_GPT_ellipse_eq_line_eq_l9_984

-- Conditions for part (I)
def cond1 (a b : ℝ) : Prop := a > 0 ∧ b > 0 ∧ a > b
def pt_p_cond (PF1 PF2 : ℝ) : Prop := PF1 = 4 / 3 ∧ PF2 = 14 / 3 ∧ PF1^2 + PF2^2 = 1

-- Theorem for part (I)
theorem ellipse_eq (a b : ℝ) (PF1 PF2 : ℝ) (h₁ : cond1 a b) (h₂ : pt_p_cond PF1 PF2) : 
  (a = 3 ∧ b = 2 ∧ PF1 = 4 / 3 ∧ PF2 = 14 / 3) → 
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

-- Conditions for part (II)
def center_circle (M : ℝ × ℝ) : Prop := M = (-2, 1)
def pts_symmetric (A B M : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * M.1 ∧ A.2 + B.2 = 2 * M.2

-- Theorem for part (II)
theorem line_eq (A B M : ℝ × ℝ) (k : ℝ) (h₁ : center_circle M) (h₂ : pts_symmetric A B M) :
  k = 8 / 9 → (∀ x y : ℝ, 8 * x - 9 * y + 25 = 0) :=
sorry

end NUMINAMATH_GPT_ellipse_eq_line_eq_l9_984


namespace NUMINAMATH_GPT_average_of_last_three_l9_924

theorem average_of_last_three (A B C D : ℝ) 
  (h1 : (A + B + C) / 3 = 6)
  (h2 : A + D = 11)
  (h3 : D = 4) : 
  (B + C + D) / 3 = 5 :=
by
  sorry

end NUMINAMATH_GPT_average_of_last_three_l9_924


namespace NUMINAMATH_GPT_day_of_week_after_6_pow_2023_l9_988

def day_of_week_after_days (start_day : ℕ) (days : ℕ) : ℕ :=
  (start_day + days) % 7

theorem day_of_week_after_6_pow_2023 :
  day_of_week_after_days 4 (6^2023) = 3 :=
by
  sorry

end NUMINAMATH_GPT_day_of_week_after_6_pow_2023_l9_988


namespace NUMINAMATH_GPT_remainder_13_pow_150_mod_11_l9_954

theorem remainder_13_pow_150_mod_11 : (13^150) % 11 = 1 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_13_pow_150_mod_11_l9_954


namespace NUMINAMATH_GPT_binom_2p_p_mod_p_l9_944

theorem binom_2p_p_mod_p (p : ℕ) (hp : p.Prime) : Nat.choose (2 * p) p ≡ 2 [MOD p] := 
by
  sorry

end NUMINAMATH_GPT_binom_2p_p_mod_p_l9_944


namespace NUMINAMATH_GPT_tan_alpha_value_l9_995

theorem tan_alpha_value (α : ℝ) (h : Real.tan (π / 4 + α) = 1 / 2) : Real.tan α = -1 / 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_alpha_value_l9_995


namespace NUMINAMATH_GPT_find_M_at_x_eq_3_l9_957

noncomputable def M (a b c d x : ℝ) := a * x^5 + b * x^3 + c * x + d

theorem find_M_at_x_eq_3
  (a b c d M : ℝ)
  (h₀ : d = -5)
  (h₁ : 243 * a + 27 * b + 3 * c = -12) :
  M = -17 :=
by
  sorry

end NUMINAMATH_GPT_find_M_at_x_eq_3_l9_957


namespace NUMINAMATH_GPT_johns_average_speed_l9_907

-- Definitions of conditions
def total_time_hours : ℝ := 6.5
def total_distance_miles : ℝ := 255

-- Stating the problem to be proven
theorem johns_average_speed :
  (total_distance_miles / total_time_hours) = 39.23 := 
sorry

end NUMINAMATH_GPT_johns_average_speed_l9_907


namespace NUMINAMATH_GPT_sandy_friday_hours_l9_986

-- Define the conditions
def hourly_rate := 15
def saturday_hours := 6
def sunday_hours := 14
def total_earnings := 450

-- Define the proof problem
theorem sandy_friday_hours (F : ℝ) (h1 : F * hourly_rate + saturday_hours * hourly_rate + sunday_hours * hourly_rate = total_earnings) : F = 10 :=
sorry

end NUMINAMATH_GPT_sandy_friday_hours_l9_986


namespace NUMINAMATH_GPT_ratio_tough_to_good_sales_l9_978

-- Define the conditions
def tough_week_sales : ℤ := 800
def total_sales : ℤ := 10400
def good_weeks : ℕ := 5
def tough_weeks : ℕ := 3

-- Define the problem in Lean 4:
theorem ratio_tough_to_good_sales : ∃ G : ℤ, (good_weeks * G) + (tough_weeks * tough_week_sales) = total_sales ∧ 
  (tough_week_sales : ℚ) / (G : ℚ) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_ratio_tough_to_good_sales_l9_978


namespace NUMINAMATH_GPT_inequality_proof_l9_904

theorem inequality_proof (a b c d : ℝ) (h_nonneg : 0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d)
    (h_sum : a + b + c + d = 8) :
    (a^3 / (a^2 + b + c) + b^3 / (b^2 + c + d) + c^3 / (c^2 + d + a) + d^3 / (d^2 + a + b)) ≥ 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l9_904


namespace NUMINAMATH_GPT_surface_area_calculation_l9_936

-- Conditions:
-- Original rectangular sheet dimensions
def length : ℕ := 25
def width : ℕ := 35
-- Dimensions of the square corners
def corner_side : ℕ := 7

-- Surface area of the interior calculation
noncomputable def surface_area_interior : ℕ :=
  let original_area := length * width
  let corner_area := corner_side * corner_side
  let total_corner_area := 4 * corner_area
  original_area - total_corner_area

-- Theorem: The surface area of the interior of the resulting box
theorem surface_area_calculation : surface_area_interior = 679 := by
  -- You can fill in the details to compute the answer
  sorry

end NUMINAMATH_GPT_surface_area_calculation_l9_936


namespace NUMINAMATH_GPT_system1_solution_correct_system2_solution_correct_l9_973

theorem system1_solution_correct (x y : ℝ) (h1 : x + y = 5) (h2 : 4 * x - 2 * y = 2) :
    x = 2 ∧ y = 3 :=
  sorry

theorem system2_solution_correct (x y : ℝ) (h1 : 3 * x - 2 * y = 13) (h2 : 4 * x + 3 * y = 6) :
    x = 3 ∧ y = -2 :=
  sorry

end NUMINAMATH_GPT_system1_solution_correct_system2_solution_correct_l9_973
