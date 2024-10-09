import Mathlib

namespace b_in_terms_of_a_l1004_100453

noncomputable def a (k : ℝ) : ℝ := 3 + 3^k
noncomputable def b (k : ℝ) : ℝ := 3 + 3^(-k)

theorem b_in_terms_of_a (k : ℝ) :
  b k = (3 * (a k) - 8) / ((a k) - 3) := 
sorry

end b_in_terms_of_a_l1004_100453


namespace hyperbola_range_of_k_l1004_100471

theorem hyperbola_range_of_k (k : ℝ) :
  (∃ x y : ℝ, (x^2)/(k + 4) + (y^2)/(k - 1) = 1) → -4 < k ∧ k < 1 :=
by 
  sorry

end hyperbola_range_of_k_l1004_100471


namespace power_identity_l1004_100488

theorem power_identity {a n m k : ℝ} (h1: a^n = 2) (h2: a^m = 3) (h3: a^k = 4) :
  a^(2 * n + m - 2 * k) = 3 / 4 :=
by
  sorry

end power_identity_l1004_100488


namespace base_11_arithmetic_l1004_100467

-- Define the base and the numbers in base 11
def base := 11

def a := 6 * base^2 + 7 * base + 4  -- 674 in base 11
def b := 2 * base^2 + 7 * base + 9  -- 279 in base 11
def c := 1 * base^2 + 4 * base + 3  -- 143 in base 11
def result := 5 * base^2 + 5 * base + 9  -- 559 in base 11

theorem base_11_arithmetic :
  (a - b + c) = result :=
sorry

end base_11_arithmetic_l1004_100467


namespace shortest_ribbon_length_l1004_100412

theorem shortest_ribbon_length :
  ∃ (L : ℕ), (∀ (n : ℕ), n = 2 ∨ n = 5 ∨ n = 7 → L % n = 0) ∧ L = 70 :=
by
  sorry

end shortest_ribbon_length_l1004_100412


namespace product_of_two_numbers_l1004_100459

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y ≠ 0) 
  (h2 : (x + y) / (x - y) = 7)
  (h3 : xy = 24 * (x - y)) : xy = 48 := 
sorry

end product_of_two_numbers_l1004_100459


namespace evaluate_expression_at_3_l1004_100411

theorem evaluate_expression_at_3 : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_3_l1004_100411


namespace third_quadrant_angles_l1004_100491

theorem third_quadrant_angles :
  {α : ℝ | ∃ k : ℤ, π + 2 * k * π < α ∧ α < 3 * π / 2 + 2 * k * π} =
  {α | π < α ∧ α < 3 * π / 2} :=
sorry

end third_quadrant_angles_l1004_100491


namespace james_trip_time_l1004_100441

def speed : ℝ := 60
def distance : ℝ := 360
def stop_time : ℝ := 1

theorem james_trip_time:
  (distance / speed) + stop_time = 7 := 
by
  sorry

end james_trip_time_l1004_100441


namespace jude_age_today_l1004_100463
-- Import the necessary libraries

-- Define the conditions as hypotheses and then state the required proof
theorem jude_age_today (heath_age_today : ℕ) (heath_age_in_5_years : ℕ) (jude_age_in_5_years : ℕ) 
  (H1 : heath_age_today = 16)
  (H2 : heath_age_in_5_years = heath_age_today + 5)
  (H3 : heath_age_in_5_years = 3 * jude_age_in_5_years) :
  jude_age_in_5_years - 5 = 2 :=
by
  -- Given conditions imply Jude's age today is 2. Proof is omitted.
  sorry

end jude_age_today_l1004_100463


namespace train_speed_l1004_100407

noncomputable def distance : ℝ := 45  -- 45 km
noncomputable def time_minutes : ℝ := 30  -- 30 minutes
noncomputable def time_hours : ℝ := time_minutes / 60  -- Convert minutes to hours

theorem train_speed (d : ℝ) (t_m : ℝ) : d = 45 → t_m = 30 → d / (t_m / 60) = 90 :=
by
  intros h₁ h₂
  sorry

end train_speed_l1004_100407


namespace base_number_pow_19_mod_10_l1004_100461

theorem base_number_pow_19_mod_10 (x : ℕ) (h : x ^ 19 % 10 = 7) : x % 10 = 3 :=
sorry

end base_number_pow_19_mod_10_l1004_100461


namespace evaluate_expr_right_to_left_l1004_100469

variable (a b c d : ℝ)

theorem evaluate_expr_right_to_left :
  (a - b * c + d) = a - b * (c + d) :=
sorry

end evaluate_expr_right_to_left_l1004_100469


namespace polygon_properties_l1004_100455

-- Assume n is the number of sides of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_of_exterior_angles : ℝ := 360

-- Given the condition
def given_condition (n : ℕ) : Prop := sum_of_interior_angles n = 5 * sum_of_exterior_angles

theorem polygon_properties (n : ℕ) (h1 : given_condition n) :
  n = 12 ∧ (n * (n - 3)) / 2 = 54 :=
by
  sorry

end polygon_properties_l1004_100455


namespace number_of_players_l1004_100449

-- Definitions based on conditions
def socks_price : ℕ := 6
def tshirt_price : ℕ := socks_price + 7
def total_cost_per_player : ℕ := 2 * (socks_price + tshirt_price)
def total_expenditure : ℕ := 4092

-- Lean theorem statement
theorem number_of_players : total_expenditure / total_cost_per_player = 108 := 
by
  sorry

end number_of_players_l1004_100449


namespace henrikh_commute_distance_l1004_100498

theorem henrikh_commute_distance (x : ℕ)
    (h1 : ∀ y : ℕ, y = x → y = x)
    (h2 : 1 * x = x)
    (h3 : 20 * x = (x : ℕ))
    (h4 : x = (x / 3) + 8) :
    x = 12 := sorry

end henrikh_commute_distance_l1004_100498


namespace option_C_qualified_l1004_100450

-- Define the acceptable range
def lower_bound : ℝ := 25 - 0.2
def upper_bound : ℝ := 25 + 0.2

-- Define the option to be checked
def option_C : ℝ := 25.1

-- The theorem stating that option C is within the acceptable range
theorem option_C_qualified : lower_bound ≤ option_C ∧ option_C ≤ upper_bound := 
by 
  sorry

end option_C_qualified_l1004_100450


namespace seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l1004_100485

-- Define the sequences
def a_sq (n : ℕ) : ℕ := n ^ 2
def a_cube (n : ℕ) : ℕ := n ^ 3

-- First proof problem statement
theorem seq_satisfies_recurrence_sq :
  (a_sq 0 = 0) ∧ (a_sq 1 = 1) ∧ (a_sq 2 = 4) ∧ (a_sq 3 = 9) ∧ (a_sq 4 = 16) →
  (∀ n : ℕ, n ≥ 3 → a_sq n = 3 * a_sq (n - 1) - 3 * a_sq (n - 2) + a_sq (n - 3)) :=
by
  sorry

-- Second proof problem statement
theorem seq_satisfies_recurrence_cube :
  (a_cube 0 = 0) ∧ (a_cube 1 = 1) ∧ (a_cube 2 = 8) ∧ (a_cube 3 = 27) ∧ (a_cube 4 = 64) →
  (∀ n : ℕ, n ≥ 4 → a_cube n = 4 * a_cube (n - 1) - 6 * a_cube (n - 2) + 4 * a_cube (n - 3) - a_cube (n - 4)) :=
by
  sorry

end seq_satisfies_recurrence_sq_seq_satisfies_recurrence_cube_l1004_100485


namespace equation_elliptic_and_canonical_form_l1004_100482

-- Defining the necessary conditions and setup
def a11 := 1
def a12 := 1
def a22 := 2

def is_elliptic (a11 a12 a22 : ℝ) : Prop :=
  a12^2 - a11 * a22 < 0

def canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) : Prop :=
  let ξ := y - x
  let η := x
  let u_ξξ := u_xx -- Assuming u_xx represents u_ξξ after change of vars
  let u_ξη := u_xy
  let u_ηη := u_yy
  let u_ξ := u_x -- Assuming u_x represents u_ξ after change of vars
  let u_η := u_y
  u_ξξ + u_ηη = -2 * u_η + u + η + (ξ + η)^2

theorem equation_elliptic_and_canonical_form (u_xx u_xy u_yy u_x u_y u x y : ℝ) :
  is_elliptic a11 a12 a22 ∧
  canonical_form u_xx u_xy u_yy u_x u_y u x y :=
by
  sorry -- Proof to be completed

end equation_elliptic_and_canonical_form_l1004_100482


namespace bus_problem_l1004_100427

theorem bus_problem (x : ℕ)
  (h1 : 28 + 82 - x = 30) :
  82 - x = 2 :=
by {
  sorry
}

end bus_problem_l1004_100427


namespace integer_modulo_solution_l1004_100493

theorem integer_modulo_solution :
  ∃ n : ℤ, 0 ≤ n ∧ n < 137 ∧ 12345 ≡ n [ZMOD 137] ∧ n = 15 :=
sorry

end integer_modulo_solution_l1004_100493


namespace average_population_is_1000_l1004_100462

-- Define the populations of the villages.
def populations : List ℕ := [803, 900, 1100, 1023, 945, 980, 1249]

-- Define the number of villages.
def num_villages : ℕ := 7

-- Define the total population.
def total_population (pops : List ℕ) : ℕ :=
  pops.foldl (λ acc x => acc + x) 0

-- Define the average population computation.
def average_population (pops : List ℕ) (n : ℕ) : ℕ :=
  total_population pops / n

-- Prove that the average population of the 7 villages is 1000.
theorem average_population_is_1000 :
  average_population populations num_villages = 1000 := by
  -- Proof omitted.
  sorry

end average_population_is_1000_l1004_100462


namespace solve_problem_l1004_100447

noncomputable def f : ℝ → ℝ := sorry

axiom functional_equation (x y : ℝ) :
  f (x^3 + y^3) = (x + y) * ((f x)^2 - (f x) * (f y) + (f (f y))^2)

theorem solve_problem (x : ℝ) : f (1996 * x) = 1996 * f x :=
sorry

end solve_problem_l1004_100447


namespace abs_lt_one_suff_but_not_necc_l1004_100497

theorem abs_lt_one_suff_but_not_necc (x : ℝ) : (|x| < 1 → x^2 + x - 2 < 0) ∧ ¬(x^2 + x - 2 < 0 → |x| < 1) :=
by
  sorry

end abs_lt_one_suff_but_not_necc_l1004_100497


namespace undefined_value_l1004_100438

theorem undefined_value (x : ℝ) : (x^2 - 16 * x + 64 = 0) → (x = 8) := by
  sorry

end undefined_value_l1004_100438


namespace tree_heights_l1004_100466

theorem tree_heights (T S : ℕ) (h1 : T - S = 20) (h2 : T - 10 = 3 * (S - 10)) : T = 40 := 
by
  sorry

end tree_heights_l1004_100466


namespace square_of_number_ending_in_5_l1004_100413

theorem square_of_number_ending_in_5 (a : ℤ) :
  (10 * a + 5) * (10 * a + 5) = 100 * a * (a + 1) + 25 := by
  sorry

end square_of_number_ending_in_5_l1004_100413


namespace length_four_implies_value_twenty_four_l1004_100442

-- Definition of prime factors of an integer
def prime_factors (n : ℕ) : List ℕ := sorry

-- Definition of the length of an integer
def length_of_integer (n : ℕ) : ℕ :=
  List.length (prime_factors n)

-- Statement of the problem
theorem length_four_implies_value_twenty_four (k : ℕ) (h1 : k > 1) (h2 : length_of_integer k = 4) : k = 24 :=
by
  sorry

end length_four_implies_value_twenty_four_l1004_100442


namespace probability_accurate_forecast_l1004_100465

theorem probability_accurate_forecast (p q : ℝ) (h1 : 0 ≤ p ∧ p ≤ 1) (h2 : 0 ≤ q ∧ q ≤ 1) : 
  p * (1 - q) = p * (1 - q) :=
by {
  sorry
}

end probability_accurate_forecast_l1004_100465


namespace classroom_gpa_l1004_100475

theorem classroom_gpa (x : ℝ) (h1 : (1 / 3) * x + (2 / 3) * 18 = 17) : x = 15 := 
by 
    sorry

end classroom_gpa_l1004_100475


namespace find_inverse_sum_l1004_100420

def f (x : ℝ) : ℝ := x * |x|^2

theorem find_inverse_sum :
  (∃ x : ℝ, f x = 8) ∧ (∃ y : ℝ, f y = -64) → 
  (∃ a b : ℝ, f a = 8 ∧ f b = -64 ∧ a + b = 6) :=
sorry

end find_inverse_sum_l1004_100420


namespace problem1_problem2_l1004_100448

noncomputable def f (x : ℝ) : ℝ := x / Real.exp x

-- Problem 1: (0 < m < 1/e) implies g(x) = f(x) - m has two zeros
theorem problem1 (m : ℝ) (h1 : 0 < m) (h2 : m < 1 / Real.exp 1) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = m ∧ f x2 = m :=
sorry

-- Problem 2: (2/e^2 ≤ a < 1/e) implies f^2(x) - af(x) > 0 has only one integer solution
theorem problem2 (a : ℝ) (h1 : 2 / (Real.exp 2) ≤ a) (h2 : a < 1 / Real.exp 1) :
  ∃! x : ℤ, ∀ y : ℤ, (f y)^2 - a * (f y) > 0 → y = x :=
sorry

end problem1_problem2_l1004_100448


namespace frequency_count_third_group_l1004_100439

theorem frequency_count_third_group 
  (x n : ℕ)
  (h1 : n = 420 - x)
  (h2 : x / (n:ℚ) = 0.20) :
  x = 70 :=
by sorry

end frequency_count_third_group_l1004_100439


namespace negation_of_existence_l1004_100437

theorem negation_of_existence :
  ¬ (∃ x : ℝ, x^2 + 3*x + 2 < 0) ↔ ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
sorry

end negation_of_existence_l1004_100437


namespace age_ratio_l1004_100492

variable (p q : ℕ)

-- Conditions
def condition1 := p - 6 = (q - 6) / 2
def condition2 := p + q = 21

-- Theorem stating the desired ratio
theorem age_ratio (h1 : condition1 p q) (h2 : condition2 p q) : p / Nat.gcd p q = 3 ∧ q / Nat.gcd p q = 4 :=
by
  sorry

end age_ratio_l1004_100492


namespace sum_div_minuend_eq_two_l1004_100405

variable (Subtrahend Minuend Difference : ℝ)

theorem sum_div_minuend_eq_two
  (h : Subtrahend + Difference = Minuend) :
  (Subtrahend + Minuend + Difference) / Minuend = 2 :=
by
  sorry

end sum_div_minuend_eq_two_l1004_100405


namespace quadratic_function_value_at_2_l1004_100446

theorem quadratic_function_value_at_2 
  (a b c : ℝ) (h_a : a ≠ 0) 
  (h1 : 7 = a * (-3)^2 + b * (-3) + c)
  (h2 : 7 = a * (5)^2 + b * 5 + c)
  (h3 : -8 = c) :
  a * 2^2 + b * 2 + c = -8 := by 
  sorry

end quadratic_function_value_at_2_l1004_100446


namespace min_value_expression_l1004_100487

theorem min_value_expression (x y : ℝ) : (x^2 + y^2 - 6 * x + 4 * y + 18) ≥ 5 :=
sorry

end min_value_expression_l1004_100487


namespace total_cost_of_antibiotics_l1004_100443

-- Definitions based on the conditions
def cost_A_per_dose : ℝ := 3
def cost_B_per_dose : ℝ := 4.50
def doses_per_day_A : ℕ := 2
def days_A : ℕ := 3
def doses_per_day_B : ℕ := 1
def days_B : ℕ := 4

-- Total cost calculations
def total_cost_A : ℝ := days_A * doses_per_day_A * cost_A_per_dose
def total_cost_B : ℝ := days_B * doses_per_day_B * cost_B_per_dose

-- Final proof statement
theorem total_cost_of_antibiotics : total_cost_A + total_cost_B = 36 :=
by
  -- The proof goes here
  sorry

end total_cost_of_antibiotics_l1004_100443


namespace sally_lost_two_balloons_l1004_100401

-- Condition: Sally originally had 9 orange balloons.
def original_orange_balloons := 9

-- Condition: Sally now has 7 orange balloons.
def current_orange_balloons := 7

-- Problem: Prove that Sally lost 2 orange balloons.
theorem sally_lost_two_balloons : original_orange_balloons - current_orange_balloons = 2 := by
  sorry

end sally_lost_two_balloons_l1004_100401


namespace concentric_circles_circumference_difference_and_area_l1004_100460

theorem concentric_circles_circumference_difference_and_area {r_inner r_outer : ℝ} (h1 : r_inner = 25) (h2 : r_outer = r_inner + 15) :
  2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi ∧ Real.pi * r_outer^2 - Real.pi * r_inner^2 = 975 * Real.pi :=
by
  sorry

end concentric_circles_circumference_difference_and_area_l1004_100460


namespace inequality_pos_real_l1004_100410

theorem inequality_pos_real (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) : 
  (a / (b + 2*c + 3*d) + b / (c + 2*d + 3*a) + c / (d + 2*a + 3*b) + d / (a + 2*b + 3*c)) ≥ (2 / 3) := 
sorry

end inequality_pos_real_l1004_100410


namespace worm_length_difference_is_correct_l1004_100403

-- Define the lengths of the worms
def worm1_length : ℝ := 0.8
def worm2_length : ℝ := 0.1

-- Define the difference in length between the longer worm and the shorter worm
def length_difference (a b : ℝ) : ℝ := a - b

-- State the theorem that the length difference is 0.7 inches
theorem worm_length_difference_is_correct (h1 : worm1_length = 0.8) (h2 : worm2_length = 0.1) :
  length_difference worm1_length worm2_length = 0.7 :=
by
  sorry

end worm_length_difference_is_correct_l1004_100403


namespace find_three_numbers_l1004_100483

theorem find_three_numbers :
  ∃ (a₁ a₄ a₂₅ : ℕ), a₁ + a₄ + a₂₅ = 114 ∧
    ( ∃ r ≠ 1, a₄ = a₁ * r ∧ a₂₅ = a₄ * r * r ) ∧
    ( ∃ d, a₄ = a₁ + 3 * d ∧ a₂₅ = a₁ + 24 * d ) ∧
    a₁ = 2 ∧ a₄ = 14 ∧ a₂₅ = 98 :=
by
  sorry

end find_three_numbers_l1004_100483


namespace smallest_number_of_pencils_l1004_100440

theorem smallest_number_of_pencils 
  (p : ℕ) 
  (h1 : p % 6 = 5)
  (h2 : p % 7 = 3)
  (h3 : p % 8 = 7) :
  p = 35 := 
sorry

end smallest_number_of_pencils_l1004_100440


namespace louisa_second_day_miles_l1004_100481

theorem louisa_second_day_miles (T1 T2 : ℕ) (speed miles_first_day miles_second_day : ℕ)
  (h1 : speed = 25) 
  (h2 : miles_first_day = 100)
  (h3 : T1 = miles_first_day / speed) 
  (h4 : T2 = T1 + 3) 
  (h5 : miles_second_day = speed * T2) :
  miles_second_day = 175 := 
by
  -- We can add the necessary calculations here, but for now, sorry is used to skip the proof.
  sorry

end louisa_second_day_miles_l1004_100481


namespace solve_inequality_l1004_100436

theorem solve_inequality (x : ℝ) : 2 * x ^ 2 - 7 * x - 30 < 0 ↔ - (5 / 2) < x ∧ x < 6 := 
sorry

end solve_inequality_l1004_100436


namespace min_value_inv_sum_l1004_100404

theorem min_value_inv_sum (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y = 12) : 
  ∃ z, (∀ x y : ℝ, 0 < x → 0 < y → x + y = 12 → z ≤ (1/x + 1/y)) ∧ z = 1/3 :=
sorry

end min_value_inv_sum_l1004_100404


namespace age_of_son_l1004_100486

theorem age_of_son (D S : ℕ) (h₁ : S = D / 4) (h₂ : D - S = 27) (h₃ : D = 36) : S = 9 :=
by
  sorry

end age_of_son_l1004_100486


namespace range_of_a_l1004_100409

variable (a : ℝ)

def p : Prop := (0 < a) ∧ (a < 1)
def q : Prop := (a > (1 / 2))

theorem range_of_a (hpq_true: p a ∨ q a) (hpq_false: ¬ (p a ∧ q a)) :
  (0 < a ∧ a ≤ (1 / 2)) ∨ (a ≥ 1) :=
sorry

end range_of_a_l1004_100409


namespace remainder_3211_div_103_l1004_100452

theorem remainder_3211_div_103 :
  3211 % 103 = 18 :=
by
  sorry

end remainder_3211_div_103_l1004_100452


namespace fraction_doubled_unchanged_l1004_100477

theorem fraction_doubled_unchanged (x y : ℝ) (h : x ≠ y) : 
  (2 * x) / (2 * x - 2 * y) = x / (x - y) :=
by
  sorry

end fraction_doubled_unchanged_l1004_100477


namespace july_husband_current_age_l1004_100474

-- Define the initial ages and the relationship between Hannah and July's age
def hannah_initial_age : ℕ := 6
def hannah_july_age_relation (hannah_age july_age : ℕ) : Prop := hannah_age = 2 * july_age

-- Define the time that has passed and the age difference between July and her husband
def time_passed : ℕ := 20
def july_husband_age_relation (july_age husband_age : ℕ) : Prop := husband_age = july_age + 2

-- Lean statement to prove July's husband's current age
theorem july_husband_current_age : ∃ (july_age husband_age : ℕ),
  hannah_july_age_relation hannah_initial_age july_age ∧
  july_husband_age_relation (july_age + time_passed) husband_age ∧
  husband_age = 25 :=
by
  sorry

end july_husband_current_age_l1004_100474


namespace solve_equation_l1004_100434

-- Define the given equation
def equation (x : ℝ) : Prop := (x^3 - 3 * x^2) / (x^2 - 4 * x + 4) + x = -3

-- State the theorem indicating the solutions to the equation
theorem solve_equation (x : ℝ) (h : x ≠ 2) : 
  equation x ↔ x = -2 ∨ x = 3 / 2 :=
sorry

end solve_equation_l1004_100434


namespace susie_vacuums_each_room_in_20_minutes_l1004_100408

theorem susie_vacuums_each_room_in_20_minutes
  (total_time_hours : ℕ)
  (number_of_rooms : ℕ)
  (total_time_minutes : ℕ)
  (time_per_room : ℕ)
  (h1 : total_time_hours = 2)
  (h2 : number_of_rooms = 6)
  (h3 : total_time_minutes = total_time_hours * 60)
  (h4 : time_per_room = total_time_minutes / number_of_rooms) :
  time_per_room = 20 :=
by
  sorry

end susie_vacuums_each_room_in_20_minutes_l1004_100408


namespace seating_arrangement_l1004_100494

theorem seating_arrangement (x y z : ℕ) (h1 : z = x + y) (h2 : x*10 + y*9 = 67) : x = 4 :=
by
  sorry

end seating_arrangement_l1004_100494


namespace part1_part2_l1004_100458

-- Definitions from conditions
def U := ℝ
def A := {x : ℝ | -x^2 + 12*x - 20 > 0}
def B (a : ℝ) := {x : ℝ | 5 - a < x ∧ x < a}

-- (1) If "x ∈ A" is a necessary condition for "x ∈ B", find the range of a
theorem part1 (a : ℝ) : (∀ x : ℝ, x ∈ B a → x ∈ A) → a ≤ 3 :=
by sorry

-- (2) If A ∩ B ≠ ∅, find the range of a
theorem part2 (a : ℝ) : (∃ x : ℝ, x ∈ A ∧ x ∈ B a) → a > 5 / 2 :=
by sorry

end part1_part2_l1004_100458


namespace tan_pi_over_4_plus_alpha_l1004_100431

theorem tan_pi_over_4_plus_alpha (α : ℝ) 
  (h : Real.tan (Real.pi / 4 + α) = 2) : 
  1 / (2 * Real.sin α * Real.cos α + Real.cos α ^ 2) = 5 / 7 := 
by {
  sorry
}

end tan_pi_over_4_plus_alpha_l1004_100431


namespace clowns_to_guppies_ratio_l1004_100414

theorem clowns_to_guppies_ratio
  (C : ℕ)
  (tetra : ℕ)
  (guppies : ℕ)
  (total_animals : ℕ)
  (h1 : tetra = 4 * C)
  (h2 : guppies = 30)
  (h3 : total_animals = 330)
  (h4 : total_animals = tetra + C + guppies) :
  C / guppies = 2 :=
by
  sorry

end clowns_to_guppies_ratio_l1004_100414


namespace connor_cats_l1004_100499

theorem connor_cats (j : ℕ) (a : ℕ) (m : ℕ) (c : ℕ) (co : ℕ) (x : ℕ) 
  (h1 : a = j / 3)
  (h2 : m = 2 * a)
  (h3 : c = a / 2)
  (h4 : c = co + 5)
  (h5 : j = 90)
  (h6 : x = j + a + m + c + co) : 
  co = 10 := 
by
  sorry

end connor_cats_l1004_100499


namespace max_value_min_4x_y_4y_x2_5y2_l1004_100424

theorem max_value_min_4x_y_4y_x2_5y2 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ t, t = min (4 * x + y) (4 * y / (x^2 + 5 * y^2)) ∧ t ≤ 2 :=
by
  sorry

end max_value_min_4x_y_4y_x2_5y2_l1004_100424


namespace range_of_a_l1004_100432

open Real

noncomputable def f (x : ℝ) := x - sqrt (x^2 + x)

noncomputable def g (x a : ℝ) := log x / log 27 - log x / log 9 + a * log x / log 3

theorem range_of_a (a : ℝ) : (∀ x1 ∈ Set.Ioi 1, ∃ x2 ∈ Set.Icc 3 9, f x1 > g x2 a) → a ≤ -1/12 :=
by
  intro h
  sorry

end range_of_a_l1004_100432


namespace toms_crab_buckets_l1004_100496

def crabs_per_bucket := 12
def price_per_crab := 5
def weekly_earnings := 3360

theorem toms_crab_buckets : (weekly_earnings / (crabs_per_bucket * price_per_crab)) = 56 := by
  sorry

end toms_crab_buckets_l1004_100496


namespace parallel_line_slope_l1004_100472

theorem parallel_line_slope (x y : ℝ) (h : 3 * x - 6 * y = 12) : 
  ∃ m : ℝ, m = 1 / 2 ∧ (∀ x1 y1 : ℝ, 3 * x1 - 6 * y1 = 12 → 
    ∃ k : ℝ, y1 = m * x1 + k) :=
by
  sorry

end parallel_line_slope_l1004_100472


namespace complex_identity_l1004_100454

open Complex

noncomputable def z := 1 + 2 * I
noncomputable def z_inv := (1 - 2 * I) / 5
noncomputable def z_conj := 1 - 2 * I

theorem complex_identity : 
  (z + z_inv) * z_conj = (22 / 5 : ℂ) - (4 / 5) * I := 
by
  sorry

end complex_identity_l1004_100454


namespace tetrahedron_altitudes_l1004_100444

theorem tetrahedron_altitudes (r h₁ h₂ h₃ h₄ : ℝ)
  (h₁_def : h₁ = 3 * r)
  (h₂_def : h₂ = 4 * r)
  (h₃_def : h₃ = 4 * r)
  (altitude_sum : 1/h₁ + 1/h₂ + 1/h₃ + 1/h₄ = 1/r) : 
  h₄ = 6 * r :=
by
  rw [h₁_def, h₂_def, h₃_def] at altitude_sum
  sorry

end tetrahedron_altitudes_l1004_100444


namespace fireworks_display_l1004_100402

-- Define numbers and conditions
def display_fireworks_for_number (n : ℕ) : ℕ := 6
def display_fireworks_for_letter (c : Char) : ℕ := 5
def fireworks_per_box : ℕ := 8
def number_boxes : ℕ := 50

-- Calculate fireworks for the year 2023
def fireworks_for_year : ℕ :=
  display_fireworks_for_number 2 * 2 +
  display_fireworks_for_number 0 * 1 +
  display_fireworks_for_number 3 * 1

-- Calculate fireworks for "HAPPY NEW YEAR"
def fireworks_for_phrase : ℕ :=
  12 * display_fireworks_for_letter 'H'

-- Calculate fireworks for 50 boxes
def fireworks_for_boxes : ℕ := number_boxes * fireworks_per_box

-- Total fireworks calculation
def total_fireworks : ℕ := fireworks_for_year + fireworks_for_phrase + fireworks_for_boxes

-- Proof statement
theorem fireworks_display : total_fireworks = 476 := 
  by
  -- This is where the proof would go.
  sorry

end fireworks_display_l1004_100402


namespace inequality_solution_l1004_100478

theorem inequality_solution (x : ℝ) : 
  (0 < (x + 2) / ((x - 3)^3)) ↔ (x < -2 ∨ x > 3)  :=
by
  sorry

end inequality_solution_l1004_100478


namespace max_regions_divided_l1004_100433

theorem max_regions_divided (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ r : ℕ, r = 50 :=
by
  have non_parallel_lines := n - m
  have regions_non_parallel := (non_parallel_lines * (non_parallel_lines + 1)) / 2 + 1
  have regions_parallel := m * non_parallel_lines + m
  have total_regions := regions_non_parallel + regions_parallel
  use total_regions
  sorry

end max_regions_divided_l1004_100433


namespace fraction_C_D_l1004_100406

noncomputable def C : ℝ := ∑' n, if n % 6 = 0 then 0 else if n % 2 = 0 then ((-1)^(n/2 + 1) / (↑n^2)) else 0
noncomputable def D : ℝ := ∑' n, if n % 6 = 0 then ((-1)^(n/6 + 1) / (↑n^2)) else 0

theorem fraction_C_D : C / D = 37 := sorry

end fraction_C_D_l1004_100406


namespace convert_246_octal_to_decimal_l1004_100468

theorem convert_246_octal_to_decimal : 2 * (8^2) + 4 * (8^1) + 6 * (8^0) = 166 := 
by
  -- We skip the proof part as it is not required in the task
  sorry

end convert_246_octal_to_decimal_l1004_100468


namespace natural_number_x_l1004_100429

theorem natural_number_x (x : ℕ) (A : ℕ → ℕ) (h : 3 * (A (x + 1))^3 = 2 * (A (x + 2))^2 + 6 * (A (x + 1))^2) : x = 4 :=
sorry

end natural_number_x_l1004_100429


namespace mass_of_man_l1004_100423

theorem mass_of_man (L B h ρ V m: ℝ) (boat_length: L = 3) (boat_breadth: B = 2) 
  (boat_sink_depth: h = 0.01) (water_density: ρ = 1000) 
  (displaced_volume: V = L * B * h) (displaced_mass: m = ρ * V): m = 60 := 
by 
  sorry

end mass_of_man_l1004_100423


namespace total_handshakes_l1004_100476

-- Define the conditions
def number_of_players_per_team : Nat := 11
def number_of_referees : Nat := 3
def total_number_of_players : Nat := number_of_players_per_team * 2

-- Prove the total number of handshakes
theorem total_handshakes : 
  (number_of_players_per_team * number_of_players_per_team) + (total_number_of_players * number_of_referees) = 187 := 
by {
  sorry
}

end total_handshakes_l1004_100476


namespace sue_nuts_count_l1004_100489

theorem sue_nuts_count (B H S : ℕ) 
  (h1 : B = 6 * H) 
  (h2 : H = 2 * S) 
  (h3 : B + H = 672) : S = 48 := 
by
  sorry

end sue_nuts_count_l1004_100489


namespace integer_solution_for_system_l1004_100400

theorem integer_solution_for_system 
    (x y z : ℕ) 
    (h1 : 3 * x - 4 * y + 5 * z = 10) 
    (h2 : 7 * y + 8 * x - 3 * z = 13) : 
    x = 1 ∧ y = 2 ∧ z = 3 :=
by 
  sorry

end integer_solution_for_system_l1004_100400


namespace cucumbers_after_purchase_l1004_100425

theorem cucumbers_after_purchase (C U : ℕ) (h1 : C + U = 10) (h2 : C = 4) : U + 2 = 8 := by
  sorry

end cucumbers_after_purchase_l1004_100425


namespace jellybeans_left_in_jar_l1004_100445

def original_jellybeans : ℕ := 250
def class_size : ℕ := 24
def sick_children : ℕ := 2
def sick_jellybeans_each : ℕ := 7
def first_group_size : ℕ := 12
def first_group_jellybeans_each : ℕ := 5
def second_group_size : ℕ := 10
def second_group_jellybeans_each : ℕ := 4

theorem jellybeans_left_in_jar : 
  original_jellybeans - ((first_group_size * first_group_jellybeans_each) + 
  (second_group_size * second_group_jellybeans_each)) = 150 := by
  sorry

end jellybeans_left_in_jar_l1004_100445


namespace MishaTotalMoney_l1004_100490

-- Define Misha's initial amount of money
def initialMoney : ℕ := 34

-- Define the amount of money Misha earns
def earnedMoney : ℕ := 13

-- Define the total amount of money Misha will have
def totalMoney : ℕ := initialMoney + earnedMoney

-- Statement to prove
theorem MishaTotalMoney : totalMoney = 47 := by
  sorry

end MishaTotalMoney_l1004_100490


namespace intersection_sets_l1004_100428

-- defining sets A and B
def A : Set ℤ := {-1, 2, 4}
def B : Set ℤ := {0, 2, 6}

-- the theorem to be proved
theorem intersection_sets:
  A ∩ B = {2} :=
sorry

end intersection_sets_l1004_100428


namespace connie_blue_markers_l1004_100464

theorem connie_blue_markers :
  ∀ (total_markers red_markers blue_markers : ℕ),
    total_markers = 105 →
    red_markers = 41 →
    blue_markers = total_markers - red_markers →
    blue_markers = 64 :=
by
  intros total_markers red_markers blue_markers htotal hred hblue
  rw [htotal, hred] at hblue
  exact hblue

end connie_blue_markers_l1004_100464


namespace extra_birds_l1004_100415

def num_sparrows : ℕ := 10
def num_robins : ℕ := 5
def num_bluebirds : ℕ := 3
def nests_for_sparrows : ℕ := 4
def nests_for_robins : ℕ := 2
def nests_for_bluebirds : ℕ := 2

theorem extra_birds (num_sparrows : ℕ)
                    (num_robins : ℕ)
                    (num_bluebirds : ℕ)
                    (nests_for_sparrows : ℕ)
                    (nests_for_robins : ℕ)
                    (nests_for_bluebirds : ℕ) :
    num_sparrows = 10 ∧ 
    num_robins = 5 ∧ 
    num_bluebirds = 3 ∧ 
    nests_for_sparrows = 4 ∧ 
    nests_for_robins = 2 ∧ 
    nests_for_bluebirds = 2 ->
    num_sparrows - nests_for_sparrows = 6 ∧ 
    num_robins - nests_for_robins = 3 ∧ 
    num_bluebirds - nests_for_bluebirds = 1 :=
by sorry

end extra_birds_l1004_100415


namespace ratio_of_products_l1004_100457

theorem ratio_of_products (a b c d : ℝ) 
  (h : (a - b) * (c - d) / ((b - c) * (d - a)) = 3 / 7) :
  ((a - c) * (b - d)) / ((a - b) * (c - d)) = -4 / 3 :=
by 
  sorry

end ratio_of_products_l1004_100457


namespace circle_center_radius_sum_18_l1004_100418

-- Conditions from the problem statement
def circle_eq (x y : ℝ) : Prop := x^2 + 2 * y - 9 = -y^2 + 18 * x + 9

-- Goal is to prove a + b + r = 18
theorem circle_center_radius_sum_18 :
  (∃ a b r : ℝ, 
     (∀ x y : ℝ, circle_eq x y ↔ (x - a)^2 + (y - b)^2 = r^2) ∧ 
     a + b + r = 18) :=
sorry

end circle_center_radius_sum_18_l1004_100418


namespace matrix_zero_product_or_rank_one_l1004_100430

variables {n : ℕ}
variables (A B C : matrix (fin n) (fin n) ℝ)

theorem matrix_zero_product_or_rank_one
  (h1 : A * B * C = 0)
  (h2 : B.rank = 1) :
  A * B = 0 ∨ B * C = 0 :=
sorry

end matrix_zero_product_or_rank_one_l1004_100430


namespace wendy_albums_used_l1004_100422

def total_pictures : ℕ := 45
def pictures_in_one_album : ℕ := 27
def pictures_per_album : ℕ := 2

theorem wendy_albums_used :
  let remaining_pictures := total_pictures - pictures_in_one_album
  let albums_used := remaining_pictures / pictures_per_album
  albums_used = 9 :=
by
  sorry

end wendy_albums_used_l1004_100422


namespace intersection_eq_l1004_100451

def M : Set ℝ := {x | x < 3}
def N : Set ℝ := {x | x^2 - 6*x + 8 < 0}
def intersection : Set ℝ := {x | 2 < x ∧ x < 3}

theorem intersection_eq : M ∩ N = intersection := by
  sorry

end intersection_eq_l1004_100451


namespace large_font_pages_l1004_100435

theorem large_font_pages (L S : ℕ) (h1 : L + S = 21) (h2 : 3 * L = 2 * S) : L = 8 :=
by {
  sorry -- Proof can be filled in Lean; this ensures the statement aligns with problem conditions.
}

end large_font_pages_l1004_100435


namespace inequality_has_no_solutions_l1004_100473

theorem inequality_has_no_solutions (x : ℝ) : ¬ (3 * x^2 + 9 * x + 12 ≤ 0) :=
by {
  sorry
}

end inequality_has_no_solutions_l1004_100473


namespace integers_in_range_of_f_l1004_100416

noncomputable def f (x : ℝ) := x^2 + x + 1/2

def count_integers_in_range (n : ℕ) : ℕ :=
  2 * (n + 1)

theorem integers_in_range_of_f (n : ℕ) :
  (count_integers_in_range n) = (2 * (n + 1)) :=
by
  sorry

end integers_in_range_of_f_l1004_100416


namespace probability_of_selecting_green_ball_l1004_100470

def container_I :  ℕ × ℕ := (5, 5) -- (red balls, green balls)
def container_II : ℕ × ℕ := (3, 3) -- (red balls, green balls)
def container_III : ℕ × ℕ := (4, 2) -- (red balls, green balls)
def container_IV : ℕ × ℕ := (6, 6) -- (red balls, green balls)

def total_containers : ℕ := 4

def probability_of_green_ball (red_green : ℕ × ℕ) : ℚ :=
  let (red, green) := red_green
  green / (red + green)

noncomputable def combined_probability_of_green_ball : ℚ :=
  (1 / total_containers) *
  (probability_of_green_ball container_I +
   probability_of_green_ball container_II +
   probability_of_green_ball container_III +
   probability_of_green_ball container_IV)

theorem probability_of_selecting_green_ball : 
  combined_probability_of_green_ball = 11 / 24 :=
sorry

end probability_of_selecting_green_ball_l1004_100470


namespace parallel_lines_l1004_100484

theorem parallel_lines (a : ℝ) : 
  (∀ x y : ℝ, x + 2 * a * y - 1 = 0 → (3 * a - 1) * x - 4 * a * y - 1 = 0 → False) → 
  (a = 0 ∨ a = -1/3) :=
sorry

end parallel_lines_l1004_100484


namespace trapezoid_bisector_segment_length_l1004_100426

-- Definitions of the conditions
variables {a b c d t : ℝ}

noncomputable def semiperimeter (a b c d : ℝ) : ℝ := (a + b + c + d) / 2

-- The theorem statement
theorem trapezoid_bisector_segment_length
  (p : ℝ)
  (h_p : p = semiperimeter a b c d) :
  t^2 = (4 * b * d) / (b + d)^2 * (p - a) * (p - c) :=
sorry

end trapezoid_bisector_segment_length_l1004_100426


namespace continuous_arrow_loop_encircling_rectangle_l1004_100419

def total_orientations : ℕ := 2^4

def favorable_orientations : ℕ := 2 * 2

def probability_loop : ℚ := favorable_orientations / total_orientations

theorem continuous_arrow_loop_encircling_rectangle : probability_loop = 1 / 4 := by
  sorry

end continuous_arrow_loop_encircling_rectangle_l1004_100419


namespace find_a_value_l1004_100480

theorem find_a_value (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) 
  (h3 : (∃ l : ℝ, ∃ f : ℝ → ℝ, f x = a^x ∧ deriv f 0 = -1)) :
  a = 1 / Real.exp 1 := by
  sorry

end find_a_value_l1004_100480


namespace tree_sidewalk_space_l1004_100495

theorem tree_sidewalk_space
  (num_trees : ℕ)
  (distance_between_trees : ℝ)
  (total_road_length : ℝ)
  (total_gaps : ℝ)
  (space_each_tree : ℝ)
  (H1 : num_trees = 11)
  (H2 : distance_between_trees = 14)
  (H3 : total_road_length = 151)
  (H4 : total_gaps = (num_trees - 1) * distance_between_trees)
  (H5 : space_each_tree = (total_road_length - total_gaps) / num_trees)
  : space_each_tree = 1 := 
by
  sorry

end tree_sidewalk_space_l1004_100495


namespace math_problem_l1004_100421

noncomputable def problem_statement : Prop :=
  let A : ℝ × ℝ := (5, 6)
  let B : ℝ × ℝ := (8, 3)
  let slope : ℝ := (B.snd - A.snd) / (B.fst - A.fst)
  let y_intercept : ℝ := A.snd - slope * A.fst
  slope + y_intercept = 10

theorem math_problem : problem_statement := sorry

end math_problem_l1004_100421


namespace slope_of_parallel_line_l1004_100479

theorem slope_of_parallel_line (x y : ℝ) :
  (∃ (b : ℝ), 3 * x - 6 * y = 12) → ∀ (m₁ x₁ y₁ x₂ y₂ : ℝ), (y₁ = (1/2) * x₁ + b) ∧ (y₂ = (1/2) * x₂ + b) → (x₁ ≠ x₂) → m₁ = 1/2 :=
by 
  sorry

end slope_of_parallel_line_l1004_100479


namespace difference_of_cats_l1004_100417

-- Definitions based on given conditions
def number_of_cats_sheridan : ℕ := 11
def number_of_cats_garrett : ℕ := 24

-- Theorem statement (proof problem) based on the question and correct answer
theorem difference_of_cats : (number_of_cats_garrett - number_of_cats_sheridan) = 13 := by
  sorry

end difference_of_cats_l1004_100417


namespace communication_system_connections_l1004_100456

theorem communication_system_connections (n : ℕ) (h : ∀ k < 2001, ∃ l < 2001, l ≠ k ∧ k ≠ l) :
  (∀ k < 2001, ∃ l < 2001, k ≠ l) → (n % 2 = 0 ∧ n ≤ 2000) ∨ n = 0 :=
sorry

end communication_system_connections_l1004_100456
