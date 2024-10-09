import Mathlib

namespace total_number_of_possible_outcomes_l823_82326

-- Define the conditions
def num_faces_per_die : ℕ := 6
def num_dice : ℕ := 2

-- Define the question as a hypothesis and the answer as the conclusion
theorem total_number_of_possible_outcomes :
  (num_faces_per_die * num_faces_per_die) = 36 := 
by
  -- Provide a proof outline, this is used to skip the actual proof
  sorry

end total_number_of_possible_outcomes_l823_82326


namespace events_A_B_mutually_exclusive_events_A_C_independent_l823_82329

-- Definitions for events A, B, and C
def event_A (x y : ℕ) : Prop := x + y = 7
def event_B (x y : ℕ) : Prop := (x * y) % 2 = 1
def event_C (x : ℕ) : Prop := x > 3

-- Proof problems to decide mutual exclusivity and independence
theorem events_A_B_mutually_exclusive :
  ∀ (x y : ℕ), event_A x y → ¬ event_B x y := 
by sorry

theorem events_A_C_independent :
  ∀ (x y : ℕ), (event_A x y) ↔ ∀ x y, event_C x ↔ event_A x y ∧ event_C x := 
by sorry

end events_A_B_mutually_exclusive_events_A_C_independent_l823_82329


namespace initial_water_amount_l823_82316

variable (W : ℝ)
variable (evap_per_day : ℝ := 0.014)
variable (days : ℕ := 50)
variable (evap_percent : ℝ := 7.000000000000001)

theorem initial_water_amount :
  evap_per_day * (days : ℝ) = evap_percent / 100 * W → W = 10 :=
by
  sorry

end initial_water_amount_l823_82316


namespace connected_graphs_bound_l823_82353

noncomputable def num_connected_graphs (n : ℕ) : ℕ := sorry
  
theorem connected_graphs_bound (n : ℕ) : 
  num_connected_graphs n ≥ (1/2) * 2^(n*(n-1)/2) := 
sorry

end connected_graphs_bound_l823_82353


namespace tim_weekly_earnings_l823_82379

-- Define the constants based on the conditions
def tasks_per_day : ℕ := 100
def pay_per_task : ℝ := 1.2
def days_per_week : ℕ := 6

-- Statement to prove Tim's weekly earnings
theorem tim_weekly_earnings : tasks_per_day * pay_per_task * days_per_week = 720 := by
  sorry

end tim_weekly_earnings_l823_82379


namespace sequence_positive_and_divisible_l823_82359

theorem sequence_positive_and_divisible:
  ∃ (a : ℕ → ℕ), 
    (a 1 = 2) ∧ (a 2 = 500) ∧ (a 3 = 2000) ∧ 
    (∀ n ≥ 2, (a (n + 2) + a (n + 1)) * a (n - 1) = a (n + 1) * (a (n + 1) + a (n - 1))) ∧ 
    (∀ n, a n > 0) ∧ 
    (2 ^ 2000 ∣ a 2000) := 
sorry

end sequence_positive_and_divisible_l823_82359


namespace problem_1_system_solution_problem_2_system_solution_l823_82320

theorem problem_1_system_solution (x y : ℝ)
  (h1 : x - 2 * y = 1)
  (h2 : 4 * x + 3 * y = 26) :
  x = 5 ∧ y = 2 :=
sorry

theorem problem_2_system_solution (x y : ℝ)
  (h1 : 2 * x + 3 * y = 3)
  (h2 : 5 * x - 3 * y = 18) :
  x = 3 ∧ y = -1 :=
sorry

end problem_1_system_solution_problem_2_system_solution_l823_82320


namespace ahmed_goats_is_13_l823_82339

def adam_goats : ℕ := 7

def andrew_goats : ℕ := 2 * adam_goats + 5

def ahmed_goats : ℕ := andrew_goats - 6

theorem ahmed_goats_is_13 : ahmed_goats = 13 :=
by
  sorry

end ahmed_goats_is_13_l823_82339


namespace probability_of_drawing_three_white_marbles_l823_82367

noncomputable def probability_of_three_white_marbles : ℚ :=
  let total_marbles := 5 + 7 + 15
  let prob_first_white := 15 / total_marbles
  let prob_second_white := 14 / (total_marbles - 1)
  let prob_third_white := 13 / (total_marbles - 2)
  prob_first_white * prob_second_white * prob_third_white

theorem probability_of_drawing_three_white_marbles :
  probability_of_three_white_marbles = 2 / 13 := 
by 
  sorry

end probability_of_drawing_three_white_marbles_l823_82367


namespace delta_comparison_eps_based_on_gamma_l823_82355

-- Definitions for the problem
variable {α β γ δ ε : ℝ}
variable {A B C : Type}
variable (s f m : Type)

-- Conditions from problem
variable (triangle_ABC : α ≠ β)
variable (median_s_from_C : s)
variable (angle_bisector_f : f)
variable (altitude_m : m)
variable (angle_between_f_m : δ = sorry)
variable (angle_between_f_s : ε = sorry)
variable (angle_at_vertex_C : γ = sorry)

-- Main statement to prove
theorem delta_comparison_eps_based_on_gamma (h1 : α ≠ β) (h2 : δ = sorry) (h3 : ε = sorry) (h4 : γ = sorry) :
  if γ < 90 then δ < ε else if γ = 90 then δ = ε else δ > ε :=
sorry

end delta_comparison_eps_based_on_gamma_l823_82355


namespace identity_holds_l823_82348

noncomputable def identity_proof : Prop :=
∀ (x : ℝ), (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x

theorem identity_holds : identity_proof :=
by
  sorry

end identity_holds_l823_82348


namespace twelve_people_pairing_l823_82371

noncomputable def num_ways_to_pair : ℕ := sorry

theorem twelve_people_pairing :
  (∀ (n : ℕ), n = 12 → (∃ f : ℕ → ℕ, ∀ i, f i = 2 ∨ f i = 12 ∨ f i = 7) → num_ways_to_pair = 3) := 
sorry

end twelve_people_pairing_l823_82371


namespace tetrahedron_volume_l823_82375

theorem tetrahedron_volume (R S1 S2 S3 S4 : ℝ) : 
    V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end tetrahedron_volume_l823_82375


namespace bicycle_weight_l823_82325

theorem bicycle_weight (b s : ℕ) (h1 : 10 * b = 5 * s) (h2 : 5 * s = 200) : b = 20 := 
by 
  sorry

end bicycle_weight_l823_82325


namespace james_total_points_l823_82349

def points_per_correct_answer : ℕ := 2
def bonus_points_per_round : ℕ := 4
def total_rounds : ℕ := 5
def questions_per_round : ℕ := 5
def total_questions : ℕ := total_rounds * questions_per_round
def questions_missed_by_james : ℕ := 1
def questions_answered_by_james : ℕ := total_questions - questions_missed_by_james
def points_for_correct_answers : ℕ := questions_answered_by_james * points_per_correct_answer
def complete_rounds_by_james : ℕ := total_rounds - 1  -- Since James missed one question, he has 4 complete rounds
def bonus_points_by_james : ℕ := complete_rounds_by_james * bonus_points_per_round
def total_points : ℕ := points_for_correct_answers + bonus_points_by_james

theorem james_total_points : total_points = 64 := by
  sorry

end james_total_points_l823_82349


namespace prime_in_A_l823_82383

open Nat

def is_in_A (x : ℕ) : Prop :=
  ∃ (a b : ℤ), x = a^2 + 2*b^2 ∧ a * b ≠ 0

theorem prime_in_A (p : ℕ) [Fact (Nat.Prime p)] (h : is_in_A (p^2)) : is_in_A p :=
  sorry

end prime_in_A_l823_82383


namespace caps_percentage_l823_82382

open Real

-- Define the conditions as given in part (a)
def total_caps : ℝ := 575
def red_caps : ℝ := 150
def green_caps : ℝ := 120
def blue_caps : ℝ := 175
def yellow_caps : ℝ := total_caps - (red_caps + green_caps + blue_caps)

-- Define the problem asking for the percentages of each color and proving the answer
theorem caps_percentage :
  (red_caps / total_caps) * 100 = 26.09 ∧
  (green_caps / total_caps) * 100 = 20.87 ∧
  (blue_caps / total_caps) * 100 = 30.43 ∧
  (yellow_caps / total_caps) * 100 = 22.61 :=
by
  -- proof steps would go here
  sorry

end caps_percentage_l823_82382


namespace bird_migration_difference_correct_l823_82322

def bird_migration_difference : ℕ := 54

/--
There are 250 bird families consisting of 3 different bird species, each with varying migration patterns.

Species A: 100 bird families; 35% fly to Africa, 65% fly to Asia
Species B: 120 bird families; 50% fly to Africa, 50% fly to Asia
Species C: 30 bird families; 10% fly to Africa, 90% fly to Asia

Prove that the difference in the number of bird families migrating to Asia and Africa is 54.
-/
theorem bird_migration_difference_correct (A_Africa_percent : ℕ := 35) (A_Asia_percent : ℕ := 65)
  (B_Africa_percent : ℕ := 50) (B_Asia_percent : ℕ := 50)
  (C_Africa_percent : ℕ := 10) (C_Asia_percent : ℕ := 90)
  (A_count : ℕ := 100) (B_count : ℕ := 120) (C_count : ℕ := 30) :
    bird_migration_difference = 
      (A_count * A_Asia_percent / 100 + B_count * B_Asia_percent / 100 + C_count * C_Asia_percent / 100) - 
      (A_count * A_Africa_percent / 100 + B_count * B_Africa_percent / 100 + C_count * C_Africa_percent / 100) :=
by sorry

end bird_migration_difference_correct_l823_82322


namespace problem1_problem2_l823_82342

-- Definition for the first proof problem
theorem problem1 (a b : ℝ) (h : a ≠ b) :
  (a^2 / (a - b) - b^2 / (a - b)) = a + b :=
by
  sorry

-- Definition for the second proof problem
theorem problem2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  ((x^2 - 1) / ((x^2 + 2 * x + 1)) / (x^2 - x) / (x + 1)) = 1 / x :=
by
  sorry

end problem1_problem2_l823_82342


namespace common_difference_d_l823_82312

open Real

-- Define the arithmetic sequence and relevant conditions
variable (a : ℕ → ℝ) -- Define the sequence as a function from natural numbers to real numbers
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific conditions from our problem
def problem_conditions (a : ℕ → ℝ) (d : ℝ) : Prop :=
  is_arithmetic_sequence a d ∧
  a 1 = 1 ∧
  (a 2) ^ 2 = a 1 * a 6

-- The goal is to prove that the common difference d is either 0 or 3
theorem common_difference_d (a : ℕ → ℝ) (d : ℝ) :
  problem_conditions a d → (d = 0 ∨ d = 3) := by
  sorry

end common_difference_d_l823_82312


namespace no_nonzero_integer_solution_l823_82323

theorem no_nonzero_integer_solution (x y z : ℤ) (h : x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) :
  x^2 + y^2 ≠ 3 * z^2 :=
by
  sorry

end no_nonzero_integer_solution_l823_82323


namespace polynomial_equality_l823_82318

def P (x : ℝ) : ℝ := x ^ 3 - 3 * x ^ 2 - 3 * x - 1

noncomputable def x1 : ℝ := 1 - Real.sqrt 2
noncomputable def x2 : ℝ := 1 + Real.sqrt 2
noncomputable def x3 : ℝ := 1 - 2 * Real.sqrt 2
noncomputable def x4 : ℝ := 1 + 2 * Real.sqrt 2

theorem polynomial_equality :
  P x1 + P x2 = P x3 + P x4 :=
sorry

end polynomial_equality_l823_82318


namespace find_a_c_l823_82366

theorem find_a_c (a c : ℝ) (h_discriminant : ∀ x : ℝ, a * x^2 + 10 * x + c = 0 → ∃ k : ℝ, a * k^2 + 10 * k + c = 0 ∧ (a * x^2 + 10 * k + c = 0 → x = k))
  (h_sum : a + c = 12) (h_lt : a < c) : (a, c) = (6 - Real.sqrt 11, 6 + Real.sqrt 11) :=
sorry

end find_a_c_l823_82366


namespace students_with_screws_neq_bolts_l823_82386

-- Let's define the main entities
def total_students : ℕ := 40
def nails_neq_bolts : ℕ := 15
def screws_eq_nails : ℕ := 10

-- Main theorem statement
theorem students_with_screws_neq_bolts (total : ℕ) (neq_nails_bolts : ℕ) (eq_screws_nails : ℕ) :
  total = 40 → neq_nails_bolts = 15 → eq_screws_nails = 10 → ∃ k, k ≥ 15 ∧ k ≤ 40 - eq_screws_nails - neq_nails_bolts := 
by
  intros
  sorry

end students_with_screws_neq_bolts_l823_82386


namespace find_m_of_odd_function_l823_82364

theorem find_m_of_odd_function (m : ℝ) (f : ℝ → ℝ) (h₁ : ∀ x, f x = ((x + 3) * (x + m)) / x)
  (h₂ : ∀ x, f (-x) = -f x) : m = -3 :=
sorry

end find_m_of_odd_function_l823_82364


namespace polygon_num_sides_l823_82361

theorem polygon_num_sides (s : ℕ) (h : 180 * (s - 2) > 2790) : s = 18 :=
sorry

end polygon_num_sides_l823_82361


namespace negation_of_proposition_l823_82328

theorem negation_of_proposition :
  (∀ x y : ℝ, (x * y = 0 → x = 0 ∨ y = 0)) →
  (∃ x y : ℝ, x * y = 0 ∧ x ≠ 0 ∧ y ≠ 0) :=
sorry

end negation_of_proposition_l823_82328


namespace gerbil_weights_l823_82317

theorem gerbil_weights
  (puffy muffy scruffy fluffy tuffy : ℕ)
  (h1 : puffy = 2 * muffy)
  (h2 : muffy = scruffy - 3)
  (h3 : scruffy = 12)
  (h4 : fluffy = muffy + tuffy)
  (h5 : fluffy = puffy / 2)
  (h6 : tuffy = puffy / 2) :
  puffy + muffy + tuffy = 36 := by
  sorry

end gerbil_weights_l823_82317


namespace range_of_f_l823_82381

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) + 2

theorem range_of_f (h : ∀ x : ℝ, x ≤ 1) : (f '' {x : ℝ | x ≤ 1}) = {y : ℝ | 1 ≤ y ∧ y ≤ 2} :=
by
  sorry

end range_of_f_l823_82381


namespace factor_polynomial_l823_82311

theorem factor_polynomial {x : ℝ} : 4 * x^3 - 16 * x = 4 * x * (x + 2) * (x - 2) := 
sorry

end factor_polynomial_l823_82311


namespace parallel_vectors_eq_l823_82334

theorem parallel_vectors_eq (t : ℝ) : ∀ (m n : ℝ × ℝ), m = (2, 8) → n = (-4, t) → (∃ k : ℝ, n = k • m) → t = -16 :=
by 
  intros m n hm hn h_parallel
  -- proof goes here
  sorry

end parallel_vectors_eq_l823_82334


namespace book_profit_percentage_l823_82378

noncomputable def profit_percentage (cost_price marked_price : ℝ) (discount_rate : ℝ) : ℝ :=
  let discount := discount_rate / 100 * marked_price
  let selling_price := marked_price - discount
  let profit := selling_price - cost_price
  (profit / cost_price) * 100

theorem book_profit_percentage :
  profit_percentage 47.50 69.85 15 = 24.994736842105263 :=
by
  sorry

end book_profit_percentage_l823_82378


namespace no_nontrivial_solutions_in_integers_l823_82365

theorem no_nontrivial_solutions_in_integers (a b c n : ℤ) 
  (h : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
  by
    sorry

end no_nontrivial_solutions_in_integers_l823_82365


namespace probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l823_82319

namespace ProbabilityKeys

-- Define the problem conditions and the probability computations
def keys : ℕ := 4
def successful_keys : ℕ := 2
def unsuccessful_keys : ℕ := 2

def probability_first_fail (k : ℕ) (s : ℕ) : ℚ := (s : ℚ) / (k : ℚ)
def probability_second_success_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (s + 1 - 1: ℚ) 
def probability_second_success_not_discarded (s : ℕ) (k : ℕ) : ℚ := (s : ℚ) / (k : ℚ)

-- The statements to be proved
theorem probability_door_opened_second_attempt_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_discarded unsuccessful_keys keys) = (1 : ℚ) / (3 : ℚ) :=
by sorry

theorem probability_door_opened_second_attempt_not_discarded : 
  (probability_first_fail keys successful_keys) * (probability_second_success_not_discarded successful_keys keys) = (1 : ℚ) / (4 : ℚ) :=
by sorry

end ProbabilityKeys

end probability_door_opened_second_attempt_discarded_probability_door_opened_second_attempt_not_discarded_l823_82319


namespace inradius_of_triangle_l823_82387

theorem inradius_of_triangle (A p s r : ℝ) (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 :=
by
  sorry

end inradius_of_triangle_l823_82387


namespace trip_cost_l823_82303

theorem trip_cost (original_price : ℕ) (discount : ℕ) (num_people : ℕ)
  (h1 : original_price = 147) (h2 : discount = 14) (h3 : num_people = 2) :
  num_people * (original_price - discount) = 266 :=
by
  sorry

end trip_cost_l823_82303


namespace baker_earnings_l823_82338

-- Define the number of cakes and pies sold
def cakes_sold := 453
def pies_sold := 126

-- Define the prices per cake and pie
def price_per_cake := 12
def price_per_pie := 7

-- Calculate the total earnings
def total_earnings : ℕ := (cakes_sold * price_per_cake) + (pies_sold * price_per_pie)

-- Theorem stating the baker's earnings
theorem baker_earnings : total_earnings = 6318 := by
  unfold total_earnings cakes_sold pies_sold price_per_cake price_per_pie
  sorry

end baker_earnings_l823_82338


namespace greatest_whole_number_satisfies_inequality_l823_82380

theorem greatest_whole_number_satisfies_inequality : 
  ∃ (x : ℕ), (∀ (y : ℕ), (6 * y - 4 < 5 - 3 * y) → y ≤ x) ∧ x = 0 := 
sorry

end greatest_whole_number_satisfies_inequality_l823_82380


namespace cost_price_to_selling_price_ratio_l823_82333

variable (CP SP : ℝ)
variable (profit_percent : ℝ)

theorem cost_price_to_selling_price_ratio
  (h1 : profit_percent = 0.25)
  (h2 : SP = (1 + profit_percent) * CP) :
  (CP / SP) = 4 / 5 := by
  sorry

end cost_price_to_selling_price_ratio_l823_82333


namespace find_BC_l823_82395

variable (A B C : Type)
variables (a b : ℝ) -- Angles
variables (AB BC CA : ℝ) -- Sides of the triangle

-- Given conditions:
-- 1: Triangle ABC
-- 2: cos(a - b) + sin(a + b) = 2
-- 3: AB = 4

theorem find_BC (hAB : AB = 4) (hTrig : Real.cos (a - b) + Real.sin (a + b) = 2) :
  BC = 2 * Real.sqrt 2 := 
sorry

end find_BC_l823_82395


namespace john_drive_time_l823_82306

theorem john_drive_time
  (t : ℝ)
  (h1 : 60 * t + 90 * (15 / 4 - t) = 300)
  (h2 : 1 / 4 = 15 / 60)
  (h3 : 4 = 15 / 4 + t + 1 / 4)
  :
  t = 1.25 :=
by
  -- This introduces the hypothesis and begins the Lean proof.
  sorry

end john_drive_time_l823_82306


namespace isosceles_triangle_vertex_angle_l823_82399

theorem isosceles_triangle_vertex_angle (exterior_angle : ℝ) (h1 : exterior_angle = 40) : 
  ∃ vertex_angle : ℝ, vertex_angle = 140 :=
by
  sorry

end isosceles_triangle_vertex_angle_l823_82399


namespace amount_saved_l823_82356

-- Initial conditions as definitions
def initial_amount : ℕ := 6000
def cost_ballpoint_pen : ℕ := 3200
def cost_eraser : ℕ := 1000
def cost_candy : ℕ := 500

-- Mathematical equivalent proof problem as a Lean theorem statement
theorem amount_saved : initial_amount - (cost_ballpoint_pen + cost_eraser + cost_candy) = 1300 := 
by 
  -- Proof is omitted
  sorry

end amount_saved_l823_82356


namespace toothpaste_usage_l823_82398

-- Define the variables involved
variables (t : ℕ) -- total toothpaste in grams
variables (d : ℕ) -- grams used by dad per brushing
variables (m : ℕ) -- grams used by mom per brushing
variables (b : ℕ) -- grams used by Anne + brother per brushing
variables (r : ℕ) -- brushing rate per day
variables (days : ℕ) -- days for toothpaste to run out
variables (N : ℕ) -- family members

-- Given conditions
variables (ht : t = 105)         -- Total toothpaste is 105 grams
variables (hd : d = 3)           -- Dad uses 3 grams per brushing
variables (hm : m = 2)           -- Mom uses 2 grams per brushing
variables (hr : r = 3)           -- Each member brushes three times a day
variables (hdays : days = 5)     -- Toothpaste runs out in 5 days

-- Additional calculations
variable (total_brushing : ℕ)
variable (total_usage_d: ℕ)
variable (total_usage_m: ℕ)
variable (total_usage_parents: ℕ)
variable (total_usage_family: ℕ)

-- Helper expressions
def total_brushing_expr := days * r * 2
def total_usage_d_expr := d * r
def total_usage_m_expr := m * r
def total_usage_parents_expr := (total_usage_d_expr + total_usage_m_expr) * days
def total_usage_family_expr := t - total_usage_parents_expr

-- Assume calculations
variables (h1: total_usage_d = total_usage_d_expr)  
variables (h2: total_usage_m = total_usage_m_expr)
variables (h3: total_usage_parents = total_usage_parents_expr)
variables (h4: total_usage_family = total_usage_family_expr)
variables (h5 : total_brushing = total_brushing_expr)

-- Define the proof
theorem toothpaste_usage : 
  b = total_usage_family / total_brushing := 
  sorry

end toothpaste_usage_l823_82398


namespace correct_statement_C_l823_82341

-- Define the function
def linear_function (x : ℝ) : ℝ := -3 * x + 1

-- Define the condition for statement C
def statement_C (x : ℝ) : Prop := x > 1 / 3 → linear_function x < 0

-- The theorem to be proved
theorem correct_statement_C : ∀ x : ℝ, statement_C x := by
  sorry

end correct_statement_C_l823_82341


namespace real_part_fraction_l823_82302

theorem real_part_fraction {i : ℂ} (h : i^2 = -1) : (
  let numerator := 1 - i
  let denominator := (1 + i) ^ 2
  let fraction := numerator / denominator
  let real_part := (fraction.re)
  real_part
) = -1/2 := sorry

end real_part_fraction_l823_82302


namespace opposite_event_is_at_least_one_hit_l823_82363

def opposite_event_of_missing_both_times (hit1 hit2 : Prop) : Prop :=
  ¬(¬hit1 ∧ ¬hit2)

theorem opposite_event_is_at_least_one_hit (hit1 hit2 : Prop) :
  opposite_event_of_missing_both_times hit1 hit2 = (hit1 ∨ hit2) :=
by
  sorry

end opposite_event_is_at_least_one_hit_l823_82363


namespace repeating_decimal_sum_as_fraction_l823_82350

theorem repeating_decimal_sum_as_fraction :
  let d1 := 1 / 9    -- Representation of 0.\overline{1}
  let d2 := 1 / 99   -- Representation of 0.\overline{01}
  d1 + d2 = (4 : ℚ) / 33 := by
{
  sorry
}

end repeating_decimal_sum_as_fraction_l823_82350


namespace max_imaginary_part_of_roots_l823_82390

noncomputable def find_phi : Prop :=
  ∃ z : ℂ, z^6 - z^4 + z^2 - 1 = 0 ∧ (∀ w : ℂ, w^6 - w^4 + w^2 - 1 = 0 → z.im ≤ w.im) ∧ z.im = Real.sin (Real.pi / 4)

theorem max_imaginary_part_of_roots : find_phi :=
sorry

end max_imaginary_part_of_roots_l823_82390


namespace minimum_value_of_polynomial_l823_82376

-- Define the polynomial expression
def polynomial_expr (x : ℝ) : ℝ := (8 - x) * (6 - x) * (8 + x) * (6 + x)

-- State the theorem with the minimum value
theorem minimum_value_of_polynomial : ∃ x : ℝ, polynomial_expr x = -196 := by
  sorry

end minimum_value_of_polynomial_l823_82376


namespace problem_statement_l823_82358

theorem problem_statement (m n c d a : ℝ)
  (h1 : m = -n)
  (h2 : c * d = 1)
  (h3 : a = 2) :
  Real.sqrt (c * d) + 2 * (m + n) - a = -1 :=
by
  -- Proof steps are skipped with sorry 
  sorry

end problem_statement_l823_82358


namespace time_saved_is_35_minutes_l823_82370

-- Define the speed and distances for each day
def monday_distance := 3
def wednesday_distance := 3
def friday_distance := 3
def sunday_distance := 4
def speed_monday := 6
def speed_wednesday := 4
def speed_friday := 5
def speed_sunday := 3
def speed_uniform := 5

-- Calculate the total time spent on the treadmill originally
def time_monday := monday_distance / speed_monday
def time_wednesday := wednesday_distance / speed_wednesday
def time_friday := friday_distance / speed_friday
def time_sunday := sunday_distance / speed_sunday
def total_time := time_monday + time_wednesday + time_friday + time_sunday

-- Calculate the total time if speed was uniformly 5 mph 
def total_distance := monday_distance + wednesday_distance + friday_distance + sunday_distance
def total_time_uniform := total_distance / speed_uniform

-- Time saved if walking at 5 mph every day
def time_saved := total_time - total_time_uniform

-- Convert time saved to minutes
def minutes_saved := time_saved * 60

theorem time_saved_is_35_minutes : minutes_saved = 35 := by
  sorry

end time_saved_is_35_minutes_l823_82370


namespace neg_proposition_equiv_l823_82362

theorem neg_proposition_equiv (p : Prop) : (¬ (∃ n : ℕ, 2^n > 1000)) = (∀ n : ℕ, 2^n ≤ 1000) :=
by
  sorry

end neg_proposition_equiv_l823_82362


namespace number_of_short_trees_to_plant_l823_82373

-- Definitions of the conditions
def current_short_trees : ℕ := 41
def current_tall_trees : ℕ := 44
def total_short_trees_after_planting : ℕ := 98

-- The statement to be proved
theorem number_of_short_trees_to_plant :
  total_short_trees_after_planting - current_short_trees = 57 :=
by
  -- Proof goes here
  sorry

end number_of_short_trees_to_plant_l823_82373


namespace smallest_repunit_divisible_by_97_l823_82393

theorem smallest_repunit_divisible_by_97 :
  ∃ n : ℕ, (∃ d : ℤ, 10^n - 1 = 97 * 9 * d) ∧ (∀ m : ℕ, (∃ d : ℤ, 10^m - 1 = 97 * 9 * d) → n ≤ m) :=
by
  sorry

end smallest_repunit_divisible_by_97_l823_82393


namespace problem_statement_l823_82343

-- Define that the function f is even
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define that the function f satisfies f(x) = f(2 - x)
def satisfies_symmetry (f : ℝ → ℝ) : Prop := ∀ x, f x = f (2 - x)

-- Define that the function f is decreasing on a given interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Define that the function f is increasing on a given interval
def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Given hypotheses and the theorem to prove. We use two statements for clarity.
theorem problem_statement (f : ℝ → ℝ) 
  (h_even : is_even f) 
  (h_symmetry : satisfies_symmetry f) 
  (h_decreasing_1_2 : is_decreasing_on f 1 2) : 
  is_increasing_on f (-2) (-1) ∧ is_decreasing_on f 3 4 := 
by 
  sorry

end problem_statement_l823_82343


namespace intersection_of_complements_l823_82388

open Set

theorem intersection_of_complements (U : Set ℕ) (A B : Set ℕ)
  (hU : U = {1,2,3,4,5,6,7,8})
  (hA : A = {3,4,5})
  (hB : B = {1,3,6}) :
  (U \ A) ∩ (U \ B) = {2,7,8} := by
  rw [hU, hA, hB]
  sorry

end intersection_of_complements_l823_82388


namespace highest_nitrogen_percentage_l823_82307

-- Define molar masses for each compound
def molar_mass_NH2OH : Float := 33.0
def molar_mass_NH4NO2 : Float := 64.1 
def molar_mass_N2O3 : Float := 76.0
def molar_mass_NH4NH2CO2 : Float := 78.1

-- Define mass of nitrogen atoms
def mass_of_nitrogen : Float := 14.0

-- Define the percentage calculations
def percentage_NH2OH : Float := (mass_of_nitrogen / molar_mass_NH2OH) * 100.0
def percentage_NH4NO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NO2) * 100.0
def percentage_N2O3 : Float := (2 * mass_of_nitrogen / molar_mass_N2O3) * 100.0
def percentage_NH4NH2CO2 : Float := (2 * mass_of_nitrogen / molar_mass_NH4NH2CO2) * 100.0

-- Define the proof problem
theorem highest_nitrogen_percentage : percentage_NH4NO2 > percentage_NH2OH ∧
                                      percentage_NH4NO2 > percentage_N2O3 ∧
                                      percentage_NH4NO2 > percentage_NH4NH2CO2 :=
by 
  sorry

end highest_nitrogen_percentage_l823_82307


namespace find_smallest_int_cube_ends_368_l823_82327

theorem find_smallest_int_cube_ends_368 :
  ∃ n : ℕ, n > 0 ∧ n^3 % 500 = 368 ∧ n = 14 :=
by
  sorry

end find_smallest_int_cube_ends_368_l823_82327


namespace bob_daily_work_hours_l823_82392

theorem bob_daily_work_hours
  (total_hours_in_month : ℕ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (total_working_days : ℕ)
  (daily_working_hours : ℕ)
  (h1 : total_hours_in_month = 200)
  (h2 : days_per_week = 5)
  (h3 : weeks_per_month = 4)
  (h4 : total_working_days = days_per_week * weeks_per_month)
  (h5 : daily_working_hours = total_hours_in_month / total_working_days) :
  daily_working_hours = 10 := 
sorry

end bob_daily_work_hours_l823_82392


namespace find_k_l823_82397

theorem find_k (x y k : ℝ) (h₁ : 3 * x + y = k) (h₂ : -1.2 * x + y = -20) (hx : x = 7) : k = 9.4 :=
by
  sorry

end find_k_l823_82397


namespace tedra_tomato_harvest_l823_82374

theorem tedra_tomato_harvest (W T F : ℝ) 
    (h1 : T = W / 2) 
    (h2 : W + T + F = 2000) 
    (h3 : F - 700 = 700) : 
    W = 400 := 
sorry

end tedra_tomato_harvest_l823_82374


namespace solve_equation_l823_82354

theorem solve_equation : ∃ x : ℚ, (2*x + 1) / 4 - 1 = x - (10*x + 1) / 12 ∧ x = 5 / 2 :=
by
  sorry

end solve_equation_l823_82354


namespace find_B_l823_82304

theorem find_B : 
  ∀ (A B : ℕ), A ≤ 9 → B ≤ 9 → (600 + 10 * A + 5) + (100 + B) = 748 → B = 3 :=
by
  intros A B hA hB hEq
  sorry

end find_B_l823_82304


namespace find_k_l823_82315

open Real

-- Define the operation "※"
def star (a b : ℝ) : ℝ := a * b + a + b^2

-- Define the main theorem stating the problem
theorem find_k (k : ℝ) (hk : k > 0) (h : star 1 k = 3) : k = 1 := by
  sorry

end find_k_l823_82315


namespace twin_primes_divisible_by_12_l823_82336

def isTwinPrime (p q : ℕ) : Prop :=
  p < q ∧ Nat.Prime p ∧ Nat.Prime q ∧ q - p = 2

theorem twin_primes_divisible_by_12 {p q r s : ℕ} 
  (h1 : isTwinPrime p q) 
  (h2 : p > 3) 
  (h3 : isTwinPrime r s) 
  (h4 : r > 3) :
  12 ∣ (p * r - q * s) := by
  sorry

end twin_primes_divisible_by_12_l823_82336


namespace trapezium_side_length_l823_82324

theorem trapezium_side_length (a b h A x : ℝ) 
  (ha : a = 20) (hh : h = 15) (hA : A = 285) 
  (h_formula : A = 1 / 2 * (a + b) * h) : 
  b = 18 :=
by
  sorry

end trapezium_side_length_l823_82324


namespace ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l823_82372

-- Definitions based on the given conditions.
def total_students : ℕ := 25
def percent_girls : ℕ := 60
def percent_boys_like_bb : ℕ := 40
def percent_girls_like_bb : ℕ := 80

-- Results from those conditions.
def num_girls : ℕ := percent_girls * total_students / 100
def num_boys : ℕ := total_students - num_girls
def num_boys_like_bb : ℕ := percent_boys_like_bb * num_boys / 100
def num_boys_dont_like_bb : ℕ := num_boys - num_boys_like_bb
def num_girls_like_bb : ℕ := percent_girls_like_bb * num_girls / 100

-- Proof Problem Statement
theorem ratio_of_girls_who_like_bb_to_boys_dont_like_bb :
  (num_girls_like_bb : ℕ) / num_boys_dont_like_bb = 2 / 1 :=
by
  sorry

end ratio_of_girls_who_like_bb_to_boys_dont_like_bb_l823_82372


namespace greatest_non_sum_complex_l823_82391

def is_complex (n : ℕ) : Prop :=
  ∃ p q : ℕ, p ≠ q ∧ Nat.Prime p ∧ Nat.Prime q ∧ p ∣ n ∧ q ∣ n

theorem greatest_non_sum_complex : ∀ n : ℕ, (¬ ∃ a b : ℕ, is_complex a ∧ is_complex b ∧ a + b = n) → n ≤ 23 :=
by {
  sorry
}

end greatest_non_sum_complex_l823_82391


namespace fencing_required_l823_82368

theorem fencing_required
  (L : ℝ) (W : ℝ) (A : ℝ) (F : ℝ)
  (hL : L = 25)
  (hA : A = 880)
  (hArea : A = L * W)
  (hF : F = L + 2 * W) :
  F = 95.4 :=
by
  sorry

end fencing_required_l823_82368


namespace biggest_number_in_ratio_l823_82384

theorem biggest_number_in_ratio (A B C D : ℕ) (h1 : 2 * D = 5 * A) (h2 : 3 * D = 5 * B) (h3 : 4 * D = 5 * C) (h_sum : A + B + C + D = 1344) : D = 480 := 
by
  sorry

end biggest_number_in_ratio_l823_82384


namespace probability_of_hitting_target_at_least_once_l823_82357

theorem probability_of_hitting_target_at_least_once :
  (∀ (p1 p2 : ℝ), p1 = 0.5 → p2 = 0.7 → (1 - (1 - p1) * (1 - p2)) = 0.85) :=
by
  intros p1 p2 h1 h2
  rw [h1, h2]
  -- This rw step simplifies (1 - (1 - 0.5) * (1 - 0.7)) to the desired result.
  sorry

end probability_of_hitting_target_at_least_once_l823_82357


namespace sum_of_integers_sqrt_485_l823_82385

theorem sum_of_integers_sqrt_485 (x y : ℕ) (h1 : x^2 + y^2 = 245) (h2 : x * y = 120) : x + y = Real.sqrt 485 :=
sorry

end sum_of_integers_sqrt_485_l823_82385


namespace problem_statement_l823_82394

noncomputable def calculateValue (n : ℕ) : ℕ :=
  Nat.choose (3 * n) (38 - n) + Nat.choose (n + 21) (3 * n)

theorem problem_statement : calculateValue 10 = 466 := by
  sorry

end problem_statement_l823_82394


namespace saly_needs_10_eggs_per_week_l823_82301

theorem saly_needs_10_eggs_per_week :
  let Saly_needs_per_week := S
  let Ben_needs_per_week := 14
  let Ked_needs_per_week := Ben_needs_per_week / 2
  let total_eggs_in_month := 124
  let weeks_per_month := 4
  let Ben_needs_per_month := Ben_needs_per_week * weeks_per_month
  let Ked_needs_per_month := Ked_needs_per_week * weeks_per_month
  let Saly_needs_per_month := total_eggs_in_month - (Ben_needs_per_month + Ked_needs_per_month)
  let S := Saly_needs_per_month / weeks_per_month
  Saly_needs_per_week = 10 :=
by
  sorry

end saly_needs_10_eggs_per_week_l823_82301


namespace sfl_entrances_l823_82340

theorem sfl_entrances (people_per_entrance total_people entrances : ℕ) 
  (h1: people_per_entrance = 283) 
  (h2: total_people = 1415) 
  (h3: total_people = people_per_entrance * entrances) 
  : entrances = 5 := 
  by 
  rw [h1, h2] at h3
  sorry

end sfl_entrances_l823_82340


namespace amount_after_3_years_l823_82300

theorem amount_after_3_years (P t A' : ℝ) (R : ℝ) :
  P = 800 → t = 3 → A' = 992 →
  (800 * ((R + 3) / 100) * 3 = 192) →
  (A = P * (1 + (R / 100) * t)) →
  A = 1160 := by
  intros hP ht hA' hR hA
  sorry

end amount_after_3_years_l823_82300


namespace find_dividend_l823_82352

theorem find_dividend (divisor : ℕ) (partial_quotient : ℕ) (dividend : ℕ) 
                       (h_divisor : divisor = 12)
                       (h_partial_quotient : partial_quotient = 909809) 
                       (h_calculation : dividend = divisor * partial_quotient) : 
                       dividend = 10917708 :=
by
  rw [h_divisor, h_partial_quotient] at h_calculation
  exact h_calculation


end find_dividend_l823_82352


namespace right_angled_triangle_count_in_pyramid_l823_82396

-- Define the cuboid and the triangular pyramid within it
variables (A B C D A₁ B₁ C₁ D₁ : Type)

-- Assume there exists a cuboid ABCD-A₁B₁C₁D₁
axiom cuboid : Prop

-- Define the triangular pyramid A₁-ABC
structure triangular_pyramid (A₁ A B C : Type) : Type :=
  (vertex₁ : A₁)
  (vertex₂ : A)
  (vertex₃ : B)
  (vertex4 : C)
  
-- The mathematical statement to prove: the number of right-angled triangles in A₁-ABC is 4
theorem right_angled_triangle_count_in_pyramid (A : Type) (B : Type) (C : Type) (A₁ : Type)
  (h_pyramid : triangular_pyramid A₁ A B C) (h_cuboid : cuboid) :
  ∃ n : ℕ, n = 4 :=
by
  sorry

end right_angled_triangle_count_in_pyramid_l823_82396


namespace tunnel_length_scale_l823_82389

theorem tunnel_length_scale (map_length_cm : ℝ) (scale_ratio : ℝ) (convert_factor : ℝ) : 
  map_length_cm = 7 → scale_ratio = 38000 → convert_factor = 100000 →
  (map_length_cm * scale_ratio / convert_factor) = 2.66 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end tunnel_length_scale_l823_82389


namespace total_price_correct_l823_82332

-- Definitions of given conditions
def original_price : Float := 120
def discount_rate : Float := 0.30
def tax_rate : Float := 0.08

-- Definition of the final selling price
def sale_price : Float := original_price * (1 - discount_rate)
def total_selling_price : Float := sale_price * (1 + tax_rate)

-- Lean 4 statement to prove the total selling price is 90.72
theorem total_price_correct : total_selling_price = 90.72 := by
  sorry

end total_price_correct_l823_82332


namespace lcm_factor_l823_82321

theorem lcm_factor (A B : ℕ) (hcf : ℕ) (factor1 : ℕ) (factor2 : ℕ) 
  (hcf_eq : hcf = 15) (factor1_eq : factor1 = 11) (A_eq : A = 225) 
  (hcf_divides_A : hcf ∣ A) (lcm_eq : Nat.lcm A B = hcf * factor1 * factor2) : 
  factor2 = 15 :=
by
  sorry

end lcm_factor_l823_82321


namespace original_total_price_l823_82313

theorem original_total_price (total_selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) 
  (selling_price_with_profit : total_selling_price/2 = original_price * (1 + profit_percent))
  (selling_price_with_loss : total_selling_price/2 = original_price * (1 - loss_percent)) :
  (original_price / (1 + profit_percent) + original_price / (1 - loss_percent) = 1333 + 1 / 3) := 
by
  sorry

end original_total_price_l823_82313


namespace John_days_per_week_l823_82305

theorem John_days_per_week
    (patients_first : ℕ := 20)
    (patients_increase_rate : ℕ := 20)
    (patients_second : ℕ := (20 + (20 * 20 / 100)))
    (total_weeks_year : ℕ := 50)
    (total_patients_year : ℕ := 11000) :
    ∃ D : ℕ, (20 * D + (20 + (20 * 20 / 100)) * D) * total_weeks_year = total_patients_year ∧ D = 5 := by
  sorry

end John_days_per_week_l823_82305


namespace fuel_a_added_l823_82308

theorem fuel_a_added (capacity : ℝ) (ethanolA : ℝ) (ethanolB : ℝ) (total_ethanol : ℝ) (x : ℝ) : 
  capacity = 200 ∧ ethanolA = 0.12 ∧ ethanolB = 0.16 ∧ total_ethanol = 28 →
  0.12 * x + 0.16 * (200 - x) = 28 → x = 100 :=
sorry

end fuel_a_added_l823_82308


namespace exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l823_82344

theorem exists_integers_for_x_squared_minus_y_squared_eq_a_fifth (a : ℤ) : 
  ∃ x y : ℤ, x^2 - y^2 = a^5 :=
sorry

end exists_integers_for_x_squared_minus_y_squared_eq_a_fifth_l823_82344


namespace percentage_people_taking_bus_l823_82309

-- Definitions
def population := 80
def car_pollution := 10 -- pounds of carbon per car per year
def bus_pollution := 100 -- pounds of carbon per bus per year
def bus_capacity := 40 -- people per bus
def carbon_reduction := 100 -- pounds of carbon reduced per year after the bus is introduced

-- Problem statement in Lean 4
theorem percentage_people_taking_bus :
  (10 / 80 : ℝ) = 0.125 :=
by
  sorry

end percentage_people_taking_bus_l823_82309


namespace math_problem_l823_82369

theorem math_problem (x y : ℝ) (h1 : x - 2 * y = 4) (h2 : x * y = 8) :
  x^2 + 4 * y^2 = 48 :=
sorry

end math_problem_l823_82369


namespace min_people_liking_both_l823_82330

theorem min_people_liking_both {A B U : Finset ℕ} (hU : U.card = 150) (hA : A.card = 130) (hB : B.card = 120) :
  (A ∩ B).card ≥ 100 :=
by
  -- Proof to be filled later
  sorry

end min_people_liking_both_l823_82330


namespace find_initial_candies_l823_82347

-- Define the initial number of candies as x
def initial_candies (x : ℕ) : ℕ :=
  let first_day := (3 * x) / 4 - 3
  let second_day := (3 * first_day) / 5 - 5
  let third_day := second_day - 7
  let final_candies := (5 * third_day) / 6
  final_candies

-- Formal statement of the theorem
theorem find_initial_candies (x : ℕ) (h : initial_candies x = 10) : x = 44 :=
  sorry

end find_initial_candies_l823_82347


namespace sequence_contains_perfect_square_l823_82346

noncomputable def f (n : ℕ) : ℕ := n + Nat.floor (Real.sqrt n)

theorem sequence_contains_perfect_square (m : ℕ) : ∃ k : ℕ, ∃ p : ℕ, f^[k] m = p * p := by
  sorry

end sequence_contains_perfect_square_l823_82346


namespace num_tables_l823_82351

/-- Given conditions related to tables, stools, and benches, we want to prove the number of tables -/
theorem num_tables 
  (t s b : ℕ) 
  (h1 : s = 8 * t)
  (h2 : b = 2 * t)
  (h3 : 3 * s + 6 * b + 4 * t = 816) : 
  t = 20 := 
sorry

end num_tables_l823_82351


namespace find_smallest_n_l823_82337

def is_perfect_square (x : ℕ) : Prop := ∃ k : ℕ, k * k = x
def is_perfect_cube (x : ℕ) : Prop := ∃ k : ℕ, k * k * k = x

theorem find_smallest_n (n : ℕ) : 
  (is_perfect_square (5 * n) ∧ is_perfect_cube (3 * n)) ∧ n = 225 :=
by
  sorry

end find_smallest_n_l823_82337


namespace f_seven_l823_82345

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function (h : ℝ) : f (-h) = -f (h)
axiom periodic_function (h : ℝ) : f (h + 4) = f (h)
axiom f_one : f 1 = 2

theorem f_seven : f (7) = -2 :=
by
  sorry

end f_seven_l823_82345


namespace volume_of_remaining_solid_after_removing_tetrahedra_l823_82310

theorem volume_of_remaining_solid_after_removing_tetrahedra :
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  cube_volume - 8 * tetrahedron_volume = 5 / 6 := by
  let cube_volume := 1
  let tetrahedron_volume := (1/3) * (1/2) * (1/2) * (1/2) * (1/2)
  have h : cube_volume - 8 * tetrahedron_volume = 5 / 6 := sorry
  exact h

end volume_of_remaining_solid_after_removing_tetrahedra_l823_82310


namespace log_relation_l823_82335

noncomputable def a := Real.log 3 / Real.log 4
noncomputable def b := Real.log 3 / Real.log 0.4
def c := (1 / 2) ^ 2

theorem log_relation (h1 : a = Real.log 3 / Real.log 4)
                     (h2 : b = Real.log 3 / Real.log 0.4)
                     (h3 : c = (1 / 2) ^ 2) : a > c ∧ c > b :=
by
  sorry

end log_relation_l823_82335


namespace range_of_a_l823_82331

theorem range_of_a (f : ℝ → ℝ) (h1 : ∀ x, f (x - 3) = f (3 - (x - 3))) (h2 : ∀ x, 0 ≤ x → f x = x^2 + 2 * x) :
  {a : ℝ | f (2 - a^2) > f a} = {a | -2 < a ∧ a < 1} :=
by
  sorry

end range_of_a_l823_82331


namespace f_zero_eq_one_positive_for_all_x_l823_82360

variables {R : Type*} [LinearOrderedField R] (f : R → R)

-- Conditions
axiom domain (x : R) : true -- This translates that f has domain (-∞, ∞)
axiom non_constant (x1 x2 : R) (h : x1 ≠ x2) : f x1 ≠ f x2
axiom functional_eq (x y : R) : f (x + y) = f x * f y

-- Questions
theorem f_zero_eq_one : f 0 = 1 :=
sorry

theorem positive_for_all_x (x : R) : f x > 0 :=
sorry

end f_zero_eq_one_positive_for_all_x_l823_82360


namespace marie_needs_8_days_to_pay_for_cash_register_l823_82314

-- Definitions of the conditions
def cost_of_cash_register : ℕ := 1040
def price_per_loaf : ℕ := 2
def loaves_per_day : ℕ := 40
def price_per_cake : ℕ := 12
def cakes_per_day : ℕ := 6
def daily_rent : ℕ := 20
def daily_electricity : ℕ := 2

-- Derive daily income and expenses
def daily_income : ℕ := (price_per_loaf * loaves_per_day) + (price_per_cake * cakes_per_day)
def daily_expenses : ℕ := daily_rent + daily_electricity
def daily_profit : ℕ := daily_income - daily_expenses

-- Define days needed to pay for the cash register
def days_needed : ℕ := cost_of_cash_register / daily_profit

-- Proof goal
theorem marie_needs_8_days_to_pay_for_cash_register : days_needed = 8 := by
  sorry

end marie_needs_8_days_to_pay_for_cash_register_l823_82314


namespace question1_solution_question2_solution_l823_82377

-- Definitions of the problem conditions
def f (x a : ℝ) : ℝ := abs (x - a)

-- First proof problem (Question 1)
theorem question1_solution (x : ℝ) : (f x 2) ≥ (4 - abs (x - 4)) ↔ (x ≥ 5 ∨ x ≤ 1) :=
by sorry

-- Second proof problem (Question 2)
theorem question2_solution (x : ℝ) (a : ℝ) (h_sol : 1 ≤ x ∧ x ≤ 2) 
  (h_ineq : abs (f (2 * x + a) a - 2 * f x a) ≤ 2) : a = 3 :=
by sorry

end question1_solution_question2_solution_l823_82377
