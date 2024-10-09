import Mathlib

namespace sqrt_sum_eq_ten_l997_99730

theorem sqrt_sum_eq_ten :
  Real.sqrt ((5 - 4*Real.sqrt 2)^2) + Real.sqrt ((5 + 4*Real.sqrt 2)^2) = 10 := 
by 
  sorry

end sqrt_sum_eq_ten_l997_99730


namespace weight_of_new_person_l997_99712

theorem weight_of_new_person (A : ℤ) (avg_weight_dec : ℤ) (n : ℤ) (new_avg : ℤ)
  (h1 : A = 102)
  (h2 : avg_weight_dec = 2)
  (h3 : n = 30) 
  (h4 : new_avg = A - avg_weight_dec) : 
  (31 * new_avg) - (30 * A) = 40 := 
by 
  sorry

end weight_of_new_person_l997_99712


namespace general_term_arithmetic_sequence_sum_first_n_terms_l997_99725

noncomputable def a_n (n : ℕ) : ℤ :=
  3 * n - 1

def b_n (n : ℕ) (b : ℕ → ℚ) : Prop :=
  (b 1 = 1) ∧ (b 2 = 1 / 3) ∧ ∀ n : ℕ, a_n n * b (n + 1) = n * b n

def sum_b_n (n : ℕ) (b : ℕ → ℚ) : ℚ :=
  (3 / 2) - (1 / (2 * (3 ^ (n - 1))))

theorem general_term_arithmetic_sequence (n : ℕ) :
  a_n n = 3 * n - 1 := by sorry

theorem sum_first_n_terms (n : ℕ) (b : ℕ → ℚ) (h : b_n n b) :
  sum_b_n n b = (3 / 2) - (1 / (2 * (3 ^ (n - 1)))) := by sorry

end general_term_arithmetic_sequence_sum_first_n_terms_l997_99725


namespace solve_for_m_l997_99721

noncomputable def f (x : ℝ) : ℝ := 2^x + x - 4

theorem solve_for_m (m : ℤ) (h : ∃ x : ℝ, 2^x + x = 4 ∧ m ≤ x ∧ x ≤ m + 1) : m = 1 :=
by
  sorry

end solve_for_m_l997_99721


namespace train_crosses_pole_in_time_l997_99738

noncomputable def time_to_cross_pole (length : ℝ) (speed_kmh : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  length / speed_ms

theorem train_crosses_pole_in_time :
  ∀ (length speed_kmh : ℝ), length = 240 → speed_kmh = 126 →
    time_to_cross_pole length speed_kmh = 6.8571 :=
by
  intros length speed_kmh h_length h_speed
  rw [h_length, h_speed, time_to_cross_pole]
  sorry

end train_crosses_pole_in_time_l997_99738


namespace first_term_arithmetic_sum_l997_99729

theorem first_term_arithmetic_sum 
  (T : ℕ → ℚ) (b : ℚ) (d : ℚ) (h₁ : ∀ n, T n = n * (2 * b + (n - 1) * d) / 2)
  (h₂ : d = 5)
  (h₃ : ∀ n, (T (4 * n)) / (T n) = (16 : ℚ)) : 
  b = 5 / 2 :=
sorry

end first_term_arithmetic_sum_l997_99729


namespace find_k_range_l997_99722

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 1 then -x^2 + x else Real.log x / Real.log (1 / 3)

def g (x k : ℝ) : ℝ :=
abs (x - k) + abs (x - 1)

theorem find_k_range (k : ℝ) :
  (∀ x1 x2 : ℝ, f x1 ≤ g x2 k) → (k ≤ 3 / 4 ∨ k ≥ 5 / 4) :=
by
  sorry

end find_k_range_l997_99722


namespace find_range_of_a_l997_99711

variable {f : ℝ → ℝ}
noncomputable def domain_f : Set ℝ := {x | 7 ≤ x ∧ x < 15}
noncomputable def domain_f_2x_plus_1 : Set ℝ := {x | 3 ≤ x ∧ x < 7}
noncomputable def B (a : ℝ) : Set ℝ := {x | x < a ∨ x > a + 1}
noncomputable def A_or_B_eq_r (a : ℝ) : Prop := domain_f_2x_plus_1 ∪ B a = Set.univ

theorem find_range_of_a (a : ℝ) : 
  A_or_B_eq_r a → 3 ≤ a ∧ a < 6 := 
sorry

end find_range_of_a_l997_99711


namespace find_number_of_students_l997_99705

-- Parameters
variable (n : ℕ) (C : ℕ)
def first_and_last_picked_by_sam (n : ℕ) (C : ℕ) : Prop := 
  C + 1 = 2 * n

-- Conditions: number of candies is 120, the bag completes 2 full rounds at the table.
theorem find_number_of_students
  (C : ℕ) (h_C: C = 120) (h_rounds: 2 * n = C):
  n = 60 :=
by
  sorry

end find_number_of_students_l997_99705


namespace inequality_solution_l997_99770

theorem inequality_solution (x : ℝ) (h : x ≠ 5) :
    (15 ≤ x * (x - 2) / (x - 5) ^ 2) ↔ (4.1933 ≤ x ∧ x < 5 ∨ 5 < x ∧ x ≤ 6.3767) :=
by
  sorry

end inequality_solution_l997_99770


namespace additional_men_joined_l997_99728

noncomputable def solve_problem := 
  let M := 1000
  let days_initial := 17
  let days_new := 11.333333333333334
  let total_provisions := M * days_initial
  let additional_men := (total_provisions / days_new) - M
  additional_men

theorem additional_men_joined : solve_problem = 500 := by
  sorry

end additional_men_joined_l997_99728


namespace sum_of_all_different_possible_areas_of_cool_rectangles_l997_99703

-- Define the concept of a cool rectangle
def is_cool_rectangle (a b : ℕ) : Prop :=
  a * b = 2 * (2 * a + 2 * b)

-- Define the function to calculate the area of a rectangle
def area (a b : ℕ) : ℕ := a * b

-- Define the set of pairs (a, b) that satisfy the cool rectangle condition
def cool_rectangle_pairs : List (ℕ × ℕ) :=
  [(5, 20), (6, 12), (8, 8)]

-- Calculate the sum of all different possible areas of cool rectangles
def sum_of_cool_rectangle_areas : ℕ :=
  List.sum (cool_rectangle_pairs.map (λ p => area p.fst p.snd))

-- Theorem statement
theorem sum_of_all_different_possible_areas_of_cool_rectangles :
  sum_of_cool_rectangle_areas = 236 :=
by
  -- This is where the proof would go based on the given solution.
  sorry

end sum_of_all_different_possible_areas_of_cool_rectangles_l997_99703


namespace symmetric_points_y_axis_l997_99714

theorem symmetric_points_y_axis (a b : ℝ) (h1 : a - b = -3) (h2 : 2 * a + b = 2) :
  a = -1 / 3 ∧ b = 8 / 3 :=
by
  sorry

end symmetric_points_y_axis_l997_99714


namespace smaller_fraction_l997_99753

variable (x y : ℚ)

theorem smaller_fraction (h1 : x + y = 13 / 14) (h2 : x * y = 1 / 8) : min x y = 1 / 6 :=
by
  sorry

end smaller_fraction_l997_99753


namespace range_of_fraction_l997_99702

theorem range_of_fraction (x y : ℝ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 3 ≤ y ∧ y ≤ 6) :
  ∀ z, z = x / y → (1 / 6 ≤ z ∧ z ≤ 4 / 3) :=
sorry

end range_of_fraction_l997_99702


namespace equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l997_99785

theorem equation1_solutions (x : ℝ) : x^2 - 4 * x - 1 = 0 ↔ (x = 2 + Real.sqrt 5 ∨ x = 2 - Real.sqrt 5) :=
by sorry

theorem equation2_solutions (x : ℝ) : x * (3 * x + 1) = 2 * (3 * x + 1) ↔ (x = -1 / 3 ∨ x = 2) :=
by sorry

theorem equation3_solutions (x : ℝ) : 2 * x^2 + x - 4 = 0 ↔ (x = (-1 + Real.sqrt 33) / 4 ∨ x = (-1 - Real.sqrt 33) / 4) :=
by sorry

theorem equation4_no_real_solutions (x : ℝ) : ¬ ∃ x, 4 * x^2 - 3 * x + 1 = 0 :=
by sorry

end equation1_solutions_equation2_solutions_equation3_solutions_equation4_no_real_solutions_l997_99785


namespace x_plus_y_value_l997_99791

def sum_evens_40_to_60 : ℕ :=
  (40 + 42 + 44 + 46 + 48 + 50 + 52 + 54 + 56 + 58 + 60)

def num_evens_40_to_60 : ℕ := 11

theorem x_plus_y_value : sum_evens_40_to_60 + num_evens_40_to_60 = 561 := by
  sorry

end x_plus_y_value_l997_99791


namespace sin_cos_eq_one_sol_set_l997_99710

-- Define the interval
def in_interval (x : ℝ) : Prop := 0 ≤ x ∧ x < 2 * Real.pi

-- Define the condition
def satisfies_eq (x : ℝ) : Prop := Real.sin x + Real.cos x = 1

-- Theorem statement: prove that the solution set is {0, π/2}
theorem sin_cos_eq_one_sol_set :
  ∀ (x : ℝ), in_interval x → satisfies_eq x ↔ x = 0 ∨ x = Real.pi / 2 := by
  sorry

end sin_cos_eq_one_sol_set_l997_99710


namespace find_angle_A_l997_99706

theorem find_angle_A 
  (a b c A B C : ℝ)
  (h₀ : a = Real.sqrt 2)
  (h₁ : b = 2)
  (h₂ : Real.sin B - Real.cos B = Real.sqrt 2)
  (h₃ : a^2 = b^2 + c^2 - 2 * b * c * Real.cos A)
  : A = Real.pi / 6 := 
  sorry

end find_angle_A_l997_99706


namespace corner_movement_l997_99772

-- Definition of corner movement problem
def canMoveCornerToBottomRight (m n : ℕ) : Prop :=
  m ≥ 2 ∧ n ≥ 2 ∧ (m % 2 = 1 ∧ n % 2 = 1)

theorem corner_movement (m n : ℕ) (h1 : m ≥ 2) (h2 : n ≥ 2) :
  (canMoveCornerToBottomRight m n ↔ (m % 2 = 1 ∧ n % 2 = 1)) :=
by
  sorry  -- Proof is omitted

end corner_movement_l997_99772


namespace abs_inequality_l997_99700

theorem abs_inequality (x : ℝ) (h : |x - 2| < 1) : 1 < x ∧ x < 3 := by
  sorry

end abs_inequality_l997_99700


namespace problem_l997_99720

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (2 * x + Real.pi / 4)

theorem problem
  (h1 : f (Real.pi / 8) = 2)
  (h2 : f (5 * Real.pi / 8) = -2) :
  (∀ x : ℝ, f x = 1 ↔ 
    (∃ k : ℤ, x = -Real.pi / 24 + k * Real.pi) ∨
    (∃ k : ℤ, x = 7 * Real.pi / 24 + k * Real.pi)) :=
by
  sorry

end problem_l997_99720


namespace simplify_fraction_l997_99789

theorem simplify_fraction :
  (1 : ℚ) / ((1 / (1 / 3 : ℚ) ^ 1) + (1 / (1 / 3 : ℚ) ^ 2) + (1 / (1 / 3 : ℚ) ^ 3) + (1 / (1 / 3 : ℚ) ^ 4)) = 1 / 120 := 
by 
  sorry

end simplify_fraction_l997_99789


namespace sum_eq_prod_S1_sum_eq_prod_S2_l997_99773

def S1 : List ℕ := [1, 1, 1, 1, 1, 1, 2, 8]
def S2 : List ℕ := [1, 1, 1, 1, 1, 2, 2, 3]

def sum_list (l : List ℕ) : ℕ := l.foldr Nat.add 0
def prod_list (l : List ℕ) : ℕ := l.foldr Nat.mul 1

theorem sum_eq_prod_S1 : sum_list S1 = prod_list S1 := 
by
  sorry

theorem sum_eq_prod_S2 : sum_list S2 = prod_list S2 := 
by
  sorry

end sum_eq_prod_S1_sum_eq_prod_S2_l997_99773


namespace polynomial_solution_l997_99749

theorem polynomial_solution (P : Polynomial ℝ) (h_0 : P.eval 0 = 0) (h_func : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) :
  P = Polynomial.X :=
sorry

end polynomial_solution_l997_99749


namespace parabola_focus_l997_99726

-- Definitions used in the conditions
def parabola_eq (p : ℝ) (x : ℝ) : ℝ := 2 * p * x^2
def passes_through (p : ℝ) : Prop := parabola_eq p 1 = 4

-- The proof that the coordinates of the focus are (0, 1/16) given the conditions
theorem parabola_focus (p : ℝ) (h : passes_through p) : p = 2 → (0, 1 / 16) = (0, 1 / (4 * p)) :=
by
  sorry

end parabola_focus_l997_99726


namespace squirrel_climb_l997_99782

-- Define the problem conditions and the goal
variable (x : ℝ)

-- net_distance_climbed_every_two_minutes
def net_distance_climbed_every_two_minutes : ℝ := x - 2

-- distance_climbed_in_14_minutes
def distance_climbed_in_14_minutes : ℝ := 7 * (x - 2)

-- distance_climbed_in_15th_minute
def distance_climbed_in_15th_minute : ℝ := x

-- total_distance_climbed_in_15_minutes
def total_distance_climbed_in_15_minutes : ℝ := 26

-- Theorem: proving x based on the conditions
theorem squirrel_climb : 
  7 * (x - 2) + x = 26 -> x = 5 := by
  intros h
  sorry

end squirrel_climb_l997_99782


namespace frank_problems_per_type_l997_99759

-- Definitions based on the problem conditions
def bill_problems : ℕ := 20
def ryan_problems : ℕ := 2 * bill_problems
def frank_problems : ℕ := 3 * ryan_problems
def types_of_problems : ℕ := 4

-- The proof statement equivalent to the math problem
theorem frank_problems_per_type : frank_problems / types_of_problems = 30 :=
by 
  -- skipping the proof steps
  sorry

end frank_problems_per_type_l997_99759


namespace friends_attended_reception_l997_99781

-- Definition of the given conditions
def total_guests : ℕ := 180
def couples_per_side : ℕ := 20

-- Statement based on the given problem
theorem friends_attended_reception : 
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  let friends := total_guests - family_guests
  friends = 100 :=
by
  -- We define the family_guests calculation
  let family_guests := 2 * couples_per_side + 2 * couples_per_side
  -- We define the friends calculation
  let friends := total_guests - family_guests
  -- We state the conclusion
  show friends = 100
  sorry

end friends_attended_reception_l997_99781


namespace average_height_corrected_l997_99783

-- Defining the conditions as functions and constants
def incorrect_average_height : ℝ := 175
def number_of_students : ℕ := 30
def incorrect_height : ℝ := 151
def actual_height : ℝ := 136

-- The target average height to prove
def target_actual_average_height : ℝ := 174.5

-- Main theorem stating the problem
theorem average_height_corrected : 
  (incorrect_average_height * number_of_students - (incorrect_height - actual_height)) / number_of_students = target_actual_average_height :=
by
  sorry

end average_height_corrected_l997_99783


namespace arithmetic_sequence_a10_l997_99793

theorem arithmetic_sequence_a10 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : (S 9) / 9 - (S 5) / 5 = 4)
  (hSn : ∀ n, S n = n * (2 + (n - 1) / 2 * (a 2 - a 1) )) : 
  a 10 = 20 := 
sorry

end arithmetic_sequence_a10_l997_99793


namespace johns_horses_l997_99709

theorem johns_horses 
  (feeding_per_day : ℕ := 2) 
  (food_per_feeding : ℝ := 20) 
  (bag_weight : ℝ := 1000) 
  (num_bags : ℕ := 60) 
  (days : ℕ := 60)
  (total_food : ℝ := num_bags * bag_weight) 
  (daily_food_consumption : ℝ := total_food / days) 
  (food_per_horse_per_day : ℝ := food_per_feeding * feeding_per_day) :
  ∀ H : ℝ, (daily_food_consumption / food_per_horse_per_day = H) → H = 25 := 
by
  intros H hH
  sorry

end johns_horses_l997_99709


namespace pages_per_brochure_l997_99787

-- Define the conditions
def single_page_spreads := 20
def double_page_spreads := 2 * single_page_spreads
def pages_per_double_spread := 2
def pages_from_single := single_page_spreads
def pages_from_double := double_page_spreads * pages_per_double_spread
def total_pages_from_spreads := pages_from_single + pages_from_double
def ads_per_4_pages := total_pages_from_spreads / 4
def total_ads_pages := ads_per_4_pages
def total_pages := total_pages_from_spreads + total_ads_pages
def brochures := 25

-- The theorem we want to prove
theorem pages_per_brochure : total_pages / brochures = 5 :=
by
  -- This is a placeholder for the actual proof
  sorry

end pages_per_brochure_l997_99787


namespace dividend_in_terms_of_a_l997_99746

variable (a Q R D : ℕ)

-- Given conditions as hypotheses
def condition1 : Prop := D = 25 * Q
def condition2 : Prop := D = 7 * R
def condition3 : Prop := Q - R = 15
def condition4 : Prop := R = 3 * a

-- Prove that the dividend given these conditions equals the expected expression
theorem dividend_in_terms_of_a (a : ℕ) (Q : ℕ) (R : ℕ) (D : ℕ) :
  condition1 D Q → condition2 D R → condition3 Q R → condition4 R a →
  (D * Q + R) = 225 * a^2 + 1128 * a + 5625 :=
by
  intro h1 h2 h3 h4
  sorry

end dividend_in_terms_of_a_l997_99746


namespace min_value_of_function_l997_99707

theorem min_value_of_function : 
  ∃ (c : ℝ), (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2) ≥ c) ∧
             (∀ x : ℝ, (x ∈ Set.Icc (Real.pi / 6) (5 * Real.pi / 6)) → (2 * (Real.sin x) ^ 2 + 2 * Real.sin x - 1 / 2 = c) → c = 1) := 
sorry

end min_value_of_function_l997_99707


namespace solve_fraction_l997_99768

theorem solve_fraction (x : ℝ) (h : 2 / (x - 3) = 2) : x = 4 :=
by
  sorry

end solve_fraction_l997_99768


namespace hat_cost_l997_99771

theorem hat_cost (total_hats blue_hat_cost green_hat_cost green_hats : ℕ) 
  (h1 : total_hats = 85)
  (h2 : blue_hat_cost = 6)
  (h3 : green_hat_cost = 7)
  (h4 : green_hats = 20) :
  (total_hats - green_hats) * blue_hat_cost + green_hats * green_hat_cost = 530 := 
by sorry

end hat_cost_l997_99771


namespace find_relationship_l997_99775

variables (x y : ℝ)

def AB : ℝ × ℝ := (6, 1)
def BC : ℝ × ℝ := (x, y)
def CD : ℝ × ℝ := (-2, -3)
def DA : ℝ × ℝ := (4 - x, -2 - y)
def parallel (u v : ℝ × ℝ) : Prop := ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_relationship (h_parallel : parallel (x, y) (4 - x, -2 - y)) : x + 2 * y = 0 :=
sorry

end find_relationship_l997_99775


namespace intersection_point_proof_l997_99750

def intersect_point : Prop := 
  ∃ x y : ℚ, (5 * x - 6 * y = 3) ∧ (8 * x + 2 * y = 22) ∧ x = 69 / 29 ∧ y = 43 / 29

theorem intersection_point_proof : intersect_point :=
  sorry

end intersection_point_proof_l997_99750


namespace minimum_period_tan_2x_l997_99723

noncomputable def f (x : ℝ) : ℝ := Real.tan (2 * x)

theorem minimum_period_tan_2x : ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = Real.pi / 2 :=
by
  sorry

end minimum_period_tan_2x_l997_99723


namespace arithmetic_seq_property_l997_99741

theorem arithmetic_seq_property (a : ℕ → ℝ) (d : ℝ)
  (h_arith : ∀ n, a (n + 1) = a n + d)
  (h_cond : a 1 + 3 * a 8 + a 15 = 120) :
  2 * a 9 - a 10 = 24 := 
sorry

end arithmetic_seq_property_l997_99741


namespace total_legs_of_all_animals_l997_99727

def num_kangaroos : ℕ := 23
def num_goats : ℕ := 3 * num_kangaroos
def legs_of_kangaroo : ℕ := 2
def legs_of_goat : ℕ := 4

theorem total_legs_of_all_animals : num_kangaroos * legs_of_kangaroo + num_goats * legs_of_goat = 322 :=
by 
  sorry

end total_legs_of_all_animals_l997_99727


namespace triangle_QR_length_l997_99762

/-- Conditions for the triangles PQR and SQR sharing a side QR with given side lengths. -/
structure TriangleSetup where
  (PQ PR SR SQ QR : ℝ)
  (PQ_pos : PQ > 0)
  (PR_pos : PR > 0)
  (SR_pos : SR > 0)
  (SQ_pos : SQ > 0)
  (shared_side_QR : QR = QR)

/-- The problem statement asserting the least possible length of QR. -/
theorem triangle_QR_length (t : TriangleSetup) 
  (h1 : t.PQ = 8)
  (h2 : t.PR = 15)
  (h3 : t.SR = 10)
  (h4 : t.SQ = 25) :
  t.QR = 15 :=
by
  sorry

end triangle_QR_length_l997_99762


namespace remainder_div_1234567_256_l997_99764

theorem remainder_div_1234567_256 : 1234567 % 256 = 45 :=
by
  sorry

end remainder_div_1234567_256_l997_99764


namespace graph_must_pass_l997_99733

variable (f : ℝ → ℝ)
variable (finv : ℝ → ℝ)
variable (h_inv : ∀ y, f (finv y) = y ∧ finv (f y) = y)
variable (h_point : (2 - f 2) = 5)

theorem graph_must_pass : finv (-3) + 3 = 5 :=
by
  -- Proof to be filled in
  sorry

end graph_must_pass_l997_99733


namespace subcommittee_count_l997_99778

theorem subcommittee_count :
  let republicans := 10
  let democrats := 8
  let select_republicans := 4
  let select_democrats := 3
  let num_ways_republicans := Nat.choose republicans select_republicans
  let num_ways_democrats := Nat.choose democrats select_democrats
  let num_ways := num_ways_republicans * num_ways_democrats
  num_ways = 11760 :=
by
  sorry

end subcommittee_count_l997_99778


namespace parallel_lines_slope_eq_l997_99737

theorem parallel_lines_slope_eq (k : ℚ) :
  (5 = 3 * k) → k = 5 / 3 :=
by
  intros h
  sorry

end parallel_lines_slope_eq_l997_99737


namespace base7_addition_l997_99776

theorem base7_addition : (26:ℕ) + (245:ℕ) = 304 :=
  sorry

end base7_addition_l997_99776


namespace find_optimal_addition_l997_99734

theorem find_optimal_addition (m : ℝ) : 
  (1000 + (m - 1000) * 0.618 = 1618 ∨ 1000 + (m - 1000) * 0.618 = 2618) →
  (m = 2000 ∨ m = 2618) :=
sorry

end find_optimal_addition_l997_99734


namespace symmetric_about_z_correct_l997_99724

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def symmetric_about_z (p : Point3D) : Point3D :=
  { x := -p.x, y := -p.y, z := p.z }

theorem symmetric_about_z_correct (p : Point3D) :
  p = {x := 3, y := 4, z := 5} → symmetric_about_z p = {x := -3, y := -4, z := 5} :=
by
  sorry

end symmetric_about_z_correct_l997_99724


namespace spinner_probability_C_l997_99792

theorem spinner_probability_C 
  (P_A : ℚ) (P_B : ℚ) (P_C : ℚ) (P_D : ℚ)
  (hA : P_A = 1/3)
  (hB : P_B = 1/4)
  (hD : P_D = 1/6)
  (hSum : P_A + P_B + P_C + P_D = 1) :
  P_C = 1 / 4 := 
sorry

end spinner_probability_C_l997_99792


namespace range_of_a_l997_99716

noncomputable def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
noncomputable def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem range_of_a (a : ℝ) : (A a ∩ B a = {-2}) ↔ (a = -1) :=
by {
  sorry
}

end range_of_a_l997_99716


namespace gcd_max_two_digits_l997_99739

theorem gcd_max_two_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^11 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^12) :
  Nat.gcd a b < 100 :=
by
  -- Definitions
  sorry

end gcd_max_two_digits_l997_99739


namespace regular_polygon_interior_angle_ratio_l997_99777

theorem regular_polygon_interior_angle_ratio (r k : ℕ) (h1 : 180 - 360 / r = (5 : ℚ) / (3 : ℚ) * (180 - 360 / k)) (h2 : r = 2 * k) :
  r = 8 ∧ k = 4 :=
sorry

end regular_polygon_interior_angle_ratio_l997_99777


namespace B_share_after_tax_l997_99790

noncomputable def B_share (x : ℝ) : ℝ := 3 * x
noncomputable def salary_proportion (A B C D : ℝ) (x : ℝ) :=
  A = 2 * x ∧ B = 3 * x ∧ C = 4 * x ∧ D = 6 * x
noncomputable def D_more_than_C (D C : ℝ) : Prop :=
  D - C = 700
noncomputable def meets_minimum_wage (B : ℝ) : Prop :=
  B ≥ 1000
noncomputable def tax_deduction (B : ℝ) : ℝ :=
  if B > 1500 then B - 0.15 * (B - 1500) else B

theorem B_share_after_tax (A B C D : ℝ) (x : ℝ) (h1 : salary_proportion A B C D x)
  (h2 : D_more_than_C D C) (h3 : meets_minimum_wage B) :
  tax_deduction B = 1050 :=
by
  sorry

end B_share_after_tax_l997_99790


namespace cos_2000_eq_neg_inv_sqrt_l997_99774

theorem cos_2000_eq_neg_inv_sqrt (a : ℝ) (h : Real.tan (20 * Real.pi / 180) = a) :
  Real.cos (2000 * Real.pi / 180) = -1 / Real.sqrt (1 + a^2) :=
sorry

end cos_2000_eq_neg_inv_sqrt_l997_99774


namespace sum_of_reciprocals_of_shifted_roots_l997_99760

noncomputable def cubic_poly (x : ℝ) := 45 * x^3 - 75 * x^2 + 33 * x - 2

theorem sum_of_reciprocals_of_shifted_roots (a b c : ℝ) 
  (ha : cubic_poly a = 0) 
  (hb : cubic_poly b = 0) 
  (hc : cubic_poly c = 0)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ b ≠ c)
  (h_bounds_a : 0 < a ∧ a < 1)
  (h_bounds_b : 0 < b ∧ b < 1)
  (h_bounds_c : 0 < c ∧ c < 1) :
  (1 / (1 - a)) + (1 / (1 - b)) + (1 / (1 - c)) = 4 / 3 := 
sorry

end sum_of_reciprocals_of_shifted_roots_l997_99760


namespace find_m_l997_99719

theorem find_m (m : ℤ) (h1 : -180 < m ∧ m < 180) : 
  ((m = 45) ∨ (m = -135)) ↔ (Real.tan (m * Real.pi / 180) = Real.tan (225 * Real.pi / 180)) := 
by 
  sorry

end find_m_l997_99719


namespace range_of_a_l997_99788

noncomputable def f (a x : ℝ) : ℝ := a ^ x

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 2 → f a x < 2) : 
  (a ∈ Set.Ioo (Real.sqrt 2 / 2) 1 ∨ a ∈ Set.Ioo 1 (Real.sqrt 2)) :=
by
  sorry

end range_of_a_l997_99788


namespace sum_of_divisors_2000_l997_99701

theorem sum_of_divisors_2000 (n : ℕ) (h : n < 2000) :
  ∃ (s : Finset ℕ), (s ⊆ {1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100, 125, 200, 250, 400, 500, 1000, 2000}) ∧ s.sum id = n :=
by
  -- Proof goes here
  sorry

end sum_of_divisors_2000_l997_99701


namespace students_got_off_l997_99794

-- Define the number of students originally on the bus
def original_students : ℕ := 10

-- Define the number of students left on the bus after the first stop
def students_left : ℕ := 7

-- Prove that the number of students who got off the bus at the first stop is 3
theorem students_got_off : original_students - students_left = 3 :=
by
  sorry

end students_got_off_l997_99794


namespace compute_2a_minus_b_l997_99715

noncomputable def conditions (a b : ℝ) : Prop :=
  a^3 - 12 * a^2 + 47 * a - 60 = 0 ∧
  -b^3 + 12 * b^2 - 47 * b + 180 = 0

theorem compute_2a_minus_b (a b : ℝ) (h : conditions a b) : 2 * a - b = 2 := 
  sorry

end compute_2a_minus_b_l997_99715


namespace oliver_money_left_l997_99758

-- Definitions based on the given conditions
def initial_amount : ℕ := 9
def allowance_saved : ℕ := 5
def frisbee_cost : ℕ := 4
def puzzle_cost : ℕ := 3
def birthday_gift : ℕ := 8

-- The proof problem statement
theorem oliver_money_left : 
  initial_amount + allowance_saved - (frisbee_cost + puzzle_cost) + birthday_gift = 15 := 
by
  sorry

end oliver_money_left_l997_99758


namespace problem1_l997_99742

theorem problem1 (x : ℝ) (hx : x < 5/4) : 
  ∃ y : ℝ, y = 4 * x - 2 + 1 / (4 * x - 5) ∧ y ≤ 1 := 
sorry

end problem1_l997_99742


namespace find_y_l997_99754

theorem find_y (a b y : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : y > 0)
  (h4 : (2 * a)^(4 * b) = a^b * y^(3 * b)) : y = 2^(4 / 3) * a :=
by
  sorry

end find_y_l997_99754


namespace total_books_on_shelves_l997_99766

-- Definitions based on conditions
def num_shelves : Nat := 150
def books_per_shelf : Nat := 15

-- The statement to be proved
theorem total_books_on_shelves : num_shelves * books_per_shelf = 2250 := by
  sorry

end total_books_on_shelves_l997_99766


namespace largest_multiple_of_7_less_than_100_l997_99769

theorem largest_multiple_of_7_less_than_100 : ∃ n : ℕ, 7 * n < 100 ∧ ∀ m : ℕ, 7 * m < 100 → 7 * m ≤ 7 * n := by
  sorry

end largest_multiple_of_7_less_than_100_l997_99769


namespace problem_1_problem_2_l997_99751

noncomputable def O := (0, 0)
noncomputable def A := (1, 2)
noncomputable def B := (-3, 4)

noncomputable def vector_AB := (B.1 - A.1, B.2 - A.2)
noncomputable def magnitude_AB := Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

noncomputable def dot_OA_OB := A.1 * B.1 + A.2 * B.2
noncomputable def magnitude_OA := Real.sqrt (A.1^2 + A.2^2)
noncomputable def magnitude_OB := Real.sqrt (B.1^2 + B.2^2)
noncomputable def cosine_angle := dot_OA_OB / (magnitude_OA * magnitude_OB)

theorem problem_1 : vector_AB = (-4, 2) ∧ magnitude_AB = 2 * Real.sqrt 5 := sorry

theorem problem_2 : cosine_angle = Real.sqrt 5 / 5 := sorry

end problem_1_problem_2_l997_99751


namespace quadrilateral_area_is_33_l997_99744

-- Definitions for the points and their coordinates
structure Point where
  x : ℝ
  y : ℝ

def A : Point := {x := 4, y := 0}
def B : Point := {x := 0, y := 12}
def C : Point := {x := 10, y := 0}
def E : Point := {x := 3, y := 3}

-- Define the quadrilateral area computation
noncomputable def areaQuadrilateral (O B E C : Point) : ℝ :=
  let triangle_area (p1 p2 p3 : Point) :=
    abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y)) / 2
  triangle_area O B E + triangle_area O E C

-- Statement to prove
theorem quadrilateral_area_is_33 : areaQuadrilateral {x := 0, y := 0} B E C = 33 := by
  sorry

end quadrilateral_area_is_33_l997_99744


namespace base5_first_digit_of_1024_l997_99780

theorem base5_first_digit_of_1024: 
  ∀ (d : ℕ), (d * 5^4 ≤ 1024) ∧ (1024 < (d+1) * 5^4) → d = 1 :=
by
  sorry

end base5_first_digit_of_1024_l997_99780


namespace calculateRequiredMonthlyRent_l997_99797

noncomputable def requiredMonthlyRent (purchase_price : ℝ) (annual_return_rate : ℝ) (annual_taxes : ℝ) (repair_percentage : ℝ) : ℝ :=
  let annual_return := annual_return_rate * purchase_price
  let total_annual_need := annual_return + annual_taxes
  let monthly_requirement := total_annual_need / 12
  let monthly_rent := monthly_requirement / (1 - repair_percentage)
  monthly_rent

theorem calculateRequiredMonthlyRent : requiredMonthlyRent 20000 0.06 450 0.10 = 152.78 := by
  sorry

end calculateRequiredMonthlyRent_l997_99797


namespace complement_U_M_l997_99756

def U : Set ℕ := {x | x > 0 ∧ ∃ y : ℝ, y = Real.sqrt (5 - x)}
def M : Set ℕ := {x ∈ U | 4^x ≤ 16}

theorem complement_U_M : U \ M = {3, 4, 5} := by
  sorry

end complement_U_M_l997_99756


namespace tan_equality_condition_l997_99745

open Real

theorem tan_equality_condition (α β : ℝ) :
  (α = β) ↔ (tan α = tan β) :=
sorry

end tan_equality_condition_l997_99745


namespace correct_exponent_operation_l997_99713

theorem correct_exponent_operation (a : ℝ) : a^4 / a^3 = a := 
by
  sorry

end correct_exponent_operation_l997_99713


namespace smallest_X_l997_99743

noncomputable def T : ℕ := 1110
noncomputable def X : ℕ := T / 6

theorem smallest_X (hT_digits : (∀ d ∈ T.digits 10, d = 0 ∨ d = 1))
  (hT_positive : T > 0)
  (hT_div_6 : T % 6 = 0) :
  X = 185 := by
  sorry

end smallest_X_l997_99743


namespace exposed_circular_segment_sum_l997_99708

theorem exposed_circular_segment_sum (r h : ℕ) (angle : ℕ) (a b c : ℕ) :
    r = 8 ∧ h = 10 ∧ angle = 90 ∧ a = 16 ∧ b = 0 ∧ c = 0 → a + b + c = 16 :=
by
  intros
  sorry

end exposed_circular_segment_sum_l997_99708


namespace find_BD_l997_99798

theorem find_BD 
  (A B C D : Type)
  (AC BC : ℝ) (h₁ : AC = 10) (h₂ : BC = 10)
  (AD CD : ℝ) (h₃ : AD = 12) (h₄ : CD = 5) :
  ∃ (BD : ℝ), BD = 152 / 24 := 
sorry

end find_BD_l997_99798


namespace number_of_sheep_l997_99763

def ratio_sheep_horses (S H : ℕ) : Prop := S / H = 3 / 7
def horse_food_per_day := 230 -- ounces
def total_food_per_day := 12880 -- ounces

theorem number_of_sheep (S H : ℕ) 
  (h1 : ratio_sheep_horses S H) 
  (h2 : H * horse_food_per_day = total_food_per_day) 
  : S = 24 :=
sorry

end number_of_sheep_l997_99763


namespace weston_academy_geography_players_l997_99755

theorem weston_academy_geography_players
  (total_players : ℕ)
  (history_players : ℕ)
  (both_players : ℕ) :
  total_players = 18 →
  history_players = 10 →
  both_players = 6 →
  ∃ (geo_players : ℕ), geo_players = 14 := 
by 
  intros h1 h2 h3
  use 18 - (10 - 6) + 6
  sorry

end weston_academy_geography_players_l997_99755


namespace leak_empty_time_l997_99795

/-- 
The time taken for a leak to empty a full tank, given that an electric pump can fill a tank in 7 hours and it takes 14 hours to fill the tank with the leak present, is 14 hours.
 -/
theorem leak_empty_time (P L : ℝ) (hP : P = 1 / 7) (hCombined : P - L = 1 / 14) : L = 1 / 14 ∧ 1 / L = 14 :=
by
  sorry

end leak_empty_time_l997_99795


namespace average_speed_of_trip_l997_99761

theorem average_speed_of_trip :
  let total_distance := 50 -- in kilometers
  let distance1 := 25 -- in kilometers
  let speed1 := 66 -- in kilometers per hour
  let distance2 := 25 -- in kilometers
  let speed2 := 33 -- in kilometers per hour
  let time1 := distance1 / speed1 -- time taken for the first part
  let time2 := distance2 / speed2 -- time taken for the second part
  let total_time := time1 + time2 -- total time for the trip
  let average_speed := total_distance / total_time -- average speed of the trip
  average_speed = 44 := by
{
  sorry
}

end average_speed_of_trip_l997_99761


namespace projection_matrix_solution_l997_99747

theorem projection_matrix_solution 
  (a c : ℚ) 
  (P : Matrix (Fin 2) (Fin 2) ℚ := ![![a, 18/45], ![c, 27/45]])
  (hP : P * P = P) :
  (a, c) = (9/25, 12/25) :=
by
  sorry

end projection_matrix_solution_l997_99747


namespace other_root_of_quadratic_l997_99736

theorem other_root_of_quadratic (a b : ℝ) (h₀ : a ≠ 0) (h₁ : ∃ x : ℝ, (a * x ^ 2 = b) ∧ (x = 2)) : 
  ∃ m : ℝ, (a * m ^ 2 = b) ∧ (m = -2) := 
sorry

end other_root_of_quadratic_l997_99736


namespace smallest_multiple_of_45_and_75_not_20_l997_99784

-- Definitions of the conditions
def isMultipleOf (a b : ℕ) : Prop := ∃ k : ℕ, a = k * b
def notMultipleOf (a b : ℕ) : Prop := ¬ (isMultipleOf a b)

-- The proof statement
theorem smallest_multiple_of_45_and_75_not_20 :
  ∃ N : ℕ, isMultipleOf N 45 ∧ isMultipleOf N 75 ∧ notMultipleOf N 20 ∧ N = 225 :=
by
  -- sorry is used to indicate that the proof needs to be filled here
  sorry

end smallest_multiple_of_45_and_75_not_20_l997_99784


namespace complement_A_l997_99799

open Set

variable (A : Set ℝ) (x : ℝ)
def A_def : Set ℝ := { x | x ≥ 1 }

theorem complement_A : Aᶜ = { y | y < 1 } :=
by
  sorry

end complement_A_l997_99799


namespace marcy_multiple_tickets_l997_99732

theorem marcy_multiple_tickets (m : ℕ) : 
  (26 + (m * 26 - 6) = 150) → m = 5 :=
by
  intro h
  sorry

end marcy_multiple_tickets_l997_99732


namespace P_lt_Q_l997_99717

theorem P_lt_Q (x : ℝ) (hx : x > 0) (P : ℝ) (Q : ℝ) 
  (hP : P = Real.sqrt (1 + x)) 
  (hQ : Q = 1 + x / 2) : P < Q := 
by
  sorry

end P_lt_Q_l997_99717


namespace students_from_second_grade_l997_99757

theorem students_from_second_grade (r1 r2 r3 : ℕ) (total_students sample_size : ℕ) (h_ratio: r1 = 3 ∧ r2 = 3 ∧ r3 = 4 ∧ r1 + r2 + r3 = 10) (h_sample_size: sample_size = 50) : 
  (r2 * sample_size / (r1 + r2 + r3)) = 15 :=
by
  sorry

end students_from_second_grade_l997_99757


namespace gcd_50420_35313_l997_99740

theorem gcd_50420_35313 : Int.gcd 50420 35313 = 19 := 
sorry

end gcd_50420_35313_l997_99740


namespace probability_eight_distinct_numbers_l997_99786

theorem probability_eight_distinct_numbers :
  let total_ways := 10^8
  let ways_distinct := (10 * 9 * 8 * 7 * 6 * 5 * 4 * 3)
  (ways_distinct / total_ways : ℚ) = 18144 / 500000 := 
by
  sorry

end probability_eight_distinct_numbers_l997_99786


namespace unique_solution_l997_99796

noncomputable def f : ℝ → ℝ :=
sorry

theorem unique_solution (x : ℝ) (hx : 0 ≤ x) : 
  (f : ℝ → ℝ) (2 * x + 1) = 3 * (f x) + 5 ↔ f x = -5 / 2 :=
by 
  sorry

end unique_solution_l997_99796


namespace solution_set_f_div_x_lt_zero_l997_99718

noncomputable def f : ℝ → ℝ := sorry

theorem solution_set_f_div_x_lt_zero :
  (∀ x, f (2 + (2 - x)) = f x) ∧
  (∀ x1 x2 : ℝ, x1 < 2 ∧ x2 < 2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0) ∧
  f 4 = 0 →
  { x : ℝ | f x / x < 0 } = { x | x < 0 } ∪ { x | 0 < x ∧ x < 4 } :=
sorry

end solution_set_f_div_x_lt_zero_l997_99718


namespace third_competitor_hot_dogs_l997_99752

theorem third_competitor_hot_dogs (first_ate : ℕ) (second_multiplier : ℕ) (third_percent_less : ℕ) (second_ate third_ate : ℕ) 
  (H1 : first_ate = 12)
  (H2 : second_multiplier = 2)
  (H3 : third_percent_less = 25)
  (H4 : second_ate = first_ate * second_multiplier)
  (H5 : third_ate = second_ate - (third_percent_less * second_ate / 100)) : 
  third_ate = 18 := 
by 
  sorry

end third_competitor_hot_dogs_l997_99752


namespace gcd_pow_diff_l997_99748

theorem gcd_pow_diff (m n: ℤ) (H1: m = 2^2025 - 1) (H2: n = 2^2016 - 1) : Int.gcd m n = 511 := by
  sorry

end gcd_pow_diff_l997_99748


namespace part1_part2_l997_99765

-- Part 1
theorem part1 (x y : ℝ) : (2 * x - 3 * y) ^ 2 - (y + 3 * x) * (3 * x - y) = -5 * x ^ 2 - 12 * x * y + 10 * y ^ 2 := 
sorry

-- Part 2
theorem part2 : (2 + 1) * (2^2 + 1) * (2^4 + 1) * (2^8 + 1) - 2 ^ 16 = -1 := 
sorry

end part1_part2_l997_99765


namespace inequality_solution_set_l997_99704

def f (x : ℝ) : ℝ := x^3

theorem inequality_solution_set (x : ℝ) :
  (f (2 * x) + f (x - 1) < 0) ↔ (x < (1 / 3)) := 
sorry

end inequality_solution_set_l997_99704


namespace abs_value_identity_l997_99731

theorem abs_value_identity (a : ℝ) (h : a + |a| = 0) : a - |2 * a| = 3 * a :=
by
  sorry

end abs_value_identity_l997_99731


namespace part1_part2_l997_99767

-- Definition of the branches of the hyperbola
def C1 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1
def C2 (P : ℝ × ℝ) : Prop := P.1 * P.2 = 1

-- Problem Part 1: Proving that P, Q, and R cannot lie on the same branch
theorem part1 (P Q R : ℝ × ℝ) (hP : C1 P) (hQ : C1 Q) (hR : C1 R) : False := by
  sorry

-- Problem Part 2: Finding the coordinates of Q and R
theorem part2 : 
  ∃ Q R : ℝ × ℝ, C1 Q ∧ C1 R ∧ 
                (Q = (2 - Real.sqrt 3, 1 / (2 - Real.sqrt 3))) ∧ 
                (R = (2 + Real.sqrt 3, 1 / (2 + Real.sqrt 3))) := 
by
  sorry

end part1_part2_l997_99767


namespace binary_11011011_to_base4_is_3123_l997_99779

def binary_to_base4 (b : Nat) : Nat :=
  -- Function to convert binary number to base 4
  -- This will skip implementation details
  sorry

theorem binary_11011011_to_base4_is_3123 :
  binary_to_base4 0b11011011 = 0x3123 := 
sorry

end binary_11011011_to_base4_is_3123_l997_99779


namespace find_range_for_two_real_solutions_l997_99735

noncomputable def f (k x : ℝ) := k * x
noncomputable def g (x : ℝ) := (Real.log x) / x

noncomputable def h (x : ℝ) := (Real.log x) / (x^2)

theorem find_range_for_two_real_solutions :
  (∃ k : ℝ, ∀ x : ℝ, (1 / Real.exp 1) ≤ x ∧ x ≤ Real.exp 1 → (f k x = g x ↔ k ∈ Set.Icc (1 / Real.exp 2) (1 / (2 * Real.exp 1)))) :=
sorry

end find_range_for_two_real_solutions_l997_99735
