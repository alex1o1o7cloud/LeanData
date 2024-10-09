import Mathlib

namespace not_integer_fraction_l675_67574

theorem not_integer_fraction (a b : ℤ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) (hrelprime : Nat.gcd a.natAbs b.natAbs = 1) : 
  ¬(∃ (k : ℤ), 2 * a * (a^2 + b^2) = k * (a^2 - b^2)) :=
  sorry

end not_integer_fraction_l675_67574


namespace b_2016_eq_neg_4_l675_67538

def b : ℕ → ℤ
| 0     => 1
| 1     => 5
| (n+2) => b (n+1) - b n

theorem b_2016_eq_neg_4 : b 2015 = -4 :=
sorry

end b_2016_eq_neg_4_l675_67538


namespace P_zero_value_l675_67591

noncomputable def P (x b c : ℚ) : ℚ := x ^ 2 + b * x + c

theorem P_zero_value (b c : ℚ)
  (h1 : P (P 1 b c) b c = 0)
  (h2 : P (P (-2) b c) b c = 0)
  (h3 : P 1 b c ≠ P (-2) b c) :
  P 0 b c = -5 / 2 :=
sorry

end P_zero_value_l675_67591


namespace perfect_square_trinomial_implies_value_of_a_l675_67529

theorem perfect_square_trinomial_implies_value_of_a (a : ℝ) :
  (∃ (b : ℝ), (∃ (x : ℝ), (x^2 - ax + 9 = 0) ∧ (x + b)^2 = x^2 - ax + 9)) ↔ a = 6 ∨ a = -6 :=
by
  sorry

end perfect_square_trinomial_implies_value_of_a_l675_67529


namespace interest_rate_condition_l675_67558

theorem interest_rate_condition 
    (P1 P2 : ℝ) 
    (R2 : ℝ) 
    (T1 T2 : ℝ) 
    (SI500 SI160 : ℝ) 
    (H1: SI500 = (P1 * R2 * T1) / 100) 
    (H2: SI160 = (P2 * (25 / 100))):
  25 * (160 / 100) / 12.5  = 6.4 :=
by
  sorry

end interest_rate_condition_l675_67558


namespace shared_friends_l675_67514

theorem shared_friends (crackers total_friends : ℕ) (each_friend_crackers : ℕ) 
  (h1 : crackers = 22) 
  (h2 : each_friend_crackers = 2)
  (h3 : crackers = each_friend_crackers * total_friends) 
  : total_friends = 11 := by 
  sorry

end shared_friends_l675_67514


namespace adams_father_total_amount_l675_67576

noncomputable def annual_interest (principal : ℝ) (rate : ℝ) : ℝ :=
  principal * rate

noncomputable def total_interest (annual_interest : ℝ) (years : ℝ) : ℝ :=
  annual_interest * years

noncomputable def total_amount (principal : ℝ) (total_interest : ℝ) : ℝ :=
  principal + total_interest

theorem adams_father_total_amount :
  let principal := 2000
  let rate := 0.08
  let years := 2.5
  let annualInterest := annual_interest principal rate
  let interest := total_interest annualInterest years
  let amount := total_amount principal interest
  amount = 2400 :=
by sorry

end adams_father_total_amount_l675_67576


namespace average_weight_increase_per_month_l675_67550

theorem average_weight_increase_per_month (w_initial w_final : ℝ) (t : ℝ) 
  (h_initial : w_initial = 3.25) (h_final : w_final = 7) (h_time : t = 3) :
  (w_final - w_initial) / t = 1.25 := 
by 
  sorry

end average_weight_increase_per_month_l675_67550


namespace tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l675_67521

variable (α : ℝ)
variable (h : (2 * Real.sin α + 3 * Real.cos α) / (Real.sin α - 2 * Real.cos α) = 1 / 4)

theorem tan_alpha_eq_neg2 : Real.tan α = -2 :=
  sorry

theorem sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2 :
  (Real.sin (2 * α) + 1) / (1 + Real.sin (2 * α) + Real.cos (2 * α)) = -1 / 2 :=
  sorry

end tan_alpha_eq_neg2_sin2a_plus_1_over_1_plus_sin2a_plus_cos2a_eq_neg1_over_2_l675_67521


namespace base_conversion_least_sum_l675_67551

theorem base_conversion_least_sum :
  ∃ (c d : ℕ), (5 * c + 8 = 8 * d + 5) ∧ c > 0 ∧ d > 0 ∧ (c + d = 15) := by
sorry

end base_conversion_least_sum_l675_67551


namespace total_eggs_collected_l675_67502

def benjamin_collects : Nat := 6
def carla_collects := 3 * benjamin_collects
def trisha_collects := benjamin_collects - 4

theorem total_eggs_collected :
  benjamin_collects + carla_collects + trisha_collects = 26 := by
  sorry

end total_eggs_collected_l675_67502


namespace intersection_of_A_and_B_l675_67582

-- Definitions of sets A and B based on the conditions
def A : Set ℝ := {x | 0 < x}
def B : Set ℝ := {0, 1, 2}

-- Theorem statement to prove A ∩ B = {1, 2}
theorem intersection_of_A_and_B : A ∩ B = {1, 2} := 
  sorry

end intersection_of_A_and_B_l675_67582


namespace sum_of_roots_of_cis_equation_l675_67532

theorem sum_of_roots_of_cis_equation 
  (cis : ℝ → ℂ)
  (phi : ℕ → ℝ)
  (h_conditions : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → 0 ≤ phi k ∧ phi k < 360)
  (h_equation : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 5 → (cis (phi k)) ^ 5 = (1 / Real.sqrt 2) + (Complex.I / Real.sqrt 2))
  : (phi 1 + phi 2 + phi 3 + phi 4 + phi 5) = 450 :=
by
  sorry

end sum_of_roots_of_cis_equation_l675_67532


namespace find_base_k_l675_67568

theorem find_base_k : ∃ k : ℕ, 6 * k^2 + 6 * k + 4 = 340 ∧ k = 7 := 
by 
  sorry

end find_base_k_l675_67568


namespace pages_left_to_read_l675_67520

theorem pages_left_to_read (total_pages : ℕ) (pages_read : ℕ) (pages_skipped : ℕ) : 
  total_pages = 372 → pages_read = 125 → pages_skipped = 16 → (total_pages - (pages_read + pages_skipped)) = 231 :=
by
  intros
  sorry

end pages_left_to_read_l675_67520


namespace sam_annual_income_l675_67548

theorem sam_annual_income
  (q : ℝ) (I : ℝ)
  (h1 : 30000 * 0.01 * q + 15000 * 0.01 * (q + 3) + (I - 45000) * 0.01 * (q + 5) = (q + 0.35) * 0.01 * I) :
  I = 48376 := 
sorry

end sam_annual_income_l675_67548


namespace total_cost_l675_67534

theorem total_cost (cost_pencil cost_pen : ℕ) 
(h1 : cost_pen = cost_pencil + 9) 
(h2 : cost_pencil = 2) : 
cost_pencil + cost_pen = 13 := 
by 
  -- Proof would go here 
  sorry

end total_cost_l675_67534


namespace fill_tank_with_leak_l675_67523

namespace TankFilling

-- Conditions
def pump_fill_rate (P : ℝ) : Prop := P = 1 / 4
def leak_drain_rate (L : ℝ) : Prop := L = 1 / 5
def net_fill_rate (P L R : ℝ) : Prop := P - L = R
def fill_time (R T : ℝ) : Prop := T = 1 / R

-- Statement
theorem fill_tank_with_leak (P L R T : ℝ) (hP : pump_fill_rate P) (hL : leak_drain_rate L) (hR : net_fill_rate P L R) (hT : fill_time R T) :
  T = 20 :=
  sorry

end TankFilling

end fill_tank_with_leak_l675_67523


namespace find_m_for_parallel_lines_l675_67543

-- The given lines l1 and l2
def line1 (m: ℝ) : Prop := ∀ x y : ℝ, (3 + m) * x - 4 * y = 5 - 3 * m
def line2 : Prop := ∀ x y : ℝ, 2 * x - y = 8

-- Definition for parallel lines
def parallel_lines (l₁ l₂ : Prop) : Prop := 
  ∃ m : ℝ, (3 + m) / 4 = 2

-- The main theorem to prove
theorem find_m_for_parallel_lines (m: ℝ) (h: parallel_lines (line1 m) line2) : m = 5 :=
by sorry

end find_m_for_parallel_lines_l675_67543


namespace xiaoma_miscalculation_l675_67501

theorem xiaoma_miscalculation (x : ℤ) (h : 40 + x = 35) : 40 / x = -8 := by
  sorry

end xiaoma_miscalculation_l675_67501


namespace valentines_distribution_l675_67535

theorem valentines_distribution (valentines_initial : ℝ) (valentines_needed : ℝ) (students : ℕ) 
  (h_initial : valentines_initial = 58.0) (h_needed : valentines_needed = 16.0) (h_students : students = 74) : 
  (valentines_initial + valentines_needed) / students = 1 :=
by
  sorry

end valentines_distribution_l675_67535


namespace golf_problem_l675_67547

variable (D : ℝ)

theorem golf_problem (h1 : D / 2 + D = 270) : D = 180 :=
by
  sorry

end golf_problem_l675_67547


namespace find_divisors_l675_67508

theorem find_divisors (N : ℕ) :
  (∃ k : ℕ, 2014 = k * (N + 1) ∧ k < N) ↔ (N = 2013 ∨ N = 1006 ∨ N = 105 ∨ N = 52) := by
  sorry

end find_divisors_l675_67508


namespace initial_num_files_l675_67544

-- Define the conditions: number of files organized in the morning, files to organize in the afternoon, and missing files.
def num_files_organized_in_morning (X : ℕ) : ℕ := X / 2
def num_files_to_organize_in_afternoon : ℕ := 15
def num_files_missing : ℕ := 15

-- Theorem to prove the initial number of files is 60.
theorem initial_num_files (X : ℕ) 
  (h1 : num_files_organized_in_morning X = X / 2)
  (h2 : num_files_to_organize_in_afternoon = 15)
  (h3 : num_files_missing = 15) :
  X = 60 :=
by
  sorry

end initial_num_files_l675_67544


namespace arithmetic_sequence_sum_l675_67562

theorem arithmetic_sequence_sum : 
  ∀ (a : ℕ → ℝ) (d : ℝ), (a 1 = 2 ∨ a 1 = 8) → (a 2017 = 2 ∨ a 2017 = 8) → 
  (∀ n : ℕ, a (n + 1) = a n + d) →
  a 2 + a 1009 + a 2016 = 15 := 
by
  intro a d h1 h2017 ha
  sorry

end arithmetic_sequence_sum_l675_67562


namespace limonia_largest_none_providable_amount_l675_67583

def is_achievable (n : ℕ) (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), x = a * (6 * n + 1) + b * (6 * n + 4) + c * (6 * n + 7) + d * (6 * n + 10)

theorem limonia_largest_none_providable_amount (n : ℕ) : 
  ∃ s, ¬ is_achievable n s ∧ (∀ t, t > s → is_achievable n t) ∧ s = 12 * n^2 + 14 * n - 1 :=
by
  sorry

end limonia_largest_none_providable_amount_l675_67583


namespace reducible_iff_form_l675_67519

def isReducible (a : ℕ) : Prop :=
  ∃ d : ℕ, d ≠ 1 ∧ d ∣ (2 * a + 5) ∧ d ∣ (3 * a + 4)

theorem reducible_iff_form (a : ℕ) : isReducible a ↔ ∃ k : ℕ, a = 7 * k + 1 := by
  sorry

end reducible_iff_form_l675_67519


namespace part1_part2_l675_67593

theorem part1 : (2 / 9 - 1 / 6 + 1 / 18) * (-18) = -2 := 
by
  sorry

theorem part2 : 54 * (3 / 4 + 1 / 2 - 1 / 4) = 54 := 
by
  sorry

end part1_part2_l675_67593


namespace B_pow_16_eq_I_l675_67567

noncomputable def B : Matrix (Fin 4) (Fin 4) ℝ := 
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4), 0 , 0],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4), 0 , 0],
    ![0, 0, Real.cos (Real.pi / 4), Real.sin (Real.pi / 4)],
    ![0, 0, -Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem B_pow_16_eq_I : B^16 = 1 := by
  sorry

end B_pow_16_eq_I_l675_67567


namespace five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l675_67503

theorem five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand :
  5.8 / 0.001 = 5.8 * 1000 :=
by
  -- This is where the proof would go
  sorry

end five_point_eight_divide_by_point_zero_zero_one_eq_five_point_eight_multiply_by_thousand_l675_67503


namespace max_distinct_colorings_5x5_l675_67585

theorem max_distinct_colorings_5x5 (n : ℕ) :
  ∃ N, N ≤ (n^25 + 4 * n^15 + n^13 + 2 * n^7) / 8 :=
sorry

end max_distinct_colorings_5x5_l675_67585


namespace find_range_a_l675_67522

-- Define the proposition p
def p (m : ℝ) : Prop :=
1 < m ∧ m < 3 / 2

-- Define the proposition q
def q (m a : ℝ) : Prop :=
(m - a) * (m - (a + 1)) < 0

-- Define the sufficient but not necessary condition
def sufficient (a : ℝ) : Prop :=
(a ≤ 1) ∧ (3 / 2 ≤ a + 1)

theorem find_range_a (a : ℝ) :
  (∀ m, p m → q m a) → sufficient a → (1 / 2 ≤ a ∧ a ≤ 1) :=
sorry

end find_range_a_l675_67522


namespace gumballs_initial_count_l675_67553

noncomputable def initial_gumballs := (34.3 / (0.7 ^ 3))

theorem gumballs_initial_count :
  initial_gumballs = 100 :=
sorry

end gumballs_initial_count_l675_67553


namespace domain_of_f_l675_67512

-- Define the conditions
def sqrt_domain (x : ℝ) : Prop := x + 1 ≥ 0
def log_domain (x : ℝ) : Prop := 3 - x > 0

-- Define the function
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 1) + Real.log (3 - x)

-- Statement of the theorem
theorem domain_of_f : ∀ x, sqrt_domain x ∧ log_domain x ↔ -1 ≤ x ∧ x < 3 := by
  sorry

end domain_of_f_l675_67512


namespace product_of_solutions_l675_67528

theorem product_of_solutions : 
  ∀ x₁ x₂ : ℝ, (|6 * x₁| + 5 = 47) ∧ (|6 * x₂| + 5 = 47) → x₁ * x₂ = -49 :=
by
  sorry

end product_of_solutions_l675_67528


namespace inequality_proof_l675_67545

theorem inequality_proof
  (a b c d : ℝ)
  (ha : abs a > 1)
  (hb : abs b > 1)
  (hc : abs c > 1)
  (hd : abs d > 1)
  (h : a * b * c + a * b * d + a * c * d + b * c * d + a + b + c + d = 0) :
  1 / (a - 1) + 1 / (b - 1) + 1 / (c - 1) + 1 / (d - 1) > 0 :=
sorry

end inequality_proof_l675_67545


namespace minimum_value_of_quadratic_l675_67588

theorem minimum_value_of_quadratic (p q : ℝ) (hp : 0 < p) (hq : 0 < q) : 
  ∃ x : ℝ, x = -p / 2 ∧ (∀ y : ℝ, (y - x) ^ 2 + 2*q ≥ (x ^ 2 + p * x + 2*q)) :=
by
  sorry

end minimum_value_of_quadratic_l675_67588


namespace count_negative_rationals_is_two_l675_67549

theorem count_negative_rationals_is_two :
  let a := (-1 : ℚ) ^ 2007
  let b := (|(-1 : ℚ)| ^ 3)
  let c := -(1 : ℚ) ^ 18
  let d := (18 : ℚ)
  (if a < 0 then 1 else 0) + (if b < 0 then 1 else 0) + (if c < 0 then 1 else 0) + (if d < 0 then 1 else 0) = 2 := by
  sorry

end count_negative_rationals_is_two_l675_67549


namespace no_real_solutions_for_g_g_x_l675_67565

theorem no_real_solutions_for_g_g_x (d : ℝ) :
  ¬ ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1^2 + 4 * x1 + d)^2 + 4 * (x1^2 + 4 * x1 + d) + d = 0 ∧
                                (x2^2 + 4 * x2 + d)^2 + 4 * (x2^2 + 4 * x2 + d) + d = 0 :=
by
  sorry

end no_real_solutions_for_g_g_x_l675_67565


namespace find_function_solution_l675_67539

def satisfies_condition (f : ℝ → ℝ) :=
  ∀ (x y : ℝ), f (f (x * y)) = |x| * f y + 3 * f (x * y)

theorem find_function_solution (f : ℝ → ℝ) :
  satisfies_condition f → (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 4 * |x|) ∨ (∀ x : ℝ, f x = -4 * |x|) :=
by
  sorry

end find_function_solution_l675_67539


namespace decreases_as_x_increases_graph_passes_through_origin_l675_67527

-- Proof Problem 1: Show that y decreases as x increases if and only if k > 2
theorem decreases_as_x_increases (k : ℝ) : (∀ x1 x2 : ℝ, (x1 < x2) → ((2 - k) * x1 - k^2 + 4) > ((2 - k) * x2 - k^2 + 4)) ↔ (k > 2) := 
  sorry

-- Proof Problem 2: Show that the graph passes through the origin if and only if k = -2
theorem graph_passes_through_origin (k : ℝ) : ((2 - k) * 0 - k^2 + 4 = 0) ↔ (k = -2) :=
  sorry

end decreases_as_x_increases_graph_passes_through_origin_l675_67527


namespace two_distinct_real_roots_of_modified_quadratic_l675_67561

theorem two_distinct_real_roots_of_modified_quadratic (a b k : ℝ) (h1 : a^2 - b > 0) (h2 : k > 0) :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 + 2 * a * x₁ + b + k * (x₁ + a)^2 = 0) ∧ (x₂^2 + 2 * a * x₂ + b + k * (x₂ + a)^2 = 0) :=
by
  sorry

end two_distinct_real_roots_of_modified_quadratic_l675_67561


namespace designer_suit_size_l675_67573

theorem designer_suit_size : ∀ (waist_in_inches : ℕ) (comfort_in_inches : ℕ) 
  (inches_per_foot : ℕ) (cm_per_foot : ℝ), 
  waist_in_inches = 34 →
  comfort_in_inches = 2 →
  inches_per_foot = 12 →
  cm_per_foot = 30.48 →
  (((waist_in_inches + comfort_in_inches) / inches_per_foot : ℝ) * cm_per_foot) = 91.4 :=
by
  intros waist_in_inches comfort_in_inches inches_per_foot cm_per_foot
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  norm_cast
  norm_num
  sorry

end designer_suit_size_l675_67573


namespace john_tips_problem_l675_67537

theorem john_tips_problem
  (A M : ℝ)
  (H1 : ∀ (A : ℝ), M * A = 0.5 * (6 * A + M * A)) :
  M = 6 := 
by
  sorry

end john_tips_problem_l675_67537


namespace pencils_count_l675_67597

theorem pencils_count (initial_pencils additional_pencils : ℕ) (h1 : initial_pencils = 115) (h2 : additional_pencils = 100) : initial_pencils + additional_pencils = 215 :=
by sorry

end pencils_count_l675_67597


namespace exists_pairs_angle_120_degrees_l675_67500

theorem exists_pairs_angle_120_degrees :
  ∃ a b : ℤ, a + b ≠ 0 ∧ a + b ≠ a ^ 2 - a * b + b ^ 2 ∧ (a + b) * 13 = 3 * (a ^ 2 - a * b + b ^ 2) :=
sorry

end exists_pairs_angle_120_degrees_l675_67500


namespace intersection_A_B_l675_67530

-- Define sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x < 2}
def B : Set ℝ := {-3, -2, -1, 0, 1, 2}

-- Define the intersection we need to prove
def A_cap_B_target : Set ℝ := {-2, -1, 0, 1}

-- Prove the intersection of A and B equals the target set
theorem intersection_A_B :
  A ∩ B = A_cap_B_target := 
sorry

end intersection_A_B_l675_67530


namespace strips_area_coverage_l675_67559

-- Define paper strips and their properties
def length_strip : ℕ := 8
def width_strip : ℕ := 2
def number_of_strips : ℕ := 5

-- Total area without considering overlaps
def area_one_strip : ℕ := length_strip * width_strip
def total_area_without_overlap : ℕ := number_of_strips * area_one_strip

-- Overlapping areas
def area_center_overlap : ℕ := 4 * (2 * 2)
def area_additional_overlap : ℕ := 2 * (2 * 2)
def total_overlap_area : ℕ := area_center_overlap + area_additional_overlap

-- Actual area covered
def actual_area_covered : ℕ := total_area_without_overlap - total_overlap_area

-- Theorem stating the required proof
theorem strips_area_coverage : actual_area_covered = 56 :=
by sorry

end strips_area_coverage_l675_67559


namespace expression_evaluation_l675_67587

def a : ℚ := 8 / 9
def b : ℚ := 5 / 6
def c : ℚ := 2 / 3
def d : ℚ := -5 / 18
def lhs : ℚ := (a - b + c) / d
def rhs : ℚ := -13 / 5

theorem expression_evaluation : lhs = rhs := by
  sorry

end expression_evaluation_l675_67587


namespace sum_of_first_6n_integers_l675_67595

theorem sum_of_first_6n_integers (n : ℕ) (h1 : (5 * n * (5 * n + 1)) / 2 = (n * (n + 1)) / 2 + 200) :
  (6 * n * (6 * n + 1)) / 2 = 300 :=
by
  sorry

end sum_of_first_6n_integers_l675_67595


namespace condo_total_units_l675_67598

-- Definitions from conditions
def total_floors := 23
def regular_units_per_floor := 12
def penthouse_units_per_floor := 2
def penthouse_floors := 2
def regular_floors := total_floors - penthouse_floors

-- Definition for total units
def total_units := (regular_floors * regular_units_per_floor) + (penthouse_floors * penthouse_units_per_floor)

-- Theorem statement: prove total units is 256
theorem condo_total_units : total_units = 256 :=
by
  sorry

end condo_total_units_l675_67598


namespace kolya_sheets_exceed_500_l675_67540

theorem kolya_sheets_exceed_500 :
  ∃ k : ℕ, (10 + k * (k + 1) / 2 > 500) :=
sorry

end kolya_sheets_exceed_500_l675_67540


namespace area_covered_three_layers_l675_67525

noncomputable def auditorium_width : ℕ := 10
noncomputable def auditorium_height : ℕ := 10

noncomputable def first_rug_width : ℕ := 6
noncomputable def first_rug_height : ℕ := 8
noncomputable def second_rug_width : ℕ := 6
noncomputable def second_rug_height : ℕ := 6
noncomputable def third_rug_width : ℕ := 5
noncomputable def third_rug_height : ℕ := 7

-- Prove that the area of part of the auditorium covered with rugs in three layers is 6 square meters.
theorem area_covered_three_layers : 
  let horizontal_overlap_second_third := 5
  let vertical_overlap_second_third := 3
  let area_overlap_second_third := horizontal_overlap_second_third * vertical_overlap_second_third
  let horizontal_overlap_all := 3
  let vertical_overlap_all := 2
  let area_overlap_all := horizontal_overlap_all * vertical_overlap_all
  area_overlap_all = 6 := 
by
  sorry

end area_covered_three_layers_l675_67525


namespace appropriate_sampling_method_l675_67517

def total_families := 500
def high_income_families := 125
def middle_income_families := 280
def low_income_families := 95
def sample_size := 100
def influenced_by_income := True

theorem appropriate_sampling_method
  (htotal : total_families = 500)
  (hhigh : high_income_families = 125)
  (hmiddle : middle_income_families = 280)
  (hlow : low_income_families = 95)
  (hsample : sample_size = 100)
  (hinfluence : influenced_by_income = True) :
  ∃ method, method = "Stratified sampling method" :=
sorry

end appropriate_sampling_method_l675_67517


namespace minimum_boxes_needed_l675_67590

theorem minimum_boxes_needed (small_box_capacity medium_box_capacity large_box_capacity : ℕ)
    (max_small_boxes max_medium_boxes max_large_boxes : ℕ)
    (total_dozens: ℕ) :
  small_box_capacity = 2 → 
  medium_box_capacity = 3 → 
  large_box_capacity = 4 → 
  max_small_boxes = 6 → 
  max_medium_boxes = 5 → 
  max_large_boxes = 4 → 
  total_dozens = 40 → 
  ∃ (small_boxes_needed medium_boxes_needed large_boxes_needed : ℕ), 
    small_boxes_needed = 5 ∧ 
    medium_boxes_needed = 5 ∧ 
    large_boxes_needed = 4 := 
by
  sorry

end minimum_boxes_needed_l675_67590


namespace smallest_z_value_l675_67552

theorem smallest_z_value :
  ∃ (w x y z : ℕ), w < x ∧ x < y ∧ y < z ∧
  w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧
  w^3 + x^3 + y^3 = z^3 ∧ z = 6 := by
  sorry

end smallest_z_value_l675_67552


namespace market_value_correct_l675_67556

noncomputable def face_value : ℝ := 100
noncomputable def dividend_per_share : ℝ := 0.14 * face_value
noncomputable def yield : ℝ := 0.08

theorem market_value_correct :
  (dividend_per_share / yield) * 100 = 175 := by
  sorry

end market_value_correct_l675_67556


namespace eq_has_positive_integer_solution_l675_67531

theorem eq_has_positive_integer_solution (a : ℤ) :
  (∃ x : ℕ+, (x : ℤ) - 4 - 2 * (a * x - 1) = 2) → a = 0 :=
by
  sorry

end eq_has_positive_integer_solution_l675_67531


namespace semiperimeter_inequality_l675_67569

theorem semiperimeter_inequality (p R r : ℝ) (hp : p ≥ 0) (hR : R ≥ 0) (hr : r ≥ 0) :
  p ≥ (3 / 2) * Real.sqrt (6 * R * r) :=
sorry

end semiperimeter_inequality_l675_67569


namespace minimum_number_of_tiles_l675_67557

def tile_width_in_inches : ℕ := 6
def tile_height_in_inches : ℕ := 4
def region_width_in_feet : ℕ := 3
def region_height_in_feet : ℕ := 8

def inches_to_feet (i : ℕ) : ℚ :=
  i / 12

def tile_width_in_feet : ℚ :=
  inches_to_feet tile_width_in_inches

def tile_height_in_feet : ℚ :=
  inches_to_feet tile_height_in_inches

def tile_area_in_square_feet : ℚ :=
  tile_width_in_feet * tile_height_in_feet

def region_area_in_square_feet : ℚ :=
  region_width_in_feet * region_height_in_feet

def number_of_tiles : ℚ :=
  region_area_in_square_feet / tile_area_in_square_feet

theorem minimum_number_of_tiles :
  number_of_tiles = 144 := by
    sorry

end minimum_number_of_tiles_l675_67557


namespace james_fence_problem_l675_67560

theorem james_fence_problem (w : ℝ) (hw : 0 ≤ w) (h_area : w * (2 * w + 10) ≥ 120) : w = 5 :=
by
  sorry

end james_fence_problem_l675_67560


namespace cylindrical_to_rectangular_l675_67505

theorem cylindrical_to_rectangular :
  ∀ (r θ z : ℝ), r = 5 → θ = (3 * Real.pi) / 4 → z = 2 →
    (r * Real.cos θ, r * Real.sin θ, z) = (-5 * Real.sqrt 2 / 2, 5 * Real.sqrt 2 / 2, 2) :=
by
  intros r θ z hr hθ hz
  rw [hr, hθ, hz]
  -- Proof steps would go here, but are omitted as they are not required.
  sorry

end cylindrical_to_rectangular_l675_67505


namespace inequality_comparison_l675_67533

theorem inequality_comparison (x y : ℝ) (h : x ≠ y) : x^4 + y^4 > x^3 * y + x * y^3 :=
  sorry

end inequality_comparison_l675_67533


namespace probability_of_square_product_l675_67510

theorem probability_of_square_product :
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9 -- (1,1), (1,4), (2,2), (4,1), (3,3), (9,1), (4,4), (5,5), (6,6)
  favorable_outcomes / total_outcomes = 1 / 8 :=
by
  let num_tiles := 12
  let num_faces := 6
  let total_outcomes := num_tiles * num_faces
  let favorable_outcomes := 9
  have h1 : favorable_outcomes / total_outcomes = 1 / 8 := sorry
  exact h1

end probability_of_square_product_l675_67510


namespace cakes_served_today_l675_67572

def lunch_cakes := 6
def dinner_cakes := 9
def total_cakes := lunch_cakes + dinner_cakes

theorem cakes_served_today : total_cakes = 15 := by
  sorry

end cakes_served_today_l675_67572


namespace parabola_min_y1_y2_squared_l675_67599

theorem parabola_min_y1_y2_squared (x1 x2 y1 y2 : ℝ) :
  (y1^2 = 4 * x1) ∧
  (y2^2 = 4 * x2) ∧
  (x1 * x2 = 16) →
  (y1^2 + y2^2 ≥ 32) :=
by
  intro h
  sorry

end parabola_min_y1_y2_squared_l675_67599


namespace change_in_expression_l675_67507

theorem change_in_expression (x b : ℝ) (hb : 0 < b) : 
    (2 * (x + b) ^ 2 + 5 - (2 * x ^ 2 + 5) = 4 * x * b + 2 * b ^ 2) ∨ 
    (2 * (x - b) ^ 2 + 5 - (2 * x ^ 2 + 5) = -4 * x * b + 2 * b ^ 2) := 
by
    sorry

end change_in_expression_l675_67507


namespace certain_number_is_two_l675_67578

variable (x : ℕ)  -- x is the certain number

-- Condition: Given that adding 6 incorrectly results in 8
axiom h1 : x + 6 = 8

-- The mathematically equivalent proof problem Lean statement
theorem certain_number_is_two : x = 2 :=
by
  sorry

end certain_number_is_two_l675_67578


namespace cost_of_fencing_each_side_l675_67513

theorem cost_of_fencing_each_side (total_cost : ℕ) (num_sides : ℕ) (h1 : total_cost = 288) (h2 : num_sides = 4) : (total_cost / num_sides) = 72 := by
  sorry

end cost_of_fencing_each_side_l675_67513


namespace tom_made_washing_cars_l675_67504

-- Definitions of the conditions
def initial_amount : ℕ := 74
def final_amount : ℕ := 86

-- Statement to be proved
theorem tom_made_washing_cars : final_amount - initial_amount = 12 := by
  sorry

end tom_made_washing_cars_l675_67504


namespace remove_terms_sum_equals_one_l675_67592

theorem remove_terms_sum_equals_one :
  let seq := [1/3, 1/6, 1/9, 1/12, 1/15, 1/18]
  let remove := [1/12, 1/15]
  (seq.sum - remove.sum) = 1 :=
by
  sorry

end remove_terms_sum_equals_one_l675_67592


namespace lines_positional_relationship_l675_67566

-- Defining basic geometric entities and their properties
structure Line :=
  (a b : ℝ)
  (point_on_line : ∃ x, a * x + b = 0)

-- Defining skew lines (two lines that do not intersect and are not parallel)
def skew_lines (l1 l2 : Line) : Prop :=
  ¬(∀ x, l1.a * x + l1.b = l2.a * x + l2.b) ∧ ¬(l1.a = l2.a)

-- Defining intersecting lines
def intersect (l1 l2 : Line) : Prop :=
  ∃ x, l1.a * x + l1.b = l2.a * x + l2.b

-- Main theorem to prove
theorem lines_positional_relationship (l1 l2 k m : Line) 
  (hl1: intersect l1 k) (hl2: intersect l2 k) (hk: skew_lines l1 m) (hm: skew_lines l2 m) :
  (intersect l1 l2) ∨ (skew_lines l1 l2) :=
sorry

end lines_positional_relationship_l675_67566


namespace compute_expression_l675_67581

-- Define the conditions as specific values and operations within the theorem itself
theorem compute_expression : 5 + 7 * (2 - 9)^2 = 348 := 
  by
  sorry

end compute_expression_l675_67581


namespace average_of_remaining_two_numbers_l675_67518

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
(h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
(h_avg_2_1 : (a + b) / 2 = 3.4)
(h_avg_2_2 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 := 
sorry

end average_of_remaining_two_numbers_l675_67518


namespace p_implies_q_and_not_converse_l675_67554

def p (a : ℝ) := a ≤ 1
def q (a : ℝ) := abs a ≤ 1

theorem p_implies_q_and_not_converse (a : ℝ) : (p a → q a) ∧ ¬(q a → p a) :=
by
  repeat { sorry }

end p_implies_q_and_not_converse_l675_67554


namespace solve_inequality_l675_67546

noncomputable def solution_set : Set ℝ := {x | x < -4/3 ∨ x > -13/9}

theorem solve_inequality (x : ℝ) : 
  2 - 1 / (3 * x + 4) < 5 → x ∈ solution_set :=
by
  sorry

end solve_inequality_l675_67546


namespace prove_f_3_equals_11_l675_67596

-- Assuming the given function definition as condition
def f (y : ℝ) : ℝ := sorry

-- The condition provided: f(x - 1/x) = x^2 + 1/x^2.
axiom function_definition (x : ℝ) (h : x ≠ 0): f (x - 1 / x) = x^2 + 1 / x^2

-- The goal is to prove that f(3) = 11
theorem prove_f_3_equals_11 : f 3 = 11 :=
by
  sorry

end prove_f_3_equals_11_l675_67596


namespace min_max_a_e_l675_67555

noncomputable def find_smallest_largest (a b c d e : ℝ) : ℝ × ℝ :=
  if a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e
    then (a, e)
    else (-1, -1) -- using -1 to indicate invalid input

theorem min_max_a_e (a b c d e : ℝ) : a + b < c + d ∧ c + d < e + a ∧ e + a < b + c ∧ b + c < d + e → 
    find_smallest_largest a b c d e = (a, e) :=
  by
    -- Proof to be filled in by user
    sorry

end min_max_a_e_l675_67555


namespace f_of_3_l675_67580

theorem f_of_3 (f : ℕ → ℕ) (h : ∀ x, f (x + 1) = 2 * x + 3) : f 3 = 7 := 
sorry

end f_of_3_l675_67580


namespace product_of_primes_l675_67516

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

noncomputable def smallest_one_digit_primes (p₁ p₂ : ℕ) : Prop :=
  is_prime p₁ ∧ is_prime p₂ ∧ p₁ < p₂ ∧ p₂ < 10 ∧ ∀ p : ℕ, is_prime p → p < 10 → p = p₁ ∨ p = p₂

noncomputable def smallest_two_digit_prime (p : ℕ) : Prop :=
  is_prime p ∧ p ≥ 10 ∧ p < 100 ∧ ∀ q : ℕ, is_prime q → q ≥ 10 → q < p → q = 11

theorem product_of_primes : ∃ p₁ p₂ p₃ : ℕ, smallest_one_digit_primes p₁ p₂ ∧ smallest_two_digit_prime p₃ ∧ p₁ * p₂ * p₃ = 66 := 
by
  sorry

end product_of_primes_l675_67516


namespace faye_age_l675_67589
open Nat

theorem faye_age :
  ∃ (C D E F : ℕ), 
    (D = E - 3) ∧ 
    (E = C + 4) ∧ 
    (F = C + 3) ∧ 
    (D = 14) ∧ 
    (F = 16) :=
by
  sorry

end faye_age_l675_67589


namespace probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l675_67524

noncomputable def total_cassettes : ℕ := 30
noncomputable def disco_cassettes : ℕ := 12
noncomputable def classical_cassettes : ℕ := 18

-- Part (a): DJ returns the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_returned :
  (disco_cassettes / total_cassettes) * (disco_cassettes / total_cassettes) = 4 / 25 :=
by
  sorry

-- Part (b): DJ does not return the first cassette before taking the second one
theorem probability_disco_music_two_cassettes_not_returned :
  (disco_cassettes / total_cassettes) * ((disco_cassettes - 1) / (total_cassettes - 1)) = 22 / 145 :=
by
  sorry

end probability_disco_music_two_cassettes_returned_probability_disco_music_two_cassettes_not_returned_l675_67524


namespace malcolm_needs_more_lights_l675_67511

def red_lights := 12
def blue_lights := 3 * red_lights
def green_lights := 6
def white_lights := 59

def colored_lights := red_lights + blue_lights + green_lights
def need_more_lights := white_lights - colored_lights

theorem malcolm_needs_more_lights :
  need_more_lights = 5 :=
by
  sorry

end malcolm_needs_more_lights_l675_67511


namespace petya_time_comparison_l675_67571

theorem petya_time_comparison (V : ℝ) (a : ℝ) (hV_pos : 0 < V) (ha_pos : 0 < a) :
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  T_real > T_planned :=
by
  let T_planned := a / V
  let T_real := (a / (1.25 * V) / 2) + (a / (0.8 * V) / 2) 
  have T_real_eq : T_real = (a / (1.25 * V)) + (a / (0.8 * V)) := by sorry
  have T_real_simplified : T_real = (2 * a / 5 / V) + (5 * a / 8 / V) := by sorry
  have T_real_combined : T_real = (41 * a) / (40 * V) := by sorry
  have planned_vs_real : T_real > a / V := by sorry
  exact planned_vs_real

end petya_time_comparison_l675_67571


namespace value_decrease_proof_l675_67526

noncomputable def value_comparison (diana_usd : ℝ) (etienne_eur : ℝ) (eur_to_usd : ℝ) : ℝ :=
  let etienne_usd := etienne_eur * eur_to_usd
  let percentage_decrease := ((diana_usd - etienne_usd) / diana_usd) * 100
  percentage_decrease

theorem value_decrease_proof :
  value_comparison 700 300 1.5 = 35.71 :=
by
  sorry

end value_decrease_proof_l675_67526


namespace imo_42_problem_l675_67579

theorem imo_42_problem {a b c : ℝ} (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a / Real.sqrt (a^2 + 8 * b * c) + b / Real.sqrt (b^2 + 8 * a * c) + c / Real.sqrt (c^2 + 8 * a * b) >= 1 :=
sorry

end imo_42_problem_l675_67579


namespace intercept_sum_equation_l675_67506

theorem intercept_sum_equation (c : ℝ) (h₀ : 3 * x + 4 * y + c = 0)
  (h₁ : (-(c / 3)) + (-(c / 4)) = 28) : c = -48 := 
by
  sorry

end intercept_sum_equation_l675_67506


namespace smallest_n_condition_l675_67594

noncomputable def distance_origin_to_point (n : ℕ) : ℝ := Real.sqrt (n)

noncomputable def radius_Bn (n : ℕ) : ℝ := distance_origin_to_point n - 1

def condition_Bn_contains_point_with_coordinate_greater_than_2 (n : ℕ) : Prop :=
  radius_Bn n > 2

theorem smallest_n_condition : ∃ n : ℕ, n ≥ 10 ∧ condition_Bn_contains_point_with_coordinate_greater_than_2 n :=
  sorry

end smallest_n_condition_l675_67594


namespace range_of_f_l675_67542

noncomputable def f (x : ℤ) : ℤ := x ^ 2 + 1

def domain : Set ℤ := {-1, 0, 1, 2}

def range_f : Set ℤ := {1, 2, 5}

theorem range_of_f : Set.image f domain = range_f :=
by
  sorry

end range_of_f_l675_67542


namespace problem1_problem2_l675_67509

open Real

noncomputable def alpha (hα : 0 < α ∧ α < π / 3) :=
  α

noncomputable def vec_a (hα : 0 < α ∧ α < π / 3) :=
  (sqrt 6 * sin (alpha hα), sqrt 2)

noncomputable def vec_b (hα : 0 < α ∧ α < π / 3) :=
  (1, cos (alpha hα) - sqrt 6 / 2)

theorem problem1 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  tan (alpha hα + π / 6) = sqrt 15 / 5 :=
sorry

theorem problem2 (hα : 0 < α ∧ α < π / 3) (h_orth : (sqrt 6 * sin (alpha hα)) + sqrt 2 * (cos (alpha hα) - sqrt 6 / 2) = 0) :
  cos (2 * alpha hα + 7 * π / 12) = (sqrt 2 - sqrt 30) / 8 :=
sorry

end problem1_problem2_l675_67509


namespace harry_geckos_count_l675_67575

theorem harry_geckos_count 
  (G : ℕ)
  (iguanas : ℕ := 2)
  (snakes : ℕ := 4)
  (cost_snake : ℕ := 10)
  (cost_iguana : ℕ := 5)
  (cost_gecko : ℕ := 15)
  (annual_cost : ℕ := 1140) :
  12 * (snakes * cost_snake + iguanas * cost_iguana + G * cost_gecko) = annual_cost → 
  G = 3 := 
by 
  intros h
  sorry

end harry_geckos_count_l675_67575


namespace days_required_for_C_l675_67563

noncomputable def rate_A (r_A r_B r_C : ℝ) : Prop := r_A + r_B = 1 / 3
noncomputable def rate_B (r_A r_B r_C : ℝ) : Prop := r_B + r_C = 1 / 6
noncomputable def rate_C (r_A r_B r_C : ℝ) : Prop := r_C + r_A = 1 / 4
noncomputable def days_for_C (r_C : ℝ) : ℝ := 1 / r_C

theorem days_required_for_C
  (r_A r_B r_C : ℝ)
  (h1 : rate_A r_A r_B r_C)
  (h2 : rate_B r_A r_B r_C)
  (h3 : rate_C r_A r_B r_C) :
  days_for_C r_C = 4.8 :=
sorry

end days_required_for_C_l675_67563


namespace vasya_fraction_l675_67564

-- Define the variables for distances and total distance
variables {a b c d s : ℝ}

-- Define conditions
def anton_distance (a b : ℝ) : Prop := a = b / 2
def sasha_distance (c a d : ℝ) : Prop := c = a + d
def dima_distance (d s : ℝ) : Prop := d = s / 10
def total_distance (a b c d s : ℝ) : Prop := a + b + c + d = s

-- The main theorem 
theorem vasya_fraction (a b c d s : ℝ) (h1 : anton_distance a b) 
  (h2 : sasha_distance c a d) (h3 : dima_distance d s)
  (h4 : total_distance a b c d s) : b / s = 0.4 :=
sorry

end vasya_fraction_l675_67564


namespace jacket_initial_reduction_l675_67536

theorem jacket_initial_reduction (P : ℝ) (x : ℝ) :
  P * (1 - x / 100) * 0.9 * 1.481481481481481 = P → x = 25 :=
by
  sorry

end jacket_initial_reduction_l675_67536


namespace clothes_in_total_l675_67584

-- Define the conditions as constants since they are fixed values
def piecesInOneLoad : Nat := 17
def numberOfSmallLoads : Nat := 5
def piecesPerSmallLoad : Nat := 6

-- Noncomputable for definition involving calculation
noncomputable def totalClothes : Nat :=
  piecesInOneLoad + (numberOfSmallLoads * piecesPerSmallLoad)

-- The theorem to prove Luke had 47 pieces of clothing in total
theorem clothes_in_total : totalClothes = 47 := by
  sorry

end clothes_in_total_l675_67584


namespace line_parallel_plane_l675_67541

axiom line (m : Type) : Prop
axiom plane (α : Type) : Prop
axiom has_no_common_points (m : Type) (α : Type) : Prop
axiom parallel (m : Type) (α : Type) : Prop

theorem line_parallel_plane
  (m : Type) (α : Type)
  (h : has_no_common_points m α) : parallel m α := sorry

end line_parallel_plane_l675_67541


namespace inequality_solution_l675_67515

theorem inequality_solution (x : ℝ) (h : x ≠ 4) : (x^2 - 16) / (x - 4) ≤ 0 ↔ x ∈ Set.Iic (-4) :=
by
  sorry

end inequality_solution_l675_67515


namespace jordan_meets_emily_after_total_time_l675_67570

noncomputable def meet_time
  (initial_distance : ℝ)
  (speed_ratio : ℝ)
  (decrease_rate : ℝ)
  (time_until_break : ℝ)
  (break_duration : ℝ)
  (total_meet_time : ℝ) : Prop :=
  initial_distance = 30 ∧
  speed_ratio = 2 ∧
  decrease_rate = 2 ∧
  time_until_break = 10 ∧
  break_duration = 5 ∧
  total_meet_time = 17

theorem jordan_meets_emily_after_total_time :
  meet_time 30 2 2 10 5 17 := 
by {
  -- The conditions directly state the requirements needed for the proof.
  exact ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩ -- This line confirms that all inputs match the given conditions.
}

end jordan_meets_emily_after_total_time_l675_67570


namespace student_correct_answers_l675_67586

theorem student_correct_answers 
(C W : ℕ) 
(h1 : C + W = 80) 
(h2 : 4 * C - W = 120) : 
C = 40 :=
by
  sorry 

end student_correct_answers_l675_67586


namespace volume_related_to_area_l675_67577

theorem volume_related_to_area (x y z : ℝ) 
  (bottom_area_eq : 3 * x * y = 3 * x * y)
  (front_area_eq : 2 * y * z = 2 * y * z)
  (side_area_eq : 3 * x * z = 3 * x * z) :
  (3 * x * y) * (2 * y * z) * (3 * x * z) = 18 * (x * y * z) ^ 2 := 
by sorry

end volume_related_to_area_l675_67577
