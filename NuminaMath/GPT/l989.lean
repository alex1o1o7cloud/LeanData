import Mathlib

namespace total_hours_worked_l989_98937

theorem total_hours_worked
  (x : ℕ)
  (h1 : 5 * x = 55)
  : 2 * x + 3 * x + 5 * x = 110 :=
by 
  sorry

end total_hours_worked_l989_98937


namespace sum_of_next_17_consecutive_integers_l989_98991

theorem sum_of_next_17_consecutive_integers (x : ℤ) (h₁ : (List.range 17).sum + 17 * x = 306) :
  (List.range 17).sum + 17 * (x + 17)  = 595 := 
sorry

end sum_of_next_17_consecutive_integers_l989_98991


namespace problem_1_problem_2_problem_3_l989_98930

variable (α : ℝ)
variable (h1 : 0 < α ∧ α < π)
variable (h2 : Real.tan α = -2)

theorem problem_1 : Real.sin (α + (π / 6)) = (2 * Real.sqrt 15 - Real.sqrt 5) / 10 := by
  sorry

theorem problem_2 : (2 * Real.cos ((π / 2) + α) - Real.cos (π - α)) / (Real.sin ((π / 2) - α) - 3 * Real.sin (π + α)) = 5 / 7 := by
  sorry

theorem problem_3 : 
  2 * Real.sin α ^ 2 - Real.sin α * Real.cos α + Real.cos α ^ 2 = 11 / 5 := by
  sorry

end problem_1_problem_2_problem_3_l989_98930


namespace find_k_l989_98983

theorem find_k (k t : ℝ) (h1 : t = 5) (h2 : (1/2) * (t^2) / ((k-1) * (k+1)) = 10) : 
  k = 3/2 := 
  sorry

end find_k_l989_98983


namespace percentage_decrease_l989_98953

theorem percentage_decrease (original_price new_price : ℝ) (h1 : original_price = 1400) (h2 : new_price = 1064) :
  ((original_price - new_price) / original_price * 100) = 24 :=
by
  sorry

end percentage_decrease_l989_98953


namespace product_of_roots_l989_98919

theorem product_of_roots (Q : Polynomial ℚ) (hQ : Q.degree = 1) (h_root : Q.eval 6 = 0) :
  (Q.roots : Multiset ℚ).prod = 6 :=
sorry

end product_of_roots_l989_98919


namespace add_ab_values_l989_98956

theorem add_ab_values (a b : ℝ) (h1 : ∀ x : ℝ, (x^2 + 4*x + 3) = (a*x + b)^2 + 4*(a*x + b) + 3) :
  a + b = -8 ∨ a + b = 4 :=
  by sorry

end add_ab_values_l989_98956


namespace science_students_count_l989_98912

def total_students := 400 + 120
def local_arts_students := 0.50 * 400
def local_commerce_students := 0.85 * 120
def total_local_students := 327

theorem science_students_count :
  0.25 * S = 25 →
  S = 100 :=
by
  sorry

end science_students_count_l989_98912


namespace amount_after_2_years_l989_98974

noncomputable def amount_after_n_years (present_value : ℝ) (rate_of_increase : ℝ) (years : ℕ) : ℝ :=
  present_value * (1 + rate_of_increase)^years

theorem amount_after_2_years :
  amount_after_n_years 6400 (1/8) 2 = 8100 :=
by
  sorry

end amount_after_2_years_l989_98974


namespace julie_upstream_distance_l989_98968

noncomputable def speed_of_stream : ℝ := 0.5
noncomputable def distance_downstream : ℝ := 72
noncomputable def time_spent : ℝ := 4
noncomputable def speed_of_julie_in_still_water : ℝ := 17.5
noncomputable def distance_upstream : ℝ := 68

theorem julie_upstream_distance :
  (distance_upstream / (speed_of_julie_in_still_water - speed_of_stream) = time_spent) ∧
  (distance_downstream / (speed_of_julie_in_still_water + speed_of_stream) = time_spent) →
  distance_upstream = 68 :=
by 
  sorry

end julie_upstream_distance_l989_98968


namespace distinct_sum_l989_98999

theorem distinct_sum (a b c d e : ℤ) (h1 : (7 - a) * (7 - b) * (7 - c) * (7 - d) * (7 - e) = 0)
  (h2 : a ≠ b) (h3 : a ≠ c) (h4 : a ≠ d) (h5 : a ≠ e) (h6 : b ≠ c) (h7 : b ≠ d) (h8 : b ≠ e) (h9 : c ≠ d) (h10 : c ≠ e) (h11 : d ≠ e) :
  a + b + c + d + e = 35 :=
sorry

end distinct_sum_l989_98999


namespace tan_ratio_l989_98984

theorem tan_ratio (x y : ℝ) (h1 : Real.sin (x + y) = 5 / 8) (h2 : Real.sin (x - y) = 1 / 4) : 
  (Real.tan x / Real.tan y) = 2 := 
by
  sorry 

end tan_ratio_l989_98984


namespace f_fraction_neg_1987_1988_l989_98901

-- Define the function f and its properties
def f : ℚ → ℝ := sorry

axiom functional_eq (x y : ℚ) : f (x + y) = f x * f y - f (x * y) + 1
axiom not_equal_f : f 1988 ≠ f 1987

-- Prove the desired equality
theorem f_fraction_neg_1987_1988 : f (-1987 / 1988) = 1 / 1988 :=
by
  sorry

end f_fraction_neg_1987_1988_l989_98901


namespace four_gt_sqrt_fifteen_l989_98972

theorem four_gt_sqrt_fifteen : 4 > Real.sqrt 15 := 
sorry

end four_gt_sqrt_fifteen_l989_98972


namespace sausage_left_l989_98938

variables (S x y : ℝ)

-- Conditions
axiom dog_bites : y = x + 300
axiom cat_bites : x = y + 500

-- Theorem Statement
theorem sausage_left {S x y : ℝ}
  (h1 : y = x + 300)
  (h2 : x = y + 500) : S - x - y = 400 :=
by
  sorry

end sausage_left_l989_98938


namespace jerusha_earnings_l989_98931

theorem jerusha_earnings (L : ℕ) (h1 : 5 * L = 85) : 4 * L = 68 := 
by
  sorry

end jerusha_earnings_l989_98931


namespace larger_number_ratio_l989_98998

theorem larger_number_ratio (x : ℕ) (a b : ℕ) (h1 : a = 3 * x) (h2 : b = 8 * x) 
(h3 : (a - 24) * 9 = (b - 24) * 4) : b = 192 :=
sorry

end larger_number_ratio_l989_98998


namespace Ann_is_16_l989_98917

variable (A S : ℕ)

theorem Ann_is_16
  (h1 : A = S + 5)
  (h2 : A + S = 27) :
  A = 16 :=
by
  sorry

end Ann_is_16_l989_98917


namespace train_cross_time_l989_98951

open Real

noncomputable def length_train1 := 190 -- in meters
noncomputable def length_train2 := 160 -- in meters
noncomputable def speed_train1 := 60 * (5/18) --speed_kmhr_to_msec 60 km/hr to m/s
noncomputable def speed_train2 := 40 * (5/18) -- speed_kmhr_to_msec 40 km/hr to m/s
noncomputable def relative_speed := speed_train1 + speed_train2 -- relative speed

theorem train_cross_time :
  (length_train1 + length_train2) / relative_speed = 350 / ((60 * (5/18)) + (40 * (5/18))) :=
by
  sorry -- The proof will be here initially just to validate the Lean statement

end train_cross_time_l989_98951


namespace slope_of_line_joining_solutions_l989_98945

theorem slope_of_line_joining_solutions (x1 x2 y1 y2 : ℝ) :
  (4 / x1 + 5 / y1 = 1) → (4 / x2 + 5 / y2 = 1) →
  (x1 ≠ x2) → (y1 = 5 * x1 / (4 * x1 - 1)) → (y2 = 5 * x2 / (4 * x2 - 1)) →
  (x1 ≠ 1 / 4) → (x2 ≠ 1 / 4) →
  ((y2 - y1) / (x2 - x1) = - (5 / 21)) :=
by
  intros h_eq1 h_eq2 h_neq h_y1 h_y2 h_x1 h_x2
  -- Proof omitted for brevity
  sorry

end slope_of_line_joining_solutions_l989_98945


namespace find_a_l989_98914

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.sin (2 * x) - (1 / 3) * Real.sin (3 * x)

noncomputable def f_prime (a : ℝ) (x : ℝ) : ℝ :=
  2 * a * Real.cos (2 * x) - Real.cos (3 * x)

theorem find_a (a : ℝ) (h : f_prime a (Real.pi / 3) = 0) : a = 1 :=
by
  sorry

end find_a_l989_98914


namespace weight_of_B_l989_98971

theorem weight_of_B (A B C : ℝ) 
  (h1 : A + B + C = 135) 
  (h2 : A + B = 80) 
  (h3 : B + C = 86) : 
  B = 31 :=
sorry

end weight_of_B_l989_98971


namespace domain_of_function_l989_98996

theorem domain_of_function (x : ℝ) (k : ℤ) :
  ∃ x, (2 * Real.sin x + 1 ≥ 0) ↔ (- (Real.pi / 6) + 2 * k * Real.pi ≤ x ∧ x ≤ (7 * Real.pi / 6) + 2 * k * Real.pi) :=
sorry

end domain_of_function_l989_98996


namespace find_xyz_l989_98975

def divisible_by (n k : ℕ) : Prop := k % n = 0

def is_7_digit_number (a b c d e f g : ℕ) : ℕ := 
  10^6 * a + 10^5 * b + 10^4 * c + 10^3 * d + 10^2 * e + 10 * f + g

theorem find_xyz
  (x y z : ℕ)
  (h : divisible_by 792 (is_7_digit_number 1 4 x y 7 8 z))
  : (100 * x + 10 * y + z) = 644 :=
by
  sorry

end find_xyz_l989_98975


namespace find_a_even_function_l989_98976

theorem find_a_even_function (f : ℝ → ℝ) (a : ℝ)
  (h_even : ∀ x, f x = f (-x))
  (h_domain : ∀ x, 2 * a + 1 ≤ x ∧ x ≤ a + 5) :
  a = -2 :=
sorry

end find_a_even_function_l989_98976


namespace medical_bills_value_l989_98913

variable (M : ℝ)
variable (property_damage : ℝ := 40000)
variable (insurance_coverage : ℝ := 0.80)
variable (carl_coverage : ℝ := 0.20)
variable (carl_owes : ℝ := 22000)

theorem medical_bills_value : 0.20 * (property_damage + M) = carl_owes → M = 70000 := 
by
  intro h
  sorry

end medical_bills_value_l989_98913


namespace smallest_n_l989_98939

theorem smallest_n (n : ℕ) (h : n ≥ 2) : 
  (∃ m : ℕ, m * m = (n + 1) * (2 * n + 1) / 6) ↔ n = 337 :=
by
  sorry

end smallest_n_l989_98939


namespace shirt_cost_l989_98990

theorem shirt_cost (J S : ℝ) (h1 : 3 * J + 2 * S = 69) (h2 : 2 * J + 3 * S = 86) : S = 24 :=
by
  sorry

end shirt_cost_l989_98990


namespace abs_diff_of_pq_eq_6_and_pq_sum_7_l989_98973

variable (p q : ℝ)

noncomputable def abs_diff (a b : ℝ) := |a - b|

theorem abs_diff_of_pq_eq_6_and_pq_sum_7 (hpq : p * q = 6) (hpq_sum : p + q = 7) : abs_diff p q = 5 :=
by
  sorry

end abs_diff_of_pq_eq_6_and_pq_sum_7_l989_98973


namespace repair_cost_total_l989_98927

def hourly_labor_cost : ℝ := 75
def labor_hours : ℝ := 16
def part_cost : ℝ := 1200
def labor_cost : ℝ := hourly_labor_cost * labor_hours
def total_cost : ℝ := labor_cost + part_cost

theorem repair_cost_total : total_cost = 2400 := 
by
  -- Proof omitted
  sorry

end repair_cost_total_l989_98927


namespace book_pages_l989_98921

theorem book_pages (total_pages : ℝ) : 
  (0.1 * total_pages + 0.25 * total_pages + 30 = 0.5 * total_pages) → 
  total_pages = 240 :=
by
  sorry

end book_pages_l989_98921


namespace monotonic_conditions_fixed_point_property_l989_98982

noncomputable
def f (x a b c : ℝ) : ℝ := x^3 - a * x^2 - b * x + c

theorem monotonic_conditions (a b c : ℝ) :
  a = 0 ∧ c = 0 ∧ b ≤ 3 ↔ ∀ x : ℝ, (x ≥ 1 → (f x a b c) ≥ 1) → ∀ x y: ℝ, (x ≥ y ↔ f x a b c ≤ f y a b c) := sorry

theorem fixed_point_property (a b c : ℝ) :
  (∀ x : ℝ, (x ≥ 1 ∧ (f x a b c) ≥ 1) → f (f x a b c) a b c = x) ↔ (f x 0 b 0 = x) := sorry

end monotonic_conditions_fixed_point_property_l989_98982


namespace molecular_weight_is_171_35_l989_98977

def atomic_weight_ba : ℝ := 137.33
def atomic_weight_o : ℝ := 16.00
def atomic_weight_h : ℝ := 1.01

def molecular_weight : ℝ :=
  (1 * atomic_weight_ba) + (2 * atomic_weight_o) + (2 * atomic_weight_h)

-- The goal is to prove that the molecular weight is 171.35
theorem molecular_weight_is_171_35 : molecular_weight = 171.35 :=
by
  sorry

end molecular_weight_is_171_35_l989_98977


namespace diameter_of_triple_sphere_l989_98929

noncomputable def radius_of_sphere : ℝ := 6

noncomputable def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

noncomputable def triple_volume_of_sphere (r : ℝ) : ℝ := 3 * volume_of_sphere r

noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3)

theorem diameter_of_triple_sphere (r : ℝ) (V1 V2 : ℝ) (a b : ℝ) 
  (h_r : r = radius_of_sphere)
  (h_V1 : V1 = volume_of_sphere r)
  (h_V2 : V2 = triple_volume_of_sphere r)
  (h_d : 12 * cube_root 3 = 2 * (6 * cube_root 3))
  : a + b = 15 :=
sorry

end diameter_of_triple_sphere_l989_98929


namespace range_of_m_l989_98909

theorem range_of_m (m : ℝ) :
  (¬(∀ x : ℝ, x^2 + m * x + 1 = 0 → x ≠ 0) ∧ ∀ x : ℝ, 4 * x^2 + 4 * (m - 2) * x + 1 ≠ 0) → (1 < m ∧ m ≤ 2) :=
by
  sorry

end range_of_m_l989_98909


namespace part1_part2_l989_98967

noncomputable def A : Set ℝ := {x | x^2 + 4 * x = 0 }

noncomputable def B (a : ℝ) : Set ℝ := {x | x^2 + 2 * (a + 1) * x + a^2 - 1 = 0 }

-- Part (1): Prove a = 1 given A ∪ B = B
theorem part1 (a : ℝ) (h : A ∪ B a = B a) : a = 1 :=
sorry

-- Part (2): Prove the set C composed of the values of a given A ∩ B = B
def C : Set ℝ := {a | a ≤ -1 ∨ a = 1}

theorem part2 (h : ∀ a, A ∩ B a = B a ↔ a ∈ C) : forall a, A ∩ B a = B a ↔ a ∈ C :=
sorry

end part1_part2_l989_98967


namespace find_probability_between_0_and_1_l989_98908

-- Define a random variable X following a normal distribution N(μ, σ²)
variables {X : ℝ → ℝ} {μ σ : ℝ}
-- Define conditions:
-- Condition 1: X follows a normal distribution with mean μ and variance σ²
def normal_dist (X : ℝ → ℝ) (μ σ : ℝ) : Prop :=
  sorry  -- Assume properties of normal distribution are satisfied

-- Condition 2: P(X < 1) = 1/2
def P_X_lt_1 : Prop := 
  sorry  -- Assume that P(X < 1) = 1/2

-- Condition 3: P(X > 2) = p
def P_X_gt_2 (p : ℝ) : Prop := 
  sorry  -- Assume that P(X > 2) = p

noncomputable
def probability_X_between_0_and_1 (p : ℝ) : ℝ :=
  1/2 - p

theorem find_probability_between_0_and_1 (X : ℝ → ℝ) {μ σ p : ℝ} 
  (hX : normal_dist X μ σ)
  (h1 : P_X_lt_1)
  (h2 : P_X_gt_2 p) :
  probability_X_between_0_and_1 p = 1/2 - p := 
  sorry

end find_probability_between_0_and_1_l989_98908


namespace sunday_price_correct_l989_98966

def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.60
def second_discount_rate : ℝ := 0.25
def discounted_price : ℝ := original_price * (1 - first_discount_rate)
def sunday_price : ℝ := discounted_price * (1 - second_discount_rate)

theorem sunday_price_correct :
  sunday_price = 75 := by
  sorry

end sunday_price_correct_l989_98966


namespace eq_nine_l989_98922

theorem eq_nine (x y : ℝ) (h1 : x^2 + y^2 = 15) (h2 : x * y = 3) : (x - y)^2 = 9 := by
  sorry

end eq_nine_l989_98922


namespace jane_total_drawing_paper_l989_98916

theorem jane_total_drawing_paper (brown_sheets : ℕ) (yellow_sheets : ℕ) 
    (h1 : brown_sheets = 28) (h2 : yellow_sheets = 27) : 
    brown_sheets + yellow_sheets = 55 := 
by
    sorry

end jane_total_drawing_paper_l989_98916


namespace length_AB_indeterminate_l989_98969

theorem length_AB_indeterminate
  (A B C : Type)
  (AC : ℝ) (BC : ℝ)
  (AC_eq_1 : AC = 1)
  (BC_eq_3 : BC = 3) :
  (2 < AB ∧ AB < 4) ∨ (AB = 2 ∨ AB = 4) → false :=
by sorry

end length_AB_indeterminate_l989_98969


namespace bead_problem_l989_98985

theorem bead_problem 
  (x y : ℕ) 
  (hx : 19 * x + 17 * y = 2017): 
  (x + y = 107) ∨ (x + y = 109) ∨ (x + y = 111) ∨ (x + y = 113) ∨ (x + y = 115) ∨ (x + y = 117) := 
sorry

end bead_problem_l989_98985


namespace cone_height_l989_98934

theorem cone_height (V : ℝ) (h : ℝ) (r : ℝ) (vertex_angle : ℝ) 
  (H1 : V = 16384 * Real.pi)
  (H2 : vertex_angle = 90) 
  (H3 : V = (1 / 3) * Real.pi * r^2 * h)
  (H4 : h = r) : 
  h = 36.6 :=
by
  sorry

end cone_height_l989_98934


namespace random_point_between_R_S_l989_98997

theorem random_point_between_R_S {P Q R S : ℝ} (PQ PR RS : ℝ) (h1 : PQ = 4 * PR) (h2 : PQ = 8 * RS) :
  let PS := PR + RS
  let probability := RS / PQ
  probability = 5 / 8 :=
by
  let PS := PR + RS
  let probability := RS / PQ
  sorry

end random_point_between_R_S_l989_98997


namespace unique_positive_integer_b_quadratic_solution_l989_98902

theorem unique_positive_integer_b_quadratic_solution (c : ℝ) :
  (∃! (b : ℕ), ∀ (x : ℝ), x^2 + (b^2 + (1 / b^2)) * x + c = 3) ↔ c = 5 :=
sorry

end unique_positive_integer_b_quadratic_solution_l989_98902


namespace mrs_hilt_total_candy_l989_98962

theorem mrs_hilt_total_candy :
  (2 * 3) + (4 * 2) + (6 * 4) = 38 :=
by
  -- here, skip the proof as instructed
  sorry

end mrs_hilt_total_candy_l989_98962


namespace unique_a_for_three_distinct_real_solutions_l989_98915

theorem unique_a_for_three_distinct_real_solutions (a : ℝ) :
  (∃ f : ℝ → ℝ, (∀ x, f x = a * x^2 - 2 * x + 1 - 3 * |x|) ∧
  ((∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 ∧ f x1 = 0 ∧ f x2 = 0 ∧ f x3 = 0) ∧
  (∀ x4 : ℝ, f x4 = 0 → (x4 = x1 ∨ x4 = x2 ∨ x4 = x3) )) ) ↔
  a = 1 / 4 :=
sorry

end unique_a_for_three_distinct_real_solutions_l989_98915


namespace parallel_lines_slopes_l989_98936

theorem parallel_lines_slopes (k : ℝ) :
  (∀ x y : ℝ, x + (1 + k) * y = 2 - k → k * x + 2 * y + 8 = 0 → k = 1) :=
by
  intro h1 h2
  -- We can see that there should be specifics here about how the conditions lead to k = 1
  sorry

end parallel_lines_slopes_l989_98936


namespace lcm_of_product_of_mutually_prime_l989_98988

theorem lcm_of_product_of_mutually_prime (a b : ℕ) (h : Nat.gcd a b = 1) : Nat.lcm a b = a * b :=
by
  sorry

end lcm_of_product_of_mutually_prime_l989_98988


namespace inequalities_correct_l989_98940

theorem inequalities_correct (a b : ℝ) (h : a * b > 0) :
  |b| > |a| ∧ |a + b| < |b| := sorry

end inequalities_correct_l989_98940


namespace urn_gold_coins_percentage_l989_98989

theorem urn_gold_coins_percentage (obj_perc_beads : ℝ) (coins_perc_gold : ℝ) : 
    obj_perc_beads = 0.15 → coins_perc_gold = 0.65 → 
    (1 - obj_perc_beads) * coins_perc_gold = 0.5525 := 
by
  intros h_obj_perc_beads h_coins_perc_gold
  sorry

end urn_gold_coins_percentage_l989_98989


namespace decrease_A_share_l989_98986

theorem decrease_A_share :
  ∃ (a b x : ℝ),
    a + b + 495 = 1010 ∧
    (a - x) / 3 = 96 ∧
    (b - 10) / 2 = 96 ∧
    x = 25 :=
by
  sorry

end decrease_A_share_l989_98986


namespace total_planks_l989_98904

-- Define the initial number of planks
def initial_planks : ℕ := 15

-- Define the planks Charlie got
def charlie_planks : ℕ := 10

-- Define the planks Charlie's father got
def father_planks : ℕ := 10

-- Prove the total number of planks
theorem total_planks : (initial_planks + charlie_planks + father_planks) = 35 :=
by sorry

end total_planks_l989_98904


namespace incorrect_statement_l989_98918

theorem incorrect_statement : ¬ (∀ x : ℝ, x ≠ 0 → (1 / x = 1 ∨ 1 / x = -1)) :=
by
  -- Proof goes here
  sorry

end incorrect_statement_l989_98918


namespace least_perimeter_of_triangle_l989_98970

theorem least_perimeter_of_triangle (cosA cosB cosC : ℝ)
  (h₁ : cosA = 13 / 16)
  (h₂ : cosB = 4 / 5)
  (h₃ : cosC = -3 / 5) :
  ∃ a b c : ℕ, a + b + c = 28 ∧ 
  a^2 + b^2 - c^2 = 2 * a * b * cosC ∧ 
  b^2 + c^2 - a^2 = 2 * b * c * cosA ∧ 
  c^2 + a^2 - b^2 = 2 * c * a * cosB :=
sorry

end least_perimeter_of_triangle_l989_98970


namespace palindrome_probability_divisible_by_11_l989_98992

namespace PalindromeProbability

-- Define the concept of a five-digit palindrome and valid digits
def is_five_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), a ≠ 0 ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ n = 10001 * a + 1010 * b + 100 * c

-- Define the condition for a number being divisible by 11
def is_divisible_by_11 (n : ℕ) : Prop :=
  n % 11 = 0

-- Count all five-digit palindromes
def count_five_digit_palindromes : ℕ :=
  9 * 10 * 10  -- There are 9 choices for a (1-9), and 10 choices for b and c (0-9)

-- Count five-digit palindromes that are divisible by 11
def count_divisible_by_11_five_digit_palindromes : ℕ :=
  9 * 10  -- There are 9 choices for a, and 10 valid (b, c) pairs for divisibility by 11

-- Calculate the probability
theorem palindrome_probability_divisible_by_11 :
  (count_divisible_by_11_five_digit_palindromes : ℚ) / count_five_digit_palindromes = 1 / 10 :=
  by sorry -- Proof goes here

end PalindromeProbability

end palindrome_probability_divisible_by_11_l989_98992


namespace min_max_values_on_interval_l989_98995

noncomputable def f (x : ℝ) : ℝ := (Real.cos x) + (x + 1)*(Real.sin x) + 1

theorem min_max_values_on_interval :
  (∀ x ∈ Set.Icc 0 (2*Real.pi), f x ≥ -(3*Real.pi/2) ∧ f x ≤ (Real.pi/2 + 2)) ∧
  ( ∃ a ∈ Set.Icc 0 (2*Real.pi), f a = -(3*Real.pi/2) ) ∧
  ( ∃ b ∈ Set.Icc 0 (2*Real.pi), f b = (Real.pi/2 + 2) ) :=
by
  sorry

end min_max_values_on_interval_l989_98995


namespace leap_day_2040_is_tuesday_l989_98959

def days_in_non_leap_year := 365
def days_in_leap_year := 366
def leap_years_between_2000_and_2040 := 10

def total_days_between_2000_and_2040 := 
  30 * days_in_non_leap_year + leap_years_between_2000_and_2040 * days_in_leap_year

theorem leap_day_2040_is_tuesday :
  (total_days_between_2000_and_2040 % 7) = 0 :=
by
  sorry

end leap_day_2040_is_tuesday_l989_98959


namespace rounding_example_l989_98955

theorem rounding_example (x : ℝ) (h : x = 8899.50241201) : round x = 8900 :=
by
  sorry

end rounding_example_l989_98955


namespace sum_n_10_terms_progression_l989_98980

noncomputable def sum_arith_progression (n a d : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem sum_n_10_terms_progression :
  ∃ (a : ℕ), (∃ (n : ℕ), sum_arith_progression n a 3 = 220) ∧
  (2 * a + (10 - 1) * 3) = 43 ∧
  sum_arith_progression 10 a 3 = 215 :=
by sorry

end sum_n_10_terms_progression_l989_98980


namespace reduced_rates_start_l989_98920

theorem reduced_rates_start (reduced_fraction : ℝ) (total_hours : ℝ) (weekend_hours : ℝ) (weekday_hours : ℝ) 
  (start_time : ℝ) (end_time : ℝ) : 
  reduced_fraction = 0.6428571428571429 → 
  total_hours = 168 → 
  weekend_hours = 48 → 
  weekday_hours = 60 - weekend_hours → 
  end_time = 8 → 
  start_time = end_time - weekday_hours → 
  start_time = 20 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end reduced_rates_start_l989_98920


namespace probability_detecting_drunk_driver_l989_98910

namespace DrunkDrivingProbability

def P_A : ℝ := 0.05
def P_B_given_A : ℝ := 0.99
def P_B_given_not_A : ℝ := 0.01

def P_not_A : ℝ := 1 - P_A

def P_B : ℝ := P_A * P_B_given_A + P_not_A * P_B_given_not_A

theorem probability_detecting_drunk_driver :
  P_B = 0.059 :=
by
  sorry

end DrunkDrivingProbability

end probability_detecting_drunk_driver_l989_98910


namespace smallest_positive_integer_satisfying_conditions_l989_98978

theorem smallest_positive_integer_satisfying_conditions :
  ∃ (x : ℕ),
    x % 4 = 1 ∧
    x % 5 = 2 ∧
    x % 7 = 3 ∧
    ∀ y : ℕ, (y % 4 = 1 ∧ y % 5 = 2 ∧ y % 7 = 3) → y ≥ x ∧ x = 93 :=
by
  sorry

end smallest_positive_integer_satisfying_conditions_l989_98978


namespace find_a_l989_98905

theorem find_a (a b c : ℂ) (ha : a.re = a) (h1 : a + b + c = 5) (h2 : a * b + b * c + c * a = 7) (h3 : a * b * c = 6) : a = 1 :=
by
  sorry

end find_a_l989_98905


namespace sqrt_expression_value_l989_98924

theorem sqrt_expression_value :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 :=
by
  sorry

end sqrt_expression_value_l989_98924


namespace stratified_sampling_example_l989_98948

theorem stratified_sampling_example 
    (high_school_students : ℕ)
    (junior_high_students : ℕ) 
    (sampled_high_school_students : ℕ)
    (sampling_ratio : ℚ)
    (total_students : ℕ)
    (n : ℕ)
    (h1 : high_school_students = 3500)
    (h2 : junior_high_students = 1500)
    (h3 : sampled_high_school_students = 70)
    (h4 : sampling_ratio = sampled_high_school_students / high_school_students)
    (h5 : total_students = high_school_students + junior_high_students) :
    n = total_students * sampling_ratio → 
    n = 100 :=
by
  sorry

end stratified_sampling_example_l989_98948


namespace max_gold_coins_l989_98958

-- Define the conditions as predicates
def divides_with_remainder (n : ℕ) (d r : ℕ) : Prop := n % d = r
def less_than (n k : ℕ) : Prop := n < k

-- Main statement incorporating the conditions and the conclusion
theorem max_gold_coins (n : ℕ) :
  divides_with_remainder n 15 3 ∧ less_than n 120 → n ≤ 105 :=
by
  sorry

end max_gold_coins_l989_98958


namespace weight_of_packet_a_l989_98943

theorem weight_of_packet_a
  (A B C D E F : ℝ)
  (h1 : (A + B + C) / 3 = 84)
  (h2 : (A + B + C + D) / 4 = 80)
  (h3 : E = D + 3)
  (h4 : (B + C + D + E) / 4 = 79)
  (h5 : F = (A + E) / 2)
  (h6 : (B + C + D + E + F) / 5 = 81) :
  A = 75 :=
by sorry

end weight_of_packet_a_l989_98943


namespace hadassah_additional_paintings_l989_98993

noncomputable def hadassah_initial_paintings : ℕ := 12
noncomputable def hadassah_initial_hours : ℕ := 6
noncomputable def hadassah_total_hours : ℕ := 16

theorem hadassah_additional_paintings 
  (initial_paintings : ℕ)
  (initial_hours : ℕ)
  (total_hours : ℕ) :
  initial_paintings = hadassah_initial_paintings →
  initial_hours = hadassah_initial_hours →
  total_hours = hadassah_total_hours →
  let additional_hours := total_hours - initial_hours
  let painting_rate := initial_paintings / initial_hours
  let additional_paintings := painting_rate * additional_hours
  additional_paintings = 20 :=
by
  sorry

end hadassah_additional_paintings_l989_98993


namespace closest_point_to_line_l989_98965

theorem closest_point_to_line {x y : ℝ} :
  (y = 2 * x - 7) → (∃ p : ℝ × ℝ, p.1 = 5 ∧ p.2 = 3 ∧ (p.1, p.2) ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} ∧ (∀ q : ℝ × ℝ, q ∈ {q : ℝ × ℝ | q.2 = 2 * q.1 - 7} → dist ⟨x, y⟩ p ≤ dist ⟨x, y⟩ q)) :=
by
  -- proof goes here
  sorry

end closest_point_to_line_l989_98965


namespace find_B_l989_98933

theorem find_B (A B C : ℝ) (h1 : A = B + C) (h2 : A + B = 1/25) (h3 : C = 1/35) : B = 1/175 :=
by
  sorry

end find_B_l989_98933


namespace tank_capacity_l989_98947

theorem tank_capacity (C : ℝ) 
  (h1 : 10 > 0) 
  (h2 : 16 > (10 : ℝ))
  (h3 : ((C/10) - 480 = (C/16))) : C = 1280 := 
by 
  sorry

end tank_capacity_l989_98947


namespace find_t_for_area_of_triangle_l989_98928

theorem find_t_for_area_of_triangle :
  ∃ (t : ℝ), 
  (∀ (A B C T U: ℝ × ℝ),
    A = (0, 10) → 
    B = (3, 0) → 
    C = (9, 0) → 
    T = (3/10 * (10 - t), t) →
    U = (9/10 * (10 - t), t) →
    2 * 15 = 3/10 * (10 - t) ^ 2) →
  t = 2.93 :=
by sorry

end find_t_for_area_of_triangle_l989_98928


namespace emily_gardens_and_seeds_l989_98994

variables (total_seeds planted_big_garden tom_seeds lettuce_seeds pepper_seeds tom_gardens lettuce_gardens pepper_gardens : ℕ)

def seeds_left (total_seeds planted_big_garden : ℕ) : ℕ :=
  total_seeds - planted_big_garden

def seeds_used_tomatoes (tom_seeds tom_gardens : ℕ) : ℕ :=
  tom_seeds * tom_gardens

def seeds_used_lettuce (lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  lettuce_seeds * lettuce_gardens

def seeds_used_peppers (pepper_seeds pepper_gardens : ℕ) : ℕ :=
  pepper_seeds * pepper_gardens

def remaining_seeds (total_seeds planted_big_garden tom_seeds tom_gardens lettuce_seeds lettuce_gardens : ℕ) : ℕ :=
  seeds_left total_seeds planted_big_garden - (seeds_used_tomatoes tom_seeds tom_gardens + seeds_used_lettuce lettuce_seeds lettuce_gardens)

def total_small_gardens (tom_gardens lettuce_gardens pepper_gardens : ℕ) : ℕ :=
  tom_gardens + lettuce_gardens + pepper_gardens

theorem emily_gardens_and_seeds :
  total_seeds = 42 ∧
  planted_big_garden = 36 ∧
  tom_seeds = 4 ∧
  lettuce_seeds = 3 ∧
  pepper_seeds = 2 ∧
  tom_gardens = 3 ∧
  lettuce_gardens = 2 →
  seeds_used_peppers pepper_seeds pepper_gardens = 0 ∧
  total_small_gardens tom_gardens lettuce_gardens pepper_gardens = 5 :=
by
  sorry

end emily_gardens_and_seeds_l989_98994


namespace prob_both_successful_prob_at_least_one_successful_l989_98944

variables (P_A P_B : ℚ)
variables (h1 : P_A = 1 / 2)
variables (h2 : P_B = 2 / 5)

/-- Prove that the probability that both A and B score in one shot each is 1 / 5. -/
theorem prob_both_successful (P_A P_B : ℚ) (h1 : P_A = 1 / 2) (h2 : P_B = 2 / 5) :
  P_A * P_B = 1 / 5 :=
by sorry

variables (P_A_miss P_B_miss : ℚ)
variables (h3 : P_A_miss = 1 / 2)
variables (h4 : P_B_miss = 3 / 5)

/-- Prove that the probability that at least one shot is successful is 7 / 10. -/
theorem prob_at_least_one_successful (P_A_miss P_B_miss : ℚ) (h3 : P_A_miss = 1 / 2) (h4 : P_B_miss = 3 / 5) :
  1 - P_A_miss * P_B_miss = 7 / 10 :=
by sorry

end prob_both_successful_prob_at_least_one_successful_l989_98944


namespace sixth_number_of_11_consecutive_odd_sum_1991_is_181_l989_98906

theorem sixth_number_of_11_consecutive_odd_sum_1991_is_181 :
  (∃ (n : ℤ), (2 * n + 1) + (2 * n + 3) + (2 * n + 5) + (2 * n + 7) + (2 * n + 9) + (2 * n + 11) + (2 * n + 13) + (2 * n + 15) + (2 * n + 17) + (2 * n + 19) + (2 * n + 21) = 1991) →
  2 * 85 + 11 = 181 := 
by
  sorry

end sixth_number_of_11_consecutive_odd_sum_1991_is_181_l989_98906


namespace number_of_rallies_l989_98949

open Nat

def X_rallies : Nat := 10
def O_rallies : Nat := 100
def sequence_Os : Nat := 3
def sequence_Xs : Nat := 7

theorem number_of_rallies : 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≤ 379) ∧ 
  (sequence_Os * O_rallies + sequence_Xs * X_rallies ≥ 370) := 
by
  sorry

end number_of_rallies_l989_98949


namespace total_candles_in_small_boxes_l989_98960

-- Definitions of the conditions
def num_small_boxes_per_big_box := 4
def num_big_boxes := 50
def candles_per_small_box := 40

-- The total number of small boxes
def total_small_boxes : Nat := num_small_boxes_per_big_box * num_big_boxes

-- The statement to prove the total number of candles in all small boxes is 8000
theorem total_candles_in_small_boxes : candles_per_small_box * total_small_boxes = 8000 :=
by 
  sorry

end total_candles_in_small_boxes_l989_98960


namespace parabola_condition_l989_98935

/-- Given the point (3,0) lies on the parabola y = 2x^2 + (k + 2)x - k,
    prove that k = -12. -/
theorem parabola_condition (k : ℝ) (h : 0 = 2 * 3^2 + (k + 2) * 3 - k) : k = -12 :=
by 
  sorry

end parabola_condition_l989_98935


namespace quotient_calculation_l989_98923

theorem quotient_calculation (dividend divisor remainder expected_quotient : ℕ)
  (h₁ : dividend = 166)
  (h₂ : divisor = 18)
  (h₃ : remainder = 4)
  (h₄ : dividend = divisor * expected_quotient + remainder) :
  expected_quotient = 9 :=
by
  sorry

end quotient_calculation_l989_98923


namespace equivalent_functions_l989_98946

theorem equivalent_functions :
  ∀ (x t : ℝ), (x^2 - 2*x - 1 = t^2 - 2*t + 1) := 
by
  intros x t
  sorry

end equivalent_functions_l989_98946


namespace alpha_beta_sum_l989_98957

theorem alpha_beta_sum (α β : ℝ) (h : ∀ x : ℝ, x ≠ 54 → x ≠ -60 → (x - α) / (x + β) = (x^2 - 72 * x + 945) / (x^2 + 45 * x - 3240)) :
  α + β = 81 :=
sorry

end alpha_beta_sum_l989_98957


namespace find_x_orthogonal_l989_98952

theorem find_x_orthogonal :
  ∃ x : ℝ, (2 * x + 5 * (-3) = 0) ∧ x = 15 / 2 :=
by
  sorry

end find_x_orthogonal_l989_98952


namespace mean_cars_l989_98911

theorem mean_cars (a b c d e : ℝ) (h1 : a = 30) (h2 : b = 14) (h3 : c = 14) (h4 : d = 21) (h5 : e = 25) : 
  (a + b + c + d + e) / 5 = 20.8 :=
by
  -- The proof will be provided here
  sorry

end mean_cars_l989_98911


namespace total_distance_walked_l989_98979

noncomputable def hazel_total_distance : ℕ := 3

def distance_first_hour := 2  -- The distance traveled in the first hour (in kilometers)
def distance_second_hour := distance_first_hour * 2  -- The distance traveled in the second hour
def distance_third_hour := distance_second_hour / 2  -- The distance traveled in the third hour, with a 50% speed decrease

theorem total_distance_walked :
  distance_first_hour + distance_second_hour + distance_third_hour = 8 :=
  by
    sorry

end total_distance_walked_l989_98979


namespace roots_cubic_sum_l989_98925

theorem roots_cubic_sum:
  (∃ p q r : ℝ, 
     (p^3 - p^2 + p - 2 = 0) ∧ 
     (q^3 - q^2 + q - 2 = 0) ∧ 
     (r^3 - r^2 + r - 2 = 0)) 
  → 
  (∃ p q r : ℝ, p^3 + q^3 + r^3 = 4) := 
by 
  sorry

end roots_cubic_sum_l989_98925


namespace Monica_tiles_count_l989_98987

noncomputable def total_tiles (length width : ℕ) := 
  let double_border_tiles := (2 * ((length - 4) + (width - 4)) + 8)
  let inner_area := (length - 4) * (width - 4)
  let three_foot_tiles := (inner_area + 8) / 9
  double_border_tiles + three_foot_tiles

theorem Monica_tiles_count : total_tiles 18 24 = 183 := 
by
  sorry

end Monica_tiles_count_l989_98987


namespace olivine_more_stones_l989_98964

theorem olivine_more_stones (x O D : ℕ) (h1 : O = 30 + x) (h2 : D = O + 11)
  (h3 : 30 + O + D = 111) : x = 5 :=
by
  sorry

end olivine_more_stones_l989_98964


namespace muffins_count_l989_98942

-- Lean 4 Statement
theorem muffins_count (doughnuts muffins : ℕ) (ratio_doughnuts_muffins : ℕ → ℕ → Prop)
  (h_ratio : ratio_doughnuts_muffins 5 1) (h_doughnuts : doughnuts = 50) :
  muffins = 10 :=
by
  sorry

end muffins_count_l989_98942


namespace paul_is_19_years_old_l989_98903

theorem paul_is_19_years_old
  (mark_age : ℕ)
  (alice_age : ℕ)
  (paul_age : ℕ)
  (h1 : mark_age = 20)
  (h2 : alice_age = mark_age + 4)
  (h3 : paul_age = alice_age - 5) : 
  paul_age = 19 := by 
  sorry

end paul_is_19_years_old_l989_98903


namespace second_chick_eats_52_l989_98900

theorem second_chick_eats_52 (days : ℕ) (first_chick_eats : ℕ → ℕ) (second_chick_eats : ℕ → ℕ) :
  (∀ n, first_chick_eats n + second_chick_eats n = 12) →
  (∃ a b, first_chick_eats a = 7 ∧ second_chick_eats a = 5 ∧
          first_chick_eats b = 7 ∧ second_chick_eats b = 5 ∧
          12 * days = first_chick_eats a * 2 + first_chick_eats b * 6 + second_chick_eats a * 2 + second_chick_eats b * 6) →
  (first_chick_eats a * 2 + first_chick_eats b * 6 = 44) →
  (second_chick_eats a * 2 + second_chick_eats b * 6 = 52) :=
by
  sorry

end second_chick_eats_52_l989_98900


namespace least_positive_x_multiple_of_53_l989_98932

theorem least_positive_x_multiple_of_53 :
  ∃ (x : ℕ), (x > 0) ∧ ((2 * x)^2 + 2 * 47 * (2 * x) + 47^2) % 53 = 0 ∧ x = 6 :=
by
  sorry

end least_positive_x_multiple_of_53_l989_98932


namespace cooking_oil_distribution_l989_98963

theorem cooking_oil_distribution (total_oil : ℝ) (oil_A : ℝ) (oil_B : ℝ) (oil_C : ℝ)
    (h_total_oil : total_oil = 3 * 1000) -- Total oil is 3000 milliliters
    (h_A_B : oil_A = oil_B + 200) -- A receives 200 milliliters more than B
    (h_B_C : oil_B = oil_C + 200) -- B receives 200 milliliters more than C
    : oil_B = 1000 :=              -- We need to prove B receives 1000 milliliters
by
  sorry

end cooking_oil_distribution_l989_98963


namespace probability_all_truth_l989_98954

noncomputable def probability_A : ℝ := 0.55
noncomputable def probability_B : ℝ := 0.60
noncomputable def probability_C : ℝ := 0.45
noncomputable def probability_D : ℝ := 0.70

theorem probability_all_truth : 
  (probability_A * probability_B * probability_C * probability_D = 0.10395) := 
by 
  sorry

end probability_all_truth_l989_98954


namespace factors_and_divisors_l989_98941

theorem factors_and_divisors :
  (∃ n : ℕ, 25 = 5 * n) ∧
  (¬(∃ n : ℕ, 209 = 19 * n ∧ ¬ (∃ m : ℕ, 57 = 19 * m))) ∧
  (¬(¬(∃ n : ℕ, 90 = 30 * n) ∧ ¬(∃ m : ℕ, 75 = 30 * m))) ∧
  (¬(∃ n : ℕ, 51 = 17 * n ∧ ¬ (∃ m : ℕ, 68 = 17 * m))) ∧
  (∃ n : ℕ, 171 = 9 * n) :=
by {
  sorry
}

end factors_and_divisors_l989_98941


namespace num_ordered_triples_unique_l989_98981

theorem num_ordered_triples_unique : 
  (∃! (x y z : ℝ), x + y = 2 ∧ xy - z^2 = 1) := 
by 
  sorry 

end num_ordered_triples_unique_l989_98981


namespace ratio_of_blue_marbles_l989_98950

theorem ratio_of_blue_marbles {total_marbles red_marbles orange_marbles blue_marbles : ℕ} 
  (h_total : total_marbles = 24)
  (h_red : red_marbles = 6)
  (h_orange : orange_marbles = 6)
  (h_blue : blue_marbles = total_marbles - red_marbles - orange_marbles) : 
  (blue_marbles : ℚ) / (total_marbles : ℚ) = 1 / 2 := 
by
  sorry -- the proof is omitted as per instructions

end ratio_of_blue_marbles_l989_98950


namespace inequality_solution_l989_98907

theorem inequality_solution (x : ℝ) :
  (3 / 16) + abs (x - 17 / 64) < 7 / 32 ↔ (15 / 64) < x ∧ x < (19 / 64) :=
by
  sorry

end inequality_solution_l989_98907


namespace probability_A_wins_probability_A_wins_2_l989_98926

def binomial (n k : ℕ) := Nat.choose n k

noncomputable def P (n : ℕ) : ℚ := 
  1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n))

theorem probability_A_wins (n : ℕ) : P n = 1/2 * (1 - binomial (2 * n) n / 2 ^ (2 * n)) := 
by sorry

theorem probability_A_wins_2 : P 2 = 5 / 16 := 
by sorry

end probability_A_wins_probability_A_wins_2_l989_98926


namespace cone_sphere_ratio_l989_98961

/-- A right circular cone and a sphere have bases with the same radius r. 
If the volume of the cone is one-third that of the sphere, find the ratio of 
the altitude of the cone to the radius of its base. -/
theorem cone_sphere_ratio (r h : ℝ) (h_pos : 0 < r) 
    (volume_cone : ℝ) (volume_sphere : ℝ)
    (cone_volume_formula : volume_cone = (1 / 3) * π * r^2 * h) 
    (sphere_volume_formula : volume_sphere = (4 / 3) * π * r^3) 
    (volume_relation : volume_cone = (1 / 3) * volume_sphere) : 
    h / r = 4 / 3 :=
by
    sorry

end cone_sphere_ratio_l989_98961
