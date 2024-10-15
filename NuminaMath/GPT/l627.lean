import Mathlib

namespace NUMINAMATH_GPT_verify_statements_l627_62767

theorem verify_statements (a b : ℝ) :
  ( (ab < 0 ∧ (a < 0 ∧ b > 0 ∨ a > 0 ∧ b < 0)) → (a / b = -1)) ∧
  ( (a + b < 0 ∧ ab > 0) → (|2 * a + 3 * b| = -(2 * a + 3 * b)) ) ∧
  ( (|a - b| + a - b = 0) → (b > a) = False ) ∧
  ( (|a| > |b|) → ((a + b) * (a - b) < 0) = False ) :=
by
  sorry

end NUMINAMATH_GPT_verify_statements_l627_62767


namespace NUMINAMATH_GPT_find_integer_in_range_divisible_by_18_l627_62708

theorem find_integer_in_range_divisible_by_18 
  (n : ℕ) (h1 : 900 ≤ n) (h2 : n ≤ 912) (h3 : n % 18 = 0) : n = 900 :=
sorry

end NUMINAMATH_GPT_find_integer_in_range_divisible_by_18_l627_62708


namespace NUMINAMATH_GPT_same_terminal_side_angle_in_range_0_to_2pi_l627_62744

theorem same_terminal_side_angle_in_range_0_to_2pi :
  ∃ k : ℤ, 0 ≤ 2 * k * π + (-4) * π / 3 ∧ 2 * k * π + (-4) * π / 3 ≤ 2 * π ∧
  2 * k * π + (-4) * π / 3 = 2 * π / 3 :=
by
  use 1
  sorry

end NUMINAMATH_GPT_same_terminal_side_angle_in_range_0_to_2pi_l627_62744


namespace NUMINAMATH_GPT_inequality_solution_l627_62762

theorem inequality_solution :
  {x : ℝ // -1 < (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) ∧ (x^2 - 10 * x + 9) / (x^2 - 4 * x + 8) < 1} = 
  {x : ℝ // x > 1/6} :=
sorry

end NUMINAMATH_GPT_inequality_solution_l627_62762


namespace NUMINAMATH_GPT_stratified_sampling_second_class_l627_62751

theorem stratified_sampling_second_class (total_products : ℕ) (first_class : ℕ) (second_class : ℕ) (third_class : ℕ) (sample_size : ℕ) (h_total : total_products = 200) (h_first : first_class = 40) (h_second : second_class = 60) (h_third : third_class = 100) (h_sample : sample_size = 40) :
  (second_class * sample_size) / total_products = 12 :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_second_class_l627_62751


namespace NUMINAMATH_GPT_smallest_value_c_plus_d_l627_62782

noncomputable def problem1 (c d : ℝ) : Prop :=
c > 0 ∧ d > 0 ∧ (c^2 ≥ 12 * d) ∧ ((3 * d)^2 ≥ 4 * c)

theorem smallest_value_c_plus_d : ∃ c d : ℝ, problem1 c d ∧ c + d = 4 / Real.sqrt 3 + 4 / 9 :=
sorry

end NUMINAMATH_GPT_smallest_value_c_plus_d_l627_62782


namespace NUMINAMATH_GPT_part_a_part_b_part_c_l627_62765

-- Definitions for the problem
def hard_problem_ratio_a := 2 / 3
def unsolved_problem_ratio_a := 2 / 3
def well_performing_students_ratio_a := 2 / 3

def hard_problem_ratio_b := 3 / 4
def unsolved_problem_ratio_b := 3 / 4
def well_performing_students_ratio_b := 3 / 4

def hard_problem_ratio_c := 7 / 10
def unsolved_problem_ratio_c := 7 / 10
def well_performing_students_ratio_c := 7 / 10

-- Theorems to prove
theorem part_a : 
  ∃ (hard_problem_ratio_a unsolved_problem_ratio_a well_performing_students_ratio_a : ℚ),
  hard_problem_ratio_a == 2 / 3 ∧
  unsolved_problem_ratio_a == 2 / 3 ∧
  well_performing_students_ratio_a == 2 / 3 →
  (True) := sorry

theorem part_b : 
  ∀ (hard_problem_ratio_b : ℚ),
  hard_problem_ratio_b == 3 / 4 →
  (False) := sorry

theorem part_c : 
  ∀ (hard_problem_ratio_c : ℚ),
  hard_problem_ratio_c == 7 / 10 →
  (False) := sorry

end NUMINAMATH_GPT_part_a_part_b_part_c_l627_62765


namespace NUMINAMATH_GPT_luke_points_per_round_l627_62704

-- Define the total number of points scored 
def totalPoints : ℕ := 8142

-- Define the number of rounds played
def rounds : ℕ := 177

-- Define the points gained per round which we need to prove
def pointsPerRound : ℕ := 46

-- Now, we can state: if Luke played 177 rounds and scored a total of 8142 points, then he gained 46 points per round
theorem luke_points_per_round :
  (totalPoints = 8142) → (rounds = 177) → (totalPoints / rounds = pointsPerRound) := by
  sorry

end NUMINAMATH_GPT_luke_points_per_round_l627_62704


namespace NUMINAMATH_GPT_largest_n_for_divisibility_l627_62735

theorem largest_n_for_divisibility :
  ∃ n : ℕ, (n + 15) ∣ (n^3 + 250) ∧ ∀ m : ℕ, ((m + 15) ∣ (m^3 + 250)) → (m ≤ 10) → (n = 10) :=
by {
  sorry
}

end NUMINAMATH_GPT_largest_n_for_divisibility_l627_62735


namespace NUMINAMATH_GPT_necessary_and_sufficient_condition_l627_62726

variable (p q : Prop)

theorem necessary_and_sufficient_condition (h1 : p → q) (h2 : q → p) : (p ↔ q) :=
by 
  sorry

end NUMINAMATH_GPT_necessary_and_sufficient_condition_l627_62726


namespace NUMINAMATH_GPT_greatest_partition_l627_62756

-- Define the condition on the partitions of the positive integers
def satisfies_condition (A : ℕ → Prop) (n : ℕ) : Prop :=
∃ a b : ℕ, a ≠ b ∧ A a ∧ A b ∧ a + b = n

-- Define what it means for k subsets to meet the requirements
def partition_satisfies (k : ℕ) : Prop :=
∃ A : ℕ → ℕ → Prop,
  (∀ i : ℕ, i < k → ∀ n ≥ 15, satisfies_condition (A i) n)

-- Our conjecture is that k can be at most 3 for the given condition
theorem greatest_partition (k : ℕ) : k ≤ 3 :=
sorry

end NUMINAMATH_GPT_greatest_partition_l627_62756


namespace NUMINAMATH_GPT_cosine_double_angle_tangent_l627_62759

theorem cosine_double_angle_tangent (θ : ℝ) (h : Real.tan θ = -1/3) : Real.cos (2 * θ) = 4/5 :=
by
  sorry

end NUMINAMATH_GPT_cosine_double_angle_tangent_l627_62759


namespace NUMINAMATH_GPT_cleaning_time_l627_62710

def lara_rate := 1 / 4
def chris_rate := 1 / 6
def combined_rate := lara_rate + chris_rate

theorem cleaning_time (t : ℝ) : 
  (combined_rate * (t - 2) = 1) ↔ (t = 22 / 5) :=
by
  sorry

end NUMINAMATH_GPT_cleaning_time_l627_62710


namespace NUMINAMATH_GPT_tangent_line_at_M_l627_62713

noncomputable def isOnCircle (x y : ℝ) : Prop := x^2 + y^2 = 1

noncomputable def M : ℝ × ℝ := (Real.sqrt 2 / 2, Real.sqrt 2 / 2)

theorem tangent_line_at_M (hM : isOnCircle (M.1) (M.2)) : (∀ x y, M.1 = x ∨ M.2 = y → x + y = Real.sqrt 2) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_at_M_l627_62713


namespace NUMINAMATH_GPT_gcd_x_y_not_8_l627_62711

theorem gcd_x_y_not_8 (x y : ℕ) (hx : x > 0) (hy : y = x^2 + 8) : ¬ ∃ d, d = 8 ∧ d ∣ x ∧ d ∣ y :=
by
  sorry

end NUMINAMATH_GPT_gcd_x_y_not_8_l627_62711


namespace NUMINAMATH_GPT_largest_divisor_39_l627_62720

theorem largest_divisor_39 (m : ℕ) (hm : 0 < m) (h : 39 ∣ m ^ 2) : 39 ∣ m :=
by sorry

end NUMINAMATH_GPT_largest_divisor_39_l627_62720


namespace NUMINAMATH_GPT_proof_x1_x2_squared_l627_62745

theorem proof_x1_x2_squared (x1 x2 : ℝ) (h1 : (Real.exp 1 * x1)^x2 = (Real.exp 1 * x2)^x1)
  (h2 : 0 < x1) (h3 : 0 < x2) (h4 : x1 ≠ x2) : x1^2 + x2^2 > 2 :=
sorry

end NUMINAMATH_GPT_proof_x1_x2_squared_l627_62745


namespace NUMINAMATH_GPT_history_only_students_l627_62728

theorem history_only_students 
  (total_students : ℕ)
  (history_students stats_students physics_students chem_students : ℕ) 
  (hist_stats hist_phys hist_chem stats_phys stats_chem phys_chem all_four : ℕ) 
  (h1 : total_students = 500)
  (h2 : history_students = 150)
  (h3 : stats_students = 130)
  (h4 : physics_students = 120)
  (h5 : chem_students = 100)
  (h6 : hist_stats = 60)
  (h7 : hist_phys = 50)
  (h8 : hist_chem = 40)
  (h9 : stats_phys = 35)
  (h10 : stats_chem = 30)
  (h11 : phys_chem = 25)
  (h12 : all_four = 20) : 
  (history_students - hist_stats - hist_phys - hist_chem + all_four) = 20 := 
by 
  sorry

end NUMINAMATH_GPT_history_only_students_l627_62728


namespace NUMINAMATH_GPT_difference_max_min_planes_l627_62797

open Set

-- Defining the regular tetrahedron and related concepts
noncomputable def tetrahedron := Unit -- Placeholder for the tetrahedron

def union_faces (T : Unit) : Set Point := sorry -- Placeholder for union of faces definition

noncomputable def simple_trace (p : Plane) (T : Unit) : Set Point := sorry -- Placeholder for planes intersecting faces

-- Calculating number of planes
def maximum_planes (T : Unit) : Nat :=
  4 -- One for each face of the tetrahedron

def minimum_planes (T : Unit) : Nat :=
  2 -- Each plane covers traces on two adjacent faces if oriented appropriately

-- Statement of the problem
theorem difference_max_min_planes (T : Unit) :
  maximum_planes T - minimum_planes T = 2 :=
by
  -- Proof skipped
  sorry

end NUMINAMATH_GPT_difference_max_min_planes_l627_62797


namespace NUMINAMATH_GPT_people_per_seat_l627_62777

def ferris_wheel_seats : ℕ := 4
def total_people_riding : ℕ := 20

theorem people_per_seat : total_people_riding / ferris_wheel_seats = 5 := by
  sorry

end NUMINAMATH_GPT_people_per_seat_l627_62777


namespace NUMINAMATH_GPT_basil_pots_count_l627_62769

theorem basil_pots_count (B : ℕ) (h1 : 9 * 18 + 6 * 30 + 4 * B = 354) : B = 3 := 
by 
  -- This is just the signature of the theorem. The proof is omitted.
  sorry

end NUMINAMATH_GPT_basil_pots_count_l627_62769


namespace NUMINAMATH_GPT_arithmetic_sequence_identity_l627_62702

theorem arithmetic_sequence_identity (a : ℕ → ℝ) (d : ℝ)
    (h_arith : ∀ n, a (n + 1) = a 1 + n * d)
    (h_sum : a 4 + a 7 + a 10 = 30) :
    a 1 - a 3 - a 6 - a 8 - a 11 + a 13 = -20 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_identity_l627_62702


namespace NUMINAMATH_GPT_chelsea_sugar_problem_l627_62709

variable (initial_sugar : ℕ)
variable (num_bags : ℕ)
variable (sugar_lost_fraction : ℕ)

def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (sugar_lost_fraction : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let sugar_lost := sugar_per_bag / sugar_lost_fraction
  let remaining_bags_sugar := (num_bags - 1) * sugar_per_bag
  remaining_bags_sugar + (sugar_per_bag - sugar_lost)

theorem chelsea_sugar_problem : 
  remaining_sugar 24 4 2 = 21 :=
by
  sorry

end NUMINAMATH_GPT_chelsea_sugar_problem_l627_62709


namespace NUMINAMATH_GPT_total_gymnasts_l627_62796

theorem total_gymnasts (n : ℕ) : 
  (∃ (t : ℕ) (c : t = 4) (h : n * (n-1) / 2 + 4 * 6 = 595), n = 34) :=
by {
  -- skipping the detailed proof here, just ensuring the problem is stated as a theorem
  sorry
}

end NUMINAMATH_GPT_total_gymnasts_l627_62796


namespace NUMINAMATH_GPT_fraction_comparison_l627_62714

theorem fraction_comparison :
  let d := 0.33333333
  let f := (1 : ℚ) / 3
  f > d ∧ f - d = 1 / (3 * (10^8 : ℚ)) :=
by
  sorry

end NUMINAMATH_GPT_fraction_comparison_l627_62714


namespace NUMINAMATH_GPT_g_value_l627_62712

theorem g_value (g : ℝ → ℝ)
  (h0 : g 0 = 0)
  (h_mono : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x < y → y ≤ 1 → g x ≤ g y)
  (h_symm : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (1 - x) = 1 - g x)
  (h_prop : ∀ x : ℝ, 0 ≤ x → x ≤ 1 → g (x / 4) = g x / 3) :
  g (2 / 5) = 1 / 2 :=
sorry

end NUMINAMATH_GPT_g_value_l627_62712


namespace NUMINAMATH_GPT_three_distinct_real_solutions_l627_62747

theorem three_distinct_real_solutions (b c : ℝ):
  (∀ x : ℝ, x^2 + b * |x| + c = 0 → x = 0) ∧ (∃! x : ℝ, x^2 + b * |x| + c = 0) →
  b < 0 ∧ c = 0 :=
by {
  sorry
}

end NUMINAMATH_GPT_three_distinct_real_solutions_l627_62747


namespace NUMINAMATH_GPT_keep_oranges_per_day_l627_62772

def total_oranges_harvested (sacks_per_day : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  sacks_per_day * oranges_per_sack

def oranges_discarded (discarded_sacks : ℕ) (oranges_per_sack : ℕ) : ℕ :=
  discarded_sacks * oranges_per_sack

def oranges_kept_per_day (total_oranges : ℕ) (discarded_oranges : ℕ) : ℕ :=
  total_oranges - discarded_oranges

theorem keep_oranges_per_day 
  (sacks_per_day : ℕ)
  (oranges_per_sack : ℕ)
  (discarded_sacks : ℕ)
  (h1 : sacks_per_day = 76)
  (h2 : oranges_per_sack = 50)
  (h3 : discarded_sacks = 64) :
  oranges_kept_per_day (total_oranges_harvested sacks_per_day oranges_per_sack) 
  (oranges_discarded discarded_sacks oranges_per_sack) = 600 :=
by
  sorry

end NUMINAMATH_GPT_keep_oranges_per_day_l627_62772


namespace NUMINAMATH_GPT_min_max_value_in_interval_l627_62719

theorem min_max_value_in_interval : ∀ (x : ℝ),
  -2 < x ∧ x < 5 →
  ∃ (y : ℝ), (y = -1.5 ∨ y = 1.5) ∧ y = (x^2 - 4 * x + 6) / (2 * x - 4) := 
by sorry

end NUMINAMATH_GPT_min_max_value_in_interval_l627_62719


namespace NUMINAMATH_GPT_quadratic_conditions_l627_62764

open Polynomial

noncomputable def exampleQuadratic (x : ℝ) : ℝ :=
-2 * x^2 + 12 * x - 10

theorem quadratic_conditions :
  (exampleQuadratic 1 = 0) ∧ (exampleQuadratic 5 = 0) ∧ (exampleQuadratic 3 = 8) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_conditions_l627_62764


namespace NUMINAMATH_GPT_problem_a_problem_b_l627_62705

-- Definition for real roots condition in problem A
def has_real_roots (k : ℝ) : Prop :=
  let a := 1
  let b := -3
  let c := k
  b^2 - 4 * a * c ≥ 0

-- Problem A: Proving the range of k
theorem problem_a (k : ℝ) : has_real_roots k ↔ k ≤ 9 / 4 :=
by
  sorry

-- Definition for a quadratic equation having a given root
def has_root (a b c : ℝ) (x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

-- Problem B: Proving the value of m given a common root condition
theorem problem_b (m : ℝ) : 
  (has_root 1 (-3) 2 1 ∧ has_root (m-1) 1 (m-3) 1) ↔ m = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_problem_a_problem_b_l627_62705


namespace NUMINAMATH_GPT_even_sum_probability_l627_62757

-- Conditions
def prob_even_first_wheel : ℚ := 1 / 4
def prob_odd_first_wheel : ℚ := 3 / 4
def prob_even_second_wheel : ℚ := 2 / 3
def prob_odd_second_wheel : ℚ := 1 / 3

-- Statement: Theorem that the probability of the sum being even is 5/12
theorem even_sum_probability : 
  (prob_even_first_wheel * prob_even_second_wheel) + 
  (prob_odd_first_wheel * prob_odd_second_wheel) = 5 / 12 :=
by
  -- Proof steps would go here
  sorry

end NUMINAMATH_GPT_even_sum_probability_l627_62757


namespace NUMINAMATH_GPT_roses_in_february_l627_62781

-- Define initial counts of roses
def roses_oct : ℕ := 80
def roses_nov : ℕ := 98
def roses_dec : ℕ := 128
def roses_jan : ℕ := 170

-- Define the differences
def diff_on : ℕ := roses_nov - roses_oct -- 18
def diff_nd : ℕ := roses_dec - roses_nov -- 30
def diff_dj : ℕ := roses_jan - roses_dec -- 42

-- The increment in differences
def inc : ℕ := diff_nd - diff_on -- 12

-- Express the difference from January to February
def diff_jf : ℕ := diff_dj + inc -- 54

-- The number of roses in February
def roses_feb : ℕ := roses_jan + diff_jf -- 224

theorem roses_in_february : roses_feb = 224 := by
  -- Provide the expected value for Lean to verify
  sorry

end NUMINAMATH_GPT_roses_in_february_l627_62781


namespace NUMINAMATH_GPT_driver_license_advantage_l627_62721

def AdvantageousReasonsForEarlyLicenseObtaining 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) : Prop :=
  ∀ age1 age2 : ℕ, (eligible age1 ∧ eligible age2 ∧ age1 < age2) →
  (effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1) →
  effectiveInsurance age1 ∧ rentalCarFlexibility age1 ∧ employmentOpportunity age1

theorem driver_license_advantage 
  (eligible : ℕ → Prop)
  (effectiveInsurance : ℕ → Prop)
  (rentalCarFlexibility : ℕ → Prop)
  (employmentOpportunity : ℕ → Prop) :
  AdvantageousReasonsForEarlyLicenseObtaining eligible effectiveInsurance rentalCarFlexibility employmentOpportunity :=
by
  sorry

end NUMINAMATH_GPT_driver_license_advantage_l627_62721


namespace NUMINAMATH_GPT_max_value_abs_x_sub_3y_l627_62718

theorem max_value_abs_x_sub_3y 
  (x y : ℝ)
  (h1 : y ≥ x)
  (h2 : x + 3 * y ≤ 4)
  (h3 : x ≥ -2) : 
  ∃ z, z = |x - 3 * y| ∧ ∀ (x y : ℝ), (y ≥ x) → (x + 3 * y ≤ 4) → (x ≥ -2) → |x - 3 * y| ≤ 4 :=
sorry

end NUMINAMATH_GPT_max_value_abs_x_sub_3y_l627_62718


namespace NUMINAMATH_GPT_roger_cookie_price_l627_62707

noncomputable def price_per_roger_cookie (A_cookies: ℕ) (A_price_per_cookie: ℕ) (A_area_per_cookie: ℕ) (R_cookies: ℕ) (R_area_per_cookie: ℕ): ℕ :=
  by
  let A_total_earnings := A_cookies * A_price_per_cookie
  let R_total_area := A_cookies * A_area_per_cookie
  let price_per_R_cookie := A_total_earnings / R_cookies
  exact price_per_R_cookie
  
theorem roger_cookie_price {A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie : ℕ}
  (h1 : A_cookies = 12)
  (h2 : A_price_per_cookie = 60)
  (h3 : A_area_per_cookie = 12)
  (h4 : R_cookies = 18) -- assumed based on area calculation 144 / 8 (we need this input to match solution context)
  (h5 : R_area_per_cookie = 8) :
  price_per_roger_cookie A_cookies A_price_per_cookie A_area_per_cookie R_cookies R_area_per_cookie = 40 :=
  by
  sorry

end NUMINAMATH_GPT_roger_cookie_price_l627_62707


namespace NUMINAMATH_GPT_find_unique_number_l627_62799

def is_three_digit_number (N : ℕ) : Prop := 100 ≤ N ∧ N < 1000

def nonzero_digits (A B C : ℕ) : Prop := A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0

def digits_of_number (N A B C : ℕ) : Prop := N = 100 * A + 10 * B + C

def product (N A B : ℕ) := N * (10 * A + B) * A

def divides (n m : ℕ) := ∃ k, n * k = m

theorem find_unique_number (N A B C : ℕ) (h1 : is_three_digit_number N)
    (h2 : nonzero_digits A B C) (h3 : digits_of_number N A B C)
    (h4 : divides 1000 (product N A B)) : N = 875 :=
sorry

end NUMINAMATH_GPT_find_unique_number_l627_62799


namespace NUMINAMATH_GPT_packages_ratio_l627_62724

theorem packages_ratio (packages_yesterday packages_today : ℕ)
  (h1 : packages_yesterday = 80)
  (h2 : packages_today + packages_yesterday = 240) :
  (packages_today / packages_yesterday) = 2 :=
by
  sorry

end NUMINAMATH_GPT_packages_ratio_l627_62724


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l627_62770

-- Define the arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Define the problem conditions
def problem_conditions (a : ℕ → ℝ) : Prop :=
  (a 3 + a 8 = 3) ∧ is_arithmetic_sequence a

-- State the theorem to be proved
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h : problem_conditions a) : a 1 + a 10 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l627_62770


namespace NUMINAMATH_GPT_bus_return_trip_fraction_l627_62738

theorem bus_return_trip_fraction :
  (3 / 4 * 200 + x * 200 = 310) → (x = 4 / 5) := by
  sorry

end NUMINAMATH_GPT_bus_return_trip_fraction_l627_62738


namespace NUMINAMATH_GPT_total_hours_worked_l627_62703

theorem total_hours_worked (Amber_hours : ℕ) (h_Amber : Amber_hours = 12) 
  (Armand_hours : ℕ) (h_Armand : Armand_hours = Amber_hours / 3)
  (Ella_hours : ℕ) (h_Ella : Ella_hours = Amber_hours * 2) : 
  Amber_hours + Armand_hours + Ella_hours = 40 :=
by
  rw [h_Amber, h_Armand, h_Ella]
  norm_num
  sorry

end NUMINAMATH_GPT_total_hours_worked_l627_62703


namespace NUMINAMATH_GPT_determine_y_l627_62786

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0)
variable (hxy : x = 2 + (1 / y))
variable (hyx : y = 2 + (2 / x))

theorem determine_y (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x = 2 + (1 / y)) (hyx : y = 2 + (2 / x)) :
  y = (5 + Real.sqrt 41) / 4 ∨ y = (5 - Real.sqrt 41) / 4 := 
sorry

end NUMINAMATH_GPT_determine_y_l627_62786


namespace NUMINAMATH_GPT_minimize_quadratic_l627_62754

theorem minimize_quadratic (c : ℝ) : ∃ b : ℝ, (∀ x : ℝ, 3 * x^2 + 2 * x + c ≥ 3 * b^2 + 2 * b + c) ∧ b = -1/3 :=
by
  sorry

end NUMINAMATH_GPT_minimize_quadratic_l627_62754


namespace NUMINAMATH_GPT_lisa_eats_one_candy_on_other_days_l627_62737

def candies_total : ℕ := 36
def candies_per_day_on_mondays_and_wednesdays : ℕ := 2
def weeks : ℕ := 4
def days_in_a_week : ℕ := 7
def mondays_and_wednesdays_in_4_weeks : ℕ := 2 * weeks
def total_candies_mondays_and_wednesdays : ℕ := mondays_and_wednesdays_in_4_weeks * candies_per_day_on_mondays_and_wednesdays
def total_other_candies : ℕ := candies_total - total_candies_mondays_and_wednesdays
def total_other_days : ℕ := weeks * (days_in_a_week - 2)
def candies_per_other_day : ℕ := total_other_candies / total_other_days

theorem lisa_eats_one_candy_on_other_days :
  candies_per_other_day = 1 :=
by
  -- Prove the theorem with conditions defined
  sorry

end NUMINAMATH_GPT_lisa_eats_one_candy_on_other_days_l627_62737


namespace NUMINAMATH_GPT_smallest_coin_remainder_l627_62755

theorem smallest_coin_remainder
  (c : ℕ)
  (h1 : c % 8 = 6)
  (h2 : c % 7 = 5)
  (h3 : ∀ d : ℕ, (d % 8 = 6) → (d % 7 = 5) → d ≥ c) :
  c % 9 = 2 :=
sorry

end NUMINAMATH_GPT_smallest_coin_remainder_l627_62755


namespace NUMINAMATH_GPT_a_10_is_100_l627_62773

-- Define the sequence a_n as a function from ℕ+ (the positive naturals) to ℤ
axiom a : ℕ+ → ℤ

-- Given assumptions
axiom seq_relation : ∀ m n : ℕ+, a m + a n = a (m + n) - 2 * m.val * n.val
axiom a1 : a 1 = 1

-- Goal statement
theorem a_10_is_100 : a 10 = 100 :=
by
  -- proof goes here, this is just the statement
  sorry

end NUMINAMATH_GPT_a_10_is_100_l627_62773


namespace NUMINAMATH_GPT_value_of_a3_a5_l627_62736

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = a n * r

theorem value_of_a3_a5 
  (a : ℕ → ℝ) 
  (h_geom : geometric_sequence a) 
  (h_pos : ∀ n, a n > 0) 
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : 
  a 3 + a 5 = 5 :=
  sorry

end NUMINAMATH_GPT_value_of_a3_a5_l627_62736


namespace NUMINAMATH_GPT_true_statements_about_f_l627_62791

noncomputable def f (x : ℝ) := 2 * abs (Real.cos x) * Real.sin x + Real.sin (2 * x)

theorem true_statements_about_f :
  (∀ x y : ℝ, -π/4 ≤ x ∧ x < y ∧ y ≤ π/4 → f x < f y) ∧
  (∀ y : ℝ, -2 ≤ y ∧ y ≤ 2 → (∃ x : ℝ, f x = y)) :=
by
  sorry

end NUMINAMATH_GPT_true_statements_about_f_l627_62791


namespace NUMINAMATH_GPT_solution_is_correct_l627_62753

-- Define the conditions of the problem.
variable (x y z : ℝ)

-- The system of equations given in the problem
def system_of_equations (x y z : ℝ) :=
  (1/x + 1/(y+z) = 6/5) ∧
  (1/y + 1/(x+z) = 3/4) ∧
  (1/z + 1/(x+y) = 2/3)

-- The desired solution
def solution (x y z : ℝ) := x = 2 ∧ y = 3 ∧ z = 1

-- The theorem to prove
theorem solution_is_correct (h : system_of_equations x y z) : solution x y z :=
sorry

end NUMINAMATH_GPT_solution_is_correct_l627_62753


namespace NUMINAMATH_GPT_avg_payment_correct_l627_62743

def first_payment : ℕ := 410
def additional_amount : ℕ := 65
def num_first_payments : ℕ := 8
def num_remaining_payments : ℕ := 44
def total_installments : ℕ := num_first_payments + num_remaining_payments

def total_first_payments : ℕ := num_first_payments * first_payment
def remaining_payment : ℕ := first_payment + additional_amount
def total_remaining_payments : ℕ := num_remaining_payments * remaining_payment

def total_payment : ℕ := total_first_payments + total_remaining_payments
def average_payment : ℚ := total_payment / total_installments

theorem avg_payment_correct : average_payment = 465 := by
  sorry

end NUMINAMATH_GPT_avg_payment_correct_l627_62743


namespace NUMINAMATH_GPT_Tony_total_payment_l627_62775

-- Defining the cost of items
def lego_block_cost : ℝ := 250
def toy_sword_cost : ℝ := 120
def play_dough_cost : ℝ := 35

-- Quantities of each item
def total_lego_blocks : ℕ := 3
def total_toy_swords : ℕ := 5
def total_play_doughs : ℕ := 10

-- Quantities purchased on each day
def first_day_lego_blocks : ℕ := 2
def first_day_toy_swords : ℕ := 3
def second_day_lego_blocks : ℕ := total_lego_blocks - first_day_lego_blocks
def second_day_toy_swords : ℕ := total_toy_swords - first_day_toy_swords
def second_day_play_doughs : ℕ := total_play_doughs

-- Discounts and tax rates
def first_day_discount : ℝ := 0.20
def second_day_discount : ℝ := 0.10
def sales_tax : ℝ := 0.05

-- Calculating first day purchase amounts
def first_day_cost_before_discount : ℝ := (first_day_lego_blocks * lego_block_cost) + (first_day_toy_swords * toy_sword_cost)
def first_day_discount_amount : ℝ := first_day_cost_before_discount * first_day_discount
def first_day_cost_after_discount : ℝ := first_day_cost_before_discount - first_day_discount_amount
def first_day_sales_tax_amount : ℝ := first_day_cost_after_discount * sales_tax
def first_day_total_cost : ℝ := first_day_cost_after_discount + first_day_sales_tax_amount

-- Calculating second day purchase amounts
def second_day_cost_before_discount : ℝ := (second_day_lego_blocks * lego_block_cost) + (second_day_toy_swords * toy_sword_cost) + 
                                           (second_day_play_doughs * play_dough_cost)
def second_day_discount_amount : ℝ := second_day_cost_before_discount * second_day_discount
def second_day_cost_after_discount : ℝ := second_day_cost_before_discount - second_day_discount_amount
def second_day_sales_tax_amount : ℝ := second_day_cost_after_discount * sales_tax
def second_day_total_cost : ℝ := second_day_cost_after_discount + second_day_sales_tax_amount

-- Total cost
def total_cost : ℝ := first_day_total_cost + second_day_total_cost

-- Lean theorem statement
theorem Tony_total_payment : total_cost = 1516.20 := by
  sorry

end NUMINAMATH_GPT_Tony_total_payment_l627_62775


namespace NUMINAMATH_GPT_system_of_equations_correct_l627_62798

theorem system_of_equations_correct (x y : ℤ) :
  (8 * x - 3 = y) ∧ (7 * x + 4 = y) :=
sorry

end NUMINAMATH_GPT_system_of_equations_correct_l627_62798


namespace NUMINAMATH_GPT_salesmans_profit_l627_62766

-- Define the initial conditions and given values
def backpacks_bought : ℕ := 72
def cost_price : ℕ := 1080
def swap_meet_sales : ℕ := 25
def swap_meet_price : ℕ := 20
def department_store_sales : ℕ := 18
def department_store_price : ℕ := 30
def online_sales : ℕ := 12
def online_price : ℕ := 28
def shipping_expenses : ℕ := 40
def local_market_price : ℕ := 24

-- Calculate the total revenue from each channel
def swap_meet_revenue : ℕ := swap_meet_sales * swap_meet_price
def department_store_revenue : ℕ := department_store_sales * department_store_price
def online_revenue : ℕ := (online_sales * online_price) - shipping_expenses

-- Calculate remaining backpacks and local market revenue
def backpacks_sold : ℕ := swap_meet_sales + department_store_sales + online_sales
def backpacks_left : ℕ := backpacks_bought - backpacks_sold
def local_market_revenue : ℕ := backpacks_left * local_market_price

-- Calculate total revenue and profit
def total_revenue : ℕ := swap_meet_revenue + department_store_revenue + online_revenue + local_market_revenue
def profit : ℕ := total_revenue - cost_price

-- State the theorem for the salesman's profit
theorem salesmans_profit : profit = 664 := by
  sorry

end NUMINAMATH_GPT_salesmans_profit_l627_62766


namespace NUMINAMATH_GPT_sum_factors_30_less_15_l627_62793

theorem sum_factors_30_less_15 : (1 + 2 + 3 + 5 + 6 + 10) = 27 := by
  sorry

end NUMINAMATH_GPT_sum_factors_30_less_15_l627_62793


namespace NUMINAMATH_GPT_surface_area_of_rectangular_solid_is_334_l627_62788

theorem surface_area_of_rectangular_solid_is_334
  (l w h : ℕ)
  (h_l_prime : Prime l)
  (h_w_prime : Prime w)
  (h_h_prime : Prime h)
  (volume_eq_385 : l * w * h = 385) : 
  2 * (l * w + l * h + w * h) = 334 := 
sorry

end NUMINAMATH_GPT_surface_area_of_rectangular_solid_is_334_l627_62788


namespace NUMINAMATH_GPT_range_of_a_l627_62778

noncomputable def f (x a : ℝ) : ℝ := Real.exp x - a * x - 1
noncomputable def g (x : ℝ) : ℝ := Real.log (Real.exp x - 1) - Real.log x

theorem range_of_a (a : ℝ) :
  (∃ x0 : ℝ, 0 < x0 ∧ f (g x0) a > f x0 a) ↔ 1 < a := sorry

end NUMINAMATH_GPT_range_of_a_l627_62778


namespace NUMINAMATH_GPT_geometric_sequence_exists_l627_62732

theorem geometric_sequence_exists 
  (a r : ℚ)
  (h1 : a = 3)
  (h2 : a * r = 8 / 9)
  (h3 : a * r^2 = 32 / 81) : 
  r = 8 / 27 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_exists_l627_62732


namespace NUMINAMATH_GPT_total_students_is_2000_l627_62752

theorem total_students_is_2000
  (S : ℝ) 
  (h1 : 0.10 * S = chess_students) 
  (h2 : 0.50 * chess_students = swimming_students) 
  (h3 : swimming_students = 100) 
  (chess_students swimming_students : ℝ) 
  : S = 2000 := 
by 
  sorry

end NUMINAMATH_GPT_total_students_is_2000_l627_62752


namespace NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l627_62771

theorem sum_of_reciprocals_of_roots (r1 r2 : ℝ) (h1 : r1 * r2 = 7) (h2 : r1 + r2 = 16) :
  (1 / r1) + (1 / r2) = 16 / 7 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_of_roots_l627_62771


namespace NUMINAMATH_GPT_spherical_to_rectangular_coords_l627_62795

theorem spherical_to_rectangular_coords :
  let ρ := 5
  let θ := Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  x = 5 * Real.sin (Real.pi / 3) * Real.cos (Real.pi / 4) ∧
  y = 5 * Real.sin (Real.pi / 3) * Real.sin (Real.pi / 4) ∧
  z = 5 * Real.cos (Real.pi / 3) ∧
  x = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  y = 5 * (Real.sqrt 3) / 2 * (Real.sqrt 2) / 2 ∧
  z = 2.5 ∧
  (x = (5 * Real.sqrt 6) / 4 ∧ y = (5 * Real.sqrt 6) / 4 ∧ z = 2.5) :=
by {
  sorry
}

end NUMINAMATH_GPT_spherical_to_rectangular_coords_l627_62795


namespace NUMINAMATH_GPT_An_nonempty_finite_l627_62717

def An (n : ℕ) : Set (ℕ × ℕ) :=
  { p : ℕ × ℕ | ∃ (k : ℕ), ∃ (a : ℕ), ∃ (b : ℕ), a = Nat.sqrt (p.1^2 + p.2 + n) ∧ b = Nat.sqrt (p.2^2 + p.1 + n) ∧ k = a + b }

theorem An_nonempty_finite (n : ℕ) (h : n ≥ 1) : Set.Nonempty (An n) ∧ Set.Finite (An n) :=
by
  sorry -- The proof goes here

end NUMINAMATH_GPT_An_nonempty_finite_l627_62717


namespace NUMINAMATH_GPT_medicine_duration_l627_62748

theorem medicine_duration (days_per_third_pill : ℕ) (pills : ℕ) (days_per_month : ℕ)
  (h1 : days_per_third_pill = 3)
  (h2 : pills = 90)
  (h3 : days_per_month = 30) :
  ((pills * (days_per_third_pill * 3)) / days_per_month) = 27 :=
sorry

end NUMINAMATH_GPT_medicine_duration_l627_62748


namespace NUMINAMATH_GPT_express_y_in_terms_of_x_l627_62784

theorem express_y_in_terms_of_x (x y : ℝ) (h : x + 2 * y = 6) : y = (-x + 6) / 2 := 
by { sorry }

end NUMINAMATH_GPT_express_y_in_terms_of_x_l627_62784


namespace NUMINAMATH_GPT_reduced_price_is_correct_l627_62750

-- Definitions for the conditions in the problem
def original_price_per_dozen (P : ℝ) : Prop :=
∀ (X : ℝ), X * P = 40.00001

def reduced_price_per_dozen (P R : ℝ) : Prop :=
R = 0.60 * P

def bananas_purchased_additional (P R : ℝ) : Prop :=
∀ (X Y : ℝ), (Y = X + (64 / 12)) → (X * P = Y * R) 

-- Assertion of the proof problem
theorem reduced_price_is_correct : 
  ∃ (R : ℝ), 
  (∀ P, original_price_per_dozen P ∧ reduced_price_per_dozen P R ∧ bananas_purchased_additional P R) → 
  R = 3.00000075 := 
by sorry

end NUMINAMATH_GPT_reduced_price_is_correct_l627_62750


namespace NUMINAMATH_GPT_mean_score_of_seniors_l627_62725

variable (s n : ℕ)  -- Number of seniors and non-seniors
variable (m_s m_n : ℝ)  -- Mean scores of seniors and non-seniors
variable (total_mean : ℝ) -- Mean score of all students
variable (total_students : ℕ) -- Total number of students

theorem mean_score_of_seniors :
  total_students = 100 → total_mean = 100 →
  n = 3 * s / 2 →
  s * m_s + n * m_n = total_students * total_mean →
  m_s = (3 * m_n / 2) →
  m_s = 125 :=
by
  intros
  sorry

end NUMINAMATH_GPT_mean_score_of_seniors_l627_62725


namespace NUMINAMATH_GPT_school_stats_l627_62739

-- Defining the conditions
def girls_grade6 := 315
def boys_grade6 := 309
def girls_grade7 := 375
def boys_grade7 := 341
def drama_club_members := 80
def drama_club_boys_percent := 30 / 100

-- Calculate the derived numbers
def students_grade6 := girls_grade6 + boys_grade6
def students_grade7 := girls_grade7 + boys_grade7
def total_students := students_grade6 + students_grade7
def drama_club_boys := drama_club_boys_percent * drama_club_members
def drama_club_girls := drama_club_members - drama_club_boys

-- Theorem
theorem school_stats :
  total_students = 1340 ∧
  drama_club_girls = 56 ∧
  boys_grade6 = 309 ∧
  boys_grade7 = 341 :=
by
  -- We provide the proof steps inline with sorry placeholders.
  -- In practice, these would be filled with appropriate proofs.
  sorry

end NUMINAMATH_GPT_school_stats_l627_62739


namespace NUMINAMATH_GPT_average_of_combined_samples_l627_62727

theorem average_of_combined_samples 
  (a : Fin 10 → ℝ)
  (b : Fin 10 → ℝ)
  (ave_a : ℝ := (1 / 10) * (Finset.univ.sum (fun i => a i)))
  (ave_b : ℝ := (1 / 10) * (Finset.univ.sum (fun i => b i)))
  (combined_average : ℝ := (1 / 20) * (Finset.univ.sum (fun i => a i) + Finset.univ.sum (fun i => b i))) :
  combined_average = (1 / 2) * (ave_a + ave_b) := 
  by
    sorry

end NUMINAMATH_GPT_average_of_combined_samples_l627_62727


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l627_62740

theorem quadratic_inequality_solution (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) :=
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l627_62740


namespace NUMINAMATH_GPT_factor_of_polynomial_l627_62776

theorem factor_of_polynomial (x : ℝ) : 
  (x^2 - 2*x + 2) ∣ (29 * 39 * x^4 + 4) :=
sorry

end NUMINAMATH_GPT_factor_of_polynomial_l627_62776


namespace NUMINAMATH_GPT_cheaper_store_price_in_cents_l627_62723

/-- List price of Book Y -/
def list_price : ℝ := 24.95

/-- Discount at Readers' Delight -/
def readers_delight_discount : ℝ := 5

/-- Discount rate at Book Bargains -/
def book_bargains_discount_rate : ℝ := 0.2

/-- Calculate sale price at Readers' Delight -/
def sale_price_readers_delight : ℝ := list_price - readers_delight_discount

/-- Calculate sale price at Book Bargains -/
def sale_price_book_bargains : ℝ := list_price * (1 - book_bargains_discount_rate)

/-- Difference in price between Book Bargains and Readers' Delight in cents -/
theorem cheaper_store_price_in_cents :
  (sale_price_book_bargains - sale_price_readers_delight) * 100 = 1 :=
by
  sorry

end NUMINAMATH_GPT_cheaper_store_price_in_cents_l627_62723


namespace NUMINAMATH_GPT_pears_for_36_bananas_l627_62758

theorem pears_for_36_bananas (p : ℕ) (bananas : ℕ) (pears : ℕ) (h : 9 * pears = 6 * bananas) :
  36 * pears = 9 * 24 :=
by
  sorry

end NUMINAMATH_GPT_pears_for_36_bananas_l627_62758


namespace NUMINAMATH_GPT_shaded_area_l627_62779

theorem shaded_area (r1 r2 : ℝ) (h1 : r2 = 3 * r1) (h2 : r1 = 2) : 
  π * (r2 ^ 2) - π * (r1 ^ 2) = 32 * π :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_l627_62779


namespace NUMINAMATH_GPT_false_propositions_count_l627_62792

-- Definitions of the propositions
def proposition1 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition2 (A B : Prop) (P : Prop) : Prop :=
  P ∧ ¬ P

def proposition3 (A B : Prop) : Prop :=
  ¬ (A ∧ B)

def proposition4 (A B : Prop) : Prop :=
  A ∧ B

-- Theorem to prove the total number of false propositions
theorem false_propositions_count (A B : Prop) (P1 P2 P3 P4 : Prop) :
  ¬ (proposition1 A B P1) ∧ ¬ (proposition2 A B P2) ∧ ¬ (proposition3 A B) ∧ proposition4 A B → 3 = 3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_false_propositions_count_l627_62792


namespace NUMINAMATH_GPT_donuts_selection_l627_62733

def number_of_selections (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem donuts_selection : number_of_selections 6 4 = 84 := by
  sorry

end NUMINAMATH_GPT_donuts_selection_l627_62733


namespace NUMINAMATH_GPT_find_n_pos_int_l627_62700

theorem find_n_pos_int (n : ℕ) (h1 : n ^ 3 + 2 * n ^ 2 + 9 * n + 8 = k ^ 3) : n = 7 := 
sorry

end NUMINAMATH_GPT_find_n_pos_int_l627_62700


namespace NUMINAMATH_GPT_socks_selection_l627_62787

theorem socks_selection :
  let red_socks := 120
  let green_socks := 90
  let blue_socks := 70
  let black_socks := 50
  let yellow_socks := 30
  let total_socks :=  red_socks + green_socks + blue_socks + black_socks + yellow_socks 
  (∀ k : ℕ, k ≥ 1 → k ≤ total_socks → (∃ p : ℕ, p = 12 → (p ≥ k / 2)) → k = 28) :=
by
  sorry

end NUMINAMATH_GPT_socks_selection_l627_62787


namespace NUMINAMATH_GPT_set_intersection_l627_62783

def A : Set ℝ := {1, 2, 3, 4, 5}
def B : Set ℝ := {x | x * (4 - x) < 0}
def C_R_B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_intersection :
  A ∩ C_R_B = {1, 2, 3, 4} :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_set_intersection_l627_62783


namespace NUMINAMATH_GPT_factor_expression_l627_62746

theorem factor_expression (x : ℝ) :
  80 * x ^ 5 - 250 * x ^ 9 = -10 * x ^ 5 * (25 * x ^ 4 - 8) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l627_62746


namespace NUMINAMATH_GPT_derivative_f_at_2_l627_62715

noncomputable def f (x : ℝ) : ℝ := (x + 1) * (x - 1)

theorem derivative_f_at_2 : (deriv f 2) = 4 := by
  sorry

end NUMINAMATH_GPT_derivative_f_at_2_l627_62715


namespace NUMINAMATH_GPT_min_value_expression_l627_62706

noncomputable def sinSquare (θ : ℝ) : ℝ :=
  Real.sin (θ) ^ 2

theorem min_value_expression (θ₁ θ₂ θ₃ θ₄ : ℝ) 
  (h₁ : θ₁ > 0) (h₂ : θ₂ > 0) (h₃ : θ₃ > 0) (h₄ : θ₄ > 0)
  (sum_eq_pi : θ₁ + θ₂ + θ₃ + θ₄ = Real.pi) :
  (2 * sinSquare θ₁ + 1 / sinSquare θ₁) *
  (2 * sinSquare θ₂ + 1 / sinSquare θ₂) *
  (2 * sinSquare θ₃ + 1 / sinSquare θ₃) *
  (2 * sinSquare θ₄ + 1 / sinSquare θ₁) ≥ 81 := 
by
  sorry

end NUMINAMATH_GPT_min_value_expression_l627_62706


namespace NUMINAMATH_GPT_totalInterest_l627_62701

-- Definitions for the amounts and interest rates
def totalInvestment : ℝ := 22000
def investedAt18 : ℝ := 7000
def rate18 : ℝ := 0.18
def rate14 : ℝ := 0.14

-- Calculations as conditions
def interestFrom18 (p r : ℝ) : ℝ := p * r
def investedAt14 (total inv18 : ℝ) : ℝ := total - inv18
def interestFrom14 (p r : ℝ) : ℝ := p * r

-- Proof statement
theorem totalInterest : interestFrom18 investedAt18 rate18 + interestFrom14 (investedAt14 totalInvestment investedAt18) rate14 = 3360 :=
by
  sorry

end NUMINAMATH_GPT_totalInterest_l627_62701


namespace NUMINAMATH_GPT_max_red_balls_l627_62785

theorem max_red_balls (r w : ℕ) (h1 : r = 3 * w) (h2 : r + w ≤ 50) : r = 36 :=
sorry

end NUMINAMATH_GPT_max_red_balls_l627_62785


namespace NUMINAMATH_GPT_domain_of_f_eq_R_l627_62731

noncomputable def f (x m : ℝ) : ℝ := (x - 4) / (m * x^2 + 4 * m * x + 3)

theorem domain_of_f_eq_R (m : ℝ) : 
  (∀ x : ℝ, m * x^2 + 4 * m * x + 3 ≠ 0) ↔ (0 ≤ m ∧ m < 3 / 4) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_eq_R_l627_62731


namespace NUMINAMATH_GPT_minimum_value_of_a_plus_b_l627_62794

noncomputable def f (x : ℝ) := Real.log x - (1 / x)
noncomputable def f' (x : ℝ) := 1 / x + 1 / (x^2)

theorem minimum_value_of_a_plus_b (a b m : ℝ) (h1 : a = 1 / m + 1 / (m^2)) 
  (h2 : b = Real.log m - 2 / m - 1) : a + b = -1 :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_a_plus_b_l627_62794


namespace NUMINAMATH_GPT_total_songs_l627_62722

variable (H : String) (M : String) (A : String) (T : String)

def num_songs (s : String) : ℕ :=
  if s = H then 9 else
  if s = M then 5 else
  if s = A ∨ s = T then 
    if H ≠ s ∧ M ≠ s then 6 else 7 
  else 0

theorem total_songs 
  (hH : num_songs H = 9)
  (hM : num_songs M = 5)
  (hA : 5 < num_songs A ∧ num_songs A < 9)
  (hT : 5 < num_songs T ∧ num_songs T < 9) :
  (num_songs H + num_songs M + num_songs A + num_songs T) / 3 = 10 :=
sorry

end NUMINAMATH_GPT_total_songs_l627_62722


namespace NUMINAMATH_GPT_remainder_of_55_power_55_plus_55_div_56_l627_62760

theorem remainder_of_55_power_55_plus_55_div_56 :
  (55 ^ 55 + 55) % 56 = 54 :=
by
  -- to be filled with the proof
  sorry

end NUMINAMATH_GPT_remainder_of_55_power_55_plus_55_div_56_l627_62760


namespace NUMINAMATH_GPT_exists_unique_subset_X_l627_62742

theorem exists_unique_subset_X :
  ∃ (X : Set ℤ), ∀ n : ℤ, ∃! (a b : ℤ), a ∈ X ∧ b ∈ X ∧ a + 2 * b = n :=
sorry

end NUMINAMATH_GPT_exists_unique_subset_X_l627_62742


namespace NUMINAMATH_GPT_evaluate_f_at_neg3_l627_62716

def f (x : ℝ) : ℝ := -2 * x^3 + 5 * x^2 - 3 * x + 2

theorem evaluate_f_at_neg3 : f (-3) = 110 :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_f_at_neg3_l627_62716


namespace NUMINAMATH_GPT_symmetric_points_l627_62789

theorem symmetric_points (a b : ℤ) (h1 : (a, -2) = (1, -2)) (h2 : (-1, b) = (-1, -2)) :
  (a + b) ^ 2023 = -1 := by
  -- We know from the conditions:
  -- (a, -2) and (1, -2) implies a = 1
  -- (-1, b) and (-1, -2) implies b = -2
  -- Thus it follows that:
  sorry

end NUMINAMATH_GPT_symmetric_points_l627_62789


namespace NUMINAMATH_GPT_minimum_value_of_c_l627_62749

noncomputable def triangle_area (a b c : ℝ) : ℝ :=
  (Real.sqrt 3 / 12) * (a^2 + b^2 - c^2)

noncomputable def tan_formula (a b c B : ℝ) : Prop :=
  24 * (b * c - a) = b * Real.tan B

noncomputable def min_value_c (a b c : ℝ) : ℝ :=
  (2 * Real.sqrt 3) / 3

theorem minimum_value_of_c (a b c B : ℝ) (h₁ : 0 < B ∧ B < π / 2) (h₂ : 24 * (b * c - a) = b * Real.tan B)
  (h₃ : triangle_area a b c = (1/2) * a * b * Real.sin (π / 6)) :
  c ≥ min_value_c a b c :=
by
  sorry

end NUMINAMATH_GPT_minimum_value_of_c_l627_62749


namespace NUMINAMATH_GPT_sum_of_h_values_l627_62774

variable (f h : ℤ → ℤ)

-- Function definition for f and h
def f_def : ∀ x, 0 ≤ x → f x = f (x + 2) := sorry
def h_def : ∀ x, x < 0 → h x = f x := sorry

-- Symmetry condition for f being odd
def f_odd : ∀ x, f (-x) = -f x := sorry

-- Given value
def f_at_5 : f 5 = 1 := sorry

-- The proof statement we need:
theorem sum_of_h_values :
  h (-2022) + h (-2023) + h (-2024) = -1 :=
sorry

end NUMINAMATH_GPT_sum_of_h_values_l627_62774


namespace NUMINAMATH_GPT_total_length_of_fence_l627_62780

theorem total_length_of_fence
  (x : ℝ)
  (h1 : (2 : ℝ) * x ^ 2 = 200) :
  (2 * x + 2 * x) = 40 :=
by
sorry

end NUMINAMATH_GPT_total_length_of_fence_l627_62780


namespace NUMINAMATH_GPT_speed_is_90_l627_62741

namespace DrivingSpeedProof

/-- Given the observation times and marker numbers, prove the speed of the car is 90 km/hr. -/
theorem speed_is_90 
  (X Y : ℕ)
  (h0 : X ≥ 0) (h1 : X ≤ 9)
  (h2 : Y = 8 * X)
  (h3 : Y ≥ 0) (h4 : Y ≤ 9)
  (noon_marker : 10 * X + Y = 18)
  (second_marker : 10 * Y + X = 81)
  (third_marker : 100 * X + Y = 108)
  : 90 = 90 :=
by {
  sorry
}

end DrivingSpeedProof

end NUMINAMATH_GPT_speed_is_90_l627_62741


namespace NUMINAMATH_GPT_ratio_consequent_l627_62761

theorem ratio_consequent (a b x : ℕ) (h_ratio : a = 4) (h_b : b = 6) (h_x : x = 30) :
  (a : ℚ) / b = x / 45 := 
by 
  -- add here the necessary proof steps 
  sorry

end NUMINAMATH_GPT_ratio_consequent_l627_62761


namespace NUMINAMATH_GPT_largest_angle_in_pentagon_l627_62768

def pentagon_angle_sum : ℝ := 540

def angle_A : ℝ := 70
def angle_B : ℝ := 90
def angle_C (x : ℝ) : ℝ := x
def angle_D (x : ℝ) : ℝ := x
def angle_E (x : ℝ) : ℝ := 3 * x - 10

theorem largest_angle_in_pentagon
  (x : ℝ)
  (h_sum : angle_A + angle_B + angle_C x + angle_D x + angle_E x = pentagon_angle_sum) :
  angle_E x = 224 :=
sorry

end NUMINAMATH_GPT_largest_angle_in_pentagon_l627_62768


namespace NUMINAMATH_GPT_good_or_bad_of_prime_divides_l627_62730

-- Define the conditions
variables (k n n' : ℕ)
variables (h1 : k ≥ 2) (h2 : n ≥ k) (h3 : n' ≥ k)
variables (prime_divides : ∀ p, prime p → p ≤ k → (p ∣ n ↔ p ∣ n'))

-- Define what it means for a number to be good or bad
def is_good (m : ℕ) : Prop := ∃ strategy : ℕ → Prop, strategy m

-- Prove that either both n and n' are good or both are bad
theorem good_or_bad_of_prime_divides :
  (is_good n ∧ is_good n') ∨ (¬is_good n ∧ ¬is_good n') :=
sorry

end NUMINAMATH_GPT_good_or_bad_of_prime_divides_l627_62730


namespace NUMINAMATH_GPT_largest_number_l627_62790

def A : ℚ := 97 / 100
def B : ℚ := 979 / 1000
def C : ℚ := 9709 / 10000
def D : ℚ := 907 / 1000
def E : ℚ := 9089 / 10000

theorem largest_number : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end NUMINAMATH_GPT_largest_number_l627_62790


namespace NUMINAMATH_GPT_range_of_half_alpha_minus_beta_l627_62763

theorem range_of_half_alpha_minus_beta (α β : ℝ) (h1 : 1 < α) (h2 : α < 3) (h3 : -4 < β) (h4 : β < 2) :
  -3/2 < (1/2) * α - β ∧ (1/2) * α - β < 11/2 :=
by
  -- sorry to skip the proof
  sorry

end NUMINAMATH_GPT_range_of_half_alpha_minus_beta_l627_62763


namespace NUMINAMATH_GPT_linear_function_m_value_l627_62734

theorem linear_function_m_value (m : ℝ) (h : abs (m + 1) = 1) : m = -2 :=
sorry

end NUMINAMATH_GPT_linear_function_m_value_l627_62734


namespace NUMINAMATH_GPT_triangle_angle_not_less_than_60_l627_62729

theorem triangle_angle_not_less_than_60 
  (a b c : ℝ) 
  (h1 : a + b + c = 180) 
  (h2 : a < 60) 
  (h3 : b < 60) 
  (h4 : c < 60) : 
  false := 
by
  sorry

end NUMINAMATH_GPT_triangle_angle_not_less_than_60_l627_62729
