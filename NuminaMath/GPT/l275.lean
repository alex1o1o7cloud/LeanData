import Mathlib

namespace NUMINAMATH_GPT_solve_equation_l275_27526

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem solve_equation:
    (7.331 * ((log_base 3 x - 1) / (log_base 3 (x / 3))) - 
    2 * (log_base 3 (Real.sqrt x)) + (log_base 3 x)^2 = 3) → 
    (x = 1 / 3 ∨ x = 9) := by
  sorry

end NUMINAMATH_GPT_solve_equation_l275_27526


namespace NUMINAMATH_GPT_diagonals_in_octagon_l275_27533

/-- The formula to calculate the number of diagonals in a polygon -/
def number_of_diagonals (n : Nat) : Nat :=
  (n * (n - 3)) / 2

/-- The number of sides in an octagon -/
def sides_of_octagon : Nat := 8

/-- The number of diagonals in an octagon is 20. -/
theorem diagonals_in_octagon : number_of_diagonals sides_of_octagon = 20 :=
by
  sorry

end NUMINAMATH_GPT_diagonals_in_octagon_l275_27533


namespace NUMINAMATH_GPT_groupB_is_basis_l275_27585

section
variables (eA1 eA2 : ℝ × ℝ) (eB1 eB2 : ℝ × ℝ) (eC1 eC2 : ℝ × ℝ) (eD1 eD2 : ℝ × ℝ)

def is_collinear (v w : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v = (k • w) ∨ w = (k • v)

-- Define each vector group
def groupA := eA1 = (0, 0) ∧ eA2 = (1, -2)
def groupB := eB1 = (-1, 2) ∧ eB2 = (5, 7)
def groupC := eC1 = (3, 5) ∧ eC2 = (6, 10)
def groupD := eD1 = (2, -3) ∧ eD2 = (1/2, -3/4)

-- The goal is to prove that group B vectors can serve as a basis
theorem groupB_is_basis : ¬ is_collinear eB1 eB2 :=
sorry
end

end NUMINAMATH_GPT_groupB_is_basis_l275_27585


namespace NUMINAMATH_GPT_geo_prog_sum_463_l275_27501

/-- Given a set of natural numbers forming an increasing geometric progression with an integer
common ratio where the sum equals 463, prove that these numbers must be {463}, {1, 462}, or {1, 21, 441}. -/
theorem geo_prog_sum_463 (n : ℕ) (b₁ q : ℕ) (s : Finset ℕ) (hgeo : ∀ i j, i < j → s.toList.get? i = some (b₁ * q^i) ∧ s.toList.get? j = some (b₁ * q^j))
  (hsum : s.sum id = 463) : 
  s = {463} ∨ s = {1, 462} ∨ s = {1, 21, 441} :=
sorry

end NUMINAMATH_GPT_geo_prog_sum_463_l275_27501


namespace NUMINAMATH_GPT_interest_difference_l275_27542

def principal : ℝ := 3600
def rate : ℝ := 0.25
def time : ℕ := 2

def simple_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * r * t

def compound_interest (P : ℝ) (r : ℝ) (t : ℕ) : ℝ :=
  P * (1 + r)^t - P

theorem interest_difference :
  let SI := simple_interest principal rate time;
  let CI := compound_interest principal rate time;
  CI - SI = 225 :=
by
  sorry

end NUMINAMATH_GPT_interest_difference_l275_27542


namespace NUMINAMATH_GPT_num_sets_N_l275_27521

open Set

noncomputable def M : Set ℤ := {-1, 0}

theorem num_sets_N (N : Set ℤ) : M ∪ N = {-1, 0, 1} → 
  (N = {1} ∨ N = {0, 1} ∨ N = {-1, 1} ∨ N = {0, -1, 1}) := 
sorry

end NUMINAMATH_GPT_num_sets_N_l275_27521


namespace NUMINAMATH_GPT_box_triple_count_l275_27541

theorem box_triple_count (a b c : ℕ) (h1 : 2 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : a * b * c = 2 * (a * b + b * c + c * a)) :
  (a = 2 ∧ b = 8 ∧ c = 8) ∨ (a = 3 ∧ b = 6 ∧ c = 6) ∨ (a = 4 ∧ b = 4 ∧ c = 4) ∨ (a = 5 ∧ b = 5 ∧ c = 5) ∨ (a = 6 ∧ b = 6 ∧ c = 6) :=
sorry

end NUMINAMATH_GPT_box_triple_count_l275_27541


namespace NUMINAMATH_GPT_find_room_width_l275_27595

theorem find_room_width
  (length : ℝ)
  (cost_per_sqm : ℝ)
  (total_cost : ℝ)
  (h_length : length = 10)
  (h_cost_per_sqm : cost_per_sqm = 900)
  (h_total_cost : total_cost = 42750) :
  ∃ width : ℝ, width = 4.75 :=
by
  sorry

end NUMINAMATH_GPT_find_room_width_l275_27595


namespace NUMINAMATH_GPT_solve_problem_l275_27551

theorem solve_problem (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  7^m - 3 * 2^n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) := sorry

end NUMINAMATH_GPT_solve_problem_l275_27551


namespace NUMINAMATH_GPT_circus_tent_sections_l275_27512

noncomputable def sections_in_circus_tent (total_capacity : ℕ) (section_capacity : ℕ) : ℕ :=
  total_capacity / section_capacity

theorem circus_tent_sections : sections_in_circus_tent 984 246 = 4 := 
  by 
  sorry

end NUMINAMATH_GPT_circus_tent_sections_l275_27512


namespace NUMINAMATH_GPT_largest_n_satisfying_conditions_l275_27569

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, n = 181 ∧
    (∃ m : ℤ, n^2 = (m + 1)^3 - m^3) ∧
    ∃ k : ℤ, 2 * n + 79 = k^2 :=
by
  sorry

end NUMINAMATH_GPT_largest_n_satisfying_conditions_l275_27569


namespace NUMINAMATH_GPT_infinite_series_eq_15_l275_27598

theorem infinite_series_eq_15 (x : ℝ) :
  (∑' (n : ℕ), (5 + n * x) / 3^n) = 15 ↔ x = 10 :=
by
  sorry

end NUMINAMATH_GPT_infinite_series_eq_15_l275_27598


namespace NUMINAMATH_GPT_problem_statement_l275_27589

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - 2 * a * x

theorem problem_statement (a : ℝ) (x1 x2 : ℝ) (h_a : a > 1) (h1 : x1 < x2) (h_extreme : f a x1 = 0 ∧ f a x2 = 0) : 
  f a x2 < -3/2 :=
sorry

end NUMINAMATH_GPT_problem_statement_l275_27589


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l275_27586

-- Define the arithmetic sequence and related sum functions
def a_n (a1 d : ℤ) (n : ℕ) : ℤ := a1 + (n - 1) * d

def S (a1 d : ℤ) (n : ℕ) : ℤ :=
  (a1 + a_n a1 d n) * n / 2

-- Problem statement: proving a_5 = -1 given the conditions
theorem arithmetic_sequence_problem :
  (∃ (a1 d : ℕ), S a1 d 2 = S a1 d 6 ∧ a_n a1 d 4 = 1) → a_n a1 d 5 = -1 :=
by
  -- Assume the statement and then skip the proof
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l275_27586


namespace NUMINAMATH_GPT_find_smallest_natural_number_l275_27565

theorem find_smallest_natural_number :
  ∃ x : ℕ, (2 * x = b^2 ∧ 3 * x = c^3) ∧ (∀ y : ℕ, (2 * y = d^2 ∧ 3 * y = e^3) → x ≤ y) := by
  sorry

end NUMINAMATH_GPT_find_smallest_natural_number_l275_27565


namespace NUMINAMATH_GPT_amount_each_girl_receives_l275_27567

theorem amount_each_girl_receives (total_amount : ℕ) (total_children : ℕ) (amount_per_boy : ℕ) (num_boys : ℕ) (remaining_amount : ℕ) (num_girls : ℕ) (amount_per_girl : ℕ) 
  (h1 : total_amount = 460) 
  (h2 : total_children = 41)
  (h3 : amount_per_boy = 12)
  (h4 : num_boys = 33)
  (h5 : remaining_amount = total_amount - num_boys * amount_per_boy)
  (h6 : num_girls = total_children - num_boys)
  (h7 : amount_per_girl = remaining_amount / num_girls) :
  amount_per_girl = 8 := 
sorry

end NUMINAMATH_GPT_amount_each_girl_receives_l275_27567


namespace NUMINAMATH_GPT_fraction_absent_l275_27568

theorem fraction_absent (p : ℕ) (x : ℝ) (h1 : 0 ≤ x ∧ x ≤ 1) (h2 : p * 1 = (1 - x) * p * 1.5) : x = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_absent_l275_27568


namespace NUMINAMATH_GPT_geometric_sequence_product_proof_l275_27581

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n, a (n + 1) = a n * q

theorem geometric_sequence_product_proof (a : ℕ → ℝ) (q : ℝ)
  (h_geo : geometric_sequence a q) 
  (h1 : a 2010 * a 2011 * a 2012 = 3)
  (h2 : a 2013 * a 2014 * a 2015 = 24) :
  a 2016 * a 2017 * a 2018 = 192 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_proof_l275_27581


namespace NUMINAMATH_GPT_train_speed_in_m_per_s_l275_27560

-- Define the given train speed in kmph
def train_speed_kmph : ℕ := 72

-- Define the conversion factor from kmph to m/s
def km_per_hour_to_m_per_second (speed_in_kmph : ℕ) : ℕ := (speed_in_kmph * 1000) / 3600

-- State the theorem
theorem train_speed_in_m_per_s (h : train_speed_kmph = 72) : km_per_hour_to_m_per_second train_speed_kmph = 20 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_m_per_s_l275_27560


namespace NUMINAMATH_GPT_intersection_eq_l275_27539

open Set

-- Define the sets A and B according to the given conditions
def A : Set ℤ := {-2, -1, 0, 1, 2}
def B : Set ℝ := {x | (x - 1) * (x + 2) < 0}

-- Define the intended intersection result
def C : Set ℤ := {-1, 0}

-- The theorem to prove
theorem intersection_eq : A ∩ {x | (x - 1) * (x + 2) < 0} = C := by
  sorry

end NUMINAMATH_GPT_intersection_eq_l275_27539


namespace NUMINAMATH_GPT_ratio_is_one_to_two_l275_27543

def valentina_share_to_whole_ratio (valentina_share : ℕ) (whole_burger : ℕ) : ℕ × ℕ :=
  (valentina_share / (Nat.gcd valentina_share whole_burger), 
   whole_burger / (Nat.gcd valentina_share whole_burger))

theorem ratio_is_one_to_two : valentina_share_to_whole_ratio 6 12 = (1, 2) := 
  by
  sorry

end NUMINAMATH_GPT_ratio_is_one_to_two_l275_27543


namespace NUMINAMATH_GPT_domain_of_function_l275_27519

theorem domain_of_function (x : ℝ) :
  (2 - x ≥ 0) ∧ (x - 1 > 0) ↔ (1 < x ∧ x ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_function_l275_27519


namespace NUMINAMATH_GPT_problem_statement_l275_27550

noncomputable def a := Real.sqrt 3 + Real.sqrt 2
noncomputable def b := Real.sqrt 3 - Real.sqrt 2
noncomputable def expression := a^(2 * Real.log (Real.sqrt 5) / Real.log b)

theorem problem_statement : expression = 1 / 5 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l275_27550


namespace NUMINAMATH_GPT_converse_prop_inverse_prop_contrapositive_prop_l275_27530

-- Given condition: the original proposition is true
axiom original_prop : ∀ (x y : ℝ), x * y = 0 → x = 0 ∨ y = 0

-- Converse: If x=0 or y=0, then xy=0 - prove this is true
theorem converse_prop (x y : ℝ) : (x = 0 ∨ y = 0) → x * y = 0 :=
by
  sorry

-- Inverse: If xy ≠ 0, then x ≠ 0 and y ≠ 0 - prove this is true
theorem inverse_prop (x y : ℝ) : x * y ≠ 0 → x ≠ 0 ∧ y ≠ 0 :=
by
  sorry

-- Contrapositive: If x ≠ 0 and y ≠ 0, then xy ≠ 0 - prove this is true
theorem contrapositive_prop (x y : ℝ) : (x ≠ 0 ∧ y ≠ 0) → x * y ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_converse_prop_inverse_prop_contrapositive_prop_l275_27530


namespace NUMINAMATH_GPT_remainder_4_power_100_div_9_l275_27572

theorem remainder_4_power_100_div_9 : (4^100) % 9 = 4 :=
by
  sorry

end NUMINAMATH_GPT_remainder_4_power_100_div_9_l275_27572


namespace NUMINAMATH_GPT_calculate_f_f_neg3_l275_27516

def f (x : ℚ) : ℚ := (1 / x) + (1 / (x + 1))

theorem calculate_f_f_neg3 : f (f (-3)) = 24 / 5 := by
  sorry

end NUMINAMATH_GPT_calculate_f_f_neg3_l275_27516


namespace NUMINAMATH_GPT_number_of_terminating_decimals_l275_27507

theorem number_of_terminating_decimals : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 299 → (∃ k : ℕ, n = 9 * k) → 
  ∃ count : ℕ, count = 33 := 
sorry

end NUMINAMATH_GPT_number_of_terminating_decimals_l275_27507


namespace NUMINAMATH_GPT_gcd_4536_8721_l275_27538

theorem gcd_4536_8721 : Nat.gcd 4536 8721 = 3 := by
  sorry

end NUMINAMATH_GPT_gcd_4536_8721_l275_27538


namespace NUMINAMATH_GPT_no_rational_solution_l275_27503

theorem no_rational_solution 
  (a b c : ℤ) 
  (ha : a % 2 = 1) 
  (hb : b % 2 = 1) 
  (hc : c % 2 = 1) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 :=
by sorry

end NUMINAMATH_GPT_no_rational_solution_l275_27503


namespace NUMINAMATH_GPT_decrease_of_negative_five_l275_27511

-- Definition: Positive and negative numbers as explained
def increase (n: ℤ) : Prop := n > 0
def decrease (n: ℤ) : Prop := n < 0

-- Conditions
def condition : Prop := increase 17

-- Theorem stating the solution
theorem decrease_of_negative_five (h : condition) : decrease (-5) ∧ -5 = -5 :=
by
  sorry

end NUMINAMATH_GPT_decrease_of_negative_five_l275_27511


namespace NUMINAMATH_GPT_fraction_of_occupied_student_chairs_is_4_over_5_l275_27500

-- Definitions based on the conditions provided
def total_chairs : ℕ := 10 * 15
def awardees_chairs : ℕ := 15
def admin_teachers_chairs : ℕ := 2 * 15
def parents_chairs : ℕ := 2 * 15
def student_chairs : ℕ := total_chairs - (awardees_chairs + admin_teachers_chairs + parents_chairs)
def vacant_student_chairs_given_to_parents : ℕ := 15
def occupied_student_chairs : ℕ := student_chairs - vacant_student_chairs_given_to_parents

-- Theorem statement based on the problem
theorem fraction_of_occupied_student_chairs_is_4_over_5 :
    (occupied_student_chairs : ℚ) / student_chairs = 4 / 5 :=
by
    sorry

end NUMINAMATH_GPT_fraction_of_occupied_student_chairs_is_4_over_5_l275_27500


namespace NUMINAMATH_GPT_race_length_l275_27577

theorem race_length
  (B_s : ℕ := 50) -- Biff's speed in yards per minute
  (K_s : ℕ := 51) -- Kenneth's speed in yards per minute
  (D_above_finish : ℕ := 10) -- distance Kenneth is past the finish line when Biff finishes
  : {L : ℕ // L = 500} := -- the length of the race is 500 yards.
  sorry

end NUMINAMATH_GPT_race_length_l275_27577


namespace NUMINAMATH_GPT_multiply_101_self_l275_27590

theorem multiply_101_self : 101 * 101 = 10201 := 
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_multiply_101_self_l275_27590


namespace NUMINAMATH_GPT_find_k_and_angle_l275_27558

def vector := ℝ × ℝ

def dot_product (u v: vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def orthogonal (u v: vector) : Prop :=
  dot_product u v = 0

theorem find_k_and_angle (k : ℝ) :
  let a : vector := (3, -1)
  let b : vector := (1, k)
  orthogonal a b →
  (k = 3 ∧ dot_product (3+1, -1+3) (3-1, -1-3) = 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_k_and_angle_l275_27558


namespace NUMINAMATH_GPT_rope_length_l275_27553

theorem rope_length (x : ℝ) 
  (h : 10^2 + (x - 4)^2 = x^2) : 
  x = 14.5 :=
sorry

end NUMINAMATH_GPT_rope_length_l275_27553


namespace NUMINAMATH_GPT_total_young_fish_l275_27593

-- Define conditions
def tanks : ℕ := 3
def fish_per_tank : ℕ := 4
def young_per_fish : ℕ := 20

-- Define the main proof statement
theorem total_young_fish : tanks * fish_per_tank * young_per_fish = 240 := by
  sorry

end NUMINAMATH_GPT_total_young_fish_l275_27593


namespace NUMINAMATH_GPT_fifth_number_selected_l275_27514

-- Define the necessary conditions
def num_students : ℕ := 60
def sample_size : ℕ := 5
def first_selected_number : ℕ := 4
def interval : ℕ := num_students / sample_size

-- Define the proposition to be proved
theorem fifth_number_selected (h1 : 1 ≤ first_selected_number) (h2 : first_selected_number ≤ num_students)
    (h3 : sample_size > 0) (h4 : num_students % sample_size = 0) :
  first_selected_number + 4 * interval = 52 :=
by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_fifth_number_selected_l275_27514


namespace NUMINAMATH_GPT_triangle_median_inequality_l275_27580

-- Defining the parameters and the inequality theorem.
theorem triangle_median_inequality
  (a b c : ℝ)
  (ma mb mc : ℝ)
  (Δ : ℝ)
  (median_medians : ∀ {a b c : ℝ}, ma ≤ mb ∧ mb ≤ mc ∧ a ≥ b ∧ b ≥ c)  :
  a * (-ma + mb + mc) + b * (ma - mb + mc) + c * (ma + mb - mc) ≥ 6 * Δ := 
sorry

end NUMINAMATH_GPT_triangle_median_inequality_l275_27580


namespace NUMINAMATH_GPT_problem_statement_l275_27575

theorem problem_statement (p : ℕ) (hprime : Prime p) :
  (∃ m n : ℕ, m > 0 ∧ n > 0 ∧ p = m^2 + n^2 ∧ p ∣ (m^3 + n^3 + 8 * m * n)) → p = 5 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l275_27575


namespace NUMINAMATH_GPT_pencils_given_out_l275_27574

theorem pencils_given_out
  (num_children : ℕ)
  (pencils_per_student : ℕ)
  (dozen : ℕ)
  (children : num_children = 46)
  (dozen_def : dozen = 12)
  (pencils_def : pencils_per_student = 4 * dozen) :
  num_children * pencils_per_student = 2208 :=
by {
  sorry
}

end NUMINAMATH_GPT_pencils_given_out_l275_27574


namespace NUMINAMATH_GPT_find_N_l275_27571

-- Definition of the conditions
def is_largest_divisor_smaller_than (m N : ℕ) : Prop := m < N ∧ Nat.gcd m N = m

def produces_power_of_ten (N m : ℕ) : Prop := ∃ k : ℕ, k > 0 ∧ N + m = 10^k

-- Final statement to prove
theorem find_N (N : ℕ) : (∃ m : ℕ, is_largest_divisor_smaller_than m N ∧ produces_power_of_ten N m) → N = 75 :=
by
  sorry

end NUMINAMATH_GPT_find_N_l275_27571


namespace NUMINAMATH_GPT_base6_addition_problem_l275_27515

-- Definitions to capture the base-6 addition problem components.
def base6₀ := 0
def base6₁ := 1
def base6₂ := 2
def base6₃ := 3
def base6₄ := 4
def base6₅ := 5

-- The main hypothesis about the base-6 addition
theorem base6_addition_problem (diamond : ℕ) (h : diamond ∈ [base6₀, base6₁, base6₂, base6₃, base6₄, base6₅]) :
  ((diamond + base6₅) % 6 = base6₃ ∨ (diamond + base6₅) % 6 = (base6₃ + 6 * 1 % 6)) ∧
  (diamond + base6₂ + base6₂ = diamond % 6) →
  diamond = base6₄ :=
sorry

end NUMINAMATH_GPT_base6_addition_problem_l275_27515


namespace NUMINAMATH_GPT_probability_at_least_two_red_balls_l275_27510

noncomputable def prob_red_balls (total_balls : ℕ) (red_balls : ℕ) (white_balls : ℕ) (black_balls : ℕ) (drawn_balls : ℕ) : ℚ :=
if total_balls = 6 ∧ red_balls = 3 ∧ white_balls = 2 ∧ black_balls = 1 ∧ drawn_balls = 3 then
  1 / 2
else
  0

theorem probability_at_least_two_red_balls :
  prob_red_balls 6 3 2 1 3 = 1 / 2 :=
by 
  sorry

end NUMINAMATH_GPT_probability_at_least_two_red_balls_l275_27510


namespace NUMINAMATH_GPT_statement_B_is_algorithm_l275_27592

def is_algorithm (statement : String) : Prop := 
  statement = "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."

def condition_A : String := "At home, it is generally the mother who cooks."
def condition_B : String := "Cooking rice involves the steps of washing the pot, rinsing the rice, adding water, and heating."
def condition_C : String := "Cooking outdoors is called a picnic."
def condition_D : String := "Rice is necessary for cooking."

theorem statement_B_is_algorithm : is_algorithm condition_B :=
by
  sorry

end NUMINAMATH_GPT_statement_B_is_algorithm_l275_27592


namespace NUMINAMATH_GPT_moon_speed_conversion_l275_27517

def moon_speed_km_sec : ℝ := 1.04
def seconds_per_hour : ℝ := 3600

theorem moon_speed_conversion :
  (moon_speed_km_sec * seconds_per_hour) = 3744 := by
  sorry

end NUMINAMATH_GPT_moon_speed_conversion_l275_27517


namespace NUMINAMATH_GPT_negation_proposition_p_l275_27523

open Classical

variable (n : ℕ)

def proposition_p : Prop := ∃ n : ℕ, 2^n > 100

theorem negation_proposition_p : ¬ proposition_p ↔ ∀ n : ℕ, 2^n ≤ 100 := 
by sorry

end NUMINAMATH_GPT_negation_proposition_p_l275_27523


namespace NUMINAMATH_GPT_miranda_savings_l275_27579

theorem miranda_savings:
  ∀ (months : ℕ) (sister_contribution price shipping total paid_per_month : ℝ),
    months = 3 →
    sister_contribution = 50 →
    price = 210 →
    shipping = 20 →
    total = 230 →
    total - sister_contribution = price + shipping →
    paid_per_month = (total - sister_contribution) / months →
    paid_per_month = 60 :=
by
  intros months sister_contribution price shipping total paid_per_month h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_miranda_savings_l275_27579


namespace NUMINAMATH_GPT_problem_statement_l275_27584

theorem problem_statement (n k : ℕ) (h1 : n = 2^2007 * k + 1) (h2 : k % 2 = 1) : ¬ n ∣ 2^(n-1) + 1 := by
  sorry

end NUMINAMATH_GPT_problem_statement_l275_27584


namespace NUMINAMATH_GPT_more_trees_died_than_survived_l275_27597

def haley_trees : ℕ := 14
def died_in_typhoon : ℕ := 9
def survived_trees := haley_trees - died_in_typhoon

theorem more_trees_died_than_survived : (died_in_typhoon - survived_trees) = 4 := by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_more_trees_died_than_survived_l275_27597


namespace NUMINAMATH_GPT_value_of_t_l275_27525

theorem value_of_t (t : ℝ) (x y : ℝ) (h : 3 * x^(t-1) + y - 5 = 0) :
  t = 2 :=
sorry

end NUMINAMATH_GPT_value_of_t_l275_27525


namespace NUMINAMATH_GPT_eggs_needed_for_recipe_l275_27599

noncomputable section

theorem eggs_needed_for_recipe 
  (total_eggs : ℕ) 
  (rotten_eggs : ℕ) 
  (prob_all_rotten : ℝ)
  (h_total : total_eggs = 36)
  (h_rotten : rotten_eggs = 3)
  (h_prob : prob_all_rotten = 0.0047619047619047615) 
  : (2 : ℕ) = 2 :=
by
  sorry

end NUMINAMATH_GPT_eggs_needed_for_recipe_l275_27599


namespace NUMINAMATH_GPT_distinct_m_count_l275_27564

noncomputable def countDistinctMValues : Nat :=
  let pairs := [(1, 36), (2, 18), (3, 12), (4, 9), (6, 6), 
                (-1, -36), (-2, -18), (-3, -12), (-4, -9), (-6, -6)]
  let ms := pairs.map (λ p => p.1 + p.2)
  ms.eraseDups.length

theorem distinct_m_count :
  countDistinctMValues = 10 := sorry

end NUMINAMATH_GPT_distinct_m_count_l275_27564


namespace NUMINAMATH_GPT_solve_system_l275_27534

theorem solve_system :
  ∃ x y : ℝ, (x + 2*y = 1 ∧ 3*x - 2*y = 7) → (x = 2 ∧ y = -1/2) :=
by
  sorry

end NUMINAMATH_GPT_solve_system_l275_27534


namespace NUMINAMATH_GPT_quadratic_no_real_solution_l275_27547

theorem quadratic_no_real_solution (a : ℝ) : (∀ x : ℝ, x^2 - x + a ≠ 0) → a > 1 / 4 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_quadratic_no_real_solution_l275_27547


namespace NUMINAMATH_GPT_find_rs_l275_27502

theorem find_rs (r s : ℝ) (h1 : 0 < r) (h2 : 0 < s) (h3 : r^2 + s^2 = 1) (h4 : r^4 + s^4 = 7/8) : 
  r * s = 1/4 :=
sorry

end NUMINAMATH_GPT_find_rs_l275_27502


namespace NUMINAMATH_GPT_percentage_of_total_l275_27554

theorem percentage_of_total (total part : ℕ) (h₁ : total = 100) (h₂ : part = 30):
  (part / total) * 100 = 30 := by
  sorry

end NUMINAMATH_GPT_percentage_of_total_l275_27554


namespace NUMINAMATH_GPT_sqrt_defined_range_l275_27583

theorem sqrt_defined_range (x : ℝ) : (∃ y : ℝ, y = Real.sqrt (x - 2)) → (x ≥ 2) := by
  sorry

end NUMINAMATH_GPT_sqrt_defined_range_l275_27583


namespace NUMINAMATH_GPT_same_terminal_side_l275_27527

theorem same_terminal_side : ∃ k : ℤ, k * 360 - 60 = 300 := by
  sorry

end NUMINAMATH_GPT_same_terminal_side_l275_27527


namespace NUMINAMATH_GPT_find_side_b_in_triangle_l275_27535

noncomputable def triangle_side_b (a A : ℝ) (cosB : ℝ) : ℝ :=
  let sinB := Real.sqrt (1 - cosB^2)
  let sinA := Real.sin A
  (a * sinB) / sinA

theorem find_side_b_in_triangle :
  triangle_side_b 5 (Real.pi / 4) (3 / 5) = 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_find_side_b_in_triangle_l275_27535


namespace NUMINAMATH_GPT_points_calculation_l275_27504

def points_per_enemy : ℕ := 9
def total_enemies : ℕ := 11
def enemies_destroyed : ℕ := total_enemies - 3
def total_points_earned : ℕ := enemies_destroyed * points_per_enemy

theorem points_calculation :
  total_points_earned = 72 := by
  sorry

end NUMINAMATH_GPT_points_calculation_l275_27504


namespace NUMINAMATH_GPT_operation_on_original_number_l275_27591

theorem operation_on_original_number (f : ℕ → ℕ) (x : ℕ) (h : 3 * (f x + 9) = 51) (hx : x = 4) :
  f x = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_operation_on_original_number_l275_27591


namespace NUMINAMATH_GPT_base_conversion_min_sum_l275_27548

theorem base_conversion_min_sum (c d : ℕ) (h : 5 * c + 8 = 8 * d + 5) : c + d = 15 := by
  sorry

end NUMINAMATH_GPT_base_conversion_min_sum_l275_27548


namespace NUMINAMATH_GPT_set_C_cannot_form_right_triangle_l275_27546

def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem set_C_cannot_form_right_triangle :
  ¬ is_right_triangle 7 8 9 :=
by
  sorry

end NUMINAMATH_GPT_set_C_cannot_form_right_triangle_l275_27546


namespace NUMINAMATH_GPT_money_collected_is_correct_l275_27528

-- Define the conditions as constants and definitions in Lean
def ticket_price_adult : ℝ := 0.60
def ticket_price_child : ℝ := 0.25
def total_persons : ℕ := 280
def children_attended : ℕ := 80

-- Define the number of adults
def adults_attended : ℕ := total_persons - children_attended

-- Define the total money collected
def total_money_collected : ℝ :=
  (adults_attended * ticket_price_adult) + (children_attended * ticket_price_child)

-- Statement to prove
theorem money_collected_is_correct :
  total_money_collected = 140 := by
  sorry

end NUMINAMATH_GPT_money_collected_is_correct_l275_27528


namespace NUMINAMATH_GPT_profit_ratio_l275_27559

theorem profit_ratio (I_P I_Q : ℝ) (t_P t_Q : ℕ) 
  (h1 : I_P / I_Q = 7 / 5)
  (h2 : t_P = 5)
  (h3 : t_Q = 14) : 
  (I_P * t_P) / (I_Q * t_Q) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_profit_ratio_l275_27559


namespace NUMINAMATH_GPT_penelope_food_intake_l275_27505

theorem penelope_food_intake
(G P M E : ℕ) -- Representing amount of food each animal eats per day
(h1 : P = 10 * G) -- Penelope eats 10 times Greta's food
(h2 : M = G / 100) -- Milton eats 1/100 of Greta's food
(h3 : E = 4000 * M) -- Elmer eats 4000 times what Milton eats
(h4 : E = P + 60) -- Elmer eats 60 pounds more than Penelope
(G_val : G = 2) -- Greta eats 2 pounds per day
: P = 20 := -- Prove Penelope eats 20 pounds per day
by
  rw [G_val] at h1 -- Replace G with 2 in h1
  norm_num at h1 -- Evaluate the expression in h1
  exact h1 -- Conclude P = 20

end NUMINAMATH_GPT_penelope_food_intake_l275_27505


namespace NUMINAMATH_GPT_rancher_no_cows_l275_27524

theorem rancher_no_cows (s c : ℕ) (h1 : 30 * s + 31 * c = 1200) 
  (h2 : 15 ≤ s) (h3 : s ≤ 35) : c = 0 :=
by
  sorry

end NUMINAMATH_GPT_rancher_no_cows_l275_27524


namespace NUMINAMATH_GPT_relation_between_a_b_l275_27506

variables {x y a b : ℝ}

theorem relation_between_a_b 
  (h1 : a = (x^2 + y^2) * (x - y))
  (h2 : b = (x^2 - y^2) * (x + y))
  (h3 : x < y) 
  (h4 : y < 0) : 
  a > b := 
by sorry

end NUMINAMATH_GPT_relation_between_a_b_l275_27506


namespace NUMINAMATH_GPT_replace_stars_with_identity_l275_27520

theorem replace_stars_with_identity:
  ∃ (a b : ℝ), 
  (12 * a = b - 13) ∧ 
  (6 * a^2 = 7 - b) ∧ 
  (a^3 = -b) ∧ 
  a = -1 ∧ b = 1 := 
by
  sorry

end NUMINAMATH_GPT_replace_stars_with_identity_l275_27520


namespace NUMINAMATH_GPT_cost_of_hiring_actors_l275_27552

theorem cost_of_hiring_actors
  (A : ℕ)
  (CostOfFood : ℕ := 150)
  (EquipmentRental : ℕ := 300 + 2 * A)
  (TotalCost : ℕ := 3 * A + 450)
  (SellingPrice : ℕ := 10000)
  (Profit : ℕ := 5950) :
  TotalCost = SellingPrice - Profit → A = 1200 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cost_of_hiring_actors_l275_27552


namespace NUMINAMATH_GPT_range_of_squared_function_l275_27508

theorem range_of_squared_function (x : ℝ) (hx : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end NUMINAMATH_GPT_range_of_squared_function_l275_27508


namespace NUMINAMATH_GPT_maximum_value_of_w_l275_27578

variables (x y : ℝ)

def condition : Prop := x^2 + y^2 = 18 * x + 8 * y + 10

def w (x y : ℝ) := 4 * x + 3 * y

theorem maximum_value_of_w : ∃ x y, condition x y ∧ w x y = 74 :=
sorry

end NUMINAMATH_GPT_maximum_value_of_w_l275_27578


namespace NUMINAMATH_GPT_raisin_cost_fraction_l275_27561

theorem raisin_cost_fraction
  (R : ℝ)                -- cost of a pound of raisins
  (cost_nuts : ℝ := 2 * R)  -- cost of a pound of nuts
  (cost_raisins : ℝ := 3 * R)  -- cost of 3 pounds of raisins
  (cost_nuts_total : ℝ := 4 * cost_nuts)  -- cost of 4 pounds of nuts
  (total_cost : ℝ := cost_raisins + cost_nuts_total)  -- total cost of the mixture
  (fraction_of_raisins : ℝ := cost_raisins / total_cost)  -- fraction of cost of raisins
  : fraction_of_raisins = 3 / 11 := 
by
  sorry

end NUMINAMATH_GPT_raisin_cost_fraction_l275_27561


namespace NUMINAMATH_GPT_probability_of_not_all_8_sided_dice_rolling_the_same_number_l275_27555

def probability_not_all_same (total_faces : ℕ) (num_dice : ℕ) : ℚ :=
  let total_outcomes := total_faces ^ num_dice
  let same_number_outcomes := total_faces
  let p_same := same_number_outcomes / total_outcomes
  1 - p_same

theorem probability_of_not_all_8_sided_dice_rolling_the_same_number :
  probability_not_all_same 8 5 = 4095 / 4096 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_not_all_8_sided_dice_rolling_the_same_number_l275_27555


namespace NUMINAMATH_GPT_students_in_both_band_and_chorus_l275_27545

-- Definitions of conditions
def total_students := 250
def band_students := 90
def chorus_students := 120
def band_or_chorus_students := 180

-- Theorem statement to prove the number of students in both band and chorus
theorem students_in_both_band_and_chorus : 
  (band_students + chorus_students - band_or_chorus_students) = 30 := 
by sorry

end NUMINAMATH_GPT_students_in_both_band_and_chorus_l275_27545


namespace NUMINAMATH_GPT_find_m_plus_c_l275_27576

-- We need to define the conditions first
variable {A : ℝ × ℝ} {B : ℝ × ℝ} {c : ℝ} {m : ℝ}

-- Given conditions from part a)
def A_def : Prop := A = (1, 3)
def B_def : Prop := B = (m, -1)
def centers_line : Prop := ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)

-- Define the theorem for the proof problem
theorem find_m_plus_c (A_def : A = (1, 3)) (B_def : B = (m, -1)) (centers_line : ∀ C : ℝ × ℝ, (C.1 - C.2 + c = 0)) : m + c = 3 :=
sorry

end NUMINAMATH_GPT_find_m_plus_c_l275_27576


namespace NUMINAMATH_GPT_cylinder_lateral_area_l275_27588

-- Define the cylindrical lateral area calculation
noncomputable def lateral_area_of_cylinder (d h : ℝ) : ℝ := (2 * Real.pi * (d / 2)) * h

-- The statement of the problem in Lean 4.
theorem cylinder_lateral_area : lateral_area_of_cylinder 4 4 = 16 * Real.pi := by
  sorry

end NUMINAMATH_GPT_cylinder_lateral_area_l275_27588


namespace NUMINAMATH_GPT_possible_triplets_l275_27536

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem possible_triplets (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (is_power_of_two (a * b - c) ∧ is_power_of_two (b * c - a) ∧ is_power_of_two (c * a - b)) ↔ 
  (a = 2 ∧ b = 2 ∧ c = 2) ∨
  (a = 2 ∧ b = 2 ∧ c = 3) ∨
  (a = 2 ∧ b = 3 ∧ c = 6) ∨
  (a = 3 ∧ b = 5 ∧ c = 7) :=
by
  sorry

end NUMINAMATH_GPT_possible_triplets_l275_27536


namespace NUMINAMATH_GPT_log_comparison_l275_27513

theorem log_comparison : Real.log 675 / Real.log 135 > Real.log 75 / Real.log 45 := 
sorry

end NUMINAMATH_GPT_log_comparison_l275_27513


namespace NUMINAMATH_GPT_average_sweater_less_by_21_after_discount_l275_27563

theorem average_sweater_less_by_21_after_discount
  (shirt_count sweater_count jeans_count : ℕ)
  (total_shirt_price total_sweater_price total_jeans_price : ℕ)
  (shirt_discount sweater_discount jeans_discount : ℕ)
  (shirt_avg_before_discount sweater_avg_before_discount jeans_avg_before_discount 
   shirt_avg_after_discount sweater_avg_after_discount jeans_avg_after_discount : ℕ) :
  shirt_count = 20 →
  sweater_count = 45 →
  jeans_count = 30 →
  total_shirt_price = 360 →
  total_sweater_price = 900 →
  total_jeans_price = 1200 →
  shirt_discount = 2 →
  sweater_discount = 4 →
  jeans_discount = 3 →
  shirt_avg_before_discount = total_shirt_price / shirt_count →
  sweater_avg_before_discount = total_sweater_price / sweater_count →
  jeans_avg_before_discount = total_jeans_price / jeans_count →
  shirt_avg_after_discount = shirt_avg_before_discount - shirt_discount →
  sweater_avg_after_discount = sweater_avg_before_discount - sweater_discount →
  jeans_avg_after_discount = jeans_avg_before_discount - jeans_discount →
  sweater_avg_after_discount = shirt_avg_after_discount →
  jeans_avg_after_discount - sweater_avg_after_discount = 21 :=
by
  intros
  sorry

end NUMINAMATH_GPT_average_sweater_less_by_21_after_discount_l275_27563


namespace NUMINAMATH_GPT_find_number_l275_27509

theorem find_number
    (x: ℝ)
    (h: 0.60 * x = 0.40 * 30 + 18) : x = 50 :=
    sorry

end NUMINAMATH_GPT_find_number_l275_27509


namespace NUMINAMATH_GPT_remainder_77_pow_77_minus_15_mod_19_l275_27556

theorem remainder_77_pow_77_minus_15_mod_19 : (77^77 - 15) % 19 = 5 := by
  sorry

end NUMINAMATH_GPT_remainder_77_pow_77_minus_15_mod_19_l275_27556


namespace NUMINAMATH_GPT_greatest_fourth_term_arith_seq_sum_90_l275_27537

theorem greatest_fourth_term_arith_seq_sum_90 :
  ∃ a d : ℕ, 6 * a + 15 * d = 90 ∧ (∀ n : ℕ, n < 6 → a + n * d > 0) ∧ (a + 3 * d = 17) :=
by
  sorry

end NUMINAMATH_GPT_greatest_fourth_term_arith_seq_sum_90_l275_27537


namespace NUMINAMATH_GPT_duration_of_resulting_video_l275_27531

theorem duration_of_resulting_video 
    (vasya_walk_time : ℕ) (petya_walk_time : ℕ) 
    (sync_meet_point : ℕ) :
    vasya_walk_time = 8 → petya_walk_time = 5 → sync_meet_point = sync_meet_point → 
    (vasya_walk_time - sync_meet_point + petya_walk_time) = 5 :=
by
  intros
  sorry

end NUMINAMATH_GPT_duration_of_resulting_video_l275_27531


namespace NUMINAMATH_GPT_coris_aunt_age_today_l275_27587

variable (Cori_age_now : ℕ) (age_diff : ℕ)

theorem coris_aunt_age_today (H1 : Cori_age_now = 3) (H2 : ∀ (Cori_age5 Aunt_age5 : ℕ), Cori_age5 = Cori_age_now + 5 → Aunt_age5 = 3 * Cori_age5 → Aunt_age5 - 5 = age_diff) :
  age_diff = 19 := 
by
  intros
  sorry

end NUMINAMATH_GPT_coris_aunt_age_today_l275_27587


namespace NUMINAMATH_GPT_inner_tetrahedron_volume_l275_27562

def volume_of_inner_tetrahedron(cube_side : ℕ) : ℚ :=
  let base_area := (cube_side * cube_side) / 2
  let height := cube_side
  let original_tetra_volume := (1 / 3) * base_area * height
  let inner_tetra_volume := original_tetra_volume / 8
  inner_tetra_volume

theorem inner_tetrahedron_volume {cube_side : ℕ} (h : cube_side = 2) : 
  volume_of_inner_tetrahedron cube_side = 1 / 6 := 
by
  rw [h]
  unfold volume_of_inner_tetrahedron 
  norm_num
  sorry

end NUMINAMATH_GPT_inner_tetrahedron_volume_l275_27562


namespace NUMINAMATH_GPT_michael_completes_in_50_days_l275_27544

theorem michael_completes_in_50_days :
  ∀ {M A W : ℝ},
    (W / M + W / A = W / 20) →
    (14 * W / 20 + 10 * W / A = W) →
    M = 50 :=
by
  sorry

end NUMINAMATH_GPT_michael_completes_in_50_days_l275_27544


namespace NUMINAMATH_GPT_locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l275_27582

/-- Given a circle with center at point P passes through point A (1,0) 
    and is tangent to the line x = -1, the locus of point P is the parabola C. -/
theorem locus_of_P_is_parabola (P A : ℝ × ℝ) (x y : ℝ):
  (A = (1, 0)) → (P.1 + 1)^2 + P.2^2 = 0 → y^2 = 4 * x := 
sorry

/-- If the line passing through point H(4, 0) intersects the parabola 
    C (denoted by y^2 = 4x) at points M and N, and T is any point on 
    the line x = -4, then the slopes of lines TM, TH, and TN form an 
    arithmetic sequence. -/
theorem slopes_form_arithmetic_sequence (H M N T : ℝ × ℝ) (m n k : ℝ): 
  (H = (4, 0)) → (T.1 = -4) → 
  (M.1, M.2) = (k^2, 4*k) ∧ (N.1, N.2) = (m^2, 4*m) → 
  ((T.2 - M.2) / (T.1 - M.1) + (T.2 - N.2) / (T.1 - N.1)) = 
  2 * (T.2 / -8) := 
sorry

end NUMINAMATH_GPT_locus_of_P_is_parabola_slopes_form_arithmetic_sequence_l275_27582


namespace NUMINAMATH_GPT_sin_alpha_cos_half_beta_minus_alpha_l275_27540

open Real

noncomputable def problem_condition (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧
  0 < β ∧ β < π / 2 ∧
  sin (π / 3 - α) = 3 / 5 ∧
  cos (β / 2 - π / 3) = 2 * sqrt 5 / 5

theorem sin_alpha (α β : ℝ) (h : problem_condition α β) : 
  sin α = (4 * sqrt 3 - 3) / 10 := sorry

theorem cos_half_beta_minus_alpha (α β : ℝ) (h : problem_condition α β) :
  cos (β / 2 - α) = 11 * sqrt 5 / 25 := sorry

end NUMINAMATH_GPT_sin_alpha_cos_half_beta_minus_alpha_l275_27540


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l275_27549

theorem hyperbola_eccentricity (a b c : ℝ) (h : (c^2 - a^2 = 5 * a^2)) (hb : a / b = 2) :
  (c / a = Real.sqrt 5) :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l275_27549


namespace NUMINAMATH_GPT_rectangle_area_coefficient_l275_27573

theorem rectangle_area_coefficient (length width d k : ℝ) 
(h1 : length / width = 5 / 2) 
(h2 : d^2 = length^2 + width^2) 
(h3 : k = 10 / 29) :
  (length * width = k * d^2) :=
by
  sorry

end NUMINAMATH_GPT_rectangle_area_coefficient_l275_27573


namespace NUMINAMATH_GPT_ones_digit_of_7_pow_53_l275_27522

theorem ones_digit_of_7_pow_53 : (7^53 % 10) = 7 := by
  sorry

end NUMINAMATH_GPT_ones_digit_of_7_pow_53_l275_27522


namespace NUMINAMATH_GPT_quadratic_properties_l275_27529

noncomputable def quadratic (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

theorem quadratic_properties (a b c : ℝ)
  (root_neg1 : quadratic a b c (-1) = 0)
  (ineq_condition : ∀ x : ℝ, (quadratic a b c x - x) * (quadratic a b c x - (x^2 + 1) / 2) ≤ 0) :
  quadratic a b c 1 = 1 ∧ ∀ x : ℝ, quadratic a b c x = (1 / 4) * x^2 + (1 / 2) * x + (1 / 4) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_properties_l275_27529


namespace NUMINAMATH_GPT_am_gm_inequality_l275_27596

theorem am_gm_inequality (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  (a - b)^2 / (8 * a) < (a + b) / 2 - (Real.sqrt (a * b)) ∧ 
  (a + b) / 2 - (Real.sqrt (a * b)) < (a - b)^2 / (8 * b) := 
sorry

end NUMINAMATH_GPT_am_gm_inequality_l275_27596


namespace NUMINAMATH_GPT_impossibility_of_sum_sixteen_l275_27594

open Nat

def max_roll_value : ℕ := 6
def sum_of_two_rolls (a b : ℕ) : ℕ := a + b

theorem impossibility_of_sum_sixteen :
  ∀ a b : ℕ, (1 ≤ a ∧ a ≤ max_roll_value) ∧ (1 ≤ b ∧ b ≤ max_roll_value) → sum_of_two_rolls a b ≠ 16 :=
by
  intros a b h
  sorry

end NUMINAMATH_GPT_impossibility_of_sum_sixteen_l275_27594


namespace NUMINAMATH_GPT_exists_finite_set_with_subset_relation_l275_27518

-- Definition of an ordered set (E, ≤)
variable {E : Type} [LE E]

theorem exists_finite_set_with_subset_relation (E : Type) [LE E] :
  ∃ (F : Set (Set E)) (X : E → Set E), 
  (∀ (e1 e2 : E), e1 ≤ e2 ↔ X e2 ⊆ X e1) :=
by
  -- The proof is initially skipped, as per instructions
  sorry

end NUMINAMATH_GPT_exists_finite_set_with_subset_relation_l275_27518


namespace NUMINAMATH_GPT_actual_distance_traveled_l275_27557

theorem actual_distance_traveled (D : ℕ) (h : (D:ℚ) / 12 = (D + 20) / 16) : D = 60 :=
sorry

end NUMINAMATH_GPT_actual_distance_traveled_l275_27557


namespace NUMINAMATH_GPT_silver_value_percentage_l275_27570

theorem silver_value_percentage
  (side_length : ℝ) (weight_per_cubic_inch : ℝ) (price_per_ounce : ℝ) 
  (selling_price : ℝ) (volume : ℝ) (weight : ℝ) (silver_value : ℝ) 
  (percentage_sold : ℝ ) 
  (h1 : side_length = 3) 
  (h2 : weight_per_cubic_inch = 6) 
  (h3 : price_per_ounce = 25)
  (h4 : selling_price = 4455)
  (h5 : volume = side_length^3)
  (h6 : weight = volume * weight_per_cubic_inch)
  (h7 : silver_value = weight * price_per_ounce)
  (h8 : percentage_sold = (selling_price / silver_value) * 100) :
  percentage_sold = 110 :=
by
  sorry

end NUMINAMATH_GPT_silver_value_percentage_l275_27570


namespace NUMINAMATH_GPT_lcm_multiplied_by_2_is_72x_l275_27566

-- Define the denominators
def denom1 (x : ℕ) := 4 * x
def denom2 (x : ℕ) := 6 * x
def denom3 (x : ℕ) := 9 * x

-- Define the least common multiple of three natural numbers
def lcm_three (a b c : ℕ) := Nat.lcm a (Nat.lcm b c)

-- Define the multiplication by 2
def multiply_by_2 (n : ℕ) := 2 * n

-- Define the final result
def final_result (x : ℕ) := 72 * x

-- The proof statement
theorem lcm_multiplied_by_2_is_72x (x : ℕ): 
  multiply_by_2 (lcm_three (denom1 x) (denom2 x) (denom3 x)) = final_result x := 
by
  sorry

end NUMINAMATH_GPT_lcm_multiplied_by_2_is_72x_l275_27566


namespace NUMINAMATH_GPT_sum_of_r_s_l275_27532

theorem sum_of_r_s (m : ℝ) (x : ℝ) (y : ℝ) (r s : ℝ) 
  (parabola_eqn : y = x^2 + 4) 
  (point_Q : (x, y) = (10, 5)) 
  (roots_rs : ∀ (m : ℝ), m^2 - 40*m + 4 = 0 → r < m → m < s)
  : r + s = 40 := 
sorry

end NUMINAMATH_GPT_sum_of_r_s_l275_27532
