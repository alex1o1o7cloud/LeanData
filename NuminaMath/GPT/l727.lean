import Mathlib

namespace NUMINAMATH_GPT_jellybean_problem_l727_72776

theorem jellybean_problem:
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  black + green + orange = 27 →
  green - orange = 1 :=
by
  intros black green orange h_black h_green h_total
  sorry

end NUMINAMATH_GPT_jellybean_problem_l727_72776


namespace NUMINAMATH_GPT_range_of_a_intersection_nonempty_range_of_a_intersection_A_l727_72740

noncomputable def A (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 3}
def B : Set ℝ := {x | x < -1 ∨ x > 5}

theorem range_of_a_intersection_nonempty (a : ℝ) : (A a ∩ B ≠ ∅) ↔ (a < -1 ∨ a > 2) :=
sorry

theorem range_of_a_intersection_A (a : ℝ) : (A a ∩ B = A a) ↔ (a < -4 ∨ a > 5) :=
sorry

end NUMINAMATH_GPT_range_of_a_intersection_nonempty_range_of_a_intersection_A_l727_72740


namespace NUMINAMATH_GPT_radius_of_tangent_sphere_l727_72797

theorem radius_of_tangent_sphere (r1 r2 : ℝ) (h : r1 = 12 ∧ r2 = 3) :
  ∃ r : ℝ, (r = 6) :=
by
  sorry

end NUMINAMATH_GPT_radius_of_tangent_sphere_l727_72797


namespace NUMINAMATH_GPT_minimum_AP_BP_l727_72791

def A : ℝ × ℝ := (2, 0)
def B : ℝ × ℝ := (7, 3)
def parabola (P : ℝ × ℝ) : Prop := P.2 * P.2 = 8 * P.1

noncomputable def distance (P Q : ℝ × ℝ) : ℝ := ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem minimum_AP_BP : 
  ∀ (P : ℝ × ℝ), parabola P → distance A P + distance B P ≥ 3 * Real.sqrt 10 :=
by 
  intros P hP
  sorry

end NUMINAMATH_GPT_minimum_AP_BP_l727_72791


namespace NUMINAMATH_GPT_brooke_butter_price_l727_72728

variables (price_per_gallon_of_milk : ℝ)
variables (gallons_to_butter_conversion : ℝ)
variables (number_of_cows : ℕ)
variables (milk_per_cow : ℝ)
variables (number_of_customers : ℕ)
variables (milk_demand_per_customer : ℝ)
variables (total_earnings : ℝ)

theorem brooke_butter_price :
    price_per_gallon_of_milk = 3 →
    gallons_to_butter_conversion = 2 →
    number_of_cows = 12 →
    milk_per_cow = 4 →
    number_of_customers = 6 →
    milk_demand_per_customer = 6 →
    total_earnings = 144 →
    (total_earnings - number_of_customers * milk_demand_per_customer * price_per_gallon_of_milk) /
    (number_of_cows * milk_per_cow - number_of_customers * milk_demand_per_customer) *
    gallons_to_butter_conversion = 1.50 :=
by { sorry }

end NUMINAMATH_GPT_brooke_butter_price_l727_72728


namespace NUMINAMATH_GPT_number_of_girls_in_school_l727_72723

theorem number_of_girls_in_school :
  ∃ G B : ℕ, 
    G + B = 1600 ∧
    (G * 200 / 1600) - 20 = (B * 200 / 1600) ∧
    G = 860 :=
by
  sorry

end NUMINAMATH_GPT_number_of_girls_in_school_l727_72723


namespace NUMINAMATH_GPT_sequence_a_n_l727_72794

theorem sequence_a_n (S : ℕ → ℕ) (a : ℕ → ℕ) :
  (∀ n : ℕ, S n = 3 + 2^n) →
  (a 1 = 5) ∧ (∀ n : ℕ, n ≥ 2 → a n = 2^(n-1)) ↔ 
  (∀ n : ℕ, a n = if n = 1 then 5 else 2^(n-1)) :=
by
  sorry

end NUMINAMATH_GPT_sequence_a_n_l727_72794


namespace NUMINAMATH_GPT_abs_sq_lt_self_iff_l727_72716

theorem abs_sq_lt_self_iff {x : ℝ} : abs x * abs x < x ↔ (0 < x ∧ x < 1) ∨ (x < -1) :=
by
  sorry

end NUMINAMATH_GPT_abs_sq_lt_self_iff_l727_72716


namespace NUMINAMATH_GPT_gcd_polynomials_l727_72707

-- State the problem in Lean 4.
theorem gcd_polynomials (b : ℤ) (h : ∃ k : ℤ, b = 7768 * 2 * k) : 
  Int.gcd (7 * b^2 + 55 * b + 125) (3 * b + 10) = 10 :=
by
  sorry

end NUMINAMATH_GPT_gcd_polynomials_l727_72707


namespace NUMINAMATH_GPT_line_through_P_perpendicular_l727_72739

theorem line_through_P_perpendicular 
  (P : ℝ × ℝ) (a b c : ℝ) (hP : P = (-1, 3)) (hline : a = 1 ∧ b = -2 ∧ c = 3) :
  ∃ (a' b' c' : ℝ), (a' * P.1 + b' * P.2 + c' = 0) ∧ (a = b' ∧ b = -a') ∧ (a' = 2 ∧ b' = 1 ∧ c' = -1) := 
by
  use 2, 1, -1
  sorry

end NUMINAMATH_GPT_line_through_P_perpendicular_l727_72739


namespace NUMINAMATH_GPT_sale_in_first_month_l727_72766

theorem sale_in_first_month 
  (sale_2 : ℝ) (sale_3 : ℝ) (sale_4 : ℝ) (sale_5 : ℝ) (sale_6 : ℝ) (avg_sale : ℝ)
  (h_sale_2 : sale_2 = 5366) (h_sale_3 : sale_3 = 5808) 
  (h_sale_4 : sale_4 = 5399) (h_sale_5 : sale_5 = 6124) 
  (h_sale_6 : sale_6 = 4579) (h_avg_sale : avg_sale = 5400) :
  ∃ (sale_1 : ℝ), sale_1 = 5124 :=
by
  let total_sales := avg_sale * 6
  let known_sales := sale_2 + sale_3 + sale_4 + sale_5 + sale_6
  have h_total_sales : total_sales = 32400 := by sorry
  have h_known_sales : known_sales = 27276 := by sorry
  let sale_1 := total_sales - known_sales
  use sale_1
  have h_sale_1 : sale_1 = 5124 := by sorry
  exact h_sale_1

end NUMINAMATH_GPT_sale_in_first_month_l727_72766


namespace NUMINAMATH_GPT_tree_sidewalk_space_l727_72773

theorem tree_sidewalk_space (num_trees : ℕ) (tree_distance: ℝ) (total_road_length: ℝ): 
  num_trees = 13 → 
  tree_distance = 12 → 
  total_road_length = 157 → 
  (total_road_length - tree_distance * (num_trees - 1)) / num_trees = 1 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  simp
  sorry

end NUMINAMATH_GPT_tree_sidewalk_space_l727_72773


namespace NUMINAMATH_GPT_percentage_temporary_employees_is_correct_l727_72765

noncomputable def percentage_temporary_employees
    (technicians_percentage : ℝ) (skilled_laborers_percentage : ℝ) (unskilled_laborers_percentage : ℝ)
    (permanent_technicians_percentage : ℝ) (permanent_skilled_laborers_percentage : ℝ)
    (permanent_unskilled_laborers_percentage : ℝ) : ℝ :=
  let total_workers : ℝ := 100
  let total_temporary_technicians := technicians_percentage * (1 - permanent_technicians_percentage / 100)
  let total_temporary_skilled_laborers := skilled_laborers_percentage * (1 - permanent_skilled_laborers_percentage / 100)
  let total_temporary_unskilled_laborers := unskilled_laborers_percentage * (1 - permanent_unskilled_laborers_percentage / 100)
  let total_temporary_workers := total_temporary_technicians + total_temporary_skilled_laborers + total_temporary_unskilled_laborers
  (total_temporary_workers / total_workers) * 100

theorem percentage_temporary_employees_is_correct :
  percentage_temporary_employees 40 35 25 60 45 35 = 51.5 :=
by
  sorry

end NUMINAMATH_GPT_percentage_temporary_employees_is_correct_l727_72765


namespace NUMINAMATH_GPT_fg_difference_l727_72742

noncomputable def f (x : ℝ) : ℝ := x^2 - 3 * x + 7
noncomputable def g (x : ℝ) : ℝ := 2 * x + 4

theorem fg_difference : f (g 3) - g (f 3) = 59 :=
by
  sorry

end NUMINAMATH_GPT_fg_difference_l727_72742


namespace NUMINAMATH_GPT_g_triple_3_eq_31_l727_72715

def g (n : ℕ) : ℕ :=
  if n ≤ 5 then n^2 + 1 else 2 * n - 3

theorem g_triple_3_eq_31 : g (g (g 3)) = 31 := by
  sorry

end NUMINAMATH_GPT_g_triple_3_eq_31_l727_72715


namespace NUMINAMATH_GPT_faster_speed_l727_72779

variable (v : ℝ)
variable (distance fasterDistance speed time : ℝ)
variable (h_distance : distance = 24)
variable (h_speed : speed = 4)
variable (h_fasterDistance : fasterDistance = distance + 6)
variable (h_time : time = distance / speed)

theorem faster_speed (h : 6 = fasterDistance / v) : v = 5 :=
by
  sorry

end NUMINAMATH_GPT_faster_speed_l727_72779


namespace NUMINAMATH_GPT_ratio_of_members_l727_72785

theorem ratio_of_members (r p : ℕ) (h1 : 5 * r + 12 * p = 8 * (r + p)) : (r / p : ℚ) = 4 / 3 := by
  sorry -- This is a placeholder for the actual proof.

end NUMINAMATH_GPT_ratio_of_members_l727_72785


namespace NUMINAMATH_GPT_find_r_l727_72783

-- Define the basic conditions based on the given problem.
def pr (r : ℕ) := 360 / 6
def p := pr 4 / 4
def cr (c r : ℕ) := 6 * c * r

-- Prove that r = 4 given the conditions.
theorem find_r (r : ℕ) : r = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_r_l727_72783


namespace NUMINAMATH_GPT_convert_234_base5_to_binary_l727_72753

def base5_to_decimal (n : Nat) : Nat :=
  2 * 5^2 + 3 * 5^1 + 4 * 5^0

def decimal_to_binary (n : Nat) : List Nat :=
  let rec to_binary_aux (n : Nat) (accum : List Nat) : List Nat :=
    if n = 0 then accum
    else to_binary_aux (n / 2) ((n % 2) :: accum)
  to_binary_aux n []

theorem convert_234_base5_to_binary :
  (base5_to_decimal 234 = 69) ∧ (decimal_to_binary 69 = [1,0,0,0,1,0,1]) :=
by
  sorry

end NUMINAMATH_GPT_convert_234_base5_to_binary_l727_72753


namespace NUMINAMATH_GPT_books_sold_over_summer_l727_72706

theorem books_sold_over_summer (n l t : ℕ) (h1 : n = 37835) (h2 : l = 143) (h3 : t = 271) : 
  t - l = 128 :=
by
  sorry

end NUMINAMATH_GPT_books_sold_over_summer_l727_72706


namespace NUMINAMATH_GPT_jake_work_hours_l727_72749

-- Definitions for the conditions
def initial_debt : ℝ := 100
def amount_paid : ℝ := 40
def work_rate : ℝ := 15

-- The main theorem stating the number of hours Jake needs to work
theorem jake_work_hours : ∃ h : ℝ, initial_debt - amount_paid = h * work_rate ∧ h = 4 :=
by 
  -- sorry placeholder indicating the proof is not required
  sorry

end NUMINAMATH_GPT_jake_work_hours_l727_72749


namespace NUMINAMATH_GPT_lily_received_books_l727_72708

def mike_books : ℕ := 45
def corey_books : ℕ := 2 * mike_books
def mike_gave_lily : ℕ := 10
def corey_gave_lily : ℕ := mike_gave_lily + 15
def lily_books_received : ℕ := mike_gave_lily + corey_gave_lily

theorem lily_received_books : lily_books_received = 35 := by
  sorry

end NUMINAMATH_GPT_lily_received_books_l727_72708


namespace NUMINAMATH_GPT_integer_solutions_count_l727_72703

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end NUMINAMATH_GPT_integer_solutions_count_l727_72703


namespace NUMINAMATH_GPT_simplify_expression_l727_72787

def a : ℚ := (3 / 4) * 60
def b : ℚ := (8 / 5) * 60
def c : ℚ := 63

theorem simplify_expression : a - b + c = 12 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l727_72787


namespace NUMINAMATH_GPT_problem_solution_l727_72764

theorem problem_solution (a : ℚ) (h : 3 * a + 6 * a / 4 = 6) : a = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_problem_solution_l727_72764


namespace NUMINAMATH_GPT_range_of_expr_l727_72755

noncomputable def expr (x y : ℝ) : ℝ := (x + 2 * y + 3) / (x + 1)

theorem range_of_expr : 
  (∀ x y : ℝ, x ≥ 0 → y ≥ x → 4 * x + 3 * y ≤ 12 → 3 ≤ expr x y ∧ expr x y ≤ 11) :=
by
  sorry

end NUMINAMATH_GPT_range_of_expr_l727_72755


namespace NUMINAMATH_GPT_area_triangle_AMC_l727_72750

open Real

-- Definitions: Define the points A, B, C, D such that they form a rectangle
-- Define midpoint M of \overline{AD}

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def A : Point := {x := 0, y := 0}
noncomputable def B : Point := {x := 6, y := 0}
noncomputable def D : Point := {x := 0, y := 8}
noncomputable def C : Point := {x := 6, y := 8}
noncomputable def M : Point := {x := 0, y := 4} -- midpoint of AD

-- Function to compute the area of triangle AMC
noncomputable def triangle_area (A M C : Point) : ℝ :=
  (1 / 2 : ℝ) * abs ((A.x - C.x) * (M.y - A.y) - (A.x - M.x) * (C.y - A.y))

-- The theorem to prove
theorem area_triangle_AMC : triangle_area A M C = 12 :=
by
  sorry

end NUMINAMATH_GPT_area_triangle_AMC_l727_72750


namespace NUMINAMATH_GPT_distinct_pairs_l727_72705

theorem distinct_pairs (x y : ℝ) (h : x ≠ y) :
  x^100 - y^100 = 2^99 * (x - y) ∧ x^200 - y^200 = 2^199 * (x - y) ↔ (x = 2 ∧ y = 0) ∨ (x = 0 ∧ y = 2) :=
by
  sorry

end NUMINAMATH_GPT_distinct_pairs_l727_72705


namespace NUMINAMATH_GPT_sequence_a_n_a31_l727_72762

theorem sequence_a_n_a31 (a : ℕ → ℤ) 
  (h_initial : a 1 = 2)
  (h_recurrence : ∀ n : ℕ, a n + a (n + 1) + n^2 = 0) :
  a 31 = -463 :=
sorry

end NUMINAMATH_GPT_sequence_a_n_a31_l727_72762


namespace NUMINAMATH_GPT_find_f_neg2014_l727_72745

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x^3 + b * x - 2

theorem find_f_neg2014 (a b : ℝ) (h : f 2014 a b = 3) : f (-2014) a b = -7 :=
by sorry

end NUMINAMATH_GPT_find_f_neg2014_l727_72745


namespace NUMINAMATH_GPT_infinitely_many_primes_satisfying_condition_l727_72777

theorem infinitely_many_primes_satisfying_condition :
  ∀ k : Nat, ∃ p : Nat, Nat.Prime p ∧ ∃ n : Nat, n > 0 ∧ p ∣ (2014^(2^n) + 2014) := 
sorry

end NUMINAMATH_GPT_infinitely_many_primes_satisfying_condition_l727_72777


namespace NUMINAMATH_GPT_sym_diff_A_B_l727_72796

def set_diff (M N : Set ℝ) : Set ℝ := {x | x ∈ M ∧ x ∉ N}
def sym_diff (M N : Set ℝ) : Set ℝ := set_diff M N ∪ set_diff N M

def A : Set ℝ := {x | -1 ≤ x ∧ x < 1}
def B : Set ℝ := {x | x < 0}

theorem sym_diff_A_B :
  sym_diff A B = {x | x < -1} ∪ {x | 0 ≤ x ∧ x < 1} := by
  sorry

end NUMINAMATH_GPT_sym_diff_A_B_l727_72796


namespace NUMINAMATH_GPT_homework_problems_l727_72741

theorem homework_problems (p t : ℕ) (h1 : p >= 10) (h2 : pt = (2 * p + 2) * (t + 1)) : p * t = 60 :=
by
  sorry

end NUMINAMATH_GPT_homework_problems_l727_72741


namespace NUMINAMATH_GPT_problem_trip_l727_72711

noncomputable def validate_trip (a b c : ℕ) (t : ℕ) : Prop :=
  a ≥ 1 ∧ a + b + c ≤ 10 ∧ 60 * t = 9 * c - 10 * b

theorem problem_trip (a b c t : ℕ) (h : validate_trip a b c t) : a^2 + b^2 + c^2 = 26 :=
sorry

end NUMINAMATH_GPT_problem_trip_l727_72711


namespace NUMINAMATH_GPT_total_votes_election_l727_72709

theorem total_votes_election 
  (votes_A : ℝ) 
  (valid_votes_percentage : ℝ) 
  (invalid_votes_percentage : ℝ)
  (votes_candidate_A : ℝ) 
  (total_votes : ℝ) 
  (h1 : votes_A = 0.60) 
  (h2 : invalid_votes_percentage = 0.15) 
  (h3 : votes_candidate_A = 285600) 
  (h4 : valid_votes_percentage = 0.85) 
  (h5 : total_votes = 560000) 
  : 
  ((votes_A * valid_votes_percentage * total_votes) = votes_candidate_A) 
  := 
  by sorry

end NUMINAMATH_GPT_total_votes_election_l727_72709


namespace NUMINAMATH_GPT_largest_A_divisible_by_8_equal_quotient_remainder_l727_72717

theorem largest_A_divisible_by_8_equal_quotient_remainder :
  ∃ (A B C : ℕ), A = 8 * B + C ∧ B = C ∧ C < 8 ∧ A = 63 := by
  sorry

end NUMINAMATH_GPT_largest_A_divisible_by_8_equal_quotient_remainder_l727_72717


namespace NUMINAMATH_GPT_probability_red_or_white_l727_72718

def total_marbles : ℕ := 50
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def white_marbles : ℕ := total_marbles - (blue_marbles + red_marbles)

theorem probability_red_or_white : 
  (red_marbles + white_marbles) / total_marbles = 9 / 10 := 
  sorry

end NUMINAMATH_GPT_probability_red_or_white_l727_72718


namespace NUMINAMATH_GPT_length_PQ_calc_l727_72734

noncomputable def length_PQ 
  (F : ℝ × ℝ) 
  (P Q : ℝ × ℝ) 
  (hF : F = (1, 0)) 
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1) 
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1) 
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1) 
  (hx1x2 : P.1 + Q.1 = 9) : ℝ :=
|P.1 - Q.1|

theorem length_PQ_calc : ∀ F P Q
  (hF : F = (1, 0))
  (hP_on_parabola : P.2 ^ 2 = 4 * P.1)
  (hQ_on_parabola : Q.2 ^ 2 = 4 * Q.1)
  (hLine_through_focus : F.1 = ((P.2 - Q.2) / (P.1 - Q.1)) * 1 + P.1)
  (hx1x2 : P.1 + Q.1 = 9),
  length_PQ F P Q hF hP_on_parabola hQ_on_parabola hLine_through_focus hx1x2 = 11 := 
by
  sorry

end NUMINAMATH_GPT_length_PQ_calc_l727_72734


namespace NUMINAMATH_GPT_inequality_solution_set_l727_72763

open Set

theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | (x - 5 * a) * (x + a) > 0} = {x | x < 5 * a ∨ x > -a} :=
sorry

end NUMINAMATH_GPT_inequality_solution_set_l727_72763


namespace NUMINAMATH_GPT_solution_set_of_inequality_cauchy_schwarz_application_l727_72700

theorem solution_set_of_inequality (c : ℝ) (h1 : c > 0) (h2 : ∀ x : ℝ, x + |x - 2 * c| ≥ 2) : 
  c ≥ 1 :=
by
  sorry

theorem cauchy_schwarz_application (m p q r : ℝ) (h1 : m ≥ 1) (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : p + q + r = 3 * m) : 
  p^2 + q^2 + r^2 ≥ 3 :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_cauchy_schwarz_application_l727_72700


namespace NUMINAMATH_GPT_four_distinct_sum_equal_l727_72770

theorem four_distinct_sum_equal (S : Finset ℕ) (hS : S.card = 10) (hS_subset : S ⊆ Finset.range 38) :
  ∃ a b c d : ℕ, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧ a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by
  sorry

end NUMINAMATH_GPT_four_distinct_sum_equal_l727_72770


namespace NUMINAMATH_GPT_circles_intersect_if_and_only_if_l727_72774

theorem circles_intersect_if_and_only_if (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 = m ∧ x^2 + y^2 + 6 * x - 8 * y - 11 = 0) ↔ (1 < m ∧ m < 121) :=
by
  sorry

end NUMINAMATH_GPT_circles_intersect_if_and_only_if_l727_72774


namespace NUMINAMATH_GPT_factor_expression_l727_72746

noncomputable def factored_expression (x : ℝ) : ℝ :=
  5 * x * (x + 2) + 9 * (x + 2)

theorem factor_expression (x : ℝ) : 
  factored_expression x = (x + 2) * (5 * x + 9) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l727_72746


namespace NUMINAMATH_GPT_find_numer_denom_n_l727_72757

theorem find_numer_denom_n (n : ℕ) 
    (h : (2 + n) / (7 + n) = (3 : ℤ) / 4) : n = 13 := sorry

end NUMINAMATH_GPT_find_numer_denom_n_l727_72757


namespace NUMINAMATH_GPT_combine_sum_l727_72726

def A (n m : Nat) : Nat := n.factorial / (n - m).factorial
def C (n m : Nat) : Nat := n.factorial / (m.factorial * (n - m).factorial)

theorem combine_sum (n m : Nat) (hA : A n m = 272) (hC : C n m = 136) : m + n = 19 := by
  sorry

end NUMINAMATH_GPT_combine_sum_l727_72726


namespace NUMINAMATH_GPT_largest_consecutive_even_integer_l727_72760

theorem largest_consecutive_even_integer (n : ℕ) (h : 5 * n - 20 = 2 * 15 * 16 / 2) : n = 52 :=
sorry

end NUMINAMATH_GPT_largest_consecutive_even_integer_l727_72760


namespace NUMINAMATH_GPT_originally_anticipated_profit_margin_l727_72780

theorem originally_anticipated_profit_margin (decrease_percent increase_percent : ℝ) (original_price current_price : ℝ) (selling_price : ℝ) :
  decrease_percent = 6.4 → 
  increase_percent = 8 → 
  original_price = 1 → 
  current_price = original_price - original_price * decrease_percent / 100 → 
  selling_price = original_price * (1 + x / 100) → 
  selling_price = current_price * (1 + (x + increase_percent) / 100) →
  x = 117 :=
by
  intros h_dec_perc h_inc_perc h_org_price h_cur_price h_selling_price_orig h_selling_price_cur
  sorry

end NUMINAMATH_GPT_originally_anticipated_profit_margin_l727_72780


namespace NUMINAMATH_GPT_how_many_bananas_l727_72710

theorem how_many_bananas (total_fruit apples oranges : ℕ) 
  (h_total : total_fruit = 12) (h_apples : apples = 3) (h_oranges : oranges = 5) :
  total_fruit - apples - oranges = 4 :=
by
  sorry

end NUMINAMATH_GPT_how_many_bananas_l727_72710


namespace NUMINAMATH_GPT_three_digit_log3_eq_whole_and_log3_log9_eq_whole_l727_72747

noncomputable def logBase (b : ℝ) (x : ℝ) : ℝ :=
  Real.log x / Real.log b

theorem three_digit_log3_eq_whole_and_log3_log9_eq_whole (n : ℕ) (hn : 100 ≤ n ∧ n ≤ 999) (hlog3 : ∃ x : ℤ, logBase 3 n = x) (hlog3log9 : ∃ k : ℤ, logBase 3 n + logBase 9 n = k) :
  n = 729 := sorry

end NUMINAMATH_GPT_three_digit_log3_eq_whole_and_log3_log9_eq_whole_l727_72747


namespace NUMINAMATH_GPT_altitudes_not_form_triangle_l727_72719

theorem altitudes_not_form_triangle (h₁ h₂ h₃ : ℝ) :
  ¬(h₁ = 5 ∧ h₂ = 12 ∧ h₃ = 13 ∧ ∃ a b c : ℝ, a * h₁ = b * h₂ ∧ b * h₂ = c * h₃ ∧
    a < b + c ∧ b < a + c ∧ c < a + b) :=
by sorry

end NUMINAMATH_GPT_altitudes_not_form_triangle_l727_72719


namespace NUMINAMATH_GPT_incorrect_transformation_l727_72768

theorem incorrect_transformation (a b : ℤ) : ¬ (a / b = (a + 1) / (b + 1)) :=
sorry

end NUMINAMATH_GPT_incorrect_transformation_l727_72768


namespace NUMINAMATH_GPT_gcd_360_1260_l727_72713

theorem gcd_360_1260 : gcd 360 1260 = 180 := by
  /- 
  Prime factorization of 360 and 1260 is given:
  360 = 2^3 * 3^2 * 5
  1260 = 2^2 * 3^2 * 5 * 7
  These conditions are implicitly used to deduce the answer.
  -/
  sorry

end NUMINAMATH_GPT_gcd_360_1260_l727_72713


namespace NUMINAMATH_GPT_total_animals_correct_l727_72729

-- Define the number of aquariums and the number of animals per aquarium.
def num_aquariums : ℕ := 26
def animals_per_aquarium : ℕ := 2

-- Define the total number of saltwater animals.
def total_animals : ℕ := num_aquariums * animals_per_aquarium

-- The statement we want to prove.
theorem total_animals_correct : total_animals = 52 := by
  -- Proof is omitted.
  sorry

end NUMINAMATH_GPT_total_animals_correct_l727_72729


namespace NUMINAMATH_GPT_largest_five_digit_integer_congruent_to_16_mod_25_l727_72798

theorem largest_five_digit_integer_congruent_to_16_mod_25 :
  ∃ x : ℤ, x % 25 = 16 ∧ x < 100000 ∧ ∀ y : ℤ, y % 25 = 16 → y < 100000 → y ≤ x :=
by
  sorry

end NUMINAMATH_GPT_largest_five_digit_integer_congruent_to_16_mod_25_l727_72798


namespace NUMINAMATH_GPT_total_amount_received_l727_72722

theorem total_amount_received (B : ℝ) (h1 : (1/3) * B = 36) : (2/3 * B) * 4 = 288 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_received_l727_72722


namespace NUMINAMATH_GPT_mikes_ride_is_46_miles_l727_72775

-- Define the conditions and the question in Lean 4
variable (M : ℕ)

-- Mike's cost formula
def mikes_cost (M : ℕ) : ℚ := 2.50 + 0.25 * M

-- Annie's total cost
def annies_miles : ℕ := 26
def annies_cost : ℚ := 2.50 + 5.00 + 0.25 * annies_miles

-- The proof statement
theorem mikes_ride_is_46_miles (h : mikes_cost M = annies_cost) : M = 46 :=
by sorry

end NUMINAMATH_GPT_mikes_ride_is_46_miles_l727_72775


namespace NUMINAMATH_GPT_triangle_inequality_l727_72756

variables {a b c : ℝ} {α : ℝ}

-- Assuming a, b, c are sides of a triangle
def triangle_sides (a b c : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (c > 0)

-- Cosine rule definition
noncomputable def cos_alpha (a b c : ℝ) : ℝ := (b^2 + c^2 - a^2) / (2 * b * c)

theorem triangle_inequality (h_sides: triangle_sides a b c) (h_cos : α = cos_alpha a b c) :
  (2 * b * c * (cos_alpha a b c)) / (b + c) < b + c - a
  ∧ b + c - a < 2 * b * c / a :=
by
  sorry

end NUMINAMATH_GPT_triangle_inequality_l727_72756


namespace NUMINAMATH_GPT_initial_blocks_l727_72724

variable (x : ℕ)

theorem initial_blocks (h : x + 30 = 65) : x = 35 := by
  sorry

end NUMINAMATH_GPT_initial_blocks_l727_72724


namespace NUMINAMATH_GPT_interval_where_f_increasing_l727_72769

noncomputable def f (x : ℝ) : ℝ := Real.log (4 * x - x^2) / Real.log (1 / 2)

theorem interval_where_f_increasing : ∀ x : ℝ, 2 ≤ x ∧ x < 4 → f x < f (x + 1) :=
by 
  sorry

end NUMINAMATH_GPT_interval_where_f_increasing_l727_72769


namespace NUMINAMATH_GPT_compute_a_d_sum_l727_72772

variables {a1 a2 a3 d1 d2 d3 : ℝ}

theorem compute_a_d_sum
  (h : ∀ x : ℝ, x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3)) :
  a1 * d1 + a2 * d2 + a3 * d3 = 1 :=
  sorry

end NUMINAMATH_GPT_compute_a_d_sum_l727_72772


namespace NUMINAMATH_GPT_trig_identity_solutions_l727_72790

open Real

theorem trig_identity_solutions (x : ℝ) (k n : ℤ) :
  (4 * sin x * cos (π / 2 - x) + 4 * sin (π + x) * cos x + 2 * sin (3 * π / 2 - x) * cos (π + x) = 1) ↔ 
  (∃ k : ℤ, x = arctan (1 / 3) + π * k) ∨ (∃ n : ℤ, x = π / 4 + π * n) := 
sorry

end NUMINAMATH_GPT_trig_identity_solutions_l727_72790


namespace NUMINAMATH_GPT_tom_buys_oranges_l727_72712

theorem tom_buys_oranges (o a : ℕ) (h₁ : o + a = 7) (h₂ : (90 * o + 60 * a) % 100 = 0) : o = 6 := 
by 
  sorry

end NUMINAMATH_GPT_tom_buys_oranges_l727_72712


namespace NUMINAMATH_GPT_remove_terms_sum_l727_72714

theorem remove_terms_sum :
  let s := (1/3 + 1/5 + 1/7 + 1/9 + 1/11 + 1/13 + 1/15 : ℚ)
  s = 16339/15015 →
  (1/13 + 1/15 = 2061/5005) →
  s - (1/13 + 1/15) = 3/2 :=
by
  intros s hs hremove
  have hrem : (s - (1/13 + 1/15 : ℚ) = 3/2) ↔ (16339/15015 - 2061/5005 = 3/2) := sorry
  exact hrem.mpr sorry

end NUMINAMATH_GPT_remove_terms_sum_l727_72714


namespace NUMINAMATH_GPT_simplify_expression_l727_72733

theorem simplify_expression (x y : ℝ) : (x^2 + y^2)⁻¹ * (x⁻¹ + y⁻¹) = (x^3 * y + x * y^3)⁻¹ * (x + y) :=
by sorry

end NUMINAMATH_GPT_simplify_expression_l727_72733


namespace NUMINAMATH_GPT_ratio_of_areas_l727_72731

theorem ratio_of_areas (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : r₁ = (Real.sqrt 2) / 4)
  (h₂ : A₁ = π * r₁^2) (h₃ : r₂ = (Real.sqrt 2) * r₁) (h₄ : A₂ = π * r₂^2) :
  A₂ / A₁ = 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l727_72731


namespace NUMINAMATH_GPT_inscribed_sphere_tetrahedron_volume_l727_72786

theorem inscribed_sphere_tetrahedron_volume
  (R : ℝ) (S1 S2 S3 S4 : ℝ) :
  ∃ V : ℝ, V = (1 / 3) * R * (S1 + S2 + S3 + S4) :=
sorry

end NUMINAMATH_GPT_inscribed_sphere_tetrahedron_volume_l727_72786


namespace NUMINAMATH_GPT__l727_72704

lemma triangle_inequality_theorem (a b c : ℝ) : 
  a + b > c ∧ a + c > b ∧ b + c > a ↔ 
  (a > 0 ∧ b > 0 ∧ c > 0) := sorry

lemma no_triangle_1_2_3 : ¬ (1 + 2 > 3 ∧ 1 + 3 > 2 ∧ 2 + 3 > 1) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_3_8_5 : ¬ (3 + 8 > 5 ∧ 3 + 5 > 8 ∧ 8 + 5 > 3) := 
by simp [triangle_inequality_theorem]

lemma no_triangle_4_5_10 : ¬ (4 + 5 > 10 ∧ 4 + 10 > 5 ∧ 5 + 10 > 4) := 
by simp [triangle_inequality_theorem]

lemma triangle_4_5_6 : 4 + 5 > 6 ∧ 4 + 6 > 5 ∧ 5 + 6 > 4 := 
by simp [triangle_inequality_theorem]

end NUMINAMATH_GPT__l727_72704


namespace NUMINAMATH_GPT_no_triangle_satisfies_sine_eq_l727_72782

theorem no_triangle_satisfies_sine_eq (A B C : ℝ) (a b c : ℝ) 
  (hA: 0 < A) (hB: 0 < B) (hC: 0 < C) 
  (hA_ineq: A < π) (hB_ineq: B < π) (hC_ineq: C < π) 
  (h_sum: A + B + C = π) 
  (sin_eq: Real.sin A + Real.sin B = Real.sin C)
  (h_tri_ineq: a + b > c ∧ a + c > b ∧ b + c > a) 
  (h_sines: a = 2 * (1) * Real.sin A ∧ b = 2 * (1) * Real.sin B ∧ c = 2 * (1) * Real.sin C) :
  False :=
sorry

end NUMINAMATH_GPT_no_triangle_satisfies_sine_eq_l727_72782


namespace NUMINAMATH_GPT_additional_charge_per_international_letter_l727_72793

-- Definitions based on conditions
def standard_postage_per_letter : ℕ := 108
def num_international_letters : ℕ := 2
def total_cost : ℕ := 460
def num_letters : ℕ := 4

-- Theorem stating the question
theorem additional_charge_per_international_letter :
  (total_cost - (num_letters * standard_postage_per_letter)) / num_international_letters = 14 :=
by
  sorry

end NUMINAMATH_GPT_additional_charge_per_international_letter_l727_72793


namespace NUMINAMATH_GPT_engineers_meeting_probability_l727_72737

theorem engineers_meeting_probability :
  ∀ (x y z : ℝ), 
    (0 ≤ x ∧ x ≤ 2) → 
    (0 ≤ y ∧ y ≤ 2) → 
    (0 ≤ z ∧ z ≤ 2) → 
    (abs (x - y) ≤ 0.5) → 
    (abs (y - z) ≤ 0.5) → 
    (abs (z - x) ≤ 0.5) → 
    Π (volume_region : ℝ) (total_volume : ℝ),
    (volume_region = 1.5 * 1.5 * 1.5) → 
    (total_volume = 2 * 2 * 2) → 
    (volume_region / total_volume = 0.421875) :=
by
  intros x y z hx hy hz hxy hyz hzx volume_region total_volume hr ht
  sorry

end NUMINAMATH_GPT_engineers_meeting_probability_l727_72737


namespace NUMINAMATH_GPT_geometric_sequence_sum_l727_72752

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

variables {a : ℕ → ℝ}

theorem geometric_sequence_sum (h1 : is_geometric_sequence a) (h2 : a 1 * a 2 = 8 * a 0)
  (h3 : (a 3 + 2 * a 4) / 2 = 20) :
  (a 0 * (2^5 - 1)) = 31 :=
by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l727_72752


namespace NUMINAMATH_GPT_good_students_l727_72701

theorem good_students (E B : ℕ) (h1 : E + B = 25) (h2 : 12 < B) (h3 : B = 3 * (E - 1)) :
  E = 5 ∨ E = 7 :=
by 
  sorry

end NUMINAMATH_GPT_good_students_l727_72701


namespace NUMINAMATH_GPT_center_of_circle_sum_eq_seven_l727_72702

theorem center_of_circle_sum_eq_seven 
  (h k : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 = 6 * x + 8 * y - 15 → (x - h)^2 + (y - k)^2 = 10) :
  h + k = 7 := 
sorry

end NUMINAMATH_GPT_center_of_circle_sum_eq_seven_l727_72702


namespace NUMINAMATH_GPT_avg_age_of_community_l727_72727

def ratio_of_populations (w m : ℕ) : Prop := w * 2 = m * 3
def avg_age (total_age population : ℚ) : ℚ := total_age / population

theorem avg_age_of_community 
    (k : ℕ)
    (total_women : ℕ := 3 * k) 
    (total_men : ℕ := 2 * k)
    (total_children : ℚ := (2 * k : ℚ) / 3)
    (avg_women_age : ℚ := 40)
    (avg_men_age : ℚ := 36)
    (avg_children_age : ℚ := 10)
    (total_women_age : ℚ := 40 * (3 * k))
    (total_men_age : ℚ := 36 * (2 * k))
    (total_children_age : ℚ := 10 * (total_children)) : 
    avg_age (total_women_age + total_men_age + total_children_age) (total_women + total_men + total_children) = 35 := 
    sorry

end NUMINAMATH_GPT_avg_age_of_community_l727_72727


namespace NUMINAMATH_GPT_distance_between_stations_l727_72743

theorem distance_between_stations 
  (distance_P_to_meeting : ℝ)
  (distance_Q_to_meeting : ℝ)
  (h1 : distance_P_to_meeting = 20 * 3)
  (h2 : distance_Q_to_meeting = 25 * 2)
  (h3 : distance_P_to_meeting + distance_Q_to_meeting = D) :
  D = 110 :=
by
  sorry

end NUMINAMATH_GPT_distance_between_stations_l727_72743


namespace NUMINAMATH_GPT_inequality_x_y_z_l727_72748

open Real

theorem inequality_x_y_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
    (x ^ 3) / ((1 + y) * (1 + z)) + (y ^ 3) / ((1 + z) * (1 + x)) + (z ^ 3) / ((1 + x) * (1 + y)) ≥ 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_inequality_x_y_z_l727_72748


namespace NUMINAMATH_GPT_perimeter_of_intersection_triangle_l727_72751

theorem perimeter_of_intersection_triangle :
  ∀ (P Q R : Type) (dist : P → Q → ℝ) (length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR : ℝ),
  (length_PQ = 150) →
  (length_QR = 250) →
  (length_PR = 200) →
  (seg_ellP = 75) →
  (seg_ellQ = 50) →
  (seg_ellR = 25) →
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  TU + US + ST = 266.67 :=
by
  intros P Q R dist length_PQ length_QR length_PR seg_ellP seg_ellQ seg_ellR hPQ hQR hPR hP hQ hR
  let TU := seg_ellP + seg_ellQ
  let US := seg_ellQ + seg_ellR
  let ST := seg_ellR + (seg_ellR * (length_QR / length_PQ))
  have : TU + US + ST = 266.67 := sorry
  exact this

end NUMINAMATH_GPT_perimeter_of_intersection_triangle_l727_72751


namespace NUMINAMATH_GPT_total_canoes_built_l727_72732

-- Definition of the conditions as suggested by the problem
def num_canoes_in_february : Nat := 5
def growth_rate : Nat := 3
def number_of_months : Nat := 5

-- Final statement to prove
theorem total_canoes_built : (num_canoes_in_february * (growth_rate^number_of_months - 1)) / (growth_rate - 1) = 605 := 
by sorry

end NUMINAMATH_GPT_total_canoes_built_l727_72732


namespace NUMINAMATH_GPT_pinedale_mall_distance_l727_72735

theorem pinedale_mall_distance 
  (speed : ℝ) (time_between_stops : ℝ) (num_stops : ℕ) (distance : ℝ) 
  (h_speed : speed = 60) 
  (h_time_between_stops : time_between_stops = 5 / 60) 
  (h_num_stops : ↑num_stops = 5) :
  distance = 25 :=
by
  sorry

end NUMINAMATH_GPT_pinedale_mall_distance_l727_72735


namespace NUMINAMATH_GPT_intersection_P_Q_l727_72795

def P : Set ℝ := {x | Real.log x / Real.log 2 < -1}
def Q : Set ℝ := {x | abs x < 1}

theorem intersection_P_Q : P ∩ Q = {x | 0 < x ∧ x < 1 / 2} := by
  sorry

end NUMINAMATH_GPT_intersection_P_Q_l727_72795


namespace NUMINAMATH_GPT_contrapositive_of_inequality_l727_72778

variable {a b c : ℝ}

theorem contrapositive_of_inequality (h : a + c ≤ b + c) : a ≤ b :=
sorry

end NUMINAMATH_GPT_contrapositive_of_inequality_l727_72778


namespace NUMINAMATH_GPT_largest_number_4597_l727_72736

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end NUMINAMATH_GPT_largest_number_4597_l727_72736


namespace NUMINAMATH_GPT_max_value_sqrt_abc_expression_l727_72721

theorem max_value_sqrt_abc_expression (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                       (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                       (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1) :=
sorry

end NUMINAMATH_GPT_max_value_sqrt_abc_expression_l727_72721


namespace NUMINAMATH_GPT_difference_one_third_0_333_l727_72744

theorem difference_one_third_0_333 :
  let one_third : ℚ := 1 / 3
  let three_hundred_thirty_three_thousandth : ℚ := 333 / 1000
  one_third - three_hundred_thirty_three_thousandth = 1 / 3000 :=
by
  sorry

end NUMINAMATH_GPT_difference_one_third_0_333_l727_72744


namespace NUMINAMATH_GPT_cost_comparison_l727_72767

-- Definitions based on the given conditions
def suit_price : ℕ := 200
def tie_price : ℕ := 40
def num_suits : ℕ := 20
def discount_rate : ℚ := 0.9

-- Define cost expressions for the two options
def option1_cost (x : ℕ) : ℕ :=
  (suit_price * num_suits) + (tie_price * (x - num_suits))

def option2_cost (x : ℕ) : ℚ :=
  ((suit_price * num_suits + tie_price * x) * discount_rate : ℚ)

-- Main theorem to prove the given answers
theorem cost_comparison (x : ℕ) (hx : x > 20) :
  option1_cost x = 40 * x + 3200 ∧
  option2_cost x = 3600 + 36 * x ∧
  (x = 30 → option1_cost 30 < option2_cost 30) :=
by
  sorry

end NUMINAMATH_GPT_cost_comparison_l727_72767


namespace NUMINAMATH_GPT_total_weight_l727_72730

variable (a b c d : ℝ)

-- Conditions
axiom h1 : a + b = 250
axiom h2 : b + c = 235
axiom h3 : c + d = 260
axiom h4 : a + d = 275

-- Proving the total weight
theorem total_weight : a + b + c + d = 510 := by
  sorry

end NUMINAMATH_GPT_total_weight_l727_72730


namespace NUMINAMATH_GPT_sequence_term_4th_l727_72784

theorem sequence_term_4th (a_n : ℕ → ℝ) (h : ∀ n, a_n n = 2 / (n^2 + n)) :
  ∃ n, a_n n = 1 / 10 ∧ n = 4 :=
by
  sorry

end NUMINAMATH_GPT_sequence_term_4th_l727_72784


namespace NUMINAMATH_GPT_coord_relationship_M_l727_72738

theorem coord_relationship_M (x y z : ℝ) (A B : ℝ × ℝ × ℝ)
  (hA : A = (1, 2, -1)) (hB : B = (2, 0, 2))
  (hM : ∃ M : ℝ × ℝ × ℝ, M = (x, y, z) ∧ y = 0 ∧ |(1 - x)^2 + 2^2 + (-1 - z)^2| = |(2 - x)^2 + (0 - z)^2|) :
  x + 3 * z - 1 = 0 ∧ y = 0 := 
sorry

end NUMINAMATH_GPT_coord_relationship_M_l727_72738


namespace NUMINAMATH_GPT_sum_of_midpoint_coordinates_l727_72789

theorem sum_of_midpoint_coordinates (x1 y1 x2 y2 : ℝ) (h1 : x1 = 8) (h2 : y1 = 16) (h3 : x2 = -2) (h4 : y2 = -8) :
  (x1 + x2) / 2 + (y1 + y2) / 2 = 7 := by
  sorry

end NUMINAMATH_GPT_sum_of_midpoint_coordinates_l727_72789


namespace NUMINAMATH_GPT_cookies_sum_l727_72771

theorem cookies_sum (C : ℕ) (h1 : C % 6 = 5) (h2 : C % 9 = 7) (h3 : C < 80) :
  C = 29 :=
by sorry

end NUMINAMATH_GPT_cookies_sum_l727_72771


namespace NUMINAMATH_GPT_Polyas_probability_relation_l727_72759

variable (Z : ℕ → ℤ → ℝ)

theorem Polyas_probability_relation (n : ℕ) (k : ℤ) :
  Z n k = (1/2) * (Z (n-1) (k-1) + Z (n-1) (k+1)) :=
by
  sorry

end NUMINAMATH_GPT_Polyas_probability_relation_l727_72759


namespace NUMINAMATH_GPT_min_neighbor_pairs_l727_72720

theorem min_neighbor_pairs (n : ℕ) (h : n = 2005) :
  ∃ (pairs : ℕ), pairs = 56430 :=
by
  sorry

end NUMINAMATH_GPT_min_neighbor_pairs_l727_72720


namespace NUMINAMATH_GPT_sum_of_consecutive_evens_l727_72792

/-- 
  Prove that the sum of five consecutive even integers 
  starting from 2n, with a common difference of 2, is 10n + 20.
-/
theorem sum_of_consecutive_evens (n : ℕ) :
  (2 * n) + (2 * n + 2) + (2 * n + 4) + (2 * n + 6) + (2 * n + 8) = 10 * n + 20 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_consecutive_evens_l727_72792


namespace NUMINAMATH_GPT_tiling_impossible_2003x2003_l727_72761

theorem tiling_impossible_2003x2003 :
  ¬ (∃ (f : Fin 2003 × Fin 2003 → ℕ),
  (∀ p : Fin 2003 × Fin 2003, f p = 1 ∨ f p = 2) ∧
  (∀ p : Fin 2003, (f (p, 0) + f (p, 1)) % 3 = 0) ∧
  (∀ p : Fin 2003, (f (0, p) + f (1, p) + f (2, p)) % 3 = 0)) := 
sorry

end NUMINAMATH_GPT_tiling_impossible_2003x2003_l727_72761


namespace NUMINAMATH_GPT_true_proposition_l727_72788

noncomputable def prop_p (x : ℝ) : Prop := x > 0 → x^2 - 2*x + 1 > 0

noncomputable def prop_q (x₀ : ℝ) : Prop := x₀ > 0 ∧ x₀^2 - 2*x₀ + 1 ≤ 0

theorem true_proposition : ¬ (∀ x > 0, x^2 - 2*x + 1 > 0) ∧ (∃ x₀ > 0, x₀^2 - 2*x₀ + 1 ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_true_proposition_l727_72788


namespace NUMINAMATH_GPT_differentiable_inequality_l727_72725

theorem differentiable_inequality 
  {a b : ℝ} 
  {f g : ℝ → ℝ} 
  (hdiff_f : DifferentiableOn ℝ f (Set.Icc a b))
  (hdiff_g : DifferentiableOn ℝ g (Set.Icc a b))
  (hderiv_ineq : ∀ x ∈ Set.Ioo a b, (deriv f x > deriv g x)) :
  ∀ x ∈ Set.Ioo a b, f x + g a > g x + f a :=
by 
  sorry

end NUMINAMATH_GPT_differentiable_inequality_l727_72725


namespace NUMINAMATH_GPT_square_ratio_l727_72758

def area (side_length : ℝ) : ℝ := side_length^2

theorem square_ratio (x : ℝ) (x_pos : 0 < x) :
  let A := area x
  let B := area (3*x)
  let C := area (2*x)
  A / (B + C) = 1 / 13 :=
by
  sorry

end NUMINAMATH_GPT_square_ratio_l727_72758


namespace NUMINAMATH_GPT_one_number_greater_than_one_l727_72781

theorem one_number_greater_than_one
  (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_prod : a * b * c = 1)
  (h_sum : a + b + c > 1/a + 1/b + 1/c) :
  ((1 < a ∧ b ≤ 1 ∧ c ≤ 1) ∨ (a ≤ 1 ∧ 1 < b ∧ c ≤ 1) ∨ (a ≤ 1 ∧ b ≤ 1 ∧ 1 < c)) 
  ∧ (¬ ((1 < a ∧ 1 < b) ∨ (1 < b ∧ 1 < c) ∨ (1 < a ∧ 1 < c))) :=
sorry

end NUMINAMATH_GPT_one_number_greater_than_one_l727_72781


namespace NUMINAMATH_GPT_negation_of_existence_is_universal_l727_72754

theorem negation_of_existence_is_universal (p : Prop) :
  (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) ↔ (∀ x : ℝ, x^2 + 2 * x + 2 > 0) :=
sorry

end NUMINAMATH_GPT_negation_of_existence_is_universal_l727_72754


namespace NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l727_72799

variable (x y : ℝ)

theorem sufficient_but_not_necessary (h1: x ≥ 2) (h2: y ≥ 2): x^2 + y^2 ≥ 4 :=
by
  sorry

theorem not_necessary (hx4 : x^2 + y^2 ≥ 4) : ¬ (x ≥ 2 ∧ y ≥ 2) → ∃ x y, (x^2 + y^2 ≥ 4) ∧ (¬ (x ≥ 2) ∨ ¬ (y ≥ 2)) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_but_not_necessary_not_necessary_l727_72799
