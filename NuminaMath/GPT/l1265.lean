import Mathlib

namespace minimum_width_l1265_126561

theorem minimum_width (w : ℝ) (h_area : w * (w + 15) ≥ 200) : w ≥ 10 :=
by
  sorry

end minimum_width_l1265_126561


namespace solve_arithmetic_sequence_l1265_126533

theorem solve_arithmetic_sequence (y : ℝ) (h : 0 < y) (h_arith : ∃ (d : ℝ), 4 + d = y^2 ∧ y^2 + d = 16 ∧ 16 + d = 36) :
  y = Real.sqrt 10 := by
  sorry

end solve_arithmetic_sequence_l1265_126533


namespace solve_for_a_l1265_126586

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then 2^x + 1 else x^2 + a * x

theorem solve_for_a : 
  (∃ a : ℝ, f (f 0 a) a = 4 * a) → (a = 2) :=
by
  sorry

end solve_for_a_l1265_126586


namespace arithmetic_sequence_problem_l1265_126529

variable (a : ℕ → ℕ)
variable (d : ℕ) -- common difference for the arithmetic sequence
variable (h1 : ∀ n : ℕ, a (n + 1) = a n + d)
variable (h2 : a 1 - a 9 + a 17 = 7)

theorem arithmetic_sequence_problem : a 3 + a 15 = 14 := by
  sorry

end arithmetic_sequence_problem_l1265_126529


namespace quadratic_has_distinct_real_roots_l1265_126528

theorem quadratic_has_distinct_real_roots (q : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x^2 + 8 * x + q = 0) ↔ q < 16 :=
by
  -- only the statement is provided, the proof is omitted
  sorry

end quadratic_has_distinct_real_roots_l1265_126528


namespace numPeopleToLeftOfKolya_l1265_126502

-- Definitions based on the conditions.
def peopleToRightOfKolya := 12
def peopleToLeftOfSasha := 20
def peopleToRightOfSasha := 8

-- Theorem statement with the given conditions and conclusion.
theorem numPeopleToLeftOfKolya 
  (h1 : peopleToRightOfKolya = 12)
  (h2 : peopleToLeftOfSasha = 20)
  (h3 : peopleToRightOfSasha = 8) :
  ∃ n, n = 16 :=
by
  -- Proving the theorem will be done here.
  sorry

end numPeopleToLeftOfKolya_l1265_126502


namespace largest_possible_perimeter_l1265_126559

theorem largest_possible_perimeter :
  ∃ (l w : ℕ), 8 * l + 8 * w = l * w - 1 ∧ 2 * l + 2 * w = 164 :=
sorry

end largest_possible_perimeter_l1265_126559


namespace range_of_a_for_monotonically_decreasing_l1265_126524

noncomputable def f (a x: ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - 2 * x

theorem range_of_a_for_monotonically_decreasing (a : ℝ) : 
  (∀ x : ℝ, x > 0 → (1/x - a*x - 2 < 0)) ↔ (a ∈ Set.Ioi (-1)) := 
sorry

end range_of_a_for_monotonically_decreasing_l1265_126524


namespace find_value_of_f_l1265_126507

axiom f : ℝ → ℝ

theorem find_value_of_f :
  (∀ x : ℝ, f (Real.cos x) = Real.sin (3 * x)) →
  f (Real.sin (Real.pi / 9)) = -1 / 2 :=
sorry

end find_value_of_f_l1265_126507


namespace eight_pow_91_gt_seven_pow_92_l1265_126597

theorem eight_pow_91_gt_seven_pow_92 : 8^91 > 7^92 :=
  sorry

end eight_pow_91_gt_seven_pow_92_l1265_126597


namespace workbook_problems_l1265_126519

theorem workbook_problems (P : ℕ)
  (h1 : (1/2 : ℚ) * P = (1/2 : ℚ) * P)
  (h2 : (1/4 : ℚ) * P = (1/4 : ℚ) * P)
  (h3 : (1/6 : ℚ) * P = (1/6 : ℚ) * P)
  (h4 : ((1/2 : ℚ) * P + (1/4 : ℚ) * P + (1/6 : ℚ) * P + 20 = P)) : 
  P = 240 :=
sorry

end workbook_problems_l1265_126519


namespace female_rainbow_trout_l1265_126577

-- Define the conditions given in the problem
variables (F_s M_s M_r F_r T : ℕ)
variables (h1 : F_s + M_s = 645)
variables (h2 : M_s = 2 * F_s + 45)
variables (h3 : 4 * M_r = 3 * F_s)
variables (h4 : 20 * M_r = 3 * T)
variables (h5 : T = 645 + F_r + M_r)

theorem female_rainbow_trout :
  F_r = 205 :=
by
  sorry

end female_rainbow_trout_l1265_126577


namespace weight_of_sparrow_l1265_126585

variable (a b : ℝ)

-- Define the conditions as Lean statements
-- 1. Six sparrows and seven swallows are balanced
def balanced_initial : Prop :=
  6 * b = 7 * a

-- 2. Sparrows are heavier than swallows
def sparrows_heavier : Prop :=
  b > a

-- 3. If one sparrow and one swallow are exchanged, the balance is maintained
def balanced_after_exchange : Prop :=
  5 * b + a = 6 * a + b

-- The theorem to prove the weight of one sparrow in terms of the weight of one swallow
theorem weight_of_sparrow (h1 : balanced_initial a b) (h2 : sparrows_heavier a b) (h3 : balanced_after_exchange a b) : 
  b = (5 / 4) * a :=
sorry

end weight_of_sparrow_l1265_126585


namespace reverse_digits_multiplication_l1265_126578

theorem reverse_digits_multiplication (a b : ℕ) (h₁ : a < 10) (h₂ : b < 10) : 
  (10 * a + b) * (10 * b + a) = 101 * a * b + 10 * (a^2 + b^2) :=
by 
  sorry

end reverse_digits_multiplication_l1265_126578


namespace find_d_l1265_126557

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ k : Nat, k > 1 → k < n → n % k ≠ 0)

def less_than_10_primes (n : Nat) : Prop :=
  n < 10 ∧ is_prime n

theorem find_d (d e f : Nat) (hd : less_than_10_primes d) (he : less_than_10_primes e) (hf : less_than_10_primes f) :
  d + e = f → d < e → d = 2 :=
by
  sorry

end find_d_l1265_126557


namespace problem_statement_l1265_126501

variables (u v w : ℝ)

theorem problem_statement (h₁: u + v + w = 3) : 
  (1 / (u^2 + 7) + 1 / (v^2 + 7) + 1 / (w^2 + 7) ≤ 3 / 8) :=
sorry

end problem_statement_l1265_126501


namespace lens_discount_l1265_126569

theorem lens_discount :
  ∃ (P : ℚ), ∀ (D : ℚ),
    (300 - D = 240) →
    (P = (D / 300) * 100) →
    P = 20 :=
by
  sorry

end lens_discount_l1265_126569


namespace exists_int_less_than_sqrt_twenty_three_l1265_126563

theorem exists_int_less_than_sqrt_twenty_three : ∃ n : ℤ, n < Real.sqrt 23 := 
  sorry

end exists_int_less_than_sqrt_twenty_three_l1265_126563


namespace sequence_n_500_l1265_126518

theorem sequence_n_500 (a : ℕ → ℤ) 
  (h1 : a 1 = 1010) 
  (h2 : a 2 = 1011) 
  (h3 : ∀ n ≥ 1, a n + a (n + 1) + a (n + 2) = 2 * n + 3) : 
  a 500 = 3003 := 
sorry

end sequence_n_500_l1265_126518


namespace angle_B_degrees_l1265_126509

theorem angle_B_degrees (A B C : ℕ) (h1 : A < B) (h2 : B < C) (h3 : 4 * C = 7 * A) (h4 : A + B + C = 180) : B = 59 :=
sorry

end angle_B_degrees_l1265_126509


namespace remainder_correct_l1265_126549

noncomputable def p : Polynomial ℝ := Polynomial.C 3 * Polynomial.X^5 + Polynomial.C 2 * Polynomial.X^3 - Polynomial.C 5 * Polynomial.X + Polynomial.C 8
noncomputable def d : Polynomial ℝ := (Polynomial.X - Polynomial.C 1) ^ 2
noncomputable def r : Polynomial ℝ := Polynomial.C 16 * Polynomial.X - Polynomial.C 8 

theorem remainder_correct : (p % d) = r := by sorry

end remainder_correct_l1265_126549


namespace prove_a_plus_b_l1265_126523

-- Defining the function f(x)
def f (a b x: ℝ) : ℝ := a * x^2 + b * x

-- The given conditions
variable (a b : ℝ)
variable (h1 : f a b (a - 1) = f a b (2 * a))
variable (h2 : ∀ x : ℝ, f a b x = f a b (-x))

-- The objective is to show a + b = 1/3
theorem prove_a_plus_b (a b : ℝ) (h1 : f a b (a - 1) = f a b (2 * a)) (h2 : ∀ x : ℝ, f a b x = f a b (-x)) :
  a + b = 1 / 3 := 
sorry

end prove_a_plus_b_l1265_126523


namespace dot_not_line_l1265_126512

variable (D S DS T : Nat)
variable (h1 : DS = 20) (h2 : S = 36) (h3 : T = 60)
variable (h4 : T = D + S - DS)

theorem dot_not_line : (D - DS) = 24 :=
by
  sorry

end dot_not_line_l1265_126512


namespace sum_of_abs_arithmetic_sequence_l1265_126599

theorem sum_of_abs_arithmetic_sequence {a_n : ℕ → ℤ} {S_n : ℕ → ℤ} 
  (hS3 : S_n 3 = 21) (hS9 : S_n 9 = 9) :
  ∃ (T_n : ℕ → ℤ), 
    (∀ (n : ℕ), n ≤ 5 → T_n n = -n^2 + 10 * n) ∧
    (∀ (n : ℕ), n ≥ 6 → T_n n = n^2 - 10 * n + 50) :=
sorry

end sum_of_abs_arithmetic_sequence_l1265_126599


namespace quadratic_completion_l1265_126506

theorem quadratic_completion (a b : ℤ) (h_eq : (x : ℝ) → x^2 - 10 * x + 25 = 0) :
  (∃ a b : ℤ, ∀ x : ℝ, (x + a) ^ 2 = b) → a + b = -5 := by
  sorry

end quadratic_completion_l1265_126506


namespace not_solvable_equations_l1265_126504

theorem not_solvable_equations :
  ¬(∃ x : ℝ, (x - 5) ^ 2 = -1) ∧ ¬(∃ x : ℝ, |2 * x| + 3 = 0) :=
by
  sorry

end not_solvable_equations_l1265_126504


namespace maintain_income_with_new_demand_l1265_126508

variable (P D : ℝ) -- Original Price and Demand
def new_price := 1.20 * P -- New Price after 20% increase
def new_demand := 1.12 * D -- New Demand after 12% increase due to advertisement
def original_income := P * D -- Original income
def new_income := new_price * new_demand -- New income after changes

theorem maintain_income_with_new_demand :
  ∀ P D : ℝ, P * D = 1.20 * P * 1.12 * (D_new : ℝ) → (D_new = 14/15 * D) :=
by
  intro P D h
  sorry

end maintain_income_with_new_demand_l1265_126508


namespace least_number_div_condition_l1265_126592

theorem least_number_div_condition (m : ℕ) : 
  (∃ k r : ℕ, m = 34 * k + r ∧ m = 5 * (r + 8) ∧ r < 34) → m = 162 := 
by
  sorry

end least_number_div_condition_l1265_126592


namespace smallest_n_l1265_126540

theorem smallest_n (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 2012) :
  ∃ m n : ℕ, a.factorial * b.factorial * c.factorial = m * 10 ^ n ∧ ¬ (10 ∣ m) ∧ n = 501 :=
by
  sorry

end smallest_n_l1265_126540


namespace prove_equation_C_l1265_126514

theorem prove_equation_C (m : ℝ) : -(m - 2) = -m + 2 := 
  sorry

end prove_equation_C_l1265_126514


namespace T_description_l1265_126584

def is_single_point {x y : ℝ} : Prop := (x = 2) ∧ (y = 11)

theorem T_description :
  ∀ (T : Set (ℝ × ℝ)),
  (∀ x y : ℝ, 
    (T (x, y) ↔ 
    ((5 = x + 3 ∧ 5 = y - 6) ∨ 
     (5 = x + 3 ∧ x + 3 = y - 6) ∨ 
     (5 = y - 6 ∧ x + 3 = y - 6)) ∧ 
    ((x = 2) ∧ (y = 11))
    )
  ) →
  (T = { (2, 11) }) :=
by
  sorry

end T_description_l1265_126584


namespace relationship_of_y_values_l1265_126558

def parabola_y (x : ℝ) (c : ℝ) : ℝ :=
  2 * (x + 1)^2 + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  y1 = parabola_y (-2) c →
  y2 = parabola_y 1 c →
  y3 = parabola_y 2 c →
  y3 > y2 ∧ y2 > y1 :=
by
  intros h1 h2 h3
  sorry

end relationship_of_y_values_l1265_126558


namespace john_safety_percentage_l1265_126576

def bench_max_weight : ℕ := 1000
def john_weight : ℕ := 250
def weight_on_bar : ℕ := 550
def total_weight := john_weight + weight_on_bar
def percentage_of_max_weight := (total_weight * 100) / bench_max_weight
def percentage_under_max_weight := 100 - percentage_of_max_weight

theorem john_safety_percentage : percentage_under_max_weight = 20 := by
  sorry

end john_safety_percentage_l1265_126576


namespace initial_student_count_l1265_126536

theorem initial_student_count
  (n : ℕ)
  (T : ℝ)
  (h1 : T = 60.5 * (n : ℝ))
  (h2 : T - 8 = 64 * ((n - 1) : ℝ))
  : n = 16 :=
sorry

end initial_student_count_l1265_126536


namespace odd_function_evaluation_l1265_126572

theorem odd_function_evaluation (f : ℝ → ℝ) (hf : ∀ x, f (-x) = -f x) (h : f (-3) = -2) : f 3 + f 0 = 2 :=
by 
  sorry

end odd_function_evaluation_l1265_126572


namespace equation_is_hyperbola_l1265_126513

-- Define the equation
def equation (x y : ℝ) : Prop :=
  4 * x^2 - 9 * y^2 + 3 * x = 0

-- Theorem stating that the given equation represents a hyperbola
theorem equation_is_hyperbola : ∀ x y : ℝ, equation x y → (∃ A B : ℝ, A * x^2 - B * y^2 = 1) :=
by
  sorry

end equation_is_hyperbola_l1265_126513


namespace greatest_b_value_l1265_126579

def equation_has_integer_solutions (b : ℕ) : Prop :=
  ∃ (x : ℤ), x * (x + b) = -20

theorem greatest_b_value : ∃ (b : ℕ), b = 21 ∧ equation_has_integer_solutions b :=
by
  sorry

end greatest_b_value_l1265_126579


namespace work_hours_l1265_126594

-- Let h be the number of hours worked
def hours_worked (total_paid part_cost hourly_rate : ℕ) : ℕ :=
  (total_paid - part_cost) / hourly_rate

-- Given conditions
def total_paid : ℕ := 300
def part_cost : ℕ := 150
def hourly_rate : ℕ := 75

-- The statement to be proved
theorem work_hours :
  hours_worked total_paid part_cost hourly_rate = 2 :=
by
  -- Provide the proof here
  sorry

end work_hours_l1265_126594


namespace mouse_lives_difference_l1265_126530

-- Definitions of variables and conditions
def cat_lives : ℕ := 9
def dog_lives : ℕ := cat_lives - 3
def mouse_lives : ℕ := 13

-- Theorem to prove
theorem mouse_lives_difference : mouse_lives - dog_lives = 7 := by
  -- This is where the proof would go, but we use sorry to skip it.
  sorry

end mouse_lives_difference_l1265_126530


namespace solution_set_of_inequalities_l1265_126541

theorem solution_set_of_inequalities :
  (∅ ≠ {x : ℝ | x - 2 > 1 ∧ x < 4} ∧ (∀ x, (x - 2 > 1 ∧ x < 4) ↔ (3 < x ∧ x < 4))) :=
by
  sorry

end solution_set_of_inequalities_l1265_126541


namespace circumscribed_circle_area_l1265_126537

noncomputable def circumradius (s : ℝ) : ℝ := s / Real.sqrt 3
noncomputable def circle_area (r : ℝ) : ℝ := Real.pi * r^2

theorem circumscribed_circle_area (s : ℝ) (hs : s = 15) : circle_area (circumradius s) = 75 * Real.pi :=
by
  sorry

end circumscribed_circle_area_l1265_126537


namespace additional_number_is_31_l1265_126510

theorem additional_number_is_31
(six_numbers_sum : ℕ)
(seven_numbers_avg : ℕ)
(h1 : six_numbers_sum = 144)
(h2 : seven_numbers_avg = 25)
: ∃ x : ℕ, ((six_numbers_sum + x) / 7 = 25) ∧ x = 31 := 
by
  sorry

end additional_number_is_31_l1265_126510


namespace proof_C_ST_l1265_126589

-- Definitions for sets and their operations
def A1 : Set ℕ := {0, 1}
def A2 : Set ℕ := {1, 2}
def S : Set ℕ := A1 ∪ A2
def T : Set ℕ := A1 ∩ A2
def C_ST : Set ℕ := S \ T

theorem proof_C_ST : 
  C_ST = {0, 2} := 
by 
  sorry

end proof_C_ST_l1265_126589


namespace problem_part1_problem_part2_l1265_126535

noncomputable def f (m x : ℝ) := Real.log (m * x) - x + 1
noncomputable def g (m x : ℝ) := (x - 1) * Real.exp x - m * x

theorem problem_part1 (m : ℝ) (h : m > 0) (hf : ∀ x, f m x ≤ 0) : m = 1 :=
sorry

theorem problem_part2 (m : ℝ) (h : m > 0) :
  ∃ x₀, (∀ x, g m x ≤ g m x₀) ∧ (1 / 2 * Real.log (m + 1) < x₀ ∧ x₀ < m) :=
sorry

end problem_part1_problem_part2_l1265_126535


namespace cost_of_student_ticket_l1265_126582

theorem cost_of_student_ticket
  (cost_adult : ℤ)
  (total_tickets : ℤ)
  (total_revenue : ℤ)
  (adult_tickets : ℤ)
  (student_tickets : ℤ)
  (H1 : cost_adult = 6)
  (H2 : total_tickets = 846)
  (H3 : total_revenue = 3846)
  (H4 : adult_tickets = 410)
  (H5 : student_tickets = 436)
  : (total_revenue = adult_tickets * cost_adult + student_tickets * (318 / 100)) :=
by
  -- mathematical proof steps would go here
  sorry

end cost_of_student_ticket_l1265_126582


namespace geom_seq_min_value_l1265_126588

noncomputable def minimum_sum (m n : ℕ) (a : ℕ → ℝ) : ℝ :=
  if (a 7 = a 6 + 2 * a 5) ∧ (a m * a n = 16 * (a 1) ^ 2) ∧ (m > 0) ∧ (n > 0) then
    (1 / m) + (4 / n)
  else
    0

theorem geom_seq_min_value (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = q * a n) →
  a 7 = a 6 + 2 * a 5 →
  (∃ m n, a m * a n = 16 * (a 1) ^ 2 ∧ m > 0 ∧ n > 0) →
  (minimum_sum m n a = 3 / 2) := sorry

end geom_seq_min_value_l1265_126588


namespace puppy_sleep_duration_l1265_126534

-- Definitions based on conditions
def connor_sleep : ℕ := 6
def luke_sleep : ℕ := connor_sleep + 2
def puppy_sleep : ℕ := 2 * luke_sleep

-- Theorem stating that the puppy sleeps for 16 hours
theorem puppy_sleep_duration : puppy_sleep = 16 := by
  sorry

end puppy_sleep_duration_l1265_126534


namespace min_value_of_m_cauchy_schwarz_inequality_l1265_126521

theorem min_value_of_m (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : m = a + 1 / ((a - b) * b)) : 
  ∃ t, t = 3 ∧ ∀ a b : ℝ, a > b → b > 0 → m = a + 1 / ((a - b) * b) → m ≥ t :=
sorry

theorem cauchy_schwarz_inequality (x y z : ℝ) :
  (x^2 + 4 * y^2 + z^2 = 3) → |x + 2 * y + z| ≤ 3 :=
sorry

end min_value_of_m_cauchy_schwarz_inequality_l1265_126521


namespace problem_solution_l1265_126517

theorem problem_solution (x : ℝ) (N : ℝ) (h1 : 625 ^ (-x) + N ^ (-2 * x) + 5 ^ (-4 * x) = 11) (h2 : x = 0.25) :
  N = 25 / 2809 :=
by
  sorry

end problem_solution_l1265_126517


namespace parallel_lines_l1265_126565

theorem parallel_lines (m : ℝ) 
  (h : 3 * (m - 2) + m * (m + 2) = 0) 
  : m = 1 ∨ m = -6 := 
by 
  sorry

end parallel_lines_l1265_126565


namespace generalized_inequality_l1265_126593

theorem generalized_inequality (x : ℝ) (n : ℕ) (h1 : x > 0) : x^n + (n : ℝ) / x > n + 1 := 
sorry

end generalized_inequality_l1265_126593


namespace seashell_count_l1265_126573

def initialSeashells : Nat := 5
def givenSeashells : Nat := 2
def remainingSeashells : Nat := initialSeashells - givenSeashells

theorem seashell_count : remainingSeashells = 3 := by
  sorry

end seashell_count_l1265_126573


namespace find_p_l1265_126503

-- Define the conditions for the problem.
-- Random variable \xi follows binomial distribution B(n, p).
axiom binomial_distribution (n : ℕ) (p : ℝ) : Type
variables (ξ : binomial_distribution n p)

-- Given conditions: Eξ = 300 and Dξ = 200.
axiom Eξ (ξ : binomial_distribution n p) : ℝ
axiom Dξ (ξ : binomial_distribution n p) : ℝ

-- Given realizations of expectations and variance.
axiom h1 : Eξ ξ = 300
axiom h2 : Dξ ξ = 200

-- Prove that p = 1/3
theorem find_p (n : ℕ) (p : ℝ) (ξ : binomial_distribution n p)
  (h1 : Eξ ξ = 300) (h2 : Dξ ξ = 200) : p = 1 / 3 :=
sorry

end find_p_l1265_126503


namespace y_intercept_of_line_l1265_126543

theorem y_intercept_of_line (y : ℝ) (h : 3 * 0 - 4 * y = 12) : y = -3 := 
by sorry

end y_intercept_of_line_l1265_126543


namespace perpendicular_lines_slope_eq_l1265_126591

theorem perpendicular_lines_slope_eq (m : ℝ) :
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → 
               2 * x + m * y - 6 = 0 → 
               (1 / 2) * (-2 / m) = -1) →
  m = 1 := 
by sorry

end perpendicular_lines_slope_eq_l1265_126591


namespace profit_percentage_l1265_126574

theorem profit_percentage (cost_price selling_price : ℝ) (h_cost : cost_price = 60) (h_selling : selling_price = 75) : 
  ((selling_price - cost_price) / cost_price) * 100 = 25 := by
  sorry

end profit_percentage_l1265_126574


namespace min_value_of_x_plus_y_l1265_126587

theorem min_value_of_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y = x * y) : x + y ≥ 9 :=
by
  sorry

end min_value_of_x_plus_y_l1265_126587


namespace class_average_weight_l1265_126595

theorem class_average_weight :
  (24 * 40 + 16 * 35 + 18 * 42 + 22 * 38) / (24 + 16 + 18 + 22) = 38.9 :=
by
  -- skipped proof
  sorry

end class_average_weight_l1265_126595


namespace toy_factory_max_profit_l1265_126544

theorem toy_factory_max_profit :
  ∃ x y : ℕ,    -- x: number of bears, y: number of cats
  15 * x + 10 * y ≤ 450 ∧    -- labor hours constraint
  20 * x + 5 * y ≤ 400 ∧     -- raw materials constraint
  80 * x + 45 * y = 2200 :=  -- total selling price
by
  sorry

end toy_factory_max_profit_l1265_126544


namespace roots_in_interval_l1265_126567

def P (x : ℝ) : ℝ := x^2014 - 100 * x + 1

theorem roots_in_interval : 
  ∀ x : ℝ, P x = 0 → (1/100) ≤ x ∧ x ≤ 100^(1 / 2013) := 
  sorry

end roots_in_interval_l1265_126567


namespace perimeter_of_square_l1265_126555

theorem perimeter_of_square (s : ℝ) (h : s^2 = s * Real.sqrt 2) (h_ne_zero : s ≠ 0) :
    4 * s = 4 * Real.sqrt 2 := by
  sorry

end perimeter_of_square_l1265_126555


namespace cost_per_trip_l1265_126505

theorem cost_per_trip
  (pass_cost : ℝ)
  (oldest_trips : ℕ)
  (youngest_trips : ℕ)
  (h_pass_cost : pass_cost = 100.0)
  (h_oldest_trips : oldest_trips = 35)
  (h_youngest_trips : youngest_trips = 15) :
  (2 * pass_cost) / (oldest_trips + youngest_trips) = 4.0 :=
by
  sorry

end cost_per_trip_l1265_126505


namespace a_2_pow_100_value_l1265_126570

theorem a_2_pow_100_value
  (a : ℕ → ℕ)
  (h1 : a 1 = 2)
  (h2 : ∀ n : ℕ, a (2 * n) = 3 * n * a n) :
  a (2^100) = 2^4852 * 3^4950 :=
by
  sorry

end a_2_pow_100_value_l1265_126570


namespace no_int_solutions_for_cubic_eqn_l1265_126580

theorem no_int_solutions_for_cubic_eqn :
  ¬ ∃ (m n : ℤ), m^3 = 3 * n^2 + 3 * n + 7 := by
  sorry

end no_int_solutions_for_cubic_eqn_l1265_126580


namespace production_rate_l1265_126564

theorem production_rate (minutes: ℕ) (machines1 machines2 paperclips1 paperclips2 : ℕ)
  (h1 : minutes = 1) (h2 : machines1 = 8) (h3 : machines2 = 18) (h4 : paperclips1 = 560) 
  (h5 : paperclips2 = (paperclips1 / machines1) * machines2 * minutes) : 
  paperclips2 = 7560 :=
by
  sorry

end production_rate_l1265_126564


namespace smaller_number_l1265_126548

theorem smaller_number (x y : ℕ) (h1 : x + y = 40) (h2 : x - y = 10) : y = 15 :=
sorry

end smaller_number_l1265_126548


namespace power_function_convex_upwards_l1265_126596

noncomputable def f (x : ℝ) : ℝ :=
  x ^ (4 / 5)

theorem power_function_convex_upwards (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : x1 < x2) :
  f ((x1 + x2) / 2) > (f x1 + f x2) / 2 :=
sorry

end power_function_convex_upwards_l1265_126596


namespace Person3IsTriussian_l1265_126562

def IsTriussian (person : ℕ) : Prop := if person = 3 then True else False

def Person1Statement : Prop := ∀ i j k : ℕ, i = 1 → j = 2 → k = 3 → (IsTriussian i = (IsTriussian j ∧ IsTriussian k) ∨ (¬IsTriussian j ∧ ¬IsTriussian k))

def Person2Statement : Prop := ∀ i j : ℕ, i = 2 → j = 3 → (IsTriussian j = False)

def Person3Statement : Prop := ∀ i j : ℕ, i = 3 → j = 1 → (IsTriussian j = False)

theorem Person3IsTriussian : (Person1Statement ∧ Person2Statement ∧ Person3Statement) → IsTriussian 3 :=
by 
  sorry

end Person3IsTriussian_l1265_126562


namespace trig_function_value_l1265_126566

noncomputable def f : ℝ → ℝ := sorry

theorem trig_function_value:
  (∀ x, f (Real.cos x) = Real.cos (3 * x)) →
  f (Real.sin (Real.pi / 6)) = -1 :=
by
  intro h
  sorry

end trig_function_value_l1265_126566


namespace track_length_l1265_126571

variable {x : ℕ}

-- Conditions
def runs_distance_jacob (x : ℕ) := 120
def runs_distance_liz (x : ℕ) := (x / 2 - 120)

def runs_second_meeting_jacob (x : ℕ) := x + 120 -- Jacob's total distance by second meeting
def runs_second_meeting_liz (x : ℕ) := (x / 2 + 60) -- Liz's total distance by second meeting

-- The relationship is simplified into the final correct answer
theorem track_length (h1 : 120 / (x / 2 - 120) = (x / 2 + 60) / 180) :
  x = 340 := 
sorry

end track_length_l1265_126571


namespace ninth_observation_l1265_126590

theorem ninth_observation (avg1 : ℝ) (avg2 : ℝ) (n1 n2 : ℝ) 
  (sum1 : n1 * avg1 = 120) 
  (sum2 : n2 * avg2 = 117) 
  (avg_decrease : avg1 - avg2 = 2) 
  (obs_count_change : n1 + 1 = n2) 
  : n2 * avg2 - n1 * avg1 = -3 :=
by
  sorry

end ninth_observation_l1265_126590


namespace cone_volume_ratio_l1265_126516

theorem cone_volume_ratio (rC hC rD hD : ℝ) (h_rC : rC = 10) (h_hC : hC = 20) (h_rD : rD = 20) (h_hD : hD = 10) :
  ((1/3) * π * rC^2 * hC) / ((1/3) * π * rD^2 * hD) = 1/2 :=
by 
  sorry

end cone_volume_ratio_l1265_126516


namespace find_missing_number_l1265_126539

-- Define the known values
def numbers : List ℕ := [1, 22, 24, 25, 26, 27, 2]
def specified_mean : ℕ := 20
def total_counts : ℕ := 8

-- The theorem statement
theorem find_missing_number : (∀ (x : ℕ), (List.sum (x :: numbers) = specified_mean * total_counts) → x = 33) :=
by
  sorry

end find_missing_number_l1265_126539


namespace total_seeds_eaten_l1265_126551

-- Definitions and conditions
def first_player_seeds : ℕ := 78
def second_player_seeds : ℕ := 53
def third_player_seeds : ℕ := second_player_seeds + 30
def fourth_player_seeds : ℕ := 2 * third_player_seeds
def first_four_players_seeds : ℕ := first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds
def average_seeds : ℕ := first_four_players_seeds / 4
def fifth_player_seeds : ℕ := average_seeds

-- Statement to prove
theorem total_seeds_eaten :
  first_player_seeds + second_player_seeds + third_player_seeds + fourth_player_seeds + fifth_player_seeds = 475 :=
by {
  sorry
}

end total_seeds_eaten_l1265_126551


namespace smallest_n_value_existence_l1265_126515

-- Define a three-digit positive integer n such that the conditions hold
def is_three_digit (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def problem_conditions (n : ℕ) : Prop :=
  n % 9 = 3 ∧ n % 6 = 3

-- Main statement: There exists a three-digit positive integer n satisfying the conditions and is equal to 111
theorem smallest_n_value_existence : ∃ n : ℕ, is_three_digit n ∧ problem_conditions n ∧ n = 111 :=
by
  sorry

end smallest_n_value_existence_l1265_126515


namespace toothpicks_in_300th_stage_l1265_126546

/-- 
Prove that the number of toothpicks needed for the 300th stage is 1201, given:
1. The first stage has 5 toothpicks.
2. Each subsequent stage adds 4 toothpicks to the previous stage.
-/
theorem toothpicks_in_300th_stage :
  let a_1 := 5
  let d := 4
  let a_n (n : ℕ) := a_1 + (n - 1) * d
  a_n 300 = 1201 := by
  sorry

end toothpicks_in_300th_stage_l1265_126546


namespace product_of_real_roots_l1265_126560

theorem product_of_real_roots (x1 x2 : ℝ) (h1 : x1^2 - 6 * x1 + 8 = 0) (h2 : x2^2 - 6 * x2 + 8 = 0) :
  x1 * x2 = 8 := 
sorry

end product_of_real_roots_l1265_126560


namespace total_receipts_l1265_126520

theorem total_receipts 
  (x y : ℕ) 
  (h1 : x + y = 64)
  (h2 : y ≥ 8) 
  : 3 * x + 4 * y = 200 := 
by
  sorry

end total_receipts_l1265_126520


namespace factorize_expression_l1265_126531

theorem factorize_expression (a b : ℤ) (h1 : 3 * b + a = -1) (h2 : a * b = -18) : a - b = -11 :=
by
  sorry

end factorize_expression_l1265_126531


namespace people_left_after_first_stop_l1265_126538

def initial_people_on_train : ℕ := 48
def people_got_off_train : ℕ := 17

theorem people_left_after_first_stop : (initial_people_on_train - people_got_off_train) = 31 := by
  sorry

end people_left_after_first_stop_l1265_126538


namespace gcd_factorial_8_10_l1265_126511

theorem gcd_factorial_8_10 : Nat.gcd (Nat.factorial 8) (Nat.factorial 10) = Nat.factorial 8 := by
  sorry

end gcd_factorial_8_10_l1265_126511


namespace chinese_carriage_problem_l1265_126556

theorem chinese_carriage_problem (x : ℕ) : 
  (3 * (x - 2) = 2 * x + 9) :=
sorry

end chinese_carriage_problem_l1265_126556


namespace intersection_is_correct_l1265_126553

-- Defining sets A and B
def setA : Set ℝ := {x : ℝ | x^2 - 2 * x ≤ 0}
def setB : Set ℝ := {x : ℝ | -1 < x ∧ x < 1}

-- Target intersection set
def setIntersection : Set ℝ := {x : ℝ | 0 ≤ x ∧ x < 1}

-- Theorem to be proved
theorem intersection_is_correct : (setA ∩ setB) = setIntersection :=
by
  -- Proof steps will go here
  sorry

end intersection_is_correct_l1265_126553


namespace problem_l1265_126522

theorem problem (m n r t : ℚ) 
  (h1 : m / n = 5 / 2) 
  (h2 : r / t = 7 / 5) 
: (5 * m * r - 2 * n * t) / (7 * n * t - 10 * m * r) = -31 / 56 := 
sorry

end problem_l1265_126522


namespace find_a_minus_b_l1265_126550

theorem find_a_minus_b
  (a b : ℝ)
  (f g h h_inv : ℝ → ℝ)
  (hf : ∀ x, f x = a * x + b)
  (hg : ∀ x, g x = -4 * x + 3)
  (hh : ∀ x, h x = f (g x))
  (hinv : ∀ x, h_inv x = 2 * x + 6)
  (h_comp : ∀ x, h x = (x - 6) / 2) :
  a - b = 5 / 2 :=
sorry

end find_a_minus_b_l1265_126550


namespace three_consecutive_odd_numbers_l1265_126575

theorem three_consecutive_odd_numbers (x : ℤ) (h : x - 2 + x + x + 2 = 27) : 
  (x + 2, x, x - 2) = (11, 9, 7) :=
by
  sorry

end three_consecutive_odd_numbers_l1265_126575


namespace ram_salary_percentage_more_l1265_126545

theorem ram_salary_percentage_more (R r : ℝ) (h : r = 0.8 * R) :
  ((R - r) / r) * 100 = 25 := 
sorry

end ram_salary_percentage_more_l1265_126545


namespace balls_sold_eq_13_l1265_126568

-- Let SP be the selling price, CP be the cost price per ball, and loss be the loss incurred.
def SP : ℕ := 720
def CP : ℕ := 90
def loss : ℕ := 5 * CP
def total_CP (n : ℕ) : ℕ := n * CP

-- Given the conditions:
axiom loss_eq : loss = 5 * CP
axiom ball_CP_value : CP = 90
axiom selling_price_value : SP = 720

-- Loss is defined as total cost price minus selling price
def calculated_loss (n : ℕ) : ℕ := total_CP n - SP

-- The proof statement:
theorem balls_sold_eq_13 (n : ℕ) (h1 : calculated_loss n = loss) : n = 13 :=
by sorry

end balls_sold_eq_13_l1265_126568


namespace find_z_plus_1_over_y_l1265_126554

theorem find_z_plus_1_over_y (x y z : ℝ) (h1 : x * y * z = 1) (h2 : x + 1 / z = 7) (h3 : y + 1 / x = 20) : 
  z + 1 / y = 29 / 139 := 
by 
  sorry

end find_z_plus_1_over_y_l1265_126554


namespace dana_total_earnings_l1265_126526

-- Define the constants for Dana's hourly rate and hours worked each day
def hourly_rate : ℝ := 13
def friday_hours : ℝ := 9
def saturday_hours : ℝ := 10
def sunday_hours : ℝ := 3

-- Define the total earnings calculation function
def total_earnings (rate : ℝ) (hours1 hours2 hours3 : ℝ) : ℝ :=
  rate * hours1 + rate * hours2 + rate * hours3

-- The main statement
theorem dana_total_earnings : total_earnings hourly_rate friday_hours saturday_hours sunday_hours = 286 := by
  sorry

end dana_total_earnings_l1265_126526


namespace company_picnic_attendance_l1265_126527

theorem company_picnic_attendance :
  ∀ (employees men women men_attending women_attending : ℕ)
  (h_employees : employees = 100)
  (h_men : men = 55)
  (h_women : women = 45)
  (h_men_attending: men_attending = 11)
  (h_women_attending: women_attending = 18),
  (100 * (men_attending + women_attending) / employees) = 29 := 
by
  intros employees men women men_attending women_attending 
         h_employees h_men h_women h_men_attending h_women_attending
  sorry

end company_picnic_attendance_l1265_126527


namespace remaining_movie_duration_l1265_126500

/--
Given:
1. The laptop was fully charged at 3:20 pm.
2. Hannah started watching a 3-hour series.
3. The laptop turned off at 5:44 pm (fully discharged).

Prove:
The remaining duration of the movie Hannah needs to watch is 36 minutes.
-/
theorem remaining_movie_duration
    (start_full_charge : ℕ := 200)  -- representing 3:20 pm as 200 (20 minutes past 3:00)
    (end_discharge : ℕ := 344)  -- representing 5:44 pm as 344 (44 minutes past 5:00)
    (total_duration_minutes : ℕ := 180)  -- 3 hours in minutes
    (start_time_minutes : ℕ := 200)  -- convert 3:20 pm to minutes past noon
    (end_time_minutes : ℕ := 344)  -- convert 5:44 pm to minutes past noon
    : (total_duration_minutes - (end_time_minutes - start_time_minutes)) = 36 :=
by
  sorry

end remaining_movie_duration_l1265_126500


namespace parallel_lines_slope_l1265_126583

theorem parallel_lines_slope (a : ℝ) :
  (∃ b : ℝ, ( ∀ x y : ℝ, a*x - 5*y - 9 = 0 → b*x - 3*y - 10 = 0) → a = 10/3) :=
sorry

end parallel_lines_slope_l1265_126583


namespace car_total_travel_time_l1265_126542

def T_NZ : ℕ := 60

def T_NR : ℕ := 8 / 10 * T_NZ -- 80% of T_NZ

def T_ZV : ℕ := 3 / 4 * T_NR -- 75% of T_NR

theorem car_total_travel_time :
  T_NZ + T_NR + T_ZV = 144 := by
  sorry

end car_total_travel_time_l1265_126542


namespace asian_games_discount_equation_l1265_126532

variable (a : ℝ)

theorem asian_games_discount_equation :
  168 * (1 - a / 100)^2 = 128 :=
sorry

end asian_games_discount_equation_l1265_126532


namespace repeatable_transformation_l1265_126581

theorem repeatable_transformation (a b c : ℝ) (h₁ : a + b > c) (h₂ : b + c > a) (h₃ : c + a > b) :
  (2 * c > a + b) ∧ (2 * a > b + c) ∧ (2 * b > c + a) := 
sorry

end repeatable_transformation_l1265_126581


namespace puppies_in_each_cage_l1265_126552

theorem puppies_in_each_cage (initial_puppies sold_puppies cages : ℕ)
  (h_initial : initial_puppies = 18)
  (h_sold : sold_puppies = 3)
  (h_cages : cages = 3) :
  (initial_puppies - sold_puppies) / cages = 5 :=
by
  sorry

end puppies_in_each_cage_l1265_126552


namespace correct_exponentiation_l1265_126598

theorem correct_exponentiation (x : ℝ) : x^2 * x^3 = x^5 :=
by sorry

end correct_exponentiation_l1265_126598


namespace half_sum_of_squares_l1265_126525

theorem half_sum_of_squares (n m : ℕ) (h : n ≠ m) :
  ∃ a b : ℕ, ( (2 * n)^2 + (2 * m)^2) / 2 = a^2 + b^2 := by
  sorry

end half_sum_of_squares_l1265_126525


namespace missing_water_calculation_l1265_126547

def max_capacity : ℝ := 350000
def loss_rate1 : ℝ := 32000
def time1 : ℝ := 5
def loss_rate2 : ℝ := 10000
def time2 : ℝ := 10
def fill_rate : ℝ := 40000
def fill_time : ℝ := 3

theorem missing_water_calculation :
  350000 - ((350000 - (32000 * 5 + 10000 * 10)) + 40000 * 3) = 140000 :=
by
  sorry

end missing_water_calculation_l1265_126547
