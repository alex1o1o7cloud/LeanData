import Mathlib

namespace NUMINAMATH_GPT_general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l2235_223569

def seq_a : ℕ → ℕ 
| 0 => 0  -- a_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2^(n+1)

def seq_b : ℕ → ℕ 
| 0 => 0  -- b_0 is not defined in natural numbers, put it as zero for base case
| (n+1) => 2*(n+1) -1

def sum_S (n : ℕ) : ℕ := (seq_a (n+1) * 2) - 2

def sum_T : ℕ → ℕ 
| 0 => 0  -- T_0 is not defined in natural numbers, put it as zero for base case too
| (n+1) => (n+1)^2

def sum_D : ℕ → ℕ
| 0 => 0
| (n+1) => (seq_a (n+1) * seq_b (n+1)) + sum_D n

theorem general_term_a_n (n : ℕ) : seq_a n = 2^n := sorry

theorem general_term_b_n (n : ℕ) : seq_b n = 2*n - 1 := sorry

theorem sum_of_first_n_terms_D_n (n : ℕ) : sum_D n = (2*n - 3)*2^(n+1) + 6 := sorry

end NUMINAMATH_GPT_general_term_a_n_general_term_b_n_sum_of_first_n_terms_D_n_l2235_223569


namespace NUMINAMATH_GPT_square_problem_solution_l2235_223560

theorem square_problem_solution
  (x : ℝ)
  (h1 : ∃ s1 : ℝ, s1^2 = x^2 + 12*x + 36)
  (h2 : ∃ s2 : ℝ, s2^2 = 4*x^2 - 12*x + 9)
  (h3 : 4 * (s1 + s2) = 64) :
  x = 13 / 3 :=
by
  sorry

end NUMINAMATH_GPT_square_problem_solution_l2235_223560


namespace NUMINAMATH_GPT_last_matching_date_2008_l2235_223564

noncomputable def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

/-- The last date in 2008 when the sum of the first four digits equals the sum of the last four digits is 25 December 2008. -/
theorem last_matching_date_2008 :
  ∃ d m y, d = 25 ∧ m = 12 ∧ y = 2008 ∧
            sum_of_digits 2512 = sum_of_digits 2008 :=
by {
  sorry
}

end NUMINAMATH_GPT_last_matching_date_2008_l2235_223564


namespace NUMINAMATH_GPT_pow_mod_remainder_l2235_223529

theorem pow_mod_remainder (n : ℕ) (h : 9 ≡ 2 [MOD 7]) (h2 : 9^2 ≡ 4 [MOD 7]) (h3 : 9^3 ≡ 1 [MOD 7]) : 9^123 % 7 = 1 := by
  sorry

end NUMINAMATH_GPT_pow_mod_remainder_l2235_223529


namespace NUMINAMATH_GPT_investment_double_l2235_223582

theorem investment_double (A : ℝ) (r t : ℝ) (hA : 0 < A) (hr : 0 < r) :
  2 * A ≤ A * (1 + r)^t ↔ t ≥ (Real.log 2) / (Real.log (1 + r)) := 
by
  sorry

end NUMINAMATH_GPT_investment_double_l2235_223582


namespace NUMINAMATH_GPT_arithmetic_seq_value_zero_l2235_223531

theorem arithmetic_seq_value_zero (a b c : ℝ) (a_seq : ℕ → ℝ)
    (l m n : ℕ) (h_arith : ∀ k, a_seq (k + 1) - a_seq k = a_seq 1 - a_seq 0)
    (h_l : a_seq l = 1 / a)
    (h_m : a_seq m = 1 / b)
    (h_n : a_seq n = 1 / c) :
    (l - m) * a * b + (m - n) * b * c + (n - l) * c * a = 0 := 
sorry

end NUMINAMATH_GPT_arithmetic_seq_value_zero_l2235_223531


namespace NUMINAMATH_GPT_value_of_polynomial_at_2_l2235_223576

def f (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3 * x^3 - 2 * x^2 - 2500 * x + 434

theorem value_of_polynomial_at_2 : f 2 = -3390 := by
  -- proof would go here
  sorry

end NUMINAMATH_GPT_value_of_polynomial_at_2_l2235_223576


namespace NUMINAMATH_GPT_find_m_l2235_223500

theorem find_m (x y m : ℝ) (h1 : x = 1) (h2 : y = 3) (h3 : m * x - y = 3) : m = 6 := 
by
  sorry

end NUMINAMATH_GPT_find_m_l2235_223500


namespace NUMINAMATH_GPT_largest_circle_radius_l2235_223547

theorem largest_circle_radius (a b c : ℝ) (h : a > b ∧ b > c) :
  ∃ radius : ℝ, radius = b :=
by
  sorry

end NUMINAMATH_GPT_largest_circle_radius_l2235_223547


namespace NUMINAMATH_GPT_ratio_distance_traveled_by_foot_l2235_223578

theorem ratio_distance_traveled_by_foot (D F B C : ℕ) (hD : D = 40) 
(hB : B = D / 2) (hC : C = 10) (hF : F = D - (B + C)) : F / D = 1 / 4 := 
by sorry

end NUMINAMATH_GPT_ratio_distance_traveled_by_foot_l2235_223578


namespace NUMINAMATH_GPT_peaches_left_l2235_223523

/-- Brenda picks 3600 peaches, 37.5% are fresh, and 250 are disposed of. Prove that Brenda has 1100 peaches left. -/
theorem peaches_left (total_peaches : ℕ) (percent_fresh : ℚ) (peaches_disposed : ℕ) (h1 : total_peaches = 3600) (h2 : percent_fresh = 3 / 8) (h3 : peaches_disposed = 250) : 
  total_peaches * percent_fresh - peaches_disposed = 1100 := 
by
  sorry

end NUMINAMATH_GPT_peaches_left_l2235_223523


namespace NUMINAMATH_GPT_students_taking_history_l2235_223566

-- Defining the conditions
def num_students (total_students history_students statistics_students both_students : ℕ) : Prop :=
  total_students = 89 ∧
  statistics_students = 32 ∧
  (history_students + statistics_students - both_students) = 59 ∧
  (history_students - both_students) = 27

-- The theorem stating that given the conditions, the number of students taking history is 54
theorem students_taking_history :
  ∃ history_students, ∃ statistics_students, ∃ both_students, 
  num_students 89 history_students statistics_students both_students ∧ history_students = 54 :=
by
  sorry

end NUMINAMATH_GPT_students_taking_history_l2235_223566


namespace NUMINAMATH_GPT_counts_of_arson_l2235_223583

-- Define variables A (arson), B (burglary), L (petty larceny)
variables (A B L : ℕ)

-- Conditions given in the problem
def burglary_charges : Prop := B = 2
def petty_larceny_charges_relation : Prop := L = 6 * B
def total_sentence_calculation : Prop := 36 * A + 18 * B + 6 * L = 216

-- Prove that given these conditions, the counts of arson (A) is 3
theorem counts_of_arson (h1 : burglary_charges B)
                        (h2 : petty_larceny_charges_relation B L)
                        (h3 : total_sentence_calculation A B L) :
                        A = 3 :=
sorry

end NUMINAMATH_GPT_counts_of_arson_l2235_223583


namespace NUMINAMATH_GPT_count_multiples_4_or_5_not_20_l2235_223539

-- We define the necessary ranges and conditions
def is_multiple_of (n k : ℕ) := n % k = 0

def count_multiples (n k : ℕ) := (n / k)

def not_multiple_of (n k : ℕ) := ¬ is_multiple_of n k

def count_multiples_excluding (n k l : ℕ) :=
  count_multiples n k + count_multiples n l - count_multiples n (Nat.lcm k l)

theorem count_multiples_4_or_5_not_20 : count_multiples_excluding 3010 4 5 = 1204 := 
by
  sorry

end NUMINAMATH_GPT_count_multiples_4_or_5_not_20_l2235_223539


namespace NUMINAMATH_GPT_total_balls_in_bag_l2235_223504

theorem total_balls_in_bag (x : ℕ) (H : 3/(4 + x) = x/(4 + x)) : 3 + 1 + x = 7 :=
by
  -- We would provide the proof here, but it's not required as per the instructions.
  sorry

end NUMINAMATH_GPT_total_balls_in_bag_l2235_223504


namespace NUMINAMATH_GPT_who_made_statements_and_fate_l2235_223511

namespace IvanTsarevichProblem

-- Define the characters and their behaviors
inductive Animal
| Bear : Animal
| Fox : Animal
| Wolf : Animal

def always_true (s : Prop) : Prop := s
def always_false (s : Prop) : Prop := ¬s
def alternates (s1 s2 : Prop) : Prop := s1 ∧ ¬s2

-- Statements made by the animals
def statement1 (save_die : Bool) : Prop := save_die = true
def statement2 (safe_sound_save : Bool) : Prop := safe_sound_save = true
def statement3 (safe_lose : Bool) : Prop := safe_lose = true

-- Analyze truth based on behaviors
noncomputable def belongs_to (a : Animal) (s : Prop) : Prop :=
  match a with
  | Animal.Bear => always_true s
  | Animal.Fox => always_false s
  | Animal.Wolf =>
    match s with
    | ss => alternates (ss = true) (ss = false)

-- Given conditions
axiom h1 : statement1 false -- Fox lies, so "You will save the horse. But you will die." is false
axiom h2 : statement2 false -- Wolf alternates, so "You will stay safe and sound. And you will save the horse." is a mix
axiom h3 : statement3 true  -- Bear tells the truth, so "You will survive. But you will lose the horse." is true

-- Conclusion: Animal who made each statement
theorem who_made_statements_and_fate : 
  belongs_to Animal.Fox (statement1 false) ∧ 
  belongs_to Animal.Wolf (statement2 false) ∧ 
  belongs_to Animal.Bear (statement3 true) ∧ 
  (¬safe_lose) := sorry

end IvanTsarevichProblem

end NUMINAMATH_GPT_who_made_statements_and_fate_l2235_223511


namespace NUMINAMATH_GPT_neg_of_proposition_l2235_223510

variable (a : ℝ)

def proposition := ∀ x : ℝ, 0 < a^x

theorem neg_of_proposition (h₀ : 0 < a) (h₁ : a ≠ 1) : ¬proposition a ↔ ∃ x : ℝ, a^x ≤ 0 :=
by
  sorry

end NUMINAMATH_GPT_neg_of_proposition_l2235_223510


namespace NUMINAMATH_GPT_mariel_dogs_count_l2235_223526

theorem mariel_dogs_count
  (num_dogs_other: Nat)
  (num_legs_tangled: Nat)
  (num_legs_per_dog: Nat)
  (num_legs_per_human: Nat)
  (num_dog_walkers: Nat)
  (num_dogs_mariel: Nat):
  num_dogs_other = 3 →
  num_legs_tangled = 36 →
  num_legs_per_dog = 4 →
  num_legs_per_human = 2 →
  num_dog_walkers = 2 →
  4*num_dogs_mariel + 4*num_dogs_other + 2*num_dog_walkers = num_legs_tangled →
  num_dogs_mariel = 5 :=
by 
  intros h_other h_tangled h_legs_dog h_legs_human h_walkers h_eq
  sorry

end NUMINAMATH_GPT_mariel_dogs_count_l2235_223526


namespace NUMINAMATH_GPT_intersect_points_count_l2235_223503

open Classical
open Real

noncomputable def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := sorry

axiom f_invertible : ∀ x y : ℝ, f x = f y ↔ x = y

theorem intersect_points_count : ∃ (count : ℕ), count = 3 ∧ ∀ x : ℝ, (f (x ^ 3) = f (x ^ 5)) ↔ (x = 0 ∨ x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_GPT_intersect_points_count_l2235_223503


namespace NUMINAMATH_GPT_machine_C_time_l2235_223546

theorem machine_C_time (T_c : ℝ) : 
  (1/4) + (1/3) + (1/T_c) = (3/4) → T_c = 6 := 
by 
  sorry

end NUMINAMATH_GPT_machine_C_time_l2235_223546


namespace NUMINAMATH_GPT_b_plus_c_eq_neg3_l2235_223548

theorem b_plus_c_eq_neg3 (b c : ℝ)
  (h1 : ∀ x : ℝ, x^2 + b * x + c > 0 ↔ (x < -1 ∨ x > 2)) :
  b + c = -3 :=
sorry

end NUMINAMATH_GPT_b_plus_c_eq_neg3_l2235_223548


namespace NUMINAMATH_GPT_find_a_if_perpendicular_l2235_223591

def m (a : ℝ) : ℝ × ℝ := (3, a - 1)
def n (a : ℝ) : ℝ × ℝ := (a, -2)

theorem find_a_if_perpendicular (a : ℝ) (h : (m a).fst * (n a).fst + (m a).snd * (n a).snd = 0) : a = -2 :=
by sorry

end NUMINAMATH_GPT_find_a_if_perpendicular_l2235_223591


namespace NUMINAMATH_GPT_length_of_parallel_at_60N_l2235_223505

noncomputable def parallel_length (R : ℝ) (lat_deg : ℝ) : ℝ :=
  2 * Real.pi * R * Real.cos (Real.pi * lat_deg / 180)

theorem length_of_parallel_at_60N :
  parallel_length 20 60 = 20 * Real.pi :=
by
  sorry

end NUMINAMATH_GPT_length_of_parallel_at_60N_l2235_223505


namespace NUMINAMATH_GPT_arcsin_sqrt3_div_2_eq_pi_div_3_l2235_223567

theorem arcsin_sqrt3_div_2_eq_pi_div_3 : Real.arcsin (Real.sqrt 3 / 2) = Real.pi / 3 :=
by
  sorry

end NUMINAMATH_GPT_arcsin_sqrt3_div_2_eq_pi_div_3_l2235_223567


namespace NUMINAMATH_GPT_problem1_problem2_l2235_223524

noncomputable def g (x : ℝ) : ℝ := 3 * x^5 - 4 * x^4 + 2 * x^3 - 28 * x^2 + 15 * x - 90

noncomputable def g' (x : ℝ) : ℝ := 15 * x^4 - 16 * x^3 + 6 * x^2 - 56 * x + 15

theorem problem1 : g 6 = 17568 := 
by {
  sorry
}

theorem problem2 : g' 6 = 15879 := 
by {
  sorry
}

end NUMINAMATH_GPT_problem1_problem2_l2235_223524


namespace NUMINAMATH_GPT_train_speed_in_kmph_l2235_223555

noncomputable def motorbike_speed : ℝ := 64
noncomputable def overtaking_time : ℝ := 40
noncomputable def train_length_meters : ℝ := 400.032

theorem train_speed_in_kmph :
  let train_length_km := train_length_meters / 1000
  let overtaking_time_hours := overtaking_time / 3600
  let relative_speed := train_length_km / overtaking_time_hours
  let train_speed := motorbike_speed + relative_speed
  train_speed = 100.00288 := by
  sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l2235_223555


namespace NUMINAMATH_GPT_price_of_pastries_is_5_l2235_223593

noncomputable def price_of_reuben : ℕ := 3
def price_of_pastries (price_reuben : ℕ) : ℕ := price_reuben + 2

theorem price_of_pastries_is_5 
    (reuben_price cost_pastries : ℕ) 
    (h1 : cost_pastries = reuben_price + 2) 
    (h2 : 10 * reuben_price + 5 * cost_pastries = 55) :
    cost_pastries = 5 :=
by
    sorry

end NUMINAMATH_GPT_price_of_pastries_is_5_l2235_223593


namespace NUMINAMATH_GPT_class_mean_l2235_223536

theorem class_mean
  (num_students_1 : ℕ)
  (num_students_2 : ℕ)
  (total_students : ℕ)
  (mean_score_1 : ℚ)
  (mean_score_2 : ℚ)
  (new_mean_score : ℚ)
  (h1 : num_students_1 + num_students_2 = total_students)
  (h2 : total_students = 30)
  (h3 : num_students_1 = 24)
  (h4 : mean_score_1 = 80)
  (h5 : num_students_2 = 6)
  (h6 : mean_score_2 = 85) :
  new_mean_score = 81 :=
by
  sorry

end NUMINAMATH_GPT_class_mean_l2235_223536


namespace NUMINAMATH_GPT_seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l2235_223585

-- Problem 1
theorem seq_inv_an_is_arithmetic (a : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) :
  ∃ d, ∀ n, n ≥ 2 → (1 / a n) = 2 + (n - 1) * d :=
sorry

-- Problem 2
theorem seq_fn_over_an_has_minimum (a f : ℕ → ℝ) (h1 : a 1 = 1/2) (h2 : ∀ n, n ≥ 2 → a (n - 1) / a n = (a (n - 1) + 2) / (2 - a n)) (h3 : ∀ n, f n = (9 / 10) ^ n) :
  ∃ m, ∀ n, n ≠ m → f n / a n ≥ f m / a m :=
sorry

end NUMINAMATH_GPT_seq_inv_an_is_arithmetic_seq_fn_over_an_has_minimum_l2235_223585


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l2235_223595

-- Define the function f(x)
def f (a x : ℝ) := |a - 3 * x|

-- Define the condition for the function to be monotonically increasing on [1, +∞)
def is_monotonically_increasing_on_interval (a : ℝ) : Prop :=
  ∀ (x y : ℝ), 1 ≤ x → x ≤ y → (f a x ≤ f a y)

-- Define the condition that a must be 3
def condition_a_eq_3 (a : ℝ) : Prop := (a = 3)

-- Prove that condition_a_eq_3 is a necessary but not sufficient condition
theorem necessary_but_not_sufficient (a : ℝ) :
  (is_monotonically_increasing_on_interval a) →
  condition_a_eq_3 a ↔ (∀ (b : ℝ), b ≠ a → is_monotonically_increasing_on_interval b → false) := 
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l2235_223595


namespace NUMINAMATH_GPT_range_of_m_l2235_223588

-- Define the first circle
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 10*y + 1 = 0

-- Define the second circle
def circle2 (x y m : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y - m = 0

-- Lean statement for the proof problem
theorem range_of_m (m : ℝ) : 
  (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) ↔ -1 < m ∧ m < 79 :=
by sorry

end NUMINAMATH_GPT_range_of_m_l2235_223588


namespace NUMINAMATH_GPT_min_value_problem1_l2235_223520

theorem min_value_problem1 (x : ℝ) (hx : x > -1) : 
  ∃ m, m = 2 * Real.sqrt 2 + 1 ∧ (∀ y, y = (x^2 + 3 * x + 4) / (x + 1) ∧ x > -1 → y ≥ m) :=
sorry

end NUMINAMATH_GPT_min_value_problem1_l2235_223520


namespace NUMINAMATH_GPT_factor_cubic_expression_l2235_223530

theorem factor_cubic_expression :
  ∃ a b c : ℕ, 
  a > b ∧ b > c ∧ 
  x^3 - 16 * x^2 + 65 * x - 80 = (x - a) * (x - b) * (x - c) ∧ 
  3 * b - c = 12 := 
sorry

end NUMINAMATH_GPT_factor_cubic_expression_l2235_223530


namespace NUMINAMATH_GPT_calculate_difference_of_squares_l2235_223521

theorem calculate_difference_of_squares : (153^2 - 147^2) = 1800 := by
  sorry

end NUMINAMATH_GPT_calculate_difference_of_squares_l2235_223521


namespace NUMINAMATH_GPT_alec_correct_problems_l2235_223515

-- Definitions of conditions and proof problem
theorem alec_correct_problems (c w : ℕ) (s : ℕ) (H1 : s = 30 + 4 * c - w) (H2 : s > 90)
  (H3 : ∀ s', 90 < s' ∧ s' < s → ¬(∃ c', ∃ w', s' = 30 + 4 * c' - w')) :
  c = 16 :=
by
  sorry

end NUMINAMATH_GPT_alec_correct_problems_l2235_223515


namespace NUMINAMATH_GPT_max_value_S_n_S_m_l2235_223528

noncomputable def a (n : ℕ) : ℤ := -(n : ℤ)^2 + 12 * n - 32

noncomputable def S : ℕ → ℤ
| 0       => 0
| (n + 1) => S n + a (n + 1)

theorem max_value_S_n_S_m : ∀ m n : ℕ, m < n → m > 0 → S n - S m ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_max_value_S_n_S_m_l2235_223528


namespace NUMINAMATH_GPT_loss_is_selling_price_of_16_pencils_l2235_223562

theorem loss_is_selling_price_of_16_pencils
  (S : ℝ) -- Assume the selling price of one pencil is S
  (C : ℝ) -- Assume the cost price of one pencil is C
  (h₁ : 80 * C = 1.2 * 80 * S) -- The cost of 80 pencils is 1.2 times the selling price of 80 pencils
  : (80 * C - 80 * S) = 16 * S := -- The loss for selling 80 pencils equals the selling price of 16 pencils
  sorry

end NUMINAMATH_GPT_loss_is_selling_price_of_16_pencils_l2235_223562


namespace NUMINAMATH_GPT_shepherds_sheep_l2235_223584

theorem shepherds_sheep (x y : ℕ) 
  (h1 : x - 4 = y + 4) 
  (h2 : x + 4 = 3 * (y - 4)) : 
  x = 20 ∧ y = 12 := 
by 
  sorry

end NUMINAMATH_GPT_shepherds_sheep_l2235_223584


namespace NUMINAMATH_GPT_earnings_per_puppy_l2235_223553

def daily_pay : ℝ := 40
def total_earnings : ℝ := 76
def num_puppies : ℕ := 16

theorem earnings_per_puppy : (total_earnings - daily_pay) / num_puppies = 2.25 := by
  sorry

end NUMINAMATH_GPT_earnings_per_puppy_l2235_223553


namespace NUMINAMATH_GPT_total_charge_3_hours_l2235_223535

-- Define the charges for the first hour (F) and additional hours (A)
variable (F A : ℝ)

-- Given conditions
axiom charge_relation : F = A + 20
axiom total_charge_5_hours : F + 4 * A = 300

-- The theorem stating the total charge for 3 hours of therapy
theorem total_charge_3_hours : 
  (F + 2 * A) = 188 :=
by
  -- Insert the proof here
  sorry

end NUMINAMATH_GPT_total_charge_3_hours_l2235_223535


namespace NUMINAMATH_GPT_congruence_problem_l2235_223542

theorem congruence_problem {x : ℤ} (h : 4 * x + 5 ≡ 3 [ZMOD 20]) : 3 * x + 8 ≡ 2 [ZMOD 10] :=
sorry

end NUMINAMATH_GPT_congruence_problem_l2235_223542


namespace NUMINAMATH_GPT_point_P_quadrant_IV_l2235_223508

theorem point_P_quadrant_IV (x y : ℝ) (h1 : x > 0) (h2 : y < 0) : x > 0 ∧ y < 0 :=
by
  sorry

end NUMINAMATH_GPT_point_P_quadrant_IV_l2235_223508


namespace NUMINAMATH_GPT_fabric_problem_l2235_223572

theorem fabric_problem
  (x y : ℝ)
  (h1 : y > 0)
  (cost_second_piece := x)
  (cost_first_piece := x + 126)
  (cost_per_meter_first := (x + 126) / y)
  (cost_per_meter_second := x / y)
  (h2 : 4 * cost_per_meter_first - 3 * cost_per_meter_second = 135)
  (h3 : 3 * cost_per_meter_first + 4 * cost_per_meter_second = 382.5) :
  y = 5.6 ∧ cost_per_meter_first = 67.5 ∧ cost_per_meter_second = 45 :=
sorry

end NUMINAMATH_GPT_fabric_problem_l2235_223572


namespace NUMINAMATH_GPT_percentage_vehicles_updated_2003_l2235_223518

theorem percentage_vehicles_updated_2003 (a : ℝ) (h1 : 1.1^4 = 1.46) (h2 : 1.1^5 = 1.61) :
  (a * 1 / (a * 1.61) * 100 = 16.4) :=
  by sorry

end NUMINAMATH_GPT_percentage_vehicles_updated_2003_l2235_223518


namespace NUMINAMATH_GPT_julio_twice_james_in_years_l2235_223570

noncomputable def years_until_julio_twice_james := 
  let x := 14
  (36 + x = 2 * (11 + x))

theorem julio_twice_james_in_years : 
  years_until_julio_twice_james := 
  by 
  sorry

end NUMINAMATH_GPT_julio_twice_james_in_years_l2235_223570


namespace NUMINAMATH_GPT_arithmetic_mean_of_1_and_4_l2235_223587

theorem arithmetic_mean_of_1_and_4 : 
  (1 + 4) / 2 = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_arithmetic_mean_of_1_and_4_l2235_223587


namespace NUMINAMATH_GPT_lattice_points_on_hyperbola_l2235_223579

open Real

theorem lattice_points_on_hyperbola : 
  ∃ (S : Finset (ℤ × ℤ)), 
  (∀ (x y : ℤ), ((x, y) ∈ S ↔ x^2 - y^2 = 65)) ∧ S.card = 8 :=
by
  sorry

end NUMINAMATH_GPT_lattice_points_on_hyperbola_l2235_223579


namespace NUMINAMATH_GPT_honey_harvested_correct_l2235_223540

def honey_harvested_last_year : ℕ := 2479
def honey_increase_this_year : ℕ := 6085
def honey_harvested_this_year : ℕ := 8564

theorem honey_harvested_correct :
  honey_harvested_last_year + honey_increase_this_year = honey_harvested_this_year :=
sorry

end NUMINAMATH_GPT_honey_harvested_correct_l2235_223540


namespace NUMINAMATH_GPT_expression_simplifies_to_one_l2235_223581

theorem expression_simplifies_to_one :
  ( (105^2 - 8^2) / (80^2 - 13^2) ) * ( (80 - 13) * (80 + 13) / ( (105 - 8) * (105 + 8) ) ) = 1 :=
by
  sorry

end NUMINAMATH_GPT_expression_simplifies_to_one_l2235_223581


namespace NUMINAMATH_GPT_person_Y_share_l2235_223558

theorem person_Y_share (total_amount : ℝ) (r1 r2 r3 r4 r5 : ℝ) (ratio_Y : ℝ) 
  (h1 : total_amount = 1390) 
  (h2 : r1 = 13) 
  (h3 : r2 = 17)
  (h4 : r3 = 23) 
  (h5 : r4 = 29) 
  (h6 : r5 = 37) 
  (h7 : ratio_Y = 29): 
  (total_amount / (r1 + r2 + r3 + r4 + r5) * ratio_Y) = 338.72 :=
by
  sorry

end NUMINAMATH_GPT_person_Y_share_l2235_223558


namespace NUMINAMATH_GPT_max_sum_arith_seq_l2235_223559

theorem max_sum_arith_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) :
  (∀ n, a n = 8 + (n - 1) * d) →
  d ≠ 0 →
  a 1 = 8 →
  a 5 ^ 2 = a 1 * a 7 →
  S n = n * a 1 + (n * (n - 1) * d) / 2 →
  ∃ n : ℕ, S n = 36 :=
by
  intros
  sorry

end NUMINAMATH_GPT_max_sum_arith_seq_l2235_223559


namespace NUMINAMATH_GPT_system_of_equations_solutions_l2235_223561

theorem system_of_equations_solutions (x y a b : ℝ) 
  (h1 : 2 * x + y = b) 
  (h2 : x - b * y = a) 
  (hx : x = 1)
  (hy : y = 0) : a - b = -1 :=
by 
  sorry

end NUMINAMATH_GPT_system_of_equations_solutions_l2235_223561


namespace NUMINAMATH_GPT_roots_quadratic_square_diff_10_l2235_223598

-- Definition and theorem statement in Lean 4
theorem roots_quadratic_square_diff_10 :
  ∀ x1 x2 : ℝ, (2 * x1^2 + 4 * x1 - 3 = 0) ∧ (2 * x2^2 + 4 * x2 - 3 = 0) →
  (x1 - x2)^2 = 10 :=
by
  sorry

end NUMINAMATH_GPT_roots_quadratic_square_diff_10_l2235_223598


namespace NUMINAMATH_GPT_friend_gain_is_20_percent_l2235_223514

noncomputable def original_cost : ℝ := 52325.58
noncomputable def loss_percentage : ℝ := 0.14
noncomputable def friend_selling_price : ℝ := 54000
noncomputable def friend_percentage_gain : ℝ :=
  ((friend_selling_price - (original_cost * (1 - loss_percentage))) / (original_cost * (1 - loss_percentage))) * 100

theorem friend_gain_is_20_percent :
  friend_percentage_gain = 20 := by
  sorry

end NUMINAMATH_GPT_friend_gain_is_20_percent_l2235_223514


namespace NUMINAMATH_GPT_correct_statements_B_and_C_l2235_223592

-- Given real numbers a, b, c satisfying the conditions
variables (a b c : ℝ)
variables (h1 : a > b)
variables (h2 : b > c)
variables (h3 : a + b + c = 0)

theorem correct_statements_B_and_C : (a - c > 2 * b) ∧ (a ^ 2 > b ^ 2) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_B_and_C_l2235_223592


namespace NUMINAMATH_GPT_unique_bijective_function_l2235_223571

noncomputable def find_bijective_function {n : ℕ}
  (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ)
  (f : Fin n → ℝ) : Prop :=
∀ i : Fin n, f i = x i

theorem unique_bijective_function (n : ℕ) (hn : n ≥ 3) (hodd : n % 2 = 1)
  (x : Fin n → ℝ) (f : Fin n → ℝ)
  (hf_bij : Function.Bijective f)
  (h_abs_diff : ∀ i, |f i - x i| = 0) : find_bijective_function hn hodd x f :=
by
  sorry

end NUMINAMATH_GPT_unique_bijective_function_l2235_223571


namespace NUMINAMATH_GPT_workout_days_l2235_223575

theorem workout_days (n : ℕ) (squats : ℕ → ℕ) 
  (h1 : squats 1 = 30)
  (h2 : ∀ k, squats (k + 1) = squats k + 5)
  (h3 : squats 4 = 45) :
  n = 4 :=
sorry

end NUMINAMATH_GPT_workout_days_l2235_223575


namespace NUMINAMATH_GPT_polynomial_value_l2235_223599

theorem polynomial_value (x : ℝ) (hx : x^2 - 4*x + 1 = 0) : 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 - 32*Real.sqrt 3 ∨ 
  x^4 - 8*x^3 + 10*x^2 - 8*x + 1 = -56 + 32*Real.sqrt 3 :=
sorry

end NUMINAMATH_GPT_polynomial_value_l2235_223599


namespace NUMINAMATH_GPT_motorcyclist_cross_time_l2235_223590

/-- Definitions and conditions -/
def speed_X := 2 -- Rounds per hour
def speed_Y := 4 -- Rounds per hour

/-- Proof statement -/
theorem motorcyclist_cross_time : (1 / (speed_X + speed_Y) * 60 = 10) :=
by
  sorry

end NUMINAMATH_GPT_motorcyclist_cross_time_l2235_223590


namespace NUMINAMATH_GPT_percentage_reduction_price_increase_for_profit_price_increase_max_profit_l2235_223565

-- Define the conditions
def original_price : ℝ := 50
def final_price : ℝ := 32
def daily_sales : ℝ := 500
def profit_per_kg : ℝ := 10
def sales_decrease_per_yuan : ℝ := 20
def required_daily_profit : ℝ := 6000
def max_possible_profit : ℝ := 6125

-- Proving the percentage reduction each time
theorem percentage_reduction (a : ℝ) :
  (original_price * (1 - a) ^ 2 = final_price) → (a = 0.2) :=
sorry

-- Proving the price increase per kilogram to ensure a daily profit of 6000 yuan
theorem price_increase_for_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = required_daily_profit) → (x = 5) :=
sorry

-- Proving the price increase per kilogram to maximize daily profit
theorem price_increase_max_profit (x : ℝ) :
  ((profit_per_kg + x) * (daily_sales - sales_decrease_per_yuan * x) = max_possible_profit) → (x = 7.5) :=
sorry

end NUMINAMATH_GPT_percentage_reduction_price_increase_for_profit_price_increase_max_profit_l2235_223565


namespace NUMINAMATH_GPT_carolyn_initial_marbles_l2235_223541

theorem carolyn_initial_marbles (x : ℕ) (h1 : x - 42 = 5) : x = 47 :=
by
  sorry

end NUMINAMATH_GPT_carolyn_initial_marbles_l2235_223541


namespace NUMINAMATH_GPT_max_value_k_l2235_223525

noncomputable def seq (n : ℕ) : ℕ :=
  match n with
  | 0     => 4
  | (n+1) => 3 * seq n - 2

theorem max_value_k (k : ℝ) :
  (∀ n : ℕ, n > 0 → k * (seq n) ≤ 9^n) → k ≤ 9 / 4 :=
sorry

end NUMINAMATH_GPT_max_value_k_l2235_223525


namespace NUMINAMATH_GPT_no_four_consecutive_powers_l2235_223509

/-- 
  There do not exist four consecutive natural numbers 
  such that each of them is a power (greater than 1) of another natural number.
-/
theorem no_four_consecutive_powers : 
  ¬ ∃ (n : ℕ), (∀ (i : ℕ), i < 4 → ∃ (a k : ℕ), k > 1 ∧ n + i = a^k) := sorry

end NUMINAMATH_GPT_no_four_consecutive_powers_l2235_223509


namespace NUMINAMATH_GPT_jeans_cost_l2235_223551

-- Definitions based on conditions
def shirt_cost : ℕ := 5
def hat_cost : ℕ := 4
def total_cost : ℕ := 51
def n_shirts : ℕ := 3
def n_hats : ℕ := 4
def n_jeans : ℕ := 2

-- The goal is to prove that the cost of one pair of jeans (J) is 10
theorem jeans_cost (J : ℕ) (h : n_shirts * shirt_cost + n_jeans * J + n_hats * hat_cost = total_cost) : J = 10 :=
  sorry

end NUMINAMATH_GPT_jeans_cost_l2235_223551


namespace NUMINAMATH_GPT_value_of_three_inch_cube_l2235_223594

theorem value_of_three_inch_cube (value_two_inch: ℝ) (volume_two_inch: ℝ) (volume_three_inch: ℝ) (cost_two_inch: ℝ):
  value_two_inch = cost_two_inch * ((volume_three_inch / volume_two_inch): ℝ) := 
by
  have volume_two_inch := 2^3 -- Volume of two-inch cube
  have volume_three_inch := 3^3 -- Volume of three-inch cube
  let volume_ratio := (volume_three_inch / volume_two_inch: ℝ)
  have := cost_two_inch * volume_ratio
  norm_num
  sorry

end NUMINAMATH_GPT_value_of_three_inch_cube_l2235_223594


namespace NUMINAMATH_GPT_recurrence_solution_proof_l2235_223545

noncomputable def recurrence_relation (a : ℕ → ℚ) : Prop :=
  (∀ n ≥ 2, a n = 5 * a (n - 1) - 6 * a (n - 2) + n + 2) ∧
  a 0 = 27 / 4 ∧
  a 1 = 49 / 4

noncomputable def solution (a : ℕ → ℚ) : Prop :=
  ∀ n, a n = 3 * 2^n + 3^n + n / 2 + 11 / 4

theorem recurrence_solution_proof : ∃ a : ℕ → ℚ, recurrence_relation a ∧ solution a :=
by { sorry }

end NUMINAMATH_GPT_recurrence_solution_proof_l2235_223545


namespace NUMINAMATH_GPT_cupcake_price_l2235_223516

theorem cupcake_price
  (x : ℝ)
  (h1 : 5 * x + 6 * 1 + 4 * 2 + 15 * 0.6 = 33) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_cupcake_price_l2235_223516


namespace NUMINAMATH_GPT_total_soccer_balls_purchased_l2235_223507

theorem total_soccer_balls_purchased : 
  (∃ (x : ℝ), 
    800 / x * 2 = 1560 / (x - 2)) → 
  (800 / x + 1560 / (x - 2) = 30) :=
by
  sorry

end NUMINAMATH_GPT_total_soccer_balls_purchased_l2235_223507


namespace NUMINAMATH_GPT_lcm_36_65_l2235_223501

-- Definitions based on conditions
def number1 : ℕ := 36
def number2 : ℕ := 65

-- The prime factorization conditions can be implied through deriving LCM hence added as comments to clarify the conditions.
-- 36 = 2^2 * 3^2
-- 65 = 5 * 13

-- Theorem statement that the LCM of number1 and number2 is 2340
theorem lcm_36_65 : Nat.lcm number1 number2 = 2340 := 
by 
  sorry

end NUMINAMATH_GPT_lcm_36_65_l2235_223501


namespace NUMINAMATH_GPT_triangle_is_right_triangle_l2235_223512

theorem triangle_is_right_triangle 
  {A B C : ℝ} {a b c : ℝ} 
  (h₁ : b - a * Real.cos B = a * Real.cos C - c) 
  (h₂ : ∀ (angle : ℝ), 0 < angle ∧ angle < π) : A = π / 2 := 
sorry

end NUMINAMATH_GPT_triangle_is_right_triangle_l2235_223512


namespace NUMINAMATH_GPT_simplify_fractions_l2235_223544

theorem simplify_fractions :
  (30 / 45) * (75 / 128) * (256 / 150) = 1 / 6 := 
by
  sorry

end NUMINAMATH_GPT_simplify_fractions_l2235_223544


namespace NUMINAMATH_GPT_range_zero_of_roots_l2235_223573

theorem range_zero_of_roots (x y z w : ℝ) (h1 : x + y + z + w = 0) 
                            (h2 : x^7 + y^7 + z^7 + w^7 = 0) :
  w * (w + x) * (w + y) * (w + z) = 0 :=
  sorry

end NUMINAMATH_GPT_range_zero_of_roots_l2235_223573


namespace NUMINAMATH_GPT_price_difference_l2235_223552

noncomputable def original_price (discounted_price : ℝ) : ℝ :=
  discounted_price / 0.85

noncomputable def final_price (discounted_price : ℝ) : ℝ :=
  discounted_price * 1.25

theorem price_difference (discounted_price : ℝ) (h : discounted_price = 71.4) : 
  (final_price discounted_price) - (original_price discounted_price) = 5.25 := 
by
  sorry

end NUMINAMATH_GPT_price_difference_l2235_223552


namespace NUMINAMATH_GPT_octal_742_to_decimal_l2235_223527

theorem octal_742_to_decimal : (7 * 8^2 + 4 * 8^1 + 2 * 8^0 = 482) :=
by
  sorry

end NUMINAMATH_GPT_octal_742_to_decimal_l2235_223527


namespace NUMINAMATH_GPT_second_pipe_filling_time_l2235_223538

theorem second_pipe_filling_time :
  ∃ T : ℝ, (1/20 + 1/T) * 2/3 * 16 = 1 ∧ T = 160/7 :=
by
  use 160 / 7
  sorry

end NUMINAMATH_GPT_second_pipe_filling_time_l2235_223538


namespace NUMINAMATH_GPT_min_a1_a7_l2235_223589

noncomputable def geom_seq (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ r : ℝ, r > 0 ∧ a (n + 1) = a n * r

theorem min_a1_a7 (a : ℕ → ℝ) (h : geom_seq a)
  (h1 : a 3 * a 5 = 64) :
  ∃ m, m = (a 1 + a 7) ∧ m = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_a1_a7_l2235_223589


namespace NUMINAMATH_GPT_probability_of_both_selected_l2235_223532

noncomputable def ramSelectionProbability : ℚ := 1 / 7
noncomputable def raviSelectionProbability : ℚ := 1 / 5

theorem probability_of_both_selected : 
  ramSelectionProbability * raviSelectionProbability = 1 / 35 :=
by sorry

end NUMINAMATH_GPT_probability_of_both_selected_l2235_223532


namespace NUMINAMATH_GPT_perpendicular_unit_vector_exists_l2235_223519

theorem perpendicular_unit_vector_exists :
  ∃ (m n : ℝ), (2 * m + n = 0) ∧ (m^2 + n^2 = 1) ∧ (m = (Real.sqrt 5) / 5) ∧ (n = -(2 * (Real.sqrt 5)) / 5) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_unit_vector_exists_l2235_223519


namespace NUMINAMATH_GPT_passengers_on_ship_l2235_223549

theorem passengers_on_ship (P : ℕ)
  (h1 : P / 12 + P / 4 + P / 9 + P / 6 + 42 = P) :
  P = 108 := 
by sorry

end NUMINAMATH_GPT_passengers_on_ship_l2235_223549


namespace NUMINAMATH_GPT_problem_statement_l2235_223534

def prop_p (x : ℝ) : Prop := x^2 >= x
def prop_q : Prop := ∃ x : ℝ, x^2 >= x

theorem problem_statement : (∀ x : ℝ, prop_p x) = false ∧ prop_q = true :=
by 
  sorry

end NUMINAMATH_GPT_problem_statement_l2235_223534


namespace NUMINAMATH_GPT_gcd_84_294_315_l2235_223597

def gcd_3_integers : ℕ := Nat.gcd (Nat.gcd 84 294) 315

theorem gcd_84_294_315 : gcd_3_integers = 21 :=
by
  sorry

end NUMINAMATH_GPT_gcd_84_294_315_l2235_223597


namespace NUMINAMATH_GPT_sum_primes_reversed_l2235_223563

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def reverse_digits (n : ℕ) : ℕ := 
  let tens := n / 10
  let ones := n % 10
  10 * ones + tens

def valid_primes : List ℕ := [31, 37, 71, 73]

theorem sum_primes_reversed :
  (∀ p ∈ valid_primes, 20 < p ∧ p < 80 ∧ is_prime p ∧ is_prime (reverse_digits p)) ∧
  (valid_primes.sum = 212) :=
by
  sorry

end NUMINAMATH_GPT_sum_primes_reversed_l2235_223563


namespace NUMINAMATH_GPT_average_run_per_day_l2235_223557

theorem average_run_per_day (n6 n7 n8 : ℕ) 
  (h1 : 3 * n7 = n6) 
  (h2 : 3 * n8 = n7) 
  (h3 : n6 * 20 + n7 * 18 + n8 * 16 = 250 * n8) : 
  (n6 * 20 + n7 * 18 + n8 * 16) / (n6 + n7 + n8) = 250 / 13 :=
by sorry

end NUMINAMATH_GPT_average_run_per_day_l2235_223557


namespace NUMINAMATH_GPT_MrsBrownCarrotYield_l2235_223556

theorem MrsBrownCarrotYield :
  let pacesLength := 25
  let pacesWidth := 30
  let strideLength := 2.5
  let yieldPerSquareFoot := 0.5
  let lengthInFeet := pacesLength * strideLength
  let widthInFeet := pacesWidth * strideLength
  let area := lengthInFeet * widthInFeet
  let yield := area * yieldPerSquareFoot
  yield = 2343.75 :=
by
  sorry

end NUMINAMATH_GPT_MrsBrownCarrotYield_l2235_223556


namespace NUMINAMATH_GPT_pages_printed_l2235_223577

theorem pages_printed (P : ℕ) 
  (H1 : P % 7 = 0)
  (H2 : P % 3 = 0)
  (H3 : P - (P / 7 + P / 3 - P / 21) = 24) : 
  P = 42 :=
sorry

end NUMINAMATH_GPT_pages_printed_l2235_223577


namespace NUMINAMATH_GPT_maddox_more_profit_than_theo_l2235_223522

-- Definitions (conditions)
def cost_per_camera : ℕ := 20
def num_cameras : ℕ := 3
def total_cost : ℕ := num_cameras * cost_per_camera

def maddox_selling_price_per_camera : ℕ := 28
def theo_selling_price_per_camera : ℕ := 23

-- Total selling price
def maddox_total_selling_price : ℕ := num_cameras * maddox_selling_price_per_camera
def theo_total_selling_price : ℕ := num_cameras * theo_selling_price_per_camera

-- Profits
def maddox_profit : ℕ := maddox_total_selling_price - total_cost
def theo_profit : ℕ := theo_total_selling_price - total_cost

-- Proof Statement
theorem maddox_more_profit_than_theo : maddox_profit - theo_profit = 15 := by
  sorry

end NUMINAMATH_GPT_maddox_more_profit_than_theo_l2235_223522


namespace NUMINAMATH_GPT_students_absent_percentage_l2235_223550

theorem students_absent_percentage (total_students present_students : ℕ) (h_total : total_students = 50) (h_present : present_students = 45) :
  (total_students - present_students) * 100 / total_students = 10 := 
by
  sorry

end NUMINAMATH_GPT_students_absent_percentage_l2235_223550


namespace NUMINAMATH_GPT_maria_remaining_money_l2235_223574

theorem maria_remaining_money (initial_amount ticket_cost : ℕ) (h_initial : initial_amount = 760) (h_ticket : ticket_cost = 300) :
  let hotel_cost := ticket_cost / 2
  let total_spent := ticket_cost + hotel_cost
  let remaining := initial_amount - total_spent
  remaining = 310 :=
by
  intros
  sorry

end NUMINAMATH_GPT_maria_remaining_money_l2235_223574


namespace NUMINAMATH_GPT_percentage_deposited_l2235_223554

theorem percentage_deposited (amount_deposited income : ℝ) 
  (h1 : amount_deposited = 2500) (h2 : income = 10000) : 
  (amount_deposited / income) * 100 = 25 :=
by
  have amount_deposited_val : amount_deposited = 2500 := h1
  have income_val : income = 10000 := h2
  sorry

end NUMINAMATH_GPT_percentage_deposited_l2235_223554


namespace NUMINAMATH_GPT_sum_of_max_min_a_l2235_223517

theorem sum_of_max_min_a (a : ℝ) (x : ℝ) :
  (x^2 - a * x - 20 * a^2 < 0) →
  (∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 ∧ x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) →
  (∀ max_min_sum : ℝ, max_min_sum = 1 + (-1) → max_min_sum = 0) := 
sorry

end NUMINAMATH_GPT_sum_of_max_min_a_l2235_223517


namespace NUMINAMATH_GPT_inequality_solution_l2235_223543

theorem inequality_solution (x : ℝ) : (-3 * x^2 - 9 * x - 6 ≥ -12) ↔ (-2 ≤ x ∧ x ≤ 1) := sorry

end NUMINAMATH_GPT_inequality_solution_l2235_223543


namespace NUMINAMATH_GPT_value_of_r_minus_q_l2235_223502

variable (q r : ℝ)
variable (slope : ℝ)
variable (h_parallel : slope = 3 / 2)
variable (h_points : (r - q) / (-2) = slope)

theorem value_of_r_minus_q (h_parallel : slope = 3 / 2) (h_points : (r - q) / (-2) = slope) : 
  r - q = -3 := by
  sorry

end NUMINAMATH_GPT_value_of_r_minus_q_l2235_223502


namespace NUMINAMATH_GPT_find_smallest_n_l2235_223513

theorem find_smallest_n 
  (n : ℕ) 
  (hn : 23 * n ≡ 789 [MOD 8]) : 
  ∃ n : ℕ, n > 0 ∧ n ≡ 3 [MOD 8] :=
sorry

end NUMINAMATH_GPT_find_smallest_n_l2235_223513


namespace NUMINAMATH_GPT_gcd_seven_eight_fact_l2235_223533

-- Definitions based on the problem conditions
def seven_fact : ℕ := 1 * 2 * 3 * 4 * 5 * 6 * 7
def eight_fact : ℕ := 8 * seven_fact

-- Statement of the theorem
theorem gcd_seven_eight_fact : Nat.gcd seven_fact eight_fact = seven_fact := by
  sorry

end NUMINAMATH_GPT_gcd_seven_eight_fact_l2235_223533


namespace NUMINAMATH_GPT_johns_grandpa_money_l2235_223568

theorem johns_grandpa_money :
  ∃ G : ℝ, (G + 3 * G = 120) ∧ (G = 30) := 
by
  sorry

end NUMINAMATH_GPT_johns_grandpa_money_l2235_223568


namespace NUMINAMATH_GPT_teresa_jogged_distance_l2235_223580

-- Define the conditions as Lean constants.
def teresa_speed : ℕ := 5 -- Speed in kilometers per hour
def teresa_time : ℕ := 5 -- Time in hours

-- Define the distance formula.
def teresa_distance (speed time : ℕ) : ℕ := speed * time

-- State the theorem.
theorem teresa_jogged_distance : teresa_distance teresa_speed teresa_time = 25 := by
  -- Proof is skipped using 'sorry'.
  sorry

end NUMINAMATH_GPT_teresa_jogged_distance_l2235_223580


namespace NUMINAMATH_GPT_total_selling_price_correct_l2235_223537

def cost_price_1 := 750
def cost_price_2 := 1200
def cost_price_3 := 500

def loss_percent_1 := 10
def loss_percent_2 := 15
def loss_percent_3 := 5

noncomputable def selling_price_1 := cost_price_1 - ((loss_percent_1 / 100) * cost_price_1)
noncomputable def selling_price_2 := cost_price_2 - ((loss_percent_2 / 100) * cost_price_2)
noncomputable def selling_price_3 := cost_price_3 - ((loss_percent_3 / 100) * cost_price_3)

noncomputable def total_selling_price := selling_price_1 + selling_price_2 + selling_price_3

theorem total_selling_price_correct : total_selling_price = 2170 := by
  sorry

end NUMINAMATH_GPT_total_selling_price_correct_l2235_223537


namespace NUMINAMATH_GPT_find_number_l2235_223586

theorem find_number (x : ℝ) (h : 97 * x - 89 * x = 4926) : x = 615.75 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l2235_223586


namespace NUMINAMATH_GPT_arithmetic_problem_l2235_223596

theorem arithmetic_problem : 1357 + 3571 + 5713 - 7135 = 3506 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_problem_l2235_223596


namespace NUMINAMATH_GPT_math_problem_mod_1001_l2235_223506

theorem math_problem_mod_1001 :
  (2^6 * 3^10 * 5^12 - 75^4 * (26^2 - 1)^2 + 3^10 - 50^6 + 5^12) % 1001 = 400 := by
  sorry

end NUMINAMATH_GPT_math_problem_mod_1001_l2235_223506
