import Mathlib

namespace NUMINAMATH_GPT_largest_even_k_for_sum_of_consecutive_integers_l1769_176963

theorem largest_even_k_for_sum_of_consecutive_integers (k n : ℕ) (h_k_even : k % 2 = 0) :
  (3^10 = k * (2 * n + k + 1)) → k ≤ 162 :=
sorry

end NUMINAMATH_GPT_largest_even_k_for_sum_of_consecutive_integers_l1769_176963


namespace NUMINAMATH_GPT_multiple_of_interest_rate_l1769_176974

theorem multiple_of_interest_rate (P r m : ℝ) (h1 : P * r^2 = 40) (h2 : P * (m * r)^2 = 360) : m = 3 :=
by
  sorry

end NUMINAMATH_GPT_multiple_of_interest_rate_l1769_176974


namespace NUMINAMATH_GPT_products_B_correct_l1769_176921

-- Define the total number of products
def total_products : ℕ := 4800

-- Define the sample size and the number of pieces from equipment A in the sample
def sample_size : ℕ := 80
def sample_A : ℕ := 50

-- Define the number of products produced by equipment A and B
def products_A : ℕ := 3000
def products_B : ℕ := total_products - products_A

-- The target number of products produced by equipment B
def target_products_B : ℕ := 1800

-- The theorem we need to prove
theorem products_B_correct :
  products_B = target_products_B := by
  sorry

end NUMINAMATH_GPT_products_B_correct_l1769_176921


namespace NUMINAMATH_GPT_find_ck_l1769_176940

def arithmetic_seq (d : ℕ) (n : ℕ) : ℕ := 1 + (n - 1) * d
def geometric_seq (r : ℕ) (n : ℕ) : ℕ := r^(n - 1)
def c_seq (a_seq : ℕ → ℕ) (b_seq : ℕ → ℕ) (n : ℕ) := a_seq n + b_seq n

theorem find_ck (d r k : ℕ) (a_seq := arithmetic_seq d) (b_seq := geometric_seq r) :
  c_seq a_seq b_seq (k - 1) = 200 →
  c_seq a_seq b_seq (k + 1) = 400 →
  c_seq a_seq b_seq k = 322 :=
by
  sorry

end NUMINAMATH_GPT_find_ck_l1769_176940


namespace NUMINAMATH_GPT_range_of_m_l1769_176958

variable (f : ℝ → ℝ) (m : ℝ)

-- Given conditions
def condition1 := ∀ x, f (-x) = -f x -- f(x) is an odd function
def condition2 := ∀ x, f (x + 3) = f x -- f(x) has a minimum positive period of 3
def condition3 := f 2015 > 1 -- f(2015) > 1
def condition4 := f 1 = (2 * m + 3) / (m - 1) -- f(1) = (2m + 3) / (m - 1)

-- We aim to prove that -2/3 < m < 1 given these conditions.
theorem range_of_m : condition1 f → condition2 f → condition3 f → condition4 f m → -2 / 3 < m ∧ m < 1 := by
  intros
  sorry

end NUMINAMATH_GPT_range_of_m_l1769_176958


namespace NUMINAMATH_GPT_magnitude_a_eq_3sqrt2_l1769_176943

open Real

def a (x: ℝ) : ℝ × ℝ := (3, x)
def b : ℝ × ℝ := (-1, 1)
def perpendicular (u v : ℝ × ℝ) : Prop := u.1 * v.1 + u.2 * v.2 = 0

theorem magnitude_a_eq_3sqrt2 (x : ℝ) (h : perpendicular (a x) b) :
  ‖a 3‖ = 3 * sqrt 2 := by
  sorry

end NUMINAMATH_GPT_magnitude_a_eq_3sqrt2_l1769_176943


namespace NUMINAMATH_GPT_remainder_97_pow_103_mul_7_mod_17_l1769_176954

theorem remainder_97_pow_103_mul_7_mod_17 :
  (97 ^ 103 * 7) % 17 = 13 := by
  have h1 : 97 % 17 = -3 % 17 := by sorry
  have h2 : 9 % 17 = -8 % 17 := by sorry
  have h3 : 64 % 17 = 13 % 17 := by sorry
  have h4 : -21 % 17 = 13 % 17 := by sorry
  sorry

end NUMINAMATH_GPT_remainder_97_pow_103_mul_7_mod_17_l1769_176954


namespace NUMINAMATH_GPT_minimum_value_l1769_176962

noncomputable def f (x : ℝ) (a b : ℝ) := a^x - b
noncomputable def g (x : ℝ) := x + 1

theorem minimum_value (a b : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : f (0 : ℝ) a b * g 0 ≤ 0)
  (h4 : ∀ x : ℝ, f x a b * g x ≤ 0) : (1 / a + 4 / b) ≥ 4 :=
sorry

end NUMINAMATH_GPT_minimum_value_l1769_176962


namespace NUMINAMATH_GPT_binom_eq_sum_l1769_176997

theorem binom_eq_sum (x : ℕ) : (∃ x : ℕ, Nat.choose 7 x = 21) ∧ Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4 :=
by
  sorry

end NUMINAMATH_GPT_binom_eq_sum_l1769_176997


namespace NUMINAMATH_GPT_interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l1769_176999

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem interval_monotonic_increase (k : ℤ) :
  ∀ x : ℝ, -Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ Real.pi / 3 + k * Real.pi →
    ∃ I : Set ℝ, I = Set.Icc (-Real.pi / 6 + k * Real.pi) (Real.pi / 3 + k * Real.pi) ∧
      (∀ x1 x2 : ℝ, x1 ∈ I ∧ x2 ∈ I → x1 ≤ x2 → f x1 ≤ f x2) := sorry

theorem axis_of_symmetry (k : ℤ) :
  ∃ x : ℝ, x = Real.pi / 3 + k * (Real.pi / 2) := sorry

theorem max_and_min_values :
  ∃ (max_val min_val : ℝ), max_val = 2 ∧ min_val = -1 ∧
    (∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 →
      ((f x = 2 ∧ x = Real.pi / 3) ∨ (f x = -1 ∧ x = 0))) := sorry

end NUMINAMATH_GPT_interval_monotonic_increase_axis_of_symmetry_max_and_min_values_l1769_176999


namespace NUMINAMATH_GPT_trajectory_of_T_l1769_176929

-- Define coordinates for points A, T, and M
variables {x x0 y y0 : ℝ}
def A (x0: ℝ) (y0: ℝ) := (x0, y0)
def T (x: ℝ) (y: ℝ) := (x, y)
def M : ℝ × ℝ := (-2, 0)

-- Conditions
def curve (x : ℝ) (y : ℝ) := 4 * x^2 - y + 1 = 0
def vector_condition (x x0 y y0 : ℝ) := (x - x0, y - y0) = 2 * (-2 - x, -y)

theorem trajectory_of_T (x y x0 y0 : ℝ) (hA : curve x0 y0) (hV : vector_condition x x0 y y0) :
  4 * (3 * x + 4)^2 - 3 * y + 1 = 0 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_T_l1769_176929


namespace NUMINAMATH_GPT_distance_along_stream_1_hour_l1769_176908

noncomputable def boat_speed_still_water : ℝ := 4
noncomputable def stream_speed : ℝ := 2
noncomputable def effective_speed_against_stream : ℝ := boat_speed_still_water - stream_speed
noncomputable def effective_speed_along_stream : ℝ := boat_speed_still_water + stream_speed

theorem distance_along_stream_1_hour : 
  effective_speed_agains_stream = 2 → effective_speed_along_stream * 1 = 6 :=
by
  sorry

end NUMINAMATH_GPT_distance_along_stream_1_hour_l1769_176908


namespace NUMINAMATH_GPT_sum_of_radii_tangent_circles_l1769_176933

theorem sum_of_radii_tangent_circles :
  ∃ (r1 r2 : ℝ), 
  (∀ r, (r = (6 + 2*Real.sqrt 6) ∨ r = (6 - 2*Real.sqrt 6)) → (r = r1 ∨ r = r2)) ∧ 
  ((r1 - 4)^2 + r1^2 = (r1 + 2)^2) ∧ 
  ((r2 - 4)^2 + r2^2 = (r2 + 2)^2) ∧ 
  (r1 + r2 = 12) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_radii_tangent_circles_l1769_176933


namespace NUMINAMATH_GPT_bananas_to_mush_l1769_176935

theorem bananas_to_mush (x : ℕ) (h1 : 3 * (20 / x) = 15) : x = 4 :=
by
  sorry

end NUMINAMATH_GPT_bananas_to_mush_l1769_176935


namespace NUMINAMATH_GPT_largest_final_digit_l1769_176982

theorem largest_final_digit (seq : Fin 1002 → Fin 10) 
  (h1 : seq 0 = 2) 
  (h2 : ∀ n : Fin 1001, (17 ∣ (10 * seq n + seq (n + 1))) ∨ (29 ∣ (10 * seq n + seq (n + 1)))) : 
  seq 1001 = 5 :=
sorry

end NUMINAMATH_GPT_largest_final_digit_l1769_176982


namespace NUMINAMATH_GPT_all_three_items_fans_l1769_176983

theorem all_three_items_fans 
  (h1 : ∀ n, n = 4800 % 80 → n = 0)
  (h2 : ∀ n, n = 4800 % 40 → n = 0)
  (h3 : ∀ n, n = 4800 % 60 → n = 0)
  (h4 : ∀ n, n = 4800):
  (∃ k, k = 20):=
by
  sorry

end NUMINAMATH_GPT_all_three_items_fans_l1769_176983


namespace NUMINAMATH_GPT_find_n_l1769_176917

theorem find_n (n : ℕ) (h : n > 0) :
  (n * (n - 1) * (n - 2)) / (6 * n^3) = 1 / 16 ↔ n = 4 :=
by sorry

end NUMINAMATH_GPT_find_n_l1769_176917


namespace NUMINAMATH_GPT_lcm_of_9_12_18_l1769_176989

-- Let's declare the numbers involved
def num1 : ℕ := 9
def num2 : ℕ := 12
def num3 : ℕ := 18

-- Define what it means for a number to be the LCM of num1, num2, and num3
def is_lcm (a b c l : ℕ) : Prop :=
  l % a = 0 ∧ l % b = 0 ∧ l % c = 0 ∧
  ∀ m, (m % a = 0 ∧ m % b = 0 ∧ m % c = 0) → l ≤ m

-- Now state the theorem
theorem lcm_of_9_12_18 : is_lcm num1 num2 num3 36 :=
by
  sorry

end NUMINAMATH_GPT_lcm_of_9_12_18_l1769_176989


namespace NUMINAMATH_GPT_Veenapaniville_high_schools_l1769_176971

theorem Veenapaniville_high_schools :
  ∃ (districtA districtB districtC : ℕ),
    districtA + districtB + districtC = 50 ∧
    (districtA + districtB + districtC = 50) ∧
    (∃ (publicB parochialB privateB : ℕ), 
      publicB + parochialB + privateB = 17 ∧ privateB = 2) ∧
    (∃ (publicC parochialC privateC : ℕ),
      publicC = 9 ∧ parochialC = 9 ∧ privateC = 9 ∧ publicC + parochialC + privateC = 27) ∧
    districtB = 17 ∧
    districtC = 27 →
    districtA = 6 := by
  sorry

end NUMINAMATH_GPT_Veenapaniville_high_schools_l1769_176971


namespace NUMINAMATH_GPT_find_rope_costs_l1769_176922

theorem find_rope_costs (x y : ℕ) (h1 : 10 * x + 5 * y = 175) (h2 : 15 * x + 10 * y = 300) : x = 10 ∧ y = 15 :=
    sorry

end NUMINAMATH_GPT_find_rope_costs_l1769_176922


namespace NUMINAMATH_GPT_problem_part_1_problem_part_2_l1769_176970

noncomputable def f (x : ℝ) (m : ℝ) := |x + 1| + |x - 2| - m

theorem problem_part_1 : 
  {x : ℝ | f x 5 > 0} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 3} :=
by sorry

theorem problem_part_2 (h : ∀ x : ℝ, f x m ≥ 2) : m ≤ 1 :=
by sorry

end NUMINAMATH_GPT_problem_part_1_problem_part_2_l1769_176970


namespace NUMINAMATH_GPT_find_E_l1769_176995

variable (A H C S M N E : ℕ)
variable (x y z l : ℕ)

theorem find_E (h1 : A * x + H * y + C * z = l)
 (h2 : S * x + M * y + N * z = l)
 (h3 : E * x = l)
 (h4 : A ≠ S ∧ A ≠ H ∧ A ≠ C ∧ A ≠ M ∧ A ≠ N ∧ A ≠ E ∧ H ≠ C ∧ H ≠ M ∧ H ≠ N ∧ H ≠ E ∧ C ≠ M ∧ C ≠ N ∧ C ≠ E ∧ M ≠ N ∧ M ≠ E ∧ N ≠ E)
 : E = (A * M + C * N - S * H - N * H) / (M + N - H) := 
sorry

end NUMINAMATH_GPT_find_E_l1769_176995


namespace NUMINAMATH_GPT_five_student_committees_from_ten_select_two_committees_with_three_overlap_l1769_176932

-- Lean statement for the first part: number of different five-student committees from ten students.
theorem five_student_committees_from_ten : 
  (Nat.choose 10 5) = 252 := 
by
  sorry

-- Lean statement for the second part: number of ways to choose two five-student committees with exactly three overlapping members.
theorem select_two_committees_with_three_overlap :
  ( (Nat.choose 10 5) * ( (Nat.choose 5 3) * (Nat.choose 5 2) ) ) / 2 = 12600 := 
by
  sorry

end NUMINAMATH_GPT_five_student_committees_from_ten_select_two_committees_with_three_overlap_l1769_176932


namespace NUMINAMATH_GPT_min_sum_of_integers_cauchy_schwarz_l1769_176916

theorem min_sum_of_integers_cauchy_schwarz :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ 
  (1 / x + 4 / y + 9 / z = 1) ∧ 
  ((x + y + z) = 36) :=
  sorry

end NUMINAMATH_GPT_min_sum_of_integers_cauchy_schwarz_l1769_176916


namespace NUMINAMATH_GPT_power_inequality_l1769_176979

theorem power_inequality (a b n : ℕ) (h_ab : a > b) (h_b1 : b > 1)
  (h_odd_b : b % 2 = 1) (h_n_pos : 0 < n) (h_div : b^n ∣ a^n - 1) :
  a^b > 3^n / n :=
by
  sorry

end NUMINAMATH_GPT_power_inequality_l1769_176979


namespace NUMINAMATH_GPT_range_of_a_l1769_176947

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 + a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1769_176947


namespace NUMINAMATH_GPT_find_p_of_binomial_distribution_l1769_176966

noncomputable def binomial_mean (n : ℕ) (p : ℝ) : ℝ :=
  n * p

theorem find_p_of_binomial_distribution (p : ℝ) (h : binomial_mean 5 p = 2) : p = 0.4 :=
by
  sorry

end NUMINAMATH_GPT_find_p_of_binomial_distribution_l1769_176966


namespace NUMINAMATH_GPT_speed_of_A_l1769_176900

theorem speed_of_A (V_B : ℝ) (h_VB : V_B = 4.555555555555555)
  (h_B_overtakes: ∀ (t_A t_B : ℝ), t_A = t_B + 0.5 → t_B = 1.8) 
  : ∃ V_A : ℝ, V_A = 3.57 :=
by
  sorry

end NUMINAMATH_GPT_speed_of_A_l1769_176900


namespace NUMINAMATH_GPT_temperature_on_friday_l1769_176902

theorem temperature_on_friday 
  (M T W Th F : ℝ)
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 42) : 
  F = 34 :=
by
  sorry

end NUMINAMATH_GPT_temperature_on_friday_l1769_176902


namespace NUMINAMATH_GPT_percentage_increase_in_items_sold_l1769_176910

-- Definitions
variables (P N M : ℝ)
-- Given conditions:
-- The new price of an item
def new_price := P * 0.90
-- The relationship between incomes
def income_increase := (P * 0.90) * M = P * N * 1.125

-- The problem statement
theorem percentage_increase_in_items_sold (h : income_increase P N M) :
  M = N * 1.25 :=
sorry

end NUMINAMATH_GPT_percentage_increase_in_items_sold_l1769_176910


namespace NUMINAMATH_GPT_laura_running_speed_l1769_176934

theorem laura_running_speed (x : ℝ) (hx : 3 * x + 1 > 0) : 
    (30 / (3 * x + 1)) + (10 / x) = 31 / 12 → x = 7.57 := 
by 
  sorry

end NUMINAMATH_GPT_laura_running_speed_l1769_176934


namespace NUMINAMATH_GPT_ratio_of_numbers_l1769_176906

theorem ratio_of_numbers (a b : ℕ) (hHCF : Nat.gcd a b = 4) (hLCM : Nat.lcm a b = 48) : a / b = 3 / 4 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_numbers_l1769_176906


namespace NUMINAMATH_GPT_books_remaining_in_special_collection_l1769_176980

theorem books_remaining_in_special_collection
  (initial_books : ℕ)
  (loaned_books : ℕ)
  (returned_percentage : ℕ)
  (initial_books_eq : initial_books = 75)
  (loaned_books_eq : loaned_books = 45)
  (returned_percentage_eq : returned_percentage = 80) :
  ∃ final_books : ℕ, final_books = initial_books - (loaned_books - (loaned_books * returned_percentage / 100)) ∧ final_books = 66 :=
by
  sorry

end NUMINAMATH_GPT_books_remaining_in_special_collection_l1769_176980


namespace NUMINAMATH_GPT_investment_total_amount_l1769_176926

noncomputable def compoundedInvestment (principal : ℝ) (rate : ℝ) (tax : ℝ) (years : ℕ) : ℝ :=
let yearlyNetInterest := principal * rate * (1 - tax)
let rec calculate (year : ℕ) (accumulated : ℝ) : ℝ :=
  if year = 0 then accumulated else
    let newPrincipal := accumulated + yearlyNetInterest
    calculate (year - 1) newPrincipal
calculate years principal

theorem investment_total_amount :
  let finalAmount := compoundedInvestment 15000 0.05 0.10 4
  round finalAmount = 17607 :=
by
  sorry

end NUMINAMATH_GPT_investment_total_amount_l1769_176926


namespace NUMINAMATH_GPT_percentage_decrease_of_b_l1769_176918

variables (a b x m : ℝ) (p : ℝ)

-- Given conditions
def ratio_ab : Prop := a / b = 4 / 5
def expression_x : Prop := x = 1.25 * a
def expression_m : Prop := m = b * (1 - p / 100)
def ratio_mx : Prop := m / x = 0.6

-- The theorem to be proved
theorem percentage_decrease_of_b 
  (h1 : ratio_ab a b)
  (h2 : expression_x a x)
  (h3 : expression_m b m p)
  (h4 : ratio_mx m x) 
  : p = 40 :=
sorry

end NUMINAMATH_GPT_percentage_decrease_of_b_l1769_176918


namespace NUMINAMATH_GPT_total_mission_days_l1769_176931

variable (initial_days_first_mission : ℝ := 5)
variable (percentage_longer : ℝ := 0.60)
variable (days_second_mission : ℝ := 3)

theorem total_mission_days : 
  let days_first_mission_extra := initial_days_first_mission * percentage_longer
  let total_days_first_mission := initial_days_first_mission + days_first_mission_extra
  (total_days_first_mission + days_second_mission) = 11 := by
  sorry

end NUMINAMATH_GPT_total_mission_days_l1769_176931


namespace NUMINAMATH_GPT_percentage_students_school_A_l1769_176951

theorem percentage_students_school_A
  (A B : ℝ)
  (h1 : A + B = 100)
  (h2 : 0.30 * A + 0.40 * B = 34) :
  A = 60 :=
sorry

end NUMINAMATH_GPT_percentage_students_school_A_l1769_176951


namespace NUMINAMATH_GPT_total_caps_produced_l1769_176959

-- Define the production of each week as given in the conditions.
def week1_caps : ℕ := 320
def week2_caps : ℕ := 400
def week3_caps : ℕ := 300

-- Define the average of the first three weeks.
def average_caps : ℕ := (week1_caps + week2_caps + week3_caps) / 3

-- Define the production increase for the fourth week.
def increase_caps : ℕ := average_caps / 5  -- 20% is equivalent to dividing by 5

-- Calculate the total production for the fourth week (including the increase).
def week4_caps : ℕ := average_caps + increase_caps

-- Calculate the total number of caps produced in four weeks.
def total_caps : ℕ := week1_caps + week2_caps + week3_caps + week4_caps

-- Theorem stating the total production over the four weeks.
theorem total_caps_produced : total_caps = 1428 := by sorry

end NUMINAMATH_GPT_total_caps_produced_l1769_176959


namespace NUMINAMATH_GPT_unique_solution_condition_l1769_176944

theorem unique_solution_condition (p q : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by
  sorry

end NUMINAMATH_GPT_unique_solution_condition_l1769_176944


namespace NUMINAMATH_GPT_ramsey_theorem_six_people_l1769_176981

theorem ramsey_theorem_six_people (S : Finset Person)
  (hS: S.card = 6)
  (R : Person → Person → Prop): 
  (∃ (has_relation : Person → Person → Prop), 
    ∀ A B : Person, A ≠ B → R A B ∨ ¬ R A B) →
  (∃ (T : Finset Person), T.card = 3 ∧ 
    ((∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → R x y) ∨ 
     (∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → ¬ R x y))) :=
by
  sorry

end NUMINAMATH_GPT_ramsey_theorem_six_people_l1769_176981


namespace NUMINAMATH_GPT_impossible_transformation_l1769_176965

def f (x : ℝ) := x^2 + 5 * x + 4
def g (x : ℝ) := x^2 + 10 * x + 8

theorem impossible_transformation :
  (∀ x, f (x) = x^2 + 5 * x + 4) →
  (∀ x, g (x) = x^2 + 10 * x + 8) →
  (¬ ∃ t : ℝ → ℝ → ℝ, (∀ x, t (f x) x = g x)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_transformation_l1769_176965


namespace NUMINAMATH_GPT_complement_of_A_in_U_l1769_176978

def U : Set ℤ := {x | -2 ≤ x ∧ x ≤ 6}
def A : Set ℤ := {x | ∃ n : ℕ, (x = 2 * n ∧ n ≤ 3)}

theorem complement_of_A_in_U : (U \ A) = {-2, -1, 1, 3, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_A_in_U_l1769_176978


namespace NUMINAMATH_GPT_f_neg_one_l1769_176952

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1/x else - (x^2 + 1/(-x))

theorem f_neg_one : f (-1) = -2 :=
by
  -- This is where the proof would go, but it is left as a sorry
  sorry

end NUMINAMATH_GPT_f_neg_one_l1769_176952


namespace NUMINAMATH_GPT_circles_internally_tangent_l1769_176960

theorem circles_internally_tangent :
  let C1 := (3, -2)
  let r1 := 1
  let C2 := (7, 1)
  let r2 := 6
  let d := Real.sqrt (((7 - 3)^2 + (1 - (-2))^2) : ℝ)
  d = r2 - r1 :=
by
  sorry

end NUMINAMATH_GPT_circles_internally_tangent_l1769_176960


namespace NUMINAMATH_GPT_odd_function_expression_l1769_176991

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^2 - 2*x else -((-x)^2 - 2*(-x))

theorem odd_function_expression (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_pos : ∀ x : ℝ, 0 ≤ x → f x = x^2 - 2*x) :
  ∀ x : ℝ, f x = x * (|x| - 2) :=
by
  sorry

end NUMINAMATH_GPT_odd_function_expression_l1769_176991


namespace NUMINAMATH_GPT_A_inter_B_empty_iff_l1769_176903

variable (m : ℝ)

def A : Set ℝ := {x | x^2 - 3 * x - 10 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem A_inter_B_empty_iff : A ∩ B m = ∅ ↔ m < 2 ∨ m > 4 := by
  sorry

end NUMINAMATH_GPT_A_inter_B_empty_iff_l1769_176903


namespace NUMINAMATH_GPT_P_not_77_for_all_integers_l1769_176913

def P (x y : ℤ) : ℤ := x^5 - 4 * x^4 * y - 5 * y^2 * x^3 + 20 * y^3 * x^2 + 4 * y^4 * x - 16 * y^5

theorem P_not_77_for_all_integers (x y : ℤ) : P x y ≠ 77 :=
sorry

end NUMINAMATH_GPT_P_not_77_for_all_integers_l1769_176913


namespace NUMINAMATH_GPT_jason_earnings_l1769_176936

theorem jason_earnings :
  let fred_initial := 49
  let jason_initial := 3
  let emily_initial := 25
  let fred_increase := 1.5 
  let jason_increase := 0.625 
  let emily_increase := 0.40 
  let fred_new := fred_initial * fred_increase
  let jason_new := jason_initial * (1 + jason_increase)
  let emily_new := emily_initial * (1 + emily_increase)
  fred_new = fred_initial * fred_increase ->
  jason_new = jason_initial * (1 + jason_increase) ->
  emily_new = emily_initial * (1 + emily_increase) ->
  jason_new - jason_initial == 1.875 :=
by
  intros
  sorry

end NUMINAMATH_GPT_jason_earnings_l1769_176936


namespace NUMINAMATH_GPT_bet_strategy_possible_l1769_176939

def betting_possibility : Prop :=
  (1 / 6 + 1 / 2 + 1 / 9 + 1 / 8 <= 1)

theorem bet_strategy_possible : betting_possibility :=
by
  -- Proof is intentionally omitted
  sorry

end NUMINAMATH_GPT_bet_strategy_possible_l1769_176939


namespace NUMINAMATH_GPT_inscribed_triangle_area_l1769_176973

noncomputable def triangle_area (r : ℝ) (A B C : ℝ) : ℝ :=
  (1 / 2) * r^2 * (Real.sin A + Real.sin B + Real.sin C)

theorem inscribed_triangle_area :
  ∀ (r : ℝ), r = 12 / Real.pi →
  ∀ (A B C : ℝ), A = 40 * Real.pi / 180 → B = 80 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  triangle_area r A B C = 359.4384 / Real.pi^2 :=
by
  intros
  unfold triangle_area
  sorry

end NUMINAMATH_GPT_inscribed_triangle_area_l1769_176973


namespace NUMINAMATH_GPT_machine_a_produces_18_sprockets_per_hour_l1769_176942

theorem machine_a_produces_18_sprockets_per_hour :
  ∃ (A : ℝ), (∀ (B C : ℝ),
  B = 1.10 * A ∧
  B = 1.20 * C ∧
  990 / A = 990 / B + 10 ∧
  990 / C = 990 / A - 5) →
  A = 18 :=
by { sorry }

end NUMINAMATH_GPT_machine_a_produces_18_sprockets_per_hour_l1769_176942


namespace NUMINAMATH_GPT_triangle_height_l1769_176907

theorem triangle_height (area base : ℝ) (h_area : area = 9.31) (h_base : base = 4.9) : (2 * area) / base = 3.8 :=
by
  sorry

end NUMINAMATH_GPT_triangle_height_l1769_176907


namespace NUMINAMATH_GPT_nathan_correct_answers_l1769_176915

theorem nathan_correct_answers (c w : ℤ) (h1 : c + w = 15) (h2 : 6 * c - 3 * w = 45) : c = 10 := 
by sorry

end NUMINAMATH_GPT_nathan_correct_answers_l1769_176915


namespace NUMINAMATH_GPT_group_sizes_correct_l1769_176987

-- Define the number of fruits and groups
def num_bananas : Nat := 527
def num_oranges : Nat := 386
def num_apples : Nat := 319

def groups_bananas : Nat := 11
def groups_oranges : Nat := 103
def groups_apples : Nat := 17

-- Define the expected sizes of each group
def bananas_per_group : Nat := 47
def oranges_per_group : Nat := 3
def apples_per_group : Nat := 18

-- Prove the sizes of the groups are as expected
theorem group_sizes_correct :
  (num_bananas / groups_bananas = bananas_per_group) ∧
  (num_oranges / groups_oranges = oranges_per_group) ∧
  (num_apples / groups_apples = apples_per_group) :=
by
  -- Division in Nat rounds down
  have h1 : num_bananas / groups_bananas = 47 := by sorry
  have h2 : num_oranges / groups_oranges = 3 := by sorry
  have h3 : num_apples / groups_apples = 18 := by sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_group_sizes_correct_l1769_176987


namespace NUMINAMATH_GPT_rectangle_perimeter_126_l1769_176909

/-- Define the sides of the rectangle in terms of a common multiplier -/
def sides (x : ℝ) : ℝ × ℝ := (4 * x, 3 * x)

/-- Define the area of the rectangle given the common multiplier -/
def area (x : ℝ) : ℝ := (4 * x) * (3 * x)

example : ∃ (x : ℝ), area x = 972 :=
by
  sorry

/-- Calculate the perimeter of the rectangle given the common multiplier -/
def perimeter (x : ℝ) : ℝ := 2 * ((4 * x) + (3 * x))

/-- The final proof statement, stating that the perimeter of the rectangle is 126 meters,
    given the ratio of its sides and its area. -/
theorem rectangle_perimeter_126 (x : ℝ) (h: area x = 972) : perimeter x = 126 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_perimeter_126_l1769_176909


namespace NUMINAMATH_GPT_distance_by_which_A_beats_B_l1769_176976

noncomputable def speed_of_A : ℝ := 1000 / 192
noncomputable def time_difference : ℝ := 8
noncomputable def distance_beaten : ℝ := speed_of_A * time_difference

theorem distance_by_which_A_beats_B :
  distance_beaten = 41.67 := by
  sorry

end NUMINAMATH_GPT_distance_by_which_A_beats_B_l1769_176976


namespace NUMINAMATH_GPT_Brenda_mice_left_l1769_176912

theorem Brenda_mice_left :
  ∀ (total_litters total_each sixth factor remaining : ℕ),
    total_litters = 3 → 
    total_each = 8 →
    sixth = total_litters * total_each / 6 →
    factor = 3 * (total_litters * total_each / 6) →
    remaining = total_litters * total_each - sixth - factor →
    remaining / 2 = ((total_litters * total_each - sixth - factor) / 2) →
    total_litters * total_each - sixth - factor - ((total_litters * total_each - sixth - factor) / 2) = 4 :=
by
  intros total_litters total_each sixth factor remaining h_litters h_each h_sixth h_factor h_remaining h_half
  sorry

end NUMINAMATH_GPT_Brenda_mice_left_l1769_176912


namespace NUMINAMATH_GPT_solve_abs_eq_l1769_176904

theorem solve_abs_eq (x : ℝ) : (|x + 4| = 3 - x) → (x = -1/2) := by
  intro h
  sorry

end NUMINAMATH_GPT_solve_abs_eq_l1769_176904


namespace NUMINAMATH_GPT_evaluate_fraction_l1769_176923

theorem evaluate_fraction : (1 / (3 - 1 / (3 - 1 / (3 - 1 / 3)))) = (8 / 21) :=
by
  sorry

end NUMINAMATH_GPT_evaluate_fraction_l1769_176923


namespace NUMINAMATH_GPT_ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l1769_176998

theorem ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one
  (m n : ℕ) : (10 ^ m + 1) % (10 ^ n - 1) ≠ 0 := 
  sorry

end NUMINAMATH_GPT_ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l1769_176998


namespace NUMINAMATH_GPT_fraction_equation_solution_l1769_176988

theorem fraction_equation_solution (x : ℝ) (h : x ≠ 3) : (2 - x) / (x - 3) + 3 = 2 / (3 - x) ↔ x = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_fraction_equation_solution_l1769_176988


namespace NUMINAMATH_GPT_diameter_outer_boundary_correct_l1769_176930

noncomputable def diameter_outer_boundary 
  (D_fountain : ℝ)
  (w_gardenRing : ℝ)
  (w_innerPath : ℝ)
  (w_outerPath : ℝ) : ℝ :=
  let R_fountain := D_fountain / 2
  let R_innerPath := R_fountain + w_gardenRing
  let R_outerPathInner := R_innerPath + w_innerPath
  let R_outerPathOuter := R_outerPathInner + w_outerPath
  2 * R_outerPathOuter

theorem diameter_outer_boundary_correct :
  diameter_outer_boundary 10 12 3 4 = 48 := by
  -- skipping proof
  sorry

end NUMINAMATH_GPT_diameter_outer_boundary_correct_l1769_176930


namespace NUMINAMATH_GPT_original_number_is_85_l1769_176905

theorem original_number_is_85
  (x : ℤ) (h_sum : 10 ≤ x ∧ x < 100) 
  (h_condition1 : (x / 10) + (x % 10) = 13)
  (h_condition2 : 10 * (x % 10) + (x / 10) = x - 27) :
  x = 85 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_85_l1769_176905


namespace NUMINAMATH_GPT_tangent_and_normal_lines_l1769_176938

theorem tangent_and_normal_lines (x y : ℝ → ℝ) (t : ℝ) (t₀ : ℝ) 
  (h0 : t₀ = 0) 
  (h1 : ∀ t, x t = (1/2) * t^2 - (1/4) * t^4) 
  (h2 : ∀ t, y t = (1/2) * t^2 + (1/3) * t^3) :
  (∃ m : ℝ, y (x t₀) = m * (x t₀) ∧ m = 1) ∧
  (∃ n : ℝ, y (x t₀) = n * (x t₀) ∧ n = -1) :=
by 
  sorry

end NUMINAMATH_GPT_tangent_and_normal_lines_l1769_176938


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l1769_176911

-- Definition of the proposition p
def prop_p (m : ℝ) := ∀ x : ℝ, x^2 - 4 * x + 2 * m ≥ 0

-- Statement of the proof problem
theorem sufficient_not_necessary_condition (m : ℝ) : 
  (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 → m ≥ 2) ∧ (m ≥ 2 → prop_p m) → (m ≥ 3 → prop_p m) ∧ ¬(m ≥ 3 ↔ prop_p m) :=
sorry

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l1769_176911


namespace NUMINAMATH_GPT_increase_80_by_50_percent_l1769_176975

theorem increase_80_by_50_percent : 
  let original_number := 80
  let percentage_increase := 0.5
  let increase := original_number * percentage_increase
  let final_number := original_number + increase
  final_number = 120 := 
by 
  sorry

end NUMINAMATH_GPT_increase_80_by_50_percent_l1769_176975


namespace NUMINAMATH_GPT_range_of_a_l1769_176949

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then (1 - 2 * a) * x + 3 * a else Real.log x

theorem range_of_a (a : ℝ) : (-1 ≤ a ∧ a < 1/2) ↔
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1769_176949


namespace NUMINAMATH_GPT_stuffed_animal_cost_l1769_176950

theorem stuffed_animal_cost
  (M S A C : ℝ)
  (h1 : M = 3 * S)
  (h2 : M = (1/2) * A)
  (h3 : C = (1/2) * A)
  (h4 : C = 2 * S)
  (h5 : M = 6) :
  A = 8 :=
by
  sorry

end NUMINAMATH_GPT_stuffed_animal_cost_l1769_176950


namespace NUMINAMATH_GPT_logarithm_argument_positive_l1769_176961

open Real

theorem logarithm_argument_positive (a : ℝ) : 
  (∀ x : ℝ, sin x ^ 6 + cos x ^ 6 + a * sin x * cos x > 0) ↔ -1 / 2 < a ∧ a < 1 / 2 :=
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_logarithm_argument_positive_l1769_176961


namespace NUMINAMATH_GPT_solve_equation_l1769_176937

theorem solve_equation (x : ℝ) (h : x > 0) :
  25^(Real.log x / Real.log 4) - 5^(Real.log (x^2) / Real.log 16 + 1) = Real.log (9 * Real.sqrt 3) / Real.log (Real.sqrt 3) - 25^(Real.log x / Real.log 16) ->
  x = 4 :=
by
  sorry

end NUMINAMATH_GPT_solve_equation_l1769_176937


namespace NUMINAMATH_GPT_sliding_window_sash_translation_l1769_176919

def is_translation (movement : Type) : Prop := sorry

def ping_pong_ball_movement : Type := sorry
def sliding_window_sash_movement : Type := sorry
def kite_flight_movement : Type := sorry
def basketball_movement : Type := sorry

axiom ping_pong_not_translation : ¬ is_translation ping_pong_ball_movement
axiom kite_not_translation : ¬ is_translation kite_flight_movement
axiom basketball_not_translation : ¬ is_translation basketball_movement
axiom window_sash_is_translation : is_translation sliding_window_sash_movement

theorem sliding_window_sash_translation :
  is_translation sliding_window_sash_movement :=
by 
  exact window_sash_is_translation

end NUMINAMATH_GPT_sliding_window_sash_translation_l1769_176919


namespace NUMINAMATH_GPT_required_HCl_moles_l1769_176977

-- Definitions of chemical substances:
def HCl: Type := Unit
def NaHCO3: Type := Unit
def H2O: Type := Unit
def CO2: Type := Unit
def NaCl: Type := Unit

-- The reaction as a balanced chemical equation:
def balanced_eq (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl) : Prop :=
  ∃ (m: ℕ), m = 1

-- Given conditions:
def condition1: Prop := balanced_eq () () () () ()
def condition2 (moles_H2O moles_CO2 moles_NaCl: ℕ): Prop :=
  moles_H2O = moles_CO2 ∧ moles_CO2 = moles_NaCl ∧ moles_NaCl = moles_H2O

def condition3: ℕ := 3  -- moles of NaHCO3

-- The theorem statement:
theorem required_HCl_moles (moles_HCl moles_NaHCO3: ℕ)
  (hcl: HCl) (nahco3: NaHCO3) (h2o: H2O) (co2: CO2) (nacl: NaCl)
  (balanced: balanced_eq hcl nahco3 h2o co2 nacl)
  (equal_moles: condition2 moles_H2O moles_CO2 moles_NaCl)
  (nahco3_eq_3: moles_NaHCO3 = condition3):
  moles_HCl = 3 :=
sorry

end NUMINAMATH_GPT_required_HCl_moles_l1769_176977


namespace NUMINAMATH_GPT_person_age_l1769_176925

theorem person_age (A : ℕ) (h : 6 * (A + 6) - 6 * (A - 6) = A) : A = 72 := 
by
  sorry

end NUMINAMATH_GPT_person_age_l1769_176925


namespace NUMINAMATH_GPT_Caroline_lost_4_pairs_of_socks_l1769_176968

theorem Caroline_lost_4_pairs_of_socks 
  (initial_pairs : ℕ) (pairs_donated_fraction : ℚ)
  (new_pairs_purchased : ℕ) (new_pairs_gifted : ℕ)
  (final_pairs : ℕ) (L : ℕ) :
  initial_pairs = 40 →
  pairs_donated_fraction = 2/3 →
  new_pairs_purchased = 10 →
  new_pairs_gifted = 3 →
  final_pairs = 25 →
  (initial_pairs - L) * (1 - pairs_donated_fraction) + new_pairs_purchased + new_pairs_gifted = final_pairs →
  L = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_Caroline_lost_4_pairs_of_socks_l1769_176968


namespace NUMINAMATH_GPT_sum_of_exterior_segment_angles_is_540_l1769_176992

-- Define the setup of the problem
def quadrilateral_inscribed_in_circle (A B C D : Type) : Prop := sorry
def angle_externally_inscribed (segment : Type) : ℝ := sorry

-- Main theorem statement
theorem sum_of_exterior_segment_angles_is_540
  (A B C D : Type)
  (h_quad : quadrilateral_inscribed_in_circle A B C D)
  (alpha beta gamma delta : ℝ)
  (h_alpha : alpha = angle_externally_inscribed A)
  (h_beta : beta = angle_externally_inscribed B)
  (h_gamma : gamma = angle_externally_inscribed C)
  (h_delta : delta = angle_externally_inscribed D) :
  alpha + beta + gamma + delta = 540 :=
sorry

end NUMINAMATH_GPT_sum_of_exterior_segment_angles_is_540_l1769_176992


namespace NUMINAMATH_GPT_clock_angle_4_oclock_l1769_176972

theorem clock_angle_4_oclock :
  let total_degrees := 360
  let hours := 12
  let degree_per_hour := total_degrees / hours
  let hour_position := 4
  let minute_hand_position := 0
  let hour_hand_angle := hour_position * degree_per_hour
  hour_hand_angle = 120 := sorry

end NUMINAMATH_GPT_clock_angle_4_oclock_l1769_176972


namespace NUMINAMATH_GPT_positive_difference_l1769_176901

theorem positive_difference (a b : ℕ) (h1 : a = (6^2 + 6^2) / 6) (h2 : b = (6^2 * 6^2) / 6) : a < b ∧ b - a = 204 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_l1769_176901


namespace NUMINAMATH_GPT_sequence_general_term_l1769_176967

open Nat

/-- Define the sequence recursively -/
def a : ℕ → ℤ
| 0     => -1
| (n+1) => 3 * a n - 1

/-- The general term of the sequence is given by - (3^n - 1) / 2 -/
theorem sequence_general_term (n : ℕ) : a n = - (3^n - 1) / 2 := 
by
  sorry

end NUMINAMATH_GPT_sequence_general_term_l1769_176967


namespace NUMINAMATH_GPT_value_of_a_l1769_176957

theorem value_of_a :
  ∀ (g : ℝ → ℝ), (∀ x, g x = 5*x - 7) → ∃ a, g a = 0 ∧ a = 7 / 5 :=
by
  sorry

end NUMINAMATH_GPT_value_of_a_l1769_176957


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1769_176946

-- Define set A
def A : Set ℤ := {-1, 0, 1, 2, 3, 4, 5}

-- Define set B
def B : Set ℤ := {2, 4, 6, 8}

-- Prove that the intersection of set A and set B is {2, 4}.
theorem intersection_of_A_and_B : A ∩ B = {2, 4} :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1769_176946


namespace NUMINAMATH_GPT_second_train_speed_is_correct_l1769_176996

noncomputable def speed_of_second_train (length_first : ℝ) (speed_first : ℝ) (time_cross : ℝ) (length_second : ℝ) : ℝ :=
let total_distance := length_first + length_second
let relative_speed := total_distance / time_cross
let relative_speed_kmph := relative_speed * 3.6
relative_speed_kmph - speed_first

theorem second_train_speed_is_correct :
  speed_of_second_train 270 120 9 230.04 = 80.016 :=
by
  sorry

end NUMINAMATH_GPT_second_train_speed_is_correct_l1769_176996


namespace NUMINAMATH_GPT_max_value_of_abs_z_plus_4_l1769_176924

open Complex
noncomputable def max_abs_z_plus_4 {z : ℂ} (h : abs (z + 3 * I) = 5) : ℝ :=
sorry

theorem max_value_of_abs_z_plus_4 (z : ℂ) (h : abs (z + 3 * I) = 5) : abs (z + 4) ≤ 10 :=
sorry

end NUMINAMATH_GPT_max_value_of_abs_z_plus_4_l1769_176924


namespace NUMINAMATH_GPT_probability_odd_3_in_6_rolls_l1769_176956

-- Definitions based on problem conditions
def probability_of_odd (outcome: ℕ) : ℚ := if outcome % 2 = 1 then 1/2 else 0 

def binomial_probability (n k : ℕ) (p : ℚ) : ℚ := 
  ((Nat.choose n k : ℚ) * (p^k) * ((1 - p)^(n - k)))

-- Given problem
theorem probability_odd_3_in_6_rolls : 
  binomial_probability 6 3 (1/2) = 5 / 16 :=
by
  sorry

end NUMINAMATH_GPT_probability_odd_3_in_6_rolls_l1769_176956


namespace NUMINAMATH_GPT_commute_time_variance_l1769_176969

theorem commute_time_variance
  (x y : ℝ)
  (h1 : x + y = 20)
  (h2 : (x - 10)^2 + (y - 10)^2 = 8) :
  x^2 + y^2 = 208 :=
by
  sorry

end NUMINAMATH_GPT_commute_time_variance_l1769_176969


namespace NUMINAMATH_GPT_division_equals_fraction_l1769_176945

theorem division_equals_fraction:
  180 / (8 + 9 * 3 - 4) = 180 / 31 := 
by
  sorry

end NUMINAMATH_GPT_division_equals_fraction_l1769_176945


namespace NUMINAMATH_GPT_example_problem_l1769_176953

-- Definitions and conditions derived from the original problem statement
def smallest_integer_with_two_divisors (m : ℕ) : Prop := m = 2
def second_largest_integer_with_three_divisors_less_than_100 (n : ℕ) : Prop := n = 25

theorem example_problem (m n : ℕ) 
    (h1 : smallest_integer_with_two_divisors m) 
    (h2 : second_largest_integer_with_three_divisors_less_than_100 n) : 
    m + n = 27 :=
by sorry

end NUMINAMATH_GPT_example_problem_l1769_176953


namespace NUMINAMATH_GPT_coprime_pairs_solution_l1769_176964

theorem coprime_pairs_solution (x y : ℕ) (hx : x ∣ y^2 + 210) (hy : y ∣ x^2 + 210) (hxy : Nat.gcd x y = 1) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
by sorry

end NUMINAMATH_GPT_coprime_pairs_solution_l1769_176964


namespace NUMINAMATH_GPT_more_knights_than_liars_l1769_176914

theorem more_knights_than_liars 
  (k l : Nat)
  (h1 : (k + l) % 2 = 1)
  (h2 : ∀ i : Nat, i < k → ∃ j : Nat, j < l)
  (h3 : ∀ j : Nat, j < l → ∃ i : Nat, i < k) :
  k > l := 
sorry

end NUMINAMATH_GPT_more_knights_than_liars_l1769_176914


namespace NUMINAMATH_GPT_comb_15_6_eq_5005_perm_6_eq_720_l1769_176990

open Nat

-- Prove that \frac{15!}{6!(15-6)!} = 5005
theorem comb_15_6_eq_5005 : (factorial 15) / (factorial 6 * factorial (15 - 6)) = 5005 := by
  sorry

-- Prove that the number of ways to arrange 6 items in a row is 720
theorem perm_6_eq_720 : factorial 6 = 720 := by
  sorry

end NUMINAMATH_GPT_comb_15_6_eq_5005_perm_6_eq_720_l1769_176990


namespace NUMINAMATH_GPT_cuboid_edge_length_l1769_176986

-- This is the main statement we want to prove
theorem cuboid_edge_length (L : ℝ) (w : ℝ) (h : ℝ) (V : ℝ) (w_eq : w = 5) (h_eq : h = 3) (V_eq : V = 30) :
  V = L * w * h → L = 2 :=
by
  -- Adding the sorry allows us to compile and acknowledge the current placeholder for the proof.
  sorry

end NUMINAMATH_GPT_cuboid_edge_length_l1769_176986


namespace NUMINAMATH_GPT_sum_squares_l1769_176993

theorem sum_squares (w x y z : ℝ) (h1 : w + x + y + z = 0) (h2 : w^2 + x^2 + y^2 + z^2 = 1) :
  -1 ≤ w * x + x * y + y * z + z * w ∧ w * x + x * y + y * z + z * w ≤ 0 := 
by 
  sorry

end NUMINAMATH_GPT_sum_squares_l1769_176993


namespace NUMINAMATH_GPT_complement_union_l1769_176928

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union : (U \ (A ∪ B)) = {1, 2, 6} :=
by simp only [U, A, B, Set.mem_union, Set.mem_compl, Set.mem_diff];
   sorry

end NUMINAMATH_GPT_complement_union_l1769_176928


namespace NUMINAMATH_GPT_find_Sum_4n_l1769_176920

variable {a : ℕ → ℕ} -- Define a sequence a_n

-- Define our conditions about the sums Sn and S3n
axiom Sum_n : ℕ → ℕ 
axiom Sum_3n : ℕ → ℕ 
axiom Sum_4n : ℕ → ℕ 

def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) - a n = d

noncomputable def arithmetic_sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (n * (a n + a 0)) / 2

axiom h1 : is_arithmetic_sequence a
axiom h2 : Sum_n 1 = 2
axiom h3 : Sum_3n 3 = 12

theorem find_Sum_4n : Sum_4n 4 = 20 :=
sorry

end NUMINAMATH_GPT_find_Sum_4n_l1769_176920


namespace NUMINAMATH_GPT_train_length_calculation_l1769_176984

noncomputable def train_length (speed_km_hr : ℕ) (time_sec : ℕ) : ℝ :=
  (speed_km_hr * 1000 / 3600) * time_sec

theorem train_length_calculation :
  train_length 250 6 = 416.67 :=
by
  sorry

end NUMINAMATH_GPT_train_length_calculation_l1769_176984


namespace NUMINAMATH_GPT_floor_area_cannot_exceed_10_square_meters_l1769_176955

theorem floor_area_cannot_exceed_10_square_meters
  (a b : ℝ)
  (h : 3 > 0)
  (floor_lt_wall1 : a * b < 3 * a)
  (floor_lt_wall2 : a * b < 3 * b) :
  a * b ≤ 9 :=
by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_floor_area_cannot_exceed_10_square_meters_l1769_176955


namespace NUMINAMATH_GPT_assembly_shortest_time_l1769_176927

-- Define the times taken for each assembly path
def time_ACD : ℕ := 3 + 4
def time_EDF : ℕ := 4 + 2

-- State the theorem for the shortest time required to assemble the product
theorem assembly_shortest_time : max time_ACD time_EDF + 4 = 13 :=
by {
  -- Introduction of the given conditions and simplified value calculation
  sorry
}

end NUMINAMATH_GPT_assembly_shortest_time_l1769_176927


namespace NUMINAMATH_GPT_find_d_l1769_176941

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (a₁ d : ℝ) : Prop :=
∀ n, a n = a₁ + d * (n - 1)

theorem find_d
  (a : ℕ → ℝ)
  (a₁ d : ℝ)
  (h_arith : arithmetic_sequence a a₁ d)
  (h₁ : a₁ = 1)
  (h_geom_mean : a 2 ^ 2 = a 1 * a 4)
  (h_d_neq_zero : d ≠ 0):
  d = 1 :=
sorry

end NUMINAMATH_GPT_find_d_l1769_176941


namespace NUMINAMATH_GPT_floor_ceiling_sum_l1769_176985

theorem floor_ceiling_sum : 
    Int.floor (0.998 : ℝ) + Int.ceil (2.002 : ℝ) = 3 := by
  sorry

end NUMINAMATH_GPT_floor_ceiling_sum_l1769_176985


namespace NUMINAMATH_GPT_multiply_polynomials_l1769_176994

theorem multiply_polynomials (x : ℝ) : 2 * x * (5 * x ^ 2) = 10 * x ^ 3 := by
  sorry

end NUMINAMATH_GPT_multiply_polynomials_l1769_176994


namespace NUMINAMATH_GPT_paul_runs_41_miles_l1769_176948

-- Conditions as Definitions
def movie1_length : ℕ := (1 * 60) + 36
def movie2_length : ℕ := (2 * 60) + 18
def movie3_length : ℕ := (1 * 60) + 48
def movie4_length : ℕ := (2 * 60) + 30
def total_watch_time : ℕ := movie1_length + movie2_length + movie3_length + movie4_length
def time_per_mile : ℕ := 12

-- Proof Statement
theorem paul_runs_41_miles : total_watch_time / time_per_mile = 41 :=
by
  -- Proof would be provided here
  sorry 

end NUMINAMATH_GPT_paul_runs_41_miles_l1769_176948
