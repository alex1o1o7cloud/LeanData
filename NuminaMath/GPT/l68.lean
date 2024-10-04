import Mathlib

namespace prism_intersection_volume_l68_68014

theorem prism_intersection_volume (a h : ℝ) (V1 V2 : ℝ) :
  let A1 B1 C1 := equilateral_triangle a, 
      S_ABC := (a^2 * real.sqrt 3) / 2,
      S_BMC := 2 / 3 * S_ABC,
      S_BN1K1 := 1 / 3 / 2 * S_ABC,
      S_np := S_BMC - S_BN1K1,
      cos_alpha := S_np / S_ABC,
      B_L := 2 * h,
      B_H := (2 / 3 * a) * (real.sqrt(3) * a / 2) / (real.sqrt((real.sqrt(3) * a / 2)^2 + (a / 6)^2)),
      sin_alpha := 2 * real.sqrt 2 / 3,
      tan_alpha := 2 * real.sqrt 2,
      BL_MCA := (a^2 * real.sqrt 3)/18 * B_L,
      V_LBMC := BL_MCA * B_L,
      V_A_M_C_A_N_K_C1 := 77 * real.sqrt 3 / 54,
  V1 = (49 * real.sqrt 3) / 54 ∧ V2 = (77 * real.sqrt 3) / 54
  := V_LBMC = V1 ∨ V_LBMC = V2

end prism_intersection_volume_l68_68014


namespace max_sum_plus_permutations_l68_68288

def max_sum_val (l : List ℕ) : ℕ :=
  match l with
  | [x1, x2, x3, x4, x5, x6] := x1 * x2 + x2 * x3 + x3 * x4 + x4 * x5 + x5 * x6 + x6 * x1
  | _ := 0

def max_permutations_value : ℕ := 81
def num_permutations_achieving_max : ℕ := 12

theorem max_sum_plus_permutations : 
  let M' := max_permutations_value,
      N' := num_permutations_achieving_max
  in M' + N' = 93 :=
by 
  sorry

end max_sum_plus_permutations_l68_68288


namespace solve_constants_l68_68639

theorem solve_constants (a b : ℚ) : 
  a • vector.ofFn [3, 4] + b • vector.ofFn [1, 6] = vector.ofFn [7, 45] →
  a = -3 / 14 ∧ b = 107 / 14 :=
sorry

end solve_constants_l68_68639


namespace fraction_subtraction_l68_68866

theorem fraction_subtraction :
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 :=
by
  sorry

end fraction_subtraction_l68_68866


namespace reasoning_incorrect_due_to_major_premise_l68_68553

-- Definitions of the conditions
def exponential_function (a : ℝ) (x : ℝ) : ℝ := a ^ x

def is_increasing_function {f : ℝ → ℝ} : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

-- The theorem to state that the reasoning is incorrect because the major premise is wrong.
theorem reasoning_incorrect_due_to_major_premise (a : ℝ) (ha : 0 < a) (a_ne_one : a ≠ 1) :
  ¬ (is_increasing_function (exponential_function a) ∧ (exponential_function a = exponential_function 2))
  :=
begin
  sorry,
end

end reasoning_incorrect_due_to_major_premise_l68_68553


namespace find_special_numbers_l68_68031

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end find_special_numbers_l68_68031


namespace base_of_1024_in_ABAB_form_l68_68530

theorem base_of_1024_in_ABAB_form :
  ∃ b : ℕ, (b ^ 3 ≤ 1024 ∧ 1024 < b ^ 4) ∧ (let n := 1024 in
  let q3 := n / b^3, rem3 := n % b^3 in
  let q2 := rem3 / b^2, rem2 := rem3 % b^2 in
  let q1 := rem2 / b, rem1 := rem2 % b in
  [q3, q2, q1, rem1] = [q3, q2, q3, q2] 
  ∧ q3 ≠ q2))
:=
begin
  existsi 6,
  split,
  { split,
    { norm_num, },
    { norm_num, } },
  { let n := 1024,
    let q3 := n / 6^3,
    let rem3 := n % 6^3,
    let q2 := rem3 / 6^2,
    let rem2 := rem3 % 6^2,
    let q1 := rem2 / 6,
    let rem1 := rem2 % 6,
    have : [q3, q2, q1, rem1] = [4, 4, 2, 4], by sorry,
    have :  4 ≠ 2, by norm_num,
    split; assumption, }
end

end base_of_1024_in_ABAB_form_l68_68530


namespace impossible_to_ensure_continuous_coverage_l68_68912

-- Definitions for conditions
def length_of_track : ℝ := 2  -- in kilometers
def arc_length_of_stands : ℝ := 100 / 1000  -- in kilometers (100 meters)
def number_of_runners : ℕ := 20
def speeds : List ℝ := List.range number_of_runners |>.map (λ n => 10 + n)  -- speeds from 10 to 29 km/h
def passing_time_fraction : ℝ := arc_length_of_stands / length_of_track  -- fraction of time passing stands

-- Proof problem setup
theorem impossible_to_ensure_continuous_coverage :
  ∃ (positions : List ℝ), (length positions = number_of_runners) ∧
  (∀ t : ℝ, ∃ i : ℕ, i < number_of_runners ∧ (positions.nth i).val + speeds.nth i.val * t % length_of_track < arc_length_of_stands) → False :=
by
  sorry

end impossible_to_ensure_continuous_coverage_l68_68912


namespace four_digit_palindrome_square_count_l68_68997

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68997


namespace sum_of_possible_k_values_l68_68239

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l68_68239


namespace Tn_lt_Sn_div_2_l68_68311

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68311


namespace Tn_lt_Sn_div_2_l68_68312

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68312


namespace four_digit_palindromic_perfect_square_l68_68954

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68954


namespace fifty_three_days_from_friday_is_tuesday_l68_68472

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68472


namespace trajectory_of_point_B_in_parallelogram_l68_68598

theorem trajectory_of_point_B_in_parallelogram :
  ∀ (A C : ℝ × ℝ) (D : ℝ × ℝ), 
    A = (-1, 3) → 
    C = (-3, 2) → 
    (x - 3 * y = 1) → 
    ∃ B : ℝ × ℝ, trajectory_point_B_eq : (B.1 - 3 * B.2 + 20 = 0) :=
by
  intros A C D hA hC hD
  sorry

end trajectory_of_point_B_in_parallelogram_l68_68598


namespace day_53_days_from_friday_l68_68446

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68446


namespace four_digit_perfect_square_palindrome_count_l68_68971

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68971


namespace pow_one_pow_greater_than_one_pow_less_than_one_l68_68398

theorem pow_one (n : ℤ) : (1 : ℝ) ^ n = 1 := 
sorry

theorem pow_greater_than_one (α : ℝ) (hα : α > 0) : 
   filter.tendsto (λ n : ℕ, (1 + α) ^ n) filter.at_top filter.at_top :=
sorry

theorem pow_less_than_one (α : ℝ) (hα : 0 < α ∧ α < 1) : 
   filter.tendsto (λ n : ℕ, (1 - α) ^ n) filter.at_top (nhds 0) :=
sorry

end pow_one_pow_greater_than_one_pow_less_than_one_l68_68398


namespace find_f_value_l68_68168

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x^α

theorem find_f_value (α : ℝ) (h : f 3 α = Real.sqrt 3) : f (1 / 4) α = 1 / 2 :=
by
  sorry

end find_f_value_l68_68168


namespace solve_for_r_l68_68195

noncomputable def k (r : ℝ) : ℝ := 5 / (2 ^ r)

theorem solve_for_r (r : ℝ) :
  (5 = k r * 2 ^ r) ∧ (45 = k r * 8 ^ r) → r = (Real.log 9 / Real.log 2) / 2 :=
by
  intro h
  sorry

end solve_for_r_l68_68195


namespace find_b_age_l68_68008

variable (a b c : ℕ)
-- Condition 1: a is two years older than b
variable (h1 : a = b + 2)
-- Condition 2: b is twice as old as c
variable (h2 : b = 2 * c)
-- Condition 3: The total of the ages of a, b, and c is 17
variable (h3 : a + b + c = 17)

theorem find_b_age (a b c : ℕ) (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 17) : b = 6 :=
by
  sorry

end find_b_age_l68_68008


namespace clearance_sale_gain_percent_l68_68010

/-- Given conditions -/
variables (SP CP MP Discount SP_sale Gain GainPercent : ℝ)
variables (h1 : SP = 30) (h2 : SP = CP + 0.20 * CP) (h3 : Discount = 0.10 * MP)
variables (h4 : MP = 30) (h5 : SP_sale = MP - Discount)
variables (h6 : Gain = SP_sale - CP) (h7 : GainPercent = (Gain / CP) * 100)

/-- Prove the shopkeeper's gain percent during the clearance sale is 8% -/
theorem clearance_sale_gain_percent : GainPercent = 8 :=
by sorry

end clearance_sale_gain_percent_l68_68010


namespace least_n_squares_l68_68683

open Nat

theorem least_n_squares (p : ℕ) (hp : Prime p) (hp_gt_two : p > 2) :
  ∃ n_least : ℕ, (∀ (s : Finset ℕ), (∀ x ∈ s, ¬ p ∣ x^2) → (∃ t ⊆ s, ∏ x in t, x^2 ≡ 1 [MOD p]) ↔ n_least = (p - 1) / 2) :=
by
  sorry

end least_n_squares_l68_68683


namespace Tn_lt_half_Sn_l68_68345

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68345


namespace parallel_lines_a_perpendicular_lines_a_l68_68188

section ParallelAndPerpendicularLines

variables {a : ℝ}

-- Define lines l₁ and l₂
def line1 (a : ℝ) : Prop := ∀ x y : ℝ, x + (1 + a) * y + a - 1 = 0
def line2 (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 2 * y + 6 = 0

-- Define slope functions
def slope_l1 (a : ℝ) : ℝ := -1 / (1 + a)
def slope_l2 (a : ℝ) : ℝ := -a / 2

-- Requirement: If the lines are parallel, slopes are equal
def are_parallel (a : ℝ) : Prop := slope_l1 a = slope_l2 a

-- Requirement: If the lines are perpendicular, product of slopes is -1
def are_perpendicular (a : ℝ) : Prop := slope_l1 a * slope_l2 a = -1

-- Proof statements
theorem parallel_lines_a : are_parallel a → a = 1 := by
  sorry

theorem perpendicular_lines_a : are_perpendicular a → a = -2 / 3 := by
  sorry

end ParallelAndPerpendicularLines

end parallel_lines_a_perpendicular_lines_a_l68_68188


namespace T_lt_S_div_2_l68_68339

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68339


namespace general_form_of_line_l_standard_equation_of_circle_C_l68_68693

/--
Given points A (-3, -1) and B (5, 5),
(Ⅰ) Find the equation in general form of the line l that passes through point A and is perpendicular to line AB.
(Ⅱ) Find the standard equation of the circle C with diameter AB.
--/

def point : Type := (ℤ × ℤ)

def points_A_B := (A : point) (B : point), A = (-3, -1) ∧ B = (5, 5)

noncomputable def slope_AB (A B : point) : ℚ :=
  let (x1, y1) := A
  let (x2, y2) := B
  (y2 - y1) / (x2 - x1)

noncomputable def slope_l (k_AB : ℚ) : ℚ :=
  -1 / k_AB

theorem general_form_of_line_l (A B : point) (h: points_A_B A B) :
  let k_l := slope_l (slope_AB A B)
  ∃ x y : ℤ, let y := k_l * (x + 3) + (-1) in 4 * x + 3 * y + 15 = 0 :=
by
  sorry

noncomputable def center_C (A B : point) : point :=
  let (x1, y1) := A
  let (x2, y2) := B
  ((x1 + x2) / 2, (y1 + y2) / 2)

noncomputable def distance (p1 p2 : point) : ℚ :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  (Math.sqrt ((x2 - x1)^2 + (y2 - y1)^2))

theorem standard_equation_of_circle_C (A B : point) (h: points_A_B A B) :
  let C := center_C A B
  let r := distance A C
  ∃ x y : ℤ, (x - fst C)² + (y - snd C)² = r :=
by
  sorry

end general_form_of_line_l_standard_equation_of_circle_C_l68_68693


namespace abs_expression_equality_l68_68627

def pi : ℝ := Real.pi

theorem abs_expression_equality : abs (2 * pi - abs (pi - 9)) = 3 * pi - 9 := by
  sorry

end abs_expression_equality_l68_68627


namespace day_53_days_from_friday_l68_68441

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68441


namespace max_value_of_sum_l68_68689

theorem max_value_of_sum
  (n : ℕ) (h : n ≥ 3)
  (A : Fin (2 * n) → Set (Fin n))
  (A_nonempty : ∀ i, i < (2 * n) → (A i).Nonempty)
  (A_distinct : ∀ i j, i < (2 * n) → j < (2 * n) → i ≠ j → A i ≠ A j)
  (A_eq : A (2 * n + 1 - 1) = A 0) : 
  ∃ (max_val : ℝ), max_val = n :=
begin
  sorry,
end

end max_value_of_sum_l68_68689


namespace quadratic_inequality_solution_l68_68099

theorem quadratic_inequality_solution (k : ℝ) :
  (∀ x : ℝ, x^2 - (k - 4) * x - k + 8 > 0) ↔ -8/3 < k ∧ k < 6 :=
by
  sorry

end quadratic_inequality_solution_l68_68099


namespace train_travel_distance_l68_68575

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l68_68575


namespace eleventh_number_is_149_l68_68050

def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |>.sum

def eleventh_number_in_list :=
  { n : Nat // sum_of_digits n = 14 ∧ n > 0 } |>.fintype
  { x // x ≤ ∅.eleven }

-- The statement specifying the problem to be proven
theorem eleventh_number_is_149 :
  (list.fin_range 200) { n : Σ 59 ≤ xs : x } fna
  get_curr vs.read { n : xs.to_nat | x+: fin.tail }细目 that n.mex_mod = 14
  sorry

end eleventh_number_is_149_l68_68050


namespace problem_proof_l68_68612

-- Define the mixed numbers and their conversions to improper fractions
def mixed_number_1 := 84 * 19 + 4  -- 1600
def mixed_number_2 := 105 * 19 + 5 -- 2000 

-- Define the improper fractions
def improper_fraction_1 := mixed_number_1 / 19
def improper_fraction_2 := mixed_number_2 / 19

-- Define the decimals and their conversions to fractions
def decimal_1 := 11 / 8  -- 1.375
def decimal_2 := 9 / 10  -- 0.9

-- Perform the multiplications
def multiplication_1 := (improper_fraction_1 * decimal_1 : ℚ)
def multiplication_2 := (improper_fraction_2 * decimal_2 : ℚ)

-- Perform the addition
def addition_result := multiplication_1 + multiplication_2

-- The final result is converted to a fraction for comparison
def final_result := 4000 / 19

-- Define and state the theorem
theorem problem_proof : addition_result = final_result := by
  sorry

end problem_proof_l68_68612


namespace sum_of_possible_k_l68_68250

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l68_68250


namespace number_of_intriguing_quintuples_l68_68086

def is_intriguing_quintuple (a b c d e : ℕ) : Prop :=
  1 ≤ a ∧ a < b ∧ b < c ∧ c < d ∧ d < e ∧ e ≤ 12 ∧ a + e > b + c + d

theorem number_of_intriguing_quintuples : 
  (finset.univ.image (λ (t : ℕ × ℕ × ℕ × ℕ × ℕ), (t.1, t.2.1, t.2.2.1, t.2.2.2.1, t.2.2.2.2))).count (λ (t : ℕ × ℕ × ℕ × ℕ × ℕ), is_intriguing_quintuple t.1 t.2.1 t.2.2.1 t.2.2.2.1 t.2.2.2.2) = 385 :=
by
-- Proof would be provided here, but is omitted for brevity
sorry

end number_of_intriguing_quintuples_l68_68086


namespace work_together_l68_68888

theorem work_together (A B : ℝ) (hA : A = 1/3) (hB : B = 1/6) : (1 / (A + B)) = 2 := by
  sorry

end work_together_l68_68888


namespace find_number_l68_68200

theorem find_number (N : ℝ) (h : 0.15 * 0.30 * 0.50 * N = 108) : N = 4800 :=
by
  sorry

end find_number_l68_68200


namespace mod_exp_subtract_lemma_problem_statement_l68_68072

theorem mod_exp_subtract_lemma : 
  ∀ (a b c : ℕ), 
  a ≡ c [MOD 12] → b ≡ c [MOD 12] → a^2903 - b^2903 ≡ 0 [MOD 12] := 
by
  intros a b c h1 h2
  sorry

theorem problem_statement : 
  51^2903 - 27^2903 ≡ 0 [MOD 12] := 
by
  have h51 : 51 ≡ 3 [MOD 12] := by sorry,
  have h27 : 27 ≡ 3 [MOD 12] := by sorry,
  exact mod_exp_subtract_lemma 51 27 3 h51 h27

end mod_exp_subtract_lemma_problem_statement_l68_68072


namespace inverse_function_ratio_l68_68775

noncomputable def g (x : ℚ) : ℚ := (3 * x + 2) / (2 * x - 5)

noncomputable def g_inv (x : ℚ) : ℚ := (-5 * x + 2) / (-2 * x + 3)

theorem inverse_function_ratio :
  ∀ x : ℚ, g (g_inv x) = x ∧ (∃ a b c d : ℚ, a = -5 ∧ b = 2 ∧ c = -2 ∧ d = 3 ∧ a / c = 2.5) :=
by
  sorry

end inverse_function_ratio_l68_68775


namespace collections_in_bag_l68_68367

noncomputable def distinct_collections : ℕ :=
  let vowels := ['A', 'I', 'O']
  let consonants := ['M', 'H', 'C', 'N', 'T', 'T']
  let case1 := Nat.choose 3 2 * Nat.choose 6 3 -- when 0 or 1 T falls off
  let case2 := Nat.choose 3 2 * Nat.choose 5 1 -- when both T's fall off
  case1 + case2

theorem collections_in_bag : distinct_collections = 75 := 
  by
  -- proof goes here
  sorry

end collections_in_bag_l68_68367


namespace exists_n_of_form_2k_l68_68097

theorem exists_n_of_form_2k (n : ℕ) (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h_sum : x + y + z = 0) (h_recip : 1/x + 1/y + 1/z = 1/(n : ℤ)) : ∃ k : ℕ, n = 2 * k :=
sorry

end exists_n_of_form_2k_l68_68097


namespace sum_of_possible_k_l68_68251

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l68_68251


namespace avg_salary_of_Raj_and_Roshan_l68_68382

variable (R S : ℕ)

theorem avg_salary_of_Raj_and_Roshan (h1 : (R + S + 7000) / 3 = 5000) : (R + S) / 2 = 4000 := by
  sorry

end avg_salary_of_Raj_and_Roshan_l68_68382


namespace find_m_n_l68_68647

theorem find_m_n : ∃ (m n : ℕ), m^m + (m * n)^n = 1984 ∧ m = 4 ∧ n = 3 := by
  sorry

end find_m_n_l68_68647


namespace proof_triangle_l68_68143

noncomputable def triangle_ABC (a b c B : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (h4 : B > 0) (h5 : B ≤ π / 3) (h6 : b^2 = a * c) : Prop :=
  ∃ (f : ℝ → ℝ), f(B) = 3 * Real.sin B + 4 * Real.cos B ∧
  (∀ x, f(x) ≤ 5) ∧ 
  f(B) = 5 ∧ 
  Real.tan B = 3 / 4

-- Statement of the theorem
theorem proof_triangle (a b c B : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h6 : b^2 = a * c) 
  (hB : B > 0 ∧ B ≤ π / 3) :
  triangle_ABC a b c B h1 h2 h3 hB.1 hB.2 h6 :=
begin
  -- Proof omitted
  sorry,
end

end proof_triangle_l68_68143


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68464

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68464


namespace greatest_prime_factor_of_144_l68_68524

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l68_68524


namespace T_lt_half_S_l68_68329

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68329


namespace pints_per_glass_l68_68597

theorem pints_per_glass (total_pints : ℝ) (glasses : ℕ) :
  total_pints = 237.5 ∧ glasses = 13 →
  (total_pints / glasses) ≈ 18.27 :=
by
  sorry

end pints_per_glass_l68_68597


namespace four_digit_perfect_square_palindrome_count_l68_68977

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68977


namespace identify_quadratic_equation_l68_68879

theorem identify_quadratic_equation :
  (¬(∃ x y : ℝ, x^2 - 2*x*y + y^2 = 0) ∧  -- Condition A is not a quadratic equation
   ¬(∃ x : ℝ, x*(x + 3) = x^2 - 1) ∧      -- Condition B is not a quadratic equation
   (∃ x : ℝ, x^2 - 2*x - 3 = 0) ∧         -- Condition C is a quadratic equation
   ¬(∃ x : ℝ, x + (1/x) = 0)) →           -- Condition D is not a quadratic equation
  (true) := sorry

end identify_quadratic_equation_l68_68879


namespace imaginary_part_of_fraction_is_correct_l68_68001

-- Define the given complex number
def complex_num : ℂ := 5 / (3 + 4 * complex.i)

-- Define the expected imaginary part
def expected_imaginary_part : ℝ := -4 / 5

-- Construct the theorem statement to prove that the imaginary part is as expected
theorem imaginary_part_of_fraction_is_correct : complex.im complex_num = expected_imaginary_part := by
  sorry -- Proof to be filled in

end imaginary_part_of_fraction_is_correct_l68_68001


namespace tn_lt_sn_div_2_l68_68305

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68305


namespace complement_union_l68_68666

open Finset

def I := {0, 1, 2, 3, 4, 5, 6, 7, 8}
def M := {1, 2, 4, 5}
def N := {0, 3, 5, 7}

theorem complement_union (hI : I = {0, 1, 2, 3, 4, 5, 6, 7, 8})
                        (hM : M = {1, 2, 4, 5})
                        (hN : N = {0, 3, 5, 7}) :
  (I \ (M ∪ N)) = {6, 8} :=
by
  rw [hI, hM, hN]
  sorry

end complement_union_l68_68666


namespace original_number_l68_68422

theorem original_number (sum_orig : ℕ) (sum_new : ℕ) (changed_value : ℕ) (avg_orig : ℕ) (avg_new : ℕ) (n : ℕ) :
    sum_orig = n * avg_orig →
    sum_new = sum_orig - changed_value + 9 →
    avg_new = 8 →
    avg_orig = 7 →
    n = 7 →
    sum_new = n * avg_new →
    changed_value = 2 := 
by
  sorry

end original_number_l68_68422


namespace arithmetic_sequence_c_d_sum_l68_68846

theorem arithmetic_sequence_c_d_sum :
  let c := 19 + (11 - 3)
  let d := c + (11 - 3)
  c + d = 62 :=
by
  sorry

end arithmetic_sequence_c_d_sum_l68_68846


namespace Tn_lt_Sn_div_2_l68_68309

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68309


namespace nth_monomial_l68_68042

-- Define the sequence of monomials
def monomial_seq : ℕ → ℝ → ℝ :=
  λ n a, (Real.sqrt n) * (a ^ (n + 1))

-- Prove that the nth monomial in the sequence matches the given pattern
theorem nth_monomial (n : ℕ) (a : ℝ) : monomial_seq (n + 1) a = (Real.sqrt (n + 1)) * (a ^ (n + 2)) :=
by
  sorry

end nth_monomial_l68_68042


namespace largest_6_digit_int_divisible_by_5_l68_68088

theorem largest_6_digit_int_divisible_by_5 :
  ∃ n, (n ≤ 999999) ∧ (999999 ≤ 999999) ∧ (n % 5 = 0) ∧ (∀ m, (m ≤ 999999) ∧ (m % 5 = 0) → n ≥ m) :=
begin
  use 999995,
  repeat {split},
  dec_trivial,
  dec_trivial,
  dec_trivial,
  intros m m_cond,
  cases m_cond,
  cases m_cond_right,
  cases m_cond_right_right,
  sorry
end

end largest_6_digit_int_divisible_by_5_l68_68088


namespace example_problem_l68_68709

def f (x : ℝ) : ℝ := real.sqrt 3 * real.sin (2 * x) + real.cos (2 * x)
def g (x : ℝ) : ℝ := 2 * real.sin (2 * x)

theorem example_problem : ¬ (∀ x : ℝ, f x = g (x + π / 6)) := by
  -- We claim that option B is incorrect
  sorry

end example_problem_l68_68709


namespace elongation_rate_significantly_improved_l68_68025

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end elongation_rate_significantly_improved_l68_68025


namespace max_sums_proof_l68_68919

-- Define the types of coins and their values
def Coin := {n : ℕ // n = 5 ∨ n = 10 ∨ n = 25 ∨ n = 50}

-- List of available coins
def available_coins : List Coin := 
  [{val:= 5, property:=Or.inl rfl}, {val:= 5, property:=Or.inl rfl}, {val:= 5, property:=Or.inl rfl},
   {val:= 10, property:=Or.inr (Or.inl rfl)}, {val:= 10, property:=Or.inr (Or.inl rfl)}, {val:= 10, property:=Or.inr (Or.inl rfl)},
   {val:= 25, property:=Or.inr (Or.inr (Or.inl rfl))}, {val:= 25, property:=Or.inr (Or.inr (Or.inl rfl))},
   {val:= 50, property:=Or.inr (Or.inr (Or.inr rfl))}, {val:= 50, property:=Or.inr (Or.inr (Or.inr rfl))}]

-- Function to calculate the possible sums
def possible_sums (coins : List Coin) : Set ℕ :=
  {s | ∃ (a b : Coin), a ∈ coins ∧ b ∈ coins ∧ s = a.val + b.val }

-- Definition of the maximum number of different sums
def max_different_sums := (possible_sums available_coins).card

-- Theorem to prove the maximum number of different sums
theorem max_sums_proof : max_different_sums = 10 :=
by sorry

end max_sums_proof_l68_68919


namespace unique_four_digit_palindromic_square_l68_68932

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68932


namespace shelves_needed_l68_68045

theorem shelves_needed (initial_stock : ℕ) (additional_shipment : ℕ) (bears_per_shelf : ℕ) (total_bears : ℕ) (shelves : ℕ) :
  initial_stock = 4 → 
  additional_shipment = 10 → 
  bears_per_shelf = 7 → 
  total_bears = initial_stock + additional_shipment →
  total_bears / bears_per_shelf = shelves →
  shelves = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end shelves_needed_l68_68045


namespace jenny_kenny_time_l68_68273

def JennyKennyVisibility (t : ℚ) : Prop :=
  let R := 100
  let d := 300
  let vj := 2
  let vk := 4
  let start_pos_jenny := (-100, 150)
  let start_pos_kenny := (-100, -150)
  let pos_jenny := (start_pos_jenny.1 + vj * t, start_pos_jenny.2)
  let pos_kenny := (start_pos_kenny.1 + vk * t, start_pos_kenny.2)
  let line_slope := -(d / (vk - vj) * t)
  let y_intercept := start_pos_kenny.2 - line_slope * start_pos_kenny.1
  let line_eq := fun x => line_slope * x + y_intercept
  let circle_eq := fun (x y : ℚ) => x^2 + y^2 = R^2
  ∀ (x y : ℚ), y = line_eq x ∧ circle_eq x y → t = 48

theorem jenny_kenny_time (t : ℚ) (h : JennyKennyVisibility t) :
  let n := t.num
  let d := t.denom
  n + d = 49 :=
sorry

end jenny_kenny_time_l68_68273


namespace solution_exists_if_a_ge_4_l68_68118

variable [Real x a : ℝ]

theorem solution_exists_if_a_ge_4 (a : ℝ) : 
  (∃ x : ℝ, (sqrt x - sqrt(x - a) = 2) ∧ (x ≥ 0) ∧ (x ≥ a)) → (a ≥ 4) :=
  by
    sorry

end solution_exists_if_a_ge_4_l68_68118


namespace find_day_53_days_from_friday_l68_68480

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68480


namespace four_digit_palindromic_perfect_square_l68_68952

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68952


namespace angle_between_BC_and_plane_PAC_l68_68230

/-- Given a regular tetrahedron S-ABCD, where 
      - O is the projection of the apex S onto the base ABCD
      - P is the midpoint of the lateral edge SD
      - SO = OD,
    the angle between line BC and plane PAC is 45 degrees. -/
theorem angle_between_BC_and_plane_PAC (S A B C D : Point) (O P : Point)
  (h_tetrahedron : RegularTetrahedron S A B C D)
  (h_projection : O = projection S (Plane.mk A B C D))
  (h_midpoint : P = midpoint S D)
  (h_lengths : distance S O = distance O D) :
  AngleBetweenLineAndPlane (Line.mk B C) (Plane.mk P A C) = degrees 45 :=
sorry

end angle_between_BC_and_plane_PAC_l68_68230


namespace triangle_area_proof_l68_68381

def line1 : ℝ → ℝ := λ x, 5
def line2 : ℝ → ℝ := λ x, 1 + x
def line3 : ℝ → ℝ := λ x, 1 - x

def point : Type := ℝ × ℝ

def intersection (l1 l2 : ℝ → ℝ) (x : ℝ) : point := (x, l1 x)

def area_of_triangle (p1 p2 p3 : point) : ℝ :=
  0.5 * abs ((p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2)))

def triangle_area_condition : Prop :=
  let A : point := (4, 5) in
  let B : point := (-4, 5) in
  let C : point := (0, 1) in
  area_of_triangle A B C = 16

theorem triangle_area_proof : triangle_area_condition :=
  sorry

end triangle_area_proof_l68_68381


namespace fewest_operations_return_16_l68_68023

noncomputable def f (x : ℝ) : ℝ := 1 / x
noncomputable def g (x : ℝ) : ℝ := x^2

theorem fewest_operations_return_16 :
  ∃ (seq : List (ℝ → ℝ)), 
    seq.length = 6 ∧ 
    (List.foldl (λ x f => f x) 16 seq) = 16 :=
begin
  -- We need to show that we can find such a sequence
  sorry
end

end fewest_operations_return_16_l68_68023


namespace probability_of_valid_pair_l68_68535

-- Definitions based on conditions:
def is_valid_pair (a b : ℕ) : Prop :=
  1 < a^2 + b^2 ∧ a^2 + b^2 ≤ 9 ∧ 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6

theorem probability_of_valid_pair : 
  let total_pairs := 36 in
  let valid_pairs := [(1,1), (1,2), (2,1), (2,2)] in
  (∀ (a b : ℕ), (a, b) ∈ valid_pairs ↔ is_valid_pair a b) →
  (valid_pairs.length / total_pairs = 1 / 9) :=
sorry

end probability_of_valid_pair_l68_68535


namespace TableCostEquals_l68_68365

-- Define the given conditions and final result
def total_spent : ℕ := 56
def num_chairs : ℕ := 2
def chair_cost : ℕ := 11
def table_cost : ℕ := 34

-- State the assertion to be proved
theorem TableCostEquals :
  table_cost = total_spent - (num_chairs * chair_cost) := 
by 
  sorry

end TableCostEquals_l68_68365


namespace coefficient_x3_in_expansion_l68_68850

theorem coefficient_x3_in_expansion :
  let f := (1 + 2 * x : ℝ) ^ 6 in
  (coeff f 3) = 160 :=
by {
  sorry
}

end coefficient_x3_in_expansion_l68_68850


namespace animal_fish_consumption_l68_68855

-- Definitions for the daily consumption of each animal
def daily_trout_polar1 := 0.2
def daily_salmon_polar1 := 0.4

def daily_trout_polar2 := 0.3
def daily_salmon_polar2 := 0.5

def daily_trout_polar3 := 0.25
def daily_salmon_polar3 := 0.45

def daily_trout_sealion1 := 0.1
def daily_salmon_sealion1 := 0.15

def daily_trout_sealion2 := 0.2
def daily_salmon_sealion2 := 0.25

-- Calculate total daily consumption
def total_daily_trout :=
  daily_trout_polar1 + daily_trout_polar2 + daily_trout_polar3 + daily_trout_sealion1 + daily_trout_sealion2

def total_daily_salmon :=
  daily_salmon_polar1 + daily_salmon_polar2 + daily_salmon_polar3 + daily_salmon_sealion1 + daily_salmon_sealion2

-- Calculate total monthly consumption
def total_monthly_trout := total_daily_trout * 30
def total_monthly_salmon := total_daily_salmon * 30

-- Total monthly fish bucket consumption
def total_monthly_fish := total_monthly_trout + total_monthly_salmon

-- The statement to prove the total consumption
theorem animal_fish_consumption : total_monthly_fish = 84 := by
  sorry

end animal_fish_consumption_l68_68855


namespace solve_for_x_l68_68659

theorem solve_for_x (y : ℝ) (x : ℝ) : (sqrt (x + y - 3) = 10) → (x = 103 - y) :=
begin
  sorry
end

end solve_for_x_l68_68659


namespace perimeter_equals_interior_tiles_l68_68631

theorem perimeter_equals_interior_tiles (m n : ℕ) (h : m ≤ n) :
  (2 * m + 2 * n - 4 = 2 * (m * n) - (2 * m + 2 * n - 4)) ↔ (m = 5 ∧ n = 12 ∨ m = 6 ∧ n = 8) :=
by sorry

end perimeter_equals_interior_tiles_l68_68631


namespace james_jail_time_l68_68268

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end james_jail_time_l68_68268


namespace probability_no_same_color_block_l68_68425

-- Definitions: conditions of the problem
structure PeopleBlocks (Person : Type) (Block : Type) :=
  (adaBlocks : Person → Block)
  (benBlocks : Person → Block)
  (cindyBlocks : Person → Block)
  (red : Block)
  (blue : Block)
  (yellow : Block)
  (black : Block)
  
-- Question translated to proof problem
theorem probability_no_same_color_block (Person : Type) (Block : Type) [Finite Person] [Fintype Block]
  [∀ (b : Block), Decidable (b = PeopleBlocks.red ∨ b = PeopleBlocks.blue ∨ b = PeopleBlocks.yellow ∨ b = PeopleBlocks.black)] :
  let num_people := 3
  let num_boxes := 4
  let prob := 14811 / 65536 in
    (∑ (person1 : Person), (∑ (person2 : Person), (∑ (person3 : Person), 
        if PeopleBlocks.adaBlocks person1 = PeopleBlocks.benBlocks person2 
          ∧ PeopleBlocks.benBlocks person2 = PeopleBlocks.cindyBlocks person3 then 1 else 0))) / 
    (num_people ^ num_boxes) = prob := 
  sorry -- proof goes here

end probability_no_same_color_block_l68_68425


namespace find_b21_l68_68124

def a (n : ℕ) : ℕ := n * (n + 1) / 2

def b (m : ℕ) : ℕ := 
  let filtered := List.filter (λ x, ¬ (x % 2 = 0)) (List.map a (List.range (2 * m + 1)))
  List.get₀ (List.reverse (List.sort (filtered))) m

theorem find_b21 : b 21 = 861 := 
  sorry

end find_b21_l68_68124


namespace min_distance_to_origin_l68_68698

theorem min_distance_to_origin (x y : ℝ) (h : 2 * x + y + 5 = 0) : ∃d, d = real.sqrt (x^2 + y^2) ∧ d = real.sqrt 5 :=
by
  existsi real.sqrt (x^2 + y^2),
  existsi real.sqrt 5,
  sorry

end min_distance_to_origin_l68_68698


namespace inequality_g_l68_68114

def f (x : ℝ) : ℝ := 1 - sin x

noncomputable def g (x : ℝ) : ℝ := ∫ t in 0..x, (x - t) * f t

theorem inequality_g (x y : ℝ) : g (x + y) + g (x - y) ≥ 2 * g x :=
  sorry

end inequality_g_l68_68114


namespace Tn_lt_Sn_over_2_l68_68322

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68322


namespace max_distinct_positive_integers_sum_of_squares_l68_68869

theorem max_distinct_positive_integers_sum_of_squares (n : ℕ) :
  (∃ (k : fin n → ℕ), (∀ i j, i ≠ j → k i ≠ k j) ∧ ∑ i in finset.range n, k i ^ 2 = 2531) → n ≤ 18 :=
sorry

end max_distinct_positive_integers_sum_of_squares_l68_68869


namespace max_stamps_purchasable_l68_68209

-- Conditions
def stamp_price : ℕ → ℕ
| n := if n > 50 then 45 else 50

def total_cost (n : ℕ) : ℕ :=
if n ≤ 50 then n * 50 else 50 * 50 + (n - 50) * 45

-- The question is to find the maximum number of stamps purchasable with $50
theorem max_stamps_purchasable :
  ∃ n : ℕ, total_cost n ≤ 5000 ∧ ∀ m : ℕ, total_cost m ≤ 5000 → m ≤ n :=
begin
  sorry
end

end max_stamps_purchasable_l68_68209


namespace ratio_alison_brittany_l68_68603

def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := 4000

theorem ratio_alison_brittany : alison_money * 2 = brittany_money :=
by
  sorry

end ratio_alison_brittany_l68_68603


namespace Tn_lt_Sn_div_2_l68_68313

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68313


namespace shots_per_puppy_l68_68071

-- Definitions
def num_pregnant_dogs : ℕ := 3
def puppies_per_dog : ℕ := 4
def cost_per_shot : ℕ := 5
def total_shot_cost : ℕ := 120

-- Total number of puppies
def total_puppies : ℕ := num_pregnant_dogs * puppies_per_dog

-- Total number of shots
def total_shots : ℕ := total_shot_cost / cost_per_shot

-- The theorem to prove
theorem shots_per_puppy : total_shots / total_puppies = 2 :=
by
  sorry

end shots_per_puppy_l68_68071


namespace log_sum_eq_23_over_4_fraction_expr_eq_1_over_2_l68_68015

theorem log_sum_eq_23_over_4 :
  log 3 (427 / 3) + log 10 25 + log 10 4 + log 7 (7^2) + log 2 3 - log 3 4 = 23 / 4 :=
sorry

theorem fraction_expr_eq_1_over_2 :
  (9 / 4)^(1 / 2) - (-0.96)^0 - (27 / 8)^(-2 / 3) + (3 / 2)^(-2) = 1 / 2 :=
sorry

end log_sum_eq_23_over_4_fraction_expr_eq_1_over_2_l68_68015


namespace find_day_53_days_from_friday_l68_68489

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68489


namespace Allison_sewing_time_l68_68605

theorem Allison_sewing_time 
  (A : ℕ) 
  (A_sewing_time : ∀ (Al_rate : ℝ), Al_rate = 1 / 12 → 
    ∀ (together_time : ℝ), together_time = 3 →
    ∀ (alone_time : ℝ), alone_time = 3.75 →
    (3 / A + 3 * (1 / 12) + 3.75 / A = 1) → 
    A = 9) : 
  ∃ A : ℕ, A = 9 := 
sorry

end Allison_sewing_time_l68_68605


namespace apple_ratio_simplest_form_l68_68376

theorem apple_ratio_simplest_form (sarah_apples brother_apples cousin_apples : ℕ) 
  (h1 : sarah_apples = 630)
  (h2 : brother_apples = 270)
  (h3 : cousin_apples = 540)
  (gcd_simplified : Nat.gcd (Nat.gcd sarah_apples brother_apples) cousin_apples = 90) :
  (sarah_apples / 90, brother_apples / 90, cousin_apples / 90) = (7, 3, 6) := 
by
  sorry

end apple_ratio_simplest_form_l68_68376


namespace exists_lattice_midpoint_among_five_points_l68_68134

-- Definition of lattice points
structure LatticePoint where
  x : ℤ
  y : ℤ

open LatticePoint

-- The theorem we want to prove
theorem exists_lattice_midpoint_among_five_points (A B C D E : LatticePoint) :
    ∃ P Q : LatticePoint, P ≠ Q ∧ (P.x + Q.x) % 2 = 0 ∧ (P.y + Q.y) % 2 = 0 := 
  sorry

end exists_lattice_midpoint_among_five_points_l68_68134


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68465

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68465


namespace find_day_53_days_from_friday_l68_68484

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68484


namespace carry_all_containers_l68_68017

theorem carry_all_containers (containers : Fin 35 → ℝ) (total_weight : containers.toList.sum = 18)
  (max_flight_weight : ∀ (subset : Finset (Fin 35)), subset.card = 34 → (∑ i in subset, containers i) ≤ 21) :
  ∃ (flight_schedule : Fin 35 → Fin 7), (∀ f : Fin 7, (∑ i in (Finset.univ.filter (λ x, flight_schedule x = f)), containers i) ≤ 3) :=
by
  sorry

end carry_all_containers_l68_68017


namespace probability_func_increasing_l68_68155

def isEven {a : Int} (a : Int) (b : Int) : Prop := a % 2 = 0
def pairSetA : Set Int := {0, 1, 2}
def pairSetB : Set Int := {-1, 1, 3, 5}

def isIncreasing (a b : Int) : Prop := 
  if a = 0 then (b <= 0)
  else (b / a <= 1)

theorem probability_func_increasing :
  let validPairs := Set.prod pairSetA pairSetB
  let satisfyingPairs := { (a, b) ∈ validPairs | isIncreasing a b }
  satisfyingPairs.card = 5 →
  validPairs.card = 9 →
  (satisfyingPairs.card : Real) / (validPairs.card : Real) = (5 / 9 : Real) :=
by
  sorry

end probability_func_increasing_l68_68155


namespace convex_polygon_not_divisible_into_non_convex_quadrilaterals_l68_68617

theorem convex_polygon_not_divisible_into_non_convex_quadrilaterals (p : ℕ) (polygon : polygon) 
  (Hconvex : ∀ (vertex : polygon.vertices), interior_angle polygon vertex < 180°) :
  ¬ ∃ (quads : list quadrilateral), (∀ q ∈ quads, non_convex q) ∧ (polygon_divided_into_congruent_quads p quads) :=
by
  sorry

end convex_polygon_not_divisible_into_non_convex_quadrilaterals_l68_68617


namespace actual_average_height_l68_68826

theorem actual_average_height :
  let avg_height : ℝ := 175
  let num_students : ℕ := 60
  let initial_total_height := avg_height * num_students
  let recorded_heights : ℕ → ℝ := ![160, 180, 155, 170, 200]
  let actual_heights : ℕ → ℝ := ![145, 165, 175, 155, 185]
  let corrections := ∑ i in finrange 5, recorded_heights i - actual_heights i
  let correct_total_height := initial_total_height - corrections
  let actual_avg_height := correct_total_height / num_students
  actual_avg_height = 174.33 :=
by
  sorry

end actual_average_height_l68_68826


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68457

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68457


namespace original_number_is_045_l68_68660

def greatest_even_le (y : ℝ) : ℝ := 
  if even (floor y) then floor y 
  else floor y - 1

theorem original_number_is_045 (y : ℝ) (h0 : greatest_even_le 6.45 = 6) (h1 : 6.45 - greatest_even_le 6.45 = 0.4500000000000002) : y = 0.45 := 
  sorry

end original_number_is_045_l68_68660


namespace solution_exists_l68_68285

open Set

-- Definitions
def cross_product_vec (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.2 * w.2 - v.1 * w.2.2, v.1 * w.2.1 - v.2.1 * w.1, v.2.1 * w.2.2 - v.2 * w.1)

def valid_set (S : Set (ℝ × ℝ × ℝ)) : Prop :=
  S = {v | ∃ v1 v2, v1 ∈ S ∧ v2 ∈ S ∧ v = cross_product_vec v1 v2}

-- Theorem statement
theorem solution_exists (n : ℕ) (S : Set (ℝ × ℝ × ℝ)) (h : valid_set S) (hn : S.card = n) :
  n = 1 ∨ n = 7 :=
by sorry

end solution_exists_l68_68285


namespace joan_gave_mike_seashells_l68_68275

-- Definitions based on the conditions
def original_seashells : ℕ := 79
def remaining_seashells : ℕ := 16
def given_seashells := original_seashells - remaining_seashells

-- The theorem we want to prove
theorem joan_gave_mike_seashells : given_seashells = 63 := by
  sorry

end joan_gave_mike_seashells_l68_68275


namespace magicians_trick_success_l68_68857

-- Define the set of dice
def dice_faces := Finset.range 6 + 1

-- Define the possible pairs
def pairs : Finset (ℕ × ℕ) :=
  Finset.product dice_faces dice_faces

-- Define the conditions
variables (total_dice : ℕ) (taken_dice : ℕ) (pairs_encoded : Finset (ℕ × ℕ))

-- Assume the total number of dice is 23
axiom total_dice_eq : total_dice = 23

-- Assume the magicians have predefined coding for pairs
axiom pairs_coding : ∀ (p : ℕ × ℕ), p ∈ pairs_encoded ↔ p ∈ pairs

-- Define the function that encodes the number of dice in pocket using pairs
noncomputable def encode_dice_number (n : ℕ) : (ℕ × ℕ) :=
  if h : 3 ≤ n ∧ n ≤ 23 then classical.some (pairs_encoded.choose (λ p, p ∈ pairs_encoded))
  else (0, 0)

-- Define the property that ensures the trick is successful
def trick_successful : Prop :=
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 23 →
  (∃ (p : ℕ × ℕ), (encode_dice_number n) = p ∧ p ∈ pairs_encoded)

-- State the theorem
theorem magicians_trick_success :
  total_dice = 23 →
  trick_successful :=
by {
  assume h_total,
  rw [total_dice_eq] at h_total,
  sorry
}

end magicians_trick_success_l68_68857


namespace find_difference_l68_68199

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l68_68199


namespace day_of_week_in_53_days_l68_68495

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68495


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68460

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68460


namespace max_police_officers_needed_l68_68363

theorem max_police_officers_needed (n : ℕ) (h1 : n = 10) (h2 : ∀ (i j : ℕ), i ≠ j → street_crosses i j) :
  choose n 2 = 45 :=
by
  sorry

end max_police_officers_needed_l68_68363


namespace circle_line_distance_l68_68743

theorem circle_line_distance (a : ℝ) :
  let center : ℝ × ℝ := (1, 2)
  let line : ℝ × ℝ → ℝ := λ p, p.1 - p.2 + a
  let distance : ℝ := (abs (center.1 - center.2 + a)) / (real.sqrt (1^2 + (-1)^2))
  distance = 2 ↔ a = 3 ∨ a = -1 := by
{
  sorry
}

end circle_line_distance_l68_68743


namespace palindromic_squares_count_l68_68947

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68947


namespace diamond_2_3_l68_68435

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_2_3 : diamond 2 3 = 16 :=
by
  -- Imported definition and theorem structure.
  sorry

end diamond_2_3_l68_68435


namespace Tn_lt_Sn_div2_l68_68298

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68298


namespace four_digit_perfect_square_palindrome_count_l68_68979

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68979


namespace min_parcels_to_cover_cost_l68_68361

theorem min_parcels_to_cover_cost (n : ℕ) : (∀ n, 8 * n ≥ 6000) → n = 750 :=
begin
  sorry
end

end min_parcels_to_cover_cost_l68_68361


namespace collinear_F_I_O_l68_68159

theorem collinear_F_I_O (A B C H E C' B' F I O : Point) 
  (h1 : is_orthocenter H A B C)
  (h2 : E = midpoint A H)
  (h3 : touches_incircle C' B' A B C)
  (h4 : F = reflection E (line_through B' C'))
  (h5 : is_incenter I A B C)
  (h6 : is_circumcenter O A B C) : collinear F I O :=
sorry

end collinear_F_I_O_l68_68159


namespace count_palindromic_four_digit_perfect_squares_l68_68982

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68982


namespace find_t_as_fraction_find_t_as_fraction_l68_68094

theorem find_t_as_fraction (t : ℝ) (h1 : EthanUtility t (10 - 2 * t) = EthanUtility (4 - t) (2 * t + 2)) : t = 2 :=
by
  sorry

def EthanUtility (hours_coding hours_swimming : ℝ) : ℝ :=
  2 * hours_coding * hours_swimming + 1

-- Definitions of utility functions for Saturday and Sunday
def SaturdayUtility (t : ℝ) := EthanUtility t (10 - 2 * t)
def SundayUtility (t : ℝ) := EthanUtility (4 - t) (2 * t + 2)

-- Assume the utilities for both days are equal
theorem find_t_as_fraction (t : ℝ)
  (h1 : SaturdayUtility t = SundayUtility t) : t = 2 :=
by
  sorry

end find_t_as_fraction_find_t_as_fraction_l68_68094


namespace seventh_observation_l68_68011

theorem seventh_observation (avg6 : ℕ) (new_avg7 : ℕ) (old_avg : ℕ) (new_avg_diff : ℕ) (n : ℕ) (m : ℕ) (h1 : avg6 = 12) (h2 : new_avg_diff = 1) (h3 : n = 6) (h4 : m = 7) :
  ((n * old_avg = avg6 * old_avg) ∧ (m * new_avg7 = avg6 * old_avg + m - n)) →
  m * new_avg7 = 77 →
  avg6 * old_avg = 72 →
  77 - 72 = 5 :=
by
  sorry

end seventh_observation_l68_68011


namespace no_counterexample_for_statement_S_l68_68353

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_divisible_by (m n : ℕ) : Prop :=
  n % m = 0

theorem no_counterexample_for_statement_S :
  ∀ n ∈ {90, 99, 108, 117}, is_divisible_by 9 (sum_of_digits n) → is_divisible_by 3 n :=
by
  intros n hn h
  sorry

end no_counterexample_for_statement_S_l68_68353


namespace Dana_hourly_wage_l68_68084

theorem Dana_hourly_wage :
  ∀ (hours_worked_Friday hours_worked_Saturday hours_worked_Sunday total_earnings : ℝ) (x : ℝ),
    hours_worked_Friday = 9 →
    hours_worked_Saturday = 10 →
    hours_worked_Sunday = 3 →
    total_earnings = 286 →
    (hours_worked_Friday + hours_worked_Saturday + hours_worked_Sunday) * x = total_earnings →
    x = 13 :=
by
  intros hours_worked_Friday hours_worked_Saturday hours_worked_Sunday total_earnings x
  assume H1 : hours_worked_Friday = 9
  assume H2 : hours_worked_Saturday = 10
  assume H3 : hours_worked_Sunday = 3
  assume H4 : total_earnings = 286
  assume H5 : (hours_worked_Friday + hours_worked_Saturday + hours_worked_Sunday) * x = total_earnings
  sorry

end Dana_hourly_wage_l68_68084


namespace exists_positive_integer_m_l68_68151

theorem exists_positive_integer_m (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0) (h_cd : c * d = 1) : 
  ∃ m : ℕ, (a * b ≤ ↑m * ↑m) ∧ (↑m * ↑m ≤ (a + c) * (b + d)) :=
by
  sorry

end exists_positive_integer_m_l68_68151


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68451

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68451


namespace weighted_valid_votes_l68_68756

theorem weighted_valid_votes :
  let total_votes := 10000
  let invalid_vote_rate := 0.25
  let valid_votes := total_votes * (1 - invalid_vote_rate)
  let v_b := (valid_votes - 2 * (valid_votes * 0.15 + valid_votes * 0.07) + valid_votes * 0.05) / 4
  let v_a := v_b + valid_votes * 0.15
  let v_c := v_a + valid_votes * 0.07
  let v_d := v_b - valid_votes * 0.05
  let weighted_votes_A := v_a * 3.0
  let weighted_votes_B := v_b * 2.5
  let weighted_votes_C := v_c * 2.75
  let weighted_votes_D := v_d * 2.25
  weighted_votes_A = 7200 ∧
  weighted_votes_B = 3187.5 ∧
  weighted_votes_C = 8043.75 ∧
  weighted_votes_D = 2025 :=
by
  sorry

end weighted_valid_votes_l68_68756


namespace combined_area_is_correct_l68_68918

-- Define the base and area of the first triangle
def base1 : ℝ := 50
def area1 : ℝ := 800

-- Define the base of the second triangle
def base2 : ℝ := 40

-- Calculate the common altitude using the first triangle's area
def altitude : ℝ := (2 * area1) / base1

-- Calculate the area of the second triangle using the same altitude
def area2 : ℝ := (1 / 2) * base2 * altitude

-- Define the total combined area
def total_area : ℝ := area1 + area2

-- The theorem proving the total combined area is 1440 square feet
theorem combined_area_is_correct : total_area = 1440 := by
  -- We use equations to find that total_area = 1440
  -- This is just set up for the formal proof
  sorry

end combined_area_is_correct_l68_68918


namespace sqrt_expression_range_l68_68208

theorem sqrt_expression_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) → x ≥ 2 := 
by 
  sorry

end sqrt_expression_range_l68_68208


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68468

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68468


namespace sum_of_possible_values_of_k_l68_68247

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l68_68247


namespace squats_yesterday_l68_68093

theorem squats_yesterday (S : ℕ)
  (h1 : ∀ (n : ℕ), n ≥ 0 → S + 5 * n)
  (h2 : S + 15 = 45)
  : S = 30 := sorry

end squats_yesterday_l68_68093


namespace hyperbola_eccentricity_l68_68680

variables (a b : ℝ)
def hyperbola_eq (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

theorem hyperbola_eccentricity 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (asymptote_slope : b / a = Real.tan (Real.pi / 6)) :
  let e := Real.sqrt (1 + (b^2 / a^2)) in
  e = (2 * Real.sqrt 3) / 3 :=
by
  sorry

end hyperbola_eccentricity_l68_68680


namespace part1_part2_17_74_part2_18_71_part2_19_68_part2_20_65_l68_68538

noncomputable section

-- Helper to handle division and multiplication calculations
def is_solution_part1 (x : Real) : Prop :=
  (144 * (1 + x)^2 = 225) → ((225 : Real) * (1 + 0.25) = 281)

-- Helper to handle integer constraints in part 2
def is_solution_part2 (a b : Nat) : Prop :=
  (6000 * a + 2000 * b = 250000) ∧ (3 * a ≤ b) ∧ (b ≤ 4.5 * a)

-- Part 1: Prove the number of family cars by end of 2013 is 281
theorem part1  : is_solution_part1 0.25 :=
  sorry

-- Part 2: Prove the possible schemes of constructing parking spaces
theorem part2_17_74 : is_solution_part2 17 74 :=
  sorry

theorem part2_18_71 : is_solution_part2 18 71 :=
  sorry

theorem part2_19_68 : is_solution_part2 19 68 :=
  sorry

theorem part2_20_65 : is_solution_part2 20 65 :=
  sorry

end part1_part2_17_74_part2_18_71_part2_19_68_part2_20_65_l68_68538


namespace range_of_b_l68_68394

noncomputable def range_b := {b | ∀ x, (e ^ x + b ≠ e ^ x) ∨ (e ^ x + b ≠ log x)}

theorem range_of_b (b : ℝ) : b ∈ range_b ↔ b ∈ Set.Icc (-2 : ℝ) (0 : ℝ) :=
by {
  sorry
}

end range_of_b_l68_68394


namespace amount_a_put_in_correct_l68_68007

noncomputable def amount_a_put_in (total_profit managing_fee total_received_by_a profit_remaining: ℝ) : ℝ :=
  let capital_b := 2500
  let a_receives_from_investment := total_received_by_a - managing_fee
  let profit_ratio := a_receives_from_investment / profit_remaining
  profit_ratio * capital_b

theorem amount_a_put_in_correct :
  amount_a_put_in 9600 960 6000 8640 = 3500 :=
by
  dsimp [amount_a_put_in]
  sorry

end amount_a_put_in_correct_l68_68007


namespace day_after_53_days_from_Friday_l68_68515

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68515


namespace find_day_53_days_from_friday_l68_68483

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68483


namespace days_from_friday_l68_68501

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68501


namespace train_travel_distance_l68_68572

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l68_68572


namespace largest_base_b_digit_sum_neq_17_l68_68651

noncomputable def base_b_digit_sum (b : ℕ) (n : ℕ) : ℕ :=
  let digits := List.unfoldr (λ x, if x = 0 then none else some (x % b, x / b)) n
  digits.sum

theorem largest_base_b_digit_sum_neq_17 :
  ∀ b: ℕ, ∃ n: ℕ, 
    (n = 12^4) → 
    (b = 32) → 
    (base_b_digit_sum b n ≠ 17) :=
by
  sorry

end largest_base_b_digit_sum_neq_17_l68_68651


namespace correct_sampling_method_order_l68_68047

-- Definitions for sampling methods
def simple_random_sampling (method : ℕ) : Bool :=
  method = 1

def systematic_sampling (method : ℕ) : Bool :=
  method = 2

def stratified_sampling (method : ℕ) : Bool :=
  method = 3

-- Main theorem stating the correct method order
theorem correct_sampling_method_order : simple_random_sampling 1 ∧ stratified_sampling 3 ∧ systematic_sampling 2 :=
by
  sorry

end correct_sampling_method_order_l68_68047


namespace find_x1_l68_68153

theorem find_x1 
  (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1^2) + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 1 / 4) :
  x1 = sqrt(15) / 4 :=
sorry

end find_x1_l68_68153


namespace satisfies_conditions_l68_68103

def f (x : ℝ) : ℝ := (x^2 + 1) / x

theorem satisfies_conditions : f (Real.log 2 / Real.log 3) = f (Real.log 3 / Real.log 2) :=
by 
    -- Assuming the definition of f from the problem
    have f_eq : ∀ x, f x = f (1 / x), from sorry,
    exact f_eq (Real.log 2 / Real.log 3)

end satisfies_conditions_l68_68103


namespace son_present_age_l68_68923

variable (S F : ℕ)

-- Given conditions
def father_age := F = S + 34
def future_age_rel := F + 2 = 2 * (S + 2)

-- Theorem to prove the son's current age
theorem son_present_age (h₁ : father_age S F) (h₂ : future_age_rel S F) : S = 32 := by
  sorry

end son_present_age_l68_68923


namespace a6_equals_5_l68_68170

variable {α : Type*}
variable [Add α] [Mul α] [One α] [Inv α] [Zero α] 

-- Define an arithmetic sequence a with common difference d
def arithmetic_seq (a : ℕ → α) (d : α) : Prop :=
  ∀ n k : ℕ, a (n + k) = a n + k * d

theorem a6_equals_5
  (a : ℕ → ℝ) (d : ℝ)
  (h_arith : arithmetic_seq a d)
  (h_sum_terms : a 2 + a 6 + a 10 = 15) :
  a 6 = 5 :=
by
  sorry

end a6_equals_5_l68_68170


namespace conference_handshakes_l68_68557

theorem conference_handshakes (n_leaders n_participants : ℕ) (n_total : ℕ) 
  (h_total : n_total = n_leaders + n_participants) 
  (h_leaders : n_leaders = 5) 
  (h_participants : n_participants = 25) 
  (h_total_people : n_total = 30) : 
  (n_leaders * (n_total - 1) - (n_leaders * (n_leaders - 1) / 2)) = 135 := 
by 
  sorry

end conference_handshakes_l68_68557


namespace charcoal_drawings_correct_l68_68423

-- Define the constants based on the problem conditions
def total_drawings : ℕ := 120
def colored_pencils : ℕ := 35
def blending_markers : ℕ := 22
def pastels : ℕ := 15
def watercolors : ℕ := 12

-- Calculate the total number of charcoal drawings
def charcoal_drawings : ℕ := total_drawings - (colored_pencils + blending_markers + pastels + watercolors)

-- The theorem we want to prove is that the number of charcoal drawings is 36
theorem charcoal_drawings_correct : charcoal_drawings = 36 :=
by
  -- The proof goes here (we skip it with 'sorry')
  sorry

end charcoal_drawings_correct_l68_68423


namespace day_of_week_in_53_days_l68_68497

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68497


namespace total_space_needed_for_trees_l68_68375

def appleTreeWidth : ℕ := 10
def spaceBetweenAppleTrees : ℕ := 12
def numAppleTrees : ℕ := 2

def peachTreeWidth : ℕ := 12
def spaceBetweenPeachTrees : ℕ := 15
def numPeachTrees : ℕ := 2

def totalSpace : ℕ :=
  numAppleTrees * appleTreeWidth + spaceBetweenAppleTrees +
  numPeachTrees * peachTreeWidth + spaceBetweenPeachTrees

theorem total_space_needed_for_trees : totalSpace = 71 := by
  sorry

end total_space_needed_for_trees_l68_68375


namespace sum_of_smallest_five_args_l68_68068

-- Define the polynomial Q(x)
def Q (x : ℂ) : ℂ :=
  let P := (1 + x + x^2 + x^3 + ... + x^20)^2
  P - x^18

-- The main theorem to prove
theorem sum_of_smallest_five_args : 
  let α1 := 1 / 22
  let α2 := 1 / 20
  let α3 := 2 / 22
  let α4 := 2 / 20
  let α5 := 3 / 22
  α1 + α2 + α3 + α4 + α5 = (83 / 220) :=
begin
  -- Simpler definition of the polynomial
  have hQ : Q(x) =  ((x^20 - 1) * (x^22 - 1)) / (x - 1)^2, sorry,
  -- Side conditions
  have hα : 0 < α1 ∧ α1 ≤ α2 ∧ α2 ≤ α3 ∧ α3 ≤ α4 ∧ α4 ≤ α5 ∧ α5 < 1, sorry,
  -- Main calculation
  calc
    α1 + α2 + α3 + α4 + α5
      = (1 / 22) + (1 / 20) + (2 / 22) + (2 / 20) + (3 / 22) : by sorry
  ... = (5 / 22) + (3 / 20) : by sorry
  ... = (50 / 220) + (33 / 220) : by sorry
  ... = 83 / 220 : by sorry
end

end sum_of_smallest_five_args_l68_68068


namespace sum_possible_values_k_l68_68254

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l68_68254


namespace ratio_of_volumes_l68_68038

theorem ratio_of_volumes (a : ℝ) (h_a : a > 0) :
  let side_octahedron := (a * Real.sqrt 2) / 2 in
  let volume_octahedron := (side_octahedron ^ 3 * Real.sqrt 2) / 3 in
  let radius_sphere := (a * Real.sqrt 3) / 2 in
  let volume_sphere := (4 / 3) * Real.pi * (radius_sphere ^ 3) in
  volume_octahedron / volume_sphere = 2 * Real.sqrt 6 / (27 * Real.pi) :=
by
  sorry

end ratio_of_volumes_l68_68038


namespace option_A_is_linear_equation_l68_68878

-- Definitions for considering an equation being linear in two variables
def is_linear_equation (e : Prop) : Prop :=
  ∃ (a b c : ℝ), e = (a = b + c) ∧ a ≠ 0 ∧ b ≠ 0

-- The given equation in option A
def Eq_A : Prop := ∀ (x y : ℝ), (2 * y - 1) / 5 = 2 - (3 * x - 2) / 4

-- Proof problem statement
theorem option_A_is_linear_equation : is_linear_equation Eq_A :=
sorry

end option_A_is_linear_equation_l68_68878


namespace rehabilitation_centers_total_l68_68876

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end rehabilitation_centers_total_l68_68876


namespace retirement_benefits_l68_68362

-- Given conditions
def aged_55 (Maria_Ivanovna : Type) : Prop :=
  Maria_Ivanovna.recently_turned_fifty_five

def works (Maria_Ivanovna : Type) : Prop := 
  Maria_Ivanovna.continues_to_work

def receives_additional_payments_from_state (Maria_Ivanovna : Type) : Prop := 
  aged_55 Maria_Ivanovna ∧ works Maria_Ivanovna

-- Desired conclusion
def additional_payments_called (Maria_Ivanovna : Type) : String :=
  if receives_additional_payments_from_state Maria_Ivanovna then "пенсия" else ""

-- The formal statement
theorem retirement_benefits (Maria_Ivanovna : Type) :
  receives_additional_payments_from_state Maria_Ivanovna → additional_payments_called Maria_Ivanovna = "пенсия" :=
sorry

end retirement_benefits_l68_68362


namespace Tn_lt_Sn_over_2_l68_68317

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68317


namespace trig_identity_proof_l68_68092

theorem trig_identity_proof :
  let sin := Real.sin
  let cos := Real.cos
  let deg_to_rad := fun θ : ℝ => θ * Real.pi / 180
  sin (deg_to_rad 30) * sin (deg_to_rad 75) - sin (deg_to_rad 60) * cos (deg_to_rad 105) = Real.sqrt 2 / 2 :=
by
  sorry

end trig_identity_proof_l68_68092


namespace squirrel_acorns_l68_68754

-- Definitions based on the problem conditions
variables {ℕ : Type} [linear_ordered_comm_ring ℕ]

definition chipmunk_holes (h_c : ℕ) (x : ℕ) : Prop :=
  4 * h_c = x

definition squirrel_holes (h_s : ℕ) (x : ℕ) : Prop :=
  5 * h_s = x

definition rabbit_holes (h_r : ℕ) (x : ℕ) : Prop :=
  6 * h_r = x

definition holes_relationship_ch_s (h_c h_s : ℕ) : Prop :=
  h_s = h_c - 1

definition holes_relationship_ch_r (h_c h_r : ℕ) : Prop :=
  h_r = h_c - 2

-- Theorem we want to prove
theorem squirrel_acorns (h_c h_s h_r x : ℕ) (H1 : chipmunk_holes h_c x) (H2 : squirrel_holes h_s x)
  (H3 : rabbit_holes h_r x) (H4 : holes_relationship_ch_s h_c h_s) (H5 : holes_relationship_ch_r h_c h_r)
  : 5 * h_s = 30 :=
sorry

end squirrel_acorns_l68_68754


namespace triangle_area_approx_143_58_l68_68211

/-- The area of a triangle with sides 26 cm, 24 cm, and 12 cm, is approximately 143.58 square centimeters. -/
theorem triangle_area_approx_143_58 :
  let a := 26
  let b := 24
  let c := 12
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  area ≈ 143.58 :=
by
  let a := 26
  let b := 24
  let c := 12
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  have hs : s = 31 := 
  sorry
  have harea : area ≈ 143.58 := 
  sorry
  exact harea

end triangle_area_approx_143_58_l68_68211


namespace palindromic_squares_count_l68_68942

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68942


namespace jane_oldest_child_age_l68_68890

-- Define the conditions
def jane_start_age : ℕ := 20
def jane_current_age : ℕ := 32
def stopped_babysitting_years_ago : ℕ := 10
def baby_sat_condition (jane_age child_age : ℕ) : Prop := child_age ≤ jane_age / 2

-- Define the proof problem
theorem jane_oldest_child_age :
  (∃ age_stopped child_age,
    stopped_babysitting_years_ago = jane_current_age - age_stopped ∧
    baby_sat_condition age_stopped child_age ∧
    (32 - stopped_babysitting_years_ago = 22) ∧ -- Jane's age when she stopped baby-sitting
    child_age = 22 / 2 ∧ -- Oldest child she could have baby-sat at age 22
    child_age + stopped_babysitting_years_ago = 21) --  current age of the oldest person for whom Jane could have baby-sat
:= sorry

end jane_oldest_child_age_l68_68890


namespace unique_four_digit_palindromic_square_l68_68934

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68934


namespace segmentOS_length_l68_68911

noncomputable def lengthOS (O P: ℝ × ℝ) (Q: ℝ × ℝ) (rO rP: ℝ) (OT TS: ℝ) : ℝ :=
  Real.sqrt ((OT ^ 2) + (TS ^ 2))

theorem segmentOS_length : 
  (radiusO radiusP : ℝ) (O P Q : ℝ × ℝ) (OT TS : ℝ) 
  (hO : O = (0, 0)) 
  (hP : P = (11, 0)) 
  (hQ : (radiusO = 5) ∧ (radiusP = 3) ∧ (TS = (4 * Real.sqrt 7)) ∧ (OT = 5)) :
  lengthOS O P Q radiusO radiusP OT TS = Real.sqrt 137 :=
by 
  simp [lengthOS]
  sorry

end segmentOS_length_l68_68911


namespace remainder_division_zero_l68_68534

noncomputable theory

def polynomial (R : Type*) [CommRing R] := R[X]

open polynomial

def P : polynomial ℚ := X^3 + 3 * X^2 - 4
def Q : polynomial ℚ := X^2 + X - 2

theorem remainder_division_zero : P % Q = 0 :=
by sorry

end remainder_division_zero_l68_68534


namespace sum_even_binom_sum_odd_binom_sum_mod4_binom_l68_68062

-- Problem (a): Sum of even indexed binomial coefficients
theorem sum_even_binom {n : ℕ} : (∑ k in (range (n + 1)).filter (λ k, k % 2 = 0), (nat.choose n k)) = 2^(n - 1) :=
sorry

-- Problem (b): Sum of odd indexed binomial coefficients
theorem sum_odd_binom {n : ℕ} : (∑ k in (range (n + 1)).filter (λ k, k % 2 = 1), (nat.choose n k)) = 2^(n - 1) :=
sorry

-- Problem (c): Sum of binomial coefficients where index is a multiple of 4
theorem sum_mod4_binom {n : ℕ} : (∑ k in (range (n + 1)).filter (λ k, k % 4 = 0), (nat.choose n k)) = (∑ k in (range (n+1)).filter (λ k, k % 4 = 0), (nat.choose n k)) :=
sorry

end sum_even_binom_sum_odd_binom_sum_mod4_binom_l68_68062


namespace sum_divisible_by_101_l68_68798

-- Define the set of digits from 1 to 8
def valid_digits := {d : ℕ | 1 ≤ d ∧ d ≤ 8}

-- Define a valid four-digit number
def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧
  (∀ k ∈ [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10], k ∈ valid_digits)

-- Define the sum of all valid four-digit numbers
def sum_valid_numbers : ℕ :=
  ∑ n in (finset.filter is_valid_number (finset.range 10000)), n

-- Statement: Prove that the sum of all valid four-digit numbers is divisible by 101
theorem sum_divisible_by_101 : sum_valid_numbers % 101 = 0 :=
begin
  sorry
end

end sum_divisible_by_101_l68_68798


namespace coefficient_of_x5_in_expansion_l68_68827

/-- The coefficient of x^5 in the expansion of (x^2 - 1/√x)^5 is 10. -/
theorem coefficient_of_x5_in_expansion :
  (let f := (fun x : ℚ => x^2 - (1 / x^(1/2))) in
   let expansion := (fun x : ℚ => (f x)^5) in
   polynomial.coeff (polynomial.expand ℚ expansion) 5 = 10) :=
sorry

end coefficient_of_x5_in_expansion_l68_68827


namespace sum_of_mapped_elements_is_ten_l68_68681

theorem sum_of_mapped_elements_is_ten (a b : ℝ) (h1 : a = 1) (h2 : b = 9) : a + b = 10 := by
  sorry

end sum_of_mapped_elements_is_ten_l68_68681


namespace train_travel_distance_l68_68573

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l68_68573


namespace palindromic_squares_count_l68_68939

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68939


namespace sum_possible_values_k_l68_68257

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l68_68257


namespace unique_four_digit_palindromic_square_l68_68930

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68930


namespace imo_1991_prel_32_l68_68691

theorem imo_1991_prel_32 {n : ℕ} (a : ℕ → ℤ) 
    (h1 : n ≥ 3) 
    (h2 : ∀ i, i < n → ¬ n ∣ a i) 
    (h3 : ¬ n ∣ (Finset.range n).sum (λ i, a i)) : 
    ∃ (e : ℕ → ℕ) (S : Finset (ℕ → ℕ)), 
      S.card = n ∧ 
      (∀ f ∈ S, ∀ i, e i ∈ {0, 1}) ∧ 
      (∀ f ∈ S, n ∣ (Finset.range n).sum (λ i, (e i) * a i)) :=
sorry

end imo_1991_prel_32_l68_68691


namespace form_hcl_l68_68653

def chemical_reaction (c2h6 cl2 c2h5cl hcl : Nat) : Prop :=
  c2h6 + cl2 = c2h5cl + hcl

theorem form_hcl (moles_c2h6 moles_cl2 : Nat) (h : chemical_reaction moles_c2h6 moles_cl2 0 (min moles_c2h6 moles_cl2)) : moles_c2h6 = 3 ∧ moles_cl2 = 3 → min moles_c2h6 moles_cl2 = 3 :=
by
  intros h1
  cases h1
  rw [h1_left, h1_right]
  simp
  sorry

end form_hcl_l68_68653


namespace correct_articles_for_generic_sentence_l68_68059

theorem correct_articles_for_generic_sentence
  (crossroads_nonspecific : bool)
  (bank_nonspecific : bool)
  (crossroads_correct_article : string)
  (bank_correct_article : string) :
  crossroads_nonspecific = true →
  bank_nonspecific = true →
  crossroads_correct_article = "a" →
  bank_correct_article = "a" →
  crossroads_correct_article = "a" ∧ bank_correct_article = "a" :=
by
  intros _ _ _ _
  split
  . apply rfl
  . apply rfl

end correct_articles_for_generic_sentence_l68_68059


namespace slope_of_line_PQ_is_8_over_3_l68_68746

noncomputable def f(x : ℝ) : ℝ := 2 * Real.sin x
noncomputable def g(x : ℝ) : ℝ := 2 * Real.sqrt x * (x / 3 + 1)

noncomputable def f' (x : ℝ) : ℝ := 2 * Real.cos x
noncomputable def g' (x : ℝ) : ℝ := (Real.sqrt x + (2 * x) / (3 * Real.sqrt x))

theorem slope_of_line_PQ_is_8_over_3 :
  ∀ x_P x_Q : ℝ, (0 ≤ x_P ∧ x_P ≤ π) → (0 ≤ x_Q ∧ x_Q ≤ π) →
  f' x_P = 2 ∧ g' x_Q = 2 →
  let P := (x_P, f x_P),
      Q := (x_Q, g x_Q)
  in ((Q.2 - P.2) / (Q.1 - P.1) = 8 / 3) := by
  sorry

end slope_of_line_PQ_is_8_over_3_l68_68746


namespace trains_clear_time_l68_68892

theorem trains_clear_time :
  ∀ (length_train1 length_train2 : ℝ) (speed_train1 speed_train2 : ℝ),
  length_train1 = 111 ∧ length_train2 = 165 ∧ speed_train1 = 100 ∧ speed_train2 = 120 →
  (length_train1 + length_train2) / ((speed_train1 + speed_train2) * 5 / 18) = 4.51 :=
by
  intros length_train1 length_train2 speed_train1 speed_train2 h,
  rcases h with ⟨h1, h2, h3, h4⟩,
  rw [h1, h2, h3, h4],
  sorry

end trains_clear_time_l68_68892


namespace incircle_radius_of_45_45_90_triangle_l68_68758

theorem incircle_radius_of_45_45_90_triangle
  (AC BC AB : ℝ)
  (hAC : AC = x)
  (hBC : BC = x)
  (hAB : AB = x * Real.sqrt 2)
  (right_triangle : ∃ C : ℝ, ∠C = 90)
  (angle_A : ∃ A : ℝ, A = 45) :
  let area := (1 / 2) * AC * BC,
      s := (AC + BC + AB) / 2 in
  (area = s * r) → 
  r = x * (1 - Real.sqrt 2 / 2) :=
by 
  sorry

end incircle_radius_of_45_45_90_triangle_l68_68758


namespace find_m_from_parallel_vectors_l68_68721

variables (m : ℝ)

def a : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (2, -3)

-- The condition that vectors a and b are parallel
def vectors_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u.1 = k * v.1 ∧ u.2 = k * v.2

-- Given that a and b are parallel, prove that m = -3/2
theorem find_m_from_parallel_vectors
  (h : vectors_parallel (1, m) (2, -3)) :
  m = -3 / 2 :=
sorry

end find_m_from_parallel_vectors_l68_68721


namespace exists_m_divisible_by_n_with_digit_sum_l68_68695

theorem exists_m_divisible_by_n_with_digit_sum 
  (n k : ℕ) 
  (hn_pos : 0 < n)
  (hk_pos : 0 < k) 
  (hn_not_div_by_3 : ¬ (3 ∣ n)) 
  (hk_ge_n : k ≥ n) 
  : ∃ (m : ℕ), (n ∣ m) ∧ (Nat.digits 10 m).Sum = k := 
sorry

end exists_m_divisible_by_n_with_digit_sum_l68_68695


namespace find_a_l68_68178

noncomputable def tangent_line (a : ℝ) (x : ℝ) := (3 * a * (1:ℝ)^2 + 1) * (x - 1) + (a * (1:ℝ)^3 + (1:ℝ) + 1)

theorem find_a : ∃ a : ℝ, tangent_line a 2 = 7 := 
sorry

end find_a_l68_68178


namespace root_inequality_l68_68810

noncomputable theory

open Real

theorem root_inequality (a : inout ℕ) (k : ℕ) (hk: k ≠ 0) (a b : fin k.succ → ℝ) (ha : ∀ i, 0 < a i) (hb : ∀ i, 0 < b i) :
  (∏ i, (a i + b i)).pow(1.0 / k.succ) ≥ (∏ i, a i).pow (1.0 / k.succ) + (∏ i, b i).pow (1.0 / k.succ) :=
sorry

end root_inequality_l68_68810


namespace boss_spends_7600_per_month_l68_68224

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end boss_spends_7600_per_month_l68_68224


namespace rectangle_area_l68_68037

theorem rectangle_area (y : ℝ) (w : ℝ) (h : 3 * w) (d : Real.sqrt (w^2 + (3 * w)^2) = y) :
  3 * w^2 = 3 * y^2 / 10 :=
by
  sorry

end rectangle_area_l68_68037


namespace probability_same_color_l68_68021

-- The problem statement in Lean 4
theorem probability_same_color (white_ball black_ball red_ball : ℕ) (h1 : white_ball = 1) (h2 : black_ball = 2) (h3 : red_ball = 3) :
  let total_balls := white_ball + black_ball + red_ball in
  let total_ways := Nat.choose total_balls 2 in
  total_balls = 6 → total_ways = 15 → 
  let ways_two_black := Nat.choose black_ball 2 in
  let ways_two_red := Nat.choose red_ball 2 in
  let ways_same_color := ways_two_black + ways_two_red in
  (ways_two_black = 1 ∧ ways_two_red = 3) → ways_same_color = 4 →
  (ways_same_color : ℚ) / (total_ways : ℚ) = 4 / 15 :=
by 
  intro h_total_balls
  intro h_total_ways
  intro h_ways
  intro h_same_color
  sorry

end probability_same_color_l68_68021


namespace days_from_friday_l68_68509

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68509


namespace order_of_numbers_l68_68424

theorem order_of_numbers (a b c : ℝ) 
  (ha : 0 ≤ a ∧ a ≤ (real.pi / 2) ∧ real.cos a = a) 
  (hb : 0 ≤ b ∧ b ≤ (real.pi / 2) ∧ real.sin (real.cos b) = b) 
  (hc : 0 ≤ c ∧ c ≤ (real.pi / 2) ∧ real.cos (real.sin c) = c) : 
  b < a ∧ a < c := 
by
  sorry

end order_of_numbers_l68_68424


namespace deepak_age_l68_68843

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l68_68843


namespace square_side_length_l68_68740

theorem square_side_length (area_eq : ∀ {a : ℝ}, ∃ (w l : ℝ), w = 4 ∧ l = 9 ∧ a = w * l) : 
  ∃ (s : ℝ), s ∈ ℝ ∧ s^2 = 36 ∧ s = 6 :=
by
  have A : ∃ (w l : ℝ), w = 4 ∧ l = 9 ∧ 36 = w * l := 
    area_eq,
  cases A with w hw,
  cases hw with l hl,
  cases hl with hw4 hl9,
  cases hl9 with hlval harea,
  use 6,
  split,
  -- Proof that 6 is a real number
  apply Real.r,
  split,
  -- Proof that 6^2 = 36
  exact pow_two 6,
  -- Proof that 6 is the side length of the square
  sorry

end square_side_length_l68_68740


namespace eccentricity_of_ellipse_l68_68393

noncomputable def e (a b c : ℝ) : ℝ := c / a

theorem eccentricity_of_ellipse (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : (a - c) * (a + c) = (2 * c)^2) : e a b c = (Real.sqrt 5) / 5 := 
by
  sorry

end eccentricity_of_ellipse_l68_68393


namespace printing_machine_completion_constant_rate_l68_68027

theorem printing_machine_completion_constant_rate :
  ∃ t : ℕ, t = 6 ∧ (machine_completion_time 3 9 12 15 t) :=
by
    let starting_time := 9 -- 9:00 AM
    let first_half_completed := 12 -- 12:00 PM
    let additional_quarter_completed := 15 -- 3:00 PM
    let completion_time := 18 -- 6:00 PM

    have half_work_done_in := first_half_completed - starting_time -- 3 hours
    have additional_quarter_done_in := additional_quarter_completed - first_half_completed -- 3 hours

    have total_done_in := 2 * half_work_done_in -- as half time computes half work, 3 * 2 = 6 hours
    have total_done_by := starting_time + total_done_in -- by 9 + 6 = 15:00 = 3:00 PM
    have remaining_work := (completion_time - additional_quarter_completed)

    existsi completion_time -- 6:00 PM
    exact ⟨
        rfl,
        sorry -- Proof of correct computation
    ⟩

end printing_machine_completion_constant_rate_l68_68027


namespace part1_part2_l68_68894

-- Definitions involving geometric configurations
variables {A B C P D E F G H K M N : Point}
variables {α : Angle}
variables [noncomputable] -- allows us to not concern ourselves with explicit construction

-- Geometric conditions
axiom point_in_triangle : ∀ {A B C P : Point}, point_in_triangle P A B C → true
axiom circumcircle_PBC_tangent_AB : ∀ {P B C D : Point}, on_circumcircle P B C D → tangent_at D (line_through A B) → true
axiom circumcircle_PBC_tangent_AC : ∀ {P B C E : Point}, on_circumcircle P B C E → tangent_at E (line_through A C) → true
axiom circumcircle_PAC_tangent_AB : ∀ {P A C F : Point}, on_circumcircle P A C F → tangent_at F (line_through A B) → true
axiom circumcircle_PAC_tangent_BC : ∀ {P A C G : Point}, on_circumcircle P A C G → tangent_at G (line_through B C) → true
axiom circumcircle_PAB_tangent_AC : ∀ {P A B H : Point}, on_circumcircle P A B H → tangent_at H (line_through A C) → true
axiom circumcircle_PAB_tangent_BC : ∀ {P A B K : Point}, on_circumcircle P A B K → tangent_at K (line_through B C) → true
axiom intersection_FK_DE : ∀ {F K D E M : Point}, line_intersection F K D E M → true
axiom intersection_HG_DE : ∀ {H G D E N : Point}, line_intersection H G D E N → true
axiom midpoint_P_DE : ∀ {P D E : Point}, midpoint P D E → true

-- Theorem 1: 
theorem part1 (h1 : point_in_triangle P A B C)
             (h2 : on_circumcircle P B C D ∧ tangent_at D (line_through A B))
             (h3 : on_circumcircle P B C E ∧ tangent_at E (line_through A C))
             (h4 : on_circumcircle P A C F ∧ tangent_at F (line_through A B))
             (h5 : on_circumcircle P A C G ∧ tangent_at G (line_through B C))
             (h6 : on_circumcircle P A B H ∧ tangent_at H (line_through A C))
             (h7 : on_circumcircle P A B K ∧ tangent_at K (line_through B C))
             (h8 : line_intersection F K D E M)
             (h9 : line_intersection H G D E N) :
    (1 / distance P M) - (1 / distance P N) = (1 / distance P D) - (1 / distance P E) := 
sorry

-- Theorem 2:
theorem part2 (h1 : point_in_triangle P A B C)
             (h2 : on_circumcircle P B C D ∧ tangent_at D (line_through A B))
             (h3 : on_circumcircle P B C E ∧ tangent_at E (line_through A C))
             (h4 : on_circumcircle P A C F ∧ tangent_at F (line_through A B))
             (h5 : on_circumcircle P A C G ∧ tangent_at G (line_through B C))
             (h6 : on_circumcircle P A B H ∧ tangent_at H (line_through A C))
             (h7 : on_circumcircle P A B K ∧ tangent_at K (line_through B C))
             (h8 : line_intersection F K D E M)
             (h9 : line_intersection H G D E N)
             (h10 : midpoint P D E) :
    distance P M = distance P N :=
sorry

end part1_part2_l68_68894


namespace largest_n_dividing_factorial_l68_68104

theorem largest_n_dividing_factorial (n : ℕ) (h : ((n!)!)! ∣ (2004!)!) : n ≤ 6 :=
sorry

example : ((6!)!)! ∣ (2004!)! :=
sorry

end largest_n_dividing_factorial_l68_68104


namespace select_100000_phone_numbers_l68_68265

theorem select_100000_phone_numbers :
  ∃ S : Finset (Fin 1000000), S.card = 100000 ∧
    (∀ k : Fin 6, Finset.univ.image (λ n : Fin 1000000, (n / 10^k + n % 10^(5 - k))) = Finset.univ : Fin 100000) :=
begin
  sorry
end

end select_100000_phone_numbers_l68_68265


namespace T_lt_half_S_l68_68328

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68328


namespace compute_BC_l68_68428

noncomputable theory

open_locale classical

variables {A B C D E : Type*} [linear_ordered_field ABC] [linear_ordered_field BC]
variables (AB AC BC : ℝ)

def is_triangle (A B C : Type*) (AB AC BC: ℝ) := 
  ∃ (A B C : ABC), AB = 5 ∧ AC = 7

def is_medians_divide (BD DE EC: ℝ) :=
  BD = DE ∧ DE = EC

axiom big_angle (angle_BAC angle_DAE : ℝ) : (angle_BAC : ℝ) > 90

axiom supplementary_angles (angle_BAC angle_DAE : ℝ) (h_angle: ∀ BAC DAE : ABC, angle_BAC + angle_DAE = 180)

theorem compute_BC 
  (h_triangle: is_triangle A B C AB AC BC)
  (h_medians: is_medians_divide BD DE EC)
  (h_angleBAC_gt_ninety: big_angle (angle_BAC: ℝ) > 90)
  (h_supplementary: supplementary_angles angle_BAC angle_DAE) : 
    BC = real.sqrt 111 :=
sorry

end compute_BC_l68_68428


namespace infinite_terms_divisible_by_3_2011_l68_68138

noncomputable def d (n : ℕ) : ℕ :=
if n = 1 then 1 else (n / (2 : ℕ).choose (∃ k, k ∣ n ∧ k < n ∧ k > 0))

def seq (a : ℕ) : ℕ → ℕ
| 0     := a
| (i+1) := seq i + d (seq i)

theorem infinite_terms_divisible_by_3_2011 (a1 : ℕ) : 
  ∃ᶠ i in at_top, 3^2011 ∣ seq a1 i :=
sorry

end infinite_terms_divisible_by_3_2011_l68_68138


namespace find_x_l68_68110

theorem find_x :
  ∃ x : ℝ, (0 < x) ∧ (⌊x⌋ * x + x^2 = 93) ∧ (x = 7.10) :=
by {
   sorry
}

end find_x_l68_68110


namespace find_x_l68_68722

variables {x : ℝ}

def a := (x, 2)
def b := (3, 6)

theorem find_x (h : a.1 * b.1 + a.2 * b.2 = 0) : x = -4 :=
by sorry

end find_x_l68_68722


namespace T_lt_half_S_l68_68330

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68330


namespace map_distance_representation_l68_68803

theorem map_distance_representation
  (cm_to_km_ratio : 15 = 90)
  (km_to_m_ratio : 1000 = 1000) :
  20 * (90 / 15) * 1000 = 120000 := by
  sorry

end map_distance_representation_l68_68803


namespace no_four_digit_perfect_square_palindromes_l68_68961

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68961


namespace Rachel_chairs_l68_68811

theorem Rachel_chairs (minutes_per_piece : ℕ) (total_minutes : ℕ) (tables : ℕ) : 
  minutes_per_piece = 4 → 
  total_minutes = 40 → 
  tables = 3 → 
  (total_minutes - tables * minutes_per_piece) / minutes_per_piece = 7 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Rachel_chairs_l68_68811


namespace points_lie_on_parabola_l68_68116

-- Definitions from the problem
def point (u : ℝ) : ℝ × ℝ :=
  (3^u - 4, 9^u - 7 * 3^u - 2)

-- Theorem Statement
theorem points_lie_on_parabola : ∀ u : ℝ, ∃ a b c, let (x, y) := point u in y = a * x^2 + b * x + c :=
by
  -- Since we are not proving, we place "sorry" here to indicate the proof is omitted
  sorry

end points_lie_on_parabola_l68_68116


namespace question1_question2_l68_68830

variable {n : ℕ}
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Condition 1: The first term of the sequence {$a_n$} is a_1 = 1
def a1 := (a 1 = 1)

-- Condition 2: The relationship between the n-th term a_n and the sum of the first n terms S_n satisfies a_n = 2 * (S n)^2 / (2 * (S n) - 1) for n ≥ 2
def an_relation := ∀ n : ℕ, n ≥ 2 → (a n = 2 * (S n) ^ 2 / (2 * S n - 1))

-- Question 1: Prove that the sequence {1 / S n} is an arithmetic sequence.
theorem question1 (h1: a1) (h2: an_relation) : ∃ d : ℝ, ∀ n : ℕ, n ≥ 1 → (1 / S (n + 1) - 1 / S n = d) := sorry

-- Suppose there exists a positive number k such that the inequality (1 + S 1)(1 + S 2)...(1 + S n) ≥ k * sqrt(2n + 1) holds for all n ∈ ℕ*
def inequality_condition (k : ℝ) := ∀ n : ℕ, n > 0 → ((List.product (List.map (λ m, 1 + S m) (List.range n))) ≥ k * real.sqrt (2 * n + 1))

-- Question 2: Find the maximum value of k.
theorem question2 : ∃ k : ℝ, 0 < k ∧ (inequality_condition k) ∧ (∀ k' : ℝ,  inequality_condition k' → k' ≤ 2 / real.sqrt 3) := sorry

end question1_question2_l68_68830


namespace day_after_53_days_from_Friday_l68_68517

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68517


namespace palindromic_squares_count_l68_68941

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68941


namespace six_contestants_speaking_orders_l68_68418

theorem six_contestants_speaking_orders :
  ∃ orders : ℕ, 
    (∀ (A_first A_last : ℕ), A_first ≠ 1 → A_last ≠ 6 → 
       orders = 4 * Nat.factorial 5) ∧ orders = 480 :=
by
  sorry

end six_contestants_speaking_orders_l68_68418


namespace total_area_of_paper_l68_68741

theorem total_area_of_paper (area_of_folded_triangle : ℝ) (h : area_of_folded_triangle = 7) : 
  ∃ total_area : ℝ, total_area = 2 * area_of_folded_triangle ∧ total_area = 14 :=
by
  use 2 * area_of_folded_triangle
  split
  { -- Prove that the total area is 2 times the area of the folded triangle
    rw mul_comm, },
  { -- Prove that the total area is 14 cm²
    rw [h, mul_comm],
    norm_num, }
  sorry

end total_area_of_paper_l68_68741


namespace find_number_l68_68907

theorem find_number (x : ℝ) : (8 * x = 0.4 * 900) -> x = 45 :=
by
  sorry

end find_number_l68_68907


namespace max_green_socks_l68_68562

theorem max_green_socks (g y : ℕ) (h1 : g + y ≤ 2025)
  (h2 : (g * (g - 1))/(g + y) * (g + y - 1) = 1/3) : 
  g ≤ 990 := 
sorry

end max_green_socks_l68_68562


namespace exponential_sum_identity_l68_68197

theorem exponential_sum_identity (γ δ : ℝ)
  (h : complex.exp (complex.I * γ) + complex.exp (complex.I * δ) = (1 / 3 : ℂ) + (2 / 5 : ℂ) * complex.I) :
  complex.exp (-complex.I * γ) + complex.exp (-complex.I * δ) = (1 / 3 : ℂ) - (2 / 5 : ℂ) * complex.I :=
by sorry

end exponential_sum_identity_l68_68197


namespace number_of_valid_A_l68_68115

-- Given conditions
def is_digit (A : ℕ) : Prop := A < 10
def divisible_by (n m : ℕ) : Prop := m % n = 0
def last_two_digits_div_by_4 (A : ℕ) : Prop := divisible_by 4 (A * 10 + 4)

-- Main statement
theorem number_of_valid_A : 
  (finset.filter (λ A, is_digit A ∧ divisible_by A 75 ∧ last_two_digits_div_by_4 A) (finset.range 10)).card = 0 :=
by
  -- Proof goes here
  sorry

end number_of_valid_A_l68_68115


namespace solution_set_of_inequality_l68_68359

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

axiom condition1 (x : ℝ) : f''(x) / Real.log 2 > f(x)
axiom condition2 : f(1) = 4

theorem solution_set_of_inequality :
  {x : ℝ | f(x) ≥ 2^(x + 1)} = {x : ℝ | x ≥ 1} :=
sorry

end solution_set_of_inequality_l68_68359


namespace day_53_days_from_friday_l68_68449

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68449


namespace matrix_vector_multiplication_l68_68629

open Matrix

-- Define the types and matrix/vector dimensions
def n : ℕ := 2
def m : ℕ := 2

-- Define the matrix A
def A : Matrix (Fin m) (Fin n) ℤ := !![3, 2; -4, 5]

-- Define the vector v
def v : Fin n → ℤ := ![4, -2]

-- Define the vector w (multiplier vector)
def w : Fin n → ℤ := ![2, -3]

-- Define the expected result
def expected_result : Fin n → ℤ := ![16, 78]

-- Statement of the theorem
theorem matrix_vector_multiplication :
  (λ i : Fin n, (A.mulVec v) i * w i) = expected_result :=
by
  sorry

end matrix_vector_multiplication_l68_68629


namespace find_day_53_days_from_friday_l68_68482

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68482


namespace intersection_M_N_l68_68206

def M : Set ℝ :=
  {x | |x| ≤ 2}

def N : Set ℝ :=
  {x | Real.exp x ≥ 1}

theorem intersection_M_N :
  (M ∩ N) = {x | 0 ≤ x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l68_68206


namespace task2_X_alone_l68_68887

namespace TaskWork

variables (r_X r_Y r_Z : ℝ)

-- Task 1 conditions
axiom task1_XY : r_X + r_Y = 1 / 4
axiom task1_YZ : r_Y + r_Z = 1 / 6
axiom task1_XZ : r_X + r_Z = 1 / 3

-- Task 2 condition
axiom task2_XYZ : r_X + r_Y + r_Z = 1 / 2

-- Theorem to be proven
theorem task2_X_alone : 1 / r_X = 4.8 :=
sorry

end TaskWork

end task2_X_alone_l68_68887


namespace allen_blocks_l68_68052

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l68_68052


namespace cube_root_one_eighth_sqrt_of_sqrt_square_neg_six_l68_68101

theorem cube_root_one_eighth : ∃ x : ℝ, x^3 = 1 / 8 ∧ x = 1 / 2 := 
by
  use 1 / 2
  split
  · norm_num
  · norm_num

theorem sqrt_of_sqrt_square_neg_six : ∃ x : ℝ, x^2 = 36 ∧ (x = 6 ∨ x = -6) := 
by
  use 6
  split
  · norm_num
  · left
    norm_num

end cube_root_one_eighth_sqrt_of_sqrt_square_neg_six_l68_68101


namespace cost_of_fish_is_80_l68_68201

theorem cost_of_fish_is_80
  (cost_of_fish_per_kg : ℝ)
  (h_cost_of_fish : cost_of_fish_per_kg = 80)
  (cost_of_f_purchases : ℝ)
  (cost_of_f_purchases = 530)
  (amount_of_fish_first_case : ℝ)
  (amount_of_fish_first_case = 4)
  (amount_of_pork_first_case : ℝ)
  (amount_of_pork_first_case = 2)
  (cost_of_s_purchases : ℝ)
  (cost_of_s_purchases = 875)
  (amount_of_pork_second_case : ℝ)
  (amount_of_pork_second_case = 3) :
  cost_of_fish_per_kg = 80 := 
  sorry

end cost_of_fish_is_80_l68_68201


namespace fifty_three_days_from_friday_is_tuesday_l68_68470

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68470


namespace probability_two_non_standard_most_probable_non_standard_probability_most_probable_l68_68058

theorem probability_two_non_standard (n : ℕ) (p : ℚ) (q : ℚ) (h1 : n = 30) (h2 : p = 0.04) (h3 : q = 1 - p) : 
  (nat.choose 30 2) * (p^2) * (q^28) = 0.202 := 
sorry

theorem most_probable_non_standard (n : ℕ) (p : ℚ) (h1 : n = 30) (h2 : p = 0.04) : 
  (n * p + p).floor = 1 := 
sorry

theorem probability_most_probable (n : ℕ) (p : ℚ) (q : ℚ) (h1 : n = 30) (h2 : p = 0.04) (h3 : q = 1 - p) : 
  (nat.choose 30 1) * (p^1) * (q^29) = 0.305 := 
sorry

end probability_two_non_standard_most_probable_non_standard_probability_most_probable_l68_68058


namespace M_eq_N_l68_68790

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12 * m + 8 * n + 4 * l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20 * p + 16 * q + 12 * r}

theorem M_eq_N : M = N :=
by
  sorry

end M_eq_N_l68_68790


namespace no_Mem_is_Veen_l68_68716

-- Assuming definitions for Mem, En, and Veen as sets
variables (Mem En Veen : Type)
variables (M : Mem → En) -- All Mems are Ens
variables (N : ¬ ∃ e : En, Veen e) -- No Ens are Veens

theorem no_Mem_is_Veen
: ∀ (m : Mem), ¬ Veen (M m) :=
by 
  intro m
  apply N
  existsi (M m)
  sorry

end no_Mem_is_Veen_l68_68716


namespace relationship_among_x_y_z_l68_68674

-- Define the variables x, y, z according to the given conditions
def x : ℝ := 2^(0.5)
def y : ℝ := Real.log 2 / Real.log 5
def z : ℝ := Real.log 0.7 / Real.log 5

-- Prove the relationship among x, y, and z
theorem relationship_among_x_y_z : z < y ∧ y < x := by
  -- We need to insert the detailed proof here, but it will be skipped for this task
  sorry

end relationship_among_x_y_z_l68_68674


namespace image_of_point1_preimage_of_point2_image_l68_68705

-- Define the mapping
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Define the points
def point1 := (-2, 3)
def image1 := (1, -6)

noncomputable def point2_image := (2, -3)
noncomputable def pre_image_points := [(-1, 3), (3, -1)]

-- Prove the image of (-2, 3) under f is (1, -6)
theorem image_of_point1 : f point1.1 point1.2 = image1 := 
sorry

-- Prove the pre-image of (2, -3) under f is (-1, 3) or (3, -1)
theorem preimage_of_point2_image : 
  let (p1, p2) := pre_image_points in
  f p1.1 p1.2 = point2_image ∨ f p2.1 p2.2 = point2_image :=
sorry

end image_of_point1_preimage_of_point2_image_l68_68705


namespace days_from_friday_l68_68504

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68504


namespace triangle_area_proof_l68_68400

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end triangle_area_proof_l68_68400


namespace discount_rate_l68_68600

theorem discount_rate (cost_price marked_price desired_profit_margin selling_price : ℝ)
  (h1 : cost_price = 160)
  (h2 : marked_price = 240)
  (h3 : desired_profit_margin = 0.2)
  (h4 : selling_price = cost_price * (1 + desired_profit_margin)) :
  marked_price * (1 - ((marked_price - selling_price) / marked_price)) = selling_price :=
by
  sorry

end discount_rate_l68_68600


namespace number_of_valid_six_tuples_l68_68287

def is_valid_six_tuple (p : ℕ) (a b c d e f : ℕ) : Prop :=
  a + b + c + d + e + f = 3 * p ∧
  (a + b) % (c + d) = 0 ∧
  (b + c) % (d + e) = 0 ∧
  (c + d) % (e + f) = 0 ∧
  (d + e) % (f + a) = 0 ∧
  (e + f) % (a + b) = 0

theorem number_of_valid_six_tuples (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : 
  ∃! n, n = p + 2 ∧ ∀ (a b c d e f : ℕ), is_valid_six_tuple p a b c d e f → n = p + 2 :=
sorry

end number_of_valid_six_tuples_l68_68287


namespace T_lt_S_div_2_l68_68334

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68334


namespace continuous_function_form_l68_68135

noncomputable def f (t : ℝ) : ℝ := sorry

theorem continuous_function_form (f : ℝ → ℝ) (h1 : f 0 = -1 / 2) (h2 : ∀ x y, f (x + y) ≥ f x + f y + f (x * y) + 1) :
  ∃ (a : ℝ), ∀ x, f x = 1 / 2 + a * x + (a/2) * x ^ 2 := sorry

end continuous_function_form_l68_68135


namespace zero_in_interval_l68_68416

noncomputable def f (x : ℝ) : ℝ := x^3 - (1 / 2)^(x - 2)

theorem zero_in_interval : ∃ x ∈ (1, 2), f x = 0 :=
begin
  -- Define the function
  have h : continuous f := sorry, -- Assume continuity 
  -- Evaluate the function at given points
  have f1 : f 1 < 0 := by norm_num,
  have f2 : f 2 > 0 := by norm_num,
  -- Apply the intermediate value theorem (IVT)
  exact IVT h f1 f2,
end

end zero_in_interval_l68_68416


namespace four_digit_palindrome_square_count_l68_68993

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68993


namespace find_center_of_circle_l68_68910

-- Given conditions about the circle intersecting the coordinate axes.
variables (a b c d : ℝ)
(def A := (a,0)) (def B := (b,0)) (def C := (0,c)) (def D := (0,d))

-- The center of the circle based on the given conditions
def center_of_circle (a b c d : ℝ) : ℝ × ℝ := ((a + b) / 2, (c + d) / 2)

-- The theorem to prove
theorem find_center_of_circle (a b c d : ℝ) :
  ∃ (x y : ℝ), (x, y) = center_of_circle a b c d :=
by {
  use ((a + b) / 2, (c + d) / 2),
  exact ⟨rfl, rfl⟩,
}

end find_center_of_circle_l68_68910


namespace rope_segments_l68_68636

theorem rope_segments (total_length : ℝ) (n : ℕ) (h1 : total_length = 3) (h2 : n = 7) :
  (∃ segment_fraction : ℝ, segment_fraction = 1 / n ∧
   ∃ segment_length : ℝ, segment_length = total_length / n) :=
sorry

end rope_segments_l68_68636


namespace solve_c_is_one_l68_68768

theorem solve_c_is_one 
  (n : ℕ) 
  (a : ℝ) 
  (b : Fin n → ℝ) 
  (c : Fin n → ℝ)
  (h : ∀ x : ℝ, x^(2 * n) + a * x^(2 * n - 1) + a * x^(2 * n - 2) + ... + a * x + 1 = 
      ∏ i, (x^2 + (b i) * x + (c i))) : 
  ∀ i, c i = 1 :=
sorry

end solve_c_is_one_l68_68768


namespace problem_part_1_problem_part_2_l68_68713

open Real

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := abs (x + 2) - abs (x - 2) + m
noncomputable def g (x : ℝ) (m : ℝ) : ℝ := f x m - x

theorem problem_part_1 :
  ∀ x : ℝ, (1 : ℝ) → f x 1 ≥ 0 ↔ x ≥ -1 / 2 :=
by sorry

theorem problem_part_2 :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ g x1 m = 0 ∧ g x2 m = 0 ∧ g x3 m = 0) ↔ -2 < m ∧ m < 2 :=
by sorry

end problem_part_1_problem_part_2_l68_68713


namespace johns_contribution_l68_68735

theorem johns_contribution (A : ℝ) (J : ℝ) : 
  (1.7 * A = 85) ∧ ((5 * A + J) / 6 = 85) → J = 260 := 
by
  sorry

end johns_contribution_l68_68735


namespace total_number_of_ways_to_form_pairs_l68_68355

theorem total_number_of_ways_to_form_pairs (n : Nat) (h : 0 < n) :
  let total_ways := ((3 * n)! * (3 * n)!) / (2^(2 * n) * (n! * n! * n!))
  total_ways = ((3 * n)! * (3 * n)!) / (2^(2 * n) * (n!)^3) :=
by
  sorry

end total_number_of_ways_to_form_pairs_l68_68355


namespace day_53_days_from_friday_l68_68448

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68448


namespace find_day_53_days_from_friday_l68_68481

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68481


namespace probability_third_smallest_is_4_l68_68799

theorem probability_third_smallest_is_4 :
  (∃ S : Finset ℕ, S.card = 9 ∧ S ⊆ Finset.range 16 ∧ 4 ∈ S ∧ 
    ∀ T : Finset ℕ, T.card = 3 ∧ T ⊆ S → 4 ∈ T → (T.erase 4).min' (by simp) = 3) → 
  ((Finset.card {S : Finset ℕ | S.card = 9 ∧ S ⊆ Finset.range 16 ∧ 4 ∈ S ∧ 
    ∀ T : Finset ℕ, T.card = 3 ∧ T ⊆ S → 4 ∈ T → (T.erase 4).min' (by simp) = 3 }).to_real / 
  (nat.choose 15 9).to_real) = (2 / 15 : ℝ) := 
sorry

end probability_third_smallest_is_4_l68_68799


namespace three_digit_not_multiple_4_or_6_l68_68192

theorem three_digit_not_multiple_4_or_6 :
  let num_total := 999 - 100 + 1,
      num_div_4 := (996 / 4).to_nat - (100 / 4).to_nat + 1,
      num_div_6 := (996 / 6).to_nat - (102 / 6).to_nat + 1,
      num_div_12 := (996 / 12).to_nat - (108 / 12).to_nat + 1,
      num_div_4_or_6 := num_div_4 + num_div_6 - num_div_12 in
  num_total - num_div_4_or_6 = 600 :=
by sorry

end three_digit_not_multiple_4_or_6_l68_68192


namespace largest_angle_of_triangle_l68_68594

noncomputable def triangle_sides := (6 : ℝ, 7 : ℝ, 8 : ℝ)

theorem largest_angle_of_triangle :
  ∃ (C : ℝ), 
    (C ≈ 76) ∧ 
    let (a, b, c) := triangle_sides in 
    a + 1 = b ∧ 
    b + 1 = c ∧ 
    a = 6 ∧ 
    a + b > c ∧ 
    a + c > b ∧ 
    b + c > a ∧ 
    C = Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b)) :=
by 
  let (a, b, c) := triangle_sides
  dsimp [triangle_sides]
  sorry

end largest_angle_of_triangle_l68_68594


namespace find_f_neg_half_l68_68158

def is_odd_function {α β : Type*} [AddGroup α] [Neg β] (f : α → β) : Prop :=
  ∀ x : α, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2 else 0

theorem find_f_neg_half (f_odd : is_odd_function f) (f_pos : ∀ x > 0, f x = Real.log x / Real.log 2) :
  f (-1/2) = 1 := by
  sorry

end find_f_neg_half_l68_68158


namespace arithmetic_sequence_value_at_15_l68_68759

axiom arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n : ℕ, ∃ a1 d : ℤ, a n = a1 + (n - 1) * d

theorem arithmetic_sequence_value_at_15
  (a : ℕ → ℤ)
  (ha : arithmetic_sequence a)
  (h5 : a 5 = 5)
  (h10 : a 10 = 15) :
  a 15 = 25 :=
by sorry

end arithmetic_sequence_value_at_15_l68_68759


namespace Tn_lt_Sn_div_2_l68_68316

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68316


namespace train_length_proof_l68_68593

noncomputable def train_length (speed_kmph : ℕ) (time_seconds : ℕ) : ℝ :=
  (speed_kmph * 1000 / 3600) * time_seconds

theorem train_length_proof : train_length 100 18 = 500.04 :=
  sorry

end train_length_proof_l68_68593


namespace find_n_l68_68184

theorem find_n (n : ℕ) 
  (h1 : (1 + 1) + (1 + 1)^2 + (1 + 1)^3 + ⋯ + (1 + 1)^n = n + 1013 + 1013) 
  (h2 : 2 * (2^n - 1) = n + 1013) : 
  n = 9 := 
  by sorry

end find_n_l68_68184


namespace line_equation_l68_68685

theorem line_equation (a b : ℝ)
(h1 : a * -1 + b * 2 = 0) 
(h2 : a = b) :
((a = 1 ∧ b = -1) ∨ (a = 2 ∧ b = -1)) := 
by
  sorry

end line_equation_l68_68685


namespace four_digit_perfect_square_palindrome_count_l68_68975

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68975


namespace part1_part2_part3_l68_68167

noncomputable theory

-- Given points (-m, 0) and (3m, 0) lie on y = ax^2 + bx + 3 where a ≠ 0:
-- (1) When m = -1, a = -1 and b = -2
theorem part1 (a b : ℝ) (a_ne_zero : a ≠ 0) (h1 : a * 1 * 1 + b * 1 + 3 = 0) (h2 : a * 9 + b * -3 + 3 = 0) : 
  a = -1 ∧ b = -2 :=
sorry

-- (2) If the quadratic function passes through the point A(n, 3), given -2 < m < -1, then -4 < n < -2
theorem part2 (a b : ℝ) (a_ne_zero : a ≠ 0) (n : ℝ) (h1 : a * 1 * 1 + b * 1 + 3 = 0) (h2 : a * 9 + b * -3 + 3 = 0) 
  (h3 : -2 < m ∧ m < -1) (h4 : a * n^2 + b * n + 3 = 3) : -4 < n ∧ n < -2 :=
sorry

-- (3) Prove that b^2 + 4a = 0
theorem part3 (a b m : ℝ) (a_ne_zero : a ≠ 0) (h1 : a * m * m + b * m + 3 = 0) (h2 : a * 9 * m * m + 3 * b * m + 3 = 0) :
  b^2 + 4 * a = 0 :=
sorry

end part1_part2_part3_l68_68167


namespace tn_lt_sn_div_2_l68_68308

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68308


namespace fold_decreases_perimeter_l68_68036

-- Define the perimeter function
def perimeter (P : Polygon) : ℝ := sorry

-- Define the folding function
def fold (P : Polygon) (A B : Point) : Polygon := sorry

-- Main theorem statement
theorem fold_decreases_perimeter (P : Polygon) (A B : Point) :
  A ∈ boundary P ∧ B ∈ boundary P →
  perimeter (fold P A B) < perimeter P :=
sorry

end fold_decreases_perimeter_l68_68036


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68463

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68463


namespace composite_sum_l68_68599

-- Define the first three positive composite integers
def first_three_composites : List ℕ := [4, 6, 8]

-- Define the next two positive composite integers
def next_two_composites : List ℕ := [9, 10]

-- Define the products
def product_first_three : ℕ := first_three_composites.product
def product_next_two : ℕ := next_two_composites.product

-- State the theorem
theorem composite_sum :
  product_first_three + product_next_two = 282 :=
by
  sorry

end composite_sum_l68_68599


namespace income_is_12000_l68_68924

theorem income_is_12000 (P : ℝ) : (P * 1.02 = 12240) → (P = 12000) :=
by
  intro h
  sorry

end income_is_12000_l68_68924


namespace greatest_prime_factor_of_144_l68_68523

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l68_68523


namespace square_partition_four_parts_l68_68764

-- Define the problem of cutting a square into four parts touching each other
theorem square_partition_four_parts :
  ∃ (parts : set (set ℝ)), 
    (∀ p ∈ parts, ∃ (a b c : set ℝ), a ≠ p ∧ b ≠ p ∧ c ≠ p ∧ 
      (p ∩ a).nonempty ∧ (p ∩ b).nonempty ∧ (p ∩ c).nonempty) ∧
    (parts.card = 4) ∧
    (∀ p ∈ parts, is_square p) :=
sorry

end square_partition_four_parts_l68_68764


namespace part1_part2_l68_68708

-- Define the piecewise function f
def f (x : ℝ) : ℝ :=
if x ≤ 0 then x^2 + 1 else -2 * x

-- Proof of the first part: f(f(-2)) = -10
theorem part1 : f (f (-2)) = -10 := by
  sorry

-- Proof of the second part: the value of x when f x = 10
theorem part2 (x : ℝ) : f x = 10 → x = -3 := by
  intro h
  sorry

end part1_part2_l68_68708


namespace percentage_increase_Anthony_to_Mabel_l68_68800

noncomputable def transactions_Mabel : ℕ := 90
noncomputable def transactions_Jade : ℕ := 80
noncomputable def difference_Jade_Cal : ℕ := 14
noncomputable def fraction_Anthony_Cal : ℚ := 2 / 3

theorem percentage_increase_Anthony_to_Mabel :
  let transactions_Cal := transactions_Jade - difference_Jade_Cal in
  let transactions_Anthony := transactions_Cal * (3/2 : ℚ) in
  let P := (transactions_Anthony - transactions_Mabel) / transactions_Mabel * 100 in
  P = 10 :=
by
  sorry

end percentage_increase_Anthony_to_Mabel_l68_68800


namespace smallest_positive_angle_l68_68630

open Real

theorem smallest_positive_angle :
  ∃ x : ℝ, x > 0 ∧ x < 90 ∧ tan (4 * x * degree) = (cos (x * degree) - sin (x * degree)) / (cos (x * degree) + sin (x * degree)) ∧ x = 9 :=
sorry

end smallest_positive_angle_l68_68630


namespace maximum_a_value_condition_l68_68730

theorem maximum_a_value_condition (x a : ℝ) :
  (∀ x, (x^2 - 2 * x - 3 > 0 → x < a)) ↔ a ≤ -1 :=
by sorry

end maximum_a_value_condition_l68_68730


namespace find_k_l68_68190

def vector_perp (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)
  
theorem find_k :
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (-1, 2)
  ∀ k : ℝ, vector_perp (vector_add (2 * a) b) (k, 1) → k = -4 :=
begin
  intros a b k h,
  sorry,
end

end find_k_l68_68190


namespace find_cd_l68_68531

def g (c d x : ℝ) := c * x^3 - 7 * x^2 + d * x - 4

theorem find_cd : ∃ c d : ℝ, (g c d 2 = -4) ∧ (g c d (-1) = -22) ∧ (c = 19/3) ∧ (d = -8/3) := 
by
  sorry

end find_cd_l68_68531


namespace min_radius_l68_68694

theorem min_radius (r : ℝ) (P : ℝ × ℝ) :
  let O := (0, 0)
  let M := (1, 0)
  ∃ (P : ℝ × ℝ), (P.1 - 5)^2 + (P.2 - 4)^2 = r^2 ∧ r > 0 ∧ (P.1^2 + P.2^2 = 2 * ((P.1 - 1)^2 + P.2^2)) →
  r = 5 - real.sqrt 2 :=
sorry

end min_radius_l68_68694


namespace transformed_roots_polynomial_l68_68778

noncomputable def transformed_polynomial (k : ℝ) : Polynomial ℝ :=
  k * Polynomial.X^2 + 3 * Polynomial.X + 1

theorem transformed_roots_polynomial (p q r s k : ℝ)
  (hpqr : Polynomial.eval p (Polynomial.X^4 - 3 * Polynomial.X^2 - Polynomial.C k) = 0)
  (hqs : Polynomial.eval q (Polynomial.X^4 - 3 * Polynomial.X^2 - Polynomial.C k) = 0)
  (har : Polynomial.eval r (Polynomial.X^4 - 3 * Polynomial.X^2 - Polynomial.C k) = 0)
  (hps : Polynomial.eval s (Polynomial.X^4 - 3 * Polynomial.X^2 - Polynomial.C k) = 0)
  (vieta : p + q + r + s = 0) :
  ∀ (x : ℝ), Polynomial.eval x (transformed_polynomial k) = 0 ↔
    x = -1 / p^2 ∨ x = -1 / q^2 ∨ x = -1 / r^2 ∨ x = -1 / s^2 :=
by
  sorry

end transformed_roots_polynomial_l68_68778


namespace circle_through_points_and_line_circle_with_conditions_l68_68016

-- Problem (1)
theorem circle_through_points_and_line (A B : ℝ × ℝ) (l : ℝ → ℝ) :
  A = (5, 2) → B = (3, 2) → l = λ x, 2 * x - 3 → 
  ∃ (C : ℝ × ℝ) (r : ℝ), 
  C.1 = 4 ∧ C.2 = 5 ∧ r = sqrt 10 ∧ 
  ∀ (x y : ℝ), (x - 4)^2 + (y - 5)^2 = 10 :=
-- sorry, no proof required
sorry 

-- Problem (2)
theorem circle_with_conditions (A : ℝ × ℝ) (line_symmetry chord_intersection : ℝ → ℝ) (chord_length : ℝ) :
  A = (2, 3) → line_symmetry = λ x, -x / 2 →
  chord_intersection = λ x, x - 1 →
  chord_length = 2 * sqrt 2 →
  ∃ (C : ℝ × ℝ) (r : ℝ), 
  (C.1 = 6 ∧ C.2 = -3 ∧ r^2 = 52 ∨ 
   C.1 = 14 ∧ C.2 = -7 ∧ r^2 = 244) ∧ 
  ∀ (x y : ℝ), 
  (x - C.1)^2 + (y - C.2)^2 = r^2 :=
-- sorry, no proof required
sorry

end circle_through_points_and_line_circle_with_conditions_l68_68016


namespace which_point_is_in_fourth_quadrant_l68_68057

def point (x: ℝ) (y: ℝ) : Prop := x > 0 ∧ y < 0

theorem which_point_is_in_fourth_quadrant :
  point 5 (-4) :=
by {
  -- proofs for each condition can be added,
  sorry
}

end which_point_is_in_fourth_quadrant_l68_68057


namespace determine_k_l68_68641

theorem determine_k (k : ℝ) : 
  (2 * k * (-1/2) - 3 = -7 * 3) → k = 18 :=
by
  intro h
  sorry

end determine_k_l68_68641


namespace knight_reach_position_if_even_l68_68063

-- Definitions for knight's move characteristics and proof statement

def knight_can_move (n : ℕ) : Prop :=
  ∀ (i j : ℤ), ∃ (m k : ℤ), m ≠ k ∧ (abs (m - i) = n ∧ abs (k - j) = 1) ∨ (abs (m - i) = 1 ∧ abs (k - j) = n)

theorem knight_reach_position_if_even (n : ℕ) : 
  (∀ i j : ℕ, ∃ (a b : ℤ), knight_can_move (a - i) (b - j)) ↔ n % 2 = 0 :=
sorry

end knight_reach_position_if_even_l68_68063


namespace option_d_correct_l68_68536

variable (a b m n : ℝ)

theorem option_d_correct :
  6 * a + a ≠ 6 * a ^ 2 ∧
  -2 * a + 5 * b ≠ 3 * a * b ∧
  4 * m ^ 2 * n - 2 * m * n ^ 2 ≠ 2 * m * n ∧
  3 * a * b ^ 2 - 5 * b ^ 2 * a = -2 * a * b ^ 2 := by
  sorry

end option_d_correct_l68_68536


namespace find_constant_C_l68_68532

def polynomial_remainder (C : ℝ) (x : ℝ) : ℝ :=
  C * x^3 - 3 * x^2 + x - 1

theorem find_constant_C :
  (polynomial_remainder 2 (-1) = -7) → 2 = 2 :=
by
  sorry

end find_constant_C_l68_68532


namespace hyperbola_eccentricity_l68_68771

theorem hyperbola_eccentricity 
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (F1 F2 O P : EuclideanSpace ℝ (Fin 2))
  (hC : ∀ x y, (x - O.1)^2 / a^2 - (y - O.2)^2 / b^2 = 1) 
  (h_focus : ∃ c, F1.1 = -c ∧ F2.1 = c ∧ F1.2 = 0 ∧ F2.2 = 0 ∧ c^2 = a^2 + b^2)
  (h_perpendicular : ∃ x0 : ℝ, P = ⟨x0, (b / a) * x0⟩)
  (h_distance : (dist P F1) = real.sqrt 6 * (dist O P)) :
  ∃ e : ℝ, e = real.sqrt 3 := 
sorry

end hyperbola_eccentricity_l68_68771


namespace sum_of_possible_values_of_k_l68_68243

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l68_68243


namespace circles_intersect_l68_68640

theorem circles_intersect :
  ∀ x y : ℝ,
  (x^2 + y^2 - 4 = 0 ∧ x^2 + y^2 - 4x - 5 = 0) →
  (∃ d R r : ℝ,
    d = 2 ∧ R = 2 ∧ r = 3 ∧
    R + r = 5 ∧
    |R - r| = 1 ∧
    |R - r| < d ∧ d < R + r) :=
begin
  sorry -- Proof details go here
end

end circles_intersect_l68_68640


namespace days_from_friday_l68_68503

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68503


namespace sum_possible_values_k_l68_68256

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l68_68256


namespace Tn_lt_Sn_over_2_l68_68324

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68324


namespace pool_filling_time_l68_68886

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l68_68886


namespace wall_building_time_l68_68733

variables (f b c y : ℕ) 

theorem wall_building_time :
  (y = 2 * f * c / b) 
  ↔ 
  (f > 0 ∧ b > 0 ∧ c > 0 ∧ (f * b * y = 2 * b * c)) := 
sorry

end wall_building_time_l68_68733


namespace problem1_problem2_l68_68175

noncomputable def f (x a b : ℝ) : ℝ := (x + a) / (x + b)

-- Problem (1): Prove the inequality f(x-1) > 0 given b = 1.
theorem problem1 (a x : ℝ) : f (x - 1) a 1 > 0 := sorry

-- Problem (2): Prove the values of a and b such that the range of f(x) for x ∈ [-1, 2] is [5/4, 2].
theorem problem2 (a b : ℝ) (H₁ : f (-1) a b = 5 / 4) (H₂ : f 2 a b = 2) :
    (a = 3 ∧ b = 2) ∨ (a = -4 ∧ b = -3) := sorry

end problem1_problem2_l68_68175


namespace area_of_trapezoid_PQRS_l68_68260

-- Define the points P, Q, R, S
structure Point where
  x : ℝ
  y : ℝ

def P : Point := ⟨2, -1⟩
def Q : Point := ⟨2, 3⟩
def R : Point := ⟨7, 10⟩
def S : Point := ⟨7, 2⟩

-- Define the function to calculate the distance between two points
def distance (A B : Point) : ℝ :=
  Real.sqrt ((A.x - B.x) ^ 2 + (A.y - B.y) ^ 2)

-- Define the bases PQ and RS
def base_PQ : ℝ := distance P Q
def base_RS : ℝ := distance R S

-- Define the height of the trapezoid
def height : ℝ := (R.x - P.x).abs

-- Define the area of the trapezoid
def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

-- State the theorem we want to prove
theorem area_of_trapezoid_PQRS :
  area_trapezoid base_PQ base_RS height = 30 :=
by
  sorry

end area_of_trapezoid_PQRS_l68_68260


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68466

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68466


namespace Tn_lt_Sn_div_2_l68_68310

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68310


namespace root_difference_l68_68385

theorem root_difference (p : ℝ) (r s : ℝ) :
  (r + s = p) ∧ (r * s = (p^2 - 1) / 4) ∧ (r ≥ s) → r - s = 1 :=
by
  intro h
  sorry

end root_difference_l68_68385


namespace intersection_A_B_l68_68186

-- Define sets A and B
def A : Set ℕ := {0, 1, 2, 8}
def B : Set ℕ := {-1, 1, 6, 8}

-- Prove A ∩ B = {1, 8}
theorem intersection_A_B : A ∩ B = {1, 8} :=
by sorry

end intersection_A_B_l68_68186


namespace complex_point_in_second_quadrant_l68_68554

noncomputable theory

-- Definitions based on the conditions
def z : ℂ := complex.I * (1 + complex.I)

-- Corresponding point in the complex plane
def point_of_z (z : ℂ) : ℂ := ⟨-1, 1⟩

-- Define what it means to be in the second quadrant
def in_second_quadrant (p : ℂ) : Prop :=
  p.re < 0 ∧ p.im > 0

-- The theorem to be proven
theorem complex_point_in_second_quadrant : in_second_quadrant (point_of_z z) :=
by sorry

end complex_point_in_second_quadrant_l68_68554


namespace find_special_numbers_l68_68030

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end find_special_numbers_l68_68030


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68462

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68462


namespace green_pill_cost_l68_68601

namespace PillCost

-- Define the conditions as Lean definitions
def total_days : ℕ := 18
def total_cost : ℕ := 738
def cost_difference : ℕ := 2

-- Define a variable for the cost of a green pill
variable (x : ℝ)

-- Problem statement for proving the cost of the green pill
theorem green_pill_cost :
  (∑ i in finset.range total_days, (x + (x - cost_difference))) = total_cost
  → x = 21.5 :=
by
  intros h
  sorry
  
end PillCost

end green_pill_cost_l68_68601


namespace consumption_reduction_l68_68546

variable (P C : ℝ)

theorem consumption_reduction (h : P > 0 ∧ C > 0) : 
  (1.25 * P * (0.8 * C) = P * C) :=
by
  -- Conditions: original price P, original consumption C
  -- New price 1.25 * P, New consumption 0.8 * C
  sorry

end consumption_reduction_l68_68546


namespace pascal_fifth_number_l68_68000

def binom (n k : Nat) : Nat := Nat.choose n k

theorem pascal_fifth_number (n r : Nat) (h1 : n = 50) (h2 : r = 4) : binom n r = 230150 := by
  sorry

end pascal_fifth_number_l68_68000


namespace greatest_prime_factor_of_144_l68_68520

theorem greatest_prime_factor_of_144 : ∃ p, prime p ∧ p ∣ 144 ∧ (∀ q, prime q ∧ q ∣ 144 → q ≤ p) :=
sorry

end greatest_prime_factor_of_144_l68_68520


namespace contracting_schemes_l68_68621

theorem contracting_schemes :
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  (Nat.choose total_projects a_contracts) *
  (Nat.choose (total_projects - a_contracts) b_contracts) *
  (Nat.choose ((total_projects - a_contracts) - b_contracts) c_contracts) = 60 :=
by
  let total_projects := 6
  let a_contracts := 3
  let b_contracts := 2
  let c_contracts := 1
  sorry

end contracting_schemes_l68_68621


namespace Mathematics_Culture_Festival_max_k_l68_68041

theorem Mathematics_Culture_Festival_max_k (n : ℕ) (h : 980 ≤ n ∧ n < 990) :
  ∃ k : ℕ, k = 32 ∧
  (∀ t, ∃ S : Finset ℕ, S.card = k ∧
  (∀ a ∈ S, ∃ b ∈ S, a = b ∨ @disjoint (Finset ℕ) _ (Finset.singleton a) (Finset.singleton b))) :=
sorry

end Mathematics_Culture_Festival_max_k_l68_68041


namespace non_intersecting_matching_exists_l68_68356

open List

-- Assume given conditions
variables (n : Nat) (h₁ : n ≥ 1) (points : List ℝ × ℝ) 
          (h₂ : points.length = 2 * n) 
          (h₃ : ∀ p ∈ points, ∀ q ∈ points, ∀ r ∈ points, 
              p ≠ q → q ≠ r → r ≠ p → ¬ collinear p q r)
          (colors : List Color) (h₄ : colors.length = points.length)
          (h₅ : count colors Blue = n) (h₆ : count colors Red = n)

-- Define what it means for segments to be non-intersecting
def non_intersecting_segments (segments : List ((ℝ × ℝ) × (ℝ × ℝ))) : Prop := 
  ∀ (s1 s2 ∈ segments), s1 ≠ s2 → no_intersection s1 s2

-- Problem statement
theorem non_intersecting_matching_exists :
  ∃ segments : List ((ℝ × ℝ) × (ℝ × ℝ)), 
    segments.length = n ∧ 
    (∀ p ∈ points, ∃ s ∈ segments, p ∈ s) ∧ 
    (∀ (s : (ℝ × ℝ) × (ℝ × ℝ)) (hs : s ∈ segments), 
      color_of (s.1) = Blue ∧ color_of (s.2) = Red) ∧
    non_intersecting_segments segments :=
sorry

end non_intersecting_matching_exists_l68_68356


namespace select_AEDs_l68_68408

theorem select_AEDs
  (n : ℕ) (r : ℕ) (options : list ℕ)
  (h_n : n = 5)
  (h_r_A : r = 3)
  (h_r_B : r = 3)
  (h_A : option_A = (nat.factorial n) / ((nat.factorial r) * (nat.factorial (n - r))) = 10)
  (h_B : option_B = (nat.factorial n) / (nat.factorial (n - r)) = 60)
  (h_D : ∃ k₁ k₂ k₃, k₁ + k₂ + k₃ = n ∧ k₁ ≥ 1 ∧ k₂ ≥ 1 ∧ k₃ ≥ 1 ∧
    (options.count k₁ = k₁ ∨ options.count k₁ = k₂ ∨ options.count k₁ = k₃) ∧
    (options.count k₂ = k₁ ∨ options.count k₂ = k₂ ∨ options.count k₂ = k₃) ∧
    (options.count k₃ = k₁ ∨ options.count k₃ = k₂ ∨ options.count k₃ = k₃) ∧
    option_D = (nat.choose n k₃) * (nat.factorial 3) + ((nat.choose n 1) * (nat.choose (n - 1) 2) * (nat.factorial 3) / 2) = 150) :
  (h_A ∧ h_B ∧ h_D) := by
  sorry

end select_AEDs_l68_68408


namespace groceries_cost_40_l68_68767

-- Define the initial amount
def initial_amount : ℤ := 100

-- Define the fraction given to his sister
def fraction_to_sister : ℝ := 1 / 4

-- Define the remaining amount
def remaining_amount : ℤ := 35

-- Define John's total expenditure after giving to his sister
def amount_after_giving_to_sister (initial : ℤ) (fraction : ℝ) : ℝ :=
  (initial : ℝ) - (fraction * (initial : ℝ))

-- Define the cost of groceries
def cost_of_groceries (initial : ℤ) (fraction : ℝ) (remaining : ℤ) : ℝ :=
  amount_after_giving_to_sister initial fraction - (remaining : ℝ)

-- Prove the cost of groceries is 40
theorem groceries_cost_40 : cost_of_groceries initial_amount fraction_to_sister remaining_amount = 40 :=
by
  sorry

end groceries_cost_40_l68_68767


namespace sum_of_elements_of_S_l68_68292

def is_repeating_decimal (r : ℚ) (a b : ℕ) : Prop :=
  r = (10 * a + b) / 99 ∧ a ≠ b ∧ a < 10 ∧ b < 10

def S : Set ℚ := {r | ∃ a b : ℕ, is_repeating_decimal r a b}

theorem sum_of_elements_of_S : ∑ r in S, r = 45 :=
  sorry

end sum_of_elements_of_S_l68_68292


namespace evaluate_expression_l68_68643

theorem evaluate_expression (x y z : ℕ) (hx : x = 5) (hy : y = 10) (hz : z = 3) : z * (y - 2 * x) = 0 := by
  sorry

end evaluate_expression_l68_68643


namespace probability_a_eq_1_l68_68156

theorem probability_a_eq_1 (a b c : ℕ) (h1 : a + b + c = 6) (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (∑ x in finset.range 6, ∑ y in finset.range (6 - x), 
      if x + y + (6 - (x + y)) = 6 ∧ x > 0 ∧ y > 0 ∧ (6 - (x + y)) > 0 then 1 else 0) = 10 → 
  (∑ y in finset.range 5, 
      if 1 + y + (5 - y) = 6 ∧ 1 > 0 ∧ y > 0 ∧ (5 - y) > 0 then 1 else 0) = 4 →
  (4 / 10 : ℚ) = (2 / 5 : ℚ) :=
by
  sorry

end probability_a_eq_1_l68_68156


namespace right_triangle_log_hypotenuse_l68_68075

def log_base (b x : ℝ) := Real.log x / Real.log b

theorem right_triangle_log_hypotenuse :
  let leg1 := log_base 8 27 in
  let leg2 := log_base 4 81 in
  let h := Real.sqrt (leg1^2 + leg2^2) in
  8^h = 3^((3/2) * Real.sqrt 20) :=
by
  sorry

end right_triangle_log_hypotenuse_l68_68075


namespace problem_proof_l68_68142

-- Define the sequence a_n and sum S_n, with given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom a_2 : a 2 = 3
axiom sum_relation : ∀ n : ℕ, n > 0 → n * a n + n = 2 * S n

-- Define b_n based on a_n and find sum S_20
def b (n : ℕ) : ℝ := (2:ℝ)^(a n) * Real.cos ((n^2 : ℕ) * Real.pi)

-- The main theorem combining both parts of the problem
theorem problem_proof : 
  (∀ n : ℕ, n > 0 → a (n + 1) - a n = a (2) - a 1) ∧ 
  (S 20 = ∑ i in Finset.range 20 + 1, b i) :=
by
  sorry

end problem_proof_l68_68142


namespace find_difference_l68_68198

theorem find_difference (x y : ℝ) (h1 : x + y = 8) (h2 : x^2 - y^2 = 24) : x - y = 3 := 
by
  sorry

end find_difference_l68_68198


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68458

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68458


namespace monkey_tree_height_l68_68029

-- Define the height of the tree and the monkey's behavior.
def monkey_climbing_tree(height : ℕ) : Prop :=
  ∃ (hours : ℕ), hours = 15 ∧ 
  ∃ (climb : ℕ → ℕ), (∀ t < 14, climb t = 1 ∧ climb 14 = 3) ∧
  height = (∑ t in finset.range 14, climb t) + climb 14

theorem monkey_tree_height : monkey_climbing_tree 17 :=
by {
  use 15,      -- It takes 15 hours
  split,
  exact rfl,   -- Prove hours = 15
  use λ t, if t < 14 then 1 else 3,
  split,
  intros t ht,
  split,
  -- Prove that for the first 14 hours, the monkey makes a net progress of 1 ft per hour
  { simp only [if_pos ht] },
  -- Prove that on the 15th hour, the monkey makes a jump of 3 ft
  { simp only [if_neg (not_lt_of_gt (nat.succ_pos 14))], },
  -- Prove the total height
  {
    rw [finset.sum_ite_eq', finset.sum_const, finset.card_range, finset.sum_singleton],
    simp,
  }
}

end monkey_tree_height_l68_68029


namespace day_53_days_from_friday_l68_68447

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68447


namespace discriminant_nonnegative_l68_68633

theorem discriminant_nonnegative {x : ℤ} (a : ℝ) (h₁ : x^2 * (49 - 40 * x^2) ≥ 0) :
  a = 0 ∨ a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -5/2 := sorry

end discriminant_nonnegative_l68_68633


namespace sufficient_but_not_necessary_condition_l68_68673

theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ≥ 1) → (|x + 1| + |x - 1| = 2 * |x|) ∧ ¬((x ≥ 1) ↔ (|x + 1| + |x - 1| = 2 * |x|)) := by
  sorry

end sufficient_but_not_necessary_condition_l68_68673


namespace four_digit_palindrome_square_count_l68_68998

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68998


namespace total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l68_68560

def chocolate_sold : ℕ := 6 + 7 + 4 + 8 + 9 + 10 + 5
def vanilla_sold : ℕ := 4 + 5 + 3 + 7 + 6 + 8 + 4
def strawberry_sold : ℕ := 3 + 2 + 6 + 4 + 5 + 7 + 4

theorem total_chocolate_sold : chocolate_sold = 49 :=
by
  unfold chocolate_sold
  rfl

theorem total_vanilla_sold : vanilla_sold = 37 :=
by
  unfold vanilla_sold
  rfl

theorem total_strawberry_sold : strawberry_sold = 31 :=
by
  unfold strawberry_sold
  rfl

end total_chocolate_sold_total_vanilla_sold_total_strawberry_sold_l68_68560


namespace questionnaire_visitors_l68_68415

theorem questionnaire_visitors (V E : ℕ) (H1 : 140 = V - E) 
  (H2 : E = (3 * V) / 4) : V = 560 :=
by
  sorry

end questionnaire_visitors_l68_68415


namespace tan_beta_when_alpha_pi_over_6_max_tan_beta_l68_68675

theorem tan_beta_when_alpha_pi_over_6 (α β : ℝ) (h_alpha : α = π / 6) 
   (h_cond : sin β / sin α = cos (α + β)) : tan β = sqrt 3 / 5 := sorry

theorem max_tan_beta (α β : ℝ) (h_cond : sin β / sin α = cos (α + β)) : 
  ∃ β, (tan β = sqrt 2 / 4) := sorry

end tan_beta_when_alpha_pi_over_6_max_tan_beta_l68_68675


namespace perpendicular_lines_sin_2alpha_l68_68692

theorem perpendicular_lines_sin_2alpha (α : ℝ) 
  (l1 : ∀ (x y : ℝ), x * (Real.sin α) + y - 1 = 0) 
  (l2 : ∀ (x y : ℝ), x - 3 * y * Real.cos α + 1 = 0) 
  (perp : ∀ (x1 y1 x2 y2 : ℝ), 
        (x1 * (Real.sin α) + y1 - 1 = 0) ∧ 
        (x2 - 3 * y2 * Real.cos α + 1 = 0) → 
        ((-Real.sin α) * (1 / (3 * Real.cos α)) = -1)) :
  Real.sin (2 * α) = (3/5) :=
sorry

end perpendicular_lines_sin_2alpha_l68_68692


namespace isosceles_triangle_perimeter_l68_68147

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l68_68147


namespace ellipse_standard_eq_and_mn_value_l68_68688

theorem ellipse_standard_eq_and_mn_value (a b c m n : ℝ) (h1 : 2 * a = 8) (h2 : a = 4)
  (h3 : c / a = sqrt 7 / 4) (h4 : a^2 = b^2 + c^2) (h5 : b^2 = 9)
  (h6 : ∀ k : ℝ, (9 + 16 * k^2) * x^2 - 32 * k^2 * m * x + 16 * k^2 * m^2 - 144 = 0)
  (h7 : k ≠ 0) (h8 : x1 + x2 = 32 * k^2 * m / (9 + 16 * k^2))
  (h9 : x1 * x2 = (16 * k^2 * m^2 - 144) / (9 + 16 * k^2))
  (h10 : k1 + k2 = 0)
  (h11 : 2 * x1 * x2 - (m + n) * (x1 + x2) + 2 * mn = 0) : 
  (∃ (a' b' : ℝ), (a' * b' = 16 ∧ (x / 16 + y / 9 = 1))) :=
by
  use [4, 3]
  split
  exact (4 * 3 = 12)

  sorry

end ellipse_standard_eq_and_mn_value_l68_68688


namespace smallest_n_S_n_l68_68140

noncomputable theory
open BigOperators

def a_seq (a : ℕ → ℝ) : Prop :=
  a 1 = 9 ∧ ∀ n : ℕ, 0 < n → 3 * a (n + 1) + a n = 4

def S_n (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

theorem smallest_n_S_n (a : ℕ → ℝ) (h : a_seq a) :
  ∃ n : ℕ, 0 < n ∧ |S_n a n - n - 6| < 1/125 ∧ n = 7 :=
sorry

end smallest_n_S_n_l68_68140


namespace magnitude_of_z_sqrt2_l68_68161

open Complex

noncomputable def magnitude_of_z (z : ℂ) := Complex.abs z

theorem magnitude_of_z_sqrt2 (z : ℂ) (h : (-1 + Complex.i) * z = (1 + Complex.i) * (1 + Complex.i)) : magnitude_of_z z = Real.sqrt 2 :=
by
  sorry

end magnitude_of_z_sqrt2_l68_68161


namespace marbles_left_percentage_l68_68122

variable (M : ℝ)
variable (initial : M > 0)
variable (pedroShare : PedroShare = 0.20 * M)
variable (remainingAfterPedro : RemainingAfterPedro = 0.80 * M)
variable (ebonyShare : EbonyShare = 0.10 * RemainingAfterPedro)
variable (remainingAfterEbony : RemainingAfterEbony = 0.90 * RemainingAfterPedro)
variable (jimmyShare : JimmyShare = 0.25 * RemainingAfterEbony)
variable (remainingAfterJimmy : RemainingAfterJimmy = 0.75 * RemainingAfterEbony)

theorem marbles_left_percentage : remainingAfterJimmy / M * 100 = 54 := 
by sorry

end marbles_left_percentage_l68_68122


namespace ball_hits_ground_in_3_seconds_l68_68608

noncomputable def ball_height (t : ℝ) : ℝ := -16 * t^2 - 32 * t + 240

theorem ball_hits_ground_in_3_seconds :
  ∃ t : ℝ, ball_height t = 0 ∧ t = 3 :=
sorry

end ball_hits_ground_in_3_seconds_l68_68608


namespace max_homework_ratio_l68_68364

theorem max_homework_ratio 
  (H : ℕ) -- time spent on history tasks
  (biology_time : ℕ)
  (total_homework_time : ℕ)
  (geography_time : ℕ)
  (history_geography_relation : geography_time = 3 * H)
  (total_time_relation : total_homework_time = 180)
  (biology_time_known : biology_time = 20)
  (sum_time_relation : H + geography_time + biology_time = total_homework_time) :
  H / biology_time = 2 :=
by
  sorry

end max_homework_ratio_l68_68364


namespace trajectory_of_G_is_ellipse_l68_68148

noncomputable def trajectory (G : Point) : Prop :=
  ∃ P Q : Point,
  (P ∈ M) ∧
  (N = (√7, 0)) ∧
  (Q ∈ segment N P) ∧
  (vector.same_dir (vector NP) (2 * vector NQ)) ∧
  (G ∈ segment M P) ∧
  (vector.dot_product (vector GQ) (vector NP) = 0)

theorem trajectory_of_G_is_ellipse :
  ∀ G : Point, trajectory G →
  ∃ a b : ℝ, a = 4 ∧ b = 3 ∧
  equation_of_trajectory G = 
  (λ x y, x^2 / a^2 + y^2 / b^2 = 1) :=
by sorry

end trajectory_of_G_is_ellipse_l68_68148


namespace james_received_stickers_l68_68271

theorem james_received_stickers (initial_stickers given_away final_stickers received_stickers : ℕ) 
  (h_initial : initial_stickers = 269)
  (h_given : given_away = 48)
  (h_final : final_stickers = 423)
  (h_total_before_giving_away : initial_stickers + received_stickers = given_away + final_stickers) :
  received_stickers = 202 :=
by
  sorry

end james_received_stickers_l68_68271


namespace T_lt_S_div_2_l68_68338

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68338


namespace parabola_directrix_l68_68387

theorem parabola_directrix (a : ℝ) : 
  (∃ y, (y ^ 2 = 4 * a * (-2))) → a = 2 :=
by
  sorry

end parabola_directrix_l68_68387


namespace unique_four_digit_palindromic_square_l68_68927

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68927


namespace day_after_53_days_from_Friday_l68_68512

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68512


namespace total_students_in_class_l68_68795

-- No need for noncomputable def here as we're dealing with basic arithmetic

theorem total_students_in_class (jellybeans_total jellybeans_left boys_girls_diff : ℕ)
  (girls boys students : ℕ) :
  jellybeans_total = 450 →
  jellybeans_left = 10 →
  boys_girls_diff = 3 →
  boys = girls + boys_girls_diff →
  students = girls + boys →
  (girls * girls) + (boys * boys) = jellybeans_total - jellybeans_left →
  students = 29 := 
by
  intro h_total h_left h_diff h_boys h_students h_distribution
  sorry

end total_students_in_class_l68_68795


namespace four_digit_perfect_square_palindrome_count_l68_68973

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68973


namespace train_distance_l68_68570

theorem train_distance (train_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  (train_speed = 1) → (total_time = 180) → (distance = train_speed * total_time) → 
  distance = 180 :=
by
  intros train_speed_eq total_time_eq dist_eq
  rw [train_speed_eq, total_time_eq] at dist_eq
  exact dist_eq

end train_distance_l68_68570


namespace largest_n_dividing_factorial_l68_68105

theorem largest_n_dividing_factorial (n : ℕ) (h : ((n!)!)! ∣ (2004!)!) : n ≤ 6 :=
sorry

example : ((6!)!)! ∣ (2004!)! :=
sorry

end largest_n_dividing_factorial_l68_68105


namespace coefficient_of_y_is_8_l68_68112

-- We set up the given conditions as hypotheses
variable (y b : ℝ)
variable h1 : y = 2
variable h2 : 5 * y^2 - b * y + 55 = 59

-- To find the coefficient of y which is -b and prove that b = 8
theorem coefficient_of_y_is_8 (h1 : y = 2) (h2 : 5 * y^2 - b * y + 55 = 59) : b = 8 :=
by
  -- This is the statement that we need to eventually prove
  sorry

end coefficient_of_y_is_8_l68_68112


namespace percent_without_conditions_l68_68591

theorem percent_without_conditions (total_teachers hbp ht d hbp_ht hbp_d ht_d all_three : ℕ)
  (h_total_teachers : total_teachers = 150)
  (h_hbp : hbp = 90)
  (h_ht : ht = 50)
  (h_d : d = 30)
  (h_hbp_ht : hbp_ht = 25)
  (h_hbp_d : hbp_d = 10)
  (h_ht_d : ht_d = 15)
  (h_all_three : all_three = 5) :
  ((total_teachers 
     - (hbp - hbp_ht - hbp_d + all_three) 
     - (ht - hbp_ht - ht_d + all_three) 
     - (d - hbp_d - ht_d + all_three) 
     - (hbp_ht - all_three) 
     - (hbp_d - all_three) 
     - (ht_d - all_three) 
     - all_three) * 100 / total_teachers) = 16.67 := 
by
  sorry

end percent_without_conditions_l68_68591


namespace water_added_l68_68022

theorem water_added (initial_fullness : ℚ) (final_fullness : ℚ) (capacity : ℚ)
  (h1 : initial_fullness = 0.40) (h2 : final_fullness = 3 / 4) (h3 : capacity = 80) :
  (final_fullness * capacity - initial_fullness * capacity) = 28 := by
  sorry

end water_added_l68_68022


namespace Tn_lt_half_Sn_l68_68343

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68343


namespace woody_saving_weeks_l68_68540

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end woody_saving_weeks_l68_68540


namespace find_a_l68_68637

def new_operation (a b : ℝ) : ℝ := 3 * a - 2 * b^2

theorem find_a (a : ℝ) (b : ℝ) (h : b = 4) (h2 : new_operation a b = 10) : a = 14 := by
  have h' : new_operation a 4 = 10 := by rw [h] at h2; exact h2
  unfold new_operation at h'
  linarith

end find_a_l68_68637


namespace ratio_XR_RZ_l68_68262

noncomputable def triangle_XYZ (X Y Z Q R : Point) : Prop :=
  right_triangle X Y Z ∧
  angle Y X Z = 30 ∧ segment_length X Z = 8 ∧
  point_on_line Q X Z ∧ segment_length X Q = 3 * segment_length Y Z ∧
  point_on_segment R X Z ∧ angle Y Z R = 3 * angle Y R Z

theorem ratio_XR_RZ (X Y Z Q R : Point) (h : triangle_XYZ X Y Z Q R) :
  segment_ratio X R R Z = 1 := by
  sorry

end ratio_XR_RZ_l68_68262


namespace min_sum_of_areas_l68_68083

theorem min_sum_of_areas : ∃ x y : ℕ, (x + y = 156 ∧ min_area x y = 761) :=
by
  -- all necessary conditions and definitions
  def min_area (x y : ℕ) : ℕ := (x / 4)^2 + (y / 4)^2
  sorry

end min_sum_of_areas_l68_68083


namespace function_passes_through_fixed_point_l68_68831

noncomputable def given_function (a : ℝ) (x : ℝ) : ℝ :=
  a^(x - 1) + 7

theorem function_passes_through_fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  given_function a 1 = 8 :=
by
  sorry

end function_passes_through_fixed_point_l68_68831


namespace count_palindromic_four_digit_perfect_squares_l68_68985

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68985


namespace f_7_minus_a_eq_neg_7_over_4_l68_68179

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x - 2 else -Real.logb 3 x

variable (a : ℝ)

-- Given conditions
axiom h1 : f a = -2

-- The proof of the required condition
theorem f_7_minus_a_eq_neg_7_over_4 (h1 : f a = -2) : f (7 - a) = -7 / 4 := sorry

end f_7_minus_a_eq_neg_7_over_4_l68_68179


namespace train_distance_l68_68569

theorem train_distance (train_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  (train_speed = 1) → (total_time = 180) → (distance = train_speed * total_time) → 
  distance = 180 :=
by
  intros train_speed_eq total_time_eq dist_eq
  rw [train_speed_eq, total_time_eq] at dist_eq
  exact dist_eq

end train_distance_l68_68569


namespace exists_child_with_two_mittens_between_l68_68579

theorem exists_child_with_two_mittens_between :
  ∃ (children : Array Nat), 
    (children.size = 25) ∧ 
    (∀ c ∈ children, ∃ c1 c2 : Nat, c1 < c2 ∧ c1 ≥ 1 ∧ c2 ≤ 50 ∧ c2 - c1 > 1 ∧ ∀ (x : Nat), x ≠ c1 ∧ x ≠ c2 → prime (abs (x - c1 - 1))) ∧
    (∃ c ∈ children, ∀ c1 c2 : Nat, c1 < c2 ∧ c1 ≥ 1 ∧ c2 ≤ 50 ∧ c2 - c1 = 3) :=
sorry

end exists_child_with_two_mittens_between_l68_68579


namespace four_digit_palindromic_perfect_square_l68_68955

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68955


namespace four_digit_palindromic_perfect_square_l68_68957

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68957


namespace arithmetic_sequence_sum_l68_68613

theorem arithmetic_sequence_sum :
  let seq := [102, 104, 106, 108, 110, 112, 114, 116, 118, 120]
  let common_difference := 2
  2 * (list.sum seq) = 2220 :=
by
  sorry

end arithmetic_sequence_sum_l68_68613


namespace solution_part_for_a_l68_68286

noncomputable def find_k (k x y n : ℕ) : Prop :=
  gcd x y = 1 ∧ 
  x > 0 ∧ y > 0 ∧ 
  k % (x^2) = 0 ∧ 
  k % (y^2) = 0 ∧ 
  k / (x^2) = n ∧ 
  k / (y^2) = n + 148

theorem solution_part_for_a (k x y n : ℕ) (h : find_k k x y n) : k = 467856 :=
sorry

end solution_part_for_a_l68_68286


namespace max_sqrt_sum_l68_68129

theorem max_sqrt_sum (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 5) :
  (sqrt (a + 1) + sqrt (b + 3)) ≤ 3 * sqrt 2 := 
sorry

end max_sqrt_sum_l68_68129


namespace last_three_digits_product_l68_68860

theorem last_three_digits_product (a b c : ℕ) 
  (h1 : (a + b) % 10 = c % 10) 
  (h2 : (b + c) % 10 = a % 10) 
  (h3 : (c + a) % 10 = b % 10) :
  (a * b * c) % 1000 = 250 ∨ (a * b * c) % 1000 = 500 ∨ (a * b * c) % 1000 = 750 ∨ (a * b * c) % 1000 = 0 := 
by
  sorry

end last_three_digits_product_l68_68860


namespace power_mod_l68_68529

theorem power_mod (n : ℕ) : 2^99 % 7 = 1 := 
by {
  sorry
}

end power_mod_l68_68529


namespace number_of_revolutions_wheel_half_mile_l68_68838

theorem number_of_revolutions_wheel_half_mile :
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  (half_mile_in_feet / circumference) = 264 / Real.pi :=
by
  let diameter := 10 * (1 : ℝ)
  let radius := diameter / 2
  let circumference := 2 * Real.pi * radius
  let half_mile_in_feet := 2640
  have h : (half_mile_in_feet / circumference) = 264 / Real.pi := by
    sorry
  exact h

end number_of_revolutions_wheel_half_mile_l68_68838


namespace days_from_friday_l68_68502

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68502


namespace triangle_perimeter_APR_eq_60_l68_68566

-- Define Circle and Tangent relationships in a structured proof format
structure CircleConfiguration (A B C P R Q : Point) (circle : Circle) : Prop :=
(tangent_AB : tangent_from A B circle)
(tangent_AC : tangent_from A C circle)
(tangent_Q : tangent_to Q circle)
(intersection_P : on_line A B P)
(intersection_R : on_line A C R)
(tangent_BP : tangent_from B P circle)
(tangent_CR : tangent_from C R circle)
(BP_PQ_equal : BP = PQ)
(QR_CR_equal : QR = CR)
(AB_AC_equal : AB = 30)
(AC_AB_equal : AC = 30)

-- Formal statement to prove in Lean
theorem triangle_perimeter_APR_eq_60 (A B C P R Q : Point) (circle : Circle)
  (config : CircleConfiguration A B C P R Q circle) : 
  (perimeter (triangle A P R) = 60) :=
by sorry

end triangle_perimeter_APR_eq_60_l68_68566


namespace triangle_area_and_angle_l68_68750

noncomputable def triangle_properties (a b c A B C : ℝ) : Prop := 
  (b ≠ 0 ∧ (2 * a + c) ≠ 0) ∧ 
  (cos B / b = - cos C / (2 * a + c)) ∧ 
  (b * b = a * a + c * c - 2 * a * c * cos B) ∧ 
  (A + B + C = π)

theorem triangle_area_and_angle (a b c A B C : ℝ) 
  (h : triangle_properties a b c A B C) :
  (B = 2 * π / 3) → 
  (b = sqrt 13) → 
  (a + c = 4) → 
  (1/2 * a * c * sin B = 3 * sqrt 3 / 4) := 
by
  intro h1 h2 h3
  sorry

end triangle_area_and_angle_l68_68750


namespace four_digit_palindromic_perfect_square_l68_68949

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68949


namespace integer_value_l68_68748

theorem integer_value (x y z : ℕ) (h1 : 2 * x = 5 * y) (h2 : 5 * y = 6 * z) (h3 : x > 0) (h4 : y > 0) (h5 : z > 0) :
  ∃ a : ℕ, a + y + z = 26 ∧ a = 15 := by
  sorry

end integer_value_l68_68748


namespace four_digit_palindromic_perfect_square_l68_68950

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68950


namespace quadratic_root_shift_l68_68845

theorem quadratic_root_shift (A B p : ℤ) (α β : ℤ) 
  (h1 : ∀ x, x^2 + p * x + 19 = 0 → x = α + 1 ∨ x = β + 1)
  (h2 : ∀ x, x^2 - A * x + B = 0 → x = α ∨ x = β)
  (h3 : α + β = A)
  (h4 : α * β = B) :
  A + B = 18 := 
sorry

end quadratic_root_shift_l68_68845


namespace day_of_week_in_53_days_l68_68499

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68499


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68452

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68452


namespace palindromic_squares_count_l68_68944

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68944


namespace mass_percentage_C_in_CuCO3_l68_68089

def molar_mass_Cu := 63.546 -- g/mol
def molar_mass_C := 12.011 -- g/mol
def molar_mass_O := 15.999 -- g/mol
def molar_mass_CuCO3 := molar_mass_Cu + molar_mass_C + 3 * molar_mass_O

theorem mass_percentage_C_in_CuCO3 : 
  (molar_mass_C / molar_mass_CuCO3) * 100 = 9.72 :=
by
  sorry

end mass_percentage_C_in_CuCO3_l68_68089


namespace ratio_of_pens_to_pencils_l68_68427

-- Define the conditions
def total_items : ℕ := 13
def pencils : ℕ := 4
def eraser : ℕ := 1
def pens : ℕ := total_items - pencils - eraser

-- Prove the ratio of pens to pencils is 2:1
theorem ratio_of_pens_to_pencils : pens = 2 * pencils :=
by
  -- indicate that the proof is omitted
  sorry

end ratio_of_pens_to_pencils_l68_68427


namespace no_four_digit_perfect_square_palindromes_l68_68968

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68968


namespace inequality_proof_l68_68736

theorem inequality_proof (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  (1 / Real.sqrt (x + y)) + (1 / Real.sqrt (y + z)) + (1 / Real.sqrt (z + x)) ≤ 1 / Real.sqrt (2 * x * y * z) :=
by
  sorry

end inequality_proof_l68_68736


namespace sum_possible_values_k_l68_68253

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l68_68253


namespace complement_M_union_N_is_5_l68_68187

universe u

variable {α : Type u}

def U : Set α := {1, 2, 3, 4, 5, 6}
def M : Set α := {1, 3, 6}
def N : Set α := {2, 3, 4}

theorem complement_M_union_N_is_5 : U \ (M ∪ N) = {5} := by
  sorry

end complement_M_union_N_is_5_l68_68187


namespace linear_equation_a_zero_l68_68744

theorem linear_equation_a_zero (a : ℝ) : 
  ((a - 2) * x ^ (abs (a - 1)) + 3 = 9) ∧ (abs (a - 1) = 1) → a = 0 := by
  sorry

end linear_equation_a_zero_l68_68744


namespace monotone_decreasing_interval_3_l68_68137

variable {f : ℝ → ℝ}

theorem monotone_decreasing_interval_3 
  (h1 : ∀ x, f (x + 3) = f (x - 3))
  (h2 : ∀ x, f (x + 3) = f (-x + 3))
  (h3 : ∀ ⦃x y⦄, 0 < x → x < 3 → 0 < y → y < 3 → x < y → f y < f x) :
  f 3.5 < f (-4.5) ∧ f (-4.5) < f 12.5 :=
sorry

end monotone_decreasing_interval_3_l68_68137


namespace common_chord_eq_l68_68719

theorem common_chord_eq :
  ∀ (x y : ℝ),
    (x^2 + y^2 + 2*x + 8*y - 8 = 0) → (x^2 + y^2 - 4*x - 4*y - 2 = 0) →
    x + 2*y - 1 = 0 :=
by
  intros x y h1 h2
  sorry

end common_chord_eq_l68_68719


namespace tan_problem_l68_68668

theorem tan_problem (m : ℝ) (α : ℝ) (h1 : Real.tan α = m / 3) (h2 : Real.tan (α + Real.pi / 4) = 2 / m) :
  m = -6 ∨ m = 1 :=
sorry

end tan_problem_l68_68668


namespace eval_f_pi_over_3_l68_68157

noncomputable def f (α : ℝ) : ℝ := 
  (sin (2 * π - α) * cos (π / 2 + α)) / (cos (-π / 2 + α) * tan (π + α))

theorem eval_f_pi_over_3 : 
  f (π / 3) = 1 / 2 := by
  sorry

end eval_f_pi_over_3_l68_68157


namespace strap_mask_probability_l68_68925

theorem strap_mask_probability 
  (p_regular_medical : ℝ)
  (p_surgical : ℝ)
  (p_strap_regular : ℝ)
  (p_strap_surgical : ℝ)
  (h_regular_medical : p_regular_medical = 0.8)
  (h_surgical : p_surgical = 0.2)
  (h_strap_regular : p_strap_regular = 0.1)
  (h_strap_surgical : p_strap_surgical = 0.2) :
  (p_regular_medical * p_strap_regular + p_surgical * p_strap_surgical) = 0.12 :=
by
  rw [h_regular_medical, h_surgical, h_strap_regular, h_strap_surgical]
  -- proof will go here
  sorry

end strap_mask_probability_l68_68925


namespace constant_term_expansion_l68_68384

theorem constant_term_expansion : 
  let expansion_term (r : ℕ) := 
    (Nat.choose 6 r) * (2 : ℕ)^(6 - r) * (-1 : ℚ)^r * x^(6 - (3 * r / 2)) in
  (∃ r : ℕ, 6 - (3 * r / 2) = 0 ∧ expansion_term r = 60) :=
by 
  let expansion_term (r : ℕ) := 
    (Nat.choose 6 r) * (2 : ℕ)^(6 - r) * (-1 : ℚ)^r * x^(6 - (3 * r / 2))
  sorry

end constant_term_expansion_l68_68384


namespace count_palindromic_four_digit_perfect_squares_l68_68986

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68986


namespace correct_setting_0001001011_l68_68040

-- Define the function that calculates the time change given a switch setting sequence
def time_travel_change (switches : List (Fin 2)) : Int :=
  List.sum (List.map (λ (n : Nat × Fin 2), 
    if n.snd = 1 
    then if (n.fst + 1) % 2 = 1  -- odd
         then 2 ^ n.fst -- advance
         else -(2 ^ n.fst) -- go back
    else 0) (List.zip (List.range 10) switches))

-- The given switch setting: 0001001011
def switches_0001001011 : List (Fin 2) := 
  [0, 0, 0, 1, 0, 0, 1, 0, 1, 1].map (λ b, Fin.mk b (by decide))

-- Statement that proves the switch setting "0001001011" results in the time traveler going back 200 years
theorem correct_setting_0001001011 :
  time_travel_change switches_0001001011 = -200 :=
by
  sorry

end correct_setting_0001001011_l68_68040


namespace largest_nonempty_domain_l68_68784

def g1 (x : ℝ) : ℝ := real.sqrt (2 - x)

def gn : ℕ → (ℝ → ℝ)
| 1 => g1
| n+2 => λ x, gn (n+1) (real.sqrt ((n+3)^2 - x))

theorem largest_nonempty_domain :
  ∃ (M d : ℕ), M = 5 ∧ d = -589 ∧ 
  (∀ x, gn M x = g1 (real.sqrt (16 - x))) := sorry

end largest_nonempty_domain_l68_68784


namespace count_palindromic_four_digit_perfect_squares_l68_68991

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68991


namespace magnitude_of_complex_number_l68_68207

theorem magnitude_of_complex_number :
  let z := (⟨0, real.sqrt 3 + 1⟩ / ⟨1, 1⟩) in
  complex.abs z = real.sqrt 2 :=
by
  sorry

end magnitude_of_complex_number_l68_68207


namespace angle_acb_l68_68261

/-- In triangle ABC, the angle bisectors of ∠BAC and ∠ABC intersect at point O.
Given that ∠AOB = 125°. Prove that the measure of ∠ACB = 70°. -/
theorem angle_acb (A B C O : Type) [incidence_geometry A B C O] :
  (bisects_angle A O B) ∧
  (bisects_angle B O A) ∧
  (angle O A B = 125) →
  (angle C A B = 70) :=
sorry

end angle_acb_l68_68261


namespace count_palindromic_four_digit_perfect_squares_l68_68989

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68989


namespace count_palindromic_four_digit_perfect_squares_l68_68984

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68984


namespace complex_fraction_simplification_l68_68623

theorem complex_fraction_simplification :
  ((10^4 + 324) * (22^4 + 324) * (34^4 + 324) * (46^4 + 324) * (58^4 + 324)) /
  ((4^4 + 324) * (16^4 + 324) * (28^4 + 324) * (40^4 + 324) * (52^4 + 324)) = 373 :=
by
  sorry

end complex_fraction_simplification_l68_68623


namespace sin_theta_through_point_l68_68852

theorem sin_theta_through_point (x y : ℤ) (r : ℝ)
  (h_point : x = 2 ∧ y = -1 ∧ r = Real.sqrt (x^2 + y^2)) :
  Real.sin (Real.atan2 y x) = - (Real.sqrt 5) / 5 :=
by
  cases h_point with hx hy;
  cases hy with hy hr;
  rw [hx, hy, hr];
  sorry

end sin_theta_through_point_l68_68852


namespace clothing_price_reduction_l68_68043

def price_reduction (original_profit_per_piece : ℕ) (original_sales_volume : ℕ) (target_profit : ℕ) (increase_in_sales_per_unit_price_reduction : ℕ) : ℕ :=
  sorry

theorem clothing_price_reduction :
  ∃ x : ℕ, (40 - x) * (20 + 2 * x) = 1200 :=
sorry

end clothing_price_reduction_l68_68043


namespace circle_equation_tangent_to_y_axis_l68_68102

theorem circle_equation_tangent_to_y_axis (x y : ℝ) :
  ∃ r, 
    let center := (-1, 5) in
    let radius := real.abs (center.1) in
    r = radius ∧ 
    (x + 1)^2 + (y - 5)^2 = r^2 :=
begin
  sorry
end

end circle_equation_tangent_to_y_axis_l68_68102


namespace woody_saves_l68_68542

variable (C A W : ℕ)

theorem woody_saves (C A W : ℕ) (H1 : C = 282) (H2 : A = 42) (H3 : W = 24) :
  let additional_amount_needed := C - A in
  let weeks := additional_amount_needed / W in
  weeks = 10 :=
by
  unfold additional_amount_needed weeks
  rw [H1, H2, H3]
  simp
  norm_num
  sorry -- Proof not provided in this exercise

end woody_saves_l68_68542


namespace woody_saving_weeks_l68_68539

variable (cost_needed current_savings weekly_allowance : ℕ)

theorem woody_saving_weeks (h₁ : cost_needed = 282)
                           (h₂ : current_savings = 42)
                           (h₃ : weekly_allowance = 24) :
  (cost_needed - current_savings) / weekly_allowance = 10 := by
  sorry

end woody_saving_weeks_l68_68539


namespace sum_of_all_real_solutions_l68_68091

theorem sum_of_all_real_solutions (x : ℝ) :
  (x^2 - 6 * x + 3)^(x^2 - 7 * x + 6) = 1 → (∑ x in {x : ℝ | (x^2 - 6 * x + 3)^(x^2 - 7 * x + 6) = 1}, x) = 19 :=
by 
  sorry

end sum_of_all_real_solutions_l68_68091


namespace day_after_53_days_from_Friday_l68_68510

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68510


namespace find_x_l68_68018

theorem find_x (x : ℤ) (h : 9873 + x = 13800) : x = 3927 :=
by {
  sorry
}

end find_x_l68_68018


namespace Tn_lt_Sn_over_2_l68_68319

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68319


namespace remainder_of_product_l68_68872

theorem remainder_of_product (a b n : ℕ) (h1 : a = 2431) (h2 : b = 1587) (h3 : n = 800) : 
  (a * b) % n = 397 := 
by
  sorry

end remainder_of_product_l68_68872


namespace remainder_f_2007_mod_2008_l68_68284

noncomputable def phi : ℝ := (1 + Real.sqrt 5) / 2

def f : ℕ → ℕ
| 0 => 1
| (2*x) => (⌊phi * f(x)⌋ : ℕ)
| (2*x + 1) => f(2*x) + f(x)

theorem remainder_f_2007_mod_2008 : (f 2007) % 2008 = 2007 := 
  sorry

end remainder_f_2007_mod_2008_l68_68284


namespace solve_arcsin_arccos_eq_l68_68901

theorem solve_arcsin_arccos_eq (x : ℝ) :
  (arcsin (2 * x) + arcsin (1 - 2 * x) = arccos (2 * x)) ↔ (x = 0 ∨ x = 1 / 2 ∨ x = -1 / 2) :=
by
  sorry

end solve_arcsin_arccos_eq_l68_68901


namespace evaluate_expression_l68_68080

def f (x : ℝ) : ℝ := x^3 + 3 * real.sqrt x

theorem evaluate_expression :
  3 * f 3 - 2 * f 9 = -1395 + 9 * real.sqrt 3 :=
by
  sorry

end evaluate_expression_l68_68080


namespace ellipse_equation_l68_68154

-- Definitions from conditions
variable (O : Point)
def is_origin (P : Point) : Prop := P = O

variables (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) (a_gt_b : a > b)
def ellipse (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

variable (A B C : Point)
def is_vertex_on_ellipse (P : Point) : Prop := ∃ x y, P = (x, y) ∧ ellipse x y

variables (slope_AB slope_OC : ℝ)
def product_of_slopes : Prop := slope_AB * slope_OC = -1 / 2

variable (area_ABC : ℝ)
def area_condition : Prop := area_ABC = 3 * Real.sqrt 6 / 2

-- Statement to prove the equation of the ellipse
theorem ellipse_equation (hO : is_origin O)
    (hA : is_vertex_on_ellipse A) (hB : is_vertex_on_ellipse B) (hC : is_vertex_on_ellipse C)
    (hSlope : product_of_slopes slope_AB slope_OC)
    (hArea : area_condition area_ABC) :
    ellipse 6 3 := sorry

end ellipse_equation_l68_68154


namespace boss_monthly_pay_l68_68222

theorem boss_monthly_pay
  (fiona_hours_per_week : ℕ)
  (john_hours_per_week : ℕ)
  (jeremy_hours_per_week : ℕ)
  (hourly_rate : ℕ)
  (weeks_in_month : ℕ)
  (fiona_income : ℕ := fiona_hours_per_week * hourly_rate)
  (john_income : ℕ := john_hours_per_week * hourly_rate)
  (jeremy_income : ℕ := jeremy_hours_per_week * hourly_rate) :
  fiona_hours_per_week = 40 →
  john_hours_per_week = 30 →
  jeremy_hours_per_week = 25 →
  hourly_rate = 20 →
  weeks_in_month = 4 →
  (fiona_income + john_income + jeremy_income) * weeks_in_month = 7600 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry -- This is the point where the proof would start
end

end boss_monthly_pay_l68_68222


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68450

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68450


namespace day_of_week_in_53_days_l68_68498

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68498


namespace six_digit_numbers_with_zero_l68_68191

theorem six_digit_numbers_with_zero :
  let total_numbers := 9 * 10^5 in
  let total_non_zero_numbers := 9^6 in
  total_numbers - total_non_zero_numbers = 368559 := 
by
  let total_numbers := 9 * 10^5
  let total_non_zero_numbers := 9^6
  calc
    total_numbers - total_non_zero_numbers = 900000 - 531441 : by rfl
    ... = 368559 : by decide

end six_digit_numbers_with_zero_l68_68191


namespace max_separators_l68_68266

theorem max_separators (n : ℕ) (a : Fin n → ℕ) (S : ℕ) (H1 : 0 < n)
  (H2 : ∀ i, 0 < a i) (H3 : (∑ i, a i) = 2 * S) :
  ∃ k₁ k₂, k₁ ≠ k₂ ∧ (∃ (s : Finset (Fin n)), s.card = k₁ ∧ (∑ x in s, a x) = S)
                     ∧ (∃ (s : Finset (Fin n)), s.card = k₂ ∧ (∑ x in s, a x) = S) :=
sorry

end max_separators_l68_68266


namespace find_FG_l68_68380

noncomputable def triangle_abc : Type :=
{
  A : Point,
  B : Point,
  C : Point,
  acute_angle : function B -- Definition for acute-angled triangle
}
 
def H (triangle : triangle_abc) : Point :=
  -- Construction of orthocenter H, intersection of altitudes AD and BE
  sorry 

def circumcircle (triangle : triangle_abc) (H : Point) : circle :=
  -- Construction of circumcircle of triangle ABH
  sorry

def fg_length (triangle : triangle_abc) (H : Point) (cc : circle) : real :=
  -- Length of segment FG where F and G are intersections of circumscribing circle with AC, BC
  sorry

theorem find_FG (triangle : triangle_abc)
  (H : Point)
  (cc : circle)
  (D E F G : Point)
  (DE : length_segment D E = 5)
  (midpoint_of_CG : D = midpoint C G)
  (midpoint_of_CF : E = midpoint C F)
  : fg_length triangle H cc = 10 :=
begin
  sorry
end

end find_FG_l68_68380


namespace min_value_l68_68787

theorem min_value (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h : 1 / (a + 3) + 1 / (b + 3) + 1 / (c + 3) = 1 / 4) : 
  22.75 ≤ a + 3 * b + 2 * c := 
sorry

end min_value_l68_68787


namespace juniors_score_l68_68226

theorem juniors_score (n : ℕ) (j s : ℕ) (avg_score students_avg seniors_avg : ℕ)
  (h1 : 0 < n)
  (h2 : j = n / 5)
  (h3 : s = 4 * n / 5)
  (h4 : avg_score = 80)
  (h5 : seniors_avg = 78)
  (h6 : students_avg = avg_score)
  (h7 : n * students_avg = n * avg_score)
  (h8 : s * seniors_avg = 78 * s) :
  (800 - 624) / j = 88 := by
  sorry

end juniors_score_l68_68226


namespace percent_absent_students_l68_68587

def total_students : ℕ := 180
def num_boys : ℕ := 100
def num_girls : ℕ := 80
def fraction_boys_absent : ℚ := 1 / 5
def fraction_girls_absent : ℚ := 1 / 4

theorem percent_absent_students : 
  (fraction_boys_absent * num_boys + fraction_girls_absent * num_girls) / total_students = 22.22 / 100 := 
  sorry

end percent_absent_students_l68_68587


namespace JoysFourthRod_l68_68278

theorem JoysFourthRod : 
  let rods := list.range' 1 30 -- Rods from 1 to 30 cm (1-based inclusive)
  let used := [5, 10, 20]
  let remaining := rods.filter (λ x, ¬ x ∈ used)
  let valid_rods := remaining.filter (λ x, 6 ≤ x ∧ x ≤ 29)
  valid_rods.length = 21 :=
by
  sorry

end JoysFourthRod_l68_68278


namespace Joe_Roy_diff_l68_68814

variable (Sara Joe Roy : ℕ)

-- Conditions
def Sara_height : Sara = 45 := sorry
def Sara_Joe_diff : Sara = Joe + 6 := sorry
def Roy_height : Roy = 36 := sorry

-- Goal
theorem Joe_Roy_diff : Joe = Roy + 3 :=
  by 
    have h₁ : Joe = Sara - 6 := by rw [Sara_Joe_diff, Sara_height]; sorry
    have h₂ : Sara - 6 = Roy + 3 := by rw [Roy_height]; sorry
    rw [← h₁, h₂]
    sorry
 
end Joe_Roy_diff_l68_68814


namespace day_after_53_days_from_Friday_l68_68514

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68514


namespace solve_f_greater_exp_neg_x_l68_68136

noncomputable def solution_set (f : ℝ → ℝ) (h_diff : Differentiable ℝ f) 
  (h_pos : ∀ x, f x + (deriv f x) > 0) (h_value : f (1/2) = 1 / Real.sqrt Real.exp 1) : Set ℝ :=
{ x | f x > Real.exp (-x) }

theorem solve_f_greater_exp_neg_x 
  (f : ℝ → ℝ) 
  (h_diff : Differentiable ℝ f) 
  (h_pos : ∀ x, f x + (deriv f x) > 0) 
  (h_value : f (1 / 2) = 1 / Real.sqrt Real.exp 1) :
  solution_set f h_diff h_pos h_value = {x | x > 1 / 2} := 
sorry

end solve_f_greater_exp_neg_x_l68_68136


namespace polynomial_is_fourth_degree_quadrinomial_l68_68403

noncomputable def polynomial_example : Polynomial ℚ := 3 * X^2 * Y - X * Y^3 + 5 * X * Y - 1

theorem polynomial_is_fourth_degree_quadrinomial :
  (degree polynomial_example = 4) ∧ (polynomial_example.sum (λd coeff, if coeff ≠ 0 then 1 else 0) = 4) := sorry

end polynomial_is_fourth_degree_quadrinomial_l68_68403


namespace inequality_holds_l68_68205

open Real

theorem inequality_holds (a b : ℝ) (h : a > b) : (1/2)^b > (1/2)^a := 
by
  sorry

end inequality_holds_l68_68205


namespace minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q_l68_68780

theorem minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q :
  let Q (z : ℂ) := z^8 + (6 * (2:ℂ).sqrt + 8) * z^4 - (6 * (2:ℂ).sqrt + 9)
  in ∃ polygon : list ℂ, 
      (∀ vertex, vertex ∈ polygon ↔ (Q vertex = 0)) ∧
      (polygon.length = 8) ∧
      (is_valid_polygon polygon) ∧
      (perimeter polygon = 8 * (2:ℂ).sqrt) := sorry

end minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q_l68_68780


namespace cos_double_angle_l68_68701

noncomputable def tan_theta := 2
def theta : Real := Classical.choice (exists_tan_real tan_theta)
def theta_range : theta ∈ set.Ioo 0 (Real.pi / 2) := sorry

theorem cos_double_angle (h1 : Real.tan theta = tan_theta) (h2 : theta ∈ set.Ioo 0 (Real.pi / 2)) :
  Real.cos (2 * theta) = -3/5 :=
by
  sorry

end cos_double_angle_l68_68701


namespace unique_four_digit_palindromic_square_l68_68929

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68929


namespace total_rehabilitation_centers_l68_68874

noncomputable def center_visits: ℕ := 6 -- Lisa's visits

def jude_visits (l: ℕ) : ℕ := l / 2
def han_visits (j: ℕ) : ℕ := 2 * j - 2
def jane_visits (h: ℕ) : ℕ := 6 + 2 * h

theorem total_rehabilitation_centers :
  let l := center_visits in
  let j := jude_visits l in
  let h := han_visits j in
  let n := jane_visits h in
  l + j + h + n = 27 :=
by
  sorry

end total_rehabilitation_centers_l68_68874


namespace problem1_problem2_l68_68152

open Set Real

-- Definition of sets A, B, and C
def A : Set ℝ := { x | 1 ≤ x ∧ x < 7 }
def B : Set ℝ := { x | 2 < x ∧ x < 10 }
def C (a : ℝ) : Set ℝ := { x | x < a }

-- Problem 1: Prove A ∪ B = { x | 1 ≤ x < 10 }
theorem problem1 : A ∪ B = { x : ℝ | 1 ≤ x ∧ x < 10 } :=
sorry

-- Problem 2: Prove the range of a given the conditions
theorem problem2 (a : ℝ) (h1 : (A ∩ C a) ≠ ∅) (h2 : (B ∩ C a) = ∅) : 1 < a ∧ a ≤ 2 :=
sorry

end problem1_problem2_l68_68152


namespace sum_of_possible_values_of_k_l68_68244

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l68_68244


namespace no_four_digit_perfect_square_palindromes_l68_68960

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68960


namespace smallest_n_value_l68_68545

-- Define the given expression
def exp := (2^5) * (6^2) * (7^3) * (13^4)

-- Define the conditions
def condition_5_2 (n : ℕ) := ∃ k, n * exp = k * 5^2
def condition_3_3 (n : ℕ) := ∃ k, n * exp = k * 3^3
def condition_11_2 (n : ℕ) := ∃ k, n * exp = k * 11^2

-- Define the smallest possible value of n
def smallest_n (n : ℕ) : Prop :=
  condition_5_2 n ∧ condition_3_3 n ∧ condition_11_2 n ∧ ∀ m, (condition_5_2 m ∧ condition_3_3 m ∧ condition_11_2 m) → m ≥ n

-- The theorem statement
theorem smallest_n_value : smallest_n 9075 :=
  by
    sorry

end smallest_n_value_l68_68545


namespace sum_of_possible_k_values_l68_68242

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l68_68242


namespace line_intersects_y_axis_at_l68_68656

noncomputable def line {α : Type*} [LinearOrderedField α] (p1 p2 : α × α) : α → α :=
  λ x, ((p2.snd - p1.snd) / (p2.fst - p1.fst)) * (x - p1.fst) + p1.snd

theorem line_intersects_y_axis_at (α : Type*) [LinearOrderedField α] :
  let p1 := (5 : α, 25 : α)
  let p2 := (-5 : α, -5 : α)
in line p1 p2 0 = 10 :=
by {
  sorry
}

end line_intersects_y_axis_at_l68_68656


namespace allen_total_blocks_l68_68053

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l68_68053


namespace four_points_form_convex_quadrilateral_l68_68896

open Set

structure EuclideanPlane (α : Type*) :=
(Point : Type*)
(Line : Type*)
(collinear : Point → Point → Point → Prop)
(not_collinear : ∀ (a b c : Point), ¬ collinear a b c)

variables {α : Type*} [EuclideanPlane α]

theorem four_points_form_convex_quadrilateral
  (P1 P2 P3 P4 P5 : EuclideanPlane.Point α)
  (h_no_collinear : ∀ (a b c : EuclideanPlane.Point α), ¬ EuclideanPlane.collinear a b c) :
  ∃ (Q1 Q2 Q3 Q4 : EuclideanPlane.Point α), 
    ConvexQuadrilateral Q1 Q2 Q3 Q4 :=
begin
  sorry
end

end four_points_form_convex_quadrilateral_l68_68896


namespace tn_lt_sn_div_2_l68_68306

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68306


namespace number_of_solutions_l68_68774

noncomputable def f (x : ℝ) := x * Real.log x

theorem number_of_solutions (a : ℝ) (x : ℝ) (h : x > 1) : 
  ∃ (n : ℕ), n = if a ≤ 0 then 0 
                 else if a > 0 ∧ 1 < 1/(2*a) then (if 0 < a ∧ a < 1/2 then 0 ∨ 1 else 1) 
                 else 1 :=
sorry

end number_of_solutions_l68_68774


namespace circle_problem_l68_68678

-- Step (1): Define the initial problem and its transformation to standard form
def circle_equation_standard_form (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 2)^2 = 9

-- Step (2): Define the line l with a slope of 1
def line_l_intersects_circle (x y : ℝ) : Prop :=
  ∃ A B : ℝ × ℝ, (A.1^2 + A.2^2 - 2 * A.1 + 4 * A.2 - 4 = 0) ∧
                  (B.1^2 + B.2^2 - 2 * B.1 + 4 * B.2 - 4 = 0) ∧
                  A ≠ B ∧
                  ∀ p : ℝ × ℝ, (p.2 - A.2) = (p.1 - A.1) ∧ (p.2 - B.2) = (p.1 - B.1)

-- Step (3): Define the existence of lines and area of triangle
def exists_line_and_max_area : Prop :=
  ∃ (a b : ℝ), (a + b + 1 = 0) ∧
               ((a = 3/2 ∧ b = -5/2 ∧ (λ x y : ℝ, x - y - 4 = 0)) ∨
                (a = -1 ∧ b = 0 ∧ (λ x y : ℝ, x - y + 1 = 0))) ∧
               (1 / 2 * (2 * sqrt(9 - ((b - a + 3) / sqrt 2)^2)) *
               ((2 - 2 * a) / sqrt 2) = 9 / 2)

-- The main theorem to be proven
theorem circle_problem :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 2 * x + 4 * y - 4 = 0 → circle_equation_standard_form x y) ∧
    line_l_intersects_circle x y ∧
    exists_line_and_max_area :=
begin
  sorry
end

end circle_problem_l68_68678


namespace find_d_for_lattice_probability_l68_68584

noncomputable def lattice_probability_d :=
  let square_side := 100
  let lattice_prob := 1/4
  let π := Real.pi
  let radius := 0.3
  calc
    radius = (1 / (2 * Real.sqrt π)) : by sorry
    _ ≈ 0.3 : by sorry

theorem find_d_for_lattice_probability :
  let square_side := 100
  let lattice_prob := 1/4
  let π := Real.pi
  let radius := 0.3
  ∃ d : ℝ, (π * d^2 = lattice_prob) ∧ Real.floor (d*10)/10 = radius :=
  by sorry

end find_d_for_lattice_probability_l68_68584


namespace four_digit_perfect_square_palindrome_count_l68_68980

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68980


namespace day_after_53_days_from_Friday_l68_68519

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68519


namespace unit_prices_and_purchasing_schemes_l68_68906

-- Define the conditions
axiom eq1 (a b : ℕ) : 2 * a + 3 * b = 510
axiom eq2 (a b : ℕ) : 3 * a + 5 * b = 810
axiom total_balls : ℕ := 50
axiom min_basketballs : ℕ := 30
axiom max_cost : ℕ := 5500

-- Define the main theorem
theorem unit_prices_and_purchasing_schemes (a b x : ℕ) :
  (2 * a + 3 * b = 510) →
  (3 * a + 5 * b = 810) →
  (30 ≤ x ∧ x ≤ 33 ∧ x ≤ (max_cost - 4500) / 30) →
  (∃ a b, a = 120 ∧ b = 90) ∧ 
  (∃ x, 30 ≤ x ∧ x ≤ 33) :=
by 
  sorry

end unit_prices_and_purchasing_schemes_l68_68906


namespace nonnegative_difference_of_roots_l68_68870

theorem nonnegative_difference_of_roots :
  let f : ℚ → ℚ := λ x, x^2 + 42*x + 468 in
  ∃ a b : ℚ, f a = 0 ∧ f b = 0 ∧ abs (a - b) = 6 :=
by
  sorry

end nonnegative_difference_of_roots_l68_68870


namespace pastries_left_l68_68609

-- Definitions for the given conditions
def pastries_initial : ℕ := 56
def cakes_initial : ℕ := 124
def pastries_sold_day1 : ℕ := (0.35 * pastries_initial).to_nat -- round down
def cakes_sold_day1 : ℕ := (0.80 * cakes_initial).to_nat -- round down
def pastries_remaining_day1 : ℕ := pastries_initial - pastries_sold_day1
def cakes_remaining_day1 : ℕ := cakes_initial - cakes_sold_day1
def pastries_sold_day2 : ℕ := (2 / 5 * pastries_remaining_day1).to_nat -- round down
def cakes_sold_day2 : ℕ := 15
def pastries_remaining_day2 : ℕ := pastries_remaining_day1 - pastries_sold_day2

-- Main theorem statement
theorem pastries_left : pastries_remaining_day2 = 23 := by
  sorry

end pastries_left_l68_68609


namespace sequence_bound_l68_68139

variable {a : ℕ+ → ℝ}

theorem sequence_bound (h : ∀ k m : ℕ+, |a (k + m) - a k - a m| ≤ 1) :
    ∀ (p q : ℕ+), |a p / p - a q / q| < 1 / p + 1 / q :=
by
  sorry

end sequence_bound_l68_68139


namespace problem_statement_l68_68697

lemma sin_lt_x (x : ℝ) (hx : 0 < x ∧ x < real.pi / 2) : real.sin x <= x :=
sorry

lemma tan_gt_x (x : ℝ) (hx : 0 < x ∧ x < real.pi / 2) : real.tan x > x :=
sorry

def p (x : ℝ) : Prop := 0 < x ∧ x < real.pi / 2 → real.sin x > x
def q (x : ℝ) : Prop := 0 < x ∧ x < real.pi / 2 → real.tan x > x

theorem problem_statement (x : ℝ) (hx : 0 < x ∧ x < real.pi / 2) :
  (p x ∨ q x) ∧ ((¬ p x) ∨ q x) :=
by
  have hp : ¬ p x := λ hpx, not_le_of_gt (hpx hx) (sin_lt_x x hx)
  have hq : q x := tan_gt_x x hx
  exact ⟨or.inr hq, or.inr hq⟩

end problem_statement_l68_68697


namespace num_correct_propositions_is_one_l68_68430

def non_overlapping (A B : Type) := ¬(∃ x, x ∈ A ∧ x ∈ B)
def perpendicular (A B : Type) := ¬(∃ x, x ∈ A ∧ x ∈ B ∧ ∃ n, n ⊥ x)

variables 
  (m n : Set ℝ^3) -- Define lines m and n
  (α β : Set ℝ^3) -- Define planes α and β
  (h_lines_non_overlap : non_overlapping m n) -- Lines m and n do not overlap
  (h_planes_non_overlap : non_overlapping α β) -- Planes α and β do not overlap
  (h_prop1 : perpendicular m α ∧ perpendicular n β ∧ perpendicular m n → perpendicular α β)
  (h_prop2 : parallel m α ∧ parallel n β ∧ parallel m n → parallel α β)
  (h_prop3 : perpendicular m α ∧ parallel n β ∧ perpendicular m n → perpendicular α β)
  (h_prop4 : perpendicular m α ∧ parallel n β ∧ parallel m n → parallel α β)

theorem num_correct_propositions_is_one :
  ( (perpendicular m α ∧ perpendicular n β ∧ perpendicular m n → perpendicular α β)
  ∧ ¬(parallel m α ∧ parallel n β ∧ parallel m n → parallel α β)
  ∧ ¬(perpendicular m α ∧ parallel n β ∧ perpendicular m n → perpendicular α β)
  ∧ ¬(perpendicular m α ∧ parallel n β ∧ parallel m n → parallel α β) ) :=
by
  sorry

end num_correct_propositions_is_one_l68_68430


namespace T_lt_S_div_2_l68_68333

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68333


namespace domain_of_function_l68_68406

theorem domain_of_function {x : ℝ} : (∃ y, y = (sqrt (x - 1)) / (x - 3)) ↔ (x ≥ 1 ∧ x ≠ 3) :=
by
  sorry

end domain_of_function_l68_68406


namespace T_lt_half_S_l68_68327

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68327


namespace convex_polygon_not_divisible_l68_68616

-- Defining our function f
def f (N : Polygon) : ℝ :=
  let angles := N.interiorAngles
  let lessThan180 := angles.filter (λ α, α < 180)
  let greaterThan180Supp := angles.filter (λ α, α > 180) |>.map (λ α => 360 - α)
  list.sum lessThan180 - list.sum greaterThan180Supp

-- Defining our problem statement
theorem convex_polygon_not_divisible (M : Polygon) (h₁ : M.isConvex) :
  ¬ ∃ (M₁ M₂ ... Mn : Polygon), (∀ i, M₁.isNonConvex) ∧ (M = M₁ ∪ M₂ ∪ ... ∪ Mn) :=
begin
  sorry
end

end convex_polygon_not_divisible_l68_68616


namespace perfect_square_in_interval_l68_68772

theorem perfect_square_in_interval (k : ℕ → ℕ) (s : ℕ → ℕ)
    (h1 : ∀ n, k n > 0)                      -- all k_i are positive integers
    (h2 : ∀ n, k n < k (n + 1))              -- k_1 < k_2 < k_3 < ...
    (h3 : ∀ n, k n + 1 < k (n + 1))          -- no two k_i are consecutive
    (h4 : ∀ n, s n = (List.range n).sum (λ i, k (i + 1))) :  -- s_m = k_1 + k_2 + ... + k_m
  ∀ n, ∃ x, s n ≤ x^2 ∧ x^2 < s (n + 1) := 
by
  sorry

end perfect_square_in_interval_l68_68772


namespace ef_inequality_solution_set_l68_68162

noncomputable def f : ℝ → ℝ := sorry
variable (hf : ∀ x : ℝ, differentiable_at ℝ f x)
variable (hfd : ∀ x : ℝ, deriv f x > f x)

theorem ef_inequality_solution_set :
  { x : ℝ | e * f x > f 1 * exp x } = { x : ℝ | x > 1 } :=
by
  sorry

end ef_inequality_solution_set_l68_68162


namespace find_angle_A_l68_68264

-- Define the geometric setup
variables {α β γ : Type}
variables (A B C : α)
variables (M : α)
variables [InGroupTheoryInversive]
variables [Equiv A B C α]
variables [IsSegment A B α] 
variables [IsAngleBisector A α]
variables [IsAltitude B α]
variables [IsPerpendicularBisector (A, B) α]

-- Define the hypothesis of the problem
hypothesis
  (h1 : IsTriangle A B C)
  (h2 : IntersectsAtSinglePoint (AngleBisector A A B C) (Altitude B B C A) (PerpendicularBisector (A, B) A B α) M)

-- The goal
theorem find_angle_A (h1 : IsTriangle A B C)
  (h2 : IntersectsAtSinglePoint (AngleBisector A A B C) (Altitude B B C A) (PerpendicularBisector (A, B) A B α) M) : 
  angle BAC = 60 := sorry

end find_angle_A_l68_68264


namespace pool_filling_time_l68_68885

theorem pool_filling_time :
  let pool_capacity := 12000 -- in cubic meters
  let first_valve_time := 120 -- in minutes
  let first_valve_rate := pool_capacity / first_valve_time -- in cubic meters per minute
  let second_valve_rate := first_valve_rate + 50 -- in cubic meters per minute
  let combined_rate := first_valve_rate + second_valve_rate -- in cubic meters per minute
  let time_to_fill := pool_capacity / combined_rate -- in minutes
  time_to_fill = 48 :=
by
  sorry

end pool_filling_time_l68_68885


namespace buttons_pattern_total_buttons_sum_l68_68884

-- Define the sequence of the number of buttons in each box
def buttons_in_box (n : ℕ) : ℕ := 3^(n-1)

-- Define the sum of buttons up to the n-th box
def total_buttons (n : ℕ) : ℕ := (3^n - 1) / 2

-- Theorem statements to prove
theorem buttons_pattern (n : ℕ) : buttons_in_box n = 3^(n-1) := by
  sorry

theorem total_buttons_sum (n : ℕ) : total_buttons n = (3^n - 1) / 2 := by
  sorry

end buttons_pattern_total_buttons_sum_l68_68884


namespace only_ten_perfect_square_l68_68095

theorem only_ten_perfect_square (n : ℤ) :
  ∃ k : ℤ, n^4 + 6 * n^3 + 11 * n^2 + 3 * n + 31 = k^2 ↔ n = 10 :=
by
  sorry

end only_ten_perfect_square_l68_68095


namespace original_number_is_40_l68_68013

theorem original_number_is_40 (x : ℝ) (h : 1.25 * x - 0.70 * x = 22) : x = 40 :=
by
  sorry

end original_number_is_40_l68_68013


namespace at_least_three_same_value_l68_68113

-- Defining the situation where we have five fair six-sided dice
def fiveFairDice : Type := Fin 6 → Fin 6 → Fin 6 → Fin 6 → Fin 6

-- Defining the probability of at least three of the five dice showing the same value
def probability_at_least_three_same : Rational :=
  23 / 108

-- The main statement to prove
   
theorem at_least_three_same_value : fiveFairDice → (probability_at_least_three_same = 23 / 108) :=
sorry

end at_least_three_same_value_l68_68113


namespace triangle_ABC_area_l68_68761

variable (A B C P M O : Type)
variable [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variable (AP : Line ℝ A P) (BM : Line ℝ B M)
variable {O: Intersection AP BM} 

-- Conditions
variable (tri_similar1 : Similar (Triangle B O P) (Triangle B O A))
variable (tri_similar2 : Similar (Triangle A O M) (Triangle B O A))
variable (BM_eq_1 : SegmentLength BM = 1)
variable (cos_angle_B : Cosine (Angle B) = 0.6)

-- Question to Prove
theorem triangle_ABC_area (A B C P M O : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (AP : Line ℝ A P) (BM : Line ℝ B M)
  {O: Intersection AP BM} 
  (tri_similar1 : Similar (Triangle B O P) (Triangle B O A))
  (tri_similar2 : Similar (Triangle A O M) (Triangle B O A))
  (BM_eq_1 : SegmentLength BM = 1)
  (cos_angle_B : Cosine (Angle B) = 0.6) :
  Area (Triangle ABC) = 8/15 := sorry

end triangle_ABC_area_l68_68761


namespace nicky_running_time_l68_68797

noncomputable def cristina_speed : ℝ := 5
noncomputable def nicky_speed : ℝ := 3
noncomputable def head_start : ℝ := 12

theorem nicky_running_time : ∃ t : ℝ, (cristina_speed * t = nicky_speed * (t + head_start)) ∧ (t + head_start = 30) :=
by
  let cristina_speed := 5
  let nicky_speed := 3
  let head_start := 12
  let t := 18
  have h₁ : cristina_speed * t = nicky_speed * (t + head_start) := by simp [cristina_speed, nicky_speed, head_start, t]
  have h₂ : (t + head_start) = 30 := by simp [t, head_start]
  exact ⟨t, h₁, h₂⟩, sorry

end nicky_running_time_l68_68797


namespace palindromic_squares_count_l68_68937

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68937


namespace Tn_lt_Sn_div_2_l68_68314

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68314


namespace tn_lt_sn_div_2_l68_68303

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68303


namespace num_tickets_bought_l68_68277

def ticket_price : ℝ := 7
def discount : ℝ := 0.50
def total_spent : ℝ := 84
def discounted_price : ℝ := ticket_price * discount
def num_tickets : ℝ := total_spent / discounted_price

theorem num_tickets_bought : num_tickets = 24 := by
  sorry

end num_tickets_bought_l68_68277


namespace number_of_solutions_l68_68109

noncomputable def count_solutions : ℕ :=
  Nat.card { x // 0 ≤ x.1 ∧ x.1 ≤ 14 ∧
                  0 ≤ x.2 ∧ x.2 ≤ 14 ∧
                  0 ≤ x.3 ∧ x.3 ≤ 14 ∧
                  x.1 + x.2 + x.3 = 14 }

theorem number_of_solutions : count_solutions = 57 := by
  sorry

end number_of_solutions_l68_68109


namespace circle_area_ratio_l68_68074

theorem circle_area_ratio (r R : ℝ) (h : π * R^2 - π * r^2 = (3/4) * π * r^2) :
  R / r = Real.sqrt 7 / 2 :=
by
  sorry

end circle_area_ratio_l68_68074


namespace number_of_true_propositions_l68_68360

-- Definitions
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

def Sn (a : ℕ → ℝ) (n : ℕ) :=
  ∑ i in range (n + 1), a i

-- Propositions
def proposition1 (a : ℕ → ℝ) : Prop :=
  (is_arithmetic_sequence a ∧ is_geometric_sequence a) → ∀ n, a n = a (n + 1)

def proposition2 (a : ℕ → ℝ) (a_coeff b_coeff : ℝ) : Prop :=
  (∀ n, Sn a n = a_coeff * n^2 + b_coeff * n) → is_arithmetic_sequence a

def proposition3 (a : ℕ → ℝ) : Prop :=
  (∀ n, Sn a n = 1 - (-1)^n) → is_geometric_sequence a

-- To prove that the number of true propositions is 2
theorem number_of_true_propositions (a : ℕ → ℝ) (a_coeff b_coeff : ℝ) :
  (proposition1 a) ∧ (¬ proposition2 a a_coeff b_coeff) ∧ (proposition3 a) :=
by
  sorry

end number_of_true_propositions_l68_68360


namespace fifty_three_days_from_friday_is_tuesday_l68_68473

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68473


namespace derivative_y_l68_68649

noncomputable def y (x : ℝ) : ℝ :=
  (Real.sqrt (9 * x^2 - 12 * x + 5)) * Real.arctan (3 * x - 2) - 
  Real.log (3 * x - 2 + Real.sqrt (9 * x^2 - 12 * x + 5))

theorem derivative_y (x : ℝ) :
  ∃ (f' : ℝ → ℝ), deriv y x = f' x ∧ f' x = (9 * x - 6) * Real.arctan (3 * x - 2) / 
  Real.sqrt (9 * x^2 - 12 * x + 5) :=
sorry

end derivative_y_l68_68649


namespace even_pos_3_digit_int_divisible_by_5_and_no_5_l68_68726

noncomputable def count_valid_numbers : ℕ :=
  let A_choices := {1, 2, 3, 4, 6, 7, 8, 9}.card
  let B_choices := {0, 1, 2, 3, 4, 6, 7, 8, 9}.card
  A_choices * B_choices

theorem even_pos_3_digit_int_divisible_by_5_and_no_5 : count_valid_numbers = 72 := sorry

end even_pos_3_digit_int_divisible_by_5_and_no_5_l68_68726


namespace day_of_week_in_53_days_l68_68492

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68492


namespace not_in_triples_777_l68_68392

def joined_int_sequence (n : Nat) : String :=
  (List.range (n+1)).map toString |>.foldr (++) ""

def split_into_triples (s : String) : List String :=
  let rec split_inner (acc : List String) (s : String) : List String :=
    if s.length < 3 then acc
    else split_inner (acc ++ [s.take 3]) (s.drop 3)
  split_inner [] s

def sequence_triples_upto (n : Nat) : List String :=
  split_into_triples (joined_int_sequence n)

theorem not_in_triples_777 :
  ¬ (List.mem "777" (sequence_triples_upto 99)) :=
by
  sorry

end not_in_triples_777_l68_68392


namespace general_term_of_arithmetic_seq_l68_68077

variable {a : ℕ → ℤ}

def arithmetic_seq (a : ℕ → ℤ) := ∃ d, ∀ n, a n = a 0 + n * d

theorem general_term_of_arithmetic_seq :
  arithmetic_seq a →
  a 2 = 9 →
  (∃ x y, (x ^ 2 - 16 * x + 60 = 0) ∧ (a 0 = x) ∧ (a 4 = y)) →
  ∀ n, a n = -n + 11 :=
by
  intros h_arith h_a2 h_root
  sorry

end general_term_of_arithmetic_seq_l68_68077


namespace magicians_trick_success_l68_68856

-- Define the set of dice
def dice_faces := Finset.range 6 + 1

-- Define the possible pairs
def pairs : Finset (ℕ × ℕ) :=
  Finset.product dice_faces dice_faces

-- Define the conditions
variables (total_dice : ℕ) (taken_dice : ℕ) (pairs_encoded : Finset (ℕ × ℕ))

-- Assume the total number of dice is 23
axiom total_dice_eq : total_dice = 23

-- Assume the magicians have predefined coding for pairs
axiom pairs_coding : ∀ (p : ℕ × ℕ), p ∈ pairs_encoded ↔ p ∈ pairs

-- Define the function that encodes the number of dice in pocket using pairs
noncomputable def encode_dice_number (n : ℕ) : (ℕ × ℕ) :=
  if h : 3 ≤ n ∧ n ≤ 23 then classical.some (pairs_encoded.choose (λ p, p ∈ pairs_encoded))
  else (0, 0)

-- Define the property that ensures the trick is successful
def trick_successful : Prop :=
  ∀ (n : ℕ), 3 ≤ n ∧ n ≤ 23 →
  (∃ (p : ℕ × ℕ), (encode_dice_number n) = p ∧ p ∈ pairs_encoded)

-- State the theorem
theorem magicians_trick_success :
  total_dice = 23 →
  trick_successful :=
by {
  assume h_total,
  rw [total_dice_eq] at h_total,
  sorry
}

end magicians_trick_success_l68_68856


namespace tangent_line_x_intercept_l68_68833

noncomputable def f (x : ℝ) : ℝ := x^3 + 4 * x + 5

-- Define the tangent line at x = 1
noncomputable def tangent_line (x : ℝ) : ℝ :=
  let slope := 3 * 1^2 + 4 in -- f'(1) = 7
  let point := (1, f 1) in
  slope * (x - 1) + (f 1)

-- Prove the x-intercept of the tangent line is -3/7
theorem tangent_line_x_intercept : 
  ∃ x : ℝ, tangent_line x = 0 ∧ x = -3/7 :=
by 
  sorry

end tangent_line_x_intercept_l68_68833


namespace count_ways_to_form_square_l68_68854

theorem count_ways_to_form_square :
  let segments := [1, 2, 3, 4, 5, 6, 7, 8, 9]
  let sum_segments := List.sum segments
  ∃ n, n = 9 ∧ (∃ (s : Finset (Fin 9)), 
    s.sum ∈ [7, 8, 9, 10, 11] ∧ 
    (∀ i ∈ s, i.val + 1 ∈ segments)) :=
by
  sorry

end count_ways_to_form_square_l68_68854


namespace sum_smallest_betas_l68_68635

noncomputable def Q (x : ℂ) : ℂ := (∑ i in (range 16), x^i)^2 - x^15

theorem sum_smallest_betas :
  let β : ℕ → ℝ := λ k, k match
    | 1 => 1 / 17
    | 2 => 1 / 15
    | 3 => 2 / 17
    | 4 => 2 / 15
    | 5 => 3 / 17
    | _ => 0
  in β 1 + β 2 + β 3 + β 4 + β 5 = 47 / 85 :=
by sorry

end sum_smallest_betas_l68_68635


namespace probability_not_sit_next_to_each_other_l68_68793

theorem probability_not_sit_next_to_each_other (total_chairs : ℕ) (chairs_chosen : ℕ) (pairs_adjacent : ℕ)
  (total_ways : ℕ := Nat.choose total_chairs chairs_chosen)
  (adjacent_ways : ℕ := pairs_adjacent)
  (prob_adjacent : ℚ := adjacent_ways / total_ways)
  (prob_not_adjacent : ℚ := 1 - prob_adjacent) :
  total_chairs = 12 → 
  chairs_chosen = 2 → 
  pairs_adjacent = 11 → 
  prob_not_adjacent = 5 / 6 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rw [Nat.choose, Nat.succ_mul_sub_one]
  norm_num
  sorry

end probability_not_sit_next_to_each_other_l68_68793


namespace frustum_volume_l68_68437

theorem frustum_volume (r : ℝ) (h : ℝ) :
  (4 * r) = h * sqrt 2 → 
  V = (1 / 3) * h * ((2 * sqrt 2 * r)^2 * 2 * sqrt 2 * r + (2 * sqrt 2 * r)^2 ) = 28 * r^3/3 :=
sorry

end frustum_volume_l68_68437


namespace union_sets_complement_set_l68_68718

namespace SetOperations

open Set

variable (U A B : Set ℕ)

def U := {1, 2, 3, 4, 5, 6, 7}
def A := {2, 4, 5}
def B := {2, 7}

theorem union_sets :
  A ∪ B = {2, 4, 5, 7} :=
sorry

theorem complement_set :
  U \ A = {1, 3, 6, 7} :=
sorry

end SetOperations

end union_sets_complement_set_l68_68718


namespace four_digit_palindrome_square_count_l68_68999

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68999


namespace sum_of_possible_k_values_l68_68238

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l68_68238


namespace abs_diff_c_d_l68_68662

def tau (n : ℕ) : ℕ := n.divisors.card

def S (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), tau i

def c : ℕ := (finset.range 1000).count (λ n, S n % 2 = 1)

def d : ℕ := (finset.range 1000).count (λ n, S n % 2 = 0)

theorem abs_diff_c_d : |c - d| = 33 :=
by
  sorry

end abs_diff_c_d_l68_68662


namespace largest_n_factorial_divisor_l68_68106

theorem largest_n_factorial_divisor (n : ℕ) : ((2004!)! % ((n!)!)! = 0) → n ≤ 6 :=
by {
  sorry
}

end largest_n_factorial_divisor_l68_68106


namespace train_length_l68_68046

noncomputable def length_of_train (speed_train speed_man : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_train + speed_man
  let relative_speed_mps := relative_speed_kmph * (1000 / 3600)
  relative_speed_mps * time

theorem train_length (speed_train speed_man time length : ℝ)
  (h_speed_train : speed_train = 60)
  (h_speed_man : speed_man = 6)
  (h_time : time = 15)
  (h_length : length = 275) :
  length_of_train speed_train speed_man time = length :=
by
  rw [h_speed_train, h_speed_man, h_time, h_length]
  unfold length_of_train
  -- Calculate relative speed in m/s
  have rel_speed_kmph : 60 + 6 = 66 := by norm_num
  have rel_speed_mps : 66 * (1000 / 3600) = 18.333... := by norm_num
  -- Calculate the length of the train
  have length_calc : 18.333... * 15 = 275 := by norm_num
  rw [rel_speed_kmph, rel_speed_mps, length_calc]
  sorry

end train_length_l68_68046


namespace fifty_three_days_from_friday_is_tuesday_l68_68479

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68479


namespace four_digit_palindrome_square_count_l68_68996

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68996


namespace prove_statement_l68_68702

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x
noncomputable def monotone_increasing (f : ℝ → ℝ) := ∀ x y, x ≤ y → f x ≤ f y

def problem_statement (f : ℝ → ℝ) : Prop :=
  odd_function f ∧
  monotone_increasing (λ x, if x < 0 then -f (-x) else f x) ∧
  ∀ x, 0 < x → ∀ y, y = 1/x → (|f (log x) - f (log y)| / 2 < f 1) → (1/e < x ∧ x < e)

theorem prove_statement (f : ℝ → ℝ) (h1 : odd_function f) (h2 : monotone_increasing (λ x, if x < 0 then -f (-x) else f x)) :
  ∀ x, 0 < x → ∀ y, y = 1/x → (|f (log x) - f (log y)| / 2 < f 1) → (1/e < x ∧ x < e) :=
begin
  sorry
end

end prove_statement_l68_68702


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68467

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68467


namespace smallest_b_satisfying_inequality_l68_68657

theorem smallest_b_satisfying_inequality : ∀ b : ℝ, (b^2 - 16 * b + 55) ≥ 0 ↔ b ≤ 5 ∨ b ≥ 11 := sorry

end smallest_b_satisfying_inequality_l68_68657


namespace parabola_tangent_angle_probability_l68_68259

noncomputable def parabola_probability (a b c d : ℝ) (range_alpha : Set ℝ) : ℝ :=
let parabola := { p : ℝ × ℝ | ∃ x, p = (x, x^2 / 4) }
let tangent_slope_at (x : ℝ) := 1 / 2 * x
let valid_xs := { x : ℝ | x ∈ Icc a b ∧ (tangent_slope_at x ∈ range_alpha) }
(valid_xs.card : ℝ) / (Icc a b).card

theorem parabola_tangent_angle_probability :
  parabola_probability (-6) 6 (π / 4) (3 * π / 4) = 2 / 3 := sorry

end parabola_tangent_angle_probability_l68_68259


namespace correct_propositions_count_l68_68432

def line := Type
def plane := Type

variables (m n : line) (α β : plane)

def is_perpendicular_line_plane (m : line) (α : plane) : Prop := sorry
def is_parallel_line_plane (m : line) (α : plane) : Prop := sorry
def is_perpendicular_lines (m n : line) : Prop := sorry
def is_parallel_lines (m n : line) : Prop := sorry
def is_perpendicular_planes (α β : plane) : Prop := sorry
def is_parallel_planes (α β : plane) : Prop := sorry

def correct_number_of_propositions : Prop := 
  ( (is_perpendicular_line_plane m α ∧ is_perpendicular_line_plane n β ∧ is_perpendicular_lines m n → is_perpendicular_planes α β)
  ∧ (is_parallel_line_plane m α ∧ is_parallel_line_plane n β ∧ is_parallel_lines m n → is_parallel_planes α β = false)
  ∧ (is_perpendicular_line_plane m α ∧ is_parallel_line_plane n β ∧ is_perpendicular_lines m n → is_perpendicular_planes α β = false)
  ∧ (is_perpendicular_line_plane m α ∧ is_parallel_line_plane n β ∧ is_parallel_lines m n → is_parallel_planes α β = false) 
  )

theorem correct_propositions_count : correct_number_of_propositions :=
by {
  -- assume the propositions are given as described
  intros,
  sorry
}

end correct_propositions_count_l68_68432


namespace monotonic_decreasing_interval_l68_68395

noncomputable def log_base3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def f (x : ℝ) : ℝ := log_base3 (x^2 - 2 * x)

theorem monotonic_decreasing_interval : ∀ x, x < 0 → ∀ x', x' < 0 → f x > f x' :=
sorry

end monotonic_decreasing_interval_l68_68395


namespace unique_four_digit_palindromic_square_l68_68931

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68931


namespace max_gcd_consecutive_terms_l68_68090

def sequence (n : ℕ) : ℕ := n! + 2 * n

theorem max_gcd_consecutive_terms : ∃ m, ∀ n ≥ 0, gcd (sequence n) (sequence (n + 1)) ≤ m ∧ (∀ n ≥ 1, gcd (sequence n) (sequence (n + 1)) ≤ 2) :=
by
  existsi 2
  sorry

end max_gcd_consecutive_terms_l68_68090


namespace maximum_value_of_omega_l68_68177

variable (A ω : ℝ)

theorem maximum_value_of_omega (hA : 0 < A) (hω_pos : 0 < ω)
  (h1 : ω * (-π / 2) ≥ -π / 2) 
  (h2 : ω * (2 * π / 3) ≤ π / 2) :
  ω = 3 / 4 :=
sorry

end maximum_value_of_omega_l68_68177


namespace trigonometric_identities_l68_68127

variable (α β : Real)
variable (h1 : tan α = 2)
variable (h2 : sin β = 3 / 5)
variable (h3 : β ∈ Set.Ioc (pi / 2) pi)

theorem trigonometric_identities
  (h_cos_beta : cos β = -4 / 5)
  (h_tan_2alpha : tan (2 * α) = -4 / 3)
  (h_tan_2alpha_minus_beta : tan (2 * α - β) = -7 / 24) :
  True :=
by
  sorry

end trigonometric_identities_l68_68127


namespace card_top_left_value_l68_68537

theorem card_top_left_value (a b c d : ℕ) (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_three_known : ∃ x y z, {a, b, c, d} = {x, y, z, 6})
  (h_two_sum_row : c + d = 6)
  (h_two_sum_column : b + d = 10) : a = 3 := sorry

end card_top_left_value_l68_68537


namespace unique_four_digit_palindromic_square_l68_68928

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68928


namespace find_f_2018_l68_68670

noncomputable def f (x : ℝ) : ℝ := x / (1 + x)

def f_seq : ℕ+ → ℝ → ℝ
| ⟨1, _⟩ := f
| ⟨n+1, h⟩ := λ x, f (f_seq ⟨n, n.succ_pos⟩ x)

theorem find_f_2018 (x : ℝ) (hx : x ≥ 0) : f_seq 2018 x = x / (1 + 2018 * x) :=
sorry

end find_f_2018_l68_68670


namespace day_53_days_from_friday_l68_68444

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68444


namespace find_max_marks_l68_68044

variable (M : ℝ)
variable (pass_mark : ℝ := 60 / 100)
variable (obtained_marks : ℝ := 200)
variable (additional_marks_needed : ℝ := 80)

theorem find_max_marks (h1 : pass_mark * M = obtained_marks + additional_marks_needed) : M = 467 := 
by
  sorry

end find_max_marks_l68_68044


namespace no_four_digit_perfect_square_palindromes_l68_68964

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68964


namespace Tn_lt_half_Sn_l68_68341

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68341


namespace milk_production_l68_68824

variables (a b c d e : ℕ) (h1 : a > 0) (h2 : c > 0)

def summer_rate := b / (a * c) -- Rate in summer per cow per day
def winter_rate := 2 * summer_rate -- Rate in winter per cow per day

noncomputable def total_milk_produced := (d * summer_rate * e) + (d * winter_rate * e)

theorem milk_production (h : d > 0) : total_milk_produced a b c d e = 3 * b * d * e / (a * c) :=
by sorry

end milk_production_l68_68824


namespace unique_four_digit_palindromic_square_l68_68935

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68935


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68454

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68454


namespace day_after_53_days_from_Friday_l68_68516

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68516


namespace runners_near_stands_l68_68915

theorem runners_near_stands (track_length : ℝ) (arc_length : ℝ) 
  (n_runners : ℕ) (speeds : Fin n_runners → ℝ) :
  track_length = 2 →
  arc_length = 0.1 →
  n_runners = 20 →
  (∀ i, speeds i ∈ Set.range (λ n : ℕ, (10 + n : ℝ)) (Finset.range 20)) →
  ¬ (∀ t, ∃ i, let distance := speeds i * t % track_length in 
               distance < arc_length ∨ distance > track_length - arc_length) :=
by
  intros h1 h2 h3 h4
  sorry

end runners_near_stands_l68_68915


namespace total_surface_area_proof_l68_68039

-- Define given conditions in Lean
def cylinder_height : ℝ := 8
def cylinder_radius : ℝ := 3
def cone_height : ℝ := 5
def cone_radius : ℝ := cylinder_radius -- same radius as the cylinder

-- Assuming the correct answer as a hypothesis to be proved
def total_exposed_surface_area : ℝ := (57 * Real.pi) + (3 * Real.pi * Real.sqrt 34)

-- Formal Lean statement for the proof problem
theorem total_surface_area_proof :
  let cylinder_lateral_area := 2 * Real.pi * cylinder_radius * cylinder_height
  let l := Real.sqrt (cone_radius^2 + cone_height^2)
  let cone_lateral_area := Real.pi * cone_radius * l
  let base_area := Real.pi * cone_radius^2
  cylinder_lateral_area + cone_lateral_area + base_area = total_exposed_surface_area := by
  sorry

end total_surface_area_proof_l68_68039


namespace day_53_days_from_friday_l68_68440

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68440


namespace trailing_zeros_1_to_100_l68_68193

def count_multiples (n : ℕ) (k : ℕ) : ℕ :=
  if k = 0 then 0 else n / k

def trailing_zeros_in_range (n : ℕ) : ℕ :=
  let multiples_of_5 := count_multiples n 5
  let multiples_of_25 := count_multiples n 25
  multiples_of_5 + multiples_of_25

theorem trailing_zeros_1_to_100 : trailing_zeros_in_range 100 = 24 := by
  sorry

end trailing_zeros_1_to_100_l68_68193


namespace allen_total_blocks_l68_68054

/-- 
  If there are 7 blocks for every color of paint used and Shiela used 7 colors, 
  then the total number of blocks Allen has is 49.
-/
theorem allen_total_blocks
  (blocks_per_color : ℕ) 
  (number_of_colors : ℕ)
  (h1 : blocks_per_color = 7) 
  (h2 : number_of_colors = 7) : 
  blocks_per_color * number_of_colors = 49 := 
by 
  sorry

end allen_total_blocks_l68_68054


namespace work_problem_solution_l68_68020

theorem work_problem_solution :
  (∃ C: ℝ, 
    B_work_days = 8 ∧ 
    (1 / A_work_rate + 1 / B_work_days + C = 1 / 3) ∧ 
    C = 1 / 8
  ) → 
  A_work_days = 12 :=
by
  sorry

end work_problem_solution_l68_68020


namespace four_digit_palindromic_perfect_square_l68_68951

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68951


namespace union_segments_lebesgue_measurable_and_minimal_measure_l68_68283

noncomputable def continuous_function (f : ℝ → ℝ) : Prop :=
  continuous_on f (Icc 0 1)

def At (t : ℝ) : set (ℝ × ℝ) :=
  {(t, 0)}

def Bt (f : ℝ → ℝ) (t : ℝ) : set (ℝ × ℝ) :=
  {(f t, 1)}

def segment (p q : ℝ × ℝ) : set (ℝ × ℝ) :=
  { x | ∃ (t : ℝ), 0 ≤ t ∧ t ≤ 1 ∧ x = (t * p.1 + (1 - t) * q.1, t * p.2 + (1 - t) * q.2) }

def union_segments (f : ℝ → ℝ) : set (ℝ × ℝ) :=
  ⋃ t ∈ (Icc 0 1), segment (t, 0) (f t, 1)

theorem union_segments_lebesgue_measurable_and_minimal_measure (f : ℝ → ℝ) (hf : continuous_function f) :
  (measurable_set (union_segments f)) ∧ (measure_space.measure (union_segments f) = real.sqrt 2 - 1) :=
by 
  sorry

end union_segments_lebesgue_measurable_and_minimal_measure_l68_68283


namespace find_day_53_days_from_friday_l68_68488

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68488


namespace relationship_between_abc_l68_68130

noncomputable def a : ℝ := Real.exp 0.9 + 1
def b : ℝ := 2.9
noncomputable def c : ℝ := Real.log (0.9 * Real.exp 3)

theorem relationship_between_abc : a > b ∧ b > c :=
by {
  sorry
}

end relationship_between_abc_l68_68130


namespace acceleration_at_3_seconds_l68_68737

theorem acceleration_at_3_seconds :
  let s := λ t : ℝ, 2 * t^3,
      v := λ t : ℝ, (deriv s t),
      a := λ t : ℝ, (deriv v t)
  in a 3 = 36 :=
by
  sorry

end acceleration_at_3_seconds_l68_68737


namespace least_positive_integer_a_l68_68652

theorem least_positive_integer_a (a : ℕ) (n : ℕ) 
  (h1 : 2001 = 3 * 23 * 29)
  (h2 : 55 % 3 = 1)
  (h3 : 32 % 3 = -1)
  (h4 : 55 % 23 = 32 % 23)
  (h5 : 55 % 29 = -32 % 29)
  (h6 : n % 2 = 1)
  : a = 436 := 
sorry

end least_positive_integer_a_l68_68652


namespace paint_segments_l68_68816

-- Definitions of the basic components involved
variables (G B : Type) [Fintype G] [Fintype B] (E : G → B → Prop)

-- Hypothesis conditions translated to Lean context
variables (h1 : ∀ (g : G), ∃ (b : B), E g b) -- Segments exist between good and bad points
variables (h2 : ∀ (g : G), Fintype.card { b : B // E g b } ≤ 100) -- At most 100 segments begin at each good point
variables (h3 : ∀ (b : B), Fintype.card { g : G // E g b } ≤ 100) -- At most 100 segments end at each bad point

-- Definition of the coloring predicate
definition segment_coloring (c1 c2 : Fin 200) (g : G) (b : B) : Prop := true 

theorem paint_segments :
  ∃ (c : G → B → Fin 200 × Fin 200), 
    (∀ (g : G) (b : B), E g b → segment_coloring (fst (c g b)) (snd (c g b)) g b) ∧
    (∀ (g : G) (b₁ b₂ : B), E g b₁ → E g b₂ → b₁ ≠ b₂ → 
      (fst (c g b₁) ≠ fst (c g b₂) ∧ fst (c g b₁) ≠ snd (c g b₂)) ∧ 
      (snd (c g b₁) ≠ fst (c g b₂) ∧ snd (c g b₁) ≠ snd (c g b₂))) ∧
    (∀ (b : B) (g₁ g₂ : G), E g₁ b → E g₂ b → g₁ ≠ g₂ → 
      (fst (c g₁ b) ≠ fst (c g₂ b) ∧ fst (c g₁ b) ≠ snd (c g₂ b)) ∧ 
      (snd (c g₁ b) ≠ fst (c g₂ b) ∧ snd (c g₁ b) ≠ snd (c g₂ b))) :=
sorry

end paint_segments_l68_68816


namespace problem_statement_l68_68357

noncomputable def c : ℕ → ℝ
| 0       := -3
| (n + 1) := c n + d n + 2 * real.sqrt (c n ^ 2 + d n ^ 2)

noncomputable def d : ℕ → ℝ
| 0       := 2
| (n + 1) := c n + d n - 2 * real.sqrt (c n ^ 2 + d n ^ 2)

theorem problem_statement : 1 / c 2012 + 1 / d 2012 = -1 / 6 :=
by
  sorry

end problem_statement_l68_68357


namespace initial_distance_between_trains_l68_68434

theorem initial_distance_between_trains :
  let length_train1 := 100 -- meters
  let length_train2 := 200 -- meters
  let speed_train1_kmph := 54 -- km/h
  let speed_train2_kmph := 72 -- km/h
  let time_hours := 1.999840012798976 -- hours
  
  -- Conversion to meters per second
  let speed_train1_mps := speed_train1_kmph * 1000 / 3600 -- 15 m/s
  let speed_train2_mps := speed_train2_kmph * 1000 / 3600 -- 20 m/s

  -- Conversion of time to seconds
  let time_seconds := time_hours * 3600 -- 7199.4240460755136 seconds

  -- Relative speed in meters per second
  let relative_speed := speed_train1_mps + speed_train2_mps -- 35 m/s

  -- Distance covered by both trains
  let distance_covered := relative_speed * time_seconds -- 251980.84161264498 meters

  -- Initial distance between the trains
  let initial_distance := distance_covered - (length_train1 + length_train2) -- 251680.84161264498 meters

  initial_distance = 251680.84161264498 := 
by
  sorry

end initial_distance_between_trains_l68_68434


namespace Tn_lt_Sn_over_2_l68_68321

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68321


namespace sum_of_possible_values_of_k_l68_68245

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l68_68245


namespace no_four_digit_perfect_square_palindromes_l68_68963

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68963


namespace matrices_product_l68_68066

theorem matrices_product :
  let sequence := list.map (fun n => ⟨⟨1, n⟩, ⟨0, 1⟩⟩) [2, 4..50]
  let result := sequence.foldl (λ acc mat, ⟨⟨1, acc.1.2 + mat.1.2⟩, ⟨0, 1⟩⟩) ⟨⟨1, 0⟩, ⟨0, 1⟩⟩
  result = ⟨⟨1, 650⟩, ⟨0, 1⟩⟩ :=
by
  sorry

end matrices_product_l68_68066


namespace volume_ratio_l68_68592

-- Define the geometric setup
structure Tetrahedron :=
  (A B C D : ℝ)
  (distance_to_plane : ℝ → ℝ)

variable {ABCD : Tetrahedron}

-- Define the ratio condition
def ratio_condition (k : ℝ) (plane : Tetrahedron → ℝ → ℝ) : Prop :=
  plane ABCD k = plane ABCD 1 / k

-- Define the volume ratio problem in Lean 4 statement
theorem volume_ratio (k : ℝ) (h : ratio_condition k ABCD.distance_to_plane) :
  (volume (tet1 ABCD k) / volume (tet2 ABCD k)) = (k^2 * (k + 3))/(3*k + 1) :=
by
  sorry

end volume_ratio_l68_68592


namespace greatest_prime_factor_of_144_l68_68526

theorem greatest_prime_factor_of_144 : 
  ∃ p, p = 3 ∧ prime p ∧ ∀ q, prime q ∧ (q ∣ 144) → q ≤ p := 
by
  sorry

end greatest_prime_factor_of_144_l68_68526


namespace arithmetic_geometric_mean_unique_solution_base_no_solution_in_base_12_more_solutions_bases_l68_68825

theorem arithmetic_geometric_mean_unique_solution_base
  (x y : ℕ) (hxy : x ≠ y) 
  (h_arith_mean : 10 ≤ (x + y) / 2 ∧ (x + y) / 2 < 100) 
  (h_geom_mean_exists : ∃ a b, 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ (x + y) / 2 = 10 * a + b ∧ nat.sqrt (x * y) = 10 * b + a) : 
  (x = 98 ∧ y = 32) ∨ (x = 32 ∧ y = 98) :=
sorry

theorem no_solution_in_base_12
  (x y : ℕ) (hxy : x ≠ y) : 
  ¬(10 ≤ (x + y) / 2 ∧ (x + y) / 2 < 144) :=
sorry

theorem more_solutions_bases (g : ℕ) :
  (g = 9 ∨ g = 14 → ¬(exists (x y : ℕ), 
    x ≠ y ∧ 10 ≤ (x + y) / 2 ∧ (x + y) / 2 < g * g 
    ∧ nat.sqrt (x * y) = nat.sqrt ((g * ((x + y) / 2)))) :=
sorry

end arithmetic_geometric_mean_unique_solution_base_no_solution_in_base_12_more_solutions_bases_l68_68825


namespace min_value_f_l68_68836

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3) * Real.sin x + Real.sin (Real.pi / 2 + x)

theorem min_value_f : ∃ x : ℝ, f x = -2 := by
  sorry

end min_value_f_l68_68836


namespace correct_propositions_count_l68_68433

def line := Type
def plane := Type

variables (m n : line) (α β : plane)

def is_perpendicular_line_plane (m : line) (α : plane) : Prop := sorry
def is_parallel_line_plane (m : line) (α : plane) : Prop := sorry
def is_perpendicular_lines (m n : line) : Prop := sorry
def is_parallel_lines (m n : line) : Prop := sorry
def is_perpendicular_planes (α β : plane) : Prop := sorry
def is_parallel_planes (α β : plane) : Prop := sorry

def correct_number_of_propositions : Prop := 
  ( (is_perpendicular_line_plane m α ∧ is_perpendicular_line_plane n β ∧ is_perpendicular_lines m n → is_perpendicular_planes α β)
  ∧ (is_parallel_line_plane m α ∧ is_parallel_line_plane n β ∧ is_parallel_lines m n → is_parallel_planes α β = false)
  ∧ (is_perpendicular_line_plane m α ∧ is_parallel_line_plane n β ∧ is_perpendicular_lines m n → is_perpendicular_planes α β = false)
  ∧ (is_perpendicular_line_plane m α ∧ is_parallel_line_plane n β ∧ is_parallel_lines m n → is_parallel_planes α β = false) 
  )

theorem correct_propositions_count : correct_number_of_propositions :=
by {
  -- assume the propositions are given as described
  intros,
  sorry
}

end correct_propositions_count_l68_68433


namespace savings_l68_68280

def distance_each_way : ℕ := 150
def round_trip_distance : ℕ := 2 * distance_each_way
def rental_cost_first_option : ℕ := 50
def rental_cost_second_option : ℕ := 90
def gasoline_efficiency : ℕ := 15
def gasoline_cost_per_liter : ℚ := 0.90
def gasoline_needed_for_trip : ℚ := round_trip_distance / gasoline_efficiency
def total_gasoline_cost : ℚ := gasoline_needed_for_trip * gasoline_cost_per_liter
def total_cost_first_option : ℚ := rental_cost_first_option + total_gasoline_cost
def total_cost_second_option : ℚ := rental_cost_second_option

theorem savings : total_cost_second_option - total_cost_first_option = 22 := by
  sorry

end savings_l68_68280


namespace alpha_magnitude_condition_l68_68638

open Complex

theorem alpha_magnitude_condition (α : ℂ) : (∀ (z1 z2 : ℂ), |z1| < 1 → |z2| < 1 → z1 ≠ z2 → ((z1 + α)^2 + α * conj z1 ≠ (z2 + α)^2 + α * conj z2)) ↔ |α| ≥ 2 :=
by
  sorry

end alpha_magnitude_condition_l68_68638


namespace magicians_can_determine_dice_in_pocket_l68_68859

-- Definitions based on conditions
def dice_faces : List Nat := [1, 2, 3, 4, 5, 6]

-- Generate all pairs (a, b) with a <= b
def pairs := List.bind dice_faces (λ a, List.map (λ b => (a, b)) (dice_faces.filter (λ b => a <= b)))

-- The preassigned unique index for each pair
def pair_index (pair : Nat × Nat) : Option Nat :=
  pairs.indexOf pair

-- Given conditions
theorem magicians_can_determine_dice_in_pocket
  (n : Nat) -- number of dice taken by spectator
  (h1 : 3 ≤ n) -- at least three dice taken
  (h2 : n ≤ 23) -- at most twenty-three dice taken
  (d1 d2 : Nat) -- numbers on top faces of two dice shown
  (hp : (d1, d2) ∈ pairs ∨ (d2, d1) ∈ pairs) -- the pair exists in the list
  : ∃ k, pair_index (d1, d2) = some k ∨ pair_index (d2, d1) = some k ∧ k = (n - 2) := sorry

end magicians_can_determine_dice_in_pocket_l68_68859


namespace semicircle_centers_form_parallelogram_l68_68368

variables {A B C D : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D]

/-- A quadrilateral with semicircles constructed on its sides as diameters. Two opposite semicircles 
    are directed inward while the other two are directed outward. The centers of these semicircles 
    form a parallelogram. -/
theorem semicircle_centers_form_parallelogram 
  (ABCD : quadrilateral A B C D)
  (O1 O2 O3 O4 : point)
  (semicircle_O1_diameter : A → B → semicircle)
  (semicircle_O2_diameter : B → C → semicircle)
  (semicircle_O3_diameter : C → D → semicircle)
  (semicircle_O4_diameter : D → A → semicircle)
  (O1_midpoint : midpoint A B O1)
  (O2_midpoint : midpoint B C O2)
  (O3_midpoint : midpoint C D O3)
  (O4_midpoint : midpoint D A O4)
  (opposite_semicircles_inward : semicircle_directed_inward semicircle_O1_diameter semicircle_O3_diameter)
  (opposite_semicircles_outward : semicircle_directed_outward semicircle_O2_diameter semicircle_O4_diameter) :
  parallelogram O1 O2 O3 O4 :=
sorry

end semicircle_centers_form_parallelogram_l68_68368


namespace negation_exists_implies_forall_l68_68397

theorem negation_exists_implies_forall (x_0 : ℝ) (h : ∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) : 
  ¬ (∃ x_0 : ℝ, x_0^3 - x_0 + 1 > 0) ↔ ∀ x : ℝ, x^3 - x + 1 ≤ 0 :=
by 
  sorry

end negation_exists_implies_forall_l68_68397


namespace diameter_of_small_plate_l68_68026

theorem diameter_of_small_plate (D_big : ℝ) (frac_uncovered : ℝ) (d_small : ℝ) : 
  D_big = 12 → frac_uncovered = 0.3055555555555555 → 
  (π * d_small^2 / 4 = π * (1 - frac_uncovered) * (D_big / 2)^2) → 
  d_small = 10 := 
by {
  intros h1 h2 h3,
  sorry
}

end diameter_of_small_plate_l68_68026


namespace find_ab_l68_68812

theorem find_ab (a b : ℝ) (h1 : 2^a = 64^(b + 1)) (h2 : 8^b = 2^(a - 2)) : a * b = 8 / 3 :=
by
  sorry

end find_ab_l68_68812


namespace four_digit_palindromic_perfect_square_l68_68953

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68953


namespace calculated_points_l68_68055

def circumference (r : ℝ) : ℝ := 2 * Real.pi * r
def modified_area (r : ℝ) : ℝ := 2 * Real.pi * r^2

theorem calculated_points :
  (∀ r ∈ ({1, 3, 5, 7, 9} : set ℝ), ∃ C A : ℝ, (C = circumference r) ∧ (A = modified_area r)) →
  ({(circumference 1, modified_area 1), 
    (circumference 3, modified_area 3), 
    (circumference 5, modified_area 5), 
    (circumference 7, modified_area 7), 
    (circumference 9, modified_area 9)} :
    set (ℝ × ℝ)) = 
  ({(2 * Real.pi, 2 * Real.pi), 
    (6 * Real.pi, 18 * Real.pi), 
    (10 * Real.pi, 50 * Real.pi), 
    (14 * Real.pi, 98 * Real.pi), 
    (18 * Real.pi, 162 * Real.pi)} : set (ℝ × ℝ)) := 
by sorry

end calculated_points_l68_68055


namespace game_worth_l68_68426

noncomputable def superNintendoValue : ℕ := 150
noncomputable def storeCreditPercentage : ℚ := 0.8
noncomputable def actualPayment : ℕ := 70
noncomputable def NESPrice : ℕ := 160

theorem game_worth : 
  let credit := (storeCreditPercentage * superNintendoValue : ℚ).toReal in
  let amountCoveredByCredit := NESPrice - actualPayment in
  let remainingCredit := credit - amountCoveredByCredit in
  remainingCredit = 30 :=
by
  sorry

end game_worth_l68_68426


namespace abs_expression_value_l68_68624

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end abs_expression_value_l68_68624


namespace golden_section_AP_l68_68204

-- Definitions of the golden ratio and its reciprocal
noncomputable def phi := (1 + Real.sqrt 5) / 2
noncomputable def phi_inv := (Real.sqrt 5 - 1) / 2

-- Conditions of the problem
def isGoldenSectionPoint (A B P : ℝ) := ∃ AP BP AB, AP < BP ∧ BP = 10 ∧ P = AB ∧ AP = BP * phi_inv

theorem golden_section_AP (A B P : ℝ) (h1 : isGoldenSectionPoint A B P) : 
  ∃ AP, AP = 5 * Real.sqrt 5 - 5 :=
by
  sorry

end golden_section_AP_l68_68204


namespace guess_matches_draw_exactly_thirteen_times_l68_68421

theorem guess_matches_draw_exactly_thirteen_times :
  ∃ n ≥ 13, n = 13 ∧ (let deck := [13, 13, 13, 13] in
                       ∀ (guess draw : ℕ), guess ∈ {i | i ∈ deck ∧ ∀ j ∈ deck, i ≥ j} →
                       draw ∈ {0, 1, 2, 3} →
                       guess = draw → n = 13) := 
sorry

end guess_matches_draw_exactly_thirteen_times_l68_68421


namespace det_matrix_A_l68_68073

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![8, 4], ![-2, 3]]

def determinant_2x2 (A : Matrix (Fin 2) (Fin 2) ℤ) : ℤ :=
  A 0 0 * A 1 1 - A 0 1 * A 1 0

theorem det_matrix_A : determinant_2x2 matrix_A = 32 := by
  sorry

end det_matrix_A_l68_68073


namespace four_digit_perfect_square_palindrome_count_l68_68972

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68972


namespace mutually_exclusive_not_contradictory_l68_68377

def Group : Type := { x // x > 0 }
def Boys : ℕ := 5
def Girls : ℕ := 3
def TotalStudents : ℕ := Boys + Girls

-- Defining event for selecting exactly 1 boy
def EventOneBoy : Set (Finset Group) := {s | s.card = 2 ∧ s.filter (λ x, x = Boys).card = 1}

-- Defining event for selecting exactly 2 girls
def EventTwoGirls : Set (Finset Group) := {s | s.card = 2 ∧ s.filter (λ x, x = Girls).card = 2}

-- Proving the events are mutually exclusive but not contradictory
theorem mutually_exclusive_not_contradictory :
  (disjoint EventOneBoy EventTwoGirls) ∧ ¬(EventOneBoy ∪ EventTwoGirls = {s | s.card = 2}) :=
begin
  sorry
end

end mutually_exclusive_not_contradictory_l68_68377


namespace wheel_radius_l68_68916

theorem wheel_radius
  (revolutions : ℝ) (distance_km : ℝ) (pi : ℝ)
  (h_rev : revolutions = 11085.714285714286)
  (h_dist_km : distance_km = 3)
  (h_pi : pi = Real.pi)

  (h_circumference : ∀ (r : ℝ), 2 * pi * r = (distance_km * 1000) / revolutions) :
  ∃ (r : ℝ), r ≈ 0.043262 :=
by
  -- Define the radius using the provided formula
  let r := (distance_km * 1000) / (revolutions * 2 * pi)
  -- Approximate the value to check
  have hr : r ≈ 0.043262 := by sorry
  -- Existential quantification proof
  exact ⟨r, hr⟩

end wheel_radius_l68_68916


namespace perimeter_of_square_l68_68828

theorem perimeter_of_square (d : ℝ) (h : d = 2 * Real.sqrt 2) : ∃ P : ℝ, P = 8 :=
by
  -- Define the side length s of the square, using the relation between diagonal d and side length s
  let s := d / Real.sqrt 2
  -- Compute the perimeter P
  let P := 4 * s
  -- Calculate the specific values and prove P equals 8
  use P
  rw [h, s]
  simp [Real.sqrt]
  sorry

end perimeter_of_square_l68_68828


namespace no_symmetric_a_l68_68712

noncomputable def f (a x : ℝ) : ℝ := Real.log (((x + 1) / (x - 1)) * (x - 1) * (a - x))

theorem no_symmetric_a (a : ℝ) (h_a : 1 < a) : ¬ ∃ c : ℝ, ∀ d : ℝ, 1 < c - d ∧ c - d < a ∧ 1 < c + d ∧ c + d < a → f a (c - d) = f a (c + d) :=
sorry

end no_symmetric_a_l68_68712


namespace train_travel_distance_l68_68576

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l68_68576


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68461

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68461


namespace four_digit_palindromic_perfect_square_l68_68958

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68958


namespace frog_escape_probability_l68_68229

noncomputable def P : ℕ → ℚ := sorry

axiom start_on_pad: P 2 = ?  -- frog starts on pad 2

axiom recursive_formula : ∀ N, 0 < N ∧ N < 6 → 
  P N = (N / 6) * P (N - 1) + (1 - N / 6) * P (N + 1)

axiom boundary_conditions_0 : P 0 = 0 -- frog gets trapped at position 0
axiom boundary_conditions_6 : P 6 = 1 -- frog escapes at position 6

theorem frog_escape_probability : P 2 = 1 / 3 :=
sorry

end frog_escape_probability_l68_68229


namespace entrance_exam_results_l68_68232

-- Given the conditions
variables {C I U : ℕ} -- Defining C, I, U as natural numbers
variable h1 : C + I = 70
variable h2 : 3 * C - I = 38

-- Prove the questions
theorem entrance_exam_results : C = 27 ∧ I = 43 ∧ U = 0 :=
by {
  have hU : U = 70 - (C + I),
  sorry
}

end entrance_exam_results_l68_68232


namespace commodity_price_difference_l68_68404

theorem commodity_price_difference :
  ∃ (n : ℕ), let price_x := 5.20 + 0.45 * n in
              let price_y := 7.30 + 0.20 * n in
              n = 7 ∧ price_x + 0.35 = price_y := 
sorry

end commodity_price_difference_l68_68404


namespace area_of_triangle_AQB_l68_68589

variables {Point : Type}
variables (A B C D H Q : Point)
variables (side_length : ℝ)
variables [metric_space Point]

noncomputable def distance (p q : Point) : ℝ := dist p q

-- Given conditions
axiom square_with_side_eight :
  side_length = 8 ∧
  distance A B = side_length ∧
  distance A H = side_length ∧
  distance H D = side_length ∧
  distance D B = side_length

axiom point_Q_condition :
  distance Q A = distance Q B ∧
  distance Q B = distance Q C ∧
  distance Q C = distance Q A

axiom QC_perpendicular_HD :
  ∠ Q C H = 90

-- The problem statement to be proved
theorem area_of_triangle_AQB :
  ∃ (area : ℝ), area = 12 ∧
  (distance A B * (8 - distance Q A)) / 2 = area :=
sorry

end area_of_triangle_AQB_l68_68589


namespace problem_1_problem_2_problem_3_problem_4_l68_68815

theorem problem_1 : 
  let persons := [1, 2, 3, 4, 5, 6, 7] in
  (∃ pA pB, pA ≠ pB ∧ permutation persons = some (list.perm persons (list.update_nth (list.update_nth persons pA 0) pB 0))) →
  num_ways_next_to_each_other persons = 1440 :=
sorry

theorem problem_2 :
  let persons := [1, 2, 3, 4, 5, 6, 7] in
  (∀ pA pB, pA ≠ pB → permutation persons = some (list.perm persons (list.update_nth (list.update_nth persons pA 0) pB 0) = false )) →
  num_ways_not_next_to_each_other persons = 3600 :=
sorry

theorem problem_3 :
  let persons := [1, 2, 3, 4, 5, 6, 7] in
  (∃ pA pB pC, pA ≠ pB ∧ pA ≠ pC ∧ pB ≠ pC ∧ ∀ (x y), x ∈ [pA, pB, pC] ∧ y ∈ [pA, pB, pC] → (x = y ∨ |x - y| > 1) ∧ permutation persons = some (list.perm persons (list.update_nth (list.update_nth (list.update_nth persons pA 0) pB 0) pC 0))) →
  num_ways_no_two_next_to_each_other persons = 1440 :=
sorry

theorem problem_4 :
  let persons := [1, 2, 3, 4, 5, 6, 7] in
  (∃ pA pB pC, pA ≠ pB ∧ pA ≠ pC ∧ pB ≠ pC ∧ count_at_most_two_next_to_each_other persons [pA, pB, pC] ≤ 2 ∧ permutation persons = some (list.perm persons (list.update_nth (list.update_nth (list.update_nth persons pA 0) pB 0) pC 0))) →
  num_ways_at_most_two_not_next_to_each_other persons = 4320 :=
sorry

end problem_1_problem_2_problem_3_problem_4_l68_68815


namespace palindromic_squares_count_l68_68940

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68940


namespace correct_propositions_l68_68720

-- Definitions of the problem
variables (a b : Type) [LinearOrder a] [LinearOrder b]
variables (α β : Type) [LinearOrder α] [LinearOrder β]

-- Define parallel and perpendicular predicates
variable (parallel : (Type → Type → Prop))
variable (perpendicular : (Type → Type → Prop))

-- Define the propositions
def proposition1 : Prop := (parallel α a) ∧ (parallel β a) → (parallel α β)
def proposition2 : Prop := (perpendicular α a) ∧ (perpendicular β a) → (parallel α β)
def proposition3 : Prop := (parallel a α) ∧ (parallel b α) → (parallel a b)
def proposition4 : Prop := (perpendicular a α) ∧ (perpendicular b α) → (parallel a b)

-- The correct propositions are 2 and 4
theorem correct_propositions :
  (proposition2 parallel perpendicular) ∧ (proposition4 parallel perpendicular) := 
sorry

end correct_propositions_l68_68720


namespace complement_of_alpha_l68_68126

-- Define that the angle α is given as 44 degrees 36 minutes
def alpha : ℚ := 44 + 36 / 60  -- using rational numbers to represent the degrees and minutes

-- Define the complement function
def complement (angle : ℚ) : ℚ := 90 - angle

-- State the proposition to prove
theorem complement_of_alpha : complement alpha = 45 + 24 / 60 := 
by
  sorry

end complement_of_alpha_l68_68126


namespace correct_statements_count_l68_68882

theorem correct_statements_count : 
  let statements_correct :=
    [ 
      "The probability of a random event A is the stable value of frequency, and frequency is an approximate value of probability.",
      "In a single experiment, different basic events cannot occur at the same time.",
      "The probability of any event A, P(A), satisfies 0 < P(A) < 1.",
      "If the probability of event A approaches 0, then event A is an impossible event."
    ]
  in 
    (statements_correct.filter
      (λ s, match s with 
            | "The probability of a random event A is the stable value of frequency, and frequency is an approximate value of probability." => true
            | "In a single experiment, different basic events cannot occur at the same time." => true
            | "The probability of any event A, P(A), satisfies 0 < P(A) < 1." => false
            | "If the probability of event A approaches 0, then event A is an impossible event." => false
            | _ => false
      )
    ).length = 2 :=
sorry

end correct_statements_count_l68_68882


namespace sum_of_possible_k_values_l68_68240

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l68_68240


namespace Tn_lt_Sn_div2_l68_68297

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68297


namespace no_four_digit_perfect_square_palindromes_l68_68969

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68969


namespace elongation_rate_significantly_improved_l68_68024

noncomputable def elongation_improvement : Prop :=
  let x : List ℝ := [545, 533, 551, 522, 575, 544, 541, 568, 596, 548]
  let y : List ℝ := [536, 527, 543, 530, 560, 533, 522, 550, 576, 536]
  let z := List.zipWith (λ xi yi => xi - yi) x y
  let n : ℝ := 10
  let mean_z := (List.sum z) / n
  let variance_z := (List.sum (List.map (λ zi => (zi - mean_z)^2) z)) / n
  mean_z = 11 ∧ 
  variance_z = 61 ∧ 
  mean_z ≥ 2 * Real.sqrt (variance_z / n)

-- We state the theorem without proof
theorem elongation_rate_significantly_improved : elongation_improvement :=
by
  -- Proof can be written here
  sorry

end elongation_rate_significantly_improved_l68_68024


namespace Tn_lt_half_Sn_l68_68342

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68342


namespace verify_exact_countries_attended_l68_68121

theorem verify_exact_countries_attended :
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  (attended_countries = 68) :=
by
  let start_year := 1990
  let years_between_festivals := 3
  let total_festivals := 12
  let attended_countries := 68
  have : attended_countries = 68 := rfl
  exact this

end verify_exact_countries_attended_l68_68121


namespace solution_set_of_inequality_l68_68849

theorem solution_set_of_inequality (x : ℝ) : (1 / 2 < x ∧ x < 1) ↔ (x / (2 * x - 1) > 1) :=
by { sorry }

end solution_set_of_inequality_l68_68849


namespace bread_slices_l68_68861

theorem bread_slices (c : ℕ) (cost_each_slice_in_cents : ℕ)
  (total_paid_in_cents : ℕ) (change_in_cents : ℕ) (n : ℕ) (slices_per_loaf : ℕ) :
  c = 3 →
  cost_each_slice_in_cents = 40 →
  total_paid_in_cents = 2 * 2000 →
  change_in_cents = 1600 →
  total_paid_in_cents - change_in_cents = n * cost_each_slice_in_cents →
  n = c * slices_per_loaf →
  slices_per_loaf = 20 :=
by sorry

end bread_slices_l68_68861


namespace minimum_value_of_PF_plus_PA_is_3_l68_68182

-- Definitions of the given conditions
def parabola (C : Type) : Prop := ∃ y x : ℝ, y^2 = -4 * x
def point (A : Type) : Prop := ∃ x y : ℝ, A = (-2, 1)
def moving_point_on_parabola (P : Type) (C : Type) : Prop := ∃ y x : ℝ, y^2 = -4 * x
def focus (F : Type) (C : Type) : Prop := ∃ x y : ℝ, F = (0, 0)  -- Focus for parabola y^2 = -4x is (0, 0)

-- Proven statement
theorem minimum_value_of_PF_plus_PA_is_3
  (P A F : Type)
  (HC : parabola C)
  (HA : point A)
  (HP : moving_point_on_parabola P C)
  (HF : focus F C) :
  ∃ m : ℝ, m = 3 ∧ ∀ P : Type, true → |PF| + |PA| ≥ m := 
sorry

end minimum_value_of_PF_plus_PA_is_3_l68_68182


namespace ganesh_overall_average_speed_l68_68550

noncomputable def average_speed (D : ℝ) : ℝ :=
  let speed_xy := 43
  let speed_yx := 34
  let time_xy := D / speed_xy
  let time_yx := D / speed_yx
  let total_distance := 2 * D
  let total_time := time_xy + time_yx
  total_distance / total_time

theorem ganesh_overall_average_speed (D : ℝ) (hD : D > 0) : 
  average_speed D = 2924 / 77 :=
by
  unfold average_speed
  linarith [hD]

end ganesh_overall_average_speed_l68_68550


namespace georgie_ghost_ways_l68_68920

-- Define the total number of windows and locked windows
def total_windows : ℕ := 8
def locked_windows : ℕ := 2

-- Define the number of usable windows
def usable_windows : ℕ := total_windows - locked_windows

-- Define the theorem to prove the number of ways Georgie the Ghost can enter and exit
theorem georgie_ghost_ways :
  usable_windows * (usable_windows - 1) = 30 := by
  sorry

end georgie_ghost_ways_l68_68920


namespace tn_lt_sn_div_2_l68_68307

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68307


namespace T_always_positive_l68_68290

noncomputable def expression_T (α : ℝ) : ℝ :=
  (sin α + tan α) / (cos α + cot α)

theorem T_always_positive (α : ℝ) (h : α ≠ k * Real.pi / 2 ∀ k : ℤ) : expression_T α > 0 :=
sorry

end T_always_positive_l68_68290


namespace julia_trip_time_l68_68802

variable (v : ℝ)
variable (time_mountain_pass distance_mountain_pass distance_freeway : ℝ)
variable (time_mountain_pass == 40)
variable (distance_mountain_pass == 20)
variable (distance_freeway == 60)
variable (speed_freeway == 4 * v)
variable (v == distance_mountain_pass / time_mountain_pass)

theorem julia_trip_time (h1 : time_mountain_pass == (distance_mountain_pass / v)) (h2 : speed_freeway == 4 * v) : 
  (time_mountain_pass + distance_freeway / speed_freeway) == 70 := by
    sorry

end julia_trip_time_l68_68802


namespace max_distance_pq_l68_68823

-- Define the circle and ellipse
def is_on_circle (P : ℝ × ℝ) : Prop :=
  P.1 ^ 2 + (P.2 - 1) ^ 2 = 3

def is_on_ellipse (Q : ℝ × ℝ) : Prop :=
  (Q.1 ^ 2) / 4 + Q.2 ^ 2 = 1

-- Define the function to calculate the distance between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)

-- Prove the maximum distance
theorem max_distance_pq : 
  ∀ (P Q : ℝ × ℝ), is_on_circle P → is_on_ellipse Q → 
  distance P Q ≤ 7 * Real.sqrt 3 / 3 := 
  by sorry

end max_distance_pq_l68_68823


namespace find_xy_log9_2_l68_68731

theorem find_xy_log9_2 (x y : ℝ) (hx : 16^x = 4) (hy : 9^y = 4) : 
  x * y = log 9 2 := 
sorry

end find_xy_log9_2_l68_68731


namespace Tn_lt_Sn_div2_l68_68299

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68299


namespace train_cross_platform_time_l68_68019

def train_length : ℝ := 150
def signal_pole_cross_time : ℝ := 18
def platform_length : ℝ := 175

def train_speed : ℝ := train_length / signal_pole_cross_time
def total_distance : ℝ := train_length + platform_length
def time_to_cross_platform : ℝ := total_distance / train_speed

theorem train_cross_platform_time :
  abs (time_to_cross_platform - 39) < 1 :=
by
  sorry

end train_cross_platform_time_l68_68019


namespace average_salary_of_all_workers_l68_68383

def totalTechnicians : Nat := 6
def avgSalaryTechnician : Nat := 12000
def restWorkers : Nat := 6
def avgSalaryRest : Nat := 6000
def totalWorkers : Nat := 12
def totalSalary := (totalTechnicians * avgSalaryTechnician) + (restWorkers * avgSalaryRest)

theorem average_salary_of_all_workers : totalSalary / totalWorkers = 9000 := 
by
    -- replace with mathematical proof once available
    sorry

end average_salary_of_all_workers_l68_68383


namespace runners_near_stands_l68_68914

theorem runners_near_stands (track_length : ℝ) (arc_length : ℝ) 
  (n_runners : ℕ) (speeds : Fin n_runners → ℝ) :
  track_length = 2 →
  arc_length = 0.1 →
  n_runners = 20 →
  (∀ i, speeds i ∈ Set.range (λ n : ℕ, (10 + n : ℝ)) (Finset.range 20)) →
  ¬ (∀ t, ∃ i, let distance := speeds i * t % track_length in 
               distance < arc_length ∨ distance > track_length - arc_length) :=
by
  intros h1 h2 h3 h4
  sorry

end runners_near_stands_l68_68914


namespace probability_m_eq_2n_l68_68917

theorem probability_m_eq_2n :
  let outcomes := [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                   (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6),
                   (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6),
                   (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6),
                   (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6),
                   (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)] in
  let count_m_eq_2n := (outcomes.countp (λ (m, n) => m = 2 * n)) in
  let total_outcomes := 36 in
  count_m_eq_2n / total_outcomes = 1 / 12 :=
by
  sorry

end probability_m_eq_2n_l68_68917


namespace sum_of_possible_x_l68_68847

theorem sum_of_possible_x 
  (x : ℝ) 
  (h : 2 * (x - 2)^2 = (x - 3) * (x + 4)) : 
  finset.sum {x | 2 * (x - 2)^2 = (x - 3) * (x + 4)} id = 9 := 
begin
  sorry
end

end sum_of_possible_x_l68_68847


namespace eccentricity_value_l68_68166

noncomputable def eccentricity_of_hyperbola (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) : ℝ :=
  sorry

-- Statement of the problem
theorem eccentricity_value {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b)
    (h₂: ∃ P: ℝ × ℝ, P.1 = 2 * a ∧ ∃ l: ℝ, y = (b / a) * x)
    (f1 : ℝ × ℝ) (f2 : ℝ × ℝ)
    (angle_pf2f1 : ℝ)
    (hcos_angle : cos(angle_pf2f1) = -1 / 4) :
    eccentricity_of_hyperbola a b h₀ h₁ = 16 / 11 :=
  sorry

end eccentricity_value_l68_68166


namespace reduced_number_l68_68032

theorem reduced_number (N : ℕ) (m a n : ℕ) (k : ℕ) (h1 : N = m + 10^k * a + 10^(k+1) * n)
  (h2 : a < 10) (h3 : m < 10^k) (h4 : N' = m + 10^(k+1) * n) (h5 : N = 6 * N') :
  N ∈ {12, 24, 36, 48} :=
sorry

end reduced_number_l68_68032


namespace Tn_lt_Sn_over_2_l68_68323

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68323


namespace percentage_of_absent_students_l68_68749

theorem percentage_of_absent_students (total_students boys girls : ℕ) (fraction_boys_absent fraction_girls_absent : ℚ)
  (total_students_eq : total_students = 180)
  (boys_eq : boys = 120)
  (girls_eq : girls = 60)
  (fraction_boys_absent_eq : fraction_boys_absent = 1/6)
  (fraction_girls_absent_eq : fraction_girls_absent = 1/4) :
  let boys_absent := fraction_boys_absent * boys
  let girls_absent := fraction_girls_absent * girls
  let total_absent := boys_absent + girls_absent
  let absent_percentage := (total_absent / total_students) * 100
  abs (absent_percentage - 19) < 1 :=
by
  sorry

end percentage_of_absent_students_l68_68749


namespace sum_possible_values_k_l68_68255

theorem sum_possible_values_k :
  (∀ j k : ℕ, j > 0 → k > 0 → (1 / j + 1 / k = 1 / 5) → k ∈ {30, 10, 6}) →
  ∑ k in {30, 10, 6}, k = 46 :=
by {
  assume h,
  sorry
}

end sum_possible_values_k_l68_68255


namespace range_floor_function_l68_68389

theorem range_floor_function :
  (∀ x, -2.5 < x ∧ x ≤ 3 → (f x = Int.floor x)) →
  (∃ y, y ∈ set.range (λ x : ℝ, ∃ h: -2.5 < x ∧ x ≤ 3, Int.floor x = y)) =
  ( -3 ∧ -2 ∧ -1 ∧ 0 ∧ 1 ∧ 2 ∧ 3) :=
by
  sorry

end range_floor_function_l68_68389


namespace min_value_l68_68405

variable {X : ℝ → ℝ} {σ : ℝ} (μ := 10)

-- The random variable X follows a normal distribution N(10, σ^2)
axiom norm_dist : ∀ x, X x = μ + σ * (Gaussian pdf 0 1).normalize

-- Probabilities given in the problem
variable (m n : ℝ)

-- The given probabilities P(X > 12) = m and P(8 ≤ X < 10) = n
variable [Pm : m = P (X > 12)]
variable [Pn : n = P (8 ≤ X < 10)]

theorem min_value : ∃ (m n : ℝ), m = P (X > 12) ∧ n = P (8 ≤ X < 10) ∧ (2 / m + 1 / n = 6 + 4 * real.sqrt 2) := sorry

end min_value_l68_68405


namespace find_q_0_l68_68350


variable (p q r : Polynomial ℝ)

theorem find_q_0 
  (h1 : r = p * q)
  (h2 : constant_term p = 5)
  (h3 : leading_coeff p = 2)
  (h4 : constant_term r = -15) : q.eval 0 = -3 := by
  sorry

end find_q_0_l68_68350


namespace nancy_total_nv_l68_68796

-- Define initial conditions and assumptions
def num_initial_carrots_picked := 12
def num_carrots_kept_immediate := 2
def num_carrot_seeds_planted := 5
def num_initial_carrots_kept_away := 10
def growth_factor := 3
def base_nv_per_seed := 1
def nv_rate_per_cm := 0.5
def avg_growth_cm_per_carrot := 12
def discard_rate := 3

-- Calculation of new carrots and their nutritional value
def num_new_carrots := num_carrot_seeds_planted * growth_factor
def nv_per_new_carrot := avg_growth_cm_per_carrot * nv_rate_per_cm
def total_nv_new_carrots := num_new_carrots * nv_per_new_carrot
def total_nv_kept_carrots := num_carrots_kept_immediate * base_nv_per_seed

-- Summing up the total number of carrots and their usable count after discarding poor quality
def total_carrots := num_initial_carrots_kept_away + num_new_carrots
def num_poor_quality_carrots := total_carrots / discard_rate
def num_good_quality_carrots := total_carrots - num_poor_quality_carrots

-- Total nutritional value after accounting for poor quality
def total_nv := total_nv_new_carrots + total_nv_kept_carrots

theorem nancy_total_nv : total_nv = 92 := by
  sorry

end nancy_total_nv_l68_68796


namespace work_schedules_lcm_l68_68602

theorem work_schedules_lcm : Nat.lcm (Nat.lcm 5 6) (Nat.lcm 8 9) = 360 := 
by 
  sorry

end work_schedules_lcm_l68_68602


namespace find_m_value_l68_68687

-- Definitions for the problem conditions are given below
variables (m : ℝ)

-- Conditions
def conditions := (6 < m) ∧ (m < 10) ∧ (4 = 2 * 2) ∧ (4 = (m - 2) - (10 - m))

-- Proof statement
theorem find_m_value : conditions m → m = 8 :=
sorry

end find_m_value_l68_68687


namespace Tn_lt_half_Sn_l68_68348

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68348


namespace brand_ratio_l68_68753

theorem brand_ratio (total_respondents : ℕ) (preferred_X : ℕ) (preferred_Y : ℕ)
  (h_total : total_respondents = 400)
  (h_preferred_X : preferred_X = 360)
  (h_preferred_Y : preferred_Y = total_respondents - preferred_X) : 
  preferred_X / preferred_Y = 9 := 
by
  rw [h_total, h_preferred_X, h_preferred_Y]
  -- proof steps would go here, ending with "sorry" to skip the actual proof
  sorry

end brand_ratio_l68_68753


namespace tn_lt_sn_div_2_l68_68302

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68302


namespace original_speed_l68_68351

   -- Define the original problem
   def circumference_feet : ℝ := 15
   def time_decrement_seconds : ℝ := 1 / 3
   def speed_increment_mph : ℝ := 4

   def feet_to_miles (feet : ℝ) : ℝ := feet / 5280
   def seconds_to_hours (sec : ℝ) : ℝ := sec / 3600

   -- Circumference in miles
   def circumference_miles : ℝ := feet_to_miles circumference_feet

   -- Prove the original speed
   theorem original_speed (r : ℝ) (t : ℝ) :
     (r * (t - seconds_to_hours time_decrement_seconds) = circumference_miles * 3600 ∧
      (r + speed_increment_mph) * t = circumference_miles * 3600) →
     r = 12 :=
   sorry
   
end original_speed_l68_68351


namespace max_tank_volume_l68_68049

noncomputable def tank_volume (r : ℝ) : ℝ := (π / 5) * (300 * r - 4 * r^3)

theorem max_tank_volume (r : ℝ) (h : ℝ) 
  (h_cost: 200 * π * r * h + 160 * π * r^2 = 12000 * π)
  (h_r_pos : 0 < r) (h_r_dom : r < 5 * Real.sqrt 3)
  (h_height : h = (1 / (5 * r)) * (300 - 4 * r^2)) :
  ∃ r' h', h' = 8 ∧ r' = 5 ∧ tank_volume r' = 200 * π :=
begin
  sorry
end

end max_tank_volume_l68_68049


namespace product_probability_less_than_50_l68_68808

theorem product_probability_less_than_50 :
  let Paco := Finset.range (6 + 1)
  let Manu := Finset.range (15 + 1)
  let favorable_events := Paco.product Manu filter (λ p, p.1 * p.2 < 50)
  let total_events := Paco.card * Manu.card
  favorable_events.card = 37 ∧ total_events = 45 → 
  mkRat favorable_events.card total_events = Rat.mk 37 45 :=
by
  sorry

end product_probability_less_than_50_l68_68808


namespace necessary_and_sufficient_condition_l68_68669

theorem necessary_and_sufficient_condition (a b : ℝ) : 
  (a > 0 ∧ b > 0) ↔ (a + b > 0 ∧ a * b > 0) := 
by 
  sorry

end necessary_and_sufficient_condition_l68_68669


namespace total_machines_sold_130_l68_68586

noncomputable def total_machines_sold (commission: ℝ) : ℕ :=
  let first_100_commission   := 100 * 0.03 * 10000
  let remaining_commission   := commission - first_100_commission
  let machines_after_100     := remaining_commission / (0.04 * 10000)
  100 + floor machines_after_100

theorem total_machines_sold_130 :
  total_machines_sold 42000 = 130 :=
by
  unfold total_machines_sold
  have h1 : 100 * 0.03 * 10000 = 30000 := by norm_num
  rw [h1]
  have h2 : 42000 - 30000 = 12000 := by norm_num
  rw [h2]
  have h3 : 0.04 * 10000 = 400 := by norm_num
  rw [h3]
  have h4 : 12000 / 400 = 30 := by norm_num
  rw [h4]
  have h5 : 100 + 30 = 130 := by norm_num
  rw [h5]
  simp
  sorry

end total_machines_sold_130_l68_68586


namespace find_integer_to_fifth_power_l68_68194

theorem find_integer_to_fifth_power :
  (∃ x : ℕ, x^5 = 2^(10 + 15)) → 32^5 = 2^(10 + 15) :=
by
  intros h
  match h with
  | ⟨x, hx⟩ =>
    have hx_eq : x = 32 := sorry
    rw [hx_eq]
    rfl

end find_integer_to_fifth_power_l68_68194


namespace triangle_square_ratio_l68_68061

theorem triangle_square_ratio (t s : ℝ) 
  (h1 : 3 * t = 15) 
  (h2 : 4 * s = 12) : 
  t / s = 5 / 3 :=
by 
  -- skipping the proof
  sorry

end triangle_square_ratio_l68_68061


namespace range_of_a_l68_68150

noncomputable def f (x : ℝ) : ℝ := real.log x + 1 / x - 5
noncomputable def g (x a : ℝ) : ℝ := x^2 - 2 * a * x

theorem range_of_a (a : ℝ) : 
  (∀ x1 ∈ Set.Ioi 0, ∃ x2 : ℝ, f x1 > g x2 a) ↔ a ∈ Set.Iio -2 ∪ Set.Ioi 2 :=
by
  sorry

end range_of_a_l68_68150


namespace day_53_days_from_friday_l68_68445

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68445


namespace count_palindromic_four_digit_perfect_squares_l68_68983

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68983


namespace num_solutions_eq_6_l68_68728

theorem num_solutions_eq_6 :
  ∃ (n : ℕ), n = 6 ∧
    (finset.card (finset.filter (λ ⟨x, y⟩, x + y + 2 * x * y = 2023)
      (finset.product (finset.range 2024) (finset.range 2024)))) = n := 
sorry

end num_solutions_eq_6_l68_68728


namespace days_from_friday_l68_68508

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68508


namespace indistinguishable_times_l68_68834

theorem indistinguishable_times : ∀ t ∈ Ico 0 12, 
  (fractional_part (t / 12) = fractional_part (hour_to_minute t)) ↔ 
  (number_of_times t = 132) := sorry

end indistinguishable_times_l68_68834


namespace ellipse_equation_area_range_l68_68125

-- Define the ellipse
def ellipse (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b

-- Point on ellipse P(-1, √2/2)
def point_on_ellipse (a b : ℝ) (P : ℝ × ℝ) : Prop :=
  P = (-1, real.sqrt 2 / 2) ∧ (P.1^2 / a^2 + P.2^2 / b^2 = 1)

-- Midpoint condition
def midpoint_condition (P F2 M : ℝ × ℝ) : Prop :=
  M = ((P.1 + F2.1) / 2, (P.2 + F2.2) / 2)

-- Equation fitting for ellipse
theorem ellipse_equation (a b c : ℝ) (P M F1 F2 : ℝ × ℝ) :
  ellipse a b → point_on_ellipse a b P → midpoint_condition P F2 M →
  (c = 1 ∧ 1 / a^2 + 1 / (2 * b^2) = 1 ∧ a^2 = b^2 + c^2) →
  (∃ x y : ℝ, x^2 / 2 + y^2 = 1) :=
by
  sorry

-- Dot product and area range conditions
def dot_product_condition (OA OB : ℝ × ℝ) (λ : ℝ) : Prop :=
  (OA.1 * OB.1 + OA.2 * OB.2 = λ)

theorem area_range (λ : ℝ) (S : ℝ) :
  (2 / 3 ≤ λ ∧ λ ≤ 3 / 4) →
  (sqrt 6 / 4 ≤ S ∧ S ≤ 2 / 3) :=
by
  sorry

end ellipse_equation_area_range_l68_68125


namespace practice_hours_l68_68611

-- Define the starting and ending hours, and the break duration
def start_hour : ℕ := 8
def end_hour : ℕ := 16
def break_duration : ℕ := 2

-- Compute the total practice hours
def total_practice_time : ℕ := (end_hour - start_hour) - break_duration

-- State that the computed practice time is equal to 6 hours
theorem practice_hours :
  total_practice_time = 6 := 
by
  -- Using the definitions provided to state the proof
  sorry

end practice_hours_l68_68611


namespace transformed_roots_l68_68732

noncomputable def specific_polynomial : Polynomial ℝ :=
  Polynomial.C 1 - Polynomial.C 4 * Polynomial.X + Polynomial.C 6 * Polynomial.X ^ 2 - Polynomial.C 4 * Polynomial.X ^ 3 + Polynomial.C 1 * Polynomial.X ^ 4

theorem transformed_roots (a b c d : ℝ) :
  (a^4 - b*a - 5 = 0) ∧ (b^4 - b*b - 5 = 0) ∧ (c^4 - b*c - 5 = 0) ∧ (d^4 - b*d - 5 = 0) →
  specific_polynomial.eval ((a + b + c) / d)^2 = 0 ∧
  specific_polynomial.eval ((a + b + d) / c)^2 = 0 ∧
  specific_polynomial.eval ((a + c + d) / b)^2 = 0 ∧
  specific_polynomial.eval ((b + c + d) / a)^2 = 0 :=
  by
    sorry

end transformed_roots_l68_68732


namespace days_from_friday_l68_68505

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68505


namespace jane_can_afford_book_l68_68272

theorem jane_can_afford_book (n : ℕ) : 4 * 10 + 5 * 0.25 + n * 0.05 ≥ 42.50 → n ≥ 25 :=
  by
    intro h
    calc
      0.05 * n ≥ 42.50 - (4 * 10 + 5 * 0.25) : by linarith
      0.05 * n ≥ 1.25 : by norm_num
      n ≥ 25 : by linarith

-- Proof is not required as specified in the task statement.

end jane_can_afford_book_l68_68272


namespace x_varies_as_half_power_of_z_l68_68203

variable {x y z : ℝ} -- declare variables as real numbers

-- Assume the conditions, which are the relationships between x, y, and z
variable (k j : ℝ) (k_pos : k > 0) (j_pos : j > 0)
axiom xy_relationship : ∀ y, x = k * y^2
axiom yz_relationship : ∀ z, y = j * z^(1/4)

-- The theorem we want to prove
theorem x_varies_as_half_power_of_z (z : ℝ) (h : z ≥ 0) : ∃ m, m > 0 ∧ x = m * z^(1/2) :=
sorry

end x_varies_as_half_power_of_z_l68_68203


namespace area_enclosed_curves_l68_68851

theorem area_enclosed_curves (a : ℝ) (h1 : (1 + 1/a)^5 = 1024) :
  ∫ x in (0 : ℝ)..1, (x^(1/3) - x^2) = 5/12 :=
sorry

end area_enclosed_curves_l68_68851


namespace fifty_three_days_from_friday_is_tuesday_l68_68474

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68474


namespace Tn_lt_half_Sn_l68_68344

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68344


namespace count_palindromic_four_digit_perfect_squares_l68_68981

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68981


namespace log_ratio_l68_68172

theorem log_ratio (x y : ℝ) (h : 2 * log (x - 2 * y) = log x + log y) : x / y = 4 ∨ x / y = 1 :=
by sorry

end log_ratio_l68_68172


namespace train_travel_distance_l68_68574

theorem train_travel_distance (m : ℝ) (h : 3 * 60 * 1 = m) : m = 180 :=
by
  sorry

end train_travel_distance_l68_68574


namespace unique_four_digit_palindromic_square_l68_68926

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68926


namespace function_minimum_value_function_minimum_value_equality_l68_68853

open Real

theorem function_minimum_value (x : ℝ) (hx : 0 < x) : x^2 + 2 / x ≥ 3 :=
begin
  -- Proof here
  sorry
end

theorem function_minimum_value_equality (x : ℝ) (hx : 0 < x) : x^2 + 2 / x = 3 ↔ x = 1 :=
begin
  -- Proof here
  sorry
end

end function_minimum_value_function_minimum_value_equality_l68_68853


namespace area_triangle_MNI_l68_68762
noncomputable theory

-- Define points A, B, and C
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 0)
def C : ℝ × ℝ := (24, 0)

-- Define midpoints M and N
def M : ℝ × ℝ := ((B.1 + C.1) / 2, (B.2 + C.2) / 2) -- Midpoint of BC
def N : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) -- Midpoint of AB

-- Define point I by using the section formula
def ratio : ℝ := 5 / 18
def I : ℝ × ℝ := (ratio * C.1 + (1 - ratio) * A.1, ratio * C.2 + (1 - ratio) * A.2)

-- Function to calculate the area of a triangle given three points
def area (P Q R : ℝ × ℝ) : ℝ := 0.5 * |P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)|

-- Theorem stating the area of triangle MNI
theorem area_triangle_MNI : area M N I = 30 := 
sorry

end area_triangle_MNI_l68_68762


namespace find_s_l68_68196

theorem find_s (s t : ℝ) (h1 : 8 * s + 4 * t = 160) (h2 : t = 2 * s - 3) : s = 10.75 :=
by
  sorry

end find_s_l68_68196


namespace selling_price_is_correct_l68_68274

noncomputable def original_cost : ℝ := 496.30
noncomputable def profit_percentage : ℝ := 0.35
noncomputable def profit : ℝ := original_cost * profit_percentage
noncomputable def selling_price : ℝ := original_cost + profit
noncomputable def rounded_selling_price : ℝ := Real.round(selling_price * 100) / 100

theorem selling_price_is_correct :
  rounded_selling_price = 670.01 :=
by
  sorry

end selling_price_is_correct_l68_68274


namespace part1_part2_l68_68174

def f (x : ℝ) := 2 * x^3 - 3 * x^2 + 1
def g (k x : ℝ) := k * x + 1 - Math.log x
def h (k x : ℝ) := min (f x) (g k x)

theorem part1 (a : ℝ) :
  (∃ (P : ℝ × ℝ), P = (a, -4) ∧ 
   (∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
    (∃ (k₁ k₂ : ℝ), 
     k₁ = 6 * t₁^2 - 6 * t₁ ∧ k₂ = 6 * t₂^2 - 6 * t₂ ∧
     (∃ (l₁ l₂ : ℝ), 
      l₁ = f t₁ - k₁ * t₁ ∧ l₂ = f t₂ - k₂ * t₂ ∧
      (-4 = k₁ * a + l₁) ∧ (-4 = k₂ * a + l₂) ∧ 
      (∀ t, (4 * t^3 - (3 + 6 * a) * t^2 + 6 * a * t - 5 = 0) → 
       t = t₁ ∨ t = t₂))
   )
  ) ↔ a = -1 ∨ a = 7 / 2 := sorry

theorem part2 (k : ℝ) :
  (∀ x : ℝ, x > 0 → 
   (∃ (x₁ x₂ x₃ : ℝ), 
    x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    h k x₁ = 0 ∧ h k x₂ = 0 ∧ h k x₃ = 0
   )
  ) ↔ 0 < k ∧ k < 1 / Real.exp 2 := sorry

end part1_part2_l68_68174


namespace discount_per_issue_l68_68607

theorem discount_per_issue
  (normal_subscription_cost : ℝ) (months : ℕ) (issues_per_month : ℕ) 
  (promotional_discount : ℝ) :
  normal_subscription_cost = 34 →
  months = 18 →
  issues_per_month = 2 →
  promotional_discount = 9 →
  (normal_subscription_cost - promotional_discount) / (months * issues_per_month) = 0.25 :=
by
  intros h1 h2 h3 h4
  sorry

end discount_per_issue_l68_68607


namespace impossible_to_ensure_continuous_coverage_l68_68913

-- Definitions for conditions
def length_of_track : ℝ := 2  -- in kilometers
def arc_length_of_stands : ℝ := 100 / 1000  -- in kilometers (100 meters)
def number_of_runners : ℕ := 20
def speeds : List ℝ := List.range number_of_runners |>.map (λ n => 10 + n)  -- speeds from 10 to 29 km/h
def passing_time_fraction : ℝ := arc_length_of_stands / length_of_track  -- fraction of time passing stands

-- Proof problem setup
theorem impossible_to_ensure_continuous_coverage :
  ∃ (positions : List ℝ), (length positions = number_of_runners) ∧
  (∀ t : ℝ, ∃ i : ℕ, i < number_of_runners ∧ (positions.nth i).val + speeds.nth i.val * t % length_of_track < arc_length_of_stands) → False :=
by
  sorry

end impossible_to_ensure_continuous_coverage_l68_68913


namespace total_cost_mangos_rice_flour_l68_68012

noncomputable def mango_cost (kg : ℕ) : ℝ := kg * 165.6
noncomputable def rice_cost (kg : ℕ) : ℝ := kg * 69
noncomputable def flour_cost (kg : ℕ) : ℝ := kg * 23

theorem total_cost_mangos_rice_flour :
  let total_cost := (4 * 165.6) + (3 * 69) + (5 * 23)
  in total_cost = 984.4 :=
by
  let total_cost := (4 * 165.6) + (3 * 69) + (5 * 23)
  have h : total_cost = 984.4 := by
    calc
      total_cost = (4 * 165.6) + (3 * 69) + (5 * 23) : rfl
               ... = 662.4 + (3 * 69) + (5 * 23)     : rfl
               ... = 662.4 + 207 + (5 * 23)          : rfl
               ... = 662.4 + 207 + 115               : rfl
               ... = 984.4                           : rfl
  exact h

end total_cost_mangos_rice_flour_l68_68012


namespace slices_remaining_l68_68214

theorem slices_remaining (large_pizza_slices : ℕ) (xl_pizza_slices : ℕ) (large_pizza_ordered : ℕ) (xl_pizza_ordered : ℕ) (mary_eats_large : ℕ) (mary_eats_xl : ℕ) :
  large_pizza_slices = 8 →
  xl_pizza_slices = 12 →
  large_pizza_ordered = 1 →
  xl_pizza_ordered = 1 →
  mary_eats_large = 7 →
  mary_eats_xl = 3 →
  (large_pizza_slices * large_pizza_ordered - mary_eats_large + xl_pizza_slices * xl_pizza_ordered - mary_eats_xl) = 10 := 
by
  intros
  sorry

end slices_remaining_l68_68214


namespace integer_pairs_satisfying_condition_l68_68087

theorem integer_pairs_satisfying_condition :
  { (m, n) : ℤ × ℤ | ∃ k : ℤ, (n^3 + 1) = k * (m * n - 1) } =
  { (1, 2), (1, 3), (2, 1), (3, 1), (2, 5), (3, 5), (5, 2), (5, 3), (2, 2) } :=
sorry

end integer_pairs_satisfying_condition_l68_68087


namespace find_number_l68_68111

theorem find_number (x : ℤ) (h : x = 1) : x + 1 = 2 :=
  by
  sorry

end find_number_l68_68111


namespace rho_square_max_value_l68_68349

variable {a b x y c : ℝ}
variable (ha_pos : a > 0) (hb_pos : b > 0)
variable (ha_ge_b : a ≥ b)
variable (hx_range : 0 ≤ x ∧ x < a)
variable (hy_range : 0 ≤ y ∧ y < b)
variable (h_eq1 : a^2 + y^2 = b^2 + x^2)
variable (h_eq2 : b^2 + x^2 = (a - x)^2 + (b - y)^2 + c^2)

theorem rho_square_max_value : (a / b) ^ 2 ≤ 4 / 3 :=
sorry

end rho_square_max_value_l68_68349


namespace solve_equation_1_solve_equation_2_l68_68819

theorem solve_equation_1 (x : Real) : 
  (1/3) * (x - 3)^2 = 12 → x = 9 ∨ x = -3 :=
by
  sorry

theorem solve_equation_2 (x : Real) : 
  (2 * x - 1)^2 = (1 - x)^2 → x = 0 ∨ x = 2/3 :=
by
  sorry

end solve_equation_1_solve_equation_2_l68_68819


namespace value_of_a_l68_68210

theorem value_of_a (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → m > 1 :=
sorry

end value_of_a_l68_68210


namespace min_distance_from_curve_to_line_l68_68370

noncomputable def point_on_curve (x : ℝ) : ℝ := x^2 - real.log x

theorem min_distance_from_curve_to_line (P : ℝ × ℝ)
  (h : P.2 = point_on_curve P.1)
  (hx_pos : 0 < P.1) :
  let line_eq := (P.1, P.2) in 
  let distance := (|(P.1 - P.2 - 4)| / real.sqrt 2) in
  distance = 2 * real.sqrt 2 :=
by
  sorry

end min_distance_from_curve_to_line_l68_68370


namespace line_segment_bisects_side_l68_68263

open EuclideanGeometry

/-- Given a triangle ABC with orthocenter H and angle bisectors BB₁ (internal) and BB₂ (external)
at vertex B. The perpendiculars from H to these bisectors intersect them at H₁ and H₂ respectively.
Prove that the line segment H₁H₂ bisects the side AC. -/
theorem line_segment_bisects_side 
  {A B C H B₁ B₂ H₁ H₂ : Point} 
  (hABC : Triangle A B C) 
  (hH : Orthocenter H A B C) 
  (hBB₁ : AngleBisector B B₁)
  (hBB₂ : ExtAngleBisector B B₂) 
  (hH₁ : Perpendicular H H₁ B B₁)
  (hH₂ : Perpendicular H H₂ B B₂) : 
  Bisects H₁ H₂ (Segment A C) :=
sorry

end line_segment_bisects_side_l68_68263


namespace max_value_of_a_l68_68634

theorem max_value_of_a :
  ∀ (m : ℚ) (x : ℤ),
    (0 < x ∧ x ≤ 50) →
    (1 / 2 < m ∧ m < 25 / 49) →
    (∀ k : ℤ, m * x + 3 ≠ k) →
  m < 25 / 49 :=
sorry

end max_value_of_a_l68_68634


namespace cyclic_inequality_l68_68558

theorem cyclic_inequality (n : ℕ) (hn : n ≥ 2) (x : Fin n → ℝ) 
  (hx : ∀ i, 0 < x i) :
  4 * (∑ i in Finset.range n, (x i)^3 - (x (i+1 % n))^3) / (x i + x (i+1 % n))
  ≤ ∑ i in Finset.range n, (x i - x (i+1 % n))^2 :=
sorry

end cyclic_inequality_l68_68558


namespace part_a_part_b_l68_68820

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.foldl (λ x y => x + y) 0

-- Part A: There exists a sequence of 158 consecutive integers where the sum of digits is not divisible by 17
theorem part_a : ∃ (n : ℕ), ∀ (k : ℕ), k < 158 → sum_of_digits (n + k) % 17 ≠ 0 := by
  sorry

-- Part B: Among any 159 consecutive integers, there exists at least one integer whose sum of digits is divisible by 17
theorem part_b : ∀ (n : ℕ), ∃ (k : ℕ), k < 159 ∧ sum_of_digits (n + k) % 17 = 0 := by
  sorry

end part_a_part_b_l68_68820


namespace problem_statement_l68_68788

noncomputable def a := Real.logBase Real.pi Real.exp 1
noncomputable def b := 2 ^ Real.cos (7 * Real.pi / 3)
noncomputable def c := Real.logBase 3 (Real.sin (17 * Real.pi / 6))

theorem problem_statement : b > a ∧ a > c := by
  sorry

end problem_statement_l68_68788


namespace zero_count_at_end_of_45_320_125_product_l68_68729

theorem zero_count_at_end_of_45_320_125_product :
  let p := 45 * 320 * 125
  45 = 5 * 3^2 ∧ 320 = 2^6 * 5 ∧ 125 = 5^3 →
  p = 2^6 * 3^2 * 5^5 →
  p % 10^5 = 0 ∧ p % 10^6 ≠ 0 :=
by
  sorry

end zero_count_at_end_of_45_320_125_product_l68_68729


namespace roots_pure_imaginary_if_negative_real_k_l68_68079

theorem roots_pure_imaginary_if_negative_real_k (k : ℝ) (h_neg : k < 0) :
  (∃ (z : ℂ), 10 * z^2 - 3 * Complex.I * z - (k : ℂ) = 0 ∧ z.im ≠ 0 ∧ z.re = 0) :=
sorry

end roots_pure_imaginary_if_negative_real_k_l68_68079


namespace triangle_bottom_row_l68_68595

theorem triangle_bottom_row :
  let initial_row := list.range' 1 2000 in
  let triangle := list.range' 1 2000 >>= (λ n, list.nth_le ({n, n-1} : finset ℕ) 0 sorry +
                                               list.nth_le ({n + 1, n} : finset ℕ) 1 sorry) in
  ∃ k, triangle[n] = 2^(2000-2) * 2001 :=
by
  sorry

end triangle_bottom_row_l68_68595


namespace max_height_reached_l68_68903

noncomputable def height (t : ℝ) : ℝ :=
  -20 * t^2 + 100 * t + 36

theorem max_height_reached : ∃ t : ℝ, height t = 161 :=
by
  sorry

end max_height_reached_l68_68903


namespace no_such_natural_numbers_l68_68642

theorem no_such_natural_numbers :
  ¬(∃ (a b c : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧
  (b ∣ a^2 - 1) ∧ (c ∣ a^2 - 1) ∧
  (a ∣ b^2 - 1) ∧ (c ∣ b^2 - 1) ∧
  (a ∣ c^2 - 1) ∧ (b ∣ c^2 - 1)) :=
by sorry

end no_such_natural_numbers_l68_68642


namespace find_smallest_n_l68_68555

theorem find_smallest_n : 
  ∃ (n : ℕ), (∀ k < n, ¬ (|S k - k - 6| < 1 / 125)) ∧ (|S n - n - 6| < 1 / 125)
:= sorry

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 9 ∧ ∀ n : ℕ, n ≥ 1 → (3 * a (n + 1) + a n = 4)

def sum_seq (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, S n = ∑ i in range (n+1), a i

noncomputable def S : ℕ → ℝ := sorry -- The exact sequence formula not required here
noncomputable def a : ℕ → ℝ := sorry -- The exact sequence formula not required here

example (h_seq : seq a) (h_sum : sum_seq S a) : 
  ∃ (n : ℕ), (∀ k < n, ¬ (|S k - k - 6| < 1 / 125)) ∧ (|S n - n - 6| < 1 / 125)
:= sorry

end find_smallest_n_l68_68555


namespace root_inequality_l68_68131

noncomputable def f (x : ℝ) : ℝ := (Real.log x / Real.log 2) - (1 / 2) ^ x

theorem root_inequality 
  (a b c x₀ : ℝ) 
  (ha : 0 < a)
  (h₁ : a < b)
  (h₂ : b < c)
  (hac : f(a) * f(b) * f(c) < 0)
  (hx₀ : f(x₀) = 0) : x₀ > a :=
sorry

end root_inequality_l68_68131


namespace smallest_addition_to_8261955_l68_68002

def odd_digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.enum.filter (λ p, p.1 % 2 = 0).sum (λ p, p.2)

def even_digit_sum (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.enum.filter (λ p, p.1 % 2 = 1).sum (λ p, p.2)

def difference_of_sums (n : ℕ) : ℤ :=
  odd_digit_sum n - even_digit_sum n

def is_divisible_by_11 (n : ℕ) : Prop :=
  difference_of_sums n % 11 = 0

theorem smallest_addition_to_8261955 :
  ∃ x : ℕ, is_divisible_by_11 (8261955 + x) ∧ ∀ y : ℕ, y < x → ¬ is_divisible_by_11 (8261955 + y) :=
sorry

end smallest_addition_to_8261955_l68_68002


namespace james_total_jail_time_l68_68269

theorem james_total_jail_time (arson_count arson_sentence explosive_multiplier domestic_terrorism_sentence : ℕ) :
    arson_count * arson_sentence + (2 * arson_count * arson_sentence) + domestic_terrorism_sentence = 56 :=
by
  -- Given conditions
  let arson_count := 2
  let arson_sentence := 6
  let explosive_multiplier := 2
  let domestic_terrorism_sentence := 20

  -- Compute the total jail time James might face
  let arson_total := arson_count * arson_sentence
  let explosive_sentence := explosive_multiplier * arson_total
  let total_sentence := arson_total + explosive_sentence + domestic_terrorism_sentence

  -- Verify the total sentence is as expected
  have h : total_sentence = 56 := sorry

  exact h

end james_total_jail_time_l68_68269


namespace Pat_worked_days_eq_57_l68_68661

def Pat_earnings (x : ℕ) : ℤ := 100 * x
def Pat_food_costs (x : ℕ) : ℤ := 20 * (70 - x)
def total_balance (x : ℕ) : ℤ := Pat_earnings x - Pat_food_costs x

theorem Pat_worked_days_eq_57 (x : ℕ) (h : total_balance x = 5440) : x = 57 :=
by
  sorry

end Pat_worked_days_eq_57_l68_68661


namespace count_non_increasing_5digit_numbers_l68_68724

theorem count_non_increasing_5digit_numbers : 
  ∃ n : ℕ, n = 715 ∧ ∀ (a b c d e : ℕ), 
  a ≥ b ∧ b ≥ c ∧ c ≥ d ∧ d ≥ e ∧ a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10 ∧ e < 10 ∧ a > 0 →
  -- count the number of such sequences --
  n = (Nat.choose (9 + 5 - 1) (5 - 1)) :=
begin
  sorry
end

end count_non_increasing_5digit_numbers_l68_68724


namespace Prakash_work_in_2_hours_l68_68378

variables (Subash_rate : ℝ) (combined_rate : ℝ) (time_interval : ℝ) 
variables (Subash_work : ℝ) (Prakash_work : ℝ)
variables (time_Subash : ℝ) (time_combined : ℝ)

-- Assume Subash's rate of copying is 50 pages in 10 hours.
def Subash_copies_in_10_hours : Prop :=
  Subash_rate = 50 / 10

-- Assume Subash and Prakash together can copy 300 pages in 40 hours.
def combined_copies_in_40_hours : Prop :=
  combined_rate = 300 / 40

-- Calculate work done by Subash in 40 hours.
def Subash_work_in_40_hours : Prop := 
  Subash_work = Subash_rate * 40

-- Determine the total work done by Prakash in 40 hours.
def Prakash_work_in_40_hours : Prop := 
  Prakash_work = (combined_rate * 40) - (Subash_rate * 40)

-- Determine Prakash's productivity per 2-hour interval and verify it's 5 pages.
theorem Prakash_work_in_2_hours
  (h1 : Subash_copies_in_10_hours Subash_rate)
  (h2 : combined_copies_in_40_hours combined_rate)
  (h3 : Subash_work_in_40_hours Subash_work)
  (h4 : Prakash_work_in_40_hours Prakash_work)
  (time_interval : ℝ := 2) :
  (Prakash_work / (40 / time_interval)) = 5 :=
sorry

end Prakash_work_in_2_hours_l68_68378


namespace day_of_week_in_53_days_l68_68496

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68496


namespace day_53_days_from_friday_l68_68443

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68443


namespace slopes_not_equal_l68_68078

noncomputable def point := (ℝ × ℝ)

noncomputable def A (s : ℝ) : point := (0, 0)
noncomputable def B (s : ℝ) : point := (s, 0)
noncomputable def C (s : ℝ) : point := (s, s)
noncomputable def D (s : ℝ) : point := (0, s)

noncomputable def reflect_y_equals_x (p : point) : point :=
  (p.2, p.1)

noncomputable def A' (s : ℝ) : point := reflect_y_equals_x (A s)
noncomputable def B' (s : ℝ) : point := reflect_y_equals_x (B s)
noncomputable def C' (s : ℝ) : point := reflect_y_equals_x (C s)
noncomputable def D' (s : ℝ) : point := reflect_y_equals_x (D s)

noncomputable def slope (p1 p2 : point) : ℝ :=
  (p2.2 - p1.2) / (p2.1 - p1.1)

theorem slopes_not_equal (s : ℝ) (h : s ≠ 0) :
  slope (A s) (D s) ≠ slope (A' s) (D' s) :=
by
  sorry

end slopes_not_equal_l68_68078


namespace isosceles_triangle_perimeter_l68_68145

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l68_68145


namespace a_is_even_positive_integer_l68_68388

noncomputable def quadratic_is_factored (a : ℕ) : Prop :=
  ∃ m n p q : ℕ,
  21 * m * p = 21 ∧ -- m * p = 21
  m * q + n * p = a ∧ -- mq + np = a
  n * q = 21 -- n * q = 21

theorem a_is_even_positive_integer (a : ℕ) 
  (h : quadratic_is_factored a) :
  ∃ k : ℕ, a = 2 * k ∧ k > 0 :=
begin
  sorry
end

end a_is_even_positive_integer_l68_68388


namespace range_of_a_l68_68181

open Real

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, a ≤ abs (x - 5) + abs (x - 3)) → a ≤ 2 := by
  sorry

end range_of_a_l68_68181


namespace children_flags_l68_68552

variable {T : Type} [Group T]

theorem children_flags (E N : ℕ) 
  (h1 : E = N / 5) 
  (h2 : 0 = N / 2) 
  (h3 : N' = 6 * E) 
  (h4 : 0 = 3 * E) :
  ∃ (k : ℕ), k ≤ T / 2 → k ≠ (N - k) :=
by 
  sorry

end children_flags_l68_68552


namespace ratio_time_A_to_B_l68_68751

-- Definition of total examination time in minutes
def total_time : ℕ := 180

-- Definition of time spent on type A problems
def time_A : ℕ := 40

-- Definition of time spent on type B problems as total_time - time_A
def time_B : ℕ := total_time - time_A

-- Statement that we need to prove
theorem ratio_time_A_to_B : time_A * 7 = time_B * 2 :=
by
  -- Implementation of the proof will go here
  sorry

end ratio_time_A_to_B_l68_68751


namespace solve_fraction_eq_l68_68818

theorem solve_fraction_eq (x : ℚ) (h : (x^2 + 3 * x + 4) / (x + 3) = x + 6) : x = -7 / 3 :=
sorry

end solve_fraction_eq_l68_68818


namespace Rademacher_definition_l68_68817

def Rademacher_function (n : ℕ) (x : ℝ) : ℝ :=
  if (0 ≤ x ∧ x ≤ 1 ∧ n > 0) then 
    sign (sin (2 ^ n * π * x)) 
  else 
    0

theorem Rademacher_definition (x : ℝ) (n : ℕ) (h₀ : 0 ≤ x) (h₁ : x ≤ 1) (h₂ : n > 0) :
  Rademacher_function n x = sign (sin (2 ^ n * π * x)) := 
by
  sorry

end Rademacher_definition_l68_68817


namespace good_permutations_bound_l68_68837

/-!
  Prove that the number of good arrangements (those that do not contain any subsequence of 10 numbers in decreasing order) of 
  the set {1, 2, ..., n} does not exceed 81^n.
 -/

-- Define the concept of a good permutation
def good_permutation (n : ℕ) (perm : list ℕ) : Prop :=
  ∀ seq : list ℕ, seq.length = 10 → seq <+ perm → ¬(strictly_decreasing seq)

-- Count the good permutations
noncomputable def count_good_permutations (n : ℕ) : ℕ :=
  (list.permutations (list.range n)).count (good_permutation n)

-- The main theorem
theorem good_permutations_bound (n : ℕ) : count_good_permutations n ≤ 81^n :=
sorry

end good_permutations_bound_l68_68837


namespace john_ate_half_package_l68_68034

def fraction_of_package_john_ate (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) : ℚ :=
  calories_consumed / (servings * calories_per_serving : ℚ)

theorem john_ate_half_package (servings : ℕ) (calories_per_serving : ℕ) (calories_consumed : ℕ) 
    (h_servings : servings = 3) (h_calories_per_serving : calories_per_serving = 120) (h_calories_consumed : calories_consumed = 180) :
    fraction_of_package_john_ate servings calories_per_serving calories_consumed = 1 / 2 :=
by
  -- Replace the actual proof with sorry to ensure the statement compiles.
  sorry

end john_ate_half_package_l68_68034


namespace angle_sum_difference_l68_68583

variables {x y : ℝ}

def point_O_inside_square :=  (x > -1) ∧ (x < 1) ∧ (y > -1) ∧ (y < 1)

noncomputable def angle_sum (x y : ℝ) : ℝ :=
  let tan_OAB := (y - 1) / (x - 1) in
  let tan_OBC := (y - 1) / (x + 1) in
  let tan_OCD := (y + 1) / (x + 1) in
  let tan_ODA := (y + 1) / (x - 1) in
  atan(tan_OAB) + atan(tan_OBC) + atan(tan_OCD) + atan(tan_ODA)

theorem angle_sum_difference (hx : -1 < x) (hx' : x < 1) (hy : -1 < y) (hy' : y < 1) :
  abs (angle_sum x y - 180) ≤ 45 :=
begin
  sorry
end

end angle_sum_difference_l68_68583


namespace area_enclosed_by_polar_curves_l68_68895

noncomputable def enclosed_area : ℝ := 
  let f₁ := λ φ : ℝ, real.cos φ
  let f₂ := λ φ : ℝ, real.sqrt 2 * real.sin (φ - real.pi / 4)
  let region_1 := 1 / 2 * ∫ φ in -real.pi / 4..0, f₁ φ^2
  let region_2 := 1 / 2 * ∫ φ in 0..real.pi / 2, f₂ φ^2
  region_1 + region_2

theorem area_enclosed_by_polar_curves :
  enclosed_area = (real.pi / 4) - (1 / 4) :=
sorry

end area_enclosed_by_polar_curves_l68_68895


namespace complement_union_l68_68717

def U := {0, 1, 2, 3, 4}
def A := {0, 1, 3}
def B := {2, 3}

theorem complement_union (c_U : Set ℕ) (A B U : Set ℕ) (h1 : U = {0, 1, 2, 3, 4}) (h2 : A = {0, 1, 3}) (h3 : B = {2, 3}) (h4 : c_U = U \ (A ∪ B)) :
  c_U = {4} :=
by
  rw [h1, h2, h3] at h4
  sorry

end complement_union_l68_68717


namespace Tn_lt_Sn_div2_l68_68294

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68294


namespace fifty_three_days_from_friday_is_tuesday_l68_68476

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68476


namespace egg_prices_l68_68619

theorem egg_prices (x y z : ℕ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z)
  (h_sum : x + y + z = 100) (h_two_equal : x = y ∨ y = z ∨ x = z) 
  (h_cost : 0.5 * x + 0.6 * y + 0.9 * z = 60) :
  x = 60 ∧ y = 20 :=
by
  sorry

end egg_prices_l68_68619


namespace hex_lattice_has_12_equilateral_triangles_l68_68028

def Point := ℕ

def hexagonal_lattice (n : ℕ) : set Point :=
  {1} ∪ {2, 3, 4, 5, 6, 7} ∪ {8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}

def equilateral_triangle (a b c : Point) : Prop :=
  ((a, b) ∈ {(2, 6), (6, 7), (7, 3), (3, 4), (4, 1), (1, 2)} ∨
   (a, b) ∈ {(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8)}) ∧
  ((b, c) ∈ {(2, 6), (6, 7), (7, 3), (3, 4), (4, 1), (1, 2)} ∨
   (b, c) ∈ {(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8)}) ∧
  ((c, a) ∈ {(2, 6), (6, 7), (7, 3), (3, 4), (4, 1), (1, 2)} ∨
   (c, a) ∈ {(8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 8)})

theorem hex_lattice_has_12_equilateral_triangles :
  ∃ S : finset (Point × Point × Point), 
  (∀ t ∈ S, equilateral_triangle t.1 t.2 t.3) ∧ S.card = 12 :=
by
  sorry

end hex_lattice_has_12_equilateral_triangles_l68_68028


namespace shaded_rectangle_total_area_l68_68366

theorem shaded_rectangle_total_area (n : ℕ) (h : n ≥ 1):
    (∑ i in Finset.range n, (1 : ℝ) / (i + 1) - 1 / (i + 2)) + 1 / n = 1 := 
sorry

end shaded_rectangle_total_area_l68_68366


namespace deepak_age_l68_68844

variable (R D : ℕ)

theorem deepak_age (h1 : R / D = 4 / 3) (h2 : R + 6 = 26) : D = 15 :=
sorry

end deepak_age_l68_68844


namespace greatest_prime_factor_of_144_l68_68528

theorem greatest_prime_factor_of_144 : 
  ∃ p, p = 3 ∧ prime p ∧ ∀ q, prime q ∧ (q ∣ 144) → q ≤ p := 
by
  sorry

end greatest_prime_factor_of_144_l68_68528


namespace T_lt_S_div_2_l68_68340

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68340


namespace total_rehabilitation_centers_l68_68875

noncomputable def center_visits: ℕ := 6 -- Lisa's visits

def jude_visits (l: ℕ) : ℕ := l / 2
def han_visits (j: ℕ) : ℕ := 2 * j - 2
def jane_visits (h: ℕ) : ℕ := 6 + 2 * h

theorem total_rehabilitation_centers :
  let l := center_visits in
  let j := jude_visits l in
  let h := han_visits j in
  let n := jane_visits h in
  l + j + h + n = 27 :=
by
  sorry

end total_rehabilitation_centers_l68_68875


namespace shipping_cost_l68_68006

theorem shipping_cost (price_per_pineapple : ℚ) (number_of_pineapples : ℕ) (total_cost_per_pineapple : ℚ) 
  (cost_of_pineapples : ℚ) (shipping_cost : ℚ) : 
  price_per_pineapple = 1.25 → number_of_pineapples = 12 → total_cost_per_pineapple = 3 → 
  cost_of_pineapples = price_per_pineapple * number_of_pineapples → 
  shipping_cost = total_cost_per_pineapple * number_of_pineapples - cost_of_pineapples → 
  shipping_cost = 21 := 
by 
  intro h1 h2 h3 h4 h5
  rw [h1, h2, h3] at *
  simp at *
  exact h5.symm 

end shipping_cost_l68_68006


namespace Tn_lt_Sn_div_2_l68_68315

def a₁ := 1
def q := (1 : ℝ) / 3
def a (n : ℕ) : ℝ := q^(n-1)

def b (n : ℕ) : ℝ := (n : ℝ) * a n / 3

def S (n : ℕ) : ℝ := (∑ i in Finset.range n, a (i + 1))

def T (n : ℕ) : ℝ := (∑ i in Finset.range n, b (i + 1))

theorem Tn_lt_Sn_div_2 (n : ℕ) : 
  T n < S n / 2 := 
sorry

end Tn_lt_Sn_div_2_l68_68315


namespace tan_eq_sin3x_solutions_l68_68655

open Real

theorem tan_eq_sin3x_solutions : 
  ∃ (s : Finset ℝ), (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ tan x = sin (3 * x)) ∧ s.card = 6 :=
sorry

end tan_eq_sin3x_solutions_l68_68655


namespace solve_for_number_l68_68644

theorem solve_for_number :
  let n := 54 in
  fifteen_percent_less_than_80 = one_fourth_more_than_n n :=
by
  /-
    Definitions based on conditions:
    - Fifteen percent less than 80
    - One-fourth more than the number n 
  -/
  let fifteen_percent_less_than_80 := 68
  let one_fourth_more_than_n (x : ℝ) := (5/4) * x
  
  show fifteen_percent_less_than_80 = one_fourth_more_than_n n from
  sorry -- Proof not required per problem statement

end solve_for_number_l68_68644


namespace total_surface_area_is_correct_l68_68060

-- Define the problem constants and structure
def num_cubes := 20
def edge_length := 1
def bottom_layer := 9
def middle_layer := 8
def top_layer := 3
def total_painted_area : ℕ := 55

-- Define a function to calculate the exposed surface area
noncomputable def calc_exposed_area (num_bottom : ℕ) (num_middle : ℕ) (num_top : ℕ) (edge : ℕ) : ℕ := 
    let bottom_exposed := num_bottom * (edge * edge)
    let middle_corners_exposed := 4 * 3 * edge
    let middle_edges_exposed := (num_middle - 4) * (2 * edge)
    let top_exposed := num_top * (5 * edge)
    bottom_exposed + middle_corners_exposed + middle_edges_exposed + top_exposed

-- Statement to prove the total painted area
theorem total_surface_area_is_correct : calc_exposed_area bottom_layer middle_layer top_layer edge_length = total_painted_area :=
by
  -- The proof itself is omitted, focus is on the statement.
  sorry

end total_surface_area_is_correct_l68_68060


namespace Tn_lt_Sn_div2_l68_68300

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68300


namespace intersection_complement_A_B_l68_68665

def A : set ℝ := {x | 2^(2*x - 1) ≤ 1 / 4}
def B : set ℝ := {x | real.log x / real.log (1/16) ≥ 1 / 2}
def R : set ℝ := set.univ
def compl_R_A : set ℝ := R \ A

theorem intersection_complement_A_B :
  (compl_R_A ∩ B) = { x : ℝ | 0 < x ∧ x ≤ 1 / 4} :=
begin
  sorry
end

end intersection_complement_A_B_l68_68665


namespace total_sum_of_ages_l68_68411

theorem total_sum_of_ages (Y : ℕ) (interval : ℕ) (age1 age2 age3 age4 age5 : ℕ)
  (h1 : Y = 2) 
  (h2 : interval = 8) 
  (h3 : age1 = Y) 
  (h4 : age2 = Y + interval) 
  (h5 : age3 = Y + 2 * interval) 
  (h6 : age4 = Y + 3 * interval) 
  (h7 : age5 = Y + 4 * interval) : 
  age1 + age2 + age3 + age4 + age5 = 90 := 
by
  sorry

end total_sum_of_ages_l68_68411


namespace f_one_zero_range_of_a_l68_68164

variable (f : ℝ → ℝ) (a : ℝ)

-- Conditions
def odd_function : Prop := ∀ x : ℝ, x ≠ 0 → f (-x) = -f x
def increasing_on_pos : Prop := ∀ x y : ℝ, 0 < x → x < y → f x < f y
def f_neg_one_zero : Prop := f (-1) = 0
def f_a_minus_half_neg : Prop := f (a - 1/2) < 0

-- Questions
theorem f_one_zero (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) : f 1 = 0 := 
sorry

theorem range_of_a (h1 : odd_function f) (h2 : increasing_on_pos f) (h3 : f_neg_one_zero f) (h4 : f_a_minus_half_neg f a) :
  1/2 < a ∧ a < 3/2 ∨ a < -1/2 :=
sorry

end f_one_zero_range_of_a_l68_68164


namespace artifact_discovery_year_possibilities_l68_68568

theorem artifact_discovery_year_possibilities :
  let digits := [1, 1, 1, 5, 8, 9]
  let starting_digit := 8
  (number_of_possibilities starting_digit digits) = 20 :=
sorry

end artifact_discovery_year_possibilities_l68_68568


namespace tn_lt_sn_div_2_l68_68304

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68304


namespace Haley_weight_l68_68864

variables (V H S : ℝ)

theorem Haley_weight :
  (V = H + 17) ∧ 
  (V = S / 2) ∧ 
  (V + S = 360) → 
  H = 103 :=
by 
  intro h 
  cases h with h1 h'
  cases h' with h2 h3
  sorry

end Haley_weight_l68_68864


namespace product_same_value_for_j_l68_68769

theorem product_same_value_for_j (a b : ℕ → ℝ) (n : ℕ) 
  (h_distinct : ∀ i j, i ≠ j → a i ≠ a j) 
  (h_same_value : ∀ i, (∏ k in finset.range n, (a i + b k)) = c) : 
  ∀ j, (∏ i in finset.range n, (a i + b j)) = c := 
by 
  sorry

end product_same_value_for_j_l68_68769


namespace sacks_per_day_l68_68723

theorem sacks_per_day (total_sacks : ℕ) (total_days : ℕ) (harvest_per_day : ℕ) : 
  total_sacks = 56 → 
  total_days = 14 → 
  harvest_per_day = total_sacks / total_days → 
  harvest_per_day = 4 := 
by
  intros h_total_sacks h_total_days h_harvest_per_day
  rw [h_total_sacks, h_total_days] at h_harvest_per_day
  simp at h_harvest_per_day
  exact h_harvest_per_day

end sacks_per_day_l68_68723


namespace find_a_b_find_difference_l68_68390

variable (f : ℝ → ℝ)
variable (a b c : ℝ)

-- The function definition
def func (x : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x + c

-- Condition 1: extremum at x = 2
def extremum_at_2 : Prop := (deriv (func a b c) 2) = 0

-- Condition 2: parallel tangent at x = 1
def parallel_tangent_at_1 : Prop := (deriv (func a b c) 1) = -3

-- Question (1): Find the values of a and b
theorem find_a_b : 
  (extremum_at_2 a b c) ∧ (parallel_tangent_at_1 a b c) → a = -1 ∧ b = 0 := 
by 
  sorry

-- Condition 3: use the values of a and b found above
variable (a_found : a = -1)
variable (b_found : b = 0)

-- Question (2): Find the difference between maximum and minimum values
theorem find_difference (c : ℝ) : 
  extremum_at_2 a b c → parallel_tangent_at_1 a b c → (a = -1) → (b = 0) → 
  let f := func a b c in
  (f 0) - (f 2) = 4 := 
by 
  sorry

end find_a_b_find_difference_l68_68390


namespace max_sum_at_11_l68_68231

noncomputable def is_arithmetic_seq (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_seq (a : ℕ → ℚ) (n : ℕ) : ℚ :=
  (n + 1) * a 0 + (n * (n + 1) / 2) * (a 1 - a 0)

theorem max_sum_at_11 (a : ℕ → ℚ) (d : ℚ) (h_arith : is_arithmetic_seq a) (h_a1_gt_0 : a 0 > 0)
 (h_sum_eq : sum_seq a 13 = sum_seq a 7) : 
  ∃ n : ℕ, sum_seq a n = sum_seq a 10 + (a 10 + a 11) := sorry


end max_sum_at_11_l68_68231


namespace john_multiple_is_correct_l68_68276

noncomputable def compute_multiple (cost_per_computer : ℝ) 
                                   (num_computers : ℕ)
                                   (rent : ℝ)
                                   (non_rent_expenses : ℝ)
                                   (profit : ℝ) : ℝ :=
  let total_revenue := (num_computers : ℝ) * cost_per_computer
  let total_expenses := (num_computers : ℝ) * 800 + rent + non_rent_expenses
  let x := (total_expenses + profit) / total_revenue
  x

theorem john_multiple_is_correct :
  compute_multiple 800 60 5000 3000 11200 = 1.4 := by
  sorry

end john_multiple_is_correct_l68_68276


namespace distance_from_P_to_neg5_0_l68_68829

def hyperbola (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

def distance (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2)

theorem distance_from_P_to_neg5_0 {P : Point} (hP : hyperbola P.x P.y) (h_dist_to_5_0 : distance P ⟨5, 0⟩ = 15) :
  distance P ⟨-5, 0⟩ = 7 ∨ distance P ⟨-5, 0⟩ = 23 :=
sorry

end distance_from_P_to_neg5_0_l68_68829


namespace last_number_is_41_l68_68069

open Nat

-- Define the procedure of marking out numbers
def mark_out_numbers (numbers : List ℕ) (start_skip : ℕ) : ℕ :=
  let rec helper (remaining : List ℕ) (skip : ℕ) : List ℕ :=
    match remaining with
    | [] => []
    | _ => 
      let marked := (remaining.drop (skip - 1)).headI
      let new_list := remaining.filter (λ x => x != marked)
      helper new_list (skip + 1)
  (helper numbers start_skip).headI

-- Theorem to prove that given the conditions, the last remaining number is 41
theorem last_number_is_41 : mark_out_numbers (List.range 100).tail 4 = 41 := by
  sorry

end last_number_is_41_l68_68069


namespace sum_reciprocals_of_factors_of_28_l68_68585

theorem sum_reciprocals_of_factors_of_28 :
  (1:ℝ) + 1/2 + 1/4 + 1/7 + 1/14 + 1/28 = 2 :=
begin
  sorry
end

end sum_reciprocals_of_factors_of_28_l68_68585


namespace comp_functions_l68_68180

def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 5

theorem comp_functions (x : ℝ) : f (g x) = 6 * x - 7 :=
by
  sorry

end comp_functions_l68_68180


namespace triangle_area_proof_l68_68401

-- Conditions
variables (P r : ℝ) (semi_perimeter : ℝ)
-- The perimeter of the triangle is 40 cm
def perimeter_condition : Prop := P = 40
-- The inradius of the triangle is 2.5 cm
def inradius_condition : Prop := r = 2.5
-- The semi-perimeter is half of the perimeter
def semi_perimeter_def : Prop := semi_perimeter = P / 2

-- The area of the triangle
def area_of_triangle : ℝ := r * semi_perimeter

-- Proof Problem
theorem triangle_area_proof (hP : perimeter_condition P) (hr : inradius_condition r) (hsemi : semi_perimeter_def P semi_perimeter) :
  area_of_triangle r semi_perimeter = 50 :=
  sorry

end triangle_area_proof_l68_68401


namespace simplify_exponent_multiplication_l68_68067

theorem simplify_exponent_multiplication :
  (10 ^ 0.25) * (10 ^ 0.15) * (10 ^ 0.35) * (10 ^ 0.05) * (10 ^ 0.85) * (10 ^ 0.35) = 10 ^ 2 := by
  sorry

end simplify_exponent_multiplication_l68_68067


namespace weight_of_tin_of_cookies_l68_68902

def weight_of_bag_of_chips := 20 -- in ounces
def weight_jasmine_carries := 336 -- converting 21 pounds to ounces
def bags_jasmine_buys := 6
def tins_multiplier := 4

theorem weight_of_tin_of_cookies 
  (weight_of_bag_of_chips : ℕ := weight_of_bag_of_chips)
  (weight_jasmine_carries : ℕ := weight_jasmine_carries)
  (bags_jasmine_buys : ℕ := bags_jasmine_buys)
  (tins_multiplier : ℕ := tins_multiplier) : 
  ℕ :=
  let total_weight_bags := bags_jasmine_buys * weight_of_bag_of_chips
  let total_weight_cookies := weight_jasmine_carries - total_weight_bags
  let num_of_tins := bags_jasmine_buys * tins_multiplier
  total_weight_cookies / num_of_tins

example : weight_of_tin_of_cookies weight_of_bag_of_chips weight_jasmine_carries bags_jasmine_buys tins_multiplier = 9 :=
by sorry

end weight_of_tin_of_cookies_l68_68902


namespace largest_equal_cost_l68_68862

def digit_sum (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def binary_digit_sum (n : ℕ) : ℕ :=
  n.digits 2 |>.sum

theorem largest_equal_cost :
  ∃ (n : ℕ), n < 500 ∧ digit_sum n = binary_digit_sum n ∧ ∀ m < 500, digit_sum m = binary_digit_sum m → m ≤ 247 :=
by
  sorry

end largest_equal_cost_l68_68862


namespace angle_BAC_eq_arcsin_4_5_l68_68582

theorem angle_BAC_eq_arcsin_4_5 :
  ∀ (A B C O : Point) (OA OB OC : ℝ) (h : OA = 15 ∧ OB = 12 ∧ OC = 20)
  (equilateral_foot_perpendiculars : is_equilateral (foot_perpendiculars O A B C)),
  angle A B C = Real.arcsin (4 / 5) :=
by
  sorry

end angle_BAC_eq_arcsin_4_5_l68_68582


namespace count_four_digit_numbers_div_by_3_with_45_last_digits_l68_68725

theorem count_four_digit_numbers_div_by_3_with_45_last_digits :
  (∃ (count : ℕ), count = 30 ∧ 
  ∀ (num : ℕ), 1000 ≤ num ∧ num < 10000 ∧ num % 100 = 45 ∧ num % 3 = 0 → count) :=
sorry

end count_four_digit_numbers_div_by_3_with_45_last_digits_l68_68725


namespace days_from_friday_l68_68500

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68500


namespace sum_of_possible_k_l68_68248

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l68_68248


namespace max_divisor_of_f_l68_68215

theorem max_divisor_of_f (n : ℕ) (hn : 0 < n) : ∃ m, (∀ n > 0, ((2 * n - 7) * 3 ^ n + 9) % m = 0) ∧ (∀ k, (∀ n > 0, ((2 * n - 7) * 3 ^ n + 9) % k = 0) → k ≤ m) :=
begin
  use 6,
  split,
  { intro n,
    intro hn,
    sorry, -- proof omitted
  },
  { intros k hk,
    sorry, -- proof omitted
  }
end

end max_divisor_of_f_l68_68215


namespace fuel_efficiency_problem_l68_68563

theorem fuel_efficiency_problem :
  let F_highway := 30
  let F_urban := 25
  let F_hill := 20
  let D_highway := 100
  let D_urban := 60
  let D_hill := 40
  let gallons_highway := D_highway / F_highway
  let gallons_urban := D_urban / F_urban
  let gallons_hill := D_hill / F_hill
  let total_gallons := gallons_highway + gallons_urban + gallons_hill
  total_gallons = 7.73 := 
by 
  sorry

end fuel_efficiency_problem_l68_68563


namespace geometric_sequence_formula_l68_68773

theorem geometric_sequence_formula (a : ℕ → ℝ)
  (q : ℝ)
  (h1 : a 1 = 3 / 2)
  (h2 : a 1 + a 1 * q + a 1 * q^2 = 9 / 2)
  (geo : ∀ n, a (n + 1) = a n * q) :
  ∀ n, a n = 3 / 2 * (-2)^(n-1) ∨ a n = 3 / 2 :=
by sorry

end geometric_sequence_formula_l68_68773


namespace complex_number_identity_l68_68734

-- Definition of the imaginary unit
def i : ℂ := complex.I

-- Statement about the multiplication of complex numbers involving the imaginary unit
theorem complex_number_identity : i * (2 - i) = -1 + 2 * i := 
by
  -- proof goes here
  sorry

end complex_number_identity_l68_68734


namespace sequence_general_formula_sequence_sum_formula_l68_68141

open Nat

theorem sequence_general_formula (a : ℕ+ → ℤ) (n : ℕ+) :
  (a n) ^ 2 + 2 * (a n) - n ^ 2 + 2 * n = 0 → a n = n := by
  sorry

theorem sequence_sum_formula (a : ℕ+ → ℤ) (n : ℕ+) :
  (∀ m : ℕ+, a m = m) → 
  (∑ i in Finset.range n, a (i.succ)) = (n * (n + 1)) / 2 := by
  sorry

end sequence_general_formula_sequence_sum_formula_l68_68141


namespace at_least_two_equal_l68_68765

theorem at_least_two_equal (x y z : ℝ) (h : (x - y) / (2 + x * y) + (y - z) / (2 + y * z) + (z - x) / (2 + z * x) = 0) : 
x = y ∨ y = z ∨ z = x := 
by
  sorry

end at_least_two_equal_l68_68765


namespace area_of_shaded_part_l68_68237

open Real

-- Given conditions
def side_length_of_octagon : ℝ := 60
def equilateral_octagon (n : ℕ) : Prop := (n = 8 ∧ ∀ (i : ℕ), 0 ≤ i < n → ∃ s : ℝ, s = side_length_of_octagon)

-- Defining the area of the shaded part as the main goal
theorem area_of_shaded_part :
  equilateral_octagon 8 →
  ∃ A B C, metric_space.dist A B = 60 ∧ metric_space.dist B C = 60 ∧ metric_space.dist C A = 60 ∧ 
  let area := (1 / 2) * 60 * 30 in area = 900 :=
sorry

end area_of_shaded_part_l68_68237


namespace T_lt_half_S_l68_68332

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68332


namespace circumscribe_hexagon_with_properties_around_convex_curve_l68_68551

theorem circumscribe_hexagon_with_properties_around_convex_curve (K : Type) [convex_curve K] :
  ∃ H : hexagon, circumscribed_by K H ∧ ∀ (i : ℕ), H.angle i = π / 3 ∧ 
  (H.side i = H.side (i + 3) % 6) :=
sorry

end circumscribe_hexagon_with_properties_around_convex_curve_l68_68551


namespace allen_blocks_l68_68051

def blocks_per_color : Nat := 7
def colors_used : Nat := 7

theorem allen_blocks : (blocks_per_color * colors_used) = 49 :=
by
  sorry

end allen_blocks_l68_68051


namespace line_through_points_zero_four_four_m_ten_negative_two_l68_68580

theorem line_through_points_zero_four_four_m_ten_negative_two (m : ℚ) :
  let slope1 := (m - 4) / 4 in
  let slope2 := (-2 - m) / 6 in
  slope1 = slope2 -> m = 8 / 5 :=
by
  intro h
  -- proof would go here
  sorry

end line_through_points_zero_four_four_m_ten_negative_two_l68_68580


namespace incorrect_propositions_l68_68606

-- Definitions based on conditions
def prop1 (P Q : Prop) : Prop := (¬ P → ¬ Q) → (Q → P)
def prop2 (a b : ℝ) : Prop := ¬((a + b ≠ 6) → (a ≠ 3 ∨ b ≠ 3))
def sufficient_condition (x : ℝ) : Prop := (x > 2) → (1 / x < 1 / 2)
def necessary_condition (x : ℝ) : Prop := (1 / x < 1 / 2) → (x > 2)
def prop4 (P Q : Prop) : Prop := (¬ P → Q) = (¬ Q → ¬ P)

-- The main theorem: proving that propositions 1 and 2 are incorrect
theorem incorrect_propositions : (prop1 P Q = false) ∧ (prop2 a b = false) := 
by
  sorry

end incorrect_propositions_l68_68606


namespace who_answered_2_questions_correctly_l68_68119

-- Define types for friends and scores.
inductive Friend : Type
  | A 
  | B 
  | C 
  | D

def Score : Type := ℕ

-- Definition of the four friends and their scores.
variables (sA sB sC sD : Score)
variable h : sA ≠ sB ∧ sA ≠ sC ∧ sA ≠ sD ∧ sB ≠ sC ∧ sB ≠ sD ∧ sC ≠ sD 
variable hs : sA + sB + sC + sD = 6
variable hs_nonzero : 0 ≤ sA ∧ sA ≤ 3 ∧ 0 ≤ sB ∧ sB ≤ 3 ∧ 0 ≤ sC ∧ sC ≤ 3 ∧ 0 ≤ sD ∧ sD ≤ 3

-- Statements made by the friends.
def friendA_statements := (sA = 2) ∧ (sA > sB) ∧ (sC < sD)
def friendB_statements := (sB = 3) ∧ (sC = 0) ∧ (sB > sA)
def friendC_statements := (sC = 1) ∧ (sD = 2) ∧ (sB < sA)
def friendD_statements := (sD = 3) ∧ (sC < sD) ∧ (sA < sB)

-- Truth condition for each friend's statements based on their score.
def truth_condition (score : Score) (statements : Prop) : Prop :=
  (if score = 3 then 3 else if score = 2 then 2 else if score = 1 then 1 else 0) = count_truths statements

def count_truths (statements : Prop) : ℕ := sorry

-- The main proposition to prove.
theorem who_answered_2_questions_correctly :
  (truth_condition sA friendA_statements) →
  (truth_condition sB friendB_statements) →
  (truth_condition sC friendC_statements) →
  (truth_condition sD friendD_statements) →
  sA = 2 :=
by
  sorry

end who_answered_2_questions_correctly_l68_68119


namespace least_positive_t_geometric_progression_l68_68628

noncomputable def least_positive_t( α : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) : ℝ :=
  9 - 4 * Real.sqrt 5

theorem least_positive_t_geometric_progression ( α t : ℝ ) ( h : 0 < α ∧ α < Real.pi / 2 ) :
  least_positive_t α h = t ↔
  ∃ r : ℝ, r > 0 ∧
    Real.arcsin (Real.sin α) = α ∧
    Real.arcsin (Real.sin (2 * α)) = 2 * α ∧
    Real.arcsin (Real.sin (7 * α)) = 7 * α ∧
    Real.arcsin (Real.sin (t * α)) = t * α ∧
    (α * r = 2 * α) ∧
    (2 * α * r = 7 * α ) ∧
    (7 * α * r = t * α) :=
sorry

end least_positive_t_geometric_progression_l68_68628


namespace cleaning_time_together_l68_68791

/-
  Problem Statement:
  Lisa and Kay were asked by their mother to clean their room.
  Lisa can clean her room in 8 hours, and Kay in 12 hours.
  Prove that if they work together, it will take them 4.8 hours to clean the entire room.
-/

-- Given the rates at which Lisa and Kay can clean their room
def lisa_rate := 1 / 8
def kay_rate := 1 / 12

-- Combined rate of Lisa and Kay working together
def combined_rate := lisa_rate + kay_rate

-- Time for both to clean the room together
def combined_time := 1 / combined_rate

theorem cleaning_time_together :
  combined_time = 4.8 :=
by
  -- Definitions of rates
  let lisa_rate := 1 / 8
  let kay_rate := 1 / 12
  let combined_rate := lisa_rate + kay_rate

  -- Calculation of combined rate
  have h1 : combined_rate = (1 / 8) + (1 / 12) := rfl
  rw h1

  -- Convert to common denominator and sum
  have h2 : (1 / 8) = 3 / 24 := by norm_num [div_eq_div_iff]
  have h3 : (1 / 12) = 2 / 24 := by norm_num [div_eq_div_iff]
  rw [h2, h3]

  have h4 : (3 / 24) + (2 / 24) = 5 / 24 := by norm_num
  rw h4

  -- Calculate time to clean together
  have h5 : 1 / (5 / 24) = 24 / 5 := by norm_num [div_eq_mul_inv]
  rw h5

  -- Simplify to find the answer
  have h6 : 24 / 5 = 4.8 := by norm_num
  rw h6
  refl

end cleaning_time_together_l68_68791


namespace num_birds_is_six_l68_68420

-- Define the number of nests
def N : ℕ := 3

-- Define the difference between the number of birds and nests
def diff : ℕ := 3

-- Prove that the number of birds is 6
theorem num_birds_is_six (B : ℕ) (h1 : N = 3) (h2 : B - N = diff) : B = 6 := by
  -- Placeholder for the proof
  sorry

end num_birds_is_six_l68_68420


namespace radius_of_inscribed_circle_l68_68871

theorem radius_of_inscribed_circle (d1 d2: ℝ) (h1: d1 = 8) (h2: d2 = 30) : 
  let r := 30 / Real.sqrt 241 in
  r = 30 / Real.sqrt 241 :=
by
  sorry

end radius_of_inscribed_circle_l68_68871


namespace triangle_inequality_circumradius_l68_68160

theorem triangle_inequality_circumradius (a b c R : ℝ) (h_triangle : a + b > c ∧ a + c > b ∧ b + c > a) :
  \(\frac{1}{a b} + \frac{1}{b c} + \frac{1}{c a} \geq \frac{1}{R^{2}}\) :=
sorry

end triangle_inequality_circumradius_l68_68160


namespace interview_problem_l68_68227

noncomputable def judge_scores : List ℝ := [70, 85, 86, 88, 90, 90, 92, 94, 95, 100]

theorem interview_problem :
  let sorted_scores := List.sort judge_scores
  let n := sorted_scores.length
  let median := (sorted_scores.get! (n/2 - 1) + sorted_scores.get! (n/2)) / 2
  let mean := sorted_scores.sum / n
  let removed_scores := List.filter (λ x, x ≠ List.minimum judge_scores ∧ x ≠ List.maximum judge_scores) judge_scores
  let new_mean := removed_scores.sum / removed_scores.length
  let variance (l : List ℝ) : ℝ := (l.map (λ x, (x - mean) ^ 2)).sum / l.length
  let new_variance := variance removed_scores
  Prop := (
    (∃ p : ℝ, p = (2 / (judge_scores.length * (judge_scores.length - 1))) ∧ p ≠ 1 / judge_scores.length) ∧
    (sorted_scores.get! (6 - 1) + sorted_scores.get! 6) / 2 = 91 ∧
    mean < median ∧
    new_mean > mean ∧
    new_variance < variance judge_scores
  )
:= sorry

end interview_problem_l68_68227


namespace quadratic_equation_D_l68_68004

theorem quadratic_equation_D (k : ℝ) : 
  ∀ x : ℝ, (k^2 + 1) * x^2 - 2 * x + 1 = 0 → 
    ∃ a b c : ℝ, a ≠ 0 ∧ (k^2 + 1) = a ∧ (-2) = b ∧ 1 = c :=
begin
  sorry,
end

end quadratic_equation_D_l68_68004


namespace greatest_prime_factor_of_144_l68_68525

-- Define the number 144
def num : ℕ := 144

-- Define what it means for a number to be a prime factor of num
def is_prime_factor (p n : ℕ) : Prop :=
  Prime p ∧ p ∣ n

-- Define what it means to be the greatest prime factor
def greatest_prime_factor (p n : ℕ) : Prop :=
  is_prime_factor p n ∧ (∀ q, is_prime_factor q n → q ≤ p)

-- Prove that the greatest prime factor of 144 is 3
theorem greatest_prime_factor_of_144 : greatest_prime_factor 3 num :=
sorry

end greatest_prime_factor_of_144_l68_68525


namespace logarithm_monotonically_increasing_l68_68128

theorem logarithm_monotonically_increasing {a : ℝ} (h : a ≥ 1) :
  ∀ x y : ℝ, (1 < x ∧ 1 < y ∧ x < y) → log 2 (a * x^2 - x) < log 2 (a * y^2 - y) :=
begin
  sorry

end logarithm_monotonically_increasing_l68_68128


namespace convex_polygon_not_divisible_into_non_convex_quadrilaterals_l68_68618

theorem convex_polygon_not_divisible_into_non_convex_quadrilaterals (p : ℕ) (polygon : polygon) 
  (Hconvex : ∀ (vertex : polygon.vertices), interior_angle polygon vertex < 180°) :
  ¬ ∃ (quads : list quadrilateral), (∀ q ∈ quads, non_convex q) ∧ (polygon_divided_into_congruent_quads p quads) :=
by
  sorry

end convex_polygon_not_divisible_into_non_convex_quadrilaterals_l68_68618


namespace Binet_formula_l68_68809

noncomputable def φ : ℝ := (1 + Real.sqrt 5) / 2
noncomputable def ψ : ℝ := (1 - Real.sqrt 5) / 2

def Fibonacci (n : ℕ) : ℕ 
| 0       => 0
| 1       => 1
| (n + 2) => Fibonacci n + Fibonacci (n + 1)

theorem Binet_formula (n : ℕ) : (Fibonacci n : ℝ) = (φ^n - ψ^n) / Real.sqrt 5 :=
by
  sorry

end Binet_formula_l68_68809


namespace boss_spends_7600_per_month_l68_68223

def hoursPerWeekFiona : ℕ := 40
def hoursPerWeekJohn : ℕ := 30
def hoursPerWeekJeremy : ℕ := 25
def hourlyRate : ℕ := 20
def weeksPerMonth : ℕ := 4

def weeklyEarnings (hours : ℕ) (rate : ℕ) : ℕ := hours * rate
def monthlyEarnings (weekly : ℕ) (weeks : ℕ) : ℕ := weekly * weeks

def totalMonthlyExpenditure : ℕ :=
  monthlyEarnings (weeklyEarnings hoursPerWeekFiona hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJohn hourlyRate) weeksPerMonth +
  monthlyEarnings (weeklyEarnings hoursPerWeekJeremy hourlyRate) weeksPerMonth

theorem boss_spends_7600_per_month :
  totalMonthlyExpenditure = 7600 :=
by
  sorry

end boss_spends_7600_per_month_l68_68223


namespace S_equals_x4_l68_68289

-- Define the expression for S
def S (x : ℝ) : ℝ := (x - 1)^4 + 4 * (x - 1)^3 + 6 * (x - 1)^2 + 4 * x - 3

-- State the theorem to be proved
theorem S_equals_x4 (x : ℝ) : S x = x^4 :=
by
  sorry

end S_equals_x4_l68_68289


namespace P_equidistant_from_A_B_satisfies_condition_l68_68648

noncomputable def point_equidistant_condition (P : ℝ × ℝ × ℝ) : Prop :=
  let A := (2, 3, 0)
  let B := (5, 1, 0)
  let dist_A := (P.1 - A.1)^2 + (P.2 - A.2)^2 + (P.3 - A.3)^2
  let dist_B := (P.1 - B.1)^2 + (P.2 - B.2)^2 + (P.3 - B.3)^2
  dist_A = dist_B → 6 * P.1 - 4 * P.2 - 13 = 0

theorem P_equidistant_from_A_B_satisfies_condition :
  ∀ (P : ℝ × ℝ × ℝ), point_equidistant_condition P :=
by
  sorry

end P_equidistant_from_A_B_satisfies_condition_l68_68648


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68456

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68456


namespace steve_has_7_fewer_b_berries_l68_68821

-- Define the initial number of berries Stacy has
def stacy_initial_berries : ℕ := 32

-- Define the number of berries Steve takes from Stacy
def steve_takes : ℕ := 4

-- Define the initial number of berries Steve has
def steve_initial_berries : ℕ := 21

-- Using the given conditions, prove that Steve has 7 fewer berries compared to Stacy's initial amount
theorem steve_has_7_fewer_b_berries :
  stacy_initial_berries - (steve_initial_berries + steve_takes) = 7 := 
by
  sorry

end steve_has_7_fewer_b_berries_l68_68821


namespace no_closed_path_diagonals_l68_68120

theorem no_closed_path_diagonals (cubes : Set (ℕ × ℕ × ℕ)) :
  (∀ (x y z : ℕ), 1 ≤ x ∧ x ≤ 3 ∧ 1 ≤ y ∧ y ≤ 3 ∧ 1 ≤ z ∧ z ≤ 3 ∧ (x, y, z) ≠ (2, 2, 2)) →
  ¬(∃ (path : List (ℕ × ℕ × ℕ)), (∀ v ∈ path, v ∈ cubes) ∧ path.head = path.last ∧ (∀ i < path.length - 1, (path.nth i).is_diagonal (path.nth (i + 1))) ∧ (has_no_repeated_vertices path)) :=
by
  sorry

end no_closed_path_diagonals_l68_68120


namespace convex_polygon_not_divisible_l68_68615

-- Defining our function f
def f (N : Polygon) : ℝ :=
  let angles := N.interiorAngles
  let lessThan180 := angles.filter (λ α, α < 180)
  let greaterThan180Supp := angles.filter (λ α, α > 180) |>.map (λ α => 360 - α)
  list.sum lessThan180 - list.sum greaterThan180Supp

-- Defining our problem statement
theorem convex_polygon_not_divisible (M : Polygon) (h₁ : M.isConvex) :
  ¬ ∃ (M₁ M₂ ... Mn : Polygon), (∀ i, M₁.isNonConvex) ∧ (M = M₁ ∪ M₂ ∪ ... ∪ Mn) :=
begin
  sorry
end

end convex_polygon_not_divisible_l68_68615


namespace correct_propositions_l68_68707

-- Define the propositions
def prop1 := ∀ (l₁ l₂ : Line), ¬ ∃ p : Point, lies_on p l₁ ∧ lies_on p l₂ → parallel l₁ l₂
def prop2 := ∀ (l₁ l₂ : Line), perpendicular l₁ l₂ → ∃ p : Point, lies_on p l₁ ∧ lies_on p l₂
def prop3 := ∀ (l₁ l₂ : Line), ¬parallel l₁ l₂ ∧ ¬ ∃ p : Point, lies_on p l₁ ∧ lies_on p l₂ → skew l₁ l₂
def prop4 := ∀ (l₁ l₂ : Line), ¬same_plane l₁ l₂ → skew l₁ l₂

-- The main theorem statement to prove correctness
theorem correct_propositions : ({ prop3, prop4 } : set (Prop)) = ({ prop1, prop2, prop3, prop4 }.filter (λ p, (p = prop3) ∨ (p = prop4))) :=
by {
 sorry -- proof can be provided
}

end correct_propositions_l68_68707


namespace intersection_points_product_equality_l68_68391

-- Define the hexagon and inscribed condition
variables (A B C D L K M N P Q : ℂ)
variables 
  (h1 : ∃ r : ℝ, ∀ x ∈ {A, B, C, D, L, K}, |x| = r)
  (h2 : M = (A * D * (K + L) - K * L * (A + D)) / (A * D - K * L))
  (h3 : N = (B * C * (K + L) - K * L * (B + C)) / (B * C - K * L))
  (h4 : P = (A * C * (K + L) - K * L * (A + C)) / (A * C - K * L))
  (h5 : Q = (B * D * (K + L) - K * L * (B + D)) / (B * D - K * L))

-- Goal: Prove the required equality
theorem intersection_points_product_equality : 
  |N - L| * |K - P| * |M - Q| = |K - M| * |P - N| * |L - Q| :=
sorry

end intersection_points_product_equality_l68_68391


namespace cos_alpha_through_point_l68_68213

theorem cos_alpha_through_point (a : ℝ) (ha : a < 0) : cos (atan2 (2 * a) a) = -(real.sqrt 5) / 5 :=
by sorry

end cos_alpha_through_point_l68_68213


namespace find_a_l68_68664

-- Points A and B on the x-axis
def point_A (a : ℝ) : (ℝ × ℝ) := (a, 0)
def point_B : (ℝ × ℝ) := (-3, 0)

-- Distance condition
def distance_condition (a : ℝ) : Prop := abs (a + 3) = 5

-- The proof problem: find a such that distance condition holds
theorem find_a (a : ℝ) : distance_condition a ↔ (a = -8 ∨ a = 2) :=
by
  sorry

end find_a_l68_68664


namespace rectangle_area_l68_68064

/--
Assume we have a rectangle with perimeter 32 cm and a length-to-width ratio of 5:3.
Prove that the area of the rectangle is 60 cm².
-/
theorem rectangle_area (perimeter length_ratio width_ratio : ℕ)
  (h1 : perimeter = 32)
  (h2 : length_ratio = 5)
  (h3 : width_ratio = 3)
  (h4 : length_ratio + width_ratio = 8) -- 5x + 3x = 8x, necessary condition to make this proof relevant
  : 5 * (h1 / h4) * 3 * (h1 / h4) = 60 := 
by 
  sorry

end rectangle_area_l68_68064


namespace total_students_is_40_l68_68225

variable (students_total : ℕ) -- Total number of students in the school
variables (students_below8 students_8 students_above8 : ℕ) -- Count of students in different age groups

-- Conditions as hypotheses
hypothesis h1 : students_below8 = (students_total * 1 / 5)
hypothesis h2 : students_8 = 12
hypothesis h3 : students_above8 = (2 / 3) * students_8

-- Main theorem statement
theorem total_students_is_40 (h1 : students_below8 = (students_total * 1 / 5))
                             (h2 : students_8 = 12)
                             (h3 : students_above8 = (2 / 3) * students_8) :
                             students_total = 40 :=
                             sorry

end total_students_is_40_l68_68225


namespace siblings_of_kevin_l68_68410

-- Define traits of each child
structure Child where
  eye_color : String
  hair_color : String

def Oliver : Child := ⟨"Green", "Red"⟩
def Kevin : Child := ⟨"Grey", "Brown"⟩
def Lily : Child := ⟨"Grey", "Red"⟩
def Emma : Child := ⟨"Green", "Brown"⟩
def Noah : Child := ⟨"Green", "Red"⟩
def Mia : Child := ⟨"Green", "Brown"⟩

-- Define the condition that siblings must share at least one trait
def share_at_least_one_trait (c1 c2 : Child) : Prop :=
  c1.eye_color = c2.eye_color ∨ c1.hair_color = c2.hair_color

-- Prove that Emma and Mia are Kevin's siblings
theorem siblings_of_kevin : share_at_least_one_trait Kevin Emma ∧ share_at_least_one_trait Kevin Mia ∧ share_at_least_one_trait Emma Mia :=
  sorry

end siblings_of_kevin_l68_68410


namespace octagon_side_length_l68_68236

theorem octagon_side_length (AB BC : ℝ) (AP BQ : ℝ) (k m n : ℤ) (x : ℝ)
  (h₁ : AB = 10) (h₂ : BC = 8) (h₃ : AP = BQ ∧ AP < 5) 
  (h₄ : x = -9 + Real.sqrt 163) :
  (let side_length_form := k + m * Real.sqrt n in
   side_length_form = x ∧ n = 163 ∧ m = 1 ∧ k = -9) →
  k + m + n = 155 :=
by
  sorry

end octagon_side_length_l68_68236


namespace correct_parallel_postulate_l68_68899

-- Definitions corresponding to the conditions
def lines_do_not_have_common_points_but_can_be_parallel_or_skew :=
  ∀ {L1 L2 : Type} [line L1] [line L2], 
  (¬ ∃ (P : Type), (P ∈ L1) ∧ (P ∈ L2)) → 
  (parallel L1 L2 ∨ skew L1 L2)

def lines_perpendicular_to_same_line_can_intersect_parallel_or_skew :=
  ∀ {L1 L2 L3 : Type} [line L1] [line L2] [line L3], 
  (perpendicular L1 L3 ∧ perpendicular L2 L3) → 
  (intersect L1 L2 ∨ parallel L1 L2 ∨ skew L1 L2)

def lines_parallel_to_same_line_are_parallel :=
  ∀ {L1 L2 L3 : Type} [line L1] [line L2] [line L3], 
  (parallel L1 L3 ∧ parallel L2 L3) → parallel L1 L2

def line_not_in_plane_can_be_parallel_or_intersect_plane :=
  ∀ {a α : Type} [line a] [plane α], 
  ¬ (a ⊆ α) → 
  (parallel_to_plane a α ∨ intersects_plane a α)

-- Problem statement to prove
theorem correct_parallel_postulate :
  lines_do_not_have_common_points_but_can_be_parallel_or_skew →
  lines_perpendicular_to_same_line_can_intersect_parallel_or_skew →
  lines_parallel_to_same_line_are_parallel →
  line_not_in_plane_can_be_parallel_or_intersect_plane →
  ∃! (C : Prop), 
  C = ∀ {L1 L2 L3 : Type} [line L1] [line L2] [line L3], 
  (parallel L1 L3 ∧ parallel L2 L3) → parallel L1 L2 
sorry -- Proof is omitted

end correct_parallel_postulate_l68_68899


namespace mollys_present_age_l68_68009

theorem mollys_present_age (x : ℤ) (h : x + 18 = 5 * (x - 6)) : x = 12 := by
  sorry

end mollys_present_age_l68_68009


namespace sum_of_possible_k_l68_68252

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l68_68252


namespace number_of_possible_n_l68_68841

theorem number_of_possible_n :
  ∃ (a : ℕ), (∀ n, (n = a^3) ∧ 
  ((∃ b c : ℕ, b ≠ c ∧ b ≠ a ∧ c ≠ a ∧ a = b * c)) ∧ 
  (a + b + c = 2010) ∧ 
  (a > 0) ∧
  (b > 0) ∧
  (c > 0)) → 
  ∃ (num_n : ℕ), num_n = 2009 :=
  sorry

end number_of_possible_n_l68_68841


namespace area_PST_l68_68763

variables (P Q R S T : Type) [Point P] [Point Q] [Point R] [Point S] [Point T]

/-- Given a triangle PQR with specified side lengths and points S and T on segments PQ and PR respectively, 
    prove the area of triangle PST is 225/17 --/
theorem area_PST 
  (PQ QR PR : ℝ)
  (PS PT : ℝ)
  (PQR_area : real) 
  (sin_Q : real)
  (hPQ : PQ = 8)
  (hQR : QR = 15)
  (hPR : PR = 17)
  (hPS : PS = 3)
  (hPT : PT = 10)
  (hPQR_area : PQR_area = 60)
  (hsin_Q : sin_Q = 15/17) :
  let PST_area := (1/2) * PS * PT * sin_Q in
  PST_area = 225 / 17 :=
by
  sorry

end area_PST_l68_68763


namespace parabola_tangent_to_line_determines_a_l68_68745

theorem parabola_tangent_to_line_determines_a (a : ℝ) :
  (∀ x : ℝ, ax^2 + 4 = 2x + 1) →
  a = 1/3 :=
sorry

end parabola_tangent_to_line_determines_a_l68_68745


namespace Karson_current_books_l68_68279

theorem Karson_current_books :
  ∀ (library_capacity current_books books_to_buy : ℕ),
    library_capacity = 400 →
    books_to_buy = 240 →
    (current_books + books_to_buy = 0.9 * library_capacity) →
    current_books = 120 := 
by
  intros library_capacity current_books books_to_buy H_capacity H_buy H_eq.
  have H_90perc : 0.9 * 400 = 360 := by norm_num,
  rw [H_capacity, H_90perc] at H_eq,
  linarith,
  sorry

end Karson_current_books_l68_68279


namespace time_to_cross_platform_l68_68578

def speed_kmph := 72 -- Speed of the train in kmph
def length_platform := 240 -- Length of the platform in meters
def length_train := 280.04 -- Length of the goods train in meters

def total_distance := length_train + length_platform

def speed_mps := speed_kmph * 1000 / 3600 -- Convert speed from kmph to mps

def time := total_distance / speed_mps

theorem time_to_cross_platform : time = 26.002 :=
by
  -- Placeholder for the proof, which we'll leave out as per the instructions
  sorry

end time_to_cross_platform_l68_68578


namespace constant_sum_l68_68082

-- Point and triangle are defined as structures or elements of a field
structure Point := (x y z : ℝ)
structure Triangle := (A B C : Point)

-- Define P as a point on the inscribed circle of equilateral triangle ABC.
-- We represent the conditions in terms of types and fields.
noncomputable def P_on_inscribed_circle (P A B C : Point) : Prop :=
  let d1 := (P.x - A.x)^2 + (P.y - A.y)^2 + (P.z - A.z)^2
  let d2 := (P.x - B.x)^2 + (P.y - B.y)^2 + (P.z - B.z)^2
  let d3 := (P.x - C.x)^2 + (P.y - C.y)^2 + (P.z - C.z)^2
  ((1 : ℝ) = (A.x + A.y + A.z + B.x + B.y + B.z + C.x + C.y + C.z)) ∧  -- Sum of coordinates constraint for equilateral triangle
  (∀ P, (A.x^2 + A.y^2 + A.z^2) = (B.x^2 + B.y^2 + B.z^2) = (C.x^2 + C.y^2 + C.z^2)) -- Points lie on the same sphere inscribed in the equilateral triangle
  
-- The theorem proving the quantity is constant.
theorem constant_sum (P A B C : Point) (h1 : P_on_inscribed_circle P A B C) : 
  (P_on_inscribed_circle P A B C) →  ((P.x - A.x)^2 + (P.y - A.y)^2 + (P.z - A.z)^2 + (P.x - B.x)^2 + (P.y - B.y)^2 + (P.z - B.z)^2 + (P.x - C.x)^2 + (P.y - C.y)^2 + (P.z - C.z)^2 = 3 * (A.x^2 + A.y^2 + A.z^2) - 2 * (A.x + A.y + A.z)) :=
by sorry

end constant_sum_l68_68082


namespace train_travel_distance_l68_68577

def speed (miles : ℕ) (minutes : ℕ) : ℕ :=
  miles / minutes

def minutes_in_hours (hours : ℕ) : ℕ :=
  hours * 60

def distance_traveled (rate : ℕ) (time : ℕ) : ℕ :=
  rate * time

theorem train_travel_distance :
  (speed 2 2 = 1) →
  (minutes_in_hours 3 = 180) →
  distance_traveled (speed 2 2) (minutes_in_hours 3) = 180 :=
by
  intros h_speed h_minutes
  rw [h_speed, h_minutes]
  sorry

end train_travel_distance_l68_68577


namespace four_digit_palindromic_perfect_square_l68_68948

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68948


namespace option_b_represents_factoring_l68_68056

theorem option_b_represents_factoring (x y : ℤ) :
  x^2 - 2*x*y = x * (x - 2*y) :=
sorry

end option_b_represents_factoring_l68_68056


namespace samantha_routes_l68_68813

theorem samantha_routes : 
  let house_to_sw = Nat.choose (3 + 2) 2 in
  let ne_to_school = Nat.choose (3 + 3) 3 in
  house_to_sw * 1 * 1 * ne_to_school = 200 := 
by
  let house_to_sw := Nat.choose (3 + 2) 2
  let ne_to_school := Nat.choose (3 + 3) 3
  have : house_to_sw * 1 * 1 * ne_to_school = 200 := by sorry
  exact this

end samantha_routes_l68_68813


namespace max_value_at_x0_l68_68704

noncomputable def f (x : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * Real.sin x + (1 / 2) * Real.cos x

theorem max_value_at_x0 : ∃ k : ℤ, x0 = (π / 3) + 2 * k * π ∧ ∀ x, f(x0) ≥ f(x) := by
    sorry

end max_value_at_x0_l68_68704


namespace sheila_hourly_wage_l68_68891

-- Sheila works 8 hours per day on Monday, Wednesday, and Friday
-- Sheila works 6 hours per day on Tuesday and Thursday
-- Sheila does not work on Saturday and Sunday
-- Sheila earns $288 per week

def hours_worked (monday_wednesday_friday_hours : Nat) (tuesday_thursday_hours : Nat) : Nat :=
  (monday_wednesday_friday_hours * 3) + (tuesday_thursday_hours * 2)

def weekly_earnings : Nat := 288
def total_hours_worked : Nat := hours_worked 8 6
def hourly_wage : Nat := weekly_earnings / total_hours_worked

theorem sheila_hourly_wage : hourly_wage = 8 := by
  -- Proof (omitted)
  sorry

end sheila_hourly_wage_l68_68891


namespace sum_of_possible_k_l68_68249

noncomputable def find_sum_k : Nat :=
  let S := { k | ∃ j : Nat, k > 0 ∧ j > 0 ∧ (1 / (j : ℚ) + 1 / (k : ℚ) = 1 / 5) }
  S.to_finset.sum id

theorem sum_of_possible_k : find_sum_k = 46 :=
by
  sorry

end sum_of_possible_k_l68_68249


namespace sum_of_four_digit_numbers_l68_68873

open Nat

theorem sum_of_four_digit_numbers (s : Finset ℤ) :
  (∀ x, x ∈ s → (∃ k, x = 30 * k + 2) ∧ 1000 ≤ x ∧ x ≤ 9999) →
  s.sum id = 1652100 := by
  sorry

end sum_of_four_digit_numbers_l68_68873


namespace four_digit_palindrome_square_count_l68_68995

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68995


namespace sum_of_B_elements_l68_68358

def A : Set ℤ := {2, 0, 1, 3}

def B : Set ℤ := {x | x ∈ A ∧ (2 - x^2) ∉ A}

theorem sum_of_B_elements : (∑ x in B, x) = -5 := by
  -- Proof goes here
  sorry

end sum_of_B_elements_l68_68358


namespace valid_k_values_l68_68096

theorem valid_k_values :
  ∀ k : ℕ, 0 < k ∧ k ≤ 10 →
  (∃ (m : ℕ), (∏ i in finset.range k, m + i) % 10 = k) ↔ (k = 1 ∨ k = 2 ∨ k = 4) :=
by
  sorry

end valid_k_values_l68_68096


namespace inequality_solution_set_l68_68714

theorem inequality_solution_set (a : ℝ) :
  ∀ x : ℝ, ((ax - 1) * (x + 1) < 0 → (x ∈ (- ∞, 1 / a) ∪ (-1, ∞))) ↔ (-1 ≤ a ∧ a < 0) :=
by
  sorry

end inequality_solution_set_l68_68714


namespace construction_company_sand_weight_l68_68567

theorem construction_company_sand_weight :
  ∀ (total_weight gravel_weight : ℝ), total_weight = 14.02 → gravel_weight = 5.91 → 
  total_weight - gravel_weight = 8.11 :=
by 
  intros total_weight gravel_weight h_total h_gravel 
  sorry

end construction_company_sand_weight_l68_68567


namespace percentage_non_silver_new_shipment_l68_68905

theorem percentage_non_silver_new_shipment (initial_cars : ℕ) (initial_percentage_silver : ℝ) 
    (new_shipment : ℕ) (total_percentage_silver : ℝ) (initial_silver : ℕ) (total_silver : ℕ) (new_shipment_silver : ℕ) 
    (non_silver_percentage : ℝ) :
  initial_cars = 40 →
  initial_percentage_silver = 0.15 →
  new_shipment = 80 →
  total_percentage_silver = 0.25 →
  initial_silver = (initial_percentage_silver * initial_cars).nat_abs →
  total_silver = (total_percentage_silver * (initial_cars + new_shipment)).nat_abs →
  new_shipment_silver = total_silver - initial_silver →
  non_silver_percentage = ((new_shipment - new_shipment_silver : ℝ) / new_shipment) * 100 →
  non_silver_percentage = 70 :=
by
  intros
  sorry

end percentage_non_silver_new_shipment_l68_68905


namespace sum_of_roots_of_polynomial_eq_2008_l68_68658

noncomputable def polynomial := 
  (λ (x : ℝ), (x-1)^(2010) + 2 * (x-2)^(2009) + 3 * (x-3)^(2008) + 
    ∑ i in (finset.range 2006), (i + 3) * (x - (i + 3))^(2008 - i))

theorem sum_of_roots_of_polynomial_eq_2008 :
  polynomial.sum_roots polynomial = 2008 :=
by
  sorry

end sum_of_roots_of_polynomial_eq_2008_l68_68658


namespace Tn_lt_Sn_div2_l68_68296

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68296


namespace four_digit_perfect_square_palindrome_count_l68_68974

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68974


namespace four_digit_perfect_square_palindrome_count_l68_68970

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68970


namespace inequality_proof_l68_68373

variable {x y z : ℝ}

theorem inequality_proof (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x * y * z = 1) :
  ¬ (1 / (1 + x + x * y) > 1 / 3 ∧ 
     y / (1 + y + y * z) > 1 / 3 ∧
     (x * z) / (1 + z + x * z) > 1 / 3) :=
by
  sorry

end inequality_proof_l68_68373


namespace find_day_53_days_from_friday_l68_68487

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68487


namespace max_painted_cells_l68_68897

/-- This definition states that a rectangle inside a 100x100 grid is 'good' if the sum of 
the numbers in its cells is divisible by 17 -/
def good_rectangle (grid : ℕ → ℕ → ℕ) (x1 y1 x2 y2 : ℕ) : Prop :=
  (∑ i in Finset.range (x2 - x1 + 1), ∑ j in Finset.range (y2 - y1 + 1), grid (x1 + i) (y1 + j)) % 17 = 0

/-- The main proposition is to prove that it is possible to paint at least 9744 cells in a 
100x100 grid where each cell contains a natural number, and only good rectangles can be painted, 
and no cell is painted twice -/
theorem max_painted_cells (grid : ℕ → ℕ → ℕ) : ∃ d, d ≥ 9744 ∧ (∀ (cells : ℕ → ℕ → bool), 
  (∀ x y, cells x y → good_rectangle grid x y x y) ∧ 
  (∑ x in Finset.range 100, ∑ y in Finset.range 100, if cells x y then 1 else 0) = d)
:= sorry

end max_painted_cells_l68_68897


namespace Tn_lt_half_Sn_l68_68346

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68346


namespace day_after_53_days_from_Friday_l68_68513

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68513


namespace chandler_bike_purchase_l68_68070

theorem chandler_bike_purchase : 
  ∀ (x : ℕ), (120 + 20 * x = 640) → x = 26 := 
by
  sorry

end chandler_bike_purchase_l68_68070


namespace cube_root_of_0_000343_is_0_07_l68_68868

theorem cube_root_of_0_000343_is_0_07 :
  real.cbrt (0.000343) = 0.07 :=
sorry

end cube_root_of_0_000343_is_0_07_l68_68868


namespace find_sin_alpha_and_beta_l68_68699

theorem find_sin_alpha_and_beta (α β : ℝ) (h₁ : 0 < α ∧ α < π / 2 ∧ π / 2 < β ∧ β < π)
  (h₂ : real.tan (α / 2) = 1 / 2)
  (h₃ : real.cos (β - α) = real.sqrt 2 / 10) :
  real.sin α = 4 / 5 ∧ β = 3 * π / 4 :=
by
  sorry

end find_sin_alpha_and_beta_l68_68699


namespace isosceles_triangle_perimeter_l68_68146

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l68_68146


namespace find_larger_number_l68_68904

variable (b : ℕ) (h₁ : b = 57) (h₂ : b - 44 = 70 - b)

theorem find_larger_number (X : ℕ) (hX : X = 70) : X = 44 + (X - 57) :=
by
  rw [h₁] at h₂
  rw [ ←hX]
  sorry

end find_larger_number_l68_68904


namespace point_in_fourth_quadrant_l68_68789

open Complex

-- Definitions from conditions
def z : ℂ := 1 - I
def z_conj : ℂ := conj z
def z_mod : ℝ := abs z

-- The complex number we are interested in
def w : ℂ := (z_conj / z) + (z ^ 2) + z_mod

-- Lean statement for the proof problem
theorem point_in_fourth_quadrant : w.re > 0 ∧ w.im < 0 := by
  sorry

end point_in_fourth_quadrant_l68_68789


namespace find_f_x_l68_68672

theorem find_f_x (f : ℝ → ℝ) (h : ∀ x, f(x - 2) = x^2 - 4x) : ∀ x, f x = x^2 - 4 :=
by
  sorry

end find_f_x_l68_68672


namespace greatest_prime_factor_of_144_l68_68522

theorem greatest_prime_factor_of_144 : ∃ p, prime p ∧ p ∣ 144 ∧ (∀ q, prime q ∧ q ∣ 144 → q ≤ p) :=
sorry

end greatest_prime_factor_of_144_l68_68522


namespace vicente_total_spent_in_usd_l68_68865

theorem vicente_total_spent_in_usd :
  let rice_kg := 5
  let meat_lb := 3
  let cost_per_kg_rice := 2
  let cost_per_lb_meat := 5
  let discount_rate_rice := 0.10
  let tax_rate_meat := 0.05
  let exchange_rate_eur_usd := 1.20
  let exchange_rate_gbp_usd := 1.35 in
  let cost_rice_eur := rice_kg * cost_per_kg_rice
  let cost_meat_gbp := meat_lb * cost_per_lb_meat in
  let discounted_cost_rice_eur := cost_rice_eur * (1 - discount_rate_rice)
  let taxed_cost_meat_gbp := cost_meat_gbp * (1 + tax_rate_meat) in
  let cost_rice_usd := discounted_cost_rice_eur * exchange_rate_eur_usd
  let cost_meat_usd := taxed_cost_meat_gbp * exchange_rate_gbp_usd in
  cost_rice_usd + cost_meat_usd = 32.06 :=
by
  sorry

end vicente_total_spent_in_usd_l68_68865


namespace find_x6_l68_68379

-- Definition of the variables xi for i = 1, ..., 10.
variables {x1 x2 x3 x4 x5 x6 x7 x8 x9 x10 : ℝ}

-- Given conditions as equations.
axiom eq1 : (x2 + x4) / 2 = 3
axiom eq2 : (x4 + x6) / 2 = 5
axiom eq3 : (x6 + x8) / 2 = 7
axiom eq4 : (x8 + x10) / 2 = 9
axiom eq5 : (x10 + x2) / 2 = 1

axiom eq6 : (x1 + x3) / 2 = 2
axiom eq7 : (x3 + x5) / 2 = 4
axiom eq8 : (x5 + x7) / 2 = 6
axiom eq9 : (x7 + x9) / 2 = 8
axiom eq10 : (x9 + x1) / 2 = 10

-- The theorem to prove.
theorem find_x6 : x6 = 1 :=
by
  sorry

end find_x6_l68_68379


namespace evaluate_f_l68_68671

def f : ℝ → ℝ
| x => if x > 0 then 2 * x else f (x + 1)

theorem evaluate_f: f (f (2 / 3)) + f (-4 / 3) = 4 := by
  sorry

end evaluate_f_l68_68671


namespace strictly_increasing_on_interval_l68_68650

noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/3) (x^2 - 9)

theorem strictly_increasing_on_interval :
  ∀ x y : ℝ, x < y → x < -3 → y < -3 → f(x) < f(y) := by
  sorry

end strictly_increasing_on_interval_l68_68650


namespace math_proof_l68_68807

def problem_statement : Prop :=
  ∃ x : ℕ, (2 * x + 3 = 19) ∧ (x + (2 * x + 3) = 27)

theorem math_proof : problem_statement :=
  sorry

end math_proof_l68_68807


namespace _l68_68684

noncomputable def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ ∀ n, a (n + 1) = 2 * a n + 2^(n + 1)

noncomputable def a_n_formula (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = (2 * n - 1) * 2^(n - 1)

noncomputable theorem general_formula (a : ℕ → ℕ) :
  sequence a → a_n_formula a :=
sorry

noncomputable def S_n_formula (S a : ℕ → ℕ) : Prop :=
  ∀ n, S n = (2 * n - 3) * 2^n + 3

noncomputable theorem sum_formula (S a : ℕ → ℕ) :
  (∀ n, a n = (2 * n - 1) * 2^(n - 1)) →
  (∀ n, S n = ∑ k in finset.range n + 1, a k) →
  S_n_formula S a :=
sorry

end _l68_68684


namespace calculate_expression_value_l68_68614

-- Define the components of the expression
def x := 8.1^0
def y := (1 / 2)^(-2)
def z := Real.log 25
def w := 2 * Real.log 2

-- Define the main theorem to prove the expression equals -1
theorem calculate_expression_value : x - y + z + w = -1 :=
by
  sorry

end calculate_expression_value_l68_68614


namespace slope_angle_of_line_l68_68848

theorem slope_angle_of_line : 
  ∃ α : Real, (∃ (f : Real → Real) (a b: Real), (∀ x, f x = -x - 1) ∧ ∀ x, f(x) = a * x + b ∧ (a = -1) ∧ (α = 3 * Real.pi / 4) := 
begin
  sorry
end

end slope_angle_of_line_l68_68848


namespace parallel_tangents_solution_l68_68690

noncomputable def curve_1 (x : ℝ) (n : ℝ) : ℝ := x^n * Real.log x

noncomputable def curve_2 (x : ℝ) : ℝ := 2 * Real.exp 1 * Real.log x

noncomputable def tangent_parallel (x : ℝ) (n : ℝ) : Prop := 
  let y₁' := n * x^(n - 1) * Real.log x + x^(n - 1)
  let y₂' := (2 * Real.exp 1) / x
  y₁' = y₂'

noncomputable def S_n (xs : List ℝ) : ℝ := xs.sum

noncomputable def T_n (xs : List ℝ) : ℝ := xs.foldl (· * ·) 1

theorem parallel_tangents_solution 
  (n : ℝ) 
  (xs : List ℝ)
  (H_parallel : ∀ x ∈ xs, tangent_parallel x n) 
  (H_xn_positive : ∀ x ∈ xs, x > 0) 
  (H_length : xs.length = n.toNat) :
  T_n xs > n + 1 ∧ S_n xs < T_n xs + Real.log (T_n xs) + 1 :=
sorry

end parallel_tangents_solution_l68_68690


namespace product_ne_sum_11_times_l68_68706

def is_prime (n : ℕ) : Prop := ∀ m, m > 1 → m < n → n % m ≠ 0
def prime_sum_product_condition (a b c d : ℕ) : Prop := 
  a * b * c * d = 11 * (a + b + c + d)

theorem product_ne_sum_11_times (a b c d : ℕ)
  (ha : is_prime a) (hb : is_prime b) (hc : is_prime c) (hd : is_prime d)
  (h : prime_sum_product_condition a b c d) :
  (a + b + c + d ≠ 46) ∧ (a + b + c + d ≠ 47) ∧ (a + b + c + d ≠ 48) :=
by  
  sorry

end product_ne_sum_11_times_l68_68706


namespace incorrect_operation_l68_68005

theorem incorrect_operation :
  ¬(let a := 5 - (-2) in a = 7) ∧
  ¬(let b := -9 / (-3) in b = 3) ∧
  ¬(let c := -5 + 3 in c = 8) ∧
  ¬(let d := -4 * (-5) in d = 20) ∧
  (-5 + 3 = -2) ->
  c.
Proof := sorry

end incorrect_operation_l68_68005


namespace palindromic_squares_count_l68_68938

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68938


namespace find_day_53_days_from_friday_l68_68486

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68486


namespace inequality_proof_l68_68374

theorem inequality_proof (a b : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) :
  (sqrt a + sqrt b)^8 ≥ 64 * a * b * (a + b)^2 :=
by
  sorry

end inequality_proof_l68_68374


namespace find_a_plus_b_l68_68711

theorem find_a_plus_b : 
  ∃ a b m n : ℝ, a > 0 ∧ b > 0 ∧ (m^2 - a*m + b = 0) ∧ (n^2 - a*n + b = 0) ∧ (m ≠ n) ∧ (m + n = a) ∧ (mn = b) ∧ 
  ((2*n = m + 2 ∧ m*n = 4) ∨ (2*m = n + 2 ∧ m*n = 4)) ∧ (a + b = 9) :=
begin
  sorry
end

end find_a_plus_b_l68_68711


namespace tan_plus_cot_l68_68645

-- Define trigonometric identities for secant and cosecant
def sec (x : ℝ) : ℝ := 1 / Real.cos x
def csc (x : ℝ) : ℝ := 1 / Real.sin x
def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x
def cot (x : ℝ) : ℝ := Real.cos x / Real.sin x

-- Main theorem statement
theorem tan_plus_cot (x : ℝ) (h : sec x - csc x = 2 * Real.sqrt 6) : tan x + cot x = 6 ∨ tan x + cot x = -4 :=
sorry

end tan_plus_cot_l68_68645


namespace arithmetic_seq_divisibility_l68_68429

-- Definitions and assumptions based on conditions.
def a_seq (a₁ d_a i : ℕ) : ℕ := a₁ + (i - 1) * d_a
def b_seq (b₁ d_b j : ℕ) : ℕ := b₁ + (j - 1) * d_b

theorem arithmetic_seq_divisibility (a₁ d_a b₁ d_b : ℕ) (hyp : ∃∞ x : ℕ, ∃ y : ℕ,  x ≤ y ∧ y ≤ x + 2021 ∧ a_seq a₁ d_a x ∣ b_seq b₁ d_b y): 
  ∀ i : ℕ, ∃ j : ℕ, a_seq a₁ d_a i ∣ b_seq b₁ d_b j :=
by 
  sorry

end arithmetic_seq_divisibility_l68_68429


namespace angle_C_side_c_l68_68219

variable (A B C a b c : ℝ)

-- Given conditions
axiom root_conditions : a^2 - 2 * real.sqrt(5) * a + 4 = 0 ∧ b^2 - 2 * real.sqrt(5) * b + 4 = 0
axiom cos_sum : 2 * real.cos (A + B) = 1

-- Objectives
theorem angle_C :
  C = 120 := 
by
  -- Given the condition on the sum of cosines, we derive that angle C is 120 degrees
  sorry

theorem side_c:
  c = 4 :=
by
  -- Given the cosine rule and derived angle C to be 120 degrees, calculate the side c
  have h1_C : 2 * real.cos (A + B) = 1 := cos_sum,
  have h2_cos_ApB : real.cos (A + B) = 1 / 2 := by sorry,
  have h3_AplusB_60 : A + B = 60 := by sorry,
  have h4_C_120 : C = 180 - (A + B) := by sorry,
  have h5_roots_ab : a + b = 2 * real.sqrt(5) ∧ a * b = 4 := by sorry,
  have h6_a_b_reln : a^2 + b^2 = (a + b)^2 - 2 * a * b := by sorry,
  have h7_c_squared  : c^2 = a^2 + b^2 + a * b := by sorry,
  have h8_c_val : c = 4 := by sorry,
  apply h8_c_val,
  sorry
  

end angle_C_side_c_l68_68219


namespace snowball_melting_percentage_l68_68588

-- Define necessary constants and variables
variables (m : ℝ) (v : ℝ) (λ : ℝ) (k : ℝ)
-- Initial conditions
def initial_conditions : Prop :=
  λ = 330 ∧ k = 0.0002

-- The percentage of the snowball that melts upon impact given initial speed v
def melting_percentage (v : ℝ) (λ : ℝ) (k : ℝ) : ℝ :=
  (4 * k)

theorem snowball_melting_percentage 
  (m v λ k : ℝ)
  (h_init : initial_conditions λ k) :
  melting_percentage (2 * v) λ k = 0.008 :=
by sorry

end snowball_melting_percentage_l68_68588


namespace max_k_value_l68_68173

noncomputable def max_k : ℝ := sorry 

theorem max_k_value :
  ∀ (k : ℝ),
  (∃ (x y : ℝ), (x - 4)^2 + y^2 = 1 ∧ y = k * x - 2 ∧  (x - 4)^2 + y^2 ≤ 4) ↔ 
  k ≤ 4 / 3 := sorry

end max_k_value_l68_68173


namespace largest_n_factorial_divisor_l68_68107

theorem largest_n_factorial_divisor (n : ℕ) : ((2004!)! % ((n!)!)! = 0) → n ≤ 6 :=
by {
  sorry
}

end largest_n_factorial_divisor_l68_68107


namespace paint_paths_l68_68228

noncomputable def color_paths (m n : ℕ) (G : Type) [Fintype G] (edges : G → G → Prop) : Prop :=
∃ (coloring : G → G → Fin m), 
  (∀ (g : G) (h1 h2 : G), h1 ≠ h2 → coloring g h1 ≠ coloring g h2) ∧ 
  (∀ (c : Fin m), (fintype.card (coloring '' {p : G → G → Prop | edges p}) = n))

theorem paint_paths (m n : ℕ) (G : Type) [Fintype G] (edges : G → G → Prop) :
  color_paths m n G edges := 
sorry

end paint_paths_l68_68228


namespace linear_regression_eq_l68_68835

-- Define the points given in the problem
def points : List (ℝ × ℝ) := [(3, 10), (7, 20), (11, 24)]

-- Calculate averages
def mean_x : ℝ := (3 + 7 + 11) / 3
def mean_y : ℝ := (10 + 20 + 24) / 3

-- Calculate the slope b and intercept a
def b : ℝ := (3 * 10 + 7 * 20 + 11 * 24 - 7 * 18 * 3) / (3^2 + 7^2 + 11^2 - 7^2 * 3)
def a : ℝ := 18 - b * 7

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := a + b * x

-- The proof problem: proving the calculated results match the expected regression line equation
theorem linear_regression_eq :
  a = 5.75 ∧ b = 1.75 ∧ (∀ x, regression_eq x = 5.75 + 1.75 * x) :=
by
  split
  { -- Prove a = 5.75 
    sorry }
  split
  { -- Prove b = 1.75
    sorry }
  { -- Prove regression equation form
    intro x
    sorry }

end linear_regression_eq_l68_68835


namespace point_in_fourth_quadrant_l68_68402

theorem point_in_fourth_quadrant : 
  let z : ℂ := (2 - complex.i)^2
  in (z.re, z.im) = (3, -4) → z.re > 0 ∧ z.im < 0 :=
by
  intros z_def
  rw z_def
  simp
  refine ⟨by norm_num, by norm_num⟩
  sorry

end point_in_fourth_quadrant_l68_68402


namespace license_plates_count_l68_68048

theorem license_plates_count :
  let vowels := 5 -- choices for the first vowel
  let other_letters := 25 -- choices for the second and third letters
  let digits := 10 -- choices for each digit
  (vowels * other_letters * other_letters * (digits * digits * digits)) = 3125000 :=
by
  -- proof steps will go here
  sorry

end license_plates_count_l68_68048


namespace log_equality_l68_68291

theorem log_equality (x k : ℝ) (h1 : real.log 3 / real.log 8 = x) (h2 : real.log 9 / real.log 2 = k * x) : k = 6 :=
by
  sorry

end log_equality_l68_68291


namespace palindromic_squares_count_l68_68946

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68946


namespace garden_length_proof_l68_68889

variable (rows columns : ℕ) (dist boundary : ℝ) (garden_length : ℝ) 

def garden_length_correct (h_rows : rows = 10) 
  (h_columns : columns = 12)
  (h_dist : dist = 2)
  (h_boundary : boundary = 5)
  (h_garden_length : garden_length = 32) : Prop :=
  garden_length = 2 * (columns - 1) + 2 * boundary

theorem garden_length_proof : garden_length_correct 10 12 2 5 32 :=
by 
  unfold garden_length_correct
  sorry

end garden_length_proof_l68_68889


namespace bike_race_difference_l68_68220

-- Define the conditions
def carlos_miles : ℕ := 70
def dana_miles : ℕ := 50
def time_period : ℕ := 5

-- State the theorem to prove the difference in miles biked
theorem bike_race_difference :
  carlos_miles - dana_miles = 20 := 
sorry

end bike_race_difference_l68_68220


namespace totalCostOfAllPuppies_l68_68620

noncomputable def goldenRetrieverCost : ℕ :=
  let numberOfGoldenRetrievers := 3
  let puppiesPerGoldenRetriever := 4
  let shotsPerPuppy := 2
  let costPerShot := 5
  let vitaminCostPerMonth := 12
  let monthsOfSupplements := 6
  numberOfGoldenRetrievers * puppiesPerGoldenRetriever *
  (shotsPerPuppy * costPerShot + vitaminCostPerMonth * monthsOfSupplements)

noncomputable def germanShepherdCost : ℕ :=
  let numberOfGermanShepherds := 2
  let puppiesPerGermanShepherd := 5
  let shotsPerPuppy := 3
  let costPerShot := 8
  let microchipCost := 25
  let toyCost := 15
  numberOfGermanShepherds * puppiesPerGermanShepherd *
  (shotsPerPuppy * costPerShot + microchipCost + toyCost)

noncomputable def bulldogCost : ℕ :=
  let numberOfBulldogs := 4
  let puppiesPerBulldog := 3
  let shotsPerPuppy := 4
  let costPerShot := 10
  let collarCost := 20
  let chewToyCost := 18
  numberOfBulldogs * puppiesPerBulldog *
  (shotsPerPuppy * costPerShot + collarCost + chewToyCost)

theorem totalCostOfAllPuppies : goldenRetrieverCost + germanShepherdCost + bulldogCost = 2560 :=
by
  sorry

end totalCostOfAllPuppies_l68_68620


namespace triangular_plot_size_in_acres_l68_68076

theorem triangular_plot_size_in_acres
  (scale_cm_to_miles : ℕ)
  (base_cm height_cm : ℕ)
  (sq_mile_to_acres : ℕ)
  (h_scale : scale_cm_to_miles = 3)
  (h_base : base_cm = 8)
  (h_height : height_cm = 6)
  (h_sq_mile_to_acres : sq_mile_to_acres = 640) :
  (1 / 2 * base_cm * height_cm * scale_cm_to_miles ^ 2 * sq_mile_to_acres) = 138240 :=
by
  -- Definitions and conditions provided in the problem
  have h_area_cm2 : 1 / 2 * base_cm * height_cm = 24,
    by simp [h_base, h_height];
  have h_area_miles2 : scale_cm_to_miles ^ 2 = 9, 
    by simp [h_scale];
  have h_area_miles : h_area_cm2 * h_area_miles2 = 216,
    by simp [h_area_cm2, h_area_miles2];
  have h_area_acres : 216 * sq_mile_to_acres = 138240,
    by simp [h_sq_mile_to_acres];
  assumption -- final result follows from the conversions
  sorry

end triangular_plot_size_in_acres_l68_68076


namespace minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q_l68_68779

theorem minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q :
  let Q (z : ℂ) := z^8 + (6 * (2:ℂ).sqrt + 8) * z^4 - (6 * (2:ℂ).sqrt + 9)
  in ∃ polygon : list ℂ, 
      (∀ vertex, vertex ∈ polygon ↔ (Q vertex = 0)) ∧
      (polygon.length = 8) ∧
      (is_valid_polygon polygon) ∧
      (perimeter polygon = 8 * (2:ℂ).sqrt) := sorry

end minimum_perimeter_of_8_sided_polygon_with_zeros_of_Q_l68_68779


namespace T_lt_S_div_2_l68_68335

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68335


namespace Tn_lt_Sn_over_2_l68_68318

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68318


namespace number_of_ways_to_assign_roles_l68_68149

-- Definitions of conditions
def members : Finset String := {"Alice", "Bob", "Carol", "Dave"}
def roles : Finset String := {"president", "vice president", "secretary", "treasurer"}

-- Main proof statement
theorem number_of_ways_to_assign_roles : (members.card = 4 ∧ roles.card = 4) →
  (members.sum (λ _, 1) * roles.sum (λ _, 1) = 4!):= by
  sorry

end number_of_ways_to_assign_roles_l68_68149


namespace james_total_jail_time_l68_68270

theorem james_total_jail_time (arson_count arson_sentence explosive_multiplier domestic_terrorism_sentence : ℕ) :
    arson_count * arson_sentence + (2 * arson_count * arson_sentence) + domestic_terrorism_sentence = 56 :=
by
  -- Given conditions
  let arson_count := 2
  let arson_sentence := 6
  let explosive_multiplier := 2
  let domestic_terrorism_sentence := 20

  -- Compute the total jail time James might face
  let arson_total := arson_count * arson_sentence
  let explosive_sentence := explosive_multiplier * arson_total
  let total_sentence := arson_total + explosive_sentence + domestic_terrorism_sentence

  -- Verify the total sentence is as expected
  have h : total_sentence = 56 := sorry

  exact h

end james_total_jail_time_l68_68270


namespace palindromic_squares_count_l68_68943

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68943


namespace monotonic_decreasing_interval_l68_68396

noncomputable def log_base3 (x : ℝ) : ℝ := Real.log x / Real.log 3

def f (x : ℝ) : ℝ := log_base3 (x^2 - 2 * x)

theorem monotonic_decreasing_interval : ∀ x, x < 0 → ∀ x', x' < 0 → f x > f x' :=
sorry

end monotonic_decreasing_interval_l68_68396


namespace sum_of_prime_factors_of_2018_l68_68842

theorem sum_of_prime_factors_of_2018 : 
  ∃ p1 p2 : ℕ, prime p1 ∧ prime p2 ∧ 2018 = p1 * p2 ∧ p1 + p2 = 1011 :=
by
  use [2, 1009]
  split; exact prime_two
  split; try { sorry }
  split; try { sorry }
  exact sorry

end sum_of_prime_factors_of_2018_l68_68842


namespace min_value_least_one_l68_68742

noncomputable def min_modulus (z : ℂ) : ℝ :=
  abs (z + 1 + complex.i)

theorem min_value_least_one (z : ℂ) (h : complex.abs (z + 3 * complex.i) + complex.abs (z - 3 * complex.i) = 6) :
  ∃ (w : ℂ), min_modulus w = 1 :=
begin
  sorry
end

end min_value_least_one_l68_68742


namespace range_PA_PB_dot_product_l68_68235

theorem range_PA_PB_dot_product (a : ℝ) (SQRT3 : ℝ) :
  (1 ≤ a) → (a ≤ 5) → SQRT3 = real.sqrt 3 → 
  ∃ x, x = ((a - 2)^2 - 1) ∧ (-1 ≤ x) ∧ (x ≤ 8) :=
by
  sorry

end range_PA_PB_dot_product_l68_68235


namespace Tn_lt_Sn_div2_l68_68293

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68293


namespace twice_one_fifth_of_10_times_15_l68_68900

theorem twice_one_fifth_of_10_times_15 : 
  let a := 10 
  let b := 15 
  let mul := a * b 
  let one_fifth := (1 / 5) * mul 
  let result := 2 * one_fifth 
in result = 60 :=
by
  let a := 10 
  let b := 15
  let mul := a * b
  let one_fifth := (1 / 5) * mul
  let result := 2 * one_fifth
  sorry

end twice_one_fifth_of_10_times_15_l68_68900


namespace num_correct_propositions_is_one_l68_68431

def non_overlapping (A B : Type) := ¬(∃ x, x ∈ A ∧ x ∈ B)
def perpendicular (A B : Type) := ¬(∃ x, x ∈ A ∧ x ∈ B ∧ ∃ n, n ⊥ x)

variables 
  (m n : Set ℝ^3) -- Define lines m and n
  (α β : Set ℝ^3) -- Define planes α and β
  (h_lines_non_overlap : non_overlapping m n) -- Lines m and n do not overlap
  (h_planes_non_overlap : non_overlapping α β) -- Planes α and β do not overlap
  (h_prop1 : perpendicular m α ∧ perpendicular n β ∧ perpendicular m n → perpendicular α β)
  (h_prop2 : parallel m α ∧ parallel n β ∧ parallel m n → parallel α β)
  (h_prop3 : perpendicular m α ∧ parallel n β ∧ perpendicular m n → perpendicular α β)
  (h_prop4 : perpendicular m α ∧ parallel n β ∧ parallel m n → parallel α β)

theorem num_correct_propositions_is_one :
  ( (perpendicular m α ∧ perpendicular n β ∧ perpendicular m n → perpendicular α β)
  ∧ ¬(parallel m α ∧ parallel n β ∧ parallel m n → parallel α β)
  ∧ ¬(perpendicular m α ∧ parallel n β ∧ perpendicular m n → perpendicular α β)
  ∧ ¬(perpendicular m α ∧ parallel n β ∧ parallel m n → parallel α β) ) :=
by
  sorry

end num_correct_propositions_is_one_l68_68431


namespace james_jail_time_l68_68267

-- Definitions based on the conditions
def arson_sentence := 6
def arson_count := 2
def total_arson_sentence := arson_sentence * arson_count

def explosives_sentence := 2 * total_arson_sentence
def terrorism_sentence := 20

-- Total sentence calculation
def total_jail_time := total_arson_sentence + explosives_sentence + terrorism_sentence

-- The theorem we want to prove
theorem james_jail_time : total_jail_time = 56 := by
  sorry

end james_jail_time_l68_68267


namespace check_perfect_squares_l68_68880

-- Define the prime factorizations of each option
def optionA := 3^3 * 4^5 * 7^7
def optionB := 3^4 * 4^4 * 7^6
def optionC := 3^6 * 4^3 * 7^8
def optionD := 3^5 * 4^6 * 7^5
def optionE := 3^4 * 4^6 * 7^7

-- Definition of a perfect square (all exponents in prime factorization are even)
def is_perfect_square (n : ℕ) : Prop :=
  ∀ p : ℕ, (p ^ 2 ∣ n) -> (p ∣ n)

-- The Lean statement asserting which options are perfect squares
theorem check_perfect_squares :
  (is_perfect_square optionB) ∧ (is_perfect_square optionC) ∧
  ¬(is_perfect_square optionA) ∧ ¬(is_perfect_square optionD) ∧ ¬(is_perfect_square optionE) :=
by sorry

end check_perfect_squares_l68_68880


namespace days_from_friday_l68_68507

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68507


namespace fifty_three_days_from_Friday_is_Tuesday_l68_68469

def days_of_week : Type := {Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}

def modulo_days (n : ℕ) : ℕ := n % 7

def day_of_week_after_days (start_day : days_of_week) (n : ℕ) : days_of_week :=
  list.nth_le (list.rotate [Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday] (modulo_days n)) sorry -- using sorry for index proof

theorem fifty_three_days_from_Friday_is_Tuesday :
  day_of_week_after_days Friday 53 = Tuesday :=
sorry

end fifty_three_days_from_Friday_is_Tuesday_l68_68469


namespace sum_of_possible_k_values_l68_68241

theorem sum_of_possible_k_values (j k : ℕ) (h : j > 0 ∧ k > 0 ∧ (1 / j : ℚ) + (1 / k : ℚ) = 1 / 5) : 
  (k = 26 ∨ k = 10 ∨ k = 6) := sorry

example : ∑ (k ∈ {26, 10, 6}) = 42 := by
  simp

end sum_of_possible_k_values_l68_68241


namespace general_term_formula_l68_68682

open Real

noncomputable def sequence (c : ℕ) : ℕ → ℕ 
| 1 := c
| (n+1) := 
  let x_n := sequence n in
  x_n + (⌊ (2*x_n - (n+2) : ℚ) / n ⌋ : ℤ).to_nat + 1

theorem general_term_formula (c : ℕ) (k : ℕ) :
  ∀ n, 
  x_n = 
    (1/2) * (n+2) * (n+1) * k +
    1 + 
    match c % 3 with
    | 1 := 0
    | 2 := n
    | 0 := ⌊ (n+2)^2 / 4 ⌋
    end := 
sorry

end general_term_formula_l68_68682


namespace find_angle4_l68_68667

-- Definitions of angles and conditions
variables (angle1 angle2 angle3 angle4 : ℝ)
variable (deg70 : ℝ := 70)
variable (deg40 : ℝ := 40)
variable (deg180 : ℝ := 180)

-- Conditions
def condition1 : Prop := angle1 + angle2 = deg180
def condition2 : Prop := angle3 = angle4
def condition3 : Prop := angle1 + deg70 + deg40 = deg180
def condition4 : Prop := angle2 + angle3 + angle4 = deg180

-- Prove angle4 = 35 degrees
theorem find_angle4
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4)
  : angle4 = 35 := 
  sorry

end find_angle4_l68_68667


namespace inequality_transformation_l68_68132

variable {x y : ℝ}

theorem inequality_transformation (h : x > y) : x + 5 > y + 5 :=
by
  sorry

end inequality_transformation_l68_68132


namespace exist_integer_and_sequence_sum_cubes_eq_l68_68646

theorem exist_integer_and_sequence_sum_cubes_eq :
  ∃ (N : ℕ) (a : ℕ → ℤ), 
  (∀ k, k ∈ range (N + 1) → (a k = 1 ∨ a k = -1)) ∧ 
  (∑ k in range (N + 1), a k * (k : ℤ)^3 = 20162016) :=
by
  sorry

end exist_integer_and_sequence_sum_cubes_eq_l68_68646


namespace T_lt_half_S_l68_68326

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68326


namespace perfect_square_K_l68_68439

-- Definitions based on the conditions of the problem
variables (Z K : ℕ)
variables (h1 : 1000 < Z ∧ Z < 5000)
variables (h2 : K > 1)
variables (h3 : Z = K^3)

-- The statement we need to prove
theorem perfect_square_K :
  (∃ K : ℕ, 1000 < K^3 ∧ K^3 < 5000 ∧ K^3 = Z ∧ (∃ a : ℕ, K = a^2)) → K = 16 :=
sorry

end perfect_square_K_l68_68439


namespace coin_and_drink_values_l68_68922

theorem coin_and_drink_values 
  (k : ℕ)
  (a : Fin k → ℕ)
  (b : Fin k → ℕ)
  (h_a : ∀ i, a i < a (i + 1))
  (h_b : ∀ i, b i < b (i + 1))
  (h : ∀ n, ∃ j, ∑ i in Finset.range n, a ⟨i % k, sorry⟩ = b j) :
  ( ∃ n : ℕ, ∀ i, a i = i + 1 ∧ b i = i + n + 1 ) ∨ ( ∀ i, a i = b i ) :=
  sorry

end coin_and_drink_values_l68_68922


namespace percentage_of_games_lost_is_30_percent_l68_68407

theorem percentage_of_games_lost_is_30_percent
  (ratio_won_lost : ℕ → ℕ)
  (tie_games : ℕ)
  (y : ℕ)
  (h_ratio : ratio_won_lost 7 3)
  (h_tie : tie_games = 6) :
  let total_games := 7 * y + 3 * y + tie_games,
      lost_games : ℕ := 3 * y,
      percentage_lost := (lost_games * 100) / total_games
  in round (percentage_lost : ℝ) = 30 :=
by
  sorry

end percentage_of_games_lost_is_30_percent_l68_68407


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68455

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68455


namespace find_f1_f9_f96_l68_68832

def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f(x) < f(y)

def functional_condition (f : ℕ → ℕ) : Prop :=
  ∀ k : ℕ, f(f(k)) = 3 * k

theorem find_f1_f9_f96 (f : ℕ → ℕ) (h1 : strictly_increasing f) (h2 : functional_condition f) :
  f 1 + f 9 + f 96 = 197 :=
sorry

end find_f1_f9_f96_l68_68832


namespace unique_four_digit_palindromic_square_l68_68936

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68936


namespace perpendicular_EB_FC_l68_68770

noncomputable def Square (A B C D : Type) [EuclideanGeometry] :=
  square A B C D

noncomputable def Intersects (X Y Z : Type) [EuclideanGeometry] :=
  ∃ (P : Type), P ∈ (X Y) ∧ P ∈ (Y Z)

theorem perpendicular_EB_FC 
  (A B C D M E F : Type) 
  [EuclideanGeometry]
  (h1 : Square A B C D)
  (h2 : M ∈ segment B C) 
  (h3 : Intersects A M C D)
  (h4 : Intersects D M A B)
  (hE : E ∈ intersection A M C D)
  (hF : F ∈ intersection D M A B) :
  perpendicular (line E B) (line F C) :=
sorry

end perpendicular_EB_FC_l68_68770


namespace number_of_valid_rearrangements_l68_68117

def letters := ["a", "b", "c", "d"]
def adjacent (x : String) (y : String) : Bool :=
  (x = "a" ∧ y = "b") ∨ (x = "b" ∧ y = "c") ∨ (x = "c" ∧ y = "d")

noncomputable def valid_permutations (lst : List String) : List (List String) :=
  lst.permutations.filter (λ perm, (perm.zip perm.tail).all (λ p, ¬adjacent p.1 p.2))

theorem number_of_valid_rearrangements : valid_permutations letters = 2 := sorry

end number_of_valid_rearrangements_l68_68117


namespace palindromic_squares_count_l68_68945

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def count_palindromic_squares : ℕ :=
  (List.range' 32 (99 - 32 + 1)).count (λ n, 
    let sq := n * n in is_four_digit sq ∧ is_palindrome sq)

theorem palindromic_squares_count : count_palindromic_squares = 2 := by
  sorry

end palindromic_squares_count_l68_68945


namespace day_of_week_in_53_days_l68_68493

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68493


namespace greatest_prime_factor_of_144_l68_68527

theorem greatest_prime_factor_of_144 : 
  ∃ p, p = 3 ∧ prime p ∧ ∀ q, prime q ∧ (q ∣ 144) → q ≤ p := 
by
  sorry

end greatest_prime_factor_of_144_l68_68527


namespace reduced_number_l68_68033

theorem reduced_number (N : ℕ) (m a n : ℕ) (k : ℕ) (h1 : N = m + 10^k * a + 10^(k+1) * n)
  (h2 : a < 10) (h3 : m < 10^k) (h4 : N' = m + 10^(k+1) * n) (h5 : N = 6 * N') :
  N ∈ {12, 24, 36, 48} :=
sorry

end reduced_number_l68_68033


namespace four_digit_palindrome_square_count_l68_68992

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68992


namespace isosceles_triangle_perimeter_l68_68144

-- Define the given conditions
def is_isosceles_triangle (a b c : ℕ) : Prop :=
  (a = b ∨ b = c ∨ c = a)

def valid_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

-- Theorem based on the problem statement and conditions
theorem isosceles_triangle_perimeter (a b : ℕ) (P : is_isosceles_triangle a b 5) (Q : is_isosceles_triangle b a 10) :
  valid_triangle a b 5 → valid_triangle b a 10 → a + b + 5 = 25 :=
by sorry

end isosceles_triangle_perimeter_l68_68144


namespace no_four_digit_perfect_square_palindromes_l68_68959

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68959


namespace correct_representation_l68_68881

variable (Φ : Set ℕ)

theorem correct_representation :
  (Φ ⊆ {0}) ↔ 
  (Φ = {0} ∨ Φ = ∅) ∧ 
  ¬ (Φ ∈ {0}) ∧ 
  (Φ ⊆ {0}) ∧ 
  ∀ (x : ℕ), (x = 0) → x ∈ Φ → 0 = 0 :=
by
  sorry

end correct_representation_l68_68881


namespace fifty_three_days_from_friday_is_tuesday_l68_68477

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68477


namespace sum_of_possible_values_l68_68777

open Real

theorem sum_of_possible_values (p q r s : ℝ) (h1 : |p - q| = 1) (h2 : |q - r| = 5) (h3 : |r - s| = 6) :
    let possible_values := [abs(p - s), abs(p - (s + 12)), abs(p - (s - 12)), 
                            abs(p - (s + 10)), abs(p - (s - 10)), abs(p - (s + 2)), abs(p - (s - 2)), 0]
    possible_values.sum = 24 :=
  sorry

end sum_of_possible_values_l68_68777


namespace limit_fraction_l68_68065

open Real

theorem limit_fraction :
  tendsto (fun x => (x^3 - 4 * x^2 - 3 * x + 18) / (x^3 - 5 * x^2 + 3 * x + 9)) (𝓝 3) (𝓝 (5 / 4)) :=
sorry

end limit_fraction_l68_68065


namespace triangle_angles_sum_lt_90_l68_68755

open Real

variables {A B C O P : Point}
variable {triangle_ABC : acute_triangle A B C} 
variable {circumcenter_O : is_circumcenter O A B C}
variable {altitude_AP : is_altitude A P B C}
variable {angle_CB_plus_30 : angle C_triangle > angle B_triangle + 30}

theorem triangle_angles_sum_lt_90 :
  angle A + angle COP < 90 :=
sorry

end triangle_angles_sum_lt_90_l68_68755


namespace find_c_to_minimize_at_2_l68_68710

variable (c : ℝ)
def f (x : ℝ) := x * (x - c) ^ 2

-- Add theorem which proves the value of c for which f(x) has a minimum at x = 2
theorem find_c_to_minimize_at_2 (h : ∀ x: ℝ, has_deriv_at (f c) (3 * x^2 - 4 * c * x + c^2) x) : 
  (∀ x : ℝ, f' c 2 = 0 → c = 2) := 
sorry

end find_c_to_minimize_at_2_l68_68710


namespace no_four_digit_perfect_square_palindromes_l68_68962

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68962


namespace income_increase_by_parental_support_l68_68590

variables (a b c S : ℝ)

theorem income_increase_by_parental_support 
  (h1 : S = a + b + c)
  (h2 : 2 * a + b + c = 1.05 * S)
  (h3 : a + 2 * b + c = 1.15 * S) :
  (a + b + 2 * c) = 1.8 * S :=
sorry

end income_increase_by_parental_support_l68_68590


namespace greatest_prime_factor_of_144_l68_68521

theorem greatest_prime_factor_of_144 : ∃ p, prime p ∧ p ∣ 144 ∧ (∀ q, prime q ∧ q ∣ 144 → q ≤ p) :=
sorry

end greatest_prime_factor_of_144_l68_68521


namespace rehabilitation_centers_total_l68_68877

noncomputable def jane_visits (han_visits : ℕ) : ℕ := 2 * han_visits + 6
noncomputable def han_visits (jude_visits : ℕ) : ℕ := 2 * jude_visits - 2
noncomputable def jude_visits (lisa_visits : ℕ) : ℕ := lisa_visits / 2
def lisa_visits : ℕ := 6

def total_visits (jane_visits han_visits jude_visits lisa_visits : ℕ) : ℕ :=
  jane_visits + han_visits + jude_visits + lisa_visits

theorem rehabilitation_centers_total :
  total_visits (jane_visits (han_visits (jude_visits lisa_visits))) 
               (han_visits (jude_visits lisa_visits))
               (jude_visits lisa_visits) 
               lisa_visits = 27 :=
by
  sorry

end rehabilitation_centers_total_l68_68877


namespace train_distance_l68_68571

theorem train_distance (train_speed : ℝ) (total_time : ℝ) (distance : ℝ) :
  (train_speed = 1) → (total_time = 180) → (distance = train_speed * total_time) → 
  distance = 180 :=
by
  intros train_speed_eq total_time_eq dist_eq
  rw [train_speed_eq, total_time_eq] at dist_eq
  exact dist_eq

end train_distance_l68_68571


namespace day_of_week_in_53_days_l68_68494

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68494


namespace integer_solutions_b_l68_68419

theorem integer_solutions_b (b : ℤ) :
  (∃ x1 x2 : ℤ, x1 ≠ x2 ∧ ∀ x : ℤ, x1 ≤ x ∧ x ≤ x2 → x^2 + b * x + 3 ≤ 0) ↔ b = -4 ∨ b = 4 := 
sorry

end integer_solutions_b_l68_68419


namespace boss_monthly_pay_l68_68221

theorem boss_monthly_pay
  (fiona_hours_per_week : ℕ)
  (john_hours_per_week : ℕ)
  (jeremy_hours_per_week : ℕ)
  (hourly_rate : ℕ)
  (weeks_in_month : ℕ)
  (fiona_income : ℕ := fiona_hours_per_week * hourly_rate)
  (john_income : ℕ := john_hours_per_week * hourly_rate)
  (jeremy_income : ℕ := jeremy_hours_per_week * hourly_rate) :
  fiona_hours_per_week = 40 →
  john_hours_per_week = 30 →
  jeremy_hours_per_week = 25 →
  hourly_rate = 20 →
  weeks_in_month = 4 →
  (fiona_income + john_income + jeremy_income) * weeks_in_month = 7600 :=
begin
  intros h1 h2 h3 h4 h5,
  rw [h1, h2, h3, h4, h5],
  sorry -- This is the point where the proof would start
end

end boss_monthly_pay_l68_68221


namespace valid_range_of_m_l68_68696

variable (m : ℝ)
def p := m > 2 ∨ m < -2
def q := 1 < m ∧ m < 3

theorem valid_range_of_m (hpq_true : p ∨ q) (hpq_false : ¬(p ∧ q)) : 
  m < -2 ∨ m ≥ 3 ∨ (1 < m ∧ m ≤ 2) :=
  sorry

end valid_range_of_m_l68_68696


namespace no_solution_for_any_a_gt_1_l68_68100

theorem no_solution_for_any_a_gt_1 (b : ℝ) :
  b ∈ Icc 4 ∞ ∨ b ∈ Ioc (-4/3) 0 → 
  ∀ (a : ℝ), a > 1 → ¬∃ x : ℝ, a^(2-2*x^2) + (b+4) * a^(1-x^2) + 3*b + 4 = 0 := 
by
  sorry

end no_solution_for_any_a_gt_1_l68_68100


namespace four_digit_perfect_square_palindrome_count_l68_68978

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68978


namespace system_solution_l68_68133

theorem system_solution (x y : ℝ) (h1 : 2 * x + y = 6) (h2 : x + 2 * y = 3) : x - y = 3 :=
by
  -- proof goes here
  sorry

end system_solution_l68_68133


namespace ratio_of_logs_l68_68677

theorem ratio_of_logs (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (hlog1 : log 9 a = log 12 b) (hlog2 : log 12 b = log 16 (a + b)) :
  b / a = (1 + Real.sqrt 5) / 2 := 
sorry

end ratio_of_logs_l68_68677


namespace chocolate_chips_needed_l68_68908

theorem chocolate_chips_needed (cups_per_recipe : ℝ) (num_recipes : ℝ) (total_chips : ℝ) :
  cups_per_recipe = 3.5 → num_recipes = 37 → total_chips = cups_per_recipe * num_recipes → total_chips = 129.5 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end chocolate_chips_needed_l68_68908


namespace decagon_intersection_points_l68_68604

-- Define what it means to be a regular decagon and the concept of diagonals
def is_regular_decagon (P : Type) [fintype P] [decidable_eq P] (vertices : Π (n : ℕ), n = 10 → list P) : Prop :=
  list.length (vertices 10 rfl) = 10

-- Define the formula for the number of diagonals in an n-gon
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 3) / 2

-- Define the binomial coefficient which will be used to find intersections
def binom (n k : ℕ) : ℕ :=
  nat.choose n k

-- Define the main theorem
theorem decagon_intersection_points (P : Type) [fintype P] [decidable_eq P] (vertices : Π (n : ℕ), n = 10 → list P)
  (h : is_regular_decagon P vertices) :
  (number_of_diagonals 10 + binom 10 4) - 35 = 210 :=
begin
  sorry
end

end decagon_intersection_points_l68_68604


namespace fifty_three_days_from_friday_is_tuesday_l68_68471

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68471


namespace fifty_three_days_from_friday_is_tuesday_l68_68475

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68475


namespace Tn_lt_half_Sn_l68_68347

noncomputable def a_n (n : ℕ) : ℝ := (1/3)^(n-1)
noncomputable def b_n (n : ℕ) : ℝ := n * (1/3)^n
noncomputable def S_n (n : ℕ) : ℝ := 3/2 - 1/2 * (1/3)^(n-1)
noncomputable def T_n (n : ℕ) : ℝ := 3/4 - 1/4 * (1/3)^(n-1) - n/2 * (1/3)^n

theorem Tn_lt_half_Sn (n : ℕ) : T_n n < S_n n / 2 :=
by
  sorry

end Tn_lt_half_Sn_l68_68347


namespace number_of_solutions_l68_68727

theorem number_of_solutions : ∃! n, n = 3 ∧ ∀ (x y : ℤ), x^2019 + y^2 = 2 * y ↔ (x, y) = (0, 0) ∨ (x, y) = (0, 2) ∨ (x, y) = (1, 1) :=
by {
  -- We introduce the integer variables x and y
  existsi 3,
  split,
  -- Prove the cardinality part
  { -- Here you would fill in the detailed calculation steps to show there are exactly 3 solutions
    sorry },
  -- Prove the part with detailed matching of solutions with the conditions
  { intros x y,
    split;
    -- Here you would establish the equivalence of solutions
    -- (x, y) = (0, 0), (0, 2), and (1, 1)
    intro h,
    sorry }
}

end number_of_solutions_l68_68727


namespace david_dogs_left_l68_68085

def total_dogs_left (boxes_small: Nat) (dogs_per_small: Nat) (boxes_large: Nat) (dogs_per_large: Nat) (giveaway_small: Nat) (giveaway_large: Nat): Nat :=
  let total_small := boxes_small * dogs_per_small
  let total_large := boxes_large * dogs_per_large
  let remaining_small := total_small - giveaway_small
  let remaining_large := total_large - giveaway_large
  remaining_small + remaining_large

theorem david_dogs_left :
  total_dogs_left 7 4 5 3 2 1 = 40 := by
  sorry

end david_dogs_left_l68_68085


namespace day_53_days_from_friday_l68_68442

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l68_68442


namespace parabola_AB_distance_AOC_lambda_l68_68183

/-- Given the parabola y^2 = 8x, with points A(x1, y1) and B(x2, y2) on it, where the line passing through these points has a slope 2√2:
  Prove that the distance between points A and B is 4.
  Prove that if \(\overrightarrow{OC} = \overrightarrow{OA} + \lambda \overrightarrow{OB}\), then \(\lambda\) can be 0 or 2. --/
theorem parabola_AB_distance_AOC_lambda :
  (∃ (x₁ x₂ y₁ y₂ : ℝ),
    (y₁^2 = 8*x₁) ∧
    (y₂^2 = 8*x₂) ∧
    ((y₂ - y₁) / (x₂ - x₁) = 2*real.sqrt 2) ∧
    ((x₁ < x₂) ∧
      ((real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 4) ∧
      (∀ (λ : ℝ) (x₃ y₃ : ℝ),
        (x₃ = x₁ + λ*(x₂ - x₁)) ∧
        (y₃ = y₁ + λ*(y₂ - y₁)) ∧
        (y₃^2 = 8*x₃) → 
          ((λ = 0) ∨ (λ = 2))))) :=
sorry

end parabola_AB_distance_AOC_lambda_l68_68183


namespace files_remaining_l68_68898

def totalFilesOnDrive (musicFiles : ℕ) (videoFiles : ℕ) (deletedFiles : ℕ) : ℕ :=
  (musicFiles + videoFiles) - deletedFiles

theorem files_remaining (musicFiles : ℕ) (videoFiles : ℕ) (deletedFiles : ℕ) (h_music : musicFiles = 27) (h_video : videoFiles = 42) (h_delete : deletedFiles = 11) :
  totalFilesOnDrive musicFiles videoFiles deletedFiles = 58 :=
  by
  simp [totalFilesOnDrive, h_music, h_video, h_delete]
  sorry

end files_remaining_l68_68898


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68453

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68453


namespace T_lt_S_div_2_l68_68337

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68337


namespace ball_hits_ground_time_l68_68561

noncomputable def h (t : ℝ) : ℝ := -16 * t^2 - 30 * t + 180

theorem ball_hits_ground_time :
  ∃ t : ℝ, h t = 0 ∧ t = 2.545 :=
by
  sorry

end ball_hits_ground_time_l68_68561


namespace max_candies_in_26_minutes_l68_68804

/-- On the board, 26 ones are written. Every minute, Karlson erases two arbitrary numbers
and writes their sum on the board, and then eats a number of candies equal to the product
of the two erased numbers. Prove that the maximum number of candies he could eat 
in 26 minutes is 325. -/
theorem max_candies_in_26_minutes : 
  ∀ (nums : list ℕ) (h_len : nums.length = 26) (h_all_ones : ∀ n ∈ nums, n = 1), 
  (∃ max_candies, max_candies = 325) := 
sorry

end max_candies_in_26_minutes_l68_68804


namespace woody_saves_l68_68541

variable (C A W : ℕ)

theorem woody_saves (C A W : ℕ) (H1 : C = 282) (H2 : A = 42) (H3 : W = 24) :
  let additional_amount_needed := C - A in
  let weeks := additional_amount_needed / W in
  weeks = 10 :=
by
  unfold additional_amount_needed weeks
  rw [H1, H2, H3]
  simp
  norm_num
  sorry -- Proof not provided in this exercise

end woody_saves_l68_68541


namespace discard_one_point_l68_68676

-- Let P be a predicate representing points in a plane
def points_on_line (P : Set Point) : Prop :=
  ∀ A B C : Point, A ∈ P → B ∈ P → C ∈ P → collinear {A, B, C}

-- Given a set of n points satisfying the conditions
def satisfies_condition (P : Set Point) : Prop :=
  ∀ A B C D : Point, A ∈ P → B ∈ P → C ∈ P → D ∈ P → 
  (∃ X : Point, X ∈ {A, B, C, D} ∧ collinear ({A, B, C, D} \ {X}))

-- Proposition: If a set P of n points satisfies the condition, then there exists one point 
-- that can be removed so that all remaining points are collinear.
theorem discard_one_point {P : Set Point} (h : satisfies_condition P) : 
  ∃ X : Point, X ∈ P ∧ points_on_line (P \ {X}) :=
sorry

end discard_one_point_l68_68676


namespace find_f_log_value_l68_68715

/- Define the function f with given properties -/
noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x ≤ 1 then 3^x - 1 else 0 -- this is a partial definition aiding the statement. The definition of f outside [0, 1] can be adjusted if needed based on periodic property. 

/-- Formal statement to prove that f(log_(1/3) 36) equals the given value if f is periodic and given other conditions --/
theorem find_f_log_value :
  (f : ℝ → ℝ) ) 
  (∀ x, f(x + 3) = f(x)) 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f(x) = 3^x - 1 ) 
  : f (log ₃ 36) 3^{3}_mathbb= - 
begin 
 sorry 
 end 
 
 
end find_f_log_value_l68_68715


namespace Billy_hiking_distance_l68_68610

noncomputable def distance_from_start (east1 north: ℕ) (east2: ℝ) : ℝ :=
  let total_east := east1 + east2
  let total_north := north
  Real.sqrt (total_east^2 + total_north^2)

theorem Billy_hiking_distance : distance_from_start 8 (4 * Real.sqrt 2) (4 * Real.sqrt 2) = 4 * Real.sqrt (6 + 4 * Real.sqrt 2) := by
  sorry

end Billy_hiking_distance_l68_68610


namespace y_minus_x_eq_seven_point_five_l68_68547

theorem y_minus_x_eq_seven_point_five (x y : ℚ) (h1 : x + y = 8) (h2 : y - 3 * x = 7) :
  y - x = 7.5 :=
by sorry

end y_minus_x_eq_seven_point_five_l68_68547


namespace xyw_square_sum_l68_68081

noncomputable def N (x y w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 3 * y, w], ![x, y, -2 * w], ![x, -y, w]]

def M (x y w : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![2 * x^2, 0, 0], ![0, 10 * y^2, 0], ![0, 0, 6 * w^2]]

def N_transpose_N_eq_2I (x y w : ℝ) : Prop :=
  (N x y w)ᵀ ⬝ (N x y w) = 2 • (1 : Matrix (Fin 3) (Fin 3) ℝ)

theorem xyw_square_sum (x y w : ℝ) (h : N_transpose_N_eq_2I x y w) :
  x^2 + y^2 + w^2 = (23 / 15 : ℝ) :=
sorry

end xyw_square_sum_l68_68081


namespace count_palindromic_four_digit_perfect_squares_l68_68988

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68988


namespace coin_order_l68_68417

-- Define the coins as types for simplicity
inductive Coin
| F | E | C | D | A | B

open Coin

-- Define the conditions as relations between coins
axiom h1 : ¬ (∃ x, x = F ∧ (x = E ∨ x = C ∨ x = D ∨ x = A ∨ x = B))  
axiom h2 : (∀ x, x ≠ F → E = x ∨ E = C ∨ E = D)
axiom h3 : (∀ x, x ≠ F ∧ x ≠ E → C = x ∨ C = A)
axiom h4 : (∀ x, x ≠ F ∧ x ≠ E → D = x ∨ D = B) 
axiom h5 : A = B
axiom h6 : (∀ x, x ≠ F ∧ x ≠ E ∧ x ≠ C ∧ x ≠ D ∧ x ≠ A ∧ x ≠ B)

-- Define the goal that we want to prove
theorem coin_order : 
  ∀ o : list Coin, o = [F, E, C, D, A, B] :=
by 
  sorry

end coin_order_l68_68417


namespace number_of_schedules_l68_68806

-- Definitions based on conditions
def classes := { "Chinese", "Mathematics", "Physical Education", "Computer Science" }
def can_schedule (schedule : List String) : Prop :=
  "Physical Education" ∈ schedule.drop 1 -- Physical Education cannot be scheduled first

-- Theorem statement
theorem number_of_schedules : ∃ (schedules : List (List String)), 
  (∀ s ∈ schedules, can_schedule s) ∧ schedules.length = 18 :=
begin
  sorry
end

end number_of_schedules_l68_68806


namespace no_four_digit_perfect_square_palindromes_l68_68967

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68967


namespace f_at_neg1_l68_68703

-- Define the function f and the conditions
def f (x : ℝ) : ℝ := if x > 0 then x^2 + (1 / x) else -(x^2 + (1 / -x))

-- Declare the property that f is an odd function
lemma f_is_odd (x : ℝ) : f (-x) = -f x :=
by
  sorry

-- The main statement to prove
theorem f_at_neg1 : f (-1) = -2 :=
by
  sorry

end f_at_neg1_l68_68703


namespace find_BN_l68_68218

-- Let the variables and points be defined
variables (A B C M N : Type)
variables [has_eq A] [has_eq B] [has_eq C] [has_eq M] [has_eq N]

-- Define the conditions as per the problem
def midpoint (M B C : Type): Prop := sorry 
def bisects_angle (A N : Type) (BAC : Type) : Prop := sorry
def perp (B N : Type) (AN : Type) : Prop := sorry

axiom AB_length : nat
axiom AC_length : nat

def problem_statement (A B C M N : Type) (AB_length AC_length : nat) : Prop :=
  midpoint M B C ∧
  bisects_angle A N BAC ∧
  perp B N AN ∧
  AB_length = 13 ∧
  AC_length = 17 ∧
  BN = 221/30

theorem find_BN (A B C M N: Type) (AB_length AC_length : nat)
  (h1 : midpoint M B C)
  (h2 : bisects_angle A N BAC)
  (h3 : perp B N AN)
  (h4 : AB_length = 13)
  (h5 : AC_length = 17)
  : problem_statement A B C M N AB_length AC_length :=
begin
  sorry
end

end find_BN_l68_68218


namespace E_Z_l68_68438

variables {n : ℕ} {P : ℝ}
variables (X : ℕ → ℝ) (Y Z : ℕ → ℝ)

-- Event A occurs with probability P in each Bernoulli trial
axiom hP_A : 0 < P ∧ P < 1

-- X follows a Binomial distribution B(n, P)
axiom X_binom : ∀ k, X k = -- Insert the required definition for Binomial distribution

-- Y is geometrically distributed with probability P
axiom Y_geom : ∀ k, k ≥ 1 → Y k = (P * (1 - P)^(k - 1))

-- Expectation of Y is 1/P
axiom E_Y : ∑ k, Y k * k = 1/P

-- Z follows the given distribution
axiom Z_dist : ∀ k, k ≥ 2 → Z k = P * (1 - P)^(k - 1) + (1 - P) * P^(k - 1)

theorem E_Z : (∑ k, Z k * k) = 1 / (P * (1 - P)) - 1 := sorry

end E_Z_l68_68438


namespace min_dot_product_of_tangents_l68_68679

-- Definition of the problem conditions
def circle_radius : ℝ := 2
def points_tangency (O A B P : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace P] :=
  (tangent_circle O A) ∧ (tangent_circle O B) ∧ (A ≠ B)

-- Statement of the proof problem
theorem min_dot_product_of_tangents
  (O A B P : Type) [MetricSpace O] [MetricSpace A] [MetricSpace B] [MetricSpace P]
  (h : points_tangency O A B P) :
  ∃ α : ℝ, α = 1 - sqrt 2 ∧ 
  (4 * (1 - α) + 8 / (1 - α) - 12 = 8 * sqrt 2 - 12) :=
sorry

end min_dot_product_of_tangents_l68_68679


namespace factor_expression_l68_68622

theorem factor_expression (x : ℝ) : 12 * x^2 - 6 * x = 6 * x * (2 * x - 1) :=
by
sorry

end factor_expression_l68_68622


namespace min_value_of_E_l68_68556

theorem min_value_of_E (a b : ℕ) (h1 : 0 < b) (h2 : b < a) :
  let E := abs ((a + 2 * b) / (a - b) + (a - b) / (a + 2 * b)) in
  E = 2 :=
by {
  -- Proof omitted
  sorry
}

end min_value_of_E_l68_68556


namespace number_of_oranges_l68_68794

-- Definitions of the conditions
def peaches : ℕ := 9
def pears : ℕ := 18
def greatest_num_per_basket : ℕ := 3
def num_baskets_peaches := peaches / greatest_num_per_basket
def num_baskets_pears := pears / greatest_num_per_basket
def min_num_baskets := min num_baskets_peaches num_baskets_pears

-- Proof problem statement
theorem number_of_oranges (O : ℕ) (h1 : O % greatest_num_per_basket = 0) 
  (h2 : O / greatest_num_per_basket = min_num_baskets) : 
  O = 9 :=
by {
  sorry
}

end number_of_oranges_l68_68794


namespace four_digit_palindrome_square_count_l68_68994

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

theorem four_digit_palindrome_square_count : 
  ∃! (n : ℕ), is_four_digit n ∧ is_palindrome n ∧ (∃ (m : ℕ), n = m * m) := by
  sorry

end four_digit_palindrome_square_count_l68_68994


namespace unique_four_digit_palindromic_square_l68_68933

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_palindromic_square :
  ∃! n : ℕ, is_palindrome n ∧ is_four_digit n ∧ ∃ m : ℕ, m^2 = n :=
sorry

end unique_four_digit_palindromic_square_l68_68933


namespace cyclic_permutations_of_repeating_decimals_l68_68840

theorem cyclic_permutations_of_repeating_decimals {p : ℕ} [hp : Fact p.Prime] :
  (period_of_decimal_expansion (1 / p) = p - 1) →
  ∀ k : ℕ, k ∈ set.Ico 1 p.to_nat → 
  ∃ n : ℕ, decimal_expansion (k / p) = cyclic_permutation n (decimal_expansion (1 / p)) :=
begin
  sorry
end

end cyclic_permutations_of_repeating_decimals_l68_68840


namespace count_palindromic_four_digit_perfect_squares_l68_68987

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68987


namespace beta_speed_l68_68822

theorem beta_speed (d : ℕ) (S_s : ℕ) (t : ℕ) (S_b : ℕ) :
  d = 490 ∧ S_s = 37 ∧ t = 7 ∧ (S_s * t) + (S_b * t) = d → S_b = 33 := by
  sorry

end beta_speed_l68_68822


namespace proof_problem_l68_68783

noncomputable def polynomial_roots : (a b c : ℝ) → Prop :=
λ a b c, a^3 - 8 * a^2 + 10 * a - 1 = 0 ∧ b^3 - 8 * b^2 + 10 * b - 1 = 0 ∧ c^3 - 8 * c^2 + 10 * c - 1 = 0

noncomputable def p (a b c : ℝ) : ℝ := sqrt a + sqrt b + sqrt c

theorem proof_problem (a b c : ℝ) (h : polynomial_roots a b c) :
  let p := p a b c in
  p^4 - 16 * p^2 - 8 * p = -24 :=
sorry

end proof_problem_l68_68783


namespace inverse_proportion_function_m_neg_l68_68165

theorem inverse_proportion_function_m_neg
  (x : ℝ) (y : ℝ) (m : ℝ)
  (h1 : y = m / x)
  (h2 : (x < 0 → y > 0) ∧ (x > 0 → y < 0)) :
  m < 0 :=
sorry

end inverse_proportion_function_m_neg_l68_68165


namespace abs_expression_value_l68_68625

theorem abs_expression_value : (abs (2 * Real.pi - abs (Real.pi - 9))) = 3 * Real.pi - 9 := 
by
  sorry

end abs_expression_value_l68_68625


namespace inequality_proof_l68_68352

-- Define the variables and conditions
variables (x y z k : ℝ)
def positive (a : ℝ) := 0 < a

-- Add the non-computable annotation where needed
noncomputable def log2 := real.log 2
noncomputable def log3 := real.log 3
noncomputable def log5 := real.log 5

-- Assume conditions
variable (hx : positive x)
variable (hy : positive y)
variable (hz : positive z)
variable (heq : 2^x = 3^y ∧ 3^y = 5^z)

-- State the theorem
theorem inequality_proof : 3 * y < 2 * x ∧ 2 * x < 5 * z :=
by
  -- Proof statement to be completed
  sorry

end inequality_proof_l68_68352


namespace average_30_matches_is_25_l68_68549

noncomputable def average_runs_in_30_matches (average_20_matches average_10_matches : ℝ) (total_matches_20 total_matches_10 : ℕ) : ℝ :=
  let total_runs_20 := total_matches_20 * average_20_matches
  let total_runs_10 := total_matches_10 * average_10_matches
  (total_runs_20 + total_runs_10) / (total_matches_20 + total_matches_10)

theorem average_30_matches_is_25 (h1 : average_runs_in_30_matches 30 15 20 10 = 25) : 
  average_runs_in_30_matches 30 15 20 10 = 25 := 
  by
    exact h1

end average_30_matches_is_25_l68_68549


namespace part1_confidence_part2_probability_part2_expectation_l68_68565

noncomputable def contingency_table : (ℕ × ℕ × ℕ × ℕ) :=
  (15, 5, 10, 20)

noncomputable def sample_size : ℕ := 50

noncomputable def K_squared (n a b c d : ℕ) : ℝ :=
  let num := (n : ℝ) * ((a * d - b * c : ℕ) : ℝ) ^ 2
  let denom := ((a + b) * (c + d) * (a + c) * (b + d) : ℕ) : ℝ
  num / denom

noncomputable def confidence_level (k_squared : ℝ) : ℝ :=
  if k_squared ≥ 7.879 then 0.995 else if k_squared ≥ 6.635 then 0.99 else 0.95

theorem part1_confidence :
  let (a, b, c, d) := contingency_table in
  let k_squared := K_squared sample_size a b c d in
  confidence_level k_squared = 0.995 := sorry

noncomputable def P_X (x : ℕ) : ℚ :=
  match x with
  | 0 => 1 / 15
  | 1 => 8 / 15
  | 2 => 6 / 15
  | _ => 0

noncomputable def expected_value_X : ℚ :=
  ∑ i in ({0, 1, 2} : finset ℕ), (i : ℚ) * P_X i

theorem part2_probability :
  (P_X 0 = 1 / 15) ∧ (P_X 1 = 8 / 15) ∧ (P_X 2 = 6 / 15) :=
    by simp

theorem part2_expectation : expected_value_X = 4 / 3 := sorry

end part1_confidence_part2_probability_part2_expectation_l68_68565


namespace main_theorem_l68_68123

-- Define a function to evaluate the expression by interpreting "+" and "×" as given
def eval_expression (e : Expr) : ℕ := 
  match e with
  | Expr.one        => 1
  | Expr.add e1 e2  => eval_expression e1 + eval_expression e2
  | Expr.mul e1 e2  => eval_expression e1 * eval_expression e2
  end

-- Define a function to transform the expression by swapping "+" and "×"
def transform_expression (e : Expr) : Expr := 
  match e with
  | Expr.one       => Expr.one
  | Expr.add e1 e2 => Expr.mul (transform_expression e1) (transform_expression e2)
  | Expr.mul e1 e2 => Expr.add (transform_expression e1) (transform_expression e2)
  end

-- Define the expression data type
inductive Expr
| one : Expr
| add : Expr → Expr → Expr
| mul : Expr → Expr → Expr

-- Example expression for the problem
def example_expr : Expr := 
  Expr.add (Expr.add (Expr.add (Expr.add (Expr.add (Expr.add (Expr.add (Expr.add (Expr.add (Expr.add 
  Expr.one Expr.one) Expr.one) Expr.one) Expr.one) Expr.one) Expr.one) Expr.one) Expr.one) Expr.one) Expr.one

-- Prove the main theorem
theorem main_theorem : 
  eval_expression example_expr = 11 ∧ eval_expression (transform_expression example_expr) = 11 :=
by 
  -- These steps are omitted and replaced with sorry for the example's purpose
  sorry

end main_theorem_l68_68123


namespace find_num_nickels_l68_68863

-- Define the values of the coins in dollars
def value_of_quarter : ℝ := 0.25
def value_of_dime : ℝ := 0.10
def value_of_nickel : ℝ := 0.05
def value_of_penny : ℝ := 0.01

-- Define the quantities of each type of coin
def num_quarters : ℕ := 10
def num_dimes : ℕ := 3
def num_pennies : ℕ := 200

-- Define the total amount of money found
def total_amount_found : ℝ := 5.00

-- Calculate the total value of the known coins
def total_value_known_coins : ℝ :=
  (num_quarters * value_of_quarter) + (num_dimes * value_of_dime) + (num_pennies * value_of_penny)

-- Calculate the value of the nickels
def value_of_nickels : ℝ := total_amount_found - total_value_known_coins

-- Calculate the number of nickels
def num_nickels : ℕ := value_of_nickels / value_of_nickel

-- Prove that the number of nickels found is 4
theorem find_num_nickels : num_nickels = 4 := by
  sorry

end find_num_nickels_l68_68863


namespace day_after_53_days_from_Friday_l68_68511

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68511


namespace abs_expression_equality_l68_68626

def pi : ℝ := Real.pi

theorem abs_expression_equality : abs (2 * pi - abs (pi - 9)) = 3 * pi - 9 := by
  sorry

end abs_expression_equality_l68_68626


namespace christina_age_fraction_l68_68234

theorem christina_age_fraction {C : ℕ} (h1 : ∃ C : ℕ, (6 + 15) = (3/5 : ℚ) * C)
  (h2 : C + 5 = 40) : (C + 5) / 80 = 1 / 2 :=
by
  sorry

end christina_age_fraction_l68_68234


namespace probability_between_lines_l68_68216

theorem probability_between_lines : 
  let l := { p : ℝ × ℝ | p.2 = -p.1 + 6 }
  let m := { p : ℝ × ℝ | p.2 = -4 * p.1 + 6 }
  let area_triangle {x1 x2 y1 y2 : ℝ} (x1 ≠ x2) (y1 ≠ y2) := 
        0.5 * (abs (x2 - x1) * abs (y2 - y1))
  let area_under_l := area_triangle 0 6 0 6
  let area_under_m := area_triangle 0 (3 / 2) 0 6
  let area_between := area_under_l - area_under_m
  area_between / area_under_l = 0.75 :=
by
  sorry

end probability_between_lines_l68_68216


namespace inclination_sine_l68_68386

-- Define the ellipse C
def ellipse (x y : ℝ) := (x^2) / 4 + (y^2) / 3 = 1

-- Define the hyperbola E with parameters a, b > 0
def hyperbola (x y a b : ℝ) (ha : a > 0) (hb : b > 0) := (x^2) / (a^2) - (y^2) / (b^2) = 1

-- Condition: the ellipse and hyperbola have the same foci
def same_foci (f1 f2 : ℝ × ℝ) := f1 = (1, 0) ∧ f2 = (-1, 0)

-- Condition: the eccentricities are reciprocal
def reciprocal_eccentricities (e_ellipse e_hyperbola : ℝ) := e_ellipse = 1 / e_hyperbola

-- The final goal is to prove the sine of the angle of inclination is sqrt(3) / 2
theorem inclination_sine {a b e_ellipse e_hyperbola : ℝ} (h_ellipse : ∀ x y, ellipse x y)
  (h_hyperbola : ∀ x y, hyperbola x y a b) (h_foci : same_foci (1, 0) (-1, 0))
  (h_recip : reciprocal_eccentricities e_ellipse e_hyperbola) :
  a = 1 / 2 → b = real.sqrt 3 / 2 → real.sin (real.atan (real.sqrt 3)) = real.sqrt 3 / 2 :=
begin
  sorry
end

end inclination_sine_l68_68386


namespace value_of_A_l68_68217

theorem value_of_A (A : ℕ) : (A * 1000 + 567) % 100 < 50 → (A * 1000 + 567) / 10 * 10 = 2560 → A = 2 :=
by
  intro h1 h2
  sorry

end value_of_A_l68_68217


namespace f_defined_iff_a_ge_neg_three_fourths_l68_68354

theorem f_defined_iff_a_ge_neg_three_fourths 
  (a : ℝ) 
  (f : ℝ → ℝ := λ x, real.log10 ((1 + 2^x + 4^x * a) / 3)) 
  (h : ∀ x < 1, (1 + 2^x + 4^x * a) / 3 > 0) :
  a ≥ -3 / 4 :=
sorry

end f_defined_iff_a_ge_neg_three_fourths_l68_68354


namespace no_four_digit_perfect_square_palindromes_l68_68966

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68966


namespace four_digit_palindromic_perfect_square_l68_68956

def is_palindrome (n : ℕ) : Prop :=
  n.to_string = n.to_string.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n

theorem four_digit_palindromic_perfect_square :
  {n : ℕ | is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n}.card = 1 :=
sorry

end four_digit_palindromic_perfect_square_l68_68956


namespace arithmetic_sequence_common_difference_l68_68752
-- Lean 4 Proof Statement


theorem arithmetic_sequence_common_difference 
  (a : ℕ) (n : ℕ) (d : ℕ) (S_n : ℕ) (a_n : ℕ) 
  (h1 : a = 2) 
  (h2 : a_n = 29) 
  (h3 : S_n = 155) 
  (h4 : S_n = n * (a + a_n) / 2) 
  (h5 : a_n = a + (n - 1) * d) 
  : d = 3 := 
by 
  sorry

end arithmetic_sequence_common_difference_l68_68752


namespace jangshe_clothing_cost_l68_68766

theorem jangshe_clothing_cost
  (total_spent : ℝ)
  (untaxed_piece1 : ℝ)
  (untaxed_piece2 : ℝ)
  (total_pieces : ℕ)
  (remaining_pieces : ℕ)
  (remaining_pieces_price : ℝ)
  (sales_tax : ℝ)
  (price_multiple_of_five : ℝ) :
  total_spent = 610 ∧
  untaxed_piece1 = 49 ∧
  untaxed_piece2 = 81 ∧
  total_pieces = 7 ∧
  remaining_pieces = 5 ∧
  sales_tax = 0.10 ∧
  (∃ k : ℕ, remaining_pieces_price = k * 5) →
  remaining_pieces_price / remaining_pieces = 87 :=
by
  sorry

end jangshe_clothing_cost_l68_68766


namespace ellipse_problem_l68_68686

theorem ellipse_problem
  (m : ℝ)
  (h_ellipse : ∀ x y : ℝ, (x^2) / m + y^2 = 1)
  (h_P : (m = 4) → (eccentricity m = (sqrt 3) / 2))
  (eccentricity : ℝ → ℝ := λ a, sqrt (1 - 1/a)) :
  let f_P := number_of_true_forms h_P in
  f_P = 2 :=
by
  sorry

end ellipse_problem_l68_68686


namespace min_recolored_cells_l68_68801

/-- On a 25 × 25 grid with exactly 9 black cells in each row and column,
the minimum number of black cells that must be recolored white to ensure
that there are no 2 × 2 black squares is 48. -/
theorem min_recolored_cells (grid : matrix (fin 25) (fin 25) bool)
  (h_row : ∀ i : fin 25, finset.filter (λ j, grid i j = tt) finset.univ.card = 9)
  (h_col : ∀ j : fin 25, finset.filter (λ i, grid i j = tt) finset.univ.card = 9) :
  ∃ k : nat, k = 48 ∧ (∀ i j : fin 24, 
    grid i j = tt ∧ grid i (j + 1) = tt ∧ grid (i + 1) j = tt ∧ grid (i + 1) (j + 1) = tt → false) :=
sorry

end min_recolored_cells_l68_68801


namespace no_four_digit_perfect_square_palindromes_l68_68965

def is_palindrome (n : ℕ) : Prop :=
  (n.toString = n.toString.reverse)

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem no_four_digit_perfect_square_palindromes :
  ¬ ∃ n : ℕ, is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n :=
by introv H, cases H with n Hn sorry

end no_four_digit_perfect_square_palindromes_l68_68965


namespace determinant_ge_bound_for_any_triangle_l68_68757

-- Define a triangle with sides a, b, c, and inradius r
variables {a b c r : ℝ}

-- Define the determinant expression
def det_expr (a b c : ℝ) : ℝ :=
  Matrix.det ![
    ![(b + c)^2, a^2, (a * b)^2],
    ![(c + a)^2, b^2, (b * c)^2],
    ![(a + b)^2, c^2, (c * a)^2]
  ]

-- Define the inradius r in terms of a, b, c using the formula for r
def inradius (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2  -- semi-perimeter
  (Math.sqrt (s * (s - a) * (s - b) * (s - c))) / s

theorem determinant_ge_bound_for_any_triangle (a b c r : ℝ) (h : r = inradius a b c) :
  det_expr a b c ≥ 93312 * (r^6) :=
by sorry

end determinant_ge_bound_for_any_triangle_l68_68757


namespace unique_prime_factors_l68_68738

theorem unique_prime_factors (n : ℕ) (hn_pos : 0 < n) (h_factors : ∃ k : ℕ, k = 12320 ∧ ∏ i in (finset.range k), (n.factorization i + 1) = 12320) :
  (n.factorization.support.card = 4) :=
sorry

end unique_prime_factors_l68_68738


namespace magicians_can_determine_dice_in_pocket_l68_68858

-- Definitions based on conditions
def dice_faces : List Nat := [1, 2, 3, 4, 5, 6]

-- Generate all pairs (a, b) with a <= b
def pairs := List.bind dice_faces (λ a, List.map (λ b => (a, b)) (dice_faces.filter (λ b => a <= b)))

-- The preassigned unique index for each pair
def pair_index (pair : Nat × Nat) : Option Nat :=
  pairs.indexOf pair

-- Given conditions
theorem magicians_can_determine_dice_in_pocket
  (n : Nat) -- number of dice taken by spectator
  (h1 : 3 ≤ n) -- at least three dice taken
  (h2 : n ≤ 23) -- at most twenty-three dice taken
  (d1 d2 : Nat) -- numbers on top faces of two dice shown
  (hp : (d1, d2) ∈ pairs ∨ (d2, d1) ∈ pairs) -- the pair exists in the list
  : ∃ k, pair_index (d1, d2) = some k ∨ pair_index (d2, d1) = some k ∧ k = (n - 2) := sorry

end magicians_can_determine_dice_in_pocket_l68_68858


namespace combined_garden_area_l68_68792

def garden_area (length width : ℕ) : ℕ :=
  length * width

def total_area (count length width : ℕ) : ℕ :=
  count * garden_area length width

theorem combined_garden_area :
  let M_length := 16
  let M_width := 5
  let M_count := 3
  let Ma_length := 8
  let Ma_width := 4
  let Ma_count := 2
  total_area M_count M_length M_width + total_area Ma_count Ma_length Ma_width = 304 :=
by
  sorry

end combined_garden_area_l68_68792


namespace day_of_week_in_53_days_l68_68491

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68491


namespace min_value_l68_68258

-- Define the problem conditions
variable (a : ℕ → ℝ)
variable (m n : ℕ)
variable (q : ℝ)

-- Given conditions
def condition1 := a 5 = 1/2
def condition2 := 4 * (a 4) + (a 6) = 2
def condition3 := ∀ m n, (Real.sqrt (a m * a n) = 4 * a 1)

-- The main proposition to prove
theorem min_value (cond1 : condition1) (cond2 : condition2) (cond3 : condition3) : 
  ∃ m n : ℕ, m + n = 6 ∧ min (1 / m + 4 / n) = 3 / 2 := 
sorry

end min_value_l68_68258


namespace NaCl_yield_l68_68654

structure Reaction :=
  (reactant1 : ℕ)
  (reactant2 : ℕ)
  (product : ℕ)

def NaOH := 3
def HCl := 3

theorem NaCl_yield : ∀ (R : Reaction), R.reactant1 = NaOH → R.reactant2 = HCl → R.product = 3 :=
by
  sorry

end NaCl_yield_l68_68654


namespace unique_solution_iff_a_eq_2_l68_68663

noncomputable def inequality_expr (a x : ℝ) :=
  log (sqrt (x^2 + a*x + 5) + 1) / log (1/4) * log (x^2 + a*x + 6) / log 5 + log 3 / log a

theorem unique_solution_iff_a_eq_2 :
  ∀ (a : ℝ), (∃! x : ℝ, inequality_expr a x ≥ 0) ↔ a = 2 :=
by
  intro a
  apply Iff.intro
  {
    intro h
    sorry
  }
  {
    intro ha
    rw ha
    sorry
  }

end unique_solution_iff_a_eq_2_l68_68663


namespace number_of_factors_of_m_l68_68776

def m : ℕ := 2^3 * 3^4 * 5^6 * 7^7

theorem number_of_factors_of_m : (Nat.factors_count m = 140) :=
sorry

end number_of_factors_of_m_l68_68776


namespace T_lt_half_S_l68_68331

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68331


namespace weight_loss_percentage_l68_68544

noncomputable def actual_weight_loss (W : ℝ) (x : ℝ) : ℝ := 
  (W - (x / 100) * W) + 0.02 * (W - (x / 100) * W)

theorem weight_loss_percentage (W : ℝ) (x : ℝ) (h : actual_weight_loss W x = 0.9078 * W) :
  x ≈ 5.55 :=
by
  sorry

end weight_loss_percentage_l68_68544


namespace four_digit_perfect_square_palindrome_count_l68_68976

def is_palindrome (n : ℕ) : Prop :=
  let str := n.repr
  str = str.reverse

def is_perfect_square (n : ℕ) : Prop :=
  ∃ (m : ℕ), m * m = n
  
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def is_four_digit_perfect_square_palindrome (n : ℕ) : Prop :=
  is_four_digit n ∧ is_perfect_square n ∧ is_palindrome n

theorem four_digit_perfect_square_palindrome_count : 
  { n : ℕ | is_four_digit_perfect_square_palindrome n }.card = 4 := 
sorry

end four_digit_perfect_square_palindrome_count_l68_68976


namespace maximize_expected_value_theorem_l68_68559

noncomputable def g (n : ℕ) : ℝ := n + 20 / n

def maximize_expected_value (n : ℕ) : Prop :=
  n ∈ {4, 5} ∧ 
  (∀ m : ℕ, m > 0 → g(n) ≤ g(m))

theorem maximize_expected_value_theorem :
  maximize_expected_value 4 ∨ maximize_expected_value 5 :=
sorry

end maximize_expected_value_theorem_l68_68559


namespace find_day_53_days_from_friday_l68_68485

-- Define the days of the week
inductive day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open day

-- Define a function that calculates the day of the week after a certain number of days
def add_days (d : day) (n : ℕ) : day :=
  match d, n % 7 with
  | Sunday, 0 => Sunday
  | Monday, 0 => Monday
  | Tuesday, 0 => Tuesday
  | Wednesday, 0 => Wednesday
  | Thursday, 0 => Thursday
  | Friday, 0 => Friday
  | Saturday, 0 => Saturday
  | Sunday, 1 => Monday
  | Monday, 1 => Tuesday
  | Tuesday, 1 => Wednesday
  | Wednesday, 1 => Thursday
  | Thursday, 1 => Friday
  | Friday, 1 => Saturday
  | Saturday, 1 => Sunday
  | Sunday, 2 => Tuesday
  | Monday, 2 => Wednesday
  | Tuesday, 2 => Thursday
  | Wednesday, 2 => Friday
  | Thursday, 2 => Saturday
  | Friday, 2 => Sunday
  | Saturday, 2 => Monday
  | Sunday, 3 => Wednesday
  | Monday, 3 => Thursday
  | Tuesday, 3 => Friday
  | Wednesday, 3 => Saturday
  | Thursday, 3 => Sunday
  | Friday, 3 => Monday
  | Saturday, 3 => Tuesday
  | Sunday, 4 => Thursday
  | Monday, 4 => Friday
  | Tuesday, 4 => Saturday
  | Wednesday, 4 => Sunday
  | Thursday, 4 => Monday
  | Friday, 4 => Tuesday
  | Saturday, 4 => Wednesday
  | Sunday, 5 => Friday
  | Monday, 5 => Saturday
  | Tuesday, 5 => Sunday
  | Wednesday, 5 => Monday
  | Thursday, 5 => Tuesday
  | Friday, 5 => Wednesday
  | Saturday, 5 => Thursday
  | Sunday, 6 => Saturday
  | Monday, 6 => Sunday
  | Tuesday, 6 => Monday
  | Wednesday, 6 => Tuesday
  | Thursday, 6 => Wednesday
  | Friday, 6 => Thursday
  | Saturday, 6 => Friday
  | _, _ => Sunday -- This case is unreachable, but required to complete the pattern match
  end

-- Lean statement for the proof problem:
theorem find_day_53_days_from_friday : add_days Friday 53 = Tuesday :=
  sorry

end find_day_53_days_from_friday_l68_68485


namespace sum_of_possible_values_of_k_l68_68246

theorem sum_of_possible_values_of_k :
  (∀ j k : ℕ, 0 < j ∧ 0 < k → (1 / (j:ℚ)) + (1 / (k:ℚ)) = (1 / 5) → k = 6 ∨ k = 10 ∨ k = 30) ∧ 
  (46 = 6 + 10 + 30) :=
by
  sorry

end sum_of_possible_values_of_k_l68_68246


namespace smallest_possible_perimeter_l68_68003

def triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

def consecutive_even_integers (a b c : ℕ) : Prop :=
  2 ∣ a ∧ 2 ∣ b ∧ 2 ∣ c ∧ b = a + 2 ∧ c = b + 2

def perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem smallest_possible_perimeter :
  ∃ (a b c : ℕ), consecutive_even_integers a b c ∧ triangle a b c ∧ perimeter a b c = 12 :=
by {
  use [2, 4, 6],
  split,
  { -- prove consecutive_even_integers 2 4 6
    split, exact dvd_refl 2,
    split, exact dvd_mul_right 2 2,
    split, exact dvd_mul_right 2 3,
    split, refl,
    refl
  },
  split,
  { -- prove triangle 2 4 6
    split, linarith,
    split, linarith,
    linarith
  },
  -- prove perimeter 2 4 6 = 12
  refl
}

end smallest_possible_perimeter_l68_68003


namespace least_subtraction_for_divisibility_l68_68108

theorem least_subtraction_for_divisibility (n : ℕ) (h : n = 964807) : ∃ k, k = 7 ∧ (n - k) % 8 = 0 :=
by 
  sorry

end least_subtraction_for_divisibility_l68_68108


namespace proj_norms_l68_68189

open Real

variables (u v w : ℝ^3)
variables (hu : ‖u‖ = 4) (hv : ‖v‖ = 5) (hw : ‖w‖ = 6)
variables (uv_dot : inner u v = 12) (vw_dot : inner v w = 15)

theorem proj_norms :
  (‖(u ⋅ v) / (‖v‖)‖ = 12 / 5) ∧ (‖(v ⋅ w) / (‖w‖)‖ = 5 / 2) :=
by sorry

end proj_norms_l68_68189


namespace ratio_of_shares_l68_68564

-- Definitions
variable (A B C : ℝ)   -- Representing the shares of a, b, and c
variable (x : ℝ)       -- Fraction

-- Conditions
axiom h1 : A = 80
axiom h2 : A + B + C = 200
axiom h3 : A = x * (B + C)
axiom h4 : B = (6 / 9) * (A + C)

-- Statement to prove
theorem ratio_of_shares : A / (B + C) = 2 / 3 :=
by sorry

end ratio_of_shares_l68_68564


namespace areaRatio_findAngle_l68_68805

-- Point M on side BC such that BM:MC=2:5
def pointM (A B C M : Point) (h_bc : collinear B C M) (h_rat : ratio B M M C = 2 / 5) : Prop := sorry

-- Bisector BL of triangle A intersects segment AM at point P at 90 degrees
def bisectorAngle (A B C M L P : Point) (h_bis : isBisector B L A) (h_ratio : angle (line A M) (line B L) = angle.right) : Prop := sorry

-- Point F on segment MC such that MF:FC=1:4
def segmentRatio (C F M : Point) (h_seg : segment F in (segment M C)) (h_rat2 : ratio M F F C = 1 / 4) : Prop := sorry

-- Line LF perpendicular to BC
def perpLine (L F B C : Point) (h_perp : perpendicular (line L F) (line B C)) : Prop := sorry

-- Given points and hypotheses
variable (A B C M L P F : Point)

-- Proof part a: Finding the ratio of areas of triangle ABP to quadrilateral LPMC
theorem areaRatio (h_bc : collinear B C M) (h_rat1 : ratio B M M C = 2 / 5)
  (h_bis : isBisector B L A) (h_angle : angle (line A M) (line B L) = angle.right)
  (h_seg : segment F in (segment M C)) (h_rat2 : ratio M F F C = 1 / 4)
  (h_perp : perpendicular (line L F) (line B C)) :
  area A B P / area L P M C = 9 / 40 := sorry

-- Proof part b: Finding the angle CBL
theorem findAngle (h_bc : collinear B C M) (h_rat1 : ratio B M M C = 2 / 5)
  (h_bis : isBisector B L A) (h_angle : angle (line A M) (line B L) = angle.right)
  (h_seg : segment F in (segment M C)) (h_rat2 : ratio M F F C = 1 / 4)
  (h_perp : perpendicular (line L F) (line B C)) :
  angle C B L = arccos (3 * sqrt 3 / (2 * sqrt 7)) := sorry

end areaRatio_findAngle_l68_68805


namespace sufficient_not_necessary_condition_l68_68893

theorem sufficient_not_necessary_condition (x : ℝ) : (x < -1 → x^2 - 1 > 0) ∧ (x^2 - 1 > 0 → x < -1 ∨ x > 1) :=
by
  sorry

end sufficient_not_necessary_condition_l68_68893


namespace arithmetic_sequence_sum_l68_68171

variable (a : ℕ → ℝ) (d : ℝ)
-- Conditions
def is_arithmetic_sequence : Prop := ∀ n : ℕ, a (n + 1) = a n + d
def condition : Prop := a 4 + a 8 = 8

-- Question
theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a d →
  condition a →
  (11 / 2) * (a 1 + a 11) = 44 :=
by
  sorry

end arithmetic_sequence_sum_l68_68171


namespace T_lt_S_div_2_l68_68336

noncomputable def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
noncomputable def b (n : ℕ) : ℝ := n * (1 / 3)^n
noncomputable def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)
noncomputable def T (n : ℕ) : ℝ := ∑ i in Finset.range n, b (i + 1)
noncomputable def S_general (n : ℕ) : ℝ := 3/2 - (1/2) * (1/3)^(n-1)
noncomputable def T_general (n : ℕ) : ℝ := 3/4 - (1/4) * (1/3)^(n-1) - (n * (1/3)^(n-1))/2

theorem T_lt_S_div_2 (n : ℕ) : T n < S n / 2 :=
by sorry

end T_lt_S_div_2_l68_68336


namespace problem_l68_68760

noncomputable def polar_eq_C1 (θ : ℝ) : ℝ := 4 * Real.sin θ

structure ParametricC2 (t : ℝ) (m α : ℝ) :=
(x y : ℝ)
(x_eq : x = m + t * Real.cos α)
(y_eq : y = t * Real.sin α)

theorem problem (φ m α : ℝ) (A B C : ℝ × ℝ) (hA : A = (polar_eq_C1 φ, φ))
  (hB : B = (polar_eq_C1 (φ + π / 4), φ + π / 4))
  (hC : C = (polar_eq_C1 (φ - π / 4), φ - π / 4))
  (h_alpha_range : 0 ≤ α ∧ α < π)
  (φ_5π_over_12 : φ = 5 * π / 12)
  (hB_on_C2 : ParametricC2 (polar_eq_C1 (φ + π / 4)) m α B.1 B.2)  -- B on C2
  (hC_on_C2 : ParametricC2 (polar_eq_C1 (φ - π / 4)) m α C.1 C.2)  -- C on C2
  : B.1 + C.1 = √2 * A.1 ∧ m = 2 * √3 ∧ α = 5 * π / 6 := sorry

end problem_l68_68760


namespace impossible_to_zero_all_integers_l68_68436

theorem impossible_to_zero_all_integers (n : ℕ)
  (h : n ≥ 2)
  (board : (ℕ × ℕ) → ℤ)
  (sum_board_eq_zero : ∑ i in finset.range n, ∑ j in finset.range n, board (i, j) = 0)
  (action : ∀ i j, board (i, j) := board (i, j) - (count_neighbors (i, j)) + (∑ nb in neighbors (i, j), board nb + 1)) :
  ¬ (∃ k, k > 0 ∧ ∀ i j, board (i, j) = 0) :=
sorry

def count_neighbors (i j : ℕ) : ℕ :=
sorry -- Implementing the counting of neighbors is skipped

def neighbors (i j : ℕ) : finset (ℕ × ℕ) :=
sorry -- Implementing the neighbors function is skipped

end impossible_to_zero_all_integers_l68_68436


namespace problem_expansion_l68_68163

-- Define the necessary conditions and the main theorem
theorem problem_expansion {n : ℕ} :
  (∃ (a b c : ℕ), 
    (a + 1 = b ∧ b + 1 = c ∧ 
    (n.choose 0) + (1 / 4) * (n.choose 2) = 2 * (1 / 2) * (n.choose 1)) ∧ n = 8) ∧
  ((∃ (r : ℕ), 
    (r = 3 ∨ r = 4) ∧ 
    (((1 / 2 ^ r) * (8.choose r) ≥ (1 / 2 ^ (r + 1)) * (8.choose (r + 1))) ∧
    ((1 / 2 ^ r) * (8.choose r) ≥ (1 / 2 ^ (r - 1)) * (8.choose (r - 1))) ∧ 
    7 * x ^ 5 ≥ 7 * x ^ (6 - 1) ∧ 
    r.choose r * x ^ (8 - r) = 7 * x ^ (8 - r)))) :=
begin
   sorry
end

end problem_expansion_l68_68163


namespace days_from_friday_l68_68506

theorem days_from_friday (n : ℕ) (h : n = 53) : 
  ∃ k m, (53 = 7 * k + m) ∧ m = 4 ∧ (4 + 1 = 5) ∧ (5 = 1) := 
sorry

end days_from_friday_l68_68506


namespace range_of_a_l68_68169

theorem range_of_a (a : ℝ) :
  (¬ ( ∃ x : ℝ, -1 ≤ x ∧ x ≤ 1 ∧ a^2 * x^2 + a * x - 2 = 0 ) 
    ∨ 
   ¬ ( ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0 )) 
→ a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end range_of_a_l68_68169


namespace labor_arrangement_count_l68_68921

theorem labor_arrangement_count (volunteers : ℕ) (choose_one_day : ℕ) (days : ℕ) 
    (h_volunteers : volunteers = 7) 
    (h_choose_one_day : choose_one_day = 3) 
    (h_days : days = 2) : 
    (Nat.choose volunteers choose_one_day) * (Nat.choose (volunteers - choose_one_day) choose_one_day) = 140 := 
by
  sorry

end labor_arrangement_count_l68_68921


namespace leo_new_average_l68_68281

def leo_scores := [92, 87, 91]
def fourth_score := 95
def previous_average := (92 + 87 + 91) / 3
def new_average := (92 + 87 + 91 + 95) / 4
def change_in_average := new_average - previous_average

theorem leo_new_average : (previous_average = 90) → (new_average = 91.25) ∧ (change_in_average = 1.25) :=
by
  intro h
  sorry

end leo_new_average_l68_68281


namespace day_of_week_in_53_days_l68_68490

/-- Prove that 53 days from Friday is Tuesday given today is Friday and a week has 7 days --/
theorem day_of_week_in_53_days
    (today : String)
    (days_in_week : ℕ)
    (day_count : ℕ)
    (target_day : String)
    (h1 : today = "Friday")
    (h2 : days_in_week = 7)
    (h3 : day_count = 53)
    (h4 : target_day = "Tuesday") :
    let remainder := day_count % days_in_week
    let computed_day := match remainder with
                        | 0 => "Friday"
                        | 1 => "Saturday"
                        | 2 => "Sunday"
                        | 3 => "Monday"
                        | 4 => "Tuesday"
                        | 5 => "Wednesday"
                        | 6 => "Thursday"
                        | _ => "Invalid"
    in computed_day = target_day := 
by
  sorry

end day_of_week_in_53_days_l68_68490


namespace tn_lt_sn_div_2_l68_68301

section geometric_sequence

open_locale big_operators

def a (n : ℕ) : ℝ := (1 / 3)^(n - 1)
def b (n : ℕ) : ℝ := (n : ℝ) * (1 / 3)^n

def S (n : ℕ) : ℝ := ∑ i in finset.range n, a (i + 1)
def T (n : ℕ) : ℝ := ∑ i in finset.range n, b (i + 1)

theorem tn_lt_sn_div_2 (n : ℕ) : T n < S n / 2 := sorry

end geometric_sequence

end tn_lt_sn_div_2_l68_68301


namespace problem_k_value_l68_68747

theorem problem_k_value (k x1 x2 : ℝ) 
  (h_eq : 8 * x1^2 + 2 * k * x1 + k - 1 = 0) 
  (h_eq2 : 8 * x2^2 + 2 * k * x2 + k - 1 = 0) 
  (h_sum_sq : x1^2 + x2^2 = 1) : 
  k = -2 :=
sorry

end problem_k_value_l68_68747


namespace fg_sqrt3_eq_neg3_minus_2sqrt3_l68_68202
noncomputable def f (x : ℝ) : ℝ := 5 - 2 * x
noncomputable def g (x : ℝ) : ℝ := x^2 + x + 1

theorem fg_sqrt3_eq_neg3_minus_2sqrt3 : f (g (Real.sqrt 3)) = -3 - 2 * Real.sqrt 3 := 
by sorry

end fg_sqrt3_eq_neg3_minus_2sqrt3_l68_68202


namespace find_f_neg4_l68_68282

def f (x : ℝ) : ℝ :=
if x < 0 then 3 * x + 10
else x^2 - 4 * x + 4

theorem find_f_neg4 : f (-4) = -2 :=
by
  -- proof goes here
  sorry

end find_f_neg4_l68_68282


namespace percentage_regular_cars_l68_68369

theorem percentage_regular_cars (total_cars : ℕ) (truck_percentage : ℚ) (convertibles : ℕ) 
  (h1 : total_cars = 125) (h2 : truck_percentage = 0.08) (h3 : convertibles = 35) : 
  (80 / 125 : ℚ) * 100 = 64 := 
by 
  sorry

end percentage_regular_cars_l68_68369


namespace right_triangle_sides_l68_68412

theorem right_triangle_sides (x y z : ℕ) (h1 : x + y + z = 30)
    (h2 : x^2 + y^2 + z^2 = 338) (h3 : x^2 + y^2 = z^2) :
    (x = 5 ∧ y = 12 ∧ z = 13) ∨ (x = 12 ∧ y = 5 ∧ z = 13) :=
by
  sorry

end right_triangle_sides_l68_68412


namespace fiftyThreeDaysFromFridayIsTuesday_l68_68459

-- Domain definition: Days of the week
inductive Day : Type
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Day

-- Function to compute the day of the week after a given number of days
def dayAfter (start: Day) (n: ℕ) : Day :=
  match start with
  | Sunday    => Day.ofNat ((0 + n) % 7)
  | Monday    => Day.ofNat ((1 + n) % 7)
  | Tuesday   => Day.ofNat ((2 + n) % 7)
  | Wednesday => Day.ofNat ((3 + n) % 7)
  | Thursday  => Day.ofNat ((4 + n) % 7)
  | Friday    => Day.ofNat ((5 + n) % 7)
  | Saturday  => Day.ofNat ((6 + n) % 7)

namespace Day

-- Helper function to convert natural numbers back to day of the week
def ofNat : ℕ → Day
| 0 => Sunday
| 1 => Monday
| 2 => Tuesday
| 3 => Wednesday
| 4 => Thursday
| 5 => Friday
| 6 => Saturday
| _ => Day.ofNat (n % 7)

-- Lean statement: Prove that 53 days from Friday is Tuesday
theorem fiftyThreeDaysFromFridayIsTuesday :
  dayAfter Friday 53 = Tuesday :=
sorry

end fiftyThreeDaysFromFridayIsTuesday_l68_68459


namespace problem_1_problem_2_l68_68176

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt 2) * Real.cos (x - π / 12)

theorem problem_1 : f (-π / 6) = 1 := 
by sorry -- Proof skipped

theorem problem_2 (θ : ℝ) (h₁ : Real.cos θ = 3 / 5) (h₂ : θ ∈ Ioo (3 * π / 2) (2 * π)) :
  f (2 * θ + π / 3) = 17 / 25 := 
by sorry -- Proof skipped

end problem_1_problem_2_l68_68176


namespace picking_ball_is_random_event_l68_68233

-- Definitions based on problem conditions
def total_balls := 201
def black_balls := 200
def white_balls := 1

-- The goal to prove
theorem picking_ball_is_random_event : 
  (total_balls = black_balls + white_balls) ∧ 
  (black_balls > 0) ∧ 
  (white_balls > 0) → 
  random_event :=
by sorry

end picking_ball_is_random_event_l68_68233


namespace minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2_l68_68781

noncomputable def Q (z : ℂ) : ℂ := z^8 + (6 * real.sqrt 2 + 8) * z^4 - (6 * real.sqrt 2 + 9)

theorem minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2 :
  ∃ (vertices : fin 8 → ℂ), (∀ v, Q v = 0) ∧ 
  minimum_perimeter (vertices '' univ) = 8 * real.sqrt 2 :=
sorry

end minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2_l68_68781


namespace abs_eq_2_iff_l68_68839

theorem abs_eq_2_iff (a : ℚ) : abs a = 2 ↔ a = 2 ∨ a = -2 :=
by
  sorry

end abs_eq_2_iff_l68_68839


namespace tetrahedron_inscribed_in_pyramid_edge_length_l68_68596

noncomputable def edge_length_of_tetrahedron := (Real.sqrt 2) / 2

theorem tetrahedron_inscribed_in_pyramid_edge_length :
  let A := (0,0,0)
  let B := (1,0,0)
  let C := (1,1,0)
  let D := (0,1,0)
  let E := (0.5, 0.5, 1)
  let v₁ := (0.5, 0, 0)
  let v₂ := (1, 0.5, 0)
  let v₃ := (0, 0.5, 0)
  dist (v₁ : ℝ × ℝ × ℝ) v₂ = edge_length_of_tetrahedron ∧
  dist v₂ v₃ = edge_length_of_tetrahedron ∧
  dist v₃ v₁ = edge_length_of_tetrahedron ∧
  dist E v₁ = dist E v₂ ∧
  dist E v₂ = dist E v₃ :=
by
  sorry

end tetrahedron_inscribed_in_pyramid_edge_length_l68_68596


namespace angle_between_b_and_c_l68_68700

open Real

-- Definitions
def vec_a : ℝ × ℝ := sorry
def vec_b : ℝ × ℝ := sorry
def vec_c : ℝ × ℝ := (1, sqrt 3)

-- Given Conditions
axiom cond1 : 2 • vec_a - vec_b = (-1, sqrt 3)
axiom cond2 : vec_a.1 * vec_c.1 + vec_a.2 * vec_c.2 = 3
axiom cond3 : sqrt (vec_b.1 * vec_b.1 + vec_b.2 * vec_b.2) = 4

-- Dot Product
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Magnitude of c
def magnitude_vec_c : ℝ := sqrt (vec_c.1 * vec_c.1 + vec_c.2 * vec_c.2)

-- Magnitude of b
def magnitude_vec_b : ℝ := sqrt (vec_b.1 * vec_b.1 + vec_b.2 * vec_b.2)

-- Theorem
theorem angle_between_b_and_c : 
  ∃ θ : ℝ, θ = acos ((dot_product vec_b vec_c) / (magnitude_vec_b * magnitude_vec_c)) ∧ θ = 60 * (π / 180) :=
by
  sorry

end angle_between_b_and_c_l68_68700


namespace one_div_seventeen_125th_digit_is_4_l68_68867

theorem one_div_seventeen_125th_digit_is_4 :
  ∀ (n : ℕ), (n = 125) → (0.0588235294117647).nth_digit (125) = 4 := 
by
  intro n
  intro n_eq
  sorry

end one_div_seventeen_125th_digit_is_4_l68_68867


namespace fraction_white_part_l68_68581

theorem fraction_white_part (total_length : ℝ) (black_fraction : ℝ) (blue_length : ℝ)
  (h_total_length : total_length = 8) (h_black_fraction : black_fraction = 1/8)
  (h_blue_length : blue_length = 7/2) :
  let black_length := black_fraction * total_length in
  let remaining_length := total_length - black_length in
  let white_length := remaining_length - blue_length in
  (white_length / remaining_length) = 1/2 :=
by
  sorry

end fraction_white_part_l68_68581


namespace no_real_solutions_sqrt_eqn_l68_68098

theorem no_real_solutions_sqrt_eqn (z : ℝ) : ¬(sqrt (5 - 4 * z) = 7) := 
by
  sorry

end no_real_solutions_sqrt_eqn_l68_68098


namespace ellipse_semi_minor_axis_l68_68632

theorem ellipse_semi_minor_axis (a b p : ℝ) 
  (h1 : p = π * (3 * (a + b) - sqrt ((3 * a + b) * (a + 3 * b))))
  (h2 : 2 * a = 2 * a)
  : b = 3 * a - p / π :=
sorry

end ellipse_semi_minor_axis_l68_68632


namespace father_catches_up_l68_68543

noncomputable def min_steps_to_catch_up : Prop :=
  let x := 30
  let father_steps := 5
  let xiaoming_steps := 8
  let distance_ratio := 2 / 5
  let xiaoming_headstart := 27
  ((xiaoming_headstart + (xiaoming_steps / father_steps) * x) / distance_ratio) = x

theorem father_catches_up : min_steps_to_catch_up :=
  by
  sorry

end father_catches_up_l68_68543


namespace fraction_scaling_l68_68739

theorem fraction_scaling (x y : ℝ) :
  ((5 * x - 5 * 5 * y) / ((5 * x) ^ 2 + (5 * y) ^ 2)) = (1 / 5) * ((x - 5 * y) / (x ^ 2 + y ^ 2)) :=
by
  sorry

end fraction_scaling_l68_68739


namespace radius_increase_by_100_percent_l68_68548

theorem radius_increase_by_100_percent (A A' r r' : ℝ) (π : ℝ)
  (h1 : A = π * r^2) -- initial area of the circle
  (h2 : A' = 4 * A) -- new area is 4 times the original area
  (h3 : A' = π * r'^2) -- new area formula with new radius
  : r' = 2 * r :=
by
  sorry

end radius_increase_by_100_percent_l68_68548


namespace sum_of_first_2n_when_n_eq_1_l68_68533

theorem sum_of_first_2n_when_n_eq_1 : (∑ i in finset.range (2*1 + 1), i) = 3 := 
by
  sorry

end sum_of_first_2n_when_n_eq_1_l68_68533


namespace Tn_lt_Sn_over_2_l68_68320

theorem Tn_lt_Sn_over_2 (a b : ℕ → ℝ) (S T : ℕ → ℝ) (n : ℕ) :
  (a 1 = 1) →
  (∀ n, b n = n * (a n) / 3) →
  (a 1, 3 * a 2, 9 * a 3 are in arithmetic_sequence) →
  (S n = ∑ i in (range n), a i) →
  (T n = ∑ i in (range n), b i) →
  T n < S n / 2 :=
sorry

end Tn_lt_Sn_over_2_l68_68320


namespace T_lt_half_S_l68_68325

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3) ^ n
def S (n : ℕ) : ℝ := (3/2) - (1/2) * (1/3)^(n-1)
def T (n : ℕ) : ℝ := (3/4) - (1/4) * (1/3)^(n-1) - (n/2) * (1/3)^n

theorem T_lt_half_S (n : ℕ) (hn : n ≥ 1) : T n < (S n) / 2 :=
by
  sorry

end T_lt_half_S_l68_68325


namespace smallest_xyz_sum_l68_68785

theorem smallest_xyz_sum (x y z : ℕ) (h1 : (x + y) * (y + z) = 2016) (h2 : (x + y) * (z + x) = 1080) :
  x > 0 → y > 0 → z > 0 → x + y + z = 61 :=
  sorry

end smallest_xyz_sum_l68_68785


namespace day_after_53_days_from_Friday_l68_68518

def Day : Type := ℕ -- Representing days of the week as natural numbers (0 for Sunday, 1 for Monday, ..., 6 for Saturday)

def Friday : Day := 5
def Tuesday : Day := 2

def days_after (start_day : Day) (n : ℕ) : Day :=
  (start_day + n) % 7

theorem day_after_53_days_from_Friday : days_after Friday 53 = Tuesday :=
  by
  -- proof goes here
  sorry

end day_after_53_days_from_Friday_l68_68518


namespace count_palindromic_four_digit_perfect_squares_l68_68990

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem count_palindromic_four_digit_perfect_squares : 
  (finset.card {n ∈ (finset.range 10000).filter (λ x, is_four_digit x ∧ is_perfect_square x ∧ is_palindrome x)} = 5) :=
by
  sorry

end count_palindromic_four_digit_perfect_squares_l68_68990


namespace fifty_three_days_from_friday_is_tuesday_l68_68478

inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

open DayOfWeek

def dayAfter (d : DayOfWeek) (n : ℕ) : DayOfWeek :=
match n % 7 with
| 0 => d
| 1 => match d with
       | Sunday    => Monday
       | Monday    => Tuesday
       | Tuesday   => Wednesday
       | Wednesday => Thursday
       | Thursday  => Friday
       | Friday    => Saturday
       | Saturday  => Sunday
| 2 => match d with
       | Sunday    => Tuesday
       | Monday    => Wednesday
       | Tuesday   => Thursday
       | Wednesday => Friday
       | Thursday  => Saturday
       | Friday    => Sunday
       | Saturday  => Monday
| 3 => match d with
       | Sunday    => Wednesday
       | Monday    => Thursday
       | Tuesday   => Friday
       | Wednesday => Saturday
       | Thursday  => Sunday
       | Friday    => Monday
       | Saturday  => Tuesday
| 4 => match d with
       | Sunday    => Thursday
       | Monday    => Friday
       | Tuesday   => Saturday
       | Wednesday => Sunday
       | Thursday  => Monday
       | Friday    => Tuesday
       | Saturday  => Wednesday
| 5 => match d with
       | Sunday    => Friday
       | Monday    => Saturday
       | Tuesday   => Sunday
       | Wednesday => Monday
       | Thursday  => Tuesday
       | Friday    => Wednesday
       | Saturday  => Thursday
| 6 => match d with
       | Sunday    => Saturday
       | Monday    => Sunday
       | Tuesday   => Monday
       | Wednesday => Tuesday
       | Thursday  => Wednesday
       | Friday    => Thursday
       | Saturday  => Friday
| _ => d  -- although all cases are covered

theorem fifty_three_days_from_friday_is_tuesday :
  dayAfter Friday 53 = Tuesday :=
by
  sorry

end fifty_three_days_from_friday_is_tuesday_l68_68478


namespace tangent_parallel_x_axis_coord_l68_68212

def f (x : ℝ) : ℝ := x^4 - 4 * x
def f' (x : ℝ) : ℝ := 4 * x^3 - 4

theorem tangent_parallel_x_axis_coord :
  ∃ (m n : ℝ), f m = n ∧ f' m = 0 ∧ m = 1 ∧ n = -3 :=
by {
  use 1,
  use -3,
  split,
  { -- f(1) = -3
    calc f 1 = 1^4 - 4 * 1 : by rw f
    ... = 1 - 4 : by norm_num
    ... = -3 : by norm_num },
  split,
  { -- f'(1) = 0
    calc f' 1 = 4 * 1^3 - 4 : by rw f'
    ... = 4 - 4 : by norm_num
    ... = 0 : by norm_num },
  { split; simp }
}

end tangent_parallel_x_axis_coord_l68_68212


namespace same_color_points_exist_l68_68035

theorem same_color_points_exist (d : ℝ) (colored_plane : ℝ × ℝ → Prop) :
  (∃ (p1 p2 : ℝ × ℝ), p1 ≠ p2 ∧ colored_plane p1 = colored_plane p2 ∧ dist p1 p2 = d) := 
sorry

end same_color_points_exist_l68_68035


namespace minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2_l68_68782

noncomputable def Q (z : ℂ) : ℂ := z^8 + (6 * real.sqrt 2 + 8) * z^4 - (6 * real.sqrt 2 + 9)

theorem minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2 :
  ∃ (vertices : fin 8 → ℂ), (∀ v, Q v = 0) ∧ 
  minimum_perimeter (vertices '' univ) = 8 * real.sqrt 2 :=
sorry

end minimum_perimeter_of_polygon_with_zeros_of_Q_is_8sqrt2_l68_68782


namespace max_area_of_equilateral_triangle_in_rectangle_l68_68409

def rectangle (A B C D : Type) := 
  -- assume that A, B, C, D are points defining the vertices of the rectangle with lengths 10 and 11
  (is_rectangle : True) ∧
  (side_lengths : (length AB = 10) ∧ (length BC = 11))

theorem max_area_of_equilateral_triangle_in_rectangle 
  (A B C D : Type) (p q r : ℕ) 
  [is_rectangle : rectangle A B C D]
  (h : ∃ (p q r : ℕ), area_equilateral_triangle_inside_rectangle A B C D = p * (sqrt q) - r) :
  p = 221 ∧ q = 3 ∧ r = 330 ∧ p + q + r = 554 := 
sorry

end max_area_of_equilateral_triangle_in_rectangle_l68_68409


namespace length_of_BC_l68_68185

noncomputable def triangle_ABC (A B C : Type) :=
  right_triangle A B C

noncomputable def triangle_ABD (A B D : Type) :=
  right_triangle A B D

variables (A B C D : Type)

axiom h1 : triangle_ABC A B C
axiom h2 : triangle_ABD A B D
axiom h3 : dist A B = 15
axiom h4 : dist A D = 50
axiom h5 : dist C D = 25
axiom h6 : dist A C = 20

theorem length_of_BC : dist B C = 25 :=
sorry

end length_of_BC_l68_68185


namespace value_of_expression_l68_68414

theorem value_of_expression (x : ℕ) (h : x = 2) : x + x * x^x = 10 := by
  rw [h] -- Substituting x = 2
  sorry

end value_of_expression_l68_68414


namespace mutually_exclusive_prob_l68_68883

/-- Define mutually exclusive events A and B -/
variables {Ω : Type*} {P : Ω → Prop} {A B : Ω → Prop}

/-- Definition of mutually exclusive events -/
def mutually_exclusive (A B : Ω → Prop) : Prop :=
  ∀ ω, ¬(A ω ∧ B ω)

/-- Prove that if A and B are mutually exclusive events,
then the probability of A plus the probability of B is less than or equal to 1 -/
theorem mutually_exclusive_prob (P : Ω → Prop) (A B : Ω → Prop) :
  mutually_exclusive A B → P(A) + P(B) ≤ 1 :=
by
  intros h
  sorry 

end mutually_exclusive_prob_l68_68883


namespace difference_between_integers_l68_68413

theorem difference_between_integers (a b : ℕ) (ha : a > 0) (hb : b > 0) (h1 : a + b = 20) (h2 : a * b = 96) :
  |a - b| = 4 :=
sorry

end difference_between_integers_l68_68413


namespace sin_double_angle_BPC_l68_68372

theorem sin_double_angle_BPC (a b c d : ℝ) (P : Point)
  (h1 : dist A B = dist B C) (h2 : dist B C = dist C D)
  (h3 : ∀ P, cos (∠ P A C) = 3 / 5) (h4 : ∀ P, cos (∠ P B D) = 1 / 5) :
  sin (2 * ∠ B P C) = 24 / 25 :=
by sorry

end sin_double_angle_BPC_l68_68372


namespace angle_BAC_50_degrees_l68_68909

theorem angle_BAC_50_degrees 
  (O A B C : Type)
  [MetricSpace O]
  (circumscribed : ∀ {X Y Z : O}, is_circumscribed_around_triangle O A B C)
  (angle_AOB : angle A O B = 120)
  (angle_AOC : angle A O C = 140) :
  angle B A C = 50 :=
by
  sorry

end angle_BAC_50_degrees_l68_68909


namespace gcd_impossible_l68_68786

-- Define the natural numbers a, b, and c
variable (a b c : ℕ)

-- Define the factorial values
def fact_30 := Nat.factorial 30
def fact_40 := Nat.factorial 40
def fact_50 := Nat.factorial 50

-- Define the gcd values to be checked
def gcd_ab := fact_30 + 111
def gcd_bc := fact_40 + 234
def gcd_ca := fact_50 + 666

-- The main theorem to prove the impossibility
theorem gcd_impossible (h1 : Nat.gcd a b = gcd_ab) (h2 : Nat.gcd b c = gcd_bc) (h3 : Nat.gcd c a = gcd_ca) : False :=
by
  -- Proof omitted
  sorry

end gcd_impossible_l68_68786


namespace max_points_5cm_away_from_P_l68_68371

noncomputable def maximum_points_on_circle (P : EuclideanGeometry.Point) (C : EuclideanGeometry.Circle) : ℕ :=
  2

theorem max_points_5cm_away_from_P (P : EuclideanGeometry.Point) (C : EuclideanGeometry.Circle) 
  (hP_outside_C : ¬ EuclideanGeometry.mem P C) : 
  ∃ n ≤ 2, maximum_points_on_circle P C = n ∧ n = 2 :=
by
  use 2
  sorry

end max_points_5cm_away_from_P_l68_68371


namespace Tn_lt_Sn_div2_l68_68295

open_locale big_operators

def a (n : ℕ) : ℝ := (1/3)^(n-1)
def b (n : ℕ) : ℝ := n * (1/3)^n

def S (n : ℕ) : ℝ := 
  ∑ i in finset.range n, a (i + 1)

def T (n : ℕ) : ℝ := 
  ∑ i in finset.range n, b (i + 1)

theorem Tn_lt_Sn_div2 (n : ℕ) : T n < S n / 2 := 
sorry

end Tn_lt_Sn_div2_l68_68295


namespace sum_nontrivial_factors_2015_l68_68399

def is_trivial_factor (n d : ℕ) : Prop :=
  d = 1 ∨ d = n

def sum_of_factors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0).sum

def sum_of_nontrivial_factors (n : ℕ) : ℕ :=
  let total_sum := sum_of_factors n
  let trivial_sum := (List.range (n + 1)).filter (λ d => is_trivial_factor n d).sum
  total_sum - trivial_sum

theorem sum_nontrivial_factors_2015 : sum_of_nontrivial_factors 2015 = 672 :=
  by
  sorry

end sum_nontrivial_factors_2015_l68_68399
